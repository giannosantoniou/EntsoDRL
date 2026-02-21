"""
Training Script for Unified Multi-Market Battery Trading Model

I train a single MaskablePPO model that handles ALL markets:
- DAM commitment execution
- IntraDay energy trading (separate from mFRR)
- aFRR capacity commitment and activation
- mFRR energy trading (separate from IntraDay)

Key Features:
1. MultiDiscrete action space [5, 5, 11, 11]
2. Hierarchical action masking with capacity cascading: DAM -> aFRR -> IntraDay -> mFRR
3. 63-feature observation space
4. VecNormalize for observation normalization
5. Curriculum learning for degradation cost
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# I add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

# I use MaskablePPO for action masking support
from sb3_contrib import MaskablePPO

from gym_envs.battery_env_unified import BatteryEnvUnified


class EntropyAnnealingCallback(BaseCallback):
    """
    I anneal the entropy coefficient from start_ent to end_ent over training.

    SB3 doesn't support ent_coef schedules natively, so I modify the model's
    ent_coef attribute directly. This allows the agent to explore broadly early
    in training (high entropy) and converge to a sharp policy later (low entropy).
    """

    def __init__(
        self,
        start_ent: float = 0.03,
        end_ent: float = 0.005,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.start_ent = start_ent
        self.end_ent = end_ent

    def _on_step(self) -> bool:
        # I compute progress_remaining (1→0) matching SB3 convention
        progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)
        progress_remaining = max(0.0, progress_remaining)

        # I linearly interpolate: start at start_ent, end at end_ent
        self.model.ent_coef = self.start_ent * progress_remaining + self.end_ent * (1.0 - progress_remaining)

        # I log the current ent_coef every 10k steps
        if self.num_timesteps % 10000 == 0:
            self.logger.record('train/ent_coef_current', self.model.ent_coef)

        return True


class DegradationCurriculumCallback(BaseCallback):
    """
    I implement curriculum learning for degradation cost.

    The degradation cost starts at 0 and ramps up to the target value
    over the warmup period. This helps the agent learn profitable trading
    before considering battery wear.

    I also monitor explained_variance during the ramp phase — if it drops
    below 0.7 the curriculum may be ramping too fast.
    """

    def __init__(
        self,
        target_degradation: float = 15.0,
        warmup_steps: int = 500_000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.target_degradation = target_degradation
        self.warmup_steps = warmup_steps
        self.current_degradation = 0.0
        self._warned_ev_drop = False

    def _on_step(self) -> bool:
        # I calculate current degradation based on training progress
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        self.current_degradation = progress * self.target_degradation

        # I update the environment's reward calculator
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'reward_calculator'):
                    env.reward_calculator.degradation_cost = self.current_degradation

        # I monitor explained_variance during the ramp phase (250k-750k)
        if (not self._warned_ev_drop
                and self.warmup_steps * 0.5 < self.num_timesteps < self.warmup_steps * 1.5):
            ev = self.logger.name_to_value.get('train/explained_variance', None)
            if ev is not None and ev < 0.7:
                print(f"\n[!] WARNING: explained_variance={ev:.3f} dropped below 0.7 "
                      f"during degradation ramp (step {self.num_timesteps:,}, "
                      f"degradation={self.current_degradation:.1f} EUR/MWh). "
                      f"Consider extending warmup_steps.")
                self._warned_ev_drop = True

        # I log current degradation cost
        if self.num_timesteps % 10000 == 0:
            self.logger.record('curriculum/degradation_cost', self.current_degradation)

        return True


class TensorboardLoggingCallback(BaseCallback):
    """I log additional metrics to Tensorboard including per-market statistics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # I track episode-level metrics
        self.episode_profits = []
        self.episode_cycles = []

        # I track per-market profits
        self.dam_profits = []
        self.intraday_profits = []
        self.afrr_capacity_profits = []
        self.afrr_energy_profits = []
        self.mfrr_profits = []

        # I track full-market episode-level profits
        self.ida_profits = []
        self.xbid_profits = []
        self.free_bid_profits = []

        # I track participation counts per episode
        self.afrr_bids_per_ep = []
        self.afrr_selections_per_ep = []
        self.afrr_activations_per_ep = []

        # I track step-level counts (reset each logging period)
        self.step_afrr_bids = 0
        self.step_afrr_selections = 0
        self.step_afrr_activations = 0
        self.step_intraday_trades = 0
        self.step_mfrr_trades = 0
        self.step_mfrr_sells = 0
        self.step_mfrr_buys = 0
        self.step_free_bid_trades = 0
        self.step_free_bid_sells = 0
        self.step_free_bid_buys = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        # I extract info from the environment
        for info in self.locals.get('infos', []):
            self.step_count += 1

            # I track step-level participation
            if info.get('afrr_commitment', 0) > 0.1:
                self.step_afrr_bids += 1
            if info.get('is_selected', False):
                self.step_afrr_selections += 1
            if info.get('afrr_activated', False):
                self.step_afrr_activations += 1
            if abs(info.get('intraday_energy_mw', 0)) > 0.1:
                self.step_intraday_trades += 1
            mfrr_mw = info.get('mfrr_energy_mw', 0)
            if abs(mfrr_mw) > 0.1:
                self.step_mfrr_trades += 1
                if mfrr_mw > 0:
                    self.step_mfrr_sells += 1
                else:
                    self.step_mfrr_buys += 1

            # I track Free Bid trades (full-market mode routes mFRR to free_bid_energy_mw)
            free_bid_mw = info.get('free_bid_energy_mw', 0)
            if abs(free_bid_mw) > 0.1:
                self.step_free_bid_trades += 1
                if free_bid_mw > 0:
                    self.step_free_bid_sells += 1
                else:
                    self.step_free_bid_buys += 1

            # I check for episode end (when total_profit is reported)
            if 'total_profit' in info:
                self.episode_profits.append(info['total_profit'])
            if 'total_cycles' in info:
                self.episode_cycles.append(info['total_cycles'])

            # I track per-market profits
            if 'dam_profit' in info:
                self.dam_profits.append(info['dam_profit'])
            if 'intraday_profit' in info:
                self.intraday_profits.append(info['intraday_profit'])
            if 'afrr_capacity_profit' in info:
                self.afrr_capacity_profits.append(info['afrr_capacity_profit'])
            if 'afrr_energy_profit' in info:
                self.afrr_energy_profits.append(info['afrr_energy_profit'])
            if 'mfrr_profit' in info:
                self.mfrr_profits.append(info['mfrr_profit'])

            # I collect full-market episode profits
            if 'ida_profit' in info:
                self.ida_profits.append(info['ida_profit'])
            if 'xbid_profit' in info:
                self.xbid_profits.append(info['xbid_profit'])
            if 'free_bid_profit' in info:
                self.free_bid_profits.append(info['free_bid_profit'])

        # I log periodically
        if self.num_timesteps % 10000 == 0 and len(self.episode_profits) > 0:
            # I log overall metrics
            avg_profit = np.mean(self.episode_profits[-100:])
            avg_cycles = np.mean(self.episode_cycles[-100:])
            self.logger.record('custom/avg_profit', avg_profit)
            self.logger.record('custom/avg_cycles', avg_cycles)
            if avg_cycles > 0:
                self.logger.record('custom/profit_per_cycle', avg_profit / avg_cycles)

            # I log per-market profits
            if len(self.dam_profits) > 0:
                self.logger.record('markets/dam_profit', np.mean(self.dam_profits[-100:]))
            if len(self.intraday_profits) > 0:
                self.logger.record('markets/intraday_profit', np.mean(self.intraday_profits[-100:]))
            if len(self.afrr_capacity_profits) > 0:
                self.logger.record('markets/afrr_capacity_profit', np.mean(self.afrr_capacity_profits[-100:]))
            if len(self.afrr_energy_profits) > 0:
                self.logger.record('markets/afrr_energy_profit', np.mean(self.afrr_energy_profits[-100:]))
            if len(self.mfrr_profits) > 0:
                self.logger.record('markets/mfrr_profit', np.mean(self.mfrr_profits[-100:]))

            # I log full-market mode metrics from episode-level accumulators
            if self.ida_profits:
                self.logger.record('markets/ida_profit', np.mean(self.ida_profits[-100:]))
            if self.xbid_profits:
                self.logger.record('markets/xbid_profit', np.mean(self.xbid_profits[-100:]))
            if self.free_bid_profits:
                self.logger.record('markets/free_bid_profit', np.mean(self.free_bid_profits[-100:]))

            # I log participation rates
            if self.step_count > 0:
                self.logger.record('participation/afrr_bid_rate', self.step_afrr_bids / self.step_count)
                self.logger.record('participation/afrr_selection_rate', self.step_afrr_selections / self.step_count)
                self.logger.record('participation/afrr_activation_rate', self.step_afrr_activations / self.step_count)
                self.logger.record('participation/intraday_trade_rate', self.step_intraday_trades / self.step_count)
                self.logger.record('participation/mfrr_trade_rate', self.step_mfrr_trades / self.step_count)
                self.logger.record('participation/mfrr_sell_rate', self.step_mfrr_sells / self.step_count)
                self.logger.record('participation/mfrr_buy_rate', self.step_mfrr_buys / self.step_count)
                self.logger.record('participation/free_bid_trade_rate', self.step_free_bid_trades / self.step_count)
                self.logger.record('participation/free_bid_sell_rate', self.step_free_bid_sells / self.step_count)
                self.logger.record('participation/free_bid_buy_rate', self.step_free_bid_buys / self.step_count)

                # I calculate conditional rates
                if self.step_afrr_bids > 0:
                    self.logger.record('participation/selection_given_bid',
                                       self.step_afrr_selections / self.step_afrr_bids)
                if self.step_afrr_selections > 0:
                    self.logger.record('participation/activation_given_selection',
                                       self.step_afrr_activations / self.step_afrr_selections)

            # I reset step counters
            self.step_afrr_bids = 0
            self.step_afrr_selections = 0
            self.step_afrr_activations = 0
            self.step_intraday_trades = 0
            self.step_mfrr_trades = 0
            self.step_mfrr_sells = 0
            self.step_mfrr_buys = 0
            self.step_free_bid_trades = 0
            self.step_free_bid_sells = 0
            self.step_free_bid_buys = 0
            self.step_count = 0

        return True


class MaskableVecEnv(DummyVecEnv):
    """
    I extend DummyVecEnv to properly support action masking with VecNormalize.

    MaskablePPO calls env_method("action_masks") which needs to work through
    the VecNormalize wrapper. This class ensures action_masks is accessible.
    """

    def action_masks(self):
        """I return action masks from all environments."""
        return np.array([env.action_masks() for env in self.envs])


def create_training_env(
    data_path: str,
    battery_params: Optional[Dict] = None,
    reward_config: Optional[Dict] = None,
    episode_length_hours: int = 168,  # 1 week in hours
    time_step_hours: float = 0.25    # 15-min default
) -> gym.Env:
    """I create and wrap the training environment."""
    # I load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded training data: {len(df)} rows")

    # I set default battery params
    if battery_params is None:
        battery_params = {
            'capacity_mwh': 146.0,
            'max_power_mw': 30.0,
            'efficiency': 0.94
        }

    # I set default reward config — penalties reduced to fix the 37:1
    # penalty/profit asymmetry that destroyed the gradient signal.
    # reward_scale raised from 0.001 to 0.01 for stronger NN signal.
    if reward_config is None:
        reward_config = {
            'degradation_cost': 5.0,
            'dam_violation_penalty': 800.0,
            'afrr_nonresponse_penalty': 500.0,
            'reward_scale': 0.01
        }

    # I convert episode length from hours to steps
    episode_length_steps = int(episode_length_hours / time_step_hours)

    # I create environment (it already implements action_masks())
    env = BatteryEnvUnified(
        df=df,
        capacity_mwh=battery_params['capacity_mwh'],
        max_power_mw=battery_params['max_power_mw'],
        efficiency=battery_params['efficiency'],
        time_step_hours=time_step_hours,
        episode_length=episode_length_steps,
        random_start=True,
        reward_config=reward_config
    )

    # I wrap with Monitor for logging
    env = Monitor(env)

    return env


def create_eval_env(
    data_path: str,
    battery_params: Optional[Dict] = None,
    time_step_hours: float = 0.25
) -> gym.Env:
    """I create the evaluation environment (deterministic)."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if battery_params is None:
        battery_params = {
            'capacity_mwh': 146.0,
            'max_power_mw': 30.0,
            'efficiency': 0.94
        }

    # I convert 2-week evaluation from hours to steps
    episode_length_steps = int(336 / time_step_hours)

    env = BatteryEnvUnified(
        df=df,
        capacity_mwh=battery_params['capacity_mwh'],
        max_power_mw=battery_params['max_power_mw'],
        efficiency=battery_params['efficiency'],
        time_step_hours=time_step_hours,
        episode_length=episode_length_steps,
        random_start=False
    )

    env = Monitor(env)

    return env


def train_unified_model(
    data_path: str = "data/unified_multimarket_training_v2.csv",
    output_dir: str = "models/unified",
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,  # I use 8 parallel environments for speedup
    learning_rate: float = 1e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.97,  # I raised from 0.95 to extend advantage horizon to ~18h (was ~12h)
    clip_range: float = 0.2,
    ent_coef: float = 0.03,
    seed: int = 42,
    device: str = "auto",
    resume_from: str = None,
    time_step_hours: float = 0.25,  # 15-min resolution default
    enable_full_market: bool = False,
    enable_forecast: bool = False,
    forecaster_path: str = "models/intraday_forecaster.pkl",
    enable_market_forecast: bool = False,
    market_forecaster_path: str = "models/market_forecaster.pkl",
    enable_endogenous_dam: bool = False,
    dam_bidder_min_spread: float = 30.0,
    mfrr_activation_rate: float = 0.35,
    mfrr_price_cap: float = 500.0
) -> str:
    """
    I train the unified multi-market model.

    Args:
        data_path: Path to unified training data
        output_dir: Directory to save model and logs
        total_timesteps: Total training steps
        n_envs: Number of parallel environments for speedup
        learning_rate: Learning rate for optimizer
        n_steps: Steps per update
        batch_size: Batch size for updates
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        seed: Random seed
        device: Training device ("auto", "cpu", "cuda")

    Returns:
        Path to saved model
    """
    print("=" * 60)
    print("UNIFIED MULTI-MARKET MODEL TRAINING")
    print("=" * 60)

    # I create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(output_dir) / f"unified_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {model_dir}")
    print(f"Training timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Time step: {time_step_hours*60:.0f}-min resolution ({1/time_step_hours:.0f} steps/hour)")
    if enable_full_market:
        print(f"Full Market Mode: IDA1/2/3/XBID + Free Bids (75 obs, 5 action dims)")
    if enable_forecast:
        print(f"Forecast Mode: IntraDay LightGBM (+9 obs, path={forecaster_path})")
    if enable_market_forecast:
        print(f"Market Forecast Mode: DAM+ID+Imbalance (+15 obs, path={market_forecaster_path})")
    if enable_endogenous_dam:
        print(f"Endogenous DAM: forecast-powered commitments (min_spread={dam_bidder_min_spread} EUR)")

    # I resolve data path
    if not Path(data_path).exists():
        data_path = project_root / data_path

    # I load data once and split temporally to avoid data leakage
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded full dataset: {len(df)} rows")

    # I perform temporal train/test split (80/20 by date)
    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx].copy()
    df_eval = df.iloc[split_idx:].copy()
    print(f"Temporal split: train={len(df_train)} rows ({df_train.index[0]} to {df_train.index[-1]})")
    print(f"                eval ={len(df_eval)} rows ({df_eval.index[0]} to {df_eval.index[-1]})")

    # I create multiple parallel training environments for speedup
    print(f"\nCreating {n_envs} parallel training environments...")

    # I convert episode lengths from hours to steps
    train_episode_steps = int(168 / time_step_hours)  # 1 week
    eval_episode_steps = int(336 / time_step_hours)   # 2 weeks

    def make_train_env(env_id: int):
        """I create a training environment with unique random seed."""
        def _init():
            env = BatteryEnvUnified(
                df=df_train,
                time_step_hours=time_step_hours,
                episode_length=train_episode_steps,
                random_start=True,
                enable_full_market=enable_full_market,
                enable_forecast=enable_forecast,
                forecaster_path=forecaster_path,
                forecast_noise=True,   # Training: noise on
                enable_market_forecast=enable_market_forecast,
                market_forecaster_path=market_forecaster_path,
                enable_endogenous_dam=enable_endogenous_dam,
                dam_bidder_min_spread=dam_bidder_min_spread,
                mfrr_activation_rate=mfrr_activation_rate,
                mfrr_price_cap=mfrr_price_cap,
            )
            # I set unique seed for each environment
            env.reset(seed=seed + env_id if seed else env_id)
            return env
        return _init

    # I create vectorized environment with multiple parallel envs
    train_env = MaskableVecEnv([make_train_env(i) for i in range(n_envs)])

    # I add VecNormalize for observation normalization only.
    # I disabled norm_reward because the reward_scale (0.01) in the
    # calculator already handles scaling, and double normalization
    # (scale + VecNorm + clip) was creating vanishing gradients and
    # stale return statistics when the degradation curriculum shifted
    # the reward distribution mid-training.
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # I create evaluation environment on SEPARATE held-out data
    print("Creating evaluation environment (held-out data)...")

    def make_eval_env():
        return BatteryEnvUnified(
            df=df_eval,
            time_step_hours=time_step_hours,
            episode_length=eval_episode_steps,
            random_start=True,
            enable_full_market=enable_full_market,
            enable_forecast=enable_forecast,
            forecaster_path=forecaster_path,
            forecast_noise=False,  # Eval: deterministic forecasts
            enable_market_forecast=enable_market_forecast,
            market_forecaster_path=market_forecaster_path,
            enable_endogenous_dam=enable_endogenous_dam,
            dam_bidder_min_spread=dam_bidder_min_spread,
            mfrr_activation_rate=mfrr_activation_rate,
            mfrr_price_cap=mfrr_price_cap,
        )

    eval_env = MaskableVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for eval
        training=False
    )

    # I create callbacks
    print("Setting up callbacks...")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="unified_model"
    )

    curriculum_callback = DegradationCurriculumCallback(
        target_degradation=5.0,  # I reduced from 15 to 5 EUR/MWh (realistic Li-ion BESS)
        warmup_steps=min(500_000, total_timesteps // 5)
    )

    logging_callback = TensorboardLoggingCallback()

    entropy_callback = EntropyAnnealingCallback(
        start_ent=ent_coef,   # I start from the user-specified ent_coef (default 0.03)
        end_ent=0.005         # I decay to 0.005 by end of training
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        curriculum_callback,
        entropy_callback,
        logging_callback
    ])

    # I create or load the model
    if resume_from:
        # I resume training from a saved checkpoint with fallback search
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            # I search for the best available model file in priority order
            candidates = [
                resume_path / "final_model.zip",
                resume_path / "best" / "best_model.zip",
            ]
            # I also check for checkpoints (latest first)
            cp_dir = resume_path / "checkpoints"
            if cp_dir.exists():
                checkpoints = sorted(cp_dir.glob("*.zip"), reverse=True)
                candidates.extend(checkpoints)

            model_file = None
            for c in candidates:
                if c.exists():
                    model_file = c
                    break
            if model_file is None:
                raise FileNotFoundError(f"No model found in {resume_path}")
            vec_file = resume_path / "vec_normalize_unified.pkl"
        else:
            model_file = resume_path
            vec_file = resume_path.parent / "vec_normalize_unified.pkl"

        print(f"\nResuming training from: {model_file}")
        model = MaskablePPO.load(
            str(model_file),
            env=train_env,
            device=device,
            tensorboard_log=str(model_dir / "tensorboard")
        )

        # I load VecNormalize stats to continue with the same normalization
        if vec_file.exists():
            import pickle
            with open(vec_file, 'rb') as f:
                old_vec = pickle.load(f)
            train_env.obs_rms = old_vec.obs_rms
            train_env.ret_rms = old_vec.ret_rms
            print(f"  Loaded VecNormalize stats from: {vec_file}")

        print(f"  Learning rate: {model.learning_rate}")
        print(f"  Entropy coefficient: {model.ent_coef}")
    else:
        print("\nCreating MaskablePPO model...")
        print(f"  Learning rate: {learning_rate}")
        print(f"  N steps: {n_steps}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gamma: {gamma}")
        print(f"  Entropy coefficient: {ent_coef} (annealing to 0.005)")
        print(f"  Value function coefficient: 0.25 (reduced from 0.5 default)")

        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=0.25,  # I reduced from 0.5 default — value_loss was 1000x policy_loss
            verbose=1,
            tensorboard_log=str(model_dir / "tensorboard"),
            seed=seed,
            device=device,
            policy_kwargs={
                "net_arch": dict(pi=[256, 256], vf=[256, 256])
            }
        )

    # I train the model
    reset_timesteps = resume_from is None
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    if resume_from:
        print("Continuing from previous checkpoint (reset_num_timesteps=False)")
    print("This will take several hours. Check tensorboard for progress.")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # I save the final model
    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to {final_model_path}")

    # I save the VecNormalize
    vec_normalize_path = model_dir / "vec_normalize_unified.pkl"
    train_env.save(str(vec_normalize_path))
    print(f"VecNormalize saved to {vec_normalize_path}")

    # I save training configuration
    config = {
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'seed': seed,
        'data_path': str(data_path),
        'time_step_hours': time_step_hours,
        'enable_full_market': enable_full_market,
        'enable_forecast': enable_forecast,
        'forecaster_path': forecaster_path,
        'enable_market_forecast': enable_market_forecast,
        'market_forecaster_path': market_forecaster_path,
        'enable_endogenous_dam': enable_endogenous_dam,
        'dam_bidder_min_spread': dam_bidder_min_spread,
        'mfrr_activation_rate': mfrr_activation_rate,
        'mfrr_price_cap': mfrr_price_cap,
        'timestamp': timestamp
    }

    config_path = model_dir / "training_config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {final_model_path}")
    print(f"Best model: {model_dir / 'best' / 'best_model.zip'}")
    print(f"Tensorboard: tensorboard --logdir {model_dir / 'tensorboard'}")

    return str(final_model_path)


def evaluate_model(
    model_path: str,
    data_path: str = "data/unified_multimarket_training_v2.csv",
    n_episodes: int = 10,
    use_eval_split: bool = True
) -> Dict:
    """
    I evaluate a trained unified model.

    I load training_config.json to reconstruct the exact env settings
    (time_step_hours, enable_full_market, enable_forecast) that were
    used during training.  I use the held-out eval split (last 20%) by
    default so the evaluation is on unseen data.
    """
    import json
    from sb3_contrib import MaskablePPO

    print(f"Evaluating model: {model_path}")

    model_dir = Path(model_path).parent

    # ----------------------------------------------------------------
    # 1. I load training config to match env settings exactly
    # ----------------------------------------------------------------
    config_path = model_dir / "training_config.json"
    train_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            train_cfg = json.load(f)
        print(f"  Loaded training config from {config_path}")

    time_step_hours = train_cfg.get('time_step_hours', 1.0)
    enable_full_market = train_cfg.get('enable_full_market', False)
    enable_forecast = train_cfg.get('enable_forecast', False)
    forecaster_path = train_cfg.get('forecaster_path', 'models/intraday_forecaster.pkl')
    enable_market_forecast = train_cfg.get('enable_market_forecast', False)
    market_forecaster_path = train_cfg.get('market_forecaster_path', 'models/market_forecaster.pkl')
    enable_endogenous_dam = train_cfg.get('enable_endogenous_dam', False)
    dam_bidder_min_spread = train_cfg.get('dam_bidder_min_spread', 30.0)
    mfrr_activation_rate = train_cfg.get('mfrr_activation_rate', 0.35)
    mfrr_price_cap = train_cfg.get('mfrr_price_cap', 500.0)

    print(f"  time_step_hours={time_step_hours}, "
          f"full_market={enable_full_market}, "
          f"forecast={enable_forecast}, "
          f"market_forecast={enable_market_forecast}")
    print(f"  endogenous_dam={enable_endogenous_dam}, "
          f"mfrr: activation_rate={mfrr_activation_rate:.0%}, "
          f"price_cap={mfrr_price_cap:.0f} EUR/MWh")

    # ----------------------------------------------------------------
    # 2. I resolve data path and load data
    # ----------------------------------------------------------------
    resolved_path = Path(data_path)
    if not resolved_path.exists():
        resolved_path = project_root / data_path
    if not resolved_path.exists():
        cfg_data = train_cfg.get('data_path', data_path)
        resolved_path = Path(cfg_data)
        if not resolved_path.exists():
            resolved_path = project_root / cfg_data

    df = pd.read_csv(resolved_path, index_col=0, parse_dates=True)
    print(f"  Loaded data: {len(df)} rows from {resolved_path}")

    # I use the held-out eval split (last 20%) for fair comparison
    if use_eval_split:
        split_idx = int(len(df) * 0.80)
        df_eval = df.iloc[split_idx:].copy()
        print(f"  Using eval split: {len(df_eval)} rows "
              f"({df_eval.index[0]} to {df_eval.index[-1]})")
    else:
        df_eval = df
        print(f"  Using full dataset for evaluation")

    # ----------------------------------------------------------------
    # 3. I load the MaskablePPO model
    # ----------------------------------------------------------------
    model = MaskablePPO.load(model_path)

    # ----------------------------------------------------------------
    # 4. I create evaluation environment with MATCHING settings
    # ----------------------------------------------------------------
    eval_episode_steps = int(336 / time_step_hours)  # 2 weeks

    def _make_eval():
        env = BatteryEnvUnified(
            df=df_eval,
            time_step_hours=time_step_hours,
            episode_length=eval_episode_steps,
            random_start=True,
            enable_full_market=enable_full_market,
            enable_forecast=enable_forecast,
            forecaster_path=forecaster_path,
            forecast_noise=False,
            enable_market_forecast=enable_market_forecast,
            market_forecaster_path=market_forecaster_path,
            enable_endogenous_dam=enable_endogenous_dam,
            dam_bidder_min_spread=dam_bidder_min_spread,
            mfrr_activation_rate=mfrr_activation_rate,
            mfrr_price_cap=mfrr_price_cap,
        )
        return Monitor(env)

    eval_vec = MaskableVecEnv([_make_eval])

    # ----------------------------------------------------------------
    # 5. I load VecNormalize stats and wrap the eval env
    # ----------------------------------------------------------------
    vec_path = model_dir / "vec_normalize_unified.pkl"
    if vec_path.exists():
        eval_vec = VecNormalize.load(str(vec_path), eval_vec)
        eval_vec.training = False
        eval_vec.norm_reward = False
        print(f"  VecNormalize loaded from {vec_path} "
              f"(obs_rms mean[:3]={eval_vec.obs_rms.mean[:3].round(3)})")
    else:
        print(f"  WARNING: No VecNormalize found at {vec_path} "
              f"-- model will receive un-normalized observations!")

    # ----------------------------------------------------------------
    # 6. I run evaluation episodes
    # ----------------------------------------------------------------
    results = {
        'episode_profits': [],
        'episode_cycles': [],
        'episode_lengths': [],
        'afrr_activations': [],
        'dam_violations': [],
        'per_market': []
    }

    for episode in range(n_episodes):
        obs = eval_vec.reset()
        done = False
        episode_profit = 0.0
        episode_cycles = 0.0
        episode_length = 0
        afrr_activations = 0
        dam_violations = 0
        last_info = {}

        while not done:
            action_masks = eval_vec.env_method("action_masks")[0]
            action, _ = model.predict(obs, deterministic=True,
                                      action_masks=action_masks)
            obs, reward, done, info = eval_vec.step(action)

            info = info[0]
            last_info = info
            episode_profit += info.get('net_profit', 0)
            episode_length += 1

            if info.get('afrr_activated', False):
                afrr_activations += 1
            if info.get('dam_shortfall_mw', 0) > 0.1:
                dam_violations += 1

        # I use the env's own cumulative cycle counter (accurate)
        episode_cycles = last_info.get('total_cycles', 0)

        results['episode_profits'].append(episode_profit)
        results['episode_cycles'].append(episode_cycles)
        results['episode_lengths'].append(episode_length)
        results['afrr_activations'].append(afrr_activations)
        results['dam_violations'].append(dam_violations)

        # I log per-market breakdown from the final info
        market_summary = {
            'dam': last_info.get('dam_profit', 0),
            'intraday': last_info.get('intraday_profit', 0),
            'afrr_cap': last_info.get('afrr_capacity_profit', 0),
            'afrr_energy': last_info.get('afrr_energy_profit', 0),
            'mfrr': last_info.get('mfrr_profit', 0),
        }
        results['per_market'].append(market_summary)

        print(f"Episode {episode + 1}: Profit={episode_profit:,.0f} EUR, "
              f"Cycles={episode_cycles:.2f}, "
              f"DAM={market_summary['dam']:,.0f}, "
              f"aFRR_cap={market_summary['afrr_cap']:,.0f}, "
              f"ID={market_summary['intraday']:,.0f}")

    # ----------------------------------------------------------------
    # 7. I calculate summary statistics
    # ----------------------------------------------------------------
    results['mean_profit'] = np.mean(results['episode_profits'])
    results['std_profit'] = np.std(results['episode_profits'])
    results['mean_cycles'] = np.mean(results['episode_cycles'])
    results['profit_per_cycle'] = (
        results['mean_profit'] / max(0.01, results['mean_cycles'])
    )
    total_steps = max(1, np.sum(results['episode_lengths']))
    results['dam_violation_rate'] = np.sum(results['dam_violations']) / total_steps
    results['afrr_activation_rate'] = np.sum(results['afrr_activations']) / total_steps

    # I aggregate per-market means
    market_keys = ['dam', 'intraday', 'afrr_cap', 'afrr_energy', 'mfrr']
    results['mean_per_market'] = {
        k: np.mean([m[k] for m in results['per_market']]) for k in market_keys
    }

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes:       {n_episodes}")
    print(f"Mean Profit:    {results['mean_profit']:,.0f} EUR "
          f"(+/-{results['std_profit']:,.0f})")
    print(f"Mean Cycles:    {results['mean_cycles']:.2f}")
    print(f"Profit/Cycle:   {results['profit_per_cycle']:,.0f} EUR")
    print(f"DAM Violations: {results['dam_violation_rate']:.2%}")
    print(f"aFRR Activations: {results['afrr_activation_rate']:.2%}")
    print(f"\nPer-Market Breakdown (mean):")
    for k, v in results['mean_per_market'].items():
        print(f"  {k:>12s}: {v:>+10,.0f} EUR")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Unified Multi-Market Model")
    parser.add_argument("--data", type=str, default="data/unified_multimarket_training_v2.csv",
                        help="Path to training data")
    parser.add_argument("--output", type=str, default="models/unified",
                        help="Output directory")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--ent_coef", type=float, default=0.03,
                        help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Training device (auto/cpu/cuda)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model directory to resume training from")
    parser.add_argument("--time_step", type=float, default=0.25,
                        help="Time step in hours (0.25 = 15-min, 1.0 = hourly)")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to model for evaluation (skip training)")
    parser.add_argument("--full-market", action='store_true',
                        help="Enable IDA1/2/3/XBID + Free Bids (75 obs, 5 action dims)")
    parser.add_argument("--enable-forecast", action='store_true',
                        help="Enable IntraDay price forecast features (+9 obs)")
    parser.add_argument("--forecaster-path", type=str,
                        default="models/intraday_forecaster.pkl",
                        help="Path to trained IntraDayForecaster pickle")
    parser.add_argument("--enable-market-forecast", action='store_true',
                        help="Enable DAM+ID+Imbalance forecast features (+15 obs)")
    parser.add_argument("--market-forecaster-path", type=str,
                        default="models/market_forecaster.pkl",
                        help="Path to trained MarketForecaster pickle")
    parser.add_argument("--enable-endogenous-dam", action='store_true',
                        help="Enable forecast-powered DAM commitment generation")
    parser.add_argument("--dam-bidder-min-spread", type=float, default=30.0,
                        help="Minimum DAM spread (EUR) to trigger trading (default: 30)")
    parser.add_argument("--mfrr-activation-rate", type=float, default=0.35,
                        help="mFRR bid activation probability (default: 0.35)")
    parser.add_argument("--mfrr-price-cap", type=float, default=500.0,
                        help="mFRR settlement price cap EUR/MWh (default: 500)")

    args = parser.parse_args()

    if args.eval:
        # I run evaluation only (config auto-loaded from training_config.json)
        results = evaluate_model(args.eval, args.data, use_eval_split=True)
    else:
        # I run training
        model_path = train_unified_model(
            data_path=args.data,
            output_dir=args.output,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device=args.device,
            resume_from=args.resume,
            time_step_hours=args.time_step,
            enable_full_market=args.full_market,
            enable_forecast=args.enable_forecast,
            forecaster_path=args.forecaster_path,
            enable_market_forecast=args.enable_market_forecast,
            market_forecaster_path=args.market_forecaster_path,
            enable_endogenous_dam=args.enable_endogenous_dam,
            dam_bidder_min_spread=args.dam_bidder_min_spread,
            mfrr_activation_rate=args.mfrr_activation_rate,
            mfrr_price_cap=args.mfrr_price_cap
        )

        # I run evaluation on trained model
        print("\nRunning evaluation on trained model...")
        evaluate_model(model_path, args.data)
