"""
Training Script for SAC Unified Multi-Market Battery Trading Model

I train a single SAC model with continuous action space for all markets:
- DAM commitment execution
- IntraDay energy trading
- aFRR capacity commitment and activation
- mFRR energy trading

Key Differences from PPO pipeline (train_unified.py):
1. Continuous Box(4,) action space via BatteryEnvUnifiedSAC wrapper
2. Off-policy SAC algorithm with replay buffer (sample efficient)
3. Penalty-based soft constraints instead of hard action masking
4. Automatic entropy tuning (no EntropyAnnealingCallback needed)
5. Standard DummyVecEnv (no MaskableVecEnv needed)
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
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from gym_envs.battery_env_unified import BatteryEnvUnified
from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC


class DegradationCurriculumCallback(BaseCallback):
    """
    I implement curriculum learning for degradation cost.

    The degradation cost starts at 0 and ramps up to the target value
    over the warmup period. This helps the agent learn profitable trading
    before considering battery wear.

    I access the inner env's reward_calculator through the SAC wrapper's
    property delegation.
    """

    def __init__(
        self,
        target_degradation: float = 5.0,
        warmup_steps: int = 500_000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.target_degradation = target_degradation
        self.warmup_steps = warmup_steps
        self.current_degradation = 0.0
        self._warned_ev_drop = False

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        self.current_degradation = progress * self.target_degradation

        # I use env_method() for SubprocVecEnv compatibility
        try:
            self.training_env.env_method('set_degradation_cost', self.current_degradation)
        except AttributeError:
            # I fall back to direct access for DummyVecEnv without env_method
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    if hasattr(env, 'set_degradation_cost'):
                        env.set_degradation_cost(self.current_degradation)

        if self.num_timesteps % 10000 == 0:
            self.logger.record('curriculum/degradation_cost', self.current_degradation)

        return True


class SACLoggingCallback(BaseCallback):
    """
    I log additional metrics to Tensorboard including per-market statistics
    and SAC-specific penalty metrics.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
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

        # I track SAC-specific penalty metrics
        self.step_penalties = []
        self.step_count = 0

        # I track participation
        self.step_afrr_bids = 0
        self.step_intraday_trades = 0
        self.step_mfrr_trades = 0
        self.step_free_bid_trades = 0
        self.step_free_bid_sells = 0
        self.step_free_bid_buys = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            self.step_count += 1

            # I track SAC penalties
            if 'sac_penalty' in info:
                self.step_penalties.append(info['sac_penalty'])

            # I track participation via decoded_mw (v3 direct execution)
            decoded = info.get('decoded_mw', {})
            if decoded:
                if abs(decoded.get('afrr', 0)) > 0.5:
                    self.step_afrr_bids += 1
                if abs(decoded.get('xbid', 0)) > 0.5:
                    self.step_intraday_trades += 1
                if abs(decoded.get('mfrr', 0)) > 0.5:
                    self.step_mfrr_trades += 1
            else:
                # I fallback for legacy discrete_action format
                discrete = info.get('discrete_action', [0, 0, 5, 5])
                if len(discrete) >= 4:
                    if discrete[0] > 0: self.step_afrr_bids += 1
                    if discrete[2] != 5: self.step_intraday_trades += 1
                    if discrete[3] != 5: self.step_mfrr_trades += 1

            # I track Free Bid trades (full-market mode routes mFRR to free_bid_energy_mw)
            free_bid_mw = info.get('free_bid_energy_mw', 0)
            if abs(free_bid_mw) > 0.1:
                self.step_free_bid_trades += 1
                if free_bid_mw > 0:
                    self.step_free_bid_sells += 1
                else:
                    self.step_free_bid_buys += 1

            # I check for episode end
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
            avg_profit = np.mean(self.episode_profits[-100:])
            avg_cycles = np.mean(self.episode_cycles[-100:]) if self.episode_cycles else 0
            self.logger.record('custom/avg_profit', avg_profit)
            self.logger.record('custom/avg_cycles', avg_cycles)
            if avg_cycles > 0:
                self.logger.record('custom/profit_per_cycle', avg_profit / avg_cycles)

            # I log per-market profits
            if self.dam_profits:
                self.logger.record('markets/dam_profit', np.mean(self.dam_profits[-100:]))
            if self.intraday_profits:
                self.logger.record('markets/intraday_profit', np.mean(self.intraday_profits[-100:]))
            if self.afrr_capacity_profits:
                self.logger.record('markets/afrr_capacity_profit', np.mean(self.afrr_capacity_profits[-100:]))
            if self.afrr_energy_profits:
                self.logger.record('markets/afrr_energy_profit', np.mean(self.afrr_energy_profits[-100:]))
            if self.mfrr_profits:
                self.logger.record('markets/mfrr_profit', np.mean(self.mfrr_profits[-100:]))

            # I log full-market mode metrics from episode-level accumulators
            if self.ida_profits:
                self.logger.record('markets/ida_profit', np.mean(self.ida_profits[-100:]))
            if self.xbid_profits:
                self.logger.record('markets/xbid_profit', np.mean(self.xbid_profits[-100:]))
            if self.free_bid_profits:
                self.logger.record('markets/free_bid_profit', np.mean(self.free_bid_profits[-100:]))

            # I log SAC penalty metrics
            if self.step_penalties:
                recent = self.step_penalties[-1000:]
                self.logger.record('sac/mean_penalty', np.mean(recent))
                self.logger.record('sac/max_penalty', np.max(recent))
                self.logger.record('sac/penalty_nonzero_rate',
                                   np.mean([p > 0 for p in recent]))

            # I log participation rates
            if self.step_count > 0:
                self.logger.record('participation/afrr_bid_rate',
                                   self.step_afrr_bids / self.step_count)
                self.logger.record('participation/intraday_trade_rate',
                                   self.step_intraday_trades / self.step_count)
                self.logger.record('participation/mfrr_trade_rate',
                                   self.step_mfrr_trades / self.step_count)
                self.logger.record('participation/free_bid_trade_rate',
                                   self.step_free_bid_trades / self.step_count)
                self.logger.record('participation/free_bid_sell_rate',
                                   self.step_free_bid_sells / self.step_count)
                self.logger.record('participation/free_bid_buy_rate',
                                   self.step_free_bid_buys / self.step_count)

            # I reset step counters
            self.step_afrr_bids = 0
            self.step_intraday_trades = 0
            self.step_mfrr_trades = 0
            self.step_free_bid_trades = 0
            self.step_free_bid_sells = 0
            self.step_free_bid_buys = 0
            self.step_count = 0
            self.step_penalties = []

        return True


def train_sac_unified(
    data_path: str = "data/unified_multimarket_training_v2.csv",
    output_dir: str = "models/unified_sac",
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,  # I use 8 parallel envs for faster buffer filling
    algorithm: str = "SAC",  # I support SAC or TD3
    learning_rate: float = 3e-4,
    buffer_size: int = 2_000_000,
    learning_starts: int = 10_000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.999,  # ~1000 steps horizon = 250h = 10.4 days at 15-min
    ent_coef: str = "auto",
    seed: int = 42,
    device: str = "auto",
    resume_from: str = None,
    time_step_hours: float = 0.25,
    enable_full_market: bool = False,
    enable_forecast: bool = False,
    forecaster_path: str = "models/intraday_forecaster.pkl",
    enable_market_forecast: bool = False,
    market_forecaster_path: str = "models/market_forecaster.pkl",
    enable_endogenous_dam: bool = True,  # DAM optimizer manages daily schedule
    dam_bidder_min_spread: float = 30.0,
    mfrr_activation_rate: float = 0.20,  # I reduced from 0.35 — real ADMIE avg is 15-25%
    mfrr_price_cap: float = 1000.0,  # I raised from 500 — real Greek mFRR reaches ~1190 EUR/MWh
    # SAC-specific soft penalty overrides (hard violations handled by action filtering)
    soc_soft_margin_penalty: float = 60.0,
    cycle_penalty: float = 2000.0,
) -> str:
    """
    I train the unified multi-market SAC model.

    Returns:
        Path to saved model
    """
    print("=" * 60)
    print("SAC UNIFIED MULTI-MARKET MODEL TRAINING")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(output_dir) / f"sac_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {model_dir}")
    print(f"Training timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Time step: {time_step_hours*60:.0f}-min resolution")
    print(f"Algorithm: SAC (continuous action space, off-policy)")
    print(f"Penalties: SoC_margin={soc_soft_margin_penalty}, "
          f"cycle={cycle_penalty} (action filtering for hard constraints)")

    # I resolve data path
    if not Path(data_path).exists():
        data_path = project_root / data_path

    # I load data and split temporally (80/20)
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded full dataset: {len(df)} rows")

    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx].copy()
    df_eval = df.iloc[split_idx:].copy()
    print(f"Temporal split: train={len(df_train)} rows "
          f"({df_train.index[0]} to {df_train.index[-1]})")
    print(f"                eval ={len(df_eval)} rows "
          f"({df_eval.index[0]} to {df_eval.index[-1]})")

    # I prepare soft penalty config (hard violations handled by action filtering)
    penalties = {
        'soc_soft_margin': soc_soft_margin_penalty,
        'cycle_excess': cycle_penalty,
    }

    # I convert episode lengths from hours to steps
    train_episode_steps = int(168 / time_step_hours)  # 1 week
    eval_episode_steps = int(168 / time_step_hours)   # 1 week (same as training)

    # I build shared env kwargs (excluding random_start and forecast_noise
    # which differ between train and eval)
    env_kwargs = dict(
        time_step_hours=time_step_hours,
        enable_full_market=enable_full_market,
        enable_forecast=enable_forecast,
        forecaster_path=forecaster_path,
        enable_market_forecast=enable_market_forecast,
        market_forecaster_path=market_forecaster_path,
        enable_endogenous_dam=enable_endogenous_dam,
        dam_bidder_min_spread=dam_bidder_min_spread,
        mfrr_activation_rate=mfrr_activation_rate,
        mfrr_price_cap=mfrr_price_cap,
    )

    print(f"\nCreating {n_envs} training environment(s)...")

    def make_train_env(env_id: int):
        """I create a SAC-wrapped training environment."""
        def _init():
            inner = BatteryEnvUnified(
                df=df_train,
                episode_length=train_episode_steps,
                random_start=True,
                forecast_noise=True,
                **env_kwargs
            )
            inner.reset(seed=seed + env_id if seed else env_id)
            sac_env = BatteryEnvUnifiedSAC(inner_env=inner, penalties=penalties)
            return Monitor(sac_env)
        return _init

    # I use SubprocVecEnv for true parallelism when n_envs > 1
    if n_envs > 1:
        train_env = SubprocVecEnv([make_train_env(i) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_train_env(i) for i in range(n_envs)])

    # I add VecNormalize for observation normalization (no reward norm)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # I create evaluation environment
    print("Creating evaluation environment (held-out data)...")

    def make_eval_env():
        inner = BatteryEnvUnified(
            df=df_eval,
            episode_length=eval_episode_steps,
            random_start=True,
            forecast_noise=False,
            **env_kwargs
        )
        return Monitor(BatteryEnvUnifiedSAC(inner_env=inner, penalties=penalties))

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
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
        name_prefix="sac_unified"
    )

    curriculum_callback = DegradationCurriculumCallback(
        target_degradation=5.0,
        warmup_steps=min(500_000, total_timesteps // 5)
    )

    logging_callback = SACLoggingCallback()

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        curriculum_callback,
        logging_callback
    ])

    # I create or load the model
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            candidates = [
                resume_path / "final_model.zip",
                resume_path / "best" / "best_model.zip",
            ]
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
            vec_file = resume_path / "vec_normalize_sac.pkl"
        else:
            model_file = resume_path
            vec_file = resume_path.parent / "vec_normalize_sac.pkl"

        print(f"\nResuming training from: {model_file}")
        model = SAC.load(
            str(model_file),
            env=train_env,
            device=device,
            tensorboard_log=str(model_dir / "tensorboard")
        )

        if vec_file.exists():
            import pickle
            with open(vec_file, 'rb') as f:
                old_vec = pickle.load(f)
            train_env.obs_rms = old_vec.obs_rms
            train_env.ret_rms = old_vec.ret_rms
            print(f"  Loaded VecNormalize stats from: {vec_file}")
    else:
        print("\nCreating SAC model...")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Buffer size: {buffer_size:,}")
        print(f"  Learning starts: {learning_starts:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Tau: {tau}")
        print(f"  Gamma: {gamma}")
        print(f"  Entropy coefficient: {ent_coef}")

        # I select algorithm based on parameter
        algo_class = SAC if algorithm.upper() == "SAC" else TD3
        algo_kwargs = dict(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            verbose=1,
            tensorboard_log=str(model_dir / "tensorboard"),
            seed=seed,
            device=device,
            policy_kwargs={
                "net_arch": dict(pi=[512, 256], qf=[512, 256])
            }
        )
        # I add ent_coef only for SAC (TD3 uses deterministic policy + noise)
        if algorithm.upper() == "SAC":
            algo_kwargs["ent_coef"] = ent_coef

        model = algo_class(**algo_kwargs)

    # I train the model
    reset_timesteps = resume_from is None
    print(f"\nStarting {algorithm.upper()} training for {total_timesteps:,} timesteps...")
    if resume_from:
        print("Continuing from previous checkpoint (reset_num_timesteps=False)")
    print("This will take several hours. Check tensorboard for progress.")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,  # I disabled for background training compatibility
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # I save the final model
    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to {final_model_path}")

    # I save VecNormalize
    vec_normalize_path = model_dir / "vec_normalize_sac.pkl"
    train_env.save(str(vec_normalize_path))
    print(f"VecNormalize saved to {vec_normalize_path}")

    # I save training configuration
    config = {
        'algorithm': 'SAC',
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'tau': tau,
        'gamma': gamma,
        'algorithm': algorithm.upper(),
        'ent_coef': str(ent_coef),
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
        'penalties': penalties,
        'timestamp': timestamp
    }

    config_path = model_dir / "training_config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")

    print("\n" + "=" * 60)
    print("SAC TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {final_model_path}")
    print(f"Best model: {model_dir / 'best' / 'best_model.zip'}")
    print(f"Tensorboard: tensorboard --logdir {model_dir / 'tensorboard'}")

    return str(final_model_path)


def evaluate_sac_model(
    model_path: str,
    data_path: str = "data/unified_multimarket_training_v2.csv",
    n_episodes: int = 10,
    use_eval_split: bool = True
) -> Dict:
    """
    I evaluate a trained SAC unified model.

    I load training_config.json to reconstruct the exact env settings
    used during training, then run deterministic evaluation episodes.
    """
    import json

    print(f"Evaluating SAC model: {model_path}")

    model_dir = Path(model_path).parent

    # I load training config
    config_path = model_dir / "training_config.json"
    train_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            train_cfg = json.load(f)
        print(f"  Loaded training config from {config_path}")

    time_step_hours = train_cfg.get('time_step_hours', 0.25)
    enable_full_market = train_cfg.get('enable_full_market', False)
    enable_forecast = train_cfg.get('enable_forecast', False)
    forecaster_path = train_cfg.get('forecaster_path', 'models/intraday_forecaster.pkl')
    enable_market_forecast = train_cfg.get('enable_market_forecast', False)
    market_forecaster_path = train_cfg.get('market_forecaster_path', 'models/market_forecaster.pkl')
    enable_endogenous_dam = train_cfg.get('enable_endogenous_dam', False)
    dam_bidder_min_spread = train_cfg.get('dam_bidder_min_spread', 30.0)
    mfrr_activation_rate = train_cfg.get('mfrr_activation_rate', 0.20)
    mfrr_price_cap = train_cfg.get('mfrr_price_cap', 1000.0)
    penalties = train_cfg.get('penalties', {})

    print(f"  time_step_hours={time_step_hours}, "
          f"full_market={enable_full_market}, "
          f"forecast={enable_forecast}")

    # I resolve data path
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

    if use_eval_split:
        split_idx = int(len(df) * 0.80)
        df_eval = df.iloc[split_idx:].copy()
        print(f"  Using eval split: {len(df_eval)} rows")
    else:
        df_eval = df

    # I load model (SAC or TD3) — I detect from training config
    algo_name = train_cfg.get('algorithm', 'SAC').upper()
    algo_class = TD3 if algo_name == 'TD3' else SAC
    print(f"  Algorithm: {algo_name}")
    model = algo_class.load(model_path)

    # I create evaluation environment
    eval_episode_steps = int(336 / time_step_hours)

    def _make_eval():
        inner = BatteryEnvUnified(
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
        return Monitor(BatteryEnvUnifiedSAC(inner_env=inner, penalties=penalties))

    eval_vec = DummyVecEnv([_make_eval])

    vec_path = model_dir / "vec_normalize_sac.pkl"
    if vec_path.exists():
        eval_vec = VecNormalize.load(str(vec_path), eval_vec)
        eval_vec.training = False
        eval_vec.norm_reward = False
        print(f"  VecNormalize loaded from {vec_path}")
    else:
        print(f"  WARNING: No VecNormalize found at {vec_path}")

    # I run evaluation episodes
    results = {
        'episode_profits': [],
        'episode_cycles': [],
        'episode_lengths': [],
        'total_penalties': [],
        'per_market': []
    }

    for episode in range(n_episodes):
        obs = eval_vec.reset()
        done = False
        episode_profit = 0.0
        episode_length = 0
        total_penalty = 0.0
        last_info = {}

        while not done:
            # I use deterministic=True for evaluation (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_vec.step(action)

            info = info[0]
            last_info = info
            episode_profit += info.get('net_profit', 0)
            total_penalty += info.get('sac_penalty', 0)
            episode_length += 1

        episode_cycles = last_info.get('total_cycles', 0)

        results['episode_profits'].append(episode_profit)
        results['episode_cycles'].append(episode_cycles)
        results['episode_lengths'].append(episode_length)
        results['total_penalties'].append(total_penalty)

        market_summary = {
            'dam': last_info.get('dam_profit', 0),
            'intraday': last_info.get('intraday_profit', 0),
            'afrr_cap': last_info.get('afrr_capacity_profit', 0),
            'afrr_energy': last_info.get('afrr_energy_profit', 0),
            'mfrr': last_info.get('mfrr_profit', 0),
            'ida': last_info.get('ida_profit', 0),
            'xbid': last_info.get('xbid_profit', 0),
            'free_bid': last_info.get('free_bid_profit', 0),
        }
        results['per_market'].append(market_summary)

        print(f"Episode {episode + 1}: Profit={episode_profit:,.0f} EUR, "
              f"Cycles={episode_cycles:.2f}, "
              f"Penalty={total_penalty:,.0f}, "
              f"DAM={market_summary['dam']:,.0f}, "
              f"aFRR_cap={market_summary['afrr_cap']:,.0f}, "
              f"ID={market_summary['intraday']:,.0f}, "
              f"mFRR={market_summary['mfrr']:,.0f}, "
              f"FreeBid={market_summary['free_bid']:,.0f}")

    # I calculate summary statistics
    results['mean_profit'] = np.mean(results['episode_profits'])
    results['std_profit'] = np.std(results['episode_profits'])
    results['mean_cycles'] = np.mean(results['episode_cycles'])
    results['profit_per_cycle'] = (
        results['mean_profit'] / max(0.01, results['mean_cycles'])
    )
    results['mean_penalty'] = np.mean(results['total_penalties'])

    market_keys = ['dam', 'intraday', 'afrr_cap', 'afrr_energy', 'mfrr',
                    'ida', 'xbid', 'free_bid']
    results['mean_per_market'] = {
        k: np.mean([m[k] for m in results['per_market']]) for k in market_keys
    }

    print("\n" + "=" * 50)
    print("SAC EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes:       {n_episodes}")
    print(f"Mean Profit:    {results['mean_profit']:,.0f} EUR "
          f"(+/-{results['std_profit']:,.0f})")
    print(f"Mean Cycles:    {results['mean_cycles']:.2f}")
    print(f"Profit/Cycle:   {results['profit_per_cycle']:,.0f} EUR")
    print(f"Mean Penalty:   {results['mean_penalty']:,.0f} EUR")
    print(f"\nPer-Market Breakdown (mean):")
    for k, v in results['mean_per_market'].items():
        print(f"  {k:>12s}: {v:>+10,.0f} EUR")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SAC Unified Multi-Market Model")
    parser.add_argument("--data", type=str, default="data/unified_multimarket_training_v2.csv",
                        help="Path to training data")
    parser.add_argument("--output", type=str, default="models/unified_sac",
                        help="Output directory")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
                        help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="Steps before learning begins")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for updates")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="Discount factor (0.999 = ~250h horizon at 15-min steps)")
    parser.add_argument("--ent-coef", type=str, default="auto",
                        help="Entropy coefficient ('auto' for learned temperature)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--algorithm", type=str, default="SAC",
                        choices=["SAC", "TD3"],
                        help="RL algorithm to use (SAC or TD3)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Training device (auto/cpu/cuda)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model directory to resume training from")
    parser.add_argument("--time_step", type=float, default=0.25,
                        help="Time step in hours (0.25 = 15-min, 1.0 = hourly)")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to model for evaluation (skip training)")
    parser.add_argument("--full-market", action='store_true',
                        help="Enable IDA1/2/3/XBID + Free Bids")
    parser.add_argument("--enable-forecast", action='store_true',
                        help="Enable IntraDay price forecast features")
    parser.add_argument("--forecaster-path", type=str,
                        default="models/intraday_forecaster.pkl")
    parser.add_argument("--enable-market-forecast", action='store_true',
                        help="Enable DAM+ID+Imbalance forecast features")
    parser.add_argument("--market-forecaster-path", type=str,
                        default="models/market_forecaster.pkl")
    parser.add_argument("--enable-endogenous-dam", action='store_true',
                        help="Enable forecast-powered DAM commitment generation")
    parser.add_argument("--dam-bidder-min-spread", type=float, default=30.0)
    parser.add_argument("--mfrr-activation-rate", type=float, default=0.20)
    parser.add_argument("--mfrr-price-cap", type=float, default=1000.0)
    # SAC-specific penalty args
    parser.add_argument("--soc-soft-margin-penalty", type=float, default=60.0,
                        help="SoC soft margin penalty EUR/MW (proximity to limits)")
    parser.add_argument("--cycle-penalty", type=float, default=2000.0,
                        help="Daily cycle excess penalty EUR/cycle (super-linear)")

    args = parser.parse_args()

    # I parse ent_coef: "auto" stays as string, numeric becomes float
    ent_coef = args.ent_coef
    if ent_coef != "auto":
        try:
            ent_coef = float(ent_coef)
        except ValueError:
            pass

    if args.eval:
        results = evaluate_sac_model(args.eval, args.data, use_eval_split=True)
    else:
        model_path = train_sac_unified(
            data_path=args.data,
            output_dir=args.output,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            algorithm=args.algorithm,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            ent_coef=ent_coef,
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
            mfrr_price_cap=args.mfrr_price_cap,
            soc_soft_margin_penalty=args.soc_soft_margin_penalty,
            cycle_penalty=args.cycle_penalty,
        )

        print("\nRunning evaluation on trained model...")
        evaluate_sac_model(model_path, args.data)
