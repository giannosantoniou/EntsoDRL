"""
Training Script for Unified Multi-Market Battery Trading Model

I train a single MaskablePPO model that handles ALL markets:
- DAM commitment execution
- IntraDay energy trading
- aFRR capacity commitment and activation
- mFRR energy trading

Key Features:
1. MultiDiscrete action space [5, 5, 21]
2. Hierarchical action masking
3. 58-feature observation space
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


class DegradationCurriculumCallback(BaseCallback):
    """
    I implement curriculum learning for degradation cost.

    The degradation cost starts at 0 and ramps up to the target value
    over the warmup period. This helps the agent learn profitable trading
    before considering battery wear.
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

    def _on_step(self) -> bool:
        # I calculate current degradation based on training progress
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        self.current_degradation = progress * self.target_degradation

        # I update the environment's reward calculator
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'reward_calculator'):
                    env.reward_calculator.degradation_cost = self.current_degradation

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
        self.afrr_capacity_profits = []
        self.afrr_energy_profits = []
        self.mfrr_profits = []

        # I track participation counts per episode
        self.afrr_bids_per_ep = []
        self.afrr_selections_per_ep = []
        self.afrr_activations_per_ep = []

        # I track step-level counts (reset each logging period)
        self.step_afrr_bids = 0
        self.step_afrr_selections = 0
        self.step_afrr_activations = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        # I extract info from the environment
        for info in self.locals.get('infos', []):
            self.step_count += 1

            # I track step-level aFRR participation
            if info.get('afrr_commitment', 0) > 0.1:
                self.step_afrr_bids += 1
            if info.get('is_selected', False):
                self.step_afrr_selections += 1
            if info.get('afrr_activated', False):
                self.step_afrr_activations += 1

            # I check for episode end (when total_profit is reported)
            if 'total_profit' in info:
                self.episode_profits.append(info['total_profit'])
            if 'total_cycles' in info:
                self.episode_cycles.append(info['total_cycles'])

            # I track per-market profits
            if 'dam_profit' in info:
                self.dam_profits.append(info['dam_profit'])
            if 'afrr_capacity_profit' in info:
                self.afrr_capacity_profits.append(info['afrr_capacity_profit'])
            if 'afrr_energy_profit' in info:
                self.afrr_energy_profits.append(info['afrr_energy_profit'])
            if 'mfrr_profit' in info:
                self.mfrr_profits.append(info['mfrr_profit'])

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
            if len(self.afrr_capacity_profits) > 0:
                self.logger.record('markets/afrr_capacity_profit', np.mean(self.afrr_capacity_profits[-100:]))
            if len(self.afrr_energy_profits) > 0:
                self.logger.record('markets/afrr_energy_profit', np.mean(self.afrr_energy_profits[-100:]))
            if len(self.mfrr_profits) > 0:
                self.logger.record('markets/mfrr_profit', np.mean(self.mfrr_profits[-100:]))

            # I log participation rates
            if self.step_count > 0:
                self.logger.record('participation/afrr_bid_rate', self.step_afrr_bids / self.step_count)
                self.logger.record('participation/afrr_selection_rate', self.step_afrr_selections / self.step_count)
                self.logger.record('participation/afrr_activation_rate', self.step_afrr_activations / self.step_count)

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
    episode_length: int = 168  # 1 week
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

    # I set default reward config
    if reward_config is None:
        reward_config = {
            'degradation_cost': 15.0,
            'dam_violation_penalty': 1500.0,
            'afrr_nonresponse_penalty': 2000.0,
            'reward_scale': 0.001
        }

    # I create environment (it already implements action_masks())
    env = BatteryEnvUnified(
        df=df,
        capacity_mwh=battery_params['capacity_mwh'],
        max_power_mw=battery_params['max_power_mw'],
        efficiency=battery_params['efficiency'],
        episode_length=episode_length,
        random_start=True,
        reward_config=reward_config
    )

    # I wrap with Monitor for logging
    env = Monitor(env)

    return env


def create_eval_env(
    data_path: str,
    battery_params: Optional[Dict] = None
) -> gym.Env:
    """I create the evaluation environment (deterministic)."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if battery_params is None:
        battery_params = {
            'capacity_mwh': 146.0,
            'max_power_mw': 30.0,
            'efficiency': 0.94
        }

    env = BatteryEnvUnified(
        df=df,
        capacity_mwh=battery_params['capacity_mwh'],
        max_power_mw=battery_params['max_power_mw'],
        efficiency=battery_params['efficiency'],
        episode_length=336,  # 2 weeks for evaluation
        random_start=False
    )

    env = Monitor(env)

    return env


def train_unified_model(
    data_path: str = "data/unified_multimarket_training.csv",
    output_dir: str = "models/unified",
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,  # I use 8 parallel environments for speedup
    learning_rate: float = 1e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.03,
    seed: int = 42,
    device: str = "auto",
    resume_from: str = None
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

    # I resolve data path
    if not Path(data_path).exists():
        data_path = project_root / data_path

    # I load data once to share across environments
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded training data: {len(df)} rows")

    # I create multiple parallel training environments for speedup
    print(f"\nCreating {n_envs} parallel training environments...")

    def make_train_env(env_id: int):
        """I create a training environment with unique random seed."""
        def _init():
            env = BatteryEnvUnified(
                df=df,
                episode_length=168,  # 1 week episodes
                random_start=True
            )
            # I set unique seed for each environment
            env.reset(seed=seed + env_id if seed else env_id)
            return env
        return _init

    # I create vectorized environment with multiple parallel envs
    train_env = MaskableVecEnv([make_train_env(i) for i in range(n_envs)])

    # I add VecNormalize for observation normalization
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    # I create evaluation environment (single env is fine for eval)
    print("Creating evaluation environment...")

    def make_eval_env():
        return BatteryEnvUnified(
            df=df,
            episode_length=336,  # 2 weeks for eval
            random_start=True
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
        target_degradation=15.0,
        warmup_steps=min(500_000, total_timesteps // 5)
    )

    logging_callback = TensorboardLoggingCallback()

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        curriculum_callback,
        logging_callback
    ])

    # I create or load the model
    if resume_from:
        # I resume training from a saved checkpoint
        resume_path = Path(resume_from)
        model_file = resume_path / "final_model.zip" if resume_path.is_dir() else resume_path
        vec_file = resume_path.parent / "vec_normalize_unified.pkl" if not resume_path.is_dir() else resume_path / "vec_normalize_unified.pkl"

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
        print(f"  Entropy coefficient: {ent_coef}")

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
    data_path: str = "data/unified_multimarket_training.csv",
    n_episodes: int = 10
) -> Dict:
    """I evaluate a trained unified model."""
    from sb3_contrib import MaskablePPO

    print(f"Evaluating model: {model_path}")

    # I load model
    model = MaskablePPO.load(model_path)

    # I try to load VecNormalize
    vec_path = Path(model_path).parent / "vec_normalize_unified.pkl"
    vec_normalize = None
    if vec_path.exists():
        vec_normalize = VecNormalize.load(str(vec_path), DummyVecEnv([lambda: create_eval_env(data_path)]))
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    # I create evaluation environment
    if not Path(data_path).exists():
        data_path = project_root / data_path

    eval_env = create_eval_env(data_path)
    eval_env = DummyVecEnv([lambda: eval_env])

    if vec_normalize is not None:
        eval_env = vec_normalize

    # I run evaluation
    results = {
        'episode_profits': [],
        'episode_cycles': [],
        'episode_lengths': [],
        'afrr_activations': [],
        'dam_violations': []
    }

    for episode in range(n_episodes):
        obs = eval_env.reset()
        done = False
        episode_profit = 0.0
        episode_cycles = 0.0
        episode_length = 0
        afrr_activations = 0
        dam_violations = 0

        while not done:
            action_masks = eval_env.env_method("action_masks")[0]
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, done, info = eval_env.step(action)

            info = info[0]
            episode_profit += info.get('net_profit', 0)
            episode_cycles += info.get('cycle_fraction', 0)
            episode_length += 1

            if info.get('afrr_activated', False):
                afrr_activations += 1
            if info.get('dam_shortfall_mw', 0) > 0.1:
                dam_violations += 1

        results['episode_profits'].append(episode_profit)
        results['episode_cycles'].append(episode_cycles)
        results['episode_lengths'].append(episode_length)
        results['afrr_activations'].append(afrr_activations)
        results['dam_violations'].append(dam_violations)

        print(f"Episode {episode + 1}: Profit={episode_profit:,.0f} EUR, Cycles={episode_cycles:.2f}")

    # I calculate summary statistics
    results['mean_profit'] = np.mean(results['episode_profits'])
    results['std_profit'] = np.std(results['episode_profits'])
    results['mean_cycles'] = np.mean(results['episode_cycles'])
    results['profit_per_cycle'] = results['mean_profit'] / max(0.01, results['mean_cycles'])
    results['dam_violation_rate'] = np.sum(results['dam_violations']) / np.sum(results['episode_lengths'])
    results['afrr_activation_rate'] = np.sum(results['afrr_activations']) / np.sum(results['episode_lengths'])

    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Mean Profit: {results['mean_profit']:,.0f} EUR (±{results['std_profit']:,.0f})")
    print(f"Mean Cycles: {results['mean_cycles']:.2f}")
    print(f"Profit per Cycle: {results['profit_per_cycle']:,.0f} EUR")
    print(f"DAM Violation Rate: {results['dam_violation_rate']:.2%}")
    print(f"aFRR Activation Rate: {results['afrr_activation_rate']:.2%}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Unified Multi-Market Model")
    parser.add_argument("--data", type=str, default="data/unified_multimarket_training.csv",
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
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to model for evaluation (skip training)")

    args = parser.parse_args()

    if args.eval:
        # I run evaluation only
        results = evaluate_model(args.eval, args.data)
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
            resume_from=args.resume
        )

        # I run evaluation on trained model
        print("\nRunning evaluation on trained model...")
        evaluate_model(model_path, args.data)
