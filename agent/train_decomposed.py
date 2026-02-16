"""
Sequential Training Pipeline for 4-Model Market Decomposition

I train 4 specialized MaskablePPO models in top-down order:
1. DAM (Day-Ahead Market) - standalone, uses price forecasts
2. aFRR (automatic FRR) - uses frozen DAM model
3. IntraDay (XBID) - uses frozen DAM + aFRR models
4. mFRR (manual FRR) - uses frozen DAM + aFRR + IntraDay models

Each downstream model uses frozen upstream models for training stability.

Usage:
    # Full pipeline
    python agent/train_decomposed.py --data data/unified_multimarket_training_15min.csv

    # Individual phases
    python agent/train_decomposed.py --phase dam --timesteps 3000000
    python agent/train_decomposed.py --phase afrr --dam-model models/decomposed/dam/best_model.zip
    python agent/train_decomposed.py --phase intraday --dam-model ... --afrr-model ...
    python agent/train_decomposed.py --phase mfrr --dam-model ... --afrr-model ... --id-model ...
"""

import os
import sys
import argparse
import json
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

from sb3_contrib import MaskablePPO

from gym_envs.battery_env_dam import BatteryEnvDAM
from gym_envs.battery_env_afrr import BatteryEnvAFRR
from gym_envs.battery_env_intraday import BatteryEnvIntraDay
from gym_envs.battery_env_mfrr import BatteryEnvMFRR
from gym_envs.shared_battery_state import UpstreamSimulator


# I define per-model MaskablePPO hyperparameters
MODEL_CONFIGS = {
    'dam': {
        'net_arch': [128, 128],
        'n_steps': 2048,
        'batch_size': 256,
        'gamma': 0.95,
        'ent_coef': 0.05,
        'learning_rate': 3e-4,
        'n_envs': 8,
        'default_timesteps': 3_000_000,
    },
    'afrr': {
        'net_arch': [128, 128],
        'n_steps': 2048,
        'batch_size': 128,
        'gamma': 0.98,
        'ent_coef': 0.03,
        'learning_rate': 1e-4,
        'n_envs': 8,
        'default_timesteps': 2_000_000,
    },
    'intraday': {
        'net_arch': [256, 256],
        'n_steps': 4096,
        'batch_size': 512,
        'gamma': 0.995,
        'ent_coef': 0.02,
        'learning_rate': 1e-4,
        'n_envs': 8,
        'default_timesteps': 5_000_000,
    },
    'mfrr': {
        'net_arch': [128, 128],
        'n_steps': 2048,
        'batch_size': 256,
        'gamma': 0.95,
        'ent_coef': 0.04,
        'learning_rate': 1e-4,
        'n_envs': 8,
        'default_timesteps': 3_000_000,
    },
}


class MaskableVecEnv(DummyVecEnv):
    """I extend DummyVecEnv to support action masking with VecNormalize."""

    def action_masks(self):
        return np.array([env.action_masks() for env in self.envs])


class DegradationCurriculumCallback(BaseCallback):
    """
    I implement curriculum learning for degradation cost.
    Starts at 0 and ramps to target over warmup steps.
    """

    def __init__(self, target_degradation: float = 15.0,
                 warmup_steps: int = 500_000, verbose: int = 0):
        super().__init__(verbose)
        self.target_degradation = target_degradation
        self.warmup_steps = warmup_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        current_deg = progress * self.target_degradation

        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'reward_calculator'):
                    env.reward_calculator.degradation_cost = current_deg
        return True


class DecomposedLoggingCallback(BaseCallback):
    """I log market-specific metrics to TensorBoard."""

    def __init__(self, market_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.market_name = market_name
        self.episode_rewards = []
        self.episode_socs = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'soc' in info:
                self.episode_socs.append(info['soc'])

        if self.num_timesteps % 10000 == 0:
            if len(self.episode_socs) > 0:
                self.logger.record(f'{self.market_name}/avg_soc',
                                   np.mean(self.episode_socs[-1000:]))
                self.logger.record(f'{self.market_name}/min_soc',
                                   np.min(self.episode_socs[-1000:]))
                self.logger.record(f'{self.market_name}/max_soc',
                                   np.max(self.episode_socs[-1000:]))
        return True


def load_data(data_path: str) -> tuple:
    """I load and temporally split data into train/eval sets."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  Loaded data: {len(df)} rows, columns: {list(df.columns[:10])}...")

    # I use temporal 80/20 split (NO random mixing to avoid leakage)
    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx].copy()
    df_eval = df.iloc[split_idx:].copy()
    print(f"  Train: {len(df_train)} rows, Eval: {len(df_eval)} rows")

    return df_train, df_eval


def train_dam(
    data_path: str,
    output_dir: str = "models/decomposed/dam",
    timesteps: Optional[int] = None,
    seed: int = 42,
) -> str:
    """I train the DAM model (Phase 1 - standalone)."""
    print("\n" + "=" * 60)
    print("PHASE 1: Training DAM Model")
    print("=" * 60)

    config = MODEL_CONFIGS['dam']
    timesteps = timesteps or config['default_timesteps']
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_eval = load_data(data_path)

    # I create parallel training environments
    def make_train_env(env_id: int):
        def _init():
            env = BatteryEnvDAM(df=df_train, random_start=True)
            env.reset(seed=seed + env_id)
            return Monitor(env)
        return _init

    train_env = MaskableVecEnv([make_train_env(i) for i in range(config['n_envs'])])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    # I create evaluation environment
    def make_eval_env():
        def _init():
            env = BatteryEnvDAM(df=df_eval, random_start=False)
            return Monitor(env)
        return _init

    eval_env = MaskableVecEnv([make_eval_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    # I create callbacks
    callbacks = CallbackList([
        EvalCallback(eval_env, eval_freq=50_000, n_eval_episodes=5,
                     best_model_save_path=str(model_dir),
                     log_path=str(model_dir / "eval_logs")),
        CheckpointCallback(save_freq=100_000,
                           save_path=str(model_dir / "checkpoints")),
        DegradationCurriculumCallback(target_degradation=15.0,
                                      warmup_steps=min(500_000, timesteps // 3)),
        DecomposedLoggingCallback(market_name='dam'),
    ])

    # I create MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        gae_lambda=0.95,
        ent_coef=config['ent_coef'],
        n_epochs=10,
        seed=seed,
        tensorboard_log=str(model_dir / "tensorboard"),
        policy_kwargs={"net_arch": dict(
            pi=config['net_arch'], vf=config['net_arch']
        )},
        verbose=1,
    )

    print(f"  Training DAM for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    # I save final model + vecnorm
    final_path = str(model_dir / "final_model")
    model.save(final_path)
    train_env.save(str(model_dir / "vec_normalize.pkl"))

    # I save training config
    with open(model_dir / "training_config.json", 'w') as f:
        json.dump({
            'market': 'dam',
            'timesteps': timesteps,
            'config': config,
            'data_path': data_path,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"  DAM model saved to: {final_path}")
    return final_path


def train_afrr(
    data_path: str,
    dam_model_path: Optional[str] = None,
    output_dir: str = "models/decomposed/afrr",
    timesteps: Optional[int] = None,
    seed: int = 42,
) -> str:
    """I train the aFRR model (Phase 2 - uses frozen DAM)."""
    print("\n" + "=" * 60)
    print("PHASE 2: Training aFRR Model")
    print("=" * 60)

    config = MODEL_CONFIGS['afrr']
    timesteps = timesteps or config['default_timesteps']
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_eval = load_data(data_path)

    # I create upstream simulator with frozen DAM model
    upstream = UpstreamSimulator(
        dam_model_path=dam_model_path,
        use_data_fallback=True,
    )

    def make_train_env(env_id: int):
        def _init():
            env = BatteryEnvAFRR(
                df=df_train, random_start=True,
                upstream_simulator=upstream,
            )
            env.reset(seed=seed + env_id)
            return Monitor(env)
        return _init

    train_env = MaskableVecEnv([make_train_env(i) for i in range(config['n_envs'])])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    def make_eval_env():
        def _init():
            env = BatteryEnvAFRR(
                df=df_eval, random_start=False,
                upstream_simulator=upstream,
            )
            return Monitor(env)
        return _init

    eval_env = MaskableVecEnv([make_eval_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    callbacks = CallbackList([
        EvalCallback(eval_env, eval_freq=50_000, n_eval_episodes=5,
                     best_model_save_path=str(model_dir),
                     log_path=str(model_dir / "eval_logs")),
        CheckpointCallback(save_freq=100_000,
                           save_path=str(model_dir / "checkpoints")),
        DegradationCurriculumCallback(target_degradation=15.0,
                                      warmup_steps=min(500_000, timesteps // 3)),
        DecomposedLoggingCallback(market_name='afrr'),
    ])

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        gae_lambda=0.95,
        ent_coef=config['ent_coef'],
        n_epochs=10,
        seed=seed,
        tensorboard_log=str(model_dir / "tensorboard"),
        policy_kwargs={"net_arch": dict(
            pi=config['net_arch'], vf=config['net_arch']
        )},
        verbose=1,
    )

    print(f"  Training aFRR for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    final_path = str(model_dir / "final_model")
    model.save(final_path)
    train_env.save(str(model_dir / "vec_normalize.pkl"))

    with open(model_dir / "training_config.json", 'w') as f:
        json.dump({
            'market': 'afrr',
            'timesteps': timesteps,
            'config': config,
            'dam_model_path': dam_model_path,
            'data_path': data_path,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"  aFRR model saved to: {final_path}")
    return final_path


def train_intraday(
    data_path: str,
    dam_model_path: Optional[str] = None,
    afrr_model_path: Optional[str] = None,
    output_dir: str = "models/decomposed/intraday",
    timesteps: Optional[int] = None,
    seed: int = 42,
) -> str:
    """I train the IntraDay model (Phase 3 - uses frozen DAM + aFRR)."""
    print("\n" + "=" * 60)
    print("PHASE 3: Training IntraDay Model")
    print("=" * 60)

    config = MODEL_CONFIGS['intraday']
    timesteps = timesteps or config['default_timesteps']
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_eval = load_data(data_path)

    upstream = UpstreamSimulator(
        dam_model_path=dam_model_path,
        afrr_model_path=afrr_model_path,
        use_data_fallback=True,
    )

    def make_train_env(env_id: int):
        def _init():
            env = BatteryEnvIntraDay(
                df=df_train, random_start=True,
                upstream_simulator=upstream,
            )
            env.reset(seed=seed + env_id)
            return Monitor(env)
        return _init

    train_env = MaskableVecEnv([make_train_env(i) for i in range(config['n_envs'])])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    def make_eval_env():
        def _init():
            env = BatteryEnvIntraDay(
                df=df_eval, random_start=False,
                upstream_simulator=upstream,
            )
            return Monitor(env)
        return _init

    eval_env = MaskableVecEnv([make_eval_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    callbacks = CallbackList([
        EvalCallback(eval_env, eval_freq=50_000, n_eval_episodes=5,
                     best_model_save_path=str(model_dir),
                     log_path=str(model_dir / "eval_logs")),
        CheckpointCallback(save_freq=100_000,
                           save_path=str(model_dir / "checkpoints")),
        DegradationCurriculumCallback(target_degradation=15.0,
                                      warmup_steps=min(500_000, timesteps // 3)),
        DecomposedLoggingCallback(market_name='intraday'),
    ])

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        gae_lambda=0.95,
        ent_coef=config['ent_coef'],
        n_epochs=10,
        seed=seed,
        tensorboard_log=str(model_dir / "tensorboard"),
        policy_kwargs={"net_arch": dict(
            pi=config['net_arch'], vf=config['net_arch']
        )},
        verbose=1,
    )

    print(f"  Training IntraDay for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    final_path = str(model_dir / "final_model")
    model.save(final_path)
    train_env.save(str(model_dir / "vec_normalize.pkl"))

    with open(model_dir / "training_config.json", 'w') as f:
        json.dump({
            'market': 'intraday',
            'timesteps': timesteps,
            'config': config,
            'dam_model_path': dam_model_path,
            'afrr_model_path': afrr_model_path,
            'data_path': data_path,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"  IntraDay model saved to: {final_path}")
    return final_path


def train_mfrr(
    data_path: str,
    dam_model_path: Optional[str] = None,
    afrr_model_path: Optional[str] = None,
    id_model_path: Optional[str] = None,
    output_dir: str = "models/decomposed/mfrr",
    timesteps: Optional[int] = None,
    seed: int = 42,
) -> str:
    """I train the mFRR model (Phase 4 - uses frozen DAM + aFRR + IntraDay)."""
    print("\n" + "=" * 60)
    print("PHASE 4: Training mFRR Model")
    print("=" * 60)

    config = MODEL_CONFIGS['mfrr']
    timesteps = timesteps or config['default_timesteps']
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_eval = load_data(data_path)

    upstream = UpstreamSimulator(
        dam_model_path=dam_model_path,
        afrr_model_path=afrr_model_path,
        intraday_model_path=id_model_path,
        use_data_fallback=True,
    )

    def make_train_env(env_id: int):
        def _init():
            env = BatteryEnvMFRR(
                df=df_train, random_start=True,
                upstream_simulator=upstream,
            )
            env.reset(seed=seed + env_id)
            return Monitor(env)
        return _init

    train_env = MaskableVecEnv([make_train_env(i) for i in range(config['n_envs'])])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=10.0)

    def make_eval_env():
        def _init():
            env = BatteryEnvMFRR(
                df=df_eval, random_start=False,
                upstream_simulator=upstream,
            )
            return Monitor(env)
        return _init

    eval_env = MaskableVecEnv([make_eval_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    callbacks = CallbackList([
        EvalCallback(eval_env, eval_freq=50_000, n_eval_episodes=5,
                     best_model_save_path=str(model_dir),
                     log_path=str(model_dir / "eval_logs")),
        CheckpointCallback(save_freq=100_000,
                           save_path=str(model_dir / "checkpoints")),
        DegradationCurriculumCallback(target_degradation=15.0,
                                      warmup_steps=min(500_000, timesteps // 3)),
        DecomposedLoggingCallback(market_name='mfrr'),
    ])

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        gae_lambda=0.95,
        ent_coef=config['ent_coef'],
        n_epochs=10,
        seed=seed,
        tensorboard_log=str(model_dir / "tensorboard"),
        policy_kwargs={"net_arch": dict(
            pi=config['net_arch'], vf=config['net_arch']
        )},
        verbose=1,
    )

    print(f"  Training mFRR for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps, callback=callbacks)

    final_path = str(model_dir / "final_model")
    model.save(final_path)
    train_env.save(str(model_dir / "vec_normalize.pkl"))

    with open(model_dir / "training_config.json", 'w') as f:
        json.dump({
            'market': 'mfrr',
            'timesteps': timesteps,
            'config': config,
            'dam_model_path': dam_model_path,
            'afrr_model_path': afrr_model_path,
            'id_model_path': id_model_path,
            'data_path': data_path,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"  mFRR model saved to: {final_path}")
    return final_path


def train_full_pipeline(
    data_path: str = "data/unified_multimarket_training_15min.csv",
    output_base: str = "models/decomposed",
    dam_timesteps: Optional[int] = None,
    afrr_timesteps: Optional[int] = None,
    id_timesteps: Optional[int] = None,
    mfrr_timesteps: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, str]:
    """I run the full sequential training pipeline."""
    print("\n" + "#" * 60)
    print("# 4-Model Market Decomposition Training Pipeline")
    print("#" * 60)
    print(f"  Data: {data_path}")
    print(f"  Output: {output_base}")
    start_time = datetime.now()

    results = {}

    # Phase 1: DAM
    dam_path = train_dam(
        data_path=data_path,
        output_dir=f"{output_base}/dam",
        timesteps=dam_timesteps,
        seed=seed,
    )
    results['dam'] = dam_path

    # Phase 2: aFRR (uses frozen DAM)
    afrr_path = train_afrr(
        data_path=data_path,
        dam_model_path=dam_path,
        output_dir=f"{output_base}/afrr",
        timesteps=afrr_timesteps,
        seed=seed,
    )
    results['afrr'] = afrr_path

    # Phase 3: IntraDay (uses frozen DAM + aFRR)
    id_path = train_intraday(
        data_path=data_path,
        dam_model_path=dam_path,
        afrr_model_path=afrr_path,
        output_dir=f"{output_base}/intraday",
        timesteps=id_timesteps,
        seed=seed,
    )
    results['intraday'] = id_path

    # Phase 4: mFRR (uses frozen DAM + aFRR + IntraDay)
    mfrr_path = train_mfrr(
        data_path=data_path,
        dam_model_path=dam_path,
        afrr_model_path=afrr_path,
        id_model_path=id_path,
        output_dir=f"{output_base}/mfrr",
        timesteps=mfrr_timesteps,
        seed=seed,
    )
    results['mfrr'] = mfrr_path

    elapsed = datetime.now() - start_time
    print("\n" + "#" * 60)
    print(f"# Pipeline completed in {elapsed}")
    print(f"# Models saved to: {output_base}/")
    for market, path in results.items():
        print(f"#   {market}: {path}")
    print("#" * 60)

    # I save pipeline summary
    with open(f"{output_base}/pipeline_summary.json", 'w') as f:
        json.dump({
            'models': results,
            'data_path': data_path,
            'elapsed_seconds': elapsed.total_seconds(),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train decomposed 4-model market architecture'
    )
    parser.add_argument('--data', type=str,
                        default='data/unified_multimarket_training_15min.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/decomposed',
                        help='Output directory for models')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'dam', 'afrr', 'intraday', 'mfrr'],
                        help='Training phase to run')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override default timesteps')
    parser.add_argument('--dam-model', type=str, default=None,
                        help='Path to frozen DAM model')
    parser.add_argument('--afrr-model', type=str, default=None,
                        help='Path to frozen aFRR model')
    parser.add_argument('--id-model', type=str, default=None,
                        help='Path to frozen IntraDay model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.phase == 'all':
        train_full_pipeline(
            data_path=args.data,
            output_base=args.output,
            seed=args.seed,
        )
    elif args.phase == 'dam':
        train_dam(
            data_path=args.data,
            output_dir=f"{args.output}/dam",
            timesteps=args.timesteps,
            seed=args.seed,
        )
    elif args.phase == 'afrr':
        train_afrr(
            data_path=args.data,
            dam_model_path=args.dam_model,
            output_dir=f"{args.output}/afrr",
            timesteps=args.timesteps,
            seed=args.seed,
        )
    elif args.phase == 'intraday':
        train_intraday(
            data_path=args.data,
            dam_model_path=args.dam_model,
            afrr_model_path=args.afrr_model,
            output_dir=f"{args.output}/intraday",
            timesteps=args.timesteps,
            seed=args.seed,
        )
    elif args.phase == 'mfrr':
        train_mfrr(
            data_path=args.data,
            dam_model_path=args.dam_model,
            afrr_model_path=args.afrr_model,
            id_model_path=args.id_model,
            output_dir=f"{args.output}/mfrr",
            timesteps=args.timesteps,
            seed=args.seed,
        )
