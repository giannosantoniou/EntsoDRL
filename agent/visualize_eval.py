"""
Visual Evaluation — I render battery trading as a Gymnasium-style game.

I run a trained SAC model (or baseline) step-by-step and display
the results in a pygame window in real-time.

Usage:
    python agent/visualize_eval.py --baseline         # IDLE strategy
    python agent/visualize_eval.py --model path/to/model.zip
    python agent/visualize_eval.py --random            # random actions

Controls:
    SPACE  = pause/resume
    +/-    = speed up/slow down
    LEFT/RIGHT = step when paused
    R = restart
    Q/ESC = quit
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gym_envs.battery_env_unified import BatteryEnvUnified
from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC
from gym_envs.battery_env_ppo_discrete import BatteryEnvPPODiscrete
from gym_envs.pygame_renderer import BatteryRenderer


def create_env(df, time_step_hours=0.25, use_ppo=False):
    """I create an env for visual evaluation (SAC or PPO)."""
    env_kwargs = dict(
        df=df,
        episode_length=672,
        random_start=False,
        time_step_hours=time_step_hours,  # 0.25 = 15-min resolution
        enable_endogenous_dam=True,
        dam_bidder_min_spread=5.0,
        mfrr_activation_rate=0.05,
    )
    if use_ppo:
        env = BatteryEnvPPODiscrete(**env_kwargs)
        print(f"  PPO env: time_step={env._time_step}h, actions={env.action_space}")
        return env
    else:
        inner = BatteryEnvUnified(**env_kwargs, forecast_noise=False)
        penalties = {'soc_soft_margin': 60.0, 'cycle_excess': 2000.0}
        return BatteryEnvUnifiedSAC(inner_env=inner, penalties=penalties)


def main():
    parser = argparse.ArgumentParser(description='Visual Battery Trading Evaluation')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained SAC model .zip')
    parser.add_argument('--data', type=str,
                        default='data/unified_multimarket_training_v5_raw_features.csv')
    parser.add_argument('--baseline', action='store_true',
                        help='Run IDLE baseline')
    parser.add_argument('--random', action='store_true',
                        help='Run random strategy')
    parser.add_argument('--ppo', action='store_true',
                        help='Use PPO discrete environment')
    parser.add_argument('--speed', type=int, default=4,
                        help='Initial speed (steps/sec, default 4)')
    args = parser.parse_args()

    is_ppo = args.ppo

    # I load data
    data_path = project_root / args.data
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    split_idx = int(len(df) * 0.80)
    df_eval = df.iloc[split_idx:].copy()
    print(f"Eval data: {len(df_eval)} rows")

    # I load model
    model = None
    if args.model and not args.baseline and not args.random:
        model_path = Path(args.model)
        if model_path.exists():
            print(f"Loading model: {model_path}")
            if is_ppo:
                from sb3_contrib import MaskablePPO
                model = MaskablePPO.load(model_path)
            else:
                from stable_baselines3 import SAC
                model = SAC.load(model_path)
        else:
            import glob
            candidates = glob.glob(str(model_path))
            if candidates:
                if is_ppo:
                    from sb3_contrib import MaskablePPO
                    model = MaskablePPO.load(candidates[0])
                else:
                    from stable_baselines3 import SAC
                    model = SAC.load(candidates[0])
                print(f"Loaded: {candidates[0]}")
            else:
                print(f"Model not found: {args.model}")
                return

    # I determine strategy
    if args.random:
        strategy = "RANDOM"
    elif args.baseline or model is None:
        strategy = "IDLE"
    else:
        strategy = "TRAINED"
    print(f"Strategy: {strategy} ({'PPO' if is_ppo else 'SAC'})")

    # I create a SINGLE env wrapped in VecNormalize for trained models
    # The key: model sees normalized obs, but I extract raw info from inner env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if is_ppo and strategy == "TRAINED":
        from sb3_contrib.common.wrappers import ActionMasker
        def mask_fn(env): return env.action_masks()
        def make_vec():
            e = create_env(df_eval, use_ppo=True)
            return ActionMasker(e, mask_fn)
        vec_env = DummyVecEnv([make_vec])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=True)

        # I warmup VecNormalize with random actions to learn obs statistics
        print("Warming up VecNormalize (3000 random steps)...")
        obs = vec_env.reset()
        for _ in range(3000):
            a = [vec_env.envs[0].action_space.sample()]
            obs, _, done, _ = vec_env.step(a)
            if done[0]: obs = vec_env.reset()
        vec_env.training = False
        print("VecNormalize ready.")

        # I reset for the actual episode
        obs = vec_env.reset()
        use_vec = True
    else:
        raw_env = create_env(df_eval, use_ppo=is_ppo)
        obs_raw, _ = raw_env.reset(seed=42)
        use_vec = False

    renderer = BatteryRenderer()
    renderer.speed = args.speed
    rng = np.random.RandomState(42)

    print("Starting visualization... (Q to quit, SPACE to pause)")

    try:
        step = 0
        last_info = None
        done = False

        while step < 672 and not done:
            if not renderer.handle_events():
                break

            if not renderer.should_step():
                continue

            # I compute action and step
            if use_vec:
                # I get action mask from the inner (unwrapped) env
                inner_env = vec_env.envs[0]
                # ActionMasker wraps the real env, I need to get masks
                mask = inner_env.action_masks()

                if strategy == "TRAINED":
                    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                else:
                    action = [0]

                obs, reward, done_arr, infos = vec_env.step(action)
                # I get raw info from the VecNormalize wrapper (it passes through info dict)
                info = infos[0]
                terminated = done_arr[0]
                truncated = False
                if terminated:
                    obs = vec_env.reset()
            else:
                if is_ppo:
                    if strategy == "RANDOM":
                        mask = raw_env.action_masks()
                        action = rng.choice(np.where(mask)[0])
                    elif strategy == "IDLE":
                        action = 0
                    else:
                        mask = raw_env.action_masks()
                        action, _ = model.predict(obs_raw, deterministic=True, action_masks=mask)
                        action = int(action)
                    obs_raw, reward, terminated, truncated, info = raw_env.step(action)
                else:
                    if strategy == "RANDOM":
                        action = rng.uniform(-1, 1, size=4).astype(np.float32)
                    elif strategy == "IDLE":
                        action = np.array([0.0, 0.0, 0.0, 0.0])
                    else:
                        action, _ = model.predict(obs_raw, deterministic=True)
                    obs_raw, reward, terminated, truncated, info = raw_env.step(action)

            last_info = info
            step += 1

            renderer.render(info)

            if terminated or truncated:
                done = True

        if done and last_info:
            print(f"\nEpisode ended at step {step}")
            print(f"Total profit: {last_info.get('total_profit', 0):,.0f} EUR")
            print(f"Press Q to close, or SPACE + LEFT/RIGHT to replay...")

            # I let the user replay the episode before closing
            import pygame
            while True:
                if not renderer.handle_events():
                    break
                renderer.should_step()  # I handle replay navigation

    finally:
        renderer.close()
        print("Visualization closed.")


if __name__ == "__main__":
    main()
