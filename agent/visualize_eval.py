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
from gym_envs.pygame_renderer import BatteryRenderer


def create_env(df, time_step_hours=0.25):
    """I create a SAC env for visual evaluation."""
    inner = BatteryEnvUnified(
        df=df,
        episode_length=672,
        random_start=False,
        forecast_noise=False,
        time_step_hours=time_step_hours,
        enable_endogenous_dam=True,
        dam_bidder_min_spread=5.0,
        mfrr_activation_rate=0.05,
    )
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
    parser.add_argument('--speed', type=int, default=4,
                        help='Initial speed (steps/sec, default 4)')
    args = parser.parse_args()

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
        from stable_baselines3 import SAC
        model_path = Path(args.model)
        if model_path.exists():
            print(f"Loading model: {model_path}")
            model = SAC.load(model_path)
        else:
            import glob
            candidates = glob.glob(str(model_path))
            if candidates:
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
    print(f"Strategy: {strategy}")

    # I create env and renderer
    env = create_env(df_eval)
    renderer = BatteryRenderer()
    renderer.speed = args.speed

    rng = np.random.RandomState(42)
    obs, _ = env.reset(seed=42)

    print("Starting visualization... (Q to quit, SPACE to pause)")

    try:
        step = 0
        last_info = None
        done = False

        while step < 672 and not done:
            # I handle pygame events (returns False on quit)
            if not renderer.handle_events():
                break

            # I check if we should step (handles pause, replay, speed)
            # should_step() already re-draws the frame when paused
            if not renderer.should_step():
                continue

            # I compute action
            if strategy == "RANDOM":
                action = rng.uniform(-1, 1, size=4).astype(np.float32)
            elif strategy == "IDLE":
                action = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                action, _ = model.predict(obs, deterministic=True)

            # I step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            step += 1

            # I render the new frame (updates history + draws)
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
