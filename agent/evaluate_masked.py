"""
Evaluation for MaskablePPO on Unseen Data - FIXED VERSION
"""
import os
import sys
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from data.dam_commitment_generator import FeasibleDAMGenerator


def get_action_mask(vec_env):
    """Traverse wrappers to get action_masks from the base environment."""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        if hasattr(env, 'action_masks'):
            return env.action_masks()
        env = env.env
    if hasattr(env, 'action_masks'):
        return env.action_masks()
    raise RuntimeError("Could not find action_masks method!")


def mask_fn(env):
    """Mask function for ActionMasker wrapper."""
    return env.action_masks()


def evaluate_masked(model_path=None, test_ratio=0.2):
    DATA_PATH = "data/feasible_data_with_dam.csv"
    
    if model_path is None:
        checkpoint_dir = "models/checkpoints/ppo_masked/"
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')],
                           key=lambda x: int(x.split('_')[-2]), reverse=True)
        model_path = os.path.join(checkpoint_dir, checkpoints[0])
    
    print("=" * 60)
    print("MaskablePPO EVALUATION ON UNSEEN DATA (FIXED)")
    print("=" * 60)
    print(f"Model: {model_path}")
    
    # Load data
    df = load_historical_data(DATA_PATH)
    split_idx = int(len(df) * (1 - test_ratio))
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Test size: {len(test_df):,} hours")
    
    # Generate DAM commitments for test
    battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}
    gen = FeasibleDAMGenerator(
        capacity_mwh=battery_params['capacity_mwh'], 
        max_power_mw=battery_params['max_discharge_mw'],
        efficiency=battery_params['efficiency']
    )
    test_df['dam_commitment'] = gen.generate(test_df)
    
    # Create env - CORRECT WRAPPER ORDER!
    # BatteryEnvMasked -> Monitor -> ActionMasker
    # ActionMasker must be OUTERMOST so it can access action_masks
    def make_env():
        env = BatteryEnvMasked(test_df, battery_params, n_actions=21)
        env = Monitor(env)  # Monitor first
        env = ActionMasker(env, lambda e: e.env.action_masks())  # ActionMasker wraps Monitor
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # Load model
    model = MaskablePPO.load(model_path, env=env)
    
    # Evaluate
    print("\nRunning evaluation with EXPLICIT action masks...")
    obs = env.reset()
    done = False
    total_reward = 0
    violations = 0
    steps = 0
    actions = {i: 0 for i in range(21)}
    
    while not done:
        # CRITICAL: Get mask and pass explicitly to predict()
        mask = get_action_mask(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=np.array([mask]))
        
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        actions[int(action[0])] += 1
        if info[0].get('violation', 0) > 0.5:
            violations += 1
        steps += 1
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total Reward:  {total_reward:>15,.2f}")
    print(f"  Violations:    {violations:>15,} ({violations/steps*100:.1f}%)")
    print(f"  Steps:         {steps:>15,}")
    
    # Action distribution
    discharge = sum(actions[i] for i in range(11, 21))
    idle = actions[10]
    charge = sum(actions[i] for i in range(0, 10))
    print(f"\n  Actions:")
    print(f"    Discharge:   {discharge/steps*100:>6.1f}%")
    print(f"    Idle:        {idle/steps*100:>6.1f}%")
    print(f"    Charge:      {charge/steps*100:>6.1f}%")
    print("=" * 60)
    
    if violations == 0:
        print("✅ ZERO VIOLATIONS! Action masking works perfectly!")
    elif violations / steps < 0.05:
        print("✅ Less than 5% violations - excellent!")
    
    return total_reward, violations


if __name__ == "__main__":
    evaluate_masked()
