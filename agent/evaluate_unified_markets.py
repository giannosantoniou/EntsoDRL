"""
Quick Evaluation of Unified Model Market Participation

I evaluate the current unified model to see participation statistics across all markets:
- DAM (Day-Ahead Market)
- aFRR Capacity bidding
- aFRR Energy activation
- mFRR Energy trading
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# I add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from gym_envs.battery_env_unified import BatteryEnvUnified


def find_latest_model():
    """I find the latest unified model checkpoint."""
    models_dir = project_root / "models" / "unified"

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return None, None

    # I find all unified model directories
    model_dirs = sorted(models_dir.glob("unified_*"), reverse=True)

    for model_dir in model_dirs:
        # I look for best/best_model.zip (current structure)
        best_model = model_dir / "best" / "best_model.zip"
        if best_model.exists():
            vec_norm = model_dir / "best" / "vec_normalize.pkl"
            print(f"Found best model: {best_model}")
            return best_model, vec_norm if vec_norm.exists() else None

        # I look for best_model/best_model.zip (alternative structure)
        best_model = model_dir / "best_model" / "best_model.zip"
        if best_model.exists():
            vec_norm = model_dir / "best_model" / "vec_normalize.pkl"
            print(f"Found best model: {best_model}")
            return best_model, vec_norm if vec_norm.exists() else None

        # I check for checkpoints
        checkpoints = sorted(model_dir.glob("checkpoints/model_*.zip"), reverse=True)
        if checkpoints:
            vec_norm = checkpoints[0].parent / "vec_normalize.pkl"
            print(f"Found checkpoint: {checkpoints[0]}")
            return checkpoints[0], vec_norm if vec_norm.exists() else None

        # I check for model.zip
        model_file = model_dir / "model.zip"
        if model_file.exists():
            vec_norm = model_dir / "vec_normalize.pkl"
            print(f"Found model: {model_file}")
            return model_file, vec_norm if vec_norm.exists() else None

    return None, None


def evaluate_market_participation(model_path: Path, vec_norm_path: Path = None, n_episodes: int = 20):
    """I evaluate market participation across all markets."""

    # I load the data
    data_path = project_root / "data" / "unified_multimarket_training.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded data: {len(df)} rows")

    # I create the environment
    env = BatteryEnvUnified(
        df=df,
        episode_length=336,  # 2 weeks
        random_start=True
    )

    # I wrap the environment
    def make_env():
        return env

    vec_env = DummyVecEnv([make_env])

    # I load normalization if available
    if vec_norm_path and vec_norm_path.exists():
        print(f"Loading VecNormalize from {vec_norm_path}")
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # I load the model
    print(f"Loading model from {model_path}")
    model = MaskablePPO.load(str(model_path), env=vec_env)

    # I run evaluation episodes
    print(f"\nRunning {n_episodes} evaluation episodes...")

    # I track statistics
    stats = {
        'total_profit': [],
        'dam_profit': [],
        'afrr_capacity_profit': [],
        'afrr_energy_profit': [],
        'mfrr_profit': [],
        'total_cycles': [],

        # I track participation counts
        'afrr_bids': [],  # Number of steps with aFRR > 0
        'afrr_selections': [],  # Number of times selected for aFRR
        'afrr_activations': [],  # Number of times aFRR activated
        'mfrr_trades': [],  # Number of mFRR trades
        'energy_actions': [],  # Non-zero energy actions

        # I track action distribution
        'afrr_level_counts': [0, 0, 0, 0, 0],  # 5 levels
        'price_tier_counts': [0, 0, 0, 0, 0],  # 5 tiers
        'energy_action_hist': np.zeros(21),  # 21 energy actions
    }

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False

        ep_afrr_bids = 0
        ep_afrr_selections = 0
        ep_afrr_activations = 0
        ep_mfrr_trades = 0
        ep_energy_actions = 0

        while not done:
            # I get action masks
            action_masks = vec_env.env_method("action_masks")[0]

            # I get action from model
            action, _ = model.predict(obs, deterministic=False, action_masks=action_masks)

            # I track action distribution
            afrr_level, price_tier, energy_action = action[0]
            stats['afrr_level_counts'][afrr_level] += 1
            stats['price_tier_counts'][price_tier] += 1
            stats['energy_action_hist'][energy_action] += 1

            # I step the environment
            obs, reward, done, info = vec_env.step(action)
            info = info[0]

            # I track participation
            if info.get('afrr_commitment', 0) > 0.1:
                ep_afrr_bids += 1
            if info.get('is_selected', False):
                ep_afrr_selections += 1
            if info.get('afrr_activated', False):
                ep_afrr_activations += 1
            if abs(info.get('mfrr_energy_mw', 0)) > 0.1:
                ep_mfrr_trades += 1
            if abs(info.get('actual_energy_mw', 0)) > 0.1:
                ep_energy_actions += 1

        # I get final episode stats
        final_info = info
        stats['total_profit'].append(final_info.get('total_profit', 0))
        stats['dam_profit'].append(final_info.get('dam_profit', 0))
        stats['afrr_capacity_profit'].append(final_info.get('afrr_capacity_profit', 0))
        stats['afrr_energy_profit'].append(final_info.get('afrr_energy_profit', 0))
        stats['mfrr_profit'].append(final_info.get('mfrr_profit', 0))
        stats['total_cycles'].append(final_info.get('total_cycles', 0))

        stats['afrr_bids'].append(ep_afrr_bids)
        stats['afrr_selections'].append(ep_afrr_selections)
        stats['afrr_activations'].append(ep_afrr_activations)
        stats['mfrr_trades'].append(ep_mfrr_trades)
        stats['energy_actions'].append(ep_energy_actions)

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} complete")

    vec_env.close()

    return stats


def print_statistics(stats: dict):
    """I print comprehensive market participation statistics."""

    print("\n" + "=" * 70)
    print("UNIFIED MODEL MARKET PARTICIPATION STATISTICS")
    print("=" * 70)

    n_eps = len(stats['total_profit'])
    ep_length = 336  # Approximate

    # I calculate averages
    print("\n[PROFIT] REVENUE BY MARKET (per episode)")
    print("-" * 50)
    print(f"  Total Profit:        {np.mean(stats['total_profit']):>12,.0f} EUR")
    print(f"  DAM Profit:          {np.mean(stats['dam_profit']):>12,.0f} EUR")
    print(f"  aFRR Capacity:       {np.mean(stats['afrr_capacity_profit']):>12,.0f} EUR")
    print(f"  aFRR Energy:         {np.mean(stats['afrr_energy_profit']):>12,.0f} EUR")
    print(f"  mFRR Trading:        {np.mean(stats['mfrr_profit']):>12,.0f} EUR")
    print(f"  Cycles:              {np.mean(stats['total_cycles']):>12.2f}")

    # I calculate participation rates
    print("\n[RATES] MARKET PARTICIPATION RATES")
    print("-" * 50)
    avg_afrr_bids = np.mean(stats['afrr_bids'])
    avg_afrr_selections = np.mean(stats['afrr_selections'])
    avg_afrr_activations = np.mean(stats['afrr_activations'])
    avg_mfrr_trades = np.mean(stats['mfrr_trades'])
    avg_energy_actions = np.mean(stats['energy_actions'])

    print(f"  aFRR Bids:           {avg_afrr_bids:>8.1f} / {ep_length} steps ({100*avg_afrr_bids/ep_length:.1f}%)")
    print(f"  aFRR Selections:     {avg_afrr_selections:>8.1f} / {ep_length} steps ({100*avg_afrr_selections/ep_length:.1f}%)")
    print(f"  aFRR Activations:    {avg_afrr_activations:>8.1f} / {ep_length} steps ({100*avg_afrr_activations/ep_length:.1f}%)")
    print(f"  mFRR Trades:         {avg_mfrr_trades:>8.1f} / {ep_length} steps ({100*avg_mfrr_trades/ep_length:.1f}%)")
    print(f"  Energy Actions:      {avg_energy_actions:>8.1f} / {ep_length} steps ({100*avg_energy_actions/ep_length:.1f}%)")

    if avg_afrr_bids > 0:
        selection_rate = avg_afrr_selections / avg_afrr_bids * 100
        print(f"\n  aFRR Selection Rate: {selection_rate:.1f}% of bids")

    if avg_afrr_selections > 0:
        activation_rate = avg_afrr_activations / avg_afrr_selections * 100
        print(f"  aFRR Activation Rate: {activation_rate:.1f}% of selections")

    # I show action distribution
    print("\n[ACTIONS] ACTION DISTRIBUTION")
    print("-" * 50)

    afrr_levels = stats['afrr_level_counts']
    total_actions = sum(afrr_levels)
    print("\n  aFRR Commitment Levels:")
    level_names = ['0%', '25%', '50%', '75%', '100%']
    for i, (name, count) in enumerate(zip(level_names, afrr_levels)):
        pct = 100 * count / total_actions if total_actions > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {name:>5}: {count:>6} ({pct:>5.1f}%) {bar}")

    price_tiers = stats['price_tier_counts']
    print("\n  aFRR Price Tiers:")
    tier_names = ['Very Aggressive', 'Aggressive', 'Neutral', 'Conservative', 'Very Conservative']
    for i, (name, count) in enumerate(zip(tier_names, price_tiers)):
        pct = 100 * count / total_actions if total_actions > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {name:>18}: {count:>6} ({pct:>5.1f}%) {bar}")

    # I show energy action distribution summary
    energy_hist = stats['energy_action_hist']
    print("\n  Energy Actions (-30 to +30 MW):")
    charge_actions = sum(energy_hist[:10])  # -30 to -3 MW (charging)
    idle_action = energy_hist[10]  # 0 MW
    discharge_actions = sum(energy_hist[11:])  # +3 to +30 MW (discharging)

    print(f"    Charging (-30 to -3 MW):   {charge_actions:>6} ({100*charge_actions/total_actions:.1f}%)")
    print(f"    Idle (0 MW):               {idle_action:>6} ({100*idle_action/total_actions:.1f}%)")
    print(f"    Discharging (+3 to +30 MW):{discharge_actions:>6} ({100*discharge_actions/total_actions:.1f}%)")

    # I calculate revenue breakdown
    print("\n[REVENUE] REVENUE BREAKDOWN (% of total)")
    print("-" * 50)
    total_rev = np.mean(stats['total_profit'])
    if total_rev != 0:
        dam_pct = 100 * np.mean(stats['dam_profit']) / abs(total_rev)
        afrr_cap_pct = 100 * np.mean(stats['afrr_capacity_profit']) / abs(total_rev)
        afrr_energy_pct = 100 * np.mean(stats['afrr_energy_profit']) / abs(total_rev)
        mfrr_pct = 100 * np.mean(stats['mfrr_profit']) / abs(total_rev)

        print(f"  DAM:           {dam_pct:>8.1f}%")
        print(f"  aFRR Capacity: {afrr_cap_pct:>8.1f}%")
        print(f"  aFRR Energy:   {afrr_energy_pct:>8.1f}%")
        print(f"  mFRR:          {mfrr_pct:>8.1f}%")

    # I show summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    markets_used = []
    if np.mean(stats['afrr_capacity_profit']) != 0:
        markets_used.append("aFRR Capacity")
    if np.mean(stats['afrr_energy_profit']) != 0:
        markets_used.append("aFRR Energy")
    if np.mean(stats['mfrr_profit']) != 0:
        markets_used.append("mFRR")
    if np.mean(stats['dam_profit']) != 0:
        markets_used.append("DAM")

    if markets_used:
        print(f"[OK] Model participates in: {', '.join(markets_used)}")
    else:
        print("[WARN]  Model not actively trading in any market")

    if avg_afrr_bids / ep_length > 0.5:
        print("[OK] Strong aFRR participation (>50% of steps)")
    elif avg_afrr_bids / ep_length > 0.2:
        print("[WARN]  Moderate aFRR participation (20-50%)")
    else:
        print("[X] Low aFRR participation (<20%)")

    print()


def main():
    """I run the evaluation."""
    print("=" * 70)
    print("UNIFIED MODEL MARKET PARTICIPATION EVALUATION")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # I find the latest model
    model_path, vec_norm_path = find_latest_model()

    if model_path is None:
        print("\n[X] No unified model found. Training may still be in progress.")
        print("   The first checkpoint is saved at 50,000 timesteps.")
        return

    # I run evaluation
    stats = evaluate_market_participation(model_path, vec_norm_path, n_episodes=20)

    # I print results
    print_statistics(stats)


if __name__ == "__main__":
    main()
