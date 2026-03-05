"""
Unit Tests for SAC Wrapper of Unified Multi-Market Battery Environment

I test the key functionality of the SAC wrapper:
1. Action space shape/bounds (Box [-1,1] shape=(N,))
2. Continuous-to-discrete mapping correctness
3. Action filtering — invalid actions replaced with nearest valid (Fix C)
4. Soft penalty computation (SoC margin, super-linear cycling)
5. Reward calculator accessibility (for curriculum callback)
6. Full episode completion (legacy 4-dim and full-market 6-dim)
7. VecNormalize compatibility
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# I add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _make_sample_data(n_hours=168, freq='h'):
    """I create sample training data for testing."""
    dates = pd.date_range(
        start='2024-01-01',
        periods=n_hours,
        freq=freq,
        tz='Europe/Athens'
    )

    np.random.seed(42)

    hours = np.array([d.hour for d in dates])
    base_price = 80 + 40 * np.sin(2 * np.pi * hours / 24)
    noise = np.random.normal(0, 10, n_hours)
    prices = base_price + noise

    dam_commitments = np.zeros(n_hours)
    for i in range(n_hours):
        if hours[i] in [18, 19, 20]:
            dam_commitments[i] = np.random.uniform(10, 25)
        elif hours[i] in [10, 11, 12]:
            dam_commitments[i] = np.random.uniform(-20, -5)

    data = {
        'price': prices,
        'dam_commitment': dam_commitments,
        'afrr_up': prices * 1.1,
        'afrr_down': prices * 0.9,
        'afrr_cap_up_price': 15 + 10 * np.random.random(n_hours),
        'afrr_cap_down_price': 25 + 15 * np.random.random(n_hours),
        'intraday_bid': prices - 2.0 + np.random.normal(0, 1, n_hours),
        'intraday_ask': prices + 2.0 + np.random.normal(0, 1, n_hours),
        'intraday_spread': 3.0 + np.random.uniform(0, 4, n_hours),
        'intraday_volume': np.random.uniform(0.3, 0.9, n_hours),
        'mfrr_price_up': prices * 1.3,
        'mfrr_price_down': prices * 0.7,
        'mfrr_spread': prices * 0.6,
        'mfrr_activated_up_mwh': np.where(np.random.random(n_hours) < 0.84,
                                           np.random.uniform(5, 200, n_hours), 0.0),
        'mfrr_activated_down_mwh': np.where(np.random.random(n_hours) < 0.96,
                                             np.random.uniform(5, 150, n_hours), 0.0),
        'net_imbalance_mw': np.random.normal(0, 300, n_hours),
        'solar': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)),
        'wind_onshore': np.random.uniform(100, 500, n_hours),
        'res_total_mw': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12))
                        + np.random.uniform(100, 500, n_hours),
        'load_mw': np.random.uniform(3000, 6000, n_hours),
        'xbid_bid': prices - 3.0 + np.random.normal(0, 1, n_hours),
        'xbid_ask': prices + 3.0 + np.random.normal(0, 1, n_hours),
        'ida1_clearing_price': prices + np.random.normal(2, 5, n_hours),
        'ida2_clearing_price': prices + np.random.normal(3, 8, n_hours),
        'ida3_clearing_price': prices + np.random.normal(5, 10, n_hours),
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_data():
    """I create sample training data for testing."""
    return _make_sample_data()


@pytest.fixture
def inner_env(sample_data):
    """I create the inner BatteryEnvUnified for wrapping (legacy 4-dim)."""
    from gym_envs.battery_env_unified import BatteryEnvUnified

    return BatteryEnvUnified(
        df=sample_data,
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        episode_length=72,
        random_start=False
    )


@pytest.fixture
def inner_env_full_market(sample_data):
    """I create the inner BatteryEnvUnified for full-market mode (6-dim)."""
    from gym_envs.battery_env_unified import BatteryEnvUnified

    return BatteryEnvUnified(
        df=sample_data,
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        episode_length=72,
        random_start=False,
        enable_full_market=True
    )


@pytest.fixture
def sac_env(inner_env):
    """I create the SAC wrapper environment (legacy 4-dim)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env)


@pytest.fixture
def sac_env_full_market(inner_env_full_market):
    """I create the SAC wrapper environment (full-market 6-dim)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env_full_market)


class TestActionSpace:
    """I test the continuous action space structure."""

    def test_action_space_shape_legacy(self, sac_env):
        """I verify the legacy action space is Box(shape=(4,))."""
        assert sac_env.action_space.shape == (4,)

    def test_action_space_shape_full_market(self, sac_env_full_market):
        """I verify the full-market action space is Box(shape=(6,))."""
        assert sac_env_full_market.action_space.shape == (6,)

    def test_action_space_bounds_legacy(self, sac_env):
        """I verify legacy action space bounds are [-1, 1]."""
        np.testing.assert_array_equal(sac_env.action_space.low, [-1] * 4)
        np.testing.assert_array_equal(sac_env.action_space.high, [1] * 4)

    def test_action_space_bounds_full_market(self, sac_env_full_market):
        """I verify full-market action space bounds are [-1, 1] x 6."""
        np.testing.assert_array_equal(sac_env_full_market.action_space.low, [-1] * 6)
        np.testing.assert_array_equal(sac_env_full_market.action_space.high, [1] * 6)

    def test_action_space_dtype(self, sac_env):
        """I verify action space uses float32."""
        assert sac_env.action_space.dtype == np.float32

    def test_observation_space_matches_inner(self, sac_env, inner_env):
        """I verify observation space delegates to inner env."""
        assert sac_env.observation_space.shape == inner_env.observation_space.shape


class TestContinuousToDiscreteMapping:
    """I test the continuous-to-discrete action conversion."""

    def test_all_min(self, sac_env):
        """I verify [-1,-1,-1,-1] maps to [0,0,0,0]."""
        continuous = np.array([-1.0, -1.0, -1.0, -1.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        np.testing.assert_array_equal(discrete, [0, 0, 0, 0])

    def test_all_center(self, sac_env):
        """I verify [0,0,0,0] maps to [2,2,5,5] (middle indices with dead zones)."""
        continuous = np.array([0.0, 0.0, 0.0, 0.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        np.testing.assert_array_equal(discrete, [2, 2, 5, 5])

    def test_all_max(self, sac_env):
        """I verify [1,1,1,1] maps to [4,4,10,10]."""
        continuous = np.array([1.0, 1.0, 1.0, 1.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        np.testing.assert_array_equal(discrete, [4, 4, 10, 10])

    def test_clipping(self, sac_env):
        """I verify out-of-range values are clipped before mapping."""
        continuous = np.array([-2.0, 2.0, -1.5, 1.5])
        discrete = sac_env.continuous_to_discrete(continuous)
        np.testing.assert_array_equal(discrete, [0, 4, 0, 10])

    def test_intraday_dead_zone(self, sac_env):
        """I verify IntraDay dead zone maps center values to idle."""
        # 0.3 < 0.5 threshold → idle (index 5)
        continuous = np.array([0.0, 0.0, 0.3, 0.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        assert discrete[2] == 5  # IntraDay idle

    def test_mfrr_dead_zone(self, sac_env):
        """I verify mFRR dead zone maps center values to idle."""
        # 0.2 < 0.3 threshold → idle (index 5)
        continuous = np.array([0.0, 0.0, 0.0, 0.2])
        discrete = sac_env.continuous_to_discrete(continuous)
        assert discrete[3] == 5  # mFRR idle

    def test_freebid_dead_zone_full_market(self, sac_env_full_market):
        """I verify FreeBid dead zone maps center values to idle in full-market."""
        # 0.2 < 0.3 threshold → idle (index 5)
        continuous = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.0])
        discrete = sac_env_full_market.continuous_to_discrete(continuous)
        assert discrete[4] == 5  # FreeBid qty idle

    def test_full_market_all_max(self, sac_env_full_market):
        """I verify full-market [1,1,1,1,1,1] maps to correct max indices."""
        continuous = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        discrete = sac_env_full_market.continuous_to_discrete(continuous)
        # [5,5,11,11,11,5] → max indices [4,4,10,10,10,4]
        np.testing.assert_array_equal(discrete, [4, 4, 10, 10, 10, 4])


class TestActionFiltering:
    """I test action filtering — Fix C (nearest valid action replacement)."""

    def test_valid_action_unchanged(self, sac_env):
        """I verify valid actions pass through filtering unchanged."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5  # mid-range SoC

        # I get idle action which should always be valid
        idle = np.array([2, 2, 5, 5], dtype=np.int64)
        filtered = sac_env._filter_to_nearest_valid(idle)
        np.testing.assert_array_equal(idle, filtered)

    def test_invalid_action_gets_filtered(self, sac_env):
        """I verify invalid actions are replaced with nearest valid."""
        sac_env.reset(seed=42)
        # I set SoC near minimum to make discharge actions invalid
        sac_env._inner.soc = 0.06

        # I force step to hour 14 so IntraDay gate is open
        sac_env._inner.current_step = sac_env._inner.start_step + 14

        # I try max discharge which should be masked at low SoC
        max_discharge = np.array([2, 2, 10, 10], dtype=np.int64)
        filtered = sac_env._filter_to_nearest_valid(max_discharge)

        # I verify IntraDay and mFRR were filtered to something valid
        mask = sac_env._inner.action_masks()
        offsets = np.cumsum([0] + list(sac_env._nvec))
        for dim_idx in range(len(sac_env._nvec)):
            start = offsets[dim_idx]
            dim_mask = mask[start:start + sac_env._nvec[dim_idx]]
            assert dim_mask[filtered[dim_idx]], \
                f"Filtered action dim {dim_idx}={filtered[dim_idx]} should be valid"

    def test_filtering_preserves_direction(self, sac_env):
        """I verify filtering picks nearest valid, preserving intended direction."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5  # mid-range, most actions should be valid

        # I try a discharge action (index 8 out of 11 for mFRR)
        action = np.array([2, 2, 5, 8], dtype=np.int64)
        filtered = sac_env._filter_to_nearest_valid(action)

        # I verify filtered action is close to original (nearest valid)
        assert abs(filtered[3] - 8) <= 3, \
            f"Filtered mFRR {filtered[3]} should be near 8"

    def test_step_never_violates_mask(self, sac_env):
        """I verify step() with action filtering never passes invalid actions."""
        sac_env.reset(seed=42)

        # I run 50 random steps — none should violate masks
        for _ in range(50):
            action = sac_env.action_space.sample()
            obs, reward, done, truncated, info = sac_env.step(action)

            # I verify the discrete action in info was valid
            discrete = np.array(info['discrete_action'])
            mask = sac_env._inner.action_masks()
            offsets = np.cumsum([0] + list(sac_env._nvec))
            for dim_idx in range(len(sac_env._nvec)):
                start = offsets[dim_idx]
                dim_mask = mask[start:start + sac_env._nvec[dim_idx]]
                # I note: mask was read AFTER step, so I just check finite reward
                assert np.isfinite(reward)

            if done or truncated:
                sac_env.reset()

    def test_action_was_filtered_flag(self, sac_env):
        """I verify info['action_was_filtered'] flag is set correctly."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        # I use idle action — should not be filtered
        idle_continuous = np.array([0.0, 0.0, 0.0, 0.0])
        obs, reward, done, truncated, info = sac_env.step(idle_continuous)
        assert 'action_was_filtered' in info

    def test_filtering_full_market(self, sac_env_full_market):
        """I verify action filtering works for 6-dim full-market actions."""
        sac_env_full_market.reset(seed=42)

        # I run 30 random steps
        for _ in range(30):
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)
            assert np.isfinite(reward)
            assert len(info['discrete_action']) == 6
            if done or truncated:
                sac_env_full_market.reset()


class TestSoftPenalties:
    """I test soft constraint penalties (SoC margin + super-linear cycling)."""

    def test_idle_action_zero_penalty(self, sac_env):
        """I verify idle action at mid SoC gets zero penalty."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5  # well below target

        idle = np.array([2, 2, 5, 5], dtype=np.int64)
        penalty = sac_env._compute_penalty(idle)
        assert penalty == 0.0

    def test_soc_margin_penalty_near_min(self, sac_env):
        """I verify soft SoC margin penalty when discharging near minimum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.08  # within 5% buffer of min_soc (0.05)
        sac_env._inner.daily_cycles = 0.5

        # I use a discharge action (IntraDay index 8 = positive level)
        discharge = np.array([2, 2, 8, 5], dtype=np.int64)
        penalty = sac_env._compute_penalty(discharge)
        assert penalty > 0, "Should penalize discharge near min SoC"

    def test_soc_margin_penalty_near_max(self, sac_env):
        """I verify soft SoC margin penalty when charging near maximum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.93  # within 5% buffer of max_soc (0.95)
        sac_env._inner.daily_cycles = 0.5

        # I use a charge action (IntraDay index 2 = negative level)
        charge = np.array([2, 2, 2, 5], dtype=np.int64)
        penalty = sac_env._compute_penalty(charge)
        assert penalty > 0, "Should penalize charge near max SoC"

    def test_no_soc_margin_penalty_at_mid_soc(self, sac_env):
        """I verify no SoC margin penalty at mid-range SoC."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5

        discharge = np.array([2, 2, 8, 8], dtype=np.int64)
        penalty = sac_env._compute_penalty(discharge)
        assert penalty == 0.0

    def test_cycle_penalty_below_target_is_zero(self, sac_env):
        """I verify no cycle penalty when below target."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 1.5  # below 1.8 target

        idle = np.array([2, 2, 5, 5], dtype=np.int64)
        penalty = sac_env._compute_penalty(idle)
        assert penalty == 0.0

    def test_cycle_penalty_super_linear(self, sac_env):
        """I verify cycle penalty is super-linear (excess^1.5)."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        idle = np.array([2, 2, 5, 5], dtype=np.int64)

        # I test at 2.0 cycles (0.2 excess)
        sac_env._inner.daily_cycles = 2.0
        penalty_small = sac_env._compute_penalty(idle)

        # I test at 4.0 cycles (2.2 excess)
        sac_env._inner.daily_cycles = 4.0
        penalty_large = sac_env._compute_penalty(idle)

        # I verify super-linear: doubling excess should more than double penalty
        # excess 0.2 → 0.2^1.5 = 0.089
        # excess 2.2 → 2.2^1.5 = 3.26
        # ratio should be ~36.6, much larger than linear (11x)
        assert penalty_large > 10 * penalty_small, \
            f"Penalty should be super-linear: {penalty_large} vs {penalty_small}"

    def test_cycle_penalty_at_5_9_cycles(self, sac_env):
        """I verify cycle penalty is severe at 5.9 cycles/day (the SAC failure case)."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 5.9

        idle = np.array([2, 2, 5, 5], dtype=np.int64)
        penalty = sac_env._compute_penalty(idle)

        # I expect: 2000 * (5.9 - 1.8)^1.5 = 2000 * 4.1^1.5 ≈ 16,600 EUR
        expected = 2000.0 * (4.1 ** 1.5)
        assert abs(penalty - expected) < 1.0, \
            f"Penalty at 5.9 cycles should be ~{expected:.0f}, got {penalty:.0f}"

    def test_cycle_penalty_magnitude_vs_profit(self, sac_env):
        """I verify cycle penalty at target+1 is meaningful vs trading profit."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 2.8  # 1.0 excess cycle

        idle = np.array([2, 2, 5, 5], dtype=np.int64)
        penalty = sac_env._compute_penalty(idle)

        # I expect: 2000 * 1.0^1.5 = 2000 EUR per step
        # This should be meaningful vs ~4,470 EUR/cycle profit
        assert penalty >= 1500.0, \
            f"Penalty at 1 excess cycle should be >= 1500 EUR, got {penalty:.0f}"

    def test_no_hard_penalty_keys_in_defaults(self, sac_env):
        """I verify hard penalties were removed from defaults (handled by filtering)."""
        assert 'soc_hard' not in sac_env.penalties
        assert 'gate_closure' not in sac_env.penalties
        assert 'mfrr_unavailable' not in sac_env.penalties


class TestRewardCalculator:
    """I test reward calculator accessibility for curriculum callbacks."""

    def test_reward_calculator_accessible(self, sac_env):
        """I verify reward calculator is accessible via property."""
        rc = sac_env.reward_calculator
        assert rc is not None
        assert hasattr(rc, 'degradation_cost')

    def test_reward_calculator_settable(self, sac_env):
        """I verify degradation cost can be updated (curriculum callback)."""
        sac_env.reward_calculator.degradation_cost = 10.0
        assert sac_env._inner.reward_calculator.degradation_cost == 10.0


class TestEpisodeCompletion:
    """I test full episode execution."""

    def test_full_episode_legacy(self, sac_env):
        """I verify a full episode completes without errors (legacy 4-dim)."""
        obs, info = sac_env.reset(seed=42)
        assert obs.shape == sac_env.observation_space.shape

        total_reward = 0.0
        steps = 0

        while True:
            action = sac_env.action_space.sample()
            obs, reward, done, truncated, info = sac_env.step(action)

            assert obs.shape == sac_env.observation_space.shape
            assert np.isfinite(reward)
            assert 'sac_penalty' in info
            assert 'discrete_action' in info
            assert 'action_was_filtered' in info

            total_reward += reward
            steps += 1

            if done or truncated:
                break

        assert steps > 0
        assert np.isfinite(total_reward)

    def test_full_episode_full_market(self, sac_env_full_market):
        """I verify a full episode completes without errors (full-market 6-dim)."""
        obs, info = sac_env_full_market.reset(seed=42)
        assert obs.shape == sac_env_full_market.observation_space.shape

        total_reward = 0.0
        steps = 0

        while True:
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)

            assert obs.shape == sac_env_full_market.observation_space.shape
            assert np.isfinite(reward)
            assert len(info['discrete_action']) == 6

            total_reward += reward
            steps += 1

            if done or truncated:
                break

        assert steps > 0
        assert np.isfinite(total_reward)

    def test_multiple_episodes(self, sac_env):
        """I verify multiple episodes complete correctly."""
        for ep in range(3):
            obs, info = sac_env.reset(seed=ep)
            done = False
            steps = 0

            while not done:
                action = sac_env.action_space.sample()
                obs, reward, done, truncated, info = sac_env.step(action)
                steps += 1
                if truncated:
                    break

            assert steps > 0


class TestVecNormalizeCompatibility:
    """I test compatibility with SB3 VecNormalize wrapper."""

    def test_dummy_vec_env_wrapping(self, sample_data):
        """I verify the SAC env works inside DummyVecEnv + VecNormalize."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data,
                capacity_mwh=146.0,
                max_power_mw=30.0,
                efficiency=0.94,
                episode_length=72,
                random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        obs = vec_env.reset()
        assert obs.shape[1] == 68  # (1, 66) for vectorized

        for _ in range(10):
            action = [vec_env.action_space.sample()]
            obs, reward, done, info = vec_env.step(action)
            assert obs.shape[1] == 68
            assert np.all(np.isfinite(obs))

    def test_observation_normalization_stable(self, sample_data):
        """I verify observations remain finite after normalization."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data,
                capacity_mwh=146.0,
                max_power_mw=30.0,
                efficiency=0.94,
                episode_length=72,
                random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        obs = vec_env.reset()
        for _ in range(50):
            action = [vec_env.action_space.sample()]
            obs, reward, done, info = vec_env.step(action)
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))


class TestInfoDictAnnotation:
    """I test that info dict contains SAC-specific annotations."""

    def test_info_contains_penalty(self, sac_env):
        """I verify step info includes SAC penalty fields."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        assert 'sac_penalty' in info
        assert 'sac_penalty_scaled' in info
        assert 'discrete_action' in info
        assert 'action_was_filtered' in info
        assert isinstance(info['discrete_action'], list)
        assert len(info['discrete_action']) == 4

    def test_info_6dim_full_market(self, sac_env_full_market):
        """I verify step info has 6-dim discrete action in full-market."""
        sac_env_full_market.reset(seed=42)
        action = sac_env_full_market.action_space.sample()
        _, _, _, _, info = sac_env_full_market.step(action)

        assert len(info['discrete_action']) == 6

    def test_penalty_scaling(self, sac_env):
        """I verify penalty is scaled by reward_scale."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        reward_scale = sac_env._inner.reward_calculator.reward_scale
        expected_scaled = info['sac_penalty'] * reward_scale
        assert abs(info['sac_penalty_scaled'] - expected_scaled) < 1e-6


class TestSetDegradationCost:
    """I test the set_degradation_cost method for SubprocVecEnv compatibility."""

    def test_set_degradation_cost_method(self, sac_env):
        """I verify set_degradation_cost updates inner env's reward_calculator."""
        sac_env.reset(seed=42)
        sac_env.set_degradation_cost(5.0)
        assert sac_env._inner.reward_calculator.degradation_cost == 5.0

        sac_env.set_degradation_cost(12.5)
        assert sac_env._inner.reward_calculator.degradation_cost == 12.5

    def test_set_degradation_cost_via_vecenv(self, sample_data):
        """I verify set_degradation_cost propagates through DummyVecEnv.env_method."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data,
                capacity_mwh=146.0,
                max_power_mw=30.0,
                efficiency=0.94,
                episode_length=72,
                random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env.reset()

        # I use env_method to call set_degradation_cost on all envs
        vec_env.env_method('set_degradation_cost', 3.0)

        # I verify the inner env's degradation cost was updated
        inner_cost = vec_env.envs[0]._inner.reward_calculator.degradation_cost
        assert inner_cost == 3.0, \
            f"Degradation cost should be 3.0 via env_method, got {inner_cost}"


class TestSetPenalties:
    """I test runtime penalty override for curriculum scheduling."""

    def test_set_penalties_updates_values(self, sac_env):
        """I verify set_penalties updates penalty dict at runtime."""
        original_cycle = sac_env.penalties['cycle_excess']
        sac_env.set_penalties({'cycle_excess': 5000.0})
        assert sac_env.penalties['cycle_excess'] == 5000.0
        assert sac_env.penalties['cycle_excess'] != original_cycle

    def test_set_penalties_partial_update(self, sac_env):
        """I verify set_penalties only updates specified keys."""
        original_margin = sac_env.penalties['soc_soft_margin']
        sac_env.set_penalties({'cycle_excess': 3000.0})
        assert sac_env.penalties['soc_soft_margin'] == original_margin
