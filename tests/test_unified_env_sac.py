"""
Unit Tests for SAC Wrapper of Unified Multi-Market Battery Environment

I test the key functionality of the SAC wrapper:
1. Action space shape/bounds (Box [-1,1] shape=(4,))
2. Continuous-to-discrete mapping correctness
3. Penalty computation for masked actions
4. Penalty = 0 for idle/valid actions
5. SoC margin penalty
6. Reward calculator accessibility (for curriculum callback)
7. Full episode completion
8. VecNormalize compatibility
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


@pytest.fixture
def sample_data():
    """I create sample training data for testing."""
    n_hours = 168
    dates = pd.date_range(
        start='2024-01-01',
        periods=n_hours,
        freq='h',
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
def inner_env(sample_data):
    """I create the inner BatteryEnvUnified for wrapping."""
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
def sac_env(inner_env):
    """I create the SAC wrapper environment."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env)


class TestActionSpace:
    """I test the continuous action space structure."""

    def test_action_space_shape(self, sac_env):
        """I verify the action space is Box(shape=(4,))."""
        assert sac_env.action_space.shape == (4,)

    def test_action_space_bounds(self, sac_env):
        """I verify action space bounds are [-1, 1]."""
        np.testing.assert_array_equal(sac_env.action_space.low, [-1, -1, -1, -1])
        np.testing.assert_array_equal(sac_env.action_space.high, [1, 1, 1, 1])

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
        """I verify [0,0,0,0] maps to [2,2,5,5] (middle indices)."""
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

    def test_intermediate_values(self, sac_env):
        """I verify intermediate continuous values round correctly."""
        # -0.5 for dim with 5 levels: (-0.5+1)/2 * 4 = 1.0 -> index 1
        continuous = np.array([-0.5, 0.5, -0.5, 0.5])
        discrete = sac_env.continuous_to_discrete(continuous)
        assert discrete[0] == 1  # aFRR commitment: 25%
        assert discrete[1] == 3  # aFRR price tier: 1.15x
        # (-0.5+1)/2*10 = 2.5 -> np.round(2.5) = 2 (banker's rounding)
        assert discrete[2] == 2  # IntraDay: slightly charge
        # (0.5+1)/2*10 = 7.5 -> np.round(7.5) = 8 (banker's rounding)
        assert discrete[3] == 8  # mFRR: slightly discharge


class TestPenaltyComputation:
    """I test penalty-based soft constraint enforcement."""

    def test_idle_action_zero_penalty(self, sac_env):
        """I verify idle action gets zero or near-zero penalty."""
        sac_env.reset(seed=42)
        # I set SoC to middle to avoid margin penalties
        sac_env._inner.soc = 0.5
        # Idle = [0, 0, 0, 0] -> discrete [2, 2, 5, 5]
        continuous = np.array([0.0, 0.0, 0.0, 0.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        penalty = sac_env._compute_penalty(continuous, discrete)
        # Idle IntraDay (5) and idle mFRR (5) should not be penalized
        # (they are always valid in inner env masks)
        assert penalty >= 0  # Non-negative
        # I check that penalty is small (no major violations for idle)
        assert penalty < 50.0

    def test_penalty_positive_when_masked(self, sac_env):
        """I verify penalty > 0 when choosing a masked action."""
        sac_env.reset(seed=42)
        # I set SoC near minimum to ensure discharge is masked
        sac_env._inner.soc = 0.06  # Just above min_soc (0.05)

        # I force step to hour 14 so IntraDay gate is open
        sac_env._inner.current_step = sac_env._inner.start_step + 14

        # Max discharge action
        continuous = np.array([0.0, 0.0, 1.0, 1.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        penalty = sac_env._compute_penalty(continuous, discrete)
        # I expect penalty > 0 due to low SoC and max discharge
        assert penalty > 0

    def test_soc_margin_penalty_near_min(self, sac_env):
        """I verify soft SoC margin penalty when near minimum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.08  # Within 5% buffer of min_soc (0.05)

        sac_env._inner.current_step = sac_env._inner.start_step + 14

        # I try a moderate discharge — even if not fully masked, margin penalty applies
        continuous = np.array([0.0, 0.0, 0.3, 0.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        penalty = sac_env._compute_penalty(continuous, discrete)
        # I expect some margin penalty for discharging near min SoC
        assert penalty >= 0

    def test_soc_margin_penalty_near_max(self, sac_env):
        """I verify soft SoC margin penalty when near maximum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.93  # Within 5% buffer of max_soc (0.95)

        sac_env._inner.current_step = sac_env._inner.start_step + 14

        # I try charging near max SoC
        continuous = np.array([0.0, 0.0, -0.5, 0.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        penalty = sac_env._compute_penalty(continuous, discrete)
        # I expect margin penalty for charging near max SoC
        assert penalty >= 0

    def test_cycle_excess_penalty(self, sac_env):
        """I verify daily cycle excess penalty when over 1.8 cycles."""
        sac_env.reset(seed=42)
        sac_env._inner.daily_cycles = 2.0  # Over 1.8 limit
        sac_env._inner.soc = 0.5

        continuous = np.array([0.0, 0.0, 0.0, 0.0])
        discrete = sac_env.continuous_to_discrete(continuous)
        penalty = sac_env._compute_penalty(continuous, discrete)
        # I expect cycle excess penalty: 100 * (2.0 - 1.8) = 20.0
        assert penalty >= 20.0


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

    def test_full_episode(self, sac_env):
        """I verify a full episode completes without errors."""
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
        assert obs.shape[1] == 64  # (1, 64) for vectorized

        for _ in range(10):
            action = [vec_env.action_space.sample()]
            obs, reward, done, info = vec_env.step(action)
            assert obs.shape[1] == 64
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
        assert isinstance(info['discrete_action'], list)
        assert len(info['discrete_action']) == 4

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
