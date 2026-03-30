"""
Unit Tests for SAC Wrapper v2 — Absolute MW + XBID Price Control

I test the key functionality of the redesigned SAC wrapper:
1. Action space shape/bounds (Box [-1,1] shape=(7,))
2. Continuous-to-absolute MW mapping correctness
3. Proportional capacity scaling
4. XBID price offset decoding
5. Action filtering — invalid actions replaced with nearest valid
6. Soft penalty computation (SoC margin, super-linear cycling)
7. Full episode completion (legacy 4-dim inner + full-market 6-dim inner)
8. VecNormalize compatibility
9. Info dict with decoded MW and price offset
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

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
    """I create the SAC wrapper environment (legacy 4-dim inner)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env)


@pytest.fixture
def sac_env_full_market(inner_env_full_market):
    """I create the SAC wrapper environment (full-market 6-dim inner)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env_full_market)


class TestActionSpace:
    """I test the continuous action space structure."""

    def test_action_space_always_7dim(self, sac_env):
        """I verify action space is always Box(shape=(7,)) regardless of inner env."""
        assert sac_env.action_space.shape == """
Unit Tests for SAC Wrapper v2 — Absolute MW + XBID Price Control

I test the key functionality of the redesigned SAC wrapper:
1. Action space shape/bounds (Box [-1,1] shape=(7,))
2. Continuous-to-absolute MW mapping correctness
3. Proportional capacity scaling
4. XBID price offset decoding
5. Action filtering — invalid actions replaced with nearest valid
6. Soft penalty computation (SoC margin, super-linear cycling)
7. Full episode completion (legacy 4-dim inner + full-market 6-dim inner)
8. VecNormalize compatibility
9. Info dict with decoded MW and price offset
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

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
    """I create the SAC wrapper environment (legacy 4-dim inner)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env)


@pytest.fixture
def sac_env_full_market(inner_env_full_market):
    """I create the SAC wrapper environment (full-market 6-dim inner)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env_full_market)


class TestActionSpace:
    """I test the continuous action space structure."""

    def test_action_space_always_7dim(self, sac_env):
        """I verify action space is always Box(shape=(7,)) regardless of inner env."""
        assert sac_env.action_space.shape == (7,)

    def test_action_space_7dim_full_market(self, sac_env_full_market):
        """I verify full-market also has Box(shape=(7,))."""
        assert sac_env_full_market.action_space.shape == (7,)

    def test_action_space_bounds(self, sac_env):
        """I verify action space bounds are [-1, 1] x 6."""
        np.testing.assert_array_equal(sac_env.action_space.low, [-1] * 7)
        np.testing.assert_array_equal(sac_env.action_space.high, [1] * 7)

    def test_action_space_dtype(self, sac_env):
        """I verify action space uses float32."""
        assert sac_env.action_space.dtype == np.float32

    def test_observation_space_matches_inner(self, sac_env, inner_env):
        """I verify observation space delegates to inner env."""
        assert sac_env.observation_space.shape == inner_env.observation_space.shape


class TestContinuousToAbsolute:
    """I test the continuous-to-absolute MW conversion."""

    def test_all_zeros_gives_mid_values(self, sac_env):
        """I verify [0,0,0,0,0,0] gives mid-range values (aFRR=15MW, others=0)."""
        action = np.zeros(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(15.0, abs=0.1)  # (0+1)/2 * 30
        assert decoded['afrr_price_tier'] == pytest.approx(1.0, abs=0.01)  # middle tier
        assert decoded['xbid_mw'] == pytest.approx(0.0, abs=0.1)
        assert decoded['xbid_price_offset'] == pytest.approx(0.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(0.0, abs=0.1)

    def test_all_max(self, sac_env):
        """I verify [1,1,1,1,1,1] gives max values."""
        action = np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['afrr_price_tier'] == pytest.approx(1.3, abs=0.01)
        assert decoded['xbid_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['xbid_price_offset'] == pytest.approx(10.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(30.0, abs=0.1)

    def test_all_min(self, sac_env):
        """I verify [-1,-1,-1,-1,-1,-1] gives min values."""
        action = -np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(0.0, abs=0.1)  # aFRR min is 0
        assert decoded['afrr_price_tier'] == pytest.approx(0.7, abs=0.01)
        assert decoded['xbid_mw'] == pytest.approx(-30.0, abs=0.1)  # full charge
        assert decoded['xbid_price_offset'] == pytest.approx(-10.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(-30.0, abs=0.1)

    def test_clipping(self, sac_env):
        """I verify out-of-range values are clipped."""
        action = np.array([2.0, -3.0, 1.5, -1.5, 2.0, -2.0, 0.0], dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['xbid_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(30.0, abs=0.1)

    def test_freebid_zero_when_not_full_market(self, sac_env):
        """I verify FreeBid MW is 0 when inner env is legacy (not full-market)."""
        action = np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['freebid_mw'] == 0.0

    def test_xbid_price_offset_range(self, sac_env):
        """I verify XBID price offset maps correctly within range."""
        # I test several values
        for cont_val, expected_offset in [(-1.0, -10.0), (0.0, 0.0), (0.5, 5.0), (1.0, 10.0)]:
            action = np.array([0.0, 0.0, 0.0, cont_val, 0.0, 0.0, 0.0], dtype=np.float32)
            decoded = sac_env._decode_action(action)
            assert decoded['xbid_price_offset'] == pytest.approx(expected_offset, abs=0.1)


class TestCapacityEnforcement:
    """I test proportional capacity scaling."""

    def test_within_capacity_unchanged(self, sac_env):
        """I verify actions within capacity are not scaled."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {
            'afrr_mw': 5.0,
            'afrr_price_tier': 1.0,
            'xbid_mw': 5.0,
            'xbid_price_offset': 0.0,
            'mfrr_mw': 5.0,
            'freebid_mw': 0.0,
        }
        enforced = sac_env._enforce_constraints(decoded)

        # I verify values are unchanged (total 15MW < 30MW available)
        assert enforced['xbid_mw'] == pytest.approx(5.0, abs=0.5)
        assert enforced['mfrr_mw'] == pytest.approx(5.0, abs=0.5)

    def test_over_capacity_scaled_down(self, sac_env):
        """I verify actions exceeding capacity are scaled proportionally."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {
            'afrr_mw': 20.0,
            'afrr_price_tier': 1.0,
            'xbid_mw': 20.0,
            'xbid_price_offset': 0.0,
            'mfrr_mw': 20.0,
            'freebid_mw': 0.0,
        }
        enforced = sac_env._enforce_constraints(decoded)

        # I verify total MW is within max_power
        total = abs(enforced['afrr_mw']) + abs(enforced['xbid_mw']) + abs(enforced['mfrr_mw'])
        assert total <= 30.0 + 0.5, f"Total {total:.1f} should be <= 30.5"


class TestActionFiltering:
    """I test action filtering — nearest valid action replacement."""

    def test_step_never_violates_mask(self, sac_env):
        """I verify step() with random actions never crashes."""
        sac_env.reset(seed=42)

        for _ in range(50):
            action = sac_env.action_space.sample()
            obs, reward, done, truncated, info = sac_env.step(action)
            assert np.isfinite(reward)
            if done or truncated:
                sac_env.reset()

    def test_info_contains_decoded_mw(self, sac_env):
        """I verify info dict contains decoded MW values."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        assert 'decoded_mw' in info
        assert 'xbid' in info['decoded_mw']
        assert 'mfrr' in info['decoded_mw']
        assert 'xbid_price_offset' in info['decoded_mw']

    def test_filtering_full_market(self, sac_env_full_market):
        """I verify full-market mode works with direct execution."""
        sac_env_full_market.reset(seed=42)

        for _ in range(30):
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)
            assert np.isfinite(reward)
            assert 'decoded_mw' in info
            if done or truncated:
                sac_env_full_market.reset()


class TestSoftPenalties:
    """I test soft constraint penalties (SoC margin + super-linear cycling)."""

    def test_idle_action_zero_penalty(self, sac_env):
        """I verify idle action at mid SoC gets zero penalty."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_soc_margin_penalty_near_min(self, sac_env):
        """I verify soft SoC margin penalty when discharging near minimum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.08
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 10.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty > 0, "Should penalize discharge near min SoC"

    def test_soc_margin_penalty_near_max(self, sac_env):
        """I verify soft SoC margin penalty when charging near maximum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.93
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': -10.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty > 0, "Should penalize charge near max SoC"

    def test_no_soc_margin_penalty_at_mid_soc(self, sac_env):
        """I verify no SoC margin penalty at mid-range SoC."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 15.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 10.0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_cycle_penalty_below_target_is_zero(self, sac_env):
        """I verify no cycle penalty when below target."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 3.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_cycle_penalty_super_linear(self, sac_env):
        """I verify cycle penalty is super-linear (excess^1.5)."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}

        sac_env._inner.daily_cycles = 5.5
        penalty_small = sac_env._compute_penalty(decoded)

        sac_env._inner.daily_cycles = 8.0
        penalty_large = sac_env._compute_penalty(decoded)

        assert penalty_large > 5 * penalty_small, \
            f"Penalty should be super-linear: {penalty_large} vs {penalty_small}"

    def test_cycle_penalty_at_8_cycles(self, sac_env):
        """I verify cycle penalty is severe at 8 cycles/day."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 8.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)

        expected = 2000.0 * (3.0 ** 1.5)
        assert abs(penalty - expected) < 1.0, \
            f"Penalty at 8 cycles should be ~{expected:.0f}, got {penalty:.0f}"

    def test_cycle_penalty_magnitude_vs_profit(self, sac_env):
        """I verify cycle penalty at target+1 is meaningful."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 6.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)

        assert penalty >= 1500.0, \
            f"Penalty at 1 excess cycle should be >= 1500 EUR, got {penalty:.0f}"

    def test_no_hard_penalty_keys_in_defaults(self, sac_env):
        """I verify hard penalties were removed from defaults."""
        assert 'soc_hard' not in sac_env.penalties
        assert 'gate_closure' not in sac_env.penalties


class TestRewardCalculator:
    """I test reward calculator accessibility for curriculum callbacks."""

    def test_reward_calculator_accessible(self, sac_env):
        """I verify reward calculator is accessible via property."""
        rc = sac_env.reward_calculator
        assert rc is not None
        assert hasattr(rc, 'degradation_cost')

    def test_reward_calculator_settable(self, sac_env):
        """I verify degradation cost can be updated."""
        sac_env.reward_calculator.degradation_cost = 10.0
        assert sac_env._inner.reward_calculator.degradation_cost == 10.0


class TestEpisodeCompletion:
    """I test full episode execution."""

    def test_full_episode_legacy(self, sac_env):
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
            assert 'decoded_mw' in info

            total_reward += reward
            steps += 1

            if done or truncated:
                break

        assert steps > 0
        assert np.isfinite(total_reward)

    def test_full_episode_full_market(self, sac_env_full_market):
        """I verify a full episode completes in full-market mode."""
        obs, info = sac_env_full_market.reset(seed=42)
        assert obs.shape == sac_env_full_market.observation_space.shape

        total_reward = 0.0
        steps = 0

        while True:
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)

            assert obs.shape == sac_env_full_market.observation_space.shape
            assert np.isfinite(reward)
            assert 'decoded_mw' in info

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
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        obs = vec_env.reset()
        assert obs.shape[1] == 72

        for _ in range(10):
            action = [vec_env.action_space.sample()]
            obs, reward, done, info = vec_env.step(action)
            assert obs.shape[1] == 72
            assert np.all(np.isfinite(obs))

    def test_observation_normalization_stable(self, sample_data):
        """I verify observations remain finite after normalization."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
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

    def test_info_contains_all_fields(self, sac_env):
        """I verify step info includes all SAC v2 fields."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        assert 'sac_penalty' in info
        assert 'sac_penalty_scaled' in info
        assert 'discrete_action' in info
        assert 'decoded_mw' in info

    def test_decoded_mw_fields(self, sac_env):
        """I verify decoded_mw contains MW and price offset."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        decoded = info['decoded_mw']
        assert 'afrr' in decoded
        assert 'xbid' in decoded
        assert 'xbid_price_offset' in decoded
        assert 'mfrr' in decoded
        assert 'freebid' in decoded

    def test_info_full_market_has_freebid(self, sac_env_full_market):
        """I verify full-market info includes freebid MW."""
        sac_env_full_market.reset(seed=42)
        action = sac_env_full_market.action_space.sample()
        _, _, _, _, info = sac_env_full_market.step(action)

        assert 'decoded_mw' in info
        assert 'freebid' in info['decoded_mw']

    def test_penalty_scaling(self, sac_env):
        """I verify penalty is scaled by reward_scale."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        reward_scale = sac_env._inner.reward_calculator.reward_scale
        expected_scaled = info['sac_penalty'] * reward_scale
        assert abs(info['sac_penalty_scaled'] - expected_scaled) < 1e-6


class TestSetDegradationCost:
    """I test the set_degradation_cost method."""

    def test_set_degradation_cost_method(self, sac_env):
        """I verify set_degradation_cost updates inner env."""
        sac_env.reset(seed=42)
        sac_env.set_degradation_cost(5.0)
        assert sac_env._inner.reward_calculator.degradation_cost == 5.0

    def test_set_degradation_cost_via_vecenv(self, sample_data):
        """I verify set_degradation_cost propagates through DummyVecEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env.reset()
        vec_env.env_method('set_degradation_cost', 3.0)

        inner_cost = vec_env.envs[0]._inner.reward_calculator.degradation_cost
        assert inner_cost == 3.0


class TestSetPenalties:
    """I test runtime penalty override."""

    def test_set_penalties_updates_values(self, sac_env):
        """I verify set_penalties updates penalty dict."""
        original_cycle = sac_env.penalties['cycle_excess']
        sac_env.set_penalties({'cycle_excess': 5000.0})
        assert sac_env.penalties['cycle_excess'] == 5000.0
        assert sac_env.penalties['cycle_excess'] != original_cycle

    def test_set_penalties_partial_update(self, sac_env):
        """I verify set_penalties only updates specified keys."""
        original_margin = sac_env.penalties['soc_soft_margin']
        sac_env.set_penalties({'cycle_excess': 3000.0})
        assert sac_env.penalties['soc_soft_margin'] == original_margin


    def test_action_space_7dim_full_market(self, sac_env_full_market):
        """I verify full-market also has Box(shape=(7,))."""
        assert sac_env_full_market.action_space.shape == """
Unit Tests for SAC Wrapper v2 — Absolute MW + XBID Price Control

I test the key functionality of the redesigned SAC wrapper:
1. Action space shape/bounds (Box [-1,1] shape=(7,))
2. Continuous-to-absolute MW mapping correctness
3. Proportional capacity scaling
4. XBID price offset decoding
5. Action filtering — invalid actions replaced with nearest valid
6. Soft penalty computation (SoC margin, super-linear cycling)
7. Full episode completion (legacy 4-dim inner + full-market 6-dim inner)
8. VecNormalize compatibility
9. Info dict with decoded MW and price offset
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

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
    """I create the SAC wrapper environment (legacy 4-dim inner)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env)


@pytest.fixture
def sac_env_full_market(inner_env_full_market):
    """I create the SAC wrapper environment (full-market 6-dim inner)."""
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    return BatteryEnvUnifiedSAC(inner_env=inner_env_full_market)


class TestActionSpace:
    """I test the continuous action space structure."""

    def test_action_space_always_7dim(self, sac_env):
        """I verify action space is always Box(shape=(7,)) regardless of inner env."""
        assert sac_env.action_space.shape == (7,)

    def test_action_space_7dim_full_market(self, sac_env_full_market):
        """I verify full-market also has Box(shape=(7,))."""
        assert sac_env_full_market.action_space.shape == (7,)

    def test_action_space_bounds(self, sac_env):
        """I verify action space bounds are [-1, 1] x 6."""
        np.testing.assert_array_equal(sac_env.action_space.low, [-1] * 7)
        np.testing.assert_array_equal(sac_env.action_space.high, [1] * 7)

    def test_action_space_dtype(self, sac_env):
        """I verify action space uses float32."""
        assert sac_env.action_space.dtype == np.float32

    def test_observation_space_matches_inner(self, sac_env, inner_env):
        """I verify observation space delegates to inner env."""
        assert sac_env.observation_space.shape == inner_env.observation_space.shape


class TestContinuousToAbsolute:
    """I test the continuous-to-absolute MW conversion."""

    def test_all_zeros_gives_mid_values(self, sac_env):
        """I verify [0,0,0,0,0,0] gives mid-range values (aFRR=15MW, others=0)."""
        action = np.zeros(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(15.0, abs=0.1)  # (0+1)/2 * 30
        assert decoded['afrr_price_tier'] == pytest.approx(1.0, abs=0.01)  # middle tier
        assert decoded['xbid_mw'] == pytest.approx(0.0, abs=0.1)
        assert decoded['xbid_price_offset'] == pytest.approx(0.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(0.0, abs=0.1)

    def test_all_max(self, sac_env):
        """I verify [1,1,1,1,1,1] gives max values."""
        action = np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['afrr_price_tier'] == pytest.approx(1.3, abs=0.01)
        assert decoded['xbid_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['xbid_price_offset'] == pytest.approx(10.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(30.0, abs=0.1)

    def test_all_min(self, sac_env):
        """I verify [-1,-1,-1,-1,-1,-1] gives min values."""
        action = -np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(0.0, abs=0.1)  # aFRR min is 0
        assert decoded['afrr_price_tier'] == pytest.approx(0.7, abs=0.01)
        assert decoded['xbid_mw'] == pytest.approx(-30.0, abs=0.1)  # full charge
        assert decoded['xbid_price_offset'] == pytest.approx(-10.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(-30.0, abs=0.1)

    def test_clipping(self, sac_env):
        """I verify out-of-range values are clipped."""
        action = np.array([2.0, -3.0, 1.5, -1.5, 2.0, -2.0, 0.0], dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['xbid_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(30.0, abs=0.1)

    def test_freebid_zero_when_not_full_market(self, sac_env):
        """I verify FreeBid MW is 0 when inner env is legacy (not full-market)."""
        action = np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['freebid_mw'] == 0.0

    def test_xbid_price_offset_range(self, sac_env):
        """I verify XBID price offset maps correctly within range."""
        # I test several values
        for cont_val, expected_offset in [(-1.0, -10.0), (0.0, 0.0), (0.5, 5.0), (1.0, 10.0)]:
            action = np.array([0.0, 0.0, 0.0, cont_val, 0.0, 0.0, 0.0], dtype=np.float32)
            decoded = sac_env._decode_action(action)
            assert decoded['xbid_price_offset'] == pytest.approx(expected_offset, abs=0.1)


class TestCapacityEnforcement:
    """I test proportional capacity scaling."""

    def test_within_capacity_unchanged(self, sac_env):
        """I verify actions within capacity are not scaled."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {
            'afrr_mw': 5.0,
            'afrr_price_tier': 1.0,
            'xbid_mw': 5.0,
            'xbid_price_offset': 0.0,
            'mfrr_mw': 5.0,
            'freebid_mw': 0.0,
        }
        enforced = sac_env._enforce_constraints(decoded)

        # I verify values are unchanged (total 15MW < 30MW available)
        assert enforced['xbid_mw'] == pytest.approx(5.0, abs=0.5)
        assert enforced['mfrr_mw'] == pytest.approx(5.0, abs=0.5)

    def test_over_capacity_scaled_down(self, sac_env):
        """I verify actions exceeding capacity are scaled proportionally."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {
            'afrr_mw': 20.0,
            'afrr_price_tier': 1.0,
            'xbid_mw': 20.0,
            'xbid_price_offset': 0.0,
            'mfrr_mw': 20.0,
            'freebid_mw': 0.0,
        }
        enforced = sac_env._enforce_constraints(decoded)

        # I verify total MW is within max_power
        total = abs(enforced['afrr_mw']) + abs(enforced['xbid_mw']) + abs(enforced['mfrr_mw'])
        assert total <= 30.0 + 0.5, f"Total {total:.1f} should be <= 30.5"


class TestActionFiltering:
    """I test action filtering — nearest valid action replacement."""

    def test_step_never_violates_mask(self, sac_env):
        """I verify step() with random actions never crashes."""
        sac_env.reset(seed=42)

        for _ in range(50):
            action = sac_env.action_space.sample()
            obs, reward, done, truncated, info = sac_env.step(action)
            assert np.isfinite(reward)
            if done or truncated:
                sac_env.reset()

    def test_info_contains_decoded_mw(self, sac_env):
        """I verify info dict contains decoded MW values."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        assert 'decoded_mw' in info
        assert 'xbid' in info['decoded_mw']
        assert 'mfrr' in info['decoded_mw']
        assert 'xbid_price_offset' in info['decoded_mw']

    def test_filtering_full_market(self, sac_env_full_market):
        """I verify full-market mode works with direct execution."""
        sac_env_full_market.reset(seed=42)

        for _ in range(30):
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)
            assert np.isfinite(reward)
            assert 'decoded_mw' in info
            if done or truncated:
                sac_env_full_market.reset()


class TestSoftPenalties:
    """I test soft constraint penalties (SoC margin + super-linear cycling)."""

    def test_idle_action_zero_penalty(self, sac_env):
        """I verify idle action at mid SoC gets zero penalty."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_soc_margin_penalty_near_min(self, sac_env):
        """I verify soft SoC margin penalty when discharging near minimum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.08
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 10.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty > 0, "Should penalize discharge near min SoC"

    def test_soc_margin_penalty_near_max(self, sac_env):
        """I verify soft SoC margin penalty when charging near maximum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.93
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': -10.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty > 0, "Should penalize charge near max SoC"

    def test_no_soc_margin_penalty_at_mid_soc(self, sac_env):
        """I verify no SoC margin penalty at mid-range SoC."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 15.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 10.0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_cycle_penalty_below_target_is_zero(self, sac_env):
        """I verify no cycle penalty when below target."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 3.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_cycle_penalty_super_linear(self, sac_env):
        """I verify cycle penalty is super-linear (excess^1.5)."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}

        sac_env._inner.daily_cycles = 5.5
        penalty_small = sac_env._compute_penalty(decoded)

        sac_env._inner.daily_cycles = 8.0
        penalty_large = sac_env._compute_penalty(decoded)

        assert penalty_large > 5 * penalty_small, \
            f"Penalty should be super-linear: {penalty_large} vs {penalty_small}"

    def test_cycle_penalty_at_8_cycles(self, sac_env):
        """I verify cycle penalty is severe at 8 cycles/day."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 8.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)

        expected = 2000.0 * (3.0 ** 1.5)
        assert abs(penalty - expected) < 1.0, \
            f"Penalty at 8 cycles should be ~{expected:.0f}, got {penalty:.0f}"

    def test_cycle_penalty_magnitude_vs_profit(self, sac_env):
        """I verify cycle penalty at target+1 is meaningful."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 6.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)

        assert penalty >= 1500.0, \
            f"Penalty at 1 excess cycle should be >= 1500 EUR, got {penalty:.0f}"

    def test_no_hard_penalty_keys_in_defaults(self, sac_env):
        """I verify hard penalties were removed from defaults."""
        assert 'soc_hard' not in sac_env.penalties
        assert 'gate_closure' not in sac_env.penalties


class TestRewardCalculator:
    """I test reward calculator accessibility for curriculum callbacks."""

    def test_reward_calculator_accessible(self, sac_env):
        """I verify reward calculator is accessible via property."""
        rc = sac_env.reward_calculator
        assert rc is not None
        assert hasattr(rc, 'degradation_cost')

    def test_reward_calculator_settable(self, sac_env):
        """I verify degradation cost can be updated."""
        sac_env.reward_calculator.degradation_cost = 10.0
        assert sac_env._inner.reward_calculator.degradation_cost == 10.0


class TestEpisodeCompletion:
    """I test full episode execution."""

    def test_full_episode_legacy(self, sac_env):
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
            assert 'decoded_mw' in info

            total_reward += reward
            steps += 1

            if done or truncated:
                break

        assert steps > 0
        assert np.isfinite(total_reward)

    def test_full_episode_full_market(self, sac_env_full_market):
        """I verify a full episode completes in full-market mode."""
        obs, info = sac_env_full_market.reset(seed=42)
        assert obs.shape == sac_env_full_market.observation_space.shape

        total_reward = 0.0
        steps = 0

        while True:
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)

            assert obs.shape == sac_env_full_market.observation_space.shape
            assert np.isfinite(reward)
            assert 'decoded_mw' in info

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
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        obs = vec_env.reset()
        assert obs.shape[1] == 72

        for _ in range(10):
            action = [vec_env.action_space.sample()]
            obs, reward, done, info = vec_env.step(action)
            assert obs.shape[1] == 72
            assert np.all(np.isfinite(obs))

    def test_observation_normalization_stable(self, sample_data):
        """I verify observations remain finite after normalization."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
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

    def test_info_contains_all_fields(self, sac_env):
        """I verify step info includes all SAC v2 fields."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        assert 'sac_penalty' in info
        assert 'sac_penalty_scaled' in info
        assert 'discrete_action' in info
        assert 'decoded_mw' in info

    def test_decoded_mw_fields(self, sac_env):
        """I verify decoded_mw contains MW and price offset."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        decoded = info['decoded_mw']
        assert 'afrr' in decoded
        assert 'xbid' in decoded
        assert 'xbid_price_offset' in decoded
        assert 'mfrr' in decoded
        assert 'freebid' in decoded

    def test_info_full_market_has_freebid(self, sac_env_full_market):
        """I verify full-market info includes freebid MW."""
        sac_env_full_market.reset(seed=42)
        action = sac_env_full_market.action_space.sample()
        _, _, _, _, info = sac_env_full_market.step(action)

        assert 'decoded_mw' in info
        assert 'freebid' in info['decoded_mw']

    def test_penalty_scaling(self, sac_env):
        """I verify penalty is scaled by reward_scale."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        reward_scale = sac_env._inner.reward_calculator.reward_scale
        expected_scaled = info['sac_penalty'] * reward_scale
        assert abs(info['sac_penalty_scaled'] - expected_scaled) < 1e-6


class TestSetDegradationCost:
    """I test the set_degradation_cost method."""

    def test_set_degradation_cost_method(self, sac_env):
        """I verify set_degradation_cost updates inner env."""
        sac_env.reset(seed=42)
        sac_env.set_degradation_cost(5.0)
        assert sac_env._inner.reward_calculator.degradation_cost == 5.0

    def test_set_degradation_cost_via_vecenv(self, sample_data):
        """I verify set_degradation_cost propagates through DummyVecEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env.reset()
        vec_env.env_method('set_degradation_cost', 3.0)

        inner_cost = vec_env.envs[0]._inner.reward_calculator.degradation_cost
        assert inner_cost == 3.0


class TestSetPenalties:
    """I test runtime penalty override."""

    def test_set_penalties_updates_values(self, sac_env):
        """I verify set_penalties updates penalty dict."""
        original_cycle = sac_env.penalties['cycle_excess']
        sac_env.set_penalties({'cycle_excess': 5000.0})
        assert sac_env.penalties['cycle_excess'] == 5000.0
        assert sac_env.penalties['cycle_excess'] != original_cycle

    def test_set_penalties_partial_update(self, sac_env):
        """I verify set_penalties only updates specified keys."""
        original_margin = sac_env.penalties['soc_soft_margin']
        sac_env.set_penalties({'cycle_excess': 3000.0})
        assert sac_env.penalties['soc_soft_margin'] == original_margin


    def test_action_space_bounds(self, sac_env):
        """I verify action space bounds are [-1, 1] x 6."""
        np.testing.assert_array_equal(sac_env.action_space.low, [-1] * 7)
        np.testing.assert_array_equal(sac_env.action_space.high, [1] * 7)

    def test_action_space_dtype(self, sac_env):
        """I verify action space uses float32."""
        assert sac_env.action_space.dtype == np.float32

    def test_observation_space_matches_inner(self, sac_env, inner_env):
        """I verify observation space delegates to inner env."""
        assert sac_env.observation_space.shape == inner_env.observation_space.shape


class TestContinuousToAbsolute:
    """I test the continuous-to-absolute MW conversion."""

    def test_all_zeros_gives_mid_values(self, sac_env):
        """I verify [0,0,0,0,0,0] gives mid-range values (aFRR=15MW, others=0)."""
        action = np.zeros(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(15.0, abs=0.1)  # (0+1)/2 * 30
        assert decoded['afrr_price_tier'] == pytest.approx(1.0, abs=0.01)  # middle tier
        assert decoded['xbid_mw'] == pytest.approx(0.0, abs=0.1)
        assert decoded['xbid_price_offset'] == pytest.approx(0.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(0.0, abs=0.1)

    def test_all_max(self, sac_env):
        """I verify [1,1,1,1,1,1] gives max values."""
        action = np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['afrr_price_tier'] == pytest.approx(1.3, abs=0.01)
        assert decoded['xbid_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['xbid_price_offset'] == pytest.approx(10.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(30.0, abs=0.1)

    def test_all_min(self, sac_env):
        """I verify [-1,-1,-1,-1,-1,-1] gives min values."""
        action = -np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(0.0, abs=0.1)  # aFRR min is 0
        assert decoded['afrr_price_tier'] == pytest.approx(0.7, abs=0.01)
        assert decoded['xbid_mw'] == pytest.approx(-30.0, abs=0.1)  # full charge
        assert decoded['xbid_price_offset'] == pytest.approx(-10.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(-30.0, abs=0.1)

    def test_clipping(self, sac_env):
        """I verify out-of-range values are clipped."""
        action = np.array([2.0, -3.0, 1.5, -1.5, 2.0, -2.0, 0.0], dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['afrr_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['xbid_mw'] == pytest.approx(30.0, abs=0.1)
        assert decoded['mfrr_mw'] == pytest.approx(30.0, abs=0.1)

    def test_freebid_zero_when_not_full_market(self, sac_env):
        """I verify FreeBid MW is 0 when inner env is legacy (not full-market)."""
        action = np.ones(7, dtype=np.float32)
        decoded = sac_env._decode_action(action)

        assert decoded['freebid_mw'] == 0.0

    def test_xbid_price_offset_range(self, sac_env):
        """I verify XBID price offset maps correctly within range."""
        # I test several values
        for cont_val, expected_offset in [(-1.0, -10.0), (0.0, 0.0), (0.5, 5.0), (1.0, 10.0)]:
            action = np.array([0.0, 0.0, 0.0, cont_val, 0.0, 0.0, 0.0], dtype=np.float32)
            decoded = sac_env._decode_action(action)
            assert decoded['xbid_price_offset'] == pytest.approx(expected_offset, abs=0.1)


class TestCapacityEnforcement:
    """I test proportional capacity scaling."""

    def test_within_capacity_unchanged(self, sac_env):
        """I verify actions within capacity are not scaled."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {
            'afrr_mw': 5.0,
            'afrr_price_tier': 1.0,
            'xbid_mw': 5.0,
            'xbid_price_offset': 0.0,
            'mfrr_mw': 5.0,
            'freebid_mw': 0.0,
        }
        enforced = sac_env._enforce_constraints(decoded)

        # I verify values are unchanged (total 15MW < 30MW available)
        assert enforced['xbid_mw'] == pytest.approx(5.0, abs=0.5)
        assert enforced['mfrr_mw'] == pytest.approx(5.0, abs=0.5)

    def test_over_capacity_scaled_down(self, sac_env):
        """I verify actions exceeding capacity are scaled proportionally."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {
            'afrr_mw': 20.0,
            'afrr_price_tier': 1.0,
            'xbid_mw': 20.0,
            'xbid_price_offset': 0.0,
            'mfrr_mw': 20.0,
            'freebid_mw': 0.0,
        }
        enforced = sac_env._enforce_constraints(decoded)

        # I verify total MW is within max_power
        total = abs(enforced['afrr_mw']) + abs(enforced['xbid_mw']) + abs(enforced['mfrr_mw'])
        assert total <= 30.0 + 0.5, f"Total {total:.1f} should be <= 30.5"


class TestActionFiltering:
    """I test action filtering — nearest valid action replacement."""

    def test_step_never_violates_mask(self, sac_env):
        """I verify step() with random actions never crashes."""
        sac_env.reset(seed=42)

        for _ in range(50):
            action = sac_env.action_space.sample()
            obs, reward, done, truncated, info = sac_env.step(action)
            assert np.isfinite(reward)
            if done or truncated:
                sac_env.reset()

    def test_info_contains_decoded_mw(self, sac_env):
        """I verify info dict contains decoded MW values."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        assert 'decoded_mw' in info
        assert 'xbid' in info['decoded_mw']
        assert 'mfrr' in info['decoded_mw']
        assert 'xbid_price_offset' in info['decoded_mw']

    def test_filtering_full_market(self, sac_env_full_market):
        """I verify full-market mode works with direct execution."""
        sac_env_full_market.reset(seed=42)

        for _ in range(30):
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)
            assert np.isfinite(reward)
            assert 'decoded_mw' in info
            if done or truncated:
                sac_env_full_market.reset()


class TestSoftPenalties:
    """I test soft constraint penalties (SoC margin + super-linear cycling)."""

    def test_idle_action_zero_penalty(self, sac_env):
        """I verify idle action at mid SoC gets zero penalty."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_soc_margin_penalty_near_min(self, sac_env):
        """I verify soft SoC margin penalty when discharging near minimum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.08
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 10.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty > 0, "Should penalize discharge near min SoC"

    def test_soc_margin_penalty_near_max(self, sac_env):
        """I verify soft SoC margin penalty when charging near maximum."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.93
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': -10.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty > 0, "Should penalize charge near max SoC"

    def test_no_soc_margin_penalty_at_mid_soc(self, sac_env):
        """I verify no SoC margin penalty at mid-range SoC."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 15.0, 'xbid_price_offset': 0,
                   'mfrr_mw': 10.0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_cycle_penalty_below_target_is_zero(self, sac_env):
        """I verify no cycle penalty when below target."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 3.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)
        assert penalty == 0.0

    def test_cycle_penalty_super_linear(self, sac_env):
        """I verify cycle penalty is super-linear (excess^1.5)."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}

        sac_env._inner.daily_cycles = 5.5
        penalty_small = sac_env._compute_penalty(decoded)

        sac_env._inner.daily_cycles = 8.0
        penalty_large = sac_env._compute_penalty(decoded)

        assert penalty_large > 5 * penalty_small, \
            f"Penalty should be super-linear: {penalty_large} vs {penalty_small}"

    def test_cycle_penalty_at_8_cycles(self, sac_env):
        """I verify cycle penalty is severe at 8 cycles/day."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 8.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)

        expected = 2000.0 * (3.0 ** 1.5)
        assert abs(penalty - expected) < 1.0, \
            f"Penalty at 8 cycles should be ~{expected:.0f}, got {penalty:.0f}"

    def test_cycle_penalty_magnitude_vs_profit(self, sac_env):
        """I verify cycle penalty at target+1 is meaningful."""
        sac_env.reset(seed=42)
        sac_env._inner.soc = 0.5
        sac_env._inner.daily_cycles = 6.0

        decoded = {'afrr_mw': 0, 'xbid_mw': 0, 'xbid_price_offset': 0,
                   'mfrr_mw': 0, 'freebid_mw': 0, 'afrr_price_tier': 1.0}
        penalty = sac_env._compute_penalty(decoded)

        assert penalty >= 1500.0, \
            f"Penalty at 1 excess cycle should be >= 1500 EUR, got {penalty:.0f}"

    def test_no_hard_penalty_keys_in_defaults(self, sac_env):
        """I verify hard penalties were removed from defaults."""
        assert 'soc_hard' not in sac_env.penalties
        assert 'gate_closure' not in sac_env.penalties


class TestRewardCalculator:
    """I test reward calculator accessibility for curriculum callbacks."""

    def test_reward_calculator_accessible(self, sac_env):
        """I verify reward calculator is accessible via property."""
        rc = sac_env.reward_calculator
        assert rc is not None
        assert hasattr(rc, 'degradation_cost')

    def test_reward_calculator_settable(self, sac_env):
        """I verify degradation cost can be updated."""
        sac_env.reward_calculator.degradation_cost = 10.0
        assert sac_env._inner.reward_calculator.degradation_cost == 10.0


class TestEpisodeCompletion:
    """I test full episode execution."""

    def test_full_episode_legacy(self, sac_env):
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
            assert 'decoded_mw' in info

            total_reward += reward
            steps += 1

            if done or truncated:
                break

        assert steps > 0
        assert np.isfinite(total_reward)

    def test_full_episode_full_market(self, sac_env_full_market):
        """I verify a full episode completes in full-market mode."""
        obs, info = sac_env_full_market.reset(seed=42)
        assert obs.shape == sac_env_full_market.observation_space.shape

        total_reward = 0.0
        steps = 0

        while True:
            action = sac_env_full_market.action_space.sample()
            obs, reward, done, truncated, info = sac_env_full_market.step(action)

            assert obs.shape == sac_env_full_market.observation_space.shape
            assert np.isfinite(reward)
            assert 'decoded_mw' in info

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
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        obs = vec_env.reset()
        assert obs.shape[1] == 72

        for _ in range(10):
            action = [vec_env.action_space.sample()]
            obs, reward, done, info = vec_env.step(action)
            assert obs.shape[1] == 72
            assert np.all(np.isfinite(obs))

    def test_observation_normalization_stable(self, sample_data):
        """I verify observations remain finite after normalization."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
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

    def test_info_contains_all_fields(self, sac_env):
        """I verify step info includes all SAC v2 fields."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        assert 'sac_penalty' in info
        assert 'sac_penalty_scaled' in info
        assert 'discrete_action' in info
        assert 'decoded_mw' in info

    def test_decoded_mw_fields(self, sac_env):
        """I verify decoded_mw contains MW and price offset."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        decoded = info['decoded_mw']
        assert 'afrr' in decoded
        assert 'xbid' in decoded
        assert 'xbid_price_offset' in decoded
        assert 'mfrr' in decoded
        assert 'freebid' in decoded

    def test_info_full_market_has_freebid(self, sac_env_full_market):
        """I verify full-market info includes freebid MW."""
        sac_env_full_market.reset(seed=42)
        action = sac_env_full_market.action_space.sample()
        _, _, _, _, info = sac_env_full_market.step(action)

        assert 'decoded_mw' in info
        assert 'freebid' in info['decoded_mw']

    def test_penalty_scaling(self, sac_env):
        """I verify penalty is scaled by reward_scale."""
        sac_env.reset(seed=42)
        action = sac_env.action_space.sample()
        _, _, _, _, info = sac_env.step(action)

        reward_scale = sac_env._inner.reward_calculator.reward_scale
        expected_scaled = info['sac_penalty'] * reward_scale
        assert abs(info['sac_penalty_scaled'] - expected_scaled) < 1e-6


class TestSetDegradationCost:
    """I test the set_degradation_cost method."""

    def test_set_degradation_cost_method(self, sac_env):
        """I verify set_degradation_cost updates inner env."""
        sac_env.reset(seed=42)
        sac_env.set_degradation_cost(5.0)
        assert sac_env._inner.reward_calculator.degradation_cost == 5.0

    def test_set_degradation_cost_via_vecenv(self, sample_data):
        """I verify set_degradation_cost propagates through DummyVecEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from gym_envs.battery_env_unified import BatteryEnvUnified
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

        def make_env():
            inner = BatteryEnvUnified(
                df=sample_data, capacity_mwh=146.0, max_power_mw=30.0,
                efficiency=0.94, episode_length=72, random_start=False
            )
            return BatteryEnvUnifiedSAC(inner_env=inner)

        vec_env = DummyVecEnv([make_env])
        vec_env.reset()
        vec_env.env_method('set_degradation_cost', 3.0)

        inner_cost = vec_env.envs[0]._inner.reward_calculator.degradation_cost
        assert inner_cost == 3.0


class TestSetPenalties:
    """I test runtime penalty override."""

    def test_set_penalties_updates_values(self, sac_env):
        """I verify set_penalties updates penalty dict."""
        original_cycle = sac_env.penalties['cycle_excess']
        sac_env.set_penalties({'cycle_excess': 5000.0})
        assert sac_env.penalties['cycle_excess'] == 5000.0
        assert sac_env.penalties['cycle_excess'] != original_cycle

    def test_set_penalties_partial_update(self, sac_env):
        """I verify set_penalties only updates specified keys."""
        original_margin = sac_env.penalties['soc_soft_margin']
        sac_env.set_penalties({'cycle_excess': 3000.0})
        assert sac_env.penalties['soc_soft_margin'] == original_margin
