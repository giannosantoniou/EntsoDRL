"""
Unit Tests for Unified Multi-Market Battery Environment

I test the key functionality of the unified environment:
1. Action space structure (MultiDiscrete [5, 5, 11, 11])
2. Observation space (63 features)
3. Action masking (capacity cascading: DAM -> aFRR -> IntraDay -> mFRR)
4. aFRR activation simulation
5. Reward calculation
6. Separate IntraDay/mFRR capacity cascading
7. Separate IntraDay/mFRR revenue tracking
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
    # I create 1 week of hourly data
    n_hours = 168
    dates = pd.date_range(
        start='2024-01-01',
        periods=n_hours,
        freq='H',
        tz='Europe/Athens'
    )

    np.random.seed(42)

    # I create realistic price patterns
    hours = np.array([d.hour for d in dates])
    base_price = 80 + 40 * np.sin(2 * np.pi * hours / 24)  # Daily pattern
    noise = np.random.normal(0, 10, n_hours)
    prices = base_price + noise

    # I create DAM commitments (sparse)
    dam_commitments = np.zeros(n_hours)
    for i in range(n_hours):
        if hours[i] in [18, 19, 20]:  # Peak discharge
            dam_commitments[i] = np.random.uniform(10, 25)
        elif hours[i] in [10, 11, 12]:  # Solar charge
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
        'net_imbalance_mw': np.random.normal(0, 300, n_hours),
        'solar': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)),
        'wind_onshore': np.random.uniform(100, 500, n_hours),
        'res_total_mw': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)) + np.random.uniform(100, 500, n_hours),
        'load_mw': np.random.uniform(3000, 6000, n_hours)
    }

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def unified_env(sample_data):
    """I create a unified environment for testing."""
    from gym_envs.battery_env_unified import BatteryEnvUnified

    env = BatteryEnvUnified(
        df=sample_data,
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        episode_length=72,
        random_start=False
    )
    return env


class TestActionSpace:
    """I test the action space structure."""

    def test_action_space_shape(self, unified_env):
        """I verify the action space is MultiDiscrete([5, 5, 11, 11])."""
        assert unified_env.action_space.nvec.tolist() == [5, 5, 11, 11]

    def test_afrr_levels(self, unified_env):
        """I verify aFRR commitment levels."""
        expected = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        np.testing.assert_array_equal(unified_env.AFRR_LEVELS, expected)

    def test_price_tiers(self, unified_env):
        """I verify aFRR price tiers."""
        expected = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
        np.testing.assert_array_equal(unified_env.AFRR_PRICE_TIERS, expected)

    def test_intraday_levels(self, unified_env):
        """I verify IntraDay action levels span -1 to +1 with 11 levels."""
        levels = unified_env.INTRADAY_LEVELS
        assert len(levels) == 11
        assert levels[0] == pytest.approx(-1.0)
        assert levels[5] == pytest.approx(0.0)
        assert levels[10] == pytest.approx(1.0)

    def test_mfrr_levels(self, unified_env):
        """I verify mFRR action levels span -1 to +1 with 11 levels."""
        levels = unified_env.MFRR_LEVELS
        assert len(levels) == 11
        assert levels[0] == pytest.approx(-1.0)
        assert levels[5] == pytest.approx(0.0)
        assert levels[10] == pytest.approx(1.0)


class TestObservationSpace:
    """I test the observation space structure."""

    def test_observation_shape(self, unified_env):
        """I verify the observation space is 63 features."""
        assert unified_env.observation_space.shape == (63,)

    def test_observation_reset(self, unified_env):
        """I verify observation is correctly shaped after reset."""
        obs, info = unified_env.reset(seed=42)
        assert obs.shape == (63,)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_observation_step(self, unified_env):
        """I verify observation is correctly shaped after step."""
        unified_env.reset(seed=42)
        action = np.array([0, 2, 5, 5])  # No aFRR, neutral price, idle ID, idle mFRR
        obs, reward, done, truncated, info = unified_env.step(action)
        assert obs.shape == (63,)
        assert isinstance(reward, (int, float))


class TestActionMasking:
    """I test hierarchical action masking."""

    def test_mask_shape(self, unified_env):
        """I verify the action mask is correctly shaped (32 = 5+5+11+11)."""
        unified_env.reset(seed=42)
        mask = unified_env.action_masks()
        assert len(mask) == 5 + 5 + 11 + 11  # 32 total

    def test_idle_always_valid(self, unified_env):
        """I verify idle actions are always valid for both IntraDay and mFRR."""
        unified_env.reset(seed=42)
        mask = unified_env.action_masks()

        # IntraDay idle: index 5 in the IntraDay section (offset 10)
        intraday_mask = mask[10:21]
        assert intraday_mask[5] == True

        # mFRR idle: index 5 in the mFRR section (offset 21)
        mfrr_mask = mask[21:32]
        assert mfrr_mask[5] == True

    def test_zero_afrr_always_valid(self, unified_env):
        """I verify zero aFRR commitment is always valid."""
        unified_env.reset(seed=42)
        mask = unified_env.action_masks()
        assert mask[0] == True  # First aFRR action is zero commitment

    def test_capacity_cascading(self, unified_env, sample_data):
        """I verify capacity cascading respects DAM commitment."""
        dam_steps = sample_data[sample_data['dam_commitment'].abs() > 5].index

        if len(dam_steps) > 0:
            unified_env.reset(seed=42)
            unified_env.soc = 0.5  # Middle SoC for flexibility

            mask = unified_env.action_masks()

            # I check that some actions are masked but not all
            intraday_mask = mask[10:21]
            mfrr_mask = mask[21:32]
            assert np.any(intraday_mask)
            assert np.any(mfrr_mask)

    def test_soc_limits_masking(self, unified_env):
        """I verify SoC limits affect action masking."""
        unified_env.reset(seed=42)

        # I set SoC near minimum
        unified_env.soc = 0.08  # Near min_soc (0.05)

        # I set imbalance to deficit so mFRR UP (discharge) is allowed by direction
        # This isolates the SoC constraint from the direction constraint
        step = unified_env.current_step
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = -200.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('mfrr_price_up')] = 500.0

        mask = unified_env.action_masks()
        intraday_mask = mask[10:21]
        mfrr_mask = mask[21:32]

        # Max discharge (index 10) should be masked in both markets due to low SoC
        assert intraday_mask[10] == False  # Max IntraDay discharge
        assert mfrr_mask[10] == False  # Max mFRR discharge

        # Charge should be valid
        assert intraday_mask[0] == True or intraday_mask[1] == True


class TestAFRRSimulation:
    """I test aFRR activation simulation."""

    def test_afrr_activation_possible(self, unified_env):
        """I verify aFRR activation can occur."""
        unified_env.reset(seed=42)

        activations = 0
        for _ in range(100):
            action = np.array([4, 2, 5, 5])  # Max aFRR, neutral price, idle ID, idle mFRR
            obs, reward, done, truncated, info = unified_env.step(action)

            if info.get('afrr_activated', False):
                activations += 1

            if done or truncated:
                unified_env.reset(seed=42)

        # I expect some activations (15% base rate)
        assert activations > 0

    def test_afrr_direction(self, unified_env):
        """I verify aFRR activation has direction."""
        unified_env.reset(seed=42)

        for _ in range(200):
            action = np.array([4, 2, 5, 5])  # Max aFRR commitment
            obs, reward, done, truncated, info = unified_env.step(action)

            if info.get('afrr_activated', False):
                direction = info.get('afrr_direction')
                assert direction in ['up', 'down']
                break

            if done or truncated:
                unified_env.reset(seed=42)


class TestRewardCalculation:
    """I test reward calculation."""

    def test_reward_finite(self, unified_env):
        """I verify rewards are finite."""
        unified_env.reset(seed=42)

        for _ in range(50):
            action = unified_env.action_space.sample()
            obs, reward, done, truncated, info = unified_env.step(action)

            assert np.isfinite(reward)

            if done or truncated:
                break

    def test_afrr_capacity_revenue(self, unified_env):
        """I verify aFRR capacity revenue when selected."""
        unified_env.reset(seed=42)

        for _ in range(50):
            action = np.array([4, 2, 5, 5])  # Max aFRR, neutral, idle, idle
            obs, reward, done, truncated, info = unified_env.step(action)

            if info.get('is_selected', False):
                # I should get capacity revenue
                assert info.get('afrr_capacity_revenue', 0) > 0
                break

            if done or truncated:
                unified_env.reset(seed=42)

    def test_dam_violation_penalty(self, unified_env, sample_data):
        """I verify DAM violations incur penalty."""
        unified_env.reset(seed=42)

        for _ in range(100):
            # I try to not meet the commitment by charging instead
            action = np.array([0, 2, 0, 0])  # No aFRR, max charge ID, max charge mFRR
            obs, reward, done, truncated, info = unified_env.step(action)

            dam_shortfall = info.get('dam_shortfall_mw', 0)
            if dam_shortfall > 0.1:
                dam_violation_cost = info.get('dam_violation_cost', 0)
                assert dam_violation_cost > 0
                break

            if done or truncated:
                unified_env.reset(seed=42)


class TestEpisodeFlow:
    """I test episode flow."""

    def test_episode_terminates(self, unified_env):
        """I verify episode terminates correctly."""
        obs, info = unified_env.reset(seed=42)

        steps = 0
        max_steps = 1000

        while steps < max_steps:
            action = np.array([0, 2, 5, 5])  # Idle
            obs, reward, done, truncated, info = unified_env.step(action)
            steps += 1

            if done or truncated:
                break

        assert done or truncated

    def test_soc_bounds_maintained(self, unified_env):
        """I verify SoC stays within bounds."""
        unified_env.reset(seed=42)

        for _ in range(100):
            action = unified_env.action_space.sample()
            obs, reward, done, truncated, info = unified_env.step(action)

            soc = info.get('soc', obs[0])
            assert 0.05 <= soc <= 0.95

            if done or truncated:
                unified_env.reset(seed=42)

    def test_info_contains_profits(self, unified_env):
        """I verify info contains profit tracking including IntraDay."""
        unified_env.reset(seed=42)

        action = np.array([2, 2, 8, 5])  # 50% aFRR, neutral, moderate ID discharge, idle mFRR
        obs, reward, done, truncated, info = unified_env.step(action)

        # I check required info keys
        assert 'soc' in info
        assert 'afrr_commitment' in info
        assert 'total_profit' in info
        assert 'intraday_profit' in info
        assert 'intraday_energy_mw' in info
        assert 'mfrr_energy_mw' in info


class TestUnifiedRewardCalculator:
    """I test the unified reward calculator."""

    def test_import(self):
        """I verify the reward calculator imports correctly."""
        from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState

    def test_reward_components(self):
        """I verify reward has expected components including intraday_revenue."""
        from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState

        calculator = UnifiedRewardCalculator()

        market = UnifiedMarketState(
            dam_price=100.0,
            dam_commitment=10.0,
            intraday_bid=98.0,
            intraday_ask=102.0,
            intraday_spread=4.0,
            afrr_cap_up_price=20.0,
            afrr_cap_down_price=30.0,
            afrr_energy_up_price=80.0,
            afrr_energy_down_price=80.0,
            afrr_activated=False,
            afrr_activation_direction=None,
            mfrr_price_up=120.0,
            mfrr_price_down=60.0
        )

        result = calculator.calculate(
            market=market,
            actual_energy_mw=10.0,
            afrr_capacity_committed_mw=5.0,
            afrr_energy_delivered_mw=0.0,
            intraday_energy_mw=5.0,
            mfrr_energy_mw=0.0,
            current_soc=0.5,
            capacity_mwh=146.0,
            time_step_hours=1.0,
            is_selected_for_afrr=True
        )

        assert 'reward' in result
        assert 'components' in result
        assert 'intraday_revenue' in result['components']
        assert isinstance(result['reward'], (int, float))
        assert np.isfinite(result['reward'])

    def test_intraday_revenue_sell(self):
        """I verify IntraDay sell revenue uses bid price."""
        from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState

        calculator = UnifiedRewardCalculator()
        market = UnifiedMarketState(
            dam_price=100.0, dam_commitment=0.0,
            intraday_bid=98.0, intraday_ask=102.0, intraday_spread=4.0,
            afrr_cap_up_price=20.0, afrr_cap_down_price=30.0,
            afrr_energy_up_price=80.0, afrr_energy_down_price=80.0,
            afrr_activated=False, afrr_activation_direction=None,
            mfrr_price_up=120.0, mfrr_price_down=60.0
        )

        result = calculator.calculate(
            market=market, actual_energy_mw=10.0,
            afrr_capacity_committed_mw=0.0, afrr_energy_delivered_mw=0.0,
            intraday_energy_mw=10.0, mfrr_energy_mw=0.0,
            current_soc=0.5, capacity_mwh=146.0
        )

        # Sell 10 MW at bid=98 -> 980 EUR
        assert result['components']['intraday_revenue'] == pytest.approx(980.0, abs=1.0)

    def test_intraday_revenue_buy(self):
        """I verify IntraDay buy cost uses ask price."""
        from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState

        calculator = UnifiedRewardCalculator()
        market = UnifiedMarketState(
            dam_price=100.0, dam_commitment=0.0,
            intraday_bid=98.0, intraday_ask=102.0, intraday_spread=4.0,
            afrr_cap_up_price=20.0, afrr_cap_down_price=30.0,
            afrr_energy_up_price=80.0, afrr_energy_down_price=80.0,
            afrr_activated=False, afrr_activation_direction=None,
            mfrr_price_up=120.0, mfrr_price_down=60.0
        )

        result = calculator.calculate(
            market=market, actual_energy_mw=-10.0,
            afrr_capacity_committed_mw=0.0, afrr_energy_delivered_mw=0.0,
            intraday_energy_mw=-10.0, mfrr_energy_mw=0.0,
            current_soc=0.5, capacity_mwh=146.0
        )

        # Buy 10 MW at ask=102 -> -1020 EUR (cost)
        assert result['components']['intraday_revenue'] == pytest.approx(-1020.0, abs=1.0)


class TestDAMSoCReservation:
    """I test the SoC-aware action masking that prevents DAM violations."""

    def test_dam_soc_reserve_no_commitment(self, unified_env):
        """I verify that without DAM commitments, reserves stay at defaults."""
        unified_env.reset(seed=42)

        # I set all future DAM commitments to 0
        step = unified_env.current_step
        for i in range(1, 5):
            if step + i < len(unified_env.df):
                unified_env.df.iloc[step + i, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        min_soc, max_soc = unified_env._calculate_dam_soc_reserve()
        assert min_soc == pytest.approx(unified_env.min_soc, abs=1e-6)
        assert max_soc == pytest.approx(unified_env.max_soc, abs=1e-6)

    def test_dam_soc_reserve_sell_commitment(self, unified_env):
        """I verify that a sell commitment raises the minimum SoC reserve."""
        unified_env.reset(seed=42)

        step = unified_env.current_step
        # I clear all future commitments first
        for i in range(1, 5):
            if step + i < len(unified_env.df):
                unified_env.df.iloc[step + i, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I set a 30 MW sell commitment at H+1
        if step + 1 < len(unified_env.df):
            unified_env.df.iloc[step + 1, unified_env.df.columns.get_loc('dam_commitment')] = 30.0

        min_soc, max_soc = unified_env._calculate_dam_soc_reserve()

        # min_soc should be higher than default
        assert min_soc > unified_env.min_soc + 0.01
        # max_soc should stay at default
        assert max_soc == pytest.approx(unified_env.max_soc, abs=1e-6)

        expected_min_soc = unified_env.min_soc + (30.0 / unified_env.eff_sqrt) / unified_env.capacity_mwh
        assert min_soc == pytest.approx(expected_min_soc, abs=0.01)

    def test_dam_mask_prevents_over_discharge(self, unified_env):
        """I verify the mask blocks discharge when SoC is near the DAM reserve."""
        unified_env.reset(seed=42)

        step = unified_env.current_step
        for i in range(1, 5):
            if step + i < len(unified_env.df):
                unified_env.df.iloc[step + i, unified_env.df.columns.get_loc('dam_commitment')] = 0.0
        if step + 1 < len(unified_env.df):
            unified_env.df.iloc[step + 1, unified_env.df.columns.get_loc('dam_commitment')] = 30.0

        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        min_soc_reserve, _ = unified_env._calculate_dam_soc_reserve()
        unified_env.soc = min_soc_reserve + 0.01

        mask = unified_env.action_masks()
        intraday_mask = mask[10:21]  # IntraDay section
        mfrr_mask = mask[21:32]  # mFRR section

        # Max discharge should be blocked in both markets
        assert intraday_mask[10] == False, "Max IntraDay discharge should be masked near DAM reserve"
        assert mfrr_mask[10] == False, "Max mFRR discharge should be masked near DAM reserve"
        # Idle should always be allowed
        assert intraday_mask[5] == True, "IntraDay idle should always be allowed"
        assert mfrr_mask[5] == True, "mFRR idle should always be allowed"
        # Charge should be allowed
        assert intraday_mask[0] == True, "Max IntraDay charge should be allowed"

    def test_dam_mask_buy_commitment(self, unified_env):
        """I verify the mask blocks excessive charge when a buy commitment is ahead."""
        unified_env.reset(seed=42)

        step = unified_env.current_step
        for i in range(1, 5):
            if step + i < len(unified_env.df):
                unified_env.df.iloc[step + i, unified_env.df.columns.get_loc('dam_commitment')] = 0.0
        if step + 1 < len(unified_env.df):
            unified_env.df.iloc[step + 1, unified_env.df.columns.get_loc('dam_commitment')] = -30.0

        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        _, max_soc_reserve = unified_env._calculate_dam_soc_reserve()
        unified_env.soc = max_soc_reserve - 0.01

        mask = unified_env.action_masks()
        intraday_mask = mask[10:21]
        mfrr_mask = mask[21:32]

        # Max charge should be blocked
        assert intraday_mask[0] == False, "Max IntraDay charge should be masked near buy-reserve ceiling"
        assert mfrr_mask[0] == False, "Max mFRR charge should be masked near buy-reserve ceiling"
        # Idle should always be allowed
        assert intraday_mask[5] == True
        assert mfrr_mask[5] == True
        # Discharge should be allowed
        assert intraday_mask[10] == True, "IntraDay discharge should be allowed with high SoC"


class TestCapacityCascading:
    """I test that IntraDay and mFRR properly cascade capacity."""

    def test_intraday_reduces_mfrr_capacity(self, unified_env):
        """I verify that IntraDay trading reduces available mFRR capacity."""
        unified_env.reset(seed=42)
        unified_env.soc = 0.5
        # I clear DAM commitment for this step
        step = unified_env.current_step
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I request full IntraDay discharge + full mFRR discharge
        # [0 aFRR, neutral price, max ID discharge (10), max mFRR discharge (10)]
        action = np.array([0, 2, 10, 10])
        obs, reward, done, truncated, info = unified_env.step(action)

        id_mw = abs(info.get('intraday_energy_mw', 0))
        mfrr_mw = abs(info.get('mfrr_energy_mw', 0))
        total_mw = id_mw + mfrr_mw

        # I verify combined power does not exceed max_power_mw
        assert total_mw <= unified_env.max_power_mw + 0.1, \
            f"|IntraDay| + |mFRR| = {total_mw:.1f} exceeds max_power {unified_env.max_power_mw}"

    def test_cross_direction_trades(self, unified_env):
        """I verify buy ID + sell mFRR = valid (cross-direction)."""
        unified_env.reset(seed=42)
        unified_env.soc = 0.5
        step = unified_env.current_step
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I buy on IntraDay (action 0 = max charge), sell on mFRR (action 10 = max discharge)
        action = np.array([0, 2, 0, 10])
        obs, reward, done, truncated, info = unified_env.step(action)

        id_mw = info.get('intraday_energy_mw', 0)
        mfrr_mw = info.get('mfrr_energy_mw', 0)

        # I verify both trades happened (they have opposite signs)
        # IntraDay charge is negative, mFRR discharge is positive
        if abs(id_mw) > 0.1 and abs(mfrr_mw) > 0.1:
            assert id_mw < 0, "IntraDay should be charging (negative)"
            assert mfrr_mw > 0, "mFRR should be discharging (positive)"


class TestSeparateRevenue:
    """I test that IntraDay and mFRR track revenue separately."""

    def test_separate_profit_tracking(self, unified_env):
        """I verify IntraDay and mFRR have separate profit counters."""
        unified_env.reset(seed=42)
        step = unified_env.current_step
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I discharge on IntraDay only (mFRR idle)
        action = np.array([0, 2, 9, 5])  # ~80% ID discharge, mFRR idle
        obs, reward, done, truncated, info = unified_env.step(action)

        id_profit = info.get('intraday_profit', 0)
        mfrr_rev = info.get('mfrr_revenue', 0)

        if abs(info.get('intraday_energy_mw', 0)) > 0.1:
            assert abs(id_profit) > 0, "IntraDay profit should be non-zero when trading"
        assert mfrr_rev == pytest.approx(0.0, abs=0.01), "mFRR revenue should be zero when idle"

    def test_intraday_uses_bid_ask(self, unified_env):
        """I verify IntraDay sells at bid and buys at ask."""
        unified_env.reset(seed=42)
        unified_env.soc = 0.5
        step = unified_env.current_step
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I sell on IntraDay
        action = np.array([0, 2, 9, 5])  # ~80% ID discharge, mFRR idle
        obs, reward, done, truncated, info = unified_env.step(action)

        id_revenue = info.get('intraday_revenue', 0)
        id_mw = info.get('intraday_energy_mw', 0)

        if id_mw > 0.1:
            # Selling should generate positive revenue
            assert id_revenue > 0, "IntraDay sell should generate positive revenue"

    def test_mfrr_uses_up_down(self, unified_env):
        """I verify mFRR sells at mfrr_up and buys at mfrr_down."""
        unified_env.reset(seed=42)
        unified_env.soc = 0.5
        step = unified_env.current_step
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I sell on mFRR (IntraDay idle)
        action = np.array([0, 2, 5, 9])  # ID idle, ~80% mFRR discharge
        obs, reward, done, truncated, info = unified_env.step(action)

        mfrr_rev = info.get('mfrr_revenue', 0)
        mfrr_mw = info.get('mfrr_energy_mw', 0)

        if mfrr_mw > 0.1:
            assert mfrr_rev > 0, "mFRR sell should generate positive revenue"


class TestMFRRDirectionConstraint:
    """I test that mFRR actions are constrained by system imbalance direction."""

    def test_mfrr_idle_only_when_balanced(self, unified_env):
        """I verify all non-idle mFRR actions are masked when |imbalance| < threshold."""
        unified_env.reset(seed=42)
        step = unified_env.current_step

        # I set balanced imbalance (within threshold)
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 10.0

        mask = unified_env.action_masks()
        mfrr_mask = mask[21:32]  # mFRR section

        # Only idle (index 5) should be allowed
        for i in range(11):
            if i == 5:
                assert mfrr_mask[i] == True, "mFRR idle must always be allowed"
            else:
                assert mfrr_mask[i] == False, \
                    f"mFRR action {i} should be masked when balanced (imb=10 MW)"

    def test_mfrr_discharge_only_during_deficit(self, unified_env):
        """I verify only discharge (level > 0) actions allowed during deficit."""
        unified_env.reset(seed=42)
        unified_env.soc = 0.5  # Enough SoC for discharge
        step = unified_env.current_step

        # I set deficit imbalance and valid price
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = -200.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('mfrr_price_up')] = 500.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        mask = unified_env.action_masks()
        mfrr_mask = mask[21:32]

        # Idle (5) must be allowed
        assert mfrr_mask[5] == True

        # Discharge levels (6-10, level > 0) should be allowed (SoC permitting)
        has_discharge = any(mfrr_mask[6:11])
        assert has_discharge, "At least one discharge action should be allowed during deficit"

        # Charge levels (0-4, level < 0) must be masked
        for i in range(5):
            assert mfrr_mask[i] == False, \
                f"mFRR charge action {i} should be masked during deficit"

    def test_mfrr_charge_only_during_surplus(self, unified_env):
        """I verify only charge (level < 0) actions allowed during surplus."""
        unified_env.reset(seed=42)
        unified_env.soc = 0.5  # Enough headroom for charge
        step = unified_env.current_step

        # I set surplus imbalance and valid price
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 200.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('mfrr_price_down')] = 300.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        mask = unified_env.action_masks()
        mfrr_mask = mask[21:32]

        # Idle (5) must be allowed
        assert mfrr_mask[5] == True

        # Charge levels (0-4, level < 0) should be allowed (SoC permitting)
        has_charge = any(mfrr_mask[0:5])
        assert has_charge, "At least one charge action should be allowed during surplus"

        # Discharge levels (6-10, level > 0) must be masked
        for i in range(6, 11):
            assert mfrr_mask[i] == False, \
                f"mFRR discharge action {i} should be masked during surplus"

    def test_mfrr_direction_feature_in_observation(self, unified_env):
        """I verify obs[58] encodes mFRR direction: +1 (UP), -1 (DOWN), 0 (closed)."""
        unified_env.reset(seed=42)
        step = unified_env.current_step

        # I test deficit → +1.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = -200.0
        obs = unified_env._build_observation()
        assert obs[58] == 1.0, f"Expected mfrr_direction=+1.0 for deficit, got {obs[58]}"

        # I test surplus → -1.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 200.0
        obs = unified_env._build_observation()
        assert obs[58] == -1.0, f"Expected mfrr_direction=-1.0 for surplus, got {obs[58]}"

        # I test balanced → 0.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 10.0
        obs = unified_env._build_observation()
        assert obs[58] == 0.0, f"Expected mfrr_direction=0.0 for balanced, got {obs[58]}"

    def test_step_enforces_direction_constraint(self, unified_env):
        """I verify step() blocks wrong-direction mFRR trades (defense-in-depth)."""
        unified_env.reset(seed=42)
        unified_env.soc = 0.5
        step = unified_env.current_step

        # I set balanced imbalance — mFRR should be blocked entirely
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 5.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I force a max discharge mFRR action (action 10) despite balanced conditions
        action = np.array([0, 2, 5, 10])  # No aFRR, idle ID, max mFRR discharge
        obs, reward, done, truncated, info = unified_env.step(action)

        # mFRR should be zero because of direction enforcement
        assert abs(info.get('mfrr_energy_mw', 0)) < 0.01, \
            f"mFRR should be blocked during balanced conditions, got {info.get('mfrr_energy_mw', 0)}"


class TestUnifiedStrategy:
    """I test the unified strategy."""

    def test_import(self):
        """I verify strategy imports correctly."""
        from agent.unified_strategy import UnifiedStrategy, UnifiedRuleBasedStrategy

    def test_rule_based_action(self, unified_env):
        """I verify rule-based strategy produces valid 4-element actions."""
        from agent.unified_strategy import UnifiedRuleBasedStrategy

        strategy = UnifiedRuleBasedStrategy()
        strategy.load()

        unified_env.reset(seed=42)
        obs = unified_env._build_observation()
        mask = unified_env.action_masks()

        action = strategy.predict_action(obs, mask)

        assert len(action) == 4
        assert 0 <= action[0] < 5   # aFRR
        assert 0 <= action[1] < 5   # Price tier
        assert 0 <= action[2] < 11  # IntraDay
        assert 0 <= action[3] < 11  # mFRR

    def test_conservative_action(self, unified_env):
        """I verify conservative strategy produces minimal 4-element actions."""
        from agent.unified_strategy import UnifiedConservativeStrategy

        strategy = UnifiedConservativeStrategy()
        strategy.load()

        unified_env.reset(seed=42)
        obs = unified_env._build_observation()
        mask = unified_env.action_masks()

        action = strategy.predict_action(obs, mask)

        assert len(action) == 4
        # Conservative should prefer zero aFRR and idle energy
        assert action[0] == 0  # Zero aFRR commitment
        assert action[2] == 5  # Idle IntraDay action
        assert action[3] == 5  # Idle mFRR action


class TestMarketPhaseFeatures:
    """I test the new market phase features that replaced gate closure."""

    def test_ida3_correction_before_noon(self, unified_env):
        """I verify obs[55] = 0.0 when hour < 12 (IDA3 not yet applicable)."""
        unified_env.reset(seed=42)
        step = unified_env.current_step

        # I find a step with hour < 12
        for i in range(step, min(step + 24, len(unified_env.df))):
            ts = unified_env.df.index[i]
            if ts.hour < 12:
                unified_env.current_step = i
                obs = unified_env._build_observation()
                assert obs[55] == 0.0, \
                    f"Expected ida3_applies=0.0 for hour {ts.hour}, got {obs[55]}"
                return

        pytest.skip("No hour < 12 found in test window")

    def test_ida3_correction_after_noon(self, unified_env):
        """I verify obs[55] = 1.0 when hour >= 12 (IDA3 prices available)."""
        unified_env.reset(seed=42)
        step = unified_env.current_step

        # I find a step with hour >= 12
        for i in range(step, min(step + 24, len(unified_env.df))):
            ts = unified_env.df.index[i]
            if ts.hour >= 12:
                unified_env.current_step = i
                obs = unified_env._build_observation()
                assert obs[55] == 1.0, \
                    f"Expected ida3_applies=1.0 for hour {ts.hour}, got {obs[55]}"
                return

        pytest.skip("No hour >= 12 found in test window")

    def test_system_stress_feature(self, unified_env):
        """I verify obs[56] scales with |imbalance| magnitude."""
        unified_env.reset(seed=42)
        step = unified_env.current_step

        # I test zero imbalance → stress = 0.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 0.0
        obs = unified_env._build_observation()
        assert obs[56] == pytest.approx(0.0, abs=0.01), \
            f"Expected system_stress=0.0 for zero imbalance, got {obs[56]}"

        # I test 150 MW imbalance → stress = 0.5
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 150.0
        obs = unified_env._build_observation()
        assert obs[56] == pytest.approx(0.5, abs=0.01), \
            f"Expected system_stress=0.5 for 150 MW, got {obs[56]}"

        # I test 600 MW imbalance → stress = 1.0 (capped)
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = 600.0
        obs = unified_env._build_observation()
        assert obs[56] == pytest.approx(1.0, abs=0.01), \
            f"Expected system_stress=1.0 for 600 MW, got {obs[56]}"

        # I test negative imbalance → stress uses abs()
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('net_imbalance_mw')] = -300.0
        obs = unified_env._build_observation()
        assert obs[56] == pytest.approx(1.0, abs=0.01), \
            f"Expected system_stress=1.0 for -300 MW, got {obs[56]}"

    def test_res_penetration_feature(self, unified_env):
        """I verify obs[57] reflects res_total/load ratio."""
        unified_env.reset(seed=42)
        step = unified_env.current_step

        # I set known RES and load values
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('res_total_mw')] = 3000.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('load_mw')] = 6000.0
        obs = unified_env._build_observation()
        # res_penetration = clip(3000/6000, 0, 1.5) / 1.5 = 0.5 / 1.5 = 0.333
        expected = (3000.0 / 6000.0) / 1.5
        assert obs[57] == pytest.approx(expected, abs=0.01), \
            f"Expected res_penetration={expected:.3f}, got {obs[57]}"

        # I test high RES (above 100% of load) → capped at 1.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('res_total_mw')] = 10000.0
        unified_env.df.iloc[step, unified_env.df.columns.get_loc('load_mw')] = 5000.0
        obs = unified_env._build_observation()
        # res_penetration = clip(10000/5000, 0, 1.5) / 1.5 = 1.5 / 1.5 = 1.0
        assert obs[57] == pytest.approx(1.0, abs=0.01), \
            f"Expected res_penetration=1.0 for high RES, got {obs[57]}"


class TestAFRRMarginalPricing:
    """I test that aFRR capacity profit uses marginal pricing."""

    def test_capacity_profit_independent_of_price_tier(self, unified_env):
        """I verify that per-selection aFRR capacity revenue does NOT vary with price_tier."""
        # I collect per-step revenue when selected for two different price tiers.
        # With marginal pricing, the revenue per MW should be identical
        # regardless of the bid tier — only selection probability differs.
        revenues_by_tier = {}

        for price_tier_action in [0, 4]:  # Most aggressive vs most conservative
            unified_env.reset(seed=42)
            step = unified_env.current_step
            row = unified_env.df.iloc[step]

            # I clear DAM to get full capacity
            unified_env.df.iloc[step, unified_env.df.columns.get_loc('dam_commitment')] = 0.0

            # I directly test the revenue formula by forcing selection
            unified_env.afrr_capacity_profit = 0.0
            unified_env.is_selected_for_afrr = False

            # I set a known aFRR capacity price
            unified_env.df.iloc[step, unified_env.df.columns.get_loc('afrr_cap_up_price')] = 25.0

            # I force selection by temporarily making selection probability 1.0
            old_prob = unified_env.selection_probability
            unified_env.selection_probability = 1.0

            action = np.array([4, price_tier_action, 5, 5])  # Max aFRR, vary tier, idle
            obs, reward, done, truncated, info = unified_env.step(action)

            unified_env.selection_probability = old_prob

            if info.get('is_selected', False):
                revenues_by_tier[price_tier_action] = unified_env.afrr_capacity_profit

        # I verify both tiers were selected and got identical revenue
        assert len(revenues_by_tier) == 2, \
            f"Expected both tiers to be selected, got {list(revenues_by_tier.keys())}"

        assert revenues_by_tier[0] == pytest.approx(revenues_by_tier[4], abs=0.01), \
            f"Capacity revenue should be identical with marginal pricing: " \
            f"tier 0={revenues_by_tier[0]:.2f}, tier 4={revenues_by_tier[4]:.2f}"


class Test15MinResolution:
    """I test the 15-minute ISP resolution support."""

    @pytest.fixture
    def sample_data_15min(self):
        """I create 15-min sample data (1 week = 672 slots)."""
        n_slots = 672  # 7 days × 96 slots/day
        dates = pd.date_range(
            start='2024-01-01',
            periods=n_slots,
            freq='15T',
            tz='Europe/Athens'
        )

        np.random.seed(42)

        hours = np.array([d.hour for d in dates])
        base_price = 80 + 40 * np.sin(2 * np.pi * hours / 24)
        noise = np.random.normal(0, 10, n_slots)
        prices = base_price + noise

        # I create DAM commitments (same value for all 4 quarter-hours within an hour)
        dam_commitments = np.zeros(n_slots)
        for i in range(n_slots):
            if hours[i] in [18, 19, 20]:
                dam_commitments[i] = np.random.uniform(10, 25)
            elif hours[i] in [10, 11, 12]:
                dam_commitments[i] = np.random.uniform(-20, -5)

        data = {
            'price': prices,
            'dam_commitment': dam_commitments,
            'afrr_up': prices * 1.1,
            'afrr_down': prices * 0.9,
            'afrr_cap_up_price': 15 + 10 * np.random.random(n_slots),
            'afrr_cap_down_price': 25 + 15 * np.random.random(n_slots),
            'intraday_bid': prices - 2.0 + np.random.normal(0, 1, n_slots),
            'intraday_ask': prices + 2.0 + np.random.normal(0, 1, n_slots),
            'intraday_spread': 3.0 + np.random.uniform(0, 4, n_slots),
            'intraday_volume': np.random.uniform(0.3, 0.9, n_slots),
            'mfrr_price_up': prices * 1.3,
            'mfrr_price_down': prices * 0.7,
            'mfrr_spread': prices * 0.6,
            'net_imbalance_mw': np.random.normal(0, 300, n_slots),
            'solar': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)),
            'wind_onshore': np.random.uniform(100, 500, n_slots),
            'res_total_mw': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)) + np.random.uniform(100, 500, n_slots),
            'load_mw': np.random.uniform(3000, 6000, n_slots)
        }

        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def env_15min(self, sample_data_15min):
        """I create a 15-min resolution environment for testing."""
        from gym_envs.battery_env_unified import BatteryEnvUnified
        env = BatteryEnvUnified(
            df=sample_data_15min,
            capacity_mwh=146.0,
            max_power_mw=30.0,
            efficiency=0.94,
            time_step_hours=0.25,
            episode_length=288,  # 3 days × 96 slots/day
            random_start=False
        )
        return env

    def test_obs_shape_15min(self, env_15min):
        """I verify observation space is still 63 features at 15-min resolution."""
        assert env_15min.observation_space.shape == (63,)
        obs, info = env_15min.reset(seed=42)
        assert obs.shape == (63,)
        assert not np.any(np.isnan(obs))

    def test_sph_value(self, env_15min):
        """I verify _sph is 4 for 15-min resolution."""
        assert env_15min._sph == 4

    def test_dam_lookahead_hourly_sampling(self, env_15min):
        """I verify DAM lookahead obs[19] matches price 1 HOUR ahead (not 15 min)."""
        env_15min.reset(seed=42)
        step = env_15min.current_step

        obs = env_15min._build_observation()

        # obs[19] is the first DAM lookahead (index 18 = current DAM, 19 = +1h)
        # I verify it samples from step + 1*_sph (= step + 4), not step + 1
        target_step = min(step + 4, len(env_15min.df) - 1)
        target_ts = env_15min.df.index[target_step]
        current_ts = env_15min.df.index[step]

        if target_ts.date() == current_ts.date():
            expected_dam = np.clip(
                env_15min.df.iloc[target_step].get('dam_commitment', 0.0),
                -env_15min.max_power_mw, env_15min.max_power_mw
            )
            assert obs[19] == pytest.approx(expected_dam / env_15min.max_power_mw, abs=0.001)

    def test_time_encoding_discriminates_quarters(self, env_15min):
        """I verify obs[14] differs between HH:00, HH:15, HH:30, HH:45."""
        env_15min.reset(seed=42)
        step = env_15min.current_step

        # I find 4 consecutive steps within the same hour
        ts = env_15min.df.index[step]
        # I find a step at HH:00
        while ts.minute != 0 and step < len(env_15min.df) - 5:
            step += 1
            ts = env_15min.df.index[step]

        if ts.minute != 0:
            pytest.skip("Could not find HH:00 step in range")

        sin_values = []
        for offset in range(4):
            env_15min.current_step = step + offset
            obs = env_15min._build_observation()
            sin_values.append(obs[14])  # hour_sin

        # I verify all 4 quarter-hour sin values are distinct
        for i in range(len(sin_values)):
            for j in range(i + 1, len(sin_values)):
                assert sin_values[i] != pytest.approx(sin_values[j], abs=1e-4), \
                    f"Time encoding must discriminate quarters: {sin_values}"

    def test_4x_more_steps_per_day(self, env_15min):
        """I verify 96 steps completes exactly 1 day at 15-min resolution."""
        env_15min.reset(seed=42)
        start_step = env_15min.current_step
        start_ts = env_15min.df.index[start_step]

        # I advance 96 steps (= 24 hours at 15-min)
        target_step = start_step + 96
        if target_step < len(env_15min.df):
            target_ts = env_15min.df.index[target_step]
            hours_elapsed = (target_ts - start_ts).total_seconds() / 3600
            assert hours_elapsed == pytest.approx(24.0, abs=0.1), \
                f"96 steps should be 24h, got {hours_elapsed:.1f}h"

    def test_backward_compat_hourly(self, sample_data):
        """I verify env with time_step_hours=1.0 still works identically."""
        from gym_envs.battery_env_unified import BatteryEnvUnified

        env_1h = BatteryEnvUnified(
            df=sample_data,
            capacity_mwh=146.0,
            max_power_mw=30.0,
            efficiency=0.94,
            time_step_hours=1.0,
            episode_length=72,
            random_start=False
        )

        assert env_1h._sph == 1
        assert env_1h.observation_space.shape == (63,)

        obs, info = env_1h.reset(seed=42)
        assert obs.shape == (63,)

        # I verify a full episode runs without errors
        for _ in range(50):
            action = np.array([0, 2, 5, 5])
            obs, reward, done, truncated, info = env_1h.step(action)
            assert np.isfinite(reward)
            if done or truncated:
                break

    def test_episode_runs_full(self, env_15min):
        """I verify a full 15-min episode runs without errors."""
        obs, info = env_15min.reset(seed=42)
        steps = 0
        for _ in range(300):
            action = np.array([0, 2, 5, 5])  # Idle
            obs, reward, done, truncated, info = env_15min.step(action)
            steps += 1
            assert np.isfinite(reward)
            assert 0.05 <= info['soc'] <= 0.95
            if done or truncated:
                break
        assert steps > 0


# =========================================================================
# FULL MARKET MODE TESTS (IDA + XBID + Free Bids)
# =========================================================================

@pytest.fixture
def sample_data_full_market():
    """I create sample training data with IDA/XBID/FreeBid columns."""
    n_hours = 168
    dates = pd.date_range(
        start='2024-01-01',
        periods=n_hours,
        freq='H',
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
        'net_imbalance_mw': np.random.normal(0, 300, n_hours),
        'solar': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)),
        'wind_onshore': np.random.uniform(100, 500, n_hours),
        'res_total_mw': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)) + np.random.uniform(100, 500, n_hours),
        'load_mw': np.random.uniform(3000, 6000, n_hours),
        # IDA columns
        'ida1_clearing_price': prices + np.random.normal(0, 12, n_hours),
        'ida2_clearing_price': prices + np.random.normal(0, 8, n_hours),
        'ida3_clearing_price': np.where(hours >= 12, prices + np.random.normal(0, 4, n_hours), np.nan),
        'ida1_position': np.clip(np.random.normal(0, 2.0, n_hours), -5, 5),
        'ida2_position': np.clip(np.random.normal(0, 1.5, n_hours), -3, 3),
        'ida3_position': np.where(hours >= 12, np.clip(np.random.normal(0, 1.0, n_hours), -2, 2), 0.0),
        'net_ida_position': np.zeros(n_hours),  # I fill this below
        # XBID columns
        'xbid_bid': prices - 3.0 + np.random.normal(0, 1, n_hours),
        'xbid_ask': prices + 3.0 + np.random.normal(0, 1, n_hours),
        'xbid_spread': 4.0 + np.random.uniform(0, 4, n_hours),
        # Free Bid columns
        'free_bid_activation_base': np.clip(np.abs(np.random.normal(0, 300, n_hours)) / 500.0, 0, 0.8),
        'free_bid_reference_price': prices * 1.3,
    }

    df = pd.DataFrame(data, index=dates)
    df['net_ida_position'] = df['ida1_position'] + df['ida2_position'] + df['ida3_position']
    return df


@pytest.fixture
def full_market_env(sample_data_full_market):
    """I create a full-market-mode environment for testing."""
    from gym_envs.battery_env_unified import BatteryEnvUnified

    env = BatteryEnvUnified(
        df=sample_data_full_market,
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        episode_length=72,
        random_start=False,
        enable_full_market=True
    )
    return env


class TestIDADecomposition:
    """I test IDA sub-market decomposition."""

    def test_ida_columns_exist(self, sample_data_full_market):
        """I verify IDA position and price columns in training data."""
        df = sample_data_full_market
        for col in ['ida1_clearing_price', 'ida2_clearing_price', 'ida3_clearing_price',
                     'ida1_position', 'ida2_position', 'ida3_position', 'net_ida_position']:
            assert col in df.columns, f"Missing IDA column: {col}"

    def test_ida_position_execution(self, full_market_env):
        """I verify IDA positions execute as mandatory (like DAM)."""
        full_market_env.reset(seed=42)
        full_market_env.soc = 0.5

        # I run a few steps and check IDA execution
        for _ in range(20):
            action = np.array([0, 2, 5, 5, 2])  # All idle except IDA (auto-executed)
            obs, reward, done, truncated, info = full_market_env.step(action)

            ida_mw = info.get('ida_energy_mw', 0)
            # IDA positions should be executed when non-zero in data
            # I verify it was attempted (may be 0 if data position is 0)
            assert isinstance(ida_mw, (int, float))

            if done or truncated:
                break

    def test_ida_pay_as_cleared(self):
        """I verify IDA revenue uses clearing price."""
        from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState

        calculator = UnifiedRewardCalculator()
        market = UnifiedMarketState(
            dam_price=100.0, dam_commitment=0.0,
            intraday_bid=98.0, intraday_ask=102.0, intraday_spread=4.0,
            afrr_cap_up_price=20.0, afrr_cap_down_price=30.0,
            afrr_energy_up_price=80.0, afrr_energy_down_price=80.0,
            afrr_activated=False, afrr_activation_direction=None,
            mfrr_price_up=120.0, mfrr_price_down=60.0,
            ida1_clearing_price=95.0, ida2_clearing_price=97.0,
            ida3_clearing_price=99.0, net_ida_position=5.0,
        )

        result = calculator.calculate(
            market=market, actual_energy_mw=5.0,
            afrr_capacity_committed_mw=0.0, afrr_energy_delivered_mw=0.0,
            current_soc=0.5, capacity_mwh=146.0,
            ida_energy_mw=5.0
        )

        # IDA revenue should use weighted average of clearing prices
        assert result['components']['ida_revenue'] > 0

    def test_ida_phase_feature(self, full_market_env):
        """I verify obs[55] encodes IDA phase correctly."""
        full_market_env.reset(seed=42)
        step = full_market_env.current_step

        # I find a step with hour < 15
        for i in range(step, min(step + 24, len(full_market_env.df))):
            ts = full_market_env.df.index[i]
            if ts.hour < 15:
                full_market_env.current_step = i
                obs = full_market_env._build_observation()
                assert obs[55] == pytest.approx(0.0, abs=0.01), \
                    f"Expected ida_phase=0.0 for hour {ts.hour}, got {obs[55]}"
                return

        pytest.skip("No hour < 15 found in test window")

    def test_ida3_nan_morning(self, sample_data_full_market):
        """I verify IDA3 is NaN/0 for hours < 12."""
        df = sample_data_full_market
        morning = df[df.index.hour < 12]
        assert morning['ida3_clearing_price'].isna().all(), \
            "IDA3 clearing price should be NaN for hours < 12"

    def test_capacity_cascade_with_ida(self, full_market_env):
        """I verify DAM -> aFRR -> IDA -> XBID -> FreeBid cascade."""
        full_market_env.reset(seed=42)
        full_market_env.soc = 0.5
        step = full_market_env.current_step
        full_market_env.df.iloc[step, full_market_env.df.columns.get_loc('dam_commitment')] = 0.0

        # I set a non-zero IDA position
        full_market_env.df.iloc[step, full_market_env.df.columns.get_loc('net_ida_position')] = 5.0

        # I request max XBID discharge + max balancing discharge
        action = np.array([0, 2, 10, 10, 2])
        obs, reward, done, truncated, info = full_market_env.step(action)

        ida_mw = abs(info.get('ida_energy_mw', 0))
        xbid_mw = abs(info.get('xbid_energy_mw', 0))
        free_bid_mw = abs(info.get('free_bid_energy_mw', 0))
        total_mw = ida_mw + xbid_mw + free_bid_mw

        # I verify combined power does not exceed max_power_mw
        assert total_mw <= full_market_env.max_power_mw + 0.1, \
            f"IDA + XBID + FreeBid = {total_mw:.1f} exceeds max_power {full_market_env.max_power_mw}"


class TestFreeBids:
    """I test Free Bid (Balancing Energy Offers)."""

    def test_action_space_5dim(self, full_market_env):
        """I verify action space is MultiDiscrete([5,5,11,11,5])."""
        assert full_market_env.action_space.nvec.tolist() == [5, 5, 11, 11, 5]

    def test_mask_shape_37(self, full_market_env):
        """I verify mask length is 5+5+11+11+5=37."""
        full_market_env.reset(seed=42)
        mask = full_market_env.action_masks()
        assert len(mask) == 5 + 5 + 11 + 11 + 5  # 37 total

    def test_free_bid_price_sensitivity(self, full_market_env):
        """I verify lower bid prices -> higher activation probability."""
        full_market_env.reset(seed=42)
        row = full_market_env.df.iloc[full_market_env.current_step]

        # I test aggressive pricing (0.8x)
        activations_aggressive = 0
        for _ in range(1000):
            result = full_market_env._simulate_free_bid_activation(
                bid_price=80.0, ref_price=100.0, imbalance=-200.0, row=row
            )
            if result:
                activations_aggressive += 1

        # I test conservative pricing (1.2x)
        activations_conservative = 0
        for _ in range(1000):
            result = full_market_env._simulate_free_bid_activation(
                bid_price=120.0, ref_price=100.0, imbalance=-200.0, row=row
            )
            if result:
                activations_conservative += 1

        assert activations_aggressive > activations_conservative, \
            f"Aggressive ({activations_aggressive}) should activate more than conservative ({activations_conservative})"

    def test_free_bid_pay_as_bid(self):
        """I verify Free Bid revenue uses agent's price, not clearing."""
        from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState

        calculator = UnifiedRewardCalculator()
        agent_price = 95.0

        market = UnifiedMarketState(
            dam_price=100.0, dam_commitment=0.0,
            intraday_bid=98.0, intraday_ask=102.0, intraday_spread=4.0,
            afrr_cap_up_price=20.0, afrr_cap_down_price=30.0,
            afrr_energy_up_price=80.0, afrr_energy_down_price=80.0,
            afrr_activated=False, afrr_activation_direction=None,
            mfrr_price_up=120.0, mfrr_price_down=60.0,
            free_bid_activated=True, free_bid_agent_price=agent_price,
        )

        result = calculator.calculate(
            market=market, actual_energy_mw=10.0,
            afrr_capacity_committed_mw=0.0, afrr_energy_delivered_mw=0.0,
            current_soc=0.5, capacity_mwh=146.0,
            free_bid_energy_mw=10.0
        )

        # Free Bid sells 10 MW at agent_price=95 -> 950 EUR
        assert result['components']['free_bid_revenue'] == pytest.approx(950.0, abs=1.0)

    def test_direction_constraint(self, full_market_env):
        """I verify Free Bid respects imbalance direction."""
        full_market_env.reset(seed=42)
        step = full_market_env.current_step

        # I set balanced imbalance
        full_market_env.df.iloc[step, full_market_env.df.columns.get_loc('net_imbalance_mw')] = 5.0

        # I test activation with low imbalance — should not activate
        row = full_market_env.df.iloc[step]
        result = full_market_env._simulate_free_bid_activation(
            bid_price=90.0, ref_price=100.0, imbalance=5.0, row=row
        )
        assert result == False, "Free Bid should not activate when |imbalance| < 30 MW"

    def test_price_tier_masked_when_idle(self, full_market_env):
        """I verify price tier forced to neutral when no balancing need."""
        full_market_env.reset(seed=42)
        step = full_market_env.current_step

        # I set balanced imbalance
        full_market_env.df.iloc[step, full_market_env.df.columns.get_loc('net_imbalance_mw')] = 10.0

        mask = full_market_env.action_masks()
        # Free Bid price tier mask is last 5 elements
        freebid_mask = mask[-5:]

        # Only neutral (index 2) should be allowed
        assert freebid_mask[2] == True, "Neutral price tier (1.0x) must be allowed"
        for i in [0, 1, 3, 4]:
            assert freebid_mask[i] == False, \
                f"Price tier {i} should be masked when balanced (no balancing need)"


class TestBackwardCompat:
    """I verify legacy mode (enable_full_market=False) is identical."""

    def test_legacy_action_space_4dim(self, unified_env):
        """I verify 4-dim action space when flag is False."""
        assert unified_env.action_space.nvec.tolist() == [5, 5, 11, 11]

    def test_legacy_obs_63(self, unified_env):
        """I verify 63-feature observation when flag is False."""
        assert unified_env.observation_space.shape == (63,)
        obs, info = unified_env.reset(seed=42)
        assert obs.shape == (63,)

    def test_legacy_mask_32(self, unified_env):
        """I verify 32-element mask when flag is False."""
        unified_env.reset(seed=42)
        mask = unified_env.action_masks()
        assert len(mask) == 5 + 5 + 11 + 11  # 32

    def test_legacy_episode_runs(self, unified_env):
        """I verify legacy mode episode runs identically."""
        obs, info = unified_env.reset(seed=42)

        for _ in range(30):
            action = np.array([0, 2, 5, 5])  # Idle
            obs, reward, done, truncated, info = unified_env.step(action)
            assert obs.shape == (63,)
            assert np.isfinite(reward)
            if done or truncated:
                break


class TestSeparateRevenueTracking:
    """I test per-market revenue tracking."""

    def test_info_contains_all_markets(self, full_market_env):
        """I verify info has dam, ida, xbid, afrr, free_bid, mfrr profits."""
        full_market_env.reset(seed=42)
        action = np.array([0, 2, 5, 5, 2])
        obs, reward, done, truncated, info = full_market_env.step(action)

        for key in ['dam_profit', 'ida_profit', 'xbid_profit',
                     'afrr_capacity_profit', 'afrr_energy_profit',
                     'free_bid_profit']:
            assert key in info, f"Missing market profit key: {key}"

    def test_full_market_obs_shape(self, full_market_env):
        """I verify 75-feature observation in full market mode."""
        obs, info = full_market_env.reset(seed=42)
        assert obs.shape == (75,)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_full_market_step(self, full_market_env):
        """I verify full market mode step works correctly."""
        full_market_env.reset(seed=42)

        for _ in range(30):
            action = full_market_env.action_space.sample()
            obs, reward, done, truncated, info = full_market_env.step(action)
            assert obs.shape == (75,)
            assert np.isfinite(reward)
            assert 0.05 <= info['soc'] <= 0.95
            if done or truncated:
                break

    def test_xbid_revenue_tracked(self, full_market_env):
        """I verify XBID revenue is tracked separately from IntraDay."""
        full_market_env.reset(seed=42)
        full_market_env.soc = 0.5
        step = full_market_env.current_step
        full_market_env.df.iloc[step, full_market_env.df.columns.get_loc('dam_commitment')] = 0.0
        full_market_env.df.iloc[step, full_market_env.df.columns.get_loc('net_ida_position')] = 0.0

        # I discharge on XBID
        action = np.array([0, 2, 9, 5, 2])  # ~80% XBID discharge, idle balancing
        obs, reward, done, truncated, info = full_market_env.step(action)

        xbid_mw = info.get('xbid_energy_mw', 0)
        if abs(xbid_mw) > 0.1:
            assert full_market_env.xbid_profit != 0, \
                "XBID profit should be tracked when trading"


class TestFullMarketStrategy:
    """I test strategy with full market mode."""

    def test_rule_based_5dim_action(self, full_market_env):
        """I verify rule-based strategy produces 5-element actions in full market mode."""
        from agent.unified_strategy import UnifiedRuleBasedStrategy

        strategy = UnifiedRuleBasedStrategy(enable_full_market=True)
        strategy.load()

        full_market_env.reset(seed=42)
        obs = full_market_env._build_observation()
        mask = full_market_env.action_masks()

        action = strategy.predict_action(obs, mask)
        assert len(action) == 5
        assert 0 <= action[4] < 5  # Free Bid price tier

    def test_conservative_5dim_action(self, full_market_env):
        """I verify conservative strategy produces 5-element actions in full market mode."""
        from agent.unified_strategy import UnifiedConservativeStrategy

        strategy = UnifiedConservativeStrategy(enable_full_market=True)
        strategy.load()

        full_market_env.reset(seed=42)
        obs = full_market_env._build_observation()
        mask = full_market_env.action_masks()

        action = strategy.predict_action(obs, mask)
        assert len(action) == 5
        assert action[0] == 0   # Zero aFRR
        assert action[2] == 5   # Idle XBID
        assert action[3] == 5   # Idle balancing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
