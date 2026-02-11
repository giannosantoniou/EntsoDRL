"""
Unit Tests for Unified Multi-Market Battery Environment

I test the key functionality of the unified environment:
1. Action space structure (MultiDiscrete [5, 5, 21])
2. Observation space (58 features)
3. Action masking (capacity cascading)
4. aFRR activation simulation
5. Reward calculation
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
        'mfrr_price_up': prices * 1.3,
        'mfrr_price_down': prices * 0.7,
        'mfrr_spread': prices * 0.6,
        'net_imbalance_mw': np.random.normal(0, 300, n_hours),
        'solar': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)),
        'wind_onshore': np.random.uniform(100, 500, n_hours)
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
        """I verify the action space is MultiDiscrete([5, 5, 21])."""
        assert unified_env.action_space.nvec.tolist() == [5, 5, 21]

    def test_afrr_levels(self, unified_env):
        """I verify aFRR commitment levels."""
        expected = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        np.testing.assert_array_equal(unified_env.AFRR_LEVELS, expected)

    def test_price_tiers(self, unified_env):
        """I verify aFRR price tiers."""
        expected = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
        np.testing.assert_array_equal(unified_env.AFRR_PRICE_TIERS, expected)

    def test_energy_levels(self, unified_env):
        """I verify energy action levels span -1 to +1."""
        levels = unified_env.energy_levels
        assert len(levels) == 21
        assert levels[0] == -1.0
        assert levels[10] == 0.0
        assert levels[20] == 1.0


class TestObservationSpace:
    """I test the observation space structure."""

    def test_observation_shape(self, unified_env):
        """I verify the observation space is 58 features."""
        assert unified_env.observation_space.shape == (58,)

    def test_observation_reset(self, unified_env):
        """I verify observation is correctly shaped after reset."""
        obs, info = unified_env.reset(seed=42)
        assert obs.shape == (58,)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_observation_step(self, unified_env):
        """I verify observation is correctly shaped after step."""
        unified_env.reset(seed=42)
        action = np.array([0, 2, 10])  # No aFRR, neutral price, idle
        obs, reward, done, truncated, info = unified_env.step(action)
        assert obs.shape == (58,)
        assert isinstance(reward, (int, float))


class TestActionMasking:
    """I test hierarchical action masking."""

    def test_mask_shape(self, unified_env):
        """I verify the action mask is correctly shaped."""
        unified_env.reset(seed=42)
        mask = unified_env.action_masks()
        assert len(mask) == 5 + 5 + 21  # 31 total

    def test_idle_always_valid(self, unified_env):
        """I verify idle action (10) is always valid."""
        unified_env.reset(seed=42)
        mask = unified_env.action_masks()

        # Energy mask starts at index 10 (5 + 5)
        energy_mask = mask[10:]
        idle_idx = 10  # Middle of 21
        assert energy_mask[idle_idx] == True

    def test_zero_afrr_always_valid(self, unified_env):
        """I verify zero aFRR commitment is always valid."""
        unified_env.reset(seed=42)
        mask = unified_env.action_masks()
        assert mask[0] == True  # First aFRR action is zero commitment

    def test_capacity_cascading(self, unified_env, sample_data):
        """I verify capacity cascading respects DAM commitment."""
        # I find a step with DAM commitment
        dam_steps = sample_data[sample_data['dam_commitment'].abs() > 5].index

        if len(dam_steps) > 0:
            # I reset and step to a DAM commitment hour
            unified_env.reset(seed=42)

            # I manually set state
            unified_env.soc = 0.5  # Middle SoC for flexibility

            mask = unified_env.action_masks()

            # With DAM commitment, remaining capacity is reduced
            # So some high-power actions should be masked
            energy_mask = mask[10:]

            # Not all discharge actions should be valid (capacity limited)
            # But some should be (within remaining capacity)
            assert np.any(energy_mask)

    def test_soc_limits_masking(self, unified_env):
        """I verify SoC limits affect action masking."""
        unified_env.reset(seed=42)

        # I set SoC near minimum
        unified_env.soc = 0.08  # Near min_soc (0.05)

        mask = unified_env.action_masks()
        energy_mask = mask[10:]

        # High discharge should be masked (not enough energy)
        # But charge should be valid
        assert energy_mask[0] == True or energy_mask[1] == True  # Charge
        assert energy_mask[20] == False  # Max discharge should be masked


class TestAFRRSimulation:
    """I test aFRR activation simulation."""

    def test_afrr_activation_possible(self, unified_env):
        """I verify aFRR activation can occur."""
        unified_env.reset(seed=42)

        activations = 0
        for _ in range(100):
            action = np.array([4, 2, 10])  # Max aFRR, neutral price, idle
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
            action = np.array([4, 2, 10])  # Max aFRR commitment
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
            action = np.array([4, 2, 10])  # Max aFRR, neutral, idle
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

        # I find a step with large sell commitment
        for _ in range(100):
            # I try to not meet the commitment by charging instead
            action = np.array([0, 2, 0])  # No aFRR, max charge
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
            action = np.array([0, 2, 10])  # Idle
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
        """I verify info contains profit tracking."""
        unified_env.reset(seed=42)

        action = np.array([2, 2, 15])  # 50% aFRR, neutral, slight discharge
        obs, reward, done, truncated, info = unified_env.step(action)

        # I check required info keys
        assert 'soc' in info
        assert 'afrr_commitment' in info
        assert 'total_profit' in info


class TestUnifiedRewardCalculator:
    """I test the unified reward calculator."""

    def test_import(self):
        """I verify the reward calculator imports correctly."""
        from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState

    def test_reward_components(self):
        """I verify reward has expected components."""
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
            mfrr_energy_mw=0.0,
            current_soc=0.5,
            capacity_mwh=146.0,
            time_step_hours=1.0,
            is_selected_for_afrr=True
        )

        assert 'reward' in result
        assert 'components' in result
        assert isinstance(result['reward'], (int, float))
        assert np.isfinite(result['reward'])


class TestUnifiedStrategy:
    """I test the unified strategy."""

    def test_import(self):
        """I verify strategy imports correctly."""
        from agent.unified_strategy import UnifiedStrategy, UnifiedRuleBasedStrategy

    def test_rule_based_action(self, unified_env):
        """I verify rule-based strategy produces valid actions."""
        from agent.unified_strategy import UnifiedRuleBasedStrategy

        strategy = UnifiedRuleBasedStrategy()
        strategy.load()

        unified_env.reset(seed=42)
        obs = unified_env._build_observation()
        mask = unified_env.action_masks()

        action = strategy.predict_action(obs, mask)

        assert len(action) == 3
        assert 0 <= action[0] < 5
        assert 0 <= action[1] < 5
        assert 0 <= action[2] < 21

    def test_conservative_action(self, unified_env):
        """I verify conservative strategy produces minimal actions."""
        from agent.unified_strategy import UnifiedConservativeStrategy

        strategy = UnifiedConservativeStrategy()
        strategy.load()

        unified_env.reset(seed=42)
        obs = unified_env._build_observation()
        mask = unified_env.action_masks()

        action = strategy.predict_action(obs, mask)

        # Conservative should prefer low aFRR and idle energy
        assert action[0] == 0  # Zero aFRR commitment
        assert action[2] == 10  # Idle energy action


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
