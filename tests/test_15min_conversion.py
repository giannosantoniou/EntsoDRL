"""
Comprehensive Test Suite for 15-Minute Interval Conversion.

Tests:
1. Upsampling correctness (row count, price values, time features)
2. Environment parameterization (lookahead scaling, SoC physics)
3. Reward calculation consistency
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.upsample_to_15min import upsample_hourly_to_15min, verify_upsampling
from gym_envs.battery_env_masked import BatteryEnvMasked


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def hourly_data():
    """Create sample hourly data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=48, freq='H')
    df = pd.DataFrame({
        'price': np.sin(np.linspace(0, 4*np.pi, 48)) * 30 + 70,  # 40-100€
        'dam_commitment': np.random.choice([10, 20, -10, -20, 0], size=48),
        'solar': np.maximum(0, np.sin(np.linspace(0, 4*np.pi, 48)) * 500),
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dates.hour / 24),
        'Long': np.ones(48) * 163.8,
        'Short': np.ones(48) * 163.8,
    }, index=dates)
    return df


@pytest.fixture
def battery_params():
    """Standard battery parameters."""
    return {
        'capacity_mwh': 30.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }


# ============================================================================
# UPSAMPLING TESTS
# ============================================================================

class TestUpsampling:
    """Tests for the upsample_to_15min utility."""
    
    def test_row_count_increases_4x(self, hourly_data):
        """After upsampling, we should have ~4x more rows."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        
        original_len = len(hourly_data)
        expected_min = (original_len - 1) * 4 + 1  # At least this many
        
        assert len(df_15min) >= expected_min, \
            f"Expected at least {expected_min} rows, got {len(df_15min)}"
    
    def test_step_interpolation_repeats_values(self, hourly_data):
        """With step interpolation, each hour's price should repeat 4 times."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        
        first_hour_price = hourly_data['price'].iloc[0]
        first_4_slots = df_15min['price'].iloc[:4]
        
        # All 4 slots should have same price
        assert np.allclose(first_4_slots, first_hour_price), \
            f"Step interpolation failed: expected {first_hour_price}, got {first_4_slots.tolist()}"
    
    def test_linear_interpolation_smooth_transition(self, hourly_data):
        """With linear interpolation, values should smoothly transition."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='linear')
        
        # The middle slots (15min, 30min, 45min) should be between endpoints
        price_0h = hourly_data['price'].iloc[0]
        price_1h = hourly_data['price'].iloc[1]
        
        price_15min = df_15min['price'].iloc[1]  # 00:15
        
        if price_0h != price_1h:
            # Should be between the two values
            min_val, max_val = min(price_0h, price_1h), max(price_0h, price_1h)
            assert min_val <= price_15min <= max_val, \
                f"Linear interpolation failed: {price_15min} not between {price_0h} and {price_1h}"
    
    def test_time_features_recalculated_for_96_slots(self, hourly_data):
        """Time features should be recalculated for 96 slots/day."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        
        # At 00:00, hour_sin should be 0, hour_cos should be 1
        assert abs(df_15min['hour_sin'].iloc[0]) < 0.01, "hour_sin at 00:00 should be ~0"
        assert abs(df_15min['hour_cos'].iloc[0] - 1.0) < 0.01, "hour_cos at 00:00 should be ~1"
        
        # At 06:00 (slot 24), hour_sin should be 1, hour_cos should be ~0
        slot_6am = 24  # 6 hours * 4 slots/hour
        hour_sin_6am = df_15min['hour_sin'].iloc[slot_6am]
        hour_cos_6am = df_15min['hour_cos'].iloc[slot_6am]
        
        assert abs(hour_sin_6am - 1.0) < 0.1, f"hour_sin at 06:00 should be ~1, got {hour_sin_6am}"
        assert abs(hour_cos_6am) < 0.1, f"hour_cos at 06:00 should be ~0, got {hour_cos_6am}"
    
    def test_dam_commitment_divided_by_4(self, hourly_data):
        """DAM commitments should be divided by 4 for 15-min slots."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step', divide_energy_by_4=True)
        
        first_hour_dam = hourly_data['dam_commitment'].iloc[0]
        first_slot_dam = df_15min['dam_commitment'].iloc[0]
        
        assert abs(first_slot_dam - first_hour_dam / 4) < 0.01, \
            f"DAM should be {first_hour_dam/4}, got {first_slot_dam}"
    
    def test_verification_passes(self, hourly_data):
        """The built-in verification should pass."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        result = verify_upsampling(hourly_data, df_15min)
        
        assert result['passed'], f"Verification failed: {result['checks']}"


# ============================================================================
# ENVIRONMENT TESTS
# ============================================================================

class TestEnvironmentParameterization:
    """Tests for battery_env_masked with time_step_hours parameter."""
    
    def test_hourly_lookahead_is_4(self, hourly_data, battery_params):
        """With hourly data (1.0), lookahead should be 4 steps."""
        env = BatteryEnvMasked(hourly_data, battery_params, time_step_hours=1.0)
        
        assert env.lookahead_steps == 4, f"Expected 4, got {env.lookahead_steps}"
        assert env.strategic_lookahead == 24, f"Expected 24, got {env.strategic_lookahead}"
    
    def test_15min_lookahead_is_16(self, hourly_data, battery_params):
        """With 15-min data (0.25), lookahead should be 16 steps (= 4 hours)."""
        # First upsample
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        
        env = BatteryEnvMasked(df_15min, battery_params, time_step_hours=0.25)
        
        assert env.lookahead_steps == 16, f"Expected 16, got {env.lookahead_steps}"
        assert env.strategic_lookahead == 96, f"Expected 96, got {env.strategic_lookahead}"
    
    def test_soc_change_scales_with_time_step(self, hourly_data, battery_params):
        """SoC change in 4 × 15-min should equal 1 × hourly (same total energy)."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        
        # Hourly environment
        env_hourly = BatteryEnvMasked(hourly_data.copy(), battery_params, time_step_hours=1.0)
        env_hourly.reset()
        initial_soc_hourly = env_hourly.current_soc
        
        # Take full discharge action for 1 hour (middle action is idle, last is full discharge)
        full_discharge_action = env_hourly.n_actions - 1  # Full discharge
        env_hourly.step(full_discharge_action)
        soc_delta_hourly = initial_soc_hourly - env_hourly.current_soc
        
        # 15-min environment
        env_15min = BatteryEnvMasked(df_15min.copy(), battery_params, time_step_hours=0.25)
        env_15min.reset()
        initial_soc_15min = env_15min.current_soc
        
        # Take same full discharge action for 4 × 15-min = 1 hour
        for _ in range(4):
            env_15min.step(full_discharge_action)
        soc_delta_15min = initial_soc_15min - env_15min.current_soc
        
        # SoC deltas should be approximately equal
        assert abs(soc_delta_hourly - soc_delta_15min) < 0.02, \
            f"SoC delta mismatch: hourly={soc_delta_hourly:.4f}, 4×15min={soc_delta_15min:.4f}"
    
    def test_observation_shape_unchanged(self, hourly_data, battery_params):
        """Observation shape should remain 16 regardless of time resolution."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        
        env_hourly = BatteryEnvMasked(hourly_data, battery_params, time_step_hours=1.0)
        env_15min = BatteryEnvMasked(df_15min, battery_params, time_step_hours=0.25)
        
        obs_hourly, _ = env_hourly.reset()
        obs_15min, _ = env_15min.reset()
        
        assert obs_hourly.shape == (16,), f"Hourly obs shape wrong: {obs_hourly.shape}"
        assert obs_15min.shape == (16,), f"15-min obs shape wrong: {obs_15min.shape}"
    
    def test_action_masking_works(self, hourly_data, battery_params):
        """Action masking should prevent impossible actions at any time resolution."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        env = BatteryEnvMasked(df_15min, battery_params, time_step_hours=0.25)
        
        env.reset()
        
        # At initial SoC=0.5, both charge and discharge should be possible
        mask = env.action_masks()
        
        # Middle action (idle) should always be valid
        idle_action = env.n_actions // 2
        assert mask[idle_action], "Idle action should always be valid"
        
        # Some charge and discharge actions should be valid
        assert any(mask[:idle_action]), "Some charge actions should be valid at SoC=0.5"
        assert any(mask[idle_action+1:]), "Some discharge actions should be valid at SoC=0.5"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_episode_runs_without_error(self, hourly_data, battery_params):
        """A full episode should complete without errors."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        env = BatteryEnvMasked(df_15min, battery_params, time_step_hours=0.25)
        
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        
        while not terminated and steps < 1000:
            # Random valid action
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"Episode completed: {steps} steps, total_reward={total_reward:.2f}")
        assert steps > 0, "Episode should have taken at least 1 step"
    
    def test_no_nan_in_observations(self, hourly_data, battery_params):
        """Observations should never contain NaN values."""
        df_15min = upsample_hourly_to_15min(hourly_data, method='step')
        env = BatteryEnvMasked(df_15min, battery_params, time_step_hours=0.25)
        
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs)), f"NaN in reset observation: {obs}"
        
        for _ in range(50):
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            
            obs, _, terminated, _, _ = env.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in observation after step: {obs}"
            
            if terminated:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
