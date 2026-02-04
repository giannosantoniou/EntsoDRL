import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env import BatteryTradingEnvV2
from gym_envs.action_mask import ActionMasker

@pytest.fixture
def mock_data():
    dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
    df = pd.DataFrame({
        'price': np.random.rand(24) * 100,
        'forecast_price': np.random.rand(24) * 100,
        'hour': dates.hour / 23.0,
        'day_of_week': dates.dayofweek / 6.0
    }, index=dates)
    return df

@pytest.fixture
def battery_params():
    return {
        'capacity_mwh': 1.0,
        'max_charge_mw': 1.0,
        'max_discharge_mw': 1.0,
        'efficiency': 0.9,
        'cost_per_mwh_throughput': 0.0
    }

def test_initialization(mock_data, battery_params):
    env = BatteryTradingEnv(mock_data, battery_params)
    obs, info = env.reset()
    assert env.current_soc == 0.5
    assert obs.shape == (5,)

def test_charge_physics(mock_data, battery_params):
    env = BatteryTradingEnv(mock_data, battery_params, initial_soc=0.5)
    env.reset()
    
    # Charge Max (Action = -1.0)
    action = np.array([-1.0], dtype=np.float32)
    obs, reward, terminated, _, info = env.step(action)
    
    # Input Power = 1.0 MW. 
    # Effective Factor = sqrt(0.9) approx 0.948
    # SoC Change = (1.0 * 0.948) / 1.0 = 0.948
    # New SoC = 0.5 + 0.948 = 1.448 -> Clipped to 1.0
    
    assert env.current_soc == 1.0
    
def test_masking(mock_data, battery_params):
    env = BatteryTradingEnv(mock_data, battery_params, initial_soc=1.0)
    wrapped_env = ActionMasker(env)
    wrapped_env.reset()
    
    # Try to Charge when Full (Action = -1.0)
    action = np.array([-1.0], dtype=np.float32)
    
    # Should be masked to 0.0
    # Wrapper calls env.step(masked_action)
    # But ActionMasker return value from action() is not directly passed to step in Gym wrappers easily
    # Wait, ActionWrapper.action() modifies the action passed TO step().
    
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    
    # Action should have been 0.0
    assert info['action_mw'] == 0.0
    assert wrapped_env.unwrapped.current_soc == 1.0
