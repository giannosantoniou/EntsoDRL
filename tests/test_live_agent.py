import sys
import os
sys.path.append(os.getcwd())

from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from agent.live_trading_agent import LiveTradingAgent
from data.data_provider import LiveDataProvider, MarketState, BatteryTelemetry, PriceForecast

def test_live_agent():
    print("Testing LiveTradingAgent logic...")
    
    # Mock Data Provider
    mock_provider = MagicMock(spec=LiveDataProvider)
    mock_provider.action_levels = np.linspace(-1, 1, 21)
    mock_provider.max_power_mw = 30.0
    mock_provider.capacity_mwh = 50.0
    mock_provider.efficiency = 0.94
    mock_provider.time_step_hours = 1.0
    
    # Mock return values
    mock_provider.get_current_market_state.return_value = MarketState(
        timestamp=pd.Timestamp.now().to_pydatetime(),
        dam_price=50.0,
        intraday_price=55.0,
        dam_commitment_mw=0.0,
        best_bid=54.0,
        best_ask=56.0,
        bid_volume=10.0,
        ask_volume=10.0,
        spread=2.0,
        imbalance_price_long=0.0,
        imbalance_price_short=0.0,
        hour_sin=0.0,
        hour_cos=1.0
    )
    
    mock_provider.get_imbalance_risk.return_value = {
        'gradient': 10.0,
        'res_deviation': -20.0,
        'risk_score': 0.5
    }
    
    mock_provider.get_battery_telemetry.return_value = BatteryTelemetry(
        soc=0.5,
        max_charge_mw=30.0,
        max_discharge_mw=30.0,
        health_status=True
    )
    
    mock_provider.get_forecast.return_value = PriceForecast(
        prices=[50.0]*24,
        dam_commitments=[0.0]*24
    )
    
    mock_provider.get_action_mask.return_value = np.ones(21, dtype=bool)
    
    # Patch MaskablePPO.load to return a mock model
    with patch('sb3_contrib.MaskablePPO.load') as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([10]), None) # Action 10 (Idle)
        mock_load.return_value = mock_model
        
        # Instantiate Agent
        agent = LiveTradingAgent("dummy_path", mock_provider)
        
        # Run tick
        info = agent.run_tick()
        
        print("\nRun tick successful!")
        print("Agent Output Info:", info)
        
        # Verify interactions
        mock_provider.get_current_market_state.assert_called_once()
        mock_provider.dispatch_battery.assert_called_once()
        print("\nVerifications passed: Data fetched and Battery dispatched.")

if __name__ == "__main__":
    test_live_agent()
