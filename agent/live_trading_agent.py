import time
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sb3_contrib import MaskablePPO
from agent.feature_engineer import FeatureEngineer, MarketStateDTO
from data.data_provider import LiveDataProvider

class LiveTradingAgent:
    """
    Production Agent for Live Trading.
    Orchestrates:
    1. Data Fetching (LiveDataProvider)
    2. Feature Engineering (FeatureEngineer)
    3. Model Inference (MaskablePPO)
    4. Execution (LiveDataProvider/BMS)
    """
    def __init__(self, model_path: str, data_provider: LiveDataProvider):
        print(f"Loading model from {model_path}...")
        self.model = MaskablePPO.load(model_path)
        self.data_provider = data_provider
        self.feature_engine = FeatureEngineer(config={
            'capacity_mwh': data_provider.capacity_mwh,
            'max_power_mw': data_provider.max_power_mw,
            'efficiency': data_provider.efficiency,
            'time_step_hours': data_provider.time_step_hours
        })
        self.last_step_time = 0
        print("LiveTradingAgent initialized.")
        
    def run_tick(self) -> Dict:
        """
        Execute one trading cycle (tick).
        """
        # 1. Fetch Live Data
        # ------------------------------------------------
        market_state = self.data_provider.get_current_market_state() 
        imbalance_risk = self.data_provider.get_imbalance_risk()
        telemetry = self.data_provider.get_battery_telemetry()
        
        # 2. Prepare Market State DTO (Adapter)
        # ------------------------------------------------
        # Get 24h forecast for strategic features
        forecast = self.data_provider.get_forecast(hours_ahead=24)
        prices_24h = forecast.prices
        dam_commits = forecast.dam_commitments
        
        # Calculate stats
        if prices_24h:
            p_max = max(prices_24h)
            p_min = min(prices_24h)
            p_mean = float(np.mean(prices_24h))
            next_price = prices_24h[0]
        else:
            p_max = market_state.dam_price
            p_min = market_state.dam_price
            p_mean = market_state.dam_price
            next_price = market_state.dam_price
        
        dto = MarketStateDTO(
            timestamp=pd.Timestamp(market_state.timestamp),
            price=market_state.intraday_price, # Current trading price
            spread=market_state.spread,
            dam_commitment_now=market_state.dam_commitment_mw, # Live commitment
            next_price=next_price,
            price_max_24h=p_max,
            price_min_24h=p_min,
            price_mean_24h=p_mean,
            dam_commitments=dam_commits[:4], # First 4 hours
            volatility=5.0 # Placeholder: LiveDataProvider could provide this
        )
        
        # 3. Build Observation
        # ------------------------------------------------
        obs = self.feature_engine.build_observation(
            market=dto,
            battery_soc=telemetry.soc,
            imbalance_info=imbalance_risk
        )
        
        # 4. Predict Action
        # ------------------------------------------------
        # Get valid actions mask from BMS/Provider
        mask = self.data_provider.get_action_mask()
        
        # Predict (deterministic for production)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        
        # 5. Execute Action
        # ------------------------------------------------
        action_idx = int(action)
        action_level = self.data_provider.action_levels[action_idx]
        mw_setpoint = action_level * self.data_provider.max_power_mw
        
        # Dispatch to battery
        self.data_provider.dispatch_battery(mw_setpoint)
        
        return {
            'timestamp': market_state.timestamp,
            'price': market_state.intraday_price,
            'soc': telemetry.soc,
            'action_idx': action_idx,
            'action_level': action_level,
            'mw_setpoint': mw_setpoint,
            'risk_score': imbalance_risk['risk_score'],
            'gradient': imbalance_risk['gradient']
        }

    def run_loop(self, interval_seconds: int = 300):
        """
        Main operation loop.
        """
        print(f"Starting trading loop (interval={interval_seconds}s)...")
        while True:
            try:
                start_time = time.time()
                
                info = self.run_tick()
                
                # Log status
                log_msg = (
                    f"[{info['timestamp']:%H:%M}] "
                    f"Price:{info['price']:>6.2f}€ | "
                    f"SoC:{info['soc']:>5.1%} | "
                    f"Risk:{info['risk_score']:>4.2f} | "
                    f"Act:{info['action_level']:>+5.2f} ({info['mw_setpoint']:>+6.1f} MW)"
                )
                print(log_msg)
                
                # Wait for next tick
                elapsed = time.time() - start_time
                sleep_time = max(1.0, interval_seconds - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("\nStopping agent...")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60) # Retry delay
