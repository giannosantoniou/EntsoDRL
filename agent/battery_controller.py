"""
Battery Controller - The Central Control Loop.

Connects Data Provider (body) with Decision Strategy (brain).
This is the main orchestration layer for both Training and Production.
"""
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass

from data.data_provider import IDataProvider, NormalizationConfig
from agent.decision_strategy import IDecisionStrategy


@dataclass
class ControlCycleResult:
    """Result of a single control cycle."""
    action_index: int
    mw_setpoint: float
    price: float
    soc: float
    strategy_name: str
    timestamp: str


class BatteryController:
    """
    The central controller that connects Data and Decisions.
    
    Responsibilities:
    1. Fetch data from IDataProvider
    2. Build observation for agent
    3. Calculate action mask for safety
    4. Get decision from IDecisionStrategy
    5. Execute dispatch command
    """
    
    def __init__(
        self, 
        data_provider: IDataProvider,
        strategy: IDecisionStrategy,
        n_actions: int = 21,
        lookahead_hours: int = 4
    ):
        self.provider = data_provider
        self.strategy = strategy
        self.n_actions = n_actions
        self.lookahead_hours = lookahead_hours
        
        # Pre-compute action levels
        self.action_levels = np.linspace(-1.0, 1.0, n_actions)
        
        # Get normalization config from provider
        self.norm_config = data_provider.get_normalization_config()
        
        print(f"🎮 BatteryController initialized")
        print(f"   Data Provider: {type(data_provider).__name__}")
        print(f"   Strategy: {strategy.name}")
        print(f"   Actions: {n_actions}")
    
    def run_cycle(self) -> ControlCycleResult:
        """
        Execute one control cycle.
        
        Returns:
            ControlCycleResult with details of the action taken
        """
        # 1. Fetch current state from provider
        market = self.provider.get_current_market_state()
        telemetry = self.provider.get_battery_telemetry()
        forecast = self.provider.get_forecast(self.lookahead_hours)
        
        # 2. Build observation array (must match training!)
        obs = self._build_observation(market, telemetry, forecast)
        
        # 3. Calculate action mask (safety constraints)
        mask = self._calculate_action_mask(telemetry)
        
        # 4. Get decision from strategy (AI, Rules, or Random)
        action_idx = self.strategy.predict_action(obs, mask)
        
        # 5. Convert action to MW setpoint
        action_level = self.action_levels[action_idx]
        mw_setpoint = action_level * self.norm_config.power_scale
        
        # 6. Apply to battery
        self.provider.dispatch_battery(mw_setpoint)
        
        # 7. Return result
        return ControlCycleResult(
            action_index=action_idx,
            mw_setpoint=mw_setpoint,
            price=market.intraday_price,
            soc=telemetry.soc,
            strategy_name=self.strategy.name,
            timestamp=market.timestamp.isoformat() if market.timestamp else ""
        )
    
    def _build_observation(self, market, telemetry, forecast) -> np.ndarray:
        """
        Build observation array for agent.
        MUST match the format used during training!
        
        Features (15 total):
        1. SoC
        2. max_discharge (normalized)
        3. max_charge (normalized)
        4. current_price (normalized)
        5. dam_commitment (normalized)
        6. hour_sin
        7. hour_cos
        8-11. price_lookahead (4h)
        12-15. dam_lookahead (4h)
        """
        obs = [
            telemetry.soc,
            telemetry.max_discharge_mw / self.norm_config.power_scale,
            telemetry.max_charge_mw / self.norm_config.power_scale,
            market.intraday_price / self.norm_config.price_scale,
            market.dam_commitment_mw / self.norm_config.power_scale,
            market.hour_sin,
            market.hour_cos,
        ]
        
        # Price lookahead
        for price in forecast.prices:
            obs.append(price / self.norm_config.price_scale)
        
        # DAM commitment lookahead
        for dam in forecast.dam_commitments:
            obs.append(dam / self.norm_config.power_scale)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_action_mask(self, telemetry) -> np.ndarray:
        """
        Calculate which actions are valid based on battery state.
        This is a SAFETY LAYER - prevents impossible actions.
        """
        max_discharge_action = telemetry.max_discharge_mw / self.norm_config.power_scale
        max_charge_action = telemetry.max_charge_mw / self.norm_config.power_scale
        
        mask = np.zeros(self.n_actions, dtype=bool)
        
        for i, action_level in enumerate(self.action_levels):
            if action_level >= 0:  # Discharge
                mask[i] = action_level <= max_discharge_action + 0.01
            else:  # Charge
                mask[i] = abs(action_level) <= max_charge_action + 0.01
        
        # Always allow idle
        center_idx = self.n_actions // 2
        mask[center_idx] = True
        
        return mask
    
    def run_continuous(self, interval_seconds: float = 900.0, max_cycles: Optional[int] = None):
        """
        Run control loop continuously.
        
        Args:
            interval_seconds: Time between cycles (default 15 minutes)
            max_cycles: Maximum number of cycles (None = infinite)
        """
        print(f"\n🚀 Starting continuous control loop")
        print(f"   Interval: {interval_seconds}s ({interval_seconds/60:.1f} min)")
        print(f"   Max cycles: {max_cycles or 'infinite'}")
        print("=" * 50)
        
        cycle_count = 0
        
        try:
            while max_cycles is None or cycle_count < max_cycles:
                cycle_count += 1
                
                print(f"\n--- Cycle {cycle_count} ---")
                result = self.run_cycle()
                
                print(f"  📊 Price: {result.price:.2f}€/MWh")
                print(f"  🔋 SoC: {result.soc:.2%}")
                print(f"  ⚡ Action: {result.action_index} ({result.mw_setpoint:+.1f} MW)")
                print(f"  🧠 Strategy: {result.strategy_name}")
                
                # Advance to next step (for simulation)
                has_more = self.provider.step()
                if not has_more:
                    print("\n⚠️ No more data available. Stopping.")
                    break
                
                # Wait for next cycle (skip in simulation)
                if interval_seconds > 0:
                    time.sleep(interval_seconds)
                    
        except KeyboardInterrupt:
            print("\n\n⛔ Control loop interrupted by user.")
        
        print(f"\n✅ Control loop completed. Total cycles: {cycle_count}")
