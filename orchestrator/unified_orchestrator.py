"""
Unified Multi-Market Battery Trading Orchestrator

I orchestrate all market activities (DAM, IntraDay, aFRR, mFRR) using a single
unified model. This is the production controller that coordinates:

1. DAM commitment execution (from previous day's bidding)
2. aFRR capacity commitment and activation response
3. mFRR/IntraDay energy trading

The orchestrator manages capacity cascading:
Total Capacity (30 MW)
  └─> DAM Commitment (binding)
      └─> Remaining Capacity
          └─> aFRR Reservation
              └─> mFRR/IntraDay Trading
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, Optional, List
from pathlib import Path

from orchestrator.interfaces import (
    IUnifiedOrchestrator, IDAMForecaster, IDAMBidder, ICommitmentExecutor,
    PriceForecast, DAMCommitment, BatteryState, UnifiedDecision
)
from agent.unified_strategy import UnifiedStrategy, create_unified_strategy


class UnifiedOrchestrator(IUnifiedOrchestrator):
    """
    I orchestrate unified multi-market battery trading.

    This is the main production controller that uses the trained unified model
    to make coordinated decisions across all markets.
    """

    # Action space structure
    AFRR_LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    ENERGY_LEVELS = np.linspace(-1.0, 1.0, 21)

    def __init__(
        self,
        # Model
        strategy: Optional[UnifiedStrategy] = None,
        model_path: Optional[str] = None,

        # Components
        dam_forecaster: Optional[IDAMForecaster] = None,
        dam_bidder: Optional[IDAMBidder] = None,
        commitment_executor: Optional[ICommitmentExecutor] = None,

        # Battery configuration
        capacity_mwh: float = 146.0,
        max_power_mw: float = 30.0,
        efficiency: float = 0.94,
        min_soc: float = 0.05,
        max_soc: float = 0.95,

        # aFRR configuration
        selection_probability: float = 0.80,
        afrr_activation_rate: float = 0.15
    ):
        # I initialize strategy
        if strategy is not None:
            self.strategy = strategy
        elif model_path is not None:
            self.strategy = create_unified_strategy("AI", model_path)
        else:
            # Fallback to rule-based
            self.strategy = create_unified_strategy("RULE_BASED")

        # I store components
        self.dam_forecaster = dam_forecaster
        self.dam_bidder = dam_bidder
        self.commitment_executor = commitment_executor

        # I store battery configuration
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.eff_sqrt = np.sqrt(efficiency)
        self.min_soc = min_soc
        self.max_soc = max_soc

        # I store aFRR configuration
        self.selection_probability = selection_probability
        self.afrr_activation_rate = afrr_activation_rate

        # I initialize state
        self.current_soc = 0.5
        self.current_dam_commitment = 0.0
        self.current_afrr_commitment_mw = 0.0
        self.is_selected_for_afrr = False
        self.hours_since_afrr_activation = 24

        # I track performance
        self.total_profit = 0.0
        self.dam_profit = 0.0
        self.afrr_capacity_profit = 0.0
        self.afrr_energy_profit = 0.0
        self.mfrr_profit = 0.0
        self.total_cycles = 0.0

        # I track history
        self.decision_history: List[UnifiedDecision] = []
        self.market_history: List[Dict] = []

        print(f"Unified Orchestrator initialized")
        print(f"  Strategy: {self.strategy.name}")
        print(f"  Battery: {capacity_mwh} MWh, {max_power_mw} MW")

    def run_day_ahead(self, target_date: date) -> DAMCommitment:
        """I run the day-ahead bidding process."""
        if self.dam_forecaster is None or self.dam_bidder is None:
            raise RuntimeError("DAM forecaster and bidder required for day-ahead")

        # I get price forecast
        forecast = self.dam_forecaster.forecast(target_date)

        # I get current battery state
        battery_state = BatteryState(
            timestamp=datetime.now(),
            soc=self.current_soc,
            available_power_mw=self.max_power_mw,
            capacity_mwh=self.capacity_mwh
        )

        # I generate DAM bids
        constraints = {
            'max_daily_cycles': 2.0,
            'min_soc': self.min_soc,
            'max_soc': self.max_soc
        }

        commitment = self.dam_bidder.generate_bids(forecast, battery_state, constraints)

        # I load commitment for execution
        if self.commitment_executor is not None:
            self.commitment_executor.load_commitment(commitment)

        return commitment

    def run_realtime(self, timestamp: datetime) -> dict:
        """I run real-time unified trading."""
        decision = self.run_unified_step(timestamp)
        return self._decision_to_dict(decision)

    def run_unified_step(
        self,
        timestamp: datetime,
        battery_state: Optional[BatteryState] = None,
        market_data: Optional[Dict] = None
    ) -> UnifiedDecision:
        """
        I execute a single unified trading step.

        This is the main decision function that:
        1. Gets current DAM commitment
        2. Calculates remaining capacity
        3. Builds observation
        4. Gets action from model
        5. Interprets and executes action
        """
        # I get current battery state
        if battery_state is not None:
            self.current_soc = battery_state.soc

        # I get current DAM commitment
        if self.commitment_executor is not None:
            self.current_dam_commitment = self.commitment_executor.get_current_commitment(timestamp)
        else:
            self.current_dam_commitment = market_data.get('dam_commitment', 0.0) if market_data else 0.0

        # I calculate remaining capacity
        remaining_capacity = self.max_power_mw - abs(self.current_dam_commitment)

        # I build observation
        observation = self._build_observation(timestamp, market_data)

        # I build action mask
        action_mask = self._build_action_mask(remaining_capacity)

        # I get action from strategy
        action = self.strategy.predict_action(observation, action_mask)

        # I unpack action
        afrr_action = int(action[0])
        price_action = int(action[1])
        energy_action = int(action[2])

        # I interpret action
        afrr_level = self.AFRR_LEVELS[afrr_action]
        afrr_commitment_mw = afrr_level * remaining_capacity

        price_tier = self.PRICE_TIERS[price_action]

        energy_level = self.ENERGY_LEVELS[energy_action]
        energy_power_mw = energy_level * remaining_capacity

        # I determine aFRR selection
        adjusted_selection_prob = self.selection_probability * (1.5 - price_tier * 0.5)
        adjusted_selection_prob = np.clip(adjusted_selection_prob, 0.1, 0.95)
        self.is_selected_for_afrr = (
            afrr_commitment_mw > 0.1 and
            np.random.random() < adjusted_selection_prob
        )

        # I check for aFRR activation
        afrr_activated = False
        afrr_direction = None
        actual_energy_mw = 0.0

        if self.is_selected_for_afrr and afrr_commitment_mw > 0.1:
            afrr_activated, afrr_direction = self._check_afrr_activation(timestamp, market_data)

            if afrr_activated:
                # I execute aFRR response
                actual_energy_mw = self._execute_afrr(afrr_commitment_mw, afrr_direction, market_data)
                self.hours_since_afrr_activation = 0
            else:
                self.hours_since_afrr_activation += 1
        else:
            self.hours_since_afrr_activation += 1

        # I execute mFRR/IntraDay if capacity remains and no aFRR activation
        if not afrr_activated and abs(energy_power_mw) > 0.1:
            actual_energy_mw = self._execute_energy(energy_power_mw, market_data)

        # I create decision record
        decision = UnifiedDecision(
            timestamp=timestamp,
            afrr_commitment_level=afrr_action,
            afrr_commitment_mw=afrr_commitment_mw,
            afrr_price_tier=price_action,
            energy_action=energy_action,
            energy_power_mw=energy_power_mw,
            dam_commitment_mw=self.current_dam_commitment,
            remaining_capacity_mw=remaining_capacity,
            is_selected_for_afrr=self.is_selected_for_afrr,
            afrr_activated=afrr_activated,
            afrr_activation_direction=afrr_direction,
            actual_energy_mw=actual_energy_mw
        )

        # I store history
        self.decision_history.append(decision)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]

        return decision

    def _build_observation(
        self,
        timestamp: datetime,
        market_data: Optional[Dict]
    ) -> np.ndarray:
        """I build the 58-feature observation vector."""
        features = []

        # I get market data with defaults
        data = market_data or {}

        # 1. Battery State (3)
        features.append(self.current_soc)

        remaining_capacity = self.max_power_mw - abs(self.current_dam_commitment)
        available_discharge = (self.current_soc - self.min_soc) * self.capacity_mwh
        available_charge = (self.max_soc - self.current_soc) * self.capacity_mwh

        max_discharge = min(remaining_capacity, available_discharge * self.eff_sqrt)
        max_charge = min(remaining_capacity, available_charge / self.eff_sqrt)

        features.append(max_discharge / self.max_power_mw)
        features.append(max_charge / self.max_power_mw)

        # 2. Market Prices (6)
        dam_price = data.get('dam_price', 100.0)
        features.append(dam_price / 100.0)
        features.append(data.get('intraday_price', dam_price) / 100.0)
        features.append(data.get('spread', 4.0) / 100.0)
        features.append(data.get('mfrr_price_up', 120.0) / 100.0)
        features.append(data.get('mfrr_price_down', 60.0) / 100.0)
        features.append(data.get('mfrr_spread', 60.0) / 100.0)

        # 3. Time Encoding (4)
        hour = timestamp.hour
        dow = timestamp.weekday()
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # 4. DAM Lookahead (13) - I use zeros for unknown future
        features.append(self.current_dam_commitment / self.max_power_mw)
        for _ in range(12):
            features.append(0.0)  # Unknown future commitments

        # 5. Price Lookahead (12) - I use current price as estimate
        for _ in range(12):
            features.append(dam_price / 100.0)

        # 6. aFRR State (8)
        features.append(data.get('afrr_cap_up_price', 20.0) / 50.0)
        features.append(data.get('afrr_cap_down_price', 30.0) / 50.0)
        features.append(data.get('net_imbalance', 0) / 1000.0)
        features.append(self.current_afrr_commitment_mw / self.max_power_mw)
        features.append(0.5)  # Default selection probability estimate
        features.append(1.0 if self.hours_since_afrr_activation < 4 else 0.0)
        features.append(min(self.hours_since_afrr_activation, 24) / 24.0)
        features.append(1.0 if self.is_selected_for_afrr else 0.0)

        # 7. Risk/Imbalance (4)
        features.append(data.get('net_imbalance', 0) / 1000.0)
        features.append(0.0)  # Shortfall risk
        features.append(data.get('price_momentum', 0.0))
        features.append(data.get('volatility', 0.3))

        # 8. Gate Closure (4)
        features.append(1.0 if hour < 14 else 0.0)  # DAM gate
        features.append(max(0, 24 - hour) / 24.0)  # IntraDay hours
        features.append(max(0, 23.75 - hour) / 24.0)  # aFRR hours
        features.append(1.0 if hour < 23 else 0.0)  # mFRR gate

        # 9. Timing Signals (4)
        features.append(data.get('price_vs_typical', 0.0))
        features.append(1.0 if 17 <= hour <= 21 else 0.0)  # is_peak
        features.append(1.0 if 9 <= hour <= 15 else 0.0)   # is_solar
        features.append(min(abs(19 - hour), abs(19 - hour + 24)) / 12.0)

        obs = np.array(features, dtype=np.float32)
        assert len(obs) == 58, f"Expected 58 features, got {len(obs)}"

        return obs

    def _build_action_mask(self, remaining_capacity: float) -> np.ndarray:
        """I build the action mask for valid actions."""
        # I calculate SoC-based limits
        available_discharge = (self.current_soc - self.min_soc) * self.capacity_mwh
        available_charge = (self.max_soc - self.current_soc) * self.capacity_mwh

        max_discharge = min(remaining_capacity, available_discharge * self.eff_sqrt)
        max_charge = min(remaining_capacity, available_charge / self.eff_sqrt)

        # aFRR mask (5)
        afrr_mask = np.ones(5, dtype=bool)
        for i, level in enumerate(self.AFRR_LEVELS):
            required = level * remaining_capacity
            if required > max_discharge or required > max_charge:
                afrr_mask[i] = False
        afrr_mask[0] = True  # Always allow zero

        # Price tier mask (5) - always all valid
        price_mask = np.ones(5, dtype=bool)

        # Energy mask (21)
        energy_mask = np.zeros(21, dtype=bool)
        for i, level in enumerate(self.ENERGY_LEVELS):
            power = level * remaining_capacity
            if level >= 0:
                if power <= max_discharge + 0.01:
                    energy_mask[i] = True
            else:
                if abs(power) <= max_charge + 0.01:
                    energy_mask[i] = True
        energy_mask[10] = True  # Always allow idle

        return np.concatenate([afrr_mask, price_mask, energy_mask])

    def _check_afrr_activation(
        self,
        timestamp: datetime,
        market_data: Optional[Dict]
    ) -> tuple:
        """I check if aFRR is activated."""
        activation_prob = self.afrr_activation_rate

        # I increase during peak hours
        if 17 <= timestamp.hour <= 21:
            activation_prob *= 1.3

        # I increase during high imbalance
        if market_data and abs(market_data.get('net_imbalance', 0)) > 500:
            activation_prob *= 1.5

        if np.random.random() < activation_prob:
            # I determine direction
            if market_data:
                imbalance = market_data.get('net_imbalance', 0)
                if imbalance > 100:
                    return True, 'up'
                elif imbalance < -100:
                    return True, 'down'

            # Default based on price
            afrr_up = market_data.get('afrr_up', 80) if market_data else 80
            afrr_down = market_data.get('afrr_down', 80) if market_data else 80
            return True, 'up' if afrr_up > afrr_down else 'down'

        return False, None

    def _execute_afrr(
        self,
        commitment_mw: float,
        direction: str,
        market_data: Optional[Dict]
    ) -> float:
        """I execute aFRR activation response."""
        if direction == 'up':
            # Discharge
            available = (self.current_soc - self.min_soc) * self.capacity_mwh * self.eff_sqrt
            actual_mw = min(commitment_mw, available)
            if actual_mw > 0.1:
                energy_mwh = actual_mw
                soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                self.current_soc = max(self.min_soc, self.current_soc - soc_delta)

                price = market_data.get('afrr_up', 80) if market_data else 80
                revenue = actual_mw * price
                self.afrr_energy_profit += revenue
                self.total_cycles += actual_mw / self.capacity_mwh

            return actual_mw
        else:
            # Charge
            available = (self.max_soc - self.current_soc) * self.capacity_mwh / self.eff_sqrt
            actual_mw = min(commitment_mw, available)
            if actual_mw > 0.1:
                energy_mwh = actual_mw
                soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                self.current_soc = min(self.max_soc, self.current_soc + soc_delta)

                price = market_data.get('afrr_down', 80) if market_data else 80
                cost = actual_mw * price
                self.afrr_energy_profit -= cost
                self.total_cycles += actual_mw / self.capacity_mwh

            return -actual_mw

    def _execute_energy(
        self,
        power_mw: float,
        market_data: Optional[Dict]
    ) -> float:
        """I execute mFRR/IntraDay energy trading."""
        if power_mw > 0:
            # Discharge
            available = (self.current_soc - self.min_soc) * self.capacity_mwh * self.eff_sqrt
            actual_mw = min(power_mw, available)
            if actual_mw > 0.1:
                soc_delta = actual_mw / self.eff_sqrt / self.capacity_mwh
                self.current_soc = max(self.min_soc, self.current_soc - soc_delta)

                price = market_data.get('mfrr_price_up', 120) if market_data else 120
                revenue = actual_mw * price
                self.mfrr_profit += revenue
                self.total_cycles += actual_mw / self.capacity_mwh

            return actual_mw
        else:
            # Charge
            available = (self.max_soc - self.current_soc) * self.capacity_mwh / self.eff_sqrt
            actual_mw = min(abs(power_mw), available)
            if actual_mw > 0.1:
                soc_delta = actual_mw * self.eff_sqrt / self.capacity_mwh
                self.current_soc = min(self.max_soc, self.current_soc + soc_delta)

                price = market_data.get('mfrr_price_down', 60) if market_data else 60
                cost = actual_mw * price
                self.mfrr_profit -= cost
                self.total_cycles += actual_mw / self.capacity_mwh

            return -actual_mw

    def _decision_to_dict(self, decision: UnifiedDecision) -> dict:
        """I convert a decision to a dictionary."""
        return {
            'timestamp': decision.timestamp.isoformat(),
            'afrr_commitment_mw': decision.afrr_commitment_mw,
            'afrr_price_tier': decision.afrr_price_tier,
            'energy_power_mw': decision.energy_power_mw,
            'dam_commitment_mw': decision.dam_commitment_mw,
            'actual_energy_mw': decision.actual_energy_mw,
            'is_selected_for_afrr': decision.is_selected_for_afrr,
            'afrr_activated': decision.afrr_activated,
            'afrr_direction': decision.afrr_activation_direction,
            'total_power_mw': decision.total_power_mw
        }

    def get_current_state(self) -> dict:
        """I return the current state of all markets."""
        return {
            'soc': self.current_soc,
            'dam_commitment_mw': self.current_dam_commitment,
            'afrr_commitment_mw': self.current_afrr_commitment_mw,
            'is_selected_for_afrr': self.is_selected_for_afrr,
            'hours_since_afrr_activation': self.hours_since_afrr_activation,
            'total_profit': self.total_profit,
            'dam_profit': self.dam_profit,
            'afrr_capacity_profit': self.afrr_capacity_profit,
            'afrr_energy_profit': self.afrr_energy_profit,
            'mfrr_profit': self.mfrr_profit,
            'total_cycles': self.total_cycles
        }

    def get_profit_breakdown(self) -> dict:
        """I return detailed profit breakdown by market."""
        return {
            'total_profit': self.total_profit,
            'dam_profit': self.dam_profit,
            'afrr_capacity_profit': self.afrr_capacity_profit,
            'afrr_energy_profit': self.afrr_energy_profit,
            'mfrr_profit': self.mfrr_profit,
            'total_cycles': self.total_cycles,
            'profit_per_cycle': self.total_profit / max(1, self.total_cycles)
        }
