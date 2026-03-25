"""
Battery Controller - The Central Control Loop.

Connects Data Provider (body) with Decision Strategy (brain).
This is the main orchestration layer for both Training and Production.

v23: I rewritten _build_observation() to use FeatureEngineer for training/production parity.
"""
import time
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from collections import deque

from data.data_provider import IDataProvider, NormalizationConfig
from agent.decision_strategy import IDecisionStrategy
from agent.feature_engineer import FeatureEngineer, MarketStateDTO


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

    v23: I now use FeatureEngineer to ensure observation parity between
    training (BatteryEnvMasked) and production (this controller).

    Responsibilities:
    1. Fetch data from IDataProvider
    2. Build observation for agent (using FeatureEngineer — same as training!)
    3. Calculate action mask for safety
    4. Get decision from IDecisionStrategy
    5. Execute dispatch command
    """

    def __init__(
        self,
        data_provider: IDataProvider,
        strategy: IDecisionStrategy,
        n_actions: int = 21,
        lookahead_hours: int = 12,
        battery_params: dict = None,
        include_price_awareness: bool = True,
        enable_gate_closure: bool = True,
        gate_closure_hours: float = 1.0
    ):
        self.provider = data_provider
        self.strategy = strategy
        self.n_actions = n_actions
        self.lookahead_hours = lookahead_hours

        # Pre-compute action levels
        self.action_levels = np.linspace(-1.0, 1.0, n_actions)

        # Get normalization config from provider
        self.norm_config = data_provider.get_normalization_config()

        # I initialize FeatureEngineer with the same config as training
        battery_params = battery_params or {
            'capacity_mwh': 146.0,
            'max_power_mw': 30.0,
            'efficiency': 0.94,
            'time_step_hours': 1.0,
            'include_price_awareness': include_price_awareness
        }
        self.feature_engine = FeatureEngineer(config=battery_params)
        self.enable_gate_closure = enable_gate_closure
        self.gate_closure_hours = gate_closure_hours

        # I maintain price history for backward-looking statistics
        self.price_history = deque(maxlen=720)  # 30 days at hourly
        self.daily_spreads = deque(maxlen=30)    # 30-day daily spreads
        self.hourly_means = {}  # hour → running mean for price_vs_typical_hour

        print(f" BatteryController initialized (v23 — FeatureEngineer parity)")
        print(f"   Data Provider: {type(data_provider).__name__}")
        print(f"   Strategy: {strategy.name}")
        print(f"   Actions: {n_actions}, Lookahead: {lookahead_hours}h")

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

        # 2. Build observation array (uses FeatureEngineer — matches training!)
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
        Build observation array using FeatureEngineer.

        v23: I now construct a MarketStateDTO and delegate to FeatureEngineer,
        ensuring IDENTICAL observations between training and production.
        """
        # I update price history for backward-looking statistics
        self.price_history.append(market.intraday_price)
        prices = np.array(self.price_history)

        # I compute backward-looking 24h statistics
        window = min(24, len(prices))
        recent = prices[-window:]
        price_max_24h = float(np.max(recent))
        price_min_24h = float(np.min(recent))
        price_mean_24h = float(np.mean(recent))

        # I compute 6h backward price momentum
        if len(prices) >= 7:
            price_momentum = (prices[-1] - prices[-7]) / max(abs(prices[-7]), 1.0)
        else:
            price_momentum = 0.0

        # I compute price_vs_typical_hour from running hourly means
        current_hour = market.timestamp.hour if market.timestamp else 0
        if current_hour in self.hourly_means:
            old_mean = self.hourly_means[current_hour]
            typical = old_mean
            pvt = (market.intraday_price - typical) / max(typical, 1.0)
            # I update the running mean with exponential weighting
            self.hourly_means[current_hour] = 0.99 * old_mean + 0.01 * market.intraday_price
        else:
            self.hourly_means[current_hour] = market.intraday_price
            pvt = 0.0

        # I compute trade_worthiness from daily spreads
        if len(prices) >= 24:
            daily_range = float(np.max(prices[-24:])) - float(np.min(prices[-24:]))
            tw = min(1.0, daily_range / 100.0)
        else:
            tw = 0.5

        # I prepare price lookahead (pad to 12 if shorter)
        price_lookahead = list(forecast.prices[:12])
        while len(price_lookahead) < 12:
            price_lookahead.append(market.intraday_price)

        # I prepare DAM commitment lookahead (pad to 12)
        dam_commitments = list(forecast.dam_commitments[:12])
        while len(dam_commitments) < 12:
            dam_commitments.append(0.0)

        # I construct the MarketStateDTO with all fields
        market_dto = MarketStateDTO(
            timestamp=market.timestamp or pd.Timestamp.now(),
            price=market.intraday_price,
            spread=getattr(market, 'spread', 4.0),
            dam_commitment_now=market.dam_commitment_mw,
            next_price=price_lookahead[0] if price_lookahead else market.intraday_price,
            price_max_24h=price_max_24h,
            price_min_24h=price_min_24h,
            price_mean_24h=price_mean_24h,
            dam_commitments=dam_commitments,
            price_lookahead=price_lookahead,
            volatility=0.0,
            afrr_up_mw=getattr(market, 'afrr_up_mw', 600.0),
            afrr_down_mw=getattr(market, 'afrr_down_mw', 130.0),
            price_vs_typical_hour=max(-1.0, min(1.0, pvt)),
            trade_worthiness=tw,
            price_momentum=float(np.clip(price_momentum, -1.0, 1.0))
        )

        # I compute imbalance risk from recent price history
        price_hist_list = list(prices[-3:]) if len(prices) >= 2 else [market.intraday_price] * 2
        imbalance_info = self.feature_engine.calculate_imbalance_risk(
            price_history=price_hist_list,
            res_history=[0.0] * len(price_hist_list),
            current_hour=current_hour
        )

        # I delegate to FeatureEngineer (same as training!)
        obs = self.feature_engine.build_observation(
            market=market_dto,
            battery_soc=telemetry.soc,
            imbalance_info=imbalance_info
        )

        # v22: I add gate closure feature if enabled (to match training)
        if self.enable_gate_closure:
            hours_until_closure = self._get_hours_until_gate_closure(
                dam_commitments, current_hour
            )
            gate_closure_normalized = min(1.0, hours_until_closure / 24.0)
            obs = np.append(obs, gate_closure_normalized)

        return obs

    def _get_hours_until_gate_closure(self, dam_commitments, current_hour) -> float:
        """I estimate hours until next gate closure for DAM commitment."""
        for i, dam_val in enumerate(dam_commitments):
            if abs(dam_val) > 0.1:
                hours_until_delivery = i + 1
                hours_until_closure = hours_until_delivery - self.gate_closure_hours
                return max(0.0, hours_until_closure)
        return 24.0

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
        print(f"\n Starting continuous control loop")
        print(f"   Interval: {interval_seconds}s ({interval_seconds/60:.1f} min)")
        print(f"   Max cycles: {max_cycles or 'infinite'}")
        print("=" * 50)

        cycle_count = 0

        try:
            while max_cycles is None or cycle_count < max_cycles:
                cycle_count += 1

                print(f"\n--- Cycle {cycle_count} ---")
                result = self.run_cycle()

                print(f"   Price: {result.price:.2f}€/MWh")
                print(f"   SoC: {result.soc:.2%}")
                print(f"   Action: {result.action_index} ({result.mw_setpoint:+.1f} MW)")
                print(f"   Strategy: {result.strategy_name}")

                # Advance to next step (for simulation)
                has_more = self.provider.step()
                if not has_more:
                    print("\n No more data available. Stopping.")
                    break

                # Wait for next cycle (skip in simulation)
                if interval_seconds > 0:
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n Control loop interrupted by user.")

        print(f"\n Control loop completed. Total cycles: {cycle_count}")
