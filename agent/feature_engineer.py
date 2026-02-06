"""
Feature Engineering Module
Decouples observation construction logic from the Environment/Agent.
Ensures parity between Training (Simulated) and Production (Live) data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class MarketStateDTO:
    """Standardized market state for feature engineering"""
    # Current
    timestamp: pd.Timestamp
    price: float        # Intraday/Current Price
    spread: float
    dam_commitment_now: float # Current step DAM commitment

    # Forecast (Next Step)
    next_price: float

    # v21: Backward-looking statistics (past-only, no data leakage)
    # I replaced forward-looking price_max/min/mean_24h with backward stats
    price_max_24h: float   # Past 24h max (backward-looking)
    price_min_24h: float   # Past 24h min (backward-looking)
    price_mean_24h: float  # Past 24h mean (backward-looking)

    # Lookahead (DAM Commitments) - Extended to 12 hours
    dam_commitments: List[float]  # Next 12 hours

    # Lookahead (Price Forecasts) - NEW: 12 hours of prices
    price_lookahead: List[float] = None  # Next 12 hours of prices

    # Volatility (optional, for potential future use)
    volatility: float = 0.0

    # aFRR Market Stress Indicators
    # High aFRR_Up = system expects deficit = good for discharge
    # High aFRR_Down = system expects surplus = good for charge
    afrr_up_mw: float = 600.0
    afrr_down_mw: float = 130.0

    # v21: Trader-inspired signals (past-only, no leakage)
    # I compute these from historical data, not future prices
    price_vs_typical_hour: float = 0.0   # Current vs 30-day mean at this hour [-1, 1]
    trade_worthiness: float = 0.0        # 30-day avg daily spread / 100 [0, 1]

    def __post_init__(self):
        if self.price_lookahead is None:
            self.price_lookahead = [self.price] * 12


class FeatureEngineer:
    """
    Constructs the 19-float observation vector for the DRL agent.
    Ensures parity between Training (Simulated) and Production (Live) data.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Battery Specs (Defaults match typical training env)
        self.capacity_mwh = self.config.get('capacity_mwh', 146.0)
        self.max_power_mw = self.config.get('max_power_mw', 30.0)
        self.efficiency = self.config.get('efficiency', 0.94)
        self.time_step_hours = self.config.get('time_step_hours', 1.0)

        # v20: I control whether price awareness features are included
        self.include_price_awareness = self.config.get('include_price_awareness', True)

        # SoC constraints
        self.min_soc = 0.05
        self.max_soc = 0.95
        
        # Normalization constants
        self.NORM_PRICE = 100.0
        self.NORM_POWER = 30.0   # Usually normalized by 30MW (or max power)
        self.NORM_GRADIENT = 500.0
        self.NORM_RES = 200.0
        self.NORM_AFRR_UP = 1000.0    # Max aFRR Up ~1000 MW
        self.NORM_AFRR_DOWN = 200.0   # Max aFRR Down ~200 MW
    
    def calculate_shortfall_risk(
        self,
        battery_soc: float,
        dam_commitments: List[float],
        current_dam_commitment: float = 0.0  # I added current step commitment
    ) -> Dict:
        """
        I calculate the risk of failing to deliver on DAM commitments.

        This is the CRITICAL feature that tells the agent:
        "You have X MWh available, but you promised Y MWh in the next hours"

        I now include the CURRENT commitment, not just future ones!

        Returns:
            - upcoming_discharge_mwh: Total positive commitments in lookahead
            - available_energy_mwh: Energy available above min_soc
            - shortfall_risk: 0 if safe, >0 if at risk (capped at 1.0)
            - coverage_ratio: available / required (>1 = safe, <1 = risk)
        """
        eff_sqrt = np.sqrt(self.efficiency)

        # Include CURRENT commitment + future commitments
        all_commitments = [current_dam_commitment] + list(dam_commitments)

        # Sum of ALL DISCHARGE commitments (positive = selling)
        upcoming_discharge_mw = sum(max(0, c) for c in all_commitments)
        # Convert to MWh (assuming 1h steps for each commitment)
        upcoming_discharge_mwh = upcoming_discharge_mw * self.time_step_hours

        # Available energy above minimum SoC (with 5% safety margin)
        safety_margin = 0.05
        available_energy_mwh = max(0, (battery_soc - self.min_soc - safety_margin) * self.capacity_mwh)
        # Account for efficiency losses when discharging
        deliverable_energy_mwh = available_energy_mwh * eff_sqrt

        # Coverage ratio: how much of our commitments can we fulfill?
        if upcoming_discharge_mwh > 0.1:  # Avoid division by zero
            coverage_ratio = deliverable_energy_mwh / upcoming_discharge_mwh
        else:
            coverage_ratio = 10.0  # No commitments = very safe

        # I made risk calculation more sensitive
        # Shortfall risk: 0 if coverage >= 1.5 (50% buffer), scales up to 1.0
        if coverage_ratio >= 1.5:
            shortfall_risk = 0.0
        elif coverage_ratio >= 1.2:
            # Safe zone but watch out (120-150% coverage)
            shortfall_risk = (1.5 - coverage_ratio) / 0.3 * 0.15  # 0 to 0.15
        elif coverage_ratio >= 1.0:
            # Marginal safety zone (100-120% coverage)
            shortfall_risk = 0.15 + (1.2 - coverage_ratio) / 0.2 * 0.25  # 0.15 to 0.4
        elif coverage_ratio >= 0.7:
            # Danger zone (70-100% coverage)
            shortfall_risk = 0.4 + (1.0 - coverage_ratio) / 0.3 * 0.35  # 0.4 to 0.75
        else:
            # Critical zone (<70% coverage)
            shortfall_risk = 0.75 + (0.7 - coverage_ratio) / 0.7 * 0.25  # 0.75 to 1.0

        return {
            'upcoming_discharge_mwh': upcoming_discharge_mwh,
            'available_energy_mwh': deliverable_energy_mwh,
            'shortfall_risk': min(shortfall_risk, 1.0),
            'coverage_ratio': min(coverage_ratio, 10.0)  # Cap for normalization
        }

    def calculate_imbalance_risk(self, price_history: List[float], res_history: List[float], current_hour: int) -> Dict:
        """
        Calculate imbalance risk metrics.
        Same logic as originally in BatteryEnvMasked._calculate_imbalance_risk
        """
        # 1. Net Load Gradient (price change per hour)
        if len(price_history) >= 2:
            gradient = price_history[-1] - price_history[0]
        else:
            gradient = 0.0
            
        # 2. RES Deviation Trend
        res_deviation = 0.0
        if len(res_history) >= 2:
            current_res = res_history[-1]
            avg_res = np.mean(res_history[:-1])
            res_deviation = current_res - avg_res
            
        # 3. Risk Score Calculation
        risk_score = 0.0
        
        # Factor 1: Price gradient
        if gradient > 20:
            risk_score += min(0.4, gradient / 50.0)
        
        # Factor 2: Time of day (sunset risk)
        if 17 <= current_hour <= 20:
            risk_score += 0.3
        
        # Factor 3: Current price level
        current_price = price_history[-1] if len(price_history) > 0 else 0
        if current_price > 150:
            risk_score += min(0.3, (current_price - 150) / 100.0)
        
        # Factor 4: RES underperformance
        if res_deviation < -50:
            risk_score += min(0.2, abs(res_deviation) / 200.0)
            
        return {
            'gradient': gradient,
            'res_deviation': res_deviation,
            'risk_score': min(risk_score, 1.0)
        }

    def build_observation(
        self,
        market: MarketStateDTO,
        battery_soc: float,
        imbalance_info: Dict,
        shortfall_info: Dict = None
    ) -> np.ndarray:
        """
        Constructs the 22-feature observation vector used in training.

        I added aFRR market stress indicators as features 21-22.
        These help the agent anticipate system imbalance direction.

        Features:
        1. SoC
        2. max_discharge (normalized)
        3. max_charge (normalized)
        4. best_bid (normalized)
        5. best_ask (normalized)
        6. dam_commitment (normalized)
        7. hour_sin
        8. hour_cos
        9-20. price_lookahead (12 hours of prices)
        21-32. dam_lookahead (12 hours of commitments)
        33. net_load_gradient
        34. res_deviation_trend
        35. imbalance_risk (market-based)
        36. shortfall_risk (SoC vs commitments)
        37. aFRR_Up_MW (system deficit indicator)
        38. aFRR_Down_MW (system surplus indicator)

        Total: 38 features (was 22 with 4h lookahead)
        """
        
        # 1. Calculate Physical Limits (Dynamic based on SoC and Time Step)
        eff_sqrt = np.sqrt(self.efficiency)
        
        # Available energy in MWh
        available_discharge_mwh = max(0.0, (battery_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - battery_soc) * self.capacity_mwh)
        
        # Max power for this step
        max_discharge_mw = min(self.max_power_mw, (available_discharge_mwh * eff_sqrt) / self.time_step_hours)
        max_charge_mw = min(self.max_power_mw, (available_charge_mwh / eff_sqrt) / self.time_step_hours)
        
        # 2. Calculate Market Prices (Bid/Ask) derived from Price + Spread
        # Parity with Env logic: best_bid = max(0.1, price - spread/2)
        best_bid = max(0.1, market.price - market.spread / 2)
        best_ask = market.price + market.spread / 2
        
        # 3. Time Encoding
        hour = market.timestamp.hour + market.timestamp.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # 4. Construct Observation Vector
        obs = [
            battery_soc,                            # 1. SoC
            max_discharge_mw / self.max_power_mw,   # 2. Max Discharge
            max_charge_mw / self.max_power_mw,      # 3. Max Charge
            best_bid / 100.0,                       # 4. Best Bid
            best_ask / 100.0,                       # 5. Best Ask
            market.dam_commitment_now / 30.0,       # 6. DAM Commitment (Current)
            hour_sin,                               # 7. Hour Sin
            hour_cos,                               # 8. Hour Cos
        ]

        # 9-20. Price Lookahead (12 hours)
        price_lookahead = (market.price_lookahead + [market.price]*12)[:12]
        for val in price_lookahead:
            obs.append(val / 100.0)

        # 21-32. DAM Commitment Lookahead (12 hours)
        dam_lookahead = (market.dam_commitments + [0.0]*12)[:12]
        for val in dam_lookahead:
            obs.append(val / 30.0)
            
        # 17-19. Imbalance Risk Features
        obs.extend([
            imbalance_info['gradient'] / self.NORM_GRADIENT,      # 17. Gradient
            imbalance_info['res_deviation'] / self.NORM_RES,      # 18. RES Deviation
            imbalance_info['risk_score']                          # 19. Risk Score
        ])

        # 20. Shortfall Risk (CRITICAL for avoiding DAM failures)
        # I calculate this here if not provided, for backward compatibility
        if shortfall_info is None:
            shortfall_info = self.calculate_shortfall_risk(
                battery_soc=battery_soc,
                dam_commitments=market.dam_commitments
            )
        obs.append(shortfall_info['shortfall_risk'])              # 20. Shortfall Risk

        # 21-22. aFRR Market Stress Indicators
        # High aFRR_Up = system anticipates deficit = discharge opportunity
        # High aFRR_Down = system anticipates surplus = charge opportunity
        obs.append(market.afrr_up_mw / self.NORM_AFRR_UP)         # 21. aFRR Up
        obs.append(market.afrr_down_mw / self.NORM_AFRR_DOWN)     # 22. aFRR Down

        # 23-26. v13 TIMING FEATURES - I help agent understand when to sell
        # These derived features make lookahead information more actionable
        current_price = market.price
        max_future_price = max(price_lookahead) if price_lookahead else current_price

        # Price ratio: >1 means better prices coming, <1 means now is good
        price_ratio = max_future_price / max(current_price, 1.0)
        obs.append(min(price_ratio, 3.0) / 3.0)  # 23. Normalized price ratio (cap at 3x)

        # Hours to peak: when is the best price?
        if max_future_price > current_price and price_lookahead:
            hours_to_peak = price_lookahead.index(max_future_price) + 1
        else:
            hours_to_peak = 0  # We're at peak now
        obs.append(hours_to_peak / 12.0)  # 24. Normalized hours to peak

        # Is at peak: binary signal - should I sell now?
        is_at_peak = 1.0 if current_price >= max_future_price * 0.95 else 0.0
        obs.append(is_at_peak)  # 25. At peak indicator

        # DAM slot comparison: if DAM discharge coming, compare prices
        # Find first DAM discharge commitment in lookahead
        dam_slot_ratio = 1.0  # Default: neutral
        for i, dam_val in enumerate(dam_lookahead):
            if dam_val > 0.1:  # Positive = discharge commitment
                dam_slot_price = price_lookahead[i] if i < len(price_lookahead) else current_price
                dam_slot_ratio = dam_slot_price / max(current_price, 1.0)
                break
        obs.append(min(dam_slot_ratio, 2.0) / 2.0)  # 26. DAM slot ratio (cap at 2x)

        # 27-28. v21 TRADER-INSPIRED SIGNALS (conditional, past-only)
        # I replaced v20's leaky price_percentile/price_vs_mean with causal features.
        # These use ONLY past data — no future price information leaks into observations.
        if self.include_price_awareness:
            # Feature 43: price_vs_typical_hour
            # I compare current price to the 30-day mean at this hour of day.
            # A trader would know "electricity at 18:00 usually costs 85€, now it's 120€".
            # Positive = expensive for this hour (sell), Negative = cheap (buy).
            price_vs_typical = max(-1.0, min(1.0, market.price_vs_typical_hour))
            obs.append(price_vs_typical)  # 27. Price vs typical hour [-1, 1]

            # Feature 44: trade_worthiness
            # I use the 30-day average daily spread (max-min per day) as a measure of
            # "is today a good day to trade?". High spread = volatile = profitable day.
            # Normalized by 100€ to keep in [0, 1] range.
            trade_worth = max(0.0, min(1.0, market.trade_worthiness))
            obs.append(trade_worth)  # 28. Trade worthiness [0, 1]

        return np.array(obs, dtype=np.float32)
