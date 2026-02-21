"""
Intraday Re-forecast & Position Correction (IDA1/IDA2/IDA3)

I re-run price forecasts before each IDA session with updated data,
detect deviations from the original DAM commitment, and submit
corrective orders through the IntraDayTrader when profitable.

Timeline:
- D-1 ~14:00: Greek DAM actual results published
- D-1 15:00: IDA1 gate closure - first corrective orders
- D-1 22:00: IDA2 gate closure - better info (IDA1 results, XBID, evening load)
- D  10:00: IDA3 gate closure - covers only ISPs 12:00-23:45 of delivery day
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from orchestrator.interfaces import BatteryState
from orchestrator.intraday_trader import IntraDayTrader, IntraDayOrder
from models.market_forecaster import DAMDayAheadForecaster


# I define the valid IDA sessions and their hour ranges
IDA_SESSIONS = {
    'IDA1': {'hour_range': range(0, 24), 'gate_hour': 15},
    'IDA2': {'hour_range': range(0, 24), 'gate_hour': 22},
    'IDA3': {'hour_range': range(12, 24), 'gate_hour': 10},
}

# I define HEnEx fee in EUR/MWh
HENEX_FEE_EUR_MWH = 0.04


@dataclass
class CorrectionOpportunity:
    """I represent a single hour's correction opportunity."""
    hour: int
    original_commitment_mw: float
    recommended_commitment_mw: float
    correction_mw: float              # recommended - original
    original_price_forecast: float
    updated_price_forecast: float
    price_deviation_eur: float
    p10_price: float
    p90_price: float
    expected_gain_eur: float
    confidence: float                 # 0-1, from quantile spread


@dataclass
class CorrectionResult:
    """I represent the outcome of a correction cycle for one IDA session."""
    session: str                      # 'IDA1', 'IDA2', 'IDA3'
    trigger_time: datetime
    n_opportunities: int
    n_corrections_applied: int
    total_correction_mw: float
    expected_pnl_eur: float
    orders: List[IntraDayOrder] = field(default_factory=list)


class PositionCorrector:
    """
    I detect deviations between original DAM forecasts and updated forecasts,
    then submit corrective IntraDay orders when profitable.

    I run once before each IDA gate closure (IDA1, IDA2, IDA3) with
    progressively better information.
    """

    def __init__(
        self,
        day_ahead_forecaster: DAMDayAheadForecaster,
        intraday_trader: IntraDayTrader,
        min_price_deviation_eur: float = 8.0,
        min_expected_gain_eur: float = 15.0,
        max_correction_per_session_mw: float = 10.0,
        max_correction_per_hour_mw: float = 5.0,
        confidence_threshold: float = 0.6,
        max_power_mw: float = 30.0,
    ):
        """
        Args:
            day_ahead_forecaster: I use this to re-forecast prices with updated data
            intraday_trader: I use this to create and fill corrective orders
            min_price_deviation_eur: Minimum price change to trigger correction
            min_expected_gain_eur: Minimum profit after transaction costs
            max_correction_per_session_mw: Cumulative limit per IDA session
            max_correction_per_hour_mw: Per-hour cap
            confidence_threshold: I reject corrections below this confidence
            max_power_mw: Maximum battery power (for position limits)
        """
        self.forecaster = day_ahead_forecaster
        self.trader = intraday_trader
        self.min_price_deviation_eur = min_price_deviation_eur
        self.min_expected_gain_eur = min_expected_gain_eur
        self.max_correction_per_session_mw = max_correction_per_session_mw
        self.max_correction_per_hour_mw = max_correction_per_hour_mw
        self.confidence_threshold = confidence_threshold
        self.max_power_mw = max_power_mw

    def run_correction_cycle(
        self,
        session: str,
        target_date: date,
        df: pd.DataFrame,
        serbia_24h: np.ndarray,
        original_forecast: Dict[str, np.ndarray],
        original_commitment: np.ndarray,
        battery_state: BatteryState,
        sph: int = 1,
        ida1_results: Optional[np.ndarray] = None,
        dam_actuals: Optional[np.ndarray] = None,
    ) -> CorrectionResult:
        """
        I run one full correction cycle for an IDA session.

        Args:
            session: 'IDA1', 'IDA2', or 'IDA3'
            target_date: Delivery date
            df: Historical data DataFrame
            serbia_24h: 24 hourly Serbia D+1 prices
            original_forecast: Dict with 'hourly', 'p10', 'p90' from original DAM forecast
            original_commitment: 24 hourly commitment values (positive=sell, negative=buy)
            battery_state: Current battery state
            sph: Steps per hour
            ida1_results: 24 hourly IDA1 clearing prices (for IDA2/IDA3)
            dam_actuals: 24 hourly actual DAM prices (known after D-1 14:00)
        """
        if session not in IDA_SESSIONS:
            raise ValueError(f"Invalid session '{session}'. Must be one of {list(IDA_SESSIONS.keys())}")

        session_config = IDA_SESSIONS[session]
        hour_range = session_config['hour_range']

        # I re-forecast with updated information
        updated_forecast = self._reforecast(
            session=session,
            target_date=target_date,
            df=df,
            serbia_24h=serbia_24h,
            sph=sph,
            ida1_results=ida1_results,
            dam_actuals=dam_actuals,
        )

        # I detect deviations between original and updated forecast
        opportunities = self._detect_deviations(
            original_forecast=original_forecast,
            updated_forecast=updated_forecast,
            original_commitment=original_commitment,
            battery_state=battery_state,
            hour_range=hour_range,
        )

        # I apply corrections for the best opportunities
        current_time = datetime.combine(
            target_date - timedelta(days=1) if session != 'IDA3' else target_date,
            datetime.min.time().replace(hour=session_config['gate_hour'])
        )

        orders = self._apply_corrections(
            opportunities=opportunities,
            original_commitment=original_commitment,
            battery_state=battery_state,
            session=session,
            current_time=current_time,
        )

        # I compute totals
        total_correction = sum(abs(o.power_mw) for o in orders)
        expected_pnl = sum(
            opp.expected_gain_eur
            for opp in opportunities[:len(orders)]
        )

        return CorrectionResult(
            session=session,
            trigger_time=current_time,
            n_opportunities=len(opportunities),
            n_corrections_applied=len(orders),
            total_correction_mw=total_correction,
            expected_pnl_eur=expected_pnl,
            orders=orders,
        )

    def _reforecast(
        self,
        session: str,
        target_date: date,
        df: pd.DataFrame,
        serbia_24h: np.ndarray,
        sph: int = 1,
        ida1_results: Optional[np.ndarray] = None,
        dam_actuals: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        I re-run the price forecast with updated data injected into df.

        IDA1: I inject Greek DAM actuals (known since ~14:00)
        IDA2: I inject DAM actuals + blend with IDA1 results (70% IDA1 / 30% model)
        IDA3: I inject DAM actuals + IDA1 results + morning actuals
        """
        # I create a copy so I don't mutate the original data
        df_updated = df.copy()

        # I find the target day in the data
        day_mask = df_updated.index.normalize() == pd.Timestamp(target_date)
        day_indices = np.where(day_mask.values)[0] if hasattr(day_mask, 'values') else np.where(day_mask)[0]

        if len(day_indices) == 0:
            # I fall back to predicting from the last available data
            day_idx = len(df_updated) - 24 * sph
        else:
            day_idx = day_indices[0]

        # I inject updated information based on session
        if dam_actuals is not None:
            # I inject actual DAM prices into the data - this improves the forecast
            # because the model can see the realized prices as recent history
            for h in range(24):
                row_idx = day_idx + h * sph
                if row_idx < len(df_updated):
                    df_updated.iloc[row_idx, df_updated.columns.get_loc('price')] = dam_actuals[h]

        if session in ('IDA2', 'IDA3') and ida1_results is not None:
            # I blend IDA1 results into the price column for better context
            blend_weight = 0.7 if session == 'IDA2' else 0.5
            for h in range(24):
                row_idx = day_idx + h * sph
                if row_idx < len(df_updated):
                    current = df_updated.iloc[row_idx]['price']
                    blended = blend_weight * ida1_results[h] + (1 - blend_weight) * current
                    df_updated.iloc[row_idx, df_updated.columns.get_loc('price')] = blended

        # I re-run the DAM forecaster on the updated data
        result = self.forecaster.predict_day(df_updated, day_idx, serbia_24h, sph)

        # I restrict IDA3 to afternoon hours only (12-23)
        if session == 'IDA3':
            for key in ('hourly', 'p10', 'p90'):
                arr = result[key].copy()
                arr[:12] = np.nan
                result[key] = arr

        return result

    def _detect_deviations(
        self,
        original_forecast: Dict[str, np.ndarray],
        updated_forecast: Dict[str, np.ndarray],
        original_commitment: np.ndarray,
        battery_state: BatteryState,
        hour_range: range,
    ) -> List[CorrectionOpportunity]:
        """
        I compare hourly forecasts and identify profitable correction opportunities.

        I filter by min_price_deviation_eur and confidence threshold,
        then sort by expected gain descending.
        """
        opportunities = []

        for hour in hour_range:
            original_price = original_forecast['hourly'][hour]
            updated_price = updated_forecast['hourly'][hour]

            # I skip NaN values (IDA3 morning hours)
            if np.isnan(updated_price):
                continue

            price_deviation = updated_price - original_price

            # I check if deviation is large enough
            if abs(price_deviation) < self.min_price_deviation_eur:
                continue

            # I compute confidence from quantile spread
            p10 = updated_forecast['p10'][hour]
            p90 = updated_forecast['p90'][hour]
            if np.isnan(p10) or np.isnan(p90):
                continue

            quantile_spread = p90 - p10
            # I define confidence as inverse of relative spread
            # Narrow spread = high confidence
            if quantile_spread > 0 and updated_price > 0:
                confidence = 1.0 - min(quantile_spread / (2 * abs(updated_price)), 1.0)
            else:
                confidence = 0.5

            # I reject low-confidence corrections
            if confidence < self.confidence_threshold:
                continue

            # I compute the optimal commitment adjustment
            # If price went up, I should sell more (positive correction)
            # If price went down, I should buy more (negative correction)
            original_commit = original_commitment[hour]

            # I compute recommended commitment based on price direction
            if price_deviation > 0:
                # Price went up - I should sell more or reduce buying
                correction = min(abs(price_deviation) / 20.0 * self.max_correction_per_hour_mw,
                                 self.max_correction_per_hour_mw)
            else:
                # Price went down - I should buy more or reduce selling
                correction = -min(abs(price_deviation) / 20.0 * self.max_correction_per_hour_mw,
                                  self.max_correction_per_hour_mw)

            recommended_commit = original_commit + correction

            # I compute expected gain: |deviation| * |correction| * 0.25h - tx_cost
            energy_mwh = abs(correction) * 0.25  # 15-min equivalent
            tx_cost = abs(correction) * abs(updated_price) * 0.02 * 0.25 + HENEX_FEE_EUR_MWH * energy_mwh
            expected_gain = abs(price_deviation) * energy_mwh - tx_cost

            if expected_gain < self.min_expected_gain_eur:
                continue

            opportunities.append(CorrectionOpportunity(
                hour=hour,
                original_commitment_mw=original_commit,
                recommended_commitment_mw=recommended_commit,
                correction_mw=correction,
                original_price_forecast=original_price,
                updated_price_forecast=updated_price,
                price_deviation_eur=price_deviation,
                p10_price=p10,
                p90_price=p90,
                expected_gain_eur=expected_gain,
                confidence=confidence,
            ))

        # I sort by expected gain descending
        opportunities.sort(key=lambda x: x.expected_gain_eur, reverse=True)
        return opportunities

    def _apply_corrections(
        self,
        opportunities: List[CorrectionOpportunity],
        original_commitment: np.ndarray,
        battery_state: BatteryState,
        session: str,
        current_time: datetime,
    ) -> List[IntraDayOrder]:
        """
        I iterate sorted opportunities, apply per-hour and per-session limits,
        enforce total position within +/-max_power_mw, and create orders.
        """
        orders = []
        session_total_mw = 0.0
        corrected_hours = set()

        for opp in opportunities:
            # I check per-session cumulative limit
            if session_total_mw + abs(opp.correction_mw) > self.max_correction_per_session_mw:
                remaining = self.max_correction_per_session_mw - session_total_mw
                if remaining < 0.5:
                    break
                # I clip the correction to fit within remaining budget
                correction = np.sign(opp.correction_mw) * remaining
            else:
                correction = opp.correction_mw

            # I enforce per-hour limit
            correction = np.clip(correction,
                                 -self.max_correction_per_hour_mw,
                                 self.max_correction_per_hour_mw)

            # I enforce total position (DAM + correction) within +/-max_power_mw
            new_total = opp.original_commitment_mw + correction
            if abs(new_total) > self.max_power_mw:
                correction = np.sign(correction) * max(
                    0, self.max_power_mw - abs(opp.original_commitment_mw)
                )

            if abs(correction) < 0.1:
                continue

            # I skip hours already corrected
            if opp.hour in corrected_hours:
                continue

            # I create delivery time for this hour
            delivery_date = current_time.date()
            if session != 'IDA3':
                delivery_date = delivery_date + timedelta(days=1)
            delivery_time = datetime.combine(
                delivery_date,
                datetime.min.time().replace(hour=opp.hour)
            )

            # I compute price limit based on direction and updated forecast
            if correction > 0:
                # Selling - I set price slightly below updated forecast
                price_limit = opp.updated_price_forecast * 0.98
            else:
                # Buying - I set price slightly above updated forecast
                price_limit = opp.updated_price_forecast * 1.02

            # I create order via the trader
            order = self.trader.create_order(
                delivery_time=delivery_time,
                power_mw=correction,
                price_limit=price_limit,
                order_type="limit",
            )

            # I simulate fill at the updated price with session-specific cost
            tx_cost_pct = self._compute_transaction_cost(
                session, abs(correction), opp.updated_price_forecast
            )
            fill_price = opp.updated_price_forecast * (1 + tx_cost_pct * np.sign(correction))
            self.trader.fill_order(order, fill_price)

            orders.append(order)
            session_total_mw += abs(correction)
            corrected_hours.add(opp.hour)

        return orders

    def _compute_transaction_cost(
        self,
        session: str,
        power_mw: float,
        price: float,
    ) -> float:
        """
        I compute transaction cost as a fraction of price.

        IDA1: ~2% (furthest from delivery, most uncertainty)
        IDA2: ~1.5% (better information)
        IDA3: ~1% (closest to delivery)

        Plus HEnEx fee ~0.04 EUR/MWh.
        """
        session_costs = {
            'IDA1': 0.02,
            'IDA2': 0.015,
            'IDA3': 0.01,
        }
        base_pct = session_costs.get(session, 0.02)

        # I add the fixed fee as a percentage
        if abs(price) > 0:
            fee_pct = HENEX_FEE_EUR_MWH / abs(price)
        else:
            fee_pct = 0.0

        return base_pct + fee_pct
