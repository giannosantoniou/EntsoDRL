"""
DAM Forecaster - D+1 Price Forecasting for Greek DAM

I provide DAM price forecasts using either:
1. DAMDayAheadForecaster (LightGBM, trained on historical data)
2. Serbia regression fallback (hourly linear regression)

Output: PriceForecast with 96 values (24h x 4 quarters) at 15-min resolution
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

from .interfaces import IDAMForecaster, PriceForecast


# Hourly regression parameters: Greek = slope * Serbia + intercept
# Based on 6-month correlation analysis (R^2 varies 0.28-0.60 by hour)
HOURLY_GR_RS_PARAMS = {
    0:  {'slope': 0.8649, 'intercept':  21.93},  # Night
    1:  {'slope': 0.6405, 'intercept':  38.62},
    2:  {'slope': 0.6709, 'intercept':  33.56},  # Best R^2=0.59
    3:  {'slope': 0.6249, 'intercept':  35.26},  # Best R^2=0.60
    4:  {'slope': 0.5040, 'intercept':  41.91},
    5:  {'slope': 0.4231, 'intercept':  43.88},  # Morning ramp starts
    6:  {'slope': 0.2152, 'intercept':  64.67},  # Low correlation
    7:  {'slope': 0.2974, 'intercept':  67.46},  # Low correlation
    8:  {'slope': 0.8470, 'intercept':  32.92},  # Morning peak
    9:  {'slope': 0.9424, 'intercept':  13.57},
    10: {'slope': 0.8600, 'intercept':   5.32},
    11: {'slope': 0.7031, 'intercept':   2.39},  # Solar starts
    12: {'slope': 0.5603, 'intercept':   5.24},  # Solar peak - GR much cheaper
    13: {'slope': 0.5545, 'intercept':   2.45},  # Solar peak
    14: {'slope': 0.5263, 'intercept':   2.97},  # Solar peak
    15: {'slope': 0.4407, 'intercept':  12.58},
    16: {'slope': 0.3844, 'intercept':  24.74},  # Evening ramp
    17: {'slope': 0.4893, 'intercept':  37.96},
    18: {'slope': 0.3996, 'intercept':  74.02},  # Low correlation
    19: {'slope': 0.5689, 'intercept':  58.52},
    20: {'slope': 1.3837, 'intercept': -25.64},  # Evening peak - GR spikes!
    21: {'slope': 1.2570, 'intercept':   5.70},  # Evening peak
    22: {'slope': 0.9221, 'intercept':  29.32},
    23: {'slope': 0.7662, 'intercept':  43.76},
}


class DAMForecaster(IDAMForecaster):
    """DAM Price Forecaster backed by DAMDayAheadForecaster or Serbia regression.

    Key insight: Serbia DAM clears ~12:00, before Greek gate closure at 13:00.
    I use Serbia D+1 prices as a strong feature for Greek price prediction.

    Hourly patterns (from correlation analysis):
    - Solar hours (11-16h): Greek much cheaper than Serbia (more solar capacity)
    - Evening peak (20-21h): Greek spikes higher than Serbia
    - Night hours: Similar prices, good correlation
    """

    def __init__(
        self,
        day_ahead_model_path: Optional[str] = None,
        historical_df: Optional[pd.DataFrame] = None,
        spread_percent: float = 0.02,
    ):
        """Initialize the forecaster.

        Args:
            day_ahead_model_path: Path to trained DAMDayAheadForecaster pickle
            historical_df: Historical Greek DAM DataFrame (for feature computation)
            spread_percent: Bid-ask spread as percentage of price
        """
        self.spread_percent = spread_percent
        self._day_ahead_forecaster = None
        self._historical_df = historical_df
        self._serbia_prices = None

        if day_ahead_model_path is not None:
            model_path = Path(day_ahead_model_path)
            if model_path.exists():
                try:
                    from models.market_forecaster import DAMDayAheadForecaster
                    self._day_ahead_forecaster = DAMDayAheadForecaster.load(str(model_path))
                    print(f"DAM Forecaster: loaded DAMDayAheadForecaster from {model_path}")
                except Exception as e:
                    print(f"DAM Forecaster: could not load D+1 model ({e}), using regression fallback")

        print(f"DAM Forecaster initialized (spread={spread_percent:.1%})")

    def forecast(
        self,
        target_date: date,
        serbia_24h: Optional[np.ndarray] = None,
    ) -> PriceForecast:
        """Generate price forecast for target date.

        Args:
            target_date: Date to forecast
            serbia_24h: Optional 24 hourly Serbia D+1 prices

        Returns:
            PriceForecast with 96 buy/sell prices
        """
        # I generate 96 timestamps (15-min intervals)
        timestamps = []
        base_dt = datetime.combine(target_date, datetime.min.time())
        for i in range(96):
            timestamps.append(base_dt + timedelta(minutes=15 * i))

        if serbia_24h is None:
            serbia_24h = self.fetch_serbia_dam(target_date)

        # I try DAMDayAheadForecaster first, then regression fallback
        if (
            self._day_ahead_forecaster is not None
            and self._day_ahead_forecaster.is_trained
            and self._historical_df is not None
            and serbia_24h is not None
            and len(serbia_24h) == 24
        ):
            sph = 1
            if len(self._historical_df) > 1:
                delta = (self._historical_df.index[1] - self._historical_df.index[0]).total_seconds()
                sph = max(1, int(round(3600 / delta)))

            # I find the day index in historical data
            day_mask = self._historical_df.index.normalize() == pd.Timestamp(target_date)
            day_indices = np.where(day_mask.values)[0] if hasattr(day_mask, 'values') else np.where(day_mask)[0]

            if len(day_indices) > 0:
                day_idx = day_indices[0]
                result = self._day_ahead_forecaster.predict_96qh(
                    self._historical_df, day_idx, serbia_24h, sph
                )
                prices_15min = result['qh']
            else:
                prices_15min = self._regression_forecast_96qh(target_date, serbia_24h)
        elif serbia_24h is not None and len(serbia_24h) == 24:
            prices_15min = self._regression_forecast_96qh(target_date, serbia_24h)
        else:
            prices_15min = np.repeat(self._generate_fallback_forecast(target_date), 4)

        # I calculate buy/sell prices with spread
        half_spread = prices_15min * self.spread_percent / 2
        buy_prices = prices_15min + half_spread
        sell_prices = prices_15min - half_spread

        return PriceForecast(
            target_date=target_date,
            timestamps=timestamps,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
        )

    def _regression_forecast_96qh(
        self,
        target_date: date,
        serbia_24h: np.ndarray,
    ) -> np.ndarray:
        """I generate 96 QH forecast using Serbia regression + cubic spline."""
        hourly = np.zeros(24)
        for h in range(24):
            params = HOURLY_GR_RS_PARAMS[h]
            hourly[h] = serbia_24h[h] * params['slope'] + params['intercept']
        hourly = np.maximum(hourly, 0.0)

        # I use cubic spline for smooth 96 QH expansion
        try:
            from scipy.interpolate import CubicSpline
            hour_midpoints = np.arange(24) * 4 + 1.5
            extended_x = np.concatenate([[-2.5], hour_midpoints, [97.5]])
            extended_y = np.concatenate([[hourly[-1]], hourly, [hourly[0]]])
            cs = CubicSpline(extended_x, extended_y, bc_type='not-a-knot')
            qh = np.maximum(cs(np.arange(96)), 0.0)
            return qh
        except ImportError:
            return np.repeat(hourly, 4)

    def fetch_serbia_dam(self, target_date: date) -> Optional[np.ndarray]:
        """Fetch Serbia DAM prices for D+1.

        Serbia DAM clears ~12:00 Greek time, before Greek gate closure at 13:00.
        This gives us a strong signal for Greek prices.

        Args:
            target_date: The delivery date (tomorrow)

        Returns:
            24 hourly prices or None if not available
        """
        try:
            # Import connector
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.entsoe_connector import EntsoeConnector

            connector = EntsoeConnector()

            # Fetch Serbia D+1
            start = pd.Timestamp(target_date, tz='Europe/Athens').normalize()
            end = start + pd.Timedelta(days=1)

            df = connector.fetch_neighbor_dam_prices('RS', start, end)

            if df is not None and not df.empty:
                # Resample to hourly if needed
                if len(df) > 24:
                    prices = df['price'].resample('h').mean().values[:24]
                else:
                    prices = df['price'].values[:24]

                self._serbia_prices = prices
                print(f"  Fetched Serbia DAM: {prices.min():.1f} - {prices.max():.1f} EUR")
                return prices

        except Exception as e:
            print(f"  Could not fetch Serbia DAM: {e}")

        return None

    def _generate_fallback_forecast(self, target_date: date) -> np.ndarray:
        """Generate fallback forecast based on typical daily pattern.

        I use a realistic daily price pattern for the Greek market:
        - Low prices: 02:00-06:00 (night), 12:00-14:00 (solar peak)
        - High prices: 08:00-10:00 (morning), 18:00-21:00 (evening peak)

        If Serbia DAM prices are available, I use them to adjust the forecast:
        - Greek prices are typically correlated with Serbia (~0.7-0.8 correlation)
        - Greek prices tend to be slightly higher than Serbia (offset ~5-15 EUR)
        """
        # Try to fetch Serbia prices first
        serbia_prices = self.fetch_serbia_dam(target_date)

        if serbia_prices is not None and len(serbia_prices) == 24:
            # Use hourly-specific regression parameters
            # Each hour has different GR-RS relationship due to:
            # - Solar generation (GR cheaper midday)
            # - Evening ramp (GR spikes more at 20-21h)
            hourly_pattern = np.zeros(24)

            for h in range(24):
                params = HOURLY_GR_RS_PARAMS[h]
                hourly_pattern[h] = serbia_prices[h] * params['slope'] + params['intercept']

            # Add small noise (±3% since we have calibrated model)
            noise = np.random.uniform(0.97, 1.03, 24)
            hourly_pattern *= noise

            # Ensure non-negative prices
            hourly_pattern = np.maximum(hourly_pattern, 0)

            print(f"  Using Serbia DAM with hourly regression model")
            return hourly_pattern

        # Fallback: Typical daily pattern (24 hourly values)
        # Based on Greek market historical averages
        hourly_pattern = np.array([
            85,   # 00:00 - night
            80,   # 01:00
            75,   # 02:00 - low
            72,   # 03:00 - lowest
            73,   # 04:00
            78,   # 05:00
            90,   # 06:00 - morning ramp
            110,  # 07:00
            130,  # 08:00 - morning peak
            125,  # 09:00
            115,  # 10:00
            100,  # 11:00
            85,   # 12:00 - solar dip
            80,   # 13:00 - solar peak (low prices)
            85,   # 14:00
            95,   # 15:00
            110,  # 16:00 - evening ramp
            135,  # 17:00
            155,  # 18:00 - evening peak
            160,  # 19:00 - peak
            150,  # 20:00
            130,  # 21:00
            110,  # 22:00
            95,   # 23:00
        ], dtype=np.float64)

        # Add day-of-week variation
        dow = target_date.weekday()
        if dow >= 5:  # Weekend
            hourly_pattern *= 0.85  # Lower weekend prices

        # Add some randomness (±10%)
        noise = np.random.uniform(0.9, 1.1, 24)
        hourly_pattern *= noise

        return hourly_pattern

    def save_forecast(self, forecast: PriceForecast, path: str) -> None:
        """Save forecast to CSV file.

        Format: 2 rows × 96 columns
        Row 1: Buy prices
        Row 2: Sell prices
        """
        # Create DataFrame with timestamps as columns
        df = pd.DataFrame({
            'type': ['buy', 'sell'],
        })

        for i, ts in enumerate(forecast.timestamps):
            col_name = ts.strftime('%H:%M')
            df[col_name] = [forecast.buy_prices[i], forecast.sell_prices[i]]

        df.to_csv(path, index=False)
        print(f"  Saved forecast to {path}")

    def load_forecast(self, path: str) -> PriceForecast:
        """Load forecast from CSV file."""
        df = pd.read_csv(path)

        # Extract prices
        price_cols = [c for c in df.columns if c != 'type']
        buy_prices = df[df['type'] == 'buy'][price_cols].values.flatten()
        sell_prices = df[df['type'] == 'sell'][price_cols].values.flatten()

        # Generate timestamps (assume today if not specified)
        target_date = date.today()
        timestamps = []
        base_dt = datetime.combine(target_date, datetime.min.time())
        for i in range(96):
            timestamps.append(base_dt + timedelta(minutes=15 * i))

        return PriceForecast(
            target_date=target_date,
            timestamps=timestamps,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
        )


# Test
if __name__ == "__main__":
    forecaster = DAMForecaster()

    # Generate forecast for tomorrow
    tomorrow = date.today() + timedelta(days=1)
    forecast = forecaster.forecast(tomorrow)

    print(f"\nForecast for {tomorrow}:")
    print(f"  Intervals: {forecast.n_intervals}")
    print(f"  Buy prices: {forecast.buy_prices.min():.2f} - {forecast.buy_prices.max():.2f} EUR")
    print(f"  Sell prices: {forecast.sell_prices.min():.2f} - {forecast.sell_prices.max():.2f} EUR")

    # Get hourly summary
    buy_h, sell_h = forecast.get_hourly_prices()
    print(f"\nHourly summary:")
    for h in range(24):
        print(f"  {h:02d}:00  Buy={buy_h[h]:.1f}  Sell={sell_h[h]:.1f}")

    # Save to CSV
    forecaster.save_forecast(forecast, "test_forecast.csv")
