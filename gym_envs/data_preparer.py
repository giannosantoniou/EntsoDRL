"""
Data Preparation Module for Unified Battery Environment

I handle all data validation, column defaults, rolling statistics,
and precomputation of O(1) lookup arrays. This was previously the
_prepare_data() method and precomputation logic in BatteryEnvUnified.__init__().
"""

import numpy as np
import pandas as pd
from typing import Optional

from gym_envs.market_executors import PrecomputedArrays


class DataPreparer:
    """I validate, enrich, and precompute training data for the unified env.

    I extract all data preparation logic from BatteryEnvUnified so the env
    can focus on simulation. I return the prepared DataFrame and precomputed
    arrays for O(1) step-level lookups.
    """

    def __init__(
        self,
        sph: int,
        mfrr_imbalance_threshold: float,
        enable_forecast: bool = False,
        warmup_steps: int = 48,
        lookahead_hours: int = 12,
        time_step_hours: float = 1.0,
    ):
        self._sph = sph
        self._mfrr_imbalance_threshold = mfrr_imbalance_threshold
        self._enable_forecast = enable_forecast
        self._warmup_steps = warmup_steps
        self._lookahead_hours = lookahead_hours
        self._time_step_hours = time_step_hours

    def prepare(self, df: pd.DataFrame) -> tuple:
        """I prepare and validate the training data.

        Returns:
            (df, precomputed, res_mean_24h, start_step, max_steps)
        """
        # I ensure datetime index (handle various formats including timezone-aware)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='mixed', utc=True)
                df.index = df.index.tz_convert('Europe/Athens')
            except (ValueError, TypeError):
                df.index = pd.to_datetime(df.index)

        # I ensure required columns exist
        required = ['price', 'dam_commitment']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # I add aFRR/mFRR/IntraDay columns if missing (with safe constant defaults
        # that do NOT depend on DAM price to avoid data leakage)
        defaults = {
            'afrr_up': 80.0,
            'afrr_down': 80.0,
            'afrr_cap_up_price': 20.0,
            'afrr_cap_down_price': 30.0,
            'intraday_bid': 98.0,
            'intraday_ask': 102.0,
            'intraday_spread': 4.0,
            'intraday_volume': 0.5,
            'mfrr_price_up': 120.0,
            'mfrr_price_down': 60.0,
            'mfrr_spread': 60.0,
            'net_imbalance_mw': 0.0
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default() if callable(default) else default

        # I reconstruct net_imbalance_mw from activated balancing energy when
        # the column exists but is all-zero (data pipeline gap before Sep 2025).
        imb = df['net_imbalance_mw']
        if imb.abs().max() < 1.0 and 'net_balancing_energy_mwh' in df.columns:
            df['net_imbalance_mw'] = df['net_balancing_energy_mwh']
            n_active = (df['net_imbalance_mw'].abs() > self._mfrr_imbalance_threshold).sum()
            print(f"  Reconstructed net_imbalance_mw from net_balancing_energy_mwh "
                  f"({n_active}/{len(df)} rows exceed {self._mfrr_imbalance_threshold} MW threshold)")

        # I add time features if missing
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek

        # I add lagged features (backward-looking only!)
        lag_cols = ['price', 'afrr_up', 'afrr_down', 'mfrr_price_up', 'mfrr_price_down',
                    'intraday_bid', 'intraday_ask', 'intraday_spread', 'intraday_volume']
        for col in lag_cols:
            if col in df.columns and f'{col}_lag_1h' not in df.columns:
                df[f'{col}_lag_1h'] = df[col].shift(1)

        # I scale rolling windows by steps-per-hour for resolution independence
        w24 = 24 * self._sph
        w168 = 168 * self._sph
        lag6 = 6 * self._sph

        # I add rolling statistics (backward-looking only!)
        if 'price_mean_24h' not in df.columns:
            df['price_mean_24h'] = df['price'].rolling(w24, min_periods=1).mean()
        if 'price_std_24h' not in df.columns:
            df['price_std_24h'] = df['price'].rolling(w24, min_periods=1).std().fillna(10)
        if 'price_min_24h' not in df.columns:
            df['price_min_24h'] = df['price'].rolling(w24, min_periods=1).min()
        if 'price_max_24h' not in df.columns:
            df['price_max_24h'] = df['price'].rolling(w24, min_periods=1).max()

        # I compute price momentum (backward-looking)
        if 'price_momentum' not in df.columns:
            price_6h_ago = df['price'].shift(lag6)
            df['price_momentum'] = (df['price'] - price_6h_ago) / (price_6h_ago + 1)
            df['price_momentum'] = df['price_momentum'].fillna(0).clip(-1, 1)

        # I compute aFRR capacity price percentile
        if 'afrr_cap_percentile' not in df.columns:
            df['afrr_cap_percentile'] = df['afrr_cap_up_price'].rolling(
                w168, min_periods=w24
            ).rank(pct=True).fillna(0.5)

        # I pre-compute RES 24h rolling mean for forecast deviation feature
        res_mean_24h = None
        if self._enable_forecast:
            if 'res_total_mw' in df.columns:
                res_mean_24h = df['res_total_mw'].rolling(w24, min_periods=1).mean().values
            elif 'solar' in df.columns and 'wind_onshore' in df.columns:
                res = df['solar'].fillna(0) + df['wind_onshore'].fillna(0)
                res_mean_24h = res.rolling(w24, min_periods=1).mean().values

        # I fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # I build precomputed arrays
        precomputed = self._build_precomputed_arrays(df)

        # I calculate step limits
        lookahead_steps = int(self._lookahead_hours / self._time_step_hours)
        start_step = self._warmup_steps
        max_steps = len(df) - 1 - lookahead_steps

        return df, precomputed, res_mean_24h, start_step, max_steps

    def _build_precomputed_arrays(self, df: pd.DataFrame) -> PrecomputedArrays:
        """I precompute date-based mappings for O(1) step-level lookups.

        Previously, _get_dam_commitment() called `self.df.index.date == current_day`
        on every step, converting all 57k timestamps. This was 90% of step time.
        """
        dates = df.index.date
        n = len(df)

        step_day_offset = np.zeros(n, dtype=np.int32)
        step_day_id = np.zeros(n, dtype=np.int32)
        current_date = None
        day_id = -1
        day_start_idx = 0
        day_starts = []

        for i, d in enumerate(dates):
            if d != current_date:
                current_date = d
                day_id += 1
                day_start_idx = i
                day_starts.append(i)
            step_day_offset[i] = i - day_start_idx
            step_day_id[i] = day_id

        # I build day start/length arrays for O(1) day_indices lookup
        n_days = day_id + 1
        day_start = np.array(day_starts, dtype=np.int32)
        day_length = np.zeros(n_days, dtype=np.int32)
        for d_id in range(n_days - 1):
            day_length[d_id] = day_start[d_id + 1] - day_start[d_id]
        day_length[n_days - 1] = n - day_start[n_days - 1]

        # I precompute dam_commitment as numpy view for O(1) access
        dam_commitment_values = df['dam_commitment'].values

        return PrecomputedArrays(
            step_day_offset=step_day_offset,
            step_day_id=step_day_id,
            day_start=day_start,
            day_length=day_length,
            dam_commitment_values=dam_commitment_values,
        )
