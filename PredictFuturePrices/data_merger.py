"""Data merger — joins all collector CSVs into a single training DataFrame.

I merge ENTSO-E (DAM prices, Serbia, load/RES forecasts), ADMIE (balancing),
ISP clearing prices, weather, and commodity data on timestamp. The result
is suitable for training all forecasters.

I trim the dataset to start from when most data sources are available
(2023-01-01) to avoid training on rows where features are all zeros.

Pattern adapted from data/create_unified_training_data.py merge strategy.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from PredictFuturePrices.config import PATHS

logger = logging.getLogger(__name__)

# I set the minimum start date to when weather/load/commodities become available.
# I keep 2023 data for DAM lag features (7-day lags need history), but ADMIE
# features will be NaN-filled for 2023 rows. The forecasters' vectorized
# build_features_matrix already handles NaN→0 gracefully.
MIN_START_DATE = "2023-01-01"

# I check for ISP clearing data in the package first, then the project root
ISP_CLEARING_CANDIDATES = [
    PATHS.isp_clearing_csv,
    PATHS.data_dir.parent.parent / "data" / "isp_clearing_prices.csv",
]


class DataMerger:
    """I merge all collector outputs into a unified training DataFrame.

    The merge strategy:
    1. GR DAM prices as the primary time axis (hourly)
    2. Left join Serbia DAM (hourly)
    3. Left join ADMIE balancing (hourly)
    4. Left join ISP1 clearing prices (resampled from 15-min to hourly)
    5. Left join ENTSO-E load/RES forecasts (hourly)
    6. Left join weather (hourly)
    7. Left join commodities (hourly, from daily ffill)
    8. Add derived features (rolling stats, lags, calendar, net_imbalance)
    9. Trim to MIN_START_DATE to avoid dead rows
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or PATHS.data_dir

    def merge_all(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """I load all CSVs and merge into a single DataFrame.

        Args:
            output_path: Where to save the merged CSV. Defaults to PATHS.merged_csv.

        Returns:
            Merged DataFrame with DatetimeIndex.
        """
        output_path = output_path or PATHS.merged_csv

        # I load each data source
        gr_dam = self._load_csv(PATHS.gr_dam_csv, "GR DAM")
        rs_dam = self._load_csv(PATHS.rs_dam_csv, "RS DAM")
        admie = self._load_csv(PATHS.admie_balancing_csv, "ADMIE")
        load_fc = self._load_csv(PATHS.load_forecast_csv, "Load Forecast")
        res_fc = self._load_csv(PATHS.res_forecast_csv, "RES Forecast")
        weather = self._load_csv(PATHS.weather_csv, "Weather")
        commodity = self._load_csv(PATHS.commodity_csv, "Commodity")
        isp_clearing = self._load_isp_clearing()

        if gr_dam.empty:
            logger.error("GR DAM prices not found — cannot merge without primary data source")
            return pd.DataFrame()

        # I rename GR DAM column for clarity
        if "price" in gr_dam.columns:
            gr_dam = gr_dam.rename(columns={"price": "dam_price_gr"})

        # I start with GR DAM as the base
        merged = gr_dam.copy()

        # I left-join all other sources
        if not rs_dam.empty:
            merged = merged.join(rs_dam, how="left", rsuffix="_rs")
            logger.info(f"  + RS DAM: {rs_dam.shape}")

        if not admie.empty:
            merged = merged.join(admie, how="left", rsuffix="_admie")
            logger.info(f"  + ADMIE: {admie.shape}")

        if not isp_clearing.empty:
            merged = merged.join(isp_clearing, how="left", rsuffix="_isp")
            logger.info(f"  + ISP Clearing: {isp_clearing.shape}")

        if not load_fc.empty:
            merged = merged.join(load_fc, how="left", rsuffix="_lf")
            logger.info(f"  + Load Forecast: {load_fc.shape}")

        if not res_fc.empty:
            merged = merged.join(res_fc, how="left", rsuffix="_rf")
            logger.info(f"  + RES Forecast: {res_fc.shape}")

        if not weather.empty:
            merged = merged.join(weather, how="left", rsuffix="_wx")
            logger.info(f"  + Weather: {weather.shape}")

        if not commodity.empty:
            merged = merged.join(commodity, how="left", rsuffix="_cm")
            logger.info(f"  + Commodity: {commodity.shape}")

        # I add derived features
        merged = self._add_derived_features(merged)

        # I forward-fill short gaps (max 3 hours)
        merged = merged.ffill(limit=3)

        # I trim to start date to avoid rows where most features are NaN→0
        merged = merged.loc[MIN_START_DATE:]
        logger.info(f"Trimmed to {MIN_START_DATE}+: {len(merged)} rows")

        logger.info(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
        logger.info(f"Date range: {merged.index.min()} → {merged.index.max()}")
        logger.info(f"NaN rate: {merged.isna().mean().mean():.1%}")

        # I save the merged result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path)
        logger.info(f"Saved to {output_path}")

        return merged

    def _load_isp_clearing(self) -> pd.DataFrame:
        """I load ISP clearing prices and resample from 15-min to hourly.

        I keep only ISP1 columns (ISP2/ISP3 are closer to delivery and
        would introduce data leakage for DAM-time forecasting). I compute
        hourly means from the 4 quarter-hourly values per hour.
        """
        isp_path = None
        for candidate in ISP_CLEARING_CANDIDATES:
            if candidate.exists():
                isp_path = candidate
                break
        if isp_path is None:
            logger.warning(f"[ISP] No ISP clearing file found in {ISP_CLEARING_CANDIDATES}")
            return pd.DataFrame()

        try:
            isp = pd.read_csv(isp_path, index_col=0, parse_dates=True)
            if isp.index.tz is not None:
                isp.index = isp.index.tz_localize(None)

            # I keep only ISP1 columns to avoid data leakage
            isp1_cols = [c for c in isp.columns if c.startswith("isp1_")]
            if not isp1_cols:
                logger.warning("[ISP] No ISP1 columns found")
                return pd.DataFrame()

            isp = isp[isp1_cols]

            # I resample from 15-min to hourly using mean
            isp_hourly = isp.resample("h").mean()
            isp_hourly.index.name = "timestamp"
            logger.info(f"[ISP] Loaded {len(isp)} 15-min rows → {len(isp_hourly)} hourly rows")
            return isp_hourly
        except Exception as e:
            logger.error(f"[ISP] Failed to load: {e}")
            return pd.DataFrame()

    def _load_csv(self, path: Path, name: str) -> pd.DataFrame:
        """I load a CSV with error handling."""
        if not path.exists():
            logger.warning(f"[{name}] File not found: {path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "timestamp"
            logger.info(f"[{name}] Loaded {len(df)} rows from {path.name}")
            return df
        except Exception as e:
            logger.error(f"[{name}] Failed to load {path}: {e}")
            return pd.DataFrame()

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """I add rolling stats, lags, calendar, and derived market features.

        I follow the same pattern as create_unified_training_data.py
        but adapted for the merged dataset columns. I detect data resolution
        and use sph (steps-per-hour) to scale all window/lag sizes.
        """
        # I detect resolution: sph=4 for 15-min data, sph=1 for hourly
        sph = 1
        if len(df) > 10:
            freq = pd.infer_freq(df.index[:100])
            if freq and ("15" in str(freq) or "min" in str(freq)):
                sph = 4
            elif hasattr(df.index, 'freq') and df.index.freq is not None:
                freq_str = str(df.index.freq)
                if "15" in freq_str or "min" in freq_str:
                    sph = 4

        # I add calendar features
        if hasattr(df.index, "hour"):
            hour = df.index.hour
            dow = df.index.dayofweek
            month = df.index.month

            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            df["month_sin"] = np.sin(2 * np.pi * month / 12)
            df["month_cos"] = np.cos(2 * np.pi * month / 12)
            df["is_weekend"] = (dow >= 5).astype(float)

        # I compute window sizes based on resolution
        window_24h = 24 * sph
        window_6h = 6 * sph

        # I add DAM price rolling stats if available
        dam_col = "dam_price_gr"
        if dam_col in df.columns:
            df[f"{dam_col}_mean_24h"] = df[dam_col].rolling(window_24h, min_periods=1).mean()
            df[f"{dam_col}_std_24h"] = df[dam_col].rolling(window_24h, min_periods=1).std()
            df[f"{dam_col}_min_24h"] = df[dam_col].rolling(window_24h, min_periods=1).min()
            df[f"{dam_col}_max_24h"] = df[dam_col].rolling(window_24h, min_periods=1).max()
            df[f"{dam_col}_std_6h"] = df[dam_col].rolling(window_6h, min_periods=1).std()

            # I add DAM price lags (1d, 2d, 3d, 7d same period)
            for lag_days in [1, 2, 3, 7]:
                df[f"{dam_col}_lag_{lag_days}d"] = df[dam_col].shift(24 * sph * lag_days)

        # I add Serbia-GR spread if both are available
        if dam_col in df.columns and "rs_dam_price" in df.columns:
            df["rs_gr_spread"] = df["rs_dam_price"] - df[dam_col]

        # I compute res_forecast_mw as sum of solar + wind if separate columns exist
        if "res_forecast_mw" not in df.columns:
            solar_col = "solar" if "solar" in df.columns else None
            wind_col = "wind_onshore" if "wind_onshore" in df.columns else None
            if solar_col and wind_col:
                df["res_forecast_mw"] = df[solar_col].fillna(0) + df[wind_col].fillna(0)
                logger.info("  + Computed res_forecast_mw = solar + wind_onshore")

        # I compute net_imbalance_mw from system_deviation_mwh if available
        if "net_imbalance_mw" not in df.columns and "system_deviation_mwh" in df.columns:
            df["net_imbalance_mw"] = df["system_deviation_mwh"]
            logger.info("  + Derived net_imbalance_mw from system_deviation_mwh")

        # I add ISP1 spread (up - down) if ISP1 columns are available
        if "isp1_price_up" in df.columns and "isp1_price_down" in df.columns:
            df["isp1_spread"] = df["isp1_price_up"] - df["isp1_price_down"]

        # I add mFRR spread if individual prices are available
        if "mfrr_price_up" in df.columns and "mfrr_price_down" in df.columns:
            if "mfrr_spread" not in df.columns:
                df["mfrr_spread"] = df["mfrr_price_up"] - df["mfrr_price_down"]

        # I add load/RES forecast errors if actual + forecast are available
        if "load_forecast" in df.columns and "system_load_mw" in df.columns:
            df["load_forecast_error"] = df["system_load_mw"] - df["load_forecast"]
            df["load_forecast_error_lag"] = df["load_forecast_error"].shift(24 * sph)

        if "res_forecast_mw" in df.columns and "res_production_mw" in df.columns:
            df["res_forecast_error"] = df["res_production_mw"] - df["res_forecast_mw"]
            df["res_forecast_error_lag"] = df["res_forecast_error"].shift(24 * sph)

        return df
