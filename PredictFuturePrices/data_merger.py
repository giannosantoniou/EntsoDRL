"""Data merger — joins all collector CSVs into a single training DataFrame.

I merge ENTSO-E (DAM prices, Serbia, load/RES forecasts), ADMIE (balancing),
weather, and commodity data on timestamp. The result is suitable for
training all forecasters.

Pattern adapted from data/create_unified_training_data.py merge strategy.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from PredictFuturePrices.config import PATHS

logger = logging.getLogger(__name__)


class DataMerger:
    """I merge all collector outputs into a unified training DataFrame.

    The merge strategy:
    1. GR DAM prices as the primary time axis (hourly)
    2. Left join Serbia DAM (hourly)
    3. Left join ADMIE balancing (hourly, from 15-min resampled)
    4. Left join ENTSO-E load/RES forecasts (hourly)
    5. Left join weather (hourly)
    6. Left join commodities (hourly, from daily ffill)
    7. Add derived features (rolling stats, lags, calendar)
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

        logger.info(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
        logger.info(f"Date range: {merged.index.min()} → {merged.index.max()}")
        logger.info(f"NaN rate: {merged.isna().mean().mean():.1%}")

        # I save the merged result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path)
        logger.info(f"Saved to {output_path}")

        return merged

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
        """I add rolling stats, lags, and calendar features.

        I follow the same pattern as create_unified_training_data.py
        but adapted for the merged dataset columns.
        """
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

        # I add DAM price rolling stats if available
        dam_col = "dam_price_gr"
        if dam_col in df.columns:
            df[f"{dam_col}_mean_24h"] = df[dam_col].rolling(24, min_periods=1).mean()
            df[f"{dam_col}_std_24h"] = df[dam_col].rolling(24, min_periods=1).std()
            df[f"{dam_col}_min_24h"] = df[dam_col].rolling(24, min_periods=1).min()
            df[f"{dam_col}_max_24h"] = df[dam_col].rolling(24, min_periods=1).max()
            df[f"{dam_col}_std_6h"] = df[dam_col].rolling(6, min_periods=1).std()

            # I add DAM price lags (1d, 2d, 3d, 7d same hour)
            for lag_days in [1, 2, 3, 7]:
                df[f"{dam_col}_lag_{lag_days}d"] = df[dam_col].shift(24 * lag_days)

        # I add Serbia-GR spread if both are available
        if dam_col in df.columns and "rs_dam_price" in df.columns:
            df["rs_gr_spread"] = df["rs_dam_price"] - df[dam_col]

        # I add load/RES forecast errors if actual + forecast are available
        if "load_forecast" in df.columns and "system_load_mw" in df.columns:
            df["load_forecast_error"] = df["system_load_mw"] - df["load_forecast"]
            df["load_forecast_error_lag"] = df["load_forecast_error"].shift(24)

        if "res_forecast_mw" in df.columns and "res_production_mw" in df.columns:
            df["res_forecast_error"] = df["res_production_mw"] - df["res_forecast_mw"]
            df["res_forecast_error_lag"] = df["res_forecast_error"].shift(24)

        return df
