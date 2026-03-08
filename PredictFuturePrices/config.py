"""Central configuration for PredictFuturePrices.

I keep all paths, constants, and feature definitions in one place
so that collectors, forecasters, and accuracy tracker stay in sync.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# I resolve the package root relative to this file
PACKAGE_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_ROOT.parent.resolve()

# I point to existing project data for reuse
EXISTING_DATA_DIR = PROJECT_ROOT / "data"
EXISTING_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class PathConfig:
    """All filesystem paths used by the package."""

    # Raw CSV storage (incremental)
    data_dir: Path = PACKAGE_ROOT / "data"
    entsoe_dir: Path = PACKAGE_ROOT / "data" / "entsoe"
    admie_dir: Path = PACKAGE_ROOT / "data" / "admie"
    admie_cache_dir: Path = PACKAGE_ROOT / "data" / "admie" / "cache"
    weather_dir: Path = PACKAGE_ROOT / "data" / "weather"
    commodities_dir: Path = PACKAGE_ROOT / "data" / "commodities"

    # Trained models
    models_dir: Path = PACKAGE_ROOT / "models"

    # Accuracy reports
    reports_dir: Path = PACKAGE_ROOT / "accuracy" / "reports"

    # Merged training data
    merged_csv: Path = PACKAGE_ROOT / "data" / "merged_training.csv"

    # Individual collector outputs
    gr_dam_csv: Path = PACKAGE_ROOT / "data" / "entsoe" / "gr_dam_prices.csv"
    rs_dam_csv: Path = PACKAGE_ROOT / "data" / "entsoe" / "rs_dam_prices.csv"
    load_forecast_csv: Path = PACKAGE_ROOT / "data" / "entsoe" / "load_forecast.csv"
    res_forecast_csv: Path = PACKAGE_ROOT / "data" / "entsoe" / "res_forecast.csv"
    isp_clearing_csv: Path = PACKAGE_ROOT / "data" / "admie" / "isp_clearing_prices.csv"
    admie_balancing_csv: Path = PACKAGE_ROOT / "data" / "admie" / "admie_balancing.csv"
    system_load_csv: Path = PACKAGE_ROOT / "data" / "admie" / "system_load.csv"
    res_actual_csv: Path = PACKAGE_ROOT / "data" / "admie" / "res_actual.csv"
    weather_csv: Path = PACKAGE_ROOT / "data" / "weather" / "weather_hourly.csv"
    commodity_csv: Path = PACKAGE_ROOT / "data" / "commodities" / "commodity_prices.csv"


@dataclass(frozen=True)
class EntsoeConfig:
    """ENTSO-E API configuration."""

    api_key: str = field(default_factory=lambda: os.environ.get("ENTSOE_API_KEY", ""))
    country_code_gr: str = "GR"
    country_code_rs: str = "RS"
    timezone: str = "Europe/Athens"
    # I set conservative rate limiting (ENTSO-E allows ~400/min)
    request_delay_s: float = 0.3
    max_retries: int = 3


@dataclass(frozen=True)
class AdmieConfig:
    """ADMIE API configuration."""

    base_url: str = "https://www.admie.gr/getOperationMarketFilewRange"
    request_delay_s: float = 0.5
    chunk_days: int = 30
    # I map category keys to file patterns (same as ADMIEDataFetcher)
    categories: tuple = (
        "isp1", "isp2", "isp3", "imbabe",
        "system_load", "res_production", "load_forecast", "res_forecast",
    )


@dataclass(frozen=True)
class WeatherConfig:
    """Open-Meteo API configuration (free, no key needed)."""

    base_url: str = "https://api.open-meteo.com/v1/forecast"
    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    # I use Athens coordinates as representative for Greek grid
    latitude: float = 37.98
    longitude: float = 23.73
    # I fetch these variables for energy-relevant weather
    hourly_variables: tuple = (
        "temperature_2m", "wind_speed_100m", "direct_radiation",
        "cloud_cover",
    )
    timezone: str = "Europe/Athens"


@dataclass(frozen=True)
class CommodityConfig:
    """Yahoo Finance tickers for commodities."""

    # TTF Natural Gas Front-Month
    ttf_ticker: str = "TTF=F"
    # EU Carbon Allowances (EUA)
    eua_ticker: str = "CKZ25.ICE"
    # I use daily data forward-filled to hourly
    resample_method: str = "ffill"


@dataclass(frozen=True)
class DAMForecasterConfig:
    """DAM forecaster hyperparameters."""

    n_hours: int = 24
    # I train 3 models per hour: point, P10, P90
    quantiles: tuple = (0.1, 0.5, 0.9)
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    })
    # I define the 29 feature names for documentation and validation
    n_features: int = 29


@dataclass(frozen=True)
class IDAForecasterConfig:
    """IDA forecaster hyperparameters."""

    # I train one model per IDA auction
    ida_numbers: tuple = (1, 2, 3)
    gate_hours: dict = field(default_factory=lambda: {1: 15, 2: 22, 3: 10})
    # I use blending alphas from existing IDASubForecaster
    blending_alphas: dict = field(default_factory=lambda: {1: 0.85, 2: 0.60, 3: 0.30})
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    })
    n_features: int = 30


@dataclass(frozen=True)
class LoadForecasterConfig:
    """Load forecaster hyperparameters."""

    n_hours: int = 24
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    })
    n_features: int = 20


@dataclass(frozen=True)
class XBIDForecasterConfig:
    """XBID forecaster hyperparameters."""

    # I predict at 4 horizons: 1h, 2h, 4h, 8h ahead
    horizons_hours: tuple = (1, 2, 4, 8)
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 7,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    })
    n_features: int = 37


@dataclass(frozen=True)
class AccuracyConfig:
    """Accuracy tracking configuration."""

    # I compute rolling metrics over these windows
    rolling_windows_days: tuple = (7, 30)
    # I track these metrics
    metrics: tuple = ("mae", "mape", "rmse", "bias")
    # I consider MAPE undefined when actual is below this threshold
    mape_floor: float = 1.0


# I create default instances for easy import
PATHS = PathConfig()
ENTSOE = EntsoeConfig()
ADMIE = AdmieConfig()
WEATHER = WeatherConfig()
COMMODITY = CommodityConfig()
DAM_FORECASTER = DAMForecasterConfig()
IDA_FORECASTER = IDAForecasterConfig()
LOAD_FORECASTER = LoadForecasterConfig()
XBID_FORECASTER = XBIDForecasterConfig()
ACCURACY = AccuracyConfig()
