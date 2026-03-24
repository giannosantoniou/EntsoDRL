"""Central configuration for PredictFuturePrices.

I keep all paths, constants, and feature definitions in one place
so that collectors, forecasters, and accuracy tracker stay in sync.
"""

import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    # EU Carbon Allowances (EUA) — CO2.L tracks EU ETS closely
    eua_ticker: str = "CO2.L"
    # I use daily data forward-filled to hourly
    resample_method: str = "ffill"


@dataclass(frozen=True)
class DAMForecasterConfig:
    """DAM forecaster hyperparameters."""

    n_periods: int = 96
    resolution_minutes: int = 15
    # I train 3 models per period: point, P10, P90
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

    n_periods: int = 96
    resolution_minutes: int = 15
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


class MarketEvent(Enum):
    """I represent the market event that triggered a prediction cycle.

    PRE_DAM  — 12:55: GR+RS DAM prices available (~12:50). Run all forecasters.
    PRE_IDA1 — 14:30: Before IDA1 gate closure (15:00). XBID + IDA1.
    PRE_IDA2 — 21:30: Before IDA2 gate closure (22:00). XBID + IDA2.
    ALL      — Manual/--once mode. Run everything (backward compat).
    """

    PRE_DAM = "pre_dam"
    PRE_IDA1 = "pre_ida1"
    PRE_IDA2 = "pre_ida2"
    ALL = "all"


# I map each market event to a hard deadline (hour, minute).
# If the cycle starts too close to the deadline, I enter fast mode.
_DEFAULT_EVENT_DEADLINES: Dict[MarketEvent, Tuple[int, int]] = {
    MarketEvent.PRE_DAM: (13, 30),    # DAM results available ~12:50, buffer until 13:30
    MarketEvent.PRE_IDA1: (15, 0),    # IDA1 gate 15:00
    MarketEvent.PRE_IDA2: (22, 0),    # IDA2 gate 22:00
}

# I map each market event to the collectors that should run.
# PRE_DAM fetches everything (GR+RS DAM just published). PRE_IDA1/2 only need ADMIE.
_DEFAULT_EVENT_COLLECTORS: Dict[MarketEvent, Tuple[str, ...]] = {
    MarketEvent.PRE_DAM: ("weather", "commodity", "entsoe_gr_dam", "entsoe_rs_dam", "admie_balancing"),
    MarketEvent.PRE_IDA1: ("admie_balancing",),
    MarketEvent.PRE_IDA2: ("admie_balancing",),
    MarketEvent.ALL: ("weather", "commodity", "entsoe_gr_dam", "entsoe_rs_dam", "admie_balancing"),
}

# I map each market event to the forecaster keys that should run.
# Keys: "dam", "load", "xbid", "ida" (all 3), "ida1", "ida2", "ida3"
_DEFAULT_EVENT_FORECASTERS: Dict[MarketEvent, Tuple[str, ...]] = {
    MarketEvent.PRE_DAM: ("dam", "load", "xbid", "ida3"),
    MarketEvent.PRE_IDA1: ("xbid", "ida1"),
    MarketEvent.PRE_IDA2: ("xbid", "ida2"),
    MarketEvent.ALL: ("dam", "load", "xbid", "ida"),
}

# I map (hour, minute) → MarketEvent for automatic event identification
_DEFAULT_EVENT_TIMES: Dict[Tuple[int, int], MarketEvent] = {
    (12, 55): MarketEvent.PRE_DAM,
    (14, 30): MarketEvent.PRE_IDA1,
    (21, 30): MarketEvent.PRE_IDA2,
}


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler configuration for the continuous prediction pipeline."""

    # I use Greek timezone because all market events are in Athens time
    timezone: str = "Europe/Athens"
    # I set the default interval to 1 hour between cycles
    default_interval_seconds: int = 3600
    # I enforce a minimum interval to prevent API abuse
    min_interval_seconds: int = 300
    # I set a maximum cycle duration (10 min) to prevent runaway cycles
    cycle_timeout_seconds: int = 600
    # I enter fast mode when fewer than 5 minutes remain before deadline
    deadline_fast_mode_seconds: int = 300
    # I align cycles to market events (hour, minute):
    # 12:55 = pre-DAM (GR+RS published ~12:50), 14:30 = pre-IDA1, 21:30 = pre-IDA2
    market_events: tuple = ((12, 55), (14, 30), (21, 30))
    # I map each market event to the forecasters that should run
    event_forecasters: Dict[MarketEvent, Tuple[str, ...]] = field(
        default_factory=lambda: dict(_DEFAULT_EVENT_FORECASTERS)
    )
    # I map (hour, minute) → MarketEvent for automatic identification
    event_times: Dict[Tuple[int, int], MarketEvent] = field(
        default_factory=lambda: dict(_DEFAULT_EVENT_TIMES)
    )
    # I map each market event to its hard deadline
    event_deadlines: Dict[MarketEvent, Tuple[int, int]] = field(
        default_factory=lambda: dict(_DEFAULT_EVENT_DEADLINES)
    )
    # I map each market event to the collectors it needs
    event_collectors: Dict[MarketEvent, Tuple[str, ...]] = field(
        default_factory=lambda: dict(_DEFAULT_EVENT_COLLECTORS)
    )
    # I allow a ±10 minute tolerance window when matching market events
    event_tolerance_minutes: int = 10
    # I log scheduler activity to this file
    log_file: str = "scheduler.log"


SCHEDULER = SchedulerConfig()
