"""Weather data collector using Open-Meteo API (free, no API key needed).

I collect temperature, wind speed, solar radiation, and cloud cover
for the Athens area as proxies for Greek grid-wide weather impact.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from PredictFuturePrices.collectors.base_collector import BaseCollector
from PredictFuturePrices.config import WEATHER, PATHS

logger = logging.getLogger(__name__)


class WeatherCollector(BaseCollector):
    """I collect hourly weather data from Open-Meteo.

    I use two endpoints:
    - archive-api for historical data (>5 days old)
    - api for recent + forecast data (last 5 days + 7 days ahead)
    """

    def __init__(self, output_path: Optional[Path] = None):
        super().__init__(
            name="weather",
            output_path=output_path or PATHS.weather_csv,
        )
        self.session = requests.Session()

    def get_default_start(self) -> datetime:
        return datetime(2023, 1, 1)

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """I fetch weather data, using archive API for old data and forecast API for recent.

        Open-Meteo archive supports up to 2 years back. For data older than ~5 days,
        I use the archive endpoint. For recent data, I use the forecast endpoint.
        """
        # I determine which endpoint(s) to use
        now = datetime.now()
        cutoff = now - timedelta(days=5)

        dfs = []

        # I fetch historical data from archive if start is before cutoff
        if start < cutoff:
            archive_end = min(end, cutoff)
            archive_df = self._fetch_archive(start, archive_end)
            if not archive_df.empty:
                dfs.append(archive_df)

        # I fetch recent + forecast data if end is after cutoff
        if end >= cutoff:
            forecast_start = max(start, cutoff)
            forecast_df = self._fetch_forecast(forecast_start, end)
            if not forecast_df.empty:
                dfs.append(forecast_df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs)
        result = result[~result.index.duplicated(keep="last")]
        return result.sort_index()

    def _fetch_archive(self, start: datetime, end: datetime) -> pd.DataFrame:
        """I fetch from the Open-Meteo archive API (historical data)."""
        params = {
            "latitude": WEATHER.latitude,
            "longitude": WEATHER.longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": ",".join(WEATHER.hourly_variables),
            "timezone": WEATHER.timezone,
        }

        try:
            resp = self.session.get(WEATHER.archive_url, params=params, timeout=30)
            resp.raise_for_status()
            return self._parse_response(resp.json())
        except Exception as e:
            logger.error(f"[weather] Archive fetch failed: {e}")
            return pd.DataFrame()

    def _fetch_forecast(self, start: datetime, end: datetime) -> pd.DataFrame:
        """I fetch from the Open-Meteo forecast API (recent + future)."""
        # I compute past_days and forecast_days
        now = datetime.now()
        past_days = min(max((now - start).days + 1, 0), 92)
        forecast_days = min(max((end - now).days + 1, 1), 16)

        params = {
            "latitude": WEATHER.latitude,
            "longitude": WEATHER.longitude,
            "hourly": ",".join(WEATHER.hourly_variables),
            "timezone": WEATHER.timezone,
            "past_days": past_days,
            "forecast_days": forecast_days,
        }

        try:
            resp = self.session.get(WEATHER.base_url, params=params, timeout=30)
            resp.raise_for_status()
            return self._parse_response(resp.json())
        except Exception as e:
            logger.error(f"[weather] Forecast fetch failed: {e}")
            return pd.DataFrame()

    def _parse_response(self, data: dict) -> pd.DataFrame:
        """I parse the Open-Meteo JSON response into a DataFrame."""
        if "hourly" not in data:
            logger.warning("[weather] Response missing 'hourly' key")
            return pd.DataFrame()

        hourly = data["hourly"]
        if "time" not in hourly:
            return pd.DataFrame()

        timestamps = pd.to_datetime(hourly["time"])
        df = pd.DataFrame(index=timestamps)
        df.index.name = "timestamp"

        # I map Open-Meteo variable names to our column names
        column_map = {
            "temperature_2m": "temperature",
            "wind_speed_100m": "wind_100m",
            "direct_radiation": "solar_radiation",
            "cloud_cover": "cloud_cover",
        }

        for api_name, col_name in column_map.items():
            if api_name in hourly:
                df[col_name] = hourly[api_name]

        # I drop rows where all weather columns are NaN
        df = df.dropna(how="all")
        return df
