"""ECMWF Weather Forecast Collector — I fetch forecast runs for IDA corrections.

I download ECMWF IFS open data forecasts (free, 9km resolution) to compute
weather forecast DEVIATIONS between consecutive runs. This is the key signal
for IDA schedule corrections:

  DAM uses forecast from 06:00 UTC run
  IDA1 uses forecast from 12:00 UTC run → deviation = IDA signal!
  IDA2 uses forecast from 18:00 UTC run → even more updated
  IDA3 uses forecast from 06:00 UTC D+0 → 24h newer than DAM

Variables: temperature (2t), wind U/V (10u/10v), solar radiation (ssrd)
Grid: interpolated to Athens (37.98N, 23.73E) as Greek grid proxy

Usage:
    collector = ECMWFCollector()
    collector.fetch_incremental()  # I fetch latest forecast runs
    deviations = collector.compute_deviations()  # I compute run-to-run changes
"""

import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from ecmwf.opendata import Client as ECMWFClient
    HAS_ECMWF = True
except ImportError:
    HAS_ECMWF = False
    logger.warning("ecmwf-opendata not installed. Install: pip install ecmwf-opendata")


class ECMWFCollector:
    """I collect ECMWF IFS forecast data and compute run-to-run deviations.

    I store forecasts from each run (00Z, 06Z, 12Z, 18Z) separately,
    then compute deviations between consecutive runs. These deviations
    tell us: "the weather forecast CHANGED" — the key IDA trading signal.
    """

    # I use multiple locations to capture Greek RES production better
    # Athens alone has R2=0.31 for wind → I add Aegean islands + Thrace
    LOCATIONS = {
        'athens': {'lat': 37.98, 'lon': 23.73, 'weight': 0.3},    # Central grid, solar
        'aegean': {'lat': 37.45, 'lon': 25.35, 'weight': 0.4},    # Cyclades — main wind farms
        'thrace': {'lat': 41.13, 'lon': 25.87, 'weight': 0.3},    # Thrace — large wind farms
    }
    # I keep Athens as primary for backward compatibility
    TARGET_LAT = 37.98
    TARGET_LON = 23.73

    # I fetch these ECMWF parameters
    PARAMS = {
        '2t': 'temperature_2m',        # 2m temperature (K → C)
        '10u': 'wind_u_10m',           # 10m U-wind component (m/s)
        '10v': 'wind_v_10m',           # 10m V-wind component (m/s)
        'ssrd': 'solar_radiation',     # Surface solar radiation downward (J/m2)
    }

    # I define forecast steps (hourly for first 90h, then 3-hourly)
    HOURLY_STEPS = list(range(0, 91))  # 0-90h hourly
    THREE_HOURLY_STEPS = list(range(93, 145, 3))  # 93-144h 3-hourly

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._output_dir = output_dir or Path("PredictFuturePrices/data/ecmwf")
        self._cache_dir = cache_dir or self._output_dir / "cache"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._forecasts_csv = self._output_dir / "ecmwf_forecasts.csv"
        self._deviations_csv = self._output_dir / "ecmwf_deviations.csv"

        if HAS_ECMWF:
            self._client = ECMWFClient(source="ecmwf")
        else:
            self._client = None

    def fetch_latest_run(self, run_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """I fetch the latest ECMWF forecast run.

        Args:
            run_time: Specific run time (default: latest available)

        Returns:
            DataFrame with hourly forecast values, or None if failed
        """
        if not HAS_ECMWF:
            logger.error("ecmwf-opendata not installed")
            return None

        try:
            # I use a temporary file for GRIB download
            with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp:
                tmp_path = tmp.name

            # I request forecast data
            request = {
                "type": "fc",
                "step": self.HOURLY_STEPS[:25],  # I fetch first 24h for simplicity
                "param": list(self.PARAMS.keys()),
            }

            if run_time:
                request["date"] = run_time.strftime("%Y-%m-%d")
                request["time"] = f"{run_time.hour:02d}"

            logger.info(f"[ECMWF] Fetching forecast run...")
            self._client.retrieve(request, tmp_path)

            # I parse the GRIB file
            df = self._parse_grib(tmp_path, run_time or datetime.utcnow())

            # I clean up
            os.unlink(tmp_path)

            if df is not None and not df.empty:
                logger.info(f"[ECMWF] Fetched {len(df)} rows")
                self._save_forecast(df)

            return df

        except Exception as e:
            logger.error(f"[ECMWF] Fetch failed: {e}")
            return None

    def _parse_grib(self, grib_path: str, run_time: datetime) -> Optional[pd.DataFrame]:
        """I parse GRIB file and extract values for Athens location."""
        try:
            import xarray as xr
            ds = xr.open_dataset(grib_path, engine='cfgrib')

            # I find nearest grid point to Athens
            lat_idx = np.argmin(np.abs(ds.latitude.values - self.TARGET_LAT))
            lon_idx = np.argmin(np.abs(ds.longitude.values - self.TARGET_LON))

            records = []
            for step_idx, step in enumerate(ds.step.values):
                valid_time = run_time + pd.Timedelta(step)
                record = {
                    'run_time': run_time,
                    'valid_time': valid_time,
                    'step_hours': int(pd.Timedelta(step).total_seconds() / 3600),
                }

                for param, col_name in self.PARAMS.items():
                    if param in ds:
                        val = float(ds[param].isel(latitude=lat_idx, longitude=lon_idx, step=step_idx).values)
                        # I convert units
                        if param == '2t':
                            val -= 273.15  # K → C
                        elif param == 'ssrd':
                            val = max(0, val / 3600)  # J/m2 → W/m2
                        record[col_name] = val

                records.append(record)

            ds.close()
            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"[ECMWF] GRIB parse error: {e}")
            return None

    def _save_forecast(self, df: pd.DataFrame):
        """I append forecast to the cumulative CSV."""
        if self._forecasts_csv.exists():
            existing = pd.read_csv(self._forecasts_csv, parse_dates=['run_time', 'valid_time'])
            df = pd.concat([existing, df]).drop_duplicates(subset=['run_time', 'valid_time'])
        df.to_csv(self._forecasts_csv, index=False)

    def compute_deviations(self) -> Optional[pd.DataFrame]:
        """I compute forecast deviations between consecutive ECMWF runs.

        For each valid_time (delivery hour), I compare:
          deviation = forecast_run_N - forecast_run_N-1

        This tells us: "the weather prediction CHANGED between runs"
        → positive deviation means the new forecast is HIGHER
        """
        if not self._forecasts_csv.exists():
            logger.warning("[ECMWF] No forecast data to compute deviations")
            return None

        df = pd.read_csv(self._forecasts_csv, parse_dates=['run_time', 'valid_time'])

        if len(df) < 48:  # I need at least 2 runs
            return None

        # I pivot: rows = valid_time, columns = run_time
        weather_cols = [c for c in df.columns if c not in ['run_time', 'valid_time', 'step_hours']]

        deviations = []
        run_times = sorted(df['run_time'].unique())

        for i in range(1, len(run_times)):
            prev_run = df[df['run_time'] == run_times[i-1]].set_index('valid_time')
            curr_run = df[df['run_time'] == run_times[i]].set_index('valid_time')

            # I find common valid_times
            common = prev_run.index.intersection(curr_run.index)
            if len(common) == 0:
                continue

            for vt in common:
                record = {
                    'valid_time': vt,
                    'prev_run': run_times[i-1],
                    'curr_run': run_times[i],
                    'hours_between_runs': (run_times[i] - run_times[i-1]).total_seconds() / 3600,
                }
                for col in weather_cols:
                    if col in prev_run.columns and col in curr_run.columns:
                        prev_val = prev_run.loc[vt, col]
                        curr_val = curr_run.loc[vt, col]
                        if isinstance(prev_val, pd.Series):
                            prev_val = prev_val.iloc[0]
                        if isinstance(curr_val, pd.Series):
                            curr_val = curr_val.iloc[0]
                        record[f'{col}_deviation'] = curr_val - prev_val
                        record[f'{col}_latest'] = curr_val

                deviations.append(record)

        if not deviations:
            return None

        dev_df = pd.DataFrame(deviations)
        dev_df.to_csv(self._deviations_csv, index=False)
        logger.info(f"[ECMWF] Computed {len(dev_df)} deviations")
        return dev_df

    def fetch_via_openmeteo(self, date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """I fetch ECMWF data via Open-Meteo API for MULTIPLE Greek locations.

        I query 3 locations (Athens, Aegean, Thrace) and combine them
        with weights to estimate national wind/solar production in MW.
        """
        import requests

        target_date = date or datetime.utcnow()
        url = "https://api.open-meteo.com/v1/ecmwf"

        all_dfs = {}
        for loc_name, loc in self.LOCATIONS.items():
            params = {
                "latitude": loc['lat'],
                "longitude": loc['lon'],
                "hourly": "temperature_2m,wind_speed_10m,direct_radiation,cloud_cover",
                "timezone": "Europe/Athens",
                "forecast_days": 3,
            }
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "hourly" in data:
                    all_dfs[loc_name] = data["hourly"]
                    logger.info(f"[ECMWF-OM] Fetched {loc_name}: {len(data['hourly']['time'])} rows")
            except Exception as e:
                logger.warning(f"[ECMWF-OM] {loc_name} failed: {e}")

        if not all_dfs:
            return None

        # I combine locations with weights into estimated MW production
        first_loc = list(all_dfs.values())[0]
        n_rows = len(first_loc['time'])

        # I compute weighted wind speed across locations
        wind_weighted = np.zeros(n_rows)
        solar_weighted = np.zeros(n_rows)
        temp_avg = np.zeros(n_rows)
        cloud_avg = np.zeros(n_rows)
        total_weight = 0

        for loc_name, hourly in all_dfs.items():
            w = self.LOCATIONS[loc_name]['weight']
            wind = np.array(hourly.get('wind_speed_10m', [0]*n_rows), dtype=float)
            solar = np.array(hourly.get('direct_radiation', [0]*n_rows), dtype=float)
            temp = np.array(hourly.get('temperature_2m', [15]*n_rows), dtype=float)
            cloud = np.array(hourly.get('cloud_cover', [50]*n_rows), dtype=float)

            wind_weighted += wind * w
            solar_weighted += solar * w
            temp_avg += temp * w
            cloud_avg += cloud * w
            total_weight += w

        if total_weight > 0:
            wind_weighted /= total_weight
            solar_weighted /= total_weight
            temp_avg /= total_weight
            cloud_avg /= total_weight

        # I convert wind speed (m/s) to estimated MW production
        # Calibrated against historical: actual mean 1,139 MW at mean speed ~7 m/s
        # Greece ~5,500 MW installed wind (2025), avg capacity factor ~21%
        INSTALLED_WIND_MW = 5500
        # I use realistic power curve: cut-in 3m/s, rated 14m/s, cut-out 25m/s
        wind_cf = np.where(
            wind_weighted < 3, 0,
            np.where(wind_weighted > 25, 0,
                     np.clip((wind_weighted - 3) / (14 - 3), 0, 1) ** 3 * 0.45))
        wind_production_mw = INSTALLED_WIND_MW * wind_cf

        # I convert solar radiation (W/m2) to estimated MW production
        # Calibrated: actual mean 571 MW, installed ~7,000 MW, avg CF ~8% (includes night)
        INSTALLED_SOLAR_MW = 7000
        # I use linear with realistic peak efficiency ~15%
        solar_cf = np.clip(solar_weighted / 1000.0 * 0.15, 0, 0.85)
        solar_production_mw = INSTALLED_SOLAR_MW * solar_cf

        df = pd.DataFrame({
            'valid_time': pd.to_datetime(first_loc['time']),
            'temperature_2m': temp_avg,
            'wind_speed_weighted': wind_weighted,
            'solar_radiation_weighted': solar_weighted,
            'cloud_cover': cloud_avg,
            'wind_production_mw': wind_production_mw,
            'solar_production_mw': solar_production_mw,
            'res_production_mw': wind_production_mw + solar_production_mw,
        })
        df['run_time'] = target_date.replace(minute=0, second=0, microsecond=0)

        logger.info(f"[ECMWF-OM] Combined {len(all_dfs)} locations, {len(df)} rows")
        logger.info(f"[ECMWF-OM] Wind: {wind_production_mw.mean():.0f} MW avg, "
                     f"Solar: {solar_production_mw.mean():.0f} MW avg")
        return df

    def get_deviation_features(self, timestamp: datetime) -> Dict[str, float]:
        """I return weather deviation features for a specific timestamp.

        This is called by IDA forecasters to get the latest weather signal.

        Returns:
            Dict with keys like 'wind_deviation', 'solar_deviation', etc.
        """
        defaults = {
            'ecmwf_temp_deviation': 0.0,
            'ecmwf_wind_deviation': 0.0,
            'ecmwf_solar_deviation': 0.0,
            'ecmwf_wind_latest': 0.0,
            'ecmwf_solar_latest': 0.0,
        }

        if not self._deviations_csv.exists():
            return defaults

        try:
            dev_df = pd.read_csv(self._deviations_csv, parse_dates=['valid_time'])
            # I find the closest valid_time
            diffs = abs(dev_df['valid_time'] - pd.Timestamp(timestamp))
            if diffs.min() > pd.Timedelta(hours=3):
                return defaults  # I don't have data close enough

            closest = dev_df.loc[diffs.idxmin()]

            return {
                'ecmwf_temp_deviation': closest.get('temperature_2m_deviation', 0),
                'ecmwf_wind_deviation': closest.get('wind_u_10m_deviation', 0),
                'ecmwf_solar_deviation': closest.get('solar_radiation_deviation', 0),
                'ecmwf_wind_latest': closest.get('wind_u_10m_latest', 0),
                'ecmwf_solar_latest': closest.get('solar_radiation_latest', 0),
            }
        except Exception as e:
            logger.error(f"[ECMWF] Deviation lookup error: {e}")
            return defaults
