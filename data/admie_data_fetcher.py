"""
ADMIE Market Data Fetcher
=========================
I fetch and parse real market data from ADMIE (Greek TSO) API for training
the unified multi-market DRL agent.

Data sources (all from https://www.admie.gr):
- ISP1ISPResults: aFRR/mFRR capacity prices (15-min, full day)
- IMBABE: mFRR energy prices, imbalance prices, system deviation (15-min)
- BalancingEnergyProduct: mFRR/aFRR activation volumes (15-min)
- RealTimeSCADASystemLoad: System load (hourly)
- RealTimeSCADARES: RES injections (hourly)
- DayAheadLoadForecast: Load forecasts (hourly)
- DayAheadRESForecast: RES forecasts (hourly)

Usage:
    python data/admie_data_fetcher.py --start 2024-01-01 --end 2026-01-31 --output data/admie_market_data.csv
"""

import argparse
import io
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ADMIEDataFetcher:
    """
    I fetch, parse, and merge ADMIE market data into a unified DataFrame
    suitable for DRL training.
    """

    BASE_URL = "https://www.admie.gr/getOperationMarketFilewRange"

    # I define all file types we need
    FILE_CATEGORIES = {
        'isp1': 'ISP1ISPResults',
        'imbabe': 'IMBABE',
        'balancing_energy': 'BalancingEnergyProduct',
        'system_load': 'RealTimeSCADASystemLoad',
        'res_production': 'RealTimeSCADARES',
        'load_forecast': 'DayAheadLoadForecast',
        'res_forecast': 'DayAheadRESForecast',
    }

    # I limit requests to avoid overwhelming the server
    REQUEST_DELAY = 0.5  # seconds between requests
    CHUNK_DAYS = 30  # I fetch in monthly chunks to keep response sizes manageable

    def __init__(self, cache_dir: Optional[str] = None):
        """
        I initialize the fetcher with an optional cache directory.
        Cached Excel files prevent re-downloading on repeated runs.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/admie_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EntsoDRL-Research/1.0'
        })

    def fetch_all(
        self,
        start_date: str,
        end_date: str,
        resample_to: str = '1h'
    ) -> pd.DataFrame:
        """
        I fetch all data types, parse them, merge into a unified DataFrame.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resample_to: Target time resolution ('15min' or '1h')

        Returns:
            Unified DataFrame with all market data
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        logger.info(f"Fetching ADMIE data from {start_date} to {end_date}")

        # I fetch each data type independently
        isp_data = self._fetch_and_parse_category(
            'isp1', start, end, self._parse_isp_file
        )
        imbabe_data = self._fetch_and_parse_category(
            'imbabe', start, end, self._parse_imbabe_file
        )
        balancing_data = self._fetch_and_parse_category(
            'balancing_energy', start, end, self._parse_balancing_energy_file
        )
        load_data = self._fetch_and_parse_category(
            'system_load', start, end, self._parse_scada_load_file
        )
        res_data = self._fetch_and_parse_category(
            'res_production', start, end, self._parse_scada_res_file
        )
        load_forecast_data = self._fetch_and_parse_category(
            'load_forecast', start, end, self._parse_scada_load_file
        )
        res_forecast_data = self._fetch_and_parse_category(
            'res_forecast', start, end, self._parse_scada_res_file
        )

        # I rename forecast columns to distinguish from actuals
        if load_forecast_data is not None and not load_forecast_data.empty:
            load_forecast_data = load_forecast_data.rename(columns={
                'system_load_mw': 'load_forecast_mw'
            })
        if res_forecast_data is not None and not res_forecast_data.empty:
            res_forecast_data = res_forecast_data.rename(columns={
                'res_production_mw': 'res_forecast_mw'
            })

        # I filter each DataFrame to the requested date range before merging
        # (weekly files like IMBABE may contain data outside our range)
        all_sources = [
            isp_data, imbabe_data, balancing_data,
            load_data, res_data, load_forecast_data, res_forecast_data
        ]
        filtered = []
        for df in all_sources:
            if df is not None and not df.empty:
                mask = (df.index >= start) & (df.index <= end + timedelta(days=1))
                filtered.append(df.loc[mask])
            else:
                filtered.append(df)

        # I resample 15-min sources to hourly BEFORE merging
        # to avoid index mismatch between 15-min and hourly data
        if resample_to == '1h':
            for i in range(len(filtered)):
                df = filtered[i]
                if df is not None and not df.empty:
                    # I detect if this has 15-min data by checking index frequency
                    if len(df) > 2:
                        median_diff = df.index.to_series().diff().median()
                        if median_diff <= timedelta(minutes=20):
                            filtered[i] = self._resample_to_hourly(df)

        # I merge all DataFrames on timestamp
        unified = self._merge_dataframes(*filtered)

        if unified.empty:
            logger.warning("No data retrieved!")
            return unified

        # I do a final resample to ensure clean hourly alignment
        # (the outer join may introduce sub-hourly timestamps from misaligned sources)
        if resample_to == '1h' and len(unified) > 0:
            pre_len = len(unified)
            unified = self._resample_to_hourly(unified)
            if len(unified) != pre_len:
                logger.info(
                    f"Final hourly resample: {pre_len} -> {len(unified)} rows"
                )

        # I forward-fill hourly sources that were joined at :00 timestamps only.
        # When resample_to='15min', the outer join between 15-min sources
        # (ISP, IMBABE) and hourly sources (SCADA load, RES) leaves NaN at
        # :15/:30/:45 for the hourly columns. Forward-fill propagates the
        # :00 value across the quarter-hours within that hour.
        if resample_to == '15min' and len(unified) > 0:
            pre_nan = unified.isna().sum().sum()
            unified = unified.ffill()
            post_nan = unified.isna().sum().sum()
            if pre_nan != post_nan:
                logger.info(
                    f"15-min ffill: {pre_nan} -> {post_nan} NaN values"
                )

        # I add derived features
        unified = self._add_derived_features(unified)

        logger.info(
            f"Final dataset: {len(unified)} rows, "
            f"{unified.columns.tolist()}"
        )

        return unified

    # =========================================================================
    # FETCHING
    # =========================================================================

    def _fetch_and_parse_category(
        self,
        category_key: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        parser_func
    ) -> Optional[pd.DataFrame]:
        """
        I fetch all files for a category in chunked date ranges,
        download the Excel files, and parse them into a DataFrame.
        """
        category = self.FILE_CATEGORIES[category_key]
        logger.info(f"Fetching {category_key} ({category})...")

        all_dfs = []
        current = start

        while current <= end:
            chunk_end = min(current + timedelta(days=self.CHUNK_DAYS), end)
            date_start = current.strftime('%Y-%m-%d')
            date_end = chunk_end.strftime('%Y-%m-%d')

            try:
                file_infos = self._get_file_list(category, date_start, date_end)
                logger.info(
                    f"  {category_key} [{date_start} -> {date_end}]: "
                    f"{len(file_infos)} files"
                )

                for file_info in file_infos:
                    file_path = file_info.get('file_path', '')
                    if not file_path:
                        continue

                    try:
                        local_path = self._download_file(file_path, category_key)
                        df = parser_func(local_path)
                        if df is not None and not df.empty:
                            all_dfs.append(df)
                    except Exception as e:
                        logger.warning(
                            f"  Error parsing {file_path}: {e}"
                        )

            except Exception as e:
                logger.warning(
                    f"  Error fetching {category_key} "
                    f"[{date_start} -> {date_end}]: {e}"
                )

            current = chunk_end + timedelta(days=1)
            time.sleep(self.REQUEST_DELAY)

        if not all_dfs:
            logger.warning(f"  No data parsed for {category_key}")
            return None

        result = pd.concat(all_dfs, ignore_index=False)
        result = result[~result.index.duplicated(keep='last')]
        result = result.sort_index()
        logger.info(f"  {category_key}: {len(result)} rows total")
        return result

    def _get_file_list(
        self, category: str, date_start: str, date_end: str
    ) -> List[Dict]:
        """I query the ADMIE API to get the list of available files."""
        params = {
            'dateStart': date_start,
            'dateEnd': date_end,
            'FileCategory': category
        }
        resp = self.session.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _download_file(self, url: str, category_key: str) -> Path:
        """
        I download a file to cache, skipping if already cached.
        Returns the local file path.
        """
        filename = url.split('/')[-1]
        cache_subdir = self.cache_dir / category_key
        cache_subdir.mkdir(parents=True, exist_ok=True)
        local_path = cache_subdir / filename

        if local_path.exists():
            return local_path

        time.sleep(self.REQUEST_DELAY)
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(resp.content)

        return local_path

    # =========================================================================
    # PARSERS
    # =========================================================================

    def _parse_isp_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        I parse ISP1/ISP2/ISP3 Excel files to extract aFRR and mFRR
        capacity prices and requirements at 15-min resolution.

        Structure per sheet (aFRR/mFRR):
        - Row 0: Title row (skip)
        - Row 1: [category_name, timestamp_1, timestamp_2, ...]
        - Row 2: Requirements Up values
        - Row 3: Requirements Down values
        - Row 4: Price Up values (EUR/MW/h)
        - Row 5: Price Down values (EUR/MW/h)
        - Rows 6+: Per-unit data (we skip these)
        """
        ext = file_path.suffix.lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'

        try:
            xls = pd.ExcelFile(file_path, engine=engine)
        except Exception as e:
            logger.warning(f"Cannot open {file_path}: {e}")
            return None

        records = []

        for sheet_type in ['aFRR', 'mFRR']:
            # I find the matching sheet name
            matching = [s for s in xls.sheet_names if sheet_type in s]
            if not matching:
                continue

            df = pd.read_excel(xls, sheet_name=matching[0], header=None)

            # I extract timestamps from row 1 (cols 1 onwards)
            timestamps = []
            for j in range(1, df.shape[1]):
                ts = df.iloc[1, j]
                if pd.notna(ts):
                    if isinstance(ts, datetime):
                        timestamps.append((j, pd.Timestamp(ts)))
                    elif isinstance(ts, str):
                        try:
                            timestamps.append((j, pd.Timestamp(ts)))
                        except Exception:
                            pass

            if not timestamps:
                continue

            # I find key rows by label
            row_indices = {}
            target_labels = [
                'Requirements Up', 'Requirements Down',
                'Price Up', 'Price Down'
            ]
            for i in range(min(10, len(df))):
                label = str(df.iloc[i, 0]).strip()
                if label in target_labels:
                    row_indices[label] = i

            # I build records for each timestamp
            prefix = sheet_type.lower().replace(' ', '')  # 'afrr' or 'mfrr'

            for col_idx, ts in timestamps:
                record = {'timestamp': ts}

                for label, row_idx in row_indices.items():
                    val = df.iloc[row_idx, col_idx]
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = np.nan

                    col_name = f"{prefix}_{label.lower().replace(' ', '_')}"
                    record[col_name] = val

                records.append(record)

        if not records:
            return None

        result = pd.DataFrame(records)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result = result.set_index('timestamp')

        # I aggregate if both aFRR and mFRR data exist for the same timestamp
        result = result.groupby(level=0).first()

        return result

    def _parse_imbabe_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        I parse IMBABE Excel files to extract imbalance prices,
        mFRR energy prices, and system deviation at 15-min resolution.

        Columns:
        [0] STARTDATE
        [1] ENDDATE
        [2] Total Activated Balancing Energy UP (MWh)
        [3] Total Activated Balancing Energy Down (MWh)
        [4] (ΔP + kΔf) "ISP energy" (MWh)
        [5] System deviation (MWh)
        [6] Curt_Flag
        [7] Imbalance Price (€/MWh)
        [8] mFRR Price UP (€/MWh)
        [9] mFRR Price Down (€/MWh)
        [10] Uplift Account 1 (€/MWh)
        [11] Uplift Account 2 (€/MWh)
        [12] Uplift Account 3 (€/MWh)
        """
        ext = file_path.suffix.lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'

        try:
            xls = pd.ExcelFile(file_path, engine=engine)
        except Exception as e:
            logger.warning(f"Cannot open {file_path}: {e}")
            return None

        # I find the data sheet (first one, not metadata)
        data_sheets = [
            s for s in xls.sheet_names
            if 'metadata' not in s.lower() and 'xdo' not in s.lower()
            and 'graph' not in s.lower()
        ]
        if not data_sheets:
            return None

        df = pd.read_excel(xls, sheet_name=data_sheets[0], header=None)

        # I skip the header row (row 0)
        if len(df) < 2:
            return None

        data = df.iloc[1:].copy()
        data.columns = range(data.shape[1])

        result = pd.DataFrame()
        result['timestamp'] = pd.to_datetime(data[0], errors='coerce')
        result['activated_energy_up_mwh'] = pd.to_numeric(
            data[2], errors='coerce'
        )
        result['activated_energy_down_mwh'] = pd.to_numeric(
            data[3], errors='coerce'
        )
        result['system_deviation_mwh'] = pd.to_numeric(
            data[5], errors='coerce'
        )
        result['imbalance_price'] = pd.to_numeric(data[7], errors='coerce')
        result['mfrr_energy_price_up'] = pd.to_numeric(
            data[8], errors='coerce'
        )
        result['mfrr_energy_price_down'] = pd.to_numeric(
            data[9], errors='coerce'
        )

        # I include uplift for full cost picture
        if data.shape[1] > 10:
            result['uplift_1'] = pd.to_numeric(data[10], errors='coerce')
        if data.shape[1] > 11:
            result['uplift_2'] = pd.to_numeric(data[11], errors='coerce')

        result = result.dropna(subset=['timestamp'])
        result = result.set_index('timestamp')

        return result

    def _parse_balancing_energy_file(
        self, file_path: Path
    ) -> Optional[pd.DataFrame]:
        """
        I parse BalancingEnergyProduct Excel files to extract
        mFRR and aFRR activation volumes at 15-min resolution.

        Columns:
        [0] STARTDATE
        [1] ENDDATE
        [2] Total mFRR UP (MWh)
        [3] Total DA mFRR UP (MWh)
        [4] Total mFRR DN (MWh)
        [5] Total DA mFRR DN (MWh)
        [6] Total aFRR UP (MWh)
        [7] Total aFRR DN (MWh)
        """
        ext = file_path.suffix.lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'

        try:
            xls = pd.ExcelFile(file_path, engine=engine)
        except Exception as e:
            logger.warning(f"Cannot open {file_path}: {e}")
            return None

        data_sheets = [
            s for s in xls.sheet_names
            if 'metadata' not in s.lower() and 'xdo' not in s.lower()
        ]
        if not data_sheets:
            return None

        df = pd.read_excel(xls, sheet_name=data_sheets[0], header=None)
        if len(df) < 2:
            return None

        data = df.iloc[1:].copy()
        data.columns = range(data.shape[1])

        result = pd.DataFrame()
        result['timestamp'] = pd.to_datetime(data[0], errors='coerce')
        result['mfrr_activated_up_mwh'] = pd.to_numeric(
            data[2], errors='coerce'
        )
        result['mfrr_activated_down_mwh'] = pd.to_numeric(
            data[4], errors='coerce'
        )

        if data.shape[1] > 6:
            result['afrr_activated_up_mwh'] = pd.to_numeric(
                data[6], errors='coerce'
            )
        if data.shape[1] > 7:
            result['afrr_activated_down_mwh'] = pd.to_numeric(
                data[7], errors='coerce'
            )

        result = result.dropna(subset=['timestamp'])
        result = result.set_index('timestamp')

        return result

    def _parse_scada_load_file(
        self, file_path: Path
    ) -> Optional[pd.DataFrame]:
        """
        I parse SCADA system load files.
        Format: wide table with date in col 0, hours 1-24 in cols 1-24.
        Values are in MWh (hourly).
        """
        ext = file_path.suffix.lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'

        try:
            xls = pd.ExcelFile(file_path, engine=engine)
        except Exception as e:
            logger.warning(f"Cannot open {file_path}: {e}")
            return None

        data_sheets = [
            s for s in xls.sheet_names
            if 'metadata' not in s.lower() and 'xdo' not in s.lower()
        ]
        if not data_sheets:
            return None

        df = pd.read_excel(xls, sheet_name=data_sheets[0], header=None)

        records = []

        # I find rows with dates and hourly data
        for i in range(len(df)):
            date_val = df.iloc[i, 0]
            if pd.isna(date_val):
                continue

            # I try to parse the date
            try:
                if isinstance(date_val, datetime):
                    date = pd.Timestamp(date_val).normalize()
                elif isinstance(date_val, str):
                    # I handle various date formats
                    for fmt in ['%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d/%m/%y']:
                        try:
                            date = pd.Timestamp(
                                datetime.strptime(date_val.strip(), fmt)
                            )
                            break
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue
            except Exception:
                continue

            # I check if there's numeric data in hourly columns
            # I skip descriptor rows (check if col 1 has a number)
            try:
                first_val = float(df.iloc[i, 1])
            except (ValueError, TypeError):
                continue

            # I read hourly values (cols 1-24)
            for hour_idx in range(min(24, df.shape[1] - 1)):
                try:
                    val = float(df.iloc[i, hour_idx + 1])
                    ts = date + timedelta(hours=hour_idx)
                    records.append({
                        'timestamp': ts,
                        'system_load_mw': val
                    })
                except (ValueError, TypeError):
                    pass

        if not records:
            return None

        result = pd.DataFrame(records)
        result = result.set_index('timestamp')
        # I take only the first row that looks like load data
        # (Net Load Without Crete is typically the first data row)
        result = result[~result.index.duplicated(keep='first')]
        return result

    def _parse_scada_res_file(
        self, file_path: Path
    ) -> Optional[pd.DataFrame]:
        """
        I parse SCADA RES production files.
        Same wide format as load files.
        """
        ext = file_path.suffix.lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'

        try:
            xls = pd.ExcelFile(file_path, engine=engine)
        except Exception as e:
            logger.warning(f"Cannot open {file_path}: {e}")
            return None

        data_sheets = [
            s for s in xls.sheet_names
            if 'metadata' not in s.lower() and 'xdo' not in s.lower()
        ]
        if not data_sheets:
            return None

        df = pd.read_excel(xls, sheet_name=data_sheets[0], header=None)

        records = []

        for i in range(len(df)):
            date_val = df.iloc[i, 0]
            if pd.isna(date_val):
                continue

            try:
                if isinstance(date_val, datetime):
                    date = pd.Timestamp(date_val).normalize()
                elif isinstance(date_val, str):
                    for fmt in ['%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d/%m/%y']:
                        try:
                            date = pd.Timestamp(
                                datetime.strptime(date_val.strip(), fmt)
                            )
                            break
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    continue
            except Exception:
                continue

            try:
                first_val = float(df.iloc[i, 1])
            except (ValueError, TypeError):
                continue

            for hour_idx in range(min(24, df.shape[1] - 1)):
                try:
                    val = float(df.iloc[i, hour_idx + 1])
                    ts = date + timedelta(hours=hour_idx)
                    records.append({
                        'timestamp': ts,
                        'res_production_mw': val
                    })
                except (ValueError, TypeError):
                    pass

        if not records:
            return None

        result = pd.DataFrame(records)
        result = result.set_index('timestamp')
        result = result[~result.index.duplicated(keep='first')]
        return result

    # =========================================================================
    # MERGE & TRANSFORM
    # =========================================================================

    def _merge_dataframes(self, *dataframes) -> pd.DataFrame:
        """I merge all DataFrames on their timestamp index."""
        valid_dfs = [df for df in dataframes if df is not None and not df.empty]

        if not valid_dfs:
            return pd.DataFrame()

        # I start with the first valid DataFrame
        result = valid_dfs[0]

        for df in valid_dfs[1:]:
            result = result.join(df, how='outer')

        return result.sort_index()

    def _resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        I resample 15-min data to hourly by taking the mean of prices
        and summing volumes.
        """
        # I separate price columns (mean) from volume columns (sum)
        price_cols = [
            c for c in df.columns
            if 'price' in c.lower() or 'uplift' in c.lower()
        ]
        volume_cols = [
            c for c in df.columns
            if 'mwh' in c.lower() or 'energy' in c.lower()
            or 'deviation' in c.lower()
        ]
        other_cols = [
            c for c in df.columns
            if c not in price_cols and c not in volume_cols
        ]

        agg_dict = {}
        for c in price_cols:
            agg_dict[c] = 'mean'
        for c in volume_cols:
            agg_dict[c] = 'sum'
        for c in other_cols:
            agg_dict[c] = 'mean'

        if not agg_dict:
            return df

        result = df.resample('1h').agg(agg_dict)
        return result

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """I add derived features useful for DRL training."""

        # I calculate mFRR energy spread (from IMBABE actual prices)
        if ('mfrr_energy_price_up' in df.columns
                and 'mfrr_energy_price_down' in df.columns):
            df['mfrr_energy_spread'] = (
                df['mfrr_energy_price_up'] - df['mfrr_energy_price_down']
            )

        # I calculate aFRR capacity spread (from ISP)
        if 'afrr_price_up' in df.columns and 'afrr_price_down' in df.columns:
            df['afrr_capacity_spread'] = (
                df['afrr_price_up'] - df['afrr_price_down']
            )

        # I calculate mFRR capacity spread (from ISP)
        if 'mfrr_price_up' in df.columns and 'mfrr_price_down' in df.columns:
            df['mfrr_capacity_spread'] = (
                df['mfrr_price_up'] - df['mfrr_price_down']
            )

        # I calculate net balancing energy
        if ('activated_energy_up_mwh' in df.columns
                and 'activated_energy_down_mwh' in df.columns):
            df['net_balancing_energy_mwh'] = (
                df['activated_energy_up_mwh']
                - df['activated_energy_down_mwh']
            )

        # I calculate RES forecast error
        if ('res_production_mw' in df.columns
                and 'res_forecast_mw' in df.columns):
            df['res_forecast_error_mw'] = (
                df['res_production_mw'] - df['res_forecast_mw']
            )

        # I calculate load forecast error
        if ('system_load_mw' in df.columns
                and 'load_forecast_mw' in df.columns):
            df['load_forecast_error_mw'] = (
                df['system_load_mw'] - df['load_forecast_mw']
            )

        # I add time features
        if not df.empty:
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # I create lagged features (1h lag for observation safety)
        lag_cols = [
            'mfrr_energy_price_up', 'mfrr_energy_price_down',
            'imbalance_price', 'mfrr_energy_spread',
            'afrr_price_up', 'afrr_price_down',
            'system_deviation_mwh'
        ]
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_lag_1h'] = df[col].shift(1)

        return df

    # =========================================================================
    # STATISTICS & REPORTING
    # =========================================================================

    def print_summary(self, df: pd.DataFrame):
        """I print a summary of the fetched dataset."""
        print("\n" + "=" * 70)
        print("ADMIE DATA SUMMARY")
        print("=" * 70)
        print(f"Date range: {df.index.min()} -> {df.index.max()}")
        print(f"Total rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Missing data rate: {df.isna().mean().mean():.1%}")
        print()

        # I group columns by category
        categories = {
            'aFRR Capacity': [c for c in df.columns if c.startswith('afrr')],
            'mFRR Capacity': [
                c for c in df.columns
                if c.startswith('mfrr') and 'energy' not in c
                and 'activated' not in c
            ],
            'mFRR Energy': [
                c for c in df.columns
                if 'mfrr_energy' in c or 'mfrr_activated' in c
            ],
            'Imbalance': [
                c for c in df.columns
                if 'imbalance' in c or 'deviation' in c
                or 'uplift' in c
            ],
            'Load/RES': [
                c for c in df.columns
                if 'load' in c or 'res' in c
            ],
        }

        for cat_name, cols in categories.items():
            if not cols:
                continue
            print(f"\n{cat_name}:")
            for col in sorted(cols):
                if col in df.columns:
                    valid = df[col].dropna()
                    if len(valid) > 0:
                        print(
                            f"  {col:40s}: "
                            f"mean={valid.mean():8.2f}, "
                            f"std={valid.std():8.2f}, "
                            f"min={valid.min():8.2f}, "
                            f"max={valid.max():8.2f}, "
                            f"missing={df[col].isna().mean():.1%}"
                        )

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch ADMIE market data for DRL training'
    )
    parser.add_argument(
        '--start', type=str, required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str, required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/admie_market_data.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--resolution', type=str, default='1h',
        choices=['15min', '1h'],
        help='Time resolution (default: 1h)'
    )
    parser.add_argument(
        '--cache-dir', type=str, default='data/admie_cache',
        help='Cache directory for downloaded Excel files'
    )

    args = parser.parse_args()

    fetcher = ADMIEDataFetcher(cache_dir=args.cache_dir)
    df = fetcher.fetch_all(args.start, args.end, resample_to=args.resolution)

    if df.empty:
        logger.error("No data retrieved. Check date range and connectivity.")
        return

    fetcher.print_summary(df)

    # I resolve the output path — if user didn't override and resolution is
    # 15min, I append _15min to the default filename to avoid overwriting
    # the hourly version
    output_path = Path(args.output)
    if (args.resolution == '15min'
            and args.output == 'data/admie_market_data.csv'):
        output_path = Path('data/admie_market_data_combined_15min.csv')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info(f"Saved to {output_path} ({len(df)} rows)")


if __name__ == '__main__':
    main()
