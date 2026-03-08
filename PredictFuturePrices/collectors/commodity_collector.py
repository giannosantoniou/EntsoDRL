"""Commodity price collector using Yahoo Finance (yfinance).

I collect TTF natural gas futures and EU carbon allowance (EUA) prices.
These are daily prices forward-filled to hourly resolution.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from PredictFuturePrices.collectors.base_collector import BaseCollector
from PredictFuturePrices.config import COMMODITY, PATHS

logger = logging.getLogger(__name__)


def _import_yfinance():
    """Lazy import of yfinance to avoid hard dependency."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "yfinance is required for commodity data. "
            "Install it with: pip install yfinance"
        )


class CommodityCollector(BaseCollector):
    """I collect TTF gas and EUA carbon prices from Yahoo Finance.

    I download daily OHLCV data and extract Close prices,
    then forward-fill to hourly resolution for merging with energy data.
    """

    def __init__(self, output_path: Optional[Path] = None):
        super().__init__(
            name="commodity",
            output_path=output_path or PATHS.commodity_csv,
        )

    def get_default_start(self) -> datetime:
        return datetime(2023, 1, 1)

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """I fetch daily commodity prices and resample to hourly.

        I download TTF gas futures and EUA carbon allowances from Yahoo Finance,
        merge them, and forward-fill to hourly.
        """
        yf = _import_yfinance()

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        dfs = {}

        # I fetch TTF gas
        try:
            ttf = yf.download(
                COMMODITY.ttf_ticker,
                start=start_str,
                end=end_str,
                progress=False,
            )
            if not ttf.empty:
                # I handle both MultiIndex and flat columns from yfinance
                if isinstance(ttf.columns, pd.MultiIndex):
                    ttf_close = ttf[("Close", COMMODITY.ttf_ticker)]
                else:
                    ttf_close = ttf["Close"]
                dfs["ttf_gas"] = ttf_close
                logger.info(f"[commodity] TTF: {len(ttf)} daily rows")
        except Exception as e:
            logger.warning(f"[commodity] TTF fetch failed: {e}")

        # I fetch EUA carbon
        try:
            eua = yf.download(
                COMMODITY.eua_ticker,
                start=start_str,
                end=end_str,
                progress=False,
            )
            if not eua.empty:
                if isinstance(eua.columns, pd.MultiIndex):
                    eua_close = eua[("Close", COMMODITY.eua_ticker)]
                else:
                    eua_close = eua["Close"]
                dfs["eua_carbon"] = eua_close
                logger.info(f"[commodity] EUA: {len(eua)} daily rows")
        except Exception as e:
            logger.warning(f"[commodity] EUA fetch failed: {e}")

        if not dfs:
            return pd.DataFrame()

        # I merge all commodity series
        daily = pd.DataFrame(dfs)
        daily.index.name = "timestamp"

        # I strip timezone if present
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)

        # I resample daily → hourly with forward-fill
        hourly_idx = pd.date_range(
            daily.index.min(),
            daily.index.max() + pd.Timedelta(hours=23),
            freq="h",
        )
        hourly = daily.reindex(hourly_idx, method="ffill")
        hourly.index.name = "timestamp"

        # I drop rows that couldn't be filled (before first data point)
        hourly = hourly.dropna(how="all")

        return hourly
