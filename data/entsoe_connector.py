import os
import logging
import pandas as pd
from entsoe import EntsoePandasClient
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# I configure module-level logger for production-grade logging
logger = logging.getLogger(__name__)


class EntsoeAPIError(Exception):
    """Custom exception for ENTSO-E API errors with context."""
    def __init__(self, message: str, endpoint: str, original_error: Optional[Exception] = None):
        self.endpoint = endpoint
        self.original_error = original_error
        super().__init__(f"[{endpoint}] {message}")


class EntsoeConnector:
    """
    Client for fetching data from ENTSO-E Transparency Platform.
    Focuses on Day-Ahead Prices for Greece (GR).

    I implement:
    - Proper logging instead of print statements
    - Response validation (check for expected columns)
    - Detailed error context for debugging
    - Data quality checks
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("ENTSOE_API_KEY must be provided or set in environment variables.")

        # I validate API key format (basic sanity check)
        if len(self.api_key) < 20:
            logger.warning("API key appears unusually short - verify it's correct")

        self.client = EntsoePandasClient(api_key=self.api_key)
        self.country_code = 'GR'  # Greece

        # Neighboring countries for cross-border analysis
        self.neighbor_codes = {
            'RS': 'Serbia',      # DAM clears before Greece - useful for forecasting
            'BG': 'Bulgaria',
            'MK': 'North Macedonia',
            'AL': 'Albania',
            'TR': 'Turkey',
            'IT': 'Italy',
        }

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        endpoint: str,
        required_columns: Optional[list] = None,
        min_rows: int = 1
    ) -> Tuple[bool, str]:
        """
        I validate API response DataFrames meet minimum quality standards.

        Returns:
            (is_valid, error_message)
        """
        if df is None:
            return False, "DataFrame is None"

        if df.empty:
            return False, "DataFrame is empty (no rows returned)"

        if len(df) < min_rows:
            return False, f"DataFrame has only {len(df)} rows (minimum: {min_rows})"

        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                return False, f"Missing required columns: {missing}. Got: {list(df.columns)}"

        # I check for excessive NaN values (>50% indicates data quality issue)
        for col in df.columns:
            nan_pct = df[col].isna().sum() / len(df)
            if nan_pct > 0.5:
                logger.warning(
                    f"[{endpoint}] Column '{col}' has {nan_pct:.1%} NaN values - data quality concern"
                )

        return True, ""

    def fetch_day_ahead_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch Day-Ahead Market Prices.
        Returns DataFrame with index as timezone-aware timestamps and column 'price'.

        Raises:
            EntsoeAPIError: If API call fails or returns invalid data
        """
        endpoint = "DAM_PRICES"
        logger.debug(f"Fetching DAM Prices for {self.country_code} from {start_date} to {end_date}")

        try:
            series = self.client.query_day_ahead_prices(
                self.country_code, start=start_date, end=end_date
            )

            if series is None or (hasattr(series, 'empty') and series.empty):
                raise EntsoeAPIError(
                    "API returned empty series", endpoint
                )

            df = series.to_frame(name='price')

            # I validate the response
            is_valid, error_msg = self._validate_dataframe(
                df, endpoint, required_columns=['price'], min_rows=1
            )
            if not is_valid:
                raise EntsoeAPIError(error_msg, endpoint)

            # I check for reasonable price values
            if df['price'].min() < -500 or df['price'].max() > 5000:
                logger.warning(
                    f"[{endpoint}] Unusual price range detected: "
                    f"min={df['price'].min():.2f}, max={df['price'].max():.2f}"
                )

            logger.info(f"[{endpoint}] Successfully fetched {len(df)} rows")
            return df

        except EntsoeAPIError:
            raise
        except Exception as e:
            logger.error(f"[{endpoint}] API call failed: {type(e).__name__}: {e}")
            raise EntsoeAPIError(str(e), endpoint, original_error=e)

    def fetch_neighbor_dam_prices(
        self,
        country_code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Fetch Day-Ahead prices from neighboring country.

        Useful for Greece because Serbia DAM clears ~12:00,
        before Greek gate closure at 13:00.

        Args:
            country_code: 'RS' for Serbia, 'BG' for Bulgaria, etc.
            start_date: Start timestamp
            end_date: End timestamp

        Returns:
            DataFrame with 'price' column
        """
        endpoint = f"DAM_PRICES_{country_code}"
        country_name = self.neighbor_codes.get(country_code, country_code)
        logger.debug(f"Fetching DAM Prices for {country_name} ({country_code})")

        try:
            series = self.client.query_day_ahead_prices(
                country_code, start=start_date, end=end_date
            )

            if series is None or (hasattr(series, 'empty') and series.empty):
                raise EntsoeAPIError(
                    f"API returned empty series for {country_name}", endpoint
                )

            df = series.to_frame(name='price')

            is_valid, error_msg = self._validate_dataframe(
                df, endpoint, required_columns=['price'], min_rows=1
            )
            if not is_valid:
                raise EntsoeAPIError(error_msg, endpoint)

            logger.info(f"[{endpoint}] Successfully fetched {len(df)} rows for {country_name}")
            return df

        except EntsoeAPIError:
            raise
        except Exception as e:
            logger.error(f"[{endpoint}] API call failed: {type(e).__name__}: {e}")
            raise EntsoeAPIError(str(e), endpoint, original_error=e)

    def fetch_serbia_dam_d1(self, target_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch Serbia DAM prices for D+1.

        Call this before 13:00 Greek time to get Serbia's
        already-published prices for tomorrow.

        Args:
            target_date: The delivery date (tomorrow)

        Returns:
            DataFrame with hourly prices for target_date
        """
        # Serbia DAM is for the full day
        start = target_date.normalize()  # 00:00
        end = start + pd.Timedelta(days=1)  # 24:00

        return self.fetch_neighbor_dam_prices('RS', start, end)

    def fetch_load_forecast(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Fetch Total Load Forecast."""
        endpoint = "LOAD_FORECAST"
        logger.debug(f"Fetching Load Forecast for {self.country_code}")

        try:
            series = self.client.query_load_forecast(
                self.country_code, start=start_date, end=end_date
            )
            df = series.to_frame(name='load_forecast')

            is_valid, error_msg = self._validate_dataframe(df, endpoint)
            if not is_valid:
                logger.warning(f"[{endpoint}] Validation failed: {error_msg}")
                return pd.DataFrame()

            logger.info(f"[{endpoint}] Successfully fetched {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"[{endpoint}] API call failed: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def fetch_wind_solar_forecast(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Fetch Wind and Solar Generation Forecast."""
        endpoint = "RES_FORECAST"
        logger.debug(f"Fetching Wind/Solar Forecast for {self.country_code}")

        try:
            df = self.client.query_wind_and_solar_forecast(
                self.country_code, start=start_date, end=end_date, psr_type=None
            )

            if df is None or df.empty:
                logger.warning(f"[{endpoint}] No RES forecast data available")
                return pd.DataFrame()

            # Rename for consistency
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            logger.info(f"[{endpoint}] Successfully fetched {len(df)} rows, columns: {list(df.columns)}")
            return df

        except Exception as e:
            logger.error(f"[{endpoint}] API call failed: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def fetch_imbalance_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch Imbalance Prices (ISP).
        Critical for Balancing Market strategy.
        """
        endpoint = "IMBALANCE_PRICES"
        logger.debug(f"Fetching Imbalance Prices for {self.country_code}")

        try:
            df = self.client.query_imbalance_prices(
                self.country_code, start=start_date, end=end_date, psr_type=None
            )

            if df is None or df.empty:
                logger.warning(f"[{endpoint}] No imbalance price data available")
                return pd.DataFrame()

            # I check for expected columns
            expected_cols = {'Long', 'Short'}
            if not expected_cols.issubset(set(df.columns)):
                logger.warning(
                    f"[{endpoint}] Unexpected columns: {list(df.columns)}. Expected: {expected_cols}"
                )

            logger.info(f"[{endpoint}] Successfully fetched {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"[{endpoint}] API call failed: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filepath: str):
        """Save fetched data to CSV."""
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")


if __name__ == "__main__":
    # I configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    key = os.getenv('ENTSOE_API_KEY')

    if not key:
        logger.error("ENTSOE_API_KEY not set in environment. Set it in .env file.")
        exit(1)

    try:
        connector = EntsoeConnector(api_key=key)
        start = pd.Timestamp.now(tz='Europe/Athens').floor('D') - pd.Timedelta(days=1)
        end = pd.Timestamp.now(tz='Europe/Athens').floor('D') + pd.Timedelta(days=1)

        df = connector.fetch_day_ahead_prices(start, end)
        logger.info(f"Sample data:\n{df.head()}")

        os.makedirs("data/downloads", exist_ok=True)
        connector.save_to_csv(df, "data/downloads/dam_prices_gr.csv")

    except EntsoeAPIError as e:
        logger.error(f"ENTSO-E API error: {e}")
        if e.original_error:
            logger.debug(f"Original error: {e.original_error}")
        exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {type(e).__name__}: {e}")
        exit(1)
