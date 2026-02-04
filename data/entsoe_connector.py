import os
import pandas as pd
from entsoe import EntsoePandasClient
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class EntsoeConnector:
    """
    Client for fetching data from ENTSO-E Transparency Platform.
    Focuses on Day-Ahead Prices for Greece (GR).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("ENTSOE_API_KEY must be provided or set in environment variables.")
            
        self.client = EntsoePandasClient(api_key=self.api_key)
        self.country_code = 'GR' # Greece
        
    def fetch_day_ahead_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch Day-Ahead Market Prices.
        Returns DataFrame with index as timezone-aware timestamps and column 'price'.
        """
        print(f"Fetching DAM Prices for {self.country_code} from {start_date} to {end_date}...")
        try:
            # query_day_ahead_prices returns a Series
            series = self.client.query_day_ahead_prices(self.country_code, start=start_date, end=end_date)
            df = series.to_frame(name='price')
            return df
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def fetch_load_forecast(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Fetch Total Load Forecast."""
        print(f"Fetching Load Forecast for {self.country_code}...")
        try:
            series = self.client.query_load_forecast(self.country_code, start=start_date, end=end_date)
            df = series.to_frame(name='load_forecast')
            return df
        except Exception as e:
            print(f"Error fetching load forecast: {e}")
            return pd.DataFrame()

    def fetch_wind_solar_forecast(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Fetch Wind and Solar Generation Forecast."""
        print(f"Fetching Wind/Solar Forecast for {self.country_code}...")
        try:
            # query_wind_and_solar_forecast returns DataFrame with columns like 'Solar', 'Wind Onshore'
            df = self.client.query_wind_and_solar_forecast(self.country_code, start=start_date, end=end_date, psr_type=None)
            # Rename for consistency
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            return df
        except Exception as e:
            print(f"Error fetching RES forecast: {e}")
            return pd.DataFrame()

    def fetch_imbalance_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch Imbalance Prices (ISP).
        Critical for Balancing Market strategy.
        """
        print(f"Fetching Imbalance Prices for {self.country_code}...")
        try:
            # query_imbalance_prices returns DataFrame with 'Long', 'Short' prices
            df = self.client.query_imbalance_prices(self.country_code, start=start_date, end=end_date, psr_type=None)
            # Rename for consistency: 'Short' -> 'upward_regulation', 'Long' -> 'downward_regulation' or similar
            # Usually: 
            # Positive Imbalance Price (System Long) -> Sell to TSO (Down Reg)
            # Negative Imbalance Price (System Short) -> Buy from TSO (Up Reg)
            return df
        except Exception as e:
            print(f"Error fetching Imbalance prices: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filepath: str):
        """Save fetched data to CSV."""
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")

if __name__ == "__main__":
    # Example Usage (Mocking or requiring real key)
    # python data/entsoe_connector.py
    
    key = os.getenv('ENTSOE_API_KEY', 'MOCK_KEY') # Provide key in .env
    
    if key == 'MOCK_KEY':
        print("Warning: Using Mock Key, connection will likely fail unless library mocks it.")
    
    try:
        connector = EntsoeConnector(api_key=key)
        start = pd.Timestamp.now(tz='Europe/Athens').floor('D') - pd.Timedelta(days=1)
        end = pd.Timestamp.now(tz='Europe/Athens').floor('D') + pd.Timedelta(days=1)
        
        df = connector.fetch_day_ahead_prices(start, end)
        print(df.head())
        
        os.makedirs("data/downloads", exist_ok=True)
        connector.save_to_csv(df, "data/downloads/dam_prices_gr.csv")
        
    except Exception as e:
        print(f"Execution failed: {e}")
