import os
import pandas as pd
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.entsoe_connector import EntsoeConnector

def fetch_history():
    load_dotenv()
    
    api_key = os.getenv('ENTSOE_API_KEY')
    if not api_key:
        print("Error: ENTSOE_API_KEY not found in .env")
        return

    connector = EntsoeConnector(api_key=api_key)
    
    # Define periods
    # Training: 2021-2023
    # Testing: 2024
    
    # Let's fetch everything in one go or year by year to be safe
    years = [2021, 2022, 2023, 2024, 2025, 2026] # Including 2026 to be up to date
    
    all_data = []
    
    for year in years:
        start = pd.Timestamp(f'{year}-01-01', tz='Europe/Athens')
        end = pd.Timestamp(f'{year}-12-31 23:00', tz='Europe/Athens')
        
        # Clip end to now if in future
        if end > pd.Timestamp.now(tz='Europe/Athens'):
            end = pd.Timestamp.now(tz='Europe/Athens')
            
        if start > end:
            continue
            
        print(f"Fetching data for {year}...")
        try:
            df = connector.fetch_day_ahead_prices(start, end)
            if not df.empty:
                all_data.append(df)
                print(f"Got {len(df)} records for {year}")
            else:
                print(f"No data found for {year}")
        except Exception as e:
            print(f"Failed to fetch {year}: {e}")
            
    if all_data:
        full_df = pd.concat(all_data)
        # Sort and remove duplicates
        full_df = full_df.sort_index()
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        
        output_path = "data/dam_prices_2021_2026.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        full_df.to_csv(output_path)
        print(f"Successfully saved {len(full_df)} records to {output_path}")
        print("Sample data:")
        print(full_df.head())
        print(full_df.tail())
    else:
        print("No data fetched.")

if __name__ == "__main__":
    fetch_history()
