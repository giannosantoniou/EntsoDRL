"""
Fetch real Balancing Data from ENTSO-E.

mFRR (manual Frequency Restoration Reserve):
- mFRR Up price = cost to buy energy when short (best_ask)
- mFRR Down price = price received when selling surplus (best_bid)

aFRR (automatic Frequency Restoration Reserve):
- aFRR Up Amount = system need for upward regulation (market stress indicator)
- aFRR Down Amount = system need for downward regulation (oversupply indicator)
- aFRR Up/Down Price = capacity prices (EUR/MW/h)
"""
import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def fetch_balancing_prices(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch mFRR activated balancing energy prices from ENTSO-E.

    Returns DataFrame with columns:
        - mFRR_Up: Price for upward regulation (buying when short)
        - mFRR_Down: Price for downward regulation (selling when long)
    """
    api_key = api_key or os.getenv('ENTSOE_API_KEY')
    if not api_key:
        raise ValueError("ENTSOE_API_KEY required")

    client = EntsoePandasClient(api_key=api_key)

    print(f"Fetching mFRR Balancing Prices for GR: {start_date.date()} to {end_date.date()}")

    try:
        # I query activated balancing energy prices
        raw_df = client.query_activated_balancing_energy_prices(
            'GR',
            start=start_date,
            end=end_date
        )

        # Raw data has columns: Direction, Price, ReserveType
        # I need to pivot this to get Up and Down prices per timestamp

        # Filter for mFRR only (main balancing mechanism)
        mfrr_df = raw_df[raw_df['ReserveType'] == 'mFRR'].copy()

        # Pivot to get Up and Down as columns
        pivoted = mfrr_df.pivot_table(
            index=mfrr_df.index,
            columns='Direction',
            values='Price',
            aggfunc='mean'  # In case of duplicates
        )

        # Rename for clarity
        result = pd.DataFrame(index=pivoted.index)
        result['mFRR_Up'] = pivoted.get('Up', pd.Series(dtype=float))
        result['mFRR_Down'] = pivoted.get('Down', pd.Series(dtype=float))

        # Fill NaN with DAM price proxy (we'll handle this in integration)
        # For now, forward fill gaps
        result = result.ffill().bfill()

        print(f"  Retrieved {len(result)} rows")
        print(f"  mFRR Up   - Mean: {result['mFRR_Up'].mean():.2f} EUR/MWh")
        print(f"  mFRR Down - Mean: {result['mFRR_Down'].mean():.2f} EUR/MWh")
        print(f"  Spread    - Mean: {(result['mFRR_Up'] - result['mFRR_Down']).mean():.2f} EUR/MWh")

        return result

    except Exception as e:
        print(f"Error fetching balancing prices: {e}")
        return pd.DataFrame()


def fetch_afrr_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch aFRR contracted amounts and prices from ENTSO-E.

    Returns DataFrame with columns:
        - aFRR_Up_MW: Contracted upward regulation capacity (MW)
        - aFRR_Down_MW: Contracted downward regulation capacity (MW)
        - aFRR_Up_Price: Capacity price for upward (EUR/MW/h)
        - aFRR_Down_Price: Capacity price for downward (EUR/MW/h)

    I use these as market stress indicators:
    - High aFRR_Up_MW = system expects deficit = discharge opportunity
    - High aFRR_Down_MW = system expects surplus = charge opportunity
    """
    api_key = api_key or os.getenv('ENTSOE_API_KEY')
    if not api_key:
        raise ValueError("ENTSOE_API_KEY required")

    client = EntsoePandasClient(api_key=api_key)

    print(f"Fetching aFRR Data for GR: {start_date.date()} to {end_date.date()}")

    result = pd.DataFrame()

    try:
        # Fetch contracted amounts (A51 = aFRR, A01 = Daily)
        amounts = client.query_contracted_reserve_amount(
            'GR',
            process_type='A51',
            type_marketagreement_type='A01',
            start=start_date,
            end=end_date
        )

        if not amounts.empty:
            result['aFRR_Up_MW'] = amounts['Up']
            result['aFRR_Down_MW'] = amounts['Down']
            print(f"  Amounts: {len(amounts)} rows")
            print(f"    Up   - Mean: {amounts['Up'].mean():.1f} MW")
            print(f"    Down - Mean: {amounts['Down'].mean():.1f} MW")

    except Exception as e:
        print(f"  Warning: Could not fetch aFRR amounts: {e}")

    try:
        # Fetch contracted prices
        prices = client.query_contracted_reserve_prices(
            'GR',
            process_type='A51',
            type_marketagreement_type='A01',
            start=start_date,
            end=end_date
        )

        if not prices.empty:
            result['aFRR_Up_Price'] = prices['Up']
            result['aFRR_Down_Price'] = prices['Down']
            print(f"  Prices: {len(prices)} rows")
            print(f"    Up   - Mean: {prices['Up'].mean():.2f} EUR/MW/h")
            print(f"    Down - Mean: {prices['Down'].mean():.2f} EUR/MW/h")

    except Exception as e:
        print(f"  Warning: Could not fetch aFRR prices: {e}")

    # Forward fill gaps
    if not result.empty:
        result = result.ffill().bfill()

    return result


def fetch_all_historical_afrr(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunk_months: int = 6
) -> pd.DataFrame:
    """
    Fetch aFRR data in chunks to avoid API timeouts.
    aFRR data available from 2022 onwards for Greece.
    """
    all_data = []
    current = start_date

    # I start from 2022 since that's when data became available
    min_available = pd.Timestamp('2022-01-01', tz='Europe/Athens')
    if current < min_available:
        print(f"Note: aFRR data only available from 2022.")
        current = min_available

    while current < end_date:
        chunk_end = min(current + pd.DateOffset(months=chunk_months), end_date)
        try:
            chunk_data = fetch_afrr_data(current, chunk_end)
            if not chunk_data.empty:
                all_data.append(chunk_data)
        except Exception as e:
            print(f"  Warning: Failed to fetch aFRR {current.date()} to {chunk_end.date()}: {e}")

        current = chunk_end

    if all_data:
        return pd.concat(all_data).sort_index()
    return pd.DataFrame()


def integrate_balancing_prices(
    data_path: str = "data/feasible_data_with_dam.csv",
    output_path: str = "data/feasible_data_with_balancing.csv"
) -> pd.DataFrame:
    """
    Load existing training data and add real mFRR balancing prices.

    I replace synthetic bid/ask spreads with real mFRR Up/Down prices.
    For 2021 data (before mFRR was available), I use synthetic spread
    based on average historical patterns.
    """
    print("\n=== Integrating Real Balancing Prices ===\n")

    # Load existing data
    print(f"Loading existing data: {data_path}")
    df = pd.read_csv(data_path, index_col=0)

    # I handle timezone properly - the index strings contain timezone info
    # e.g., "2021-01-01 00:00:00+02:00"
    df.index = pd.to_datetime(df.index, utc=True)  # I parse as UTC first
    df.index = df.index.tz_convert('Europe/Athens')  # Then convert to Athens

    print(f"  Data range: {df.index.min()} to {df.index.max()}")
    print(f"  Total rows: {len(df)}")

    # Fetch balancing prices for period from 2022 onwards
    start = pd.Timestamp('2022-01-01', tz='Europe/Athens')
    end = pd.Timestamp(df.index.max()).ceil('D') + pd.Timedelta(days=1)

    print(f"\nFetching mFRR data from {start.date()} to {end.date()}...")
    balancing = fetch_all_historical_balancing(start, end, chunk_months=6)

    # I initialize mFRR columns with synthetic spread for all data
    # This handles 2021 and any gaps
    print("\nInitializing with synthetic spread for baseline...")
    df['mFRR_Up'] = df['price'] * 1.3    # 30% above DAM as default
    df['mFRR_Down'] = df['price'] * 0.7  # 30% below DAM as default

    if not balancing.empty:
        print(f"\nMerging real mFRR data ({len(balancing)} rows)...")

        # I resample to hourly to match training data (15-min -> 1h)
        balancing_hourly = balancing.resample('h').mean()

        # I update only where we have real data
        for col in ['mFRR_Up', 'mFRR_Down']:
            if col in balancing_hourly.columns:
                # I find matching timestamps
                common_idx = df.index.intersection(balancing_hourly.index)
                if len(common_idx) > 0:
                    df.loc[common_idx, col] = balancing_hourly.loc[common_idx, col]
                    print(f"  Updated {len(common_idx)} rows for {col}")

    # I apply data quality filters before using the values
    # Some ENTSO-E data has extreme/erroneous values
    print("\nApplying data quality filters...")

    # I cap prices at reasonable bounds
    # Historical max DAM price in Greece was around 1000 EUR/MWh
    # I use more generous bounds for balancing prices
    price_min = -100   # Negative prices possible but not below -100
    price_max = 2000   # Cap at 2000 EUR/MWh

    extreme_up = (df['mFRR_Up'] < price_min) | (df['mFRR_Up'] > price_max)
    extreme_down = (df['mFRR_Down'] < price_min) | (df['mFRR_Down'] > price_max)

    if extreme_up.sum() > 0:
        print(f"  Resetting {extreme_up.sum()} extreme mFRR_Up values to synthetic")
        df.loc[extreme_up, 'mFRR_Up'] = df.loc[extreme_up, 'price'] * 1.3

    if extreme_down.sum() > 0:
        print(f"  Resetting {extreme_down.sum()} extreme mFRR_Down values to synthetic")
        df.loc[extreme_down, 'mFRR_Down'] = df.loc[extreme_down, 'price'] * 0.7

    # I ensure mFRR_Up >= mFRR_Down (no arbitrage condition)
    swap_mask = df['mFRR_Up'] < df['mFRR_Down']
    if swap_mask.sum() > 0:
        print(f"  Fixing {swap_mask.sum()} rows where Up < Down")
        temp = df.loc[swap_mask, 'mFRR_Up'].copy()
        df.loc[swap_mask, 'mFRR_Up'] = df.loc[swap_mask, 'mFRR_Down']
        df.loc[swap_mask, 'mFRR_Down'] = temp

    # I ensure minimum spread of 5 EUR/MWh (realistic market friction)
    min_spread = 5.0
    low_spread_mask = (df['mFRR_Up'] - df['mFRR_Down']) < min_spread
    if low_spread_mask.sum() > 0:
        print(f"  Adjusting {low_spread_mask.sum()} rows with spread < {min_spread}")
        mid_price = (df.loc[low_spread_mask, 'mFRR_Up'] + df.loc[low_spread_mask, 'mFRR_Down']) / 2
        df.loc[low_spread_mask, 'mFRR_Up'] = mid_price + min_spread / 2
        df.loc[low_spread_mask, 'mFRR_Down'] = mid_price - min_spread / 2

    # I also cap the max spread at 500 EUR/MWh (very conservative upper bound)
    max_spread = 500.0
    high_spread_mask = (df['mFRR_Up'] - df['mFRR_Down']) > max_spread
    if high_spread_mask.sum() > 0:
        print(f"  Capping {high_spread_mask.sum()} rows with spread > {max_spread}")
        mid_price = (df.loc[high_spread_mask, 'mFRR_Up'] + df.loc[high_spread_mask, 'mFRR_Down']) / 2
        df.loc[high_spread_mask, 'mFRR_Up'] = mid_price + max_spread / 2
        df.loc[high_spread_mask, 'mFRR_Down'] = mid_price - max_spread / 2

    # ========================================
    # aFRR Integration (market stress signals)
    # ========================================
    print(f"\n=== Fetching aFRR Data ===")
    afrr_data = fetch_all_historical_afrr(start, end, chunk_months=6)

    # I initialize aFRR columns with typical values for 2021 (before data available)
    # Typical values based on 2022-2024 averages
    df['aFRR_Up_MW'] = 600.0      # Typical upward capacity
    df['aFRR_Down_MW'] = 130.0    # Typical downward capacity
    df['aFRR_Up_Price'] = 5.0     # Typical capacity price
    df['aFRR_Down_Price'] = 1.5   # Typical capacity price

    if not afrr_data.empty:
        print(f"\nMerging real aFRR data ({len(afrr_data)} rows)...")

        # I resample to hourly to match training data
        afrr_hourly = afrr_data.resample('h').mean()

        # I update only where we have real data
        for col in ['aFRR_Up_MW', 'aFRR_Down_MW', 'aFRR_Up_Price', 'aFRR_Down_Price']:
            if col in afrr_hourly.columns:
                common_idx = df.index.intersection(afrr_hourly.index)
                if len(common_idx) > 0:
                    df.loc[common_idx, col] = afrr_hourly.loc[common_idx, col]
                    print(f"  Updated {len(common_idx)} rows for {col}")

    # I apply data quality filters for aFRR
    # Cap amounts at reasonable bounds (0 to 2000 MW)
    for col in ['aFRR_Up_MW', 'aFRR_Down_MW']:
        df[col] = df[col].clip(lower=0, upper=2000)

    # Cap prices at reasonable bounds (0 to 100 EUR/MW/h)
    for col in ['aFRR_Up_Price', 'aFRR_Down_Price']:
        df[col] = df[col].clip(lower=0, upper=100)

    # Fill any remaining NaN
    afrr_cols = ['aFRR_Up_MW', 'aFRR_Down_MW', 'aFRR_Up_Price', 'aFRR_Down_Price']
    df[afrr_cols] = df[afrr_cols].ffill().bfill()

    # Save
    df.to_csv(output_path)
    print(f"\nSaved enhanced data to: {output_path}")
    print(f"  New columns: mFRR_Up, mFRR_Down, aFRR_Up_MW, aFRR_Down_MW, aFRR_Up_Price, aFRR_Down_Price")

    # Summary statistics
    print("\n=== Final Data Summary ===")
    print(f"  DAM Price  - Mean: {df['price'].mean():.2f} EUR/MWh")
    print(f"  mFRR Up    - Mean: {df['mFRR_Up'].mean():.2f} EUR/MWh")
    print(f"  mFRR Down  - Mean: {df['mFRR_Down'].mean():.2f} EUR/MWh")
    spread = df['mFRR_Up'] - df['mFRR_Down']
    print(f"  Spread     - Mean: {spread.mean():.2f}, Max: {spread.max():.2f} EUR/MWh")

    print("\n=== aFRR Summary ===")
    print(f"  aFRR Up MW     - Mean: {df['aFRR_Up_MW'].mean():.1f} MW")
    print(f"  aFRR Down MW   - Mean: {df['aFRR_Down_MW'].mean():.1f} MW")
    print(f"  aFRR Up Price  - Mean: {df['aFRR_Up_Price'].mean():.2f} EUR/MW/h")
    print(f"  aFRR Down Price- Mean: {df['aFRR_Down_Price'].mean():.2f} EUR/MW/h")

    # I show breakdown by year
    print("\n=== Data by Year ===")
    df['year'] = df.index.year
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        year_spread = year_data['mFRR_Up'] - year_data['mFRR_Down']
        year_afrr_up = year_data['aFRR_Up_MW'].mean()
        print(f"  {year}: mFRR spread = {year_spread.mean():.0f} EUR, aFRR Up = {year_afrr_up:.0f} MW ({len(year_data)} rows)")

    df = df.drop(columns=['year'])

    return df


def fetch_all_historical_balancing(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunk_months: int = 3
) -> pd.DataFrame:
    """
    Fetch balancing prices in chunks to avoid API timeouts.
    mFRR data available from 2022 onwards for Greece.
    """
    all_data = []
    current = start_date

    # I start from 2022 since that's when data became available
    min_available = pd.Timestamp('2022-01-01', tz='Europe/Athens')
    if current < min_available:
        print(f"Note: mFRR data only available from 2022. Will use synthetic for earlier dates.")
        current = min_available

    while current < end_date:
        chunk_end = min(current + pd.DateOffset(months=chunk_months), end_date)
        try:
            chunk_data = fetch_balancing_prices(current, chunk_end)
            if not chunk_data.empty:
                all_data.append(chunk_data)
        except Exception as e:
            print(f"  Warning: Failed to fetch {current.date()} to {chunk_end.date()}: {e}")

        current = chunk_end

    if all_data:
        return pd.concat(all_data).sort_index()
    return pd.DataFrame()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'integrate':
        # Run full integration
        integrate_balancing_prices()
    else:
        # Test fetching recent data
        end = pd.Timestamp.now(tz='Europe/Athens').floor('D')
        start = end - pd.Timedelta(days=7)

        prices = fetch_balancing_prices(start, end)
        if not prices.empty:
            print("\nSample data:")
            print(prices.head(10))
