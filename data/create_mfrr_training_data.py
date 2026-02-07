"""
Create Merged Training Dataset for mFRR Bidding

I combine all ENTSO-E balancing data into a single training dataset
with hourly resolution, suitable for DRL agent training.

Features included:
- Imbalance prices (Long/Short)
- Imbalance volumes (system imbalance direction)
- mFRR/aFRR activation prices (Up/Down)
- Total system load
- RES forecasts (Solar + Wind)
- Cross-border flows (scheduled exchanges)
- NTC capacity (import/export availability)
- Hydro reservoir levels

Derived features:
- Net imbalance (Up - Down)
- Net cross-border flow (imports - exports)
- RES penetration (RES / Load)
- Imbalance spread (Short - Long price)
- Time features (hour, day_of_week, month, is_weekend)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 70)
    print("Creating Merged Training Dataset for mFRR Bidding")
    print("=" * 70)

    balancing_dir = Path(__file__).parent / "balancing"

    # 1. IMBALANCE PRICES (Long/Short)
    print("\n1. Loading Imbalance Prices...")
    df_imb_prices = pd.read_csv(balancing_dir / "imbalance_prices.csv")
    df_imb_prices["timestamp"] = pd.to_datetime(df_imb_prices.iloc[:, 0], utc=True)
    df_imb_prices = df_imb_prices.set_index("timestamp")
    df_imb_prices = df_imb_prices[["Long", "Short"]]
    df_imb_prices.columns = ["imbalance_price_long", "imbalance_price_short"]
    df_imb_prices_h = df_imb_prices.resample("h").mean()
    print(f"   {len(df_imb_prices_h)} hourly records")

    # 2. IMBALANCE VOLUMES (A24) - System Long/Short
    print("\n2. Loading Imbalance Volumes (A24)...")
    df_imb_vol = pd.read_csv(balancing_dir / "imbalance_volumes_a24.csv")
    df_imb_vol["timestamp"] = pd.to_datetime(df_imb_vol["timestamp"], utc=True)
    df_imb_vol = df_imb_vol.pivot_table(
        index="timestamp",
        columns="direction",
        values="quantity_mw",
        aggfunc="sum"
    )
    df_imb_vol.columns = ["imbalance_volume_down", "imbalance_volume_up"]
    df_imb_vol["net_imbalance_mw"] = (
        df_imb_vol["imbalance_volume_up"] - df_imb_vol["imbalance_volume_down"]
    )
    df_imb_vol_h = df_imb_vol.resample("h").mean()
    print(f"   {len(df_imb_vol_h)} hourly records")

    # 3. ACTIVATED BALANCING PRICES (mFRR/aFRR Up/Down)
    print("\n3. Loading Activated Balancing Prices...")
    df_bal = pd.read_csv(balancing_dir / "activated_balancing_energy_prices.csv")
    df_bal["timestamp"] = pd.to_datetime(df_bal.iloc[:, 0], utc=True)
    df_bal = df_bal.set_index("timestamp")
    df_bal["col_name"] = (
        df_bal["ReserveType"].str.lower() + "_price_" + df_bal["Direction"].str.lower()
    )
    df_bal_pivot = df_bal.pivot_table(
        index=df_bal.index,
        columns="col_name",
        values="Price",
        aggfunc="mean"
    )
    df_bal_h = df_bal_pivot.resample("h").mean()
    print(f"   {len(df_bal_h)} hourly records")
    print(f"   Columns: {list(df_bal_h.columns)}")

    # 4. TOTAL LOAD (A65)
    print("\n4. Loading Total Load (A65)...")
    df_load = pd.read_csv(balancing_dir / "total_load_a65.csv")
    df_load["timestamp"] = pd.to_datetime(df_load["timestamp"], utc=True)
    df_load = df_load.set_index("timestamp")[["load_mw"]]
    df_load_h = df_load.resample("h").mean()
    print(f"   {len(df_load_h)} hourly records")

    # 5. SOLAR FORECAST (A69)
    print("\n5. Loading Solar Forecast (A69)...")
    df_solar = pd.read_csv(balancing_dir / "solar_forecast_a69.csv")
    df_solar["timestamp"] = pd.to_datetime(df_solar["timestamp"], utc=True)
    df_solar = df_solar.set_index("timestamp")[["forecast_mw"]]
    df_solar.columns = ["solar_forecast_mw"]
    df_solar_h = df_solar.resample("h").mean()
    print(f"   {len(df_solar_h)} hourly records")

    # 6. WIND FORECAST (A69)
    print("\n6. Loading Wind Forecast (A69)...")
    df_wind = pd.read_csv(balancing_dir / "wind_forecast_a69.csv")
    df_wind["timestamp"] = pd.to_datetime(df_wind["timestamp"], utc=True)
    df_wind = df_wind.set_index("timestamp")[["forecast_mw"]]
    df_wind.columns = ["wind_forecast_mw"]
    df_wind_h = df_wind.resample("h").mean()
    print(f"   {len(df_wind_h)} hourly records")

    # 7. SCHEDULED EXCHANGES (A11) - Net flow
    print("\n7. Loading Scheduled Exchanges (A11)...")
    df_flows = pd.read_csv(balancing_dir / "scheduled_exchanges_2y.csv")
    df_flows["timestamp"] = pd.to_datetime(df_flows["timestamp"], utc=True)

    # Calculate signed MW (positive = import, negative = export)
    df_flows["signed_mw"] = df_flows.apply(
        lambda r: r["scheduled_mw"] if r["direction"] == "import" else -r["scheduled_mw"],
        axis=1
    )

    # Total net flow (positive = net import)
    df_net_flow = df_flows.groupby("timestamp")["signed_mw"].sum().to_frame("net_cross_border_mw")
    df_net_flow_h = df_net_flow.resample("h").mean()

    # Per-neighbor net flows
    df_neighbor_net = df_flows.pivot_table(
        index="timestamp",
        columns="neighbor",
        values="signed_mw",
        aggfunc="sum"
    )
    df_neighbor_net.columns = [f"flow_{n}_mw" for n in df_neighbor_net.columns]
    df_neighbor_h = df_neighbor_net.resample("h").mean()

    print(f"   {len(df_net_flow_h)} hourly records")

    # 8. NTC CAPACITY (A61)
    print("\n8. Loading NTC Capacity (A61)...")
    df_ntc = pd.read_csv(balancing_dir / "ntc_capacity_a61.csv")
    df_ntc["timestamp"] = pd.to_datetime(df_ntc["timestamp"], utc=True)
    df_ntc_total = df_ntc.pivot_table(
        index="timestamp",
        columns="direction",
        values="capacity_mw",
        aggfunc="sum"
    )
    df_ntc_total.columns = ["total_ntc_export_mw", "total_ntc_import_mw"]
    df_ntc_h = df_ntc_total.resample("h").mean()
    print(f"   {len(df_ntc_h)} hourly records")

    # 9. HYDRO RESERVOIR (A72) - Weekly, forward fill
    print("\n9. Loading Hydro Reservoir (A72)...")
    df_hydro = pd.read_csv(balancing_dir / "hydro_reservoir_a72.csv")
    df_hydro["week_start"] = pd.to_datetime(df_hydro["week_start"], utc=True)
    df_hydro = df_hydro.set_index("week_start")
    df_hydro.columns = ["hydro_reservoir_gwh"]
    df_hydro["hydro_reservoir_gwh"] = df_hydro["hydro_reservoir_gwh"] / 1000  # to GWh
    df_hydro_h = df_hydro.resample("h").ffill()
    print(f"   {len(df_hydro_h)} hourly records (forward-filled)")

    # MERGE ALL
    print("\n" + "=" * 70)
    print("Merging all datasets...")

    dfs_to_merge = [
        df_imb_prices_h,
        df_imb_vol_h,
        df_bal_h,
        df_load_h,
        df_solar_h,
        df_wind_h,
        df_net_flow_h,
        df_neighbor_h,
        df_ntc_h,
        df_hydro_h
    ]

    # Merge all using outer join
    df_merged = dfs_to_merge[0]
    for df in dfs_to_merge[1:]:
        df_merged = df_merged.join(df, how="outer")

    # Add time features
    df_merged["hour"] = df_merged.index.hour
    df_merged["day_of_week"] = df_merged.index.dayofweek
    df_merged["month"] = df_merged.index.month
    df_merged["is_weekend"] = (df_merged["day_of_week"] >= 5).astype(int)

    # Calculate derived features
    df_merged["res_total_mw"] = (
        df_merged["solar_forecast_mw"].fillna(0) +
        df_merged["wind_forecast_mw"].fillna(0)
    )
    df_merged["res_penetration"] = (
        df_merged["res_total_mw"] / df_merged["load_mw"].replace(0, np.nan)
    )
    df_merged["imbalance_spread"] = (
        df_merged["imbalance_price_short"] - df_merged["imbalance_price_long"]
    )

    # Sort by timestamp
    df_merged = df_merged.sort_index()

    # Remove rows with too many NaNs (keep rows with at least 5 valid features)
    min_valid = 5
    df_merged = df_merged.dropna(thresh=min_valid)

    print(f"\nFinal dataset: {len(df_merged)} rows x {len(df_merged.columns)} columns")
    print(f"Date range: {df_merged.index.min()} to {df_merged.index.max()}")

    print(f"\nColumns ({len(df_merged.columns)}):")
    for col in df_merged.columns:
        non_null = df_merged[col].notna().sum()
        pct = 100 * non_null / len(df_merged)
        print(f"   {col}: {pct:.1f}% complete")

    # Save
    output_path = Path(__file__).parent / "mfrr_training_data.csv"
    df_merged.to_csv(output_path)

    print("\n" + "=" * 70)
    print(f"SAVED: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 70)

    return df_merged


if __name__ == "__main__":
    main()
