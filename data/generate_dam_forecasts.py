"""
Batch Generate D-1 DAM Price Forecasts using EntsoE3 Models

I use the trained XGBoost ensemble from EntsoE3/production_v2 to generate
retrospective D-1 DAM price forecasts for the full ADMIE training period
(2024-01-01 to 2026-01-17).

These forecasts represent what the forecaster WOULD HAVE predicted on each D-1,
using only data available at 12:55 CET (before DAM gate closure at 13:00).

Output: data/dam_forecasts_2024_2026.csv
    - timestamp: hourly datetime
    - forecast_price: predicted DAM price (EUR/MWh)
    - Used by create_unified_training_data.py to generate realistic DAM commitments
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# I set up paths to EntsoE3
ENTSOE3_DIR = Path(__file__).parent.parent.parent / "EntsoE3"
PRODUCTION_DIR = ENTSOE3_DIR / "production_v2"
DATA_RAW_DIR = ENTSOE3_DIR / "data" / "raw"
MODELS_DIR = PRODUCTION_DIR / "models"

# I add EntsoE3 src to path
sys.path.insert(0, str(PRODUCTION_DIR))
sys.path.insert(0, str(PRODUCTION_DIR / "src"))


def generate_all_forecasts(
    start_date: str = "2024-01-01",
    end_date: str = "2026-01-18",
    output_path: str = None
) -> pd.DataFrame:
    """
    I generate D-1 DAM price forecasts for every day in the date range.

    I load all data and models once, then loop through dates for efficiency.
    """
    import polars as pl
    import xgboost as xgb
    from train_v2 import DAMSampleV2, extract_features_v2, load_all_data

    if output_path is None:
        output_path = str(Path(__file__).parent / "dam_forecasts_2024_2026.csv")

    print("=" * 60)
    print("BATCH GENERATING D-1 DAM PRICE FORECASTS")
    print("=" * 60)

    # Step 1: I load all data ONCE (this is the expensive part)
    logger.info("Loading all data from EntsoE3...")
    greek_df, neighbor_df = load_all_data(DATA_RAW_DIR)

    # I fix timezones and resample to hourly
    greek_df = greek_df.with_columns(
        pl.col("timestamp").dt.cast_time_unit("ns").dt.replace_time_zone("UTC")
    ).sort("timestamp")

    neighbor_df = neighbor_df.with_columns(
        pl.col("timestamp").dt.cast_time_unit("ns").dt.replace_time_zone("UTC")
    ).sort("timestamp")

    greek_df = greek_df.group_by_dynamic("timestamp", every="1h").agg(pl.all().mean())
    neighbor_df = neighbor_df.group_by_dynamic("timestamp", every="1h").agg(pl.all().mean())

    logger.info(f"Greek data: {greek_df['timestamp'].min()} to {greek_df['timestamp'].max()}")

    # I ensure required columns exist
    if "wind_onshore_forecast_mw" in greek_df.columns and "wind_forecast_mw" not in greek_df.columns:
        greek_df = greek_df.rename({"wind_onshore_forecast_mw": "wind_forecast_mw"})

    for col in ["wind_forecast_mw", "solar_forecast_mw", "load_forecast_mw"]:
        if col not in greek_df.columns:
            logger.warning(f"Column {col} missing — filling with zeros")
            greek_df = greek_df.with_columns(pl.lit(0.0).alias(col))

    # Step 2: I load all 24 XGBoost models ONCE
    logger.info("Loading 24 XGBoost hourly models...")
    models = []
    for hour in range(24):
        model_path = MODELS_DIR / f"xgboost_hour_{hour:02d}.json"
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        models.append(model)
    logger.info("All models loaded")

    # Step 3: I loop through all target dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start

    all_forecasts = []
    n_success = 0
    n_fail = 0

    def safe_extract_pad(df, col_name):
        """I extract a column and pad to exactly 24 values."""
        arr = df[col_name].to_numpy()
        if len(arr) == 24:
            return arr
        elif len(arr) > 24:
            return arr[:24]
        elif len(arr) > 0:
            return np.pad(arr, (0, 24 - len(arr)), mode='edge')
        else:
            return np.zeros(24)

    total_days = (end - start).days
    logger.info(f"Generating forecasts for {total_days} days...")

    while current < end:
        target_date = current.replace(tzinfo=timezone.utc)
        submission_date = target_date - timedelta(days=1)

        try:
            # I build the encoder (7 days of history up to D-1 12:00)
            encoder_start = submission_date - timedelta(days=7)
            encoder_end = submission_date.replace(hour=12, minute=0, second=0, microsecond=0)

            greek_enc = greek_df.filter(
                (pl.col("timestamp") >= encoder_start) &
                (pl.col("timestamp") <= encoder_end)
            )

            if len(greek_enc) == 0:
                n_fail += 1
                current += timedelta(days=1)
                continue

            encoder_prices = greek_enc["price_eur_mwh"].to_numpy()

            # I get decoder data (target day forecasts)
            greek_dec = greek_df.filter(
                pl.col("timestamp").dt.date() == target_date.date()
            )

            if len(greek_dec) == 0:
                n_fail += 1
                current += timedelta(days=1)
                continue

            decoder_load = safe_extract_pad(greek_dec, "load_forecast_mw")
            decoder_solar = safe_extract_pad(greek_dec, "solar_forecast_mw")
            decoder_wind = safe_extract_pad(greek_dec, "wind_forecast_mw")

            # I get neighbor prices
            neighbor_dec = neighbor_df.filter(
                pl.col("timestamp").dt.date() == target_date.date()
            )

            dam_bg = safe_extract_pad(neighbor_dec, "dam_price_bg") if "dam_price_bg" in neighbor_dec.columns else np.zeros(24)
            dam_ro = safe_extract_pad(neighbor_dec, "dam_price_ro") if "dam_price_ro" in neighbor_dec.columns else np.zeros(24)
            dam_it = safe_extract_pad(neighbor_dec, "dam_price_it_south") if "dam_price_it_south" in neighbor_dec.columns else np.zeros(24)

            # I build the sample
            sample = DAMSampleV2(
                submission_date=submission_date,
                target_date=target_date,
                encoder_prices=encoder_prices,
                encoder_load=np.zeros(1),
                encoder_solar=np.zeros(1),
                encoder_wind=np.zeros(1),
                encoder_gas=np.zeros(1),
                encoder_timestamps=[],
                neighbor_dam_bg=dam_bg,
                neighbor_dam_ro=dam_ro,
                neighbor_dam_hu=np.zeros(24),
                neighbor_dam_rs=np.zeros(24),
                neighbor_dam_mk=np.zeros(24),
                neighbor_dam_it=dam_it,
                neighbor_dam_de=np.zeros(24),
                neighbor_balkans_avg=np.zeros(24),
                decoder_load=decoder_load,
                decoder_solar=decoder_solar,
                decoder_wind=decoder_wind,
                target_prices=np.zeros(24)
            )

            # I run predictions for all 24 hours
            for hour in range(24):
                features = extract_features_v2(sample, hour)
                pred = float(models[hour].predict(np.array([features]))[0])
                ts = datetime(current.year, current.month, current.day, hour)
                all_forecasts.append({
                    'timestamp': ts,
                    'forecast_price': pred
                })

            n_success += 1

        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                logger.warning(f"Failed for {current.date()}: {e}")

        current += timedelta(days=1)

        # I log progress every 100 days
        if n_success % 100 == 0 and n_success > 0:
            logger.info(f"  Progress: {n_success}/{total_days} days done")

    logger.info(f"Completed: {n_success} success, {n_fail} failed out of {total_days} days")

    # Step 4: I save results
    result_df = pd.DataFrame(all_forecasts)
    result_df = result_df.set_index('timestamp').sort_index()
    result_df.to_csv(output_path)
    logger.info(f"Saved {len(result_df)} hourly forecasts to {output_path}")

    # I print summary statistics
    print(f"\nForecast Statistics:")
    print(f"  Rows: {len(result_df)}")
    print(f"  Date range: {result_df.index.min()} to {result_df.index.max()}")
    print(f"  Mean: {result_df['forecast_price'].mean():.2f} EUR/MWh")
    print(f"  Std: {result_df['forecast_price'].std():.2f}")
    print(f"  Min: {result_df['forecast_price'].min():.2f}")
    print(f"  Max: {result_df['forecast_price'].max():.2f}")

    return result_df


if __name__ == "__main__":
    generate_all_forecasts()
