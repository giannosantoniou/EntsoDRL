"""Train all forecasters on merged data.

Usage:
    python -m PredictFuturePrices.train [--data PATH] [--output-dir PATH] [--forecasters dam,load,ida,xbid]

I load the merged training CSV, split into train/val/test, and train
each forecaster. Models are saved to the models/ directory.
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from PredictFuturePrices.config import PATHS

logger = logging.getLogger(__name__)

ALL_FORECASTERS = ["dam", "load", "ida", "xbid"]


def load_training_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """I load and validate the merged training DataFrame."""
    path = data_path or PATHS.merged_csv

    if not path.exists():
        # I try to fall back to the project's existing training data
        fallbacks = [
            PATHS.data_dir.parent.parent / "data" / "unified_multimarket_training_v2.csv",
            PATHS.data_dir.parent.parent / "data" / "unified_multimarket_training.csv",
        ]
        for fb in fallbacks:
            if fb.exists():
                logger.info(f"Using fallback data: {fb}")
                path = fb
                break
        else:
            raise FileNotFoundError(
                f"No training data found at {path}. "
                "Run data collection and merger first."
            )

    logger.info(f"Loading training data from {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # I map column names from the existing project data to what forecasters expect
    column_aliases = {
        "serbia_dam_price": "rs_dam_price",
        "load_mw": "system_load_mw",
        "price_std_24h": "dam_price_gr_std_24h",
    }
    for old_name, new_name in column_aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # I also create dam_price_gr alias if missing
    if "dam_price_gr" not in df.columns and "price" in df.columns:
        df["dam_price_gr"] = df["price"]

    # I compute rs_gr_spread if both sides are available
    if "rs_dam_price" in df.columns and "dam_price_gr" in df.columns:
        if "rs_gr_spread" not in df.columns:
            df["rs_gr_spread"] = df["rs_dam_price"] - df["dam_price_gr"]

    # I detect resolution: sph=4 for 15-min, sph=1 for hourly
    sph = 1
    freq = pd.infer_freq(df.index[:100])
    if freq and ("15" in str(freq) or "T" in str(freq) or "min" in str(freq)):
        sph = 4
        logger.info(f"Detected 15-min resolution (sph={sph}, {len(df)} rows)")

    # I compute window sizes based on resolution
    window_24h = 24 * sph

    # I add rolling stats if missing (needed by DAM forecaster)
    dam_col = "dam_price_gr"
    if dam_col in df.columns:
        if f"{dam_col}_mean_24h" not in df.columns:
            df[f"{dam_col}_mean_24h"] = df[dam_col].rolling(window_24h, min_periods=1).mean()
        if f"{dam_col}_std_24h" not in df.columns:
            df[f"{dam_col}_std_24h"] = df[dam_col].rolling(window_24h, min_periods=1).std()
        if f"{dam_col}_min_24h" not in df.columns:
            df[f"{dam_col}_min_24h"] = df[dam_col].rolling(window_24h, min_periods=1).min()
        if f"{dam_col}_max_24h" not in df.columns:
            df[f"{dam_col}_max_24h"] = df[dam_col].rolling(window_24h, min_periods=1).max()

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def split_data(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15):
    """I split data into train/val/test by time."""
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    logger.info(
        f"Split: train=0:{train_end} ({train_end}), "
        f"val={train_end}:{val_end} ({val_end - train_end}), "
        f"test={val_end}:{n} ({n - val_end})"
    )

    return train_end, val_end


def train_forecaster(name: str, df: pd.DataFrame, train_end: int, val_end: int, output_dir: Path):
    """I train a single forecaster and save the model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {name.upper()} forecaster")
    logger.info(f"{'='*60}")

    start_time = time.time()

    if name == "dam":
        from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster
        fc = DAMForecaster()
    elif name == "load":
        from PredictFuturePrices.forecasters.load_forecaster import LoadForecaster
        fc = LoadForecaster()
    elif name == "ida":
        from PredictFuturePrices.forecasters.ida_forecaster import IDAForecaster
        fc = IDAForecaster()
    elif name == "xbid":
        from PredictFuturePrices.forecasters.xbid_forecaster import XBIDForecaster
        fc = XBIDForecaster()
    else:
        logger.error(f"Unknown forecaster: {name}")
        return None

    try:
        fc.fit(df, train_end, val_end)
    except ImportError as e:
        logger.error(f"[{name}] Missing dependency: {e}")
        return None

    elapsed = time.time() - start_time
    logger.info(f"[{name}] Training completed in {elapsed:.1f}s")

    # I save the model
    model_path = output_dir / f"{name}_forecaster.pkl"
    fc.save(model_path)

    # I compute test metrics
    test_end = min(len(df), val_end + 500)
    if test_end > val_end + 50:
        logger.info(f"[{name}] Computing test metrics ({val_end}:{test_end})")
        kwargs = {}
        if name == "dam":
            # I use midday period, adapting to detected resolution (12 for hourly, 48 for 15-min)
            kwargs = {"period": fc.n_periods // 2}
        elif name == "load":
            kwargs = {"period": fc.n_periods // 2}
        elif name == "ida":
            kwargs = {"ida_number": 1}
        elif name == "xbid":
            kwargs = {"horizon": 1}

        metrics = fc.evaluate(df, val_end, test_end, **kwargs)
        logger.info(f"[{name}] Test metrics: {metrics}")
    else:
        metrics = fc.train_metrics

    n_models = len(fc.models)

    # I free the forecaster and its models immediately after saving
    del fc
    gc.collect()

    return {
        "name": name,
        "model_path": str(model_path),
        "elapsed_s": elapsed,
        "metrics": metrics,
        "n_models": n_models,
    }


def train_all(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    forecasters: Optional[List[str]] = None,
):
    """I train all selected forecasters and return a summary."""
    output_dir = output_dir or PATHS.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasters = forecasters or ALL_FORECASTERS

    df = load_training_data(data_path)
    train_end, val_end = split_data(df)

    results = []
    for name in forecasters:
        result = train_forecaster(name, df, train_end, val_end, output_dir)
        if result:
            results.append(result)

    # I print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        metrics = r.get("metrics", {})
        mae = metrics.get("mae", float("nan"))
        print(f"  {r['name']:8s} | {r['n_models']:3d} models | "
              f"MAE={mae:8.2f} | {r['elapsed_s']:6.1f}s | {r['model_path']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train all forecasters")
    parser.add_argument("--data", type=str, help="Path to merged training CSV")
    parser.add_argument("--output-dir", type=str, help="Directory for saved models")
    parser.add_argument(
        "--forecasters", type=str, default=",".join(ALL_FORECASTERS),
        help=f"Comma-separated forecaster names ({','.join(ALL_FORECASTERS)})"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # I suppress LightGBM's internal logger to prevent stdout/stderr flood
    # that can crash Claude Code's Node.js process with heap OOM
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

    # I suppress sklearn's "X does not have valid feature names" warnings
    # that flood output during evaluation (500k+ lines)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    data_path = Path(args.data) if args.data else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    forecasters = args.forecasters.split(",")

    train_all(data_path=data_path, output_dir=output_dir, forecasters=forecasters)


if __name__ == "__main__":
    main()
