"""Run predictions and record accuracy.

Usage:
    python -m PredictFuturePrices.predict [--data PATH] [--models-dir PATH] [--fill-actuals]

I load trained models, generate predictions for the latest data point,
record them in the accuracy tracker, and optionally fill actuals from
available data.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from PredictFuturePrices.accuracy.tracker import AccuracyTracker
from PredictFuturePrices.config import PATHS

logger = logging.getLogger(__name__)


def load_models(models_dir: Optional[Path] = None) -> Dict:
    """I load all trained forecaster models from disk."""
    models_dir = models_dir or PATHS.models_dir
    loaded = {}

    model_configs = {
        "dam": ("dam_forecaster.pkl", "PredictFuturePrices.forecasters.dam_forecaster", "DAMForecaster"),
        "load": ("load_forecaster.pkl", "PredictFuturePrices.forecasters.load_forecaster", "LoadForecaster"),
        "ida": ("ida_forecaster.pkl", "PredictFuturePrices.forecasters.ida_forecaster", "IDAForecaster"),
        "xbid": ("xbid_forecaster.pkl", "PredictFuturePrices.forecasters.xbid_forecaster", "XBIDForecaster"),
    }

    for name, (filename, module_path, class_name) in model_configs.items():
        model_path = models_dir / filename
        if not model_path.exists():
            logger.warning(f"[{name}] Model not found: {model_path}")
            continue

        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            fc = cls()
            fc.load(model_path)
            loaded[name] = fc
            logger.info(f"[{name}] Loaded ({len(fc.models)} models)")
        except Exception as e:
            logger.error(f"[{name}] Failed to load: {e}")

    return loaded


def run_predictions(
    df: pd.DataFrame,
    forecasters: Dict,
    tracker: AccuracyTracker,
    predict_idx: Optional[int] = None,
) -> Dict[str, Dict]:
    """I generate predictions from all loaded forecasters and record them.

    Args:
        df: Merged DataFrame with latest data.
        forecasters: Dict of name → fitted forecaster.
        tracker: AccuracyTracker instance.
        predict_idx: Index to predict from (default: latest available).

    Returns:
        Dict of forecaster name → prediction results.
    """
    if predict_idx is None:
        predict_idx = len(df) - 25  # I leave room for target extraction

    if predict_idx < 168:
        logger.error("Not enough data to predict (need at least 168 rows)")
        return {}

    now = datetime.now()
    predict_ts = df.index[predict_idx]
    results = {}

    for name, fc in forecasters.items():
        logger.info(f"[{name}] Predicting from index {predict_idx} ({predict_ts})")

        try:
            preds = fc.predict(df, predict_idx)
            results[name] = preds

            # I record predictions in accuracy tracker
            _record_predictions(name, fc, preds, tracker, now, predict_ts)

        except Exception as e:
            logger.error(f"[{name}] Prediction failed: {e}")

    return results


def _record_predictions(
    name: str,
    fc,
    preds: Dict,
    tracker: AccuracyTracker,
    now: datetime,
    predict_ts: pd.Timestamp,
):
    """I record predictions in the accuracy tracker."""
    if name == "dam":
        # I record 24 hourly predictions
        point = preds.get("point", np.zeros(24))
        targets = [predict_ts + pd.Timedelta(hours=h + 1) for h in range(24)]
        horizons = [float(h + 1) for h in range(24)]
        tracker.record_batch("dam", now, targets, point, horizons)
        logger.info(f"[dam] Recorded {len(targets)} predictions "
                    f"(mean={np.mean(point):.1f} EUR/MWh)")

    elif name == "load":
        point = preds.get("point", np.zeros(24))
        targets = [predict_ts + pd.Timedelta(hours=h + 1) for h in range(24)]
        horizons = [float(h + 1) for h in range(24)]
        tracker.record_batch("load", now, targets, point, horizons)
        logger.info(f"[load] Recorded {len(targets)} predictions "
                    f"(mean={np.mean(point):.0f} MW)")

    elif name == "ida":
        for ida_num in [1, 2, 3]:
            key = f"ida{ida_num}"
            if key in preds:
                value = float(preds[key][0])
                target_ts = predict_ts + pd.Timedelta(hours=1)
                tracker.record_prediction(
                    f"ida{ida_num}", now, target_ts, value,
                    horizon_hours=1.0,
                )
                logger.info(f"[ida{ida_num}] Recorded prediction: {value:.1f}")

    elif name == "xbid":
        for horizon_key, pred_arr in preds.items():
            horizon_h = float(horizon_key.replace("h", ""))
            value = float(pred_arr[0])
            target_ts = predict_ts + pd.Timedelta(hours=horizon_h)
            tracker.record_prediction(
                "xbid", now, target_ts, value,
                horizon_hours=horizon_h,
            )
        logger.info(f"[xbid] Recorded {len(preds)} horizon predictions")


def fill_actuals_from_data(
    df: pd.DataFrame,
    tracker: AccuracyTracker,
):
    """I fill actual values from the merged DataFrame.

    I scan the DataFrame for timestamps that match unfilled predictions
    and update the accuracy tracker.
    """
    dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
    load_col = None
    for col in ["system_load_mw", "load_mw", "load_forecast"]:
        if col in df.columns:
            load_col = col
            break

    # I build actuals dicts from the DataFrame
    if dam_col in df.columns:
        dam_actuals = {
            ts.to_pydatetime(): float(row[dam_col])
            for ts, row in df.iterrows()
            if not pd.isna(row[dam_col])
        }
        updated = tracker.fill_actuals("dam", dam_actuals)
        if updated:
            logger.info(f"[dam] Filled {updated} actuals")

        # I also use DAM for IDA (as proxy)
        for ida_name in ["ida1", "ida2", "ida3"]:
            updated = tracker.fill_actuals(ida_name, dam_actuals)
            if updated:
                logger.info(f"[{ida_name}] Filled {updated} actuals")

    if load_col:
        load_actuals = {
            ts.to_pydatetime(): float(row[load_col])
            for ts, row in df.iterrows()
            if not pd.isna(row[load_col])
        }
        updated = tracker.fill_actuals("load", load_actuals)
        if updated:
            logger.info(f"[load] Filled {updated} actuals")

    # I fill XBID actuals from xbid_price_bid or DAM
    xbid_col = None
    for col in ["xbid_price_bid", "xbid_price_mid", dam_col]:
        if col in df.columns:
            xbid_col = col
            break

    if xbid_col:
        xbid_actuals = {
            ts.to_pydatetime(): float(row[xbid_col])
            for ts, row in df.iterrows()
            if not pd.isna(row[xbid_col])
        }
        updated = tracker.fill_actuals("xbid", xbid_actuals)
        if updated:
            logger.info(f"[xbid] Filled {updated} actuals")


def main():
    parser = argparse.ArgumentParser(description="Run predictions and track accuracy")
    parser.add_argument("--data", type=str, help="Path to merged CSV")
    parser.add_argument("--models-dir", type=str, help="Directory with trained models")
    parser.add_argument("--fill-actuals", action="store_true", help="Fill actuals from data")
    parser.add_argument("--summary", action="store_true", help="Print accuracy summary")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # I load data
    from PredictFuturePrices.train import load_training_data
    data_path = Path(args.data) if args.data else None
    df = load_training_data(data_path)

    # I load models
    models_dir = Path(args.models_dir) if args.models_dir else None
    forecasters = load_models(models_dir)

    if not forecasters:
        logger.error("No forecasters loaded. Train models first.")
        return

    # I set up tracker
    tracker = AccuracyTracker()

    # I run predictions
    results = run_predictions(df, forecasters, tracker)

    # I optionally fill actuals
    if args.fill_actuals:
        fill_actuals_from_data(df, tracker)

    # I optionally print summary
    if args.summary or args.fill_actuals:
        summary = tracker.generate_summary()
        if not summary.empty:
            print("\n" + "=" * 60)
            print("ACCURACY SUMMARY")
            print("=" * 60)
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
