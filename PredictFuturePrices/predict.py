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
from PredictFuturePrices.config import (
    PATHS, IDA_FORECASTER,
    MarketEvent, SCHEDULER,
)

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


def _should_run_forecaster(
    name: str,
    market_event: MarketEvent,
) -> bool:
    """I check whether a forecaster should run for the given market event.

    The IDA forecaster has special handling: "ida" (full) matches any
    IDA-related key, and individual "ida1"/"ida2"/"ida3" keys match exactly.
    """
    allowed = SCHEDULER.event_forecasters.get(
        market_event, SCHEDULER.event_forecasters[MarketEvent.ALL]
    )

    if name == "ida":
        # I run the IDA forecaster if any IDA key is allowed
        return any(k.startswith("ida") for k in allowed)
    return name in allowed


def _get_ida_filter(market_event: MarketEvent) -> Optional[List[int]]:
    """I return the IDA auction numbers to predict, or None for all.

    When the event specifies a specific IDA (e.g. "ida1"), I filter to
    only that auction. When "ida" (full) is in the list, I return None
    (= run all 3).
    """
    if market_event == MarketEvent.ALL:
        return None

    allowed = SCHEDULER.event_forecasters.get(
        market_event, SCHEDULER.event_forecasters[MarketEvent.ALL]
    )

    if "ida" in allowed:
        return None  # I run all 3 auctions

    # I collect specific IDA numbers (e.g. "ida1" → 1, "ida3" → 3)
    ida_nums = []
    for key in allowed:
        if key.startswith("ida") and key[3:].isdigit():
            ida_nums.append(int(key[3:]))

    return ida_nums if ida_nums else None


def _predict_filtered_ida(
    fc,
    df: pd.DataFrame,
    predict_idx: int,
    ida_filter: Optional[List[int]],
) -> Dict:
    """I run IDA prediction, optionally filtering to specific auctions.

    When ida_filter is None, I run the full predict (all 3 auctions).
    Otherwise, I predict only the specified auctions.
    """
    full_preds = fc.predict(df, predict_idx)
    if ida_filter is None:
        return full_preds

    # I keep only the requested auctions
    return {k: v for k, v in full_preds.items()
            if any(k == f"ida{n}" for n in ida_filter)}


def run_predictions(
    df: pd.DataFrame,
    forecasters: Dict,
    tracker: AccuracyTracker,
    predict_idx: Optional[int] = None,
    market_event: MarketEvent = MarketEvent.ALL,
) -> Dict[str, Dict]:
    """I generate predictions from loaded forecasters and record them.

    Args:
        df: Merged DataFrame with latest data.
        forecasters: Dict of name → fitted forecaster.
        tracker: AccuracyTracker instance.
        predict_idx: Index to predict from (default: latest available).
        market_event: Which market event triggered this cycle.
            Controls which forecasters actually run.

    Returns:
        Dict of forecaster name → prediction results.
    """
    # I derive min_history and predict_idx from loaded models' metadata
    # so hourly models (sph=1) don't require 672 rows they can't fill
    max_sph = max(
        (getattr(fc, 'sph', 1) for fc in forecasters.values()),
        default=1,
    )
    min_history = 7 * 24 * max_sph
    if predict_idx is None:
        max_periods = max(
            (getattr(fc, 'n_periods', 24) for fc in forecasters.values()),
            default=24,
        )
        predict_idx = len(df) - max_periods - 1  # I leave room for target extraction

    if predict_idx < min_history:
        logger.error(f"Not enough data to predict (need at least {min_history} rows)")
        return {}

    now = datetime.now()
    predict_ts = df.index[predict_idx]
    results = {}

    for name, fc in forecasters.items():
        if not _should_run_forecaster(name, market_event):
            logger.info(f"[{name}] Skipped (event={market_event.value})")
            continue

        logger.info(f"[{name}] Predicting from index {predict_idx} ({predict_ts})")

        try:
            if name == "ida":
                ida_filter = _get_ida_filter(market_event)
                preds = _predict_filtered_ida(fc, df, predict_idx, ida_filter)
            else:
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
    """I record predictions in the accuracy tracker.

    I derive resolution and period count from the loaded forecaster's
    metadata (fc.sph, fc.n_periods) instead of config constants.  This
    handles the case where models were trained on hourly data (sph=1,
    n_periods=24) but config expects 15-min (sph=4, n_periods=96).
    """
    # I read the model's actual resolution, not the config target
    res_min = 60 // getattr(fc, 'sph', 1)
    n_periods = getattr(fc, 'n_periods', 24)

    if name == "dam":
        point = preds.get("point", np.zeros(n_periods))
        targets = [predict_ts + pd.Timedelta(minutes=(p + 1) * res_min) for p in range(n_periods)]
        horizons = [float((p + 1) * res_min / 60) for p in range(n_periods)]
        tracker.record_batch("dam", now, targets, point, horizons)
        logger.info(f"[dam] Recorded {n_periods} predictions "
                    f"(mean={np.mean(point):.1f} EUR/MWh)")

    elif name == "load":
        point = preds.get("point", np.zeros(n_periods))
        targets = [predict_ts + pd.Timedelta(minutes=(p + 1) * res_min) for p in range(n_periods)]
        horizons = [float((p + 1) * res_min / 60) for p in range(n_periods)]
        tracker.record_batch("load", now, targets, point, horizons)
        logger.info(f"[load] Recorded {n_periods} predictions "
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

    # I build actuals dicts from the DataFrame.
    # For IDA, I use the same blended formula as IDAForecaster._extract_target():
    #   ida_actual = alpha * DAM + (1 - alpha) * ISP1_mid
    has_isp1 = ("isp1_price_up" in df.columns and "isp1_price_down" in df.columns)
    ida_actuals = {n: {} for n in [1, 2, 3]}

    if dam_col in df.columns:
        dam_actuals = {}
        for ts, row in df.iterrows():
            dam_val = row[dam_col]
            if pd.isna(dam_val):
                continue
            dam_actuals[ts.to_pydatetime()] = float(dam_val)

            # I compute blended IDA actuals in the same pass
            isp1_mid = float(dam_val)  # fallback when ISP1 unavailable
            if has_isp1:
                up = row["isp1_price_up"]
                down = row["isp1_price_down"]
                if not (pd.isna(up) or pd.isna(down)):
                    isp1_mid = (float(up) + float(down)) / 2

            for ida_num in [1, 2, 3]:
                alpha = IDA_FORECASTER.blending_alphas.get(ida_num, 0.7)
                blended = alpha * float(dam_val) + (1 - alpha) * isp1_mid
                ida_actuals[ida_num][ts.to_pydatetime()] = blended

        updated = tracker.fill_actuals("dam", dam_actuals)
        if updated:
            logger.info(f"[dam] Filled {updated} actuals")

        # I fill IDA actuals with blended values
        for ida_num in [1, 2, 3]:
            ida_name = f"ida{ida_num}"
            updated = tracker.fill_actuals(ida_name, ida_actuals[ida_num])
            if updated:
                logger.info(f"[{ida_name}] Filled {updated} actuals (blended α={IDA_FORECASTER.blending_alphas[ida_num]:.2f})")

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
