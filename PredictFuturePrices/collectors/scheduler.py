"""Scheduler that orchestrates all data collectors.

I run all collectors in sequence, handling failures gracefully.
Each collector is independent — if one fails, the rest still run.

Usage:
    python -m PredictFuturePrices.collectors.scheduler --once
"""

import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from PredictFuturePrices.config import PATHS, SCHEDULER, MarketEvent

logger = logging.getLogger(__name__)


def run_all_collectors(
    end: Optional[datetime] = None,
    market_event: Optional[MarketEvent] = None,
) -> Dict[str, int]:
    """I run all collectors and return a summary of rows fetched.

    I import collectors lazily to avoid import errors when
    optional dependencies (yfinance, entsoe-py) aren't installed.

    Args:
        end: End date for incremental fetch. Defaults to now.
        market_event: If provided, I only run collectors relevant to this
            event (from SCHEDULER.event_collectors). None = run all (backward compat).

    Returns:
        Dict mapping collector name → rows fetched (or -1 on error).
    """
    if end is None:
        end = datetime.now()

    results = {}

    # I define collectors with lazy imports to handle missing dependencies
    collector_factories = {
        "weather": _create_weather_collector,
        "commodity": _create_commodity_collector,
        "entsoe_gr_dam": _create_entsoe_gr_dam_collector,
        "entsoe_rs_dam": _create_entsoe_rs_dam_collector,
        "admie_balancing": _create_admie_balancing_collector,
        "henex_ida_clearing": _create_henex_ida_collector,
    }

    # I filter collectors based on market event if provided
    if market_event is not None:
        allowed = SCHEDULER.event_collectors.get(market_event)
        if allowed is not None:
            collector_factories = {
                name: factory
                for name, factory in collector_factories.items()
                if name in allowed
            }
            logger.info(
                f"[collectors] Event {market_event.value} → "
                f"running {list(collector_factories.keys())}"
            )

    for name, factory in collector_factories.items():
        logger.info(f"--- Running collector: {name} ---")
        try:
            collector = factory()
            new_data = collector.fetch_incremental(end=end)
            rows = len(new_data) if new_data is not None else 0
            results[name] = rows
            logger.info(f"[{name}] Fetched {rows} new rows")
        except ImportError as e:
            logger.warning(f"[{name}] Skipped (missing dependency): {e}")
            results[name] = -1
        except Exception as e:
            logger.error(f"[{name}] Failed: {e}")
            results[name] = -1

        # I add a small delay between collectors to be a good API citizen
        time.sleep(1)

    return results


def _create_weather_collector():
    from PredictFuturePrices.collectors.weather_collector import WeatherCollector
    return WeatherCollector()


def _create_commodity_collector():
    from PredictFuturePrices.collectors.commodity_collector import CommodityCollector
    return CommodityCollector()


def _create_entsoe_gr_dam_collector():
    from PredictFuturePrices.collectors.entsoe_collector import EntsoeDAMCollector
    return EntsoeDAMCollector()


def _create_entsoe_rs_dam_collector():
    from PredictFuturePrices.collectors.entsoe_collector import EntsoeSerbiaDAMCollector
    collector = EntsoeSerbiaDAMCollector()
    # I wrap it to expose fetch_incremental
    from PredictFuturePrices.collectors.entsoe_collector import _IncrementalWrapper
    return _IncrementalWrapper(collector)


def _create_admie_balancing_collector():
    from PredictFuturePrices.collectors.admie_collector import AdmieBalancingCollector
    return AdmieBalancingCollector()


def _create_henex_ida_collector():
    from PredictFuturePrices.collectors.henex_collector import HenexIDACollector
    return HenexIDACollector()


def main():
    parser = argparse.ArgumentParser(description="Run all data collectors")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=3600, help="Interval in seconds (default: 1h)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.once:
        results = run_all_collectors()
        print("\n=== Collection Summary ===")
        for name, rows in results.items():
            status = f"{rows} rows" if rows >= 0 else "FAILED"
            print(f"  {name}: {status}")
    else:
        logger.info(f"Starting scheduler (interval={args.interval}s)")
        while True:
            run_all_collectors()
            logger.info(f"Sleeping {args.interval}s until next run...")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
