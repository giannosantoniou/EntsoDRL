"""
IDA Schedule Forecasters — I predict updated prices at each IDA gate closure.

I implement 3 forecasters, one per IDA gate:
  IDA1 (D-1 15:00): +GR DAM actual results, +Serbia D+1 confirmed
  IDA2 (D-1 22:00): +ISP1 clearing, +evening actuals
  IDA3 (D+0 10:00): +morning actuals, +ISP2, +actual SoC

Each forecaster predicts 24 (or 12) hourly prices. The ScheduleOptimizer
then converts to a schedule and computes delta vs the locked cumulative.

Architecture: 1 LightGBM model per gate × 1 model per gate = 3 models total.
Each model includes hour as feature (same pattern as DAMScheduleForecaster).

Target: Actual DAM price (same as DAM forecaster). The improvement comes from
BETTER prediction with MORE information → better schedule → profitable delta.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class IDAScheduleForecaster:
    """I predict updated hourly prices at an IDA gate closure.

    I extend the DAM feature set with new information available at each gate.
    The key principle: each later gate has STRICTLY MORE features than the previous.

    IDA1 (gate=1): DAM features + GR DAM actual + Serbia D+1 hourly (confirmed)
    IDA2 (gate=2): IDA1 features + ISP1 clearing + evening load/RES actuals
    IDA3 (gate=3): IDA2 features + morning price actuals + ISP2 + actual SoC
    """

    # I define the feature groups per gate
    # Each gate's features = previous gate + new group
    BASE_FEATURES = [
        # Same as DAM (43 features, indices 0-42)
        "dam_lag_1d", "dam_lag_2d", "dam_lag_3d", "dam_lag_7d",
        "dam_mean_24h", "dam_std_24h", "dam_min_24h", "dam_max_24h", "dam_std_6h",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_peak_hour", "is_solar_hour",
        "load_forecast_norm", "res_forecast_norm",
        "temperature", "wind_100m", "solar_radiation", "cloud_cover",
        "ttf_gas", "eua_carbon",
        "ttf_gas_change_24h", "ttf_gas_change_7d",
        "dam_bg", "dam_ro", "dam_hu", "dam_it_south",
        "dam_bg_d1", "dam_ro_d1", "dam_hu_d1", "dam_it_south_d1",
        "rs_d1_target_hour",
        "rs_d1_mean", "rs_d1_max", "rs_d1_min", "rs_d1_spread",
        "rs_gr_spread",
        "load_fcst_error_lag", "res_fcst_error_lag",
        "target_hour_sin", "target_hour_cos",
    ]

    IDA1_NEW_FEATURES = [
        # GR DAM actual D+1 (4) — published ~14:00, available at 15:00
        "dam_actual_target_hour",     # Actual DAM price for THIS target hour
        "dam_actual_mean", "dam_actual_spread", "dam_actual_peak_price",
        # DAM schedule context (3) — what did the optimizer commit?
        "dam_sched_this_hour",        # MW committed for this hour
        "dam_sched_total_buy_mwh", "dam_sched_total_sell_mwh",
        # Weather forecast deviation (4) — key IDA signal!
        # At 15:00, we see 3h of actual weather → deviation from DAM-time forecast
        "wind_deviation_recent",      # Wind changed since DAM decision
        "solar_deviation_recent",     # Solar changed since DAM decision
        "load_deviation_recent",      # Load changed since DAM decision
        "res_surprise_recent",        # RES surprise magnitude
    ]

    IDA2_NEW_FEATURES = [
        # ISP1 clearing (3) — published ~18:00 for D delivery
        "isp1_up", "isp1_down", "isp1_spread",
        # Weather deviation UPDATED at 22:00 (5) — 7h more data since IDA1
        "evening_wind_deviation",     # Wind changed since morning forecast
        "evening_solar_deviation",    # Solar actual vs expected
        "evening_load_deviation",     # Load actual vs expected
        "wind_trend_evening",         # Wind direction (rising/falling)
        "res_surprise_evening",       # How wrong was DAM's RES assumption?
        # Imbalance signal (2)
        "recent_imbalance_mean", "recent_imbalance_price",
    ]

    IDA3_NEW_FEATURES = [
        # Morning actual prices (3) — D+0 00:00-10:00 known!
        "morning_price_mean", "morning_price_max", "morning_price_vs_dam_forecast",
        # ISP2 clearing (2)
        "isp2_up", "isp2_down",
        # Morning actuals (2) — D+0 actual RES
        "morning_wind_actual", "morning_solar_actual",
        # Afternoon weather proxy (3) — morning actuals predict afternoon (corr 0.83-0.88)
        # At 10:00, short-range weather forecast for 12:00-24:00 is very accurate
        "afternoon_solar_proxy",      # Morning solar × seasonal afternoon ratio
        "afternoon_wind_proxy",       # Morning wind (persistent, corr 0.83)
        "afternoon_res_impact",       # Expected RES impact on afternoon prices
        # Execution reality (2) — how did the night go?
        "current_soc", "overnight_imbalance_mean",
    ]

    GATE_CONFIG = {
        1: {"gate_hour": 15, "n_target_hours": 24, "label": "IDA1"},
        2: {"gate_hour": 22, "n_target_hours": 24, "label": "IDA2"},
        3: {"gate_hour": 10, "n_target_hours": 12, "label": "IDA3"},
    }

    def __init__(self, ida_number: int):
        """I initialize for a specific IDA gate (1, 2, or 3)."""
        if ida_number not in self.GATE_CONFIG:
            raise ValueError(f"ida_number must be 1, 2, or 3, got {ida_number}")

        self.ida_number = ida_number
        self.config = self.GATE_CONFIG[ida_number]

        # I build the full feature list for this gate
        self.feature_names = list(self.BASE_FEATURES)
        self.feature_names.extend(self.IDA1_NEW_FEATURES)
        if ida_number >= 2:
            self.feature_names.extend(self.IDA2_NEW_FEATURES)
        if ida_number >= 3:
            self.feature_names.extend(self.IDA3_NEW_FEATURES)

        self.n_features = len(self.feature_names)
        self.model = None
        self.is_fitted = False
        self.train_metrics: Dict[str, float] = {}

        logger.info(f"[{self.config['label']}] Initialized: {self.n_features} features, "
                     f"gate={self.config['gate_hour']}:00, "
                     f"target_hours={self.config['n_target_hours']}")

    def build_features_for_day(
        self,
        df: pd.DataFrame,
        decision_idx: int,
        dam_schedule: Optional[np.ndarray] = None,
        current_soc: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """I build feature matrix for all target hours from one IDA decision point.

        Args:
            df: DataFrame with price/market data
            decision_idx: Index at gate closure time
            dam_schedule: Locked DAM schedule [96 slots] (needed for IDA1+)
            current_soc: Battery SoC at decision time (needed for IDA3)

        Returns:
            (X, hours) where X.shape = (n_hours, n_features)
        """
        sph = self._detect_sph(df)
        min_history = 7 * 24 * sph
        if decision_idx < min_history:
            return None

        row = df.iloc[decision_idx]
        ts = df.index[decision_idx]
        w24 = 24 * sph
        n_hours = self.config["n_target_hours"]
        hour_offset = 12 if self.ida_number == 3 else 0

        pcol = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        prices = df[pcol].values if pcol in df.columns else None

        X = np.zeros((n_hours, self.n_features))

        for hi, h in enumerate(range(hour_offset, hour_offset + n_hours)):
            fi = 0

            # === BASE FEATURES (same as DAM, 45 features) ===

            # GR DAM lags (4)
            if prices is not None:
                for lag_days in [1, 2, 3, 7]:
                    lag_idx = decision_idx - lag_days * w24 + h * sph
                    if 0 <= lag_idx < len(prices):
                        X[hi, fi] = prices[lag_idx] / 200.0
                    fi += 1
            else:
                fi += 4

            # Rolling stats (5)
            if prices is not None:
                window = prices[max(0, decision_idx - w24):decision_idx]
                if len(window) > 0:
                    X[hi, fi] = np.mean(window)
                    X[hi, fi+1] = np.std(window) if len(window) > 1 else 0
                    X[hi, fi+2] = np.min(window)
                    X[hi, fi+3] = np.max(window)
                w6 = 6 * sph
                w6_data = prices[max(0, decision_idx - w6):decision_idx]
                if len(w6_data) > 1:
                    X[hi, fi+4] = np.std(w6_data)
            fi += 5

            # Calendar (6)
            X[hi, fi] = np.sin(2 * np.pi * h / 24)
            X[hi, fi+1] = np.cos(2 * np.pi * h / 24)
            X[hi, fi+2] = np.sin(2 * np.pi * ts.dayofweek / 7)
            X[hi, fi+3] = np.cos(2 * np.pi * ts.dayofweek / 7)
            X[hi, fi+4] = np.sin(2 * np.pi * ts.month / 12)
            X[hi, fi+5] = np.cos(2 * np.pi * ts.month / 12)
            fi += 6

            # Peak indicators (2)
            X[hi, fi] = 1.0 if 17 <= h <= 21 else 0.0
            X[hi, fi+1] = 1.0 if 9 <= h <= 15 else 0.0
            fi += 2

            # Load/RES forecast (2)
            X[hi, fi] = row.get("load_mw", 0) / 10000.0
            X[hi, fi+1] = row.get("res_total_mw", 0) / 5000.0
            fi += 2

            # Weather (4)
            for col, norm in [("temperature", 40.0), ("wind_100m", 25.0),
                              ("solar_radiation", 1000.0), ("cloud_cover", 100.0)]:
                X[hi, fi] = row.get(col, 0) / norm if col in df.columns else 0
                fi += 1

            # Commodities (2)
            X[hi, fi] = row.get("ttf_gas_price", 0) / 50.0
            X[hi, fi+1] = row.get("eua_carbon", 0) / 100.0
            fi += 2

            # Gas momentum (2)
            X[hi, fi] = row.get("ttf_gas_change_24h", 0) / 50.0
            X[hi, fi+1] = row.get("ttf_gas_change_7d", 0) / 50.0
            fi += 2

            # Neighbor D-0 (4)
            for col in ["dam_price_bg", "dam_price_ro", "dam_price_hu", "dam_price_it_south"]:
                X[hi, fi] = row.get(col, 0) / 200.0 if col in df.columns else 0
                fi += 1

            # Neighbor D+1 (4)
            for col in ["dam_price_bg_d1", "dam_price_ro_d1", "dam_price_hu_d1", "dam_price_it_south_d1"]:
                X[hi, fi] = row.get(col, 0) / 200.0 if col in df.columns else 0
                fi += 1

            # Serbia D+1 target hour (1)
            rs_col = f"serbia_d1_h{h:02d}"
            X[hi, fi] = row.get(rs_col, 0) / 200.0 if rs_col in df.columns else 0
            fi += 1

            # Serbia D+1 stats (4)
            rs_vals = []
            for hh in range(24):
                v = row.get(f"serbia_d1_h{hh:02d}", 0)
                if v > 1.0:
                    rs_vals.append(v)
            if rs_vals:
                ra = np.array(rs_vals)
                X[hi, fi] = np.mean(ra) / 200.0
                X[hi, fi+1] = np.max(ra) / 200.0
                X[hi, fi+2] = np.min(ra) / 200.0
                X[hi, fi+3] = (np.max(ra) - np.min(ra)) / 200.0
            fi += 4

            # RS-GR spread (1)
            rs_d1_mean = X[hi, fi-4] * 200.0
            gr_mean = X[hi, 4]  # dam_mean_24h
            X[hi, fi] = (rs_d1_mean / 200.0 - gr_mean) / 100.0 if rs_d1_mean > 0 else 0
            fi += 1

            # Forecast errors (2)
            X[hi, fi] = row.get("load_forecast_error_lag", 0) / 500.0
            X[hi, fi+1] = row.get("res_forecast_error_lag", 0) / 500.0
            fi += 2

            # Target hour encoding (2)
            X[hi, fi] = np.sin(2 * np.pi * h / 24)
            X[hi, fi+1] = np.cos(2 * np.pi * h / 24)
            fi += 2

            # === IDA1 NEW FEATURES (7) ===

            # GR DAM actual for the DELIVERY day (4)
            # I find the delivery day start: from decision point, go to next midnight
            # IDA1 at D-1 15:00 → delivery day D starts at decision_idx + 9h
            # IDA2 at D-1 22:00 → delivery day D starts at decision_idx + 2h
            # IDA3 at D+0 10:00 → delivery day is TODAY, started 10h ago
            hours_to_midnight = (24 - ts.hour) % 24
            delivery_day_start = decision_idx + hours_to_midnight * sph
            if self.ida_number == 3:
                # I am ON the delivery day — today started 10h ago
                delivery_day_start = decision_idx - ts.hour * sph

            if prices is not None and delivery_day_start + 24 * sph <= len(prices) and delivery_day_start >= 0:
                day_prices = prices[delivery_day_start:delivery_day_start + 24 * sph]
                hourly = [np.mean(day_prices[hh*sph:(hh+1)*sph]) for hh in range(24)]
                hourly = np.array(hourly)

                X[hi, fi] = hourly[h] / 200.0  # dam_actual_target_hour
                X[hi, fi+1] = np.mean(hourly) / 200.0  # dam_actual_mean
                X[hi, fi+2] = (np.max(hourly) - np.min(hourly)) / 200.0  # dam_actual_spread
                X[hi, fi+3] = np.max(hourly) / 200.0  # dam_actual_peak_price
            fi += 4

            # DAM schedule context (3)
            if dam_schedule is not None and len(dam_schedule) > h * sph:
                X[hi, fi] = dam_schedule[h * sph] / 30.0  # dam_sched_this_hour
                buy_mwh = np.sum(np.minimum(dam_schedule, 0)) * 0.25
                sell_mwh = np.sum(np.maximum(dam_schedule, 0)) * 0.25
                X[hi, fi+1] = abs(buy_mwh) / 100.0  # dam_sched_total_buy_mwh
                X[hi, fi+2] = sell_mwh / 100.0  # dam_sched_total_sell_mwh
            fi += 3

            # Weather forecast deviation (4) — IDA1 key signal
            # At 15:00, we have 3h of actual weather since DAM decision at 12:00
            # I use the deviation columns from the dataset
            if "wind_forecast_deviation" in df.columns:
                X[hi, fi] = row.get("wind_forecast_deviation", 0) / 500.0
            fi += 1
            if "solar_forecast_deviation" in df.columns:
                X[hi, fi] = row.get("solar_forecast_deviation", 0) / 500.0
            fi += 1
            if "load_forecast_deviation" in df.columns:
                X[hi, fi] = row.get("load_forecast_deviation", 0) / 1000.0
            fi += 1
            if "res_surprise" in df.columns:
                X[hi, fi] = row.get("res_surprise", 0)
            fi += 1

            # === IDA2 NEW FEATURES (7) — only for IDA2, IDA3 ===
            if self.ida_number >= 2:
                # ISP1 clearing (3)
                X[hi, fi] = row.get("isp1_price_up", 0) / 200.0
                X[hi, fi+1] = row.get("isp1_price_down", 0) / 200.0
                isp1_up = row.get("isp1_price_up", 100)
                isp1_dn = row.get("isp1_price_down", 100)
                X[hi, fi+2] = (isp1_up - isp1_dn) / 200.0
                fi += 3

                # Weather deviation UPDATED at 22:00 (5)
                # I use the pre-computed deviation columns for cleaner signal
                if "wind_deviation_24h" in df.columns:
                    X[hi, fi] = row.get("wind_deviation_24h", 0) / 1000.0
                fi += 1
                if "solar_deviation_24h" in df.columns:
                    X[hi, fi] = row.get("solar_deviation_24h", 0) / 500.0
                fi += 1
                if "load_deviation_24h" in df.columns:
                    X[hi, fi] = row.get("load_deviation_24h", 0) / 1000.0
                fi += 1
                if "wind_trend_3h" in df.columns:
                    X[hi, fi] = row.get("wind_trend_3h", 0) / 500.0
                fi += 1
                if "res_surprise" in df.columns:
                    X[hi, fi] = row.get("res_surprise", 0)
                fi += 1

                # Imbalance signal (2)
                if "net_imbalance_mw" in df.columns:
                    recent_imb = df["net_imbalance_mw"].iloc[max(0, decision_idx-w6):decision_idx]
                    X[hi, fi] = recent_imb.mean() / 500.0 if len(recent_imb) > 0 else 0
                fi += 1
                X[hi, fi] = row.get("imbalance_price", 0) / 200.0
                fi += 1

            # === IDA3 NEW FEATURES (9) — only for IDA3 ===
            if self.ida_number >= 3:
                # Morning actual prices (3) — TODAY 00:00-10:00 (BEFORE decision point)
                # I am at D+0 10:00, so morning = decision_idx - 10h to decision_idx
                if prices is not None:
                    morning_start = max(0, decision_idx - 10 * sph)
                    morning_end = decision_idx
                    if morning_end > morning_start:
                        morning = prices[morning_start:morning_end]
                        X[hi, fi] = np.mean(morning) / 200.0
                        X[hi, fi+1] = np.max(morning) / 200.0
                        # I compare morning actual vs DAM published price for same hours
                        dam_morning_mean = X[hi, fi-16] if X[hi, fi-16] > 0 else np.mean(morning) / 200.0
                        X[hi, fi+2] = (np.mean(morning) / 200.0 - dam_morning_mean)
                fi += 3

                # ISP2 clearing (2)
                X[hi, fi] = row.get("isp2_price_up", 0) / 200.0
                X[hi, fi+1] = row.get("isp2_price_down", 0) / 200.0
                fi += 2

                # Morning wind/solar actual (2) — TODAY, before decision point
                morning_start = max(0, decision_idx - 10 * sph)
                if "wind_onshore" in df.columns:
                    morning_wind = df["wind_onshore"].iloc[morning_start:decision_idx]
                    X[hi, fi] = morning_wind.mean() / 2000.0 if len(morning_wind) > 0 else 0
                fi += 1
                if "solar" in df.columns:
                    morning_solar = df["solar"].iloc[morning_start:decision_idx]
                    X[hi, fi] = morning_solar.mean() / 2000.0 if len(morning_solar) > 0 else 0
                fi += 1

                # Afternoon weather proxy (3) — I use morning actuals to predict afternoon
                # Solar: morning solar × 1.5 (afternoon typically 50% higher in summer)
                # At 10:00, the weather pattern is established for the day
                morning_solar_val = X[hi, fi-1] * 2000.0  # I denormalize morning_solar
                month = ts.month
                # I estimate afternoon solar ratio based on season
                # Summer (5-9): afternoon solar ~ 1.5× morning. Winter: ~ 0.8×
                solar_ratio = 1.5 if 5 <= month <= 9 else 0.8
                X[hi, fi] = morning_solar_val * solar_ratio / 2000.0
                fi += 1

                # Wind: morning wind is a good proxy (persistent, corr 0.83)
                morning_wind_val = X[hi, fi-3] * 2000.0  # I denormalize morning_wind
                X[hi, fi] = morning_wind_val / 2000.0  # I pass through — wind is persistent
                fi += 1

                # RES impact: estimated total afternoon RES / mean demand
                # High RES → low afternoon prices, Low RES → high prices
                est_afternoon_res = morning_solar_val * solar_ratio + morning_wind_val
                load_est = row.get("load_mw", 5000)
                X[hi, fi] = est_afternoon_res / max(load_est, 1000)  # RES penetration ratio
                fi += 1

                # Current SoC (1)
                X[hi, fi] = current_soc
                fi += 1

                # Overnight imbalance mean (1) — TODAY 00:00-06:00
                overnight_start = max(0, decision_idx - 10 * sph)  # same as morning_start
                overnight_end = max(0, decision_idx - 4 * sph)     # 00:00-06:00 roughly
                if "net_imbalance_mw" in df.columns and overnight_end > overnight_start:
                    overnight = df["net_imbalance_mw"].iloc[overnight_start:overnight_end]
                    X[hi, fi] = overnight.mean() / 500.0 if len(overnight) > 0 else 0
                fi += 1

        hours = np.arange(hour_offset, hour_offset + n_hours)
        return X, hours

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
        dam_forecaster=None,
    ) -> "IDAScheduleForecaster":
        """I train the IDA forecaster.

        Args:
            df: Full dataset
            train_end_idx: End of training period
            val_end_idx: End of validation period
            dam_forecaster: Trained DAM forecaster (to generate DAM schedules for features)
        """
        if lgb is None:
            raise ImportError("LightGBM required")

        sph = self._detect_sph(df)
        day_steps = 24 * sph
        pcol = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        n_hours = self.config["n_target_hours"]
        hour_offset = 12 if self.ida_number == 3 else 0
        label = self.config["label"]

        logger.info(f"[{label}] Building training data...")

        from PredictFuturePrices.forecasters.schedule_optimizer import ScheduleOptimizer
        opt = ScheduleOptimizer(max_power_mw=30, capacity_mwh=146, efficiency=0.94)

        X_all, y_all = [], []
        min_history = 7 * day_steps

        for day_start in range(min_history, train_end_idx, day_steps):
            # I simulate the gate closure time
            gate_hour = self.config["gate_hour"]
            decision_idx = day_start + gate_hour * sph

            if decision_idx >= train_end_idx:
                break

            # I generate DAM schedule (using DAM forecaster or simple heuristic)
            dam_schedule = None
            if dam_forecaster is not None:
                predicted_prices = dam_forecaster.predict(df, day_start)
                if predicted_prices is not None:
                    result = opt.optimize(predicted_prices, current_soc=0.5, n_slots=96)
                    dam_schedule = result.schedule

            if dam_schedule is None:
                dam_schedule = np.zeros(96)

            result = self.build_features_for_day(df, decision_idx, dam_schedule)
            if result is None:
                continue

            X_day, hours = result

            # I extract targets: actual prices for D+1
            targets = np.full(n_hours, np.nan)
            next_day_start = day_start + day_steps
            for hi, h in enumerate(range(hour_offset, hour_offset + n_hours)):
                target_idx = next_day_start + h * sph
                if target_idx < len(df):
                    targets[hi] = df[pcol].iloc[target_idx]

            valid = ~np.isnan(targets)
            if valid.sum() > 0:
                X_all.append(X_day[valid])
                y_all.append(targets[valid])

        if not X_all:
            logger.warning(f"[{label}] No valid training data!")
            return self

        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)

        # I build validation set
        val_X, val_y = [], []
        for day_start in range(train_end_idx, val_end_idx, day_steps):
            gate_hour = self.config["gate_hour"]
            decision_idx = day_start + gate_hour * sph
            if decision_idx >= val_end_idx:
                break

            dam_schedule = np.zeros(96)
            if dam_forecaster is not None:
                predicted = dam_forecaster.predict(df, day_start)
                if predicted is not None:
                    r = opt.optimize(predicted, current_soc=0.5, n_slots=96)
                    dam_schedule = r.schedule

            result = self.build_features_for_day(df, decision_idx, dam_schedule)
            if result is None:
                continue

            X_day, hours = result
            targets = np.full(n_hours, np.nan)
            next_day_start = day_start + day_steps
            for hi, h in enumerate(range(hour_offset, hour_offset + n_hours)):
                target_idx = next_day_start + h * sph
                if target_idx < len(df):
                    targets[hi] = df[pcol].iloc[target_idx]
            valid = ~np.isnan(targets)
            if valid.sum() > 0:
                val_X.append(X_day[valid])
                val_y.append(targets[valid])

        X_val = np.vstack(val_X) if val_X else X_train[-500:]
        y_val = np.concatenate(val_y) if val_y else y_train[-500:]

        logger.info(f"[{label}] Training: {len(X_train)} samples, "
                     f"Validation: {len(X_val)}, Features: {X_train.shape[1]}")

        train_set = lgb.Dataset(X_train, label=y_train,
                                feature_name=self.feature_names)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "huber",
            "metric": "mae",
            "n_estimators": 800,
            "max_depth": 8,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
        }

        callbacks = [lgb.log_evaluation(period=200)]
        self.model = lgb.train(
            params, train_set,
            valid_sets=[val_set],
            num_boost_round=params.pop("n_estimators"),
            callbacks=callbacks,
        )

        val_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(val_pred - y_val))
        rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        self.train_metrics = {"val_mae": mae, "val_rmse": rmse, "n_train": len(X_train)}
        self.is_fitted = True

        logger.info(f"[{label}] Trained: MAE={mae:.2f}, RMSE={rmse:.2f}")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        decision_idx: int,
        dam_schedule: Optional[np.ndarray] = None,
        current_soc: float = 0.5,
    ) -> Optional[np.ndarray]:
        """I predict hourly prices at this IDA gate closure.

        Returns:
            Array of predicted prices (24 for IDA1/2, 12 for IDA3)
        """
        if not self.is_fitted or self.model is None:
            return None

        result = self.build_features_for_day(
            df, decision_idx, dam_schedule, current_soc)
        if result is None:
            return None

        X, hours = result
        prices = self.model.predict(X)
        prices = np.maximum(prices, 0)
        return prices

    def save(self, path: str):
        """I save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "ida_number": self.ida_number,
                "metrics": self.train_metrics,
                "feature_names": self.feature_names,
            }, f)

    def load(self, path: str) -> "IDAScheduleForecaster":
        """I load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.train_metrics = data.get("metrics", {})
        self.is_fitted = True
        return self

    def _detect_sph(self, df: pd.DataFrame) -> int:
        """I detect steps-per-hour."""
        if len(df) < 10:
            return 1
        diffs = pd.Series(df.index[:100]).diff().dropna()
        median_min = diffs.dt.total_seconds().median() / 60
        if median_min <= 20: return 4
        if median_min <= 45: return 2
        return 1
