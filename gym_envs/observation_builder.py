"""
Observation Builder for Unified Battery Environment

I build the observation vector (70-112 features) from env state and market data.
This was previously the _build_observation() method (529 lines) in BatteryEnvUnified.

Feature Groups:
1. Battery State (4): soc, max_discharge, max_charge, cycles_remaining
2. Market Prices (13): dam, imb_lag, sys_dir, imb_mag, volume, mfrr_up/down/spread, spreads, hvr
3. Time Encoding (4): hour_sin/cos, dow_sin/cos
4. DAM Lookahead (13): current + 12h commitments
5. Price Lookahead (12): 12h price forecasts
6. aFRR State (8): cap_prices, activation, commitment, selection, signal, hours_since
7. Risk/Imbalance (8): dam_reserve, shortfall, momentum, worthiness, soc_eod, net_energy, position, vol_regime
8. Market Phase (4): ida3_correction, system_stress, res_penetration, mfrr_direction
9. Timing Signals (4): price_vs_typical, is_peak, is_solar, hours_to_max
10. IntraDay Forecast + RES (9, enable_forecast)
10b. Market Forecasts (20, enable_market_forecast)
11-12. Full Market (13, enable_full_market)
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gym_envs.battery_env_unified import BatteryEnvUnified


class ObservationBuilder:
    """I build observation vectors from env state for the RL agent.

    I hold a reference to the env and read its state when build() is called.
    This extracts the 529-line _build_observation() method for maintainability.
    """

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def build(self) -> np.ndarray:
        """I build the observation vector (70-112 features depending on flags)."""
        env = self._env
        row = env.df.iloc[env.current_step]
        ts = env.df.index[env.current_step]

        features = []

        # I build each feature group
        self._add_battery_state(features, env, row)
        self._add_market_prices(features, env, row)
        self._add_time_encoding(features, ts)
        self._add_dam_lookahead(features, env, ts)
        self._add_price_lookahead(features, env, ts, row)
        self._add_afrr_state(features, env, row)
        self._add_risk_imbalance(features, env, row)
        self._add_market_phase(features, env, row, ts)
        self._add_timing_signals(features, env, row, ts)

        if env.enable_forecast:
            self._add_forecast_features(features, env, row)
        if env.enable_market_forecast:
            self._add_market_forecast_features(features, env, row, ts)
        if env.enable_full_market:
            self._add_full_market_features(features, env, row, ts)

        # I validate and clean
        obs = np.array(features, dtype=np.float32)
        expected_features = 72  # I updated from 70: +2 DAM acceptance features in Group 7
        if env.enable_forecast:
            expected_features += 9
        if env.enable_market_forecast:
            expected_features += 20
        if env.enable_full_market:
            expected_features += 13
        assert len(obs) == expected_features, f"Expected {expected_features} features, got {len(obs)}"

        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _add_battery_state(self, features, env, row):
        """Group 1: Battery State (4 features)."""
        features.append(env.soc)

        dam_commitment = np.clip(env._get_dam_commitment(env.current_step),
                                  -env.max_power_mw, env.max_power_mw)
        remaining_capacity = max(0, env.max_power_mw - abs(dam_commitment) - env.afrr_commitment_mw)

        available_discharge = (env.soc - env.min_soc) * env.capacity_mwh
        available_charge = (env.max_soc - env.soc) * env.capacity_mwh

        max_discharge = min(remaining_capacity,
                           available_discharge * env.eff_sqrt / env.time_step_hours)
        max_charge = min(remaining_capacity,
                        available_charge / env.eff_sqrt / env.time_step_hours)

        features.append(max_discharge / env.max_power_mw)
        features.append(max_charge / env.max_power_mw)

        cycles_remaining = max(0, env.max_daily_cycles - env.daily_cycles) / env.max_daily_cycles
        features.append(cycles_remaining)

    def _add_market_prices(self, features, env, row):
        """Group 2: Market Prices (13 features)."""
        dam_price = row.get('price', 100.0)
        features.append(dam_price / 100.0)

        imb_price_lag = row.get('imbalance_price_lag_1h', dam_price)
        net_imb_lag = row.get('net_imbalance_mw_lag_1h',
                              row.get('net_imbalance_mw', 0.0))
        features.append(imb_price_lag / 100.0)
        features.append(np.sign(net_imb_lag))
        features.append(min(abs(net_imb_lag) / 500.0, 1.0))
        features.append(row.get('intraday_volume_lag_1h', 0.5))

        mfrr_up = row.get('mfrr_price_up_lag_1h', 0.0)
        mfrr_down = row.get('mfrr_price_down_lag_1h', 0.0)
        features.append(mfrr_up / 100.0)
        features.append(mfrr_down / 100.0)
        features.append((mfrr_up - mfrr_down) / 100.0)

        features.append((imb_price_lag - dam_price) / 100.0)
        features.append((mfrr_up - dam_price) / 100.0)
        features.append((mfrr_up - imb_price_lag) / 100.0)

        afrr_up_lag = row.get('afrr_up_lag_1h', row.get('afrr_up', 80.0))
        afrr_down_lag = row.get('afrr_down_lag_1h', row.get('afrr_down', 80.0))
        features.append((mfrr_up - afrr_up_lag) / 100.0)
        features.append((mfrr_down - afrr_down_lag) / 100.0)

    def _add_time_encoding(self, features, ts):
        """Group 3: Time Encoding (4 features)."""
        hour = ts.hour
        minute = ts.minute
        fractional_hour = hour + minute / 60.0
        dow = ts.dayofweek

        features.append(np.sin(2 * np.pi * fractional_hour / 24))
        features.append(np.cos(2 * np.pi * fractional_hour / 24))
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

    def _add_dam_lookahead(self, features, env, ts):
        """Group 4: DAM Lookahead (13 features)."""
        dam_commitment = np.clip(env._get_dam_commitment(env.current_step),
                                  -env.max_power_mw, env.max_power_mw)
        features.append(dam_commitment / env.max_power_mw)

        current_day = ts.date()
        for i in range(1, 13):
            target = min(env.current_step + i * env._sph, len(env.df) - 1)
            target_ts = env.df.index[target]
            if target_ts.date() == current_day:
                future_dam = np.clip(env._get_dam_commitment(target),
                                     -env.max_power_mw, env.max_power_mw)
            else:
                future_dam = 0.0
            features.append(future_dam / env.max_power_mw)

    def _add_price_lookahead(self, features, env, ts, row):
        """Group 5: Price Lookahead (12 features)."""
        dam_price = row.get('price', 100.0)
        current_hour = ts.hour
        current_day = ts.date()
        from datetime import timedelta

        if current_hour >= 14:
            last_known_day = (ts + timedelta(days=1)).date()
        else:
            last_known_day = current_day

        for i in range(1, 13):
            target = min(env.current_step + i * env._sph, len(env.df) - 1)
            target_ts = env.df.index[target]
            target_day = target_ts.date()

            if target_day <= last_known_day:
                future_price = env.df.iloc[target].get('price', dam_price)
            else:
                future_price = dam_price

            features.append(future_price / 100.0)

    def _add_afrr_state(self, features, env, row):
        """Group 6: aFRR State (8 features)."""
        cap_up = row.get('afrr_cap_up_price', 20.0)
        cap_down = row.get('afrr_cap_down_price', 30.0)
        features.append(cap_up / 50.0)
        features.append(cap_down / 50.0)

        imbalance = row.get('net_imbalance_mw', 0)
        features.append(np.clip(imbalance / 1000.0, -1, 1))

        features.append(env.afrr_commitment_mw / env.max_power_mw)

        cap_percentile = row.get('afrr_cap_percentile', 0.5)
        features.append(cap_percentile)

        features.append(1.0 if env.steps_since_afrr_activation < 4 * env._sph else 0.0)

        features.append(min(env.steps_since_afrr_activation, 24 * env._sph) / (24 * env._sph))

        features.append(1.0 if env.is_selected_for_afrr else 0.0)

    def _add_risk_imbalance(self, features, env, row):
        """Group 7: Risk/Imbalance (10 features — was 8, added 2 DAM acceptance features)."""
        dam_min_soc, dam_max_soc = env._calculate_dam_soc_reserve()
        dam_discharge_reserve = (dam_min_soc - env.min_soc) / (env.max_soc - env.min_soc)
        features.append(np.clip(dam_discharge_reserve, 0, 1))

        shortfall_risk = env._calculate_shortfall_risk()
        features.append(shortfall_risk)

        momentum = row.get('price_momentum', 0.0)
        features.append(np.clip(momentum, -1, 1))

        price_std = row.get('price_std_24h', 20.0)
        worthiness = min(price_std / 50.0, 1.0)
        features.append(worthiness)

        predicted_soc_eod, net_energy_mwh = env._predict_dam_soc()
        features.append(np.clip(predicted_soc_eod, 0.0, 1.0))

        dam_net_energy_ratio = np.clip(net_energy_mwh / env.capacity_mwh, -1.0, 1.0)
        features.append(dam_net_energy_ratio)

        total_sold = (env.dam_mwh_sold + env.intraday_mwh_sold +
                      env.afrr_mwh_sold + env.mfrr_mwh_sold +
                      env.xbid_mwh_sold + env.free_bid_mwh_sold +
                      env.ida_mwh_sold)
        total_bought = (env.dam_mwh_bought + env.intraday_mwh_bought +
                        env.afrr_mwh_bought + env.mfrr_mwh_bought +
                        env.xbid_mwh_bought + env.free_bid_mwh_bought +
                        env.ida_mwh_bought)
        net_energy_position = np.clip(
            (total_sold - total_bought) / env.capacity_mwh, -1.0, 1.0
        )
        features.append(net_energy_position)

        vol_24h = row.get('price_std_24h', 20.0)
        vol_6h = row.get('price_std_6h', vol_24h)
        if vol_24h > 0.1:
            vol_regime = np.clip(vol_6h / vol_24h, 0.0, 3.0) / 3.0
        else:
            vol_regime = 0.5
        features.append(vol_regime)

        # I add DAM bidding acceptance features (2 new features)
        # These tell the agent how well DAM bids are being accepted today
        dam_accept_rate = getattr(env, 'dam_acceptance_rate', 1.0)
        features.append(np.clip(dam_accept_rate, 0.0, 1.0))

        # I compute remaining rejected volume for rest of day (normalized)
        dam_rejected_remaining = 0.0
        if hasattr(env, 'dam_rejected_volume') and env.dam_rejected_volume is not None:
            day_indices = env._get_day_indices_for_step(env.current_step)
            if len(day_indices) > 0:
                day_offset = env.current_step - day_indices[0]
                remaining = env.dam_rejected_volume[day_offset:]
                dam_rejected_remaining = abs(remaining).sum() / max(env.max_power_mw, 1.0)
        features.append(np.clip(dam_rejected_remaining, 0.0, 5.0) / 5.0)

    def _add_market_phase(self, features, env, row, ts):
        """Group 8: Market Phase & System State (4 features)."""
        hour = ts.hour
        if env.enable_full_market:
            features.append(env._get_ida_phase(row))
        else:
            ida3_applies = 1.0 if hour >= 12 else 0.0
            features.append(ida3_applies)

        imbalance_mag = row.get('net_imbalance_mw', 0.0)
        system_stress = min(abs(imbalance_mag) / 300.0, 1.0)
        features.append(system_stress)

        res_total = row.get('res_total_mw', 0.0)
        load_total = row.get('load_mw', 1.0)
        res_penetration = np.clip(res_total / max(load_total, 1.0), 0.0, 1.5) / 1.5
        features.append(res_penetration)

        if 'mfrr_activated_up_mwh' in row.index if hasattr(row, 'index') else 'mfrr_activated_up_mwh' in row:
            mfrr_up_active = 1.0 if row.get('mfrr_activated_up_mwh', 0) > 0 else 0.0
            mfrr_down_active = 1.0 if row.get('mfrr_activated_down_mwh', 0) > 0 else 0.0
            mfrr_direction = mfrr_up_active - mfrr_down_active
        else:
            imbalance = row.get('net_imbalance_mw', 0.0)
            if imbalance < -env.mfrr_imbalance_threshold:
                mfrr_direction = 1.0
            elif imbalance > env.mfrr_imbalance_threshold:
                mfrr_direction = -1.0
            else:
                mfrr_direction = 0.0
        features.append(mfrr_direction)

    def _add_timing_signals(self, features, env, row, ts):
        """Group 9: Timing Signals (4 features)."""
        dam_price = row.get('price', 100.0)
        hour = ts.hour

        mean_24h = row.get('price_mean_24h', 100.0)
        if mean_24h > 1:
            price_vs_typical = (dam_price - mean_24h) / mean_24h
        else:
            price_vs_typical = 0.0
        features.append(np.clip(price_vs_typical, -1, 1))

        is_peak = 1.0 if 17 <= hour <= 21 else 0.0
        features.append(is_peak)

        is_solar = 1.0 if 9 <= hour <= 15 else 0.0
        features.append(is_solar)

        hours_to_peak = min(abs(19 - hour), abs(19 - hour + 24)) / 12.0
        features.append(hours_to_peak)

    def _add_forecast_features(self, features, env, row):
        """Group 10: IntraDay Forecast + RES Fundamentals (9 features)."""
        dam_price = row.get('price', 100.0)

        if env.price_forecaster is not None:
            all_fc = env.price_forecaster.predict(
                env.df, env.current_step, add_noise=env.forecast_noise
            )
            fc_1h = all_fc[0]
            fc_2h = all_fc[1]
            fc_4h = all_fc[3] if len(all_fc) > 3 else all_fc[-1]
            fc_8h = all_fc[7] if len(all_fc) > 7 else all_fc[-1]
        else:
            fc_1h = fc_2h = fc_4h = fc_8h = dam_price
            if env.forecast_noise:
                for i, h in enumerate([1, 2, 4, 8]):
                    err = 0.08 + 0.12 * (h - 1) / 11
                    val = max(0.1, dam_price + np.random.normal(0, abs(dam_price) * err))
                    if i == 0: fc_1h = val
                    elif i == 1: fc_2h = val
                    elif i == 2: fc_4h = val
                    elif i == 3: fc_8h = val

        features.append(fc_1h / 100.0)
        features.append(fc_2h / 100.0)
        features.append(fc_4h / 100.0)
        features.append(fc_8h / 100.0)
        features.append((fc_1h - dam_price) / 100.0)
        fc_list = [fc_1h, fc_2h, fc_4h, fc_8h]
        features.append((max(fc_list) - min(fc_list)) / 100.0)
        features.append(row.get('solar', 0.0) / 3000.0)
        features.append(row.get('wind_onshore', 0.0) / 3000.0)
        res = row.get('res_total_mw', row.get('solar', 0) + row.get('wind_onshore', 0))
        res_mean = env._res_mean_24h[env.current_step] if env._res_mean_24h is not None else res
        features.append((res - res_mean) / 1000.0)

    def _add_market_forecast_features(self, features, env, row, ts):
        """Group 10b: Market Forecasts (20 features)."""
        dam_price = row.get('price', 100.0)

        if 'dam_forecast_next_4h_mean' in env.df.columns:
            self._add_precomputed_market_forecast(features, env, row, ts, dam_price)
        elif env.market_forecaster is not None:
            self._add_live_market_forecast(features, env, row, dam_price)
        else:
            # I use zeros as placeholder (20 features)
            features.extend([dam_price / 100.0] * 3 + [0.0] * 2 +
                            [dam_price / 100.0] * 2 + [0.0] * 8 +
                            [dam_price / 100.0] * 2 + [0.0] * 3)

    def _add_precomputed_market_forecast(self, features, env, row, ts, dam_price):
        """I read pre-computed forecast columns from the DataFrame."""
        features.append(row.get('dam_forecast_next_4h_mean', dam_price) / 100.0)
        features.append(row.get('dam_forecast_next_4h_max', dam_price) / 100.0)
        features.append(row.get('dam_forecast_next_4h_min', dam_price) / 100.0)
        fc_spread = row.get('dam_forecast_spread', 0.0)
        features.append(fc_spread / 100.0)
        features.append(row.get('dam_forecast_vs_current', 0.0) / 100.0)
        features.append(row.get('id_forecast_1h', dam_price) / 100.0)
        features.append(row.get('id_forecast_4h', dam_price) / 100.0)
        id_fc_1h = row.get('id_forecast_1h', dam_price)
        dam_fc_mean = row.get('dam_forecast_next_4h_mean', dam_price)
        features.append((id_fc_1h - dam_fc_mean) / 100.0)
        features.append(row.get('imbalance_forecast_dir', 0.0))
        features.append(row.get('imbalance_forecast_mag', 0.0) / 500.0)
        features.append(row.get('mfrr_opportunity_score', 0.0))
        serbia = row.get('serbia_dam_price', np.nan)
        if not np.isnan(serbia):
            features.append((serbia - dam_price) / 100.0)
        else:
            features.append(0.0)
        features.append(row.get('forecast_confidence', 0.5))

        # I compute dynamic hours to peak/trough respecting DAM publication window
        from datetime import timedelta
        current_hour = ts.hour
        current_day = ts.date()
        if current_hour >= 14:
            last_known_day = (ts + timedelta(days=1)).date()
        else:
            last_known_day = current_day

        known_prices = []
        for i in range(96):
            target = min(env.current_step + i, len(env.df) - 1)
            target_day = env.df.index[target].date()
            if target_day <= last_known_day:
                known_prices.append(env.df.iloc[target].get('price', dam_price))
            else:
                known_prices.append(dam_price)

        if len(known_prices) > 4:
            hours_to_peak = np.argmax(known_prices) * env.time_step_hours
            hours_to_trough = np.argmin(known_prices) * env.time_step_hours
        else:
            hours_to_peak, hours_to_trough = 12.0, 6.0
        features.append(hours_to_peak / 24.0)
        features.append(hours_to_trough / 24.0)

        # I add ISP features (5 features) — ISP1 only
        isp1_up = row.get('isp1_price_up', dam_price)
        isp1_down = row.get('isp1_price_down', dam_price)
        features.append(isp1_up / 100.0)
        features.append(isp1_down / 100.0)
        features.append((isp1_up - dam_price) / 100.0)
        features.append((isp1_up - isp1_down) / 100.0)
        features.append(row.get('isp_cascade_spread_up_lag_24h', 0.0) / 100.0)

    def _add_live_market_forecast(self, features, env, row, dam_price):
        """I call the live forecaster (slower but works without pre-computed columns)."""
        try:
            fc = env.market_forecaster.predict_all(env.df, env.current_step)
            dam_4h = fc['dam_forecast_24h'][:4]
            features.append(np.mean(dam_4h) / 100.0)
            features.append(np.max(dam_4h) / 100.0)
            features.append(np.min(dam_4h) / 100.0)
            features.append((np.max(dam_4h) - np.min(dam_4h)) / 100.0)
            features.append((np.mean(dam_4h) - dam_price) / 100.0)
            id_fc = fc['id_forecast']
            features.append(id_fc.get(1, dam_price) / 100.0)
            features.append(id_fc.get(4, dam_price) / 100.0)
            features.append((id_fc.get(1, dam_price) - np.mean(dam_4h)) / 100.0)
            imb = fc['imbalance']
            features.append(imb['direction'])
            features.append(imb['magnitude'] / 500.0)
            features.append(imb['opportunity_score'])
            serbia = row.get('serbia_dam_price', np.nan)
            features.append((serbia - dam_price) / 100.0 if not np.isnan(serbia) else 0.0)
            avg_std = np.mean(fc['dam_forecast_std'][:4])
            features.append(1.0 / (1.0 + avg_std / 20.0))
            dam_24h = fc['dam_forecast_24h']
            features.append(float(np.argmax(dam_24h) + 1) / 24.0)
            features.append(float(np.argmin(dam_24h) + 1) / 24.0)

            isp1_up = row.get('isp1_price_up', dam_price)
            isp1_down = row.get('isp1_price_down', dam_price)
            features.append(isp1_up / 100.0)
            features.append(isp1_down / 100.0)
            features.append((isp1_up - dam_price) / 100.0)
            features.append((isp1_up - isp1_down) / 100.0)
            features.append(row.get('isp_cascade_spread_up_lag_24h', 0.0) / 100.0)
        except Exception:
            features.extend([dam_price / 100.0] * 2 + [dam_price / 100.0] +
                            [0.0] * 2 + [dam_price / 100.0] * 2 + [0.0] * 8 +
                            [dam_price / 100.0] * 2 + [0.0] * 3)

    def _add_full_market_features(self, features, env, row, ts):
        """Groups 11-12: Full Market Features (13 features)."""
        dam_price = row.get('price', 100.0)
        dam_commitment = np.clip(env._get_dam_commitment(env.current_step),
                                  -env.max_power_mw, env.max_power_mw)
        imbalance_fm = row.get('net_imbalance_mw', 0.0)

        # IDA State (8 features)
        features.append(env._get_ida_phase(row))
        features.append(env._get_hours_to_next_ida_gate() / 24.0)

        ida_net = env._get_ida_commitment(env.current_step)
        features.append(np.clip(ida_net / env.max_power_mw, -1.0, 1.0))

        total_net = dam_commitment + ida_net
        features.append(np.clip(total_net / env.max_power_mw, -1.0, 1.0))

        ida_fc_spread = env._get_ida_forecast_vs_dam_spread(row)
        features.append(np.clip(ida_fc_spread / 50.0, -1.0, 1.0))

        isp1_up_obs = row.get('isp1_price_up', dam_price)
        isp1_down_obs = row.get('isp1_price_down', dam_price)
        isp1_bal_spread = max(0, isp1_up_obs - isp1_down_obs)
        features.append(min(isp1_bal_spread / 50.0, 1.0))

        soc_feasible = 1.0 if env._check_soc_trajectory_feasible() else 0.0
        features.append(soc_feasible)

        correction_needed = env._get_ida_correction_signal()
        features.append(np.clip(correction_needed, -1.0, 1.0))

        isp1_mid_obs = (isp1_up_obs + isp1_down_obs) / 2.0
        isp1_dam_spread = (isp1_mid_obs - dam_price) / 50.0
        features.append(np.clip(isp1_dam_spread, -1.0, 1.0))

        # Free Bid State (3 features)
        features.append(1.0 if abs(imbalance_fm) > env.mfrr_imbalance_threshold else 0.0)
        features.append(row.get('free_bid_activation_base', 0.3))
        features.append(row.get('free_bid_reference_price', 100.0) / 100.0)

        # ISP Cascade Quality (1 feature)
        features.append(row.get('isp_cascade_spread_up_lag_24h', 0.0) / 100.0)
