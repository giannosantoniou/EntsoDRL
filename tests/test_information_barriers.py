"""
Tests for Information Barriers in Decomposed Market Models

I verify that each model's observation builder only accesses data available
at its decision time, preventing information leakage:

- DAM: Only sees price FORECASTS (lag_24h), NOT actual clearing prices
- aFRR: Sees DAM results but NOT future aFRR activations or IntraDay/mFRR prices
- IntraDay: Sees DAM+aFRR commitments but NOT future XBID prices or mFRR
- mFRR: Uses LAGGED imbalance, NOT real-time imbalance data
"""

import pytest
import numpy as np
import pandas as pd

from gym_envs.battery_env_dam import BatteryEnvDAM
from gym_envs.battery_env_afrr import BatteryEnvAFRR
from gym_envs.battery_env_intraday import BatteryEnvIntraDay
from gym_envs.battery_env_mfrr import BatteryEnvMFRR
from gym_envs.shared_battery_state import SharedBatteryState


@pytest.fixture
def barrier_test_df():
    """
    I create a DataFrame with EXTREME future prices to detect information leakage.

    If a model's observation contains future information, the extreme values
    will appear in the observation and fail the barrier test.
    """
    n_rows = 7 * 24 * 4 + 48 * 4  # 7 days + warmup
    dates = pd.date_range(
        start='2025-01-01', periods=n_rows, freq='15min', tz='Europe/Athens'
    )
    np.random.seed(42)

    # I create normal prices for the first 3 days
    normal_prices = 80.0 + np.random.normal(0, 5, n_rows)

    # I set EXTREME future prices (days 5-7) that would be obvious if leaked
    extreme_start = 5 * 24 * 4  # day 5 start
    extreme_prices = normal_prices.copy()
    extreme_prices[extreme_start:] = 9999.0  # extreme future value

    df = pd.DataFrame({
        'price': extreme_prices,
        'dam_commitment': np.zeros(n_rows),
        'afrr_up': 80.0 * np.ones(n_rows),
        'afrr_down': 80.0 * np.ones(n_rows),
        'afrr_cap_up_price': 15.0 * np.ones(n_rows),
        'afrr_cap_down_price': 12.0 * np.ones(n_rows),
        'intraday_bid': 78.0 * np.ones(n_rows),
        'intraday_ask': 82.0 * np.ones(n_rows),
        'intraday_spread': 4.0 * np.ones(n_rows),
        'intraday_volume': 0.5 * np.ones(n_rows),
        'net_imbalance_mw': np.zeros(n_rows),
        'mfrr_price_up': 100.0 * np.ones(n_rows),
        'mfrr_price_down': 80.0 * np.ones(n_rows),
    }, index=dates)

    # I also set EXTREME future balancing data
    df.loc[df.index[extreme_start:], 'mfrr_price_up'] = 9999.0
    df.loc[df.index[extreme_start:], 'afrr_up'] = 9999.0
    df.loc[df.index[extreme_start:], 'net_imbalance_mw'] = 9999.0

    return df


class TestDAMInformationBarrier:

    def test_dam_uses_forecast_not_actual(self, barrier_test_df):
        """I verify DAM model uses price_lag_24h, NOT current price."""
        env = BatteryEnvDAM(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I check that the forecast prices in obs[7:11] are NOT extreme
        # (They should come from lag_24h, which is yesterday's price)
        forecast_features = obs[7:11]
        assert np.all(forecast_features < 50.0), \
            "DAM observation contains actual prices instead of forecasts!"

    def test_dam_no_intraday_prices(self, barrier_test_df):
        """I verify DAM observation doesn't contain IntraDay prices."""
        env = BatteryEnvDAM(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify no observation feature exceeds reasonable range
        # (IntraDay prices would be 78-82, but if leaked from future, 9999)
        assert np.all(obs < 100.0), \
            "DAM observation may contain leaked IntraDay/future prices!"

    def test_dam_no_afrr_data(self, barrier_test_df):
        """I verify DAM observation doesn't contain aFRR data."""
        env = BatteryEnvDAM(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify obs doesn't contain aFRR capacity prices
        assert np.all(obs < 100.0), \
            "DAM observation may contain leaked aFRR data!"


class TestAFRRInformationBarrier:

    def test_afrr_no_future_activation(self, barrier_test_df):
        """I verify aFRR observation has no future activation data."""
        env = BatteryEnvAFRR(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify observation doesn't leak future activation states
        assert np.all(np.abs(obs) < 100.0), \
            "aFRR observation may contain future activation data!"

    def test_afrr_uses_lagged_prices(self, barrier_test_df):
        """I verify aFRR uses lagged capacity prices, not current."""
        env = BatteryEnvAFRR(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify features 10-11 (afrr cap prices lag) are reasonable
        # If using current prices from extreme future, they'd be 9999/50 = 200
        afrr_price_features = obs[10:12]
        assert np.all(afrr_price_features < 10.0), \
            "aFRR observation may use non-lagged capacity prices!"

    def test_afrr_no_intraday_data(self, barrier_test_df):
        """I verify aFRR observation doesn't contain IntraDay prices."""
        env = BatteryEnvAFRR(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify no leaked IntraDay features
        assert np.all(np.abs(obs) < 100.0)


class TestIntraDayInformationBarrier:

    def test_intraday_uses_lagged_xbid(self, barrier_test_df):
        """I verify IntraDay uses lagged XBID prices, not current."""
        env = BatteryEnvIntraDay(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify features 11-14 (XBID lag) are reasonable
        xbid_features = obs[11:15]
        assert np.all(np.abs(xbid_features) < 10.0), \
            "IntraDay observation may use non-lagged XBID prices!"

    def test_intraday_no_mfrr_data(self, barrier_test_df):
        """I verify IntraDay observation doesn't contain mFRR data."""
        env = BatteryEnvIntraDay(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify no observation feature is extreme
        assert np.all(np.abs(obs) < 100.0), \
            "IntraDay observation may contain leaked mFRR data!"


class TestMFRRInformationBarrier:

    def test_mfrr_uses_lagged_imbalance(self, barrier_test_df):
        """I verify mFRR uses LAGGED imbalance, not real-time."""
        env = BatteryEnvMFRR(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify feature 7 (imbalance_lag / 1000) is reasonable
        imbalance_feature = obs[7]
        assert abs(imbalance_feature) < 5.0, \
            "mFRR observation may use real-time imbalance instead of lagged!"

    def test_mfrr_uses_lagged_prices(self, barrier_test_df):
        """I verify mFRR uses lagged prices, not actual clearing."""
        env = BatteryEnvMFRR(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify features 9-11 (mfrr prices lag) are reasonable
        mfrr_price_features = obs[9:12]
        assert np.all(np.abs(mfrr_price_features) < 10.0), \
            "mFRR observation may use non-lagged prices!"

    def test_mfrr_direction_from_lagged(self, barrier_test_df):
        """I verify mFRR direction constraint uses lagged imbalance."""
        env = BatteryEnvMFRR(df=barrier_test_df, random_start=False)
        obs, _ = env.reset(seed=42)

        # I verify direction feature (obs[8]) is based on lagged data
        direction = obs[8]
        assert direction in [-1.0, 0.0, 1.0], \
            f"mFRR direction should be -1/0/+1, got {direction}"
