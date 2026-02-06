"""
Tests for v21: Verify NO future data leaks into observations.

I test three critical properties:
1. Changing future prices must NOT affect the current observation
2. Backward stats must only use past data
3. New trader-inspired features must be in expected ranges
"""
import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('.')

from gym_envs.battery_env_masked import BatteryEnvMasked


@pytest.fixture
def battery_params():
    return {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }


@pytest.fixture
def sample_df():
    """I create a realistic 2000-row hourly DataFrame for testing."""
    n = 2000
    np.random.seed(42)

    # I generate realistic price patterns with daily cycles
    hours = np.arange(n) % 24
    base_price = 80 + 30 * np.sin(2 * np.pi * hours / 24 - np.pi / 3)
    noise = np.random.normal(0, 10, n)
    prices = np.clip(base_price + noise, 10, 300)

    # I create a DataFrame with datetime index (hourly)
    dates = pd.date_range('2023-01-01', periods=n, freq='h')
    df = pd.DataFrame({
        'price': prices,
        'dam_commitment': np.random.choice([-10, 0, 0, 0, 10], n),
        'solar': np.maximum(0, 100 * np.sin(2 * np.pi * hours / 24 - np.pi / 4)),
        'wind_onshore': np.random.uniform(20, 80, n),
    }, index=dates)

    return df


def _make_env(df, battery_params, **kwargs):
    """I create a test environment with price awareness enabled."""
    defaults = dict(
        n_actions=21,
        use_ml_forecaster=False,
        include_price_awareness=True,
        noise_std=0.0,
        use_perfect_forecast=True,
    )
    defaults.update(kwargs)
    return BatteryEnvMasked(df, battery_params, **defaults)


class TestNoFutureLeakage:
    """I verify that future prices do NOT affect current observations."""

    def test_future_price_change_no_effect_on_obs(self, sample_df, battery_params):
        """
        Core leakage test: I modify future prices and check that the
        observation at a PAST step remains IDENTICAL.

        If this test fails, the observation is leaking future data.
        """
        test_step = 100  # I observe at step 100

        # I create environment with original prices
        env1 = _make_env(sample_df.copy(), battery_params)
        env1.reset()
        env1.current_step = test_step
        obs1 = env1._get_observation()

        # I modify ALL future prices (step 101+) to be wildly different
        df_modified = sample_df.copy()
        df_modified.iloc[test_step + 1:, df_modified.columns.get_loc('price')] = 999.0

        env2 = _make_env(df_modified, battery_params)
        env2.reset()
        env2.current_step = test_step
        obs2 = env2._get_observation()

        # I compare only the features that should be identical:
        # Features 0-8: battery state, market, time encoding
        # Features 33-38: imbalance risk, shortfall, afrr, timing
        # Features 43-44: trader-inspired (backward-only)
        #
        # Features 9-32 (price/DAM lookahead) use future data by design
        # (via forecaster or noisy peek) — that's expected
        #
        # But the backward stats (used in features 43-44) MUST be identical

        # I check features 43 (price_vs_typical_hour) and 44 (trade_worthiness)
        # These are the last 2 features in 44-feature obs
        np.testing.assert_allclose(
            obs1[-2:], obs2[-2:],
            rtol=1e-5,
            err_msg="Features 43-44 (trader signals) changed when future prices changed — DATA LEAKAGE!"
        )

        # I also check backward stats don't leak (features used in timing calculations)
        # Battery state features (0-2) should be identical
        np.testing.assert_allclose(
            obs1[:3], obs2[:3],
            rtol=1e-5,
            err_msg="Battery state features changed when future prices changed"
        )

        # Time encoding (6-7) should be identical
        np.testing.assert_allclose(
            obs1[6:8], obs2[6:8],
            rtol=1e-5,
            err_msg="Time encoding changed when future prices changed"
        )

    def test_backward_stats_only_use_past(self, sample_df, battery_params):
        """
        I verify that price_max/min/mean_24h at step i only depend
        on prices at steps [max(0, i-23) ... i], not steps [i+1 ...].
        """
        env = _make_env(sample_df, battery_params)

        prices = sample_df['price'].values
        test_step = 50

        # I manually compute the backward 24h stats for step 50
        window_start = max(0, test_step - 23)
        past_prices = prices[window_start:test_step + 1]

        expected_max = np.max(past_prices)
        expected_min = np.min(past_prices)
        expected_mean = np.mean(past_prices)

        np.testing.assert_allclose(env.price_max_24h[test_step], expected_max, rtol=1e-5,
                                   err_msg="price_max_24h uses future data!")
        np.testing.assert_allclose(env.price_min_24h[test_step], expected_min, rtol=1e-5,
                                   err_msg="price_min_24h uses future data!")
        np.testing.assert_allclose(env.price_mean_24h[test_step], expected_mean, rtol=1e-5,
                                   err_msg="price_mean_24h uses future data!")

    def test_early_steps_backward_stats(self, sample_df, battery_params):
        """
        I verify that early steps (< 24h of history) use whatever
        history is available, without errors or future peeking.
        """
        env = _make_env(sample_df, battery_params)
        prices = sample_df['price'].values

        # I check step 5 (only 6 prices available: 0..5)
        test_step = 5
        past = prices[:test_step + 1]

        np.testing.assert_allclose(env.price_max_24h[test_step], np.max(past), rtol=1e-5)
        np.testing.assert_allclose(env.price_min_24h[test_step], np.min(past), rtol=1e-5)
        np.testing.assert_allclose(env.price_mean_24h[test_step], np.mean(past), rtol=1e-5)

    def test_step_0_no_crash(self, sample_df, battery_params):
        """I verify that step 0 works (edge case: no history at all)."""
        env = _make_env(sample_df, battery_params)
        env.reset()
        env.current_step = 0
        obs = env._get_observation()

        assert obs.shape == (44,), f"Expected 44 features, got {obs.shape}"
        assert not np.any(np.isnan(obs)), "NaN found in observation at step 0"
        assert not np.any(np.isinf(obs)), "Inf found in observation at step 0"


class TestTraderFeatures:
    """I verify the new trader-inspired features (43, 44) are correct."""

    def test_price_vs_typical_hour_range(self, sample_df, battery_params):
        """I check that price_vs_typical_hour is always in [-1, 1]."""
        env = _make_env(sample_df, battery_params)

        assert np.all(env.price_vs_typical_hour >= -1.0), \
            f"price_vs_typical_hour below -1: min={env.price_vs_typical_hour.min()}"
        assert np.all(env.price_vs_typical_hour <= 1.0), \
            f"price_vs_typical_hour above 1: max={env.price_vs_typical_hour.max()}"

    def test_trade_worthiness_range(self, sample_df, battery_params):
        """I check that trade_worthiness is always in [0, 1]."""
        env = _make_env(sample_df, battery_params)

        assert np.all(env.trade_worthiness >= 0.0), \
            f"trade_worthiness below 0: min={env.trade_worthiness.min()}"
        assert np.all(env.trade_worthiness <= 1.0), \
            f"trade_worthiness above 1: max={env.trade_worthiness.max()}"

    def test_price_vs_typical_hour_is_backward(self, sample_df, battery_params):
        """
        I verify price_vs_typical_hour at step i doesn't depend on
        prices after step i by modifying future prices.
        """
        test_step = 200

        env1 = _make_env(sample_df.copy(), battery_params)
        val1 = env1.price_vs_typical_hour[test_step]

        # I corrupt all future prices
        df_mod = sample_df.copy()
        df_mod.iloc[test_step + 1:, df_mod.columns.get_loc('price')] = 500.0

        env2 = _make_env(df_mod, battery_params)
        val2 = env2.price_vs_typical_hour[test_step]

        np.testing.assert_allclose(
            val1, val2, rtol=1e-5,
            err_msg="price_vs_typical_hour changed with future prices — DATA LEAKAGE!"
        )

    def test_trade_worthiness_is_backward(self, sample_df, battery_params):
        """
        I verify trade_worthiness at step i doesn't depend on
        prices after step i.
        """
        test_step = 200

        env1 = _make_env(sample_df.copy(), battery_params)
        val1 = env1.trade_worthiness[test_step]

        # I corrupt all future prices
        df_mod = sample_df.copy()
        df_mod.iloc[test_step + 1:, df_mod.columns.get_loc('price')] = 500.0

        env2 = _make_env(df_mod, battery_params)
        val2 = env2.trade_worthiness[test_step]

        np.testing.assert_allclose(
            val1, val2, rtol=1e-5,
            err_msg="trade_worthiness changed with future prices — DATA LEAKAGE!"
        )

    def test_high_price_hour_positive_signal(self, sample_df, battery_params):
        """
        I verify that a spike price at a normally cheap hour produces
        a positive price_vs_typical_hour value (above typical = sell signal).
        """
        df = sample_df.copy()

        # I inject a spike at step 800 (should have enough history by then)
        # I set a very high price that is far above the 30-day hourly average
        original_price = df.iloc[800]['price']
        df.iloc[800, df.columns.get_loc('price')] = original_price + 200.0

        env = _make_env(df, battery_params)

        # I expect the signal to be positive (price above typical for this hour)
        assert env.price_vs_typical_hour[800] > 0.0, \
            f"Spike should produce positive signal, got {env.price_vs_typical_hour[800]}"


class TestObservationShape:
    """I verify the observation vector shape and basic properties."""

    def test_obs_shape_with_price_awareness(self, sample_df, battery_params):
        """I check 44 features when include_price_awareness=True."""
        env = _make_env(sample_df, battery_params, include_price_awareness=True)
        env.reset()
        obs = env._get_observation()
        assert obs.shape == (44,), f"Expected 44 features, got {obs.shape}"

    def test_obs_shape_without_price_awareness(self, sample_df, battery_params):
        """I check 42 features when include_price_awareness=False."""
        env = _make_env(sample_df, battery_params, include_price_awareness=False)
        env.reset()
        obs = env._get_observation()
        assert obs.shape == (42,), f"Expected 42 features, got {obs.shape}"

    def test_obs_no_nan(self, sample_df, battery_params):
        """I run a few steps and verify no NaN values appear."""
        env = _make_env(sample_df, battery_params)
        obs, _ = env.reset()

        for _ in range(100):
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)

            assert not np.any(np.isnan(obs)), f"NaN in observation at step {env.current_step}"
            assert not np.any(np.isinf(obs)), f"Inf in observation at step {env.current_step}"
            assert np.isfinite(reward), f"Non-finite reward at step {env.current_step}"

            if terminated:
                obs, _ = env.reset()

    def test_full_episode_no_crash(self, sample_df, battery_params):
        """I run a complete episode to verify no crashes."""
        env = _make_env(sample_df, battery_params)
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated:
                break

        assert steps > 100, f"Episode too short: {steps} steps"
        assert np.isfinite(total_reward), f"Total reward not finite: {total_reward}"


class TestBackwardStatsConsistency:
    """I verify backward stats are consistent across the entire dataset."""

    def test_max_never_less_than_min(self, sample_df, battery_params):
        """I verify max_24h >= min_24h at every step."""
        env = _make_env(sample_df, battery_params)

        diff = env.price_max_24h - env.price_min_24h
        assert np.all(diff >= -1e-10), \
            f"max_24h < min_24h at some steps: min diff = {diff.min()}"

    def test_mean_between_min_max(self, sample_df, battery_params):
        """I verify min_24h <= mean_24h <= max_24h at every step."""
        env = _make_env(sample_df, battery_params)

        assert np.all(env.price_mean_24h >= env.price_min_24h - 1e-10), \
            "mean_24h < min_24h at some steps"
        assert np.all(env.price_mean_24h <= env.price_max_24h + 1e-10), \
            "mean_24h > max_24h at some steps"

    def test_monotonically_expanding_early(self, sample_df, battery_params):
        """
        I verify that in the first 24 steps, the max is monotonically
        increasing (or equal), because the window is still growing.
        """
        env = _make_env(sample_df, battery_params)
        prices = sample_df['price'].values

        for i in range(1, 24):
            # I check: max at step i must be >= max at step i-1 OR
            # the new price is smaller than the running max
            running_max = np.max(prices[:i + 1])
            np.testing.assert_allclose(
                env.price_max_24h[i], running_max, rtol=1e-5,
                err_msg=f"max_24h at step {i} doesn't match running max"
            )
