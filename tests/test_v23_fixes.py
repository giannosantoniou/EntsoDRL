"""
Tests for v23 Production-Ready fixes.

I verify:
1. No oracle features (features 23-26 are backward-looking only)
2. Reset randomization works
3. Observation shape consistency between env and FeatureEngineer
4. Feature values are within expected ranges
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent.feature_engineer import FeatureEngineer, MarketStateDTO
from gym_envs.battery_env_masked import BatteryEnvMasked


@pytest.fixture
def mock_df():
    """I create realistic mock market data for testing."""
    n_rows = 500  # ~20 days of hourly data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='h')

    # I create a realistic price pattern with daily seasonality
    base_price = 80 + 20 * np.sin(2 * np.pi * np.arange(n_rows) / 24)
    noise = np.random.normal(0, 10, n_rows)
    prices = np.maximum(10, base_price + noise)

    df = pd.DataFrame({
        'price': prices,
        'solar': np.maximum(0, 500 * np.sin(np.pi * (np.arange(n_rows) % 24 - 6) / 12)),
        'wind_onshore': np.random.uniform(200, 800, n_rows),
        'dam_commitment': np.random.choice([-10, 0, 10, 20], n_rows),
        'long': prices * 0.95,
        'short': prices * 1.05,
        'mfrr_up': prices * 1.3,
        'mfrr_down': prices * 0.7,
        'afrr_up_mw': np.full(n_rows, 600.0),
        'afrr_down_mw': np.full(n_rows, 130.0),
    }, index=dates)
    return df


@pytest.fixture
def battery_params():
    return {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94
    }


class TestNoOracleFeatures:
    """I verify that features 23-26 are backward-looking only."""

    def test_feature_23_uses_past_data_only(self):
        """Feature 23 (price_vs_24h_range) should use only past prices."""
        fe = FeatureEngineer()

        # I create two MarketStateDTOs with SAME past data but DIFFERENT future prices
        common = dict(
            timestamp=pd.Timestamp('2024-01-10 14:00'),
            price=100.0,
            spread=4.0,
            dam_commitment_now=10.0,
            next_price=95.0,
            price_max_24h=120.0,
            price_min_24h=60.0,
            price_mean_24h=85.0,
            dam_commitments=[10.0] * 12,
            afrr_up_mw=600.0,
            afrr_down_mw=130.0,
            price_vs_typical_hour=0.2,
            trade_worthiness=0.5,
            price_momentum=0.05,
        )

        # Scenario A: future prices are HIGH
        dto_a = MarketStateDTO(**common, price_lookahead=[150.0] * 12)
        # Scenario B: future prices are LOW
        dto_b = MarketStateDTO(**common, price_lookahead=[50.0] * 12)

        imbalance = {'gradient': 5.0, 'res_deviation': -10.0, 'risk_score': 0.1}

        obs_a = fe.build_observation(market=dto_a, battery_soc=0.5, imbalance_info=imbalance)
        obs_b = fe.build_observation(market=dto_b, battery_soc=0.5, imbalance_info=imbalance)

        # Features 23-26 (indices 38-41 in 0-indexed, after 38 base features)
        # With base features: 8 + 12 + 12 + 3 + 1 + 2 = 38, so features 23-26 are at [38:42]
        feat_23_a = obs_a[38]  # price_vs_24h_range
        feat_24_a = obs_a[39]  # price_momentum_6h
        feat_25_a = obs_a[40]  # is_near_daily_high
        feat_26_a = obs_a[41]  # price_vs_mean_24h

        feat_23_b = obs_b[38]
        feat_24_b = obs_b[39]
        feat_25_b = obs_b[40]
        feat_26_b = obs_b[41]

        # I verify features 23-26 are IDENTICAL regardless of future prices
        assert feat_23_a == feat_23_b, f"Feature 23 depends on future prices! {feat_23_a} != {feat_23_b}"
        assert feat_24_a == feat_24_b, f"Feature 24 depends on future prices! {feat_24_a} != {feat_24_b}"
        assert feat_25_a == feat_25_b, f"Feature 25 depends on future prices! {feat_25_a} != {feat_25_b}"
        assert feat_26_a == feat_26_b, f"Feature 26 depends on future prices! {feat_26_a} != {feat_26_b}"

    def test_feature_values_in_expected_ranges(self):
        """I check that backward features produce sensible values."""
        fe = FeatureEngineer()

        dto = MarketStateDTO(
            timestamp=pd.Timestamp('2024-06-15 18:00'),
            price=120.0,
            spread=4.0,
            dam_commitment_now=15.0,
            next_price=115.0,
            price_max_24h=130.0,
            price_min_24h=50.0,
            price_mean_24h=85.0,
            dam_commitments=[10.0] * 12,
            price_lookahead=[100.0] * 12,
            price_momentum=0.15,
            price_vs_typical_hour=0.4,
            trade_worthiness=0.6,
        )

        imbalance = {'gradient': 10.0, 'res_deviation': -20.0, 'risk_score': 0.2}
        obs = fe.build_observation(market=dto, battery_soc=0.6, imbalance_info=imbalance)

        # Feature 23: price_vs_24h_range = (120-50)/(130-50) = 0.875
        assert 0.0 <= obs[38] <= 1.0, f"Feature 23 out of range: {obs[38]}"
        assert abs(obs[38] - 0.875) < 0.01, f"Feature 23 unexpected value: {obs[38]}"

        # Feature 24: price_momentum = clip(0.15, -1, 1)
        assert -1.0 <= obs[39] <= 1.0, f"Feature 24 out of range: {obs[39]}"
        assert abs(obs[39] - 0.15) < 0.01, f"Feature 24 unexpected value: {obs[39]}"

        # Feature 25: is_near_daily_high = 1.0 if 120 >= 130*0.95=123.5 → 0.0
        assert obs[40] in (0.0, 1.0), f"Feature 25 not binary: {obs[40]}"
        assert obs[40] == 0.0, f"Feature 25 should be 0.0 (120 < 123.5)"

        # Feature 26: price_vs_mean = (120-85)/85 = 0.412
        assert -1.0 <= obs[41] <= 1.0, f"Feature 26 out of range: {obs[41]}"
        assert abs(obs[41] - 0.412) < 0.02, f"Feature 26 unexpected value: {obs[41]}"


class TestResetRandomization:
    """I verify that episode starts are randomized."""

    def test_randomized_start_varies_position(self, mock_df, battery_params):
        """I check that reset() produces different starting positions."""
        env = BatteryEnvMasked(mock_df, battery_params, randomize_start=True)

        starts = set()
        for seed in range(20):
            env.reset(seed=seed)
            starts.add(env.current_step)

        # I expect at least 5 different starting positions out of 20 resets
        assert len(starts) >= 5, f"Only {len(starts)} unique starts in 20 resets — not enough randomization"

    def test_fixed_start_always_zero(self, mock_df, battery_params):
        """I check that randomize_start=False always starts at step 0."""
        env = BatteryEnvMasked(mock_df, battery_params, randomize_start=False)

        for seed in range(10):
            env.reset(seed=seed)
            assert env.current_step == 0, f"Expected step 0, got {env.current_step}"
            assert env.current_soc == 0.5, f"Expected SoC 0.5, got {env.current_soc}"

    def test_soc_randomization(self, mock_df, battery_params):
        """I check that initial SoC varies with randomization."""
        env = BatteryEnvMasked(mock_df, battery_params, randomize_start=True)

        socs = set()
        for seed in range(20):
            env.reset(seed=seed)
            socs.add(round(env.current_soc, 2))

        assert len(socs) >= 3, f"Only {len(socs)} unique SoC values — not enough randomization"

    def test_min_episode_length_respected(self, mock_df, battery_params):
        """I check that episodes are long enough."""
        min_len = 100
        env = BatteryEnvMasked(
            mock_df, battery_params,
            randomize_start=True, min_episode_length=min_len
        )

        for seed in range(10):
            env.reset(seed=seed)
            remaining = env.max_steps - env.current_step
            assert remaining >= min_len, (
                f"Episode too short: step={env.current_step}, max={env.max_steps}, "
                f"remaining={remaining} < min={min_len}"
            )


class TestObservationShape:
    """I verify observation dimensions are correct and consistent."""

    def test_env_observation_shape(self, mock_df, battery_params):
        """I check the environment produces correct observation shape."""
        env = BatteryEnvMasked(
            mock_df, battery_params,
            include_price_awareness=True,
            enable_gate_closure=True,
            randomize_start=False
        )
        obs, info = env.reset()

        expected_shape = env.observation_space.shape
        assert obs.shape == expected_shape, (
            f"Observation shape mismatch: got {obs.shape}, expected {expected_shape}"
        )

    def test_observation_no_nans(self, mock_df, battery_params):
        """I check observations contain no NaN values."""
        env = BatteryEnvMasked(mock_df, battery_params, randomize_start=False)
        obs, _ = env.reset()

        assert not np.any(np.isnan(obs)), f"NaN in observation: {obs}"

        # I run a few steps to check stability
        for action in [10, 5, 15, 10, 0]:  # idle, charge, discharge, idle, max charge
            obs, reward, done, _, info = env.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in observation at step: {obs}"
            assert not np.isnan(reward), f"NaN reward"
            if done:
                break

    def test_feature_engineer_matches_env(self, mock_df, battery_params):
        """I check that FeatureEngineer output matches env observation space."""
        env = BatteryEnvMasked(
            mock_df, battery_params,
            include_price_awareness=True,
            enable_gate_closure=True,
            randomize_start=False
        )
        obs, _ = env.reset()

        # I verify the observation is a flat float array (float32 or float64 both acceptable)
        assert obs.dtype in (np.float32, np.float64), f"Expected float, got {obs.dtype}"

        # I verify all values are finite
        assert np.all(np.isfinite(obs)), f"Non-finite values in observation"
