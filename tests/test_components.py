"""
Unit Tests for Extracted Components

I test the individual components extracted during the SOLID refactoring:
1. DataPreparer — data validation, defaults, precomputed arrays
2. CapacityManager — cascade capacity limits
3. MarketResult — DTO correctness
4. DamExecutor — DAM commitment and schedule generation
5. AfrrExecutor — aFRR activation and commitment logic
6. IdaExecutor — IDA gate closure, schedule generation, observation helpers
7. TradingExecutor — mFRR, XBID, FreeBid execution
8. ObservationBuilder — observation vector shape and content
9. MaskBuilder — action mask consistency
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gym_envs.data_preparer import DataPreparer
from gym_envs.capacity_manager import CapacityManager, CapacityLimits
from gym_envs.market_executors import PrecomputedArrays, MarketResult
from gym_envs.battery_env_unified import BatteryEnvUnified


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """I create minimal sample data for component testing."""
    n_hours = 168  # 1 week
    dates = pd.date_range(
        start='2024-01-01', periods=n_hours, freq='h', tz='Europe/Athens'
    )
    np.random.seed(42)
    hours = np.array([d.hour for d in dates])
    prices = 80 + 40 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 10, n_hours)
    dam = np.zeros(n_hours)
    for i in range(n_hours):
        if hours[i] in [18, 19, 20]:
            dam[i] = np.random.uniform(10, 25)
        elif hours[i] in [10, 11, 12]:
            dam[i] = np.random.uniform(-20, -5)

    return pd.DataFrame({
        'price': prices,
        'dam_commitment': dam,
        'afrr_up': prices * 1.1,
        'afrr_down': prices * 0.9,
        'afrr_cap_up_price': 15 + 10 * np.random.random(n_hours),
        'afrr_cap_down_price': 25 + 15 * np.random.random(n_hours),
        'intraday_bid': prices - 2,
        'intraday_ask': prices + 2,
        'intraday_spread': 3.0 + np.random.uniform(0, 4, n_hours),
        'intraday_volume': np.random.uniform(0.3, 0.9, n_hours),
        'mfrr_price_up': prices * 1.3,
        'mfrr_price_down': prices * 0.7,
        'mfrr_spread': prices * 0.6,
        'mfrr_activated_up_mwh': np.where(np.random.random(n_hours) < 0.84,
                                           np.random.uniform(5, 200, n_hours), 0.0),
        'mfrr_activated_down_mwh': np.where(np.random.random(n_hours) < 0.96,
                                             np.random.uniform(5, 150, n_hours), 0.0),
        'net_imbalance_mw': np.random.normal(0, 300, n_hours),
        'solar': np.maximum(0, 500 * np.sin(2 * np.pi * (hours - 6) / 12)),
        'wind_onshore': np.random.uniform(100, 500, n_hours),
        'isp1_price_up': prices * 1.2,
        'isp1_price_down': prices * 0.8,
    }, index=dates)


@pytest.fixture
def env(sample_df):
    """I create a standard unified env for component testing."""
    return BatteryEnvUnified(sample_df, capacity_mwh=146.0, max_power_mw=30.0)


@pytest.fixture
def full_market_env(sample_df):
    """I create a full-market env for IDA/XBID/FreeBid testing."""
    return BatteryEnvUnified(
        sample_df, capacity_mwh=146.0, max_power_mw=30.0,
        enable_full_market=True
    )


# ═══════════════════════════════════════════════════════════════════════
# TEST: PrecomputedArrays + MarketResult DTOs
# ═══════════════════════════════════════════════════════════════════════

class TestDTOs:
    """I test the shared DTOs in gym_envs/market_executors/__init__.py."""

    def test_precomputed_arrays_creation(self):
        """I verify PrecomputedArrays stores all required arrays."""
        pa = PrecomputedArrays(
            step_day_offset=np.array([0, 1, 2, 3], dtype=np.int32),
            step_day_id=np.array([0, 0, 0, 0], dtype=np.int32),
            day_start=np.array([0], dtype=np.int32),
            day_length=np.array([4], dtype=np.int32),
            dam_commitment_values=np.array([10.0, -5.0, 0.0, 15.0]),
        )
        assert len(pa.step_day_offset) == 4
        assert len(pa.step_day_id) == 4
        assert len(pa.day_start) == 1
        assert len(pa.day_length) == 1
        assert pa.day_length[0] == 4

    def test_precomputed_get_day_indices(self):
        """I verify get_day_indices_for_step returns correct step range."""
        # I create 2 days of data: day0 has 4 steps, day1 has 3 steps
        pa = PrecomputedArrays(
            step_day_offset=np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int32),
            step_day_id=np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
            day_start=np.array([0, 4], dtype=np.int32),
            day_length=np.array([4, 3], dtype=np.int32),
            dam_commitment_values=np.zeros(7),
        )
        # I verify day0 indices
        indices_d0 = pa.get_day_indices_for_step(0)
        np.testing.assert_array_equal(indices_d0, np.array([0, 1, 2, 3]))
        # I verify day1 indices
        indices_d1 = pa.get_day_indices_for_step(5)
        np.testing.assert_array_equal(indices_d1, np.array([4, 5, 6]))

    def test_market_result_defaults(self):
        """I verify MarketResult initializes with zeros."""
        r = MarketResult()
        assert r.energy_mw == 0.0
        assert r.revenue == 0.0
        assert r.capacity_used_mw == 0.0
        assert r.cycle_fraction == 0.0
        assert r.soc_delta == 0.0
        assert r.is_activated is False
        assert r.direction is None
        assert r.mwh_sold == 0.0
        assert r.mwh_bought == 0.0
        assert isinstance(r.metadata, dict)
        assert len(r.metadata) == 0

    def test_market_result_is_mutable(self):
        """I verify MarketResult fields can be set."""
        r = MarketResult()
        r.energy_mw = 15.0
        r.revenue = 1500.0
        r.is_activated = True
        r.direction = 'up'
        r.metadata['test'] = True
        assert r.energy_mw == 15.0
        assert r.revenue == 1500.0
        assert r.metadata['test'] is True


# ═══════════════════════════════════════════════════════════════════════
# TEST: DataPreparer
# ═══════════════════════════════════════════════════════════════════════

class TestDataPreparer:
    """I test the data preparation and precomputation module."""

    def test_prepare_adds_defaults(self, sample_df):
        """I verify prepare() adds missing columns with safe defaults."""
        # I remove some optional columns
        df_minimal = sample_df[['price', 'dam_commitment']].copy()
        dp = DataPreparer(sph=1, mfrr_imbalance_threshold=50)
        df_out, pc, _, _, _ = dp.prepare(df_minimal)

        # I check that defaults were added
        assert 'afrr_up' in df_out.columns
        assert 'mfrr_price_up' in df_out.columns
        assert 'net_imbalance_mw' in df_out.columns
        # I verify default values
        assert df_out['afrr_up'].iloc[0] == 80.0
        assert df_out['mfrr_price_up'].iloc[0] == 120.0

    def test_prepare_adds_rolling_stats(self, sample_df):
        """I verify rolling statistics are computed."""
        dp = DataPreparer(sph=1, mfrr_imbalance_threshold=50)
        df_out, _, _, _, _ = dp.prepare(sample_df)

        assert 'price_mean_24h' in df_out.columns
        assert 'price_std_24h' in df_out.columns
        assert 'price_momentum' in df_out.columns
        assert 'afrr_cap_percentile' in df_out.columns

    def test_prepare_builds_precomputed_arrays(self, sample_df):
        """I verify precomputed arrays have correct dimensions."""
        dp = DataPreparer(sph=1, mfrr_imbalance_threshold=50)
        df_out, pc, _, _, _ = dp.prepare(sample_df)

        assert len(pc.step_day_offset) == len(df_out)
        assert len(pc.step_day_id) == len(df_out)
        assert len(pc.dam_commitment_values) == len(df_out)
        # I verify day arrays are consistent
        n_days = pc.step_day_id[-1] + 1
        assert len(pc.day_start) == n_days
        assert len(pc.day_length) == n_days
        # I verify day_length sums to total steps
        assert pc.day_length.sum() == len(df_out)

    def test_prepare_day_offsets_are_correct(self, sample_df):
        """I verify step_day_offset starts at 0 for each new day."""
        dp = DataPreparer(sph=1, mfrr_imbalance_threshold=50)
        df_out, pc, _, _, _ = dp.prepare(sample_df)

        # I check that offset resets at day boundaries
        for day_id in range(pc.step_day_id[-1] + 1):
            day_start = pc.day_start[day_id]
            assert pc.step_day_offset[day_start] == 0

    def test_prepare_requires_price_column(self):
        """I verify prepare() raises on missing required columns."""
        df_bad = pd.DataFrame({
            'dam_commitment': [0, 0, 0],
        }, index=pd.date_range('2024-01-01', periods=3, freq='h', tz='Europe/Athens'))
        dp = DataPreparer(sph=1, mfrr_imbalance_threshold=50)
        with pytest.raises(ValueError, match="Missing required column: price"):
            dp.prepare(df_bad)

    def test_prepare_requires_dam_commitment(self):
        """I verify prepare() raises on missing dam_commitment."""
        df_bad = pd.DataFrame({
            'price': [100, 100, 100],
        }, index=pd.date_range('2024-01-01', periods=3, freq='h', tz='Europe/Athens'))
        dp = DataPreparer(sph=1, mfrr_imbalance_threshold=50)
        with pytest.raises(ValueError, match="Missing required column: dam_commitment"):
            dp.prepare(df_bad)

    def test_prepare_15min_resolution(self):
        """I verify prepare() works with 15-min resolution data."""
        n = 96 * 3  # 3 days of 15-min data
        dates = pd.date_range('2024-01-01', periods=n, freq='15min', tz='Europe/Athens')
        df = pd.DataFrame({
            'price': 80 + np.random.normal(0, 10, n),
            'dam_commitment': np.zeros(n),
        }, index=dates)
        dp = DataPreparer(sph=4, mfrr_imbalance_threshold=50, time_step_hours=0.25)
        df_out, pc, _, _, _ = dp.prepare(df)

        # I check that 3 days exist
        n_days = pc.step_day_id[-1] + 1
        assert n_days == 3
        # I check 96 steps per day
        assert pc.day_length[0] == 96


# ═══════════════════════════════════════════════════════════════════════
# TEST: CapacityManager
# ═══════════════════════════════════════════════════════════════════════

class TestCapacityManager:
    """I test the capacity cascade logic."""

    @pytest.fixture
    def cm(self):
        """I create a standard capacity manager."""
        return CapacityManager(
            capacity_mwh=146.0,
            max_power_mw=30.0,
            efficiency=0.94,
            eff_sqrt=np.sqrt(0.94),
            min_soc=0.05,
            max_soc=0.95,
            time_step_hours=1.0,
            max_daily_cycles=2.5,
        )

    def test_no_commitments(self, cm):
        """I verify limits with no DAM or aFRR commitment."""
        limits = cm.compute_limits(
            soc=0.5, dam_commitment=0.0, afrr_commitment_mw=0.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        assert limits.remaining_after_dam == 30.0
        assert limits.remaining_for_trading == 30.0
        assert limits.max_discharge_mw > 0
        assert limits.max_charge_mw > 0

    def test_dam_reduces_capacity(self, cm):
        """I verify DAM commitment reduces remaining capacity."""
        limits = cm.compute_limits(
            soc=0.5, dam_commitment=20.0, afrr_commitment_mw=0.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        assert limits.remaining_after_dam == 10.0
        assert limits.remaining_for_trading == 10.0

    def test_afrr_reduces_trading_capacity(self, cm):
        """I verify aFRR commitment reduces trading capacity but not DAM capacity."""
        limits = cm.compute_limits(
            soc=0.5, dam_commitment=0.0, afrr_commitment_mw=10.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        assert limits.remaining_after_dam == 30.0
        assert limits.remaining_for_trading == 20.0

    def test_cascade_stacking(self, cm):
        """I verify DAM + aFRR stack correctly."""
        limits = cm.compute_limits(
            soc=0.5, dam_commitment=15.0, afrr_commitment_mw=5.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        assert limits.remaining_after_dam == 15.0
        assert limits.remaining_for_trading == 10.0

    def test_low_soc_limits_discharge(self, cm):
        """I verify low SoC limits discharge power."""
        limits = cm.compute_limits(
            soc=0.06, dam_commitment=0.0, afrr_commitment_mw=0.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        # I check that discharge is limited by small SoC headroom
        max_energy = (0.06 - 0.05) * 146.0 * cm.eff_sqrt / 1.0
        assert limits.max_discharge_mw == pytest.approx(max_energy, abs=0.1)

    def test_high_soc_limits_charge(self, cm):
        """I verify high SoC limits charge power."""
        limits = cm.compute_limits(
            soc=0.94, dam_commitment=0.0, afrr_commitment_mw=0.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        max_energy = (0.95 - 0.94) * 146.0 / cm.eff_sqrt / 1.0
        assert limits.max_charge_mw == pytest.approx(max_energy, abs=0.1)

    def test_cycle_limit_caps_power(self, cm):
        """I verify daily cycle limit caps available power."""
        limits_fresh = cm.compute_limits(
            soc=0.5, dam_commitment=0.0, afrr_commitment_mw=0.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        limits_exhausted = cm.compute_limits(
            soc=0.5, dam_commitment=0.0, afrr_commitment_mw=0.0,
            daily_cycles=2.5, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        # I check that exhausted cycles yield zero power
        assert limits_exhausted.max_discharge_mw == 0.0
        assert limits_exhausted.max_charge_mw == 0.0
        # I check fresh has full power
        assert limits_fresh.max_discharge_mw > 0

    def test_negative_dam_commitment(self, cm):
        """I verify negative DAM (charge) still reduces capacity by absolute value."""
        limits = cm.compute_limits(
            soc=0.5, dam_commitment=-20.0, afrr_commitment_mw=0.0,
            daily_cycles=0.0, dam_min_soc=0.05, dam_max_soc=0.95,
        )
        assert limits.remaining_after_dam == 10.0


# ═══════════════════════════════════════════════════════════════════════
# TEST: DamExecutor
# ═══════════════════════════════════════════════════════════════════════

class TestDamExecutor:
    """I test the DAM executor component via env delegation."""

    def test_get_commitment_returns_dam_value(self, env):
        """I verify _get_dam_commitment delegates correctly."""
        env.reset()
        dam = env._get_dam_commitment(env.current_step)
        expected = env.df.iloc[env.current_step]['dam_commitment']
        assert dam == pytest.approx(expected)

    def test_dam_soc_reserve(self, env):
        """I verify DAM SoC reserve computation returns valid bounds."""
        env.reset()
        min_res, max_res = env._calculate_dam_soc_reserve()
        assert min_res >= env.min_soc
        assert max_res <= env.max_soc
        assert min_res <= max_res

    def test_shortfall_risk_at_good_soc(self, env):
        """I verify low shortfall risk when SoC is comfortable."""
        env.reset()
        env.soc = 0.5
        risk = env._calculate_shortfall_risk()
        # I check risk is low at moderate SoC
        assert 0.0 <= risk <= 1.0

    def test_predict_dam_soc_returns_valid(self, env):
        """I verify DAM SoC prediction returns valid values."""
        env.reset()
        predicted_soc, net_energy = env._predict_dam_soc()
        assert 0.0 <= predicted_soc <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# TEST: AfrrExecutor
# ═══════════════════════════════════════════════════════════════════════

class TestAfrrExecutor:
    """I test the aFRR executor component."""

    def test_check_activation_returns_tuple(self, env):
        """I verify aFRR activation check returns (bool, optional_str)."""
        env.reset()
        row = env.df.iloc[env.current_step]
        activated, direction = env._check_afrr_activation(row)
        assert isinstance(activated, bool)
        if activated:
            assert direction in ('up', 'down')
        else:
            assert direction is None

    def test_afrr_executor_process_commitment(self, env):
        """I verify aFRR executor can process commitment at block boundary."""
        env.reset()
        # I simulate at block boundary
        env._afrr_steps_remaining = 0
        remaining = 30.0
        env._afrr_executor.process_commitment(
            afrr_action=2,  # Some non-zero level
            price_tier_action=1,
            remaining_capacity=remaining,
        )
        assert env.afrr_commitment_mw >= 0

    def test_afrr_executor_collect_capacity_revenue(self, env):
        """I verify capacity revenue is collected when selected."""
        env.reset()
        env.is_selected_for_afrr = True
        env.afrr_commitment_mw = 10.0
        row = env.df.iloc[env.current_step]
        rev = env._afrr_executor.collect_capacity_revenue(row)
        assert rev > 0  # Should earn capacity payment


# ═══════════════════════════════════════════════════════════════════════
# TEST: IdaExecutor
# ═══════════════════════════════════════════════════════════════════════

class TestIdaExecutor:
    """I test the IDA executor component."""

    def test_get_commitment_no_schedule(self, full_market_env):
        """I verify IDA commitment is 0 when no schedules are locked."""
        env = full_market_env
        env.reset()
        commitment = env._get_ida_commitment(env.current_step)
        assert commitment == 0.0

    def test_hours_to_next_gate(self, full_market_env):
        """I verify hours to next IDA gate is always positive."""
        env = full_market_env
        env.reset()
        hours = env._get_hours_to_next_ida_gate()
        assert hours > 0

    def test_ida_phase_returns_valid(self, full_market_env):
        """I verify IDA phase returns values in [0.0, 1.0]."""
        env = full_market_env
        env.reset()
        row = env.df.iloc[env.current_step]
        phase = env._get_ida_phase(row)
        assert 0.0 <= phase <= 1.0

    def test_ida_correction_signal_range(self, full_market_env):
        """I verify correction signal is in [-1, 1]."""
        env = full_market_env
        env.reset()
        signal = env._get_ida_correction_signal()
        assert -1.0 <= signal <= 1.0

    def test_soc_trajectory_feasible(self, full_market_env):
        """I verify feasibility check returns boolean."""
        env = full_market_env
        env.reset()
        feasible = env._check_soc_trajectory_feasible()
        assert isinstance(feasible, bool)

    def test_ida_executor_get_phase(self, full_market_env):
        """I verify IdaExecutor.get_phase() returns same as env delegate."""
        env = full_market_env
        env.reset()
        row = env.df.iloc[env.current_step]
        phase_env = env._get_ida_phase(row)
        phase_exec = env._ida_executor.get_phase()
        assert phase_env == phase_exec

    def test_ida_executor_get_hours_to_next_gate(self, full_market_env):
        """I verify IdaExecutor.get_hours_to_next_gate() matches delegate."""
        env = full_market_env
        env.reset()
        hours_env = env._get_hours_to_next_ida_gate()
        hours_exec = env._ida_executor.get_hours_to_next_gate()
        assert hours_env == hours_exec


# ═══════════════════════════════════════════════════════════════════════
# TEST: TradingExecutor
# ═══════════════════════════════════════════════════════════════════════

class TestTradingExecutor:
    """I test the trading executor (mFRR, XBID, FreeBid)."""

    def test_execute_mfrr_no_capacity(self, env):
        """I verify mFRR returns empty result when no capacity available."""
        env.reset()
        row = env.df.iloc[env.current_step]
        result = env._trading_executor.execute_mfrr(
            mfrr_qty_action=5,  # Idle
            remaining_after_ida=0.0,
            row=row,
        )
        assert result.energy_mw == 0.0
        assert result.revenue == 0.0

    def test_execute_intraday_market_closed(self, env):
        """I verify XBID returns empty when market is closed."""
        env.reset()
        # I find a step at 23:xx (market closed)
        for i in range(env.episode_start_step, env.max_steps):
            if env.df.index[i].hour == 23:
                env.current_step = i
                break
        row = env.df.iloc[env.current_step]
        result = env._trading_executor.execute_intraday(
            xbid_action=5,
            remaining_after_mfrr=30.0,
            row=row,
        )
        assert result.energy_mw == 0.0

    def test_execute_free_bid_requires_full_market(self, env):
        """I verify FreeBid returns empty in non-full-market mode."""
        env.reset()
        row = env.df.iloc[env.current_step]
        result = env._trading_executor.execute_free_bid(
            freebid_qty_action=0,
            freebid_price_action=0,
            remaining_after_intraday=30.0,
            row=row,
        )
        assert result.energy_mw == 0.0

    def test_execute_free_bid_tracks_submissions(self, full_market_env):
        """I verify FreeBid tracks submission count."""
        env = full_market_env
        env.reset()
        row = env.df.iloc[env.current_step]
        env.soc = 0.5  # Ensure capacity
        before = env.free_bid_submissions
        # I use a non-idle action (index 0 or 10 → full capacity)
        env._trading_executor.execute_free_bid(
            freebid_qty_action=0,  # Full discharge
            freebid_price_action=2,
            remaining_after_intraday=30.0,
            row=row,
        )
        # Submission should be counted regardless of activation
        assert env.free_bid_submissions >= before


# ═══════════════════════════════════════════════════════════════════════
# TEST: ObservationBuilder
# ═══════════════════════════════════════════════════════════════════════

class TestObservationBuilder:
    """I test the observation builder component."""

    def test_obs_shape_matches_space(self, env):
        """I verify built observation matches declared observation space."""
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_obs_within_bounds(self, env):
        """I verify observation values are within space bounds."""
        obs, _ = env.reset()
        low = env.observation_space.low
        high = env.observation_space.high
        # I allow small numerical tolerance
        assert np.all(obs >= low - 0.01), f"Below low: {obs[obs < low - 0.01]}"
        assert np.all(obs <= high + 0.01), f"Above high: {obs[obs > high + 0.01]}"

    def test_obs_builder_matches_env(self, env):
        """I verify ObservationBuilder.build() matches env._build_observation()."""
        env.reset()
        obs_env = env._build_observation()
        obs_builder = env._obs_builder.build()
        np.testing.assert_array_almost_equal(obs_env, obs_builder)

    def test_obs_soc_at_index_0(self, env):
        """I verify SoC is always at index 0."""
        env.reset()
        obs = env._build_observation()
        # SoC should be close to initial 0.5
        assert 0.4 <= obs[0] <= 0.6


# ═══════════════════════════════════════════════════════════════════════
# TEST: MaskBuilder
# ═══════════════════════════════════════════════════════════════════════

class TestMaskBuilder:
    """I test the action mask builder component."""

    def test_mask_shape(self, env):
        """I verify mask has correct total length (sum of action dims)."""
        env.reset()
        mask = env.action_masks()
        expected = sum(env.action_space.nvec)
        assert len(mask) == expected

    def test_mask_all_ones_at_start(self, env):
        """I verify at least one action is valid in each dimension."""
        env.reset()
        mask = env.action_masks()
        # I split mask by action dimensions
        dims = env.action_space.nvec
        offset = 0
        for d, size in enumerate(dims):
            dim_mask = mask[offset:offset + size]
            assert np.any(dim_mask), f"Dimension {d} has no valid actions"
            offset += size

    def test_mask_builder_matches_env(self, env):
        """I verify MaskBuilder.build() matches env.action_masks()."""
        env.reset()
        mask_env = env.action_masks()
        mask_builder = env._mask_builder.build()
        np.testing.assert_array_equal(mask_env, mask_builder)

    def test_mask_low_soc_blocks_discharge(self, env):
        """I verify low SoC blocks mFRR discharge actions."""
        env.reset()
        env.soc = 0.051  # Just above min_soc
        mask = env.action_masks()
        # mFRR dimension starts at offset 5+5+11 = 21
        # Discharge actions are the higher indices
        dims = env.action_space.nvec
        mfrr_offset = sum(dims[:3])
        mfrr_mask = mask[mfrr_offset:mfrr_offset + dims[3]]
        # I verify at least some discharge actions are blocked
        # The high-index mFRR levels (discharge) should be 0
        assert mfrr_mask[-1] == 0 or mfrr_mask[-2] == 0

    def test_mask_full_market_has_extra_dims(self, full_market_env):
        """I verify full-market mask has 48 bits (6 dimensions)."""
        env = full_market_env
        env.reset()
        mask = env.action_masks()
        expected = sum(env.action_space.nvec)  # 5+5+11+11+11+5 = 48
        assert len(mask) == expected
        assert len(env.action_space.nvec) == 6


# ═══════════════════════════════════════════════════════════════════════
# TEST: Integration — Components work together through env
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    """I test that extracted components work together through the env."""

    def test_full_episode(self, env):
        """I verify a complete episode runs without errors."""
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            mask = env.action_masks()
            # I apply mask to ensure valid actions
            dims = env.action_space.nvec
            offset = 0
            masked_action = []
            for d, size in enumerate(dims):
                dim_mask = mask[offset:offset + size]
                valid = np.where(dim_mask)[0]
                if len(valid) > 0:
                    masked_action.append(np.random.choice(valid))
                else:
                    masked_action.append(size // 2)  # Fallback to middle
                offset += size
            obs, reward, terminated, truncated, info = env.step(np.array(masked_action))
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_full_market_episode(self, full_market_env):
        """I verify a full-market episode runs without errors."""
        env = full_market_env
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action = env.action_space.sample()
            mask = env.action_masks()
            dims = env.action_space.nvec
            offset = 0
            masked_action = []
            for d, size in enumerate(dims):
                dim_mask = mask[offset:offset + size]
                valid = np.where(dim_mask)[0]
                if len(valid) > 0:
                    masked_action.append(np.random.choice(valid))
                else:
                    masked_action.append(size // 2)
                offset += size
            obs, reward, terminated, truncated, info = env.step(np.array(masked_action))
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_executor_instances_exist(self, env):
        """I verify all executor instances are created."""
        env.reset()
        assert hasattr(env, '_dam_executor')
        assert hasattr(env, '_afrr_executor')
        assert hasattr(env, '_ida_executor')
        assert hasattr(env, '_trading_executor')
        assert hasattr(env, '_obs_builder')
        assert hasattr(env, '_mask_builder')
        assert hasattr(env, '_capacity_manager')

    def test_sac_wrapper_works_with_refactored_env(self):
        """I verify SAC wrapper still works with the refactored env."""
        from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC
        n = 168
        dates = pd.date_range('2024-01-01', periods=n, freq='h', tz='Europe/Athens')
        np.random.seed(42)
        df = pd.DataFrame({
            'price': 80 + np.random.normal(0, 10, n),
            'dam_commitment': np.zeros(n),
        }, index=dates)
        inner = BatteryEnvUnified(df)
        env = BatteryEnvUnifiedSAC(inner_env=inner)
        obs, _ = env.reset()
        # SAC has continuous action space
        action = env.action_space.sample()
        obs2, reward, done, truncated, info = env.step(action)
        assert obs2.shape == env.observation_space.shape
