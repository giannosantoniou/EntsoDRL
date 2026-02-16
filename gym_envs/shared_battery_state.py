"""
Shared Battery State & Upstream Simulator for Decomposed Market Models

I implement the shared battery state that flows across all 4 decomposed market
environments (DAM -> aFRR -> IntraDay -> mFRR). Each downstream model sees the
commitments made by upstream models and the resulting SoC.

The UpstreamSimulator provides upstream market decisions for downstream model
training - either from frozen trained models or from data/heuristic fallbacks.

Key Design Decisions:
1. SharedBatteryState is a dataclass with methods for energy application
2. Capacity cascading: each downstream model only sees REMAINING capacity
3. UpstreamSimulator can use frozen models OR data column fallbacks
4. SoC trajectory simulation for forward feasibility checking
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
from copy import deepcopy


@dataclass
class SharedBatteryState:
    """
    I hold the physical battery state shared across all 4 market environments.

    Each market model reads and updates this state in sequence:
    DAM (hourly) -> aFRR (4h blocks) -> IntraDay (per ISP) -> mFRR (per ISP)
    """

    # I define physical parameters
    soc: float = 0.5
    capacity_mwh: float = 146.0
    max_power_mw: float = 30.0
    efficiency: float = 0.94
    min_soc: float = 0.05
    max_soc: float = 0.95

    # I track cycling for degradation
    total_cycles: float = 0.0
    daily_cycles: float = 0.0
    max_daily_cycles: float = 2.0
    daily_energy_throughput: float = 0.0

    # I store upstream commitments for the current day
    # DAM: 24 hourly MW values (+ = sell/discharge, - = buy/charge)
    dam_schedule: np.ndarray = field(
        default_factory=lambda: np.zeros(24, dtype=np.float32)
    )
    # aFRR: 6 blocks per day, MW capacity reserved
    afrr_commitments: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    # aFRR price tiers: 6 blocks per day
    afrr_price_tiers: np.ndarray = field(
        default_factory=lambda: np.ones(6, dtype=np.int32) * 2  # neutral
    )
    # IntraDay: 96 ISPs per day (15-min), MW positions
    intraday_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(96, dtype=np.float32)
    )

    # I track aFRR activation state (set by UpstreamSimulator or env)
    afrr_is_selected: bool = False
    afrr_is_activated: bool = False
    afrr_activation_direction: str = 'none'  # 'up', 'down', 'none'
    afrr_activated_mw: float = 0.0

    def __post_init__(self):
        """I pre-compute efficiency square root for energy calculations."""
        self.eff_sqrt = np.sqrt(self.efficiency)

    def apply_energy(self, power_mw: float, duration_h: float) -> float:
        """
        I apply an energy transfer to the battery and update SoC + cycles.

        Args:
            power_mw: Power in MW (+ = discharge/sell, - = charge/buy)
            duration_h: Duration in hours (e.g. 0.25 for 15-min ISP)

        Returns:
            Actual energy transferred in MWh (after efficiency + SoC clipping)
        """
        if abs(power_mw) < 0.01:
            return 0.0

        if power_mw > 0:
            # I discharge: energy leaves battery (with efficiency loss)
            max_discharge_energy = (
                (self.soc - self.min_soc) * self.capacity_mwh * self.eff_sqrt
            )
            requested_energy = power_mw * duration_h
            actual_energy = min(requested_energy, max_discharge_energy)
            actual_energy = max(actual_energy, 0.0)
            soc_delta = actual_energy / (self.eff_sqrt * self.capacity_mwh)
            self.soc -= soc_delta
        else:
            # I charge: energy enters battery (with efficiency loss)
            max_charge_energy = (
                (self.max_soc - self.soc) * self.capacity_mwh / self.eff_sqrt
            )
            requested_energy = abs(power_mw) * duration_h
            actual_energy = min(requested_energy, max_charge_energy)
            actual_energy = max(actual_energy, 0.0)
            soc_delta = actual_energy * self.eff_sqrt / self.capacity_mwh
            self.soc += soc_delta
            actual_energy = -actual_energy  # I keep sign convention

        # I update cycling counters
        energy_mwh = abs(actual_energy)
        self.daily_energy_throughput += energy_mwh
        cycle_fraction = energy_mwh / (self.capacity_mwh * (self.max_soc - self.min_soc))
        self.total_cycles += cycle_fraction
        self.daily_cycles += cycle_fraction

        # I clamp SoC to valid range
        self.soc = np.clip(self.soc, self.min_soc, self.max_soc)

        return actual_energy

    def remaining_capacity(self, after_market: str, isp: int = 0,
                           hour: int = 0) -> Tuple[float, float]:
        """
        I calculate remaining discharge/charge capacity after upstream markets.

        Args:
            after_market: 'dam', 'afrr', 'intraday', or 'mfrr'
            isp: Current ISP index (0-95) for sub-hourly markets
            hour: Current hour (0-23) for hourly markets

        Returns:
            (max_discharge_mw, max_charge_mw) available for this market
        """
        used_power = 0.0

        if after_market in ('afrr', 'intraday', 'mfrr'):
            # I subtract DAM commitment (absolute value, it uses power)
            dam_mw = abs(self.dam_schedule[hour]) if hour < 24 else 0.0
            used_power += dam_mw

        if after_market in ('intraday', 'mfrr'):
            # I subtract aFRR capacity reservation
            block_idx = hour // 4 if hour < 24 else 0
            afrr_mw = abs(self.afrr_commitments[block_idx]) if block_idx < 6 else 0.0
            used_power += afrr_mw

        if after_market == 'mfrr':
            # I subtract IntraDay position
            id_mw = abs(self.intraday_positions[isp]) if isp < 96 else 0.0
            used_power += id_mw

        remaining_power = max(0.0, self.max_power_mw - used_power)

        # I also check SoC-based limits
        available_discharge_energy = (self.soc - self.min_soc) * self.capacity_mwh
        available_charge_energy = (self.max_soc - self.soc) * self.capacity_mwh

        # I convert energy to power (for the shortest ISP = 15 min)
        time_step = 0.25  # 15-min ISP
        max_discharge = min(
            remaining_power,
            available_discharge_energy * self.eff_sqrt / time_step
        )
        max_charge = min(
            remaining_power,
            available_charge_energy / (self.eff_sqrt * time_step)
        )

        return max(0.0, max_discharge), max(0.0, max_charge)

    def simulate_soc_trajectory(
        self, schedule: np.ndarray, initial_soc: float,
        duration_h_per_step: float = 1.0
    ) -> Tuple[np.ndarray, bool]:
        """
        I simulate the SoC trajectory for a given schedule without mutating state.

        Args:
            schedule: Array of MW values per step (+ = discharge, - = charge)
            initial_soc: Starting SoC
            duration_h_per_step: Duration of each step in hours

        Returns:
            (trajectory, is_feasible): SoC trajectory and feasibility flag
        """
        trajectory = np.zeros(len(schedule) + 1, dtype=np.float32)
        trajectory[0] = initial_soc
        soc = initial_soc
        is_feasible = True

        for i, power_mw in enumerate(schedule):
            if abs(power_mw) < 0.01:
                trajectory[i + 1] = soc
                continue

            if power_mw > 0:  # discharge
                energy = power_mw * duration_h_per_step
                soc_delta = energy / (self.eff_sqrt * self.capacity_mwh)
                soc -= soc_delta
            else:  # charge
                energy = abs(power_mw) * duration_h_per_step
                soc_delta = energy * self.eff_sqrt / self.capacity_mwh
                soc += soc_delta

            if soc < self.min_soc - 0.001 or soc > self.max_soc + 0.001:
                is_feasible = False

            soc = np.clip(soc, self.min_soc, self.max_soc)
            trajectory[i + 1] = soc

        return trajectory, is_feasible

    def check_day_boundary(self, hour: int, prev_hour: int) -> None:
        """I reset daily counters when crossing midnight."""
        if hour < prev_hour:  # midnight crossing
            self.daily_cycles = 0.0
            self.daily_energy_throughput = 0.0

    def clone(self) -> 'SharedBatteryState':
        """I create a deep copy for forward simulation without mutating state."""
        new_state = SharedBatteryState(
            soc=self.soc,
            capacity_mwh=self.capacity_mwh,
            max_power_mw=self.max_power_mw,
            efficiency=self.efficiency,
            min_soc=self.min_soc,
            max_soc=self.max_soc,
            total_cycles=self.total_cycles,
            daily_cycles=self.daily_cycles,
            max_daily_cycles=self.max_daily_cycles,
            daily_energy_throughput=self.daily_energy_throughput,
            dam_schedule=self.dam_schedule.copy(),
            afrr_commitments=self.afrr_commitments.copy(),
            afrr_price_tiers=self.afrr_price_tiers.copy(),
            intraday_positions=self.intraday_positions.copy(),
            afrr_is_selected=self.afrr_is_selected,
            afrr_is_activated=self.afrr_is_activated,
            afrr_activation_direction=self.afrr_activation_direction,
            afrr_activated_mw=self.afrr_activated_mw,
        )
        return new_state

    def reset(self, soc: Optional[float] = None) -> None:
        """I reset state for a new episode."""
        self.soc = soc if soc is not None else 0.5
        self.total_cycles = 0.0
        self.daily_cycles = 0.0
        self.daily_energy_throughput = 0.0
        self.dam_schedule = np.zeros(24, dtype=np.float32)
        self.afrr_commitments = np.zeros(6, dtype=np.float32)
        self.afrr_price_tiers = np.ones(6, dtype=np.int32) * 2
        self.intraday_positions = np.zeros(96, dtype=np.float32)
        self.afrr_is_selected = False
        self.afrr_is_activated = False
        self.afrr_activation_direction = 'none'
        self.afrr_activated_mw = 0.0


class UpstreamSimulator:
    """
    I provide upstream market decisions for downstream model training.

    During training of downstream models (e.g., aFRR), I need DAM decisions.
    I can use either:
    1. Frozen trained models (for sequential training pipeline)
    2. Data column fallbacks (dam_commitment from CSV)
    3. Heuristic defaults (for initial bootstrapping)
    """

    # I define aFRR price tiers matching the aFRR env
    AFRR_PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])

    def __init__(
        self,
        dam_model_path: Optional[str] = None,
        afrr_model_path: Optional[str] = None,
        intraday_model_path: Optional[str] = None,
        use_data_fallback: bool = True,
    ):
        """
        I initialize with optional frozen model paths.

        Args:
            dam_model_path: Path to frozen DAM model .zip
            afrr_model_path: Path to frozen aFRR model .zip
            intraday_model_path: Path to frozen IntraDay model .zip
            use_data_fallback: If True, use CSV data columns when models unavailable
        """
        self.use_data_fallback = use_data_fallback
        self.dam_model = None
        self.afrr_model = None
        self.intraday_model = None

        # I load frozen models if paths provided
        if dam_model_path:
            self.dam_model = self._load_model(dam_model_path)
        if afrr_model_path:
            self.afrr_model = self._load_model(afrr_model_path)
        if intraday_model_path:
            self.intraday_model = self._load_model(intraday_model_path)

    def _load_model(self, model_path: str):
        """I load a frozen MaskablePPO model for inference."""
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path)
            return model
        except Exception as e:
            print(f"  Warning: Could not load model {model_path}: {e}")
            return None

    def get_dam_schedule(
        self, df: pd.DataFrame, day_start_idx: int, sph: int = 4
    ) -> np.ndarray:
        """
        I provide a 24-hour DAM schedule for the given day.

        Falls back to dam_commitment column from CSV if no model available.

        Args:
            df: Training DataFrame
            day_start_idx: Index of first row of the day
            sph: Steps per hour (4 for 15-min, 1 for 1-hour)

        Returns:
            24-element array of hourly MW commitments
        """
        schedule = np.zeros(24, dtype=np.float32)

        if self.dam_model is not None:
            # I would run the frozen DAM model here
            # For now, this is a placeholder - actual implementation in train_decomposed.py
            pass
        elif self.use_data_fallback and 'dam_commitment' in df.columns:
            # I extract hourly DAM commitments from CSV
            for h in range(24):
                row_idx = day_start_idx + h * sph
                if row_idx < len(df):
                    schedule[h] = np.clip(
                        df.iloc[row_idx].get('dam_commitment', 0.0),
                        -30.0, 30.0
                    )
        else:
            # I use zero schedule as safe default
            pass

        return schedule

    def get_afrr_commitment(
        self, state: SharedBatteryState, df: pd.DataFrame,
        block_start_idx: int, block_idx: int
    ) -> Tuple[float, int]:
        """
        I provide aFRR capacity commitment for a 4h block.

        Args:
            state: Current battery state
            df: Training DataFrame
            block_start_idx: Index of first ISP in this block
            block_idx: Block index (0-5)

        Returns:
            (capacity_mw, price_tier_idx)
        """
        if self.afrr_model is not None:
            # I would run frozen aFRR model here
            pass

        # I use heuristic: 50% of remaining capacity, neutral price
        hour = block_idx * 4
        remaining_discharge, remaining_charge = state.remaining_capacity(
            'afrr', hour=hour
        )
        capacity_mw = min(remaining_discharge, remaining_charge) * 0.5
        price_tier = 2  # neutral (1.0x multiplier)

        return capacity_mw, price_tier

    def get_intraday_position(
        self, state: SharedBatteryState, df: pd.DataFrame,
        isp_idx: int
    ) -> float:
        """
        I provide IntraDay position for a given ISP.

        Args:
            state: Current battery state
            df: Training DataFrame
            isp_idx: ISP index (0-95)

        Returns:
            MW position (+ = sell, - = buy)
        """
        if self.intraday_model is not None:
            # I would run frozen IntraDay model here
            pass

        # I default to idle (no IntraDay trading)
        return 0.0

    def simulate_afrr_activation(
        self, commitment_mw: float, df: pd.DataFrame,
        block_start_idx: int, sph: int = 4,
        price_tier: int = 2
    ) -> Tuple[bool, str, float]:
        """
        I simulate stochastic aFRR activation within a 4h block.

        Args:
            commitment_mw: Committed capacity in MW
            df: Training DataFrame
            block_start_idx: Index of first ISP in this block
            price_tier: Price tier index (0-4)
            sph: Steps per hour

        Returns:
            (was_selected, direction, activated_mw)
        """
        if commitment_mw < 0.1:
            return False, 'none', 0.0

        # I calculate selection probability based on price tier
        price_mult = self.AFRR_PRICE_TIERS[price_tier]
        base_selection_prob = 0.80
        # I adjust: aggressive bid (low mult) = higher selection
        selection_prob = base_selection_prob * (1.3 - price_mult) / 0.6
        selection_prob = np.clip(selection_prob, 0.3, 0.95)

        was_selected = np.random.random() < selection_prob
        if not was_selected:
            return False, 'none', 0.0

        # I determine activation based on system imbalance in block
        block_isps = 4 * sph  # 16 ISPs per 4h block at 15-min resolution
        imbalance_sum = 0.0
        count = 0
        for i in range(block_isps):
            idx = block_start_idx + i
            if idx < len(df):
                imbalance_sum += df.iloc[idx].get('net_imbalance_mw', 0.0)
                count += 1

        avg_imbalance = imbalance_sum / max(count, 1)

        # I calculate activation probability
        activation_prob = 0.15
        if abs(avg_imbalance) > 100:
            activation_prob += 0.15
        if abs(avg_imbalance) > 500:
            activation_prob += 0.10

        activated = np.random.random() < activation_prob

        if not activated:
            return True, 'none', 0.0  # selected but not activated

        # I determine direction
        if avg_imbalance > 0:
            direction = 'up'  # discharge
        elif avg_imbalance < 0:
            direction = 'down'  # charge
        else:
            direction = 'up' if np.random.random() > 0.5 else 'down'

        return True, direction, commitment_mw
