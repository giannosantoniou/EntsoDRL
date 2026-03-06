"""
Capacity Manager for Unified Battery Environment

I centralize the capacity cascade logic that determines how much power
is available for each market after prior commitments are deducted.

The cascade order is: DAM → aFRR → IDA → mFRR → XBID → FreeBid
Each market can only use capacity not consumed by prior markets.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CapacityLimits:
    """I hold the computed capacity limits for the current step.

    These are used by both action_masks() and step() to enforce
    consistent physical constraints.
    """
    # I track capacity at each cascade level
    remaining_after_dam: float       # max_power_mw - |dam_commitment|
    remaining_for_trading: float     # remaining_after_dam - afrr_commitment_mw

    # I track SoC-based limits (adjusted for DAM reserves)
    max_discharge_mw: float         # min(capacity, SoC-based, cycle-based)
    max_charge_mw: float            # min(capacity, SoC-based, cycle-based)
    adjusted_max_discharge: float   # after aFRR reservation
    adjusted_max_charge: float      # after aFRR reservation

    # I track underlying energy
    available_discharge_mwh: float  # (soc - min_soc) * capacity
    available_charge_mwh: float     # (max_soc - soc) * capacity


class CapacityManager:
    """I compute available capacity for each market in the cascade.

    I extract the duplicated capacity computation from action_masks() and
    step() so both use consistent limits. The cascade order ensures that
    mandatory markets (DAM, aFRR) get priority over voluntary markets.
    """

    def __init__(
        self,
        capacity_mwh: float,
        max_power_mw: float,
        efficiency: float,
        eff_sqrt: float,
        min_soc: float,
        max_soc: float,
        time_step_hours: float,
        max_daily_cycles: float,
    ):
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.eff_sqrt = eff_sqrt
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.time_step_hours = time_step_hours
        self.max_daily_cycles = max_daily_cycles

    def compute_limits(
        self,
        soc: float,
        dam_commitment: float,
        afrr_commitment_mw: float,
        daily_cycles: float,
        dam_min_soc: float,
        dam_max_soc: float,
    ) -> CapacityLimits:
        """I compute the full capacity cascade limits for a step.

        Args:
            soc: Current state of charge (0-1)
            dam_commitment: DAM commitment MW (positive=sell, negative=buy)
            afrr_commitment_mw: Current aFRR capacity commitment MW
            daily_cycles: Cycles used today
            dam_min_soc: Minimum SoC after DAM reserve (from _calculate_dam_soc_reserve)
            dam_max_soc: Maximum SoC after DAM reserve
        """
        # I calculate remaining capacity after DAM
        remaining_after_dam = self.max_power_mw - abs(dam_commitment)

        # I calculate SoC-based limits using DAM-aware reserves
        available_discharge_mwh = max(0.0, (soc - dam_min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (dam_max_soc - soc) * self.capacity_mwh)

        max_discharge_mw = min(
            remaining_after_dam,
            available_discharge_mwh * self.eff_sqrt / self.time_step_hours
        )
        max_charge_mw = min(
            remaining_after_dam,
            available_charge_mwh / self.eff_sqrt / self.time_step_hours
        )

        # I apply daily cycle limits, pre-subtracting DAM cycle cost and aFRR buffer
        dam_cycle_cost = abs(dam_commitment) * self.time_step_hours / self.capacity_mwh
        afrr_buffer = afrr_commitment_mw * self.time_step_hours / self.capacity_mwh * 0.15
        cycles_remaining = max(0, self.max_daily_cycles - daily_cycles
                               - dam_cycle_cost - afrr_buffer)
        max_cycle_power = cycles_remaining * self.capacity_mwh / self.time_step_hours
        max_discharge_mw = min(max_discharge_mw, max_cycle_power)
        max_charge_mw = min(max_charge_mw, max_cycle_power)

        # I subtract aFRR commitment to prevent cross-market over-commitment
        remaining_for_trading = max(0, remaining_after_dam - afrr_commitment_mw)

        # I also reduce SoC-based limits to reserve energy for aFRR activation
        afrr_energy_reserve = afrr_commitment_mw * self.time_step_hours
        adjusted_max_discharge = min(
            remaining_for_trading,
            max(0, available_discharge_mwh - afrr_energy_reserve / self.eff_sqrt)
            * self.eff_sqrt / self.time_step_hours
        )
        adjusted_max_charge = min(
            remaining_for_trading,
            max(0, available_charge_mwh - afrr_energy_reserve * self.eff_sqrt)
            / self.eff_sqrt / self.time_step_hours
        )

        return CapacityLimits(
            remaining_after_dam=remaining_after_dam,
            remaining_for_trading=remaining_for_trading,
            max_discharge_mw=max_discharge_mw,
            max_charge_mw=max_charge_mw,
            adjusted_max_discharge=adjusted_max_discharge,
            adjusted_max_charge=adjusted_max_charge,
            available_discharge_mwh=available_discharge_mwh,
            available_charge_mwh=available_charge_mwh,
        )
