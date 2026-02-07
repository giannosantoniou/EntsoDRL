"""
Battery Simulator for Backtesting

I simulate a battery energy storage system with realistic constraints:
- State of Charge (SoC) limits
- Charge/discharge efficiency
- Power limits
- Degradation costs
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class BatteryConfig:
    """Battery configuration parameters."""
    capacity_mwh: float = 146.0      # Total energy capacity
    max_power_mw: float = 30.0       # Max charge/discharge rate
    efficiency: float = 0.94         # Round-trip efficiency (sqrt for one-way)
    soc_min: float = 0.05            # Minimum SoC (5%)
    soc_max: float = 0.95            # Maximum SoC (95%)
    degradation_cost: float = 15.0   # EUR/MWh cycled (battery wear)
    initial_soc: float = 0.50        # Starting SoC


class Battery:
    """
    I simulate battery operations with realistic constraints.

    Positive power = discharge (sell to grid)
    Negative power = charge (buy from grid)
    """

    def __init__(self, config: BatteryConfig = None):
        self.config = config or BatteryConfig()
        self.soc = self.config.initial_soc
        self.total_energy_cycled = 0.0
        self.total_degradation_cost = 0.0

        # One-way efficiency
        self.charge_eff = np.sqrt(self.config.efficiency)
        self.discharge_eff = np.sqrt(self.config.efficiency)

        # History tracking
        self.history = {
            'soc': [],
            'power': [],
            'energy_cycled': [],
        }

    @property
    def energy_mwh(self) -> float:
        """Current stored energy in MWh."""
        return self.soc * self.config.capacity_mwh

    @property
    def available_discharge_mw(self) -> float:
        """Maximum power available for discharge."""
        # Energy available above min SoC
        available_energy = (self.soc - self.config.soc_min) * self.config.capacity_mwh
        # Convert to power (1 hour assumed)
        max_from_soc = available_energy * self.discharge_eff
        return min(self.config.max_power_mw, max_from_soc)

    @property
    def available_charge_mw(self) -> float:
        """Maximum power available for charging."""
        # Energy space below max SoC
        available_space = (self.config.soc_max - self.soc) * self.config.capacity_mwh
        # Convert to power (1 hour assumed)
        max_from_soc = available_space / self.charge_eff
        return min(self.config.max_power_mw, max_from_soc)

    def step(self, power_mw: float, duration_hours: float = 1.0) -> Tuple[float, float]:
        """
        Execute a charge/discharge action.

        Args:
            power_mw: Power in MW (positive=discharge, negative=charge)
            duration_hours: Duration of the action

        Returns:
            Tuple of (actual_power_mw, degradation_cost_eur)
        """
        # Clip power to available limits
        if power_mw > 0:  # Discharge
            actual_power = min(power_mw, self.available_discharge_mw)
            energy_from_battery = actual_power * duration_hours / self.discharge_eff
            self.soc -= energy_from_battery / self.config.capacity_mwh
        elif power_mw < 0:  # Charge
            actual_power = max(power_mw, -self.available_charge_mw)
            energy_to_battery = abs(actual_power) * duration_hours * self.charge_eff
            self.soc += energy_to_battery / self.config.capacity_mwh
        else:
            actual_power = 0.0

        # Calculate degradation cost
        energy_cycled = abs(actual_power) * duration_hours
        degradation_cost = energy_cycled * self.config.degradation_cost

        # Update totals
        self.total_energy_cycled += energy_cycled
        self.total_degradation_cost += degradation_cost

        # Record history
        self.history['soc'].append(self.soc)
        self.history['power'].append(actual_power)
        self.history['energy_cycled'].append(energy_cycled)

        return actual_power, degradation_cost

    def reset(self):
        """Reset battery to initial state."""
        self.soc = self.config.initial_soc
        self.total_energy_cycled = 0.0
        self.total_degradation_cost = 0.0
        self.history = {'soc': [], 'power': [], 'energy_cycled': []}

    def get_action_mask(self) -> Tuple[bool, bool]:
        """
        Return which actions are available.

        Returns:
            Tuple of (can_discharge, can_charge)
        """
        can_discharge = self.available_discharge_mw > 0.1  # Small threshold
        can_charge = self.available_charge_mw > 0.1
        return can_discharge, can_charge
