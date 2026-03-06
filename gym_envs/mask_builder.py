"""
Action Mask Builder for Unified Battery Environment

I build hierarchical action masks for MaskablePPO's MultiDiscrete action space.
The mask ensures the agent never selects physically impossible actions.

Mask dimensions:
  Legacy: 5 + 5 + 11 + 11 = 32 bits
  Full market: 5 + 5 + 11 + 11 + 11 + 5 = 48 bits
"""

import numpy as np
from typing import TYPE_CHECKING

from gym_envs.capacity_manager import CapacityManager, CapacityLimits

if TYPE_CHECKING:
    from gym_envs.battery_env_unified import BatteryEnvUnified


class MaskBuilder:
    """I build action masks using capacity limits and market state.

    I extract the action_masks() method from BatteryEnvUnified so the
    mask logic can be tested independently and reused.
    """

    def __init__(self, env: 'BatteryEnvUnified'):
        """I hold a reference to the env for state access.

        I don't copy state — I read it live from the env so the mask
        always reflects the current step's constraints.
        """
        self._env = env

    def build(self) -> np.ndarray:
        """I return hierarchical action masks for the MultiDiscrete space.

        Returns a flattened mask for sb3_contrib.MaskablePPO compatibility.
        """
        env = self._env
        row = env.df.iloc[env.current_step]

        # I get current DAM commitment
        dam_commitment = env._get_dam_commitment(env.current_step)
        dam_commitment = np.clip(dam_commitment, -env.max_power_mw, env.max_power_mw)

        # I compute capacity limits using the CapacityManager
        dam_min_soc, dam_max_soc = env._calculate_dam_soc_reserve()
        limits = env._capacity_manager.compute_limits(
            soc=env.soc,
            dam_commitment=dam_commitment,
            afrr_commitment_mw=env.afrr_commitment_mw,
            daily_cycles=env.daily_cycles,
            dam_min_soc=dam_min_soc,
            dam_max_soc=dam_max_soc,
        )

        # Mask 1: aFRR commitment levels (5 options)
        afrr_mask = np.ones(len(env.AFRR_LEVELS), dtype=bool)
        for i, level in enumerate(env.AFRR_LEVELS):
            required_capacity = level * limits.remaining_after_dam
            if required_capacity > limits.max_discharge_mw or required_capacity > limits.max_charge_mw:
                afrr_mask[i] = False
        afrr_mask[0] = True  # I always allow zero commitment

        # Mask 2: aFRR price tiers (5 options) — no physical constraints
        price_mask = np.ones(len(env.AFRR_PRICE_TIERS), dtype=bool)

        # Mask 3: IntraDay actions (11 options)
        intraday_mask = np.zeros(env.N_INTRADAY_ACTIONS, dtype=bool)
        intraday_open = env._is_intraday_open(row)
        if intraday_open:
            for i, level in enumerate(env.INTRADAY_LEVELS):
                power_mw = level * limits.remaining_for_trading
                if level >= 0:
                    if power_mw <= limits.adjusted_max_discharge + 0.01:
                        intraday_mask[i] = True
                else:
                    if abs(power_mw) <= limits.adjusted_max_charge + 0.01:
                        intraday_mask[i] = True
        intraday_mask[env.N_INTRADAY_ACTIONS // 2] = True  # I always allow idle

        # Mask 4: mFRR actions (11 options)
        mfrr_mask = np.zeros(env.N_MFRR_ACTIONS, dtype=bool)
        mfrr_activated_up = False
        mfrr_activated_down = False

        has_activation_data = ('mfrr_activated_up_mwh' in row.index
                               if hasattr(row, 'index') else
                               'mfrr_activated_up_mwh' in row)
        if has_activation_data:
            mfrr_activated_up = row.get('mfrr_activated_up_mwh', 0.0) > 0
            mfrr_activated_down = row.get('mfrr_activated_down_mwh', 0.0) > 0

            if mfrr_activated_up:
                for i, level in enumerate(env.MFRR_LEVELS):
                    if level > 0:
                        power_mw = level * limits.remaining_for_trading
                        if power_mw <= limits.adjusted_max_discharge + 0.01:
                            mfrr_mask[i] = True

            if mfrr_activated_down:
                for i, level in enumerate(env.MFRR_LEVELS):
                    if level < 0:
                        power_mw = abs(level) * limits.remaining_for_trading
                        if power_mw <= limits.adjusted_max_charge + 0.01:
                            mfrr_mask[i] = True
        else:
            imbalance = row.get('net_imbalance_mw_lag_1h',
                                row.get('system_deviation_mwh_lag_1h',
                                        row.get('net_imbalance_mw', 0.0)))
            mfrr_up_price = row.get('mfrr_price_up_lag_1h',
                                    row.get('mfrr_price_up', 0.0))
            mfrr_down_price = row.get('mfrr_price_down_lag_1h',
                                      row.get('mfrr_price_down', 0.0))

            if imbalance < -env.mfrr_imbalance_threshold and mfrr_up_price > 1.0:
                for i, level in enumerate(env.MFRR_LEVELS):
                    if level > 0:
                        power_mw = level * limits.remaining_for_trading
                        if power_mw <= limits.adjusted_max_discharge + 0.01:
                            mfrr_mask[i] = True
            elif imbalance > env.mfrr_imbalance_threshold and mfrr_down_price > 1.0:
                for i, level in enumerate(env.MFRR_LEVELS):
                    if level < 0:
                        power_mw = abs(level) * limits.remaining_for_trading
                        if power_mw <= limits.adjusted_max_charge + 0.01:
                            mfrr_mask[i] = True

        mfrr_mask[env.N_MFRR_ACTIONS // 2] = True  # I always allow idle

        # I concatenate masks for MultiDiscrete format
        if env.enable_full_market:
            # Mask 5: Free Bid qty (11 options)
            freebid_qty_mask = np.zeros(env.N_FREEBID_QTY_ACTIONS, dtype=bool)

            if has_activation_data:
                if mfrr_activated_up:
                    for i, level in enumerate(env.FREEBID_QTY_LEVELS):
                        if level > 0:
                            power_mw = level * limits.remaining_for_trading
                            if power_mw <= limits.adjusted_max_discharge + 0.01:
                                freebid_qty_mask[i] = True
                if mfrr_activated_down:
                    for i, level in enumerate(env.FREEBID_QTY_LEVELS):
                        if level < 0:
                            power_mw = abs(level) * limits.remaining_for_trading
                            if power_mw <= limits.adjusted_max_charge + 0.01:
                                freebid_qty_mask[i] = True
            else:
                if imbalance < -env.mfrr_imbalance_threshold and mfrr_up_price > 1.0:
                    for i, level in enumerate(env.FREEBID_QTY_LEVELS):
                        if level > 0:
                            power_mw = level * limits.remaining_for_trading
                            if power_mw <= limits.adjusted_max_discharge + 0.01:
                                freebid_qty_mask[i] = True
                elif imbalance > env.mfrr_imbalance_threshold and mfrr_down_price > 1.0:
                    for i, level in enumerate(env.FREEBID_QTY_LEVELS):
                        if level < 0:
                            power_mw = abs(level) * limits.remaining_for_trading
                            if power_mw <= limits.adjusted_max_charge + 0.01:
                                freebid_qty_mask[i] = True

            freebid_qty_mask[env.N_FREEBID_QTY_ACTIONS // 2] = True

            # Mask 6: Free Bid price tier (5 options)
            price_tier_mask = np.ones(env.N_FREEBID_PRICE_TIERS, dtype=bool)
            freebid_has_direction = any(
                freebid_qty_mask[i] for i in range(env.N_FREEBID_QTY_ACTIONS)
                if i != env.N_FREEBID_QTY_ACTIONS // 2
            )
            if not freebid_has_direction:
                price_tier_mask = np.zeros(env.N_FREEBID_PRICE_TIERS, dtype=bool)
                price_tier_mask[2] = True

            full_mask = np.concatenate([
                afrr_mask, price_mask, intraday_mask, mfrr_mask,
                freebid_qty_mask, price_tier_mask
            ])
        else:
            full_mask = np.concatenate([afrr_mask, price_mask, intraday_mask, mfrr_mask])

        return full_mask
