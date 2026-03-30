"""
SAC/TD3 Continuous Environment for Multi-Market Battery Trading (v3 — Direct Execution)

I execute market actions in ABSOLUTE MW directly, bypassing the inner env's
MultiDiscrete action space. This eliminates rounding noise and cascade dependencies.

Architecture:
  - I use BatteryEnvUnified for data prep, observation building, and state tracking
  - I BYPASS its step() — instead, I compute SoC/revenue directly from absolute MW
  - ObservationBuilder works unchanged (reads my synced state from inner env)
  - UnifiedRewardCalculator works unchanged (receives my computed results)

Action Space: Box(6,) continuous [-1, +1]
  [0] aFRR Commitment:  0..max_power MW
  [1] aFRR Price Tier:  0.7..1.3 multiplier
  [2] XBID Quantity:    -max_power..+max_power MW (neg=buy)
  [3] XBID Price Offset: -10..+10 EUR/MWh from mid-price
  [4] mFRR Quantity:    -max_power..+max_power MW
  [5] FreeBid Quantity: -max_power..+max_power MW
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from gym_envs.battery_env_unified import BatteryEnvUnified


class BatteryEnvUnifiedSAC(gym.Env):
    """I provide true continuous action execution for SAC/TD3."""

    metadata = {'render_modes': ['human']}

    DEFAULT_PENALTIES = {
        'soc_soft_margin': 60.0,
        'cycle_excess': 2000.0,
    }
    CYCLE_TARGET = 5.0
    XBID_PRICE_OFFSET_MAX = 50.0  # EUR/MWh (±50 for volatile intraday near gate closure)
    N_ACTIONS = 7

    # Fix 1: Network charges (Use-of-System fees)
    # I apply ~4 EUR/MWh per direction (charge and discharge)
    # This reflects real ADMIE/HEDNO transmission + distribution charges
    NETWORK_CHARGE_EUR_PER_MWH = 4.0  # EUR/MWh per direction

    # Fix 3: Minimum bid sizes (HEnEx rules)
    MIN_BID_DAM_MW = 0.1     # DAM minimum bid
    MIN_BID_BALANCING_MW = 1.0  # aFRR/mFRR minimum bid

    # Fix 5: Prequalification flag
    # I assume prequalified by default but can be disabled
    is_prequalified_afrr = True
    is_prequalified_mfrr = True

    # Fix 8: Calendar aging cost (EUR per step, depends on SoC)
    # I model that holding battery at extreme SoC accelerates degradation
    # LFP optimal storage: 30-70% SoC. Penalty outside this range.
    # Battery ~22M EUR / 20 years = ~3,000 EUR/day total aging budget.
    # Calendar aging = ~30% of total = ~900 EUR/day = ~9.4 EUR/step (at 96 steps/day)
    CALENDAR_AGING_BASE_EUR_PER_STEP = 0.3  # ~29 EUR/day baseline (minimal)
    CALENDAR_AGING_SOC_PENALTY_FACTOR = 0.001  # extra cost per (SoC%-50)^2
    # At SoC 95%: 0.3 + 0.001*2025 = 2.33 EUR/step = 224 EUR/day (reasonable)

    # Fix 12: Availability contract revenue
    # I model the fixed availability payment from EU-funded scheme
    # ~115,000 EUR/MW/year = ~315 EUR/MW/day = ~13 EUR/MW/hour
    # For 30 MW: ~394 EUR/hour = ~99 EUR per 15-min step
    AVAILABILITY_CONTRACT_EUR_PER_MW_PER_HOUR = 13.15  # EUR/MW/h

    def __init__(
        self,
        inner_env: Optional[BatteryEnvUnified] = None,
        penalties: Optional[Dict[str, float]] = None,
        network_charge_eur_per_mwh: float = 4.0,
        enable_availability_contract: bool = True,
        availability_eur_per_mw_hour: float = 13.15,
        prequalified_afrr: bool = True,
        prequalified_mfrr: bool = True,
        **kwargs
    ):
        super().__init__()

        # I store the inner env for data/observations/state tracking
        if inner_env is not None:
            self._inner = inner_env
        else:
            self._inner = BatteryEnvUnified(**kwargs)

        self.penalties = dict(self.DEFAULT_PENALTIES)
        if penalties:
            self.penalties.update(penalties)

        # Fix 1: Network charges
        self.NETWORK_CHARGE_EUR_PER_MWH = network_charge_eur_per_mwh

        # Fix 5: Prequalification
        self.is_prequalified_afrr = prequalified_afrr
        self.is_prequalified_mfrr = prequalified_mfrr

        # Fix 12: Availability contract
        self._enable_availability_contract = enable_availability_contract
        self.AVAILABILITY_CONTRACT_EUR_PER_MW_PER_HOUR = availability_eur_per_mw_hour

        # I cache physics constants from inner env
        self._max_power = self._inner.max_power_mw
        self._capacity = self._inner.capacity_mwh
        self._eff_sqrt = self._inner.eff_sqrt
        self._min_soc = self._inner.min_soc
        self._max_soc = self._inner.max_soc
        self._time_step = self._inner.time_step_hours
        self._is_full_market = self._inner.enable_full_market

        # I validate episode_length matches time_step_hours
        ep_len = self._inner.episode_length
        if ep_len is not None:
            ep_hours = ep_len * self._time_step
            ep_days = ep_hours / 24.0
            if ep_days < 1.0 or ep_days > 60.0:
                import warnings
                warnings.warn(
                    f"Episode length {ep_len} steps = {ep_hours:.0f}h = {ep_days:.1f} days. "
                    f"Expected 1-60 days. Check episode_length vs time_step_hours={self._time_step}."
                )

        # I define the continuous action space (always 6 dims)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.N_ACTIONS,), dtype=np.float32
        )
        self.observation_space = self._inner.observation_space

    # ─── Properties for monitoring ─────────────────────────────────

    @property
    def reward_calculator(self):
        return self._inner.reward_calculator

    @reward_calculator.setter
    def reward_calculator(self, value):
        self._inner.reward_calculator = value

    @property
    def soc(self):
        return self._inner.soc

    @property
    def daily_cycles(self):
        return self._inner.daily_cycles

    # ─── Action decoding ──────────────────────────────────────────

    def _decode_action(self, action: np.ndarray) -> dict:
        """I convert continuous [-1,+1] to absolute MW + price offset."""
        a = np.clip(action, -1.0, 1.0)
        return {
            'afrr_mw': (a[0] + 1.0) / 2.0 * self._max_power,
            'afrr_price_tier': 0.7 + (a[1] + 1.0) / 2.0 * 0.6,
            'xbid_mw': a[2] * self._max_power,
            'xbid_price_offset': a[3] * self.XBID_PRICE_OFFSET_MAX,
            'mfrr_mw': a[4] * self._max_power,
            'freebid_mw': a[5] * self._max_power if self._is_full_market else 0.0,
            'dam_aggressiveness': a[6],  # -1 (conservative) to +1 (aggressive)
        }

    def _enforce_constraints(self, d: dict) -> dict:
        """I enforce capacity + SoC limits via proportional scaling."""
        # I get DAM commitment (mandatory, pre-set)
        dam_mw = abs(self._inner._get_dam_commitment(self._inner.current_step))
        available = max(0.0, self._max_power - dam_mw)

        # I scale down if total > available
        total = abs(d['afrr_mw']) + abs(d['xbid_mw']) + abs(d['mfrr_mw']) + abs(d['freebid_mw'])
        if total > available and total > 0.1:
            scale = available / total
            d['afrr_mw'] *= scale
            d['xbid_mw'] *= scale
            d['mfrr_mw'] *= scale
            d['freebid_mw'] *= scale

        # I enforce SoC limits
        soc = self._inner.soc
        max_discharge_mw = min(
            self._max_power,
            (soc - self._min_soc) * self._capacity * self._eff_sqrt / self._time_step
        )
        max_charge_mw = min(
            self._max_power,
            (self._max_soc - soc) * self._capacity / self._eff_sqrt / self._time_step
        )

        # I compute total discharge and charge across all markets
        total_discharge = max(0, d['xbid_mw']) + max(0, d['mfrr_mw']) + max(0, d['freebid_mw'])
        total_charge = abs(min(0, d['xbid_mw'])) + abs(min(0, d['mfrr_mw'])) + abs(min(0, d['freebid_mw']))

        # I add DAM contribution
        total_discharge += max(0, dam_mw)  # DAM sell = discharge
        total_charge += abs(min(0, dam_mw))  # DAM buy = charge (negative commitment)

        if total_discharge > max_discharge_mw and total_discharge > 0.1:
            # I only scale agent-controlled markets (not DAM)
            agent_discharge = total_discharge - max(0, dam_mw)
            if agent_discharge > 0.1:
                scale = max(0, max_discharge_mw - max(0, dam_mw)) / agent_discharge
                scale = max(0, min(1, scale))
                if d['xbid_mw'] > 0: d['xbid_mw'] *= scale
                if d['mfrr_mw'] > 0: d['mfrr_mw'] *= scale
                if d['freebid_mw'] > 0: d['freebid_mw'] *= scale

        if total_charge > max_charge_mw and total_charge > 0.1:
            agent_charge = total_charge - abs(min(0, dam_mw))
            if agent_charge > 0.1:
                scale = max(0, max_charge_mw - abs(min(0, dam_mw))) / agent_charge
                scale = max(0, min(1, scale))
                if d['xbid_mw'] < 0: d['xbid_mw'] *= scale
                if d['mfrr_mw'] < 0: d['mfrr_mw'] *= scale
                if d['freebid_mw'] < 0: d['freebid_mw'] *= scale

        # Fix 3: I enforce minimum bid sizes (HEnEx rules)
        # aFRR/mFRR: minimum 1 MW, DAM: minimum 0.1 MW
        if abs(d['afrr_mw']) < self.MIN_BID_BALANCING_MW:
            d['afrr_mw'] = 0.0
        if abs(d['xbid_mw']) < self.MIN_BID_DAM_MW:
            d['xbid_mw'] = 0.0
        if abs(d['mfrr_mw']) < self.MIN_BID_BALANCING_MW:
            d['mfrr_mw'] = 0.0
        if abs(d['freebid_mw']) < self.MIN_BID_BALANCING_MW:
            d['freebid_mw'] = 0.0

        return d

    # ─── SoC update primitives ────────────────────────────────────

    def _execute_energy(self, power_mw: float) -> Tuple[float, float]:
        """I update SoC for given power and return (signed_energy, soc_delta).

        power_mw > 0: discharge (sell), SoC decreases
        power_mw < 0: charge (buy), SoC increases

        Returns SIGNED actual_energy in MWh:
          positive = energy DELIVERED to grid (sell → revenue)
          negative = energy TAKEN from grid (buy → cost)

        CRITICAL: Revenue = signed_energy * price gives correct sign:
          sell 10MW at 100 EUR: +2.42 MWh * 100 = +242 EUR (income)
          buy 10MW at 100 EUR:  -2.58 MWh * 100 = -258 EUR (cost)
        """
        if abs(power_mw) < 0.1:
            return 0.0, 0.0

        env = self._inner
        old_soc = env.soc

        if power_mw > 0:  # Discharge (sell) → positive energy
            energy_mwh = power_mw * self._time_step
            soc_delta = energy_mwh / self._eff_sqrt / self._capacity
            env.soc = max(self._min_soc, env.soc - soc_delta)
            actual_energy = (old_soc - env.soc) * self._capacity * self._eff_sqrt
        else:  # Charge (buy) → negative energy (COST)
            energy_mwh = abs(power_mw) * self._time_step
            soc_delta = energy_mwh * self._eff_sqrt / self._capacity
            env.soc = min(self._max_soc, env.soc + soc_delta)
            actual_energy = -((env.soc - old_soc) * self._capacity / self._eff_sqrt)

        return actual_energy, env.soc - old_soc

    # ─── Market determination (NO SoC updates here) ────────────

    def _determine_dam(self, row) -> dict:
        """I determine DAM commitment MW and price. No SoC change."""
        dam_mw = self._inner._get_dam_commitment(self._inner.current_step)
        dam_mw = np.clip(dam_mw, -self._max_power, self._max_power)
        return {'dam_mw': dam_mw, 'dam_price': row.get('price', 100.0)}

    def _determine_afrr(self, afrr_mw: float, price_tier: float, row) -> dict:
        """I determine aFRR commitment and activation. No SoC change.

        I handle 4-hour block selection and stochastic activation using ADMIE data,
        but I do NOT update SoC — that happens in the single net execution step.
        """
        env = self._inner
        result = {
            'afrr_commitment_mw': 0.0,
            'afrr_energy_mw': 0.0,
            'afrr_capacity_revenue': 0.0,
            'is_selected': False,
            'is_activated': False,
            'settlement_price': 0.0,
        }

        # Fix 5: I check prequalification before allowing aFRR
        if not self.is_prequalified_afrr:
            return result

        # I check if new 4-hour block starts (block boundary)
        if env._afrr_steps_remaining <= 0:
            committed = min(abs(afrr_mw), self._max_power)
            env.afrr_commitment_mw = committed
            tso_requirement = row.get('afrr_cap_up_qty', 300.0)
            estimated_pool = tso_requirement * 1.5
            base_prob = min(0.85, committed / max(estimated_pool, 1.0) + 0.3)
            tier_factor = 1.4 - price_tier * 0.4
            selection_prob = np.clip(base_prob * tier_factor, 0.05, 0.90)
            env.is_selected_for_afrr = np.random.random() < selection_prob
            env._afrr_steps_remaining = env._afrr_block_steps
        else:
            env._afrr_steps_remaining -= 1

        result['afrr_commitment_mw'] = env.afrr_commitment_mw
        result['is_selected'] = env.is_selected_for_afrr

        # I calculate capacity revenue (financial, independent of physical)
        if env.is_selected_for_afrr and env.afrr_commitment_mw > 0.1:
            cap_price = row.get('afrr_cap_up_price', 20.0)
            result['afrr_capacity_revenue'] = env.afrr_commitment_mw * cap_price * self._time_step

        # I check activation using REAL ADMIE volumes
        if env.is_selected_for_afrr and env.afrr_commitment_mw > 0.1:
            afrr_act_up = row.get('afrr_activated_up_mwh', 0.0)
            afrr_act_down = row.get('afrr_activated_down_mwh', 0.0)

            if afrr_act_up > 0.5:
                prob = min(0.70, env.afrr_commitment_mw / max(afrr_act_up, 1.0))
                if np.random.random() < prob:
                    result['is_activated'] = True
                    result['afrr_energy_mw'] = min(env.afrr_commitment_mw, self._max_power)
                    result['settlement_price'] = max(
                        row.get('afrr_up', 100.0), row.get('mfrr_price_up', 100.0))
                    env.steps_since_afrr_activation = 0

            elif afrr_act_down > 0.5:
                prob = min(0.70, env.afrr_commitment_mw / max(afrr_act_down, 1.0))
                if np.random.random() < prob:
                    result['is_activated'] = True
                    result['afrr_energy_mw'] = -min(env.afrr_commitment_mw, self._max_power)
                    result['settlement_price'] = max(
                        row.get('afrr_down', 80.0), row.get('mfrr_price_down', 80.0))
                    env.steps_since_afrr_activation = 0

            if not result['is_activated']:
                env.steps_since_afrr_activation += 1
        else:
            env.steps_since_afrr_activation += 1

        return result

    def _determine_mfrr(self, mfrr_mw: float, row) -> dict:
        """I determine mFRR activation using ADMIE data. No SoC change."""
        result = {'mfrr_mw': 0.0, 'mfrr_price': 0.0, 'activated': False}

        # Fix 5: I check prequalification before allowing mFRR
        if not self.is_prequalified_mfrr:
            return result

        if abs(mfrr_mw) < 0.1:
            return result

        real_activated_up = row.get('mfrr_activated_up_mwh', 0.0)
        real_activated_down = row.get('mfrr_activated_down_mwh', 0.0)

        if mfrr_mw > 0 and real_activated_up > 0.5:
            prob = min(0.60, abs(mfrr_mw) / max(real_activated_up, 1.0))
            if np.random.random() < prob:
                result['activated'] = True
                result['mfrr_mw'] = mfrr_mw
                result['mfrr_price'] = min(row.get('mfrr_price_up', 100.0),
                                           self._inner.mfrr_price_cap)

        elif mfrr_mw < 0 and real_activated_down > 0.5:
            prob = min(0.60, abs(mfrr_mw) / max(real_activated_down, 1.0))
            if np.random.random() < prob:
                result['activated'] = True
                result['mfrr_mw'] = mfrr_mw
                result['mfrr_price'] = min(row.get('mfrr_price_down', 80.0),
                                           self._inner.mfrr_price_cap)

        return result

    def _determine_xbid_price(self, xbid_mw: float, price_offset: float, row) -> dict:
        """I determine XBID execution price. No SoC change."""
        xbid_bid = row.get('xbid_price_bid', row.get('price', 100.0) - 1.5)
        xbid_ask = row.get('xbid_price_ask', row.get('price', 100.0) + 1.5)

        if xbid_mw > 0.1:
            return {'price': xbid_bid + price_offset}
        elif xbid_mw < -0.1:
            return {'price': xbid_ask + price_offset}
        else:
            return {'price': (xbid_bid + xbid_ask) / 2.0}

    # ─── Penalty computation ──────────────────────────────────────

    def _compute_penalty(self, d: dict) -> float:
        """I compute soft constraint penalties."""
        penalty = 0.0
        soc = self._inner.soc
        margin = 0.05

        net_discharge = max(0, d['xbid_mw']) + max(0, d['mfrr_mw']) + max(0, d['freebid_mw'])
        net_charge = abs(min(0, d['xbid_mw'])) + abs(min(0, d['mfrr_mw'])) + abs(min(0, d['freebid_mw']))
        discharge_intensity = net_discharge / self._max_power
        charge_intensity = net_charge / self._max_power

        if soc < self._min_soc + margin and discharge_intensity > 0:
            proximity = np.clip(1.0 - (soc - self._min_soc) / margin, 0, 1)
            penalty += self.penalties['soc_soft_margin'] * proximity * discharge_intensity

        elif soc > self._max_soc - margin and charge_intensity > 0:
            proximity = np.clip(1.0 - (self._max_soc - soc) / margin, 0, 1)
            penalty += self.penalties['soc_soft_margin'] * proximity * charge_intensity

        if self._inner.daily_cycles > self.CYCLE_TARGET:
            excess = self._inner.daily_cycles - self.CYCLE_TARGET
            penalty += self.penalties['cycle_excess'] * (excess ** 1.5)

        return penalty

    # ─── Main step ────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """I delegate reset to inner env."""
        self._inner._xbid_price_offset = 0.0
        obs, info = self._inner.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """I execute one step with NET PHYSICAL execution.

        CRITICAL DESIGN: The battery has ONE inverter. It either charges or discharges.
        I cannot do both simultaneously. So I:
        1. Determine all market MW positions (financial commitments)
        2. Calculate NET physical MW (single inverter setpoint)
        3. Execute ONE SoC update with net_mw
        4. Settle each market financially at its own price

        This ensures energy balance: energy_in - energy_out = SoC_change × capacity
        """
        # I decode and constrain
        decoded = self._decode_action(action)
        decoded = self._enforce_constraints(decoded)

        # I pass agent's DAM aggressiveness to inner env before day reset
        # (day reset triggers DAM bid generation which uses aggressiveness)
        self._inner.dam_aggressiveness = decoded.get('dam_aggressiveness', 0.0)

        # I check day boundary (resets daily cycles, IDA schedules, DAM bids)
        self._inner._check_day_reset()

        # I get current market data
        row = self._inner.df.iloc[self._inner.current_step]

        # ── Phase 1: Determine market positions with PRIORITY CASCADE ──
        # Priority: DAM (mandatory) → aFRR (if selected) → mFRR → XBID
        # I allocate inverter capacity in priority order. Lower-priority markets
        # get only the REMAINING capacity after higher-priority markets.

        dam = self._determine_dam(row)
        dam_mw = dam['dam_mw']

        # I calculate remaining inverter capacity after DAM
        remaining_capacity = self._max_power - abs(dam_mw)
        remaining_capacity = max(0.0, remaining_capacity)

        # aFRR (2nd priority — mandatory if selected by TSO)
        afrr = self._determine_afrr(
            min(abs(decoded['afrr_mw']), remaining_capacity),
            decoded['afrr_price_tier'], row)
        afrr_energy_mw = afrr['afrr_energy_mw']
        remaining_capacity -= abs(afrr_energy_mw)
        remaining_capacity = max(0.0, remaining_capacity)

        # mFRR (3rd priority — TSO-activated)
        mfrr_requested = np.clip(decoded['mfrr_mw'], -remaining_capacity, remaining_capacity)
        mfrr = self._determine_mfrr(mfrr_requested, row)
        mfrr_mw = mfrr['mfrr_mw']
        remaining_capacity -= abs(mfrr_mw)
        remaining_capacity = max(0.0, remaining_capacity)

        # XBID (lowest priority — agent-controlled, uses remaining capacity)
        xbid_mw = np.clip(decoded['xbid_mw'], -remaining_capacity, remaining_capacity)

        # XBID fill probability — orders may not fill in continuous market
        xbid_price_info = self._determine_xbid_price(xbid_mw, decoded['xbid_price_offset'], row)
        if abs(xbid_mw) > 0.1:
            xbid_mid = (row.get('xbid_price_bid', 98.0) + row.get('xbid_price_ask', 102.0)) / 2
            price_distance = abs(xbid_price_info['price'] - xbid_mid)
            # I model fill probability: near mid-price = high fill, far = low fill
            # At 0 EUR distance: 95% fill. At 20 EUR: 50%. At 50 EUR: 10%.
            fill_prob = np.clip(0.95 - price_distance * 0.02, 0.05, 0.95)
            if np.random.random() > fill_prob:
                xbid_mw = 0.0  # Order not filled

        # ── Phase 2: Calculate NET physical MW (already capacity-constrained) ──
        # No clipping needed — priority cascade ensures sum ≤ max_power
        net_mw = dam_mw + afrr_energy_mw + xbid_mw + mfrr_mw

        # ── Phase 3: Single SoC update ──

        actual_energy, soc_delta = self._execute_energy(net_mw)

        # I track cycles from the SINGLE net SoC change (no double-counting)
        cycle_frac = abs(soc_delta) / (self._max_soc - self._min_soc) / 2.0
        self._inner.daily_cycles += cycle_frac
        self._inner.total_cycles += cycle_frac

        # I track volumes
        if net_mw > 0.1:
            self._inner.intraday_mwh_sold += abs(actual_energy)
        elif net_mw < -0.1:
            self._inner.intraday_mwh_bought += abs(actual_energy)

        # ── Phase 4: Financial settlement per market ──
        # Each market is settled at its committed MW × its price × time_step
        # This is independent of physical execution (financial netting)

        dam_revenue = dam_mw * dam['dam_price'] * self._time_step
        afrr_energy_revenue = 0.0
        if afrr['is_activated'] and abs(afrr_energy_mw) > 0.1:
            afrr_energy_revenue = afrr_energy_mw * afrr['settlement_price'] * self._time_step
        xbid_revenue = xbid_mw * xbid_price_info['price'] * self._time_step
        mfrr_revenue = 0.0
        if mfrr['activated'] and abs(mfrr_mw) > 0.1:
            mfrr_revenue = mfrr_mw * mfrr['mfrr_price'] * self._time_step
        afrr_cap_revenue = afrr['afrr_capacity_revenue']

        # Fix 12: Availability contract revenue (fixed payment, independent of trading)
        availability_revenue = 0.0
        if self._enable_availability_contract:
            availability_revenue = (self._max_power *
                                    self.AVAILABILITY_CONTRACT_EUR_PER_MW_PER_HOUR *
                                    self._time_step)

        # I compute total market revenue for this step
        total_revenue = (dam_revenue + afrr_energy_revenue + afrr_cap_revenue +
                        xbid_revenue + mfrr_revenue + availability_revenue)

        # I account for degradation cost based on actual cycling
        deg_cost = abs(actual_energy) * self._inner.reward_calculator.degradation_cost

        # Fix 1: Network charges — I charge for energy flowing through the grid
        # Both charge and discharge incur UoS fees (metered at connection point)
        network_cost = abs(actual_energy) * self.NETWORK_CHARGE_EUR_PER_MWH

        # Fix 8: Calendar aging — single source (NOT double-counted with reward calculator)
        # I compute aging here and pass soc_penalty_coeff=0 to reward calculator
        # to avoid double-counting. Battery ~22M EUR / 20 years ≈ 3,000 EUR/day total.
        soc_pct = self._inner.soc * 100  # 0-100
        soc_deviation = abs(soc_pct - 50.0)  # distance from optimal 50%
        calendar_aging = (self.CALENDAR_AGING_BASE_EUR_PER_STEP +
                         self.CALENDAR_AGING_SOC_PENALTY_FACTOR * soc_deviation ** 2)

        # I compute net profit for this step
        step_profit = total_revenue - deg_cost - network_cost - calendar_aging

        # ── Phase 5: Build reward ──

        from gym_envs.unified_reward_calculator import UnifiedMarketState
        dam_price = dam['dam_price']
        id_bid = row.get('xbid_price_bid', dam_price - 1.5)
        id_ask = row.get('xbid_price_ask', dam_price + 1.5)

        # Fix 2: Single imbalance price (ADMIE uses weighted average of activated balancing)
        # I use the single imbalance_price column if available, otherwise fall back to DAM price
        single_imbalance_price = row.get('imbalance_price', dam_price)

        market_state = UnifiedMarketState(
            dam_price=dam_price,
            dam_commitment=dam_mw,
            intraday_bid=id_bid,
            intraday_ask=id_ask,
            intraday_spread=id_ask - id_bid,
            mfrr_price_up=min(row.get('mfrr_price_up', 100.0), self._inner.mfrr_price_cap),
            mfrr_price_down=min(row.get('mfrr_price_down', 80.0), self._inner.mfrr_price_cap),
            afrr_cap_up_price=row.get('afrr_cap_up_price', 20.0),
            afrr_cap_down_price=row.get('afrr_cap_down_price', 25.0),
            afrr_energy_up_price=row.get('afrr_up', 100.0),
            afrr_energy_down_price=row.get('afrr_down', 80.0),
            afrr_activated=afrr['is_activated'],
            afrr_activation_direction='up' if afrr_energy_mw > 0 else 'down',
            xbid_bid=id_bid,
            xbid_ask=id_ask,
        )

        reward_info = self._inner.reward_calculator.calculate(
            market=market_state,
            actual_energy_mw=net_mw,
            afrr_capacity_committed_mw=afrr['afrr_commitment_mw'],
            afrr_energy_delivered_mw=afrr_energy_mw,
            current_soc=self._inner.soc,
            capacity_mwh=self._capacity,
            time_step_hours=self._time_step,
            intraday_energy_mw=xbid_mw,
            mfrr_energy_mw=mfrr_mw,
            dam_executed_mw=dam_mw,
            afrr_max_deliverable_mw=min(
                self._inner.afrr_commitment_mw,
                (self._inner.soc - self._min_soc) * self._capacity * self._eff_sqrt / self._time_step
            ) if self._inner.is_selected_for_afrr else 0,
            daily_cycles=self._inner.daily_cycles,
            is_selected_for_afrr=self._inner.is_selected_for_afrr,
        )

        reward = reward_info['reward']

        # I apply soft penalty
        penalty = self._compute_penalty(decoded)
        reward_scale = self._inner.reward_calculator.reward_scale
        scaled_penalty = penalty * reward_scale
        reward -= scaled_penalty

        # ── Phase 6: Track profits ──

        self._inner.total_profit += step_profit
        self._inner.episode_profit += step_profit
        self._inner.episode_net_profit += step_profit

        self._inner.dam_profit += dam_revenue
        self._inner.afrr_capacity_profit += afrr_cap_revenue
        self._inner.afrr_energy_profit += afrr_energy_revenue
        self._inner.intraday_profit += xbid_revenue
        self._inner.mfrr_profit += mfrr_revenue

        # ── Phase 7: Advance step ──

        self._inner.current_step += 1
        ep_len = self._inner.episode_length
        if ep_len is not None:
            steps_in_ep = self._inner.current_step - self._inner.start_step
            terminated = steps_in_ep >= ep_len
        else:
            terminated = self._inner.current_step >= len(self._inner.df) - 2
        truncated = self._inner.current_step >= len(self._inner.df) - 2

        if not (terminated or truncated):
            obs = self._inner._obs_builder.build()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # ── Phase 8: Info dict ──

        info = {
            'soc': self._inner.soc,
            'dam_commitment_mw': dam_mw,
            'actual_mw': net_mw,
            'net_mw': net_mw,
            'total_profit': self._inner.total_profit,
            'dam_profit': self._inner.dam_profit,
            'intraday_profit': self._inner.intraday_profit,
            'afrr_capacity_profit': self._inner.afrr_capacity_profit,
            'afrr_energy_profit': self._inner.afrr_energy_profit,
            'mfrr_profit': self._inner.mfrr_profit,
            'total_cycles': self._inner.total_cycles,
            'daily_cycles': self._inner.daily_cycles,
            'net_profit': step_profit,
            'step_revenue': total_revenue,
            'step_degradation': deg_cost,
            'step_network_cost': network_cost,
            'step_calendar_aging': calendar_aging,
            'step_availability_revenue': availability_revenue,
            'sac_penalty': penalty,
            'sac_penalty_scaled': scaled_penalty,
            'discrete_action': [],
            'decoded_mw': {
                'afrr': decoded['afrr_mw'],
                'xbid': decoded['xbid_mw'],
                'xbid_price_offset': decoded['xbid_price_offset'],
                'mfrr': decoded['mfrr_mw'],
                'freebid': decoded['freebid_mw'],
                'dam': dam_mw,
                'afrr_energy': afrr_energy_mw,
                'mfrr_activated': mfrr_mw,
                'net_physical': net_mw,
            },
        }

        return obs, reward, terminated, truncated, info

    # ─── Utilities ────────────────────────────────────────────────

    def render(self):
        return self._inner.render()

    def close(self):
        return self._inner.close()

    def action_masks(self):
        return self._inner.action_masks()

    def set_degradation_cost(self, cost: float):
        self._inner.reward_calculator.degradation_cost = cost

    def set_penalties(self, penalty_overrides: Dict[str, float]):
        self.penalties.update(penalty_overrides)
