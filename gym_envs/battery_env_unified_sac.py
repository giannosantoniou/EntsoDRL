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
    N_ACTIONS = 8  # 4 daily (DAM blocks) + 4 intraday (aFRR+XBID)

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
    # Realistic: 15 MW available 25% of the time (not full 30MW 24/7)
    # ~115,000 EUR/MW/year = ~13.15 EUR/MW/hour
    # Effective: 15 MW × 25% = 3.75 MW equivalent
    AVAILABILITY_MW = 15.0        # MW committed to availability
    AVAILABILITY_FRACTION = 0.25  # 25% of the time
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

        # Fix 3: I set soc_penalty_coeff=0 in reward calculator to avoid double
        # calendar aging (SAC wrapper has its own, more realistic model)
        if hasattr(self._inner, 'reward_calculator') and self._inner.reward_calculator is not None:
            self._inner.reward_calculator.soc_penalty_coeff = 0.0

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
        """I convert continuous [-1,+1] to absolute MW + price offset.

        Action space (8D):
          DAILY (applied at day boundary only):
          [0] charge_block_start:     0..20 (hour to start charging)
          [1] discharge_block_start:  4..20 (hour to start discharging)
          [2] second_cycle:           0 or 1
          [3] aggressiveness:         0.0..1.0 (MW per block)

          INTRADAY (applied every step):
          [4] aFRR commitment:        0..max_power MW
          [5] aFRR price tier:        0.7..1.3
          [6] XBID quantity:          -max_power..+max_power MW
          [7] XBID price offset:      -50..+50 EUR/MWh
        """
        a = np.clip(action, -1.0, 1.0)
        return {
            # Daily DAM decisions (used at day boundary)
            'charge_block_start': int((a[0] + 1.0) / 2.0 * 20),       # 0-20
            'discharge_block_start': int((a[1] + 1.0) / 2.0 * 16 + 4), # 4-20
            'second_cycle': 1 if a[2] > 0.0 else 0,                    # binary
            'aggressiveness': (a[3] + 1.0) / 2.0,                      # 0-1

            # Intraday decisions (every step)
            'afrr_mw': (a[4] + 1.0) / 2.0 * self._max_power,
            'afrr_price_tier': 0.7 + (a[5] + 1.0) / 2.0 * 0.6,
            'xbid_mw': a[6] * self._max_power,
            'xbid_price_offset': a[7] * self.XBID_PRICE_OFFSET_MAX,
        }

    def _enforce_constraints(self, d: dict) -> dict:
        """I enforce capacity + SoC limits for agent-controlled actions only.

        The agent controls ONLY aFRR commitment and XBID.
        DAM is set by optimizer, mFRR is auto-response — not in this dict.
        """
        # I get DAM commitment (mandatory, set by DamOptimizer)
        dam_mw = abs(self._inner._get_dam_commitment(self._inner.current_step))
        # I reserve capacity for potential mFRR auto-response (~5 MW buffer)
        mfrr_buffer = 5.0
        available = max(0.0, self._max_power - dam_mw - mfrr_buffer)

        # I scale down agent actions if total > available
        total = abs(d['afrr_mw']) + abs(d['xbid_mw'])
        if total > available and total > 0.1:
            scale = available / total
            d['afrr_mw'] *= scale
            d['xbid_mw'] *= scale

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

        # I enforce SoC limits on agent-controlled actions (XBID only — aFRR is commitment, not energy)
        dam_mw_signed = self._inner._get_dam_commitment(self._inner.current_step)
        total_discharge = max(0, d['xbid_mw']) + max(0, dam_mw_signed)
        total_charge = abs(min(0, d['xbid_mw'])) + abs(min(0, dam_mw_signed))

        if total_discharge > max_discharge_mw and total_discharge > 0.1:
            agent_discharge = max(0, d['xbid_mw'])
            if agent_discharge > 0.1:
                headroom = max(0, max_discharge_mw - max(0, dam_mw_signed))
                d['xbid_mw'] = min(d['xbid_mw'], headroom)

        if total_charge > max_charge_mw and total_charge > 0.1:
            agent_charge = abs(min(0, d['xbid_mw']))
            if agent_charge > 0.1:
                headroom = max(0, max_charge_mw - abs(min(0, dam_mw_signed)))
                d['xbid_mw'] = max(d['xbid_mw'], -headroom)

        # I enforce minimum bid sizes (HEnEx rules)
        if abs(d['afrr_mw']) < self.MIN_BID_BALANCING_MW:
            d['afrr_mw'] = 0.0
        if abs(d['xbid_mw']) < self.MIN_BID_DAM_MW:
            d['xbid_mw'] = 0.0

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

    def _apply_agent_dam_schedule(self, decoded: dict):
        """I generate DAM schedule from agent's daily block choices.

        The agent decides:
          - charge_block_start: which hour to start charging (0-20)
          - discharge_block_start: which hour to start discharging (4-20)
          - second_cycle: 0 or 1
          - aggressiveness: 0-1 (how much MW per block)

        I create a 96-slot (15-min) schedule from these block choices.
        The DamOptimizer already ran in _check_day_reset() but I OVERRIDE
        with the agent's choice — the agent controls DAM, not the optimizer.
        """
        env = self._inner
        if env.dam_schedule is None:
            return  # No day boundary this step

        cs = decoded['charge_block_start']     # 0-20
        ds = decoded['discharge_block_start']  # 4-20
        second = decoded['second_cycle']       # 0 or 1
        agg = decoded['aggressiveness']        # 0-1

        # I ensure discharge starts AFTER charge ends
        if ds < cs + 4:
            ds = min(cs + 4, 20)

        sph = env._sph  # Steps per hour (4 for 15-min)
        n_slots = len(env.dam_schedule)
        base_power = self._max_power * max(0.3, agg)  # At least 30% power

        # I build schedule: charge block → idle → discharge block
        schedule = np.zeros(n_slots)

        # Cycle 1: charge at cs, discharge at ds
        for h in range(4):
            for q in range(sph):
                slot = (cs + h) * sph + q
                if slot < n_slots:
                    schedule[slot] = -base_power  # Charge (negative = buy)
                slot = (ds + h) * sph + q
                if slot < n_slots:
                    schedule[slot] = base_power   # Discharge (positive = sell)

        # Cycle 2 (if agent chose it AND blocks fit)
        if second == 1 and ds + 4 + 8 <= 24:
            cs2 = ds + 4  # Second charge starts after first discharge
            ds2 = cs2 + 4  # Second discharge
            if ds2 + 4 <= 24:
                for h in range(4):
                    for q in range(sph):
                        slot = (cs2 + h) * sph + q
                        if slot < n_slots:
                            schedule[slot] = -base_power * 0.8  # Slightly less aggressive
                        slot = (ds2 + h) * sph + q
                        if slot < n_slots:
                            schedule[slot] = base_power * 0.8

        # I validate SoC trajectory (cap infeasible commitments)
        if hasattr(env, '_dam_optimizer') and env._dam_optimizer is not None:
            schedule = env._dam_optimizer._cap_by_soc_trajectory(
                schedule, env.soc, n_slots)

        # I override the optimizer's schedule with agent's choice
        env.dam_schedule = schedule
        env._dam_schedule_day_id = env._step_day_id[env.current_step]

    def _determine_dam(self, row) -> dict:
        """I determine DAM commitment MW and price. No SoC change."""
        dam_mw = self._inner._get_dam_commitment(self._inner.current_step)
        dam_mw = np.clip(dam_mw, -self._max_power, self._max_power)
        return {'dam_mw': dam_mw, 'dam_price': row.get('price', 100.0)}

    def _determine_afrr(self, afrr_mw: float, price_tier: float, row) -> dict:
        """I determine aFRR commitment and activation using MERIT ORDER logic.

        Selection: I compare agent's bid price vs market clearing price.
          If agent bid <= clearing → selected (cheapest wins).
        Activation: DETERMINISTIC from ADMIE data (no random rolls).
          If TSO activated aFRR this step → battery must respond.
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

        if not self.is_prequalified_afrr:
            return result

        # I check if new 4-hour block starts (block boundary)
        if env._afrr_steps_remaining <= 0:
            committed = min(abs(afrr_mw), self._max_power)
            env.afrr_commitment_mw = committed

            # SMOOTH MERIT ORDER: I compare agent bid vs clearing price
            # Instead of hard cutoff, I use a sigmoid probability based on
            # how far the bid is from clearing. This gives smooth gradient.
            clearing_price = row.get('afrr_cap_up_price', 20.0)
            agent_bid_price = clearing_price * price_tier  # tier 0.7-1.3 × clearing

            # I compute selection probability: sigmoid centered at clearing price
            # bid << clearing → prob ≈ 1.0 (cheap bid, always wins)
            # bid ≈ clearing → prob ≈ 0.5 (marginal bid)
            # bid >> clearing → prob ≈ 0.0 (expensive bid, never wins)
            if clearing_price > 0.1 and committed > self.MIN_BID_BALANCING_MW:
                ratio = agent_bid_price / clearing_price  # <1 = cheap, >1 = expensive
                # I use logistic function: p = 1 / (1 + exp(k*(ratio - 1)))
                # k=10 gives smooth transition: 0.73→1.00 ratio maps to 95%→50% prob
                selection_prob = 1.0 / (1.0 + np.exp(10.0 * (ratio - 1.0)))
                env.is_selected_for_afrr = np.random.random() < selection_prob
            else:
                env.is_selected_for_afrr = False
            env._afrr_steps_remaining = env._afrr_block_steps
        else:
            env._afrr_steps_remaining -= 1

        result['afrr_commitment_mw'] = env.afrr_commitment_mw
        result['is_selected'] = env.is_selected_for_afrr

        # I calculate capacity revenue (if selected)
        if env.is_selected_for_afrr and env.afrr_commitment_mw > 0.1:
            cap_price = row.get('afrr_cap_up_price', 20.0)
            result['afrr_capacity_revenue'] = env.afrr_commitment_mw * cap_price * self._time_step

        # PROPORTIONAL activation: afrr_activated_up/down_mwh is SYSTEM-LEVEL data.
        # Our battery is one provider among ~600 MW total aFRR capacity.
        # I activate proportionally to our committed share.
        if env.is_selected_for_afrr and env.afrr_commitment_mw > 0.1:
            afrr_act_up = row.get('afrr_activated_up_mwh', 0.0)
            afrr_act_down = row.get('afrr_activated_down_mwh', 0.0)
            system_cap_up = max(row.get('afrr_cap_up_qty', 600.0), 100.0)
            system_cap_down = max(row.get('afrr_cap_down_qty', 120.0), 50.0)

            if afrr_act_up > 0.5:
                # I calculate our proportional share of system activation
                our_share = min(0.50, env.afrr_commitment_mw / system_cap_up)
                if np.random.random() < our_share:
                    proportional_mw = min(env.afrr_commitment_mw,
                                         afrr_act_up * env.afrr_commitment_mw / system_cap_up)
                    proportional_mw = max(proportional_mw, 1.0)
                    result['is_activated'] = True
                    result['afrr_energy_mw'] = proportional_mw
                    result['settlement_price'] = max(
                        row.get('afrr_up', 100.0), row.get('mfrr_price_up', 100.0))
                    env.steps_since_afrr_activation = 0

            elif afrr_act_down > 0.5:
                our_share = min(0.50, env.afrr_commitment_mw / system_cap_down)
                if np.random.random() < our_share:
                    proportional_mw = min(env.afrr_commitment_mw,
                                         afrr_act_down * env.afrr_commitment_mw / system_cap_down)
                    proportional_mw = max(proportional_mw, 1.0)
                    result['is_activated'] = True
                    result['afrr_energy_mw'] = -proportional_mw
                    result['settlement_price'] = max(
                        row.get('afrr_down', 80.0), row.get('mfrr_price_down', 80.0))
                    env.steps_since_afrr_activation = 0

            if not result['is_activated']:
                env.steps_since_afrr_activation += 1
        else:
            env.steps_since_afrr_activation += 1

        return result

    def _determine_mfrr_auto(self, remaining_capacity_mw: float, row) -> dict:
        """I auto-respond to TSO mFRR activation. Agent does NOT control this.

        mFRR is TSO-activated: when the grid needs balancing, the TSO calls
        on available reserves. The battery must respond with full available
        capacity. This is deterministic from ADMIE data — no random rolls.

        I also check SoC limits — can't discharge if SoC too low, can't charge if too high.
        """
        result = {'mfrr_mw': 0.0, 'mfrr_price': 0.0, 'activated': False}

        if not self.is_prequalified_mfrr:
            return result

        if remaining_capacity_mw < self.MIN_BID_BALANCING_MW:
            return result

        # I compute SoC-constrained capacity
        soc = self._inner.soc
        max_discharge_mw = min(
            remaining_capacity_mw,
            (soc - self._min_soc) * self._capacity * self._eff_sqrt / self._time_step
        )
        max_charge_mw = min(
            remaining_capacity_mw,
            (self._max_soc - soc) * self._capacity / self._eff_sqrt / self._time_step
        )

        # I check if TSO activated mFRR this step (SYSTEM-LEVEL data)
        # CRITICAL: mfrr_activated_up/down_mwh is the TOTAL system activation,
        # NOT our battery specifically. Our 30MW battery is one provider among ~600MW total.
        # I activate PROPORTIONALLY to our share of system capacity.
        real_up = row.get('mfrr_activated_up_mwh', 0.0)
        real_down = row.get('mfrr_activated_down_mwh', 0.0)
        system_req_up = max(row.get('mfrr_requirements_up', 600.0), 100.0)
        system_req_down = max(row.get('mfrr_requirements_down', 200.0), 100.0)

        if real_up > 0.5 and max_discharge_mw >= self.MIN_BID_BALANCING_MW:
            # I calculate our share: 30MW out of ~600MW system = ~5%
            our_share = min(0.30, remaining_capacity_mw / system_req_up)
            if np.random.random() < our_share:
                # I activate with proportional MW (not full capacity)
                proportional_mw = min(max_discharge_mw, real_up * remaining_capacity_mw / system_req_up)
                proportional_mw = max(proportional_mw, self.MIN_BID_BALANCING_MW)
                result['activated'] = True
                result['mfrr_mw'] = proportional_mw
                result['mfrr_price'] = min(row.get('mfrr_price_up', 100.0),
                                           self._inner.mfrr_price_cap)

        elif real_down > 0.5 and max_charge_mw >= self.MIN_BID_BALANCING_MW:
            our_share = min(0.30, remaining_capacity_mw / system_req_down)
            if np.random.random() < our_share:
                proportional_mw = min(max_charge_mw, real_down * remaining_capacity_mw / system_req_down)
                proportional_mw = max(proportional_mw, self.MIN_BID_BALANCING_MW)
                result['activated'] = True
                result['mfrr_mw'] = -proportional_mw
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

    def _compute_penalty(self, d: dict, row=None) -> float:
        """I compute soft constraint penalties + XBID buy incentive."""
        penalty = 0.0
        soc = self._inner.soc

        # FIX 3a: STRONGER SoC penalty below 20% (agent drains battery too much)
        if soc < 0.20:
            # I heavily penalize being below 20% — battery needs energy to sell later
            low_soc_severity = (0.20 - soc) / 0.15  # 0 at 20%, 1 at 5%
            penalty += 100.0 * low_soc_severity ** 2  # Quadratic, up to 100 EUR

        # Standard margin penalties (near 5% or 95%)
        margin = 0.05
        net_discharge = max(0, d.get('xbid_mw', 0))
        discharge_intensity = net_discharge / self._max_power

        if soc < self._min_soc + margin and discharge_intensity > 0:
            proximity = np.clip(1.0 - (soc - self._min_soc) / margin, 0, 1)
            penalty += self.penalties['soc_soft_margin'] * proximity * discharge_intensity

        # Cycle excess
        if self._inner.daily_cycles > self.CYCLE_TARGET:
            excess = self._inner.daily_cycles - self.CYCLE_TARGET
            penalty += self.penalties['cycle_excess'] * (excess ** 1.5)

        # FIX 3b: XBID buy incentive — bonus (negative penalty) for buying cheap
        if row is not None and d.get('xbid_mw', 0) < -0.5:
            price = row.get('price', 100)
            mean_24h = row.get('price_mean_24h', price)
            if price < mean_24h * 0.85:
                # I reward buying when price is 15%+ below daily mean
                buy_bonus = 5.0 * (mean_24h - price) / max(mean_24h, 1.0)
                penalty -= buy_bonus  # Negative penalty = bonus

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
        # I decode agent's decisions (daily DAM + intraday aFRR/XBID)
        decoded = self._decode_action(action)
        decoded = self._enforce_constraints(decoded)

        # I store daily actions for DAM schedule generation
        self._last_daily_action = decoded

        # I check day boundary (resets daily state, generates DAM schedule)
        self._inner._check_day_reset()

        # AGENT-CONTROLLED DAM: I override the optimizer schedule with agent's choice
        # The agent decides charge/discharge block timing via daily actions [D0-D3]
        self._apply_agent_dam_schedule(decoded)

        # I get current market data
        row = self._inner.df.iloc[self._inner.current_step]

        # ── Phase 1: HIERARCHICAL PRIORITY CASCADE ──
        # Priority: DAM (mandatory) → aFRR (mandatory if activated) → mFRR (auto) → XBID (agent)

        # 1. DAM (mandatory, set by DamOptimizer at day boundary)
        dam = self._determine_dam(row)
        dam_mw = dam['dam_mw']
        remaining_capacity = max(0.0, self._max_power - abs(dam_mw))

        # 2. aFRR (agent controls commitment, activation is deterministic/merit-order)
        afrr = self._determine_afrr(
            min(abs(decoded['afrr_mw']), remaining_capacity),
            decoded['afrr_price_tier'], row)
        afrr_energy_mw = afrr['afrr_energy_mw']
        remaining_capacity = max(0.0, remaining_capacity - abs(afrr_energy_mw))

        # 3. IDA1/2/3 (rule-based, locked at gate closures — like DAM but intraday)
        # I trigger IDA gate closures at the correct timestamps
        ts = self._inner.df.index[self._inner.current_step]
        hour_min = ts.hour * 60 + ts.minute

        # IDA1: D-1 15:00 (checks both exact and within 15-min window)
        if 900 <= hour_min < 915 and not self._inner.ida1_locked_today:
            try:
                self._inner._trigger_ida_gate(1, participation_level=1.0)
            except Exception:
                pass

        # IDA2: D-1 22:00
        if 1320 <= hour_min < 1335 and not self._inner.ida2_locked_today:
            try:
                self._inner._trigger_ida_gate(2, participation_level=1.0)
            except Exception:
                pass

        # IDA3: D+0 10:00
        if 600 <= hour_min < 615 and not self._inner.ida3_locked_today:
            try:
                self._inner._trigger_ida_gate(3, participation_level=1.0)
            except Exception:
                pass

        # I execute locked IDA commitments (mandatory, like DAM)
        ida_mw = self._inner._get_ida_commitment(self._inner.current_step)
        ida_mw = np.clip(ida_mw, -remaining_capacity, remaining_capacity)
        remaining_capacity = max(0.0, remaining_capacity - abs(ida_mw))

        # 4. mFRR (AUTO-RESPONSE — agent does NOT control, TSO deterministic)
        mfrr = self._determine_mfrr_auto(remaining_capacity, row)
        mfrr_mw = mfrr['mfrr_mw']
        remaining_capacity = max(0.0, remaining_capacity - abs(mfrr_mw))

        # 5. XBID (agent-controlled, uses remaining capacity)
        # XBID gate closure: no trading within 1 hour of delivery
        # ts already set above (for IDA gate checks)
        xbid_gate_open = True
        step_in_day = self._inner._step_day_offset[self._inner.current_step]
        day_id = self._inner._step_day_id[self._inner.current_step]
        if day_id < len(self._inner._day_length):
            day_length = self._inner._day_length[day_id]
            if day_length - step_in_day <= 4:  # Last 1 hour (4 × 15min)
                xbid_gate_open = False

        if xbid_gate_open:
            xbid_mw = np.clip(decoded['xbid_mw'], -remaining_capacity, remaining_capacity)
        else:
            xbid_mw = 0.0  # XBID gate closed

        xbid_price_info = self._determine_xbid_price(xbid_mw, decoded['xbid_price_offset'], row)

        # I model XBID fill probability (continuous market, not guaranteed)
        if abs(xbid_mw) > 0.1:
            xbid_mid = (row.get('xbid_price_bid', 98.0) + row.get('xbid_price_ask', 102.0)) / 2
            price_distance = abs(xbid_price_info['price'] - xbid_mid)
            fill_prob = np.clip(0.95 - price_distance * 0.02, 0.05, 0.95)
            if np.random.random() > fill_prob:
                xbid_mw = 0.0  # Order not filled

        # ── Phase 2: Calculate NET physical MW (already capacity-constrained) ──
        # No clipping needed — priority cascade ensures sum ≤ max_power
        net_mw = dam_mw + afrr_energy_mw + ida_mw + xbid_mw + mfrr_mw

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

        # IDA revenue: settled at clearing prices from data
        ida_revenue = 0.0
        if abs(ida_mw) > 0.1:
            # I compute revenue from each IDA auction's clearing price
            step_offset = self._inner._step_day_offset[self._inner.current_step]
            for ida_num, sched_attr in [(1, 'ida1_schedule'), (2, 'ida2_schedule'), (3, 'ida3_schedule')]:
                sched = getattr(self._inner, sched_attr, None)
                if sched is not None and step_offset < len(sched):
                    pos = float(sched[step_offset])
                    if abs(pos) > 0.1:
                        clearing = row.get(f'ida{ida_num}_clearing_price', row.get('price', 100.0))
                        ida_revenue += pos * clearing * self._time_step

        # Availability contract: I track for accounting but EXCLUDE from RL reward
        # It's a constant +99 EUR/step regardless of agent action — adds no learning signal
        availability_revenue = 0.0
        if self._enable_availability_contract:
            # Realistic: 15 MW committed, available 25% of the time
            availability_revenue = (self.AVAILABILITY_MW *
                                    self.AVAILABILITY_FRACTION *
                                    self.AVAILABILITY_CONTRACT_EUR_PER_MW_PER_HOUR *
                                    self._time_step)

        # FULL REVENUE: Agent controls ALL markets now
        # DAM schedule = agent's daily block choice
        # XBID + aFRR = agent's intraday decisions
        # mFRR = auto-response (agent positions SoC to benefit)
        # IDA = EXCLUDED from reward (synthetic clearing prices unreliable 2021-2023)
        #       Still tracked in info dict for accounting
        total_revenue = (dam_revenue + afrr_energy_revenue +
                        afrr_cap_revenue + xbid_revenue + mfrr_revenue)
        total_revenue_with_ida = total_revenue + ida_revenue  # For accounting only
        # Note: availability_revenue NOT included — it's reported in info dict only

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

        # I compute net profit for this step (FULL accounting, all markets)
        step_profit = total_revenue - deg_cost - network_cost - calendar_aging
        full_net_profit = step_profit

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

        # I use FULL profit for RL reward (agent controls ALL markets)
        reward_scale = self._inner.reward_calculator.reward_scale
        reward = full_net_profit * reward_scale

        # I apply soft penalty
        penalty = self._compute_penalty(decoded, row=row)
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
        self._inner.ida_profit += ida_revenue

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
            # FIX 4: I add small noise to observations during TRAINING
            # This prevents the agent from memorizing exact price patterns
            # and forces it to learn robust strategies
            if self._inner.np_random is not None:
                obs_noise = self._inner.np_random.normal(0, 0.01, size=obs.shape)
                obs = obs + obs_noise.astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # ── Phase 8: Info dict ──

        # I get timestamp for logging
        step_idx = self._inner.current_step - 1  # Already advanced
        step_ts = str(self._inner.df.index[step_idx]) if step_idx < len(self._inner.df) else ''

        # I get predicted price for this hour (from EntsoE3 cache)
        predicted_price = 0.0
        if hasattr(self._inner, '_entsoe3_daily_cache') and self._inner._entsoe3_daily_cache:
            day_id = self._inner._step_day_id[step_idx] if step_idx < len(self._inner._step_day_id) else -1
            if day_id in self._inner._entsoe3_daily_cache:
                fc = self._inner._entsoe3_daily_cache[day_id]
                hour = self._inner.df.index[step_idx].hour if step_idx < len(self._inner.df) else 0
                predicted_price = float(fc[hour]) if hour < len(fc) else 0.0

        info = {
            # State
            'timestamp': step_ts,
            'soc': self._inner.soc,
            'dam_price': row.get('price', 0),
            'predicted_price': predicted_price,

            # Agent decisions
            'dam_commitment_mw': dam_mw,
            'ida_mw': ida_mw,
            'afrr_commitment_mw': afrr.get('afrr_commitment_mw', 0),
            'afrr_energy_mw': afrr_energy_mw,
            'xbid_mw': xbid_mw,
            'xbid_price_offset': decoded['xbid_price_offset'],
            'mfrr_auto_mw': mfrr_mw,
            'net_mw': net_mw,

            # Daily actions
            'charge_block_start': decoded.get('charge_block_start', 0),
            'discharge_block_start': decoded.get('discharge_block_start', 0),
            'second_cycle': decoded.get('second_cycle', 0),
            'aggressiveness': decoded.get('aggressiveness', 0),

            # Financial
            'step_dam_revenue': dam_revenue,
            'step_ida_revenue': ida_revenue,
            'step_afrr_cap_revenue': afrr_cap_revenue,
            'step_afrr_energy_revenue': afrr_energy_revenue,
            'step_xbid_revenue': xbid_revenue,
            'step_mfrr_revenue': mfrr_revenue,
            'step_revenue': total_revenue,
            'step_degradation': deg_cost,
            'step_network_cost': network_cost,
            'step_calendar_aging': calendar_aging,
            'step_availability_revenue': availability_revenue,
            'step_profit': step_profit,
            'reward': reward,

            # Cumulative
            'total_profit': self._inner.total_profit,
            'dam_profit': self._inner.dam_profit,
            'intraday_profit': self._inner.intraday_profit,
            'afrr_capacity_profit': self._inner.afrr_capacity_profit,
            'afrr_energy_profit': self._inner.afrr_energy_profit,
            'mfrr_profit': self._inner.mfrr_profit,
            'ida_profit': self._inner.ida_profit,
            'total_cycles': self._inner.total_cycles,
            'daily_cycles': self._inner.daily_cycles,

            # Penalties
            'sac_penalty': penalty,

            # Raw decoded actions
            'decoded_mw': {
                'afrr': decoded['afrr_mw'],
                'xbid': decoded['xbid_mw'],
                'xbid_price_offset': decoded['xbid_price_offset'],
                'dam': dam_mw,
                'afrr_energy': afrr_energy_mw,
                'mfrr_auto': mfrr_mw,
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

    @staticmethod
    def log_episode_to_csv(infos: list, output_path: str):
        """I save per-step decision logs to CSV for analysis.

        Usage during evaluation:
            infos = []
            for step in range(672):
                obs, reward, done, truncated, info = env.step(action)
                infos.append(info)
            BatteryEnvUnifiedSAC.log_episode_to_csv(infos, 'eval_decisions.csv')
        """
        import pandas as pd

        # I extract flat columns from info dicts (skip nested 'decoded_mw')
        rows = []
        for info in infos:
            row = {k: v for k, v in info.items() if not isinstance(v, dict)}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return df

    def set_degradation_cost(self, cost: float):
        self._inner.reward_calculator.degradation_cost = cost

    def set_penalties(self, penalty_overrides: Dict[str, float]):
        self.penalties.update(penalty_overrides)
