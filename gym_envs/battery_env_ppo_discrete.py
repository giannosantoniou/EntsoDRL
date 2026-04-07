"""
PPO Discrete Environment for Intraday Battery Trading (v1)

I implement a simple 5-action discrete space for MaskablePPO.
The agent decides WHEN to act, not how much — always max power.

Actions:
  0: IDLE          → 0 MW
  1: BUY MAX       → -30 MW XBID (charge)
  2: SELL MAX      → +30 MW XBID (discharge)
  3: aFRR COMMIT   → commit 15MW aFRR capacity (idle XBID)
  4: aFRR + BUY    → commit 15MW aFRR + buy remaining capacity

Action Masking:
  - Mask BUY (1,4) if SoC > 90%
  - Mask SELL (2) if SoC < 10%
  - Mask aFRR (3,4) if already committed in current 4h block

DAM is handled by LP Optimizer (locked, agent can't override).
Observation: same 20 focused features as SAC v6.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

from gym_envs.battery_env_unified import BatteryEnvUnified


class BatteryEnvPPODiscrete(gym.Env):
    """I provide a simple discrete action space for MaskablePPO."""

    metadata = {'render_modes': ['human']}

    N_ACTIONS = 7  # aFRR commitment levels: 0, 5, 10, 15, 20, 25, 30 MW
    CYCLE_TARGET = 5.0
    NETWORK_CHARGE_EUR_PER_MWH = 4.0
    CALENDAR_AGING_BASE_EUR_PER_STEP = 0.3
    CALENDAR_AGING_SOC_PENALTY_FACTOR = 0.001
    AFRR_LEVELS = [0, 5, 10, 15, 20, 25, 30]  # MW commitment options

    def __init__(self, inner_env: Optional[BatteryEnvUnified] = None, **kwargs):
        super().__init__()

        if inner_env is not None:
            self._inner = inner_env
        else:
            self._inner = BatteryEnvUnified(**kwargs)

        # I disable reward calculator's SoC penalty (I handle it myself)
        if hasattr(self._inner, 'reward_calculator') and self._inner.reward_calculator is not None:
            self._inner.reward_calculator.soc_penalty_coeff = 0.0

        self._max_power = self._inner.max_power_mw
        self._capacity = self._inner.capacity_mwh
        self._eff_sqrt = self._inner.eff_sqrt
        self._min_soc = self._inner.min_soc
        self._max_soc = self._inner.max_soc
        self._time_step = self._inner.time_step_hours

        # I define action and observation spaces
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.N_OBS = 20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.N_OBS,), dtype=np.float32
        )

        # I track aFRR commitment blocks (4h = 16 steps at 15-min)
        self._afrr_block_steps = 16
        self._afrr_steps_remaining = 0
        self._afrr_committed = False

    def action_masks(self) -> np.ndarray:
        """I return a boolean mask for valid aFRR commitment levels."""
        mask = np.ones(self.N_ACTIONS, dtype=bool)  # [0,5,10,15,20,25,30]
        soc = self._inner.soc
        dam_mw = abs(self._inner._get_dam_commitment(self._inner.current_step))
        remaining = max(0, self._max_power - dam_mw)

        # I mask if already committed in this 4h block
        if self._afrr_committed:
            for i in range(self.N_ACTIONS):
                mask[i] = False
            # I keep only the current commitment level active (no change mid-block)
            current_idx = min(range(len(self.AFRR_LEVELS)),
                            key=lambda i: abs(self.AFRR_LEVELS[i] - self._current_commitment))
            mask[current_idx] = True
            return mask

        # I mask commitment levels that exceed available capacity
        for i, mw in enumerate(self.AFRR_LEVELS):
            if mw > remaining + 0.5:
                mask[i] = False

        # I mask high commitments if SoC too low for delivery
        if soc < 0.15:
            for i in range(3, self.N_ACTIONS):  # I mask 15+ MW
                mask[i] = False
        elif soc < 0.25:
            for i in range(5, self.N_ACTIONS):  # I mask 25+ MW
                mask[i] = False

        # I always allow 0 MW
        mask[0] = True
        return mask

    def _decode_action(self, action: int) -> dict:
        """I convert discrete action to aFRR commitment MW.

        Actions 0-6 map to [0, 5, 10, 15, 20, 25, 30] MW.
        At block boundaries, I lock the commitment for 4 hours.
        """
        target_mw = self.AFRR_LEVELS[min(action, len(self.AFRR_LEVELS) - 1)]
        dam_mw = abs(self._inner._get_dam_commitment(self._inner.current_step))
        remaining = max(0, self._max_power - dam_mw)
        afrr_mw = min(target_mw, remaining)

        # I lock commitment at block boundary
        if not self._afrr_committed and afrr_mw > 0:
            self._afrr_committed = True
            self._afrr_steps_remaining = self._afrr_block_steps
            self._current_commitment = afrr_mw

        return {'xbid_mw': 0.0, 'afrr_mw': afrr_mw}

    def _build_obs(self) -> np.ndarray:
        """I build the same 20 focused features as SAC v6."""
        env = self._inner
        row = env.df.iloc[env.current_step]
        ts = env.df.index[env.current_step]
        sph = env._sph

        obs = np.zeros(self.N_OBS, dtype=np.float32)

        # 1. Battery state (2)
        obs[0] = env.soc
        obs[1] = env.daily_cycles / max(self.CYCLE_TARGET, 1)

        # 2. Available capacity (2)
        dam_commitment = env._get_dam_commitment(env.current_step)
        remaining = max(0, self._max_power - abs(dam_commitment))
        obs[2] = remaining / self._max_power
        obs[3] = dam_commitment / self._max_power

        # 3. Locked schedule lookahead (4)
        schedule_next = np.zeros(4)
        for i in range(4):
            step = env.current_step + i
            if step < len(env.df):
                schedule_next[i] = env._get_dam_commitment(step)
        obs[4] = np.mean(schedule_next) / self._max_power
        obs[5] = np.max(schedule_next) / self._max_power
        obs[6] = np.min(schedule_next) / self._max_power
        obs[7] = (np.max(schedule_next) - np.min(schedule_next)) / self._max_power

        # 4. SoC headroom (2)
        obs[8] = (env.soc - self._min_soc) * self._capacity / self._capacity
        obs[9] = (self._max_soc - env.soc) * self._capacity / self._capacity

        # 5. Market prices (4)
        dam_price = row.get('price', 100)
        xbid_bid = row.get('xbid_price_bid', dam_price - 1.5)
        xbid_ask = row.get('xbid_price_ask', dam_price + 1.5)
        obs[10] = dam_price / 200.0
        obs[11] = xbid_bid / 200.0
        obs[12] = xbid_ask / 200.0
        obs[13] = (xbid_bid - xbid_ask) / 50.0

        # 6. aFRR market (2)
        obs[14] = row.get('afrr_cap_up_price', 20) / 200.0
        obs[15] = float(self._afrr_committed)

        # 7. System state (2)
        obs[16] = row.get('net_imbalance_mw', 0) / 500.0
        obs[17] = row.get('imbalance_price', dam_price) / 200.0

        # 8. Time (2)
        hour = ts.hour + ts.minute / 60.0
        obs[18] = np.sin(2 * np.pi * hour / 24)
        obs[19] = np.cos(2 * np.pi * hour / 24)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(self, seed=None, options=None):
        """I reset environment."""
        _, info = self._inner.reset(seed=seed, options=options)
        self._inner._check_day_reset()
        self._afrr_committed = False
        self._afrr_steps_remaining = 0
        self._current_commitment = 0.0
        self._initial_soc = self._inner.soc
        obs = self._build_obs()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """I execute one step with discrete action."""

        # I decode action
        decoded = self._decode_action(action)
        xbid_mw = decoded['xbid_mw']
        afrr_mw = decoded['afrr_mw']

        # I update aFRR block timer
        if self._afrr_steps_remaining > 0:
            self._afrr_steps_remaining -= 1
            if self._afrr_steps_remaining == 0:
                self._afrr_committed = False

        # I check day boundary (DAM schedule from LP)
        self._inner._check_day_reset()

        # I get market data
        row = self._inner.df.iloc[self._inner.current_step]

        # I compute net MW (DAM + aFRR energy + XBID)
        dam_mw = self._inner._get_dam_commitment(self._inner.current_step)

        # I sync aFRR commitment to inner env so selection/activation works
        self._inner.afrr_commitment_mw = afrr_mw

        # I handle aFRR selection (merit order based on price tier)
        afrr_energy_mw = 0.0
        afrr_cap_revenue = 0.0
        afrr_energy_revenue = 0.0
        if afrr_mw > 0.5:
            # I simulate merit-order selection: competitive bid at 85% of clearing
            clearing_price = row.get('afrr_cap_up_price', 20.0)
            price_tier = 0.85  # I bid 15% below clearing for high selection probability
            ratio = price_tier
            # I use sigmoid selection (same as SAC wrapper)
            selection_prob = 1.0 / (1.0 + np.exp(10.0 * (ratio - 1.0)))
            is_selected = np.random.random() < selection_prob

            if is_selected:
                self._inner.is_selected_for_afrr = True
                # I earn capacity payment (pay-as-cleared, not pay-as-bid)
                afrr_cap_revenue = afrr_mw * clearing_price * self._time_step

                # I check activation (proportional to system needs)
                system_up = row.get('afrr_activated_up_mwh', 0)
                system_down = row.get('afrr_activated_down_mwh', 0)
                system_cap = row.get('afrr_cap_up_qty', 600)

                if system_up > 0.5:
                    our_share = min(0.50, afrr_mw / max(system_cap, 100))
                    if np.random.random() < our_share:
                        afrr_energy_mw = min(afrr_mw, system_up * afrr_mw / max(system_cap, 100))
                        afrr_price = max(row.get('afrr_up', 80), row.get('mfrr_price_up', 0))
                        afrr_energy_revenue = afrr_energy_mw * afrr_price * self._time_step
                elif system_down > 0.5:
                    our_share = min(0.50, afrr_mw / max(system_cap, 100))
                    if np.random.random() < our_share:
                        afrr_energy_mw = -min(afrr_mw, system_down * afrr_mw / max(system_cap, 100))
                        afrr_price = row.get('afrr_down', 80)
                        afrr_energy_revenue = abs(afrr_energy_mw) * afrr_price * self._time_step
            else:
                self._inner.is_selected_for_afrr = False

        # I handle mFRR auto-response
        mfrr_mw = 0.0
        mfrr_revenue = 0.0
        remaining_after_dam = max(0, self._max_power - abs(dam_mw) - abs(afrr_energy_mw) - abs(xbid_mw))
        system_mfrr_up = row.get('mfrr_activated_up_mwh', 0)
        if system_mfrr_up > 0.5 and remaining_after_dam > 1.0:
            our_share = min(0.15, remaining_after_dam / max(row.get('mfrr_requirements_up', 600), 100))
            if np.random.random() < our_share:
                mfrr_mw = min(remaining_after_dam, 10.0)
                mfrr_revenue = mfrr_mw * min(row.get('mfrr_price_up', 100), 1000) * self._time_step

        # I compute net physical MW
        net_mw = dam_mw + afrr_energy_mw + xbid_mw + mfrr_mw

        # I execute SoC change
        soc_before = self._inner.soc
        actual_energy, soc_delta = self._execute_energy(net_mw, soc_before)

        # I track cycles
        cycle_frac = abs(soc_delta) / (self._max_soc - self._min_soc) / 2.0
        self._inner.daily_cycles += cycle_frac
        self._inner.total_cycles += cycle_frac

        # I compute revenues
        dam_price = row.get('price', 100)
        dam_revenue = dam_mw * dam_price * self._time_step

        xbid_price = row.get('xbid_price_bid', dam_price) if xbid_mw > 0 else row.get('xbid_price_ask', dam_price)
        xbid_revenue = xbid_mw * xbid_price * self._time_step

        afrr_energy_revenue = 0.0
        if abs(afrr_energy_mw) > 0.1:
            afrr_energy_revenue = afrr_energy_mw * row.get('afrr_up', 100) * self._time_step

        total_revenue = dam_revenue + afrr_cap_revenue + afrr_energy_revenue + xbid_revenue + mfrr_revenue

        # I track cumulative aFRR profits
        if hasattr(self._inner, 'afrr_capacity_profit'):
            self._inner.afrr_capacity_profit += afrr_cap_revenue
        if hasattr(self._inner, 'afrr_energy_profit'):
            self._inner.afrr_energy_profit += afrr_energy_revenue

        # I compute costs
        deg_cost = abs(actual_energy) * 5.0  # degradation
        network_cost = abs(actual_energy) * self.NETWORK_CHARGE_EUR_PER_MWH
        soc_pct = self._inner.soc * 100
        calendar_aging = self.CALENDAR_AGING_BASE_EUR_PER_STEP + self.CALENDAR_AGING_SOC_PENALTY_FACTOR * (soc_pct - 50) ** 2

        # I apply per-market energy cost: XBID energy valued at DAM price
        # Agent profits ONLY from XBID-DAM spread, not from selling "free" energy
        xbid_energy_cost = xbid_mw * dam_price * self._time_step

        # I compute reward
        step_profit = total_revenue - xbid_energy_cost - deg_cost - network_cost - calendar_aging
        reward_scale = self._inner.reward_calculator.reward_scale
        reward = step_profit * reward_scale

        # I apply penalty
        penalty = 0.0
        soc = self._inner.soc
        margin = 0.05
        if soc < self._min_soc + margin:
            penalty += 30.0 * np.clip(1.0 - (soc - self._min_soc) / margin, 0, 1)
        if soc > self._max_soc - margin:
            penalty += 30.0 * np.clip(1.0 - (self._max_soc - soc) / margin, 0, 1)
        if self._inner.daily_cycles > self.CYCLE_TARGET:
            excess = self._inner.daily_cycles - self.CYCLE_TARGET
            penalty += 2000.0 * (excess ** 1.5)
        reward -= penalty * reward_scale

        # I track profits per market for P&L display
        self._inner.total_profit += step_profit
        self._inner.episode_profit += step_profit
        self._inner.dam_profit += dam_revenue
        self._inner.intraday_profit += xbid_revenue
        self._inner.mfrr_profit += mfrr_revenue

        # I advance step
        self._inner.current_step += 1
        ep_len = self._inner.episode_length
        if ep_len is not None:
            steps_in_ep = self._inner.current_step - self._inner.start_step
            terminated = steps_in_ep >= ep_len
        else:
            terminated = self._inner.current_step >= len(self._inner.df) - 2
        truncated = self._inner.current_step >= len(self._inner.df) - 2

        if not (terminated or truncated):
            obs = self._build_obs()
        else:
            obs = np.zeros(self.N_OBS, dtype=np.float32)

        # I get predicted price for visualization (from EntsoE3 cache)
        step_idx = self._inner.current_step - 1
        predicted_price = dam_price  # Default: same as actual
        if hasattr(self._inner, '_entsoe3_daily_cache') and self._inner._entsoe3_daily_cache:
            day_id = self._inner._step_day_id[step_idx] if step_idx < len(self._inner._step_day_id) else -1
            if day_id in self._inner._entsoe3_daily_cache:
                fc = self._inner._entsoe3_daily_cache[day_id]
                hour = self._inner.df.index[step_idx].hour if step_idx < len(self._inner.df) else 0
                predicted_price = float(fc[hour]) if hour < len(fc) else dam_price

        # I build info dict compatible with pygame_renderer
        ts = self._inner.df.index[self._inner.current_step - 1]
        info = {
            # State
            'soc': self._inner.soc,
            'timestamp': ts,
            'dam_price': dam_price,
            'predicted_price': predicted_price,
            'net_mw': net_mw,

            # Agent decisions
            'action': action,
            'action_name': f"aFRR_{self.AFRR_LEVELS[min(action, len(self.AFRR_LEVELS)-1)]}MW",
            'dam_commitment_mw': dam_mw,
            'ida1_mw': 0,  # IDA not used in PPO aFRR env
            'ida2_mw': 0,
            'xbid_mw': xbid_mw,
            'afrr_commitment_mw': afrr_mw,
            'afrr_mw': afrr_mw,
            'mfrr_auto_mw': mfrr_mw,

            # Step financials
            'step_profit': step_profit,
            'step_dam_revenue': dam_revenue,
            'step_xbid_revenue': xbid_revenue,
            'step_afrr_cap_revenue': afrr_cap_revenue,
            'step_afrr_energy_revenue': afrr_energy_revenue,
            'step_mfrr_revenue': mfrr_revenue,
            'step_ida1_revenue': 0,
            'step_ida2_revenue': 0,
            'step_degradation': deg_cost,
            'step_network_cost': network_cost,
            'step_calendar_aging': calendar_aging,
            'step_imbalance_cost': 0,
            'xbid_energy_cost': xbid_energy_cost,
            'penalty': penalty,

            # Cumulative
            'total_profit': self._inner.total_profit,
            'dam_profit': getattr(self._inner, 'dam_profit', 0),
            'intraday_profit': getattr(self._inner, 'intraday_profit', 0),
            'afrr_capacity_profit': getattr(self._inner, 'afrr_capacity_profit', 0),
            'afrr_energy_profit': getattr(self._inner, 'afrr_energy_profit', 0),
            'mfrr_profit': getattr(self._inner, 'mfrr_profit', 0),
            'ida_profit': getattr(self._inner, 'ida_profit', 0),
            'daily_cycles': self._inner.daily_cycles,
            'delivery_ratio': 1.0,
        }

        return obs, reward, terminated, truncated, info

    def _execute_energy(self, power_mw: float, soc_before: float) -> Tuple[float, float]:
        """I execute energy transfer and update SoC. Returns (actual_energy_mwh, soc_delta)."""
        if abs(power_mw) < 0.1:
            return 0.0, 0.0

        power_mw = np.clip(power_mw, -self._max_power, self._max_power)
        energy_mwh = power_mw * self._time_step

        if power_mw > 0:  # Discharge
            soc_drop = energy_mwh / (self._eff_sqrt * self._capacity)
            available = self._inner.soc - self._min_soc
            if soc_drop > available:
                soc_drop = available
                energy_mwh = soc_drop * self._eff_sqrt * self._capacity
            self._inner.soc -= soc_drop
        else:  # Charge
            soc_gain = abs(energy_mwh) * self._eff_sqrt / self._capacity
            available = self._max_soc - self._inner.soc
            if soc_gain > available:
                soc_gain = available
                energy_mwh = -soc_gain * self._capacity / self._eff_sqrt
            self._inner.soc += soc_gain

        self._inner.soc = np.clip(self._inner.soc, self._min_soc, self._max_soc)
        soc_delta = self._inner.soc - soc_before  # Simple: after - before

        return energy_mwh, soc_delta

    def render(self):
        pass
