"""
SAC Wrapper for Unified Multi-Market Battery Trading Environment

I wrap BatteryEnvUnified (composition) to provide a continuous Box action space
for SAC training while reusing all existing market simulation logic.

Key Design Decisions:
1. Composition over inheritance — I hold an inner BatteryEnvUnified instance
2. Continuous actions [-1,1] are discretized to nearest MultiDiscrete index
3. Penalty-based soft constraints replace PPO's hard action masking
4. Inner env's action_masks() is read to detect invalid actions and penalize

Action Space: Box(low=-1, high=1, shape=(4,))
  a[0]: aFRR commitment fraction (-1=0%, +1=100%)
  a[1]: aFRR price tier (-1=aggressive 0.7x, +1=conservative 1.3x)
  a[2]: IntraDay energy (-1=full charge, +1=full discharge)
  a[3]: mFRR energy (-1=full charge, +1=full discharge)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

from gym_envs.battery_env_unified import BatteryEnvUnified


class BatteryEnvUnifiedSAC(gym.Env):
    """
    I wrap BatteryEnvUnified with a continuous action space for SAC.

    I convert continuous [-1, 1] actions to discrete MultiDiscrete indices,
    delegate to the inner env for market simulation, and apply penalty-based
    soft constraints instead of hard action masking.
    """

    metadata = {'render_modes': ['human']}

    # I define penalty costs (EUR) for constraint violations
    DEFAULT_PENALTIES = {
        'soc_hard': 200.0,       # EUR/MW for hard SoC violation
        'soc_soft_margin': 20.0, # EUR/MW for soft SoC margin (5% buffer)
        'gate_closure': 50.0,    # EUR/MW for IntraDay after 23:00
        'mfrr_unavailable': 50.0,# EUR/MW for mFRR when no activation
        'cycle_excess': 100.0,   # EUR per excess cycle above 1.8/day
    }

    def __init__(
        self,
        inner_env: Optional[BatteryEnvUnified] = None,
        penalties: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        I initialize the SAC wrapper.

        Args:
            inner_env: Pre-built BatteryEnvUnified instance (preferred).
            penalties: Override default penalty costs.
            **kwargs: Passed to BatteryEnvUnified if inner_env is None.
        """
        super().__init__()

        # I store the inner env (composition)
        if inner_env is not None:
            self._inner = inner_env
        else:
            self._inner = BatteryEnvUnified(**kwargs)

        # I set up penalty costs
        self.penalties = dict(self.DEFAULT_PENALTIES)
        if penalties:
            self.penalties.update(penalties)

        # I cache discrete action dimensions from inner env
        self._nvec = self._inner.action_space.nvec  # [5, 5, 11, 11] or [5, 5, 11, 11, 5]

        # I define continuous action space: N dims in [-1, 1] matching inner env
        n_dims = len(self._nvec)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_dims,), dtype=np.float32
        )

        # I delegate observation space from inner env
        self.observation_space = self._inner.observation_space

    @property
    def reward_calculator(self):
        """I expose the inner env's reward calculator for curriculum callbacks."""
        return self._inner.reward_calculator

    @reward_calculator.setter
    def reward_calculator(self, value):
        """I allow setting the reward calculator on the inner env."""
        self._inner.reward_calculator = value

    @property
    def soc(self):
        """I expose inner env SoC for monitoring."""
        return self._inner.soc

    @property
    def daily_cycles(self):
        """I expose inner env daily cycles for monitoring."""
        return self._inner.daily_cycles

    # I define the IntraDay dead zone threshold for SAC's tanh-squashed actions.
    # SAC uses tanh(Normal(0,1)) which creates a U-shaped distribution concentrated
    # near ±1. Without this, only ~5% of actions land in the no-trade zone.
    # With threshold=0.5, ~42% of initial random actions map to no-trade,
    # letting the agent learn that IntraDay is only profitable selectively.
    INTRADAY_DEAD_ZONE = 0.5

    def continuous_to_discrete(self, continuous_action: np.ndarray) -> np.ndarray:
        """
        I convert continuous [-1, 1] actions to discrete MultiDiscrete indices.

        Mapping for each dimension:
          continuous -1.0 -> discrete index 0
          continuous  0.0 -> discrete index (n-1)/2  (middle)
          continuous +1.0 -> discrete index n-1

        For IntraDay (dim 2), I apply a wider dead zone so that SAC's
        tanh-squashed policy has a meaningful no-trade region.
        """
        continuous_action = np.clip(continuous_action, -1.0, 1.0)
        discrete = np.zeros(len(self._nvec), dtype=np.int64)

        for i, n in enumerate(self._nvec):
            # I map [-1, 1] to [0, n-1] with rounding to nearest index
            idx = (continuous_action[i] + 1.0) / 2.0 * (n - 1)
            discrete[i] = int(np.round(np.clip(idx, 0, n - 1)))

        # I apply IntraDay dead zone: continuous values near center → no trade
        if abs(continuous_action[2]) < self.INTRADAY_DEAD_ZONE:
            discrete[2] = self._nvec[2] // 2  # center index = no trade

        return discrete

    def _compute_penalty(
        self, continuous_action: np.ndarray, discrete_action: np.ndarray
    ) -> float:
        """
        I compute penalty for constraint violations using inner env's action masks.

        I read the masks that PPO would use, then penalize proportionally
        when the SAC agent selects actions that would have been masked.
        """
        mask = self._inner.action_masks()
        penalty = 0.0

        # I split the flat mask into per-dimension segments
        offsets = np.cumsum([0] + list(self._nvec))
        for dim_idx in range(len(self._nvec)):
            start = offsets[dim_idx]
            end = offsets[dim_idx + 1]
            dim_mask = mask[start:end]
            chosen_idx = discrete_action[dim_idx]

            if not dim_mask[chosen_idx]:
                # I determine penalty type based on which dimension is violated
                if dim_idx == 0:
                    # aFRR commitment violation (SoC-related)
                    penalty += self.penalties['soc_hard']
                elif dim_idx == 1:
                    # aFRR price tier — rarely masked, minimal penalty
                    pass
                elif dim_idx == 2:
                    # IntraDay violation
                    # I check if it's a gate closure issue or SoC issue
                    row = self._inner.df.iloc[self._inner.current_step]
                    intraday_open = self._inner._is_intraday_open(row)
                    if not intraday_open:
                        penalty += self.penalties['gate_closure']
                    else:
                        penalty += self.penalties['soc_hard']
                elif dim_idx == 3:
                    # mFRR violation (direction/availability)
                    penalty += self.penalties['mfrr_unavailable']

        # I add soft SoC margin penalty (penalize approaching limits)
        soc = self._inner.soc
        margin = 0.05  # 5% buffer zone
        if soc < self._inner.min_soc + margin:
            # I penalize discharge actions near min SoC
            id_level = self._inner.INTRADAY_LEVELS[discrete_action[2]]
            mfrr_level = self._inner.MFRR_LEVELS[discrete_action[3]]
            discharge_intensity = max(0, id_level) + max(0, mfrr_level)
            if discharge_intensity > 0:
                proximity = 1.0 - (soc - self._inner.min_soc) / margin
                proximity = np.clip(proximity, 0, 1)
                penalty += self.penalties['soc_soft_margin'] * proximity * discharge_intensity

        elif soc > self._inner.max_soc - margin:
            # I penalize charge actions near max SoC
            id_level = self._inner.INTRADAY_LEVELS[discrete_action[2]]
            mfrr_level = self._inner.MFRR_LEVELS[discrete_action[3]]
            charge_intensity = max(0, -id_level) + max(0, -mfrr_level)
            if charge_intensity > 0:
                proximity = 1.0 - (self._inner.max_soc - soc) / margin
                proximity = np.clip(proximity, 0, 1)
                penalty += self.penalties['soc_soft_margin'] * proximity * charge_intensity

        # I add daily cycle excess penalty
        if self._inner.daily_cycles > 1.8:
            excess = self._inner.daily_cycles - 1.8
            penalty += self.penalties['cycle_excess'] * excess

        return penalty

    def reset(self, seed=None, options=None):
        """I delegate reset to the inner env."""
        obs, info = self._inner.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        I execute one step: convert continuous action, compute penalty, delegate.
        """
        # I convert continuous action to discrete
        discrete_action = self.continuous_to_discrete(action)

        # I compute penalty BEFORE stepping (using current state masks)
        penalty = self._compute_penalty(action, discrete_action)

        # I delegate to inner env
        obs, reward, done, truncated, info = self._inner.step(discrete_action)

        # I apply penalty to reward (using inner env's reward_scale)
        reward_scale = self._inner.reward_calculator.reward_scale
        scaled_penalty = penalty * reward_scale
        reward -= scaled_penalty

        # I add penalty info for logging
        info['sac_penalty'] = penalty
        info['sac_penalty_scaled'] = scaled_penalty
        info['discrete_action'] = discrete_action.tolist()

        return obs, reward, done, truncated, info

    def render(self):
        """I delegate rendering to the inner env."""
        return self._inner.render()

    def close(self):
        """I delegate close to the inner env."""
        return self._inner.close()

    def action_masks(self):
        """I expose inner env's action masks for logging/analysis."""
        return self._inner.action_masks()
