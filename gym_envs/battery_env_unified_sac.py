"""
SAC Wrapper for Unified Multi-Market Battery Trading Environment

I wrap BatteryEnvUnified (composition) to provide a continuous Box action space
for SAC training while reusing all existing market simulation logic.

Key Design Decisions:
1. Composition over inheritance — I hold an inner BatteryEnvUnified instance
2. Continuous actions [-1,1] are discretized to nearest MultiDiscrete index
3. Action filtering: invalid actions are replaced with nearest valid action (Fix C)
4. Soft constraint penalties for SoC margin proximity and cycle excess
5. Inner env's action_masks() enforce hard constraints via filtering, not penalties

Action Space: Box(low=-1, high=1, shape=(N,))
  Legacy (4-dim): [aFRR commit, aFRR price, IntraDay, mFRR]
  Full-market (6-dim): [aFRR commit, aFRR price, IntraDay, mFRR, FreeBid qty, FreeBid price]
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

    # I define penalty costs (EUR) for soft constraint violations.
    # Hard mask violations are eliminated by action filtering (Fix C),
    # so these only apply to soft constraints (SoC margin proximity, cycling).
    DEFAULT_PENALTIES = {
        'soc_soft_margin': 60.0,  # EUR/MW for soft SoC margin (5% buffer)
        'cycle_excess': 2000.0,   # EUR per excess cycle above target (super-linear)
    }

    # I define the target cycling rate (cycles/day)
    CYCLE_TARGET = 1.8

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
        # Legacy: [5,5,11,11], full-market: [5,5,11,11,11,5]
        self._nvec = self._inner.action_space.nvec

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

    # I define dead zone thresholds for SAC's tanh-squashed actions.
    # SAC uses tanh(Normal(0,1)) which creates a U-shaped distribution concentrated
    # near ±1. Without dead zones, only ~5% of actions land in the no-trade zone.
    INTRADAY_DEAD_ZONE = 0.5   # ~42% of random actions map to idle
    MFRR_DEAD_ZONE = 0.3       # ~26% of random actions map to idle
    FREEBID_DEAD_ZONE = 0.3    # ~26% of random actions map to idle

    def continuous_to_discrete(self, continuous_action: np.ndarray) -> np.ndarray:
        """
        I convert continuous [-1, 1] actions to discrete MultiDiscrete indices.

        Mapping for each dimension:
          continuous -1.0 -> discrete index 0
          continuous  0.0 -> discrete index (n-1)/2  (middle)
          continuous +1.0 -> discrete index n-1

        I apply dead zones for IntraDay, mFRR, and FreeBid so that SAC's
        tanh-squashed policy has meaningful no-trade regions.
        """
        continuous_action = np.clip(continuous_action, -1.0, 1.0)
        discrete = np.zeros(len(self._nvec), dtype=np.int64)

        for i, n in enumerate(self._nvec):
            # I map [-1, 1] to [0, n-1] with rounding to nearest index
            idx = (continuous_action[i] + 1.0) / 2.0 * (n - 1)
            discrete[i] = int(np.round(np.clip(idx, 0, n - 1)))

        # I apply dead zones: continuous values near center → no trade (idle)
        if abs(continuous_action[2]) < self.INTRADAY_DEAD_ZONE:
            discrete[2] = self._nvec[2] // 2  # IntraDay idle

        if abs(continuous_action[3]) < self.MFRR_DEAD_ZONE:
            discrete[3] = self._nvec[3] // 2  # mFRR idle

        # I handle FreeBid dims only in full-market mode (6-dim)
        if len(self._nvec) >= 6:
            if abs(continuous_action[4]) < self.FREEBID_DEAD_ZONE:
                discrete[4] = self._nvec[4] // 2  # FreeBid qty idle

        return discrete

    def _filter_to_nearest_valid(self, discrete_action: np.ndarray) -> np.ndarray:
        """
        I replace masked (invalid) discrete actions with the nearest valid action
        per dimension (Fix C). This eliminates hard constraint violations at the
        source, so the agent never executes an illegal action.

        For each dimension, if the chosen index is masked:
        1. I search outward (±1, ±2, ...) for the nearest valid index
        2. If no valid index exists in that dimension, I use the center (idle)
        """
        mask = self._inner.action_masks()
        offsets = np.cumsum([0] + list(self._nvec))
        filtered = discrete_action.copy()

        for dim_idx in range(len(self._nvec)):
            start = offsets[dim_idx]
            end = offsets[dim_idx + 1]
            dim_mask = mask[start:end]
            chosen = filtered[dim_idx]
            n = self._nvec[dim_idx]

            if dim_mask[chosen]:
                # I keep the action — it's valid
                continue

            # I search outward for the nearest valid action
            best_idx = None
            for delta in range(1, n):
                for candidate in [chosen - delta, chosen + delta]:
                    if 0 <= candidate < n and dim_mask[candidate]:
                        best_idx = candidate
                        break
                if best_idx is not None:
                    break

            if best_idx is not None:
                filtered[dim_idx] = best_idx
            else:
                # I fall back to center (idle) — should always be valid
                filtered[dim_idx] = n // 2

        return filtered

    def _compute_penalty(self, discrete_action: np.ndarray) -> float:
        """
        I compute soft constraint penalties only. Hard mask violations are
        eliminated by _filter_to_nearest_valid() (Fix C), so I only penalize:
        1. SoC margin proximity (approaching min/max SoC)
        2. Daily cycle excess (super-linear penalty above target)
        """
        penalty = 0.0

        # I compute total discharge/charge intensity across trading dims
        id_level = self._inner.INTRADAY_LEVELS[discrete_action[2]]
        mfrr_level = self._inner.MFRR_LEVELS[discrete_action[3]]
        discharge_intensity = max(0, id_level) + max(0, mfrr_level)
        charge_intensity = max(0, -id_level) + max(0, -mfrr_level)

        # I include FreeBid in intensity calculation for full-market mode
        if len(self._nvec) >= 6:
            freebid_level = self._inner.FREEBID_QTY_LEVELS[discrete_action[4]]
            discharge_intensity += max(0, freebid_level)
            charge_intensity += max(0, -freebid_level)

        # I add soft SoC margin penalty (penalize approaching limits)
        soc = self._inner.soc
        margin = 0.05  # 5% buffer zone
        if soc < self._inner.min_soc + margin and discharge_intensity > 0:
            proximity = 1.0 - (soc - self._inner.min_soc) / margin
            proximity = np.clip(proximity, 0, 1)
            penalty += self.penalties['soc_soft_margin'] * proximity * discharge_intensity

        elif soc > self._inner.max_soc - margin and charge_intensity > 0:
            proximity = 1.0 - (self._inner.max_soc - soc) / margin
            proximity = np.clip(proximity, 0, 1)
            penalty += self.penalties['soc_soft_margin'] * proximity * charge_intensity

        # I add daily cycle excess penalty with super-linear scaling (Fix B).
        # Small excess (0.2 over target) is gentle; large excess (4.1 over) is severe.
        # Formula: penalty = base * excess^1.5
        if self._inner.daily_cycles > self.CYCLE_TARGET:
            excess = self._inner.daily_cycles - self.CYCLE_TARGET
            penalty += self.penalties['cycle_excess'] * (excess ** 1.5)

        return penalty

    def reset(self, seed=None, options=None):
        """I delegate reset to the inner env."""
        obs, info = self._inner.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        I execute one step:
        1. Convert continuous action to discrete indices
        2. Filter invalid actions to nearest valid (Fix C — zero hard violations)
        3. Compute soft penalties (SoC margin, cycling)
        4. Delegate to inner env with guaranteed-valid action
        """
        # I convert continuous action to discrete
        discrete_action = self.continuous_to_discrete(action)

        # I filter to nearest valid action (Fix C — eliminates hard violations)
        filtered_action = self._filter_to_nearest_valid(discrete_action)

        # I compute soft penalty BEFORE stepping (SoC margin + cycling only)
        penalty = self._compute_penalty(filtered_action)

        # I delegate to inner env with guaranteed-valid action
        obs, reward, done, truncated, info = self._inner.step(filtered_action)

        # I apply penalty to reward (using inner env's reward_scale)
        reward_scale = self._inner.reward_calculator.reward_scale
        scaled_penalty = penalty * reward_scale
        reward -= scaled_penalty

        # I add penalty info for logging
        info['sac_penalty'] = penalty
        info['sac_penalty_scaled'] = scaled_penalty
        info['discrete_action'] = filtered_action.tolist()
        info['action_was_filtered'] = not np.array_equal(discrete_action, filtered_action)

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

    def set_degradation_cost(self, cost: float):
        """I set degradation cost for curriculum learning (SubprocVecEnv compatible)."""
        self._inner.reward_calculator.degradation_cost = cost

    def set_penalties(self, penalty_overrides: Dict[str, float]):
        """I update penalty values at runtime (for curriculum penalty scheduling)."""
        self.penalties.update(penalty_overrides)
