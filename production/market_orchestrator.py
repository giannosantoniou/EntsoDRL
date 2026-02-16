"""
4-Model Production Orchestrator

I orchestrate the 4 decomposed market models in correct timeline order:
DAM (D-1 12:00) -> aFRR (per 4h block) -> IntraDay (per ISP) -> mFRR (per ISP)

Each model runs at its correct decision point with the right information
barriers and capacity cascading.

Usage:
    from production.market_orchestrator import MarketOrchestrator

    orchestrator = MarketOrchestrator(
        dam_path="models/decomposed/dam/best_model.zip",
        afrr_path="models/decomposed/afrr/best_model.zip",
        id_path="models/decomposed/intraday/best_model.zip",
        mfrr_path="models/decomposed/mfrr/best_model.zip",
    )

    # At D-1 12:00
    dam_schedule = orchestrator.run_dam_decision(forecast_data)

    # Every 4h block
    afrr_cap, afrr_price = orchestrator.run_afrr_decision(market_data, block_idx)

    # Every ISP (15 min)
    decisions = orchestrator.run_isp_decisions(market_data, isp)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

from gym_envs.shared_battery_state import SharedBatteryState
from gym_envs.battery_env_dam import BatteryEnvDAM, DAM_LEVELS
from gym_envs.battery_env_afrr import BatteryEnvAFRR, AFRR_CAP_LEVELS, AFRR_PRICE_TIERS
from gym_envs.battery_env_intraday import BatteryEnvIntraDay, ID_LEVELS
from gym_envs.battery_env_mfrr import BatteryEnvMFRR, MFRR_LEVELS, MFRR_PRICE_TIERS

warnings.filterwarnings('ignore')


class MarketOrchestrator:
    """
    I orchestrate 4 models in correct market-timeline order.

    I ensure:
    1. Each model runs at its correct decision point
    2. Information barriers are respected (no future data leakage)
    3. Capacity cascading flows correctly (DAM -> aFRR -> ID -> mFRR)
    4. Shared battery state is updated consistently
    """

    def __init__(
        self,
        dam_path: str,
        afrr_path: str,
        id_path: str,
        mfrr_path: str,
        dam_vecnorm_path: Optional[str] = None,
        afrr_vecnorm_path: Optional[str] = None,
        id_vecnorm_path: Optional[str] = None,
        mfrr_vecnorm_path: Optional[str] = None,
        battery_config: Optional[Dict] = None,
    ):
        """
        I load all 4 MaskablePPO models and their VecNormalize wrappers.

        Args:
            dam_path: Path to DAM model .zip
            afrr_path: Path to aFRR model .zip
            id_path: Path to IntraDay model .zip
            mfrr_path: Path to mFRR model .zip
            dam_vecnorm_path: Path to DAM VecNormalize .pkl
            afrr_vecnorm_path: Path to aFRR VecNormalize .pkl
            id_vecnorm_path: Path to IntraDay VecNormalize .pkl
            mfrr_vecnorm_path: Path to mFRR VecNormalize .pkl
            battery_config: Battery parameters dict
        """
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import VecNormalize

        # I load all 4 models
        self.dam_model = MaskablePPO.load(dam_path)
        self.afrr_model = MaskablePPO.load(afrr_path)
        self.id_model = MaskablePPO.load(id_path)
        self.mfrr_model = MaskablePPO.load(mfrr_path)

        # I load VecNormalize stats if available
        self.dam_vecnorm = self._load_vecnorm(dam_vecnorm_path)
        self.afrr_vecnorm = self._load_vecnorm(afrr_vecnorm_path)
        self.id_vecnorm = self._load_vecnorm(id_vecnorm_path)
        self.mfrr_vecnorm = self._load_vecnorm(mfrr_vecnorm_path)

        # I create shared battery state
        config = battery_config or {}
        self.battery = SharedBatteryState(
            capacity_mwh=config.get('capacity_mwh', 146.0),
            max_power_mw=config.get('max_power_mw', 30.0),
            efficiency=config.get('efficiency', 0.94),
        )

        # I track daily decisions
        self.daily_log = []

    def _load_vecnorm(self, path: Optional[str]):
        """I load VecNormalize stats for observation normalization."""
        if path is None:
            return None
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            # I load the stats only (mean/var) for normalization
            import pickle
            with open(path, 'rb') as f:
                vecnorm = pickle.load(f)
            return vecnorm
        except Exception as e:
            print(f"  Warning: Could not load VecNormalize from {path}: {e}")
            return None

    def _normalize_obs(self, obs: np.ndarray, vecnorm) -> np.ndarray:
        """I normalize an observation using VecNormalize statistics."""
        if vecnorm is None:
            return obs
        try:
            # I use the running mean/var from VecNormalize
            obs_mean = vecnorm.obs_rms.mean
            obs_var = vecnorm.obs_rms.var
            clip = vecnorm.clip_obs
            normalized = np.clip(
                (obs - obs_mean) / np.sqrt(obs_var + 1e-8),
                -clip, clip
            )
            return normalized.astype(np.float32)
        except Exception:
            return obs

    def run_dam_decision(self, forecast_data: pd.DataFrame) -> np.ndarray:
        """
        I run the DAM model at D-1 12:00 to produce a 24-hour schedule.

        Args:
            forecast_data: DataFrame with price forecasts for 24 hours

        Returns:
            24-element array of hourly MW commitments
        """
        # I create a temporary DAM environment for observation building
        env = BatteryEnvDAM(df=forecast_data, random_start=False)
        obs, _ = env.reset()
        schedule = np.zeros(24, dtype=np.float32)

        for h in range(24):
            # I normalize observation
            norm_obs = self._normalize_obs(obs, self.dam_vecnorm)

            # I get action masks
            masks = env.action_masks()

            # I predict action using frozen model
            action, _ = self.dam_model.predict(
                norm_obs, deterministic=True, action_masks=masks
            )

            obs, reward, terminated, truncated, info = env.step(int(action))
            schedule[h] = DAM_LEVELS[int(action)]

            if terminated:
                break

        # I store in shared state
        self.battery.dam_schedule = schedule.copy()
        return schedule

    def run_afrr_decision(
        self, market_data: pd.DataFrame, block_idx: int
    ) -> Tuple[float, int]:
        """
        I run the aFRR model before each 4h block.

        Args:
            market_data: Current market data
            block_idx: Block index (0-5)

        Returns:
            (capacity_mw, price_tier_idx)
        """
        # I create temporary aFRR environment
        env = BatteryEnvAFRR(df=market_data, random_start=False)
        obs, _ = env.reset()

        # I sync battery state
        env.battery = self.battery.clone()

        # I build observation for this block
        obs = env._build_observation()
        norm_obs = self._normalize_obs(obs, self.afrr_vecnorm)
        masks = env.action_masks()

        action, _ = self.afrr_model.predict(
            norm_obs, deterministic=True, action_masks=masks
        )

        cap_idx, price_idx = int(action[0]), int(action[1])

        # I calculate actual commitment
        hour = block_idx * 4
        remaining_d, remaining_c = self.battery.remaining_capacity('afrr', hour=hour)
        remaining = min(remaining_d, remaining_c)
        capacity_mw = AFRR_CAP_LEVELS[cap_idx] * remaining

        # I store in shared state
        self.battery.afrr_commitments[block_idx] = capacity_mw
        self.battery.afrr_price_tiers[block_idx] = price_idx

        return capacity_mw, price_idx

    def run_isp_decisions(
        self, market_data: pd.DataFrame, isp: int
    ) -> Dict:
        """
        I execute all market actions for a single ISP (15 min).

        Order: DAM execution -> aFRR check -> IntraDay trade -> mFRR bid

        Args:
            market_data: Current market data
            isp: ISP index within the day (0-95)

        Returns:
            Dict with all decisions and results
        """
        hour = isp // 4
        block = hour // 4
        results = {'isp': isp, 'hour': hour, 'block': block}

        # STAGE 1: Execute DAM commitment (mandatory)
        dam_mw = self.battery.dam_schedule[hour]
        dam_fraction = dam_mw / 4  # 4 ISPs per hour
        if abs(dam_fraction) > 0.01:
            actual_dam = self.battery.apply_energy(dam_fraction, 0.25)
            results['dam_executed_mw'] = actual_dam

        # STAGE 2: Check aFRR activation (from TSO - external event)
        # In production, this would come from TSO signal
        results['afrr_activated'] = self.battery.afrr_is_activated
        results['afrr_activated_mw'] = self.battery.afrr_activated_mw

        # STAGE 3: IntraDay trade (if not blocked by aFRR)
        if not self.battery.afrr_is_activated:
            env = BatteryEnvIntraDay(df=market_data, random_start=False)
            obs, _ = env.reset()
            env.battery = self.battery.clone()
            obs = env._build_observation()
            norm_obs = self._normalize_obs(obs, self.id_vecnorm)
            masks = env.action_masks()

            action, _ = self.id_model.predict(
                norm_obs, deterministic=True, action_masks=masks
            )

            level = ID_LEVELS[int(action)]
            remaining_d, remaining_c = self.battery.remaining_capacity(
                'intraday', isp=isp, hour=hour
            )

            if level >= 0:
                id_mw = level * remaining_d
            else:
                id_mw = level * remaining_c

            if abs(id_mw) > 0.01:
                actual_id = self.battery.apply_energy(id_mw, 0.25)
                self.battery.intraday_positions[isp] = id_mw
                results['intraday_mw'] = actual_id
        else:
            results['intraday_mw'] = 0.0

        # STAGE 4: mFRR Free Bid
        env = BatteryEnvMFRR(df=market_data, random_start=False)
        obs, _ = env.reset()
        env.battery = self.battery.clone()
        obs = env._build_observation()
        norm_obs = self._normalize_obs(obs, self.mfrr_vecnorm)
        masks = env.action_masks()

        action, _ = self.mfrr_model.predict(
            norm_obs, deterministic=True, action_masks=masks
        )

        qty_idx, price_idx = int(action[0]), int(action[1])
        results['mfrr_qty_level'] = float(MFRR_LEVELS[qty_idx])
        results['mfrr_price_tier'] = float(MFRR_PRICE_TIERS[price_idx])

        results['soc'] = self.battery.soc
        results['daily_cycles'] = self.battery.daily_cycles

        self.daily_log.append(results)
        return results

    def run_daily_cycle(self, data_provider) -> Dict:
        """
        I orchestrate a full day D timeline.

        Args:
            data_provider: Data source providing market data

        Returns:
            Dict with day summary
        """
        self.daily_log = []

        # I get forecast data for DAM (D-1 12:00)
        forecast_data = data_provider.get_dam_forecast()
        dam_schedule = self.run_dam_decision(forecast_data)

        # I run 6 aFRR blocks
        for block in range(6):
            market_data = data_provider.get_block_data(block)
            self.run_afrr_decision(market_data, block)

        # I run 96 ISPs
        for isp in range(96):
            market_data = data_provider.get_isp_data(isp)

            # I check for aFRR activation (from external TSO signal)
            if data_provider.is_afrr_activated(isp):
                self.battery.afrr_is_activated = True
                afrr_mw = self.battery.afrr_commitments[isp // 16]
                self.battery.afrr_activated_mw = afrr_mw
                # I execute aFRR activation
                self.battery.apply_energy(afrr_mw, 0.25)
            else:
                self.battery.afrr_is_activated = False
                self.battery.afrr_activated_mw = 0.0

            self.run_isp_decisions(market_data, isp)

        # I reset daily counters at end of day
        self.battery.check_day_boundary(0, 23)

        return {
            'dam_schedule': dam_schedule.tolist(),
            'afrr_commitments': self.battery.afrr_commitments.tolist(),
            'final_soc': self.battery.soc,
            'total_cycles': self.battery.total_cycles,
            'n_decisions': len(self.daily_log),
        }

    def get_status(self) -> Dict:
        """I return current battery and commitment status."""
        return {
            'soc': self.battery.soc,
            'total_cycles': self.battery.total_cycles,
            'daily_cycles': self.battery.daily_cycles,
            'dam_schedule': self.battery.dam_schedule.tolist(),
            'afrr_commitments': self.battery.afrr_commitments.tolist(),
            'afrr_is_activated': self.battery.afrr_is_activated,
        }
