import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class BatteryTradingEnvV2(gym.Env):
    """
    Custom Environment for Battery Trading compatible with OpenAI Gymnasium.
    Simulates a standalone battery storage system participating in Day-Ahead (DAM) and Intraday markets.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame, battery_params: Dict[str, float], initial_soc: float = 0.5):
        super().__init__()
        
        self.df = df
        # Normalize columns to lowercase for consistency
        self.df.columns = [c.lower() for c in self.df.columns]
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        
        # Battery Parameters
        self.capacity_mwh = battery_params.get('capacity_mwh', 1.0)
        self.max_charge_mw = battery_params.get('max_charge_mw', 1.0)
        self.max_discharge_mw = battery_params.get('max_discharge_mw', 1.0)
        self.efficiency = battery_params.get('efficiency', 0.9)
        self.efficiency = battery_params.get('efficiency', 0.9)
        self.cost_per_mwh_throughput = battery_params.get('cost_per_mwh_throughput', 0.0)
        self.transaction_fee = battery_params.get('transaction_fee', 0.0)
        
        # Adversarial Params (Phase 6)
        # Market Impact: Drop/Rise in Price per MW traded. e.g. 0.001 (0.1%) per MW.
        self.market_impact = battery_params.get('market_impact', 0.0)
        # Liquidity Risk: Probability that balancing trade fails (0..1)
        self.liquidity_risk = battery_params.get('liquidity_risk', 0.0)
        # Forecast Noise: Std Dev of noise added to observation features
        self.observation_noise = battery_params.get('observation_noise', 0.0)
        
        # Action Space: (2,)
        # 0: DAM Commitment (MW) [-1, 1]
        # 1: Physical Flow Request (MW) [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation Space - Dynamic
        # We need to define size based on what features are expected.
        # SoC + features list below
        # Defined later in _get_observation, but we need size here.
        # Let's define features here to avoid mismatch.
        self.features = [
            'price', 'load_forecast', 'residual_load_forecast', 
            # 'short', 'long', # REMOVED LEAKAGE: Agent must infer risk!
            'solar', 'wind_onshore',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        feature_count = 1 + len(self.features) # SoC + features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(feature_count,), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.max_steps = len(self.df) - 1
        
    def set_cost(self, cost: float):
        """Update degradation cost for curriculum learning."""
        self.cost_per_mwh_throughput = cost

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_soc = self.initial_soc
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Parse Action
        # Handle both 1D (Legacy/Fallback) and 2D (Multi-Market)
        if action.shape == (1,):
            dam_mw = 0.0 # No DAM commitment
            phy_req_mw = float(action[0])
        else:
            dam_mw = float(action[0])
            phy_req_mw = float(action[1])
        
        # 2. Scale Actions
        # In this env, 1.0 = MaxPower.
        dam_mw_scaled = dam_mw * (self.max_discharge_mw if dam_mw > 0 else self.max_charge_mw)
        # Note: Usually DAM is symmetrical bid, let's assume max_discharge for sizing or just raw 1.0=1MW.
        # Let's stick to: Action is fraction of Max Power.
        
        phy_mw_scaled = phy_req_mw * (self.max_discharge_mw if phy_req_mw > 0 else self.max_charge_mw)
        
        # 3. Apply Physics to Physical Request
        # (ActionMasker helps, but we enforce strictly here too)
        
        # Calculate max possible charge/discharge based on SoC
        # Charge limit: (1.0 - soc) * Cap / Eff
        # Discharge limit: soc * Cap * Eff
        
        # Actually, simpler:
        # If charging:
        # Max Energy In = (MAX_SOC - current_soc) * Cap / Eff_sqrt
        # New Constraint: MAX_SOC = 0.95
        max_soc_limit = 0.95
        min_soc_limit = 0.05
        
        max_energy_in = max(0.0, (max_soc_limit - self.current_soc) * self.capacity_mwh / (np.sqrt(self.efficiency)))
        
        # If discharging:
        # Max Energy Out = (current_soc - MIN_SOC) * Cap * Eff_sqrt
        # New Constraint: MIN_SOC = 0.05
        max_energy_out = max(0.0, (self.current_soc - min_soc_limit) * self.capacity_mwh * np.sqrt(self.efficiency))
        
        # Clip Physical Flow
        if phy_mw_scaled < 0: # Charging
             # Limited by Max Charge Rate AND Max Energy Capacity
             limit = max(-self.max_charge_mw, -max_energy_in)
             actual_phy_mw = max(phy_mw_scaled, limit) # closer to 0
        else: # Discharging
             limit = min(self.max_discharge_mw, max_energy_out)
             actual_phy_mw = min(phy_mw_scaled, limit)
             
        # 4. Update SoC
        eff_factor = np.sqrt(self.efficiency)
        if actual_phy_mw < 0: # Charge
            soc_change = (abs(actual_phy_mw) * eff_factor) / self.capacity_mwh
            self.current_soc += soc_change
        else: # Discharge
            soc_change = -(actual_phy_mw / eff_factor) / self.capacity_mwh
            self.current_soc += soc_change
            
        # 5% - 95% Hard Clip
        self.current_soc = np.clip(self.current_soc, 0.0, 1.0) # Physical limit is still 0-1, but logic above prevents moving past 0.05/0.95
        
        # 5. Financial Settlement
        market_data = self.df.iloc[self.current_step]
        base_dam_price = market_data['price']
        
        # Apply Market Impact (Slippage) on DAM
        # If Buying (dam_mw > 0 ?? No. DAM bid > 0 is usually sell in this env? 
        # Check step(): dam_mw_scaled = dam_mw * max_discharge.
        # usually +ve is discharge (sell), -ve is charge (buy).
        # Sell -> Supply Up -> Price Down. 
        # Buy -> Demand Up -> Price Up.
        # Formula: Realized = Base * (1 - impact * signed_MW)
        # If Sell (+50): 1 - 0.01 * 50 = 1 - 0.5 = 0.5x Price (Crash)
        # If Buy (-50): 1 - 0.01 * (-50) = 1 + 0.5 = 1.5x Price (Spike)
        slippage_factor = 1.0 - (self.market_impact * dam_mw_scaled)
        dam_price = base_dam_price * slippage_factor
        
        # ---------------------------------------------------------
        # FINAL SAFETY GUARDRAIL REMOVED
        # We allow the agent to commit (dam_mw) whatever it wants.
        # If it cannot deliver physically, it will face Imbalance Settlement below.
        # This allows the Policy to LEARN the constraints via penalties.
        # ---------------------------------------------------------
        
        # Imbalance Calculation
        # Imbalance = Actual - Scheduled
        # If Actual > Scheduled (Long): We produced more / consumed less.
        #   Note: Charging is negative generation (Consumption).
        #   Example: DAM = -10 (Buy 10). Actual = -8 (Buy 8). Imbalance = -8 - (-10) = +2. Long. Correct.
        #   Example: DAM = 10 (Sell 10). Actual = 12 (Sell 12). Imbalance = 12 - 10 = +2. Long. Correct.
        imbalance_mw = actual_phy_mw - dam_mw_scaled
        
        # Determine Imbalance Price
        # If Net System is Short (needs energy), Price is High (Up Regulation).
        # If Net System is Long (excess energy), Price is Low (Down Regulation).
        # We don't know Net System State perfectly, but we have 'Long' and 'Short' columns.
        # 'Long': Price paid to TSO for surplus? Or Price TSO pays us?
        ### ENTSO-E: 
        # "Positive Imbalance Price" (Long): Price for pos imbalance.
        # "Negative Imbalance Price" (Short): Price for neg imbalance.
        
        # Let's assume columns 'long' and 'short' are these prices.
        # Usually: Short Price >= Long Price (Spread).
        # If I am Long (Imbalance > 0): I sell spill to TSO. usually at Long Price (Lower).
        # If I am Short (Imbalance < 0): I buy deficit from TSO. usually at Short Price (Higher).
        
        long_price = market_data.get('long', dam_price * 0.5) # Fallback
        short_price = market_data.get('short', dam_price * 1.5) # Fallback
        
        # Revenue Streams
        # 1. DAM Revenue
        dam_revenue = dam_mw_scaled * dam_price
        
        # 2. Balancing Revenue/Cost
        # 2. Balancing Revenue/Cost
        bal_revenue = 0.0
        
        # Apply Liquidity Risk to Balancing
        # If we have a trade (imbalance != 0)
        trade_executed = True
        if abs(imbalance_mw) > 0.1 and self.liquidity_risk > 0:
            if np.random.random() < self.liquidity_risk:
                trade_executed = False
                # What happens if rejected?
                # Usually you are forced to pay a penalty or simpler: You get 0 revenue/cost benefit, just stuck.
                # However, imbalance implies physical constraint mismatch.
                # If TSO rejects your balancing OFFER, you are just penalized for imbalance at worst price?
                # Let's simplify: Imbalance happened physically. But TSO didn't pay you for helping (if Long). 
                # Or TSO charged you extra (if Short)?
                # Simplest Adversary: You get 0 Revenue for Positive Imbalance (Spill), but Pay Full for Negative.
                pass
        
        if imbalance_mw > 0: # Long (Surplus)
            if trade_executed:
                bal_revenue = imbalance_mw * long_price
            else:
                bal_revenue = 0.0 # TSO refused to buy your spill. You wasted energy.
        else: # Short (Deficit)
            # You ALWAYS pay for deficit. TSO forces you.
            bal_revenue = imbalance_mw * short_price
            
        # 3. Costs
        # Degradation (on Physical Flow only)
        deg_cost = abs(actual_phy_mw) * self.cost_per_mwh_throughput
        
        # Transaction Fees (on DAM + Balancing? Or just commercial trades?)
        # Typically fees are on cleared volumes.
        # HEnEx: Fees on DAM Trades. Balancing might have different.
        # Let's apply FEE to DAM Vol + Absolute Imbalance Vol (everything processed)
        tx_cost = (abs(dam_mw_scaled) + abs(imbalance_mw)) * self.transaction_fee
        
        reward = dam_revenue + bal_revenue - deg_cost - tx_cost
        
        # 5. Move to next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        observation = self._get_observation()
        
        info = {
            'step': self.current_step,
            'soc': self.current_soc,
            'dam_rev': dam_revenue,
            'bal_rev': bal_revenue,
            'dam_mw': dam_mw_scaled,
            'phy_mw': actual_phy_mw,
            'imbalance_mw': imbalance_mw
        }
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Check if we are at end
        if self.current_step >= len(self.df):
             return np.zeros(self.observation_space.shape, dtype=np.float32)
             
        row = self.df.iloc[self.current_step]
        
        # Construct observation vector dynamically based on available columns
        # Core: SoC
        obs_list = [self.current_soc]
        
        # Features to look for
        # Use pre-defined features list
        features = self.features
        
        for f in features:
            if f in row:
                val = row[f]
                # Scaling REQUIRED to match Training Distribution (VecNormalize stats)
                if 'price' in f:
                    val = val / 100.0
                elif 'load' in f or 'solar' in f or 'wind' in f:
                    val = val / 10000.0
                    
                # Apply Observation Noise (Phase 6)
                if self.observation_noise > 0:
                    # Only apply to forecasts/prices, not sine/cosine?
                    if 'price' in f or 'load' in f or 'solar' in f or 'wind' in f:
                        noise = np.random.normal(0, self.observation_noise)
                        val += noise
                
                # Handle NaNs
                if pd.isna(val): val = 0.0
                obs_list.append(val)
            else:
                # Missing feature -> fill with 0.0 to maintain shape consistency
                obs_list.append(0.0)
        
        obs = np.array(obs_list, dtype=np.float32)
        return obs
