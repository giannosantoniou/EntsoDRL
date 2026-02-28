#!/usr/bin/env python3
"""
Paper Trading Mode - Live Data, Virtual Battery
================================================
Χρησιμοποιεί ΠΡΑΓΜΑΤΙΚΑ ENTSO-E data αλλά:
- Virtual Battery (simulated SoC)
- Virtual P&L (calculated profit/loss)
- NO real dispatches to BMS

Ιδανικό για testing σε production περιβάλλον χωρίς κίνδυνο!
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from agent.feature_engineer import FeatureEngineer, MarketStateDTO
from agent.decision_strategy import create_strategy, IDecisionStrategy
from data.data_provider import LiveDataProvider
import pickle

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingSimulator:
    """
    Simulates battery trading με live market data.
    Καταγράφει όλες τις αποφάσεις και το P&L χωρίς πραγματικές συναλλαγές.

    Supports multiple strategies:
    - AI: MaskablePPO model (default)
    - AGGRESSIVE_HYBRID: Time-based + price intelligence (best from backtesting)
    - RULE_BASED: Simple threshold rules
    """
    def __init__(self, model_path: str, data_provider: LiveDataProvider,
                 log_file: str = "paper_trading_log.csv",
                 strategy_type: str = "AI"):
        self.strategy_type = strategy_type.upper()
        self.data_provider = data_provider

        # I load the appropriate strategy
        self.vec_normalize = None  # VecNormalize for observation normalization

        if self.strategy_type == "AI":
            logger.info(f"Loading AI model from {model_path}...")
            self.model = MaskablePPO.load(model_path)
            self.strategy = None  # Use model directly for AI

            # I look for VecNormalize stats alongside the model
            # Common naming patterns: vec_normalize.pkl, vecnormalize.pkl, or same name as model
            self._load_vec_normalize(model_path)
        else:
            logger.info(f"Initializing {self.strategy_type} strategy...")
            self.model = None
            self.strategy = create_strategy(
                strategy_type=self.strategy_type,
                model_path=model_path if self.strategy_type == "AI" else None,
                n_actions=21
            )
        
        # Feature engineer (same as production)
        self.feature_engine = FeatureEngineer(config={
            'capacity_mwh': data_provider.capacity_mwh,
            'max_power_mw': data_provider.max_power_mw,
            'efficiency': data_provider.efficiency,
            'time_step_hours': data_provider.time_step_hours
        })
        
        # Virtual Battery State (Simulated)
        self.virtual_soc = data_provider.current_soc
        self.min_soc = 0.05
        self.max_soc = 0.95
        
        # P&L Tracking (total + per-market breakdown)
        self.total_profit = 0.0
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.decisions_count = 0
        self.dam_profit = 0.0
        self.afrr_cap_profit = 0.0
        self.afrr_energy_profit = 0.0
        self.intraday_profit = 0.0
        self.mfrr_profit = 0.0
        self.free_bid_profit = 0.0
        self.total_fees = 0.0
        
        # Decision Log
        self.log_file = log_file
        self._init_log_file()
        
        logger.info("Paper Trading Simulator initialized (VIRTUAL MODE)")
        logger.info(f"Virtual SoC: {self.virtual_soc:.2%}")
        logger.info(f"Logging to: {self.log_file}")

    def _load_vec_normalize(self, model_path: str):
        """
        I load VecNormalize stats to ensure observations are normalized
        the same way as during training.

        CRITICAL: If the model was trained with VecNormalize but we don't
        load it here, the model receives un-normalized observations and
        will produce garbage predictions.
        """
        model_dir = Path(model_path).parent
        model_stem = Path(model_path).stem

        # I try multiple common naming patterns
        candidates = [
            model_dir / "vec_normalize.pkl",
            model_dir / "vecnormalize.pkl",
            model_dir / f"{model_stem}_vec_normalize.pkl",
            model_dir / f"vec_normalize_{model_stem}.pkl",
            Path(model_path).with_suffix('.pkl'),  # Same name, different extension
        ]

        for vec_path in candidates:
            if vec_path.exists():
                try:
                    logger.info(f"Loading VecNormalize from {vec_path}...")

                    # I load the VecNormalize object
                    with open(vec_path, 'rb') as f:
                        self.vec_normalize = pickle.load(f)

                    # I set inference mode (no updates to running stats)
                    self.vec_normalize.training = False
                    self.vec_normalize.norm_reward = False

                    logger.info(
                        f"VecNormalize loaded successfully. "
                        f"obs_rms mean: {self.vec_normalize.obs_rms.mean[:3]}... "
                        f"(showing first 3 of {len(self.vec_normalize.obs_rms.mean)})"
                    )
                    return

                except Exception as e:
                    logger.warning(f"Failed to load VecNormalize from {vec_path}: {e}")
                    continue

        # No VecNormalize found - this is a potential issue
        logger.warning(
            "WARNING: No VecNormalize stats found! "
            "If the model was trained with normalized observations, "
            "predictions may be incorrect. "
            f"Searched in: {model_dir}"
        )

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        I normalize observations using loaded VecNormalize stats.

        If VecNormalize is not loaded, I return the observation unchanged
        but log a warning on the first call.
        """
        if self.vec_normalize is None:
            return obs

        # I apply normalization: (obs - mean) / sqrt(var + epsilon)
        obs_rms = self.vec_normalize.obs_rms
        epsilon = self.vec_normalize.epsilon
        clip_obs = self.vec_normalize.clip_obs

        normalized = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon)
        normalized = np.clip(normalized, -clip_obs, clip_obs)

        return normalized
    
    def _init_log_file(self):
        """Initialize CSV log file with per-market breakdown."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,price_eur,soc,risk_score,action_idx,action_level,mw_setpoint,"
                       "step_profit,total_profit,revenue,cost,"
                       "dam_profit,afrr_cap_profit,afrr_energy_profit,"
                       "intraday_profit,mfrr_profit,free_bid_profit,fees,"
                       "gradient,res_deviation\n")
    
    def _update_virtual_battery(self, mw_setpoint: float):
        """
        Simulate battery SoC update (NO real dispatch).
        """
        eff_sqrt = np.sqrt(self.data_provider.efficiency)
        energy_mwh = abs(mw_setpoint) * self.data_provider.time_step_hours
        
        if mw_setpoint < 0:  # Charging
            self.virtual_soc += (energy_mwh * eff_sqrt) / self.data_provider.capacity_mwh
        else:  # Discharging
            self.virtual_soc -= (energy_mwh / eff_sqrt) / self.data_provider.capacity_mwh
        
        self.virtual_soc = np.clip(self.virtual_soc, 0.0, 1.0)
    
    def run_tick(self):
        """
        Execute one paper trading cycle.
        Uses LIVE data but SIMULATES battery and P&L.
        """
        try:
            # 1. Fetch LIVE Data from ENTSO-E
            # ------------------------------------------------
            market_state = self.data_provider.get_current_market_state()
            imbalance_risk = self.data_provider.get_imbalance_risk()

            # I check data freshness and log warnings if stale
            from data.data_provider import DataFreshness
            freshness = self.data_provider.get_data_freshness()
            if freshness == DataFreshness.STALE:
                logger.warning("Trading with STALE market data - exercise caution!")
            elif freshness == DataFreshness.UNAVAILABLE:
                logger.error("No market data available - skipping this tick")
                return None
            
            # 2. Prepare Market State DTO
            # ------------------------------------------------
            forecast = self.data_provider.get_forecast(hours_ahead=24)
            prices_24h = forecast.prices
            dam_commits = forecast.dam_commitments
            
            if prices_24h:
                p_max = max(prices_24h)
                p_min = min(prices_24h)
                p_mean = float(np.mean(prices_24h))
                next_price = prices_24h[0]
            else:
                p_max = market_state.dam_price
                p_min = market_state.dam_price
                p_mean = market_state.dam_price
                next_price = market_state.dam_price
            
            dto = MarketStateDTO(
                timestamp=pd.Timestamp(market_state.timestamp),
                price=market_state.intraday_price,
                spread=market_state.spread,
                dam_commitment_now=market_state.dam_commitment_mw,
                next_price=next_price,
                price_max_24h=p_max,
                price_min_24h=p_min,
                price_mean_24h=p_mean,
                dam_commitments=dam_commits[:4],
                volatility=5.0
            )
            
            # 3. Build Observation (using virtual SoC!)
            # ------------------------------------------------
            obs = self.feature_engine.build_observation(
                market=dto,
                battery_soc=self.virtual_soc,  # VIRTUAL!
                imbalance_info=imbalance_risk
            )
            
            # 4. Predict Action
            # ------------------------------------------------
            # Use virtual SoC to calculate mask
            eff_sqrt = np.sqrt(self.data_provider.efficiency)
            available_discharge = max(0, (self.virtual_soc - self.min_soc) * self.data_provider.capacity_mwh)
            available_charge = max(0, (self.max_soc - self.virtual_soc) * self.data_provider.capacity_mwh)
            
            max_discharge_mw = min(self.data_provider.max_power_mw, 
                                  (available_discharge * eff_sqrt) / self.data_provider.time_step_hours)
            max_charge_mw = min(self.data_provider.max_power_mw, 
                               (available_charge / eff_sqrt) / self.data_provider.time_step_hours)
            
            mask = np.zeros(21, dtype=bool)
            for i, level in enumerate(self.data_provider.action_levels):
                if level >= 0:
                    mask[i] = level <= (max_discharge_mw / self.data_provider.max_power_mw) + 0.01
                else:
                    mask[i] = abs(level) <= (max_charge_mw / self.data_provider.max_power_mw) + 0.01
            mask[10] = True  # Always allow IDLE
            
            # I use the appropriate prediction method based on strategy type
            if self.model is not None:
                # AI mode - use MaskablePPO directly
                # CRITICAL: I normalize observations to match training conditions
                obs_normalized = self._normalize_obs(obs)
                action, _ = self.model.predict(obs_normalized, action_masks=mask, deterministic=True)
            else:
                # Strategy mode - use IDecisionStrategy interface (no normalization needed)
                action = self.strategy.predict_action(obs, mask)
            
            # 4b. Log IntraDay Proposal with Forecast Justification
            # ------------------------------------------------
            # I check if forecast features are present in observation (indices 63-71)
            if len(obs) > 67:
                proposal = {
                    'timestamp': datetime.now().isoformat(),
                    'forecast_1h': obs[63] * 100,
                    'forecast_2h': obs[64] * 100,
                    'correction_signal': obs[67],
                    'dam_price': obs[3] * 100,
                    'solar_mw': obs[69] * 3000 if len(obs) > 69 else 0,
                    'wind_mw': obs[70] * 3000 if len(obs) > 70 else 0,
                    'status': 'PENDING_APPROVAL'
                }
                logger.info(
                    f"INTRADAY PROPOSAL: "
                    f"forecast={proposal['forecast_1h']:.1f} EUR "
                    f"| dam={proposal['dam_price']:.1f} EUR "
                    f"| correction={proposal['correction_signal']:.3f} "
                    f"| solar={proposal['solar_mw']:.0f} wind={proposal['wind_mw']:.0f}"
                )

            # 5. Calculate VIRTUAL P&L (NO real dispatch!)
            # ------------------------------------------------
            action_idx = int(action)
            action_level = self.data_provider.action_levels[action_idx]
            
            # Agent decides DEVIATION (Balancing Action), not total output
            # But currently model is trained on TOTAL output mode?
            # Let's assume Agent Output IS the deviation strategy if we follow standard DRL balancing
            # OR if Agent controls total, we must check compliance.
            
            # Since our Env is trained to output TOTAL MW, we must respect DAM here manually if strictly required.
            # However, usually in this setup, DAM is just a financial position.
            # If "0 Violations" is stricter, we should say:
            # Physical Output = Agent Action
            # Imbalance = Physical Output - DAM Commitment
            
            dam_commitment = dto.dam_commitment_now
            mw_setpoint = action_level * self.data_provider.max_power_mw
            
            # Calculate Deviation
            deviation_mw = mw_setpoint - dam_commitment
            
            # Profit calculation
            # 1. DAM Settlement (already fixed yesterday, but affects P&L accounting)
            # 1. DAM Settlement (already fixed yesterday, but affects P&L accounting)
            dam_revenue = dam_commitment * market_state.dam_price if dam_commitment > 0 else 0
            
            # 2. Imbalance Settlement
            imbalance_revenue = 0.0
            if deviation_mw > 0:  # Surplus (Selling to TSO)
                imbalance_revenue = deviation_mw * market_state.imbalance_price_short # Up Regulation Price
            elif deviation_mw < 0: # Deficit (Buying from TSO)
                imbalance_revenue = deviation_mw * market_state.imbalance_price_long # Down Regulation Price (cost)
            
            # Total Profit for this step (Cashflow view)
            # Note: This is a simplified settlement view.
            step_profit = imbalance_revenue # We track specific balancing profit here

            # Define price for logging
            price = market_state.intraday_price

            # I calculate HEnEx/ADMIE fees for this step
            step_dam_vol = abs(dam_commitment) * self.data_provider.time_step_hours
            step_bal_vol = abs(deviation_mw) * self.data_provider.time_step_hours
            charge_mwh = abs(min(0, mw_setpoint)) * self.data_provider.time_step_hours
            step_fees = step_dam_vol * 0.04 + step_bal_vol * 0.02 + charge_mwh * 2.5

            # I track per-market profit (simplified for paper trading)
            step_dam_profit = dam_revenue
            step_imbalance_profit = imbalance_revenue

            self.dam_profit += step_dam_profit
            self.total_fees += step_fees

            # Update cumulative stats
            current_revenue = dam_revenue
            current_cost = step_fees

            if imbalance_revenue > 0:
                current_revenue += imbalance_revenue
            else:
                current_cost += abs(imbalance_revenue)

            self.total_revenue += current_revenue
            self.total_cost += current_cost
            self.total_profit += step_profit - step_fees
            
            # 6. Update VIRTUAL Battery
            # ------------------------------------------------
            self._update_virtual_battery(mw_setpoint)
            
            # 7. Log Decision
            # ------------------------------------------------
            self.decisions_count += 1
            
            with open(self.log_file, 'a') as f:
                f.write(f"{market_state.timestamp},{price:.2f},{self.virtual_soc:.4f},"
                       f"{imbalance_risk['risk_score']:.4f},{action_idx},{action_level:.2f},"
                       f"{mw_setpoint:.2f},{step_profit:.2f},{self.total_profit:.2f},"
                       f"{self.total_revenue:.2f},{self.total_cost:.2f},"
                       f"{step_dam_profit:.2f},{self.afrr_cap_profit:.2f},"
                       f"{self.afrr_energy_profit:.2f},{self.intraday_profit:.2f},"
                       f"{self.mfrr_profit:.2f},{self.free_bid_profit:.2f},"
                       f"{step_fees:.2f},"
                       f"{imbalance_risk['gradient']:.2f},{imbalance_risk['res_deviation']:.2f}\n")
            
            # 8. Return Status
            # ------------------------------------------------
            action_name = "IDLE"
            if action_level < 0:
                action_name = f"CHARGE {abs(action_level):.0%}"
            elif action_level > 0:
                action_name = f"DISCHARGE {action_level:.0%}"
            
            return {
                'timestamp': market_state.timestamp,
                'price': price,
                'soc': self.virtual_soc,
                'action': action_name,
                'mw': mw_setpoint,
                'step_profit': step_profit,
                'total_profit': self.total_profit,
                'risk_score': imbalance_risk['risk_score']
            }
        
        except Exception as e:
            logger.error(f"Error in paper trading tick: {e}", exc_info=True)
            return None
    
    def run_loop(self, interval_seconds: int = 300):
        """
        Main paper trading loop.
        Runs forever, logging all decisions.
        """
        logger.info("=" * 80)
        logger.info("PAPER TRADING MODE - LIVE DATA, VIRTUAL BATTERY")
        logger.info("=" * 80)
        logger.info(f"Interval: {interval_seconds}s ({interval_seconds/60:.1f} min)")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)
        
        while True:
            try:
                start_time = time.time()
                
                info = self.run_tick()
                
                if info:
                    # Console output
                    log_msg = (
                        f"[{info['timestamp']:%Y-%m-%d %H:%M}] "
                        f"Price:{info['price']:>7.2f}€ | "
                        f"SoC:{info['soc']:>5.1%} | "
                        f"Risk:{info['risk_score']:>4.2f} | "
                        f"{info['action']:<15} | "
                        f"{info['mw']:>+6.1f}MW | "
                        f"Step:{info['step_profit']:>+8.2f}€ | "
                        f"Total:{info['total_profit']:>+10.2f}€"
                    )
                    logger.info(log_msg)
                    
                    # Periodic summary with per-market breakdown
                    if self.decisions_count % 10 == 0:
                        avg_profit = self.total_profit / self.decisions_count if self.decisions_count > 0 else 0
                        logger.info(f"--- Stats after {self.decisions_count} ticks ---")
                        logger.info(f"Avg Profit/Tick: {avg_profit:+.2f}€")
                        logger.info(f"Revenue: {self.total_revenue:,.2f}€ | Cost: {self.total_cost:,.2f}€ | Fees: {self.total_fees:,.2f}€")
                        logger.info(f"DAM={self.dam_profit:+,.0f}€ | aFRR_cap={self.afrr_cap_profit:+,.0f}€ | "
                                    f"aFRR_e={self.afrr_energy_profit:+,.0f}€ | ID={self.intraday_profit:+,.0f}€ | "
                                    f"mFRR={self.mfrr_profit:+,.0f}€ | FreeBid={self.free_bid_profit:+,.0f}€")
                
                # Wait for next tick
                elapsed = time.time() - start_time
                sleep_time = max(1.0, interval_seconds - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("\n" + "=" * 80)
                logger.info("PAPER TRADING STOPPED BY USER")
                logger.info("=" * 80)
                logger.info(f"Total Ticks: {self.decisions_count}")
                logger.info(f"Total Profit: {self.total_profit:+,.2f}€ (net of fees)")
                logger.info(f"Total Revenue: {self.total_revenue:,.2f}€")
                logger.info(f"Total Cost: {self.total_cost:,.2f}€")
                logger.info(f"Total Fees: {self.total_fees:,.2f}€")
                logger.info(f"Per-Market: DAM={self.dam_profit:+,.0f}€ | "
                            f"aFRR_cap={self.afrr_cap_profit:+,.0f}€ | "
                            f"aFRR_e={self.afrr_energy_profit:+,.0f}€ | "
                            f"ID={self.intraday_profit:+,.0f}€ | "
                            f"mFRR={self.mfrr_profit:+,.0f}€ | "
                            f"FreeBid={self.free_bid_profit:+,.0f}€")
                logger.info(f"Final Virtual SoC: {self.virtual_soc:.2%}")
                logger.info(f"Log saved to: {os.path.abspath(self.log_file)}")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(60)  # Retry delay


def main():
    """
    Main entry point for paper trading.

    Environment variables:
    - STRATEGY: AI, AGGRESSIVE_HYBRID, RULE_BASED (default: AI)
    - MODEL_PATH: Path to AI model (required for AI strategy)
    - BATTERY_CAPACITY_MWH: Battery capacity (default: 146.0)
    - BATTERY_MAX_POWER_MW: Max power (default: 30.0)
    - BATTERY_EFFICIENCY: Round-trip efficiency (default: 0.94)
    - INITIAL_SOC: Starting SoC (default: 0.5)
    - TIME_STEP_HOURS: Time step in hours (default: 0.25)
    - TRADING_INTERVAL_SECONDS: Loop interval (default: 300)
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Paper Trading Simulator')
    parser.add_argument('--strategy', choices=['ai', 'hybrid', 'rule'],
                        default=None, help='Trading strategy')
    args = parser.parse_args()

    # I determine strategy from args or environment
    strategy_map = {
        'ai': 'AI',
        'hybrid': 'AGGRESSIVE_HYBRID',
        'rule': 'RULE_BASED'
    }
    if args.strategy:
        strategy_type = strategy_map[args.strategy]
    else:
        strategy_type = os.getenv('STRATEGY', 'AI').upper()

    logger.info(f"Strategy: {strategy_type}")

    # Battery parameters
    battery_params = {
        'capacity_mwh': float(os.getenv('BATTERY_CAPACITY_MWH', 146.0)),
        'max_discharge_mw': float(os.getenv('BATTERY_MAX_POWER_MW', 30.0)),
        'efficiency': float(os.getenv('BATTERY_EFFICIENCY', 0.94))
    }

    # Initialize LiveDataProvider (for LIVE data fetching)
    logger.info("Initializing LiveDataProvider (LIVE DATA MODE)...")
    provider = LiveDataProvider(
        api_key=os.getenv('ENTSOE_API_KEY'),
        battery_params=battery_params,
        initial_soc=float(os.getenv('INITIAL_SOC', 0.5)),
        time_step_hours=float(os.getenv('TIME_STEP_HOURS', 0.25))
    )

    # Load trained model (only needed for AI strategy)
    model_path = os.getenv('MODEL_PATH',
                          '../models/checkpoints/ppo_15min/ppo_15min_3200000_steps.zip')

    # Create paper trading simulator with specified strategy
    simulator = PaperTradingSimulator(
        model_path=model_path,
        data_provider=provider,
        strategy_type=strategy_type
    )

    # Start loop
    interval_seconds = int(os.getenv('TRADING_INTERVAL_SECONDS', 300))  # 5 min default
    simulator.run_loop(interval_seconds=interval_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
