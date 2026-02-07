"""
Battery Trading System v2.0 - Production Main Entry Point.

This is the production-ready main.py that uses:
- IDataProvider (Data Layer)
- IDecisionStrategy (Logic Layer)
- BatteryController (Control Layer)

Usage:
    python main.py                    # Default: Training mode with AI
    python main.py --mode production  # Production mode
    python main.py --strategy rule    # Use rule-based strategy
"""
import sys
import argparse
import pandas as pd

sys.path.insert(0, '.')

from data.data_loader import load_historical_data
from data.data_provider import SimulatedDataProvider, LiveDataProvider, NormalizationConfig
from data.dam_commitment_generator import FeasibleDAMGenerator
from agent.decision_strategy import create_strategy
from agent.battery_controller import BatteryController


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Mode: TRAINING (CSV simulation) or PRODUCTION (Live API)
    "MODE": "TRAINING",
    
    # Strategy: AI, RULE_BASED, RANDOM, CONSERVATIVE, AGGRESSIVE_HYBRID
    "STRATEGY": "AI",
    
    # Paths
    "MODEL_PATH": "models/best_model_5M_profit.zip",
    "DATA_FILE": "data/feasible_data_with_dam.csv",
    
    # Battery Parameters
    "BATTERY": {
        "capacity_mwh": 146.0,      # Updated: 146 MWh capacity (~5h duration)
        "max_discharge_mw": 30.0,   # Inverter rating: 30 MW (unchanged)
        "efficiency": 0.94
    },
    
    # Control Loop
    "CYCLE_INTERVAL": 0,      # 0 for fast simulation, 900 (15 min) for live
    "MAX_CYCLES": 1000,       # None for infinite (production)
    "LOOKAHEAD_HOURS": 4,
    "N_ACTIONS": 21,
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Battery Trading System v2.0')
    parser.add_argument('--mode', choices=['training', 'production'], 
                        default='training', help='Operating mode')
    parser.add_argument('--strategy', choices=['ai', 'rule', 'random', 'conservative', 'hybrid'],
                        default='ai', help='Decision strategy (hybrid = AggressiveHybrid from backtesting)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to AI model (overrides config)')
    parser.add_argument('--cycles', type=int, default=None,
                        help='Max cycles to run')
    args = parser.parse_args()
    
    # Override config with args
    if args.mode:
        CONFIG["MODE"] = args.mode.upper()
    if args.strategy:
        strategy_map = {
            'ai': 'AI',
            'rule': 'RULE_BASED',
            'random': 'RANDOM',
            'conservative': 'CONSERVATIVE',
            'hybrid': 'AGGRESSIVE_HYBRID'
        }
        CONFIG["STRATEGY"] = strategy_map[args.strategy]
    if args.model:
        CONFIG["MODEL_PATH"] = args.model
    if args.cycles:
        CONFIG["MAX_CYCLES"] = args.cycles
    
    print("=" * 60)
    print(" Battery Trading System v2.0")
    print("=" * 60)
    print(f"   Mode:     {CONFIG['MODE']}")
    print(f"   Strategy: {CONFIG['STRATEGY']}")
    print(f"   Model:    {CONFIG['MODEL_PATH']}")
    print("=" * 60)

    # ------------------------------------------------
    # 1. SETUP DATA PROVIDER (THE BODY)
    # ------------------------------------------------
    if CONFIG["MODE"] == "TRAINING":
        print("\n Loading training data...")
        try:
            df = load_historical_data(CONFIG["DATA_FILE"])
            
            # Use test split (last 20%)
            split_idx = int(len(df) * 0.8)
            test_df = df.iloc[split_idx:].copy()
            
            # Regenerate DAM commitments
            gen = FeasibleDAMGenerator(
                capacity_mwh=CONFIG["BATTERY"]["capacity_mwh"],
                max_power_mw=CONFIG["BATTERY"]["max_discharge_mw"],
                efficiency=CONFIG["BATTERY"]["efficiency"]
            )
            test_df['dam_commitment'] = gen.generate(test_df)
            
            print(f"   Data: {test_df.index[0]} to {test_df.index[-1]}")
            print(f"   Rows: {len(test_df):,}")
            
        except FileNotFoundError:
            print(f" Error: Data file {CONFIG['DATA_FILE']} not found.")
            sys.exit(1)
        
        provider = SimulatedDataProvider(
            df=test_df,
            battery_params=CONFIG["BATTERY"],
            noise_std=5.0  # Add noise for robustness
        )
        
    else:  # PRODUCTION
        print("\n Initializing production mode...")
        print("  WARNING: LiveDataProvider requires API/PLC clients!")
        
        # In production, you would pass real clients here:
        # provider = LiveDataProvider(
        #     market_api_client=EnExAPIClient(api_key="..."),
        #     battery_plc_client=ModbusClient(ip="192.168.1.50"),
        #     forecast_service=ForecastService(endpoint="...")
        # )
        
        print(" Production mode not yet implemented.")
        print("   Please configure API clients in main.py")
        sys.exit(1)

    # ------------------------------------------------
    # 2. SETUP STRATEGY (THE BRAIN)
    # ------------------------------------------------
    print(f"\n Loading strategy: {CONFIG['STRATEGY']}...")
    try:
        strategy = create_strategy(
            strategy_type=CONFIG["STRATEGY"],
            model_path=CONFIG["MODEL_PATH"],
            use_maskable=True,
            n_actions=CONFIG["N_ACTIONS"]
        )
    except Exception as e:
        print(f" Error loading strategy: {e}")
        sys.exit(1)

    # ------------------------------------------------
    # 3. SETUP CONTROLLER (THE NERVOUS SYSTEM)
    # ------------------------------------------------
    print("\n Initializing controller...")
    controller = BatteryController(
        data_provider=provider,
        strategy=strategy,
        n_actions=CONFIG["N_ACTIONS"],
        lookahead_hours=CONFIG["LOOKAHEAD_HOURS"]
    )

    # ------------------------------------------------
    # 4. RUN (LOOP)
    # ------------------------------------------------
    print(f"\n System Online!")
    print(f"   Interval: {CONFIG['CYCLE_INTERVAL']}s")
    print(f"   Max Cycles: {CONFIG['MAX_CYCLES'] or 'infinite'}")
    
    try:
        controller.run_continuous(
            interval_seconds=CONFIG["CYCLE_INTERVAL"],
            max_cycles=CONFIG["MAX_CYCLES"]
        )
    except KeyboardInterrupt:
        print("\n\n System stopped by user.")
    
    print("\n Battery Trading System shutdown complete.")


if __name__ == "__main__":
    main()
