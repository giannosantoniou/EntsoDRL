"""
Generate 15-Minute Training Data — Native ADMIE Pipeline

I generate native 15-min training data from real ADMIE balancing market data,
replacing the Phase 1 approach (forward-fill from hourly).

Native 15-min sources (mFRR, aFRR, imbalance) provide real intra-hour price
variation. Only DAM prices and SCADA load/RES (hourly by market design) are
forward-filled.

Usage:
    python data/generate_15min_training_data.py
    python data/generate_15min_training_data.py --output data/custom_15min.csv

Phase 1 upsampling is still available via --legacy flag.
"""

import sys
from pathlib import Path

# I add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate native 15-min training data from ADMIE"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: data/unified_multimarket_training_15min.csv)"
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy Phase 1 upsampling (forward-fill from hourly) instead "
             "of native 15-min pipeline"
    )
    parser.add_argument(
        "--input", type=str,
        default="data/unified_multimarket_training.csv",
        help="Input hourly CSV for legacy mode only"
    )
    args = parser.parse_args()

    if args.legacy:
        # I fall back to Phase 1 upsampling for backward compatibility
        print("Using LEGACY Phase 1 upsampling (forward-fill from hourly)...")
        import pandas as pd
        from data.upsample_to_15min import upsample_hourly_to_15min, verify_upsampling

        input_path = Path(args.input)
        if not input_path.exists():
            input_path = project_root / args.input
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        output_path = Path(args.output) if args.output else input_path.with_name(
            input_path.stem + "_15min" + input_path.suffix
        )

        print(f"Loading hourly data from {input_path}...")
        df_hourly = pd.read_csv(input_path, index_col=0, parse_dates=True)
        print(f"  Loaded: {len(df_hourly)} rows, {len(df_hourly.columns)} columns")

        print(f"\nUpsampling to 15-min using 'step' method...")
        df_15min = upsample_hourly_to_15min(
            df_hourly, method='step',
            energy_columns=[], power_columns=['dam_commitment']
        )

        verification = verify_upsampling(df_hourly, df_15min)
        if verification['passed']:
            print("  All verification checks PASSED")

        df_15min.to_csv(output_path)
        print(f"\nSaved: {len(df_15min)} rows to {output_path}")
    else:
        # I use the native 15-min pipeline with real ADMIE data
        from data.create_unified_training_data import create_unified_dataset_from_admie_15min

        kwargs = {}
        if args.output:
            kwargs['output_path'] = args.output

        print("Using NATIVE 15-min pipeline (real ADMIE data)...")
        create_unified_dataset_from_admie_15min(**kwargs)

    print("\nDone! Use this data with:")
    print("  python agent/train_unified.py --data data/unified_multimarket_training_15min.csv --time_step 0.25")


if __name__ == "__main__":
    main()
