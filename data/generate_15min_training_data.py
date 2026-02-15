"""
Generate 15-Minute Training Data from Hourly Data

I load existing hourly unified training data and upsample to 15-minute
resolution using step interpolation (forward fill).

This is a Phase 1 solution — prices are forward-filled from hourly values.
Phase 2 will use native 15-min data from ADMIE fetcher.

Usage:
    python data/generate_15min_training_data.py
    python data/generate_15min_training_data.py --input data/unified_multimarket_training.csv
    python data/generate_15min_training_data.py --method linear
"""

import sys
from pathlib import Path

# I add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from data.upsample_to_15min import upsample_hourly_to_15min, verify_upsampling


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate 15-min training data from hourly")
    parser.add_argument("--input", type=str,
                        default="data/unified_multimarket_training.csv",
                        help="Path to hourly training data CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: adds _15min suffix)")
    parser.add_argument("--method", type=str, default="step",
                        choices=["step", "linear"],
                        help="Interpolation method: step (forward fill) or linear")
    args = parser.parse_args()

    # I resolve paths
    input_path = Path(args.input)
    if not input_path.exists():
        input_path = project_root / args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            input_path.stem + "_15min" + input_path.suffix
        )

    # I load the hourly data
    print(f"Loading hourly data from {input_path}...")
    df_hourly = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"  Loaded: {len(df_hourly)} rows, {len(df_hourly.columns)} columns")
    print(f"  Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")

    # I upsample to 15-min
    # dam_commitment is power (MW), NOT energy — forward-filled by default
    # No energy columns to divide by 4 in our standard dataset
    print(f"\nUpsampling to 15-min using '{args.method}' method...")
    df_15min = upsample_hourly_to_15min(
        df_hourly,
        method=args.method,
        energy_columns=[],        # No energy columns to divide
        power_columns=['dam_commitment']  # Power columns forward-filled as-is
    )

    # I verify the upsampling
    print("\nVerifying upsampling quality...")
    verification = verify_upsampling(df_hourly, df_15min)
    if verification['passed']:
        print("  All verification checks PASSED")
    else:
        print("  WARNING: Some verification checks failed:")
        for check in verification['checks']:
            status = 'PASS' if check['passed'] else 'FAIL'
            print(f"    [{status}] {check['name']}")

    # I save the result
    print(f"\nSaving 15-min data to {output_path}...")
    df_15min.to_csv(output_path)
    print(f"  Saved: {len(df_15min)} rows")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")

    print("\nDone! Use this data with:")
    print(f"  python agent/train_unified.py --data {output_path} --time_step 0.25")


if __name__ == "__main__":
    main()
