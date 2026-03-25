"""Clean Raman spectroscopy datasets using configurable strategies."""

import argparse
import json
import os

import numpy as np
import pandas as pd


STRATEGIES = {
    "aggressive": {
        "missing": "drop",
        "duplicates": "drop",
        "outliers": "drop_iqr",
        "description": "Remove all problematic rows",
    },
    "conservative": {
        "missing": "keep",
        "duplicates": "keep",
        "outliers": "keep",
        "description": "Keep everything, minimal changes",
    },
    "balanced": {
        "missing": "drop",
        "duplicates": "drop",
        "outliers": "clip_zscore",
        "description": "Drop missing/duplicates, clip extreme outliers",
    },
}


def clean_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle missing values."""
    if strategy == "drop":
        before = len(df)
        # Drop rows with NaN
        df = df.dropna(subset=["spectrum", "label"])
        # Drop rows with empty spectra
        df = df[df["spectrum"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        print(f"  Missing → dropped {before - len(df)} rows")
    return df


def clean_duplicates(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle duplicate spectra."""
    if strategy == "drop":
        before = len(df)
        spec_str = df["spectrum"].apply(lambda x: str(x[:20]) if isinstance(x, list) and len(x) > 0 else "")
        df = df[~spec_str.duplicated(keep="first")]
        print(f"  Duplicates → dropped {before - len(df)} rows")
    return df


def clean_outliers(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle outliers."""
    lengths = df["spectrum"].apply(len)

    if strategy == "drop_iqr":
        before = len(df)
        q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        iqr = q3 - q1
        mask = (lengths >= q1 - 1.5 * iqr) & (lengths <= q3 + 1.5 * iqr)
        df = df[mask]
        print(f"  Outliers (IQR) → dropped {before - len(df)} rows")

    elif strategy == "clip_zscore":
        before = len(df)
        mean_len = lengths.mean()
        std_len = lengths.std()
        if std_len > 0:
            z_scores = (lengths - mean_len) / std_len
            mask = z_scores.abs() <= 3
            df = df[mask]
        print(f"  Outliers (z>3) → dropped {before - len(df)} rows")

    return df


def clean_data(df: pd.DataFrame, strategy_name: str) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning strategy."""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from: {list(STRATEGIES.keys())}")

    strategy = STRATEGIES[strategy_name]
    before_count = len(df)
    stats = {"strategy": strategy_name, "before": before_count}

    print(f"\nApplying '{strategy_name}' strategy: {strategy['description']}")

    df = clean_missing(df, strategy["missing"])
    stats["after_missing"] = len(df)

    df = clean_duplicates(df, strategy["duplicates"])
    stats["after_duplicates"] = len(df)

    df = clean_outliers(df, strategy["outliers"])
    stats["after_outliers"] = len(df)

    stats["after"] = len(df)
    stats["removed"] = before_count - len(df)
    stats["removed_pct"] = round(stats["removed"] / before_count * 100, 1) if before_count > 0 else 0

    print(f"\n  Result: {before_count} → {len(df)} rows ({stats['removed']} removed, {stats['removed_pct']}%)")

    return df.reset_index(drop=True), stats


def main():
    parser = argparse.ArgumentParser(description="Clean dataset with chosen strategy")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--strategy", default="balanced", choices=list(STRATEGIES.keys()))
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    df_clean, stats = clean_data(df, args.strategy)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_clean.to_parquet(args.output, index=False)

    # Save stats
    stats_path = os.path.join(os.path.dirname(args.output), "cleaning_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
