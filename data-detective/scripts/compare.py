"""Compare dataset before and after cleaning."""

import argparse
import os

import numpy as np
import pandas as pd


def compare_datasets(before_path: str, after_path: str) -> str:
    """Generate comparison report."""
    df_before = pd.read_parquet(before_path)
    df_after = pd.read_parquet(after_path)

    # Basic stats
    n_before = len(df_before)
    n_after = len(df_after)
    n_removed = n_before - n_after

    # Class distribution
    classes_before = df_before["label"].value_counts()
    classes_after = df_after["label"].value_counts()

    # Spectrum stats
    len_before = df_before["spectrum"].apply(len)
    len_after = df_after["spectrum"].apply(len)

    # Missing values
    missing_before = df_before.isna().sum().sum()
    missing_after = df_after.isna().sum().sum()

    # Duplicates
    dup_before = df_before["spectrum"].apply(lambda x: str(x[:20]) if isinstance(x, list) and len(x) > 0 else "").duplicated().sum()
    dup_after = df_after["spectrum"].apply(lambda x: str(x[:20]) if isinstance(x, list) and len(x) > 0 else "").duplicated().sum()

    report = f"""# Data Quality Comparison Report

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total rows | {n_before} | {n_after} | -{n_removed} ({n_removed/n_before*100:.1f}%) |
| Missing values | {missing_before} | {missing_after} | -{missing_before - missing_after} |
| Duplicates | {dup_before} | {dup_after} | -{dup_before - dup_after} |
| Mean spectrum length | {len_before.mean():.0f} | {len_after.mean():.0f} | {len_after.mean() - len_before.mean():+.0f} |
| Std spectrum length | {len_before.std():.1f} | {len_after.std():.1f} | {len_after.std() - len_before.std():+.1f} |

## Class Distribution

| Class | Before | After | Change |
|-------|--------|-------|--------|
"""
    all_classes = sorted(set(classes_before.index) | set(classes_after.index))
    for cls in all_classes:
        cb = classes_before.get(cls, 0)
        ca = classes_after.get(cls, 0)
        report += f"| {cls} | {cb} | {ca} | {ca - cb:+d} |\n"

    # Imbalance ratio
    ratio_before = classes_before.max() / classes_before.min() if classes_before.min() > 0 else float("inf")
    ratio_after = classes_after.max() / classes_after.min() if classes_after.min() > 0 else float("inf")

    report += f"""
## Imbalance

| Metric | Before | After |
|--------|--------|-------|
| Imbalance ratio | {ratio_before:.2f}x | {ratio_after:.2f}x |
| Number of classes | {len(classes_before)} | {len(classes_after)} |

## Conclusion

Cleaning removed **{n_removed}** rows ({n_removed/n_before*100:.1f}% of dataset).
"""
    if ratio_after < ratio_before:
        report += "Class imbalance improved.\n"
    elif ratio_after > ratio_before:
        report += "Class imbalance slightly worsened (some classes lost more samples).\n"

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare before/after cleaning")
    parser.add_argument("--before", required=True, help="Before parquet file")
    parser.add_argument("--after", required=True, help="After parquet file")
    parser.add_argument("--output", default="data/detective/comparison.md")
    args = parser.parse_args()

    report = compare_datasets(args.before, args.after)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
