"""Detect data quality issues in Raman spectroscopy datasets."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_missing(df: pd.DataFrame) -> dict:
    """Detect missing values."""
    missing = {}
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if col == "spectrum":
            n_missing += df[col].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()
        if n_missing > 0:
            missing[col] = int(n_missing)
    return missing


def detect_duplicates(df: pd.DataFrame) -> int:
    """Detect duplicate spectra (by converting to string for comparison)."""
    spec_strings = df["spectrum"].apply(lambda x: str(x[:20]) if isinstance(x, list) and len(x) > 0 else "")
    return int(spec_strings.duplicated().sum())


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """Detect outliers in spectrum length and intensity using IQR."""
    # Spectrum length outliers
    lengths = df["spectrum"].apply(len)
    q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
    iqr = q3 - q1
    length_outliers = int(((lengths < q1 - 1.5 * iqr) | (lengths > q3 + 1.5 * iqr)).sum())

    # Intensity outliers (mean intensity per spectrum)
    mean_intensities = df["spectrum"].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    q1_i, q3_i = mean_intensities.quantile(0.25), mean_intensities.quantile(0.75)
    iqr_i = q3_i - q1_i
    intensity_outliers = int(((mean_intensities < q1_i - 1.5 * iqr_i) | (mean_intensities > q3_i + 1.5 * iqr_i)).sum())

    return {
        "length_outliers": length_outliers,
        "intensity_outliers": intensity_outliers,
        "total": length_outliers + intensity_outliers,
    }


def detect_class_imbalance(df: pd.DataFrame) -> dict:
    """Detect class imbalance."""
    counts = df["label"].value_counts()
    ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
    return {
        "class_counts": counts.to_dict(),
        "imbalance_ratio": round(ratio, 2),
        "majority_class": str(counts.index[0]),
        "minority_class": str(counts.index[-1]),
    }


def detect_all(df: pd.DataFrame) -> dict:
    """Run all detection checks."""
    return {
        "missing": detect_missing(df),
        "duplicates": detect_duplicates(df),
        "outliers": detect_outliers_iqr(df),
        "imbalance": detect_class_imbalance(df),
        "total_rows": len(df),
    }


def visualize_problems(df: pd.DataFrame, problems: dict, output_dir: str):
    """Generate visualization for each problem type."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Missing values
    missing = problems["missing"]
    if missing:
        fig, ax = plt.subplots(figsize=(8, 5))
        cols = list(missing.keys())
        vals = list(missing.values())
        ax.barh(cols, vals, color="tomato", edgecolor="black")
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Count")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "missing_values.png"), dpi=150)
        plt.close(fig)
    else:
        # No missing — create placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(os.path.join(output_dir, "missing_values.png"), dpi=150)
        plt.close(fig)

    # 2. Class balance
    counts = problems["imbalance"]["class_counts"]
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(counts.keys())
    values = list(counts.values())
    ax.bar(classes, values, color="steelblue", edgecolor="black")
    ax.set_title(f"Class Balance (imbalance ratio: {problems['imbalance']['imbalance_ratio']:.1f}x)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "class_balance.png"), dpi=150)
    plt.close(fig)

    # 3. Outliers
    lengths = df["spectrum"].apply(len)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(lengths, vert=False)
    ax.set_title(f"Spectrum Length Outliers ({problems['outliers']['length_outliers']} detected)")
    ax.set_xlabel("Spectrum Length")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "outliers.png"), dpi=150)
    plt.close(fig)

    # 4. Duplicates
    fig, ax = plt.subplots(figsize=(6, 4))
    n_dup = problems["duplicates"]
    n_unique = len(df) - n_dup
    ax.pie([n_unique, n_dup], labels=["Unique", "Duplicates"],
           colors=["steelblue", "tomato"], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Duplicates: {n_dup} of {len(df)}")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "duplicates.png"), dpi=150)
    plt.close(fig)

    print(f"  Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Detect data quality issues")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output-dir", default="data/detective", help="Output directory")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    problems = detect_all(df)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "problems.json"), "w") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"\nQuality Report:")
    print(f"  Missing values: {problems['missing']}")
    print(f"  Duplicates: {problems['duplicates']}")
    print(f"  Outliers: {problems['outliers']['total']} (length: {problems['outliers']['length_outliers']}, intensity: {problems['outliers']['intensity_outliers']})")
    print(f"  Imbalance ratio: {problems['imbalance']['imbalance_ratio']}x ({problems['imbalance']['majority_class']} vs {problems['imbalance']['minority_class']})")

    visualize_problems(df, problems, args.output_dir)


if __name__ == "__main__":
    main()
