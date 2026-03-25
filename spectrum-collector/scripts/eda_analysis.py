"""Exploratory Data Analysis for Raman spectroscopy datasets."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_eda(df: pd.DataFrame, output_dir: str) -> dict:
    """Run full EDA on unified Raman dataset."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # --- 1. Class distribution ---
    label_counts = df["label"].value_counts()
    results["class_distribution"] = label_counts.to_dict()
    results["total_samples"] = len(df)
    results["num_classes"] = len(label_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    label_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Material Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved class_distribution.png")

    # --- 2. Spectrum length distribution ---
    spec_lengths = df["spectrum"].apply(len)
    results["spectrum_length"] = {
        "mean": float(spec_lengths.mean()),
        "std": float(spec_lengths.std()),
        "min": int(spec_lengths.min()),
        "max": int(spec_lengths.max()),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(spec_lengths, bins=30, color="steelblue", edgecolor="black")
    ax.set_title("Spectrum Length Distribution")
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrum_stats.png"), dpi=150)
    plt.close(fig)
    print("  Saved spectrum_stats.png")

    # --- 3. Example spectra by class ---
    classes = label_counts.index.tolist()[:6]  # Top 6 classes
    n_classes = len(classes)
    if n_classes > 0:
        fig, axes = plt.subplots(
            min(n_classes, 3), max(1, (n_classes + 2) // 3),
            figsize=(14, 4 * min(n_classes, 3)),
            squeeze=False,
        )
        for i, cls in enumerate(classes):
            row, col = divmod(i, max(1, (n_classes + 2) // 3))
            ax = axes[row][col]
            sample = df[df["label"] == cls].iloc[0]
            spectrum = np.array(sample["spectrum"])
            wavenumber = np.array(sample["wavenumber"]) if len(sample["wavenumber"]) == len(spectrum) else np.arange(len(spectrum))
            ax.plot(wavenumber, spectrum, linewidth=0.8)
            ax.set_title(f"{cls}")
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity")
        # Hide unused axes
        for i in range(n_classes, axes.shape[0] * axes.shape[1]):
            row, col = divmod(i, axes.shape[1])
            axes[row][col].set_visible(False)
        plt.suptitle("Example Spectra by Class", fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "spectrum_examples.png"), dpi=150)
        plt.close(fig)
        print("  Saved spectrum_examples.png")

    # --- 4. Sources ---
    source_counts = df["source"].value_counts()
    results["sources"] = source_counts.to_dict()

    # --- 5. Intensity stats ---
    all_intensities = []
    for spec in df["spectrum"].head(500):
        arr = np.array(spec)
        all_intensities.extend([float(arr.mean()), float(arr.max()), float(arr.std())])

    results["intensity_stats"] = {
        "global_mean": float(np.mean(all_intensities[::3])),
        "global_max_mean": float(np.mean(all_intensities[1::3])),
        "global_std_mean": float(np.mean(all_intensities[2::3])),
    }

    # Save results JSON
    with open(os.path.join(output_dir, "eda_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  Saved eda_results.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="EDA for Raman spectroscopy dataset")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output-dir", default="data/eda", help="Output directory")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    results = run_eda(df, args.output_dir)

    print(f"\nEDA Summary:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Classes: {results['num_classes']}")
    print(f"  Class distribution: {results['class_distribution']}")
    print(f"  Spectrum length: {results['spectrum_length']}")


if __name__ == "__main__":
    main()
