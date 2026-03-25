"""Generate EDA markdown report for Raman spectroscopy dataset."""

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd


def generate_report(input_path: str, eda_dir: str, output_path: str):
    """Generate markdown report from EDA results."""
    df = pd.read_parquet(input_path)

    eda_json = os.path.join(eda_dir, "eda_results.json")
    with open(eda_json) as f:
        eda = json.load(f)

    report = f"""# EDA Report: Raman Spectroscopy Dataset

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total samples | {eda['total_samples']} |
| Number of classes | {eda['num_classes']} |
| Sources | {len(eda.get('sources', {}))} |

## Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
"""
    total = eda["total_samples"]
    for cls, count in sorted(eda["class_distribution"].items(), key=lambda x: -x[1]):
        pct = count / total * 100
        report += f"| {cls} | {count} | {pct:.1f}% |\n"

    report += f"""
![Class Distribution](class_distribution.png)

## Spectrum Statistics

| Metric | Value |
|--------|-------|
| Mean length | {eda['spectrum_length']['mean']:.0f} points |
| Std length | {eda['spectrum_length']['std']:.1f} |
| Min length | {eda['spectrum_length']['min']} |
| Max length | {eda['spectrum_length']['max']} |

![Spectrum Length Distribution](spectrum_stats.png)

## Example Spectra

![Example Spectra by Class](spectrum_examples.png)

## Data Sources

| Source | Count |
|--------|-------|
"""
    for src, count in eda.get("sources", {}).items():
        report += f"| {src} | {count} |\n"

    report += f"""
## Intensity Statistics

| Metric | Value |
|--------|-------|
| Mean intensity | {eda['intensity_stats']['global_mean']:.2f} |
| Mean max intensity | {eda['intensity_stats']['global_max_mean']:.2f} |
| Mean std intensity | {eda['intensity_stats']['global_std_mean']:.2f} |
"""

    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate EDA report")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--eda-dir", default="data/eda", help="EDA results directory")
    parser.add_argument("--output", default="data/eda/REPORT.md", help="Output report path")
    args = parser.parse_args()

    generate_report(args.input, args.eda_dir, args.output)


if __name__ == "__main__":
    main()
