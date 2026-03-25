"""DataQualityAgent — detects and fixes data quality issues.

Technical contract (Assignment 2):
    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    df_clean = agent.fix(df, strategy={'missing': 'drop', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
    comparison = agent.compare(df, df_clean)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGIES = {
    "aggressive": {"missing": "drop", "duplicates": "drop", "outliers": "drop_iqr"},
    "conservative": {"missing": "keep", "duplicates": "keep", "outliers": "keep"},
    "balanced": {"missing": "drop", "duplicates": "drop", "outliers": "clip_zscore"},
}


class DataQualityAgent:
    """Agent that detects and fixes data quality issues."""

    def __init__(self, config="config.yaml"):
        if isinstance(config, dict):
            self.config = config
        else:
            import yaml
            with open(config) as f:
                self.config = yaml.safe_load(f) or {}

    def detect_issues(self, df: pd.DataFrame, output_dir: str = "data/detective") -> dict:
        """Detect missing values, duplicates, outliers, class imbalance."""
        # Missing
        missing = {}
        for col in df.columns:
            n = df[col].isna().sum()
            if col == "text":
                n += (df[col].astype(str).str.strip().str.len() == 0).sum()
            if n > 0:
                missing[col] = int(n)

        # Duplicates
        duplicates = int(df["text"].duplicated().sum())

        # Outliers (text length IQR)
        lengths = df["text"].str.len()
        q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        iqr = q3 - q1
        n_outliers = int(((lengths < q1 - 1.5 * iqr) | (lengths > q3 + 1.5 * iqr)).sum())

        # Class imbalance
        counts = df["label"].value_counts()
        ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")

        problems = {
            "missing": missing,
            "duplicates": duplicates,
            "outliers": {"length_outliers": n_outliers, "total": n_outliers},
            "imbalance": {
                "class_counts": counts.to_dict(),
                "imbalance_ratio": round(ratio, 2),
                "majority_class": str(counts.index[0]),
                "minority_class": str(counts.index[-1]),
            },
            "total_rows": len(df),
        }

        # Visualize
        os.makedirs(output_dir, exist_ok=True)
        self._visualize(df, problems, output_dir)

        with open(os.path.join(output_dir, "problems.json"), "w") as f:
            json.dump(problems, f, indent=2)

        return problems

    def fix(self, df: pd.DataFrame, strategy=None) -> pd.DataFrame:
        """Fix quality issues. strategy can be a string or dict."""
        if strategy is None:
            strategy = self.config.get("cleaning", {}).get("strategy", "balanced")

        if isinstance(strategy, str):
            if strategy not in STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy}. Choose: {list(STRATEGIES.keys())}")
            strat = STRATEGIES[strategy]
            label = strategy
        else:
            strat = strategy
            label = "custom"

        before = len(df)
        print(f"\nApplying '{label}' strategy:")

        # Missing
        if strat.get("missing") == "drop":
            b = len(df)
            df = df.dropna(subset=["text", "label"])
            df = df[df["text"].astype(str).str.strip().str.len() > 0]
            print(f"  Missing → dropped {b - len(df)} rows")

        # Duplicates
        if strat.get("duplicates") == "drop":
            b = len(df)
            df = df[~df["text"].duplicated(keep="first")]
            print(f"  Duplicates → dropped {b - len(df)} rows")

        # Outliers
        lengths = df["text"].str.len()
        if strat.get("outliers") == "drop_iqr":
            b = len(df)
            q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
            iqr = q3 - q1
            df = df[(lengths >= q1 - 1.5 * iqr) & (lengths <= q3 + 1.5 * iqr)]
            print(f"  Outliers (IQR) → dropped {b - len(df)} rows")
        elif strat.get("outliers") == "clip_zscore" or strat.get("outliers") == "clip_iqr":
            b = len(df)
            mean_l, std_l = lengths.mean(), lengths.std()
            if std_l > 0:
                df = df[((lengths - mean_l) / std_l).abs() <= 3]
            print(f"  Outliers (z>3) → dropped {b - len(df)} rows")

        removed = before - len(df)
        print(f"\n  Result: {before} → {len(df)} rows ({removed} removed, {removed/before*100:.1f}%)")

        # Save
        cleaned_dir = os.path.join(self.config.get("pipeline", {}).get("data_dir", "data"), "cleaned")
        os.makedirs(cleaned_dir, exist_ok=True)
        df = df.reset_index(drop=True)
        df.to_parquet(os.path.join(cleaned_dir, "cleaned.parquet"), index=False)

        return df

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> str:
        """Generate markdown comparison report."""
        n_b, n_a = len(df_before), len(df_after)
        cb = df_before["label"].value_counts()
        ca = df_after["label"].value_counts()
        lb = df_before["text"].str.len()
        la = df_after["text"].str.len()

        report = f"""# Data Quality Comparison Report

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total rows | {n_b} | {n_a} | -{n_b - n_a} ({(n_b-n_a)/n_b*100:.1f}%) |
| Missing | {df_before.isna().sum().sum()} | {df_after.isna().sum().sum()} | |
| Duplicates | {df_before['text'].duplicated().sum()} | {df_after['text'].duplicated().sum()} | |
| Mean text length | {lb.mean():.0f} | {la.mean():.0f} | |

## Class Distribution

| Class | Before | After |
|-------|--------|-------|
"""
        for cls in sorted(set(cb.index) | set(ca.index)):
            report += f"| {cls} | {cb.get(cls,0)} | {ca.get(cls,0)} |\n"
        return report

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _visualize(df, problems, output_dir):
        # Missing
        fig, ax = plt.subplots(figsize=(8, 5))
        m = problems["missing"]
        if m:
            ax.barh(list(m.keys()), list(m.values()), color="tomato", edgecolor="black")
            ax.set_title("Missing Values")
        else:
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
        fig.savefig(os.path.join(output_dir, "missing_values.png"), dpi=150); plt.close(fig)

        # Class balance
        c = problems["imbalance"]["class_counts"]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(list(c.keys()), list(c.values()), color="steelblue", edgecolor="black")
        ax.set_title(f"Class Balance ({problems['imbalance']['imbalance_ratio']:.1f}x)")
        ax.tick_params(axis="x", rotation=45); plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "class_balance.png"), dpi=150); plt.close(fig)

        # Outliers
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(df["text"].str.len(), vert=False)
        ax.set_title(f"Text Length Outliers ({problems['outliers']['total']})")
        fig.savefig(os.path.join(output_dir, "outliers.png"), dpi=150); plt.close(fig)

        # Duplicates
        nd = problems["duplicates"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie([len(df)-nd, nd], labels=["Unique","Duplicates"], colors=["steelblue","tomato"], autopct="%1.1f%%")
        ax.set_title(f"Duplicates: {nd}/{len(df)}")
        fig.savefig(os.path.join(output_dir, "duplicates.png"), dpi=150); plt.close(fig)

        print(f"  Visualizations saved to {output_dir}")
