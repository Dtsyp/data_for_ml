"""Unify dataset schema to standard format for Raman spectroscopy pipeline."""

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np


def unify_dataframe(
    df: pd.DataFrame,
    spectrum_col: str | None = None,
    wavenumber_col: str | None = None,
    label_col: str | None = None,
    source_name: str = "unknown",
) -> pd.DataFrame:
    """Convert a DataFrame to the unified schema.

    Unified schema:
        spectrum: list[float] — intensity values
        wavenumber: list[float] — wavenumber values
        label: str — material class
        source: str — data origin
        collected_at: str — ISO timestamp
    """
    unified = pd.DataFrame()

    # Handle spectrum column
    if spectrum_col and spectrum_col in df.columns:
        col = df[spectrum_col]
        if col.dtype == object:
            # Try parsing JSON strings
            unified["spectrum"] = col.apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        else:
            unified["spectrum"] = col.apply(lambda x: [float(x)] if np.isscalar(x) else list(x))
    else:
        # If no spectrum column, try to detect numeric columns as spectrum
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)
        if len(numeric_cols) > 5:
            # Many numeric columns = likely spectrum data in wide format
            print(f"  Detected wide-format spectrum: {len(numeric_cols)} numeric columns")
            unified["spectrum"] = df[numeric_cols].values.tolist()
            # Use column names as wavenumber if they look numeric
            try:
                wn = [float(c) for c in numeric_cols]
                unified["wavenumber"] = [wn] * len(df)
            except (ValueError, TypeError):
                unified["wavenumber"] = [list(range(len(numeric_cols)))] * len(df)
        else:
            print("  WARNING: Could not detect spectrum columns")
            unified["spectrum"] = [[] for _ in range(len(df))]

    # Handle wavenumber
    if "wavenumber" not in unified.columns:
        if wavenumber_col and wavenumber_col in df.columns:
            col = df[wavenumber_col]
            if col.dtype == object:
                unified["wavenumber"] = col.apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            else:
                unified["wavenumber"] = col.apply(lambda x: [float(x)] if np.isscalar(x) else list(x))
        else:
            # Generate default wavenumber range
            spec_len = len(unified["spectrum"].iloc[0]) if len(unified) > 0 else 0
            if spec_len > 0:
                wn = np.linspace(200, 3500, spec_len).tolist()
                unified["wavenumber"] = [wn] * len(df)
            else:
                unified["wavenumber"] = [[] for _ in range(len(df))]

    # Handle label
    if label_col and label_col in df.columns:
        unified["label"] = df[label_col].astype(str)
    else:
        unified["label"] = "unknown"

    # Metadata
    unified["source"] = source_name
    unified["collected_at"] = datetime.now(timezone.utc).isoformat()

    # Filter out empty spectra
    unified = unified[unified["spectrum"].apply(lambda x: len(x) > 0)]

    return unified


def main():
    parser = argparse.ArgumentParser(description="Unify dataset to standard schema")
    parser.add_argument("--input", required=True, help="Input CSV/parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--spectrum-col", default=None, help="Column with spectrum data")
    parser.add_argument("--wavenumber-col", default=None, help="Column with wavenumber data")
    parser.add_argument("--label-col", default=None, help="Column with labels")
    parser.add_argument("--source-name", default="unknown", help="Source identifier")
    args = parser.parse_args()

    # Load input
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    elif args.input.endswith(".json"):
        df = pd.read_json(args.input)
    else:
        df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Columns: {list(df.columns)}")

    unified = unify_dataframe(
        df,
        spectrum_col=args.spectrum_col,
        wavenumber_col=args.wavenumber_col,
        label_col=args.label_col,
        source_name=args.source_name,
    )

    unified.to_parquet(args.output, index=False)
    print(f"Saved {len(unified)} rows to {args.output}")
    print(f"Labels: {unified['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
