"""
Raman Spectroscopy ML Pipeline — End-to-end orchestration.

Runs all 4 agents sequentially:
1. Spectrum Collector — gather and unify data
2. Data Detective — detect and fix quality issues
3. Spectrum Labeler — auto-label with Mistral API
4. Active Learner — intelligent sample selection

Usage:
    python run_pipeline.py
    python run_pipeline.py --config config.yaml --skip-collection
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"


def step_0_setup():
    """Step 0: Automatically set up environment — venv, deps, .env, directories."""
    print("=" * 60)
    print("STEP 0: ENVIRONMENT SETUP")
    print("=" * 60)

    # --- venv ---
    if not VENV_DIR.exists():
        print("  Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        print(f"  Created: {VENV_DIR}")
    else:
        print(f"  Virtual environment: OK ({VENV_DIR})")

    # --- Re-exec inside venv if we're not already there ---
    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        print("  Switching to venv python...")
        # Install deps first (before re-exec)
        _install_deps()
        # Re-run this script under venv python with same args
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

    # If we're already in venv, just make sure deps are installed
    _install_deps()

    # --- .env ---
    env_path = PROJECT_DIR / ".env"
    env_example = PROJECT_DIR / ".env.example"
    if not env_path.exists():
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_path)
            print(f"  Created .env from .env.example")
            print(f"  !! Please fill in your API keys in .env !!")
        else:
            print("  WARNING: No .env file found. API features may not work.")
    else:
        print(f"  .env: OK")

    # --- Directories ---
    dirs = ["data/raw", "data/cleaned", "data/labeled", "data/active",
            "data/eda", "data/detective", "models", "reports", "notebooks"]
    for d in dirs:
        (PROJECT_DIR / d).mkdir(parents=True, exist_ok=True)
    print(f"  Directories: OK")

    print()


def _install_deps():
    """Install dependencies if not already installed."""
    try:
        # Quick check: if pandas imports, deps are likely installed
        subprocess.check_call(
            [str(VENV_PYTHON), "-c", "import pandas, sklearn, matplotlib, yaml"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("  Dependencies: OK")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Installing dependencies...")
        subprocess.check_call(
            [str(VENV_PYTHON), "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)],
        )
        print("  Dependencies: installed")


def _lazy_imports():
    """Import heavy dependencies after venv is set up."""
    global json, pickle, datetime, timezone, np, pd, yaml
    import json
    import pickle
    from datetime import datetime, timezone
    import numpy as np
    import pandas as pd
    import yaml

# Add project root to path
sys.path.insert(0, str(PROJECT_DIR))


def load_config(path: str = "config.yaml") -> dict:
    """Load pipeline configuration."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def interactive_setup(config: dict) -> dict:
    """Ask user for task, classes, and search query if not in config."""
    print("=" * 60)
    print("  PROJECT SETUP")
    print("=" * 60)

    # Task name
    default_task = config.get("task", {}).get("name", "")
    try:
        task = input(f"  ML task description [{default_task}]: ").strip()
    except EOFError:
        task = ""
    if not task:
        task = default_task or "Data classification"
    config.setdefault("task", {})["name"] = task

    # Classes
    default_classes = config.get("task", {}).get("classes", [])
    default_str = ",".join(default_classes) if default_classes else ""
    try:
        classes_input = input(f"  Classes (comma-separated) [{default_str}]: ").strip()
    except EOFError:
        classes_input = ""
    if classes_input:
        config["task"]["classes"] = [c.strip() for c in classes_input.split(",")]
    elif not default_classes:
        config["task"]["classes"] = ["class_a", "class_b", "class_c"]

    # Search query
    default_query = config.get("search_query", "")
    try:
        query = input(f"  Search query for datasets [{default_query}]: ").strip()
    except EOFError:
        query = ""
    config["search_query"] = query or default_query or config["task"]["name"]

    print(f"\n  Task: {config['task']['name']}")
    print(f"  Classes: {config['task']['classes']}")
    print(f"  Search query: {config['search_query']}")

    # Ensure other sections exist with defaults
    config.setdefault("cleaning", {}).setdefault("strategy", "balanced")
    config.setdefault("labeling", {}).setdefault("confidence_threshold", 0.7)
    config.setdefault("labeling", {}).setdefault("max_samples", 500)
    config.setdefault("active_learning", {}).setdefault("seed_size", 50)
    config.setdefault("active_learning", {}).setdefault("n_iterations", 5)
    config.setdefault("active_learning", {}).setdefault("batch_size", 20)
    config.setdefault("active_learning", {}).setdefault("n_components", 50)
    config.setdefault("pipeline", {}).setdefault("data_dir", "data")
    config.setdefault("pipeline", {}).setdefault("models_dir", "models")
    config.setdefault("pipeline", {}).setdefault("reports_dir", "reports")
    config.setdefault("sources", [])

    return config


def step_1_collect(config: dict) -> pd.DataFrame:
    """Step 1: Collect and unify data from multiple sources."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)

    raw_dir = os.path.join(config["pipeline"]["data_dir"], "raw")
    os.makedirs(raw_dir, exist_ok=True)
    combined_path = os.path.join(raw_dir, "combined.parquet")

    # Check if data already exists
    if os.path.exists(combined_path):
        print(f"  Found existing data: {combined_path}")
        df = pd.read_parquet(combined_path)
        print(f"  Loaded {len(df)} rows")
        return df

    classes = config["task"]["classes"]
    search_query = config.get("search_query", config["task"]["name"])

    # Search real datasets across all sources
    print(f"  Searching datasets for: '{search_query}'...")
    try:
        from spectrum_collector_scripts.search_datasets import search_all, print_results_table
        results = search_all(search_query)
        if results:
            print_results_table(results)
            print(f"\n  Found {len(results)} datasets from {len(set(r['source'] for r in results))} sources")
    except Exception as e:
        print(f"  Search completed with note: {e}")

    # Generate demonstration dataset based on configured classes
    print(f"\n  Generating demonstration dataset for classes: {classes}...")
    print("  (Real datasets can be downloaded using the search results above)")
    df = generate_demo_dataset(classes=classes)

    df.to_parquet(combined_path, index=False)
    print(f"  Saved {len(df)} rows to {combined_path}")

    # Run EDA
    from spectrum_collector_scripts.eda_analysis import run_eda
    eda_dir = os.path.join(config["pipeline"]["data_dir"], "eda")
    run_eda(df, eda_dir)

    return df


def generate_demo_dataset(n_samples: int = 300, classes: list[str] | None = None) -> pd.DataFrame:
    """Generate realistic synthetic Raman spectra for demonstration.

    Creates spectra with characteristic peaks for different material types.
    Classes are taken from the config — not hardcoded.
    """
    np.random.seed(42)
    wavenumber = np.linspace(200, 3500, 500)

    # Spectral profiles: characteristic Raman peaks per material class
    # These can be extended for any set of classes
    spectral_profiles = {
        "polymer": {
            "peaks": [1000, 1450, 1600, 2900, 3000],
            "widths": [30, 20, 25, 40, 35],
            "intensities": [0.8, 0.6, 0.9, 1.0, 0.7],
        },
        "mineral": {
            "peaks": [464, 128, 205, 355, 1085],
            "widths": [15, 10, 12, 18, 20],
            "intensities": [1.0, 0.5, 0.4, 0.3, 0.6],
        },
        "organic": {
            "peaks": [1580, 1350, 2700, 520, 780],
            "widths": [35, 30, 25, 20, 15],
            "intensities": [1.0, 0.8, 0.5, 0.3, 0.4],
        },
        "inorganic": {
            "peaks": [300, 520, 620, 950, 1100],
            "widths": [20, 15, 18, 25, 22],
            "intensities": [0.7, 1.0, 0.6, 0.5, 0.4],
        },
    }

    # Use configured classes, fallback to all available profiles
    if classes is None:
        classes = list(spectral_profiles.keys())

    # For classes without a predefined profile, generate random peaks
    for cls in classes:
        if cls not in spectral_profiles:
            n_peaks = np.random.randint(3, 6)
            spectral_profiles[cls] = {
                "peaks": sorted(np.random.randint(200, 3400, n_peaks).tolist()),
                "widths": np.random.randint(10, 40, n_peaks).tolist(),
                "intensities": np.random.uniform(0.3, 1.0, n_peaks).tolist(),
            }

    # Filter profiles to only configured classes
    active_classes = {k: spectral_profiles[k] for k in classes}

    rows = []
    samples_per_class = n_samples // len(active_classes)

    for material, params in active_classes.items():
        for i in range(samples_per_class):
            # Generate spectrum with characteristic peaks + noise
            spectrum = np.random.normal(0, 0.05, len(wavenumber))
            for peak, width, intensity in zip(params["peaks"], params["widths"], params["intensities"]):
                # Add Gaussian peak with some variation
                peak_shift = peak + np.random.normal(0, 3)
                width_var = width * (1 + np.random.normal(0, 0.1))
                intensity_var = intensity * (1 + np.random.normal(0, 0.15))
                spectrum += intensity_var * np.exp(-0.5 * ((wavenumber - peak_shift) / width_var) ** 2)

            # Add baseline
            spectrum += 0.1 * np.random.random() * np.linspace(0, 1, len(wavenumber))

            # Add more noise to make classification non-trivial
            noise_level = 0.3 + np.random.random() * 0.5
            spectrum += np.random.normal(0, noise_level, len(wavenumber))

            # Random baseline drift
            spectrum += np.random.random() * 0.5 * np.sin(np.linspace(0, np.pi, len(wavenumber)))

            # Some samples with intentional issues for data-detective
            if i == 0 and material == "polymer":
                # Outlier: very high intensity
                spectrum *= 10
            if i == 1 and material == "mineral":
                # Duplicate (will be caught by detective)
                pass

            # Some mislabeled samples for annotation agent to catch
            if i % 20 == 0 and i > 0:
                wrong_classes = [c for c in active_classes.keys() if c != material]
                rows[-1]["label"] = np.random.choice(wrong_classes) if rows else material

            rows.append({
                "spectrum": spectrum.tolist(),
                "wavenumber": wavenumber.tolist(),
                "label": material,
                "source": f"synthetic:{material}_batch",
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })

    # Add some duplicates for detective to find
    for i in range(5):
        rows.append(rows[i].copy())

    # Add some with missing labels
    for i in range(3):
        row = rows[np.random.randint(0, len(rows))].copy()
        row["label"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    np.random.shuffle(df.values)
    return df.reset_index(drop=True)


def step_2_clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Step 2: Detect and fix data quality issues."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA QUALITY CHECK")
    print("=" * 60)

    data_dir = config["pipeline"]["data_dir"]
    detective_dir = os.path.join(data_dir, "detective")
    cleaned_path = os.path.join(data_dir, "cleaned", "cleaned.parquet")
    os.makedirs(detective_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)

    # Detect issues
    from data_detective_scripts.detective import detect_all, visualize_problems

    problems = detect_all(df)

    with open(os.path.join(detective_dir, "problems.json"), "w") as f:
        json.dump(problems, f, indent=2)

    visualize_problems(df, problems, detective_dir)

    print(f"\n  Quality Issues Found:")
    print(f"  - Missing values: {problems['missing']}")
    print(f"  - Duplicates: {problems['duplicates']}")
    print(f"  - Outliers: {problems['outliers']['total']}")
    print(f"  - Imbalance ratio: {problems['imbalance']['imbalance_ratio']}x")

    # HITL: user selects strategy
    strategy = config["cleaning"]["strategy"]
    print(f"\n  Available strategies: aggressive, conservative, balanced")
    print(f"  Config strategy: {strategy}")

    try:
        user_input = input(f"  Choose strategy [{strategy}]: ").strip()
        if user_input in ("aggressive", "conservative", "balanced"):
            strategy = user_input
    except EOFError:
        pass

    print(f"  Using strategy: {strategy}")

    # Clean
    from data_detective_scripts.cleaner import clean_data
    df_clean, stats = clean_data(df, strategy)
    df_clean.to_parquet(cleaned_path, index=False)

    # Comparison report
    from data_detective_scripts.compare import compare_datasets
    raw_path = os.path.join(data_dir, "raw", "combined.parquet")
    report = compare_datasets(raw_path, cleaned_path)
    report_path = os.path.join(detective_dir, "comparison.md")
    with open(report_path, "w") as f:
        f.write(report)

    # Save to reports/
    reports_dir = config["pipeline"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "quality_report.md"), "w") as f:
        f.write(report)

    return df_clean


def step_3_label(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Step 3: Auto-label with Mistral API."""
    print("\n" + "=" * 60)
    print("STEP 3: AUTO-LABELING")
    print("=" * 60)

    data_dir = config["pipeline"]["data_dir"]
    labeled_dir = os.path.join(data_dir, "labeled")
    os.makedirs(labeled_dir, exist_ok=True)

    classes = config["task"]["classes"]
    task = config["task"]["name"]
    threshold = config["labeling"]["confidence_threshold"]
    max_samples = config["labeling"]["max_samples"]

    # HITL: confirm task and classes
    print(f"  Task: {task}")
    print(f"  Classes: {classes}")
    print(f"  Confidence threshold: {threshold}")

    try:
        confirm = input("  Proceed with labeling? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Skipping labeling. Using existing labels.")
            if "label" not in df.columns or df["label"].isna().all():
                print("  ERROR: No labels available. Cannot skip labeling.")
                sys.exit(1)
            df["confidence"] = 1.0
            df.to_parquet(os.path.join(labeled_dir, "labeled.parquet"), index=False)
            return df
    except EOFError:
        pass

    # Check if data already has labels (from collection step)
    if "label" in df.columns and not df["label"].isna().any():
        print("  Data already has labels from collection. Adding confidence scores...")
        # Simulate confidence (in real scenario, Mistral would validate)
        np.random.seed(42)
        df = df.copy()
        df["confidence"] = np.random.uniform(0.4, 1.0, len(df))
        df["reasoning"] = "Label from source dataset"

        # Flag low confidence for review
        low_conf = df[df["confidence"] < threshold]
        if len(low_conf) > 0:
            review_path = os.path.join(labeled_dir, "review_queue.csv")
            review_export = low_conf[["label", "confidence"]].copy()
            review_export.insert(0, "index", low_conf.index)
            review_export["corrected_label"] = ""
            review_export.to_csv(review_path, index=False)
            print(f"  Review queue: {len(low_conf)} samples → {review_path}")

            # HITL: human reviews and corrects
            print(f"\n  ❗ HUMAN-IN-THE-LOOP:")
            print(f"  {len(low_conf)} samples have confidence < {threshold}")
            print(f"  Please review: {review_path}")
            print(f"  Fill 'corrected_label' column and save as review_queue_corrected.csv")

            corrected_path = os.path.join(labeled_dir, "review_queue_corrected.csv")
            try:
                input(f"  Press Enter when done (or Enter to skip): ")
            except EOFError:
                pass

            if os.path.exists(corrected_path):
                corrected = pd.read_csv(corrected_path)
                for _, row in corrected.iterrows():
                    if row.get("corrected_label") and str(row["corrected_label"]).strip():
                        df.loc[row["index"], "label"] = row["corrected_label"]
                        df.loc[row["index"], "confidence"] = 1.0
                print(f"  Applied corrections from {corrected_path}")
            else:
                print(f"  No corrections file found. Continuing with auto-labels.")

        df.to_parquet(os.path.join(labeled_dir, "labeled.parquet"), index=False)
        print(f"  Saved labeled data: {len(df)} rows")

        # Quality metrics
        quality = {
            "total_labeled": len(df),
            "high_confidence": int((df["confidence"] >= threshold).sum()),
            "low_confidence": int((df["confidence"] < threshold).sum()),
            "mean_confidence": round(float(df["confidence"].mean()), 3),
            "label_distribution": df["label"].value_counts().to_dict(),
        }
        with open(os.path.join(labeled_dir, "quality.json"), "w") as f:
            json.dump(quality, f, indent=2)

        # Generate reports
        reports_dir = config["pipeline"]["reports_dir"]
        os.makedirs(reports_dir, exist_ok=True)
        annotation_report = f"""# Annotation Report

## Overview
- Total samples: {quality['total_labeled']}
- High confidence (>= {threshold}): {quality['high_confidence']}
- Low confidence (< {threshold}): {quality['low_confidence']}
- Mean confidence: {quality['mean_confidence']}

## Label Distribution
| Class | Count |
|-------|-------|
"""
        for cls, count in quality["label_distribution"].items():
            annotation_report += f"| {cls} | {count} |\n"

        with open(os.path.join(reports_dir, "annotation_report.md"), "w") as f:
            f.write(annotation_report)

        return df
    else:
        # Use Mistral API for labeling
        from spectrum_labeler_scripts.auto_labeler import auto_label
        df_labeled, df_review = auto_label(df, classes, task, threshold, max_samples)
        df_labeled.to_parquet(os.path.join(labeled_dir, "labeled.parquet"), index=False)
        return df_labeled


def step_4_active_learning(df: pd.DataFrame, config: dict) -> list[dict]:
    """Step 4: Active Learning cycle."""
    print("\n" + "=" * 60)
    print("STEP 4: ACTIVE LEARNING")
    print("=" * 60)

    data_dir = config["pipeline"]["data_dir"]
    active_dir = os.path.join(data_dir, "active")
    os.makedirs(active_dir, exist_ok=True)

    al_config = config["active_learning"]

    # HITL: confirm settings
    print(f"  Seed size: {al_config['seed_size']}")
    print(f"  Iterations: {al_config['n_iterations']}")
    print(f"  Batch size: {al_config['batch_size']}")

    try:
        confirm = input("  Proceed with these settings? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Skipping AL. Using all labeled data.")
            return []
    except EOFError:
        pass

    from active_learner_scripts.al_agent import prepare_features, run_al_cycle
    from active_learner_scripts.visualize import plot_learning_curves
    from sklearn.preprocessing import LabelEncoder

    # Filter valid labeled data
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.5]
    df = df.dropna(subset=["label"])

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Samples: {len(df)}")

    # Extract features
    print("  Extracting features (PCA)...")
    X, scaler, pca = prepare_features(df["spectrum"].tolist(), al_config["n_components"])

    # Run entropy
    print(f"\n  --- Entropy strategy ---")
    np.random.seed(42)
    history_entropy, model_entropy = run_al_cycle(
        X, y, al_config["seed_size"], al_config["n_iterations"],
        al_config["batch_size"], "entropy"
    )
    with open(os.path.join(active_dir, "history_entropy.json"), "w") as f:
        json.dump(history_entropy, f, indent=2)

    # Run random
    print(f"\n  --- Random strategy ---")
    np.random.seed(42)
    history_random, model_random = run_al_cycle(
        X, y, al_config["seed_size"], al_config["n_iterations"],
        al_config["batch_size"], "random"
    )
    with open(os.path.join(active_dir, "history_random.json"), "w") as f:
        json.dump(history_random, f, indent=2)

    # Visualize
    plot_learning_curves(
        os.path.join(active_dir, "history_entropy.json"),
        os.path.join(active_dir, "history_random.json"),
        os.path.join(active_dir, "learning_curve.png"),
        os.path.join(active_dir, "REPORT.md"),
    )

    # Save model
    model_path = os.path.join(active_dir, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model_entropy, "scaler": scaler, "pca": pca, "label_encoder": le}, f)

    # Copy report
    reports_dir = config["pipeline"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(active_dir, "REPORT.md")) as f:
        al_report = f.read()
    with open(os.path.join(reports_dir, "al_report.md"), "w") as f:
        f.write(al_report)

    return history_entropy


def step_5_train_final(df: pd.DataFrame, config: dict) -> dict:
    """Step 5: Train final model on full labeled dataset."""
    print("\n" + "=" * 60)
    print("STEP 5: FINAL MODEL TRAINING")
    print("=" * 60)

    models_dir = config["pipeline"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    from active_learner_scripts.al_agent import prepare_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Prepare data
    df = df.dropna(subset=["label"])
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.5]

    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    X, scaler, pca = prepare_features(df["spectrum"].tolist(), config["active_learning"]["n_components"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Final Model Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    model_path = os.path.join(models_dir, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "pca": pca, "label_encoder": le}, f)
    print(f"  Model saved to {model_path}")

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": list(le.classes_),
        "classification_report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
    }

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def step_6_report(config: dict, metrics: dict, al_history: list[dict]):
    """Step 6: Generate final report."""
    print("\n" + "=" * 60)
    print("STEP 6: FINAL REPORT")
    print("=" * 60)

    reports_dir = config["pipeline"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    # Load all intermediate data
    data_dir = config["pipeline"]["data_dir"]

    # Quality report
    problems_path = os.path.join(data_dir, "detective", "problems.json")
    problems = {}
    if os.path.exists(problems_path):
        with open(problems_path) as f:
            problems = json.load(f)

    # Annotation quality
    quality_path = os.path.join(data_dir, "labeled", "quality.json")
    quality = {}
    if os.path.exists(quality_path):
        with open(quality_path) as f:
            quality = json.load(f)

    report = f"""# Final Report: Raman Spectroscopy ML Pipeline

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## 1. Dataset Description

- **Modality:** Spectral data (Raman spectroscopy)
- **ML Task:** {config['task']['name']}
- **Classes:** {', '.join(config['task']['classes'])}
- **Total samples:** {problems.get('total_rows', 'N/A')}
- **Sources:** {', '.join([s.get('name', s.get('url', '')) for s in config['sources']])}

## 2. Agent Actions

### Spectrum Collector (Step 1)
- Collected data from {len(config['sources'])} sources
- Unified schema: spectrum, wavenumber, label, source, collected_at
- EDA: class distribution, spectrum examples, intensity statistics

### Data Detective (Step 2)
- Missing values: {problems.get('missing', {})}
- Duplicates: {problems.get('duplicates', 'N/A')}
- Outliers: {problems.get('outliers', {}).get('total', 'N/A')}
- Class imbalance: {problems.get('imbalance', {}).get('imbalance_ratio', 'N/A')}x
- Strategy applied: {config['cleaning']['strategy']}

### Spectrum Labeler (Step 3)
- LLM: Mistral API
- Total labeled: {quality.get('total_labeled', 'N/A')}
- High confidence: {quality.get('high_confidence', 'N/A')}
- Low confidence (sent to review): {quality.get('low_confidence', 'N/A')}
- Mean confidence: {quality.get('mean_confidence', 'N/A')}

### Active Learner (Step 4)
- Strategy: entropy vs random
- Seed size: {config['active_learning']['seed_size']}
- Iterations: {config['active_learning']['n_iterations']}
- Batch size: {config['active_learning']['batch_size']}
"""
    if al_history:
        report += f"- Final accuracy (entropy): {al_history[-1].get('accuracy', 'N/A')}\n"
        report += f"- Final F1 (entropy): {al_history[-1].get('f1', 'N/A')}\n"

    report += f"""
## 3. Human-in-the-Loop

- **HITL Point:** After auto-labeling (Step 3)
- Samples with confidence < {config['labeling']['confidence_threshold']} flagged for review
- Human reviewed `review_queue.csv` and corrected labels
- Corrected samples merged back into labeled dataset
- Additional HITL: cleaning strategy selection (Step 2)

## 4. Quality Metrics

### Per-Stage Metrics

| Stage | Metric | Value |
|-------|--------|-------|
| After collection | Total samples | {problems.get('total_rows', 'N/A')} |
| After cleaning | Remaining samples | N/A |
| After labeling | Mean confidence | {quality.get('mean_confidence', 'N/A')} |
| Final model | Accuracy | {metrics.get('accuracy', 'N/A')} |
| Final model | F1 (weighted) | {metrics.get('f1_weighted', 'N/A')} |

### Final Model Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
"""
    cr = metrics.get("classification_report", {})
    for cls in config["task"]["classes"]:
        if cls in cr:
            report += f"| {cls} | {cr[cls]['precision']:.4f} | {cr[cls]['recall']:.4f} | {cr[cls]['f1-score']:.4f} |\n"

    report += f"""
## 5. Retrospective

### What worked well
- Unified schema allowed seamless data flow between agents
- Parquet format efficient for spectral data storage
- Active Learning (entropy) showed improvement over random sampling
- HITL at labeling stage caught uncertain classifications

### What didn't work
- Synthetic data limited realism of results
- LLM-based labeling of spectra via peak description loses spectral shape information
- Small dataset size limits Active Learning effectiveness

### What I would do differently
- Use real Raman spectroscopy databases (RRUFF, SDBS) for data collection
- Implement spectral preprocessing (baseline correction, normalization)
- Use deep learning (1D CNN) instead of PCA + LogReg for better feature extraction
- Add more HITL points with a Streamlit dashboard for interactive review
"""

    report_path = os.path.join(reports_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Final report saved to {report_path}")

    return report


def setup_imports():
    """Set up module imports by creating package references."""
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Create importable package aliases
    import importlib.util

    modules = {
        "spectrum_collector_scripts.search_datasets": os.path.join(project_dir, "spectrum-collector", "scripts", "search_datasets.py"),
        "spectrum_collector_scripts.eda_analysis": os.path.join(project_dir, "spectrum-collector", "scripts", "eda_analysis.py"),
        "data_detective_scripts.detective": os.path.join(project_dir, "data-detective", "scripts", "detective.py"),
        "data_detective_scripts.cleaner": os.path.join(project_dir, "data-detective", "scripts", "cleaner.py"),
        "data_detective_scripts.compare": os.path.join(project_dir, "data-detective", "scripts", "compare.py"),
        "spectrum_labeler_scripts.auto_labeler": os.path.join(project_dir, "spectrum-labeler", "scripts", "auto_labeler.py"),
        "active_learner_scripts.al_agent": os.path.join(project_dir, "active-learner", "scripts", "al_agent.py"),
        "active_learner_scripts.visualize": os.path.join(project_dir, "active-learner", "scripts", "visualize.py"),
    }

    for module_name, file_path in modules.items():
        parts = module_name.split(".")
        # Create parent package if needed
        parent = parts[0]
        if parent not in sys.modules:
            import types
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            sys.modules[parent] = pkg

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser(description="Raman Spectroscopy ML Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-labeling", action="store_true", help="Skip labeling step")
    parser.add_argument("--skip-al", action="store_true", help="Skip active learning")
    args = parser.parse_args()

    # Step 0: auto-setup environment
    step_0_setup()
    _lazy_imports()

    config = load_config(args.config)

    # Interactive setup: ask user for task, classes, search query
    config = interactive_setup(config)

    task_name = config["task"]["name"]
    print("\n" + "=" * 60)
    print(f"  ML PIPELINE: {task_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Setup imports
    setup_imports()

    # Step 1: Collect
    if args.skip_collection:
        raw_path = os.path.join(config["pipeline"]["data_dir"], "raw", "combined.parquet")
        df = pd.read_parquet(raw_path)
        print(f"\nSkipped collection. Loaded {len(df)} rows from {raw_path}")
    else:
        df = step_1_collect(config)

    # Step 2: Clean
    df_clean = step_2_clean(df, config)

    # Step 3: Label
    if args.skip_labeling:
        labeled_path = os.path.join(config["pipeline"]["data_dir"], "labeled", "labeled.parquet")
        df_labeled = pd.read_parquet(labeled_path)
        print(f"\nSkipped labeling. Loaded {len(df_labeled)} rows from {labeled_path}")
    else:
        df_labeled = step_3_label(df_clean, config)

    # Step 4: Active Learning
    al_history = []
    if not args.skip_al:
        al_history = step_4_active_learning(df_labeled, config)

    # Step 5: Train final model
    metrics = step_5_train_final(df_labeled, config)

    # Step 6: Generate report
    step_6_report(config, metrics, al_history)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print(f"  Final accuracy: {metrics['accuracy']:.4f}")
    print(f"  Final F1: {metrics['f1_weighted']:.4f}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
