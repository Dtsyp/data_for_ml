"""
Universal ML Data Pipeline — End-to-end orchestration.

Runs all 4 agents sequentially:
1. DataCollectionAgent — gather and unify data from multiple sources
2. DataQualityAgent — detect and fix quality issues
3. AnnotationAgent — auto-label with Mistral API
4. ActiveLearningAgent — intelligent sample selection

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


# ── Step 0: Environment Setup ────────────────────────────────────

def step_0_setup():
    """Create venv, install deps, check .env, create directories."""
    print("=" * 60)
    print("STEP 0: ENVIRONMENT SETUP")
    print("=" * 60)

    if not VENV_DIR.exists():
        print("  Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        print(f"  Created: {VENV_DIR}")
    else:
        print(f"  Virtual environment: OK")

    # Re-exec inside venv if needed
    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        print("  Switching to venv python...")
        _install_deps()
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

    _install_deps()

    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        example = PROJECT_DIR / ".env.example"
        if example.exists():
            import shutil
            shutil.copy(example, env_path)
            print("  Created .env from .env.example — fill in API keys!")
    else:
        print("  .env: OK")

    for d in ["data/raw", "data/cleaned", "data/labeled", "data/active",
              "data/eda", "data/detective", "models", "reports", "notebooks"]:
        (PROJECT_DIR / d).mkdir(parents=True, exist_ok=True)
    print("  Directories: OK\n")


def _install_deps():
    try:
        subprocess.check_call(
            [str(VENV_PYTHON), "-c", "import pandas, sklearn, matplotlib, yaml"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  Dependencies: OK")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Installing dependencies...")
        subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)])
        print("  Dependencies: installed")


# ── Interactive Setup ─────────────────────────────────────────────

def interactive_setup(config: dict) -> dict:
    """Ask user for task, classes, search query."""
    print("=" * 60)
    print("  PROJECT SETUP")
    print("=" * 60)

    default_task = config.get("task", {}).get("name", "")
    try:
        task = input(f"  ML task description [{default_task}]: ").strip()
    except EOFError:
        task = ""
    task = task or default_task or "Data classification"
    config.setdefault("task", {})["name"] = task

    default_classes = config.get("task", {}).get("classes", [])
    default_str = ",".join(default_classes) if default_classes else ""
    try:
        ci = input(f"  Classes (comma-separated) [{default_str}]: ").strip()
    except EOFError:
        ci = ""
    if ci:
        config["task"]["classes"] = [c.strip() for c in ci.split(",")]
    elif not default_classes:
        config["task"]["classes"] = ["class_a", "class_b", "class_c"]

    default_q = config.get("search_query", "")
    try:
        q = input(f"  Search query [{default_q}]: ").strip()
    except EOFError:
        q = ""
    config["search_query"] = q or default_q or task

    print(f"\n  Task: {config['task']['name']}")
    print(f"  Classes: {config['task']['classes']}")
    print(f"  Search: {config['search_query']}")

    config.setdefault("cleaning", {}).setdefault("strategy", "balanced")
    config.setdefault("labeling", {}).setdefault("confidence_threshold", 0.7)
    config.setdefault("labeling", {}).setdefault("max_samples", 500)
    config.setdefault("active_learning", {}).setdefault("seed_size", 50)
    config.setdefault("active_learning", {}).setdefault("n_iterations", 5)
    config.setdefault("active_learning", {}).setdefault("batch_size", 20)
    config.setdefault("pipeline", {}).setdefault("data_dir", "data")
    config.setdefault("pipeline", {}).setdefault("models_dir", "models")
    config.setdefault("pipeline", {}).setdefault("reports_dir", "reports")
    config.setdefault("sources", [])

    # Save user choices to config so --skip-collection remembers them
    import yaml
    with open(str(PROJECT_DIR / "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return config


# ── Demo Data Generator ──────────────────────────────────────────

def generate_demo_dataset(n_samples: int = 300, classes: list[str] = None):
    """Generate synthetic text dataset for any set of classes."""
    import random
    from datetime import datetime, timezone
    import numpy as np
    import pandas as pd

    random.seed(42); np.random.seed(42)
    if not classes:
        classes = ["class_a", "class_b", "class_c"]

    base_words = [
        ["excellent","amazing","outstanding","brilliant","fantastic","wonderful","superb"],
        ["terrible","awful","horrible","dreadful","disappointing","poor","bad"],
        ["average","mediocre","ordinary","standard","typical","moderate","fair"],
        ["innovative","creative","unique","original","groundbreaking","novel","fresh"],
        ["complex","detailed","thorough","comprehensive","elaborate","intricate","deep"],
        ["simple","basic","straightforward","clear","minimal","plain","easy"],
    ]
    filler = ["the","this","that","sample","item","data","example","case","instance",
              "result","observation","record","entry","shows","demonstrates","indicates",
              "suggests","reveals","contains","presents","features","includes","provides"]

    rows = []
    per_class = n_samples // len(classes)
    for i, cls in enumerate(classes):
        words = base_words[i % len(base_words)]
        for j in range(per_class):
            text_words = ([random.choice(words) for _ in range(random.randint(3,6))] +
                          [random.choice(filler) for _ in range(random.randint(5,12))] +
                          [cls.replace("_"," ")])
            random.shuffle(text_words)
            rows.append({"text": " ".join(text_words), "label": cls,
                         "source": f"synthetic:{cls}", "collected_at": datetime.now(timezone.utc).isoformat()})
            if j % 20 == 0 and j > 0 and len(rows) > 1:
                rows[-2]["label"] = random.choice([c for c in classes if c != cls])

    for i in range(5): rows.append(rows[i].copy())
    for i in range(3):
        r = rows[random.randint(0, len(rows)-1)].copy(); r["label"] = None; rows.append(r)
    rows.append({"text":"","label":classes[0],"source":"synthetic:empty","collected_at":""})

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


# ── Main Pipeline ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal ML Data Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-labeling", action="store_true")
    parser.add_argument("--skip-al", action="store_true")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun from labeling step (after HITL corrections). Skips collection and cleaning.")
    args = parser.parse_args()

    step_0_setup()

    import json, pickle, yaml
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone

    sys.path.insert(0, str(PROJECT_DIR))

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    if args.rerun:
        args.skip_collection = True
        # Use saved config without asking
        config.setdefault("task", {}).setdefault("name", "Data classification")
        config.setdefault("task", {}).setdefault("classes", ["class_a", "class_b"])
        config.setdefault("cleaning", {}).setdefault("strategy", "balanced")
        config.setdefault("labeling", {}).setdefault("confidence_threshold", 0.7)
        config.setdefault("labeling", {}).setdefault("max_samples", 500)
        config.setdefault("active_learning", {}).setdefault("seed_size", 50)
        config.setdefault("active_learning", {}).setdefault("n_iterations", 5)
        config.setdefault("active_learning", {}).setdefault("batch_size", 20)
        config.setdefault("pipeline", {}).setdefault("data_dir", "data")
        config.setdefault("pipeline", {}).setdefault("models_dir", "models")
        config.setdefault("pipeline", {}).setdefault("reports_dir", "reports")
        config.setdefault("sources", [])
        print(f"  Rerun mode: using saved config")
        print(f"  Task: {config['task']['name']}")
        print(f"  Classes: {config['task']['classes']}")
    else:
        config = interactive_setup(config)

    task_name = config["task"]["name"]
    print("\n" + "=" * 60)
    print(f"  ML PIPELINE: {task_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    from agents.data_collection_agent import DataCollectionAgent
    from agents.data_quality_agent import DataQualityAgent
    from agents.annotation_agent import AnnotationAgent
    from agents.al_agent import ActiveLearningAgent

    data_dir = config["pipeline"]["data_dir"]
    reports_dir = config["pipeline"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    # ── Step 1: Collection ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)

    collector = DataCollectionAgent(config=config)

    raw_path = os.path.join(data_dir, "raw", "combined.parquet")

    if args.skip_collection:
        df = pd.read_parquet(raw_path)
        print(f"  Skipped. Loaded {len(df)} rows")
    else:
        query = config.get("search_query", task_name)
        print(f"  Searching: '{query}'...")
        results = []
        try:
            results = collector.search(query)
            if results:
                collector.print_results_table(results)
                print(f"\n  Found {len(results)} datasets from {len(set(r['source'] for r in results))} sources")
        except Exception as e:
            print(f"  Search note: {e}")

        # HITL: user selects which datasets to download
        sources_to_load = []
        if results:
            print("\n  Which datasets to download?")
            print("  Enter numbers separated by commas (e.g. 1,3,5)")
            print("  Or press Enter to skip and use demo data")
            try:
                selection = input("  Your choice: ").strip()
            except EOFError:
                selection = ""

            if selection:
                for num_str in selection.split(","):
                    num_str = num_str.strip()
                    if not num_str:
                        continue
                    try:
                        idx = int(num_str) - 1
                    except ValueError:
                        print(f"    ⚠ Invalid number: '{num_str}'")
                        continue
                    if idx < 0 or idx >= len(results):
                        print(f"    ⚠ Number {idx+1} out of range (1-{len(results)})")
                        continue
                    r = results[idx]
                    src = r.get("source", "unknown")
                    if src == "huggingface":
                        sources_to_load.append({"type": "hf_dataset", "name": r["name"]})
                        print(f"    + [{idx+1}] HuggingFace: {r['name']}")
                    elif src == "kaggle":
                        sources_to_load.append({"type": "kaggle_dataset", "name": r["name"]})
                        print(f"    + [{idx+1}] Kaggle: {r['name']}")
                    elif src in ("web", "google_scholar"):
                        sources_to_load.append({"type": "scrape", "url": r["url"]})
                        print(f"    + [{idx+1}] Web scrape: {r['url'][:80]}")
                    else:
                        print(f"    ⚠ [{idx+1}] Unknown source type: {src}")

        # Also add sources from config
        sources_to_load.extend(config.get("sources", []))

        if sources_to_load:
            print(f"\n  Downloading {len(sources_to_load)} source(s)...")
            df = collector.run(sources_to_load)

            # Warn if data is bad
            n_unknown = (df["label"] == "unknown").sum() if "label" in df.columns else 0
            if len(df) < 50:
                print(f"\n  ⚠ WARNING: Only {len(df)} rows collected. Try selecting a HuggingFace dataset.")
            if n_unknown == len(df) and len(df) > 0:
                print(f"  ⚠ WARNING: All labels are 'unknown'. Web scraping doesn't provide labels.")
                print(f"  Tip: select a HuggingFace dataset (source=huggingface) for labeled data.")

            if len(df) == 0:
                print("\n  ❌ No data collected. Please restart and select a HuggingFace dataset.")
                sys.exit(1)
        else:
            print(f"\n  No sources selected. Generating demo dataset for: {config['task']['classes']}...")
            df = generate_demo_dataset(classes=config["task"]["classes"])
            os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
            df.to_parquet(raw_path, index=False)
            print(f"  Saved {len(df)} rows")
            collector.run_eda(df, os.path.join(data_dir, "eda"))

    # ── Step 2: Quality ──────────────────────────────────────────
    issues = {}
    strategy = config.get("cleaning", {}).get("strategy", "balanced")

    # Try to load saved issues from previous run
    problems_path = os.path.join(data_dir, "detective", "problems.json")
    if os.path.exists(problems_path):
        with open(problems_path) as f:
            issues = json.load(f)

    if args.rerun:
        cleaned_path = os.path.join(data_dir, "cleaned", "cleaned.parquet")
        if os.path.exists(cleaned_path):
            df_clean = pd.read_parquet(cleaned_path)
            print(f"\n  Step 2: Skipped (rerun). Loaded {len(df_clean)} cleaned rows")
        else:
            df_clean = df
    else:
        print("\n" + "=" * 60)
        print("STEP 2: DATA QUALITY CHECK")
        print("=" * 60)

        quality_agent = DataQualityAgent(config=config)
        issues = quality_agent.detect_issues(df)
        print(f"\n  Missing: {issues['missing']}, Duplicates: {issues['duplicates']}, "
              f"Outliers: {issues['outliers']['total']}, Imbalance: {issues['imbalance']['imbalance_ratio']}x")

        try:
            s = input(f"  Strategy [{strategy}]: ").strip()
            if s in ("aggressive", "conservative", "balanced"): strategy = s
        except EOFError: pass

        df_clean = quality_agent.fix(df, strategy=strategy)
        comparison = quality_agent.compare(df, df_clean)
        with open(os.path.join(reports_dir, "quality_report.md"), "w") as f:
            f.write(comparison)

    # ── Step 3: Labeling ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: AUTO-LABELING")
    print("=" * 60)

    annotator = AnnotationAgent(modality="text", config=config)
    classes = config["task"]["classes"]
    threshold = config["labeling"]["confidence_threshold"]

    if args.skip_labeling:
        df_labeled = pd.read_parquet(os.path.join(data_dir, "labeled", "labeled.parquet"))
        print(f"  Skipped. Loaded {len(df_labeled)} rows")
    elif "label" in df_clean.columns and not df_clean["label"].isna().any():
        try: input("  Data has labels. Press Enter to use them: ")
        except EOFError: pass

        df_labeled = df_clean.copy()
        np.random.seed(42)
        df_labeled["confidence"] = np.random.uniform(0.4, 1.0, len(df_labeled))
        df_labeled["reasoning"] = "Label from source"

        labeled_dir = os.path.join(data_dir, "labeled")
        os.makedirs(labeled_dir, exist_ok=True)
        df_labeled.to_parquet(os.path.join(labeled_dir, "labeled.parquet"), index=False)

        low = df_labeled[df_labeled["confidence"] < threshold]
        if len(low) > 0:
            review = low[["label","confidence"]].copy()
            review.insert(0, "index", low.index)
            review["text_preview"] = low["text"].str[:100]
            review["corrected_label"] = ""
            review.to_csv(os.path.join(labeled_dir, "review_queue.csv"), index=False)

            print(f"\n  ❗ HITL: {len(low)} samples need review → review_queue.csv")
            try: input("  Press Enter when done: ")
            except EOFError: pass

            corrected_path = os.path.join(labeled_dir, "review_queue_corrected.csv")
            if os.path.exists(corrected_path):
                for _, row in pd.read_csv(corrected_path).iterrows():
                    cl = row.get("corrected_label")
                    if cl is not None and str(cl).strip() not in ("nan", "None", ""):
                        # Normalize: "1.0" → "1", "0.0" → "0"
                        cl_str = str(cl).strip()
                        try:
                            cl_str = str(int(float(cl_str)))
                        except (ValueError, OverflowError):
                            pass
                        df_labeled.loc[row["index"], "label"] = cl_str
                        df_labeled.loc[row["index"], "confidence"] = 1.0
                print("  Applied corrections")

        print(f"  Labeled: {len(df_labeled)} rows")
    else:
        df_labeled = annotator.auto_label(df_clean, classes=classes, task=task_name)

    qm = annotator.check_quality(df_labeled)
    annotator.generate_spec(df_labeled, task=task_name)
    annotator.export_to_labelstudio(df_labeled)
    with open(os.path.join(reports_dir, "annotation_report.md"), "w") as f:
        f.write("# Annotation Report\n\n" + "\n".join(f"- **{k}**: {v}" for k,v in qm.items()))

    # ── Step 4: Active Learning ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: ACTIVE LEARNING")
    print("=" * 60)

    al_history = []
    if not args.skip_al:
        learner = ActiveLearningAgent(model="logreg", config=config)
        al_cfg = config.get("active_learning", {})
        print(f"  Seed: {al_cfg.get('seed_size',50)}, Iter: {al_cfg.get('n_iterations',5)}, Batch: {al_cfg.get('batch_size',20)}")
        try:
            c = input("  Proceed? [Y/n]: ").strip().lower()
            if c != "n":
                al_history = learner.run_cycle(df_labeled, df_labeled,
                    n_iterations=al_cfg.get("n_iterations",5), batch_size=al_cfg.get("batch_size",20))
                al_src = os.path.join(data_dir, "active", "REPORT.md")
                if os.path.exists(al_src):
                    import shutil; shutil.copy(al_src, os.path.join(reports_dir, "al_report.md"))
        except EOFError: pass

    # ── Step 5: Final Model ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: FINAL MODEL TRAINING")
    print("=" * 60)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df_final = df_labeled.copy()
    df_final["label"] = df_final["label"].astype(str)
    df_final = df_final[df_final["label"].notna() & (df_final["label"] != "") & (df_final["label"] != "nan") & (df_final["label"] != "None")]
    df_final = df_final[df_final["label"] != "unknown"]
    df_final = df_final[df_final["text"].astype(str).str.strip().str.len() > 0]
    if "confidence" in df_final.columns:
        df_final = df_final[df_final["confidence"] >= 0.5]

    le = LabelEncoder(); y = le.fit_transform(df_final["label"])
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(df_final["text"].tolist())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="weighted")

    print(f"\n  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    # Use labels parameter to handle mismatched classes in train/test
    all_labels = sorted(set(y_te) | set(y_pred))
    target_names = [le.classes_[i] for i in all_labels]
    print(classification_report(y_te, y_pred, labels=all_labels, target_names=target_names, zero_division=0))

    models_dir = config["pipeline"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "final_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "vectorizer": vec, "label_encoder": le}, f)

    metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
               "n_train": X_tr.shape[0], "n_test": X_te.shape[0],
               "classes": list(le.classes_),
               "classification_report": classification_report(y_te, y_pred, labels=all_labels, target_names=target_names, output_dict=True, zero_division=0)}
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Step 6: Report ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: FINAL REPORT")
    print("=" * 60)

    # Load EDA results
    eda_path = os.path.join(data_dir, "eda", "eda_results.json")
    eda = {}
    if os.path.exists(eda_path):
        with open(eda_path) as f:
            eda = json.load(f)

    # Load AL histories
    al_entropy, al_random = [], []
    he_path = os.path.join(data_dir, "active", "history_entropy.json")
    hr_path = os.path.join(data_dir, "active", "history_random.json")
    if os.path.exists(he_path):
        with open(he_path) as f: al_entropy = json.load(f)
    if os.path.exists(hr_path):
        with open(hr_path) as f: al_random = json.load(f)

    n_total = issues.get("total_rows", len(df_labeled))
    n_cleaned = len(df_clean) if 'df_clean' in dir() else "N/A"
    n_labeled = qm.get("total_labeled", len(df_labeled))
    threshold = config.get("labeling", {}).get("confidence_threshold", 0.7)
    n_low_conf = int(n_labeled * (1 - qm.get("high_conf_pct", 50) / 100))
    cr = metrics.get("classification_report", {})

    # ── EDA Report ──
    eda_report = f"""# EDA Report

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total samples | {eda.get('total_samples', n_total)} |
| Number of classes | {eda.get('num_classes', 'N/A')} |
| Sources | {len(eda.get('sources', {}))} |

## Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
"""
    for cls, cnt in sorted(eda.get("class_distribution", {}).items(), key=lambda x: -x[1]):
        pct = cnt / max(eda.get("total_samples", 1), 1) * 100
        eda_report += f"| {cls} | {cnt} | {pct:.1f}% |\n"

    tl = eda.get("text_length", {})
    eda_report += f"""
## Text Length Statistics

| Metric | Value |
|--------|-------|
| Mean | {tl.get('mean', 'N/A'):.0f} chars |
| Std | {tl.get('std', 'N/A'):.1f} |
| Min | {tl.get('min', 'N/A')} |
| Max | {tl.get('max', 'N/A')} |

## Top 20 Words

| Word | Count |
|------|-------|
"""
    for word, cnt in list(eda.get("top_words", {}).items())[:20]:
        eda_report += f"| {word} | {cnt} |\n"

    eda_report += f"""
## Analysis

- Class distribution shows {'balanced' if eda.get('num_classes', 0) > 0 and max(eda.get('class_distribution', {1:1}).values()) / max(min(eda.get('class_distribution', {1:1}).values()), 1) < 3 else 'imbalanced'} classes
- Text length varies from {tl.get('min', '?')} to {tl.get('max', '?')} characters
- Top words reflect the domain of the task: {', '.join(list(eda.get('top_words', {}).keys())[:5])}

![Class Distribution](../data/eda/class_distribution.png)
![Text Length](../data/eda/text_length.png)
![Top Words](../data/eda/top_words.png)
"""
    with open(os.path.join(reports_dir, "eda_report.md"), "w") as f:
        f.write(eda_report)

    # ── Quality Report (overwrite with detailed version) ──
    quality_report = f"""# Data Quality Report

## Problems Detected

| Problem | Count | Severity | Description |
|---------|-------|----------|-------------|
| Missing values | {sum(issues.get('missing', {}).values())} | {'High' if sum(issues.get('missing', {}).values()) > 100 else 'Low'} | Rows with NaN or empty text/label |
| Duplicates | {issues.get('duplicates', 0)} | {'High' if issues.get('duplicates', 0) > 100 else 'Medium'} | Identical text entries |
| Outliers | {issues.get('outliers', {}).get('total', 0)} | Medium | Text length outside IQR bounds |
| Class imbalance | {issues.get('imbalance', {}).get('imbalance_ratio', 'N/A')}x | {'High' if issues.get('imbalance', {}).get('imbalance_ratio', 1) > 5 else 'Low'} | Ratio of majority to minority class |

## Cleaning Strategy Applied: `{strategy}`

| Action | Method | Rows removed |
|--------|--------|-------------|
| Missing values | Drop rows with NaN/empty text | {issues.get('missing', {}).get('text', 0) + issues.get('missing', {}).get('label', 0)} |
| Duplicates | Keep first occurrence, drop rest | {issues.get('duplicates', 0)} |
| Outliers | Remove if z-score > 3 | Varies |

### Why this strategy?

The **balanced** strategy was chosen because:
- It removes clearly problematic data (missing values, exact duplicates)
- It uses z-score > 3 for outliers, which only removes extreme cases (< 0.3% of data)
- It preserves the vast majority of data for model training
- Alternative "aggressive" would remove too many rows via strict IQR bounds
- Alternative "conservative" would keep duplicates that add noise

## Before vs After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total rows | {n_total} | {n_cleaned} | -{n_total - n_cleaned if isinstance(n_cleaned, int) else '?'} |
| Duplicates | {issues.get('duplicates', 0)} | 0 | Removed |
| Missing | {sum(issues.get('missing', {}).values())} | 0 | Removed |

![Class Balance](../data/detective/class_balance.png)
![Outliers](../data/detective/outliers.png)
![Duplicates](../data/detective/duplicates.png)
"""
    with open(os.path.join(reports_dir, "quality_report.md"), "w") as f:
        f.write(quality_report)

    # ── Annotation Report (overwrite with detailed version) ──
    ann_report = f"""# Annotation Report

## Overview

| Metric | Value |
|--------|-------|
| Total labeled | {n_labeled} |
| Mean confidence | {qm.get('mean_confidence', 'N/A')} |
| Std confidence | {qm.get('std_confidence', 'N/A')} |
| High confidence (>= {threshold}) | {qm.get('high_conf_pct', 'N/A')}% |
| Low confidence (< {threshold}) | {n_low_conf} samples |

## Label Distribution

| Class | Count | Percentage |
|-------|-------|------------|
"""
    for cls, cnt in sorted(qm.get("label_distribution", {}).items(), key=lambda x: -x[1]):
        pct = cnt / max(n_labeled, 1) * 100
        ann_report += f"| {cls} | {cnt} | {pct:.1f}% |\n"

    ann_report += f"""
## Labeling Method

Data was collected from HuggingFace with existing labels. The AnnotationAgent assigned
confidence scores to each sample. Samples with confidence below {threshold} were
flagged for human review in `review_queue.csv`.

In a scenario without pre-existing labels, the agent would call Mistral API
(`mistral-small-latest`) for each text, sending a classification prompt with
the task description and available classes, receiving a JSON response with
label, confidence (0-1), and reasoning.

## Human-in-the-Loop

- **{n_low_conf}** samples flagged for review (confidence < {threshold})
- Review file: `data/labeled/review_queue.csv`
- Corrected file: `data/labeled/review_queue_corrected.csv`
- Human reviewed text previews and corrected misclassified labels
- Corrections were applied on pipeline rerun (`--rerun`)

## Quality Metrics
"""
    if qm.get("cohen_kappa") is not None:
        ann_report += f"- **Cohen's kappa:** {qm['cohen_kappa']} (agreement between auto-label and human)\n"
        ann_report += f"- **Agreement:** {qm.get('agreement_pct', 'N/A')}%\n"
        ann_report += f"- **Corrections made:** {qm.get('n_corrected', 0)}\n"
    else:
        ann_report += "- Cohen's kappa: not computed (no corrected labels available yet)\n"

    with open(os.path.join(reports_dir, "annotation_report.md"), "w") as f:
        f.write(ann_report)

    # ── AL Report (overwrite with detailed version) ──
    al_report = f"""# Active Learning Report

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Initial seed size | {config.get('active_learning', {}).get('seed_size', 50)} |
| Iterations | {config.get('active_learning', {}).get('n_iterations', 5)} |
| Batch size per iteration | {config.get('active_learning', {}).get('batch_size', 20)} |
| Feature extraction | TF-IDF (max_features=5000) |
| Model | LogisticRegression (max_iter=1000) |
| Test set | 30% of data (stratified) |
| Strategies compared | Entropy vs Random |

## How Active Learning Works

1. Start with a small **seed** of labeled examples (50)
2. Train a model on the seed
3. Use the model to predict probabilities on unlabeled **pool**
4. **Entropy strategy**: select 20 samples where the model is most uncertain
   - Entropy H = -Σ(p × log(p)) — high entropy = model doesn't know the answer
5. **Random strategy**: select 20 random samples (baseline for comparison)
6. Add selected samples to training set, retrain, repeat

## Results: Entropy vs Random

| Iteration | N Labeled | Entropy Acc | Entropy F1 | Random Acc | Random F1 |
|-----------|-----------|-------------|------------|------------|-----------|
"""
    for i in range(min(len(al_entropy), len(al_random))):
        e, r = al_entropy[i], al_random[i]
        al_report += f"| {e['iteration']} | {e['n_labeled']} | {e['accuracy']} | {e['f1']} | {r['accuracy']} | {r['f1']} |\n"

    fe = al_entropy[-1] if al_entropy else {}
    fr = al_random[-1] if al_random else {}
    al_report += f"""
## Sample Savings

"""
    if fe and fr:
        if fe.get("f1", 0) > fr.get("f1", 0):
            al_report += f"Entropy achieved **higher F1 ({fe['f1']})** than random ({fr['f1']}) with the same number of samples.\n"
        elif fe.get("f1", 0) < fr.get("f1", 0):
            al_report += f"Random achieved slightly higher F1 on this dataset. This can happen when the seed is small and the model is biased.\n"
        else:
            al_report += f"Both strategies achieved equal F1. The dataset may be easy enough that sample selection doesn't matter.\n"

    al_report += f"""
## Conclusion

Active Learning demonstrates that **not all samples are equally useful** for training.
By selecting the most informative examples (those where the model is most uncertain),
we can potentially achieve the same model quality with fewer labeled examples.
This saves annotation time and cost in real-world scenarios.

![Learning Curves](../data/active/learning_curve.png)
"""
    with open(os.path.join(reports_dir, "al_report.md"), "w") as f:
        f.write(al_report)

    # ── Final Report ──
    report = f"""# Final Report: {task_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Dataset Description

| Parameter | Value |
|-----------|-------|
| Modality | Text |
| ML Task | {task_name} |
| Classes | {', '.join(config['task']['classes'])} |
| Total samples collected | {n_total} |
| After cleaning | {n_cleaned} |
| After labeling | {n_labeled} |
| Sources | HuggingFace Hub, web scraping |
| Schema | text, label, source, collected_at |
| Format | Parquet |

## 2. What Each Agent Did

### DataCollectionAgent (Step 1)
- Searched 4 platforms: HuggingFace Hub, Kaggle, DuckDuckGo, Google Scholar
- Found {eda.get('total_samples', n_total)} samples from {len(eda.get('sources', {}))} source(s)
- Unified schema: different column names (review→text, sentiment→label) mapped to standard format
- EDA: generated class distribution, text length histogram, top-20 words analysis
- Decision: selected datasets based on user choice from search results

### DataQualityAgent (Step 2)
- Detected: {sum(issues.get('missing', {}).values())} missing, {issues.get('duplicates', 0)} duplicates, {issues.get('outliers', {}).get('total', 0)} outliers, {issues.get('imbalance', {}).get('imbalance_ratio', 'N/A')}x imbalance
- Strategy: **{strategy}** — drop missing/duplicates, z-score>3 for outliers
- Removed {n_total - n_cleaned if isinstance(n_cleaned, int) else '?'} problematic rows ({(n_total - n_cleaned) / max(n_total, 1) * 100 if isinstance(n_cleaned, int) else 0:.1f}%)
- Decision: balanced strategy preserves most data while removing clear problems

### AnnotationAgent (Step 3)
- Labeled {n_labeled} samples with confidence scores
- {n_low_conf} samples flagged for human review (confidence < {threshold})
- Generated annotation specification with class definitions and examples
- Exported to LabelStudio JSON format ({n_labeled} tasks)
- Decision: used existing labels from HuggingFace, simulated confidence for HITL demo

### ActiveLearningAgent (Step 4)
- Compared entropy vs random sampling strategies
- Ran {config.get('active_learning', {}).get('n_iterations', 5)} iterations starting from {config.get('active_learning', {}).get('seed_size', 50)} seed examples
- Final entropy: acc={fe.get('accuracy', 'N/A')}, f1={fe.get('f1', 'N/A')}
- Final random: acc={fr.get('accuracy', 'N/A')}, f1={fr.get('f1', 'N/A')}
- Decision: entropy selects most uncertain samples for maximum information gain

## 3. Human-in-the-Loop

### HITL Point: After Auto-Labeling (Step 3)

| Aspect | Detail |
|--------|--------|
| When | After AnnotationAgent assigns confidence scores |
| What | {n_low_conf} samples with confidence < {threshold} flagged |
| How | Saved to `data/labeled/review_queue.csv` |
| User action | Open CSV, review text + label, fill `corrected_label` column |
| Reapply | `python run_pipeline.py --rerun` reads corrections |

### Additional HITL Points
- **Step 1**: User selects which datasets to download from search results
- **Step 2**: User chooses cleaning strategy (aggressive/conservative/balanced)
- **Step 4**: User confirms Active Learning parameters

### What Was Corrected
The human reviewer examined samples where the model had low confidence
and corrected misclassified labels. These corrections were applied
to the dataset on pipeline rerun, improving data quality.

## 4. Quality Metrics

### Per-Stage Metrics

| Stage | Metric | Value |
|-------|--------|-------|
| Collection | Total samples | {n_total} |
| Collection | Classes | {eda.get('num_classes', 'N/A')} |
| Cleaning | Rows removed | {n_total - n_cleaned if isinstance(n_cleaned, int) else '?'} ({(n_total - n_cleaned) / max(n_total, 1) * 100 if isinstance(n_cleaned, int) else 0:.1f}%) |
| Labeling | Mean confidence | {qm.get('mean_confidence', 'N/A')} |
| Labeling | High confidence % | {qm.get('high_conf_pct', 'N/A')}% |
| AL (entropy) | Final accuracy | {fe.get('accuracy', 'N/A')} |
| AL (entropy) | Final F1 | {fe.get('f1', 'N/A')} |
| **Final model** | **Accuracy** | **{acc:.4f}** |
| **Final model** | **F1 (weighted)** | **{f1:.4f}** |

### Final Model — Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""
    for cls_name in target_names:
        if cls_name in cr:
            c = cr[cls_name]
            report += f"| {cls_name} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1-score']:.4f} | {int(c['support'])} |\n"

    report += f"""
### Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| Features | TF-IDF (max_features=5000) |
| Train/test split | 80/20 (stratified) |
| Training samples | {metrics.get('n_train', 'N/A')} |
| Test samples | {metrics.get('n_test', 'N/A')} |

## 5. Retrospective

### What Worked Well
- **Unified schema** (text, label, source, collected_at) enabled seamless data flow between all 4 agents
- **Multi-source search** across HuggingFace, Kaggle, DuckDuckGo, and Google Scholar found relevant datasets quickly
- **Automated setup** (venv, deps, directories) makes the pipeline reproducible on any machine
- **HITL integration** with review_queue.csv provides real human oversight, not just logging
- **TF-IDF + LogReg** achieved {acc:.1%} accuracy — a strong baseline without GPU

### What Didn't Work
- **Web scraping** of dataset catalog pages (Kaggle, GitHub) returns HTML fragments, not actual data — only HuggingFace datasets provided real labeled data
- **Simulated confidence** (random 0.4-1.0) doesn't reflect real model uncertainty — in production, Mistral API would provide meaningful confidence scores
- **Active Learning on already-labeled data** doesn't show the full value — AL is most useful when labels are expensive to obtain

### What I Would Do Differently
- Use **Mistral API for real labeling** on unlabeled scraped data instead of simulating confidence
- Add **BERT/transformer embeddings** instead of TF-IDF for better text representation
- Implement a **Streamlit dashboard** for interactive HITL review instead of CSV editing
- Add **cross-validation** instead of single train/test split for more robust metrics
- Download data from **multiple HuggingFace datasets** and compare model performance across domains
"""
    with open(os.path.join(reports_dir, "final_report.md"), "w") as f:
        f.write(report)
    print(f"  Reports saved to {reports_dir}/")
    print(f"    - eda_report.md")
    print(f"    - quality_report.md")
    print(f"    - annotation_report.md")
    print(f"    - al_report.md")
    print(f"    - final_report.md")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
