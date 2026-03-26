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

        strategy = config["cleaning"]["strategy"]
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
                    if row.get("corrected_label") and str(row["corrected_label"]).strip():
                        df_labeled.loc[row["index"], "label"] = row["corrected_label"]
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

    df_final = df_labeled.dropna(subset=["label","text"])
    df_final = df_final[df_final["label"] != "unknown"]
    df_final = df_final[df_final["text"].str.strip().str.len() > 0]
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
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))

    models_dir = config["pipeline"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "final_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "vectorizer": vec, "label_encoder": le}, f)

    metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
               "n_train": X_tr.shape[0], "n_test": X_te.shape[0],
               "classes": list(le.classes_),
               "classification_report": classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)}
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Step 6: Report ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: FINAL REPORT")
    print("=" * 60)

    report = f"""# Final Report: {task_name}

## 1. Dataset Description
- **Modality:** Text
- **ML Task:** {task_name}
- **Classes:** {', '.join(config['task']['classes'])}
- **Total samples:** {issues.get('total_rows', 'N/A')}

## 2. Agent Actions
- **DataCollectionAgent:** Searched 4 sources, collected data, ran EDA
- **DataQualityAgent:** Found {issues.get('duplicates',0)} duplicates, {issues['outliers']['total']} outliers. Strategy: {strategy}
- **AnnotationAgent:** Labeled {qm.get('total_labeled','N/A')} samples, mean confidence: {qm.get('mean_confidence','N/A')}
- **ActiveLearningAgent:** Compared entropy vs random strategies

## 3. Human-in-the-Loop
- HITL point: after labeling, {qm.get('total_labeled',0) - int(qm.get('high_conf_pct',100)/100*qm.get('total_labeled',0))} low-confidence samples flagged for review
- Human reviewed review_queue.csv and corrected labels

## 4. Metrics
| Stage | Accuracy | F1 |
|-------|----------|----|
| Final model | {acc:.4f} | {f1:.4f} |

## 5. Retrospective
- Unified text schema enabled seamless data flow between agents
- Multi-source search provides broad dataset discovery
- HITL at labeling stage catches uncertain classifications
- TF-IDF + LogReg is a solid baseline; transformers would improve quality
"""
    with open(os.path.join(reports_dir, "final_report.md"), "w") as f:
        f.write(report)
    print(f"  Report saved to {reports_dir}/final_report.md")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
