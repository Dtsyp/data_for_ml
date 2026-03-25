"""Active Learning agent for Raman spectroscopy classification."""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_features(spectra: list[list], n_components: int = 50) -> tuple:
    """Extract features from spectra using PCA."""
    # Pad/truncate spectra to same length
    max_len = max(len(s) for s in spectra)
    padded = np.array([
        np.pad(s, (0, max(0, max_len - len(s))), mode="constant")[:max_len]
        for s in spectra
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(padded)

    n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, scaler, pca


def entropy_query(model, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
    """Select samples with highest entropy."""
    proba = model.predict_proba(X_pool)
    # Entropy: H(p) = -sum(p * log(p))
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    indices = np.argsort(entropy)[-batch_size:]
    return indices


def margin_query(model, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
    """Select samples with smallest margin between top-2 predictions."""
    proba = model.predict_proba(X_pool)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    indices = np.argsort(margin)[:batch_size]
    return indices


def random_query(X_pool: np.ndarray, batch_size: int) -> np.ndarray:
    """Select random samples (baseline)."""
    indices = np.random.choice(len(X_pool), size=min(batch_size, len(X_pool)), replace=False)
    return indices


STRATEGIES = {
    "entropy": entropy_query,
    "margin": margin_query,
}


def run_al_cycle(
    X: np.ndarray,
    y: np.ndarray,
    seed_size: int = 50,
    n_iterations: int = 5,
    batch_size: int = 20,
    strategy: str = "entropy",
    test_size: float = 0.3,
) -> tuple[list[dict], object]:
    """Run Active Learning cycle.

    Returns:
        history: list of {iteration, n_labeled, accuracy, f1}
        model: final trained model
    """
    # Split into test set (fixed) and pool
    X_pool_all, X_test, y_pool_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Initial seed
    seed_idx = np.random.choice(len(X_pool_all), size=min(seed_size, len(X_pool_all)), replace=False)
    pool_mask = np.ones(len(X_pool_all), dtype=bool)
    pool_mask[seed_idx] = False

    X_labeled = X_pool_all[seed_idx]
    y_labeled = y_pool_all[seed_idx]
    X_pool = X_pool_all[pool_mask]
    y_pool = y_pool_all[pool_mask]

    history = []

    # Initial evaluation
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_labeled, y_labeled)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    history.append({
        "iteration": 0,
        "n_labeled": len(X_labeled),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
    })
    print(f"  Iter 0: n_labeled={len(X_labeled)}, acc={acc:.4f}, f1={f1:.4f}")

    # AL iterations
    for i in range(1, n_iterations + 1):
        if len(X_pool) == 0:
            print(f"  Iter {i}: pool exhausted")
            break

        actual_batch = min(batch_size, len(X_pool))

        if strategy == "random":
            selected_idx = random_query(X_pool, actual_batch)
        else:
            query_fn = STRATEGIES[strategy]
            selected_idx = query_fn(model, X_pool, actual_batch)

        # Move selected from pool to labeled
        X_labeled = np.vstack([X_labeled, X_pool[selected_idx]])
        y_labeled = np.concatenate([y_labeled, y_pool[selected_idx]])

        # Remove from pool
        X_pool = np.delete(X_pool, selected_idx, axis=0)
        y_pool = np.delete(y_pool, selected_idx, axis=0)

        # Retrain
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_labeled, y_labeled)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        history.append({
            "iteration": i,
            "n_labeled": len(X_labeled),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
        })
        print(f"  Iter {i}: n_labeled={len(X_labeled)}, acc={acc:.4f}, f1={f1:.4f}")

    return history, model


def main():
    parser = argparse.ArgumentParser(description="Active Learning for Raman spectra")
    parser.add_argument("--input", required=True, help="Input labeled parquet")
    parser.add_argument("--output-dir", default="data/active", help="Output directory")
    parser.add_argument("--seed-size", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--strategy", default="entropy", choices=["entropy", "margin", "random"])
    parser.add_argument("--n-components", type=int, default=50, help="PCA components")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Filter out unknown labels and low confidence
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.5]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"Classes: {list(le.classes_)}")

    # Extract features
    print("Extracting features (PCA)...")
    X, scaler, pca = prepare_features(df["spectrum"].tolist(), args.n_components)

    os.makedirs(args.output_dir, exist_ok=True)

    # Run entropy strategy
    print(f"\n--- Strategy: entropy ---")
    np.random.seed(42)
    history_entropy, model_entropy = run_al_cycle(
        X, y, args.seed_size, args.iterations, args.batch_size, "entropy"
    )

    with open(os.path.join(args.output_dir, "history_entropy.json"), "w") as f:
        json.dump(history_entropy, f, indent=2)

    # Run random strategy (baseline)
    print(f"\n--- Strategy: random ---")
    np.random.seed(42)
    history_random, model_random = run_al_cycle(
        X, y, args.seed_size, args.iterations, args.batch_size, "random"
    )

    with open(os.path.join(args.output_dir, "history_random.json"), "w") as f:
        json.dump(history_random, f, indent=2)

    # Save final model (entropy is usually better)
    model_path = os.path.join(args.output_dir, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model_entropy,
            "scaler": scaler,
            "pca": pca,
            "label_encoder": le,
        }, f)
    print(f"\nFinal model saved to {model_path}")

    # Summary
    final_entropy = history_entropy[-1]
    final_random = history_random[-1]
    print(f"\n--- Comparison ---")
    print(f"  Entropy: acc={final_entropy['accuracy']:.4f}, f1={final_entropy['f1']:.4f}")
    print(f"  Random:  acc={final_random['accuracy']:.4f}, f1={final_random['f1']:.4f}")

    if final_entropy["f1"] > final_random["f1"]:
        # Find how many samples random needs to reach entropy's f1
        target_f1 = final_entropy["f1"]
        for step in history_random:
            if step["f1"] >= target_f1:
                saved = step["n_labeled"] - final_entropy["n_labeled"]
                print(f"  Entropy saved ~{max(0, saved)} samples vs random for same F1")
                break
        else:
            print(f"  Random never reached entropy's F1 — entropy is strictly better")


if __name__ == "__main__":
    main()
