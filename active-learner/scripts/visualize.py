"""Visualize Active Learning results: learning curves comparison."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(
    entropy_path: str,
    random_path: str,
    output_path: str,
    report_path: str | None = None,
):
    """Plot learning curves for entropy vs random strategies."""
    with open(entropy_path) as f:
        history_entropy = json.load(f)
    with open(random_path) as f:
        history_random = json.load(f)

    # Extract data
    n_entropy = [h["n_labeled"] for h in history_entropy]
    acc_entropy = [h["accuracy"] for h in history_entropy]
    f1_entropy = [h["f1"] for h in history_entropy]

    n_random = [h["n_labeled"] for h in history_random]
    acc_random = [h["accuracy"] for h in history_random]
    f1_random = [h["f1"] for h in history_random]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    ax1.plot(n_entropy, acc_entropy, "o-", color="steelblue", label="Entropy", linewidth=2)
    ax1.plot(n_random, acc_random, "s--", color="tomato", label="Random", linewidth=2)
    ax1.set_title("Accuracy vs Number of Labeled Samples")
    ax1.set_xlabel("N labeled")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1
    ax2.plot(n_entropy, f1_entropy, "o-", color="steelblue", label="Entropy", linewidth=2)
    ax2.plot(n_random, f1_random, "s--", color="tomato", label="Random", linewidth=2)
    ax2.set_title("F1 Score vs Number of Labeled Samples")
    ax2.set_xlabel("N labeled")
    ax2.set_ylabel("F1 (weighted)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Active Learning: Entropy vs Random Sampling", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to {output_path}")

    # Generate report
    if report_path:
        final_e = history_entropy[-1]
        final_r = history_random[-1]

        report = f"""# Active Learning Report

## Strategy Comparison

| Metric | Entropy | Random |
|--------|---------|--------|
| Final Accuracy | {final_e['accuracy']:.4f} | {final_r['accuracy']:.4f} |
| Final F1 | {final_e['f1']:.4f} | {final_r['f1']:.4f} |
| Samples used | {final_e['n_labeled']} | {final_r['n_labeled']} |

## Learning Curves

![Learning Curves](learning_curve.png)

## Iteration Details

### Entropy Strategy
| Iteration | N Labeled | Accuracy | F1 |
|-----------|-----------|----------|----|
"""
        for h in history_entropy:
            report += f"| {h['iteration']} | {h['n_labeled']} | {h['accuracy']:.4f} | {h['f1']:.4f} |\n"

        report += """
### Random Strategy (Baseline)
| Iteration | N Labeled | Accuracy | F1 |
|-----------|-----------|----------|----|
"""
        for h in history_random:
            report += f"| {h['iteration']} | {h['n_labeled']} | {h['accuracy']:.4f} | {h['f1']:.4f} |\n"

        # Conclusion
        if final_e["f1"] > final_r["f1"]:
            diff = final_e["f1"] - final_r["f1"]
            report += f"""
## Conclusion

Entropy-based Active Learning outperformed random sampling by **{diff:.4f} F1** with the same number of labeled samples.
This demonstrates that intelligent sample selection can improve model quality without additional labeling cost.
"""
        else:
            report += """
## Conclusion

Random sampling performed comparably to entropy-based selection on this dataset.
This may indicate that the dataset is relatively homogeneous or that the model
can learn well from any subset of the data.
"""

        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AL learning curves")
    parser.add_argument("--entropy", required=True, help="Entropy history JSON")
    parser.add_argument("--random", required=True, help="Random history JSON")
    parser.add_argument("--output", default="data/active/learning_curve.png")
    parser.add_argument("--report", default="data/active/REPORT.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot_learning_curves(args.entropy, args.random, args.output, args.report)


if __name__ == "__main__":
    main()
