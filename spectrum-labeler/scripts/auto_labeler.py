"""Auto-label Raman spectra using Mistral API."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_mistral_client():
    """Lazy import and create Mistral client."""
    try:
        from mistralai import Mistral
    except ImportError:
        print("ERROR: mistralai not installed. Run: pip install mistralai")
        sys.exit(1)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY not set in .env")
        sys.exit(1)
    return Mistral(api_key=api_key)


def extract_peaks(spectrum: list, wavenumber: list, n_peaks: int = 10) -> list[dict]:
    """Extract top N peaks from spectrum for LLM classification."""
    arr = np.array(spectrum)
    wn = np.array(wavenumber) if len(wavenumber) == len(spectrum) else np.linspace(200, 3500, len(spectrum))

    # Simple peak detection: local maxima
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append({"wavenumber": round(float(wn[i]), 1), "intensity": round(float(arr[i]), 2)})

    # Sort by intensity, take top N
    peaks.sort(key=lambda x: x["intensity"], reverse=True)
    return peaks[:n_peaks]


def classify_with_mistral(
    peaks: list[dict],
    classes: list[str],
    task: str,
    client,
    model: str = "mistral-small-latest",
) -> dict:
    """Classify spectrum using Mistral API based on peaks."""
    peaks_str = ", ".join([f"{p['wavenumber']} cm⁻¹ (I={p['intensity']})" for p in peaks[:8]])

    prompt = f"""You are an expert in Raman spectroscopy.
Task: {task}
Available classes: {', '.join(classes)}

Raman spectrum peaks: {peaks_str}

Based on the characteristic Raman peaks, classify this spectrum.
Respond ONLY with valid JSON:
{{"label": "<one of the classes>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    try:
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON from response
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        # Validate label
        if result.get("label") not in classes:
            # Try to find closest match
            label_lower = result.get("label", "").lower()
            for cls in classes:
                if cls.lower() in label_lower or label_lower in cls.lower():
                    result["label"] = cls
                    break
            else:
                result["label"] = classes[0]
                result["confidence"] = 0.1

        return result

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {"label": classes[0], "confidence": 0.0, "reasoning": f"Parse error: {e}"}
    except Exception as e:
        return {"label": classes[0], "confidence": 0.0, "reasoning": f"API error: {e}"}


def auto_label(
    df: pd.DataFrame,
    classes: list[str],
    task: str,
    confidence_threshold: float = 0.7,
    max_samples: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Auto-label dataset using Mistral API.

    Returns:
        labeled_df: DataFrame with labels and confidence
        review_df: Low-confidence samples for manual review
    """
    client = get_mistral_client()

    df = df.head(max_samples).copy()
    labels = []
    confidences = []
    reasonings = []

    total = len(df)
    print(f"Labeling {total} samples with Mistral API...")

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{total}")

        peaks = extract_peaks(row["spectrum"], row.get("wavenumber", []))
        result = classify_with_mistral(peaks, classes, task, client)

        labels.append(result.get("label", classes[0]))
        confidences.append(result.get("confidence", 0.0))
        reasonings.append(result.get("reasoning", ""))

        # Rate limiting
        time.sleep(0.5)

    df["label"] = labels
    df["confidence"] = confidences
    df["reasoning"] = reasonings

    # Split by confidence
    high_conf = df[df["confidence"] >= confidence_threshold].copy()
    low_conf = df[df["confidence"] < confidence_threshold].copy()

    print(f"\nLabeling complete:")
    print(f"  High confidence (>= {confidence_threshold}): {len(high_conf)}")
    print(f"  Low confidence (< {confidence_threshold}): {len(low_conf)} → review_queue")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    return df, low_conf


def generate_spec(classes: list[str], task: str, df_labeled: pd.DataFrame) -> str:
    """Generate annotation specification."""
    spec = f"""# Annotation Specification

## Task
{task}

## Classes

"""
    for cls in classes:
        examples = df_labeled[df_labeled["label"] == cls].head(3)
        spec += f"### {cls}\n"
        spec += f"- Count in dataset: {len(df_labeled[df_labeled['label'] == cls])}\n"
        if len(examples) > 0:
            for _, ex in examples.iterrows():
                peaks = extract_peaks(ex["spectrum"], ex.get("wavenumber", []), n_peaks=5)
                peaks_str = ", ".join([f"{p['wavenumber']} cm⁻¹" for p in peaks])
                spec += f"- Example peaks: {peaks_str}\n"
                if "reasoning" in ex and ex["reasoning"]:
                    spec += f"  - Reasoning: {ex['reasoning']}\n"
        spec += "\n"

    spec += """## Edge Cases
- If spectrum has very few peaks → low confidence, send to manual review
- If peaks match multiple classes → choose most likely, mark confidence < 0.5
- If spectrum appears noisy → flag for review
"""
    return spec


def export_labelstudio(df: pd.DataFrame, output_path: str):
    """Export to LabelStudio JSON format."""
    tasks = []
    for i, (_, row) in enumerate(df.iterrows()):
        peaks = extract_peaks(row["spectrum"], row.get("wavenumber", []), n_peaks=5)
        peaks_str = ", ".join([f"{p['wavenumber']} cm⁻¹ (I={p['intensity']})" for p in peaks])

        task = {
            "id": i,
            "data": {
                "text": f"Raman peaks: {peaks_str}",
                "spectrum_length": len(row["spectrum"]),
                "source": row.get("source", "unknown"),
            },
        }
        if "label" in row and pd.notna(row["label"]):
            task["predictions"] = [{
                "result": [{
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                    "value": {"choices": [row["label"]]},
                }],
                "score": float(row.get("confidence", 0.5)),
            }]
        tasks.append(task)

    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"LabelStudio export: {len(tasks)} tasks → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-label Raman spectra with Mistral")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--classes", required=True, help="Comma-separated class names")
    parser.add_argument("--task", default="Classify Raman spectrum material type")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Auto-label
    df_labeled, df_review = auto_label(df, classes, args.task, args.confidence_threshold, args.max_samples)

    # Save outputs
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    df_labeled.to_parquet(args.output, index=False)
    print(f"Labeled data saved to {args.output}")

    # Review queue
    review_path = os.path.join(output_dir, "review_queue.csv")
    if len(df_review) > 0:
        review_export = df_review[["label", "confidence"]].copy()
        review_export.insert(0, "index", df_review.index)
        # Add peaks for human review
        review_export["spectrum_peaks"] = df_review["spectrum"].apply(
            lambda x: str([round(float(v), 1) for v in extract_peaks(x, [], 5)])
        )
        review_export["corrected_label"] = ""
        review_export.to_csv(review_path, index=False)
        print(f"Review queue saved to {review_path}")

    # Spec
    spec = generate_spec(classes, args.task, df_labeled)
    spec_path = os.path.join(output_dir, "spec.md")
    with open(spec_path, "w") as f:
        f.write(spec)
    print(f"Annotation spec saved to {spec_path}")

    # Quality metrics
    quality = {
        "total_labeled": len(df_labeled),
        "high_confidence": int((df_labeled["confidence"] >= args.confidence_threshold).sum()),
        "low_confidence": int((df_labeled["confidence"] < args.confidence_threshold).sum()),
        "mean_confidence": round(float(df_labeled["confidence"].mean()), 3),
        "label_distribution": df_labeled["label"].value_counts().to_dict(),
    }
    quality_path = os.path.join(output_dir, "quality.json")
    with open(quality_path, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"Quality metrics saved to {quality_path}")

    # LabelStudio export
    ls_path = os.path.join(output_dir, "labelstudio.json")
    export_labelstudio(df_labeled, ls_path)


if __name__ == "__main__":
    main()
