"""AnnotationAgent — auto-labels data, generates specs, checks quality.

Technical contract (Assignment 3):
    agent = AnnotationAgent(modality='text')
    df_labeled = agent.auto_label(df)
    spec = agent.generate_spec(df, task='sentiment_classification')
    metrics = agent.check_quality(df_labeled)
    agent.export_to_labelstudio(df_labeled)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _get_mistral_client():
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


class AnnotationAgent:
    """Agent that auto-labels data with LLM and manages annotation quality."""

    def __init__(self, modality: str = "text", config="config.yaml"):
        self.modality = modality
        if isinstance(config, dict):
            self.config = config
        else:
            import yaml
            with open(config) as f:
                self.config = yaml.safe_load(f) or {}

    def auto_label(self, df: pd.DataFrame, classes: list[str] = None,
                   task: str = None, confidence_threshold: float = None,
                   max_samples: int = None) -> pd.DataFrame:
        """Auto-label data using Mistral API."""
        classes = classes or self.config.get("task", {}).get("classes", [])
        task = task or self.config.get("task", {}).get("name", "Classify this text")
        confidence_threshold = confidence_threshold or self.config.get("labeling", {}).get("confidence_threshold", 0.7)
        max_samples = max_samples or self.config.get("labeling", {}).get("max_samples", 500)

        client = _get_mistral_client()
        df = df.head(max_samples).copy()
        labels, confidences, reasonings = [], [], []

        print(f"Labeling {len(df)} samples with Mistral API...")
        for i, (_, row) in enumerate(df.iterrows()):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(df)}")
            result = self._classify(row["text"], classes, task, client)
            labels.append(result["label"])
            confidences.append(result["confidence"])
            reasonings.append(result["reasoning"])
            time.sleep(0.5)

        df["label"] = labels
        df["confidence"] = confidences
        df["reasoning"] = reasonings

        # Save
        labeled_dir = os.path.join(self.config.get("pipeline", {}).get("data_dir", "data"), "labeled")
        os.makedirs(labeled_dir, exist_ok=True)
        df.to_parquet(os.path.join(labeled_dir, "labeled.parquet"), index=False)

        # Review queue
        low = df[df["confidence"] < confidence_threshold]
        if len(low) > 0:
            review = low[["label", "confidence"]].copy()
            review.insert(0, "index", low.index)
            review["text_preview"] = low["text"].str[:100]
            review["corrected_label"] = ""
            review.to_csv(os.path.join(labeled_dir, "review_queue.csv"), index=False)

        print(f"  High confidence: {len(df) - len(low)}, Low: {len(low)} → review_queue")
        return df

    def generate_spec(self, df: pd.DataFrame, task: str = None) -> str:
        """Generate annotation specification with classes, examples, edge cases."""
        classes = df["label"].unique().tolist() if "label" in df.columns else []
        task = task or self.config.get("task", {}).get("name", "Classification")

        spec = f"# Annotation Specification\n\n## Task\n{task}\n\n## Classes\n"
        for cls in classes:
            examples = df[df["label"] == cls].head(3)
            spec += f"\n### {cls}\n- Count: {len(df[df['label'] == cls])}\n"
            for _, ex in examples.iterrows():
                preview = ex["text"][:100] + "..." if len(ex["text"]) > 100 else ex["text"]
                spec += f'- Example: "{preview}"\n'

        spec += "\n## Edge Cases\n"
        spec += "- If text is ambiguous or could belong to multiple classes → choose most likely, mark low confidence\n"
        spec += "- If text is too short to classify → flag for manual review\n"
        spec += "- If text contains mixed signals → assign primary class, note in reasoning\n"

        labeled_dir = os.path.join(self.config.get("pipeline", {}).get("data_dir", "data"), "labeled")
        os.makedirs(labeled_dir, exist_ok=True)
        with open(os.path.join(labeled_dir, "spec.md"), "w") as f:
            f.write(spec)
        return spec

    def check_quality(self, df_labeled: pd.DataFrame) -> dict:
        """Check quality: Cohen's kappa, confidence stats, label distribution."""
        result = {
            "label_distribution": df_labeled["label"].value_counts().to_dict(),
            "total_labeled": len(df_labeled),
        }

        if "confidence" in df_labeled.columns:
            conf = df_labeled["confidence"]
            threshold = self.config.get("labeling", {}).get("confidence_threshold", 0.7)
            result["mean_confidence"] = round(float(conf.mean()), 3)
            result["std_confidence"] = round(float(conf.std()), 3)
            result["high_conf_pct"] = round(float((conf >= threshold).mean() * 100), 1)

        if "corrected_label" in df_labeled.columns:
            mask = df_labeled["corrected_label"].notna() & (df_labeled["corrected_label"] != "")
            if mask.sum() > 0:
                from sklearn.metrics import cohen_kappa_score
                auto = df_labeled.loc[mask, "label"]
                corrected = df_labeled.loc[mask, "corrected_label"]
                try:
                    result["cohen_kappa"] = round(float(cohen_kappa_score(auto, corrected)), 3)
                    result["n_corrected"] = int(mask.sum())
                    result["agreement_pct"] = round(float((auto == corrected).mean() * 100), 1)
                except Exception:
                    result["cohen_kappa"] = None

        labeled_dir = os.path.join(self.config.get("pipeline", {}).get("data_dir", "data"), "labeled")
        os.makedirs(labeled_dir, exist_ok=True)
        with open(os.path.join(labeled_dir, "quality.json"), "w") as f:
            json.dump(result, f, indent=2)
        return result

    def export_to_labelstudio(self, df: pd.DataFrame, output_path: str = None) -> str:
        """Export to LabelStudio JSON format."""
        if output_path is None:
            labeled_dir = os.path.join(self.config.get("pipeline", {}).get("data_dir", "data"), "labeled")
            output_path = os.path.join(labeled_dir, "labelstudio.json")

        tasks = []
        for i, (_, row) in enumerate(df.iterrows()):
            task = {"id": i, "data": {"text": row["text"][:1000], "source": row.get("source", "")}}
            if "label" in row and pd.notna(row["label"]):
                task["predictions"] = [{"result": [{"from_name": "label", "to_name": "text",
                    "type": "choices", "value": {"choices": [row["label"]]}}],
                    "score": float(row.get("confidence", 0.5))}]
            tasks.append(task)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        print(f"  LabelStudio export: {len(tasks)} tasks → {output_path}")
        return output_path

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _classify(text, classes, task, client, model="mistral-small-latest"):
        preview = text[:500] + "..." if len(text) > 500 else text
        prompt = f"""Task: {task}
Available classes: {', '.join(classes)}

Text: {preview}

Classify this text. Respond ONLY with valid JSON:
{{"label": "<class>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}"""
        try:
            resp = client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            t = resp.choices[0].message.content.strip()
            if "```" in t:
                t = t.split("```")[1]
                if t.startswith("json"): t = t[4:]
                t = t.strip()
            result = json.loads(t)
            if result.get("label") not in classes:
                for c in classes:
                    if c.lower() in result.get("label","").lower():
                        result["label"] = c; break
                else:
                    result["label"] = classes[0]; result["confidence"] = 0.1
            return result
        except Exception as e:
            return {"label": classes[0], "confidence": 0.0, "reasoning": str(e)}
