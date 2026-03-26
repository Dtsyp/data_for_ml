"""DataCollectionAgent — collects data from multiple sources and returns a unified dataset.

Technical contract (Assignment 1):
    agent = DataCollectionAgent(config='config.yaml')
    df = agent.run(sources=[
        {'type': 'hf_dataset', 'name': 'imdb'},
        {'type': 'scrape', 'url': '...', 'selector': '...'},
    ])
    # → pd.DataFrame with columns: text, label, source, collected_at
"""

import json
import os
from collections import Counter
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


class DataCollectionAgent:
    """Agent that collects data from multiple sources and unifies the schema."""

    def __init__(self, config="config.yaml"):
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
                self.config = yaml.safe_load(f) or {}

    # ── Skills ────────────────────────────────────────────────────

    def scrape(self, url: str, selector: str = None) -> pd.DataFrame:
        """Scrape a web page and return a DataFrame with unified schema."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            print("  [Scrape] requests/beautifulsoup4 not installed")
            return self._empty_df()

        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        try:
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            if selector:
                elements = soup.select(selector)
                text = "\n".join(el.get_text(strip=True) for el in elements[:30])
            else:
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)[:5000]

            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 10]
            if not paragraphs:
                return self._empty_df()

            return pd.DataFrame({
                "text": paragraphs,
                "label": "unknown",
                "source": f"scrape:{url}",
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            print(f"  [Scrape] Error: {e}")
            return self._empty_df()

    def fetch_api(self, endpoint: str, params: dict = None) -> pd.DataFrame:
        """Fetch data from an API endpoint and return a unified DataFrame."""
        import requests
        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "results" in data:
            df = pd.DataFrame(data["results"])
        else:
            df = pd.DataFrame([data])

        return self._unify(df, source_name=f"api:{endpoint}")

    def load_dataset(self, name: str, source: str = "hf") -> pd.DataFrame:
        """Load a dataset from HuggingFace or Kaggle."""
        if source == "hf":
            try:
                from datasets import load_dataset as hf_load
                ds = hf_load(name, split="train")
                df = ds.to_pandas()
                return self._unify(df, source_name=f"hf:{name}")
            except Exception as e:
                print(f"  [HF] Error loading {name}: {e}")
                return self._empty_df()

        elif source == "kaggle":
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                dl_dir = os.path.join("data", "raw", name.replace("/", "_"))
                os.makedirs(dl_dir, exist_ok=True)
                api.dataset_download_files(name, path=dl_dir, unzip=True)
                for f in os.listdir(dl_dir):
                    fpath = os.path.join(dl_dir, f)
                    if f.endswith(".csv"):
                        return self._unify(pd.read_csv(fpath), source_name=f"kaggle:{name}")
                    elif f.endswith(".parquet"):
                        return self._unify(pd.read_parquet(fpath), source_name=f"kaggle:{name}")
            except Exception as e:
                print(f"  [Kaggle] Error loading {name}: {e}")
            return self._empty_df()
        else:
            raise ValueError(f"Unknown source: {source}")

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple DataFrames into a single unified dataset."""
        valid = [df for df in sources if len(df) > 0]
        if not valid:
            return self._empty_df()
        return pd.concat(valid, ignore_index=True)

    # ── Main entry point ──────────────────────────────────────────

    def run(self, sources: list[dict]) -> pd.DataFrame:
        """Run collection: dispatch to skills, merge, run EDA."""
        dfs = []
        for src in sources:
            t = src.get("type", "")
            print(f"  Collecting from {t}: {src.get('name', src.get('url', ''))}")
            if t == "hf_dataset":
                df = self.load_dataset(src["name"], source="hf")
            elif t == "kaggle_dataset":
                df = self.load_dataset(src["name"], source="kaggle")
            elif t == "scrape":
                df = self.scrape(src["url"], selector=src.get("selector"))
            elif t == "api":
                df = self.fetch_api(src["endpoint"], params=src.get("params"))
            else:
                print(f"    Unknown type: {t}, skipping")
                continue
            print(f"    → {len(df)} rows")
            dfs.append(df)

        merged = self.merge(dfs)

        if len(merged) == 0:
            print("  ⚠ WARNING: No data collected! Check your source selection.")
            return merged

        raw_dir = os.path.join(self.config.get("pipeline", {}).get("data_dir", "data"), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        merged.to_parquet(os.path.join(raw_dir, "combined.parquet"), index=False)
        print(f"  Saved {len(merged)} rows")

        eda_dir = os.path.join(self.config.get("pipeline", {}).get("data_dir", "data"), "eda")
        if len(merged) > 0:
            self.run_eda(merged, eda_dir)

        return merged

    # ── Search ────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """Search for datasets across HuggingFace, Kaggle, DuckDuckGo, Google Scholar."""
        all_results = []
        print(f"Searching for: '{query}'\n")

        print("  [1/4] HuggingFace Hub...")
        r = self._search_hf(query, max_results)
        print(f"         Found: {len(r)}")
        all_results.extend(r)

        print("  [2/4] Kaggle...")
        r = self._search_kaggle(query, max_results)
        print(f"         Found: {len(r)}")
        all_results.extend(r)

        print("  [3/4] DuckDuckGo...")
        r = self._search_web(f"{query} dataset download", max_results)
        print(f"         Found: {len(r)}")
        all_results.extend(r)

        print("  [4/4] Google Scholar...")
        r = self._search_scholar(f"{query} dataset", min(max_results, 5))
        print(f"         Found: {len(r)}")
        all_results.extend(r)

        return all_results

    @staticmethod
    def print_results_table(results: list[dict]):
        print(f"\n{'#':<4} {'Source':<16} {'Name':<45} {'Downloads':<10}")
        print("-" * 80)
        for i, r in enumerate(results, 1):
            print(f"{i:<4} {r.get('source',''):<16} {r.get('name','')[:44]:<45} {r.get('downloads',0):<10}")

    # ── EDA ───────────────────────────────────────────────────────

    def run_eda(self, df: pd.DataFrame, output_dir: str) -> dict:
        """Run EDA: class distribution, text lengths, top-20 words."""
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        label_counts = df["label"].value_counts()
        results["class_distribution"] = label_counts.to_dict()
        results["total_samples"] = len(df)
        results["num_classes"] = len(label_counts)

        fig, ax = plt.subplots(figsize=(10, 6))
        label_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title("Class Distribution"); ax.set_xlabel("Class"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45); plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150); plt.close(fig)
        print("  Saved class_distribution.png")

        text_lengths = df["text"].str.len()
        results["text_length"] = {"mean": float(text_lengths.mean()), "std": float(text_lengths.std()),
                                   "min": int(text_lengths.min()), "max": int(text_lengths.max())}
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(text_lengths, bins=30, color="steelblue", edgecolor="black")
        ax.set_title("Text Length Distribution"); ax.set_xlabel("Characters"); ax.set_ylabel("Count")
        plt.tight_layout(); fig.savefig(os.path.join(output_dir, "text_length.png"), dpi=150); plt.close(fig)
        print("  Saved text_length.png")

        all_words = " ".join(df["text"].str.lower()).split()
        word_counts = Counter(all_words)
        for sw in {"the","a","an","is","it","to","of","and","in","for","on","with","that","this","was","are","be","has","have","had","not","but","or","at","by","from","as","i"}:
            word_counts.pop(sw, None)
        top_20 = word_counts.most_common(20)
        results["top_words"] = dict(top_20)
        if top_20:
            words, counts = zip(*top_20)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(list(reversed(words)), list(reversed(counts)), color="steelblue", edgecolor="black")
            ax.set_title("Top 20 Words"); ax.set_xlabel("Count"); plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "top_words.png"), dpi=150); plt.close(fig)
            print("  Saved top_words.png")

        results["sources"] = df["source"].value_counts().to_dict()
        with open(os.path.join(output_dir, "eda_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("  Saved eda_results.json")
        return results

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _empty_df():
        return pd.DataFrame(columns=["text", "label", "source", "collected_at"])

    @staticmethod
    def _unify(df: pd.DataFrame, text_col=None, label_col=None, source_name="unknown") -> pd.DataFrame:
        unified = pd.DataFrame()
        if text_col and text_col in df.columns:
            unified["text"] = df[text_col].astype(str)
        else:
            for col in ["text","content","review","comment","body","sentence","description"]:
                if col in df.columns:
                    unified["text"] = df[col].astype(str); break
            else:
                str_cols = df.select_dtypes(include=["object","string"]).columns
                unified["text"] = df[str_cols[0]].astype(str) if len(str_cols) > 0 else df.iloc[:,0].astype(str)

        if label_col and label_col in df.columns:
            unified["label"] = df[label_col].astype(str)
        else:
            for col in ["label","class","category","sentiment","target","y","rating","rating_str","rating_int","score"]:
                if col in df.columns:
                    unified["label"] = df[col].astype(str); break
            else:
                unified["label"] = "unknown"

        unified["source"] = source_name
        unified["collected_at"] = datetime.now(timezone.utc).isoformat()
        return unified[unified["text"].str.strip().str.len() > 0]

    @staticmethod
    def _search_hf(query, max_results=10):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            return [{"name": ds.id, "source": "huggingface",
                     "url": f"https://huggingface.co/datasets/{ds.id}",
                     "description": ""[:200], "downloads": getattr(ds, "downloads", 0)}
                    for ds in list(api.list_datasets(search=query, limit=max_results, sort="downloads"))]
        except Exception as e:
            print(f"  [HF] Error: {e}"); return []

    @staticmethod
    def _search_kaggle(query, max_results=10):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            return [{"name": ds.ref, "source": "kaggle",
                     "url": f"https://www.kaggle.com/datasets/{ds.ref}",
                     "description": (getattr(ds,"subtitle","") or "")[:200],
                     "downloads": getattr(ds,"downloadCount",0)}
                    for ds in list(api.dataset_list(search=query))[:max_results]]
        except Exception as e:
            print(f"  [Kaggle] Error: {e}"); return []

    @staticmethod
    def _search_web(query, max_results=10):
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                return [{"name": r["title"][:80], "source": "web", "url": r["href"],
                         "description": r["body"][:200], "downloads": 0}
                        for r in ddgs.text(query, max_results=max_results)]
        except Exception as e:
            print(f"  [Web] Error: {e}"); return []

    @staticmethod
    def _search_scholar(query, max_results=5):
        try:
            import itertools
            from scholarly import scholarly
            return [{"name": r.get("bib",{}).get("title","")[:80], "source": "google_scholar",
                     "url": r.get("pub_url",""), "description": r.get("bib",{}).get("abstract","")[:200],
                     "downloads": r.get("num_citations",0)}
                    for r in itertools.islice(scholarly.search_pubs(query), max_results)]
        except Exception as e:
            print(f"  [Scholar] Error: {e}"); return []
