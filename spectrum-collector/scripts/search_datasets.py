"""Search for Raman spectroscopy datasets on HuggingFace Hub."""

import argparse
import json
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def search_hf(query: str, limit: int = 10) -> list[dict]:
    """Search HuggingFace Hub for datasets matching query."""
    api = HfApi()
    results = []

    datasets = list(api.list_datasets(search=query, limit=limit, sort="downloads"))

    for ds in datasets:
        results.append({
            "name": ds.id,
            "downloads": ds.downloads or 0,
            "tags": ds.tags or [],
            "description": (ds.card_data.get("description", "") if ds.card_data else "")[:200],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Search Raman spectroscopy datasets")
    parser.add_argument("--query", default="raman spectroscopy", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    print(f"Searching HuggingFace for: '{args.query}' ...")
    results = search_hf(args.query, args.limit)

    if not results:
        # Try broader search
        print("No results. Trying broader search: 'spectroscopy'")
        results = search_hf("spectroscopy", args.limit)

    if not results:
        print("No results. Trying: 'raman'")
        results = search_hf("raman", args.limit)

    print(f"\nFound {len(results)} datasets:\n")
    print(f"{'#':<4} {'Name':<45} {'Downloads':<12}")
    print("-" * 65)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['name'][:44]:<45} {r['downloads']:<12}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()
