"""Search for Raman spectroscopy datasets across multiple sources.

Sources:
  1. HuggingFace Hub — open datasets
  2. Kaggle — competition & community datasets
  3. DuckDuckGo — web search for niche repositories
  4. Google Scholar — academic papers with datasets
  5. Web scraping — extract data from found pages
"""

import argparse
import json
import sys
import os

# Add project root to path so we can import agent tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def search_huggingface(query: str, max_results: int = 10) -> list[dict]:
    """Search HuggingFace Hub for datasets."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        datasets = list(api.list_datasets(search=query, limit=max_results, sort="downloads"))

        results = []
        for ds in datasets:
            description = ""
            if hasattr(ds, "card_data") and ds.card_data:
                description = (
                    ds.card_data.get("description", "") if isinstance(ds.card_data, dict) else ""
                )

            results.append({
                "name": ds.id,
                "source": "huggingface",
                "url": f"https://huggingface.co/datasets/{ds.id}",
                "description": (description or getattr(ds, "description", "") or "")[:200],
                "downloads": getattr(ds, "downloads", 0),
            })
        return results
    except Exception as e:
        print(f"  [HuggingFace] Error: {e}")
        return []


def search_kaggle(query: str, max_results: int = 10) -> list[dict]:
    """Search Kaggle for datasets."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search=query)

        results = []
        for ds in datasets[:max_results]:
            results.append({
                "name": ds.ref,
                "source": "kaggle",
                "url": f"https://www.kaggle.com/datasets/{ds.ref}",
                "description": (getattr(ds, "subtitle", "") or "")[:200],
                "downloads": getattr(ds, "downloadCount", 0),
                "size": str(getattr(ds, "size", "unknown")),
            })
        return results
    except Exception as e:
        print(f"  [Kaggle] Error: {e}")
        return []


def search_web(query: str, max_results: int = 10) -> list[dict]:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        results = []
        for r in raw_results:
            results.append({
                "name": r["title"][:80],
                "source": "web",
                "url": r["href"],
                "description": r["body"][:200],
                "downloads": 0,
            })
        return results
    except Exception as e:
        print(f"  [DuckDuckGo] Error: {e}")
        return []


def search_google_scholar(query: str, max_results: int = 5) -> list[dict]:
    """Search Google Scholar for academic papers with datasets."""
    try:
        import itertools
        from scholarly import scholarly

        results_gen = scholarly.search_pubs(query)
        raw_results = list(itertools.islice(results_gen, max_results))

        results = []
        for r in raw_results:
            bib = r.get("bib", {})
            results.append({
                "name": bib.get("title", "Unknown")[:80],
                "source": "google_scholar",
                "url": r.get("pub_url") or r.get("eprint_url", ""),
                "description": bib.get("abstract", "")[:200],
                "downloads": r.get("num_citations", 0),
                "authors": ", ".join(bib.get("author", [])[:3]),
                "year": bib.get("pub_year", ""),
            })
        return results
    except Exception as e:
        print(f"  [Google Scholar] Error: {e}")
        return []


def scrape_url(url: str, css_selector: str = None) -> dict:
    """Fetch a web page and extract text content and links."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        if css_selector:
            elements = soup.select(css_selector)
            text = "\n".join(el.get_text(strip=True) for el in elements[:30])
        else:
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        text = text[:5000]

        links = []
        for a in soup.find_all("a", href=True):
            link_text = a.get_text(strip=True)
            if link_text:
                href = a["href"]
                if href.startswith("/"):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                links.append({"text": link_text[:100], "href": href})
                if len(links) >= 30:
                    break

        return {"url": url, "text": text, "links": links}

    except Exception as e:
        print(f"  [Scrape] Error scraping {url}: {e}")
        return {"url": url, "error": str(e), "text": "", "links": []}


def search_all(query: str, max_results: int = 10) -> list[dict]:
    """Search all available sources."""
    all_results = []

    print(f"Searching for: '{query}'\n")

    # 1. HuggingFace
    print("  [1/4] Searching HuggingFace Hub...")
    hf_results = search_huggingface(query, max_results)
    print(f"         Found: {len(hf_results)} datasets")
    all_results.extend(hf_results)

    # 2. Kaggle
    print("  [2/4] Searching Kaggle...")
    kaggle_results = search_kaggle(query, max_results)
    print(f"         Found: {len(kaggle_results)} datasets")
    all_results.extend(kaggle_results)

    # 3. DuckDuckGo web search
    print("  [3/4] Searching DuckDuckGo...")
    web_results = search_web(f"{query} dataset download", max_results)
    print(f"         Found: {len(web_results)} results")
    all_results.extend(web_results)

    # 4. Google Scholar
    print("  [4/4] Searching Google Scholar (may be slow)...")
    scholar_results = search_google_scholar(f"{query} dataset", min(max_results, 5))
    print(f"         Found: {len(scholar_results)} papers")
    all_results.extend(scholar_results)

    return all_results


def print_results_table(results: list[dict]):
    """Print results as a formatted table."""
    print(f"\n{'#':<4} {'Source':<16} {'Name':<45} {'Downloads':<10}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        name = r.get("name", "")[:44]
        downloads = r.get("downloads", 0)
        print(f"{i:<4} {source:<16} {name:<45} {downloads:<10}")


def main():
    parser = argparse.ArgumentParser(description="Search Raman spectroscopy datasets (all sources)")
    parser.add_argument("--query", default="raman spectroscopy", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Max results per source")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--sources", default="all",
                        help="Comma-separated sources: hf,kaggle,web,scholar,all")
    args = parser.parse_args()

    sources = args.sources.split(",")
    all_results = []

    print(f"Searching for: '{args.query}'\n")

    if "all" in sources or "hf" in sources:
        print("  [HuggingFace] Searching...")
        results = search_huggingface(args.query, args.limit)
        print(f"  [HuggingFace] Found: {len(results)}")
        all_results.extend(results)

    if "all" in sources or "kaggle" in sources:
        print("  [Kaggle] Searching...")
        results = search_kaggle(args.query, args.limit)
        print(f"  [Kaggle] Found: {len(results)}")
        all_results.extend(results)

    if "all" in sources or "web" in sources:
        print("  [DuckDuckGo] Searching...")
        results = search_web(f"{args.query} dataset download", args.limit)
        print(f"  [DuckDuckGo] Found: {len(results)}")
        all_results.extend(results)

    if "all" in sources or "scholar" in sources:
        print("  [Google Scholar] Searching (may be slow)...")
        results = search_google_scholar(f"{args.query} dataset", min(args.limit, 5))
        print(f"  [Google Scholar] Found: {len(results)}")
        all_results.extend(results)

    print(f"\nTotal: {len(all_results)} results from {len(set(r['source'] for r in all_results))} sources")
    print_results_table(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    return all_results


if __name__ == "__main__":
    main()
