import itertools


def search_google(query: str, max_results: int = 10) -> dict:
    """Search Google Scholar for academic papers and datasets related to the query."""
    try:
        from scholarly import scholarly

        results_gen = scholarly.search_pubs(query)
        results = list(itertools.islice(results_gen, max_results))

        parsed = []
        for r in results:
            bib = r.get("bib", {})
            parsed.append(
                {
                    "title": bib.get("title", "Unknown"),
                    "url": r.get("pub_url") or r.get("eprint_url", ""),
                    "snippet": bib.get("abstract", "")[:300],
                    "authors": ", ".join(bib.get("author", [])[:3]),
                    "year": bib.get("pub_year", ""),
                    "venue": bib.get("venue", ""),
                    "citations": r.get("num_citations", 0),
                }
            )

        return {
            "source": "google_scholar",
            "count": len(parsed),
            "results": parsed,
        }

    except Exception as e:
        return {"source": "google_scholar", "error": str(e), "results": []}
