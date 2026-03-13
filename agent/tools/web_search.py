def search_web(query: str, max_results: int = 10) -> dict:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        return {
            "source": "web",
            "count": len(results),
            "results": [
                {"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results
            ],
        }
    except Exception as e:
        return {"source": "web", "error": str(e), "results": []}
