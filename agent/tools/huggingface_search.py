import os


def search_huggingface(query: str, max_results: int = 10) -> dict:
    """Search HuggingFace Hub for datasets matching the query."""
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

            results.append(
                {
                    "name": ds.id,
                    "url": f"https://huggingface.co/datasets/{ds.id}",
                    "description": description or getattr(ds, "description", "") or "",
                    "downloads": getattr(ds, "downloads", 0),
                    "tags": (ds.tags[:5] if ds.tags else []),
                }
            )
        return {"source": "huggingface", "count": len(results), "results": results}

    except Exception as e:
        return {"source": "huggingface", "error": str(e), "results": []}


def download_huggingface(name: str, download_dir: str = "./downloads") -> dict:
    """Download a HuggingFace dataset by repo ID."""
    try:
        from huggingface_hub import snapshot_download

        target_dir = os.path.join(download_dir, name.replace("/", "_"))
        os.makedirs(target_dir, exist_ok=True)

        path = snapshot_download(repo_id=name, repo_type="dataset", local_dir=target_dir)
        files = os.listdir(path)
        return {
            "success": True,
            "path": path,
            "files": files[:20],
            "message": f"Downloaded to {path}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
