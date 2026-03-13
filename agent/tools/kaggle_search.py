import os


def search_kaggle(query: str, max_results: int = 10) -> dict:
    """Search Kaggle for datasets matching the query."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search=query)

        results = []
        for ds in datasets[:max_results]:
            results.append(
                {
                    "name": ds.ref,
                    "title": getattr(ds, "title", ds.ref),
                    "url": f"https://www.kaggle.com/datasets/{ds.ref}",
                    "size": str(getattr(ds, "size", "unknown")),
                    "description": getattr(ds, "subtitle", "") or "",
                    "downloads": getattr(ds, "downloadCount", 0),
                }
            )
        return {"source": "kaggle", "count": len(results), "results": results}

    except Exception as e:
        return {"source": "kaggle", "error": str(e), "results": []}


def download_kaggle(name: str, download_dir: str = "./downloads") -> dict:
    """Download a Kaggle dataset by reference (e.g., 'owner/dataset-name')."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        target_dir = os.path.join(download_dir, name.replace("/", "_"))
        os.makedirs(target_dir, exist_ok=True)
        api.dataset_download_files(name, path=target_dir, unzip=True)

        files = os.listdir(target_dir)
        return {
            "success": True,
            "path": target_dir,
            "files": files[:20],
            "message": f"Downloaded {len(files)} files to {target_dir}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
