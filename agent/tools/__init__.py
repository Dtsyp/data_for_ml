from agent.tools.kaggle_search import search_kaggle, download_kaggle
from agent.tools.huggingface_search import search_huggingface, download_huggingface
from agent.tools.web_search import search_web
from agent.tools.google_search import search_google
from agent.tools.web_scrape import scrape_url
from agent.tools.present_datasets import present_datasets

# Mistral uses OpenAI-compatible function calling format
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_kaggle",
            "description": (
                "Search Kaggle for datasets matching a query. "
                "Returns dataset names, URLs, descriptions, and sizes. "
                "Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for Kaggle datasets"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_huggingface",
            "description": (
                "Search HuggingFace Hub for datasets matching a query. "
                "Returns dataset IDs, URLs, descriptions, and download counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for HuggingFace datasets"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web using DuckDuckGo for datasets, data repositories, "
                "and download pages. Good for finding university datasets, "
                "government data portals, and niche repositories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": (
                "Search Google Scholar for academic papers and datasets. "
                "Returns papers with titles, authors, abstracts, citations. "
                "Great for finding research datasets referenced in papers. "
                "Note: may be slow (5-10 seconds)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Google Scholar search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": (
                "Fetch a web page and extract its text content and links. "
                "Use this to explore dataset pages found via web search, "
                "find download links, or read dataset descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "css_selector": {
                        "type": "string",
                        "description": "Optional CSS selector to extract specific elements",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "present_datasets",
            "description": (
                "Present found datasets to the user in a formatted table "
                "and ask them to select which ones to download. "
                "Call this AFTER searching all sources. "
                "Returns the user's selection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "datasets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "source": {"type": "string", "description": "kaggle, huggingface, or web"},
                                "url": {"type": "string"},
                                "description": {"type": "string"},
                                "format": {"type": "string", "description": "File format (CSV, HDF5, etc.)"},
                                "size": {"type": "string", "description": "Approximate size"},
                            },
                            "required": ["name", "source", "url"],
                        },
                        "description": "List of datasets to present",
                    }
                },
                "required": ["datasets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_dataset",
            "description": (
                "Download a dataset. Supports Kaggle datasets (by ref), "
                "HuggingFace datasets (by repo ID), and direct URL downloads."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name or identifier"},
                    "source": {
                        "type": "string",
                        "enum": ["kaggle", "huggingface", "url"],
                        "description": "Source type",
                    },
                    "url": {"type": "string", "description": "URL (for direct downloads)"},
                },
                "required": ["name", "source"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "search_kaggle": search_kaggle,
    "search_huggingface": search_huggingface,
    "search_web": search_web,
    "search_google": search_google,
    "scrape_url": scrape_url,
    "present_datasets": present_datasets,
    "download_dataset": None,  # set dynamically in loop.py
}
