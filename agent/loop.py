import json
import os

from mistralai.client.sdk import Mistral

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS_SCHEMA, TOOL_HANDLERS
from agent.tools.kaggle_search import download_kaggle
from agent.tools.huggingface_search import download_huggingface
from agent.ui import console, show_tool_call, show_thinking, show_error, show_success

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

MODEL = "mistral-medium-latest"


def download_url(url: str, download_dir: str, name: str) -> dict:
    """Download a file from a direct URL."""
    try:
        target_dir = os.path.join(download_dir, name.replace("/", "_").replace(" ", "_"))
        os.makedirs(target_dir, exist_ok=True)

        resp = requests.get(url, timeout=60, headers=HEADERS, stream=True)
        resp.raise_for_status()

        filename = url.split("/")[-1].split("?")[0] or "data"
        filepath = os.path.join(target_dir, filename)

        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size = os.path.getsize(filepath)
        return {
            "success": True,
            "path": filepath,
            "size": f"{size / 1024:.1f} KB",
            "message": f"Downloaded {filename} ({size / 1024:.1f} KB) to {target_dir}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def handle_download(name: str, source: str, url: str = None, download_dir: str = "./downloads") -> dict:
    """Route download to the appropriate handler."""
    if source == "kaggle":
        return download_kaggle(name, download_dir)
    elif source == "huggingface":
        return download_huggingface(name, download_dir)
    elif source == "url" and url:
        return download_url(url, download_dir, name)
    else:
        return {"success": False, "error": f"Unknown source or missing URL: {source}"}


def run_agent(query: str, download_dir: str = "./downloads"):
    """Run the dataset search agent loop."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        show_error(
            "MISTRAL_API_KEY not set.\n"
            "Get a free key at: https://console.mistral.ai/\n"
            "Then: export MISTRAL_API_KEY=your_key_here"
        )
        return

    client = Mistral(api_key=api_key)
    os.makedirs(download_dir, exist_ok=True)

    console.print(f"\n[bold magenta]Dataset Search Agent[/bold magenta] (model: {MODEL})")
    console.print(f"[dim]Query: {query}[/dim]")
    console.print(f"[dim]Download directory: {download_dir}[/dim]\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    max_iterations = 20  # safety limit

    for _ in range(max_iterations):
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
        except Exception as e:
            show_error(f"API error: {e}")
            break

        choice = response.choices[0]
        message = choice.message

        # Add assistant message to history
        messages.append(message)

        # Print any text content
        if message.content and message.content.strip():
            show_thinking(message.content)

        # If no tool calls, the agent is done
        if not message.tool_calls:
            break

        # Execute each tool call
        for tool_call in message.tool_calls:
            func = tool_call.function
            tool_name = func.name

            try:
                tool_input = json.loads(func.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            show_tool_call(tool_name, tool_input)

            # Execute the tool
            if tool_name == "download_dataset":
                result = handle_download(
                    name=tool_input.get("name", ""),
                    source=tool_input.get("source", "url"),
                    url=tool_input.get("url"),
                    download_dir=download_dir,
                )
                if result.get("success"):
                    show_success(result.get("message", "Downloaded"))
                else:
                    show_error(result.get("error", "Download failed"))
            elif tool_name in TOOL_HANDLERS and TOOL_HANDLERS[tool_name]:
                result = TOOL_HANDLERS[tool_name](**tool_input)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Add tool result to messages
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

    console.print("\n[bold magenta]Agent finished.[/bold magenta]\n")
