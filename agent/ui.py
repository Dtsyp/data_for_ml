from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def show_tool_call(name: str, inputs: dict):
    """Display which tool the agent is calling."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
    console.print(
        Panel(f"[bold cyan]{name}[/]({args_str})", title="Tool Call", border_style="cyan")
    )


def show_thinking(text: str):
    """Display agent's reasoning text."""
    console.print(f"[dim]{text}[/dim]")


def show_error(msg: str):
    """Display an error message."""
    console.print(Panel(msg, title="Error", border_style="red"))


def show_success(msg: str):
    """Display a success message."""
    console.print(Panel(msg, title="Success", border_style="green"))


def show_datasets_table(datasets: list) -> None:
    """Render a table of found datasets."""
    table = Table(title="Found Datasets", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Name", max_width=30)
    table.add_column("Source", width=12)
    table.add_column("URL", max_width=50)
    table.add_column("Description", max_width=40)
    table.add_column("Format", width=10)

    for i, ds in enumerate(datasets, 1):
        table.add_row(
            str(i),
            ds.get("name", "N/A"),
            ds.get("source", "N/A"),
            ds.get("url", "N/A"),
            (ds.get("description", "") or "")[:80],
            ds.get("format", "N/A"),
        )
    console.print(table)
