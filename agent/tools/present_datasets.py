from rich.prompt import Prompt
from agent.ui import console, show_datasets_table


def present_datasets(datasets: list) -> dict:
    """Present datasets to user and get their selection."""
    if not datasets:
        console.print("[yellow]No datasets found.[/yellow]")
        return {"selected": [], "message": "No datasets were found"}

    show_datasets_table(datasets)

    console.print()
    choice = Prompt.ask(
        "[bold]Select datasets to download (e.g. 1,3,5) or 'none' to skip[/bold]",
        default="none",
    )

    if choice.strip().lower() in ("none", "n", "skip", ""):
        return {"selected": [], "message": "User chose not to download any datasets"}

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = [datasets[i] for i in indices if 0 <= i < len(datasets)]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection. Skipping download.[/red]")
        return {"selected": [], "message": "Invalid selection from user"}

    names = [ds.get("name", "?") for ds in selected]
    console.print(f"[green]Selected {len(selected)} dataset(s): {', '.join(names)}[/green]")
    return {
        "selected": selected,
        "message": f"User selected {len(selected)} datasets for download: {', '.join(names)}",
    }
