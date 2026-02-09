"""User-friendly output formatters."""

from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green][bold]OK[/bold][/green] {message}")


def print_error(message: str, hint: Optional[str] = None) -> None:
    """Print an error message with optional hint."""
    console.print(f"[red][bold]ERROR[/bold][/red] {message}")
    if hint:
        console.print(f"[dim]Hint: {hint}[/dim]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow][bold]WARN[/bold][/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue][bold]INFO[/bold][/blue] {message}")


def print_step(step: int, total: int, message: str) -> None:
    """Print a step progress message."""
    console.print(f"[dim]({step}/{total})[/dim] {message}")


def print_check_result(name: str, passed: bool, details: str = "") -> None:
    """Print a check result with pass/fail indicator."""
    icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
    status = f"[green]{details}[/green]" if passed else f"[red]{details}[/red]"

    # Pad name for alignment
    padded_name = f"{name:.<30}"
    console.print(f"  {icon} {padded_name} {status}")


def create_progress() -> Progress:
    """Create a styled progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        console=console,
    )


def show_status_table(services: list) -> None:
    """Display service status in a table."""
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Service", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Port", justify="center")
    table.add_column("PID", justify="right")
    table.add_column("Health", justify="center")

    for service in services:
        status_style = "green" if service["running"] else "red"
        status_text = "Running" if service["running"] else "Stopped"
        health_icon = "[green]●[/green]" if service.get("healthy") else "[red]●[/red]" if service["running"] else "[dim]○[/dim]"

        table.add_row(
            service["name"],
            f"[{status_style}]{status_text}[/{status_style}]",
            str(service.get("port", "-")),
            str(service.get("pid", "-")),
            health_icon,
        )

    console.print(table)


def show_dependency_table(deps: list) -> None:
    """Display dependency check results."""
    for dep in deps:
        status = "Found" if dep["found"] else "NOT FOUND"
        print_check_result(
            dep["name"],
            dep["found"],
            f"{dep.get('version', status)}" if dep["found"] else status
        )


def confirm(message: str, default: bool = True) -> bool:
    """Ask for user confirmation."""
    default_hint = "[Y/n]" if default else "[y/N]"
    response = console.input(f"\n{message} {default_hint}: ").strip().lower()

    if not response:
        return default
    return response in ("y", "yes")


def show_urls_panel(urls: dict) -> None:
    """Show panel with accessible URLs."""
    from jones_framework.cli.ui.banner import show_success_banner
    show_success_banner(urls)
