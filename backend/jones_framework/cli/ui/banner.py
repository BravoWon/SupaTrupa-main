"""ASCII art banners and branding."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

JONES_BANNER = r"""
       ██╗ ██████╗ ███╗   ██╗███████╗███████╗
       ██║██╔═══██╗████╗  ██║██╔════╝██╔════╝
       ██║██║   ██║██╔██╗ ██║█████╗  ███████╗
  ██   ██║██║   ██║██║╚██╗██║██╔══╝  ╚════██║
  ╚█████╔╝╚██████╔╝██║ ╚████║███████╗███████║
   ╚════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚══════╝
"""

SUBTITLE = "Unified Activity:State Platform - Beta"


def show_banner() -> None:
    """Display the main Jones Framework banner."""
    banner_text = Text()
    banner_text.append(JONES_BANNER, style="bold blue")
    banner_text.append(f"\n{SUBTITLE:^50}", style="dim")

    panel = Panel(
        banner_text,
        border_style="blue",
        padding=(0, 2),
    )
    console.print(panel)
    console.print()


def show_success_banner(urls: dict) -> None:
    """Display success banner with service URLs."""
    content = Text()
    content.append("Framework is Running!\n\n", style="bold green")

    if urls.get("backend"):
        content.append("  Backend API:       ", style="dim")
        content.append(f"{urls['backend']}\n", style="cyan underline")

    if urls.get("frontend"):
        content.append("  Web Interface:     ", style="dim")
        content.append(f"{urls['frontend']}\n", style="cyan underline")

    if urls.get("docs"):
        content.append("  API Documentation: ", style="dim")
        content.append(f"{urls['docs']}\n", style="cyan underline")

    content.append("\n")
    content.append("  To stop: ", style="dim")
    content.append("jones stop", style="yellow")
    content.append("    To see status: ", style="dim")
    content.append("jones status --watch", style="yellow")

    panel = Panel(
        content,
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


def show_error_banner(title: str, message: str, actions: list = None) -> None:
    """Display error banner with helpful actions."""
    content = Text()
    content.append(f"{title}\n\n", style="bold red")
    content.append(message, style="white")

    if actions:
        content.append("\n\nTo fix:\n", style="bold")
        for i, action in enumerate(actions, 1):
            content.append(f"  {i}. {action}\n", style="dim")

    panel = Panel(
        content,
        border_style="red",
        title="[bold red]Problem[/bold red]",
        padding=(1, 2),
    )
    console.print(panel)
