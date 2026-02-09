"""Wizard screen definitions and display logic."""

from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from jones_framework.cli.constants import InstallMode, INSTALL_MODES
from jones_framework.cli.ui.banner import JONES_BANNER


class WizardScreens:
    """Screen display logic for the installation wizard."""

    def __init__(self, console: Console):
        self.console = console

    def show_welcome(self) -> None:
        """Display the welcome screen."""
        self.console.clear()

        banner = Text()
        banner.append(JONES_BANNER, style="bold blue")
        banner.append("\n\n")
        banner.append("Welcome to the Jones Framework Installation Wizard!\n\n", style="bold")
        banner.append("This wizard will help you:\n", style="dim")
        banner.append("  1. Check your computer meets the requirements\n", style="dim")
        banner.append("  2. Choose how much to install\n", style="dim")
        banner.append("  3. Set up everything automatically\n", style="dim")
        banner.append("\n")
        banner.append("No technical knowledge required - just follow the prompts.", style="italic dim")

        panel = Panel(banner, border_style="blue", padding=(1, 2))
        self.console.print(panel)

    def show_step(self, current: int, total: int, title: str) -> None:
        """Display a step header."""
        self.console.print()
        self.console.print(f"[bold]Step {current} of {total}:[/bold] {title}")
        self.console.print("─" * 50)
        self.console.print()

    def show_check_results(self, results: list) -> None:
        """Display system check results."""
        self.console.print("[bold]Checking your computer...[/bold]\n")

        for result in results:
            icon = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            status = result.details

            # Pad name for alignment
            name = f"{result.name:.<35}"

            if result.passed:
                self.console.print(f"  {icon} {name} [green]{status}[/green]")
            else:
                self.console.print(f"  {icon} {name} [red]{status}[/red]")

    def show_mode_selection(self) -> InstallMode:
        """Display mode selection and return choice."""
        self.console.print("[bold]How would you like to use the framework?[/bold]\n")

        modes = list(INSTALL_MODES.items())

        for i, (mode, info) in enumerate(modes, 1):
            recommended = " [green](Recommended)[/green]" if mode == InstallMode.STANDARD else ""
            self.console.print(f"  [{i}] [bold]{info.name}[/bold]{recommended}")
            self.console.print(f"      [dim]{info.description}[/dim]")
            self.console.print(f"      [dim]Size: {info.size} | Time: {info.time}[/dim]")
            self.console.print()

        while True:
            choice = self.console.input("Enter your choice (1-3) [2]: ").strip()

            if not choice:
                return InstallMode.STANDARD

            try:
                idx = int(choice)
                if 1 <= idx <= 3:
                    return modes[idx - 1][0]
            except ValueError:
                pass

            self.console.print("[yellow]Please enter 1, 2, or 3[/yellow]")

    def show_installation_progress(self, message: str, percent: int) -> None:
        """Display installation progress."""
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        self.console.print(f"\r  [{bar}] {percent:3d}%  {message}", end="")

    def show_completion(self) -> None:
        """Display completion screen."""
        self.console.print()

        content = Text()
        content.append("✓ ", style="bold green")
        content.append("Installation Complete!\n\n", style="bold green")
        content.append("The Jones Framework has been installed on your computer.\n\n")
        content.append("You can now:\n")
        content.append("  • Run ", style="dim")
        content.append("jones start", style="cyan")
        content.append(" to launch the framework\n", style="dim")
        content.append("  • Run ", style="dim")
        content.append("jones", style="cyan")
        content.append(" to see the main menu\n", style="dim")
        content.append("  • Run ", style="dim")
        content.append("jones help", style="cyan")
        content.append(" for more options\n", style="dim")

        panel = Panel(content, border_style="green", padding=(1, 2))
        self.console.print(panel)

    def show_error(self, title: str, message: str, actions: List[str] = None) -> None:
        """Display an error screen."""
        content = Text()
        content.append(f"✗ {title}\n\n", style="bold red")
        content.append(message, style="white")

        if actions:
            content.append("\n\nTo fix this:\n", style="bold")
            for i, action in enumerate(actions, 1):
                content.append(f"  {i}. {action}\n", style="dim")

        panel = Panel(content, border_style="red", title="[bold red]Error[/bold red]", padding=(1, 2))
        self.console.print(panel)

    def show_cancelled(self) -> None:
        """Display cancellation message."""
        self.console.print()
        self.console.print("[yellow]Installation cancelled.[/yellow]")
        self.console.print()
        self.console.print("[dim]You can run the wizard again anytime with:[/dim]")
        self.console.print("  [cyan]jones wizard[/cyan]")
        self.console.print()
