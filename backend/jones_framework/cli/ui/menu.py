"""Interactive menu system for user-friendly navigation."""

from typing import Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from jones_framework.cli.constants import MAIN_MENU_OPTIONS

console = Console()


def show_menu_options() -> None:
    """Display the main menu options."""
    console.print()
    for key, name, description in MAIN_MENU_OPTIONS:
        console.print(f"  [bold cyan][{key}][/bold cyan] {name:12} [dim]- {description}[/dim]")
    console.print()


def get_menu_choice() -> str:
    """Get user's menu choice with validation."""
    valid_choices = [opt[0] for opt in MAIN_MENU_OPTIONS]

    while True:
        try:
            choice = console.input("Enter your choice (1-8): ").strip()

            if choice in valid_choices:
                return choice

            console.print("[yellow]Please enter a number from 1 to 8[/yellow]")
        except KeyboardInterrupt:
            return "8"  # Exit


def handle_menu_choice(choice: str) -> bool:
    """
    Handle a menu choice and execute the corresponding action.

    Returns True to continue the menu loop, False to exit.
    """
    import typer

    if choice == "1":
        # Quick Start
        from jones_framework.cli.main import quick
        try:
            quick()
        except typer.Exit:
            pass
        return True

    elif choice == "2":
        # Install
        from jones_framework.cli.main import install
        try:
            install(mode=None, check_only=False)
        except typer.Exit:
            pass
        return True

    elif choice == "3":
        # Start
        from jones_framework.cli.main import start
        try:
            start(backend_only=False, frontend_only=False, wait=False)
        except typer.Exit:
            pass
        return True

    elif choice == "4":
        # Status
        from jones_framework.cli.main import status
        try:
            status(watch=False, json_output=False)
        except typer.Exit:
            pass
        return True

    elif choice == "5":
        # Stop
        from jones_framework.cli.main import stop
        try:
            stop(force=False)
        except typer.Exit:
            pass
        return True

    elif choice == "6":
        # Doctor
        from jones_framework.cli.main import doctor
        try:
            doctor(fix=False, verbose=False)
        except typer.Exit:
            pass
        return True

    elif choice == "7":
        # Help
        show_help()
        return True

    elif choice == "8":
        # Exit
        console.print("\n[dim]Goodbye![/dim]")
        return False

    return True


def show_help() -> None:
    """Display help information."""
    help_text = Text()
    help_text.append("Jones Framework Help\n\n", style="bold blue")

    help_text.append("Getting Started:\n", style="bold")
    help_text.append("  1. Run ", style="dim")
    help_text.append("Quick Start", style="cyan")
    help_text.append(" to set up and launch everything automatically\n", style="dim")
    help_text.append("  2. Or use ", style="dim")
    help_text.append("Install", style="cyan")
    help_text.append(" for step-by-step setup with more options\n\n", style="dim")

    help_text.append("If Something Goes Wrong:\n", style="bold")
    help_text.append("  - Run ", style="dim")
    help_text.append("Doctor", style="yellow")
    help_text.append(" to diagnose and fix common problems\n", style="dim")
    help_text.append("  - Check ", style="dim")
    help_text.append("Status", style="yellow")
    help_text.append(" to see what's running\n\n", style="dim")

    help_text.append("Command Line Usage:\n", style="bold")
    help_text.append("  jones quick       ", style="cyan")
    help_text.append("- Quick setup and start\n", style="dim")
    help_text.append("  jones wizard      ", style="cyan")
    help_text.append("- Interactive guided setup\n", style="dim")
    help_text.append("  jones start       ", style="cyan")
    help_text.append("- Start services\n", style="dim")
    help_text.append("  jones stop        ", style="cyan")
    help_text.append("- Stop services\n", style="dim")
    help_text.append("  jones status      ", style="cyan")
    help_text.append("- Check status\n", style="dim")
    help_text.append("  jones doctor      ", style="cyan")
    help_text.append("- Diagnose problems\n", style="dim")
    help_text.append("  jones --help      ", style="cyan")
    help_text.append("- Show all options\n", style="dim")

    panel = Panel(help_text, border_style="blue", padding=(1, 2))
    console.print(panel)


def interactive_menu() -> None:
    """Run the main interactive menu loop."""
    show_menu_options()

    while True:
        choice = get_menu_choice()

        if not handle_menu_choice(choice):
            break

        # Show menu again after action completes
        if choice != "8":
            console.print("\n" + "-" * 50)
            show_menu_options()
