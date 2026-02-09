"""
Main CLI entry point for the Jones Framework.

Provides both command-line interface and interactive menu system.
"""

import sys
from typing import Optional

import typer
from rich.console import Console

from jones_framework.cli.constants import MAIN_MENU_OPTIONS

# Create the main Typer app
app = typer.Typer(
    name="jones",
    help="Jones Framework - Unified Activity:State Platform",
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
)

console = Console()


def show_banner() -> None:
    """Display the Jones Framework banner."""
    from jones_framework.cli.ui.banner import show_banner as _show_banner
    _show_banner()


def interactive_menu() -> None:
    """Run the interactive menu system."""
    from jones_framework.cli.ui.menu import interactive_menu as _interactive_menu
    _interactive_menu()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive menu"),
) -> None:
    """
    Jones Framework CLI - Unified Activity:State Platform

    Run without arguments or with --interactive for the menu system.
    """
    if version:
        console.print("[bold blue]Jones Framework[/bold blue] version [green]0.1.0-beta[/green]")
        raise typer.Exit()

    # If no command provided, show interactive menu
    if ctx.invoked_subcommand is None:
        show_banner()
        interactive_menu()


@app.command()
def quick() -> None:
    """
    Quick Start - Get everything running with defaults.

    This command:
    1. Checks system requirements
    2. Installs dependencies if needed
    3. Starts all services
    4. Opens the web interface
    """
    from jones_framework.cli.ui.output import print_info, print_success, print_error
    from jones_framework.cli.services.dependency_checker import DependencyChecker
    from jones_framework.cli.services.service_manager import ServiceManager

    show_banner()
    print_info("Starting Quick Setup...")

    # Check dependencies
    checker = DependencyChecker()
    if not checker.check_all():
        print_error("Some requirements are missing. Run 'jones doctor' for details.")
        raise typer.Exit(1)

    # Start services
    manager = ServiceManager()
    if manager.start_all():
        print_success("Framework is ready!")
        manager.show_urls()
    else:
        print_error("Failed to start services. Run 'jones doctor' for diagnostics.")
        raise typer.Exit(1)


@app.command()
def wizard() -> None:
    """
    Interactive guided setup wizard.

    Walk through installation step by step with helpful explanations.
    """
    from jones_framework.cli.wizard.engine import WizardEngine

    show_banner()
    engine = WizardEngine()
    engine.run()


@app.command()
def install(
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m",
        help="Installation mode: simple, standard, or full"
    ),
    check_only: bool = typer.Option(
        False, "--check", "-c",
        help="Only check requirements, don't install"
    ),
) -> None:
    """
    Install the framework on your computer.

    Modes:
    - simple: Basic setup (~100 MB)
    - standard: Backend + Web interface (~300 MB)
    - full: Everything including AI/ML (~3 GB)
    """
    from jones_framework.cli.commands.install import run_install
    run_install(mode=mode, check_only=check_only)


@app.command()
def start(
    backend_only: bool = typer.Option(False, "--backend", "-b", help="Start backend only"),
    frontend_only: bool = typer.Option(False, "--frontend", "-f", help="Start frontend only"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait until services are healthy"),
) -> None:
    """
    Start the framework services.

    By default, starts both backend and frontend.
    """
    from jones_framework.cli.commands.start import run_start
    run_start(backend_only=backend_only, frontend_only=frontend_only, wait=wait)


@app.command()
def stop(
    force: bool = typer.Option(False, "--force", "-f", help="Force stop services"),
) -> None:
    """
    Stop all running services.
    """
    from jones_framework.cli.commands.start import run_stop
    run_stop(force=force)


@app.command()
def restart() -> None:
    """
    Restart all services.
    """
    from jones_framework.cli.commands.start import run_restart
    run_restart()


@app.command()
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuously monitor status"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """
    Check what's running and system health.
    """
    from jones_framework.cli.commands.status import run_status
    run_status(watch=watch, json_output=json_output)


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", "-f", help="Attempt automatic fixes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """
    Diagnose problems and suggest fixes.

    Run this if something isn't working correctly.
    """
    from jones_framework.cli.commands.doctor import run_doctor
    run_doctor(fix=fix, verbose=verbose)


@app.command()
def logs(
    service: Optional[str] = typer.Option(None, "--service", "-s", help="Service to show logs for"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
) -> None:
    """
    View service logs.
    """
    from jones_framework.cli.commands.logs import run_logs
    run_logs(service=service, follow=follow, lines=lines)


def main() -> None:
    """Main entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("[dim]Run 'jones doctor' for diagnostics[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
