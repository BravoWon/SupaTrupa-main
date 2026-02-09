"""Installation command implementation."""

import typer
from typing import Optional

from jones_framework.cli.constants import InstallMode, INSTALL_MODES
from jones_framework.cli.ui.output import (
    console,
    print_info,
    print_success,
    print_error,
    print_warning,
    print_check_result,
    confirm,
)

install_app = typer.Typer(name="install", help="Installation commands")


def run_install(mode: Optional[str] = None, check_only: bool = False) -> None:
    """Run the installation process."""
    from jones_framework.cli.services.dependency_checker import DependencyChecker
    from jones_framework.cli.services.install_manager import InstallManager

    checker = DependencyChecker()

    # Parse mode
    install_mode = InstallMode.STANDARD
    if mode:
        mode_lower = mode.lower()
        if mode_lower == "simple":
            install_mode = InstallMode.SIMPLE
        elif mode_lower == "full":
            install_mode = InstallMode.FULL
        elif mode_lower != "standard":
            print_error(f"Unknown mode: {mode}. Use simple, standard, or full.")
            raise typer.Exit(1)

    # Show mode info
    mode_info = INSTALL_MODES[install_mode]
    console.print(f"\n[bold]{mode_info.name}[/bold]")
    console.print(f"[dim]{mode_info.description}[/dim]")
    console.print(f"[dim]Size: {mode_info.size} | Time: {mode_info.time}[/dim]\n")

    # Run checks
    console.print("[bold]Checking your computer...[/bold]\n")
    report = checker.get_full_report(install_mode)

    for check in report:
        print_check_result(check.name, check.passed, check.details)

    # Check for failures
    failed = [r for r in report if not r.passed]

    if failed:
        console.print()
        print_warning(f"{len(failed)} issue(s) found")

        for check in failed:
            if check.error_message:
                console.print(f"\n[red]{check.name}:[/red]")
                console.print(f"[dim]{check.error_message}[/dim]")

        if check_only:
            raise typer.Exit(1)

        # Some issues are blocking
        critical = [r for r in failed if r.name in ["Python version", "Disk space"]]
        if critical:
            print_error("Cannot continue due to critical issues above.")
            raise typer.Exit(1)

    if check_only:
        console.print()
        print_success("All checks passed!")
        return

    # Confirm installation
    console.print()
    if not confirm("Ready to install. Continue?"):
        print_info("Installation cancelled.")
        raise typer.Exit(0)

    # Run installation
    console.print()
    manager = InstallManager()

    if manager.install_all(install_mode):
        console.print()
        print_success("Installation complete!")
        console.print()
        console.print("[dim]To start the framework, run:[/dim]")
        console.print("  [cyan]jones start[/cyan]")
    else:
        print_error("Installation failed. Run 'jones doctor' for diagnostics.")
        raise typer.Exit(1)


def show_mode_selection() -> InstallMode:
    """Show interactive mode selection."""
    console.print("\n[bold]How would you like to use the framework?[/bold]\n")

    for mode, info in INSTALL_MODES.items():
        marker = "●" if mode == InstallMode.STANDARD else "○"
        recommended = " [green](Recommended)[/green]" if mode == InstallMode.STANDARD else ""
        console.print(f"  {marker} [bold]{info.name}[/bold]{recommended}")
        console.print(f"    [dim]{info.description}[/dim]")
        console.print(f"    [dim]Size: {info.size} | Time: {info.time}[/dim]")
        console.print()

    while True:
        choice = console.input("Enter your choice (1=Simple, 2=Standard, 3=Full) [2]: ").strip()

        if not choice or choice == "2":
            return InstallMode.STANDARD
        elif choice == "1":
            return InstallMode.SIMPLE
        elif choice == "3":
            return InstallMode.FULL
        else:
            console.print("[yellow]Please enter 1, 2, or 3[/yellow]")
