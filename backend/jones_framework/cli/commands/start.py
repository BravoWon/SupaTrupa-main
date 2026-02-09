"""Service start/stop command implementation."""

import typer

from jones_framework.cli.ui.output import print_info, print_success, print_error

start_app = typer.Typer(name="start", help="Service management commands")


def run_start(
    backend_only: bool = False,
    frontend_only: bool = False,
    wait: bool = False
) -> None:
    """Start framework services."""
    from jones_framework.cli.services.service_manager import ServiceManager
    from jones_framework.cli.services.dependency_checker import DependencyChecker

    # Quick dependency check
    checker = DependencyChecker()
    if not checker.check_all():
        print_error("Some requirements are missing. Run 'jones install' first.")
        raise typer.Exit(1)

    manager = ServiceManager()

    if backend_only:
        if manager.start_backend(wait=wait):
            manager.show_urls()
        else:
            raise typer.Exit(1)
    elif frontend_only:
        if manager.start_frontend(wait=wait):
            manager.show_urls()
        else:
            raise typer.Exit(1)
    else:
        if manager.start_all(wait=wait):
            manager.show_urls()
        else:
            raise typer.Exit(1)


def run_stop(force: bool = False) -> None:
    """Stop framework services."""
    from jones_framework.cli.services.service_manager import ServiceManager

    manager = ServiceManager()
    services = manager.get_all_status()

    running = [s for s in services if s.running]
    if not running:
        print_info("No services are running")
        return

    if manager.stop_all(force=force):
        print_success("All services stopped")
    else:
        print_error("Some services could not be stopped")
        if not force:
            print_info("Try: jones stop --force")
        raise typer.Exit(1)


def run_restart() -> None:
    """Restart all services."""
    from jones_framework.cli.services.service_manager import ServiceManager

    print_info("Restarting services...")
    manager = ServiceManager()

    if manager.restart_all():
        manager.show_urls()
    else:
        print_error("Failed to restart services")
        raise typer.Exit(1)
