"""Status command implementation."""

import json
import time
import typer
from rich.console import Console
from rich.live import Live

from jones_framework.cli.ui.output import show_status_table

status_app = typer.Typer(name="status", help="Status and monitoring commands")

console = Console()


def run_status(watch: bool = False, json_output: bool = False) -> None:
    """Show status of all services."""
    from jones_framework.cli.services.service_manager import ServiceManager

    manager = ServiceManager()

    if json_output:
        services = manager.get_all_status()
        output = {
            "services": [
                {
                    "name": s.name,
                    "running": s.running,
                    "pid": s.pid,
                    "port": s.port,
                    "healthy": s.healthy,
                    "url": s.url,
                }
                for s in services
            ]
        }
        console.print(json.dumps(output, indent=2))
        return

    if watch:
        run_watch_mode(manager)
    else:
        console.print("\n[bold]Service Status[/bold]\n")
        services = manager.get_all_status()
        show_status_table([
            {
                "name": s.name.capitalize(),
                "running": s.running,
                "pid": s.pid,
                "port": s.port,
                "healthy": s.healthy,
            }
            for s in services
        ])
        console.print()

        # Show URLs for running services
        running = [s for s in services if s.running]
        if running:
            console.print("[dim]Access URLs:[/dim]")
            for s in running:
                if s.url:
                    console.print(f"  {s.name}: [cyan underline]{s.url}[/cyan underline]")
            console.print()


def run_watch_mode(manager) -> None:
    """Run continuous status monitoring."""
    console.print("[dim]Watching status... Press Ctrl+C to stop[/dim]\n")

    try:
        while True:
            # Clear and redraw
            console.clear()
            console.print("[bold]Service Status[/bold] [dim](watching, Ctrl+C to stop)[/dim]\n")

            services = manager.get_all_status()
            show_status_table([
                {
                    "name": s.name.capitalize(),
                    "running": s.running,
                    "pid": s.pid,
                    "port": s.port,
                    "healthy": s.healthy,
                }
                for s in services
            ])

            # Show health details for backend
            from jones_framework.cli.services.health_monitor import HealthMonitor
            monitor = HealthMonitor()
            backend_status = monitor.get_backend_status()

            if backend_status:
                console.print("\n[dim]Backend Details:[/dim]")
                for key, value in backend_status.items():
                    if key != "status":
                        console.print(f"  {key}: {value}")

            time.sleep(2)

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching[/dim]")
