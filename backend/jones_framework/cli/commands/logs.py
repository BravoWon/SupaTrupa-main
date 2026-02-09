"""Log viewing command implementation."""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console

from jones_framework.cli.ui.output import print_info, print_error, print_warning
from jones_framework.cli.services.platform_adapter import get_platform_adapter

logs_app = typer.Typer(name="logs", help="Log viewing commands")

console = Console()


def run_logs(
    service: Optional[str] = None,
    follow: bool = False,
    lines: int = 50
) -> None:
    """View service logs."""
    adapter = get_platform_adapter()
    project_root = adapter.get_project_root()

    if not project_root:
        print_error("Could not find project root")
        raise typer.Exit(1)

    # Determine log paths
    log_dir = project_root / ".jones" / "logs"

    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        print_info("No logs found yet. Start services to generate logs.")
        return

    # Find log files
    if service:
        log_files = list(log_dir.glob(f"{service}*.log"))
    else:
        log_files = list(log_dir.glob("*.log"))

    if not log_files:
        # Try to find logs in standard locations
        possible_logs = [
            project_root / "backend" / "logs",
            project_root / "logs",
        ]

        for log_path in possible_logs:
            if log_path.exists():
                if service:
                    log_files = list(log_path.glob(f"{service}*.log"))
                else:
                    log_files = list(log_path.glob("*.log"))
                if log_files:
                    break

    if not log_files:
        print_info("No log files found.")
        console.print("\n[dim]Logs are created when services are running.[/dim]")
        console.print("[dim]Start services with: jones start[/dim]")
        return

    # Sort by modification time (newest first)
    log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    if follow:
        follow_logs(log_files[0])
    else:
        show_logs(log_files, lines)


def show_logs(log_files: list, lines: int) -> None:
    """Display log file contents."""
    for log_file in log_files:
        console.print(f"\n[bold cyan]═══ {log_file.name} ═══[/bold cyan]\n")

        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()

                # Show last N lines
                display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

                if len(all_lines) > lines:
                    console.print(f"[dim]... showing last {lines} of {len(all_lines)} lines ...[/dim]\n")

                for line in display_lines:
                    format_log_line(line.rstrip())

        except Exception as e:
            print_error(f"Could not read {log_file.name}: {e}")


def follow_logs(log_file: Path) -> None:
    """Follow a log file (like tail -f)."""
    import time

    console.print(f"[dim]Following {log_file.name}... Press Ctrl+C to stop[/dim]\n")

    try:
        with open(log_file, "r") as f:
            # Go to end of file
            f.seek(0, 2)

            while True:
                line = f.readline()
                if line:
                    format_log_line(line.rstrip())
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped following logs[/dim]")
    except Exception as e:
        print_error(f"Error following logs: {e}")


def format_log_line(line: str) -> None:
    """Format and colorize a log line."""
    if not line:
        return

    # Color based on log level
    if "ERROR" in line or "error" in line.lower():
        console.print(f"[red]{line}[/red]")
    elif "WARNING" in line or "WARN" in line or "warning" in line.lower():
        console.print(f"[yellow]{line}[/yellow]")
    elif "DEBUG" in line:
        console.print(f"[dim]{line}[/dim]")
    elif "INFO" in line:
        console.print(line)
    else:
        console.print(line)
