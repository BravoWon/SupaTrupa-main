"""Doctor command for diagnostics and troubleshooting."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from jones_framework.cli.ui.output import (
    print_info,
    print_success,
    print_error,
    print_warning,
    print_check_result,
    confirm,
)
from jones_framework.cli.constants import DEFAULT_BACKEND_PORT, DEFAULT_FRONTEND_PORT

doctor_app = typer.Typer(name="doctor", help="Diagnostic commands")

console = Console()


def run_doctor(fix: bool = False, verbose: bool = False) -> None:
    """Run diagnostic checks and optionally fix issues."""
    from jones_framework.cli.services.dependency_checker import DependencyChecker
    from jones_framework.cli.services.platform_adapter import get_platform_adapter

    checker = DependencyChecker()
    adapter = get_platform_adapter()

    console.print("\n[bold]Running diagnostic checks...[/bold]\n")

    issues = []
    checks_run = 0

    # 1. Python version
    checks_run += 1
    py_check = checker.check_python_version()
    print_check_result("Python version", py_check.passed, py_check.details)
    if not py_check.passed:
        issues.append(("HIGH", "Python version", py_check.error_message, None))

    # 2. Virtual environment
    checks_run += 1
    venv_check = checker.check_virtual_env()
    print_check_result("Virtual environment", venv_check.passed, venv_check.details)
    if not venv_check.passed:
        issues.append(("MEDIUM", "Virtual environment", venv_check.error_message, "jones install"))

    # 3. Node.js
    checks_run += 1
    node_check = checker.check_node()
    print_check_result("Node.js", node_check.found, node_check.version or "NOT FOUND")
    if not node_check.found:
        issues.append(("MEDIUM", "Node.js", node_check.error, None))

    # 4. pnpm
    checks_run += 1
    pnpm_check = checker.check_pnpm()
    print_check_result("pnpm", pnpm_check.found, pnpm_check.version or "NOT FOUND")
    if not pnpm_check.found:
        issues.append(("LOW", "pnpm", pnpm_check.error, pnpm_check.fix_command))

    # 5. Port 8000 (backend)
    checks_run += 1
    port_8000 = checker.check_port_available(DEFAULT_BACKEND_PORT, "backend")
    print_check_result(f"Port {DEFAULT_BACKEND_PORT}", port_8000.passed, port_8000.details)
    if not port_8000.passed:
        pid = adapter.find_process_by_port(DEFAULT_BACKEND_PORT)
        fix_action = f"kill {pid}" if pid else None
        issues.append(("HIGH", f"Port {DEFAULT_BACKEND_PORT}", port_8000.error_message, fix_action))

    # 6. Port 5173 (frontend)
    checks_run += 1
    port_5173 = checker.check_port_available(DEFAULT_FRONTEND_PORT, "frontend")
    print_check_result(f"Port {DEFAULT_FRONTEND_PORT}", port_5173.passed, port_5173.details)
    if not port_5173.passed:
        pid = adapter.find_process_by_port(DEFAULT_FRONTEND_PORT)
        fix_action = f"kill {pid}" if pid else None
        issues.append(("HIGH", f"Port {DEFAULT_FRONTEND_PORT}", port_5173.error_message, fix_action))

    # 7. Disk space
    checks_run += 1
    disk_check = checker.check_disk_space()
    print_check_result("Disk space", disk_check.passed, disk_check.details)
    if not disk_check.passed:
        issues.append(("MEDIUM", "Disk space", disk_check.error_message, None))

    # 8. Git
    checks_run += 1
    git_check = checker.check_git()
    print_check_result("Git", git_check.found, git_check.version or "NOT FOUND")

    # Summary
    console.print()

    if not issues:
        print_success(f"All {checks_run} checks passed!")
        console.print("\n[dim]Everything looks good. If you're still having problems,")
        console.print("try running: jones stop && jones start[/dim]")
        return

    # Show issues
    console.print(f"[yellow]Problems Found: {len(issues)}[/yellow]\n")

    for severity, name, message, fix_action in issues:
        severity_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}.get(severity, "white")
        console.print(f"  [{severity_color}][{severity}][/{severity_color}] {name}")

        if verbose and message:
            console.print(f"  [dim]{message}[/dim]")

        if fix_action:
            console.print(f"  [dim]Fix: {fix_action}[/dim]")

        console.print()

    # Offer to fix
    if fix:
        run_fixes(issues, adapter)
    else:
        fixable = [i for i in issues if i[3] is not None]
        if fixable:
            console.print(f"[dim]{len(fixable)} issue(s) can be fixed automatically.[/dim]")
            console.print("[dim]Run: jones doctor --fix[/dim]")


def run_fixes(issues: list, adapter) -> None:
    """Attempt to fix issues automatically."""
    import subprocess

    fixable = [i for i in issues if i[3] is not None]

    if not fixable:
        print_info("No issues can be fixed automatically.")
        return

    console.print("\n[bold]Attempting fixes...[/bold]\n")

    for severity, name, message, fix_action in fixable:
        console.print(f"Fixing: {name}...")

        try:
            if fix_action.startswith("kill "):
                # Kill process
                pid = int(fix_action.split()[1])
                if adapter.kill_process(pid):
                    print_success(f"  Killed process {pid}")
                else:
                    print_warning(f"  Could not kill process {pid}")

            elif fix_action == "jones install":
                # Run installation
                print_info("  Run 'jones install' to set up the virtual environment")

            else:
                # Run command
                result = subprocess.run(
                    fix_action,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    print_success(f"  Fixed: {fix_action}")
                else:
                    print_warning(f"  Command failed: {fix_action}")

        except Exception as e:
            print_error(f"  Error: {e}")

    console.print("\n[dim]Run 'jones doctor' again to verify fixes.[/dim]")
