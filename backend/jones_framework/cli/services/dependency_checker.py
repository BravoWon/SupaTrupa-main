"""Pre-flight dependency and environment checks."""

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from jones_framework.cli.constants import (
    MIN_PYTHON_VERSION,
    MIN_DISK_SPACE,
    InstallMode,
    ERROR_MESSAGES,
)
from jones_framework.cli.services.platform_adapter import get_platform_adapter


@dataclass
class DependencyResult:
    """Result of a dependency check."""
    name: str
    found: bool
    version: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None
    fix_command: Optional[str] = None


@dataclass
class CheckResult:
    """Result of a system check."""
    name: str
    passed: bool
    details: str
    error_message: Optional[str] = None
    fix_actions: Optional[List[str]] = None


class DependencyChecker:
    """Check system dependencies and requirements."""

    def __init__(self):
        self.adapter = get_platform_adapter()
        self.project_root = self.adapter.get_project_root()

    def check_python_version(self) -> CheckResult:
        """Check if Python version meets requirements."""
        current = sys.version_info[:2]
        required = MIN_PYTHON_VERSION

        if current >= required:
            return CheckResult(
                name="Python version",
                passed=True,
                details=f"Python {current[0]}.{current[1]}",
            )
        else:
            return CheckResult(
                name="Python version",
                passed=False,
                details=f"Python {current[0]}.{current[1]}",
                error_message=ERROR_MESSAGES["python_version"].format(
                    found=f"{current[0]}.{current[1]}",
                    required=f"{required[0]}.{required[1]}"
                ),
            )

    def check_node(self) -> DependencyResult:
        """Check if Node.js is installed."""
        version = self.adapter.get_command_version("node", "--version")
        if version:
            # Extract version number (e.g., "v20.10.0" -> "20.10.0")
            clean_version = version.lstrip("v").split()[0]
            return DependencyResult(
                name="Node.js",
                found=True,
                version=clean_version,
            )
        return DependencyResult(
            name="Node.js",
            found=False,
            error=ERROR_MESSAGES["node_missing"],
        )

    def check_pnpm(self) -> DependencyResult:
        """Check if pnpm is installed."""
        version = self.adapter.get_command_version("pnpm", "--version")
        if version:
            return DependencyResult(
                name="pnpm",
                found=True,
                version=version.split()[0],
            )
        return DependencyResult(
            name="pnpm",
            found=False,
            error=ERROR_MESSAGES["pnpm_missing"],
            fix_command="npm install -g pnpm",
        )

    def check_git(self) -> DependencyResult:
        """Check if Git is installed."""
        version = self.adapter.get_command_version("git", "--version")
        if version:
            # "git version 2.39.0" -> "2.39.0"
            match = re.search(r"(\d+\.\d+\.\d+)", version)
            clean_version = match.group(1) if match else version
            return DependencyResult(
                name="Git",
                found=True,
                version=clean_version,
            )
        return DependencyResult(
            name="Git",
            found=False,
        )

    def check_virtual_env(self) -> CheckResult:
        """Check if virtual environment exists and is usable."""
        if not self.project_root:
            return CheckResult(
                name="Virtual environment",
                passed=False,
                details="Project root not found",
            )

        venv_path = self.project_root / "backend" / ".venv"
        if not venv_path.exists():
            venv_path = self.project_root / ".venv"

        if venv_path.exists():
            python_path = self.adapter.get_venv_python(venv_path)
            if python_path.exists():
                return CheckResult(
                    name="Virtual environment",
                    passed=True,
                    details=f".venv exists at {venv_path.name}",
                )
            return CheckResult(
                name="Virtual environment",
                passed=False,
                details="venv folder exists but Python not found",
            )

        return CheckResult(
            name="Virtual environment",
            passed=False,
            details="Not set up",
            error_message=ERROR_MESSAGES["venv_missing"],
            fix_actions=["Run: jones install"],
        )

    def check_disk_space(self, mode: InstallMode = InstallMode.STANDARD) -> CheckResult:
        """Check available disk space."""
        if not self.project_root:
            path = Path.cwd()
        else:
            path = self.project_root

        try:
            total, free = self.adapter.get_disk_space(path)
            required = MIN_DISK_SPACE[mode]

            # Format sizes for display
            free_gb = free / (1024**3)
            required_gb = required / (1024**3)

            if free >= required:
                return CheckResult(
                    name="Disk space",
                    passed=True,
                    details=f"{free_gb:.1f} GB available",
                )
            else:
                return CheckResult(
                    name="Disk space",
                    passed=False,
                    details=f"{free_gb:.1f} GB available",
                    error_message=ERROR_MESSAGES["disk_space"].format(
                        required=f"{required_gb:.1f} GB",
                        available=f"{free_gb:.1f} GB"
                    ),
                )
        except Exception as e:
            return CheckResult(
                name="Disk space",
                passed=True,  # Assume OK if check fails
                details="Could not check",
            )

    def check_port_available(self, port: int, service_name: str = "") -> CheckResult:
        """Check if a port is available."""
        if not self.adapter.is_port_in_use(port):
            return CheckResult(
                name=f"Port {port}" + (f" ({service_name})" if service_name else ""),
                passed=True,
                details="Available",
            )

        pid = self.adapter.find_process_by_port(port)
        pid_info = f" (PID {pid})" if pid else ""

        return CheckResult(
            name=f"Port {port}" + (f" ({service_name})" if service_name else ""),
            passed=False,
            details=f"IN USE{pid_info}",
            error_message=ERROR_MESSAGES["port_in_use"].format(port=port),
        )

    def check_all(self) -> bool:
        """
        Run all critical checks and return True if all pass.

        This is the main pre-flight check that should pass before starting.
        """
        checks = [
            self.check_python_version(),
            self.check_virtual_env(),
        ]

        return all(check.passed for check in checks)

    def get_full_report(self, mode: InstallMode = InstallMode.STANDARD) -> List[CheckResult]:
        """Get a complete dependency check report."""
        results = []

        # System checks
        results.append(self.check_python_version())
        results.append(self.check_disk_space(mode))

        # Dependencies
        node = self.check_node()
        results.append(CheckResult(
            name="Node.js",
            passed=node.found,
            details=node.version if node.found else "NOT FOUND",
            error_message=node.error if not node.found else None,
        ))

        pnpm = self.check_pnpm()
        results.append(CheckResult(
            name="pnpm",
            passed=pnpm.found,
            details=pnpm.version if pnpm.found else "NOT FOUND",
            error_message=pnpm.error if not pnpm.found else None,
        ))

        git = self.check_git()
        results.append(CheckResult(
            name="Git",
            passed=git.found,
            details=git.version if git.found else "NOT FOUND",
        ))

        # Environment
        results.append(self.check_virtual_env())

        return results

    def get_dependency_list(self) -> List[DependencyResult]:
        """Get list of all dependency check results."""
        return [
            self.check_node(),
            self.check_pnpm(),
            self.check_git(),
        ]
