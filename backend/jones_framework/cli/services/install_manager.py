"""Installation management for the framework."""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Callable

from jones_framework.cli.constants import InstallMode, INSTALL_MODES
from jones_framework.cli.services.platform_adapter import get_platform_adapter


class InstallManager:
    """Manage framework installation."""

    def __init__(self, progress_callback: Optional[Callable[[str, int], None]] = None):
        self.adapter = get_platform_adapter()
        self.project_root = self.adapter.get_project_root()
        self.progress_callback = progress_callback

    def _report_progress(self, message: str, percent: int) -> None:
        """Report installation progress."""
        if self.progress_callback:
            self.progress_callback(message, percent)

    def create_virtual_env(self, path: Optional[Path] = None) -> bool:
        """Create a Python virtual environment."""
        if path is None:
            if self.project_root:
                path = self.project_root / "backend" / ".venv"
            else:
                return False

        if path.exists():
            return True

        self._report_progress("Creating virtual environment...", 10)

        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(path)],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def install_python_deps(self, mode: InstallMode = InstallMode.STANDARD) -> bool:
        """Install Python dependencies."""
        if not self.project_root:
            return False

        backend_dir = self.project_root / "backend"
        venv_path = backend_dir / ".venv"
        python_path = self.adapter.get_venv_python(venv_path)

        if not python_path.exists():
            if not self.create_virtual_env(venv_path):
                return False

        self._report_progress("Installing Python dependencies...", 30)

        # Determine extras based on mode
        extras = "cli"
        if mode == InstallMode.STANDARD:
            extras = "cli,api"
        elif mode == InstallMode.FULL:
            extras = "all"

        try:
            subprocess.run(
                [str(python_path), "-m", "pip", "install", "-e", f".[{extras}]"],
                cwd=backend_dir,
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def install_node_deps(self) -> bool:
        """Install Node.js dependencies for frontend."""
        if not self.project_root:
            return False

        frontend_dir = self.project_root / "frontend"
        if not frontend_dir.exists():
            return True  # No frontend to install

        self._report_progress("Installing frontend dependencies...", 60)

        try:
            subprocess.run(
                ["pnpm", "install"],
                cwd=frontend_dir,
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_all(self, mode: InstallMode = InstallMode.STANDARD) -> bool:
        """Run complete installation."""
        from jones_framework.cli.ui.output import print_info, print_success, print_error

        mode_info = INSTALL_MODES[mode]
        print_info(f"Installing {mode_info.name}...")

        # Create venv
        if not self.create_virtual_env():
            print_error("Failed to create virtual environment")
            return False

        # Install Python deps
        if not self.install_python_deps(mode):
            print_error("Failed to install Python dependencies")
            return False

        # Install Node deps (for standard and full modes)
        if mode in (InstallMode.STANDARD, InstallMode.FULL):
            if not self.install_node_deps():
                print_error("Failed to install frontend dependencies")
                return False

        self._report_progress("Installation complete!", 100)
        print_success("Installation complete!")
        return True
