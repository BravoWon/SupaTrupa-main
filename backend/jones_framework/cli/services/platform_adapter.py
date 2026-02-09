"""Cross-platform adaptations for Windows, macOS, and Linux."""

import os
import sys
import shutil
import signal
import subprocess
import socket
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific operations."""

    @abstractmethod
    def get_venv_activate_command(self, venv_path: Path) -> str:
        """Get the command to activate a virtual environment."""
        pass

    @abstractmethod
    def get_venv_python(self, venv_path: Path) -> Path:
        """Get the path to the Python executable in a venv."""
        pass

    @abstractmethod
    def run_background_process(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
    ) -> subprocess.Popen:
        """Start a process in the background."""
        pass

    @abstractmethod
    def kill_process(self, pid: int, force: bool = False) -> bool:
        """Kill a process by PID."""
        pass

    @abstractmethod
    def find_process_by_port(self, port: int) -> Optional[int]:
        """Find the PID of a process using a specific port."""
        pass

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except socket.error:
                return True

    def check_command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH."""
        return shutil.which(command) is not None

    def get_command_version(self, command: str, version_flag: str = "--version") -> Optional[str]:
        """Get the version of a command."""
        try:
            result = subprocess.run(
                [command, version_flag],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Extract version from output (first line, usually)
                output = result.stdout.strip() or result.stderr.strip()
                return output.split("\n")[0]
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

    def get_disk_space(self, path: Path) -> Tuple[int, int]:
        """Get total and free disk space in bytes."""
        if hasattr(os, 'statvfs'):
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            return total, free
        else:
            # Windows fallback
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                str(path), None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes)
            )
            return total_bytes.value, free_bytes.value

    def get_project_root(self) -> Optional[Path]:
        """Find the project root by looking for marker files."""
        markers = [".git", "pyproject.toml", "backend"]
        current = Path.cwd()

        for _ in range(10):  # Max depth
            for marker in markers:
                if (current / marker).exists():
                    return current
            if current.parent == current:
                break
            current = current.parent

        return None


class UnixAdapter(PlatformAdapter):
    """Platform adapter for Unix-like systems (Linux, macOS)."""

    def get_venv_activate_command(self, venv_path: Path) -> str:
        return f"source {venv_path}/bin/activate"

    def get_venv_python(self, venv_path: Path) -> Path:
        return venv_path / "bin" / "python"

    def run_background_process(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
    ) -> subprocess.Popen:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        return subprocess.Popen(
            command,
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    def kill_process(self, pid: int, force: bool = False) -> bool:
        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def find_process_by_port(self, port: int) -> Optional[int]:
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{port}", "-t"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split("\n")[0])
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            return None


class WindowsAdapter(PlatformAdapter):
    """Platform adapter for Windows systems."""

    def get_venv_activate_command(self, venv_path: Path) -> str:
        return str(venv_path / "Scripts" / "activate.bat")

    def get_venv_python(self, venv_path: Path) -> Path:
        return venv_path / "Scripts" / "python.exe"

    def run_background_process(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
    ) -> subprocess.Popen:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        # Windows-specific flags for background process
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008

        return subprocess.Popen(
            command,
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
        )

    def kill_process(self, pid: int, force: bool = False) -> bool:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F" if force else ""],
                capture_output=True,
                timeout=5,
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def find_process_by_port(self, port: int) -> Optional[int]:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.split()
                        if parts:
                            return int(parts[-1])
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            return None


class MacOSAdapter(UnixAdapter):
    """Platform adapter for macOS with Mac-specific tweaks."""

    def get_command_version(self, command: str, version_flag: str = "--version") -> Optional[str]:
        # Handle macOS-specific version commands
        if command == "python3":
            # Try python3 first, then python
            version = super().get_command_version("python3", version_flag)
            if version:
                return version
            return super().get_command_version("python", version_flag)
        return super().get_command_version(command, version_flag)


def get_platform_adapter() -> PlatformAdapter:
    """Get the appropriate platform adapter for the current system."""
    if sys.platform == "win32":
        return WindowsAdapter()
    elif sys.platform == "darwin":
        return MacOSAdapter()
    else:
        return UnixAdapter()
