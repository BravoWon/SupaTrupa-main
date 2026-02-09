"""Process management for framework services."""

import os
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

from jones_framework.cli.constants import (
    DEFAULT_BACKEND_PORT,
    DEFAULT_FRONTEND_PORT,
    SERVICE_BACKEND,
    SERVICE_FRONTEND,
)
from jones_framework.cli.services.platform_adapter import get_platform_adapter
from jones_framework.cli.services.health_monitor import HealthMonitor


@dataclass
class ServiceInfo:
    """Information about a running service."""
    name: str
    running: bool
    pid: Optional[int] = None
    port: Optional[int] = None
    healthy: bool = False
    url: Optional[str] = None


class ServiceManager:
    """Manage framework service processes."""

    def __init__(self):
        self.adapter = get_platform_adapter()
        self.project_root = self.adapter.get_project_root()
        self.health_monitor = HealthMonitor()
        self._pid_file = self._get_pid_file_path()
        self._pids: Dict[str, int] = {}
        self._load_pids()

    def _get_pid_file_path(self) -> Path:
        """Get path to PID file for tracking services."""
        if self.project_root:
            pid_dir = self.project_root / ".jones"
            pid_dir.mkdir(exist_ok=True)
            return pid_dir / "services.json"
        return Path.home() / ".jones" / "services.json"

    def _load_pids(self) -> None:
        """Load saved PIDs from file."""
        if self._pid_file.exists():
            try:
                with open(self._pid_file) as f:
                    self._pids = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._pids = {}

    def _save_pids(self) -> None:
        """Save PIDs to file."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._pid_file, "w") as f:
            json.dump(self._pids, f)

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def get_service_status(self, service: str) -> ServiceInfo:
        """Get status of a specific service."""
        port = DEFAULT_BACKEND_PORT if service == SERVICE_BACKEND else DEFAULT_FRONTEND_PORT
        pid = self._pids.get(service)

        # Check if process is running
        if pid and self._is_process_running(pid):
            url = f"http://localhost:{port}"
            healthy = self.health_monitor.check_health(service)
            return ServiceInfo(
                name=service,
                running=True,
                pid=pid,
                port=port,
                healthy=healthy,
                url=url,
            )

        # Check if something else is using the port
        port_pid = self.adapter.find_process_by_port(port)
        if port_pid:
            return ServiceInfo(
                name=service,
                running=True,
                pid=port_pid,
                port=port,
                healthy=self.health_monitor.check_health(service),
                url=f"http://localhost:{port}",
            )

        return ServiceInfo(name=service, running=False, port=port)

    def get_all_status(self) -> List[ServiceInfo]:
        """Get status of all services."""
        return [
            self.get_service_status(SERVICE_BACKEND),
            self.get_service_status(SERVICE_FRONTEND),
        ]

    def start_backend(self, wait: bool = True) -> bool:
        """Start the backend API server."""
        from jones_framework.cli.ui.output import print_info, print_success, print_error

        if not self.project_root:
            print_error("Could not find project root")
            return False

        backend_dir = self.project_root / "backend"
        if not backend_dir.exists():
            print_error(f"Backend directory not found: {backend_dir}")
            return False

        # Check if already running
        status = self.get_service_status(SERVICE_BACKEND)
        if status.running:
            print_info("Backend is already running")
            return True

        # Find Python in venv
        venv_path = backend_dir / ".venv"
        if not venv_path.exists():
            venv_path = self.project_root / ".venv"

        python_path = self.adapter.get_venv_python(venv_path)
        if not python_path.exists():
            print_error("Virtual environment not found. Run 'jones install' first.")
            return False

        # Start uvicorn
        print_info("Starting backend server...")
        cmd = [
            str(python_path),
            "-m", "uvicorn",
            "jones_framework.api.server:app",
            "--host", "0.0.0.0",
            "--port", str(DEFAULT_BACKEND_PORT),
        ]

        try:
            process = self.adapter.run_background_process(cmd, cwd=backend_dir)
            self._pids[SERVICE_BACKEND] = process.pid
            self._save_pids()

            if wait:
                if self.health_monitor.wait_for_health(SERVICE_BACKEND, timeout=30):
                    print_success(f"Backend started on port {DEFAULT_BACKEND_PORT}")
                    return True
                else:
                    print_error("Backend started but health check failed")
                    return False

            print_success(f"Backend starting on port {DEFAULT_BACKEND_PORT}")
            return True

        except Exception as e:
            print_error(f"Failed to start backend: {e}")
            return False

    def start_frontend(self, wait: bool = True) -> bool:
        """Start the frontend dev server."""
        from jones_framework.cli.ui.output import print_info, print_success, print_error

        if not self.project_root:
            print_error("Could not find project root")
            return False

        frontend_dir = self.project_root / "frontend"
        if not frontend_dir.exists():
            print_error(f"Frontend directory not found: {frontend_dir}")
            return False

        # Check if already running
        status = self.get_service_status(SERVICE_FRONTEND)
        if status.running:
            print_info("Frontend is already running")
            return True

        # Start pnpm dev
        print_info("Starting frontend server...")
        cmd = ["pnpm", "dev"]

        try:
            process = self.adapter.run_background_process(cmd, cwd=frontend_dir)
            self._pids[SERVICE_FRONTEND] = process.pid
            self._save_pids()

            if wait:
                # Frontend takes longer to start
                if self.health_monitor.wait_for_health(SERVICE_FRONTEND, timeout=60):
                    print_success(f"Frontend started on port {DEFAULT_FRONTEND_PORT}")
                    return True
                else:
                    # May still be starting, give it the benefit of the doubt
                    print_success(f"Frontend starting on port {DEFAULT_FRONTEND_PORT}")
                    return True

            print_success(f"Frontend starting on port {DEFAULT_FRONTEND_PORT}")
            return True

        except Exception as e:
            print_error(f"Failed to start frontend: {e}")
            return False

    def start_all(self, wait: bool = True) -> bool:
        """Start all services in correct order."""
        from jones_framework.cli.ui.output import print_step

        print_step(1, 2, "Starting backend API...")
        if not self.start_backend(wait=wait):
            return False

        print_step(2, 2, "Starting frontend...")
        if not self.start_frontend(wait=wait):
            return False

        return True

    def stop_service(self, service: str, force: bool = False) -> bool:
        """Stop a specific service."""
        from jones_framework.cli.ui.output import print_info, print_success, print_warning

        status = self.get_service_status(service)
        if not status.running:
            print_info(f"{service.capitalize()} is not running")
            return True

        pid = status.pid
        if pid:
            print_info(f"Stopping {service} (PID {pid})...")
            if self.adapter.kill_process(pid, force=force):
                # Wait for process to stop
                for _ in range(10):
                    if not self._is_process_running(pid):
                        break
                    time.sleep(0.5)

                if service in self._pids:
                    del self._pids[service]
                    self._save_pids()

                print_success(f"{service.capitalize()} stopped")
                return True
            else:
                print_warning(f"Could not stop {service}")
                return False

        return False

    def stop_all(self, force: bool = False) -> bool:
        """Stop all running services."""
        success = True

        # Stop frontend first, then backend
        for service in [SERVICE_FRONTEND, SERVICE_BACKEND]:
            if not self.stop_service(service, force=force):
                success = False

        return success

    def restart_all(self) -> bool:
        """Restart all services."""
        self.stop_all()
        time.sleep(1)
        return self.start_all()

    def show_urls(self) -> None:
        """Display URLs for running services."""
        from jones_framework.cli.ui.banner import show_success_banner

        backend = self.get_service_status(SERVICE_BACKEND)
        frontend = self.get_service_status(SERVICE_FRONTEND)

        urls = {}
        if backend.running:
            urls["backend"] = f"http://localhost:{DEFAULT_BACKEND_PORT}"
            urls["docs"] = f"http://localhost:{DEFAULT_BACKEND_PORT}/docs"
        if frontend.running:
            urls["frontend"] = f"http://localhost:{DEFAULT_FRONTEND_PORT}"

        if urls:
            show_success_banner(urls)
