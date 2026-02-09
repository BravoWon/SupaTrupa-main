"""Health monitoring for framework services."""

import time
from typing import Optional
import urllib.request
import urllib.error

from jones_framework.cli.constants import (
    DEFAULT_BACKEND_PORT,
    DEFAULT_FRONTEND_PORT,
    SERVICE_BACKEND,
    SERVICE_FRONTEND,
)


class HealthMonitor:
    """Monitor health of framework services."""

    def __init__(self):
        self.endpoints = {
            SERVICE_BACKEND: f"http://localhost:{DEFAULT_BACKEND_PORT}/health",
            SERVICE_FRONTEND: f"http://localhost:{DEFAULT_FRONTEND_PORT}/",
        }
        self.timeout = 5

    def check_health(self, service: str) -> bool:
        """Check if a service is healthy."""
        endpoint = self.endpoints.get(service)
        if not endpoint:
            return False

        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False

    def check_all(self) -> dict:
        """Check health of all services."""
        return {
            service: self.check_health(service)
            for service in self.endpoints
        }

    def wait_for_health(
        self,
        service: str,
        timeout: int = 30,
        poll_interval: float = 1.0
    ) -> bool:
        """Wait for a service to become healthy."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.check_health(service):
                return True
            time.sleep(poll_interval)

        return False

    def get_backend_status(self) -> Optional[dict]:
        """Get detailed backend status from /health endpoint."""
        endpoint = f"http://localhost:{DEFAULT_BACKEND_PORT}/health"

        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    import json
                    return json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass

        return None

    def get_detailed_status(self) -> dict:
        """Get detailed status for all services."""
        backend_health = self.get_backend_status()

        return {
            SERVICE_BACKEND: {
                "healthy": backend_health is not None,
                "details": backend_health,
            },
            SERVICE_FRONTEND: {
                "healthy": self.check_health(SERVICE_FRONTEND),
                "details": None,
            },
        }
