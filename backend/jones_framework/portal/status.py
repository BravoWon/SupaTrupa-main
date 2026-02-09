"""
Portal Status: Component status tracking and aggregation.

Provides:
- ComponentStatus: Enum of possible component states
- StatusReport: Data class for component status snapshots
- StatusAggregator: Central collector for all component statuses

Usage:
    from jones_framework.portal.status import StatusAggregator, ComponentStatus, StatusReport

    aggregator = StatusAggregator()

    # Update component status
    aggregator.update('novelty', StatusReport(
        component_name='novelty',
        status=ComponentStatus.RUNNING,
        metrics={'iterations': 42, 'novelty': 0.82}
    ))

    # Subscribe to updates
    aggregator.subscribe(lambda report: print(f"{report.component_name}: {report.status}"))

    # Get all statuses
    all_statuses = aggregator.get_all()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum, auto
from datetime import datetime
import threading

from jones_framework.portal.errors import PortalError


class ComponentStatus(Enum):
    """
    Possible states for a portal component.
    """
    IDLE = 'idle'           # Registered but not started
    STARTING = 'starting'   # In process of starting
    RUNNING = 'running'     # Actively processing
    PAUSED = 'paused'       # Temporarily stopped
    ERROR = 'error'         # Encountered an error
    STOPPED = 'stopped'     # Gracefully stopped

    @property
    def is_active(self) -> bool:
        """Check if component is in an active state."""
        return self in (ComponentStatus.STARTING, ComponentStatus.RUNNING)

    @property
    def is_healthy(self) -> bool:
        """Check if component is in a healthy state."""
        return self in (
            ComponentStatus.IDLE,
            ComponentStatus.STARTING,
            ComponentStatus.RUNNING,
            ComponentStatus.PAUSED,
            ComponentStatus.STOPPED
        )

    @property
    def indicator(self) -> str:
        """Get a status indicator character."""
        indicators = {
            ComponentStatus.IDLE: 'o',
            ComponentStatus.STARTING: '>',
            ComponentStatus.RUNNING: '*',
            ComponentStatus.PAUSED: '|',
            ComponentStatus.ERROR: 'X',
            ComponentStatus.STOPPED: '.',
        }
        return indicators.get(self, '?')


@dataclass
class StatusReport:
    """
    A snapshot of a component's status.

    Contains:
    - Component identity
    - Current status
    - Metrics (component-specific key-value pairs)
    - Timing information
    - Error information (if any)
    """
    component_name: str
    status: ComponentStatus
    metrics: Dict[str, float] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    error: Optional[PortalError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uptime_seconds(self) -> Optional[float]:
        """Get uptime in seconds if started."""
        if self.started_at is None:
            return None
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Get seconds since last activity."""
        return (datetime.now() - self.last_activity).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component_name': self.component_name,
            'status': self.status.value,
            'status_indicator': self.status.indicator,
            'is_active': self.status.is_active,
            'is_healthy': self.status.is_healthy,
            'metrics': self.metrics,
            'last_activity': self.last_activity.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'uptime_seconds': self.uptime_seconds,
            'idle_seconds': self.idle_seconds,
            'error': self.error.to_dict() if self.error else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatusReport':
        """Create from dictionary."""
        return cls(
            component_name=data['component_name'],
            status=ComponentStatus(data['status']),
            metrics=data.get('metrics', {}),
            last_activity=datetime.fromisoformat(data['last_activity']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            error=PortalError.from_dict(data['error']) if data.get('error') else None,
            metadata=data.get('metadata', {})
        )

    def with_status(self, status: ComponentStatus) -> 'StatusReport':
        """Create a copy with updated status."""
        return StatusReport(
            component_name=self.component_name,
            status=status,
            metrics=self.metrics.copy(),
            last_activity=datetime.now(),
            started_at=self.started_at,
            error=None if status != ComponentStatus.ERROR else self.error,
            metadata=self.metadata.copy()
        )

    def with_error(self, error: PortalError) -> 'StatusReport':
        """Create a copy with error status."""
        return StatusReport(
            component_name=self.component_name,
            status=ComponentStatus.ERROR,
            metrics=self.metrics.copy(),
            last_activity=datetime.now(),
            started_at=self.started_at,
            error=error,
            metadata=self.metadata.copy()
        )

    def with_metrics(self, **metrics: float) -> 'StatusReport':
        """Create a copy with updated metrics."""
        new_metrics = self.metrics.copy()
        new_metrics.update(metrics)
        return StatusReport(
            component_name=self.component_name,
            status=self.status,
            metrics=new_metrics,
            last_activity=datetime.now(),
            started_at=self.started_at,
            error=self.error,
            metadata=self.metadata.copy()
        )


class StatusAggregator:
    """
    Central collector for component status updates.

    Features:
    - Thread-safe status updates
    - Subscription system for real-time notifications
    - History tracking (last N updates per component)
    - Overall system health computation
    """

    def __init__(self, history_size: int = 100):
        self._statuses: Dict[str, StatusReport] = {}
        self._history: Dict[str, List[StatusReport]] = {}
        self._history_size = history_size
        self._subscribers: List[Callable[[StatusReport], None]] = []
        self._lock = threading.RLock()

    def update(self, component: str, report: StatusReport) -> None:
        """
        Update status for a component.

        Args:
            component: Component name
            report: New status report
        """
        with self._lock:
            # Ensure component name matches
            if report.component_name != component:
                report = StatusReport(
                    component_name=component,
                    status=report.status,
                    metrics=report.metrics,
                    last_activity=report.last_activity,
                    started_at=report.started_at,
                    error=report.error,
                    metadata=report.metadata
                )

            # Store current status
            self._statuses[component] = report

            # Add to history
            if component not in self._history:
                self._history[component] = []
            self._history[component].append(report)

            # Trim history
            if len(self._history[component]) > self._history_size:
                self._history[component] = self._history[component][-self._history_size:]

        # Notify subscribers (outside lock to prevent deadlocks)
        for subscriber in self._subscribers:
            try:
                subscriber(report)
            except Exception:
                pass  # Don't let subscriber errors propagate

    def get(self, component: str) -> Optional[StatusReport]:
        """Get current status for a component."""
        with self._lock:
            return self._statuses.get(component)

    def get_all(self) -> Dict[str, StatusReport]:
        """Get all current component statuses."""
        with self._lock:
            return self._statuses.copy()

    def get_history(self, component: str, limit: int = 10) -> List[StatusReport]:
        """Get recent history for a component."""
        with self._lock:
            history = self._history.get(component, [])
            return history[-limit:]

    def subscribe(self, callback: Callable[[StatusReport], None]) -> int:
        """
        Subscribe to status updates.

        Args:
            callback: Function to call on each status update

        Returns:
            Subscription ID (index)
        """
        self._subscribers.append(callback)
        return len(self._subscribers) - 1

    def unsubscribe(self, subscription_id: int) -> None:
        """Remove a subscription by ID."""
        if 0 <= subscription_id < len(self._subscribers):
            self._subscribers[subscription_id] = lambda _: None  # Replace with no-op

    @property
    def overall_status(self) -> ComponentStatus:
        """
        Compute overall system status.

        Rules:
        - If any component is ERROR -> ERROR
        - If any component is STARTING -> STARTING
        - If all components are RUNNING -> RUNNING
        - If all components are STOPPED/IDLE -> IDLE
        - Otherwise -> PAUSED
        """
        with self._lock:
            if not self._statuses:
                return ComponentStatus.IDLE

            statuses = [r.status for r in self._statuses.values()]

            if ComponentStatus.ERROR in statuses:
                return ComponentStatus.ERROR
            if ComponentStatus.STARTING in statuses:
                return ComponentStatus.STARTING
            if all(s == ComponentStatus.RUNNING for s in statuses):
                return ComponentStatus.RUNNING
            if all(s in (ComponentStatus.STOPPED, ComponentStatus.IDLE) for s in statuses):
                return ComponentStatus.IDLE

            return ComponentStatus.PAUSED

    @property
    def health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        with self._lock:
            total = len(self._statuses)
            running = sum(1 for r in self._statuses.values() if r.status == ComponentStatus.RUNNING)
            errors = sum(1 for r in self._statuses.values() if r.status == ComponentStatus.ERROR)
            healthy = sum(1 for r in self._statuses.values() if r.status.is_healthy)

            return {
                'overall_status': self.overall_status.value,
                'total_components': total,
                'running': running,
                'errors': errors,
                'healthy': healthy,
                'health_percent': (healthy / total * 100) if total > 0 else 100.0,
                'components': {
                    name: report.to_dict()
                    for name, report in self._statuses.items()
                }
            }

    def clear(self) -> None:
        """Clear all status data."""
        with self._lock:
            self._statuses.clear()
            self._history.clear()

    def remove(self, component: str) -> None:
        """Remove a component from tracking."""
        with self._lock:
            self._statuses.pop(component, None)
            self._history.pop(component, None)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_status(
    component: str,
    status: ComponentStatus = ComponentStatus.IDLE,
    **metrics: float
) -> StatusReport:
    """
    Create a new StatusReport.

    Args:
        component: Component name
        status: Initial status
        **metrics: Initial metrics as keyword arguments

    Returns:
        New StatusReport instance
    """
    return StatusReport(
        component_name=component,
        status=status,
        metrics=metrics
    )


def running_status(component: str, **metrics: float) -> StatusReport:
    """Create a RUNNING status report."""
    return StatusReport(
        component_name=component,
        status=ComponentStatus.RUNNING,
        metrics=metrics
    )


def error_status(component: str, error: PortalError) -> StatusReport:
    """Create an ERROR status report."""
    return StatusReport(
        component_name=component,
        status=ComponentStatus.ERROR,
        error=error
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ComponentStatus',
    'StatusReport',
    'StatusAggregator',
    # Helpers
    'create_status',
    'running_status',
    'error_status',
]
