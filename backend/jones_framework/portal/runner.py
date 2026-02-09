"""
Portal Runner: Unified execution orchestrator for the framework.

Coordinates all independent modules, provides:
- Component registration and lifecycle management
- Unified status broadcasting
- Console logging integration
- Both sync (CLI) and async (server) execution modes

Usage:
    # CLI mode (synchronous)
    from jones_framework.portal import PortalRunner

    runner = PortalRunner()
    runner.register('novelty', create_novelty_loop())
    runner.register('knowledge', KnowledgeFlow())
    runner.run_sync(states)  # Batch processing

    # Server mode (asynchronous)
    await runner.start_server(port=8765)  # WebSocket streaming
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Protocol, runtime_checkable
from enum import Enum, auto
from datetime import datetime
import asyncio
import threading
import json

from jones_framework.portal.errors import (
    ErrorCode, PortalError,
    component_init_failed, processing_failed
)
from jones_framework.portal.status import (
    ComponentStatus, StatusReport, StatusAggregator,
    create_status, running_status, error_status
)
from jones_framework.portal.console import (
    PortalConsole, LogLevel, get_console
)


# =============================================================================
# Component Protocol
# =============================================================================

@runtime_checkable
class PortalComponent(Protocol):
    """
    Protocol for components that can be registered with PortalRunner.

    Components should implement at minimum:
    - get_statistics() -> Dict[str, Any]

    Optional methods:
    - reset() -> None
    - set_event_store(store) -> None
    - enable_metrics(registry) -> None
    """

    def get_statistics(self) -> Dict[str, Any]:
        """Get component statistics/metrics."""
        ...


# =============================================================================
# Execution Mode
# =============================================================================

class ExecutionMode(Enum):
    """Portal execution mode."""
    CLI = 'cli'         # Synchronous batch processing
    SERVER = 'server'   # Async WebSocket server
    HYBRID = 'hybrid'   # Both CLI output and server


# =============================================================================
# Registered Component Wrapper
# =============================================================================

@dataclass
class RegisteredComponent:
    """Wrapper for a registered component with metadata."""
    name: str
    component: Any
    status: ComponentStatus = ComponentStatus.IDLE
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    iteration_count: int = 0
    error: Optional[PortalError] = None

    def to_status_report(self) -> StatusReport:
        """Convert to StatusReport."""
        metrics = {}
        try:
            if hasattr(self.component, 'get_statistics'):
                stats = self.component.get_statistics()
                # Flatten numeric values into metrics
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
        except Exception:
            pass

        metrics['iterations'] = float(self.iteration_count)

        return StatusReport(
            component_name=self.name,
            status=self.status,
            metrics=metrics,
            last_activity=self.last_activity or datetime.now(),
            started_at=self.started_at,
            error=self.error
        )


# =============================================================================
# Portal Runner
# =============================================================================

class PortalRunner:
    """
    Unified execution orchestrator for all framework components.

    Manages:
    - Component registration and lifecycle
    - Status aggregation and broadcasting
    - Console logging
    - Sync and async execution modes
    """

    def __init__(
        self,
        console: Optional[PortalConsole] = None,
        mode: ExecutionMode = ExecutionMode.CLI
    ):
        self._components: Dict[str, RegisteredComponent] = {}
        self._status_aggregator = StatusAggregator()
        self._console = console or get_console()
        self._mode = mode

        # Execution state
        self._running = False
        self._paused = False
        self._stop_requested = False

        # Event store and metrics (optional integrations)
        self._event_store = None
        self._metrics_registry = None

        # WebSocket broadcast callbacks
        self._ws_broadcasters: List[Callable[[str, Dict[str, Any]], None]] = []

        # Lock for thread safety
        self._lock = threading.RLock()

        self._console.header('Portal Runner Initialized')

    # =========================================================================
    # Component Registration
    # =========================================================================

    def register(
        self,
        name: str,
        component: Any,
        auto_wire: bool = True
    ) -> None:
        """
        Register a component for orchestration.

        Args:
            name: Unique component name
            component: The component instance
            auto_wire: If True, automatically wire event store and metrics
        """
        with self._lock:
            if name in self._components:
                self._console.warn('Runner', f"Replacing existing component: {name}")

            wrapped = RegisteredComponent(name=name, component=component)
            self._components[name] = wrapped

            # Update status aggregator
            self._status_aggregator.update(name, wrapped.to_status_report())

            self._console.info('Runner', f"Registered component: {name}", type=type(component).__name__)

            # Auto-wire integrations
            if auto_wire:
                self._wire_component(name, component)

    def unregister(self, name: str) -> None:
        """Remove a component from orchestration."""
        with self._lock:
            if name in self._components:
                del self._components[name]
                self._status_aggregator.remove(name)
                self._console.info('Runner', f"Unregistered component: {name}")

    def _wire_component(self, name: str, component: Any) -> None:
        """Wire event store and metrics to component if supported."""
        try:
            if self._event_store and hasattr(component, 'set_event_store'):
                component.set_event_store(self._event_store)
                self._console.debug('Runner', f"Wired event store to {name}")

            if self._metrics_registry and hasattr(component, 'enable_metrics'):
                component.enable_metrics(self._metrics_registry)
                self._console.debug('Runner', f"Wired metrics to {name}")
        except Exception as e:
            self._console.warn('Runner', f"Failed to wire integrations for {name}: {e}")

    def set_event_store(self, event_store: Any) -> None:
        """Set the event store for all components."""
        self._event_store = event_store
        # Wire to existing components
        for name, wrapped in self._components.items():
            if hasattr(wrapped.component, 'set_event_store'):
                wrapped.component.set_event_store(event_store)

    def set_metrics_registry(self, registry: Any) -> None:
        """Set the metrics registry for all components."""
        self._metrics_registry = registry
        # Wire to existing components
        for name, wrapped in self._components.items():
            if hasattr(wrapped.component, 'enable_metrics'):
                wrapped.component.enable_metrics(registry)

    # =========================================================================
    # Status Management
    # =========================================================================

    def _update_component_status(
        self,
        name: str,
        status: ComponentStatus,
        error: Optional[PortalError] = None
    ) -> None:
        """Update a component's status."""
        with self._lock:
            if name not in self._components:
                return

            wrapped = self._components[name]
            wrapped.status = status
            wrapped.last_activity = datetime.now()

            if status == ComponentStatus.RUNNING and wrapped.started_at is None:
                wrapped.started_at = datetime.now()

            if error:
                wrapped.error = error
            elif status != ComponentStatus.ERROR:
                wrapped.error = None

            # Update aggregator
            report = wrapped.to_status_report()
            self._status_aggregator.update(name, report)

            # Log status change
            self._console.status(name, status)

            # Broadcast to WebSocket clients
            self._broadcast('portal.status', report.to_dict())

    def get_unified_status(self) -> Dict[str, Any]:
        """Get aggregated status from all components."""
        return self._status_aggregator.health_summary

    def get_component_status(self, name: str) -> Optional[StatusReport]:
        """Get status for a specific component."""
        return self._status_aggregator.get(name)

    # =========================================================================
    # Synchronous Execution (CLI Mode)
    # =========================================================================

    def run_sync(
        self,
        states: List[Any],
        processor: Optional[Callable[[str, Any, Any], Any]] = None
    ) -> Dict[str, Any]:
        """
        Run synchronous batch processing.

        Args:
            states: List of states to process
            processor: Optional custom processor function(component_name, component, state) -> result

        Returns:
            Results dictionary with component outputs
        """
        self._console.header('Starting Synchronous Execution')
        self._running = True
        self._stop_requested = False
        results: Dict[str, List[Any]] = {name: [] for name in self._components}

        try:
            # Mark all components as starting
            for name in self._components:
                self._update_component_status(name, ComponentStatus.STARTING)

            # Mark all as running
            for name in self._components:
                self._update_component_status(name, ComponentStatus.RUNNING)

            # Process states
            total = len(states)
            for i, state in enumerate(states):
                if self._stop_requested:
                    self._console.warn('Runner', 'Stop requested, halting execution')
                    break

                if self._paused:
                    self._console.info('Runner', 'Execution paused')
                    while self._paused and not self._stop_requested:
                        import time
                        time.sleep(0.1)
                    self._console.info('Runner', 'Execution resumed')

                # Process through each component
                for name, wrapped in self._components.items():
                    try:
                        if processor:
                            result = processor(name, wrapped.component, state)
                        else:
                            result = self._default_process(name, wrapped.component, state)

                        results[name].append(result)
                        wrapped.iteration_count += 1
                        wrapped.last_activity = datetime.now()

                        # Update status with new metrics
                        self._status_aggregator.update(name, wrapped.to_status_report())

                    except Exception as e:
                        error = PortalError.from_exception(e, name)
                        self._update_component_status(name, ComponentStatus.ERROR, error)
                        self._console.portal_error(error)
                        results[name].append(None)

                # Log progress periodically
                if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
                    self._console.progress('Runner', i + 1, total)

            # Mark all as stopped
            for name in self._components:
                self._update_component_status(name, ComponentStatus.STOPPED)

        finally:
            self._running = False

        self._console.header('Execution Complete')
        return {
            'results': results,
            'status': self.get_unified_status()
        }

    def _default_process(self, name: str, component: Any, state: Any) -> Any:
        """Default processing logic for a component."""
        # Try common method names
        if hasattr(component, 'iterate'):
            return component.iterate(state)
        elif hasattr(component, 'process'):
            return component.process(state)
        elif hasattr(component, 'export_from_layer'):
            # KnowledgeFlow special case
            import numpy as np
            from jones_framework.core.knowledge_flow import LayerType, Role
            features = state.to_numpy() if hasattr(state, 'to_numpy') else np.array(state)
            return component.export_from_layer(LayerType.STRUCTURAL, features, Role.ANALYST)
        else:
            self._console.warn(name, "No process method found, skipping")
            return None

    # =========================================================================
    # Asynchronous Execution (Server Mode)
    # =========================================================================

    async def start_async(self) -> None:
        """Start async execution loop."""
        self._console.header('Starting Async Execution')
        self._running = True
        self._stop_requested = False

        # Mark all components as starting
        for name in self._components:
            self._update_component_status(name, ComponentStatus.STARTING)

        # Mark all as running
        for name in self._components:
            self._update_component_status(name, ComponentStatus.RUNNING)

        # Start status broadcast loop
        asyncio.create_task(self._status_broadcast_loop())

        self._console.info('Runner', 'Async execution started')

    async def stop_async(self) -> None:
        """Stop async execution."""
        self._stop_requested = True
        self._running = False

        for name in self._components:
            self._update_component_status(name, ComponentStatus.STOPPED)

        self._console.info('Runner', 'Async execution stopped')

    async def _status_broadcast_loop(self, interval: float = 1.0) -> None:
        """Periodically broadcast status to WebSocket clients."""
        while self._running and not self._stop_requested:
            try:
                status = self.get_unified_status()
                self._broadcast('portal.status', status)
            except Exception as e:
                self._console.warn('Runner', f"Status broadcast failed: {e}")

            await asyncio.sleep(interval)

    async def process_async(self, state: Any) -> Dict[str, Any]:
        """
        Process a single state asynchronously.

        Args:
            state: State to process

        Returns:
            Results from all components
        """
        results = {}

        for name, wrapped in self._components.items():
            try:
                result = self._default_process(name, wrapped.component, state)
                results[name] = result
                wrapped.iteration_count += 1
                wrapped.last_activity = datetime.now()
                self._status_aggregator.update(name, wrapped.to_status_report())

            except Exception as e:
                error = PortalError.from_exception(e, name)
                self._update_component_status(name, ComponentStatus.ERROR, error)
                self._console.portal_error(error)
                results[name] = None

        return results

    # =========================================================================
    # WebSocket Broadcasting
    # =========================================================================

    def add_ws_broadcaster(self, broadcaster: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a WebSocket broadcast function."""
        self._ws_broadcasters.append(broadcaster)

    def _broadcast(self, message_type: str, data: Dict[str, Any]) -> None:
        """Broadcast to all WebSocket clients."""
        message = {
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        for broadcaster in self._ws_broadcasters:
            try:
                broadcaster(message_type, message)
            except Exception:
                pass

    # =========================================================================
    # Control Methods
    # =========================================================================

    def pause(self) -> None:
        """Pause execution."""
        self._paused = True
        for name in self._components:
            self._update_component_status(name, ComponentStatus.PAUSED)
        self._console.info('Runner', 'Execution paused')

    def resume(self) -> None:
        """Resume execution."""
        self._paused = False
        for name in self._components:
            self._update_component_status(name, ComponentStatus.RUNNING)
        self._console.info('Runner', 'Execution resumed')

    def stop(self) -> None:
        """Request execution stop."""
        self._stop_requested = True
        self._console.info('Runner', 'Stop requested')

    def reset(self) -> None:
        """Reset all components."""
        for name, wrapped in self._components.items():
            if hasattr(wrapped.component, 'reset'):
                wrapped.component.reset()
            wrapped.iteration_count = 0
            wrapped.started_at = None
            wrapped.error = None
            self._update_component_status(name, ComponentStatus.IDLE)

        self._console.info('Runner', 'All components reset')

    @property
    def is_running(self) -> bool:
        """Check if runner is currently executing."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if runner is paused."""
        return self._paused


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for portal runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Portal Runner - Framework Orchestrator')
    parser.add_argument('--mode', choices=['cli', 'server'], default='cli',
                        help='Execution mode')
    parser.add_argument('--port', type=int, default=8765,
                        help='WebSocket port for server mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    console = PortalConsole(
        min_level=LogLevel.DEBUG if args.verbose else LogLevel.INFO
    )

    runner = PortalRunner(
        console=console,
        mode=ExecutionMode(args.mode)
    )

    console.header('Portal Runner')
    console.info('Runner', f"Mode: {args.mode}")

    if args.mode == 'server':
        console.info('Runner', f"Starting server on port {args.port}")
        # Server mode would start WebSocket server here
        console.warn('Runner', 'Server mode not fully implemented in CLI entry point')
    else:
        console.info('Runner', 'CLI mode - register components and call run_sync()')


if __name__ == '__main__':
    main()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'PortalComponent',
    'ExecutionMode',
    'RegisteredComponent',
    'PortalRunner',
]
