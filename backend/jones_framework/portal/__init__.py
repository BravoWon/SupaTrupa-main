"""
Portal Module: Unified execution orchestration and status management.

This module provides:
- PortalRunner: Central orchestrator for all framework components
- StatusAggregator: Component status collection and broadcasting
- PortalConsole: Formatted console logging with colors
- ErrorCode/PortalError: Standardized error classification

Usage:
    from jones_framework.portal import PortalRunner, PortalConsole

    runner = PortalRunner()
    runner.register('novelty', create_novelty_loop())
    runner.register('knowledge', KnowledgeFlow())

    # CLI mode
    results = runner.run_sync(states)

    # Server mode
    await runner.start_async()
"""

from jones_framework.portal.errors import (
    ErrorCode,
    PortalError,
    format_error,
    component_init_failed,
    processing_failed,
    timeout_error,
    continuity_violation,
    invalid_state_vector,
)

from jones_framework.portal.status import (
    ComponentStatus,
    StatusReport,
    StatusAggregator,
    create_status,
    running_status,
    error_status,
)

from jones_framework.portal.console import (
    Color,
    colorize,
    LogLevel,
    LogEntry,
    PortalConsole,
    get_console,
    set_console,
)

from jones_framework.portal.runner import (
    PortalComponent,
    ExecutionMode,
    RegisteredComponent,
    PortalRunner,
)


__all__ = [
    # Errors
    'ErrorCode',
    'PortalError',
    'format_error',
    'component_init_failed',
    'processing_failed',
    'timeout_error',
    'continuity_violation',
    'invalid_state_vector',
    # Status
    'ComponentStatus',
    'StatusReport',
    'StatusAggregator',
    'create_status',
    'running_status',
    'error_status',
    # Console
    'Color',
    'colorize',
    'LogLevel',
    'LogEntry',
    'PortalConsole',
    'get_console',
    'set_console',
    # Runner
    'PortalComponent',
    'ExecutionMode',
    'RegisteredComponent',
    'PortalRunner',
]
