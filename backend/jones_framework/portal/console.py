"""
Portal Console: Formatted console output for portal events.

Provides human-readable, colored console output with:
- Timestamps
- Component prefixes
- Status indicators
- Progress bars
- Error formatting

Usage:
    from jones_framework.portal.console import PortalConsole

    console = PortalConsole()
    console.log('NoveltySearch', 'Processing iteration 42...')
    console.status('NoveltySearch', ComponentStatus.RUNNING)
    console.error('NoveltySearch', ErrorCode.E4001_NOVELTY_ARCHIVE_FULL, 'Archive limit reached')
    console.progress('NoveltySearch', 42, 100)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TextIO, Callable
from enum import Enum
from datetime import datetime
import sys
import threading

from jones_framework.portal.status import ComponentStatus, StatusReport
from jones_framework.portal.errors import ErrorCode, PortalError, format_error


# =============================================================================
# ANSI Color Codes
# =============================================================================

class Color(Enum):
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


def colorize(text: str, *colors: Color, enabled: bool = True) -> str:
    """Apply ANSI colors to text."""
    if not enabled:
        return text
    prefix = ''.join(c.value for c in colors)
    return f"{prefix}{text}{Color.RESET.value}"


# =============================================================================
# Log Levels
# =============================================================================

class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARN = 'WARN'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'

    @property
    def color(self) -> Color:
        """Get color for this level."""
        colors = {
            LogLevel.DEBUG: Color.DIM,
            LogLevel.INFO: Color.CYAN,
            LogLevel.WARN: Color.YELLOW,
            LogLevel.ERROR: Color.RED,
            LogLevel.CRITICAL: Color.BRIGHT_RED,
        }
        return colors.get(self, Color.WHITE)

    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        priorities = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARN: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
        }
        return priorities.get(self, 1)


# =============================================================================
# Log Entry
# =============================================================================

@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: datetime
    component: str
    level: LogLevel
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'level': self.level.value,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


# =============================================================================
# Portal Console
# =============================================================================

class PortalConsole:
    """
    Formatted console output for portal events.

    Features:
    - Colored output (can be disabled)
    - Component-prefixed messages
    - Status change logging
    - Progress indicators
    - Error formatting with codes
    - Log history
    """

    # Component colors for visual distinction
    COMPONENT_COLORS = [
        Color.BRIGHT_CYAN,
        Color.BRIGHT_GREEN,
        Color.BRIGHT_MAGENTA,
        Color.BRIGHT_YELLOW,
        Color.BRIGHT_BLUE,
    ]

    def __init__(
        self,
        output: TextIO = sys.stdout,
        use_colors: bool = True,
        min_level: LogLevel = LogLevel.INFO,
        history_size: int = 1000
    ):
        self._output = output
        self._use_colors = use_colors
        self._min_level = min_level
        self._history: List[LogEntry] = []
        self._history_size = history_size
        self._component_colors: Dict[str, Color] = {}
        self._color_index = 0
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[LogEntry], None]] = []

    def _get_component_color(self, component: str) -> Color:
        """Get or assign a color for a component."""
        if component not in self._component_colors:
            self._component_colors[component] = self.COMPONENT_COLORS[
                self._color_index % len(self.COMPONENT_COLORS)
            ]
            self._color_index += 1
        return self._component_colors[component]

    def _format_timestamp(self) -> str:
        """Format current timestamp."""
        now = datetime.now()
        return now.strftime('%H:%M:%S.') + f'{now.microsecond // 1000:03d}'

    def _write(self, text: str) -> None:
        """Write to output with thread safety."""
        with self._lock:
            self._output.write(text + '\n')
            self._output.flush()

    def _add_to_history(self, entry: LogEntry) -> None:
        """Add entry to history and notify subscribers."""
        with self._lock:
            self._history.append(entry)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

        # Notify subscribers
        for sub in self._subscribers:
            try:
                sub(entry)
            except Exception:
                pass

    def log(
        self,
        component: str,
        message: str,
        level: LogLevel = LogLevel.INFO,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a message.

        Args:
            component: Component name
            message: Log message
            level: Log level
            details: Optional additional details
        """
        if level.priority < self._min_level.priority:
            return

        timestamp = self._format_timestamp()
        comp_color = self._get_component_color(component)

        # Build formatted line
        ts_str = colorize(f'[{timestamp}]', Color.DIM, enabled=self._use_colors)
        level_str = colorize(f'[{level.value:8s}]', level.color, enabled=self._use_colors)
        comp_str = colorize(f'[{component:15s}]', comp_color, enabled=self._use_colors)

        line = f"{ts_str} {level_str} {comp_str} {message}"

        if details:
            details_str = ' | '.join(f"{k}={v}" for k, v in details.items())
            line += colorize(f' ({details_str})', Color.DIM, enabled=self._use_colors)

        self._write(line)

        # Add to history
        entry = LogEntry(
            timestamp=datetime.now(),
            component=component,
            level=level,
            message=message,
            details=details
        )
        self._add_to_history(entry)

    def debug(self, component: str, message: str, **details) -> None:
        """Log a debug message."""
        self.log(component, message, LogLevel.DEBUG, details if details else None)

    def info(self, component: str, message: str, **details) -> None:
        """Log an info message."""
        self.log(component, message, LogLevel.INFO, details if details else None)

    def warn(self, component: str, message: str, **details) -> None:
        """Log a warning message."""
        self.log(component, message, LogLevel.WARN, details if details else None)

    def status(self, component: str, status: ComponentStatus) -> None:
        """
        Log a status change.

        Args:
            component: Component name
            status: New status
        """
        indicator = status.indicator
        status_name = status.value.upper()

        level = LogLevel.WARN if status == ComponentStatus.ERROR else LogLevel.INFO

        # Color the indicator
        indicator_colors = {
            ComponentStatus.IDLE: Color.DIM,
            ComponentStatus.STARTING: Color.YELLOW,
            ComponentStatus.RUNNING: Color.BRIGHT_GREEN,
            ComponentStatus.PAUSED: Color.YELLOW,
            ComponentStatus.ERROR: Color.BRIGHT_RED,
            ComponentStatus.STOPPED: Color.DIM,
        }
        ind_color = indicator_colors.get(status, Color.WHITE)
        colored_indicator = colorize(indicator, ind_color, enabled=self._use_colors)

        message = f"{colored_indicator} Status changed to {status_name}"
        self.log(component, message, level)

    def error(
        self,
        component: str,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error with code.

        Args:
            component: Component name
            error_code: Error code enum
            message: Error message
            details: Optional error details
        """
        timestamp = self._format_timestamp()
        comp_color = self._get_component_color(component)

        # Build formatted line
        ts_str = colorize(f'[{timestamp}]', Color.DIM, enabled=self._use_colors)
        code_str = colorize(f'[{error_code.value}]', Color.BRIGHT_RED, Color.BOLD, enabled=self._use_colors)
        level_str = colorize(f'[{error_code.severity:8s}]', Color.RED, enabled=self._use_colors)
        comp_str = colorize(f'[{component:15s}]', comp_color, enabled=self._use_colors)

        line = f"{ts_str} {code_str} {level_str} {comp_str} {message}"

        if details:
            details_str = ' | '.join(f"{k}={v}" for k, v in details.items())
            line += colorize(f' ({details_str})', Color.DIM, enabled=self._use_colors)

        self._write(line)

        # Add to history
        entry = LogEntry(
            timestamp=datetime.now(),
            component=component,
            level=LogLevel.ERROR,
            message=message,
            error_code=error_code.value,
            details=details
        )
        self._add_to_history(entry)

    def portal_error(self, error: PortalError) -> None:
        """Log a PortalError."""
        self.error(
            error.component,
            error.code,
            error.message,
            error.details
        )

    def progress(
        self,
        component: str,
        current: int,
        total: int,
        suffix: str = ''
    ) -> None:
        """
        Log a progress update.

        Args:
            component: Component name
            current: Current progress value
            total: Total/target value
            suffix: Optional suffix text
        """
        if total <= 0:
            pct = 100.0
        else:
            pct = (current / total) * 100

        # Build progress bar
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else bar_width
        bar = '#' * filled + '-' * (bar_width - filled)

        # Color based on completion
        if pct >= 100:
            bar_color = Color.BRIGHT_GREEN
        elif pct >= 50:
            bar_color = Color.BRIGHT_CYAN
        else:
            bar_color = Color.YELLOW

        colored_bar = colorize(bar, bar_color, enabled=self._use_colors)
        pct_str = colorize(f'{pct:6.1f}%', bar_color, enabled=self._use_colors)

        message = f"[{colored_bar}] {pct_str} ({current}/{total})"
        if suffix:
            message += f" {suffix}"

        self.log(component, message, LogLevel.INFO)

    def divider(self, char: str = '-', width: int = 80) -> None:
        """Print a divider line."""
        line = colorize(char * width, Color.DIM, enabled=self._use_colors)
        self._write(line)

    def header(self, title: str) -> None:
        """Print a header with title."""
        self.divider()
        centered = title.center(78)
        line = colorize(f'|{centered}|', Color.BRIGHT_CYAN, enabled=self._use_colors)
        self._write(line)
        self.divider()

    def status_report(self, report: StatusReport) -> None:
        """
        Log a full status report.

        Args:
            report: StatusReport to log
        """
        self.status(report.component_name, report.status)

        # Log metrics if any
        if report.metrics:
            metrics_str = ' | '.join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in report.metrics.items())
            self.debug(report.component_name, f"Metrics: {metrics_str}")

        # Log error if any
        if report.error:
            self.portal_error(report.error)

    def get_history(self, limit: int = 100) -> List[LogEntry]:
        """Get recent log history."""
        with self._lock:
            return self._history[-limit:]

    def subscribe(self, callback: Callable[[LogEntry], None]) -> int:
        """Subscribe to log entries. Returns subscription ID."""
        self._subscribers.append(callback)
        return len(self._subscribers) - 1

    def clear_history(self) -> None:
        """Clear log history."""
        with self._lock:
            self._history.clear()

    def set_min_level(self, level: LogLevel) -> None:
        """Set minimum log level."""
        self._min_level = level


# =============================================================================
# Global Console Instance
# =============================================================================

_global_console: Optional[PortalConsole] = None


def get_console() -> PortalConsole:
    """Get the global console instance."""
    global _global_console
    if _global_console is None:
        _global_console = PortalConsole()
    return _global_console


def set_console(console: PortalConsole) -> None:
    """Set the global console instance."""
    global _global_console
    _global_console = console


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Color',
    'colorize',
    'LogLevel',
    'LogEntry',
    'PortalConsole',
    'get_console',
    'set_console',
]
