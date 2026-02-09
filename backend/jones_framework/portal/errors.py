"""
Portal Error Codes: Standardized error classification for the framework.

Error code ranges:
- E1xxx: Initialization errors
- E2xxx: Runtime/processing errors
- E3xxx: Communication errors
- E4xxx: Component-specific errors
- E5xxx: Data/validation errors

Usage:
    from jones_framework.portal.errors import ErrorCode, PortalError

    raise PortalError(
        code=ErrorCode.E2001_PROCESSING_FAILED,
        component='NoveltySearch',
        message='Failed to compute novelty score',
        details={'iteration': 42, 'state_dim': 16}
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime


class ErrorCode(Enum):
    """
    Standardized error codes for the framework.

    Format: E{category}{sequence}_{description}
    """

    # ==========================================================================
    # E1xxx: Initialization Errors
    # ==========================================================================
    E1001_COMPONENT_INIT_FAILED = 'E1001'
    E1002_DEPENDENCY_MISSING = 'E1002'
    E1003_CONFIG_INVALID = 'E1003'
    E1004_REGISTRY_FAILED = 'E1004'
    E1005_EVENT_STORE_INIT_FAILED = 'E1005'
    E1006_METRICS_INIT_FAILED = 'E1006'

    # ==========================================================================
    # E2xxx: Runtime/Processing Errors
    # ==========================================================================
    E2001_PROCESSING_FAILED = 'E2001'
    E2002_STATE_INVALID = 'E2002'
    E2003_TIMEOUT = 'E2003'
    E2004_ITERATION_LIMIT = 'E2004'
    E2005_RESOURCE_EXHAUSTED = 'E2005'
    E2006_CONCURRENT_MODIFICATION = 'E2006'

    # ==========================================================================
    # E3xxx: Communication Errors
    # ==========================================================================
    E3001_WEBSOCKET_DISCONNECTED = 'E3001'
    E3002_EVENT_STORE_FAILED = 'E3002'
    E3003_BROADCAST_FAILED = 'E3003'
    E3004_SUBSCRIPTION_FAILED = 'E3004'
    E3005_MESSAGE_PARSE_ERROR = 'E3005'
    E3006_CONNECTION_REFUSED = 'E3006'

    # ==========================================================================
    # E4xxx: Component-Specific Errors
    # ==========================================================================
    # Novelty Search (E40xx)
    E4001_NOVELTY_ARCHIVE_FULL = 'E4001'
    E4002_NOVELTY_GRADIENT_DIVERGED = 'E4002'
    E4003_LAYER_PROCESSOR_FAILED = 'E4003'

    # Knowledge Flow (E41xx)
    E4101_KNOWLEDGE_FLOW_BLOCKED = 'E4101'
    E4102_ROLE_AFFINITY_FAILED = 'E4102'
    E4103_VIEW_PROJECTION_FAILED = 'E4103'

    # Mixture of Experts (E42xx)
    E4201_MOE_EXPERT_UNAVAILABLE = 'E4201'
    E4202_MOE_HOT_SWAP_FAILED = 'E4202'
    E4203_REGIME_CLASSIFICATION_FAILED = 'E4203'

    # Continuity Guard (E43xx)
    E4301_CONTINUITY_VIOLATION = 'E4301'
    E4302_SAFETY_THRESHOLD_BREACH = 'E4302'
    E4303_TRANSITION_BLOCKED = 'E4303'

    # TDA Pipeline (E44xx)
    E4401_TDA_COMPUTATION_FAILED = 'E4401'
    E4402_PERSISTENCE_DIAGRAM_INVALID = 'E4402'
    E4403_INSUFFICIENT_DATA = 'E4403'

    # ==========================================================================
    # E5xxx: Data/Validation Errors
    # ==========================================================================
    E5001_INVALID_STATE_VECTOR = 'E5001'
    E5002_DIMENSION_MISMATCH = 'E5002'
    E5003_NAN_DETECTED = 'E5003'
    E5004_INF_DETECTED = 'E5004'
    E5005_SCHEMA_VALIDATION_FAILED = 'E5005'

    @property
    def category(self) -> str:
        """Get the error category."""
        code = self.value
        cat_num = int(code[1])
        categories = {
            1: 'Initialization',
            2: 'Runtime',
            3: 'Communication',
            4: 'Component',
            5: 'Data/Validation'
        }
        return categories.get(cat_num, 'Unknown')

    @property
    def severity(self) -> str:
        """Get suggested severity level."""
        code = self.value
        cat_num = int(code[1])
        # E1xxx and E3xxx are typically critical
        # E2xxx and E4xxx are typically errors
        # E5xxx are typically warnings
        severities = {
            1: 'CRITICAL',
            2: 'ERROR',
            3: 'CRITICAL',
            4: 'ERROR',
            5: 'WARNING'
        }
        return severities.get(cat_num, 'ERROR')


@dataclass
class PortalError(Exception):
    """
    Standardized error for the portal system.

    Carries error code, component name, message, and optional details.
    Can be serialized for transmission over WebSocket.
    """
    code: ErrorCode
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return f"[{self.code.value}] [{self.component}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"PortalError(code={self.code.value}, "
            f"component={self.component!r}, message={self.message!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'code': self.code.value,
            'category': self.code.category,
            'severity': self.code.severity,
            'component': self.component,
            'message': self.message,
            'details': self.details or {},
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortalError':
        """Create from dictionary."""
        code_value = data['code']
        code = None
        for ec in ErrorCode:
            if ec.value == code_value:
                code = ec
                break
        if code is None:
            raise ValueError(f"Unknown error code: {code_value}")

        return cls(
            code=code,
            component=data['component'],
            message=data['message'],
            details=data.get('details'),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        component: str,
        code: ErrorCode = ErrorCode.E2001_PROCESSING_FAILED
    ) -> 'PortalError':
        """Create from a standard Python exception."""
        return cls(
            code=code,
            component=component,
            message=str(exc),
            details={
                'exception_type': type(exc).__name__,
                'exception_args': [str(a) for a in exc.args]
            }
        )


def format_error(error: PortalError, include_details: bool = False) -> str:
    """
    Format a PortalError for console output.

    Args:
        error: The error to format
        include_details: Whether to include the details dict

    Returns:
        Formatted error string
    """
    base = f"[{error.code.value}] [{error.code.severity}] [{error.component}] {error.message}"
    if include_details and error.details:
        details_str = ', '.join(f"{k}={v}" for k, v in error.details.items())
        base += f" ({details_str})"
    return base


# =============================================================================
# Error Helpers
# =============================================================================

def component_init_failed(component: str, reason: str) -> PortalError:
    """Create a component initialization error."""
    return PortalError(
        code=ErrorCode.E1001_COMPONENT_INIT_FAILED,
        component=component,
        message=f"Failed to initialize: {reason}"
    )


def processing_failed(component: str, operation: str, reason: str) -> PortalError:
    """Create a processing error."""
    return PortalError(
        code=ErrorCode.E2001_PROCESSING_FAILED,
        component=component,
        message=f"Failed during {operation}: {reason}"
    )


def timeout_error(component: str, operation: str, timeout_ms: int) -> PortalError:
    """Create a timeout error."""
    return PortalError(
        code=ErrorCode.E2003_TIMEOUT,
        component=component,
        message=f"Operation '{operation}' timed out after {timeout_ms}ms",
        details={'timeout_ms': timeout_ms, 'operation': operation}
    )


def continuity_violation(
    component: str,
    kl_divergence: float,
    threshold: float
) -> PortalError:
    """Create a continuity violation error."""
    return PortalError(
        code=ErrorCode.E4301_CONTINUITY_VIOLATION,
        component=component,
        message=f"Continuity violation: KL={kl_divergence:.4f} exceeds threshold={threshold:.4f}",
        details={'kl_divergence': kl_divergence, 'threshold': threshold}
    )


def invalid_state_vector(
    component: str,
    reason: str,
    vector_shape: Optional[tuple] = None
) -> PortalError:
    """Create an invalid state vector error."""
    return PortalError(
        code=ErrorCode.E5001_INVALID_STATE_VECTOR,
        component=component,
        message=f"Invalid state vector: {reason}",
        details={'vector_shape': vector_shape} if vector_shape else None
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ErrorCode',
    'PortalError',
    'format_error',
    # Helpers
    'component_init_failed',
    'processing_failed',
    'timeout_error',
    'continuity_violation',
    'invalid_state_vector',
]
