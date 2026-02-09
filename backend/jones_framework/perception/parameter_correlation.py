"""Parameter Resonance Network — cross-channel correlation engine.

Computes the full network graph from windowed drilling telemetry:
  • NxM data matrix from selected channels
  • Pearson cross-correlation via np.corrcoef
  • Per-channel dominant frequency via scipy.signal.welch (FFT PSD)
  • Z-score anomaly detection (|z| > 3.0)
  • Health status (OPTIMAL/CAUTION/WARNING/CRITICAL) per channel
  • Eigenvector centrality for node importance
  • Edge filtering by significance threshold
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import welch as _welch

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from jones_framework.core.manifold_bridge import bridge, ConnectionType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ParameterCategory(str, Enum):
    MECHANICAL = "mechanical"
    HYDRAULIC = "hydraulic"
    FORMATION = "formation"
    DIRECTIONAL = "directional"
    VIBRATION = "vibration"
    PERFORMANCE = "performance"


class HealthStatus(str, Enum):
    OPTIMAL = "optimal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParameterNode:
    id: str
    full_name: str
    category: str  # ParameterCategory value
    current_value: float
    unit: str
    health: str  # HealthStatus value
    anomaly_flag: bool
    z_score: float
    dominant_frequency_hz: float
    mean: float
    std: float
    importance: float


@dataclass
class CorrelationEdge:
    source: str
    target: str
    pearson_r: float
    is_significant: bool


@dataclass
class NetworkGraph:
    nodes: List[ParameterNode]
    edges: List[CorrelationEdge]
    strong_count: int
    anomaly_count: int
    system_health: str
    computation_time_ms: float


# ---------------------------------------------------------------------------
# 18-channel taxonomy
# ---------------------------------------------------------------------------

PARAMETER_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "WOB": {"full_name": "Weight on Bit", "category": ParameterCategory.MECHANICAL, "unit": "klbs", "record_field": "wob"},
    "TRQ": {"full_name": "Torque", "category": ParameterCategory.MECHANICAL, "unit": "ft-lb", "record_field": "torque"},
    "RPM": {"full_name": "Rotary Speed", "category": ParameterCategory.MECHANICAL, "unit": "rpm", "record_field": "rpm"},
    "HKLD": {"full_name": "Hookload", "category": ParameterCategory.MECHANICAL, "unit": "klbs", "record_field": "hookload"},
    "SPP": {"full_name": "Standpipe Pressure", "category": ParameterCategory.HYDRAULIC, "unit": "psi", "record_field": "spp"},
    "FLOW": {"full_name": "Flow Rate", "category": ParameterCategory.HYDRAULIC, "unit": "gpm", "record_field": None},
    "ECD": {"full_name": "Equiv Circ Density", "category": ParameterCategory.HYDRAULIC, "unit": "ppg", "record_field": None},
    "PUMP": {"full_name": "Pump Strokes", "category": ParameterCategory.HYDRAULIC, "unit": "spm", "record_field": None},
    "GR": {"full_name": "Gamma Ray", "category": ParameterCategory.FORMATION, "unit": "gAPI", "record_field": None},
    "RES": {"full_name": "Resistivity", "category": ParameterCategory.FORMATION, "unit": "ohm-m", "record_field": None},
    "INC": {"full_name": "Inclination", "category": ParameterCategory.DIRECTIONAL, "unit": "deg", "record_field": None},
    "AZI": {"full_name": "Azimuth", "category": ParameterCategory.DIRECTIONAL, "unit": "deg", "record_field": None},
    "DLS": {"full_name": "Dogleg Severity", "category": ParameterCategory.DIRECTIONAL, "unit": "deg/100ft", "record_field": None},
    "VIB": {"full_name": "Lateral Vibration", "category": ParameterCategory.VIBRATION, "unit": "g", "record_field": None},
    "SS": {"full_name": "Stick-Slip Index", "category": ParameterCategory.VIBRATION, "unit": "\u2014", "record_field": None},
    "ROP": {"full_name": "Rate of Penetration", "category": ParameterCategory.PERFORMANCE, "unit": "ft/hr", "record_field": "rop"},
    "MSE": {"full_name": "Mech Specific Energy", "category": ParameterCategory.PERFORMANCE, "unit": "psi", "record_field": None},
    "DEPTH": {"full_name": "Measured Depth", "category": ParameterCategory.PERFORMANCE, "unit": "ft", "record_field": "depth"},
}

# Fields directly extractable from DrillingRecord dicts
_RECORD_FIELDS = {pid: meta["record_field"] for pid, meta in PARAMETER_TAXONOMY.items() if meta["record_field"]}


def _health_from_z(z: float) -> HealthStatus:
    az = abs(z)
    if az < 1.0:
        return HealthStatus.OPTIMAL
    if az < 2.0:
        return HealthStatus.CAUTION
    if az < 3.0:
        return HealthStatus.WARNING
    return HealthStatus.CRITICAL


_HEALTH_SEVERITY = {
    HealthStatus.OPTIMAL: 0,
    HealthStatus.CAUTION: 1,
    HealthStatus.WARNING: 2,
    HealthStatus.CRITICAL: 3,
}


def _worst_health(statuses: List[HealthStatus]) -> HealthStatus:
    if not statuses:
        return HealthStatus.OPTIMAL
    return max(statuses, key=lambda h: _HEALTH_SEVERITY[h])


def _dominant_frequency(series: np.ndarray, fs: float = 1.0) -> float:
    """Return dominant frequency (Hz) of *series* using Welch PSD.

    Falls back to 0.0 when scipy is unavailable or the series is too short.
    """
    if not HAS_SCIPY or len(series) < 8:
        return 0.0
    try:
        nperseg = min(len(series), 256)
        freqs, psd = _welch(series, fs=fs, nperseg=nperseg)
        if len(psd) == 0:
            return 0.0
        return float(freqs[np.argmax(psd)])
    except Exception:
        return 0.0


def _eigenvector_centrality(corr_matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvector centrality from |correlation_matrix|.

    Returns a 1-D array of importance scores normalised to [0, 1].
    """
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 0.0)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(abs_corr)
        # Take eigenvector of the largest eigenvalue
        v = np.abs(eigenvectors[:, -1])
        vmax = v.max()
        if vmax > 0:
            v = v / vmax
        return v
    except Exception:
        return np.ones(corr_matrix.shape[0])


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@bridge(
    connects_to=["ConditionState", "TDAPipeline"],
    connection_types={
        "ConditionState": ConnectionType.USES,
        "TDAPipeline": ConnectionType.TRANSFORMS,
    },
)
class ParameterCorrelationEngine:
    """Computes the parameter resonance network from drilling records."""

    def __init__(self, sample_rate_hz: float = 1.0):
        self.sample_rate_hz = sample_rate_hz

    # ---- public API -------------------------------------------------------

    def compute_network(
        self,
        records: List[Dict[str, float]],
        channels: Optional[List[str]] = None,
        correlation_threshold: float = 0.3,
        window_size: int = 50,
    ) -> NetworkGraph:
        """Compute the full network graph from *records*.

        Parameters
        ----------
        records:
            List of drilling record dicts (keys matching DrillingRecord fields).
        channels:
            Subset of parameter IDs to include.  ``None`` ➜ all channels
            for which data is available.
        correlation_threshold:
            Minimum |r| to emit an edge.
        window_size:
            Number of most-recent records to use.
        """
        t0 = time.perf_counter()

        # -- 1. resolve channels --------------------------------------------
        if channels is None:
            channels = list(PARAMETER_TAXONOMY.keys())

        # Use most recent window
        window = records[-window_size:]
        n_rows = len(window)
        if n_rows < 3:
            return NetworkGraph(
                nodes=[], edges=[], strong_count=0, anomaly_count=0,
                system_health=HealthStatus.OPTIMAL.value,
                computation_time_ms=0.0,
            )

        # -- 2. Build NxM data matrix --------------------------------------
        available_channels: List[str] = []
        column_data: List[np.ndarray] = []

        for ch in channels:
            meta = PARAMETER_TAXONOMY.get(ch)
            if meta is None:
                continue
            rf = meta["record_field"]
            if rf is not None:
                col = np.array([r.get(rf, np.nan) for r in window], dtype=np.float64)
            else:
                # Computed channels
                col = self._compute_channel(ch, window)
            # Only include if we have real data
            if col is not None and not np.all(np.isnan(col)):
                available_channels.append(ch)
                column_data.append(col)

        m = len(available_channels)
        if m < 2:
            return NetworkGraph(
                nodes=[], edges=[], strong_count=0, anomaly_count=0,
                system_health=HealthStatus.OPTIMAL.value,
                computation_time_ms=_ms(t0),
            )

        data_matrix = np.column_stack(column_data)  # (n_rows, m)

        # -- 3. Cross-correlation -------------------------------------------
        # Replace NaN with column mean for correlation
        col_means = np.nanmean(data_matrix, axis=0)
        col_stds = np.nanstd(data_matrix, axis=0)
        data_clean = np.where(np.isnan(data_matrix), col_means, data_matrix)
        corr_matrix = np.corrcoef(data_clean, rowvar=False)  # (m, m)

        # -- 4. Eigenvector centrality --------------------------------------
        importance = _eigenvector_centrality(corr_matrix.copy())

        # -- 5. Build nodes -------------------------------------------------
        nodes: List[ParameterNode] = []
        healths: List[HealthStatus] = []
        anomaly_count = 0

        for j, ch in enumerate(available_channels):
            meta = PARAMETER_TAXONOMY[ch]
            current = float(data_clean[-1, j])
            mu = float(col_means[j])
            sigma = float(col_stds[j])
            z = (current - mu) / sigma if sigma > 1e-12 else 0.0
            health = _health_from_z(z)
            healths.append(health)
            anomaly = abs(z) > 3.0
            if anomaly:
                anomaly_count += 1
            freq = _dominant_frequency(data_clean[:, j], fs=self.sample_rate_hz)

            nodes.append(ParameterNode(
                id=ch,
                full_name=meta["full_name"],
                category=meta["category"].value,
                current_value=round(current, 4),
                unit=meta["unit"],
                health=health.value,
                anomaly_flag=anomaly,
                z_score=round(z, 3),
                dominant_frequency_hz=round(freq, 4),
                mean=round(mu, 4),
                std=round(sigma, 4),
                importance=round(float(importance[j]), 4),
            ))

        # -- 6. Build edges -------------------------------------------------
        edges: List[CorrelationEdge] = []
        strong_count = 0
        for i in range(m):
            for j in range(i + 1, m):
                r = float(corr_matrix[i, j])
                if np.isnan(r):
                    continue
                sig = abs(r) >= correlation_threshold
                if sig:
                    edges.append(CorrelationEdge(
                        source=available_channels[i],
                        target=available_channels[j],
                        pearson_r=round(r, 4),
                        is_significant=True,
                    ))
                    if abs(r) > 0.6:
                        strong_count += 1

        system_health = _worst_health(healths)

        return NetworkGraph(
            nodes=nodes,
            edges=edges,
            strong_count=strong_count,
            anomaly_count=anomaly_count,
            system_health=system_health.value,
            computation_time_ms=_ms(t0),
        )

    # ---- computed channels ------------------------------------------------

    def _compute_channel(self, ch: str, records: List[Dict[str, float]]) -> Optional[np.ndarray]:
        """Derive a computed channel from raw record fields."""
        n = len(records)
        if ch == "SS":
            # Stick-Slip Index: rolling torque coefficient of variation
            trq = np.array([r.get("torque", np.nan) for r in records], dtype=np.float64)
            if np.all(np.isnan(trq)):
                return None
            mu = np.nanmean(trq)
            if mu < 1e-6:
                return np.zeros(n)
            return np.abs(trq - mu) / mu
        if ch == "MSE":
            # Mechanical Specific Energy = (WOB * RPM) / (ROP * bit_area)
            # Simplified: assume 8.5-in bit → area ~ 56.7 in²
            bit_area = 56.7
            wob = np.array([r.get("wob", np.nan) for r in records], dtype=np.float64)
            rpm = np.array([r.get("rpm", np.nan) for r in records], dtype=np.float64)
            rop = np.array([r.get("rop", np.nan) for r in records], dtype=np.float64)
            denom = rop * bit_area
            denom[denom < 1e-6] = np.nan
            return (wob * rpm) / denom
        # Channels that require LAS data — return None (unavailable from records)
        return None


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
