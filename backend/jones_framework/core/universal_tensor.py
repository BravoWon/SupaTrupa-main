"""
Universal Tensor Space — CTS Section 3 (Definitions 3.2, 3.3, 3.7)

Implements the structured decomposition U = P × T × M × F
where P = pattern, T = temporal, M = magnitude, F = frequency.

The tensor coordinate mapping φ: S → U decomposes any signal into
four interpretable components, enabling cross-domain comparison
via the weighted product metric d_U (Eq 8).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from jones_framework.core.condition_state import ConditionState
from jones_framework.core.manifold_bridge import bridge, ConnectionType


class TensorComponent(Enum):
    """The four components of Universal Tensor Space U = P × T × M × F."""
    PATTERN = auto()      # Shape descriptors, persistence diagrams, topological features
    TEMPORAL = auto()     # Duration, periodicity, phase
    MAGNITUDE = auto()    # Mean, std, range, energy
    FREQUENCY = auto()    # Dominant frequency, bandwidth, spectral centroid


@dataclass
class TensorDecomposition:
    """Result of decomposing a signal into U = P × T × M × F.

    Each component is a feature vector extracted by the corresponding
    φ_P, φ_T, φ_M, φ_F mapping (CTS Definition 3.3).
    """
    pattern: np.ndarray       # φ_P: persistence features
    temporal: np.ndarray      # φ_T: autocorrelation, periodicity, phase
    magnitude: np.ndarray     # φ_M: statistical moments
    frequency: np.ndarray     # φ_F: spectral features
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Concatenate all components into a single feature vector."""
        return np.concatenate([self.pattern, self.temporal,
                               self.magnitude, self.frequency])

    def get_component(self, component: TensorComponent) -> np.ndarray:
        """Access a specific component by enum."""
        return {
            TensorComponent.PATTERN: self.pattern,
            TensorComponent.TEMPORAL: self.temporal,
            TensorComponent.MAGNITUDE: self.magnitude,
            TensorComponent.FREQUENCY: self.frequency,
        }[component]

    def component_dimensions(self) -> Dict[TensorComponent, int]:
        """Return dimensionality of each component."""
        return {
            TensorComponent.PATTERN: len(self.pattern),
            TensorComponent.TEMPORAL: len(self.temporal),
            TensorComponent.MAGNITUDE: len(self.magnitude),
            TensorComponent.FREQUENCY: len(self.frequency),
        }


@bridge(
    connects_to=['ConditionState', 'TDAPipeline', 'ShadowTensor'],
    connection_types={
        'ConditionState': ConnectionType.USES,
        'TDAPipeline': ConnectionType.USES,
        'ShadowTensor': ConnectionType.EXTENDS,
    },
    metadata={'domain': 'core', 'version': '1.0.0', 'cts_section': '3'}
)
class UniversalTensorSpace:
    """Implements the Universal Tensor Space U = P × T × M × F.

    CTS Definition 3.2: U is the product space where each component
    captures a distinct characteristic domain of the signal.

    CTS Definition 3.3: The tensor coordinate mapping φ: S → U
    decomposes as φ(s) = (φ_P(s), φ_T(s), φ_M(s), φ_F(s)).

    CTS Definition 3.7: The tensor distance d_U uses a weighted
    product metric (Eq 8).
    """

    def __init__(
        self,
        tda_pipeline=None,
        embedding_dim: int = 3,
        delay: int = 1,
        autocorr_lags: int = 10,
        fft_top_k: int = 5,
        component_weights: Optional[Dict[TensorComponent, float]] = None,
    ):
        self._tda = tda_pipeline
        self._embedding_dim = embedding_dim
        self._delay = delay
        self._autocorr_lags = autocorr_lags
        self._fft_top_k = fft_top_k

        # Default weights for tensor distance (Eq 8)
        self._weights = component_weights or {
            TensorComponent.PATTERN: 1.0,
            TensorComponent.TEMPORAL: 1.0,
            TensorComponent.MAGNITUDE: 1.0,
            TensorComponent.FREQUENCY: 1.0,
        }

    # ------------------------------------------------------------------
    # Core decomposition: φ: S → U
    # ------------------------------------------------------------------

    def decompose(self, time_series: np.ndarray) -> TensorDecomposition:
        """Decompose a signal into U = P × T × M × F (CTS Def 3.3).

        Args:
            time_series: 1D array of signal values, or 2D (N, d) multivariate.

        Returns:
            TensorDecomposition with all four components.
        """
        if time_series.ndim == 1:
            series = time_series
        else:
            # For multivariate: use first channel for scalar features,
            # full array for pattern extraction
            series = time_series[:, 0] if time_series.shape[1] > 0 else time_series.ravel()

        pattern = self._extract_pattern(time_series)
        temporal = self._extract_temporal(series)
        magnitude = self._extract_magnitude(series)
        frequency = self._extract_frequency(series)

        return TensorDecomposition(
            pattern=pattern,
            temporal=temporal,
            magnitude=magnitude,
            frequency=frequency,
            metadata={
                'series_length': len(series),
                'embedding_dim': self._embedding_dim,
            }
        )

    def decompose_from_states(
        self, states: List[ConditionState]
    ) -> TensorDecomposition:
        """Decompose from a sequence of ConditionStates."""
        if not states:
            raise ValueError('Cannot decompose empty state list')
        vectors = np.array([s.vector for s in states])
        decomp = self.decompose(vectors)
        decomp.metadata['timestamp_range'] = (
            states[0].timestamp, states[-1].timestamp
        )
        return decomp

    # ------------------------------------------------------------------
    # φ_P: Pattern component (persistence diagrams, topological features)
    # ------------------------------------------------------------------

    def _extract_pattern(self, data: np.ndarray) -> np.ndarray:
        """φ_P: Shape descriptors via persistent homology.

        If TDA pipeline is available, computes persistence features.
        Otherwise falls back to basic shape statistics.
        """
        if data.ndim == 1:
            point_cloud = self._time_delay_embed(data)
        else:
            point_cloud = data

        if self._tda is not None:
            features = self._tda.extract_features(point_cloud)
            return np.array([
                features.get('betti_0', 0),
                features.get('betti_1', 0),
                features.get('entropy_h0', 0),
                features.get('entropy_h1', 0),
                features.get('max_lifetime_h0', 0),
                features.get('max_lifetime_h1', 0),
                features.get('mean_lifetime_h0', 0),
                features.get('mean_lifetime_h1', 0),
                features.get('n_features_h0', 0),
                features.get('n_features_h1', 0),
            ])

        # Fallback: basic shape statistics from the point cloud
        if len(point_cloud) < 2:
            return np.zeros(10)
        from scipy.spatial.distance import pdist
        dists = pdist(point_cloud)
        return np.array([
            1.0,                          # β₀ = 1 (assume connected)
            0.0,                          # β₁ = 0 (unknown without TDA)
            float(np.std(dists)),         # distance dispersion
            0.0,                          # placeholder
            float(np.max(dists)),         # max pairwise distance
            0.0,
            float(np.mean(dists)),        # mean pairwise distance
            0.0,
            float(len(point_cloud)),      # point count
            0.0,
        ])

    # ------------------------------------------------------------------
    # φ_T: Temporal component (periodicity, phase, autocorrelation)
    # ------------------------------------------------------------------

    def _extract_temporal(self, series: np.ndarray) -> np.ndarray:
        """φ_T: Duration, periodicity, phase characteristics."""
        n = len(series)
        if n < 4:
            return np.zeros(self._autocorr_lags + 3)

        features = []

        # Autocorrelation coefficients (lags 1..k)
        mean = np.mean(series)
        var = np.var(series)
        if var < 1e-12:
            autocorr = np.zeros(self._autocorr_lags)
        else:
            centered = series - mean
            autocorr = np.array([
                np.sum(centered[:n - lag] * centered[lag:]) / (var * n)
                for lag in range(1, self._autocorr_lags + 1)
            ])
        features.extend(autocorr)

        # Dominant period via FFT peak detection
        fft_vals = np.fft.rfft(series - mean)
        power = np.abs(fft_vals[1:])  # skip DC
        if len(power) > 0 and np.max(power) > 0:
            dominant_idx = np.argmax(power)
            dominant_period = n / (dominant_idx + 1)
            phase_angle = np.angle(fft_vals[dominant_idx + 1])
        else:
            dominant_period = 0.0
            phase_angle = 0.0

        features.append(dominant_period / n)  # normalized period
        features.append(phase_angle)
        features.append(float(n))  # duration (series length)

        return np.array(features)

    # ------------------------------------------------------------------
    # φ_M: Magnitude component (statistical moments)
    # ------------------------------------------------------------------

    def _extract_magnitude(self, series: np.ndarray) -> np.ndarray:
        """φ_M: Mean, std, energy, skewness, kurtosis, min, max."""
        if len(series) < 2:
            return np.zeros(7)

        mean_val = float(np.mean(series))
        std_val = float(np.std(series))
        energy = float(np.sum(series ** 2) / len(series))
        min_val = float(np.min(series))
        max_val = float(np.max(series))

        # Skewness
        if std_val > 1e-12:
            skewness = float(np.mean(((series - mean_val) / std_val) ** 3))
            kurtosis = float(np.mean(((series - mean_val) / std_val) ** 4) - 3.0)
        else:
            skewness = 0.0
            kurtosis = 0.0

        return np.array([mean_val, std_val, energy, skewness, kurtosis,
                         min_val, max_val])

    # ------------------------------------------------------------------
    # φ_F: Frequency component (spectral features via FFT)
    # ------------------------------------------------------------------

    def _extract_frequency(self, series: np.ndarray) -> np.ndarray:
        """φ_F: Dominant frequency, bandwidth, spectral centroid."""
        n = len(series)
        if n < 4:
            return np.zeros(self._fft_top_k + 3)

        fft_vals = np.fft.rfft(series - np.mean(series))
        magnitude = np.abs(fft_vals[1:])  # skip DC
        freqs = np.fft.rfftfreq(n)[1:]    # normalized frequencies

        features = []

        # Top-k frequency magnitudes
        if len(magnitude) >= self._fft_top_k:
            top_indices = np.argsort(magnitude)[-self._fft_top_k:][::-1]
            for idx in top_indices:
                features.append(float(magnitude[idx]))
        else:
            padded = np.zeros(self._fft_top_k)
            padded[:len(magnitude)] = magnitude
            features.extend(padded.tolist())

        # Spectral centroid
        total_power = np.sum(magnitude)
        if total_power > 1e-12:
            spectral_centroid = float(np.sum(freqs * magnitude) / total_power)
        else:
            spectral_centroid = 0.0
        features.append(spectral_centroid)

        # Bandwidth (spectral spread)
        if total_power > 1e-12:
            bandwidth = float(np.sqrt(
                np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / total_power
            ))
        else:
            bandwidth = 0.0
        features.append(bandwidth)

        # Spectral rolloff (frequency below which 85% of energy is concentrated)
        if total_power > 1e-12:
            cumulative = np.cumsum(magnitude)
            rolloff_idx = np.searchsorted(cumulative, 0.85 * total_power)
            rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        else:
            rolloff = 0.0
        features.append(rolloff)

        return np.array(features)

    # ------------------------------------------------------------------
    # Tensor Distance d_U (CTS Eq 8)
    # ------------------------------------------------------------------

    def compute_tensor_distance(
        self,
        u1: TensorDecomposition,
        u2: TensorDecomposition,
        weights: Optional[Dict[TensorComponent, float]] = None,
    ) -> float:
        """Weighted product metric d_U (CTS Definition 3.7, Eq 8).

        d_U(u1, u2) = sqrt(w_P * d_P² + w_T * d_T² + w_M * d_M² + w_F * d_F²)
        """
        w = weights or self._weights

        d_p = np.linalg.norm(u1.pattern - u2.pattern)
        d_t = np.linalg.norm(u1.temporal - u2.temporal)
        d_m = np.linalg.norm(u1.magnitude - u2.magnitude)
        d_f = np.linalg.norm(u1.frequency - u2.frequency)

        return float(np.sqrt(
            w[TensorComponent.PATTERN] * d_p ** 2 +
            w[TensorComponent.TEMPORAL] * d_t ** 2 +
            w[TensorComponent.MAGNITUDE] * d_m ** 2 +
            w[TensorComponent.FREQUENCY] * d_f ** 2
        ))

    # ------------------------------------------------------------------
    # Time-delay embedding (shared with ShadowTensorBuilder)
    # ------------------------------------------------------------------

    def embed_to_manifold(self, time_series: np.ndarray) -> np.ndarray:
        """Time-delay embedding: maps 1D signal to point cloud in R^d.

        This produces the point cloud M = μ(X_persist) ⊂ U on which
        persistent homology is computed.
        """
        return self._time_delay_embed(time_series)

    def _time_delay_embed(self, series: np.ndarray) -> np.ndarray:
        """Takens delay-coordinate embedding."""
        if series.ndim > 1:
            # Already a point cloud
            return series

        n = len(series)
        required = (self._embedding_dim - 1) * self._delay + 1
        if n < required:
            # Pad or return as single-column
            return series.reshape(-1, 1)

        n_points = n - (self._embedding_dim - 1) * self._delay
        embedded = np.zeros((n_points, self._embedding_dim))
        for i in range(self._embedding_dim):
            start = i * self._delay
            embedded[:, i] = series[start:start + n_points]
        return embedded

    # ------------------------------------------------------------------
    # Component weight access
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[TensorComponent, float]:
        """Return current component weights for the tensor distance."""
        return dict(self._weights)

    def set_weights(self, weights: Dict[TensorComponent, float]):
        """Update component weights."""
        self._weights.update(weights)
