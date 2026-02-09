"""Cycle 5: Predictive Topology — forecast topological state evolution.

Provides sliding-window signature extrapolation and regime transition
probability estimation from historical windowed TDA features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


FEATURE_NAMES = [
    "betti_0",
    "betti_1",
    "entropy_h0",
    "entropy_h1",
    "total_persistence_h0",
    "total_persistence_h1",
]


@dataclass
class ForecastPoint:
    """Single forecasted window."""

    window_index: int
    betti_0: float
    betti_1: float
    entropy_h0: float
    entropy_h1: float
    total_persistence_h0: float
    total_persistence_h1: float
    confidence_upper: Dict[str, float] = field(default_factory=dict)
    confidence_lower: Dict[str, float] = field(default_factory=dict)


@dataclass
class TopologyForecast:
    """Complete topology forecast result."""

    current: Dict[str, float]
    forecast: List[ForecastPoint]
    velocity: Dict[str, float]
    acceleration: Dict[str, float]
    trend_direction: str  # 'stable', 'diverging', 'converging'
    stability_index: float  # 0=chaotic, 1=stable
    n_windows_used: int
    n_ahead: int


@dataclass
class TransitionProbability:
    """Regime transition probability result."""

    current_regime: str
    probabilities: Dict[str, float]
    trending_toward: str
    trending_away: str
    velocity_magnitude: float
    estimated_windows_to_transition: Optional[int]
    risk_level: str  # 'low', 'medium', 'high', 'critical'


class TopologyForecaster:
    """Forecast topological state evolution using windowed signature extrapolation."""

    def __init__(self, min_windows: int = 4):
        self._min_windows = min_windows

    def forecast_trajectory(
        self,
        windowed_features: List[Dict[str, float]],
        n_ahead: int = 5,
    ) -> TopologyForecast:
        """Extrapolate topological features n_ahead windows using weighted linear regression.

        Parameters
        ----------
        windowed_features : list of dict
            Each dict has keys from FEATURE_NAMES with float values.
            Must have at least ``min_windows`` entries.
        n_ahead : int
            Number of future windows to forecast.

        Returns
        -------
        TopologyForecast
        """
        n = len(windowed_features)
        if n < self._min_windows:
            raise ValueError(
                f"Need at least {self._min_windows} windows, got {n}"
            )

        # Extract time-series per feature
        series: Dict[str, np.ndarray] = {}
        for name in FEATURE_NAMES:
            series[name] = np.array(
                [w.get(name, 0.0) for w in windowed_features], dtype=np.float64
            )

        # Current values (last window)
        current = {name: float(series[name][-1]) for name in FEATURE_NAMES}

        # Weighted linear regression (exponential recency weights)
        t = np.arange(n, dtype=np.float64)
        weights = np.exp(np.linspace(-1, 0, n))  # recent windows weighted more

        slopes: Dict[str, float] = {}
        intercepts: Dict[str, float] = {}
        residual_stds: Dict[str, float] = {}

        for name in FEATURE_NAMES:
            y = series[name]
            # Weighted least squares
            W = np.diag(weights)
            X = np.column_stack([t, np.ones(n)])
            XtW = X.T @ W
            try:
                beta = np.linalg.solve(XtW @ X, XtW @ y)
            except np.linalg.LinAlgError:
                beta = np.array([0.0, float(y[-1])])
            slopes[name] = float(beta[0])
            intercepts[name] = float(beta[1])
            residuals = y - (beta[0] * t + beta[1])
            residual_stds[name] = float(np.std(residuals)) if n > 2 else 0.0

        # Velocity and acceleration
        velocity = {name: slopes[name] for name in FEATURE_NAMES}
        acceleration: Dict[str, float] = {}
        for name in FEATURE_NAMES:
            y = series[name]
            if n >= 3:
                v1 = y[-1] - y[-2]
                v2 = y[-2] - y[-3]
                acceleration[name] = float(v1 - v2)
            else:
                acceleration[name] = 0.0

        # Forecast
        forecast_points: List[ForecastPoint] = []
        for step in range(1, n_ahead + 1):
            future_t = n - 1 + step
            predicted = {}
            upper = {}
            lower = {}
            for name in FEATURE_NAMES:
                val = slopes[name] * future_t + intercepts[name]
                # Confidence band widens with forecast horizon
                band = residual_stds[name] * 1.96 * np.sqrt(step)
                predicted[name] = float(val)
                upper[name] = float(val + band)
                lower[name] = float(max(0, val - band))

            forecast_points.append(
                ForecastPoint(
                    window_index=n - 1 + step,
                    betti_0=predicted["betti_0"],
                    betti_1=predicted["betti_1"],
                    entropy_h0=predicted["entropy_h0"],
                    entropy_h1=predicted["entropy_h1"],
                    total_persistence_h0=predicted["total_persistence_h0"],
                    total_persistence_h1=predicted["total_persistence_h1"],
                    confidence_upper=upper,
                    confidence_lower=lower,
                )
            )

        # Trend direction
        total_velocity = sum(abs(v) for v in velocity.values())
        betti_velocity = abs(velocity.get("betti_0", 0)) + abs(
            velocity.get("betti_1", 0)
        )

        if total_velocity < 0.05:
            trend_direction = "stable"
        elif betti_velocity > 0.3:
            trend_direction = "diverging"
        else:
            trend_direction = "converging"

        # Stability index: inverse of coefficient of variation across features
        cvs = []
        for name in FEATURE_NAMES:
            y = series[name]
            mean = np.mean(y)
            if abs(mean) > 1e-10:
                cvs.append(np.std(y) / abs(mean))
            else:
                cvs.append(0.0)
        mean_cv = np.mean(cvs) if cvs else 0.0
        stability_index = float(max(0.0, min(1.0, 1.0 - mean_cv)))

        return TopologyForecast(
            current=current,
            forecast=forecast_points,
            velocity=velocity,
            acceleration=acceleration,
            trend_direction=trend_direction,
            stability_index=stability_index,
            n_windows_used=n,
            n_ahead=n_ahead,
        )

    def compute_transition_probabilities(
        self,
        current_features: Dict[str, float],
        regime_signatures: Dict[str, np.ndarray],
        norm_scale: Optional[np.ndarray],
        velocity: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TransitionProbability:
        """Compute probability of transitioning to each regime.

        Uses softmax over negative normalized distances, adjusted by
        velocity direction (approaching vs receding).

        Parameters
        ----------
        current_features : dict
            Current observed TDA features (10-dim).
        regime_signatures : dict
            Mapping of regime_name -> feature_vector (numpy array).
        norm_scale : array or None
            Per-dimension normalization scale from classifier.
        velocity : dict or None
            Rate of change per feature (from forecast).
        feature_names : list or None
            Feature names corresponding to vector indices.
        """
        if feature_names is None:
            feature_names = [
                "betti_0", "betti_1", "entropy_h0", "entropy_h1",
                "max_lifetime_h0", "max_lifetime_h1",
                "mean_lifetime_h0", "mean_lifetime_h1",
                "n_features_h0", "n_features_h1",
            ]

        # Compute normalized distances to each regime
        distances: Dict[str, float] = {}
        for regime_name, sig_vec in regime_signatures.items():
            sq_dist = 0.0
            for i, name in enumerate(feature_names):
                obs = current_features.get(name, 0.0)
                sig_v = float(sig_vec[i]) if i < len(sig_vec) else 0.0
                scale = float(norm_scale[i]) if norm_scale is not None and i < len(norm_scale) else 1.0
                normed_diff = (obs - sig_v) / scale if scale > 1e-10 else 0.0
                sq_dist += normed_diff ** 2
            distances[regime_name] = float(np.sqrt(sq_dist))

        # Velocity adjustment: reduce distance to regimes we're moving toward
        if velocity:
            for regime_name, sig_vec in regime_signatures.items():
                dot = 0.0
                for i, name in enumerate(feature_names):
                    obs = current_features.get(name, 0.0)
                    sig_v = float(sig_vec[i]) if i < len(sig_vec) else 0.0
                    v = velocity.get(name, 0.0)
                    direction = sig_v - obs
                    # Positive dot = moving toward this regime
                    dot += v * direction
                # Reduce effective distance if moving toward
                adjustment = np.tanh(dot * 0.5)  # bounded [-1, 1]
                distances[regime_name] *= max(0.3, 1.0 - adjustment * 0.3)

        # Softmax over negative distances -> probabilities
        neg_dists = np.array([-distances[r] for r in distances])
        # Temperature scaling for sharper/softer probabilities
        temperature = 1.0
        exp_vals = np.exp((neg_dists - np.max(neg_dists)) / temperature)
        probs = exp_vals / (np.sum(exp_vals) + 1e-10)

        regime_names = list(distances.keys())
        probabilities = {regime_names[i]: float(probs[i]) for i in range(len(regime_names))}

        # Find current regime (highest probability)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        current_regime = sorted_probs[0][0]

        # Trending toward/away
        trending_toward = sorted_probs[1][0] if len(sorted_probs) > 1 else current_regime
        trending_away = sorted_probs[-1][0] if sorted_probs else current_regime

        # Velocity magnitude
        vel_mag = 0.0
        if velocity:
            vel_mag = float(np.sqrt(sum(v ** 2 for v in velocity.values())))

        # Estimate windows to transition (rough: distance / velocity)
        dist_to_next = distances.get(trending_toward, 1.0)
        windows_to_transition: Optional[int] = None
        if vel_mag > 0.01:
            windows_to_transition = max(1, int(dist_to_next / vel_mag))

        # Risk level
        top_prob = sorted_probs[0][1]
        second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
        gap = top_prob - second_prob

        if gap > 0.4:
            risk_level = "low"
        elif gap > 0.2:
            risk_level = "medium"
        elif gap > 0.1:
            risk_level = "high"
        else:
            risk_level = "critical"

        return TransitionProbability(
            current_regime=current_regime,
            probabilities=probabilities,
            trending_toward=trending_toward,
            trending_away=trending_away,
            velocity_magnitude=vel_mag,
            estimated_windows_to_transition=windows_to_transition,
            risk_level=risk_level,
        )
