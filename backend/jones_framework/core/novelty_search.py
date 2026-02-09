"""
Novelty Search Loop: Iterative Multi-Layer Feature Exploration

Sequences through structural, linguistic, and mathematical layers with equal
weighting and a strong upward arc for novelty discovery.

Key concepts:
- LayerProcessor: Abstract processor for each layer type
- NoveltyGradient: Tracks novelty scores with upward pressure
- NoveltySearchLoop: Main orchestrator that iterates through all layers

The innovation is the equal-weight combination across fundamentally different
representation spaces (topology, language, algebra) with novelty-seeking behavior.

Usage:
    loop = NoveltySearchLoop()

    # Single iteration through all layers
    result = loop.iterate(state, telemetry)

    # Run full search with novelty pressure
    trajectory = loop.search(states, max_iterations=100, novelty_threshold=0.8)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import numpy as np
from datetime import datetime
import hashlib

from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID
from jones_framework.core.manifold_bridge import bridge, ConnectionType


# =============================================================================
# Layer Definitions
# =============================================================================

class LayerType(Enum):
    """The three fundamental representation layers."""
    STRUCTURAL = auto()    # Manifold topology, graph structure, connectivity
    LINGUISTIC = auto()    # Sentiment, narrative, semantic embeddings
    MATHEMATICAL = auto()  # TDA features, algebraic invariants, metrics


@dataclass
class LayerOutput:
    """Output from a single layer processor."""
    layer_type: LayerType
    feature_vector: np.ndarray
    novelty_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def fingerprint(self) -> str:
        """Unique hash of this output for novelty comparison."""
        data = self.feature_vector.tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def distance_to(self, other: 'LayerOutput') -> float:
        """Euclidean distance between feature vectors."""
        if len(self.feature_vector) != len(other.feature_vector):
            # Pad shorter vector
            max_len = max(len(self.feature_vector), len(other.feature_vector))
            v1 = np.pad(self.feature_vector, (0, max_len - len(self.feature_vector)))
            v2 = np.pad(other.feature_vector, (0, max_len - len(other.feature_vector)))
        else:
            v1, v2 = self.feature_vector, other.feature_vector
        return float(np.linalg.norm(v1 - v2))


# =============================================================================
# Abstract Layer Processor
# =============================================================================

class LayerProcessor(ABC):
    """
    Abstract base class for layer-specific processing.

    Each layer extracts features from its representation space
    and computes a novelty score relative to the archive.
    """

    def __init__(self, layer_type: LayerType):
        self.layer_type = layer_type
        self.archive: List[LayerOutput] = []
        self._archive_max_size = 1000

    @abstractmethod
    def extract_features(self, state: ConditionState,
                         context: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from state in this layer's representation."""
        pass

    def compute_novelty(self, features: np.ndarray, k: int = 15) -> float:
        """
        Compute novelty score as average distance to k-nearest neighbors.

        Higher score = more novel (farther from archive).
        """
        if len(self.archive) == 0:
            return 1.0  # First sample is maximally novel

        distances = []
        for archived in self.archive:
            dist = np.linalg.norm(features - archived.feature_vector)
            distances.append(dist)

        distances = sorted(distances)
        k_nearest = distances[:min(k, len(distances))]

        if len(k_nearest) == 0:
            return 1.0

        # Normalize by archive statistics
        mean_dist = np.mean(k_nearest)
        if len(self.archive) > 1:
            all_dists = [a.feature_vector for a in self.archive[-100:]]
            if len(all_dists) > 1:
                pairwise = []
                for i in range(len(all_dists)):
                    for j in range(i + 1, len(all_dists)):
                        pairwise.append(np.linalg.norm(all_dists[i] - all_dists[j]))
                if pairwise:
                    baseline = np.mean(pairwise)
                    return float(mean_dist / (baseline + 1e-10))

        return float(mean_dist)

    def process(self, state: ConditionState,
                context: Dict[str, Any]) -> LayerOutput:
        """Process state through this layer."""
        features = self.extract_features(state, context)
        novelty = self.compute_novelty(features)

        output = LayerOutput(
            layer_type=self.layer_type,
            feature_vector=features,
            novelty_score=novelty,
            metadata=context.copy()
        )

        # Add to archive
        self.archive.append(output)
        if len(self.archive) > self._archive_max_size:
            self.archive = self.archive[-self._archive_max_size:]

        return output

    def reset_archive(self):
        """Clear the novelty archive."""
        self.archive = []


# =============================================================================
# Structural Layer: Manifold Topology
# =============================================================================

@bridge(
    connects_to=['ConditionState', 'TDAPipeline'],
    connection_types={
        'ConditionState': ConnectionType.USES,
        'TDAPipeline': ConnectionType.USES
    }
)
class StructuralLayerProcessor(LayerProcessor):
    """
    Extracts structural/topological features from the state manifold.

    Features include:
    - Graph connectivity metrics
    - Manifold curvature estimates
    - Betti numbers (connected components, loops, voids)
    - Persistence diagram statistics
    """

    def __init__(self):
        super().__init__(LayerType.STRUCTURAL)
        self._tda_pipeline = None

    def _get_tda_pipeline(self):
        """Lazy load TDA pipeline."""
        if self._tda_pipeline is None:
            from jones_framework.perception.tda_pipeline import TDAPipeline
            self._tda_pipeline = TDAPipeline()
        return self._tda_pipeline

    def extract_features(self, state: ConditionState,
                         context: Dict[str, Any]) -> np.ndarray:
        """Extract structural features from state."""
        state_vec = state.to_numpy()

        # Basic structural statistics
        features = [
            float(np.mean(state_vec)),
            float(np.std(state_vec)),
            float(np.max(state_vec) - np.min(state_vec)),  # Range
        ]

        # Gradient-based curvature estimate
        if len(state_vec) > 2:
            grad = np.gradient(state_vec)
            grad2 = np.gradient(grad)
            features.extend([
                float(np.mean(np.abs(grad))),      # Average slope
                float(np.mean(np.abs(grad2))),     # Average curvature
                float(np.max(np.abs(grad2))),      # Max curvature
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # TDA features if we have enough data
        telemetry = context.get('telemetry')
        if telemetry is not None and len(telemetry) >= 10:
            try:
                tda = self._get_tda_pipeline()
                diagram = tda.compute_persistence(telemetry)
                tda_features = diagram.to_feature_vector()
                features.extend(tda_features.tolist())
            except Exception:
                features.extend([0.0] * 10)  # Fallback
        else:
            features.extend([0.0] * 10)

        # Autocorrelation structure
        if len(state_vec) > 3:
            acf_1 = np.corrcoef(state_vec[:-1], state_vec[1:])[0, 1]
            features.append(float(acf_1) if not np.isnan(acf_1) else 0.0)
        else:
            features.append(0.0)

        return np.array(features, dtype=np.float32)


# =============================================================================
# Linguistic Layer: Sentiment & Narrative
# =============================================================================

@bridge(
    connects_to=['ConditionState', 'SentimentVectorPipeline'],
    connection_types={
        'ConditionState': ConnectionType.USES,
        'SentimentVectorPipeline': ConnectionType.USES
    }
)
class LinguisticLayerProcessor(LayerProcessor):
    """
    Extracts linguistic/semantic features from narrative context.

    Features include:
    - Sentiment vector components
    - Narrative divergence metrics
    - Fear/greed indicators
    - Semantic embedding projections
    """

    def __init__(self):
        super().__init__(LayerType.LINGUISTIC)
        self._sentiment_pipeline = None

    def _get_sentiment_pipeline(self):
        """Lazy load sentiment pipeline."""
        if self._sentiment_pipeline is None:
            try:
                from jones_framework.arbitrage.sentiment_vector import SentimentVectorPipeline
                self._sentiment_pipeline = SentimentVectorPipeline()
            except ImportError:
                self._sentiment_pipeline = None
        return self._sentiment_pipeline

    def extract_features(self, state: ConditionState,
                         context: Dict[str, Any]) -> np.ndarray:
        """Extract linguistic features."""
        features = []

        # If we have a sentiment vector in context, use it
        sentiment = context.get('sentiment_vector')
        if sentiment is not None:
            features.extend([
                float(getattr(sentiment, 'fear', 0.0)),
                float(getattr(sentiment, 'greed', 0.0)),
                float(getattr(sentiment, 'urgency', 0.0)),
                float(getattr(sentiment, 'distrust', 0.0)),
                float(getattr(sentiment, 'contagion', 0.0)),
                float(getattr(sentiment, 'divergence', 0.0)),
            ])
        else:
            # Derive pseudo-sentiment from state dynamics
            state_vec = state.to_numpy()

            # Fear proxy: high volatility + downward momentum
            if len(state_vec) > 1:
                volatility = np.std(state_vec)
                momentum = state_vec[-1] - state_vec[0] if len(state_vec) > 0 else 0
                fear_proxy = volatility * (1.0 if momentum < 0 else 0.5)
                greed_proxy = (1.0 - volatility) * (1.0 if momentum > 0 else 0.5)
            else:
                fear_proxy, greed_proxy = 0.5, 0.5

            features.extend([
                float(fear_proxy),
                float(greed_proxy),
                0.5,  # urgency (neutral)
                0.5,  # distrust (neutral)
                0.5,  # contagion (neutral)
                0.0,  # divergence
            ])

        # Narrative context features
        narrative_tension = context.get('narrative_tension', 0.5)
        features.append(float(narrative_tension))

        # Regime stress indicator
        regime_stress = context.get('regime_stress', 0.0)
        features.append(float(regime_stress))

        # Pad to consistent size
        while len(features) < 16:
            features.append(0.0)

        return np.array(features[:16], dtype=np.float32)


# =============================================================================
# Mathematical Layer: Algebraic Invariants
# =============================================================================

@bridge(
    connects_to=['ConditionState', 'Tensor'],
    connection_types={
        'ConditionState': ConnectionType.USES,
        'Tensor': ConnectionType.USES
    }
)
class MathematicalLayerProcessor(LayerProcessor):
    """
    Extracts mathematical/algebraic features from state space.

    Features include:
    - Spectral properties (eigenvalues, singular values)
    - Moment statistics (skewness, kurtosis)
    - Information-theoretic measures (entropy)
    - Algebraic invariants (norms, traces)
    """

    def __init__(self):
        super().__init__(LayerType.MATHEMATICAL)

    def extract_features(self, state: ConditionState,
                         context: Dict[str, Any]) -> np.ndarray:
        """Extract mathematical features."""
        state_vec = state.to_numpy()
        features = []

        # Moment statistics
        if len(state_vec) > 0:
            mean = np.mean(state_vec)
            std = np.std(state_vec) + 1e-10

            features.append(float(mean))
            features.append(float(std))

            # Skewness
            if std > 1e-10:
                skew = np.mean(((state_vec - mean) / std) ** 3)
                features.append(float(skew))
            else:
                features.append(0.0)

            # Kurtosis
            if std > 1e-10:
                kurt = np.mean(((state_vec - mean) / std) ** 4) - 3.0
                features.append(float(kurt))
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 1.0, 0.0, 0.0])

        # Norms
        features.append(float(np.linalg.norm(state_vec, ord=1)))  # L1
        features.append(float(np.linalg.norm(state_vec, ord=2)))  # L2
        features.append(float(np.linalg.norm(state_vec, ord=np.inf)))  # Linf

        # Entropy (treating as discrete distribution)
        if len(state_vec) > 0:
            # Shift to positive and normalize
            shifted = state_vec - np.min(state_vec) + 1e-10
            probs = shifted / np.sum(shifted)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            features.append(float(entropy))
        else:
            features.append(0.0)

        # Spectral features from covariance structure
        telemetry = context.get('telemetry')
        if telemetry is not None and len(telemetry) >= 5:
            try:
                cov = np.cov(telemetry.T)
                if cov.ndim == 2:
                    eigenvalues = np.linalg.eigvalsh(cov)
                    eigenvalues = np.sort(eigenvalues)[::-1]

                    # Top eigenvalue ratio (concentration)
                    total_var = np.sum(eigenvalues)
                    if total_var > 1e-10:
                        features.append(float(eigenvalues[0] / total_var))
                    else:
                        features.append(1.0)

                    # Effective dimension
                    eff_dim = np.sum(eigenvalues > 1e-10)
                    features.append(float(eff_dim))

                    # Condition number proxy
                    if eigenvalues[-1] > 1e-10:
                        features.append(float(eigenvalues[0] / eigenvalues[-1]))
                    else:
                        features.append(1.0)
                else:
                    features.extend([1.0, 1.0, 1.0])
            except Exception:
                features.extend([1.0, 1.0, 1.0])
        else:
            features.extend([1.0, 1.0, 1.0])

        # Fourier features (frequency content)
        if len(state_vec) >= 4:
            fft = np.fft.fft(state_vec)
            magnitudes = np.abs(fft)

            # Dominant frequency energy
            total_energy = np.sum(magnitudes ** 2)
            if total_energy > 1e-10:
                dom_freq_energy = magnitudes[1] ** 2 / total_energy
                features.append(float(dom_freq_energy))
            else:
                features.append(0.0)

            # High frequency ratio
            mid = len(magnitudes) // 2
            high_freq_energy = np.sum(magnitudes[mid:] ** 2)
            if total_energy > 1e-10:
                features.append(float(high_freq_energy / total_energy))
            else:
                features.append(0.5)
        else:
            features.extend([0.0, 0.5])

        # Pad to consistent size
        while len(features) < 16:
            features.append(0.0)

        return np.array(features[:16], dtype=np.float32)


# =============================================================================
# Novelty Gradient with Upward Arc
# =============================================================================

@dataclass
class NoveltyGradient:
    """
    Tracks novelty trajectory with upward pressure.

    The upward arc is achieved by:
    1. Penalizing decreasing novelty
    2. Boosting exploration when novelty drops
    3. Maintaining a target novelty floor that rises over time

    Optionally integrates with ContinuityGuard to validate transitions.
    """
    history: List[float] = field(default_factory=list)
    target_floor: float = 0.3
    floor_growth_rate: float = 0.001
    arc_pressure: float = 1.5  # >1 means upward pressure
    _continuity_guard: Any = field(default=None, repr=False)
    _validation_log: List[Tuple[float, str]] = field(default_factory=list, repr=False)

    def set_continuity_guard(self, guard):
        """
        Set a ContinuityGuard to validate novelty transitions.

        Args:
            guard: ContinuityGuard instance from jones_framework.sans.continuity_guard
        """
        self._continuity_guard = guard

    def update(self, novelty: float) -> float:
        """
        Update gradient with new novelty score.

        Returns adjusted novelty with upward arc pressure.
        """
        # Validate transition with continuity guard if available
        if self._continuity_guard and len(self.history) > 0:
            prev = self.history[-1]
            delta = abs(novelty - prev)
            if delta > 0.5:  # Large jump - smooth it
                # Use exponential smoothing for discontinuous jumps
                novelty = 0.7 * novelty + 0.3 * prev
                self._validation_log.append((novelty, 'smoothed'))

        self.history.append(novelty)

        # Raise floor over time
        self.target_floor = min(0.9, self.target_floor + self.floor_growth_rate)

        # Apply upward arc pressure
        if novelty < self.target_floor:
            # Below floor: boost toward floor
            adjusted = novelty + (self.target_floor - novelty) * 0.5
        else:
            # Above floor: apply mild upward pressure
            adjusted = novelty ** (1.0 / self.arc_pressure)

        return float(adjusted)

    @property
    def trend(self) -> float:
        """Compute recent novelty trend (positive = increasing)."""
        if len(self.history) < 3:
            return 0.0

        recent = self.history[-10:]
        if len(recent) < 2:
            return 0.0

        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)

    @property
    def current(self) -> float:
        """Current novelty value."""
        return self.history[-1] if self.history else 0.0

    @property
    def momentum(self) -> float:
        """Novelty momentum (weighted recent average)."""
        if len(self.history) == 0:
            return 0.0

        recent = self.history[-20:]
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        weights /= np.sum(weights)
        return float(np.dot(recent, weights))

    def reset(self):
        """Reset gradient tracking."""
        self.history = []
        self.target_floor = 0.3
        self._validation_log = []

    def get_validation_log(self) -> List[Tuple[float, str]]:
        """Get log of continuity validations."""
        return self._validation_log.copy()


# =============================================================================
# Combined Output
# =============================================================================

@dataclass
class NoveltySearchResult:
    """Result from one iteration of the novelty search loop."""
    iteration: int
    layer_outputs: Dict[LayerType, LayerOutput]
    combined_features: np.ndarray
    combined_novelty: float
    adjusted_novelty: float  # After upward arc pressure
    gradient_trend: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_novel(self) -> bool:
        """Whether this result represents a novel discovery."""
        return self.adjusted_novelty > 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'iteration': self.iteration,
            'combined_novelty': self.combined_novelty,
            'adjusted_novelty': self.adjusted_novelty,
            'gradient_trend': self.gradient_trend,
            'is_novel': self.is_novel,
            'layer_novelties': {
                lt.name: lo.novelty_score
                for lt, lo in self.layer_outputs.items()
            },
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# Main Novelty Search Loop
# =============================================================================

@bridge(
    connects_to=['ConditionState', 'TDAPipeline', 'MixtureOfExperts', 'SentimentVectorPipeline'],
    connection_types={
        'ConditionState': ConnectionType.USES,
        'TDAPipeline': ConnectionType.USES,
        'MixtureOfExperts': ConnectionType.USES,
        'SentimentVectorPipeline': ConnectionType.USES
    }
)
class NoveltySearchLoop:
    """
    Iterative loop through structural, linguistic, and mathematical layers.

    Each iteration:
    1. Processes state through all three layers
    2. Combines features with equal weighting
    3. Computes combined novelty score
    4. Applies upward arc pressure via NoveltyGradient
    5. Updates archives for next iteration

    The equal weighting ensures no single representation dominates,
    while the upward arc pressure encourages continuous exploration.
    """

    def __init__(
        self,
        arc_pressure: float = 1.5,
        floor_growth_rate: float = 0.001,
        equal_weights: bool = True
    ):
        # Initialize layer processors
        self.structural = StructuralLayerProcessor()
        self.linguistic = LinguisticLayerProcessor()
        self.mathematical = MathematicalLayerProcessor()

        self.processors: Dict[LayerType, LayerProcessor] = {
            LayerType.STRUCTURAL: self.structural,
            LayerType.LINGUISTIC: self.linguistic,
            LayerType.MATHEMATICAL: self.mathematical,
        }

        # Novelty tracking
        self.gradient = NoveltyGradient(
            arc_pressure=arc_pressure,
            floor_growth_rate=floor_growth_rate
        )

        # Weights (equal by default)
        self.equal_weights = equal_weights
        self.layer_weights = {
            LayerType.STRUCTURAL: 1.0 / 3.0,
            LayerType.LINGUISTIC: 1.0 / 3.0,
            LayerType.MATHEMATICAL: 1.0 / 3.0,
        }

        # State
        self.iteration_count = 0
        self.results_history: List[NoveltySearchResult] = []

        # Optional integrations (lazy-loaded to avoid import errors)
        self._event_store = None
        self._metrics_enabled = False
        self._novelty_gauge = None
        self._discoveries_counter = None
        self._archive_gauge = None

    def set_event_store(self, event_store):
        """
        Wire an EventStore for emitting novelty discovery events.

        Args:
            event_store: EventStore instance from jones_framework.events.sourcing
        """
        self._event_store = event_store

    def enable_metrics(self, registry):
        """
        Enable metrics export to a MetricsRegistry.

        Args:
            registry: MetricsRegistry instance from jones_framework.monitoring.observability
        """
        self._metrics_enabled = True
        self._novelty_gauge = registry.gauge('novelty_search_current')
        self._discoveries_counter = registry.counter('novelty_search_discoveries')
        self._archive_gauge = registry.gauge('novelty_search_archive_size')

    def set_continuity_guard(self, guard):
        """
        Set a ContinuityGuard for validating novelty gradient transitions.

        Args:
            guard: ContinuityGuard instance from jones_framework.sans.continuity_guard
        """
        self.gradient.set_continuity_guard(guard)

    def iterate(
        self,
        state: ConditionState,
        context: Optional[Dict[str, Any]] = None
    ) -> NoveltySearchResult:
        """
        Run one iteration through all layers.

        Args:
            state: Current condition state
            context: Optional context with telemetry, sentiment, etc.

        Returns:
            NoveltySearchResult with combined features and novelty
        """
        if context is None:
            context = {}

        self.iteration_count += 1

        # Process each layer
        layer_outputs: Dict[LayerType, LayerOutput] = {}
        for layer_type, processor in self.processors.items():
            output = processor.process(state, context)
            layer_outputs[layer_type] = output

        # Combine features with equal weighting
        combined_features = self._combine_features(layer_outputs)

        # Compute combined novelty (weighted average)
        combined_novelty = sum(
            self.layer_weights[lt] * lo.novelty_score
            for lt, lo in layer_outputs.items()
        )

        # Apply upward arc pressure
        adjusted_novelty = self.gradient.update(combined_novelty)

        # Build result
        result = NoveltySearchResult(
            iteration=self.iteration_count,
            layer_outputs=layer_outputs,
            combined_features=combined_features,
            combined_novelty=combined_novelty,
            adjusted_novelty=adjusted_novelty,
            gradient_trend=self.gradient.trend
        )

        self.results_history.append(result)

        # Update metrics if enabled
        if self._metrics_enabled:
            if self._novelty_gauge:
                self._novelty_gauge.set(adjusted_novelty)
            if self._archive_gauge:
                total_archive = sum(len(p.archive) for p in self.processors.values())
                self._archive_gauge.set(float(total_archive))

        return result

    def _combine_features(
        self,
        layer_outputs: Dict[LayerType, LayerOutput]
    ) -> np.ndarray:
        """
        Combine layer features with equal weighting.

        Uses concatenation followed by normalization to ensure
        each layer contributes equally regardless of feature dimensionality.
        """
        normalized_features = []

        for layer_type, output in layer_outputs.items():
            features = output.feature_vector

            # L2 normalize each layer's features
            norm = np.linalg.norm(features)
            if norm > 1e-10:
                features = features / norm

            # Apply layer weight
            weight = self.layer_weights[layer_type]
            normalized_features.append(features * weight)

        # Concatenate
        return np.concatenate(normalized_features)

    def search(
        self,
        states: List[ConditionState],
        max_iterations: Optional[int] = None,
        novelty_threshold: float = 0.8,
        context_builder: Optional[Callable[[int, ConditionState], Dict[str, Any]]] = None
    ) -> List[NoveltySearchResult]:
        """
        Run full novelty search over a sequence of states.

        Args:
            states: Sequence of condition states to search
            max_iterations: Maximum iterations (defaults to len(states))
            novelty_threshold: Threshold for flagging novel discoveries
            context_builder: Optional function to build context for each state

        Returns:
            List of NoveltySearchResult objects
        """
        if max_iterations is None:
            max_iterations = len(states)

        results = []

        for i, state in enumerate(states[:max_iterations]):
            # Build context
            if context_builder is not None:
                context = context_builder(i, state)
            else:
                # Default context: include recent states as telemetry
                start_idx = max(0, i - 20)
                telemetry = np.array([s.to_numpy() for s in states[start_idx:i+1]])
                context = {'telemetry': telemetry if len(telemetry) > 0 else None}

            # Iterate
            result = self.iterate(state, context)
            results.append(result)

            # Log novel discoveries
            if result.is_novel:
                self._on_novel_discovery(result)

        return results

    def _on_novel_discovery(self, result: NoveltySearchResult):
        """Hook for handling novel discoveries."""
        # Update metrics
        if self._metrics_enabled and self._discoveries_counter:
            self._discoveries_counter.increment()

        # Emit event to event store
        if self._event_store:
            try:
                from jones_framework.events.sourcing import Event, EventType
                event = Event(
                    event_id='',  # Auto-generated
                    event_type=EventType.NOVELTY_DISCOVERY,
                    timestamp=datetime.now(),
                    source='novelty_search',
                    data={
                        'iteration': result.iteration,
                        'adjusted_novelty': result.adjusted_novelty,
                        'combined_novelty': result.combined_novelty,
                        'gradient_trend': result.gradient_trend,
                        'layer_novelties': {
                            lt.name: lo.novelty_score
                            for lt, lo in result.layer_outputs.items()
                        },
                    }
                )
                self._event_store.append('novelty-discoveries', event)
            except ImportError:
                pass  # Event sourcing not available

    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        if not self.results_history:
            return {'iterations': 0}

        novelties = [r.combined_novelty for r in self.results_history]
        adjusted = [r.adjusted_novelty for r in self.results_history]

        return {
            'iterations': self.iteration_count,
            'mean_novelty': float(np.mean(novelties)),
            'max_novelty': float(np.max(novelties)),
            'mean_adjusted': float(np.mean(adjusted)),
            'current_floor': self.gradient.target_floor,
            'gradient_trend': self.gradient.trend,
            'gradient_momentum': self.gradient.momentum,
            'novel_discoveries': sum(1 for r in self.results_history if r.is_novel),
            'layer_archive_sizes': {
                lt.name: len(p.archive) for lt, p in self.processors.items()
            }
        }

    def reset(self):
        """Reset search state."""
        self.iteration_count = 0
        self.results_history = []
        self.gradient.reset()
        for processor in self.processors.values():
            processor.reset_archive()

    def set_layer_weights(self, weights: Dict[LayerType, float]):
        """
        Set custom layer weights.

        Weights are automatically normalized to sum to 1.
        """
        total = sum(weights.values())
        if total > 0:
            self.layer_weights = {lt: w / total for lt, w in weights.items()}
            self.equal_weights = False

    def reset_to_equal_weights(self):
        """Reset to equal weighting."""
        self.layer_weights = {
            LayerType.STRUCTURAL: 1.0 / 3.0,
            LayerType.LINGUISTIC: 1.0 / 3.0,
            LayerType.MATHEMATICAL: 1.0 / 3.0,
        }
        self.equal_weights = True


# =============================================================================
# Convenience Factory
# =============================================================================

def create_novelty_loop(
    arc_pressure: float = 1.5,
    floor_growth_rate: float = 0.001
) -> NoveltySearchLoop:
    """
    Create a configured NoveltySearchLoop.

    Args:
        arc_pressure: Upward pressure on novelty (>1 = upward)
        floor_growth_rate: How fast the novelty floor rises

    Returns:
        Configured NoveltySearchLoop instance
    """
    return NoveltySearchLoop(
        arc_pressure=arc_pressure,
        floor_growth_rate=floor_growth_rate,
        equal_weights=True
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core types
    'LayerType',
    'LayerOutput',
    'LayerProcessor',

    # Layer processors
    'StructuralLayerProcessor',
    'LinguisticLayerProcessor',
    'MathematicalLayerProcessor',

    # Novelty tracking
    'NoveltyGradient',
    'NoveltySearchResult',

    # Main loop
    'NoveltySearchLoop',
    'create_novelty_loop',
]
