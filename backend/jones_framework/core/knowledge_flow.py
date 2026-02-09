"""
Knowledge Flow: Export/Import Dimensions with Role-Based View Preferences

Models the flow of knowledge across different dimensions, accounting for:
- Export vs Import directionality
- Role-based filtering and transformation
- View preference likelihood estimation
- Multi-dimensional knowledge representation

This extends the novelty search by tracking HOW knowledge moves between
representations, not just WHAT features are extracted.

Usage:
    flow = KnowledgeFlow()

    # Export knowledge from a layer
    export = flow.export_from_layer(LayerType.STRUCTURAL, features, role=Role.ANALYST)

    # Import into another representation
    imported = flow.import_to_layer(LayerType.LINGUISTIC, export, target_role=Role.TRADER)

    # Get view preference likelihood
    likelihood = flow.view_preference_likelihood(export, role=Role.RISK_MANAGER)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from enum import Enum, auto
import numpy as np
from datetime import datetime

from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.core.novelty_search import LayerType, LayerOutput


# =============================================================================
# Role Definitions
# =============================================================================

class Role(Enum):
    """
    Roles define different perspectives on knowledge.

    Each role has different:
    - Information needs (what matters)
    - Transformation preferences (how to view)
    - Export/import patterns (what flows where)
    """
    # Core analytical roles
    ANALYST = auto()        # Deep feature examination
    TRADER = auto()         # Action-oriented signals
    RISK_MANAGER = auto()   # Tail risks, correlations
    STRATEGIST = auto()     # Big picture, regime detection

    # Domain roles
    ENGINEER = auto()       # Technical implementation
    RESEARCHER = auto()     # Novel pattern discovery
    OPERATOR = auto()       # Real-time monitoring

    # Meta roles
    AGGREGATOR = auto()     # Combines all views
    AUDITOR = auto()        # Validates consistency


class ViewDimension(Enum):
    """
    Dimensions along which knowledge can be viewed.

    These are orthogonal to LayerTypes - they represent
    HOW to interpret, not WHAT representation to use.
    """
    TEMPORAL = auto()       # Time-series view
    SPATIAL = auto()        # Cross-sectional view
    SPECTRAL = auto()       # Frequency domain view
    TOPOLOGICAL = auto()    # Shape/connectivity view
    STATISTICAL = auto()    # Distribution view
    CAUSAL = auto()         # Cause-effect view
    SEMANTIC = auto()       # Meaning/narrative view


# =============================================================================
# Knowledge Packet
# =============================================================================

@dataclass
class KnowledgePacket:
    """
    A discrete unit of knowledge that can be exported/imported.

    Contains:
    - Feature content (the actual knowledge)
    - Source metadata (where it came from)
    - Role affinity scores (how relevant for each role)
    - View projections (how to view in each dimension)
    """
    content: np.ndarray
    source_layer: LayerType
    source_role: Role
    timestamp: datetime = field(default_factory=datetime.now)

    # Role affinity: how relevant is this knowledge for each role
    role_affinities: Dict[Role, float] = field(default_factory=dict)

    # View projections: pre-computed projections for each view dimension
    view_projections: Dict[ViewDimension, np.ndarray] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimensionality(self) -> int:
        """Content vector dimensionality."""
        return len(self.content)

    def get_view(self, dimension: ViewDimension) -> np.ndarray:
        """Get content projected to a specific view dimension."""
        if dimension in self.view_projections:
            return self.view_projections[dimension]
        return self.content  # Fallback to raw content

    def relevance_for_role(self, role: Role) -> float:
        """Get relevance score for a specific role."""
        return self.role_affinities.get(role, 0.5)


# =============================================================================
# Export/Import Transformers
# =============================================================================

class KnowledgeTransformer(ABC):
    """
    Abstract transformer for knowledge export/import operations.

    Transformers handle the conversion between different
    representations while preserving semantic content.
    """

    @abstractmethod
    def transform(
        self,
        content: np.ndarray,
        from_context: Dict[str, Any],
        to_context: Dict[str, Any]
    ) -> np.ndarray:
        """Transform content between contexts."""
        pass


class RoleBasedTransformer(KnowledgeTransformer):
    """
    Transforms knowledge based on source and target roles.

    Different roles emphasize different aspects of the knowledge:
    - TRADER: Amplifies actionable signals
    - RISK_MANAGER: Amplifies tail risk indicators
    - ANALYST: Preserves full detail
    - etc.
    """

    def __init__(self):
        # Role-specific transformation weights
        self._role_weights = {
            Role.ANALYST: self._analyst_weights,
            Role.TRADER: self._trader_weights,
            Role.RISK_MANAGER: self._risk_weights,
            Role.STRATEGIST: self._strategist_weights,
            Role.ENGINEER: self._engineer_weights,
            Role.RESEARCHER: self._researcher_weights,
            Role.OPERATOR: self._operator_weights,
            Role.AGGREGATOR: self._aggregator_weights,
            Role.AUDITOR: self._auditor_weights,
        }

    def transform(
        self,
        content: np.ndarray,
        from_context: Dict[str, Any],
        to_context: Dict[str, Any]
    ) -> np.ndarray:
        """Transform content based on role transition."""
        from_role = from_context.get('role', Role.ANALYST)
        to_role = to_context.get('role', Role.ANALYST)

        # Get transformation matrix
        weights = self._role_weights.get(to_role, self._analyst_weights)
        transformed = weights(content, from_role)

        return transformed

    def _analyst_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Analyst: Full detail preservation."""
        return content.copy()

    def _trader_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Trader: Amplify signal, suppress noise."""
        if len(content) < 4:
            return content.copy()

        # Emphasize high-magnitude components (likely signals)
        magnitudes = np.abs(content)
        threshold = np.percentile(magnitudes, 75)
        mask = magnitudes > threshold
        result = content.copy()
        result[~mask] *= 0.5  # Suppress low-magnitude
        result[mask] *= 1.5   # Amplify high-magnitude
        return result

    def _risk_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Risk Manager: Amplify tail indicators."""
        if len(content) < 4:
            return content.copy()

        # Emphasize extreme values (tail risks)
        z_scores = (content - np.mean(content)) / (np.std(content) + 1e-10)
        tail_mask = np.abs(z_scores) > 1.5
        result = content.copy()
        result[tail_mask] *= 2.0  # Amplify tails
        return result

    def _strategist_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Strategist: Smooth for big picture."""
        if len(content) < 5:
            return content.copy()

        # Apply smoothing (moving average equivalent)
        kernel_size = min(5, len(content))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(content, kernel, mode='same')
        return smoothed

    def _engineer_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Engineer: Preserve structure, normalize scale."""
        norm = np.linalg.norm(content)
        if norm > 1e-10:
            return content / norm
        return content.copy()

    def _researcher_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Researcher: Emphasize novelty indicators."""
        # Compute local variance as novelty proxy
        if len(content) < 3:
            return content.copy()

        result = content.copy()
        for i in range(1, len(content) - 1):
            local_var = np.var([content[i-1], content[i], content[i+1]])
            result[i] *= (1.0 + local_var)  # Amplify high-variance regions
        return result

    def _operator_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Operator: Recent values weighted higher."""
        if len(content) < 2:
            return content.copy()

        # Exponential decay weights (recent = important)
        weights = np.exp(np.linspace(-1, 0, len(content)))
        return content * weights

    def _aggregator_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Aggregator: Average of all role perspectives."""
        results = [
            self._trader_weights(content, from_role),
            self._risk_weights(content, from_role),
            self._strategist_weights(content, from_role),
        ]
        return np.mean(results, axis=0)

    def _auditor_weights(self, content: np.ndarray, from_role: Role) -> np.ndarray:
        """Auditor: Preserve with consistency metrics."""
        # Add consistency indicator to end
        consistency = 1.0 - np.std(content) / (np.mean(np.abs(content)) + 1e-10)
        return np.append(content, consistency)


# =============================================================================
# View Preference Model
# =============================================================================

@dataclass
class ViewPreference:
    """
    A role's preference for viewing knowledge in different dimensions.

    Preferences are learned from interaction patterns.
    """
    role: Role
    dimension_weights: Dict[ViewDimension, float] = field(default_factory=dict)
    layer_preferences: Dict[LayerType, float] = field(default_factory=dict)
    history: List[Tuple[ViewDimension, float]] = field(default_factory=list)

    def likelihood(self, dimension: ViewDimension) -> float:
        """Likelihood this role will prefer viewing in given dimension."""
        return self.dimension_weights.get(dimension, 1.0 / len(ViewDimension))

    def layer_likelihood(self, layer: LayerType) -> float:
        """Likelihood this role prefers given layer."""
        return self.layer_preferences.get(layer, 1.0 / len(LayerType))

    def update(self, dimension: ViewDimension, success: float):
        """Update preference based on observed interaction success."""
        self.history.append((dimension, success))

        # Simple exponential moving average update
        current = self.dimension_weights.get(dimension, 0.5)
        alpha = 0.1
        self.dimension_weights[dimension] = (1 - alpha) * current + alpha * success


class ViewPreferenceModel:
    """
    Model of role-specific view preferences.

    Tracks and predicts which view dimensions each role
    is likely to find useful.
    """

    def __init__(self):
        self.preferences: Dict[Role, ViewPreference] = {}
        self._initialize_priors()

    def _initialize_priors(self):
        """Initialize with domain knowledge priors."""
        # ANALYST: prefers all views equally
        self.preferences[Role.ANALYST] = ViewPreference(
            role=Role.ANALYST,
            dimension_weights={d: 1.0 / len(ViewDimension) for d in ViewDimension},
            layer_preferences={l: 1.0 / len(LayerType) for l in LayerType}
        )

        # TRADER: prefers temporal, statistical
        self.preferences[Role.TRADER] = ViewPreference(
            role=Role.TRADER,
            dimension_weights={
                ViewDimension.TEMPORAL: 0.35,
                ViewDimension.STATISTICAL: 0.25,
                ViewDimension.CAUSAL: 0.20,
                ViewDimension.SPECTRAL: 0.10,
                ViewDimension.SPATIAL: 0.05,
                ViewDimension.TOPOLOGICAL: 0.03,
                ViewDimension.SEMANTIC: 0.02,
            },
            layer_preferences={
                LayerType.MATHEMATICAL: 0.5,
                LayerType.STRUCTURAL: 0.3,
                LayerType.LINGUISTIC: 0.2,
            }
        )

        # RISK_MANAGER: prefers statistical, topological
        self.preferences[Role.RISK_MANAGER] = ViewPreference(
            role=Role.RISK_MANAGER,
            dimension_weights={
                ViewDimension.STATISTICAL: 0.30,
                ViewDimension.TOPOLOGICAL: 0.25,
                ViewDimension.TEMPORAL: 0.20,
                ViewDimension.CAUSAL: 0.15,
                ViewDimension.SPECTRAL: 0.05,
                ViewDimension.SPATIAL: 0.03,
                ViewDimension.SEMANTIC: 0.02,
            },
            layer_preferences={
                LayerType.MATHEMATICAL: 0.4,
                LayerType.STRUCTURAL: 0.4,
                LayerType.LINGUISTIC: 0.2,
            }
        )

        # STRATEGIST: prefers semantic, causal
        self.preferences[Role.STRATEGIST] = ViewPreference(
            role=Role.STRATEGIST,
            dimension_weights={
                ViewDimension.SEMANTIC: 0.30,
                ViewDimension.CAUSAL: 0.25,
                ViewDimension.TEMPORAL: 0.20,
                ViewDimension.TOPOLOGICAL: 0.15,
                ViewDimension.STATISTICAL: 0.05,
                ViewDimension.SPECTRAL: 0.03,
                ViewDimension.SPATIAL: 0.02,
            },
            layer_preferences={
                LayerType.LINGUISTIC: 0.5,
                LayerType.STRUCTURAL: 0.3,
                LayerType.MATHEMATICAL: 0.2,
            }
        )

        # RESEARCHER: prefers topological, spectral
        self.preferences[Role.RESEARCHER] = ViewPreference(
            role=Role.RESEARCHER,
            dimension_weights={
                ViewDimension.TOPOLOGICAL: 0.30,
                ViewDimension.SPECTRAL: 0.25,
                ViewDimension.CAUSAL: 0.20,
                ViewDimension.STATISTICAL: 0.15,
                ViewDimension.TEMPORAL: 0.05,
                ViewDimension.SEMANTIC: 0.03,
                ViewDimension.SPATIAL: 0.02,
            },
            layer_preferences={
                LayerType.STRUCTURAL: 0.5,
                LayerType.MATHEMATICAL: 0.35,
                LayerType.LINGUISTIC: 0.15,
            }
        )

        # Default for other roles
        for role in Role:
            if role not in self.preferences:
                self.preferences[role] = ViewPreference(
                    role=role,
                    dimension_weights={d: 1.0 / len(ViewDimension) for d in ViewDimension},
                    layer_preferences={l: 1.0 / len(LayerType) for l in LayerType}
                )

    def get_preference(self, role: Role) -> ViewPreference:
        """Get preference model for a role."""
        return self.preferences.get(role, self.preferences[Role.ANALYST])

    def likelihood(self, role: Role, dimension: ViewDimension) -> float:
        """Get likelihood of role preferring a view dimension."""
        pref = self.get_preference(role)
        return pref.likelihood(dimension)

    def layer_likelihood(self, role: Role, layer: LayerType) -> float:
        """Get likelihood of role preferring a layer."""
        pref = self.get_preference(role)
        return pref.layer_likelihood(layer)

    def best_view_for_role(self, role: Role) -> ViewDimension:
        """Get the most likely preferred view for a role."""
        pref = self.get_preference(role)
        return max(pref.dimension_weights.keys(), key=lambda d: pref.likelihood(d))

    def best_layer_for_role(self, role: Role) -> LayerType:
        """Get the most likely preferred layer for a role."""
        pref = self.get_preference(role)
        return max(pref.layer_preferences.keys(), key=lambda l: pref.layer_likelihood(l))


# =============================================================================
# Main Knowledge Flow Manager
# =============================================================================

@bridge(
    connects_to=['ConditionState', 'TDAPipeline', 'MixtureOfExperts'],
    connection_types={
        'ConditionState': ConnectionType.TRANSFORMS,
        'TDAPipeline': ConnectionType.USES,
        'MixtureOfExperts': ConnectionType.USES
    }
)
class KnowledgeFlow:
    """
    Manages the flow of knowledge across layers, roles, and view dimensions.

    Handles:
    - Export from one layer/role
    - Import to another layer/role
    - View preference likelihood computation
    - Role-based transformations
    """

    def __init__(self):
        self.transformer = RoleBasedTransformer()
        self.preference_model = ViewPreferenceModel()

        # Flow history
        self.export_history: List[KnowledgePacket] = []
        self.import_history: List[Tuple[KnowledgePacket, LayerType, Role]] = []

        # View projectors
        self._projectors = self._initialize_projectors()

        # Optional event store integration
        self._event_store = None

    def set_event_store(self, event_store):
        """
        Wire an EventStore for emitting knowledge flow events.

        Args:
            event_store: EventStore instance from jones_framework.events.sourcing
        """
        self._event_store = event_store

    def _initialize_projectors(self) -> Dict[ViewDimension, Callable[[np.ndarray], np.ndarray]]:
        """Initialize view dimension projectors."""
        return {
            ViewDimension.TEMPORAL: self._temporal_projection,
            ViewDimension.SPATIAL: self._spatial_projection,
            ViewDimension.SPECTRAL: self._spectral_projection,
            ViewDimension.TOPOLOGICAL: self._topological_projection,
            ViewDimension.STATISTICAL: self._statistical_projection,
            ViewDimension.CAUSAL: self._causal_projection,
            ViewDimension.SEMANTIC: self._semantic_projection,
        }

    def export_from_layer(
        self,
        layer_type: LayerType,
        features: np.ndarray,
        role: Role = Role.ANALYST,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgePacket:
        """
        Export knowledge from a layer for a specific role.

        Creates a KnowledgePacket with:
        - Role affinities computed
        - View projections pre-computed
        """
        if metadata is None:
            metadata = {}

        # Compute role affinities
        role_affinities = self._compute_role_affinities(features, layer_type)

        # Compute view projections
        view_projections = {
            dim: projector(features)
            for dim, projector in self._projectors.items()
        }

        packet = KnowledgePacket(
            content=features.copy(),
            source_layer=layer_type,
            source_role=role,
            role_affinities=role_affinities,
            view_projections=view_projections,
            metadata=metadata
        )

        self.export_history.append(packet)

        # Emit event if event store is wired
        if self._event_store:
            self._emit_export_event(packet, layer_type, role)

        return packet

    def _emit_export_event(self, packet: KnowledgePacket, layer_type: LayerType, role: Role):
        """Emit knowledge export event to event store."""
        try:
            from jones_framework.events.sourcing import Event, EventType
            event = Event(
                event_id='',  # Auto-generated
                event_type=EventType.KNOWLEDGE_EXPORTED,
                timestamp=datetime.now(),
                source='knowledge_flow',
                data={
                    'action': 'export',
                    'source_layer': layer_type.name,
                    'source_role': role.name,
                    'dimensionality': len(packet.content),
                    'role_affinities': {r.name: v for r, v in packet.role_affinities.items()},
                }
            )
            self._event_store.append('knowledge-flow', event)
        except ImportError:
            pass  # Event sourcing not available

    def import_to_layer(
        self,
        target_layer: LayerType,
        packet: KnowledgePacket,
        target_role: Role = Role.ANALYST
    ) -> np.ndarray:
        """
        Import knowledge to a target layer for a specific role.

        Applies role-based transformation to match target perspective.
        """
        # Transform based on role transition
        transformed = self.transformer.transform(
            packet.content,
            from_context={'role': packet.source_role, 'layer': packet.source_layer},
            to_context={'role': target_role, 'layer': target_layer}
        )

        self.import_history.append((packet, target_layer, target_role))

        # Emit event if event store is wired
        if self._event_store:
            self._emit_import_event(packet, target_layer, target_role)

        return transformed

    def _emit_import_event(self, packet: KnowledgePacket, target_layer: LayerType, target_role: Role):
        """Emit knowledge import event to event store."""
        try:
            from jones_framework.events.sourcing import Event, EventType
            event = Event(
                event_id='',  # Auto-generated
                event_type=EventType.KNOWLEDGE_IMPORTED,
                timestamp=datetime.now(),
                source='knowledge_flow',
                data={
                    'action': 'import',
                    'source_layer': packet.source_layer.name,
                    'source_role': packet.source_role.name,
                    'target_layer': target_layer.name,
                    'target_role': target_role.name,
                    'dimensionality': len(packet.content),
                }
            )
            self._event_store.append('knowledge-flow', event)
        except ImportError:
            pass  # Event sourcing not available

    def view_preference_likelihood(
        self,
        packet: KnowledgePacket,
        role: Role,
        dimension: Optional[ViewDimension] = None
    ) -> Union[float, Dict[ViewDimension, float]]:
        """
        Get likelihood of a role preferring to view this knowledge.

        If dimension is specified, returns single likelihood.
        Otherwise returns likelihoods for all dimensions.
        """
        if dimension is not None:
            base_likelihood = self.preference_model.likelihood(role, dimension)
            # Modulate by content relevance
            relevance = packet.relevance_for_role(role)
            return base_likelihood * relevance

        # Return all dimension likelihoods
        return {
            dim: self.preference_model.likelihood(role, dim) * packet.relevance_for_role(role)
            for dim in ViewDimension
        }

    def best_export_target(
        self,
        packet: KnowledgePacket,
        candidate_roles: Optional[List[Role]] = None
    ) -> Tuple[Role, LayerType, float]:
        """
        Find the best role and layer to export knowledge to.

        Returns (role, layer, score) tuple.
        """
        if candidate_roles is None:
            candidate_roles = list(Role)

        best_role, best_layer, best_score = None, None, -1.0

        for role in candidate_roles:
            for layer in LayerType:
                # Score = role affinity * layer preference
                affinity = packet.relevance_for_role(role)
                layer_pref = self.preference_model.layer_likelihood(role, layer)
                score = affinity * layer_pref

                if score > best_score:
                    best_role, best_layer, best_score = role, layer, score

        return best_role, best_layer, best_score

    def _compute_role_affinities(
        self,
        features: np.ndarray,
        source_layer: LayerType
    ) -> Dict[Role, float]:
        """Compute how relevant these features are for each role."""
        affinities = {}

        # Feature statistics
        if len(features) == 0:
            return {role: 0.5 for role in Role}

        mean_val = np.mean(features)
        std_val = np.std(features) + 1e-10
        max_val = np.max(np.abs(features))
        skew = np.mean(((features - mean_val) / std_val) ** 3) if std_val > 1e-10 else 0

        for role in Role:
            # Base affinity from layer preference
            base = self.preference_model.layer_likelihood(role, source_layer)

            # Modulate by feature characteristics
            if role == Role.TRADER:
                # Traders like high signal (high max/mean ratio)
                signal_ratio = max_val / (np.abs(mean_val) + 1e-10)
                affinity = base * min(2.0, signal_ratio / 2.0)

            elif role == Role.RISK_MANAGER:
                # Risk managers care about tails (high kurtosis / skew)
                tail_indicator = np.abs(skew) + 0.5
                affinity = base * min(2.0, tail_indicator)

            elif role == Role.STRATEGIST:
                # Strategists prefer smooth, interpretable
                smoothness = 1.0 / (std_val + 0.1)
                affinity = base * min(2.0, smoothness)

            elif role == Role.RESEARCHER:
                # Researchers like variance (interesting patterns)
                variance_indicator = std_val / (np.abs(mean_val) + 1e-10)
                affinity = base * min(2.0, variance_indicator)

            else:
                affinity = base

            affinities[role] = float(min(1.0, affinity))

        return affinities

    # View Projectors
    def _temporal_projection(self, features: np.ndarray) -> np.ndarray:
        """Project to temporal view (preserve order, emphasize changes)."""
        if len(features) < 2:
            return features.copy()
        diff = np.diff(features, prepend=features[0])
        return np.stack([features, diff]).flatten()

    def _spatial_projection(self, features: np.ndarray) -> np.ndarray:
        """Project to spatial view (preserve structure)."""
        return features.copy()

    def _spectral_projection(self, features: np.ndarray) -> np.ndarray:
        """Project to frequency domain."""
        if len(features) < 2:
            return features.copy()
        fft = np.fft.fft(features)
        return np.abs(fft)

    def _topological_projection(self, features: np.ndarray) -> np.ndarray:
        """Project to topological view (shape invariants)."""
        if len(features) < 3:
            return features.copy()

        # Approximate Betti-like invariants
        crossings = np.sum(np.diff(np.sign(features - np.mean(features))) != 0)
        peaks = np.sum((features[1:-1] > features[:-2]) & (features[1:-1] > features[2:]))
        return np.array([crossings, peaks, np.std(features), np.max(features) - np.min(features)])

    def _statistical_projection(self, features: np.ndarray) -> np.ndarray:
        """Project to statistical summary."""
        if len(features) == 0:
            return np.zeros(6)

        mean = np.mean(features)
        std = np.std(features)
        skew = np.mean(((features - mean) / (std + 1e-10)) ** 3) if std > 1e-10 else 0
        kurt = np.mean(((features - mean) / (std + 1e-10)) ** 4) - 3 if std > 1e-10 else 0

        return np.array([
            mean, std, skew, kurt,
            np.min(features), np.max(features)
        ])

    def _causal_projection(self, features: np.ndarray) -> np.ndarray:
        """Project to causal view (lag relationships)."""
        if len(features) < 3:
            return features.copy()

        # Simple auto-correlation as causal proxy
        lags = []
        for lag in range(1, min(5, len(features))):
            corr = np.corrcoef(features[:-lag], features[lag:])[0, 1]
            lags.append(corr if not np.isnan(corr) else 0.0)

        return np.array(lags)

    def _semantic_projection(self, features: np.ndarray) -> np.ndarray:
        """Project to semantic view (high-level abstractions)."""
        # Cluster into semantic categories
        if len(features) < 3:
            return features.copy()

        # Simple binning as semantic abstraction
        bins = np.percentile(features, [25, 50, 75])
        categories = np.digitize(features, bins)
        return categories.astype(np.float32)

    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge flow."""
        return {
            'total_exports': len(self.export_history),
            'total_imports': len(self.import_history),
            'exports_by_layer': {
                lt.name: sum(1 for p in self.export_history if p.source_layer == lt)
                for lt in LayerType
            },
            'exports_by_role': {
                r.name: sum(1 for p in self.export_history if p.source_role == r)
                for r in Role
            },
            'imports_by_target_role': {
                r.name: sum(1 for _, _, tr in self.import_history if tr == r)
                for r in Role
            },
        }

    def reset(self):
        """Reset flow history."""
        self.export_history = []
        self.import_history = []


# =============================================================================
# Integration with NoveltySearchLoop
# =============================================================================

def integrate_knowledge_flow(novelty_loop, knowledge_flow: KnowledgeFlow):
    """
    Integrate knowledge flow tracking into a novelty search loop.

    This wraps the iterate method to automatically export knowledge
    packets for each layer output.
    """
    original_iterate = novelty_loop.iterate

    def wrapped_iterate(state, context=None):
        result = original_iterate(state, context)

        # Export each layer output to knowledge flow
        for layer_type, layer_output in result.layer_outputs.items():
            knowledge_flow.export_from_layer(
                layer_type=layer_type,
                features=layer_output.feature_vector,
                role=Role.ANALYST,  # Default export role
                metadata={
                    'novelty_score': layer_output.novelty_score,
                    'iteration': result.iteration,
                }
            )

        return result

    novelty_loop.iterate = wrapped_iterate
    return novelty_loop


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    'Role',
    'ViewDimension',

    # Data classes
    'KnowledgePacket',
    'ViewPreference',

    # Transformers
    'KnowledgeTransformer',
    'RoleBasedTransformer',

    # Core classes
    'ViewPreferenceModel',
    'KnowledgeFlow',

    # Integration
    'integrate_knowledge_flow',
]
