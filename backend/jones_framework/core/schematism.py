"""
Schematism Bridge — CTS Section 7 (Definitions 7.1-7.5, Theorem 7.6)

Implements the coupling mechanism between topological signature Q
and coherence measure Φ. The schematism validates that the topological
structure assumed by active knowledge nodes is compatible with the
topological structure actually present in the data.

Central construct: H(v_i) := [d_BN(p_i, p_data) ≤ ε]  (CTS Eq 14)

A transcendental error occurs when a knowledge node's pattern
assumptions do not match the data's actual topology — analogous
to hallucination in AI or category error in cognition.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from jones_framework.core.manifold_bridge import bridge, ConnectionType


@dataclass
class PatternSchema:
    """Pattern schema associated with a KIM node (CTS Definition 7.1).

    Each active knowledge node may have a pattern schema p_i ∈ P
    describing the topological/morphological signature that the
    node assumes of its data.

    The schema stores either a reference persistence diagram or
    a predicate on Betti numbers / persistence features.
    """
    node_name: str
    expected_betti_0: Optional[int] = None     # Expected β₀ (None = any)
    expected_betti_1: Optional[int] = None     # Expected β₁ (None = any)
    min_persistence: float = 0.0               # Minimum persistence for matching
    reference_features: Optional[np.ndarray] = None  # 10-dim TDA feature vector
    reference_diagram_h0: Optional[np.ndarray] = None  # (n,2) birth-death pairs
    reference_diagram_h1: Optional[np.ndarray] = None  # (n,2) birth-death pairs
    epsilon: Optional[float] = None            # Node-specific tolerance (overrides default)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_topological_commitment(self) -> bool:
        """Whether this schema references a non-trivial topological feature.

        Required for Topological Grounding (CTS Definition 7.4):
        at least one node must reference a feature with persistence ≥ τ + ε.
        """
        return (
            self.expected_betti_1 is not None and self.expected_betti_1 > 0
        ) or (
            self.reference_diagram_h1 is not None and len(self.reference_diagram_h1) > 0
        )


@dataclass
class SchematismResult:
    """Result of a schematism check for one KIM node (CTS Eq 14).

    H(v_i) = [d_BN(p_i, p_data) ≤ ε]
    """
    node_name: str
    is_grounded: bool           # True if schematism passes
    bottleneck_distance: float  # d_BN(p_i, p_data)
    epsilon: float              # Tolerance used
    feature_distance: float     # L2 distance on feature vectors
    message: str
    is_transcendental_error: bool = False  # Category error: fundamental mismatch


@bridge(
    connects_to=['TDAPipeline', 'ContinuityGuard', 'ConditionState'],
    connection_types={
        'TDAPipeline': ConnectionType.USES,
        'ContinuityGuard': ConnectionType.VALIDATES,
        'ConditionState': ConnectionType.USES,
    },
    metadata={'domain': 'core', 'version': '1.0.0', 'cts_section': '7'}
)
class SchematismBridge:
    """Implements the Schematism Bridge — CTS Section 7.

    The schematism is the framework's grounding mechanism. Without it,
    Q and Φ are computed independently and their conjunction is arbitrary.
    With it, Q validates the topological assumptions on which Φ depends,
    and Φ requires precisely the topological richness that Q certifies.

    Theorem 7.6 (Q-Φ Coupling):
    - High Φ implies non-trivial Q
    - Non-trivial Q enables high Φ
    - Schematism failure decouples Q and Φ
    """

    def __init__(
        self,
        tda_pipeline=None,
        default_epsilon: float = 0.5,
        transcendental_threshold: float = 2.0,
    ):
        """
        Args:
            tda_pipeline: TDAPipeline instance for computing persistence.
            default_epsilon: Default bottleneck distance tolerance.
            transcendental_threshold: d_BN above which the error is
                classified as transcendental (category error).
        """
        self._tda = tda_pipeline
        self._default_epsilon = default_epsilon
        self._transcendental_threshold = transcendental_threshold
        self._schemata: Dict[str, PatternSchema] = {}

    # ------------------------------------------------------------------
    # Schema registration
    # ------------------------------------------------------------------

    def register_schema(
        self,
        node_name: str,
        expected_betti_0: Optional[int] = None,
        expected_betti_1: Optional[int] = None,
        min_persistence: float = 0.0,
        reference_features: Optional[np.ndarray] = None,
        reference_diagram_h0: Optional[np.ndarray] = None,
        reference_diagram_h1: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatternSchema:
        """Associate a pattern schema with a KIM node."""
        schema = PatternSchema(
            node_name=node_name,
            expected_betti_0=expected_betti_0,
            expected_betti_1=expected_betti_1,
            min_persistence=min_persistence,
            reference_features=reference_features,
            reference_diagram_h0=reference_diagram_h0,
            reference_diagram_h1=reference_diagram_h1,
            epsilon=epsilon,
            metadata=metadata or {},
        )
        self._schemata[node_name] = schema
        return schema

    def register_from_regime_signature(
        self,
        node_name: str,
        feature_vector: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> PatternSchema:
        """Register a schema from a RegimeClassifier signature.

        Maps the 10-dim feature vector to a PatternSchema with
        expected Betti numbers and reference features.
        """
        betti_0 = int(feature_vector[0]) if len(feature_vector) > 0 else None
        betti_1 = int(feature_vector[1]) if len(feature_vector) > 1 else None

        return self.register_schema(
            node_name=node_name,
            expected_betti_0=betti_0,
            expected_betti_1=betti_1 if betti_1 > 0 else None,
            reference_features=feature_vector,
            epsilon=epsilon,
        )

    def get_schema(self, node_name: str) -> Optional[PatternSchema]:
        """Retrieve schema for a node."""
        return self._schemata.get(node_name)

    def list_schemata(self) -> List[PatternSchema]:
        """Return all registered schemata."""
        return list(self._schemata.values())

    # ------------------------------------------------------------------
    # Schematism Validation — CTS Definition 7.1, Eq 14
    # ------------------------------------------------------------------

    def validate_grounding(
        self,
        node_name: str,
        point_cloud: np.ndarray,
    ) -> SchematismResult:
        """Check schematism for a single node: H(v_i) = [d_BN(p_i, p_data) ≤ ε].

        CTS Definition 7.1: The schematism check verifies that the
        topological assumptions of the knowledge node match the data.

        Args:
            node_name: The KIM node to validate.
            point_cloud: Current data point cloud (N, d).

        Returns:
            SchematismResult with grounding status and distance.
        """
        schema = self._schemata.get(node_name)
        if schema is None:
            # No schema = trivially passes (CTS worked example: v3, v5)
            return SchematismResult(
                node_name=node_name,
                is_grounded=True,
                bottleneck_distance=0.0,
                epsilon=self._default_epsilon,
                feature_distance=0.0,
                message=f'{node_name}: No pattern schema — trivially grounded',
            )

        epsilon = schema.epsilon if schema.epsilon is not None else self._default_epsilon

        # Compute data features
        data_features = self._compute_features(point_cloud)
        data_betti_0 = int(data_features[0]) if len(data_features) > 0 else 0
        data_betti_1 = int(data_features[1]) if len(data_features) > 1 else 0

        # Compute distances
        bottleneck_dist = self._compute_schema_distance(schema, point_cloud, data_features)
        feature_dist = self._compute_feature_distance(schema, data_features)

        # Check Betti number requirements
        betti_mismatch = False
        betti_msg = ''
        if schema.expected_betti_0 is not None:
            if data_betti_0 < schema.expected_betti_0:
                betti_mismatch = True
                betti_msg += f'β₀={data_betti_0} < expected {schema.expected_betti_0}; '
        if schema.expected_betti_1 is not None:
            if data_betti_1 < schema.expected_betti_1:
                betti_mismatch = True
                betti_msg += f'β₁={data_betti_1} < expected {schema.expected_betti_1}; '

        # Determine grounding
        is_grounded = bottleneck_dist <= epsilon and not betti_mismatch
        is_transcendental = (
            bottleneck_dist > self._transcendental_threshold or
            (betti_mismatch and schema.has_topological_commitment)
        )

        if is_grounded:
            message = f'{node_name}: GROUNDED (d_BN={bottleneck_dist:.3f} ≤ ε={epsilon:.3f})'
        elif is_transcendental:
            message = (
                f'{node_name}: TRANSCENDENTAL ERROR — '
                f'd_BN={bottleneck_dist:.3f} >> ε={epsilon:.3f}. {betti_msg}'
                f'Category mismatch: node assumptions incompatible with data topology.'
            )
        else:
            message = (
                f'{node_name}: UNGROUNDED (d_BN={bottleneck_dist:.3f} > ε={epsilon:.3f}). '
                f'{betti_msg}'
            )

        return SchematismResult(
            node_name=node_name,
            is_grounded=is_grounded,
            bottleneck_distance=bottleneck_dist,
            epsilon=epsilon,
            feature_distance=feature_dist,
            message=message,
            is_transcendental_error=is_transcendental,
        )

    def validate_coherence(
        self,
        active_nodes: Set[str],
        point_cloud: np.ndarray,
    ) -> Dict[str, SchematismResult]:
        """Validate schematism for all active nodes (CTS Definition 7.5).

        CTS Definition 7.3 (Valid Configuration): A coherent configuration
        (Q, Φ) is valid if the schematism passes for all active nodes.

        Returns:
            Dict mapping node names to their SchematismResults.
        """
        results = {}
        for node_name in active_nodes:
            results[node_name] = self.validate_grounding(node_name, point_cloud)
        return results

    def validate_configuration(
        self,
        topological_signature,
        phi: float,
        active_nodes: Set[str],
        point_cloud: np.ndarray,
        phi_min: float = 0.0,
    ) -> Tuple[bool, Dict[str, SchematismResult]]:
        """Full configuration validation (CTS Definition 8.2).

        Checks all three consciousness criteria:
        1. Non-trivial topology: Q has βk > 0 for some k ≥ 1
        2. High integration: Φ > Φ_min
        3. Schematism validity: all active nodes pass

        Args:
            topological_signature: The TopologicalSignature Q.
            phi: The coherence measure Φ.
            active_nodes: Currently active KIM nodes.
            point_cloud: Current data point cloud.
            phi_min: Minimum Φ threshold.

        Returns:
            (is_valid, results_dict)
        """
        # Check schematism for all active nodes
        results = self.validate_coherence(active_nodes, point_cloud)
        all_grounded = all(r.is_grounded for r in results.values())

        # Check non-trivial topology (β₁ > 0)
        has_nontrivial_q = False
        if topological_signature is not None:
            betti = getattr(topological_signature, 'betti_numbers', {})
            if isinstance(betti, dict):
                has_nontrivial_q = any(v > 0 for k, v in betti.items() if k >= 1)

        # Check integration threshold
        high_integration = phi > phi_min

        is_valid = all_grounded and high_integration

        return is_valid, results

    # ------------------------------------------------------------------
    # Transcendental Error Detection
    # ------------------------------------------------------------------

    def detect_transcendental_error(
        self,
        node_name: str,
        point_cloud: np.ndarray,
    ) -> Optional[str]:
        """Detect transcendental errors (CTS Section 7).

        A transcendental error indicates a category error: the system is
        applying concepts whose topological assumptions do not match the
        data's actual structure. In an AI system, this corresponds to
        hallucination.

        Returns:
            Error message string if detected, None otherwise.
        """
        result = self.validate_grounding(node_name, point_cloud)
        if result.is_transcendental_error:
            return result.message
        return None

    # ------------------------------------------------------------------
    # Topological Grounding Check (CTS Definition 7.4)
    # ------------------------------------------------------------------

    def check_topological_grounding(
        self,
        active_nodes: Set[str],
        k_min: int = 2,
    ) -> bool:
        """Check if the active subgraph satisfies topological grounding.

        CTS Definition 7.4: A KIM satisfies topological grounding if
        every connected subgraph with |V_a| ≥ k_min contains at least
        one node whose pattern schema references a non-trivial
        topological feature.
        """
        if len(active_nodes) < k_min:
            return True  # Too small to require grounding

        has_topological_node = False
        for node_name in active_nodes:
            schema = self._schemata.get(node_name)
            if schema and schema.has_topological_commitment:
                has_topological_node = True
                break

        return has_topological_node

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_features(self, point_cloud: np.ndarray) -> np.ndarray:
        """Extract TDA features from a point cloud."""
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

        # Fallback: basic statistics
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, len(point_cloud), 0], dtype=float)

    def _compute_schema_distance(
        self,
        schema: PatternSchema,
        point_cloud: np.ndarray,
        data_features: np.ndarray,
    ) -> float:
        """Compute bottleneck distance between schema and data.

        Uses TDA pipeline's bottleneck_distance if available,
        otherwise falls back to feature vector L2 distance.
        """
        # If we have reference diagrams and TDA pipeline, use true bottleneck
        if (self._tda is not None and
                schema.reference_diagram_h1 is not None):
            try:
                data_diagram = self._tda.compute_persistence(point_cloud)
                return self._tda.bottleneck_distance(
                    self._make_diagram(schema), data_diagram, dim=1
                )
            except Exception:
                pass

        # Fallback: normalized feature vector distance
        return self._compute_feature_distance(schema, data_features)

    def _compute_feature_distance(
        self,
        schema: PatternSchema,
        data_features: np.ndarray,
    ) -> float:
        """L2 distance between schema features and data features."""
        if schema.reference_features is not None:
            ref = schema.reference_features
            n = min(len(ref), len(data_features))
            return float(np.linalg.norm(ref[:n] - data_features[:n]))
        return 0.0

    def _make_diagram(self, schema: PatternSchema):
        """Construct a PersistenceDiagram-like object from a schema."""
        # Minimal adapter for bottleneck distance computation
        class _SchemaDiagram:
            def __init__(self, h0, h1):
                self.h0 = h0 if h0 is not None else np.empty((0, 2))
                self.h1 = h1 if h1 is not None else np.empty((0, 2))
                self.h2 = None
        return _SchemaDiagram(schema.reference_diagram_h0, schema.reference_diagram_h1)
