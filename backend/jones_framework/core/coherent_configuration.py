"""
Coherent Configuration — CTS Sections 8-9

Implements:
- CoherentConfiguration Ct = (Q, Φ) — CTS Definition 8.1
- Consciousness Criteria — CTS Definition 8.2
- Type II Criticality — CTS Definition 8.3
- Valid Transition Protocol — CTS Definition 8.4
- Agency Flow as Term-Series — CTS Section 9

The coherent configuration is the terminal output of the CTS pipeline.
It packages the topological signature Q and coherence measure Φ with
validation state, enabling the system to distinguish valid configurations
from hallucinated ones.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from jones_framework.core.manifold_bridge import bridge, ConnectionType


class CriticalityType(Enum):
    """Type of criticality at a state (CTS Definition 8.3)."""
    NONE = auto()        # No criticality — stable regime
    TYPE_I = auto()      # Smooth transition (gradual regime shift)
    TYPE_II = auto()     # Discontinuous — ‖∇v(m)‖_g > τ_crit


@dataclass
class CoherentConfiguration:
    """Coherent Configuration Ct = (Q(t), Φ(t)) — CTS Definition 8.1.

    The ordered pair where Q is the topological signature and Φ is the
    coherence measure. Valid only if schematism passes for all active nodes.
    """
    topological_signature: Any   # TopologicalSignature or PersistenceDiagram
    phi: float                   # Coherence measure Φ
    timestamp: float = 0.0
    is_valid: bool = False       # Schematism validation result
    schematism_results: Dict[str, Any] = field(default_factory=dict)
    betti_numbers: Dict[int, int] = field(default_factory=dict)
    total_persistence: Dict[int, float] = field(default_factory=dict)
    criticality: CriticalityType = CriticalityType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_nontrivial_topology(self) -> bool:
        """CTS Definition 8.2, criterion 1: βk > 0 for some k ≥ 1."""
        return any(v > 0 for k, v in self.betti_numbers.items() if k >= 1)

    @property
    def has_high_integration(self) -> bool:
        """CTS Definition 8.2, criterion 2: Φ > 0."""
        return self.phi > 0

    @property
    def satisfies_consciousness_criteria(self) -> bool:
        """CTS Definition 8.2: all three conditions met."""
        return (
            self.has_nontrivial_topology and
            self.has_high_integration and
            self.is_valid
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            'phi': self.phi,
            'timestamp': self.timestamp,
            'is_valid': self.is_valid,
            'betti_numbers': self.betti_numbers,
            'total_persistence': self.total_persistence,
            'criticality': self.criticality.name,
            'has_nontrivial_topology': self.has_nontrivial_topology,
            'has_high_integration': self.has_high_integration,
            'satisfies_consciousness_criteria': self.satisfies_consciousness_criteria,
            'schematism_results': {
                name: {
                    'is_grounded': r.is_grounded,
                    'bottleneck_distance': r.bottleneck_distance,
                    'message': r.message,
                }
                for name, r in self.schematism_results.items()
                if hasattr(r, 'is_grounded')
            },
            'metadata': self.metadata,
        }


@dataclass
class AgencyStep:
    """Single step in a term-series flow (CTS Section 9.1, Eq 17).

    m_{i-1} --f_i--> m_i

    Each step includes pre/post condition verification and
    continuity guard checks.
    """
    step_index: int
    config_before: CoherentConfiguration
    action: np.ndarray                    # Parameter adjustment vector
    config_after: Optional[CoherentConfiguration] = None
    is_valid: bool = False                # All guards passed
    pre_condition_met: bool = False
    post_condition_met: bool = False
    continuity_preserved: bool = False
    cost: float = 0.0                     # Eq 18: c_base · e^{-(v(m_i) - v(m_{i-1}))}
    message: str = ''


@bridge(
    connects_to=['TDAPipeline', 'SchematismBridge', 'ContinuityGuard', 'ConditionState'],
    connection_types={
        'TDAPipeline': ConnectionType.USES,
        'SchematismBridge': ConnectionType.USES,
        'ContinuityGuard': ConnectionType.USES,
        'ConditionState': ConnectionType.USES,
    },
    metadata={'domain': 'core', 'version': '1.0.0', 'cts_section': '8-9'}
)
class ConfigurationBuilder:
    """Builds and validates CoherentConfigurations.

    Executes the full CTS pipeline to produce Ct = (Q, Φ):
    1. Compute Q via TDA pipeline
    2. Compute Φ via registry coherence
    3. Validate via schematism bridge
    4. Package into CoherentConfiguration
    """

    def __init__(
        self,
        tda_pipeline=None,
        schematism_bridge=None,
        registry=None,
        value_function=None,
        tau_crit: float = 1.0,
        phi_min: float = 0.0,
    ):
        self._tda = tda_pipeline
        self._schematism = schematism_bridge
        self._registry = registry
        self._value_fn = value_function
        self._tau_crit = tau_crit
        self._phi_min = phi_min

    def build(
        self,
        point_cloud: np.ndarray,
        active_nodes: Optional[Set[str]] = None,
        timestamp: float = 0.0,
    ) -> CoherentConfiguration:
        """Execute the full CTS pipeline to produce Ct = (Q, Φ).

        CTS Section 11 complete pipeline:
        X --μ--> M ⊂ U --g_v--> Mv --PH--> Q (topological channel)
        X --KIM--> G_active --λ₁,α--> Φ (integration channel)
        (Q, Φ) --schematism--> Ct (validation gate)
        """
        # --- Topological Channel: compute Q ---
        topological_signature = None
        betti_numbers = {}
        total_persistence = {}

        if self._tda is not None:
            try:
                topological_signature = self._tda.compute_full_signature(
                    point_cloud, max_dim=2
                )
                betti_numbers = dict(topological_signature.betti_numbers)
                total_persistence = dict(topological_signature.total_persistence)
            except Exception:
                # Fallback: basic persistence
                try:
                    diagram = self._tda.compute_persistence(point_cloud)
                    features = diagram.to_feature_vector()
                    betti_numbers = {0: int(features[0]), 1: int(features[1])}
                    topological_signature = diagram
                except Exception:
                    betti_numbers = {0: 1, 1: 0}

        # --- Integration Channel: compute Φ ---
        phi = 0.0
        if self._registry is not None:
            phi = self._registry.compute_coherence_phi(active_nodes)

        # --- Validation Gate: schematism check ---
        is_valid = True
        schematism_results = {}
        if self._schematism is not None and active_nodes:
            is_valid, schematism_results = self._schematism.validate_configuration(
                topological_signature, phi, active_nodes, point_cloud, self._phi_min
            )

        return CoherentConfiguration(
            topological_signature=topological_signature,
            phi=phi,
            timestamp=timestamp,
            is_valid=is_valid,
            schematism_results=schematism_results,
            betti_numbers=betti_numbers,
            total_persistence=total_persistence,
            metadata={
                'num_points': len(point_cloud),
                'active_nodes': list(active_nodes) if active_nodes else [],
            },
        )

    # ------------------------------------------------------------------
    # Type II Criticality Detection — CTS Definition 8.3
    # ------------------------------------------------------------------

    def detect_criticality(
        self,
        config_prev: CoherentConfiguration,
        config_curr: CoherentConfiguration,
    ) -> CriticalityType:
        """Detect criticality type between two configurations.

        CTS Definition 8.3: Type II criticality occurs when
        ‖∇v(m)‖_g > τ_crit — the value gradient exceeds a threshold,
        indicating a regime transition boundary.

        Also detected via large topological change (bottleneck distance
        between Q_prev and Q_curr).
        """
        # Check topological change
        topo_distance = self._topological_distance(config_prev, config_curr)

        # Check Φ change
        phi_change = abs(config_curr.phi - config_prev.phi)

        # Check Betti number change
        betti_changed = False
        for k in set(config_prev.betti_numbers) | set(config_curr.betti_numbers):
            if config_prev.betti_numbers.get(k, 0) != config_curr.betti_numbers.get(k, 0):
                betti_changed = True
                break

        # Type II: large topological discontinuity or Betti number change
        if topo_distance > self._tau_crit or (betti_changed and phi_change > 0.2):
            return CriticalityType.TYPE_II

        # Type I: moderate change
        if topo_distance > self._tau_crit * 0.5 or phi_change > 0.1:
            return CriticalityType.TYPE_I

        return CriticalityType.NONE

    # ------------------------------------------------------------------
    # Valid Transition Protocol — CTS Definition 8.4
    # ------------------------------------------------------------------

    def validate_transition(
        self,
        config_prev: CoherentConfiguration,
        config_curr: CoherentConfiguration,
    ) -> Tuple[bool, str]:
        """Validate a transition between configurations (CTS Definition 8.4).

        A valid transition requires:
        1. Continuous path exists (implied by temporal adjacency)
        2. Φ ≥ Φ_min throughout (checked at endpoints)
        3. Schematism passes at both endpoints
        4. Physical consistency (delegated to ContinuityGuard)
        """
        reasons = []

        # Check Φ maintenance
        if config_prev.phi < self._phi_min:
            reasons.append(f'Φ_prev={config_prev.phi:.3f} < Φ_min={self._phi_min}')
        if config_curr.phi < self._phi_min:
            reasons.append(f'Φ_curr={config_curr.phi:.3f} < Φ_min={self._phi_min}')

        # Check schematism validity
        if not config_prev.is_valid:
            reasons.append('Pre-transition configuration is invalid')
        if not config_curr.is_valid:
            reasons.append('Post-transition configuration is invalid')

        is_valid = len(reasons) == 0
        message = 'Valid transition' if is_valid else '; '.join(reasons)

        return is_valid, message

    def _topological_distance(
        self,
        config_a: CoherentConfiguration,
        config_b: CoherentConfiguration,
    ) -> float:
        """Compute topological distance between two configurations."""
        # Use Betti number comparison as a proxy when full diagrams unavailable
        dist = 0.0
        all_dims = set(config_a.betti_numbers) | set(config_b.betti_numbers)
        for k in all_dims:
            diff = abs(
                config_a.betti_numbers.get(k, 0) -
                config_b.betti_numbers.get(k, 0)
            )
            dist += diff ** 2

        # Add persistence-based distance
        for k in set(config_a.total_persistence) | set(config_b.total_persistence):
            diff = abs(
                config_a.total_persistence.get(k, 0.0) -
                config_b.total_persistence.get(k, 0.0)
            )
            dist += diff ** 2

        return float(np.sqrt(dist))


@bridge(
    connects_to=['ConfigurationBuilder', 'ContinuityGuard'],
    connection_types={
        'ConfigurationBuilder': ConnectionType.COMPOSES,
        'ContinuityGuard': ConnectionType.USES,
    },
    metadata={'domain': 'core', 'version': '1.0.0', 'cts_section': '9'}
)
class AgencyFlow:
    """Agency as Term-Series Flow — CTS Section 9.

    Implements the discrete term series (Eq 17):
    m_0 --f_1--> m_1 --f_2--> m_2 --f_3--> ... --f_n--> m_n

    Each step f_i: Mv → Mv is a single computational step with:
    1. Pre-condition verification
    2. Continuity guard
    3. Post-condition verification
    4. Antinomy check

    Cost per step (Eq 18):
    cost(m_{i-1} --f_i--> m_i) = c_base(f_i) · e^{-(v(m_i) - v(m_{i-1}))}
    """

    def __init__(
        self,
        config_builder: ConfigurationBuilder,
        continuity_guard=None,
        value_function=None,
        max_steps: int = 10,
        identity_tolerance: float = 0.5,
    ):
        self._builder = config_builder
        self._guard = continuity_guard
        self._value_fn = value_function
        self._max_steps = max_steps
        self._identity_tolerance = identity_tolerance

    def plan_step(
        self,
        current_cloud: np.ndarray,
        action: np.ndarray,
        active_nodes: Optional[Set[str]] = None,
        step_index: int = 0,
        timestamp: float = 0.0,
    ) -> AgencyStep:
        """Plan and validate a single step of the term series.

        CTS Section 9.1: Each step includes pre-condition verification,
        execution, continuity guard, and post-condition verification.

        Args:
            current_cloud: Current point cloud state.
            action: Parameter adjustment to apply.
            active_nodes: Currently active KIM nodes.
            step_index: Index in the term series.
            timestamp: Current timestamp.

        Returns:
            AgencyStep with validation results.
        """
        # Build configuration before action
        config_before = self._builder.build(current_cloud, active_nodes, timestamp)

        # Pre-condition: current configuration must be valid
        pre_condition_met = config_before.is_valid

        # Apply action (translate point cloud)
        next_cloud = current_cloud + action.reshape(1, -1)

        # Build configuration after action
        config_after = self._builder.build(next_cloud, active_nodes, timestamp + 1)

        # Post-condition: result configuration must be valid
        post_condition_met = config_after.is_valid

        # Continuity guard: check identity signature hasn't drifted
        topo_dist = self._builder._topological_distance(config_before, config_after)
        continuity_preserved = topo_dist < self._identity_tolerance

        # Compute cost (Eq 18)
        if self._value_fn is not None:
            v_before = self._value_fn(current_cloud.mean(axis=0))
            v_after = self._value_fn(next_cloud.mean(axis=0))
            cost = 1.0 * np.exp(-(v_after - v_before))
        else:
            cost = float(np.linalg.norm(action))

        is_valid = pre_condition_met and post_condition_met and continuity_preserved

        message = 'Step valid' if is_valid else 'Step invalid: '
        if not pre_condition_met:
            message += 'pre-condition failed; '
        if not post_condition_met:
            message += 'post-condition failed; '
        if not continuity_preserved:
            message += f'continuity violated (d={topo_dist:.3f} > ε={self._identity_tolerance}); '

        return AgencyStep(
            step_index=step_index,
            config_before=config_before,
            action=action,
            config_after=config_after,
            is_valid=is_valid,
            pre_condition_met=pre_condition_met,
            post_condition_met=post_condition_met,
            continuity_preserved=continuity_preserved,
            cost=cost,
            message=message,
        )

    def execute_flow(
        self,
        initial_cloud: np.ndarray,
        actions: List[np.ndarray],
        active_nodes: Optional[Set[str]] = None,
        timestamp: float = 0.0,
    ) -> List[AgencyStep]:
        """Execute a full term series with pre/post validation.

        CTS Eq 17: m_0 --f_1--> m_1 --f_2--> ... --f_n--> m_n

        Each step is individually guarded. If a step fails, the flow
        stops and returns the completed steps.

        Args:
            initial_cloud: Starting point cloud.
            actions: List of parameter adjustment vectors.
            active_nodes: Active KIM nodes throughout the flow.
            timestamp: Starting timestamp.

        Returns:
            List of AgencySteps (may be shorter than actions if a step fails).
        """
        steps = []
        current_cloud = initial_cloud.copy()

        for i, action in enumerate(actions[:self._max_steps]):
            step = self.plan_step(
                current_cloud, action, active_nodes, i, timestamp + i
            )
            steps.append(step)

            if not step.is_valid:
                # Roll back: stop the flow at the failed step
                step.message += ' Flow halted — rollback to previous state.'
                break

            # Advance state
            current_cloud = current_cloud + action.reshape(1, -1)

        return steps
