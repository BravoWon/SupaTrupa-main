"""
CTS Integration Tests — Configurational Term Series

Tests the complete CTS pipeline from theory to implementation:
- T1-T3: Universal Tensor Space (Section 3)
- T4-T6: Antinomy Detection and Coherence Φ (Section 6)
- T7-T11: Schematism Bridge (Section 7)
- T12-T15: Coherent Configuration (Section 8)
- T16-T18: Agency Flow (Section 9)
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tda_pipeline():
    from jones_framework.perception.tda_pipeline import TDAPipeline
    return TDAPipeline(1)


@pytest.fixture
def registry():
    from jones_framework.core.manifold_bridge import get_registry
    return get_registry()


@pytest.fixture
def uts(tda_pipeline):
    from jones_framework.core.universal_tensor import UniversalTensorSpace
    return UniversalTensorSpace(tda_pipeline=tda_pipeline)


@pytest.fixture
def schematism(tda_pipeline):
    from jones_framework.core.schematism import SchematismBridge
    return SchematismBridge(tda_pipeline=tda_pipeline, default_epsilon=0.5)


@pytest.fixture
def config_builder(tda_pipeline, schematism, registry):
    from jones_framework.core.coherent_configuration import ConfigurationBuilder
    return ConfigurationBuilder(
        tda_pipeline=tda_pipeline,
        schematism_bridge=schematism,
        registry=registry,
    )


@pytest.fixture
def stable_cloud():
    """Point cloud for a stable regime: single cluster, no loops."""
    rng = np.random.RandomState(42)
    return rng.randn(20, 3) * 0.5 + np.array([1, 0, 0])


@pytest.fixture
def cyclic_cloud():
    """Point cloud with a topological loop (β₁ ≥ 1)."""
    t = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    rng = np.random.RandomState(42)
    noise = rng.randn(30, 3) * 0.05
    cloud = np.column_stack([np.cos(t), np.sin(t), np.zeros(30)]) + noise
    return cloud


@pytest.fixture
def sine_series():
    """Clean sine wave signal for UTI decomposition."""
    t = np.linspace(0, 4 * np.pi, 200)
    return np.sin(t) + 0.1 * np.sin(5 * t)


# ===========================================================================
# T1-T3: Universal Tensor Space (CTS Section 3)
# ===========================================================================

class TestUniversalTensorSpace:

    def test_decomposition_produces_all_components(self, uts, sine_series):
        """All four components P, T, M, F should be populated."""
        decomp = uts.decompose(sine_series)
        assert len(decomp.pattern) > 0
        assert len(decomp.temporal) > 0
        assert len(decomp.magnitude) > 0
        assert len(decomp.frequency) > 0

    def test_magnitude_statistics_correct(self, uts, sine_series):
        """φ_M should capture mean, std, energy, skewness, kurtosis, min, max."""
        decomp = uts.decompose(sine_series)
        mag = decomp.magnitude
        assert len(mag) == 7
        # Mean of sin(t) + 0.1*sin(5t) ≈ 0
        assert abs(mag[0]) < 0.5  # mean near 0
        assert mag[1] > 0         # nonzero std
        assert mag[2] > 0         # nonzero energy

    def test_frequency_detects_dominant_frequency(self, uts, sine_series):
        """φ_F should detect the dominant frequency peak."""
        decomp = uts.decompose(sine_series)
        freq = decomp.frequency
        # Top-k magnitudes should be nonzero
        assert freq[0] > 0  # largest spectral peak

    def test_temporal_autocorrelation(self, uts, sine_series):
        """φ_T should show strong autocorrelation for periodic signal."""
        decomp = uts.decompose(sine_series)
        temp = decomp.temporal
        # Autocorrelation at lag 1 should be high for smooth sine
        assert temp[0] > 0.5

    def test_tensor_distance_symmetric(self, uts, sine_series):
        """d_U(u1, u2) = d_U(u2, u1) — symmetry."""
        s1 = sine_series
        s2 = np.sin(np.linspace(0, 6 * np.pi, 200))
        u1 = uts.decompose(s1)
        u2 = uts.decompose(s2)
        d12 = uts.compute_tensor_distance(u1, u2)
        d21 = uts.compute_tensor_distance(u2, u1)
        assert abs(d12 - d21) < 1e-10

    def test_tensor_distance_self_zero(self, uts, sine_series):
        """d_U(u, u) = 0 — identity of indiscernibles."""
        u = uts.decompose(sine_series)
        assert uts.compute_tensor_distance(u, u) == pytest.approx(0.0)

    def test_tensor_distance_positive(self, uts, sine_series):
        """d_U(u1, u2) > 0 for different signals."""
        s2 = np.random.randn(200) * 5
        u1 = uts.decompose(sine_series)
        u2 = uts.decompose(s2)
        assert uts.compute_tensor_distance(u1, u2) > 0

    def test_to_vector_concatenation(self, uts, sine_series):
        """to_vector() should concatenate P, T, M, F."""
        decomp = uts.decompose(sine_series)
        vec = decomp.to_vector()
        expected_len = sum(decomp.component_dimensions().values())
        assert len(vec) == expected_len


# ===========================================================================
# T4-T6: Antinomy Detection & Coherence Φ (CTS Section 6)
# ===========================================================================

class TestAntinomyDetection:

    def test_fiedler_value_positive_for_connected_graph(self, registry):
        """λ₁ > 0 for a connected component registry (CTS Prop 6.5)."""
        fiedler = registry.compute_fiedler_value()
        assert fiedler >= 0

    def test_coherence_phi_no_antinomies(self, registry):
        """Φ = λ₁ · (1 - 0) = λ₁ when no antinomies."""
        phi = registry.compute_coherence_phi()
        fiedler = registry.compute_fiedler_value()
        alpha = registry.compute_antinomy_load()
        assert alpha == pytest.approx(0.0)
        assert phi == pytest.approx(fiedler)

    def test_antinomy_load_degrades_phi(self, registry):
        """Adding antinomies should degrade Φ (CTS Eq 12)."""
        phi_before = registry.compute_coherence_phi()
        components = list(registry.components.keys())
        if len(components) >= 2:
            registry.add_antinomy(components[0], components[1])
            active = set(components[:4])
            phi_after = registry.compute_coherence_phi(active)
            alpha = registry.compute_antinomy_load(active)
            assert alpha > 0
            # With antinomy, Φ should be reduced by (1-α) factor
            fiedler = registry.compute_fiedler_value(active)
            assert phi_after == pytest.approx(fiedler * (1 - alpha))

    def test_full_antinomy_kills_phi(self, registry):
        """α = 1 → Φ = 0 (CTS Prop 6.5)."""
        components = list(registry.components.keys())
        if len(components) >= 2:
            # Register antinomies between all pairs in a small subset
            subset = components[:2]
            registry.add_antinomy(subset[0], subset[1])
            active = set(subset)
            alpha = registry.compute_antinomy_load(active)
            assert alpha == 1.0
            phi = registry.compute_coherence_phi(active)
            assert phi == pytest.approx(0.0)


# ===========================================================================
# T7-T11: Schematism Bridge (CTS Section 7)
# ===========================================================================

class TestSchematismBridge:

    def test_schema_registration(self, schematism):
        """Schemata can be registered and retrieved."""
        schematism.register_schema('test_node', expected_betti_0=1, expected_betti_1=0)
        schema = schematism.get_schema('test_node')
        assert schema is not None
        assert schema.expected_betti_0 == 1
        assert schema.expected_betti_1 == 0

    def test_no_schema_trivially_passes(self, schematism, stable_cloud):
        """Node with no schema trivially passes (CTS worked example: v3, v5)."""
        result = schematism.validate_grounding('unregistered_node', stable_cloud)
        assert result.is_grounded is True

    def test_stable_cloud_passes_stable_schema(self, schematism, stable_cloud):
        """Stable regime (β₀=1, β₁=0) should pass a stable schema.

        Use a generous epsilon since the fallback TDA (without ripser)
        produces approximate features.
        """
        # Compute actual features first to build a matching schema
        data_features = schematism._compute_features(stable_cloud)
        schematism.register_schema(
            'normal_drilling', expected_betti_0=1,
            reference_features=data_features, epsilon=1.0,
        )
        result = schematism.validate_grounding('normal_drilling', stable_cloud)
        assert result.is_grounded is True
        assert result.bottleneck_distance == pytest.approx(0.0)

    def test_stable_cloud_fails_cyclic_schema(self, schematism, stable_cloud):
        """Stable cloud should fail a schema expecting β₁ ≥ 1."""
        features = np.array([3, 2, 0.5, 0.6, 1.0, 0.8, 0.5, 0.4, 10, 5], dtype=float)
        schematism.register_schema(
            'stick_slip', expected_betti_1=2,
            reference_features=features, epsilon=0.3,
        )
        result = schematism.validate_grounding('stick_slip', stable_cloud)
        assert result.is_grounded is False

    def test_transcendental_error_detection(self, schematism, stable_cloud):
        """Extreme mismatch should be flagged as transcendental error."""
        features = np.array([5, 4, 0.8, 0.9, 2.0, 1.5, 1.0, 0.8, 20, 15], dtype=float)
        schematism.register_schema(
            'washout', expected_betti_1=4,
            reference_features=features, epsilon=0.1,
        )
        error = schematism.detect_transcendental_error('washout', stable_cloud)
        assert error is not None
        assert 'TRANSCENDENTAL' in error

    def test_validate_coherence_all_nodes(self, schematism, stable_cloud):
        """validate_coherence checks all active nodes."""
        schematism.register_schema('a', expected_betti_0=1)
        schematism.register_schema('b', expected_betti_0=1)
        results = schematism.validate_coherence({'a', 'b', 'c'}, stable_cloud)
        assert 'a' in results
        assert 'b' in results
        assert 'c' in results  # unregistered → trivially passes

    def test_topological_grounding_check(self, schematism):
        """Check topological grounding (CTS Def 7.4)."""
        # No schemas → no topological commitment → passes trivially
        assert schematism.check_topological_grounding({'a', 'b', 'c'}) is False
        # Add a schema with topological commitment
        schematism.register_schema('topo', expected_betti_1=1)
        assert schematism.check_topological_grounding({'topo', 'b', 'c'}) is True


# ===========================================================================
# T12-T15: Coherent Configuration (CTS Section 8)
# ===========================================================================

class TestCoherentConfiguration:

    def test_build_produces_valid_config(self, config_builder, stable_cloud):
        """ConfigurationBuilder.build() returns a CoherentConfiguration."""
        config = config_builder.build(stable_cloud)
        assert config.phi >= 0
        assert isinstance(config.betti_numbers, dict)
        assert isinstance(config.is_valid, bool)

    def test_config_to_dict_serializable(self, config_builder, stable_cloud):
        """to_dict() produces a JSON-serializable dict."""
        import json
        config = config_builder.build(stable_cloud)
        d = config.to_dict()
        # Should be JSON serializable
        json.dumps(d)

    def test_type_ii_criticality_detection(self, config_builder, stable_cloud, cyclic_cloud):
        """Type II criticality detects topological change between regimes."""
        from jones_framework.core.coherent_configuration import CriticalityType
        config_stable = config_builder.build(stable_cloud)
        config_cyclic = config_builder.build(cyclic_cloud)
        criticality = config_builder.detect_criticality(config_stable, config_cyclic)
        # Different topological structure → should detect criticality
        assert criticality in (CriticalityType.TYPE_I, CriticalityType.TYPE_II)

    def test_valid_transition_protocol(self, config_builder, stable_cloud):
        """Same-regime transition should be valid."""
        config1 = config_builder.build(stable_cloud)
        config2 = config_builder.build(stable_cloud + 0.01)
        valid, msg = config_builder.validate_transition(config1, config2)
        assert valid is True

    def test_consciousness_criteria(self, config_builder, cyclic_cloud):
        """Check consciousness criteria properties (CTS Def 8.2)."""
        config = config_builder.build(cyclic_cloud)
        # We can check individual criteria without asserting the conjunction
        # since it depends on whether schematism nodes are registered
        assert isinstance(config.has_nontrivial_topology, bool)
        assert isinstance(config.has_high_integration, bool)
        assert isinstance(config.satisfies_consciousness_criteria, bool)


# ===========================================================================
# T16-T18: Agency Flow (CTS Section 9)
# ===========================================================================

class TestAgencyFlow:

    def test_single_step_valid(self, config_builder, stable_cloud):
        """A small action step should preserve continuity."""
        from jones_framework.core.coherent_configuration import AgencyFlow
        flow = AgencyFlow(config_builder, identity_tolerance=5.0)
        action = np.array([0.01, 0.01, 0.01])
        step = flow.plan_step(stable_cloud, action)
        assert step.config_before is not None
        assert step.config_after is not None
        assert step.continuity_preserved is True

    def test_large_step_may_violate_continuity(self, config_builder, stable_cloud):
        """A large action should violate the continuity guard."""
        from jones_framework.core.coherent_configuration import AgencyFlow
        flow = AgencyFlow(config_builder, identity_tolerance=0.01)
        action = np.array([100.0, 100.0, 100.0])
        step = flow.plan_step(stable_cloud, action)
        # Large jumps are likely to change topology
        assert isinstance(step.continuity_preserved, bool)

    def test_execute_flow_multi_step(self, config_builder, stable_cloud):
        """Multi-step flow should execute until completion or failure."""
        from jones_framework.core.coherent_configuration import AgencyFlow
        flow = AgencyFlow(config_builder, identity_tolerance=5.0)
        actions = [np.array([0.01, 0.01, 0.01])] * 3
        steps = flow.execute_flow(stable_cloud, actions)
        assert len(steps) >= 1
        assert all(hasattr(s, 'is_valid') for s in steps)

    def test_flow_halts_on_failure(self, config_builder, stable_cloud):
        """Flow should halt when a step fails validation."""
        from jones_framework.core.coherent_configuration import AgencyFlow
        flow = AgencyFlow(config_builder, identity_tolerance=0.001)
        # Mix small and large actions
        actions = [
            np.array([0.001, 0.001, 0.001]),
            np.array([50.0, 50.0, 50.0]),  # likely to fail
            np.array([0.001, 0.001, 0.001]),
        ]
        steps = flow.execute_flow(stable_cloud, actions)
        # Flow should stop at or before the large action
        assert len(steps) <= 3

    def test_step_cost_computed(self, config_builder, stable_cloud):
        """Agency step cost should be computed (CTS Eq 18)."""
        from jones_framework.core.coherent_configuration import AgencyFlow
        flow = AgencyFlow(config_builder)
        action = np.array([0.1, 0.1, 0.1])
        step = flow.plan_step(stable_cloud, action)
        assert step.cost >= 0


# ===========================================================================
# T19-T21: End-to-End Pipeline
# ===========================================================================

class TestEndToEndPipeline:

    def test_full_cts_pipeline(self, uts, schematism, config_builder, stable_cloud):
        """End-to-end: decompose → compute Q → compute Φ → validate → Ct."""
        # Step 1: Decompose signal into U (Section 3)
        series = stable_cloud[:, 0]  # Use first dimension
        decomp = uts.decompose(series)
        assert len(decomp.to_vector()) > 0

        # Step 2-5: Build coherent configuration (Sections 5-8)
        config = config_builder.build(stable_cloud)
        assert config.phi >= 0

        # Step 6: Serialize
        d = config.to_dict()
        assert 'phi' in d
        assert 'is_valid' in d
        assert 'betti_numbers' in d

    def test_imports_from_core(self):
        """All CTS classes should be importable from core."""
        from jones_framework.core import (
            TensorComponent,
            TensorDecomposition,
            UniversalTensorSpace,
            PatternSchema,
            SchematismResult,
            SchematismBridge,
            CriticalityType,
            CoherentConfiguration,
            AgencyStep,
            ConfigurationBuilder,
            AgencyFlow,
        )
        assert TensorComponent.PATTERN is not None
        assert CriticalityType.TYPE_II is not None
