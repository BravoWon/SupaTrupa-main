# CTS Theory Implementation Plan

## Overview

Implement the 5 priority items from the CTS gap analysis across backend core, API, frontend types, and documentation. Each priority becomes a work phase executed in order, with tests for each new module.

---

## Phase 1: Universal Tensor Space (Priority 3 moved up — prerequisite for Schematism)

**Why first:** The U = P × T × M × F decomposition is required by the Schematism Bridge (it needs pattern schemata derived from U-space features). Building it first gives us the formal feature decomposition that downstream modules consume.

**File:** `backend/jones_framework/core/universal_tensor.py` (new)

### What to build:

1. **`TensorComponent` enum** — `PATTERN`, `TEMPORAL`, `MAGNITUDE`, `FREQUENCY`

2. **`UniversalTensorSpace` class** (bridged to `ConditionState`, `TDAPipeline`, `ShadowTensor`):
   - `__init__(self, tda_pipeline, embedding_dim=3, delay=1)`
   - `decompose(self, time_series: np.ndarray) -> Dict[TensorComponent, np.ndarray]`
     - **P (Pattern):** delegates to `TDAPipeline.extract_features()` on delay-embedded point cloud → returns feature dict as array
     - **T (Temporal):** autocorrelation coefficients (lags 1..k), dominant period via FFT peak, phase angle
     - **M (Magnitude):** mean, std, energy (sum of squares), skewness, kurtosis, min, max
     - **F (Frequency):** FFT magnitude spectrum (top-k bins), spectral centroid, bandwidth, rolloff
   - `compute_tensor_distance(self, u1, u2, weights=None) -> float` — weighted product metric (Eq 8)
   - `embed_to_manifold(self, time_series) -> np.ndarray` — time-delay embedding (delegates to existing ShadowTensorBuilder.get_metric_at)

3. **Integration:**
   - Register in `manifold_bridge.py` core components
   - Export from `core/__init__.py`
   - ShadowTensorBuilder remains as-is (backward compat), UniversalTensorSpace is the CTS-formal replacement

---

## Phase 2: Antinomy Detection (Priority 4 — needed by Coherence Φ)

**Why second:** The full Φ = λ₁ · (1 − α) formula requires antinomy load α. This enhances `manifold_bridge.py` before we build the CoherentConfiguration that consumes Φ.

**File:** Enhance `backend/jones_framework/core/manifold_bridge.py`

### What to build:

1. **Add `INHIBITS` to `ConnectionType` enum** (already has 11 types but no explicit INHIBITS — check if present, add if missing)

2. **Add to `ComponentRegistry`:**
   - `add_antinomy(self, comp_a: str, comp_b: str)` — register mutual exclusion between two nodes
   - `_antinomies: List[Tuple[str, str]]` — internal storage
   - `compute_antinomy_load(self, active_nodes: Set[str]) -> float` — α(t) = count of active antinomy pairs / total antinomy pairs
   - `compute_coherence_phi(self, active_nodes: Set[str]) -> float` — Φ = λ₁ · (1 − α), where λ₁ is the Fiedler value from existing spectral embedding

3. **Compute λ₁ (Fiedler value):**
   - The existing `_f1I1cOf` computes the full Laplacian eigenvectors but doesn't expose the eigenvalues
   - Add `_fiedler_value` property that extracts the second-smallest eigenvalue from the graph Laplacian (store during `_f1I1cOf`)

---

## Phase 3: Schematism Bridge (Priority 1 — the central CTS innovation)

**File:** `backend/jones_framework/core/schematism.py` (new)

### What to build:

1. **`PatternSchema` dataclass:**
   - `node_name: str` — KIM node this schema belongs to
   - `expected_diagram: PersistenceDiagram` — reference persistence diagram
   - `epsilon: float` — bottleneck distance tolerance
   - `metadata: Dict[str, Any]`

2. **`SchematismResult` dataclass:**
   - `node_name: str`
   - `is_grounded: bool` — d_BN(p_i, p_data) ≤ ε
   - `bottleneck_distance: float`
   - `epsilon: float`
   - `message: str`

3. **`SchematismBridge` class** (bridged to `TDAPipeline`, `ContinuityGuard`, `ConditionState`):
   - `__init__(self, tda_pipeline: TDAPipeline, default_epsilon: float = 0.5)`
   - `register_schema(self, node_name, reference_diagram, epsilon=None)` — associate pattern schema with KIM node
   - `validate_grounding(self, node_name, point_cloud) -> SchematismResult` — Def 7.4: H(v_i) = [d_BN(p_i, p_data) ≤ ε]
   - `validate_coherence(self, active_schemas, point_cloud) -> Dict[str, SchematismResult]` — Def 7.5: check all active nodes
   - `validate_configuration(self, Q, Phi, active_schemas, point_cloud) -> bool` — Def 7.3: non-trivial Q AND Φ > threshold AND all schemata grounded
   - `detect_transcendental_error(self, node_name, point_cloud) -> Optional[str]` — flag when schema fundamentally mismatches data topology
   - `_schemata: Dict[str, PatternSchema]` — internal storage

4. **Pre-register drilling regime schemata** — use the 12 existing regime signatures from `RegimeClassifier` as initial pattern schemata (their `reference_diagram` fields map directly)

---

## Phase 4: Coherent Configuration (Priority 2)

**File:** `backend/jones_framework/core/coherent_configuration.py` (new)

### What to build:

1. **`CoherentConfiguration` dataclass:**
   - `Q: TopologicalSignature` — topological channel output
   - `Phi: float` — coherence measure (from manifold_bridge)
   - `timestamp: float`
   - `is_valid: bool` — passed schematism validation
   - `schematism_results: Dict[str, SchematismResult]`
   - `metadata: Dict[str, Any]`

2. **`CriticalityType` enum** — `NONE`, `TYPE_I` (smooth), `TYPE_II` (discontinuous, ‖∇v‖_g > τ_crit)

3. **`ConfigurationBuilder` class** (bridged to `TDAPipeline`, `SchematismBridge`, `ContinuityGuard`, `ConditionState`):
   - `__init__(self, tda_pipeline, schematism_bridge, registry, value_function=None)`
   - `build(self, point_cloud, active_nodes=None) -> CoherentConfiguration` — execute full CTS pipeline:
     1. Compute Q via `tda_pipeline.compute_full_signature(point_cloud)`
     2. Compute Φ via `registry.compute_coherence_phi(active_nodes)`
     3. Validate via `schematism_bridge.validate_configuration(Q, Phi, ...)`
     4. Package into CoherentConfiguration
   - `detect_criticality(self, config_prev, config_curr, value_function=None) -> CriticalityType` — Def 8.3: Type II if ‖∇v‖ > τ_crit or bottleneck distance between Q_prev and Q_curr exceeds threshold
   - `validate_transition(self, config_prev, config_curr) -> bool` — Def 8.4: continuous transition check using ContinuityGuard + topological distance

4. **`AgencyStep` dataclass:**
   - `config_before: CoherentConfiguration`
   - `action: np.ndarray` — parameter adjustment
   - `config_after: CoherentConfiguration`
   - `is_valid: bool`

5. **`AgencyFlow` class** (bridged to `ConfigurationBuilder`, `ContinuityGuard`):
   - `__init__(self, config_builder, continuity_guard)`
   - `plan_step(self, current_config, target_state) -> AgencyStep` — compute one step of the term-series
   - `execute_flow(self, initial_cloud, target_state, max_steps=10) -> List[AgencyStep]` — full term-series m₀ → f₁ → m₁ → ... → mₙ with pre/post validation at each step

---

## Phase 5: API Endpoints & TypeScript Types

### Backend API (`backend/jones_framework/api/server.py`)

Add these endpoints:

```
POST /api/v1/cts/decompose          — UniversalTensorSpace.decompose()
POST /api/v1/cts/tensor-distance    — UniversalTensorSpace.compute_tensor_distance()
POST /api/v1/cts/validate-schema    — SchematismBridge.validate_grounding()
POST /api/v1/cts/coherent-config    — ConfigurationBuilder.build()
POST /api/v1/cts/detect-criticality — ConfigurationBuilder.detect_criticality()
POST /api/v1/cts/agency-step        — AgencyFlow.plan_step()
GET  /api/v1/cts/coherence-phi      — registry.compute_coherence_phi()
GET  /api/v1/cts/schemata           — list registered schemata
```

### TypeScript Types (`shared/types/index.ts`)

Add interfaces:
- `TensorDecomposition` — { pattern, temporal, magnitude, frequency }
- `SchematismResult` — { node_name, is_grounded, bottleneck_distance, epsilon }
- `CoherentConfiguration` — { Q (signature), Phi, is_valid, schematism_results, timestamp }
- `CriticalityType` — enum
- `AgencyStep` — { config_before, action, config_after, is_valid }

---

## Phase 6: Axiom Restructuring (Priority 5 — documentation)

**File:** Update `EPISTEMIC_ENGINE.md`

### What to change:

1. Expand Section 1 from G1-G5 to A1-A11 axioms with formal statements
2. Add new Section "Formal Definitions (D1-D17)" with mathematical definitions
3. Add new Section "Theorems (T1-T10)" with theorem statements
4. Add new Section "Design Principles (P1-P21)" formalizing the existing informal principles
5. Add new Section "Heuristics (H1-H8)" documenting empirical rules
6. Add mapping table from old G1-G5 IDs to new A-IDs
7. Reference the 4 new CTS modules in the Integration section

---

## Phase 7: Tests

**File:** `backend/tests/test_cts_integration.py` (new)

### Test classes:

1. `TestUniversalTensorSpace` — decomposition produces all 4 components, distance metric is symmetric/non-negative, empty input handling
2. `TestAntinomyDetection` — antinomy registration, load computation, full Φ formula
3. `TestSchematismBridge` — schema registration, grounding validation (pass/fail), transcendental error detection
4. `TestCoherentConfiguration` — full pipeline build, criticality detection, transition validation
5. `TestAgencyFlow` — single step planning, multi-step flow with pre/post guards

---

## Component Registration Summary

All new classes will be registered in the manifold via `@bridge`:

| New Component | Connects To | Connection Type |
|---|---|---|
| `UniversalTensorSpace` | `ConditionState`, `TDAPipeline`, `ShadowTensor` | TRANSFORMS, USES, EXTENDS |
| `SchematismBridge` | `TDAPipeline`, `ContinuityGuard`, `ConditionState` | VALIDATES, USES, USES |
| `CoherentConfiguration` | `TDAPipeline`, `SchematismBridge` | PRODUCES, USES |
| `ConfigurationBuilder` | `TDAPipeline`, `SchematismBridge`, `ContinuityGuard` | COMPOSES |
| `AgencyFlow` | `ConfigurationBuilder`, `ContinuityGuard` | COMPOSES, USES |

---

## File Change Summary

| File | Action |
|---|---|
| `backend/jones_framework/core/universal_tensor.py` | CREATE |
| `backend/jones_framework/core/schematism.py` | CREATE |
| `backend/jones_framework/core/coherent_configuration.py` | CREATE |
| `backend/jones_framework/core/manifold_bridge.py` | EDIT (antinomy + Fiedler) |
| `backend/jones_framework/core/__init__.py` | EDIT (add exports) |
| `backend/jones_framework/api/server.py` | EDIT (add CTS endpoints) |
| `shared/types/index.ts` | EDIT (add CTS types) |
| `EPISTEMIC_ENGINE.md` | EDIT (restructure axioms) |
| `backend/tests/test_cts_integration.py` | CREATE |
