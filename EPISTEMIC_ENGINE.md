# Analysis Framework Reference (v3.0)

## The Jones Framework

> *"The system does not assume a pre-existing model; it constructs a state representation from raw input data before performing analysis."*

---

## Design Rationale & Core Directive

The Analysis Framework is a **data-driven processing pipeline** governed by the Jones Framework. This document defines the analytical architecture that underlies system behavior.

### Context-Free Initialization

- **No Implicit Defaults:** Rather than starting from a fixed model (point A to point B), the system constructs a state space from the input data, identifies relationships, and computes gradients.
- **Core Directive:** Do not assume a default operating context. The system must construct a state representation at the start of every analysis session. Assuming a fixed starting state may produce misleading results.

---

## 1. Design Principles

The system is governed by the following design principles:

| Principle | Name | Definition |
|-----------|------|------------|
| **G1** | Input Classification | Raw input is unstructured signal. Apply a classification scheme to produce structured observations. |
| **G2** | Dynamic Weighting | Relevance is context-dependent, not fixed. Operational objectives change the relative importance of parameters. |
| **G3** | Optimal Path Computation | There may not be a single correct output; the system computes the lowest-cost path through the parameter space given current constraints. |
| **G4** | State Continuity | Outputs must be consistent with prior state. The system should not produce discontinuous state transitions without supporting evidence. |
| **G5** | Cross-Domain Applicability | The same analytical methods can apply across domains. Use structural analogies between different operational contexts. |

### Principle-to-Implementation Mapping

| Principle | Implementation | Location |
|-----------|----------------|----------|
| G1 | `ConditionState` construction from raw telemetry | `core/condition_state.py` |
| G2 | Configurable metric tensors in `ActivityState` | `core/activity_state.py` |
| G3 | TDA-based path computation via persistent homology | `perception/tda_pipeline.py` |
| G4 | `ContinuityGuard` validates state transition consistency | `sans/continuity_guard.py` |
| G5 | Domain adapters share core tensor operations | `domains/*.py` |

---

## 2. Processing Pipeline

All input processing follows three phases:

### Phase I: State Construction

```
Raw Input → Normalization → Tensor Decomposition → State Space Model
```

1. **Normalization:** Standardize input to remove format-specific artifacts
2. **Tensor Decomposition:** Decompose signal into `P x T x M x F` (Pattern, Time, Magnitude, Frequency)
3. **State Space Construction:** Given feature coordinates, construct a state space model with associated topology
4. **Validation:** Verify that the data supports the constructed state space model

**Implementation:** `ShadowTensor.from_timeseries()` → `TDAPipeline.extract_features()`

### Phase II: Regime Classification & Path Computation

```
Regime Detection → Metric Selection → Optimization → Trajectory Output
```

1. **Detect the Regime:** Classify the current operating state (Stable, Unstable, Transitioning)
2. **Select the Metric:** Apply the appropriate parameter weighting for the detected regime
3. **Compute Path:** Calculate the parameter trajectory that minimizes a cost function

**Implementation:** `RegimeClassifier.classify()` → `MixtureOfExperts.select_expert()`

### Phase III: Output Generation

```
NLP Processing → Output Format Selection → Response Construction → Validation
```

1. **NLP Processing:** Select output language and detail level appropriate to the detected regime
2. **Error Handling:** Treat error signals as valid feedback; re-run state construction if needed

**Implementation:** `LinguisticArbitrageEngine` → `SentimentVectorPipeline`

---

## 3. Operating Modes

The system selects operating mode based on input characteristics:

### Mode A: Exploration (State Construction)

- **Trigger:** Ambiguous data or insufficient context
- **Action:** Construct a state representation from available data
- **Output:** "State space defined by [Variables]..."
- **Implementation:** Full TDA pipeline with regime detection

### Mode B: Navigation (Path Computation)

- **Trigger:** User provides a target state or constraint
- **Action:** Compute the parameter path to the target
- **Output:** "Suggested path from [State A] to [State B]. High sensitivity detected at..."
- **Implementation:** Expert-guided inference with LoRA adaptation

### Mode C: Validation (Continuity Check)

- **Trigger:** Proposed state transition appears discontinuous
- **Action:** Validate transition feasibility
- **Output:** "Transition from A to B requires intermediate states. Direct transition is not supported by the data."
- **Implementation:** `ContinuityGuard.validate_transition()`

---

## 4. Output Format

Structured response format:

```
1. State Model
   "Current state space defined as..."

2. Objective Gradient
   "The operational objective implies movement toward..."

3. Suggested Path
   Step 1: [Observation] → [Parameter Mapping]
   Step 2: [Assessment] → [Recommended Action]

4. Summary
   Consolidated recommendation.
```

---

## 5. Regime-Specific Behavior

| Regime | Betti Signature | Parameter Weighting | Expert Behavior |
|--------|-----------------|---------------------|-----------------|
| STABLE | β₀=1, β₁=0 | Uniform (linear) | Conservative, trend-following |
| MOMENTUM | β₀=1, β₁≥1 | Directionally weighted | Trend-aligned |
| MEAN_REVERTING | β₀≥2, β₁=0 | Oscillatory | Range-bound, mean-seeking |
| HIGH_VOLATILITY | β₀≥1, β₁≥2 | High-sensitivity | Defensive, variance-aware |
| TRANSITION | Rapidly changing | Non-stationary | Cautious, regime-detecting |
| CRISIS | β₀≥3, β₁≥3 | Extreme-sensitivity | Emergency protocols |

---

## 6. Cross-Domain Applicability (Principle G5)

The same topological analysis methods are designed to apply across domains:

| Domain | State Space | Observables | Regimes |
|--------|-------------|-------------|---------|
| **Drilling** | BHA dynamics | WOB, RPM, Torque | Stable/Whirl/Stick-Slip |
| **Finance** | Price data | OHLCV, Greeks | Bull/Bear/Volatile |
| **Climate** | Atmospheric state | Temp, Pressure, Humidity | Normal/Storm/Transition |
| **Healthcare** | Physiological state | Vitals, Biomarkers | Healthy/Acute/Chronic |

The `@bridge` decorator supports this by requiring all components to connect to the core data model.

---

## 7. Failure Modes & Recovery

### State Continuity Violations

If `ContinuityGuard` detects a discontinuous transition:
1. Log the violation with full state context
2. Attempt gradient-based recovery (interpolate intermediate states)
3. If recovery fails, trigger regime re-detection
4. If re-detection fails, escalate to operator review

### Invalid State Prevention (Principle G4)

The system guards against invalid state jumps through:
- Persistent homology tracking of state evolution
- Minimum path length requirements between states
- Confidence thresholds for regime transitions

---

## 8. Integration with Codebase

This framework document maps to the implementation as follows:

```
EPISTEMIC_ENGINE.md (this file)
    ↓
CLAUDE.md (development guidelines)
    ↓
backend/jones_framework/
    ├── core/           # G1, G2: State primitives
    ├── perception/     # G3: Path computation
    ├── sans/           # G4, G5: Expert routing
    └── domains/        # G5: Cross-domain adapters
```

**Fallback:** If the constructed state model does not fit the incoming data, the system re-runs state construction from the current input window.
