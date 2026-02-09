# Iterative Upgrade Cycle Manifest

> **Platform:** Unified Activity:State (SupaTrupa)
> **Framework:** Jones Axiomatic (G1-G5)
> **Process:** Manifest-driven progressive surfacing of computed capabilities

---

## Development Phase Summary

| Field | Value |
|-------|-------|
| **Status** | DEVELOPMENT COMPLETE |
| **Date** | 2026-02-09 |
| **Cycles Delivered** | 0-11 (all 12 cycles) |
| **API Endpoints** | 45 (34 POST, 9 GET, 1 WS, 1 multipart) |
| **UI Tabs** | 12 (DASHBOARD, CTS, WELL PATH, WIRE MESH, NETWORK, SENSITIVITY, SIGNATURE, DYNAMICS, FORECAST, ADVISORY, FIELD MAP, ANALYZER) |
| **Handoff** | `docs/TEST_PLAN.md` created for test team |

**Polish pass completed:**
- Dashboard crash fixes (TopologyForecaster args, ShadowTensorBuilder obfuscated names)
- ~70 TypeScript errors fixed (unused imports/vars, R3F args, shadcn rewrites, deobfuscation)
- 15 orphaned components deleted (BettiPanel, DepthFormationBar, LiveMonitor, ManusDialog, MissionManager, PortalStatus, StateSpaceViz, TopologicalMapper, TraderView, RSSSimulator, RecommendationPanel, ParametersPanel, PersistenceControls, RegimePanel, VoiceCommand)
- tsconfig.node.json fix (`composite: true` + `declaration: true`)
- 10 bare `except Exception: pass` replaced with `logger.debug()` + `exc_info=True`
- 15 silent `.catch(() => {})` in Home.tsx replaced with `console.warn`
- vite-env.d.ts added for `import.meta.env` typing
- const.ts fixed (removed broken `@shared/const` import)

**Build status:** All green (`tsc --noEmit` passes, `pnpm build` succeeds, all consumer test endpoints return 200).

---

## Insight Level Progression

```
Level 1: "What IS the state?"           -> Cycles 0-1 (DONE)
Level 2: "How does the state EVOLVE?"   -> Cycles 2-3
Level 3: "WHY is the state what it is?" -> Cycles 4, 6
Level 4: "What WILL the state be?"      -> Cycle 5
Level 5: "What SHOULD we do?"           -> Cycle 7
Level 6: "What do ALL wells tell us?"   -> Cycle 8
Level 7: "How do I USE it operationally?" -> Cycles 9-11
```

---

## Cycle Process (Repeatable)

Each cycle follows this loop:

```
1. SCOPE     - Read cycle definition, confirm scope
2. BACKEND   - Create/modify perception engines + API endpoints
3. FRONTEND  - Create visualization components + utility libs
4. INTEGRATE - Wire into Home.tsx tabs/panels
5. VERIFY    - Backend curl tests + pnpm build + visual check
6. RECORD    - Update CYCLE_MANIFEST.md status + MEMORY.md
```

---

## Dependency Graph

```
Cycle 0 (Foundation) --- DONE
  |
  +-- Cycle 1 (PRN) --- DONE
  |     |
  |     +-- Cycle 2 (Topological Time Machine) -- Level 2
  |     |     |
  |     |     +-- Cycle 4 (Persistence Fingerprinting) -- Level 3
  |     |     |     |
  |     |     |     +-- Cycle 5 (Predictive Topology) -- Level 4
  |     |     |     |     |
  |     |     |     |     +-- Cycle 7 (Autonomous Advisory) -- Level 5
  |     |     |     |     |
  |     |     |     |     +-- Cycle 8 (Field Intelligence) -- Level 6
  |     |     |     |
  |     |     |     +-- Cycle 8 (also needs Cycle 4)
  |     |     |
  |     |     +-- Cycle 6 (Shadow Tensor) -- Level 3
  |     |
  |     +-- Cycle 7 (also needs Cycle 1 correlations)
  |
  +-- Cycle 3 (Manifold Geometry) -- Level 2
        |
        +-- Cycle 7 (also needs Cycle 3 geodesics)
  |
  +-- Cycle 9 (LAS Sliding Analyzer) -- Level 7
  |     |
  |     +-- Cycle 10 (Master Dashboard) -- Level 7
  |           |
  |           +-- Cycle 11 (Panel Language Refinement) -- Level 7
```

**Execution order:** 2 -> 3 -> 4 -> 6 -> 5 -> 7 -> 8 -> 9 -> 10 -> 11

---

## Tracking Table

| Cycle | Name                        | Level | Status      | Date       |
|-------|-----------------------------|-------|-------------|------------|
| 0     | Foundation (CTS Interface)  | 1     | DONE        | 2025-01    |
| 1     | Parameter Resonance Network | 1     | DONE        | 2025-01    |
| 2     | Topological Time Machine    | 2     | DONE        | 2025-02    |
| 3     | Manifold Geometry Engine    | 2     | DONE        | 2025-02    |
| 4     | Persistence Fingerprinting  | 3     | DONE        | 2025-02    |
| 5     | Predictive Topology         | 4     | DONE        | 2025-02    |
| 6     | Shadow Tensor Integration   | 3     | DONE        | 2025-02    |
| 7     | Autonomous Advisory         | 5     | DONE        | 2026-02    |
| 8     | Field-Level Intelligence    | 6     | DONE        | 2026-02    |
| 9     | LAS Sliding Analyzer        | 7     | DONE        | 2026-02    |
| 10    | Master Dashboard            | 7     | DONE        | 2026-02    |
| 11    | Panel Language Refinement   | 7     | DONE        | 2026-02    |

---

## Cycle Definitions

### Cycle 0: Foundation (CTS Operator Interface) -- DONE

**Insight level:** 1 -- "What IS the state?"
**Industry disruption:** Replaces flat dashboards with topology-aware state display.

**Delivered:**
- Backend: `POST /api/v1/tda/persistence-diagram`, persistence_diagram in ingest response
- Theme: CTS dark mode (ISO 11064), near-black bg, teal/cyan primary
- Components: AttractorManifold, PersistenceBarcode, TrustGauge, CTSPipelineBar, KPICards
- Home.tsx: 3-panel fixed-viewport (40/35/25%), tabs (CTS/WELL PATH/WIRE MESH)

---

### Cycle 1: Parameter Resonance Network (PRN) -- DONE

**Insight level:** 1 -- "What IS the state?" (cross-channel view)
**Industry disruption:** Replaces isolated parameter monitoring with correlation topology.

**Delivered:**
- Backend: `POST /api/v1/network/compute`, ParameterCorrelationEngine (18 channels, 6 categories)
- Force simulation: custom Verlet in forceSimulation.ts
- Components: ParameterNetworkGraph, ParameterDetailCard, NetworkStatsBar, ChannelSelector
- Home.tsx: 4th tab NETWORK, channel presets

---

### Cycle 2: Topological Time Machine -- DONE

**Insight level:** 2 -- "How does the state EVOLVE?"
**Complexity:** M (medium)
**Dependencies:** Cycles 0, 1

**Insight target:** Track drilling state evolution through topological space over time.

**Industry disruption:** Replaces static SPC charts and threshold alarms with continuous
topological trajectory tracking. Current practice: alarm when WOB > X. New practice:
alarm when the *shape of the data* is deforming toward a dangerous topology, before any
single parameter crosses a threshold.

**Backend scope:**
- Modified: `server.py` (4 new endpoints, 4 Pydantic models)
- Methods surfaced from `tda_pipeline.py`:
  - `compute_full_signature()` -> `POST /api/v1/tda/full-signature`
  - `compute_betti_curve()` -> `POST /api/v1/tda/betti-curve`
  - `compute_windowed_signature()` -> `POST /api/v1/tda/windowed-signatures`
  - `detect_topological_change()` -> `POST /api/v1/tda/change-detect`

**Frontend scope:**
- New: `BettiTimeline.tsx` (~120 lines) -- SVG sparkline of Betti_0/Betti_1 over windows
- New: `TopologicalHeatmap.tsx` (~180 lines) -- Canvas heatmap of filtration evolution
- New: `ChangeDetector.tsx` (~80 lines) -- Change magnitude bar with flash alert
- Modified: `Home.tsx` -- integrate into CTS center panel + footer
- Modified: `types.ts` -- add WindowedSignature, ChangeDetectResult, BettiCurveData

**Estimated:** 8 files (3 new, 5 modified), ~1800 lines

---

### Cycle 3: Manifold Geometry Engine -- DONE

**Insight level:** 2 -- "How does the state EVOLVE?" (geometric view)
**Complexity:** M (medium)
**Dependencies:** Cycle 0

**Insight target:** Visualize the shape of the ROP-warped drilling parameter space.

**Industry disruption:** Replaces Euclidean parameter spaces with Riemannian geometry.
Current practice: plot WOB vs ROP as flat scatter. New practice: show the *curvature*
of parameter space, revealing where small changes cause large effects (high curvature)
vs where the system is insensitive (flat regions).

**Backend scope:**
- New: Metric tensor endpoint (`POST /api/v1/geometry/metric-tensor`)
- New: Geodesic computation endpoint (`POST /api/v1/geometry/geodesic`)
- New: Curvature field endpoint (`POST /api/v1/geometry/curvature`)
- Methods surfaced from drilling geometry modules (metric tensors, geodesics, Ricci curvature)

**Frontend scope:**
- New: `CurvatureField.tsx` -- 2D heatmap of Ricci scalar over parameter space
- New: `GeodesicOverlay.tsx` -- Geodesic paths overlaid on attractor manifold
- Modified: `Home.tsx` -- new tab or sub-view in CTS panel

**Estimated:** 7 files (3 new, 4 modified), ~1500 lines

---

### Cycle 4: Persistence Fingerprinting -- DONE

**Insight level:** 3 -- "WHY is the state what it is?"
**Complexity:** M (medium)
**Dependencies:** Cycle 2

**Insight target:** Identify the topological features (specific persistence pairs) that
define each regime, enabling causal attribution.

**Industry disruption:** Replaces post-hoc log review with real-time causal fingerprinting.
Current practice: after an event, geologists review logs to find the cause. New practice:
the system identifies *which topological features* are driving the current regime in
real time, pointing to the specific drilling parameter interactions responsible.

**Delivered:**
- Backend: 4 endpoints (`POST /api/v1/tda/fingerprint`, `POST /api/v1/tda/attribute`, `POST /api/v1/tda/compare-regimes`, `GET /api/v1/tda/fingerprint-library`)
- Components: RegimeFingerprint (radar chart), AttributionBars (horizontal bars), RegimeCompare (dual radar)
- Home.tsx: 6th tab FINGERPRINT, lazy fetch on tab activation, auto-compare closest two regimes
- Types: FingerprintResponse, AttributionResponse, RegimeCompareResponse, FingerprintDriver, Attribution

---

### Cycle 5: Predictive Topology -- DONE

**Insight level:** 4 -- "What WILL the state be?"
**Complexity:** L (large)
**Dependencies:** Cycles 2, 4

**Insight target:** Forecast the topological state N windows ahead, predicting regime
transitions before they occur.

**Industry disruption:** Replaces reactive alarms with predictive topology forecasting.
Current practice: alarm fires when a threshold is crossed. New practice: forecast that
the topological trajectory is converging toward a dangerous regime, issuing a warning
minutes/hours before the event.

**Delivered:**
- Backend: New `perception/topology_forecaster.py` with TopologyForecaster class
  - `forecast_trajectory()`: weighted linear regression extrapolation of 6 TDA features, confidence bands (1.96*std*sqrt(step)), velocity/acceleration, stability index
  - `compute_transition_probabilities()`: softmax over normalized regime distances with velocity adjustment (tanh-bounded dot product), risk level classification
- 2 endpoints: `POST /api/v1/tda/forecast`, `POST /api/v1/tda/transition-probability`
- Components: TopologyForecast (SVG trajectory with history/forecast split, confidence bands), TransitionRadar (polar probability chart with regime labels, trending indicators)
- Home.tsx: 8th tab FORECAST, lazy fetch, dual-panel (60/40%), reuses windowed signatures from Cycle 2
- Types: ForecastPoint, ForecastResponse, TransitionProbResponse

---

### Cycle 6: Shadow Tensor Integration -- DONE

**Insight level:** 3 -- "WHY is the state what it is?" (hidden dynamics)
**Complexity:** M (medium)
**Dependencies:** Cycle 2

**Insight target:** Surface delay-coordinate embeddings (Takens' theorem) that reveal
hidden dynamical structure invisible in raw parameter space.

**Industry disruption:** Replaces single-parameter time series with state-space
reconstructions. Current practice: look at WOB time series. New practice: embed WOB
in delay coordinates to reveal attractor structure, identifying deterministic chaos
vs stochastic noise, and detecting early warning signals of bifurcations.

**Delivered:**
- Backend: 2 endpoints (`POST /api/v1/shadow/embed`, `POST /api/v1/shadow/attractor`)
- ShadowTensorBuilder aliases: build_from_numpy, build_from_states, concatenate
- Attractor analysis: Lyapunov exponent (Rosenstein), correlation dimension (Grassberger-Procaccia), RQA (recurrence rate, determinism, laminarity, trapping time)
- Attractor classification: fixed_point, limit_cycle, strange_attractor, quasi_periodic, stochastic, transient
- Components: DelayEmbedding (canvas 3D scatter with auto-rotation), LyapunovIndicator (gauge + RQA metrics)
- Home.tsx: 7th tab SHADOW, lazy fetch, dual-panel (60/40%)

---

### Cycle 7: Autonomous Advisory -- DONE

**Insight level:** 5 -- "What SHOULD we do?"
**Complexity:** L (large)
**Dependencies:** Cycles 1, 3, 5

**Insight target:** Compute geodesic-optimal parameter prescriptions that navigate the
drilling manifold toward the target regime.

**Industry disruption:** Replaces generic recommendations with geodesic-optimal parameter
prescriptions. Current practice: "reduce WOB by 10%." New practice: compute the shortest
path through ROP-warped parameter space from current state to target regime, prescribing
exact parameter trajectories.

**Delivered:**
- Backend: New `perception/advisory_engine.py` (AdvisoryEngine class, ParameterStep, AdvisoryResult, RiskAssessment)
  - `compute_advisory()`: regime-specific parameter deltas, correlation-aware ordering, path interpolation, risk scoring
  - `assess_risk()`: evaluates regime risk, change magnitude risk, correlation risk, generates mitigations and abort conditions
  - 8 target regime transition rules (NORMAL, OPTIMAL, DARCY_FLOW, NON_DARCY_FLOW, STICK_SLIP, BIT_BOUNCE, WHIRL, PACKOFF)
- 2 endpoints: `POST /api/v1/advisory/recommend`, `POST /api/v1/advisory/risk`
- Components: AdvisoryPanel (step-by-step prescription, risk bars, mitigations), GeodesicNavigator (SVG parameter trajectory paths)
- Home.tsx: 9th tab ADVISORY, lazy fetch, dual-panel (55/45%), auto-selects OPTIMAL target
- Types: ParameterStep, AdvisoryResponse, RiskFactor, RiskAssessmentResponse

---

### Cycle 8: Field-Level Intelligence -- DONE

**Insight level:** 6 -- "What do ALL wells tell us?"
**Complexity:** L (large)
**Dependencies:** Cycles 4, 5

**Insight target:** Build a topological atlas across multiple wells, enabling field-wide
pattern recognition and cross-well learning.

**Industry disruption:** Replaces per-well offset analysis with field-wide topological
atlas. Current practice: compare offset wells by depth-matched log curves. New practice:
compare wells by their topological signatures, identifying formation-driven vs
drilling-practice-driven regime patterns across the field.

**Delivered:**
- Backend: New `perception/field_atlas.py` (FieldAtlas class, WellEntry, WellComparison, PatternMatch)
  - `register_well()`: compute 10-dim TDA fingerprint + windowed regime distribution from drilling records
  - `register_simulated()`: register pre-computed offset wells
  - `get_atlas()`: return all wells sorted by registration time
  - `compare_wells()`: pairwise topological distance, feature deltas, regime similarity, depth overlap
  - `search_patterns()`: nearest-neighbor search over topological signatures
  - `get_field_summary()`: field-wide regime distribution, mean/spread signature, depth range
- 4 endpoints: `POST /api/v1/field/register`, `GET /api/v1/field/atlas`, `POST /api/v1/field/compare`, `POST /api/v1/field/pattern-search`
- Components: FieldAtlas (well grid with mini radar charts, regime bars, click-to-select), WellCompare (butterfly bar chart, regime sim/depth overlap/topo dist metrics)
- Home.tsx: 10th tab FIELD, auto-registers current well + 3 perturbed offset wells, lazy fetch atlas, click two wells to compare
- Types: FieldWellEntry, FieldAtlasResponse, FieldCompareResponse, PatternMatch

---

### Cycle 9: LAS Sliding Analyzer -- DONE

**Insight level:** 7 -- "How do I USE it operationally?"
**Complexity:** M (medium)
**Dependencies:** Cycles 0, 2

**Insight target:** Turn static LAS display into interactive depth-window stepping with
per-window regime/TDA analysis.

**Industry disruption:** Replaces static log display with interactive depth-window analysis.
Current practice: load LAS file, view entire well at once. New practice: slide through
depth windows re-running regime classification and TDA at each position, building a
colored regime strip showing formation behavior evolution along the wellbore.

**Delivered:**
- Backend: `POST /api/v1/las/{file_id}/analyze-window` composite endpoint
  - Chains: map_to_drilling_records → classify → windowed TDA signatures
  - Returns: records, count, regime, windowed_signatures, index_range
- Components: LASAnalyzer (dual-thumb slider, step controls, auto-play), RegimeStrip (colored regime evolution bar)
- Home.tsx: ANALYZER tab with LASAnalyzer controls + RegimeStrip + LogTrackViewer + KPICards + BettiTimeline
- Types: LASAnalyzeWindowResponse, AnalyzerHistoryEntry

---

### Cycle 10: Master Dashboard -- DONE

**Insight level:** 7 -- "How do I USE it operationally?"
**Complexity:** M (medium)
**Dependencies:** Cycles 0-8

**Insight target:** Unified "at a glance" view aggregating key indicators from all cycles
in drilling operator language.

**Industry disruption:** Replaces scattered technical panels with unified operational dashboard.
Current practice: navigate multiple specialized views to understand well status. New practice:
single dashboard shows regime, parameters, pattern analysis, predictability, and recommended
actions in plain drilling language with color-coded status tiles.

**Delivered:**
- Backend: `POST /api/v1/dashboard/summary` composite endpoint
  - Composes: regime classification, attractor analysis (Lyapunov/RQA), transition probabilities, advisory engine
  - Returns: regime, display name, color, parameters, topology metrics (drilling zones, coupling loops, stability), predictability index, behavioral consistency, transition risk, top advisory step
  - REGIME_DISPLAY_NAMES dict mapping 16 RegimeIDs to operator-friendly names
- Components: MasterDashboard (4-section grid with KPI tiles), DashboardTile (reusable status tile)
- Home.tsx: DASHBOARD as default landing tab, navigation links to specialized tabs
- Types: DashboardSummary

---

### Cycle 11: Panel Language Refinement -- DONE

**Insight level:** 7 -- "How do I USE it operationally?"
**Complexity:** S (small)
**Dependencies:** Cycles 0-10

**Insight target:** Replace mathematical labels with drilling operator terms across all
existing components. No new computations, pure UI language translation.

**Industry disruption:** Removes the last barrier to operator adoption — mathematical jargon.
Current practice: panels show β₀, Lyapunov exponents, Ricci curvature. New practice:
panels show Drilling Zones, Predictability Index, Parameter Sensitivity — terms that
operators understand without a math degree.

**Delivered:**
- Translation applied across 17 components:
  - β₀→Drilling Zones, β₁→Coupling Loops (KPICards, BettiTimeline, TopologyForecast)
  - Entropy→Stability (RegimeFingerprint, AttributionBars, RegimeCompare, FieldAtlas, WellCompare)
  - Lyapunov→Predictability Index, RQA→behavioral terms (LyapunovIndicator)
  - Ricci Scalar→Parameter Sensitivity (CurvatureField)
  - Geodesic→Optimal Path (GeodesicOverlay)
  - Persistence Barcode→Feature Lifetime Chart (PersistenceBarcode)
  - Topo Change→Pattern Shift (ChangeDetector)
  - Topological Heatmap→Feature Evolution Map (TopologicalHeatmap)
  - Delay Embedding→Hidden Dynamics (DelayEmbedding)
  - Attractor types: fixed_point→Stable, limit_cycle→Cyclic, strange_attractor→Complex, quasi_periodic→Multi-Cycle, stochastic→Noisy, transient→Transitioning
- Tab renames: GEOMETRY→SENSITIVITY, FINGERPRINT→SIGNATURE, SHADOW→DYNAMICS, FIELD→FIELD MAP
- New: InfoTooltip component for contextual hover help
- Trust Calibration→Automation Readiness (TrustGauge)
