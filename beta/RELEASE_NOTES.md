# Release Notes - Beta v0.1.0

**Date:** 2026-02-09
**Status:** Pre-Install Beta (Development Complete)
**Codename:** SupaTrupa

---

## What Is This?

A cognitive topological system (CTS) for real-time drilling intelligence. It uses Topological Data Analysis (TDA) to detect drilling regime changes from the *shape* of multi-parameter data, rather than single-parameter thresholds.

**For the test team:** This is the full platform after 11 development cycles. All features are implemented and the build is clean. Your job is to verify it works end-to-end.

---

## Capabilities Delivered

### Cycles 0-1: Foundation (Level 1 - "What IS the state?")
- 16-regime classification via persistent homology
- CTS operator interface (ISO 11064 dark mode)
- Parameter Resonance Network (18-channel correlation graph)
- Mixture of Experts with LoRA hot-swap
- WebSocket streaming for live regime updates

### Cycles 2-3: Evolution (Level 2 - "How does the state EVOLVE?")
- Windowed TDA signatures tracking topology over time
- Betti curves, change detection, topological heatmaps
- Riemannian curvature field (parameter sensitivity mapping)
- Geodesic computation (optimal paths through parameter space)

### Cycle 4 & 6: Causation (Level 3 - "WHY is the state what it is?")
- Persistence fingerprinting (10-dim regime signatures)
- Per-feature attribution (which TDA dimensions drive each regime)
- Shadow tensor / delay embedding (Takens' theorem)
- Attractor classification (6 types: Stable, Cyclic, Complex, Multi-Cycle, Noisy, Transitioning)
- Lyapunov exponent + RQA analysis

### Cycle 5: Prediction (Level 4 - "What WILL the state be?")
- Topology trajectory forecasting with confidence bands
- Transition probability (softmax over regime distances)
- Risk level classification (low/medium/high/critical)

### Cycle 7: Prescription (Level 5 - "What SHOULD we do?")
- Advisory engine with parameter prescriptions
- Correlation-aware step ordering
- Risk assessment with mitigations and abort conditions

### Cycle 8: Field Intelligence (Level 6 - "What do ALL wells tell us?")
- Multi-well topological atlas
- Pairwise well comparison (distance, regime similarity)
- Pattern search across registered wells

### Cycles 9-11: Operations (Level 7 - "How do I USE it?")
- LAS sliding window analyzer with regime strip
- Master dashboard (unified operational overview)
- All math labels translated to drilling operator terms

---

## Platform Numbers

| Metric | Count |
|--------|-------|
| API endpoints | 45 |
| UI tabs | 12 |
| Frontend components | 42 |
| Backend perception engines | 8 |
| Regime types | 16 |
| TDA feature dimensions | 10 |
| Attractor types | 6 |
| Parameter channels | 18 |

---

## Polish Pass (Post-Development)

- Fixed dashboard crash bugs (TopologyForecaster args, ShadowTensorBuilder names)
- Fixed ~70 TypeScript errors across the frontend
- Deleted 15 orphaned/unused components
- Fixed tsconfig.node.json for clean `tsc --noEmit`
- Replaced 10 silent Python `except: pass` with proper logging
- Replaced 15 silent JS `.catch(() => {})` with `console.warn`
- Added vite-env.d.ts for `import.meta.env` typing

---

## Known Issues

| Issue | Severity | Workaround |
|-------|----------|------------|
| 1.7 MB JS bundle | Low | Works fine, just slow first load |
| No persistent storage | Medium | All state lost on backend restart |
| `@bridge(metadata=...)` broken | Low | Omit metadata kwarg |
| ShadowTensorBuilder obfuscated params | Low | Public API aliases work correctly |
| LAS tabs need file upload first | Info | Upload via API or UI before using ANALYZER/WIRE MESH |

---

## Test Resources

- **Full Test Plan:** `docs/TEST_PLAN.md`
- **Consumer Tests:** `scripts/consumer-tests.sh` (15 automated tests)
- **Quick Verify:** `beta/verify.sh` (5-check smoke test)
- **Setup Script:** `beta/setup.sh` (one-command install)
- **Glossary:** `docs/user-guides/05_GLOSSARY.md`
