# Release Notes - Beta v0.1.0

**Date:** 2026-02-09
**Status:** Pre-Install Beta (Development Complete)
**Codename:** SupaTrupa

---

## What Is This?

A multi-parameter drilling monitoring and analysis platform. It uses Topological Data Analysis (TDA) to classify drilling regimes based on multi-parameter data patterns, supplementing conventional single-parameter threshold monitoring.

**For the test team:** This is the full platform after 11 development cycles. All features are implemented and the build is clean. Your job is to verify it works end-to-end.

---

## Capabilities Delivered

### Cycles 0-1: Foundation (Level 1 - "What is the current state?")
- 16-regime classification via persistent homology
- Operator interface (ISO 11064 dark mode)
- Parameter correlation network (18-channel graph)
- Mixture of Experts with LoRA adapter switching
- WebSocket streaming for regime updates

### Cycles 2-3: Trend Analysis (Level 2 - "How is the state changing?")
- Windowed TDA signatures tracking topology over time
- Betti curves, change detection, feature evolution maps
- Parameter sensitivity mapping (curvature field)
- Suggested parameter paths through state space

### Cycle 4 & 6: Attribution (Level 3 - "What factors are associated with this state?")
- Persistence fingerprinting (10-dimension regime signatures)
- Per-feature attribution (which TDA dimensions are associated with each regime)
- Delay embedding analysis (Takens' theorem)
- Attractor classification (6 types: Stable, Cyclic, Complex, Multi-Cycle, Noisy, Transitioning)
- Lyapunov exponent + Recurrence Quantification Analysis

### Cycle 5: Forecasting (Level 4 - "What might the state become?")
- Topology trajectory extrapolation with confidence bands
- Transition probability estimation (softmax over regime distances)
- Risk level classification (low/medium/high/critical)

### Cycle 7: Decision Support (Level 5 - "What are the suggested adjustments?")
- Advisory engine with parameter adjustment suggestions
- Correlation-aware step ordering
- Risk assessment with mitigations and abort conditions

### Cycle 8: Field Comparison (Level 6 - "How do wells compare?")
- Multi-well topological atlas
- Pairwise well comparison (distance, regime similarity)
- Pattern search across registered wells

### Cycles 9-11: Operations (Level 7 - "How is it used day-to-day?")
- LAS sliding window analyzer with regime strip
- Master dashboard (consolidated operational overview)
- Analytical labels translated to standard drilling terminology

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
