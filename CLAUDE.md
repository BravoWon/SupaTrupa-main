# Unified Activity:State Platform - Axiom-Driven Development

> **Foundational Theory:** See [EPISTEMIC_ENGINE.md](./EPISTEMIC_ENGINE.md) for the complete Jones Axiomatic Framework (Axioms G1-G5) that governs runtime cognition.

## Meta-Architecture: The Development Manifold

This codebase applies its own principles to its development process. The same axioms that govern runtime behavior govern how we write code.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AXIOM STACK                                  │
├─────────────────────────────────────────────────────────────────┤
│ 1. Substrate Principle    → Git commits are ConditionStates    │
│ 2. Hierarchy Principle    → Dev phases are ActivityStates      │
│ 3. Manifold Hypothesis    → Code topology defines complexity   │
│ 4. Continuity Thesis      → Changes must be continuous maps    │
│ 5. SANS Architecture      → Different tools for different tasks│
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Substrate Principle: Commits as ConditionStates

Every git commit is an immutable, timestamped, verifiable fact.

```python
# Runtime equivalent
ConditionState(timestamp=1705520000, vector=(price, volume, ...), verified=True)

# Development equivalent
git commit --message "feat: add regime classifier" # Immutable fact
```

**Rules:**
- Never rewrite published history (`--force` is a topology violation)
- Each commit is atomic and verifiable
- Commit messages are metadata, not the state itself

---

## 2. Hierarchy Principle: Development Regimes

Development has distinct phases where the rules change:

| Dev Regime | Runtime Equivalent | Metric Tensor | Expert Tool |
|------------|-------------------|---------------|-------------|
| `DEBUGGING` | `HIGH_VOLATILITY` | Error proximity | Debugger, logs |
| `FEATURE_DEV` | `MOMENTUM` | Velocity toward goal | AI assistant |
| `REFACTORING` | `MEAN_REVERTING` | Entropy reduction | Linter, types |
| `TESTING` | `STABLE` | Coverage, green | pytest, vitest |
| `REVIEW` | `TRANSITION` | Diff analysis | Human + AI |

**Regime Detection:**
```bash
# High error rate + frequent reverts → DEBUGGING regime
# Clean tests + forward progress → FEATURE_DEV regime
# No new features + structure changes → REFACTORING regime
```

---

## 3. Manifold Hypothesis: Code Topology

Code has shape. Complexity is topology, not just line count.

| Betti Number | Code Meaning | Detection |
|--------------|--------------|-----------|
| β₀ (components) | Disconnected modules | Import graph analysis |
| β₁ (cycles) | Circular dependencies | `madge --circular` |
| β₂ (voids) | Dead code regions | Coverage gaps |

**Healthy topology:**
- β₀ = 1 (one connected codebase)
- β₁ = 0 (no circular deps)
- High coverage (no voids)

**The `@bridge` decorator enforces this:**
```python
from jones_framework.core.manifold_bridge import bridge

@bridge(connects_to=['ConditionState', 'TDAPipeline'])
class NewComponent:
    """Must connect to existing manifold or registration fails."""
    pass
```

---

## 4. Continuity Thesis: Change as Continuous Map

Valid code changes are continuous deformations of the codebase manifold.

**Continuous (allowed):**
- Adding a new function that calls existing functions
- Extending a class with new methods
- Refactoring that preserves behavior (tests pass before and after)

**Discontinuous (requires explicit regime change):**
- Breaking API changes
- Removing public interfaces
- Changing data schemas

**Detection:**
```bash
# Pre-commit hook checks for discontinuities
python -c "from jones_framework import RecursiveImprover; print(RecursiveImprover().identify_gaps())"
```

---

## 5. SANS Architecture: Expert Tools

Different development tasks route to different expert tools:

```
┌─────────────────┐     ┌─────────────────┐
│ Task Detection  │────▶│ Expert Router   │
│ (What regime?)  │     │ (Which tool?)   │
└─────────────────┘     └─────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Claude Code     │ │ pytest/vitest   │ │ black/prettier  │
│ (FEATURE_DEV)   │ │ (TESTING)       │ │ (REFACTORING)   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Component Registry

All framework components are registered in the manifold:

```python
from jones_framework.core.manifold_bridge import get_registry

registry = get_registry()

# View all components
print(registry.to_json())

# Visualize as graph
print(registry.visualize())  # DOT format

# Get improvement suggestions
from jones_framework.core.manifold_bridge import RecursiveImprover
improver = RecursiveImprover()
print(improver.generate_improvement_report())
```

### Core Components (14 nodes)

```
Tensor ──────────────────────────────────────────────────┐
   │                                                     │
   ▼                                                     │
ConditionState ◀─────────────────────────────────────────┤
   │                                                     │
   ├──▶ ActivityState                                    │
   │         │                                           │
   │         ▼                                           │
   ├──▶ ShadowTensor ──▶ TDAPipeline ──▶ RegimeClassifier│
   │                          │                │         │
   │                          ▼                ▼         │
   │              LinguisticArbitrageEngine    │         │
   │                          │                │         │
   │                          ▼                ▼         │
   │              SentimentVectorPipeline  MixtureOfExperts
   │                                           │
   │                                           ▼
   └──▶ ContinuityGuard                   LoRAAdapter
```

---

## Development Commands

### Backend (Python)

```bash
# Install with all dependencies
cd backend && pip install -e ".[all]"

# Run component manifold analysis
python -c "
from jones_framework.core.manifold_bridge import RecursiveImprover
improver = RecursiveImprover()
print(improver.generate_improvement_report())
"

# Validate new component connections
python -c "
from jones_framework.core.manifold_bridge import get_registry
registry = get_registry()
errors = registry.validate_connections('YourNewComponent')
print(errors if errors else 'All connections valid')
"

# Run tests
pytest -v

# Start API server
uvicorn jones_framework.api.server:app --reload --port 8000
```

### Frontend (React)

```bash
cd frontend && pnpm install
pnpm dev      # Development
pnpm build    # Production
pnpm check    # Type check
```

### Full Stack

```bash
./scripts/dev.sh           # Start both servers
./scripts/test.sh          # Run all tests
./scripts/consumer-tests.sh # E2E validation
```

---

## Creating New Components

Every new component MUST connect to the manifold:

```python
from jones_framework.core.manifold_bridge import bridge, ConnectionType

@bridge(
    connects_to=['ConditionState', 'TDAPipeline'],
    connection_types={
        'ConditionState': ConnectionType.USES,
        'TDAPipeline': ConnectionType.TRANSFORMS
    },
    metadata={'domain': 'perception', 'version': '1.0.0'}
)
class MyNewPerceptionComponent:
    """
    This component is automatically registered and validated.

    If it doesn't connect to existing components, registration
    will warn about orphaned code.
    """

    def __init__(self, tda_pipeline: TDAPipeline):
        self.tda = tda_pipeline

    def process(self, state: ConditionState) -> dict:
        # Transform condition state through TDA
        features = self.tda.extract_features(state.to_numpy().reshape(1, -1))
        return features
```

### Decorator Shortcuts

```python
from jones_framework.core.manifold_bridge import extends, transforms, composes

@extends('ConditionState')
class EnhancedState:
    """Inherits from ConditionState."""
    pass

@transforms('ConditionState', 'ActivityState')
class StateTransformer:
    """Transforms between state types."""
    pass

@composes('TDAPipeline', 'RegimeClassifier')
class PerceptionEngine:
    """Composed of multiple components."""
    pass
```

---

## Recursive Improvement

The system can analyze itself and suggest improvements:

```python
from jones_framework.core.manifold_bridge import RecursiveImprover

improver = RecursiveImprover()

# Find gaps in the manifold
gaps = improver.identify_gaps()
for gap in gaps:
    print(f"Gap: {gap['type']} in {gap['component']}")
    print(f"Suggestion: {gap['suggestion']}")

# Get improvement priority order
priority = improver.compute_improvement_path()
print(f"Next component to improve: {priority[0]}")

# Suggest new component for a domain
suggestion = improver.suggest_new_component('domains.healthcare')
print(f"Missing types: {suggestion['suggested_types']}")
```

---

## Key Files

### Backend Core
| File | Purpose |
|------|---------|
| `core/condition_state.py` | Atomic immutable data structure |
| `core/activity_state.py` | Regime definitions with metric tensors |
| `core/manifold_bridge.py` | Component registry and recursive improvement |
| `perception/tda_pipeline.py` | Persistent homology computation |
| `sans/mixture_of_experts.py` | Expert selection and hot-swap |

### Frontend Core
| File | Purpose |
|------|---------|
| `components/StateSpaceViz.tsx` | 3D state space visualization |
| `components/TopologicalMapper.tsx` | TDA feature visualization |
| `lib/gtMoeOptimizer.ts` | TypeScript MoE implementation |

### Integration
| File | Purpose |
|------|---------|
| `backend/jones_framework/api/server.py` | FastAPI bridge |
| `shared/types/index.ts` | Cross-language types |

---

## Environment Variables

```bash
# Backend
JONES_ENV=development|production
JONES_DEVICE=auto|cuda|metal|cpu
JONES_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Frontend
VITE_API_URL=http://localhost:8000
NODE_ENV=development|production
```

---

## Philosophy

> "Every component must connect to the manifold. Orphaned code is a topology violation."

The Manifold Bridge ensures:
1. **No orphaned code** - Everything connects back to core primitives
2. **Automatic dependency tracking** - The registry knows all connections
3. **Gradient-based improvement** - The system knows where to invest effort
4. **Continuous validation** - Changes are checked for topological consistency

This is not just architecture documentation. It's executable specification.
