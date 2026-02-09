# Jones Framework

**State-Adaptive Computational Intelligence Backend**

The Jones Framework is a Python library for regime-aware computational intelligence, combining Topological Data Analysis (TDA) with Mixture of Experts (MoE) architectures for real-time state classification and adaptive processing.

---

## Architecture Overview

```
jones_framework/
├── core/                    # Foundational primitives
│   ├── condition_state.py   # Atomic immutable observations
│   ├── activity_state.py    # Regime definitions + metrics
│   ├── manifold_bridge.py   # Component registry + dependencies
│   ├── shadow_tensor.py     # Multi-representation tensors
│   ├── tensor_ops.py        # Tensor algebra operations
│   ├── novelty_search.py    # Novelty gradient computation
│   └── knowledge_flow.py    # Multi-perspective integration
│
├── perception/              # Regime detection
│   ├── tda_pipeline.py      # Persistent homology (Ripser)
│   ├── regime_classifier.py # Point cloud → regime mapping
│   └── metric_warper.py     # Conformal metric transformations
│
├── sans/                    # Specialization architecture
│   ├── mixture_of_experts.py # Regime-based expert routing
│   ├── lora_adapter.py      # Low-rank adaptation layers
│   └── continuity_guard.py  # Smooth transition enforcement
│
├── arbitrage/               # Linguistic processing
│   ├── sentiment_vector.py  # NLP sentiment extraction
│   └── linguistic_arbitrage.py # Kinetic triggers
│
├── domains/                 # Domain-specific adapters
│   ├── drilling/            # Oil & gas drilling ops
│   ├── finance/             # Market data processing
│   ├── healthcare/          # Medical signals (stub)
│   └── reservoir/           # Geological modeling (stub)
│
├── data/                    # Data layer
│   ├── connectors/          # External data sources
│   ├── point_cloud/         # Point cloud processing (16 modules)
│   └── streams/             # Real-time data streams
│
├── api/                     # API interfaces
│   ├── server.py            # FastAPI main application
│   ├── rest/                # REST endpoints
│   └── websocket/           # Real-time streaming
│
└── config/                  # Configuration management
    └── settings.py          # Environment & feature flags
```

---

## Core Concepts

### ConditionState
Atomic, immutable observations representing a point in state space:

```python
from jones_framework import ConditionState

state = ConditionState(
    timestamp=1705520000,
    vector=(25.0, 120, 85, 12000, 0.3),  # (WOB, RPM, ROP, Torque, Vibration)
    metadata={'formation': 'shale', 'tvd': 8500},
    verified=True
)
```

### ActivityState (Regimes)
Macroscopic phases where system behavior follows distinct rules:

```python
from jones_framework import ActivityState, RegimeID

regime = ActivityState(
    regime_id=RegimeID.HIGH_VOLATILITY,
    transition_threshold=0.7,
    metadata={'detected_dysfunction': 'stick_slip'}
)
```

**Available Regimes (16 total):**
- Fluid: `DARCY_FLOW`, `NON_DARCY_FLOW`, `TURBULENT`, `MULTIPHASE`
- Market: `RISK_ON`, `RISK_OFF`, `LOW_VOLATILITY`, `HIGH_VOLATILITY`, `MEAN_REVERTING`, `MOMENTUM`, `LIQUIDITY_CRISIS`, `CENTRAL_BANK_INTERVENTION`
- Meta: `STABLE`, `TRANSITION`, `CHAOS`, `UNKNOWN`

### TDA Pipeline
Extracts topological features via persistent homology:

```python
from jones_framework import TDAPipeline
import numpy as np

tda = TDAPipeline(embedding_dim=3, max_dimension=1)

# Point cloud of observations
data = np.random.randn(100, 5)

features = tda.extract_features(data)
print(f"Betti-0: {features['betti_0']}")  # Connected components
print(f"Betti-1: {features['betti_1']}")  # Loops/cycles
```

### Mixture of Experts
Routes inputs to regime-specialized expert networks:

```python
from jones_framework import MixtureOfExperts, RegimeClassifier

classifier = RegimeClassifier()
moe = MixtureOfExperts(num_experts=6)

# Classify regime
result = classifier.classify(data)
print(f"Detected regime: {result['regime']}")

# Route to expert
output = moe.process(data, regime=result['regime'])
```

---

## Installation

```bash
# Basic installation
pip install -e .

# With CLI support (recommended)
pip install -e ".[cli]"

# With all optional dependencies
pip install -e ".[all]"

# Development installation
pip install -e ".[dev]"
```

### Dependencies

**Core:**
- numpy, scipy, scikit-learn, pandas
- ripser, persim (TDA)
- pyyaml

**CLI:**
- typer (command-line interface)
- rich (terminal formatting)

**Optional:**
- torch (GPU acceleration)
- transformers (NLP models)
- gudhi (alternative TDA)
- cupy (CUDA support)

---

## CLI (Command Line Interface)

The `jones` CLI provides an accessible interface for installation, service management, and troubleshooting:

```bash
# Launch interactive menu
jones

# Quick start (checks system, starts services)
jones quick

# Guided installation wizard
jones wizard

# Service management
jones start          # Start backend and frontend
jones stop           # Stop all services
jones status         # Check what's running
jones restart        # Restart services

# Troubleshooting
jones doctor         # Run diagnostics
jones doctor --fix   # Attempt automatic fixes
jones logs           # View service logs
jones logs --follow  # Follow logs in real-time
```

### CLI Installation Modes

The `jones install` command supports different installation modes:

| Mode | Size | Includes |
|------|------|----------|
| `simple` | ~100 MB | Core framework, basic API |
| `standard` | ~300 MB | Full API, web interface, TDA pipeline |
| `full` | ~3 GB | Everything + ML models + all adapters |

```bash
jones install --mode simple    # Minimal installation
jones install --mode standard  # Recommended for most users
jones install --mode full      # For developers and power users
```

---

## Quick Start

### 1. Regime Classification

```python
from jones_framework import RegimeClassifier, TDAPipeline
import numpy as np

# Initialize
tda = TDAPipeline(embedding_dim=3, max_dimension=1)
classifier = RegimeClassifier(tda_pipeline=tda)

# Simulate telemetry (WOB, RPM, ROP, Torque, Vibration)
telemetry = np.array([
    [25.0, 120, 85, 12000, 0.3],
    [25.5, 118, 82, 12200, 0.4],
    [26.0, 115, 78, 13000, 0.8],
    [24.0, 110, 65, 14500, 1.2],
    [22.0, 105, 45, 16000, 2.1],
])

result = classifier.classify(telemetry)
print(f"Regime: {result['regime']}, Confidence: {result['confidence']:.2f}")
```

### 2. Starting the API Server

```bash
uvicorn jones_framework.api.server:app --reload --port 8000
```

### 3. Making API Requests

```bash
# Health check
curl http://localhost:8000/health

# Classify regime
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"data": [[25.0, 120, 85, 12000, 0.3], [25.5, 118, 82, 12200, 0.4]]}'
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System health check |
| `/api/v1/state/create` | POST | Create ConditionState |
| `/api/v1/classify` | POST | Classify regime from data |
| `/api/v1/tda/features` | POST | Extract TDA features |
| `/api/v1/moe/process` | POST | Process through MoE |
| `/api/v1/moe/hot-swap` | POST | Swap LoRA adapters |
| `/ws/stream` | WS | Real-time streaming |

---

## Domain Adapters

### Drilling Domain

```python
from jones_framework.domains.drilling import (
    DrillingDomainAdapter,
    DrillingManifold,
    DrillingRegime
)

adapter = DrillingDomainAdapter()
manifold = DrillingManifold()

# Create drilling-specific state
state = adapter.create_state({
    'wob': 25.0,
    'rpm': 120,
    'rop': 85,
    'torque': 12000,
    'formation': 'shale',
    'tvd': 8500
})

# Get geodesic (optimal drilling path)
geodesic = manifold.compute_geodesic(state_a, state_b)
```

---

## Component Registry

All components are registered in the manifold for dependency tracking:

```python
from jones_framework import bridge, ComponentRegistry

@bridge(connects_to=['ConditionState', 'TDAPipeline'])
class MyComponent:
    """Automatically registered with dependency graph."""
    pass

# View registry
registry = ComponentRegistry()
print(registry.visualize())  # DOT format graph
```

---

## Testing

```bash
# Run all tests
pytest -v

# With coverage
pytest -v --cov=jones_framework

# Specific module
pytest tests/test_api.py -v
```

---

## Configuration

### Environment Variables

```bash
JONES_ENV=development|production
JONES_DEVICE=auto|cuda|metal|cpu
JONES_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
```

### YAML Configuration

```yaml
environment: development
hardware:
  device_preference: auto
  batch_size: 64
sans:
  num_experts: 6
  lora_rank: 8
tda:
  embedding_dim: 3
  max_dimension: 1
```

---

## Key Modules Reference

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `core.condition_state` | `ConditionState` | Immutable atomic data |
| `core.activity_state` | `ActivityState`, `RegimeID` | Regime definitions |
| `core.manifold_bridge` | `ComponentRegistry`, `bridge` | Dependency tracking |
| `perception.tda_pipeline` | `TDAPipeline` | Persistent homology |
| `perception.regime_classifier` | `RegimeClassifier` | Regime detection |
| `sans.mixture_of_experts` | `MixtureOfExperts`, `Expert` | Expert routing |
| `sans.lora_adapter` | `LoRAAdapter`, `LoRAAdapterBank` | Fast weight switching |
| `sans.continuity_guard` | `ContinuityGuard` | Transition smoothing |
| `arbitrage.sentiment_vector` | `SentimentVectorPipeline` | NLP processing |
| `domains.drilling.adapter` | `DrillingDomainAdapter` | Drilling-specific logic |

---

## License

MIT License

---

## Related Documentation

- [Main README](../README.md) - Full platform overview
- [CLAUDE.md](../CLAUDE.md) - Axiom-driven development guide
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Enterprise deployment
- [Les Logic](../docs/LES_LOGIC_BHA_OPTIMIZATION.md) - BHA optimization methodology
- [Drilling Quickstart](../docs/DRILLING_QUICKSTART.md) - Field operations guide
