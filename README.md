# Unified Activity:State Platform

**Cognitive Command Center for State-Adaptive Computational Intelligence**

A production-ready synthesis of two engineering systems, combining the theoretical rigor of the Jones Framework with the visual power of real-time 3D dashboards.

---

## Features

### Computational Engine (Python Backend)
- **ConditionState/ActivityState Primitives**: Immutable atomic data structures with Riemannian metric tensors
- **TDA Pipeline**: Persistent homology via Ripser for regime detection
- **SANS Architecture**: Mixture of Experts with LoRA hot-swap capability
- **Linguistic Arbitrage**: NLP-based kinetic triggers for sentiment-driven regime shifts
- **Hardware Acceleration**: Automatic GPU/NPU detection (CUDA, Metal, CPU fallback)

### Visualization Layer (React Frontend)
- **Master Dashboard**: Unified operational overview with regime, topology, dynamics, and advisory status
- **CTS Operator Interface**: ISO 11064-compliant dark mode with AttractorManifold, PersistenceBarcode, TrustGauge
- **GT-MoE Optimizer**: Real-time regime classification with Betti number displays
- **12 Specialized Tabs**: Dashboard, CTS, Well Path, Wire Mesh, Network, Sensitivity, Signature, Dynamics, Forecast, Advisory, Field Map, Analyzer
- **Modern UI**: React 19, Radix UI, Tailwind CSS, Framer Motion

### Integration
- **REST/WebSocket API**: FastAPI bridge connecting Python and React
- **Shared Type System**: JSON Schemas validated across Python and TypeScript
- **Real-time Streaming**: Live regime updates and signal alerts

---

## Quick Start

### Prerequisites
- Python 3.9+ (3.10+ recommended)
- Node.js 18+
- pnpm

### Option 1: Interactive CLI (Recommended)

The `jones` CLI provides an accessible interface for all skill levels:

```bash
# Install with CLI support
cd backend
pip install -e ".[cli]"

# Launch interactive menu
jones

# Or use the guided wizard
jones wizard

# Quick start (checks, installs, and starts everything)
jones quick
```

**CLI Commands:**
| Command | Description |
|---------|-------------|
| `jones` | Interactive menu |
| `jones quick` | One-command setup and start |
| `jones wizard` | Guided installation |
| `jones install` | Install dependencies |
| `jones start` | Start services |
| `jones stop` | Stop services |
| `jones status` | Check what's running |
| `jones doctor` | Diagnose problems |
| `jones logs` | View service logs |

### Option 2: Manual Installation

```bash
# Clone the repository
git clone <repository-url>
cd unified-activity-state-platform

# Install Python backend
cd backend
pip install -e ".[all]"

# Install React frontend
cd ../frontend
pnpm install
```

### Development

```bash
# Using the CLI (recommended)
jones start

# Or manually in separate terminals:
# Terminal 1: Start Python API server
cd backend
uvicorn jones_framework.api.server:app --reload --port 8000

# Terminal 2: Start React dev server
cd frontend
pnpm dev
```

### Production

```bash
# Build frontend
cd frontend && pnpm build

# Start production server
NODE_ENV=production node dist/index.js
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Master       │  │ Attractor    │  │ Parameter Network    │  │
│  │ Dashboard    │  │ Manifold     │  │ Graph (PRN)          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Topology     │  │ Curvature    │  │ LAS Analyzer /       │  │
│  │ Forecast     │  │ Field        │  │ Field Atlas          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                   API Client (fetch + WS)                  │ │
│  └────────────────────────────┬──────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────────┘
                                │ REST/WebSocket (45 endpoints)
┌───────────────────────────────▼─────────────────────────────────┐
│                       API BRIDGE (FastAPI)                      │
│  /classify  /drilling/ingest  /tda/*  /shadow/*  /advisory/*   │
│  /geometry/*  /field/*  /dashboard/*  /las/*  /ws/stream       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      BACKEND (jones_framework)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ConditionState│  │ActivityState│  │ TDAPipeline │             │
│  │   (Atomic)   │  │  (Regimes)  │  │(Persistence)│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │MixtureOfExp.│  │LoRA Adapters│  │ Topology    │             │
│  │   (SANS)    │  │(Hot-Swap)   │  │ Forecaster  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Advisory    │  │  Shadow     │  │  Field      │             │
│  │ Engine      │  │  Tensor     │  │  Atlas      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### The Geometry of Being (GoB)

The system operates on the principle that complex systems exist in distinct "thermodynamic phases" (Activity:States), where the rules of physics themselves change between regimes.

- **ConditionState**: Atomic, immutable observations (like spin network nodes in Loop Quantum Gravity)
- **ActivityState**: Macroscopic regimes with distinct metric tensors
- **Geodesics**: Optimal paths that curve according to regime geometry

### Topological Data Analysis

The system "perceives" regime structure through persistent homology:

| Betti Number | Meaning | Drilling Interpretation | Operator Term |
|--------------|---------|------------------------|---------------|
| β₀ | Connected components | Distinct drilling zones | Drilling Zones |
| β₁ | Loops/cycles | Coupled parameter oscillations | Coupling Loops |
| β₂ | Voids | Structural gaps in parameter space | - |

### SANS Architecture

"Symbolic Abstract Neural Search" - specialized experts for each regime:

1. **Regime Detection**: TDA pipeline classifies current state
2. **Expert Selection**: Route to regime-specific expert
3. **Hot-Swap**: LoRA adapters enable instant switching
4. **Continuity Guard**: Topological safety enforcement

---

## Current UI Tabs

| Tab | Description | Key Components |
|-----|-------------|----------------|
| **DASHBOARD** | Default landing - unified operational overview | MasterDashboard, DashboardTile |
| **CTS** | Cognitive Topological System operator interface | AttractorManifold, PersistenceBarcode, TrustGauge, KPICards |
| **WELL PATH** | 3D well trajectory visualization | Three.js/R3F |
| **WIRE MESH** | LAS wire mesh visualization | LASMeshViz |
| **NETWORK** | Parameter correlation force graph | ParameterNetworkGraph, ChannelSelector |
| **SENSITIVITY** | Parameter sensitivity heatmap + optimal paths | CurvatureField, GeodesicOverlay |
| **SIGNATURE** | Regime fingerprint radar + attribution | RegimeFingerprint, AttributionBars, RegimeCompare |
| **DYNAMICS** | Hidden dynamics + attractor analysis | DelayEmbedding, LyapunovIndicator |
| **FORECAST** | Topology trajectory prediction | TopologyForecast, TransitionRadar |
| **ADVISORY** | Parameter prescriptions + risk assessment | AdvisoryPanel, GeodesicNavigator |
| **FIELD MAP** | Multi-well comparison atlas | FieldAtlas, WellCompare |
| **ANALYZER** | LAS sliding window regime analysis | LASAnalyzer, RegimeStrip |

## API Summary

45 endpoints across 16 categories (34 POST, 9 GET, 1 WS, 1 multipart upload). See [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for full endpoint documentation with curl commands.

---

## Testing

```bash
# Run all tests
./scripts/test.sh

# Backend only
cd backend && pytest -v --cov=jones_framework

# Frontend only
cd frontend && pnpm test

# E2E consumer tests (15 endpoint groups)
./scripts/consumer-tests.sh

# Full test plan for test team
# See: docs/TEST_PLAN.md
```

---

## Configuration

Create a `.env` file:

```env
# Backend
JONES_ENV=development
JONES_DEVICE=auto
JONES_LOG_LEVEL=INFO

# Frontend
VITE_API_URL=http://localhost:8000
```

Or use `config.yaml`:

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

## License

**Proprietary - All Rights Reserved.** See [LICENSE](LICENSE).

This software is source-available for viewing only. No rights are granted to use, copy, modify, or distribute. Contact the copyright holder for licensing inquiries.

---

## Credits

Synthesized from:
- **Bestest-main** (jones_framework): State-Adaptive Computational Intelligence Framework
- **activity-state-dashboard**: Drilling Intelligence Platform Enterprise Beta v1.0

---

## Les Logic: BHA Optimization Methodology

The **Les Logic** methodology applies specifically to Bottom Hole Assembly (BHA) optimization within this platform. It provides a fundamentally different approach to drilling dysfunction detection and BHA configuration recommendations.

**For comprehensive documentation, see: [docs/LES_LOGIC_BHA_OPTIMIZATION.md](docs/LES_LOGIC_BHA_OPTIMIZATION.md)**

### Quick Overview

Les Logic addresses a critical gap in current drilling optimization: traditional methods are **reactive** (detecting problems after they occur) and **threshold-based** (missing complex system dynamics).

**Current Industry Approach:**
```
Sensor Data → Threshold Check → Alert (after damage begins)
```

**Les Logic Approach:**
```
Sensor Data → Topological Analysis → Regime Detection → BHA Recommendation (before damage)
```

### The Three Axioms

| Axiom | Application to BHA Optimization |
|-------|--------------------------------|
| **Axiomatic Alignment** | Every measurement connects to physical BHA behavior—torque causes rotation, WOB causes penetration |
| **Non-Euclidean Topology** | Drilling state space is curved: ROP changes "cost" different amounts depending on where you are in the parameter space |
| **Cognitive Ergonomics** | Outputs are actionable: "Add 1 stabilizer" not "β₁ increased" |

### Key Capabilities

1. **Regime Detection via Persistent Homology**
   - Analyzes drilling data as point clouds
   - Computes Betti numbers to identify oscillatory behavior (stick-slip, whirl, bit bounce)
   - Detects regime transitions *before* traditional thresholds trigger

2. **Physics-Based Recommendations**
   | Detected Regime | Root Cause | Les Logic Recommendation |
   |-----------------|------------|--------------------------|
   | Whirl | BHA eccentricity | Add stabilizer |
   | Stick-Slip | Torsional resonance | Reduce motor bend angle |
   | Bit Bounce | Axial instability | Increase flow restrictor |

3. **Riemannian Metric for Drilling**
   - Quantifies "cost" of drilling operations in curved state space
   - Geodesics represent optimal drilling paths
   - Ricci curvature indicates formation difficulty changes

### Why a New Approach?

Current methods fail because they treat drilling parameters as independent variables with fixed thresholds. In reality:

- Parameters are **coupled** (WOB affects torque affects ROP affects vibration)
- Regime transitions are **abrupt** (small changes → large effects at critical points)
- Dysfunction is **topological** (the *shape* of data in phase space matters, not just values)

Les Logic uses Topological Data Analysis (TDA) to detect these patterns that traditional monitoring misses—often preventing $500K+ damage events from undetected whirl or stick-slip.
