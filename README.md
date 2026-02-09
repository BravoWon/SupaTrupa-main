# Unified Activity:State Platform

**Multi-Parameter Drilling Analysis & Monitoring Platform**

A full-stack system combining the Jones Framework analytical backend with real-time dashboard visualizations for drilling operations monitoring.

---

## Features

### Analytical Engine (Python Backend)
- **ConditionState/ActivityState Primitives**: Immutable data structures with configurable metric tensors
- **TDA Pipeline**: Persistent homology via Ripser for regime classification
- **SANS Architecture**: Mixture of Experts with LoRA adapter switching
- **NLP Event Detection**: Text-based triggers for regime shift identification
- **Hardware Support**: GPU/NPU detection (CUDA, Metal, CPU fallback)

### Visualization Layer (React Frontend)
- **Master Dashboard**: Operational overview with regime, topology, dynamics, and advisory status
- **Operator Interface**: ISO 11064-compliant dark mode with attractor display, persistence barcode, and trust gauge
- **Regime Classifier Display**: Regime classification with Betti number readouts
- **12 Specialized Tabs**: Dashboard, CTS, Well Path, Wire Mesh, Network, Sensitivity, Signature, Dynamics, Forecast, Advisory, Field Map, Analyzer
- **Frontend Stack**: React 19, Radix UI, Tailwind CSS, Framer Motion

### Integration
- **REST/WebSocket API**: FastAPI bridge connecting Python and React
- **Shared Type System**: JSON Schemas validated across Python and TypeScript
- **Streaming Updates**: WebSocket-based regime updates and alerts

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

### Analysis Framework

The system models drilling operations as existing in distinct operational regimes (Activity:States), where parameter relationships and sensitivities differ between regimes.

- **ConditionState**: Immutable, timestamped observation records
- **ActivityState**: Operational regimes with associated parameter weighting
- **Optimal Paths**: Computed parameter trajectories between states based on regime geometry

### Topological Data Analysis (TDA)

The system uses persistent homology to characterize regime structure from multi-parameter data:

| Betti Number | Mathematical Meaning | Drilling Interpretation | Operator Term |
|--------------|----------------------|------------------------|---------------|
| β₀ | Connected components | Distinct drilling zones | Drilling Zones |
| β₁ | Loops/cycles | Coupled parameter oscillations | Coupling Loops |
| β₂ | Voids | Gaps in parameter space coverage | - |

### SANS Architecture (Mixture of Experts)

Specialized analytical models selected per regime:

1. **Regime Detection**: TDA pipeline classifies current operational state
2. **Expert Selection**: Routes to regime-specific analytical model
3. **Adapter Switching**: LoRA adapters allow model switching without full reload
4. **Continuity Validation**: Checks for valid transitions between states

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
- **Bestest-main** (jones_framework): Multi-parameter analysis framework
- **activity-state-dashboard**: Drilling monitoring platform beta v1.0

---

## Les Logic: BHA Optimization Methodology

The **Les Logic** methodology applies to Bottom Hole Assembly (BHA) optimization within this platform. It uses multi-parameter topological analysis to support drilling dysfunction detection and BHA configuration recommendations.

**For comprehensive documentation, see: [docs/LES_LOGIC_BHA_OPTIMIZATION.md](docs/LES_LOGIC_BHA_OPTIMIZATION.md)**

### Quick Overview

Les Logic supplements conventional threshold-based monitoring with topological analysis of multi-parameter drilling data. Where conventional methods check individual parameters against fixed limits, Les Logic analyzes the relationships between parameters to identify regime changes.

**Conventional Approach:**
```
Sensor Data → Threshold Check → Alert
```

**Les Logic Approach:**
```
Sensor Data → Topological Analysis → Regime Detection → BHA Recommendation
```

### Design Principles

| Principle | Application to BHA Optimization |
|-----------|--------------------------------|
| **Physical Grounding** | Every measurement connects to physical BHA behavior -- torque causes rotation, WOB causes penetration |
| **Multi-Parameter Analysis** | Drilling parameter space has non-linear relationships: ROP sensitivity varies depending on operating point |
| **Operator-Readable Output** | Results are actionable: "Add 1 stabilizer" rather than raw Betti numbers |

### Key Capabilities

1. **Regime Classification via Persistent Homology**
   - Analyzes drilling data as multi-dimensional point clouds
   - Computes Betti numbers to identify oscillatory behavior (stick-slip, whirl, bit bounce)
   - May provide earlier indication of regime transitions compared to single-parameter thresholds

2. **Rule-Based Recommendations**
   | Detected Regime | Root Cause | Les Logic Recommendation |
   |-----------------|------------|--------------------------|
   | Whirl | BHA eccentricity | Add stabilizer |
   | Stick-Slip | Torsional resonance | Reduce motor bend angle |
   | Bit Bounce | Axial instability | Increase flow restrictor |

3. **Parameter Space Modeling**
   - Models drilling parameter interactions using metric tensors
   - Computes suggested parameter trajectories between operating states
   - Curvature mapping indicates regions of high parameter sensitivity

### Rationale

Conventional single-parameter threshold methods treat drilling parameters as independent variables. In practice:

- Parameters are **coupled** (WOB affects torque affects ROP affects vibration)
- Regime transitions can be **non-linear** (small changes can produce large effects near critical points)
- Multi-parameter patterns (the *shape* of data in phase space) may carry information that individual parameter values do not

Les Logic uses Topological Data Analysis (TDA) to characterize these multi-parameter patterns, with the goal of supplementing conventional monitoring.
