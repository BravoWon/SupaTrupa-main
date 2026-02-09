# SupaTrupa Beta v0.1.0 - Pre-Install Test Package

**Cognitive Topological System for Drilling Intelligence**

This is the pre-install beta release for test team evaluation. All 11 development cycles are complete and the system is ready for structured testing.

---

## Quick Start (5 minutes)

### Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.9+ (3.10+ recommended) | `python3 --version` |
| Node.js | 20+ | `node --version` |
| pnpm | 9+ | `pnpm --version` |
| git | any | `git --version` |

### One-Command Setup

```bash
# From the project root:
bash beta/setup.sh
```

This installs all dependencies and starts both servers. Once running:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **Health Check:** http://localhost:8000/health

### Manual Setup

If the setup script doesn't work on your system:

```bash
# Terminal 1: Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
uvicorn jones_framework.api.server:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
pnpm install
pnpm dev
```

---

## Verify Installation

```bash
# Quick smoke test (requires backend running)
bash beta/verify.sh
```

Or manually:
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","framework":true,"pointcloud_api":true}
```

---

## What to Test

### API Endpoints (45 total)

Run the full consumer test suite:
```bash
bash scripts/consumer-tests.sh
```

For detailed per-endpoint curl commands and expected responses, see:
**[docs/TEST_PLAN.md](../docs/TEST_PLAN.md)**

### UI Tabs (12 total)

Open http://localhost:3000 and verify each tab:

| Tab | What to Check |
|-----|---------------|
| **DASHBOARD** | Default landing page, 4-section status grid |
| **CTS** | 3D attractor manifold, barcode, gauges |
| **WELL PATH** | 3D trajectory visualization |
| **WIRE MESH** | Wire mesh (needs LAS file loaded) |
| **NETWORK** | Force-directed parameter graph, channel presets |
| **SENSITIVITY** | Curvature heatmap, geodesic path overlay |
| **SIGNATURE** | Radar fingerprint, attribution bars |
| **DYNAMICS** | 3D scatter, predictability gauge |
| **FORECAST** | Trajectory with confidence bands, polar chart |
| **ADVISORY** | Parameter prescription steps, risk assessment |
| **FIELD MAP** | Multi-well comparison grid |
| **ANALYZER** | LAS sliding window (needs LAS file loaded) |

---

## Test Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Full Test Plan | `docs/TEST_PLAN.md` | Complete test procedures for all 45 endpoints and 12 tabs |
| Consumer Tests | `scripts/consumer-tests.sh` | Automated 15-test smoke suite |
| Glossary | `docs/user-guides/05_GLOSSARY.md` | All drilling and system terminology |
| Cycle Manifest | `CYCLE_MANIFEST.md` | Development history and cycle details |

---

## Known Limitations

1. **Bundle size:** ~1.7 MB JS chunk (no code-splitting yet)
2. **No persistent storage:** Restarting backend clears all in-memory state
3. **Single-user:** No authentication or multi-tenant support
4. **LAS/ANALYZER/WIRE MESH tabs** require uploading a `.las` file first
5. **Field Map compare** requires registering wells first via the API
6. **First load** may be slow due to bundle size

---

## Reporting Issues

When reporting issues, include:
1. Which endpoint or tab
2. Steps to reproduce
3. Expected vs actual behavior
4. Browser console errors (F12 > Console)
5. Backend terminal output

---

## Architecture Overview

```
Frontend (React 19)          Backend (Python/FastAPI)
  12 UI Tabs                   45 REST/WS Endpoints
  42 Components                TDA Pipeline (Ripser)
  Force Simulation             Regime Classifier (16 regimes)
  Canvas/SVG/WebGL             Advisory Engine
       |                       Shadow Tensor Builder
       |--- REST/WebSocket ----|
       |    Port 3000          |    Port 8000
```

**Tech stack:** React 19, Three.js/R3F, Tailwind CSS, Radix UI | Python 3.10+, FastAPI, NumPy, SciPy, Ripser
