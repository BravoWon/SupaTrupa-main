# Test Plan - Unified Activity:State Platform (SupaTrupa)

**Version:** 1.0
**Date:** 2026-02-09
**Status:** Development Complete (Cycles 0-11), ready for test team handoff

---

## Section A: Environment Setup

### A.1 Backend (Python)

```bash
# Activate virtual environment
cd backend
source .venv/bin/activate

# Install dependencies
pip install -e ".[all]"

# Start API server
uvicorn jones_framework.api.server:app --reload --port 8000
```

### A.2 Frontend (React/TypeScript)

```bash
# Use Node 20 via nvm
nvm use 20

# Enable pnpm via corepack
corepack enable

# Install dependencies
cd frontend
pnpm install

# Development server (http://localhost:3000)
pnpm dev

# Production build
pnpm build

# Type check
pnpm check
```

### A.3 Environment Variables

Create `frontend/client/.env`:
```
VITE_API_URL=http://localhost:8000
```

Backend environment (optional):
```
JONES_ENV=development
JONES_DEVICE=auto
JONES_LOG_LEVEL=INFO
```

### A.4 Health Check Verification

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","framework":"jones_framework","pointcloud_api":true}
```

---

## Section B: Developer Tests (Automated)

### B.1 Python Import Check

```bash
cd backend
source .venv/bin/activate
python3 -c "
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState
from jones_framework.perception.tda_pipeline import TDAPipeline
from jones_framework.perception.regime_classifier import RegimeClassifier
from jones_framework.sans.mixture_of_experts import MixtureOfExperts
from jones_framework.core.manifold_bridge import get_registry
print('All 6 core imports OK')
"
```
**Pass:** prints "All 6 core imports OK" with exit code 0.

### B.2 TypeScript Type Check

```bash
cd frontend
pnpm check
```
**Pass:** `tsc --noEmit` exits 0 with no errors.

### B.3 Production Build

```bash
cd frontend
pnpm build
```
**Pass:** Vite build succeeds; `dist/` directory created. Warning about bundle size (>1.5 MB) is expected and acceptable.

### B.4 Backend Pytest

```bash
cd backend
pytest -v
```
**Pass:** All tests pass.

---

## Section C: Consumer Tests - API Endpoints (45 endpoints)

**Prerequisites:** Backend running on `http://localhost:8000`.

The sample point cloud used across tests:

```json
{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}
```

The sample drilling records used across tests:

```json
{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]}
```

---

### C.1 Health & Status (2 endpoints)

**GET /health**
```bash
curl -s http://localhost:8000/health
```
Expected: `200` with `status`, `framework`, `pointcloud_api` fields.

**GET /api/v1/status**
```bash
curl -s http://localhost:8000/api/v1/status
```
Expected: `200` with `status`, `framework_available`, `active_regime`, `num_experts`, `uptime_seconds` fields.

---

### C.2 State & Regime (5 endpoints)

**POST /api/v1/state/create**
```bash
curl -s -X POST http://localhost:8000/api/v1/state/create \
  -H "Content-Type: application/json" \
  -d '{"vector": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```
Expected: `200` with `state_id`, `timestamp`, `vector`, `dimension` fields.

**POST /api/v1/state/market**
```bash
curl -s -X POST http://localhost:8000/api/v1/state/market \
  -H "Content-Type: application/json" \
  -d '{"price": 100.0, "volume": 5000, "bid": 99.5, "ask": 100.5, "symbol": "TEST"}'
```
Expected: `200` with `state_id`, `timestamp`, `vector`, `metadata` fields.

**GET /api/v1/regimes**
```bash
curl -s http://localhost:8000/api/v1/regimes
```
Expected: `200` with `regimes` array (16 regime names).

**GET /api/v1/regime**
```bash
curl -s http://localhost:8000/api/v1/regime
```
Expected: `200` with `regime`, `confidence`, `is_transition` fields.

**GET /api/v1/regime/history**
```bash
curl -s http://localhost:8000/api/v1/regime/history
```
Expected: `200` with `transitions` array of `{regime, timestamp, confidence}` objects.

---

### C.3 Classification & TDA (7 endpoints)

**POST /api/v1/classify**
```bash
curl -s -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}'
```
Expected: `200` with `regime_id`, `confidence`, `betti_0`, `betti_1`, `entropy_h1`, `features` fields.

**POST /api/v1/tda/features**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/features \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}'
```
Expected: `200` with 10 TDA features: `betti_0`, `betti_1`, `entropy_h0`, `entropy_h1`, `max_lifetime_h0`, `max_lifetime_h1`, `mean_lifetime_h0`, `mean_lifetime_h1`, `n_features_h0`, `n_features_h1`.

**POST /api/v1/tda/persistence-diagram**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/persistence-diagram \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}'
```
Expected: `200` with persistence diagram data (birth/death pairs).

**POST /api/v1/tda/full-signature**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/full-signature \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}'
```
Expected: `200` with full topological signature.

**POST /api/v1/tda/betti-curve**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/betti-curve \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}'
```
Expected: `200` with `h0` and `h1` Betti curve arrays.

**POST /api/v1/tda/windowed-signatures**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/windowed-signatures \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]], "window_size": 3, "step_size": 1}'
```
Expected: `200` with `signatures` array of windowed TDA snapshots.

**POST /api/v1/tda/change-detect**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/change-detect \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]], "window_size": 3}'
```
Expected: `200` with `change_detected`, `magnitude`, `threshold` fields.

---

### C.4 MoE (3 endpoints)

**POST /api/v1/moe/process**
```bash
curl -s -X POST http://localhost:8000/api/v1/moe/process \
  -H "Content-Type: application/json" \
  -d '{"state": {"vector": [15,120,45,8500,3200]}, "point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100]]}'
```
Expected: `200` with `output`, `active_regime`, `expert_used` fields.

**POST /api/v1/moe/hot-swap/{regime_name}**
```bash
curl -s -X POST http://localhost:8000/api/v1/moe/hot-swap/NORMAL
```
Expected: `200` with `success`, `active_regime` fields.

**GET /api/v1/moe/experts**
```bash
curl -s http://localhost:8000/api/v1/moe/experts
```
Expected: `200` with `experts` array of `{regime, description, is_active}` objects.

---

### C.5 Drilling (3 endpoints)

**POST /api/v1/drilling/ingest**
```bash
curl -s -X POST http://localhost:8000/api/v1/drilling/ingest \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]}'
```
Expected: `200` with `regime`, `confidence`, `betti_0`, `betti_1`, `color`, `recommendation` fields.

**GET /api/v1/drilling/metrics**
```bash
curl -s http://localhost:8000/api/v1/drilling/metrics
```
Expected: `200` with `current_regime`, `total_transitions`, `regime_counts`, `num_experts` fields.

**POST /api/v1/drilling/bha/recommend**
```bash
curl -s -X POST http://localhost:8000/api/v1/drilling/bha/recommend \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100}], "current_config": {"bitType":"PDC","motorBendAngle":1.5,"stabilizers":2,"rssType":"push-the-bit","flowRestrictor":0.5}}'
```
Expected: `200` with `regime`, `confidence`, `betti_numbers`, `suggestions`, `reasoning` fields.

---

### C.6 Parameter Resonance Network (1 endpoint)

**POST /api/v1/network/compute**
```bash
curl -s -X POST http://localhost:8000/api/v1/network/compute \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]}'
```
Expected: `200` with `nodes`, `edges`, `categories` fields. Nodes contain channel names, edges contain correlation strengths.

---

### C.7 Geometry / Sensitivity (3 endpoints)

**POST /api/v1/geometry/metric-field**
```bash
curl -s -X POST http://localhost:8000/api/v1/geometry/metric-field \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100}], "grid_size": 5}'
```
Expected: `200` with `points` array; each point has `x`, `y`, `metric_tensor`, `ricci_scalar`.

**POST /api/v1/geometry/geodesic**
```bash
curl -s -X POST http://localhost:8000/api/v1/geometry/geodesic \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100}], "start": [15, 120], "end": [20, 110]}'
```
Expected: `200` with `path` array of `[x, y]` points and `length`.

**POST /api/v1/geometry/curvature-field**
```bash
curl -s -X POST http://localhost:8000/api/v1/geometry/curvature-field \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100}], "grid_size": 5}'
```
Expected: `200` with `points` array; each point has `x`, `y`, `curvature`.

---

### C.8 Fingerprinting / Signature (4 endpoints)

**POST /api/v1/tda/fingerprint**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/fingerprint \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}'
```
Expected: `200` with `fingerprint` (10-dim vector), `regime_id`, `confidence` fields.

**POST /api/v1/tda/attribute**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/attribute \
  -H "Content-Type: application/json" \
  -d '{"point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100],[12,160,60,10000,3600],[16,130,48,8800,3300]]}'
```
Expected: `200` with `attributions` array of `{feature, value, percentage}` objects.

**POST /api/v1/tda/compare-regimes**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/compare-regimes \
  -H "Content-Type: application/json" \
  -d '{"point_cloud_a": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100]], "point_cloud_b": [[25,90,30,6000,2800],[28,95,35,6500,2900],[22,85,28,5800,2700]]}'
```
Expected: `200` with `regime_a`, `regime_b`, `fingerprint_a`, `fingerprint_b`, `distance` fields.

**GET /api/v1/tda/fingerprint-library**
```bash
curl -s http://localhost:8000/api/v1/tda/fingerprint-library
```
Expected: `200` with `library` object mapping regime names to reference fingerprints.

---

### C.9 Forecast (2 endpoints)

**POST /api/v1/tda/forecast**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/forecast \
  -H "Content-Type: application/json" \
  -d '{"signatures": [{"betti_0":3,"betti_1":1,"entropy_h0":0.5,"entropy_h1":0.3,"max_lifetime_h0":2.1,"max_lifetime_h1":1.2,"mean_lifetime_h0":0.8,"mean_lifetime_h1":0.4,"n_features_h0":5,"n_features_h1":2},{"betti_0":4,"betti_1":1,"entropy_h0":0.6,"entropy_h1":0.35,"max_lifetime_h0":2.3,"max_lifetime_h1":1.3,"mean_lifetime_h0":0.9,"mean_lifetime_h1":0.45,"n_features_h0":6,"n_features_h1":2}], "horizon": 3}'
```
Expected: `200` with `forecast` array of `{step, predicted, upper_band, lower_band}` objects, plus `velocity`, `acceleration`, `stability_index`.

**POST /api/v1/tda/transition-probability**
```bash
curl -s -X POST http://localhost:8000/api/v1/tda/transition-probability \
  -H "Content-Type: application/json" \
  -d '{"signatures": [{"betti_0":3,"betti_1":1,"entropy_h0":0.5,"entropy_h1":0.3,"max_lifetime_h0":2.1,"max_lifetime_h1":1.2,"mean_lifetime_h0":0.8,"mean_lifetime_h1":0.4,"n_features_h0":5,"n_features_h1":2},{"betti_0":4,"betti_1":1,"entropy_h0":0.6,"entropy_h1":0.35,"max_lifetime_h0":2.3,"max_lifetime_h1":1.3,"mean_lifetime_h0":0.9,"mean_lifetime_h1":0.45,"n_features_h0":6,"n_features_h1":2}]}'
```
Expected: `200` with `probabilities` object mapping regime names to probability values, `risk_level` field.

---

### C.10 Shadow / Dynamics (2 endpoints)

**POST /api/v1/shadow/embed**
```bash
curl -s -X POST http://localhost:8000/api/v1/shadow/embed \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]}'
```
Expected: `200` with `embedding` (3D coordinate array), `dimension`, `delay_lag` fields.

**POST /api/v1/shadow/attractor**
```bash
curl -s -X POST http://localhost:8000/api/v1/shadow/attractor \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]}'
```
Expected: `200` with `attractor_type` (one of: fixed_point, limit_cycle, strange_attractor, quasi_periodic, stochastic, transient), `lyapunov_exponent`, `correlation_dimension`, `rqa` (recurrence_rate, determinism, laminarity, trapping_time) fields.

---

### C.11 Advisory (2 endpoints)

**POST /api/v1/advisory/recommend**
```bash
curl -s -X POST http://localhost:8000/api/v1/advisory/recommend \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}], "target_regime": "OPTIMAL"}'
```
Expected: `200` with `steps` array of `{parameter, current, target, delta, unit, order}` objects, `current_regime`, `target_regime`, `path_efficiency` fields.

**POST /api/v1/advisory/risk**
```bash
curl -s -X POST http://localhost:8000/api/v1/advisory/risk \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}], "target_regime": "OPTIMAL"}'
```
Expected: `200` with `risk_factors` array of `{category, score, description}` objects, `overall_risk`, `mitigations`, `abort_conditions` fields.

---

### C.12 Field / Field Map (4 endpoints)

**POST /api/v1/field/register**
```bash
curl -s -X POST http://localhost:8000/api/v1/field/register \
  -H "Content-Type: application/json" \
  -d '{"well_name": "Well-A", "records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]}'
```
Expected: `200` with `well_id`, `well_name`, `fingerprint` (10-dim vector), `regime_distribution` fields.

**GET /api/v1/field/atlas**
```bash
curl -s http://localhost:8000/api/v1/field/atlas
```
Expected: `200` with `wells` array and `field_summary` object.

**POST /api/v1/field/compare**
```bash
curl -s -X POST http://localhost:8000/api/v1/field/compare \
  -H "Content-Type: application/json" \
  -d '{"well_a": "Well-A", "well_b": "Well-B"}'
```
Expected: `200` with `well_a`, `well_b`, `topological_distance`, `feature_deltas`, `regime_similarity` fields. Note: requires both wells to be registered first.

**POST /api/v1/field/pattern-search**
```bash
curl -s -X POST http://localhost:8000/api/v1/field/pattern-search \
  -H "Content-Type: application/json" \
  -d '{"query_fingerprint": [3,1,0.5,0.3,2.1,1.2,0.8,0.4,5,2], "top_k": 3}'
```
Expected: `200` with `matches` array of `{well_name, distance, fingerprint}` objects. Note: requires wells to be registered first.

---

### C.13 Dashboard (1 endpoint)

**POST /api/v1/dashboard/summary**
```bash
curl -s -X POST http://localhost:8000/api/v1/dashboard/summary \
  -H "Content-Type: application/json" \
  -d '{"records": [{"wob":15,"rpm":120,"rop":45,"torque":8500,"spp":3200},{"wob":18,"rpm":140,"rop":52,"torque":9200,"spp":3400},{"wob":20,"rpm":110,"rop":38,"torque":7800,"spp":3100},{"wob":12,"rpm":160,"rop":60,"torque":10000,"spp":3600},{"wob":16,"rpm":130,"rop":48,"torque":8800,"spp":3300}]}'
```
Expected: `200` with `regime`, `display_name`, `color`, `parameters`, `topology` (drilling_zones, coupling_loops, stability), `predictability_index`, `behavioral_consistency`, `transition_risk`, `advisory_step` fields.

---

### C.14 LAS File Operations (5 endpoints)

**Note:** LAS endpoints require a file upload first. All subsequent endpoints use the `file_id` returned from upload.

**POST /api/v1/las/upload**
```bash
curl -s -X POST http://localhost:8000/api/v1/las/upload \
  -F "file=@/path/to/your/file.las"
```
Expected: `200` with `file_id`, `filename`, `curves`, `depth_range` fields.

**POST /api/v1/las/{file_id}/curves**
```bash
curl -s -X POST http://localhost:8000/api/v1/las/{file_id}/curves \
  -H "Content-Type: application/json" \
  -d '{"curves": ["DEPT", "GR", "RHOB"]}'
```
Expected: `200` with `curves` object containing requested curve data arrays.

**POST /api/v1/las/{file_id}/map-to-drilling**
```bash
curl -s -X POST http://localhost:8000/api/v1/las/{file_id}/map-to-drilling \
  -H "Content-Type: application/json" \
  -d '{}'
```
Expected: `200` with `records` array of drilling record objects, `count`, `mapping` fields.

**GET /api/v1/las/files**
```bash
curl -s http://localhost:8000/api/v1/las/files
```
Expected: `200` with `files` array of uploaded LAS file summaries.

**POST /api/v1/las/{file_id}/analyze-window**
```bash
curl -s -X POST http://localhost:8000/api/v1/las/{file_id}/analyze-window \
  -H "Content-Type: application/json" \
  -d '{"start_index": 0, "end_index": 50}'
```
Expected: `200` with `records`, `count`, `regime`, `windowed_signatures`, `index_range` fields.

---

### C.15 WebSocket (1 endpoint)

**WS /api/v1/ws/stream**

Test with `websocat` or similar:
```bash
websocat ws://localhost:8000/api/v1/ws/stream
```

Send ping:
```json
{"type": "ping"}
```
Expected response: `{"type": "pong"}`

Send classification:
```json
{"type": "classify", "point_cloud": [[15,120,45,8500,3200],[18,140,52,9200,3400],[20,110,38,7800,3100]]}
```
Expected response: `{"type": "classification", "regime_id": "...", "confidence": ..., "betti_0": ..., "betti_1": ..., "timestamp": ...}`

---

## Section D: Consumer Tests - UI Tabs (12 tabs)

**Prerequisites:** Both backend (port 8000) and frontend (port 3000) running. Open `http://localhost:3000`.

### D.1 DASHBOARD (default landing tab)

- **Navigation:** Opens by default on page load
- **Elements:** 4-section grid with KPI tiles
- **Verify:** Regime display name, color-coded status, drilling zones count, coupling loops count, predictability index, transition risk level, recommended advisory step
- **Interactive:** Click any tile section header to navigate to the corresponding specialized tab

### D.2 CTS

- **Navigation:** Click "CTS" tab
- **Elements:** 3-panel layout (40/35/25% width)
  - Left: AttractorManifold (3D WebGL)
  - Center: PersistenceBarcode (Feature Lifetime Chart), KPICards, BettiTimeline
  - Right: TrustGauge (Automation Readiness), CTSPipelineBar
- **Verify:** 3D visualization renders, barcode bars visible, KPI values populated, gauge animates

### D.3 WELL PATH

- **Navigation:** Click "WELL PATH" tab
- **Elements:** 3D well trajectory visualization (Three.js/R3F)
- **Verify:** 3D trajectory renders with depth coloring

### D.4 WIRE MESH

- **Navigation:** Click "WIRE MESH" tab
- **Elements:** LASMeshViz (R3F wire mesh visualization)
- **Verify:** Wire mesh renders when LAS data loaded; empty state message if no LAS file

### D.5 NETWORK

- **Navigation:** Click "NETWORK" tab
- **Elements:** ParameterNetworkGraph (force-directed graph), ParameterDetailCard, NetworkStatsBar, ChannelSelector
- **Verify:** Nodes appear with labels, edges connect correlated parameters, channel presets (Co Man, DD, MWD, Gen Super, Office) switch visible channels

### D.6 SENSITIVITY (formerly GEOMETRY)

- **Navigation:** Click "SENSITIVITY" tab
- **Elements:** CurvatureField (canvas heatmap), GeodesicOverlay (SVG path)
- **Verify:** Heatmap renders with color scale (high curvature = red, low = blue), geodesic path drawn as overlay. Labels show "Parameter Sensitivity" not "Ricci Scalar"

### D.7 SIGNATURE (formerly FINGERPRINT)

- **Navigation:** Click "SIGNATURE" tab
- **Elements:** RegimeFingerprint (radar chart), AttributionBars (horizontal bars), RegimeCompare (dual radar)
- **Verify:** Radar chart has 10 axes matching TDA features, attribution bars show percentage breakdown, comparison panel shows two overlaid radar charts

### D.8 DYNAMICS (formerly SHADOW)

- **Navigation:** Click "DYNAMICS" tab
- **Elements:** DelayEmbedding (canvas 3D scatter), LyapunovIndicator (gauge + metrics)
- **Verify:** 3D scatter plot with auto-rotation, attractor type label (Stable/Cyclic/Complex/Multi-Cycle/Noisy/Transitioning), Predictability Index gauge, RQA metrics visible. Labels show "Hidden Dynamics" not "Delay Embedding"

### D.9 FORECAST

- **Navigation:** Click "FORECAST" tab
- **Elements:** TopologyForecast (SVG trajectory with confidence bands), TransitionRadar (polar probability chart)
- **Verify:** History line + forecast dashed line visible, confidence bands shade around forecast, polar chart shows regime transition probabilities with trending arrows

### D.10 ADVISORY

- **Navigation:** Click "ADVISORY" tab
- **Elements:** AdvisoryPanel (step-by-step prescription), GeodesicNavigator (SVG trajectory)
- **Verify:** Parameter steps listed with current/target/delta values, risk bars visible, mitigation suggestions listed, SVG path shows trajectory from current to target

### D.11 FIELD MAP (formerly FIELD)

- **Navigation:** Click "FIELD MAP" tab
- **Elements:** FieldAtlas (well grid with mini radars), WellCompare (butterfly bar chart)
- **Verify:** Multiple well cards shown with mini radar fingerprints, click two wells to see comparison butterfly bars, regime similarity and topological distance metrics displayed

### D.12 ANALYZER

- **Navigation:** Click "ANALYZER" tab (requires LAS file loaded)
- **Elements:** LASAnalyzer (dual-thumb slider, step/play controls), RegimeStrip (colored regime evolution bar)
- **Verify:** Depth slider moves window through LAS data, step buttons increment window position, play button auto-advances, regime strip colors update per window. No LAS → empty state prompt

---

## Section E: Known Limitations

1. **Bundle size:** ~1.7 MB JavaScript chunk. No code-splitting implemented yet. First load may be slow on constrained connections.

2. **ShadowTensorBuilder obfuscated constructor:** Uses mangled parameter names internally (`_fll1c29` for embedding_dim, `_fI1Oc2A` for delay_lag). Public API works correctly via aliases.

3. **`@bridge(metadata=...)` kwarg broken:** `manifold_bridge.py:398` references an obfuscated field name. Workaround: omit the `metadata=` keyword argument from `@bridge()` calls.

4. **LAS endpoints require file upload:** All `/las/{file_id}/...` endpoints fail with 404 if no LAS file has been uploaded in the current server session. Upload a `.las` file first.

5. **Field compare/pattern-search require well registration:** The `/field/compare` and `/field/pattern-search` endpoints return errors if the referenced wells have not been registered via `/field/register` in the current session.

6. **No persistent storage:** All server-side state (LAS files, registered wells, regime history) is in-memory. Restarting the backend clears everything.

7. **Single-user design:** No authentication, no multi-tenant support. The backend assumes a single concurrent user.

---

## Endpoint Summary

| Category | Count | Methods |
|----------|-------|---------|
| Health & Status | 2 | 2 GET |
| State & Regime | 5 | 2 POST, 3 GET |
| Classification & TDA | 7 | 7 POST |
| MoE | 3 | 2 POST, 1 GET |
| Drilling | 3 | 2 POST, 1 GET |
| Network | 1 | 1 POST |
| Geometry | 3 | 3 POST |
| Fingerprinting | 4 | 3 POST, 1 GET |
| Forecast | 2 | 2 POST |
| Shadow | 2 | 2 POST |
| Advisory | 2 | 2 POST |
| Field | 4 | 3 POST, 1 GET |
| Dashboard | 1 | 1 POST |
| LAS | 5 | 4 POST, 1 GET |
| WebSocket | 1 | 1 WS |
| **Total** | **45** | **34 POST, 9 GET, 1 WS, 1 multipart** |
