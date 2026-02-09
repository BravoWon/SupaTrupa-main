from __future__ import annotations
import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
try:
    from jones_framework.core.condition_state import ConditionState
    from jones_framework.core.activity_state import ActivityState, RegimeID
    from jones_framework.perception.tda_pipeline import TDAPipeline, PersistenceDiagram
    from jones_framework.perception.regime_classifier import RegimeClassifier, RegimeSignature
    from jones_framework.perception.parameter_correlation import (
        ParameterCorrelationEngine,
        PARAMETER_TAXONOMY,
    )
    from jones_framework.sans.mixture_of_experts import MixtureOfExperts
    from jones_framework.core.shadow_tensor import ShadowTensorBuilder, ShadowTensor
    from jones_framework.perception.topology_forecaster import TopologyForecaster
    from jones_framework.perception.advisory_engine import AdvisoryEngine
    from jones_framework.perception.field_atlas import FieldAtlas
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

try:
    import importlib.util as _geo_ilu
    import sys as _geo_sys

    _geo_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "domains", "drilling", "manifold_geometry.py")
    )
    _geo_spec = _geo_ilu.spec_from_file_location("manifold_geometry", _geo_path)
    _geo_mod = _geo_ilu.module_from_spec(_geo_spec)
    _geo_sys.modules["manifold_geometry"] = _geo_mod
    _geo_spec.loader.exec_module(_geo_mod)
    DrillingMetricField = _geo_mod.DrillingMetricField
    GeodesicSolver = _geo_mod.GeodesicSolver
    default_rop_function = _geo_mod.default_rop_function
    GEOMETRY_AVAILABLE = True
except Exception as _geo_e:
    logging.warning(f"Geometry module not available: {_geo_e}")
    GEOMETRY_AVAILABLE = False

# Point cloud API imports
try:
    from jones_framework.api.rest.point_cloud_routes import (
        router as pointcloud_router,
        initialize_pointcloud_routes
    )
    POINTCLOUD_API_AVAILABLE = True
except ImportError:
    POINTCLOUD_API_AVAILABLE = False

class _cl1O492(BaseModel):
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    domain: str = 'generic'

class _cOO1493(BaseModel):
    state_id: str
    timestamp: int
    vector: List[float]
    metadata: Dict[str, Any]
    dimension: int

class _cOl1494(BaseModel):
    price: float
    volume: float
    bid: float
    ask: float
    symbol: str

class _c1O0495(BaseModel):
    point_cloud: List[List[float]]

class _cOII496(BaseModel):
    regime_id: str
    confidence: float
    betti_0: int
    betti_1: int
    entropy_h1: float
    features: Dict[str, float]

class _c011497(BaseModel):
    state: _cl1O492
    point_cloud: Optional[List[List[float]]] = None
    auto_swap: bool = True

class _cII0498(BaseModel):
    output: List[float]
    active_regime: str
    expert_used: str

class _c1II499(BaseModel):
    betti_0: int
    betti_1: int
    entropy_h0: float
    entropy_h1: float
    max_lifetime_h0: float
    max_lifetime_h1: float
    mean_lifetime_h0: float
    mean_lifetime_h1: float
    n_features_h0: int
    n_features_h1: int

class _c0ll49A(BaseModel):
    status: str
    framework_available: bool
    active_regime: Optional[str]
    num_experts: int
    uptime_seconds: float

class _cIO049B:

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.tda_pipeline: Optional[TDAPipeline] = None
        self.classifier: Optional[RegimeClassifier] = None
        self.moe: Optional[MixtureOfExperts] = None
        self.correlation_engine: Optional['ParameterCorrelationEngine'] = None
        self.field_atlas: Optional['FieldAtlas'] = None
        self.active_connections: List[WebSocket] = []

    def _f0O049c(self):
        if FRAMEWORK_AVAILABLE:
            self.tda_pipeline = TDAPipeline(1)
            self.classifier = RegimeClassifier()
            self.moe = MixtureOfExperts(classifier=self.classifier)
            self.correlation_engine = ParameterCorrelationEngine(sample_rate_hz=1.0)
            self.field_atlas = FieldAtlas()

    def _f0ll49d(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()
app_state = _cIO049B()

@asynccontextmanager
async def _fO0I49E(_fI1l49f: FastAPI):
    logging.info('Initializing Jones Framework API...')
    app_state._f0O049c()
    # Initialize point cloud API
    if POINTCLOUD_API_AVAILABLE:
        logging.info('Initializing Point Cloud API...')
        initialize_pointcloud_routes()
    logging.info('Jones Framework API ready')
    yield
    logging.info('Shutting down Jones Framework API...')
_fI1l49f = FastAPI(title='Jones Framework API', description='REST and WebSocket API for State-Adaptive Computational Intelligence', version='1.0.0', lifespan=_fO0I49E)
_fI1l49f.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000', 'http://localhost:5173', '*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
from fastapi import APIRouter
router = APIRouter(prefix='/api/v1')

@router.get('/status', response_model=_c0ll49A)
async def _fOll4AO():
    active_regime = None
    num_experts = 0
    if app_state.moe:
        active_regime = app_state.moe.current_regime.name if app_state.moe.current_regime else None
        num_experts = len(app_state.moe.experts)
    return _c0ll49A(status='online', framework_available=FRAMEWORK_AVAILABLE, active_regime=active_regime, num_experts=num_experts, uptime_seconds=app_state._f0ll49d())

@router.post('/state/create', response_model=_cOO1493)
async def _fOOO4Al(_f0I04A2: _cl1O492):
    if not FRAMEWORK_AVAILABLE:
        raise HTTPException(status_code=503, detail='Framework not available')
    state = ConditionState.from_market(np.array(_f0I04A2.vector), metadata={**_f0I04A2.metadata, 'domain': _f0I04A2.domain})
    return _cOO1493(state_id=state.state_id, timestamp=state.timestamp, vector=list(state.vector), metadata=state.metadata, dimension=state.dimension)

@router.post('/state/market', response_model=_cOO1493)
async def _f1ll4A3(_f0I04A2: _cOl1494):
    if not FRAMEWORK_AVAILABLE:
        raise HTTPException(status_code=503, detail='Framework not available')
    state = ConditionState._f0I0c6d(price=_f0I04A2.price, volume=_f0I04A2.volume, bid=_f0I04A2.bid, ask=_f0I04A2.ask, symbol=_f0I04A2.symbol)
    return _cOO1493(state_id=state.state_id, timestamp=state.timestamp, vector=list(state.vector), metadata=state.metadata, dimension=state.dimension)

@router.post('/classify', response_model=_cOII496)
async def _f1OO4A4(_f0I04A2: _c1O0495):
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail='Framework not available')
    point_cloud = np.array(_f0I04A2.point_cloud)
    if len(point_cloud) < 3:
        raise HTTPException(status_code=400, detail='Point cloud must have at least 3 points')
    result = app_state.classifier.classify(point_cloud)
    features = app_state.tda_pipeline.extract_features(point_cloud)
    return _cOII496(regime_id=result['regime'], confidence=result['confidence'], betti_0=int(features.get('betti_0', 0)), betti_1=int(features.get('betti_1', 0)), entropy_h1=features.get('entropy_h1', 0.0), features=features)

class PersistencePairResponse(BaseModel):
    birth: float
    death: float


class PersistenceDiagramResponse(BaseModel):
    h0: List[PersistencePairResponse]
    h1: List[PersistencePairResponse]
    betti_0: int
    betti_1: int
    filtration_range: List[float]


@router.post('/tda/persistence-diagram', response_model=PersistenceDiagramResponse)
async def tda_persistence_diagram(_f0I04A2: _c1O0495):
    """Return raw H0/H1 birth-death pairs for barcode visualization."""
    if not FRAMEWORK_AVAILABLE or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail='Framework not available')
    point_cloud = np.array(_f0I04A2.point_cloud)
    if len(point_cloud) < 3:
        raise HTTPException(status_code=400, detail='Point cloud must have at least 3 points')
    diagram = app_state.tda_pipeline.compute_persistence(point_cloud)
    h0_pairs = []
    filt_min = float('inf')
    filt_max = float('-inf')
    for row in diagram.h0:
        b, d = float(row[0]), float(row[1])
        if not np.isinf(d):
            h0_pairs.append(PersistencePairResponse(birth=b, death=d))
            filt_min = min(filt_min, b)
            filt_max = max(filt_max, d)
        else:
            h0_pairs.append(PersistencePairResponse(birth=b, death=-1.0))
            filt_min = min(filt_min, b)
    h1_pairs = []
    for row in diagram.h1:
        b, d = float(row[0]), float(row[1])
        if not np.isinf(d):
            h1_pairs.append(PersistencePairResponse(birth=b, death=d))
            filt_min = min(filt_min, b)
            filt_max = max(filt_max, d)
    if filt_min == float('inf'):
        filt_min = 0.0
    if filt_max == float('-inf'):
        filt_max = 1.0
    return PersistenceDiagramResponse(
        h0=h0_pairs,
        h1=h1_pairs,
        betti_0=diagram.betti_0,
        betti_1=diagram.betti_1,
        filtration_range=[filt_min, filt_max],
    )


@router.post('/tda/features', response_model=_c1II499)
async def _fll14A5(_f0I04A2: _c1O0495):
    if not FRAMEWORK_AVAILABLE or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail='Framework not available')
    point_cloud = np.array(_f0I04A2.point_cloud)
    features = app_state.tda_pipeline.extract_features(point_cloud)
    return _c1II499(betti_0=int(features.get('betti_0', 0)), betti_1=int(features.get('betti_1', 0)), entropy_h0=features.get('entropy_h0', 0.0), entropy_h1=features.get('entropy_h1', 0.0), max_lifetime_h0=features.get('max_lifetime_h0', 0.0), max_lifetime_h1=features.get('max_lifetime_h1', 0.0), mean_lifetime_h0=features.get('mean_lifetime_h0', 0.0), mean_lifetime_h1=features.get('mean_lifetime_h1', 0.0), n_features_h0=int(features.get('n_features_h0', 0)), n_features_h1=int(features.get('n_features_h1', 0)))

@router.post('/moe/process', response_model=_cII0498)
async def _f0Il4A6(_f0I04A2: _c011497):
    if not FRAMEWORK_AVAILABLE or not app_state.moe:
        raise HTTPException(status_code=503, detail='Framework not available')
    state = ConditionState.from_market(np.array(_f0I04A2.state.vector), metadata=_f0I04A2.state.metadata)
    point_cloud = np.array(_f0I04A2.point_cloud) if _f0I04A2.point_cloud else None
    output, regime = app_state.moe.process(state, telemetry=point_cloud, auto_switch=_f0I04A2.auto_swap)
    return _cII0498(output=output.tolist(), active_regime=regime.name, expert_used=f'{regime.name}_Expert')

@router.post('/moe/hot-swap/{regime_name}')
async def _fII04A7(_f0ll4A8: str):
    if not FRAMEWORK_AVAILABLE or not app_state.moe:
        raise HTTPException(status_code=503, detail='Framework not available')
    try:
        regime = RegimeID[_f0ll4A8.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f'Unknown regime: {_f0ll4A8}')
    app_state.moe.switch_to_regime(regime)
    return {'success': True, 'active_regime': regime.name}

@router.get('/moe/experts')
async def _f10I4A9():
    if not FRAMEWORK_AVAILABLE or not app_state.moe:
        raise HTTPException(status_code=503, detail='Framework not available')
    experts = app_state.moe.list_experts()
    return {'experts': [{'regime': r.name, 'description': d, 'is_active': a} for r, d, a in experts]}

@router.get('/regimes')
async def _f0104AA():
    return {'regimes': [r.name for r in RegimeID] if FRAMEWORK_AVAILABLE else []}

# --- Regime status endpoints ---

@router.get('/regime')
async def get_current_regime():
    """Get current active regime and confidence."""
    if not FRAMEWORK_AVAILABLE or not app_state.moe:
        raise HTTPException(status_code=503, detail='Framework not available')
    regime = app_state.moe.current_regime
    history = app_state.moe.get_transition_history()
    last_confidence = history[-1][2] if history else 1.0
    is_transition = len(history) >= 2 and (history[-1][1] - history[-2][1]) < 5.0
    return {
        'regime': regime.name if regime else 'UNKNOWN',
        'confidence': last_confidence,
        'is_transition': is_transition,
    }

@router.get('/regime/history')
async def get_regime_history():
    """Get regime transition history."""
    if not FRAMEWORK_AVAILABLE or not app_state.moe:
        raise HTTPException(status_code=503, detail='Framework not available')
    history = app_state.moe.get_transition_history()
    return {
        'transitions': [
            {'regime': r.name, 'timestamp': int(t * 1000), 'confidence': c}
            for r, t, c in history
        ]
    }

# --- Drilling-specific endpoints ---

class DrillingIngestRequest(BaseModel):
    """Ingest drilling telemetry data."""
    records: List[Dict[str, float]]
    window_size: int = 20

class PersistenceDiagramField(BaseModel):
    h0: List[Dict[str, float]]
    h1: List[Dict[str, float]]
    filtration_range: List[float]


class DrillingRegimeResponse(BaseModel):
    regime: str
    confidence: float
    betti_0: int
    betti_1: int
    color: str
    recommendation: str
    persistence_diagram: Optional[PersistenceDiagramField] = None

REGIME_COLORS = {
    'NORMAL': 'GREEN',
    'OPTIMAL': 'GREEN',
    'DARCY_FLOW': 'GREEN',
    'NON_DARCY_FLOW': 'YELLOW',
    'FORMATION_CHANGE': 'YELLOW',
    'WHIRL': 'YELLOW',
    'BIT_BOUNCE': 'YELLOW',
    'STICK_SLIP': 'ORANGE',
    'PACKOFF': 'ORANGE',
    'TURBULENT': 'ORANGE',
    'MULTIPHASE': 'ORANGE',
    'TRANSITION': 'YELLOW',
    'KICK': 'RED',
    'WASHOUT': 'RED',
    'LOST_CIRCULATION': 'RED',
    'UNKNOWN': 'YELLOW',
}

REGIME_RECOMMENDATIONS = {
    'NORMAL': 'Continue current parameters. Drilling within normal envelope.',
    'OPTIMAL': 'Optimal drilling conditions. Maintain current parameters.',
    'DARCY_FLOW': 'Optimal flow regime. Maintain current mud weight and flow rate.',
    'NON_DARCY_FLOW': 'Non-Darcy flow detected. Monitor ECD and adjust flow rate if needed.',
    'STICK_SLIP': 'Reduce WOB by 10%. Torsional oscillation detected — check RPM and torque.',
    'WHIRL': 'Lateral vibration detected. Check BHA balance, consider adding stabilizer.',
    'BIT_BOUNCE': 'Axial instability detected. Reduce WOB and increase flow restrictor.',
    'PACKOFF': 'Restricted annulus detected. Increase flow rate, consider short trip.',
    'FORMATION_CHANGE': 'Strong ROP trend. Monitor for bit wear and formation change.',
    'TURBULENT': 'Turbulent flow detected. Reduce flow rate, adjust mud rheology.',
    'MULTIPHASE': 'Multiphase flow detected. Monitor gas levels and ECD closely.',
    'KICK': 'IMMEDIATE ACTION: Pull off bottom, check flow sensors. Possible kick.',
    'WASHOUT': 'Hole enlargement detected. Consider increasing mud weight or LCM.',
    'LOST_CIRCULATION': 'ALERT: Lost circulation detected. Prepare LCM pill, reduce pump rate.',
    'TRANSITION': 'Formation change detected. Prepare for parameter adjustment.',
}

@router.post('/drilling/ingest')
async def drilling_ingest(request: DrillingIngestRequest):
    """Ingest drilling records and classify regime."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail='Framework not available')

    records = request.records
    if len(records) < 3:
        raise HTTPException(status_code=400, detail='Need at least 3 records')

    # Extract drilling parameters as point cloud
    fields = ['wob', 'rpm', 'rop', 'torque', 'spp']
    point_cloud = []
    for r in records[-request.window_size:]:
        point = [r.get(f, 0.0) for f in fields]
        point_cloud.append(point)

    pc_array = np.array(point_cloud)
    result = app_state.classifier.classify(pc_array)
    features = app_state.tda_pipeline.extract_features(pc_array)

    regime_name = result['regime']
    confidence = result['confidence']

    # Determine color based on confidence and regime
    if confidence < 0.4:
        color = 'RED'
    elif confidence < 0.6:
        color = 'ORANGE'
    elif confidence < 0.8:
        color = 'YELLOW'
    else:
        color = REGIME_COLORS.get(regime_name, 'YELLOW')

    recommendation = REGIME_RECOMMENDATIONS.get(regime_name, 'Monitor parameters closely.')

    # Compute raw persistence diagram for barcode visualization
    persistence_diagram = None
    try:
        diagram = app_state.tda_pipeline.compute_persistence(pc_array)
        h0_pairs = []
        filt_min = float('inf')
        filt_max = float('-inf')
        for row in diagram.h0:
            b, d = float(row[0]), float(row[1])
            if not np.isinf(d):
                h0_pairs.append({'birth': b, 'death': d})
                filt_min = min(filt_min, b)
                filt_max = max(filt_max, d)
            else:
                h0_pairs.append({'birth': b, 'death': -1.0})
                filt_min = min(filt_min, b)
        h1_pairs = []
        for row in diagram.h1:
            b, d = float(row[0]), float(row[1])
            if not np.isinf(d):
                h1_pairs.append({'birth': b, 'death': d})
                filt_min = min(filt_min, b)
                filt_max = max(filt_max, d)
        if filt_min == float('inf'):
            filt_min = 0.0
        if filt_max == float('-inf'):
            filt_max = 1.0
        persistence_diagram = PersistenceDiagramField(
            h0=h0_pairs, h1=h1_pairs, filtration_range=[filt_min, filt_max]
        )
    except Exception:
        logger.debug("Persistence diagram computation failed", exc_info=True)

    return DrillingRegimeResponse(
        regime=regime_name,
        confidence=confidence,
        betti_0=int(features.get('betti_0', 0)),
        betti_1=int(features.get('betti_1', 0)),
        color=color,
        recommendation=recommendation,
        persistence_diagram=persistence_diagram,
    )

@router.get('/drilling/metrics')
async def drilling_metrics():
    """Get current drilling regime metrics and statistics."""
    if not FRAMEWORK_AVAILABLE or not app_state.moe:
        raise HTTPException(status_code=503, detail='Framework not available')
    stats = app_state.moe.get_statistics()
    return {
        'current_regime': stats.get('current_regime', 'UNKNOWN'),
        'total_transitions': stats.get('total_transitions', 0),
        'regime_counts': stats.get('regime_counts', {}),
        'num_experts': stats.get('num_experts', 0),
    }

class BHARecommendRequest(BaseModel):
    """Request BHA optimization recommendation."""
    records: List[Dict[str, float]]
    current_config: Dict[str, Any]

@router.post('/drilling/bha/recommend')
async def bha_recommend(request: BHARecommendRequest):
    """Recommend BHA configuration based on drilling data."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail='Framework not available')

    records = request.records
    if len(records) < 5:
        raise HTTPException(status_code=400, detail='Need at least 5 records for BHA analysis')

    fields = ['wob', 'rpm', 'rop', 'torque', 'spp']
    point_cloud = np.array([[r.get(f, 0.0) for f in fields] for r in records[-30:]])

    result = app_state.classifier.classify(point_cloud)
    features = app_state.tda_pipeline.extract_features(point_cloud)

    regime = result['regime']
    config = request.current_config
    suggestions = {}
    reasoning = []

    betti_1 = int(features.get('betti_1', 0))

    if betti_1 > 2:
        reasoning.append(f'High topological complexity (β₁={betti_1}) indicates oscillatory behavior.')
        if config.get('motorBendAngle', 1.5) > 1.0:
            suggestions['motorBendAngle'] = max(0.5, config.get('motorBendAngle', 1.5) - 0.5)
            reasoning.append('Reduce motor bend angle to decrease torsional oscillation.')
        suggestions['stabilizers'] = config.get('stabilizers', 2) + 1
        reasoning.append('Add stabilizer to dampen lateral vibration.')

    if regime in ('STICK_SLIP', 'KICK'):
        suggestions['flowRestrictor'] = min(100, config.get('flowRestrictor', 50) + 15)
        reasoning.append('Increase flow restrictor to reduce axial instability.')

    if regime == 'FORMATION_CHANGE' and betti_1 == 0:
        reasoning.append('Strong drilling performance. No BHA changes recommended.')

    if not reasoning:
        reasoning.append('Drilling parameters within normal envelope. Continue monitoring.')

    return {
        'regime': regime,
        'confidence': result['confidence'],
        'betti_numbers': {'b0': int(features.get('betti_0', 0)), 'b1': betti_1},
        'suggestions': suggestions,
        'reasoning': reasoning,
    }

# --- Parameter Resonance Network endpoints ---


class NetworkComputeRequest(BaseModel):
    """Compute cross-channel correlation network."""
    records: List[Dict[str, float]]
    channels: Optional[List[str]] = None
    window_size: int = 50
    correlation_threshold: float = 0.3


@router.post('/network/compute')
async def network_compute(request: NetworkComputeRequest):
    """Compute the parameter resonance network from drilling records."""
    if not FRAMEWORK_AVAILABLE or not app_state.correlation_engine:
        raise HTTPException(status_code=503, detail='Framework not available')

    if len(request.records) < 3:
        raise HTTPException(status_code=400, detail='Need at least 3 records')

    graph = app_state.correlation_engine.compute_network(
        records=request.records,
        channels=request.channels,
        correlation_threshold=request.correlation_threshold,
        window_size=request.window_size,
    )

    from dataclasses import asdict
    return {
        'nodes': [asdict(n) for n in graph.nodes],
        'edges': [asdict(e) for e in graph.edges],
        'strong_count': graph.strong_count,
        'anomaly_count': graph.anomaly_count,
        'system_health': graph.system_health,
        'computation_time_ms': graph.computation_time_ms,
    }


# --- Cycle 2: Topological Time Machine endpoints ---


class BettiCurveResponse(BaseModel):
    """Betti curve data for H0 and H1."""

    h0: Dict[str, List[float]]  # {curve: [...], t_values: [...]}
    h1: Dict[str, List[float]]


class WindowedSignatureEntry(BaseModel):
    """Single window's topological summary."""

    window_index: int
    betti_0: int
    betti_1: int
    entropy_h0: float
    entropy_h1: float
    total_persistence_h0: float
    total_persistence_h1: float


class WindowedSignatureResponse(BaseModel):
    """Per-window topological features across sliding windows."""

    windows: List[WindowedSignatureEntry]
    window_size: int
    stride: int
    num_windows: int


class ChangeDetectRequest(BaseModel):
    """Two point clouds for topological change detection."""

    point_cloud_a: List[List[float]]
    point_cloud_b: List[List[float]]
    threshold: float = 0.5


class ChangeDetectResponse(BaseModel):
    """Topological change detection result."""

    detected_change: bool
    change_magnitude: float
    betti_change: Dict[str, float]
    landscape_distance: Dict[str, float]
    silhouette_distance: Dict[str, float]


class FullSignatureResponse(BaseModel):
    """Summarized full topological signature."""

    betti_numbers: Dict[str, int]
    persistence_entropy: Dict[str, float]
    total_persistence: Dict[str, float]
    betti_curves: Dict[str, Dict[str, List[float]]]  # dim -> {curve, t_values}
    max_dimension: int
    num_points: int


@router.post("/tda/full-signature", response_model=FullSignatureResponse)
async def tda_full_signature(request: _c1O0495):
    """Compute complete topological signature of a point cloud."""
    if not FRAMEWORK_AVAILABLE or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail="Framework not available")
    point_cloud = np.array(request.point_cloud)
    if len(point_cloud) < 3:
        raise HTTPException(status_code=400, detail="Point cloud must have at least 3 points")
    sig = app_state.tda_pipeline.compute_full_signature(point_cloud, max_dim=1, resolution=50)
    betti_curves_out: Dict[str, Dict[str, List[float]]] = {}
    for dim, bc in sig.betti_curves.items():
        betti_curves_out[str(dim)] = {
            "curve": bc.curve.tolist(),
            "t_values": bc.t_values.tolist(),
        }
    return FullSignatureResponse(
        betti_numbers={str(k): v for k, v in sig.betti_numbers.items()},
        persistence_entropy={str(k): v for k, v in sig.persistence_entropy.items()},
        total_persistence={str(k): v for k, v in sig.total_persistence.items()},
        betti_curves=betti_curves_out,
        max_dimension=sig.max_dimension,
        num_points=sig.num_points,
    )


@router.post("/tda/betti-curve", response_model=BettiCurveResponse)
async def tda_betti_curve(request: _c1O0495):
    """Compute Betti curves (Betti number as function of filtration) for H0 and H1."""
    if not FRAMEWORK_AVAILABLE or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail="Framework not available")
    point_cloud = np.array(request.point_cloud)
    if len(point_cloud) < 3:
        raise HTTPException(status_code=400, detail="Point cloud must have at least 3 points")
    diagram = app_state.tda_pipeline.compute_persistence(point_cloud)
    bc_h0 = app_state.tda_pipeline.compute_betti_curve(diagram.h0, resolution=100)
    bc_h1 = app_state.tda_pipeline.compute_betti_curve(diagram.h1, resolution=100)
    return BettiCurveResponse(
        h0={"curve": bc_h0.curve.tolist(), "t_values": bc_h0.t_values.tolist()},
        h1={"curve": bc_h1.curve.tolist(), "t_values": bc_h1.t_values.tolist()},
    )


class WindowedSignatureRequest(BaseModel):
    """Request for windowed topological signatures."""

    point_cloud: List[List[float]]
    window_size: int = 50
    stride: int = 10


@router.post("/tda/windowed-signatures", response_model=WindowedSignatureResponse)
async def tda_windowed_signatures(request: WindowedSignatureRequest):
    """Compute topological signatures over sliding windows for temporal evolution."""
    if not FRAMEWORK_AVAILABLE or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail="Framework not available")
    point_cloud = np.array(request.point_cloud)
    if len(point_cloud) < request.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {request.window_size} points for windowed analysis",
        )
    sigs = app_state.tda_pipeline.compute_windowed_signature(
        point_cloud, window_size=request.window_size, stride=request.stride, max_dim=1
    )
    windows = []
    for i, sig in enumerate(sigs):
        windows.append(
            WindowedSignatureEntry(
                window_index=i,
                betti_0=sig.betti_numbers.get(0, 0),
                betti_1=sig.betti_numbers.get(1, 0),
                entropy_h0=sig.persistence_entropy.get(0, 0.0),
                entropy_h1=sig.persistence_entropy.get(1, 0.0),
                total_persistence_h0=sig.total_persistence.get(0, 0.0),
                total_persistence_h1=sig.total_persistence.get(1, 0.0),
            )
        )
    return WindowedSignatureResponse(
        windows=windows,
        window_size=request.window_size,
        stride=request.stride,
        num_windows=len(windows),
    )


@router.post("/tda/change-detect", response_model=ChangeDetectResponse)
async def tda_change_detect(request: ChangeDetectRequest):
    """Detect topological change between two point clouds."""
    if not FRAMEWORK_AVAILABLE or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail="Framework not available")
    pc_a = np.array(request.point_cloud_a)
    pc_b = np.array(request.point_cloud_b)
    if len(pc_a) < 3 or len(pc_b) < 3:
        raise HTTPException(status_code=400, detail="Both point clouds must have at least 3 points")
    sig_a = app_state.tda_pipeline.compute_full_signature(pc_a, max_dim=1, resolution=30)
    sig_b = app_state.tda_pipeline.compute_full_signature(pc_b, max_dim=1, resolution=30)
    changes = app_state.tda_pipeline.detect_topological_change(
        sig_a, sig_b, threshold=request.threshold
    )
    return ChangeDetectResponse(
        detected_change=changes["detected_change"],
        change_magnitude=changes["change_magnitude"],
        betti_change={str(k): float(v) for k, v in changes["betti_change"].items()},
        landscape_distance={str(k): float(v) for k, v in changes["landscape_distance"].items()},
        silhouette_distance={str(k): float(v) for k, v in changes["silhouette_distance"].items()},
    )


# --- Cycle 3: Manifold Geometry Engine endpoints ---


class MetricFieldRequest(BaseModel):
    """Compute metric tensor field over a grid."""

    t_range: List[float] = Field(..., min_length=2, max_length=2)
    d_range: List[float] = Field(..., min_length=2, max_length=2)
    resolution: int = 20


class MetricFieldPoint(BaseModel):
    """Single point in the metric field."""

    t: float
    d: float
    g_tt: float
    g_dd: float
    determinant: float
    ricci_scalar: float
    rop: float


class MetricFieldResponse(BaseModel):
    """Metric field grid response."""

    points: List[MetricFieldPoint]
    t_values: List[float]
    d_values: List[float]
    resolution: int


class GeodesicRequest(BaseModel):
    """Compute geodesic between two points on the drilling manifold."""

    start: List[float] = Field(..., min_length=2, max_length=2)
    end: List[float] = Field(..., min_length=2, max_length=2)
    n_steps: int = 100


class GeodesicResponse(BaseModel):
    """Geodesic path result."""

    path: List[List[float]]
    total_length: float
    start_rop: float
    end_rop: float
    start_curvature: float
    end_curvature: float


class CurvatureFieldRequest(BaseModel):
    """Compute curvature field from drilling data."""

    records: List[Dict[str, float]]
    t_range: Optional[List[float]] = None
    d_range: Optional[List[float]] = None
    resolution: int = 20


class CurvatureFieldResponse(BaseModel):
    """Curvature field response with per-point values."""

    points: List[MetricFieldPoint]
    t_values: List[float]
    d_values: List[float]
    resolution: int
    max_curvature: float
    min_curvature: float
    mean_curvature: float


def _build_rop_from_records(records: List[Dict[str, float]]) -> callable:
    """Build an ROP interpolation function from drilling records."""
    depths = [r.get("depth", 0.0) for r in records]
    rops = [r.get("rop", 50.0) for r in records]
    if len(depths) < 2:
        return default_rop_function

    depths_arr = np.array(depths)
    rops_arr = np.array(rops)
    sort_idx = np.argsort(depths_arr)
    depths_arr = depths_arr[sort_idx]
    rops_arr = rops_arr[sort_idx]

    def rop_fn(t: float, d: float) -> float:
        if d <= depths_arr[0]:
            return max(1.0, float(rops_arr[0]))
        if d >= depths_arr[-1]:
            return max(1.0, float(rops_arr[-1]))
        return max(1.0, float(np.interp(d, depths_arr, rops_arr)))

    return rop_fn


@router.post("/geometry/metric-field", response_model=MetricFieldResponse)
async def geometry_metric_field(request: MetricFieldRequest):
    """Compute metric tensor field over a (time, depth) grid."""
    if not GEOMETRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Geometry module not available")

    t_vals = np.linspace(request.t_range[0], request.t_range[1], request.resolution)
    d_vals = np.linspace(request.d_range[0], request.d_range[1], request.resolution)

    field = DrillingMetricField(default_rop_function)
    points = []
    for t in t_vals:
        for d in d_vals:
            g = field.metric_tensor(t, d)
            det = field.metric_determinant(t, d)
            R = field.ricci_scalar(t, d)
            rop = field.get_rop(t, d)
            points.append(
                MetricFieldPoint(
                    t=float(t),
                    d=float(d),
                    g_tt=float(g[0, 0]),
                    g_dd=float(g[1, 1]),
                    determinant=float(det),
                    ricci_scalar=float(R),
                    rop=float(rop),
                )
            )

    return MetricFieldResponse(
        points=points,
        t_values=t_vals.tolist(),
        d_values=d_vals.tolist(),
        resolution=request.resolution,
    )


@router.post("/geometry/geodesic", response_model=GeodesicResponse)
async def geometry_geodesic(request: GeodesicRequest):
    """Compute geodesic (shortest path) between two points on drilling manifold."""
    if not GEOMETRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Geometry module not available")

    field = DrillingMetricField(default_rop_function)
    solver = GeodesicSolver(field)

    start = tuple(request.start)
    end = tuple(request.end)
    path = solver.solve_bvp(start, end)

    # Compute total geodesic length
    total_length = 0.0
    for i in range(1, len(path)):
        total_length += field.metric_distance(path[i - 1][0], path[i - 1][1], path[i][0], path[i][1])

    return GeodesicResponse(
        path=[[float(p[0]), float(p[1])] for p in path],
        total_length=float(total_length),
        start_rop=float(field.get_rop(start[0], start[1])),
        end_rop=float(field.get_rop(end[0], end[1])),
        start_curvature=float(field.ricci_scalar(start[0], start[1])),
        end_curvature=float(field.ricci_scalar(end[0], end[1])),
    )


@router.post("/geometry/curvature-field", response_model=CurvatureFieldResponse)
async def geometry_curvature_field(request: CurvatureFieldRequest):
    """Compute curvature field from drilling records with data-driven ROP."""
    if not GEOMETRY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Geometry module not available")

    if len(request.records) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 records")

    rop_fn = _build_rop_from_records(request.records)
    field = DrillingMetricField(rop_fn)

    # Auto-range from data if not provided
    depths = [r.get("depth", 0.0) for r in request.records]
    if request.t_range:
        t_vals = np.linspace(request.t_range[0], request.t_range[1], request.resolution)
    else:
        t_vals = np.linspace(0, len(request.records) * 10, request.resolution)

    if request.d_range:
        d_vals = np.linspace(request.d_range[0], request.d_range[1], request.resolution)
    else:
        d_vals = np.linspace(max(0, min(depths)), max(depths), request.resolution)

    points = []
    curvatures = []
    for t in t_vals:
        for d in d_vals:
            g = field.metric_tensor(t, d)
            det = field.metric_determinant(t, d)
            R = field.ricci_scalar(t, d)
            rop = field.get_rop(t, d)
            curvatures.append(R)
            points.append(
                MetricFieldPoint(
                    t=float(t),
                    d=float(d),
                    g_tt=float(g[0, 0]),
                    g_dd=float(g[1, 1]),
                    determinant=float(det),
                    ricci_scalar=float(R),
                    rop=float(rop),
                )
            )

    return CurvatureFieldResponse(
        points=points,
        t_values=t_vals.tolist(),
        d_values=d_vals.tolist(),
        resolution=request.resolution,
        max_curvature=float(max(curvatures)) if curvatures else 0.0,
        min_curvature=float(min(curvatures)) if curvatures else 0.0,
        mean_curvature=float(np.mean(curvatures)) if curvatures else 0.0,
    )


# --- Cycle 4: Persistence Fingerprinting endpoints ---

FEATURE_NAMES = [
    "betti_0",
    "betti_1",
    "entropy_h0",
    "entropy_h1",
    "max_lifetime_h0",
    "max_lifetime_h1",
    "mean_lifetime_h0",
    "mean_lifetime_h1",
    "n_features_h0",
    "n_features_h1",
]


class FingerprintEntry(BaseModel):
    """Single regime fingerprint."""

    regime: str
    description: str
    feature_vector: List[float]
    feature_names: List[str]
    threshold: float


class FingerprintResponse(BaseModel):
    """Fingerprint match result for observed data."""

    matched_regime: str
    confidence: float
    is_transition: bool
    observed_features: Dict[str, float]
    matched_signature: Dict[str, float]
    feature_distances: Dict[str, float]
    top_drivers: List[Dict[str, Any]]
    all_distances: Dict[str, float]


class AttributionResponse(BaseModel):
    """Per-feature attribution showing which TDA dimensions drive classification."""

    regime: str
    confidence: float
    attributions: List[Dict[str, Any]]
    total_distance: float
    dominant_dimension: str
    interpretation: str


class RegimeCompareResponse(BaseModel):
    """Side-by-side comparison of two regime fingerprints."""

    regime_a: str
    regime_b: str
    features_a: Dict[str, float]
    features_b: Dict[str, float]
    feature_deltas: Dict[str, float]
    topological_distance: float
    discriminating_features: List[str]
    interpretation: str


@router.post("/tda/fingerprint", response_model=FingerprintResponse)
async def tda_fingerprint(request: _c1O0495):
    """Match observed point cloud to regime fingerprints and show driving features."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail="Framework not available")
    point_cloud = np.array(request.point_cloud)
    if len(point_cloud) < 3:
        raise HTTPException(status_code=400, detail="Point cloud must have at least 3 points")

    # Classify and get features
    result = app_state.classifier.classify(point_cloud)
    features = app_state.tda_pipeline.extract_features(point_cloud)

    regime_name = result["regime"]
    sig = app_state.classifier.signatures.get(result["regime_id"])

    # Build observed feature dict
    observed = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}

    # Build matched signature dict
    matched_sig = {}
    if sig is not None:
        for i, name in enumerate(FEATURE_NAMES):
            matched_sig[name] = float(sig.feature_vector[i])
    else:
        matched_sig = {name: 0.0 for name in FEATURE_NAMES}

    # Per-feature squared distance contribution
    norm_scale = app_state.classifier._norm_scale
    feature_dists = {}
    sq_contributions = []
    for i, name in enumerate(FEATURE_NAMES):
        obs_val = observed[name]
        sig_val = matched_sig[name]
        scale = float(norm_scale[i]) if norm_scale is not None else 1.0
        normed_diff = (obs_val - sig_val) / scale if scale > 1e-10 else 0.0
        sq = normed_diff**2
        feature_dists[name] = float(abs(normed_diff))
        sq_contributions.append((name, sq, obs_val, sig_val))

    total_sq = sum(c[1] for c in sq_contributions) + 1e-10
    top_drivers = sorted(sq_contributions, key=lambda x: x[1], reverse=True)[:5]
    top_drivers_out = [
        {
            "feature": name,
            "contribution_pct": round(sq / total_sq * 100, 1),
            "observed": obs,
            "signature": sig_v,
            "direction": "higher" if obs > sig_v else "lower",
        }
        for name, sq, obs, sig_v in top_drivers
    ]

    return FingerprintResponse(
        matched_regime=regime_name,
        confidence=result["confidence"],
        is_transition=result["is_transition"],
        observed_features=observed,
        matched_signature=matched_sig,
        feature_distances=feature_dists,
        top_drivers=top_drivers_out,
        all_distances=result["all_distances"],
    )


@router.post("/tda/attribute", response_model=AttributionResponse)
async def tda_attribute(request: _c1O0495):
    """Per-feature attribution: which TDA dimensions drive regime classification."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail="Framework not available")
    point_cloud = np.array(request.point_cloud)
    if len(point_cloud) < 3:
        raise HTTPException(status_code=400, detail="Point cloud must have at least 3 points")

    result = app_state.classifier.classify(point_cloud)
    features = app_state.tda_pipeline.extract_features(point_cloud)
    sig = app_state.classifier.signatures.get(result["regime_id"])
    norm_scale = app_state.classifier._norm_scale

    attributions = []
    total_sq = 0.0
    for i, name in enumerate(FEATURE_NAMES):
        obs = float(features.get(name, 0.0))
        sig_v = float(sig.feature_vector[i]) if sig else 0.0
        scale = float(norm_scale[i]) if norm_scale is not None else 1.0
        normed_diff = (obs - sig_v) / scale if scale > 1e-10 else 0.0
        sq = normed_diff**2
        total_sq += sq
        attributions.append(
            {
                "feature": name,
                "observed": obs,
                "signature": sig_v,
                "normalized_distance": round(abs(normed_diff), 4),
                "squared_contribution": round(sq, 6),
            }
        )

    # Compute contribution percentages
    for a in attributions:
        a["contribution_pct"] = round(a["squared_contribution"] / (total_sq + 1e-10) * 100, 1)

    # Sort by contribution
    attributions.sort(key=lambda x: x["squared_contribution"], reverse=True)
    dominant = attributions[0]["feature"] if attributions else "unknown"

    # Generate interpretation
    top3 = attributions[:3]
    interp_parts = []
    for a in top3:
        direction = "higher" if a["observed"] > a["signature"] else "lower"
        interp_parts.append(
            f'{a["feature"]} is {direction} than {result["regime"]} signature '
            f'({a["observed"]:.2f} vs {a["signature"]:.2f}, {a["contribution_pct"]}%)'
        )
    interpretation = "; ".join(interp_parts) if interp_parts else "Classification matches signature closely."

    return AttributionResponse(
        regime=result["regime"],
        confidence=result["confidence"],
        attributions=attributions,
        total_distance=float(np.sqrt(total_sq)),
        dominant_dimension=dominant,
        interpretation=interpretation,
    )


class RegimeCompareRequest(BaseModel):
    """Compare two regimes by name."""

    regime_a: str
    regime_b: str


@router.post("/tda/compare-regimes", response_model=RegimeCompareResponse)
async def tda_compare_regimes(request: RegimeCompareRequest):
    """Side-by-side comparison of two regime fingerprints."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail="Framework not available")

    # Resolve regime names
    try:
        rid_a = RegimeID[request.regime_a.upper()]
        rid_b = RegimeID[request.regime_b.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail="Unknown regime name")

    sig_a = app_state.classifier.signatures.get(rid_a)
    sig_b = app_state.classifier.signatures.get(rid_b)
    if not sig_a or not sig_b:
        raise HTTPException(status_code=404, detail="Regime signature not found")

    norm_scale = app_state.classifier._norm_scale
    features_a = {}
    features_b = {}
    deltas = {}
    total_sq = 0.0

    for i, name in enumerate(FEATURE_NAMES):
        va = float(sig_a.feature_vector[i])
        vb = float(sig_b.feature_vector[i])
        features_a[name] = va
        features_b[name] = vb
        deltas[name] = round(vb - va, 4)
        scale = float(norm_scale[i]) if norm_scale is not None else 1.0
        normed_diff = (va - vb) / scale if scale > 1e-10 else 0.0
        total_sq += normed_diff**2

    # Find features that discriminate most
    abs_deltas = [(name, abs(d)) for name, d in deltas.items()]
    abs_deltas.sort(key=lambda x: x[1], reverse=True)
    discriminating = [name for name, _ in abs_deltas[:3]]

    interpretation = (
        f"{request.regime_a} vs {request.regime_b}: "
        f"most different in {', '.join(discriminating)}. "
        f"Topological distance = {np.sqrt(total_sq):.3f}."
    )

    return RegimeCompareResponse(
        regime_a=request.regime_a.upper(),
        regime_b=request.regime_b.upper(),
        features_a=features_a,
        features_b=features_b,
        feature_deltas=deltas,
        topological_distance=float(np.sqrt(total_sq)),
        discriminating_features=discriminating,
        interpretation=interpretation,
    )


@router.get("/tda/fingerprint-library")
async def tda_fingerprint_library():
    """Return all stored regime fingerprint signatures."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail="Framework not available")

    library = []
    for regime_id, sig in app_state.classifier.signatures.items():
        library.append(
            {
                "regime": regime_id.name,
                "description": sig.description,
                "feature_vector": sig.feature_vector.tolist(),
                "feature_names": FEATURE_NAMES,
                "threshold": sig.threshold,
            }
        )
    return {"signatures": library, "count": len(library), "feature_names": FEATURE_NAMES}


# --- Cycle 5: Predictive Topology endpoints ---


class ForecastRequest(BaseModel):
    """Forecast topological state evolution from point cloud."""

    point_cloud: List[List[float]]
    window_size: int = 20
    stride: int = 5
    n_ahead: int = 5


class ForecastPointModel(BaseModel):
    """Single forecasted window."""

    window_index: int
    betti_0: float
    betti_1: float
    entropy_h0: float
    entropy_h1: float
    total_persistence_h0: float
    total_persistence_h1: float
    confidence_upper: Dict[str, float]
    confidence_lower: Dict[str, float]


class ForecastResponse(BaseModel):
    """Topology forecast result."""

    current: Dict[str, float]
    forecast: List[ForecastPointModel]
    velocity: Dict[str, float]
    acceleration: Dict[str, float]
    trend_direction: str
    stability_index: float
    n_windows_used: int
    n_ahead: int


class TransitionProbRequest(BaseModel):
    """Compute regime transition probabilities."""

    point_cloud: List[List[float]]
    window_size: int = 20
    stride: int = 5


class TransitionProbResponse(BaseModel):
    """Regime transition probability result."""

    current_regime: str
    probabilities: Dict[str, float]
    trending_toward: str
    trending_away: str
    velocity_magnitude: float
    estimated_windows_to_transition: Optional[int]
    risk_level: str


@router.post("/tda/forecast", response_model=ForecastResponse)
async def tda_forecast(request: ForecastRequest):
    """Forecast topological state N windows ahead from point cloud data."""
    if not FRAMEWORK_AVAILABLE or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail="Framework not available")

    point_cloud = np.array(request.point_cloud)
    if len(point_cloud) < request.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {request.window_size} points for windowed analysis",
        )

    # Compute windowed signatures via existing TDA pipeline
    sigs = app_state.tda_pipeline.compute_windowed_signature(
        point_cloud, window_size=request.window_size, stride=request.stride, max_dim=1
    )

    if len(sigs) < 4:
        raise HTTPException(
            status_code=400,
            detail="Not enough windows for forecasting (need at least 4)",
        )

    # Convert TopologicalSignature objects to feature dicts
    windowed_features = []
    for sig in sigs:
        windowed_features.append(
            {
                "betti_0": float(sig.betti_numbers.get(0, 0)),
                "betti_1": float(sig.betti_numbers.get(1, 0)),
                "entropy_h0": float(sig.persistence_entropy.get(0, 0.0)),
                "entropy_h1": float(sig.persistence_entropy.get(1, 0.0)),
                "total_persistence_h0": float(sig.total_persistence.get(0, 0.0)),
                "total_persistence_h1": float(sig.total_persistence.get(1, 0.0)),
            }
        )

    forecaster = TopologyForecaster(min_windows=4)
    result = forecaster.forecast_trajectory(windowed_features, n_ahead=request.n_ahead)

    forecast_out = []
    for fp in result.forecast:
        forecast_out.append(
            ForecastPointModel(
                window_index=fp.window_index,
                betti_0=round(fp.betti_0, 4),
                betti_1=round(fp.betti_1, 4),
                entropy_h0=round(fp.entropy_h0, 4),
                entropy_h1=round(fp.entropy_h1, 4),
                total_persistence_h0=round(fp.total_persistence_h0, 4),
                total_persistence_h1=round(fp.total_persistence_h1, 4),
                confidence_upper={k: round(v, 4) for k, v in fp.confidence_upper.items()},
                confidence_lower={k: round(v, 4) for k, v in fp.confidence_lower.items()},
            )
        )

    return ForecastResponse(
        current={k: round(v, 4) for k, v in result.current.items()},
        forecast=forecast_out,
        velocity={k: round(v, 6) for k, v in result.velocity.items()},
        acceleration={k: round(v, 6) for k, v in result.acceleration.items()},
        trend_direction=result.trend_direction,
        stability_index=round(result.stability_index, 4),
        n_windows_used=result.n_windows_used,
        n_ahead=result.n_ahead,
    )


@router.post("/tda/transition-probability", response_model=TransitionProbResponse)
async def tda_transition_probability(request: TransitionProbRequest):
    """Compute regime transition probabilities with velocity-adjusted forecasting."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier or not app_state.tda_pipeline:
        raise HTTPException(status_code=503, detail="Framework not available")

    point_cloud = np.array(request.point_cloud)
    if len(point_cloud) < request.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {request.window_size} points",
        )

    # Current TDA features (10-dim)
    features = app_state.tda_pipeline.extract_features(point_cloud)

    # Also compute windowed signatures for velocity
    velocity = None
    if len(point_cloud) >= request.window_size * 2:
        sigs = app_state.tda_pipeline.compute_windowed_signature(
            point_cloud, window_size=request.window_size, stride=request.stride, max_dim=1
        )
        if len(sigs) >= 4:
            windowed_features = []
            for sig in sigs:
                windowed_features.append(
                    {
                        "betti_0": float(sig.betti_numbers.get(0, 0)),
                        "betti_1": float(sig.betti_numbers.get(1, 0)),
                        "entropy_h0": float(sig.persistence_entropy.get(0, 0.0)),
                        "entropy_h1": float(sig.persistence_entropy.get(1, 0.0)),
                        "total_persistence_h0": float(sig.total_persistence.get(0, 0.0)),
                        "total_persistence_h1": float(sig.total_persistence.get(1, 0.0)),
                    }
                )
            forecaster = TopologyForecaster(min_windows=4)
            try:
                forecast = forecaster.forecast_trajectory(windowed_features, n_ahead=1)
                velocity = forecast.velocity
            except ValueError:
                pass

    # Build regime signature dict from classifier
    regime_sigs: Dict[str, np.ndarray] = {}
    for rid, sig in app_state.classifier.signatures.items():
        regime_sigs[rid.name] = sig.feature_vector

    norm_scale = app_state.classifier._norm_scale

    # Map velocity to 10-dim feature names if we have 6-dim velocity
    full_velocity = None
    if velocity:
        full_velocity = {
            "betti_0": velocity.get("betti_0", 0.0),
            "betti_1": velocity.get("betti_1", 0.0),
            "entropy_h0": velocity.get("entropy_h0", 0.0),
            "entropy_h1": velocity.get("entropy_h1", 0.0),
            "max_lifetime_h0": 0.0,
            "max_lifetime_h1": 0.0,
            "mean_lifetime_h0": 0.0,
            "mean_lifetime_h1": 0.0,
            "n_features_h0": velocity.get("betti_0", 0.0),
            "n_features_h1": velocity.get("betti_1", 0.0),
        }

    forecaster = TopologyForecaster()
    trans = forecaster.compute_transition_probabilities(
        current_features={name: float(features.get(name, 0.0)) for name in FEATURE_NAMES},
        regime_signatures=regime_sigs,
        norm_scale=norm_scale,
        velocity=full_velocity,
    )

    return TransitionProbResponse(
        current_regime=trans.current_regime,
        probabilities={k: round(v, 4) for k, v in trans.probabilities.items()},
        trending_toward=trans.trending_toward,
        trending_away=trans.trending_away,
        velocity_magnitude=round(trans.velocity_magnitude, 6),
        estimated_windows_to_transition=trans.estimated_windows_to_transition,
        risk_level=trans.risk_level,
    )


# --- Cycle 6: Shadow Tensor Integration endpoints ---


class ShadowEmbedRequest(BaseModel):
    """Build delay-coordinate embedding from drilling parameter time series."""

    records: List[Dict[str, float]]
    parameter: str = "rop"
    embedding_dim: int = 3
    delay_lag: int = 1


class ShadowEmbedResponse(BaseModel):
    """Delay-coordinate embedding result."""

    point_cloud: List[List[float]]
    metric_proxy: List[float]
    tangent_proxy: List[float]
    fractal_proxy: List[float]
    embedding_dim: int
    delay_lag: int
    n_points: int
    total_dimension: int


class AttractorRequest(BaseModel):
    """Compute attractor properties from drilling records."""

    records: List[Dict[str, float]]
    parameter: str = "rop"
    embedding_dim: int = 3
    delay_lag: int = 1
    recurrence_threshold: float = 0.1


class AttractorResponse(BaseModel):
    """Attractor analysis result."""

    lyapunov_exponent: float
    lyapunov_interpretation: str
    correlation_dimension: float
    recurrence_rate: float
    determinism: float
    attractor_type: str
    embedding_dim: int
    n_points: int
    laminarity: float
    trapping_time: float


def _extract_parameter_series(records: List[Dict[str, float]], parameter: str) -> np.ndarray:
    """Extract a named parameter time series from drilling records."""
    valid_params = {"wob", "rpm", "rop", "torque", "spp", "hookload", "depth"}
    if parameter not in valid_params:
        raise ValueError(f"Unknown parameter '{parameter}'. Valid: {', '.join(sorted(valid_params))}")
    series = np.array([r.get(parameter, 0.0) for r in records], dtype=np.float64)
    return series


def _estimate_lyapunov(embedded: np.ndarray, dt: float = 1.0) -> float:
    """Estimate largest Lyapunov exponent via Rosenstein's method."""
    n_points = len(embedded)
    if n_points < 10:
        return 0.0

    # Find nearest neighbors (excluding self and temporal neighbors)
    from scipy.spatial.distance import cdist

    dists = cdist(embedded, embedded)
    np.fill_diagonal(dists, np.inf)
    # Exclude temporal neighbors within +-2 steps
    for i in range(n_points):
        for j in range(max(0, i - 2), min(n_points, i + 3)):
            if i != j:
                dists[i, j] = np.inf

    nn_idx = np.argmin(dists, axis=1)
    nn_dist = dists[np.arange(n_points), nn_idx]

    # Track divergence over time steps
    max_steps = min(n_points // 4, 20)
    if max_steps < 2:
        return 0.0

    divergence = np.zeros(max_steps)
    counts = np.zeros(max_steps)

    for i in range(n_points):
        j = nn_idx[i]
        if nn_dist[i] < 1e-10:
            continue
        for k in range(max_steps):
            ii = i + k
            jj = j + k
            if ii >= n_points or jj >= n_points:
                break
            d = np.linalg.norm(embedded[ii] - embedded[jj])
            if d > 1e-10:
                divergence[k] += np.log(d)
                counts[k] += 1

    # Compute mean log divergence
    valid = counts > 0
    if np.sum(valid) < 2:
        return 0.0

    mean_div = np.zeros(max_steps)
    mean_div[valid] = divergence[valid] / counts[valid]

    # Linear fit to get slope = largest Lyapunov exponent
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return 0.0

    t = valid_idx * dt
    y = mean_div[valid_idx]
    coeffs = np.polyfit(t, y, 1)
    return float(coeffs[0])


def _correlation_dimension(embedded: np.ndarray, n_radii: int = 15) -> float:
    """Estimate correlation dimension via Grassberger-Procaccia algorithm."""
    n_points = len(embedded)
    if n_points < 10:
        return 0.0

    from scipy.spatial.distance import pdist

    distances = pdist(embedded)
    if len(distances) == 0:
        return 0.0

    d_min = np.percentile(distances[distances > 0], 5) if np.any(distances > 0) else 1e-10
    d_max = np.percentile(distances, 95)
    if d_max <= d_min:
        return 0.0

    radii = np.logspace(np.log10(max(d_min, 1e-10)), np.log10(d_max), n_radii)
    n_pairs = len(distances)

    log_r = []
    log_c = []
    for r in radii:
        count = np.sum(distances < r)
        if count > 0:
            c_r = count / n_pairs
            log_r.append(np.log(r))
            log_c.append(np.log(c_r))

    if len(log_r) < 3:
        return 0.0

    # Linear fit in the scaling region (middle portion)
    log_r = np.array(log_r)
    log_c = np.array(log_c)
    n = len(log_r)
    start = n // 4
    end = 3 * n // 4 + 1
    if end - start < 2:
        start = 0
        end = n

    coeffs = np.polyfit(log_r[start:end], log_c[start:end], 1)
    return float(max(0.0, coeffs[0]))


def _recurrence_analysis(
    embedded: np.ndarray, threshold_frac: float = 0.1
) -> Dict[str, float]:
    """Recurrence quantification analysis (RQA)."""
    n_points = len(embedded)
    if n_points < 5:
        return {"recurrence_rate": 0.0, "determinism": 0.0, "laminarity": 0.0, "trapping_time": 0.0}

    from scipy.spatial.distance import cdist

    dists = cdist(embedded, embedded)
    threshold = threshold_frac * np.max(dists)
    recurrence = (dists < threshold).astype(int)
    np.fill_diagonal(recurrence, 0)

    total_points = n_points * (n_points - 1)
    rr = float(np.sum(recurrence)) / total_points if total_points > 0 else 0.0

    # Determinism: fraction of recurrence points on diagonal lines (length >= 2)
    diag_count = 0
    total_diag = 0
    for offset in range(1, n_points):
        diag = np.diag(recurrence, offset)
        runs = []
        current_run = 0
        for v in diag:
            if v == 1:
                current_run += 1
            else:
                if current_run >= 2:
                    runs.append(current_run)
                current_run = 0
        if current_run >= 2:
            runs.append(current_run)
        diag_count += sum(runs)
        total_diag += int(np.sum(diag))

    det = float(diag_count) / max(total_diag, 1)

    # Laminarity: fraction on vertical lines (length >= 2)
    vert_count = 0
    total_vert = 0
    for col in range(n_points):
        column = recurrence[:, col]
        runs = []
        current_run = 0
        for v in column:
            if v == 1:
                current_run += 1
            else:
                if current_run >= 2:
                    runs.append(current_run)
                current_run = 0
        if current_run >= 2:
            runs.append(current_run)
        vert_count += sum(runs)
        total_vert += int(np.sum(column))

    lam = float(vert_count) / max(total_vert, 1)

    # Trapping time: average length of vertical structures
    vert_lengths = []
    for col in range(n_points):
        column = recurrence[:, col]
        current_run = 0
        for v in column:
            if v == 1:
                current_run += 1
            else:
                if current_run >= 2:
                    vert_lengths.append(current_run)
                current_run = 0
        if current_run >= 2:
            vert_lengths.append(current_run)

    tt = float(np.mean(vert_lengths)) if vert_lengths else 0.0

    return {"recurrence_rate": rr, "determinism": det, "laminarity": lam, "trapping_time": tt}


def _classify_attractor(lyap: float, corr_dim: float, det: float) -> str:
    """Classify attractor type from dynamical invariants."""
    if lyap < -0.01:
        return "fixed_point"
    elif abs(lyap) <= 0.01 and det > 0.8:
        return "limit_cycle"
    elif lyap > 0.01 and corr_dim > 1.5 and det < 0.8:
        return "strange_attractor"
    elif lyap > 0.01 and det > 0.5:
        return "quasi_periodic"
    elif det < 0.3:
        return "stochastic"
    else:
        return "transient"


@router.post("/shadow/embed", response_model=ShadowEmbedResponse)
async def shadow_embed(request: ShadowEmbedRequest):
    """Build delay-coordinate embedding from a drilling parameter time series."""
    if not FRAMEWORK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Framework not available")

    if len(request.records) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 records")

    try:
        series = _extract_parameter_series(request.records, request.parameter)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    builder = ShadowTensorBuilder(
        _fll1c29=request.embedding_dim,
        _fI1Oc2A=request.delay_lag,
    )

    try:
        shadow = builder.build_from_numpy(series)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ShadowEmbedResponse(
        point_cloud=shadow.point_cloud.tolist(),
        metric_proxy=shadow.metric_proxy.tolist(),
        tangent_proxy=shadow.tangent_proxy.tolist(),
        fractal_proxy=shadow.fractal_proxy.tolist(),
        embedding_dim=request.embedding_dim,
        delay_lag=request.delay_lag,
        n_points=len(shadow.point_cloud),
        total_dimension=shadow.dimension,
    )


@router.post("/shadow/attractor", response_model=AttractorResponse)
async def shadow_attractor(request: AttractorRequest):
    """Compute attractor properties from delay-coordinate embedded drilling data."""
    if not FRAMEWORK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Framework not available")

    if len(request.records) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 records")

    try:
        series = _extract_parameter_series(request.records, request.parameter)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    builder = ShadowTensorBuilder(
        _fll1c29=request.embedding_dim,
        _fI1Oc2A=request.delay_lag,
    )

    try:
        shadow = builder.build_from_numpy(series)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    embedded = shadow.point_cloud

    # Compute dynamical invariants
    lyap = _estimate_lyapunov(embedded)
    corr_dim = _correlation_dimension(embedded)
    rqa = _recurrence_analysis(embedded, request.recurrence_threshold)

    attractor_type = _classify_attractor(lyap, corr_dim, rqa["determinism"])

    # Interpretation
    if lyap < -0.01:
        lyap_interp = "Stable (converging to fixed point)"
    elif abs(lyap) <= 0.01:
        lyap_interp = "Neutral (periodic or quasi-periodic)"
    elif lyap < 0.1:
        lyap_interp = "Weakly chaotic (early warning)"
    else:
        lyap_interp = "Strongly chaotic (unpredictable dynamics)"

    return AttractorResponse(
        lyapunov_exponent=round(lyap, 6),
        lyapunov_interpretation=lyap_interp,
        correlation_dimension=round(corr_dim, 3),
        recurrence_rate=round(rqa["recurrence_rate"], 4),
        determinism=round(rqa["determinism"], 4),
        attractor_type=attractor_type,
        embedding_dim=request.embedding_dim,
        n_points=len(embedded),
        laminarity=round(rqa["laminarity"], 4),
        trapping_time=round(rqa["trapping_time"], 3),
    )


# --- Cycle 7: Autonomous Advisory endpoints ---


class AdvisoryRecommendRequest(BaseModel):
    """Compute geodesic-optimal parameter prescription."""

    current_params: Dict[str, float]
    current_regime: str
    target_regime: str


class ParameterStepModel(BaseModel):
    """Single step in the advisory prescription."""

    step_index: int
    parameter: str
    current_value: float
    target_value: float
    change_amount: float
    change_pct: float
    priority: int
    rationale: str


class AdvisoryRecommendResponse(BaseModel):
    """Advisory recommendation result."""

    current_regime: str
    target_regime: str
    confidence: float
    steps: List[ParameterStepModel]
    geodesic_length: float
    euclidean_length: float
    path_efficiency: float
    risk_score: float
    risk_level: str
    estimated_transitions: int
    reasoning: List[str]
    parameter_trajectory: List[Dict[str, float]]


class AdvisoryRiskRequest(BaseModel):
    """Assess risk of proposed parameter changes."""

    current_params: Dict[str, float]
    proposed_changes: Dict[str, float]
    current_regime: str


class RiskFactorModel(BaseModel):
    """Single risk factor."""

    factor: str
    value: str
    risk: float


class AdvisoryRiskResponse(BaseModel):
    """Risk assessment result."""

    overall_risk: float
    risk_level: str
    risk_factors: List[Dict[str, Any]]
    mitigations: List[str]
    abort_conditions: List[str]
    regime_risk: float
    path_risk: float
    correlation_risk: float


@router.post("/advisory/recommend", response_model=AdvisoryRecommendResponse)
async def advisory_recommend(request: AdvisoryRecommendRequest):
    """Compute geodesic-optimal parameter prescription from current to target regime."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail="Framework not available")

    # Build regime signatures from classifier
    regime_sigs: Dict[str, np.ndarray] = {}
    for rid, sig in app_state.classifier.signatures.items():
        regime_sigs[rid.name] = sig.feature_vector

    # Get correlation edges if we have data from the correlation engine
    correlation_edges = None
    if app_state.correlation_engine:
        try:
            # Build synthetic records from current_params for correlation
            records = []
            for _ in range(20):
                rec = dict(request.current_params)
                # Add noise to get correlation structure
                for k in rec:
                    rec[k] = rec[k] * (1.0 + np.random.normal(0, 0.02))
                records.append(rec)
            graph = app_state.correlation_engine.compute_network(
                records=records, correlation_threshold=0.3, window_size=20
            )
            correlation_edges = [asdict(e) for e in graph.edges]
        except Exception:
            logger.debug("Advisory correlation computation failed", exc_info=True)

    # Get transition probabilities if possible
    transition_probs = None

    engine = AdvisoryEngine(regime_signatures=regime_sigs)
    result = engine.compute_advisory(
        current_params=request.current_params,
        current_regime=request.current_regime.upper(),
        target_regime=request.target_regime.upper(),
        correlation_edges=correlation_edges,
        transition_probs=transition_probs,
    )

    steps_out = [
        ParameterStepModel(
            step_index=s.step_index,
            parameter=s.parameter,
            current_value=s.current_value,
            target_value=s.target_value,
            change_amount=s.change_amount,
            change_pct=s.change_pct,
            priority=s.priority,
            rationale=s.rationale,
        )
        for s in result.steps
    ]

    return AdvisoryRecommendResponse(
        current_regime=result.current_regime,
        target_regime=result.target_regime,
        confidence=result.confidence,
        steps=steps_out,
        geodesic_length=result.geodesic_length,
        euclidean_length=result.euclidean_length,
        path_efficiency=result.path_efficiency,
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        estimated_transitions=result.estimated_transitions,
        reasoning=result.reasoning,
        parameter_trajectory=result.parameter_trajectory,
    )


@router.post("/advisory/risk", response_model=AdvisoryRiskResponse)
async def advisory_risk(request: AdvisoryRiskRequest):
    """Assess risk of a proposed set of parameter changes."""
    if not FRAMEWORK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Framework not available")

    # Get correlation edges if possible
    correlation_edges = None
    if app_state.correlation_engine:
        try:
            records = []
            for _ in range(20):
                rec = dict(request.current_params)
                for k in rec:
                    rec[k] = rec[k] * (1.0 + np.random.normal(0, 0.02))
                records.append(rec)
            graph = app_state.correlation_engine.compute_network(
                records=records, correlation_threshold=0.3, window_size=20
            )
            correlation_edges = [asdict(e) for e in graph.edges]
        except Exception:
            logger.debug("Risk correlation computation failed", exc_info=True)

    engine = AdvisoryEngine()
    result = engine.assess_risk(
        current_params=request.current_params,
        proposed_changes=request.proposed_changes,
        current_regime=request.current_regime.upper(),
        correlation_edges=correlation_edges,
    )

    return AdvisoryRiskResponse(
        overall_risk=result.overall_risk,
        risk_level=result.risk_level,
        risk_factors=result.risk_factors,
        mitigations=result.mitigations,
        abort_conditions=result.abort_conditions,
        regime_risk=result.regime_risk,
        path_risk=result.path_risk,
        correlation_risk=result.correlation_risk,
    )


# --- Cycle 8: Field-Level Intelligence endpoints ---


class FieldRegisterRequest(BaseModel):
    """Register a well in the field atlas."""

    name: str
    records: List[Dict[str, float]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WellEntryModel(BaseModel):
    """Single well in the atlas."""

    well_id: str
    name: str
    depth_min: float
    depth_max: float
    num_records: int
    feature_vector: List[float]
    regime: str
    confidence: float
    regime_distribution: Dict[str, float]
    windowed_betti: List[Dict[str, Any]]


class FieldAtlasResponse(BaseModel):
    """Complete field atlas."""

    wells: List[WellEntryModel]
    well_count: int
    field_summary: Dict[str, Any]


class FieldCompareRequest(BaseModel):
    """Compare two wells by ID."""

    well_id_a: str
    well_id_b: str


class FieldCompareResponse(BaseModel):
    """Well comparison result."""

    well_a: str
    well_b: str
    topological_distance: float
    feature_deltas: Dict[str, float]
    discriminating_features: List[str]
    regime_similarity: float
    depth_overlap: float
    interpretation: str


class PatternSearchRequest(BaseModel):
    """Search for wells matching a topological pattern."""

    query_features: List[float] = Field(..., min_length=10, max_length=10)
    top_k: int = 5


class PatternMatchModel(BaseModel):
    """Single pattern match."""

    well_id: str
    name: str
    distance: float
    regime: str
    confidence: float
    feature_vector: List[float]


@router.post("/field/register")
async def field_register(request: FieldRegisterRequest):
    """Register a well by computing its topological signature from drilling records."""
    if not FRAMEWORK_AVAILABLE or not app_state.field_atlas:
        raise HTTPException(status_code=503, detail="Framework not available")
    if len(request.records) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 records")

    try:
        entry = app_state.field_atlas.register_well(
            name=request.name,
            records=request.records,
            tda_pipeline=app_state.tda_pipeline,
            classifier=app_state.classifier,
            metadata=request.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "well_id": entry.well_id,
        "name": entry.name,
        "regime": entry.regime,
        "confidence": entry.confidence,
        "depth_range": [entry.depth_min, entry.depth_max],
        "num_records": entry.num_records,
        "feature_vector": entry.feature_vector,
    }


@router.get("/field/atlas", response_model=FieldAtlasResponse)
async def field_atlas():
    """Get all registered wells and field-wide summary."""
    if not FRAMEWORK_AVAILABLE or not app_state.field_atlas:
        raise HTTPException(status_code=503, detail="Framework not available")

    wells = app_state.field_atlas.get_atlas()
    summary = app_state.field_atlas.get_field_summary()

    return FieldAtlasResponse(
        wells=[
            WellEntryModel(
                well_id=w.well_id,
                name=w.name,
                depth_min=w.depth_min,
                depth_max=w.depth_max,
                num_records=w.num_records,
                feature_vector=w.feature_vector,
                regime=w.regime,
                confidence=w.confidence,
                regime_distribution=w.regime_distribution,
                windowed_betti=w.windowed_betti,
            )
            for w in wells
        ],
        well_count=len(wells),
        field_summary=summary,
    )


@router.post("/field/compare", response_model=FieldCompareResponse)
async def field_compare(request: FieldCompareRequest):
    """Compare two wells by topological signature."""
    if not FRAMEWORK_AVAILABLE or not app_state.field_atlas:
        raise HTTPException(status_code=503, detail="Framework not available")

    norm_scale = None
    if app_state.classifier:
        norm_scale = app_state.classifier._norm_scale

    try:
        result = app_state.field_atlas.compare_wells(
            request.well_id_a, request.well_id_b, norm_scale=norm_scale
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return FieldCompareResponse(
        well_a=result.well_a,
        well_b=result.well_b,
        topological_distance=result.topological_distance,
        feature_deltas=result.feature_deltas,
        discriminating_features=result.discriminating_features,
        regime_similarity=result.regime_similarity,
        depth_overlap=result.depth_overlap,
        interpretation=result.interpretation,
    )


@router.post("/field/pattern-search")
async def field_pattern_search(request: PatternSearchRequest):
    """Find wells matching a topological pattern query."""
    if not FRAMEWORK_AVAILABLE or not app_state.field_atlas:
        raise HTTPException(status_code=503, detail="Framework not available")

    norm_scale = None
    if app_state.classifier:
        norm_scale = app_state.classifier._norm_scale

    matches = app_state.field_atlas.search_patterns(
        query_features=request.query_features,
        top_k=request.top_k,
        norm_scale=norm_scale,
    )

    return {
        "matches": [
            {
                "well_id": m.well_id,
                "name": m.name,
                "distance": m.distance,
                "regime": m.regime,
                "confidence": m.confidence,
                "feature_vector": m.feature_vector,
            }
            for m in matches
        ],
        "count": len(matches),
    }


@router.websocket('/ws/stream')
async def _f0OO4AB(_fl1l4Ac: WebSocket):
    await _fl1l4Ac.accept()
    app_state.active_connections.append(_fl1l4Ac)
    try:
        while True:
            data = await _fl1l4Ac.receive_json()
            if data.get('type') == 'classify':
                point_cloud = np.array(data.get('point_cloud', []))
                if len(point_cloud) >= 3 and app_state.classifier:
                    result = app_state.classifier.classify(point_cloud)
                    features = app_state.tda_pipeline.extract_features(point_cloud)
                    await _fl1l4Ac.send_json({'type': 'classification', 'regime_id': result['regime'], 'confidence': result['confidence'], 'betti_0': int(features.get('betti_0', 0)), 'betti_1': int(features.get('betti_1', 0)), 'timestamp': datetime.utcnow().isoformat()})
            elif data.get('type') == 'process':
                vector = data.get('vector', [])
                if len(vector) > 0 and app_state.moe:
                    state = ConditionState.from_market(np.array(vector))
                    output, regime = app_state.moe.process(state)
                    await _fl1l4Ac.send_json({'type': 'processed', 'output': output.tolist(), 'regime': regime.name, 'timestamp': datetime.utcnow().isoformat()})
            elif data.get('type') == 'ping':
                await _fl1l4Ac.send_json({'type': 'pong'})
    except WebSocketDisconnect:
        app_state.active_connections.remove(_fl1l4Ac)

async def _fOII4Ad(_f0lI4AE: str, _f11l4Af: float):
    message = {'type': 'regime_change', 'regime': _f0lI4AE, 'confidence': _f11l4Af, 'timestamp': datetime.utcnow().isoformat()}
    for connection in app_state.active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            logger.debug("WebSocket send failed", exc_info=True)

# --- Cycle 10: Master Dashboard composite endpoint ---


class DashboardSummaryRequest(BaseModel):
    """Request a unified dashboard summary from drilling records."""

    records: List[Dict[str, float]]
    window_size: int = 30


REGIME_DISPLAY_NAMES = {
    "NORMAL": "Normal Drilling",
    "OPTIMAL": "Optimal Drilling",
    "DARCY_FLOW": "Laminar Flow",
    "NON_DARCY_FLOW": "Non-Darcy Flow",
    "TURBULENT": "Turbulent Flow",
    "MULTIPHASE": "Multiphase Flow",
    "BIT_BOUNCE": "Bit Bounce",
    "PACKOFF": "Pack-Off",
    "STICK_SLIP": "Stick-Slip Vibration",
    "WHIRL": "Lateral Whirl",
    "FORMATION_CHANGE": "Formation Change",
    "WASHOUT": "Hole Washout",
    "LOST_CIRCULATION": "Lost Circulation",
    "KICK": "Kick Detected",
    "TRANSITION": "Regime Transition",
    "UNKNOWN": "Unknown",
}


@router.post("/dashboard/summary")
async def dashboard_summary(request: DashboardSummaryRequest):
    """Composite endpoint producing operator-friendly dashboard metrics."""
    if not FRAMEWORK_AVAILABLE or not app_state.classifier:
        raise HTTPException(status_code=503, detail="Framework not available")
    records = request.records
    if len(records) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 records")

    fields = ["wob", "rpm", "rop", "torque", "spp"]
    pc = [[r.get(f, 0.0) for f in fields] for r in records[-request.window_size :]]
    pc_array = np.array(pc)

    # 1) Regime classification + TDA features
    result = app_state.classifier.classify(pc_array)
    features = app_state.tda_pipeline.extract_features(pc_array)
    regime_name = result["regime"]
    confidence = result["confidence"]
    if confidence < 0.4:
        color = "RED"
    elif confidence < 0.6:
        color = "ORANGE"
    elif confidence < 0.8:
        color = "YELLOW"
    else:
        color = REGIME_COLORS.get(regime_name, "YELLOW")

    # Latest drilling params
    last = records[-1]
    rop = last.get("rop", 0.0)
    wob = last.get("wob", 0.0)
    rpm = last.get("rpm", 0.0)
    torque = last.get("torque", 0.0)
    spp = last.get("spp", 0.0)

    # 2) Attractor analysis for predictability
    predictability_index = 0.5
    behavioral_consistency = 0.5
    try:
        if len(records) >= 20:
            rop_series = np.array([r.get("rop", 0.0) for r in records])
            builder = ShadowTensorBuilder(_fll1c29=3, _fI1Oc2A=1)
            shadow = builder.build_from_numpy(rop_series)
            cloud = shadow.point_cloud
            if len(cloud) >= 10:
                lyap = _estimate_lyapunov(cloud)
                # Map Lyapunov to predictability: negative=predictable, positive=chaotic
                predictability_index = float(max(0, min(1, 1.0 - (lyap + 0.5))))
                # RQA determinism
                rqa = _recurrence_analysis(cloud, threshold_frac=0.1)
                behavioral_consistency = float(rqa.get("determinism", 0.5))
    except Exception:
        logger.debug("Dashboard attractor analysis failed", exc_info=True)

    # 3) Transition probabilities
    trending_toward = regime_name
    transition_risk = "low"
    est_windows = None
    try:
        if len(pc_array) >= 20:
            forecaster = TopologyForecaster()
            sigs = app_state.tda_pipeline.compute_windowed_signature(
                pc_array, window_size=min(20, len(pc_array)), stride=5, max_dim=1
            )
            if len(sigs) >= 3:
                trans = forecaster.compute_transition_probabilities(sigs)
                trending_toward = trans.get("trending_toward", regime_name)
                transition_risk = trans.get("risk_level", "low")
                est_windows = trans.get("estimated_windows_to_transition")
    except Exception:
        logger.debug("Dashboard transition probability failed", exc_info=True)

    # 4) Top advisory step
    top_advisory = None
    advisory_risk = None
    try:
        engine = AdvisoryEngine()
        current_params = {"wob": wob, "rpm": rpm, "rop": rop, "torque": torque, "spp": spp}
        target = "OPTIMAL" if regime_name != "OPTIMAL" else "NORMAL"
        adv = engine.compute_advisory(current_params, regime_name, target)
        if adv.steps:
            step = adv.steps[0]
            direction = "Increase" if step.change_amount > 0 else "Reduce"
            top_advisory = f"{direction} {step.parameter.upper()} by {abs(step.change_amount):.1f}"
        advisory_risk = adv.risk_level
    except Exception:
        logger.debug("Dashboard advisory computation failed", exc_info=True)

    return {
        "regime": regime_name,
        "regime_display": REGIME_DISPLAY_NAMES.get(regime_name, regime_name),
        "confidence": confidence,
        "color": color,
        "recommendation": REGIME_RECOMMENDATIONS.get(regime_name, "Monitor parameters closely."),
        "rop": rop,
        "wob": wob,
        "rpm": rpm,
        "torque": torque,
        "spp": spp,
        "drilling_zones": int(features.get("betti_0", 0)),
        "coupling_loops": int(features.get("betti_1", 0)),
        "signature_stability": float(features.get("entropy_h1", 0.0)),
        "predictability_index": predictability_index,
        "behavioral_consistency": behavioral_consistency,
        "trending_toward": REGIME_DISPLAY_NAMES.get(trending_toward, trending_toward),
        "transition_risk": transition_risk,
        "estimated_windows_to_transition": est_windows,
        "top_advisory": top_advisory,
        "advisory_risk": advisory_risk,
    }

_fI1l49f.include_router(router)

# Include point cloud routes
if POINTCLOUD_API_AVAILABLE:
    _fI1l49f.include_router(pointcloud_router)

# --- LAS File API ---

try:
    import importlib.util as _ilu
    import sys as _sys

    _las_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "domains", "las_parser.py")
    )
    _las_spec = _ilu.spec_from_file_location("las_parser", _las_path)
    _las_mod = _ilu.module_from_spec(_las_spec)
    _sys.modules["las_parser"] = _las_mod  # dataclass needs this
    _las_spec.loader.exec_module(_las_mod)
    parse_las_metadata = _las_mod.parse_las_metadata
    read_las_window = _las_mod.read_las_window
    map_to_drilling_records = _las_mod.map_to_drilling_records
    LAS_AVAILABLE = True
except Exception as _e:
    logging.warning(f"LAS parser not available: {_e}")
    LAS_AVAILABLE = False

LAS_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "las_uploads")
os.makedirs(LAS_UPLOAD_DIR, exist_ok=True)

# Track uploaded file metadata
_las_file_registry: Dict[str, Dict[str, Any]] = {}

las_router = APIRouter(prefix="/api/v1/las")


class CurveDataRequest(BaseModel):
    curves: List[str]
    start_index: Optional[float] = None
    end_index: Optional[float] = None
    max_points: int = 5000


class MapToDrillingRequest(BaseModel):
    curve_mapping: Dict[str, str]
    start_index: Optional[float] = None
    end_index: Optional[float] = None
    max_points: int = 5000


@las_router.post("/upload")
async def las_upload(file: UploadFile = File(...)):
    """Upload LAS file and return metadata + curve catalog."""
    if not LAS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LAS parser not available")

    if not file.filename or not file.filename.lower().endswith(".las"):
        raise HTTPException(status_code=400, detail="File must have .las extension")

    # Save to temp directory
    dest = os.path.join(LAS_UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        metadata = parse_las_metadata(dest)
    except Exception as e:
        os.remove(dest)
        raise HTTPException(status_code=400, detail=f"Failed to parse LAS file: {str(e)}")

    # Store path in registry
    _las_file_registry[metadata.file_id] = {
        "filepath": dest,
        "filename": file.filename,
    }

    # Convert dataclass to dict for JSON serialization
    curves_list = []
    for c in metadata.curves:
        curves_list.append(
            {
                "mnemonic": c.mnemonic,
                "unit": c.unit,
                "description": c.description,
                "category": c.category.value,
                "is_numeric": c.is_numeric,
                "null_pct": c.null_pct,
                "min_val": c.min_val,
                "max_val": c.max_val,
                "auto_map_field": c.auto_map_field,
            }
        )

    return {
        "file_id": metadata.file_id,
        "status": "ready",
        "metadata": {
            "well_name": metadata.well_name,
            "company": metadata.company,
            "index_type": metadata.index_type,
            "index_unit": metadata.index_unit,
            "index_min": metadata.index_min,
            "index_max": metadata.index_max,
            "num_rows": metadata.num_rows,
            "num_curves": len(metadata.curves),
            "curves": curves_list,
        },
    }


@las_router.post("/{file_id}/curves")
async def las_curves(file_id: str, request: CurveDataRequest):
    """Fetch windowed curve data for selected mnemonics."""
    if not LAS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LAS parser not available")

    reg = _las_file_registry.get(file_id)
    if not reg:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    try:
        result = read_las_window(
            reg["filepath"],
            request.curves,
            start=request.start_index,
            end=request.end_index,
            max_points=request.max_points,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read curves: {str(e)}")

    return result


@las_router.post("/{file_id}/map-to-drilling")
async def las_map_to_drilling(file_id: str, request: MapToDrillingRequest):
    """Auto-map LAS curves to DrillingRecord[], return mapped records + optional regime classification."""
    if not LAS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LAS parser not available")

    reg = _las_file_registry.get(file_id)
    if not reg:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    try:
        records = map_to_drilling_records(
            reg["filepath"],
            request.curve_mapping,
            start=request.start_index,
            end=request.end_index,
            max_points=request.max_points,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to map curves: {str(e)}")

    # Optionally classify regime if we have enough records
    regime_result = None
    if len(records) >= 10 and FRAMEWORK_AVAILABLE and app_state.classifier:
        fields = ["wob", "rpm", "rop", "torque", "spp"]
        pc = [[r.get(f, 0.0) for f in fields] for r in records[-30:]]
        pc_array = np.array(pc)
        try:
            result = app_state.classifier.classify(pc_array)
            features = app_state.tda_pipeline.extract_features(pc_array)
            regime_name = result["regime"]
            confidence = result["confidence"]
            if confidence < 0.4:
                color = "RED"
            elif confidence < 0.6:
                color = "ORANGE"
            elif confidence < 0.8:
                color = "YELLOW"
            else:
                color = REGIME_COLORS.get(regime_name, "YELLOW")

            regime_result = {
                "regime": regime_name,
                "confidence": confidence,
                "color": color,
                "betti_0": int(features.get("betti_0", 0)),
                "betti_1": int(features.get("betti_1", 0)),
                "recommendation": REGIME_RECOMMENDATIONS.get(
                    regime_name, "Monitor parameters closely."
                ),
            }
        except Exception:
            logger.debug("LAS regime classification failed", exc_info=True)

    return {
        "records": records,
        "count": len(records),
        "regime": regime_result,
    }


@las_router.get("/files")
async def las_list_files():
    """List all uploaded LAS files with metadata summaries."""
    files = []
    for fid, reg in _las_file_registry.items():
        filepath = reg["filepath"]
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            files.append(
                {
                    "file_id": fid,
                    "filename": reg["filename"],
                    "size_mb": round(size_mb, 2),
                }
            )
    return {"files": files}


class LASAnalyzeWindowRequest(BaseModel):
    """Analyze a depth window of a loaded LAS file with full TDA pipeline."""

    curve_mapping: Dict[str, str]
    start_index: float
    end_index: float
    max_points: int = 2000
    tda_window_size: int = 20
    tda_stride: int = 5


@las_router.post("/{file_id}/analyze-window")
async def las_analyze_window(file_id: str, request: LASAnalyzeWindowRequest):
    """Composite endpoint: read LAS window -> map to drilling -> classify + windowed TDA."""
    if not LAS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LAS parser not available")

    reg = _las_file_registry.get(file_id)
    if not reg:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

    try:
        records = map_to_drilling_records(
            reg["filepath"],
            request.curve_mapping,
            start=request.start_index,
            end=request.end_index,
            max_points=request.max_points,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read window: {str(e)}")

    # Regime classification
    regime_result = None
    windowed_sigs = []
    if len(records) >= 10 and FRAMEWORK_AVAILABLE and app_state.classifier:
        fields = ["wob", "rpm", "rop", "torque", "spp"]
        pc = [[r.get(f, 0.0) for f in fields] for r in records]
        pc_array = np.array(pc)
        try:
            result = app_state.classifier.classify(pc_array[-30:] if len(pc_array) > 30 else pc_array)
            features = app_state.tda_pipeline.extract_features(
                pc_array[-30:] if len(pc_array) > 30 else pc_array
            )
            regime_name = result["regime"]
            confidence = result["confidence"]
            if confidence < 0.4:
                color = "RED"
            elif confidence < 0.6:
                color = "ORANGE"
            elif confidence < 0.8:
                color = "YELLOW"
            else:
                color = REGIME_COLORS.get(regime_name, "YELLOW")
            regime_result = {
                "regime": regime_name,
                "confidence": confidence,
                "color": color,
                "betti_0": int(features.get("betti_0", 0)),
                "betti_1": int(features.get("betti_1", 0)),
                "recommendation": REGIME_RECOMMENDATIONS.get(regime_name, "Monitor parameters closely."),
            }
        except Exception:
            logger.debug("LAS window classification failed", exc_info=True)

        # Windowed TDA signatures
        if len(pc_array) >= request.tda_window_size:
            try:
                sigs = app_state.tda_pipeline.compute_windowed_signature(
                    pc_array,
                    window_size=request.tda_window_size,
                    stride=request.tda_stride,
                    max_dim=1,
                )
                for i, sig in enumerate(sigs):
                    windowed_sigs.append(
                        {
                            "window_index": i,
                            "betti_0": sig.betti_numbers.get(0, 0),
                            "betti_1": sig.betti_numbers.get(1, 0),
                            "entropy_h0": sig.persistence_entropy.get(0, 0.0),
                            "entropy_h1": sig.persistence_entropy.get(1, 0.0),
                            "total_persistence_h0": sig.total_persistence.get(0, 0.0),
                            "total_persistence_h1": sig.total_persistence.get(1, 0.0),
                        }
                    )
            except Exception:
                logger.debug("LAS windowed TDA failed", exc_info=True)

    return {
        "records": records,
        "count": len(records),
        "regime": regime_result,
        "windowed_signatures": windowed_sigs,
        "index_range": [request.start_index, request.end_index],
    }


_fI1l49f.include_router(las_router)

@_fI1l49f.get('/health')
async def _fI0O4BO():
    return {
        'status': 'healthy',
        'framework': FRAMEWORK_AVAILABLE,
        'pointcloud_api': POINTCLOUD_API_AVAILABLE
    }
# Public API alias
app = _fI1l49f

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(_fI1l49f, host='0.0.0.0', port=8000)