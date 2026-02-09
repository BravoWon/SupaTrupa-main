from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import json
import asyncio
import threading
from functools import wraps
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import RegimeID
from jones_framework.core.shadow_tensor import ShadowTensorBuilder
from jones_framework.core.tensor_ops import Tensor
from jones_framework.core.manifold_bridge import bridge, ConnectionType, get_registry
from jones_framework.perception.tda_pipeline import TDAPipeline
from jones_framework.perception.regime_classifier import RegimeClassifier
from jones_framework.sans.mixture_of_experts import MixtureOfExperts
from jones_framework.sans.continuity_guard import ContinuityGuard
from jones_framework.domains.base import DomainAdapter, MultiDomainOrchestrator
from jones_framework.utils.config import FrameworkConfig

class _cO0O4dl(Enum):
    GET = auto()
    POST = auto()
    PUT = auto()
    DELETE = auto()
    PATCH = auto()

@dataclass
class _c00l4d2:
    path: str
    method: _cO0O4dl
    handler: Callable
    description: str = ''
    auth_required: bool = True
    rate_limit: int = 100

@dataclass
class _cI1l4d3:
    host: str = '0.0.0.0'
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ['*'])
    rate_limit_per_minute: int = 1000
    max_request_size_mb: int = 10
    auth_enabled: bool = True
    api_key_header: str = 'X-API-Key'
    valid_api_keys: List[str] = field(default_factory=list)

@dataclass
class _cI1l4d4:
    success: bool
    data: Any = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def _fOOO4d5(self) -> Dict[str, Any]:
        return {'success': self.success, 'data': self.data, 'error': self.error, 'meta': self.meta, 'timestamp': datetime.now().isoformat()}

    def _fIOO4d6(self) -> str:
        return json.dumps(self._fOOO4d5(), default=str)

class _cOl14d7:

    def __init__(self, _flOl4d8: int):
        self.limit = _flOl4d8
        self.requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def get_metric_at(self, _f1l14dA: str) -> bool:
        import time
        now = time.time()
        window = 60
        with self._lock:
            if _f1l14dA not in self.requests:
                self.requests[_f1l14dA] = []
            self.requests[_f1l14dA] = [t for t in self.requests[_f1l14dA] if now - t < window]
            if len(self.requests[_f1l14dA]) >= self.limit:
                return False
            self.requests[_f1l14dA].append(now)
            return True

@bridge(connects_to=['ConditionState', 'RegimeClassifier', 'MixtureOfExperts', 'DomainAdapter', 'FrameworkConfig', 'ContinuityGuard'], connection_types={'ConditionState': ConnectionType.USES, 'MixtureOfExperts': ConnectionType.USES, 'DomainAdapter': ConnectionType.USES})
class _c0l04dB:

    def __init__(self, _fl0O4dc: _cI1l4d3, _fIOI4dd: Optional[FrameworkConfig]=None):
        self._fl0O4dc = _fl0O4dc
        self._fIOI4dd = _fIOI4dd or FrameworkConfig()
        self._tda_pipeline = TDAPipeline()
        self._classifier = RegimeClassifier(self._tda_pipeline)
        self._moe = MixtureOfExperts(classifier=self._classifier)
        self._continuity_guard = ContinuityGuard()
        self._shadow_builder = ShadowTensorBuilder()
        self._orchestrator = MultiDomainOrchestrator()
        self._rate_limiter = _cOl14d7(_fl0O4dc.rate_limit_per_minute)
        self._routes: Dict[Tuple[str, _cO0O4dl], _c00l4d2] = {}
        self._register_routes()
        self._state_buffer: List[ConditionState] = []
        self._max_buffer = 10000
        self._app = None
        self._server = None

    def _fOl14dE(self):
        routes = [_c00l4d2('/health', _cO0O4dl.GET, self._health_check, 'Health check endpoint', auth_required=False), _c00l4d2('/status', _cO0O4dl.GET, self._get_status, 'Get server status'), _c00l4d2('/metrics', _cO0O4dl.GET, self._get_metrics, 'Get system metrics'), _c00l4d2('/states', _cO0O4dl.POST, self._ingest_state, 'Ingest new condition state'), _c00l4d2('/states/batch', _cO0O4dl.POST, self._ingest_batch, 'Ingest batch of states'), _c00l4d2('/states', _cO0O4dl.GET, self._get_states, 'Get recent states'), _c00l4d2('/states/{state_id}', _cO0O4dl.GET, self._get_state, 'Get specific state'), _c00l4d2('/regime', _cO0O4dl.GET, self._get_current_regime, 'Get current regime'), _c00l4d2('/regime/history', _cO0O4dl.GET, self._get_regime_history, 'Get regime transition history'), _c00l4d2('/regime/classify', _cO0O4dl.POST, self._classify_data, 'Classify data into regime'), _c00l4d2('/experts', _cO0O4dl.GET, self._list_experts, 'List all experts'), _c00l4d2('/experts/{regime_id}', _cO0O4dl.GET, self._get_expert, 'Get expert details'), _c00l4d2('/experts/active', _cO0O4dl.GET, self._get_active_expert, 'Get currently active expert'), _c00l4d2('/experts/{regime_id}/activate', _cO0O4dl.POST, self._activate_expert, 'Activate specific expert'), _c00l4d2('/process', _cO0O4dl.POST, self._process_state, 'Process state through MoE'), _c00l4d2('/process/batch', _cO0O4dl.POST, self._process_batch, 'Process batch of states'), _c00l4d2('/shadow-tensor', _cO0O4dl.GET, self._get_shadow_tensor, 'Get current shadow tensor'), _c00l4d2('/shadow-tensor/build', _cO0O4dl.POST, self._build_shadow_tensor, 'Build shadow tensor from data'), _c00l4d2('/tda/persistence', _cO0O4dl.POST, self._compute_persistence, 'Compute persistence diagram'), _c00l4d2('/tda/features', _cO0O4dl.POST, self._extract_tda_features, 'Extract TDA features'), _c00l4d2('/safety/validate', _cO0O4dl.POST, self._validate_transition, 'Validate state transition'), _c00l4d2('/safety/stats', _cO0O4dl.GET, self._get_safety_stats, 'Get safety statistics'), _c00l4d2('/domains', _cO0O4dl.GET, self._list_domains, 'List registered domains'), _c00l4d2('/domains/{name}', _cO0O4dl.GET, self._get_domain, 'Get domain details'), _c00l4d2('/domains/{name}/ingest', _cO0O4dl.POST, self._ingest_to_domain, 'Ingest data to domain'), _c00l4d2('/components', _cO0O4dl.GET, self._list_components, 'List all framework components'), _c00l4d2('/components/graph', _cO0O4dl.GET, self._get_component_graph, 'Get component connection graph')]
        for route in routes:
            self._routes[route.path, route.method] = route

    async def _flll4df(self, _fO1l4EO: Dict) -> _cI1l4d4:
        return _cI1l4d4(success=True, data={'status': 'healthy'})

    async def _fl014El(self, _fO1l4EO: Dict) -> _cI1l4d4:
        return _cI1l4d4(success=True, data={'uptime': 'N/A', 'buffer_size': len(self._state_buffer), 'active_regime': self._moe.active_regime.name if self._moe.active_regime else None, 'num_domains': len(self._orchestrator.adapters)})

    async def _f0lI4E2(self, _fO1l4EO: Dict) -> _cI1l4d4:
        metrics = {'states_buffered': len(self._state_buffer), 'transitions': len(self._moe.get_transition_history()), 'expert_stats': self._moe.get_expert_stats(), 'safety_stats': self._continuity_guard.get_safety_statistics()}
        return _cI1l4d4(success=True, data=metrics)

    async def _f1lO4E3(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            data = _fO1l4EO.get('body', {})
            state = ConditionState.for_market(price=data.get('price', 0), volume=data.get('volume', 0), bid=data.get('bid', 0), ask=data.get('ask', 0), symbol=data.get('symbol', 'UNKNOWN'))
            self._state_buffer.append(state)
            if len(self._state_buffer) > self._max_buffer:
                self._state_buffer = self._state_buffer[-self._max_buffer:]
            return _cI1l4d4(success=True, data={'state_id': state.state_id})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _f1OI4E4(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            states = _fO1l4EO.get('body', {}).get('states', [])
            ingested = []
            for data in states:
                state = ConditionState.for_market(price=data.get('price', 0), volume=data.get('volume', 0), bid=data.get('bid', 0), ask=data.get('ask', 0), symbol=data.get('symbol', 'UNKNOWN'))
                self._state_buffer.append(state)
                ingested.append(state.state_id)
            return _cI1l4d4(success=True, data={'ingested': len(ingested)})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _fIOl4E5(self, _fO1l4EO: Dict) -> _cI1l4d4:
        limit = _fO1l4EO.get('params', {}).get('limit', 100)
        states = self._state_buffer[-limit:]
        return _cI1l4d4(success=True, data={'states': [{'id': s.state_id, 'timestamp': s.timestamp, 'vector': s.vector} for s in states], 'total': len(self._state_buffer)})

    async def _f00O4E6(self, _fO1l4EO: Dict) -> _cI1l4d4:
        state_id = _fO1l4EO.get('path_params', {}).get('state_id')
        for state in self._state_buffer:
            if state.state_id == state_id:
                return _cI1l4d4(success=True, data={'id': state.state_id, 'timestamp': state.timestamp, 'vector': state.vector, 'metadata': state.metadata})
        return _cI1l4d4(success=False, error='State not found')

    async def _f1004E7(self, _fO1l4EO: Dict) -> _cI1l4d4:
        if len(self._state_buffer) < 20:
            return _cI1l4d4(success=True, data={'regime': 'UNKNOWN', 'confidence': 0})
        shadow = self._shadow_builder.build(self._state_buffer[-50:])
        result = self._classifier.classify(shadow.point_cloud)
        return _cI1l4d4(success=True, data={'regime': result.regime_id.name, 'confidence': result.confidence, 'is_transition': result.is_transition})

    async def _flOI4E8(self, _fO1l4EO: Dict) -> _cI1l4d4:
        history = self._moe.get_transition_history()
        return _cI1l4d4(success=True, data={'transitions': [{'regime': r.name, 'timestamp': t, 'confidence': c} for r, t, c in history]})

    async def _fO1O4E9(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            data = _fO1l4EO.get('body', {}).get('point_cloud', [])
            import numpy as np
            point_cloud = np.array(data)
            result = self._classifier.classify(point_cloud)
            return _cI1l4d4(success=True, data={'regime': result.regime_id.name, 'confidence': result.confidence, 'distances': {k.name: v for k, v in result.all_distances.items()}})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _fO114EA(self, _fO1l4EO: Dict) -> _cI1l4d4:
        experts = self._moe.list_experts()
        return _cI1l4d4(success=True, data={'experts': [{'regime': r.name, 'description': d, 'active': a} for r, d, a in experts]})

    async def _fI114EB(self, _fO1l4EO: Dict) -> _cI1l4d4:
        regime_name = _fO1l4EO.get('path_params', {}).get('regime_id')
        try:
            regime_id = RegimeID[regime_name]
            if regime_id in self._moe.experts:
                expert = self._moe.experts[regime_id]
                return _cI1l4d4(success=True, data={'regime': regime_id.name, 'description': expert.description, 'priority': expert.priority})
            return _cI1l4d4(success=False, error='Expert not found')
        except KeyError:
            return _cI1l4d4(success=False, error=f'Unknown regime: {regime_name}')

    async def _fllI4Ec(self, _fO1l4EO: Dict) -> _cI1l4d4:
        active = self._moe.active_regime
        if active:
            return _cI1l4d4(success=True, data={'regime': active.name})
        return _cI1l4d4(success=True, data={'regime': None})

    async def _fOlO4Ed(self, _fO1l4EO: Dict) -> _cI1l4d4:
        regime_name = _fO1l4EO.get('path_params', {}).get('regime_id')
        try:
            regime_id = RegimeID[regime_name]
            self._moe.hot_swap(regime_id)
            return _cI1l4d4(success=True, data={'activated': regime_id.name})
        except KeyError:
            return _cI1l4d4(success=False, error=f'Unknown regime: {regime_name}')

    async def _f1014EE(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            data = _fO1l4EO.get('body', {})
            state = ConditionState.for_market(price=data.get('price', 0), volume=data.get('volume', 0), bid=data.get('bid', 0), ask=data.get('ask', 0), symbol=data.get('symbol', 'UNKNOWN'))
            shadow = None
            if len(self._state_buffer) >= 20:
                shadow = self._shadow_builder.build(self._state_buffer[-50:])
            output, regime = self._moe.process(state, shadow.point_cloud if shadow else None)
            return _cI1l4d4(success=True, data={'output': output.tolist(), 'regime': regime.name})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _fl014Ef(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            states_data = _fO1l4EO.get('body', {}).get('states', [])
            results = []
            for data in states_data:
                state = ConditionState.for_market(price=data.get('price', 0), volume=data.get('volume', 0), bid=data.get('bid', 0), ask=data.get('ask', 0), symbol=data.get('symbol', 'UNKNOWN'))
                output, regime = self._moe.process(state, None)
                results.append({'output': output.tolist(), 'regime': regime.name})
            return _cI1l4d4(success=True, data={'results': results})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _f0004fO(self, _fO1l4EO: Dict) -> _cI1l4d4:
        if len(self._state_buffer) < 20:
            return _cI1l4d4(success=False, error='Insufficient data for shadow tensor')
        shadow = self._shadow_builder.build(self._state_buffer[-50:])
        return _cI1l4d4(success=True, data={'metric_proxy': shadow.metric_proxy.tolist(), 'tangent_proxy': shadow.tangent_proxy.tolist(), 'fractal_proxy': shadow.fractal_proxy.tolist(), 'dimension': shadow.dimension})

    async def _fIl14fl(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            import numpy as np
            prices = np.array(_fO1l4EO.get('body', {}).get('prices', []))
            shadow = self._shadow_builder.build_from_numpy(prices)
            return _cI1l4d4(success=True, data={'metric_proxy': shadow.metric_proxy.tolist(), 'tangent_proxy': shadow.tangent_proxy.tolist(), 'dimension': shadow.dimension})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _f1014f2(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            import numpy as np
            point_cloud = np.array(_fO1l4EO.get('body', {}).get('point_cloud', []))
            diagram = self._tda_pipeline.compute_persistence(point_cloud)
            return _cI1l4d4(success=True, data={'h0': diagram.h0.tolist(), 'h1': diagram.h1.tolist(), 'betti_0': diagram.betti_0, 'betti_1': diagram.betti_1})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _fII04f3(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            import numpy as np
            point_cloud = np.array(_fO1l4EO.get('body', {}).get('point_cloud', []))
            features = self._tda_pipeline.extract_features(point_cloud)
            return _cI1l4d4(success=True, data=features)
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _fllI4f4(self, _fO1l4EO: Dict) -> _cI1l4d4:
        try:
            import numpy as np
            body = _fO1l4EO.get('body', {})
            current = ConditionState.for_market(price=body.get('current', {}).get('price', 0), volume=body.get('current', {}).get('volume', 0), bid=body.get('current', {}).get('bid', 0), ask=body.get('current', {}).get('ask', 0), symbol='VALIDATE')
            proposed = np.array(body.get('proposed', []))
            result = self._continuity_guard.validate_transition(current, proposed)
            return _cI1l4d4(success=True, data={'safety_level': result.safety_level.name, 'kl_divergence': result.kl_divergence, 'message': result.message, 'suggested_action': result.suggested_action})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _fOlO4f5(self, _fO1l4EO: Dict) -> _cI1l4d4:
        stats = self._continuity_guard.get_safety_statistics()
        return _cI1l4d4(success=True, data=stats)

    async def _fOI14f6(self, _fO1l4EO: Dict) -> _cI1l4d4:
        domains = [{'name': name, 'type': adapter.domain_type.name} for name, adapter in self._orchestrator.adapters.items()]
        return _cI1l4d4(success=True, data={'domains': domains})

    async def _f1I04f7(self, _fO1l4EO: Dict) -> _cI1l4d4:
        name = _fO1l4EO.get('path_params', {}).get('name')
        if name in self._orchestrator.adapters:
            adapter = self._orchestrator.adapters[name]
            return _cI1l4d4(success=True, data=adapter.get_metrics())
        return _cI1l4d4(success=False, error='Domain not found')

    async def _fIII4f8(self, _fO1l4EO: Dict) -> _cI1l4d4:
        name = _fO1l4EO.get('path_params', {}).get('name')
        if name not in self._orchestrator.adapters:
            return _cI1l4d4(success=False, error='Domain not found')
        try:
            data = _fO1l4EO.get('body', {})
            adapter = self._orchestrator.adapters[name]
            adapter.ingest(data)
            return _cI1l4d4(success=True, data={'ingested': True})
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    async def _f1I14f9(self, _fO1l4EO: Dict) -> _cI1l4d4:
        registry = get_registry()
        components = [{'name': name, 'module': node.module_path} for name, node in registry.components.items()]
        return _cI1l4d4(success=True, data={'components': components})

    async def _fIll4fA(self, _fO1l4EO: Dict) -> _cI1l4d4:
        registry = get_registry()
        return _cI1l4d4(success=True, data={'graph': registry.visualize()})

    def _f1114fB(self, _fO1O4fc: str, _f1I04fd: DomainAdapter):
        self._orchestrator.register(_fO1O4fc, _f1I04fd)

    async def _f01I4fE(self, _flO14ff: str, _flIl5OO: str, _fO1l4EO: Dict) -> _cI1l4d4:
        http_method = _cO0O4dl[_flIl5OO.upper()]
        route_key = (_flO14ff, http_method)
        if route_key not in self._routes:
            return _cI1l4d4(success=False, error='Not found')
        route = self._routes[route_key]
        _f1l14dA = _fO1l4EO.get('client_id', 'unknown')
        if not self._rate_limiter.get_metric_at(_f1l14dA):
            return _cI1l4d4(success=False, error='Rate limit exceeded')
        if route.auth_required and self._fl0O4dc.auth_enabled:
            api_key = _fO1l4EO.get('headers', {}).get(self._fl0O4dc.api_key_header)
            if api_key not in self._fl0O4dc.valid_api_keys:
                return _cI1l4d4(success=False, error='Unauthorized')
        try:
            return await route.handler(_fO1l4EO)
        except Exception as e:
            return _cI1l4d4(success=False, error=str(e))

    def _fl005Ol(self) -> Dict[str, Any]:
        paths = {}
        for (_flO14ff, _flIl5OO), route in self._routes.items():
            if _flO14ff not in paths:
                paths[_flO14ff] = {}
            paths[_flO14ff][_flIl5OO._fO1O4fc.lower()] = {'summary': route.description, 'responses': {'200': {'description': 'Success'}, '400': {'description': 'Bad request'}, '401': {'description': 'Unauthorized'}, '500': {'description': 'Server error'}}}
        return {'openapi': '3.0.0', 'info': {'title': 'Jones Framework API', 'version': '1.0.0', 'description': 'REST API for the Jones Framework'}, 'paths': paths}