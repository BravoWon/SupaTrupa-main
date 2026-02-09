from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
import time
import threading
import asyncio
from jones_framework.core import bridge, ComponentRegistry

class _cIlIA56(Enum):
    COUNTER = 'counter'
    GAUGE = 'gauge'
    HISTOGRAM = 'histogram'
    SUMMARY = 'summary'
    TIMER = 'timer'

class _c10IA57(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class _c010A58(Enum):
    OK = 'ok'
    PENDING = 'pending'
    FIRING = 'firing'
    RESOLVED = 'resolved'

class _cI0OA59(Enum):
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'
    UNKNOWN = 'unknown'

@dataclass
class _cIllA5A:
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class _c011A5B:
    name: str
    metric_type: _cIlIA56
    description: str = ''
    labels: List[str] = field(default_factory=list)
    unit: str = ''
    buckets: List[float] = field(default_factory=list)

@dataclass
class _cll1A5c:
    alert_id: str
    name: str
    severity: _c10IA57
    state: _c010A58
    message: str
    started_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    last_evaluated: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    threshold: Optional[float] = None
    firing_count: int = 0
    acknowledged: bool = False
    silenced_until: Optional[datetime] = None

    @property
    def _flIIA5d(self) -> timedelta:
        end = self.resolved_at or datetime.now()
        return end - self.started_at

    @property
    def _fI01A5E(self) -> bool:
        return self.state == _c010A58.FIRING

    @property
    def _fllOA5f(self) -> bool:
        if self.silenced_until is None:
            return False
        return datetime.now() < self.silenced_until

@dataclass
class _clIIA6O:
    name: str
    metric_name: str
    condition: str
    threshold: float
    severity: _c10IA57
    _flIIA5d: timedelta = timedelta(minutes=5)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    evaluation_interval: timedelta = timedelta(seconds=60)
    for_duration: timedelta = timedelta(minutes=1)

@dataclass
class _cIllA6l:
    name: str
    status: _cI0OA59
    message: str = ''
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class _cOl0A62:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Tuple[datetime, str]] = field(default_factory=list)
    status: str = 'ok'
    error: Optional[str] = None

    def get_metric_at(self, _f110A64: str='ok', _flIlA65: Optional[str]=None):
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self._f110A64 = _f110A64
        self._flIlA65 = _flIlA65

    def _fOI0A66(self, _fI0OA67: str):
        self.logs.append((datetime.now(), _fI0OA67))

class _c1I1A68:

    def __init__(self, _f10lA69: str, _fOOOA6A: Dict[str, str]=None):
        self._f10lA69 = _f10lA69
        self._fOOOA6A = _fOOOA6A or {}
        self._value = 0.0
        self._lock = threading.Lock()

    def _fI0IA6B(self, _fO00A6c: float=1.0):
        with self._lock:
            self._value += _fO00A6c

    @property
    def _fO00A6c(self) -> float:
        with self._lock:
            return self._value

class _cIlIA6d:

    def __init__(self, _f10lA69: str, _fOOOA6A: Dict[str, str]=None):
        self._f10lA69 = _f10lA69
        self._fOOOA6A = _fOOOA6A or {}
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, _fO00A6c: float):
        with self._lock:
            self._value = _fO00A6c

    def _fI0IA6B(self, _fO00A6c: float=1.0):
        with self._lock:
            self._value += _fO00A6c

    def _flIOA6E(self, _fO00A6c: float=1.0):
        with self._lock:
            self._value -= _fO00A6c

    @property
    def _fO00A6c(self) -> float:
        with self._lock:
            return self._value

class _cO1lA6f:
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, _f10lA69: str, _fOOlA7O: List[float]=None, _fOOOA6A: Dict[str, str]=None):
        self._f10lA69 = _f10lA69
        self._fOOOA6A = _fOOOA6A or {}
        self._buckets = sorted(_fOOlA7O or self.DEFAULT_BUCKETS)
        self._bucket_counts = [0] * len(self._buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def _fOO0A7l(self, _fO00A6c: float):
        with self._lock:
            self._sum += _fO00A6c
            self._count += 1
            for i, bucket in enumerate(self._buckets):
                if _fO00A6c <= bucket:
                    self._bucket_counts[i] += 1

    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum

    @property
    def _fI01A72(self) -> int:
        with self._lock:
            return self._count

    @property
    def _flIlA73(self) -> float:
        with self._lock:
            return self._sum / self._count if self._count > 0 else 0

    def _fO10A74(self, _f1llA75: float) -> float:
        with self._lock:
            if self._count == 0:
                return 0
            target = _f1llA75 * self._count
            cumulative = 0
            for i, _fI01A72 in enumerate(self._bucket_counts):
                cumulative += _fI01A72
                if cumulative >= target:
                    return self._buckets[i]
            return self._buckets[-1]

class _cI11A76:

    def __init__(self, _f10lA69: str, _fOOOA6A: Dict[str, str]=None):
        self._histogram = _cO1lA6f(_f10lA69, labels=_fOOOA6A)

    def _fOIlA77(self):
        return TimerContext(self._histogram)

    def _fOO0A7l(self, _flOOA78: float):
        self._histogram._fOO0A7l(_flOOA78)

class _cI1OA79:

    def __init__(self, _f1lOA7A: _cO1lA6f):
        self._histogram = _f1lOA7A
        self._start = None

    def __enter__(self):
        self._start = _fOIlA77._fOIlA77()
        return self

    def __exit__(self, *args):
        _flIIA5d = _fOIlA77._fOIlA77() - self._start
        self._histogram._fOO0A7l(_flIIA5d)

class _c0O0A7B:

    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._definitions: Dict[str, _c011A5B] = {}
        self._lock = threading.Lock()

    def _fOIlA7c(self, _flOOA7d: _c011A5B):
        with self._lock:
            self._definitions[_flOOA7d._f10lA69] = _flOOA7d

    def _f0IlA7E(self, _f10lA69: str, _fOOOA6A: Dict[str, str]=None) -> _c1I1A68:
        key = self._make_key(_f10lA69, _fOOOA6A)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = _c1I1A68(_f10lA69, _fOOOA6A)
            return self._metrics[key]

    def _fO1IA7f(self, _f10lA69: str, _fOOOA6A: Dict[str, str]=None) -> _cIlIA6d:
        key = self._make_key(_f10lA69, _fOOOA6A)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = _cIlIA6d(_f10lA69, _fOOOA6A)
            return self._metrics[key]

    def _f1lOA7A(self, _f10lA69: str, _fOOlA7O: List[float]=None, _fOOOA6A: Dict[str, str]=None) -> _cO1lA6f:
        key = self._make_key(_f10lA69, _fOOOA6A)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = _cO1lA6f(_f10lA69, _fOOlA7O, _fOOOA6A)
            return self._metrics[key]

    def _fOI1A8O(self, _f10lA69: str, _fOOOA6A: Dict[str, str]=None) -> _cI11A76:
        key = self._make_key(_f10lA69, _fOOOA6A)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = _cI11A76(_f10lA69, _fOOOA6A)
            return self._metrics[key]

    def _f11OA8l(self, _f10lA69: str, _fOOOA6A: Dict[str, str]=None) -> str:
        if not _fOOOA6A:
            return _f10lA69
        label_str = ','.join((f'{k}={v}' for k, v in sorted(_fOOOA6A.items())))
        return f'{_f10lA69}{{{label_str}}}'

    def _fI0OA82(self) -> List[_cIllA5A]:
        points = []
        now = datetime.now()
        with self._lock:
            for key, metric in self._metrics.items():
                if isinstance(metric, (_c1I1A68, _cIlIA6d)):
                    points.append(_cIllA5A(timestamp=now, value=metric._fO00A6c, labels=metric._fOOOA6A))
                elif isinstance(metric, _cO1lA6f):
                    points.append(_cIllA5A(timestamp=now, value=metric._flIlA73, labels={**metric._fOOOA6A, 'stat': 'mean'}))
                    points.append(_cIllA5A(timestamp=now, value=metric._fO10A74(0.95), labels={**metric._fOOOA6A, 'stat': 'p95'}))
                    points.append(_cIllA5A(timestamp=now, value=metric._fO10A74(0.99), labels={**metric._fOOOA6A, 'stat': 'p99'}))
        return points

class _c1IIA83:

    def __init__(self, _fII0A84: _c0O0A7B):
        self._registry = _fII0A84
        self._rules: List[_clIIA6O] = []
        self._alerts: Dict[str, _cll1A5c] = {}
        self._alert_history: List[_cll1A5c] = []
        self._notification_handlers: List[Callable[[_cll1A5c], None]] = []
        self._lock = threading.Lock()

    def _f0lIA85(self, _fOIlA86: _clIIA6O):
        with self._lock:
            self._rules.append(_fOIlA86)

    def _f10lA87(self, _fII0A88: Callable[[_cll1A5c], None]):
        self._notification_handlers.append(_fII0A88)

    @bridge(connects_to=['MetricsRegistry', 'RiskEngine'], connection_types={'MetricsRegistry': 'monitors', 'RiskEngine': 'alerts'})
    def _f0IlA89(self) -> List[_cll1A5c]:
        new_alerts = []
        now = datetime.now()
        with self._lock:
            for _fOIlA86 in self._rules:
                metric = self._get_metric_value(_fOIlA86.metric_name)
                if metric is None:
                    continue
                firing = self._evaluate_condition(metric, _fOIlA86.condition, _fOIlA86.threshold)
                alert_id = f'{_fOIlA86._f10lA69}_{hash(frozenset(_fOIlA86._fOOOA6A.items()))}'
                if firing:
                    if alert_id not in self._alerts:
                        alert = _cll1A5c(alert_id=alert_id, name=_fOIlA86._f10lA69, severity=_fOIlA86.severity, state=_c010A58.PENDING, message=f'{_fOIlA86.metric_name} {_fOIlA86.condition} {_fOIlA86.threshold}', labels=_fOIlA86._fOOOA6A, annotations=_fOIlA86.annotations, value=metric, threshold=_fOIlA86.threshold)
                        self._alerts[alert_id] = alert
                    else:
                        alert = self._alerts[alert_id]
                        alert.firing_count += 1
                        alert.last_evaluated = now
                        alert._fO00A6c = metric
                        if alert.state == _c010A58.PENDING and now - alert.started_at >= _fOIlA86.for_duration:
                            alert.state = _c010A58.FIRING
                            new_alerts.append(alert)
                            self._notify(alert)
                elif alert_id in self._alerts:
                    alert = self._alerts[alert_id]
                    if alert.state == _c010A58.FIRING:
                        alert.state = _c010A58.RESOLVED
                        alert.resolved_at = now
                        self._alert_history.append(alert)
                        self._notify(alert)
                    del self._alerts[alert_id]
        return new_alerts

    def _fO1OA8A(self, _fIllA8B: str) -> Optional[float]:
        points = self._registry._fI0OA82()
        for point in points:
            if _fIllA8B in str(point._fOOOA6A) or True:
                return point._fO00A6c
        return None

    def _fOlOA8c(self, _fO00A6c: float, _fl1IA8d: str, _fOIIA8E: float) -> bool:
        if _fl1IA8d == 'gt':
            return _fO00A6c > _fOIIA8E
        elif _fl1IA8d == 'lt':
            return _fO00A6c < _fOIIA8E
        elif _fl1IA8d == 'ge':
            return _fO00A6c >= _fOIIA8E
        elif _fl1IA8d == 'le':
            return _fO00A6c <= _fOIIA8E
        elif _fl1IA8d == 'eq':
            return _fO00A6c == _fOIIA8E
        elif _fl1IA8d == 'ne':
            return _fO00A6c != _fOIIA8E
        return False

    def _f1OIA8f(self, _fIlIA9O: _cll1A5c):
        if _fIlIA9O._fllOA5f:
            return
        for _fII0A88 in self._notification_handlers:
            try:
                _fII0A88(_fIlIA9O)
            except Exception:
                pass

    def _f0OlA9l(self, _f1O0A92: str, _flIIA5d: timedelta):
        with self._lock:
            if _f1O0A92 in self._alerts:
                self._alerts[_f1O0A92].silenced_until = datetime.now() + _flIIA5d

    def _fOI1A93(self, _f1O0A92: str):
        with self._lock:
            if _f1O0A92 in self._alerts:
                self._alerts[_f1O0A92].acknowledged = True

    @property
    def _f10OA94(self) -> List[_cll1A5c]:
        with self._lock:
            return [a for a in self._alerts.values() if a._fI01A5E]

class _cI1lA95:

    def __init__(self):
        self._checks: Dict[str, Callable[[], _cIllA6l]] = {}
        self._results: Dict[str, _cIllA6l] = {}
        self._lock = threading.Lock()

    def _f0O1A96(self, _f10lA69: str, _fO00A97: Callable[[], _cIllA6l]):
        with self._lock:
            self._checks[_f10lA69] = _fO00A97

    @bridge(connects_to=['JonesEngine', 'DataConnector', 'ModelRegistry'], connection_types={'JonesEngine': 'monitors', 'DataConnector': 'monitors', 'ModelRegistry': 'monitors'})
    def _flO1A98(self) -> Dict[str, _cIllA6l]:
        results = {}
        with self._lock:
            for _f10lA69, _fO00A97 in self._checks.items():
                start = _fOIlA77._fOIlA77()
                try:
                    result = _fO00A97()
                    result.latency_ms = (_fOIlA77._fOIlA77() - start) * 1000
                    results[_f10lA69] = result
                except Exception as e:
                    results[_f10lA69] = _cIllA6l(name=_f10lA69, status=_cI0OA59.UNHEALTHY, message=str(e), latency_ms=(_fOIlA77._fOIlA77() - start) * 1000)
            self._results = results
        return results

    @property
    def _f0IlA99(self) -> _cI0OA59:
        with self._lock:
            if not self._results:
                return _cI0OA59.UNKNOWN
            statuses = [r._f110A64 for r in self._results.values()]
            if all((s == _cI0OA59.HEALTHY for s in statuses)):
                return _cI0OA59.HEALTHY
            elif any((s == _cI0OA59.UNHEALTHY for s in statuses)):
                return _cI0OA59.UNHEALTHY
            elif any((s == _cI0OA59.DEGRADED for s in statuses)):
                return _cI0OA59.DEGRADED
            return _cI0OA59.UNKNOWN

class _cOlOA9A:

    def __init__(self, _f0IOA9B: str):
        self._service_name = _f0IOA9B
        self._spans: Dict[str, _cOl0A62] = {}
        self._completed_spans: List[_cOl0A62] = []
        self._lock = threading.Lock()

    def _fl0OA9c(self, _f110A9d: str, _f1O1A9E: Optional[str]=None, _f1IOA9f: Dict[str, str]=None) -> _cOl0A62:
        import hashlib
        import random
        trace_id = hashlib.md5(f'{_f110A9d}{datetime.now()}{random.random()}'.encode()).hexdigest()[:16]
        span_id = hashlib.md5(f'{trace_id}{random.random()}'.encode()).hexdigest()[:8]
        span = _cOl0A62(trace_id=trace_id, span_id=span_id, parent_span_id=_f1O1A9E, operation_name=_f110A9d, service_name=self._service_name, start_time=datetime.now(), tags=_f1IOA9f or {})
        with self._lock:
            self._spans[span_id] = span
        return span

    def _fOO0AAO(self, _fI01AAl: _cOl0A62, _f110A64: str='ok', _flIlA65: Optional[str]=None):
        _fI01AAl.get_metric_at(_f110A64, _flIlA65)
        with self._lock:
            if _fI01AAl.span_id in self._spans:
                del self._spans[_fI01AAl.span_id]
            self._completed_spans.append(_fI01AAl)
            if len(self._completed_spans) > 10000:
                self._completed_spans = self._completed_spans[-5000:]

    def _fOOIAA2(self, _fI10AA3: str) -> List[_cOl0A62]:
        with self._lock:
            return [s for s in self._completed_spans if s._fI10AA3 == _fI10AA3]

    @property
    def _f010AA4(self) -> List[_cOl0A62]:
        with self._lock:
            return list(self._spans.values())

class _clI0AA5:

    def __init__(self, _fI01AA6: _c0O0A7B, _f11OAA7: _c1IIA83, _fOlOAA8: _cI1lA95, _f1O0AA9: _cOlOA9A):
        self._metrics = _fI01AA6
        self._alerts = _f11OAA7
        self._health = _fOlOAA8
        self._tracer = _f1O0AA9

    @bridge(connects_to=['MetricsRegistry', 'AlertManager', 'HealthChecker', 'Tracer'], connection_types={'MetricsRegistry': 'reads', 'AlertManager': 'reads', 'HealthChecker': 'reads', 'Tracer': 'reads'})
    def _fOO1AAA(self) -> Dict[str, Any]:
        return {'timestamp': datetime.now().isoformat(), 'health': {'status': self._health._f0IlA99._fO00A6c, 'checks': {_f10lA69: {'status': check._f110A64._fO00A6c, 'message': check._fI0OA67, 'latency_ms': check.latency_ms} for _f10lA69, check in self._health._results.items()}}, 'alerts': {'active_count': len(self._alerts._f10OA94), 'by_severity': {severity._f10lA69: sum((1 for a in self._alerts._f10OA94 if a.severity == severity)) for severity in _c10IA57}}, 'tracing': {'active_spans': len(self._tracer._f010AA4)}}

    def _f1IlAAB(self) -> Dict[str, Any]:
        points = self._metrics._fI0OA82()
        return {'count': len(points), 'metrics': [{'labels': _f1llA75._fOOOA6A, 'value': _f1llA75._fO00A6c, 'timestamp': _f1llA75.timestamp.isoformat()} for _f1llA75 in points[:100]]}

    def _fIIIAAc(self, _fIIIAAd: int=24) -> List[Dict[str, Any]]:
        cutoff = datetime.now() - timedelta(hours=_fIIIAAd)
        return [{'alert_id': a._f1O0A92, 'name': a._f10lA69, 'severity': a.severity._f10lA69, 'state': a.state._fO00A6c, 'started_at': a.started_at.isoformat(), 'resolved_at': a.resolved_at.isoformat() if a.resolved_at else None, 'duration_seconds': a._flIIA5d.total_seconds()} for a in self._alerts._alert_history if a.started_at > cutoff]

class _c1llAAE:

    def __init__(self, _f0IOA9B: str='jones_framework'):
        self.metrics = _c0O0A7B()
        self.alerts = _c1IIA83(self.metrics)
        self.health = _cI1lA95()
        self._f1O0AA9 = _cOlOA9A(_f0IOA9B)
        self.dashboard = _clI0AA5(self.metrics, self.alerts, self.health, self._f1O0AA9)
        self._running = False
        self._registry = ComponentRegistry.get_instance()
        self._register_default_metrics()

    def _f1lOAAf(self):
        self.metrics._fOIlA7c(_c011A5B(name='jones_requests_total', metric_type=_cIlIA56.COUNTER, description='Total number of requests'))
        self.metrics._fOIlA7c(_c011A5B(name='jones_request_duration_seconds', metric_type=_cIlIA56.HISTOGRAM, description='Request duration in seconds'))
        self.metrics._fOIlA7c(_c011A5B(name='jones_active_positions', metric_type=_cIlIA56.GAUGE, description='Number of active positions'))
        self.metrics._fOIlA7c(_c011A5B(name='jones_portfolio_value', metric_type=_cIlIA56.GAUGE, description='Total portfolio value'))

    @bridge(connects_to=['JonesEngine', 'RiskEngine', 'SignalEngine'], connection_types={'JonesEngine': 'monitors', 'RiskEngine': 'monitors', 'SignalEngine': 'monitors'})
    async def _f001ABO(self):
        self._running = True
        asyncio.create_task(self._alert_evaluation_loop())
        asyncio.create_task(self._health_check_loop())

    async def _fO0OABl(self):
        self._running = False

    async def _fl0OAB2(self):
        while self._running:
            self.alerts._f0IlA89()
            await asyncio.sleep(60)

    async def _fIlIAB3(self):
        while self._running:
            self.health._flO1A98()
            await asyncio.sleep(30)

def _fllIAB4(_f0IOA9B: str='jones_framework') -> _c1llAAE:
    return _c1llAAE(_f0IOA9B)

def _f1IOAB5() -> _c0O0A7B:
    return _c0O0A7B()

def _fOOlAB6(_fII0A84: _c0O0A7B) -> _c1IIA83:
    return _c1IIA83(_fII0A84)

# Public API aliases for obfuscated classes
MetricType = _cIlIA56
Counter = _c1I1A68
Gauge = _cIlIA6d
Histogram = _cO1lA6f
MetricsRegistry = _cI1OA79
AlertManager = _c0O0A7B
MonitoringService = _clI0AA5
