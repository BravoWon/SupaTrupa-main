from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic, Set, Protocol, Sequence
from enum import Enum, auto
import time
import math
import threading
from collections import deque
import hashlib
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID

class _cIlOBBO(Enum):
    GEOMETRIC = auto()
    LINGUISTIC = auto()
    VOLATILITY = auto()
    FLOW = auto()
    CORRELATION = auto()
    MOMENTUM = auto()
    MEAN_REVERSION = auto()

class _clllBBl(Enum):
    IDLE = auto()
    LOADING = auto()
    ARMED = auto()
    FIRING = auto()
    COOLDOWN = auto()

class _cOllBB2(Enum):
    REGIME_SWITCH = auto()
    POSITION_ENTRY = auto()
    POSITION_EXIT = auto()
    HEDGE_ACTIVATION = auto()
    RISK_REDUCTION = auto()
    MODEL_RETRAIN = auto()
    ALERT_ONLY = auto()
    DRILLING_STOP = auto()
    FLOW_REDIRECT = auto()

class _c10IBB3(Enum):
    STABLE = auto()
    STRESSED = auto()
    BREAKING = auto()
    INVERTED = auto()
    RECOVERING = auto()

@dataclass(frozen=True)
class _clO0BB4:
    timestamp: int
    betti_numbers: Tuple[int, ...]
    persistence_entropy: float
    bottleneck_distance: float
    bbw_squeeze: float
    manifold_curvature: float
    singularity_proximity: float
    dimension_loss: float

    @property
    def _f000BB5(self) -> float:
        squeeze_factor = 1.0 - self.bbw_squeeze
        singularity_factor = 1.0 - self.singularity_proximity
        score = 0.25 * squeeze_factor + 0.2 * singularity_factor + 0.2 * self.dimension_loss + 0.15 * min(1.0, self.manifold_curvature / 10.0) + 0.1 * min(1.0, self.persistence_entropy / 5.0) + 0.1 * min(1.0, self.bottleneck_distance / 2.0)
        return max(0.0, min(1.0, score))

    def _f10lBB6(self, _fO0OBB7: float=0.7) -> bool:
        return self._f000BB5 >= _fO0OBB7

@dataclass(frozen=True)
class _c1IlBB8:
    timestamp: int
    fear_index: float
    distrust_index: float
    narrative_velocity: float
    consensus_sentiment: float
    shadow_sentiment: float
    divergence: float
    keyword_acceleration: float

    @property
    def _fOI0BB9(self) -> float:
        divergence_factor = self.divergence
        velocity_factor = min(1.0, abs(self.narrative_velocity) / 5.0)
        fear_factor = self.fear_index
        distrust_factor = self.distrust_index
        score = 0.35 * divergence_factor + 0.25 * velocity_factor + 0.2 * fear_factor + 0.15 * distrust_factor + 0.05 * min(1.0, abs(self.keyword_acceleration) / 3.0)
        return max(0.0, min(1.0, score))

    def _fl0lBBA(self, _fO0OBB7: float=0.6) -> bool:
        return self._fOI0BB9 >= _fO0OBB7

@dataclass(frozen=True)
class _cO10BBB:
    timestamp: int
    correlation_matrix: Tuple[Tuple[float, ...], ...]
    rolling_correlation: float
    correlation_velocity: float
    eigenvalue_ratio: float
    dispersion: float

    @property
    def _f10lBBc(self) -> float:
        velocity_factor = min(1.0, abs(self.correlation_velocity) / 0.5)
        dispersion_factor = min(1.0, self.dispersion / 0.3)
        factor_breakdown = 1.0 - min(1.0, self.eigenvalue_ratio)
        score = 0.4 * velocity_factor + 0.35 * dispersion_factor + 0.25 * factor_breakdown
        return max(0.0, min(1.0, score))

@dataclass
class _c1OlBBd:
    name: str
    signal_type: _cIlOBBO
    _fO0OBB7: float
    comparison: str
    weight: float = 1.0
    required: bool = True

    def _fO1lBBE(self, _fI0IBBf: float) -> bool:
        if self.comparison == 'gt':
            return _fI0IBBf > self._fO0OBB7
        elif self.comparison == 'lt':
            return _fI0IBBf < self._fO0OBB7
        elif self.comparison == 'eq':
            return abs(_fI0IBBf - self._fO0OBB7) < 0.001
        elif self.comparison == 'gte':
            return _fI0IBBf >= self._fO0OBB7
        elif self.comparison == 'lte':
            return _fI0IBBf <= self._fO0OBB7
        return False

@dataclass
class _c0l0BcO:
    name: str
    conditions: List[_c1OlBBd]
    action: _cOllBB2
    cooldown_ms: int = 5000
    confirmation_count: int = 1
    max_fire_rate: float = 0.1
    direction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@bridge(connects_to=['ConditionState', 'ActivityState', 'TDAPipeline', 'RegimeClassifier', 'LinguisticArbitrageEngine', 'SentimentVectorPipeline', 'ContinuityGuard'], connection_types={'ConditionState': ConnectionType.TRANSFORMS, 'ActivityState': ConnectionType.PRODUCES, 'TDAPipeline': ConnectionType.USES, 'RegimeClassifier': ConnectionType.USES, 'LinguisticArbitrageEngine': ConnectionType.USES, 'SentimentVectorPipeline': ConnectionType.USES, 'ContinuityGuard': ConnectionType.VALIDATES})
class _cllOBcl:

    def __init__(self, _f1IIBc2: Optional[List[_c0l0BcO]]=None, _fOl0Bc3: float=0.7, _fI11Bc4: float=0.6, _f1I1Bc5: float=0.5, _fO10Bc6: str='multiplicative'):
        self._f1IIBc2 = _f1IIBc2 or self._default_gates()
        self._fOl0Bc3 = _fOl0Bc3
        self._fI11Bc4 = _fI11Bc4
        self._f1I1Bc5 = _f1I1Bc5
        self._fO10Bc6 = _fO10Bc6
        self._state = _clllBBl.IDLE
        self._correlation_regime = _c10IBB3.STABLE
        self._last_fire_time: Dict[str, float] = {}
        self._signal_buffer: deque = deque(maxlen=1000)
        self._confirmation_counts: Dict[str, int] = {}
        self._on_fire_callbacks: List[Callable] = []
        self._on_state_change_callbacks: List[Callable] = []
        self._lock = threading.RLock()
        self._metrics = {'total_signals': 0, 'gates_evaluated': 0, 'gates_fired': 0, 'false_positives': 0, 'confirmed_fires': 0}

    def _fl00Bc7(self) -> List[_c0l0BcO]:
        return [_c0l0BcO(name='primary_arbitrage', conditions=[_c1OlBBd('geometric_potential', _cIlOBBO.GEOMETRIC, 0.7, 'gte'), _c1OlBBd('linguistic_kinetic', _cIlOBBO.LINGUISTIC, 0.6, 'gte')], action=_cOllBB2.REGIME_SWITCH, cooldown_ms=10000, confirmation_count=2), _c0l0BcO(name='squeeze_breakout', conditions=[_c1OlBBd('bbw_squeeze', _cIlOBBO.GEOMETRIC, 0.2, 'lte', required=True), _c1OlBBd('narrative_velocity', _cIlOBBO.LINGUISTIC, 3.0, 'gte')], action=_cOllBB2.POSITION_ENTRY, cooldown_ms=30000, confirmation_count=1), _c0l0BcO(name='correlation_breakdown', conditions=[_c1OlBBd('correlation_breakdown', _cIlOBBO.CORRELATION, 0.6, 'gte'), _c1OlBBd('fear_index', _cIlOBBO.LINGUISTIC, 0.7, 'gte')], action=_cOllBB2.HEDGE_ACTIVATION, cooldown_ms=60000, confirmation_count=3), _c0l0BcO(name='crisis_detector', conditions=[_c1OlBBd('singularity_proximity', _cIlOBBO.GEOMETRIC, 0.1, 'lte'), _c1OlBBd('divergence', _cIlOBBO.LINGUISTIC, 0.8, 'gte'), _c1OlBBd('correlation_breakdown', _cIlOBBO.CORRELATION, 0.7, 'gte')], action=_cOllBB2.RISK_REDUCTION, cooldown_ms=300000, confirmation_count=5), _c0l0BcO(name='stuck_pipe_prevention', conditions=[_c1OlBBd('dimension_loss', _cIlOBBO.GEOMETRIC, 0.8, 'gte'), _c1OlBBd('flow_anomaly', _cIlOBBO.FLOW, 0.7, 'gte')], action=_cOllBB2.DRILLING_STOP, cooldown_ms=1000, confirmation_count=1, metadata={'domain': 'oil_gas', 'priority': 'critical'})]

    def _fOI0Bc8(self, _f1IOBc9: Optional[_clO0BB4]=None, _f101BcA: Optional[_c1IlBB8]=None, _f0lOBcB: Optional[_cO10BBB]=None, _fOIlBcc: Optional[Dict[str, float]]=None) -> List[Tuple[_c0l0BcO, _cOllBB2, float]]:
        with self._lock:
            self._metrics['total_signals'] += 1
            signals = self._build_signal_dict(_f1IOBc9, _f101BcA, _f0lOBcB, _fOIlBcc)
            self._signal_buffer.append({'timestamp': time.time_ns(), 'signals': signals})
            fusion_score = self._compute_fusion_score(_f1IOBc9, _f101BcA, _f0lOBcB)
            self._update_trigger_state(fusion_score, signals)
            fired_gates = []
            for gate in self._f1IIBc2:
                self._metrics['gates_evaluated'] += 1
                if self._evaluate_gate(gate, signals):
                    if self._check_cooldown(gate):
                        if self._confirm_signal(gate, signals):
                            confidence = self._compute_confidence(gate, signals, fusion_score)
                            fired_gates.append((gate, gate.action, confidence))
                            self._record_fire(gate)
                            self._metrics['gates_fired'] += 1
            for gate, action, confidence in fired_gates:
                self._trigger_callbacks(gate, action, confidence, signals)
            return fired_gates

    def _f1OlBcd(self, _f1IOBc9: Optional[_clO0BB4], _f101BcA: Optional[_c1IlBB8], _f0lOBcB: Optional[_cO10BBB], _fOIlBcc: Optional[Dict[str, float]]) -> Dict[str, float]:
        signals = {}
        if _f1IOBc9:
            signals['geometric_potential'] = _f1IOBc9._f000BB5
            signals['bbw_squeeze'] = _f1IOBc9.bbw_squeeze
            signals['singularity_proximity'] = _f1IOBc9.singularity_proximity
            signals['dimension_loss'] = _f1IOBc9.dimension_loss
            signals['manifold_curvature'] = _f1IOBc9.manifold_curvature
            signals['persistence_entropy'] = _f1IOBc9.persistence_entropy
            signals['bottleneck_distance'] = _f1IOBc9.bottleneck_distance
        if _f101BcA:
            signals['linguistic_kinetic'] = _f101BcA._fOI0BB9
            signals['fear_index'] = _f101BcA.fear_index
            signals['distrust_index'] = _f101BcA.distrust_index
            signals['narrative_velocity'] = _f101BcA.narrative_velocity
            signals['divergence'] = _f101BcA.divergence
            signals['keyword_acceleration'] = _f101BcA.keyword_acceleration
            signals['consensus_sentiment'] = _f101BcA.consensus_sentiment
            signals['shadow_sentiment'] = _f101BcA.shadow_sentiment
        if _f0lOBcB:
            signals['correlation_breakdown'] = _f0lOBcB._f10lBBc
            signals['correlation_velocity'] = _f0lOBcB.correlation_velocity
            signals['dispersion'] = _f0lOBcB.dispersion
            signals['eigenvalue_ratio'] = _f0lOBcB.eigenvalue_ratio
        if _fOIlBcc:
            signals.update(_fOIlBcc)
        return signals

    def _fI01BcE(self, _f1IOBc9: Optional[_clO0BB4], _f101BcA: Optional[_c1IlBB8], _f0lOBcB: Optional[_cO10BBB]) -> float:
        scores = []
        if _f1IOBc9:
            scores.append(_f1IOBc9._f000BB5)
        if _f101BcA:
            scores.append(_f101BcA._fOI0BB9)
        if _f0lOBcB:
            scores.append(_f0lOBcB._f10lBBc)
        if not scores:
            return 0.0
        if self._fO10Bc6 == 'multiplicative':
            product = 1.0
            for s in scores:
                product *= s
            return product ** (1.0 / len(scores))
        elif self._fO10Bc6 == 'additive':
            return sum(scores) / len(scores)
        elif self._fO10Bc6 == 'min':
            return min(scores)
        elif self._fO10Bc6 == 'max':
            return max(scores)
        else:
            return sum(scores) / len(scores)

    def _fO10Bcf(self, _flIIBdO: float, _f0lOBdl: Dict[str, float]):
        old_state = self._state
        geometric_potential = _f0lOBdl.get('geometric_potential', 0)
        linguistic_kinetic = _f0lOBdl.get('linguistic_kinetic', 0)
        if _flIIBdO < 0.3:
            self._state = _clllBBl.IDLE
        elif geometric_potential >= self._fOl0Bc3 and linguistic_kinetic < self._fI11Bc4:
            self._state = _clllBBl.ARMED
        elif geometric_potential < self._fOl0Bc3 and linguistic_kinetic >= self._fI11Bc4:
            self._state = _clllBBl.LOADING
        elif _flIIBdO >= 0.6:
            self._state = _clllBBl.FIRING
        else:
            self._state = _clllBBl.LOADING
        if old_state != self._state:
            for callback in self._on_state_change_callbacks:
                callback(old_state, self._state, _f0lOBdl)

    def _f1O1Bd2(self, _fI1IBd3: _c0l0BcO, _f0lOBdl: Dict[str, float]) -> bool:
        required_pass = True
        optional_score = 0.0
        optional_count = 0
        for condition in _fI1IBd3.conditions:
            _fI0IBBf = _f0lOBdl.get(condition.name, 0.0)
            passed = condition._fO1lBBE(_fI0IBBf)
            if condition.required:
                if not passed:
                    required_pass = False
                    break
            else:
                optional_count += 1
                if passed:
                    optional_score += condition.weight
        if not required_pass:
            return False
        if optional_count > 0:
            return optional_score >= optional_count * 0.5
        return True

    def _fIlIBd4(self, _fI1IBd3: _c0l0BcO) -> bool:
        last_fire = self._last_fire_time.get(_fI1IBd3.name, 0)
        elapsed_ms = (time.time() - last_fire) * 1000
        return elapsed_ms >= _fI1IBd3.cooldown_ms

    def _flO1Bd5(self, _fI1IBd3: _c0l0BcO, _f0lOBdl: Dict[str, float]) -> bool:
        gate_key = _fI1IBd3.name
        if gate_key not in self._confirmation_counts:
            self._confirmation_counts[gate_key] = 0
        self._confirmation_counts[gate_key] += 1
        if self._confirmation_counts[gate_key] >= _fI1IBd3.confirmation_count:
            self._confirmation_counts[gate_key] = 0
            self._metrics['confirmed_fires'] += 1
            return True
        return False

    def _fll1Bd6(self, _fI1IBd3: _c0l0BcO, _f0lOBdl: Dict[str, float], _flIIBdO: float) -> float:
        confidence = _flIIBdO
        for condition in _fI1IBd3.conditions:
            _fI0IBBf = _f0lOBdl.get(condition.name, 0)
            if condition.comparison in ('gt', 'gte'):
                excess = (_fI0IBBf - condition._fO0OBB7) / condition._fO0OBB7
                confidence += 0.1 * max(0, min(1, excess))
        return max(0.0, min(1.0, confidence))

    def _flllBd7(self, _fI1IBd3: _c0l0BcO):
        self._last_fire_time[_fI1IBd3.name] = time.time()

    def _f01OBd8(self, _fI1IBd3: _c0l0BcO, _fIl1Bd9: _cOllBB2, _f1l1BdA: float, _f0lOBdl: Dict[str, float]):
        event = {'gate_name': _fI1IBd3.name, 'action': _fIl1Bd9, 'confidence': _f1l1BdA, 'signals': _f0lOBdl, 'timestamp': time.time_ns(), 'state': self._state}
        for callback in self._on_fire_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def _fI1OBdB(self, _fI1IBd3: _c0l0BcO):
        with self._lock:
            self._f1IIBc2.append(_fI1IBd3)

    def _flO1Bdc(self, _fI01Bdd: str) -> bool:
        with self._lock:
            for i, _fI1IBd3 in enumerate(self._f1IIBc2):
                if _fI1IBd3._fI01Bdd == _fI01Bdd:
                    self._f1IIBc2.pop(i)
                    return True
            return False

    def _fll1BdE(self, _fI0OBdf: Callable):
        self._on_fire_callbacks.append(_fI0OBdf)

    def _fOl0BEO(self, _fI0OBdf: Callable):
        self._on_state_change_callbacks.append(_fI0OBdf)

    @property
    def _f0lOBEl(self) -> _clllBBl:
        return self._state

    @property
    def _fII1BE2(self) -> _c10IBB3:
        return self._correlation_regime

    def _fI0IBE3(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metrics)

    def _fI01BE4(self):
        with self._lock:
            for key in self._metrics:
                self._metrics[key] = 0

@bridge(connects_to=['CorrelationCutter', 'MarketAdapter'], connection_types={'CorrelationCutter': ConnectionType.EXTENDS, 'MarketAdapter': ConnectionType.USES})
class _cOO1BE5(_cllOBcl):

    def __init__(self, **kwargs):
        market_gates = [_c0l0BcO(name='volatility_regime_shift', conditions=[_c1OlBBd('geometric_potential', _cIlOBBO.GEOMETRIC, 0.65, 'gte'), _c1OlBBd('bbw_squeeze', _cIlOBBO.VOLATILITY, 0.15, 'lte')], action=_cOllBB2.HEDGE_ACTIVATION, cooldown_ms=60000, confirmation_count=2, metadata={'regime': 'volatility'}), _c0l0BcO(name='risk_parity_breakdown', conditions=[_c1OlBBd('correlation_breakdown', _cIlOBBO.CORRELATION, 0.7, 'gte'), _c1OlBBd('geometric_potential', _cIlOBBO.GEOMETRIC, 0.6, 'gte')], action=_cOllBB2.RISK_REDUCTION, cooldown_ms=300000, confirmation_count=4, metadata={'strategy': 'risk_parity'}), _c0l0BcO(name='carry_trade_unwind', conditions=[_c1OlBBd('fear_index', _cIlOBBO.LINGUISTIC, 0.75, 'gte'), _c1OlBBd('correlation_velocity', _cIlOBBO.CORRELATION, 0.3, 'gte')], action=_cOllBB2.POSITION_EXIT, cooldown_ms=120000, confirmation_count=3, direction='short', metadata={'strategy': 'carry'}), _c0l0BcO(name='central_bank_intervention', conditions=[_c1OlBBd('narrative_velocity', _cIlOBBO.LINGUISTIC, 5.0, 'gte'), _c1OlBBd('keyword_acceleration', _cIlOBBO.LINGUISTIC, 4.0, 'gte')], action=_cOllBB2.ALERT_ONLY, cooldown_ms=3600000, confirmation_count=2, metadata={'event': 'intervention'})]
        existing_gates = kwargs.pop('gates', None) or []
        kwargs['gates'] = existing_gates + market_gates
        super().__init__(**kwargs)

@bridge(connects_to=['CorrelationCutter', 'ReservoirAdapter'], connection_types={'CorrelationCutter': ConnectionType.EXTENDS, 'ReservoirAdapter': ConnectionType.USES})
class _cO1OBE6(_cllOBcl):

    def __init__(self, **kwargs):
        reservoir_gates = [_c0l0BcO(name='water_breakthrough_imminent', conditions=[_c1OlBBd('singularity_proximity', _cIlOBBO.GEOMETRIC, 0.15, 'lte'), _c1OlBBd('dimension_loss', _cIlOBBO.GEOMETRIC, 0.7, 'gte')], action=_cOllBB2.FLOW_REDIRECT, cooldown_ms=5000, confirmation_count=2, metadata={'event': 'water_breakthrough', 'priority': 'high'}), _c0l0BcO(name='stuck_pipe_imminent', conditions=[_c1OlBBd('geometric_potential', _cIlOBBO.GEOMETRIC, 0.85, 'gte'), _c1OlBBd('flow_anomaly', _cIlOBBO.FLOW, 0.6, 'gte')], action=_cOllBB2.DRILLING_STOP, cooldown_ms=500, confirmation_count=1, metadata={'event': 'stuck_pipe', 'priority': 'critical'}), _c0l0BcO(name='darcy_to_non_darcy', conditions=[_c1OlBBd('persistence_entropy', _cIlOBBO.GEOMETRIC, 3.0, 'gte'), _c1OlBBd('manifold_curvature', _cIlOBBO.GEOMETRIC, 8.0, 'gte')], action=_cOllBB2.REGIME_SWITCH, cooldown_ms=30000, confirmation_count=3, metadata={'transition': 'darcy_non_darcy'}), _c0l0BcO(name='wellbore_integrity_alert', conditions=[_c1OlBBd('dimension_loss', _cIlOBBO.GEOMETRIC, 0.9, 'gte')], action=_cOllBB2.ALERT_ONLY, cooldown_ms=10000, confirmation_count=5, metadata={'alert': 'wellbore_integrity', 'priority': 'high'})]
        existing_gates = kwargs.pop('gates', None) or []
        kwargs['gates'] = existing_gates + reservoir_gates
        super().__init__(**kwargs)

@bridge(connects_to=['TDAPipeline', 'RegimeClassifier', 'GeometricSignal'], connection_types={'TDAPipeline': ConnectionType.USES, 'RegimeClassifier': ConnectionType.USES, 'GeometricSignal': ConnectionType.PRODUCES})
class _cl1OBE7:

    def __init__(self, _f1IIBE8=None, _fOIOBE9=None):
        self._f1IIBE8 = _f1IIBE8
        self._fOIOBE9 = _fOIOBE9
        self._reference_topology = None

    def _fl1lBEA(self, _f0lOBEl: ConditionState, _fI00BEB: float=0.5, _fO0IBEc: float=50.0) -> _clO0BB4:
        timestamp = _f0lOBEl.timestamp
        betti_numbers = (1, 0, 0)
        persistence_entropy = 1.0
        bottleneck_distance = 0.5
        if self._f1IIBE8:
            try:
                diagram = self._f1IIBE8.compute_persistence(_f0lOBEl.vector)
                betti_numbers = tuple(diagram.betti_numbers)
                persistence_entropy = self._compute_entropy(diagram)
                if self._reference_topology:
                    bottleneck_distance = self._compute_bottleneck(diagram)
            except Exception:
                pass
        bbw_squeeze = max(0.0, min(1.0, _fI00BEB / 0.4))
        singularity_proximity = self._estimate_singularity_proximity(_f0lOBEl)
        dimension_loss = self._estimate_dimension_loss(_f0lOBEl)
        manifold_curvature = self._estimate_curvature(_f0lOBEl)
        return _clO0BB4(timestamp=timestamp, betti_numbers=betti_numbers, persistence_entropy=persistence_entropy, bottleneck_distance=bottleneck_distance, bbw_squeeze=bbw_squeeze, manifold_curvature=manifold_curvature, singularity_proximity=singularity_proximity, dimension_loss=dimension_loss)

    def _f1llBEd(self, _fO1OBEE) -> float:
        return 1.5

    def _fOI1BEf(self, _fO1OBEE) -> float:
        return 0.3

    def _f001BfO(self, _f0lOBEl: ConditionState) -> float:
        return 0.5

    def _fOOOBfl(self, _f0lOBEl: ConditionState) -> float:
        return 0.3

    def _fIl1Bf2(self, _f0lOBEl: ConditionState) -> float:
        return 2.0

@bridge(connects_to=['SentimentVectorPipeline', 'LinguisticArbitrageEngine', 'LinguisticSignal'], connection_types={'SentimentVectorPipeline': ConnectionType.USES, 'LinguisticArbitrageEngine': ConnectionType.USES, 'LinguisticSignal': ConnectionType.PRODUCES})
class _c1OlBf3:

    def __init__(self, _f1IlBf4=None, _f000Bf5=None):
        self._f1IlBf4 = _f1IlBf4
        self._f000Bf5 = _f000Bf5
        self._history = deque(maxlen=100)

    def _fl1lBEA(self, _f1IOBf6: float=0.0, _fOOOBf7: float=0.0, _f0OOBf8: float=0.0, _f001Bf9: float=0.0) -> _c1IlBB8:
        timestamp = time.time_ns()
        divergence = abs(_f1IOBf6 - _fOOOBf7) / 2.0
        self._history.append({'timestamp': timestamp, 'consensus': _f1IOBf6, 'shadow': _fOOOBf7})
        narrative_velocity = self._compute_velocity()
        keyword_acceleration = self._compute_keyword_acceleration(_f0OOBf8)
        return _c1IlBB8(timestamp=timestamp, fear_index=max(0.0, min(1.0, _f0OOBf8)), distrust_index=max(0.0, min(1.0, _f001Bf9)), narrative_velocity=narrative_velocity, consensus_sentiment=_f1IOBf6, shadow_sentiment=_fOOOBf7, divergence=divergence, keyword_acceleration=keyword_acceleration)

    def _flI1BfA(self) -> float:
        if len(self._history) < 2:
            return 0.0
        recent = list(self._history)[-10:]
        if len(recent) < 2:
            return 0.0
        start_div = abs(recent[0]['consensus'] - recent[0]['shadow'])
        end_div = abs(recent[-1]['consensus'] - recent[-1]['shadow'])
        return (end_div - start_div) * 10

    def _flI0BfB(self, _fI1IBfc: float) -> float:
        if len(self._history) < 3:
            return 0.0
        return _fI1IBfc * 2.0
__all__ = ['SignalType', 'TriggerState', 'ActionType', 'CorrelationRegime', 'GeometricSignal', 'LinguisticSignal', 'CorrelationSignal', 'GateCondition', 'GateConfiguration', 'CorrelationCutter', 'MarketCorrelationCutter', 'ReservoirCorrelationCutter', 'GeometricSignalGenerator', 'LinguisticSignalGenerator']