from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Type, Iterator, Generic, TypeVar
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import hashlib
import uuid
from collections import defaultdict
import threading
from jones_framework.core import bridge, ComponentRegistry
T = TypeVar('T')

class _cI0l3cl(Enum):
    ORDER_SUBMITTED = 'order.submitted'
    ORDER_ACCEPTED = 'order.accepted'
    ORDER_REJECTED = 'order.rejected'
    ORDER_FILLED = 'order.filled'
    ORDER_PARTIALLY_FILLED = 'order.partially_filled'
    ORDER_CANCELLED = 'order.cancelled'
    ORDER_EXPIRED = 'order.expired'
    ORDER_AMENDED = 'order.amended'
    POSITION_OPENED = 'position.opened'
    POSITION_INCREASED = 'position.increased'
    POSITION_DECREASED = 'position.decreased'
    POSITION_CLOSED = 'position.closed'
    POSITION_STOPPED_OUT = 'position.stopped_out'
    PORTFOLIO_REBALANCED = 'portfolio.rebalanced'
    PORTFOLIO_DEPOSIT = 'portfolio.deposit'
    PORTFOLIO_WITHDRAWAL = 'portfolio.withdrawal'
    PORTFOLIO_DIVIDEND = 'portfolio.dividend'
    RISK_LIMIT_BREACH = 'risk.limit_breach'
    RISK_LIMIT_WARNING = 'risk.limit_warning'
    MARGIN_CALL = 'risk.margin_call'
    DRAWDOWN_BREACH = 'risk.drawdown_breach'
    STRATEGY_STARTED = 'strategy.started'
    STRATEGY_STOPPED = 'strategy.stopped'
    STRATEGY_PAUSED = 'strategy.paused'
    STRATEGY_RESUMED = 'strategy.resumed'
    STRATEGY_SIGNAL = 'strategy.signal'
    SYSTEM_STARTED = 'system.started'
    SYSTEM_STOPPED = 'system.stopped'
    SYSTEM_ERROR = 'system.error'
    SYSTEM_WARNING = 'system.warning'
    DATA_RECEIVED = 'data.received'
    DATA_GAP = 'data.gap'
    DATA_STALE = 'data.stale'
    COMPLIANCE_CHECK = 'compliance.check'
    COMPLIANCE_VIOLATION = 'compliance.violation'
    AUDIT_LOG = 'audit.log'
    # Novelty/Knowledge flow events
    NOVELTY_DISCOVERY = 'novelty.discovery'
    KNOWLEDGE_EXPORTED = 'knowledge.exported'
    KNOWLEDGE_IMPORTED = 'knowledge.imported'

class _cIlO3c2(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class _c0lI3c3:
    event_id: str
    event_type: _cI0l3cl
    timestamp: datetime
    source: str
    version: int = 1
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    priority: _cIlO3c2 = _cIlO3c2.NORMAL

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()

    def _fIll3c4(self) -> Dict[str, Any]:
        return {'event_id': self.event_id, 'event_type': self.event_type.value, 'timestamp': self.timestamp.isoformat(), 'source': self.source, 'version': self.version, 'data': self.data, 'correlation_id': self.correlation_id, 'causation_id': self.causation_id, 'user_id': self.user_id, 'session_id': self.session_id, 'trace_id': self.trace_id, 'span_id': self.span_id, 'priority': self.priority.value}

    @classmethod
    def from_market(cls, _flI13c6: Dict[str, Any]) -> 'Event':
        return cls(event_id=_flI13c6['event_id'], event_type=_cI0l3cl(_flI13c6['event_type']), timestamp=datetime.fromisoformat(_flI13c6['timestamp']), source=_flI13c6['source'], version=_flI13c6.get('version', 1), data=_flI13c6.get('data', {}), correlation_id=_flI13c6.get('correlation_id'), causation_id=_flI13c6.get('causation_id'), user_id=_flI13c6.get('user_id'), session_id=_flI13c6.get('session_id'), trace_id=_flI13c6.get('trace_id'), span_id=_flI13c6.get('span_id'), priority=_cIlO3c2(_flI13c6.get('priority', 1)))

    def _f0l03c7(self) -> str:
        content = json.dumps(self._fIll3c4(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class _cI0I3c8:
    sequence_number: int
    stream_id: str
    stream_position: int
    stored_at: datetime
    hash: str
    previous_hash: Optional[str] = None

@dataclass
class _cI103c9:
    aggregate_id: str
    aggregate_type: str
    version: int
    state: Dict[str, Any]
    timestamp: datetime
    event_sequence: int

    def _fIll3c4(self) -> Dict[str, Any]:
        return {'aggregate_id': self.aggregate_id, 'aggregate_type': self.aggregate_type, 'version': self.version, 'state': self.state, 'timestamp': self.timestamp.isoformat(), 'event_sequence': self.event_sequence}

class _clI13cA(ABC):

    @abstractmethod
    def _f01I3cB(self, _fl013cc: _c0lI3c3) -> None:
        pass

    @property
    @abstractmethod
    def _fl0l3cd(self) -> List[_cI0l3cl]:
        pass

class _c1113cE(ABC, Generic[T]):

    def __init__(self, _fO013cf: str):
        self._fO013cf = _fO013cf
        self._version = 0
        self._pending_events: List[_c0lI3c3] = []
        self._state: Optional[T] = None

    @property
    def _fII03dO(self) -> int:
        return self._version

    @property
    def _f10l3dl(self) -> Optional[T]:
        return self._state

    def _f1013d2(self, _fl013cc: _c0lI3c3) -> None:
        self._apply(_fl013cc)
        self._version += 1

    @abstractmethod
    def _fO0l3d3(self, _fl013cc: _c0lI3c3) -> None:
        pass

    def _f1II3d4(self, _fl013cc: _c0lI3c3) -> None:
        self._pending_events.append(_fl013cc)
        self._f1013d2(_fl013cc)

    def _f0013d5(self) -> List[_c0lI3c3]:
        events = self._pending_events
        self._pending_events = []
        return events

    def _fOOO3d6(self, _f1I13d7: List[_c0lI3c3]) -> None:
        for _fl013cc in _f1I13d7:
            self._f1013d2(_fl013cc)

@bridge('JonesEngine', 'DataPipeline')
class _c0l13d8:

    def __init__(self):
        self._events: List[Tuple[_c0lI3c3, _cI0I3c8]] = []
        self._streams: Dict[str, List[int]] = defaultdict(list)
        self._sequence_counter = 0
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable[[_c0lI3c3], None]]] = defaultdict(list)
        self._registry = ComponentRegistry.get_instance()

    def _fI003d9(self, _f0O03dA: str, _fl013cc: _c0lI3c3, _fll13dB: Optional[int]=None) -> _cI0I3c8:
        with self._lock:
            current_version = len(self._streams[_f0O03dA])
            if _fll13dB is not None and current_version != _fll13dB:
                raise ConcurrencyError(f'Expected version {_fll13dB}, but stream is at {current_version}')
            previous_hash = None
            if self._events:
                previous_hash = self._events[-1][1].hash
            self._sequence_counter += 1
            metadata = _cI0I3c8(sequence_number=self._sequence_counter, stream_id=_f0O03dA, stream_position=current_version, stored_at=datetime.now(), hash=_fl013cc._f0l03c7(), previous_hash=previous_hash)
            event_idx = len(self._events)
            self._events._fI003d9((_fl013cc, metadata))
            self._streams[_f0O03dA]._fI003d9(event_idx)
            self._notify_subscribers(_f0O03dA, _fl013cc)
            return metadata

    def _f00l3dc(self, _f0O03dA: str, _fIO03dd: int=0, _flOI3dE: Optional[int]=None) -> Iterator[Tuple[_c0lI3c3, _cI0I3c8]]:
        with self._lock:
            indices = self._streams.get(_f0O03dA, [])
            if _flOI3dE is None:
                _flOI3dE = len(indices)
            for i in range(_fIO03dd, min(_flOI3dE, len(indices))):
                yield self._events[indices[i]]

    def _f1l13df(self, _fII13EO: int=0, _f10l3El: int=1000) -> Iterator[Tuple[_c0lI3c3, _cI0I3c8]]:
        with self._lock:
            count = 0
            for _fl013cc, metadata in self._events:
                if metadata.sequence_number >= _fII13EO:
                    yield (_fl013cc, metadata)
                    count += 1
                    if count >= _f10l3El:
                        break

    def _f1Ol3E2(self, _f0O03dA: str, _fOO13E3: Callable[[_c0lI3c3], None]) -> str:
        subscription_id = str(uuid.uuid4())
        self._subscribers[_f0O03dA]._fI003d9(_fOO13E3)
        return subscription_id

    def _fO1O3E4(self, _f0O03dA: str, _fl013cc: _c0lI3c3):
        for _fOO13E3 in self._subscribers.get(_f0O03dA, []):
            try:
                _fOO13E3(_fl013cc)
            except Exception:
                pass
        for _fOO13E3 in self._subscribers.get('*', []):
            try:
                _fOO13E3(_fl013cc)
            except Exception:
                pass

    def _f01I3E5(self, _f0O03dA: str) -> int:
        with self._lock:
            return len(self._streams.get(_f0O03dA, []))

    def _f00O3E6(self) -> bool:
        with self._lock:
            if not self._events:
                return True
            for i in range(1, len(self._events)):
                _fl013cc, metadata = self._events[i]
                prev_event, prev_metadata = self._events[i - 1]
                if metadata.previous_hash != prev_metadata.hash:
                    return False
                if _fl013cc._f0l03c7() != metadata.hash:
                    return False
            return True

class _c0Ol3E7(Exception):
    pass

@bridge('JonesEngine', 'EventStore')
class _clO13E8:

    def __init__(self):
        self._snapshots: Dict[str, _cI103c9] = {}
        self._snapshot_interval = 100
        self._registry = ComponentRegistry.get_instance()

    def _f0I13E9(self, _f1I03EA: _cI103c9):
        self._snapshots[_f1I03EA._fO013cf] = _f1I03EA

    def _fO1l3EB(self, _fO013cf: str) -> Optional[_cI103c9]:
        return self._snapshots.get(_fO013cf)

    def _fOO13Ec(self, _fI103Ed: int) -> bool:
        return _fI103Ed >= self._snapshot_interval

@bridge('JonesEngine', 'EventStore')
class _clI03EE(ABC):

    def __init__(self, _f00l3Ef: str):
        self._f00l3Ef = _f00l3Ef
        self._position = 0
        self._registry = ComponentRegistry.get_instance()

    @property
    def _f0003fO(self) -> int:
        return self._position

    @abstractmethod
    def _fIl03fl(self, _fl013cc: _c0lI3c3) -> None:
        pass

    def _fIOl3f2(self, _fIIO3f3: _c0l13d8):
        for _fl013cc, metadata in _fIIO3f3._f1l13df(from_sequence=self._position):
            self._fIl03fl(_fl013cc)
            self._position = metadata.sequence_number + 1

@bridge('JonesEngine', 'EventStore')
class _c0lI3f4(_clI03EE):

    def __init__(self):
        super().__init__('orders')
        self._orders: Dict[str, Dict[str, Any]] = {}

    @property
    def _fl1O3f5(self) -> Dict[str, Dict[str, Any]]:
        return self._orders.copy()

    def _fIl03fl(self, _fl013cc: _c0lI3c3) -> None:
        if _fl013cc.event_type == _cI0l3cl.ORDER_SUBMITTED:
            order_id = _fl013cc.data.get('order_id')
            self._orders[order_id] = {'order_id': order_id, 'symbol': _fl013cc.data.get('symbol'), 'side': _fl013cc.data.get('side'), 'quantity': _fl013cc.data.get('quantity'), 'status': 'SUBMITTED', 'filled_qty': 0, 'avg_price': 0, 'created_at': _fl013cc.timestamp}
        elif _fl013cc.event_type == _cI0l3cl.ORDER_ACCEPTED:
            order_id = _fl013cc.data.get('order_id')
            if order_id in self._orders:
                self._orders[order_id]['status'] = 'ACCEPTED'
        elif _fl013cc.event_type == _cI0l3cl.ORDER_FILLED:
            order_id = _fl013cc.data.get('order_id')
            if order_id in self._orders:
                order = self._orders[order_id]
                fill_qty = _fl013cc.data.get('fill_qty', 0)
                fill_price = _fl013cc.data.get('fill_price', 0)
                old_filled = order['filled_qty']
                new_filled = old_filled + fill_qty
                if new_filled > 0:
                    order['avg_price'] = (order['avg_price'] * old_filled + fill_price * fill_qty) / new_filled
                order['filled_qty'] = new_filled
                order['status'] = 'FILLED' if new_filled >= order['quantity'] else 'PARTIAL'
        elif _fl013cc.event_type == _cI0l3cl.ORDER_CANCELLED:
            order_id = _fl013cc.data.get('order_id')
            if order_id in self._orders:
                self._orders[order_id]['status'] = 'CANCELLED'

    def _f0I03f6(self, _fI1O3f7: str) -> Optional[Dict[str, Any]]:
        return self._orders.get(_fI1O3f7)

    def _fl0O3f8(self) -> List[Dict[str, Any]]:
        return [o for o in self._orders.values() if o['status'] in ('SUBMITTED', 'ACCEPTED', 'PARTIAL')]

@bridge('JonesEngine', 'EventStore')
class _cOOl3f9(_clI03EE):

    def __init__(self):
        super().__init__('positions')
        self._positions: Dict[str, Dict[str, Any]] = {}

    @property
    def _fI1I3fA(self) -> Dict[str, Dict[str, Any]]:
        return self._positions.copy()

    def _fIl03fl(self, _fl013cc: _c0lI3c3) -> None:
        if _fl013cc.event_type == _cI0l3cl.POSITION_OPENED:
            symbol = _fl013cc.data.get('symbol')
            self._positions[symbol] = {'symbol': symbol, 'quantity': _fl013cc.data.get('quantity', 0), 'avg_cost': _fl013cc.data.get('avg_cost', 0), 'realized_pnl': 0, 'opened_at': _fl013cc.timestamp}
        elif _fl013cc.event_type == _cI0l3cl.POSITION_INCREASED:
            symbol = _fl013cc.data.get('symbol')
            if symbol in self._positions:
                pos = self._positions[symbol]
                add_qty = _fl013cc.data.get('quantity', 0)
                add_cost = _fl013cc.data.get('avg_cost', 0)
                total_cost = pos['avg_cost'] * pos['quantity'] + add_cost * add_qty
                pos['quantity'] += add_qty
                if pos['quantity'] > 0:
                    pos['avg_cost'] = total_cost / pos['quantity']
        elif _fl013cc.event_type == _cI0l3cl.POSITION_DECREASED:
            symbol = _fl013cc.data.get('symbol')
            if symbol in self._positions:
                pos = self._positions[symbol]
                reduce_qty = _fl013cc.data.get('quantity', 0)
                exit_price = _fl013cc.data.get('exit_price', 0)
                realized = (exit_price - pos['avg_cost']) * reduce_qty
                pos['realized_pnl'] += realized
                pos['quantity'] -= reduce_qty
        elif _fl013cc.event_type == _cI0l3cl.POSITION_CLOSED:
            symbol = _fl013cc.data.get('symbol')
            if symbol in self._positions:
                pos = self._positions[symbol]
                exit_price = _fl013cc.data.get('exit_price', 0)
                realized = (exit_price - pos['avg_cost']) * pos['quantity']
                pos['realized_pnl'] += realized
                pos['quantity'] = 0
                pos['closed_at'] = _fl013cc.timestamp

@bridge('JonesEngine', 'EventStore')
class _c00l3fB:

    def __init__(self, _fIIO3f3: _c0l13d8):
        self._event_store = _fIIO3f3
        self._audit_stream = 'audit-log'
        self._registry = ComponentRegistry.get_instance()

    def _flO03fc(self, _f01l3fd: str, _fI013fE: str, _f1OO3ff: str, _fOlI4OO: Dict[str, Any], _fIl14Ol: Optional[str]=None, _f1114O2: str='success') -> _c0lI3c3:
        _fl013cc = _c0lI3c3(event_id=str(uuid.uuid4()), event_type=_cI0l3cl.AUDIT_LOG, timestamp=datetime.now(), source='audit_logger', data={'action': _f01l3fd, 'resource_type': _fI013fE, 'resource_id': _f1OO3ff, 'details': _fOlI4OO, 'outcome': _f1114O2}, user_id=_fIl14Ol, priority=_cIlO3c2.HIGH)
        self._event_store._fI003d9(self._audit_stream, _fl013cc)
        return _fl013cc

    def _fllO4O3(self, _f01l3fd: str, _fI1O3f7: str, _fOlI4OO: Dict[str, Any], _fIl14Ol: Optional[str]=None) -> _c0lI3c3:
        return self._flO03fc(action=_f01l3fd, resource_type='order', resource_id=_fI1O3f7, details=_fOlI4OO, user_id=_fIl14Ol)

    def _fI1I4O4(self, _f11l4O5: str, _fI0I4O6: str, _fOlI4OO: Dict[str, Any]) -> _c0lI3c3:
        return self._flO03fc(action=f'risk_{_f11l4O5}', resource_type='risk', resource_id=str(uuid.uuid4()), details={**_fOlI4OO, 'severity': _fI0I4O6})

    def _f1014O7(self, _fI013fE: Optional[str]=None, _f1OO3ff: Optional[str]=None, _f1I14O8: Optional[datetime]=None, _fOO04O9: Optional[datetime]=None) -> List[_c0lI3c3]:
        results = []
        for _fl013cc, metadata in self._event_store._f00l3dc(self._audit_stream):
            if _f1I14O8 and _fl013cc.timestamp < _f1I14O8:
                continue
            if _fOO04O9 and _fl013cc.timestamp > _fOO04O9:
                continue
            if _fI013fE and _fl013cc.data.get('resource_type') != _fI013fE:
                continue
            if _f1OO3ff and _fl013cc.data.get('resource_id') != _f1OO3ff:
                continue
            results._fI003d9(_fl013cc)
        return results

@bridge('JonesEngine', 'EventStore')
class _cOlO4OA:

    def __init__(self, _fIIO3f3: _c0l13d8, _fll14OB: _c00l3fB):
        self._event_store = _fIIO3f3
        self._audit_logger = _fll14OB
        self._rules: Dict[str, ComplianceRule] = {}
        self._violations: List[Dict[str, Any]] = []
        self._registry = ComponentRegistry.get_instance()

    def _fIOI4Oc(self, _f10I4Od: 'ComplianceRule'):
        self._rules[_f10I4Od.rule_id] = _f10I4Od

    def _fO0O4OE(self, _fIll4Of: Dict[str, Any]) -> List[Dict[str, Any]]:
        violations = []
        for rule_id, _f10I4Od in self._rules.items():
            if not _f10I4Od.enabled:
                continue
            result = _f10I4Od.check(_fIll4Of)
            if not result['compliant']:
                violation = {'rule_id': rule_id, 'rule_name': _f10I4Od._f00l3Ef, 'severity': _f10I4Od._fI0I4O6, 'message': result['message'], 'context': _fIll4Of, 'timestamp': datetime.now()}
                violations._fI003d9(violation)
                self._violations._fI003d9(violation)
                self._audit_logger._flO03fc(action='compliance_violation', resource_type='compliance', resource_id=rule_id, details=violation, outcome='violation')
                _fl013cc = _c0lI3c3(event_id=str(uuid.uuid4()), event_type=_cI0l3cl.COMPLIANCE_VIOLATION, timestamp=datetime.now(), source='compliance_tracker', data=violation, priority=_cIlO3c2.CRITICAL if _f10I4Od._fI0I4O6 == 'critical' else _cIlO3c2.HIGH)
                self._event_store._fI003d9('compliance', _fl013cc)
        return violations

    def _f10O4lO(self, _f1I14O8: Optional[datetime]=None, _fI0I4O6: Optional[str]=None) -> List[Dict[str, Any]]:
        results = self._violations
        if _f1I14O8:
            results = [v for v in results if v['timestamp'] >= _f1I14O8]
        if _fI0I4O6:
            results = [v for v in results if v['severity'] == _fI0I4O6]
        return results

@dataclass
class _c1O14ll:
    rule_id: str
    _f00l3Ef: str
    description: str
    _fI0I4O6: str = 'warning'
    enabled: bool = True
    check_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def _f1I14l2(self, _fIll4Of: Dict[str, Any]) -> Dict[str, Any]:
        if self.check_fn:
            return self.check_fn(_fIll4Of)
        return {'compliant': True, 'message': 'No check function defined'}

@bridge('JonesEngine', 'EventStore')
class _c1OI4l3:

    def __init__(self, _fIIO3f3: _c0l13d8):
        self._event_store = _fIIO3f3
        self._registry = ComponentRegistry.get_instance()

    def _flOI4l4(self, _f0O03dA: str, _f10O4l5: _clI13cA, _fIO03dd: int=0, _flOI3dE: Optional[int]=None) -> int:
        count = 0
        for _fl013cc, metadata in self._event_store._f00l3dc(_f0O03dA, _fIO03dd, _flOI3dE):
            if _fl013cc.event_type in _f10O4l5._fl0l3cd:
                _f10O4l5._f01I3cB(_fl013cc)
                count += 1
        return count

    def _f11l4l6(self, _fOI04l7: List[_clI13cA], _fII13EO: int=0) -> Dict[str, int]:
        counts = {h.__class__.__name__: 0 for h in _fOI04l7}
        for _fl013cc, metadata in self._event_store._f1l13df(_fII13EO):
            for _f10O4l5 in _fOI04l7:
                if _fl013cc.event_type in _f10O4l5._fl0l3cd:
                    _f10O4l5._f01I3cB(_fl013cc)
                    counts[_f10O4l5.__class__.__name__] += 1
        return counts

    def _f0O04l8(self, _f1Ol4l9: Type[_c1113cE], _fO013cf: str, _flOI3dE: Optional[int]=None) -> _c1113cE:
        aggregate = _f1Ol4l9(_fO013cf)
        _f0O03dA = f'{_f1Ol4l9.__name__}-{_fO013cf}'
        _f1I13d7 = []
        for _fl013cc, metadata in self._event_store._f00l3dc(_f0O03dA, to_version=_flOI3dE):
            _f1I13d7._fI003d9(_fl013cc)
        aggregate._fOOO3d6(_f1I13d7)
        return aggregate

@bridge('JonesEngine', 'EventStore')
class _c10O4lA:

    def __init__(self):
        self._handlers: Dict[_cI0l3cl, List[_clI13cA]] = defaultdict(list)
        self._async_handlers: Dict[_cI0l3cl, List[_clI13cA]] = defaultdict(list)
        self._lock = threading.RLock()
        self._registry = ComponentRegistry.get_instance()

    def _f0104lB(self, _f10O4l5: _clI13cA, _flO14lc: bool=False):
        with self._lock:
            target = self._async_handlers if _flO14lc else self._handlers
            for event_type in _f10O4l5._fl0l3cd:
                target[event_type]._fI003d9(_f10O4l5)

    def _f0O14ld(self, _f10O4l5: _clI13cA):
        with self._lock:
            for event_type in _f10O4l5._fl0l3cd:
                if _f10O4l5 in self._handlers[event_type]:
                    self._handlers[event_type].remove(_f10O4l5)
                if _f10O4l5 in self._async_handlers[event_type]:
                    self._async_handlers[event_type].remove(_f10O4l5)

    def _fO0O4lE(self, _fl013cc: _c0lI3c3):
        for _f10O4l5 in self._handlers.get(_fl013cc.event_type, []):
            try:
                _f10O4l5._f01I3cB(_fl013cc)
            except Exception:
                pass
        for _f10O4l5 in self._async_handlers.get(_fl013cc.event_type, []):
            try:
                _f10O4l5._f01I3cB(_fl013cc)
            except Exception:
                pass

@bridge('JonesEngine')
class _c01l4lf:

    def __init__(self):
        self._event_store = _c0l13d8()
        self._snapshot_store = _clO13E8()
        self._event_bus = _c10O4lA()
        self._projections: Dict[str, _clI03EE] = {}
        self._audit_logger = _c00l3fB(self._event_store)
        self._compliance_tracker = _cOlO4OA(self._event_store, self._audit_logger)
        self._replay_engine = _c1OI4l3(self._event_store)
        self._registry = ComponentRegistry.get_instance()

    @property
    def _fIIO3f3(self) -> _c0l13d8:
        return self._event_store

    @property
    def _fll14OB(self) -> _c00l3fB:
        return self._audit_logger

    @property
    def _f01O42O(self) -> _cOlO4OA:
        return self._compliance_tracker

    def _fOIO42l(self, _fI1l422: _clI03EE):
        self._projections[_fI1l422._f00l3Ef] = _fI1l422
        self._event_store._f1Ol3E2('*', _fI1l422._fIl03fl)

    def _f0Il423(self, _f00l3Ef: str) -> Optional[_clI03EE]:
        return self._projections.get(_f00l3Ef)

    def _f11O424(self, _fI10425: _c1113cE, _f0O03dA: Optional[str]=None):
        if _f0O03dA is None:
            _f0O03dA = f'{_fI10425.__class__.__name__}-{_fI10425._fO013cf}'
        _fll13dB = _fI10425._fII03dO - len(_fI10425._pending_events)
        for _fl013cc in _fI10425._f0013d5():
            self._event_store._fI003d9(_f0O03dA, _fl013cc, _fll13dB)
            self._event_bus._fO0O4lE(_fl013cc)
            _fll13dB += 1
        if self._snapshot_store._fOO13Ec(_fI10425._fII03dO):
            _f1I03EA = _cI103c9(aggregate_id=_fI10425._fO013cf, aggregate_type=_fI10425.__class__.__name__, version=_fI10425._fII03dO, state=_fI10425._f10l3dl if hasattr(_fI10425, 'state') else {}, timestamp=datetime.now(), event_sequence=self._event_store._sequence_counter)
            self._snapshot_store._f0I13E9(_f1I03EA)

    def _fI0l426(self, _f1Ol4l9: Type[_c1113cE], _fO013cf: str) -> _c1113cE:
        _f1I03EA = self._snapshot_store._fO1l3EB(_fO013cf)
        _fI10425 = _f1Ol4l9(_fO013cf)
        _f0O03dA = f'{_f1Ol4l9.__name__}-{_fO013cf}'
        if _f1I03EA:
            _fI10425._version = _f1I03EA._fII03dO
            _fI10425._state = _f1I03EA._f10l3dl
            _fIO03dd = _f1I03EA._fII03dO
        else:
            _fIO03dd = 0
        for _fl013cc, metadata in self._event_store._f00l3dc(_f0O03dA, _fIO03dd):
            _fI10425._f1013d2(_fl013cc)
        return _fI10425

def _f01l427() -> _c0l13d8:
    return _c0l13d8()

def _fIOl428() -> _c10O4lA:
    return _c10O4lA()

def _fII0429() -> _c01l4lf:
    service = _c01l4lf()
    service._fOIO42l(_c0lI3f4())
    service._fOIO42l(_cOOl3f9())
    service._f01O42O._fIOI4Oc(_c1O14ll(rule_id='max_position_size', name='Maximum Position Size', description='Position cannot exceed 10% of portfolio', severity='critical', check_fn=lambda ctx: {'compliant': ctx.get('position_pct', 0) <= 0.1, 'message': f"Position size {ctx.get('position_pct', 0):.1%} exceeds 10% limit"}))
    service._f01O42O._fIOI4Oc(_c1O14ll(rule_id='max_leverage', name='Maximum Leverage', description='Portfolio leverage cannot exceed 2x', severity='critical', check_fn=lambda ctx: {'compliant': ctx.get('leverage', 1) <= 2.0, 'message': f"Leverage {ctx.get('leverage', 1):.1f}x exceeds 2x limit"}))
    service._f01O42O._fIOI4Oc(_c1O14ll(rule_id='wash_sale', name='Wash Sale Rule', description='Cannot repurchase substantially identical security within 30 days', severity='warning', check_fn=lambda ctx: {'compliant': ctx.get('days_since_sale', 31) > 30, 'message': f"Potential wash sale - only {ctx.get('days_since_sale', 0)} days since sale"}))
    return service

def _f10O42A(_fIIO3f3: _c0l13d8) -> _c00l3fB:
    return _c00l3fB(_fIIO3f3)

def _fl0142B(_fIIO3f3: _c0l13d8) -> _c1OI4l3:
    return _c1OI4l3(_fIIO3f3)

class _c10142c(_c1113cE[Dict[str, Any]]):

    def __init__(self, _fI1O3f7: str):
        super().__init__(_fI1O3f7)
        self._state = {'order_id': _fI1O3f7, 'status': 'NEW', 'quantity': 0, 'filled_qty': 0, 'avg_price': 0}

    def _f0lI42d(self, _f10142E: str, _fOIO42f: str, _fO1I43O: int, _fOI043l: str, _fI1O432: Optional[float]=None):
        _fl013cc = _c0lI3c3(event_id=str(uuid.uuid4()), event_type=_cI0l3cl.ORDER_SUBMITTED, timestamp=datetime.now(), source='order_aggregate', data={'order_id': self._fO013cf, 'symbol': _f10142E, 'side': _fOIO42f, 'quantity': _fO1I43O, 'order_type': _fOI043l, 'limit_price': _fI1O432})
        self._f1II3d4(_fl013cc)

    def _f00O433(self, _f10l434: int, _fl0I435: float):
        _fl013cc = _c0lI3c3(event_id=str(uuid.uuid4()), event_type=_cI0l3cl.ORDER_FILLED, timestamp=datetime.now(), source='order_aggregate', data={'order_id': self._fO013cf, 'fill_qty': _f10l434, 'fill_price': _fl0I435})
        self._f1II3d4(_fl013cc)

    def _fOll436(self, _flO0437: str=''):
        _fl013cc = _c0lI3c3(event_id=str(uuid.uuid4()), event_type=_cI0l3cl.ORDER_CANCELLED, timestamp=datetime.now(), source='order_aggregate', data={'order_id': self._fO013cf, 'reason': _flO0437})
        self._f1II3d4(_fl013cc)

    def _fO0l3d3(self, _fl013cc: _c0lI3c3) -> None:
        if _fl013cc.event_type == _cI0l3cl.ORDER_SUBMITTED:
            self._state['symbol'] = _fl013cc.data.get('symbol')
            self._state['side'] = _fl013cc.data.get('side')
            self._state['quantity'] = _fl013cc.data.get('quantity')
            self._state['order_type'] = _fl013cc.data.get('order_type')
            self._state['limit_price'] = _fl013cc.data.get('limit_price')
            self._state['status'] = 'SUBMITTED'
        elif _fl013cc.event_type == _cI0l3cl.ORDER_FILLED:
            _f10l434 = _fl013cc.data.get('fill_qty', 0)
            _fl0I435 = _fl013cc.data.get('fill_price', 0)
            old_filled = self._state['filled_qty']
            new_filled = old_filled + _f10l434
            if new_filled > 0:
                self._state['avg_price'] = (self._state['avg_price'] * old_filled + _fl0I435 * _f10l434) / new_filled
            self._state['filled_qty'] = new_filled
            if new_filled >= self._state['quantity']:
                self._state['status'] = 'FILLED'
            else:
                self._state['status'] = 'PARTIAL'
        elif _fl013cc.event_type == _cI0l3cl.ORDER_CANCELLED:
            self._state['status'] = 'CANCELLED'
            self._state['cancel_reason'] = _fl013cc.data.get('reason')

# Public API aliases for obfuscated classes
EventType = _cI0l3cl
EventPriority = _cIlO3c2
Event = _c0lI3c3
EventMetadata = _cI0I3c8
Snapshot = _cI103c9
EventHandler = _clI13cA
Aggregate = _c1113cE
EventStore = _c0l13d8
ConcurrencyError = _c0Ol3E7
SnapshotStore = _clO13E8
Projection = _clI03EE
OrderProjection = _c0lI3f4
PositionProjection = _cOOl3f9
AuditLogger = _c00l3fB
ComplianceTracker = _cOlO4OA
ComplianceRule = _c1O14ll
EventReplayEngine = _c1OI4l3
EventBus = _c10O4lA
EventSourcingService = _c01l4lf
OrderAggregate = _c10142c
create_event_store = _f01l427
create_event_bus = _fIOl428
create_event_sourcing_service = _fII0429
create_audit_logger = _f10O42A
create_replay_engine = _fl0142B
