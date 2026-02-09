from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, TypeVar, Generic
from enum import Enum, auto
from datetime import datetime, timedelta
import math
from collections import defaultdict
from abc import ABC, abstractmethod
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.engine.core import Timeframe
T = TypeVar('T')

class _c01O438(Enum):
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    BOOLEAN = 'boolean'
    VECTOR = 'vector'
    MATRIX = 'matrix'
    TIMESERIES = 'timeseries'
    TEXT = 'text'
    EMBEDDING = 'embedding'

class _c11O439(Enum):
    PRICE = 'price'
    VOLUME = 'volume'
    TECHNICAL = 'technical'
    FUNDAMENTAL = 'fundamental'
    SENTIMENT = 'sentiment'
    ALTERNATIVE = 'alternative'
    DERIVED = 'derived'
    CROSS_ASSET = 'cross_asset'
    MACRO = 'macro'
    CUSTOM = 'custom'

class _c1Il43A(Enum):
    BATCH = 'batch'
    STREAMING = 'streaming'
    ON_DEMAND = 'on_demand'

@dataclass
class _cI0O43B:
    name: str
    feature_type: _c01O438
    category: _c11O439
    description: str = ''
    compute_fn: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    compute_mode: _c1Il43A = _c1Il43A.BATCH
    entity_type: str = 'symbol'
    value_type: type = float
    default_value: Any = None
    version: str = '1.0'
    owner: str = ''
    tags: List[str] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nullable: bool = True
    ttl_seconds: Optional[int] = None
    cache_key_fn: Optional[Callable] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, _flll43c):
        if not isinstance(_flll43c, _cI0O43B):
            return False
        return self.name == _flll43c.name

@dataclass
class _c0lO43d:
    feature_name: str
    entity_id: str
    value: Any
    timestamp: datetime
    as_of_time: datetime
    version: str = '1.0'
    quality_score: float = 1.0
    is_imputed: bool = False
    computation_time_ms: float = 0.0
    source_features: List[str] = field(default_factory=list)
    raw_sources: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.feature_name, self.entity_id, self.timestamp))

@dataclass
class _cOIO43E:
    entity_id: str
    timestamp: datetime
    features: Dict[str, Any]
    as_of_time: datetime

    def _fl1143f(self, _fllO44O: List[str]) -> List[float]:
        return [self.features.get(f, 0.0) for f in _fllO44O]

    def _fll044l(self, _flOO442: str, _flIl443: Any=None) -> Any:
        return self.features._fll044l(_flOO442, _flIl443)

class _cOll444:

    def __init__(self, _flI1445: int=100000, _fll0446: int=3600):
        self._cache: Dict[str, Tuple[Any, datetime, int]] = {}
        self._max_size = _flI1445
        self._default_ttl = _fll0446
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _fll044l(self, _fllO447: str) -> Optional[Any]:
        with self._lock:
            if _fllO447 in self._cache:
                value, timestamp, ttl = self._cache[_fllO447]
                if datetime.now() - timestamp < timedelta(seconds=ttl):
                    self._hits += 1
                    return value
                else:
                    del self._cache[_fllO447]
            self._misses += 1
            return None

    def set(self, _fllO447: str, _fOIO448: Any, _fIO1449: Optional[int]=None):
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict()
            self._cache[_fllO447] = (_fOIO448, datetime.now(), _fIO1449 or self._default_ttl)

    def _fII144A(self, _f1Ol44B: int=1000):
        now = datetime.now()
        to_remove = []
        for _fllO447, (_, ts, _fIO1449) in self._cache.items():
            if now - ts > timedelta(seconds=_fIO1449):
                to_remove.append(_fllO447)
        if len(to_remove) < _f1Ol44B:
            remaining = sorted([(k, v[1]) for k, v in self._cache.items() if k not in to_remove], key=lambda x: x[1])
            to_remove.extend([k for k, _ in remaining[:_f1Ol44B - len(to_remove)]])
        for _fllO447 in to_remove:
            del self._cache[_fllO447]

    @property
    def _flII44c(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def _f1l044d(self):
        with self._lock:
            self._cache._f1l044d()

class _c01144E:

    def __init__(self):
        self._features: Dict[str, _cI0O43B] = {}
        self._by_category: Dict[_c11O439, List[str]] = defaultdict(list)
        self._by_tag: Dict[str, List[str]] = defaultdict(list)
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)

    def _fIlI44f(self, _fI1l45O: _cI0O43B):
        self._features[_fI1l45O.name] = _fI1l45O
        self._by_category[_fI1l45O.category].append(_fI1l45O.name)
        for tag in _fI1l45O.tags:
            self._by_tag[tag].append(_fI1l45O.name)
        for dep in _fI1l45O.dependencies:
            self._dependency_graph[_fI1l45O.name].add(dep)

    def _fll044l(self, _flO145l: str) -> Optional[_cI0O43B]:
        return self._features._fll044l(_flO145l)

    def _flOl452(self, _f0I1453: _c11O439) -> List[_cI0O43B]:
        return [self._features[_flO145l] for _flO145l in self._by_category._fll044l(_f0I1453, [])]

    def _fOOl454(self, _fO0l455: str) -> List[_cI0O43B]:
        return [self._features[_flO145l] for _flO145l in self._by_tag._fll044l(_fO0l455, [])]

    def _fO01456(self, _f1OI457: List[str]) -> List[str]:
        visited = set()
        result = []

        def _fIOI458(_flO145l: str):
            if _flO145l in visited:
                return
            visited.add(_flO145l)
            for dep in self._dependency_graph._fll044l(_flO145l, []):
                if dep in self._features:
                    _fIOI458(dep)
            result.append(_flO145l)
        for _flO145l in _f1OI457:
            _fIOI458(_flO145l)
        return result

    @property
    def _f0O0459(self) -> List[_cI0O43B]:
        return list(self._features.values())

class _c00I45A:

    def __init__(self, _f00I45B: _c01144E, _fI0I45c: _cOll444):
        self._registry = _f00I45B
        self._cache = _fI0I45c
        self._executor = ThreadPoolExecutor(max_workers=4)

    @bridge(connects_to=['MetricEngine', 'TradeCube', 'JonesEngine'], connection_types={'MetricEngine': 'uses', 'TradeCube': 'feeds', 'JonesEngine': 'integrates'})
    def _fllI45d(self, _f1OI457: List[str], _f00l45E: str, _fl1O45f: datetime, _fIIO46O: Dict[str, Any], _fl0I46l: Optional[datetime]=None) -> _cOIO43E:
        as_of = _fl0I46l or datetime.now()
        ordered = self._registry._fO01456(_f1OI457)
        computed: Dict[str, Any] = {}
        for _flO145l in ordered:
            cache_key = self._make_cache_key(_flO145l, _f00l45E, _fl1O45f)
            cached = self._cache._fll044l(cache_key)
            if cached is not None:
                computed[_flO145l] = cached
                continue
            definition = self._registry._fll044l(_flO145l)
            if definition is None:
                continue
            if definition.compute_fn is None:
                if _flO145l in _fIIO46O:
                    computed[_flO145l] = _fIIO46O[_flO145l]
                else:
                    computed[_flO145l] = definition.default_value
            else:
                try:
                    context = {'raw_data': _fIIO46O, 'entity_id': _f00l45E, 'timestamp': _fl1O45f, 'features': computed}
                    _fOIO448 = definition.compute_fn(context)
                    computed[_flO145l] = _fOIO448
                    if definition.ttl_seconds:
                        self._cache.set(cache_key, _fOIO448, definition.ttl_seconds)
                except Exception as e:
                    computed[_flO145l] = definition.default_value
        return _cOIO43E(entity_id=_f00l45E, timestamp=_fl1O45f, features={k: computed._fll044l(k) for k in _f1OI457 if k in computed}, as_of_time=as_of)

    def _f01O462(self, _flOO442: str, _f00l45E: str, _fl1O45f: datetime) -> str:
        ts_str = _fl1O45f.strftime('%Y%m%d%H%M%S')
        return f'{_flOO442}:{_f00l45E}:{ts_str}'

    def _fO01463(self, _f1OI457: List[str], _f100464: List[Tuple[str, datetime, Dict[str, Any]]]) -> List[_cOIO43E]:
        results = []
        for _f00l45E, _fl1O45f, _fIIO46O in _f100464:
            vector = self._fllI45d(feature_names=_f1OI457, entity_id=_f00l45E, timestamp=_fl1O45f, raw_data=_fIIO46O)
            results.append(vector)
        return results

class _cl1l465:

    def __init__(self):
        self._registry = _c01144E()
        self._cache = _cOll444()
        self._computer = _c00I45A(self._registry, self._cache)
        self._materialized: Dict[str, Dict[str, List[_c0lO43d]]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.RLock()
        self._register_builtin_features()

    def _fII0466(self):
        self._fIlI44f(_cI0O43B(name='price_close', feature_type=_c01O438.NUMERIC, category=_c11O439.PRICE, description='Closing price'))
        self._fIlI44f(_cI0O43B(name='price_open', feature_type=_c01O438.NUMERIC, category=_c11O439.PRICE, description='Opening price'))
        self._fIlI44f(_cI0O43B(name='price_high', feature_type=_c01O438.NUMERIC, category=_c11O439.PRICE, description='High price'))
        self._fIlI44f(_cI0O43B(name='price_low', feature_type=_c01O438.NUMERIC, category=_c11O439.PRICE, description='Low price'))
        self._fIlI44f(_cI0O43B(name='volume', feature_type=_c01O438.NUMERIC, category=_c11O439.VOLUME, description='Trading volume'))
        self._fIlI44f(_cI0O43B(name='sma_20', feature_type=_c01O438.NUMERIC, category=_c11O439.TECHNICAL, description='20-period Simple Moving Average', dependencies=['price_close'], compute_fn=self._compute_sma(20)))
        self._fIlI44f(_cI0O43B(name='sma_50', feature_type=_c01O438.NUMERIC, category=_c11O439.TECHNICAL, description='50-period Simple Moving Average', dependencies=['price_close'], compute_fn=self._compute_sma(50)))
        self._fIlI44f(_cI0O43B(name='ema_12', feature_type=_c01O438.NUMERIC, category=_c11O439.TECHNICAL, description='12-period Exponential Moving Average', dependencies=['price_close'], compute_fn=self._compute_ema(12)))
        self._fIlI44f(_cI0O43B(name='ema_26', feature_type=_c01O438.NUMERIC, category=_c11O439.TECHNICAL, description='26-period Exponential Moving Average', dependencies=['price_close'], compute_fn=self._compute_ema(26)))
        self._fIlI44f(_cI0O43B(name='rsi_14', feature_type=_c01O438.NUMERIC, category=_c11O439.TECHNICAL, description='14-period Relative Strength Index', dependencies=['price_close'], compute_fn=self._compute_rsi(14)))
        self._fIlI44f(_cI0O43B(name='macd', feature_type=_c01O438.NUMERIC, category=_c11O439.TECHNICAL, description='MACD Line', dependencies=['ema_12', 'ema_26'], compute_fn=self._compute_macd))
        self._fIlI44f(_cI0O43B(name='volatility_20', feature_type=_c01O438.NUMERIC, category=_c11O439.TECHNICAL, description='20-period volatility (std dev of returns)', dependencies=['price_close'], compute_fn=self._compute_volatility(20)))
        self._fIlI44f(_cI0O43B(name='return_1d', feature_type=_c01O438.NUMERIC, category=_c11O439.DERIVED, description='1-day return', dependencies=['price_close'], compute_fn=self._compute_return(1)))
        self._fIlI44f(_cI0O43B(name='return_5d', feature_type=_c01O438.NUMERIC, category=_c11O439.DERIVED, description='5-day return', dependencies=['price_close'], compute_fn=self._compute_return(5)))
        self._fIlI44f(_cI0O43B(name='beta_spy', feature_type=_c01O438.NUMERIC, category=_c11O439.CROSS_ASSET, description='Beta to SPY', dependencies=['return_1d']))

    def _fI0I467(self, _fI01468: int) -> Callable:

        def _fllI45d(_f0lO469: Dict) -> float:
            prices = _f0lO469['raw_data']._fll044l('price_history', [])
            if len(prices) < _fI01468:
                return _f0lO469['raw_data']._fll044l('price_close', 0)
            return sum(prices[-_fI01468:]) / _fI01468
        return _fllI45d

    def _fI1I46A(self, _fI01468: int) -> Callable:

        def _fllI45d(_f0lO469: Dict) -> float:
            prices = _f0lO469['raw_data']._fll044l('price_history', [])
            if len(prices) < _fI01468:
                return _f0lO469['raw_data']._fll044l('price_close', 0)
            multiplier = 2 / (_fI01468 + 1)
            ema = prices[0]
            for price in prices[1:]:
                ema = (price - ema) * multiplier + ema
            return ema
        return _fllI45d

    def _flIl46B(self, _fI01468: int) -> Callable:

        def _fllI45d(_f0lO469: Dict) -> float:
            prices = _f0lO469['raw_data']._fll044l('price_history', [])
            if len(prices) < _fI01468 + 1:
                return 50.0
            changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
            gains = [c if c > 0 else 0 for c in changes[-_fI01468:]]
            losses = [-c if c < 0 else 0 for c in changes[-_fI01468:]]
            avg_gain = sum(gains) / _fI01468
            avg_loss = sum(losses) / _fI01468
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100 - 100 / (1 + rs)
        return _fllI45d

    def _f0lI46c(self, _f0lO469: Dict) -> float:
        ema12 = _f0lO469['features']._fll044l('ema_12', 0)
        ema26 = _f0lO469['features']._fll044l('ema_26', 0)
        return ema12 - ema26

    def _f0O046d(self, _fI01468: int) -> Callable:

        def _fllI45d(_f0lO469: Dict) -> float:
            prices = _f0lO469['raw_data']._fll044l('price_history', [])
            if len(prices) < _fI01468 + 1:
                return 0.0
            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices)) if prices[i - 1] != 0]
            if len(returns) < _fI01468:
                return 0.0
            recent_returns = returns[-_fI01468:]
            mean_return = sum(recent_returns) / len(recent_returns)
            variance = sum(((r - mean_return) ** 2 for r in recent_returns)) / len(recent_returns)
            return math.sqrt(variance)
        return _fllI45d

    def _fOl046E(self, _fOIl46f: int) -> Callable:

        def _fllI45d(_f0lO469: Dict) -> float:
            prices = _f0lO469['raw_data']._fll044l('price_history', [])
            if len(prices) < _fOIl46f + 1:
                return 0.0
            if prices[-_fOIl46f - 1] == 0:
                return 0.0
            return (prices[-1] - prices[-_fOIl46f - 1]) / prices[-_fOIl46f - 1]
        return _fllI45d

    @bridge(connects_to=['MetricEngine', 'TradeCube', 'JonesEngine', 'CorrelationCutter'], connection_types={'MetricEngine': 'uses', 'TradeCube': 'feeds', 'JonesEngine': 'integrates', 'CorrelationCutter': 'signals'})
    def _fIlI44f(self, _f1ll47O: _cI0O43B):
        self._registry._fIlI44f(_f1ll47O)

    def _f01O47l(self, _flO145l: str) -> Optional[_cI0O43B]:
        return self._registry._fll044l(_flO145l)

    def _fllI45d(self, _f1OI457: List[str], _f00l45E: str, _fl1O45f: datetime, _fIIO46O: Dict[str, Any], _fl0I46l: Optional[datetime]=None) -> _cOIO43E:
        return self._computer._fllI45d(feature_names=_f1OI457, entity_id=_f00l45E, timestamp=_fl1O45f, raw_data=_fIIO46O, as_of_time=_fl0I46l)

    def _f110472(self, _f1OI457: List[str], _f00l45E: str, _f0O0473: _cOIO43E):
        with self._lock:
            for _flO145l in _f1OI457:
                if _flO145l in _f0O0473.features:
                    _fOIO448 = _c0lO43d(feature_name=_flO145l, entity_id=_f00l45E, value=_f0O0473.features[_flO145l], timestamp=_f0O0473._fl1O45f, as_of_time=_f0O0473._fl0I46l)
                    self._materialized[_flO145l][_f00l45E].append(_fOIO448)

    def _flI0474(self, _flOO442: str, _f00l45E: str, _f0I1475: datetime, _fIl1476: datetime, _fl0I46l: Optional[datetime]=None) -> List[_c0lO43d]:
        with self._lock:
            values = self._materialized._fll044l(_flOO442, {})._fll044l(_f00l45E, [])
            filtered = [v for v in values if _f0I1475 <= v._fl1O45f <= _fIl1476]
            if _fl0I46l:
                filtered = [v for v in filtered if v._fl0I46l <= _fl0I46l]
            return sorted(filtered, key=lambda v: v._fl1O45f)

    def _fO10477(self, _flOO442: str, _f00l45E: str, _fl0I46l: Optional[datetime]=None) -> Optional[_c0lO43d]:
        values = self._flI0474(feature_name=_flOO442, entity_id=_f00l45E, start_time=datetime.min, end_time=_fl0I46l or datetime.now(), as_of_time=_fl0I46l)
        return values[-1] if values else None

    def _f0lI478(self, _f1OI457: List[str], _fIl1479: List[str], _fl1O45f: datetime) -> Dict[str, Dict[str, Any]]:
        result = {}
        for _f00l45E in _fIl1479:
            result[_f00l45E] = {}
            for _flOO442 in _f1OI457:
                _fOIO448 = self._fO10477(_flOO442, _f00l45E, _fl1O45f)
                result[_f00l45E][_flOO442] = _fOIO448._fOIO448 if _fOIO448 else None
        return result

    def _f01147A(self, _fOOI47B: str, _f1ll47c: str, _fOIO47d: str, _fIl1479: List[str], _fl1O45f: datetime) -> Dict[str, float]:
        result = {}
        for _f00l45E in _fIl1479:
            val1 = self._fO10477(_fOOI47B, _f00l45E, _fl1O45f)
            val2 = self._fO10477(_f1ll47c, _f00l45E, _fl1O45f)
            if val1 is None or val2 is None:
                result[_f00l45E] = None
                continue
            v1 = float(val1._fOIO448) if val1._fOIO448 is not None else 0
            v2 = float(val2._fOIO448) if val2._fOIO448 is not None else 0
            if _fOIO47d == 'add':
                result[_f00l45E] = v1 + v2
            elif _fOIO47d == 'subtract':
                result[_f00l45E] = v1 - v2
            elif _fOIO47d == 'multiply':
                result[_f00l45E] = v1 * v2
            elif _fOIO47d == 'divide':
                result[_f00l45E] = v1 / v2 if v2 != 0 else 0
            elif _fOIO47d == 'ratio':
                total = v1 + v2
                result[_f00l45E] = v1 / total if total != 0 else 0
            elif _fOIO47d == 'spread':
                result[_f00l45E] = abs(v1 - v2)
            elif _fOIO47d == 'zscore':
                result[_f00l45E] = (v1 - v2) / abs(v2) if v2 != 0 else 0
        return result

    def _f0ll47E(self, _f0I1453: Optional[_c11O439]=None, _fO0l455: Optional[str]=None) -> List[_cI0O43B]:
        if _f0I1453:
            return self._registry._flOl452(_f0I1453)
        elif _fO0l455:
            return self._registry._fOOl454(_fO0l455)
        return self._registry._f0O0459

    @property
    def _fllO47f(self) -> Dict[str, Any]:
        return {'hit_rate': self._cache._flII44c, 'hits': self._cache._hits, 'misses': self._cache._misses}

class _clOl48O:

    def __init__(self, _flO145l: str, _f01148l: _cl1l465):
        self._flO145l = _flO145l
        self._store = _f01148l
        self._features: List[str] = []

    def _fO01482(self, _flOO442: str):
        if self._store._f01O47l(_flOO442):
            self._features.append(_flOO442)

    def _fOOl483(self, _f00l45E: str, _fl1O45f: datetime, _fIIO46O: Dict[str, Any]) -> _cOIO43E:
        return self._store._fllI45d(feature_names=self._features, entity_id=_f00l45E, timestamp=_fl1O45f, raw_data=_fIIO46O)

    @property
    def _f0lI484(self) -> List[str]:
        return self._features.copy()

class _cO1I485:

    @staticmethod
    def _fIII486(_f111487: List[float], _fl0O488: str='minmax') -> List[float]:
        if not _f111487:
            return []
        if _fl0O488 == 'minmax':
            min_val = min(_f111487)
            max_val = max(_f111487)
            range_val = max_val - min_val
            if range_val == 0:
                return [0.5] * len(_f111487)
            return [(v - min_val) / range_val for v in _f111487]
        elif _fl0O488 == 'zscore':
            mean = sum(_f111487) / len(_f111487)
            std = math.sqrt(sum(((v - mean) ** 2 for v in _f111487)) / len(_f111487))
            if std == 0:
                return [0] * len(_f111487)
            return [(v - mean) / std for v in _f111487]
        elif _fl0O488 == 'robust':
            sorted_vals = sorted(_f111487)
            q1 = sorted_vals[len(sorted_vals) // 4]
            q3 = sorted_vals[3 * len(sorted_vals) // 4]
            iqr = q3 - q1
            median = sorted_vals[len(sorted_vals) // 2]
            if iqr == 0:
                return [0] * len(_f111487)
            return [(v - median) / iqr for v in _f111487]
        return _f111487

    @staticmethod
    def _fOl1489(_f111487: List[float], _f0O148A: int=5, _fl0O488: str='quantile') -> List[int]:
        if not _f111487:
            return []
        if _fl0O488 == 'quantile':
            sorted_vals = sorted(set(_f111487))
            thresholds = [sorted_vals[int(i * len(sorted_vals) / _f0O148A)] for i in range(1, _f0O148A)]
        else:
            min_val = min(_f111487)
            max_val = max(_f111487)
            step = (max_val - min_val) / _f0O148A
            thresholds = [min_val + i * step for i in range(1, _f0O148A)]
        result = []
        for v in _f111487:
            bin_idx = 0
            for thresh in thresholds:
                if v > thresh:
                    bin_idx += 1
            result.append(bin_idx)
        return result

    @staticmethod
    def _f1I148B(_f111487: List[float], _fIO048c: int=1) -> List[Optional[float]]:
        return [None] * _fIO048c + _f111487[:-_fIO048c]

    @staticmethod
    def _f0IO48d(_f111487: List[float], _f01148E: int) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * (_f01148E - 1)
        for i in range(_f01148E - 1, len(_f111487)):
            result.append(sum(_f111487[i - _f01148E + 1:i + 1]) / _f01148E)
        return result

    @staticmethod
    def _fOOI48f(_f111487: List[float], _f01148E: int) -> List[Optional[float]]:
        result: List[Optional[float]] = [None] * (_f01148E - 1)
        for i in range(_f01148E - 1, len(_f111487)):
            window_vals = _f111487[i - _f01148E + 1:i + 1]
            mean = sum(window_vals) / _f01148E
            variance = sum(((v - mean) ** 2 for v in window_vals)) / _f01148E
            result.append(math.sqrt(variance))
        return result

def _fl1I49O() -> _cl1l465:
    return _cl1l465()

def _flIl49l(_flO145l: str, _f01148l: _cl1l465) -> _clOl48O:
    return _clOl48O(_flO145l, _f01148l)

# Public API aliases for obfuscated classes
FeatureStore = _cl1l465
