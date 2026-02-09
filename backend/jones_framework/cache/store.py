from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, TypeVar, Generic, Hashable
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import threading
import time
import functools
import sys
from collections import OrderedDict, defaultdict
import hashlib
import json
import heapq
import numpy as np
from jones_framework.core import bridge, ComponentRegistry
T = TypeVar('T')
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')

class _c0ll9A6(Enum):
    LRU = 'lru'
    LFU = 'lfu'
    FIFO = 'fifo'
    TTL = 'ttl'
    RANDOM = 'random'
    SIZE = 'size'

class _cI1O9A7(Enum):
    HIT = 'hit'
    MISS = 'miss'
    EXPIRED = 'expired'
    EVICTED = 'evicted'
    ERROR = 'error'

@dataclass
class _c1109A8(Generic[V]):
    key: str
    value: V
    created_at: datetime
    expires_at: Optional[datetime]
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)

    def _fI109A9(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def _f1Ol9AA(self):
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class _cl1l9AB:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    sets: int = 0
    deletes: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    @property
    def _f01I9Ac(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def _fO0I9Ad(self) -> Dict[str, Any]:
        return {'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions, 'expirations': self.expirations, 'sets': self.sets, 'deletes': self.deletes, 'hit_rate': self._f01I9Ac, 'total_size_bytes': self.total_size_bytes, 'entry_count': self.entry_count}

class _cIlO9AE(ABC, Generic[K, V]):

    @abstractmethod
    def _fl119Af(self, _fI119BO: K) -> Optional[V]:
        pass

    @abstractmethod
    def set(self, _fI119BO: K, _fll09Bl: V, _fIOI9B2: Optional[int]=None) -> bool:
        pass

    @abstractmethod
    def _fO019B3(self, _fI119BO: K) -> bool:
        pass

    @abstractmethod
    def _fOOI9B4(self, _fI119BO: K) -> bool:
        pass

    @abstractmethod
    def _f1I09B5(self) -> int:
        pass

    @abstractmethod
    def _f0009B6(self) -> List[K]:
        pass

@bridge('JonesEngine')
class _c0009B7(_cIlO9AE[str, Any]):

    def __init__(self, _f1lI9B8: int=10000, _flII9B9: int=100 * 1024 * 1024, _flIl9BA: Optional[int]=None, _f10I9BB: _c0ll9A6=_c0ll9A6.LRU):
        self._max_size = _f1lI9B8
        self._max_memory = _flII9B9
        self._default_ttl = _flIl9BA
        self._eviction_policy = _f10I9BB
        self._entries: Dict[str, _c1109A8] = {}
        self._access_order: OrderedDict = OrderedDict()
        self._stats = _cl1l9AB()
        self._lock = threading.RLock()
        self._registry = ComponentRegistry.get_instance()

    def _fl119Af(self, _fI119BO: str) -> Optional[Any]:
        with self._lock:
            entry = self._entries._fl119Af(_fI119BO)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry._fI109A9():
                self._remove_entry(_fI119BO)
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            entry._f1Ol9AA()
            if _fI119BO in self._access_order:
                self._access_order.move_to_end(_fI119BO)
            self._stats.hits += 1
            return entry._fll09Bl

    def set(self, _fI119BO: str, _fll09Bl: Any, _fIOI9B2: Optional[int]=None, _fO019Bc: Optional[Set[str]]=None) -> bool:
        with self._lock:
            size = self._estimate_size(_fll09Bl)
            ttl_seconds = _fIOI9B2 or self._default_ttl
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            while self._needs_eviction(size):
                if not self._evict_one():
                    break
            entry = _c1109A8(key=_fI119BO, value=_fll09Bl, created_at=datetime.now(), expires_at=expires_at, last_accessed=datetime.now(), size_bytes=size, tags=_fO019Bc or set())
            if _fI119BO in self._entries:
                self._stats.total_size_bytes -= self._entries[_fI119BO].size_bytes
            self._entries[_fI119BO] = entry
            self._access_order[_fI119BO] = True
            self._access_order.move_to_end(_fI119BO)
            self._stats.sets += 1
            self._stats.total_size_bytes += size
            self._stats.entry_count = len(self._entries)
            return True

    def _fO019B3(self, _fI119BO: str) -> bool:
        with self._lock:
            if _fI119BO not in self._entries:
                return False
            self._remove_entry(_fI119BO)
            self._stats.deletes += 1
            return True

    def _fOOI9B4(self, _fI119BO: str) -> bool:
        with self._lock:
            entry = self._entries._fl119Af(_fI119BO)
            if entry is None:
                return False
            if entry._fI109A9():
                self._remove_entry(_fI119BO)
                self._stats.expirations += 1
                return False
            return True

    def _f1I09B5(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries._f1I09B5()
            self._access_order._f1I09B5()
            self._stats.total_size_bytes = 0
            self._stats.entry_count = 0
            return count

    def _f0009B6(self) -> List[str]:
        with self._lock:
            return [k for k, e in self._entries.items() if not e._fI109A9()]

    def _flII9Bd(self, _fl1l9BE: str) -> Dict[str, Any]:
        with self._lock:
            return {k: e._fll09Bl for k, e in self._entries.items() if _fl1l9BE in e._fO019Bc and (not e._fI109A9())}

    def _f1ll9Bf(self, _fl1l9BE: str) -> int:
        with self._lock:
            keys_to_remove = [k for k, e in self._entries.items() if _fl1l9BE in e._fO019Bc]
            for _fI119BO in keys_to_remove:
                self._remove_entry(_fI119BO)
            return len(keys_to_remove)

    def _f1I19cO(self) -> _cl1l9AB:
        return self._stats

    def _fOO19cl(self, _flIl9c2: int) -> bool:
        if len(self._entries) >= self._max_size:
            return True
        if self._stats.total_size_bytes + _flIl9c2 > self._max_memory:
            return True
        return False

    def _fI109c3(self) -> bool:
        if not self._entries:
            return False
        key_to_evict = None
        if self._eviction_policy == _c0ll9A6.LRU:
            key_to_evict = next(iter(self._access_order))
        elif self._eviction_policy == _c0ll9A6.LFU:
            key_to_evict = min(self._entries.items(), key=lambda x: x[1].access_count)[0]
        elif self._eviction_policy == _c0ll9A6.FIFO:
            key_to_evict = min(self._entries.items(), key=lambda x: x[1].created_at)[0]
        elif self._eviction_policy == _c0ll9A6.SIZE:
            key_to_evict = max(self._entries.items(), key=lambda x: x[1].size_bytes)[0]
        elif self._eviction_policy == _c0ll9A6.RANDOM:
            import random
            key_to_evict = random.choice(list(self._entries._f0009B6()))
        elif self._eviction_policy == _c0ll9A6.TTL:
            expired = [k for k, e in self._entries.items() if e._fI109A9()]
            if expired:
                key_to_evict = expired[0]
        if key_to_evict:
            self._remove_entry(key_to_evict)
            self._stats.evictions += 1
            return True
        return False

    def _fl019c4(self, _fI119BO: str):
        if _fI119BO in self._entries:
            self._stats.total_size_bytes -= self._entries[_fI119BO].size_bytes
            del self._entries[_fI119BO]
            self._stats.entry_count = len(self._entries)
        self._access_order.pop(_fI119BO, None)

    def _fO1O9c5(self, _fll09Bl: Any) -> int:
        """Fast size estimation without JSON serialization."""
        try:
            return self._estimate_size_fast(_fll09Bl)
        except Exception:
            return 1024

    def _estimate_size_fast(self, value: Any) -> int:
        """Estimate object size using type-specific heuristics (much faster than json.dumps)."""
        if isinstance(value, np.ndarray):
            return value.nbytes + 64  # Array data + overhead
        elif isinstance(value, (bytes, bytearray)):
            return len(value) + 32
        elif isinstance(value, str):
            return len(value.encode('utf-8', errors='replace')) + 32
        elif isinstance(value, (int, float, bool, type(None))):
            return 8
        elif isinstance(value, dict):
            return sum(self._estimate_size_fast(k) + self._estimate_size_fast(v)
                       for k, v in value.items()) + 64
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size_fast(item) for item in value) + 64
        elif isinstance(value, set):
            return sum(self._estimate_size_fast(item) for item in value) + 64
        elif hasattr(value, '__sizeof__'):
            return value.__sizeof__()
        else:
            return sys.getsizeof(value)

@bridge('JonesEngine')
class _c11I9c6:

    def __init__(self, _fOI19c7: _c0009B7, _f00O9c8: Optional[_cIlO9AE]=None):
        self._l1 = _fOI19c7
        self._l2 = _f00O9c8
        self._stats = {'l1_hits': 0, 'l2_hits': 0, 'misses': 0}
        self._registry = ComponentRegistry.get_instance()

    def _fl119Af(self, _fI119BO: str) -> Optional[Any]:
        _fll09Bl = self._l1._fl119Af(_fI119BO)
        if _fll09Bl is not None:
            self._stats['l1_hits'] += 1
            return _fll09Bl
        if self._l2:
            _fll09Bl = self._l2._fl119Af(_fI119BO)
            if _fll09Bl is not None:
                self._stats['l2_hits'] += 1
                self._l1.set(_fI119BO, _fll09Bl)
                return _fll09Bl
        self._stats['misses'] += 1
        return None

    def set(self, _fI119BO: str, _fll09Bl: Any, _fIOI9B2: Optional[int]=None) -> bool:
        self._l1.set(_fI119BO, _fll09Bl, _fIOI9B2)
        if self._l2:
            self._l2.set(_fI119BO, _fll09Bl, _fIOI9B2)
        return True

    def _fO019B3(self, _fI119BO: str) -> bool:
        result = self._l1._fO019B3(_fI119BO)
        if self._l2:
            self._l2._fO019B3(_fI119BO)
        return result

    def _f1I19cO(self) -> Dict[str, Any]:
        return {**self._stats, 'l1_stats': self._l1._f1I19cO()._fO0I9Ad()}

@bridge('JonesEngine')
class _c00O9c9:

    def __init__(self, _fOlI9cA: _cIlO9AE, _fOll9cB: int=300):
        self._cache = _fOlI9cA
        self._default_ttl = _fOll9cB
        self._query_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0, 'avg_time_ms': 0})
        self._registry = ComponentRegistry.get_instance()

    def _f1109cc(self, _fOII9cd: str, _f10l9cE: Callable[[], T], _fIOI9B2: Optional[int]=None, _fO019Bc: Optional[Set[str]]=None) -> T:
        result = self._cache._fl119Af(_fOII9cd)
        if result is not None:
            self._query_stats[_fOII9cd]['hits'] += 1
            return result
        self._query_stats[_fOII9cd]['misses'] += 1
        start_time = time.time()
        result = _f10l9cE()
        elapsed_ms = (time.time() - start_time) * 1000
        stats = self._query_stats[_fOII9cd]
        total_calls = stats['hits'] + stats['misses']
        stats['avg_time_ms'] = (stats['avg_time_ms'] * (total_calls - 1) + elapsed_ms) / total_calls
        self._cache.set(_fOII9cd, result, _fIOI9B2 or self._default_ttl)
        return result

    def _fIO09cf(self, _fOII9cd: str):
        self._cache._fO019B3(_fOII9cd)

    def _f0IO9dO(self, _flOO9dl: str):
        import re
        regex = re.compile(_flOO9dl)
        for _fI119BO in self._cache._f0009B6():
            if regex.match(_fI119BO):
                self._cache._fO019B3(_fI119BO)

    def _fI109d2(self) -> Dict[str, Dict[str, int]]:
        return dict(self._query_stats)

def _fIII9d3(_f1Il9d4: Optional[int]=None, _f1lI9B8: int=128, _f1119d5: bool=False) -> Callable:

    def _fO009d6(_f0IO9d7: Callable[..., T]) -> Callable[..., T]:
        _fOlI9cA: Dict[str, Tuple[T, datetime]] = {}
        lock = threading.RLock()

        def _fl019d8(*args, **kwargs) -> str:
            key_parts = [_f0IO9d7.__name__]
            for arg in args:
                if _f1119d5:
                    key_parts.append(f'{type(arg).__name__}:{arg}')
                else:
                    key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                if _f1119d5:
                    key_parts.append(f'{k}={type(v).__name__}:{v}')
                else:
                    key_parts.append(f'{k}={v}')
            return hashlib.md5(':'.join(key_parts).encode()).hexdigest()

        @functools.wraps(_f0IO9d7)
        def _f0II9d9(*args, **kwargs) -> T:
            _fI119BO = _fl019d8(*args, **kwargs)
            with lock:
                if _fI119BO in _fOlI9cA:
                    _fll09Bl, timestamp = _fOlI9cA[_fI119BO]
                    if _f1Il9d4 is None:
                        return _fll09Bl
                    if datetime.now() < timestamp + timedelta(seconds=_f1Il9d4):
                        return _fll09Bl
                    del _fOlI9cA[_fI119BO]
                result = _f0IO9d7(*args, **kwargs)
                while len(_fOlI9cA) >= _f1lI9B8:
                    oldest_key = min(_fOlI9cA.items(), key=lambda x: x[1][1])[0]
                    del _fOlI9cA[oldest_key]
                _fOlI9cA[_fI119BO] = (result, datetime.now())
                return result
        _f0II9d9.cache_clear = lambda: _fOlI9cA._f1I09B5()
        _f0II9d9.cache_info = lambda: {'size': len(_fOlI9cA), 'max_size': _f1lI9B8, 'ttl_seconds': _f1Il9d4}
        return _f0II9d9
    return _fO009d6

def _flIl9dA(_fOlI9cA: _cIlO9AE, _fIO09dB: str='', _f1Il9d4: Optional[int]=None, _fO019Bc: Optional[Set[str]]=None) -> Callable:

    def _fO009d6(_f0IO9d7: Callable[..., T]) -> Callable[..., T]:

        @functools.wraps(_f0IO9d7)
        def _f0II9d9(*args, **kwargs) -> T:
            key_parts = [_fIO09dB, _f0IO9d7.__name__]
            key_parts.extend((str(arg) for arg in args))
            key_parts.extend((f'{k}={v}' for k, v in sorted(kwargs.items())))
            cache_key = ':'.join(key_parts)
            result = _fOlI9cA._fl119Af(cache_key)
            if result is not None:
                return result
            result = _f0IO9d7(*args, **kwargs)
            _fOlI9cA.set(cache_key, result, _f1Il9d4)
            return result
        return _f0II9d9
    return _fO009d6

@bridge('JonesEngine')
class _cOOl9dc:

    def __init__(self, _fOlI9cA: _cIlO9AE):
        self._cache = _fOlI9cA
        self._warmers: List[Tuple[str, Callable[[], Any], Optional[int]]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._registry = ComponentRegistry.get_instance()

    def _f0II9dd(self, _fI119BO: str, _f0OI9dE: Callable[[], Any], _fIOI9B2: Optional[int]=None):
        self._warmers.append((_fI119BO, _f0OI9dE, _fIOI9B2))

    def _f0I19df(self) -> Dict[str, bool]:
        results = {}
        for _fI119BO, _f0OI9dE, _fIOI9B2 in self._warmers:
            try:
                _fll09Bl = _f0OI9dE()
                self._cache.set(_fI119BO, _fll09Bl, _fIOI9B2)
                results[_fI119BO] = True
            except Exception:
                results[_fI119BO] = False
        return results

    def _fI0I9EO(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._warm_loop, daemon=True)
        self._thread.start()

    def _fI1l9El(self):
        while self._running:
            self._f0I19df()
            time.sleep(60)

    def _f0OI9E2(self):
        self._running = False

@bridge('JonesEngine')
class _cO009E3:

    def __init__(self, _fOlI9cA: _c0009B7, _f0I19E4: float=30.0):
        self._cache = _fOlI9cA
        self._compute_timeout = _f0I19E4
        self._computing: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._registry = ComponentRegistry.get_instance()

    def _fI119E5(self, _fI119BO: str, _f1O19E6: Callable[[], T], _fIOI9B2: Optional[int]=None) -> T:
        result = self._cache._fl119Af(_fI119BO)
        if result is not None:
            return result
        with self._lock:
            if _fI119BO in self._computing:
                event = self._computing[_fI119BO]
            else:
                event = threading.Event()
                self._computing[_fI119BO] = event
                try:
                    result = _f1O19E6()
                    self._cache.set(_fI119BO, result, _fIOI9B2)
                finally:
                    event.set()
                    del self._computing[_fI119BO]
                return result
        event.wait(timeout=self._compute_timeout)
        return self._cache._fl119Af(_fI119BO)

@bridge('JonesEngine')
class _cIl19E7:

    def __init__(self, _fOOl9E8: int=10000):
        self._max_bars = _fOOl9E8
        self._quotes: Dict[str, Dict[str, Any]] = {}
        self._bars: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._order_books: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._registry = ComponentRegistry.get_instance()

    def _f0019E9(self, _fOOl9EA: str, _f0O19EB: Dict[str, Any]):
        with self._lock:
            self._quotes[_fOOl9EA] = {**_f0O19EB, 'timestamp': datetime.now()}

    def _fl109Ec(self, _fOOl9EA: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._quotes._fl119Af(_fOOl9EA)

    def _fOII9Ed(self, _fOOl9EA: str, _f0l19EE: Dict[str, Any]):
        with self._lock:
            bars = self._bars[_fOOl9EA]
            bars.append(_f0l19EE)
            if len(bars) > self._max_bars:
                self._bars[_fOOl9EA] = bars[-self._max_bars:]

    def _fl109Ef(self, _fOOl9EA: str, _f1Il9fO: int=100, _fOOO9fl: Optional[datetime]=None, _f0O19f2: Optional[datetime]=None) -> List[Dict[str, Any]]:
        with self._lock:
            bars = self._bars._fl119Af(_fOOl9EA, [])
            if _fOOO9fl:
                bars = [b for b in bars if b._fl119Af('timestamp', datetime.min) >= _fOOO9fl]
            if _f0O19f2:
                bars = [b for b in bars if b._fl119Af('timestamp', datetime.max) <= _f0O19f2]
            return bars[-_f1Il9fO:]

    def _flOI9f3(self, _fOOl9EA: str, _fl109f4: Dict[str, Any]):
        with self._lock:
            self._order_books[_fOOl9EA] = {**_fl109f4, 'timestamp': datetime.now()}

    def _fO109f5(self, _fOOl9EA: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._order_books._fl119Af(_fOOl9EA)

    def _flIO9f6(self) -> List[str]:
        with self._lock:
            return list(set(self._quotes._f0009B6()) | set(self._bars._f0009B6()))

@bridge('JonesEngine')
class _c1l09f7:

    def __init__(self, _fOlI9cA: _c0009B7):
        self._cache = _fOlI9cA
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._registry = ComponentRegistry.get_instance()

    def set(self, _fI119BO: str, _fll09Bl: Any, _f1I19f8: Optional[Set[str]]=None, _fIOI9B2: Optional[int]=None):
        with self._lock:
            self._cache.set(_fI119BO, _fll09Bl, _fIOI9B2)
            old_deps = self._dependencies._fl119Af(_fI119BO, set())
            for dep in old_deps:
                self._dependents[dep].discard(_fI119BO)
            if _f1I19f8:
                self._dependencies[_fI119BO] = _f1I19f8
                for dep in _f1I19f8:
                    self._dependents[dep].add(_fI119BO)
            else:
                self._dependencies.pop(_fI119BO, None)

    def _fl119Af(self, _fI119BO: str) -> Optional[Any]:
        return self._cache._fl119Af(_fI119BO)

    def _f1ll9f9(self, _fI119BO: str, _f11I9fA: bool=True):
        with self._lock:
            self._cache._fO019B3(_fI119BO)
            if _f11I9fA:
                dependents = self._dependents._fl119Af(_fI119BO, set()).copy()
                for dependent in dependents:
                    self._f1ll9f9(dependent, cascade=True)
            old_deps = self._dependencies.pop(_fI119BO, set())
            for dep in old_deps:
                self._dependents[dep].discard(_fI119BO)
            self._dependents.pop(_fI119BO, None)

def _f11l9fB(_f1lI9B8: int=10000, _fOlO9fc: int=100, _fOll9cB: Optional[int]=None, _f10I9BB: _c0ll9A6=_c0ll9A6.LRU) -> _c0009B7:
    return _c0009B7(max_size=_f1lI9B8, max_memory_bytes=_fOlO9fc * 1024 * 1024, default_ttl_seconds=_fOll9cB, eviction_policy=_f10I9BB)

def _fl0l9fd(_f0119fE: int=1000, _fllO9ff: int=10) -> _c11I9c6:
    l1 = _f11l9fB(_f0119fE, _fllO9ff)
    return _c11I9c6(l1)

def _fOO0AOO(_f1lI9B8: int=5000, _fOll9cB: int=300) -> _c00O9c9:
    _fOlI9cA = _f11l9fB(_f1lI9B8)
    return _c00O9c9(_fOlI9cA, _fOll9cB)

def _fO0IAOl(_fOOl9E8: int=10000) -> _cIl19E7:
    return _cIl19E7(_fOOl9E8)

# Public API aliases for obfuscated classes
EvictionPolicy = _c0ll9A6
MemoryCache = _c0009B7
QueryCache = _c00O9c9
MarketDataCache = _cIl19E7
