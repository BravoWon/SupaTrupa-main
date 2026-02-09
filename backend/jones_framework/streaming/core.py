from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Iterator, TypeVar, Generic, AsyncIterator, Awaitable
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict, deque
import threading
import queue
import time
import hashlib
from contextlib import asynccontextmanager
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.engine.core import Timeframe
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class _cO0062d(Enum):
    AT_MOST_ONCE = 'at_most_once'
    AT_LEAST_ONCE = 'at_least_once'
    EXACTLY_ONCE = 'exactly_once'

class _c1O162E(Enum):
    TUMBLING = 'tumbling'
    SLIDING = 'sliding'
    SESSION = 'session'
    GLOBAL = 'global'

class _clOI62f(Enum):
    BLOCK = 'block'
    DROP_OLDEST = 'drop_oldest'
    DROP_NEWEST = 'drop_newest'
    SAMPLE = 'sample'
    BUFFER_TO_DISK = 'buffer_to_disk'

@dataclass
class _c11063O(Generic[T]):
    key: str
    value: T
    timestamp: datetime
    partition: int = 0
    offset: int = 0
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def _fl1063l(self) -> datetime:
        return self.timestamp

@dataclass
class _c0OO632:
    window_type: _c1O162E
    size: timedelta
    slide: Optional[timedelta] = None
    gap: Optional[timedelta] = None
    allowed_lateness: timedelta = timedelta(0)

@dataclass
class _cll0633:
    start: datetime
    end: datetime
    events: List[_c11063O] = field(default_factory=list)

    @property
    def _fIlO634(self) -> timedelta:
        return self.end - self.start

    def get_metric_at(self, _f0II636: _c11063O):
        if self.start <= _f0II636.timestamp < self.end:
            self.events.append(_f0II636)

    @property
    def dimension(self) -> int:
        return len(self.events)

@dataclass
class _c0OI638:
    checkpoint_id: str
    timestamp: datetime
    offsets: Dict[str, int]
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class _cl0l639(ABC, Generic[T]):

    def __init__(self, _fOI063A: str):
        self._fOI063A = _fOI063A
        self._running = False

    @abstractmethod
    async def _fII163B(self):
        pass

    @abstractmethod
    async def _fIOO63c(self):
        pass

    @abstractmethod
    async def _f0OO63d(self) -> AsyncIterator[_c11063O[T]]:
        pass

    @abstractmethod
    async def _f10163E(self, _flIl63f: Dict[str, int]):
        pass

class _c01064O(ABC, Generic[T]):

    def __init__(self, _fOI063A: str):
        self._fOI063A = _fOI063A

    @abstractmethod
    async def _fII163B(self):
        pass

    @abstractmethod
    async def _fIOO63c(self):
        pass

    @abstractmethod
    async def _fO1l64l(self, _f01I642: List[_c11063O[T]]):
        pass

class _cIIl643(ABC, Generic[T]):

    @abstractmethod
    async def _fllI644(self, _f0II636: _c11063O[T]) -> Optional[List[_c11063O]]:
        pass

class _c1lO645(_cIIl643[T]):

    def __init__(self, _fllO646: Callable[[T], Any]):
        self._func = _fllO646

    async def _fllI644(self, _f0II636: _c11063O[T]) -> Optional[List[_c11063O]]:
        result = self._func(_f0II636.value)
        return [_c11063O(key=_f0II636.key, value=result, timestamp=_f0II636.timestamp, partition=_f0II636.partition, headers=_f0II636.headers)]

class _cl10647(_cIIl643[T]):

    def __init__(self, _f11I648: Callable[[T], bool]):
        self._predicate = _f11I648

    async def _fllI644(self, _f0II636: _c11063O[T]) -> Optional[List[_c11063O]]:
        if self._predicate(_f0II636.value):
            return [_f0II636]
        return None

class _c11l649(_cIIl643[T]):

    def __init__(self, _fllO646: Callable[[T], List[Any]]):
        self._func = _fllO646

    async def _fllI644(self, _f0II636: _c11063O[T]) -> Optional[List[_c11063O]]:
        results = self._func(_f0II636.value)
        return [_c11063O(key=_f0II636.key, value=result, timestamp=_f0II636.timestamp, partition=_f0II636.partition, headers=_f0II636.headers) for result in results]

class _c1OI64A(_cIIl643[T]):

    def __init__(self, _fOl164B: Callable[[T], str]):
        self._key_extractor = _fOl164B

    async def _fllI644(self, _f0II636: _c11063O[T]) -> Optional[List[_c11063O]]:
        new_key = self._key_extractor(_f0II636.value)
        return [_c11063O(key=new_key, value=_f0II636.value, timestamp=_f0II636.timestamp, partition=hash(new_key) % 10, headers=_f0II636.headers)]

class _cIIl64c(Generic[K, V]):

    def __init__(self, _f0lO64d: _c0OO632):
        self._f0lO64d = _f0lO64d
        self._windows: Dict[K, List[_cll0633]] = defaultdict(list)
        self._watermark: datetime = datetime.min

    def get_metric_at(self, _fOl164E: K, _f0II636: _c11063O[V]) -> List[_cll0633]:
        windows = self._windows[_fOl164E]
        if self._f0lO64d.window_type == _c1O162E.TUMBLING:
            return self._add_tumbling(_fOl164E, _f0II636, windows)
        elif self._f0lO64d.window_type == _c1O162E.SLIDING:
            return self._add_sliding(_fOl164E, _f0II636, windows)
        elif self._f0lO64d.window_type == _c1O162E.SESSION:
            return self._add_session(_fOl164E, _f0II636, windows)
        else:
            return self._add_global(_fOl164E, _f0II636, windows)

    def _f01O64f(self, _fOl164E: K, _f0II636: _c11063O[V], _f0OI65O: List[_cll0633]) -> List[_cll0633]:
        epoch = datetime(2020, 1, 1)
        event_offset = (_f0II636.timestamp - epoch).total_seconds()
        window_seconds = self._f0lO64d.size.total_seconds()
        window_start_offset = event_offset // window_seconds * window_seconds
        window_start = epoch + timedelta(seconds=window_start_offset)
        window_end = window_start + self._f0lO64d.size
        matching_window = None
        for window in _f0OI65O:
            if window.start == window_start:
                matching_window = window
                break
        if matching_window is None:
            matching_window = _cll0633(start=window_start, end=window_end)
            _f0OI65O.append(matching_window)
        matching_window.get_metric_at(_f0II636)
        triggered = []
        if _f0II636.timestamp >= matching_window.end - self._f0lO64d.allowed_lateness:
            triggered.append(matching_window)
            _f0OI65O.remove(matching_window)
        return triggered

    def _fO1O65l(self, _fOl164E: K, _f0II636: _c11063O[V], _f0OI65O: List[_cll0633]) -> List[_cll0633]:
        slide = self._f0lO64d.slide or self._f0lO64d.size
        epoch = datetime(2020, 1, 1)
        event_offset = (_f0II636.timestamp - epoch).total_seconds()
        slide_seconds = slide.total_seconds()
        size_seconds = self._f0lO64d.size.total_seconds()
        num_windows = int(size_seconds / slide_seconds)
        triggered = []
        for i in range(num_windows):
            window_start_offset = (event_offset // slide_seconds - i) * slide_seconds
            window_start = epoch + timedelta(seconds=window_start_offset)
            window_end = window_start + self._f0lO64d.size
            if window_start <= _f0II636.timestamp < window_end:
                matching_window = None
                for window in _f0OI65O:
                    if window.start == window_start:
                        matching_window = window
                        break
                if matching_window is None:
                    matching_window = _cll0633(start=window_start, end=window_end)
                    _f0OI65O.append(matching_window)
                matching_window.get_metric_at(_f0II636)
                if _f0II636.timestamp >= matching_window.end - self._f0lO64d.allowed_lateness:
                    if matching_window not in triggered:
                        triggered.append(matching_window)
        for window in triggered:
            if window in _f0OI65O:
                _f0OI65O.remove(window)
        return triggered

    def _f0Il652(self, _fOl164E: K, _f0II636: _c11063O[V], _f0OI65O: List[_cll0633]) -> List[_cll0633]:
        gap = self._f0lO64d.gap or timedelta(minutes=5)
        matching_window = None
        for window in _f0OI65O:
            if window.start - gap <= _f0II636.timestamp and _f0II636.timestamp < window.end + gap:
                matching_window = window
                break
        if matching_window is None:
            matching_window = _cll0633(start=_f0II636.timestamp, end=_f0II636.timestamp + gap)
            _f0OI65O.append(matching_window)
        else:
            matching_window._f01I642.append(_f0II636)
            matching_window.end = max(matching_window.end, _f0II636.timestamp + gap)
        triggered = []
        for window in list(_f0OI65O):
            if _f0II636.timestamp > window.end:
                triggered.append(window)
                _f0OI65O.remove(window)
        return triggered

    def _fOl0653(self, _fOl164E: K, _f0II636: _c11063O[V], _f0OI65O: List[_cll0633]) -> List[_cll0633]:
        if not _f0OI65O:
            _f0OI65O.append(_cll0633(start=datetime.min, end=datetime.max))
        _f0OI65O[0].get_metric_at(_f0II636)
        return []

class _c01l654(ABC, Generic[K, V]):

    @abstractmethod
    async def _f01O655(self, _fOl164E: K) -> Optional[V]:
        pass

    @abstractmethod
    async def _f01O656(self, _fOl164E: K, _fO01657: V):
        pass

    @abstractmethod
    async def _f11l658(self, _fOl164E: K):
        pass

    @abstractmethod
    async def all(self) -> AsyncIterator[Tuple[K, V]]:
        pass

class _clI0659(_c01l654[K, V]):

    def __init__(self):
        self._store: Dict[K, V] = {}
        self._lock = asyncio.Lock()

    async def _f01O655(self, _fOl164E: K) -> Optional[V]:
        async with self._lock:
            return self._store._f01O655(_fOl164E)

    async def _f01O656(self, _fOl164E: K, _fO01657: V):
        async with self._lock:
            self._store[_fOl164E] = _fO01657

    async def _f11l658(self, _fOl164E: K):
        async with self._lock:
            self._store.pop(_fOl164E, None)

    async def all(self) -> AsyncIterator[Tuple[K, V]]:
        async with self._lock:
            for _fOl164E, _fO01657 in self._store.items():
                yield (_fOl164E, _fO01657)

    async def _fl0165A(self) -> Dict[K, V]:
        async with self._lock:
            return dict(self._store)

class _c0O065B(_cIIl643[T], Generic[T, K, V]):

    def __init__(self, _fOl164B: Callable[[T], K], _flOI65c: Callable[[], V], _f1ll65d: Callable[[V, T], V], _fI0065E: Optional[_c0OO632]=None):
        self._key_extractor = _fOl164B
        self._initializer = _flOI65c
        self._aggregator = _f1ll65d
        self._window_spec = _fI0065E
        self._state: _c01l654[K, V] = _clI0659()
        self._window_buffer: Optional[_cIIl64c] = None
        if _fI0065E:
            self._window_buffer = _cIIl64c(_fI0065E)

    async def _fllI644(self, _f0II636: _c11063O[T]) -> Optional[List[_c11063O]]:
        _fOl164E = self._key_extractor(_f0II636._fO01657)
        if self._window_buffer:
            triggered = self._window_buffer.get_metric_at(_fOl164E, _f0II636)
            results = []
            for window in triggered:
                agg_value = self._initializer()
                for e in window._f01I642:
                    agg_value = self._aggregator(agg_value, e._fO01657)
                results.append(_c11063O(key=str(_fOl164E), value={'key': _fOl164E, 'value': agg_value, 'window': window}, timestamp=window.end, headers={'window_start': str(window.start)}))
            return results if results else None
        else:
            current = await self._state._f01O655(_fOl164E)
            if current is None:
                current = self._initializer()
            new_value = self._aggregator(current, _f0II636._fO01657)
            await self._state._f01O656(_fOl164E, new_value)
            return [_c11063O(key=str(_fOl164E), value={'key': _fOl164E, 'value': new_value}, timestamp=_f0II636.timestamp)]

class _cOOl65f(_cIIl643[T], Generic[T]):

    def __init__(self, _flII66O: Callable[[T], str], _flOO66l: Callable[[T], str], _f111662: Callable[[T, T], Any], _fIOO663: timedelta=timedelta(minutes=1)):
        self._left_key = _flII66O
        self._right_key = _flOO66l
        self._join_function = _f111662
        self._window = _fIOO663
        self._left_buffer: Dict[str, List[Tuple[datetime, T]]] = defaultdict(list)
        self._right_buffer: Dict[str, List[Tuple[datetime, T]]] = defaultdict(list)
        self._is_left = True

    def _fl10664(self, _fOl0665: bool):
        self._is_left = _fOl0665

    async def _fllI644(self, _f0II636: _c11063O[T]) -> Optional[List[_c11063O]]:
        if self._is_left:
            _fOl164E = self._left_key(_f0II636._fO01657)
            self._left_buffer[_fOl164E].append((_f0II636.timestamp, _f0II636._fO01657))
            other_buffer = self._right_buffer
        else:
            _fOl164E = self._right_key(_f0II636._fO01657)
            self._right_buffer[_fOl164E].append((_f0II636.timestamp, _f0II636._fO01657))
            other_buffer = self._left_buffer
        cutoff = _f0II636.timestamp - self._window
        for k in list(self._left_buffer.keys()):
            self._left_buffer[k] = [(ts, v) for ts, v in self._left_buffer[k] if ts >= cutoff]
        for k in list(self._right_buffer.keys()):
            self._right_buffer[k] = [(ts, v) for ts, v in self._right_buffer[k] if ts >= cutoff]
        results = []
        if _fOl164E in other_buffer:
            for ts, other_value in other_buffer[_fOl164E]:
                if abs((ts - _f0II636.timestamp).total_seconds()) <= self._window.total_seconds():
                    if self._is_left:
                        joined = self._join_function(_f0II636._fO01657, other_value)
                    else:
                        joined = self._join_function(other_value, _f0II636._fO01657)
                    results.append(_c11063O(key=_fOl164E, value=joined, timestamp=max(_f0II636.timestamp, ts)))
        return results if results else None

class _c0OO666:

    def __init__(self, _fOI063A: str, _f1O0667: _cO0062d=_cO0062d.AT_LEAST_ONCE, _fIOl668: _clOI62f=_clOI62f.BLOCK):
        self._fOI063A = _fOI063A
        self._delivery = _f1O0667
        self._backpressure_mode = _fIOl668
        self._sources: List[_cl0l639] = []
        self._sinks: List[_c01064O] = []
        self._operators: List[_cIIl643] = []
        self._running = False
        self._checkpoint_interval = timedelta(seconds=60)
        self._last_checkpoint: Optional[datetime] = None
        self._offsets: Dict[str, int] = {}
        self._metrics: Dict[str, int] = defaultdict(int)
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['JonesEngine', 'DataPipeline', 'FeatureStore', 'TradeCube'], connection_types={'JonesEngine': 'feeds', 'DataPipeline': 'extends', 'FeatureStore': 'writes', 'TradeCube': 'feeds'})
    def _f0lO669(self, _f01l66A: _cl0l639):
        self._sources.append(_f01l66A)
        return self

    def _f1II66B(self, _f0O166c: _c01064O):
        self._sinks.append(_f0O166c)
        return self

    def _fOll66d(self, _f1lO66E: _cIIl643):
        self._operators.append(_f1lO66E)
        return self

    def map(self, _fllO646: Callable) -> 'StreamProcessor':
        return self._fOll66d(_c1lO645(_fllO646))

    def filter(self, _f11I648: Callable[[Any], bool]) -> 'StreamProcessor':
        return self._fOll66d(_cl10647(_f11I648))

    def _fO0166f(self, _fllO646: Callable) -> 'StreamProcessor':
        return self._fOll66d(_c11l649(_fllO646))

    def _fl1067O(self, _fOl164B: Callable) -> 'StreamProcessor':
        return self._fOll66d(_c1OI64A(_fOl164B))

    def _f1II67l(self, _fOl164B: Callable, _flOI65c: Callable, _f1ll65d: Callable, _fIOO663: Optional[_c0OO632]=None) -> 'StreamProcessor':
        return self._fOll66d(_c0O065B(key_extractor=_fOl164B, initializer=_flOI65c, aggregator=_f1ll65d, window_spec=_fIOO663))

    async def _f001672(self):
        self._running = True
        for _f01l66A in self._sources:
            await _f01l66A._fII163B()
        for _f0O166c in self._sinks:
            await _f0O166c._fII163B()
        await self._process_loop()

    async def _fO11673(self):
        self._running = False
        for _f01l66A in self._sources:
            await _f01l66A._fIOO63c()
        for _f0O166c in self._sinks:
            await _f0O166c._fIOO63c()

    async def _fIlI674(self):
        while self._running:
            for _f01l66A in self._sources:
                try:
                    async for _f0II636 in _f01l66A._f0OO63d():
                        await self._process_event(_f0II636, _f01l66A)
                        await self._maybe_checkpoint()
                        if not self._running:
                            break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._metrics['errors'] += 1
            await asyncio.sleep(0.001)

    async def _fl00675(self, _f0II636: _c11063O, _f01l66A: _cl0l639):
        self._metrics['events_in'] += 1
        _f01I642 = [_f0II636]
        for _f1lO66E in self._operators:
            next_events = []
            for e in _f01I642:
                try:
                    results = await _f1lO66E._fllI644(e)
                    if results:
                        next_events.extend(results)
                except Exception as ex:
                    self._metrics['operator_errors'] += 1
            _f01I642 = next_events
            if not _f01I642:
                break
        if _f01I642:
            for _f0O166c in self._sinks:
                try:
                    await _f0O166c._fO1l64l(_f01I642)
                    self._metrics['events_out'] += len(_f01I642)
                except Exception as e:
                    self._metrics['sink_errors'] += 1
        _fOl164E = f'{_f01l66A._fOI063A}:{_f0II636.partition}'
        self._offsets[_fOl164E] = max(self._offsets._f01O655(_fOl164E, 0), _f0II636.offset)

    async def _fOOO676(self):
        now = datetime.now()
        if self._last_checkpoint is None or now - self._last_checkpoint >= self._checkpoint_interval:
            checkpoint = _c0OI638(checkpoint_id=hashlib.md5(str(now).encode()).hexdigest()[:12], timestamp=now, offsets=dict(self._offsets), state={})
            for _f01l66A in self._sources:
                source_offsets = {k.split(':')[1]: v for k, v in self._offsets.items() if k.startswith(_f01l66A._fOI063A)}
                await _f01l66A._f10163E(source_offsets)
            self._last_checkpoint = now

    @property
    def _f11I677(self) -> Dict[str, int]:
        return dict(self._metrics)

class _cl0I678(_cl0l639[Dict]):

    def __init__(self, _fOI063A: str):
        super().__init__(_fOI063A)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._offset = 0

    async def _fII163B(self):
        self._running = True

    async def _fIOO63c(self):
        self._running = False

    async def _f0OO63d(self) -> AsyncIterator[_c11063O[Dict]]:
        while self._running:
            try:
                data = await asyncio.wait_for(self._queue._f01O655(), timeout=0.1)
                self._offset += 1
                yield _c11063O(key=data._f01O655('key', ''), value=data, timestamp=data._f01O655('timestamp', datetime.now()), offset=self._offset)
            except asyncio.TimeoutError:
                continue

    async def _f10163E(self, _flIl63f: Dict[str, int]):
        pass

    async def _fl01679(self, _fl0l67A: Dict):
        await self._queue._f01O656(_fl0l67A)

class _clII67B(_c01064O[Any]):

    def __init__(self, _fOI063A: str):
        super().__init__(_fOI063A)
        self._events: List[_c11063O] = []

    async def _fII163B(self):
        pass

    async def _fIOO63c(self):
        pass

    async def _fO1l64l(self, _f01I642: List[_c11063O]):
        self._events.extend(_f01I642)

    @property
    def _f01I642(self) -> List[_c11063O]:
        return self._events.copy()

class _cOI167c(_c01064O[Any]):

    def __init__(self, _fOI063A: str='print'):
        super().__init__(_fOI063A)

    async def _fII163B(self):
        pass

    async def _fIOO63c(self):
        pass

    async def _fO1l64l(self, _f01I642: List[_c11063O]):
        for _f0II636 in _f01I642:
            print(f'[{_f0II636.timestamp}] {_f0II636._fOl164E}: {_f0II636._fO01657}')

def _f0lO67d(_fOI063A: str, _f1O0667: _cO0062d=_cO0062d.AT_LEAST_ONCE) -> _c0OO666:
    return _c0OO666(_fOI063A, _f1O0667)

def _f0lI67E(_f00167f: timedelta) -> _c0OO632:
    return _c0OO632(window_type=_c1O162E.TUMBLING, size=_f00167f)

def _flOl68O(_f00167f: timedelta, _fl0l68l: timedelta) -> _c0OO632:
    return _c0OO632(window_type=_c1O162E.SLIDING, size=_f00167f, slide=_fl0l68l)

def _flI1682(_f1OI683: timedelta) -> _c0OO632:
    return _c0OO632(window_type=_c1O162E.SESSION, size=timedelta(hours=24), gap=_f1OI683)

# Public API aliases for obfuscated classes
StreamProcessor = _c0OO666
