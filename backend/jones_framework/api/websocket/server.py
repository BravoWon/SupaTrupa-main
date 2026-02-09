from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set, Coroutine, AsyncIterator
from enum import Enum, auto
import asyncio
import json
import time
import hashlib
import threading
from collections import defaultdict
import struct
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID

class _c1O05O2(Enum):
    CONTINUATION = 0
    TEXT = 1
    BINARY = 2
    CLOSE = 8
    PING = 9
    PONG = 10

class _cO015O3(Enum):
    SUBSCRIBE = 'subscribe'
    UNSUBSCRIBE = 'unsubscribe'
    QUERY = 'query'
    COMMAND = 'command'
    PING = 'ping'
    STATE_UPDATE = 'state_update'
    REGIME_CHANGE = 'regime_change'
    SAFETY_ALERT = 'safety_alert'
    EXPERT_SWAP = 'expert_swap'
    TDA_UPDATE = 'tda_update'
    CORRELATION_BREAK = 'correlation_break'
    ERROR = 'error'
    ACK = 'ack'
    PONG = 'pong'
    # Portal message types
    PORTAL_STATUS = 'portal.status'
    PORTAL_ERROR = 'portal.error'
    PORTAL_PROGRESS = 'portal.progress'
    PORTAL_LOG = 'portal.log'

class _c1Ol5O4(Enum):
    STATES = 'states'
    REGIMES = 'regimes'
    SAFETY = 'safety'
    EXPERTS = 'experts'
    TDA = 'tda'
    CORRELATIONS = 'correlations'
    METRICS = 'metrics'
    PORTAL = 'portal'
    ALL = 'all'

@dataclass
class _c0015O5:
    type: _cO015O3
    payload: Dict[str, Any]
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    message_id: str = ''

    def __post_init__(self):
        if not self.message_id:
            self.message_id = hashlib.sha256(f'{self.type.value}{self.timestamp}{id(self)}'.encode()).hexdigest()[:16]

    def _f0lI5O6(self) -> str:
        return json.dumps({'type': self.type.value, 'payload': self.payload, 'timestamp': self.timestamp, 'messageId': self.message_id})

    @classmethod
    def _fl105O7(cls, _fIlO5O8: str) -> 'WebSocketMessage':
        parsed = json.loads(_fIlO5O8)
        return cls(type=_cO015O3(parsed['type']), payload=parsed.get('payload', {}), timestamp=parsed.get('timestamp', int(time.time() * 1000)), message_id=parsed.get('messageId', ''))

@dataclass
class _cl1I5O9:
    client_id: str
    channel: _c1Ol5O4
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

class _c1OI5OA:

    def __init__(self, _fl0I5OB: str, _fOI05Oc: Optional[Any]=None):
        self._fl0I5OB = _fl0I5OB
        self._fOI05Oc = _fOI05Oc
        self.subscriptions: Dict[_c1Ol5O4, _cl1I5O9] = {}
        self.connected_at = int(time.time() * 1000)
        self.last_ping = self.connected_at
        self.message_count = 0
        self.metadata: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def _fI1I5Od(self, _fI1O5OE: _c0015O5):
        async with self._lock:
            if self._fOI05Oc:
                try:
                    frame = self._encode_frame(_fI1O5OE._f0lI5O6())
                    self._fOI05Oc.write(frame)
                    await self._fOI05Oc.drain()
                    self.message_count += 1
                except Exception:
                    pass

    def _fIII5Of(self, _fIlO5O8: str) -> bytes:
        payload = _fIlO5O8.encode('utf-8')
        length = len(payload)
        frame = bytearray()
        frame.append(129)
        if length < 126:
            frame.append(length)
        elif length < 65536:
            frame.append(126)
            frame.extend(struct.pack('>H', length))
        else:
            frame.append(127)
            frame.extend(struct.pack('>Q', length))
        frame.extend(payload)
        return bytes(frame)

    def _f01I5lO(self, _fOl15ll: _c1Ol5O4, _f0Ol5l2: Optional[Dict[str, Any]]=None):
        self.subscriptions[_fOl15ll] = _cl1I5O9(client_id=self._fl0I5OB, channel=_fOl15ll, filters=_f0Ol5l2 or {})

    def _fIll5l3(self, _fOl15ll: _c1Ol5O4):
        if _fOl15ll in self.subscriptions:
            del self.subscriptions[_fOl15ll]

    def _f1105l4(self, _fOl15ll: _c1Ol5O4) -> bool:
        if _c1Ol5O4.ALL in self.subscriptions:
            return True
        return _fOl15ll in self.subscriptions

    def _fIOO5l5(self, _fOl15ll: _c1Ol5O4, _fIlO5O8: Dict[str, Any]) -> bool:
        if _fOl15ll not in self.subscriptions:
            if _c1Ol5O4.ALL not in self.subscriptions:
                return False
            sub = self.subscriptions.get(_c1Ol5O4.ALL)
        else:
            sub = self.subscriptions[_fOl15ll]
        if not sub or not sub._f0Ol5l2:
            return True
        for key, value in sub._f0Ol5l2.items():
            if key in _fIlO5O8:
                if isinstance(value, list):
                    if _fIlO5O8[key] not in value:
                        return False
                elif _fIlO5O8[key] != value:
                    return False
        return True

@bridge(connects_to=['ConditionState', 'ActivityState', 'RegimeClassifier', 'MixtureOfExperts', 'TDAPipeline', 'ContinuityGuard', 'CorrelationCutter', 'ComponentRegistry', 'JonesGraphQLSchema'], connection_types={'ConditionState': ConnectionType.USES, 'ActivityState': ConnectionType.USES, 'RegimeClassifier': ConnectionType.USES, 'MixtureOfExperts': ConnectionType.USES, 'TDAPipeline': ConnectionType.USES, 'ContinuityGuard': ConnectionType.USES, 'CorrelationCutter': ConnectionType.USES, 'ComponentRegistry': ConnectionType.USES, 'JonesGraphQLSchema': ConnectionType.USES})
class _cI015l6:

    def __init__(self, _f1O15l7: str='0.0.0.0', _f0OI5l8: int=8765, _fI0l5l9: int=1000, _f1Il5lA: int=30, _f10l5lB: int=1000):
        self._f1O15l7 = _f1O15l7
        self._f0OI5l8 = _f0OI5l8
        self._fI0l5l9 = _fI0l5l9
        self._f1Il5lA = _f1Il5lA
        self._f10l5lB = _f10l5lB
        self._connections: Dict[str, _c1OI5OA] = {}
        self._connections_lock = asyncio.Lock()
        self._channel_buffers: Dict[_c1Ol5O4, List[_c0015O5]] = {_fOl15ll: [] for _fOl15ll in _c1Ol5O4}
        self._metrics = {'total_connections': 0, 'active_connections': 0, 'messages_sent': 0, 'messages_received': 0, 'errors': 0}
        self._running = False
        self._server = None
        self._message_handlers: Dict[_cO015O3, List[Callable]] = defaultdict(list)
        self._setup_default_handlers()

    def _fIl15lc(self):
        self._message_handlers[_cO015O3.SUBSCRIBE].append(self._handle_subscribe)
        self._message_handlers[_cO015O3.UNSUBSCRIBE].append(self._handle_unsubscribe)
        self._message_handlers[_cO015O3.PING].append(self._handle_ping)
        self._message_handlers[_cO015O3.QUERY].append(self._handle_query)

    async def _f0O05ld(self):
        self._running = True
        asyncio.create_task(self._heartbeat_loop())
        self._server = await asyncio.start_server(self._handle_connection, self._f1O15l7, self._f0OI5l8)
        async with self._server:
            await self._server.serve_forever()

    async def _fOl15lE(self):
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        async with self._connections_lock:
            for conn in self._connections.values():
                if conn._fOI05Oc:
                    conn._fOI05Oc.close()
            self._connections.clear()

    async def _fO0l5lf(self, _f01O52O: asyncio.StreamReader, _fOI05Oc: asyncio.StreamWriter):
        _fl0I5OB = hashlib.sha256(f'{time.time()}{id(_fOI05Oc)}'.encode()).hexdigest()[:16]
        async with self._connections_lock:
            if len(self._connections) >= self._fI0l5l9:
                _fOI05Oc.close()
                return
            connection = _c1OI5OA(_fl0I5OB, _fOI05Oc)
            self._connections[_fl0I5OB] = connection
            self._metrics['total_connections'] += 1
            self._metrics['active_connections'] += 1
        try:
            await self._websocket_handshake(_f01O52O, _fOI05Oc)
            await connection._fI1I5Od(_c0015O5(type=_cO015O3.ACK, payload={'connectionId': _fl0I5OB, 'serverTime': int(time.time() * 1000), 'channels': [c.value for c in _c1Ol5O4]}))
            while self._running:
                try:
                    _fI1O5OE = await self._read_frame(_f01O52O)
                    if _fI1O5OE is None:
                        break
                    self._metrics['messages_received'] += 1
                    await self._process_message(connection, _fI1O5OE)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    self._metrics['errors'] += 1
                    break
        finally:
            async with self._connections_lock:
                if _fl0I5OB in self._connections:
                    del self._connections[_fl0I5OB]
                    self._metrics['active_connections'] -= 1
            _fOI05Oc.close()

    async def _f10l52l(self, _f01O52O: asyncio.StreamReader, _fOI05Oc: asyncio.StreamWriter):
        request = await _f01O52O.readuntil(b'\r\n\r\n')
        request_str = request.decode('utf-8')
        key = ''
        for line in request_str.split('\r\n'):
            if line.lower().startswith('sec-websocket-key:'):
                key = line.split(':', 1)[1].strip()
                break
        if not key:
            raise ValueError('Missing Sec-WebSocket-Key')
        import hashlib
        import base64
        magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        accept = base64.b64encode(hashlib.sha1((key + magic).encode()).digest()).decode()
        response = f'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n'
        _fOI05Oc.write(response.encode())
        await _fOI05Oc.drain()

    async def _fO1l522(self, _f01O52O: asyncio.StreamReader) -> Optional[str]:
        try:
            header = await asyncio.wait_for(_f01O52O.readexactly(2), timeout=self._f1Il5lA)
            fin = header[0] & 128 != 0
            opcode = header[0] & 15
            masked = header[1] & 128 != 0
            length = header[1] & 127
            if length == 126:
                ext = await _f01O52O.readexactly(2)
                length = struct.unpack('>H', ext)[0]
            elif length == 127:
                ext = await _f01O52O.readexactly(8)
                length = struct.unpack('>Q', ext)[0]
            mask = None
            if masked:
                mask = await _f01O52O.readexactly(4)
            payload = await _f01O52O.readexactly(length)
            if mask:
                payload = bytes((b ^ mask[i % 4] for i, b in enumerate(payload)))
            if opcode == _c1O05O2.CLOSE.value:
                return None
            elif opcode == _c1O05O2.PING.value:
                return '__ping__'
            elif opcode == _c1O05O2.TEXT.value:
                return payload.decode('utf-8')
            return None
        except asyncio.TimeoutError:
            return '__timeout__'
        except Exception:
            return None

    async def _fl0I523(self, _f1II524: _c1OI5OA, _flOO525: str):
        if _flOO525 == '__timeout__':
            return
        if _flOO525 == '__ping__':
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.PONG, payload={}))
            return
        try:
            _fI1O5OE = _c0015O5._fl105O7(_flOO525)
            handlers = self._message_handlers.get(_fI1O5OE.type, [])
            for handler in handlers:
                await handler(_f1II524, _fI1O5OE)
        except json.JSONDecodeError:
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.ERROR, payload={'error': 'Invalid JSON'}))
        except Exception as e:
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.ERROR, payload={'error': str(e)}))

    async def _fOIl526(self, _f1II524: _c1OI5OA, _fI1O5OE: _c0015O5):
        channel_name = _fI1O5OE.payload.get('channel', 'all')
        _f0Ol5l2 = _fI1O5OE.payload.get('filters', {})
        try:
            _fOl15ll = _c1Ol5O4(channel_name)
            _f1II524._f01I5lO(_fOl15ll, _f0Ol5l2)
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.ACK, payload={'action': 'subscribed', 'channel': channel_name, 'filters': _f0Ol5l2}))
            if _fI1O5OE.payload.get('replay', False):
                for buffered in self._channel_buffers.get(_fOl15ll, [])[-10:]:
                    await _f1II524._fI1I5Od(buffered)
        except ValueError:
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.ERROR, payload={'error': f'Unknown channel: {channel_name}'}))

    async def _f0IO527(self, _f1II524: _c1OI5OA, _fI1O5OE: _c0015O5):
        channel_name = _fI1O5OE.payload.get('channel')
        try:
            _fOl15ll = _c1Ol5O4(channel_name)
            _f1II524._fIll5l3(_fOl15ll)
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.ACK, payload={'action': 'unsubscribed', 'channel': channel_name}))
        except ValueError:
            pass

    async def _f0l1528(self, _f1II524: _c1OI5OA, _fI1O5OE: _c0015O5):
        _f1II524.last_ping = int(time.time() * 1000)
        await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.PONG, payload={'serverTime': int(time.time() * 1000)}))

    async def _f0lI529(self, _f1II524: _c1OI5OA, _fI1O5OE: _c0015O5):
        query_type = _fI1O5OE.payload.get('query')
        if query_type == 'status':
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.ACK, payload={'status': 'ok', 'activeConnections': self._metrics['active_connections'], 'subscriptions': [ch.value for ch in _f1II524.subscriptions.keys()]}))
        elif query_type == 'metrics':
            await _f1II524._fI1I5Od(_c0015O5(type=_cO015O3.ACK, payload={'metrics': dict(self._metrics)}))

    async def _fIIO52A(self):
        while self._running:
            await asyncio.sleep(self._f1Il5lA)
            async with self._connections_lock:
                stale = []
                for conn_id, conn in self._connections.items():
                    if time.time() * 1000 - conn.last_ping > self._f1Il5lA * 3 * 1000:
                        stale.append(conn_id)
                    else:
                        await conn._fI1I5Od(_c0015O5(type=_cO015O3.PONG, payload={'heartbeat': True}))
                for conn_id in stale:
                    if conn_id in self._connections:
                        conn = self._connections[conn_id]
                        if conn._fOI05Oc:
                            conn._fOI05Oc.close()
                        del self._connections[conn_id]
                        self._metrics['active_connections'] -= 1

    async def _fllO52B(self, _fOl15ll: _c1Ol5O4, _f1l152c: _cO015O3, _f0O152d: Dict[str, Any]):
        _fI1O5OE = _c0015O5(type=_f1l152c, payload=_f0O152d)
        buffer = self._channel_buffers.get(_fOl15ll, [])
        buffer.append(_fI1O5OE)
        if len(buffer) > self._f10l5lB:
            buffer.pop(0)
        async with self._connections_lock:
            for conn in self._connections.values():
                if conn._f1105l4(_fOl15ll) and conn._fIOO5l5(_fOl15ll, _f0O152d):
                    await conn._fI1I5Od(_fI1O5OE)
                    self._metrics['messages_sent'] += 1

    async def _fIIl52E(self, _fOII52f: ConditionState, _f1O153O: str='market'):
        await self._fllO52B(_c1Ol5O4.STATES, _cO015O3.STATE_UPDATE, {'timestamp': _fOII52f.timestamp, 'vector': list(_fOII52f.vector), 'domain': _f1O153O, 'verified': _fOII52f.verified, 'metadata': _fOII52f.metadata})

    async def _fO1O53l(self, _flII532: RegimeID, _fllI533: RegimeID, _fOII534: float, _f1O153O: str='market'):
        await self._fllO52B(_c1Ol5O4.REGIMES, _cO015O3.REGIME_CHANGE, {'oldRegime': _flII532.name, 'newRegime': _fllI533.name, 'confidence': _fOII534, 'domain': _f1O153O, 'timestamp': int(time.time() * 1000)})

    async def _f100535(self, _f00O536: str, _f0II537: float, _fI0O538: float, _fO10539: List[str], _f1O153O: str='market'):
        await self._fllO52B(_c1Ol5O4.SAFETY, _cO015O3.SAFETY_ALERT, {'level': _f00O536, 'klDivergence': _f0II537, 'riskScore': _fI0O538, 'warnings': _fO10539, 'domain': _f1O153O, 'timestamp': int(time.time() * 1000)})

    async def _fO1I53A(self, _f0I153B: RegimeID, _fII153c: str, _fll153d: str, _f1IO53E: str):
        await self._fllO52B(_c1Ol5O4.EXPERTS, _cO015O3.EXPERT_SWAP, {'regime': _f0I153B.name, 'oldExpert': _fII153c, 'newExpert': _fll153d, 'reason': _f1IO53E, 'timestamp': int(time.time() * 1000)})

    async def _f1ll53f(self, _fIOO54O: Tuple[int, ...], _f0Il54l: float, _fIII542: float):
        await self._fllO52B(_c1Ol5O4.TDA, _cO015O3.TDA_UPDATE, {'bettiNumbers': list(_fIOO54O), 'persistenceEntropy': _f0Il54l, 'bottleneckDistance': _fIII542, 'timestamp': int(time.time() * 1000)})

    async def _f001543(self, _f0Ol544: float, _fO0l545: List[str], _fO1l546: str):
        await self._fllO52B(_c1Ol5O4.CORRELATIONS, _cO015O3.CORRELATION_BREAK, {'breakdownScore': _f0Ol544, 'affectedPairs': _fO0l545, 'regimeImpact': _fO1l546, 'timestamp': int(time.time() * 1000)})

    async def _fP0rtal_status(self, _fStatus: Dict[str, Any]):
        """Broadcast portal component status update."""
        await self._fllO52B(_c1Ol5O4.PORTAL, _cO015O3.PORTAL_STATUS, {**_fStatus, 'timestamp': int(time.time() * 1000)})

    async def _fP0rtal_error(self, _fComponent: str, _fCode: str, _fMessage: str, _fDetails: Optional[Dict[str, Any]] = None):
        """Broadcast portal error."""
        await self._fllO52B(_c1Ol5O4.PORTAL, _cO015O3.PORTAL_ERROR, {'component': _fComponent, 'code': _fCode, 'message': _fMessage, 'details': _fDetails or {}, 'timestamp': int(time.time() * 1000)})

    async def _fP0rtal_progress(self, _fComponent: str, _fCurrent: int, _fTotal: int, _fSuffix: str = ''):
        """Broadcast portal progress update."""
        pct = (_fCurrent / _fTotal * 100) if _fTotal > 0 else 100.0
        await self._fllO52B(_c1Ol5O4.PORTAL, _cO015O3.PORTAL_PROGRESS, {'component': _fComponent, 'current': _fCurrent, 'total': _fTotal, 'percent': pct, 'suffix': _fSuffix, 'timestamp': int(time.time() * 1000)})

    async def _fP0rtal_log(self, _fComponent: str, _fLevel: str, _fMessage: str, _fDetails: Optional[Dict[str, Any]] = None):
        """Broadcast portal log entry."""
        await self._fllO52B(_c1Ol5O4.PORTAL, _cO015O3.PORTAL_LOG, {'component': _fComponent, 'level': _fLevel, 'message': _fMessage, 'details': _fDetails or {}, 'timestamp': int(time.time() * 1000)})

    def _f11I547(self, _f1l152c: _cO015O3, _fIOO548: Callable):
        self._message_handlers[_f1l152c].append(_fIOO548)

    def _f10l549(self) -> Dict[str, Any]:
        return dict(self._metrics)

    def _fOl054A(self) -> int:
        return self._metrics['active_connections']

class _c0lO54B:

    def __init__(self, _f0OI54c: str):
        self._f0OI54c = _f0OI54c
        self._reader = None
        self._writer = None
        self._connected = False
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def _f0Il54d(self):
        _f1O15l7, _f0OI5l8 = self._parse_uri()
        self._reader, self._writer = await asyncio.open_connection(_f1O15l7, _f0OI5l8)
        import base64
        import os
        key = base64.b64encode(os.urandom(16)).decode()
        request = f'GET / HTTP/1.1\r\nHost: {_f1O15l7}:{_f0OI5l8}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'
        self._writer.write(request.encode())
        await self._writer.drain()
        response = await self._reader.readuntil(b'\r\n\r\n')
        if b'101' not in response:
            raise ConnectionError('WebSocket handshake failed')
        self._connected = True
        asyncio.create_task(self._receive_loop())

    def _f1II54E(self) -> Tuple[str, int]:
        _f0OI54c = self._f0OI54c.replace('ws://', '').replace('wss://', '')
        if ':' in _f0OI54c:
            _f1O15l7, _f0OI5l8 = _f0OI54c.split(':')
            return (_f1O15l7, int(_f0OI5l8.split('/')[0]))
        return (_f0OI54c.split('/')[0], 80)

    async def _fIl154f(self):
        while self._connected:
            try:
                header = await self._reader.readexactly(2)
                opcode = header[0] & 15
                length = header[1] & 127
                if length == 126:
                    ext = await self._reader.readexactly(2)
                    length = struct.unpack('>H', ext)[0]
                elif length == 127:
                    ext = await self._reader.readexactly(8)
                    length = struct.unpack('>Q', ext)[0]
                _f0O152d = await self._reader.readexactly(length)
                if opcode == _c1O05O2.TEXT.value:
                    await self._message_queue.put(_f0O152d.decode('utf-8'))
                elif opcode == _c1O05O2.CLOSE.value:
                    self._connected = False
                    break
            except Exception:
                self._connected = False
                break

    async def _fI1I5Od(self, _fI1O5OE: str):
        if not self._connected:
            raise ConnectionError('Not connected')
        _f0O152d = _fI1O5OE.encode('utf-8')
        length = len(_f0O152d)
        import os
        mask = os.urandom(4)
        frame = bytearray()
        frame.append(129)
        if length < 126:
            frame.append(128 | length)
        elif length < 65536:
            frame.append(128 | 126)
            frame.extend(struct.pack('>H', length))
        else:
            frame.append(128 | 127)
            frame.extend(struct.pack('>Q', length))
        frame.extend(mask)
        frame.extend(bytes((b ^ mask[i % 4] for i, b in enumerate(_f0O152d))))
        self._writer.write(bytes(frame))
        await self._writer.drain()

    async def _flI055O(self, _f1OO55l: float=30.0) -> Optional[str]:
        try:
            return await asyncio.wait_for(self._message_queue.get(), _f1OO55l)
        except asyncio.TimeoutError:
            return None

    async def _f01I5lO(self, _fOl15ll: str, _f0Ol5l2: Optional[Dict[str, Any]]=None):
        _fI1O5OE = _c0015O5(type=_cO015O3.SUBSCRIBE, payload={'channel': _fOl15ll, 'filters': _f0Ol5l2 or {}})
        await self._fI1I5Od(_fI1O5OE._f0lI5O6())

    async def _fO01552(self):
        self._connected = False
        if self._writer:
            self._writer._fO01552()

def _f0lI553():
    init_content = '"""WebSocket and API modules for Jones Framework."""\n\nfrom jones_framework.api.websocket.server import (\n    WebSocketServer,\n    WebSocketClient,\n    WebSocketMessage,\n    MessageType,\n    SubscriptionChannel,\n)\n\n__all__ = [\n    \'WebSocketServer\',\n    \'WebSocketClient\',\n    \'WebSocketMessage\',\n    \'MessageType\',\n    \'SubscriptionChannel\',\n]\n'
    return init_content

# Public API aliases for portal methods
_cI015l6.broadcast_portal_status = _cI015l6._fP0rtal_status
_cI015l6.broadcast_portal_error = _cI015l6._fP0rtal_error
_cI015l6.broadcast_portal_progress = _cI015l6._fP0rtal_progress
_cI015l6.broadcast_portal_log = _cI015l6._fP0rtal_log

# Public API aliases
WebSocketOpcode = _c1O05O2
MessageType = _cO015O3
SubscriptionChannel = _c1Ol5O4
WebSocketMessage = _c0015O5
Subscription = _cl1I5O9
WebSocketConnection = _c1OI5OA
WebSocketServer = _cI015l6
WebSocketClient = _c0lO54B

__all__ = ['WebSocketOpcode', 'MessageType', 'SubscriptionChannel', 'WebSocketMessage', 'Subscription', 'WebSocketConnection', 'WebSocketServer', 'WebSocketClient']