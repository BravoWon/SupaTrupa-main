from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID, WarpedMetric, EuclideanMetric
from jones_framework.core.tensor_ops import Tensor
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.domains.base import DomainAdapter, DomainConfig, DomainType, DomainRegimeMapper, StateFactory
from jones_framework.perception.metric_warper import VolatilityCompressionValue

class _cOlI974(Enum):
    LOW_VOL = auto()
    NORMAL_VOL = auto()
    HIGH_VOL = auto()
    VOL_EXPLOSION = auto()
    STRONG_UPTREND = auto()
    WEAK_UPTREND = auto()
    RANGING = auto()
    WEAK_DOWNTREND = auto()
    STRONG_DOWNTREND = auto()
    RISK_ON = auto()
    RISK_OFF = auto()
    ROTATION = auto()
    LIQUID = auto()
    ILLIQUID = auto()
    LIQUIDITY_CRISIS = auto()
    EARNINGS = auto()
    CENTRAL_BANK = auto()
    GEOPOLITICAL = auto()

class _cIll975(DomainRegimeMapper):
    _mapping = {_cOlI974.LOW_VOL: RegimeID.OPTIMAL, _cOlI974.NORMAL_VOL: RegimeID.NORMAL, _cOlI974.HIGH_VOL: RegimeID.STICK_SLIP, _cOlI974.VOL_EXPLOSION: RegimeID.WASHOUT, _cOlI974.STRONG_UPTREND: RegimeID.FORMATION_CHANGE, _cOlI974.WEAK_UPTREND: RegimeID.BIT_BOUNCE, _cOlI974.RANGING: RegimeID.WHIRL, _cOlI974.WEAK_DOWNTREND: RegimeID.PACKOFF, _cOlI974.STRONG_DOWNTREND: RegimeID.FORMATION_CHANGE, _cOlI974.RISK_ON: RegimeID.BIT_BOUNCE, _cOlI974.RISK_OFF: RegimeID.PACKOFF, _cOlI974.ROTATION: RegimeID.TRANSITION, _cOlI974.LIQUID: RegimeID.NORMAL, _cOlI974.ILLIQUID: RegimeID.STICK_SLIP, _cOlI974.LIQUIDITY_CRISIS: RegimeID.WASHOUT, _cOlI974.EARNINGS: RegimeID.STICK_SLIP, _cOlI974.CENTRAL_BANK: RegimeID.LOST_CIRCULATION, _cOlI974.GEOPOLITICAL: RegimeID.KICK}
    _reverse_mapping = {v: k for k, v in _mapping.items()}

    def _fOOI976(self, _flII977: _cOlI974) -> RegimeID:
        return self._mapping.get(_flII977, RegimeID.UNKNOWN)

    def _f111978(self, _fI0O979: RegimeID) -> _cOlI974:
        return self._reverse_mapping.get(_fI0O979, _cOlI974.NORMAL_VOL)

    def _flIl97A(self) -> List[_cOlI974]:
        return list(_cOlI974)

@dataclass
class _cl0I97B(ConditionState):
    symbol: str = ''
    asset_class: str = ''
    exchange: str = ''
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    vwap: float = 0.0
    volume: float = 0.0
    trades: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    depth: int = 0
    spread: float = 0.0
    spread_bps: float = 0.0
    volatility: float = 0.0
    returns: float = 0.0

    @classmethod
    def from_market(cls, _f00097d: str, _f1O197E: float, _fIIl97f: float, _fOl198O: float, _f1Ol98l: float, _f1Il982: Optional[int]=None) -> 'MarketState':
        import time
        ts = _f1Il982 or int(time.time() * 1000)
        spread = _f1Ol98l - _fOl198O
        mid = (_fOl198O + _f1Ol98l) / 2
        vector = (_f1O197E, _fIIl97f, _fOl198O, _f1Ol98l, spread, mid)
        return cls(timestamp=ts, vector=vector, metadata={'domain': 'market', 'symbol': _f00097d, 'source': 'tick'}, symbol=_f00097d, close_price=_f1O197E, volume=_fIIl97f, bid=_fOl198O, ask=_f1Ol98l, spread=spread, spread_bps=spread / mid * 10000 if mid > 0 else 0)

    @classmethod
    def _flOI983(cls, _f00097d: str, _f01l984: float, _fI0l985: float, _flOO986: float, _f0l1987: float, _fIIl97f: float, _f1Il982: Optional[int]=None) -> 'MarketState':
        import time
        ts = _f1Il982 or int(time.time() * 1000)
        typical_price = (_fI0l985 + _flOO986 + _f0l1987) / 3
        true_range = _fI0l985 - _flOO986
        volatility = true_range / typical_price if typical_price > 0 else 0
        returns = (_f0l1987 - _f01l984) / _f01l984 if _f01l984 > 0 else 0
        vector = (_f01l984, _fI0l985, _flOO986, _f0l1987, _fIIl97f, typical_price)
        return cls(timestamp=ts, vector=vector, metadata={'domain': 'market', 'symbol': _f00097d, 'source': 'bar'}, symbol=_f00097d, open_price=_f01l984, high_price=_fI0l985, low_price=_flOO986, close_price=_f0l1987, volume=_fIIl97f, vwap=typical_price, volatility=volatility, returns=returns)

    def _f10l988(self) -> np.ndarray:
        return np.array([self.close_price, self._fIIl97f, self._fOl198O, self._f1Ol98l, self.spread, self.spread_bps, self.volatility, self.returns, self.high_price - self.low_price, self.buy_volume - self.sell_volume])

class _cO0I989(StateFactory[_cl0I97B]):

    def __init__(self, _fIll98A: str=''):
        self._fIll98A = _fIll98A

    def _fO0O98B(self, _fOIl98c: Dict[str, Any], _f1Il982: int) -> _cl0I97B:
        _f00097d = _fOIl98c.get('symbol', self._fIll98A)
        if 'open' in _fOIl98c:
            return _cl0I97B._flOI983(symbol=_f00097d, open_price=_fOIl98c['open'], high=_fOIl98c['high'], low=_fOIl98c['low'], close=_fOIl98c['close'], volume=_fOIl98c.get('volume', 0), timestamp=_f1Il982)
        else:
            return _cl0I97B.from_market(symbol=_f00097d, price=_fOIl98c.get('price', _fOIl98c.get('close', 0)), volume=_fOIl98c.get('volume', 0), bid=_fOIl98c.get('bid', 0), ask=_fOIl98c.get('ask', 0), timestamp=_f1Il982)

    def _f10O98d(self, _fOIl98c: List[Dict[str, Any]]) -> List[_cl0I97B]:
        import time
        base_ts = int(time.time() * 1000)
        return [self._fO0O98B(d, d.get('timestamp', base_ts + i * 1000)) for i, d in enumerate(_fOIl98c)]

@dataclass
class _c00l98E(DomainConfig):
    symbols: List[str] = field(default_factory=list)
    asset_class: str = 'equity'
    exchange: str = ''
    bar_interval: str = '1m'
    use_tick_data: bool = False
    vol_lookback: int = 20
    vol_threshold_low: float = 0.1
    vol_threshold_high: float = 0.3
    spread_threshold_high: float = 0.002
    volume_percentile: float = 0.2
    trend_threshold: float = 0.02
    correlation_window: int = 50

    def __post_init__(self):
        self.domain_type = DomainType.FINANCE

@bridge(connects_to=['ConditionState', 'ActivityState', 'ShadowTensorBuilder', 'TDAPipeline', 'RegimeClassifier', 'MixtureOfExperts', 'VolatilityCompressionValue', 'MetricWarper'], connection_types={'ConditionState': ConnectionType.EXTENDS, 'ShadowTensorBuilder': ConnectionType.USES, 'MixtureOfExperts': ConnectionType.USES, 'VolatilityCompressionValue': ConnectionType.USES})
class _cOOO98f(DomainAdapter[_cl0I97B]):

    def __init__(self, _flOl99O: _c00l98E):
        super().__init__(_flOl99O)
        self.market_config = _flOl99O
        self._volatility_history: List[float] = []
        self._correlation_matrix: Optional[np.ndarray] = None
        self._symbol_states: Dict[str, List[_cl0I97B]] = {}

    def _fll199l(self) -> int:
        return 10

    def _flll992(self) -> int:
        return 4

    def _f1Il993(self) -> DomainRegimeMapper:
        return _cIll975()

    def _f11I994(self) -> StateFactory[_cl0I97B]:
        return _cO0I989(default_symbol=self.market_config.symbols[0] if self.market_config.symbols else '')

    def _fl1l995(self, _f0Ol996: Dict[str, Any]) -> np.ndarray:
        return np.array([_f0Ol996.get('price', 0), _f0Ol996.get('volume', 0), _f0Ol996.get('bid', 0), _f0Ol996.get('ask', 0), _f0Ol996.get('spread', 0), _f0Ol996.get('volatility', 0)])

    def _fOIl997(self, _f0ll998: np.ndarray) -> Dict[str, Any]:
        return {'price': float(_f0ll998[0]), 'volume': float(_f0ll998[1]), 'bid': float(_f0ll998[2]), 'ask': float(_f0ll998[3]), 'spread': float(_f0ll998[4]), 'volatility': float(_f0ll998[5]) if len(_f0ll998) > 5 else 0}

    def _fOIO999(self, _fI0O979: RegimeID) -> ActivityState:
        dim = self._fll199l()
        if _fI0O979 in [RegimeID.BIT_BOUNCE, RegimeID.FORMATION_CHANGE]:
            value_fn = VolatilityCompressionValue(compression_weight=0.5)
        elif _fI0O979 == RegimeID.PACKOFF:
            value_fn = VolatilityCompressionValue(compression_weight=1.0)
        else:
            value_fn = VolatilityCompressionValue(compression_weight=0.3)
        base_metric = EuclideanMetric(dim)
        warped_metric = WarpedMetric(base_metric, lambda x: value_fn(x))
        return ActivityState(regime_id=_fI0O979, manifold_metric=warped_metric, metadata={'domain': 'market'})

    def _fOlI99A(self, _f00097d: str, _f1O197E: float, _fIIl97f: float, _fOl198O: float, _f1Ol98l: float, _f1Il982: Optional[int]=None) -> _cl0I97B:
        state = _cl0I97B.from_market(_f00097d, _f1O197E, _fIIl97f, _fOl198O, _f1Ol98l, _f1Il982)
        self._state_history.append(state)
        if _f00097d not in self._symbol_states:
            self._symbol_states[_f00097d] = []
        self._symbol_states[_f00097d].append(state)
        return state

    def _f01I99B(self, _f00097d: str, _f01l984: float, _fI0l985: float, _flOO986: float, _f0l1987: float, _fIIl97f: float, _f1Il982: Optional[int]=None) -> _cl0I97B:
        state = _cl0I97B._flOI983(_f00097d, _f01l984, _fI0l985, _flOO986, _f0l1987, _fIIl97f, _f1Il982)
        self._state_history.append(state)
        if _f00097d not in self._symbol_states:
            self._symbol_states[_f00097d] = []
        self._symbol_states[_f00097d].append(state)
        self._update_volatility(state)
        return state

    def _fO1I99c(self, _fOlI99d: _cl0I97B):
        self._volatility_history.append(_fOlI99d.volatility)
        if len(self._volatility_history) > self.market_config.vol_lookback:
            self._volatility_history = self._volatility_history[-self.market_config.vol_lookback:]

    def _fI0099E(self, _fl1099f: Optional[int]=None) -> float:
        _fl1099f = _fl1099f or self.market_config.vol_lookback
        if len(self._state_history) < 2:
            return 0.0
        returns = []
        states = self._state_history[-_fl1099f:]
        for i in range(1, len(states)):
            if states[i - 1].close_price > 0:
                ret = np.log(states[i].close_price / states[i - 1].close_price)
                returns.append(ret)
        if not returns:
            return 0.0
        return float(np.std(returns) * np.sqrt(252))

    def _fl0l9AO(self) -> _cOlI974:
        vol = self._fI0099E()
        if vol < self.market_config.vol_threshold_low:
            return _cOlI974.LOW_VOL
        elif vol > self.market_config.vol_threshold_high:
            return _cOlI974.HIGH_VOL
        else:
            return _cOlI974.NORMAL_VOL

    def _fIOl9Al(self, _fl1099f: int=20) -> _cOlI974:
        if len(self._state_history) < _fl1099f:
            return _cOlI974.RANGING
        closes = [s.close_price for s in self._state_history[-_fl1099f:]]
        first = closes[0]
        last = closes[-1]
        if first == 0:
            return _cOlI974.RANGING
        change = (last - first) / first
        if change > self.market_config.trend_threshold * 2:
            return _cOlI974.STRONG_UPTREND
        elif change > self.market_config.trend_threshold:
            return _cOlI974.WEAK_UPTREND
        elif change < -self.market_config.trend_threshold * 2:
            return _cOlI974.STRONG_DOWNTREND
        elif change < -self.market_config.trend_threshold:
            return _cOlI974.WEAK_DOWNTREND
        else:
            return _cOlI974.RANGING

    def _flOl9A2(self) -> _cOlI974:
        if not self._state_history:
            return _cOlI974.LIQUID
        recent = self._state_history[-20:]
        avg_spread_bps = np.mean([s.spread_bps for s in recent])
        avg_volume = np.mean([s._fIIl97f for s in recent])
        if len(self._state_history) > 100:
            historical = self._state_history[-100:-20]
            hist_volume = np.mean([s._fIIl97f for s in historical])
            volume_ratio = avg_volume / hist_volume if hist_volume > 0 else 1
        else:
            volume_ratio = 1
        if avg_spread_bps > self.market_config.spread_threshold_high * 10000:
            if volume_ratio < self.market_config.volume_percentile:
                return _cOlI974.LIQUIDITY_CRISIS
            return _cOlI974.ILLIQUID
        return _cOlI974.LIQUID

    def _f1IO9A3(self) -> Dict[str, _cOlI974]:
        return {'volatility': self._fl0l9AO(), 'trend': self._fIOl9Al(), 'liquidity': self._flOl9A2()}

    def _flOO9A4(self) -> Optional[np.ndarray]:
        if len(self._symbol_states) < 2:
            return None
        symbols = list(self._symbol_states.keys())
        n = len(symbols)
        window = self.market_config.correlation_window
        returns_data = {}
        for _f00097d in symbols:
            states = self._symbol_states[_f00097d][-window:]
            if len(states) < 2:
                returns_data[_f00097d] = []
                continue
            returns = []
            for i in range(1, len(states)):
                if states[i - 1].close_price > 0:
                    ret = np.log(states[i].close_price / states[i - 1].close_price)
                    returns.append(ret)
            returns_data[_f00097d] = returns
        min_len = min((len(r) for r in returns_data.values() if r))
        if min_len < 2:
            return None
        matrix = np.zeros((n, n))
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                r1 = returns_data[sym1][:min_len]
                r2 = returns_data[sym2][:min_len]
                if r1 and r2:
                    matrix[i, j] = np.corrcoef(r1, r2)[0, 1]
                else:
                    matrix[i, j] = 0
        self._correlation_matrix = matrix
        return matrix

    def _f0O09A5(self) -> Dict[str, Any]:
        base_metrics = self.get_metrics()
        base_metrics.update({'realized_vol': self._fI0099E(), 'volatility_regime': self._fl0l9AO().name, 'trend_regime': self._fIOl9Al().name, 'liquidity_regime': self._flOl9A2().name, 'num_symbols': len(self._symbol_states), 'correlation_available': self._correlation_matrix is not None})
        return base_metrics