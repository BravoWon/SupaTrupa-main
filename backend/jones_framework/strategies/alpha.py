from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import math
from collections import defaultdict
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.trading.execution.order_manager import Order, OrderSide, OrderType, Position
from jones_framework.trading.signals.signal_engine import Signal, SignalType, SignalSource

class _c100l(Enum):
    MOMENTUM = 'momentum'
    MEAN_REVERSION = 'mean_reversion'
    TREND_FOLLOWING = 'trend_following'
    STATISTICAL_ARBITRAGE = 'statistical_arbitrage'
    FACTOR = 'factor'
    EVENT_DRIVEN = 'event_driven'
    MARKET_MAKING = 'market_making'
    PAIRS = 'pairs'
    SECTOR_ROTATION = 'sector_rotation'
    MULTI_STRATEGY = 'multi_strategy'

class _c0OO2(Enum):
    INACTIVE = 'inactive'
    WARMING_UP = 'warming_up'
    ACTIVE = 'active'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    ERROR = 'error'

class _c0103(Enum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'
    STEP = 'step'
    NONE = 'none'

@dataclass
class _c1lI4:
    name: str
    category: str
    weight: float = 1.0
    decay_type: _c0103 = _c0103.EXPONENTIAL
    decay_halflife_days: float = 5.0
    zscore_normalize: bool = True
    winsorize_std: float = 3.0
    compute_fn: Optional[Callable] = None
    lookback_days: int = 20
    description: str = ''
    universe_filter: Optional[str] = None

@dataclass
class _c1O05:
    symbol: str
    factor_name: str
    raw_value: float
    normalized_value: float
    weight: float
    timestamp: datetime
    decay_factor: float = 1.0

    @property
    def _flI06(self) -> float:
        return self.normalized_value * self.weight * self.decay_factor

@dataclass
class _c0II7:
    name: str
    strategy_type: _c100l
    universe: List[str]
    warmup_periods: int = 20
    rebalance_frequency: str = 'daily'
    max_position_pct: float = 0.1
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    min_position_size: float = 1000.0
    max_drawdown_pct: float = 0.2
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    max_sector_exposure: float = 0.3
    expected_slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class _cOIl8:
    timestamp: datetime
    total_return: float = 0.0
    daily_return: float = 0.0
    mtd_return: float = 0.0
    ytd_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    num_positions: int = 0
    turnover: float = 0.0

class _c1l19(ABC):

    def __init__(self, _f100A: _c0II7):
        self._f100A = _f100A
        self._state = _c0OO2.INACTIVE
        self._positions: Dict[str, Position] = {}
        self._pending_orders: List[Order] = []
        self._signals: List[Signal] = []
        self._metrics_history: List[_cOIl8] = []
        self._warmup_count = 0
        self._high_watermark = 0.0
        self._registry = ComponentRegistry.get_instance()

    @property
    def _fI1lB(self) -> str:
        return self._f100A._fI1lB

    @property
    def _fI10c(self) -> _c0OO2:
        return self._state

    @property
    def _fl1Od(self) -> bool:
        return self._state == _c0OO2.ACTIVE

    @abstractmethod
    def _fl10E(self, _fII0f: Dict[str, Dict[str, float]], _fIIllO: Dict[str, Dict[str, float]]) -> List[Signal]:
        pass

    @abstractmethod
    def _f0I1ll(self, _fllOl2: List[Signal], _fO10l3: Dict[str, Position], _fOl0l4: float) -> Dict[str, float]:
        pass

    def _fl1Il5(self, _fII0f: Dict[str, Dict[str, float]], _fIIllO: Dict[str, Dict[str, float]], _fOl0l4: float) -> List[Order]:
        if self._state == _c0OO2.WARMING_UP:
            self._warmup_count += 1
            if self._warmup_count >= self._f100A.warmup_periods:
                self._state = _c0OO2.ACTIVE
            return []
        if not self._fl1Od:
            return []
        _fllOl2 = self._fl10E(_fII0f, _fIIllO)
        self._signals = _fllOl2
        targets = self._f0I1ll(_fllOl2, self._positions, _fOl0l4)
        targets = self._apply_risk_controls(targets, _fOl0l4)
        orders = self._generate_rebalance_orders(targets, _fII0f, _fOl0l4)
        return orders

    def _f1l1l6(self, _f101l7: Order, _f1l0l8: float, _f10Il9: int):
        symbol = _f101l7.symbol
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0, unrealized_pnl=0, realized_pnl=0)
        pos = self._positions[symbol]
        if _f101l7.side == OrderSide.BUY:
            new_qty = pos.quantity + _f10Il9
            if pos.quantity >= 0:
                total_cost = pos.avg_price * pos.quantity + _f1l0l8 * _f10Il9
                pos.avg_price = total_cost / new_qty if new_qty > 0 else 0
            pos.quantity = new_qty
        else:
            new_qty = pos.quantity - _f10Il9
            if pos.quantity > 0:
                realized = (_f1l0l8 - pos.avg_price) * _f10Il9
                pos.realized_pnl += realized
            pos.quantity = new_qty
        if pos.quantity == 0:
            del self._positions[symbol]

    def _fIl1lA(self, _flOllB: Dict[str, float], _fOl0l4: float) -> Dict[str, float]:
        adjusted = dict(_flOllB)
        for symbol, weight in adjusted.items():
            adjusted[symbol] = max(-self._f100A.max_position_pct, min(self._f100A.max_position_pct, weight))
        gross = sum((abs(w) for w in adjusted.values()))
        if gross > self._f100A.max_gross_exposure:
            scale = self._f100A.max_gross_exposure / gross
            adjusted = {s: w * scale for s, w in adjusted.items()}
        net = sum(adjusted.values())
        if abs(net) > self._f100A.max_net_exposure:
            excess = abs(net) - self._f100A.max_net_exposure
            total_abs = sum((abs(w) for w in adjusted.values()))
            if total_abs > 0:
                for symbol in adjusted:
                    sign = 1 if adjusted[symbol] > 0 else -1
                    reduction = abs(adjusted[symbol]) / total_abs * excess * sign
                    adjusted[symbol] -= reduction
        return adjusted

    def _f010lc(self, _flOllB: Dict[str, float], _fII0f: Dict[str, Dict[str, float]], _fOl0l4: float) -> List[Order]:
        orders = []
        current_weights = {}
        for symbol, pos in self._positions.items():
            price = _fII0f.get(symbol, {}).get('close', pos.avg_price)
            value = pos.quantity * price
            current_weights[symbol] = value / _fOl0l4 if _fOl0l4 > 0 else 0
        all_symbols = set(_flOllB.keys()) | set(current_weights.keys())
        for symbol in all_symbols:
            target = _flOllB.get(symbol, 0)
            current = current_weights.get(symbol, 0)
            delta = target - current
            if abs(delta) < 0.001:
                continue
            price = _fII0f.get(symbol, {}).get('close', 0)
            if price <= 0:
                continue
            value = delta * _fOl0l4
            quantity = int(abs(value) / price)
            if quantity * price < self._f100A.min_position_size:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(order_id=f'{self._fI1lB}_{symbol}_{datetime.now().timestamp()}', symbol=symbol, side=side, quantity=quantity, order_type=OrderType.MARKET))
        return orders

    def _f1Illd(self):
        self._state = _c0OO2.WARMING_UP
        self._warmup_count = 0

    def _fO11lE(self):
        self._state = _c0OO2.STOPPED

    def _f0IIlf(self):
        self._state = _c0OO2.PAUSED

    def _f11l2O(self):
        if self._state == _c0OO2.PAUSED:
            self._state = _c0OO2.ACTIVE

class _cI0I2l(_c1l19):

    def __init__(self, _f100A: _c0II7):
        super().__init__(_f100A)
        self._lookback = _f100A.parameters.get('lookback_days', 20)
        self._top_n = _f100A.parameters.get('top_n', 10)
        self._bottom_n = _f100A.parameters.get('bottom_n', 10)

    @bridge(connects_to=['FeatureStore', 'SignalEngine', 'RiskEngine'], connection_types={'FeatureStore': 'reads', 'SignalEngine': 'feeds', 'RiskEngine': 'validates'})
    def _fl10E(self, _fII0f: Dict[str, Dict[str, float]], _fIIllO: Dict[str, Dict[str, float]]) -> List[Signal]:
        _fllOl2 = []
        momentum_scores = {}
        for symbol in self._f100A.universe:
            feat = _fIIllO.get(symbol, {})
            ret = feat.get(f'return_{self._lookback}d', 0)
            vol = feat.get('volatility_20', 0.02)
            if vol > 0:
                momentum_scores[symbol] = ret / vol
            else:
                momentum_scores[symbol] = ret
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        for symbol, score in sorted_symbols[:self._top_n]:
            if score > 0:
                _fllOl2.append(Signal(signal_id=f'mom_long_{symbol}', symbol=symbol, signal_type=SignalType.ENTRY, direction=OrderSide.BUY, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(score) * 0.1), rationale=f'Top momentum: {score:.2f}'))
        for symbol, score in sorted_symbols[-self._bottom_n:]:
            if score < 0:
                _fllOl2.append(Signal(signal_id=f'mom_short_{symbol}', symbol=symbol, signal_type=SignalType.ENTRY, direction=OrderSide.SELL, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(score) * 0.1), rationale=f'Bottom momentum: {score:.2f}'))
        return _fllOl2

    def _f0I1ll(self, _fllOl2: List[Signal], _fO10l3: Dict[str, Position], _fOl0l4: float) -> Dict[str, float]:
        _flOllB = {}
        long_signals = [s for s in _fllOl2 if s.direction == OrderSide.BUY]
        short_signals = [s for s in _fllOl2 if s.direction == OrderSide.SELL]
        n_long = len(long_signals)
        n_short = len(short_signals)
        if n_long > 0:
            long_weight = self._f100A.max_gross_exposure / 2 / n_long
            for signal in long_signals:
                _flOllB[signal.symbol] = long_weight * signal.confidence
        if n_short > 0:
            short_weight = self._f100A.max_gross_exposure / 2 / n_short
            for signal in short_signals:
                _flOllB[signal.symbol] = -short_weight * signal.confidence
        return _flOllB

class _cO1022(_c1l19):

    def __init__(self, _f100A: _c0II7):
        super().__init__(_f100A)
        self._zscore_entry = _f100A.parameters.get('zscore_entry', 2.0)
        self._zscore_exit = _f100A.parameters.get('zscore_exit', 0.5)
        self._lookback = _f100A.parameters.get('lookback_days', 20)

    @bridge(connects_to=['FeatureStore', 'SignalEngine', 'RiskEngine'], connection_types={'FeatureStore': 'reads', 'SignalEngine': 'feeds', 'RiskEngine': 'validates'})
    def _fl10E(self, _fII0f: Dict[str, Dict[str, float]], _fIIllO: Dict[str, Dict[str, float]]) -> List[Signal]:
        _fllOl2 = []
        for symbol in self._f100A.universe:
            data = _fII0f.get(symbol, {})
            feat = _fIIllO.get(symbol, {})
            price = data.get('close', 0)
            sma = feat.get(f'sma_{self._lookback}', price)
            std = feat.get(f'std_{self._lookback}', 0)
            if std <= 0:
                continue
            zscore = (price - sma) / std
            if zscore < -self._zscore_entry:
                _fllOl2.append(Signal(signal_id=f'mr_long_{symbol}', symbol=symbol, signal_type=SignalType.ENTRY, direction=OrderSide.BUY, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(zscore) * 0.1), target_price=sma, rationale=f'Oversold: z-score={zscore:.2f}'))
            elif zscore > self._zscore_entry:
                _fllOl2.append(Signal(signal_id=f'mr_short_{symbol}', symbol=symbol, signal_type=SignalType.ENTRY, direction=OrderSide.SELL, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(zscore) * 0.1), target_price=sma, rationale=f'Overbought: z-score={zscore:.2f}'))
            if symbol in self._positions:
                pos = self._positions[symbol]
                if pos.quantity > 0 and zscore > -self._zscore_exit:
                    _fllOl2.append(Signal(signal_id=f'mr_exit_long_{symbol}', symbol=symbol, signal_type=SignalType.EXIT, direction=OrderSide.SELL, source=SignalSource.QUANTITATIVE, confidence=0.7, rationale=f'Mean reversion exit: z-score={zscore:.2f}'))
                elif pos.quantity < 0 and zscore < self._zscore_exit:
                    _fllOl2.append(Signal(signal_id=f'mr_exit_short_{symbol}', symbol=symbol, signal_type=SignalType.EXIT, direction=OrderSide.BUY, source=SignalSource.QUANTITATIVE, confidence=0.7, rationale=f'Mean reversion exit: z-score={zscore:.2f}'))
        return _fllOl2

    def _f0I1ll(self, _fllOl2: List[Signal], _fO10l3: Dict[str, Position], _fOl0l4: float) -> Dict[str, float]:
        _flOllB = {}
        entry_signals = [s for s in _fllOl2 if s.signal_type == SignalType.ENTRY]
        exit_signals = [s for s in _fllOl2 if s.signal_type == SignalType.EXIT]
        for signal in exit_signals:
            _flOllB[signal.symbol] = 0
        for signal in entry_signals:
            weight = self._f100A.max_position_pct * signal.confidence
            if signal.direction == OrderSide.SELL:
                weight = -weight
            _flOllB[signal.symbol] = weight
        return _flOllB

class _cl1I23(_c1l19):

    def __init__(self, _f100A: _c0II7):
        super().__init__(_f100A)
        self._pairs = _f100A.parameters.get('pairs', [])
        self._zscore_entry = _f100A.parameters.get('zscore_entry', 2.0)
        self._zscore_exit = _f100A.parameters.get('zscore_exit', 0.5)
        self._spread_history: Dict[str, List[float]] = defaultdict(list)

    @bridge(connects_to=['FeatureStore', 'SignalEngine', 'CorrelationCutter'], connection_types={'FeatureStore': 'reads', 'SignalEngine': 'feeds', 'CorrelationCutter': 'uses'})
    def _fl10E(self, _fII0f: Dict[str, Dict[str, float]], _fIIllO: Dict[str, Dict[str, float]]) -> List[Signal]:
        _fllOl2 = []
        for sym1, sym2, hedge_ratio in self._pairs:
            price1 = _fII0f.get(sym1, {}).get('close', 0)
            price2 = _fII0f.get(sym2, {}).get('close', 0)
            if price1 <= 0 or price2 <= 0:
                continue
            spread = price1 - hedge_ratio * price2
            pair_key = f'{sym1}_{sym2}'
            self._spread_history[pair_key].append(spread)
            if len(self._spread_history[pair_key]) > 60:
                self._spread_history[pair_key] = self._spread_history[pair_key][-60:]
            if len(self._spread_history[pair_key]) < 20:
                continue
            history = self._spread_history[pair_key]
            mean = sum(history) / len(history)
            std = math.sqrt(sum(((x - mean) ** 2 for x in history)) / len(history))
            if std <= 0:
                continue
            zscore = (spread - mean) / std
            if zscore > self._zscore_entry:
                _fllOl2.append(Signal(signal_id=f'pairs_short_{sym1}', symbol=sym1, signal_type=SignalType.ENTRY, direction=OrderSide.SELL, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(zscore) * 0.1), rationale=f'Pairs: short spread z={zscore:.2f}'))
                _fllOl2.append(Signal(signal_id=f'pairs_long_{sym2}', symbol=sym2, signal_type=SignalType.ENTRY, direction=OrderSide.BUY, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(zscore) * 0.1), rationale=f'Pairs: short spread z={zscore:.2f}'))
            elif zscore < -self._zscore_entry:
                _fllOl2.append(Signal(signal_id=f'pairs_long_{sym1}', symbol=sym1, signal_type=SignalType.ENTRY, direction=OrderSide.BUY, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(zscore) * 0.1), rationale=f'Pairs: long spread z={zscore:.2f}'))
                _fllOl2.append(Signal(signal_id=f'pairs_short_{sym2}', symbol=sym2, signal_type=SignalType.ENTRY, direction=OrderSide.SELL, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(zscore) * 0.1), rationale=f'Pairs: long spread z={zscore:.2f}'))
        return _fllOl2

    def _f0I1ll(self, _fllOl2: List[Signal], _fO10l3: Dict[str, Position], _fOl0l4: float) -> Dict[str, float]:
        _flOllB = {}
        for signal in _fllOl2:
            weight = self._f100A.max_position_pct / 2 * signal.confidence
            if signal.direction == OrderSide.SELL:
                weight = -weight
            if signal.symbol in _flOllB:
                _flOllB[signal.symbol] += weight
            else:
                _flOllB[signal.symbol] = weight
        return _flOllB

class _clO024(_c1l19):

    def __init__(self, _f100A: _c0II7):
        super().__init__(_f100A)
        self._factors: List[_c1lI4] = []
        self._factor_values: Dict[str, Dict[str, _c1O05]] = defaultdict(dict)

    def _fOll25(self, _fl0l26: _c1lI4):
        self._factors.append(_fl0l26)

    @bridge(connects_to=['FeatureStore', 'SignalEngine', 'RiskEngine', 'ModelRegistry'], connection_types={'FeatureStore': 'reads', 'SignalEngine': 'feeds', 'RiskEngine': 'validates', 'ModelRegistry': 'uses'})
    def _fl10E(self, _fII0f: Dict[str, Dict[str, float]], _fIIllO: Dict[str, Dict[str, float]]) -> List[Signal]:
        _fllOl2 = []
        composite_scores = {}
        for symbol in self._f100A.universe:
            feat = _fIIllO.get(symbol, {})
            total_score = 0.0
            total_weight = 0.0
            for _fl0l26 in self._factors:
                if _fl0l26.compute_fn:
                    raw_value = _fl0l26.compute_fn(feat)
                else:
                    raw_value = feat.get(_fl0l26._fI1lB, 0)
                normalized = raw_value
                self._factor_values[symbol][_fl0l26._fI1lB] = _c1O05(symbol=symbol, factor_name=_fl0l26._fI1lB, raw_value=raw_value, normalized_value=normalized, weight=_fl0l26.weight, timestamp=datetime.now())
                total_score += normalized * _fl0l26.weight
                total_weight += _fl0l26.weight
            if total_weight > 0:
                composite_scores[symbol] = total_score / total_weight
        sorted_symbols = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_symbols)
        quintile_size = n // 5 if n >= 5 else 1
        for symbol, score in sorted_symbols[:quintile_size]:
            _fllOl2.append(Signal(signal_id=f'factor_long_{symbol}', symbol=symbol, signal_type=SignalType.ENTRY, direction=OrderSide.BUY, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(score) * 0.2), rationale=f'Top factor quintile: {score:.3f}'))
        for symbol, score in sorted_symbols[-quintile_size:]:
            _fllOl2.append(Signal(signal_id=f'factor_short_{symbol}', symbol=symbol, signal_type=SignalType.ENTRY, direction=OrderSide.SELL, source=SignalSource.QUANTITATIVE, confidence=min(0.9, 0.5 + abs(score) * 0.2), rationale=f'Bottom factor quintile: {score:.3f}'))
        return _fllOl2

    def _f0I1ll(self, _fllOl2: List[Signal], _fO10l3: Dict[str, Position], _fOl0l4: float) -> Dict[str, float]:
        _flOllB = {}
        long_signals = [s for s in _fllOl2 if s.direction == OrderSide.BUY]
        short_signals = [s for s in _fllOl2 if s.direction == OrderSide.SELL]
        total_long_conf = sum((s.confidence for s in long_signals))
        total_short_conf = sum((s.confidence for s in short_signals))
        max_side = self._f100A.max_gross_exposure / 2
        for signal in long_signals:
            if total_long_conf > 0:
                weight = signal.confidence / total_long_conf * max_side
                _flOllB[signal.symbol] = weight
        for signal in short_signals:
            if total_short_conf > 0:
                weight = signal.confidence / total_short_conf * max_side
                _flOllB[signal.symbol] = -weight
        return _flOllB

class _c0II27:

    def __init__(self):
        self._strategies: Dict[str, _c1l19] = {}
        self._allocations: Dict[str, float] = {}
        self._combined_positions: Dict[str, float] = {}

    @bridge(connects_to=['JonesEngine', 'RiskEngine', 'OrderManager'], connection_types={'JonesEngine': 'integrates', 'RiskEngine': 'validates', 'OrderManager': 'feeds'})
    def _fII128(self, _flOO29: _c1l19, _f1Il2A: float=1.0):
        self._strategies[_flOO29._fI1lB] = _flOO29
        self._allocations[_flOO29._fI1lB] = _f1Il2A
        total = sum(self._allocations.values())
        if total > 0:
            self._allocations = {k: v / total for k, v in self._allocations.items()}

    def _fIl12B(self, _fI1lB: str):
        if _fI1lB in self._strategies:
            del self._strategies[_fI1lB]
            del self._allocations[_fI1lB]

    def _fl1Il5(self, _fII0f: Dict[str, Dict[str, float]], _fIIllO: Dict[str, Dict[str, float]], _fOl0l4: float) -> List[Order]:
        all_orders = []
        for _fI1lB, _flOO29 in self._strategies.items():
            _f1Il2A = self._allocations.get(_fI1lB, 0)
            allocated_value = _fOl0l4 * _f1Il2A
            orders = _flOO29._fl1Il5(_fII0f, _fIIllO, allocated_value)
            for _f101l7 in orders:
                _f101l7.quantity = int(_f101l7.quantity * _f1Il2A)
                if _f101l7.quantity > 0:
                    all_orders.append(_f101l7)
        return all_orders

    def _f1102c(self) -> Dict[str, float]:
        combined = defaultdict(float)
        for _flOO29 in self._strategies.values():
            for symbol, position in _flOO29._positions.items():
                combined[symbol] += position.quantity
        return dict(combined)

    def _f1I12d(self):
        for _flOO29 in self._strategies.values():
            _flOO29._f1Illd()

    def _fO002E(self):
        for _flOO29 in self._strategies.values():
            _flOO29._fO11lE()

    @property
    def _fIOI2f(self) -> Dict[str, str]:
        return {_fI1lB: _flOO29._fI10c.value for _fI1lB, _flOO29 in self._strategies.items()}

def _f1003O(_fI1lB: str, _fI113l: List[str], _fOO032: int=20, _fl0133: int=10) -> _cI0I2l:
    _f100A = _c0II7(name=_fI1lB, strategy_type=_c100l.MOMENTUM, universe=_fI113l, parameters={'lookback_days': _fOO032, 'top_n': _fl0133, 'bottom_n': _fl0133})
    return _cI0I2l(_f100A)

def _f0l034(_fI1lB: str, _fI113l: List[str], _f00135: float=2.0, _fOlO36: float=0.5) -> _cO1022:
    _f100A = _c0II7(name=_fI1lB, strategy_type=_c100l.MEAN_REVERSION, universe=_fI113l, parameters={'zscore_entry': _f00135, 'zscore_exit': _fOlO36})
    return _cO1022(_f100A)

def _fIlI37(_fI1lB: str, _flIl38: List[Tuple[str, str, float]]) -> _cl1I23:
    symbols = list(set((s for p in _flIl38 for s in p[:2])))
    _f100A = _c0II7(name=_fI1lB, strategy_type=_c100l.PAIRS, universe=symbols, parameters={'pairs': _flIl38})
    return _cl1I23(_f100A)

def _fOlO39(_fI1lB: str, _fI113l: List[str], _f11O3A: List[_c1lI4]) -> _clO024:
    _f100A = _c0II7(name=_fI1lB, strategy_type=_c100l.FACTOR, universe=_fI113l)
    _flOO29 = _clO024(_f100A)
    for _fl0l26 in _f11O3A:
        _flOO29._fOll25(_fl0l26)
    return _flOO29

# Public API aliases for obfuscated classes
StrategyType = _c100l
StrategyState = _c0OO2
AlphaDecay = _c0103
AlphaFactor = _c1lI4
AlphaSignal = _c1O05
StrategyConfig = _c0II7
StrategyMetrics = _cOIl8
Strategy = _c1l19
MomentumStrategy = _cI0I2l
MeanReversionStrategy = _cO1022
PairsStrategy = _cl1I23
FactorStrategy = _clO024
StrategyManager = _c0II27
create_momentum_strategy = _f1003O
create_mean_reversion_strategy = _f0l034
create_pairs_strategy = _fIlI37
create_factor_strategy = _fOlO39
