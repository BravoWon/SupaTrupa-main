from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import math
from collections import defaultdict
import heapq
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.trading.execution.order_manager import Order, OrderSide, OrderType

class _cO1O9f(Enum):
    ENTRY = 'entry'
    EXIT = 'exit'
    SCALE_IN = 'scale_in'
    SCALE_OUT = 'scale_out'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    REBALANCE = 'rebalance'
    HEDGE = 'hedge'

class _c01IAO(Enum):
    TECHNICAL = 'technical'
    FUNDAMENTAL = 'fundamental'
    SENTIMENT = 'sentiment'
    NEWS = 'news'
    QUANTITATIVE = 'quantitative'
    REGIME = 'regime'
    CORRELATION = 'correlation'
    RISK = 'risk'
    ML_MODEL = 'ml_model'
    EXTERNAL = 'external'
    COMPOSITE = 'composite'

class _cO01Al(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

class _c1O1A2(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class _c0IOA3:
    signal_id: str
    symbol: str
    signal_type: _cO1O9f
    direction: OrderSide
    source: _c01IAO
    strength: _cO01Al = _cO01Al.MODERATE
    priority: _c1O1A2 = _c1O1A2.NORMAL
    confidence: float = 0.5
    conviction: float = 0.5
    generated_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    execution_window: Optional[timedelta] = None
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    suggested_size: Optional[int] = None
    size_pct: Optional[float] = None
    risk_pct: Optional[float] = None
    rationale: str = ''
    supporting_signals: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    is_executed: bool = False
    executed_at: Optional[datetime] = None
    execution_price: Optional[float] = None

    @property
    def _fOO1A4(self) -> bool:
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until

    @property
    def _flOIA5(self) -> Optional[timedelta]:
        if self.valid_until is None:
            return None
        return self.valid_until - datetime.now()

    @property
    def _f0lOA6(self) -> timedelta:
        return datetime.now() - self.generated_at

    @property
    def _f11lA7(self) -> float:
        strength_score = self.strength.value / 5
        confidence_score = self.confidence
        priority_score = self.priority.value / 5
        if self.valid_until:
            remaining = (self.valid_until - datetime.now()).total_seconds()
            total = (self.valid_until - self.generated_at).total_seconds()
            time_factor = max(0, remaining / total) if total > 0 else 0
        else:
            age_seconds = self._f0lOA6.total_seconds()
            time_factor = max(0, 1 - age_seconds / 86400)
        return strength_score * 0.3 + confidence_score * 0.35 + priority_score * 0.15 + time_factor * 0.2

@dataclass
class _cO00A8:
    group_id: str
    signals: List[_c0IOA3]
    symbol: str
    net_direction: Optional[OrderSide] = None
    consensus_score: float = 0.0
    agreement_ratio: float = 0.0

    def __post_init__(self):
        self._calculate_consensus()

    def _fll1A9(self):
        if not self.signals:
            return
        buy_score = sum((s._f11lA7 for s in self.signals if s.direction == OrderSide.BUY))
        sell_score = sum((s._f11lA7 for s in self.signals if s.direction == OrderSide.SELL))
        self.consensus_score = abs(buy_score - sell_score)
        if buy_score > sell_score:
            self.net_direction = OrderSide.BUY
        elif sell_score > buy_score:
            self.net_direction = OrderSide.SELL
        total = buy_score + sell_score
        if total > 0:
            self.agreement_ratio = max(buy_score, sell_score) / total

class _cO0OAA(ABC):

    def __init__(self, _f1l1AB: str, _f0O0Ac: _c01IAO):
        self._f1l1AB = _f1l1AB
        self._f0O0Ac = _f0O0Ac
        self._active = True
        self._weight = 1.0
        self._signal_count = 0

    @abstractmethod
    async def _f0I0Ad(self, _flOIAE: str, _f00OAf: Dict[str, Any], _fIIIBO: Dict[str, Any]) -> List[_c0IOA3]:
        pass

    @property
    def _flllBl(self) -> float:
        return self._weight

    @_flllBl.setter
    def _flllBl(self, _flOlB2: float):
        self._weight = max(0.0, min(1.0, _flOlB2))

class _c0l1B3(_cO0OAA):

    def __init__(self):
        super().__init__('Technical', _c01IAO.TECHNICAL)

    @bridge(connects_to=['FeatureStore', 'MetricEngine'], connection_types={'FeatureStore': 'reads', 'MetricEngine': 'uses'})
    async def _f0I0Ad(self, _flOIAE: str, _f00OAf: Dict[str, Any], _fIIIBO: Dict[str, Any]) -> List[_c0IOA3]:
        signals = []
        rsi = _f00OAf.get('rsi_14', 50)
        if rsi < 30:
            signals.append(self._create_signal(symbol=_flOIAE, signal_type=_cO1O9f.ENTRY, direction=OrderSide.BUY, strength=_cO01Al.STRONG if rsi < 20 else _cO01Al.MODERATE, confidence=0.7 - rsi / 100, rationale=f'RSI oversold at {rsi:.1f}'))
        elif rsi > 70:
            signals.append(self._create_signal(symbol=_flOIAE, signal_type=_cO1O9f.EXIT, direction=OrderSide.SELL, strength=_cO01Al.STRONG if rsi > 80 else _cO01Al.MODERATE, confidence=rsi / 100 - 0.3, rationale=f'RSI overbought at {rsi:.1f}'))
        macd = _f00OAf.get('macd', 0)
        macd_signal = _f00OAf.get('macd_signal', 0)
        macd_prev = _f00OAf.get('macd_prev', macd)
        if macd > macd_signal and macd_prev <= macd_signal:
            signals.append(self._create_signal(symbol=_flOIAE, signal_type=_cO1O9f.ENTRY, direction=OrderSide.BUY, strength=_cO01Al.MODERATE, confidence=0.6, rationale='MACD bullish crossover'))
        elif macd < macd_signal and macd_prev >= macd_signal:
            signals.append(self._create_signal(symbol=_flOIAE, signal_type=_cO1O9f.EXIT, direction=OrderSide.SELL, strength=_cO01Al.MODERATE, confidence=0.6, rationale='MACD bearish crossover'))
        price = _f00OAf.get('price_close', 0)
        sma_20 = _f00OAf.get('sma_20', price)
        sma_50 = _f00OAf.get('sma_50', price)
        if sma_20 > sma_50 and _f00OAf.get('sma_20_prev', sma_20) <= _f00OAf.get('sma_50_prev', sma_50):
            signals.append(self._create_signal(symbol=_flOIAE, signal_type=_cO1O9f.ENTRY, direction=OrderSide.BUY, strength=_cO01Al.MODERATE, confidence=0.55, rationale='Golden cross (20/50 MA)'))
        elif sma_20 < sma_50 and _f00OAf.get('sma_20_prev', sma_20) >= _f00OAf.get('sma_50_prev', sma_50):
            signals.append(self._create_signal(symbol=_flOIAE, signal_type=_cO1O9f.EXIT, direction=OrderSide.SELL, strength=_cO01Al.MODERATE, confidence=0.55, rationale='Death cross (20/50 MA)'))
        return signals

    def _f0l1B4(self, _flOIAE: str, _fl01B5: _cO1O9f, _fO1lB6: OrderSide, _f1IIB7: _cO01Al, _fl00B8: float, _fl00B9: str) -> _c0IOA3:
        self._signal_count += 1
        return _c0IOA3(signal_id=f'{self._f1l1AB}_{_flOIAE}_{self._signal_count}', symbol=_flOIAE, signal_type=_fl01B5, direction=_fO1lB6, source=self._f0O0Ac, strength=_f1IIB7, confidence=_fl00B8, rationale=_fl00B9, valid_until=datetime.now() + timedelta(hours=4))

class _clO1BA(_cO0OAA):

    def __init__(self):
        super().__init__('Sentiment', _c01IAO.SENTIMENT)

    @bridge(connects_to=['NewsArticleProcessor', 'EarningsCallProcessor', 'LinguisticArbitrage'], connection_types={'NewsArticleProcessor': 'reads', 'EarningsCallProcessor': 'reads', 'LinguisticArbitrage': 'uses'})
    async def _f0I0Ad(self, _flOIAE: str, _f00OAf: Dict[str, Any], _fIIIBO: Dict[str, Any]) -> List[_c0IOA3]:
        signals = []
        news_sentiment = _f00OAf.get('news_sentiment', 0)
        news_count = _f00OAf.get('news_count', 0)
        if news_count > 0:
            if news_sentiment > 0.5:
                signals.append(self._f0l1B4(symbol=_flOIAE, signal_type=_cO1O9f.ENTRY, direction=OrderSide.BUY, strength=_cO01Al.STRONG if news_sentiment > 0.7 else _cO01Al.MODERATE, confidence=min(0.8, 0.4 + news_sentiment * 0.5), rationale=f'Strong positive news sentiment ({news_sentiment:.2f})'))
            elif news_sentiment < -0.5:
                signals.append(self._f0l1B4(symbol=_flOIAE, signal_type=_cO1O9f.EXIT, direction=OrderSide.SELL, strength=_cO01Al.STRONG if news_sentiment < -0.7 else _cO01Al.MODERATE, confidence=min(0.8, 0.4 + abs(news_sentiment) * 0.5), rationale=f'Strong negative news sentiment ({news_sentiment:.2f})'))
        social_sentiment = _f00OAf.get('social_sentiment', 0)
        social_volume = _f00OAf.get('social_volume', 0)
        if social_volume > _f00OAf.get('avg_social_volume', 0) * 2:
            if social_sentiment > 0.3:
                signals.append(self._f0l1B4(symbol=_flOIAE, signal_type=_cO1O9f.ENTRY, direction=OrderSide.BUY, strength=_cO01Al.WEAK, confidence=0.4, rationale='High positive social media activity'))
            elif social_sentiment < -0.3:
                signals.append(self._f0l1B4(symbol=_flOIAE, signal_type=_cO1O9f.EXIT, direction=OrderSide.SELL, strength=_cO01Al.WEAK, confidence=0.4, rationale='High negative social media activity'))
        return signals

    def _f0l1B4(self, _flOIAE: str, _fl01B5: _cO1O9f, _fO1lB6: OrderSide, _f1IIB7: _cO01Al, _fl00B8: float, _fl00B9: str) -> _c0IOA3:
        self._signal_count += 1
        return _c0IOA3(signal_id=f'{self._f1l1AB}_{_flOIAE}_{self._signal_count}', symbol=_flOIAE, signal_type=_fl01B5, direction=_fO1lB6, source=self._f0O0Ac, strength=_f1IIB7, confidence=_fl00B8, rationale=_fl00B9, valid_until=datetime.now() + timedelta(hours=2))

class _cO01BB(_cO0OAA):

    def __init__(self):
        super().__init__('Regime', _c01IAO.REGIME)

    @bridge(connects_to=['RegimeClassifier', 'ActivityState', 'CorrelationCutter'], connection_types={'RegimeClassifier': 'reads', 'ActivityState': 'reads', 'CorrelationCutter': 'uses'})
    async def _f0I0Ad(self, _flOIAE: str, _f00OAf: Dict[str, Any], _fIIIBO: Dict[str, Any]) -> List[_c0IOA3]:
        signals = []
        current_regime = _f00OAf.get('regime', 'neutral')
        previous_regime = _f00OAf.get('previous_regime', current_regime)
        regime_confidence = _f00OAf.get('regime_confidence', 0.5)
        if current_regime != previous_regime:
            if current_regime == 'bull' and previous_regime in ['bear', 'neutral']:
                signals.append(self._f0l1B4(symbol=_flOIAE, signal_type=_cO1O9f.ENTRY, direction=OrderSide.BUY, strength=_cO01Al.STRONG, confidence=regime_confidence, rationale=f'Regime transition to BULL from {previous_regime}'))
            elif current_regime == 'bear' and previous_regime in ['bull', 'neutral']:
                signals.append(self._f0l1B4(symbol=_flOIAE, signal_type=_cO1O9f.EXIT, direction=OrderSide.SELL, strength=_cO01Al.STRONG, confidence=regime_confidence, rationale=f'Regime transition to BEAR from {previous_regime}'))
        volatility_regime = _f00OAf.get('volatility_regime', 'normal')
        if volatility_regime == 'high':
            signals.append(self._f0l1B4(symbol=_flOIAE, signal_type=_cO1O9f.SCALE_OUT, direction=OrderSide.SELL, strength=_cO01Al.MODERATE, confidence=0.6, rationale='High volatility regime - reduce exposure', size_pct=0.3))
        return signals

    def _f0l1B4(self, _flOIAE: str, _fl01B5: _cO1O9f, _fO1lB6: OrderSide, _f1IIB7: _cO01Al, _fl00B8: float, _fl00B9: str, _fIl0Bc: Optional[float]=None) -> _c0IOA3:
        self._signal_count += 1
        return _c0IOA3(signal_id=f'{self._f1l1AB}_{_flOIAE}_{self._signal_count}', symbol=_flOIAE, signal_type=_fl01B5, direction=_fO1lB6, source=self._f0O0Ac, strength=_f1IIB7, confidence=_fl00B8, rationale=_fl00B9, size_pct=_fIl0Bc, valid_until=datetime.now() + timedelta(hours=12))

class _cO0lBd:

    def __init__(self):
        self._source_weights: Dict[_c01IAO, float] = {_c01IAO.TECHNICAL: 0.25, _c01IAO.FUNDAMENTAL: 0.3, _c01IAO.SENTIMENT: 0.15, _c01IAO.NEWS: 0.1, _c01IAO.QUANTITATIVE: 0.25, _c01IAO.REGIME: 0.2, _c01IAO.CORRELATION: 0.15, _c01IAO.RISK: 0.25, _c01IAO.ML_MODEL: 0.25, _c01IAO.EXTERNAL: 0.1}
        self._min_consensus = 0.3

    def _fl1OBE(self, _f0O0Ac: _c01IAO, _flllBl: float):
        self._source_weights[_f0O0Ac] = max(0.0, min(1.0, _flllBl))

    @bridge(connects_to=['TechnicalSignalGenerator', 'SentimentSignalGenerator', 'RegimeSignalGenerator'], connection_types={'TechnicalSignalGenerator': 'aggregates', 'SentimentSignalGenerator': 'aggregates', 'RegimeSignalGenerator': 'aggregates'})
    def _fO01Bf(self, _f010cO: List[_c0IOA3], _flOIAE: str) -> _cO00A8:
        active_signals = [s for s in _f010cO if s._flOIAE == _flOIAE and s.is_active and (not s._fOO1A4)]
        if not active_signals:
            return _cO00A8(group_id=f'group_{_flOIAE}_{datetime.now().timestamp()}', signals=[], symbol=_flOIAE)
        weighted_signals = []
        for signal in active_signals:
            _flllBl = self._source_weights.get(signal._f0O0Ac, 0.5)
            signal.metadata['applied_weight'] = _flllBl
            weighted_signals.append(signal)
        return _cO00A8(group_id=f'group_{_flOIAE}_{datetime.now().timestamp()}', signals=weighted_signals, symbol=_flOIAE)

    def _f01Icl(self, _f1O0c2: _cO00A8) -> bool:
        if not _f1O0c2._f010cO:
            return False
        return _f1O0c2.consensus_score > self._min_consensus and _f1O0c2.agreement_ratio > 0.6

class _cOOOc3:

    def __init__(self, _f11Ic4: float=0.1, _fI00c5: float=0.02, _fl10c6: bool=True):
        self._max_position_pct = _f11Ic4
        self._max_risk_pct = _fI00c5
        self._volatility_scaling = _fl10c6

    @bridge(connects_to=['RiskEngine', 'PortfolioRisk'], connection_types={'RiskEngine': 'uses', 'PortfolioRisk': 'reads'})
    def _fl01c7(self, _fIOIc8: _c0IOA3, _fO0Oc9: float, _fl11cA: float, _f00IcB: float=0.02, _flI1cc: int=0) -> int:
        max_value = _fO0Oc9 * self._max_position_pct
        base_size = int(max_value / _fl11cA)
        if _fIOIc8.stop_price and _fl11cA > 0:
            risk_per_share = abs(_fl11cA - _fIOIc8.stop_price)
            max_risk_value = _fO0Oc9 * self._max_risk_pct
            risk_based_size = int(max_risk_value / risk_per_share) if risk_per_share > 0 else base_size
            base_size = min(base_size, risk_based_size)
        if self._volatility_scaling:
            target_vol = 0.02
            vol_factor = target_vol / _f00IcB if _f00IcB > 0 else 1
            vol_factor = max(0.5, min(2.0, vol_factor))
            base_size = int(base_size * vol_factor)
        strength_factor = _fIOIc8._f1IIB7._flOlB2 / 5
        sized = int(base_size * (0.5 + strength_factor * 0.5))
        sized = int(sized * (0.5 + _fIOIc8._fl00B8 * 0.5))
        if _fIOIc8._fIl0Bc:
            suggested = int(_fO0Oc9 * _fIOIc8._fIl0Bc / _fl11cA)
            sized = min(sized, suggested)
        if _fIOIc8._fl01B5 == _cO1O9f.SCALE_OUT:
            sized = min(sized, abs(_flI1cc))
        elif _fIOIc8._fl01B5 == _cO1O9f.SCALE_IN:
            pass
        return max(0, sized)

class _cIlIcd:

    def __init__(self, _f10lcE: _cOOOc3, _fOIlcf: bool=True, _fl0OdO: float=0.001):
        self._sizer = _f10lcE
        self._use_limit_orders = _fOIlcf
        self._limit_offset_pct = _fl0OdO
        self._order_count = 0

    @bridge(connects_to=['PositionSizer', 'OrderManager'], connection_types={'PositionSizer': 'uses', 'OrderManager': 'feeds'})
    def _fI0Odl(self, _fIOIc8: _c0IOA3, _fl11cA: float, _fO0Oc9: float, _flI1cc: int=0, _f00IcB: float=0.02) -> List[Order]:
        orders = []
        size = self._sizer._fl01c7(signal=_fIOIc8, portfolio_value=_fO0Oc9, current_price=_fl11cA, volatility=_f00IcB, current_position=_flI1cc)
        if size <= 0:
            return orders
        if self._use_limit_orders and _fIOIc8._fl01B5 in [_cO1O9f.ENTRY, _cO1O9f.SCALE_IN]:
            order_type = OrderType.LIMIT
            if _fIOIc8._fO1lB6 == OrderSide.BUY:
                limit_price = _fl11cA * (1 - self._limit_offset_pct)
            else:
                limit_price = _fl11cA * (1 + self._limit_offset_pct)
        else:
            order_type = OrderType.MARKET
            limit_price = None
        if _fIOIc8.limit_price:
            order_type = OrderType.LIMIT
            limit_price = _fIOIc8.limit_price
        self._order_count += 1
        main_order = Order(order_id=f'order_{_fIOIc8.signal_id}_{self._order_count}', symbol=_fIOIc8._flOIAE, side=_fIOIc8._fO1lB6, quantity=size, order_type=order_type, limit_price=limit_price, stop_price=None)
        orders.append(main_order)
        if _fIOIc8.stop_price and _fIOIc8._fl01B5 == _cO1O9f.ENTRY:
            self._order_count += 1
            stop_side = OrderSide.SELL if _fIOIc8._fO1lB6 == OrderSide.BUY else OrderSide.BUY
            stop_order = Order(order_id=f'stop_{_fIOIc8.signal_id}_{self._order_count}', symbol=_fIOIc8._flOIAE, side=stop_side, quantity=size, order_type=OrderType.STOP, stop_price=_fIOIc8.stop_price)
            orders.append(stop_order)
        if _fIOIc8.target_price and _fIOIc8._fl01B5 == _cO1O9f.ENTRY:
            self._order_count += 1
            tp_side = OrderSide.SELL if _fIOIc8._fO1lB6 == OrderSide.BUY else OrderSide.BUY
            tp_order = Order(order_id=f'tp_{_fIOIc8.signal_id}_{self._order_count}', symbol=_fIOIc8._flOIAE, side=tp_side, quantity=size, order_type=OrderType.LIMIT, limit_price=_fIOIc8.target_price)
            orders.append(tp_order)
        return orders

class _clIId2:

    def __init__(self):
        self._generators: List[_cO0OAA] = []
        self._aggregator = _cO0lBd()
        self._sizer = _cOOOc3()
        self._order_generator = _cIlIcd(self._sizer)
        self._pending_signals: Dict[str, List[_c0IOA3]] = defaultdict(list)
        self._signal_history: List[_c0IOA3] = []
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['OrderManager', 'RiskEngine', 'FeatureStore', 'JonesEngine'], connection_types={'OrderManager': 'feeds', 'RiskEngine': 'validates', 'FeatureStore': 'reads', 'JonesEngine': 'integrates'})
    def _fI1Od3(self, _fIlId4: _cO0OAA):
        self._generators.append(_fIlId4)

    async def _fOOld5(self, _flOIAE: str, _f00OAf: Dict[str, Any], _fIIIBO: Dict[str, Any]) -> List[_c0IOA3]:
        all_signals = []
        for _fIlId4 in self._generators:
            try:
                _f010cO = await _fIlId4._f0I0Ad(_flOIAE, _f00OAf, _fIIIBO)
                all_signals.extend(_f010cO)
            except Exception as e:
                pass
        self._pending_signals[_flOIAE].extend(all_signals)
        self._signal_history.extend(all_signals)
        return all_signals

    def _fO1Od6(self, _flOIAE: str) -> _cO00A8:
        pending = self._pending_signals.get(_flOIAE, [])
        return self._aggregator._fO01Bf(pending, _flOIAE)

    def _fI0Odl(self, _flOIAE: str, _fl11cA: float, _fO0Oc9: float, _flI1cc: int=0, _f00IcB: float=0.02) -> List[Order]:
        _f1O0c2 = self._fO1Od6(_flOIAE)
        if not self._aggregator._f01Icl(_f1O0c2):
            return []
        orders = []
        if _f1O0c2.net_direction:
            composite = _c0IOA3(signal_id=f'composite_{_f1O0c2.group_id}', symbol=_flOIAE, signal_type=_cO1O9f.ENTRY, direction=_f1O0c2.net_direction, source=_c01IAO.COMPOSITE, strength=self._determine_strength(_f1O0c2), confidence=_f1O0c2.consensus_score)
            orders = self._order_generator._fI0Odl(signal=composite, current_price=_fl11cA, portfolio_value=_fO0Oc9, current_position=_flI1cc, volatility=_f00IcB)
        return orders

    def _flO1d7(self, _f1O0c2: _cO00A8) -> _cO01Al:
        if _f1O0c2.consensus_score > 0.8:
            return _cO01Al.VERY_STRONG
        elif _f1O0c2.consensus_score > 0.6:
            return _cO01Al.STRONG
        elif _f1O0c2.consensus_score > 0.4:
            return _cO01Al.MODERATE
        elif _f1O0c2.consensus_score > 0.2:
            return _cO01Al.WEAK
        return _cO01Al.VERY_WEAK

    def _f10Od8(self, _flOIAE: str):
        self._pending_signals[_flOIAE] = []

    def _flO1d9(self, _f010dA: str, _fOlldB: float):
        for _f010cO in self._pending_signals.values():
            for _fIOIc8 in _f010cO:
                if _fIOIc8._f010dA == _f010dA:
                    _fIOIc8.is_executed = True
                    _fIOIc8.executed_at = datetime.now()
                    _fIOIc8.execution_price = _fOlldB
                    _fIOIc8.is_active = False
                    return

    @property
    def _fO0Idc(self) -> Dict[str, Any]:
        total = len(self._signal_history)
        executed = sum((1 for s in self._signal_history if s.is_executed))
        expired = sum((1 for s in self._signal_history if s._fOO1A4 and (not s.is_executed)))
        by_source = defaultdict(int)
        by_type = defaultdict(int)
        for s in self._signal_history:
            by_source[s._f0O0Ac._flOlB2] += 1
            by_type[s._fl01B5._flOlB2] += 1
        return {'total_signals': total, 'executed': executed, 'expired': expired, 'active': total - executed - expired, 'execution_rate': executed / total if total > 0 else 0, 'by_source': dict(by_source), 'by_type': dict(by_type)}

def _fOl1dd() -> _clIId2:
    engine = _clIId2()
    engine._fI1Od3(_c0l1B3())
    engine._fI1Od3(_clO1BA())
    engine._fI1Od3(_cO01BB())
    return engine

def _f0IIdE(_f11Ic4: float=0.1, _fI00c5: float=0.02) -> _cOOOc3:
    return _cOOOc3(_f11Ic4, _fI00c5)

# Public API aliases for obfuscated classes
SignalType = _cO1O9f
SignalSource = _c01IAO
SignalStrength = _cO01Al
SignalPriority = _c1O1A2
Signal = _c0IOA3
SignalGroup = _cO00A8
SignalGenerator = _cO0OAA
TechnicalSignalGenerator = _c0l1B3
SentimentSignalGenerator = _clO1BA
RegimeSignalGenerator = _cO01BB
SignalAggregator = _cO0lBd
PositionSizer = _cOOOc3
OrderGenerator = _cIlIcd
SignalEngine = _clIId2
create_signal_engine = _fOl1dd
create_position_sizer = _f0IIdE