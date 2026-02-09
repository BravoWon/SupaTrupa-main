from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Iterator, TypeVar
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import math
from collections import defaultdict
import heapq
import copy
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.engine.core import Timeframe
from jones_framework.trading.execution.order_manager import Order, OrderSide, OrderType, OrderStatus, Fill, Position
from jones_framework.features.store import FeatureStore, FeatureVector

class _c1O07Bl(Enum):
    BAR = 'bar'
    TICK = 'tick'
    FILL = 'fill'
    ORDER = 'order'
    SIGNAL = 'signal'
    REBALANCE = 'rebalance'
    DIVIDEND = 'dividend'
    SPLIT = 'split'
    CUSTOM = 'custom'

class _cI117B2(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    VWAP = 'vwap'
    TWAP = 'twap'
    CLOSE = 'close'
    NEXT_OPEN = 'next_open'

@dataclass
class _cll17B3:
    timestamp: datetime
    event_type: _c1O07Bl
    symbol: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def __lt__(self, _fIlI7B4):
        if self.timestamp != _fIlI7B4.timestamp:
            return self.timestamp < _fIlI7B4.timestamp
        return self.priority < _fIlI7B4.priority

@dataclass
class _cIlI7B5:
    symbol: str
    timestamp: datetime
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None

    @property
    def _f0IO7B6(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low

@dataclass
class _c1I07B7:
    commission_per_share: float = 0.0
    commission_per_trade: float = 0.0
    commission_pct: float = 0.0
    min_commission: float = 0.0
    slippage_pct: float = 0.0
    slippage_fixed: float = 0.0
    market_impact_pct: float = 0.0

    def _fIOO7B8(self, _fOIl7B9: float, _fl1O7BA: int, _fl0l7BB: OrderSide) -> Tuple[float, float]:
        commission = max(self.min_commission, self.commission_per_trade + self.commission_per_share * abs(_fl1O7BA) + self.commission_pct * _fOIl7B9 * abs(_fl1O7BA))
        slippage = self.slippage_fixed + self.slippage_pct * _fOIl7B9
        impact = self.market_impact_pct * _fOIl7B9 * math.log1p(abs(_fl1O7BA) / 1000)
        if _fl0l7BB == OrderSide.BUY:
            adjusted_price = _fOIl7B9 + slippage + impact
        else:
            adjusted_price = _fOIl7B9 - slippage - impact
        return (commission, adjusted_price)

@dataclass
class _cIIO7Bc:
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    benchmark: Optional[str] = None
    execution_model: _cI117B2 = _cI117B2.CLOSE
    transaction_costs: _c1I07B7 = field(default_factory=_c1I07B7)
    fill_delay_bars: int = 0
    max_position_pct: float = 0.2
    max_leverage: float = 1.0
    margin_requirement: float = 0.25
    timeframe: Timeframe = Timeframe.D1
    use_adjusted_prices: bool = True
    random_seed: Optional[int] = None
    num_simulations: int = 1

@dataclass
class _cllO7Bd:
    trade_id: str
    symbol: str
    _fl0l7BB: OrderSide
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    _fl1O7BA: int
    pnl: float
    pnl_pct: float
    commission: float
    duration: timedelta
    mae: float = 0.0
    mfe: float = 0.0

@dataclass
class _cO0I7BE:
    timestamps: List[datetime] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    cash: List[float] = field(default_factory=list)
    positions_value: List[float] = field(default_factory=list)
    benchmark: List[float] = field(default_factory=list)

    def _fO0O7Bf(self, _f1lI7cO: datetime, _fOIO7cl: float, _f1lO7c2: float, _fO0I7c3: float, _f10O7c4: float=0.0):
        self.timestamps.append(_f1lI7cO)
        self._fOIO7cl.append(_fOIO7cl)
        self._f1lO7c2.append(_f1lO7c2)
        self._fO0I7c3.append(_fO0I7c3)
        self.benchmark.append(_f10O7c4)

    @property
    def _fOIO7c5(self) -> List[float]:
        if len(self._fOIO7cl) < 2:
            return []
        return [(self._fOIO7cl[i] - self._fOIO7cl[i - 1]) / self._fOIO7cl[i - 1] for i in range(1, len(self._fOIO7cl))]

@dataclass
class _cl107c6:
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: timedelta = timedelta(0)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: timedelta = timedelta(0)
    avg_exposure: float = 0.0
    max_exposure: float = 0.0
    time_in_market_pct: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    total_commission: float = 0.0
    turnover: float = 0.0

class _cIIl7c7(ABC):

    def __init__(self, _fI0I7c8: str):
        self._fI0I7c8 = _fI0I7c8
        self._parameters: Dict[str, Any] = {}

    @abstractmethod
    def _fll07c9(self, _f0l07cA: _cIlI7B5, _flI17cB: 'BacktestContext') -> List[Order]:
        pass

    def _flIl7cc(self, _f1OO7cd: Fill, _flI17cB: 'BacktestContext'):
        pass

    def _fll17cE(self, _flI17cB: 'BacktestContext'):
        pass

    def _fO1I7cf(self, _flI17cB: 'BacktestContext'):
        pass

    def _f1117dO(self, _fI0I7c8: str, _f1007dl: Any):
        self._parameters[_fI0I7c8] = _f1007dl

    def _fll17d2(self, _fI0I7c8: str, _fIll7d3: Any=None) -> Any:
        return self._parameters.get(_fI0I7c8, _fIll7d3)

class _cl117d4:

    def __init__(self, _fI107d5: _cIIO7Bc, _f0007d6: Optional[FeatureStore]=None):
        self._fI107d5 = _fI107d5
        self._f0007d6 = _f0007d6
        self.current_time: datetime = _fI107d5.start_date
        self._f1lO7c2: float = _fI107d5.initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.bars: Dict[str, List[_cIlI7B5]] = defaultdict(list)
        self.trades: List[_cllO7Bd] = []
        self.orders: List[Order] = []
        self.fills: List[Fill] = []
        self.current_prices: Dict[str, float] = {}

    def _fl017d7(self, _f0lO7d8: str) -> Optional[Position]:
        return self.positions.get(_f0lO7d8)

    def _fO0l7d9(self, _f0lO7d8: str) -> int:
        pos = self.positions.get(_f0lO7d8)
        return pos._fl1O7BA if pos else 0

    @property
    def _fOIO7cl(self) -> float:
        _fO0I7c3 = sum((pos._fl1O7BA * self.current_prices.get(_f0lO7d8, pos.avg_price) for _f0lO7d8, pos in self.positions.items()))
        return self._f1lO7c2 + _fO0I7c3

    @property
    def _f0IO7dA(self) -> float:
        _fO0I7c3 = sum((abs(pos._fl1O7BA) * self.current_prices.get(_f0lO7d8, pos.avg_price) for _f0lO7d8, pos in self.positions.items()))
        return _fO0I7c3 / self._fOIO7cl if self._fOIO7cl > 0 else 0

    def _fIOl7dB(self, _f0lO7d8: str) -> Optional[_cIlI7B5]:
        bars = self.bars.get(_f0lO7d8, [])
        return bars[-1] if bars else None

    def _flO17dc(self, _f0lO7d8: str, _fI0I7dd: int=1) -> List[_cIlI7B5]:
        bars = self.bars.get(_f0lO7d8, [])
        return bars[-_fI0I7dd:] if bars else []

    def _fl0I7dE(self, _f0lO7d8: str, _fI0I7dd: int=1) -> List[float]:
        bars = self._flO17dc(_f0lO7d8, _fI0I7dd)
        return [b.close for b in bars]

    def _fOO07df(self, _fI0I7EO: str, _f0lO7d8: str) -> Optional[float]:
        if self._f0007d6 is None:
            return None
        _f1007dl = self._f0007d6.get_latest(feature_name=_fI0I7EO, entity_id=_f0lO7d8, as_of_time=self.current_time)
        return _f1007dl._f1007dl if _f1007dl else None

class _cIlO7El:

    def __init__(self):
        self._queue: List[_cll17B3] = []
        self._counter = 0

    def _f0l07E2(self, _f0O17E3: _cll17B3):
        heapq.heappush(self._queue, _f0O17E3)

    def _f0Ol7E4(self) -> Optional[_cll17B3]:
        if self._queue:
            return heapq.heappop(self._queue)
        return None

    def _f1Il7E5(self) -> Optional[_cll17B3]:
        return self._queue[0] if self._queue else None

    @property
    def dimension(self) -> bool:
        return len(self._queue) == 0

class _cOII7E7:

    def __init__(self, _fI107d5: _cIIO7Bc, _f0007d6: Optional[FeatureStore]=None):
        self._fI107d5 = _fI107d5
        self._feature_store = _f0007d6
        self._strategies: List[_cIIl7c7] = []
        self._data: Dict[str, List[_cIlI7B5]] = {}
        self._event_queue = _cIlO7El()
        self._context: Optional[_cl117d4] = None
        self._equity_curve: Optional[_cO0I7BE] = None
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['JonesEngine', 'FeatureStore', 'OrderManager', 'TradeCube'], connection_types={'JonesEngine': 'uses', 'FeatureStore': 'reads', 'OrderManager': 'simulates', 'TradeCube': 'feeds'})
    def _fOO07E8(self, _fIl07E9: _cIIl7c7):
        self._strategies.append(_fIl07E9)

    def _flII7EA(self, _f0lO7d8: str, _fO1O7EB: List[_cIlI7B5]):
        self._data[_f0lO7d8] = sorted(_fO1O7EB, key=lambda b: b._f1lI7cO)

    def _f0OI7Ec(self) -> 'BacktestResult':
        self._context = _cl117d4(config=self._fI107d5, feature_store=self._feature_store)
        self._equity_curve = _cO0I7BE()
        for _fIl07E9 in self._strategies:
            _fIl07E9._fll17cE(self._context)
        self._build_event_queue()
        while not self._event_queue.dimension:
            _f0O17E3 = self._event_queue._f0Ol7E4()
            self._process_event(_f0O17E3)
        for _fIl07E9 in self._strategies:
            _fIl07E9._fO1I7cf(self._context)
        self._close_all_positions()
        metrics = self._calculate_metrics()
        return BacktestResult(config=self._fI107d5, equity_curve=self._equity_curve, trades=self._context.trades, metrics=metrics, orders=self._context.orders, fills=self._context.fills)

    def _fOI17Ed(self):
        for _f0lO7d8, _fO1O7EB in self._data.items():
            for _f0l07cA in _fO1O7EB:
                if self._fI107d5.start_date <= _f0l07cA._f1lI7cO <= self._fI107d5.end_date:
                    _f0O17E3 = _cll17B3(timestamp=_f0l07cA._f1lI7cO, event_type=_c1O07Bl.BAR, symbol=_f0lO7d8, data={'bar': _f0l07cA})
                    self._event_queue._f0l07E2(_f0O17E3)

    def _flIO7EE(self, _f0O17E3: _cll17B3):
        self._context.current_time = _f0O17E3._f1lI7cO
        if _f0O17E3.event_type == _c1O07Bl.BAR:
            self._process_bar_event(_f0O17E3)
        elif _f0O17E3.event_type == _c1O07Bl.FILL:
            self._process_fill_event(_f0O17E3)
        elif _f0O17E3.event_type == _c1O07Bl.ORDER:
            self._process_order_event(_f0O17E3)

    def _fI1l7Ef(self, _f0O17E3: _cll17B3):
        _f0l07cA: _cIlI7B5 = _f0O17E3.data['bar']
        self._context._fO1O7EB[_f0l07cA._f0lO7d8].append(_f0l07cA)
        self._context.current_prices[_f0l07cA._f0lO7d8] = _f0l07cA.close
        self._process_pending_orders(_f0l07cA)
        for _fIl07E9 in self._strategies:
            try:
                orders = _fIl07E9._fll07c9(_f0l07cA, self._context)
                for order in orders:
                    self._submit_order(order, _f0l07cA)
            except Exception as e:
                pass
        self._record_equity()

    def _f01O7fO(self, _f0l07cA: _cIlI7B5):
        remaining = []
        for order in self._context.pending_orders:
            if order._f0lO7d8 != _f0l07cA._f0lO7d8:
                remaining.append(order)
                continue
            _f1OO7cd = self._try_fill_order(order, _f0l07cA)
            if _f1OO7cd:
                self._process_fill(_f1OO7cd)
            else:
                remaining.append(order)
        self._context.pending_orders = remaining

    def _fI0O7fl(self, _f0I17f2: Order, _f0l07cA: _cIlI7B5) -> Optional[Fill]:
        fill_price = None
        if _f0I17f2.order_type == OrderType.MARKET:
            if self._fI107d5.execution_model == _cI117B2.CLOSE:
                fill_price = _f0l07cA.close
            elif self._fI107d5.execution_model == _cI117B2.NEXT_OPEN:
                fill_price = _f0l07cA.open
            else:
                fill_price = _f0l07cA.close
        elif _f0I17f2.order_type == OrderType.LIMIT:
            if _f0I17f2._fl0l7BB == OrderSide.BUY:
                if _f0l07cA.low <= _f0I17f2.limit_price:
                    fill_price = min(_f0I17f2.limit_price, _f0l07cA.open)
            elif _f0l07cA.high >= _f0I17f2.limit_price:
                fill_price = max(_f0I17f2.limit_price, _f0l07cA.open)
        elif _f0I17f2.order_type == OrderType.STOP:
            if _f0I17f2._fl0l7BB == OrderSide.BUY:
                if _f0l07cA.high >= _f0I17f2.stop_price:
                    fill_price = max(_f0I17f2.stop_price, _f0l07cA.open)
            elif _f0l07cA.low <= _f0I17f2.stop_price:
                fill_price = min(_f0I17f2.stop_price, _f0l07cA.open)
        if fill_price is None:
            return None
        commission, adjusted_price = self._fI107d5.transaction_costs._fIOO7B8(price=fill_price, quantity=_f0I17f2._fl1O7BA, side=_f0I17f2._fl0l7BB)
        return Fill(fill_id=f'fill_{len(self._context.fills)}', order_id=_f0I17f2.order_id, symbol=_f0I17f2._f0lO7d8, side=_f0I17f2._fl0l7BB, quantity=_f0I17f2._fl1O7BA, price=adjusted_price, commission=commission, timestamp=self._context.current_time)

    def _f1017f3(self, _f1OO7cd: Fill):
        self._context.fills.append(_f1OO7cd)
        cost = _f1OO7cd._fOIl7B9 * _f1OO7cd._fl1O7BA
        if _f1OO7cd._fl0l7BB == OrderSide.BUY:
            self._context._f1lO7c2 -= cost + _f1OO7cd.commission
        else:
            self._context._f1lO7c2 += cost - _f1OO7cd.commission
        self._update_position(_f1OO7cd)
        for _fIl07E9 in self._strategies:
            _fIl07E9._flIl7cc(_f1OO7cd, self._context)

    def _fl1I7f4(self, _f1OO7cd: Fill):
        current = self._context.positions.get(_f1OO7cd._f0lO7d8)
        if current is None:
            self._context.positions[_f1OO7cd._f0lO7d8] = Position(symbol=_f1OO7cd._f0lO7d8, quantity=_f1OO7cd._fl1O7BA if _f1OO7cd._fl0l7BB == OrderSide.BUY else -_f1OO7cd._fl1O7BA, avg_price=_f1OO7cd._fOIl7B9, unrealized_pnl=0.0, realized_pnl=0.0)
        else:
            old_qty = current._fl1O7BA
            fill_qty = _f1OO7cd._fl1O7BA if _f1OO7cd._fl0l7BB == OrderSide.BUY else -_f1OO7cd._fl1O7BA
            new_qty = old_qty + fill_qty
            if new_qty == 0:
                pnl = (_f1OO7cd._fOIl7B9 - current.avg_price) * abs(old_qty)
                if old_qty < 0:
                    pnl = -pnl
                self._record_trade(_f1OO7cd, current, pnl)
                del self._context.positions[_f1OO7cd._f0lO7d8]
            elif old_qty > 0 and fill_qty < 0 or (old_qty < 0 and fill_qty > 0):
                closed_qty = min(abs(old_qty), abs(fill_qty))
                pnl = (_f1OO7cd._fOIl7B9 - current.avg_price) * closed_qty
                if old_qty < 0:
                    pnl = -pnl
                current._fl1O7BA = new_qty
                current.realized_pnl += pnl
            else:
                total_cost = current.avg_price * abs(old_qty) + _f1OO7cd._fOIl7B9 * abs(fill_qty)
                current._fl1O7BA = new_qty
                current.avg_price = total_cost / abs(new_qty)

    def _fI0O7f5(self, _f1OO7cd: Fill, _f0lO7f6: Position, _f0OI7f7: float):
        trade = _cllO7Bd(trade_id=f'trade_{len(self._context.trades)}', symbol=_f1OO7cd._f0lO7d8, side=OrderSide.BUY if _f0lO7f6._fl1O7BA > 0 else OrderSide.SELL, entry_time=self._context.current_time - timedelta(days=1), exit_time=self._context.current_time, entry_price=_f0lO7f6.avg_price, exit_price=_f1OO7cd._fOIl7B9, quantity=abs(_f0lO7f6._fl1O7BA), pnl=_f0OI7f7, pnl_pct=_f0OI7f7 / (_f0lO7f6.avg_price * abs(_f0lO7f6._fl1O7BA)), commission=_f1OO7cd.commission, duration=timedelta(days=1))
        self._context.trades.append(trade)

    def _fI1I7f8(self, _f0I17f2: Order, _f0l07cA: _cIlI7B5):
        if not self._validate_order(_f0I17f2):
            _f0I17f2.status = OrderStatus.REJECTED
            self._context.orders.append(_f0I17f2)
            return
        _f0I17f2.status = OrderStatus.PENDING
        self._context.orders.append(_f0I17f2)
        self._context.pending_orders.append(_f0I17f2)

    def _f0lO7f9(self, _f0I17f2: Order) -> bool:
        if _f0I17f2._fl0l7BB == OrderSide.BUY:
            estimated_cost = _f0I17f2._fl1O7BA * self._context.current_prices.get(_f0I17f2._f0lO7d8, 0)
            if estimated_cost > self._context._f1lO7c2:
                return False
        current_equity = self._context._fOIO7cl
        position_value = _f0I17f2._fl1O7BA * self._context.current_prices.get(_f0I17f2._f0lO7d8, 0)
        if position_value / current_equity > self._fI107d5.max_position_pct:
            return False
        return True

    def _fI1l7fA(self):
        _fO0I7c3 = sum((pos._fl1O7BA * self._context.current_prices.get(_f0lO7d8, pos.avg_price) for _f0lO7d8, pos in self._context.positions.items()))
        self._equity_curve._fO0O7Bf(timestamp=self._context.current_time, equity=self._context._fOIO7cl, cash=self._context._f1lO7c2, positions_value=_fO0I7c3)

    def _fI107fB(self):
        for _f0lO7d8, _f0lO7f6 in list(self._context.positions.items()):
            _fOIl7B9 = self._context.current_prices.get(_f0lO7d8, _f0lO7f6.avg_price)
            _f1OO7cd = Fill(fill_id=f'close_{_f0lO7d8}', order_id='close', symbol=_f0lO7d8, side=OrderSide.SELL if _f0lO7f6._fl1O7BA > 0 else OrderSide.BUY, quantity=abs(_f0lO7f6._fl1O7BA), price=_fOIl7B9, commission=0, timestamp=self._context.current_time)
            self._f1017f3(_f1OO7cd)

    def _f1O07fc(self, _f0O17E3: _cll17B3):
        _f1OO7cd = _f0O17E3.data.get('fill')
        if _f1OO7cd:
            self._f1017f3(_f1OO7cd)

    def _fIll7fd(self, _f0O17E3: _cll17B3):
        pass

    def _f1OO7fE(self) -> _cl107c6:
        metrics = _cl107c6()
        if not self._equity_curve._fOIO7cl:
            return metrics
        initial = self._fI107d5.initial_capital
        final = self._equity_curve._fOIO7cl[-1]
        metrics.total_return = (final - initial) / initial
        days = (self._fI107d5.end_date - self._fI107d5.start_date).days
        years = days / 365.25
        if years > 0:
            metrics.cagr = (final / initial) ** (1 / years) - 1
            metrics.annualized_return = metrics.cagr
        _fOIO7c5 = self._equity_curve._fOIO7c5
        if _fOIO7c5:
            mean_return = sum(_fOIO7c5) / len(_fOIO7c5)
            variance = sum(((r - mean_return) ** 2 for r in _fOIO7c5)) / len(_fOIO7c5)
            daily_vol = math.sqrt(variance)
            metrics.volatility = daily_vol * math.sqrt(252)
            neg_returns = [r for r in _fOIO7c5 if r < 0]
            if neg_returns:
                neg_variance = sum((r ** 2 for r in neg_returns)) / len(neg_returns)
                metrics.downside_volatility = math.sqrt(neg_variance) * math.sqrt(252)
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility
            if metrics.downside_volatility > 0:
                metrics.sortino_ratio = metrics.annualized_return / metrics.downside_volatility
        peak = self._equity_curve._fOIO7cl[0]
        max_dd = 0
        for _fOIO7cl in self._equity_curve._fOIO7cl:
            peak = max(peak, _fOIO7cl)
            dd = (peak - _fOIO7cl) / peak
            max_dd = max(max_dd, dd)
        metrics.max_drawdown = max_dd
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
        if self._context.trades:
            trades = self._context.trades
            metrics.total_trades = len(trades)
            metrics.winning_trades = sum((1 for t in trades if t._f0OI7f7 > 0))
            metrics.losing_trades = sum((1 for t in trades if t._f0OI7f7 < 0))
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
            wins = [t._f0OI7f7 for t in trades if t._f0OI7f7 > 0]
            losses = [t._f0OI7f7 for t in trades if t._f0OI7f7 < 0]
            if wins:
                metrics.avg_win = sum(wins) / len(wins)
                metrics.largest_win = max(wins)
            if losses:
                metrics.avg_loss = sum(losses) / len(losses)
                metrics.largest_loss = min(losses)
            if losses and sum((abs(l) for l in losses)) > 0:
                metrics.profit_factor = sum(wins) / sum((abs(l) for l in losses))
            metrics.avg_trade_pnl = sum((t._f0OI7f7 for t in trades)) / len(trades)
            metrics.total_commission = sum((t.commission for t in trades))
        return metrics

@dataclass
class _cOII7ff:
    _fI107d5: _cIIO7Bc
    equity_curve: _cO0I7BE
    trades: List[_cllO7Bd]
    metrics: _cl107c6
    orders: List[Order]
    fills: List[Fill]

    def _f0l08OO(self) -> str:
        m = self.metrics
        return f'\nBacktest Summary\n================\nPeriod: {self._fI107d5.start_date.date()} to {self._fI107d5.end_date.date()}\nInitial Capital: ${self._fI107d5.initial_capital:,.2f}\nFinal Equity: ${self.equity_curve._fOIO7cl[-1]:,.2f}\n\nPerformance\n-----------\nTotal Return: {m.total_return:.2%}\nCAGR: {m.cagr:.2%}\nVolatility: {m.volatility:.2%}\nSharpe Ratio: {m.sharpe_ratio:.2f}\nSortino Ratio: {m.sortino_ratio:.2f}\nMax Drawdown: {m.max_drawdown:.2%}\nCalmar Ratio: {m.calmar_ratio:.2f}\n\nTrading\n-------\nTotal Trades: {m.total_trades}\nWin Rate: {m.win_rate:.1%}\nProfit Factor: {m.profit_factor:.2f}\nAvg Trade P&L: ${m.avg_trade_pnl:.2f}\nLargest Win: ${m.largest_win:.2f}\nLargest Loss: ${m.largest_loss:.2f}\nTotal Commission: ${m.total_commission:.2f}\n'

class _cO108Ol:

    def __init__(self, _f00I8O2: _cOII7E7, _fIl07E9: _cIIl7c7, _fl008O3: Dict[str, List[Any]], _fllO8O4: str='sharpe_ratio'):
        self._engine = _f00I8O2
        self._strategy = _fIl07E9
        self._parameter_space = _fl008O3
        self._optimization_metric = _fllO8O4

    @bridge(connects_to=['BacktestEngine', 'Strategy'], connection_types={'BacktestEngine': 'uses', 'Strategy': 'optimizes'})
    def _f0OI7Ec(self, _fI018O5: int=5, _fOOl8O6: float=0.7) -> 'WalkForwardResult':
        results = []
        best_params_history = []
        _fI107d5 = self._engine._fI107d5
        total_days = (_fI107d5.end_date - _fI107d5.start_date).days
        split_days = total_days // _fI018O5
        for i in range(_fI018O5):
            split_start = _fI107d5.start_date + timedelta(days=i * split_days)
            split_end = split_start + timedelta(days=split_days)
            train_end = split_start + timedelta(days=int(split_days * _fOOl8O6))
            best_params = self._optimize(split_start, train_end)
            best_params_history.append(best_params)
            self._strategy._parameters = best_params
            test_config = copy.copy(_fI107d5)
            test_config.start_date = train_end
            test_config.end_date = split_end
            test_engine = _cOII7E7(test_config, self._engine._feature_store)
            test_engine._data = self._engine._data
            test_engine._fOO07E8(self._strategy)
            result = test_engine._f0OI7Ec()
            results.append(result)
        return WalkForwardResult(splits=_fI018O5, results=results, best_params_history=best_params_history, optimization_metric=self._optimization_metric)

    def _fIlI8O7(self, _fOl08O8: datetime, _fIlI8O9: datetime) -> Dict[str, Any]:
        best_metric = float('-inf')
        best_params = {}
        param_names = list(self._parameter_space.keys())
        param_values = list(self._parameter_space.values())
        from itertools import product
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            for _fI0I7c8, _f1007dl in params.items():
                self._strategy._f1117dO(_fI0I7c8, _f1007dl)
            _fI107d5 = copy.copy(self._engine._fI107d5)
            _fI107d5._fOl08O8 = _fOl08O8
            _fI107d5._fIlI8O9 = _fIlI8O9
            _f00I8O2 = _cOII7E7(_fI107d5, self._engine._feature_store)
            _f00I8O2._data = self._engine._data
            _f00I8O2._fOO07E8(self._strategy)
            result = _f00I8O2._f0OI7Ec()
            metric_value = getattr(result.metrics, self._optimization_metric, 0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params.copy()
        return best_params

@dataclass
class _c1ll8OA:
    splits: int
    results: List[_cOII7ff]
    best_params_history: List[Dict[str, Any]]
    _fllO8O4: str

    @property
    def _f0II8OB(self) -> _cl107c6:
        if not self.results:
            return _cl107c6()
        combined_returns = []
        for result in self.results:
            combined_returns.extend(result.equity_curve._fOIO7c5)
        metrics = _cl107c6()
        if combined_returns:
            mean_return = sum(combined_returns) / len(combined_returns)
            variance = sum(((r - mean_return) ** 2 for r in combined_returns)) / len(combined_returns)
            daily_vol = math.sqrt(variance)
            metrics.annualized_return = mean_return * 252
            metrics.volatility = daily_vol * math.sqrt(252)
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility
        metrics.total_trades = sum((r.metrics.total_trades for r in self.results))
        metrics.win_rate = sum((r.metrics.win_rate * r.metrics.total_trades for r in self.results)) / max(metrics.total_trades, 1)
        return metrics

def _fI1O8Oc(_fOl08O8: datetime, _fIlI8O9: datetime, _fO0O8Od: float=100000.0) -> _cOII7E7:
    _fI107d5 = _cIIO7Bc(start_date=_fOl08O8, end_date=_fIlI8O9, initial_capital=_fO0O8Od)
    return _cOII7E7(_fI107d5)

def _fIO18OE(_fl1I8Of: float=0.001, _fl108lO: float=0.0005) -> _c1I07B7:
    return _c1I07B7(commission_pct=_fl1I8Of, slippage_pct=_fl108lO)