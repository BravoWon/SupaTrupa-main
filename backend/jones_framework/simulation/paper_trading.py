from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Iterator
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import math
from collections import defaultdict
import random
import heapq
from jones_framework.core import bridge, ComponentRegistry

class _cIlOdOf(Enum):
    BACKTEST = 'backtest'
    PAPER_TRADING = 'paper_trading'
    REPLAY = 'replay'
    MONTE_CARLO = 'monte_carlo'
    STRESS_TEST = 'stress_test'

class _c10ldlO(Enum):
    IMMEDIATE = 'immediate'
    PARTIAL = 'partial'
    PROBABILISTIC = 'probabilistic'
    QUEUE_POSITION = 'queue_position'
    MARKET_IMPACT = 'market_impact'

class _cIIldll(Enum):
    NONE = 'none'
    FIXED = 'fixed'
    PERCENTAGE = 'percentage'
    VOLUME_BASED = 'volume_based'
    VOLATILITY_BASED = 'volatility_based'
    ALMGREN_CHRISS = 'almgren_chriss'

class _c0Ildl2(Enum):
    NORMAL = 'normal'
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    CRISIS = 'crisis'
    RECOVERY = 'recovery'

@dataclass
class _c1lldl3:
    mode: _cIlOdOf
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000.0
    margin_requirement: float = 0.5
    fill_model: _c10ldlO = _c10ldlO.PARTIAL
    slippage_model: _cIIldll = _cIIldll.VOLUME_BASED
    latency_ms: float = 10.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission: float = 10.0
    borrow_rate_annual: float = 0.02
    fixed_slippage_bps: float = 5.0
    volume_impact_coef: float = 0.1
    volatility_impact_coef: float = 0.5
    max_position_pct: float = 0.1
    max_leverage: float = 2.0
    max_drawdown_pct: float = 0.25
    data_frequency: str = '1min'
    use_adjusted_prices: bool = True
    random_seed: Optional[int] = None

@dataclass
class _cIlIdl4:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = 'PENDING'
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    queue_position: int = 0

@dataclass
class _c11Idl5:
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class _cl11dl6:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    slippage: float
    timestamp: datetime

@dataclass
class _cO10dl7:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0

    @property
    def _fO0Odl8(self) -> float:
        return (self.high + self.low) / 2

@dataclass
class _cIOIdl9:
    timestamp: datetime
    cash: float
    portfolio_value: float
    total_equity: float
    positions: Dict[str, _c11Idl5]
    num_positions: int
    gross_exposure: float
    net_exposure: float
    long_value: float
    short_value: float
    daily_pnl: float
    total_pnl: float
    total_return_pct: float
    current_drawdown: float
    max_drawdown: float

@dataclass
class _c0lOdlA:
    config: _c1lldl3
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    avg_gross_exposure: float = 0.0
    avg_net_exposure: float = 0.0
    avg_turnover: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    drawdown_curve: List[Tuple[datetime, float]] = field(default_factory=list)

class _cl0IdlB:

    def __init__(self, _f1I1dlc: _cIIldll, _f11Odld: _c1lldl3):
        self._f1I1dlc = _f1I1dlc
        self._f11Odld = _f11Odld

    def _f0OIdlE(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7, _f0O1d2l: float=1000000) -> float:
        if self._f1I1dlc == _cIIldll.NONE:
            return 0.0
        if self._f1I1dlc == _cIIldll.FIXED:
            return self._f11Odld.fixed_slippage_bps / 10000 * _fI1Od2O.close
        if self._f1I1dlc == _cIIldll.PERCENTAGE:
            return self._f11Odld.fixed_slippage_bps / 10000 * _fI1Od2O.close
        if self._f1I1dlc == _cIIldll.VOLUME_BASED:
            participation = _f0I1dlf.quantity / max(1, _fI1Od2O.volume)
            impact = self._f11Odld.volume_impact_coef * math.sqrt(participation)
            return impact * _fI1Od2O.close
        if self._f1I1dlc == _cIIldll.VOLATILITY_BASED:
            volatility = (_fI1Od2O.high - _fI1Od2O.low) / _fI1Od2O.close
            participation = _f0I1dlf.quantity / max(1, _fI1Od2O.volume)
            impact = self._f11Odld.volatility_impact_coef * volatility * math.sqrt(participation)
            return impact * _fI1Od2O.close
        if self._f1I1dlc == _cIIldll.ALMGREN_CHRISS:
            sigma = (_fI1Od2O.high - _fI1Od2O.low) / _fI1Od2O.close
            participation = _f0I1dlf.quantity / max(1, _fI1Od2O.volume)
            permanent = 0.1 * sigma * participation
            temporary = 0.5 * sigma * math.sqrt(participation)
            return (permanent + temporary) * _fI1Od2O.close
        return 0.0

class _cI0Id22:

    def __init__(self, _f1I1dlc: _c10ldlO, _f11Odld: _c1lldl3):
        self._f1I1dlc = _f1I1dlc
        self._f11Odld = _f11Odld
        self._order_queues: Dict[str, List[_cIlIdl4]] = defaultdict(list)
        self._fill_counter = 0

    def _f0lOd23(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7, _fI0ld24: float) -> Optional[_cl11dl6]:
        if self._f1I1dlc == _c10ldlO.IMMEDIATE:
            return self._immediate_fill(_f0I1dlf, _fI1Od2O, _fI0ld24)
        if self._f1I1dlc == _c10ldlO.PARTIAL:
            return self._partial_fill(_f0I1dlf, _fI1Od2O, _fI0ld24)
        if self._f1I1dlc == _c10ldlO.PROBABILISTIC:
            return self._probabilistic_fill(_f0I1dlf, _fI1Od2O, _fI0ld24)
        if self._f1I1dlc == _c10ldlO.QUEUE_POSITION:
            return self._queue_fill(_f0I1dlf, _fI1Od2O, _fI0ld24)
        if self._f1I1dlc == _c10ldlO.MARKET_IMPACT:
            return self._impact_fill(_f0I1dlf, _fI1Od2O, _fI0ld24)
        return None

    def _f0Old25(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7, _fI0ld24: float) -> _cl11dl6:
        remaining = _f0I1dlf.quantity - _f0I1dlf.filled_qty
        if _f0I1dlf.order_type == 'MARKET':
            if _f0I1dlf.side == 'BUY':
                price = _fI1Od2O.close + _fI0ld24
            else:
                price = _fI1Od2O.close - _fI0ld24
        else:
            price = _f0I1dlf.limit_price or _fI1Od2O.close
        commission = self._calculate_commission(remaining, price)
        self._fill_counter += 1
        return _cl11dl6(fill_id=f'FILL-{self._fill_counter:08d}', order_id=_f0I1dlf.order_id, symbol=_f0I1dlf.symbol, side=_f0I1dlf.side, quantity=remaining, price=price, commission=commission, slippage=_fI0ld24, timestamp=_fI1Od2O.timestamp)

    def _f0lOd26(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7, _fI0ld24: float) -> Optional[_cl11dl6]:
        remaining = _f0I1dlf.quantity - _f0I1dlf.filled_qty
        max_fill = int(_fI1Od2O.volume * 0.1)
        fill_qty = min(remaining, max_fill)
        if fill_qty <= 0:
            return None
        if _f0I1dlf.order_type == 'MARKET':
            if _f0I1dlf.side == 'BUY':
                price = _fI1Od2O.close + _fI0ld24
            else:
                price = _fI1Od2O.close - _fI0ld24
        else:
            price = _f0I1dlf.limit_price or _fI1Od2O.close
        commission = self._calculate_commission(fill_qty, price)
        self._fill_counter += 1
        return _cl11dl6(fill_id=f'FILL-{self._fill_counter:08d}', order_id=_f0I1dlf.order_id, symbol=_f0I1dlf.symbol, side=_f0I1dlf.side, quantity=fill_qty, price=price, commission=commission, slippage=_fI0ld24, timestamp=_fI1Od2O.timestamp)

    def _f1IOd27(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7, _fI0ld24: float) -> Optional[_cl11dl6]:
        if _f0I1dlf.order_type != 'LIMIT':
            return self._f0Old25(_f0I1dlf, _fI1Od2O, _fI0ld24)
        if _f0I1dlf.side == 'BUY':
            price_distance = _fI1Od2O.low - _f0I1dlf.limit_price
        else:
            price_distance = _f0I1dlf.limit_price - _fI1Od2O.high
        if price_distance <= 0:
            fill_prob = 0.9
        else:
            pct_distance = price_distance / _fI1Od2O.close
            fill_prob = max(0, 0.5 - pct_distance * 10)
        if random.random() > fill_prob:
            return None
        return self._f0Old25(_f0I1dlf, _fI1Od2O, _fI0ld24)

    def _f00ld28(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7, _fI0ld24: float) -> Optional[_cl11dl6]:
        if _f0I1dlf.order_type != 'LIMIT':
            return self._f0Old25(_f0I1dlf, _fI1Od2O, _fI0ld24)
        if _f0I1dlf not in self._order_queues[_f0I1dlf.symbol]:
            _f0I1dlf.queue_position = len(self._order_queues[_f0I1dlf.symbol])
            self._order_queues[_f0I1dlf.symbol].append(_f0I1dlf)
        if _f0I1dlf.side == 'BUY':
            touched = _fI1Od2O.low <= _f0I1dlf.limit_price
        else:
            touched = _fI1Od2O.high >= _f0I1dlf.limit_price
        if not touched:
            return None
        volume_at_price = _fI1Od2O.volume * 0.1
        volume_before = _f0I1dlf.queue_position * 100
        if volume_before > volume_at_price:
            return None
        self._order_queues[_f0I1dlf.symbol].remove(_f0I1dlf)
        return self._f0Old25(_f0I1dlf, _fI1Od2O, _fI0ld24)

    def _flOOd29(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7, _fI0ld24: float) -> Optional[_cl11dl6]:
        remaining = _f0I1dlf.quantity - _f0I1dlf.filled_qty
        participation = remaining / max(1, _fI1Od2O.volume)
        if participation > 0.25:
            fill_qty = int(_fI1Od2O.volume * 0.25)
        else:
            fill_qty = remaining
        if fill_qty <= 0:
            return None
        extra_impact = _fI0ld24 * participation * 2
        if _f0I1dlf.side == 'BUY':
            price = _fI1Od2O.close + _fI0ld24 + extra_impact
        else:
            price = _fI1Od2O.close - _fI0ld24 - extra_impact
        commission = self._calculate_commission(fill_qty, price)
        self._fill_counter += 1
        return _cl11dl6(fill_id=f'FILL-{self._fill_counter:08d}', order_id=_f0I1dlf.order_id, symbol=_f0I1dlf.symbol, side=_f0I1dlf.side, quantity=fill_qty, price=price, commission=commission, slippage=_fI0ld24 + extra_impact, timestamp=_fI1Od2O.timestamp)

    def _f01Od2A(self, _flOOd2B: int, _f1OId2c: float) -> float:
        commission = _flOOd2B * self._f11Odld.commission_per_share
        commission = max(self._f11Odld.min_commission, commission)
        commission = min(self._f11Odld.max_commission, commission)
        return commission

@bridge('JonesEngine', 'OrderManager', 'RiskEngine', 'FeatureStore')
class _cOO0d2d:

    def __init__(self, _f11Odld: _c1lldl3):
        self._f11Odld = _f11Odld
        self._cash = _f11Odld.initial_capital
        self._positions: Dict[str, _c11Idl5] = {}
        self._orders: Dict[str, _cIlIdl4] = {}
        self._pending_orders: List[_cIlIdl4] = []
        self._fills: List[_cl11dl6] = []
        self._snapshots: List[_cIOIdl9] = []
        self._slippage_calc = _cl0IdlB(_f11Odld.slippage_model, _f11Odld)
        self._fill_sim = _cI0Id22(_f11Odld.fill_model, _f11Odld)
        self._current_time = _f11Odld.start_date
        self._order_counter = 0
        self._high_watermark = _f11Odld.initial_capital
        self._max_drawdown = 0.0
        self._registry = ComponentRegistry.get_instance()
        if _f11Odld.random_seed:
            random.seed(_f11Odld.random_seed)

    @property
    def _flOld2E(self) -> float:
        return self._cash

    @property
    def _f01Id2f(self) -> float:
        return sum((p.market_value for p in self._positions.values()))

    @property
    def _fIl1d3O(self) -> float:
        return self._cash + self._f01Id2f

    @property
    def _f0IId3l(self) -> Dict[str, _c11Idl5]:
        return self._positions.copy()

    def _f10Id32(self, _fO01d33: str, _fOIld34: str, _flOOd2B: int, _fI11d35: str='MARKET', _fOI0d36: Optional[float]=None, _fI11d37: Optional[float]=None) -> str:
        if _flOOd2B <= 0:
            raise ValueError('Quantity must be positive')
        if _fI11d35 == 'LIMIT' and _fOI0d36 is None:
            raise ValueError('Limit price required for limit orders')
        if not self._check_risk_limits(_fO01d33, _fOIld34, _flOOd2B):
            raise ValueError('Order exceeds risk limits')
        self._order_counter += 1
        order_id = f'SIM-{self._order_counter:08d}'
        _f0I1dlf = _cIlIdl4(order_id=order_id, symbol=_fO01d33, side=_fOIld34, order_type=_fI11d35, quantity=_flOOd2B, limit_price=_fOI0d36, stop_price=_fI11d37, created_at=self._current_time)
        self._orders[order_id] = _f0I1dlf
        self._pending_orders.append(_f0I1dlf)
        return order_id

    def _fI0Od38(self, _fI0Id39: str) -> bool:
        _f0I1dlf = self._orders.get(_fI0Id39)
        if not _f0I1dlf or _f0I1dlf.status != 'PENDING':
            return False
        _f0I1dlf.status = 'CANCELLED'
        _f0I1dlf.cancelled_at = self._current_time
        if _f0I1dlf in self._pending_orders:
            self._pending_orders.remove(_f0I1dlf)
        return True

    def _f11Od3A(self, _f0O0d3B: Dict[str, _cO10dl7]):
        self._current_time = next(iter(_f0O0d3B.values())).timestamp
        for _fO01d33, _fI1Od2O in _f0O0d3B.items():
            if _fO01d33 in self._positions:
                pos = self._positions[_fO01d33]
                pos.last_price = _fI1Od2O.close
                pos.market_value = pos._flOOd2B * _fI1Od2O.close
                pos.unrealized_pnl = (_fI1Od2O.close - pos.avg_cost) * pos._flOOd2B
                pos.last_updated = self._current_time
        for _f0I1dlf in list(self._pending_orders):
            if _f0I1dlf._fO01d33 not in _f0O0d3B:
                continue
            _fI1Od2O = _f0O0d3B[_f0I1dlf._fO01d33]
            if not self._should_trigger(_f0I1dlf, _fI1Od2O):
                continue
            _fI0ld24 = self._slippage_calc._f0OIdlE(_f0I1dlf, _fI1Od2O)
            fill = self._fill_sim._f0lOd23(_f0I1dlf, _fI1Od2O, _fI0ld24)
            if fill:
                self._process_fill(_f0I1dlf, fill)
        self._take_snapshot()

    def _fOI1d3c(self, _f0I1dlf: _cIlIdl4, _fI1Od2O: _cO10dl7) -> bool:
        if _f0I1dlf._fI11d35 == 'MARKET':
            return True
        if _f0I1dlf._fI11d35 == 'LIMIT':
            if _f0I1dlf._fOIld34 == 'BUY':
                return _fI1Od2O.low <= _f0I1dlf._fOI0d36
            else:
                return _fI1Od2O.high >= _f0I1dlf._fOI0d36
        if _f0I1dlf._fI11d35 == 'STOP':
            if _f0I1dlf._fOIld34 == 'BUY':
                return _fI1Od2O.high >= _f0I1dlf._fI11d37
            else:
                return _fI1Od2O.low <= _f0I1dlf._fI11d37
        return False

    def _f1I0d3d(self, _f0I1dlf: _cIlIdl4, _fI0Od3E: _cl11dl6):
        old_filled = _f0I1dlf.filled_qty
        _f0I1dlf.filled_qty += _fI0Od3E._flOOd2B
        _f0I1dlf.commission += _fI0Od3E.commission
        if _f0I1dlf.filled_qty > 0:
            _f0I1dlf.avg_fill_price = (_f0I1dlf.avg_fill_price * old_filled + _fI0Od3E._f1OId2c * _fI0Od3E._flOOd2B) / _f0I1dlf.filled_qty
        if _f0I1dlf.filled_qty >= _f0I1dlf._flOOd2B:
            _f0I1dlf.status = 'FILLED'
            _f0I1dlf.filled_at = _fI0Od3E.timestamp
            if _f0I1dlf in self._pending_orders:
                self._pending_orders.remove(_f0I1dlf)
        else:
            _f0I1dlf.status = 'PARTIAL'
        _fO01d33 = _fI0Od3E._fO01d33
        if _fO01d33 not in self._positions:
            self._positions[_fO01d33] = _c11Idl5(symbol=_fO01d33, opened_at=_fI0Od3E.timestamp)
        pos = self._positions[_fO01d33]
        if _fI0Od3E._fOIld34 == 'BUY':
            total_cost = pos.avg_cost * pos._flOOd2B + _fI0Od3E._f1OId2c * _fI0Od3E._flOOd2B
            pos._flOOd2B += _fI0Od3E._flOOd2B
            if pos._flOOd2B > 0:
                pos.avg_cost = total_cost / pos._flOOd2B
            self._cash -= _fI0Od3E._f1OId2c * _fI0Od3E._flOOd2B + _fI0Od3E.commission
        else:
            if pos._flOOd2B > 0:
                realized = (_fI0Od3E._f1OId2c - pos.avg_cost) * _fI0Od3E._flOOd2B
                pos.realized_pnl += realized
            pos._flOOd2B -= _fI0Od3E._flOOd2B
            self._cash += _fI0Od3E._f1OId2c * _fI0Od3E._flOOd2B - _fI0Od3E.commission
        if pos._flOOd2B == 0:
            del self._positions[_fO01d33]
        self._fills.append(_fI0Od3E)

    def _fI0Od3f(self, _fO01d33: str, _fOIld34: str, _flOOd2B: int) -> bool:
        current_qty = 0
        if _fO01d33 in self._positions:
            current_qty = self._positions[_fO01d33]._flOOd2B
        if _fOIld34 == 'BUY':
            new_qty = current_qty + _flOOd2B
        else:
            new_qty = current_qty - _flOOd2B
        est_price = 100.0
        if _fO01d33 in self._positions:
            est_price = self._positions[_fO01d33].last_price
        est_value = abs(new_qty) * est_price
        if est_value / self._fIl1d3O > self._f11Odld.max_position_pct:
            return False
        total_exposure = sum((abs(p.market_value) for p in self._positions.values()))
        total_exposure += est_value
        if total_exposure / self._fIl1d3O > self._f11Odld.max_leverage:
            return False
        if self._max_drawdown > self._f11Odld.max_drawdown_pct:
            return False
        return True

    def _fOOId4O(self):
        long_value = sum((p.market_value for p in self._positions.values() if p._flOOd2B > 0))
        short_value = sum((abs(p.market_value) for p in self._positions.values() if p._flOOd2B < 0))
        gross = long_value + short_value
        net = long_value - short_value
        equity = self._fIl1d3O
        if equity > self._high_watermark:
            self._high_watermark = equity
        current_dd = (self._high_watermark - equity) / self._high_watermark
        if current_dd > self._max_drawdown:
            self._max_drawdown = current_dd
        total_pnl = equity - self._f11Odld.initial_capital
        total_return = total_pnl / self._f11Odld.initial_capital
        daily_pnl = 0.0
        if self._snapshots:
            prev = self._snapshots[-1]
            daily_pnl = equity - prev._fIl1d3O
        snapshot = _cIOIdl9(timestamp=self._current_time, cash=self._cash, portfolio_value=self._f01Id2f, total_equity=equity, positions=self._f0IId3l, num_positions=len(self._positions), gross_exposure=gross, net_exposure=net, long_value=long_value, short_value=short_value, daily_pnl=daily_pnl, total_pnl=total_pnl, total_return_pct=total_return * 100, current_drawdown=current_dd, max_drawdown=self._max_drawdown)
        self._snapshots.append(snapshot)

    def _f00Id4l(self) -> _c0lOdlA:
        results = _c0lOdlA(config=self._f11Odld)
        if not self._snapshots:
            return results
        results.total_return = (self._fIl1d3O - self._f11Odld.initial_capital) / self._f11Odld.initial_capital
        results.max_drawdown = self._max_drawdown
        daily_returns = []
        for i in range(1, len(self._snapshots)):
            prev_eq = self._snapshots[i - 1]._fIl1d3O
            curr_eq = self._snapshots[i]._fIl1d3O
            if prev_eq > 0:
                daily_returns.append((curr_eq - prev_eq) / prev_eq)
        if daily_returns:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum(((r - mean_return) ** 2 for r in daily_returns)) / len(daily_returns)
            results.volatility = math.sqrt(variance * 252)
            if results.volatility > 0:
                results.sharpe_ratio = mean_return * 252 / results.volatility
            downside_returns = [r for r in daily_returns if r < 0]
            if downside_returns:
                downside_var = sum((r ** 2 for r in downside_returns)) / len(downside_returns)
                downside_vol = math.sqrt(downside_var * 252)
                if downside_vol > 0:
                    results.sortino_ratio = mean_return * 252 / downside_vol
            results.annualized_return = mean_return * 252
        if results.max_drawdown > 0:
            results.calmar_ratio = results.annualized_return / results.max_drawdown
        results.total_trades = len(self._fills)
        trade_pnls = []
        order_pnls: Dict[str, float] = defaultdict(float)
        for _fI0Od3E in self._fills:
            pass
        results.total_commission = sum((f.commission for f in self._fills))
        results.total_slippage = sum((f._fI0ld24 * f._flOOd2B for f in self._fills))
        results.total_costs = results.total_commission + results.total_slippage
        if self._snapshots:
            results.avg_gross_exposure = sum((s.gross_exposure for s in self._snapshots)) / len(self._snapshots)
            results.avg_net_exposure = sum((s.net_exposure for s in self._snapshots)) / len(self._snapshots)
        results.start_date = self._f11Odld.start_date
        results.end_date = self._f11Odld.end_date
        results.trading_days = len(self._snapshots)
        results.equity_curve = [(s.timestamp, s._fIl1d3O) for s in self._snapshots]
        results.drawdown_curve = [(s.timestamp, s.current_drawdown) for s in self._snapshots]
        return results

@bridge('JonesEngine', 'DataPipeline', 'FeatureStore')
class _c00Od42:

    def __init__(self, _f11Odld: _c1lldl3):
        self._f11Odld = _f11Odld
        self._paper_engine = _cOO0d2d(_f11Odld)
        self._data_cache: Dict[str, List[_cO10dl7]] = {}
        self._current_idx = 0
        self._registry = ComponentRegistry.get_instance()

    def _fI1ld43(self, _fO01d33: str, _f0O0d3B: List[_cO10dl7]):
        self._data_cache[_fO01d33] = sorted(_f0O0d3B, key=lambda b: b.timestamp)

    def _f010d44(self, _f0lld45: Callable[[Dict[str, _cO10dl7], 'BacktestEngine'], None]) -> _c0lOdlA:
        all_timestamps: Set[datetime] = set()
        for _f0O0d3B in self._data_cache.values():
            for _fI1Od2O in _f0O0d3B:
                all_timestamps.add(_fI1Od2O.timestamp)
        sorted_timestamps = sorted(all_timestamps)
        sorted_timestamps = [ts for ts in sorted_timestamps if self._f11Odld.start_date <= ts <= self._f11Odld.end_date]
        for timestamp in sorted_timestamps:
            current_bars = {}
            for _fO01d33, _f0O0d3B in self._data_cache.items():
                for _fI1Od2O in _f0O0d3B:
                    if _fI1Od2O.timestamp == timestamp:
                        current_bars[_fO01d33] = _fI1Od2O
                        break
            if current_bars:
                _f0lld45(current_bars, self)
                self._paper_engine._f11Od3A(current_bars)
            self._current_idx += 1
        return self._paper_engine._f00Id4l()

    def _f10Id32(self, _fO01d33: str, _fOIld34: str, _flOOd2B: int, _fI11d35: str='MARKET', _fOI0d36: Optional[float]=None) -> str:
        return self._paper_engine._f10Id32(_fO01d33, _fOIld34, _flOOd2B, _fI11d35, _fOI0d36)

    def _fI0Od38(self, _fI0Id39: str) -> bool:
        return self._paper_engine._fI0Od38(_fI0Id39)

    @property
    def _f0IId3l(self) -> Dict[str, _c11Idl5]:
        return self._paper_engine._f0IId3l

    @property
    def _flOld2E(self) -> float:
        return self._paper_engine._flOld2E

    @property
    def _f1IOd46(self) -> float:
        return self._paper_engine._fIl1d3O

@bridge('JonesEngine', 'DataPipeline')
class _clIld47:

    def __init__(self, _f11Odld: _c1lldl3, _flOOd48: int=1000):
        self._f11Odld = _f11Odld
        self._flOOd48 = _flOOd48
        self._results: List[_c0lOdlA] = []
        self._registry = ComponentRegistry.get_instance()

    def _f00ld49(self, _fll1d4A: List[float], _fI01d4B: int) -> List[List[float]]:
        paths = []
        mean_return = sum(_fll1d4A) / len(_fll1d4A)
        variance = sum(((r - mean_return) ** 2 for r in _fll1d4A)) / len(_fll1d4A)
        std_return = math.sqrt(variance)
        for _ in range(self._flOOd48):
            path = []
            for _ in range(_fI01d4B):
                if random.random() < 0.5:
                    ret = random.choice(_fll1d4A)
                else:
                    ret = random.gauss(mean_return, std_return)
                path.append(ret)
            paths.append(path)
        return paths

    def _fIl1d4c(self, _fl00d4d: float, _f100d4E: float, _f10Id4f: float, _fI01d4B: int, _fO0Id5O: float=1 / 252) -> List[List[float]]:
        paths = []
        for _ in range(self._flOOd48):
            path = [_fl00d4d]
            _f1OId2c = _fl00d4d
            for _ in range(_fI01d4B):
                z = random.gauss(0, 1)
                _f1OId2c = _f1OId2c * math.exp((_f100d4E - 0.5 * _f10Id4f ** 2) * _fO0Id5O + _f10Id4f * math.sqrt(_fO0Id5O) * z)
                path.append(_f1OId2c)
            paths.append(path)
        return paths

    def _f01Od5l(self, _flI1d52: List[float], _f100d53: float=0.95) -> float:
        sorted_returns = sorted(_flI1d52)
        idx = int(len(sorted_returns) * (1 - _f100d53))
        return sorted_returns[idx]

    def _fIlOd54(self, _flI1d52: List[float], _f100d53: float=0.95) -> float:
        var = self._f01Od5l(_flI1d52, _f100d53)
        tail_returns = [r for r in _flI1d52 if r <= var]
        if tail_returns:
            return sum(tail_returns) / len(tail_returns)
        return var

    def _f00Od55(self, _f0lld45: Callable, _f1O0d56: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        all_terminal_values = []
        all_max_drawdowns = []
        all_sharpe_ratios = []
        for path_idx in range(self._flOOd48):
            bt_config = _c1lldl3(mode=_cIlOdOf.BACKTEST, start_date=self._f11Odld.start_date, end_date=self._f11Odld.end_date, initial_capital=self._f11Odld.initial_capital, random_seed=path_idx)
            engine = _c00Od42(bt_config)
            for _fO01d33, paths in _f1O0d56.items():
                path = paths[path_idx]
                _f0O0d3B = []
                start = self._f11Odld.start_date
                for i, _f1OId2c in enumerate(path):
                    _fI1Od2O = _cO10dl7(symbol=_fO01d33, timestamp=start + timedelta(days=i), open=_f1OId2c, high=_f1OId2c * 1.01, low=_f1OId2c * 0.99, close=_f1OId2c, volume=1000000)
                    _f0O0d3B.append(_fI1Od2O)
                engine._fI1ld43(_fO01d33, _f0O0d3B)
            results = engine._f010d44(_f0lld45)
            all_terminal_values.append(results.equity_curve[-1][1] if results.equity_curve else self._f11Odld.initial_capital)
            all_max_drawdowns.append(results.max_drawdown)
            all_sharpe_ratios.append(results.sharpe_ratio)
            self._results.append(results)
        _flI1d52 = [(v - self._f11Odld.initial_capital) / self._f11Odld.initial_capital for v in all_terminal_values]
        return {'mean_return': sum(_flI1d52) / len(_flI1d52), 'median_return': sorted(_flI1d52)[len(_flI1d52) // 2], 'std_return': math.sqrt(sum(((r - sum(_flI1d52) / len(_flI1d52)) ** 2 for r in _flI1d52)) / len(_flI1d52)), 'var_95': self._f01Od5l(_flI1d52, 0.95), 'cvar_95': self._fIlOd54(_flI1d52, 0.95), 'max_drawdown_mean': sum(all_max_drawdowns) / len(all_max_drawdowns), 'max_drawdown_worst': max(all_max_drawdowns), 'sharpe_mean': sum(all_sharpe_ratios) / len(all_sharpe_ratios), 'probability_of_loss': sum((1 for r in _flI1d52 if r < 0)) / len(_flI1d52), 'probability_of_gain': sum((1 for r in _flI1d52 if r > 0)) / len(_flI1d52)}

@bridge('JonesEngine', 'RiskEngine')
class _cIlld57:

    def __init__(self, _f11Odld: _c1lldl3):
        self._f11Odld = _f11Odld
        self._scenarios: Dict[str, StressScenario] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._registry = ComponentRegistry.get_instance()

    def _fIlld58(self, _f100d59: 'StressScenario'):
        self._scenarios[_f100d59.name] = _f100d59

    def _fl01d5A(self, _f01Id5B: str, _flOId5c: str, _fl1Id5d: Dict[str, float], _fl01d5E: float=2.0):
        _f100d59 = StressScenario(name=_f01Id5B, description=_flOId5c, price_shocks=_fl1Id5d, volatility_multiplier=_fl01d5E)
        self._scenarios[_f01Id5B] = _f100d59

    def _fOlld5f(self, _fO00d6O: str, _fllId6l: Dict[str, _c11Idl5]) -> Dict[str, Any]:
        _f100d59 = self._scenarios.get(_fO00d6O)
        if not _f100d59:
            raise ValueError(f'Unknown scenario: {_fO00d6O}')
        total_value = sum((p.market_value for p in _fllId6l.values()))
        stressed_value = 0.0
        position_impacts = {}
        for _fO01d33, position in _fllId6l.items():
            shock = _f100d59._fl1Id5d.get(_fO01d33, _f100d59.default_shock)
            stressed_price = position.last_price * (1 + shock)
            stressed_mv = position._flOOd2B * stressed_price
            stressed_value += stressed_mv
            position_impacts[_fO01d33] = {'original_value': position.market_value, 'stressed_value': stressed_mv, 'pnl': stressed_mv - position.market_value, 'pnl_pct': (stressed_mv - position.market_value) / position.market_value if position.market_value != 0 else 0}
        portfolio_pnl = stressed_value - total_value
        portfolio_pnl_pct = portfolio_pnl / total_value if total_value > 0 else 0
        result = {'scenario': _fO00d6O, 'description': _f100d59._flOId5c, 'original_portfolio_value': total_value, 'stressed_portfolio_value': stressed_value, 'portfolio_pnl': portfolio_pnl, 'portfolio_pnl_pct': portfolio_pnl_pct, 'position_impacts': position_impacts, 'worst_position': min(position_impacts.items(), key=lambda x: x[1]['pnl_pct'])[0] if position_impacts else None, 'best_position': max(position_impacts.items(), key=lambda x: x[1]['pnl_pct'])[0] if position_impacts else None}
        self._results[_fO00d6O] = result
        return result

    def _fO0Od62(self, _fllId6l: Dict[str, _c11Idl5]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for _fO00d6O in self._scenarios:
            results[_fO00d6O] = self._fOlld5f(_fO00d6O, _fllId6l)
        return results

    def _fllId63(self) -> Tuple[str, Dict[str, Any]]:
        if not self._results:
            return (None, {})
        worst = min(self._results.items(), key=lambda x: x[1]['portfolio_pnl_pct'])
        return worst

@dataclass
class _cl00d64:
    _f01Id5B: str
    _flOId5c: str
    _fl1Id5d: Dict[str, float]
    _fl01d5E: float = 1.0
    correlation_shock: float = 0.0
    liquidity_reduction: float = 0.0
    default_shock: float = -0.1

@bridge('JonesEngine', 'OrderManager')
class _c10Id65:

    def __init__(self, _fO01d33: str):
        self._fO01d33 = _fO01d33
        self._bids: List[Tuple[float, int]] = []
        self._asks: List[Tuple[float, int]] = []
        self._last_trade_price = 0.0
        self._last_trade_qty = 0
        self._trades: List[Tuple[datetime, float, int]] = []
        self._registry = ComponentRegistry.get_instance()

    def _f101d66(self, _f1OOd67: float, _fI0Id68: float=10.0, _f1Ild69: int=10, _fIO0d6A: int=1000):
        spread = _f1OOd67 * _fI0Id68 / 10000
        half_spread = spread / 2
        bid_price = _f1OOd67 - half_spread
        ask_price = _f1OOd67 + half_spread
        self._bids = []
        self._asks = []
        tick_size = _f1OOd67 * 0.0001
        for i in range(_f1Ild69):
            size = int(_fIO0d6A * (1 - i * 0.05))
            size = max(100, size)
            self._bids.append((bid_price - i * tick_size, size))
            self._asks.append((ask_price + i * tick_size, size))
        self._bids.sort(key=lambda x: -x[0])
        self._asks.sort(key=lambda x: x[0])

    @property
    def _fll0d6B(self) -> Tuple[float, int]:
        return self._bids[0] if self._bids else (0, 0)

    @property
    def _fllId6c(self) -> Tuple[float, int]:
        return self._asks[0] if self._asks else (0, 0)

    @property
    def _f1OOd67(self) -> float:
        bb, _ = self._fll0d6B
        ba, _ = self._fllId6c
        return (bb + ba) / 2 if bb and ba else 0

    @property
    def _flOId6d(self) -> float:
        bb, _ = self._fll0d6B
        ba, _ = self._fllId6c
        return ba - bb if bb and ba else 0

    def _fl10d6E(self, _fOIld34: str, _flOOd2B: int, _f1lId6f: datetime) -> List[Tuple[float, int]]:
        fills = []
        remaining = _flOOd2B
        if _fOIld34 == 'BUY':
            levels = self._asks
        else:
            levels = self._bids
        new_levels = []
        for _f1OId2c, size in levels:
            if remaining <= 0:
                new_levels.append((_f1OId2c, size))
                continue
            fill_qty = min(remaining, size)
            fills.append((_f1OId2c, fill_qty))
            remaining -= fill_qty
            if size > fill_qty:
                new_levels.append((_f1OId2c, size - fill_qty))
            self._last_trade_price = _f1OId2c
            self._last_trade_qty = fill_qty
            self._trades.append((_f1lId6f, _f1OId2c, fill_qty))
        if _fOIld34 == 'BUY':
            self._asks = new_levels
        else:
            self._bids = new_levels
        return fills

    def _fO0Id7O(self, _fOIld34: str, _flOOd2B: int, _f1OId2c: float) -> int:
        if _fOIld34 == 'BUY':
            position = 0
            for i, (bid_price, bid_size) in enumerate(self._bids):
                if _f1OId2c >= bid_price:
                    self._bids.insert(i, (_f1OId2c, _flOOd2B))
                    return position
                position += bid_size
            self._bids.append((_f1OId2c, _flOOd2B))
            return position
        else:
            position = 0
            for i, (ask_price, ask_size) in enumerate(self._asks):
                if _f1OId2c <= ask_price:
                    self._asks.insert(i, (_f1OId2c, _flOOd2B))
                    return position
                position += ask_size
            self._asks.append((_f1OId2c, _flOOd2B))
            return position

    def _fI1Id7l(self, _fOI0d72: float=0.001):
        _fO0Odl8 = self._f1OOd67
        change = _fO0Odl8 * _fOI0d72 * random.gauss(0, 1)
        new_mid = _fO0Odl8 + change
        _flOId6d = self._flOId6d
        self._bids = [(p + change, s) for p, s in self._bids]
        self._asks = [(p + change, s) for p, s in self._asks]
        if random.random() < 0.1:
            _fOIld34 = 'BUY' if random.random() < 0.5 else 'SELL'
            _f1OId2c = new_mid * (1 + random.gauss(0, 0.001))
            size = random.randint(100, 1000)
            self._fO0Id7O(_fOIld34, size, _f1OId2c)
        if random.random() < 0.1 and len(self._bids) > 5:
            idx = random.randint(2, len(self._bids) - 1)
            self._bids.pop(idx)
        if random.random() < 0.1 and len(self._asks) > 5:
            idx = random.randint(2, len(self._asks) - 1)
            self._asks.pop(idx)

def _fIlOd73(_fO1ld74: float=1000000, _fl1Od75: _c10ldlO=_c10ldlO.PARTIAL, _fII0d76: _cIIldll=_cIIldll.VOLUME_BASED) -> _cOO0d2d:
    _f11Odld = _c1lldl3(mode=_cIlOdOf.PAPER_TRADING, start_date=datetime.now(), end_date=datetime.now() + timedelta(days=365), initial_capital=_fO1ld74, fill_model=_fl1Od75, slippage_model=_fII0d76)
    return _cOO0d2d(_f11Odld)

def _fI0Id77(_fIlOd78: datetime, _f0l1d79: datetime, _fO1ld74: float=1000000) -> _c00Od42:
    _f11Odld = _c1lldl3(mode=_cIlOdOf.BACKTEST, start_date=_fIlOd78, end_date=_f0l1d79, initial_capital=_fO1ld74)
    return _c00Od42(_f11Odld)

def _fIl0d7A(_flOOd48: int=1000, _fO1ld74: float=1000000) -> _clIld47:
    _f11Odld = _c1lldl3(mode=_cIlOdOf.MONTE_CARLO, start_date=datetime.now(), end_date=datetime.now() + timedelta(days=252), initial_capital=_fO1ld74)
    return _clIld47(_f11Odld, _flOOd48)

def _fO10d7B(_fO1ld74: float=1000000) -> _cIlld57:
    _f11Odld = _c1lldl3(mode=_cIlOdOf.STRESS_TEST, start_date=datetime.now(), end_date=datetime.now() + timedelta(days=1), initial_capital=_fO1ld74)
    simulator = _cIlld57(_f11Odld)
    simulator._fl01d5A('2008_financial_crisis', '2008 Financial Crisis - Major bank failures', {'SPY': -0.5, 'XLF': -0.7, 'XLE': -0.4}, volatility_multiplier=4.0)
    simulator._fl01d5A('2020_covid_crash', 'March 2020 COVID-19 Market Crash', {'SPY': -0.35, 'XLK': -0.3, 'XLE': -0.6, 'XLF': -0.4}, volatility_multiplier=5.0)
    simulator._fl01d5A('flash_crash', 'May 2010 Flash Crash', {'SPY': -0.1, 'default': -0.15}, volatility_multiplier=10.0)
    simulator._fl01d5A('rate_shock', 'Interest Rate Shock - 200bp increase', {'TLT': -0.2, 'XLF': 0.05, 'XLU': -0.15, 'XLRE': -0.2}, volatility_multiplier=2.0)
    simulator._fl01d5A('tech_selloff', 'Technology Sector Selloff', {'XLK': -0.3, 'QQQ': -0.35, 'SPY': -0.15}, volatility_multiplier=2.5)
    return simulator

# Public API aliases for obfuscated classes
SimulationMode = _cIlOdOf
FillModel = _c10ldlO
SlippageModel = _cIIldll
MarketRegime = _c0Ildl2
SimulationConfig = _c1lldl3
SimulatedOrder = _cIlIdl4
SimulatedPosition = _c11Idl5
SimulatedFill = _cl11dl6
SimulatedBar = _cO10dl7
SimulationSnapshot = _cIOIdl9
SimulationResults = _c0lOdlA
SlippageCalculator = _cl0IdlB
FillSimulator = _cI0Id22
PaperTradingEngine = _cOO0d2d
BacktestEngine = _c00Od42
MonteCarloSimulator = _clIld47
StressTestSimulator = _cIlld57
StressScenario = _cl00d64
OrderBookSimulator = _c10Id65
create_paper_engine = _fIlOd73
create_backtest_engine = _fI0Id77
create_monte_carlo_simulator = _fIl0d7A
create_stress_tester = _fO10d7B
