from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import math
from collections import defaultdict
import random
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.trading.execution.order_manager import Position, Order, OrderSide

class _cl0I37E(Enum):
    VAR = 'var'
    EXPECTED_SHORTFALL = 'expected_shortfall'
    VOLATILITY = 'volatility'
    BETA = 'beta'
    SHARPE = 'sharpe'
    SORTINO = 'sortino'
    MAX_DRAWDOWN = 'max_drawdown'
    CORRELATION = 'correlation'
    CONCENTRATION = 'concentration'
    LEVERAGE = 'leverage'
    LIQUIDITY = 'liquidity'

class _c1lI37f(Enum):
    HISTORICAL = 'historical'
    PARAMETRIC = 'parametric'
    MONTE_CARLO = 'monte_carlo'

class _cO1038O(Enum):
    POSITION_SIZE = 'position_size'
    POSITION_VALUE = 'position_value'
    SECTOR_EXPOSURE = 'sector_exposure'
    COUNTRY_EXPOSURE = 'country_exposure'
    SINGLE_NAME = 'single_name'
    LEVERAGE = 'leverage'
    VAR = 'var'
    DRAWDOWN = 'drawdown'
    LOSS = 'loss'
    CONCENTRATION = 'concentration'

class _c0II38l(Enum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4

@dataclass
class _c0O1382:
    limit_type: _cO1038O
    name: str
    limit_value: float
    warning_threshold: float = 0.8
    hard_limit: bool = True
    scope: str = 'portfolio'
    scope_value: Optional[str] = None

@dataclass
class _c010383:
    limit: _c0O1382
    current_value: float
    breach_time: datetime
    severity: _c0II38l
    details: str = ''

    @property
    def _fO11384(self) -> float:
        return self.current_value / self.limit.limit_value if self.limit.limit_value else 0

@dataclass
class _clll385:
    alert_id: str
    severity: _c0II38l
    metric: _cl0I37E
    message: str
    timestamp: datetime
    acknowledged: bool = False
    related_positions: List[str] = field(default_factory=list)
    recommended_action: str = ''

@dataclass
class _cIO0386:
    symbol: str
    quantity: int
    market_value: float
    weight: float
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0
    marginal_var: float = 0.0
    component_var: float = 0.0
    incremental_var: float = 0.0
    avg_daily_volume: float = 0.0
    days_to_liquidate: float = 0.0
    liquidity_score: float = 1.0

@dataclass
class _cIl0387:
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    hhi: float = 0.0
    top_5_concentration: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    position_risks: List[_cIO0386] = field(default_factory=list)

@dataclass
class _cI1O388:
    name: str
    description: str
    market_shocks: Dict[str, float]
    correlation_shock: float = 0.0
    volatility_shock: float = 0.0
    probability: float = 0.0

@dataclass
class _cOOI389:
    scenario: _cI1O388
    portfolio_loss: float
    portfolio_loss_pct: float
    position_losses: Dict[str, float]
    breached_limits: List[_c0O1382]
    timestamp: datetime

class _cO1l38A:

    def __init__(self, _fI0O38B: List[float]=None):
        self._confidence_levels = _fI0O38B or [0.95, 0.99]
        self._historical_returns: Dict[str, List[float]] = {}

    def _f0O138c(self, _fOO038d: str, _f10138E: List[float]):
        self._historical_returns[_fOO038d] = _f10138E

    @bridge(connects_to=['PortfolioRisk', 'PositionRisk', 'RiskEngine'], connection_types={'PortfolioRisk': 'calculates', 'PositionRisk': 'calculates', 'RiskEngine': 'provides'})
    def _f0II38f(self, _fOO038d: str, _fO1039O: float, _fOl039l: _c1lI37f=_c1lI37f.HISTORICAL, _f1ll392: float=0.95, _f1O0393: int=1) -> float:
        _f10138E = self._historical_returns.get(_fOO038d, [])
        if _fOl039l == _c1lI37f.HISTORICAL:
            return self._historical_var(_f10138E, _fO1039O, _f1ll392, _f1O0393)
        elif _fOl039l == _c1lI37f.PARAMETRIC:
            return self._parametric_var(_f10138E, _fO1039O, _f1ll392, _f1O0393)
        else:
            return self._monte_carlo_var(_f10138E, _fO1039O, _f1ll392, _f1O0393)

    def _f1ll394(self, _f10138E: List[float], _fO1039O: float, _f1ll392: float, _f1O0393: int) -> float:
        if not _f10138E:
            return _fO1039O * 0.05
        sorted_returns = sorted(_f10138E)
        index = int((1 - _f1ll392) * len(sorted_returns))
        percentile_return = sorted_returns[index]
        scaled_return = percentile_return * math.sqrt(_f1O0393)
        return abs(_fO1039O * scaled_return)

    def _f1I0395(self, _f10138E: List[float], _fO1039O: float, _f1ll392: float, _f1O0393: int) -> float:
        if not _f10138E or len(_f10138E) < 2:
            return _fO1039O * 0.05
        mean = sum(_f10138E) / len(_f10138E)
        variance = sum(((r - mean) ** 2 for r in _f10138E)) / (len(_f10138E) - 1)
        std = math.sqrt(variance)
        z_scores = {0.9: 1.28, 0.95: 1.645, 0.99: 2.33}
        z = z_scores.get(_f1ll392, 1.645)
        var = _fO1039O * (z * std * math.sqrt(_f1O0393) - mean * _f1O0393)
        return abs(var)

    def _flll396(self, _f10138E: List[float], _fO1039O: float, _f1ll392: float, _f1O0393: int, _f1I0397: int=10000) -> float:
        if not _f10138E or len(_f10138E) < 2:
            return _fO1039O * 0.05
        mean = sum(_f10138E) / len(_f10138E)
        variance = sum(((r - mean) ** 2 for r in _f10138E)) / (len(_f10138E) - 1)
        std = math.sqrt(variance)
        simulated_losses = []
        for _ in range(_f1I0397):
            total_return = 0
            for _ in range(_f1O0393):
                daily_return = random.gauss(mean, std)
                total_return += daily_return
            simulated_losses.append(-_fO1039O * total_return)
        sorted_losses = sorted(simulated_losses)
        index = int(_f1ll392 * len(sorted_losses))
        return max(0, sorted_losses[index])

    def _f0I1398(self, _fOO038d: str, _fO1039O: float, _f1ll392: float=0.95, _f1O0393: int=1) -> float:
        _f10138E = self._historical_returns.get(_fOO038d, [])
        if not _f10138E:
            return _fO1039O * 0.07
        sorted_returns = sorted(_f10138E)
        cutoff_index = int((1 - _f1ll392) * len(sorted_returns))
        tail_returns = sorted_returns[:max(1, cutoff_index)]
        avg_tail_return = sum(tail_returns) / len(tail_returns)
        scaled_return = avg_tail_return * math.sqrt(_f1O0393)
        return abs(_fO1039O * scaled_return)

class _cIlO399:

    def __init__(self):
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}
        self._position_var: _cO1l38A = _cO1l38A()

    def _fOlI39A(self, _f0lO39B: str, _f11I39c: str, _f00l39d: float):
        self._correlation_matrix[_f0lO39B, _f11I39c] = _f00l39d
        self._correlation_matrix[_f11I39c, _f0lO39B] = _f00l39d

    def _fIOO39E(self, _f0lO39B: str, _f11I39c: str) -> float:
        if _f0lO39B == _f11I39c:
            return 1.0
        return self._correlation_matrix.get((_f0lO39B, _f11I39c), 0.3)

    @bridge(connects_to=['VaRCalculator', 'RiskEngine'], connection_types={'VaRCalculator': 'uses', 'RiskEngine': 'provides'})
    def _f0Ol39f(self, _fIII3AO: Dict[str, float], _f1ll392: float=0.95, _f1O0393: int=1) -> float:
        if not _fIII3AO:
            return 0.0
        symbols = list(_fIII3AO.keys())
        n = len(symbols)
        individual_vars = {}
        for _fOO038d, value in _fIII3AO.items():
            individual_vars[_fOO038d] = self._position_var._f0II38f(_fOO038d, value, _c1lI37f.HISTORICAL, _f1ll392, _f1O0393)
        corr_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(self._fIOO39E(symbols[i], symbols[j]))
            corr_matrix.append(row)
        portfolio_var_sq = 0.0
        for i in range(n):
            for j in range(n):
                var_i = individual_vars[symbols[i]]
                var_j = individual_vars[symbols[j]]
                rho = corr_matrix[i][j]
                portfolio_var_sq += var_i * var_j * rho
        return math.sqrt(max(0, portfolio_var_sq))

    def _f1103Al(self, _fIII3AO: Dict[str, float], _fOO038d: str, _f1ll392: float=0.95) -> float:
        var_with = self._f0Ol39f(_fIII3AO, _f1ll392)
        positions_without = {s: v for s, v in _fIII3AO.items() if s != _fOO038d}
        var_without = self._f0Ol39f(positions_without, _f1ll392)
        return var_with - var_without

    def _fllO3A2(self, _fIII3AO: Dict[str, float], _fOO038d: str, _f1ll392: float=0.95) -> float:
        total_var = self._f0Ol39f(_fIII3AO, _f1ll392)
        marginal_var = self._f1103Al(_fIII3AO, _fOO038d, _f1ll392)
        _fO1039O = _fIII3AO.get(_fOO038d, 0)
        total_value = sum(_fIII3AO.values())
        if total_value == 0:
            return 0
        weight = _fO1039O / total_value
        return marginal_var * weight * total_value / total_var if total_var > 0 else 0

class _c0l13A3:

    def __init__(self):
        self._scenarios: List[_cI1O388] = []
        self._register_default_scenarios()

    def _fOII3A4(self):
        self._scenarios.append(_cI1O388(name='2008 Financial Crisis', description='Market crash similar to 2008', market_shocks={'SPY': -0.4, 'QQQ': -0.45, 'IWM': -0.5, 'XLF': -0.6, 'XLE': -0.35, 'GLD': 0.05, 'TLT': 0.2}, correlation_shock=0.3, volatility_shock=2.0, probability=0.02))
        self._scenarios.append(_cI1O388(name='Flash Crash', description='Sudden market drop similar to 2010', market_shocks={'SPY': -0.1, 'QQQ': -0.12, 'IWM': -0.15}, correlation_shock=0.5, volatility_shock=3.0, probability=0.05))
        self._scenarios.append(_cI1O388(name='Interest Rate Shock', description='Sudden interest rate increase', market_shocks={'SPY': -0.15, 'QQQ': -0.2, 'XLF': -0.1, 'TLT': -0.25, 'HYG': -0.15}, volatility_shock=1.5, probability=0.1))
        self._scenarios.append(_cI1O388(name='Tech Selloff', description='Technology sector correction', market_shocks={'QQQ': -0.3, 'XLK': -0.35, 'SPY': -0.15, 'XLF': -0.05}, volatility_shock=1.8, probability=0.15))

    def _fO113A5(self, _fO1I3A6: _cI1O388):
        self._scenarios.append(_fO1I3A6)

    @bridge(connects_to=['RiskEngine', 'PortfolioRisk'], connection_types={'RiskEngine': 'provides', 'PortfolioRisk': 'tests'})
    def _fO0l3A7(self, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float], _fO1I3A6: _cI1O388, _f10l3A9: List[_c0O1382]) -> _cOOI389:
        position_losses = {}
        total_loss = 0.0
        for _fOO038d, position in _fIII3AO.items():
            price = _f0103A8.get(_fOO038d, position.avg_price)
            value = position.quantity * price
            shock = _fO1I3A6.market_shocks.get(_fOO038d, -0.1)
            loss = value * abs(shock)
            position_losses[_fOO038d] = loss
            total_loss += loss
        total_value = sum((pos.quantity * _f0103A8.get(sym, pos.avg_price) for sym, pos in _fIII3AO.items()))
        loss_pct = total_loss / total_value if total_value > 0 else 0
        breached = []
        for limit in _f10l3A9:
            if limit.limit_type == _cO1038O.LOSS:
                if loss_pct > limit.limit_value:
                    breached.append(limit)
            elif limit.limit_type == _cO1038O.VAR:
                if total_loss > limit.limit_value:
                    breached.append(limit)
        return _cOOI389(scenario=_fO1I3A6, portfolio_loss=total_loss, portfolio_loss_pct=loss_pct, position_losses=position_losses, breached_limits=breached, timestamp=datetime.now())

    def _fIlO3AA(self, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float], _f10l3A9: List[_c0O1382]) -> List[_cOOI389]:
        results = []
        for _fO1I3A6 in self._scenarios:
            result = self._fO0l3A7(_fIII3AO, _f0103A8, _fO1I3A6, _f10l3A9)
            results.append(result)
        return results

    def _f1I13AB(self, _fIIO3Ac: List[_cOOI389]) -> float:
        total = 0.0
        for result in _fIIO3Ac:
            total += result.portfolio_loss * result._fO1I3A6.probability
        return total

class _cOl03Ad:

    def __init__(self):
        self._limits: List[_c0O1382] = []
        self._breaches: List[_c010383] = []
        self._alerts: List[_clll385] = []

    def _fI0I3AE(self, _f1IO3Af: _c0O1382):
        self._limits.append(_f1IO3Af)

    def _fOO03BO(self, _f11I3Bl: str):
        self._limits = [l for l in self._limits if l.name != _f11I3Bl]

    @bridge(connects_to=['RiskEngine', 'OrderManager'], connection_types={'RiskEngine': 'provides', 'OrderManager': 'validates'})
    def _f1IO3B2(self, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float], _fllO3B3: Dict[str, str]=None) -> List[_c010383]:
        breaches = []
        for _f1IO3Af in self._limits:
            current_value = self._calculate_limit_value(_f1IO3Af, _fIII3AO, _f0103A8, _fllO3B3)
            if current_value is None:
                continue
            if current_value >= _f1IO3Af.limit_value:
                breach = _c010383(limit=_f1IO3Af, current_value=current_value, breach_time=datetime.now(), severity=_c0II38l.CRITICAL if _f1IO3Af.hard_limit else _c0II38l.WARNING, details=f'{_f1IO3Af.name}: {current_value:.2f} >= {_f1IO3Af.limit_value:.2f}')
                breaches.append(breach)
                self._breaches.append(breach)
            elif current_value >= _f1IO3Af.warning_threshold * _f1IO3Af.limit_value:
                breach = _c010383(limit=_f1IO3Af, current_value=current_value, breach_time=datetime.now(), severity=_c0II38l.WARNING, details=f'{_f1IO3Af.name}: {current_value:.2f} approaching {_f1IO3Af.limit_value:.2f}')
                breaches.append(breach)
        return breaches

    def _f0IO3B4(self, _f1IO3Af: _c0O1382, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float], _fllO3B3: Dict[str, str]=None) -> Optional[float]:
        total_value = sum((pos.quantity * _f0103A8.get(sym, pos.avg_price) for sym, pos in _fIII3AO.items()))
        if _f1IO3Af.limit_type == _cO1038O.POSITION_SIZE:
            if _f1IO3Af.scope_value and _f1IO3Af.scope_value in _fIII3AO:
                pos = _fIII3AO[_f1IO3Af.scope_value]
                return abs(pos.quantity)
            return None
        elif _f1IO3Af.limit_type == _cO1038O.POSITION_VALUE:
            if _f1IO3Af.scope_value and _f1IO3Af.scope_value in _fIII3AO:
                pos = _fIII3AO[_f1IO3Af.scope_value]
                price = _f0103A8.get(_f1IO3Af.scope_value, pos.avg_price)
                return abs(pos.quantity * price)
            return None
        elif _f1IO3Af.limit_type == _cO1038O.SINGLE_NAME:
            if total_value == 0:
                return 0
            max_weight = 0
            for sym, pos in _fIII3AO.items():
                price = _f0103A8.get(sym, pos.avg_price)
                weight = abs(pos.quantity * price) / total_value
                max_weight = max(max_weight, weight)
            return max_weight
        elif _f1IO3Af.limit_type == _cO1038O.SECTOR_EXPOSURE:
            if not _fllO3B3:
                return None
            sector_value = 0
            for sym, pos in _fIII3AO.items():
                if _fllO3B3.get(sym) == _f1IO3Af.scope_value:
                    price = _f0103A8.get(sym, pos.avg_price)
                    sector_value += abs(pos.quantity * price)
            return sector_value / total_value if total_value > 0 else 0
        elif _f1IO3Af.limit_type == _cO1038O.LEVERAGE:
            gross_value = sum((abs(pos.quantity) * _f0103A8.get(sym, pos.avg_price) for sym, pos in _fIII3AO.items()))
            return gross_value / total_value if total_value > 0 else 0
        elif _f1IO3Af.limit_type == _cO1038O.CONCENTRATION:
            if total_value == 0:
                return 0
            hhi = 0
            for sym, pos in _fIII3AO.items():
                price = _f0103A8.get(sym, pos.avg_price)
                weight = abs(pos.quantity * price) / total_value
                hhi += weight ** 2
            return hhi
        return None

    def _fOOO3B5(self, _fIOO3B6: Order, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float], _fI1I3B7: float) -> Tuple[bool, List[str]]:
        reasons = []
        new_positions = dict(_fIII3AO)
        if _fIOO3B6._fOO038d in new_positions:
            current = new_positions[_fIOO3B6._fOO038d]
            if _fIOO3B6.side == OrderSide.BUY:
                new_qty = current.quantity + _fIOO3B6.quantity
            else:
                new_qty = current.quantity - _fIOO3B6.quantity
            new_positions[_fIOO3B6._fOO038d] = Position(symbol=_fIOO3B6._fOO038d, quantity=new_qty, avg_price=current.avg_price, unrealized_pnl=0, realized_pnl=0)
        else:
            qty = _fIOO3B6.quantity if _fIOO3B6.side == OrderSide.BUY else -_fIOO3B6.quantity
            new_positions[_fIOO3B6._fOO038d] = Position(symbol=_fIOO3B6._fOO038d, quantity=qty, avg_price=_f0103A8.get(_fIOO3B6._fOO038d, 0), unrealized_pnl=0, realized_pnl=0)
        breaches = self._f1IO3B2(new_positions, _f0103A8)
        for breach in breaches:
            if breach._f1IO3Af.hard_limit and breach.severity == _c0II38l.CRITICAL:
                reasons.append(breach.details)
        return (len(reasons) == 0, reasons)

    @property
    def _flIO3B8(self) -> List[_c010383]:
        return [b for b in self._breaches if b.severity == _c0II38l.CRITICAL]

class _c1I03B9:

    def __init__(self):
        self._var_calculator = _cO1l38A()
        self._portfolio_var = _cIlO399()
        self._stress_engine = _c0l13A3()
        self._limit_manager = _cOl03Ad()
        self._historical_risk: List[_cIl0387] = []
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['OrderManager', 'FeatureStore', 'TradeCube', 'JonesEngine'], connection_types={'OrderManager': 'validates', 'FeatureStore': 'reads', 'TradeCube': 'feeds', 'JonesEngine': 'integrates'})
    def _fI0I3AE(self, _f1IO3Af: _c0O1382):
        self._limit_manager._fI0I3AE(_f1IO3Af)

    def _f0O138c(self, _fOO038d: str, _f10138E: List[float]):
        self._var_calculator._f0O138c(_fOO038d, _f10138E)

    def _fOlI39A(self, _f0lO39B: str, _f11I39c: str, _f00l39d: float):
        self._portfolio_var._fOlI39A(_f0lO39B, _f11I39c, _f00l39d)

    def _fl113BA(self, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float], _fI1I3B7: float, _f0l13BB: float=0.0) -> _cIl0387:
        position_values = {sym: pos.quantity * _f0103A8.get(sym, pos.avg_price) for sym, pos in _fIII3AO.items()}
        positions_value = sum(position_values.values())
        total_value = _fI1I3B7 + positions_value
        var_95 = self._portfolio_var._f0Ol39f(position_values, 0.95)
        var_99 = self._portfolio_var._f0Ol39f(position_values, 0.99)
        position_risks = []
        for _fOO038d, pos in _fIII3AO.items():
            value = position_values[_fOO038d]
            weight = value / total_value if total_value > 0 else 0
            pos_var_95 = self._var_calculator._f0II38f(_fOO038d, value, _c1lI37f.HISTORICAL, 0.95)
            pos_var_99 = self._var_calculator._f0II38f(_fOO038d, value, _c1lI37f.HISTORICAL, 0.99)
            es = self._var_calculator._f0I1398(_fOO038d, value)
            position_risks.append(_cIO0386(symbol=_fOO038d, quantity=pos.quantity, market_value=value, weight=weight, var_95=pos_var_95, var_99=pos_var_99, expected_shortfall=es, marginal_var=self._portfolio_var._f1103Al(position_values, _fOO038d), component_var=self._portfolio_var._fllO3A2(position_values, _fOO038d)))
        gross_exposure = sum((abs(v) for v in position_values.values()))
        net_exposure = sum(position_values.values())
        leverage = gross_exposure / total_value if total_value > 0 else 0
        weights = [abs(v) / total_value for v in position_values.values()] if total_value > 0 else []
        hhi = sum((w ** 2 for w in weights))
        top_5 = sum(sorted(weights, reverse=True)[:5])
        es_95 = sum((pr.expected_shortfall for pr in position_risks)) * 0.8
        risk = _cIl0387(timestamp=datetime.now(), total_value=total_value, cash=_fI1I3B7, positions_value=positions_value, var_95=var_95, var_99=var_99, expected_shortfall_95=es_95, gross_exposure=gross_exposure, net_exposure=net_exposure, leverage=leverage, hhi=hhi, top_5_concentration=top_5, position_risks=position_risks)
        self._historical_risk.append(risk)
        return risk

    def _f1IO3B2(self, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float]) -> List[_c010383]:
        return self._limit_manager._f1IO3B2(_fIII3AO, _f0103A8)

    def _fOOO3B5(self, _fIOO3B6: Order, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float], _fI1I3B7: float) -> Tuple[bool, List[str]]:
        return self._limit_manager._fOOO3B5(_fIOO3B6, _fIII3AO, _f0103A8, _fI1I3B7)

    def _fI0I3Bc(self, _fIII3AO: Dict[str, Position], _f0103A8: Dict[str, float]) -> List[_cOOI389]:
        return self._stress_engine._fIlO3AA(_fIII3AO, _f0103A8, self._limit_manager._limits)

    def _fOll3Bd(self, _fO1I3A6: _cI1O388):
        self._stress_engine._fO113A5(_fO1I3A6)

    @property
    def _fll03BE(self) -> Dict[str, Any]:
        if not self._historical_risk:
            return {}
        latest = self._historical_risk[-1]
        return {'timestamp': latest.timestamp, 'total_value': latest.total_value, 'var_95': latest.var_95, 'var_99': latest.var_99, 'leverage': latest.leverage, 'concentration_hhi': latest.hhi, 'active_breaches': len(self._limit_manager._flIO3B8)}

def _f00l3Bf() -> _c1I03B9:
    engine = _c1I03B9()
    engine._fI0I3AE(_c0O1382(limit_type=_cO1038O.SINGLE_NAME, name='Single Name Limit', limit_value=0.2, warning_threshold=0.8))
    engine._fI0I3AE(_c0O1382(limit_type=_cO1038O.LEVERAGE, name='Leverage Limit', limit_value=2.0, warning_threshold=0.8))
    engine._fI0I3AE(_c0O1382(limit_type=_cO1038O.CONCENTRATION, name='Concentration Limit', limit_value=0.3, warning_threshold=0.8))
    return engine

def _fl1I3cO() -> _cO1l38A:
    return _cO1l38A()

# Public API aliases for obfuscated classes
RiskEngine = _c1I03B9
RiskMetrics = _cIl0387
