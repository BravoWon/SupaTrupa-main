from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import math
from collections import defaultdict
from jones_framework.core import bridge, ComponentRegistry

class _c1lI76f(Enum):
    MAX_RETURN = 'max_return'
    MIN_VARIANCE = 'min_variance'
    MAX_SHARPE = 'max_sharpe'
    MAX_SORTINO = 'max_sortino'
    RISK_PARITY = 'risk_parity'
    MIN_TRACKING_ERROR = 'min_tracking_error'
    MAX_DIVERSIFICATION = 'max_diversification'
    MIN_CVaR = 'min_cvar'

class _cIlO77O(Enum):
    WEIGHT_BOUND = 'weight_bound'
    SECTOR_EXPOSURE = 'sector_exposure'
    COUNTRY_EXPOSURE = 'country_exposure'
    FACTOR_EXPOSURE = 'factor_exposure'
    TURNOVER = 'turnover'
    HOLDING_COUNT = 'holding_count'
    LONG_ONLY = 'long_only'
    LEVERAGE = 'leverage'
    TRACKING_ERROR = 'tracking_error'

class _cO1077l(Enum):
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    ANNUAL = 'annual'
    THRESHOLD = 'threshold'

@dataclass
class _c1O0772:
    symbol: str
    name: str = ''
    sector: str = ''
    country: str = ''
    asset_class: str = 'equity'
    expected_return: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    bid_ask_spread: float = 0.001
    commission_rate: float = 0.0001
    factor_exposures: Dict[str, float] = field(default_factory=dict)

@dataclass
class _cO10773:
    constraint_type: _cIlO77O
    name: str
    value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    scope: str = 'portfolio'
    scope_value: Optional[str] = None

@dataclass
class _c101774:
    weights: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 0.0
    max_drawdown: float = 0.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    country_exposure: Dict[str, float] = field(default_factory=dict)
    factor_exposure: Dict[str, float] = field(default_factory=dict)

    @property
    def _f00l775(self) -> int:
        return sum((1 for w in self.weights.values() if abs(w) > 1e-06))

    @property
    def _fOOI776(self) -> float:
        return sum(self.weights.values())

    @property
    def _f1l0777(self) -> float:
        return sum((w for w in self.weights.values() if w > 0))

    @property
    def _f10O778(self) -> float:
        return sum((abs(w) for w in self.weights.values() if w < 0))

    @property
    def _fIlO779(self) -> float:
        return sum((abs(w) for w in self.weights.values()))

    @property
    def _fO0I77A(self) -> float:
        return sum(self.weights.values())

@dataclass
class _cI0l77B:
    symbol: str
    side: str
    quantity: float
    current_weight: float
    target_weight: float
    estimated_cost: float = 0.0
    priority: int = 0

@dataclass
class _cO1O77c:
    current_weights: _c101774
    target_weights: _c101774
    trades: List[_cI0l77B]
    total_turnover: float = 0.0
    estimated_transaction_cost: float = 0.0
    estimated_tax_impact: float = 0.0
    expected_return_change: float = 0.0
    volatility_change: float = 0.0
    sharpe_change: float = 0.0

class _cl0177d:

    def __init__(self, _f01I77E: str='sample'):
        self._method = _f01I77E

    def _f1I177f(self, _fOlO78O: Dict[str, List[float]], _f1ll78l: float=0.0) -> Dict[Tuple[str, str], float]:
        symbols = list(_fOlO78O.keys())
        n = len(symbols)
        if not _fOlO78O or not _fOlO78O[symbols[0]]:
            return {}
        T = len(_fOlO78O[symbols[0]])
        means = {s: sum(r) / len(r) for s, r in _fOlO78O.items()}
        cov = {}
        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                if j < i:
                    cov[s1, s2] = cov[s2, s1]
                else:
                    cov_sum = sum(((_fOlO78O[s1][t] - means[s1]) * (_fOlO78O[s2][t] - means[s2]) for t in range(T)))
                    cov[s1, s2] = cov_sum / (T - 1)
        if _f1ll78l > 0:
            avg_var = sum((cov[s, s] for s in symbols)) / n
            for i, s1 in enumerate(symbols):
                for j, s2 in enumerate(symbols):
                    if i == j:
                        cov[s1, s2] = (1 - _f1ll78l) * cov[s1, s2] + _f1ll78l * avg_var
                    else:
                        cov[s1, s2] = (1 - _f1ll78l) * cov[s1, s2]
        return cov

class _cII0782:

    def __init__(self, _f01I77E: str='historical'):
        self._method = _f01I77E

    def _f1I177f(self, _fOlO78O: Dict[str, List[float]], _fIOO783: float=0.0) -> Dict[str, float]:
        if self._method == 'historical':
            return {s: sum(r) / len(r) * 252 if r else 0.0 for s, r in _fOlO78O.items()}
        elif self._method == 'capm':
            market_return = 0.08
            return {s: _fIOO783 + 1.0 * (market_return - _fIOO783) for s in _fOlO78O.keys()}
        else:
            return {s: 0.0 for s in _fOlO78O.keys()}

class _c1I0784(ABC):

    def __init__(self, _flO1785: List[_c1O0772]):
        self._assets = {a.symbol: a for a in _flO1785}
        self._constraints: List[_cO10773] = []

    def _f1Il786(self, _f11l787: _cO10773):
        self._constraints.append(_f11l787)

    @abstractmethod
    def _fOl1788(self, _f0Il789: Dict[str, float], _f01078A: Dict[Tuple[str, str], float], _fI0078B: Optional[Dict[str, float]]=None) -> _c101774:
        pass

    def _fOIl78c(self, _fl1O78d: Dict[str, float]) -> List[str]:
        violations = []
        for _f11l787 in self._constraints:
            if _f11l787.constraint_type == _cIlO77O.WEIGHT_BOUND:
                for symbol, weight in _fl1O78d.items():
                    asset = self._assets.get(symbol)
                    if asset:
                        if weight < asset.min_weight:
                            violations.append(f'{symbol} weight {weight:.4f} below min {asset.min_weight}')
                        if weight > asset.max_weight:
                            violations.append(f'{symbol} weight {weight:.4f} above max {asset.max_weight}')
            elif _f11l787.constraint_type == _cIlO77O.LONG_ONLY:
                for symbol, weight in _fl1O78d.items():
                    if weight < -1e-06:
                        violations.append(f'{symbol} has negative weight {weight:.4f}')
            elif _f11l787.constraint_type == _cIlO77O.LEVERAGE:
                gross = sum((abs(w) for w in _fl1O78d.values()))
                if _f11l787.max_value and gross > _f11l787.max_value:
                    violations.append(f'Leverage {gross:.2f} exceeds max {_f11l787.max_value}')
        return violations

    def _fllI78E(self, _fl1O78d: Dict[str, float], _f0Il789: Dict[str, float], _f01078A: Dict[Tuple[str, str], float], _fIOO783: float=0.0) -> Dict[str, float]:
        symbols = list(_fl1O78d.keys())
        port_return = sum((_fl1O78d[s] * _f0Il789.get(s, 0) for s in symbols))
        port_variance = 0.0
        for s1 in symbols:
            for s2 in symbols:
                w1 = _fl1O78d[s1]
                w2 = _fl1O78d[s2]
                cov = _f01078A.get((s1, s2), 0)
                port_variance += w1 * w2 * cov
        port_vol = math.sqrt(max(0, port_variance))
        sharpe = (port_return - _fIOO783) / port_vol if port_vol > 0 else 0
        weighted_vol = sum((abs(_fl1O78d[s]) * math.sqrt(_f01078A.get((s, s), 0)) for s in symbols))
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        return {'expected_return': port_return, 'volatility': port_vol, 'sharpe_ratio': sharpe, 'diversification_ratio': div_ratio}

class _c11078f(_c1I0784):

    def __init__(self, _flO1785: List[_c1O0772], _f1OI79O: _c1lI76f=_c1lI76f.MAX_SHARPE, _fI0O79l: float=1.0):
        super().__init__(_flO1785)
        self._objective = _f1OI79O
        self._risk_aversion = _fI0O79l

    @bridge(connects_to=['RiskEngine', 'FeatureStore', 'TradeCube'], connection_types={'RiskEngine': 'uses', 'FeatureStore': 'reads', 'TradeCube': 'feeds'})
    def _fOl1788(self, _f0Il789: Dict[str, float], _f01078A: Dict[Tuple[str, str], float], _fI0078B: Optional[Dict[str, float]]=None) -> _c101774:
        symbols = list(self._assets.keys())
        n = len(symbols)
        if n == 0:
            return _c101774(weights={})
        _fl1O78d = {s: 1.0 / n for s in symbols}
        lr = 0.01
        iterations = 1000
        for _ in range(iterations):
            gradients = self._calculate_gradients(_fl1O78d, _f0Il789, _f01078A)
            for s in symbols:
                _fl1O78d[s] += lr * gradients[s]
            _fl1O78d = self._project_to_constraints(_fl1O78d)
            total = sum(_fl1O78d.values())
            if total > 0:
                _fl1O78d = {s: w / total for s, w in _fl1O78d.items()}
        metrics = self._fllI78E(_fl1O78d, _f0Il789, _f01078A)
        sector_exposure = defaultdict(float)
        country_exposure = defaultdict(float)
        for s, w in _fl1O78d.items():
            asset = self._assets.get(s)
            if asset:
                sector_exposure[asset.sector] += w
                country_exposure[asset.country] += w
        return _c101774(weights=_fl1O78d, expected_return=metrics['expected_return'], volatility=metrics['volatility'], sharpe_ratio=metrics['sharpe_ratio'], diversification_ratio=metrics['diversification_ratio'], sector_exposure=dict(sector_exposure), country_exposure=dict(country_exposure))

    def _fl1l792(self, _fl1O78d: Dict[str, float], _f0Il789: Dict[str, float], _f01078A: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        symbols = list(_fl1O78d.keys())
        gradients = {}
        for s in symbols:
            if self._objective == _c1lI76f.MAX_RETURN:
                gradients[s] = _f0Il789.get(s, 0)
            elif self._objective == _c1lI76f.MIN_VARIANCE:
                grad = 0.0
                for s2 in symbols:
                    grad += 2 * _fl1O78d[s2] * _f01078A.get((s, s2), 0)
                gradients[s] = -grad
            elif self._objective == _c1lI76f.MAX_SHARPE:
                ret = _f0Il789.get(s, 0)
                var_grad = 0.0
                for s2 in symbols:
                    var_grad += 2 * _fl1O78d[s2] * _f01078A.get((s, s2), 0)
                gradients[s] = ret - self._risk_aversion * var_grad
            else:
                gradients[s] = _f0Il789.get(s, 0)
        return gradients

    def _fl0l793(self, _fl1O78d: Dict[str, float]) -> Dict[str, float]:
        projected = dict(_fl1O78d)
        long_only = any((c.constraint_type == _cIlO77O.LONG_ONLY for c in self._constraints))
        if long_only:
            projected = {s: max(0, w) for s, w in projected.items()}
        for s, w in projected.items():
            asset = self._assets.get(s)
            if asset:
                projected[s] = max(asset.min_weight, min(asset.max_weight, w))
        return projected

class _c11O794(_c1I0784):

    def __init__(self, _flO1785: List[_c1O0772]):
        super().__init__(_flO1785)

    @bridge(connects_to=['RiskEngine', 'FeatureStore'], connection_types={'RiskEngine': 'uses', 'FeatureStore': 'reads'})
    def _fOl1788(self, _f0Il789: Dict[str, float], _f01078A: Dict[Tuple[str, str], float], _fI0078B: Optional[Dict[str, float]]=None) -> _c101774:
        symbols = list(self._assets.keys())
        n = len(symbols)
        if n == 0:
            return _c101774(weights={})
        vols = {s: math.sqrt(_f01078A.get((s, s), 0.04)) for s in symbols}
        inv_vols = {s: 1 / v if v > 0 else 0 for s, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol > 0:
            _fl1O78d = {s: iv / total_inv_vol for s, iv in inv_vols.items()}
        else:
            _fl1O78d = {s: 1 / n for s in symbols}
        metrics = self._fllI78E(_fl1O78d, _f0Il789, _f01078A)
        return _c101774(weights=_fl1O78d, expected_return=metrics['expected_return'], volatility=metrics['volatility'], sharpe_ratio=metrics['sharpe_ratio'], diversification_ratio=metrics['diversification_ratio'])

class _cl1I795(_c1I0784):

    def __init__(self, _flO1785: List[_c1O0772], _f1O0796: Dict[str, float], _fI0O79l: float=2.5, _fIll797: float=0.05):
        super().__init__(_flO1785)
        self._market_weights = _f1O0796
        self._risk_aversion = _fI0O79l
        self._tau = _fIll797
        self._views: List[Tuple[Dict[str, float], float, float]] = []

    def _fI10798(self, _fOI0799: Dict[str, float], _f0I079A: float, _fI0079B: float):
        self._views.append((_fOI0799, _f0I079A, _fI0079B))

    @bridge(connects_to=['RiskEngine', 'FeatureStore', 'ResearchReportProcessor'], connection_types={'RiskEngine': 'uses', 'FeatureStore': 'reads', 'ResearchReportProcessor': 'incorporates'})
    def _fOl1788(self, _f0Il789: Dict[str, float], _f01078A: Dict[Tuple[str, str], float], _fI0078B: Optional[Dict[str, float]]=None) -> _c101774:
        symbols = list(self._assets.keys())
        n = len(symbols)
        if n == 0:
            return _c101774(weights={})
        implied_returns = {}
        for s in symbols:
            mkt_w = self._market_weights.get(s, 1 / n)
            cov_sum = sum((self._market_weights.get(s2, 1 / n) * _f01078A.get((s, s2), 0) for s2 in symbols))
            implied_returns[s] = self._risk_aversion * cov_sum
        if not self._views:
            bl_returns = implied_returns
        else:
            bl_returns = dict(implied_returns)
            for _fOI0799, view_return, _fI0079B in self._views:
                for s, vw in _fOI0799.items():
                    if s in bl_returns:
                        blend = _fI0079B
                        bl_returns[s] = (1 - blend) * bl_returns[s] + blend * view_return * vw
        mv_optimizer = _c11078f(list(self._assets.values()), _c1lI76f.MAX_SHARPE, self._risk_aversion)
        mv_optimizer._constraints = self._constraints
        return mv_optimizer._fOl1788(bl_returns, _f01078A)

class _c01O79c:

    def __init__(self, _fl0O79d: _c1I0784, _f0I179E: _cO1077l=_cO1077l.MONTHLY, _f00I79f: float=0.05):
        self._optimizer = _fl0O79d
        self._frequency = _f0I179E
        self._threshold = _f00I79f
        self._last_rebalance: Optional[datetime] = None

    @bridge(connects_to=['OrderManager', 'RiskEngine', 'TradeCube'], connection_types={'OrderManager': 'feeds', 'RiskEngine': 'validates', 'TradeCube': 'records'})
    def _f0l17AO(self, _fI0078B: Dict[str, float], _fOOI7Al: Dict[str, float]) -> bool:
        max_drift = 0.0
        for symbol in set(_fI0078B.keys()) | set(_fOOI7Al.keys()):
            current = _fI0078B.get(symbol, 0)
            target = _fOOI7Al.get(symbol, 0)
            drift = abs(current - target)
            max_drift = max(max_drift, drift)
        if max_drift > self._threshold:
            return True
        if self._last_rebalance is None:
            return True
        elapsed = datetime.now() - self._last_rebalance
        if self._frequency == _cO1077l.DAILY:
            return elapsed >= timedelta(days=1)
        elif self._frequency == _cO1077l.WEEKLY:
            return elapsed >= timedelta(weeks=1)
        elif self._frequency == _cO1077l.MONTHLY:
            return elapsed >= timedelta(days=30)
        elif self._frequency == _cO1077l.QUARTERLY:
            return elapsed >= timedelta(days=90)
        elif self._frequency == _cO1077l.ANNUAL:
            return elapsed >= timedelta(days=365)
        return False

    def _fOIO7A2(self, _fI0078B: Dict[str, float], _fOOI7Al: Dict[str, float], _f1007A3: float, _f1Ol7A4: Dict[str, float]) -> List[_cI0l77B]:
        trades = []
        all_symbols = set(_fI0078B.keys()) | set(_fOOI7Al.keys())
        for symbol in all_symbols:
            current = _fI0078B.get(symbol, 0)
            target = _fOOI7Al.get(symbol, 0)
            delta = target - current
            if abs(delta) < 0.001:
                continue
            price = _f1Ol7A4.get(symbol, 1.0)
            quantity = delta * _f1007A3 / price
            side = 'buy' if delta > 0 else 'sell'
            asset = self._optimizer._assets.get(symbol)
            if asset:
                cost = abs(quantity * price) * (asset.bid_ask_spread + asset.commission_rate)
            else:
                cost = abs(quantity * price) * 0.001
            trades.append(_cI0l77B(symbol=symbol, side=side, quantity=abs(quantity), current_weight=current, target_weight=target, estimated_cost=cost, priority=1 if abs(delta) > 0.05 else 0))
        trades.sort(key=lambda t: t.priority, reverse=True)
        return trades

    def _f1017A5(self, _fI0078B: Dict[str, float], _f0Il789: Dict[str, float], _f01078A: Dict[Tuple[str, str], float], _f1007A3: float, _f1Ol7A4: Dict[str, float]) -> _cO1O77c:
        target = self._optimizer._fOl1788(_f0Il789, _f01078A, _fI0078B)
        trades = self._fOIO7A2(_fI0078B, target._fl1O78d, _f1007A3, _f1Ol7A4)
        total_turnover = sum((abs(t.target_weight - t.current_weight) for t in trades)) / 2
        total_cost = sum((t.estimated_cost for t in trades))
        current = _c101774(weights=_fI0078B)
        self._last_rebalance = datetime.now()
        return _cO1O77c(current_weights=current, target_weights=target, trades=trades, total_turnover=total_turnover, estimated_transaction_cost=total_cost, expected_return_change=target._f0I079A - current._f0I079A, volatility_change=target.volatility - current.volatility)

class _clOI7A6:

    def __init__(self, _f1O17A7: float=0.35, _f0ll7A8: float=0.15, _f1107A9: int=365):
        self._short_term_rate = _f1O17A7
        self._long_term_rate = _f0ll7A8
        self._holding_period = timedelta(days=_f1107A9)
        self._wash_sale_period = timedelta(days=30)

    @bridge(connects_to=['OrderManager', 'TradeCube'], connection_types={'OrderManager': 'feeds', 'TradeCube': 'records'})
    def _fI007AA(self, _flOl7AB: Dict[str, Tuple[float, float, datetime]], _f1Ol7A4: Dict[str, float], _f1O07Ac: Dict[str, str]) -> List[_cI0l77B]:
        trades = []
        now = datetime.now()
        for symbol, (quantity, cost_basis, purchase_date) in _flOl7AB.items():
            if quantity <= 0:
                continue
            price = _f1Ol7A4.get(symbol, cost_basis)
            gain_loss = (price - cost_basis) * quantity
            if gain_loss >= 0:
                continue
            holding_time = now - purchase_date
            if holding_time > self._holding_period:
                tax_rate = self._long_term_rate
            else:
                tax_rate = self._short_term_rate
            tax_benefit = abs(gain_loss) * tax_rate
            replacement = _f1O07Ac.get(symbol)
            if tax_benefit > 100:
                trades.append(_cI0l77B(symbol=symbol, side='sell', quantity=quantity, current_weight=0, target_weight=0, estimated_cost=0, priority=int(tax_benefit)))
                if replacement and replacement in _f1Ol7A4:
                    trades.append(_cI0l77B(symbol=replacement, side='buy', quantity=quantity * price / _f1Ol7A4[replacement], current_weight=0, target_weight=0, estimated_cost=0, priority=int(tax_benefit)))
        return trades

def _fOO07Ad(_flO1785: List[_c1O0772], _f1OI79O: _c1lI76f=_c1lI76f.MAX_SHARPE) -> _c11078f:
    return _c11078f(_flO1785, _f1OI79O)

def _fOI07AE(_flO1785: List[_c1O0772]) -> _c11O794:
    return _c11O794(_flO1785)

def _fII07Af(_flO1785: List[_c1O0772], _f1O0796: Dict[str, float]) -> _cl1I795:
    return _cl1I795(_flO1785, _f1O0796)

def _fO1O7BO(_fl0O79d: _c1I0784, _f0I179E: _cO1077l=_cO1077l.MONTHLY) -> _c01O79c:
    return _c01O79c(_fl0O79d, _f0I179E)

# Public API aliases for obfuscated classes
PortfolioOptimizer = _c1I0784
MeanVarianceOptimizer = _c11078f
RiskParityOptimizer = _c11O794
Rebalancer = _c01O79c
