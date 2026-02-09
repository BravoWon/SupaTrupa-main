from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Iterator
from enum import Enum, auto
from datetime import datetime, timedelta
import math
from collections import defaultdict
from abc import ABC, abstractmethod
from jones_framework.core import bridge, ComponentRegistry
from jones_framework.engine.core import Timeframe

class _cIOldf(Enum):
    TIMEFRAME = 'timeframe'
    ASSET = 'asset'
    ASSET_CLASS = 'asset_class'
    SECTOR = 'sector'
    REGIME = 'regime'
    VOLATILITY = 'volatility'
    LIQUIDITY = 'liquidity'
    CORRELATION = 'correlation'
    SENTIMENT = 'sentiment'
    FUNDAMENTAL = 'fundamental'
    TECHNICAL = 'technical'
    RISK = 'risk'
    RETURN = 'return'
    CUSTOM = 'custom'

class _cl1lEO(Enum):
    SUM = 'sum'
    MEAN = 'mean'
    MEDIAN = 'median'
    MIN = 'min'
    MAX = 'max'
    FIRST = 'first'
    LAST = 'last'
    STD = 'std'
    VAR = 'var'
    COUNT = 'count'
    WEIGHTED_MEAN = 'weighted_mean'

class _cl1lEl(Enum):
    EQ = 'eq'
    NE = 'ne'
    GT = 'gt'
    GE = 'ge'
    LT = 'lt'
    LE = 'le'
    IN = 'in'
    NOT_IN = 'not_in'
    BETWEEN = 'between'
    LIKE = 'like'

@dataclass
class _cIOOE2:
    dimension: _cIOldf
    value: Any
    label: str = ''
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label:
            self.label = str(self.value)

    def __hash__(self):
        return hash((self.dimension, self.value))

    def __eq__(self, _fOl0E3):
        if not isinstance(_fOl0E3, _cIOOE2):
            return False
        return self.dimension == _fOl0E3.dimension and self.value == _fOl0E3.value

@dataclass
class _cOIOE4:
    coordinates: Tuple[_cIOOE2, ...]
    metrics: Dict[str, float]
    count: int = 1
    weight: float = 1.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def _fl11E5(self) -> float:
        if 'pnl' in self.metrics:
            return self.metrics['pnl']
        elif 'return' in self.metrics:
            return self.metrics['return']
        return next(iter(self.metrics.values())) if self.metrics else 0.0

    def _f01OE6(self, _fOl0E3: 'TradeCell', _fO1lE7: _cl1lEO=_cl1lEO.SUM) -> 'TradeCell':
        merged_metrics = {}
        all_keys = set(self.metrics.keys()) | set(_fOl0E3.metrics.keys())
        for key in all_keys:
            v1 = self.metrics.get(key, 0)
            v2 = _fOl0E3.metrics.get(key, 0)
            if _fO1lE7 == _cl1lEO.SUM:
                merged_metrics[key] = v1 + v2
            elif _fO1lE7 == _cl1lEO.MEAN:
                merged_metrics[key] = (v1 + v2) / 2
            elif _fO1lE7 == _cl1lEO.MIN:
                merged_metrics[key] = min(v1, v2)
            elif _fO1lE7 == _cl1lEO.MAX:
                merged_metrics[key] = max(v1, v2)
            elif _fO1lE7 == _cl1lEO.LAST:
                merged_metrics[key] = v2
        return _cOIOE4(coordinates=self.coordinates, metrics=merged_metrics, count=self.count + _fOl0E3.count, weight=self.weight + _fOl0E3.weight, timestamp=_fOl0E3.timestamp or self.timestamp)

@dataclass
class _c0lIE8:
    dimension: _cIOldf
    operator: _cl1lEl
    value: Any
    secondary_value: Any = None

    def _fl0OE9(self, _f110EA: _cIOOE2) -> bool:
        if _f110EA.dimension != self.dimension:
            return True
        v = _f110EA.value
        if self.operator == _cl1lEl.EQ:
            return v == self.value
        elif self.operator == _cl1lEl.NE:
            return v != self.value
        elif self.operator == _cl1lEl.GT:
            return v > self.value
        elif self.operator == _cl1lEl.GE:
            return v >= self.value
        elif self.operator == _cl1lEl.LT:
            return v < self.value
        elif self.operator == _cl1lEl.LE:
            return v <= self.value
        elif self.operator == _cl1lEl.IN:
            return v in self.value
        elif self.operator == _cl1lEl.NOT_IN:
            return v not in self.value
        elif self.operator == _cl1lEl.BETWEEN:
            return self.value <= v <= self.secondary_value
        elif self.operator == _cl1lEl.LIKE:
            import re
            pattern = self.value.replace('%', '.*').replace('_', '.')
            return bool(re.match(pattern, str(v)))
        return False

@dataclass
class _cOlOEB:
    dimension: _cIOldf
    name: str
    values: List[Any] = field(default_factory=list)
    value_labels: Dict[Any, str] = field(default_factory=dict)
    is_hierarchical: bool = False
    parent_dimension: Optional[_cIOldf] = None
    default_aggregation: _cl1lEO = _cl1lEO.SUM
    custom_aggregator: Optional[Callable] = None

    def _fO00Ec(self, _fI0lEd: Any) -> str:
        return self.value_labels.get(_fI0lEd, str(_fI0lEd))

class _c0l1EE:

    def __init__(self, _fOIlEf: List[_cOlOEB]):
        self._dimension_specs: Dict[_cIOldf, _cOlOEB] = {spec.dimension: spec for spec in _fOIlEf}
        self._cells: Dict[Tuple[_cIOOE2, ...], _cOIOE4] = {}
        self._indices: Dict[_cIOldf, Dict[Any, Set[Tuple]]] = defaultdict(lambda: defaultdict(set))
        self._registry = ComponentRegistry.get_instance()

    @bridge(connects_to=['JonesEngine', 'OrderManager', 'ConditionState', 'ActivityState'], connection_types={'JonesEngine': 'analyzes', 'OrderManager': 'receives', 'ConditionState': 'reads', 'ActivityState': 'triggers'})
    def _f00OfO(self, _f10Ifl: Dict[str, float], _fl00f2: Dict[_cIOldf, Any], _f101f3: Optional[datetime]=None, _fIlOf4: float=1.0, _flllf5: Optional[Dict[str, Any]]=None):
        coords = []
        for dim, spec in self._dimension_specs.items():
            if dim in _fl00f2:
                _fI0lEd = _fl00f2[dim]
            else:
                _fI0lEd = None
            dv = _cIOOE2(dimension=dim, value=_fI0lEd, label=spec._fO00Ec(_fI0lEd))
            coords.append(dv)
        coord_tuple = tuple(sorted(coords, key=lambda x: x.dimension._fI0lEd))
        cell = _cOIOE4(coordinates=coord_tuple, metrics=_f10Ifl, timestamp=_f101f3 or datetime.now(), weight=_fIlOf4, metadata=_flllf5 or {})
        if coord_tuple in self._cells:
            self._cells[coord_tuple] = self._cells[coord_tuple]._f01OE6(cell)
        else:
            self._cells[coord_tuple] = cell
        for dv in coord_tuple:
            self._indices[dv.dimension][dv._fI0lEd].add(coord_tuple)

    def slice(self, _fl0Of6: List[_c0lIE8]) -> 'TradeCubeSlice':
        matching_cells = []
        for coord_tuple, cell in self._cells.items():
            matches_all = True
            for condition in _fl0Of6:
                for dv in coord_tuple:
                    if dv.dimension == condition.dimension:
                        if not condition._fl0OE9(dv):
                            matches_all = False
                            break
                if not matches_all:
                    break
            if matches_all:
                matching_cells.append(cell)
        return TradeCubeSlice(cells=matching_cells, conditions=_fl0Of6, dimension_specs=self._dimension_specs)

    def _fl0lf7(self, _fOIlEf: List[_cIOldf], _fl0Of6: Optional[List[_c0lIE8]]=None) -> 'TradeCubeDice':
        if _fl0Of6:
            filtered = self.slice(_fl0Of6).cells
        else:
            filtered = list(self._cells.values())
        return TradeCubeDice(cells=filtered, selected_dimensions=_fOIlEf, dimension_specs=self._dimension_specs)

    def _flI1f8(self, _fIlIf9: _cIOldf, _f01IfA: _cIOldf, _fOllfB: str='pnl', _fO1lE7: _cl1lEO=_cl1lEO.SUM) -> 'TradeCubePivot':
        return TradeCubePivot(cells=list(self._cells.values()), row_dimension=_fIlIf9, column_dimension=_f01IfA, metric=_fOllfB, aggregation=_fO1lE7, dimension_specs=self._dimension_specs)

    def _fO11fc(self, _fOlIfd: _cIOldf, _fO1lE7: _cl1lEO=_cl1lEO.SUM) -> Dict[Any, Dict[str, float]]:
        results: Dict[Any, List[_cOIOE4]] = defaultdict(list)
        for cell in self._cells.values():
            for dv in cell.coordinates:
                if dv._fOlIfd == _fOlIfd:
                    results[dv._fI0lEd].append(cell)
                    break
        aggregated = {}
        for _fI0lEd, cells in results.items():
            aggregated[_fI0lEd] = self._aggregate_cells(cells, _fO1lE7)
        return aggregated

    def _fI10fE(self, _f0l0ff: _cIOldf, _f1lOlOO: Any, _f100lOl: _cIOldf) -> Dict[Any, Dict[str, float]]:
        matching = []
        for cell in self._cells.values():
            for dv in cell.coordinates:
                if dv._fOlIfd == _f0l0ff and dv._fI0lEd == _f1lOlOO:
                    matching.append(cell)
                    break
        results: Dict[Any, List[_cOIOE4]] = defaultdict(list)
        for cell in matching:
            for dv in cell.coordinates:
                if dv._fOlIfd == _f100lOl:
                    results[dv._fI0lEd].append(cell)
                    break
        return {_fI0lEd: self._aggregate_cells(cells, _cl1lEO.SUM) for _fI0lEd, cells in results.items()}

    def _fI0llO2(self, _fOIOlO3: List[_cOIOE4], _fO1lE7: _cl1lEO) -> Dict[str, float]:
        if not _fOIOlO3:
            return {}
        metric_values: Dict[str, List[float]] = defaultdict(list)
        metric_weights: Dict[str, List[float]] = defaultdict(list)
        for cell in _fOIOlO3:
            for _fOllfB, _fI0lEd in cell._f10Ifl.items():
                metric_values[_fOllfB].append(_fI0lEd)
                metric_weights[_fOllfB].append(cell._fIlOf4)
        result = {}
        for _fOllfB, values in metric_values.items():
            weights = metric_weights[_fOllfB]
            if _fO1lE7 == _cl1lEO.SUM:
                result[_fOllfB] = sum(values)
            elif _fO1lE7 == _cl1lEO.MEAN:
                result[_fOllfB] = sum(values) / len(values)
            elif _fO1lE7 == _cl1lEO.MEDIAN:
                sorted_vals = sorted(values)
                mid = len(sorted_vals) // 2
                result[_fOllfB] = sorted_vals[mid]
            elif _fO1lE7 == _cl1lEO.MIN:
                result[_fOllfB] = min(values)
            elif _fO1lE7 == _cl1lEO.MAX:
                result[_fOllfB] = max(values)
            elif _fO1lE7 == _cl1lEO.FIRST:
                result[_fOllfB] = values[0]
            elif _fO1lE7 == _cl1lEO.LAST:
                result[_fOllfB] = values[-1]
            elif _fO1lE7 == _cl1lEO.STD:
                mean = sum(values) / len(values)
                variance = sum(((v - mean) ** 2 for v in values)) / len(values)
                result[_fOllfB] = math.sqrt(variance)
            elif _fO1lE7 == _cl1lEO.VAR:
                mean = sum(values) / len(values)
                result[_fOllfB] = sum(((v - mean) ** 2 for v in values)) / len(values)
            elif _fO1lE7 == _cl1lEO.COUNT:
                result[_fOllfB] = len(values)
            elif _fO1lE7 == _cl1lEO.WEIGHTED_MEAN:
                total_weight = sum(weights)
                if total_weight > 0:
                    result[_fOllfB] = sum((v * w for v, w in zip(values, weights))) / total_weight
                else:
                    result[_fOllfB] = 0
        return result

    def _fO10lO4(self, _fOlIfd: _cIOldf) -> List[Any]:
        return list(self._indices[_fOlIfd].keys())

    def _fl0OlO5(self) -> Set[str]:
        _f10Ifl = set()
        for cell in self._cells.values():
            _f10Ifl.update(cell._f10Ifl.keys())
        return _f10Ifl

    @property
    def dimension(self) -> int:
        return len(self._cells)

    @property
    def _f1OIlO7(self) -> int:
        return sum((cell.count for cell in self._cells.values()))

class _c011lO8:

    def __init__(self, _fOIOlO3: List[_cOIOE4], _fl0Of6: List[_c0lIE8], _fl1llO9: Dict[_cIOldf, _cOlOEB]):
        self._fOIOlO3 = _fOIOlO3
        self._fl0Of6 = _fl0Of6
        self._dimension_specs = _fl1llO9

    def _fl0llOA(self, _fO1lE7: _cl1lEO=_cl1lEO.SUM) -> Dict[str, float]:
        if not self._fOIOlO3:
            return {}
        metric_values: Dict[str, List[float]] = defaultdict(list)
        for cell in self._fOIOlO3:
            for _fOllfB, _fI0lEd in cell._f10Ifl.items():
                metric_values[_fOllfB].append(_fI0lEd)
        result = {}
        for _fOllfB, values in metric_values.items():
            if _fO1lE7 == _cl1lEO.SUM:
                result[_fOllfB] = sum(values)
            elif _fO1lE7 == _cl1lEO.MEAN:
                result[_fOllfB] = sum(values) / len(values)
        return result

    @property
    def _f1OIlOB(self) -> int:
        return sum((cell.count for cell in self._fOIOlO3))

    def _fII1lOc(self) -> List[Dict[str, Any]]:
        result = []
        for cell in self._fOIOlO3:
            row = dict(cell._f10Ifl)
            for dv in cell.coordinates:
                row[dv._fOlIfd._fI0lEd] = dv._fI0lEd
            row['_count'] = cell.count
            row['_weight'] = cell._fIlOf4
            result.append(row)
        return result

class _c00llOd:

    def __init__(self, _fOIOlO3: List[_cOIOE4], _fOlIlOE: List[_cIOldf], _fl1llO9: Dict[_cIOldf, _cOlOEB]):
        self._fOIOlO3 = _fOIOlO3
        self._fOlIlOE = _fOlIlOE
        self._dimension_specs = _fl1llO9

    def _fI10lOf(self, _fOlIfd: _cIOldf, _fO1lE7: _cl1lEO=_cl1lEO.SUM) -> Dict[Any, Dict[str, float]]:
        groups: Dict[Any, List[_cOIOE4]] = defaultdict(list)
        for cell in self._fOIOlO3:
            for dv in cell.coordinates:
                if dv._fOlIfd == _fOlIfd:
                    groups[dv._fI0lEd].append(cell)
                    break
        result = {}
        for _fI0lEd, group_cells in groups.items():
            _f10Ifl: Dict[str, List[float]] = defaultdict(list)
            for cell in group_cells:
                for m, v in cell._f10Ifl.items():
                    _f10Ifl[m].append(v)
            result[_fI0lEd] = {m: sum(vals) if _fO1lE7 == _cl1lEO.SUM else sum(vals) / len(vals) for m, vals in _f10Ifl.items()}
        return result

    def _fO0IllO(self, _fIOOlll: _cIOldf, _f1Olll2: _cIOldf, _fOllfB: str='pnl') -> Dict[Any, Dict[Any, float]]:
        result: Dict[Any, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        for cell in self._fOIOlO3:
            val1 = None
            val2 = None
            for dv in cell.coordinates:
                if dv._fOlIfd == _fIOOlll:
                    val1 = dv._fI0lEd
                elif dv._fOlIfd == _f1Olll2:
                    val2 = dv._fI0lEd
            if val1 is not None and val2 is not None:
                result[val1][val2] += cell._f10Ifl.get(_fOllfB, 0)
        return dict(result)

class _clOlll3:

    def __init__(self, _fOIOlO3: List[_cOIOE4], _fIlIf9: _cIOldf, _f01IfA: _cIOldf, _fOllfB: str, _fO1lE7: _cl1lEO, _fl1llO9: Dict[_cIOldf, _cOlOEB]):
        self._fIlIf9 = _fIlIf9
        self._f01IfA = _f01IfA
        self._fOllfB = _fOllfB
        self._fO1lE7 = _fO1lE7
        self._dimension_specs = _fl1llO9
        self._table: Dict[Any, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._row_values: Set[Any] = set()
        self._col_values: Set[Any] = set()
        for cell in _fOIOlO3:
            row_val = None
            col_val = None
            for dv in cell.coordinates:
                if dv._fOlIfd == _fIlIf9:
                    row_val = dv._fI0lEd
                    self._row_values.add(row_val)
                elif dv._fOlIfd == _f01IfA:
                    col_val = dv._fI0lEd
                    self._col_values.add(col_val)
            if row_val is not None and col_val is not None:
                self._table[row_val][col_val].append(cell._f10Ifl.get(_fOllfB, 0))

    def _fI10ll4(self, _f0IOll5: Any, _f0lIll6: Any) -> float:
        values = self._table.get(_f0IOll5, {}).get(_f0lIll6, [])
        if not values:
            return 0.0
        if self._fO1lE7 == _cl1lEO.SUM:
            return sum(values)
        elif self._fO1lE7 == _cl1lEO.MEAN:
            return sum(values) / len(values)
        elif self._fO1lE7 == _cl1lEO.COUNT:
            return len(values)
        return 0.0

    @property
    def _f10Oll7(self) -> List[Any]:
        return sorted(self._row_values)

    @property
    def _f011ll8(self) -> List[Any]:
        return sorted(self._col_values)

    def _f1llll9(self) -> List[List[float]]:
        matrix = []
        for _f0IOll5 in self._f10Oll7:
            row_data = []
            for _f0lIll6 in self._f011ll8:
                row_data.append(self._fI10ll4(_f0IOll5, _f0lIll6))
            matrix.append(row_data)
        return matrix

    def _f1IOllA(self) -> Dict[Any, float]:
        return {_f0IOll5: sum((self._fI10ll4(_f0IOll5, _f0lIll6) for _f0lIll6 in self._f011ll8)) for _f0IOll5 in self._f10Oll7}

    def _f0l0llB(self) -> Dict[Any, float]:
        return {_f0lIll6: sum((self._fI10ll4(_f0IOll5, _f0lIll6) for _f0IOll5 in self._f10Oll7)) for _f0lIll6 in self._f011ll8}

    def _f00lllc(self) -> float:
        return sum(self._f1IOllA().values())

class _c00Olld:
    TIMEFRAME_HIERARCHY = [Timeframe.TICK, Timeframe.S1, Timeframe.S5, Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1, Timeframe.MN1]

    def __init__(self, _fO0OllE: _c0l1EE):
        self._cube = _fO0OllE

    @bridge(connects_to=['TradeCube', 'JonesEngine', 'TimeframeAggregator'], connection_types={'TradeCube': 'extends', 'JonesEngine': 'integrates', 'TimeframeAggregator': 'uses'})
    def _fO11llf(self, _fI1ll2O: Timeframe, _fl01l2l: Timeframe, _fO1lE7: _cl1lEO=_cl1lEO.SUM) -> Dict[str, float]:
        slice_result = self._cube.slice([_c0lIE8(dimension=_cIOldf.TIMEFRAME, operator=_cl1lEl.EQ, value=_fI1ll2O._fI0lEd)])
        return slice_result._fl0llOA(_fO1lE7)

    def _f10Ol22(self, _f101l23: Optional[List[Timeframe]]=None) -> Dict[Timeframe, Dict[str, float]]:
        if _f101l23 is None:
            _f101l23 = [Timeframe.M1, Timeframe.M5, Timeframe.H1, Timeframe.D1]
        results = {}
        for tf in _f101l23:
            slice_result = self._cube.slice([_c0lIE8(dimension=_cIOldf.TIMEFRAME, operator=_cl1lEl.EQ, value=tf._fI0lEd)])
            results[tf] = slice_result._fl0llOA()
        return results

    def _fOO0l24(self, _fOllfB: str='pnl') -> Dict[Timeframe, float]:
        return {tf: self._cube.slice([_c0lIE8(dimension=_cIOldf.TIMEFRAME, operator=_cl1lEl.EQ, value=tf._fI0lEd)])._fl0llOA().get(_fOllfB, 0) for tf in self.TIMEFRAME_HIERARCHY}

class _cIl1l25:
    METRIC_CATEGORIES = {'technical': ['rsi', 'macd', 'sma', 'ema', 'bollinger', 'atr', 'volume'], 'fundamental': ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'revenue', 'earnings'], 'sentiment': ['news_sentiment', 'social_sentiment', 'analyst_sentiment'], 'risk': ['volatility', 'var', 'sharpe', 'max_drawdown', 'beta'], 'return': ['pnl', 'return_pct', 'gross_return', 'net_return']}

    def __init__(self, _fO0OllE: _c0l1EE):
        self._cube = _fO0OllE

    @bridge(connects_to=['TradeCube', 'MetricEngine', 'FeatureStore'], connection_types={'TradeCube': 'extends', 'MetricEngine': 'uses', 'FeatureStore': 'reads'})
    def _fIIOl26(self, _fOI0l27: str) -> Dict[str, float]:
        if _fOI0l27 not in self.METRIC_CATEGORIES:
            return {}
        category_metrics = self.METRIC_CATEGORIES[_fOI0l27]
        all_metrics = self._cube._fl0OlO5()
        result = {}
        for cell in self._cube._cells.values():
            for _fOllfB, _fI0lEd in cell._f10Ifl.items():
                if _fOllfB in category_metrics:
                    if _fOllfB not in result:
                        result[_fOllfB] = 0
                    result[_fOllfB] += _fI0lEd
        return result

    def _fOlIl28(self, _fl11E5: str, _f1I0l29: List[str]) -> Dict[str, Dict[str, float]]:
        ranges = self._create_ranges(_fl11E5)
        result = {}
        for range_label, (low, high) in ranges.items():
            matching_cells = [cell for cell in self._cube._cells.values() if low <= cell._f10Ifl.get(_fl11E5, 0) < high]
            secondary_values = {}
            for sec_metric in _f1I0l29:
                values = [c._f10Ifl.get(sec_metric, 0) for c in matching_cells]
                if values:
                    secondary_values[sec_metric] = sum(values) / len(values)
                else:
                    secondary_values[sec_metric] = 0
            result[range_label] = secondary_values
        return result

    def _f0OOl2A(self, _fOllfB: str, _fl01l2B: int=5) -> Dict[str, Tuple[float, float]]:
        values = [cell._f10Ifl.get(_fOllfB, 0) for cell in self._cube._cells.values() if _fOllfB in cell._f10Ifl]
        if not values:
            return {}
        min_val = min(values)
        max_val = max(values)
        step = (max_val - min_val) / _fl01l2B
        ranges = {}
        for i in range(_fl01l2B):
            low = min_val + i * step
            high = min_val + (i + 1) * step
            label = f'{low:.2f}-{high:.2f}'
            ranges[label] = (low, high)
        return ranges

class _cO1Il2c:

    def __init__(self, _fO0OllE: _c0l1EE):
        self._cube = _fO0OllE
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}

    @bridge(connects_to=['TradeCube', 'CorrelationCutter', 'ActivityState'], connection_types={'TradeCube': 'extends', 'CorrelationCutter': 'uses', 'ActivityState': 'reads'})
    def _flIOl2d(self, _fIOIl2E: Dict[Tuple[str, str], float]):
        self._correlation_matrix = _fIOIl2E

    def _fOlOl2f(self, _fOlOl3O: str, _fOl0l3l: str) -> Dict[str, Dict[str, float]]:
        corr = self._correlation_matrix.get((_fOlOl3O, _fOl0l3l), 0)
        if corr > 0.7:
            regime = 'high_positive'
        elif corr > 0.3:
            regime = 'positive'
        elif corr > -0.3:
            regime = 'neutral'
        elif corr > -0.7:
            regime = 'negative'
        else:
            regime = 'high_negative'
        matching_cells = []
        for cell in self._cube._cells.values():
            for dv in cell.coordinates:
                if dv._fOlIfd == _cIOldf.ASSET:
                    if dv._fI0lEd in [_fOlOl3O, _fOl0l3l]:
                        matching_cells.append(cell)
                        break
        if not matching_cells:
            return {regime: {}}
        _f10Ifl: Dict[str, List[float]] = defaultdict(list)
        for cell in matching_cells:
            for m, v in cell._f10Ifl.items():
                _f10Ifl[m].append(v)
        return {regime: {m: sum(vals) / len(vals) for m, vals in _f10Ifl.items()}}

def _f01ll32() -> _c0l1EE:
    _fOIlEf = [_cOlOEB(dimension=_cIOldf.TIMEFRAME, name='Timeframe', values=[tf._fI0lEd for tf in Timeframe], default_aggregation=_cl1lEO.SUM), _cOlOEB(dimension=_cIOldf.ASSET, name='Asset', default_aggregation=_cl1lEO.SUM), _cOlOEB(dimension=_cIOldf.ASSET_CLASS, name='Asset Class', values=['equity', 'fixed_income', 'commodity', 'fx', 'crypto'], default_aggregation=_cl1lEO.SUM), _cOlOEB(dimension=_cIOldf.REGIME, name='Market Regime', values=['bull', 'bear', 'neutral', 'volatile'], default_aggregation=_cl1lEO.SUM), _cOlOEB(dimension=_cIOldf.SECTOR, name='Sector', default_aggregation=_cl1lEO.SUM), _cOlOEB(dimension=_cIOldf.RISK, name='Risk Level', values=['low', 'medium', 'high'], default_aggregation=_cl1lEO.MEAN)]
    return _c0l1EE(_fOIlEf)

def _f1OIl33() -> _c0l1EE:
    _fOIlEf = [_cOlOEB(dimension=d, name=d._fI0lEd.title()) for d in _cIOldf]
    return _c0l1EE(_fOIlEf)