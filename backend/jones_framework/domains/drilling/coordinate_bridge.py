from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from functools import cached_property
try:
    from jones_framework.core.condition_state import ConditionState
    from jones_framework.core.manifold_bridge import bridge, ConnectionType
    HAS_FRAMEWORK = True
except ImportError:
    HAS_FRAMEWORK = False

    def bridge(*args, **kwargs):

        def _fO0093f(cls):
            return cls
        return _fO0093f

    class ConnectionType:
        EXTENDS = 'EXTENDS'
        TRANSFORMS = 'TRANSFORMS'

class _cI0I94O(Enum):
    TIME = auto()
    DEPTH = auto()
    DUAL = auto()

@dataclass(frozen=True)
class _cO0094l:
    rop_values: Tuple[float, ...]
    index_values: Tuple[float, ...]
    index_type: _cI0I94O

    @classmethod
    def from_numpy(cls, _fl1I943: np.ndarray, _fI10944: np.ndarray, _fIII945: _cI0I94O) -> 'ROPJacobian':
        return cls(rop_values=tuple(_fl1I943.flatten()), index_values=tuple(_fI10944.flatten()), index_type=_fIII945)

    def _fO10946(self, _fO1O947: float) -> float:
        return float(np.interp(_fO1O947, self.index_values, self.rop_values))

    def _fOI1948(self, _fO1O947: float) -> float:
        _fl1I943 = self._fO10946(_fO1O947)
        if _fl1I943 <= 0:
            return float('inf')
        return 1.0 / _fl1I943

    @cached_property
    def _f100949(self) -> float:
        return float(np.mean(self.rop_values))

    @cached_property
    def _fl1094A(self) -> float:
        return float(np.var(self.rop_values))

class _c1II94B:

    def __init__(self, _f10l94c: _cO0094l):
        self._f10l94c = _f10l94c
        self._precompute_integrals()

    def _fl0O94d(self):
        indices = np.array(self._f10l94c.index_values)
        rops = np.array(self._f10l94c.rop_values)
        d_idx = np.diff(indices)
        if self._f10l94c._fIII945 == _cI0I94O.DEPTH:
            inverse_rop = np.where(rops[:-1] > 0, 1.0 / rops[:-1], 0)
            dt = inverse_rop * d_idx
            self._cumulative_time = np.concatenate([[0], np.cumsum(dt)])
            self._cumulative_depth = indices
        else:
            dd = rops[:-1] * d_idx
            self._cumulative_depth = np.concatenate([[0], np.cumsum(dd)])
            self._cumulative_time = indices

    def _fOOI94E(self, _fllO94f: float, _f1O095O: float=0, _fIO195l: float=0) -> float:
        if self._f10l94c._fIII945 != _cI0I94O.DEPTH:
            raise ValueError('Jacobian must be depth-indexed for depth_to_time')
        idx = np.searchsorted(self._cumulative_depth, _fllO94f)
        if idx >= len(self._cumulative_time):
            idx = len(self._cumulative_time) - 1
        if idx > 0:
            frac = (_fllO94f - self._cumulative_depth[idx - 1]) / (self._cumulative_depth[idx] - self._cumulative_depth[idx - 1] + 1e-10)
            time_offset = self._cumulative_time[idx - 1] + frac * (self._cumulative_time[idx] - self._cumulative_time[idx - 1])
        else:
            time_offset = self._cumulative_time[idx]
        return _f1O095O + time_offset

    def _fI0I952(self, _fOO0953: float, _fIO195l: float=0, _f1O095O: float=0) -> float:
        if self._f10l94c._fIII945 != _cI0I94O.TIME:
            raise ValueError('Jacobian must be time-indexed for time_to_depth')
        idx = np.searchsorted(self._cumulative_time, _fOO0953)
        if idx >= len(self._cumulative_depth):
            idx = len(self._cumulative_depth) - 1
        if idx > 0:
            frac = (_fOO0953 - self._cumulative_time[idx - 1]) / (self._cumulative_time[idx] - self._cumulative_time[idx - 1] + 1e-10)
            depth_offset = self._cumulative_depth[idx - 1] + frac * (self._cumulative_depth[idx] - self._cumulative_depth[idx - 1])
        else:
            depth_offset = self._cumulative_depth[idx]
        return _fIO195l + depth_offset

@dataclass
class _cI0I954:
    _f10l94c: _cO0094l
    _f1O095O: float = 0.0
    _fIO195l: float = 0.0

    def __post_init__(self):
        self.back_integral = _c1II94B(self._f10l94c)

    def _f10I955(self, _fllO94f: np.ndarray, to_numpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        times = np.array([self.back_integral._fOOI94E(d, self._f1O095O, self._fIO195l) for d in _fllO94f])
        return (times, to_numpy)

    def _f1OO957(self, _fOO0953: np.ndarray, to_numpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        depths = np.array([self.back_integral._fI0I952(t, self._fIO195l, self._f1O095O) for t in _fOO0953])
        return (depths, to_numpy)

    def _f11O958(self, _fO0I959: np.ndarray, to_numpy: np.ndarray, _f01O95A: _cI0I94O) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if _f01O95A == _cI0I94O.DEPTH:
            times, _ = self._f10I955(_fO0I959, to_numpy)
            return (times, _fO0I959, to_numpy)
        else:
            depths, _ = self._f1OO957(_fO0I959, to_numpy)
            return (_fO0I959, depths, to_numpy)
if HAS_FRAMEWORK:

    @bridge(connects_to=['ConditionState', 'ActivityState'], connection_types={'ConditionState': ConnectionType.EXTENDS, 'ActivityState': ConnectionType.TRANSFORMS}, metadata={'domain': 'drilling', 'version': '1.0.0'})
    class _c00195B(ConditionState):

        @classmethod
        def _f1O195c(cls, to_numpy: np.ndarray, _fOO0953: Optional[float]=None, _fllO94f: Optional[float]=None, _fl1I943: Optional[float]=None, _f0IO95d: _cI0I94O=_cI0I94O.DUAL, _f11095E: Optional[Dict]=None) -> 'DrillingConditionState':
            extended = np.concatenate([[_fOO0953 or 0.0, _fllO94f or 0.0, _fl1I943 or 0.0], to_numpy])
            meta = _f11095E or {}
            meta.update({'coordinate_system': _f0IO95d.name, 'has_time': _fOO0953 is not None, 'has_depth': _fllO94f is not None, 'has_rop': _fl1I943 is not None, 'feature_dim': len(to_numpy)})
            return cls.from_numpy(extended, metadata=meta)

        @property
        def _fOO0953(self) -> Optional[float]:
            if self._f11095E.get('has_time'):
                return float(self.vector[0])
            return None

        @property
        def _fllO94f(self) -> Optional[float]:
            if self._f11095E.get('has_depth'):
                return float(self.vector[1])
            return None

        @property
        def _fl1I943(self) -> Optional[float]:
            if self._f11095E.get('has_rop'):
                return float(self.vector[2])
            return None

        @property
        def to_numpy(self) -> np.ndarray:
            return np.array(self.vector[3:])

        def _fI1095f(self, _f10O96O: _cI0I954) -> 'DrillingConditionState':
            if self._fOO0953 is not None and self._fllO94f is None:
                new_depth = _f10O96O.back_integral._fI0I952(self._fOO0953)
                return self._f1O195c(features=self.to_numpy, time=self._fOO0953, depth=new_depth, rop=self._fl1I943, coordinate_system=_cI0I94O.DUAL, metadata=self._f11095E)
            elif self._fllO94f is not None and self._fOO0953 is None:
                new_time = _f10O96O.back_integral._fOOI94E(self._fllO94f)
                return self._f1O195c(features=self.to_numpy, time=new_time, depth=self._fllO94f, rop=self._fl1I943, coordinate_system=_cI0I94O.DUAL, metadata=self._f11095E)
            else:
                return self
else:

    @dataclass
    class _c00195B:
        _fOO0953: Optional[float]
        _fllO94f: Optional[float]
        _fl1I943: Optional[float]
        to_numpy: np.ndarray
        _f11095E: Dict = field(default_factory=dict)

class _cO0I96l:

    def __init__(self, _f10O96O: _cI0I954, _f11l962: Optional[List[str]]=None):
        self._f10O96O = _f10O96O
        self._f11l962 = _f11l962 or []
        self._states: List[_c00195B] = []
        self._time_index: Dict[float, int] = {}
        self._depth_index: Dict[float, int] = {}

    def _f0l1963(self, _f10O964: _c00195B):
        if _f10O964._fOO0953 is None or _f10O964._fllO94f is None:
            _f10O964 = _f10O964._fI1095f(self._f10O96O)
        idx = len(self._states)
        self._states.append(_f10O964)
        if _f10O964._fOO0953 is not None:
            self._time_index[_f10O964._fOO0953] = idx
        if _f10O964._fllO94f is not None:
            self._depth_index[_f10O964._fllO94f] = idx

    def _fOI0965(self, _fI0l966: np.ndarray, to_numpy: np.ndarray, _fl1I943: Optional[np.ndarray]=None):
        for i, t in enumerate(_fI0l966):
            _f10O964 = _c00195B._f1O195c(features=to_numpy[i], time=t, depth=None, rop=_fl1I943[i] if _fl1I943 is not None else None, coordinate_system=_cI0I94O.TIME)
            self._f0l1963(_f10O964)

    def _f0II967(self, _f1O1968: np.ndarray, to_numpy: np.ndarray, _fl1I943: Optional[np.ndarray]=None):
        for i, d in enumerate(_f1O1968):
            _f10O964 = _c00195B._f1O195c(features=to_numpy[i], time=None, depth=d, rop=_fl1I943[i] if _fl1I943 is not None else None, coordinate_system=_cI0I94O.DEPTH)
            self._f0l1963(_f10O964)

    def _fIlO969(self, _fllO94f: float) -> Optional[_c00195B]:
        _f1O1968 = sorted(self._depth_index.keys())
        if not _f1O1968:
            return None
        idx = np.searchsorted(_f1O1968, _fllO94f)
        if idx == 0:
            return self._states[self._depth_index[_f1O1968[0]]]
        if idx >= len(_f1O1968):
            return self._states[self._depth_index[_f1O1968[-1]]]
        d_lo, d_hi = (_f1O1968[idx - 1], _f1O1968[idx])
        s_lo = self._states[self._depth_index[d_lo]]
        s_hi = self._states[self._depth_index[d_hi]]
        frac = (_fllO94f - d_lo) / (d_hi - d_lo + 1e-10)
        interp_features = (1 - frac) * s_lo.to_numpy + frac * s_hi.to_numpy
        interp_time = self._f10O96O.back_integral._fOOI94E(_fllO94f)
        return _c00195B._f1O195c(features=interp_features, time=interp_time, depth=_fllO94f, rop=(1 - frac) * (s_lo._fl1I943 or 0) + frac * (s_hi._fl1I943 or 0), coordinate_system=_cI0I94O.DUAL)

    def _fOI196A(self, _fOO0953: float) -> Optional[_c00195B]:
        _fI0l966 = sorted(self._time_index.keys())
        if not _fI0l966:
            return None
        idx = np.searchsorted(_fI0l966, _fOO0953)
        if idx == 0:
            return self._states[self._time_index[_fI0l966[0]]]
        if idx >= len(_fI0l966):
            return self._states[self._time_index[_fI0l966[-1]]]
        t_lo, t_hi = (_fI0l966[idx - 1], _fI0l966[idx])
        s_lo = self._states[self._time_index[t_lo]]
        s_hi = self._states[self._time_index[t_hi]]
        frac = (_fOO0953 - t_lo) / (t_hi - t_lo + 1e-10)
        interp_features = (1 - frac) * s_lo.to_numpy + frac * s_hi.to_numpy
        interp_depth = self._f10O96O.back_integral._fI0I952(_fOO0953)
        return _c00195B._f1O195c(features=interp_features, time=_fOO0953, depth=interp_depth, rop=(1 - frac) * (s_lo._fl1I943 or 0) + frac * (s_hi._fl1I943 or 0), coordinate_system=_cI0I94O.DUAL)

    def _f11196B(self, _f0IO95d: _cI0I94O=_cI0I94O.DUAL) -> np.ndarray:
        points = []
        for _f10O964 in self._states:
            if _f0IO95d == _cI0I94O.TIME:
                point = np.concatenate([[_f10O964._fOO0953 or 0], _f10O964.to_numpy])
            elif _f0IO95d == _cI0I94O.DEPTH:
                point = np.concatenate([[_f10O964._fllO94f or 0], _f10O964.to_numpy])
            else:
                point = np.concatenate([[_f10O964._fOO0953 or 0, _f10O964._fllO94f or 0], _f10O964.to_numpy])
            points.append(point)
        return np.array(points)

    def _f0ll96c(self, _fO1l96d: _c00195B, _f01O96E: _c00195B, _fl1096f: float=1.0, _f11l97O: float=1.0) -> float:
        dt = (_fO1l96d._fOO0953 or 0) - (_f01O96E._fOO0953 or 0)
        dd = (_fO1l96d._fllO94f or 0) - (_f01O96E._fllO94f or 0)
        avg_rop = 0.5 * ((_fO1l96d._fl1I943 or self._f10O96O._f10l94c._f100949) + (_f01O96E._fl1I943 or self._f10O96O._f10l94c._f100949))
        if avg_rop <= 0:
            avg_rop = 1.0
        g_tt = _fl1096f
        g_dd = _f11l97O
        g_td = 1.0 / avg_rop
        coord_dist_sq = g_tt * dt ** 2 + 2 * g_td * dt * dd + g_dd * dd ** 2
        feature_dist_sq = np.sum((_fO1l96d.to_numpy - _f01O96E.to_numpy) ** 2)
        return np.sqrt(coord_dist_sq + feature_dist_sq)

    @property
    def _fI1O97l(self) -> Tuple[float, float]:
        _f1O1968 = [s._fllO94f for s in self._states if s._fllO94f is not None]
        if not _f1O1968:
            return (0.0, 0.0)
        return (min(_f1O1968), max(_f1O1968))

    @property
    def _f0I1972(self) -> Tuple[float, float]:
        _fI0l966 = [s._fOO0953 for s in self._states if s._fOO0953 is not None]
        if not _fI0l966:
            return (0.0, 0.0)
        return (min(_fI0l966), max(_fI0l966))

    def _f1ll973(self) -> Dict:
        return {'n_states': len(self._states), 'depth_range': self._fI1O97l, 'time_range': self._f0I1972, 'mean_rop': self._f10O96O._f10l94c._f100949, 'rop_variance': self._f10O96O._f10l94c._fl1094A, 'feature_dim': self._states[0].to_numpy.shape[0] if self._states else 0, 'feature_names': self._f11l962}