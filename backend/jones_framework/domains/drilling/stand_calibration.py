from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict
from enum import Enum, auto
import numpy as np
from functools import cached_property

class _cI0l89f(Enum):
    CONNECTION = auto()
    SURVEY = auto()
    CIRCULATION = auto()
    REAMING = auto()
    WIPER_TRIP = auto()
    TREATMENT = auto()
    EQUIPMENT = auto()
    WAITING = auto()
    TRIPPING = auto()
    CASING = auto()
    CEMENTING = auto()
    LOGGING = auto()
    TESTING = auto()
    OTHER = auto()

@dataclass(frozen=True)
class _cO118AO:
    category: _cI0l89f
    start_time: float
    duration: float
    depth: float
    description: Optional[str] = None

    @property
    def _fIll8Al(self) -> float:
        return self.start_time + self.duration

    def _fIOl8A2(self, _f1OI8A3: float, _fIlO8A4: float) -> float:
        overlap_start = max(self.start_time, _f1OI8A3)
        overlap_end = min(self._fIll8Al, _fIlO8A4)
        return max(0.0, overlap_end - overlap_start)

@dataclass(frozen=True)
class _cIIO8A5:
    time: float
    depth: float
    is_connection: bool = True
    duration: float = 0.0
    stand_number: Optional[int] = None
    notes: Optional[str] = None

@dataclass
class _c10I8A6:
    d_start: float
    d_end: float
    _f1OI8A3: float
    _fIlO8A4: float
    depth_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    rop_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    npt_duration: float = 0.0

    def __post_init__(self):
        self._compute_calibration()

    def _flO18A7(self):
        self.delta_d = self.d_end - self.d_start
        self.delta_t_total = self._fIlO8A4 - self._f1OI8A3
        self.delta_t_drilling = max(0.0, self.delta_t_total - self.npt_duration)
        if self.delta_d <= 0 or self.delta_t_drilling <= 0:
            self.rop_avg = 0.0
            self.true_rop_avg = 0.0
            self.calibration_factor = 1.0
            return
        self.rop_avg = self.delta_d / self.delta_t_total if self.delta_t_total > 0 else 0.0
        self.true_rop_avg = self.delta_d / self.delta_t_drilling
        if len(self.rop_samples) > 0:
            raw_inverse_integral = np.trapz(1.0 / np.maximum(self.rop_samples, 0.001), self.depth_samples)
            if raw_inverse_integral > 0:
                self.calibration_factor = self.delta_t_drilling / raw_inverse_integral
            else:
                self.calibration_factor = 1.0
        else:
            self.calibration_factor = 1.0

    def _f0l18A8(self, _fIOl8A9: float) -> float:
        if len(self.rop_samples) == 0:
            return self.rop_avg
        raw_rop = np.interp(_fIOl8A9, self.depth_samples, self.rop_samples)
        return raw_rop / self.calibration_factor

    def _f0018AA(self, _fIOl8A9: float, _fl108AB: int=100) -> float:
        if _fIOl8A9 <= self.d_start:
            return self._f1OI8A3
        if _fIOl8A9 >= self.d_end:
            return self._fIlO8A4
        depths = np.linspace(self.d_start, _fIOl8A9, _fl108AB)
        inverse_rops = np.array([1.0 / max(self._f0l18A8(d), 0.001) for d in depths])
        integral = np.trapz(inverse_rops, depths)
        return self._f1OI8A3 + integral

    def _fIl18Ac(self, _flOO8Ad: float, _fl108AB: int=100) -> float:
        if _flOO8Ad <= self._f1OI8A3:
            return self.d_start
        if _flOO8Ad >= self._fIlO8A4:
            return self.d_end
        d_lo, d_hi = (self.d_start, self.d_end)
        for _ in range(50):
            d_mid = (d_lo + d_hi) / 2
            t_mid = self._f0018AA(d_mid, _fl108AB)
            if abs(t_mid - _flOO8Ad) < 0.1:
                return d_mid
            if t_mid < _flOO8Ad:
                d_lo = d_mid
            else:
                d_hi = d_mid
        return (d_lo + d_hi) / 2

@dataclass
class _cl0O8AE:
    start_boundary: _cIIO8A5
    end_boundary: _cIIO8A5
    rop_model: _c10I8A6
    npt_events: List[_cO118AO] = field(default_factory=list)
    depth_log: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_log: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: List[str] = field(default_factory=list)

    @classmethod
    def from_numpy(cls, _fOl08BO: _cIIO8A5, _f0118Bl: _cIIO8A5, _fIIl8B2: Optional[np.ndarray]=None, _fl108B3: Optional[np.ndarray]=None, _f0O08B4: Optional[np.ndarray]=None, _f0lI8B5: Optional[List[str]]=None, _f01O8B6: Optional[List[_cO118AO]]=None) -> 'StandSegment':
        total_npt = 0.0
        if _f01O8B6:
            for event in _f01O8B6:
                total_npt += event._fIOl8A2(_fOl08BO._flOO8Ad, _f0118Bl._flOO8Ad)
        rop_model = _c10I8A6(d_start=_fOl08BO._fIOl8A9, d_end=_f0118Bl._fIOl8A9, t_start=_fOl08BO._flOO8Ad, t_end=_f0118Bl._flOO8Ad, depth_samples=_fIIl8B2 if _fIIl8B2 is not None else np.array([]), rop_samples=_fl108B3 if _fl108B3 is not None else np.array([]), npt_duration=total_npt)
        return cls(start_boundary=_fOl08BO, end_boundary=_f0118Bl, rop_model=rop_model, npt_events=_f01O8B6 or [], depth_log=_fIIl8B2 if _fIIl8B2 is not None else np.array([]), feature_log=_f0O08B4 if _f0O08B4 is not None else np.array([]), feature_names=_f0lI8B5 or [])

    @property
    def _f0O18B7(self) -> float:
        return self.end_boundary._fIOl8A9 - self.start_boundary._fIOl8A9

    @property
    def _fI1l8B8(self) -> float:
        return self.end_boundary._flOO8Ad - self.start_boundary._flOO8Ad

    @property
    def _fO0I8B9(self) -> float:
        return sum((event._fIOl8A2(self.start_boundary._flOO8Ad, self.end_boundary._flOO8Ad) for event in self._f01O8B6))

    @property
    def _fOOI8BA(self) -> float:
        return self._fI1l8B8 - self._fO0I8B9

    @property
    def _fl118BB(self) -> float:
        return self.rop_model.true_rop_avg

    @property
    def _fI0O8Bc(self) -> float:
        if self._fI1l8B8 <= 0:
            return 0.0
        return self._f0O18B7 / self._fI1l8B8

    @property
    def _fIII8Bd(self) -> float:
        if self._fI1l8B8 <= 0:
            return 0.0
        return self._fO0I8B9 / self._fI1l8B8

    @property
    def _fO118BE(self) -> float:
        return 1.0 - self._fIII8Bd

    def _f0l18Bf(self) -> Dict[_cI0l89f, float]:
        result: Dict[_cI0l89f, float] = {}
        for event in self._f01O8B6:
            overlap = event._fIOl8A2(self.start_boundary._flOO8Ad, self.end_boundary._flOO8Ad)
            if overlap > 0:
                if event.category in result:
                    result[event.category] += overlap
                else:
                    result[event.category] = overlap
        return result

    def _fIll8cO(self, _fIIl8cl: _cO118AO):
        self._f01O8B6.append(_fIIl8cl)
        self.rop_model.npt_duration = self._fO0I8B9
        self.rop_model._flO18A7()

    def _f10I8c2(self, _fIOl8A9: float) -> float:
        return self.rop_model._f0018AA(_fIOl8A9)

    def _fl108c3(self, _flOO8Ad: float) -> float:
        return self.rop_model._fIl18Ac(_flOO8Ad)

    def _fl0I8c4(self, _fIOl8A9: float) -> bool:
        return self.start_boundary._fIOl8A9 <= _fIOl8A9 <= self.end_boundary._fIOl8A9

    def _fO0l8c5(self, _flOO8Ad: float) -> bool:
        return self.start_boundary._flOO8Ad <= _flOO8Ad <= self.end_boundary._flOO8Ad

class _c01l8c6:

    def __init__(self, _fI1O8c7: str=''):
        self._fI1O8c7 = _fI1O8c7
        self.stands: List[_cl0O8AE] = []
        self._depth_index: List[float] = []

    def _flIO8c8(self, _fl018c9: _cl0O8AE):
        self.stands.append(_fl018c9)
        self._depth_index.append(_fl018c9.start_boundary._fIOl8A9)
        self._depth_index.sort()

    def _f0lO8cA(self, _flOO8Ad: float, _fIOl8A9: float, _fIIl8B2: Optional[np.ndarray]=None, _fl108B3: Optional[np.ndarray]=None, **kwargs):
        new_boundary = _cIIO8A5(time=_flOO8Ad, depth=_fIOl8A9, **kwargs)
        if self.stands:
            prev_boundary = self.stands[-1].end_boundary
            _fl018c9 = _cl0O8AE.from_numpy(start=prev_boundary, end=new_boundary, depth_samples=_fIIl8B2, rop_samples=_fl108B3)
            self._flIO8c8(_fl018c9)
        elif hasattr(self, '_pending_boundary'):
            _fl018c9 = _cl0O8AE.from_numpy(start=self._pending_boundary, end=new_boundary, depth_samples=_fIIl8B2, rop_samples=_fl108B3)
            self._flIO8c8(_fl018c9)
            del self._pending_boundary
        else:
            self._pending_boundary = new_boundary

    def _fIll8cB(self, _fIOl8A9: float) -> Optional[_cl0O8AE]:
        for _fl018c9 in self.stands:
            if _fl018c9._fl0I8c4(_fIOl8A9):
                return _fl018c9
        return None

    def _fl1l8cc(self, _flOO8Ad: float) -> Optional[_cl0O8AE]:
        for _fl018c9 in self.stands:
            if _fl018c9._fO0l8c5(_flOO8Ad):
                return _fl018c9
        return None

    def _f0018AA(self, _fIOl8A9: float) -> Optional[float]:
        _fl018c9 = self._fIll8cB(_fIOl8A9)
        if _fl018c9:
            return _fl018c9._f10I8c2(_fIOl8A9)
        if self.stands:
            if _fIOl8A9 < self.stands[0].start_boundary._fIOl8A9:
                return self.stands[0].start_boundary._flOO8Ad
            if _fIOl8A9 > self.stands[-1].end_boundary._fIOl8A9:
                last = self.stands[-1]
                extra_depth = _fIOl8A9 - last.end_boundary._fIOl8A9
                extra_time = extra_depth / max(last._fl118BB, 0.001)
                return last.end_boundary._flOO8Ad + extra_time
        return None

    def _fIl18Ac(self, _flOO8Ad: float) -> Optional[float]:
        _fl018c9 = self._fl1l8cc(_flOO8Ad)
        if _fl018c9:
            return _fl018c9._fl108c3(_flOO8Ad)
        if self.stands:
            if _flOO8Ad < self.stands[0].start_boundary._flOO8Ad:
                return self.stands[0].start_boundary._fIOl8A9
            if _flOO8Ad > self.stands[-1].end_boundary._flOO8Ad:
                last = self.stands[-1]
                extra_time = _flOO8Ad - last.end_boundary._flOO8Ad
                extra_depth = extra_time * last._fl118BB
                return last.end_boundary._fIOl8A9 + extra_depth
        return None

    def _fOll8cd(self, _fOOO8cE: np.ndarray, _f0OI8cf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.array([self._f0018AA(d) or 0 for d in _fOOO8cE])
        return (times, _fOOO8cE, _f0OI8cf)

    def _fI0O8dO(self, _f0Il8dl: int=100) -> Tuple[np.ndarray, np.ndarray]:
        if not self.stands:
            return (np.array([]), np.array([]))
        d_min = self.stands[0].start_boundary._fIOl8A9
        d_max = self.stands[-1].end_boundary._fIOl8A9
        _fOOO8cE = np.linspace(d_min, d_max, _f0Il8dl)
        rops = np.array([self._fIll8cB(d).rop_model._f0l18A8(d) if self._fIll8cB(d) else 0 for d in _fOOO8cE])
        return (_fOOO8cE, rops)

    def _fI0l8d2(self) -> dict:
        if not self.stands:
            return {'n_stands': 0}
        _fI1l8B8 = self.stands[-1].end_boundary._flOO8Ad - self.stands[0].start_boundary._flOO8Ad
        total_drilling = sum((s._fOOI8BA for s in self.stands))
        total_npt = sum((s._fO0I8B9 for s in self.stands))
        return {'well_name': self._fI1O8c7, 'n_stands': len(self.stands), 'total_depth': self.stands[-1].end_boundary._fIOl8A9 - self.stands[0].start_boundary._fIOl8A9, 'total_time': _fI1l8B8, 'total_drilling_time': total_drilling, 'total_npt_time': total_npt, 'drilling_efficiency': total_drilling / _fI1l8B8 if _fI1l8B8 > 0 else 0.0, 'avg_true_rop': np.mean([s._fl118BB for s in self.stands]), 'avg_apparent_rop': np.mean([s._fI0O8Bc for s in self.stands]), 'avg_stand_length': np.mean([s._f0O18B7 for s in self.stands]), 'depth_range': (self.stands[0].start_boundary._fIOl8A9, self.stands[-1].end_boundary._fIOl8A9), 'time_range': (self.stands[0].start_boundary._flOO8Ad, self.stands[-1].end_boundary._flOO8Ad)}

    def _f0ll8d3(self) -> Dict[_cI0l89f, float]:
        result: Dict[_cI0l89f, float] = {}
        for _fl018c9 in self.stands:
            for category, duration in _fl018c9._f0l18Bf().items():
                if category in result:
                    result[category] += duration
                else:
                    result[category] = duration
        return result

    def _fIll8cO(self, _fIIl8cl: _cO118AO):
        for _fl018c9 in self.stands:
            if _fIIl8cl._fIOl8A2(_fl018c9.start_boundary._flOO8Ad, _fl018c9.end_boundary._flOO8Ad) > 0:
                _fl018c9._fIll8cO(_fIIl8cl)

def _fOlO8d4(_fOI18d5: List[float], _fO008d6: List[float], _f11I8d7: Optional[List[np.ndarray]]=None, _f0I08d8: Optional[List[np.ndarray]]=None) -> _c01l8c6:
    trajectory = _c01l8c6()
    for i, (t, d) in enumerate(zip(_fOI18d5, _fO008d6)):
        _fIIl8B2 = _f11I8d7[i - 1] if _f11I8d7 and i > 0 else None
        _fl108B3 = _f0I08d8[i - 1] if _f0I08d8 and i > 0 else None
        trajectory._f0lO8cA(time=t, depth=d, stand_number=i, depth_samples=_fIIl8B2, rop_samples=_fl108B3)
    return trajectory

def _f11O8d9(_fOOO8cE: np.ndarray, _fIOO8dA: List[Tuple[float, float]], _fII18dB: float=50.0) -> np.ndarray:
    boundaries = sorted(_fIOO8dA, key=lambda x: x[0])
    times = np.zeros_like(_fOOO8cE)
    for i, d in enumerate(_fOOO8cE):
        lower_bound = None
        upper_bound = None
        for bd, bt in boundaries:
            if bd <= d:
                lower_bound = (bd, bt)
            if bd >= d and upper_bound is None:
                upper_bound = (bd, bt)
        if lower_bound is None:
            first_d, first_t = boundaries[0]
            times[i] = first_t - (first_d - d) / _fII18dB
        elif upper_bound is None:
            last_d, last_t = boundaries[-1]
            times[i] = last_t + (d - last_d) / _fII18dB
        elif lower_bound[0] == upper_bound[0]:
            times[i] = lower_bound[1]
        else:
            d0, t0 = lower_bound
            d1, t1 = upper_bound
            frac = (d - d0) / (d1 - d0)
            times[i] = t0 + frac * (t1 - t0)
    return times
if __name__ == '__main__':
    print('=== Stand-Calibrated Well Trajectory ===\n')
    _fOI18d5 = [0, 3600, 7500, 11200, 15000, 18500]
    _fO008d6 = [0, 500, 1050, 1580, 2100, 2620]
    trajectory = _fOlO8d4(connection_times=_fOI18d5, connection_depths=_fO008d6)
    print('Well Summary:')
    for k, v in trajectory._fI0l8d2().items():
        print(f'  {k}: {v}')
    print('\nStand Details:')
    for i, _fl018c9 in enumerate(trajectory.stands):
        print(f'  Stand {i + 1}: {_fl018c9._f0O18B7:.0f}ft in {_fl018c9._fOOI8BA:.0f}s (ROP: {_fl018c9._fl118BB:.1f} ft/hr)')
    print('\nTime at specific depths:')
    test_depths = [250, 750, 1300, 1850, 2400]
    for d in test_depths:
        t = trajectory._f0018AA(d)
        print(f'  d={d}ft → t={t:.0f}s ({t / 3600:.2f}hr)')
    print('\nDepth at specific times:')
    test_times = [1800, 5400, 9000, 13000, 17000]
    for t in test_times:
        d = trajectory._fIl18Ac(t)
        print(f'  t={t}s ({t / 3600:.2f}hr) → d={d:.0f}ft')