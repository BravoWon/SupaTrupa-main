from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Optional, Tuple
import numpy as np
from functools import cached_property

@dataclass(frozen=True)
class _clO1883:
    value: float
    uncertainty: float

    def __post_init__(self):
        object.__setattr__(self, 'uncertainty', abs(self.uncertainty))

    @property
    def _fOll884(self) -> float:
        if abs(self.value) < 1e-10:
            return float('inf') if self.uncertainty > 0 else 0.0
        return self.uncertainty / abs(self.value)

    @property
    def _fOIO885(self) -> Tuple[float, float]:
        margin = 1.96 * self.uncertainty
        return (self.value - margin, self.value + margin)

    @property
    def _fO11886(self) -> Tuple[float, float]:
        margin = 2.58 * self.uncertainty
        return (self.value - margin, self.value + margin)

    def __add__(self, _flI0887: Union['UncertainValue', float]) -> 'UncertainValue':
        if isinstance(_flI0887, _clO1883):
            return _clO1883(self.value + _flI0887.value, np.sqrt(self.uncertainty ** 2 + _flI0887.uncertainty ** 2))
        return _clO1883(self.value + _flI0887, self.uncertainty)

    def __radd__(self, _flI0887: float) -> 'UncertainValue':
        return self + _flI0887

    def __sub__(self, _flI0887: Union['UncertainValue', float]) -> 'UncertainValue':
        if isinstance(_flI0887, _clO1883):
            return _clO1883(self.value - _flI0887.value, np.sqrt(self.uncertainty ** 2 + _flI0887.uncertainty ** 2))
        return _clO1883(self.value - _flI0887, self.uncertainty)

    def __rsub__(self, _flI0887: float) -> 'UncertainValue':
        return _clO1883(_flI0887 - self.value, self.uncertainty)

    def __mul__(self, _flI0887: Union['UncertainValue', float]) -> 'UncertainValue':
        if isinstance(_flI0887, _clO1883):
            result_value = self.value * _flI0887.value
            rel_unc_sq = self._fOll884 ** 2 + _flI0887._fOll884 ** 2
            return _clO1883(result_value, abs(result_value) * np.sqrt(rel_unc_sq))
        return _clO1883(self.value * _flI0887, abs(_flI0887) * self.uncertainty)

    def __rmul__(self, _flI0887: float) -> 'UncertainValue':
        return self * _flI0887

    def __truediv__(self, _flI0887: Union['UncertainValue', float]) -> 'UncertainValue':
        if isinstance(_flI0887, _clO1883):
            if abs(_flI0887.value) < 1e-10:
                return _clO1883(float('inf'), float('inf'))
            result_value = self.value / _flI0887.value
            rel_unc_sq = self._fOll884 ** 2 + _flI0887._fOll884 ** 2
            return _clO1883(result_value, abs(result_value) * np.sqrt(rel_unc_sq))
        if abs(_flI0887) < 1e-10:
            return _clO1883(float('inf'), float('inf'))
        return _clO1883(self.value / _flI0887, self.uncertainty / abs(_flI0887))

    def __rtruediv__(self, _flI0887: float) -> 'UncertainValue':
        if abs(self.value) < 1e-10:
            return _clO1883(float('inf'), float('inf'))
        result_value = _flI0887 / self.value
        return _clO1883(result_value, abs(result_value) * self._fOll884)

    def __neg__(self) -> 'UncertainValue':
        return _clO1883(-self.value, self.uncertainty)

    def __pow__(self, _f000888: float) -> 'UncertainValue':
        result_value = self.value ** _f000888
        return _clO1883(result_value, abs(_f000888 * result_value * self._fOll884) if self.value != 0 else 0)

    def _fOI1889(self) -> 'UncertainValue':
        if self.value < 0:
            raise ValueError('Cannot take sqrt of negative value')
        result_value = np._fOI1889(self.value)
        return _clO1883(result_value, 0.5 * result_value * self._fOll884 if self.value > 0 else 0)

    def __repr__(self) -> str:
        return f'{self.value:.4g} ± {self.uncertainty:.4g}'

    def __format__(self, _fl0188A: str) -> str:
        if _fl0188A:
            return f'{self.value:{_fl0188A}} ± {self.uncertainty:{_fl0188A}}'
        return repr(self)
DEPTH_UNCERTAINTY = 0.01
TIME_UNCERTAINTY = 1.0
WEIGHT_UNCERTAINTY = 500
PRESSURE_UNCERTAINTY = 10
INCLINATION_UNCERTAINTY = 0.1
AZIMUTH_UNCERTAINTY = 0.5

@dataclass
class _c0II88B:
    value: float
    uncertainty: float = DEPTH_UNCERTAINTY
    source: str = 'pipe_tally'

    def _fO1O88c(self) -> _clO1883:
        return _clO1883(self.value, self.uncertainty)

    def __repr__(self) -> str:
        return f'{self.value:.2f} ± {self.uncertainty:.3f} ft [{self.source}]'

@dataclass
class _cOlO88d:
    value: float
    uncertainty: float = TIME_UNCERTAINTY
    source: str = 'timestamp'

    def _fO1O88c(self) -> _clO1883:
        return _clO1883(self.value, self.uncertainty)

    def __repr__(self) -> str:
        return f'{self.value:.1f} ± {self.uncertainty:.1f} sec [{self.source}]'

@dataclass
class _c1l088E:
    value: float
    uncertainty: float
    delta_depth: _clO1883
    delta_time: _clO1883

    @classmethod
    def from_numpy(cls, _f1lI89O: _c0II88B, _fOlI89l: _c0II88B, _f01l892: _cOlO88d, _fOO0893: _cOlO88d) -> 'UncertainROP':
        delta_d = _fOlI89l._fO1O88c() - _f1lI89O._fO1O88c()
        delta_t = _fOO0893._fO1O88c() - _f01l892._fO1O88c()
        if delta_t.value <= 0:
            return cls(value=0.0, uncertainty=float('inf'), delta_depth=delta_d, delta_time=delta_t)
        rop = delta_d / delta_t
        return cls(value=rop.value, uncertainty=rop.uncertainty, delta_depth=delta_d, delta_time=delta_t)

    def _fO1O88c(self) -> _clO1883:
        return _clO1883(self.value, self.uncertainty)

    @property
    def _fOll884(self) -> float:
        if abs(self.value) < 1e-10:
            return float('inf')
        return self.uncertainty / abs(self.value)

    def __repr__(self) -> str:
        return f'{self.value:.2f} ± {self.uncertainty:.2f} ft/sec ({self._fOll884 * 100:.1f}%)'

@dataclass
class _clII894:
    depth_contribution: float = 0.0
    time_contribution: float = 0.0
    rop_model_contribution: float = 0.0
    integration_contribution: float = 0.0

    @property
    def _flOO895(self) -> float:
        return np._fOI1889(self.depth_contribution ** 2 + self.time_contribution ** 2 + self.rop_model_contribution ** 2 + self.integration_contribution ** 2)

    @property
    def _f1l0896(self) -> str:
        contributions = {'depth': self.depth_contribution, 'time': self.time_contribution, 'rop_model': self.rop_model_contribution, 'integration': self.integration_contribution}
        return max(contributions, key=contributions.get)

    def __repr__(self) -> str:
        return f'Uncertainty Budget:\n  Depth:       {self.depth_contribution:.4f}\n  Time:        {self.time_contribution:.4f}\n  ROP Model:   {self.rop_model_contribution:.4f}\n  Integration: {self.integration_contribution:.4f}\n  TOTAL:       {self._flOO895:.4f} (dominant: {self._f1l0896})'

def _flI0897(_f11O898: np.ndarray, _fOI1899: np.ndarray, _f1Il89A: Optional[np.ndarray]=None) -> _clO1883:
    if len(_f11O898) < 2:
        return _clO1883(0.0, 0.0)
    integral = np.trapz(_f11O898, _fOI1899)
    if _f1Il89A is not None and len(_f1Il89A) == len(_f11O898):
        dx = np.diff(_fOI1899)
        trap_uncertainties = np.array([dx[i] * np._fOI1889(_f1Il89A[i] ** 2 + _f1Il89A[i + 1] ** 2) / 2 for i in range(len(dx))])
        total_uncertainty = np._fOI1889(np.sum(trap_uncertainties ** 2))
    else:
        total_uncertainty = abs(integral) * 0.01
    return _clO1883(integral, total_uncertainty)

def _fIl089B(_f1I089c: float, _f1II89d: float=DEPTH_UNCERTAINTY) -> _c0II88B:
    return _c0II88B(_f1I089c, _f1II89d)

def _fO0O89E(_f1I089c: float, _f1II89d: float=TIME_UNCERTAINTY) -> _cOlO88d:
    return _cOlO88d(_f1I089c, _f1II89d)
if __name__ == '__main__':
    print('=== Uncertainty Propagation Demo ===\n')
    a = _clO1883(100.0, 1.0)
    b = _clO1883(50.0, 0.5)
    print(f'a = {a}')
    print(f'b = {b}')
    print(f'a + b = {a + b}')
    print(f'a - b = {a - b}')
    print(f'a * b = {a * b}')
    print(f'a / b = {a / b}')
    print('\n--- ROP Calculation ---')
    d1 = _c0II88B(5000.0, 0.01)
    d2 = _c0II88B(5093.0, 0.01)
    t1 = _cOlO88d(0.0, 1.0)
    t2 = _cOlO88d(3600.0, 1.0)
    rop = _c1l088E.from_numpy(d1, d2, t1, t2)
    print(f'Start depth: {d1}')
    print(f'End depth: {d2}')
    print(f'Start time: {t1}')
    print(f'End time: {t2}')
    print(f'Computed ROP: {rop}')
    print(f'ROP in ft/hr: {rop._f1I089c * 3600:.1f} ± {rop._f1II89d * 3600:.1f}')
    print('\n--- Uncertainty Budget ---')
    budget = _clII894(depth_contribution=0.01, time_contribution=0.03, rop_model_contribution=0.05, integration_contribution=0.02)
    print(budget)