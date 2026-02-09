from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np

class _c0l1AlE(ABC):

    @abstractmethod
    def __call__(self, _f0IOAlf: np.ndarray) -> float:
        pass

    @abstractmethod
    def distance_to(self, _f0IOAlf: np.ndarray) -> np.ndarray:
        pass

    def _f00lA2l(self, _f0IOAlf: np.ndarray) -> np.ndarray:
        eps = 1e-05
        n = len(_f0IOAlf)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                delta_i = np.zeros(n)
                delta_j = np.zeros(n)
                delta_i[i] = eps
                delta_j[j] = eps
                H[i, j] = (self(_f0IOAlf + delta_i + delta_j) - self(_f0IOAlf + delta_i - delta_j) - self(_f0IOAlf - delta_i + delta_j) + self(_f0IOAlf - delta_i - delta_j)) / (4 * eps * eps)
        return H

class _cI0lA22(_c0l1AlE):

    def __init__(self, _f0I0A23: float=1.0, _fI00A24: float=0.05):
        self._f0I0A23 = _f0I0A23
        self._fI00A24 = _fI00A24

    def __call__(self, _f0IOAlf: np.ndarray) -> float:
        if len(_f0IOAlf) == 0:
            return 0.0
        volatility = np.std(_f0IOAlf)
        if volatility < self._fI00A24:
            compression = self._fI00A24 - volatility
            return self._f0I0A23 * np.exp(compression / self._fI00A24)
        else:
            return self._f0I0A23 / (1 + volatility)

    def distance_to(self, _f0IOAlf: np.ndarray) -> np.ndarray:
        eps = 1e-06
        grad = np.zeros_like(_f0IOAlf)
        for i in range(len(_f0IOAlf)):
            delta = np.zeros_like(_f0IOAlf)
            delta[i] = eps
            grad[i] = (self(_f0IOAlf + delta) - self(_f0IOAlf - delta)) / (2 * eps)
        return grad

class _c0O0A25(_c0l1AlE):

    def __init__(self, _f1O1A26: float=0.4, _fIOOA27: float=0.3, _fIl0A28: float=0.3, _fO1lA29: float=100.0, _f000A2A: float=0.2):
        self._f1O1A26 = _f1O1A26
        self._fIOOA27 = _fIOOA27
        self._fIl0A28 = _fIl0A28
        self._fO1lA29 = _fO1lA29
        self._f000A2A = _f000A2A

    def __call__(self, _f0IOAlf: np.ndarray) -> float:
        if len(_f0IOAlf) < 4:
            return 0.0
        porosity = _f0IOAlf[1] if len(_f0IOAlf) > 1 else 0
        permeability = _f0IOAlf[2] if len(_f0IOAlf) > 2 else 0
        perm_score = 1.0 / (1.0 + abs(permeability - self._fO1lA29) / self._fO1lA29)
        poro_score = 1.0 / (1.0 + abs(porosity - self._f000A2A) / self._f000A2A)
        conn_score = 0.5
        value = self._f1O1A26 * perm_score + self._fIOOA27 * poro_score + self._fIl0A28 * conn_score
        return value

    def distance_to(self, _f0IOAlf: np.ndarray) -> np.ndarray:
        eps = 1e-06
        grad = np.zeros_like(_f0IOAlf)
        for i in range(len(_f0IOAlf)):
            delta = np.zeros_like(_f0IOAlf)
            delta[i] = eps
            grad[i] = (self(_f0IOAlf + delta) - self(_f0IOAlf - delta)) / (2 * eps)
        return grad

class _cOllA2B(_c0l1AlE):

    def __init__(self, _f0I0A2c: List[Tuple[_c0l1AlE, float]]):
        self._f0I0A2c = _f0I0A2c

    def __call__(self, _f0IOAlf: np.ndarray) -> float:
        return sum((w * f(_f0IOAlf) for f, w in self._f0I0A2c))

    def distance_to(self, _f0IOAlf: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(_f0IOAlf)
        for f, w in self._f0I0A2c:
            grad += w * f.distance_to(_f0IOAlf)
        return grad

@dataclass
class _clIlA2d:
    base_dimension: int
    value_function: _c0l1AlE

    def _fI0IA2E(self, _f0IOAlf: np.ndarray) -> float:
        v = self.value_function(_f0IOAlf)
        return float(np.exp(2 * v))

    def _fOI0A2f(self, _f0IOAlf: np.ndarray) -> np.ndarray:
        scale = self._fI0IA2E(_f0IOAlf)
        return scale * np.eye(self.base_dimension)

    def _fIOIA3O(self, _flIlA3l: np.ndarray, _f1I0A32: np.ndarray, _f00lA33: int=10) -> float:
        path = np.linspace(_flIlA3l, _f1I0A32, _f00lA33)
        total_dist = 0.0
        for i in range(len(path) - 1):
            mid = (path[i] + path[i + 1]) / 2
            scale = np.sqrt(self._fI0IA2E(mid))
            segment_dist = np.linalg.norm(path[i + 1] - path[i])
            total_dist += scale * segment_dist
        return total_dist

    def _f1I0A34(self, _fOIOA35: np.ndarray, _f1IOA36: np.ndarray, _f1OOA37: int=50, _fIl1A38: int=100, _fl11A39: float=0.1) -> np.ndarray:
        path = np.linspace(_fOIOA35, _f1IOA36, _f1OOA37)
        for _ in range(_fIl1A38):
            for i in range(1, _f1OOA37 - 1):
                value_grad = self.value_function.distance_to(path[i])
                smooth_grad = path[i - 1] + path[i + 1] - 2 * path[i]
                path[i] += _fl11A39 * (0.3 * value_grad + 0.7 * smooth_grad)
        return path

    def _fllIA3A(self, _f0IOA3B: np.ndarray) -> float:
        energy = 0.0
        for i in range(len(_f0IOA3B) - 1):
            mid = (_f0IOA3B[i] + _f0IOA3B[i + 1]) / 2
            scale = self._fI0IA2E(mid)
            segment_energy = scale * np.sum((_f0IOA3B[i + 1] - _f0IOA3B[i]) ** 2)
            energy += segment_energy
        return energy

    def _f1IIA3c(self, _f0IOAlf: np.ndarray, _fl11A39: float=0.1) -> np.ndarray:
        grad = self.value_function.distance_to(_f0IOAlf)
        scale = self._fI0IA2E(_f0IOAlf)
        new_point = _f0IOAlf + _fl11A39 * grad / np.sqrt(scale + 1e-10)
        return new_point

    def _f010A3d(self, _fOIOA35: np.ndarray, _flOIA3E: int=1000, _fIO1A3f: float=1e-06) -> Tuple[np.ndarray, List[float]]:
        _f0IOAlf = _fOIOA35.copy()
        history = [self.value_function(_f0IOAlf)]
        for i in range(_flOIA3E):
            new_point = self._f1IIA3c(_f0IOAlf)
            new_value = self.value_function(new_point)
            history.append(new_value)
            if np.linalg.norm(new_point - _f0IOAlf) < _fIO1A3f:
                break
            _f0IOAlf = new_point
        return (_f0IOAlf, history)

# Public API aliases for obfuscated classes
ValueFunction = _c0l1AlE
VolatilityFunction = _cI0lA22
DrillabilityFunction = _c0O0A25
CompositeValueFunction = _cOllA2B
MetricWarper = _clIlA2d
