from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from jones_framework.core.tensor_ops import Tensor
from jones_framework.core.manifold_bridge import bridge, ConnectionType

@dataclass
class _c1IlEO6:
    name: str
    reduction: str = 'mean'

@bridge(connects_to=['Tensor'], connection_types={'Tensor': ConnectionType.USES})
class _cO1IEO7(ABC):

    def __init__(self, _f101EO8: _c1IlEO6):
        self._f101EO8 = _f101EO8

    @abstractmethod
    def _f01IEO9(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        pass

    def __call__(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        loss = self._f01IEO9(_fOlOEOA, _flOIEOB)
        if self._f101EO8.reduction == 'mean':
            return loss.mean()
        elif self._f101EO8.reduction == 'sum':
            return loss.sum()
        return loss

class _clllEOc(_cO1IEO7):

    def _f01IEO9(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        return (_fOlOEOA - _flOIEOB) ** 2

class _c0IIEOd(_cO1IEO7):

    def _f01IEO9(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        return (_fOlOEOA - _flOIEOB).abs()

class _cIIIEOE(_cO1IEO7):

    def _f01IEO9(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        log_probs = _fOlOEOA.log_softmax(dim=-1)
        n = _fOlOEOA.shape[0] if _fOlOEOA.ndim > 1 else 1
        loss = Tensor.zeros(n)
        for i in range(n):
            idx = int(_flOIEOB._data[i])
            loss._data[i] = -log_probs._data[i, idx]
        return loss

class _cllOEOf(_cO1IEO7):

    def __init__(self, _f101EO8: _c1IlEO6, _fOO0ElO: float=1.0):
        super().__init__(_f101EO8)
        self._fOO0ElO = _fOO0ElO

    def _f01IEO9(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        diff = (_fOlOEOA - _flOIEOB).abs()
        quadratic = diff.clamp(max_val=self._fOO0ElO)
        linear = diff - quadratic
        return 0.5 * quadratic ** 2 + self._fOO0ElO * linear

class _c0l0Ell(_cO1IEO7):

    def __init__(self, _f101EO8: _c1IlEO6, _fI0OEl2: _cO1IEO7, _fIOIEl3: dict=None):
        super().__init__(_f101EO8)
        self._fI0OEl2 = _fI0OEl2
        self._fIOIEl3 = _fIOIEl3 or {}
        self._current_regime = None

    def _fOOIEl4(self, _fl01El5):
        self._current_regime = _fl01El5

    def _f01IEO9(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        base = self._fI0OEl2._f01IEO9(_fOlOEOA, _flOIEOB)
        if self._current_regime and self._current_regime in self._fIOIEl3:
            weight = self._fIOIEl3[self._current_regime]
            return base * weight
        return base

class _cOIOEl6(_cO1IEO7):

    def _f01IEO9(self, _fOlOEOA: Tensor, _flOIEOB: Tensor) -> Tensor:
        log_pred = _fOlOEOA.log()
        return _flOIEOB * (_flOIEOB.log() - log_pred)

class _cll0El7(_cO1IEO7):

    def __init__(self, _f101EO8: _c1IlEO6, _fOOIEl8: float=1.0):
        super().__init__(_f101EO8)
        self._fOOIEl8 = _fOOIEl8

    def _f01IEO9(self, _fl0OEl9: Tensor, _fl01ElA: Tensor, _f0l0ElB: Tensor=None) -> Tensor:
        pos_dist = (_fl0OEl9 - _fl01ElA).norm()
        if _f0l0ElB is not None:
            neg_dist = (_fl0OEl9 - _f0l0ElB).norm()
            loss = pos_dist + (self._fOOIEl8 - neg_dist).clamp(min_val=0)
        else:
            loss = pos_dist
        return loss