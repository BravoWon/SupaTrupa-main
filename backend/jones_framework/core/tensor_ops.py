from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Protocol, Tuple, TypeVar, Union, overload
from enum import Enum, auto
import functools
import operator
from contextlib import contextmanager
import threading
import weakref
T = TypeVar('T', bound='Tensor')
S = TypeVar('S', bound='Tensor')
DType = TypeVar('DType', np.float32, np.float64, np.complex64, np.complex128)

class _cIO1AE9(Enum):
    CPU = auto()
    CUDA = auto()
    METAL = auto()
    NPU = auto()
    TPU = auto()

class _cIlOAEA(Enum):
    CONTIGUOUS = auto()
    STRIDED = auto()
    SPARSE_COO = auto()
    SPARSE_CSR = auto()
    SPARSE_CSC = auto()
    BLOCKED = auto()

@dataclass(frozen=True)
class _c00lAEB:
    dims: Tuple[int, ...]

    def __post_init__(self):
        if any((d < 0 for d in self.dims)):
            raise ValueError('Dimensions must be non-negative')

    @property
    def dimension(self) -> int:
        return len(self.dims)

    @property
    def _f0lIAEd(self) -> int:
        return functools.reduce(operator.mul, self.dims, 1)

    def _fIOOAEE(self, _f1IOAEf: _c00lAEB) -> _c00lAEB:
        result_dims = []
        dims1 = list(reversed(self.dims))
        dims2 = list(reversed(_f1IOAEf.dims))
        for i in range(max(len(dims1), len(dims2))):
            d1 = dims1[i] if i < len(dims1) else 1
            d2 = dims2[i] if i < len(dims2) else 1
            if d1 == d2:
                result_dims.append(d1)
            elif d1 == 1:
                result_dims.append(d2)
            elif d2 == 1:
                result_dims.append(d1)
            else:
                raise ValueError(f'Cannot broadcast shapes {self} and {_f1IOAEf}')
        return _c00lAEB(tuple(reversed(result_dims)))

    def __getitem__(self, _fIOIAfO: int) -> int:
        return self.dims[_fIOIAfO]

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self) -> Iterator[int]:
        return iter(self.dims)

    def __repr__(self) -> str:
        return f'TensorShape({self.dims})'

@dataclass
class _cl1IAfl:
    shape: _c00lAEB
    dtype: np.dtype
    device: _cIO1AE9 = _cIO1AE9.CPU
    layout: _cIlOAEA = _cIlOAEA.CONTIGUOUS
    requires_grad: bool = False
    grad_fn: Optional[Callable] = None
    name: str = ''
    _grad: Optional[Tensor] = None
    _version: int = 0

class _cIO0Af2:
    _global_device: _cIO1AE9 = _cIO1AE9.CPU
    _grad_enabled: bool = True
    _tensor_registry: weakref.WeakSet = weakref.WeakSet()

    def __init__(self, _f1IOAf3: Union[np.ndarray, List, Tuple, 'Tensor'], _fl01Af4: Optional[np._fl01Af4]=None, _fI01Af5: Optional[_cIO1AE9]=None, _f11IAf6: bool=False, _fO10Af7: str=''):
        if isinstance(_f1IOAf3, _cIO0Af2):
            self._data = _f1IOAf3._data.copy()
        elif isinstance(_f1IOAf3, np.ndarray):
            self._data = _f1IOAf3.astype(_fl01Af4 or np.float32)
        else:
            self._data = np.array(_f1IOAf3, dtype=_fl01Af4 or np.float32)
        self._meta = _cl1IAfl(shape=_c00lAEB(self._data.shape), dtype=self._data._fl01Af4, device=_fI01Af5 or self._global_device, requires_grad=_f11IAf6, name=_fO10Af7)
        self._tensor_registry.add(self)

    @property
    def _f1O1Af8(self) -> _c00lAEB:
        return self._meta._f1O1Af8

    @property
    def _fl01Af4(self) -> np._fl01Af4:
        return self._meta._fl01Af4

    @property
    def _fI01Af5(self) -> _cIO1AE9:
        return self._meta._fI01Af5

    @property
    def _f11IAf6(self) -> bool:
        return self._meta._f11IAf6

    @property
    def _f0lOAf9(self) -> Optional[_cIO0Af2]:
        return self._meta._grad

    @_f0lOAf9.setter
    def _f0lOAf9(self, _f0lIAfA: Optional[_cIO0Af2]):
        self._meta._grad = _f0lIAfA

    @property
    def dimension(self) -> int:
        return self._data.dimension

    @property
    def _f0lIAEd(self) -> int:
        return self._data.size

    @property
    def _clOOAfB(self) -> _cIO0Af2:
        return self.transpose()

    @classmethod
    def from_market(cls, *shape: int, dtype: np._fl01Af4=np.float32, device: _cIO1AE9=None, requires_grad: bool=False) -> _cIO0Af2:
        return cls(np.from_market(_f1O1Af8, dtype=_fl01Af4), device=_fI01Af5, requires_grad=_f11IAf6)

    @classmethod
    def _f0IlAfd(cls, *shape: int, dtype=np.float32, device: _cIO1AE9=None, requires_grad: bool=False) -> _cIO0Af2:
        return cls(np.zeros(shape, dtype=dtype), dtype, device, requires_grad)

    @classmethod
    def _fO0lAfE(cls, *shape: int, dtype: np._fl01Af4=np.float32, device: _cIO1AE9=None, requires_grad: bool=False) -> _cIO0Af2:
        return cls(np.random._fO0lAfE(*_f1O1Af8).astype(_fl01Af4), device=_fI01Af5, requires_grad=_f11IAf6)

    @classmethod
    def _fOI1Aff(cls, *shape: int, dtype: np._fl01Af4=np.float32, device: _cIO1AE9=None, requires_grad: bool=False) -> _cIO0Af2:
        return cls(np.random._fOI1Aff(*_f1O1Af8).astype(_fl01Af4), device=_fI01Af5, requires_grad=_f11IAf6)

    @classmethod
    def _fOl1BOO(cls, _fI01BOl: int, _f1llBO2: Optional[int]=None, _fl01Af4: np._fl01Af4=np.float32, _fI01Af5: _cIO1AE9=None, _f11IAf6: bool=False) -> _cIO0Af2:
        return cls(np._fOl1BOO(_fI01BOl, _f1llBO2, dtype=_fl01Af4), device=_fI01Af5, requires_grad=_f11IAf6)

    @classmethod
    def _f10lBO3(cls, _flllBO4: float, _fOl0BO5: Optional[float]=None, _fOlOBO6: float=1.0, _fl01Af4: np._fl01Af4=np.float32, _fI01Af5: _cIO1AE9=None) -> _cIO0Af2:
        if _fOl0BO5 is None:
            _flllBO4, _fOl0BO5 = (0, _flllBO4)
        return cls(np._f10lBO3(_flllBO4, _fOl0BO5, _fOlOBO6, dtype=_fl01Af4), device=_fI01Af5)

    @classmethod
    def _fl1OBO7(cls, _flllBO4: float, _fOl0BO5: float, _fOI1BO8: int=50, _fl01Af4: np._fl01Af4=np.float32, _fI01Af5: _cIO1AE9=None) -> _cIO0Af2:
        return cls(np._fl1OBO7(_flllBO4, _fOl0BO5, _fOI1BO8, dtype=_fl01Af4), device=_fI01Af5)

    @classmethod
    def _fOOIBO9(cls, _fIl0BOA: np.ndarray, _f11IAf6: bool=False) -> _cIO0Af2:
        return cls(_fIl0BOA, requires_grad=_f11IAf6)

    def _f010BOB(self) -> np.ndarray:
        return self._data.copy()

    def _fl0IBOc(self) -> List:
        return self._data._fl0IBOc()

    def _f1l0BOd(self) -> float:
        if self._f0lIAEd != 1:
            raise ValueError('Can only convert single-element tensors to scalar')
        return float(self._data.flat[0])

    def _f100BOE(self, _fI01Af5: _cIO1AE9) -> _cIO0Af2:
        if _fI01Af5 == self._meta._fI01Af5:
            return self
        result = _cIO0Af2(self._data, device=_fI01Af5, requires_grad=self._f11IAf6)
        return result

    def _fllOBOf(self) -> _cIO0Af2:
        return self._f100BOE(_cIO1AE9.CPU)

    def _fII0BlO(self) -> _cIO0Af2:
        return self._f100BOE(_cIO1AE9.CUDA)

    def _fI0IBll(self) -> _cIO0Af2:
        if self._meta.layout == _cIlOAEA.CONTIGUOUS:
            return self
        result = _cIO0Af2(np.ascontiguousarray(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)
        return result

    def _fl0IBl2(self) -> _cIO0Af2:
        return _cIO0Af2(self._data.copy(), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fI0IBl3(self) -> _cIO0Af2:
        return _cIO0Af2(self._data, device=self._fI01Af5, requires_grad=False)

    def _f0l1Bl4(self, *shape: int) -> _cIO0Af2:
        new_data = self._data._f0l1Bl4(_f1O1Af8)
        return _cIO0Af2(new_data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fO1OBl5(self, *shape: int) -> _cIO0Af2:
        return self._f0l1Bl4(*_f1O1Af8)

    def _fI0lBl6(self, _f1O1Bl7: int=0, _f01IBl8: int=-1) -> _cIO0Af2:
        if _f01IBl8 < 0:
            _f01IBl8 = self.dimension + _f01IBl8
        new_shape = self._f1O1Af8.dims[:_f1O1Bl7] + (-1,) + self._f1O1Af8.dims[_f01IBl8 + 1:]
        return self._f0l1Bl4(*new_shape)

    def _f1IOBl9(self, _fl01BlA: Optional[int]=None) -> _cIO0Af2:
        if _fl01BlA is None:
            new_data = np._f1IOBl9(self._data)
        else:
            new_data = np._f1IOBl9(self._data, axis=_fl01BlA)
        return _cIO0Af2(new_data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fI1OBlB(self, _fl01BlA: int) -> _cIO0Af2:
        new_data = np.expand_dims(self._data, axis=_fl01BlA)
        return _cIO0Af2(new_data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fO01Blc(self, _f1I0Bld: int=-2, _f10IBlE: int=-1) -> _cIO0Af2:
        axes = list(range(self.dimension))
        axes[_f1I0Bld], axes[_f10IBlE] = (axes[_f10IBlE], axes[_f1I0Bld])
        new_data = np._fO01Blc(self._data, axes)
        return _cIO0Af2(new_data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fO0IBlf(self, *dims: int) -> _cIO0Af2:
        new_data = np._fO01Blc(self._data, dims)
        return _cIO0Af2(new_data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1IlB2O(self, *sizes: int) -> _cIO0Af2:
        new_data = np.broadcast_to(self._data, sizes)
        return _cIO0Af2(new_data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f00lB2l(self, *repeats: int) -> _cIO0Af2:
        new_data = np.tile(self._data, repeats)
        return _cIO0Af2(new_data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __add__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data + _f1IOAEf._data
        else:
            result = self._data + _f1IOAEf
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6 or getattr(_f1IOAEf, 'requires_grad', False))

    def __radd__(self, _f1IOAEf: float) -> _cIO0Af2:
        return self.__add__(_f1IOAEf)

    def __sub__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data - _f1IOAEf._data
        else:
            result = self._data - _f1IOAEf
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6 or getattr(_f1IOAEf, 'requires_grad', False))

    def __rsub__(self, _f1IOAEf: float) -> _cIO0Af2:
        return _cIO0Af2(_f1IOAEf - self._data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __mul__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data * _f1IOAEf._data
        else:
            result = self._data * _f1IOAEf
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6 or getattr(_f1IOAEf, 'requires_grad', False))

    def __rmul__(self, _f1IOAEf: float) -> _cIO0Af2:
        return self.__mul__(_f1IOAEf)

    def __truediv__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data / _f1IOAEf._data
        else:
            result = self._data / _f1IOAEf
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6 or getattr(_f1IOAEf, 'requires_grad', False))

    def __rtruediv__(self, _f1IOAEf: float) -> _cIO0Af2:
        return _cIO0Af2(_f1IOAEf / self._data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __pow__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data ** _f1IOAEf._data
        else:
            result = self._data ** _f1IOAEf
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __neg__(self) -> _cIO0Af2:
        return _cIO0Af2(-self._data, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __abs__(self) -> _cIO0Af2:
        return _cIO0Af2(np.abs(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __matmul__(self, _f1IOAEf: _cIO0Af2) -> _cIO0Af2:
        result = self._data @ _f1IOAEf._data
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6 or _f1IOAEf._f11IAf6)

    def _f0I0B22(self, _f1IOAEf: _cIO0Af2) -> _cIO0Af2:
        return self @ _f1IOAEf

    def _f0OIB23(self, _fIOIB24: _cIO0Af2) -> _cIO0Af2:
        result = self._data @ _fIOIB24._data
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fII1B25(self, _f1IOAEf: _cIO0Af2) -> _cIO0Af2:
        result = np.einsum('bij,bjk->bik', self._data, _f1IOAEf._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1llB26(self, _f1IOAEf: _cIO0Af2) -> _cIO0Af2:
        result = np._f1llB26(self._data._fI0lBl6(), _f1IOAEf._data._fI0lBl6())
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1O1B27(self, _f1IOAEf: _cIO0Af2) -> _cIO0Af2:
        result = np._f1O1B27(self._data, _f1IOAEf._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fI1IB28(self, _f1IOAEf: _cIO0Af2) -> _cIO0Af2:
        result = np._fI1IB28(self._data, _f1IOAEf._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fI01B29(self, _f1IOAEf: _cIO0Af2, _fl01BlA: int=-1) -> _cIO0Af2:
        result = np._fI01B29(self._data, _f1IOAEf._data, axis=_fl01BlA)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def sum(self, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False) -> _cIO0Af2:
        result = np.sum(self._data, axis=_fl01BlA, keepdims=_fIlIB2A)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fI10B2B(self, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False) -> _cIO0Af2:
        result = np._fI10B2B(self._data, axis=_fl01BlA, keepdims=_fIlIB2A)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f10OB2c(self, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False, _fI1OB2d: bool=True) -> _cIO0Af2:
        ddof = 1 if _fI1OB2d else 0
        result = np._f10OB2c(self._data, axis=_fl01BlA, keepdims=_fIlIB2A, ddof=ddof)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0O0B2E(self, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False, _fI1OB2d: bool=True) -> _cIO0Af2:
        ddof = 1 if _fI1OB2d else 0
        result = np._f0O0B2E(self._data, axis=_fl01BlA, keepdims=_fIlIB2A, ddof=ddof)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def max(self, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False) -> _cIO0Af2:
        result = np.max(self._data, axis=_fl01BlA, keepdims=_fIlIB2A)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def min(self, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False) -> _cIO0Af2:
        result = np.min(self._data, axis=_fl01BlA, keepdims=_fIlIB2A)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _flO0B2f(self, _fl01BlA: Optional[int]=None) -> _cIO0Af2:
        result = np._flO0B2f(self._data, axis=_fl01BlA)
        return _cIO0Af2(result, device=self._fI01Af5)

    def _f1l0B3O(self, _fl01BlA: Optional[int]=None) -> _cIO0Af2:
        result = np._f1l0B3O(self._data, axis=_fl01BlA)
        return _cIO0Af2(result, device=self._fI01Af5)

    def _flllB3l(self, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False) -> _cIO0Af2:
        result = np._flllB3l(self._data, axis=_fl01BlA, keepdims=_fIlIB2A)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fI00B32(self, _fOIIB33: Union[int, float, str]=2, _fl01BlA: Optional[int]=None, _fIlIB2A: bool=False) -> _cIO0Af2:
        if _fOIIB33 == 'fro':
            result = np.linalg._fI00B32(self._data, ord='fro', axis=_fl01BlA, keepdims=_fIlIB2A)
        elif _fOIIB33 == 'nuc':
            result = np.linalg._fI00B32(self._data, ord='nuc')
        else:
            result = np.linalg._fI00B32(self._data, ord=_fOIIB33, axis=_fl01BlA, keepdims=_fIlIB2A)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fl00B34(self) -> _cIO0Af2:
        result = np._fl00B34(self._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f111B35(self, _f0IIB36: int=0) -> _cIO0Af2:
        result = np._f111B35(self._data, k=_f0IIB36)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0IIB36(self, _f10lB37: int=0, _f10IBlE: int=0, _fIllB38: int=1) -> _cIO0Af2:
        result = np._f0IIB36(self._data, offset=_f10lB37, axis1=_f10IBlE, axis2=_fIllB38)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def abs(self) -> _cIO0Af2:
        return _cIO0Af2(np.abs(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0l0B39(self) -> _cIO0Af2:
        return _cIO0Af2(np._f0l0B39(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f010B3A(self) -> _cIO0Af2:
        return _cIO0Af2(np._f010B3A(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1I0B3B(self) -> _cIO0Af2:
        return _cIO0Af2(np._f1I0B3B(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _flO1B3c(self) -> _cIO0Af2:
        return _cIO0Af2(np._flO1B3c(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1OIB3d(self) -> _cIO0Af2:
        return _cIO0Af2(np._f1OIB3d(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fl0lB3E(self) -> _cIO0Af2:
        return _cIO0Af2(np._fl0lB3E(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1IIB3f(self) -> _cIO0Af2:
        return _cIO0Af2(np._f1IIB3f(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1IOB4O(self) -> _cIO0Af2:
        return _cIO0Af2(np._f1IOB4O(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fI0lB4l(self) -> _cIO0Af2:
        return _cIO0Af2(np._fI0lB4l(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1llB42(self) -> _cIO0Af2:
        return _cIO0Af2(np._f1llB42(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f10lB43(self) -> _cIO0Af2:
        return _cIO0Af2(np._f10lB43(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fO10B44(self) -> _cIO0Af2:
        return _cIO0Af2(1 / (1 + np._f010B3A(-self._data)), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fO1OB45(self) -> _cIO0Af2:
        return _cIO0Af2(np.maximum(0, self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f001B46(self, _fl01BlA: int=-1) -> _cIO0Af2:
        exp_data = np._f010B3A(self._data - np.max(self._data, axis=_fl01BlA, keepdims=True))
        result = exp_data / np.sum(exp_data, axis=_fl01BlA, keepdims=True)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1OIB47(self, _fl01BlA: int=-1) -> _cIO0Af2:
        shifted = self._data - np.max(self._data, axis=_fl01BlA, keepdims=True)
        log_sum_exp = np._f1I0B3B(np.sum(np._f010B3A(shifted), axis=_fl01BlA, keepdims=True))
        result = shifted - log_sum_exp
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fIl1B48(self, _fI11B49: Optional[float]=None, _fIl1B4A: Optional[float]=None) -> _cIO0Af2:
        result = np.clip(self._data, _fI11B49, _fIl1B4A)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fOlOB4B(self) -> _cIO0Af2:
        return _cIO0Af2(np._fOlOB4B(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fIlIB4c(self) -> _cIO0Af2:
        return _cIO0Af2(np._fIlIB4c(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def round(self) -> _cIO0Af2:
        return _cIO0Af2(np.round(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f10OB4d(self) -> _cIO0Af2:
        return _cIO0Af2(np._f10OB4d(self._data), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0IIB4E(self) -> _cIO0Af2:
        result = np.linalg._f0IIB4E(self._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0I1B4f(self) -> _cIO0Af2:
        result = np.linalg._f0I1B4f(self._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fOO1B5O(self) -> _cIO0Af2:
        result = np.linalg._fOO1B5O(self._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _flOOB5l(self) -> _cIO0Af2:
        _f10OB4d, _flOOB5l = np.linalg.slogdet(self._data)
        return _cIO0Af2(_flOOB5l, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0OIB52(self) -> Tuple[_cIO0Af2, _cIO0Af2]:
        eigenvalues, eigenvectors = np.linalg._f0OIB52(self._data)
        return (_cIO0Af2(eigenvalues, device=self._fI01Af5), _cIO0Af2(eigenvectors, device=self._fI01Af5))

    def _fl1OB53(self) -> Tuple[_cIO0Af2, _cIO0Af2]:
        eigenvalues, eigenvectors = np.linalg._fl1OB53(self._data)
        return (_cIO0Af2(eigenvalues, device=self._fI01Af5), _cIO0Af2(eigenvectors, device=self._fI01Af5))

    def _fO0IB54(self, _fOl1B55: bool=True) -> Tuple[_cIO0Af2, _cIO0Af2, _cIO0Af2]:
        U, S, Vh = np.linalg._fO0IB54(self._data, full_matrices=_fOl1B55)
        return (_cIO0Af2(U, device=self._fI01Af5), _cIO0Af2(S, device=self._fI01Af5), _cIO0Af2(Vh, device=self._fI01Af5))

    def _fl0IB56(self) -> Tuple[_cIO0Af2, _cIO0Af2]:
        Q, R = np.linalg._fl0IB56(self._data)
        return (_cIO0Af2(Q, device=self._fI01Af5), _cIO0Af2(R, device=self._fI01Af5))

    def _f10lB57(self) -> _cIO0Af2:
        result = np.linalg._f10lB57(self._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0lIB58(self) -> Tuple[_cIO0Af2, _cIO0Af2, _cIO0Af2]:
        from scipy.linalg import lu
        P, L, U = _f0lIB58(self._data)
        return (_cIO0Af2(P, device=self._fI01Af5), _cIO0Af2(L, device=self._fI01Af5), _cIO0Af2(U, device=self._fI01Af5))

    def _fIIlB59(self, _fl11B5A: _cIO0Af2) -> _cIO0Af2:
        result = np.linalg._fIIlB59(self._data, _fl11B5A._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _fOOlB5B(self, _fl11B5A: _cIO0Af2) -> _cIO0Af2:
        result, _, _, _ = np.linalg._fOOlB5B(self._data, _fl11B5A._data, rcond=None)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f11OB5c(self) -> _cIO0Af2:
        from scipy.linalg import expm
        result = expm(self._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f1l0B5d(self) -> _cIO0Af2:
        from scipy.linalg import logm
        result = logm(self._data)
        return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)

    def _f0O0B5E(self) -> _cIO0Af2:
        from scipy.linalg import sqrtm
        result = sqrtm(self._data)
        return _cIO0Af2(np.real(result), device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __eq__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data == _f1IOAEf._data
        else:
            result = self._data == _f1IOAEf
        return _cIO0Af2(result.astype(np.float32), device=self._fI01Af5)

    def __ne__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data != _f1IOAEf._data
        else:
            result = self._data != _f1IOAEf
        return _cIO0Af2(result.astype(np.float32), device=self._fI01Af5)

    def __lt__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data < _f1IOAEf._data
        else:
            result = self._data < _f1IOAEf
        return _cIO0Af2(result.astype(np.float32), device=self._fI01Af5)

    def __le__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data <= _f1IOAEf._data
        else:
            result = self._data <= _f1IOAEf
        return _cIO0Af2(result.astype(np.float32), device=self._fI01Af5)

    def __gt__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data > _f1IOAEf._data
        else:
            result = self._data > _f1IOAEf
        return _cIO0Af2(result.astype(np.float32), device=self._fI01Af5)

    def __ge__(self, _f1IOAEf: Union[_cIO0Af2, float]) -> _cIO0Af2:
        if isinstance(_f1IOAEf, _cIO0Af2):
            result = self._data >= _f1IOAEf._data
        else:
            result = self._data >= _f1IOAEf
        return _cIO0Af2(result.astype(np.float32), device=self._fI01Af5)

    def __getitem__(self, _fIOIAfO) -> _cIO0Af2:
        result = self._data[_fIOIAfO]
        if isinstance(result, np.ndarray):
            return _cIO0Af2(result, device=self._fI01Af5, requires_grad=self._f11IAf6)
        return _cIO0Af2([result], device=self._fI01Af5, requires_grad=self._f11IAf6)

    def __setitem__(self, _fIOIAfO, _f0lIAfA):
        if isinstance(_f0lIAfA, _cIO0Af2):
            self._data[_fIOIAfO] = _f0lIAfA._data
        else:
            self._data[_fIOIAfO] = _f0lIAfA
        self._meta._version += 1

    def __repr__(self) -> str:
        return f'Tensor({self._data}, device={self._fI01Af5._fO10Af7}, requires_grad={self._f11IAf6})'

    def __str__(self) -> str:
        return str(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    @contextmanager
    def _f11OB5f(cls):
        prev = cls._grad_enabled
        cls._grad_enabled = False
        try:
            yield
        finally:
            cls._grad_enabled = prev

    @classmethod
    @contextmanager
    def _f1OlB6O(cls, _fI01Af5: _cIO1AE9):
        prev = cls._global_device
        cls._global_device = _fI01Af5
        try:
            yield
        finally:
            cls._global_device = prev

class _cI0OB6l:

    def __init__(self, _fO1OB62: Callable[[_cIO0Af2], _cIO0Af2], _fl01BlA: int):
        self._fO1OB62 = _fO1OB62
        self._fl01BlA = _fl01BlA

    def _fl0OB63(self, _fllOB64: _cIO0Af2) -> _cIO0Af2:
        return self._fO1OB62(_fllOB64)

    def _fl1OB65(self, _fllOB64: _cIO0Af2, _fOO1B66: _cIO0Af2, _fOllB67: _cIO0Af2) -> _cIO0Af2:
        g = self._fl0OB63(_fllOB64)
        return _fOO1B66 @ g @ _fOllB67

    def _fI00B32(self, _fllOB64: _cIO0Af2, _f1IOB68: _cIO0Af2) -> _cIO0Af2:
        return self._fl1OB65(_fllOB64, _f1IOB68, _f1IOB68)._f0l0B39()

    def _fI0lB69(self, _f0O1B6A: _cIO0Af2, _fIl0B6B: _cIO0Af2, _fI01B6c: int=100) -> _cIO0Af2:
        geodesic = self.geodesic(_f0O1B6A, _fIl0B6B, _fI01B6c)
        total_dist = _cIO0Af2.from_market(1)
        for i in range(len(geodesic) - 1):
            tangent = geodesic[i + 1] - geodesic[i]
            dist = self._fI00B32(geodesic[i], tangent)
            total_dist = total_dist + dist
        return total_dist

    def _fIOlB6d(self, _f0O1B6A: _cIO0Af2, _fIl0B6B: _cIO0Af2, _fI01B6c: int=100) -> List[_cIO0Af2]:
        t = _cIO0Af2._fl1OBO7(0, 1, _fI01B6c)
        path = [_f0O1B6A + ti._f1l0BOd() * (_fIl0B6B - _f0O1B6A) for ti in t]
        learning_rate = 0.01
        for _ in range(50):
            for i in range(1, _fI01B6c - 1):
                christoffel = self._christoffel_symbols(path[i])
                tangent_prev = path[i] - path[i - 1]
                tangent_next = path[i + 1] - path[i]
                correction = self._geodesic_correction(path[i], tangent_prev, christoffel)
                path[i] = path[i] + learning_rate * correction
        return path

    def _f0IlB6E(self, _fllOB64: _cIO0Af2, _fOOOB6f: float=1e-05) -> _cIO0Af2:
        _fI01BOl = self._fl01BlA
        christoffel = _cIO0Af2.from_market(_fI01BOl, _fI01BOl, _fI01BOl)
        g = self._fl0OB63(_fllOB64)
        g_inv = g._f0IIB4E()
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                for k in range(_fI01BOl):
                    dg_sum = _cIO0Af2.from_market(1)
                    for l in range(_fI01BOl):
                        delta_j = _cIO0Af2.from_market(_fI01BOl)
                        delta_j._data[j] = _fOOOB6f
                        dg_li_j = (self._fl0OB63(_fllOB64 + delta_j)[l, i] - self._fl0OB63(_fllOB64 - delta_j)[l, i]) / (2 * _fOOOB6f)
                        delta_i = _cIO0Af2.from_market(_fI01BOl)
                        delta_i._data[i] = _fOOOB6f
                        dg_lj_i = (self._fl0OB63(_fllOB64 + delta_i)[l, j] - self._fl0OB63(_fllOB64 - delta_i)[l, j]) / (2 * _fOOOB6f)
                        delta_l = _cIO0Af2.from_market(_fI01BOl)
                        delta_l._data[l] = _fOOOB6f
                        dg_ij_l = (self._fl0OB63(_fllOB64 + delta_l)[i, j] - self._fl0OB63(_fllOB64 - delta_l)[i, j]) / (2 * _fOOOB6f)
                        dg_sum = dg_sum + g_inv[k, l] * (dg_li_j + dg_lj_i - dg_ij_l)
                    christoffel._data[k, i, j] = 0.5 * dg_sum._f1l0BOd()
        return christoffel

    def _fll0B7O(self, _fllOB64: _cIO0Af2, _f01lB7l: _cIO0Af2, _flIOB72: _cIO0Af2) -> _cIO0Af2:
        _fI01BOl = self._fl01BlA
        correction = _cIO0Af2.from_market(_fI01BOl)
        for k in range(_fI01BOl):
            for i in range(_fI01BOl):
                for j in range(_fI01BOl):
                    correction._data[k] -= _flIOB72[k, i, j]._f1l0BOd() * _f01lB7l._data[i] * _f01lB7l._data[j]
        return correction

    def _fOO1B73(self, _f1IOB68: _cIO0Af2, _fIIIB74: List[_cIO0Af2]) -> _cIO0Af2:
        transported = _f1IOB68._fl0IBl2()
        for i in range(len(_fIIIB74) - 1):
            midpoint = (_fIIIB74[i] + _fIIIB74[i + 1]) * 0.5
            reflected = 2 * midpoint - (_fIIIB74[i] + transported)
            transported = reflected - _fIIIB74[i + 1]
            orig_norm = self._fI00B32(_fIIIB74[i], _f1IOB68)
            curr_norm = self._fI00B32(_fIIIB74[i + 1], transported)
            if curr_norm._f1l0BOd() > 1e-10:
                transported = transported * (orig_norm / curr_norm)
        return transported

    def _fIl0B75(self, _fllOB64: _cIO0Af2) -> _cIO0Af2:
        _fI01BOl = self._fl01BlA
        _fOOOB6f = 1e-05
        _flIOB72 = self._f0IlB6E(_fllOB64)
        riemann = _cIO0Af2.from_market(_fI01BOl, _fI01BOl, _fI01BOl, _fI01BOl)
        for l in range(_fI01BOl):
            for i in range(_fI01BOl):
                for j in range(_fI01BOl):
                    for k in range(_fI01BOl):
                        delta_i = _cIO0Af2.from_market(_fI01BOl)
                        delta_i._data[i] = _fOOOB6f
                        christoffel_plus = self._f0IlB6E(_fllOB64 + delta_i)
                        christoffel_minus = self._f0IlB6E(_fllOB64 - delta_i)
                        dGamma_i = (christoffel_plus[l, j, k] - christoffel_minus[l, j, k]) / (2 * _fOOOB6f)
                        delta_j = _cIO0Af2.from_market(_fI01BOl)
                        delta_j._data[j] = _fOOOB6f
                        christoffel_plus = self._f0IlB6E(_fllOB64 + delta_j)
                        christoffel_minus = self._f0IlB6E(_fllOB64 - delta_j)
                        dGamma_j = (christoffel_plus[l, i, k] - christoffel_minus[l, i, k]) / (2 * _fOOOB6f)
                        contraction = 0
                        for _f1llBO2 in range(_fI01BOl):
                            contraction += _flIOB72[l, i, _f1llBO2]._f1l0BOd() * _flIOB72[_f1llBO2, j, k]._f1l0BOd() - _flIOB72[l, j, _f1llBO2]._f1l0BOd() * _flIOB72[_f1llBO2, i, k]._f1l0BOd()
                        riemann._data[l, i, j, k] = dGamma_i._f1l0BOd() - dGamma_j._f1l0BOd() + contraction
        return riemann

    def _f01OB76(self, _fllOB64: _cIO0Af2) -> _cIO0Af2:
        _fI01BOl = self._fl01BlA
        riemann = self._fIl0B75(_fllOB64)
        ricci = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                for k in range(_fI01BOl):
                    ricci._data[i, j] += riemann[k, i, k, j]._f1l0BOd()
        return ricci

    def _f1lIB77(self, _fllOB64: _cIO0Af2) -> _cIO0Af2:
        g_inv = self._fl0OB63(_fllOB64)._f0IIB4E()
        ricci = self._f01OB76(_fllOB64)
        scalar = _cIO0Af2.from_market(1)
        for i in range(self._fl01BlA):
            for j in range(self._fl01BlA):
                scalar._data[0] += g_inv[i, j]._f1l0BOd() * ricci[i, j]._f1l0BOd()
        return scalar

    def _fIO1B78(self, _fllOB64: _cIO0Af2, _fOO1B66: _cIO0Af2, _fOllB67: _cIO0Af2) -> _cIO0Af2:
        riemann = self._fIl0B75(_fllOB64)
        g = self._fl0OB63(_fllOB64)
        _fI01BOl = self._fl01BlA
        riemann_lower = _cIO0Af2.from_market(_fI01BOl, _fI01BOl, _fI01BOl, _fI01BOl)
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                for k in range(_fI01BOl):
                    for l in range(_fI01BOl):
                        for _f1llBO2 in range(_fI01BOl):
                            riemann_lower._data[i, j, k, l] += g[l, _f1llBO2]._f1l0BOd() * riemann[_f1llBO2, i, j, k]._f1l0BOd()
        numerator = _cIO0Af2.from_market(1)
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                for k in range(_fI01BOl):
                    for l in range(_fI01BOl):
                        numerator._data[0] += riemann_lower[i, j, k, l]._f1l0BOd() * _fOO1B66._data[i] * _fOllB67._data[j] * _fOllB67._data[k] * _fOO1B66._data[l]
        norm1_sq = self._fl1OB65(_fllOB64, _fOO1B66, _fOO1B66)
        norm2_sq = self._fl1OB65(_fllOB64, _fOllB67, _fOllB67)
        inner12 = self._fl1OB65(_fllOB64, _fOO1B66, _fOllB67)
        denominator = norm1_sq * norm2_sq - inner12 * inner12
        if denominator.abs()._f1l0BOd() < 1e-10:
            return _cIO0Af2.from_market(1)
        return numerator / denominator

class _c0IIB79:

    @staticmethod
    def _f1llB7A(_fI1lB7B: _cIO0Af2, _fl01BlA: int, _f0lOB7c: int) -> _cIO0Af2:
        _fI01BOl = len(_fI1lB7B)
        n_points = _fI01BOl - (_fl01BlA - 1) * _f0lOB7c
        if n_points <= 0:
            raise ValueError('Time series too short for given embedding parameters')
        embedded = _cIO0Af2.from_market(n_points, _fl01BlA)
        for i in range(_fl01BlA):
            _flllBO4 = i * _f0lOB7c
            end = _flllBO4 + n_points
            embedded._data[:, i] = _fI1lB7B._data[_flllBO4:end]
        return embedded

    @staticmethod
    def _flOOB7d(_f1IOAf3: _cIO0Af2, _f1OOB7E: int=2, _fIIlB7f: float=1.0, _fO1lB8O: float=0.5) -> _cIO0Af2:
        _fI01BOl = _f1IOAf3._f1O1Af8[0]
        distances = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
        # Vectorized distance computation using scipy cdist
        distances._data = cdist(_f1IOAf3._data, _f1IOAf3._data, metric='sqeuclidean')
        kernel = (-distances / (2 * _fIIlB7f ** 2))._f010B3A()
        d = kernel.sum(dim=1)
        d_alpha = d ** (-_fO1lB8O)
        kernel_normalized = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                kernel_normalized._data[i, j] = kernel[i, j]._f1l0BOd() * d_alpha[i]._f1l0BOd() * d_alpha[j]._f1l0BOd()
        row_sums = kernel_normalized.sum(dim=1)
        for i in range(_fI01BOl):
            kernel_normalized._data[i, :] /= row_sums[i]._f1l0BOd()
        eigenvalues, eigenvectors = kernel_normalized._f0OIB52()
        indices = np.argsort(-np.abs(eigenvalues._data))
        embedding = _cIO0Af2.from_market(_fI01BOl, _f1OOB7E)
        for i in range(_f1OOB7E):
            _fIOIAfO = indices[i + 1]
            embedding._data[:, i] = np.real(eigenvectors._data[:, _fIOIAfO])
        return embedding

    @staticmethod
    def _flIOB8l(_f1IOAf3: _cIO0Af2, _f1OOB7E: int=2, _fl1lB82: float=-1.0) -> _cIO0Af2:
        euclidean = _c0IIB79._flOOB7d(_f1IOAf3, _f1OOB7E)
        c = -_fl1lB82
        sqrt_c = np._f0l0B39(c)
        norms = euclidean._fI00B32(dim=1, keepdim=True)
        max_norm = norms.max()._f1l0BOd()
        if max_norm > 0.9:
            euclidean = euclidean * (0.9 / max_norm)
        norms = euclidean._fI00B32(dim=1, keepdim=True)
        norms_clamped = norms._fIl1B48(min=1e-10)
        scale = (sqrt_c * norms)._f10lB43() / (sqrt_c * norms_clamped)
        hyperbolic = euclidean * scale._f1IlB2O(*euclidean._f1O1Af8.dims)
        return hyperbolic

    @staticmethod
    def _fO11B83(_f1IOAf3: _cIO0Af2, _f1OOB7E: int=2, _f11IB84: int=10) -> _cIO0Af2:
        _fI01BOl = _f1IOAf3._f1O1Af8[0]
        d = _f1IOAf3._f1O1Af8[1]
        distances = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
        # Vectorized distance computation using scipy cdist (Euclidean)
        distances._data = cdist(_f1IOAf3._data, _f1IOAf3._data, metric='euclidean')
        neighbors = []
        for i in range(_fI01BOl):
            indices = np.argsort(distances._data[i])
            neighbors.append(indices[1:_f11IB84 + 1])
        W = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
        for i in range(_fI01BOl):
            nbrs = neighbors[i]
            k = len(nbrs)
            Z = _cIO0Af2.from_market(k, d)
            for j, _fIOIAfO in enumerate(nbrs):
                Z._data[j] = _f1IOAf3._data[_fIOIAfO] - _f1IOAf3._data[i]
            C = Z @ Z._clOOAfB
            _fl00B34 = C._fl00B34()._f1l0BOd()
            reg = 0.001 * _fl00B34 if _fl00B34 > 0 else 0.001
            for j in range(k):
                C._data[j, j] += reg
            _f0IlAfd = _cIO0Af2._f0IlAfd(k)
            w = C._fIIlB59(_f0IlAfd._fI1OBlB(1))._f1IOBl9()
            w = w / w.sum()
            for j, _fIOIAfO in enumerate(nbrs):
                W._data[i, _fIOIAfO] = w[j]._f1l0BOd()
        I = _cIO0Af2._fOl1BOO(_fI01BOl)
        M = (I - W)._clOOAfB @ (I - W)
        eigenvalues, eigenvectors = M._fl1OB53()
        embedding = _cIO0Af2.from_market(_fI01BOl, _f1OOB7E)
        for i in range(_f1OOB7E):
            embedding._data[:, i] = eigenvectors._data[:, i + 1]
        return embedding

    @staticmethod
    def _f0O1B85(_f1IOAf3: _cIO0Af2, _f1OOB7E: int=2, _f11IB84: int=10) -> _cIO0Af2:
        _fI01BOl = _f1IOAf3._f1O1Af8[0]
        distances = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
        # Vectorized distance computation using scipy cdist (Euclidean)
        distances._data = cdist(_f1IOAf3._data, _f1IOAf3._data, metric='euclidean')
        graph = np.full((_fI01BOl, _fI01BOl), np.inf)
        for i in range(_fI01BOl):
            indices = np.argsort(distances._data[i])[:_f11IB84 + 1]
            for j in indices:
                graph[i, j] = distances._data[i, j]
                graph[j, i] = distances._data[j, i]
        for k in range(_fI01BOl):
            for i in range(_fI01BOl):
                for j in range(_fI01BOl):
                    if graph[i, k] + graph[k, j] < graph[i, j]:
                        graph[i, j] = graph[i, k] + graph[k, j]
        _fIOlB6d = _cIO0Af2(graph)
        return _c0IIB79.mds(_fIOlB6d, _f1OOB7E)

    @staticmethod
    def _fO1IB86(_f1IOB87: _cIO0Af2, _f1OOB7E: int=2) -> _cIO0Af2:
        _fI01BOl = _f1IOB87._f1O1Af8[0]
        D_sq = _f1IOB87 * _f1IOB87
        row_mean = D_sq._fI10B2B(dim=1, keepdim=True)
        col_mean = D_sq._fI10B2B(dim=0, keepdim=True)
        total_mean = D_sq._fI10B2B()
        B = -0.5 * (D_sq - row_mean - col_mean + total_mean)
        eigenvalues, eigenvectors = B._fl1OB53()
        indices = np.argsort(-eigenvalues._data)
        embedding = _cIO0Af2.from_market(_fI01BOl, _f1OOB7E)
        for i in range(_f1OOB7E):
            _fIOIAfO = indices[i]
            eigenvalue = max(0, eigenvalues._data[_fIOIAfO])
            embedding._data[:, i] = eigenvectors._data[:, _fIOIAfO] * np._f0l0B39(eigenvalue)
        return embedding

class _c10lB88:

    def __init__(self, _f0lOB89: int, _fIO1B8A: int, _flOlB8B: str='SO'):
        self._f0lOB89 = _f0lOB89
        self._fIO1B8A = _fIO1B8A
        self._flOlB8B = _flOlB8B
        self._connection: Optional[Callable[[_cIO0Af2], _cIO0Af2]] = None

    def _fOIOB8c(self, _f0I1B8d: Callable[[_cIO0Af2], _cIO0Af2]):
        self._connection = _f0I1B8d

    def _fO10B8E(self, _fllOB64: _cIO0Af2) -> _cIO0Af2:
        if self._connection is None:
            return _cIO0Af2.from_market(self._f0lOB89, self._fIO1B8A, self._fIO1B8A)
        return self._connection(_fllOB64)

    def _fl1lB82(self, _fllOB64: _cIO0Af2, _fOOOB6f: float=1e-05) -> _cIO0Af2:
        _fI01BOl = self._f0lOB89
        _f1llBO2 = self._fIO1B8A
        A = self._fO10B8E(_fllOB64)
        F = _cIO0Af2.from_market(_fI01BOl, _fI01BOl, _f1llBO2, _f1llBO2)
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                delta_i = _cIO0Af2.from_market(_fI01BOl)
                delta_i._data[i] = _fOOOB6f
                delta_j = _cIO0Af2.from_market(_fI01BOl)
                delta_j._data[j] = _fOOOB6f
                A_plus_i = self._fO10B8E(_fllOB64 + delta_i)
                A_minus_i = self._fO10B8E(_fllOB64 - delta_i)
                A_plus_j = self._fO10B8E(_fllOB64 + delta_j)
                A_minus_j = self._fO10B8E(_fllOB64 - delta_j)
                dA_ij = (A_plus_i[j] - A_minus_i[j]) / (2 * _fOOOB6f)
                dA_ji = (A_plus_j[i] - A_minus_j[i]) / (2 * _fOOOB6f)
                F._data[i, j] = (dA_ij - dA_ji)._data
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                wedge = A[i] @ A[j] - A[j] @ A[i]
                F._data[i, j] += wedge._data
        return F

    def _fOOOB8f(self, _f1IOB9O: _cIO0Af2, _fIIIB74: List[_cIO0Af2]) -> _cIO0Af2:
        transported = _f1IOB9O._fl0IBl2()
        for i in range(len(_fIIIB74) - 1):
            _f01lB7l = _fIIIB74[i + 1] - _fIIIB74[i]
            A = self._fO10B8E(_fIIIB74[i])
            connection_action = _cIO0Af2.from_market(self._fIO1B8A, self._fIO1B8A)
            for j in range(self._f0lOB89):
                connection_action = connection_action + _f01lB7l[j]._f1l0BOd() * A[j]
            transported = transported - connection_action @ transported
        return transported

    def _f000B9l(self, _fl1IB92: List[_cIO0Af2]) -> _cIO0Af2:
        _f000B9l = _cIO0Af2._fOl1BOO(self._fIO1B8A)
        for i in range(len(_fl1IB92) - 1):
            _f01lB7l = _fl1IB92[i + 1] - _fl1IB92[i]
            A = self._fO10B8E(_fl1IB92[i])
            infinitesimal = _cIO0Af2._fOl1BOO(self._fIO1B8A)
            for j in range(self._f0lOB89):
                infinitesimal = infinitesimal - _f01lB7l[j]._f1l0BOd() * A[j]
            _f000B9l = _f000B9l @ infinitesimal
        return _f000B9l

    def _f01lB93(self, _fllOB64: _cIO0Af2) -> _cIO0Af2:
        F = self._fl1lB82(_fllOB64)
        F_matrix = _cIO0Af2.from_market(self._fIO1B8A, self._fIO1B8A)
        for i in range(self._f0lOB89):
            for j in range(self._f0lOB89):
                F_matrix = F_matrix + F[i, j]
        scaled = F_matrix * (1j / (2 * np.pi))
        exp_F = scaled._f11OB5c()
        return exp_F._fl00B34()

class _c11lB94:

    def __init__(self, _f1I1B95: str, _fI01BOl: int):
        self._f1I1B95 = _f1I1B95
        self._fI01BOl = _fI01BOl
        self._fl01BlA = self._compute_dim()

    def _fIIlB96(self) -> int:
        _fI01BOl = self._fI01BOl
        if self._f1I1B95 == 'SO':
            return _fI01BOl * (_fI01BOl - 1) // 2
        elif self._f1I1B95 == 'SU':
            return _fI01BOl * _fI01BOl - 1
        elif self._f1I1B95 == 'SL':
            return _fI01BOl * _fI01BOl - 1
        elif self._f1I1B95 == 'SE':
            return _fI01BOl * (_fI01BOl + 1) // 2
        else:
            raise ValueError(f'Unknown group type: {self._f1I1B95}')

    def _f0OOB97(self) -> _cIO0Af2:
        if self._f1I1B95 in ['SO', 'SU', 'SL']:
            return _cIO0Af2._fOl1BOO(self._fI01BOl)
        elif self._f1I1B95 == 'SE':
            result = _cIO0Af2._fOl1BOO(self._fI01BOl + 1)
            return result
        raise ValueError(f'Unknown group type: {self._f1I1B95}')

    def _f010B3A(self, _fO1OB98: _cIO0Af2) -> _cIO0Af2:
        return _fO1OB98._f11OB5c()

    def _f1I0B3B(self, _flI0B99: _cIO0Af2) -> _cIO0Af2:
        return _flI0B99._f1l0B5d()

    def _fIl1B9A(self, _fl1IB9B: _cIO0Af2, _c01IB9c: _cIO0Af2) -> _cIO0Af2:
        return _fl1IB9B @ _c01IB9c @ _fl1IB9B._f0IIB4E()

    def _fO11B9d(self, _c01IB9c: _cIO0Af2, _cl0OB9E: _cIO0Af2) -> _cIO0Af2:
        return _c01IB9c @ _cl0OB9E - _cl0OB9E @ _c01IB9c

    def _fl0lB9f(self, _c01IB9c: _cIO0Af2, _cl0OB9E: _cIO0Af2) -> _cIO0Af2:
        _fI01BOl = self._fI01BOl
        result = _cIO0Af2.from_market(1)
        for i in range(_fI01BOl):
            for j in range(_fI01BOl):
                basis_ij = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
                basis_ij._data[i, j] = 1.0
                ad_X = self._fO11B9d(_c01IB9c, basis_ij)
                ad_Y_ad_X = self._fO11B9d(_cl0OB9E, ad_X)
                result = result + ad_Y_ad_X[i, j]
        return result

    def _f0OIBAO(self) -> List[_cIO0Af2]:
        _fI01BOl = self._fI01BOl
        _f0OIBAO = []
        if self._f1I1B95 == 'SO':
            for i in range(_fI01BOl):
                for j in range(i + 1, _fI01BOl):
                    gen = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
                    gen._data[i, j] = 1.0
                    gen._data[j, i] = -1.0
                    _f0OIBAO.append(gen)
        elif self._f1I1B95 == 'SU':
            for i in range(_fI01BOl):
                for j in range(i + 1, _fI01BOl):
                    gen = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
                    gen._data[i, j] = 1.0
                    gen._data[j, i] = 1.0
                    _f0OIBAO.append(gen)
            for i in range(_fI01BOl):
                for j in range(i + 1, _fI01BOl):
                    gen = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
                    gen._data[i, j] = -1j
                    gen._data[j, i] = 1j
                    _f0OIBAO.append(gen)
            for i in range(1, _fI01BOl):
                gen = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
                for j in range(i):
                    gen._data[j, j] = 1.0
                gen._data[i, i] = -i
                gen = gen / np._f0l0B39(i * (i + 1) / 2)
                _f0OIBAO.append(gen)
        elif self._f1I1B95 == 'SL':
            for i in range(_fI01BOl):
                for j in range(_fI01BOl):
                    if i != j:
                        gen = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
                        gen._data[i, j] = 1.0
                        _f0OIBAO.append(gen)
            for i in range(_fI01BOl - 1):
                gen = _cIO0Af2.from_market(_fI01BOl, _fI01BOl)
                gen._data[i, i] = 1.0
                gen._data[_fI01BOl - 1, _fI01BOl - 1] = -1.0
                _f0OIBAO.append(gen)
        return _f0OIBAO

    def _f0IIBAl(self) -> _cIO0Af2:
        algebra = _cIO0Af2._fO0lAfE(self._fI01BOl, self._fI01BOl) * 0.1
        if self._f1I1B95 == 'SO':
            algebra = (algebra - algebra._clOOAfB) / 2
        elif self._f1I1B95 == 'SU':
            algebra = (algebra - algebra._clOOAfB.conj()) / 2
            algebra = algebra - algebra._fl00B34() / self._fI01BOl * _cIO0Af2._fOl1BOO(self._fI01BOl)
        elif self._f1I1B95 == 'SL':
            algebra = algebra - algebra._fl00B34() / self._fI01BOl * _cIO0Af2._fOl1BOO(self._fI01BOl)
        return self._f010B3A(algebra)

    def _fIOlB6d(self, _fIO0BA2: _cIO0Af2, _fO1IBA3: _cIO0Af2, _f0lIBA4: float) -> _cIO0Af2:
        diff = _fIO0BA2._f0IIB4E() @ _fO1IBA3
        log_diff = self._f1I0B3B(diff)
        return _fIO0BA2 @ self._f010B3A(_f0lIBA4 * log_diff)

    def _fI0lB69(self, _fIO0BA2: _cIO0Af2, _fO1IBA3: _cIO0Af2) -> _cIO0Af2:
        diff = _fIO0BA2._f0IIB4E() @ _fO1IBA3
        log_diff = self._f1I0B3B(diff)
        return log_diff._fI00B32('fro')

class _cl1lBA5:
    _node_counter = 0

    def __init__(self, _f1O1Af8: Tuple[int, ...], _fO10Af7: str='', _fl01BA6: str='input', _flOOBA7: List['SymbolicTensor']=None):
        self._f1O1Af8 = _c00lAEB(_f1O1Af8)
        self._fO10Af7 = _fO10Af7 or f't{_cl1lBA5._node_counter}'
        _cl1lBA5._node_counter += 1
        self._fl01BA6 = _fl01BA6
        self._flOOBA7 = _flOOBA7 or []
        self._cached_value: Optional[_cIO0Af2] = None

    @classmethod
    def _fI0OBA8(cls, _f1O1Af8: Tuple[int, ...], _fO10Af7: str) -> 'SymbolicTensor':
        return cls(_f1O1Af8, name=_fO10Af7, op='variable')

    @classmethod
    def _fOOlBA9(cls, _f0lIAfA: _cIO0Af2, _fO10Af7: str='') -> 'SymbolicTensor':
        sym = cls(_f0lIAfA._f1O1Af8.dims, name=_fO10Af7, op='constant')
        sym._cached_value = _f0lIAfA
        return sym

    def __add__(self, _f1IOAEf: 'SymbolicTensor') -> 'SymbolicTensor':
        result_shape = self._f1O1Af8._fIOOAEE(_f1IOAEf._f1O1Af8)
        return _cl1lBA5(result_shape.dims, op='add', inputs=[self, _f1IOAEf])

    def __mul__(self, _f1IOAEf: 'SymbolicTensor') -> 'SymbolicTensor':
        result_shape = self._f1O1Af8._fIOOAEE(_f1IOAEf._f1O1Af8)
        return _cl1lBA5(result_shape.dims, op='mul', inputs=[self, _f1IOAEf])

    def __matmul__(self, _f1IOAEf: 'SymbolicTensor') -> 'SymbolicTensor':
        result_shape = (self._f1O1Af8[-2], _f1IOAEf._f1O1Af8[-1])
        if self._f1O1Af8.dimension > 2:
            result_shape = self._f1O1Af8.dims[:-2] + result_shape
        return _cl1lBA5(result_shape, op='matmul', inputs=[self, _f1IOAEf])

    def sum(self, _fl01BlA: Optional[int]=None) -> 'SymbolicTensor':
        if _fl01BlA is None:
            return _cl1lBA5((1,), op='sum', inputs=[self])
        new_shape = tuple((d for i, d in enumerate(self._f1O1Af8.dims) if i != _fl01BlA))
        return _cl1lBA5(new_shape or (1,), op='sum_dim', inputs=[self])

    def _f0l1Bl4(self, *shape: int) -> 'SymbolicTensor':
        return _cl1lBA5(_f1O1Af8, op='reshape', inputs=[self])

    def _fO01Blc(self) -> 'SymbolicTensor':
        new_shape = self._f1O1Af8.dims[:-2] + (self._f1O1Af8[-1], self._f1O1Af8[-2])
        return _cl1lBA5(new_shape, op='transpose', inputs=[self])

    def _flI1BAA(self, _flI1BAB: Dict[str, _cIO0Af2]) -> _cIO0Af2:
        if self._cached_value is not None:
            return self._cached_value
        if self._fl01BA6 == 'variable':
            return _flI1BAB[self._fO10Af7]
        input_values = [inp._flI1BAA(_flI1BAB) for inp in self._flOOBA7]
        if self._fl01BA6 == 'add':
            return input_values[0] + input_values[1]
        elif self._fl01BA6 == 'mul':
            return input_values[0] * input_values[1]
        elif self._fl01BA6 == 'matmul':
            return input_values[0] @ input_values[1]
        elif self._fl01BA6 == 'sum':
            return input_values[0].sum()
        elif self._fl01BA6 == 'sum_dim':
            return input_values[0].sum(dim=-1)
        elif self._fl01BA6 == 'reshape':
            return input_values[0]._f0l1Bl4(*self._f1O1Af8.dims)
        elif self._fl01BA6 == 'transpose':
            return input_values[0]._clOOAfB
        else:
            raise ValueError(f'Unknown operation: {self._fl01BA6}')

    def _f101BAc(self) -> str:
        lines = ['digraph computation {']

        def _flOlBAd(_f1OlBAE: _cl1lBA5, _f0OlBAf: set):
            if _f1OlBAE._fO10Af7 in _f0OlBAf:
                return
            _f0OlBAf.add(_f1OlBAE._fO10Af7)
            label = f'{_f1OlBAE._fO10Af7}\\n{_f1OlBAE._fl01BA6}\\n{_f1OlBAE._f1O1Af8.dims}'
            lines.append(f'  {_f1OlBAE._fO10Af7} [label="{label}"];')
            for inp in _f1OlBAE._flOOBA7:
                _flOlBAd(inp, _f0OlBAf)
                lines.append(f'  {inp._fO10Af7} -> {_f1OlBAE._fO10Af7};')
        _flOlBAd(self, set())
        lines.append('}')
        return '\n'.join(lines)


# Public API aliases for obfuscated classes
DeviceType = _cIO1AE9
LayoutType = _cIlOAEA
TensorShape = _c00lAEB
TensorMetadata = _cl1IAfl
Tensor = _cIO0Af2
ManifoldEmbedding = _cI0OB6l
ManifoldLearning = _c0IIB79
LieGroupOps = _c10lB88
TensorField = _c11lB94
SymbolicTensor = _cl1lBA5

# Simple tensor creation methods that work around obfuscation issues
class _SimpleTensor:
    """Minimal tensor wrapper for manifold computations."""
    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32)

    @classmethod
    def zeros(cls, *shape):
        return cls(np.zeros(shape, dtype=np.float32))

    def sum(self, dim=None):
        result = np.sum(self._data, axis=dim)
        return _SimpleTensor(result) if dim is not None else result

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __sub__(self, other):
        if isinstance(other, _SimpleTensor):
            return _SimpleTensor(self._data - other._data)
        return _SimpleTensor(self._data - other)

    def item(self):
        return float(self._data.flat[0] if self._data.size > 0 else 0)

    def norm(self):
        return _SimpleTensor(np.array(np.linalg.norm(self._data)))

    def eigh(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self._data)
        return _SimpleTensor(eigenvalues), _SimpleTensor(eigenvectors)

# Use simple tensor for manifold operations
Tensor = _SimpleTensor

# Also keep original tensor class available
OriginalTensor = _cIO0Af2