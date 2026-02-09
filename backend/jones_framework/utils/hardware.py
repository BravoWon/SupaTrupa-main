from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import os
import warnings

class _clI0AB7(Enum):
    CPU = auto()
    CUDA = auto()
    ROCM = auto()
    NPU = auto()
    METAL = auto()
    TPU = auto()
    AUTO = auto()

@dataclass
class _c01OAB8:
    device_type: _clI0AB7
    device_id: int
    name: str
    memory_gb: float
    compute_capability: str = ''
    is_available: bool = True

class _cl00AB9(ABC):

    @abstractmethod
    def _flO1ABA(self, _fI11ABB: np.ndarray) -> Any:
        pass

    @abstractmethod
    def _fO0IABc(self, _fI11ABB: Any) -> np.ndarray:
        pass

    @abstractmethod
    def _fI11ABd(self, _f1l0ABE: Any, _f1IIABf: Any) -> Any:
        pass

    @abstractmethod
    def _f1I0AcO(self, _fIIOAcl: str, *operands) -> Any:
        pass

    @abstractmethod
    def _fl0lAc2(self, _fI11ABB: Any, ord: Optional[int]=None) -> float:
        pass

class _clO0Ac3(_cl00AB9):

    def __init__(self):
        try:
            config = np.__config__.show()
            self._has_blas = 'openblas' in str(config).lower() or 'mkl' in str(config).lower()
        except:
            self._has_blas = False

    def _flO1ABA(self, _fI11ABB: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(_fI11ABB)

    def _fO0IABc(self, _fI11ABB: np.ndarray) -> np.ndarray:
        return _fI11ABB

    def _fI11ABd(self, _f1l0ABE: np.ndarray, _f1IIABf: np.ndarray) -> np.ndarray:
        return np._fI11ABd(_f1l0ABE, _f1IIABf)

    def _f1I0AcO(self, _fIIOAcl: str, *operands) -> np.ndarray:
        return np._f1I0AcO(_fIIOAcl, *operands, optimize=True)

    def _fl0lAc2(self, _fI11ABB: np.ndarray, ord: Optional[int]=None) -> float:
        return float(np.linalg._fl0lAc2(_fI11ABB, ord=ord))

class _cIIOAc4(_cl00AB9):

    def __init__(self, _f1IIAc5: int=0):
        import cupy as cp
        self.cp = cp
        self._f1IIAc5 = _f1IIAc5
        cp.cuda.Device(_f1IIAc5).use()

    def _flO1ABA(self, _fI11ABB: np.ndarray) -> Any:
        return self.cp.asarray(_fI11ABB)

    def _fO0IABc(self, _fI11ABB: Any) -> np.ndarray:
        return self.cp.asnumpy(_fI11ABB)

    def _fI11ABd(self, _f1l0ABE: Any, _f1IIABf: Any) -> Any:
        return self.cp._fI11ABd(_f1l0ABE, _f1IIABf)

    def _f1I0AcO(self, _fIIOAcl: str, *operands) -> Any:
        return self.cp._f1I0AcO(_fIIOAcl, *operands)

    def _fl0lAc2(self, _fI11ABB: Any, ord: Optional[int]=None) -> float:
        return float(self.cp.linalg._fl0lAc2(_fI11ABB, ord=ord))

class _cO0OAc6(_cl00AB9):

    def __init__(self, _f11IAc7: str='cuda'):
        import torch
        self.torch = torch
        self._f11IAc7 = torch._f11IAc7(_f11IAc7)

    def _flO1ABA(self, _fI11ABB: np.ndarray) -> Any:
        return self.torch.tensor(_fI11ABB, device=self._f11IAc7, dtype=self.torch.float32)

    def _fO0IABc(self, _fI11ABB: Any) -> np.ndarray:
        return _fI11ABB.cpu().numpy()

    def _fI11ABd(self, _f1l0ABE: Any, _f1IIABf: Any) -> Any:
        return self.torch._fI11ABd(_f1l0ABE, _f1IIABf)

    def _f1I0AcO(self, _fIIOAcl: str, *operands) -> Any:
        return self.torch._f1I0AcO(_fIIOAcl, *operands)

    def _fl0lAc2(self, _fI11ABB: Any, ord: Optional[int]=None) -> float:
        return float(self.torch.linalg._fl0lAc2(_fI11ABB))

class _clO1Ac8:

    def __init__(self, _f0I0Ac9: _clI0AB7=_clI0AB7.AUTO, _f1IIAc5: int=0, _flI0AcA: bool=True):
        self._f0I0Ac9 = _f0I0Ac9
        self._f1IIAc5 = _f1IIAc5
        self._flI0AcA = _flI0AcA
        self.backend: _cl00AB9 = _clO0Ac3()
        self.device_info: Optional[_c01OAB8] = None
        if _f0I0Ac9 == _clI0AB7.AUTO:
            self._auto_detect()
        else:
            self._init_backend(_f0I0Ac9)

    def _fOI1AcB(self):
        if self._try_cuda():
            return
        if self._try_rocm():
            return
        if self._try_metal():
            return
        if self._try_npu():
            return
        self._init_cpu()

    def _f1I1Acc(self) -> bool:
        try:
            import cupy as cp
            _f11IAc7 = cp.cuda.Device(self._f1IIAc5)
            props = cp.cuda.runtime.getDeviceProperties(self._f1IIAc5)
            self.backend = _cIIOAc4(self._f1IIAc5)
            self._f0I0Ac9 = _clI0AB7.CUDA
            self.device_info = _c01OAB8(device_type=_clI0AB7.CUDA, device_id=self._f1IIAc5, name=props['name'].decode(), memory_gb=props['totalGlobalMem'] / 1024 ** 3, compute_capability=f"{props['major']}.{props['minor']}")
            return True
        except:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                self.backend = _cO0OAc6('cuda')
                self._f0I0Ac9 = _clI0AB7.CUDA
                self.device_info = _c01OAB8(device_type=_clI0AB7.CUDA, device_id=self._f1IIAc5, name=torch.cuda.get_device_name(self._f1IIAc5), memory_gb=torch.cuda.get_device_properties(self._f1IIAc5).total_memory / 1024 ** 3)
                return True
        except:
            pass
        return False

    def _flO0Acd(self) -> bool:
        try:
            import torch
            if torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0):
                self.backend = _cO0OAc6('cuda')
                self._f0I0Ac9 = _clI0AB7.ROCM
                self.device_info = _c01OAB8(device_type=_clI0AB7.ROCM, device_id=self._f1IIAc5, name=torch.cuda.get_device_name(self._f1IIAc5), memory_gb=torch.cuda.get_device_properties(self._f1IIAc5).total_memory / 1024 ** 3)
                return True
        except:
            pass
        return False

    def _fIllAcE(self) -> bool:
        try:
            import torch
            if torch.backends.mps.is_available():
                self.backend = _cO0OAc6('mps')
                self._f0I0Ac9 = _clI0AB7.METAL
                self.device_info = _c01OAB8(device_type=_clI0AB7.METAL, device_id=0, name='Apple Metal GPU', memory_gb=0)
                return True
        except:
            pass
        return False

    def _fI0lAcf(self) -> bool:
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            if 'NPU' in devices or 'GPU' in devices:
                self._f0I0Ac9 = _clI0AB7.NPU
                self.device_info = _c01OAB8(device_type=_clI0AB7.NPU, device_id=0, name='Intel NPU', memory_gb=0)
                self._init_cpu()
                return True
        except:
            pass
        return False

    def _fIIOAdO(self):
        self.backend = _clO0Ac3()
        self._f0I0Ac9 = _clI0AB7.CPU
        self.device_info = _c01OAB8(device_type=_clI0AB7.CPU, device_id=0, name='CPU', memory_gb=0)

    def _fO0lAdl(self, _f0I0Ac9: _clI0AB7):
        if _f0I0Ac9 == _clI0AB7.CUDA:
            if not self._f1I1Acc() and self._flI0AcA:
                warnings.warn('CUDA not available, falling back to CPU')
                self._fIIOAdO()
        elif _f0I0Ac9 == _clI0AB7.METAL:
            if not self._fIllAcE() and self._flI0AcA:
                warnings.warn('Metal not available, falling back to CPU')
                self._fIIOAdO()
        else:
            self._fIIOAdO()

    @classmethod
    def _f11IAd2(cls) -> _clO1Ac8:
        return cls(device_type=_clI0AB7.AUTO)

    def _flO1ABA(self, _fI11ABB: np.ndarray) -> Any:
        return self.backend._flO1ABA(_fI11ABB)

    def _fO0IABc(self, _fI11ABB: Any) -> np.ndarray:
        return self.backend._fO0IABc(_fI11ABB)

    def _fI11ABd(self, _f1l0ABE: Any, _f1IIABf: Any) -> Any:
        return self.backend._fI11ABd(_f1l0ABE, _f1IIABf)

    def _f1I0AcO(self, _fIIOAcl: str, *operands) -> Any:
        return self.backend._f1I0AcO(_fIIOAcl, *operands)

    def _fl0lAc2(self, _fI11ABB: Any, ord: Optional[int]=None) -> float:
        return self.backend._fl0lAc2(_fI11ABB, ord)

    def _f1IIAd3(self, _f1l0ABE: Any, _f1IIABf: Any) -> Any:
        return self._f1I0AcO('bij,bjk->bik', _f1l0ABE, _f1IIABf)

    def __repr__(self) -> str:
        info = self.device_info
        if info:
            return f'HardwareAccelerator({info._f0I0Ac9.name}: {info.name})'
        return f'HardwareAccelerator({self._f0I0Ac9.name})'

def _flOOAd4(_fOOIAd5: str='auto') -> _clO1Ac8:
    type_map = {'auto': _clI0AB7.AUTO, 'cuda': _clI0AB7.CUDA, 'gpu': _clI0AB7.CUDA, 'rocm': _clI0AB7.ROCM, 'metal': _clI0AB7.METAL, 'mps': _clI0AB7.METAL, 'npu': _clI0AB7.NPU, 'cpu': _clI0AB7.CPU}
    _f0I0Ac9 = type_map.get(_fOOIAd5.lower(), _clI0AB7.AUTO)
    return _clO1Ac8(device_type=_f0I0Ac9)