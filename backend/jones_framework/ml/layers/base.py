from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from jones_framework.core.tensor_ops import Tensor
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.ml.networks.base import Initializer, Activation

@dataclass
class _c0OIEcB:
    name: str
    input_dim: int
    output_dim: int
    activation: str = 'relu'
    use_bias: bool = True
    dropout: float = 0.0

@bridge(connects_to=['Tensor'], connection_types={'Tensor': ConnectionType.USES})
class _c10lEcc(ABC):

    def __init__(self, _fl10Ecd: _c0OIEcB):
        self._fl10Ecd = _fl10Ecd
        self._training = True

    @abstractmethod
    def _flOOEcE(self, _f1O1Ecf: Tensor) -> Tensor:
        pass

    def __call__(self, _f1O1Ecf: Tensor) -> Tensor:
        return self._flOOEcE(_f1O1Ecf)

    def _fIO1EdO(self, _f1lIEdl: bool=True):
        self._training = _f1lIEdl

    def eval(self):
        self._training = False

class _cOl1Ed2(_c10lEcc):

    def __init__(self, _fl10Ecd: _c0OIEcB):
        super().__init__(_fl10Ecd)
        self.weight = Initializer.xavier((_fl10Ecd.input_dim, _fl10Ecd.output_dim))
        self.bias = Tensor.zeros(_fl10Ecd.output_dim) if _fl10Ecd.use_bias else None

    def _flOOEcE(self, _f1O1Ecf: Tensor) -> Tensor:
        output = _f1O1Ecf @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

class _cl0lEd3(_c10lEcc):

    def __init__(self, _fl10Ecd: _c0OIEcB, _f0IOEd4: float=1e-05):
        super().__init__(_fl10Ecd)
        self._f0IOEd4 = _f0IOEd4
        self.gamma = Tensor.ones(_fl10Ecd.input_dim)
        self.beta = Tensor.zeros(_fl10Ecd.input_dim)

    def _flOOEcE(self, _f1O1Ecf: Tensor) -> Tensor:
        mean = _f1O1Ecf.mean(dim=-1, keepdim=True)
        var = _f1O1Ecf.var(dim=-1, keepdim=True)
        normalized = (_f1O1Ecf - mean) / (var + self._f0IOEd4).sqrt()
        return normalized * self.gamma + self.beta

class _clllEd5(_c10lEcc):

    def __init__(self, _fl10Ecd: _c0OIEcB, _fOl0Ed6: float=0.1, _f0IOEd4: float=1e-05):
        super().__init__(_fl10Ecd)
        self._fOl0Ed6 = _fOl0Ed6
        self._f0IOEd4 = _f0IOEd4
        self.gamma = Tensor.ones(_fl10Ecd.input_dim)
        self.beta = Tensor.zeros(_fl10Ecd.input_dim)
        self.running_mean = Tensor.zeros(_fl10Ecd.input_dim)
        self.running_var = Tensor.ones(_fl10Ecd.input_dim)

    def _flOOEcE(self, _f1O1Ecf: Tensor) -> Tensor:
        if self._training:
            mean = _f1O1Ecf.mean(dim=0)
            var = _f1O1Ecf.var(dim=0)
            self.running_mean = (1 - self._fOl0Ed6) * self.running_mean + self._fOl0Ed6 * mean
            self.running_var = (1 - self._fOl0Ed6) * self.running_var + self._fOl0Ed6 * var
        else:
            mean = self.running_mean
            var = self.running_var
        normalized = (_f1O1Ecf - mean) / (var + self._f0IOEd4).sqrt()
        return normalized * self.gamma + self.beta

class _cOI1Ed7(_c10lEcc):

    def __init__(self, _fl10Ecd: _c0OIEcB):
        super().__init__(_fl10Ecd)
        self.p = _fl10Ecd.dropout

    def _flOOEcE(self, _f1O1Ecf: Tensor) -> Tensor:
        if not self._training or self.p == 0:
            return _f1O1Ecf
        mask = Tensor.rand(*_f1O1Ecf.shape.dims) > self.p
        return _f1O1Ecf * mask * (1.0 / (1.0 - self.p))

class _c0llEd8(_c10lEcc):

    def __init__(self, _fl10Ecd: _c0OIEcB, _fOI1Ed9: int):
        super().__init__(_fl10Ecd)
        self._fOI1Ed9 = _fOI1Ed9
        self.weight = Initializer.xavier((_fOI1Ed9, _fl10Ecd.output_dim))

    def _flOOEcE(self, _f1O1Ecf: Tensor) -> Tensor:
        indices = _f1O1Ecf.numpy().astype(int).flatten()
        embedded = np.zeros((len(indices), self._fl10Ecd.output_dim))
        for i, idx in enumerate(indices):
            embedded[i] = self.weight._data[idx]
        return Tensor(embedded.reshape(*_f1O1Ecf.shape.dims, self._fl10Ecd.output_dim))

class _c1I0EdA(_c10lEcc):

    def __init__(self, _fl10Ecd: _c0OIEcB, _f1llEdB: int, _f1l0Edc: int=1, _fI11Edd: int=0):
        super().__init__(_fl10Ecd)
        self._f1llEdB = _f1llEdB
        self._f1l0Edc = _f1l0Edc
        self._fI11Edd = _fI11Edd
        self.weight = Initializer.he((_fl10Ecd.output_dim, _fl10Ecd.input_dim, _f1llEdB))
        self.bias = Tensor.zeros(_fl10Ecd.output_dim) if _fl10Ecd.use_bias else None

    def _flOOEcE(self, _f1O1Ecf: Tensor) -> Tensor:
        batch_size = _f1O1Ecf.shape[0] if _f1O1Ecf.ndim == 3 else 1
        in_channels = _f1O1Ecf.shape[1] if _f1O1Ecf.ndim == 3 else _f1O1Ecf.shape[0]
        length = _f1O1Ecf.shape[-1]
        if self._fI11Edd > 0:
            padded = Tensor.zeros(batch_size, in_channels, length + 2 * self._fI11Edd)
            padded._data[:, :, self._fI11Edd:-self._fI11Edd] = _f1O1Ecf._data
            _f1O1Ecf = padded
            length = length + 2 * self._fI11Edd
        out_length = (length - self._f1llEdB) // self._f1l0Edc + 1
        output = Tensor.zeros(batch_size, self._fl10Ecd.output_dim, out_length)
        for i in range(out_length):
            start = i * self._f1l0Edc
            end = start + self._f1llEdB
            window = _f1O1Ecf[:, :, start:end] if _f1O1Ecf.ndim == 3 else _f1O1Ecf[:, start:end]
            for j in range(self._fl10Ecd.output_dim):
                output._data[:, j, i] = (window._data * self.weight._data[j]).sum(axis=(1, 2))
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).unsqueeze(-1)
        return output