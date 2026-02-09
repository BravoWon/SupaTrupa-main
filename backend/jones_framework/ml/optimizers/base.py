from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from jones_framework.core.tensor_ops import Tensor
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.ml.networks.base import Parameter

@dataclass
class _cl11Elc:
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-08

@bridge(connects_to=['Tensor', 'Parameter'], connection_types={'Tensor': ConnectionType.USES})
class _clI0Eld(ABC):

    def __init__(self, _f0IOElE: List[Parameter], _f01lElf: _cl11Elc):
        self._f0IOElE = _f0IOElE
        self._f01lElf = _f01lElf
        self.state: Dict[str, Any] = {}
        self._step_count = 0

    @abstractmethod
    def _f11IE2O(self, _fOOOE2l: Dict[str, Tensor]):
        pass

    def _fOI1E22(self):
        for param in self._f0IOElE:
            param.tensor.grad = None

class _cI11E23(_clI0Eld):

    def __init__(self, _f0IOElE: List[Parameter], _f01lElf: _cl11Elc):
        super().__init__(_f0IOElE, _f01lElf)
        self.velocities: Dict[str, Tensor] = {}

    def _f11IE2O(self, _fOOOE2l: Dict[str, Tensor]):
        self._step_count += 1
        for param in self._f0IOElE:
            if param.name not in _fOOOE2l:
                continue
            grad = _fOOOE2l[param.name]
            if self._f01lElf.weight_decay > 0:
                grad = grad + self._f01lElf.weight_decay * param.tensor
            if self._f01lElf.momentum > 0:
                if param.name not in self.velocities:
                    self.velocities[param.name] = Tensor.zeros(*grad.shape.dims)
                v = self.velocities[param.name]
                v = self._f01lElf.momentum * v + grad
                self.velocities[param.name] = v
                update = v
            else:
                update = grad
            param.tensor = param.tensor - self._f01lElf.learning_rate * update

class _clI1E24(_clI0Eld):

    def __init__(self, _f0IOElE: List[Parameter], _f01lElf: _cl11Elc):
        super().__init__(_f0IOElE, _f01lElf)
        self.m: Dict[str, Tensor] = {}
        self.v: Dict[str, Tensor] = {}

    def _f11IE2O(self, _fOOOE2l: Dict[str, Tensor]):
        self._step_count += 1
        beta1, beta2 = self._f01lElf.betas
        for param in self._f0IOElE:
            if param.name not in _fOOOE2l:
                continue
            grad = _fOOOE2l[param.name]
            if param.name not in self.m:
                self.m[param.name] = Tensor.zeros(*grad.shape.dims)
                self.v[param.name] = Tensor.zeros(*grad.shape.dims)
            self.m[param.name] = beta1 * self.m[param.name] + (1 - beta1) * grad
            self.v[param.name] = beta2 * self.v[param.name] + (1 - beta2) * (grad * grad)
            m_hat = self.m[param.name] * (1 / (1 - beta1 ** self._step_count))
            v_hat = self.v[param.name] * (1 / (1 - beta2 ** self._step_count))
            update = m_hat / (v_hat.sqrt() + self._f01lElf.eps)
            if self._f01lElf.weight_decay > 0:
                update = update + self._f01lElf.weight_decay * param.tensor
            param.tensor = param.tensor - self._f01lElf.learning_rate * update

class _cOO0E25(_clI1E24):

    def _f11IE2O(self, _fOOOE2l: Dict[str, Tensor]):
        self._step_count += 1
        beta1, beta2 = self._f01lElf.betas
        for param in self._f0IOElE:
            if param.name not in _fOOOE2l:
                continue
            grad = _fOOOE2l[param.name]
            if param.name not in self.m:
                self.m[param.name] = Tensor.zeros(*grad.shape.dims)
                self.v[param.name] = Tensor.zeros(*grad.shape.dims)
            self.m[param.name] = beta1 * self.m[param.name] + (1 - beta1) * grad
            self.v[param.name] = beta2 * self.v[param.name] + (1 - beta2) * (grad * grad)
            m_hat = self.m[param.name] * (1 / (1 - beta1 ** self._step_count))
            v_hat = self.v[param.name] * (1 / (1 - beta2 ** self._step_count))
            param.tensor = param.tensor * (1 - self._f01lElf.learning_rate * self._f01lElf.weight_decay)
            update = m_hat / (v_hat.sqrt() + self._f01lElf.eps)
            param.tensor = param.tensor - self._f01lElf.learning_rate * update

class _cllOE26(_clI0Eld):

    def __init__(self, _f0IOElE: List[Parameter], _f01lElf: _cl11Elc, _fO0lE27=None):
        super().__init__(_f0IOElE, _f01lElf)
        self._fO0lE27 = _fO0lE27 or (lambda x: Tensor.eye(len(x)))

    def _f11IE2O(self, _fOOOE2l: Dict[str, Tensor]):
        self._step_count += 1
        for param in self._f0IOElE:
            if param.name not in _fOOOE2l:
                continue
            grad = _fOOOE2l[param.name]
            point = param.tensor
            metric = self._fO0lE27(point)
            metric_inv = metric.inv()
            riemannian_grad = metric_inv @ grad
            param.tensor = param.tensor - self._f01lElf.learning_rate * riemannian_grad