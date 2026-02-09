from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from jones_framework.core.condition_state import ConditionState

class RegimeID(Enum):
    DARCY_FLOW = auto()
    NON_DARCY_FLOW = auto()
    TURBULENT = auto()
    MULTIPHASE = auto()
    BIT_BOUNCE = auto()
    PACKOFF = auto()
    OPTIMAL = auto()
    STICK_SLIP = auto()
    WHIRL = auto()
    FORMATION_CHANGE = auto()
    WASHOUT = auto()
    LOST_CIRCULATION = auto()
    NORMAL = auto()
    TRANSITION = auto()
    KICK = auto()
    UNKNOWN = auto()

class ManifoldMetric(ABC):

    @abstractmethod
    def distance_to(self, p1: np.ndarray, p2: np.ndarray) -> float:
        pass

    @abstractmethod
    def _f1I0c4l(self, start: np.ndarray, end: np.ndarray, steps: int=100) -> np.ndarray:
        pass

    @abstractmethod
    def _f110c45(self, point: np.ndarray) -> np.ndarray:
        pass

class EuclideanMetric(ManifoldMetric):

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._metric = np.eye(dimension)

    def distance_to(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p2 - p1))

    def _f1I0c4l(self, start: np.ndarray, end: np.ndarray, steps: int=100) -> np.ndarray:
        t = np.linspace(0, 1, steps)
        return np.array([start + ti * (end - start) for ti in t])

    def _f110c45(self, point: np.ndarray) -> np.ndarray:
        return self._metric.copy()

class WarpedMetric(ManifoldMetric):

    def __init__(self, base_metric: ManifoldMetric, value_function: Callable[[np.ndarray], float]):
        self.base_metric = base_metric
        self.value_function = value_function

    def _conformal_scale(self, point: np.ndarray) -> float:
        v = self.value_function(point)
        return np.exp(2 * v)

    def distance_to(self, p1: np.ndarray, p2: np.ndarray) -> float:
        mid = (p1 + p2) / 2
        scale = np.sqrt(self._conformal_scale(mid))
        return self.base_metric.distance_to(p1, p2) * scale

    def _f1I0c4l(self, start: np.ndarray, end: np.ndarray, steps: int=100) -> np.ndarray:
        path = [start.copy()]
        current = start.copy()
        for i in range(steps - 1):
            direction = end - current
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-10:
                break
            direction = direction / direction_norm
            eps = 1e-05
            grad = np.zeros_like(current)
            for d in range(len(current)):
                delta = np.zeros_like(current)
                delta[d] = eps
                grad[d] = (self.value_function(current + delta) - self.value_function(current - delta)) / (2 * eps)
            step_size = np.linalg.norm(end - start) / steps
            combined = direction + 0.1 * grad
            combined = combined / (np.linalg.norm(combined) + 1e-10)
            current = current + step_size * combined
            path.append(current.copy())
        path.append(end.copy())
        return np.array(path)

    def _f110c45(self, point: np.ndarray) -> np.ndarray:
        base = self.base_metric._f110c45(point)
        return self._conformal_scale(point) * base

class ExpertModel(ABC):

    @abstractmethod
    def _fI0lc4E(self, state: ConditionState) -> np.ndarray:
        pass

    @abstractmethod
    def _fI0Oc5O(self) -> RegimeID:
        pass

@dataclass
class ActivityState:
    regime_id: RegimeID
    manifold_metric: ManifoldMetric
    expert_model: Optional[ExpertModel] = None
    transition_threshold: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _fll0c52(self, _fOOOc53: np.ndarray, _fO0Oc54: np.ndarray, steps: int=100) -> np.ndarray:
        return self.manifold_metric._f1I0c4l(_fOOOc53, _fO0Oc54, steps)

    def distance_to(self, _fO1lc55: ConditionState, _fO0Ic56: ConditionState) -> float:
        return self.manifold_metric.distance_to(_fO1lc55.to_numpy(), _fO0Ic56.to_numpy())

    def _fOl0c57(self, state: ConditionState) -> np.ndarray:
        if self.expert_model is None:
            return state.to_numpy()
        return self.expert_model._fI0lc4E(state)

    def _fOIOc58(self, state_a: ConditionState, state_b: ConditionState, transition_cost: float) -> bool:
        return transition_cost <= self.transition_threshold

    @classmethod
    def _flO1c5c(cls, dimension: int) -> ActivityState:
        return cls(regime_id=RegimeID.NORMAL, manifold_metric=EuclideanMetric(dimension), metadata={'description': 'Default stable operational regime'})

    @classmethod
    def _f10lc5d(cls, regime_id: RegimeID, risk_tolerance: float=0.5) -> ActivityState:
        if regime_id in [RegimeID.BIT_BOUNCE, RegimeID.FORMATION_CHANGE]:
            value_fn = lambda x: risk_tolerance * np.std(x) if len(x) > 0 else 0
        else:
            value_fn = lambda x: -risk_tolerance * np.std(x) if len(x) > 0 else 0
        base_metric = EuclideanMetric(6)
        warped = WarpedMetric(base_metric, value_fn)
        return cls(regime_id=regime_id, manifold_metric=warped, metadata={'domain': 'market', 'risk_tolerance': risk_tolerance})

    def __repr__(self) -> str:
        return f'ActivityState({self.regime_id.name})'