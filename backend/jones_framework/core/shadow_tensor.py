from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from jones_framework.core.condition_state import ConditionState

@dataclass
class _cl0Ic25:
    metric_proxy: np.ndarray
    tangent_proxy: np.ndarray
    fractal_proxy: np.ndarray
    point_cloud: np.ndarray
    timestamp_range: Tuple[int, int] = (0, 0)

    @property
    def dimension(self) -> int:
        return len(self.metric_proxy) + len(self.tangent_proxy) + len(self.fractal_proxy)

    def _fIl1c27(self) -> np.ndarray:
        return np.concatenate([self.metric_proxy, self.tangent_proxy, self.fractal_proxy])

class _cOI1c28:

    def __init__(self, _fll1c29: int=3, _fI1Oc2A: int=1, _fOI1c2B: Optional[List[int]]=None):
        self._fll1c29 = _fll1c29
        self._fI1Oc2A = _fI1Oc2A
        self._fOI1c2B = _fOI1c2B or [5, 20, 50]

    def get_metric_at(self, _fllIc2d: np.ndarray) -> np.ndarray:
        n = len(_fllIc2d)
        required_length = (self._fll1c29 - 1) * self._fI1Oc2A + 1
        if n < required_length:
            raise ValueError(f'Time series too short. Need at least {required_length} points for embedding_dim={self._fll1c29} and delay={self._fI1Oc2A}')
        n_points = n - (self._fll1c29 - 1) * self._fI1Oc2A
        embedded = np.zeros((n_points, self._fll1c29))
        for i in range(self._fll1c29):
            start = i * self._fI1Oc2A
            end = start + n_points
            embedded[:, i] = _fllIc2d[start:end]
        return embedded

    def _f010c2E(self, _fIl0c2f: np.ndarray, _f101c3O: int, _fI1Oc3l: float=2.0) -> np.ndarray:
        if len(_fIl0c2f) < _f101c3O:
            return np.zeros(len(_fIl0c2f))
        bbw = np.zeros(len(_fIl0c2f))
        for i in range(_f101c3O - 1, len(_fIl0c2f)):
            window_data = _fIl0c2f[i - _f101c3O + 1:i + 1]
            sma = np.mean(window_data)
            std = np.std(window_data)
            upper = sma + _fI1Oc3l * std
            lower = sma - _fI1Oc3l * std
            if sma > 0:
                bbw[i] = (upper - lower) / sma
            else:
                bbw[i] = 0
        return bbw

    def _fl0Oc32(self, _fIl0c2f: np.ndarray, _f101c3O: int=14) -> np.ndarray:
        if len(_fIl0c2f) < _f101c3O + 1:
            return np.full(len(_fIl0c2f), 50.0)
        deltas = np.diff(_fIl0c2f)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = np.full(len(_fIl0c2f), 50.0)
        avg_gain = np.mean(gains[:_f101c3O])
        avg_loss = np.mean(losses[:_f101c3O])
        for i in range(_f101c3O, len(_fIl0c2f) - 1):
            avg_gain = (avg_gain * (_f101c3O - 1) + gains[i]) / _f101c3O
            avg_loss = (avg_loss * (_f101c3O - 1) + losses[i]) / _f101c3O
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    def _f1IIc33(self, _fIl0c2f: np.ndarray, _flOIc34: Optional[List[int]]=None) -> np.ndarray:
        _flOIc34 = _flOIc34 or [5, 10, 20, 50]
        if len(_fIl0c2f) < max(_flOIc34):
            return np.zeros(len(_flOIc34) + 2)
        mas = []
        for w in _flOIc34:
            ma = np.convolve(_fIl0c2f, np.ones(w) / w, mode='valid')
            pad_left = (len(_fIl0c2f) - len(ma)) // 2
            pad_right = len(_fIl0c2f) - len(ma) - pad_left
            ma_padded = np.pad(ma, (pad_left, pad_right), mode='edge')
            mas.append(ma_padded)
        mas = np.array(mas)
        features = []
        spreads = np.diff(mas, axis=0)
        features.append(np.mean(np.abs(spreads)))
        ordered = np.all(np.diff(mas, axis=0) <= 0, axis=0)
        features.append(np.mean(ordered))
        for ma in mas:
            slope = (ma[-1] - ma[0]) / (len(ma) * ma[0]) if ma[0] != 0 else 0
            features.append(slope)
        return np.array(features)

    def _f1I0c35(self, _fIl0c2f: np.ndarray, _fIOlc36: np.ndarray, _fOI0c37: int=20) -> float:
        if len(_fIl0c2f) < _fOI0c37 or len(_fIOlc36) < _fOI0c37:
            return 0.0
        recent_prices = _fIl0c2f[-_fOI0c37:]
        recent_rsi = _fIOlc36[-_fOI0c37:]
        price_min_idx = np.argmin(recent_prices)
        price_max_idx = np.argmax(recent_prices)
        if price_min_idx > _fOI0c37 // 2:
            first_half_min = np.min(recent_prices[:_fOI0c37 // 2])
            second_half_min = np.min(recent_prices[_fOI0c37 // 2:])
            first_half_rsi_min = np.min(recent_rsi[:_fOI0c37 // 2])
            second_half_rsi_min = np.min(recent_rsi[_fOI0c37 // 2:])
            if second_half_min < first_half_min and second_half_rsi_min > first_half_rsi_min:
                return (second_half_rsi_min - first_half_rsi_min) / 100
        if price_max_idx > _fOI0c37 // 2:
            first_half_max = np.max(recent_prices[:_fOI0c37 // 2])
            second_half_max = np.max(recent_prices[_fOI0c37 // 2:])
            first_half_rsi_max = np.max(recent_rsi[:_fOI0c37 // 2])
            second_half_rsi_max = np.max(recent_rsi[_fOI0c37 // 2:])
            if second_half_max > first_half_max and second_half_rsi_max < first_half_rsi_max:
                return -(first_half_rsi_max - second_half_rsi_max) / 100
        return 0.0

    def _fIO0c38(self, _f11Oc39: List[ConditionState]) -> _cl0Ic25:
        if not _f11Oc39:
            raise ValueError('Cannot build Shadow Tensor from empty state list')
        _fIl0c2f = np.array([s.vector[0] for s in _f11Oc39])
        metric_proxies = []
        tangent_proxies = []
        for scale in self._fOI1c2B:
            bbw = self._f010c2E(_fIl0c2f, scale)
            _fIOlc36 = self._fl0Oc32(_fIl0c2f, min(scale, 14))
            metric_proxies.append(bbw[-1] if len(bbw) > 0 else 0)
            tangent_proxies.append(_fIOlc36[-1] if len(_fIOlc36) > 0 else 50)
            divergence = self._f1I0c35(_fIl0c2f, _fIOlc36, min(scale, len(_fIl0c2f)))
            tangent_proxies.append(divergence)
        fractal_proxy = self._f1IIc33(_fIl0c2f)
        try:
            point_cloud = self.get_metric_at(_fIl0c2f)
        except ValueError:
            point_cloud = _fIl0c2f.reshape(-1, 1)
        timestamps = [s.timestamp for s in _f11Oc39]
        return _cl0Ic25(metric_proxy=np.array(metric_proxies), tangent_proxy=np.array(tangent_proxies), fractal_proxy=fractal_proxy, point_cloud=point_cloud, timestamp_range=(min(timestamps), max(timestamps)))

    def _fO1Oc3A(self, _fIl0c2f: np.ndarray, _f1I0c3B: Optional[np.ndarray]=None) -> _cl0Ic25:
        _f11Oc39 = []
        for i, price in enumerate(_fIl0c2f):
            ts = int(_f1I0c3B[i]) if _f1I0c3B is not None else i * 1000
            state = ConditionState(timestamp=ts, vector=(float(price), 0.0, 0.0, 0.0, 0.0, 0.0), metadata={'synthetic': True})
            _f11Oc39.append(state)
        return self._fIO0c38(_f11Oc39)


# Public API aliases for obfuscated classes
ShadowTensor = _cl0Ic25
ShadowTensor.concatenate = _cl0Ic25._fIl1c27
ShadowTensorBuilder = _cOI1c28
ShadowTensorBuilder.build_from_numpy = _cOI1c28._fO1Oc3A
ShadowTensorBuilder.build_from_states = _cOI1c28._fIO0c38