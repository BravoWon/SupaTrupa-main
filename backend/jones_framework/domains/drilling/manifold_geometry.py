from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import numpy as np

@dataclass
class _cl0l8dc:

    def __init__(self, _fIOI8dd: Callable[[float, float], float], _fO1O8dE: float=1.0, _flOl8df: float=1.0):
        self._fIOI8dd = _fIOI8dd
        self._fO1O8dE = _fO1O8dE
        self._flOl8df = _flOl8df

    def _fIII8EO(self, _fllI8El: float, _fIII8E2: float) -> float:
        rop = self._fIOI8dd(_fllI8El, _fIII8E2) * self._fO1O8dE
        return max(rop, 1e-06)

    def _fOO18E3(self, _fllI8El: float, _fIII8E2: float) -> np.ndarray:
        rop = self._fIII8EO(_fllI8El, _fIII8E2)
        g_tt = self._flOl8df ** 2
        g_dd = (self._flOl8df / rop) ** 2
        return np.array([[g_tt, 0.0], [0.0, g_dd]])

    def _fl118E4(self, _fllI8El: float, _fIII8E2: float) -> float:
        rop = self._fIII8EO(_fllI8El, _fIII8E2)
        return self._flOl8df ** 4 / rop ** 2

    def _fOl18E5(self, _fllI8El: float, _fIII8E2: float) -> np.ndarray:
        rop = self._fIII8EO(_fllI8El, _fIII8E2)
        g_inv_tt = 1.0 / self._flOl8df ** 2
        g_inv_dd = (rop / self._flOl8df) ** 2
        return np.array([[g_inv_tt, 0.0], [0.0, g_inv_dd]])

    def _f10I8E6(self, _fl1I8E7: float, _f1008E8: float, _fI018E9: float, _f1OI8EA: float) -> float:
        dt = _fI018E9 - _fl1I8E7
        dd = _f1OI8EA - _f1008E8
        avg_rop = (self._fIII8EO(_fl1I8E7, _f1008E8) + self._fIII8EO(_fI018E9, _f1OI8EA)) / 2
        return np.sqrt(dt ** 2 + (dd / avg_rop) ** 2) * self._flOl8df

    def _fOll8EB(self, _fllI8El: float, _fIII8E2: float, _f0lO8Ec: float=0.01) -> np.ndarray:
        g = self._fOO18E3(_fllI8El, _fIII8E2)
        g_inv = self._fOl18E5(_fllI8El, _fIII8E2)
        dg_dt = (self._fOO18E3(_fllI8El + _f0lO8Ec, _fIII8E2) - self._fOO18E3(_fllI8El - _f0lO8Ec, _fIII8E2)) / (2 * _f0lO8Ec)
        dg_dd = (self._fOO18E3(_fllI8El, _fIII8E2 + _f0lO8Ec) - self._fOO18E3(_fllI8El, _fIII8E2 - _f0lO8Ec)) / (2 * _f0lO8Ec)
        dg = np.array([dg_dt, dg_dd])
        gamma = np.zeros((2, 2, 2))
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        gamma[k, i, j] += 0.5 * g_inv[k, l] * (dg[i, j, l] + dg[j, i, l] - dg[l, i, j])
        return gamma

    def _fll18Ed(self, _fllI8El: float, _fIII8E2: float, _f0lO8Ec: float=0.01) -> float:
        det = self._fl118E4(_fllI8El, _fIII8E2)
        if det <= 0:
            return 0.0
        sqrt_g = np.sqrt(det)
        log_sqrt_g = np.log(sqrt_g)

        def _fIOl8EE(_f0OO8Ef, _flOl8fO):
            det_ = self._fl118E4(_f0OO8Ef, _flOl8fO)
            if det_ <= 0:
                return 0.0
            return np.log(np.sqrt(det_))
        d2_dt2 = (_fIOl8EE(_fllI8El + _f0lO8Ec, _fIII8E2) - 2 * _fIOl8EE(_fllI8El, _fIII8E2) + _fIOl8EE(_fllI8El - _f0lO8Ec, _fIII8E2)) / (_f0lO8Ec * _f0lO8Ec)
        d2_dd2 = (_fIOl8EE(_fllI8El, _fIII8E2 + _f0lO8Ec) - 2 * _fIOl8EE(_fllI8El, _fIII8E2) + _fIOl8EE(_fllI8El, _fIII8E2 - _f0lO8Ec)) / (_f0lO8Ec * _f0lO8Ec)
        return -2 * (d2_dt2 + d2_dd2) / sqrt_g

class _cl0l8fl:

    def __init__(self, _fOll8f2: _cl0l8dc):
        self._fOll8f2 = _fOll8f2

    def _f10l8f3(self, _f0118f4: Tuple[float, float], _f10l8f5: Tuple[float, float], _f1II8f6: int=100, _fOO18f7: float=0.1) -> List[Tuple[float, float]]:
        path = [_f0118f4]
        _fllI8El, _fIII8E2 = _f0118f4
        vt, vd = _f10l8f5
        for _ in range(_f1II8f6):
            gamma = self._fOll8f2._fOll8EB(_fllI8El, _fIII8E2)
            at = -gamma[0, 0, 0] * vt * vt - 2 * gamma[0, 0, 1] * vt * vd - gamma[0, 1, 1] * vd * vd
            ad = -gamma[1, 0, 0] * vt * vt - 2 * gamma[1, 0, 1] * vt * vd - gamma[1, 1, 1] * vd * vd
            vt += at * _fOO18f7
            vd += ad * _fOO18f7
            _fllI8El += vt * _fOO18f7
            _fIII8E2 += vd * _fOO18f7
            path.append((_fllI8El, _fIII8E2))
            if _fIII8E2 < 0 or _fIII8E2 > 50000 or abs(_fllI8El) > 10000000.0:
                break
        return path

    def _fl008f8(self, _f0118f4: Tuple[float, float], _fllI8f9: Tuple[float, float], _fOOl8fA: int=50) -> List[Tuple[float, float]]:
        direction = (_fllI8f9[0] - _f0118f4[0], _fllI8f9[1] - _f0118f4[1])
        dist = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        if dist < 1e-10:
            return [_f0118f4, _fllI8f9]
        v0 = (direction[0] / dist, direction[1] / dist)
        v_scale_lo, v_scale_hi = (0.1, 10.0)
        for _ in range(_fOOl8fA):
            v_scale = (v_scale_lo + v_scale_hi) / 2
            path = self._f10l8f3(_f0118f4, (v0[0] * v_scale, v0[1] * v_scale))
            if not path:
                break
            final = path[-1]
            final_dist = np.sqrt((final[0] - _fllI8f9[0]) ** 2 + (final[1] - _fllI8f9[1]) ** 2)
            if final_dist < dist * 0.01:
                return path
            if final[1] > _fllI8f9[1]:
                v_scale_hi = v_scale
            else:
                v_scale_lo = v_scale
        return self._f10l8f3(_f0118f4, (v0[0] * v_scale, v0[1] * v_scale))

def _fIOO8fB(_fIOI8dd: Callable[[float, float], float], _f1118fc: Tuple[float, float], _f0IO8fd: Tuple[float, float], _fIOI8fE: int=20) -> str:
    _fOll8f2 = _cl0l8dc(_fIOI8dd)
    t_vals = np.linspace(_f1118fc[0], _f1118fc[1], _fIOI8fE)
    d_vals = np.linspace(_f0IO8fd[0], _f0IO8fd[1], _fIOI8fE)
    lines = []
    lines.append(f'Metric Field Visualization (t: {_f1118fc}, d: {_f0IO8fd})')
    lines.append('=' * (_fIOI8fE + 4))
    for _fIII8E2 in reversed(d_vals):
        row = '| '
        for _fllI8El in t_vals:
            R = _fOll8f2._fll18Ed(_fllI8El, _fIII8E2)
            det = _fOll8f2._fl118E4(_fllI8El, _fIII8E2)
            if det <= 0:
                char = '#'
            elif R > 0.1:
                char = '+'
            elif R < -0.1:
                char = '-'
            else:
                char = '.'
            row += char
        row += ' |'
        lines.append(row)
    lines.append('=' * (_fIOI8fE + 4))
    lines.append(f"Legend: '+' converging, '-' diverging, '.' flat, '#' singular")
    return '\n'.join(lines)

def _fI108ff(_fllI8El: float, _fIII8E2: float) -> float:
    baseline = 50.0
    depth_decay = np.exp(-_fIII8E2 / 10000)
    formation_variation = 10 * np.sin(_fIII8E2 / 500)
    return max(1.0, baseline * depth_decay + formation_variation)
# =============================================================================
# Public API Aliases
# =============================================================================

DrillingMetricField = _cl0l8dc
DrillingMetricField.get_rop = _cl0l8dc._fIII8EO
DrillingMetricField.metric_tensor = _cl0l8dc._fOO18E3
DrillingMetricField.metric_determinant = _cl0l8dc._fl118E4
DrillingMetricField.inverse_metric = _cl0l8dc._fOl18E5
DrillingMetricField.metric_distance = _cl0l8dc._f10I8E6
DrillingMetricField.christoffel_symbols = _cl0l8dc._fOll8EB
DrillingMetricField.ricci_scalar = _cl0l8dc._fll18Ed

GeodesicSolver = _cl0l8fl
GeodesicSolver.solve_ivp = _cl0l8fl._f10l8f3
GeodesicSolver.solve_bvp = _cl0l8fl._fl008f8

visualize_metric_field = _fIOO8fB
default_rop_function = _fI108ff

__all__ = [
    "DrillingMetricField",
    "GeodesicSolver",
    "visualize_metric_field",
    "default_rop_function",
]

if __name__ == '__main__':
    print(_fIOO8fB(_fI108ff, t_range=(0, 100000), d_range=(0, 10000), resolution=40))
    print('\n' + '=' * 50)
    print('Computing geodesic from surface to 5000 ft...')
    _fOll8f2 = _cl0l8dc(_fI108ff)
    solver = _cl0l8fl(_fOll8f2)
    path = solver._f10l8f3(start=(0, 0), initial_velocity=(1.0, 0.5), n_steps=100)
    print(f'Path length: {len(path)} points')
    print(f'Final point: t={path[-1][0]:.1f}s, d={path[-1][1]:.1f}ft')