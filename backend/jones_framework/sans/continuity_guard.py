from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from jones_framework.core.condition_state import ConditionState

class _cOlOlc8(Enum):
    SAFE = auto()
    CAUTION = auto()
    DANGEROUS = auto()
    BLOCKED = auto()

@dataclass
class _cl0Ilc9:
    safety_level: _cOlOlc8
    kl_divergence: float
    message: str
    suggested_action: Optional[str] = None
    intermediate_states: Optional[List[np.ndarray]] = None

class _cIIOlcA:

    def __init__(self, _fO0OlcB: float=0.5, _f0I0lcc: float=1.0, _flI0lcd: float=2.0, _fIIIlcE: bool=True):
        self.kl_caution = _fO0OlcB
        self.kl_dangerous = _f0I0lcc
        self.kl_blocked = _flI0lcd
        self._fIIIlcE = _fIIIlcE
        self._safety_rules: List[Callable[[np.ndarray, np.ndarray], Optional[str]]] = []
        self._transition_log: List[_cl0Ilc9] = []

    def distance_to(self, _fOl1ldO: np.ndarray, _fIl1ldl: np.ndarray, _f0IOld2: float=1e-10) -> float:
        _fOl1ldO = _fOl1ldO + _f0IOld2
        _fIl1ldl = _fIl1ldl + _f0IOld2
        p = np.abs(_fOl1ldO) / np.sum(np.abs(_fOl1ldO))
        q = np.abs(_fIl1ldl) / np.sum(np.abs(_fIl1ldl))
        kl_pq = np.sum(p * np.log(p / (q + _f0IOld2) + _f0IOld2))
        kl_qp = np.sum(q * np.log(q / (p + _f0IOld2) + _f0IOld2))
        return (kl_pq + kl_qp) / 2

    def _fIOOld3(self, _fO1Old4: ConditionState, _f0l0ld5: np.ndarray) -> _cl0Ilc9:
        _fOl1ldO = _fO1Old4.to_numpy()
        kl_div = self.distance_to(_fOl1ldO, _f0l0ld5)
        rule_messages = []
        for rule in self._safety_rules:
            msg = rule(_fOl1ldO, _f0l0ld5)
            if msg:
                rule_messages.append(msg)
        if kl_div >= self.kl_blocked or rule_messages:
            safety = _cOlOlc8.BLOCKED if self._fIIIlcE else _cOlOlc8.DANGEROUS
            message = f'BLOCKED: KL={kl_div:.3f} exceeds threshold'
            if rule_messages:
                message += f" | Rules: {'; '.join(rule_messages)}"
            suggested = 'Reduce step size or use intermediate transitions'
            intermediate = self._compute_intermediate_states(_fOl1ldO, _f0l0ld5)
        elif kl_div >= self.kl_dangerous:
            safety = _cOlOlc8.DANGEROUS
            message = f'DANGEROUS: KL={kl_div:.3f} indicates discontinuity'
            suggested = 'Consider gradual transition'
            intermediate = self._compute_intermediate_states(_fOl1ldO, _f0l0ld5)
        elif kl_div >= self.kl_caution:
            safety = _cOlOlc8.CAUTION
            message = f'CAUTION: KL={kl_div:.3f} approaching threshold'
            suggested = None
            intermediate = None
        else:
            safety = _cOlOlc8.SAFE
            message = f'SAFE: KL={kl_div:.3f} within bounds'
            suggested = None
            intermediate = None
        result = _cl0Ilc9(safety_level=safety, kl_divergence=kl_div, message=message, suggested_action=suggested, intermediate_states=intermediate)
        self._transition_log.append(result)
        return result

    def _fl0Old6(self, _fOl1ldO: np.ndarray, _fIl1ldl: np.ndarray, _f10Old7: int=5) -> List[np.ndarray]:
        t_values = np.linspace(0, 1, _f10Old7 + 1)[1:-1]
        return [_fOl1ldO + t * (_fIl1ldl - _fOl1ldO) for t in t_values]

    def _flIOld8(self, _fI00ld9: Callable[[np.ndarray, np.ndarray], Optional[str]]):
        self._safety_rules.append(_fI00ld9)

    def _fOllldA(self, _fO1Old4: ConditionState, _f11IldB: np.ndarray, _f110ldc: bool=False) -> Tuple[np.ndarray, _cl0Ilc9]:
        validation = self._fIOOld3(_fO1Old4, _f11IldB)
        if validation.safety_level == _cOlOlc8.BLOCKED and (not _f110ldc):
            if validation.intermediate_states:
                return (validation.intermediate_states[0], validation)
            else:
                return (_fO1Old4.to_numpy(), validation)
        return (_f11IldB, validation)

    def _f1lOldd(self) -> Dict[str, Any]:
        if not self._transition_log:
            return {'total_validations': 0}
        counts = {}
        for val in self._transition_log:
            name = val.safety_level.name
            counts[name] = counts.get(name, 0) + 1
        kl_values = [v.kl_divergence for v in self._transition_log]
        return {'total_validations': len(self._transition_log), 'safety_counts': counts, 'avg_kl_divergence': np.mean(kl_values), 'max_kl_divergence': np.max(kl_values), 'blocked_count': counts.get('BLOCKED', 0), 'dangerous_count': counts.get('DANGEROUS', 0)}

    def _fllIldE(self):
        self._transition_log = []

    @staticmethod
    def _fOOOldf(_f00IlEO: float, _f1O1lEl: Optional[int]=None) -> Callable[[np.ndarray, np.ndarray], Optional[str]]:

        def _fI00ld9(_fOl1ldO: np.ndarray, _fIl1ldl: np.ndarray) -> Optional[str]:
            if _f1O1lEl is not None:
                change = abs(_fIl1ldl[_f1O1lEl] - _fOl1ldO[_f1O1lEl])
                if change > _f00IlEO:
                    return f'Dim {_f1O1lEl} change {change:.3f} > {_f00IlEO}'
            else:
                changes = np.abs(_fIl1ldl - _fOl1ldO)
                max_idx = np.argmax(changes)
                if changes[max_idx] > _f00IlEO:
                    return f'Max change {changes[max_idx]:.3f} > {_f00IlEO} (dim {max_idx})'
            return None
        return _fI00ld9

    @staticmethod
    def _f101lE2(_f1IOlE3: np.ndarray, _f01IlE4: np.ndarray) -> Callable[[np.ndarray, np.ndarray], Optional[str]]:

        def _fI00ld9(_fOl1ldO: np.ndarray, _fIl1ldl: np.ndarray) -> Optional[str]:
            violations = []
            for i in range(min(len(_fIl1ldl), len(_f1IOlE3))):
                if _fIl1ldl[i] < _f1IOlE3[i]:
                    violations.append(f'dim{i}<{_f1IOlE3[i]}')
                if i < len(_f01IlE4) and _fIl1ldl[i] > _f01IlE4[i]:
                    violations.append(f'dim{i}>{_f01IlE4[i]}')
            if violations:
                return f"Bounds violated: {', '.join(violations)}"
            return None
        return _fI00ld9

    @staticmethod
    def _fIO1lE5(_fl0OlE6: float, _fOOOlE7: float=1.0) -> Callable[[np.ndarray, np.ndarray], Optional[str]]:

        def _fI00ld9(_fOl1ldO: np.ndarray, _fIl1ldl: np.ndarray) -> Optional[str]:
            change = np.linalg.norm(_fIl1ldl - _fOl1ldO)
            rate = change / _fOOOlE7
            if rate > _fl0OlE6:
                return f'Rate {rate:.3f} exceeds max {_fl0OlE6}'
            return None
        return _fI00ld9

# Public API aliases for obfuscated classes
SafetyLevel = _cOlOlc8
ValidationResult = _cl0Ilc9
ContinuityGuard = _cIIOlcA
