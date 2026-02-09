from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto
import numpy as np
from jones_framework.domains.base import DomainAdapter, DomainConfig, DomainType, DomainRegimeMapper, StateFactory
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.domains.drilling.coordinate_bridge import CoordinateSystem, CoordinateAtlas, ROPJacobian, DrillingManifold, DrillingConditionState
from jones_framework.domains.drilling.stand_calibration import WellTrajectory, StandBoundary, StandSegment, NPTCategory, NPTEvent
from jones_framework.domains.drilling.pipe_tally import PipeTally, create_standard_tally
from jones_framework.domains.drilling.uncertainty import UncertainValue, UncertainDepth, UncertainTime, UncertainROP, UncertaintyBudget

class _c0ll84A(Enum):
    NORMAL_DRILLING = auto()
    FAST_DRILLING = auto()
    SLOW_DRILLING = auto()
    STICK_SLIP = auto()
    WHIRL = auto()
    BIT_BOUNCE = auto()
    WASHOUT = auto()
    PACK_OFF = auto()
    KICK = auto()
    LOST_CIRCULATION = auto()
    CONNECTION = auto()
    TRIPPING = auto()
    SURVEYING = auto()
    REAMING = auto()
DRILLING_REGIME_MAP = {_c0ll84A.NORMAL_DRILLING: RegimeID.NORMAL, _c0ll84A.FAST_DRILLING: RegimeID.FORMATION_CHANGE, _c0ll84A.SLOW_DRILLING: RegimeID.WHIRL, _c0ll84A.STICK_SLIP: RegimeID.STICK_SLIP, _c0ll84A.WHIRL: RegimeID.WHIRL, _c0ll84A.BIT_BOUNCE: RegimeID.BIT_BOUNCE, _c0ll84A.WASHOUT: RegimeID.WASHOUT, _c0ll84A.PACK_OFF: RegimeID.PACKOFF, _c0ll84A.KICK: RegimeID.KICK, _c0ll84A.LOST_CIRCULATION: RegimeID.LOST_CIRCULATION, _c0ll84A.CONNECTION: RegimeID.NORMAL, _c0ll84A.TRIPPING: RegimeID.NORMAL, _c0ll84A.SURVEYING: RegimeID.NORMAL, _c0ll84A.REAMING: RegimeID.WHIRL}

class _c00184B(DomainRegimeMapper):

    def __init__(self):
        self._forward_map = DRILLING_REGIME_MAP
        self._reverse_map = {}
        for drilling_regime, framework_regime in DRILLING_REGIME_MAP.items():
            if framework_regime not in self._reverse_map:
                self._reverse_map[framework_regime] = drilling_regime

    def _fl1O84c(self, _flIl84d: _c0ll84A) -> RegimeID:
        return self._forward_map.get(_flIl84d, RegimeID.NORMAL)

    def _fOOO84E(self, _f11l84f: RegimeID) -> _c0ll84A:
        return self._reverse_map.get(_f11l84f, _c0ll84A.NORMAL_DRILLING)

    def _f1lO85O(self) -> List[_c0ll84A]:
        return list(_c0ll84A)

@dataclass
class _cOOl85l:
    feature_names: List[str] = field(default_factory=lambda: ['wob', 'rpm', 'torque', 'spp', 'flow_rate', 'rop', 'mse', 'ecd', 'differential_pressure'])

    def _fOI0852(self, _flII853: Dict[str, Any], _f01O854: int) -> DrillingConditionState:
        time_val = _flII853.get('time', _flII853.get('timestamp', _f01O854 / 1000.0))
        depth_val = _flII853.get('depth', _flII853.get('measured_depth', 0.0))
        rop_val = _flII853.get('rop', _flII853.get('rate_of_penetration', 0.0))
        features = np.array([_flII853.get(name, 0.0) for name in self.feature_names], dtype=np.float32)
        return DrillingConditionState._fOI0852(features=features, time=time_val, depth=depth_val, rop=rop_val, coordinate_system=CoordinateSystem.DUAL, metadata={'timestamp_ms': _f01O854, 'raw_data': _flII853})

    def _fI00855(self, _fIIO856: List[Dict[str, Any]]) -> List[DrillingConditionState]:
        import time
        base_ts = int(time.time() * 1000)
        return [self._fOI0852(d, base_ts + i * 1000) for i, d in enumerate(_fIIO856)]

@dataclass
class _c10l857(DomainConfig):
    default_coordinate_system: CoordinateSystem = CoordinateSystem.DUAL
    rop_interpolation: str = 'linear'
    pipe_range: str = 'RANGE_2'
    joints_per_stand: int = 3
    depth_uncertainty_ft: float = 0.01
    time_uncertainty_sec: float = 1.0
    rop_uncertainty_pct: float = 5.0
    stick_slip_threshold: float = 30.0
    whirl_threshold: float = 0.5
    kick_threshold: float = 10.0
    max_stands: int = 1000
    calibration_interval: int = 50
    feature_names: List[str] = field(default_factory=lambda: ['wob', 'rpm', 'torque', 'spp', 'flow_rate', 'rop', 'mse', 'ecd', 'differential_pressure'])

    def __post_init__(self):
        self.domain_type = DomainType.ENERGY
        if not self.name:
            self.name = 'drilling'

@bridge(connects_to=['DrillingConditionState', 'WellTrajectory', 'PipeTally', 'DrillingManifold', 'UncertaintyBudget'], connection_types={'DrillingConditionState': ConnectionType.CREATES, 'WellTrajectory': ConnectionType.USES, 'PipeTally': ConnectionType.USES})
class _c001858(DomainAdapter[DrillingConditionState]):

    def __init__(self, _fO0O859: Optional[_c10l857]=None):
        if _fO0O859 is None:
            _fO0O859 = _c10l857(domain_type=DomainType.ENERGY, name='drilling')
        super().__init__(_fO0O859)
        self.drilling_config = _fO0O859
        self._trajectory = WellTrajectory(well_name=_fO0O859.name)
        self._pipe_tally = create_standard_tally(num_stands=100)
        self._manifold: Optional[DrillingManifold] = None
        self._uncertainty_budget = UncertaintyBudget()
        self._feature_names = _fO0O859.feature_names

    def _f0I185A(self) -> int:
        return 3 + len(self._feature_names)

    def _fl0085B(self) -> int:
        return 5

    def _f10185c(self) -> DomainRegimeMapper:
        return _c00184B()

    def _fOIl85d(self) -> _cOOl85l:
        return _cOOl85l(feature_names=self._feature_names)

    def _fl1I85E(self, _f0ll85f: Dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self._f0I185A(), dtype=np.float32)
        vector[0] = _f0ll85f.get('time', 0.0)
        vector[1] = _f0ll85f.get('depth', 0.0)
        vector[2] = _f0ll85f.get('rop', 0.0)
        for i, name in enumerate(self._feature_names):
            vector[3 + i] = _f0ll85f.get(name, 0.0)
        return vector

    def _f0O186O(self, _fll186l: np.ndarray) -> Dict[str, Any]:
        _flII853 = {'time': float(_fll186l[0]), 'depth': float(_fll186l[1]), 'rop': float(_fll186l[2])}
        for i, name in enumerate(self._feature_names):
            if 3 + i < len(_fll186l):
                _flII853[name] = float(_fll186l[3 + i])
        return _flII853

    def _f1II862(self, _f11l84f: RegimeID) -> ActivityState:
        drilling_regime = self._regime_mapper._fOOO84E(_f11l84f)
        current_rop = self._get_current_rop()
        metric_tensor = self._build_metric_tensor(current_rop)
        return ActivityState.from_condition_states(states=self._state_history[-self._fO0O859.regime_window:], regime_id=_f11l84f, metadata={'drilling_regime': drilling_regime.name, 'current_rop': current_rop, 'metric_tensor': metric_tensor, 'trajectory_summary': self._trajectory.summary()})

    def _fl1O863(self) -> float:
        if not self._state_history:
            return 0.0
        return self._state_history[-1].rop or 0.0

    def _f1Ol864(self, _f0l1865: float) -> np.ndarray:
        if _f0l1865 <= 0:
            return np.eye(2)
        return np.array([[1.0, 0.0], [0.0, 1.0 / _f0l1865 ** 2]])

    def _fOO0866(self, _flII853: Dict[str, Any], _f0l1867: Optional[float]=None, _fl01868: Optional[float]=None, _f0l1865: Optional[float]=None) -> DrillingConditionState:
        if _f0l1867 is not None:
            _flII853['time'] = _f0l1867
        if _fl01868 is not None:
            _flII853['depth'] = _fl01868
        if _f0l1865 is not None:
            _flII853['rop'] = _f0l1865
        return self.ingest(_flII853)

    def _fOI1869(self, _f0l1867: float, _fl01868: float, _f11186A: Optional[List[NPTEvent]]=None):
        self._trajectory.add_boundary(time=_f0l1867, depth=_fl01868, stand_number=len(self._trajectory.stands) + 1)
        if _f11186A:
            for event in _f11186A:
                self._trajectory.add_npt_event(event)

    def _f0I186B(self, _fII086c: NPTCategory, _fll186d: float, _fOOl86E: float, _fl01868: float, _fIII86f: Optional[str]=None):
        event = NPTEvent(category=_fII086c, start_time=_fll186d, duration=_fOOl86E, depth=_fl01868, description=_fIII86f)
        self._trajectory._f0I186B(event)

    def _fIII87O(self, _fl01868: float) -> Optional[float]:
        return self._trajectory.time_at_depth(_fl01868)

    def _fO0087l(self, _f0l1867: float) -> Optional[float]:
        return self._trajectory.depth_at_time(_f0l1867)

    def _flI1872(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._trajectory.rop_profile()

    def _fI1l873(self) -> float:
        summary = self._trajectory.summary()
        return summary.get('drilling_efficiency', 1.0)

    def _fOO1874(self) -> Dict[str, float]:
        npt_summary = self._trajectory.npt_summary()
        return {cat.name: _fOOl86E for cat, _fOOl86E in npt_summary.items()}

    def _f0lO875(self, _fO00876: np.ndarray, _fI0O877: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._trajectory._f0lO875(_fO00876, _fI0O877)

    def _flOO878(self, _f0I1879: str, _f1OI87A: float) -> UncertainValue:
        if _f0I1879 == 'depth':
            return UncertainValue(_f1OI87A, self.drilling_config.depth_uncertainty_ft)
        elif _f0I1879 == 'time':
            return UncertainValue(_f1OI87A, self.drilling_config.time_uncertainty_sec)
        elif _f0I1879 == 'rop':
            return UncertainValue(_f1OI87A, _f1OI87A * self.drilling_config.rop_uncertainty_pct / 100.0)
        else:
            return UncertainValue(_f1OI87A, abs(_f1OI87A) * 0.01)

    def _fIIO87B(self) -> Dict[str, Any]:
        return {'depth_contribution': self._uncertainty_budget.depth_contribution, 'time_contribution': self._uncertainty_budget.time_contribution, 'rop_model_contribution': self._uncertainty_budget.rop_model_contribution, 'integration_contribution': self._uncertainty_budget.integration_contribution, 'total': self._uncertainty_budget.total, 'dominant_source': self._uncertainty_budget.dominant_source}

    def _f1ll87c(self) -> Tuple[_c0ll84A, float]:
        classification = self.detect_regime()
        drilling_regime = self._regime_mapper._fOOO84E(classification._f11l84f)
        return (drilling_regime, classification.confidence)

    def _f01l87d(self) -> Dict[str, Any]:
        return self._trajectory.summary()

    def _fOIO87E(self) -> Dict[str, Any]:
        return {'total_joints': self._pipe_tally.total_joints, 'total_stands': self._pipe_tally.total_stands, 'total_length_ft': self._pipe_tally.total_length, 'current_depth': self._pipe_tally.current_depth}

    def _fO1087f(self) -> Dict[str, Any]:
        base = super()._fO1087f()
        base.update({'trajectory_summary': self._trajectory.summary(), 'pipe_tally_summary': self._fOIO87E(), 'uncertainty_budget': self._fIIO87B(), 'drilling_regime': self._f1ll87c()[0].name})
        return base

    def _fIIl88O(self) -> Dict[str, Any]:
        base = super()._fIIl88O()
        drilling_regime, confidence = self._f1ll87c()
        base.update({'drilling_regime': drilling_regime.name, 'regime_confidence': confidence, 'current_depth': self._trajectory.stands[-1].end_boundary._fl01868 if self._trajectory.stands else 0.0, 'current_rop': self._fl1O863(), 'drilling_efficiency': self._fI1l873(), 'stands_drilled': len(self._trajectory.stands)})
        return base

def _fOOO88l(_fIO1882: str='default_well', **config_kwargs) -> _c001858:
    _fO0O859 = _c10l857(domain_type=DomainType.ENERGY, name=_fIO1882, **config_kwargs)
    return _c001858(_fO0O859)
if __name__ == '__main__':
    print('=== Drilling Adapter Demo ===\n')
    adapter = _fOOO88l('Test Well #1')
    drilling_data = [{'time': 0, 'depth': 0, 'rop': 0, 'wob': 0, 'rpm': 0, 'torque': 0}, {'time': 60, 'depth': 10, 'rop': 10, 'wob': 15000, 'rpm': 120, 'torque': 5000}, {'time': 120, 'depth': 25, 'rop': 15, 'wob': 18000, 'rpm': 130, 'torque': 6000}, {'time': 180, 'depth': 45, 'rop': 20, 'wob': 20000, 'rpm': 140, 'torque': 7000}, {'time': 240, 'depth': 70, 'rop': 25, 'wob': 22000, 'rpm': 150, 'torque': 8000}]
    for _flII853 in drilling_data:
        adapter.ingest(_flII853)
    adapter._fOI1869(time=240, depth=70)
    print('Adapter Metrics:')
    for key, _f1OI87A in adapter._fIIl88O().items():
        print(f'  {key}: {_f1OI87A}')
    regime, confidence = adapter._f1ll87c()
    print(f'\nDetected Regime: {regime._fIO1882} (confidence: {confidence:.2f})')
    depth_unc = adapter._flOO878('depth', 70.0)
    print(f'\nDepth uncertainty: {depth_unc}')