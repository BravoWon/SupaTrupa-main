from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum, auto
import numpy as np
from datetime import datetime

class _cO009OO(Enum):
    RANGE_1 = auto()
    RANGE_2 = auto()
    RANGE_3 = auto()

@dataclass
class _cOI19Ol:
    serial_number: str
    length: float
    grade: str = 'S-135'
    weight: float = 19.5
    od: float = 5.0
    id: float = 4.276
    time_in_hole: Optional[float] = None
    time_out_hole: Optional[float] = None
    makeup_torque: Optional[float] = None

    @property
    def _fI1O9O2(self) -> bool:
        return self.time_in_hole is not None and self.time_out_hole is None

@dataclass
class _cIO09O3:
    joints: List[_cOI19Ol] = field(default_factory=list)
    stand_number: int = 0
    time_spud: Optional[float] = None
    time_connection: Optional[float] = None

    @property
    def _f1OI9O4(self) -> float:
        return sum((j.length for j in self.joints))

    @property
    def dimension(self) -> int:
        return len(self.joints)

    @property
    def _f11l9O6(self) -> Optional[float]:
        if self.time_spud is not None and self.time_connection is not None:
            return self.time_connection - self.time_spud
        return None

    @property
    def _f0I09O7(self) -> Optional[float]:
        dt = self._f11l9O6
        if dt is not None and dt > 0:
            return self._f1OI9O4 / (dt / 3600)
        return None

class _cI119O8:

    def __init__(self, _f1O19O9: str='', _f00O9OA: int=3, _f11I9OB: float=40.0, _fl1O9Oc: float=500.0):
        self._f1O19O9 = _f1O19O9
        self._f00O9OA = _f00O9OA
        self._f11I9OB = _f11I9OB
        self._fl1O9Oc = _fl1O9Oc
        self.joints_in_hole: List[_cOI19Ol] = []
        self.joints_on_rack: List[_cOI19Ol] = []
        self.stands: List[_cIO09O3] = []
        self._current_stand_joints: List[_cOI19Ol] = []
        self._drilling_start_time: Optional[float] = None

    @property
    def _f1lI9Od(self) -> float:
        return sum((j.length for j in self.joints_in_hole))

    @property
    def _fll09OE(self) -> float:
        return self._f1lI9Od + self._fl1O9Oc

    @property
    def _fl109Of(self) -> int:
        return len(self.joints_in_hole)

    @property
    def _fIl09lO(self) -> int:
        return len(self.stands)

    def _fI1I9ll(self, _fllO9l2: _cOI19Ol):
        self.joints_on_rack.append(_fllO9l2)

    def _f1109l3(self, _fllO9l2: _cOI19Ol, _fOI09l4: float):
        _fllO9l2.time_in_hole = _fOI09l4
        self.joints_in_hole.append(_fllO9l2)
        self._current_stand_joints.append(_fllO9l2)
        if _fllO9l2 in self.joints_on_rack:
            self.joints_on_rack.remove(_fllO9l2)

    def _f0O09l5(self, _fOI09l4: float):
        self._drilling_start_time = _fOI09l4

    def _f10l9l6(self, _fOI09l4: float) -> _cIO09O3:
        stand = _cIO09O3(joints=self._current_stand_joints.copy(), stand_number=len(self.stands) + 1, time_spud=self._drilling_start_time, time_connection=_fOI09l4)
        self.stands.append(stand)
        self._current_stand_joints = []
        self._drilling_start_time = None
        return stand

    def _fOl19l7(self, _fOI09l4: float) -> Optional[_cOI19Ol]:
        if not self.joints_in_hole:
            return None
        _fllO9l2 = self.joints_in_hole.pop()
        _fllO9l2.time_out_hole = _fOI09l4
        self.joints_on_rack.append(_fllO9l2)
        return _fllO9l2

    def _f1l09l8(self, dimension: int) -> float:
        if dimension > len(self.joints_in_hole):
            dimension = len(self.joints_in_hole)
        pipe_length = sum((j.length for j in self.joints_in_hole[:dimension]))
        return pipe_length + self._fl1O9Oc

    def _fOOl9l9(self, _fO0l9lA: int) -> float:
        if _fO0l9lA > len(self.stands):
            return self._fll09OE
        total_joints = sum((s.dimension for s in self.stands[:_fO0l9lA]))
        return self._f1l09l8(total_joints)

    def _f1019lB(self) -> List[Tuple[float, float]]:
        boundaries = []
        cumulative_depth = self._fl1O9Oc
        for stand in self.stands:
            cumulative_depth += stand._f1OI9O4
            if stand.time_connection is not None:
                boundaries.append((cumulative_depth, stand.time_connection))
        return boundaries

    def _fI019lc(self):
        from jones_framework.domains.drilling.stand_calibration import WellTrajectory, StandBoundary, StandSegment
        trajectory = WellTrajectory(well_name=self._f1O19O9)
        cumulative_depth = self._fl1O9Oc
        prev_time = 0.0
        prev_depth = self._fl1O9Oc
        for stand in self.stands:
            if stand.time_spud is None or stand.time_connection is None:
                continue
            start_boundary = StandBoundary(time=stand.time_spud, depth=prev_depth, stand_number=stand._fO0l9lA, is_connection=False)
            cumulative_depth = prev_depth + stand._f1OI9O4
            end_boundary = StandBoundary(time=stand.time_connection, depth=cumulative_depth, stand_number=stand._fO0l9lA, is_connection=True)
            segment = StandSegment.create(start=start_boundary, end=end_boundary)
            trajectory.stands.append(segment)
            prev_depth = cumulative_depth
            prev_time = stand.time_connection
        return trajectory

    def _fOI19ld(self, _f0019lE: float, _fl0I9lf: float=1.0) -> bool:
        tally_depth = self._fll09OE
        error = abs(_f0019lE - tally_depth)
        return error <= _fl0I9lf

    def _f1OI92O(self, _f0019lE: float) -> float:
        tally_depth = self._fll09OE
        if _f0019lE <= 0:
            return 1.0
        return tally_depth / _f0019lE

    def _f10192l(self) -> Dict:
        return {'well_name': self._f1O19O9, 'joints_in_hole': self._fl109Of, 'joints_on_rack': len(self.joints_on_rack), 'stands_drilled': self._fIl09lO, 'total_pipe_length': self._f1lI9Od, 'bit_depth': self._fll09OE, 'bha_length': self._fl1O9Oc, 'avg_joint_length': self._f1lI9Od / self._fl109Of if self._fl109Of > 0 else 0, 'avg_stand_rop': np.mean([s._f0I09O7 for s in self.stands if s._f0I09O7 is not None]) if self.stands else 0}

def _fO00922(dimension: int, _flOI923: _cO009OO=_cO009OO.RANGE_2, _f00O9OA: int=3) -> _cI119O8:
    length_ranges = {_cO009OO.RANGE_1: (18.0, 22.0), _cO009OO.RANGE_2: (27.0, 30.0), _cO009OO.RANGE_3: (38.0, 45.0)}
    min_len, max_len = length_ranges[_flOI923]
    tally = _cI119O8(joints_per_stand=_f00O9OA)
    for i in range(dimension):
        length = np.random.uniform(min_len, max_len)
        _fllO9l2 = _cOI19Ol(serial_number=f'DP-{i + 1:04d}', length=round(length, 2))
        tally._fI1I9ll(_fllO9l2)
    return tally
if __name__ == '__main__':
    print('=== Pipe Tally Simulation ===\n')
    tally = _fO00922(100, _cO009OO.RANGE_2)
    tally._f1O19O9 = 'Test Well #1'
    tally._fl1O9Oc = 450.0
    print(f'Initial state:')
    print(f'  Joints on rack: {len(tally.joints_on_rack)}')
    print(f'  BHA length: {tally._fl1O9Oc} ft')
    current_time = 0.0
    for stand_num in range(5):
        for _ in range(3):
            _fllO9l2 = tally.joints_on_rack[0]
            tally._f1109l3(_fllO9l2, current_time)
            current_time += 60
        tally._f0O09l5(current_time)
        base_time = 3600
        depth_factor = 1 + tally._fll09OE / 5000
        _f11l9O6 = base_time * depth_factor * np.random.uniform(0.8, 1.2)
        current_time += _f11l9O6
        stand = tally._f10l9l6(current_time)
        print(f'\nStand {stand._fO0l9lA}:')
        print(f'  Length: {stand._f1OI9O4:.1f} ft')
        print(f'  Drilling time: {stand._f11l9O6:.0f}s ({stand._f11l9O6 / 3600:.2f} hr)')
        print(f'  ROP: {stand._f0I09O7:.1f} ft/hr')
        print(f'  Bit depth: {tally._fll09OE:.1f} ft')
        current_time += 600
    print('\n' + '=' * 40)
    print('Final Summary:')
    for k, v in tally._f10192l().items():
        print(f'  {k}: {v}')
    print('\nStand Boundaries (for WellTrajectory):')
    for depth, time in tally._f1019lB():
        print(f'  d={depth:.1f}ft, t={time:.0f}s ({time / 3600:.2f}hr)')
    print('\nConverting to WellTrajectory...')
    trajectory = tally._fI019lc()
    print(f'  Created trajectory with {len(trajectory.stands)} stands')
    test_depth = 1000.0
    test_time = trajectory.time_at_depth(test_depth)
    print(f'\n  Time at {test_depth}ft: {test_time:.0f}s ({test_time / 3600:.2f}hr)')