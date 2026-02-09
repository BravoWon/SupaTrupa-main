from __future__ import annotations
from dataclasses import dataclass
from typing import Union, TypeVar, Generic
from enum import Enum, auto
import numpy as np

class _c0II924(Enum):
    FEET = auto()
    METERS = auto()
    INCHES = auto()

    @property
    def _fI1O925(self) -> float:
        return {_c0II924.FEET: 0.3048, _c0II924.METERS: 1.0, _c0II924.INCHES: 0.0254}[self]

class _cl1I926(Enum):
    SECONDS = auto()
    MINUTES = auto()
    HOURS = auto()

    @property
    def _fOO1927(self) -> float:
        return {_cl1I926.SECONDS: 1.0, _cl1I926.MINUTES: 60.0, _cl1I926.HOURS: 3600.0}[self]

class _c0II928(Enum):
    FT_PER_HR = auto()
    M_PER_HR = auto()
    FT_PER_MIN = auto()
    M_PER_SEC = auto()

    @property
    def _flOO929(self) -> float:
        return {_c0II928.FT_PER_HR: 0.3048 / 3600.0, _c0II928.M_PER_HR: 1.0 / 3600.0, _c0II928.FT_PER_MIN: 0.3048 / 60.0, _c0II928.M_PER_SEC: 1.0}[self]

@dataclass(frozen=True)
class _c0ll92A:
    value: float
    unit: _c0II924 = _c0II924.FEET

    def _f0OI92B(self, _fl1l92c: _c0II924) -> 'Length':
        meters = self.value * self.unit._fI1O925
        return _c0ll92A(meters / _fl1l92c._fI1O925, _fl1l92c)

    def _fI1192d(self) -> float:
        return self.value * self.unit._fI1O925

    def __add__(self, _fl0O92E: 'Length') -> 'Length':
        other_converted = _fl0O92E._f0OI92B(self.unit)
        return _c0ll92A(self.value + other_converted.value, self.unit)

    def __sub__(self, _fl0O92E: 'Length') -> 'Length':
        other_converted = _fl0O92E._f0OI92B(self.unit)
        return _c0ll92A(self.value - other_converted.value, self.unit)

    def __mul__(self, _f0O192f: float) -> 'Length':
        return _c0ll92A(self.value * _f0O192f, self.unit)

    def __truediv__(self, _fl0O92E: Union[float, 'Length']) -> Union['Length', float]:
        if isinstance(_fl0O92E, _c0ll92A):
            return self._fI1192d() / _fl0O92E._fI1192d()
        return _c0ll92A(self.value / _fl0O92E, self.unit)

    def __repr__(self) -> str:
        unit_str = {_c0II924.FEET: 'ft', _c0II924.METERS: 'm', _c0II924.INCHES: 'in'}
        return f'{self.value:.2f} {unit_str[self.unit]}'

@dataclass(frozen=True)
class _cOOO93O:
    value: float
    unit: _cl1I926 = _cl1I926.SECONDS

    def _f0OI92B(self, _fl1l92c: _cl1I926) -> 'Time':
        seconds = self.value * self.unit._fOO1927
        return _cOOO93O(seconds / _fl1l92c._fOO1927, _fl1l92c)

    def _fI1192d(self) -> float:
        return self.value * self.unit._fOO1927

    def __add__(self, _fl0O92E: 'Time') -> 'Time':
        other_converted = _fl0O92E._f0OI92B(self.unit)
        return _cOOO93O(self.value + other_converted.value, self.unit)

    def __sub__(self, _fl0O92E: 'Time') -> 'Time':
        other_converted = _fl0O92E._f0OI92B(self.unit)
        return _cOOO93O(self.value - other_converted.value, self.unit)

    def __mul__(self, _f0O192f: float) -> 'Time':
        return _cOOO93O(self.value * _f0O192f, self.unit)

    def __truediv__(self, _fl0O92E: Union[float, 'Time']) -> Union['Time', float]:
        if isinstance(_fl0O92E, _cOOO93O):
            return self._fI1192d() / _fl0O92E._fI1192d()
        return _cOOO93O(self.value / _fl0O92E, self.unit)

    def __repr__(self) -> str:
        unit_str = {_cl1I926.SECONDS: 's', _cl1I926.MINUTES: 'min', _cl1I926.HOURS: 'hr'}
        return f'{self.value:.2f} {unit_str[self.unit]}'

@dataclass(frozen=True)
class _c11l93l:
    value: float
    unit: _c0II928 = _c0II928.FT_PER_HR

    def _f0OI92B(self, _fl1l92c: _c0II928) -> 'Velocity':
        m_per_sec = self.value * self.unit._flOO929
        return _c11l93l(m_per_sec / _fl1l92c._flOO929, _fl1l92c)

    def _fI1192d(self) -> float:
        return self.value * self.unit._flOO929

    @classmethod
    def _fI11932(cls, _f0ll933: _c0ll92A, _fI11934: _cOOO93O, _fOOO935: _c0II928=_c0II928.FT_PER_HR) -> 'Velocity':
        m_per_sec = _f0ll933._fI1192d() / _fI11934._fI1192d()
        return _c11l93l(m_per_sec / _fOOO935._flOO929, _fOOO935)

    def __mul__(self, _fI11934: _cOOO93O) -> _c0ll92A:
        m_per_sec = self._fI1192d()
        seconds = _fI11934._fI1192d()
        meters = m_per_sec * seconds
        return _c0ll92A(meters / _c0II924.FEET._fI1O925, _c0II924.FEET)

    def _fOlO936(self) -> 'InverseVelocity':
        return InverseVelocity(1.0 / self.value, self._fOOO935)

    def __repr__(self) -> str:
        unit_str = {_c0II928.FT_PER_HR: 'ft/hr', _c0II928.M_PER_HR: 'm/hr', _c0II928.FT_PER_MIN: 'ft/min', _c0II928.M_PER_SEC: 'm/s'}
        return f'{self.value:.2f} {unit_str[self._fOOO935]}'

@dataclass(frozen=True)
class _c000937:
    value: float
    velocity_unit: _c0II928 = _c0II928.FT_PER_HR

    def __mul__(self, _f0ll933: _c0ll92A) -> _cOOO93O:
        if self.velocity_unit == _c0II928.FT_PER_HR:
            length_ft = _f0ll933._f0OI92B(_c0II924.FEET).value
            hours = self.value * length_ft
            return _cOOO93O(hours, _cl1I926.HOURS)
        else:
            rop = _c11l93l(1.0 / self.value, self.velocity_unit)
            m_per_sec = rop._fI1192d()
            sec_per_m = 1.0 / m_per_sec if m_per_sec > 0 else float('inf')
            meters = _f0ll933._fI1192d()
            seconds = sec_per_m * meters
            return _cOOO93O(seconds, _cl1I926.SECONDS)

    def __repr__(self) -> str:
        unit_str = {_c0II928.FT_PER_HR: 'hr/ft', _c0II928.M_PER_HR: 'hr/m'}
        base = unit_str.get(self.velocity_unit, f'1/({self.velocity_unit})')
        return f'{self.value:.4f} {base}'

def _f01l938(_f1I0939: float) -> _c0ll92A:
    return _c0ll92A(_f1I0939, _c0II924.FEET)

def _f1lO93A(_f1I0939: float) -> _c0ll92A:
    return _c0ll92A(_f1I0939, _c0II924.METERS)

def _fI1l93B(_f1I0939: float) -> _cOOO93O:
    return _cOOO93O(_f1I0939, _cl1I926.SECONDS)

def _fl0093c(_f1I0939: float) -> _cOOO93O:
    return _cOOO93O(_f1I0939, _cl1I926.HOURS)

def _fOOO93d(_f1I0939: float) -> _c11l93l:
    return _c11l93l(_f1I0939, _c0II928.FT_PER_HR)

def _f01093E(_f1I0939: float) -> _c11l93l:
    return _c11l93l(_f1I0939, _c0II928.M_PER_HR)
if __name__ == '__main__':
    print('=== Unit-Aware Drilling Calculations ===\n')
    depth_start = _f01l938(5000)
    depth_end = _f01l938(5093)
    delta_d = depth_end - depth_start
    rop = _fOOO93d(50)
    print(f'Depth start: {depth_start}')
    print(f'Depth end: {depth_end}')
    print(f'Delta depth: {delta_d}')
    print(f'ROP: {rop}')
    inv_rop = rop._fOlO936()
    delta_t = inv_rop * delta_d
    print(f'\nInverse ROP: {inv_rop}')
    print(f'Drilling time: {delta_t}')
    print(f'Drilling time (seconds): {delta_t._f0OI92B(_cl1I926.SECONDS)}')
    reconstructed_depth = rop * delta_t
    print(f'\nRound-trip depth: {reconstructed_depth}')
    print(f'\nUnit conversions:')
    print(f'  {depth_start} = {depth_start._f0OI92B(_c0II924.METERS)}')
    print(f'  {rop} = {rop._f0OI92B(_c0II928.M_PER_HR)}')