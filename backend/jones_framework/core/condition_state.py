from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class ConditionState:
    timestamp: int
    vector: Tuple[float, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    verified: bool = True
    state_id: str = field(default='', compare=False)

    def __post_init__(self):
        if not self.state_id:
            state_hash = self._compute_hash()
            object.__setattr__(self, 'state_id', state_hash)

    def _compute_hash(self) -> str:
        data = f'{self.timestamp}:{self.vector}:{sorted(self.metadata.items())}'
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @classmethod
    def from_market(cls, vector: np.ndarray, metadata: Optional[Dict[str, Any]]=None, timestamp: Optional[int]=None, verified: bool=True) -> ConditionState:
        return cls(timestamp=timestamp or int(time.time() * 1000), vector=tuple(vector.flatten().tolist()), metadata=metadata or {}, verified=verified)

    @classmethod
    def _fll0c67(cls, lithology: str, porosity: float, permeability: float, saturation: float, location: Tuple[float, float, float], timestamp: Optional[int]=None) -> ConditionState:
        lithology_map = {'sandstone': 0.0, 'shale': 1.0, 'carbonate': 2.0}
        lithology_val = lithology_map.get(lithology.lower(), 0.0)
        vector = (lithology_val, porosity, permeability, saturation)
        metadata = {'domain': 'reservoir', 'lithology_name': lithology, 'location_x': location[0], 'location_y': location[1], 'location_z': location[2]}
        return cls(timestamp=timestamp or int(time.time() * 1000), vector=vector, metadata=metadata, verified=True)

    @classmethod
    def _f0I0c6d(cls, price: float, volume: float, bid: float, ask: float, symbol: str, timestamp: Optional[int]=None) -> ConditionState:
        spread = ask - bid
        mid = (bid + ask) / 2
        vector = (price, volume, bid, ask, spread, mid)
        metadata = {'domain': 'market', 'symbol': symbol, 'spread_bps': spread / mid * 10000 if mid > 0 else 0}
        return cls(timestamp=timestamp or int(time.time() * 1000), vector=vector, metadata=metadata, verified=True)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.vector)

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def distance_to(self, other: ConditionState) -> float:
        if self.dimension != other.dimension:
            raise ValueError('Cannot compute distance between states of different dimensions')
        return float(np.linalg.norm(self.to_numpy() - other.to_numpy()))

    def __repr__(self) -> str:
        domain = self.metadata.get('domain', 'generic')
        return f'ConditionState({self.state_id}, domain={domain}, dim={self.dimension})'