from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum, auto
import numpy as np
from pathlib import Path
import json
import gc

class _cl1Id7c(Enum):
    NONE = auto()
    LAYER_WISE = auto()
    TENSOR_WISE = auto()
    PIPELINE = auto()
    HYBRID = auto()

@dataclass
class _clO1d7d:
    strategy: _cl1Id7c = _cl1Id7c.LAYER_WISE
    num_shards: int = 2
    gpu_memory_gb: float = 10.0
    cpu_memory_gb: float = 32.0
    offload_to_cpu: bool = True
    offload_to_disk: bool = False
    disk_offload_path: Optional[Path] = None
    async_loading: bool = True
    prefetch_layers: int = 1

    def _f0lld7E(self, _fI1ld7f: float) -> int:
        available = self.gpu_memory_gb * 0.8
        if _fI1ld7f <= available:
            return 1
        return int(np.ceil(_fI1ld7f / available))

@dataclass
class _cOlId8O:
    shard_id: int
    layer_indices: List[int]
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    location: str = 'cpu'
    disk_path: Optional[Path] = None

    @property
    def _fO0Id8l(self) -> int:
        return sum((w.nbytes for w in self.weights.values()))

    @property
    def _fOl0d82(self) -> float:
        return self._fO0Id8l / 1000000000.0

    def _fOlId83(self) -> 'LayerShard':
        if self.location == 'disk':
            self._load_from_disk()
        self.location = 'gpu'
        return self

    def _fI0ld84(self) -> 'LayerShard':
        if self.location == 'disk':
            self._load_from_disk()
        self.location = 'cpu'
        return self

    def _f1l1d85(self, _f1OOd86: Path) -> 'LayerShard':
        self.disk_path = _f1OOd86 / f'shard_{self.shard_id}'
        self.disk_path.mkdir(parents=True, exist_ok=True)
        for name, weight in self.weights.items():
            np.save(self.disk_path / f'{name}.npy', weight)
        metadata = {'shard_id': self.shard_id, 'layer_indices': self.layer_indices, 'weight_names': list(self.weights.keys()), 'shapes': {k: list(v.shape) for k, v in self.weights.items()}}
        with open(self.disk_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        self.weights.clear()
        self.location = 'disk'
        gc.collect()
        return self

    def _f1Old87(self):
        if self.disk_path is None:
            raise ValueError('No disk path set for shard')
        with open(self.disk_path / 'metadata.json') as f:
            metadata = json.load(f)
        self.weights = {}
        for name in metadata['weight_names']:
            self.weights[name] = np.load(self.disk_path / f'{name}.npy')

    def _fIl1d88(self, _fOIld89: np.ndarray, _fOlOd8A: Callable) -> np.ndarray:
        original_location = self.location
        if self.location == 'disk':
            self._f1Old87()
        output = _fOIld89
        for layer_idx in self.layer_indices:
            layer_weights = {k: v for k, v in self.weights.items() if k.startswith(f'layer_{layer_idx}')}
            output = _fOlOd8A(output, layer_weights, layer_idx)
        if original_location == 'disk':
            self.weights.clear()
            gc.collect()
        return output

class _cO01d8B:

    def __init__(self, _f000d8c: _clO1d7d):
        self._f000d8c = _f000d8c
        self.shards: List[_cOlId8O] = []
        self.total_layers = 0
        self._active_shard_idx: Optional[int] = None

    def _f0I1d8d(self, _f1Old8E: List[Dict[str, np.ndarray]], _fIIId8f: Optional[int]=None):
        self.total_layers = len(_f1Old8E)
        total_size_gb = sum((sum((w.nbytes for w in layer.values())) for layer in _f1Old8E)) / 1000000000.0
        if _fIIId8f is None:
            _fIIId8f = self._f000d8c._f0lld7E(total_size_gb)
        layers_per_shard = self.total_layers // _fIIId8f
        remainder = self.total_layers % _fIIId8f
        layer_idx = 0
        for shard_id in range(_fIIId8f):
            n_layers = layers_per_shard + (1 if shard_id < remainder else 0)
            layer_indices = list(range(layer_idx, layer_idx + n_layers))
            weights = {}
            for idx in layer_indices:
                for name, weight in _f1Old8E[idx].items():
                    weights[f'layer_{idx}_{name}'] = weight
            shard = _cOlId8O(shard_id=shard_id, layer_indices=layer_indices, weights=weights, location='cpu')
            if self._f000d8c.offload_to_disk and self._f000d8c.disk_offload_path:
                shard._f1l1d85(self._f000d8c.disk_offload_path)
            self.shards.append(shard)
            layer_idx += n_layers

    def _fIl1d88(self, _fOIld89: np.ndarray, _fOlOd8A: Callable[[np.ndarray, Dict, int], np.ndarray]) -> np.ndarray:
        output = _fOIld89
        for shard_idx, shard in enumerate(self.shards):
            if shard.location != 'gpu':
                if self._active_shard_idx is not None:
                    prev_shard = self.shards[self._active_shard_idx]
                    if self._f000d8c.offload_to_cpu:
                        prev_shard._fI0ld84()
                    elif self._f000d8c.offload_to_disk:
                        prev_shard._f1l1d85(self._f000d8c.disk_offload_path)
                shard._fOlId83()
                self._active_shard_idx = shard_idx
            output = shard._fIl1d88(output, _fOlOd8A)
            if self._f000d8c.async_loading and shard_idx < len(self.shards) - 1:
                pass
        return output

    def _fOIld9O(self) -> Dict[str, Any]:
        gpu_shards = [s for s in self.shards if s.location == 'gpu']
        cpu_shards = [s for s in self.shards if s.location == 'cpu']
        disk_shards = [s for s in self.shards if s.location == 'disk']
        return {'total_shards': len(self.shards), 'total_layers': self.total_layers, 'gpu_shards': len(gpu_shards), 'gpu_memory_gb': sum((s._fOl0d82 for s in gpu_shards)), 'cpu_shards': len(cpu_shards), 'cpu_memory_gb': sum((s._fOl0d82 for s in cpu_shards)), 'disk_shards': len(disk_shards), 'active_shard': self._active_shard_idx}

class _cllld9l:

    def __init__(self, _fOlId92: int, _fI0Od93: int, _fIIId8f: int=2, _fl0Od94: str='row'):
        self._fOlId92 = _fOlId92
        self._fI0Od93 = _fI0Od93
        self._fIIId8f = _fIIId8f
        self._fl0Od94 = _fl0Od94
        self.weight_shards: List[np.ndarray] = []
        self.bias_shards: Optional[List[np.ndarray]] = None
        if _fl0Od94 == 'row':
            shard_size = _fI0Od93 // _fIIId8f
            for i in range(_fIIId8f):
                start = i * shard_size
                end = start + shard_size if i < _fIIId8f - 1 else _fI0Od93
                self.weight_shards.append(np.random.randn(_fOlId92, end - start).astype(np.float32) * 0.01)
        else:
            shard_size = _fOlId92 // _fIIId8f
            for i in range(_fIIId8f):
                start = i * shard_size
                end = start + shard_size if i < _fIIId8f - 1 else _fOlId92
                self.weight_shards.append(np.random.randn(end - start, _fI0Od93).astype(np.float32) * 0.01)

    def _fIl1d88(self, _fOIld89: np.ndarray) -> np.ndarray:
        if self._fl0Od94 == 'row':
            outputs = [_fOIld89 @ shard for shard in self.weight_shards]
            return np.concatenate(outputs, axis=-1)
        else:
            shard_size = self._fOlId92 // self._fIIId8f
            outputs = []
            for i, shard in enumerate(self.weight_shards):
                start = i * shard_size
                end = start + shard.shape[0]
                x_shard = _fOIld89[..., start:end]
                outputs.append(x_shard @ shard)
            return sum(outputs)

    @property
    def _flIOd95(self) -> int:
        return self.weight_shards[0].nbytes

class _cI11d96:

    def __init__(self, _fIlOd97: int, _fOO1d98: int, _fOO1d99: int, _fIOId9A: str='lru'):
        self._fIlOd97 = _fIlOd97
        self._fOO1d98 = _fOO1d98
        self._fOO1d99 = _fOO1d99
        self._fIOId9A = _fIOId9A
        self.experts_per_gpu = max(1, _fOO1d99 // _fOO1d98)
        self.gpu_experts: List[int] = []
        self.cpu_experts: List[int] = list(range(_fIlOd97))
        self.usage_counts: Dict[int, int] = {i: 0 for i in range(_fIlOd97)}
        self.last_used: Dict[int, float] = {i: 0.0 for i in range(_fIlOd97)}

    def _fllOd9B(self, _fIIOd9c: int, _f01Od9d: float=0.0) -> bool:
        self.usage_counts[_fIIOd9c] += 1
        self.last_used[_fIIOd9c] = _f01Od9d
        if _fIIOd9c in self.gpu_experts:
            return False
        while len(self.gpu_experts) >= self.experts_per_gpu:
            self._evict_one()
        if _fIIOd9c in self.cpu_experts:
            self.cpu_experts.remove(_fIIOd9c)
        self.gpu_experts.append(_fIIOd9c)
        return True

    def _f0IId9E(self):
        if not self.gpu_experts:
            return
        if self._fIOId9A == 'lru':
            oldest = min(self.gpu_experts, key=lambda e: self.last_used[e])
        elif self._fIOId9A == 'priority':
            oldest = min(self.gpu_experts, key=lambda e: self.usage_counts[e])
        else:
            oldest = self.gpu_experts[0]
        self.gpu_experts.remove(oldest)
        self.cpu_experts.append(oldest)

    def _f1OOd9f(self, _fO0OdAO: np.ndarray) -> Dict[str, List[int]]:
        if _fO0OdAO.ndim == 1:
            top_experts = np.argsort(_fO0OdAO)[-self.experts_per_gpu:]
        else:
            top_experts = set()
            for weights in _fO0OdAO:
                top_experts.update(np.argsort(weights)[-2:].tolist())
            top_experts = list(top_experts)[:self.experts_per_gpu]
        return {'gpu': top_experts, 'cpu': [i for i in range(self._fIlOd97) if i not in top_experts]}

    def _fO1OdAl(self) -> Dict[str, Any]:
        return {'num_experts': self._fIlOd97, 'experts_per_gpu': self.experts_per_gpu, 'gpu_experts': len(self.gpu_experts), 'cpu_experts': len(self.cpu_experts), 'strategy': self._fIOId9A, 'expert_size_mb': self._fOO1d98 / 1000000.0, 'gpu_budget_mb': self._fOO1d99 / 1000000.0}
if __name__ == '__main__':
    print('=== Model Sharding Demo ===\n')
    _f000d8c = _clO1d7d(strategy=_cl1Id7c.LAYER_WISE, gpu_memory_gb=8.0, offload_to_cpu=True)
    _f1Old8E = []
    for i in range(12):
        _f1Old8E.append({'weight': np.random.randn(1024, 1024).astype(np.float32), 'bias': np.random.randn(1024).astype(np.float32)})
    model = _cO01d8B(_f000d8c)
    model._f0I1d8d(_f1Old8E, num_shards=3)
    print('Sharded model created:')
    for k, v in model._fOIld9O().items():
        print(f'  {k}: {v}')
    print('\n--- Tensor Parallel Linear ---')
    tp_linear = _cllld9l(in_features=768, out_features=3072, num_shards=4, parallel_dim='row')
    _fOIld89 = np.random.randn(32, 512, 768).astype(np.float32)
    y = tp_linear._fIl1d88(_fOIld89)
    print(f'Input shape: {_fOIld89.shape}')
    print(f'Output shape: {y.shape}')
    print(f'Memory per shard: {tp_linear._flIOd95 / 1000000.0:.2f} MB')
    print('\n--- Expert Sharding ---')
    expert_sharding = _cI11d96(num_experts=8, expert_size_bytes=100 * 1024 * 1024, gpu_budget_bytes=400 * 1024 * 1024, strategy='lru')
    print('Initial state:')
    for k, v in expert_sharding._fO1OdAl().items():
        print(f'  {k}: {v}')
    routing = np.array([0.1, 0.3, 0.05, 0.15, 0.2, 0.1, 0.05, 0.05])
    plan = expert_sharding._f1OOd9f(routing)
    print(f'\nPlacement plan for routing: {plan}')