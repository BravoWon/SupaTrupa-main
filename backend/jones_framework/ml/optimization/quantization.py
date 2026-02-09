from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import numpy as np
from functools import lru_cache
import gc

class _c1Olddf(Enum):
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    INT8 = auto()
    INT4 = auto()
    NF4 = auto()

@dataclass
class _cIlOdEO:
    weight_dtype: _c1Olddf = _c1Olddf.INT8
    activation_dtype: _c1Olddf = _c1Olddf.FP16
    compute_dtype: _c1Olddf = _c1Olddf.FP16
    per_channel: bool = True
    calibration_samples: int = 128
    calibration_method: str = 'minmax'
    double_quant: bool = True
    skip_layers: List[str] = field(default_factory=lambda: ['input_embed', 'output_head', 'layernorm'])

    def _f0I1dEl(self, _fIlOdE2: int) -> Dict[str, float]:
        bits = {_c1Olddf.FP32: 32, _c1Olddf.FP16: 16, _c1Olddf.BF16: 16, _c1Olddf.INT8: 8, _c1Olddf.INT4: 4, _c1Olddf.NF4: 4}
        weight_bits = bits[self.weight_dtype]
        weight_mb = _fIlOdE2 * weight_bits / 8 / 1000000.0
        if self.weight_dtype in [_c1Olddf.INT8, _c1Olddf.INT4, _c1Olddf.NF4]:
            metadata_mb = weight_mb * 0.1
        else:
            metadata_mb = 0
        return {'weights_mb': weight_mb, 'metadata_mb': metadata_mb, 'total_mb': weight_mb + metadata_mb, 'compression_ratio': 32 / weight_bits}

@dataclass
class _cIOIdE3:
    data: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray
    dtype: _c1Olddf
    shape: Tuple[int, ...]

    @classmethod
    def from_numpy(cls, _flOOdE5: np.ndarray, _fIlOdE6: _c1Olddf=_c1Olddf.INT8, _fOI0dE7: bool=True, _fI0ldE8: int=0) -> 'QuantizedTensor':
        original_shape = _flOOdE5.shape
        if _fIlOdE6 == _c1Olddf.INT8:
            return cls._quantize_int8(_flOOdE5, _fOI0dE7, _fI0ldE8)
        elif _fIlOdE6 == _c1Olddf.INT4:
            return cls._quantize_int4(_flOOdE5, _fOI0dE7, _fI0ldE8)
        elif _fIlOdE6 == _c1Olddf.NF4:
            return cls._quantize_nf4(_flOOdE5, _fOI0dE7, _fI0ldE8)
        else:
            return cls(data=_flOOdE5.astype(np.float16), scale=np.array([1.0]), zero_point=np.array([0]), dtype=_fIlOdE6, shape=original_shape)

    @classmethod
    def _f1O1dE9(cls, _flOOdE5: np.ndarray, _fOI0dE7: bool, _fI0ldE8: int) -> 'QuantizedTensor':
        if _fOI0dE7:
            _flOOdE5 = np.moveaxis(_flOOdE5, _fI0ldE8, 0)
            n_channels = _flOOdE5.shape[0]
            scales = np.zeros(n_channels, dtype=np.float32)
            quantized = np.zeros(_flOOdE5.shape, dtype=np.int8)
            for i in range(n_channels):
                channel = _flOOdE5[i]
                max_val = np.max(np.abs(channel))
                scale = max_val / 127.0 if max_val > 0 else 1.0
                scales[i] = scale
                quantized[i] = np.round(channel / scale).clip(-128, 127).astype(np.int8)
            quantized = np.moveaxis(quantized, 0, _fI0ldE8)
        else:
            max_val = np.max(np.abs(_flOOdE5))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            scales = np.array([scale])
            quantized = np.round(_flOOdE5 / scale).clip(-128, 127).astype(np.int8)
        return cls(data=quantized, scale=scales, zero_point=np.zeros_like(scales, dtype=np.int8), dtype=_c1Olddf.INT8, shape=_flOOdE5.shape)

    @classmethod
    def _fI10dEA(cls, _flOOdE5: np.ndarray, _fOI0dE7: bool, _fI0ldE8: int) -> 'QuantizedTensor':
        if _fOI0dE7:
            _flOOdE5 = np.moveaxis(_flOOdE5, _fI0ldE8, 0)
            n_channels = _flOOdE5.shape[0]
            scales = np.zeros(n_channels, dtype=np.float32)
            flat_size = _flOOdE5[0].size
            packed_size = (flat_size + 1) // 2
            packed = np.zeros((n_channels, packed_size), dtype=np.uint8)
            for i in range(n_channels):
                channel = _flOOdE5[i].flatten()
                max_val = np.max(np.abs(channel))
                scale = max_val / 7.0 if max_val > 0 else 1.0
                scales[i] = scale
                quantized = np.round(channel / scale).clip(-8, 7).astype(np.int8)
                unsigned = (quantized + 8).astype(np.uint8)
                for j in range(0, len(unsigned) - 1, 2):
                    packed[i, j // 2] = unsigned[j] << 4 | unsigned[j + 1]
                if len(unsigned) % 2 == 1:
                    packed[i, -1] = unsigned[-1] << 4
            _flOOdE5 = np.moveaxis(_flOOdE5, 0, _fI0ldE8)
        else:
            max_val = np.max(np.abs(_flOOdE5))
            scale = max_val / 7.0 if max_val > 0 else 1.0
            scales = np.array([scale])
            flat = _flOOdE5.flatten()
            quantized = np.round(flat / scale).clip(-8, 7).astype(np.int8)
            unsigned = (quantized + 8).astype(np.uint8)
            packed_size = (len(unsigned) + 1) // 2
            packed = np.zeros(packed_size, dtype=np.uint8)
            for j in range(0, len(unsigned) - 1, 2):
                packed[j // 2] = unsigned[j] << 4 | unsigned[j + 1]
            if len(unsigned) % 2 == 1:
                packed[-1] = unsigned[-1] << 4
        return cls(data=packed, scale=scales, zero_point=np.zeros_like(scales, dtype=np.int8), dtype=_c1Olddf.INT4, shape=_flOOdE5.shape)

    @classmethod
    def _fO0IdEB(cls, _flOOdE5: np.ndarray, _fOI0dE7: bool, _fI0ldE8: int) -> 'QuantizedTensor':
        NF4_TABLE = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=np.float32)
        if _fOI0dE7:
            _flOOdE5 = np.moveaxis(_flOOdE5, _fI0ldE8, 0)
            n_channels = _flOOdE5.shape[0]
            scales = np.zeros(n_channels, dtype=np.float32)
            flat_size = _flOOdE5[0].size
            packed_size = (flat_size + 1) // 2
            packed = np.zeros((n_channels, packed_size), dtype=np.uint8)
            for i in range(n_channels):
                channel = _flOOdE5[i].flatten()
                absmax = np.max(np.abs(channel))
                scale = absmax if absmax > 0 else 1.0
                scales[i] = scale
                normalized = channel / scale
                indices = np.argmin(np.abs(normalized[:, np.newaxis] - NF4_TABLE), axis=1).astype(np.uint8)
                for j in range(0, len(indices) - 1, 2):
                    packed[i, j // 2] = indices[j] << 4 | indices[j + 1]
                if len(indices) % 2 == 1:
                    packed[i, -1] = indices[-1] << 4
            _flOOdE5 = np.moveaxis(_flOOdE5, 0, _fI0ldE8)
        else:
            flat = _flOOdE5.flatten()
            absmax = np.max(np.abs(flat))
            scale = absmax if absmax > 0 else 1.0
            scales = np.array([scale])
            normalized = flat / scale
            indices = np.argmin(np.abs(normalized[:, np.newaxis] - NF4_TABLE), axis=1).astype(np.uint8)
            packed_size = (len(indices) + 1) // 2
            packed = np.zeros(packed_size, dtype=np.uint8)
            for j in range(0, len(indices) - 1, 2):
                packed[j // 2] = indices[j] << 4 | indices[j + 1]
            if len(indices) % 2 == 1:
                packed[-1] = indices[-1] << 4
        return cls(data=packed, scale=scales, zero_point=np.zeros_like(scales, dtype=np.int8), dtype=_c1Olddf.NF4, shape=_flOOdE5.shape)

    def _fllIdEc(self) -> np.ndarray:
        if self._fIlOdE6 == _c1Olddf.INT8:
            return self._dequantize_int8()
        elif self._fIlOdE6 == _c1Olddf.INT4:
            return self._dequantize_int4()
        elif self._fIlOdE6 == _c1Olddf.NF4:
            return self._dequantize_nf4()
        else:
            return self.data.astype(np.float32)

    def _fIlIdEd(self) -> np.ndarray:
        result = self.data.astype(np.float32)
        if len(self.scale) > 1:
            scale_shape = [1] * len(self.shape)
            scale_shape[0] = len(self.scale)
            result = result * self.scale.reshape(scale_shape)
        else:
            result = result * self.scale[0]
        return result

    def _f0O0dEE(self) -> np.ndarray:
        total_elements = np.prod(self.shape)
        unpacked = np.zeros(total_elements, dtype=np.float32)
        packed_flat = self.data.flatten()
        for i, byte in enumerate(packed_flat):
            high = (byte >> 4 & 15) - 8
            low = (byte & 15) - 8
            if 2 * i < total_elements:
                unpacked[2 * i] = high
            if 2 * i + 1 < total_elements:
                unpacked[2 * i + 1] = low
        if len(self.scale) > 1:
            unpacked = unpacked.reshape(self.shape)
            scale_shape = [1] * len(self.shape)
            scale_shape[0] = len(self.scale)
            result = unpacked * self.scale.reshape(scale_shape)
        else:
            result = unpacked.reshape(self.shape) * self.scale[0]
        return result

    def _f0I0dEf(self) -> np.ndarray:
        NF4_TABLE = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0], dtype=np.float32)
        total_elements = np.prod(self.shape)
        unpacked = np.zeros(total_elements, dtype=np.float32)
        packed_flat = self.data.flatten()
        for i, byte in enumerate(packed_flat):
            high_idx = byte >> 4 & 15
            low_idx = byte & 15
            if 2 * i < total_elements:
                unpacked[2 * i] = NF4_TABLE[high_idx]
            if 2 * i + 1 < total_elements:
                unpacked[2 * i + 1] = NF4_TABLE[low_idx]
        if len(self.scale) > 1:
            unpacked = unpacked.reshape(self.shape)
            scale_shape = [1] * len(self.shape)
            scale_shape[0] = len(self.scale)
            result = unpacked * self.scale.reshape(scale_shape)
        else:
            result = unpacked.reshape(self.shape) * self.scale[0]
        return result

    @property
    def _fO1IdfO(self) -> int:
        return self.data.nbytes + self.scale.nbytes + self.zero_point.nbytes

class _c101dfl:

    def __init__(self, _fIlOdf2: int, _f1l0df3: int, _fl10df4: int=8, _f110df5: float=1.0, _fOlldf6: Optional[_cIlOdEO]=None, _f10Idf7: str=''):
        self._fIlOdf2 = _fIlOdf2
        self._f1l0df3 = _f1l0df3
        self._fl10df4 = _fl10df4
        self._f110df5 = _f110df5
        self._f10Idf7 = _f10Idf7
        self.config = _fOlldf6 or _cIlOdEO()
        A_fp32 = np.random.randn(_fIlOdf2, _fl10df4).astype(np.float32) * 0.01
        B_fp32 = np.zeros((_fl10df4, _f1l0df3), dtype=np.float32)
        self.A_quant = _cIOIdE3.from_numpy(A_fp32, dtype=self.config.weight_dtype, per_channel=self.config._fOI0dE7)
        self.B_quant = _cIOIdE3.from_numpy(B_fp32, dtype=self.config.weight_dtype, per_channel=self.config._fOI0dE7)

    def _fO1ldf8(self, _f0Ildf9: np.ndarray) -> np.ndarray:
        A = self.A_quant._fllIdEc()
        B = self.B_quant._fllIdEc()
        return self._f110df5 * (_f0Ildf9 @ A @ B)

    def _f10ldfA(self, _c01IdfB: np.ndarray, _cOI0dfc: np.ndarray):
        self.A_quant = _cIOIdE3.from_numpy(_c01IdfB.astype(np.float32), dtype=self.config.weight_dtype, per_channel=self.config._fOI0dE7)
        self.B_quant = _cIOIdE3.from_numpy(_cOI0dfc.astype(np.float32), dtype=self.config.weight_dtype, per_channel=self.config._fOI0dE7)

    @property
    def _fO1IdfO(self) -> int:
        return self.A_quant._fO1IdfO + self.B_quant._fO1IdfO

    @property
    def _fIl1dfd(self) -> float:
        fp32_size = (self._fIlOdf2 * self._fl10df4 + self._fl10df4 * self._f1l0df3) * 4
        return fp32_size / self._fO1IdfO

@dataclass
class _c011dfE:
    total_vram_gb: float
    reserved_system_gb: float = 0.5
    reserved_activations_gb: float = 1.0

    @property
    def _fOI1dff(self) -> float:
        return self.total_vram_gb - self.reserved_system_gb - self.reserved_activations_gb

    def _fO0IEOO(self, _fIlOdE2: int, _fOlldf6: _cIlOdEO) -> Tuple[bool, Dict[str, float]]:
        estimate = _fOlldf6._f0I1dEl(_fIlOdE2)
        model_gb = estimate['total_mb'] / 1024
        fits = model_gb <= self._fOI1dff
        return (fits, {'model_gb': model_gb, 'available_gb': self._fOI1dff, 'headroom_gb': self._fOI1dff - model_gb, 'fits': fits})

    def _fl11EOl(self, _fIlOdE2: int) -> _c1Olddf:
        available_mb = self._fOI1dff * 1024
        for _fIlOdE6 in [_c1Olddf.FP16, _c1Olddf.INT8, _c1Olddf.INT4, _c1Olddf.NF4]:
            config = _cIlOdEO(weight_dtype=_fIlOdE6)
            estimate = config._f0I1dEl(_fIlOdE2)
            if estimate['total_mb'] <= available_mb:
                return _fIlOdE6
        return _c1Olddf.NF4
VRAM_PRESETS = {'rtx_3080_10gb': _c011dfE(total_vram_gb=10.0), 'rtx_4070_12gb': _c011dfE(total_vram_gb=12.0), 'rtx_4080_16gb': _c011dfE(total_vram_gb=16.0), 'a4000_16gb': _c011dfE(total_vram_gb=16.0), 'rtx_3090_24gb': _c011dfE(total_vram_gb=24.0), 'a5000_24gb': _c011dfE(total_vram_gb=24.0), 'cpu_only': _c011dfE(total_vram_gb=0.0, reserved_system_gb=0.0)}

def _flIOEO2(_fOIOEO3: int, _fl1OEO4: str='rtx_4070_12gb', _fOI0EO5: int=32) -> Dict[str, Any]:
    budget = VRAM_PRESETS.get(_fl1OEO4, VRAM_PRESETS['rtx_4070_12gb'])
    recommended_dtype = budget._fl11EOl(_fOIOEO3)
    config = _cIlOdEO(weight_dtype=recommended_dtype, activation_dtype=_c1Olddf.FP16, compute_dtype=_c1Olddf.FP16)
    fits, memory_info = budget._fO0IEOO(_fOIOEO3, config)
    activation_per_sample_mb = 0.5
    available_for_activations_mb = budget.reserved_activations_gb * 1024
    max_batch = int(available_for_activations_mb / activation_per_sample_mb)
    return {'recommended_quantization': recommended_dtype._f10Idf7, 'config': config, 'fits_in_vram': fits, 'memory_breakdown': memory_info, 'max_batch_size': min(max_batch, 128), 'recommended_batch_size': min(_fOI0EO5, max_batch), 'gradient_checkpointing': recommended_dtype in [_c1Olddf.INT4, _c1Olddf.NF4], 'vram_preset': _fl1OEO4}
if __name__ == '__main__':
    print('=== Quantization Module Demo ===\n')
    _flOOdE5 = np.random.randn(768, 768).astype(np.float32)
    print(f'Original tensor: {_flOOdE5.shape}, {_flOOdE5.nbytes / 1000000.0:.2f} MB')
    for _fIlOdE6 in [_c1Olddf.INT8, _c1Olddf.INT4, _c1Olddf.NF4]:
        q = _cIOIdE3.from_numpy(_flOOdE5, dtype=_fIlOdE6)
        reconstructed = q._fllIdEc()
        error = np.mean(np.abs(_flOOdE5 - reconstructed))
        print(f'\n{_fIlOdE6._f10Idf7}:')
        print(f'  Compressed: {q._fO1IdfO / 1000000.0:.2f} MB')
        print(f'  Compression ratio: {_flOOdE5.nbytes / q._fO1IdfO:.1f}x')
        print(f'  Mean abs error: {error:.6f}')
    print('\n--- Enterprise Deployment ---')
    _fOIOEO3 = 100000000
    for preset in ['rtx_3080_10gb', 'rtx_4070_12gb', 'cpu_only']:
        print(f'\n{preset}:')
        result = _flIOEO2(_fOIOEO3, preset)
        print(f"  Recommended: {result['recommended_quantization']}")
        print(f"  Fits: {result['fits_in_vram']}")
        print(f"  Max batch: {result['max_batch_size']}")
        print(f"  Gradient checkpointing: {result['gradient_checkpointing']}")