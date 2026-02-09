from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from jones_framework.core.tensor_ops import Tensor
from jones_framework.core.manifold_bridge import bridge, ConnectionType
from jones_framework.ml.networks.base import NeuralNetwork, NetworkConfig, Activation, Initializer, Parameter

@dataclass
class _c01lE65(NetworkConfig):
    n_heads: int = 8
    n_layers: int = 6
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    use_rotary_embeddings: bool = True
    use_flash_attention: bool = False
    causal: bool = True

class _cI1OE66:

    def __init__(self, _f01lE67: int, _fIIlE68: int, _fO11E69: float=0.0, _flOOE6A: bool=False):
        self._f01lE67 = _f01lE67
        self._fIIlE68 = _fIIlE68
        self.d_head = _f01lE67 // _fIIlE68
        self._fO11E69 = _fO11E69
        self._flOOE6A = _flOOE6A
        self.scale = 1.0 / np.sqrt(self.d_head)
        init_fn = Initializer.xavier
        self.W_q = init_fn((_f01lE67, _f01lE67))
        self.W_k = init_fn((_f01lE67, _f01lE67))
        self.W_v = init_fn((_f01lE67, _f01lE67))
        self.W_o = init_fn((_f01lE67, _f01lE67))

    def _fIl0E6B(self, _fO00E6c: Tensor, _f000E6d: Tensor, _fII0E6E: Tensor, _f0I0E6f: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        batch_size = _fO00E6c.shape[0] if _fO00E6c.ndim > 2 else 1
        seq_len = _fO00E6c.shape[-2] if _fO00E6c.ndim > 1 else 1
        Q = _fO00E6c @ self.W_q
        K = _f000E6d @ self.W_k
        V = _fII0E6E @ self.W_v

        def _fO1IE7O(_f010E7l: Tensor) -> Tensor:
            if _f010E7l.ndim == 2:
                return _f010E7l.reshape(seq_len, self._fIIlE68, self.d_head).transpose(0, 1)
            return _f010E7l.reshape(batch_size, seq_len, self._fIIlE68, self.d_head).transpose(1, 2)
        Q = _fO1IE7O(Q)
        K = _fO1IE7O(K)
        V = _fO1IE7O(V)
        scores = Q @ K.transpose(-2, -1) * self.scale
        if self._flOOE6A:
            causal_mask = Tensor.ones(seq_len, seq_len)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    causal_mask._data[i, j] = float('-inf')
            scores = scores + causal_mask
        if _f0I0E6f is not None:
            scores = scores + _f0I0E6f
        attn_weights = scores.softmax(dim=-1)
        output = attn_weights @ V
        if output.ndim == 4:
            output = output.transpose(1, 2).reshape(batch_size, seq_len, self._f01lE67)
        else:
            output = output.transpose(0, 1).reshape(seq_len, self._f01lE67)
        output = output @ self.W_o
        return (output, attn_weights)

class _c1lIE72:

    def __init__(self, _f01lE67: int, _f00IE73: int, _fO11E69: float=0.0):
        self._f01lE67 = _f01lE67
        self._f00IE73 = _f00IE73
        self._fO11E69 = _fO11E69
        init_fn = Initializer.he
        self.W1 = init_fn((_f01lE67, _f00IE73))
        self.b1 = Tensor.zeros(_f00IE73)
        self.W2 = init_fn((_f00IE73, _f01lE67))
        self.b2 = Tensor.zeros(_f01lE67)

    def _fIl0E6B(self, _f010E7l: Tensor) -> Tensor:
        h = _f010E7l @ self.W1 + self.b1
        h = Activation.gelu(h)
        output = h @ self.W2 + self.b2
        return output

class _c1IIE74:

    def __init__(self, _f01lE67: int, _fIIlE68: int, _f00IE73: int, _fO11E69: float=0.0, _flOOE6A: bool=False):
        self.attention = _cI1OE66(_f01lE67, _fIIlE68, _fO11E69, _flOOE6A)
        self.ff = _c1lIE72(_f01lE67, _f00IE73, _fO11E69)
        self.ln1_gamma = Tensor.ones(_f01lE67)
        self.ln1_beta = Tensor.zeros(_f01lE67)
        self.ln2_gamma = Tensor.ones(_f01lE67)
        self.ln2_beta = Tensor.zeros(_f01lE67)

    def _fIIIE75(self, _f010E7l: Tensor, _f0OOE76: Tensor, _f1OOE77: Tensor) -> Tensor:
        mean = _f010E7l.mean(dim=-1, keepdim=True)
        var = _f010E7l.var(dim=-1, keepdim=True)
        normalized = (_f010E7l - mean) / (var + 1e-05).sqrt()
        return normalized * _f0OOE76 + _f1OOE77

    def _fIl0E6B(self, _f010E7l: Tensor, _f0I0E6f: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        residual = _f010E7l
        _f010E7l = self._fIIIE75(_f010E7l, self.ln1_gamma, self.ln1_beta)
        attn_out, attn_weights = self.attention._fIl0E6B(_f010E7l, _f010E7l, _f010E7l, _f0I0E6f)
        _f010E7l = residual + attn_out
        residual = _f010E7l
        _f010E7l = self._fIIIE75(_f010E7l, self.ln2_gamma, self.ln2_beta)
        ff_out = self.ff._fIl0E6B(_f010E7l)
        _f010E7l = residual + ff_out
        return (_f010E7l, attn_weights)

class _c0O1E78:

    def __init__(self, _f01lE67: int, _flIIE79: int=5000):
        self._f01lE67 = _f01lE67
        self._flIIE79 = _flIIE79
        self.encoding = self._create_encoding()

    def _flOlE7A(self) -> Tensor:
        position = np.arange(self._flIIE79)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self._f01lE67, 2) * -(np.log(10000.0) / self._f01lE67))
        pe = np.zeros((self._flIIE79, self._f01lE67))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return Tensor(pe)

    def _fIl0E6B(self, _f010E7l: Tensor) -> Tensor:
        seq_len = _f010E7l.shape[-2] if _f010E7l.ndim > 1 else _f010E7l.shape[0]
        return _f010E7l + self.encoding[:seq_len]

class _cl10E7B:

    def __init__(self, _f01lE67: int, _flIIE79: int=5000, _f1l1E7c: int=10000):
        self._f01lE67 = _f01lE67
        self._flIIE79 = _flIIE79
        self._f1l1E7c = _f1l1E7c
        inv_freq = 1.0 / _f1l1E7c ** (np.arange(0, _f01lE67, 2) / _f01lE67)
        t = np.arange(_flIIE79)
        freqs = np.outer(t, inv_freq)
        self.cos_cache = Tensor(np.cos(freqs))
        self.sin_cache = Tensor(np.sin(freqs))

    def _fIl0E6B(self, _f010E7l: Tensor, _fII1E7d: int=-2) -> Tensor:
        seq_len = _f010E7l.shape[_fII1E7d]
        d = _f010E7l.shape[-1]
        cos = self.cos_cache[:seq_len, :d // 2]
        sin = self.sin_cache[:seq_len, :d // 2]
        x1 = _f010E7l[..., ::2]
        x2 = _f010E7l[..., 1::2]
        rotated = Tensor.zeros(*_f010E7l.shape.dims)
        rotated._data[..., ::2] = (x1 * cos - x2 * sin)._data
        rotated._data[..., 1::2] = (x2 * cos + x1 * sin)._data
        return rotated

@bridge(connects_to=['NeuralNetwork', 'Tensor', 'LoRAAdapter', 'ConditionState', 'ShadowTensor', 'RegimeClassifier'], connection_types={'NeuralNetwork': ConnectionType.EXTENDS, 'ShadowTensor': ConnectionType.USES, 'RegimeClassifier': ConnectionType.USES})
class _cIOlE7E(NeuralNetwork):

    def __init__(self, _fIIIE7f: _c01lE65):
        self.transformer_config = _fIIIE7f
        self._blocks: List[_c1IIE74] = []
        self._positional_encoding: Optional[_c0O1E78] = None
        self._rotary_embedding: Optional[_cl10E7B] = None
        super().__init__(_fIIIE7f)

    def _flIIE8O(self):
        _f01lE67 = self.transformer_config._f01lE67
        init_fn = Initializer.xavier
        self.register_parameter('input_embedding', init_fn((self._fIIIE7f.input_dim, _f01lE67)))
        if self.transformer_config.use_rotary_embeddings:
            self._rotary_embedding = _cl10E7B(_f01lE67, self.transformer_config.max_seq_len)
        else:
            self._positional_encoding = _c0O1E78(_f01lE67, self.transformer_config.max_seq_len)
        for i in range(self.transformer_config.n_layers):
            block = _c1IIE74(d_model=_f01lE67, n_heads=self.transformer_config._fIIlE68, d_ff=self.transformer_config._f00IE73, dropout=self.transformer_config._fO11E69, causal=self.transformer_config._flOOE6A)
            self._blocks.append(block)
            self.register_parameter(f'block_{i}_ln1_gamma', block.ln1_gamma)
            self.register_parameter(f'block_{i}_ln1_beta', block.ln1_beta)
            self.register_parameter(f'block_{i}_ln2_gamma', block.ln2_gamma)
            self.register_parameter(f'block_{i}_ln2_beta', block.ln2_beta)
        self.register_parameter('final_ln_gamma', Tensor.ones(_f01lE67))
        self.register_parameter('final_ln_beta', Tensor.zeros(_f01lE67))
        self.register_parameter('output_projection', init_fn((_f01lE67, self._fIIIE7f.output_dim)))

    def _fIl0E6B(self, _f010E7l: Tensor, _f0I0E6f: Optional[Tensor]=None, _f1l1E8l: bool=False) -> Tensor:
        _f010E7l = _f010E7l @ self._parameters['input_embedding'].data
        if self._rotary_embedding is not None:
            _f010E7l = self._rotary_embedding._fIl0E6B(_f010E7l)
        elif self._positional_encoding is not None:
            _f010E7l = self._positional_encoding._fIl0E6B(_f010E7l)
        attention_weights = []
        for i, block in enumerate(self._blocks):
            block.ln1_gamma = self._parameters[f'block_{i}_ln1_gamma'].data
            block.ln1_beta = self._parameters[f'block_{i}_ln1_beta'].data
            block.ln2_gamma = self._parameters[f'block_{i}_ln2_gamma'].data
            block.ln2_beta = self._parameters[f'block_{i}_ln2_beta'].data
            _f010E7l, attn = block._fIl0E6B(_f010E7l, _f0I0E6f)
            attention_weights.append(attn)
        mean = _f010E7l.mean(dim=-1, keepdim=True)
        var = _f010E7l.var(dim=-1, keepdim=True)
        _f010E7l = (_f010E7l - mean) / (var + 1e-05).sqrt()
        _f010E7l = _f010E7l * self._parameters['final_ln_gamma'].data + self._parameters['final_ln_beta'].data
        if _f010E7l.ndim == 3:
            _f010E7l = _f010E7l[:, -1, :]
        elif _f010E7l.ndim == 2:
            _f010E7l = _f010E7l[-1, :]
        output = _f010E7l @ self._parameters['output_projection'].data
        if _f1l1E8l:
            return (output, attention_weights)
        return output

    def _fO0IE82(self, _f010E7l: Tensor) -> List[Tensor]:
        _, attention_weights = self._fIl0E6B(_f010E7l, return_attention=True)
        return attention_weights

class _cOOOE83(_cIOlE7E):

    def __init__(self, _fIIIE7f: _c01lE65, _flOIE84: int=6):
        self._flOIE84 = _flOIE84
        super().__init__(_fIIIE7f)

    def _flIIE8O(self):
        super()._flIIE8O()
        for scale in [1, 4, 16]:
            self.register_parameter(f'scale_{scale}_conv', Initializer.xavier((self._fIIIE7f.input_dim, self.transformer_config._f01lE67 // 3)))
        _f01lE67 = self.transformer_config._f01lE67
        self.register_parameter('regime_head', Initializer.xavier((_f01lE67, self._flOIE84)))
        self.register_parameter('confidence_head', Initializer.xavier((_f01lE67, 1)))

    def _fIl0E6B(self, _f010E7l: Tensor, _f0I0E6f: Optional[Tensor]=None) -> Dict[str, Tensor]:
        main_output = super()._fIl0E6B(_f010E7l, _f0I0E6f)
        hidden = _f010E7l @ self._parameters['input_embedding'].data
        if self._positional_encoding:
            hidden = self._positional_encoding._fIl0E6B(hidden)
        for block in self._blocks:
            hidden, _ = block._fIl0E6B(hidden, _f0I0E6f)
        if hidden.ndim == 3:
            final_hidden = hidden[:, -1, :]
        else:
            final_hidden = hidden[-1, :]
        regime_logits = final_hidden @ self._parameters['regime_head'].data
        regime_probs = regime_logits.softmax(dim=-1)
        confidence = (final_hidden @ self._parameters['confidence_head'].data).sigmoid()
        return {'output': main_output, 'regime_logits': regime_logits, 'regime_probs': regime_probs, 'confidence': confidence}