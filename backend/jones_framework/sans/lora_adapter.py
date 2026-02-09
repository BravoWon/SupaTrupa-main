"""
LoRA (Low-Rank Adaptation) for Regime-Specific Expert Switching

This is the canonical reference implementation for hot-swappable
low-rank adapters in state-adaptive systems.

Key concepts:
- LoRALayer: A single low-rank decomposition (W = A @ B)
- LoRAAdapter: A collection of layers for a specific regime
- LoRAAdapterBank: Registry for all adapters, handles activation

Usage:
    # Create adapter for stick-slip regime
    adapter = LoRAAdapter(RegimeID.STICK_SLIP)
    adapter.add_layer(input_dim=64, output_dim=32, rank=8)

    # Register and activate
    bank = LoRAAdapterBank()
    bank.register(adapter)
    bank.activate(RegimeID.STICK_SLIP)

    # Apply adaptation
    output = base_output + adapter.forward(input, layer_idx=0)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from jones_framework.core.activity_state import RegimeID


@dataclass
class LoRALayer:
    """
    Single LoRA layer implementing low-rank decomposition.

    W_adapted = W_base + alpha * (A @ B)

    Where:
    - A: (input_dim, rank) - learned down-projection
    - B: (rank, output_dim) - learned up-projection
    - alpha: scaling factor for adaptation strength

    This allows O(rank * (input + output)) parameters instead of
    O(input * output), enabling fast switching between regimes.
    """
    input_dim: int
    output_dim: int
    rank: int = 8
    alpha: float = 1.0
    name: str = ''
    A: np.ndarray = field(default_factory=lambda: np.array([]))
    B: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Initialize A with small random values, B with zeros (start as identity)."""
        if self.A.size == 0:
            # Kaiming-style initialization scaled down
            self.A = np.random.randn(self.input_dim, self.rank) * 0.01
        if self.B.size == 0:
            # Zero init so initial adaptation is zero
            self.B = np.zeros((self.rank, self.output_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the LoRA adaptation delta.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Adaptation delta to add to base output
        """
        return self.alpha * (x @ self.A @ self.B)

    def get_delta_weights(self) -> np.ndarray:
        """
        Get the full delta weight matrix (for merging into base weights).

        Returns:
            Delta weights of shape (input_dim, output_dim)
        """
        return self.alpha * (self.A @ self.B)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in this layer."""
        return self.A.size + self.B.size

    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer for storage/transmission."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'rank': self.rank,
            'alpha': self.alpha,
            'name': self.name,
            'A': self.A.tolist(),
            'B': self.B.tolist()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LoRALayer:
        """Deserialize layer from storage."""
        layer = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            rank=data['rank'],
            alpha=data['alpha'],
            name=data['name']
        )
        layer.A = np.array(data['A'])
        layer.B = np.array(data['B'])
        return layer


class LoRAAdapter:
    """
    Collection of LoRA layers for a specific regime.

    Each regime (STICK_SLIP, BIT_BOUNCE, etc.) has its own adapter
    that can be hot-swapped without reloading the base model.

    Attributes:
        regime_id: The regime this adapter specializes in
        layers: List of LoRALayer objects
        description: Human-readable description
    """

    def __init__(
        self,
        regime_id: RegimeID,
        layers: Optional[List[LoRALayer]] = None,
        description: str = ''
    ):
        self.regime_id = regime_id
        self.layers = layers or []
        self.description = description
        self._is_active = False

    def add_layer(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 1.0,
        name: str = ''
    ) -> LoRALayer:
        """
        Add a new LoRA layer to this adapter.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            rank: Rank of low-rank decomposition (lower = fewer params)
            alpha: Scaling factor for adaptation strength
            name: Optional layer name

        Returns:
            The created LoRALayer
        """
        layer = LoRALayer(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=rank,
            alpha=alpha,
            name=name or f'layer_{len(self.layers)}'
        )
        self.layers.append(layer)
        return layer

    def forward(self, x: np.ndarray, layer_idx: int = 0) -> np.ndarray:
        """
        Apply LoRA adaptation for a specific layer.

        Args:
            x: Input tensor
            layer_idx: Which layer to apply

        Returns:
            Adaptation delta to add to base output
        """
        if layer_idx >= len(self.layers):
            raise IndexError(f'Layer index {layer_idx} out of range')
        return self.layers[layer_idx].forward(x)

    def forward_all(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply all layers to corresponding inputs.

        Args:
            inputs: List of input tensors, one per layer

        Returns:
            List of adaptation deltas
        """
        if len(inputs) != len(self.layers):
            raise ValueError(f'Expected {len(self.layers)} inputs, got {len(inputs)}')
        return [layer.forward(x) for layer, x in zip(self.layers, inputs)]

    def activate(self):
        """Mark this adapter as active."""
        self._is_active = True

    def deactivate(self):
        """Mark this adapter as inactive."""
        self._is_active = False

    @property
    def is_active(self) -> bool:
        """Whether this adapter is currently active."""
        return self._is_active

    @property
    def total_parameters(self) -> int:
        """Total parameters across all layers."""
        return sum(layer.num_parameters for layer in self.layers)

    def scale_alpha(self, factor: float):
        """Scale the alpha of all layers by a factor."""
        for layer in self.layers:
            layer.alpha *= factor

    def interpolate(self, other: LoRAAdapter, t: float) -> LoRAAdapter:
        """
        Interpolate between this adapter and another.

        Useful for smooth regime transitions.

        Args:
            other: Target adapter
            t: Interpolation factor (0 = self, 1 = other)

        Returns:
            New interpolated adapter
        """
        if len(self.layers) != len(other.layers):
            raise ValueError('Adapters must have same number of layers')

        new_layers = []
        for l1, l2 in zip(self.layers, other.layers):
            if l1.input_dim != l2.input_dim or l1.output_dim != l2.output_dim:
                raise ValueError('Layer dimensions must match for interpolation')

            new_layer = LoRALayer(
                input_dim=l1.input_dim,
                output_dim=l1.output_dim,
                rank=l1.rank,
                alpha=(1 - t) * l1.alpha + t * l2.alpha,
                name=f'{l1.name}_interp'
            )
            new_layer.A = (1 - t) * l1.A + t * l2.A
            new_layer.B = (1 - t) * l1.B + t * l2.B
            new_layers.append(new_layer)

        return LoRAAdapter(
            regime_id=self.regime_id if t < 0.5 else other.regime_id,
            layers=new_layers,
            description=f'Interpolated ({1-t:.1f} {self.regime_id.name} + {t:.1f} {other.regime_id.name})'
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize adapter for storage."""
        return {
            'regime_id': self.regime_id.name,
            'description': self.description,
            'layers': [layer.to_dict() for layer in self.layers]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LoRAAdapter:
        """Deserialize adapter from storage."""
        return cls(
            regime_id=RegimeID[data['regime_id']],
            description=data['description'],
            layers=[LoRALayer.from_dict(l) for l in data['layers']]
        )

    def __repr__(self) -> str:
        return f'LoRAAdapter({self.regime_id.name}, layers={len(self.layers)}, params={self.total_parameters})'


class LoRAAdapterBank:
    """
    Registry and manager for all LoRA adapters.

    Handles:
    - Registering adapters for different regimes
    - Activating/deactivating adapters on regime change
    - Hot-swapping between regimes

    Usage:
        bank = LoRAAdapterBank()
        bank.register(high_vol_adapter)
        bank.register(risk_on_adapter)

        # Later, when regime changes:
        bank.activate(RegimeID.STICK_SLIP)
        current = bank.active_adapter
    """

    def __init__(self):
        self.adapters: Dict[RegimeID, LoRAAdapter] = {}
        self._active_regime: Optional[RegimeID] = None

    def register(self, adapter: LoRAAdapter):
        """Register an adapter for its regime."""
        self.adapters[adapter.regime_id] = adapter

    def get(self, regime_id: RegimeID) -> Optional[LoRAAdapter]:
        """Get adapter for a specific regime."""
        return self.adapters.get(regime_id)

    def activate(self, regime_id: RegimeID) -> LoRAAdapter:
        """
        Activate adapter for a regime, deactivating the current one.

        This is the hot-swap operation - O(1) complexity.

        Args:
            regime_id: Regime to activate

        Returns:
            The activated adapter

        Raises:
            KeyError: If no adapter registered for regime
        """
        # Deactivate current
        if self._active_regime and self._active_regime in self.adapters:
            self.adapters[self._active_regime].deactivate()

        # Activate new
        if regime_id not in self.adapters:
            raise KeyError(f'No adapter registered for {regime_id}')

        self._active_regime = regime_id
        self.adapters[regime_id].activate()
        return self.adapters[regime_id]

    @property
    def active_adapter(self) -> Optional[LoRAAdapter]:
        """Currently active adapter, or None."""
        if self._active_regime:
            return self.adapters.get(self._active_regime)
        return None

    @property
    def active_regime(self) -> Optional[RegimeID]:
        """Currently active regime, or None."""
        return self._active_regime

    def list_adapters(self) -> List[Tuple[RegimeID, str, bool]]:
        """
        List all registered adapters.

        Returns:
            List of (regime_id, description, is_active) tuples
        """
        return [
            (adapter.regime_id, adapter.description, adapter.is_active)
            for adapter in self.adapters.values()
        ]

    def __len__(self) -> int:
        return len(self.adapters)

    def __contains__(self, regime_id: RegimeID) -> bool:
        return regime_id in self.adapters
