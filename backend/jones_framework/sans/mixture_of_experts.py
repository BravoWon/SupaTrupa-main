"""
Mixture of Experts (MoE) with Regime-Based Expert Selection

Canonical reference implementation for state-adaptive expert routing.

Key concepts:
- Expert: A specialized model for a specific regime
- ExpertConfig: Associates an expert with regime and LoRA adapter
- MixtureOfExperts: Routes inputs to appropriate expert based on detected regime

The key innovation is combining TDA-based regime detection with
LoRA hot-swapping for instant expert switching.

Usage:
    moe = MixtureOfExperts()

    # Process with automatic regime detection and expert selection
    output, regime = moe.process(state, telemetry_window)

    # Or manually switch experts
    moe.switch_to_regime(RegimeID.STICK_SLIP)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import numpy as np
from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID, ExpertModel
from jones_framework.perception.regime_classifier import RegimeClassifier, ClassificationResult
from jones_framework.sans.lora_adapter import LoRAAdapter, LoRAAdapterBank


class Expert(ExpertModel):
    """
    A regime-specialized expert model.

    Each expert is a small MLP that can be augmented with LoRA adapters
    for regime-specific behavior without changing base weights.

    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.weights = self._init_weights()
        self._regime_id = RegimeID.NORMAL
        self._lora_adapter: Optional[LoRAAdapter] = None

    def _init_weights(self) -> List[np.ndarray]:
        """Initialize weights with Xavier initialization."""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        weights = []
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = np.random.randn(dims[i], dims[i + 1]) * scale
            weights.append(w)
        return weights

    def attach_adapter(self, adapter: LoRAAdapter):
        """Attach a LoRA adapter for regime specialization."""
        self._lora_adapter = adapter
        self._regime_id = adapter.regime_id

    def detach_adapter(self):
        """Remove the current LoRA adapter."""
        self._lora_adapter = None

    def get_regime_id(self) -> RegimeID:
        """Get the regime this expert is specialized for."""
        return self._regime_id

    # Abstract method implementations (obfuscated names from ExpertModel ABC)
    def _fI0lc4E(self, state: ConditionState) -> np.ndarray:
        return self.forward(state)

    def _fI0Oc5O(self) -> RegimeID:
        return self.get_regime_id()

    def forward(self, state: ConditionState) -> np.ndarray:
        """
        Forward pass through the expert.

        Args:
            state: Input condition state

        Returns:
            Output vector
        """
        x = state.to_numpy()

        # Pad or truncate to match input dimension
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        elif len(x) > self.input_dim:
            x = x[:self.input_dim]

        # Forward through layers with optional LoRA
        h = x
        for i, w in enumerate(self.weights[:-1]):
            layer_input = h  # Save pre-multiplication input for LoRA
            h = h @ w
            # Add LoRA adaptation if attached (applied before activation)
            if self._lora_adapter and i < len(self._lora_adapter.layers):
                h = h + self._lora_adapter.layers[i].forward(layer_input)
            h = np.maximum(0, h)  # ReLU

        # Output layer (no activation)
        output = h @ self.weights[-1]
        return output

    def forward_batch(self, states: List[ConditionState]) -> np.ndarray:
        """Process a batch of states."""
        return np.array([self.forward(s) for s in states])


@dataclass
class ExpertConfig:
    """Configuration for a registered expert."""
    regime_id: RegimeID
    expert: Expert
    adapter: Optional[LoRAAdapter] = None
    priority: int = 1
    description: str = ''


class MixtureOfExperts:
    """
    Mixture of Experts with regime-based routing.

    This is the main orchestrator that:
    1. Uses TDA pipeline to detect current regime
    2. Routes to appropriate expert
    3. Hot-swaps LoRA adapters for instant regime changes

    Attributes:
        classifier: RegimeClassifier for detecting regimes
        experts: Dict mapping RegimeID to ExpertConfig
        adapter_bank: Bank of all LoRA adapters
    """

    def __init__(
        self,
        classifier: Optional[RegimeClassifier] = None,
        input_dim: int = 6,
        output_dim: int = 4
    ):
        self.classifier = classifier or RegimeClassifier()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.experts: Dict[RegimeID, ExpertConfig] = {}
        self.adapter_bank = LoRAAdapterBank()
        self._active_expert: Optional[ExpertConfig] = None
        self._transition_history: List[Tuple[RegimeID, float, float]] = []
        self._init_default_experts()

    def _init_default_experts(self):
        """Initialize default experts for common regimes."""
        default_regimes = [
            (RegimeID.NORMAL, 'Normal drilling regime expert'),
            (RegimeID.STICK_SLIP, 'Stick-slip torsional vibration expert'),
            (RegimeID.BIT_BOUNCE, 'Bit bounce axial vibration expert'),
            (RegimeID.PACKOFF, 'Pack-off restricted annulus expert'),
            (RegimeID.FORMATION_CHANGE, 'Formation change trending expert'),
            (RegimeID.WHIRL, 'Whirl lateral vibration expert'),
        ]

        for regime_id, description in default_regimes:
            expert = Expert(self.input_dim, self.output_dim)

            # Create LoRA adapter for this regime
            adapter = LoRAAdapter(
                regime_id=regime_id,
                description=f'LoRA adapter for {regime_id.name}'
            )
            adapter.add_layer(self.input_dim, 64, rank=4)
            adapter.add_layer(64, 32, rank=4)

            self.register_expert(regime_id, expert, adapter, description)

    def register_expert(
        self,
        regime_id: RegimeID,
        expert: Expert,
        adapter: Optional[LoRAAdapter] = None,
        description: str = '',
        priority: int = 1
    ):
        """
        Register an expert for a regime.

        Args:
            regime_id: The regime this expert handles
            expert: The Expert model
            adapter: Optional LoRA adapter for specialization
            description: Human-readable description
            priority: Priority for expert selection (higher = preferred)
        """
        if adapter:
            expert.attach_adapter(adapter)
            self.adapter_bank.register(adapter)

        self.experts[regime_id] = ExpertConfig(
            regime_id=regime_id,
            expert=expert,
            adapter=adapter,
            priority=priority,
            description=description
        )

    def classify_and_select(
        self,
        telemetry: np.ndarray
    ) -> Tuple[ClassificationResult, ExpertConfig]:
        """
        Classify regime and select appropriate expert.

        Args:
            telemetry: Window of telemetry data

        Returns:
            Tuple of (classification result, selected expert config)
        """
        result = self.classifier.classify(telemetry)
        expert = self._select_expert(result)
        return result, expert

    def _select_expert(self, result: ClassificationResult) -> ExpertConfig:
        """
        Select expert based on classification result.

        Uses confidence and priority for improved selection:
        - Low confidence (<0.5): prefer STABLE expert to avoid jitter
        - During transitions: consider multiple candidate experts
        - Use priority field to break ties between valid experts
        """
        regime = result.regime_id
        confidence = getattr(result, 'confidence', 1.0)

        # Low confidence - prefer stable/safe expert to avoid regime jitter
        if confidence < 0.5 and RegimeID.NORMAL in self.experts:
            return self.experts[RegimeID.NORMAL]

        # Build candidate list based on regime and transition state
        candidates = []

        # Primary: exact regime match
        if regime in self.experts:
            candidates.append(self.experts[regime])

        # During transitions, consider alternative regimes from all_distances
        if getattr(result, 'is_transition', False) and hasattr(result, 'all_distances'):
            sorted_regimes = sorted(result.all_distances.items(), key=lambda x: x[1])
            for alt_regime, _ in sorted_regimes[:3]:  # Top 3 closest
                if alt_regime in self.experts and alt_regime != regime:
                    candidates.append(self.experts[alt_regime])

        # Select best candidate by priority (higher = better)
        if candidates:
            return max(candidates, key=lambda e: e.priority)

        # Fallback to stable
        if RegimeID.NORMAL in self.experts:
            return self.experts[RegimeID.NORMAL]

        # Last resort: first available
        return list(self.experts.values())[0]

    def switch_to_regime(self, regime_id: RegimeID) -> ExpertConfig:
        """
        Explicitly switch to a specific regime's expert.

        This performs the hot-swap operation.

        Args:
            regime_id: Target regime

        Returns:
            The activated expert config

        Raises:
            KeyError: If no expert for regime
        """
        if regime_id not in self.experts:
            raise KeyError(f'No expert registered for {regime_id}')

        old_regime = self._active_expert.regime_id if self._active_expert else None

        # Hot-swap the LoRA adapter
        if regime_id in [e.regime_id for e in self.experts.values() if e.adapter]:
            self.adapter_bank.activate(regime_id)

        self._active_expert = self.experts[regime_id]

        # Record transition
        self._transition_history.append((regime_id, time.time(), 1.0))

        return self._active_expert

    def process(
        self,
        state: ConditionState,
        telemetry: Optional[np.ndarray] = None,
        auto_switch: bool = True
    ) -> Tuple[np.ndarray, RegimeID]:
        """
        Process a state through the appropriate expert.

        This is the main entry point. It:
        1. Classifies the regime (if telemetry provided)
        2. Switches to appropriate expert (if auto_switch)
        3. Runs the state through the expert

        Args:
            state: Current condition state
            telemetry: Optional window of telemetry for classification
            auto_switch: Whether to auto-switch experts based on regime

        Returns:
            Tuple of (expert output, current regime)
        """
        # Classify and potentially switch
        if telemetry is not None and auto_switch:
            result, expert_config = self.classify_and_select(telemetry)
            if self._active_expert is None or expert_config.regime_id != self._active_expert.regime_id:
                self.switch_to_regime(expert_config.regime_id)

        # Ensure we have an active expert
        if self._active_expert is None:
            self.switch_to_regime(RegimeID.NORMAL)

        # Process through expert
        output = self._active_expert.expert.forward(state)
        return output, self._active_expert.regime_id

    def process_sequence(
        self,
        states: List[ConditionState],
        window_size: int = 20
    ) -> List[Tuple[np.ndarray, RegimeID]]:
        """
        Process a sequence of states with rolling regime detection.

        Args:
            states: List of condition states
            window_size: Window size for regime classification

        Returns:
            List of (output, regime) tuples
        """
        results = []
        for i, state in enumerate(states):
            if i >= window_size:
                recent = states[i - window_size:i]
                telemetry = np.array([s.to_numpy() for s in recent])
            else:
                telemetry = None

            output, regime = self.process(state, telemetry)
            results.append((output, regime))

        return results

    @property
    def current_regime(self) -> Optional[RegimeID]:
        """Currently active regime."""
        return self._active_expert.regime_id if self._active_expert else None

    def get_transition_history(self) -> List[Tuple[RegimeID, float, float]]:
        """Get history of regime transitions."""
        return self._transition_history.copy()

    def list_experts(self) -> List[Tuple[RegimeID, str, bool]]:
        """
        List all registered experts.

        Returns:
            List of (regime_id, description, is_active)
        """
        current = self.current_regime
        return [
            (config.regime_id, config.description, config.regime_id == current)
            for config in self.experts.values()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime transitions and expert usage."""
        regime_counts = {}
        for regime, _, _ in self._transition_history:
            regime_counts[regime.name] = regime_counts.get(regime.name, 0) + 1

        return {
            'total_transitions': len(self._transition_history),
            'regime_counts': regime_counts,
            'num_experts': len(self.experts),
            'current_regime': self.current_regime.name if self.current_regime else None,
        }
