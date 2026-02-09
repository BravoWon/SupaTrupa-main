from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from jones_framework.core.activity_state import RegimeID
from jones_framework.core.shadow_tensor import ShadowTensor
from jones_framework.perception.tda_pipeline import PersistenceDiagram
from jones_framework.arbitrage.sentiment_vector import SentimentVector, SentimentVectorPipeline, TextDocument, NarrativeType

@dataclass
class ArbitrageSignal:
    signal_type: str
    regime_from: Optional[RegimeID]
    regime_to: Optional[RegimeID]
    confidence: float
    timestamp: datetime
    sentiment_vector: SentimentVector
    geometric_potential: float
    narrative_divergence: float
    trigger_keywords: List[str] = field(default_factory=list)
    description: str = ''

    @property
    def _is_actionable(self) -> bool:
        return self.signal_type == 'trigger' and self.confidence > 0.7

@dataclass
class PotentialEnergy:
    volatility_compression: float
    topological_torsion: float
    persistence_entropy: float
    narrative_tension: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def _total_energy(self) -> float:
        return 0.3 * self.volatility_compression + 0.3 * self.topological_torsion + 0.2 * self.persistence_entropy + 0.2 * self.narrative_tension

    @property
    def _is_coiled(self) -> bool:
        return self._total_energy > 0.6

class LinguisticArbitrageEngine:

    def __init__(self, _f1O15d5: Optional[SentimentVectorPipeline]=None, _fOII5d6: float=0.7, _fll05d7: float=0.5, _f0lO5d8: int=24):
        self.sentiment = _f1O15d5 or SentimentVectorPipeline()
        self._fOII5d6 = _fOII5d6
        self._fll05d7 = _fll05d7
        self.trigger_window = timedelta(hours=_f0lO5d8)
        self._potential_energy: Optional[PotentialEnergy] = None
        self._signal_history: List[ArbitrageSignal] = []
        self._current_regime: RegimeID = RegimeID.NORMAL
        self._trigger_rules: List[Callable[[PotentialEnergy, SentimentVector], Optional[ArbitrageSignal]]] = []

    def process_batch(self, shadow_tensor: ShadowTensor, persistence_diagram: Optional[PersistenceDiagram]=None):
        bbw_values = shadow_tensor.metric_proxy
        compression = 1.0 - np.mean(bbw_values) if len(bbw_values) > 0 else 0.5
        rsi_values = shadow_tensor.tangent_proxy
        divergence_scores = rsi_values[1::2] if len(rsi_values) > 1 else np.array([0])
        torsion = np.max(np.abs(divergence_scores)) if len(divergence_scores) > 0 else 0.0
        if persistence_diagram:
            entropy = persistence_diagram.get_persistence_entropy(1)
        else:
            entropy = 0.5
        narrative_tension = self.sentiment.get_regime_stress()
        self._potential_energy = PotentialEnergy(volatility_compression=float(compression), topological_torsion=float(torsion), persistence_entropy=float(entropy), narrative_tension=narrative_tension)

    def _process_text(self, documents: List[TextDocument]) -> Tuple[SentimentVector, Optional[ArbitrageSignal]]:
        sentiment, details = self.sentiment.process_batch(documents)
        signal = self._check_for_signals(sentiment, details)
        return (sentiment, signal)

    def _check_for_signals(self, sentiment: SentimentVector, details: Dict[str, Any]) -> Optional[ArbitrageSignal]:
        if self._potential_energy is None:
            return None
        if not self._potential_energy._is_coiled:
            return None
        if details.get('divergence', 0) < self._fll05d7:
            return None
        if sentiment.fear > 0.8:
            signal_type = 'trigger'
            confidence = 0.9
            description = 'High fear + coiled spring = imminent regime break'
        elif sentiment.fear > 0.5:
            signal_type = 'warning'
            confidence = 0.7
            description = 'Elevated fear with geometric potential building'
        else:
            signal_type = 'confirmation'
            confidence = 0.5
            description = 'Conditions aligning for potential regime shift'
        regime_to = self._predict_regime_transition(sentiment)
        signal = ArbitrageSignal(signal_type=signal_type, regime_from=self._current_regime, regime_to=regime_to, confidence=confidence, timestamp=datetime.now(), sentiment_vector=sentiment, geometric_potential=self._potential_energy._total_energy, narrative_divergence=details.get('divergence', 0), trigger_keywords=self._extract_trigger_keywords(details), description=description)
        self._signal_history.append(signal)
        for rule in self._trigger_rules:
            custom_signal = rule(self._potential_energy, sentiment)
            if custom_signal:
                return custom_signal
        return signal

    def _predict_regime_transition(self, sentiment: SentimentVector) -> RegimeID:
        if sentiment.fear > 0.7 and sentiment.urgency > 0.5:
            return RegimeID.WASHOUT
        if sentiment.distrust > 0.6:
            return RegimeID.STICK_SLIP
        if sentiment.fear < 0.3 and sentiment.contagion < 0.3:
            return RegimeID.BIT_BOUNCE
        return RegimeID.TRANSITION

    def _extract_trigger_keywords(self, details: Dict[str, Any]) -> List[str]:
        keywords = []
        if details.get('consensus_fear', 0) > 0.5:
            keywords.append('consensus_fear_elevated')
        if details.get('shadow_fear', 0) > 0.5:
            keywords.append('shadow_fear_elevated')
        if details.get('divergence', 0) > 0.6:
            keywords.append('narrative_divergence_high')
        return keywords

    def _add_trigger_rule(self, _fOl15E4: Callable[[PotentialEnergy, SentimentVector], Optional[ArbitrageSignal]]):
        self._trigger_rules.append(_fOl15E4)

    def _get_state(self) -> Dict[str, Any]:
        pe = self._potential_energy
        return {'potential_energy': {'volatility_compression': pe.volatility_compression if pe else 0, 'topological_torsion': pe.topological_torsion if pe else 0, 'persistence_entropy': pe.persistence_entropy if pe else 0, 'narrative_tension': pe.narrative_tension if pe else 0, 'total': pe._total_energy if pe else 0, 'is_coiled': pe._is_coiled if pe else False}, 'regime_stress': self.sentiment.get_regime_stress(), 'current_regime': self._current_regime.name, 'recent_signals': len([s for s in self._signal_history if s.timestamp > datetime.now() - self.trigger_window])}

    def _get_signal_history(self, _f01I5E7: int=100, _fI1I5E8: bool=False) -> List[ArbitrageSignal]:
        signals = self._signal_history[-_f01I5E7:]
        if _fI1I5E8:
            signals = [s for s in signals if s._is_actionable]
        return signals

    def _set_regime(self, _fOl15EA: RegimeID):
        self._current_regime = _fOl15EA

    def _reset(self):
        self._signal_history = []
        self.sentiment._reset()

    @staticmethod
    def _create_fear_spike_rule(_f1O05Ed: float=0.9, _f1II5EE: float=0.7) -> Callable[[PotentialEnergy, SentimentVector], Optional[ArbitrageSignal]]:

        def _fOl15E4(_fOl15Ef: PotentialEnergy, _flO05fO: SentimentVector) -> Optional[ArbitrageSignal]:
            if _flO05fO.fear > _f1O05Ed and _flO05fO.urgency > _f1II5EE:
                return ArbitrageSignal(signal_type='trigger', regime_from=RegimeID.NORMAL, regime_to=RegimeID.WASHOUT, confidence=0.95, timestamp=datetime.now(), sentiment_vector=_flO05fO, geometric_potential=_fOl15Ef._total_energy, narrative_divergence=_fOl15Ef.narrative_tension, description='FEAR SPIKE: Immediate regime break likely')
            return None
        return _fOl15E4

    @staticmethod
    def _create_divergence_collapse_rule(_fll05d7: float=0.8) -> Callable[[PotentialEnergy, SentimentVector], Optional[ArbitrageSignal]]:

        def _fOl15E4(_fOl15Ef: PotentialEnergy, _flO05fO: SentimentVector) -> Optional[ArbitrageSignal]:
            if _fOl15Ef.narrative_tension > _fll05d7 and _flO05fO.divergence < 0.2:
                return ArbitrageSignal(signal_type='trigger', regime_from=RegimeID.TRANSITION, regime_to=RegimeID.STICK_SLIP, confidence=0.85, timestamp=datetime.now(), sentiment_vector=_flO05fO, geometric_potential=_fOl15Ef._total_energy, narrative_divergence=_flO05fO.divergence, description='DIVERGENCE COLLAPSE: Narratives converging on crisis')
            return None
        return _fOl15E4