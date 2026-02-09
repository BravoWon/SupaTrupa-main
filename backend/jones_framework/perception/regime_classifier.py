from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from jones_framework.core.activity_state import RegimeID
from jones_framework.perception.tda_pipeline import TDAPipeline, PersistenceDiagram

@dataclass
class _cOOlA4O:
    regime_id: RegimeID
    reference_diagram: PersistenceDiagram
    feature_vector: np.ndarray
    description: str = ''
    threshold: float = 0.5

@dataclass
class _c00OA4l:
    regime_id: RegimeID
    confidence: float
    distance: float
    all_distances: Dict[RegimeID, float] = field(default_factory=dict)
    is_transition: bool = False

class _cO10A42:

    def __init__(self, _fOO1A43: Optional[TDAPipeline]=None):
        self.tda = _fOO1A43 or TDAPipeline()
        self.signatures: Dict[RegimeID, _cOOlA4O] = {}
        self._norm_scale: Optional[np.ndarray] = None
        self._f1I1A44()

    def _f1I1A44(self):
        self.add_signature(RegimeID.NORMAL, feature_vector=np.array([1, 0, 0.1, 0.0, 1.0, 0.0, 1.0, 0.0, 1, 0]), description='Normal drilling: single connected component, no cycles', threshold=0.3)
        self.add_signature(RegimeID.STICK_SLIP, feature_vector=np.array([3, 2, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3, 5, 3]), description='Stick-slip: torsional oscillation with multiple components and cycles', threshold=0.4)
        self.add_signature(RegimeID.BIT_BOUNCE, feature_vector=np.array([2, 1, 0.5, 0.3, 0.7, 0.5, 0.5, 0.4, 3, 2]), description='Bit bounce: axial vibration with moderate topology', threshold=0.35)
        self.add_signature(RegimeID.PACKOFF, feature_vector=np.array([1, 1, 0.2, 0.2, 0.9, 0.6, 0.8, 0.5, 2, 1]), description='Pack-off: restricted annulus with low entropy', threshold=0.35)
        self.add_signature(RegimeID.WASHOUT, feature_vector=np.array([5, 4, 0.95, 0.9, 0.2, 0.15, 0.1, 0.08, 8, 6]), description='Washout: hole enlargement with fragmenting topology', threshold=0.5)
        self.add_signature(RegimeID.WHIRL, feature_vector=np.array([1, 2, 0.3, 0.5, 0.8, 0.7, 0.7, 0.6, 2, 3]), description='Whirl: lateral vibration with persistent cycles', threshold=0.35)
        self.add_signature(RegimeID.FORMATION_CHANGE, feature_vector=np.array([2, 1, 0.4, 0.2, 0.8, 0.3, 0.6, 0.2, 3, 1]), description='Formation change: directional shift in drilling parameters', threshold=0.35)
        # Fluid regime signatures
        self.add_signature(RegimeID.DARCY_FLOW, feature_vector=np.array([1, 0, 0.05, 0.0, 1.2, 0.0, 1.2, 0.0, 1, 0]), description='Darcy flow: laminar single-phase, very stable topology', threshold=0.25)
        self.add_signature(RegimeID.NON_DARCY_FLOW, feature_vector=np.array([2, 1, 0.3, 0.2, 0.8, 0.3, 0.6, 0.2, 2, 1]), description='Non-Darcy flow: inertial effects creating moderate complexity', threshold=0.35)
        self.add_signature(RegimeID.TURBULENT, feature_vector=np.array([3, 3, 0.7, 0.7, 0.4, 0.3, 0.3, 0.2, 4, 4]), description='Turbulent flow: chaotic mixing with high topological complexity', threshold=0.4)
        self.add_signature(RegimeID.MULTIPHASE, feature_vector=np.array([4, 2, 0.6, 0.5, 0.6, 0.4, 0.4, 0.3, 5, 2]), description='Multiphase flow: gas/liquid/solid phases creating disconnected clusters', threshold=0.4)
        self._compute_normalization()

    def _compute_normalization(self):
        """Compute per-dimension scale from signature std devs so L2 distance
        weights all feature dimensions equally (Betti counts, entropies, lifetimes)."""
        if not self.signatures:
            self._norm_scale = None
            return
        vectors = np.array([s.feature_vector for s in self.signatures.values()], dtype=np.float64)
        stds = np.std(vectors, axis=0)
        # Guard against zero-variance dimensions (replace with 1.0 so they pass through unchanged)
        self._norm_scale = np.where(stds > 1e-10, stds, 1.0)

    def _normalize(self, feature_vector: np.ndarray) -> np.ndarray:
        """Scale feature vector by per-dimension standard deviation."""
        if self._norm_scale is None:
            return feature_vector
        return feature_vector / self._norm_scale

    def add_signature(self, regime_id: RegimeID, reference_diagram: Optional[PersistenceDiagram]=None, feature_vector: Optional[np.ndarray]=None, description: str='', threshold: float=0.5):
        if reference_diagram is None and feature_vector is not None:
            reference_diagram = self._fI0lA4B(feature_vector)
        if reference_diagram is None:
            raise ValueError('Must provide either reference_diagram or feature_vector')
        if feature_vector is None:
            feature_vector = reference_diagram.to_feature_vector()
        self.signatures[regime_id] = _cOOlA4O(regime_id=regime_id, reference_diagram=reference_diagram, feature_vector=feature_vector, description=description, threshold=threshold)

    def _fI0lA4B(self, _fI0OA4c: np.ndarray) -> PersistenceDiagram:
        n_h0 = max(1, int(_fI0OA4c[8]))
        n_h1 = max(0, int(_fI0OA4c[9]))
        h0 = np.zeros((n_h0, 2))
        h0[:, 0] = 0
        h0[:, 1] = _fI0OA4c[4]
        if n_h1 > 0:
            h1 = np.zeros((n_h1, 2))
            h1[:, 0] = np.linspace(0, _fI0OA4c[6] * 0.5, n_h1)
            h1[:, 1] = h1[:, 0] + _fI0OA4c[7]
        else:
            h1 = np.array([]).reshape(0, 2)
        return PersistenceDiagram(h0=h0, h1=h1)

    def classify(self, data: np.ndarray, transition_threshold: float=0.1) -> dict:
        """Classify input data into a regime. Returns dict with regime, confidence, etc."""
        result = self._flIlA4d(data, transition_threshold)
        return {
            'regime': result.regime_id.name,
            'regime_id': result.regime_id,
            'confidence': result.confidence,
            'distance': result.distance,
            'all_distances': {k.name: v for k, v in result.all_distances.items()},
            'is_transition': result.is_transition,
        }

    def _flIlA4d(self, _flI1A4E: np.ndarray, _fI11A4f: float=0.1) -> _c00OA4l:
        current_diagram = self.tda.compute_persistence(_flI1A4E)
        current_features = current_diagram.to_feature_vector()
        norm_current = self._normalize(current_features)
        distances = {}
        for _f10OA46, signature in self.signatures.items():
            norm_sig = self._normalize(signature.feature_vector)
            dist = float(np.linalg.norm(norm_current - norm_sig))
            distances[_f10OA46] = dist
        best_regime = min(distances, key=distances.get)
        best_distance = distances[best_regime]
        max_dist = max(distances.values()) + 1e-10
        confidence = 1.0 - best_distance / max_dist
        sorted_dists = sorted(distances.values())
        is_transition = False
        if len(sorted_dists) >= 2:
            gap = sorted_dists[1] - sorted_dists[0]
            is_transition = gap < _fI11A4f * max_dist
        return _c00OA4l(regime_id=best_regime, confidence=confidence, distance=best_distance, all_distances=distances, is_transition=is_transition)

    def _fOl0A5O(self, _fIOOA5l: PersistenceDiagram) -> _c00OA4l:
        current_features = _fIOOA5l.to_feature_vector()
        norm_current = self._normalize(current_features)
        distances = {}
        for _f10OA46, signature in self.signatures.items():
            norm_sig = self._normalize(signature.feature_vector)
            dist = float(np.linalg.norm(norm_current - norm_sig))
            distances[_f10OA46] = dist
        best_regime = min(distances, key=distances.get)
        best_distance = distances[best_regime]
        max_dist = max(distances.values()) + 1e-10
        confidence = 1.0 - best_distance / max_dist
        return _c00OA4l(regime_id=best_regime, confidence=confidence, distance=best_distance, all_distances=distances, is_transition=False)

    def _fOl1A52(self, _f1OOA53: Dict[RegimeID, List[np.ndarray]]):
        for _f10OA46, point_clouds in _f1OOA53.items():
            if not point_clouds:
                continue
            feature_vectors = []
            for pc in point_clouds:
                _fIOOA5l = self.tda.compute_persistence(pc)
                feature_vectors.append(_fIOOA5l.to_feature_vector())
            avg_features = np.mean(feature_vectors, axis=0)
            norm_avg = self._normalize(avg_features)
            dists = [np.linalg.norm(self._normalize(fv) - norm_avg) for fv in feature_vectors]
            _fOlOA4A = np.mean(dists) + np.std(dists)
            self.add_signature(regime_id=_f10OA46, feature_vector=avg_features, threshold=_fOlOA4A, description=f'Trained signature for {_f10OA46.name}')
        self._compute_normalization()

    def _fOOlA54(self, _f10OA46: RegimeID) -> Optional[_cOOlA4O]:
        return self.signatures.get(_f10OA46)

    def _fO0OA55(self) -> List[Tuple[RegimeID, str]]:
        return [(sig._f10OA46, sig._f1l0A49) for sig in self.signatures.values()]


# Public API aliases for obfuscated classes
RegimeSignature = _cOOlA4O
ClassificationResult = _c00OA4l
RegimeClassifier = _cO10A42