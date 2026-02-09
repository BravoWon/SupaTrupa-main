"""Cycle 8: Field-Level Intelligence — multi-well topological atlas.

Builds a topological atlas across multiple wells, enabling field-wide
pattern recognition and cross-well learning.  Each well is characterized
by its 10-dim TDA feature vector (the same fingerprint used in Cycle 4),
plus regime distribution, depth statistics, and windowed evolution.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class WellEntry:
    """A single well in the field atlas."""

    well_id: str
    name: str
    depth_min: float
    depth_max: float
    num_records: int
    feature_vector: List[float]  # 10-dim TDA fingerprint
    regime: str
    confidence: float
    regime_distribution: Dict[str, float]  # regime -> fraction of windows
    windowed_betti: List[Dict[str, float]]  # per-window betti summary
    registered_at: float  # unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WellComparison:
    """Pairwise comparison between two wells."""

    well_a: str
    well_b: str
    topological_distance: float
    feature_deltas: Dict[str, float]
    discriminating_features: List[str]
    regime_similarity: float  # 0-1 (1 = identical regime distributions)
    depth_overlap: float  # fraction of depth range overlap
    interpretation: str


@dataclass
class PatternMatch:
    """A well matching a topological pattern query."""

    well_id: str
    name: str
    distance: float
    regime: str
    confidence: float
    feature_vector: List[float]


# The 10 TDA feature names (same order as Cycle 4)
FEATURE_NAMES = [
    "betti_0",
    "betti_1",
    "entropy_h0",
    "entropy_h1",
    "max_lifetime_h0",
    "max_lifetime_h1",
    "mean_lifetime_h0",
    "mean_lifetime_h1",
    "n_features_h0",
    "n_features_h1",
]


class FieldAtlas:
    """Multi-well topological signature database.

    Stores TDA fingerprints for each registered well, enabling:
    - Field-wide regime pattern visualization
    - Cross-well topological comparison
    - Formation-driven vs drilling-practice-driven pattern identification
    - Offset well pattern search
    """

    def __init__(self) -> None:
        self._wells: Dict[str, WellEntry] = {}

    @property
    def well_count(self) -> int:
        return len(self._wells)

    def register_well(
        self,
        name: str,
        records: List[Dict[str, float]],
        tda_pipeline: Any,
        classifier: Any,
        well_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WellEntry:
        """Register a well by computing its topological signature from drilling records.

        Parameters
        ----------
        name : str
            Human-readable well name.
        records : list of dict
            Drilling records with keys: wob, rpm, rop, torque, spp, depth.
        tda_pipeline : TDAPipeline
            Pipeline instance for feature extraction.
        classifier : RegimeClassifier
            Classifier instance for regime identification.
        well_id : str, optional
            Explicit well ID. Auto-generated if omitted.
        metadata : dict, optional
            Arbitrary metadata to attach.

        Returns
        -------
        WellEntry
            The registered well entry with computed signatures.
        """
        if not records or len(records) < 3:
            raise ValueError("Need at least 3 records to register a well")

        if well_id is None:
            well_id = hashlib.md5(
                f"{name}:{len(records)}:{time.time()}".encode()
            ).hexdigest()[:12]

        # Extract point cloud from drilling fields
        fields = ["wob", "rpm", "rop", "torque", "spp"]
        point_cloud = np.array(
            [[r.get(f, 0.0) for f in fields] for r in records],
            dtype=np.float64,
        )

        # Compute 10-dim TDA feature vector
        features = tda_pipeline.extract_features(point_cloud)
        feature_vector = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]

        # Classify regime
        result = classifier.classify(point_cloud)
        regime = result["regime"]
        confidence = result["confidence"]

        # Compute windowed regime distribution
        regime_dist: Dict[str, int] = {}
        windowed_betti: List[Dict[str, float]] = []

        window_size = min(20, max(3, len(records) // 5))
        stride = max(1, window_size // 2)

        for start in range(0, len(records) - window_size + 1, stride):
            window_pc = point_cloud[start : start + window_size]
            try:
                w_result = classifier.classify(window_pc)
                w_regime = w_result["regime"]
                regime_dist[w_regime] = regime_dist.get(w_regime, 0) + 1

                w_features = tda_pipeline.extract_features(window_pc)
                windowed_betti.append(
                    {
                        "window_index": len(windowed_betti),
                        "betti_0": float(w_features.get("betti_0", 0)),
                        "betti_1": float(w_features.get("betti_1", 0)),
                        "regime": w_regime,
                    }
                )
            except Exception:
                pass

        # Normalize regime distribution to fractions
        total_windows = sum(regime_dist.values()) or 1
        regime_distribution = {
            k: round(v / total_windows, 3) for k, v in regime_dist.items()
        }

        # Depth statistics
        depths = [r.get("depth", 0.0) for r in records]
        depth_min = min(depths) if depths else 0.0
        depth_max = max(depths) if depths else 0.0

        entry = WellEntry(
            well_id=well_id,
            name=name,
            depth_min=round(depth_min, 1),
            depth_max=round(depth_max, 1),
            num_records=len(records),
            feature_vector=feature_vector,
            regime=regime,
            confidence=round(confidence, 3),
            regime_distribution=regime_distribution,
            windowed_betti=windowed_betti,
            registered_at=time.time(),
            metadata=metadata or {},
        )

        self._wells[well_id] = entry
        return entry

    def register_simulated(
        self,
        name: str,
        feature_vector: List[float],
        regime: str,
        confidence: float,
        depth_range: Tuple[float, float] = (0.0, 10000.0),
        num_records: int = 100,
        regime_distribution: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WellEntry:
        """Register a well with pre-computed features (for simulated/offset wells)."""
        well_id = hashlib.md5(
            f"{name}:{time.time()}".encode()
        ).hexdigest()[:12]

        entry = WellEntry(
            well_id=well_id,
            name=name,
            depth_min=round(depth_range[0], 1),
            depth_max=round(depth_range[1], 1),
            num_records=num_records,
            feature_vector=feature_vector,
            regime=regime,
            confidence=round(confidence, 3),
            regime_distribution=regime_distribution or {regime: 1.0},
            windowed_betti=[],
            registered_at=time.time(),
            metadata=metadata or {},
        )

        self._wells[well_id] = entry
        return entry

    def get_atlas(self) -> List[WellEntry]:
        """Return all registered wells, sorted by registration time."""
        return sorted(self._wells.values(), key=lambda w: w.registered_at)

    def compare_wells(
        self,
        well_id_a: str,
        well_id_b: str,
        norm_scale: Optional[np.ndarray] = None,
    ) -> WellComparison:
        """Compare two wells by topological signature and regime distribution."""
        a = self._wells.get(well_id_a)
        b = self._wells.get(well_id_b)
        if not a or not b:
            raise ValueError("Well not found")

        vec_a = np.array(a.feature_vector)
        vec_b = np.array(b.feature_vector)

        # Compute normalized topological distance
        if norm_scale is not None and len(norm_scale) == len(vec_a):
            safe_scale = np.where(norm_scale > 1e-10, norm_scale, 1.0)
            normed_diff = (vec_a - vec_b) / safe_scale
        else:
            normed_diff = vec_a - vec_b

        topo_dist = float(np.linalg.norm(normed_diff))

        # Per-feature deltas
        deltas = {}
        sq_contribs = []
        for i, fname in enumerate(FEATURE_NAMES):
            delta = float(vec_b[i] - vec_a[i])
            deltas[fname] = round(delta, 4)
            sq_contribs.append((fname, normed_diff[i] ** 2))

        # Top 3 discriminating features
        sq_contribs.sort(key=lambda x: x[1], reverse=True)
        discriminating = [name for name, _ in sq_contribs[:3]]

        # Regime distribution similarity (1 - Jensen-Shannon-like divergence)
        all_regimes = set(a.regime_distribution.keys()) | set(
            b.regime_distribution.keys()
        )
        if all_regimes:
            overlap = 0.0
            for r in all_regimes:
                pa = a.regime_distribution.get(r, 0.0)
                pb = b.regime_distribution.get(r, 0.0)
                overlap += min(pa, pb)
            regime_sim = round(overlap, 3)
        else:
            regime_sim = 1.0

        # Depth range overlap
        overlap_min = max(a.depth_min, b.depth_min)
        overlap_max = min(a.depth_max, b.depth_max)
        total_range = max(a.depth_max, b.depth_max) - min(a.depth_min, b.depth_min)
        depth_overlap = round(
            max(0.0, overlap_max - overlap_min) / max(total_range, 1e-10), 3
        )

        # Interpretation
        if topo_dist < 0.5:
            sim_word = "very similar"
        elif topo_dist < 1.5:
            sim_word = "moderately similar"
        elif topo_dist < 3.0:
            sim_word = "different"
        else:
            sim_word = "very different"

        interpretation = (
            f"{a.name} and {b.name} are topologically {sim_word} "
            f"(distance={topo_dist:.2f}). "
            f"Most different in: {', '.join(discriminating)}. "
            f"Regime overlap: {regime_sim * 100:.0f}%. "
            f"Depth overlap: {depth_overlap * 100:.0f}%."
        )

        return WellComparison(
            well_a=well_id_a,
            well_b=well_id_b,
            topological_distance=round(topo_dist, 3),
            feature_deltas=deltas,
            discriminating_features=discriminating,
            regime_similarity=regime_sim,
            depth_overlap=depth_overlap,
            interpretation=interpretation,
        )

    def search_patterns(
        self,
        query_features: List[float],
        top_k: int = 5,
        norm_scale: Optional[np.ndarray] = None,
    ) -> List[PatternMatch]:
        """Find wells whose topological signature is closest to a query pattern."""
        if not self._wells:
            return []

        query = np.array(query_features)
        results: List[Tuple[float, WellEntry]] = []

        for entry in self._wells.values():
            vec = np.array(entry.feature_vector)
            if norm_scale is not None and len(norm_scale) == len(vec):
                safe_scale = np.where(norm_scale > 1e-10, norm_scale, 1.0)
                diff = (query - vec) / safe_scale
            else:
                diff = query - vec
            dist = float(np.linalg.norm(diff))
            results.append((dist, entry))

        results.sort(key=lambda x: x[0])
        top = results[:top_k]

        return [
            PatternMatch(
                well_id=entry.well_id,
                name=entry.name,
                distance=round(dist, 3),
                regime=entry.regime,
                confidence=entry.confidence,
                feature_vector=entry.feature_vector,
            )
            for dist, entry in top
        ]

    def get_field_summary(self) -> Dict[str, Any]:
        """Compute field-wide summary statistics."""
        if not self._wells:
            return {
                "well_count": 0,
                "regime_distribution": {},
                "mean_signature": [],
                "signature_spread": [],
                "depth_range": [0, 0],
            }

        wells = list(self._wells.values())

        # Aggregate regime distribution
        regime_counts: Dict[str, int] = {}
        for w in wells:
            for r, frac in w.regime_distribution.items():
                regime_counts[r] = regime_counts.get(r, 0) + 1

        total = sum(regime_counts.values()) or 1
        field_regime_dist = {
            k: round(v / total, 3) for k, v in regime_counts.items()
        }

        # Mean and spread of feature vectors
        vecs = np.array([w.feature_vector for w in wells])
        mean_sig = np.mean(vecs, axis=0).tolist()
        spread_sig = np.std(vecs, axis=0).tolist()

        # Depth range
        all_min = min(w.depth_min for w in wells)
        all_max = max(w.depth_max for w in wells)

        return {
            "well_count": len(wells),
            "regime_distribution": field_regime_dist,
            "mean_signature": [round(v, 4) for v in mean_sig],
            "signature_spread": [round(v, 4) for v in spread_sig],
            "depth_range": [round(all_min, 1), round(all_max, 1)],
        }
