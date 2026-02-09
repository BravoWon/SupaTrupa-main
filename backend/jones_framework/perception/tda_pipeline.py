from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from functools import lru_cache

try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
try:
    import persim
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False


# =============================================================================
# Persistence Landscape - Functional TDA Representation
# =============================================================================

@dataclass
class PersistenceLandscape:
    """
    Persistence Landscape: A stable, functional representation of persistence diagrams.

    The k-th landscape function λ_k(t) is the k-th largest value of the tent functions
    centered at each persistence pair. This enables:
    - Statistical operations (mean, variance) on persistence
    - Vectorization for ML pipelines
    - Stable distance computations

    Mathematical Definition:
        For a persistence pair (b, d), the tent function is:
        Λ(t) = max(0, min(t - b, d - t))

        λ_k(t) = k-th largest Λ_i(t) across all pairs i
    """
    landscapes: np.ndarray  # Shape: (k, resolution) - k landscape functions
    t_values: np.ndarray    # Shape: (resolution,) - evaluation points
    homology_dim: int       # Which H_n this represents
    birth_death_pairs: Optional[np.ndarray] = None  # Original diagram

    @property
    def num_landscapes(self) -> int:
        """Number of landscape functions."""
        return self.landscapes.shape[0]

    @property
    def resolution(self) -> int:
        """Number of sample points per landscape."""
        return self.landscapes.shape[1]

    def norm(self, p: float = 2) -> float:
        """
        Compute L^p norm of the landscape.

        For p=2, this is the standard Hilbert space norm.
        """
        if p == float('inf'):
            return float(np.max(np.abs(self.landscapes)))
        dt = self.t_values[1] - self.t_values[0] if len(self.t_values) > 1 else 1.0
        return float(np.sum(np.abs(self.landscapes) ** p * dt) ** (1/p))

    def inner_product(self, other: 'PersistenceLandscape') -> float:
        """Compute L^2 inner product between landscapes."""
        if self.landscapes.shape != other.landscapes.shape:
            raise ValueError("Landscapes must have same shape for inner product")
        dt = self.t_values[1] - self.t_values[0] if len(self.t_values) > 1 else 1.0
        return float(np.sum(self.landscapes * other.landscapes) * dt)

    def to_vector(self) -> np.ndarray:
        """Flatten landscape to feature vector."""
        return self.landscapes.flatten()

    @staticmethod
    def average(landscapes: List['PersistenceLandscape']) -> 'PersistenceLandscape':
        """Compute pointwise average of multiple landscapes."""
        if not landscapes:
            raise ValueError("Cannot average empty list of landscapes")
        avg = np.mean([L.landscapes for L in landscapes], axis=0)
        return PersistenceLandscape(
            landscapes=avg,
            t_values=landscapes[0].t_values.copy(),
            homology_dim=landscapes[0].homology_dim
        )


# =============================================================================
# Persistence Silhouette - Weighted Summary Statistics
# =============================================================================

@dataclass
class PersistenceSilhouette:
    """
    Persistence Silhouette: Weighted power mean of landscape functions.

    The silhouette is defined as:
        φ_p(t) = (Σ_k w_k λ_k(t)^p)^(1/p) / Σ_k w_k

    Where w_k are weights (default: persistence of k-th feature).
    This provides a single curve summarizing the entire persistence structure.

    Properties:
    - p=1: Arithmetic mean (emphasizes all features equally)
    - p=2: Root mean square (emphasizes larger features)
    - p→∞: Maximum (λ_1 dominates)
    """
    silhouette: np.ndarray      # Shape: (resolution,)
    t_values: np.ndarray        # Shape: (resolution,)
    power: float                # Weighting power parameter
    homology_dim: int
    weights: Optional[np.ndarray] = None  # Feature weights used

    @property
    def resolution(self) -> int:
        return len(self.silhouette)

    def norm(self, p: float = 2) -> float:
        """Compute L^p norm of silhouette."""
        dt = self.t_values[1] - self.t_values[0] if len(self.t_values) > 1 else 1.0
        if p == float('inf'):
            return float(np.max(np.abs(self.silhouette)))
        return float(np.sum(np.abs(self.silhouette) ** p * dt) ** (1/p))

    def to_vector(self) -> np.ndarray:
        """Return silhouette as feature vector."""
        return self.silhouette.copy()

    @staticmethod
    def distance(s1: 'PersistenceSilhouette', s2: 'PersistenceSilhouette', p: float = 2) -> float:
        """Compute L^p distance between silhouettes."""
        if len(s1.silhouette) != len(s2.silhouette):
            raise ValueError("Silhouettes must have same resolution")
        dt = s1.t_values[1] - s1.t_values[0] if len(s1.t_values) > 1 else 1.0
        diff = s1.silhouette - s2.silhouette
        if p == float('inf'):
            return float(np.max(np.abs(diff)))
        return float(np.sum(np.abs(diff) ** p * dt) ** (1/p))


# =============================================================================
# Persistence Image - 2D Discretization for ML
# =============================================================================

@dataclass
class PersistenceImage:
    """
    Persistence Image: Stable vectorization of persistence diagrams.

    Converts birth-death pairs to a 2D image by:
    1. Transform to birth-persistence coordinates: (b, d) → (b, d-b)
    2. Weight by persistence: w(b, p) = p (or custom weighting)
    3. Convolve with Gaussian kernel
    4. Discretize on a grid

    This is ML-friendly: fixed-size vectors, differentiable, stable.
    """
    image: np.ndarray           # Shape: (resolution, resolution)
    birth_range: Tuple[float, float]
    persistence_range: Tuple[float, float]
    sigma: float                # Gaussian kernel bandwidth
    homology_dim: int
    resolution: int = field(default=50)

    def to_vector(self) -> np.ndarray:
        """Flatten image to feature vector."""
        return self.image.flatten()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.image.shape

    @staticmethod
    def distance(img1: 'PersistenceImage', img2: 'PersistenceImage') -> float:
        """Frobenius norm distance between persistence images."""
        return float(np.linalg.norm(img1.image - img2.image, 'fro'))

    def entropy(self) -> float:
        """Compute image entropy (information content)."""
        img_norm = self.image / (np.sum(self.image) + 1e-10)
        img_norm = img_norm[img_norm > 0]
        return float(-np.sum(img_norm * np.log(img_norm + 1e-10)))


# =============================================================================
# Betti Curve - Betti numbers as function of filtration
# =============================================================================

@dataclass
class BettiCurve:
    """
    Betti Curve: Track Betti numbers across filtration values.

    β_n(t) = number of n-dimensional holes at filtration value t

    This captures how topology evolves through the filtration,
    complementing point summaries like total Betti numbers.
    """
    curve: np.ndarray           # Shape: (resolution,) - Betti numbers
    t_values: np.ndarray        # Shape: (resolution,) - filtration values
    homology_dim: int

    @property
    def max_betti(self) -> int:
        """Maximum Betti number across filtration."""
        return int(np.max(self.curve))

    @property
    def total_persistence(self) -> float:
        """Area under Betti curve (total persistence)."""
        dt = self.t_values[1] - self.t_values[0] if len(self.t_values) > 1 else 1.0
        return float(np.sum(self.curve) * dt)

    def to_vector(self) -> np.ndarray:
        """Return curve as feature vector."""
        return self.curve.copy()


# =============================================================================
# Extended Topological Features
# =============================================================================

@dataclass
class TopologicalSignature:
    """
    Complete topological signature combining multiple TDA representations.

    This is the full interdimensional fingerprint of a point cloud,
    capturing topology at multiple scales and dimensions.
    """
    # Core persistence data
    diagrams: Dict[int, np.ndarray]  # H_0, H_1, H_2, ... diagrams

    # Functional representations
    landscapes: Dict[int, PersistenceLandscape]
    silhouettes: Dict[int, PersistenceSilhouette]
    betti_curves: Dict[int, BettiCurve]

    # Image representation
    images: Dict[int, PersistenceImage]

    # Summary statistics
    betti_numbers: Dict[int, int]
    persistence_entropy: Dict[int, float]
    total_persistence: Dict[int, float]

    # Metadata
    max_dimension: int
    num_points: int

    def to_vector(self, include_landscapes: bool = True,
                  include_silhouettes: bool = True,
                  include_images: bool = False) -> np.ndarray:
        """
        Convert signature to fixed-size feature vector.

        This enables direct use in ML pipelines.
        """
        features = []

        # Basic statistics
        for dim in range(self.max_dimension + 1):
            features.extend([
                self.betti_numbers.get(dim, 0),
                self.persistence_entropy.get(dim, 0.0),
                self.total_persistence.get(dim, 0.0)
            ])

        # Functional features
        if include_silhouettes:
            for dim in range(self.max_dimension + 1):
                if dim in self.silhouettes:
                    features.extend(self.silhouettes[dim].to_vector())

        if include_landscapes:
            for dim in range(self.max_dimension + 1):
                if dim in self.landscapes:
                    features.extend(self.landscapes[dim].to_vector()[:100])  # Cap size

        if include_images:
            for dim in range(self.max_dimension + 1):
                if dim in self.images:
                    features.extend(self.images[dim].to_vector())

        return np.array(features)

@dataclass
class _cI1OAO2:
    h0: np.ndarray
    h1: np.ndarray
    h2: Optional[np.ndarray] = None

    @property
    def _f0IOAO3(self) -> int:
        return self._count_persistent(self.h0)

    @property
    def _fllIAO4(self) -> int:
        return self._count_persistent(self.h1)

    def distance_to(self, _flO0AO6: np.ndarray, _flOOAO7: float=0.1) -> int:
        if len(_flO0AO6) == 0:
            return 0
        lifetimes = _flO0AO6[:, 1] - _flO0AO6[:, 0]
        lifetimes = np.where(np.isinf(lifetimes), 1.0, lifetimes)
        return int(np.sum(lifetimes > _flOOAO7))

    def _f0OOAO8(self, _f1IlAO9: int=1) -> float:
        _flO0AO6 = self.h1 if _f1IlAO9 == 1 else self.h0
        if len(_flO0AO6) == 0:
            return 0.0
        lifetimes = _flO0AO6[:, 1] - _flO0AO6[:, 0]
        lifetimes = np.where(np.isinf(lifetimes), 0, lifetimes)
        lifetimes = lifetimes[lifetimes > 0]
        if len(lifetimes) == 0:
            return 0.0
        total = np.sum(lifetimes)
        probs = lifetimes / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)

    def _fO0lAOA(self) -> np.ndarray:
        features = [self._f0IOAO3, self._fllIAO4, self._f0OOAO8(0), self._f0OOAO8(1), self._get_max_lifetime(self.h0), self._get_max_lifetime(self.h1), self._get_mean_lifetime(self.h0), self._get_mean_lifetime(self.h1), len(self.h0), len(self.h1)]
        return np.array(features)

    def _flOIAOB(self, _flO0AO6: np.ndarray) -> float:
        if len(_flO0AO6) == 0:
            return 0.0
        lifetimes = _flO0AO6[:, 1] - _flO0AO6[:, 0]
        lifetimes = np.where(np.isinf(lifetimes), 0, lifetimes)
        return float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0

    def _f1lOAOc(self, _flO0AO6: np.ndarray) -> float:
        if len(_flO0AO6) == 0:
            return 0.0
        lifetimes = _flO0AO6[:, 1] - _flO0AO6[:, 0]
        lifetimes = np.where(np.isinf(lifetimes), 0, lifetimes)
        return float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0

class _clOOAOd:

    def __init__(self, _fO10AOE: int=1, _f1IOAOf: float=float('inf'), _fOl0AlO: int=1):
        self._fO10AOE = _fO10AOE
        self._f1IOAOf = _f1IOAOf
        self._fOl0AlO = _fOl0AlO
        # Internal cache for expensive computations
        self._landscape_cache: Dict[str, PersistenceLandscape] = {}
        self._silhouette_cache: Dict[str, PersistenceSilhouette] = {}
        self._image_cache: Dict[str, PersistenceImage] = {}
        self._cache_max_size = 128
        if not HAS_RIPSER:
            print('Warning: ripser not installed. Using simplified TDA fallback.')

    def _cache_key(self, diagram: np.ndarray, *args) -> str:
        """Generate a hashable cache key from diagram and parameters."""
        diagram_bytes = diagram.tobytes() if len(diagram) > 0 else b''
        args_str = str(args)
        import hashlib
        return hashlib.md5(diagram_bytes + args_str.encode()).hexdigest()

    def _cache_trim(self, cache: dict) -> None:
        """Trim cache to max size using simple FIFO."""
        while len(cache) > self._cache_max_size:
            cache.pop(next(iter(cache)))

    def _fOl0All(self, _fO10Al2: np.ndarray) -> _cI1OAO2:
        if len(_fO10Al2) < 3:
            return _cI1OAO2(h0=np.array([[0, float('inf')]]), h1=np.array([]).reshape(0, 2))
        if HAS_RIPSER:
            return self._compute_with_ripser(_fO10Al2)
        else:
            return self._compute_fallback(_fO10Al2)

    def _f1l1Al3(self, _fO10Al2: np.ndarray) -> _cI1OAO2:
        result = ripser.ripser(_fO10Al2, maxdim=self._fO10AOE, thresh=self._f1IOAOf)
        diagrams = result['dgms']
        h0 = diagrams[0] if len(diagrams) > 0 else np.array([]).reshape(0, 2)
        h1 = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
        h2 = diagrams[2] if len(diagrams) > 2 else None
        return _cI1OAO2(h0=h0, h1=h1, h2=h2)

    def _f01lAl4(self, _fO10Al2: np.ndarray) -> _cI1OAO2:
        n_points = len(_fO10Al2)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(_fO10Al2))
        h0_features = []
        sorted_edges = np.sort(distances[np.triu_indices(n_points, k=1)])
        for i, edge_len in enumerate(sorted_edges[:min(10, len(sorted_edges))]):
            h0_features.append([0, edge_len])
        if not h0_features:
            h0_features.append([0, float('inf')])
        h1_features = []
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        for scale in [0.5, 1.0, 1.5, 2.0]:
            _flOOAO7 = mean_dist + scale * std_dist
            n_edges = np.sum(distances < _flOOAO7) // 2
            if n_edges > n_points:
                birth = mean_dist * (scale - 0.5)
                death = mean_dist * scale
                h1_features.append([birth, death])
        if not h1_features:
            h1_features = [[0, 0]]
        return _cI1OAO2(h0=np.array(h0_features), h1=np.array(h1_features))

    def _fIIOAl5(self, _fO1OAl6: _cI1OAO2, _f00lAl7: _cI1OAO2, _fI11Al8: int=1) -> float:
        d1 = _fO1OAl6.h1 if _fI11Al8 == 1 else _fO1OAl6.h0
        d2 = _f00lAl7.h1 if _fI11Al8 == 1 else _f00lAl7.h0
        if HAS_PERSIM:
            return persim.bottleneck(d1, d2)
        else:
            return self._bottleneck_fallback(d1, d2)

    def _fI00Al9(self, _fO1OAl6: np.ndarray, _f00lAl7: np.ndarray) -> float:
        if len(_fO1OAl6) == 0 and len(_f00lAl7) == 0:
            return 0.0
        f1 = self._diagram_to_simple_features(_fO1OAl6)
        f2 = self._diagram_to_simple_features(_f00lAl7)
        return float(np.linalg.norm(f1 - f2))

    def _fIOlAlA(self, _flO0AO6: np.ndarray) -> np.ndarray:
        if len(_flO0AO6) == 0:
            return np.zeros(4)
        lifetimes = _flO0AO6[:, 1] - _flO0AO6[:, 0]
        lifetimes = np.where(np.isinf(lifetimes), 0, lifetimes)
        return np.array([len(_flO0AO6), np.mean(lifetimes), np.max(lifetimes) if len(lifetimes) > 0 else 0, np.std(lifetimes)])

    def _fO10AlB(self, _fO1OAl6: _cI1OAO2, _f00lAl7: _cI1OAO2, _fI11Al8: int=1, _fl11Alc: int=2) -> float:
        d1 = _fO1OAl6.h1 if _fI11Al8 == 1 else _fO1OAl6.h0
        d2 = _f00lAl7.h1 if _fI11Al8 == 1 else _f00lAl7.h0
        if HAS_PERSIM:
            return persim.wasserstein(d1, d2, matching=False)
        else:
            return self._fI00Al9(d1, d2)

    def _f0llAld(self, _fO10Al2: np.ndarray) -> Dict[str, float]:
        _flO0AO6 = self._fOl0All(_fO10Al2)
        feature_vec = _flO0AO6._fO0lAOA()
        return {'betti_0': feature_vec[0], 'betti_1': feature_vec[1], 'entropy_h0': feature_vec[2], 'entropy_h1': feature_vec[3], 'max_lifetime_h0': feature_vec[4], 'max_lifetime_h1': feature_vec[5], 'mean_lifetime_h0': feature_vec[6], 'mean_lifetime_h1': feature_vec[7], 'n_features_h0': feature_vec[8], 'n_features_h1': feature_vec[9]}

    # =========================================================================
    # Extended TDA Methods: Landscapes, Silhouettes, Images
    # =========================================================================

    def compute_landscape(self, diagram: np.ndarray, k: int = 5,
                          resolution: int = 100, t_min: float = None,
                          t_max: float = None) -> PersistenceLandscape:
        """
        Compute persistence landscape from a persistence diagram.

        Args:
            diagram: Nx2 array of (birth, death) pairs
            k: Number of landscape functions to compute
            resolution: Number of sample points
            t_min: Minimum filtration value (auto if None)
            t_max: Maximum filtration value (auto if None)

        Returns:
            PersistenceLandscape object
        """
        # Check cache first
        cache_key = self._cache_key(diagram, k, resolution, t_min, t_max)
        if cache_key in self._landscape_cache:
            return self._landscape_cache[cache_key]

        if len(diagram) == 0:
            t_values = np.linspace(0, 1, resolution)
            return PersistenceLandscape(
                landscapes=np.zeros((k, resolution)),
                t_values=t_values,
                homology_dim=0,
                birth_death_pairs=diagram
            )

        # Filter infinite values
        finite_mask = np.isfinite(diagram[:, 1])
        finite_diagram = diagram[finite_mask]

        if len(finite_diagram) == 0:
            t_values = np.linspace(0, 1, resolution)
            return PersistenceLandscape(
                landscapes=np.zeros((k, resolution)),
                t_values=t_values,
                homology_dim=0,
                birth_death_pairs=diagram
            )

        # Determine range
        if t_min is None:
            t_min = np.min(finite_diagram[:, 0])
        if t_max is None:
            t_max = np.max(finite_diagram[:, 1])

        t_values = np.linspace(t_min, t_max, resolution)
        landscapes = np.zeros((k, resolution))

        # Vectorized computation of tent functions
        # births/deaths: (n_pairs,), t_values: (resolution,)
        births = finite_diagram[:, 0]  # (n_pairs,)
        deaths = finite_diagram[:, 1]  # (n_pairs,)

        # Broadcasting: t_grid (resolution, 1) vs births/deaths (n_pairs,)
        t_grid = t_values[:, np.newaxis]  # (resolution, 1)

        # Compute valid mask where birth <= t <= death: (resolution, n_pairs)
        valid_mask = (births <= t_grid) & (t_grid <= deaths)

        # Compute tent values: min(t - birth, death - t) where valid
        tent_left = t_grid - births    # (resolution, n_pairs)
        tent_right = deaths - t_grid   # (resolution, n_pairs)
        tent_values = np.minimum(tent_left, tent_right)
        tent_values = np.where(valid_mask, tent_values, 0.0)

        # Sort each row descending and take top k landscapes
        # np.partition is O(n) vs O(n log n) for full sort
        n_pairs = tent_values.shape[1]
        for j in range(min(k, n_pairs)):
            if n_pairs > j:
                # Get the (j+1)-th largest value in each row
                idx = n_pairs - j - 1
                partitioned = np.partition(tent_values, idx, axis=1)
                landscapes[j, :] = partitioned[:, idx]

        result = PersistenceLandscape(
            landscapes=landscapes,
            t_values=t_values,
            homology_dim=0,  # Caller should set correctly
            birth_death_pairs=diagram
        )
        # Store in cache
        self._landscape_cache[cache_key] = result
        self._cache_trim(self._landscape_cache)
        return result

    def compute_silhouette(self, diagram: np.ndarray, power: float = 1.0,
                           resolution: int = 100, t_min: float = None,
                           t_max: float = None) -> PersistenceSilhouette:
        """
        Compute persistence silhouette from a persistence diagram.

        The silhouette is a weighted average of tent functions.

        Args:
            diagram: Nx2 array of (birth, death) pairs
            power: Weighting power (1=arithmetic mean, 2=RMS)
            resolution: Number of sample points
            t_min, t_max: Filtration range

        Returns:
            PersistenceSilhouette object
        """
        # Check cache first
        cache_key = self._cache_key(diagram, power, resolution, t_min, t_max)
        if cache_key in self._silhouette_cache:
            return self._silhouette_cache[cache_key]

        if len(diagram) == 0:
            t_values = np.linspace(0, 1, resolution)
            return PersistenceSilhouette(
                silhouette=np.zeros(resolution),
                t_values=t_values,
                power=power,
                homology_dim=0
            )

        # Filter infinite values
        finite_mask = np.isfinite(diagram[:, 1])
        finite_diagram = diagram[finite_mask]

        if len(finite_diagram) == 0:
            t_values = np.linspace(0, 1, resolution)
            return PersistenceSilhouette(
                silhouette=np.zeros(resolution),
                t_values=t_values,
                power=power,
                homology_dim=0
            )

        # Determine range
        if t_min is None:
            t_min = np.min(finite_diagram[:, 0])
        if t_max is None:
            t_max = np.max(finite_diagram[:, 1])

        t_values = np.linspace(t_min, t_max, resolution)
        silhouette = np.zeros(resolution)

        # Weights based on persistence
        weights = finite_diagram[:, 1] - finite_diagram[:, 0]
        total_weight = np.sum(weights) + 1e-10

        # Vectorized computation
        births = finite_diagram[:, 0]  # (n_pairs,)
        deaths = finite_diagram[:, 1]  # (n_pairs,)

        # Broadcasting: t_grid (resolution, 1) vs births/deaths (n_pairs,)
        t_grid = t_values[:, np.newaxis]  # (resolution, 1)

        # Valid mask: (resolution, n_pairs)
        valid_mask = (births <= t_grid) & (t_grid <= deaths)

        # Tent values: (resolution, n_pairs)
        tent_left = t_grid - births
        tent_right = deaths - t_grid
        tent_values = np.minimum(tent_left, tent_right)
        tent_values = np.where(valid_mask, tent_values, 0.0)

        # Weighted sum: weights (n_pairs,) broadcast with tent_values (resolution, n_pairs)
        if power != 0:
            weighted_tent = weights * (tent_values ** power)  # (resolution, n_pairs)
            weighted_sum = np.sum(weighted_tent, axis=1)      # (resolution,)
            silhouette = (weighted_sum / total_weight) ** (1/power)

        result = PersistenceSilhouette(
            silhouette=silhouette,
            t_values=t_values,
            power=power,
            homology_dim=0,
            weights=weights
        )
        # Store in cache
        self._silhouette_cache[cache_key] = result
        self._cache_trim(self._silhouette_cache)
        return result

    def compute_persistence_image(self, diagram: np.ndarray,
                                   resolution: int = 50,
                                   sigma: float = None,
                                   birth_range: Tuple[float, float] = None,
                                   persistence_range: Tuple[float, float] = None,
                                   weight_fn: Callable[[float, float], float] = None
                                   ) -> PersistenceImage:
        """
        Compute persistence image from a persistence diagram.

        Args:
            diagram: Nx2 array of (birth, death) pairs
            resolution: Image resolution (resolution x resolution)
            sigma: Gaussian kernel bandwidth (auto if None)
            birth_range: (min, max) for birth axis
            persistence_range: (min, max) for persistence axis
            weight_fn: Custom weighting function(birth, persistence) -> weight

        Returns:
            PersistenceImage object
        """
        # Check cache first (only if no custom weight_fn)
        use_cache = weight_fn is None
        if use_cache:
            cache_key = self._cache_key(diagram, resolution, sigma, birth_range, persistence_range)
            if cache_key in self._image_cache:
                return self._image_cache[cache_key]

        if len(diagram) == 0:
            return PersistenceImage(
                image=np.zeros((resolution, resolution)),
                birth_range=(0, 1),
                persistence_range=(0, 1),
                sigma=0.1,
                homology_dim=0,
                resolution=resolution
            )

        # Filter infinite values and compute persistence
        finite_mask = np.isfinite(diagram[:, 1])
        finite_diagram = diagram[finite_mask]

        if len(finite_diagram) == 0:
            return PersistenceImage(
                image=np.zeros((resolution, resolution)),
                birth_range=(0, 1),
                persistence_range=(0, 1),
                sigma=0.1,
                homology_dim=0,
                resolution=resolution
            )

        births = finite_diagram[:, 0]
        deaths = finite_diagram[:, 1]
        persistence = deaths - births

        # Determine ranges
        if birth_range is None:
            birth_range = (float(np.min(births)), float(np.max(births)))
        if persistence_range is None:
            persistence_range = (0.0, float(np.max(persistence)))

        # Auto sigma based on grid spacing (ensure minimum to prevent div-by-zero)
        if sigma is None:
            birth_span = birth_range[1] - birth_range[0]
            pers_span = persistence_range[1] - persistence_range[0]
            sigma = max(min(birth_span, pers_span) / resolution, 1e-6)
        sigma = max(sigma, 1e-10)  # Ensure sigma is never zero

        # Default weight function: linear in persistence
        if weight_fn is None:
            def weight_fn(b, p):
                return p

        # Create image grid
        birth_grid = np.linspace(birth_range[0], birth_range[1], resolution)
        pers_grid = np.linspace(persistence_range[0], persistence_range[1], resolution)
        image = np.zeros((resolution, resolution))

        # Filter points with positive persistence
        valid_mask = persistence > 0
        valid_births = births[valid_mask]
        valid_pers = persistence[valid_mask]

        if len(valid_births) == 0:
            return PersistenceImage(
                image=image,
                birth_range=birth_range,
                persistence_range=persistence_range,
                sigma=sigma,
                homology_dim=0,
                resolution=resolution
            )

        # Vectorized Gaussian computation
        # Create meshgrid: BG (resolution, resolution), PG (resolution, resolution)
        BG, PG = np.meshgrid(birth_grid, pers_grid)

        # Compute weights for all valid points
        if weight_fn is None:
            weights = valid_pers  # Default: linear in persistence
        else:
            weights = np.array([weight_fn(b, p) for b, p in zip(valid_births, valid_pers)])

        # Compute Gaussian for each point using broadcasting
        # valid_births (n,), valid_pers (n,) -> reshape for broadcasting
        # BG, PG are (resolution, resolution)
        # We compute for each point and sum
        sigma_sq_2 = 2 * sigma ** 2
        for idx in range(len(valid_births)):
            b, p, w = valid_births[idx], valid_pers[idx], weights[idx]
            dist_sq = (b - BG) ** 2 + (p - PG) ** 2
            image += w * np.exp(-dist_sq / sigma_sq_2)

        result = PersistenceImage(
            image=image,
            birth_range=birth_range,
            persistence_range=persistence_range,
            sigma=sigma,
            homology_dim=0,
            resolution=resolution
        )
        # Store in cache if applicable
        if use_cache:
            self._image_cache[cache_key] = result
            self._cache_trim(self._image_cache)
        return result

    def compute_betti_curve(self, diagram: np.ndarray,
                            resolution: int = 100,
                            t_min: float = None,
                            t_max: float = None) -> BettiCurve:
        """
        Compute Betti curve (Betti number as function of filtration).

        Args:
            diagram: Nx2 array of (birth, death) pairs
            resolution: Number of sample points
            t_min, t_max: Filtration range

        Returns:
            BettiCurve object
        """
        if len(diagram) == 0:
            t_values = np.linspace(0, 1, resolution)
            return BettiCurve(
                curve=np.zeros(resolution),
                t_values=t_values,
                homology_dim=0
            )

        # Handle infinite deaths
        diagram_copy = diagram.copy()
        max_finite = np.max(diagram_copy[np.isfinite(diagram_copy[:, 1]), 1]) if np.any(np.isfinite(diagram_copy[:, 1])) else 1.0
        diagram_copy[~np.isfinite(diagram_copy[:, 1]), 1] = max_finite * 2

        # Determine range
        if t_min is None:
            t_min = np.min(diagram_copy[:, 0])
        if t_max is None:
            t_max = np.max(diagram_copy[:, 1])

        t_values = np.linspace(t_min, t_max, resolution)
        curve = np.zeros(resolution)

        births = diagram_copy[:, 0]
        deaths = diagram_copy[:, 1]

        for i, t in enumerate(t_values):
            # Count features alive at time t
            alive = np.sum((births <= t) & (deaths > t))
            curve[i] = alive

        return BettiCurve(
            curve=curve,
            t_values=t_values,
            homology_dim=0
        )

    def compute_full_signature(self, point_cloud: np.ndarray,
                                max_dim: int = 2,
                                landscape_k: int = 5,
                                resolution: int = 50) -> TopologicalSignature:
        """
        Compute complete topological signature of a point cloud.

        This extracts ALL topological features at ALL dimensions,
        creating a comprehensive interdimensional fingerprint.

        Args:
            point_cloud: Nxd array of points
            max_dim: Maximum homology dimension to compute
            landscape_k: Number of landscape functions per dimension
            resolution: Resolution for discretized representations

        Returns:
            TopologicalSignature with complete TDA analysis
        """
        # Compute persistence with higher dimensions
        old_maxdim = self._fO10AOE
        self._fO10AOE = max_dim

        if HAS_RIPSER:
            result = ripser.ripser(point_cloud, maxdim=max_dim, thresh=self._f1IOAOf)
            dgms = result['dgms']
        else:
            # Fallback computes only H0 and H1
            basic = self._f01lAl4(point_cloud)
            dgms = [basic.h0, basic.h1]

        self._fO10AOE = old_maxdim

        # Initialize containers
        diagrams = {}
        landscapes = {}
        silhouettes = {}
        betti_curves = {}
        images = {}
        betti_numbers = {}
        persistence_entropy = {}
        total_persistence = {}

        for dim in range(min(len(dgms), max_dim + 1)):
            dgm = dgms[dim]
            diagrams[dim] = dgm

            # Compute all representations
            landscape = self.compute_landscape(dgm, k=landscape_k, resolution=resolution)
            landscape.homology_dim = dim
            landscapes[dim] = landscape

            silhouette = self.compute_silhouette(dgm, resolution=resolution)
            silhouette.homology_dim = dim
            silhouettes[dim] = silhouette

            betti_curve = self.compute_betti_curve(dgm, resolution=resolution)
            betti_curve.homology_dim = dim
            betti_curves[dim] = betti_curve

            image = self.compute_persistence_image(dgm, resolution=resolution)
            image.homology_dim = dim
            images[dim] = image

            # Summary statistics
            finite_dgm = dgm[np.isfinite(dgm[:, 1])]
            if len(finite_dgm) > 0:
                lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                betti_numbers[dim] = int(np.sum(lifetimes > 0.01))  # Threshold
                total_persistence[dim] = float(np.sum(lifetimes))

                # Entropy
                lifetimes = lifetimes[lifetimes > 0]
                if len(lifetimes) > 0:
                    total = np.sum(lifetimes)
                    probs = lifetimes / total
                    persistence_entropy[dim] = float(-np.sum(probs * np.log(probs + 1e-10)))
                else:
                    persistence_entropy[dim] = 0.0
            else:
                betti_numbers[dim] = 0
                total_persistence[dim] = 0.0
                persistence_entropy[dim] = 0.0

        return TopologicalSignature(
            diagrams=diagrams,
            landscapes=landscapes,
            silhouettes=silhouettes,
            betti_curves=betti_curves,
            images=images,
            betti_numbers=betti_numbers,
            persistence_entropy=persistence_entropy,
            total_persistence=total_persistence,
            max_dimension=max_dim,
            num_points=len(point_cloud)
        )

    def landscape_distance(self, L1: PersistenceLandscape,
                           L2: PersistenceLandscape, p: float = 2) -> float:
        """
        Compute L^p distance between persistence landscapes.

        This is a stable metric on the space of persistence diagrams.
        """
        if L1.landscapes.shape != L2.landscapes.shape:
            # Interpolate to common resolution
            resolution = max(L1.resolution, L2.resolution)
            t_min = min(L1.t_values[0], L2.t_values[0])
            t_max = max(L1.t_values[-1], L2.t_values[-1])
            # For simplicity, just compare common part
            min_res = min(L1.resolution, L2.resolution)
            diff = L1.landscapes[:, :min_res] - L2.landscapes[:, :min_res]
        else:
            diff = L1.landscapes - L2.landscapes

        dt = L1.t_values[1] - L1.t_values[0] if len(L1.t_values) > 1 else 1.0
        if p == float('inf'):
            return float(np.max(np.abs(diff)))
        return float(np.sum(np.abs(diff) ** p * dt) ** (1/p))

    def silhouette_distance(self, S1: PersistenceSilhouette,
                            S2: PersistenceSilhouette) -> float:
        """Compute L^2 distance between silhouettes."""
        return PersistenceSilhouette.distance(S1, S2, p=2)

    def image_distance(self, I1: PersistenceImage, I2: PersistenceImage) -> float:
        """Compute Frobenius distance between persistence images."""
        return PersistenceImage.distance(I1, I2)

    # =========================================================================
    # Streaming/Incremental TDA Computation
    # =========================================================================

    def create_streaming_state(self, window_size: int = 100,
                                max_dim: int = 1) -> 'StreamingTDAState':
        """
        Create a streaming TDA state for incremental computation.

        Args:
            window_size: Number of points to maintain in sliding window
            max_dim: Maximum homology dimension to track

        Returns:
            StreamingTDAState object for incremental updates
        """
        return StreamingTDAState(
            pipeline=self,
            window_size=window_size,
            max_dim=max_dim
        )

    def compute_windowed_signature(self, points: np.ndarray,
                                    window_size: int = 50,
                                    stride: int = 10,
                                    max_dim: int = 1) -> List[TopologicalSignature]:
        """
        Compute topological signatures over sliding windows.

        This is useful for detecting topological changes over time.

        Args:
            points: Nxd array of points (typically time series embedded)
            window_size: Size of sliding window
            stride: Step size between windows
            max_dim: Maximum homology dimension

        Returns:
            List of TopologicalSignature objects, one per window
        """
        signatures = []
        for i in range(0, len(points) - window_size + 1, stride):
            window = points[i:i + window_size]
            sig = self.compute_full_signature(window, max_dim=max_dim,
                                               landscape_k=3, resolution=30)
            signatures.append(sig)
        return signatures

    def detect_topological_change(self, sig1: TopologicalSignature,
                                   sig2: TopologicalSignature,
                                   threshold: float = 0.5) -> Dict[str, any]:
        """
        Detect topological changes between two signatures.

        This identifies when the topological structure has changed significantly,
        potentially indicating a regime change.

        Args:
            sig1: First topological signature
            sig2: Second topological signature
            threshold: Change detection threshold

        Returns:
            Dictionary with change metrics and detection flags
        """
        changes = {
            'betti_change': {},
            'landscape_distance': {},
            'silhouette_distance': {},
            'detected_change': False,
            'change_magnitude': 0.0
        }

        total_change = 0.0
        num_dims = 0

        for dim in range(min(sig1.max_dimension, sig2.max_dimension) + 1):
            # Betti number change
            b1 = sig1.betti_numbers.get(dim, 0)
            b2 = sig2.betti_numbers.get(dim, 0)
            changes['betti_change'][dim] = b2 - b1

            # Landscape distance
            if dim in sig1.landscapes and dim in sig2.landscapes:
                L_dist = self.landscape_distance(sig1.landscapes[dim],
                                                  sig2.landscapes[dim])
                changes['landscape_distance'][dim] = L_dist
                total_change += L_dist
                num_dims += 1

            # Silhouette distance
            if dim in sig1.silhouettes and dim in sig2.silhouettes:
                S_dist = self.silhouette_distance(sig1.silhouettes[dim],
                                                   sig2.silhouettes[dim])
                changes['silhouette_distance'][dim] = S_dist

        if num_dims > 0:
            changes['change_magnitude'] = total_change / num_dims
            changes['detected_change'] = changes['change_magnitude'] > threshold

        return changes


# =============================================================================
# Streaming TDA State - For Real-Time Processing
# =============================================================================

@dataclass
class StreamingTDAState:
    """
    Maintains state for incremental TDA computation on streaming data.

    This enables real-time topological analysis by:
    1. Maintaining a sliding window of recent points
    2. Caching persistence computations
    3. Detecting topological changes incrementally
    """
    pipeline: '_clOOAOd'
    window_size: int
    max_dim: int
    points: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    _current_signature: Optional[TopologicalSignature] = None
    _previous_signature: Optional[TopologicalSignature] = None
    _update_count: int = 0

    def add_point(self, point: np.ndarray) -> Optional[Dict[str, any]]:
        """
        Add a new point and update topological state.

        Returns change detection results if a change was detected.
        """
        point = np.atleast_1d(point)

        # Initialize or append
        if self.points.size == 0:
            self.points = point.reshape(1, -1)
        else:
            self.points = np.vstack([self.points, point])

        # Maintain window size
        if len(self.points) > self.window_size:
            self.points = self.points[-self.window_size:]

        self._update_count += 1

        # Only recompute every few updates for efficiency
        if self._update_count % 5 == 0 and len(self.points) >= 10:
            return self._update_signature()

        return None

    def add_points(self, points: np.ndarray) -> Optional[Dict[str, any]]:
        """Add multiple points at once."""
        points = np.atleast_2d(points)

        if self.points.size == 0:
            self.points = points
        else:
            self.points = np.vstack([self.points, points])

        # Maintain window size
        if len(self.points) > self.window_size:
            self.points = self.points[-self.window_size:]

        self._update_count += len(points)

        if len(self.points) >= 10:
            return self._update_signature()

        return None

    def _update_signature(self) -> Optional[Dict[str, any]]:
        """Recompute signature and check for changes."""
        self._previous_signature = self._current_signature
        self._current_signature = self.pipeline.compute_full_signature(
            self.points,
            max_dim=self.max_dim,
            landscape_k=3,
            resolution=30
        )

        # Check for topological change
        if self._previous_signature is not None:
            changes = self.pipeline.detect_topological_change(
                self._previous_signature,
                self._current_signature,
                threshold=0.3
            )
            if changes['detected_change']:
                return changes

        return None

    @property
    def current_signature(self) -> Optional[TopologicalSignature]:
        """Get current topological signature."""
        return self._current_signature

    @property
    def betti_numbers(self) -> Dict[int, int]:
        """Get current Betti numbers."""
        if self._current_signature:
            return self._current_signature.betti_numbers
        return {}

    def get_feature_vector(self) -> np.ndarray:
        """Get current feature vector for ML."""
        if self._current_signature:
            return self._current_signature.to_vector()
        return np.array([])

    def reset(self):
        """Reset streaming state."""
        self.points = np.array([]).reshape(0, 0)
        self._current_signature = None
        self._previous_signature = None
        self._update_count = 0


# =============================================================================
# Method Aliases for TDAPipeline Public API
# =============================================================================

_clOOAOd.compute_persistence = _clOOAOd._fOl0All
_clOOAOd._compute_with_ripser = _clOOAOd._f1l1Al3
_clOOAOd._compute_fallback = _clOOAOd._f01lAl4
_clOOAOd.bottleneck_distance = _clOOAOd._fIIOAl5
_clOOAOd._bottleneck_fallback = _clOOAOd._fI00Al9
_clOOAOd._diagram_to_simple_features = _clOOAOd._fIOlAlA
_clOOAOd.wasserstein_distance = _clOOAOd._fO10AlB
_clOOAOd.extract_features = _clOOAOd._f0llAld


# =============================================================================
# Public API Exports
# =============================================================================

# Core obfuscated class aliases
PersistenceDiagram = _cI1OAO2
TDAPipeline = _clOOAOd

# Additional method aliases on PersistenceDiagram
_cI1OAO2.betti_0 = property(lambda self: self._f0IOAO3)
_cI1OAO2.betti_1 = property(lambda self: self._fllIAO4)
_cI1OAO2._count_persistent = _cI1OAO2.distance_to
_cI1OAO2.persistence_entropy = _cI1OAO2._f0OOAO8
_cI1OAO2.to_feature_vector = _cI1OAO2._fO0lAOA
_cI1OAO2._get_max_lifetime = _cI1OAO2._flOIAOB
_cI1OAO2._get_mean_lifetime = _cI1OAO2._f1lOAOc

# Public exports list for module
__all__ = [
    # Core classes
    'PersistenceDiagram',
    'TDAPipeline',
    # Extended TDA representations
    'PersistenceLandscape',
    'PersistenceSilhouette',
    'PersistenceImage',
    'BettiCurve',
    'TopologicalSignature',
    # Streaming support
    'StreamingTDAState',
]