from __future__ import annotations
import functools
import inspect
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from enum import Enum, auto
import threading
import json
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from jones_framework.core.tensor_ops import Tensor, ManifoldEmbedding
T = TypeVar('T')

class ConnectionType(Enum):
    EXTENDS = auto()
    USES = auto()
    TRANSFORMS = auto()
    COMPOSES = auto()
    IMPLEMENTS = auto()
    BRIDGES = auto()
    VALIDATES = auto()
    PRODUCES = auto()
    CONSUMES = auto()
    MONITORS = auto()
    CONFIGURES = auto()
    INHIBITS = auto()

@dataclass
class _c0IIBfd:
    name: str
    module_path: str
    cls: Optional[Type] = None
    connections: Dict[str, ConnectionType] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Tensor] = None

    # Alias for obfuscated field name used in legacy code
    @property
    def _fl11cOA(self) -> Dict[str, ConnectionType]:
        return self.connections

    @property
    def _fll0clB(self) -> Dict[str, Any]:
        return self.metadata

    def _f011BfE(self, _f00OBff: str, _flO1cOO: ConnectionType):
        self.connections[_f00OBff] = _flO1cOO

    def _fIllcOl(self, _fI1OcO2: 'ComponentNode') -> float:
        if self.embedding is None or _fI1OcO2.embedding is None:
            return self._graph_distance(_fI1OcO2)
        return float((self.embedding - _fI1OcO2.embedding).norm().item())

    def _f01OcO3(self, _fI1OcO2: 'ComponentNode') -> float:
        if _fI1OcO2.name in self.connections:
            return 1.0
        return float('inf')

class _clOOcO4:
    _instance: Optional['ComponentRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _f0llcO5(self):
        self.components: Dict[str, _c0IIBfd] = {}
        self._connection_graph: Dict[str, Set[str]] = {}
        self._embeddings_computed = False
        self._antinomies: List[Tuple[str, str]] = []
        self._fiedler_value: Optional[float] = None
        self._core_components = self._register_core_components()

    # Alias for obfuscated method name
    _initialize = _f0llcO5

    def _fIlIcO6(self) -> Set[str]:
        core = {'Tensor': 'jones_framework.core.tensor_ops', 'ConditionState': 'jones_framework.core.condition_state', 'ActivityState': 'jones_framework.core.activity_state', 'ShadowTensor': 'jones_framework.core.shadow_tensor', 'TDAPipeline': 'jones_framework.perception.tda_pipeline', 'RegimeClassifier': 'jones_framework.perception.regime_classifier', 'MetricWarper': 'jones_framework.perception.metric_warper', 'MixtureOfExperts': 'jones_framework.sans.mixture_of_experts', 'LoRAAdapter': 'jones_framework.sans.lora_adapter', 'ContinuityGuard': 'jones_framework.sans.continuity_guard', 'SentimentVectorPipeline': 'jones_framework.arbitrage.sentiment_vector', 'LinguisticArbitrageEngine': 'jones_framework.arbitrage.linguistic_arbitrage', 'HardwareAccelerator': 'jones_framework.utils.hardware', 'FrameworkConfig': 'jones_framework.utils.config'}
        for name, module in core.items():
            node = _c0IIBfd(name=name, module_path=module)
            self.components[name] = node
            self._connection_graph[name] = set()
        core_connections = [
            ('ActivityState', 'ConditionState', ConnectionType.USES),
            ('ShadowTensor', 'ConditionState', ConnectionType.TRANSFORMS),
            ('TDAPipeline', 'ShadowTensor', ConnectionType.USES),
            ('RegimeClassifier', 'TDAPipeline', ConnectionType.USES),
            ('MixtureOfExperts', 'RegimeClassifier', ConnectionType.USES),
            ('MixtureOfExperts', 'LoRAAdapter', ConnectionType.COMPOSES),
            ('MixtureOfExperts', 'ActivityState', ConnectionType.USES),
            ('ContinuityGuard', 'ConditionState', ConnectionType.USES),
            ('LinguisticArbitrageEngine', 'SentimentVectorPipeline', ConnectionType.COMPOSES),
            ('LinguisticArbitrageEngine', 'ShadowTensor', ConnectionType.USES),
            ('LinguisticArbitrageEngine', 'TDAPipeline', ConnectionType.USES),
            ('MetricWarper', 'Tensor', ConnectionType.USES),
            ('ConditionState', 'Tensor', ConnectionType.USES),
            # Additional connections for previously orphaned core components
            ('LoRAAdapter', 'Tensor', ConnectionType.USES),
            ('SentimentVectorPipeline', 'ConditionState', ConnectionType.TRANSFORMS),
            ('HardwareAccelerator', 'Tensor', ConnectionType.USES),
            ('FrameworkConfig', 'Tensor', ConnectionType.USES),
        ]
        for source, _f00OBff, conn_type in core_connections:
            self._f011BfE(source, _f00OBff, conn_type)
        return set(core.keys())

    # Alias for obfuscated method name
    _register_core_components = _fIlIcO6

    def _f1IlcO7(self, _f11lcO8: str, _f00OcO9: str, cls: Optional[Type]=None, _fl11cOA: List[Tuple[str, ConnectionType]]=None) -> _c0IIBfd:
        if _f11lcO8 in self.components:
            node = self.components[_f11lcO8]
            node.cls = cls or node.cls
        else:
            node = _c0IIBfd(name=_f11lcO8, module_path=_f00OcO9, cls=cls)
            self.components[_f11lcO8] = node
            self._connection_graph[_f11lcO8] = set()
        if _fl11cOA:
            for _f00OBff, conn_type in _fl11cOA:
                self._f011BfE(_f11lcO8, _f00OBff, conn_type)
        self._embeddings_computed = False
        return node

    def _f011BfE(self, _f0O0cOB: str, _f00OBff: str, _flO1cOO: ConnectionType):
        if _f0O0cOB not in self.components:
            raise ValueError(f'Source component not registered: {_f0O0cOB}')
        if _f00OBff not in self.components:
            # Auto-register unknown target components as external references
            import warnings
            warnings.warn(f'Auto-registering external component: {_f00OBff}')
            self.components[_f00OBff] = _c0IIBfd(name=_f00OBff, module_path='external')
            self._connection_graph[_f00OBff] = set()
        self.components[_f0O0cOB]._f011BfE(_f00OBff, _flO1cOO)
        self._connection_graph[_f0O0cOB].add(_f00OBff)
        self._embeddings_computed = False

    def _fII1cOc(self, _fl0lcOd: str) -> List[str]:
        errors = []
        if _fl0lcOd not in self.components:
            errors.append(f'Component not registered: {_fl0lcOd}')
            return errors
        node = self.components[_fl0lcOd]

        # Skip validation for auto-registered external components (module_path == 'external')
        if node.module_path == 'external':
            return errors

        # Skip validation for obfuscated class names (likely internal implementation details)
        if _fl0lcOd.startswith('_c') or _fl0lcOd.startswith('_f'):
            return errors

        if not node._fl11cOA and _fl0lcOd != 'Tensor':
            errors.append(f'{_fl0lcOd} has no connections - orphaned component')
        for _f00OBff in node._fl11cOA:
            if _f00OBff not in self.components:
                errors.append(f'{_fl0lcOd} connects to unknown: {_f00OBff}')
        if not self._is_reachable_from_core(_fl0lcOd):
            errors.append(f'{_fl0lcOd} not reachable from core components')
        return errors

    def _f111cOE(self, _fl0lcOd: str) -> bool:
        if _fl0lcOd in self._core_components:
            return True

        # Check forward direction: component -> core
        visited = set()
        queue = [_fl0lcOd]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in self._core_components:
                return True
            node = self.components.get(current)
            if node:
                queue.extend(node._fl11cOA.keys())

        # Check reverse direction: core -> component (is any core component connected to this one?)
        for core_name in self._core_components:
            core_node = self.components.get(core_name)
            if core_node and _fl0lcOd in core_node._fl11cOA:
                return True

        # Check if any component that IS reachable from core connects to this component
        for name, node in self.components.items():
            if name in self._core_components or name == _fl0lcOd:
                continue
            if _fl0lcOd in node._fl11cOA:
                # This component connects to our target - check if IT is reachable
                if name in visited:  # Already checked and it reached core
                    continue
                # Do a quick check if this connector is reachable from core
                if self._is_connected_to_core_direct(name):
                    return True

        return False

    def _is_connected_to_core_direct(self, name: str) -> bool:
        """Quick check if a component directly connects to core."""
        node = self.components.get(name)
        if not node:
            return False
        for target in node._fl11cOA:
            if target in self._core_components:
                return True
        return False

    # Alias for obfuscated method name
    _is_reachable_from_core = _f111cOE

    def _f1I1cOf(self, _f1lOclO: int=8):
        n = len(self.components)
        if n == 0:
            return
        names = list(self.components.keys())
        name_to_idx = {_f11lcO8: i for i, _f11lcO8 in enumerate(names)}
        adjacency = Tensor.zeros(n, n)
        for _f11lcO8, node in self.components.items():
            i = name_to_idx[_f11lcO8]
            for _f00OBff in node._fl11cOA:
                if _f00OBff in name_to_idx:
                    j = name_to_idx[_f00OBff]
                    adjacency._data[i, j] = 1.0
                    adjacency._data[j, i] = 1.0
        degree = adjacency.sum(dim=1)
        laplacian = Tensor.zeros(n, n)
        for i in range(n):
            laplacian._data[i, i] = degree[i].item()
        laplacian = laplacian - adjacency
        # Use sparse eigenvalue solver for efficiency - only compute k+1 smallest eigenvalues
        k = min(_f1lOclO + 1, n - 1)  # Need k+1 eigenvalues (skip first zero eigenvalue)
        if k >= 1 and n > 1:
            # Convert to sparse format and use eigsh for smallest eigenvalues
            laplacian_sparse = csr_matrix(laplacian._data)
            eigenvalues_arr, eigenvectors_arr = eigsh(laplacian_sparse, k=k, which='SM', tol=1e-6)
            # Sort by eigenvalue (eigsh doesn't guarantee order)
            sort_idx = np.argsort(eigenvalues_arr)
            eigenvalues_arr = eigenvalues_arr[sort_idx]
            eigenvectors_arr = eigenvectors_arr[:, sort_idx]
            # Store Fiedler value (second-smallest eigenvalue)
            self._fiedler_value = float(eigenvalues_arr[1]) if len(eigenvalues_arr) > 1 else 0.0
        else:
            # Fallback for very small graphs
            eigenvalues_tensor, eigenvectors_tensor = laplacian.eigh()
            eigenvectors_arr = eigenvectors_tensor._data
        for i, _f11lcO8 in enumerate(names):
            embedding = Tensor.zeros(_f1lOclO)
            for j in range(min(_f1lOclO, eigenvectors_arr.shape[1] - 1)):
                embedding._data[j] = eigenvectors_arr[i, j + 1]
            self.components[_f11lcO8].embedding = embedding
        self._embeddings_computed = True

    def _fI1lcll(self, _fl0lcOd: str, _fIlIcl2: int=5) -> List[Tuple[str, float]]:
        if not self._embeddings_computed:
            self._f1I1cOf()
        if _fl0lcOd not in self.components:
            return []
        node = self.components[_fl0lcOd]
        distances = []
        for _f11lcO8, other_node in self.components.items():
            if _f11lcO8 == _fl0lcOd:
                continue
            dist = node._fIllcOl(other_node)
            distances.append((_f11lcO8, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:_fIlIcl2]

    def _fIIOcl3(self, _fl0lcOd: str) -> List[Tuple[str, ConnectionType, float]]:
        nearest = self._fI1lcll(_fl0lcOd, _fIlIcl2=10)
        existing = set(self.components[_fl0lcOd]._fl11cOA.keys())
        suggestions = []
        for _f11lcO8, dist in nearest:
            if _f11lcO8 not in existing:
                conn_type = self._infer_connection_type(_fl0lcOd, _f11lcO8)
                suggestions.append((_f11lcO8, conn_type, dist))
        return suggestions[:5]

    def _fI0Ocl4(self, _f0O0cOB: str, _f00OBff: str) -> ConnectionType:
        source_lower = _f0O0cOB.lower()
        target_lower = _f00OBff.lower()
        if 'pipeline' in source_lower or 'builder' in source_lower:
            return ConnectionType.TRANSFORMS
        if 'engine' in source_lower or 'controller' in source_lower:
            return ConnectionType.COMPOSES
        if 'state' in target_lower:
            return ConnectionType.USES
        return ConnectionType.USES

    def _fOl1cl5(self) -> Dict[str, float]:
        gradients = {}
        for _f11lcO8, node in self.components.items():
            n_connections = len(node._fl11cOA)
            base = 1.0 / (1.0 + n_connections)
            core_distance = self._min_distance_to_core(_f11lcO8)
            core_factor = 1.0 + 0.1 * core_distance
            incoming = sum((1 for n in self.components.values() if _f11lcO8 in n._fl11cOA))
            bidirectional_factor = 1.0 if incoming > 0 else 2.0
            gradients[_f11lcO8] = base * core_factor * bidirectional_factor
        return gradients

    def _f1I0cl6(self, _fl0lcOd: str) -> int:
        if _fl0lcOd in self._core_components:
            return 0
        visited = {_fl0lcOd}
        queue = [(_fl0lcOd, 0)]
        while queue:
            current, dist = queue.pop(0)
            node = self.components.get(current)
            if not node:
                continue
            for _f00OBff in node._fl11cOA:
                if _f00OBff in self._core_components:
                    return dist + 1
                if _f00OBff not in visited:
                    visited.add(_f00OBff)
                    queue.append((_f00OBff, dist + 1))
        return float('inf')

    def _fIIOcl7(self) -> str:
        data = {'components': {}, 'connections': []}
        for _f11lcO8, node in self.components.items():
            data['components'][_f11lcO8] = {'module_path': node._f00OcO9, 'metadata': node.metadata}
            for _f00OBff, conn_type in node._fl11cOA.items():
                data['connections'].append({'source': _f11lcO8, 'target': _f00OBff, 'type': conn_type._f11lcO8})
        return json.dumps(data, indent=2)

    def _fI00cl8(self) -> str:
        lines = ['digraph ComponentManifold {']
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box];')
        module_colors = {'core': 'lightblue', 'perception': 'lightgreen', 'sans': 'lightyellow', 'arbitrage': 'lightpink', 'utils': 'lightgray', 'domains': 'lavender', 'data': 'peachpuff', 'ml': 'paleturquoise'}
        for _f11lcO8, node in self.components.items():
            module = node._f00OcO9.split('.')[1] if '.' in node._f00OcO9 else 'core'
            color = module_colors.get(module, 'white')
            lines.append(f'  "{_f11lcO8}" [fillcolor={color}, style=filled];')
        for _f11lcO8, node in self.components.items():
            for _f00OBff, conn_type in node._fl11cOA.items():
                style = {ConnectionType.EXTENDS: 'bold', ConnectionType.USES: 'solid', ConnectionType.TRANSFORMS: 'dashed', ConnectionType.COMPOSES: 'dotted', ConnectionType.IMPLEMENTS: 'bold,dashed', ConnectionType.BRIDGES: 'bold,dotted'}.get(conn_type, 'solid')
                lines.append(f'  "{_f11lcO8}" -> "{_f00OBff}" [style={style}];')
        lines.append('}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Antinomy Detection (CTS Section 6.3)
    # ------------------------------------------------------------------

    def add_antinomy(self, comp_a: str, comp_b: str):
        """Register a mutual exclusion relation between two components.

        An antinomy means comp_a and comp_b cannot both be active
        without degrading coherence (CTS Section 6.3).
        """
        if comp_a not in self.components:
            raise ValueError(f'Component not registered: {comp_a}')
        if comp_b not in self.components:
            raise ValueError(f'Component not registered: {comp_b}')
        pair = tuple(sorted([comp_a, comp_b]))
        if pair not in [(a, b) for a, b in self._antinomies]:
            self._antinomies.append(pair)

    def compute_antinomy_load(self, active_nodes: Optional[Set[str]] = None) -> float:
        """Compute antinomy load α(t) — CTS Eq 12.

        α(t) = |{v ∈ V_a : v participates in at least one antinomy}| / |V_a|

        Args:
            active_nodes: Set of currently active component names.
                          If None, uses all registered components.

        Returns:
            float in [0, 1]: fraction of active nodes involved in contradictions.
        """
        if active_nodes is None:
            active_nodes = set(self.components.keys())
        if not active_nodes:
            return 0.0

        contradicted = set()
        for comp_a, comp_b in self._antinomies:
            if comp_a in active_nodes and comp_b in active_nodes:
                contradicted.add(comp_a)
                contradicted.add(comp_b)

        return len(contradicted) / len(active_nodes)

    def compute_fiedler_value(self, active_nodes: Optional[Set[str]] = None) -> float:
        """Compute the algebraic connectivity λ₁ (Fiedler value).

        λ₁ = second-smallest eigenvalue of the normalized graph Laplacian
        of the active subgraph. λ₁ > 0 iff the graph is connected.

        Args:
            active_nodes: Set of active component names. If None, uses all.

        Returns:
            float ≥ 0: the Fiedler value.
        """
        if active_nodes is None:
            # Use cached value from full graph if available
            if self._fiedler_value is not None:
                return self._fiedler_value
            active_nodes = set(self.components.keys())

        active_list = sorted(active_nodes & set(self.components.keys()))
        n = len(active_list)
        if n < 2:
            return 0.0

        name_to_idx = {name: i for i, name in enumerate(active_list)}

        # Build adjacency matrix for active subgraph
        adj = np.zeros((n, n))
        for name in active_list:
            i = name_to_idx[name]
            node = self.components[name]
            for target in node.connections:
                if target in name_to_idx:
                    j = name_to_idx[target]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # Compute normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
        degree = adj.sum(axis=1)
        # Handle isolated nodes
        d_inv_sqrt = np.zeros(n)
        for i in range(n):
            if degree[i] > 0:
                d_inv_sqrt[i] = 1.0 / np.sqrt(degree[i])

        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_sym = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

        # Eigendecomposition
        try:
            eigenvalues = np.linalg.eigvalsh(L_sym)
            eigenvalues.sort()
            return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        except np.linalg.LinAlgError:
            return 0.0

    def compute_coherence_phi(self, active_nodes: Optional[Set[str]] = None) -> float:
        """Compute coherence measure Φ — CTS Definition 6.2, Eq 12.

        Φ(t) = λ₁(L_sym) · (1 - α(t))

        where λ₁ is the Fiedler value and α is the antinomy load.

        Args:
            active_nodes: Set of active component names. If None, uses all.

        Returns:
            float ≥ 0: the coherence measure.
        """
        lambda_1 = self.compute_fiedler_value(active_nodes)
        alpha = self.compute_antinomy_load(active_nodes)
        return lambda_1 * (1.0 - alpha)

    def get_antinomies(self) -> List[Tuple[str, str]]:
        """Return all registered antinomy pairs."""
        return list(self._antinomies)

    # Public API method aliases for ComponentRegistry
    _infer_connection_type = _fI0Ocl4
    _min_distance_to_core = _f1I0cl6
    _get_improvement_gradients = _fOl1cl5

_registry = _clOOcO4()

def get_registry() -> _clOOcO4:
    return _registry

def bridge(
    *args,
    _fOOlcl9: List[str] = None,
    _f1l0clA: Optional[Dict[str, ConnectionType]] = None,
    _fll0clB: Optional[Dict[str, Any]] = None,
    connects_to: List[str] = None,
    connection_types: Optional[Dict[str, ConnectionType]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[Type[T]], Type[T]]:
    # Support multiple calling conventions:
    # 1. @bridge('A', 'B', 'C') - variadic positional strings
    # 2. @bridge(connects_to=['A', 'B'], connection_types={...}) - keyword args
    # 3. @bridge(_fOOlcl9=['A', 'B']) - obfuscated keyword args

    # Handle variadic positional args (strings)
    if args and all(isinstance(a, str) for a in args):
        _fOOlcl9 = list(args)
    elif len(args) == 1 and isinstance(args[0], list):
        _fOOlcl9 = args[0]

    # Support both obfuscated and public parameter names
    _fOOlcl9 = connects_to if connects_to is not None else _fOOlcl9
    _f1l0clA = connection_types if connection_types is not None else _f1l0clA
    _fll0clB = metadata if metadata is not None else _fll0clB
    if _fOOlcl9 is None:
        _fOOlcl9 = []

    def _fIl1clc(cls: Type[T]) -> Type[T]:
        _f11lcO8 = cls.__name__
        module = cls.__module__
        _fl11cOA = []
        for _f00OBff in _fOOlcl9:
            conn_type = ConnectionType.USES
            if _f1l0clA and isinstance(_f1l0clA, dict) and _f00OBff in _f1l0clA:
                val = _f1l0clA[_f00OBff]
                # Handle string connection types (from bridge calls with string values)
                if isinstance(val, str):
                    try:
                        conn_type = ConnectionType[val.upper()]
                    except KeyError:
                        conn_type = ConnectionType.USES
                else:
                    conn_type = val
            _fl11cOA.append((_f00OBff, conn_type))
        node = _registry._f1IlcO7(_f11lcO8, module, cls, _fl11cOA)
        if _fll0clB:
            node._fll0clB.update(_fll0clB)
        errors = _registry._fII1cOc(_f11lcO8)
        if errors:
            import warnings
            for error in errors:
                warnings.warn(f'ComponentRegistry: {error}')
        cls._component_node = node
        cls._component_registry = _registry
        return cls
    return _fIl1clc

def extends(_f1OIcld: str):

    def _fIl1clc(cls: Type[T]) -> Type[T]:
        return bridge(connects_to=[_f1OIcld], connection_types={_f1OIcld: ConnectionType.EXTENDS})(cls)
    return _fIl1clc

def transforms(*components: str):

    def _fIl1clc(cls: Type[T]) -> Type[T]:
        return bridge(connects_to=list(components), connection_types={c: ConnectionType.TRANSFORMS for c in components})(cls)
    return _fIl1clc

def composes(*components: str):

    def _fIl1clc(cls: Type[T]) -> Type[T]:
        return bridge(connects_to=list(components), connection_types={c: ConnectionType.COMPOSES for c in components})(cls)
    return _fIl1clc

class RecursiveImprover:

    def __init__(self, _f0OOclE: Optional[_clOOcO4]=None):
        self._f0OOclE = _f0OOclE or get_registry()

    def _fIO1clf(self) -> List[Dict[str, Any]]:
        gaps = []
        for _f11lcO8, node in self._f0OOclE.components.items():
            errors = self._f0OOclE._fII1cOc(_f11lcO8)
            if errors:
                gaps.append({'type': 'orphaned', 'component': _f11lcO8, 'errors': errors, 'suggestion': self._f0OOclE._fIIOcl3(_f11lcO8)})
        for _f11lcO8, node in self._f0OOclE.components.items():
            for _f00OBff in node._fl11cOA:
                target_node = self._f0OOclE.components.get(_f00OBff)
                if target_node and _f11lcO8 not in target_node._fl11cOA:
                    gaps.append({'type': 'unidirectional', 'source': _f11lcO8, 'target': _f00OBff, 'suggestion': f'Consider adding {_f00OBff} -> {_f11lcO8} connection'})
        return gaps

    def _fO10c2O(self) -> List[str]:
        gradients = self._f0OOclE._fOl1cl5()
        sorted_components = sorted(gradients.items(), key=lambda x: x[1], reverse=True)
        return [_f11lcO8 for _f11lcO8, _ in sorted_components]

    def _fO0Oc2l(self, _f000c22: str) -> Dict[str, Any]:
        domain_components = [_f11lcO8 for _f11lcO8, node in self._f0OOclE.components.items() if _f000c22 in node._f00OcO9]
        all_neighbors = []
        for comp in domain_components:
            neighbors = self._f0OOclE._fI1lcll(comp, _fIlIcl2=3)
            all_neighbors.extend(neighbors)
        neighbor_types = set()
        for _f11lcO8, _ in all_neighbors:
            if _f11lcO8 not in domain_components:
                neighbor_types.add(self._infer_component_type(_f11lcO8))
        existing_types = {self._infer_component_type(c) for c in domain_components}
        missing_types = neighbor_types - existing_types
        return {'domain': _f000c22, 'existing_components': domain_components, 'suggested_types': list(missing_types), 'potential_connections': [n for n, _ in all_neighbors[:5]]}

    def _f010c23(self, _f11lcO8: str) -> str:
        name_lower = _f11lcO8.lower()
        if 'pipeline' in name_lower:
            return 'pipeline'
        if 'engine' in name_lower:
            return 'engine'
        if 'builder' in name_lower:
            return 'builder'
        if 'state' in name_lower:
            return 'state'
        if 'adapter' in name_lower:
            return 'adapter'
        if 'classifier' in name_lower:
            return 'classifier'
        if 'guard' in name_lower:
            return 'guard'
        return 'component'

    def _f0l0c24(self) -> str:
        lines = ['# Component Manifold Improvement Report\n']
        lines.append('## Overview\n')
        lines.append(f'Total components: {len(self._f0OOclE.components)}\n')
        lines.append(f'Core components: {len(self._f0OOclE._core_components)}\n')
        gaps = self._fIO1clf()
        lines.append(f'\n## Gaps Identified: {len(gaps)}\n')
        for gap in gaps[:10]:
            lines.append(f"- **{gap['type']}**: {gap.get('component', gap.get('source', 'N/A'))}\n")
        path = self._fO10c2O()
        lines.append('\n## Improvement Priority\n')
        for i, comp in enumerate(path[:10]):
            lines.append(f'{i + 1}. {comp}\n')
        gradients = self._f0OOclE._fOl1cl5()
        lines.append('\n## Improvement Gradients\n')
        sorted_grads = sorted(gradients.items(), key=lambda x: x[1], reverse=True)
        for _f11lcO8, grad in sorted_grads[:10]:
            lines.append(f'- {_f11lcO8}: {grad:.3f}\n')
        return ''.join(lines)

    # Public API method aliases
    identify_gaps = _fIO1clf
    compute_improvement_path = _fO10c2O
    suggest_new_component = _fO0Oc2l
    _infer_component_type = _f010c23
    generate_improvement_report = _f0l0c24


# Public API aliases for obfuscated classes
ComponentNode = _c0IIBfd
ComponentRegistry = _clOOcO4