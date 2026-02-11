from jones_framework.core.condition_state import ConditionState
from jones_framework.core.activity_state import ActivityState, RegimeID
from jones_framework.core.shadow_tensor import ShadowTensorBuilder, ShadowTensor
from jones_framework.core.manifold_bridge import (
    bridge,
    ComponentRegistry,
    ComponentNode,
    ConnectionType,
    RecursiveImprover,
    get_registry,
    extends,
    transforms,
    composes,
)
from jones_framework.core.novelty_search import (
    LayerType,
    LayerOutput,
    LayerProcessor,
    StructuralLayerProcessor,
    LinguisticLayerProcessor,
    MathematicalLayerProcessor,
    NoveltyGradient,
    NoveltySearchResult,
    NoveltySearchLoop,
    create_novelty_loop,
)
from jones_framework.core.knowledge_flow import (
    Role,
    ViewDimension,
    KnowledgePacket,
    ViewPreference,
    ViewPreferenceModel,
    KnowledgeFlow,
    integrate_knowledge_flow,
)
from jones_framework.core.universal_tensor import (
    TensorComponent,
    TensorDecomposition,
    UniversalTensorSpace,
)
from jones_framework.core.schematism import (
    PatternSchema,
    SchematismResult,
    SchematismBridge,
)
from jones_framework.core.coherent_configuration import (
    CriticalityType,
    CoherentConfiguration,
    AgencyStep,
    ConfigurationBuilder,
    AgencyFlow,
)

__all__ = [
    # Core state
    'ConditionState', 'ActivityState', 'RegimeID',
    'ShadowTensorBuilder', 'ShadowTensor',

    # Manifold bridge
    'bridge', 'ComponentRegistry', 'ComponentNode', 'ConnectionType',
    'RecursiveImprover', 'get_registry', 'extends', 'transforms', 'composes',

    # Novelty search loop
    'LayerType', 'LayerOutput', 'LayerProcessor',
    'StructuralLayerProcessor', 'LinguisticLayerProcessor', 'MathematicalLayerProcessor',
    'NoveltyGradient', 'NoveltySearchResult', 'NoveltySearchLoop', 'create_novelty_loop',

    # Knowledge flow
    'Role', 'ViewDimension', 'KnowledgePacket', 'ViewPreference',
    'ViewPreferenceModel', 'KnowledgeFlow', 'integrate_knowledge_flow',

    # CTS: Universal Tensor Space (Section 3)
    'TensorComponent', 'TensorDecomposition', 'UniversalTensorSpace',

    # CTS: Schematism Bridge (Section 7)
    'PatternSchema', 'SchematismResult', 'SchematismBridge',

    # CTS: Coherent Configuration (Sections 8-9)
    'CriticalityType', 'CoherentConfiguration', 'AgencyStep',
    'ConfigurationBuilder', 'AgencyFlow',
]
