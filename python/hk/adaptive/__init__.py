"""
HK Adaptive Neural Framework: Root Exports
"""

from .appendix import (
    AppendixEntryType,
    AppendixFlags,
    AppendixMetrics,
    AppendixRecord,
    read_appendix,
    append_record,
    write_appendix_record,
    rollback_appendix,
    verify_lineage,
    compute_parent_hash
)

from .growth import (
    GrowthGovernor,
    net2wider_linear,
    net2wider_swiglu,
    net2deeper_linear,
    ModularResidualBlock,
    expand_vocab,
    expand_model_width,
    protect_base_capacity,
)

from .code_eval import (
    TestCase,
    EvalResult,
    CodeSandbox
)

from .self_play import (
    LoRAAdapter,
    SelfPlayEvolutionEngine
)

from .topology import (
    TopologyContainer,
    package_standalone_hk,
    load_standalone_topology,
    load_standalone_hk,
    build_model_from_topology
)

from .expansion_evaluator import (
    ExpansionDiagnosis,
    ExpansionEvaluator,
)

from .self_conversation import (
    ConversationalTurn,
    SelfDialogue,
    SelfConversationalEngine,
)

from .self_training import (
    SelfTrainingCurriculum,
    GenerationReport,
    SelfTrainingPipeline,
)

__all__ = [
    "AppendixEntryType",
    "AppendixFlags",
    "AppendixMetrics",
    "AppendixRecord",
    "read_appendix",
    "append_record",
    "write_appendix_record",
    "rollback_appendix",
    "verify_lineage",
    "compute_parent_hash",
    "GrowthGovernor",
    "net2wider_linear",
    "net2wider_swiglu",
    "net2deeper_linear",
    "ModularResidualBlock",
    "expand_vocab",
    "expand_model_width",
    "protect_base_capacity",
    "TestCase",
    "EvalResult",
    "CodeSandbox",
    "LoRAAdapter",
    "SelfPlayEvolutionEngine",
    "TopologyContainer",
    "package_standalone_hk",
    "load_standalone_topology",
    "load_standalone_hk",
    "build_model_from_topology",
    "ExpansionDiagnosis",
    "ExpansionEvaluator",
    "ConversationalTurn",
    "SelfDialogue",
    "SelfConversationalEngine",
    "SelfTrainingCurriculum",
    "GenerationReport",
    "SelfTrainingPipeline",
]
