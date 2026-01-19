# from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3

__all__ = [
    "LlamaForCausalLMEagle3",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
