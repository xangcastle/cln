"""
Continuous Liquid Networks (CLN)

A next-generation LLM architecture where:
- Weights are "liquid" (plastic) and evolve during inference
- Training and inference are a single unified process
- Knowledge accumulates permanently without catastrophic forgetting
- Inspired by Liquid Time-Constant (LTC) biological neural dynamics
"""

from .attention import LiquidMultiHeadAttention
from .chat import CLNChat
from .core import LiquidLinear
from .learn import learn_document, learn_file
from .loader import (
    load_hf, inject_plasticity, MLP_MODULES,
    save_plastic_state_hf, load_plastic_state_hf, consolidate_hf,
)
from .memory import TopologicalMemory
from .model import CLNModel, CLNBlock
from .viz import ConversationTracker

__version__ = "0.1.0"
__all__ = [
    "LiquidLinear",
    "LiquidMultiHeadAttention",
    "TopologicalMemory",
    "CLNModel",
    "CLNBlock",
    "ConversationTracker",
    "load_hf",
    "inject_plasticity",
    "MLP_MODULES",
    "save_plastic_state_hf",
    "load_plastic_state_hf",
    "consolidate_hf",
    "CLNChat",
    "learn_document",
    "learn_file",
]
