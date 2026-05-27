"""
Continuous Liquid Networks (CLN)

A next-generation LLM architecture where:
- Weights are "liquid" (plastic) and evolve during inference
- Training and inference are a single unified process
- Knowledge accumulates permanently without catastrophic forgetting
- Inspired by Liquid Time-Constant (LTC) biological neural dynamics
"""

from .core import LiquidLinear
from .attention import LiquidMultiHeadAttention
from .memory import TopologicalMemory
from .model import CLNModel, CLNBlock
from .viz import ConversationTracker
from .loader import (
    load_gpt2, verify_load, GPT2_CONFIGS,
    load_hf, inject_plasticity, MLP_MODULES,
    save_plastic_state_hf, load_plastic_state_hf, consolidate_hf,
)
from .chat import CLNChat
from .learn import learn_document, learn_file

__version__ = "0.1.0"
__all__ = [
    "LiquidLinear",
    "LiquidMultiHeadAttention",
    "TopologicalMemory",
    "CLNModel",
    "CLNBlock",
    "ConversationTracker",
    "load_gpt2",
    "verify_load",
    "GPT2_CONFIGS",
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
