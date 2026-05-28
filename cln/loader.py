"""
Model loader for CLN: generic HuggingFace models.

HuggingFace path (Phi-3, TinyLlama, Qwen, LLaMA, Mistral, …)
--------------------------------------------------------------
``inject_plasticity()`` walks the model tree and replaces each ``nn.Linear``
with a ``LiquidLinear`` that inherits its weights as ``W_base`` and starts
with ``ΔW = 0``. The HuggingFace model continues to operate normally but now
learns on every forward pass. See ``load_hf()``.
"""

import os
from typing import Optional

import torch
import torch.nn as nn

from .core import LiquidLinear

MLP_MODULES: frozenset = frozenset({
    "gate_proj", "up_proj", "down_proj",
    "gate_up_proj",
    "c_fc",
    "fc1", "fc2",
    "w1", "w2", "w3",
})
"""Attribute names that correspond to MLP layers across common architectures.

``inject_plasticity()`` uses this set when ``target_names=MLP_MODULES`` to
restrict plasticity to feed-forward layers only, leaving attention projections
(Q/K/V/O) as static ``nn.Linear`` layers. This is faster and more stable than
injecting plasticity into the full model.

Covered families:

- LLaMA / Mistral / Qwen (SwiGLU): ``gate_proj``, ``up_proj``, ``down_proj``
- Qwen fused gate+up: ``gate_up_proj``
- GPT-2 / NanoGPT: ``c_fc``
- Phi-3 / BERT / generic: ``fc1``, ``fc2``
- LLaMA-1 alternative naming: ``w1``, ``w2``, ``w3``
"""


def inject_plasticity(
    model: nn.Module,
    liquid_kwargs: Optional[dict] = None,
    freeze_base: bool = True,
    skip_names: Optional[set] = None,
    target_names: Optional[frozenset] = None,
) -> int:
    """Recursively replace every ``nn.Linear`` in a model with ``LiquidLinear``.

    The original weights become ``W_base`` (optionally frozen). ``ΔW``,
    ``fisher``, and ``anchor_delta`` are initialized to zero. Works with any
    HuggingFace architecture: Phi-3, LLaMA, Mistral, TinyLlama, Qwen, etc.

    ``delta_w`` is created on the same device and dtype as the replaced linear's
    weights. ``fisher`` and ``anchor_delta`` are always stored on CPU in
    ``float32`` to avoid VRAM pressure on large models.

    Args:
        model: Any ``nn.Module`` — typically a HuggingFace causal LM.
        liquid_kwargs: Plasticity hyperparameters forwarded to every new
            ``LiquidLinear``.
        freeze_base: When True, ``W_base`` and its bias are frozen so that
            gradient-based optimizers cannot modify them.
        skip_names: Set of child attribute names to leave untouched. Defaults
            to ``{"lm_head"}``, which protects the output projection.
        target_names: When provided, only children whose attribute name appears
            in this set are replaced. All other ``nn.Linear`` layers are left
            static. Pass ``MLP_MODULES`` for MLP-only plasticity, which is
            significantly faster and more stable than full injection.

    Returns:
        The total number of ``nn.Linear`` layers replaced across the entire
        model tree.
    """
    if skip_names is None:
        skip_names = {"lm_head"}

    lkw = liquid_kwargs or {}
    count = 0

    for attr_name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and attr_name not in skip_names:
            if target_names is not None and attr_name not in target_names:
                continue

            src_dtype  = child.weight.data.dtype
            weight_dev = child.weight.device

            liquid = LiquidLinear(
                in_features  = child.in_features,
                out_features = child.out_features,
                bias         = child.bias is not None,
                **lkw,
            )

            with torch.no_grad():
                liquid.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    liquid.bias.data = child.bias.data.clone()

                liquid.register_buffer(
                    "delta_w",
                    torch.zeros(
                        child.out_features, child.in_features,
                        dtype=src_dtype, device=weight_dev,
                    ),
                )
                liquid.register_buffer(
                    "fisher",
                    torch.zeros(
                        child.out_features, child.in_features,
                        dtype=torch.float32, device="cpu",
                    ),
                )
                liquid.register_buffer(
                    "anchor_delta",
                    torch.zeros(
                        child.out_features, child.in_features,
                        dtype=torch.float32, device="cpu",
                    ),
                )

            if freeze_base:
                liquid.weight.requires_grad_(False)
                if liquid.bias is not None:
                    liquid.bias.requires_grad_(False)

            setattr(model, attr_name, liquid)
            count += 1

        elif not isinstance(child, LiquidLinear):
            count += inject_plasticity(child, lkw, freeze_base, skip_names, target_names)

    return count


def save_plastic_state_hf(model: nn.Module, path: str) -> None:
    """Serialize the plastic buffers of an injected HuggingFace model.

    Saves ``delta_w``, ``fisher``, and ``anchor_delta`` for every
    ``LiquidLinear`` layer, along with a ``__model_id__`` key used to detect
    checkpoint/model mismatches on load.

    Args:
        model: HuggingFace model with ``LiquidLinear`` layers injected by
            ``inject_plasticity()``.
        path: Destination file path (passed to ``torch.save``).
    """
    state: dict = {"__model_id__": getattr(model, "_cln_model_id", None)}
    for name, m in model.named_modules():
        if isinstance(m, LiquidLinear):
            state[name] = {
                "delta_w":      m.delta_w.cpu(),
                "fisher":       m.fisher.cpu(),
                "anchor_delta": m.anchor_delta.cpu(),
            }
    torch.save(state, path)


def load_plastic_state_hf(model: nn.Module, path: str) -> bool:
    """Restore plastic buffers into an injected HuggingFace model.

    Layers present in the checkpoint but missing from the model, or whose
    ``delta_w`` shape does not match, are silently skipped. This allows
    partial loading when the model architecture has changed.

    If the checkpoint was saved from a different model (detected via
    ``__model_id__``), the load is aborted and ``False`` is returned.

    Args:
        model: HuggingFace model with ``LiquidLinear`` layers injected by
            ``inject_plasticity()``.
        path: Source file path created by ``save_plastic_state_hf()``.

    Returns:
        True if the checkpoint was found and loaded; False if the file does
        not exist, cannot be parsed, or belongs to a different model.
    """
    if not os.path.exists(path):
        return False
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [CLN] Warning: could not load '{path}' ({e}). Skipping.")
        return False

    saved_id   = state.get("__model_id__", None)
    current_id = (
        getattr(model, "_cln_model_id", None)
        or getattr(getattr(model, "config", None), "_name_or_path", None)
    )
    if saved_id and current_id and saved_id != current_id:
        print(
            f"  [CLN] Warning: checkpoint is from '{saved_id}', "
            f"current model is '{current_id}'. Ignoring prior memory."
        )
        return False

    mods = {n: m for n, m in model.named_modules() if isinstance(m, LiquidLinear)}
    skipped = 0
    for name, ls in state.items():
        if name.startswith("__") or name not in mods:
            continue
        m = mods[name]
        dw_saved = ls["delta_w"]
        if dw_saved.shape != m.delta_w.shape:
            skipped += 1
            continue
        m.delta_w.copy_(dw_saved.to(m.delta_w.device, m.delta_w.dtype))
        m.fisher.copy_(ls["fisher"].float().cpu())
        m.anchor_delta.copy_(ls["anchor_delta"].float().cpu())

    if skipped:
        print(f"  [CLN] {skipped} layers skipped (shape mismatch).")
    return True


def consolidate_hf(model: nn.Module) -> None:
    """Run EWC consolidation on all LiquidLinear layers of a HuggingFace model.

    Args:
        model: HuggingFace model with ``LiquidLinear`` layers injected by
            ``inject_plasticity()``.
    """
    for m in model.modules():
        if isinstance(m, LiquidLinear):
            m.consolidate()


def load_hf(
    model_id: str,
    liquid_kwargs: Optional[dict] = None,
    freeze_base: bool = True,
    dtype: torch.dtype = torch.float16,
    device: Optional[str] = None,
    target_names: Optional[frozenset] = MLP_MODULES,
    verbose: bool = True,
    solver_mode: str = "ml_fast",
) -> tuple:
    """Load any causal HuggingFace model and inject CLN plasticity.

    The model is moved to the target device *before* plasticity is injected so
    that ``delta_w`` buffers are created directly on the target device. This
    avoids holding both a CPU copy and a device copy in memory simultaneously,
    which would cause OOM errors on large models.

    Requires ``transformers`` and ``accelerate``
    (``pip install transformers accelerate``).

    Compatible model IDs include (but are not limited to)::

        "microsoft/Phi-3-mini-4k-instruct"
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        "Qwen/Qwen2.5-0.5B-Instruct"

    Args:
        model_id: HuggingFace Hub model identifier.
        liquid_kwargs: Plasticity hyperparameters forwarded to every
            ``LiquidLinear``. Defaults to conservative values that keep
            ``ΔW`` small relative to ``W_base``.
        freeze_base: When True, ``W_base`` parameters are frozen.
        dtype: Weight dtype for the loaded model. ``torch.float16`` halves
            memory usage compared to ``float32``.
        device: Target device string (``"cpu"``, ``"mps"``, ``"cuda"``).
            When ``None``, the best available device is selected automatically.
        target_names: Set of attribute names to plasticize. Defaults to
            ``MLP_MODULES`` (feed-forward layers only). Pass ``None`` to
            inject plasticity into every ``nn.Linear`` in the model.
        verbose: When True, prints load progress and parameter counts.
        solver_mode: Plasticity solver passed to every ``LiquidLinear``.

    Returns:
        A ``(model, tokenizer)`` tuple. ``model`` has ``LiquidLinear`` layers
        injected and is ready for CLN inference.

    Raises:
        ImportError: If ``transformers`` or ``accelerate`` are not installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers accelerate")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if verbose:
        print(f"[CLN loader] Loading {model_id}")
        print(f"             dtype={dtype}  device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()

    lkw = liquid_kwargs or {
        "tau_w":      50.0,
        "eta":        5e-5,
        "lambda_ewc": 0.25,
        "dt":         0.05,
        "max_delta":  0.1,
    }
    if "solver_mode" not in lkw:
        lkw["solver_mode"] = solver_mode

    if verbose:
        print(f"[CLN loader] Moving to device ({device})...")
    model.to(device)

    if verbose:
        print("[CLN loader] Injecting CLN plasticity...")

    n_injected = inject_plasticity(model, lkw, freeze_base, target_names=target_names)
    model._cln_model_id = model_id

    if verbose:
        total_params   = sum(p.numel() for p in model.parameters())
        plastic_params = sum(
            m.delta_w.numel()
            for m in model.modules()
            if isinstance(m, LiquidLinear)
        )
        scope = (
            "MLP-only" if target_names is MLP_MODULES
            else ("all" if target_names is None else "custom")
        )
        print(f"[CLN loader] Done.  {n_injected} layers → LiquidLinear  ({scope})")
        print(f"             Base parameters    : {total_params:>14,}")
        print(f"             Plastic parameters : {plastic_params:>12,}  (ΔW on {device})")
        print(f"             Fisher / anchor    : on CPU (saves VRAM)")

    return model, tokenizer
