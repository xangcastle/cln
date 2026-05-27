"""
Model loaders for CLN: GPT-2 and generic HuggingFace models.

GPT-2 path
----------
Manually maps Conv1D weights to LiquidLinear layers inside a CLNModel.
See ``load_gpt2()``.

HuggingFace path (Phi-3, TinyLlama, Qwen, LLaMA, Mistral, …)
--------------------------------------------------------------
``inject_plasticity()`` walks the model tree and replaces each ``nn.Linear``
with a ``LiquidLinear`` that inherits its weights as ``W_base`` and starts
with ``ΔW = 0``. The HuggingFace model continues to operate normally but now
learns on every forward pass. See ``load_hf()``.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn

from .core import LiquidLinear
from .model import CLNModel


GPT2_CONFIGS = {
    "gpt2":        dict(d_model=768,  n_layers=12, n_heads=12, d_ff=3072,  max_seq_len=1024),
    "gpt2-medium": dict(d_model=1024, n_layers=24, n_heads=16, d_ff=4096,  max_seq_len=1024),
    "gpt2-large":  dict(d_model=1280, n_layers=36, n_heads=20, d_ff=5120,  max_seq_len=1024),
    "gpt2-xl":     dict(d_model=1600, n_layers=48, n_heads=25, d_ff=6400,  max_seq_len=1024),
}
"""Architecture hyperparameters for each GPT-2 variant."""

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


def _copy_weight(
    liquid: LiquidLinear,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    transpose: bool = True,
) -> None:
    """Copy a weight tensor into a LiquidLinear layer and reset its plastic state.

    GPT-2 uses ``Conv1D`` which stores weights in ``[in, out]`` layout, while
    ``nn.Linear`` (and ``LiquidLinear``) expect ``[out, in]``. When
    ``transpose=True``, the tensor is transposed before copying.

    Args:
        liquid: Destination LiquidLinear layer.
        weight: Source weight tensor, either ``[in, out]`` (Conv1D) or
            ``[out, in]`` (Linear).
        bias: Optional bias tensor to copy into ``liquid.bias``.
        transpose: When True, transposes ``weight`` before copying. Set to
            False when the source is already in ``[out, in]`` layout.

    Raises:
        ValueError: If the (possibly transposed) weight shape does not match
            ``liquid.weight.shape``.
    """
    w = weight.T if transpose else weight
    if w.shape != liquid.weight.shape:
        raise ValueError(
            f"Shape mismatch: weight {w.shape} vs LiquidLinear {liquid.weight.shape}"
        )
    with torch.no_grad():
        liquid.weight.data.copy_(w)
        liquid.delta_w.zero_()
        liquid.fisher.zero_()
        liquid.anchor_delta.zero_()
        if bias is not None and liquid.bias is not None:
            liquid.bias.data.copy_(bias)


def _init_tau_unit(tau_module) -> None:
    """Initialize a LiquidTimeConstant so that τ ≈ 1.0 for all tokens.

    This makes the CLNBlock mathematically identical to the original GPT-2
    block on the very first forward pass. The time constant adapts from there
    as the model interacts with new text.

    Derivation::

        τ(x) = τ_min + σ(W_τ·x + b) · (τ_max − τ_min)

    To obtain τ = 1.0 at initialization:

        σ(b) = (1.0 − τ_min) / (τ_max − τ_min)
        b    = logit(target_σ)

    ``W_τ`` is zeroed so the gate does not respond to the input initially.

    Args:
        tau_module: A ``LiquidTimeConstant`` instance to initialize in place.
    """
    tau_min   = tau_module.tau_min
    tau_range = tau_module.tau_range
    if tau_range < 1e-6:
        return

    target = (1.0 - tau_min) / tau_range
    target = max(0.01, min(0.99, target))
    init_bias = math.log(target / (1.0 - target))

    with torch.no_grad():
        nn.init.zeros_(tau_module.w_tau.weight)
        nn.init.constant_(tau_module.w_tau.bias, init_bias)


def _map_block(cln_block, gpt2_block, d_model: int) -> None:
    """Transfer weights from one GPT-2 transformer block to its CLN equivalent.

    Full mapping::

        gpt2.ln_1          → cln.norm1          (layer norm, direct copy)
        gpt2.attn.c_attn   → cln.attn.{q,k,v}  (fused QKV split on dim 0)
        gpt2.attn.c_proj   → cln.attn.out_proj
        gpt2.ln_2          → cln.norm2
        gpt2.mlp.c_fc      → cln.ff.fc1
        gpt2.mlp.c_proj    → cln.ff.fc2
        (no GPT-2 equiv.)  → cln.tau            (initialized to τ = 1.0)

    GPT-2's ``c_attn`` is a fused ``Conv1D(d_model → 3·d_model)`` stored as
    ``[d_model, 3·d_model]``. After transposing to ``[3·d_model, d_model]`` it
    is split into three ``[d_model, d_model]`` matrices along dimension 0.

    Args:
        cln_block: Destination ``CLNBlock`` instance.
        gpt2_block: Source GPT-2 transformer block (``GPT2Block``).
        d_model: Embedding dimension, used to split the fused QKV matrix.
    """
    g, c = gpt2_block, cln_block

    c.norm1.weight.data.copy_(g.ln_1.weight.data)
    c.norm1.bias.data.copy_(g.ln_1.bias.data)
    c.norm2.weight.data.copy_(g.ln_2.weight.data)
    c.norm2.bias.data.copy_(g.ln_2.bias.data)

    w_qkv = g.attn.c_attn.weight.T
    b_qkv = g.attn.c_attn.bias

    q_w, k_w, v_w = w_qkv.split(d_model, dim=0)
    q_b, k_b, v_b = b_qkv.split(d_model, dim=0)

    _copy_weight(c.attn.q_proj, q_w, q_b, transpose=False)
    _copy_weight(c.attn.k_proj, k_w, k_b, transpose=False)
    _copy_weight(c.attn.v_proj, v_w, v_b, transpose=False)

    _copy_weight(c.attn.out_proj, g.attn.c_proj.weight, g.attn.c_proj.bias)

    _copy_weight(c.ff.fc1, g.mlp.c_fc.weight,   g.mlp.c_fc.bias)
    _copy_weight(c.ff.fc2, g.mlp.c_proj.weight, g.mlp.c_proj.bias)

    _init_tau_unit(c.tau)


def load_gpt2(
    variant: str = "gpt2",
    liquid_kwargs: Optional[dict] = None,
    freeze_base: bool = True,
    verbose: bool = True,
    solver_mode: str = "ml_fast",
) -> CLNModel:
    """Download GPT-2 weights and return a CLNModel ready for inference.

    Builds a ``CLNModel`` with the architecture matching ``variant``, copies
    all GPT-2 weights into the corresponding ``LiquidLinear`` layers, and
    initializes ``ΔW = 0`` throughout. The result is mathematically equivalent
    to GPT-2 on the first forward pass and diverges gradually as plasticity
    accumulates.

    Requires ``transformers`` (``pip install transformers``).

    Args:
        variant: One of ``"gpt2"``, ``"gpt2-medium"``, ``"gpt2-large"``,
            ``"gpt2-xl"``.
        liquid_kwargs: Plasticity hyperparameters forwarded to every
            ``LiquidLinear`` layer. Defaults to conservative values
            (slow decay, low learning rate, strong EWC) to preserve GPT-2
            knowledge during early interactions.
        freeze_base: When True, ``W_base`` parameters are frozen (no gradient
            updates). Only ``ΔW`` evolves during CLN inference.
        verbose: When True, prints download progress and parameter counts.
        solver_mode: Plasticity solver passed to every ``LiquidLinear``.
            ``"ml_fast"`` (default) uses the standard Hebbian ODE path.

    Returns:
        A ``CLNModel`` with ``W_base`` loaded from GPT-2 and ``ΔW = 0``.

    Raises:
        ImportError: If ``transformers`` is not installed.
        ValueError: If ``variant`` is not a recognized GPT-2 variant.
    """
    try:
        from transformers import GPT2Model
    except ImportError:
        raise ImportError(
            "Install transformers first:\n"
            "  pip install transformers"
        )

    if variant not in GPT2_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Options: {list(GPT2_CONFIGS)}"
        )

    cfg = GPT2_CONFIGS[variant]

    lkw = liquid_kwargs or {
        "tau_w":      40.0,
        "eta":        1e-4,
        "lambda_ewc": 0.2,
        "dt":         0.05,
        "max_delta":  0.15,
    }
    if "solver_mode" not in lkw:
        lkw["solver_mode"] = solver_mode

    if verbose:
        n_params = {
            "gpt2": "124M", "gpt2-medium": "355M",
            "gpt2-large": "774M", "gpt2-xl": "1.5B",
        }[variant]
        print(f"[CLN loader] Downloading {variant} ({n_params}) from HuggingFace...")

    gpt2 = GPT2Model.from_pretrained(variant)
    gpt2.eval()

    if verbose:
        print("[CLN loader] Building CLNModel...")

    model = CLNModel(
        vocab_size    = 50257,
        d_model       = cfg["d_model"],
        n_layers      = cfg["n_layers"],
        n_heads       = cfg["n_heads"],
        d_ff          = cfg["d_ff"],
        max_seq_len   = cfg["max_seq_len"],
        dropout       = 0.0,
        liquid_kwargs = lkw,
    )

    with torch.no_grad():
        model.token_emb.weight.data.copy_(gpt2.wte.weight.data)
        model.pos_emb.weight.data.copy_(gpt2.wpe.weight.data)

    n = cfg["n_layers"]
    for i, (cln_block, gpt2_block) in enumerate(zip(model.layers, gpt2.h)):
        if verbose:
            print(f"[CLN loader]   layer {i+1:2d}/{n}", end="\r")
        _map_block(cln_block, gpt2_block, cfg["d_model"])
    if verbose:
        print()

    with torch.no_grad():
        model.norm.weight.data.copy_(gpt2.ln_f.weight.data)
        model.norm.bias.data.copy_(gpt2.ln_f.bias.data)

    if freeze_base:
        for m in model.modules():
            if isinstance(m, LiquidLinear):
                m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

    del gpt2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        counts = model.param_count()
        print("[CLN loader] Done.")
        print(f"             Static weights  : {counts['static']:>12,}  (W_base, frozen={freeze_base})")
        print(f"             Plastic weights : {counts['plastic']:>12,}  (ΔW = 0, ready to learn)")

    return model


def verify_load(
    model: CLNModel,
    variant: str = "gpt2",
    verbose: bool = True,
) -> dict:
    """Verify that GPT-2 weights were loaded correctly by comparing logits.

    Runs both the original ``GPT2LMHeadModel`` and the provided ``CLNModel``
    (with ``plastic=False``) on a fixed test sentence and measures agreement.
    A successful load yields ``top1_agreement ≥ 0.95`` and ``max_diff < 2.0``.
    Small residual differences are expected due to GELU exact vs. tanh
    approximation.

    Requires ``transformers`` (``pip install transformers``).

    Args:
        model: A ``CLNModel`` returned by ``load_gpt2()``.
        variant: The same GPT-2 variant string passed to ``load_gpt2()``.
        verbose: When True, prints a human-readable verification summary.

    Returns:
        Dict with four keys:

        - ``max_diff``: Maximum absolute logit difference (< 2.0 is good).
        - ``mean_diff``: Mean absolute logit difference (< 0.5 is good).
        - ``top1_agreement``: Fraction of token positions with matching argmax
          (> 0.95 is good).
        - ``top5_agreement``: Fraction where the GPT-2 top-1 token appears in
          the CLN top-5.
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError("pip install transformers")

    gpt2_lm = GPT2LMHeadModel.from_pretrained(variant).eval()

    test_ids = torch.tensor([[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]])

    with torch.no_grad():
        gpt2_logits = gpt2_lm(test_ids).logits
        cln_logits  = model(test_ids, plastic=False)

    diff = (cln_logits - gpt2_logits).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    gpt2_top1  = gpt2_logits.argmax(dim=-1)
    cln_top1   = cln_logits.argmax(dim=-1)
    top1_agree = (gpt2_top1 == cln_top1).float().mean().item()

    gpt2_top1_tok = gpt2_top1.unsqueeze(-1)
    cln_top5      = cln_logits.topk(5, dim=-1).indices
    top5_agree = (cln_top5 == gpt2_top1_tok).any(dim=-1).float().mean().item()

    result = {
        "max_diff":       round(max_diff,  4),
        "mean_diff":      round(mean_diff, 4),
        "top1_agreement": round(top1_agree, 4),
        "top5_agreement": round(top5_agree, 4),
    }

    if verbose:
        ok = "✓" if top1_agree >= 0.95 else "✗"
        print(f"[verify] {ok} Top-1 agreement  : {top1_agree:.1%}")
        print(f"[verify] {ok} Top-5 agreement  : {top5_agree:.1%}")
        print(f"[verify]    Max logit diff    : {max_diff:.4f}")
        print(f"[verify]    Mean logit diff   : {mean_diff:.4f}")
        if top1_agree >= 0.95:
            print("[verify] Load verified successfully.")
        else:
            print("[verify] Warning: larger difference than expected.")
            print("         Likely caused by exact vs. approximate GELU (normal).")

    del gpt2_lm
    return result


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