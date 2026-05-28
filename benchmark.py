#!/usr/bin/env python3
"""
CLN Benchmark — baseline measurement before optimizations.

Covers three areas from benchmark_plan.md:
  A. Speed       — tokens/sec with plastic=False vs plastic=True
  B. Memory      — peak RAM and plastic state file size
  C. Learning    — perplexity before/after learning; EWC retention

Usage:
    python benchmark.py                   # all tests, auto device
    python benchmark.py --skip-speed      # skip speed test
    python benchmark.py --device cpu      # force device
    python benchmark.py --tokens 500      # more tokens for speed test
"""

import argparse
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from cln import CLNModel
from cln.core import LiquidLinear, set_plastic_mode


# ── documents for learning tests ──────────────────────────────────────────────

DOCUMENT_A = (
    "The transformer architecture uses multi-head self-attention mechanisms. "
    "Each attention head learns to focus on different aspects of the input sequence. "
    "The feed-forward network processes each position independently after attention. "
    "Layer normalization stabilizes training by normalizing hidden activations. "
    "Residual connections allow gradients to flow through deep networks without vanishing. "
    "Positional embeddings encode the order of tokens in the sequence. "
    "Weight tying between the embedding and the language model head reduces parameters. "
    "Transformers have become the dominant architecture for natural language processing. "
)

DOCUMENT_B = (
    "Quantum computing encodes information in qubits that exist in superposition states. "
    "Entanglement correlates qubit states regardless of physical distance between them. "
    "Quantum gates apply unitary transformations to manipulate qubit states. "
    "Shor's algorithm factors large integers exponentially faster than classical methods. "
    "Grover's algorithm provides a quadratic speedup for unstructured database search. "
    "Quantum error correction protects computation from decoherence and noise. "
    "Current quantum processors are limited to tens or hundreds of noisy qubits. "
    "Fault-tolerant quantum computers will require millions of physical qubits per logical qubit. "
)


# ── helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print()
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)


def encode(text: str) -> torch.Tensor:
    ids = list(text.encode("utf-8"))
    return torch.tensor([ids], dtype=torch.long)


def make_model(device: torch.device) -> CLNModel:
    """Small but non-trivial CLNModel for fast benchmark iteration."""
    return CLNModel(
        vocab_size=256,
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.0,
        liquid_kwargs={
            "tau_w": 20.0, "eta": 5e-4,
            "lambda_ewc": 0.05, "dt": 0.1, "max_delta": 0.3,
        },
    ).eval().to(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 ** 2
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def compute_loss(model: CLNModel, ids: torch.Tensor) -> float:
    """Cross-entropy loss over a token sequence evaluated with plastic=False."""
    ids = ids[:, :model.max_seq_len]
    logits = model(ids, plastic=False)          # [1, T, V]
    return F.cross_entropy(
        logits[0, :-1],                         # [T-1, V]
        ids[0, 1:],                             # [T-1]
    ).item()


# ── HuggingFace helpers ───────────────────────────────────────────────────────

def _hf_reset_plasticity(model) -> None:
    for m in model.modules():
        if isinstance(m, LiquidLinear):
            m.reset_plasticity()


def _hf_consolidate(model) -> None:
    for m in model.modules():
        if isinstance(m, LiquidLinear):
            m.consolidate()


def _hf_plastic_layers(model) -> list:
    return [m for m in model.modules() if isinstance(m, LiquidLinear)]


def _hf_save_plastic(model, path: str) -> None:
    from cln.loader import save_plastic_state_hf
    save_plastic_state_hf(model, path)


@torch.no_grad()
def compute_loss_hf(model, ids: torch.Tensor) -> float:
    """Cross-entropy loss on a HuggingFace causal LM with plastic=False."""
    set_plastic_mode(False)
    out = model(ids, use_cache=False)
    logits = out.logits                         # [1, T, V]
    loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
    set_plastic_mode(True)
    return loss.item()


def _hf_max_len(model) -> int:
    return getattr(model.config, "max_position_embeddings", 2048)


@torch.no_grad()
def _hf_generate(model, input_ids: torch.Tensor, max_new_tokens: int, plastic: bool) -> torch.Tensor:
    """Minimal greedy generation loop for HF models."""
    set_plastic_mode(False)
    generated = input_ids.clone()
    try:
        from transformers.cache_utils import DynamicCache
        out = model(generated, past_key_values=DynamicCache(), use_cache=True)
        past = out.past_key_values
        use_cache = True
    except Exception:
        past = None
        use_cache = False

    for _ in range(max_new_tokens):
        if use_cache:
            next_in = generated[:, -1:]
            out = model(next_in, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            out = model(generated, use_cache=False)
        next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)

    if plastic:
        # Deferred learning: one plastic forward over the full generated context
        set_plastic_mode(True)
        model(generated[:, :_hf_max_len(model)], use_cache=False)
        set_plastic_mode(False)

    return generated


# ── A. Speed ──────────────────────────────────────────────────────────────────

def bench_speed(
    model: CLNModel,
    device: torch.device,
    n_tokens: int,
    n_warmup: int = 3,
) -> dict:
    prompt_ids = encode("Liquid networks learn continuously from every token.").to(device)
    results = {}

    for plastic in (False, True):
        label = "plastic=True " if plastic else "plastic=False"
        model.reset_plasticity()

        for _ in range(n_warmup):
            with torch.no_grad():
                model.generate(prompt_ids, max_new_tokens=20, plastic=plastic)

        model.reset_plasticity()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt_ids, max_new_tokens=n_tokens, plastic=plastic)
        elapsed = time.perf_counter() - t0

        tps = n_tokens / elapsed
        results[plastic] = {"tok_s": round(tps, 2), "elapsed_s": round(elapsed, 3)}
        print(f"  {label} : {tps:8.1f} tok/s  ({elapsed:.2f}s for {n_tokens} tokens)")

    overhead = results[False]["tok_s"] / max(results[True]["tok_s"], 1e-9)
    results["overhead_x"] = round(overhead, 2)
    print(f"  Plasticity overhead  : {overhead:.2f}x slower than static")
    return results


# ── B. Memory ─────────────────────────────────────────────────────────────────

def bench_memory(
    model: CLNModel,
    device: torch.device,
    n_tokens: int = 100,
) -> dict:
    prompt_ids = encode("Liquid networks learn continuously from every token.").to(device)
    results = {}

    for plastic in (False, True):
        label = "plastic=True " if plastic else "plastic=False"
        model.reset_plasticity()
        reset_peak_memory(device)

        with torch.no_grad():
            model.generate(prompt_ids, max_new_tokens=n_tokens, plastic=plastic)

        mb = peak_memory_mb(device)
        results[plastic] = mb
        peak_str = f"{mb:.1f} MB" if mb is not None else "N/A (CPU)"
        print(f"  {label} : peak memory = {peak_str}")

    # Plastic state file size
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    model.save_plastic_state(tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.unlink(tmp_path)
    results["storage_kb"] = round(size_kb, 2)
    print(f"  Plastic state (ΔW)   : {size_kb:.1f} KB on disk")
    return results


# ── C1. Memorization ──────────────────────────────────────────────────────────

def bench_memorization(
    model: CLNModel,
    device: torch.device,
    n_passes: int,
) -> dict:
    """Perplexity of document A before and after learning it."""
    ids = encode(DOCUMENT_A).to(device)[:, :model.max_seq_len]
    model.reset_plasticity()

    loss_before = compute_loss(model, ids)
    ppl_before = torch.exp(torch.tensor(loss_before)).item()
    print(f"  Before learning : loss={loss_before:.4f}  ppl={ppl_before:.1f}")

    with torch.no_grad():
        for _ in range(n_passes):
            model(ids, plastic=True)

    loss_after = compute_loss(model, ids)
    ppl_after = torch.exp(torch.tensor(loss_after)).item()
    reduction_pct = (loss_before - loss_after) / loss_before * 100
    print(f"  After  learning : loss={loss_after:.4f}  ppl={ppl_after:.1f}")
    print(f"  Loss reduction  : {reduction_pct:.1f}%  (higher = stronger memorization)")

    return {
        "loss_before": round(loss_before, 4),
        "loss_after":  round(loss_after,  4),
        "ppl_before":  round(ppl_before,  2),
        "ppl_after":   round(ppl_after,   2),
        "reduction_pct": round(reduction_pct, 1),
    }


# ── C2. EWC retention ─────────────────────────────────────────────────────────

def bench_ewc(
    model: CLNModel,
    device: torch.device,
    n_passes: int,
) -> dict:
    """
    1. Learn document A → measure its loss.
    2. EWC consolidate.
    3. Learn document B (interference).
    4. Re-measure loss on document A → quantify forgetting.
    """
    ids_a = encode(DOCUMENT_A).to(device)[:, :model.max_seq_len]
    ids_b = encode(DOCUMENT_B).to(device)[:, :model.max_seq_len]
    model.reset_plasticity()

    # Step 1 — learn A
    with torch.no_grad():
        for _ in range(n_passes):
            model(ids_a, plastic=True)

    loss_a_after_a = compute_loss(model, ids_a)
    ppl_a_after_a  = torch.exp(torch.tensor(loss_a_after_a)).item()
    print(f"  After learning A   — doc A  ppl : {ppl_a_after_a:.1f}")

    # Step 2 — consolidate
    model.consolidate_all()
    print("  EWC consolidation applied.")

    # Step 3 — learn B
    with torch.no_grad():
        for _ in range(n_passes):
            model(ids_b, plastic=True)

    # Step 4 — re-evaluate A
    loss_a_after_b = compute_loss(model, ids_a)
    ppl_a_after_b  = torch.exp(torch.tensor(loss_a_after_b)).item()
    degradation    = (loss_a_after_b - loss_a_after_a) / (abs(loss_a_after_a) + 1e-8) * 100
    retention      = max(0.0, 100.0 - degradation)
    print(f"  After learning B   — doc A  ppl : {ppl_a_after_b:.1f}")
    print(f"  Knowledge retention : {retention:.1f}%  (100% = no forgetting)")

    return {
        "ppl_a_after_a":  round(ppl_a_after_a,  2),
        "ppl_a_after_b":  round(ppl_a_after_b,  2),
        "degradation_pct": round(degradation,    1),
        "retention_pct":   round(retention,      1),
    }


# ── HuggingFace bench functions ───────────────────────────────────────────────

def bench_speed_hf(model, tokenizer, device: torch.device, n_tokens: int, n_warmup: int = 2) -> dict:
    prompt = "Liquid networks learn continuously from every token."
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    results = {}

    for plastic in (False, True):
        label = "plastic=True " if plastic else "plastic=False"
        _hf_reset_plasticity(model)

        for _ in range(n_warmup):
            _hf_generate(model, prompt_ids, max_new_tokens=10, plastic=plastic)

        _hf_reset_plasticity(model)
        t0 = time.perf_counter()
        _hf_generate(model, prompt_ids, max_new_tokens=n_tokens, plastic=plastic)
        elapsed = time.perf_counter() - t0

        tps = n_tokens / elapsed
        results[plastic] = {"tok_s": round(tps, 2), "elapsed_s": round(elapsed, 3)}
        print(f"  {label} : {tps:8.1f} tok/s  ({elapsed:.2f}s for {n_tokens} tokens)")

    overhead = results[False]["tok_s"] / max(results[True]["tok_s"], 1e-9)
    results["overhead_x"] = round(overhead, 2)
    print(f"  Plasticity overhead  : {overhead:.2f}x slower than static")
    return results


def bench_memory_hf(model, tokenizer, device: torch.device, n_tokens: int = 50) -> dict:
    prompt = "Liquid networks learn continuously from every token."
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    results = {}

    for plastic in (False, True):
        label = "plastic=True " if plastic else "plastic=False"
        _hf_reset_plasticity(model)
        reset_peak_memory(device)
        _hf_generate(model, prompt_ids, max_new_tokens=n_tokens, plastic=plastic)

        mb = peak_memory_mb(device)
        results[plastic] = mb
        peak_str = f"{mb:.1f} MB" if mb is not None else "N/A (CPU)"
        print(f"  {label} : peak memory = {peak_str}")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    _hf_save_plastic(model, tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.unlink(tmp_path)
    results["storage_kb"] = round(size_kb, 2)
    print(f"  Plastic state (ΔW)   : {size_kb:.1f} KB on disk")
    return results


def bench_memorization_hf(model, tokenizer, device: torch.device, n_passes: int) -> dict:
    max_len = _hf_max_len(model)
    ids_a = tokenizer.encode(DOCUMENT_A, return_tensors="pt").to(device)[:, :max_len]
    _hf_reset_plasticity(model)

    loss_before = compute_loss_hf(model, ids_a)
    ppl_before = torch.exp(torch.tensor(loss_before)).item()
    print(f"  Before learning : loss={loss_before:.4f}  ppl={ppl_before:.1f}")

    set_plastic_mode(True)
    with torch.no_grad():
        for _ in range(n_passes):
            model(ids_a, use_cache=False)
    set_plastic_mode(False)

    loss_after = compute_loss_hf(model, ids_a)
    ppl_after = torch.exp(torch.tensor(loss_after)).item()
    reduction_pct = (loss_before - loss_after) / loss_before * 100
    print(f"  After  learning : loss={loss_after:.4f}  ppl={ppl_after:.1f}")
    print(f"  Loss reduction  : {reduction_pct:.1f}%  (higher = stronger memorization)")

    return {
        "loss_before": round(loss_before, 4), "loss_after": round(loss_after, 4),
        "ppl_before": round(ppl_before, 2),   "ppl_after":  round(ppl_after,  2),
        "reduction_pct": round(reduction_pct, 1),
    }


def bench_ewc_hf(model, tokenizer, device: torch.device, n_passes: int) -> dict:
    max_len = _hf_max_len(model)
    ids_a = tokenizer.encode(DOCUMENT_A, return_tensors="pt").to(device)[:, :max_len]
    ids_b = tokenizer.encode(DOCUMENT_B, return_tensors="pt").to(device)[:, :max_len]
    _hf_reset_plasticity(model)

    set_plastic_mode(True)
    with torch.no_grad():
        for _ in range(n_passes):
            model(ids_a, use_cache=False)
    set_plastic_mode(False)

    loss_a_after_a = compute_loss_hf(model, ids_a)
    ppl_a_after_a  = torch.exp(torch.tensor(loss_a_after_a)).item()
    print(f"  After learning A   — doc A  ppl : {ppl_a_after_a:.1f}")

    _hf_consolidate(model)
    print("  EWC consolidation applied.")

    set_plastic_mode(True)
    with torch.no_grad():
        for _ in range(n_passes):
            model(ids_b, use_cache=False)
    set_plastic_mode(False)

    loss_a_after_b = compute_loss_hf(model, ids_a)
    ppl_a_after_b  = torch.exp(torch.tensor(loss_a_after_b)).item()
    degradation    = (loss_a_after_b - loss_a_after_a) / (abs(loss_a_after_a) + 1e-8) * 100
    retention      = max(0.0, 100.0 - degradation)
    print(f"  After learning B   — doc A  ppl : {ppl_a_after_b:.1f}")
    print(f"  Knowledge retention : {retention:.1f}%  (100% = no forgetting)")

    return {
        "ppl_a_after_a": round(ppl_a_after_a, 2), "ppl_a_after_b": round(ppl_a_after_b, 2),
        "degradation_pct": round(degradation, 1),  "retention_pct":  round(retention, 1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

# ── plots ─────────────────────────────────────────────────────────────────────

_BLUE   = "#4C72B0"
_ORANGE = "#DD8452"
_GREEN  = "#55A868"
_RED    = "#C44E52"
_GRAY   = "#8C8C8C"


def _bar(ax, labels, values, colors, title, ylabel, fmt=".1f"):
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:{fmt}}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def plot_results(results: dict, save_path: str = "benchmark.png") -> None:
    import matplotlib.pyplot as plt

    has = {k: k in results for k in ("speed", "memory", "memorization", "ewc")}
    n_panels = sum(has.values())
    if n_panels == 0:
        return

    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    axes_flat = [axes] if n_panels == 1 else list(
        axes.flat if hasattr(axes, "flat") else [axes]
    )
    ax_iter = iter(axes_flat)

    device_label = results.get("device", "")

    # ── A: Speed ──
    if has["speed"]:
        ax = next(ax_iter)
        s = results["speed"]
        _bar(
            ax,
            labels=["Static\n(plastic=False)", "Plastic\n(plastic=True)"],
            values=[s[False]["tok_s"], s[True]["tok_s"]],
            colors=[_BLUE, _ORANGE],
            title="A · Inference Speed",
            ylabel="Tokens / second  (higher = faster)",
        )
        ax.annotate(
            f"{s['overhead_x']}× overhead",
            xy=(1, s[True]["tok_s"]),
            xytext=(0.55, s[False]["tok_s"] * 0.65),
            arrowprops=dict(arrowstyle="->", color=_RED, lw=1.5),
            fontsize=10, color=_RED, fontweight="bold",
        )

    # ── B: Memory / Storage ──
    if has["memory"]:
        ax = next(ax_iter)
        m = results["memory"]
        storage_mb = m["storage_kb"] / 1024

        mb_static  = m[False]
        mb_plastic = m[True]
        if mb_static is not None:
            _bar(
                ax,
                labels=["Static\n(peak RAM)", "Plastic\n(peak RAM)", "ΔW file\n(disk)"],
                values=[mb_static, mb_plastic, storage_mb],
                colors=[_BLUE, _ORANGE, _GRAY],
                title="B · Memory Footprint",
                ylabel="Megabytes  (lower = better)",
            )
        else:
            _bar(
                ax,
                labels=["ΔW state\n(disk)"],
                values=[storage_mb],
                colors=[_GRAY],
                title="B · Plastic State Size",
                ylabel="Megabytes  (lower = better)",
            )

    # ── C1: Memorization ──
    if has["memorization"]:
        ax = next(ax_iter)
        lrn = results["memorization"]
        _bar(
            ax,
            labels=["Before\nlearning", "After\nlearning"],
            values=[lrn["ppl_before"], lrn["ppl_after"]],
            colors=[_RED, _GREEN],
            title="C1 · Memorization  (Perplexity)",
            ylabel="Perplexity  (lower = better learned)",
            fmt=".0f",
        )
        pct = lrn["reduction_pct"]
        sign = "−" if pct >= 0 else "+"
        ax.text(
            0.97, 0.95,
            f"{sign}{abs(pct):.1f}% loss",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10, color=_GREEN if pct >= 0 else _RED,
            fontweight="bold",
        )

    # ── C2: EWC Retention ──
    if has["ewc"]:
        ax = next(ax_iter)
        ewc = results["ewc"]
        _bar(
            ax,
            labels=["After A\n(learned)", "After B\n(interference)"],
            values=[ewc["ppl_a_after_a"], ewc["ppl_a_after_b"]],
            colors=[_GREEN, _ORANGE],
            title="C2 · EWC Retention  (Doc A Perplexity)",
            ylabel="Perplexity  (lower = better retained)",
            fmt=".0f",
        )
        ax.text(
            0.97, 0.95,
            f"Retention: {ewc['retention_pct']:.1f}%",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10, color=_BLUE,
            fontweight="bold",
        )

    # Hide unused panels
    for ax in ax_iter:
        ax.set_visible(False)

    fig.suptitle(
        f"CLN Benchmark — Baseline  ({device_label})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CLN baseline benchmark")
    p.add_argument("--hf", default=None, metavar="MODEL_ID",
                   help="HuggingFace model ID to benchmark instead of the built-in CLNModel")
    p.add_argument("--lora-rank", type=int, default=0,
                   help="Liquid LoRA rank for HF models (0 = full ΔW, default: 0)")
    p.add_argument("--eta", type=float, default=None,
                   help="Hebbian learning rate (default: 5e-5 standard, 1e-6 with LoRA)")
    p.add_argument("--tokens", type=int, default=200,
                   help="Tokens to generate in speed test (default: 200)")
    p.add_argument("--passes", type=int, default=5,
                   help="Learning passes for perplexity tests (default: 5)")
    p.add_argument("--device", default=None,
                   help="cpu | mps | cuda  (default: auto-detect)")
    p.add_argument("--skip-speed",  action="store_true", help="Skip section A")
    p.add_argument("--skip-memory", action="store_true", help="Skip section B")
    p.add_argument("--skip-learn",  action="store_true", help="Skip section C")
    p.add_argument("--no-plot", action="store_true",    help="Skip chart generation")
    p.add_argument("--plot-out", default="benchmark.png",
                   help="Output path for the chart (default: benchmark.png)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  CLN Benchmark — Baseline                                      ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    results: dict = {"device": str(device)}
    tokenizer = None

    if args.hf:
        from cln.loader import load_hf

        lora_rank = args.lora_rank
        # Conservative eta: much smaller when using LoRA to protect pretrained weights
        eta = args.eta if args.eta is not None else (1e-6 if lora_rank > 0 else 5e-5)
        liquid_kwargs = {
            "eta":        eta,
            "tau_w":      200.0 if lora_rank > 0 else 50.0,
            "lambda_ewc": 0.5   if lora_rank > 0 else 0.25,
            "dt":         0.01  if lora_rank > 0 else 0.05,
            "max_delta":  0.02  if lora_rank > 0 else 0.1,
        }

        print(f"  Loading  : {args.hf}")
        if lora_rank > 0:
            print(f"  LoRA     : rank={lora_rank}  eta={eta:.2e}")
        model, tokenizer = load_hf(
            args.hf, device=str(device), verbose=True,
            lora_rank=lora_rank, liquid_kwargs=liquid_kwargs,
        )
        model.eval()
        n_plastic = len(_hf_plastic_layers(model))
        print(f"  Device   : {device}")
        print(f"  Model    : {args.hf}  ({n_plastic} plastic layers)")
        print(f"  Tokens   : {args.tokens}  (speed test)")
        print(f"  Passes   : {args.passes}  (learning tests)")

        if not args.skip_speed:
            section("A · Speed  (tokens / second)")
            results["speed"] = bench_speed_hf(model, tokenizer, device, n_tokens=args.tokens)

        if not args.skip_memory:
            section("B · Memory  (peak RAM + plastic state size)")
            results["memory"] = bench_memory_hf(model, tokenizer, device)

        if not args.skip_learn:
            section("C1 · Memorization  (perplexity before / after learning)")
            results["memorization"] = bench_memorization_hf(model, tokenizer, device, n_passes=args.passes)

            section("C2 · EWC Retention  (catastrophic forgetting)")
            results["ewc"] = bench_ewc_hf(model, tokenizer, device, n_passes=args.passes)

    else:
        model = make_model(device)
        counts = model.param_count()
        print(f"  Device  : {device}")
        print(f"  Model   : CLNModel  d=256  L=6  H=4")
        print(f"            {counts['static']:,} static params | {counts['plastic']:,} plastic params")
        print(f"  Tokens  : {args.tokens}  (speed test)")
        print(f"  Passes  : {args.passes}  (learning tests)")

        if not args.skip_speed:
            section("A · Speed  (tokens / second)")
            results["speed"] = bench_speed(model, device, n_tokens=args.tokens)

        if not args.skip_memory:
            section("B · Memory  (peak RAM + plastic state size)")
            results["memory"] = bench_memory(model, device)

        if not args.skip_learn:
            section("C1 · Memorization  (perplexity before / after learning)")
            results["memorization"] = bench_memorization(model, device, n_passes=args.passes)

            section("C2 · EWC Retention  (catastrophic forgetting)")
            results["ewc"] = bench_ewc(model, device, n_passes=args.passes)

    # ── summary ──────────────────────────────────────────────────────────────
    section("Summary — Baseline Numbers")

    if "speed" in results:
        s = results["speed"]
        print(f"  [A] Tok/s static   : {s[False]['tok_s']}")
        print(f"  [A] Tok/s plastic  : {s[True]['tok_s']}")
        print(f"  [A] Overhead       : {s['overhead_x']}x")

    if "memory" in results:
        m = results["memory"]
        mb_static  = m[False]
        mb_plastic = m[True]
        if mb_static is not None:
            print(f"  [B] Peak RAM static  : {mb_static:.1f} MB")
            print(f"  [B] Peak RAM plastic : {mb_plastic:.1f} MB")
        print(f"  [B] ΔW file size     : {m['storage_kb']:.1f} KB")

    if "memorization" in results:
        lrn = results["memorization"]
        print(f"  [C1] PPL before      : {lrn['ppl_before']}")
        print(f"  [C1] PPL after       : {lrn['ppl_after']}")
        print(f"  [C1] Loss reduction  : {lrn['reduction_pct']}%")

    if "ewc" in results:
        ewc = results["ewc"]
        print(f"  [C2] PPL after A     : {ewc['ppl_a_after_a']}")
        print(f"  [C2] PPL after B int : {ewc['ppl_a_after_b']}")
        print(f"  [C2] EWC retention   : {ewc['retention_pct']}%")

    print()

    if not args.no_plot:
        section("Generating chart")
        plot_results(results, save_path=args.plot_out)

    print()


if __name__ == "__main__":
    main()
