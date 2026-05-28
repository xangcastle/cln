"""
Intensive document learning via Hebbian plasticity.

Normal chat updates ΔW once per turn with a conservative learning rate.
To have the model *internalize* a full document into its plastic weights,
a more deliberate process is needed:

1. Split the text into overlapping chunks (sliding window).
2. Multiple epochs of exposure — the Hebbian rule requires repetition,
   just like biological memory consolidation.
3. A temporary eta multiplier — learn from the document faster without
   destabilizing later chat behavior.
4. EWC consolidation at the end — the acquired knowledge is anchored and
   survives future conversations without catastrophic forgetting.

The result is that ΔW encodes the statistical co-activations present in the
document: syntax, keyword patterns, type structure, etc. This is not RAG —
the knowledge lives in the weights, not in a retrieval index.

Quick start::

    from cln import load_hf
    from cln.learn import learn_document

    model, tokenizer = load_hf("Qwen/Qwen2.5-0.5B-Instruct")
    learn_document(model, open("manual.txt").read(), tokenizer=tokenizer)
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .core import LiquidLinear, set_plastic_mode


def _all_liquid(model: torch.nn.Module) -> List[LiquidLinear]:
    """Return a flat list of all LiquidLinear layers in the model."""
    return [m for m in model.modules() if isinstance(m, LiquidLinear)]


def _get_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter, or CPU if the model has none."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@contextlib.contextmanager
def _boosted_eta(layers: List[LiquidLinear], multiplier: float):
    """Temporarily scale the Hebbian learning rate of all layers by ``multiplier``.

    Restores the original ``eta`` values on exit, even if an exception is raised.

    Args:
        layers: List of LiquidLinear layers whose ``eta`` will be scaled.
        multiplier: Factor by which to multiply each layer's ``eta``.
    """
    originals = [layer.eta for layer in layers]
    for layer in layers:
        layer.eta = layer.eta * multiplier
    try:
        yield
    finally:
        for layer, orig in zip(layers, originals):
            layer.eta = orig


def _tokenize(
    text: str,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize text and return a 1-D token id tensor on ``device``.

    Args:
        text: Plain text to tokenize.
        tokenizer: HuggingFace tokenizer.
        model: Unused directly; present for interface symmetry.
        device: Target device for the output tensor.

    Returns:
        1-D ``torch.long`` tensor of token ids on ``device``.

    Raises:
        ValueError: If ``tokenizer`` is ``None``.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer is required.")
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long, device=device)


def _plastic_norm(layers: List[LiquidLinear]) -> float:
    """Return the sum of L2 norms of ``delta_w`` across all given layers."""
    return sum(layer.delta_w.float().norm().item() for layer in layers)


def learn_document(
    model: torch.nn.Module,
    text: str,
    tokenizer=None,
    epochs: int = 3,
    chunk_size: int = 512,
    stride: int = 256,
    eta_multiplier: float = 10.0,
    consolidate_each_epoch: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Teach a full document to the model by modifying its plastic weights.

    The text is split into overlapping chunks using a sliding window. Each
    chunk is passed through the model with ``plastic=True`` and a temporarily
    amplified eta. After all epochs, EWC consolidation anchors the acquired
    knowledge so it survives future chat interactions without being overwritten.

    Args:
        model: CLNModel or a HuggingFace causal LM with plasticity injected
            via ``inject_plasticity()``.
        text: Full text of the document to learn.
        tokenizer: HuggingFace tokenizer.
        epochs: Number of full passes over the document. 3–5 epochs is
            typically sufficient for medium-length texts.
        chunk_size: Number of tokens per context window.
        stride: Step size between consecutive chunk start positions.
            Values smaller than ``chunk_size`` create overlapping windows.
        eta_multiplier: Factor by which η is scaled during learning.
            10× is conservative; 50× for aggressive memorization.
        consolidate_each_epoch: When True, runs EWC consolidation at the end
            of each epoch. Recommended to protect knowledge between epochs.
        save_path: When provided, the plastic state is saved to this path
            after learning completes.
        verbose: When True, prints a progress bar and summary statistics.

    Returns:
        Dict with the following keys:

        - ``epochs``: Number of epochs completed.
        - ``chunks``: Total chunks processed across all epochs.
        - ``tokens``: Total tokens in the document.
        - ``plastic_norm_start``: ‖ΔW‖ before learning began.
        - ``plastic_norm_end``: ‖ΔW‖ after learning completed.
        - ``delta_norm``: Change in ‖ΔW‖ (end minus start).
        - ``time_s``: Total wall-clock time in seconds.

    Raises:
        RuntimeError: If the model contains no LiquidLinear layers.
    """
    layers = _all_liquid(model)
    if not layers:
        raise RuntimeError(
            "The model has no LiquidLinear layers. "
            "Call inject_plasticity() first."
        )

    device = _get_device(model)
    t0 = time.time()

    if verbose:
        print("\n[learn_document] Tokenizing document...")

    all_ids = _tokenize(text, tokenizer, model, device)
    n_tokens = len(all_ids)

    if n_tokens < chunk_size:
        chunk_size = n_tokens
        stride = max(1, n_tokens // 2)

    starts = list(range(0, n_tokens - chunk_size + 1, stride))
    if not starts:
        starts = [0]
    n_chunks = len(starts)

    norm_start = _plastic_norm(layers)

    if verbose:
        print(f"[learn_document] Document : {n_tokens:,} tokens")
        print(f"                 Chunks   : {n_chunks} × {chunk_size} tok  (stride={stride})")
        print(f"                 Epochs   : {epochs}  (η×{eta_multiplier})")
        print(f"                 Layers   : {len(layers)} LiquidLinear")
        print(f"                 ‖ΔW‖ ini : {norm_start:.5f}")
        print()

    model.eval()

    with _boosted_eta(layers, eta_multiplier):
        for epoch in range(epochs):
            epoch_start = time.time()
            set_plastic_mode(True)

            for ci, start in enumerate(starts):
                chunk = all_ids[start : start + chunk_size].unsqueeze(0)

                with torch.no_grad():
                    if tokenizer is not None:
                        model(chunk, use_cache=False)
                    else:
                        model(chunk, plastic=True)

                if verbose:
                    done = ci + 1
                    pct  = done / n_chunks * 100
                    bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(
                        f"\r  Epoch {epoch+1}/{epochs}  [{bar}] {pct:5.1f}%  "
                        f"chunk {done}/{n_chunks}",
                        end="", flush=True,
                    )

            set_plastic_mode(False)

            norm_now = _plastic_norm(layers)
            epoch_t  = time.time() - epoch_start

            if verbose:
                print(
                    f"\r  Epoch {epoch+1}/{epochs}  [████████████████████] 100.0%  "
                    f"‖ΔW‖={norm_now:.5f}  {epoch_t:.1f}s"
                )

            if consolidate_each_epoch:
                _consolidate(model)
                if verbose:
                    print(f"  → EWC consolidated (epoch {epoch+1} knowledge anchored)")

    if not consolidate_each_epoch:
        _consolidate(model)
        if verbose:
            print("\n  → EWC consolidated at end of training")

    norm_end = _plastic_norm(layers)
    elapsed  = time.time() - t0

    if save_path:
        _save(model, tokenizer, save_path)
        if verbose:
            print(f"\n[learn_document] Plastic state saved → {save_path}")

    stats = {
        "epochs":             epochs,
        "chunks":             n_chunks * epochs,
        "tokens":             n_tokens,
        "plastic_norm_start": round(norm_start, 6),
        "plastic_norm_end":   round(norm_end,   6),
        "delta_norm":         round(norm_end - norm_start, 6),
        "time_s":             round(elapsed, 2),
    }

    if verbose:
        print()
        print(f"[learn_document] ✓ Completed in {elapsed:.1f}s")
        print(f"                 ‖ΔW‖: {norm_start:.5f} → {norm_end:.5f}  "
              f"(Δ={stats['delta_norm']:+.5f})")
        print()

    return stats


def _consolidate(model: torch.nn.Module) -> None:
    """Run EWC consolidation on every LiquidLinear layer in the model."""
    for m in model.modules():
        if isinstance(m, LiquidLinear):
            m.consolidate()


def _save(model: torch.nn.Module, tokenizer, path: str) -> None:
    """Save the plastic state of a HuggingFace model with injected plasticity.

    Args:
        model: HuggingFace model with injected plasticity.
        tokenizer: Unused; present for interface symmetry.
        path: Destination file path.
    """
    from .loader import save_plastic_state_hf
    save_plastic_state_hf(model, path)


def learn_file(
    model: torch.nn.Module,
    path: str,
    tokenizer=None,
    encoding: str = "utf-8",
    **kwargs,
) -> Dict:
    """Learn from a text file by delegating to ``learn_document``.

    Reads the file at ``path`` and passes its contents to ``learn_document``
    with all remaining keyword arguments forwarded unchanged.

    Args:
        model: CLNModel or HuggingFace model with injected plasticity.
        path: Path to the text file to learn from. Accepts plain text,
            Markdown, reStructuredText, source code, or any UTF-8 encoded file.
        tokenizer: HuggingFace tokenizer.
        encoding: Character encoding used to read the file.
        **kwargs: Additional keyword arguments forwarded to ``learn_document``
            (e.g. ``epochs``, ``eta_multiplier``, ``save_path``).

    Returns:
        The statistics dict returned by ``learn_document``.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = p.read_text(encoding=encoding, errors="replace")

    if kwargs.get("verbose", True):
        size_kb = p.stat().st_size / 1024
        print(f"[learn_file] Reading '{p.name}' ({size_kb:.1f} KB, {len(text):,} chars)")

    return learn_document(model, text, tokenizer=tokenizer, **kwargs)