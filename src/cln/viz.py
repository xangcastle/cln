"""
ConversationTracker — records plastic state after each interaction and
generates a multi-panel figure showing how weights evolve over a conversation.

Panels produced by ``plot()``:

- **A** — Plastic norm ‖ΔW‖ per layer over time (line chart).
- **B** — Final plasticity ratio ‖ΔW‖ / ‖W_base‖ per layer (bar chart).
- **C1–C4** — ΔW heatmap snapshots at four conversation checkpoints.
- **D** — Fisher information Ω showing which connections are protected.
- **E** — ‖ΔW‖ vs ‖Ω‖ scatter revealing the learn/protect trade-off.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn

from .core import LiquidLinear


_ABBREV = {
    "q_proj":   "Aq",
    "k_proj":   "Ak",
    "v_proj":   "Av",
    "out_proj": "Ao",
    "fc1":      "F1",
    "fc2":      "F2",
}


def _short(name: str) -> str:
    """Convert a fully-qualified layer name into a compact display label.

    Strips ``layers`` and sub-module connector segments (``attn``, ``ff``,
    ``ffn``), prefixes digit segments with ``L``, expands known projection
    names via ``_ABBREV``, and joins the remaining parts with ``·``.

    Args:
        name: Fully-qualified module name as returned by ``named_modules()``.

    Returns:
        Abbreviated label string, e.g. ``"L0·Aq"`` for
        ``"layers.0.attn.q_proj"``.
    """
    parts = name.split(".")
    out = []
    for p in parts:
        if p == "layers":
            continue
        elif p.isdigit():
            out.append(f"L{p}")
        elif p in _ABBREV:
            out.append(_ABBREV[p])
        elif p in ("attn", "ff", "ffn"):
            continue
        else:
            out.append(p[:2].upper())
    return "·".join(out)


class ConversationTracker:
    """Records the plastic state of a CLNModel after each interaction.

    Snapshots are collected via ``record()`` and visualized as a six-panel
    matplotlib figure via ``plot()``. The tracker is intentionally lightweight:
    it only stores per-layer scalar norms and downsampled heatmaps for the
    first four layers, keeping memory overhead minimal even for long conversations.

    Example::

        tracker = ConversationTracker(model)
        for turn in conversation:
            ids = encode(turn)
            model(ids, plastic=True)
            tracker.record(label=turn[:30])

        fig = tracker.plot(save_path="evolution.png")
        plt.show()

    Attributes:
        model: The CLNModel being tracked.
        max_heatmap_dim: Maximum side length used when downsampling ΔW tensors
            for heatmap storage. Larger values increase memory usage.
        records: Ordered list of snapshot dicts, one per ``record()`` call.
            Each dict has keys ``norms``, ``fisher_norms``, ``ratios``, and
            ``heatmaps``.
        labels: Short text label associated with each snapshot, parallel to
            ``records``.
    """

    def __init__(self, model: nn.Module, max_heatmap_dim: int = 64):
        """Initialize the tracker.

        Args:
            model: Module whose LiquidLinear layers will be snapshotted.
            max_heatmap_dim: Maximum dimension (height and width) of stored
                heatmap arrays. Tensors larger than this are average-pooled
                before storage.
        """
        self.model = model
        self.max_heatmap_dim = max_heatmap_dim
        self.records: List[Dict] = []
        self.labels:  List[str]  = []
        self._layer_names: Optional[List[str]] = None

    def _liquid_layers(self) -> Dict[str, LiquidLinear]:
        """Return a mapping of layer names to LiquidLinear instances in the model."""
        return {
            name: m
            for name, m in self.model.named_modules()
            if isinstance(m, LiquidLinear)
        }

    def _downsample(self, t: torch.Tensor) -> np.ndarray:
        """Reduce a 2-D tensor to at most ``max_heatmap_dim × max_heatmap_dim``.

        Uses non-overlapping average pooling to preserve the overall magnitude
        distribution while keeping storage proportional to ``max_heatmap_dim²``.

        Args:
            t: Weight tensor of shape [H, W].

        Returns:
            Numpy array of shape [H', W'] where H' ≤ max_heatmap_dim and
            W' ≤ max_heatmap_dim.
        """
        arr = t.detach().float().cpu().numpy()
        H, W = arr.shape
        rh = max(1, H // self.max_heatmap_dim)
        rw = max(1, W // self.max_heatmap_dim)
        H2, W2 = (H // rh) * rh, (W // rw) * rw
        return arr[:H2, :W2].reshape(H2 // rh, rh, W2 // rw, rw).mean(axis=(1, 3))

    def record(self, label: str = "") -> None:
        """Snapshot the current plastic state of all LiquidLinear layers.

        Should be called immediately after each model forward pass or user
        interaction. Full heatmaps are stored only for the first four layers
        discovered; all layers contribute scalar norms and ratios.

        Args:
            label: Short description of the current interaction, displayed as
                a step annotation in panel A and as a subtitle in panels C1–C4.
        """
        layers = self._liquid_layers()
        if self._layer_names is None:
            self._layer_names = list(layers.keys())

        snap: Dict = {"norms": {}, "fisher_norms": {}, "ratios": {}, "heatmaps": {}}
        heatmap_candidates = list(self._layer_names)[:4]

        for name, m in layers.items():
            dn = m.delta_w.norm().item()
            bn = m.weight.data.norm().item()
            snap["norms"][name]        = dn
            snap["fisher_norms"][name] = m.fisher.norm().item()
            snap["ratios"][name]       = dn / (bn + 1e-8)
            if name in heatmap_candidates:
                snap["heatmaps"][name] = self._downsample(m.delta_w)

        self.records.append(snap)
        self.labels.append(label)

    def plot(
        self,
        save_path: Optional[str] = None,
        title: str = "CLN · Weight Evolution During Conversation",
        figsize: Tuple[int, int] = (20, 14),
    ) -> plt.Figure:
        """Generate a six-panel figure from all recorded snapshots.

        Layout (3 rows × 4 columns):

        - Row 0, cols 0–1 — **A**: plastic norm trajectory per layer.
        - Row 0, cols 2–3 — **B**: final plasticity ratio bar chart.
        - Row 1, cols 0–3 — **C1–C4**: ΔW heatmaps at four checkpoints.
        - Row 2, cols 0–1 — **D**: Fisher information heatmap (final step).
        - Row 2, cols 2–3 — **E**: ‖ΔW‖ vs ‖Ω‖ scatter plot (final step).

        Args:
            save_path: If provided, the figure is saved to this path at 150 dpi
                before being returned.
            title: Figure-level suptitle string.
            figsize: Width × height of the figure in inches.

        Returns:
            The matplotlib ``Figure`` object. Callers can further customize it
            or call ``plt.show()`` / ``fig.savefig()`` independently.

        Raises:
            RuntimeError: If ``record()`` has never been called.
        """
        if not self.records:
            raise RuntimeError("No data — call tracker.record() after each interaction first.")

        n      = len(self.records)
        steps  = list(range(n))
        names  = self._layer_names or []
        shorts = [_short(nm) for nm in names]

        palette = plt.get_cmap("tab20")
        colors  = [palette(i % 20) for i in range(len(names))]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

        gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.38)

        ax_norm   = fig.add_subplot(gs[0, :2])
        ax_ratio  = fig.add_subplot(gs[0, 2:])
        ax_heat   = [fig.add_subplot(gs[1, c]) for c in range(4)]
        ax_fisher = fig.add_subplot(gs[2, :2])
        ax_scat   = fig.add_subplot(gs[2, 2:])

        ax_norm.set_title("A · Plastic norm ‖ΔW‖ per layer over time", fontsize=10)
        ax_norm.set_xlabel("Interaction step")
        ax_norm.set_ylabel("‖ΔW‖")

        for i, (nm, sh) in enumerate(zip(names, shorts)):
            ys = [r["norms"].get(nm, 0.0) for r in self.records]
            ax_norm.plot(steps, ys, color=colors[i], linewidth=1.8, label=sh, alpha=0.9)

        for s, lbl in enumerate(self.labels):
            if lbl:
                ax_norm.axvline(s, color="#999999", linestyle=":", linewidth=0.7, alpha=0.6)

        ax_norm.legend(
            fontsize=6, ncol=max(1, len(names) // 4),
            loc="upper left", framealpha=0.75, handlelength=1.2,
        )
        ax_norm.grid(True, alpha=0.2)

        ax_ratio.set_title("B · Final plasticity ratio  ‖ΔW‖ / ‖W_base‖", fontsize=10)
        ax_ratio.set_xlabel("Layer")
        ax_ratio.set_ylabel("Ratio")

        final_ratios = [self.records[-1]["ratios"].get(nm, 0.0) for nm in names]
        ax_ratio.bar(range(len(names)), final_ratios, color=colors)
        ax_ratio.set_xticks(range(len(shorts)))
        ax_ratio.set_xticklabels(shorts, rotation=65, fontsize=6, ha="right")
        ax_ratio.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2%}"))
        ax_ratio.grid(True, axis="y", alpha=0.2)

        snap_layer = next(
            (nm for nm in names if nm in self.records[-1].get("heatmaps", {})),
            None,
        )

        checkpoints = [int(n * f) for f in (0.0, 0.33, 0.66, 1.0)]
        checkpoints[-1] = min(n - 1, checkpoints[-1])

        if snap_layer:
            all_arrs = [
                self.records[cp].get("heatmaps", {}).get(snap_layer, np.zeros((2, 2)))
                for cp in checkpoints
            ]
            vmax = max(np.abs(a).max() for a in all_arrs) + 1e-8

            for col, (cp, arr) in enumerate(zip(checkpoints, all_arrs)):
                lbl = (self.labels[cp] or f"step {cp}")[:22]
                im = ax_heat[col].imshow(
                    arr, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, interpolation="nearest",
                )
                ax_heat[col].set_title(
                    f"C{col+1} · {_short(snap_layer)}\nstep {cp}: {lbl}",
                    fontsize=8,
                )
                ax_heat[col].set_xlabel("pre →", fontsize=7)
                ax_heat[col].set_ylabel("post ↓", fontsize=7)
                ax_heat[col].tick_params(labelsize=6)
                fig.colorbar(im, ax=ax_heat[col], shrink=0.75, pad=0.02)
        else:
            for ax in ax_heat:
                ax.set_visible(False)

        ax_fisher.set_title(
            f"D · Fisher info Ω (importance) — {_short(snap_layer or '')} · final",
            fontsize=10,
        )
        if snap_layer:
            layer_obj = {
                nm: m for nm, m in self.model.named_modules()
                if isinstance(m, LiquidLinear)
            }.get(snap_layer)
            if layer_obj is not None:
                fisher_arr = self._downsample(layer_obj.fisher)
                im2 = ax_fisher.imshow(
                    fisher_arr, aspect="auto", cmap="YlOrRd",
                    vmin=0.0, interpolation="nearest",
                )
                ax_fisher.set_xlabel("pre-synaptic →")
                ax_fisher.set_ylabel("post-synaptic ↓")
                fig.colorbar(im2, ax=ax_fisher, shrink=0.8, pad=0.02)

        ax_scat.set_title("E · ‖ΔW‖ vs Fisher ‖Ω‖ per layer (final)", fontsize=10)
        ax_scat.set_xlabel("‖ΔW‖  (how much it changed)")
        ax_scat.set_ylabel("‖Ω‖   (how protected it is)")

        xs = [self.records[-1]["norms"].get(nm, 0.0)        for nm in names]
        ys = [self.records[-1]["fisher_norms"].get(nm, 0.0) for nm in names]

        ax_scat.scatter(xs, ys, color=colors, s=55, alpha=0.88, zorder=3)
        for i, sh in enumerate(shorts):
            ax_scat.annotate(
                sh, (xs[i], ys[i]),
                textcoords="offset points", xytext=(4, 3),
                fontsize=6, alpha=0.85,
            )

        if xs and ys:
            lim = max(max(xs), max(ys)) * 1.12
            ax_scat.plot([0, lim], [0, lim], "k--", linewidth=0.8,
                         alpha=0.35, label="‖Ω‖ = ‖ΔW‖")
            ax_scat.set_xlim(left=0)
            ax_scat.set_ylim(bottom=0)
            ax_scat.legend(fontsize=8, framealpha=0.7)
        ax_scat.grid(True, alpha=0.2)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved → {save_path}")

        return fig