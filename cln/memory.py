"""
TopologicalMemory — persistent storage for the network's plastic state.

This component makes CLN learning permanent across sessions:

- Saves all ΔW tensors (and Fisher information) to disk.
- Restores accumulated knowledge on startup.
- Runs periodic EWC consolidation cycles to protect important memories.
- Provides diagnostics on the current plastic state.

The name "topological" refers to the fact that effective connectivity between
neurons changes as ΔW evolves — the network's functional topology is different
after every interaction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import LiquidLinear


class TopologicalMemory:
    """Manages the persistent plastic state of a CLNModel.

    Handles serialization, restoration, periodic EWC consolidation, and
    diagnostics for all LiquidLinear layers found inside a model.

    Example::

        mem = TopologicalMemory(model, consolidation_interval=50)
        mem.load("memory.pt")
        # ... run inference ...
        mem.step("user said hello")
        mem.save("memory.pt")

    Attributes:
        model: The CLNModel (or any nn.Module containing LiquidLinear layers)
            whose plastic state this object manages.
        consolidation_interval: Number of steps between automatic EWC
            consolidation passes.
        step_count: Total number of steps recorded since creation or last load.
        interaction_log: Ordered list of logged interaction dicts, each with
            ``step`` and ``text`` keys. Capped at the last 1 000 entries on save.
    """

    def __init__(self, model: nn.Module, consolidation_interval: int = 100):
        """Initialize TopologicalMemory for a given model.

        Args:
            model: Module whose LiquidLinear layers will be tracked.
            consolidation_interval: How many steps to wait between automatic
                EWC consolidation passes.
        """
        self.model = model
        self.consolidation_interval = consolidation_interval
        self.step_count: int = 0
        self.interaction_log: List[Dict] = []

    def _liquid_layers(self) -> Dict[str, LiquidLinear]:
        """Return a mapping of fully-qualified layer names to LiquidLinear instances."""
        return {
            name: module
            for name, module in self.model.named_modules()
            if isinstance(module, LiquidLinear)
        }

    def save(self, path: str) -> None:
        """Serialize all plastic state to a file.

        Saves ``delta_w``, ``fisher``, and ``anchor_delta`` for every
        LiquidLinear layer, along with the step counter and the most recent
        1 000 interaction log entries.

        Args:
            path: Destination file path (passed directly to ``torch.save``).
        """
        state: Dict = {
            "step_count": self.step_count,
            "interaction_log": self.interaction_log[-1000:],
        }
        for name, layer in self._liquid_layers().items():
            state[name] = {
                "delta_w":      layer.delta_w.cpu(),
                "fisher":       layer.fisher.cpu(),
                "anchor_delta": layer.anchor_delta.cpu(),
            }
        torch.save(state, path)

    def load(self, path: str) -> bool:
        """Restore plastic state from a file.

        Copies saved tensors back to each LiquidLinear layer, respecting the
        device currently holding ``delta_w``. Layers present in the file but
        missing from the model (or whose entry is not a dict) are silently
        skipped.

        Args:
            path: Source file path created by a previous call to ``save``.

        Returns:
            True if the file existed and was loaded successfully; False if the
            file was not found.
        """
        if not Path(path).exists():
            return False

        state = torch.load(path, map_location="cpu", weights_only=False)
        layers = self._liquid_layers()

        for name, ls in state.items():
            if name not in layers or not isinstance(ls, dict):
                continue
            m = layers[name]
            device = m.delta_w.device
            m.delta_w.copy_(ls["delta_w"].to(device))
            m.fisher.copy_(ls["fisher"].to(device))
            m.anchor_delta.copy_(ls["anchor_delta"].to(device))

        self.step_count = state.get("step_count", 0)
        self.interaction_log = state.get("interaction_log", [])
        return True

    def consolidate(self) -> None:
        """Run EWC consolidation on every LiquidLinear layer in the model."""
        for layer in self._liquid_layers().values():
            layer.consolidate()

    def step(self, interaction: Optional[str] = None) -> None:
        """Record one interaction step and trigger consolidation when due.

        Should be called after each forward pass or user turn. Appends an
        entry to ``interaction_log`` (truncated to 200 characters) when
        ``interaction`` is provided. Calls ``consolidate`` automatically every
        ``consolidation_interval`` steps.

        Args:
            interaction: Optional text description of the current interaction,
                stored in the log for diagnostics.
        """
        self.step_count += 1
        if interaction:
            self.interaction_log.append({
                "step": self.step_count,
                "text": interaction[:200],
            })
        if self.step_count % self.consolidation_interval == 0:
            self.consolidate()

    def stats(self) -> Dict[str, Dict]:
        """Return per-layer plasticity statistics.

        Returns:
            A dict keyed by layer name. Each value contains:

            - ``delta_norm``: L2 norm of the plastic delta.
            - ``fisher_norm``: L2 norm of the Fisher information matrix.
            - ``base_norm``: L2 norm of the frozen base weights.
            - ``plasticity_ratio``: ``delta_norm / base_norm``, a measure of
              how much the layer has drifted from its pre-trained state.
        """
        result = {}
        for name, layer in self._liquid_layers().items():
            base_norm   = layer.weight.data.norm().item()
            delta_norm  = layer.delta_w.norm().item()
            fisher_norm = layer.fisher.norm().item()
            result[name] = {
                "delta_norm":       round(delta_norm, 6),
                "fisher_norm":      round(fisher_norm, 6),
                "base_norm":        round(base_norm, 6),
                "plasticity_ratio": round(delta_norm / (base_norm + 1e-8), 6),
            }
        return result

    def total_plastic_norm(self) -> float:
        """Return the sum of L2 norms of ``delta_w`` across all liquid layers."""
        return sum(
            layer.delta_w.norm().item()
            for layer in self._liquid_layers().values()
        )

    def summary(self) -> str:
        """Return a human-readable summary of the current memory state."""
        lines = [
            f"TopologicalMemory | steps={self.step_count} | "
            f"interactions={len(self.interaction_log)}",
            f"  Total plastic norm : {self.total_plastic_norm():.6f}",
            f"  Liquid layers      : {len(self._liquid_layers())}",
        ]
        return "\n".join(lines)


@dataclass
class MemoryTrace:
    """A single mnemonic trace stored in a MemoryConsolidation buffer.

    Attributes:
        pattern: Normalized activation pattern this trace encodes.
        strength: Current salience of the trace. Decays over time and is
            boosted by novelty at insertion.
        age: Number of decay steps applied since the trace was added.
        consolidated: Whether this trace has been merged into long-term memory.
    """

    pattern: torch.Tensor
    strength: float
    age: int = 0
    consolidated: bool = False

    def decay(self, factor: float = 0.99) -> None:
        """Apply exponential decay to the trace strength and increment its age.

        Args:
            factor: Multiplicative decay factor applied to ``strength``.
        """
        self.strength *= factor
        self.age += 1


class MemoryConsolidation(nn.Module):
    """Layer-level memory consolidation inspired by hippocampal-neocortical transfer.

    Maintains a ring buffer of recent activation traces and periodically
    consolidates high-strength traces into a long-term ``consolidated_memory``
    matrix. Low-strength and old traces are pruned via exponential decay.

    Consolidation runs every 100 forward steps; decay runs every 10 steps;
    memory rehearsal (random replay) runs every 50 steps at a configurable rate.

    Attributes:
        dim: Feature dimension of stored patterns.
        max_traces: Maximum number of traces held in the ring buffer.
        consolidation_threshold: Minimum strength required for a trace to
            contribute to a consolidation pass.
        rehearsal_rate: Probability of triggering a rehearsal pass on eligible
            steps.
        consolidated_memory: Long-term memory matrix, shape [dim, dim].
        trace_buffer: Ring buffer of stored patterns, shape [max_traces, dim].
        trace_strengths: Scalar strength for each buffer slot.
        trace_ages: Integer age (in decay steps) for each buffer slot.
    """

    def __init__(
        self,
        dim: int,
        max_traces: int = 1000,
        consolidation_threshold: float = 0.8,
        rehearsal_rate: float = 0.01,
    ):
        """Initialize the memory consolidation module.

        Args:
            dim: Feature dimension of the activation patterns to store.
            max_traces: Ring-buffer capacity. Older traces are overwritten
                once the buffer is full.
            consolidation_threshold: Strength threshold above which a trace
                is included in a consolidation pass.
            rehearsal_rate: Probability that a rehearsal pass fires on each
                eligible step (every 50 steps).
        """
        super().__init__()
        self.dim = dim
        self.max_traces = max_traces
        self.consolidation_threshold = consolidation_threshold
        self.rehearsal_rate = rehearsal_rate

        self.register_buffer("consolidated_memory", torch.zeros(dim, dim))
        self.register_buffer("trace_buffer",        torch.zeros(max_traces, dim))
        self.register_buffer("trace_strengths",     torch.zeros(max_traces))
        self.register_buffer("trace_ages",          torch.zeros(max_traces, dtype=torch.long))

        self.trace_count = 0
        self.time_step = 0

        self.importance_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def compute_importance(
        self, pattern: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Score the importance of a pattern given a context vector.

        Concatenates pattern and context, then passes them through the
        learned importance scorer.

        Args:
            pattern: Activation pattern, shape [..., dim].
            context: Context signal used to modulate importance, shape [..., dim].

        Returns:
            Importance score tensor, shape [..., 1], in [0, 1].
        """
        return self.importance_scorer(torch.cat([pattern, context], dim=-1))

    def add_trace(
        self, pattern: torch.Tensor, importance: Optional[torch.Tensor] = None
    ) -> None:
        """Insert one or more patterns into the ring buffer.

        Each pattern is L2-normalized before storage. When no importance is
        provided, novelty relative to ``consolidated_memory`` is used as a proxy.

        Args:
            pattern: Pattern(s) to store, shape [dim] or [batch, dim].
            importance: Optional importance scores, shape [batch, 1]. When
                omitted, novelty scores are computed automatically.
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)

        batch_size = pattern.shape[0]
        pattern = F.normalize(pattern, p=2, dim=-1)

        if importance is None:
            similarity = torch.matmul(pattern, self.consolidated_memory.t())
            novelty = 1.0 - torch.max(torch.abs(similarity), dim=-1)[0]
            importance = novelty.unsqueeze(-1)

        for i in range(batch_size):
            idx = self.trace_count % self.max_traces
            self.trace_buffer[idx]   = pattern[i]
            self.trace_strengths[idx] = importance[i].item()
            self.trace_ages[idx]     = 0
            self.trace_count += 1

    def consolidate(self, min_strength: float = 0.7) -> None:
        """Merge strong traces into the long-term consolidated memory matrix.

        Selects all active traces whose strength exceeds ``min_strength``,
        computes a strength-weighted average pattern, and updates
        ``consolidated_memory`` with an exponential moving average (α=0.05).
        Consolidated traces have their strength reduced to 30 % of the
        original to avoid repeated consolidation.

        Args:
            min_strength: Minimum strength threshold for trace inclusion.
        """
        if self.trace_count == 0:
            return

        num_active = min(self.trace_count, self.max_traces)
        candidates = self.trace_strengths[:num_active] > min_strength

        if candidates.sum() == 0:
            return

        candidate_patterns  = self.trace_buffer[:num_active][candidates]
        candidate_strengths = self.trace_strengths[:num_active][candidates]

        weights = candidate_strengths / (candidate_strengths.sum() + 1e-8)
        weighted_pattern = torch.sum(candidate_patterns * weights.unsqueeze(-1), dim=0)

        self.consolidated_memory = (
            0.95 * self.consolidated_memory
            + 0.05 * torch.outer(weighted_pattern, weighted_pattern)
        )
        self.trace_strengths[:num_active] *= (
            (~candidates).float() + candidates.float() * 0.3
        )

    def decay_traces(self, decay_factor: float = 0.995) -> None:
        """Apply exponential decay to all active traces and prune dead ones.

        A trace is considered dead when its strength drops below 0.01 or its
        age exceeds 1 000 steps. Dead traces have their strength zeroed out.

        Args:
            decay_factor: Multiplicative factor applied to all active strengths.
        """
        num_active = min(self.trace_count, self.max_traces)
        if num_active == 0:
            return

        self.trace_strengths[:num_active] *= decay_factor
        self.trace_ages[:num_active] += 1

        dead_mask = (
            (self.trace_strengths[:num_active] < 0.01)
            | (self.trace_ages[:num_active] > 1000)
        )
        if dead_mask.sum() > 0:
            self.trace_strengths[:num_active] *= (~dead_mask).float()

    def memory_protection_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute a protection mask that suppresses updates to consolidated weights.

        Returns a tensor in [0, 1] where values close to 0 indicate weights that
        are strongly aligned with consolidated memory and should be protected from
        further modification.

        Args:
            weights: Weight matrix to protect, shape [dim, dim]. If the shape
                does not match ``[dim, dim]``, a mask of all ones is returned.

        Returns:
            Protection mask of the same shape as ``weights``.
        """
        if weights.shape[0] != self.dim or weights.shape[1] != self.dim:
            return torch.ones_like(weights)

        alignment = torch.abs(weights * self.consolidated_memory)
        return 1.0 - torch.tanh(alignment * 5.0)

    def rehearse(self, num_samples: int = 5) -> Optional[torch.Tensor]:
        """Sample traces from the buffer weighted by their strength.

        Traces with higher strength are sampled more frequently, implementing
        a prioritized experience replay mechanism.

        Args:
            num_samples: Number of traces to sample without replacement.

        Returns:
            Tensor of shape [num_samples, dim] containing the sampled patterns,
            or None if the buffer is empty or the probability distribution is
            degenerate.
        """
        num_active = min(self.trace_count, self.max_traces)
        if num_active == 0:
            return None

        probs = F.softmax(self.trace_strengths[:num_active] * 10, dim=0)
        if torch.isnan(probs).any() or probs.sum() < 1e-6:
            return None

        indices = torch.multinomial(
            probs, min(num_samples, num_active), replacement=False
        )
        return self.trace_buffer[indices]

    def forward(
        self, pattern: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process a pattern: score its importance, store it, and run periodic maintenance.

        Consolidation fires every 100 steps, decay every 10 steps, and rehearsal
        every 50 steps (subject to ``rehearsal_rate``).

        Args:
            pattern: Activation pattern to store, shape [batch, dim].
            context: Optional context tensor used to modulate importance scoring.
                Defaults to ``pattern`` when not provided.

        Returns:
            Importance score tensor, shape [batch, 1], in [0, 1].
        """
        self.time_step += 1
        if context is None:
            context = pattern

        importance = self.compute_importance(pattern, context)
        self.add_trace(pattern, importance)

        if self.time_step % 100 == 0:
            self.consolidate()
        if self.time_step % 10 == 0:
            self.decay_traces()
        if self.time_step % 50 == 0 and torch.rand(1).item() < self.rehearsal_rate:
            self.rehearse()

        return importance