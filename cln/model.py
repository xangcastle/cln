"""
CLNModel — Continuous Liquid Network Language Model.

Full architecture::

    TokenEmbedding (static)
        +
    PositionalEmbedding (static)
        ↓
    N × CLNBlock:
        LiquidMultiHeadAttention   ← plastic Q, K, V, O projections
        LiquidFeedForward          ← plastic fc1, fc2
        LiquidTimeConstant         ← per-token τ gates residual strength
        LayerNorm (static)
        ↓
    LayerNorm (static)
        ↓
    LM Head  (weight-tied to embedding, static)

Each CLNBlock computes:

    τ  = LiquidTimeConstant(x)
    x ← x + (1/τ) · Attention(norm(x))
    x ← x + (1/τ) · FFN(norm(x))

This is the Euler discretization of the LTC ODE:

    dx/dt = -x/τ(x) + F(x)

Calling ``forward(input_ids, plastic=True)`` simultaneously produces
next-token logits (inference) and updates all ΔW tensors via Hebbian
plasticity (learning). There is no separate training loop.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LiquidMultiHeadAttention
from .core import LiquidLinear, LiquidTimeConstant


class LiquidFeedForward(nn.Module):
    """Two-layer feed-forward network with plastic weights.

    Projects from ``d_model`` to ``d_ff`` via ``fc1``, applies GELU
    activation, then projects back to ``d_model`` via ``fc2``. Both
    linear layers are LiquidLinear and receive Hebbian updates each
    forward pass when plasticity is active.

    Attributes:
        fc1: Plastic projection from d_model to d_ff.
        fc2: Plastic projection from d_ff back to d_model.
        act: GELU activation (tanh approximation, matching GPT-2's gelu_new).
        drop: Dropout applied between fc1 and fc2.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        liquid_kwargs: Optional[dict] = None,
    ):
        """Initialize the liquid feed-forward block.

        Args:
            d_model: Input and output feature dimension.
            d_ff: Hidden dimension of the feed-forward expansion.
            dropout: Dropout probability applied between the two projections.
            liquid_kwargs: Extra keyword arguments forwarded to both
                LiquidLinear layers.
        """
        super().__init__()
        lkw = liquid_kwargs or {}
        self.fc1  = LiquidLinear(d_model, d_ff, **lkw)
        self.fc2  = LiquidLinear(d_ff, d_model, **lkw)
        self.act  = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, plastic: bool = True) -> torch.Tensor:
        """Apply the two-layer transformation.

        Args:
            x: Input tensor of shape [..., d_model].
            plastic: When True, both projections run their Hebbian update.

        Returns:
            Output tensor of shape [..., d_model].
        """
        return self.fc2(
            self.drop(self.act(self.fc1(x, plastic=plastic))),
            plastic=plastic,
        )

    def consolidate(self, importance: Optional[torch.Tensor] = None) -> None:
        """Run EWC consolidation on both projection layers.

        Args:
            importance: Optional per-weight importance tensor forwarded to
                each LiquidLinear layer.
        """
        self.fc1.consolidate(importance)
        self.fc2.consolidate(importance)


class CLNBlock(nn.Module):
    """Single Continuous Liquid Network transformer block.

    Applies liquid-gated multi-head attention followed by a liquid-gated
    feed-forward network. Both sub-layers are scaled by the inverse of a
    per-neuron time constant τ, implementing one Euler step of the LTC ODE:

        τ   = τ_min + σ(W_τ · x) · (τ_max − τ_min)
        h   = x + (1/τ) · MHA(LayerNorm(x))
        out = h + (1/τ) · FFN(LayerNorm(h))

    Attributes:
        attn: Liquid multi-head self-attention sub-layer.
        ff: Liquid feed-forward sub-layer.
        norm1: Layer normalization applied before attention.
        norm2: Layer normalization applied before the feed-forward network.
        tau: Per-neuron adaptive time constant module.
        drop: Dropout applied to each sub-layer output before the residual add.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        liquid_kwargs: Optional[dict] = None,
    ):
        """Initialize a CLNBlock.

        Args:
            d_model: Embedding dimension shared across all sub-layers.
            n_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward expansion.
            dropout: Dropout probability used in attention, FFN, and residual.
            liquid_kwargs: Plasticity hyperparameters forwarded to all
                LiquidLinear layers within this block.
        """
        super().__init__()
        lkw = liquid_kwargs or {}
        self.attn  = LiquidMultiHeadAttention(d_model, n_heads, dropout, lkw)
        self.ff    = LiquidFeedForward(d_model, d_ff, dropout, lkw)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.tau   = LiquidTimeConstant(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        plastic: bool = True,
    ) -> torch.Tensor:
        """Apply liquid-gated attention and feed-forward transformations.

        Args:
            x: Input tensor of shape [B, T, D].
            mask: Causal mask of shape [1, 1, T, T] passed to attention.
            plastic: When True, all LiquidLinear layers inside this block
                perform their Hebbian update.

        Returns:
            Output tensor of shape [B, T, D].
        """
        tau = self.tau(x)
        x = x + (1.0 / tau) * self.drop(self.attn(self.norm1(x), mask=mask, plastic=plastic))
        x = x + (1.0 / tau) * self.drop(self.ff(self.norm2(x), plastic=plastic))
        return x

    def consolidate(self) -> None:
        """Run EWC consolidation on all liquid layers within this block."""
        self.attn.consolidate()
        self.ff.consolidate()


class CLNModel(nn.Module):
    """Continuous Liquid Network Language Model.

    A transformer language model where the feed-forward and attention
    projection layers are plastic: they accumulate Hebbian updates on every
    forward pass, making training and inference a single unified process.

    Two operating modes:

    - ``plastic=True`` (default): CLN mode. Every forward pass updates all ΔW
      tensors. The model learns continuously from the text it processes.
    - ``plastic=False``: Static mode. Behaves identically to a standard
      transformer. Useful for evaluation or gradient-based pre-training.

    Example::

        model = CLNModel(vocab_size=50257, d_model=512, n_layers=6, n_heads=8)
        logits = model(input_ids)
        model.save_plastic_state("memory.pt")
        model.load_plastic_state("memory.pt")

    Attributes:
        d_model: Embedding dimension.
        vocab_size: Size of the token vocabulary.
        max_seq_len: Maximum supported sequence length.
        token_emb: Static token embedding table.
        pos_emb: Static positional embedding table.
        layers: Stack of CLNBlock transformer layers.
        norm: Final layer normalization before the LM head.
        lm_head: Output projection weight-tied to ``token_emb``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        liquid_kwargs: Optional[dict] = None,
    ):
        """Initialize the CLNModel.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Embedding and hidden dimension throughout the model.
            n_layers: Number of stacked CLNBlocks.
            n_heads: Number of attention heads per block.
            d_ff: Feed-forward hidden dimension inside each block.
            max_seq_len: Maximum sequence length for positional embeddings.
            dropout: Dropout probability applied throughout the model.
            liquid_kwargs: Plasticity hyperparameters forwarded to every
                LiquidLinear layer. Defaults to ``{tau_w=20, eta=5e-4,
                lambda_ewc=0.05, dt=0.1, max_delta=0.3}``.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        lkw = liquid_kwargs or {
            "tau_w":      20.0,
            "eta":        5e-4,
            "lambda_ewc": 0.05,
            "dt":         0.1,
            "max_delta":  0.3,
        }

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            CLNBlock(d_model, n_heads, d_ff, dropout, lkw)
            for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply GPT-2-style weight initialization (normal with std=0.02)."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, LiquidLinear)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build a lower-triangular causal mask of shape [1, 1, T, T]."""
        return torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)

    def forward(
        self,
        input_ids: torch.Tensor,
        plastic: bool = True,
    ) -> torch.Tensor:
        """Run a forward pass that optionally learns from the input.

        When ``plastic=True``, this call simultaneously performs inference
        (returning next-token logits) and learning (updating all ΔW tensors
        via the Hebbian ODE). There is no separate training loop.

        Args:
            input_ids: Integer token ids of shape [B, T].
            plastic: When True, all LiquidLinear layers update their plastic
                weights after computing their output.

        Returns:
            Logits tensor of shape [B, T, vocab_size].
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        device = input_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        mask = self._causal_mask(T, device)

        for layer in self.layers:
            x = layer(x, mask=mask, plastic=plastic)

        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        plastic: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with optional online learning.

        When ``plastic=True``, each generated token triggers a Hebbian weight
        update — the model learns from its own outputs in real time during
        generation.

        Args:
            input_ids: Seed token ids of shape [B, T_prompt].
            max_new_tokens: Number of new tokens to append.
            temperature: Softmax temperature. Higher values increase randomness.
            top_k: Restricts sampling to the top-k logits. Pass 0 to disable.
            plastic: When True, plastic weights are updated at each step.

        Returns:
            Token id tensor of shape [B, T_prompt + max_new_tokens].
        """
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.max_seq_len:]
            logits = self(ctx, plastic=plastic)[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    def consolidate_all(self) -> None:
        """Run EWC consolidation on every CLNBlock in the model."""
        for layer in self.layers:
            layer.consolidate()

    def reset_plasticity(self) -> None:
        """Zero all plastic weights across every LiquidLinear layer."""
        for m in self.modules():
            if isinstance(m, LiquidLinear):
                m.reset_plasticity()

    def save_plastic_state(self, path: str) -> None:
        """Serialize only the plastic weights to a file.

        The saved checkpoint contains ``delta_w``, ``fisher``, and
        ``anchor_delta`` for every LiquidLinear layer. It is much smaller
        than a full model checkpoint because the frozen base weights are
        excluded and can be loaded separately from any standard checkpoint.

        Args:
            path: Destination file path.
        """
        state: Dict = {}
        for name, m in self.named_modules():
            if isinstance(m, LiquidLinear):
                entry: Dict = {}
                if m.lora_rank > 0:
                    entry["lora_A"] = m.lora_A.cpu()
                    entry["lora_B"] = m.lora_B.cpu()
                    entry["lora_rank"] = m.lora_rank
                    entry["fisher_A"] = m.fisher_A.cpu()
                    entry["fisher_B"] = m.fisher_B.cpu()
                    entry["anchor_A"] = m.anchor_A.cpu()
                    entry["anchor_B"] = m.anchor_B.cpu()
                else:
                    entry["delta_w"] = m.delta_w.cpu()
                    entry["fisher"] = m.fisher.cpu()
                    entry["anchor_delta"] = m.anchor_delta.cpu()
                state[name] = entry
        torch.save(state, path)

    def load_plastic_state(self, path: str) -> bool:
        """Restore plastic weights from a file created by ``save_plastic_state``.

        Layers present in the file but absent from the current model are
        silently skipped, allowing partial loading across architectures with
        different depths.

        Args:
            path: Source file path.

        Returns:
            True if the file was found and loaded; False if the file does
            not exist.
        """
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        mods = {n: m for n, m in self.named_modules() if isinstance(m, LiquidLinear)}
        for name, ls in state.items():
            if name not in mods:
                continue
            m = mods[name]
            if "lora_A" in ls and m.lora_rank > 0:
                m.lora_A.copy_(ls["lora_A"].to(m.lora_A.device, m.lora_A.dtype))
                m.lora_B.copy_(ls["lora_B"].to(m.lora_B.device, m.lora_B.dtype))
                if "fisher_A" in ls:
                    m.fisher_A.copy_(ls["fisher_A"].float().cpu())
                    m.fisher_B.copy_(ls["fisher_B"].float().cpu())
                    m.anchor_A.copy_(ls["anchor_A"].float().cpu())
                    m.anchor_B.copy_(ls["anchor_B"].float().cpu())
            elif "delta_w" in ls and m.lora_rank == 0:
                dev = m.plastic_device
                m.delta_w.copy_(ls["delta_w"].to(dev))
                if "fisher" in ls:
                    m.fisher.copy_(ls["fisher"])
                    m.anchor_delta.copy_(ls["anchor_delta"])
        return True

    def plasticity_stats(self) -> Dict[str, Dict]:
        """Return per-layer plasticity statistics for every LiquidLinear layer."""
        out = {}
        for name, m in self.named_modules():
            if isinstance(m, LiquidLinear):
                dn = m.plastic_norm()
                bn = m.weight.data.norm().item()
                out[name] = {
                    "delta_norm":       round(dn, 6),
                    "base_norm":        round(bn, 6),
                    "fisher_norm":      round(m.fisher_norm(), 6),
                    "plasticity_ratio": round(dn / (bn + 1e-8), 6),
                }
        return out

    def param_count(self) -> Dict[str, int]:
        """Return static, plastic, and total parameter counts."""
        static = sum(p.numel() for p in self.parameters())
        plastic = sum(m.plastic_numel() for m in self.modules() if isinstance(m, LiquidLinear))
        return {"static": static, "plastic": plastic, "total": static + plastic}