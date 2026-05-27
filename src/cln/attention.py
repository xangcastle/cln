"""
Liquid Multi-Head Attention.

All four projection matrices (Q, K, V, O) are LiquidLinear layers that
accumulate Hebbian updates on every inference pass. Token pairs with high
attention scores generate stronger plasticity signals, so the network
naturally reinforces the relationships it finds important.

The residual gate uses the liquid time constant τ:

    x ← x + (1/τ) · Attention(x)

This is an Euler discretization of the continuous ODE:

    dx/dt = -x/τ + Attention(x)

which matches the dynamics of biological liquid neural networks
(Liquid Time-Constant networks, Hasani et al. 2021).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import LiquidLinear


class LiquidMultiHeadAttention(nn.Module):
    """Multi-head self-attention where all four projections (Q, K, V, O) are plastic.

    Each forward pass triggers Hebbian weight updates on every projection layer,
    so the attention mechanism evolves continuously during inference without a
    separate training phase.

    Attributes:
        d_model: Total embedding dimension.
        n_heads: Number of attention heads.
        d_head: Per-head dimension (d_model // n_heads).
        scale: Dot-product scaling factor (sqrt of d_head).
        last_attn_weights: Attention weight tensor from the most recent forward
            pass, detached from the computation graph. Useful for visualization
            and external analysis.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        liquid_kwargs: Optional[dict] = None,
    ):
        """Initialize liquid multi-head attention.

        Args:
            d_model: Total embedding dimension. Must be divisible by n_heads.
            n_heads: Number of parallel attention heads.
            dropout: Dropout probability applied to attention weights and the
                residual output.
            liquid_kwargs: Extra keyword arguments forwarded to every
                LiquidLinear projection (e.g. learning rate, plasticity rule).
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        lkw = liquid_kwargs or {}
        self.q_proj   = LiquidLinear(d_model, d_model, **lkw)
        self.k_proj   = LiquidLinear(d_model, d_model, **lkw)
        self.v_proj   = LiquidLinear(d_model, d_model, **lkw)
        self.out_proj = LiquidLinear(d_model, d_model, **lkw)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape from [B, T, D] to [B, H, T, D/H]."""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape from [B, H, T, D/H] back to [B, T, D]."""
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        plastic: bool = True,
    ) -> torch.Tensor:
        """Run liquid multi-head attention over an input sequence.

        Args:
            x: Input tensor of shape [B, T, D].
            mask: Boolean causal mask of shape [1, 1, T, T].
                Positions with value 0 are masked out (set to -inf before softmax).
            plastic: When True, each projection layer applies its Hebbian update
                rule after the forward pass.

        Returns:
            Output tensor of shape [B, T, D] with the same dtype and device as x.
        """
        Q = self._split_heads(self.q_proj(x, plastic=plastic))
        K = self._split_heads(self.k_proj(x, plastic=plastic))
        V = self._split_heads(self.v_proj(x, plastic=plastic))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach()
        attn = self.attn_drop(attn)

        context = self._merge_heads(torch.matmul(attn, V))
        return self.resid_drop(self.out_proj(context, plastic=plastic))

    def consolidate(self, importance: Optional[torch.Tensor] = None) -> None:
        """Merge plastic deltas into the base weights of all projections.

        Args:
            importance: Optional per-parameter importance scores passed through
                to each LiquidLinear layer to weight the consolidation.
        """
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.consolidate(importance)