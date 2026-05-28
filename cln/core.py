"""
Core liquid components: LiquidLinear with plastic weight dynamics.

Mathematical foundation — the weight evolution ODE:

    dΔW/dt = -ΔW/τ_w + η·(post ⊗ pre) - λ·Ω·(ΔW - ΔW*)

Where:
    ΔW        : plastic weight delta (starts at zero, evolves during inference)
    τ_w       : weight time constant (decay / forgetting speed)
    η         : Hebbian plasticity rate
    post ⊗ pre: outer product (correlation between pre/post activations)
    λ         : EWC consolidation strength
    Ω         : Fisher information matrix (importance of each weight)
    ΔW*       : anchor delta after last consolidation

Effective weight: W_eff = W_base + ΔW
    W_base is trained via gradient descent and then frozen.
    ΔW is updated online — no gradients needed.

Improvements over the naive Hebbian rule:
  - Covariance: activations are centered over the sequence before correlation.
  - Exact per-token: torch.einsum over [B, S, D] instead of collapsing to mean first.
  - Liquid LoRA: optional low-rank factorization ΔW ≈ A·B (lora_rank > 0).
  - Episodic: accumulate Hebbian statistics for N steps before applying the ODE (accumulate_steps > 1).
  - Learnable τ and η stored as nn.Parameter, kept positive via softplus.
"""

import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PLASTIC_ENABLED: bool = True


def set_plastic_mode(enabled: bool) -> None:
    """Enable or disable online Hebbian learning globally."""
    global _PLASTIC_ENABLED
    _PLASTIC_ENABLED = enabled


def get_plastic_mode() -> bool:
    """Return the current global plasticity flag."""
    return _PLASTIC_ENABLED


@contextmanager
def plastic_off():
    """Context manager that temporarily disables plasticity for all layers."""
    global _PLASTIC_ENABLED
    prev = _PLASTIC_ENABLED
    _PLASTIC_ENABLED = False
    try:
        yield
    finally:
        _PLASTIC_ENABLED = prev


class LiquidLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tau_w: float = 20.0,
        eta: float = 5e-4,
        lambda_ewc: float = 0.05,
        dt: float = 0.1,
        max_delta: float = 0.3,
        solver_mode: str = "ml_fast",
        lora_rank: int = 0,
        accumulate_steps: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eta   = nn.Parameter(torch.tensor(eta,   dtype=torch.float32))
        self.tau_w = nn.Parameter(torch.tensor(tau_w, dtype=torch.float32))
        self.lambda_ewc = lambda_ewc
        self.dt = dt
        self.max_delta = max_delta
        self.solver_mode = solver_mode
        self.lora_rank = lora_rank
        self.accumulate_steps = accumulate_steps

        if self.solver_mode == "bio_ode":
            from .bio_core import LiquidNeuronBank
            self.bio_backend = LiquidNeuronBank(
                input_dim=in_features,
                hidden_dim=out_features,
                num_neurons=out_features,
                tau=tau_w,
                eta=eta,
                dt=dt,
            )
            self.bio_state = None
        else:
            self.bio_backend = None

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("fisher",        torch.zeros(out_features, in_features))
        self.register_buffer("anchor_delta",  torch.zeros(out_features, in_features))
        self.register_buffer("plasticity_gate", torch.ones(1))

        if lora_rank > 0:
            self.register_buffer("lora_A", torch.zeros(out_features, lora_rank))
            self.register_buffer("lora_B", torch.zeros(lora_rank, in_features))
            nn.init.kaiming_uniform_(self.lora_A)
            
            # We don't need full delta_w in memory during inference if we use LoRA
            # but we keep it for backward compatibility with older saving scripts if needed.
            # To save VRAM, we can just use lora_A @ lora_B on the fly.
        else:
            self.register_buffer("delta_w", torch.zeros(out_features, in_features))

        if accumulate_steps > 1:
            if lora_rank > 0:
                self.register_buffer("_acc_dA", torch.zeros(out_features, lora_rank))
                self.register_buffer("_acc_dB", torch.zeros(lora_rank, in_features))
            else:
                self.register_buffer("_hebbian_acc", torch.zeros(out_features, in_features))
            self._acc_count: int = 0

    @property
    def effective_weight(self) -> torch.Tensor:
        """Return the effective weight combining base weights and plastic delta."""
        if self.lora_rank > 0:
            return self.weight + (self.lora_A.float() @ self.lora_B.float()).to(self.weight.dtype)
        return self.weight + self.delta_w

    @property
    def plastic_delta(self) -> torch.Tensor:
        """Effective plastic delta as a dense tensor (works in both full and LoRA mode)."""
        if self.lora_rank > 0:
            return (self.lora_A.float() @ self.lora_B.float())
        return self.delta_w.float()

    @property
    def plastic_device(self) -> torch.device:
        """Device holding the plastic state."""
        if self.lora_rank > 0:
            return self.lora_A.device
        return self.delta_w.device

    def plastic_norm(self) -> float:
        """L2 norm of the plastic delta (works in both full and LoRA mode)."""
        return self.plastic_delta.norm().item()

    def plastic_numel(self) -> int:
        """Number of learnable plastic parameters (lora_A+lora_B or delta_w)."""
        if self.lora_rank > 0:
            return self.lora_A.numel() + self.lora_B.numel()
        return self.delta_w.numel()

    def liquid_step(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        with torch.no_grad():
            if pre.dim() == 2:
                pre  = pre.unsqueeze(1)
                post = post.unsqueeze(1)
            B, S, _ = pre.shape

            pre_f  = pre.float()
            post_f = post.float()

            pre_c  = pre_f  - pre_f.mean(dim=1, keepdim=True)
            post_c = post_f - post_f.mean(dim=1, keepdim=True)

            tau  = float(F.softplus(self.tau_w))
            eta  = float(F.softplus(self.eta))
            gate = float(self.plasticity_gate)
            
            # The effective integration step if accumulated
            eff_dt = self.dt * self.accumulate_steps

            if self.lora_rank > 0:
                A = self.lora_A.float()
                B_mat = self.lora_B.float()
                
                # Math optimization: avoid computing the [Out, In] hebbian matrix.
                # Hebbian term on B: dA_hebbian = (post_c.transpose(1, 2) @ (pre_c @ B.T))
                # Hebbian term on A: dB_hebbian = ((post_c @ A).transpose(1, 2) @ pre_c)
                
                pre_c_flat = pre_c.view(-1, self.in_features)
                post_c_flat = post_c.view(-1, self.out_features)
                
                # We do this as a batch matrix mult across the flattened B*S dimension
                # post_c_flat.T is [Out, B*S]. pre_c_flat is [B*S, In]
                # dA_hebbian = post_c_flat.T @ (pre_c_flat @ B_mat.T)  --> [Out, Rank]
                # dB_hebbian = (post_c_flat @ A).T @ pre_c_flat        --> [Rank, In]
                
                dA_hebbian = torch.matmul(post_c_flat.t(), torch.matmul(pre_c_flat, B_mat.t())) / (B * S)
                dB_hebbian = torch.matmul(torch.matmul(post_c_flat, A).t(), pre_c_flat) / (B * S)
                
                # Decay terms: A @ (B @ B.T) / tau
                B_Bt = torch.matmul(B_mat, B_mat.t())
                At_A = torch.matmul(A.t(), A)
                
                dA_decay = torch.matmul(A, B_Bt) / tau
                dB_decay = torch.matmul(At_A, B_mat) / tau
                
                dA_ewc = 0
                dB_ewc = 0
                if self.lambda_ewc > 0 and hasattr(self, 'fisher'):
                    if self.fisher.device == self.lora_A.device:
                        # EWC for LoRA is computationally heavy because it requires Fisher(O, I)
                        # We approximate or compute it exactly via broadcasting if memory permits.
                        # For now, we compute exact EWC penalty on A and B, which requires 
                        # forming the Delta W locally or running it on CPU.
                        # To keep it extremely fast and VRAM efficient, we omit EWC from LoRA by default
                        # unless explicitly computed on CPU offline during `consolidate`.
                        pass
                
                dA = eff_dt * (eta * gate * dA_hebbian - dA_decay - dA_ewc)
                dB = eff_dt * (eta * gate * dB_hebbian - dB_decay - dB_ewc)
                
                if self.accumulate_steps > 1:
                    self._acc_dA.add_(dA)
                    self._acc_dB.add_(dB)
                    self._acc_count += 1
                    if self._acc_count < self.accumulate_steps:
                        return
                    
                    self.lora_A.add_(self._acc_dA.to(self.lora_A.dtype)).clamp_(-self.max_delta, self.max_delta)
                    self.lora_B.add_(self._acc_dB.to(self.lora_B.dtype)).clamp_(-self.max_delta, self.max_delta)
                    self._acc_dA.zero_()
                    self._acc_dB.zero_()
                    self._acc_count = 0
                else:
                    self.lora_A.add_(dA.to(self.lora_A.dtype)).clamp_(-self.max_delta, self.max_delta)
                    self.lora_B.add_(dB.to(self.lora_B.dtype)).clamp_(-self.max_delta, self.max_delta)

            else:
                hebbian = torch.einsum('bso,bsi->oi', post_c, pre_c) / (B * S)
                h_norm = hebbian.norm()
                if h_norm > 1e-6:
                    hebbian = hebbian / h_norm
                
                if self.accumulate_steps > 1:
                    self._hebbian_acc.add_(hebbian)
                    self._acc_count += 1
                    if self._acc_count < self.accumulate_steps:
                        return
                    hebbian = self._hebbian_acc / self._acc_count
                    self._hebbian_acc.zero_()
                    self._acc_count = 0

                ewc_dev = self.delta_w.device
                if self.lambda_ewc > 0 and self.fisher.device == ewc_dev:
                    ewc = self.fisher.float() * (self.delta_w.float() - self.anchor_delta.float())
                else:
                    ewc = torch.zeros_like(hebbian)

                dw = self.delta_w.float()
                d_delta = -dw / tau + eta * gate * hebbian - self.lambda_ewc * ewc
                self.delta_w.add_((eff_dt * d_delta).to(self.delta_w.dtype))
                self.delta_w.clamp_(-self.max_delta, self.max_delta)

    def consolidate(self, importance: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            if self.lora_rank > 0:
                delta_cpu = (self.lora_A.float().cpu() @ self.lora_B.float().cpu())
            else:
                delta_cpu = self.delta_w.float().cpu()
                
            imp = (
                importance.float().cpu()
                if importance is not None
                else delta_cpu.abs()
            )
            self.fisher = self.fisher.cpu().mul_(0.9).add_(0.1 * imp)
            self.anchor_delta = delta_cpu.clone()

    def reset_plasticity(self) -> None:
        self.fisher.zero_()
        self.anchor_delta.zero_()
        if self.lora_rank > 0:
            self.lora_A.zero_()
            nn.init.kaiming_uniform_(self.lora_A)
            self.lora_B.zero_()
        else:
            self.delta_w.zero_()
            
        if self.accumulate_steps > 1:
            if self.lora_rank > 0:
                self._acc_dA.zero_()
                self._acc_dB.zero_()
            else:
                self._hebbian_acc.zero_()
            self._acc_count = 0

    def forward(self, x: torch.Tensor, plastic: Optional[bool] = None) -> torch.Tensor:
        should_learn = _PLASTIC_ENABLED if plastic is None else plastic

        if self.solver_mode == "bio_ode":
            if not hasattr(self, "_bio_synced"):
                with torch.no_grad():
                    self.bio_backend.liquid.W_init.copy_(self.weight)
                    if self.bias is not None:
                        self.bio_backend.liquid.bias.copy_(self.bias)
                self._bio_synced = True

            out, self.bio_state = self.bio_backend(
                x,
                state=self.bio_state,
                num_steps=2 if should_learn else 1,
            )
            return out

        if self.lora_rank > 0:
            out = F.linear(x, self.weight, self.bias)
            out_lora = F.linear(F.linear(x, self.lora_B.to(x.dtype)), self.lora_A.to(x.dtype))
            out = out + out_lora
        else:
            out = F.linear(x, self.effective_weight, self.bias)

        if should_learn:
            self.liquid_step(x.detach(), torch.tanh(out.detach()))

        return out


class LiquidTimeConstant(nn.Module):
    def __init__(self, d_model: int, tau_min: float = 0.5, tau_max: float = 8.0):
        super().__init__()
        self.tau_min = tau_min
        self.tau_range = tau_max - tau_min
        self.w_tau = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau_min + torch.sigmoid(self.w_tau(x)) * self.tau_range