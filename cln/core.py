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
"""

import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PLASTIC_ENABLED: bool = True


def set_plastic_mode(enabled: bool) -> None:
    """Enable or disable online Hebbian learning globally.

    Args:
        enabled: When False, all LiquidLinear layers skip their plastic
            update step regardless of the per-call ``plastic`` argument.
    """
    global _PLASTIC_ENABLED
    _PLASTIC_ENABLED = enabled


def get_plastic_mode() -> bool:
    """Return the current global plasticity flag."""
    return _PLASTIC_ENABLED


@contextmanager
def plastic_off():
    """Context manager that temporarily disables plasticity for all layers.

    Restores the previous flag on exit, even if an exception is raised.
    Useful for evaluation passes where weight updates are unwanted.
    """
    global _PLASTIC_ENABLED
    prev = _PLASTIC_ENABLED
    _PLASTIC_ENABLED = False
    try:
        yield
    finally:
        _PLASTIC_ENABLED = prev


class LiquidLinear(nn.Module):
    """Linear layer with plastic weights that evolve during inference.

    Maintains two weight components:

    - ``weight``: static base weights, trained offline via SGD/Adam and frozen.
    - ``delta_w``: plastic delta, evolved online via a Hebbian ODE Euler step.

    The effective computation is ``y = x @ (weight + delta_w).T + bias``.

    After each forward pass (when plasticity is active), ``delta_w`` is updated
    via one Euler step of the plasticity ODE entirely inside ``torch.no_grad()``,
    so no gradient tape is required.

    EWC (Elastic Weight Consolidation) prevents catastrophic forgetting: the
    Fisher information matrix Ω tracks weight importance, and the consolidation
    term pulls ``delta_w`` toward the last anchor weighted by that importance.
    Fisher and anchor tensors are always stored on CPU to save device memory.

    Attributes:
        weight: Static base weight parameter, shape [out_features, in_features].
        bias: Optional bias parameter, shape [out_features].
        delta_w: Plastic weight delta buffer, same shape as weight.
        fisher: Fisher information matrix buffer, kept on CPU.
        anchor_delta: Last consolidated delta snapshot, kept on CPU.
        plasticity_gate: Scalar buffer that scales the Hebbian update.
            Can be modulated externally to suppress or amplify learning.
    """

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
        accumulate_steps: int = 1,
        lora_rank: int = 0,
    ):
        """Initialize a LiquidLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: When True, a learnable bias is added to the output.
            tau_w: Weight time constant. Larger values slow the exponential
                decay of ``delta_w``, giving longer plastic memory.
            eta: Hebbian plasticity learning rate. Controls how strongly
                each forward pass modifies ``delta_w``.
            lambda_ewc: EWC consolidation strength. Zero disables the EWC
                term entirely.
            dt: Euler step size for the ODE integration.
            max_delta: Hard clamp applied to ``delta_w`` after each update
                to prevent runaway weight growth.
            solver_mode: ``"ml_fast"`` uses the standard Hebbian ODE path.
                ``"bio_ode"`` routes computation through a biologically
                inspired LiquidNeuronBank backend.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_w = tau_w
        self.eta = eta
        self.lambda_ewc = lambda_ewc
        self.dt = dt
        self.max_delta = max_delta
        self.solver_mode = solver_mode
        self.accumulate_steps = accumulate_steps
        self.lora_rank = lora_rank

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
        if lora_rank > 0:
            self.register_buffer("lora_A", torch.zeros(out_features, lora_rank))
            self.register_buffer("lora_B", torch.zeros(lora_rank, in_features))
            nn.init.kaiming_uniform_(self.lora_A)
            # Fisher and anchor are still full dense for now (Step 3)
            self.register_buffer("fisher",        torch.zeros(out_features, in_features, dtype=torch.float32, device="cpu"))
            self.register_buffer("anchor_delta",  torch.zeros(out_features, in_features, dtype=torch.float32, device="cpu"))
            self.register_buffer("plasticity_gate", torch.ones(1))
            if accumulate_steps > 1:
                self.register_buffer("_acc_dA", torch.zeros(out_features, lora_rank))
                self.register_buffer("_acc_dB", torch.zeros(lora_rank, in_features))
                self._acc_count: int = 0
        else:
            self.register_buffer("delta_w",       torch.zeros(out_features, in_features))
            self.register_buffer("fisher",        torch.zeros(out_features, in_features, dtype=torch.float32, device="cpu"))
            self.register_buffer("anchor_delta",  torch.zeros(out_features, in_features, dtype=torch.float32, device="cpu"))
            self.register_buffer("plasticity_gate", torch.ones(1))
    
            if accumulate_steps > 1:
                self.register_buffer("_hebbian_acc", torch.zeros(out_features, in_features))
                self._acc_count: int = 0

    @property
    def effective_weight(self) -> torch.Tensor:
        """Return the sum of base weights and the current plastic delta."""
        if self.lora_rank > 0:
            return self.weight + (self.lora_A.float() @ self.lora_B.float()).to(self.weight.dtype)
        return self.weight + self.delta_w

    @property
    def plastic_delta(self) -> torch.Tensor:
        if self.lora_rank > 0:
            return self.lora_A.float() @ self.lora_B.float()
        return self.delta_w.float()

    def liquid_step(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        """Apply one Euler step of the plastic weight ODE.

        Computes the Hebbian correlation between pre- and post-synaptic
        activations, normalizes it (Oja-style) to prevent weight explosion,
        optionally applies the EWC regularization term, and integrates with
        step size ``dt``. The result is clamped to ``[-max_delta, max_delta]``.

        Args:
            pre: Pre-synaptic activations, shape [..., in_features].
            post: Post-synaptic activations, shape [..., out_features].
        """
        with torch.no_grad():
            if pre.dim() == 2:
                pre  = pre.unsqueeze(1)
                post = post.unsqueeze(1)
            B, S, _ = pre.shape

            pre_f  = pre.float()
            post_f = post.float()

            pre_c  = pre_f  - pre_f.mean(dim=1, keepdim=True)
            post_c = post_f - post_f.mean(dim=1, keepdim=True)

            eff_dt = self.dt * self.accumulate_steps
            
            if self.lora_rank > 0:
                A = self.lora_A.float()
                B_mat = self.lora_B.float()
                
                pre_c_flat = pre_c.view(-1, self.in_features)
                post_c_flat = post_c.view(-1, self.out_features)
                
                dA_hebbian = torch.matmul(post_c_flat.t(), torch.matmul(pre_c_flat, B_mat.t())) / (B * S)
                dB_hebbian = torch.matmul(torch.matmul(post_c_flat, A).t(), pre_c_flat) / (B * S)
                
                joint_norm = math.sqrt(dA_hebbian.norm().item()**2 + dB_hebbian.norm().item()**2)
                if joint_norm > 1e-6:
                    dA_hebbian = dA_hebbian / joint_norm
                    dB_hebbian = dB_hebbian / joint_norm
                    
                dA_decay = A / (2 * self.tau_w)
                dB_decay = B_mat / (2 * self.tau_w)
                
                ewc_dev = self.lora_A.device
                if self.lambda_ewc > 0 and self.fisher.device == ewc_dev:
                    ewc = self.fisher.float() * (self.plastic_delta - self.anchor_delta.float())
                    # Project dense EWC penalty into A and B gradients
                    dA_ewc = self.lambda_ewc * torch.matmul(ewc, B_mat.t())
                    dB_ewc = self.lambda_ewc * torch.matmul(A.t(), ewc)
                else:
                    dA_ewc = 0
                    dB_ewc = 0
                    
                dA = self.dt * (self.eta * float(self.plasticity_gate) * dA_hebbian - dA_decay - dA_ewc)
                dB = self.dt * (self.eta * float(self.plasticity_gate) * dB_hebbian - dB_decay - dB_ewc)
                
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
                    ewc = self.fisher.float() * (
                        self.delta_w.float() - self.anchor_delta.float()
                    )
                else:
                    ewc = torch.zeros_like(hebbian)
    
                dw = self.delta_w.float()
                d_delta = (
                    -dw / self.tau_w
                    + self.eta * float(self.plasticity_gate) * hebbian
                    - self.lambda_ewc * ewc
                )
    
                self.delta_w.add_((eff_dt * d_delta).to(self.delta_w.dtype))
                self.delta_w.clamp_(-self.max_delta, self.max_delta)

    def consolidate(self, importance: Optional[torch.Tensor] = None) -> None:
        """Merge the current plastic state into the EWC anchor.

        Updates the Fisher information matrix with an exponential moving
        average and records the current ``delta_w`` as the new anchor.
        Both tensors are kept on CPU regardless of the device holding
        ``delta_w``, to avoid VRAM pressure on CUDA/MPS models.

        Args:
            importance: Optional per-weight importance scores with the same
                shape as ``delta_w``. When omitted, the absolute value of
                ``delta_w`` is used as a proxy for importance.
        """
        with torch.no_grad():
            delta_cpu = self.plastic_delta.cpu()
            imp = (
                importance.float().cpu()
                if importance is not None
                else delta_cpu.abs()
            )
            self.fisher = self.fisher.cpu().mul_(0.9).add_(0.1 * imp)
            self.anchor_delta = delta_cpu.clone()

    def reset_plasticity(self) -> None:
        """Zero out all plastic state, effectively wiping the layer's memory."""
        if self.lora_rank > 0:
            self.lora_A.zero_()
            nn.init.kaiming_uniform_(self.lora_A)
            self.lora_B.zero_()
        else:
            self.delta_w.zero_()
            
        self.fisher.zero_()
        self.anchor_delta.zero_()
        
        if getattr(self, "accumulate_steps", 1) > 1:
            if self.lora_rank > 0:
                self._acc_dA.zero_()
                self._acc_dB.zero_()
            else:
                self._hebbian_acc.zero_()
            self._acc_count = 0

    def forward(self, x: torch.Tensor, plastic: Optional[bool] = None) -> torch.Tensor:
        """Compute the linear transformation and optionally update plastic weights.

        When plasticity is active, applies a Hebbian update via ``liquid_step``
        after computing the output. The update runs outside the autograd graph.

        For the ``"bio_ode"`` solver, the first call synchronizes base weights
        into the biological backend before delegating the computation.

        Args:
            x: Input tensor of shape [..., in_features].
            plastic: Overrides the global ``_PLASTIC_ENABLED`` flag for this
                call. Pass ``True`` to force an update or ``False`` to skip it
                regardless of the global setting. ``None`` defers to the global flag.

        Returns:
            Output tensor of shape [..., out_features].
        """
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

        out = F.linear(x, self.effective_weight, self.bias)

        if should_learn:
            self.liquid_step(x.detach(), torch.tanh(out.detach()))

        return out


class LiquidTimeConstant(nn.Module):
    """Per-neuron adaptive time constant implementing the core LTC property.

    Computes a time constant τ(x) that varies with the input:

        τ(x) = τ_min + σ(W_τ · x) · (τ_max − τ_min)

    A small τ yields fast integration (high sensitivity to the current input).
    A large τ yields slow integration (smoothed state that persists over time).

    The network learns to be fast where precise reactions matter and slow
    where it should integrate information across longer contexts.

    Attributes:
        tau_min: Minimum time constant value.
        tau_range: Difference between maximum and minimum time constants.
        w_tau: Learned linear projection that gates the time constant.
    """

    def __init__(self, d_model: int, tau_min: float = 0.5, tau_max: float = 8.0):
        """Initialize the adaptive time constant module.

        Args:
            d_model: Input and output feature dimension.
            tau_min: Minimum time constant (fast neuron limit).
            tau_max: Maximum time constant (slow neuron limit).
        """
        super().__init__()
        self.tau_min = tau_min
        self.tau_range = tau_max - tau_min
        self.w_tau = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-neuron time constants from the input.

        Args:
            x: Input tensor of shape [..., d_model].

        Returns:
            Time constant tensor of shape [..., d_model], with values in
            [tau_min, tau_max].
        """
        return self.tau_min + torch.sigmoid(self.w_tau(x)) * self.tau_range