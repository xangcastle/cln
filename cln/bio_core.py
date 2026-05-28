"""
Biologically-inspired liquid neuron core.

Implements a neuron whose hidden state, synaptic weights, and mnemonic traces
all evolve simultaneously according to coupled ODEs during inference, following
the master equations of Continuous Liquid Networks (CLN).

The three state variables and their governing equations are:

    τ · dW/dt = -W + Φ(x, h, M) + η · P(W, x, h, M)   (weight dynamics)
    dh/dt     = σ(W(t) · x + b) - λ · h                 (activation dynamics)
    dM/dt     = α · (x ⊗ h) · C - β · M · D             (mnemonic dynamics)

Integration is performed with either explicit Euler or 4th-order Runge-Kutta
at each forward pass. No gradient-based training loop is needed — learning
emerges from the ODE evolution itself.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LiquidState:
    """Complete state of a liquid neuron at a single time step.

    All three tensors must be kept together and passed between time steps
    to maintain continuity of the ODE integration.

    Attributes:
        h: Neuronal hidden activation, shape [batch, hidden_dim].
        W: Liquid synaptic weights evolving in continuous time,
            shape [hidden_dim, input_dim].
        M: Mnemonic traces encoding past co-activation patterns,
            shape [hidden_dim, input_dim].
    """

    h: torch.Tensor
    W: torch.Tensor
    M: torch.Tensor

    def clone(self) -> "LiquidState":
        """Return a deep copy of this state with all tensors cloned."""
        return LiquidState(
            h=self.h.clone(),
            W=self.W.clone(),
            M=self.M.clone(),
        )


class LiquidNeuron(nn.Module):
    """A single neuron whose weights, activation, and memory all evolve over time.

    Unlike a standard neuron with fixed weights, this neuron maintains a
    ``LiquidState`` (h, W, M) that is updated at every forward call via
    numerical integration of the coupled ODE system:

        τ · dW/dt = -W + Φ(x, h, M) + η · P(W, x, h, M)
        dh/dt     = σ(W(t) · x + b) - λ · h
        dM/dt     = α · (x ⊗ h) · C - β · M · D

    - Φ (``phi``) is a bounded contextual modulation signal.
    - P (``plasticity_operator``) is a Hebbian term with mnemonic protection.
    - C gates how strongly a correlation is written into M (novelty signal).
    - D gates how aggressively M is decayed (distance-from-W signal).

    Attributes:
        input_dim: Dimension of the input vector.
        hidden_dim: Dimension of the hidden activation h.
        tau: Viscous time constant controlling how fast W evolves.
        eta: Synaptic plasticity intensity scaling P.
        alpha: Mnemonic trace formation rate.
        beta: Mnemonic decay / consolidation rate.
        gamma: Protection factor penalizing changes to consolidated weights.
        lambda_decay: Neuronal state decay rate in the h ODE.
        dt: Numerical integration step size.
        solver: Integration method — ``"rk4"`` (default) or ``"euler"``.
        bias: Learnable bias applied to the pre-activation, shape [hidden_dim].
        omega_h: Learnable scale for the h contribution to Φ.
        omega_x: Learnable scale for the x contribution to Φ.
        omega_m: Learnable scale for the M contribution to Φ.
        W_init: Buffer holding the initial weight matrix used when no prior
            state is provided, shape [hidden_dim, input_dim].
        M_init: Buffer holding the initial mnemonic matrix (zeros),
            shape [hidden_dim, input_dim].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau: float = 0.1,
        eta: float = 0.01,
        alpha: float = 0.5,
        beta: float = 0.05,
        gamma: float = 0.3,
        lambda_decay: float = 0.2,
        dt: float = 0.05,
        solver: str = "rk4",
    ):
        """Initialize a LiquidNeuron.

        Args:
            input_dim: Dimension of the input vector x.
            hidden_dim: Dimension of the hidden activation h and the rows of W.
            tau: Viscous time constant. Smaller values make W evolve faster.
            eta: Plasticity intensity. Scales the Hebbian update P in the W ODE.
            alpha: Rate at which new co-activations are written into M.
            beta: Rate at which M decays toward W (consolidation / forgetting).
            gamma: Mnemonic protection strength. Higher values penalize changes
                to weights that have strong associated memory traces.
            lambda_decay: Leak rate in the h ODE. Higher values make h forget
                its previous activation faster.
            dt: Step size for numerical integration. Smaller values increase
                accuracy at the cost of more compute per forward call.
            solver: ``"rk4"`` uses 4th-order Runge-Kutta (more accurate).
                ``"euler"`` uses explicit Euler (faster, less accurate).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        self.dt = dt
        self.solver = solver

        self.bias    = nn.Parameter(torch.zeros(hidden_dim))
        self.omega_h = nn.Parameter(torch.randn(hidden_dim, 1) * 0.01)
        self.omega_x = nn.Parameter(torch.randn(1, input_dim) * 0.01)
        self.omega_m = nn.Parameter(torch.ones(1) * 0.1)

        self.register_buffer("W_init", torch.randn(hidden_dim, input_dim) * 0.01)
        self.register_buffer("M_init", torch.zeros(hidden_dim, input_dim))

    def phi(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the contextual modulation signal Φ(x, h, M).

        Produces a bounded tensor that drives the weight evolution ODE.
        Contributions from the input, hidden state, and (normalized) mnemonic
        traces are summed and passed through tanh:

            Φ = tanh(ω_x · mean(x) + ω_h · mean(h)ᵀ + ω_m · M / (1 + |M|))

        Args:
            x: External input, shape [batch, input_dim].
            h: Hidden activation, shape [batch, hidden_dim].
            M: Mnemonic traces, shape [hidden_dim, input_dim].

        Returns:
            Contextual modulation tensor of shape [hidden_dim, input_dim],
            with values in (-1, 1).
        """
        x_contrib = (self.omega_x * x.mean(dim=0, keepdim=True)).expand(self.hidden_dim, -1)
        h_contrib = (self.omega_h * h.mean(dim=0, keepdim=True).t()).expand(-1, self.input_dim)
        M_contrib = self.omega_m * (M / (1.0 + torch.abs(M) + 1e-8))
        return torch.tanh(x_contrib + h_contrib + M_contrib)

    def plasticity_operator(
        self,
        W: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Hebbian anti-forgetting plasticity update P(W, x, h, M).

        Combines a Hebbian correlation term with mnemonic protection and soft
        L2 regularization:

            P = (hᵀ · x) / batch - γ · W · ‖M‖² - 0.001 · W

        The mnemonic protection term penalizes changes to weights whose
        associated memory traces are strong, preserving consolidated knowledge.

        Args:
            W: Current liquid weights, shape [hidden_dim, input_dim].
            x: External input, shape [batch, input_dim].
            h: Hidden activation, shape [batch, hidden_dim].
            M: Mnemonic traces, shape [hidden_dim, input_dim].

        Returns:
            Plasticity gradient tensor of shape [hidden_dim, input_dim].
        """
        hebbian      = torch.matmul(h.t(), x) / x.shape[0]
        protection   = self.gamma * W * torch.sum(M ** 2, dim=1, keepdim=True)
        regularization = 0.001 * W
        return hebbian - protection - regularization

    def derivatives(
        self,
        state: LiquidState,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the coupled ODE right-hand sides at the given state.

        Computes dh/dt, dW/dt, and dM/dt simultaneously from the current state
        and the external input. This is the core function called by both the
        Euler and RK4 integrators.

        Args:
            state: Current liquid state (h, W, M).
            x: External input, shape [batch, input_dim].

        Returns:
            Three-tuple ``(dh_dt, dW_dt, dM_dt)``, each with the same shape as
            the corresponding state tensor.
        """
        h, W, M = state.h, state.W, state.M

        dh_dt = torch.tanh(torch.matmul(x, W.t()) + self.bias) - self.lambda_decay * h

        phi_term  = self.phi(x, h, M)
        plasticity = self.plasticity_operator(W, x, h, M)
        dW_dt = (-W + phi_term + self.eta * plasticity) / self.tau

        correlation = torch.matmul(h.t(), x) / x.shape[0]
        C    = torch.sigmoid(correlation * 10)
        D    = torch.sigmoid(torch.abs(W - M) * 5)
        dM_dt = self.alpha * correlation * C - self.beta * M * D

        return dh_dt, dW_dt, dM_dt

    def step_euler(self, state: LiquidState, x: torch.Tensor) -> LiquidState:
        """Apply one explicit Euler integration step.

        Args:
            state: Current liquid state.
            x: External input, shape [batch, input_dim].

        Returns:
            Updated ``LiquidState`` after one step of size ``dt``.
        """
        dh, dW, dM = self.derivatives(state, x)
        return LiquidState(
            h=state.h + self.dt * dh,
            W=state.W + self.dt * dW,
            M=state.M + self.dt * dM,
        )

    def step_rk4(self, state: LiquidState, x: torch.Tensor) -> LiquidState:
        """Apply one 4th-order Runge-Kutta integration step.

        Evaluates derivatives at four intermediate points and combines them
        with the classical RK4 weights:

            y_{n+1} = y_n + (dt / 6) · (k1 + 2·k2 + 2·k3 + k4)

        More accurate than Euler for the same ``dt``, at the cost of four
        derivative evaluations per step instead of one.

        Args:
            state: Current liquid state.
            x: External input, shape [batch, input_dim].

        Returns:
            Updated ``LiquidState`` after one RK4 step of size ``dt``.
        """
        dh1, dW1, dM1 = self.derivatives(state, x)

        s2 = LiquidState(
            h=state.h + 0.5 * self.dt * dh1,
            W=state.W + 0.5 * self.dt * dW1,
            M=state.M + 0.5 * self.dt * dM1,
        )
        dh2, dW2, dM2 = self.derivatives(s2, x)

        s3 = LiquidState(
            h=state.h + 0.5 * self.dt * dh2,
            W=state.W + 0.5 * self.dt * dW2,
            M=state.M + 0.5 * self.dt * dM2,
        )
        dh3, dW3, dM3 = self.derivatives(s3, x)

        s4 = LiquidState(
            h=state.h + self.dt * dh3,
            W=state.W + self.dt * dW3,
            M=state.M + self.dt * dM3,
        )
        dh4, dW4, dM4 = self.derivatives(s4, x)

        return LiquidState(
            h=state.h + self.dt / 6.0 * (dh1 + 2 * dh2 + 2 * dh3 + dh4),
            W=state.W + self.dt / 6.0 * (dW1 + 2 * dW2 + 2 * dW3 + dW4),
            M=state.M + self.dt / 6.0 * (dM1 + 2 * dM2 + 2 * dM3 + dM4),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LiquidState] = None,
        num_steps: int = 1,
    ) -> Tuple[torch.Tensor, LiquidState]:
        """Run the liquid neuron for one or more integration steps.

        When no prior state is provided, h is initialized to zeros and W/M are
        cloned from their registered initial-value buffers.

        Args:
            x: Input tensor of shape [batch, input_dim].
            state: Prior ``LiquidState`` to continue from. Pass ``None`` to
                start a fresh integration from the initial conditions.
            num_steps: Number of ODE steps to integrate before returning.
                Higher values give the dynamics more time to evolve per call.

        Returns:
            Two-tuple ``(h, state)`` where ``h`` is the hidden activation after
            integration (shape [batch, hidden_dim]) and ``state`` is the updated
            ``LiquidState`` to pass to the next call.
        """
        batch_size = x.shape[0]
        device = x.device

        if state is None:
            state = LiquidState(
                h=torch.zeros(batch_size, self.hidden_dim, device=device),
                W=self.W_init.clone().to(device),
                M=self.M_init.clone().to(device),
            )

        step_fn = self.step_rk4 if self.solver == "rk4" else self.step_euler
        for _ in range(num_steps):
            state = step_fn(state, x)

        return state.h, state


class LiquidNeuronBank(nn.Module):
    """A bank of liquid neurons that processes vectors or full sequences.

    Acts as a drop-in replacement for a linear layer, but each output
    neuron is backed by a ``LiquidNeuron`` with its own evolving weight
    dynamics and mnemonic traces.

    When the input is a 3-D sequence tensor, the bank unrolls over the time
    dimension, threading the liquid state from one time step to the next.

    Attributes:
        input_dim: Dimension of each input vector.
        hidden_dim: Dimension of the output (and hidden) activation.
        num_neurons: Number of logical neurons (currently shared as one
            ``LiquidNeuron`` with ``hidden_dim`` outputs).
        liquid: The underlying ``LiquidNeuron`` instance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_neurons: int,
        **liquid_kwargs,
    ):
        """Initialize a LiquidNeuronBank.

        Args:
            input_dim: Dimension of the input vector fed to each neuron.
            hidden_dim: Output dimension of the liquid neuron.
            num_neurons: Logical neuron count. Currently all neurons are
                implemented as a single ``LiquidNeuron`` with ``hidden_dim``
                outputs.
            **liquid_kwargs: Additional keyword arguments forwarded to
                ``LiquidNeuron.__init__`` (e.g. ``tau``, ``eta``, ``solver``).
        """
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.num_neurons = num_neurons

        self.liquid = LiquidNeuron(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **liquid_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LiquidState] = None,
        num_steps: int = 1,
    ) -> Tuple[torch.Tensor, LiquidState]:
        """Process a vector or sequence through the liquid neuron bank.

        For 2-D inputs the call is delegated directly to ``LiquidNeuron``.
        For 3-D inputs the bank iterates over the sequence dimension,
        threading the liquid state across time steps.

        Args:
            x: Input tensor of shape [batch, input_dim] or
                [batch, seq_len, input_dim].
            state: Prior ``LiquidState`` to continue from. Pass ``None`` to
                start from the neuron's initial conditions.
            num_steps: Number of ODE integration steps per time position.

        Returns:
            Two-tuple ``(output, state)`` where ``output`` has shape
            [batch, hidden_dim] for 2-D input or [batch, seq_len, hidden_dim]
            for 3-D input, and ``state`` is the final ``LiquidState``.
        """
        if x.dim() == 3:
            outputs = []
            for t in range(x.shape[1]):
                out, state = self.liquid(x[:, t, :], state, num_steps)
                outputs.append(out.unsqueeze(1))
            return torch.cat(outputs, dim=1), state

        return self.liquid(x, state, num_steps)