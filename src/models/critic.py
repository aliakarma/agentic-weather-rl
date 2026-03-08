"""
src/models/critic.py
=====================
Centralised critic network for CTDE training.

Implements V_ψ(s_t) described in Section 4.3 of the paper:

  "The centralised critic is a three-layer MLP (hidden size 512)
   conditioned on the concatenated global state."

During training the critic has access to the full global state s_t
(dimension 24 in synthetic mode). At deployment each agent's
decentralised actor π_θi uses only its local observation o_{i,t};
the critic is discarded at inference time.

The critic outputs a scalar value estimate V(s_t) used to compute
the Generalised Advantage Estimate (GAE) for both the reward signal
(Â^R_t) and the per-agent constraint cost signals (Â^{C_i}_t).
Both advantage streams are needed for the cost-aware advantage
Â^L_t = Â^R_t - Σ_i λ_i Â^{C_i}_t (Eq. 7 of paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CriticNetwork(nn.Module):
    """
    Three-layer centralised MLP value network.

    Architecture
    ------------
    Input  : global state s_t  shape (batch, state_dim)
    Hidden : Linear(state_dim → 512) → LayerNorm → Tanh
             Linear(512 → 512)       → LayerNorm → Tanh
             Linear(512 → 512)       → LayerNorm → Tanh
    Output : Linear(512 → 1) → scalar V(s_t)

    The critic optionally also estimates a cost-value V^{C_i}(s_t) for
    each agent's constraint stream. This is activated by setting
    n_constraint_heads > 0, enabling cost-aware advantage computation
    without requiring a separate network per agent.

    Parameters
    ----------
    state_dim : int
        Dimension of the global state vector (default 24, from Table 1).
    hidden_size : int
        Width of each hidden layer (default 512, per paper Section 4.3).
    n_agents : int
        Number of agents; determines the number of constraint value heads.
    n_constraint_heads : int
        Number of per-agent cost-value heads V^{C_i}(s_t).
        Set to n_agents (3) to enable Lagrangian advantage computation.
        Set to 0 for a standard value-only critic (e.g., MAPPO baseline).
    """

    def __init__(
        self,
        state_dim: int = 24,
        hidden_size: int = 512,
        n_agents: int = 3,
        n_constraint_heads: int = 3,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.n_agents = n_agents
        self.n_constraint_heads = n_constraint_heads

        # ── Three-layer MLP backbone ─────────────────────────────────────
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        # ── Reward value head V(s_t) ─────────────────────────────────────
        self.value_head = nn.Linear(hidden_size, 1)

        # ── Per-agent constraint value heads V^{C_i}(s_t) ───────────────
        # One scalar output per agent for cost-aware advantage (Eq. 7)
        if n_constraint_heads > 0:
            self.cost_heads = nn.ModuleList([
                nn.Linear(hidden_size, 1)
                for _ in range(n_constraint_heads)
            ])
        else:
            self.cost_heads = nn.ModuleList()

        self._init_weights()

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the reward value and (optionally) constraint values.

        Parameters
        ----------
        state : torch.Tensor
            Global state s_t, shape (batch_size, state_dim) or (state_dim,).

        Returns
        -------
        value : torch.Tensor
            Reward value estimate V(s_t), shape (batch_size, 1).
        cost_values : torch.Tensor or None
            Per-agent cost values V^{C_i}(s_t), shape (batch_size, n_agents).
            None if n_constraint_heads == 0.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Shared backbone
        x = torch.tanh(self.ln1(self.fc1(state)))
        x = torch.tanh(self.ln2(self.fc2(x)))
        x = torch.tanh(self.ln3(self.fc3(x)))

        # Reward value head
        value = self.value_head(x)

        # Constraint value heads
        if self.cost_heads:
            cost_values = torch.cat(
                [head(x) for head in self.cost_heads], dim=-1
            )  # (batch, n_constraint_heads)
        else:
            cost_values = None

        return value, cost_values

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convenience method returning only the scalar reward value V(s_t).
        Used by GAE computation for the reward advantage stream Â^R_t.

        Returns
        -------
        value : torch.Tensor  shape (batch_size, 1)
        """
        value, _ = self.forward(state)
        return value

    def get_cost_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return per-agent cost value estimates V^{C_i}(s_t).
        Used by GAE computation for constraint advantage streams Â^{C_i}_t.

        Returns
        -------
        cost_values : torch.Tensor  shape (batch_size, n_agents)

        Raises
        ------
        RuntimeError
            If the critic was initialised with n_constraint_heads == 0.
        """
        if not self.cost_heads:
            raise RuntimeError(
                "This CriticNetwork has no constraint heads "
                "(n_constraint_heads=0). "
                "Re-initialise with n_constraint_heads=n_agents to enable "
                "cost-value estimation for Lagrangian advantage computation."
            )
        _, cost_values = self.forward(state)
        return cost_values  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # Bellman residual loss — used by CTDE trainer
    # -----------------------------------------------------------------------

    def value_loss(
        self,
        state: torch.Tensor,
        returns: torch.Tensor,
        clip_range: float = 0.2,
        old_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Clipped value function loss for the reward stream.
        Mirrors the PPO value clipping used in the actor update.

        Parameters
        ----------
        state : torch.Tensor       shape (batch, state_dim)
        returns : torch.Tensor     shape (batch, 1)   — TD(λ) targets
        clip_range : float         PPO value clipping ε (default 0.2)
        old_values : torch.Tensor | None
            Previous value estimates for clipping. If None, no clipping.

        Returns
        -------
        loss : torch.Tensor   scalar
        """
        current_values, _ = self.forward(state)

        if old_values is not None:
            # Clipped value loss (standard PPO critic update)
            values_clipped = old_values + torch.clamp(
                current_values - old_values, -clip_range, clip_range
            )
            loss_unclipped = (current_values - returns) ** 2
            loss_clipped = (values_clipped - returns) ** 2
            loss = 0.5 * torch.mean(torch.max(loss_unclipped, loss_clipped))
        else:
            loss = 0.5 * torch.mean((current_values - returns) ** 2)

        return loss

    def cost_value_loss(
        self,
        state: torch.Tensor,
        cost_returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss for all constraint value heads simultaneously.

        Parameters
        ----------
        state : torch.Tensor        shape (batch, state_dim)
        cost_returns : torch.Tensor shape (batch, n_agents) — cost TD targets

        Returns
        -------
        loss : torch.Tensor  scalar
        """
        cost_values = self.get_cost_values(state)
        return 0.5 * torch.mean((cost_values - cost_returns) ** 2)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Orthogonal initialisation for hidden layers (gain=√2).
        Output heads initialised near-zero for stable early training.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=(2 ** 0.5))
            nn.init.zeros_(layer.bias)

        for head in [self.value_head, *self.cost_heads]:
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CriticNetwork("
            f"state_dim={self.state_dim}, "
            f"hidden={self.hidden_size}, "
            f"n_constraint_heads={self.n_constraint_heads}, "
            f"params={self.count_parameters():,})"
        )
