"""
src/models/actor.py
====================
Decentralised actor network for each MARL agent.

Implements the policy π_θi(a_i | o_{i,t}) described in Section 4.3
of the paper:

  "All three agents share the same actor network architecture
   (two-layer MLP, hidden size 256) but maintain independent
   parameters θ_i."

Each agent owns its own ActorNetwork instance with separate parameters.
At execution time the actor receives only its local observation o_{i,t}
(dimension 12) and outputs a categorical action distribution over the
four discrete actions {0, 1, 2, 3}.

Action encoding (Section 3.3 of paper):
    0 — no action
    1 — issue warning
    2 — deploy resources
    3 — initiate evacuation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    Two-layer MLP policy network for a single decentralised agent.

    Architecture
    ------------
    Input  : local observation o_{i,t}  shape (batch, obs_dim)
    Hidden : Linear(obs_dim → hidden) → LayerNorm → Tanh
             Linear(hidden   → hidden) → LayerNorm → Tanh
    Output : Linear(hidden → n_actions) → logits (unnormalised)

    LayerNorm is used instead of BatchNorm because episodes are collected
    with variable batch sizes and single-sample inference is common at
    deployment time.

    Parameters
    ----------
    obs_dim : int
        Dimension of the agent's local observation vector (default 12).
    n_actions : int
        Number of discrete actions (default 4).
    hidden_size : int
        Width of each hidden layer (default 256, per paper Section 4.3).
    agent_id : int | None
        Optional identifier for logging; does not affect computation.
    """

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        hidden_size: int = 256,
        agent_id: int | None = None,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.agent_id = agent_id

        # ── Two-layer MLP ────────────────────────────────────────────────
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Output head: logits over action space
        self.output = nn.Linear(hidden_size, n_actions)

        self._init_weights()

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits from the local observation.

        Parameters
        ----------
        obs : torch.Tensor
            Shape (batch_size, obs_dim) or (obs_dim,) for single inference.

        Returns
        -------
        logits : torch.Tensor
            Shape (batch_size, n_actions) — unnormalised log-probabilities.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x = torch.tanh(self.ln1(self.fc1(obs)))
        x = torch.tanh(self.ln2(self.fc2(x)))
        logits = self.output(x)
        return logits

    # -----------------------------------------------------------------------
    # Distribution helpers — used by PPOTrainer
    # -----------------------------------------------------------------------

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        """
        Return a Categorical distribution over actions.

        Parameters
        ----------
        obs : torch.Tensor
            Local observation tensor.

        Returns
        -------
        dist : torch.distributions.Categorical
            Action distribution π_θi(· | o_{i,t}).
        """
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or greedily select) an action and return auxiliary tensors
        needed for PPO updates.

        Parameters
        ----------
        obs : torch.Tensor
            Local observation tensor.
        deterministic : bool
            If True, select the argmax action (used at evaluation time).

        Returns
        -------
        action : torch.Tensor  shape (batch,)
            Sampled or greedy action index.
        log_prob : torch.Tensor  shape (batch,)
            Log-probability of the selected action log π_θi(a | o).
        entropy : torch.Tensor  shape (batch,)
            Per-sample entropy of the distribution H[π_θi(· | o)].
        """
        dist = self.get_distribution(obs)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probabilities and entropy for a batch of (obs, action) pairs.
        Called during the PPO update step (Eq. 8 of the paper).

        Parameters
        ----------
        obs : torch.Tensor       shape (batch, obs_dim)
        actions : torch.Tensor   shape (batch,)

        Returns
        -------
        log_probs : torch.Tensor  shape (batch,)
        entropy   : torch.Tensor  shape (batch,)
        """
        dist = self.get_distribution(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Orthogonal initialisation for hidden layers (gain=√2) and small
        uniform initialisation for the output layer.
        Standard practice for PPO actor networks.
        """
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=(2 ** 0.5))
            nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ActorNetwork("
            f"agent_id={self.agent_id}, "
            f"obs_dim={self.obs_dim}, "
            f"hidden={self.hidden_size}, "
            f"n_actions={self.n_actions}, "
            f"params={self.count_parameters():,})"
        )
