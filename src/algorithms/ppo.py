"""
src/algorithms/ppo.py
======================
Proximal Policy Optimisation (PPO) trainer for a single agent.

Implements the clipped surrogate objective from Eq. 8 of the paper:

    L^CLIP(θ_i) = E[ min(ρ_t(θ_i) * Â^L_t,
                         clip(ρ_t(θ_i), 1-ε, 1+ε) * Â^L_t) ]

where ρ_t(θ_i) = π_θi(a_{i,t} | o_{i,t}) / π_θi_old(a_{i,t} | o_{i,t})
is the importance ratio and Â^L_t is the cost-aware advantage (Eq. 7).

Also implements:
  - Generalised Advantage Estimation (GAE) for both reward and cost streams
  - Entropy regularisation
  - Mini-batch gradient updates with gradient clipping

PPOTrainer is used by LagrangianCTDE as the per-agent optimisation primitive.
It is also used standalone by the IPPO and MAPPO baselines.

Paper hyperparameters (Table 3):
    lr            = 3e-4
    clip_epsilon  = 0.2
    gae_lambda    = 0.95
    gamma         = 0.99
    entropy_coef  = 0.01
    mini_batches  = 4
    batch_size    = 64
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """
    Stores a single episode trajectory for one agent.

    All lists are appended to during environment interaction and converted
    to tensors before the PPO update.

    Fields
    ------
    observations   : list of local observations o_{i,t}  shape (obs_dim,)
    actions        : list of integer actions a_{i,t}
    log_probs      : list of log π_θi(a_{i,t} | o_{i,t})  scalar
    rewards        : list of team reward r_t               scalar
    values         : list of V(s_t) from centralised critic scalar
    dones          : list of episode-end flags             bool
    cost_rewards   : list of constraint cost C_{i,t}       scalar
    cost_values    : list of V^{C_i}(s_t)                  scalar
    global_states  : list of global state s_t  shape (state_dim,)
      (used by centralised critic; None in decentralised mode)
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    cost_rewards: List[float] = field(default_factory=list)
    cost_values: List[float] = field(default_factory=list)
    global_states: List[Optional[np.ndarray]] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        cost_reward: float = 0.0,
        cost_value: float = 0.0,
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.cost_rewards.append(cost_reward)
        self.cost_values.append(cost_value)
        self.global_states.append(global_state)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)

    def to_tensors(
        self, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Convert all lists to tensors on the given device."""
        obs_t = torch.tensor(
            np.array(self.observations), dtype=torch.float32, device=device
        )
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=device)
        log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        values_t = torch.tensor(self.values, dtype=torch.float32, device=device)
        dones_t = torch.tensor(self.dones, dtype=torch.float32, device=device)
        cost_rewards_t = torch.tensor(
            self.cost_rewards, dtype=torch.float32, device=device
        )
        cost_values_t = torch.tensor(
            self.cost_values, dtype=torch.float32, device=device
        )

        result: Dict[str, torch.Tensor] = {
            "observations":  obs_t,
            "actions":       actions_t,
            "log_probs_old": log_probs_t,
            "rewards":       rewards_t,
            "values":        values_t,
            "dones":         dones_t,
            "cost_rewards":  cost_rewards_t,
            "cost_values":   cost_values_t,
        }

        # Include global states if present
        if any(s is not None for s in self.global_states):
            result["global_states"] = torch.tensor(
                np.array(self.global_states), dtype=torch.float32, device=device
            )

        return result


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation (GAE).

    Computes advantages Â_t and TD(λ) returns G_t for a single trajectory.

    Implements the recursion:
        δ_t   = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        Â_t   = δ_t + γλ * (1 - done_t) * Â_{t+1}
        G_t   = Â_t + V(s_t)

    Parameters
    ----------
    rewards : Tensor (T,)
    values  : Tensor (T,)  — V(s_t) estimates from the critic
    dones   : Tensor (T,)  — 1.0 if episode ended at step t
    next_value : float     — V(s_{T}) bootstrap value
    gamma      : float     — discount factor (default 0.99)
    gae_lambda : float     — GAE λ parameter (default 0.95)

    Returns
    -------
    advantages : Tensor (T,)
    returns    : Tensor (T,)   — TD(λ) targets for critic regression
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1].item()
        mask = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * next_val * mask - values[t].item()
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Single-agent PPO trainer.

    Used by LagrangianCTDE as the per-agent optimisation primitive.
    Also used directly by IPPO and MAPPO baselines.

    Parameters
    ----------
    actor : ActorNetwork
        The agent's decentralised policy π_θi.
    critic : CriticNetwork | None
        Critic for advantage estimation.
        If None, a separate shared critic must supply value estimates
        (as in CTDE mode where LagrangianCTDE owns the critic).
    agent_id : int
        Agent identifier (1, 2, or 3). Used for logging only.
    lr : float
        Learning rate for actor (and critic if owned). Default 3e-4.
    clip_epsilon : float
        PPO clipping threshold ε. Default 0.2.
    gamma : float
        Discount factor γ. Default 0.99.
    gae_lambda : float
        GAE λ parameter. Default 0.95.
    entropy_coef : float
        Entropy bonus coefficient. Default 0.01.
    max_grad_norm : float
        Gradient clipping norm. Default 0.5.
    mini_batches : int
        Number of mini-batch gradient steps per update. Default 4.
    device : str
        Compute device. Default 'cpu'.
    """

    def __init__(
        self,
        actor: ActorNetwork,
        critic: Optional[CriticNetwork] = None,
        agent_id: int = 1,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        mini_batches: int = 4,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.agent_id = agent_id
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.mini_batches = mini_batches
        self.device = torch.device(device)

        # Actor optimiser (always owned by PPOTrainer)
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=lr)

        # Critic optimiser (only when PPOTrainer owns the critic)
        self.critic_optimizer: Optional[optim.Adam] = None
        if critic is not None:
            self.critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

        # Rollout buffer for this agent
        self.buffer = RolloutBuffer()

        # Training statistics
        self._update_count: int = 0

    # -----------------------------------------------------------------------
    # Buffer interface — called during environment rollout
    # -----------------------------------------------------------------------

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        cost_reward: float = 0.0,
        cost_value: float = 0.0,
        global_state: Optional[np.ndarray] = None,
    ) -> None:
        """Append one transition to this agent's rollout buffer."""
        self.buffer.add(
            obs=obs, action=action, log_prob=log_prob, reward=reward,
            value=value, done=done, cost_reward=cost_reward,
            cost_value=cost_value, global_state=global_state,
        )

    # -----------------------------------------------------------------------
    # PPO update — called once per episode (or per N steps)
    # -----------------------------------------------------------------------

    def update(
        self,
        next_value: float = 0.0,
        next_cost_value: float = 0.0,
        lagrange_lambda: float = 0.0,
    ) -> Dict[str, float]:
        """
        Run the PPO update using stored rollout data.

        Implements Steps 3–7 of Algorithm 1 (paper):
          3. Compute reward advantages Â^R_t via GAE
          4. Compute cost advantages Â^{C_i}_t via GAE
          5. Compute cost-aware advantage Â^L_t = Â^R_t - λ_i * Â^{C_i}_t
          6. Update actor θ_i by maximising L^CLIP using Â^L_t
          7. Update critic ψ to minimise Bellman residual

        Parameters
        ----------
        next_value : float
            Bootstrap V(s_{T+1}) for GAE. 0.0 at terminal steps.
        next_cost_value : float
            Bootstrap V^{C_i}(s_{T+1}) for cost GAE.
        lagrange_lambda : float
            Current dual variable λ_i for this agent's constraint.
            Supplied by LagrangianCTDE; 0.0 for unconstrained baselines.

        Returns
        -------
        stats : dict
            Training statistics for logging:
            policy_loss, value_loss, entropy_loss, total_loss,
            approx_kl, clip_fraction, cost_advantage_mean.
        """
        if len(self.buffer) == 0:
            return {}

        data = self.buffer.to_tensors(self.device)

        # ── Step 3: Reward advantage via GAE ──────────────────────────────
        reward_advantages, reward_returns = compute_gae(
            rewards=data["rewards"],
            values=data["values"],
            dones=data["dones"],
            next_value=next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # ── Step 4: Cost advantage via GAE ────────────────────────────────
        cost_advantages, cost_returns = compute_gae(
            rewards=data["cost_rewards"],
            values=data["cost_values"],
            dones=data["dones"],
            next_value=next_cost_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # ── Step 5: Cost-aware advantage Â^L_t (Eq. 7 of paper) ──────────
        # Â^L_t = Â^R_t - λ_i * Â^{C_i}_t
        lagrange_advantages = (
            reward_advantages - lagrange_lambda * cost_advantages
        )

        # Normalise combined advantage for training stability
        lagrange_advantages = (
            (lagrange_advantages - lagrange_advantages.mean())
            / (lagrange_advantages.std() + 1e-8)
        )

        # ── Steps 6–7: Mini-batch PPO updates ─────────────────────────────
        T = len(data["rewards"])
        mini_batch_size = max(T // self.mini_batches, 1)

        stats_accumulator: Dict[str, list] = {
            "policy_loss": [], "value_loss": [], "entropy_loss": [],
            "total_loss": [], "approx_kl": [], "clip_fraction": [],
        }

        for _ in range(self.mini_batches):
            # Random mini-batch indices
            indices = torch.randperm(T, device=self.device)[:mini_batch_size]

            mb_obs        = data["observations"][indices]
            mb_actions    = data["actions"][indices]
            mb_log_probs_old = data["log_probs_old"][indices]
            mb_advantages = lagrange_advantages[indices]
            mb_returns    = reward_returns[indices]
            mb_cost_returns = cost_returns[indices]
            mb_values_old = data["values"][indices]

            # ── Actor update (Step 6) ─────────────────────────────────────
            log_probs_new, entropy = self.actor.evaluate_actions(
                mb_obs, mb_actions
            )

            # Importance ratio ρ_t = π_new(a|o) / π_old(a|o)
            ratio = torch.exp(log_probs_new - mb_log_probs_old)

            # Clipped surrogate objective (Eq. 8)
            surr1 = ratio * mb_advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * mb_advantages
            )
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # Entropy bonus H[π]
            entropy_loss = -torch.mean(entropy)

            # Actor total loss
            actor_total = policy_loss + self.entropy_coef * entropy_loss

            self.actor_optimizer.zero_grad()
            actor_total.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
            self.actor_optimizer.step()

            # ── Critic update (Step 7) — only when critic is owned ────────
            value_loss_val = torch.tensor(0.0, device=self.device)
            if self.critic is not None and self.critic_optimizer is not None:
                # Use global states if available, else fall back to obs
                if "global_states" in data:
                    mb_states = data["global_states"][indices]
                else:
                    mb_states = mb_obs

                value_loss_val = self.critic.value_loss(
                    state=mb_states,
                    returns=mb_returns.unsqueeze(-1),
                    clip_range=self.clip_epsilon,
                    old_values=mb_values_old.unsqueeze(-1),
                )

                # Add cost value loss if constraint heads exist
                if self.critic.n_constraint_heads > 0:
                    # mb_cost_returns: (T,) → broadcast to (T, n_agents)
                    cost_ret_expanded = mb_cost_returns.unsqueeze(-1).expand(
                        -1, self.critic.n_constraint_heads
                    )
                    cost_loss = self.critic.cost_value_loss(
                        mb_states, cost_ret_expanded
                    )
                    value_loss_val = value_loss_val + cost_loss

                self.critic_optimizer.zero_grad()
                value_loss_val.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()

            # ── Logging statistics ────────────────────────────────────────
            with torch.no_grad():
                approx_kl = torch.mean((ratio - 1.0) - torch.log(ratio)).item()
                clip_frac = torch.mean(
                    (torch.abs(ratio - 1.0) > self.clip_epsilon).float()
                ).item()

            stats_accumulator["policy_loss"].append(policy_loss.item())
            stats_accumulator["value_loss"].append(value_loss_val.item())
            stats_accumulator["entropy_loss"].append(entropy_loss.item())
            stats_accumulator["total_loss"].append(actor_total.item())
            stats_accumulator["approx_kl"].append(approx_kl)
            stats_accumulator["clip_fraction"].append(clip_frac)

        self._update_count += 1
        self.buffer.clear()

        return {
            k: float(np.mean(v))
            for k, v in stats_accumulator.items()
        } | {"cost_advantage_mean": float(cost_advantages.mean().item())}

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def get_action_and_value(
        self,
        obs: np.ndarray,
        global_state: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, float]:
        """
        Sample an action and compute value estimates for storage.

        Used during rollout collection.

        Returns
        -------
        action      : int
        log_prob    : float
        value       : float   — V(s_t) from owned critic (or 0.0)
        cost_value  : float   — V^{C_i}(s_t) from owned critic (or 0.0)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t, log_prob_t, _ = self.actor.act(obs_t, deterministic=deterministic)

        value = 0.0
        cost_value = 0.0
        if self.critic is not None:
            state_t = (
                torch.tensor(global_state, dtype=torch.float32, device=self.device)
                if global_state is not None
                else obs_t
            )
            value = float(self.critic.get_value(state_t).item())
            if self.critic.n_constraint_heads > 0:
                cv = self.critic.get_cost_values(state_t)
                # Cost value for this specific agent (index agent_id - 1)
                agent_idx = max(0, self.agent_id - 1)
                idx = min(agent_idx, cv.shape[-1] - 1)
                cost_value = float(cv[0, idx].item())

        return (
            int(action_t.item()),
            float(log_prob_t.item()),
            value,
            cost_value,
        )

    def set_device(self, device: str) -> None:
        self.device = torch.device(device)
        self.actor.to(self.device)
        if self.critic is not None:
            self.critic.to(self.device)

    @property
    def update_count(self) -> int:
        return self._update_count
