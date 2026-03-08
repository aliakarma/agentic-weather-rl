"""
src/algorithms/baselines/qmix.py
=================================
QMIX cooperative MARL baseline.

Implements a simplified QMIX (Rashid et al., 2018) with a monotonic mixing
network that combines per-agent utility values into a joint Q^{tot}.

Corresponds to Table 2 row:  QMIX: reward=69.8±3.7, VR=10.5%

Architecture:
  Agent utility networks : Q_i(o_{i,t}, a_i) — per-agent utility values
  Mixing network         : Q^{tot} = f_mix(u_1, ..., u_n | s_t)
                           with weights constrained non-negative (monotonicity)

The monotonicity constraint ensures Individual-Global Max (IGM): the joint
greedy action argmax Q^{tot} equals the combination of individual argmax_i Q_i.

Action selection: ε-greedy on per-agent utilities Q_i (equivalent under IGM).
Training: TD(0) on joint Q^{tot}, gradient flows back through mixing + utilities.

References:
  Rashid et al. "QMIX: Monotonic Value Function Factorisation for MARL." ICML 2018.
  foerster2018counterfactual — cited in paper for CTDE context.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.environment.disaster_env import DisasterEnv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-agent utility network
# ---------------------------------------------------------------------------

class AgentUtilityNet(nn.Module):
    """Q_i(o_i, a_i) — scalar utility for agent i given local obs and action."""

    def __init__(self, obs_dim: int = 12, n_actions: int = 4,
                 hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions: (B, n_actions)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)


# ---------------------------------------------------------------------------
# QMIX mixing network
# ---------------------------------------------------------------------------

class MixingNetwork(nn.Module):
    """
    Monotonic mixing: Q^{tot} = g(u_1, ..., u_n | s_t)
    Weights are non-negative (ensured by absolute value / ELU+1).
    """

    def __init__(self, n_agents: int = 3, state_dim: int = 24,
                 mixing_hidden: int = 32) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden = mixing_hidden

        # Hyper-networks that produce non-negative mixing weights from s_t
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, n_agents * mixing_hidden),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden),
            nn.ReLU(),
            nn.Linear(mixing_hidden, mixing_hidden),
        )
        # Bias hyper-networks (unconstrained)
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden), nn.ReLU(),
            nn.Linear(mixing_hidden, 1),
        )

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        agent_qs : (B, n_agents)  — per-agent chosen Q-values
        state    : (B, state_dim) — global state

        Returns
        -------
        q_tot : (B, 1)
        """
        B = agent_qs.size(0)
        agent_qs = agent_qs.unsqueeze(1)  # (B, 1, n_agents)

        # Layer 1 — non-negative weights via abs()
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(state).view(B, 1, self.mixing_hidden)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # (B, 1, mixing_hidden)

        # Layer 2 — non-negative weights via abs()
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.mixing_hidden, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)

        return q_tot.squeeze(-1)  # (B, 1)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class QMIXTransition:
    obs:        np.ndarray  # (n_agents, obs_dim)
    actions:    np.ndarray  # (n_agents,) int
    reward:     float
    next_obs:   np.ndarray  # (n_agents, obs_dim)
    global_state:      np.ndarray  # (state_dim,)
    next_global_state: np.ndarray  # (state_dim,)
    done:       bool


class QMIXReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: Deque[QMIXTransition] = deque(maxlen=capacity)

    def push(self, t: QMIXTransition) -> None:
        self._buf.append(t)

    def sample(self, n: int) -> List[QMIXTransition]:
        return random.sample(self._buf, min(n, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# QMIXAgent
# ---------------------------------------------------------------------------

class QMIXAgent:
    """
    QMIX cooperative multi-agent value-based learning.

    Parameters
    ----------
    obs_dim, n_actions, n_agents, state_dim — environment dimensions
    lr, gamma, batch_size, buffer_cap       — optimisation settings
    eps_start, eps_end, eps_decay           — ε-greedy schedule
    target_update                           — hard target-net sync frequency
    """

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_agents: int = 3,
        state_dim: int = 24,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_cap: int = 10_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 500,
        target_update: int = 100,
        mixing_hidden: int = 32,
        device: str = "cpu",
    ) -> None:
        self.n_agents   = n_agents
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.batch_size = batch_size
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.target_update = target_update
        self.device     = torch.device(device)
        self._episode   = 0
        self._steps     = 0

        # Utility networks (online + target)
        self.utility_nets = nn.ModuleList([
            AgentUtilityNet(obs_dim, n_actions).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_utility_nets = nn.ModuleList([
            AgentUtilityNet(obs_dim, n_actions).to(self.device)
            for _ in range(n_agents)
        ])

        # Mixing network (online + target)
        self.mixing_net = MixingNetwork(n_agents, state_dim, mixing_hidden).to(self.device)
        self.target_mixing_net = MixingNetwork(n_agents, state_dim, mixing_hidden).to(self.device)

        self._sync_target()

        params = list(self.utility_nets.parameters()) + list(self.mixing_net.parameters())
        self.optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
        self.buffer = QMIXReplayBuffer(buffer_cap)

    @property
    def epsilon(self) -> float:
        frac = min(self._episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _sync_target(self) -> None:
        for online, target in zip(self.utility_nets, self.target_utility_nets):
            target.load_state_dict(online.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def act(self, obs_dict: Dict[int, np.ndarray],
            deterministic: bool = False) -> Dict[int, int]:
        eps = 0.0 if deterministic else self.epsilon
        actions = {}
        for i in range(1, self.n_agents + 1):
            if not deterministic and random.random() < eps:
                actions[i] = random.randint(0, self.n_actions - 1)
            else:
                obs_t = torch.tensor(obs_dict[i], dtype=torch.float32,
                                     device=self.device)
                with torch.no_grad():
                    q = self.utility_nets[i - 1](obs_t)
                actions[i] = int(q.argmax(dim=-1).item())
        return actions

    def _update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        B = len(batch)

        obs_b      = torch.tensor(np.array([t.obs   for t in batch]),
                                  dtype=torch.float32, device=self.device)   # (B, n, obs)
        act_b      = torch.tensor(np.array([t.actions for t in batch]),
                                  dtype=torch.long,    device=self.device)   # (B, n)
        rew_b      = torch.tensor([t.reward for t in batch],
                                  dtype=torch.float32, device=self.device)   # (B,)
        next_obs_b = torch.tensor(np.array([t.next_obs for t in batch]),
                                  dtype=torch.float32, device=self.device)   # (B, n, obs)
        state_b    = torch.tensor(np.array([t.global_state for t in batch]),
                                  dtype=torch.float32, device=self.device)   # (B, state)
        next_state_b = torch.tensor(np.array([t.next_global_state for t in batch]),
                                    dtype=torch.float32, device=self.device) # (B, state)
        done_b     = torch.tensor([t.done for t in batch],
                                  dtype=torch.float32, device=self.device)   # (B,)

        # Chosen Q-values: Q_i(o_{i,t}, a_{i,t})
        chosen_qs = []
        for idx in range(self.n_agents):
            q_all = self.utility_nets[idx](obs_b[:, idx, :])         # (B, n_actions)
            q_a   = q_all.gather(1, act_b[:, idx].unsqueeze(1))      # (B, 1)
            chosen_qs.append(q_a)
        chosen_qs_t = torch.cat(chosen_qs, dim=1)                    # (B, n_agents)

        q_tot = self.mixing_net(chosen_qs_t, state_b)                # (B, 1)

        # Target Q^{tot}: greedy next actions under target nets
        with torch.no_grad():
            target_qs = []
            for idx in range(self.n_agents):
                tq = self.target_utility_nets[idx](next_obs_b[:, idx, :]).max(dim=1, keepdim=True).values
                target_qs.append(tq)
            target_qs_t = torch.cat(target_qs, dim=1)
            target_q_tot = self.target_mixing_net(target_qs_t, next_state_b)
            y = rew_b.unsqueeze(1) + self.gamma * target_q_tot * (1.0 - done_b.unsqueeze(1))

        loss = F.mse_loss(q_tot, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.utility_nets.parameters()) + list(self.mixing_net.parameters()), 10.0
        )
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update == 0:
            self._sync_target()

        return float(loss.item())

    def train(
        self,
        env: DisasterEnv,
        n_episodes: int = 1000,
        checkpoint_dir: str = "checkpoints/",
        save_interval: int = 100,
    ) -> Dict:
        history: Dict[str, list] = {"rewards": [], "losses": []}

        for ep in range(1, n_episodes + 1):
            self._episode = ep
            obs_dict, _ = env.reset(seed=ep)
            total_reward, ep_losses = 0.0, []

            for _ in range(env.episode_length):
                actions = self.act(obs_dict)
                global_state = env.get_global_state()
                next_obs, reward, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated
                next_state = env.get_global_state()

                self.buffer.push(QMIXTransition(
                    obs=np.array([obs_dict[i] for i in range(1, self.n_agents + 1)]),
                    actions=np.array([actions[i] for i in range(1, self.n_agents + 1)]),
                    reward=reward,
                    next_obs=np.array([next_obs[i] for i in range(1, self.n_agents + 1)]),
                    global_state=global_state,
                    next_global_state=next_state,
                    done=done,
                ))
                ep_losses.append(self._update())
                obs_dict = next_obs
                total_reward += reward
                if done:
                    break

            history["rewards"].append(total_reward)
            history["losses"].append(float(np.mean(ep_losses)))
            if ep % 50 == 0:
                logger.info("QMIX ep %d | R=%.2f | ε=%.4f", ep, total_reward, self.epsilon)
            if ep % save_interval == 0:
                self.save(str(Path(checkpoint_dir) / f"qmix_ep{ep}.pt"))

        return history

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "utility":  [n.state_dict() for n in self.utility_nets],
            "mixing":    self.mixing_net.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=True)
        for i, sd in enumerate(ckpt["utility"]):
            self.utility_nets[i].load_state_dict(sd)
            self.target_utility_nets[i].load_state_dict(sd)
        self.mixing_net.load_state_dict(ckpt["mixing"])
        self.target_mixing_net.load_state_dict(ckpt["mixing"])
