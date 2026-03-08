"""
src/algorithms/baselines/dqn.py
================================
Independent Deep Q-Network (DQN) baseline.

Each agent maintains its own DQN and is trained independently.
No coordination or shared critic. Corresponds to Table 2 row:
  DQN: reward=55.6±3.2, VR=14.7%

Architecture:
  Q-network: 3-layer MLP (obs_dim → 128 → 128 → n_actions)
  Replay buffer: uniform random experience replay
  Target network: hard update every target_update_freq steps

Action selection: ε-greedy (annealed from ε_start to ε_end).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment.disaster_env import DisasterEnv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Three-layer MLP Q-function for a single agent."""

    def __init__(self, obs_dim: int = 12, n_actions: int = 4,
                 hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    obs:      np.ndarray
    action:   int
    reward:   float
    next_obs: np.ndarray
    done:     bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN agent (single agent)
# ---------------------------------------------------------------------------

class _SingleDQN:
    def __init__(self, agent_id: int, obs_dim: int, n_actions: int,
                 lr: float, gamma: float, batch_size: int,
                 target_update_freq: int, buffer_capacity: int,
                 device: torch.device) -> None:
        self.agent_id = agent_id
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.n_actions = n_actions

        self.q_net     = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self._steps = 0

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q = self.q_net(obs_t)
            return int(q.argmax(dim=-1).item())

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        obs_b      = torch.tensor(np.array([t.obs      for t in batch]), dtype=torch.float32, device=self.device)
        act_b      = torch.tensor([t.action   for t in batch], dtype=torch.long,    device=self.device)
        rew_b      = torch.tensor([t.reward   for t in batch], dtype=torch.float32, device=self.device)
        next_obs_b = torch.tensor(np.array([t.next_obs for t in batch]), dtype=torch.float32, device=self.device)
        done_b     = torch.tensor([t.done     for t in batch], dtype=torch.float32, device=self.device)

        q_vals = self.q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs_b).max(dim=1).values
            targets = rew_b + self.gamma * next_q * (1.0 - done_b)

        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


# ---------------------------------------------------------------------------
# Multi-agent wrapper — DQNAgent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Independent DQN for all three agents (no coordination).

    Parameters
    ----------
    obs_dim, n_actions, n_agents — environment dimensions
    lr            : learning rate         (default 3e-4)
    gamma         : discount factor       (default 0.99)
    batch_size    : replay batch size     (default 64)
    buffer_cap    : replay buffer size    (default 10_000)
    eps_start/end : ε-greedy schedule     (1.0 → 0.05)
    eps_decay     : linear decay episodes (default 500)
    target_update : hard target-net sync  (default 100 steps)
    device        : compute device
    """

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_agents: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_cap: int = 10_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 500,
        target_update: int = 100,
        device: str = "cpu",
    ) -> None:
        dev = torch.device(device)
        self.n_agents = n_agents
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self._episode  = 0

        self._agents: Dict[int, _SingleDQN] = {
            i: _SingleDQN(
                agent_id=i, obs_dim=obs_dim, n_actions=n_actions,
                lr=lr, gamma=gamma, batch_size=batch_size,
                target_update_freq=target_update, buffer_capacity=buffer_cap,
                device=dev,
            )
            for i in range(1, n_agents + 1)
        }

    # -----------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        frac = min(self._episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs_dict: Dict[int, np.ndarray],
            deterministic: bool = False) -> Dict[int, int]:
        eps = 0.0 if deterministic else self.epsilon
        return {i: self._agents[i].select_action(obs_dict[i], eps)
                for i in range(1, self.n_agents + 1)}

    def train(
        self,
        env: DisasterEnv,
        n_episodes: int = 1000,
        checkpoint_dir: str = "checkpoints/",
        save_interval: int = 100,
    ) -> Dict:
        history: Dict[str, list] = {"rewards": [], "losses": [], "epsilons": []}

        for ep in range(1, n_episodes + 1):
            self._episode = ep
            obs_dict, _ = env.reset(seed=ep)
            total_reward = 0.0
            ep_losses: List[float] = []

            for _ in range(env.episode_length):
                actions = self.act(obs_dict)
                next_obs, reward, terminated, truncated, info = env.step(actions)

                for i in range(1, self.n_agents + 1):
                    self._agents[i].buffer.push(Transition(
                        obs=obs_dict[i], action=actions[i],
                        reward=reward, next_obs=next_obs[i],
                        done=(terminated or truncated),
                    ))
                    loss = self._agents[i].update()
                    ep_losses.append(loss)

                obs_dict = next_obs
                total_reward += reward
                if terminated or truncated:
                    break

            history["rewards"].append(total_reward)
            history["losses"].append(float(np.mean(ep_losses)))
            history["epsilons"].append(self.epsilon)

            if ep % 50 == 0:
                logger.info("DQN ep %d | R=%.2f | ε=%.4f", ep, total_reward, self.epsilon)

            if ep % save_interval == 0:
                self.save(str(Path(checkpoint_dir) / f"dqn_ep{ep}.pt"))

        return history

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {i: self._agents[i].q_net.state_dict()
             for i in range(1, self.n_agents + 1)}, path
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=True)
        for i in range(1, self.n_agents + 1):
            self._agents[i].q_net.load_state_dict(ckpt[i])
            self._agents[i].target_net.load_state_dict(ckpt[i])
