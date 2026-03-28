"""Independent DQN baseline with replay buffer and target networks."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import BaseAgent, ReplayBuffer, ValueNetwork


class DQNAgent(BaseAgent):
    _ALGO_NAME = "dqn"

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_agents: int = 3,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 3e-4,
        seed: int = 42,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.999,
        batch_size: int = 64,
        target_update_interval: int = 200,
        buffer_size: int = 100_000,
        **kwargs,
    ):
        super().__init__(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            gamma=gamma,
            lr=lr,
            seed=seed,
            **kwargs,
        )
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_nets: List[ValueNetwork] = [
            ValueNetwork(obs_dim, hidden_dim, n_actions, seed + 101 * i, lr=lr)
            for i in range(n_agents)
        ]
        self.target_nets: List[ValueNetwork] = [
            ValueNetwork(obs_dim, hidden_dim, n_actions, seed + 601 * i, lr=lr)
            for i in range(n_agents)
        ]
        for tnet, net in zip(self.target_nets, self.q_nets):
            tnet.net.copy_from(net.net)

        self.replay = ReplayBuffer(capacity=buffer_size)
        self.update_steps = 0

    def act(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True) -> Dict[int, int]:
        acts = {}
        for i in range(1, self.n_agents + 1):
            obs = obs_dict[i][None, :]
            if (not deterministic) and (self.rng.random() < self.epsilon):
                acts[i] = int(self.rng.integers(0, self.n_actions))
            else:
                q = self.q_nets[i - 1].predict(obs)[0]
                acts[i] = int(np.argmax(q))
        return acts

    def observe(self, transition: dict) -> None:
        self.replay.add(transition)

    def update(self) -> float:
        if len(self.replay) < self.batch_size:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return 0.0

        batch = self.replay.sample(self.batch_size, self.rng)
        reward = np.array([b["reward"] for b in batch], dtype=np.float32)
        done = np.array([float(b["done"]) for b in batch], dtype=np.float32)

        losses = []
        for agent_idx in range(self.n_agents):
            obs = np.stack([b["obs"][agent_idx + 1] for b in batch]).astype(np.float32)
            nxt = np.stack([b["next_obs"][agent_idx + 1] for b in batch]).astype(np.float32)
            act = np.array([int(b["actions"][agent_idx + 1]) for b in batch], dtype=np.int64)

            q_pred_all = self.q_nets[agent_idx].predict(obs)
            q_pred = q_pred_all[np.arange(self.batch_size), act]
            q_next = self.target_nets[agent_idx].predict(nxt).max(axis=1)
            target = reward + self.gamma * (1.0 - done) * q_next

            target_full = q_pred_all.copy()
            target_full[np.arange(self.batch_size), act] = target
            loss = self.q_nets[agent_idx].update_mse(obs, target_full)
            losses.append(loss)

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            for tnet, net in zip(self.target_nets, self.q_nets):
                tnet.net.copy_from(net.net)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(np.mean(losses))

    def _serialize(self) -> dict:
        payload = super()._serialize()
        payload.update(
            {
                "epsilon": self.epsilon,
                "q_nets": [n.net.state_dict() for n in self.q_nets],
                "target_nets": [n.net.state_dict() for n in self.target_nets],
            }
        )
        return payload

    def _deserialize(self, payload: dict) -> None:
        super()._deserialize(payload)
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        for net, sd in zip(self.q_nets, payload.get("q_nets", [])):
            net.net.load_state_dict(sd)
        for net, sd in zip(self.target_nets, payload.get("target_nets", [])):
            net.net.load_state_dict(sd)
