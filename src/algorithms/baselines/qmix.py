"""QMIX baseline with replay buffer and monotonic mixing network."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import Adam, BaseAgent, ReplayBuffer, ValueNetwork


class MonotonicMixer:
    """Simple monotonic mixer: Q_tot = sum_i w_i(s) * q_i + b(s), w_i >= 0."""

    def __init__(self, state_dim: int, n_agents: int, seed: int, lr: float):
        rng = np.random.default_rng(seed)
        self.HW = (rng.standard_normal((state_dim, n_agents)) * 0.05).astype(np.float32)
        self.bW = np.zeros(n_agents, dtype=np.float32)
        self.Hb = (rng.standard_normal((state_dim, 1)) * 0.05).astype(np.float32)
        self.bb = np.zeros(1, dtype=np.float32)
        self.opt = Adam([self.HW, self.bW, self.Hb, self.bb], lr=lr)

    def _weights(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = state @ self.HW + self.bW
        w = np.log1p(np.exp(z))
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return w, z, sigmoid

    def forward(self, q_agents: np.ndarray, state: np.ndarray) -> np.ndarray:
        w, _, _ = self._weights(state)
        b = state @ self.Hb + self.bb
        return (w * q_agents).sum(axis=1, keepdims=True) + b

    def update(self, q_agents: np.ndarray, state: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        pred = self.forward(q_agents, state)
        diff = pred - target
        loss = float(np.mean(diff * diff))
        grad_pred = 2.0 * diff / max(len(state), 1)

        w, _z, sigmoid = self._weights(state)
        grad_q = grad_pred * w  # dL/dq_i

        grad_w = grad_pred * q_agents
        grad_z = grad_w * sigmoid

        gHW = state.T @ grad_z
        gbW = grad_z.sum(axis=0)

        gbias = grad_pred
        gHb = state.T @ gbias
        gbb = gbias.sum(axis=0)

        self.opt.step([
            gHW.astype(np.float32),
            gbW.astype(np.float32),
            gHb.astype(np.float32),
            gbb.astype(np.float32),
        ])
        return loss, grad_q

    def copy_from(self, other: "MonotonicMixer") -> None:
        self.HW[...] = other.HW
        self.bW[...] = other.bW
        self.Hb[...] = other.Hb
        self.bb[...] = other.bb

    def state_dict(self) -> dict:
        return {
            "HW": self.HW.copy(),
            "bW": self.bW.copy(),
            "Hb": self.Hb.copy(),
            "bb": self.bb.copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.HW[...] = sd["HW"]
        self.bW[...] = sd["bW"]
        self.Hb[...] = sd["Hb"]
        self.bb[...] = sd["bb"]


class QMIXAgent(BaseAgent):
    _ALGO_NAME = "qmix"

    def __init__(
        self,
        obs_dim: int = 12,
        state_dim: int = 24,
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
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_nets: List[ValueNetwork] = [
            ValueNetwork(obs_dim, hidden_dim, n_actions, seed + 17 * i, lr=lr)
            for i in range(n_agents)
        ]
        self.target_q_nets: List[ValueNetwork] = [
            ValueNetwork(obs_dim, hidden_dim, n_actions, seed + 117 * i, lr=lr)
            for i in range(n_agents)
        ]
        for tnet, net in zip(self.target_q_nets, self.q_nets):
            tnet.net.copy_from(net.net)

        self.mixer = MonotonicMixer(obs_dim * n_agents, n_agents, seed + 901, lr=lr)
        self.target_mixer = MonotonicMixer(obs_dim * n_agents, n_agents, seed + 1901, lr=lr)
        self.target_mixer.copy_from(self.mixer)

        self.replay = ReplayBuffer(capacity=buffer_size)
        self.update_steps = 0

    def _global_state(self, obs_dict: Dict[int, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs_dict[i] for i in range(1, self.n_agents + 1)]).astype(np.float32)

    def act(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True) -> Dict[int, int]:
        actions = {}
        for i in range(1, self.n_agents + 1):
            obs = obs_dict[i][None, :]
            if (not deterministic) and (self.rng.random() < self.epsilon):
                a = int(self.rng.integers(0, self.n_actions))
            else:
                q = self.q_nets[i - 1].predict(obs)[0]
                a = int(np.argmax(q))
            actions[i] = a
        return actions

    def observe(self, transition: dict) -> None:
        item = dict(transition)
        item["state"] = self._global_state(transition["obs"])
        item["next_state"] = self._global_state(transition["next_obs"])
        self.replay.add(item)

    def update(self) -> float:
        if len(self.replay) < self.batch_size:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return 0.0

        batch = self.replay.sample(self.batch_size, self.rng)
        reward = np.array([b["reward"] for b in batch], dtype=np.float32)[:, None]
        done = np.array([float(b["done"]) for b in batch], dtype=np.float32)[:, None]
        state = np.stack([b["state"] for b in batch]).astype(np.float32)
        next_state = np.stack([b["next_state"] for b in batch]).astype(np.float32)

        q_chosen = []
        q_next_max = []
        obs_batches = []
        act_batches = []

        for agent_idx in range(self.n_agents):
            obs = np.stack([b["obs"][agent_idx + 1] for b in batch]).astype(np.float32)
            nxt = np.stack([b["next_obs"][agent_idx + 1] for b in batch]).astype(np.float32)
            act = np.array([int(b["actions"][agent_idx + 1]) for b in batch], dtype=np.int64)

            pred = self.q_nets[agent_idx].predict(obs)
            target_next = self.target_q_nets[agent_idx].predict(nxt)

            q_chosen.append(pred[np.arange(self.batch_size), act])
            q_next_max.append(target_next.max(axis=1))
            obs_batches.append(obs)
            act_batches.append(act)

        q_chosen_arr = np.stack(q_chosen, axis=1).astype(np.float32)
        q_next_arr = np.stack(q_next_max, axis=1).astype(np.float32)

        with_target = self.target_mixer.forward(q_next_arr, next_state)
        td_target = reward + self.gamma * (1.0 - done) * with_target

        mix_loss, grad_q = self.mixer.update(q_chosen_arr, state, td_target)

        q_losses = []
        for agent_idx in range(self.n_agents):
            obs = obs_batches[agent_idx]
            act = act_batches[agent_idx]
            grad_out = np.zeros((self.batch_size, self.n_actions), dtype=np.float32)
            grad_out[np.arange(self.batch_size), act] = grad_q[:, agent_idx]
            q_loss = self.q_nets[agent_idx].update_with_output_grad(obs, grad_out)
            q_losses.append(q_loss)

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            for tnet, net in zip(self.target_q_nets, self.q_nets):
                tnet.net.copy_from(net.net)
            self.target_mixer.copy_from(self.mixer)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(mix_loss + np.mean(q_losses))

    def _serialize(self) -> dict:
        payload = super()._serialize()
        payload.update(
            {
                "epsilon": self.epsilon,
                "q_nets": [n.net.state_dict() for n in self.q_nets],
                "target_q_nets": [n.net.state_dict() for n in self.target_q_nets],
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
            }
        )
        return payload

    def _deserialize(self, payload: dict) -> None:
        super()._deserialize(payload)
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        for net, sd in zip(self.q_nets, payload.get("q_nets", [])):
            net.net.load_state_dict(sd)
        for net, sd in zip(self.target_q_nets, payload.get("target_q_nets", [])):
            net.net.load_state_dict(sd)
        if "mixer" in payload:
            self.mixer.load_state_dict(payload["mixer"])
        if "target_mixer" in payload:
            self.target_mixer.load_state_dict(payload["target_mixer"])
