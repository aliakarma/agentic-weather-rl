"""Independent PPO baseline with rollout buffer and trainable actor-critic."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import BaseAgent, PolicyNetwork, RolloutBuffer, ValueNetwork


class IPPOAgent(BaseAgent):
    _ALGO_NAME = "ippo"

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_agents: int = 3,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 3e-4,
        seed: int = 42,
        rollout_len: int = 128,
        ppo_epochs: int = 4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
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
        self.rollout_len = rollout_len
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actors: List[PolicyNetwork] = [
            PolicyNetwork(obs_dim, hidden_dim, n_actions, seed + 19 * i, lr=lr)
            for i in range(n_agents)
        ]
        self.critics: List[ValueNetwork] = [
            ValueNetwork(obs_dim, hidden_dim, 1, seed + 73 * i, lr=lr)
            for i in range(n_agents)
        ]
        self.rollout = RolloutBuffer()

    def act(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True) -> Dict[int, int]:
        actions = {}
        logps = {}
        values = {}
        for i in range(1, self.n_agents + 1):
            obs = obs_dict[i]
            if deterministic:
                a = self.actors[i - 1].greedy_action(obs)
                p = self.actors[i - 1].probs(obs[None, :])[0]
                lp = float(np.log(np.clip(p[a], 1e-8, 1.0)))
            else:
                a, lp = self.actors[i - 1].sample_action(obs, self.rng)
            v = float(self.critics[i - 1].predict(obs[None, :])[0, 0])
            actions[i] = int(a)
            logps[i] = lp
            values[i] = v
        self._latest_logps = logps
        self._latest_values = values
        return actions

    def observe(self, transition: dict) -> None:
        transition = dict(transition)
        transition["logps"] = dict(getattr(self, "_latest_logps", {}))
        transition["values"] = dict(getattr(self, "_latest_values", {}))
        self.rollout.add(transition)

    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            out[t] = running
        return out

    def update(self) -> float:
        if len(self.rollout) == 0:
            return 0.0

        done = bool(self.rollout.data[-1]["done"])
        if len(self.rollout) < self.rollout_len and not done:
            return 0.0

        rewards = np.array([x["reward"] for x in self.rollout.data], dtype=np.float32)
        dones = np.array([float(x["done"]) for x in self.rollout.data], dtype=np.float32)
        returns = self._compute_returns(rewards, dones)

        losses = []
        for agent_idx in range(self.n_agents):
            obs = np.stack([x["obs"][agent_idx + 1] for x in self.rollout.data]).astype(np.float32)
            acts = np.array([x["actions"][agent_idx + 1] for x in self.rollout.data], dtype=np.int64)
            old_logp = np.array([x["logps"][agent_idx + 1] for x in self.rollout.data], dtype=np.float32)
            old_val = np.array([x["values"][agent_idx + 1] for x in self.rollout.data], dtype=np.float32)

            adv = returns - old_val
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret_col = returns[:, None]

            for _ in range(self.ppo_epochs):
                pol_loss = self.actors[agent_idx].update(
                    obs_batch=obs,
                    act_batch=acts,
                    adv_batch=adv,
                    old_logp_batch=old_logp,
                    clip_eps=self.clip_eps,
                    entropy_coef=self.entropy_coef,
                )
                val_loss = self.critics[agent_idx].update_mse(obs, ret_col)
                losses.append(pol_loss + self.value_coef * val_loss)

        self.rollout.clear()
        return float(np.mean(losses)) if losses else 0.0

    def _serialize(self) -> dict:
        payload = super()._serialize()
        payload.update(
            {
                "actors": [a.net.state_dict() for a in self.actors],
                "critics": [c.net.state_dict() for c in self.critics],
            }
        )
        return payload

    def _deserialize(self, payload: dict) -> None:
        super()._deserialize(payload)
        for net, sd in zip(self.actors, payload.get("actors", [])):
            net.net.load_state_dict(sd)
        for net, sd in zip(self.critics, payload.get("critics", [])):
            net.net.load_state_dict(sd)
