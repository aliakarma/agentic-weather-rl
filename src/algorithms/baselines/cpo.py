"""CPO-style constrained policy optimization baseline."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import BaseAgent, PolicyNetwork, RolloutBuffer, ValueNetwork


class CPOAgent(BaseAgent):
    _ALGO_NAME = "cpo"

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_agents: int = 3,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        seed: int = 42,
        rollout_len: int = 128,
        ppo_epochs: int = 4,
        clip_eps: float = 0.2,
        cost_limit: float = 0.10,
        lambda_lr: float = 5e-3,
        value_coef: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            gamma=gamma,
            lr=lr_actor,
            seed=seed,
            **kwargs,
        )
        self.rollout_len = rollout_len
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.value_coef = value_coef

        self.actors: List[PolicyNetwork] = [
            PolicyNetwork(obs_dim, hidden_dim, n_actions, seed + 41 * i, lr=lr_actor)
            for i in range(n_agents)
        ]
        self.reward_critics: List[ValueNetwork] = [
            ValueNetwork(obs_dim, hidden_dim, 1, seed + 131 * i, lr=lr_critic)
            for i in range(n_agents)
        ]
        self.cost_critics: List[ValueNetwork] = [
            ValueNetwork(obs_dim, hidden_dim, 1, seed + 231 * i, lr=lr_critic)
            for i in range(n_agents)
        ]
        self.lambdas = np.full(n_agents, 0.1, dtype=np.float32)
        self.rollout = RolloutBuffer()

    def act(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True) -> Dict[int, int]:
        actions = {}
        logps = {}
        r_values = {}
        c_values = {}
        for i in range(1, self.n_agents + 1):
            obs = obs_dict[i]
            if deterministic:
                a = self.actors[i - 1].greedy_action(obs)
                p = self.actors[i - 1].probs(obs[None, :])[0]
                lp = float(np.log(np.clip(p[a], 1e-8, 1.0)))
            else:
                a, lp = self.actors[i - 1].sample_action(obs, self.rng)
            rv = float(self.reward_critics[i - 1].predict(obs[None, :])[0, 0])
            cv = float(self.cost_critics[i - 1].predict(obs[None, :])[0, 0])
            actions[i] = int(a)
            logps[i] = lp
            r_values[i] = rv
            c_values[i] = cv

        self._latest_logps = logps
        self._latest_r_values = r_values
        self._latest_c_values = c_values
        return actions

    def observe(self, transition: dict) -> None:
        transition = dict(transition)
        transition["logps"] = dict(getattr(self, "_latest_logps", {}))
        transition["r_values"] = dict(getattr(self, "_latest_r_values", {}))
        transition["c_values"] = dict(getattr(self, "_latest_c_values", {}))
        self.rollout.add(transition)

    def _returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
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
        costs = np.array([x["violation"] for x in self.rollout.data], dtype=np.float32)
        dones = np.array([float(x["done"]) for x in self.rollout.data], dtype=np.float32)
        returns_r = self._returns(rewards, dones)
        returns_c = self._returns(costs, dones)

        losses = []
        for agent_idx in range(self.n_agents):
            obs = np.stack([x["obs"][agent_idx + 1] for x in self.rollout.data]).astype(np.float32)
            acts = np.array([x["actions"][agent_idx + 1] for x in self.rollout.data], dtype=np.int64)
            old_logp = np.array([x["logps"][agent_idx + 1] for x in self.rollout.data], dtype=np.float32)
            old_rv = np.array([x["r_values"][agent_idx + 1] for x in self.rollout.data], dtype=np.float32)
            old_cv = np.array([x["c_values"][agent_idx + 1] for x in self.rollout.data], dtype=np.float32)

            adv_r = returns_r - old_rv
            adv_c = returns_c - old_cv
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)
            adv = adv_r - float(self.lambdas[agent_idx]) * adv_c

            rr = returns_r[:, None]
            rc = returns_c[:, None]

            for _ in range(self.ppo_epochs):
                pl = self.actors[agent_idx].update(
                    obs_batch=obs,
                    act_batch=acts,
                    adv_batch=adv,
                    old_logp_batch=old_logp,
                    clip_eps=self.clip_eps,
                    entropy_coef=0.0,
                )
                vrl = self.reward_critics[agent_idx].update_mse(obs, rr)
                vcl = self.cost_critics[agent_idx].update_mse(obs, rc)
                losses.append(pl + self.value_coef * (vrl + vcl))

            mean_cost = float(np.mean(costs))
            self.lambdas[agent_idx] = max(
                0.0,
                float(self.lambdas[agent_idx] + self.lambda_lr * (mean_cost - self.cost_limit)),
            )

        self.rollout.clear()
        return float(np.mean(losses)) if losses else 0.0

    def _serialize(self) -> dict:
        payload = super()._serialize()
        payload.update(
            {
                "lambdas": self.lambdas.tolist(),
                "actors": [a.net.state_dict() for a in self.actors],
                "reward_critics": [c.net.state_dict() for c in self.reward_critics],
                "cost_critics": [c.net.state_dict() for c in self.cost_critics],
            }
        )
        return payload

    def _deserialize(self, payload: dict) -> None:
        super()._deserialize(payload)
        self.lambdas = np.array(payload.get("lambdas", self.lambdas.tolist()), dtype=np.float32)
        for net, sd in zip(self.actors, payload.get("actors", [])):
            net.net.load_state_dict(sd)
        for net, sd in zip(self.reward_critics, payload.get("reward_critics", [])):
            net.net.load_state_dict(sd)
        for net, sd in zip(self.cost_critics, payload.get("cost_critics", [])):
            net.net.load_state_dict(sd)
