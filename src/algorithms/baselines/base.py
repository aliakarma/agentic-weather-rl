"""
Common baseline infrastructure for learning-based MARL agents.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class Adam:
    """Lightweight Adam optimizer for NumPy parameter arrays."""

    def __init__(self, params: List[np.ndarray], lr: float = 3e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]) -> None:
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / b1t
            v_hat = self.v[i] / b2t
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLP:
    """2-layer MLP with tanh hidden activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_dim, out_dim)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    @property
    def params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        h = np.tanh(x @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        return y, (x, h)

    def backward(self, cache: Tuple[np.ndarray, np.ndarray], grad_y: np.ndarray) -> List[np.ndarray]:
        x, h = cache
        gW2 = h.T @ grad_y
        gb2 = grad_y.sum(axis=0)
        gh = grad_y @ self.W2.T
        gz = gh * (1.0 - h * h)
        gW1 = x.T @ gz
        gb1 = gz.sum(axis=0)
        return [gW1.astype(np.float32), gb1.astype(np.float32), gW2.astype(np.float32), gb2.astype(np.float32)]

    def copy_from(self, other: "MLP") -> None:
        for p, q in zip(self.params, other.params):
            p[...] = q

    def state_dict(self) -> dict:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.W1[...] = sd["W1"]
        self.b1[...] = sd["b1"]
        self.W2[...] = sd["W2"]
        self.b2[...] = sd["b2"]


class PolicyNetwork:
    """Discrete-action policy network with PPO-compatible gradient update."""

    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int, seed: int, lr: float):
        self.net = MLP(obs_dim, hidden_dim, n_actions, seed)
        self.opt = Adam(self.net.params, lr=lr)
        self.n_actions = n_actions

    def probs(self, obs_batch: np.ndarray) -> np.ndarray:
        logits, _ = self.net.forward(obs_batch)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-8, None)

    def sample_action(self, obs: np.ndarray, rng: np.random.Generator) -> Tuple[int, float]:
        p = self.probs(obs[None, :])[0]
        action = int(rng.choice(len(p), p=p))
        logp = float(np.log(np.clip(p[action], 1e-8, 1.0)))
        return action, logp

    def greedy_action(self, obs: np.ndarray) -> int:
        p = self.probs(obs[None, :])[0]
        return int(np.argmax(p))

    def update(self, obs_batch: np.ndarray, act_batch: np.ndarray,
               adv_batch: np.ndarray, old_logp_batch: np.ndarray,
               clip_eps: float = 0.2, entropy_coef: float = 0.0) -> float:
        logits, cache = self.net.forward(obs_batch)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-8, None)

        chosen = probs[np.arange(len(act_batch)), act_batch]
        logp = np.log(np.clip(chosen, 1e-8, 1.0))
        ratio = np.exp(logp - old_logp_batch)
        clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = np.minimum(ratio * adv_batch, clipped_ratio * adv_batch)
        policy_loss = -float(np.mean(surrogate))

        # Gradient for softmax policy: -(effective_adv) * grad(log pi(a|s)).
        effective_adv = clipped_ratio * adv_batch
        grad_logits = probs.copy()
        grad_logits[np.arange(len(act_batch)), act_batch] -= 1.0
        grad_logits *= (-effective_adv[:, None] / max(len(act_batch), 1))

        if entropy_coef > 0.0:
            entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)
            policy_loss -= entropy_coef * float(np.mean(entropy))
            # Entropy gradient is optional here; omitted for stability and speed.

        grads = self.net.backward(cache, grad_logits)
        self.opt.step(grads)
        return policy_loss


class ValueNetwork:
    """Value / Q network with MSE loss."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int, lr: float):
        self.net = MLP(in_dim, hidden_dim, out_dim, seed)
        self.opt = Adam(self.net.params, lr=lr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y, _ = self.net.forward(x)
        return y

    def update_mse(self, x: np.ndarray, target: np.ndarray) -> float:
        pred, cache = self.net.forward(x)
        diff = pred - target
        loss = float(np.mean(diff * diff))
        grad = 2.0 * diff / max(len(x), 1)
        grads = self.net.backward(cache, grad)
        self.opt.step(grads)
        return loss

    def update_with_output_grad(self, x: np.ndarray, grad_output: np.ndarray) -> float:
        pred, cache = self.net.forward(x)
        grads = self.net.backward(cache, grad_output)
        self.opt.step(grads)
        return float(np.mean(np.abs(grad_output)))


class ReplayBuffer:
    """Replay buffer for off-policy methods (DQN, QMIX)."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.storage: List[dict] = []
        self.ptr = 0

    def add(self, item: dict) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> List[dict]:
        idx = rng.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.storage)


class RolloutBuffer:
    """Rollout buffer for on-policy methods (IPPO, MAPPO, CPO)."""

    def __init__(self):
        self.data: List[dict] = []

    def add(self, item: dict) -> None:
        self.data.append(item)

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class TrainingMetrics:
    rewards: List[float]
    losses: List[float]
    violation_rates: List[float]


class BaseAgent(ABC):
    """Abstract baseline API expected by training scripts."""

    _ALGO_NAME = "base"

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_agents: int = 3,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 3e-4,
        seed: int = 42,
        device: str = "cpu",
        **_: dict,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.seed = seed
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._trained = False
        self.training_metrics = TrainingMetrics(rewards=[], losses=[], violation_rates=[])

    @abstractmethod
    def act(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True) -> Dict[int, int]:
        pass

    @abstractmethod
    def update(self) -> float:
        pass

    def observe(self, transition: dict) -> None:
        """Optional hook for algorithm-specific buffer writes."""
        _ = transition

    def _episode_seed(self, ep: int) -> int:
        # Shared seed rule across all baselines for reproducibility and fairness.
        return int(self.seed * 100_000 + ep)

    def train(self, env, n_episodes: int = 1000, checkpoint_dir: str = "checkpoints", save_interval: int = 500):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(
            f"    [{self._ALGO_NAME}] training start | episodes={n_episodes} "
            f"steps={env.episode_length} seed={self.seed}",
            flush=True,
        )

        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset(seed=self._episode_seed(ep))
            ep_reward = 0.0
            ep_viol = 0
            ep_losses: List[float] = []

            for _t in range(env.episode_length):
                actions = self.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = bool(terminated or truncated)
                transition = {
                    "obs": {k: v.copy() for k, v in obs.items()},
                    "actions": dict(actions),
                    "reward": float(reward),
                    "next_obs": {k: v.copy() for k, v in next_obs.items()},
                    "done": done,
                    "violation": float(info.get("violation", 0.0)),
                }
                self.observe(transition)
                loss = self.update()
                if loss > 0:
                    ep_losses.append(float(loss))

                ep_reward += reward
                ep_viol += int(info.get("violation", 0))
                obs = next_obs
                if done:
                    break

            ep_vr = ep_viol / max(env.episode_length, 1)
            ep_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            self.training_metrics.rewards.append(float(ep_reward))
            self.training_metrics.losses.append(ep_loss)
            self.training_metrics.violation_rates.append(ep_vr)

            if ep % max(1, n_episodes // 10) == 0 or ep == 1:
                print(
                    f"    [{self._ALGO_NAME}] ep {ep:5d}/{n_episodes} "
                    f"reward={ep_reward:7.2f} loss={ep_loss:8.4f} viol={ep_vr:6.3f}",
                    flush=True,
                )

            if ep % max(1, save_interval) == 0:
                self.save(f"{checkpoint_dir}/{self._ALGO_NAME}_ep{ep}.pt")

        self._trained = True
        print(
            f"    [{self._ALGO_NAME}] training done | "
            f"reward_mean={np.mean(self.training_metrics.rewards):.2f} "
            f"loss_mean={np.mean(self.training_metrics.losses):.4f} "
            f"viol_mean={np.mean(self.training_metrics.violation_rates):.4f}",
            flush=True,
        )
        return {
            "rewards": self.training_metrics.rewards,
            "losses": self.training_metrics.losses,
            "constraint_violations": self.training_metrics.violation_rates,
        }

    def get_actions(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True) -> Dict[int, int]:
        return self.act(obs_dict, deterministic=deterministic)

    def _serialize(self) -> dict:
        return {
            "algo": self._ALGO_NAME,
            "seed": self.seed,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "metrics": {
                "rewards": list(self.training_metrics.rewards),
                "losses": list(self.training_metrics.losses),
                "violation_rates": list(self.training_metrics.violation_rates),
            },
        }

    def _deserialize(self, payload: dict) -> None:
        metrics = payload.get("metrics", {})
        self.training_metrics = TrainingMetrics(
            rewards=list(metrics.get("rewards", [])),
            losses=list(metrics.get("losses", [])),
            violation_rates=list(metrics.get("violation_rates", [])),
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._serialize(), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._deserialize(payload)
        self._trained = True
