"""
src/algorithms/baselines/ippo.py
=================================
Independent PPO (IPPO) baseline.

Each agent trains its own actor and critic independently, using only its
local observation. No centralised critic, no coordination. Each agent's
critic conditions on o_{i,t} rather than s_t.

Corresponds to Table 2 row:  IPPO: reward=63.4±4.1, VR=12.1%

Compared with LagrangianCTDE:
  - No centralised critic (decentralised V_i(o_i) instead of V(s))
  - No Lagrange multipliers (unconstrained optimisation)
  - No cost-aware advantage (standard Â^R_t only)

Uses PPOTrainer from src.algorithms.ppo with critic=own_critic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.environment.disaster_env import DisasterEnv
from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.algorithms.ppo import PPOTrainer

logger = logging.getLogger(__name__)


class IPPOAgent:
    """
    Independent PPO: each agent has its own actor + decentralised critic.

    The decentralised critic V_i(o_{i,t}) conditions on local observations
    only. There are no constraint heads and no Lagrange multipliers.

    Parameters
    ----------
    obs_dim, n_actions, n_agents — environment dimensions
    lr             : learning rate   (default 3e-4)
    gamma          : discount factor (default 0.99)
    gae_lambda     : GAE λ           (default 0.95)
    clip_epsilon   : PPO ε           (default 0.2)
    entropy_coef   : H[π] weight     (default 0.01)
    mini_batches   : gradient steps  (default 4)
    device         : compute device
    """

    def __init__(
        self,
        obs_dim: int = 12,
        n_actions: int = 4,
        n_agents: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        mini_batches: int = 4,
        hidden_actor: int = 256,
        hidden_critic: int = 512,
        device: str = "cpu",
    ) -> None:
        self.n_agents = n_agents
        self.device_str = device
        dev = torch.device(device)

        # Independent actor + decentralised critic per agent
        # Critic input is obs_dim (local obs), not state_dim (global state)
        # n_constraint_heads=0: no constraint value estimation (unconstrained)
        self._trainers: Dict[int, PPOTrainer] = {
            i: PPOTrainer(
                actor=ActorNetwork(obs_dim, n_actions, hidden_actor, agent_id=i).to(dev),
                critic=CriticNetwork(
                    state_dim=obs_dim,       # local obs only
                    hidden_size=hidden_critic,
                    n_constraint_heads=0,    # no constraint heads
                ).to(dev),
                agent_id=i,
                lr=lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                entropy_coef=entropy_coef,
                mini_batches=mini_batches,
                device=device,
            )
            for i in range(1, n_agents + 1)
        }

    def act(self, obs_dict: Dict[int, np.ndarray],
            deterministic: bool = False) -> Dict[int, int]:
        actions = {}
        for i in range(1, self.n_agents + 1):
            action, _, _, _ = self._trainers[i].get_action_and_value(
                obs_dict[i], deterministic=deterministic
            )
            actions[i] = action
        return actions

    def train(
        self,
        env: DisasterEnv,
        n_episodes: int = 1000,
        checkpoint_dir: str = "checkpoints/",
        save_interval: int = 100,
    ) -> Dict:
        history: Dict[str, list] = {"rewards": [], "policy_losses": [], "value_losses": []}

        for ep in range(1, n_episodes + 1):
            obs_dict, _ = env.reset(seed=ep)
            total_reward = 0.0

            # ── Rollout collection ─────────────────────────────────────────
            for _ in range(env.episode_length):
                actions: Dict[int, int] = {}
                for i in range(1, self.n_agents + 1):
                    action, log_prob, value, _ = self._trainers[i].get_action_and_value(
                        obs_dict[i], global_state=obs_dict[i]  # local obs as state
                    )
                    actions[i] = action
                    self._trainers[i].store(
                        obs=obs_dict[i], action=action, log_prob=log_prob,
                        reward=0.0,  # filled below
                        value=value, done=False,
                    )

                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                total_reward += reward

                # Overwrite reward in the last-stored transition
                for i in range(1, self.n_agents + 1):
                    self._trainers[i].buffer.rewards[-1] = reward
                    self._trainers[i].buffer.dones[-1] = done

                obs_dict = next_obs
                if done:
                    break

            # ── PPO update (no Lagrange multiplier) ───────────────────────
            ep_pol_losses, ep_val_losses = [], []
            for i in range(1, self.n_agents + 1):
                stats = self._trainers[i].update(
                    next_value=0.0, lagrange_lambda=0.0
                )
                ep_pol_losses.append(stats.get("policy_loss", 0.0))
                ep_val_losses.append(stats.get("value_loss", 0.0))

            history["rewards"].append(total_reward)
            history["policy_losses"].append(float(np.mean(ep_pol_losses)))
            history["value_losses"].append(float(np.mean(ep_val_losses)))

            if ep % 50 == 0:
                logger.info("IPPO ep %d | R=%.2f", ep, total_reward)

            if ep % save_interval == 0:
                self.save(str(Path(checkpoint_dir) / f"ippo_ep{ep}.pt"))

        return history

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {i: {
                "actor":  self._trainers[i].actor.state_dict(),
                "critic": self._trainers[i].critic.state_dict(),
             } for i in range(1, self.n_agents + 1)},
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=True)
        for i in range(1, self.n_agents + 1):
            self._trainers[i].actor.load_state_dict(ckpt[i]["actor"])
            self._trainers[i].critic.load_state_dict(ckpt[i]["critic"])
