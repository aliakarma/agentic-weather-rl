"""
src/algorithms/baselines/mappo.py
==================================
Multi-Agent PPO (MAPPO) baseline.

MAPPO is identical to LagrangianCTDE except it uses a centralised critic
V(s_t) without any Lagrange multipliers or constraint optimisation.
The cost-aware advantage reduces to the standard reward advantage
(Â^L_t = Â^R_t since all λ_i = 0).

Corresponds to Table 2 row:  MAPPO: reward=74.3±3.0, VR=8.9%

Key differences from LagrangianCTDE:
  - λ_i = 0 for all agents (no constraint penalty)
  - CriticNetwork with n_constraint_heads=0 (no cost value heads)
  - No _update_lambdas() step

Uses PPOTrainer with critic=None per agent (shared centralised critic
is updated separately), exactly as in LagrangianCTDE but without the
dual variable machinery.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment.disaster_env import DisasterEnv
from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.algorithms.ppo import PPOTrainer, compute_gae

logger = logging.getLogger(__name__)


class MAPPOAgent:
    """
    MAPPO: centralised critic, decentralised actors, no constraints.

    Parameters
    ----------
    obs_dim, state_dim, n_actions, n_agents — environment dimensions
    lr_actor, lr_critic  : learning rates
    gamma                : discount factor (0.99)
    gae_lambda           : GAE λ (0.95)
    clip_epsilon         : PPO ε (0.2)
    entropy_coef         : H[π] weight (0.01)
    mini_batches         : gradient steps (4)
    """

    def __init__(
        self,
        obs_dim: int = 12,
        state_dim: int = 24,
        n_actions: int = 4,
        n_agents: int = 3,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        mini_batches: int = 4,
        max_grad_norm: float = 0.5,
        hidden_actor: int = 256,
        hidden_critic: int = 512,
        device: str = "cpu",
    ) -> None:
        self.n_agents      = n_agents
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_epsilon  = clip_epsilon
        self.entropy_coef  = entropy_coef
        self.mini_batches  = mini_batches
        self.max_grad_norm = max_grad_norm
        self.device        = torch.device(device)

        # Decentralised actors
        self.actors: Dict[int, ActorNetwork] = {
            i: ActorNetwork(obs_dim, n_actions, hidden_actor, agent_id=i).to(self.device)
            for i in range(1, n_agents + 1)
        }

        # Centralised critic (reward only — no constraint heads)
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_size=hidden_critic,
            n_constraint_heads=0,   # no Lagrange heads: key MAPPO vs LagrangianCTDE diff
        ).to(self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # PPO per-agent trainers (no owned critic)
        self.ppo_trainers: Dict[int, PPOTrainer] = {
            i: PPOTrainer(
                actor=self.actors[i],
                critic=None,
                agent_id=i,
                lr=lr_actor,
                clip_epsilon=clip_epsilon,
                gamma=gamma,
                gae_lambda=gae_lambda,
                entropy_coef=entropy_coef,
                mini_batches=mini_batches,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            for i in range(1, n_agents + 1)
        }

    # -----------------------------------------------------------------------

    def act(self, obs_dict: Dict[int, np.ndarray],
            deterministic: bool = False) -> Dict[int, int]:
        actions = {}
        for i in range(1, self.n_agents + 1):
            obs_t = torch.tensor(obs_dict[i], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                a, _, _ = self.actors[i].act(obs_t, deterministic=deterministic)
            actions[i] = int(a.item())
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
            traj: Dict = {i: {"obs": [], "actions": [], "log_probs": [],
                               "values": [], "dones": []} for i in range(1, self.n_agents + 1)}
            rewards_ep: List[float] = []
            states_ep: List[np.ndarray] = []

            # ── Collect trajectory ─────────────────────────────────────────
            for _ in range(env.episode_length):
                global_state = env.get_global_state()
                state_t = torch.tensor(global_state, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)

                with torch.no_grad():
                    value_t, _ = self.critic(state_t)
                shared_val = float(value_t.item())

                actions = {}
                for i in range(1, self.n_agents + 1):
                    obs_t = torch.tensor(obs_dict[i], dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        a, lp, _ = self.actors[i].act(obs_t)
                    actions[i] = int(a.item())
                    traj[i]["obs"].append(obs_dict[i].copy())
                    traj[i]["actions"].append(int(a.item()))
                    traj[i]["log_probs"].append(float(lp.item()))
                    traj[i]["values"].append(shared_val)

                next_obs, reward, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated
                rewards_ep.append(reward)
                states_ep.append(global_state)

                for i in range(1, self.n_agents + 1):
                    traj[i]["dones"].append(float(done))

                obs_dict = next_obs
                total_reward += reward
                if done:
                    break

            T = len(rewards_ep)
            dev = self.device

            rewards_t = torch.tensor(rewards_ep, dtype=torch.float32, device=dev)
            states_t  = torch.tensor(np.array(states_ep), dtype=torch.float32, device=dev)

            # ── GAE and PPO update per agent ───────────────────────────────
            ep_pol_losses, ep_val_losses = [], []
            for i in range(1, self.n_agents + 1):
                values_t = torch.tensor(traj[i]["values"], dtype=torch.float32, device=dev)
                dones_t  = torch.tensor(traj[i]["dones"],  dtype=torch.float32, device=dev)
                obs_t    = torch.tensor(np.array(traj[i]["obs"]), dtype=torch.float32, device=dev)
                acts_t   = torch.tensor(traj[i]["actions"], dtype=torch.long, device=dev)
                logp_t   = torch.tensor(traj[i]["log_probs"], dtype=torch.float32, device=dev)

                adv, ret = compute_gae(rewards_t, values_t, dones_t, 0.0,
                                       self.gamma, self.gae_lambda)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                mini_batch_size = max(T // self.mini_batches, 1)
                for _ in range(self.mini_batches):
                    idx = torch.randperm(T, device=dev)[:mini_batch_size]
                    lp_new, entropy = self.actors[i].evaluate_actions(obs_t[idx], acts_t[idx])
                    ratio = torch.exp(lp_new - logp_t[idx])
                    surr  = torch.min(
                        ratio * adv[idx],
                        torch.clamp(ratio, 1 - self.clip_epsilon,
                                           1 + self.clip_epsilon) * adv[idx]
                    )
                    loss = -surr.mean() - self.entropy_coef * entropy.mean()
                    self.ppo_trainers[i].actor_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.ppo_trainers[i].actor_optimizer.step()
                    ep_pol_losses.append(loss.item())

            # ── Centralised critic update ──────────────────────────────────
            ret_shared = torch.stack([
                compute_gae(rewards_t,
                            torch.tensor(traj[i]["values"], dtype=torch.float32, device=dev),
                            torch.tensor(traj[i]["dones"],  dtype=torch.float32, device=dev),
                            0.0, self.gamma, self.gae_lambda)[1]
                for i in range(1, self.n_agents + 1)
            ], dim=1).mean(dim=1, keepdim=True)

            mini_batch_size = max(T // self.mini_batches, 1)
            for _ in range(self.mini_batches):
                idx = torch.randperm(T, device=dev)[:mini_batch_size]
                v_loss = self.critic.value_loss(states_t[idx], ret_shared[idx])
                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                ep_val_losses.append(v_loss.item())

            history["rewards"].append(total_reward)
            history["policy_losses"].append(float(np.mean(ep_pol_losses)))
            history["value_losses"].append(float(np.mean(ep_val_losses)))

            if ep % 50 == 0:
                logger.info("MAPPO ep %d | R=%.2f", ep, total_reward)
            if ep % save_interval == 0:
                self.save(str(Path(checkpoint_dir) / f"mappo_ep{ep}.pt"))

        return history

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actors": {i: self.actors[i].state_dict() for i in range(1, self.n_agents + 1)},
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=True)
        for i in range(1, self.n_agents + 1):
            self.actors[i].load_state_dict(ckpt["actors"][i])
        self.critic.load_state_dict(ckpt["critic"])
