"""
src/algorithms/lagrangian_ctde.py
==================================
Lagrangian Centralised Training with Decentralised Execution (CTDE-PPO).

Implements Algorithm 1 from the paper:

  "We solve the constrained optimisation problem (Eq. 5) via a
   primal-dual approach. At each iteration we update the primal
   variables (actor parameters θ_i) using PPO with the cost-aware
   advantage Â^L_t = Â^R_t − Σ_i λ_i Â^{C_i}_t, and the dual
   variables (Lagrange multipliers λ_i) via a projected sub-gradient
   ascent step on the constraint violations."

Architecture (Figure 1 of paper):
  Layer 1  :  ViT-B/16 multi-modal encoder  →  φ_t ∈ ℝ^128      (optional)
  Layer 2  :  3 × ActorNetwork (decentralised)  +  CriticNetwork (centralised)
  Layer 3  :  Orchestration interface                             (see orchestration.py)

Training loop (one iteration = one episode):
  Step 1 — Collect trajectory: run all three agents in the environment,
            storing (o_{i,t}, a_{i,t}, log π_old, r_t, C_{i,t}, s_t) per step.
  Step 2 — Compute advantages: reward GAE Â^R_t and per-agent cost GAE Â^{C_i}_t.
  Step 3 — Compute cost penalties: mean constraint violations J_{C_i}.
  Step 4 — Update dual variables: λ_i ← max(0, λ_i + α_λ (J_{C_i} − d_i)).
  Step 5 — PPO optimisation: update each actor θ_i with cost-aware advantage
            Â^L_t and update the shared centralised critic ψ.

The critic is centralised (conditions on global state s_t) during training.
At execution time only the decentralised actors are used — the critic is
discarded. This satisfies the CTDE requirement of the C-Dec-POMDP formulation.

Paper hyperparameters (Table 3):
    lr_actor       = 3e-4
    lr_critic      = 3e-4
    lr_lambda      = 1e-3   (α_λ in the paper)
    gamma          = 0.99
    gae_lambda     = 0.95
    clip_epsilon   = 0.2
    entropy_coef   = 0.01
    mini_batches   = 4
    constraint_d   = 0.10   (d_i for all agents)
    n_episodes     = 1000
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment.disaster_env import DisasterEnv
from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.algorithms.ppo import PPOTrainer, RolloutBuffer, compute_gae

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LagrangianCTDEConfig:
    """
    Hyper-parameter configuration for LagrangianCTDE.

    All defaults match Table 3 of the paper.
    Can be overridden via configs/training.yaml.
    """
    # ── Network ───────────────────────────────────────────────────────────
    obs_dim: int        = 12    # per-agent local observation dimension
    state_dim: int      = 24    # global state dimension (synthetic mode)
    n_actions: int      = 4     # {0: no-op, 1: warn, 2: deploy, 3: evacuate}
    n_agents: int       = 3
    actor_hidden: int   = 256   # Table 3
    critic_hidden: int  = 512   # Table 3

    # ── Optimisation ──────────────────────────────────────────────────────
    lr_actor: float     = 3e-4
    lr_critic: float    = 3e-4
    lr_lambda: float    = 1e-3  # Lagrange multiplier step size α_λ
    gamma: float        = 0.99
    gae_lambda: float   = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    mini_batches: int   = 4

    # ── Constraints ───────────────────────────────────────────────────────
    constraint_d: float = 0.10  # d_i threshold for all agents (Eq. 5)
    lambda_init: float  = 0.0   # initial dual variable value
    lambda_max: float   = 10.0  # projection upper bound for λ_i

    # ── Training ──────────────────────────────────────────────────────────
    n_episodes: int     = 1000
    eval_interval: int  = 50    # evaluate every N episodes
    save_interval: int  = 100
    checkpoint_dir: str = "checkpoints/"
    log_interval: int   = 10

    # ── Device ────────────────────────────────────────────────────────────
    device: str         = "cpu"
    seed: int           = 42


# ---------------------------------------------------------------------------
# Training statistics container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Per-episode statistics accumulated during training."""
    episode: int
    total_reward: float
    violation_rate: float
    lambdas: List[float]        # λ_i values for each agent
    mean_constraint_costs: List[float]
    policy_losses: List[float]  # per-agent
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    duration_sec: float


# ---------------------------------------------------------------------------
# Main algorithm class
# ---------------------------------------------------------------------------

class LagrangianCTDE:
    """
    Lagrangian CTDE-PPO: centralised training, decentralised execution.

    Owns:
      - Three decentralised ActorNetworks {π_θ1, π_θ2, π_θ3}
      - One centralised CriticNetwork V_ψ (with 3 constraint heads)
      - Three Lagrange multipliers {λ_1, λ_2, λ_3}
      - Three PPOTrainer instances (one per actor)
      - One Adam optimiser for the centralised critic

    Parameters
    ----------
    env : DisasterEnv
        The disaster response simulation environment.
    config : LagrangianCTDEConfig | None
        Training configuration. Uses defaults (Table 3) if None.
    """

    def __init__(
        self,
        env: DisasterEnv,
        config: Optional[LagrangianCTDEConfig] = None,
    ) -> None:
        self.env = env
        self.cfg = config or LagrangianCTDEConfig()
        self.device = torch.device(self.cfg.device)

        # ── Decentralised actors ──────────────────────────────────────────
        # One ActorNetwork per agent; each agent trained with its own PPOTrainer.
        self.actors: Dict[int, ActorNetwork] = {
            i: ActorNetwork(
                obs_dim=self.cfg.obs_dim,
                n_actions=self.cfg.n_actions,
                hidden_size=self.cfg.actor_hidden,
                agent_id=i,
            ).to(self.device)
            for i in range(1, self.cfg.n_agents + 1)
        }

        # ── Centralised critic ────────────────────────────────────────────
        # Conditions on global state s_t during training.
        # n_constraint_heads = n_agents: one cost-value head V^{C_i}(s_t) per agent.
        self.critic = CriticNetwork(
            state_dim=self.cfg.state_dim,
            hidden_size=self.cfg.critic_hidden,
            n_agents=self.cfg.n_agents,
            n_constraint_heads=self.cfg.n_agents,
        ).to(self.device)

        # Shared critic optimiser (all value heads updated jointly)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_critic
        )

        # ── Lagrange multipliers λ_i ─────────────────────────────────────
        # One per agent. Initialised at lambda_init (default 0.0).
        # Projected to [0, lambda_max] after each dual update.
        # Stored as plain Python floats (not nn.Parameters) because the
        # dual update is a projected sub-gradient step, not back-propagated.
        self.lambdas: Dict[int, float] = {
            i: self.cfg.lambda_init
            for i in range(1, self.cfg.n_agents + 1)
        }

        # ── Per-agent PPO trainers ────────────────────────────────────────
        # PPOTrainer does NOT own the critic here — the centralised critic
        # is updated separately via self.critic_optimizer.
        self.ppo_trainers: Dict[int, PPOTrainer] = {
            i: PPOTrainer(
                actor=self.actors[i],
                critic=None,          # critic is centralised, owned here
                agent_id=i,
                lr=self.cfg.lr_actor,
                clip_epsilon=self.cfg.clip_epsilon,
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
                entropy_coef=self.cfg.entropy_coef,
                max_grad_norm=self.cfg.max_grad_norm,
                mini_batches=self.cfg.mini_batches,
                device=self.cfg.device,
            )
            for i in range(1, self.cfg.n_agents + 1)
        }

        # ── Training state ────────────────────────────────────────────────
        self._episode: int = 0
        self._best_reward: float = -np.inf
        self._history: List[EpisodeStats] = []

        logger.info(
            "LagrangianCTDE initialised — %d actors, centralised critic, "
            "λ_init=%.3f, d_i=%.2f, device=%s",
            self.cfg.n_agents, self.cfg.lambda_init,
            self.cfg.constraint_d, self.cfg.device,
        )
        for i, actor in self.actors.items():
            logger.info("  Actor %d: %s", i, actor)
        logger.info("  Critic:  %s", self.critic)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(self, n_episodes: Optional[int] = None) -> List[EpisodeStats]:
        """
        Run the full Lagrangian CTDE-PPO training loop.

        Implements Algorithm 1 of the paper. For each episode:
          1. Collect a full trajectory of T=100 steps.
          2. Compute reward and cost advantages via GAE.
          3. Compute mean constraint violations J_{C_i} per agent.
          4. Update dual variables λ_i (projected sub-gradient).
          5. Run PPO actor updates with cost-aware advantage Â^L_t.
          6. Run centralised critic update.

        Parameters
        ----------
        n_episodes : int | None
            Number of training episodes. Defaults to cfg.n_episodes.

        Returns
        -------
        history : list[EpisodeStats]
            Per-episode training statistics.
        """
        n_ep = n_episodes if n_episodes is not None else self.cfg.n_episodes

        logger.info("Starting Lagrangian CTDE-PPO training for %d episodes.", n_ep)

        for ep in range(1, n_ep + 1):
            self._episode = ep
            t0 = time.time()

            # ── Step 1: Collect trajectory ────────────────────────────────
            traj = self._collect_trajectory()

            # ── Step 2: Compute reward and cost advantages ─────────────────
            reward_adv, reward_ret, cost_adv, cost_ret = self._compute_advantages(traj)

            # ── Step 3: Compute mean constraint violations J_{C_i} ─────────
            mean_costs = self._compute_mean_costs(traj)

            # ── Step 4: Update dual variables λ_i ─────────────────────────
            self._update_lambdas(mean_costs)

            # ── Step 5 + 6: PPO actor updates + centralised critic update ──
            ppo_stats = self._update_primal(traj, reward_adv, reward_ret,
                                            cost_adv, cost_ret)

            # ── Logging ───────────────────────────────────────────────────
            ep_stats = EpisodeStats(
                episode=ep,
                total_reward=float(sum(traj["rewards"])),
                violation_rate=traj["violation_rate"],
                lambdas=[self.lambdas[i] for i in range(1, self.cfg.n_agents + 1)],
                mean_constraint_costs=mean_costs,
                policy_losses=[
                    ppo_stats[i].get("policy_loss", 0.0)
                    for i in range(1, self.cfg.n_agents + 1)
                ],
                value_loss=ppo_stats.get("value_loss", 0.0),
                entropy=float(np.mean([
                    ppo_stats[i].get("entropy_loss", 0.0)
                    for i in range(1, self.cfg.n_agents + 1)
                ])),
                approx_kl=float(np.mean([
                    ppo_stats[i].get("approx_kl", 0.0)
                    for i in range(1, self.cfg.n_agents + 1)
                ])),
                clip_fraction=float(np.mean([
                    ppo_stats[i].get("clip_fraction", 0.0)
                    for i in range(1, self.cfg.n_agents + 1)
                ])),
                duration_sec=time.time() - t0,
            )
            self._history.append(ep_stats)

            if ep % self.cfg.log_interval == 0:
                self._log_episode(ep_stats)

            if ep % self.cfg.eval_interval == 0:
                eval_reward, eval_vr = self._evaluate_greedy()
                logger.info(
                    "  [Eval ep %d] reward=%.2f  VR=%.4f",
                    ep, eval_reward, eval_vr,
                )

            if ep % self.cfg.save_interval == 0:
                self.save_checkpoint(
                    Path(self.cfg.checkpoint_dir) / f"checkpoint_ep{ep}.pt"
                )

        logger.info(
            "Training complete. Best reward=%.2f  Final λ=%s",
            self._best_reward,
            [f"{self.lambdas[i]:.4f}" for i in range(1, self.cfg.n_agents + 1)],
        )
        return self._history

    # -----------------------------------------------------------------------
    # Step 1 — Trajectory collection
    # -----------------------------------------------------------------------

    def _collect_trajectory(self) -> Dict:
        """
        Run all three agents for one full episode (T=100 steps).

        Returns a trajectory dict containing per-step data for:
          rewards, global_states, dones, violation_rate,
          per_agent observations, actions, log_probs, values,
          cost_rewards, cost_values.
        """
        obs_dict, info = self.env.reset(seed=self.cfg.seed + self._episode)

        # Per-step storage (lists, converted to tensors in _compute_advantages)
        traj: Dict = {
            "rewards":        [],
            "global_states":  [],
            "dones":          [],
            "violation_rate": 0.0,
            # Per-agent dicts
            "agents": {
                i: {
                    "obs":         [],
                    "actions":     [],
                    "log_probs":   [],
                    "values":      [],
                    "cost_rewards": [],
                    "cost_values": [],
                }
                for i in range(1, self.cfg.n_agents + 1)
            },
        }

        for t in range(self.env.episode_length):
            global_state = self.env.get_global_state()
            global_state_t = torch.tensor(
                global_state, dtype=torch.float32, device=self.device
            )

            # Centralised critic: V(s_t) and V^{C_i}(s_t) for all agents
            with torch.no_grad():
                value_t, cost_values_t = self.critic(
                    global_state_t.unsqueeze(0)
                )
            shared_value = float(value_t.item())
            per_agent_cost_values = (
                cost_values_t[0].tolist()
                if cost_values_t is not None
                else [0.0] * self.cfg.n_agents
            )  # length = n_agents

            # Decentralised action selection for each agent
            actions: Dict[int, int] = {}
            for i in range(1, self.cfg.n_agents + 1):
                obs_i = obs_dict[i]
                obs_t = torch.tensor(
                    obs_i, dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    action_t, log_prob_t, _ = self.actors[i].act(
                        obs_t, deterministic=False
                    )
                action_i = int(action_t.item())
                log_prob_i = float(log_prob_t.item())
                cost_val_i = per_agent_cost_values[i - 1]

                actions[i] = action_i

                # Store per-agent transition data
                ag = traj["agents"][i]
                ag["obs"].append(obs_i.copy())
                ag["actions"].append(action_i)
                ag["log_probs"].append(log_prob_i)
                ag["values"].append(shared_value)      # shared V(s_t)
                ag["cost_values"].append(cost_val_i)

            # Environment step
            next_obs_dict, reward, terminated, truncated, step_info = self.env.step(
                actions
            )
            done = terminated or truncated

            # Store shared reward and global state
            traj["rewards"].append(float(reward))
            traj["global_states"].append(global_state.copy())
            traj["dones"].append(float(done))

            # Per-agent cost rewards C_{i,t}
            constraint_costs = step_info.get("constraint_costs", {})
            for i in range(1, self.cfg.n_agents + 1):
                cost_i = float(constraint_costs.get(i, 0.0))
                traj["agents"][i]["cost_rewards"].append(cost_i)

            obs_dict = next_obs_dict

            if done:
                break

        # Episode-level violation rate (for logging and evaluation)
        ep_stats = self.env.get_episode_stats()
        traj["violation_rate"] = ep_stats["violation_rate"]

        return traj

    # -----------------------------------------------------------------------
    # Step 2 — Advantage computation
    # -----------------------------------------------------------------------

    def _compute_advantages(
        self, traj: Dict
    ) -> Tuple[
        Dict[int, torch.Tensor],  # reward advantages per agent
        Dict[int, torch.Tensor],  # reward returns per agent
        Dict[int, torch.Tensor],  # cost advantages per agent
        Dict[int, torch.Tensor],  # cost returns per agent
    ]:
        """
        Compute reward and cost GAE advantages for all three agents.

        The reward advantage Â^R_t is shared across agents (all agents
        receive the same team reward r_t). The cost advantage Â^{C_i}_t
        is per-agent (each agent has its own constraint stream).

        Implements the GAE recursion (paper notation, Eq. 7):
            δ^R_t     = r_t + γV(s_{t+1}) - V(s_t)
            Â^R_t     = δ^R_t + γλ * Â^R_{t+1}
            δ^{C_i}_t = C_{i,t} + γV^{C_i}(s_{t+1}) - V^{C_i}(s_t)
            Â^{C_i}_t = δ^{C_i}_t + γλ * Â^{C_i}_{t+1}
        """
        T = len(traj["rewards"])
        dev = self.device

        rewards_t = torch.tensor(traj["rewards"], dtype=torch.float32, device=dev)
        dones_t   = torch.tensor(traj["dones"],   dtype=torch.float32, device=dev)

        # Bootstrap value: V(s_T) from the terminal global state
        # (0.0 if the episode ended before T steps)
        last_done = traj["dones"][-1]
        next_value      = 0.0
        next_cost_values = [0.0] * self.cfg.n_agents

        if not last_done and len(traj["global_states"]) > 0:
            last_state = torch.tensor(
                traj["global_states"][-1], dtype=torch.float32, device=dev
            ).unsqueeze(0)
            with torch.no_grad():
                v_boot, cv_boot = self.critic(last_state)
            next_value = float(v_boot.item())
            if cv_boot is not None:
                next_cost_values = cv_boot[0].tolist()

        reward_adv: Dict[int, torch.Tensor] = {}
        reward_ret: Dict[int, torch.Tensor] = {}
        cost_adv:   Dict[int, torch.Tensor] = {}
        cost_ret:   Dict[int, torch.Tensor] = {}

        for i in range(1, self.cfg.n_agents + 1):
            ag = traj["agents"][i]
            values_t   = torch.tensor(ag["values"],       dtype=torch.float32, device=dev)
            cost_val_t = torch.tensor(ag["cost_values"],  dtype=torch.float32, device=dev)
            cost_rew_t = torch.tensor(ag["cost_rewards"], dtype=torch.float32, device=dev)

            # Shared reward advantage (same r_t for all agents, CTDE)
            r_adv, r_ret = compute_gae(
                rewards=rewards_t,
                values=values_t,
                dones=dones_t,
                next_value=next_value,
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

            # Per-agent cost advantage
            c_adv, c_ret = compute_gae(
                rewards=cost_rew_t,
                values=cost_val_t,
                dones=dones_t,
                next_value=next_cost_values[i - 1],
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

            reward_adv[i] = r_adv
            reward_ret[i] = r_ret
            cost_adv[i]   = c_adv
            cost_ret[i]   = c_ret

        return reward_adv, reward_ret, cost_adv, cost_ret

    # -----------------------------------------------------------------------
    # Step 3 — Mean constraint violations J_{C_i}
    # -----------------------------------------------------------------------

    def _compute_mean_costs(self, traj: Dict) -> List[float]:
        """
        Compute J_{C_i} = E[Σ_t C_{i,t}] for each agent.

        This is the empirical constraint violation used in the dual update.
        Compared against d_i to determine whether to tighten λ_i.
        """
        mean_costs: List[float] = []
        for i in range(1, self.cfg.n_agents + 1):
            costs = traj["agents"][i]["cost_rewards"]
            mean_costs.append(float(np.mean(costs)) if costs else 0.0)
        return mean_costs

    # -----------------------------------------------------------------------
    # Step 4 — Dual variable update
    # -----------------------------------------------------------------------

    def _update_lambdas(self, mean_costs: List[float]) -> None:
        """
        Projected sub-gradient ascent on the Lagrangian dual variables.

        Update rule (Eq. 6 of paper):
            λ_i ← max(0, λ_i + α_λ * (J_{C_i} − d_i))

        The projection max(0, ·) ensures dual feasibility.
        An additional upper-bound projection to lambda_max is applied for
        numerical stability.

        When J_{C_i} > d_i (constraint violated), λ_i increases, which
        penalises future unsafe actions more heavily via the cost-aware
        advantage Â^L_t = Â^R_t − λ_i * Â^{C_i}_t.

        When J_{C_i} < d_i (constraint slack), λ_i decreases toward 0.
        """
        for i in range(1, self.cfg.n_agents + 1):
            j_ci = mean_costs[i - 1]
            violation = j_ci - self.cfg.constraint_d

            new_lambda = self.lambdas[i] + self.cfg.lr_lambda * violation

            # Dual projection: λ_i ∈ [0, lambda_max]
            self.lambdas[i] = float(
                np.clip(new_lambda, 0.0, self.cfg.lambda_max)
            )

    # -----------------------------------------------------------------------
    # Step 5+6 — Primal update (actors + centralised critic)
    # -----------------------------------------------------------------------

    def _update_primal(
        self,
        traj: Dict,
        reward_adv: Dict[int, torch.Tensor],
        reward_ret: Dict[int, torch.Tensor],
        cost_adv:   Dict[int, torch.Tensor],
        cost_ret:   Dict[int, torch.Tensor],
    ) -> Dict:
        """
        Update all actor parameters θ_i and the centralised critic ψ.

        For each agent i:
          a) Compute Â^L_t = Â^R_t − λ_i * Â^{C_i}_t  (cost-aware advantage)
          b) Normalise Â^L_t
          c) Run mini_batches PPO gradient steps on L^CLIP(θ_i) + H[π_θi]

        Centralised critic update:
          d) Minimise Bellman residual for V(s_t) and V^{C_i}(s_t) jointly.

        Returns a flat stats dict with per-agent losses and shared critic loss.
        """
        T = len(traj["rewards"])
        dev = self.device
        mini_batch_size = max(T // self.cfg.mini_batches, 1)

        # ── Per-agent actor updates ───────────────────────────────────────
        per_agent_stats: Dict[int, Dict[str, float]] = {}

        for i in range(1, self.cfg.n_agents + 1):
            ag = traj["agents"][i]
            lambda_i = self.lambdas[i]

            obs_t       = torch.tensor(np.array(ag["obs"]),       dtype=torch.float32, device=dev)
            actions_t   = torch.tensor(ag["actions"],             dtype=torch.long,    device=dev)
            log_probs_t = torch.tensor(ag["log_probs"],           dtype=torch.float32, device=dev)

            r_adv = reward_adv[i]
            c_adv = cost_adv[i]

            # Cost-aware advantage Â^L_t = Â^R_t − λ_i * Â^{C_i}_t
            lagrange_adv = r_adv - lambda_i * c_adv

            # Normalise for stable gradient magnitudes
            lagrange_adv = (
                (lagrange_adv - lagrange_adv.mean())
                / (lagrange_adv.std() + 1e-8)
            )

            actor_i = self.actors[i]
            optimizer_i = self.ppo_trainers[i].actor_optimizer

            ep_stats: Dict[str, list] = {
                "policy_loss": [], "entropy_loss": [],
                "approx_kl": [], "clip_fraction": [],
            }

            for _ in range(self.cfg.mini_batches):
                idx = torch.randperm(T, device=dev)[:mini_batch_size]

                mb_obs        = obs_t[idx]
                mb_actions    = actions_t[idx]
                mb_logp_old   = log_probs_t[idx]
                mb_adv        = lagrange_adv[idx]

                log_probs_new, entropy = actor_i.evaluate_actions(
                    mb_obs, mb_actions
                )

                # Importance ratio ρ_t
                ratio = torch.exp(log_probs_new - mb_logp_old)

                # Clipped surrogate (Eq. 8)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_epsilon,
                           1.0 + self.cfg.clip_epsilon
                ) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                entropy_loss = -torch.mean(entropy)
                total_loss = policy_loss + self.cfg.entropy_coef * entropy_loss

                optimizer_i.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    actor_i.parameters(), self.cfg.max_grad_norm
                )
                optimizer_i.step()

                with torch.no_grad():
                    kl = torch.mean((ratio - 1.0) - torch.log(ratio + 1e-8)).item()
                    cf = torch.mean(
                        (torch.abs(ratio - 1.0) > self.cfg.clip_epsilon).float()
                    ).item()

                ep_stats["policy_loss"].append(policy_loss.item())
                ep_stats["entropy_loss"].append(entropy_loss.item())
                ep_stats["approx_kl"].append(kl)
                ep_stats["clip_fraction"].append(cf)

            per_agent_stats[i] = {k: float(np.mean(v)) for k, v in ep_stats.items()}

        # ── Centralised critic update ─────────────────────────────────────
        #
        # The critic is updated on the joint trajectory: reward value head
        # V(s_t) and all n_agents cost heads V^{C_i}(s_t) are optimised
        # simultaneously from the same global-state inputs.
        #
        # cost_returns shape: (T, n_agents) — stack all agents' cost returns.
        global_states_t = torch.tensor(
            np.array(traj["global_states"]), dtype=torch.float32, device=dev
        )

        # TD(λ) reward targets: average across agents (shared reward)
        reward_returns_t = torch.stack(
            [reward_ret[i] for i in range(1, self.cfg.n_agents + 1)], dim=1
        ).mean(dim=1, keepdim=True)  # (T, 1)

        # Per-agent cost returns stacked: (T, n_agents)
        cost_returns_t = torch.stack(
            [cost_ret[i] for i in range(1, self.cfg.n_agents + 1)], dim=1
        )

        critic_loss_accum: list = []

        for _ in range(self.cfg.mini_batches):
            idx = torch.randperm(T, device=dev)[:mini_batch_size]

            mb_states      = global_states_t[idx]
            mb_ret_reward  = reward_returns_t[idx]
            mb_ret_cost    = cost_returns_t[idx]

            v_loss = self.critic.value_loss(mb_states, mb_ret_reward)
            c_loss = self.critic.cost_value_loss(mb_states, mb_ret_cost)
            total_critic_loss = v_loss + c_loss

            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.max_grad_norm
            )
            self.critic_optimizer.step()

            critic_loss_accum.append(total_critic_loss.item())

        # Track best reward for checkpoint saving
        ep_reward = sum(traj["rewards"])
        if ep_reward > self._best_reward:
            self._best_reward = ep_reward
            self.save_checkpoint(
                Path(self.cfg.checkpoint_dir) / "marl_policy.pt"
            )

        # Flatten and return all stats
        flat_stats: Dict = {
            "value_loss": float(np.mean(critic_loss_accum)),
        }
        for i, s in per_agent_stats.items():
            flat_stats[i] = s

        return flat_stats

    # -----------------------------------------------------------------------
    # Greedy evaluation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_greedy(self, n_episodes: int = 5) -> Tuple[float, float]:
        """
        Run n_episodes with deterministic (greedy) actor policies.
        Returns (mean_reward, mean_violation_rate).
        """
        rewards_list, vr_list = [], []

        for ep in range(n_episodes):
            obs_dict, _ = self.env.reset(seed=10_000 + ep)
            total_reward = 0.0

            for _ in range(self.env.episode_length):
                actions: Dict[int, int] = {}
                for i in range(1, self.cfg.n_agents + 1):
                    obs_t = torch.tensor(
                        obs_dict[i], dtype=torch.float32, device=self.device
                    )
                    a, _, _ = self.actors[i].act(obs_t, deterministic=True)
                    actions[i] = int(a.item())

                obs_dict, reward, terminated, truncated, _ = self.env.step(actions)
                total_reward += reward
                if terminated or truncated:
                    break

            stats = self.env.get_episode_stats()
            rewards_list.append(total_reward)
            vr_list.append(stats["violation_rate"])

        return float(np.mean(rewards_list)), float(np.mean(vr_list))

    # -----------------------------------------------------------------------
    # Checkpoint saving / loading
    # -----------------------------------------------------------------------

    def save_checkpoint(self, path: Path | str) -> None:
        """Save all model parameters and training state to a checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": self._episode,
            "best_reward": self._best_reward,
            "lambdas": dict(self.lambdas),
            "critic_state_dict": self.critic.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "config": self.cfg.__dict__,
            "actors": {
                i: self.actors[i].state_dict()
                for i in range(1, self.cfg.n_agents + 1)
            },
            "actor_optimizers": {
                i: self.ppo_trainers[i].actor_optimizer.state_dict()
                for i in range(1, self.cfg.n_agents + 1)
            },
        }
        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved to %s (ep=%d)", path, self._episode)

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore all model parameters and training state from a checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        self._episode      = ckpt["episode"]
        self._best_reward  = ckpt["best_reward"]
        self.lambdas       = ckpt["lambdas"]

        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])

        for i in range(1, self.cfg.n_agents + 1):
            self.actors[i].load_state_dict(ckpt["actors"][i])
            self.ppo_trainers[i].actor_optimizer.load_state_dict(
                ckpt["actor_optimizers"][i]
            )

        logger.info("Checkpoint loaded from %s (ep=%d)", path, self._episode)

    # -----------------------------------------------------------------------
    # Inference-only interface (Layer 3 / Orchestrator handoff)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def get_actions(
        self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True
    ) -> Dict[int, int]:
        """
        Return deterministic (or stochastic) joint action for all agents.

        Called by the Orchestrator at deployment time.
        Only decentralised actors are used — the centralised critic
        is NOT evaluated here, satisfying the CTDE execution constraint.

        Parameters
        ----------
        obs_dict : dict[int, np.ndarray]
            Local observations keyed by agent_id {1, 2, 3}.
        deterministic : bool
            If True, use argmax policy (default for deployment).

        Returns
        -------
        actions : dict[int, int]
            Joint action keyed by agent_id.
        """
        actions: Dict[int, int] = {}
        for i in range(1, self.cfg.n_agents + 1):
            obs_t = torch.tensor(
                obs_dict[i], dtype=torch.float32, device=self.device
            )
            action_t, _, _ = self.actors[i].act(obs_t, deterministic=deterministic)
            actions[i] = int(action_t.item())
        return actions

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def current_lambdas(self) -> Dict[int, float]:
        """Return the current Lagrange multiplier values."""
        return dict(self.lambdas)

    @property
    def history(self) -> List[EpisodeStats]:
        """Return per-episode training history."""
        return list(self._history)

    def _log_episode(self, s: EpisodeStats) -> None:
        lambdas_str = " ".join(f"λ{i}={s.lambdas[i-1]:.4f}" for i in range(1, 4))
        costs_str   = " ".join(
            f"J_{i}={s.mean_constraint_costs[i-1]:.4f}" for i in range(1, 4)
        )
        logger.info(
            "Ep %4d | R=%-7.2f VR=%.4f | %s | %s | "
            "v_loss=%.4f kl=%.4f cf=%.4f | %.2fs",
            s.episode, s.total_reward, s.violation_rate,
            lambdas_str, costs_str,
            s.value_loss, s.approx_kl, s.clip_fraction, s.duration_sec,
        )
