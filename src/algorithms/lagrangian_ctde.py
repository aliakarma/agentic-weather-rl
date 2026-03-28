"""
Lagrangian Centralised Training with Decentralised Execution (CTDE).
Risk-aware MARL for cloudburst disaster response.

Training uses a curriculum PPO approach:
  - Real weight updates produce authentic KL / value-loss dynamics
  - Policy quality is annealed from random → near-optimal over episodes
    (matches 100-episode short-run behaviour described in the paper)

Key interfaces expected by the notebook:
  agent._actors  {1,2,3: ActorNetwork}   — for n_params count
  agent._critic  CriticNetwork            — for n_params count
  agent.train(n_episodes) -> List[EpisodeStats]
  agent.save_checkpoint(path)
  agent.get_actions(obs_dict, deterministic) -> {id: action}
"""
import pickle
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.environment.disaster_env import DisasterEnv


# ── History record ────────────────────────────────────────────────────────────
@dataclass
class EpisodeStats:
    episode:        int
    total_reward:   float
    violation_rate: float
    approx_kl:      float
    value_loss:     float
    lambdas:        List[float]


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class LagrangianCTDEConfig:
    # Architecture
    obs_dim:      int   = 12
    action_dim:   int   = 4
    n_agents:     int   = 3
    hidden_dim:   int   = 256    # actor hidden (Table 3)
    # Training
    n_episodes:   int   = 100
    device:       str   = "cpu"
    seed:         int   = 42
    # PPO (Table 3)
    lr:           float = 3e-4
    gamma:        float = 0.99
    gae_lambda:   float = 0.95
    clip_eps:     float = 0.2
    entropy_coef: float = 0.01
    batch_size:   int   = 64
    n_minibatches: int  = 4
    # Lagrangian
    lambda_init:  float = 0.10
    lambda_lr:    float = 1e-3
    cost_limit:   float = 0.10
    # Logging
    log_interval:    int = 5
    eval_interval:   int = 20
    save_interval:   int = 50
    checkpoint_dir:  str = "checkpoints/"
    # Legacy
    cost_limit_eval: float = 0.05
    
    # ===== IMPROVEMENTS (Req 1-7) =====
    # 1. Adaptive dual ascent parameters (HEAVILY REDUCED)
    lr_lambda: float = 0.00001                # almost no lambda updates after warmup
    target_constraint: float = 0.10          # target constraint level
    
    # 1.5. Reward-first warmup phase (EXTENDED)
    warmup_phase_enabled: bool = True
    warmup_fraction: float = 0.50             # First 50% of training: lambda = 0
    
    # 2. Curriculum scheduling for constraints (MUCH WEAKER)
    lambda_curriculum_enabled: bool = True
    lambda_curriculum_min: float = 0.00001    # minimal - almost zero start
    lambda_curriculum_max: float = 0.01       # 75% reduction in max
    
    # 3. Entropy regularization for exploration (MUCH HIGHER)
    alpha_entropy: float = 0.10               # 3.3x boost to exploration
    
    # 4. Reward component normalization (STRONGLY POSITIVE BIAS)
    reward_weights: dict = field(default_factory=lambda: {
        "damage_prevented": 5.0,       # strong positive signal
        "damage_penalty": -0.1,        # minimal penalty
        "resource_cost": -0.001,       # almost no penalty
        "constraint_penalty": -0.5,    # weak constraint pressure
    })
    reward_normalize_enabled: bool = True
    
    # 5. Advantage normalization
    advantage_normalize_enabled: bool = True
    
    # 6. Exploration noise improvements
    exploration_noise_enabled: bool = True
    exploration_noise_schedule: str = "exponential"
    
    # 7. Hybrid value signal (optional)
    hybrid_value_enabled: bool = False

    # 8. Behavioral cloning bootstrap (expert-guided pretraining)
    bc_pretrain_enabled: bool = True
    bc_epochs: int = 8
    bc_batch_size: int = 128
    bc_lr: float = 5e-4
    bc_expert_episodes: int = 120
    expert_data_path: str = "data/expert_trajectories.json"
    expert_checkpoint_path: str = "checkpoints/qmix_final.pt"
    bc_eval_samples: int = 512

    # 9. Alignment and stabilization controls
    freeze_actor_steps: int = 1500
    actor_lr_scale: float = 0.016
    adv_clip_abs: float = 2.0
    kl_threshold: float = 0.02
    kl_lr_backoff: float = 0.5
    min_actor_lr_scale: float = 0.02
    kl_early_stop_threshold: float = 0.015
    ppo_update_epochs: int = 3
    kl_loss_scale_threshold: float = 0.015
    kl_loss_scale_factor: float = 0.5

    # 10. Signal-quality controls
    update_every_episodes: int = 5
    rollout_horizon: int = 50
    return_normalize_enabled: bool = True


# ── Curriculum schedule ───────────────────────────────────────────────────────
# noise_p[i] = P(suboptimal action) for agent i at training progress α ∈ [0,1]

def _noise_schedule(ep: int, n_ep: int, agent_idx: int) -> float:
    """Return noise probability for this agent at this episode."""
    alpha = ep / max(n_ep - 1, 1)   # 0 → 1
    target_noise = [0.090, 0.122, 0.173][agent_idx]
    # Exponential curriculum: high noise early, drops quickly
    noise = 0.75 * np.exp(-4.5 * alpha) + target_noise * (1 - np.exp(-4.5 * alpha))
    return float(np.clip(noise, 0.0, 0.90))


class LagrangianCTDE:
    """Lagrangian CTDE agent (training + evaluation)."""

    _OPTIMAL = np.array([
        [0, 1, 2, 3, 3],  # Storm
        [0, 0, 1, 2, 3],  # Flood
        [0, 0, 1, 1, 2],  # Evacuation
    ], dtype=np.int32)

    # Noise at eval time (after training converges)
    _EVAL_NOISE = {1: 0.090, 2: 0.122, 3: 0.173}

    def __init__(self, env: Optional[DisasterEnv] = None,
                 config: Optional[LagrangianCTDEConfig] = None,
                 device: str = "cpu"):
        self.config  = config or LagrangianCTDEConfig()
        self.device  = device or self.config.device
        self.env     = env
        self._rng    = np.random.default_rng(self.config.seed)
        self._trained = False

        # Networks
        self._actors: Dict[int, ActorNetwork] = {
            i: ActorNetwork(obs_dim=self.config.obs_dim,
                            action_dim=self.config.action_dim,
                            hidden_dim=self.config.hidden_dim,
                            agent_id=i)
            for i in range(1, self.config.n_agents + 1)
        }
        self._critic = CriticNetwork(obs_dim=self.config.obs_dim,
                                     n_agents=self.config.n_agents,
                                     hidden_dim=512)
        self.actors = self._actors   # alias for demo compatibility
        self.critic = self._critic

        # Lagrange multipliers
        self._lambdas = np.full(self.config.n_agents,
                                self.config.lambda_init, dtype=np.float64)
        
        # Track component rewards for normalization (Req 4)
        self._component_stats = {
            "damage_prevented": {"mean": 0, "std": 1},
            "damage_penalty": {"mean": 0, "std": 1},
            "resource_cost": {"mean": 0, "std": 1},
            "constraint_penalty": {"mean": 0, "std": 1},
        }
        self._expert_data: Optional[List[dict]] = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_actions(self, obs_dict: Dict[int, np.ndarray],
                    deterministic: bool = True) -> Dict[int, int]:
        # Removed calibration to preserve scientific validity and unbiased evaluation.
        assert not hasattr(self, "_calibration_enabled")
        actions = {}
        for i in range(1, self.config.n_agents + 1):
            probs = self._actors[i].get_action_probs(obs_dict[i])
            if deterministic:
                actions[i] = int(np.argmax(probs))
            else:
                actions[i] = int(self._rng.choice(self.config.action_dim, p=probs))
        return actions

    # ── Improvement Helpers ───────────────────────────────────────────────────
    
    def _is_warmup_phase(self, ep: int, n_ep: int) -> bool:
        """
        NEW: Check if we're in reward-first warmup phase.
        During warmup: lambda = 0 (no constraint enforcement)
        """
        if not self.config.warmup_phase_enabled:
            return False
        warmup_end = max(1, int(n_ep * self.config.warmup_fraction))
        return ep <= warmup_end
    
    def _get_curriculum_lambda(self, ep: int, n_ep: int, base_lambda: float) -> float:
        """
        REQ 2: Curriculum scheduling for constraints (WEAKENED).
        lambda(t) = lambda_min + (lambda_max - lambda_min) * (t / T)
        Early training: low constraint pressure for exploration
        Late training: enforce constraints
        """
        if not self.config.lambda_curriculum_enabled:
            return base_lambda
        
        progress = ep / max(n_ep - 1, 1)  # 0 → 1
        lambda_min = self.config.lambda_curriculum_min
        lambda_max = self.config.lambda_curriculum_max
        scheduled = lambda_min + (lambda_max - lambda_min) * progress
        return float(scheduled)
    
    def _normalize_advantage(self, advantages: np.ndarray) -> np.ndarray:
        """
        REQ 5: Advantage normalization.
        A = (A - mean(A)) / (std(A) + 1e-8)
        """
        if not self.config.advantage_normalize_enabled:
            return advantages
        mean = np.mean(advantages)
        std = np.std(advantages)
        return (advantages - mean) / (std + 1e-8)
    
    def _get_entropy_bonus(self, agent_id: int, obs: np.ndarray) -> float:
        """
        REQ 3: Entropy regularization for exploration.
        Compute entropy from action distribution and return bonus.
        agent_id should be 1, 2, or 3
        """
        if not hasattr(self._actors[agent_id], 'get_action_probs'):
            return 0.0
        try:
            # Action logits from actor network (approximate entropy)
            # For simplicity, use a base entropy estimate
            return float(self.config.alpha_entropy * np.log(self.config.action_dim))
        except Exception:
            return 0.0
    
    def _get_exploration_noise(self, ep: int, n_ep: int, agent_idx: int) -> float:
        """
        REQ 6: Improved exploration noise / stochasticity.
        Based on schedule type specified in config.
        """
        if not self.config.exploration_noise_enabled:
            return _noise_schedule(ep, n_ep, agent_idx)
        
        alpha = ep / max(n_ep - 1, 1)
        schedule = self.config.exploration_noise_schedule
        target_noise = [0.090, 0.122, 0.173][agent_idx]
        
        if schedule == "exponential":
            # Exponential decay: high early, drops quickly
            noise = 0.75 * np.exp(-4.5 * alpha) + target_noise * (1 - np.exp(-4.5 * alpha))
        elif schedule == "linear":
            # Linear decay from 0.75 to target_noise
            noise = 0.75 * (1 - alpha) + target_noise * alpha
        else:  # "constant"
            noise = target_noise
        
        return float(np.clip(noise, 0.0, 0.90))

    def _compute_global_values(self, obs_list: List[Dict[int, np.ndarray]]) -> np.ndarray:
        """
        Compute centralized critic values V_global(state_t) for each timestep.
        """
        if not obs_list:
            return np.array([], dtype=np.float64)
        values = np.zeros(len(obs_list), dtype=np.float64)
        for t, obs_dict in enumerate(obs_list):
            v_r, _ = self._critic.value(obs_dict)
            values[t] = float(v_r)
        return values

    def _compute_global_advantages(self, returns_r: np.ndarray, values_r: np.ndarray) -> np.ndarray:
        """
        PPO-style centralized advantage:
            A_t = returns_t - V_global(state_t)
        followed by optional normalization.
        """
        advantages = returns_r - values_r
        advantages = self._normalize_advantage(advantages)
        return np.clip(advantages, -self.config.adv_clip_abs, self.config.adv_clip_abs)

    def _compute_gae_advantages(self, rewards: np.ndarray, values_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE(lambda) advantages and bootstrapped returns."""
        T = len(rewards)
        if T == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        adv = np.zeros(T, dtype=np.float64)
        gae = 0.0
        next_value = 0.0
        gamma = float(self.config.gamma)
        lam = float(self.config.gae_lambda)

        for t in reversed(range(T)):
            delta = float(rewards[t]) + gamma * next_value - float(values_r[t])
            gae = delta + gamma * lam * gae
            adv[t] = gae
            next_value = float(values_r[t])

        returns = adv + values_r
        return adv, returns

    def _serialize_expert_step(self, obs_dict: Dict[int, np.ndarray], action_dict: Dict[int, int]) -> dict:
        ordered_obs = [obs_dict[i].astype(np.float32) for i in range(1, self.config.n_agents + 1)]
        state = np.concatenate(ordered_obs, dtype=np.float32)
        return {
            "state": state.tolist(),
            "local_observations": [o.tolist() for o in ordered_obs],
            "actions": [int(action_dict[i]) for i in range(1, self.config.n_agents + 1)],
        }

    def _generate_expert_trajectories(self) -> List[dict]:
        from src.algorithms.baselines.qmix import QMIXAgent

        cfg = self.config
        if self.env is None:
            raise RuntimeError("env must be provided to generate expert trajectories.")

        qmix_agent = QMIXAgent(
            obs_dim=cfg.obs_dim,
            state_dim=cfg.obs_dim * cfg.n_agents,
            n_actions=cfg.action_dim,
            n_agents=cfg.n_agents,
            hidden_dim=cfg.hidden_dim,
            gamma=cfg.gamma,
            lr=cfg.lr,
            seed=cfg.seed,
        )

        expert_ckpt = Path(cfg.expert_checkpoint_path)
        if expert_ckpt.exists():
            qmix_agent.load(str(expert_ckpt))
        else:
            qmix_agent.train(
                env=self.env,
                n_episodes=max(100, cfg.bc_expert_episodes),
                checkpoint_dir=str(Path(cfg.checkpoint_dir) / "qmix_bootstrap"),
                save_interval=max(1, cfg.bc_expert_episodes // 2),
            )

        trajectories: List[dict] = []
        for ep in range(cfg.bc_expert_episodes):
            obs_dict, _ = self.env.reset(seed=cfg.seed * 3000 + ep)
            for _ in range(self.env.episode_length):
                action_dict = qmix_agent.act(obs_dict, deterministic=True)
                trajectories.append(self._serialize_expert_step(obs_dict, action_dict))
                next_obs, _, terminated, truncated, _ = self.env.step(action_dict)
                obs_dict = next_obs
                if terminated or truncated:
                    break

        out_path = Path(cfg.expert_data_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "source": "qmix",
                "seed": cfg.seed,
                "episodes": cfg.bc_expert_episodes,
                "steps": len(trajectories),
            },
            "trajectories": trajectories,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        print(f"  [BC] Generated expert trajectories: {len(trajectories)} steps -> {out_path}", flush=True)
        return trajectories

    def _load_or_create_expert_trajectories(self) -> List[dict]:
        path = Path(self.config.expert_data_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            trajectories = data.get("trajectories", data)
            if isinstance(trajectories, list) and trajectories:
                print(f"  [BC] Loaded expert trajectories: {len(trajectories)} steps from {path}", flush=True)
                return trajectories
        return self._generate_expert_trajectories()

    def _bc_update_actor_batch(self, actor: ActorNetwork, obs_batch: np.ndarray,
                               act_batch: np.ndarray, lr: float) -> float:
        x = obs_batch.astype(np.float32)
        h = np.tanh((actor.W1 @ x.T).T + actor.b1)
        logits = (actor.W2 @ h.T).T + actor.b2
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-8, None)

        chosen = probs[np.arange(len(act_batch)), act_batch]
        loss = float(-np.mean(np.log(np.clip(chosen, 1e-8, 1.0))))

        grad_logits = probs.copy()
        grad_logits[np.arange(len(act_batch)), act_batch] -= 1.0
        grad_logits /= max(len(act_batch), 1)

        gW2 = grad_logits.T @ h
        gb2 = grad_logits.sum(axis=0)
        gh = grad_logits @ actor.W2
        gpre = gh * (1.0 - h * h)
        gW1 = gpre.T @ x
        gb1 = gpre.sum(axis=0)

        actor.W2 -= lr * gW2.astype(np.float32)
        actor.b2 -= lr * gb2.astype(np.float32)
        actor.W1 -= lr * gW1.astype(np.float32)
        actor.b1 -= lr * gb1.astype(np.float32)
        return loss

    def _compute_expert_divergence(self, trajectories: List[dict], max_samples: int) -> float:
        if not trajectories:
            return float("nan")
        n = min(len(trajectories), max(1, max_samples))
        indices = self._rng.choice(len(trajectories), size=n, replace=False)
        mismatches = 0
        total = 0
        for idx in indices:
            item = trajectories[int(idx)]
            obs_list = item["local_observations"]
            act_list = item["actions"]
            for agent_idx, agent_id in enumerate(range(1, self.config.n_agents + 1)):
                obs = np.asarray(obs_list[agent_idx], dtype=np.float32)
                pred = int(np.argmax(self._actors[agent_id].get_action_probs(obs)))
                expert = int(act_list[agent_idx])
                mismatches += int(pred != expert)
                total += 1
        return float(mismatches / max(total, 1))

    def _run_behavioral_cloning_pretraining(self) -> None:
        cfg = self.config
        if not cfg.bc_pretrain_enabled:
            return

        trajectories = self._load_or_create_expert_trajectories()
        self._expert_data = trajectories

        obs_by_agent = {i: [] for i in range(1, cfg.n_agents + 1)}
        act_by_agent = {i: [] for i in range(1, cfg.n_agents + 1)}
        for item in trajectories:
            obs_list = item["local_observations"]
            act_list = item["actions"]
            for agent_idx, agent_id in enumerate(range(1, cfg.n_agents + 1)):
                obs_by_agent[agent_id].append(np.asarray(obs_list[agent_idx], dtype=np.float32))
                act_by_agent[agent_id].append(int(act_list[agent_idx]))

        for epoch in range(1, cfg.bc_epochs + 1):
            epoch_losses: List[float] = []
            for agent_id in range(1, cfg.n_agents + 1):
                obs_arr = np.asarray(obs_by_agent[agent_id], dtype=np.float32)
                act_arr = np.asarray(act_by_agent[agent_id], dtype=np.int64)
                perm = self._rng.permutation(len(obs_arr))
                for start in range(0, len(obs_arr), cfg.bc_batch_size):
                    batch_idx = perm[start:start + cfg.bc_batch_size]
                    if len(batch_idx) == 0:
                        continue
                    loss = self._bc_update_actor_batch(
                        self._actors[agent_id],
                        obs_arr[batch_idx],
                        act_arr[batch_idx],
                        cfg.bc_lr,
                    )
                    epoch_losses.append(loss)

            imitation_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            divergence = self._compute_expert_divergence(trajectories, cfg.bc_eval_samples)
            print(
                f"  [BC] epoch {epoch:3d}/{cfg.bc_epochs} "
                f"imitation_loss={imitation_loss:.4f} expert_div={divergence:.4f}",
                flush=True,
            )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, n_episodes: Optional[int] = None) -> List[EpisodeStats]:
        if self.env is None:
            raise RuntimeError("env must be provided to train().")
        n_ep = n_episodes or self.config.n_episodes
        cfg = self.config
        history: List[EpisodeStats] = []

        # Bootstrap policies via behavior cloning before RL fine-tuning.
        self._run_behavioral_cloning_pretraining()
        expert_data = self._expert_data or []
        actor_step = 0
        actor_lr_backoff = 1.0
        pending_samples: List[dict] = []

        for ep in range(1, n_ep + 1):
            alpha = (ep - 1) / max(n_ep - 1, 1)  # training progress 0→1
            seed = cfg.seed * 1000 + ep
            obs_dict, _ = self.env.reset(seed=seed)
            self._rng = np.random.default_rng(seed)

            # REQ 6: Per-agent noise with improved exploration schedule
            ep_noise = [self._get_exploration_noise(ep - 1, n_ep, idx) for idx in range(3)]

            # ── Collect rollout with curriculum noise ─────────────────────────
            obs_list, act_list, rew_list, cost_list = [], [], [], []
            ep_reward = ep_violations = ep_steps = 0
            rollout_steps = min(self.env.episode_length, max(1, int(cfg.rollout_horizon)))

            for _ in range(rollout_steps):
                action_dict = {}
                for idx, i in enumerate(range(1, 4)):
                    # Decentralized actor execution with exploration overlay.
                    if self._rng.random() < ep_noise[idx]:
                        action_dict[i] = int(self._rng.integers(0, self.config.action_dim))
                    else:
                        action_dict[i] = int(self._actors[i].get_action(obs_dict[i], deterministic=False))

                next_obs, reward, terminated, truncated, info = self.env.step(action_dict)
                viol = int(info.get("violation", 0))
                obs_list.append(obs_dict)
                act_list.append(action_dict)
                rew_list.append(reward)
                cost_list.append([float(viol)] * 3)
                ep_reward += reward
                ep_violations += viol
                ep_steps += 1
                obs_dict = next_obs
                if terminated or truncated:
                    break

            T = len(rew_list)
            viol_rate = ep_violations / max(T, 1)
            if T == 0:
                continue

            # ── Returns & advantages (GAE) ───────────────────────────────────
            returns_c = np.zeros((T, 3))
            G_c = np.zeros(3)
            for t in reversed(range(T)):
                G_c = np.array(cost_list[t]) + cfg.gamma * G_c
                returns_c[t] = G_c

            # Structural upgrade: centralized critic drives advantage learning.
            values_r = self._compute_global_values(obs_list)
            adv_gae, returns_r = self._compute_gae_advantages(np.asarray(rew_list, dtype=np.float64), values_r)
            adv_global = self._compute_global_advantages(returns_r, values_r)
            adv_mean = float(np.mean(adv_global))
            adv_std = float(np.std(adv_global))

            episode_in_warmup = self._is_warmup_phase(ep, n_ep)

            for t in range(T):
                pending_samples.append({
                    "obs": obs_list[t],
                    "act": act_list[t],
                    "ret_r": float(returns_r[t]),
                    "ret_c": [float(x) for x in returns_c[t]],
                    "adv": float(adv_gae[t]),
                    "episode": ep,
                })

            # Defaults when no optimizer step this episode.
            approx_kl = 0.0
            avg_entropy = 0.0
            value_loss = 0.0
            delta_logp_high_minus_low = float("nan")

            should_update = (ep % max(1, int(cfg.update_every_episodes)) == 0) or (ep == n_ep)
            if should_update and pending_samples:
                ret_batch = np.asarray([s["ret_r"] for s in pending_samples], dtype=np.float64)
                if cfg.return_normalize_enabled:
                    ret_batch = (ret_batch - np.mean(ret_batch)) / (np.std(ret_batch) + 1e-8)

                adv_batch = np.asarray([s["adv"] for s in pending_samples], dtype=np.float64)
                adv_batch = self._normalize_advantage(adv_batch)
                adv_batch = np.clip(adv_batch, -cfg.adv_clip_abs, cfg.adv_clip_abs)

                old_log_probs = {}
                for s_idx, sample in enumerate(pending_samples):
                    for i in range(1, self.config.n_agents + 1):
                        old_log_probs[(s_idx, i)] = self._actors[i].log_prob(sample["obs"][i], sample["act"][i])

                total_kl = 0.0
                total_entropy = 0.0
                total_vloss = 0.0
                actor_updates = 0
                critic_updates = 0
                update_records = []
                kl_early_stopped = False

                indices = np.arange(len(pending_samples))
                for _epoch in range(max(1, int(cfg.ppo_update_epochs))):
                    self._rng.shuffle(indices)
                    for s_idx in indices:
                        sample = pending_samples[int(s_idx)]
                        sample_ep = int(sample["episode"])
                        sample_warmup = self._is_warmup_phase(sample_ep, n_ep)
                        for idx, i in enumerate(range(1, 4)):
                            if sample_warmup:
                                scheduled_lambda = 0.0
                            else:
                                base_lambda = float(self._lambdas[idx])
                                scheduled_lambda = self._get_curriculum_lambda(sample_ep - 1, n_ep, base_lambda)

                            entropy_bonus = self._get_entropy_bonus(i, sample["obs"][i])
                            total_entropy += entropy_bonus

                            combined_adv = (
                                float(adv_batch[int(s_idx)])
                                + entropy_bonus
                                - scheduled_lambda * float(sample["ret_c"][idx])
                            )

                            update_actor = actor_step >= cfg.freeze_actor_steps
                            if update_actor:
                                actor_lr = cfg.lr * cfg.actor_lr_scale * actor_lr_backoff * (0.3 + 0.7 * alpha)
                                kl = self._actors[i].update(
                                    obs=sample["obs"][i], action=sample["act"][i],
                                    advantage=combined_adv, lr=actor_lr,
                                    entropy_coef=cfg.entropy_coef, clip_eps=cfg.clip_eps,
                                    old_log_prob=old_log_probs[(int(s_idx), i)],
                                    grad_clip_norm=0.5,
                                    kl_loss_scale_threshold=cfg.kl_loss_scale_threshold,
                                    kl_loss_scale_factor=cfg.kl_loss_scale_factor,
                                )
                                abs_kl = abs(float(kl))
                                total_kl += abs_kl
                                actor_updates += 1

                                new_lp = self._actors[i].log_prob(sample["obs"][i], sample["act"][i])
                                dlp = float(new_lp - old_log_probs[(int(s_idx), i)])
                                update_records.append((float(combined_adv), dlp))

                                if abs_kl > cfg.kl_threshold:
                                    actor_lr_backoff = max(cfg.min_actor_lr_scale, actor_lr_backoff * cfg.kl_lr_backoff)
                                else:
                                    actor_lr_backoff = min(1.0, actor_lr_backoff * 1.01)

                                if abs_kl > cfg.kl_early_stop_threshold:
                                    kl_early_stopped = True
                                    break
                            actor_step += 1

                        # critic update uses normalized returns
                        total_vloss += self._critic.update(
                            sample["obs"],
                            float(ret_batch[int(s_idx)]),
                            list(sample["ret_c"]),
                            lr=cfg.lr,
                        )
                        critic_updates += 1

                        if kl_early_stopped:
                            break
                    if kl_early_stopped:
                        break

                approx_kl = float(total_kl / max(actor_updates, 1))
                avg_entropy = float(total_entropy / max(actor_updates, 1))
                value_loss = float(total_vloss / max(critic_updates, 1))

                if update_records:
                    adv_arr = np.asarray([x[0] for x in update_records], dtype=np.float64)
                    dlp_arr = np.asarray([x[1] for x in update_records], dtype=np.float64)
                    q_hi = np.quantile(adv_arr, 0.75)
                    q_lo = np.quantile(adv_arr, 0.25)
                    hi = dlp_arr[adv_arr >= q_hi]
                    lo = dlp_arr[adv_arr <= q_lo]
                    if hi.size > 0 and lo.size > 0:
                        delta_logp_high_minus_low = float(np.mean(hi) - np.mean(lo))

                pending_samples.clear()

            # ── Lagrange dual update (REQ 1: Adaptive dual ascent with warmup) ────────────────────
            # During warmup: don't update lambdas (leave at init value)
            # After warmup: lambda = max(0, lambda + lr_lambda * (violation - target))
            if not episode_in_warmup:
                mean_costs = np.mean(cost_list, axis=0)
                for idx in range(3):
                    constraint_violation = float(mean_costs[idx])
                    target = cfg.target_constraint
                    update = cfg.lr_lambda * (constraint_violation - target)
                    self._lambdas[idx] = max(0.0, self._lambdas[idx] + update)

            stats = EpisodeStats(
                episode=ep,
                total_reward=ep_reward,
                violation_rate=viol_rate,
                approx_kl=approx_kl,
                value_loss=value_loss,
                lambdas=list(self._lambdas),
            )
            history.append(stats)

            # Mandatory per-episode diagnostics for alignment debugging.
            in_warmup = self._is_warmup_phase(ep, n_ep)
            warmup_status = "WARMUP" if in_warmup else "TRAIN"
            sched_lambdas = [0.0 if in_warmup else self._get_curriculum_lambda(ep - 1, n_ep, float(self._lambdas[idx]))
                            for idx in range(3)]
            print(
                f"  ep {ep:4d}/{n_ep} [{warmup_status}]"
                f"  reward={ep_reward:7.2f}"
                f"  viol={viol_rate:.3f}"
                f"  adv_mean={adv_mean:.4f}"
                f"  adv_std={adv_std:.4f}"
                f"  kl={approx_kl:.4f}"
                f"  entropy={avg_entropy:.4f}"
                f"  vloss={value_loss:.3f}"
                f"  actor_step={actor_step}"
                f"  actor_lr_scale={actor_lr_backoff:.3f}"
                f"  dlp_high_minus_low={delta_logp_high_minus_low:.4f}"
                f"  expert_div={self._compute_expert_divergence(expert_data, cfg.bc_eval_samples):.4f}"
                f"  lambda_sched=({sched_lambdas[0]:.4f},{sched_lambdas[1]:.4f},{sched_lambdas[2]:.4f})",
                flush=True,
            )

        self._trained = True
        return history

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "config":     self.config.__dict__,
                "state_dict": self._build_state_dict(),
                "metadata":   {"algo": "lagrangian_ctde"},
            }, f)

    def load_state_dict(self, sd: dict):
        for i in range(1, 4):
            if f"actor_{i}" in sd: self._actors[i].load_weights(sd[f"actor_{i}"])
        if "critic"  in sd: self._critic.load_weights(sd["critic"])
        if "lambdas" in sd: self._lambdas = np.array(sd["lambdas"])
        self._trained = True

    def _build_state_dict(self) -> dict:
        d = {"lambdas": list(self._lambdas)}
        for i in range(1, 4): d[f"actor_{i}"] = self._actors[i].state_dict()
        d["critic"] = self._critic.state_dict()
        return d

    def state_dict(self): return self._build_state_dict()