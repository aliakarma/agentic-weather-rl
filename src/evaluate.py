"""Evaluation utilities for MARL disaster response."""
import pickle
from pathlib import Path
from typing import Optional
import numpy as np

from src.environment.disaster_env import DisasterEnv
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig


def load_agent_from_checkpoint(checkpoint_path, algo="lagrangian_ctde", device="cpu"):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    config_dict = ckpt.get("config", {})
    valid = set(LagrangianCTDEConfig.__dataclass_fields__)
    config = LagrangianCTDEConfig(**{k: v for k, v in config_dict.items() if k in valid})
    agent = LagrangianCTDE(config=config, device=device)
    if "state_dict" in ckpt:
        agent.load_state_dict(ckpt["state_dict"])
    print(f"✓ Loaded checkpoint  algo={algo}  device={device}")
    return agent


def evaluate(agent, env, n_episodes=20, seed_offset=10000, deterministic=True,
             output_dir=None, algo_name="lagrangian_ctde"):
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Removed calibration to preserve scientific validity and unbiased evaluation.
    assert not hasattr(agent, "_calibration_enabled")

    rewards, violation_rates = [], []
    damage_prevented_rates = []
    damage_after_rates = []
    noise_levels = []
    action_levels = {1: [], 2: [], 3: []}

    for ep in range(n_episodes):
        seed = seed_offset + ep
        obs_dict, _ = env.reset(seed=seed)
        # Reset agent RNG if available
        if hasattr(agent, '_rng'):
            agent._rng = np.random.default_rng(seed)

        ep_reward, ep_violations, ep_steps = 0.0, 0, 0
        ep_damage_prevented, ep_damage_after = 0.0, 0.0
        ep_action_sum = {1: 0.0, 2: 0.0, 3: 0.0}

        for _ in range(env.episode_length):
            action_dict = agent.get_actions(obs_dict, deterministic=deterministic)
            obs_dict, reward, terminated, truncated, info = env.step(action_dict)
            ep_reward += reward
            ep_violations += int(info.get("violation", 0))
            ep_damage_prevented += float(info.get("damage_prevented", 0.0))
            ep_damage_after += float(info.get("damage_after", 0.0))
            for i in range(1, 4):
                ep_action_sum[i] += float(action_dict.get(i, 0))
            ep_steps += 1
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        violation_rates.append(ep_violations / max(ep_steps, 1))
        damage_prevented_rates.append(ep_damage_prevented / max(ep_steps, 1))
        damage_after_rates.append(ep_damage_after / max(ep_steps, 1))
        noise_levels.append(float(getattr(env, "observation_noise", 0.0)))
        for i in range(1, 4):
            action_levels[i].append(ep_action_sum[i] / max(ep_steps, 1))

        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:6.2f}  "
              f"viol={ep_violations/ep_steps:.3f}  "
              f"dmg_prev={damage_prevented_rates[-1]:.3f}  "
              f"dmg_after={damage_after_rates[-1]:.3f}",
              flush=True)

    metrics = {
        "cumulative_reward_mean": float(np.mean(rewards)),
        "cumulative_reward_std":  float(np.std(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std":  float(np.std(rewards)),
        "violation_rate_mean": float(np.mean(violation_rates)),
        "violation_rate_std":  float(np.std(violation_rates)),
        "avg_damage_prevented_mean": float(np.mean(damage_prevented_rates)),
        "avg_damage_prevented_std":  float(np.std(damage_prevented_rates)),
        "avg_damage_after_mean": float(np.mean(damage_after_rates)),
        "avg_damage_after_std":  float(np.std(damage_after_rates)),
        "observation_noise_mean": float(np.mean(noise_levels)),
    }

    agent_action_means = np.array([np.mean(action_levels[i]) for i in range(1, 4)], dtype=np.float64)
    jain_num = float(np.sum(agent_action_means) ** 2)
    jain_den = float(3.0 * np.sum(agent_action_means ** 2) + 1e-8)
    metrics["fairness_action_balance_std"] = float(np.std(agent_action_means))
    metrics["fairness_jain_action_index"] = float(jain_num / jain_den)
    for i in range(1, 4):
        metrics[f"agent_{i}_action_mean"] = float(np.mean(action_levels[i]))

    # Robustness metric under stronger observation noise.
    base_noise = float(getattr(env, "observation_noise", 0.0))
    robust_noise = max(base_noise * 2.0, base_noise + 0.03)
    robust_rewards = []
    robust_prevented = []

    for ep in range(n_episodes):
        robust_env = DisasterEnv(seed=seed_offset + ep + 500_000,
                                 observation_noise=robust_noise,
                                 hazard_rate=float(getattr(env, "hazard_rate", 0.22)))
        obs_dict, _ = robust_env.reset(seed=seed_offset + ep + 500_000)
        if hasattr(agent, '_rng'):
            agent._rng = np.random.default_rng(seed_offset + ep + 500_000)

        ep_reward = 0.0
        ep_damage_prevented = 0.0
        for _ in range(robust_env.episode_length):
            action_dict = agent.get_actions(obs_dict, deterministic=deterministic)
            obs_dict, reward, terminated, truncated, info = robust_env.step(action_dict)
            ep_reward += reward
            ep_damage_prevented += float(info.get("damage_prevented", 0.0))
            if terminated or truncated:
                break
        robust_rewards.append(ep_reward)
        robust_prevented.append(ep_damage_prevented / max(robust_env.episode_length, 1))

    metrics["robustness_noise_level"] = robust_noise
    metrics["robustness_reward_mean"] = float(np.mean(robust_rewards))
    metrics["robustness_damage_prevented_mean"] = float(np.mean(robust_prevented))
    metrics["robustness_reward_delta"] = float(metrics["robustness_reward_mean"] - metrics["reward_mean"])
    metrics["robustness_reward_ratio"] = float(
        metrics["robustness_reward_mean"] / (metrics["reward_mean"] + 1e-8)
    )

    if output_dir:
        np.save(Path(output_dir) / f"{algo_name}_metrics.npy", metrics)
    return metrics
