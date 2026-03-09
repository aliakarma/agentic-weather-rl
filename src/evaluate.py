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

    rewards, violation_rates = [], []
    da_per_agent = {1: [], 2: [], 3: []}

    for ep in range(n_episodes):
        seed = seed_offset + ep
        obs_dict, _ = env.reset(seed=seed)
        # Reset agent RNG if available
        if hasattr(agent, '_rng'):
            agent._rng = np.random.default_rng(seed)

        ep_reward, ep_violations, ep_steps = 0.0, 0, 0
        correct = {1: 0, 2: 0, 3: 0}

        for _ in range(env.episode_length):
            action_dict = agent.get_actions(obs_dict, deterministic=deterministic)
            obs_dict, reward, terminated, truncated, info = env.step(action_dict)
            ep_reward += reward
            ep_violations += int(info.get("violation", 0))
            ep_steps += 1
            obs_sev = int(info.get("obs_severity", info.get("severity", 0)))
            for idx, i in enumerate(range(1, 4)):
                opt = int(DisasterEnv.OPTIMAL_ACTIONS[idx, obs_sev])
                if action_dict[i] == opt:
                    correct[i] += 1
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        violation_rates.append(ep_violations / max(ep_steps, 1))
        for i in range(1, 4):
            da_per_agent[i].append(correct[i] / max(ep_steps, 1))

        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:6.2f}  "
              f"viol={ep_violations/ep_steps:.3f}  "
              f"DA=({da_per_agent[1][-1]:.2f},{da_per_agent[2][-1]:.2f},{da_per_agent[3][-1]:.2f})",
              flush=True)

    metrics = {
        "reward_mean": float(np.mean(rewards)),
        "reward_std":  float(np.std(rewards)),
        "violation_rate_mean": float(np.mean(violation_rates)),
        "violation_rate_std":  float(np.std(violation_rates)),
    }
    for i in range(1, 4):
        metrics[f"decision_accuracy_{i}_mean"] = float(np.mean(da_per_agent[i]))
        metrics[f"decision_accuracy_{i}_std"]  = float(np.std(da_per_agent[i]))
    if output_dir:
        np.save(Path(output_dir) / f"{algo_name}_metrics.npy", metrics)
    return metrics
