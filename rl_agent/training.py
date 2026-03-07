"""
Reinforcement Learning Training Module
========================================
Purpose:
    Training loop for the PPO-based DisasterResponseAgent. Handles environment
    construction, agent initialization, training, checkpointing, and result
    plotting. Supports reproducible experiments via fixed seeds.

Output:
    results/ppo_agent.zip           — trained model weights
    results/reward_curve.png        — training reward progression plot
    results/rl_training_log.csv     — per-episode reward log

Example usage:
    python rl_agent/training.py --timesteps 100000 --seed 42
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from rl_agent.environment import DisasterResponseEnv
from rl_agent.agent_ppo import DisasterResponseAgent


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def smooth_curve(values: list, window: int = 20) -> np.ndarray:
    """
    Apply a moving average to a list of scalar values.

    Args:
        values: Raw per-episode values (e.g., rewards).
        window: Smoothing window size.

    Returns:
        Smoothed numpy array of the same length.
    """
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_reward_curve(
    rewards: list,
    save_path: str,
    title: str = "PPO Training Reward Curve",
    window: int = 20,
) -> None:
    """
    Plot and save the training reward curve with a smoothed overlay.

    Args:
        rewards:   List of per-episode cumulative rewards.
        save_path: Output file path for the PNG image.
        title:     Plot title.
        window:    Smoothing window for the moving average overlay.
    """
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = smooth_curve(rewards, window=window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(episodes, smoothed, color="steelblue", linewidth=2, label=f"Smoothed (w={window})")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Reward curve saved to {save_path}")


def save_reward_log(rewards: list, save_path: str) -> None:
    """
    Save per-episode reward values to a CSV file.

    Args:
        rewards:   List of cumulative episode rewards.
        save_path: Output CSV path.
    """
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "cumulative_reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i + 1, r])
    print(f"[INFO] Reward log saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    total_timesteps: int = 100_000,
    seed: int = 42,
    n_envs: int = 1,
    eval_freq: int = 5_000,
    output_dir: str = "results",
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
) -> DisasterResponseAgent:
    """
    Full training pipeline for the PPO disaster response agent.

    Steps:
        1. Create training and evaluation environments.
        2. Initialize the PPO agent.
        3. Train for the specified number of timesteps.
        4. Save model checkpoint.
        5. Plot and save reward curve.
        6. Return the trained agent.

    Args:
        total_timesteps: Total environment interaction steps.
        seed:            Random seed for reproducibility.
        n_envs:          Number of parallel environments (SB3 VecEnv).
        eval_freq:       Evaluation frequency in steps.
        output_dir:      Directory to save results and checkpoints.
        tensorboard_log: Optional path for TensorBoard logs.
        verbose:         SB3 verbosity level.

    Returns:
        Trained DisasterResponseAgent instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "ppo_agent")

    # ----- Create environments -----
    if n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        env = make_vec_env(
            lambda: DisasterResponseEnv(max_steps=50, seed=seed),
            n_envs=n_envs,
            seed=seed,
        )
    else:
        env = DisasterResponseEnv(max_steps=50, seed=seed)

    eval_env = DisasterResponseEnv(max_steps=50, seed=seed + 100)

    # ----- Initialize agent -----
    print(f"[INFO] Initializing PPO agent (seed={seed}, timesteps={total_timesteps})")
    agent = DisasterResponseAgent(
        env=env,
        seed=seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )

    # ----- Train -----
    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=eval_freq,
        save_best_path=os.path.join(output_dir, "best_model"),
    )

    # ----- Save model -----
    agent.save(model_save_path)

    # ----- Plot reward curve -----
    rewards = agent.get_reward_history()
    if rewards:
        plot_reward_curve(
            rewards=rewards,
            save_path=os.path.join(output_dir, "reward_curve.png"),
        )
        save_reward_log(
            rewards=rewards,
            save_path=os.path.join(output_dir, "rl_training_log.csv"),
        )
    else:
        print("[WARNING] No episode reward data recorded (env may use VecEnv wrapping).")

    # ----- Final evaluation -----
    final_eval_env = DisasterResponseEnv(max_steps=50, seed=seed + 999)
    eval_stats = agent.evaluate(final_eval_env, n_episodes=20)
    print(f"\n[RESULTS] Final evaluation over 20 episodes:")
    for k, v in eval_stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    return agent


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PPO disaster response agent."
    )
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel training envs (default: 1).")
    parser.add_argument("--eval_freq", type=int, default=5_000,
                        help="Evaluation frequency in steps (default: 5000).")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save outputs (default: results/).")
    parser.add_argument("--tensorboard_log", type=str, default=None,
                        help="TensorBoard log directory (optional).")
    parser.add_argument("--verbose", type=int, default=1,
                        help="SB3 verbosity: 0=silent, 1=info, 2=debug.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        output_dir=args.output_dir,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
    )
