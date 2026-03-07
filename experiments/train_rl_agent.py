"""
RL Agent Training Entry Point
================================
Purpose:
    Top-level script to train the PPO disaster response agent. Delegates
    to rl_agent/training.py with CLI argument parsing.

Usage:
    python experiments/train_rl_agent.py --timesteps 100000 --seed 42
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PPO disaster response agent.")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel training environments.")
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--tensorboard_log", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
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
