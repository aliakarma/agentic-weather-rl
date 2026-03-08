"""
src/train.py
=============
Main training entry point for the Lagrangian CTDE-PPO algorithm
and all six baseline methods.

Responsibilities:
  1. Load configuration from YAML (configs/) or CLI arguments
  2. Initialise the DisasterEnv environment
  3. Initialise the chosen algorithm
  4. Run the training loop with periodic checkpointing
  5. Log metrics to stdout and TensorBoard

Usage
-----
    # Train the proposed LagrangianCTDE (default)
    python -m src.train

    # Train with a specific config file
    python -m src.train --config configs/training.yaml

    # Train a baseline
    python -m src.train --algo mappo --episodes 1000

    # Short run for CI / Colab demo
    python -m src.train --algo lagrangian_ctde --episodes 50

Supported --algo values:
    lagrangian_ctde  (default — proposed method)
    heuristic        (rule-based, no training)
    dqn
    ippo
    qmix
    mappo
    cpo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging setup (before any module imports so all loggers are configured)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional dependencies (graceful fallback)
# ---------------------------------------------------------------------------

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.warning("PyYAML not installed; YAML config loading disabled.")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------

from src.environment.disaster_env import DisasterEnv
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return as a flat dict."""
    if not _YAML_AVAILABLE:
        raise ImportError("Install PyYAML: pip install pyyaml")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    # Flatten nested dicts (training.yaml has nested sections)
    flat: Dict[str, Any] = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    return flat


def build_env(cfg: Dict[str, Any], seed: int) -> DisasterEnv:
    """Construct the DisasterEnv from config values."""
    return DisasterEnv(
        grid_size=cfg.get("grid_size", 20),
        n_agents=cfg.get("n_agents", 3),
        episode_length=cfg.get("episode_length", 100),
        hazard_lambda=cfg.get("hazard_lambda", 0.05),
        reward_alpha=cfg.get("reward_alpha", 1.0),
        reward_beta=cfg.get("reward_beta", 0.5),
        reward_eta=cfg.get("reward_eta", 0.3),
        constraint_threshold=cfg.get("constraint_threshold", 0.10),
        seed=seed,
    )


def build_lagrangian_ctde(env: DisasterEnv, cfg: Dict[str, Any]) -> LagrangianCTDE:
    """Construct LagrangianCTDE from merged config dict."""
    algo_cfg = LagrangianCTDEConfig(
        obs_dim=cfg.get("obs_dim", 12),
        state_dim=cfg.get("state_dim", 24),
        n_actions=cfg.get("n_actions", 4),
        n_agents=cfg.get("n_agents", 3),
        actor_hidden=cfg.get("actor_hidden", 256),
        critic_hidden=cfg.get("critic_hidden", 512),
        lr_actor=cfg.get("lr", 3e-4),
        lr_critic=cfg.get("lr", 3e-4),
        lr_lambda=cfg.get("lr_lambda", 1e-3),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_epsilon=cfg.get("clip_epsilon", 0.2),
        entropy_coef=cfg.get("entropy_coef", 0.01),
        mini_batches=cfg.get("mini_batches", 4),
        constraint_d=cfg.get("constraint_threshold", 0.10),
        n_episodes=cfg.get("n_episodes", 1000),
        eval_interval=cfg.get("eval_interval", 50),
        save_interval=cfg.get("save_interval", 100),
        log_interval=cfg.get("log_interval", 10),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/"),
        device=cfg.get("device", "cpu"),
        seed=cfg.get("seed", 42),
    )
    return LagrangianCTDE(env=env, config=algo_cfg)


def build_baseline(algo: str, env: DisasterEnv, cfg: Dict[str, Any]):
    """Construct a baseline agent from name and config dict."""
    shared = dict(
        obs_dim=cfg.get("obs_dim", 12),
        n_actions=cfg.get("n_actions", 4),
        n_agents=cfg.get("n_agents", 3),
        device=cfg.get("device", "cpu"),
    )

    if algo == "heuristic":
        from src.algorithms.baselines.heuristic import HeuristicPolicy
        return HeuristicPolicy()

    if algo == "dqn":
        from src.algorithms.baselines.dqn import DQNAgent
        return DQNAgent(lr=cfg.get("lr", 3e-4), gamma=cfg.get("gamma", 0.99), **shared)

    if algo == "ippo":
        from src.algorithms.baselines.ippo import IPPOAgent
        return IPPOAgent(lr=cfg.get("lr", 3e-4), gamma=cfg.get("gamma", 0.99), **shared)

    if algo == "qmix":
        from src.algorithms.baselines.qmix import QMIXAgent
        return QMIXAgent(
            state_dim=cfg.get("state_dim", 24),
            lr=cfg.get("lr", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            **shared,
        )

    if algo == "mappo":
        from src.algorithms.baselines.mappo import MAPPOAgent
        return MAPPOAgent(
            state_dim=cfg.get("state_dim", 24),
            lr_actor=cfg.get("lr", 3e-4),
            lr_critic=cfg.get("lr", 3e-4),
            gamma=cfg.get("gamma", 0.99),
            **shared,
        )

    if algo == "cpo":
        from src.algorithms.baselines.cpo import CPOAgent
        return CPOAgent(lr_actor=cfg.get("lr", 3e-4), gamma=cfg.get("gamma", 0.99), **shared)

    raise ValueError(
        f"Unknown algorithm '{algo}'. "
        "Choose from: lagrangian_ctde, heuristic, dqn, ippo, qmix, mappo, cpo"
    )


# ---------------------------------------------------------------------------
# TensorBoard logger wrapper
# ---------------------------------------------------------------------------

class MetricLogger:
    """Thin wrapper around SummaryWriter with a stdout fallback."""

    def __init__(self, log_dir: str, algo: str) -> None:
        self.log_dir = Path(log_dir) / algo
        self.writer = None
        if _TB_AVAILABLE:
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info("TensorBoard logging to %s", self.log_dir)
        else:
            logger.info(
                "TensorBoard not available. Install tensorboard for visual logging."
            )

    def log(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_dict(self, tag_prefix: str, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.log(f"{tag_prefix}/{k}", v, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


# ---------------------------------------------------------------------------
# Public train() function
# ---------------------------------------------------------------------------

def train(
    algo: str = "lagrangian_ctde",
    config_path: Optional[str] = None,
    n_episodes: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/",
    log_dir: str = "runs/",
    extra_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train the chosen algorithm on DisasterEnv.

    Parameters
    ----------
    algo : str
        Algorithm name. Default 'lagrangian_ctde'.
    config_path : str | None
        Path to a YAML config file. Defaults are used if None.
    n_episodes : int | None
        Override the number of training episodes.
    device : str
        Compute device ('cpu' or 'cuda').
    seed : int
        Master random seed for reproducibility.
    checkpoint_dir : str
        Directory for saving checkpoints.
    log_dir : str
        Directory for TensorBoard logs.
    extra_cfg : dict | None
        Additional config key-value pairs (override YAML values).

    Returns
    -------
    result : dict
        Training summary with keys: algo, episodes, final_reward,
        final_violation_rate, training_time_sec, checkpoint_path.
    """
    t_start = time.time()

    # ── 1. Load configuration ─────────────────────────────────────────────
    cfg: Dict[str, Any] = {
        "device":          device,
        "seed":            seed,
        "checkpoint_dir":  checkpoint_dir,
        "log_dir":         log_dir,
    }
    if config_path is not None and Path(config_path).exists():
        cfg.update(load_yaml_config(config_path))
        logger.info("Loaded config from %s", config_path)
    if extra_cfg:
        cfg.update(extra_cfg)
    if n_episodes is not None:
        cfg["n_episodes"] = n_episodes

    n_ep = cfg.get("n_episodes", 1000)

    logger.info(
        "=== Training %s for %d episodes | device=%s | seed=%d ===",
        algo.upper(), n_ep, cfg["device"], cfg["seed"],
    )

    # ── 2. Initialise environment ──────────────────────────────────────────
    env = build_env(cfg, seed=cfg["seed"])
    logger.info("Environment: DisasterEnv(grid=%d, T=%d, λ=%.3f)",
                env.grid_size, env.episode_length, env._hazard_gen.hazard_lambda)

    # ── 3. Initialise algorithm ────────────────────────────────────────────
    metric_logger = MetricLogger(cfg.get("log_dir", "runs/"), algo)

    if algo == "lagrangian_ctde":
        agent = build_lagrangian_ctde(env, cfg)

        # Run the built-in training loop (returns List[EpisodeStats])
        history = agent.train(n_episodes=n_ep)

        # Extract per-episode metrics for logging
        for ep_stats in history:
            ep = ep_stats.episode
            metric_logger.log("train/reward",          ep_stats.total_reward,     ep)
            metric_logger.log("train/violation_rate",  ep_stats.violation_rate,   ep)
            metric_logger.log("train/value_loss",      ep_stats.value_loss,       ep)
            metric_logger.log("train/approx_kl",       ep_stats.approx_kl,        ep)
            for i, lam in enumerate(ep_stats.lambdas, 1):
                metric_logger.log(f"train/lambda_{i}", lam, ep)

        final_reward = float(np.mean([s.total_reward    for s in history[-10:]]))
        final_vr     = float(np.mean([s.violation_rate  for s in history[-10:]]))
        ckpt_path    = str(Path(checkpoint_dir) / "marl_policy.pt")

    else:
        # ── Baseline training ──────────────────────────────────────────────
        agent = build_baseline(algo, env, cfg)

        if algo == "heuristic":
            # Heuristic: evaluate directly, no training loop
            logger.info("HeuristicPolicy: no training required, running evaluation...")
            rewards, vrs = _evaluate_baseline_quick(agent, env, n_episodes=20)
            final_reward = float(np.mean(rewards))
            final_vr     = float(np.mean(vrs))
            ckpt_path    = "N/A (heuristic)"
        else:
            history = agent.train(
                env=env,
                n_episodes=n_ep,
                checkpoint_dir=checkpoint_dir,
                save_interval=cfg.get("save_interval", 100),
            )

            # Log from history dict (all baselines return {"rewards": [...]})
            rewards_list = history.get("rewards", [])
            for ep, r in enumerate(rewards_list, 1):
                metric_logger.log("train/reward", r, ep)
            if "losses" in history:
                for ep, l in enumerate(history["losses"], 1):
                    metric_logger.log("train/loss", l, ep)

            final_reward = float(np.mean(rewards_list[-10:])) if rewards_list else 0.0
            final_vr     = float(np.mean(history.get("constraint_violations", [0.0])[-10:]))
            ckpt_path    = str(Path(checkpoint_dir) / f"{algo}_final.pt")
            agent.save(ckpt_path)

    training_time = time.time() - t_start
    metric_logger.close()

    result = {
        "algo":                algo,
        "episodes":            n_ep,
        "final_reward":        final_reward,
        "final_violation_rate": final_vr,
        "training_time_sec":   training_time,
        "checkpoint_path":     ckpt_path,
    }

    logger.info(
        "=== Training complete in %.1fs ===\n"
        "  Algorithm:       %s\n"
        "  Final reward:    %.2f\n"
        "  Final VR:        %.4f\n"
        "  Checkpoint:      %s",
        training_time, algo, final_reward, final_vr, ckpt_path,
    )

    return result


# ---------------------------------------------------------------------------
# Quick evaluation helper (used for heuristic baseline)
# ---------------------------------------------------------------------------

def _evaluate_baseline_quick(
    agent, env: DisasterEnv, n_episodes: int = 20
) -> tuple:
    rewards, vrs = [], []
    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=ep)
        total_reward = 0.0
        for _ in range(env.episode_length):
            actions = agent.act(obs_dict) if hasattr(agent, "act") else agent.act(obs_dict)
            obs_dict, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            if terminated or truncated:
                break
        stats = env.get_episode_stats()
        rewards.append(total_reward)
        vrs.append(stats["violation_rate"])
    return rewards, vrs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MARL agent on the disaster response environment."
    )
    parser.add_argument(
        "--algo",
        default="lagrangian_ctde",
        choices=["lagrangian_ctde", "heuristic", "dqn", "ippo", "qmix", "mappo", "cpo"],
        help="Algorithm to train (default: lagrangian_ctde)",
    )
    parser.add_argument("--config",       default=None,          help="Path to YAML config file")
    parser.add_argument("--episodes",     type=int, default=None, help="Number of training episodes")
    parser.add_argument("--device",       default="cpu",         help="Compute device (cpu/cuda)")
    parser.add_argument("--seed",         type=int, default=42,   help="Random seed")
    parser.add_argument("--checkpoint_dir", default="checkpoints/", help="Checkpoint save directory")
    parser.add_argument("--log_dir",      default="runs/",        help="TensorBoard log directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = train(
        algo=args.algo,
        config_path=args.config,
        n_episodes=args.episodes,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    # Print final summary table
    print("\n" + "=" * 50)
    print(f"  Algorithm        : {result['algo']}")
    print(f"  Episodes trained : {result['episodes']}")
    print(f"  Final reward     : {result['final_reward']:.2f}")
    print(f"  Final VR         : {result['final_violation_rate']:.4f}")
    print(f"  Training time    : {result['training_time_sec']:.1f}s")
    print(f"  Checkpoint       : {result['checkpoint_path']}")
    print("=" * 50)
