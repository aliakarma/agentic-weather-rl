"""
src/evaluate.py
================
Evaluation pipeline for the trained Lagrangian CTDE-PPO policy
and all baseline methods.

Runs 500 evaluation episodes (as reported in Table 2 of the paper)
and outputs:
  - Mean cumulative reward  (± std)
  - Violation rate          (± std)   [Eq. 9 of paper]
  - Decision accuracy       (± std)   per agent

Decision accuracy for agent i is defined (Section 4.4 of paper) as:
  "The fraction of timesteps on which agent i's action matches the
   oracle action, where the oracle selects the action that minimises
   the simulated disaster impact given perfect information."

The oracle is approximated here by a risk-threshold oracle that selects
the highest-severity justified action given the risk level from the
environment info dict.

All results are printed in a formatted table matching Table 2 / Table 3
of the manuscript and optionally saved to results/example_results/.

Usage
-----
    # Evaluate the trained LagrangianCTDE checkpoint
    python -m src.evaluate

    # Evaluate with explicit checkpoint
    python -m src.evaluate --checkpoint checkpoints/marl_policy.pt --algo lagrangian_ctde

    # Evaluate a baseline
    python -m src.evaluate --algo mappo --checkpoint checkpoints/mappo_final.pt

    # Quick run (fewer episodes)
    python -m src.evaluate --episodes 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.environment.disaster_env import DisasterEnv


# ---------------------------------------------------------------------------
# Oracle for decision accuracy
# ---------------------------------------------------------------------------

def oracle_action(obs: np.ndarray, agent_id: int, risk_level: float) -> int:
    """
    Risk-threshold oracle action for agent i.

    The oracle has access to the environment's risk_level (not available
    to real agents) and selects the highest-severity action that is
    justified by the current risk.

    Decision accuracy = fraction of steps where agent action == oracle action.
    (Table 3 of paper: Storm=0.91, Flood=0.87, Evacuation=0.84)

    Parameters
    ----------
    obs        : local observation for agent i
    agent_id   : 1, 2, or 3
    risk_level : scalar risk from environment info dict

    Returns
    -------
    oracle_action : int ∈ {0, 1, 2, 3}
    """
    # Observation index constants (must match obs_router.py)
    if agent_id == 1:
        # Agent 1: Storm Detection — oracle based on storm_probability (obs[4])
        storm_prob = float(obs[4])
        if storm_prob > 0.6 or risk_level > 0.65:
            return 3  # evacuate
        elif storm_prob > 0.3 or risk_level > 0.4:
            return 1  # warn
        return 0

    elif agent_id == 2:
        # Agent 2: Flood Risk — oracle based on rainfall + river level
        combined = float(obs[0]) + float(obs[1])
        if combined > 0.8 or risk_level > 0.65:
            return 2  # deploy
        elif combined > 0.4 or risk_level > 0.35:
            return 1  # warn
        return 0

    else:  # agent_id == 3
        # Agent 3: Evacuation — oracle based on vulnerability × compound risk
        compound = float(obs[8]) * float(obs[9])
        if compound > 0.5 or risk_level > 0.7:
            return 3  # evacuate
        elif compound > 0.2 or risk_level > 0.5:
            return 2  # deploy
        return 0


# ---------------------------------------------------------------------------
# Single-episode evaluation
# ---------------------------------------------------------------------------

def _run_episode(
    agent,
    env: DisasterEnv,
    seed: int,
    deterministic: bool = True,
    constraint_d: float = 0.10,
) -> Dict[str, Any]:
    """
    Run one evaluation episode and return per-step metrics.

    Returns
    -------
    dict with keys: total_reward, violation_rate, decision_accuracies,
                    n_steps, actions_taken, risk_levels.
    """
    obs_dict, _ = env.reset(seed=seed)
    total_reward = 0.0
    correct_actions: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    total_steps_per_agent: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    n_violations = 0
    risk_levels: List[float] = []
    actions_taken: List[Tuple[int, int, int]] = []

    for _ in range(env.episode_length):
        # Get joint action from agent
        if hasattr(agent, "get_actions"):
            action_dict = agent.get_actions(obs_dict, deterministic=deterministic)
        elif hasattr(agent, "act"):
            action_dict = agent.act(obs_dict, deterministic=deterministic)
        else:
            raise AttributeError("Agent must implement get_actions() or act().")

        next_obs, reward, terminated, truncated, info = env.step(action_dict)
        risk_level = float(info.get("risk_level", 0.0))
        risk_levels.append(risk_level)
        total_reward += reward

        # Record joint action
        actions_taken.append((
            int(action_dict[1]), int(action_dict[2]), int(action_dict[3])
        ))

        # Decision accuracy: compare agent actions to oracle
        for i in range(1, 4):
            oracle = oracle_action(obs_dict[i], i, risk_level)
            if int(action_dict[i]) == oracle:
                correct_actions[i] += 1
            total_steps_per_agent[i] += 1

        # Constraint violations
        constraint_costs = info.get("constraint_costs", {})
        if any(c > constraint_d for c in constraint_costs.values()):
            n_violations += 1

        obs_dict = next_obs
        if terminated or truncated:
            break

    n_steps = len(actions_taken)
    ep_stats = env.get_episode_stats()

    decision_accuracies = {
        i: correct_actions[i] / max(total_steps_per_agent[i], 1)
        for i in range(1, 4)
    }

    return {
        "total_reward":        total_reward,
        "violation_rate":      ep_stats["violation_rate"],
        "decision_accuracies": decision_accuracies,
        "n_steps":             n_steps,
        "risk_levels":         risk_levels,
        "actions_taken":       actions_taken,
    }


# ---------------------------------------------------------------------------
# Public evaluate() function
# ---------------------------------------------------------------------------

def evaluate(
    agent,
    env: Optional[DisasterEnv] = None,
    n_episodes: int = 500,
    seed_offset: int = 10_000,
    deterministic: bool = True,
    constraint_d: float = 0.10,
    output_dir: Optional[str] = None,
    algo_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Run full evaluation over n_episodes episodes.

    Outputs (per Table 2 / Table 3 of the paper):
      - Mean reward ± std
      - Violation rate ± std
      - Per-agent decision accuracy ± std

    Parameters
    ----------
    agent
        Trained agent with get_actions() or act() method.
    env : DisasterEnv | None
        Evaluation environment. Created with default config if None.
    n_episodes : int
        Number of evaluation episodes (default 500, per paper).
    seed_offset : int
        Starting seed for evaluation (disjoint from training seeds).
    deterministic : bool
        Use greedy policy (default True).
    constraint_d : float
        Constraint threshold d_i for violation counting.
    output_dir : str | None
        If provided, save results JSON to this directory.
    algo_name : str
        Algorithm name for display and file naming.

    Returns
    -------
    metrics : dict
        {reward_mean, reward_std, violation_rate_mean, violation_rate_std,
         decision_accuracy_{1,2,3}_mean, decision_accuracy_{1,2,3}_std,
         n_episodes, algo_name}
    """
    if env is None:
        env = DisasterEnv()

    logger.info(
        "Evaluating %s over %d episodes (deterministic=%s)...",
        algo_name, n_episodes, deterministic,
    )

    rewards:    List[float] = []
    vr_list:    List[float] = []
    acc_list:   Dict[int, List[float]] = {1: [], 2: [], 3: []}

    t0 = time.time()

    for ep in range(n_episodes):
        seed = seed_offset + ep
        result = _run_episode(
            agent=agent,
            env=env,
            seed=seed,
            deterministic=deterministic,
            constraint_d=constraint_d,
        )

        rewards.append(result["total_reward"])
        vr_list.append(result["violation_rate"])
        for i in range(1, 4):
            acc_list[i].append(result["decision_accuracies"][i])

        if (ep + 1) % 50 == 0:
            logger.info(
                "  [%d/%d] mean_reward=%.2f  mean_VR=%.4f",
                ep + 1, n_episodes,
                float(np.mean(rewards)), float(np.mean(vr_list)),
            )

    elapsed = time.time() - t0

    metrics: Dict[str, Any] = {
        "algo_name":            algo_name,
        "n_episodes":           n_episodes,
        "reward_mean":          float(np.mean(rewards)),
        "reward_std":           float(np.std(rewards)),
        "violation_rate_mean":  float(np.mean(vr_list)),
        "violation_rate_std":   float(np.std(vr_list)),
        "eval_time_sec":        elapsed,
    }

    for i in range(1, 4):
        metrics[f"decision_accuracy_{i}_mean"] = float(np.mean(acc_list[i]))
        metrics[f"decision_accuracy_{i}_std"]  = float(np.std(acc_list[i]))

    # Print formatted table (matches Table 2 / Table 3 in the paper)
    _print_results_table(metrics)

    # Save results to JSON
    if output_dir is not None:
        _save_results(metrics, output_dir, algo_name)

    return metrics


# ---------------------------------------------------------------------------
# Display and I/O helpers
# ---------------------------------------------------------------------------

def _print_results_table(metrics: Dict[str, Any]) -> None:
    """Print a formatted results table matching the paper's Table 2."""
    print("\n" + "=" * 65)
    print(f"  Evaluation Results — {metrics['algo_name'].upper()}")
    print(f"  Episodes: {metrics['n_episodes']}   "
          f"Time: {metrics['eval_time_sec']:.1f}s")
    print("=" * 65)
    print(f"  {'Metric':<35} {'Mean':>8}  {'±Std':>8}")
    print("-" * 65)
    print(f"  {'Cumulative Reward':<35} "
          f"{metrics['reward_mean']:>8.2f}  "
          f"{metrics['reward_std']:>8.2f}")
    print(f"  {'Violation Rate (VR)':<35} "
          f"{metrics['violation_rate_mean']:>8.4f}  "
          f"{metrics['violation_rate_std']:>8.4f}")
    print("-" * 65)
    agent_names = {1: "Agent 1 (Storm Detection)",
                   2: "Agent 2 (Flood Risk)",
                   3: "Agent 3 (Evacuation)"}
    for i in range(1, 4):
        mean = metrics[f"decision_accuracy_{i}_mean"]
        std  = metrics[f"decision_accuracy_{i}_std"]
        print(f"  {'Decision Accuracy — ' + agent_names[i]:<35} "
              f"{mean:>8.4f}  {std:>8.4f}")
    print("=" * 65 + "\n")


def _save_results(
    metrics: Dict[str, Any], output_dir: str, algo_name: str
) -> None:
    """Save metrics dict to a JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"eval_{algo_name}.json"
    with open(fname, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Results saved to %s", fname)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_agent_from_checkpoint(
    checkpoint_path: str,
    algo: str,
    device: str = "cpu",
) -> Any:
    """
    Load a trained agent from a checkpoint file.

    Returns the agent object ready for evaluation.
    """
    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if algo == "lagrangian_ctde":
        from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
        env = DisasterEnv()
        agent = LagrangianCTDE(env=env, config=LagrangianCTDEConfig(device=device))
        agent.load_checkpoint(checkpoint_path)
        return agent

    if algo == "heuristic":
        from src.algorithms.baselines.heuristic import HeuristicPolicy
        return HeuristicPolicy()  # stateless, no checkpoint needed

    if algo == "dqn":
        from src.algorithms.baselines.dqn import DQNAgent
        agent = DQNAgent(device=device)
        agent.load(checkpoint_path)
        return agent

    if algo == "ippo":
        from src.algorithms.baselines.ippo import IPPOAgent
        agent = IPPOAgent(device=device)
        agent.load(checkpoint_path)
        return agent

    if algo == "qmix":
        from src.algorithms.baselines.qmix import QMIXAgent
        agent = QMIXAgent(device=device)
        agent.load(checkpoint_path)
        return agent

    if algo == "mappo":
        from src.algorithms.baselines.mappo import MAPPOAgent
        agent = MAPPOAgent(device=device)
        agent.load(checkpoint_path)
        return agent

    if algo == "cpo":
        from src.algorithms.baselines.cpo import CPOAgent
        agent = CPOAgent(device=device)
        agent.load(checkpoint_path)
        return agent

    raise ValueError(f"Unknown algorithm: {algo}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MARL agent on the disaster response environment."
    )
    parser.add_argument(
        "--algo",
        default="lagrangian_ctde",
        choices=["lagrangian_ctde", "heuristic", "dqn", "ippo", "qmix", "mappo", "cpo"],
        help="Algorithm to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/marl_policy.pt",
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of evaluation episodes (default: 500)",
    )
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--seed",      type=int, default=10_000)
    parser.add_argument(
        "--output_dir",
        default="results/example_results/",
        help="Directory to save evaluation JSON",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic (sampled) rather than greedy actions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    agent = load_agent_from_checkpoint(
        checkpoint_path=args.checkpoint,
        algo=args.algo,
        device=args.device,
    )

    env = DisasterEnv(seed=args.seed)

    metrics = evaluate(
        agent=agent,
        env=env,
        n_episodes=args.episodes,
        seed_offset=args.seed,
        deterministic=not args.stochastic,
        output_dir=args.output_dir,
        algo_name=args.algo,
    )

    # Exit code 0 on success
    sys.exit(0)
