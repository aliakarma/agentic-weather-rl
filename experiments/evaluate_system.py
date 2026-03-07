"""
End-to-End System Evaluation Script
======================================
Purpose:
    Evaluates the complete three-layer architecture:
      1. Generates synthetic weather states via the perception module
      2. Runs the trained RL agent on each state
      3. Logs emergency response actions via the orchestration layer
      4. Computes decision accuracy per scenario type
      5. Saves results as CSV and accuracy plot

Output:
    results/experiment_results.csv   — per-scenario decision accuracy
    results/accuracy_plot.png        — bar chart of decision accuracy by scenario

Usage:
    python experiments/evaluate_system.py \
        --model_path results/ppo_agent \
        --n_episodes 200 \
        --seed 42
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.environment import DisasterResponseEnv, WeatherScenarioGenerator, ACTION_LABELS
from rl_agent.agent_ppo import DisasterResponseAgent
from orchestration.emergency_action_simulator import (
    EmergencyOrchestrator,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Scenario definitions
# Scenario label → (risk_level_string, expected_optimal_action)
# ---------------------------------------------------------------------------
SCENARIO_DEFINITIONS = {
    "storm_warning":   ("medium",   1),   # Issue early warning
    "flood_response":  ("high",     2),   # Prepare emergency resources
    "evacuation_decision": ("critical", 3),  # Recommend evacuation
    "no_action":       ("low",      0),   # No action
}


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def load_agent(model_path: str, env: DisasterResponseEnv) -> DisasterResponseAgent:
    """
    Load a saved PPO agent from disk, or create and briefly train a new one
    if no saved model exists (allows evaluation to run end-to-end without
    a pre-existing checkpoint).

    Args:
        model_path: Path to saved model (SB3 .zip format).
        env:        Environment instance for the agent.

    Returns:
        DisasterResponseAgent.
    """
    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if os.path.exists(zip_path):
        print(f"[INFO] Loading saved agent from {zip_path}")
        return DisasterResponseAgent.load(model_path, env)
    else:
        print(f"[WARNING] No saved model at {zip_path}. Training a quick agent (10k steps).")
        agent = DisasterResponseAgent(env, seed=42, verbose=0)
        agent.train(total_timesteps=10_000)
        return agent


def evaluate_scenario(
    agent: DisasterResponseAgent,
    scenario_gen: WeatherScenarioGenerator,
    risk_level: str,
    expected_action: int,
    n_trials: int = 100,
) -> float:
    """
    Evaluate decision accuracy for a specific scenario type.

    Generates n_trials observations at the given risk level, queries the
    agent for its action, and measures how often the agent selects the
    expected optimal action.

    Args:
        agent:           Trained DisasterResponseAgent.
        scenario_gen:    WeatherScenarioGenerator instance.
        risk_level:      Risk level string ('low', 'medium', 'high', 'critical').
        expected_action: The action index the agent ideally selects.
        n_trials:        Number of random observations to test.

    Returns:
        Decision accuracy in [0, 1].
    """
    correct = 0
    for _ in range(n_trials):
        obs = scenario_gen.sample(risk_level=risk_level)
        action, _ = agent.predict(obs, deterministic=True)
        if action == expected_action:
            correct += 1
    return correct / n_trials


def run_full_pipeline_evaluation(
    agent: DisasterResponseAgent,
    orchestrator: EmergencyOrchestrator,
    n_episodes: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Run the complete perception → RL → orchestration pipeline and collect
    decision logs and aggregate statistics.

    Args:
        agent:        Trained RL agent.
        orchestrator: EmergencyOrchestrator for simulating actions.
        n_episodes:   Number of simulated decisions to run.
        seed:         Random seed for scenario generation.

    Returns:
        Dictionary with per-action statistics and overall accuracy.
    """
    gen = WeatherScenarioGenerator(seed=seed)
    action_counts = defaultdict(int)
    correct_counts = defaultdict(int)
    total_reward = 0.0

    # Simple env for reward calculation
    env = DisasterResponseEnv(max_steps=1, seed=seed)

    for _ in range(n_episodes):
        obs = gen.sample()
        risk_level_str = gen.compute_risk_level(obs)
        risk_enum = RiskLevel[risk_level_str.upper()]

        action, _ = agent.predict(obs, deterministic=True)
        action_counts[ACTION_LABELS[action]] += 1

        # Execute simulated orchestration action
        weather_dict = {
            "storm_probability": float(obs[0]),
            "rainfall_intensity": float(obs[1]),
            "flood_risk_score": float(obs[2]),
            "regional_risk_indicator": float(obs[3]),
        }
        orchestrator.execute_action(action, risk_enum, weather_dict)

        # Check correctness against expected action mapping
        expected_map = {
            "low": 0, "medium": 1, "high": 2, "critical": 3
        }
        if action == expected_map[risk_level_str]:
            correct_counts["overall"] += 1

    overall_accuracy = correct_counts["overall"] / n_episodes

    return {
        "n_episodes": n_episodes,
        "overall_accuracy": overall_accuracy,
        "action_distribution": dict(action_counts),
    }


def plot_accuracy(scenario_results: Dict[str, float], save_path: str) -> None:
    """
    Plot a bar chart of decision accuracy per scenario type.

    Args:
        scenario_results: Dict mapping scenario name → accuracy.
        save_path:        Output path for the PNG file.
    """
    labels = list(scenario_results.keys())
    values = [scenario_results[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(labels, values, color=colors[:len(labels)], width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Decision Accuracy")
    ax.set_title("RL Agent Decision Accuracy by Scenario Type")
    ax.axhline(y=0.8, linestyle="--", color="gray", alpha=0.6, label="0.80 baseline")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Accuracy plot saved to {save_path}")


def save_results_csv(scenario_results: Dict[str, float], save_path: str) -> None:
    """
    Save per-scenario decision accuracy to a CSV file.

    Args:
        scenario_results: Dict mapping scenario name → accuracy.
        save_path:        Output CSV path.
    """
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "decision_accuracy"])
        for scenario, acc in scenario_results.items():
            writer.writerow([scenario, f"{acc:.4f}"])
    print(f"[INFO] Results saved to {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str = "results/ppo_agent",
    n_episodes: int = 200,
    n_trials_per_scenario: int = 100,
    seed: int = 42,
    output_dir: str = "results",
    region: str = "Evaluation Region",
) -> None:
    """
    Run the full end-to-end system evaluation.

    Steps:
      1. Load or train the RL agent.
      2. Evaluate per-scenario decision accuracy.
      3. Run full pipeline with orchestration.
      4. Save results CSV and accuracy plot.

    Args:
        model_path:              Path to saved PPO model.
        n_episodes:              Total pipeline evaluation episodes.
        n_trials_per_scenario:   Trials per scenario type for accuracy measurement.
        seed:                    Random seed.
        output_dir:              Where to save outputs.
        region:                  Region name for orchestrator logs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ----- Load agent -----
    env = DisasterResponseEnv(max_steps=50, seed=seed)
    agent = load_agent(model_path, env)

    # ----- Per-scenario accuracy -----
    gen = WeatherScenarioGenerator(seed=seed)
    print("\n[INFO] Evaluating per-scenario decision accuracy...")
    scenario_results = {}
    for scenario_name, (risk_level, expected_action) in SCENARIO_DEFINITIONS.items():
        acc = evaluate_scenario(agent, gen, risk_level, expected_action, n_trials_per_scenario)
        scenario_results[scenario_name] = acc
        print(f"  {scenario_name:>25}: {acc:.4f}")

    # ----- Full pipeline evaluation -----
    print("\n[INFO] Running full pipeline evaluation...")
    orchestrator = EmergencyOrchestrator(region=region, verbose=False)
    pipeline_stats = run_full_pipeline_evaluation(agent, orchestrator, n_episodes, seed)
    print(f"  Overall pipeline accuracy: {pipeline_stats['overall_accuracy']:.4f}")
    print(f"  Action distribution: {pipeline_stats['action_distribution']}")

    # ----- Save results -----
    save_results_csv(scenario_results, os.path.join(output_dir, "experiment_results.csv"))
    plot_accuracy(scenario_results, os.path.join(output_dir, "accuracy_plot.png"))

    # Print summary
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    for k, v in scenario_results.items():
        print(f"  {k:>30}: {v:.4f}")
    print("-" * 55)
    print(f"  {'Pipeline Overall Accuracy':>30}: {pipeline_stats['overall_accuracy']:.4f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the end-to-end disaster response system.")
    parser.add_argument("--model_path", type=str, default="results/ppo_agent")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--region", type=str, default="Evaluation Region")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        n_trials_per_scenario=args.n_trials,
        seed=args.seed,
        output_dir=args.output_dir,
        region=args.region,
    )
