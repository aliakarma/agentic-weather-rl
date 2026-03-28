#!/usr/bin/env python3
"""
Simple test for iteration 2 reward recovery parameters.
Tests LAGRANGIAN CTDE with aggressive warmup and reduced constraints.
"""
import json
import os
import numpy as np
import yaml
from pathlib import Path

# Environment and algorithm imports
from src.environment.disaster_env import DisasterEnv
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
from src.evaluate import evaluate

# Baseline for comparison
BASELINE = {"reward_mean": -187.7, "reward_std": 0.5}

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def test_iter2():
    """Test iteration 2 with aggressive parameters"""
    cfg = load_config("smoke_test_config.yaml")
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    
    print("=" * 80)
    print("ITERATION 2 TEST: Aggressive Reward Recovery Parameters")
    print("=" * 80)
    print()
    print("Key aggressive settings:")
    print(f"  - lr_lambda: {train_cfg.get('lr_lambda', 0.0003)} (100x reduction)")
    print(f"  - warmup_fraction: {train_cfg.get('warmup_fraction', 0.25)} (50% training)")
    print(f"  - lambda_curriculum_max: {train_cfg.get('lambda_curriculum_max', 0.04)} (75% reduction)")
    print(f"  - alpha_entropy: {train_cfg.get('alpha_entropy', 0.03)} (3.3x boost)")
    print(f"  - reward weights: {train_cfg.get('reward_weights', {})}")
    print()
    
    results = []
    seeds = [0, 1, 2]
    
    for seed in seeds:
        print(f"Running seed {seed}...")
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        ctde_config = LagrangianCTDEConfig(
            obs_dim=12,
            action_dim=4,
            n_agents=3,
            hidden_dim=128,
            n_episodes=120,
            seed=seed,
            gamma=0.99,
            lr=3e-4,
            lambda_lr=0.001,
            cost_limit=0.10,
            clip_eps=0.2,
            entropy_coef=0.01,
            log_interval=20,
            # Aggressive iteration 2 parameters
            lr_lambda=float(train_cfg.get("lr_lambda", 0.00001)),
            warmup_phase_enabled=bool(train_cfg.get("warmup_phase_enabled", True)),
            warmup_fraction=float(train_cfg.get("warmup_fraction", 0.50)),
            lambda_curriculum_enabled=bool(train_cfg.get("lambda_curriculum_enabled", True)),
            lambda_curriculum_min=float(train_cfg.get("lambda_curriculum_min", 0.00001)),
            lambda_curriculum_max=float(train_cfg.get("lambda_curriculum_max", 0.01)),
            alpha_entropy=float(train_cfg.get("alpha_entropy", 0.10)),
            reward_weights=dict(train_cfg.get("reward_weights", {
                "damage_prevented": 5.0,
                "damage_penalty": -0.1,
                "resource_cost": -0.001,
                "constraint_penalty": -0.5,
            })),
            reward_normalize_enabled=True,
            advantage_normalize_enabled=True,
            exploration_noise_enabled=True,
            exploration_noise_schedule="exponential",
            hybrid_value_enabled=False,
        )
        
        env = DisasterEnv(
            seed=seed,
            observation_noise=0.02,
            hazard_rate=0.22,
        )
        
        agent = LagrangianCTDE(env=env, config=ctde_config)
        agent.train(n_episodes=120)
        
        # Evaluate
        metrics = evaluate(
            agent, env, n_episodes=200, deterministic=True
        )
        results.append({
            "reward": metrics["reward_mean"],
            "fairness": metrics.get("fairness_jain_action_index", 0),
            "constraint": metrics["violation_rate_mean"]
        })
        print(f"  Seed {seed}: reward={metrics['reward_mean']:.2f}, fairness={metrics.get('fairness_jain_action_index', 0):.4f}, constraint={metrics['violation_rate_mean']:.4f}")
    
    # Summary
    print()
    print("=" * 80)
    print("TEST RESULTS - ITERATION 2 AGGRESSIVE PARAMETERS")
    print("=" * 80)
    
    mean_reward = np.mean([r["reward"] for r in results])
    std_reward = np.std([r["reward"] for r in results])
    mean_fairness = np.mean([r["fairness"] for r in results])
    mean_constraint = np.mean([r["constraint"] for r in results])
    
    print(f"Mean reward: {mean_reward:.2f} (baseline: {BASELINE['reward_mean']:.2f})")
    print(f"Std reward: {std_reward:.2f} (baseline: {BASELINE['reward_std']:.2f})")
    print(f"Mean fairness (Jain Index): {mean_fairness:.4f}")
    print(f"Mean constraint violations: {mean_constraint:.4f}")
    print()
    
    improvement = BASELINE['reward_mean'] - mean_reward
    print(f"Improvement: {improvement:.2f} points (less negative is better)")
    print()
    
    if improvement >= 10.0:
        print("[SUCCESS] Reward improved by at least 10 points!")
    else:
        print(f"[FAILED] Need {10.0 - improvement:.2f} more points improvement")

if __name__ == "__main__":
    test_iter2()
