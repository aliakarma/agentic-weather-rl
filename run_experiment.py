"""Canonical reproducible experiment runner.

Usage:
    python run_experiment.py --config config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from src.algorithms.baselines.cpo import CPOAgent
from src.algorithms.baselines.dqn import DQNAgent
from src.algorithms.baselines.ippo import IPPOAgent
from src.algorithms.baselines.mappo import MAPPOAgent
from src.algorithms.baselines.qmix import QMIXAgent
from src.algorithms.baselines.random_agent import RandomAgent
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
from src.environment.disaster_env import DisasterEnv
from src.evaluate import evaluate


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at top level.")
    return data


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _build_agent(method: str, seed: int, train_cfg: Dict[str, Any], env_obs_dim: int):
    n_agents = int(train_cfg.get("n_agents", 3))
    n_actions = int(train_cfg.get("n_actions", 4))
    hidden_dim = int(train_cfg.get("hidden_dim", 128))
    gamma = float(train_cfg.get("gamma", 0.99))
    lr = float(train_cfg.get("lr", 3e-4))

    if method == "lagrangian_ctde":
        cfg = LagrangianCTDEConfig(
            obs_dim=env_obs_dim,
            action_dim=n_actions,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            n_episodes=int(train_cfg.get("train_episodes", 1000)),
            seed=seed,
            gamma=gamma,
            lr=lr,
            lambda_lr=float(train_cfg.get("lambda_lr", 1e-3)),
            cost_limit=float(train_cfg.get("cost_limit", 0.10)),
            clip_eps=float(train_cfg.get("clip_eps", 0.2)),
            entropy_coef=float(train_cfg.get("entropy_coef", 0.01)),
            log_interval=int(train_cfg.get("log_interval", 50)),
            save_interval=int(train_cfg.get("save_interval", 500)),
            # New parameters (Req 1-7)
            lr_lambda=float(train_cfg.get("lr_lambda", 0.00001)),
            target_constraint=float(train_cfg.get("target_constraint", 0.10)),
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
            reward_normalize_enabled=bool(train_cfg.get("reward_normalize_enabled", True)),
            advantage_normalize_enabled=bool(train_cfg.get("advantage_normalize_enabled", True)),
            exploration_noise_enabled=bool(train_cfg.get("exploration_noise_enabled", True)),
            exploration_noise_schedule=str(train_cfg.get("exploration_noise_schedule", "exponential")),
            hybrid_value_enabled=bool(train_cfg.get("hybrid_value_enabled", False)),
            bc_pretrain_enabled=bool(train_cfg.get("bc_pretrain_enabled", True)),
            bc_epochs=int(train_cfg.get("bc_epochs", 8)),
            bc_batch_size=int(train_cfg.get("bc_batch_size", 128)),
            bc_lr=float(train_cfg.get("bc_lr", 5e-4)),
            bc_expert_episodes=int(train_cfg.get("bc_expert_episodes", 120)),
            expert_data_path=str(train_cfg.get("expert_data_path", "data/expert_trajectories.json")),
            expert_checkpoint_path=str(train_cfg.get("expert_checkpoint_path", "checkpoints/qmix_final.pt")),
            bc_eval_samples=int(train_cfg.get("bc_eval_samples", 512)),
            freeze_actor_steps=int(train_cfg.get("freeze_actor_steps", 1500)),
            actor_lr_scale=float(train_cfg.get("actor_lr_scale", 0.008)),
            adv_clip_abs=float(train_cfg.get("adv_clip_abs", 2.0)),
            kl_threshold=float(train_cfg.get("kl_threshold", 0.02)),
            kl_lr_backoff=float(train_cfg.get("kl_lr_backoff", 0.5)),
            min_actor_lr_scale=float(train_cfg.get("min_actor_lr_scale", 0.02)),
            kl_early_stop_threshold=float(train_cfg.get("kl_early_stop_threshold", 0.015)),
            ppo_update_epochs=int(train_cfg.get("ppo_update_epochs", 3)),
            kl_loss_scale_threshold=float(train_cfg.get("kl_loss_scale_threshold", 0.015)),
            kl_loss_scale_factor=float(train_cfg.get("kl_loss_scale_factor", 0.5)),
            update_every_episodes=int(train_cfg.get("update_every_episodes", 5)),
            rollout_horizon=int(train_cfg.get("rollout_horizon", 50)),
            return_normalize_enabled=bool(train_cfg.get("return_normalize_enabled", True)),
        )
        env = DisasterEnv(
            seed=seed,
            observation_noise=float(train_cfg.get("observation_noise", 0.02)),
            hazard_rate=float(train_cfg.get("hazard_rate", 0.22)),
        )
        return LagrangianCTDE(env=env, config=cfg), env

    if method == "random":
        env = DisasterEnv(
            seed=seed,
            observation_noise=float(train_cfg.get("observation_noise", 0.02)),
            hazard_rate=float(train_cfg.get("hazard_rate", 0.22)),
        )
        return RandomAgent(n_agents=n_agents, n_actions=n_actions, seed=seed), env

    kwargs = dict(
        obs_dim=env_obs_dim,
        n_actions=n_actions,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        gamma=gamma,
        seed=seed,
    )

    if method == "dqn":
        kwargs.update(
            lr=lr,
            batch_size=int(train_cfg.get("batch_size", 64)),
            target_update_interval=int(train_cfg.get("target_update_interval", 200)),
        )
        agent = DQNAgent(**kwargs)
    elif method == "ippo":
        kwargs.update(
            lr=lr,
            rollout_len=int(train_cfg.get("rollout_len", 128)),
            ppo_epochs=int(train_cfg.get("ppo_epochs", 4)),
            clip_eps=float(train_cfg.get("clip_eps", 0.2)),
        )
        agent = IPPOAgent(**kwargs)
    elif method == "qmix":
        kwargs.update(
            lr=lr,
            state_dim=int(train_cfg.get("state_dim", 24)),
            batch_size=int(train_cfg.get("batch_size", 64)),
            target_update_interval=int(train_cfg.get("target_update_interval", 200)),
        )
        agent = QMIXAgent(**kwargs)
    elif method == "mappo":
        kwargs.update(
            state_dim=int(train_cfg.get("state_dim", 24)),
            lr_actor=float(train_cfg.get("lr_actor", lr)),
            lr_critic=float(train_cfg.get("lr_critic", lr)),
            rollout_len=int(train_cfg.get("rollout_len", 128)),
            ppo_epochs=int(train_cfg.get("ppo_epochs", 4)),
            clip_eps=float(train_cfg.get("clip_eps", 0.2)),
        )
        agent = MAPPOAgent(**kwargs)
    elif method == "cpo":
        kwargs.update(
            lr_actor=float(train_cfg.get("lr_actor", lr)),
            lr_critic=float(train_cfg.get("lr_critic", lr)),
            rollout_len=int(train_cfg.get("rollout_len", 128)),
            ppo_epochs=int(train_cfg.get("ppo_epochs", 4)),
            clip_eps=float(train_cfg.get("clip_eps", 0.2)),
            cost_limit=float(train_cfg.get("cost_limit", 0.10)),
            lambda_lr=float(train_cfg.get("lambda_lr", 5e-3)),
        )
        agent = CPOAgent(**kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

    env = DisasterEnv(
        seed=seed,
        observation_noise=float(train_cfg.get("observation_noise", 0.02)),
        hazard_rate=float(train_cfg.get("hazard_rate", 0.22)),
    )
    return agent, env


def _save_minimal_checkpoint(path: Path, agent, config: Dict[str, Any], seed: int, commit_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": agent.state_dict() if hasattr(agent, "state_dict") else None,
        "config": config,
        "seed": int(seed),
        "git_commit": commit_hash,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _json_safe_config_hash(config: Dict[str, Any]) -> str:
    txt = json.dumps(config, sort_keys=True)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


def _ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"low": float("nan"), "high": float("nan")}
    if n == 1:
        return {"low": float(arr[0]), "high": float(arr[0])}
    sem = stats.sem(arr)
    low, high = stats.t.interval(0.95, n - 1, loc=float(np.mean(arr)), scale=float(sem))
    return {"low": float(low), "high": float(high)}


def _aggregate(seed_rows: List[dict]) -> dict:
    rewards = [r["reward"] for r in seed_rows]
    constraints = [r["constraint_violations"]["mean"] for r in seed_rows]
    fairness = [r["fairness_metrics"]["jain_action_index"] for r in seed_rows]

    return {
        "n_runs": len(seed_rows),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_ci95": _ci95(rewards),
        "constraint_mean": float(np.mean(constraints)),
        "constraint_std": float(np.std(constraints)),
        "constraint_ci95": _ci95(constraints),
        "fairness_mean": float(np.mean(fairness)),
        "fairness_std": float(np.std(fairness)),
        "fairness_ci95": _ci95(fairness),
    }


def _paired_t_test(primary: List[dict], secondary: List[dict]) -> dict:
    p_by_seed = {x["seed"]: x for x in primary}
    s_by_seed = {x["seed"]: x for x in secondary}
    shared = sorted(set(p_by_seed).intersection(s_by_seed))
    if len(shared) < 2:
        return {"n": len(shared), "t_stat": float("nan"), "p_value": float("nan")}
    p_reward = np.array([p_by_seed[s]["reward"] for s in shared], dtype=np.float64)
    s_reward = np.array([s_by_seed[s]["reward"] for s in shared], dtype=np.float64)
    t_stat, p_val = stats.ttest_rel(p_reward, s_reward)
    return {
        "n": len(shared),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }


def _run_single(method: str, seed: int, cfg: Dict[str, Any], out_dir: Path, commit_hash: str) -> dict:
    _seed_everything(seed)

    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    env_obs_dim = int(train_cfg.get("obs_dim", 12))
    agent, train_env = _build_agent(method, seed, train_cfg, env_obs_dim)

    n_train = int(train_cfg.get("train_episodes", 1000))
    save_interval = int(train_cfg.get("save_interval", max(1, n_train // 2)))

    if hasattr(agent, "train"):
        if method == "lagrangian_ctde":
            agent.train(n_episodes=n_train)
        else:
            agent.train(
                env=train_env,
                n_episodes=n_train,
                checkpoint_dir=str(out_dir / "checkpoints" / method),
                save_interval=save_interval,
            )

    eval_env = DisasterEnv(
        seed=int(eval_cfg.get("eval_seed_base", 10_000)) + seed,
        observation_noise=float(train_cfg.get("observation_noise", 0.02)),
        hazard_rate=float(train_cfg.get("hazard_rate", 0.22)),
    )

    metrics = evaluate(
        agent=agent,
        env=eval_env,
        n_episodes=int(eval_cfg.get("eval_episodes", 200)),
        seed_offset=int(eval_cfg.get("eval_seed_base", 10_000)) + seed,
        deterministic=bool(eval_cfg.get("deterministic", True)),
    )

    # Save checkpoint with strict metadata requirements.
    run_cfg = {
        "method": method,
        "training": train_cfg,
        "evaluation": eval_cfg,
    }
    ckpt_path = out_dir / "checkpoints" / method / f"seed_{seed}.pt"
    _save_minimal_checkpoint(ckpt_path, agent, run_cfg, seed, commit_hash)

    row = {
        "seed": int(seed),
        "method": method,
        "git_commit": commit_hash,
        "config_hash": _json_safe_config_hash(run_cfg),
        "reward": float(metrics["reward_mean"]),
        "accuracy": None,
        "fairness_metrics": {
            "jain_action_index": float(metrics.get("fairness_jain_action_index", float("nan"))),
            "action_balance_std": float(metrics.get("fairness_action_balance_std", float("nan"))),
            "agent_1_action_mean": float(metrics.get("agent_1_action_mean", float("nan"))),
            "agent_2_action_mean": float(metrics.get("agent_2_action_mean", float("nan"))),
            "agent_3_action_mean": float(metrics.get("agent_3_action_mean", float("nan"))),
        },
        "constraint_violations": {
            "mean": float(metrics["violation_rate_mean"]),
            "std": float(metrics["violation_rate_std"]),
        },
        "damage_metrics": {
            "avg_damage_prevented": float(metrics["avg_damage_prevented_mean"]),
            "avg_damage_after": float(metrics["avg_damage_after_mean"]),
        },
        "robustness_metrics": {
            "noise_level": float(metrics["robustness_noise_level"]),
            "reward_mean": float(metrics["robustness_reward_mean"]),
            "reward_delta": float(metrics["robustness_reward_delta"]),
            "reward_ratio": float(metrics["robustness_reward_ratio"]),
        },
        "raw_metrics": metrics,
    }

    seed_file = out_dir / f"seed_{seed}.json"
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    return row


def run_experiment(config_path: str, method_override: str | None = None) -> None:
    cfg = _load_yaml(config_path)
    method = method_override if method_override else str(cfg.get("method", "lagrangian_ctde"))
    seeds = list(cfg.get("seeds", []))
    if len(seeds) < 10:
        raise ValueError("At least 10 seeds are required for statistical validity.")

    out_root_dir = Path(cfg.get("output_dir", "results"))
    out_root_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_root_dir / method
    out_dir.mkdir(parents=True, exist_ok=True)

    commit_hash = _git_commit_hash()

    print(f"Running method={method} over {len(seeds)} independent seeds")
    rows = []
    for s in seeds:
        seed = int(s)
        print(f"\n=== Seed {seed} ===")
        rows.append(_run_single(method, seed, cfg, out_dir, commit_hash))

    summary = _aggregate(rows)

    secondary_rows: Optional[List[dict]] = None
    compare_method = cfg.get("compare_method")
    if compare_method:
        compare_method = str(compare_method)
        secondary_rows = []
        print(f"\nRunning paired comparison against method={compare_method}")
        for s in seeds:
            seed = int(s)
            secondary_rows.append(_run_single(compare_method, seed, cfg, out_dir / compare_method, commit_hash))
        summary["paired_t_test_reward"] = _paired_t_test(rows, secondary_rows)

    summary_payload = {
        "method": method,
        "git_commit": commit_hash,
        "seeds": [int(x) for x in seeds],
        "summary": summary,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nSaved per-seed files and summary:")
    print(f"  {out_dir / 'summary.json'}")
    print("All results are derived from independent raw runs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--method",
        required=False,
        choices=["lagrangian_ctde", "dqn", "ippo", "qmix", "mappo", "cpo", "random"],
        help="Override method from config and run only that method",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(args.config, method_override=args.method)
