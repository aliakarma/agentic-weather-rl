"""Aggregate experiment artifacts for reviewer inspection.

Usage:
  python scripts/aggregate_artifacts.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

KNOWN_METHODS = ["lagrangian_ctde", "dqn", "ippo", "qmix", "mappo", "cpo"]
SEARCH_ROOTS = ["results", "outputs", "experiments"]


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"low": float("nan"), "high": float("nan")}
    if len(arr) == 1:
        return {"low": float(arr[0]), "high": float(arr[0])}
    sem = stats.sem(arr)
    low, high = stats.t.interval(0.95, len(arr) - 1, loc=float(np.mean(arr)), scale=float(sem))
    return {"low": float(low), "high": float(high)}


def _infer_method(seed_json_path: Path, payload: dict) -> str | None:
    method = payload.get("method")
    if isinstance(method, str) and method in KNOWN_METHODS:
        return method

    p = str(seed_json_path).lower().replace("\\", "/")
    for m in KNOWN_METHODS:
        if f"/{m}/" in p:
            return m
    return None


def _seed_from_name(path: Path) -> int | None:
    name = path.stem
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("_")[1])
    except Exception:
        return None


def _find_seed_jsons(root: Path) -> List[Path]:
    files: List[Path] = []
    for rel in SEARCH_ROOTS:
        d = root / rel
        if d.exists():
            for fp in d.rglob("seed_*.json"):
                if "artifacts" not in fp.parts:
                    files.append(fp)
    return sorted(set(files))


def _validate_seed_payload(payload: dict) -> Tuple[bool, List[str]]:
    missing = []
    if "reward" not in payload:
        missing.append("reward")
    if "fairness_metrics" not in payload:
        missing.append("fairness_metrics")
    if "constraint_violations" not in payload:
        missing.append("constraint_violations")
    return len(missing) == 0, missing


def _copy_configs(root: Path, artifacts_configs: Path) -> List[str]:
    artifacts_configs.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    candidates: List[Path] = []

    top_config = root / "config.yaml"
    if top_config.exists():
        candidates.append(top_config)

    configs_dir = root / "configs"
    if configs_dir.exists():
        candidates.extend(sorted(configs_dir.glob("*.yaml")))

    for fp in candidates:
        target = artifacts_configs / fp.name
        shutil.copy2(fp, target)
        copied.append(str(fp))
    return copied


def _build_plots(rows: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    methods = sorted(by_method.keys())

    # 1) Reward distribution per method.
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [[x["reward"] for x in by_method[m]] for m in methods]
    ax.boxplot(data, tick_labels=methods)
    ax.set_title("Reward Distribution Per Method")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Method")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_distribution_per_method.png", dpi=150)
    plt.close(fig)

    # 2) Fairness vs constraint scatter.
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in methods:
        xs = [x["fairness"] for x in by_method[m]]
        ys = [x["constraint"] for x in by_method[m]]
        ax.scatter(xs, ys, label=m, alpha=0.75)
    ax.set_title("Fairness vs Constraint")
    ax.set_xlabel("Fairness")
    ax.set_ylabel("Constraint Violations")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fairness_vs_constraint.png", dpi=150)
    plt.close(fig)

    # 3) Mean +- std reward bar chart.
    means = [float(np.mean([x["reward"] for x in by_method[m]])) for m in methods]
    stds = [float(np.std([x["reward"] for x in by_method[m]])) for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(methods, means, yerr=stds, capsize=4)
    ax.set_title("Reward Mean +- Std")
    ax.set_ylabel("Reward")
    ax.set_xlabel("Method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_mean_std.png", dpi=150)
    plt.close(fig)


def main(strict: bool = False) -> int:
    root = Path.cwd()
    artifacts_dir = root / "artifacts"
    methods_dir = artifacts_dir / "methods"
    plots_dir = artifacts_dir / "plots"
    configs_dir = artifacts_dir / "configs"
    metadata_dir = artifacts_dir / "metadata"

    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []

    seed_files = _find_seed_jsons(root)
    if not seed_files:
        raise RuntimeError("No seed_*.json files found under results/ outputs/ experiments/")

    rows: List[dict] = []
    seen = set()

    for fp in seed_files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Failed to parse {fp}: {exc}")
            continue

        method = _infer_method(fp, payload)
        if not method:
            warnings.append(f"Could not infer method for {fp}")
            continue

        seed = payload.get("seed")
        if seed is None:
            seed = _seed_from_name(fp)
        try:
            seed = int(seed)
        except Exception:
            warnings.append(f"Invalid seed value in {fp}")
            continue

        valid, missing = _validate_seed_payload(payload)
        if not valid:
            warnings.append(f"Missing required fields in {fp}: {', '.join(missing)}")
            continue

        key = (method, seed)
        if key in seen:
            warnings.append(f"Duplicate method/seed pair skipped: {method} seed={seed} ({fp})")
            continue
        seen.add(key)

        method_out = methods_dir / method
        method_out.mkdir(parents=True, exist_ok=True)
        out_fp = method_out / f"seed_{seed}.json"
        shutil.copy2(fp, out_fp)

        fairness = payload["fairness_metrics"]
        constraints = payload["constraint_violations"]
        damage_prevented = None
        if isinstance(payload.get("damage_metrics"), dict):
            damage_prevented = payload["damage_metrics"].get("avg_damage_prevented")

        row = {
            "method": method,
            "seed": seed,
            "reward": float(payload["reward"]),
            "fairness": float(fairness.get("jain_action_index", np.nan)),
            "constraint": float(constraints.get("mean", np.nan)),
            "damage_prevented": damage_prevented,
        }
        rows.append(row)

    # Build summary.
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    summary = {"methods": {}}
    seeds_per_method = {}

    for method, items in sorted(by_method.items()):
        rewards = [x["reward"] for x in items]
        fairness_vals = [x["fairness"] for x in items]
        constraint_vals = [x["constraint"] for x in items]

        summary["methods"][method] = {
            "n_runs": len(items),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_ci95": _ci95(rewards),
            "fairness_mean": float(np.mean(fairness_vals)),
            "fairness_std": float(np.std(fairness_vals)),
            "constraint_mean": float(np.mean(constraint_vals)),
            "constraint_std": float(np.std(constraint_vals)),
        }
        seeds_per_method[method] = len(items)

    # all_results.csv
    csv_path = artifacts_dir / "all_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "seed", "reward", "fairness", "constraint", "damage_prevented"],
        )
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x["method"], x["seed"])):
            writer.writerow(r)

    # plots
    if rows:
        _build_plots(rows, plots_dir)

    # configs
    copied_config_paths = _copy_configs(root, configs_dir)

    # metadata
    metadata = {
        "git_commit_hash": _git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "number_of_methods": len(by_method),
        "number_of_seeds_per_method": seeds_per_method,
        "warnings": warnings,
        "copied_config_files": copied_config_paths,
    }
    with open(metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(artifacts_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"total methods collected: {len(by_method)}")
    print("total seeds per method:")
    for m in sorted(seeds_per_method):
        print(f"  {m}: {seeds_per_method[m]}")

    if strict and len(by_method) < 5:
        raise RuntimeError("FAIL: fewer than 5 methods found with valid per-seed JSON outputs.")
    if not strict and len(by_method) < 5:
        print("WARNING: fewer than 5 methods found with valid per-seed JSON outputs.")

    print("Artifact ready for external review")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true", help="Fail when fewer than 5 methods are found")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(strict=args.strict))
