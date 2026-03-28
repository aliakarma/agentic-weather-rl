"""Statistical analysis for experiment outputs.

Usage:
  python scripts/analyze_results.py --results results
  python scripts/analyze_results.py --results results --compare results/mappo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


def _seed_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))


def _load_rows(results_dir: Path) -> List[dict]:
    rows = []
    for fp in _seed_files(results_dir):
        with open(fp, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No seed_*.json files found in {results_dir}")
    return rows


def _ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return {"low": float(arr[0]), "high": float(arr[0])}
    sem = stats.sem(arr)
    low, high = stats.t.interval(0.95, len(arr) - 1, loc=float(np.mean(arr)), scale=float(sem))
    return {"low": float(low), "high": float(high)}


def _summary(rows: List[dict]) -> dict:
    rewards = [float(r["reward"]) for r in rows]
    fairness = [float(r["fairness_metrics"]["jain_action_index"]) for r in rows]
    constraints = [float(r["constraint_violations"]["mean"]) for r in rows]

    return {
        "n_runs": len(rows),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_ci95": _ci95(rewards),
        "fairness_mean": float(np.mean(fairness)),
        "fairness_std": float(np.std(fairness)),
        "fairness_ci95": _ci95(fairness),
        "constraint_mean": float(np.mean(constraints)),
        "constraint_std": float(np.std(constraints)),
        "constraint_ci95": _ci95(constraints),
    }


def _paired_t_test(a_rows: List[dict], b_rows: List[dict]) -> dict:
    a_map = {int(r["seed"]): float(r["reward"]) for r in a_rows}
    b_map = {int(r["seed"]): float(r["reward"]) for r in b_rows}
    shared = sorted(set(a_map).intersection(b_map))
    if len(shared) < 2:
        return {"n": len(shared), "t_stat": float("nan"), "p_value": float("nan")}
    a = np.array([a_map[s] for s in shared], dtype=np.float64)
    b = np.array([b_map[s] for s in shared], dtype=np.float64)
    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "n": len(shared),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }


def main(results: str, compare: str | None) -> None:
    results_dir = Path(results)
    rows = _load_rows(results_dir)

    payload = {
        "results_dir": str(results_dir),
        "summary": _summary(rows),
    }

    if compare:
        compare_rows = _load_rows(Path(compare))
        payload["paired_t_test_reward"] = _paired_t_test(rows, compare_rows)

    out_path = results_dir / "analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--compare", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.results, args.compare)
