#!/usr/bin/env bash
# =============================================================================
# train_baselines.sh
# =============================================================================
# Train and evaluate all six baseline methods from Table 2 of the paper:
#
#   Method      | Expected Reward | Expected VR
#   ------------|-----------------|------------
#   Heuristic   |  42.1 ± 1.8     |  18.3%
#   DQN         |  55.6 ± 3.2     |  14.7%
#   IPPO        |  63.4 ± 4.1     |  12.1%
#   QMIX        |  69.8 ± 3.7     |  10.5%
#   MAPPO       |  74.3 ± 3.0     |   8.9%
#   CPO         |  71.2 ± 2.8     |   4.1%
#
# The heuristic requires no training and is evaluated directly.
# All learned baselines are trained for 1000 episodes on a single seed.
# Full 5-seed reproduction of each baseline would require additional runs.
#
# Expected runtime : ~2–3 hours on A100  |  ~8 hours on CPU
#   Breakdown (single seed, 1000 episodes):
#     Heuristic : <1 min  (eval only)
#     DQN       : ~15 min
#     IPPO      : ~20 min
#     QMIX      : ~25 min
#     MAPPO     : ~20 min
#     CPO       : ~25 min
#
# Usage:
#   bash scripts/train_baselines.sh
#   bash scripts/train_baselines.sh --device cuda
#   bash scripts/train_baselines.sh --algo dqn ippo   # subset
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
EPISODES="${EPISODES:-1000}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/training.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-runs/baselines}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"

# Default: run all baselines
ALL_ALGOS=("heuristic" "dqn" "ippo" "qmix" "mappo" "cpo")
ALGOS=("${ALL_ALGOS[@]}")

# ── Parse optional CLI overrides ──────────────────────────────────────────────
EXTRA_ALGOS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)       EPISODES="$2";       shift 2 ;;
    --device)         DEVICE="$2";         shift 2 ;;
    --seed)           SEED="$2";           shift 2 ;;
    --config)         CONFIG="$2";         shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --log_dir)        LOG_DIR="$2";        shift 2 ;;
    --eval_episodes)  EVAL_EPISODES="$2";  shift 2 ;;
    --algo)           EXTRA_ALGOS+=("$2"); shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# If --algo was provided, restrict to those only
if [[ ${#EXTRA_ALGOS[@]} -gt 0 ]]; then
  ALGOS=("${EXTRA_ALGOS[@]}")
fi

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "results/example_results"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "  Risk-Aware MARL — Baseline Training"
echo "  Algorithms  : ${ALGOS[*]}"
echo "  Episodes    : ${EPISODES}"
echo "  Device      : ${DEVICE}"
echo "  Seed        : ${SEED}"
echo "  Eval eps    : ${EVAL_EPISODES}"
echo "============================================================"

# ── Track timing per algorithm ────────────────────────────────────────────────
declare -A TIMING=()

for ALGO in "${ALGOS[@]}"; do
  echo ""
  echo "────────────────────────────────────────────────────────────"
  echo "  Algorithm: ${ALGO^^}"
  echo "────────────────────────────────────────────────────────────"

  T_START=$(date +%s)

  CKPT="${CHECKPOINT_DIR}/${ALGO}_seed${SEED}.pt"

  if [[ "${ALGO}" == "heuristic" ]]; then
    # ── Heuristic: evaluate directly, no training ──────────────────────────
    echo "  No training required for heuristic. Running evaluation..."
    python -m src.evaluate \
      --algo        heuristic                    \
      --checkpoint  "${CKPT}"                    \
      --episodes    "${EVAL_EPISODES}"           \
      --device      "${DEVICE}"                  \
      --output_dir  "results/example_results"    \
      --seed        $((10000 + SEED))            \
      2>/dev/null || \
    python - << PYEOF
# Heuristic has no checkpoint requirement — run evaluate directly
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from src.environment.disaster_env import DisasterEnv
from src.algorithms.baselines.heuristic import HeuristicPolicy
from src.evaluate import evaluate
env = DisasterEnv()
agent = HeuristicPolicy()
# HeuristicPolicy.act() not .get_actions() — wrap it
class HeuristicWrapper:
    def __init__(self, policy): self._p = policy
    def get_actions(self, obs, deterministic=True): return self._p.act(obs)
metrics = evaluate(
    agent=HeuristicWrapper(agent),
    env=env,
    n_episodes=${EVAL_EPISODES},
    seed_offset=$((10000 + SEED)),
    output_dir='results/example_results',
    algo_name='heuristic',
)
PYEOF

  else
    # ── Learned baseline: train then evaluate ─────────────────────────────
    echo "  Training ${ALGO} for ${EPISODES} episodes..."

    python -m src.train \
      --algo            "${ALGO}"          \
      --config          "${CONFIG}"        \
      --episodes        "${EPISODES}"      \
      --device          "${DEVICE}"        \
      --seed            "${SEED}"          \
      --checkpoint_dir  "${CHECKPOINT_DIR}"\
      --log_dir         "${LOG_DIR}/${ALGO}_seed${SEED}"

    # train.py saves to {algo}_final.pt for baselines
    SAVED="${CHECKPOINT_DIR}/${ALGO}_final.pt"
    if [[ -f "${SAVED}" ]]; then
      mv "${SAVED}" "${CKPT}"
      echo "  Checkpoint saved: ${CKPT}"
    fi

    echo ""
    echo "  Evaluating ${ALGO} over ${EVAL_EPISODES} episodes..."

    python -m src.evaluate \
      --algo        "${ALGO}"                  \
      --checkpoint  "${CKPT}"                  \
      --episodes    "${EVAL_EPISODES}"         \
      --device      "${DEVICE}"                \
      --output_dir  "results/example_results"  \
      --seed        $((10000 + SEED))
  fi

  # Rename result file to include algorithm tag
  SRC_JSON="results/example_results/eval_${ALGO}.json"
  if [[ -f "${SRC_JSON}" ]]; then
    cp "${SRC_JSON}" "results/example_results/eval_${ALGO}_seed${SEED}.json"
  fi

  T_END=$(date +%s)
  ELAPSED=$(( T_END - T_START ))
  TIMING["${ALGO}"]="${ELAPSED}s"
  echo "  ${ALGO} completed in ${ELAPSED}s"
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Baseline Results Summary"
echo "============================================================"

python - << 'PYEOF'
import json
import numpy as np
from pathlib import Path

result_dir = Path("results/example_results")

algos = ["heuristic", "dqn", "ippo", "qmix", "mappo", "cpo"]
paper = {
    "heuristic": (42.1, 18.3),
    "dqn":       (55.6, 14.7),
    "ippo":      (63.4, 12.1),
    "qmix":      (69.8, 10.5),
    "mappo":     (74.3,  8.9),
    "cpo":       (71.2,  4.1),
}

print(f"\n  {'Algorithm':<12} {'Reward':>9} {'VR%':>8} | {'Paper R':>9} {'Paper VR%':>10}")
print("  " + "-" * 55)
for algo in algos:
    fpath = result_dir / f"eval_{algo}.json"
    if not fpath.exists():
        print(f"  {algo:<12} {'(no result)':>9}")
        continue
    with open(fpath) as f:
        m = json.load(f)
    r  = m.get("reward_mean", float("nan"))
    vr = m.get("violation_rate_mean", float("nan")) * 100
    pr, pvr = paper.get(algo, (0, 0))
    print(f"  {algo:<12} {r:>9.2f} {vr:>8.1f}% | {pr:>9.1f} {pvr:>9.1f}%")
print()
PYEOF

echo ""
echo "  Timing summary:"
for ALGO in "${ALGOS[@]}"; do
  echo "    ${ALGO}: ${TIMING[${ALGO}]:-N/A}"
done

echo ""
echo "Baseline training complete."
echo "To view training curves:"
echo "  tensorboard --logdir ${LOG_DIR}"
echo "============================================================"
