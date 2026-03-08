#!/usr/bin/env bash
# =============================================================================
# train_full.sh
# =============================================================================
# Full training of Lagrangian CTDE-PPO across 5 random seeds.
# Reproduces the results reported in Table 2 of the paper:
#   Reward = 81.5 ± 2.6,  Violation Rate = 2.3%
#
# Expected runtime : ~2 hours on A100 40 GB  |  ~8 hours on RTX 3090
# GPU required     : Strongly recommended (Colab Pro A100 or equivalent)
# Episodes         : 1000 per seed (5000 total)
# Seeds            : 42 43 44 45 46  (fixed for reproducibility)
#
# Outputs (per seed):
#   checkpoints/marl_policy_seed{N}.pt
#   runs/full_seed{N}/         (TensorBoard logs)
#
# After all seeds complete, aggregate evaluation:
#   bash scripts/demo.sh  (uses best checkpoint from seed 42 by default)
#
# Usage:
#   bash scripts/train_full.sh
#   bash scripts/train_full.sh --device cuda
#   bash scripts/train_full.sh --seeds "42 43"   # subset of seeds
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
EPISODES="${EPISODES:-1000}"
DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-42 43 44 45 46}"
CONFIG="${CONFIG:-configs/training.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-runs/full}"
EVAL_EPISODES="${EVAL_EPISODES:-500}"

# ── Parse optional CLI overrides ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)       EPISODES="$2";       shift 2 ;;
    --device)         DEVICE="$2";         shift 2 ;;
    --seeds)          SEEDS="$2";          shift 2 ;;
    --config)         CONFIG="$2";         shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --log_dir)        LOG_DIR="$2";        shift 2 ;;
    --eval_episodes)  EVAL_EPISODES="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "results/example_results"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "  Risk-Aware MARL — Full Training (5 Seeds)"
echo "  Episodes/seed : ${EPISODES}"
echo "  Seeds         : ${SEEDS}"
echo "  Device        : ${DEVICE}"
echo "  Config        : ${CONFIG}"
echo "  Checkpoints   : ${CHECKPOINT_DIR}/"
echo "  TensorBoard   : ${LOG_DIR}/"
echo "============================================================"
echo ""

# ── Array to accumulate per-seed results ─────────────────────────────────────
declare -a SEED_REWARDS=()
declare -a SEED_VR=()

SEED_COUNT=0

for SEED in ${SEEDS}; do
  SEED_COUNT=$((SEED_COUNT + 1))
  CKPT_SEED="${CHECKPOINT_DIR}/marl_policy_seed${SEED}.pt"
  LOG_SEED="${LOG_DIR}/seed${SEED}"

  echo ""
  echo "────────────────────────────────────────────────────────────"
  echo "  Seed ${SEED}  (${SEED_COUNT}/${#SEEDS[@]} — counting: $(echo ${SEEDS} | wc -w) total)"
  echo "────────────────────────────────────────────────────────────"
  echo ""

  # ── Training ────────────────────────────────────────────────────────────────
  python -m src.train \
    --algo            lagrangian_ctde     \
    --config          "${CONFIG}"         \
    --episodes        "${EPISODES}"       \
    --device          "${DEVICE}"         \
    --seed            "${SEED}"           \
    --checkpoint_dir  "${CHECKPOINT_DIR}" \
    --log_dir         "${LOG_SEED}"

  # Rename checkpoint to include seed tag
  if [[ -f "${CHECKPOINT_DIR}/marl_policy.pt" ]]; then
    cp "${CHECKPOINT_DIR}/marl_policy.pt" "${CKPT_SEED}"
    echo "Checkpoint saved: ${CKPT_SEED}"
  fi

  # ── Per-seed evaluation (500 episodes) ──────────────────────────────────────
  echo ""
  echo "Evaluating seed ${SEED} over ${EVAL_EPISODES} episodes..."

  python -m src.evaluate \
    --algo        lagrangian_ctde                             \
    --checkpoint  "${CKPT_SEED}"                             \
    --episodes    "${EVAL_EPISODES}"                         \
    --device      "${DEVICE}"                                \
    --output_dir  "results/example_results"                  \
    --seed        $((10000 + SEED))

  # Rename result file to include seed tag
  SRC_JSON="results/example_results/eval_lagrangian_ctde.json"
  if [[ -f "${SRC_JSON}" ]]; then
    cp "${SRC_JSON}" "results/example_results/eval_lagrangian_ctde_seed${SEED}.json"
  fi

  echo "Seed ${SEED} complete."
done

# ── Aggregate results across all seeds ───────────────────────────────────────
echo ""
echo "============================================================"
echo "  Aggregating results across ${SEED_COUNT} seeds..."
echo "============================================================"

python - << 'PYEOF'
import json
import numpy as np
from pathlib import Path

result_dir = Path("results/example_results")
seeds = [int(s) for s in "$SEEDS".split()]

rewards, vrs, accs = [], [], {1: [], 2: [], 3: []}

for seed in seeds:
    fpath = result_dir / f"eval_lagrangian_ctde_seed{seed}.json"
    if not fpath.exists():
        print(f"  Warning: missing result for seed {seed}")
        continue
    with open(fpath) as f:
        m = json.load(f)
    rewards.append(m["reward_mean"])
    vrs.append(m["violation_rate_mean"])
    for i in range(1, 4):
        accs[i].append(m[f"decision_accuracy_{i}_mean"])

if not rewards:
    print("  No completed seed results found.")
else:
    print(f"\n  Seeds evaluated : {len(rewards)}")
    print(f"  Reward          : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Violation Rate  : {np.mean(vrs):.3f} ± {np.std(vrs):.3f}")
    for i in range(1, 4):
        agent_name = {1: "Storm   ", 2: "Flood   ", 3: "Evacuation"}[i]
        vals = accs[i]
        print(f"  DA Agent {i} ({agent_name}): {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"\n  Paper Table 2 targets:")
    print(f"    Reward=81.5±2.6  VR=2.3%  DA: 0.91 / 0.87 / 0.84")

    # Save aggregated result
    agg = {
        "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)),
        "violation_rate_mean": float(np.mean(vrs)), "violation_rate_std": float(np.std(vrs)),
        **{f"decision_accuracy_{i}_mean": float(np.mean(accs[i])) for i in range(1,4)},
        **{f"decision_accuracy_{i}_std":  float(np.std(accs[i]))  for i in range(1,4)},
        "n_seeds": len(rewards),
    }
    out = result_dir / "eval_lagrangian_ctde_aggregate.json"
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Aggregate saved to: {out}")
PYEOF

echo ""
echo "Full training complete."
echo ""
echo "To view training curves:"
echo "  tensorboard --logdir ${LOG_DIR}"
echo "============================================================"
