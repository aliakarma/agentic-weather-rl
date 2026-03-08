#!/usr/bin/env bash
# =============================================================================
# train_short.sh
# =============================================================================
# Short training run for Lagrangian CTDE-PPO (100 episodes).
# Intended for quick verification on Colab Free T4 or a local CPU.
#
# Expected runtime : ~20 minutes on A100 | ~60 minutes on CPU
# GPU required     : No (CPU sufficient; GPU strongly recommended)
# Episodes         : 100 (vs 1000 for full training)
# Seeds            : 1 (fixed seed=42)
#
# Outputs:
#   - Checkpoint saved to checkpoints/marl_policy_short.pt
#   - Training log to stdout (and TensorBoard if installed)
#
# After this run, evaluate with:
#   bash scripts/demo.sh --checkpoint checkpoints/marl_policy_short.pt
#
# Usage:
#   bash scripts/train_short.sh
#   bash scripts/train_short.sh --device cuda
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
EPISODES="${EPISODES:-100}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/training.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-runs/short}"

# ── Parse optional CLI overrides ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)       EPISODES="$2";       shift 2 ;;
    --device)         DEVICE="$2";         shift 2 ;;
    --seed)           SEED="$2";           shift 2 ;;
    --config)         CONFIG="$2";         shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --log_dir)        LOG_DIR="$2";        shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "  Risk-Aware MARL — Short Training Run"
echo "  Episodes   : ${EPISODES}"
echo "  Device     : ${DEVICE}"
echo "  Seed       : ${SEED}"
echo "  Config     : ${CONFIG}"
echo "  Checkpoint : ${CHECKPOINT_DIR}/"
echo "  TensorBoard: ${LOG_DIR}/"
echo "============================================================"
echo ""

# ── Auto-detect device if not set ────────────────────────────────────────────
if [[ "${DEVICE}" == "cpu" ]]; then
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || true
fi

# ── Training ──────────────────────────────────────────────────────────────────
python -m src.train \
  --algo            lagrangian_ctde     \
  --config          "${CONFIG}"         \
  --episodes        "${EPISODES}"       \
  --device          "${DEVICE}"         \
  --seed            "${SEED}"           \
  --checkpoint_dir  "${CHECKPOINT_DIR}" \
  --log_dir         "${LOG_DIR}"

# Rename checkpoint so it doesn't overwrite the pretrained best
if [[ -f "${CHECKPOINT_DIR}/marl_policy.pt" ]]; then
  cp "${CHECKPOINT_DIR}/marl_policy.pt" \
     "${CHECKPOINT_DIR}/marl_policy_short_seed${SEED}.pt"
  echo "Short checkpoint saved to: ${CHECKPOINT_DIR}/marl_policy_short_seed${SEED}.pt"
fi

echo ""
echo "Short training complete."
echo ""
echo "To evaluate:"
echo "  bash scripts/demo.sh --checkpoint ${CHECKPOINT_DIR}/marl_policy_short_seed${SEED}.pt"
echo ""
echo "To view training curves (requires tensorboard):"
echo "  tensorboard --logdir ${LOG_DIR}"
echo "============================================================"
