#!/usr/bin/env bash
# =============================================================================
# demo.sh
# =============================================================================
# Load the pretrained LagrangianCTDE checkpoint and run a short evaluation.
#
# If either checkpoint is missing, scripts/create_checkpoints.py is run
# automatically to produce valid (randomly initialised) placeholder files
# so the demo works immediately after cloning.
#
# Expected runtime : ~5 minutes on CPU
# GPU required     : No (CPU sufficient for evaluation)
# Episodes         : 20 (quick demo; full paper uses 500)
#
# Outputs:
#   - Reward, violation rate, per-agent decision accuracy printed to stdout
#   - Results JSON: results/example_results/eval_lagrangian_ctde.json
#
# Usage:
#   bash scripts/demo.sh
#   bash scripts/demo.sh --episodes 50      # more episodes
#   bash scripts/demo.sh --device cuda      # GPU evaluation
#   bash scripts/demo.sh --checkpoint checkpoints/marl_policy.pt
# =============================================================================

set -euo pipefail

# ── Locate project root (works regardless of where the script is invoked from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Add project root to PYTHONPATH ────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Default configuration ─────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-checkpoints/marl_policy.pt}"
EPISODES="${EPISODES:-20}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-results/example_results}"
ALGO="lagrangian_ctde"

# ── Parse optional CLI overrides ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --episodes)   EPISODES="$2";   shift 2 ;;
    --device)     DEVICE="$2";     shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Print header ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Risk-Aware MARL — Demo Evaluation"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Episodes   : ${EPISODES}"
echo "  Device     : ${DEVICE}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "============================================================"

# ── Auto-create checkpoints if they do not exist ──────────────────────────────
# This block runs scripts/create_checkpoints.py when either required checkpoint
# is absent. The script generates deterministically seeded, randomly initialised
# weights that are structurally identical to trained checkpoints and are fully
# loadable by evaluate.py / LagrangianCTDE.load_checkpoint().
#
# Replace the generated files with trained checkpoints for meaningful results:
#   bash scripts/train_short.sh   (100 episodes, ~20 min on GPU)
#   bash scripts/train_full.sh    (1000 episodes x 5 seeds, ~2 h on A100)
# ─────────────────────────────────────────────────────────────────────────────

NEED_CHECKPOINT_GEN=false

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo ""
  echo "  Checkpoint not found: ${CHECKPOINT}"
  NEED_CHECKPOINT_GEN=true
fi

if [[ ! -f "checkpoints/perception_encoder.pt" ]]; then
  echo "  Perception encoder not found: checkpoints/perception_encoder.pt"
  NEED_CHECKPOINT_GEN=true
fi

if [[ "${NEED_CHECKPOINT_GEN}" == "true" ]]; then
  echo ""
  echo "  Generating placeholder checkpoints ..."
  echo "  (randomly initialised weights — run train_short.sh for trained weights)"
  echo ""

  # Try with real ViT encoder first; fall back to stub if timm is absent.
  if python -c "import timm" 2>/dev/null; then
    python scripts/create_checkpoints.py \
      --out_dir "checkpoints" \
      --seed    0             \
      --device  "${DEVICE}"
  else
    python scripts/create_checkpoints.py \
      --out_dir "checkpoints" \
      --seed    0             \
      --device  "${DEVICE}"
  fi

  echo ""
  echo "  Checkpoints generated."
fi

# ── Final checkpoint check ────────────────────────────────────────────────────
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo ""
  echo "  ERROR: Checkpoint still missing after generation attempt: ${CHECKPOINT}"
  echo ""
  echo "  Possible causes:"
  echo "    1. PyTorch is not installed. Install it:"
  echo "         pip install torch"
  echo "    2. The project root is not on PYTHONPATH. From project root, run:"
  echo "         PYTHONPATH=. bash scripts/demo.sh"
  echo "    3. Manual generation:"
  echo "         python scripts/create_checkpoints.py --out_dir checkpoints"
  echo ""
  exit 1
fi

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ── Run evaluation ────────────────────────────────────────────────────────────
echo ""
echo "Running evaluation (${EPISODES} episodes)..."
echo ""

python -m src.evaluate \
  --algo        "${ALGO}"       \
  --checkpoint  "${CHECKPOINT}" \
  --episodes    "${EPISODES}"   \
  --device      "${DEVICE}"     \
  --output_dir  "${OUTPUT_DIR}" \
  --seed        10000

# ── Post-evaluation summary ───────────────────────────────────────────────────
echo ""
echo "Demo complete."
echo "Results saved to: ${OUTPUT_DIR}/eval_${ALGO}.json"
echo ""

TRAINED=$(python - << 'PYEOF' 2>/dev/null
import torch, sys
try:
    ckpt = torch.load("checkpoints/marl_policy.pt", map_location="cpu", weights_only=True)
    print("yes" if ckpt.get("trained", True) else "no")
except Exception:
    print("unknown")
PYEOF
)

if [[ "${TRAINED}" == "no" ]]; then
  echo "  NOTE: The loaded checkpoint contains RANDOMLY INITIALISED weights."
  echo "        Results above are NOT comparable to paper Table 2."
  echo "        To produce meaningful results, train first:"
  echo ""
  echo "          bash scripts/train_short.sh    # ~20 min on GPU"
  echo "          bash scripts/train_full.sh     # ~2 h on A100 (5 seeds)"
  echo ""
else
  echo "  Expected results (5-seed mean, Table 2 of paper):"
  echo "    Reward         : 81.5 +/- 2.6"
  echo "    Violation Rate : 0.023 +/- 0.003"
  echo "    Decision Acc.  : Storm=0.91  Flood=0.87  Evacuation=0.84"
fi

echo "============================================================"
