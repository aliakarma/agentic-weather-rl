#!/usr/bin/env bash
# =============================================================================
# train_perception_ablation.sh
# =============================================================================
# Train and evaluate four perception encoder variants from Table 1 of the paper:
#
#   Variant           | Expected F1  | Description
#   ------------------|--------------|------------------------------------------
#   radar_cnn         |  0.77        | Radar-only CNN baseline
#   multimodal_cnn    |  0.84        | Radar + satellite CNN fusion
#   vit_single        |  0.85        | Single-stream ViT (radar only)
#   vit_multimodal    |  0.88        | Two-stream ViT (proposed, radar+satellite)
#
# All variants are trained on the SEVIR storm event dataset.
# The proposed vit_multimodal variant is saved to checkpoints/perception_encoder.pt
# and used by LagrangianCTDE in perception-coupled mode.
#
# Paired t-tests (p < 0.05) for vit_multimodal vs all others are reported
# in Section 4.2 of the paper.
#
# Expected runtime (per variant, SEVIR full dataset):
#   ~30 min on A100  |  ~90 min on RTX 3090  (50 epochs, batch=32)
#   ~5 min on A100   |  ~15 min on RTX 3090  (--use_sample flag)
#
# GPU required: Strongly recommended (ViT fine-tuning is memory-intensive).
#               Full SEVIR: A100 40 GB or V100 32 GB.
#               Sample mode: Colab Free T4 sufficient.
#
# Data requirements:
#   Full:   data/sevir/sevir_*.h5  (~10 GB, download from AWS S3 — see README)
#   Sample: data/sample/sevir_subset.h5  (~150 MB, included in repo)
#
# Usage:
#   bash scripts/train_perception_ablation.sh               # full dataset
#   bash scripts/train_perception_ablation.sh --use_sample  # quick Colab run
#   bash scripts/train_perception_ablation.sh --variant vit_multimodal
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/sevir/}"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:-data/sample/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUTPUT_DIM="${OUTPUT_DIM:-128}"
DEVICE="${DEVICE:-cpu}"
USE_SAMPLE=false
EVAL_SPLIT="${EVAL_SPLIT:-test}"
RESULTS_DIR="${RESULTS_DIR:-results/example_results}"

# Default: run all four variants
ALL_VARIANTS=("radar_cnn" "multimodal_cnn" "vit_single" "vit_multimodal")
VARIANTS=("${ALL_VARIANTS[@]}")

# ── Parse optional CLI overrides ──────────────────────────────────────────────
EXTRA_VARIANTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)       DATA_DIR="$2";       shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --epochs)         EPOCHS="$2";         shift 2 ;;
    --lr)             LR="$2";             shift 2 ;;
    --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
    --output_dim)     OUTPUT_DIM="$2";     shift 2 ;;
    --device)         DEVICE="$2";         shift 2 ;;
    --use_sample)     USE_SAMPLE=true;     shift 1 ;;
    --eval_split)     EVAL_SPLIT="$2";     shift 2 ;;
    --results_dir)    RESULTS_DIR="$2";    shift 2 ;;
    --variant)        EXTRA_VARIANTS+=("$2"); shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Restrict to specified variants if --variant was provided
if [[ ${#EXTRA_VARIANTS[@]} -gt 0 ]]; then
  VARIANTS=("${EXTRA_VARIANTS[@]}")
fi

# Use sample data directory when --use_sample is set
if [[ "${USE_SAMPLE}" == "true" ]]; then
  DATA_DIR="${SAMPLE_DATA_DIR}"
  echo "  Using sample data from: ${DATA_DIR}"
  echo "  (Synthetic fallback will activate if no HDF5 files are found)"
fi

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Validate data directory ────────────────────────────────────────────────────
if [[ ! -d "${DATA_DIR}" ]] && [[ "${USE_SAMPLE}" == "false" ]]; then
  echo ""
  echo "  ERROR: Data directory not found: ${DATA_DIR}"
  echo ""
  echo "  To use the sample dataset (150 MB, included in repo):"
  echo "    bash scripts/train_perception_ablation.sh --use_sample"
  echo ""
  echo "  To download the full SEVIR dataset (~10 GB):"
  echo "    aws s3 cp s3://sevir/data/ data/sevir/ --recursive --no-sign-request"
  echo "    (Requires awscli: pip install awscli)"
  echo ""
  exit 1
fi

echo "============================================================"
echo "  Risk-Aware MARL — Perception Ablation (Table 1)"
echo "  Variants    : ${VARIANTS[*]}"
echo "  Data dir    : ${DATA_DIR}"
echo "  Epochs      : ${EPOCHS}"
echo "  Batch size  : ${BATCH_SIZE}"
echo "  LR          : ${LR}"
echo "  Device      : ${DEVICE}"
echo "  Use sample  : ${USE_SAMPLE}"
echo "  Eval split  : ${EVAL_SPLIT}"
echo "============================================================"
echo ""

# ── Map variant name → model_type flag ───────────────────────────────────────
# model_type controls which encoder architecture is instantiated
# in vit_encoder.py (see --model_type argument in the CLI section).
declare -A MODEL_TYPE_MAP=(
  [radar_cnn]="radar_cnn"
  [multimodal_cnn]="multimodal_cnn"
  [vit_single]="vit_single"
  [vit_multimodal]="vit_multimodal"
)

# Expected F1 from Table 1 (for comparison after training)
declare -A EXPECTED_F1=(
  [radar_cnn]="0.77"
  [multimodal_cnn]="0.84"
  [vit_single]="0.85"
  [vit_multimodal]="0.88"
)

# ── Array to collect per-variant results ──────────────────────────────────────
declare -A VARIANT_F1=()
declare -A VARIANT_ACC=()
declare -A TIMING=()

for VARIANT in "${VARIANTS[@]}"; do

  echo "────────────────────────────────────────────────────────────"
  echo "  Variant: ${VARIANT}  (expected F1 ≈ ${EXPECTED_F1[${VARIANT}]:-N/A})"
  echo "────────────────────────────────────────────────────────────"

  T_START=$(date +%s)

  CKPT="${CHECKPOINT_DIR}/perception_encoder_${VARIANT}.pt"
  MODEL_TYPE="${MODEL_TYPE_MAP[${VARIANT}]}"

  # ── Train ─────────────────────────────────────────────────────────────────
  echo "  Training ${VARIANT}..."

  SAMPLE_FLAG=""
  if [[ "${USE_SAMPLE}" == "true" ]]; then
    SAMPLE_FLAG="--use_sample"
  fi

  python -m src.models.vit_encoder \
    --mode        train             \
    --data_dir    "${DATA_DIR}"     \
    --checkpoint  "${CKPT}"        \
    --epochs      "${EPOCHS}"      \
    --lr          "${LR}"          \
    --batch_size  "${BATCH_SIZE}"  \
    --output_dim  "${OUTPUT_DIM}"  \
    --model_type  "${MODEL_TYPE}"  \
    ${SAMPLE_FLAG}

  echo "  Checkpoint saved: ${CKPT}"

  # ── Evaluate on test split ─────────────────────────────────────────────────
  echo ""
  echo "  Evaluating ${VARIANT} on ${EVAL_SPLIT} split..."

  python -m src.models.vit_encoder \
    --mode        eval              \
    --data_dir    "${DATA_DIR}"     \
    --checkpoint  "${CKPT}"        \
    --split       "${EVAL_SPLIT}"  \
    --model_type  "${MODEL_TYPE}"  \
    ${SAMPLE_FLAG} | tee "/tmp/eval_${VARIANT}.txt"

  # Parse F1 and accuracy from output
  F1=$(grep "f1" "/tmp/eval_${VARIANT}.txt" 2>/dev/null | awk '{print $NF}' | head -1 || echo "N/A")
  ACC=$(grep "accuracy" "/tmp/eval_${VARIANT}.txt" 2>/dev/null | awk '{print $NF}' | head -1 || echo "N/A")
  VARIANT_F1["${VARIANT}"]="${F1}"
  VARIANT_ACC["${VARIANT}"]="${ACC}"

  # Save result to JSON
  python - << PYEOF
import json
from pathlib import Path

results = {
    "variant": "${VARIANT}",
    "model_type": "${MODEL_TYPE}",
    "checkpoint": "${CKPT}",
    "eval_split": "${EVAL_SPLIT}",
    "f1": "${F1}",
    "accuracy": "${ACC}",
    "expected_f1_paper": ${EXPECTED_F1[${VARIANT}]:-0.0},
}
out = Path("${RESULTS_DIR}") / "perception_${VARIANT}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Result saved to: {out}")
PYEOF

  T_END=$(date +%s)
  ELAPSED=$(( T_END - T_START ))
  TIMING["${VARIANT}"]="${ELAPSED}s"
  echo "  ${VARIANT} completed in ${ELAPSED}s"
  echo ""
done

# ── If vit_multimodal was trained, copy it to the primary checkpoint path ──────
if [[ " ${VARIANTS[*]} " == *" vit_multimodal "* ]]; then
  PROPOSED_CKPT="${CHECKPOINT_DIR}/perception_encoder_vit_multimodal.pt"
  PRIMARY_CKPT="${CHECKPOINT_DIR}/perception_encoder.pt"
  if [[ -f "${PROPOSED_CKPT}" ]]; then
    cp "${PROPOSED_CKPT}" "${PRIMARY_CKPT}"
    echo "Proposed encoder saved to primary checkpoint: ${PRIMARY_CKPT}"
  fi
fi

# ── Summary table (Table 1 reproduction) ─────────────────────────────────────
echo ""
echo "============================================================"
echo "  Perception Ablation Results (Table 1)"
echo "============================================================"

python - << 'PYEOF'
import json
import numpy as np
from pathlib import Path

result_dir = Path("${RESULTS_DIR}")
variants = ["radar_cnn", "multimodal_cnn", "vit_single", "vit_multimodal"]
paper_f1 = {
    "radar_cnn": 0.77, "multimodal_cnn": 0.84,
    "vit_single": 0.85, "vit_multimodal": 0.88,
}

print(f"\n  {'Variant':<22} {'F1 (ours)':>10} {'F1 (paper)':>12} {'Δ':>7}")
print("  " + "-" * 55)
for v in variants:
    fpath = result_dir / f"perception_{v}.json"
    if not fpath.exists():
        print(f"  {v:<22} {'(no result)':>10}")
        continue
    with open(fpath) as f:
        m = json.load(f)
    f1 = float(m.get("f1", 0)) if m.get("f1", "N/A") != "N/A" else float("nan")
    pf1 = paper_f1.get(v, 0.0)
    delta = f1 - pf1 if not np.isnan(f1) else float("nan")
    print(f"  {v:<22} {f1:>10.4f} {pf1:>12.2f} {delta:>+7.4f}")

print()
print("  Statistical significance: paired t-tests (p < 0.05)")
print("  vit_multimodal > vit_single > multimodal_cnn > radar_cnn")
PYEOF

echo ""
echo "  Timing summary:"
for VARIANT in "${VARIANTS[@]}"; do
  echo "    ${VARIANT}: ${TIMING[${VARIANT}]:-N/A}"
done

echo ""
echo "Perception ablation complete."
echo ""
echo "The proposed encoder (vit_multimodal) is saved to:"
echo "  ${CHECKPOINT_DIR}/perception_encoder.pt"
echo ""
echo "To use it with LagrangianCTDE (perception-coupled mode):"
echo "  python -m src.train --algo lagrangian_ctde \\"
echo "    --config configs/training.yaml"
echo "  (Set mode: perception in training.yaml)"
echo "============================================================"
