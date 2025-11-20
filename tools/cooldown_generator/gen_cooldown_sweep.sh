#!/usr/bin/env bash
set -euo pipefail

# Emit ONE command per file:
#   cooldown_id<N>_exact.sh
#   cooldown_id<N>_rollback.sh   (only if rollback > 0)
#
# ID -> data list mapping:
#   IDs 1..4 -> olmo-mix-1124.txt
#   IDs 5..7 -> aurora/dolmino-mix-1124-fused-file-list.txt
#
# Example:
#   ./run_cooldown_per_id_split.sh \
#     --load /path/to/checkpoints_parent \
#     --cool-R 2000 \
#     --emit-dir ./cooldown_out \
#     --phase1-list ALCF/data-lists/olmo-mix-1124.txt \
#     --phase2-list ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
#     --train-script train_alcf.sh \
#     --extra-args "--wandb-tag cooldown --wandb-note per_id_split"
#
# Optional knobs:
#     --tokens-per-step 50331648   # 8192*6144
#     --ttokens 8                  # produce 1..7T
#     --cooldown-percent 0.05
#     --round 50
#     --python python3

# Defaults
EMIT_DIR="${PWD}/cooldown_out"
COOL_R=""
LOAD_PATH=""
PHASE1_LIST=""
PHASE2_LIST=""
TRAIN_SCRIPT="train_alcf.sh"
EXTRA_ARGS=""
TOKENS_PER_STEP=$((8192*6144))
T_TOTAL=8
COOLDOWN_PCT="0.05"
ROUND_TO="50"
PYTHON="${PYTHON:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --emit-dir) EMIT_DIR="$2"; shift 2 ;;
    --cool-R) COOL_R="$2"; shift 2 ;;
    --load) LOAD_PATH="$2"; shift 2 ;;
    --phase1-list) PHASE1_LIST="$2"; shift 2 ;;
    --phase2-list) PHASE2_LIST="$2"; shift 2 ;;
    --train-script) TRAIN_SCRIPT="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --tokens-per-step) TOKENS_PER_STEP="$2"; shift 2 ;;
    --ttokens) T_TOTAL="$2"; shift 2 ;;
    --cooldown-percent) COOLDOWN_PCT="$2"; shift 2 ;;
    --round) ROUND_TO="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "${COOL_R}" ]] || die "--cool-R (cooldown steps) is required"
[[ -n "${LOAD_PATH}" ]] || die "--load path is required"
[[ -n "${PHASE1_LIST}" ]] || die "--phase1-list is required (IDs 1..4)"
[[ -n "${PHASE2_LIST}" ]] || die "--phase2-list is required (IDs 5..7)"

mkdir -p "${EMIT_DIR}"

# 1) Build the checkpoint table (1..7T + rollback)
${PYTHON} build_checkpoints_from_tokens.py \
  --ttokens "${T_TOTAL}" \
  --tokens-per-step "${TOKENS_PER_STEP}" \
  --cooldown-percent "${COOLDOWN_PCT}" \
  --round "${ROUND_TO}" \
  --out "${EMIT_DIR}/checkpoints.tsv"

# 2) For each row, emit two single-command files (exact + rollback if > 0)
tail -n +2 "${EMIT_DIR}/checkpoints.tsv" | while IFS=$'\t' read -r id smod srb; do
  # choose data list by ID range
  if (( id >= 1 && id <= 4 )); then
    DATA_LIST="${PHASE1_LIST}"
  elif (( id >= 5 && id <= 7 )); then
    DATA_LIST="${PHASE2_LIST}"
  else
    echo "Skipping unknown id ${id}" >&2
    continue
  fi

  # exact-T
  OUT_EX="${EMIT_DIR}/cooldown_id${id}_exact.sh"
  ${PYTHON} make_cooldown_cmds.py \
    --load "${LOAD_PATH}" \
    --data-file-list "${DATA_LIST}" \
    --train-script "${TRAIN_SCRIPT}" \
    --checkpoint-ids "${id}" \
    --checkpoint-iters "${smod}" \
    --cooldown-steps "${COOL_R}" \
    --extra-args "${EXTRA_ARGS}" \
    --emit-sh "${OUT_EX}"
  chmod +x "${OUT_EX}"
  echo "Wrote ${OUT_EX}"

  # rollback (only if positive)
  if (( srb > 0 )); then
    OUT_RB="${EMIT_DIR}/cooldown_id${id}_rollback.sh"
    ${PYTHON} make_cooldown_cmds.py \
      --load "${LOAD_PATH}" \
      --data-file-list "${DATA_LIST}" \
      --train-script "${TRAIN_SCRIPT}" \
      --checkpoint-ids "${id}" \
      --checkpoint-iters "${srb}" \
      --cooldown-steps "${COOL_R}" \
      --extra-args "${EXTRA_ARGS}" \
      --emit-sh "${OUT_RB}"
    chmod +x "${OUT_RB}"
    echo "Wrote ${OUT_RB}"
  fi
done

echo "Per-ID single-command scripts written to: ${EMIT_DIR}"

