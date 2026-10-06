#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/fuse/verl-e2e}
VERL_SOURCE=${VERL_SOURCE:-$ROOT/src/verl-direct-gin}
AWEX_SOURCE=${AWEX_SOURCE:-$ROOT/src/Awex-ring-broadcast}
VENV=${VENV:-$ROOT/envs/verl-py312-torch213-cu132-vllm027-pilot}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen3-30B-A3B}
MATRIX_ROOT=${MATRIX_ROOT:-$ROOT/runs/awex-ring-broadcast-20261006}
RING_MODE=${RING_MODE:?set RING_MODE to off, naive, or swizzle}
RUN_LABEL=${RUN_LABEL:-$RING_MODE}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TARGET_TRAIN_STEPS=${TARGET_TRAIN_STEPS:-10}
PROFILE_WARMUP_UPDATES=${AWEX_PROFILE_WARMUP_UPDATES:-3}
PROMPT_BATCH=${PROMPT_BATCH:-32}
ROLLOUT_N=${ROLLOUT_N:-4}

case "$RING_MODE" in
  off)
    ring_broadcast=0
    ring_swizzle=0
    ;;
  naive)
    ring_broadcast=1
    ring_swizzle=0
    ;;
  swizzle)
    ring_broadcast=1
    ring_swizzle=1
    ;;
  *)
    echo "RING_MODE must be off, naive, or swizzle, got: $RING_MODE" >&2
    exit 2
    ;;
esac

case "$ROLLOUT_TP" in
  2|4|8) ;;
  *)
    echo "ROLLOUT_TP must be 2, 4, or 8, got: $ROLLOUT_TP" >&2
    exit 2
    ;;
esac

case "$RUN_LABEL" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_LABEL must contain only letters, digits, dots, underscores, or dashes" >&2
    exit 2
    ;;
esac

run_name="p1t2c2e8-rtp${ROLLOUT_TP}-ring-${RUN_LABEL}"
run_root="$MATRIX_ROOT/$run_name"
manifest="$MATRIX_ROOT/manifest.tsv"

mkdir -p "$MATRIX_ROOT"
if [[ ! -f "$manifest" ]]; then
  printf 'run\tring_mode\tring_broadcast\tring_swizzle\ttarget_train_steps\tprofile_warmup_updates\tstatus\n' > "$manifest"
fi
if [[ -f "$run_root/exit_code" ]] && [[ "$(<"$run_root/exit_code")" == 0 ]]; then
  echo "Skipping completed run: $run_root"
  exit 0
fi
if [[ -e "$run_root/run.log" ]]; then
  echo "Refusing to overwrite existing run: $run_root" >&2
  exit 2
fi

mkdir -p "$run_root"
printf 'ring_mode\tring_broadcast\tring_swizzle\trollout_tp\trollout_engines\ttarget_train_steps\tprofile_warmup_updates\n%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$RING_MODE" "$ring_broadcast" "$ring_swizzle" "$ROLLOUT_TP" \
  "$((16 / ROLLOUT_TP))" "$TARGET_TRAIN_STEPS" \
  "$PROFILE_WARMUP_UPDATES" > "$run_root/protocol.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\trunning\n' \
  "$run_name" "$RING_MODE" "$ring_broadcast" "$ring_swizzle" \
  "$TARGET_TRAIN_STEPS" "$PROFILE_WARMUP_UPDATES" >> "$manifest"

if env \
  ROOT="$ROOT" \
  VERL_SOURCE="$VERL_SOURCE" \
  AWEX_SOURCE="$AWEX_SOURCE" \
  VENV="$VENV" \
  MODEL_PATH="$MODEL_PATH" \
  RUN_ROOT="$run_root" \
  PROMPT_BATCH="$PROMPT_BATCH" \
  ROLLOUT_N="$ROLLOUT_N" \
  ROLLOUT_TP="$ROLLOUT_TP" \
  ACTOR_TP=2 \
  ACTOR_PP=1 \
  ACTOR_CP=2 \
  ACTOR_EP=8 \
  ACTOR_ETP=1 \
  USE_MEGATRON_FSDP=1 \
  TARGET_TRAIN_STEPS="$TARGET_TRAIN_STEPS" \
  AWEX_PROFILE_WARMUP_UPDATES="$PROFILE_WARMUP_UPDATES" \
  AWEX_NCCL_DEVICE_V2_RING_BROADCAST="$ring_broadcast" \
  AWEX_NCCL_DEVICE_V2_RING_SWIZZLE="$ring_swizzle" \
  CHECKPOINT_BACKEND=awex_weightrail \
  NCCL_IB_HCA_MODE=balanced \
  bash "$VERL_SOURCE/scripts/awex_h20/4node_launch.sh" \
    > "$run_root/run.log" 2>&1; then
  printf '%s\n' 0 > "$run_root/exit_code"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tcomplete\n' \
    "$run_name" "$RING_MODE" "$ring_broadcast" "$ring_swizzle" \
    "$TARGET_TRAIN_STEPS" "$PROFILE_WARMUP_UPDATES" >> "$manifest"
else
  exit_code=$?
  printf '%s\n' "$exit_code" > "$run_root/exit_code"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tfailed:%s\n' \
    "$run_name" "$RING_MODE" "$ring_broadcast" "$ring_swizzle" \
    "$TARGET_TRAIN_STEPS" "$PROFILE_WARMUP_UPDATES" "$exit_code" >> "$manifest"
  exit "$exit_code"
fi
