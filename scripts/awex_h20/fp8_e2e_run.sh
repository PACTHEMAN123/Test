#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/fuse/verl-e2e}
export VERL_SOURCE=${VERL_SOURCE:-$ROOT/src/verl-fp8-e2e}
export AWEX_SOURCE=${AWEX_SOURCE:-$ROOT/src/Awex-fp8-blockwise}
export CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:?set nccl or awex_weightrail}
case "$CHECKPOINT_BACKEND" in nccl|awex_weightrail) ;; *) exit 2 ;; esac
export RUN_ROOT=${RUN_ROOT:-$ROOT/runs/verl-fp8-e2e-20261008/$CHECKPOINT_BACKEND}
export ROLLOUT_QUANTIZATION=fp8
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export TARGET_TRAIN_STEPS=${TARGET_TRAIN_STEPS:-10}
export PROMPT_BATCH=32 ROLLOUT_N=4 ACTOR_MICRO_BATCH=32
export ACTOR_TP=${ACTOR_TP:-2} ACTOR_PP=${ACTOR_PP:-1} ACTOR_CP=${ACTOR_CP:-2}
export ACTOR_EP=${ACTOR_EP:-8} ACTOR_ETP=${ACTOR_ETP:-1}
export AWEX_PROFILE=1 AWEX_PROFILE_SYNC_START=1 AWEX_PROFILE_WARMUP_UPDATES=3
export AWEX_NCCL_DEVICE_V2_FP8_BLOCKWISE=1
export AWEX_NCCL_DEVICE_V2_RING_BROADCAST=1 AWEX_NCCL_DEVICE_V2_RING_SWIZZLE=1
export AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES=131072 AWEX_NCCL_DEVICE_V2_GIN_FIFO_DEPTH=16
export AWEX_NCCL_DEVICE_V2_HCA_POLICY=balanced NCCL_IGNORE_CPU_AFFINITY=0
export VERL_FP8_UPDATE_PROFILE=${VERL_FP8_UPDATE_PROFILE:-1}
export VERL_WEIGHT_PAYLOAD_PROFILE=${VERL_WEIGHT_PAYLOAD_PROFILE:-1}
export RAY_DEDUP_LOGS=0 VERL_LOGGING_LEVEL=INFO
# Use the identical NCCL runtime for both backends.
export LD_PRELOAD=/usr/local/cuda/lib64/libnccl.so.2
export PYTHONPATH="$AWEX_SOURCE:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT"
if [[ -e "$RUN_ROOT/run.log" ]]; then
  echo "Refusing to overwrite $RUN_ROOT/run.log" >&2
  exit 2
fi
git -C "$VERL_SOURCE" rev-parse HEAD > "$RUN_ROOT/verl-revision.txt"
git -C "$AWEX_SOURCE" rev-parse HEAD > "$RUN_ROOT/awex-revision.txt"
sha256sum "$AWEX_SOURCE/awex_nccl_device_ext_v2.so" > "$RUN_ROOT/extension-sha256.txt"
if bash "$VERL_SOURCE/scripts/awex_h20/4node_launch.sh" \
  actor_rollout_ref.rollout.load_format=dummy "$@" > "$RUN_ROOT/run.log" 2>&1; then
  printf '0\n' > "$RUN_ROOT/exit_code"
else
  status=$?
  printf '%s\n' "$status" > "$RUN_ROOT/exit_code"
  exit "$status"
fi
