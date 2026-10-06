#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/fuse/verl-e2e}
VERL_SOURCE=${VERL_SOURCE:-$ROOT/src/verl-direct-gin}
AWEX_SOURCE=${AWEX_SOURCE:-$ROOT/src/Awex-nccl-device-gin}
VENV=${VENV:-$ROOT/envs/verl-py312-torch213-cu132-vllm027-pilot}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen3-30B-A3B}
MATRIX_ROOT=${MATRIX_ROOT:-$ROOT/runs/awex-topology-matrix-20261006}
TARGET_TRAIN_STEPS=${TARGET_TRAIN_STEPS:-40}
PROFILE_WARMUP_UPDATES=${AWEX_PROFILE_WARMUP_UPDATES:-8}
PROMPT_BATCH=${PROMPT_BATCH:-32}
ROLLOUT_N=${ROLLOUT_N:-4}

mkdir -p "$MATRIX_ROOT"
manifest="$MATRIX_ROOT/manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'run\ttopology\tactor_tp\tactor_pp\tactor_cp\tactor_ep\tmegatron_fsdp\trollout_tp\tbackend\tstatus\n' > "$manifest"
fi

run_one() {
  local topology=$1
  local actor_tp=$2
  local actor_pp=$3
  local actor_cp=$4
  local actor_ep=$5
  local use_megatron_fsdp=$6
  local rollout_tp=$7
  local backend=$8
  local run_name="${topology}-rtp${rollout_tp}-${backend}"
  local run_root="$MATRIX_ROOT/$run_name"

  if [[ -f "$run_root/exit_code" ]] && [[ "$(<"$run_root/exit_code")" == 0 ]]; then
    echo "Skipping completed run: $run_root"
    return 0
  fi
  if [[ -e "$run_root/run.log" ]]; then
    echo "Refusing to overwrite existing run: $run_root" >&2
    return 2
  fi

  mkdir -p "$run_root"
  printf 'target_train_steps\tprofile_warmup_updates\n%s\t%s\n' \
    "$TARGET_TRAIN_STEPS" "$PROFILE_WARMUP_UPDATES" > "$run_root/protocol.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\trunning\n' \
    "$run_name" "$topology" "$actor_tp" "$actor_pp" "$actor_cp" \
    "$actor_ep" "$use_megatron_fsdp" "$rollout_tp" "$backend" >> "$manifest"

  if env \
    ROOT="$ROOT" \
    VERL_SOURCE="$VERL_SOURCE" \
    AWEX_SOURCE="$AWEX_SOURCE" \
    VENV="$VENV" \
    MODEL_PATH="$MODEL_PATH" \
    RUN_ROOT="$run_root" \
    PROMPT_BATCH="$PROMPT_BATCH" \
    ROLLOUT_N="$ROLLOUT_N" \
    ROLLOUT_TP="$rollout_tp" \
    ACTOR_TP="$actor_tp" \
    ACTOR_PP="$actor_pp" \
    ACTOR_CP="$actor_cp" \
    ACTOR_EP="$actor_ep" \
    ACTOR_ETP=1 \
    USE_MEGATRON_FSDP="$use_megatron_fsdp" \
    TARGET_TRAIN_STEPS="$TARGET_TRAIN_STEPS" \
    AWEX_PROFILE_WARMUP_UPDATES="$PROFILE_WARMUP_UPDATES" \
    CHECKPOINT_BACKEND="$backend" \
    NCCL_IB_HCA_MODE=balanced \
    bash "$VERL_SOURCE/scripts/awex_h20/4node_launch.sh" \
      > "$run_root/run.log" 2>&1; then
    printf '%s\n' 0 > "$run_root/exit_code"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tcomplete\n' \
      "$run_name" "$topology" "$actor_tp" "$actor_pp" "$actor_cp" \
      "$actor_ep" "$use_megatron_fsdp" "$rollout_tp" "$backend" >> "$manifest"
  else
    exit_code=$?
    printf '%s\n' "$exit_code" > "$run_root/exit_code"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfailed:%s\n' \
      "$run_name" "$topology" "$actor_tp" "$actor_pp" "$actor_cp" \
      "$actor_ep" "$use_megatron_fsdp" "$rollout_tp" "$backend" "$exit_code" >> "$manifest"
    return "$exit_code"
  fi
}

# Latin-square backend order limits systematic drift across adjacent runs.
run_one p4t2c2e4 2 4 2 4 0 2 nccl
run_one p4t2c2e4 2 4 2 4 0 2 awex_nccl
run_one p4t2c2e4 2 4 2 4 0 2 awex_weightrail
run_one p4t2c2e4 2 4 2 4 0 4 awex_nccl
run_one p4t2c2e4 2 4 2 4 0 4 awex_weightrail
run_one p4t2c2e4 2 4 2 4 0 4 nccl
run_one p4t2c2e4 2 4 2 4 0 8 awex_weightrail
run_one p4t2c2e4 2 4 2 4 0 8 nccl
run_one p4t2c2e4 2 4 2 4 0 8 awex_nccl

run_one p1t4c2e8 4 1 2 8 1 2 awex_nccl
run_one p1t4c2e8 4 1 2 8 1 2 nccl
run_one p1t4c2e8 4 1 2 8 1 2 awex_weightrail
run_one p1t4c2e8 4 1 2 8 1 4 nccl
run_one p1t4c2e8 4 1 2 8 1 4 awex_weightrail
run_one p1t4c2e8 4 1 2 8 1 4 awex_nccl
run_one p1t4c2e8 4 1 2 8 1 8 awex_weightrail
run_one p1t4c2e8 4 1 2 8 1 8 awex_nccl
run_one p1t4c2e8 4 1 2 8 1 8 nccl
