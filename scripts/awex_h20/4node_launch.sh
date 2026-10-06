#!/usr/bin/env bash
set -euo pipefail

# Standalone recipe for the verl fully-async entrypoint and pluggable checkpoint backends.
# Run this from the Ray head after all four nodes have joined the cluster.

ROOT=${ROOT:-/mnt/fuse/verl-e2e}
VERL_SOURCE=${VERL_SOURCE:-$ROOT/src/verl}
AWEX_SOURCE=${AWEX_SOURCE:-$ROOT/src/Awex-nccl-device-gin}
VENV=${VENV:-$ROOT/envs/verl-py312-torch213-cu132-vllm027-pilot}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen3-30B-A3B}
MODEL_LABEL=${MODEL_LABEL:-$(basename "$MODEL_PATH")}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/datasets/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$ROOT/datasets/gsm8k/test.parquet}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/${MODEL_LABEL}-fully-async}

PROMPT_BATCH=${PROMPT_BATCH:-128}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
TARGET_TRAIN_STEPS=${TARGET_TRAIN_STEPS:-32}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-$((PROMPT_BATCH * ROLLOUT_N * TARGET_TRAIN_STEPS))}
FSDP_SHARDING_STRATEGY=${FSDP_SHARDING_STRATEGY:-optim_grads}
WEIGHT_UPDATE_BUCKET_MB=${WEIGHT_UPDATE_BUCKET_MB:-256}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_ETP=${ACTOR_ETP:-1}
USE_DEEPEP=${USE_DEEPEP:-1}
MODE=${MODE:-run}
CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:-nccl}
CHECKPOINT_CUSTOM_BACKEND_MODULE=${CHECKPOINT_CUSTOM_BACKEND_MODULE:-}

case "$ROLLOUT_TP" in
  2|4|8) ;;
  *) echo "ROLLOUT_TP must be 2, 4, or 8, got $ROLLOUT_TP" >&2; exit 2 ;;
esac

case "$FSDP_SHARDING_STRATEGY" in
  optim_grads|optim_grads_params) ;;
  *) echo "FSDP_SHARDING_STRATEGY must be optim_grads or optim_grads_params, got $FSDP_SHARDING_STRATEGY" >&2; exit 2 ;;
esac

case "$WEIGHT_UPDATE_BUCKET_MB" in
  ''|*[!0-9]*|0) echo "WEIGHT_UPDATE_BUCKET_MB must be a positive integer, got $WEIGHT_UPDATE_BUCKET_MB" >&2; exit 2 ;;
esac

for value_name in ACTOR_EP ACTOR_ETP; do
  value=${!value_name}
  case "$value" in
    ''|*[!0-9]*|0) echo "$value_name must be a positive integer, got $value" >&2; exit 2 ;;
  esac
done

case "$USE_DEEPEP" in
  0|1) ;;
  *) echo "USE_DEEPEP must be 0 or 1, got $USE_DEEPEP" >&2; exit 2 ;;
esac

case "$CHECKPOINT_BACKEND" in
  nccl) ;;
  awex_nccl|awex_weightrail)
    CHECKPOINT_CUSTOM_BACKEND_MODULE=${CHECKPOINT_CUSTOM_BACKEND_MODULE:-awex.verl_checkpoint_engine}
    ;;
  *) echo "Unsupported CHECKPOINT_BACKEND: $CHECKPOINT_BACKEND" >&2; exit 2 ;;
esac

if [[ "$USE_DEEPEP" == 1 && "$ACTOR_EP" == 1 ]]; then
  echo "USE_DEEPEP=1 requires ACTOR_EP greater than 1" >&2
  exit 2
fi

for path in "$ROOT" "$VERL_SOURCE" "$VENV" "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE" "$RUN_ROOT"; do
  case "$path" in
    *'/oss/'*) echo "OSS paths are forbidden: $path" >&2; exit 2 ;;
  esac
done

for path in "$VERL_SOURCE" "$VENV/bin/python" "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required local path is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT"
source /opt/rh/gcc-toolset-12/enable
PYTHON="$VENV/bin/python"
NVSHMEM_DIR="$VENV/lib/python3.12/site-packages/nvidia/nvshmem"
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/scripts/vllm_preload:$VERL_SOURCE:${PYTHONPATH:-}"
if [[ "$CHECKPOINT_BACKEND" == awex_* ]]; then
  if [[ ! -d "$AWEX_SOURCE/awex" ]]; then
    echo "Required Awex source is missing: $AWEX_SOURCE" >&2
    exit 2
  fi
  export PYTHONPATH="$AWEX_SOURCE:$PYTHONPATH"
fi
if [[ "$CHECKPOINT_BACKEND" == awex_weightrail ]]; then
  export AWEX_NCCL_INCLUDE=${AWEX_NCCL_INCLUDE:-/usr/local/cuda/include}
  export AWEX_NCCL_LIB=${AWEX_NCCL_LIB:-/usr/local/cuda/lib64}
  export LD_PRELOAD=${LD_PRELOAD:-$AWEX_NCCL_LIB/libnccl.so.2}
  export LD_LIBRARY_PATH="$AWEX_NCCL_LIB:$LD_LIBRARY_PATH"
  export NCCL_CUMEM_ENABLE=1
  export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$ROOT/build-cache/torch-extensions-nccl230}
  export AWEX_PROFILE=${AWEX_PROFILE:-1}
  export AWEX_PROFILE_SYNC_START=${AWEX_PROFILE_SYNC_START:-1}
  export AWEX_PROFILE_WARMUP_UPDATES=${AWEX_PROFILE_WARMUP_UPDATES:-2}
  export AWEX_NCCL_DEVICE_V2_MAX_CHANNELS=${AWEX_NCCL_DEVICE_V2_MAX_CHANNELS:-64}
  if [[ -f "$AWEX_SOURCE/awex_nccl_device_ext_v2.so" ]]; then
    export AWEX_NCCL_DEVICE_V2_EXTENSION=${AWEX_NCCL_DEVICE_V2_EXTENSION:-awex_nccl_device_ext_v2}
  fi
fi
cd "$VERL_SOURCE"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_NCCL_ENABLE_MONITORING=0
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=1
export NVSHMEM_USE_GDRCOPY=0
export EP_DISABLE_GIN=0
export EP_BUFFER_DEBUG=${EP_BUFFER_DEBUG:-1}
export PYTHONUNBUFFERED=1

ppo_mini_batch_size=$((PROMPT_BATCH * ROLLOUT_N))
max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

hydra_flags=()
if [[ "$MODE" == config ]]; then
  hydra_flags+=(--cfg job --resolve)
elif [[ "$MODE" != run ]]; then
  echo "MODE must be run or config, got $MODE" >&2
  exit 2
fi

moe_flags=()
if [[ "$USE_DEEPEP" == 1 ]]; then
  moe_flags+=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_flex_dispatcher_backend=deepep
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
  )
fi

checkpoint_engine_flags=(
  actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_BACKEND"
)
if [[ -n "$CHECKPOINT_CUSTOM_BACKEND_MODULE" ]]; then
  checkpoint_engine_flags+=(
    actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module="$CHECKPOINT_CUSTOM_BACKEND_MODULE"
  )
fi

"$PYTHON" -m verl.experimental.fully_async_policy.fully_async_main \
  "${hydra_flags[@]}" \
  --config-name=fully_async_ppo_megatron_trainer \
  model_engine=megatron \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=0 \
  data.gen_batch_size=1 \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  critic.enable=False \
  reward.reward_model.enable=False \
  reward.reward_manager.name=naive \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKENS_PER_GPU" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.lr_decay_steps="$TARGET_TRAIN_STEPS" \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.megatron.param_offload=False \
  actor_rollout_ref.actor.megatron.grad_offload=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
  actor_rollout_ref.actor.megatron.context_parallel_size=2 \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size="$ACTOR_EP" \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="$ACTOR_ETP" \
  actor_rollout_ref.actor.megatron.sequence_parallel=True \
  actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
  actor_rollout_ref.actor.megatron.use_megatron_fsdp=True \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=null \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=null \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=null \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
  +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
  "${moe_flags[@]}" \
  +actor_rollout_ref.actor.megatron.override_ddp_config.data_parallel_sharding_strategy="$FSDP_SHARDING_STRATEGY" \
  +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_grad_reduce=True \
  +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_param_gather=True \
  +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True \
  +actor_rollout_ref.actor.megatron.override_ddp_config.gradient_reduce_div_fusion=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.expert_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.enable_sleep_mode=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.max_model_len="$max_model_len" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_TOKENS_PER_GPU" \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8,16,32,64,128]' \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=FLASH_ATTN \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_TOKENS_PER_GPU" \
  "${checkpoint_engine_flags[@]}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$WEIGHT_UPDATE_BUCKET_MB" \
  async_training.trigger_parameter_sync_step=1 \
  async_training.require_batches=1 \
  async_training.partial_rollout=True \
  async_training.staleness_threshold=0.5 \
  async_training.use_trainer_do_validate=False \
  async_training.use_dynamic_resource_scheduling=False \
  trainer.nnodes=2 \
  trainer.n_gpus_per_node=8 \
  rollout.nnodes=2 \
  rollout.n_gpus_per_node=8 \
  rollout.n="$ROLLOUT_N" \
  rollout.total_rollout_steps="$TOTAL_ROLLOUT_STEPS" \
  trainer.logger='["console"]' \
  trainer.project_name=verl-end2end-h20 \
  trainer.experiment_name="${MODEL_LABEL}-${CHECKPOINT_BACKEND}-tp${ROLLOUT_TP}-pb${PROMPT_BATCH}" \
  trainer.default_local_dir="$RUN_ROOT/checkpoints" \
  trainer.resume_mode=disable \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  "$@"
