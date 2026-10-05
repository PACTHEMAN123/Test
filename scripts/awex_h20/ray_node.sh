#!/usr/bin/env bash
set -euo pipefail

role=${1:?usage: $0 head|worker NODE_IP HCA_LIST}
node_ip=${2:?usage: $0 head|worker NODE_IP HCA_LIST}
hca_list=${3:?usage: $0 head|worker NODE_IP HCA_LIST}

ROOT=${ROOT:-/mnt/fuse/verl-e2e}
VERL_SOURCE=${VERL_SOURCE:-$ROOT/src/verl}
AWEX_SOURCE=${AWEX_SOURCE:-}
VENV=${VENV:-$ROOT/envs/verl-py312-torch213-cu132-vllm027-pilot}
HEAD_ADDRESS=${HEAD_ADDRESS:-11.18.56.89:6379}
OBJECT_STORE_MEMORY=${OBJECT_STORE_MEMORY:-1610612736}

if [[ -f /opt/rh/gcc-toolset-12/enable ]]; then
  source /opt/rh/gcc-toolset-12/enable
fi
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/nvshmem/lib:$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/scripts/vllm_preload:$VERL_SOURCE:${PYTHONPATH:-}"
if [[ -n "$AWEX_SOURCE" ]]; then
  export PYTHONPATH="$AWEX_SOURCE:$PYTHONPATH"
  if [[ -f "$AWEX_SOURCE/awex_nccl_device_ext_v2.so" ]]; then
    export AWEX_NCCL_DEVICE_V2_EXTENSION=${AWEX_NCCL_DEVICE_V2_EXTENSION:-awex_nccl_device_ext_v2}
  fi
fi
if [[ -n "${NCCL_RUNTIME_LIB:-}" ]]; then
  export LD_PRELOAD="$NCCL_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
  export LD_LIBRARY_PATH="$(dirname "$NCCL_RUNTIME_LIB"):$LD_LIBRARY_PATH"
fi

export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_IB_HCA="=$hca_list"

export NVSHMEM_HCA_LIST="$hca_list"
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=1
export NVSHMEM_USE_GDRCOPY=0

unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
"$VENV/bin/ray" stop --force >/tmp/verl-ray-stop.log 2>&1 || true

common=(
  --node-ip-address="$node_ip"
  --num-cpus=64
  --num-gpus=8
  --object-store-memory="$OBJECT_STORE_MEMORY"
  --disable-usage-stats
)

case "$role" in
  head)
    "$VENV/bin/ray" start --head --port=6379 --include-dashboard=false "${common[@]}"
    ;;
  worker)
    "$VENV/bin/ray" start --address="$HEAD_ADDRESS" "${common[@]}"
    ;;
  *)
    echo "role must be head or worker, got: $role" >&2
    exit 2
    ;;
esac
