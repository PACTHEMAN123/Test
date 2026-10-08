#!/usr/bin/env python3
"""Inspect/deploy the four user-selected H20 nodes through their existing Ray cluster."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import urllib.request
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

IPS = ("11.18.50.220", "11.18.56.89", "33.0.194.195", "33.240.39.192")
ROOT = Path("/mnt/fuse/verl-e2e")
SOURCE = ROOT / "src/verl-fp8-e2e"
BRANCH = "codex/verl-fp8-e2e"


def command(args, cwd=None, timeout=30):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()


@ray.remote(num_cpus=0)
def operate(action, revision):
    idle = command(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"])
    if action == "deploy":
        if idle:
            raise RuntimeError("Refusing deployment while GPU compute processes are active")
        if not SOURCE.exists():
            command(["git", "clone", "--shared", str(ROOT / "src/verl-direct-gin"), str(SOURCE)], timeout=60)
        if command(["git", "status", "--porcelain"], cwd=SOURCE):
            raise RuntimeError(f"Refusing to overwrite dirty source: {SOURCE}")
        try:
            command(["git", "-c", "http.lowSpeedTime=10", "-c", "http.lowSpeedLimit=1", "fetch",
                     "https://github.com/PACTHEMAN123/Test.git", BRANCH], cwd=SOURCE, timeout=30)
        except (subprocess.SubprocessError, OSError):
            bundle = ROOT / "runs/verl-fp8-e2e-20261008/verl-fp8.bundle"
            bundle.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen("http://11.18.56.89:19012/verl-fp8.bundle", timeout=30) as response:
                bundle.write_bytes(response.read())
            command(["git", "fetch", str(bundle), BRANCH], cwd=SOURCE)
        actual = command(["git", "rev-parse", "FETCH_HEAD"], cwd=SOURCE)
        if actual != revision:
            raise RuntimeError((actual, revision))
        command(["git", "switch", "--detach", actual], cwd=SOURCE)
        command([str(ROOT / "envs/verl-py312-torch213-cu132-vllm027-pilot/bin/python"), "-m", "compileall", "-q",
                 str(SOURCE / "verl/utils/weight_update_profile.py"), str(SOURCE / "verl/workers/rollout/vllm_rollout/utils.py")])
    awex = ROOT / "src/Awex-fp8-blockwise"
    return {
        "ip": ray.util.get_node_ip_address(), "hostname": socket.gethostname(), "cwd": os.getcwd(),
        "user": command(["id", "-un"]), "gpu_compute_processes": idle,
        "gpus": command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"]),
        "gpu_nic_topology": command(["nvidia-smi", "topo", "-m"]),
        "cpu_allowed": sorted(os.sched_getaffinity(0)),
        "model_exists": (ROOT / "models/Qwen3-30B-A3B/config.json").is_file(),
        "data_exists": (ROOT / "datasets/gsm8k/train.parquet").is_file(),
        "oss_mount_exists": Path("/mnt/fuse/oss/xiaopac.xjy").is_dir(),
        "verl_revision": command(["git", "rev-parse", "HEAD"], cwd=SOURCE if SOURCE.exists() else ROOT / "src/verl-direct-gin"),
        "awex_revision": command(["git", "rev-parse", "HEAD"], cwd=awex),
        "extension_sha256": hashlib.sha256((awex / "awex_nccl_device_ext_v2.so").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "deploy"))
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.action == "deploy" and not args.revision:
        parser.error("deploy requires --revision")
    ray.init(address="11.18.56.89:6379", logging_level="ERROR")
    nodes = {n["NodeManagerAddress"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}
    results = ray.get([operate.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip], soft=False)).remote(
        args.action, args.revision) for ip in IPS])
    print(json.dumps(results, indent=2), flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
