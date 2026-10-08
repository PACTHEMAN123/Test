#!/usr/bin/env python3
"""Run matched real FP8 GRPO cases serially on the four established H20 nodes."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ROOT = Path("/mnt/fuse/verl-e2e")
SOURCE = ROOT / "src/verl-fp8-e2e"
AWEX = ROOT / "src/Awex-fp8-verl-e2e"
IPS = ("11.18.50.220", "11.18.56.89", "33.0.194.195", "33.240.39.192")
INFERENCE_IPS = ("11.18.56.89", "33.240.39.192")
EXTENSION_HASH = "8406492c439aa781ba1a92470212fa4b1f0886afbbac3100629422aaa71c5a87"
CASES = (("p1t4c2e8-rtp2", 4, 1, 2, 8, 1, 2),)


@ray.remote(num_cpus=0)
def inspect_node():
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return {
        "ip": ray.util.get_node_ip_address(),
        "gpu_processes": processes,
        "verl_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
        ).strip(),
        "awex_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=AWEX, text=True
        ).strip(),
        "extension_sha256": hashlib.sha256(
            (AWEX / "awex_nccl_device_ext_v2.so").read_bytes()
        ).hexdigest(),
    }


@ray.remote(num_cpus=0, num_gpus=8)
class ReserveInference:
    def ready(self):
        return ray.util.get_node_ip_address()


def preflight(nodes, revision):
    inventory = ray.get(
        [
            inspect_node.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    nodes[ip], soft=False
                )
            ).remote()
            for ip in IPS
        ]
    )
    if ray.available_resources().get("GPU", 0) != 32:
        raise RuntimeError("Expected all 32 Ray GPUs to be free")
    for node in inventory:
        if node["gpu_processes"] or node["verl_revision"] != revision:
            raise RuntimeError(f"Unexpected node runtime state: {node}")
        if (
            node["awex_revision"] != "b689ea280e17234fb9b26701fdde4a1afa1e3637"
            or node["extension_sha256"] != EXTENSION_HASH
        ):
            raise RuntimeError(f"Awex source or qualified kernel changed: {node}")
    return inventory


def run_case(matrix_root, spec, backend, nodes, revision):
    label, tp, pp, cp, ep, fsdp, rollout_tp = spec
    run = matrix_root / f"{label}-{backend}"
    if (run / "exit_code").exists() and (run / "exit_code").read_text().strip() == "0":
        print(f"SKIP complete {run.name}", flush=True)
        return
    if (run / "run.log").exists():
        raise RuntimeError(
            f"Refusing to overwrite existing failed/interrupted run: {run}"
        )
    run.mkdir(parents=True, exist_ok=True)
    inventory = preflight(nodes, revision)
    metadata = dict(
        label=label,
        backend=backend,
        actor_tp=tp,
        actor_pp=pp,
        actor_cp=cp,
        actor_ep=ep,
        actor_etp=1,
        megatron_fsdp=fsdp,
        rollout_tp=rollout_tp,
        fanout=16 // rollout_tp,
        target_steps=10,
        warmup_updates=3,
        source_revision=revision,
        preflight=inventory,
        training_ips=[ip for ip in IPS if ip not in INFERENCE_IPS],
        inference_ips=list(INFERENCE_IPS),
        native_fine_profile=False,
        quantization="fp8_e4m3_block128x128",
        ring_mode="swizzle",
    )
    (run / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    reservations, process = [], None
    try:
        reservations = [
            ReserveInference.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    nodes[ip], soft=False
                )
            ).remote()
            for ip in INFERENCE_IPS
        ]
        print(
            f"BEGIN {run.name}: reserved inference {ray.get([a.ready.remote() for a in reservations])}",
            flush=True,
        )
        env = dict(
            os.environ,
            CHECKPOINT_BACKEND=backend,
            AWEX_SOURCE=str(AWEX),
            VERL_SOURCE=str(SOURCE),
            RUN_ROOT=str(run),
            ACTOR_TP=str(tp),
            ACTOR_PP=str(pp),
            ACTOR_CP=str(cp),
            ACTOR_EP=str(ep),
            ACTOR_ETP="1",
            USE_MEGATRON_FSDP=str(fsdp),
            ROLLOUT_TP=str(rollout_tp),
            TARGET_TRAIN_STEPS="10",
            VERL_FP8_UPDATE_PROFILE="0" if backend == "nccl" else "1",
        )
        process = subprocess.Popen(
            ["bash", str(SOURCE / "scripts/awex_h20/fp8_e2e_run.sh")],
            env=env,
            start_new_session=True,
        )
        begin = time.monotonic()
        while ray.available_resources().get("GPU", 0) > 0:
            if process.poll() is not None:
                raise RuntimeError(
                    f"RL exited before trainer placement: {process.returncode}"
                )
            if time.monotonic() - begin > 300:
                raise TimeoutError(
                    "Trainer placement did not commit within five minutes"
                )
            time.sleep(2)
        for actor in reservations:
            ray.kill(actor, no_restart=True)
        reservations.clear()
        print(f"PLACED {run.name}: trainer owns the other 16 GPUs", flush=True)
        code = process.wait(timeout=1800)
        print(f"END {run.name}: exit={code}", flush=True)
        if code:
            raise RuntimeError(f"Experiment failed: {run.name}, exit={code}")
    finally:
        for actor in reservations:
            ray.kill(actor, no_restart=True)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        # Ray may release GPU reservations before worker processes finish
        # destroying CUDA contexts. Wait for both before starting the next run.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if ray.available_resources().get("GPU", 0) == 32:
                inventory = ray.get(
                    [
                        inspect_node.options(
                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                nodes[ip], soft=False
                            )
                        ).remote()
                        for ip in IPS
                    ]
                )
                if all(not node["gpu_processes"] for node in inventory):
                    break
            time.sleep(5)
        else:
            raise TimeoutError(
                "Run workers did not release all GPUs within two minutes"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "runs/verl-fp8-topology-20261008"
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--case", choices=[spec[0] for spec in CASES])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    ray.init(address="11.18.56.89:6379", logging_level="ERROR")
    nodes = {n["NodeManagerAddress"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}
    try:
        for index, spec in enumerate(CASES):
            if args.case and args.case != spec[0]:
                continue
            backends = (
                ("nccl", "awex_weightrail")
                if index % 2 == 0
                else ("awex_weightrail", "nccl")
            )
            for backend in backends:
                run_case(args.root, spec, backend, nodes, args.revision)
        preflight(nodes, args.revision)
        print(
            "MATRIX COMPLETE: all requested cases successful; all GPUs released",
            flush=True,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
