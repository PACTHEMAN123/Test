#!/usr/bin/env python3
"""Archive all profiles and compare real FP8 RL workflow and conversion timings."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_topology_matrix import ANSI_RE, DERIVED_METRICS, RAW_METRICS, enrich, parse_log, read_exit_code, stats


def read_profiles(path):
    records = []
    for line in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", line)
        for marker in ("VERL_FP8_PROFILE ", "AWEX_PROFILE "):
            if marker in line:
                try:
                    record, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
                    records.append({"profile_source": marker.strip(), **record})
                except json.JSONDecodeError:
                    pass
    return records


def summarize(values):
    return stats(values) if values else {"n": 0}


def analyze(run_dir):
    steps, progress_seconds, progress_total, transport = parse_log(run_dir / "run.log")
    profiles = read_profiles(run_dir / "run.log")
    (run_dir / "profiles.json").write_text(json.dumps(profiles, indent=2) + "\n")
    # The aggregator logs a cycle one version late, before adding the current
    # cycle in _fit_postprocess_step. Retain both logical and raw log indices.
    workflow = [{"step": i - 1, "logged_step": i, **row} for i, row in sorted(steps.items()) if 4 <= i <= 10]
    for row in workflow:
        enrich(row)
    steady = [p for p in profiles if isinstance(p.get("step_id"), int) and 3 <= p["step_id"] <= 10]
    checkpoint = [p for p in steady if p.get("event") == "checkpoint_workflow"]
    publications = [p["total_wall_ms"] / 1000 for p in checkpoint]
    reloads = [p for p in steady if p.get("event") == "vllm_fp8_reload"]
    awex = [p for p in steady if p.get("event") == "weight_transfer"]
    breakdown = {}
    for p in reloads:
        wall = p["wall_ms"]
        gpu = p["gpu_stream_ms"]
        p["reload_residual_wall_ms"] = p["total_wall_ms"] - sum(
            wall.get(name, 0) for name in ("prepare_layout", "load_including_quantization", "finalize_layout")
        )
        p["load_stream_excluding_conversion_ms"] = gpu.get("load_including_quantization", 0) - gpu.get("bf16_to_fp8", 0)
    for group, key, fields in (
        (checkpoint, "stage_wall_ms", sorted({f for p in checkpoint for f in p["stage_wall_ms"]})),
        (reloads, "wall_ms", ("bf16_to_fp8", "prepare_layout", "load_including_quantization", "finalize_layout")),
        (reloads, "gpu_stream_ms", ("bf16_to_fp8", "prepare_layout", "load_including_quantization", "finalize_layout")),
    ):
        for field in fields:
            by_step = defaultdict(list)
            for p in group:
                if field in p[key]:
                    by_step[p["step_id"]].append(p[key][field])
            # All-rank maximum for each publication, then distributions across publications.
            breakdown[f"{key}/{field}/rank_max_ms"] = summarize([max(v) for v in by_step.values()])
    for field in (
        "total_wall_ms",
        "profile_fence_ms",
        "reload_residual_wall_ms",
        "load_stream_excluding_conversion_ms",
    ):
        by_step = defaultdict(list)
        for p in reloads:
            by_step[p["step_id"]].append(p[field])
        breakdown[f"vllm_reload/{field}/rank_max_ms"] = summarize([max(v) for v in by_step.values()])
    for role in ("writer", "reader"):
        for field in (
            "convert_time_ms",
            "kernel_transfer_time_ms",
            "backend_execute_time_ms",
            "total_transfer_time_ms",
        ):
            by_step = defaultdict(list)
            for p in awex:
                if p.get("role") == role and field in p:
                    by_step[p["step_id"]].append(p[field])
            breakdown[f"awex/{role}/{field}/rank_max_ms"] = summarize([max(v) for v in by_step.values()])
    fields = ["step", "logged_step", *RAW_METRICS.values(), *DERIVED_METRICS]
    with (run_dir / "workflow-steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(workflow)
    return {
        "run": run_dir.name,
        "exit_code": read_exit_code(run_dir),
        "progress_total": progress_total,
        "progress_seconds": progress_seconds,
        "publication": summarize(publications),
        "workflow": {field: summarize([r[field] for r in workflow if field in r]) for field in fields[2:]},
        "breakdown": breakdown,
        "transport": transport,
        "checkpoint_records_per_step": dict(sorted(Counter(p["step_id"] for p in checkpoint).items())),
        "reload_records_per_step": dict(sorted(Counter(p["step_id"] for p in reloads).items())),
        "awex_records_per_step": dict(sorted(Counter(p["step_id"] for p in awex).items())),
        "reload_payloads": [
            {
                k: p[k]
                for k in (
                    "hostname",
                    "pid",
                    "rank",
                    "step_id",
                    "input_quantized_bytes",
                    "output_fp8_bytes",
                    "output_scale_bytes",
                    "quantized_tensors",
                )
            }
            for p in reloads
        ],
        "complete": read_exit_code(run_dir) == 0
        and progress_total == 10
        and len(publications) == 8
        and len(workflow) == 7,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    results = [
        analyze(p)
        for p in sorted(args.root.iterdir())
        if p.is_dir() and (p / "run.log").exists() and p.name != "config"
    ]
    (args.root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in ("run", "complete", "publication", "reload_records_per_step", "awex_records_per_step")
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
