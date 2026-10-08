#!/usr/bin/env python3
"""Archive all profiles and compare real FP8 RL workflow and conversion timings."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from analyze_topology_matrix import (
    ANSI_RE,
    DERIVED_METRICS,
    RAW_METRICS,
    enrich,
    parse_log,
    read_exit_code,
    stats,
)


def read_profiles(path):
    records = []
    for line in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", line)
        for marker in ("VERL_FP8_PROFILE ", "AWEX_PROFILE "):
            if marker in line:
                try:
                    record, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
                    ip = re.search(r"ip=([0-9.]+)\)", line)
                    record["ray_node_ip"] = ip.group(1) if ip else "11.18.56.89"
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
    # Trainer aggregation is one cycle late; rollouter reset metrics are
    # immediate. Joining both at the raw log index mixes different cycles.
    rollout_fields = (
        "rollout_active_s",
        "version_time_s",
        "rollout_idle_ratio_reported",
        "generated_samples",
        "rollout_resource_utilization",
    )
    workflow = []
    for i in range(3, 10):
        if i not in steps or i + 1 not in steps:
            continue
        row = {
            "step": i,
            "trainer_logged_step": i + 1,
            "rollout_logged_step": i,
            **steps[i + 1],
        }
        for field in rollout_fields:
            row.pop(field, None)
            if field in steps[i]:
                row[field] = steps[i][field]
        if (
            "train_resource_utilization" in row
            and "rollout_resource_utilization" in row
        ):
            row["combined_resource_utilization"] = (
                row["train_resource_utilization"] + row["rollout_resource_utilization"]
            ) / 2
        workflow.append(row)
    for row in workflow:
        enrich(row)
    steady = [
        p
        for p in profiles
        if isinstance(p.get("step_id"), int) and 3 <= p["step_id"] <= 10
    ]
    checkpoint = [p for p in steady if p.get("event") == "checkpoint_workflow"]
    publications = [p["total_wall_ms"] / 1000 for p in checkpoint]
    publication_source = "checkpoint_workflow/total_wall_ms"
    trainer_publications = {
        int(version): float(seconds)
        for seconds, version in re.findall(
            r"_fit_update_weights, timing_s/param_sync: ([0-9.]+) seconds self.current_param_version: (\d+)",
            ANSI_RE.sub("", (run_dir / "run.log").read_text(errors="replace")),
        )
        if 3 <= int(version) <= 10
    }
    if not checkpoint:
        publications = [trainer_publications[k] for k in sorted(trainer_publications)]
        publication_source = "trainer_parameter_sync/rounded_seconds"
    reloads = [p for p in steady if p.get("event") == "vllm_fp8_reload"]
    awex = [p for p in steady if p.get("event") == "weight_transfer"]
    payload = analyze_pair_payloads(run_dir, awex, publications) if awex else None
    breakdown = {}
    for event, field in (
        ("actor_export", "export_iterator_wall_ms"),
        ("nccl_send_pipeline", "pipeline_wall_ms"),
        ("nccl_receive_pipeline", "pipeline_wall_ms"),
    ):
        by_step = defaultdict(list)
        for p in steady:
            if p.get("event") == event:
                by_step[p["step_id"]].append(p[field])
        breakdown[f"{event}/{field}/rank_max_ms"] = summarize(
            [max(v) for v in by_step.values()]
        )
    for p in reloads:
        wall = p["wall_ms"]
        gpu = p["gpu_stream_ms"]
        p["reload_residual_wall_ms"] = p["total_wall_ms"] - sum(
            wall.get(name, 0)
            for name in (
                "prepare_layout",
                "load_including_quantization",
                "finalize_layout",
            )
        )
        p["load_stream_excluding_conversion_ms"] = gpu.get(
            "load_including_quantization", 0
        ) - gpu.get("bf16_to_fp8", 0)
    for group, key, fields in (
        (
            checkpoint,
            "stage_wall_ms",
            sorted({f for p in checkpoint for f in p["stage_wall_ms"]}),
        ),
        (
            reloads,
            "wall_ms",
            (
                "bf16_to_fp8",
                "prepare_layout",
                "load_including_quantization",
                "finalize_layout",
            ),
        ),
        (
            reloads,
            "gpu_stream_ms",
            (
                "bf16_to_fp8",
                "prepare_layout",
                "load_including_quantization",
                "finalize_layout",
            ),
        ),
    ):
        for field in fields:
            by_step = defaultdict(list)
            for p in group:
                if field in p[key]:
                    by_step[p["step_id"]].append(p[key][field])
            # All-rank maximum for each publication, then distributions across publications.
            breakdown[f"{key}/{field}/rank_max_ms"] = summarize(
                [max(v) for v in by_step.values()]
            )
    for field in (
        "total_wall_ms",
        "profile_fence_ms",
        "reload_residual_wall_ms",
        "load_stream_excluding_conversion_ms",
    ):
        by_step = defaultdict(list)
        for p in reloads:
            by_step[p["step_id"]].append(p[field])
        breakdown[f"vllm_reload/{field}/rank_max_ms"] = summarize(
            [max(v) for v in by_step.values()]
        )
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
            breakdown[f"awex/{role}/{field}/rank_max_ms"] = summarize(
                [max(v) for v in by_step.values()]
            )
    fields = [
        "step",
        "trainer_logged_step",
        "rollout_logged_step",
        *RAW_METRICS.values(),
        *DERIVED_METRICS,
    ]
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
        "publication_source": publication_source,
        "trainer_publications_seconds": trainer_publications,
        "workflow": {
            field: summarize([r[field] for r in workflow if field in r])
            for field in fields[3:]
        },
        "breakdown": breakdown,
        "transport": transport,
        "directed_payload": payload,
        "checkpoint_records_per_step": dict(
            sorted(Counter(p["step_id"] for p in checkpoint).items())
        ),
        "reload_records_per_step": dict(
            sorted(Counter(p["step_id"] for p in reloads).items())
        ),
        "awex_records_per_step": dict(
            sorted(Counter(p["step_id"] for p in awex).items())
        ),
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
        "export_payloads": [
            {
                k: p[k]
                for k in (
                    "step_id",
                    "rank",
                    "hostname",
                    "payload_bytes",
                    "tensor_count",
                    "dtype_bytes",
                )
            }
            for p in steady
            if p.get("event") == "actor_export"
        ],
        "complete": read_exit_code(run_dir) == 0
        and progress_total == 10
        and len(publications) == 8
        and len(workflow) == 7
        and all(row.get("generated_samples") == 128 for row in workflow)
        and (
            not reloads
            or all(sum(p["step_id"] == i for p in reloads) == 16 for i in range(3, 11))
        )
        and (
            not awex
            or all(sum(p["step_id"] == i for p in awex) == 32 for i in range(3, 11))
        ),
    }


def analyze_pair_payloads(run_dir, profiles, publications):
    """Audit this TP2 x 8 swizzle setup using actual per-rank peer payloads."""
    records = {}
    fields = (
        "role",
        "ray_node_ip",
        "payload_bytes",
        "active_peer_payload_bytes",
        "selected_hca",
        "selected_hca_bandwidth_gbps",
        "lsa_peer_count",
        "gin_peer_count",
    )
    for profile in profiles:
        assert (
            profile["fp8_blockwise"]
            and profile["ring_order_strategy"] == "root_swizzle"
        )
        static = {field: profile[field] for field in fields}
        rank = profile["rank"]
        assert rank not in records or records[rank] == static, rank
        records[rank] = static
    assert set(records) == set(range(32))
    expected = {
        int(rank): json.loads(counts)
        for rank, counts in re.findall(
            r"Lowered nccl_device_v2 plan rank=(\d+) sender=True.*?expected_counts=(\[[^\]]*\])",
            (run_dir / "run.log").read_text(errors="replace"),
        )
    }
    assert set(expected) == set(range(16, 32))
    flows = defaultdict(int)
    for root, counts in expected.items():
        peers = [rank for rank, count in enumerate(counts) if count]
        amounts = records[root]["active_peer_payload_bytes"]
        assert len(peers) == len(amounts)
        for first, amount in zip(peers, amounts):
            flows[root, first] += amount
            targets = [engine * 2 + first % 2 for engine in range(8)]
            groups = [targets[::2], targets[1::2]]
            offset = (root // 2) % 4
            order = []
            for index in range(2):
                group = groups[(root + index) % 2]
                order.extend(group[offset:] + group[:offset])
            assert order[0] == first
            for source, destination in zip(order[:-1], order[1:]):
                flows[source, destination] += amount
    combined = defaultdict(lambda: defaultdict(int))
    sent, received = defaultdict(int), defaultdict(int)
    for (source, destination), amount in flows.items():
        combined[source][destination] += amount
        combined[destination][source] += amount
        sent[source] += amount
        received[destination] += amount
    for rank, record in records.items():
        assert sorted(combined[rank].values()) == sorted(
            record["active_peer_payload_bytes"]
        ), rank
        assert (sent[rank] if record["role"] == "writer" else received[rank]) == record[
            "payload_bytes"
        ], rank
        local = sum(
            records[peer]["ray_node_ip"] == record["ray_node_ip"]
            for peer in combined[rank]
        )
        assert local == record["lsa_peer_count"], rank
        assert len(combined[rank]) - local == record["gin_peer_count"], rank
    pairs, ports = [], defaultdict(int)
    latency_s = stats(publications)["p50"]
    for (source, destination), amount in sorted(flows.items()):
        src, dst = records[source], records[destination]
        transport = "LSA" if src["ray_node_ip"] == dst["ray_node_ip"] else "GIN"
        pairs.append(
            {
                "source_rank": source,
                "destination_rank": destination,
                "source_node": src["ray_node_ip"],
                "destination_node": dst["ray_node_ip"],
                "transport": transport,
                "payload_bytes": amount,
                "step_average_GBps": amount / (latency_s * 1e9),
            }
        )
        if transport == "GIN":
            for direction, record in (("TX", src), ("RX", dst)):
                ports[record["ray_node_ip"], record["selected_hca"], direction] += (
                    amount
                )
    with (run_dir / "directed-pair-payloads.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    return {
        "validated_ranks": 32,
        "pairs": pairs,
        "GIN_bytes": sum(p["payload_bytes"] for p in pairs if p["transport"] == "GIN"),
        "LSA_bytes": sum(p["payload_bytes"] for p in pairs if p["transport"] == "LSA"),
        "assigned_port_loads": [
            {
                "node": node,
                "requested_hca": hca,
                "direction": direction,
                "payload_bytes": amount,
                "step_average_GBps": amount / (latency_s * 1e9),
                "percent_of_requested_200Gbps_cap": amount / (latency_s * 25e9) * 100,
            }
            for (node, hca, direction), amount in sorted(ports.items())
        ],
        "actual_wire_utilization_percent": None,
        "note": "Per-pair application bytes include FP8 scale/framing overhead. Requested-port capacity ratios are conditional, not measured physical NIC utilization.",
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
                    for k in (
                        "run",
                        "complete",
                        "publication",
                        "reload_records_per_step",
                        "awex_records_per_step",
                    )
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
