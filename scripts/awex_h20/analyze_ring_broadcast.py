#!/usr/bin/env python3
"""Analyze matched H20 WeightRail ring-broadcast runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from analyze_topology_matrix import (
    ANSI_RE,
    DERIVED_METRICS,
    RAW_METRICS,
    SUMMARY_METRICS,
    enrich,
    format_value,
    parse_log,
    read_exit_code,
    stats,
    write_csv,
)


RUN_RE = re.compile(r"^p1t2c2e8-rtp4-ring-(?P<label>[A-Za-z0-9._-]+)$")
PROFILE_FIELDS = (
    "role",
    "step_id",
    "phase",
    "rank",
    "ring_broadcast",
    "ring_order_strategy",
    "ring_relay_count",
    "active_peer_count",
    "lsa_peer_count",
    "gin_peer_count",
    "payload_bytes",
    "transport_total_time_ms",
    "kernel_transfer_time_ms",
    "backend_execute_time_ms",
    "effective_gbps",
    "backend_effective_gbps",
    "plan_cache_hit",
    "gin_type",
)


def read_protocol(run_dir: Path) -> dict[str, str | int]:
    with (run_dir / "protocol.tsv").open(newline="", encoding="utf-8") as handle:
        protocol = next(csv.DictReader(handle, delimiter="\t"))
    return {
        "run": run_dir.name,
        "label": RUN_RE.fullmatch(run_dir.name).group("label"),
        "ring_mode": protocol["ring_mode"],
        "ring_broadcast": int(protocol["ring_broadcast"]),
        "ring_swizzle": int(protocol["ring_swizzle"]),
        "target_train_steps": int(protocol["target_train_steps"]),
        "profile_warmup_updates": int(protocol["profile_warmup_updates"]),
        "publication_first_step": int(protocol["profile_warmup_updates"]),
        "publication_last_step": int(protocol["target_train_steps"]),
        "workflow_first_step": int(protocol["profile_warmup_updates"]),
        "workflow_last_step": int(protocol["target_train_steps"]) - 1,
    }


def parse_profiles(path: Path, run_meta: dict[str, str | int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            if "AWEX_PROFILE {" not in line:
                continue
            try:
                payload, _ = json.JSONDecoder().raw_decode(
                    line.split("AWEX_PROFILE ", 1)[1]
                )
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "weight_transfer":
                continue
            records.append(
                {
                    "run": run_meta["run"],
                    "ring_mode": run_meta["ring_mode"],
                    **{key: payload.get(key, "") for key in PROFILE_FIELDS},
                }
            )
    return records


def validate(
    run_meta: dict[str, str | int],
    steps: dict[int, dict[str, float]],
    progress_seconds: int | None,
    progress_total: int | None,
    transport_profile: dict[str, int],
    profiles: list[dict[str, object]],
    exit_code: int | None,
) -> dict[str, object]:
    publication_expected = set(
        range(
            int(run_meta["publication_first_step"]),
            int(run_meta["publication_last_step"]) + 1,
        )
    )
    workflow_expected = set(
        range(
            int(run_meta["workflow_first_step"]),
            int(run_meta["workflow_last_step"]) + 1,
        )
    )
    publication_actual = {
        step for step, values in steps.items() if "param_sync_s" in values
    }
    workflow_actual = {
        step
        for step, values in steps.items()
        if "version_time_s" in values and "trainer_iteration_s" in values
    }
    measure_profiles = [record for record in profiles if record["phase"] == "measure"]
    ring_records = [record for record in measure_profiles if record["ring_broadcast"] is True]
    relay_records = [
        record
        for record in ring_records
        if isinstance(record["ring_relay_count"], int)
        and int(record["ring_relay_count"]) > 0
    ]
    strategies = sorted(
        {
            str(record["ring_order_strategy"])
            for record in measure_profiles
            if record["ring_order_strategy"] != ""
        }
    )
    expected_strategy = {
        "off": "disabled",
        "naive": "fixed",
        "swizzle": "root_swizzle",
    }[str(run_meta["ring_mode"])]
    ring_expected = bool(run_meta["ring_broadcast"])
    evidence_ok = (
        bool(ring_records) and bool(relay_records)
        if ring_expected
        else not ring_records and not relay_records
    )
    checks = [
        exit_code == 0,
        progress_seconds is not None,
        progress_total == run_meta["target_train_steps"],
        publication_expected <= publication_actual,
        workflow_expected <= workflow_actual,
        all(steps.get(step, {}).get("generated_samples", 0) > 0 for step in workflow_expected),
        transport_profile["gin_full"] > 0,
        transport_profile["gin_non_full"] == 0,
        bool(measure_profiles),
        evidence_ok,
        expected_strategy in strategies,
    ]
    return {
        **run_meta,
        "status": "pass" if all(checks) else "incomplete_or_failed",
        "exit_code": "" if exit_code is None else exit_code,
        "progress_seconds": "" if progress_seconds is None else progress_seconds,
        "publication_sample_count": len(publication_actual & publication_expected),
        "workflow_sample_count": len(workflow_actual & workflow_expected),
        "missing_publication_steps": ",".join(
            map(str, sorted(publication_expected - publication_actual))
        ),
        "missing_workflow_steps": ",".join(
            map(str, sorted(workflow_expected - workflow_actual))
        ),
        "gin_full": transport_profile["gin_full"],
        "gin_non_full": transport_profile["gin_non_full"],
        "plan_cache_hit": transport_profile["plan_cache_hit"],
        "measure_profile_count": len(measure_profiles),
        "ring_profile_count": len(ring_records),
        "relay_profile_count": len(relay_records),
        "ring_order_strategies": ",".join(strategies),
    }


def write_report(
    path: Path,
    validations: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> None:
    lookup = {(row["run"], row["metric"]): row for row in summaries}
    passed = [row for row in validations if row["status"] == "pass"]
    lines = [
        "# H20 WeightRail ring broadcast",
        "",
        f"Validated runs: **{len(passed)}/{len(validations)}**. All measurement runs "
        "use 10 trainer steps, three warm-up updates, publication steps 3-10, and "
        "complete workflow cycles 3-9.",
        "",
        "## Correctness",
        "",
        "| Run | Mode | Status | Loop | Publication n | Workflow n | Ring profiles | Relay profiles | Strategy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in validations:
        elapsed = row["progress_seconds"]
        elapsed_text = "-" if elapsed == "" else f"{int(elapsed) // 60}:{int(elapsed) % 60:02d}"
        lines.append(
            f"| {row['run']} | {row['ring_mode']} | {row['status']} | {elapsed_text} | "
            f"{row['publication_sample_count']} | {row['workflow_sample_count']} | "
            f"{row['ring_profile_count']} | {row['relay_profile_count']} | "
            f"{row['ring_order_strategies']} |"
        )

    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Run | Mode | Publication p50 (s) | p95 (s) | Iteration p50 (s) | Aggregate throughput (sample/s) | Blocked accelerator-s p50 | Speedup vs off |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    off_rows = [row for row in passed if row["ring_mode"] == "off"]
    off_publication = None
    if off_rows:
        summary = lookup.get((off_rows[-1]["run"], "param_sync_s"))
        off_publication = float(summary["p50"]) if summary else None
    for row in validations:
        publication = lookup.get((row["run"], "param_sync_s"))
        iteration = lookup.get((row["run"], "trainer_iteration_s"))
        throughput = lookup.get((row["run"], "rollout_throughput_samples_s"))
        blocked = lookup.get((row["run"], "total_blocked_accelerator_s"))
        if not publication:
            continue
        speedup = (
            off_publication / float(publication["p50"])
            if off_publication is not None
            else None
        )
        lines.append(
            f"| {row['run']} | {row['ring_mode']} | {format_value(publication['p50'])} | "
            f"{format_value(publication['p95'])} | "
            f"{format_value(iteration['p50']) if iteration else '-'} | "
            f"{format_value(throughput['aggregate_rate']) if throughput else '-'} | "
            f"{format_value(blocked['p50']) if blocked else '-'} | "
            f"{format_value(speedup) if speedup is not None else '-'}x |"
        )

    swizzle_rows = [row for row in passed if row["ring_mode"] == "swizzle"]
    target_met = False
    if off_publication is not None and swizzle_rows:
        swizzle_summary = lookup.get((swizzle_rows[-1]["run"], "param_sync_s"))
        competing = [
            lookup.get((row["run"], "param_sync_s"))
            for row in passed
            if row["ring_mode"] != "swizzle"
        ]
        if swizzle_summary:
            swizzle_p50 = float(swizzle_summary["p50"])
            target_met = (
                off_publication / swizzle_p50 >= 2.0
                and all(
                    other is None or swizzle_p50 < float(other["p50"])
                    for other in competing
                )
            )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "Swizzle must be the fastest mode and deliver at least 2x publication "
            f"speedup over the matched no-ring baseline. **Target met: {'yes' if target_met else 'no'}**.",
            "",
            "Full distributions are in `ring-broadcast-summary.csv`; per-step workflow "
            "metrics and structured Device v2 profiles are archived alongside it.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.matrix_root
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    run_dirs = sorted(
        path
        for path in args.matrix_root.iterdir()
        if path.is_dir()
        and RUN_RE.fullmatch(path.name)
        and (path / "run.log").exists()
        and (path / "protocol.tsv").exists()
    )
    for run_dir in run_dirs:
        run_meta = read_protocol(run_dir)
        steps, progress_seconds, progress_total, transport_profile = parse_log(
            run_dir / "run.log"
        )
        if not steps:
            continue
        profiles = parse_profiles(run_dir / "run.log", run_meta)
        profile_rows.extend(profiles)
        validations.append(
            validate(
                run_meta,
                steps,
                progress_seconds,
                progress_total,
                transport_profile,
                profiles,
                read_exit_code(run_dir),
            )
        )
        rows_for_run: list[dict[str, object]] = []
        for step in sorted(steps):
            row: dict[str, object] = {**run_meta, "step": step, **steps[step]}
            enrich(row)
            rows_for_run.append(row)
            step_rows.append(row)
        for metric, window, label in SUMMARY_METRICS:
            first = int(run_meta[f"{window}_first_step"])
            last = int(run_meta[f"{window}_last_step"])
            values = [
                float(row[metric])
                for row in rows_for_run
                if first <= int(row["step"]) <= last and metric in row
            ]
            if not values:
                continue
            summary: dict[str, object] = {
                **run_meta,
                "metric": metric,
                "label": label,
                "window": window,
                "first_step": first,
                "last_step": last,
                **stats(values),
                "aggregate_rate": "",
            }
            if metric == "rollout_throughput_samples_s":
                complete = [
                    row
                    for row in rows_for_run
                    if first <= int(row["step"]) <= last
                    and "generated_samples" in row
                    and "version_time_s" in row
                ]
                summary["aggregate_rate"] = sum(
                    float(row["generated_samples"]) for row in complete
                ) / sum(float(row["version_time_s"]) for row in complete)
            summaries.append(summary)

    meta_fields = [
        "run",
        "label",
        "ring_mode",
        "ring_broadcast",
        "ring_swizzle",
        "target_train_steps",
        "profile_warmup_updates",
        "publication_first_step",
        "publication_last_step",
        "workflow_first_step",
        "workflow_last_step",
    ]
    write_csv(
        output_dir / "ring-broadcast-steps.csv",
        step_rows,
        meta_fields + ["step"] + list(RAW_METRICS.values()) + list(DERIVED_METRICS),
    )
    write_csv(
        output_dir / "ring-broadcast-summary.csv",
        summaries,
        meta_fields
        + [
            "metric",
            "label",
            "window",
            "first_step",
            "last_step",
            "n",
            "mean",
            "p50",
            "p95",
            "min",
            "max",
            "population_stddev",
            "aggregate_rate",
        ],
    )
    write_csv(
        output_dir / "ring-broadcast-validation.csv",
        validations,
        meta_fields
        + [
            "status",
            "exit_code",
            "progress_seconds",
            "publication_sample_count",
            "workflow_sample_count",
            "missing_publication_steps",
            "missing_workflow_steps",
            "gin_full",
            "gin_non_full",
            "plan_cache_hit",
            "measure_profile_count",
            "ring_profile_count",
            "relay_profile_count",
            "ring_order_strategies",
        ],
    )
    write_csv(
        output_dir / "ring-broadcast-profiles.csv",
        profile_rows,
        ["run", "ring_mode", *PROFILE_FIELDS],
    )
    write_report(output_dir / "ring-broadcast-report.md", validations, summaries)
    passed = sum(row["status"] == "pass" for row in validations)
    print(f"Analyzed {len(run_dirs)} run(s); {passed} passed ring validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
