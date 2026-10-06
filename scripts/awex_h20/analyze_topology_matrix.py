#!/usr/bin/env python3
"""Analyze the steady-state workflow impact of the H20 topology matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUN_RE = re.compile(
    r"^(?P<topology>p(?P<actor_pp>\d+)t(?P<actor_tp>\d+)c(?P<actor_cp>\d+)e(?P<actor_ep>\d+))"
    r"-rtp(?P<rollout_tp>\d+)-(?P<backend>nccl|awex_nccl|awex_weightrail)$"
)
PROGRESS_RE = re.compile(r"Training Progress:\s*100%.*?40/40 \[(\d+):(\d+)<")
STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
METRIC_RE = re.compile(r"^([^:]+):(.+)$")

PUBLICATION_FIRST = 8
PUBLICATION_LAST = 40
WORKFLOW_FIRST = 8
WORKFLOW_LAST = 39
TRAINER_GPUS = 16
ROLLOUT_GPUS = 16
EXPECTED_SAMPLES = 128

RAW_METRICS = {
    "fully_async/rollouter/active_time": "rollout_active_s",
    "fully_async/rollouter/version_time": "version_time_s",
    "fully_async/rollouter/idle_ratio": "rollout_idle_ratio_reported",
    "fully_async/rollouter/step_generated_samples": "generated_samples",
    "timing_s/timing_s/param_sync": "param_sync_s",
    "timing_s/step": "trainer_iteration_s",
    "timing_s/gen": "generation_s",
    "timing_s/update_actor": "actor_update_s",
    "dynamic_resource/train_compute_time_s": "train_compute_s",
    "fully_async/total_wait_time": "trainer_wait_s",
    "perf/mfu/actor": "actor_mfu",
    "fully_async/trainer/idle_ratio": "trainer_idle_ratio",
    "dynamic_resource/train_resource_utilization": "train_resource_utilization",
    "dynamic_resource/rollout_resource_utilization": "rollout_resource_utilization",
    "dynamic_resource/resource_utilization": "combined_resource_utilization",
    "fully_async/count/stale_trajectory_processed": "stale_trajectory_processed_cumulative",
    "fully_async/count/staleness_samples": "staleness_samples",
    "fully_async/count/dropped_stale_samples": "dropped_stale_samples_cumulative",
    "fully_async/monitor/queue/pending_queue_size": "pending_queue_size",
    "fully_async/monitor/queue/mq_queue_size": "message_queue_size",
}

DERIVED_METRICS = (
    "publication_duty_pct",
    "rollout_inactive_s",
    "rollout_throughput_samples_s",
    "trainer_blocked_accelerator_s",
    "rollout_blocked_accelerator_s",
    "total_blocked_accelerator_s",
)

SUMMARY_METRICS = (
    ("param_sync_s", "publication", "Trainer publication stall (s)"),
    ("publication_duty_pct", "workflow", "Publication duty cycle (%)"),
    ("version_time_s", "workflow", "Synchronization cycle (s)"),
    ("trainer_iteration_s", "workflow", "Trainer iteration (s)"),
    ("rollout_throughput_samples_s", "workflow", "Rollout throughput (sample/s)"),
    ("rollout_active_s", "workflow", "Rollout active interval (s)"),
    ("rollout_inactive_s", "workflow", "Rollout inactive interval (s)"),
    ("generation_s", "workflow", "Trainer-observed generation (s)"),
    ("actor_update_s", "workflow", "Actor update (s)"),
    ("train_compute_s", "workflow", "Trainer compute (s)"),
    ("trainer_wait_s", "workflow", "Trainer wait (s)"),
    ("actor_mfu", "workflow", "Actor MFU"),
    ("trainer_idle_ratio", "workflow", "Trainer idle ratio"),
    ("train_resource_utilization", "workflow", "Trainer resource utilization"),
    ("rollout_resource_utilization", "workflow", "Rollout resource utilization"),
    ("combined_resource_utilization", "workflow", "Combined resource utilization"),
    ("staleness_samples", "workflow", "Staleness samples"),
    ("pending_queue_size", "workflow", "Pending queue size"),
    ("message_queue_size", "workflow", "Message queue size"),
    ("trainer_blocked_accelerator_s", "workflow", "Trainer blocked accelerator-s"),
    ("rollout_blocked_accelerator_s", "workflow", "Rollout blocked accelerator-s"),
    ("total_blocked_accelerator_s", "workflow", "Total blocked accelerator-s"),
)


def percentile(values: list[float], quantile: float) -> float:
    """Return a NumPy-compatible linear percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "population_stddev": statistics.pstdev(values),
    }


def parse_number(value: str) -> float | None:
    try:
        result = float(value.strip())
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_log(path: Path) -> tuple[dict[int, dict[str, float]], int | None, dict[str, int]]:
    steps: dict[int, dict[str, float]] = defaultdict(dict)
    progress_seconds = None
    profile = {"gin_full": 0, "gin_non_full": 0, "plan_cache_hit": 0}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            for minutes, seconds in PROGRESS_RE.findall(line):
                progress_seconds = int(minutes) * 60 + int(seconds)
            if "AWEX_PROFILE " in line:
                try:
                    payload, _ = json.JSONDecoder().raw_decode(
                        line.split("AWEX_PROFILE ", 1)[1]
                    )
                except json.JSONDecodeError:
                    payload = {}
                if payload.get("gin_type") == 3:
                    profile["gin_full"] += 1
                elif "gin_type" in payload:
                    profile["gin_non_full"] += 1
                if payload.get("plan_cache_hit") is True:
                    profile["plan_cache_hit"] += 1

            match = STEP_RE.search(line)
            if not match or "FullyAsyncTrainer" not in line:
                continue
            step = int(match.group(1))
            for item in match.group(2).split(" - "):
                metric_match = METRIC_RE.match(item)
                if not metric_match:
                    continue
                key, raw_value = metric_match.groups()
                if key not in RAW_METRICS:
                    continue
                value = parse_number(raw_value)
                if value is not None:
                    steps[step][RAW_METRICS[key]] = value

    return steps, progress_seconds, profile


def metadata(run_dir: Path) -> dict[str, str | int]:
    match = RUN_RE.match(run_dir.name)
    if not match:
        raise ValueError(f"Unrecognized run directory: {run_dir.name}")
    result: dict[str, str | int] = match.groupdict()
    for key in ("actor_pp", "actor_tp", "actor_cp", "actor_ep", "rollout_tp"):
        result[key] = int(result[key])
    result["megatron_fsdp"] = int(result["actor_pp"] == 1)
    result["rollout_engines"] = ROLLOUT_GPUS // int(result["rollout_tp"])
    result["run"] = run_dir.name
    return result


def enrich(row: dict[str, object]) -> None:
    param_sync = row.get("param_sync_s")
    version_time = row.get("version_time_s")
    active_time = row.get("rollout_active_s")
    generated = row.get("generated_samples")
    if isinstance(param_sync, float):
        row["trainer_blocked_accelerator_s"] = TRAINER_GPUS * param_sync
    if isinstance(version_time, float) and isinstance(active_time, float):
        inactive = max(0.0, version_time - active_time)
        row["rollout_inactive_s"] = inactive
        row["rollout_blocked_accelerator_s"] = ROLLOUT_GPUS * inactive
    if isinstance(param_sync, float) and isinstance(version_time, float) and version_time:
        row["publication_duty_pct"] = 100.0 * param_sync / version_time
    if isinstance(generated, float) and isinstance(version_time, float) and version_time:
        row["rollout_throughput_samples_s"] = generated / version_time
    trainer_blocked = row.get("trainer_blocked_accelerator_s")
    rollout_blocked = row.get("rollout_blocked_accelerator_s")
    if isinstance(trainer_blocked, float) and isinstance(rollout_blocked, float):
        row["total_blocked_accelerator_s"] = trainer_blocked + rollout_blocked


def read_exit_code(run_dir: Path) -> int | None:
    path = run_dir / "exit_code"
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return -1


def run_validation(
    run_meta: dict[str, str | int],
    steps: dict[int, dict[str, float]],
    exit_code: int | None,
    progress_seconds: int | None,
    profile: dict[str, int],
) -> dict[str, object]:
    publication_steps = {
        step for step, values in steps.items() if "param_sync_s" in values
    }
    workflow_steps = {
        step
        for step, values in steps.items()
        if "version_time_s" in values and "trainer_iteration_s" in values
    }
    expected_publication = set(range(PUBLICATION_FIRST, PUBLICATION_LAST + 1))
    expected_workflow = set(range(WORKFLOW_FIRST, WORKFLOW_LAST + 1))
    bad_samples = [
        step
        for step in expected_workflow
        if steps.get(step, {}).get("generated_samples") != EXPECTED_SAMPLES
    ]
    missing_publication = sorted(expected_publication - publication_steps)
    missing_workflow = sorted(expected_workflow - workflow_steps)
    backend = str(run_meta["backend"])
    checks = [
        exit_code == 0,
        progress_seconds is not None,
        not missing_publication,
        not missing_workflow,
        not bad_samples,
    ]
    if backend == "awex_weightrail":
        checks.extend([profile["gin_full"] > 0, profile["gin_non_full"] == 0])
    return {
        **run_meta,
        "status": "pass" if all(checks) else "incomplete_or_failed",
        "exit_code": "" if exit_code is None else exit_code,
        "progress_seconds": "" if progress_seconds is None else progress_seconds,
        "parsed_step_count": len(steps),
        "publication_sample_count": len(publication_steps & expected_publication),
        "workflow_sample_count": len(workflow_steps & expected_workflow),
        "missing_publication_steps": ",".join(map(str, missing_publication)),
        "missing_workflow_steps": ",".join(map(str, missing_workflow)),
        "bad_generated_sample_steps": ",".join(map(str, bad_samples)),
        **profile,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(
    path: Path,
    validations: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> None:
    completed = [row for row in validations if row["status"] == "pass"]
    lookup = {
        (row["run"], row["metric"]): row
        for row in summaries
    }
    backend_labels = {
        "awex_weightrail": "WeightRail",
        "awex_nccl": "Awex NCCL",
        "nccl": "veRL NCCL",
    }
    backend_order = ("awex_weightrail", "awex_nccl", "nccl")
    lines = [
        "# H20 topology matrix: three-backend workflow impact",
        "",
        f"Validated runs: **{len(completed)}/18**. Publication statistics use steps "
        f"{PUBLICATION_FIRST}-{PUBLICATION_LAST}; workflow statistics use complete cycles "
        f"{WORKFLOW_FIRST}-{WORKFLOW_LAST}.",
        "",
        "Each topology is matched across veRL native NCCL, Awex NCCL, and WeightRail. "
        "The PP4 topology uses Megatron distributed optimizer because the current MCore "
        "FSDP adapter does not support that pipeline layout; PP1 uses MCore FSDP.",
        "",
        "## Validation",
        "",
        "| Run | Status | Publication n | Workflow n | Loop time | Full GIN profiles |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in validations:
        elapsed = row["progress_seconds"]
        elapsed_text = "-" if elapsed == "" else f"{int(elapsed) // 60}:{int(elapsed) % 60:02d}"
        lines.append(
            f"| {row['run']} | {row['status']} | {row['publication_sample_count']} | "
            f"{row['workflow_sample_count']} | {elapsed_text} | {row['gin_full']} |"
        )

    lines.extend(
        [
            "",
            "## Workflow impact",
            "",
            "Each subsection is one matched topology. Main workflow cells are p50; "
            "aggregate rollout throughput is total complete samples divided by total "
            "version time. Full mean/p50/p95/min/max/stddev data for every metric is "
            "in `topology-matrix-summary.csv`.",
            "",
        ]
    )
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for validation in completed:
        grouped[(str(validation["topology"]), int(validation["rollout_tp"]))][
            str(validation["backend"])
        ] = validation

    workflow_metrics = [
        ("param_sync_s", "Trainer publication stall (s)", "p50"),
        ("publication_duty_pct", "Publication duty cycle (%)", "p50"),
        ("version_time_s", "Synchronization cycle (s)", "p50"),
        ("trainer_iteration_s", "Trainer iteration (s)", "p50"),
        ("rollout_throughput_samples_s", "Rollout throughput p50 (sample/s)", "p50"),
        ("rollout_throughput_samples_s", "Rollout throughput aggregate (sample/s)", "aggregate_rate"),
        ("rollout_active_s", "Rollout active interval (s)", "p50"),
        ("rollout_inactive_s", "Rollout inactive interval (s)", "p50"),
        ("generation_s", "Trainer-observed generation (s)", "p50"),
        ("actor_update_s", "Actor update (s)", "p50"),
        ("trainer_wait_s", "Trainer wait (s)", "p50"),
        ("actor_mfu", "Actor MFU", "p50"),
        ("trainer_idle_ratio", "Trainer idle ratio", "p50"),
        ("train_resource_utilization", "Trainer resource utilization", "p50"),
        ("rollout_resource_utilization", "Rollout resource utilization", "p50"),
        ("combined_resource_utilization", "Combined resource utilization", "p50"),
        ("staleness_samples", "Staleness samples", "p50"),
        ("pending_queue_size", "Pending queue size", "p50"),
        ("trainer_blocked_accelerator_s", "Trainer blocked accelerator-s", "p50"),
        ("rollout_blocked_accelerator_s", "Rollout blocked accelerator-s", "p50"),
        ("total_blocked_accelerator_s", "Total blocked accelerator-s", "p50"),
    ]
    for (topology, rollout_tp), backends in sorted(grouped.items()):
        exemplar = next(iter(backends.values()))
        trainer_mode = "MCore FSDP" if exemplar["megatron_fsdp"] else "Megatron DDP"
        lines.append(
            f"### {topology}, rollout TP{rollout_tp} "
            f"({exemplar['rollout_engines']} engines, {trainer_mode})"
        )
        lines.append("")
        lines.append("Publication steady-state distribution (steps 8-40):")
        lines.append("")
        lines.append("| Backend | Mean (s) | p50 (s) | p95 (s) | Min (s) | Max (s) | Stddev (s) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for backend in backend_order:
            validation = backends.get(backend)
            if not validation:
                continue
            row = lookup.get((str(validation["run"]), "param_sync_s"))
            if row:
                lines.append(
                    f"| {backend_labels[backend]} | {format_value(row['mean'])} | "
                    f"{format_value(row['p50'])} | {format_value(row['p95'])} | "
                    f"{format_value(row['min'])} | {format_value(row['max'])} | "
                    f"{format_value(row['population_stddev'])} |"
                )
        lines.append("")
        lines.append("Workflow impact over complete cycles (steps 8-39):")
        lines.append("")
        lines.append("| Metric | WeightRail | Awex NCCL | veRL NCCL |")
        lines.append("| --- | ---: | ---: | ---: |")
        for metric, label, statistic in workflow_metrics:
            values = []
            for backend in backend_order:
                validation = backends.get(backend)
                row = (
                    lookup.get((str(validation["run"]), metric))
                    if validation else None
                )
                value = row.get(statistic) if row else None
                values.append("-" if value in (None, "") else format_value(value))
            lines.append(f"| {label} | {' | '.join(values)} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.matrix_root
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    run_dirs = sorted(
        path for path in args.matrix_root.iterdir()
        if path.is_dir() and RUN_RE.match(path.name) and (path / "run.log").exists()
    )
    for run_dir in run_dirs:
        run_meta = metadata(run_dir)
        steps, progress_seconds, profile = parse_log(run_dir / "run.log")
        exit_code = read_exit_code(run_dir)
        validations.append(
            run_validation(run_meta, steps, exit_code, progress_seconds, profile)
        )
        rows_for_run: list[dict[str, object]] = []
        for step in sorted(steps):
            row: dict[str, object] = {**run_meta, "step": step, **steps[step]}
            enrich(row)
            rows_for_run.append(row)
            step_rows.append(row)

        for metric, window, label in SUMMARY_METRICS:
            first, last = (
                (PUBLICATION_FIRST, PUBLICATION_LAST)
                if window == "publication"
                else (WORKFLOW_FIRST, WORKFLOW_LAST)
            )
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
                complete_rows = [
                    row for row in rows_for_run
                    if first <= int(row["step"]) <= last
                    and "generated_samples" in row and "version_time_s" in row
                ]
                summary["aggregate_rate"] = sum(
                    float(row["generated_samples"]) for row in complete_rows
                ) / sum(float(row["version_time_s"]) for row in complete_rows)
            summaries.append(summary)

    meta_fields = [
        "run", "topology", "actor_tp", "actor_pp", "actor_cp", "actor_ep",
        "megatron_fsdp", "rollout_tp", "rollout_engines", "backend",
    ]
    step_fields = meta_fields + ["step"] + list(RAW_METRICS.values()) + list(DERIVED_METRICS)
    summary_fields = meta_fields + [
        "metric", "label", "window", "first_step", "last_step", "n", "mean",
        "p50", "p95", "min", "max", "population_stddev", "aggregate_rate",
    ]
    validation_fields = meta_fields + [
        "status", "exit_code", "progress_seconds", "parsed_step_count",
        "publication_sample_count", "workflow_sample_count",
        "missing_publication_steps", "missing_workflow_steps",
        "bad_generated_sample_steps", "gin_full", "gin_non_full", "plan_cache_hit",
    ]
    write_csv(output_dir / "topology-matrix-steps.csv", step_rows, step_fields)
    write_csv(output_dir / "topology-matrix-summary.csv", summaries, summary_fields)
    write_csv(output_dir / "topology-matrix-validation.csv", validations, validation_fields)
    write_report(output_dir / "topology-matrix-report.md", validations, summaries)

    passed = sum(row["status"] == "pass" for row in validations)
    print(f"Analyzed {len(run_dirs)} run(s); {passed} passed complete-run validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
