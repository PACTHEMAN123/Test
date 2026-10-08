#!/usr/bin/env python3
"""Compare completed FP8 topology pairs with an explicitly reused baseline."""

import argparse
import copy
import csv
import json
from pathlib import Path

from analyze_fp8_e2e import analyze


def aggregate(run, path):
    if run.get("workflow_totals"):
        return run["workflow_totals"]["aggregate_samples_s"]
    rows = list(csv.DictReader((path / "workflow-steps.csv").open()))
    return sum(float(row["generated_samples"]) for row in rows) / sum(
        float(row["version_time_s"]) for row in rows
    )


def cell(value, scale=1):
    return (
        f"{value['p50'] * scale:.3f} / {value['p95'] * scale:.3f}"
        if value.get("n")
        else "—"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    status_path = args.root / "experiment-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    results = []
    baseline = json.loads((args.baseline / "summary.json").read_text())
    for name, backend in (
        ("nccl-unprofiled", "nccl"),
        ("awex_weightrail", "awex_weightrail"),
    ):
        result = copy.deepcopy(next(row for row in baseline if row["run"] == name))
        result["case"] = {
            "label": "p1t2c2e8-rtp2",
            "backend": backend,
            "actor_pp": 1,
            "actor_tp": 2,
            "actor_cp": 2,
            "actor_ep": 8,
            "rollout_tp": 2,
            "fanout": 8,
            "megatron_fsdp": 1,
        }
        result["origin"] = str(args.baseline / name)
        result["reused_baseline"] = True
        result["aggregate_samples_s"] = aggregate(result, args.baseline / name)
        results.append(result)
    for path in sorted(args.root.iterdir()):
        if (
            not path.is_dir()
            or not (path / "exit_code").exists()
            or not (path / "case.json").exists()
        ):
            continue
        result = analyze(path)
        result["origin"] = str(path)
        result["reused_baseline"] = False
        result["aggregate_samples_s"] = (
            aggregate(result, path) if result["complete"] else None
        )
        native_payloads = result.get("native_receive_payloads", [])
        if result["complete"] and result["case"]["backend"] == "nccl":
            assert len(native_payloads) == 128, path
            for publication in range(3, 11):
                ranks = [
                    p["rank"] for p in native_payloads if p["step_id"] == publication
                ]
                assert sorted(ranks) == list(range(1, 17)), (path, publication, ranks)
            assert all(
                sum(p["dtype_bytes"].values()) == p["payload_bytes"]
                for p in native_payloads
            ), path
        results.append(result)
    (args.root / "matrix-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = []
    for result in results:
        row = {
            "topology": result["case"]["label"],
            "backend": result["case"]["backend"],
            "complete": result["complete"],
            "origin": result["origin"],
            "reused_baseline": result["reused_baseline"],
            "aggregate_samples_s": result["aggregate_samples_s"],
        }
        for metric, values in {
            "publication_s": result["publication"],
            **result["workflow"],
        }.items():
            for stat, value in values.items():
                row[f"{metric}/{stat}"] = value
        payload = result.get("directed_payload")
        if payload:
            latency_s = result["publication"]["p50"]
            row["GIN_payload_bytes"] = payload["GIN_bytes"]
            row["LSA_payload_bytes"] = payload["LSA_bytes"]
            row["GIN_application_GBps"] = payload["GIN_bytes"] / latency_s / 1e9
            row["requested_port_peak_capacity_pct"] = max(
                port["percent_of_requested_200Gbps_cap"]
                for port in payload["assigned_port_loads"]
            )
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.root / "matrix-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pairs = {}
    for result in results:
        if result["complete"]:
            pairs.setdefault(result["case"]["label"], {})[result["case"]["backend"]] = (
                result
            )
    lines = [
        "# 真实 veRL FP8：训练与推理拓扑对照",
        "",
        "四台 H20：训练 11.18.50.220 + 33.0.194.195；推理 11.18.56.89 + 33.240.39.192。"
        "Qwen3-30B-A3B，BF16 训练、E4M3 128×128 blockwise FP8 推理，真实 fully-async GRPO。",
        "",
        "每组 10 steps，publication 0–2 为初始化/warmup，稳态更新 3–10 共 8 次，完整 workflow cycle 3–9 共 7 次。"
        "原生性能组关闭逐 tensor CUDA-event 计时；Awex 量化融合在原有传输 kernel 内，含 swizzle ring。"
        "未包含 awex-nccl。TP2→TP2 的基准 pair 来自已完成实验，显式标记 reused；其余为本轮新运行。",
        "",
        "## 完整更新与训练周期",
        "",
        "表中延迟为 p50 / p95；吞吐按实际 complete samples / 总 version time 计算。",
        "",
        "| 训练布局 → 推理布局 | 原生更新 ms | Awex 更新 ms | 更新 p50 加速 | 原生 step s | Awex step s | 原生 / Awex samples/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    complete_pairs = 0
    for label, pair in sorted(pairs.items()):
        if set(pair) != {"nccl", "awex_weightrail"}:
            continue
        complete_pairs += 1
        native, awex = pair["nccl"], pair["awex_weightrail"]
        speedup = native["publication"]["p50"] / awex["publication"]["p50"]
        lines.append(
            f"| {label} | {cell(native['publication'], 1000)} | {cell(awex['publication'], 1000)} | {speedup:.2f}× | "
            f"{cell(native['workflow']['trainer_iteration_s'])} | {cell(awex['workflow']['trainer_iteration_s'])} | "
            f"{native['aggregate_samples_s']:.3f} / {awex['aggregate_samples_s']:.3f} |"
        )
    lines += [
        "",
        f"最终完成配对：{complete_pairs} 组（基准复用 1 组，本轮新增 1 组）。"
        "用户已结束本轮实验；不继续运行剩余拓扑。完整分布及来源见 matrix-summary.json / CSV。"
        if status.get("status") == "stopped_by_user"
        else f"已完成配对：{complete_pairs}/4。完整分布及来源见 matrix-summary.json / CSV。",
        "",
        "两组实际训练配置均为 PP1/CP2/EP8/ETP1、MCore FSDP，仅 TP2/TP4 不同；"
        "推理均为 TP2×8。PP4 已撤出本轮，CP1 未运行。"
        "本轮未新增 off/naive/swizzle 或 BF16/FP8 的匹配消融。",
        "",
        "## Workflow impact 与 payload",
        "",
    ]
    metrics = (
        ("param_sync_s", "参数同步 stall，s", 1),
        ("publication_duty_pct", "参数同步 duty，%", 1),
        ("generation_s", "Generation，s", 1),
        ("actor_update_s", "Actor update，s", 1),
        ("rollout_active_s", "Rollout active，s", 1),
        ("rollout_inactive_s", "Rollout inactive，s", 1),
        ("combined_resource_utilization", "Combined resource utilization，%", 100),
        ("actor_mfu", "Actor MFU，%", 100),
        ("total_blocked_accelerator_s", "Blocked accelerator-seconds，GPU·s", 1),
        ("generated_samples", "每窗口 generated samples", 1),
        ("staleness_samples", "Staleness samples", 1),
    )
    for label, pair in sorted(pairs.items()):
        if set(pair) != {"nccl", "awex_weightrail"}:
            continue
        native, awex = pair["nccl"], pair["awex_weightrail"]
        lines += [
            f"### {label}",
            "",
            "| 指标，p50 / p95 | 原生 NCCL FP8 | Awex FP8/swizzle |",
            "|---|---:|---:|",
        ]
        for metric, title, scale in metrics:
            lines.append(
                f"| {title} | {cell(native['workflow'][metric], scale)} | {cell(awex['workflow'][metric], scale)} |"
            )
        payload = awex["directed_payload"]
        kernel = awex["breakdown"].get(
            "awex/all/kernel_transfer_time_ms/rank_max_ms",
            awex["breakdown"]["awex/writer/kernel_transfer_time_ms/rank_max_ms"],
        )
        lines += [
            "",
            f"Awex 融合 kernel：{cell(kernel)} ms；跨节点 GIN：{payload['GIN_bytes'] / 1e9:.6f} GB/update；"
            f"节点内 LSA：{payload['LSA_bytes'] / 1e9:.6f} GB/update。"
            "有向 pair 从实际 writer 字节数和 ring 路由重建，并与全部 32 ranks 的日志校验，包含 relay、scale/framing。",
        ]
        latency_s = awex["publication"]["p50"]
        ports = payload["assigned_port_loads"]
        lines += [
            "",
            f"按完整更新中位数计算，集群 GIN 应用吞吐为 {payload['GIN_bytes'] / latency_s / 1e9:.3f} GB/s。"
            "该总量包含 relay 的逐 hop 字节，不能当作单张网卡的吞吐。",
            "",
            "| 节点／请求 HCA／方向 | GB/update | GB/s | 请求 200 Gb/s 容量比例，% |",
            "|---|---:|---:|---:|",
        ]
        for port in ports:
            lines.append(
                f"| {port['node']} / {port['requested_hca']} / {port['direction']} | "
                f"{port['payload_bytes'] / 1e9:.3f} | {port['step_average_GBps']:.3f} | "
                f"{port['percent_of_requested_200Gbps_cap']:.2f} |"
            )
        if native.get("native_receive_payloads"):
            variants = sorted(
                {
                    (p["payload_bytes"], tuple(sorted(p["dtype_bytes"].items())))
                    for p in native["native_receive_payloads"]
                }
            )
            lines += [
                "",
                f"原生逐 worker receive 字节数／dtype：`{variants}`。这是 API 接收字节数，不能乘 16 当实际跨节点 TX。",
            ]
        lines += ["", f"来源：原生 `{native['origin']}`；Awex `{awex['origin']}`。", ""]
    lines += [
        "## TP4 推理的配置限制",
        "",
        "推理 TP4×4 在 vLLM 构建 FP8 MoE weights 时失败：模型 intermediate_size768 / TP4 =192，"
        "不整除 block_n128，错误发生在传输之前。失败日志保留，未计入性能对照。"
        "本轮保持同一 128×128 blockwise 格式和 EP1 推理，因此固定推理 TP2×8、扩展训练侧布局。",
        "",
        "## 解释与限制",
        "",
        "原生广播训练侧完整权重，推理 worker 在 TP shard 加载前量化；Awex 在训练 shard 上融合量化、"
        "以 FP8 转发并直接写入目标 shard。收益包含字节量、量化重复度、并行传输、checkpoint export/loading 的差异，"
        "不能只归因于 BF16→FP8 的约 2× 压缩。",
        "",
        "此前单独开启计时的 TP2 基准测得 conversion stream interval p50 1130 ms，含 host launch gaps；"
        "本轮新拓扑不启用该重计时，不把旧 conversion timing 当成新拓扑的实测值。",
        "",
        "物理 NIC counters 未测；requested-HCA 容量比例是条件估算，不是实际线速利用率。"
        "TP4 组个别条件比例略超 100%，说明 requested-HCA/标称容量假设不足以成为物理上限。"
        "Blocked accelerator-seconds 是工作流推导指标，resource utilization 是框架 compute-time/allocation-time。"
        "每组只有 8 次稳态更新，p95 为小样本描述性分位数；短训练验证工作流，不代表长期收敛或逐位数值等价。",
        "",
        "## 运行版本与收尾",
        "",
        "复用 TP2 基准的 veRL runtime 为 `33e9e90221007610b71deea43ac46ec16286d81d`；"
        "新增 TP4 对照为 `4acc6867cd71a0b30e4bfd53f936b91f80c48626`，"
        "新增的是训练拓扑参数和不含 CUDA event 的 native payload 计数。"
        "Awex adapter 均为 `b689ea280e17234fb9b26701fdde4a1afa1e3637`；"
        "device v2 kernel 未修改，四节点 binary SHA256 均为 "
        "`8406492c439aa781ba1a92470212fa4b1f0886afbbac3100629422aaa71c5a87`。",
        "",
        "原始日志、exit_code、逐 rank profile、workflow CSV、逐 pair payload、配置及版本均已本地归档。"
        "旧基准保存在独立目录，未重写其数据。实验停止状态见 experiment-status.json；"
        "最终节点检查见 postflight.json；两组均完成 10 步并正常退出。",
        "",
    ]
    (args.root / "REPORT.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "complete_pairs": complete_pairs,
                "complete_runs": sum(r["complete"] for r in results),
                "new_complete_runs": sum(
                    r["complete"] and not r["reused_baseline"] for r in results
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
