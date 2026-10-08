# Qwen3-30B-A3B：真实 veRL FP8 权重更新与 workflow impact

日期：2026-10-08。三组实验均完成 10 个真实 fully-async GRPO 训练 step 并正常退出，使用原生 veRL NCCL 和 Awex device v2 FP8/swizzle，不包含 awex-nccl。这不是 mock 通信测试。以关闭细粒度计时的原生组为性能基线，完整参数更新 p50 从 **8600.60 ms 降至 341.77 ms，25.17×**；完整训练 step p50 从 **30.390 s 降至 21.771 s，缩短 28.36%**；累计 rollout throughput 提升 **37.25%**。

## 配置和统计口径

四台 H20，每台 8 卡：训练为 11.18.50.220、33.0.194.195；推理为 11.18.56.89、33.240.39.192。两组实际节点角色相同。训练 BF16，TP2/PP1/CP2/EP8/ETP1；推理 TP2 × 8 replicas，vLLM 0.27 的 E4M3 FP8、128×128 blockwise、Triton linear/MoE。Router、embedding、norm、lm_head 保持 BF16。真实 GSM8K，32 prompts × 4 responses，prompt/response 上限均为 1024。

Publication 0–2 为初始化／warmup，稳态参数更新为版本 3–10，共 8 次。完整 workflow 为 cycle 3–9，共 7 个窗口，每窗 128 samples。Trainer 聚合指标延后一轮，rollouter reset 指标不延后；CSV 按 trainer log step 4–10 与 rollout log step 3–9 对齐，排除最后的 partial drain。p95 是小样本描述性分位数，不是长期稳定性保证。

## 开启细粒度计时的主对照

| 指标 | 原生 veRL NCCL FP8 | Awex device v2 FP8/swizzle |
|---|---:|---:|
| 完整参数更新 p50 / p95 | 9231.60 / 9820.74 ms | 341.77 / 347.90 ms |
| 完整训练 step p50 / p95 | 30.507 / 32.159 s | 21.771 / 23.051 s |
| 参数同步 duty p50 / p95 | 30.86 / 32.02% | 1.58 / 1.62% |
| Rollout throughput p50 / p95 | 4.195 / 4.429 samples/s | 5.878 / 6.000 samples/s |
| Rollout throughput，896 samples / 总窗口时间 | 4.185 samples/s | 5.846 samples/s |
| Trainer-observed generation p50 / p95 | 11.702 / 12.137 s | 11.958 / 11.985 s |
| Actor update p50 / p95 | 9.603 / 10.896 s | 9.499 / 12.312 s |
| Rollout inactive p50 / p95 | 18.348 / 20.075 s | 9.384 / 11.252 s |
| Combined resource utilization p50 / p95 | 32.31 / 33.36% | 45.46 / 47.21% |
| Blocked accelerator-seconds p50 / p95 | 443.04 / 476.45 GPU·s | 155.66 / 185.51 GPU·s |

主对照参数更新 p50 加速 27.01×；完整训练 step p50 缩短 28.64%；按总窗口时间计算 rollout throughput 提升约 39.7%。Blocked accelerator-seconds 为 16×trainer param-sync stall + 16×rollout inactive interval，属于工作流推导指标，不是硬件计数器。Resource utilization 是框架 compute-time/allocation-time 指标，不是 NVML SM occupancy。Trainer idle ratio 单独看会因周期缩短而上升，不能据此判定吞吐下降。

## 转换、加载与传输 breakdown

下表先对每个 publication 取所有参与 rank 的最大值，再对 8 个 publication 求分位数；不同项的最慢 rank 可以不同，不能把这些 p50 相加。加载与转换是包含关系，NCCL pipeline 与导出／加载存在重叠和 backpressure。

| 原生 veRL 阶段 | p50 / p95，ms | 含义 |
|---|---:|---|
| BF16→FP8 host interval | 1375.39 / 1394.32 | 18,624 次转换的累计 host enqueue 区间 |
| BF16→FP8 CUDA stream interval | 1130.03 / 1148.10 | 含 host launch gaps；不是 exclusive SM active time |
| vLLM load inclusive，wall | 5824.18 / 5895.27 | 包含量化、Python dispatch、权重映射／TP shard 加载 |
| vLLM load inclusive，stream | 5793.58 / 5862.37 | 包含上述量化区间 |
| 同一 worker load stream 减 conversion，再取 rank max | 4660.42 / 4716.54 | stream 上剩余加载区间，仍可能含 host gaps |
| vLLM prepare layout，wall | 4.41 / 6.32 | 量化布局准备 |
| vLLM finalize layout，wall | 1.62 / 2.39 | 布局收尾 |
| Reload residual wall | 3166.74 / 3881.54 | 总 reload 减 prepare/load/finalize，包含等待／IPC／循环开销，不能当纯网络时间 |
| 完整 vLLM reload wall | 8677.98 / 9392.77 | 每个 worker 的完整更新窗口 |
| Lazy actor export iterator，wall | 7680.07 / 7837.22 | 不含 yield 下游消费时间，但可包含 collectives 等待，不能当纯 CPU conversion |
| NCCL send pipeline | 7679.07 / 7783.16 | 包含 lazy export 和 packing |
| NCCL receive pipeline | 8633.71 / 9240.48 | 包含 source readiness 和下游 IPC/reload backpressure |
| Checkpoint manager weight_update | 9123.49 / 9746.74 | 等待全部 sender / receiver 完成 |
| 完整 checkpoint workflow | 9231.60 / 9820.74 | 包含 generation/cache/process group 控制 |

| Awex 阶段 | p50 / p95，ms | 含义 |
|---|---:|---|
| Writer kernel critical path | 264.55 / 265.84 | 融合 BF16→FP8、传输／转发、写入目标；不单独拆量化 |
| Reader kernel critical path | 260.07 / 261.45 | FP8 接收／转发与写入 |
| Writer backend execute | 266.80 / 268.04 | 包含 extension 调用开销 |
| Reader total transfer | 273.58 / 274.89 | 含控制／barrier 和返回路径 |
| Checkpoint manager weight_update | 290.70 / 293.98 | 含 adapter 的 KV cache clear 与版本发布 |
| 完整 checkpoint workflow | 341.77 / 347.90 | 同步、融合更新和 cache/generation 恢复 |

Awex 日志中的 `convert_time_ms=0` 表示没有单独的 HF-format conversion，绝不表示 FP8 量化免费；量化已包含在 kernel 时间内。本轮没有修改 device v2 kernel。

## Payload 与公平性

原生 publication 0 是 FP32，122,128,490,496 bytes，仅初始化使用，已排除；所有稳态 actor export 和 receiver payload 都是 BF16，61,064,245,248 bytes。每个推理 worker 在 TP shard loading 前转换完整的 18,624 个 eligible tensors：输入 59,793,997,824 bytes，FP8 输出 29,896,998,912 bytes，FP32 scales 7,299,072 bytes。16 个推理 worker 都执行此转换；其输出计数不能当成实际保存的 TP shard 大小。

Awex 从分布式 BF16 训练 shard 直接量化并传／转发 FP8，到原生 TP2 inference storage。逐 pair payload 不均匀，已由 writer expected_counts 重建有向流量，按 swizzle routing 重建 relay，并校验全部 32 ranks 在全部 8 次稳态更新的 peer bytes、payload totals、GIN/LSA peer count。每次 application-level GIN 流量 **62,444,060,672 bytes**，LSA 流量 **187,332,182,016 bytes**，含 scale/framing 开销。分别除以完整更新 p50 得到约 182.71 / 548.13 GB/s 的聚合业务流量速率；这是多节点／多链路累计值，不能当单卡带宽或模型唯一字节吞吐。

`directed-pair-payloads.csv` 和 `summary.json` 保存逐 pair、逐节点和 requested HCA 的流量／容量比例。各 HCA 声称 200 Gb/s，绑定日志完整；实际每连接 GIN rail routing 和物理 NIC counters 未测，容量比例只在 requested HCA 等于实际路径时成立，不能声称测得物理利用率。原生每-worker receive payload 是 API 字节量，也不能简单乘 16 当实际跨节点 TX。

两组 inference FP8 格式相同，但实现不同：原生发送 BF16 full checkpoint、在每个推理 worker 量化并加载；Awex 在训练侧 fused kernel 中量化并用 ring 转发。27× 是这些完整实现的更新延迟对照，包含消除重复量化、checkpoint export 和加载开销，不能解释成只靠 BF16→FP8 把传输时间减半。两组独立训练的轨迹随机且受调度影响，10 steps 验证真实更新／生成／训练工作流，不构成长训练收敛或逐位数值等价证明。

## 关闭计时的原生控制组

同配置、同节点角色的 10-step 控制组已完成，exit 0。`VERL_FP8_UPDATE_PROFILE=0` 关闭 per-tensor CUDA events、export profiling 和细粒度记录，保留框架原生版本化 parameter-sync wall timer；完整日志验证没有 `VERL_FP8_PROFILE` 记录。

| 指标 | 原生 veRL，关闭细粒度计时 | Awex FP8/swizzle |
|---|---:|---:|
| 完整参数更新 p50 / p95 | 8600.60 / 9218.55 ms | 341.77 / 347.90 ms |
| 完整训练 step p50 / p95 | 30.390 / 30.902 s | 21.771 / 23.051 s |
| 参数同步 duty p50 / p95 | 28.42 / 31.78% | 1.58 / 1.62% |
| Rollout throughput p50 / p95 | 4.211 / 4.464 samples/s | 5.878 / 6.000 samples/s |
| Rollout throughput，896 samples / 总窗口时间 | 4.259 samples/s | 5.846 samples/s |
| Generation p50 / p95 | 11.839 / 11.930 s | 11.958 / 11.985 s |
| Actor update p50 / p95 | 9.946 / 10.345 s | 9.499 / 12.312 s |
| Rollout inactive p50 / p95 | 18.337 / 18.865 s | 9.384 / 11.252 s |
| Combined resource utilization p50 / p95 | 32.88 / 33.79% | 45.46 / 47.21% |
| Blocked accelerator-seconds p50 / p95 | 428.35 / 441.83 GPU·s | 155.66 / 185.51 GPU·s |

开启计时的原生 publication p50 高于关闭计时组约 7.34%，说明计时开销不可忽略；两次独立运行还包含调度和训练轨迹波动，因此这个差值不是单独 CUDA events 的精确因果开销。性能加速采用关闭计时基线的 25.17×，转换／加载 breakdown 仅描述开启计时组，不能据此声称关闭计时后的纯 quant kernel 仍精确耗时 1.13 s。

## 代码、验证与归档

实测 veRL revision：`33e9e90221007610b71deea43ac46ec16286d81d`，分支 `codex/verl-fp8-e2e`，Test 仓库。后续分支提交只调整离线分析，不改变已测 runtime。

Awex 主对照 revision：`b689ea280e17234fb9b26701fdde4a1afa1e3637`，分支 `codex/device-v2-fp8-verl-e2e`。仅 adapter 增加成功传输后的 KV cache clear 和 global step 发布，符合原生 checkpoint contract。4 项 adapter 测试通过，传输失败时不会发布新版本。四节点复用相同已验收 extension：SHA256 `8406492c439aa781ba1a92470212fa4b1f0886afbbac3100629422aaa71c5a87`。

主对照每个稳态版本有原生 16 个 reload profile 或 Awex 32 个 transfer profile；原始 run.log、exit_code、source revisions、extension checksum、逐 worker JSON、workflow CSV、pair CSV、配置/preflight/deployment/test 日志和 qualified extension binary 均归档在本目录。初始化失败的 `nccl` 目录保留但不参与统计。`deployed.json` 是首次尝试的部署快照，`deployed-rerun.json` 是实际成功实验的 runtime revision 校验。实验结束后的检查确认四节点没有 GPU compute processes，Ray 的 32 张 GPU 均已释放。原始本地 veRL checkout 的用户修改未触碰。

复算：`python3 /Users/xiaojiayu/work/rl/verl-fp8-e2e/scripts/awex_h20/analyze_fp8_e2e.py /Users/xiaojiayu/work/rl/results/verl-fp8-e2e-20261008`。
