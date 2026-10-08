# Real veRL FP8 topology results, 2026-10-08

The user ended this experiment after two matched topology pairs. One pair is reused from the completed FP8 baseline; the TP4 training pair was newly measured. No remaining experiment is scheduled. All four H20 nodes have no GPU compute processes and Ray has released all 32 GPUs.

Qwen3-30B-A3B, training BF16, inference E4M3 128×128 blockwise FP8, real fully-async GRPO. Training uses two nodes / 16 GPUs, PP1/CP2/EP8/ETP1 with MCore FSDP. Inference uses two nodes / 16 GPUs, TP2 × 8 replicas. Each run completes 10 steps. Updates 0–2 are initialization/warmup; results use updates 3–10 (8 samples), and complete workflow cycles 3–9 (7 windows). Native NCCL performance runs disable per-tensor CUDA-event profiling. Awex uses the accepted device v2 fused FP8 kernel and swizzle ring, unchanged in this experiment. Awex NCCL is excluded.

| Training TP | Native update p50 / p95, ms | Awex update p50 / p95, ms | Update speedup | Native / Awex step p50, s | Step reduction | Native / Awex aggregate samples/s | Throughput gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP2, reused baseline | 8600.600 / 9218.550 | 341.766 / 347.899 | 25.17× | 30.390 / 21.771 | 28.36% | 4.259 / 5.846 | 37.25% |
| TP4, new matched pair | 8422.150 / 8712.750 | 311.099 / 319.588 | 27.07× | 32.890 / 24.640 | 25.08% | 3.914 / 5.035 | 28.64% |

Awex all-rank maximum fused kernel p50 is 264.549 ms for TP2 training and 233.032 ms for TP4 training. Checkpoint-manager weight-update p50 is 290.703 and 262.894 ms respectively. Complete update additionally includes generation/cache/process-group control; nested percentiles must not be summed. Parameter-sync duty p50 falls from 28.42% to 1.58% in TP2 and from 25.82% to 1.26% in TP4. Rollout inactive p50 falls from 18.337 to 9.384 s and from 20.883 to 12.310 s respectively.

Both Awex layouts have exactly 62,444,060,672 application-level GIN bytes and 187,332,182,016 LSA bytes per update, including relay, scale and framing overhead. Directed flows retain unequal writer-pair payloads and pass all 32-rank consistency checks over all eight steady updates. GIN bytes divided by full-update p50 yield aggregate application rates of 182.710 / 200.721 GB/s; these sum multiple links and relay hops, not a single NIC's bandwidth. Requested-HCA capacity ratios are conditional estimates; actual physical NIC counters were not measured. The TP4 maximum conditional ratio slightly exceeds 100%, so the assumed requested routing / advertised cap must not be called a measured physical limit.

Native steady BF16 full-checkpoint API receive payload is 61,064,245,248 bytes per worker. The new TP4 pair validates this for all 16 receivers over all eight steady versions. Receiver totals must not be interpreted as actual cross-node TX. The separately profiled earlier TP2 native run measured conversion stream intervals around 1130 ms, including host launch gaps. It is diagnostic evidence, not new TP4 conversion timing. Overall 25–27× update gains include source-side quantization, smaller forwarding traffic, removal of repeated receiver quantization, and different export/loading paths; they cannot be attributed solely to approximately 2× dtype compression.

Inference TP4 × 4 failed before transport: expert intermediate width768 / TP4 =192 is not divisible by block_n128. The failed log is retained and excluded. PP4 was withdrawn, CP1 was not run, and this round added no ring or quantization ablation. Short independent training runs do not establish convergence, numerical identity, or a causal reason for the approximately 9% cross-layout Awex latency difference.

Training nodes: 11.18.50.220 and 33.0.194.195. Inference nodes: 11.18.56.89 and 33.240.39.192. Reused baseline veRL runtime: `33e9e90221007610b71deea43ac46ec16286d81d`; new pair: `4acc6867cd71a0b30e4bfd53f936b91f80c48626`. Awex adapter: `b689ea280e17234fb9b26701fdde4a1afa1e3637`. Unchanged extension SHA256: `8406492c439aa781ba1a92470212fa4b1f0886afbbac3100629422aaa71c5a87`.

Local detailed report, distributions, raw logs, per-rank profiles, workflow CSV, directed pair CSV, requested-port loads, versions and final node checks: `/Users/xiaojiayu/work/rl/results/verl-fp8-topology-20261008`. Reused baseline and separate conversion profile: `/Users/xiaojiayu/work/rl/results/verl-fp8-e2e-20261008`. The earlier baseline archive remains unchanged.

Recompute: `python3 scripts/awex_h20/analyze_fp8_topology.py /Users/xiaojiayu/work/rl/results/verl-fp8-topology-20261008 --baseline /Users/xiaojiayu/work/rl/results/verl-fp8-e2e-20261008`.
