# 第六册第二章：推理基础资料驱动精修记录

## 研究范围与核心问题图谱

- 日期：2026-10-26。
- 承接 `progress_v2.md` 当前 next，精修 [`book-06-llm-deployment/chapters/02-推理基础.md`](../book-06-llm-deployment/chapters/02-推理基础.md)；之前已完成章节未重做。
- 本章要回答的核心问题：一次 autoregressive request 如何分成 prefill/decode；输入长度、输出长度、batch、权重与 KV 流量怎样分别影响运算和延迟；TTFT、TPOT、ITL、E2E、分位数和吞吐分别以什么时间点/样本为分母；如何从 trace 和 profiler 区分 kernel、带宽、排队与跨阶段干扰；代表性系统工作分别改变了哪项约束；压测到的数字如何避免把同时到达 cohort 当成稳态容量。
- 原有结构 §2.1—§2.19 全部保留；本轮补深既有 roofline、指标和 benchmark 主线，没有删掉 conditional generation、prefill/decode、KV、padding、continuous batching、排队、prefix cache、训练/推理区别、workload 分类、错误边界或容量决策等问题。

## 来源与实际核验内容

资料用于形成主题地图、核对具体机制与当前公开 benchmark 语义；动态文档按 2026-10-26 可见内容记录，之后需随目标 runtime 版本复查。

| 来源 | 已读内容与支持的判断 | 证据边界 |
| --- | --- | --- |
| [A Survey of LLM Inference Systems, arXiv v1](https://arxiv.org/abs/2506.21901) | 阅读摘要、引言及 request processing、operator、batching/scheduling、memory management、single/multi-replica taxonomy，作为本章问题地图 | 综述用于分类和追溯文献，不替代各工作原论文，也不能证明当前框架版本的行为 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 阅读方法与 IO complexity：分块/在线 softmax 避免将完整 attention 中间矩阵往返写入 HBM；目标仍为精确 attention | 主要说明内存移动和临时空间，标准 dense attention 的 FLOPs 仍随长度平方增长；论文加速结果受 GPU、形状和实现限制 |
| [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785) 与 [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 核对 arithmetic intensity、峰值计算/带宽交叉点、ridge point、achieved kernel value 与 profiling overhead；据此把 roofline 限定为匹配精度和内存层级的 kernel/阶段分析 | 峰值是上限而非持续实测；profiling 采集本身可能有开销；整条 request 的 queue/network/CPU 与单 kernel roofline 不同 |
| [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | 核对 iteration-level scheduling 的动机：在每个生成迭代边界重新调度请求，降低 request-level 静态 batch 的等待和空槽 | 论文收益绑定其模型、硬件、基线、流量和实现，不意味着它本身解决 prefill 与 decode 争用 |
| [Sarathi-Serve, arXiv v3](https://arxiv.org/abs/2403.02310) | 阅读摘要、机制和评测：chunked prefill 与 stall-free scheduling 将长 prompt 分片插入 decode 迭代以减轻 decode stall | 论文公布的 serving-capacity 倍数绑定 Mistral-7B/A100、Yi-34B/双 A100、Falcon-180B/PP 等实验条件；chunk 参数和负载变化会改变 TTFT/吞吐 |
| [Splitwise, arXiv v2](https://arxiv.org/html/2311.18677) | 阅读阶段刻画、KV transfer、机器/集群调度、实验方法和结果。论文把 prompt computation 作为偏 compute、token generation 作为偏 memory/bandwidth 的不同阶段，并跨机器拆分以分别配置资源 | 作者报告最高 1.4× throughput / 20% lower cost 或同预算 2.35× throughput；这些数字来自其生产 trace、A100/H100 及集群设计，不可外推为目标部署收益；分离带来 KV network transfer、队列配平和资源碎片 |
| [vLLM Benchmark CLI, mutable `main`](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md) | 核对客户端 TTFT/ITL/TPOT 口径、`request-rate`、`burstiness`、`max-concurrency`、`inf` 时同时送请求，以及复用 prompt 可能因 prefix cache 命中抬高吞吐 | `main` 是动态页面，文档语义会变化；需记录运行版本、commit、参数、缓存状态和负载生成器行为 |
| [TensorRT-LLM `trtllm-serve` benchmark guide, mutable `main`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-serve/run-benchmark-with-trtllm-serve.md) | 核对 TTFT、TPOT、ITL 和跨请求聚合的差异：TPOT 按请求先算再等权汇总；ITL 汇总单个间隔事件 | 页面会更新；具体工具/版本对 event bundling、单 token request 和样本过滤的定义仍须检查 |

## 研究如何改变正文

1. **把 prefill/decode FLOPs 的层数和作用范围说清。** Prefill attention 主项现在明确是单层 `4 B H_q T_in² D_h`；总层数 `L` 显式出现在全模型复杂度和 decode attention 项中。因果实现可以改变常数，但不会使标准 dense attention 的平方长度趋势消失。
2. **补齐模型级权重与 KV 账本。** 使用稠密 decoder 教学近似：prefill `2 B T_in P + 4 L B H_q T_in² D_h`；decode 单步 `2 B P + 4 L B H_q T_ctx D_h`。理想 decode 字节流量为 `P b_w + 2 L B T_ctx H_kv D_h b_kv`。明确权重在 batch 内共享、KV 读量随活跃序列与上下文增长，并说明多卡需改为 per-rank 账本。
3. **用同一硬件假设手算 batch 与 context 的共同影响。** 参数设为 `P=7B`、`L=32`、`Hq=32`、`Hkv=8`、`Dh=128`、BF16 两字节、计算峰值 120 TFLOP/s、HBM 1.6 TB/s（ridge 75 FLOPs/byte）。例：`B=64,Tctx=256` 得 904.590 GFLOPs、14 GB 权重流量、2.147 GB KV 读取、56.02 FLOPs/B、理想下界 10.09 ms（带宽侧）；`B=128,Tctx=256` 得 1.809 TFLOPs、4.295 GB KV、98.89 FLOPs/B、计算下界 15.08 ms（计算侧）；`B=64,Tctx=8192` 得约 68.72 GB KV 读取与 51.70 ms 带宽下界（带宽侧）。这些是公式推演，不是 benchmark。
4. **比较不同层次的代表方案。** 正文新增 FlashAttention、Orca、Sarathi-Serve、Splitwise 对照：它们依次主要改变 attention kernel 的 HBM/片上 IO、请求调度粒度、长 prefill 对 decode 的阻塞、以及 P/D 两阶段的机器资源配置。每项都写明作用机制、没解决的瓶颈、代价与目标测量；论文倍率没有跨工作排名。
5. **修正指标分母和时间边界。** 客户端 TTFT 从请求发送到首个 content event，不再用可能早于 token 内容的 first byte；TPOT 与 ITL 增加等权请求/合并 interval 两种统计的手算差异；E2E 明确由同一 client clock 上 send 到 response complete，并用 event 数而非假设每个事件恰好一个 token，避免重复计入网络/前端时间。
6. **修正分位数边界。** 线性插值用零基 `k=floor((n-1)p)`，通过上界夹取定义 `n=1` 和 `p=1`，并让代码检查 bool、NaN/Inf 概率与样本。
7. **让代码输出匹配它真正计算的对象。** 原 demo 的 prefill attention 强度分母只是 activation proxy、decode 分母只有单层 KV bytes，却把它们标成 whole-phase bound；现改成 per-layer attention intensity proxy 和 ridge relation，不再叫阶段瓶颈。8 个请求假设同时到达，吞吐重命名为 finite cohort rate；所有最终上下文长度之和改为假设全重叠的 KV envelope，不再称完成时的真实 cache 峰值。
8. **补足可复现压测协议。** 增加 open-loop 到达与 closed-loop 并发负载区分、burstiness/arrival process、warm-up、重复 seed、缓存冷热状态、测量窗口、错误/超时/取消、client/server 时间戳；提醒 profiler 采集开销，benchmark workload 和诊断 run 分开。

## 原有知识点覆盖图

| 原位置 | 既有问题与知识点 | 本轮处理 |
| --- | --- | --- |
| §2.1 | 条件概率、自回归依赖、请求完整生命周期、模板与停止条件 | 全部保留；作为后续 FLOPs/事件口径前提 |
| §2.2 | prefill 输入并行、因果 mask、KV 写入、attention 与线性项、计算密集的条件 | 保留；补单层/全模型范围及 `L`，解释因果 kernel 只改变常数因子 |
| §2.3 | decode 单请求串行、多请求并行、上下文逐步增长、KV 读取、长输出/长上下文差异 | 保留；补全 attention 层数，后续加权重读流量估算 |
| §2.4 | KV Cache 以空间换重算、cache 与分页细节边界 | 保留；进一步链接第 3 章的详细主讲位置 |
| §2.5 | TTFT、TPOT、ITL、E2E、finish reason、QPS/TPS、分位数 | 保留；修时间戳、聚合样本权重、E2E 分解和 percentiles 端点 |
| §2.6 | sequence batch/token work、padding、活跃 KV 状态、静态/动态/continuous batching | 保留；不移除调度基础，后续仍由第 6 章展开实现 |
| §2.7 | latency/throughput 关系、Little 定律、饱和、过载 | 保留；Little 定律统一系统边界与流量分母，单工作单元 `rho` 与简化多 worker 近似分开 |
| §2.8 | arithmetic intensity、roofline、阶段经验边界 | 深化为权重+KV 账本、手算表和 4 项论文方法比较；明确 roofline 的 kernel 范围和实际 counters 需求 |
| §2.9 | 双请求 trace 手算 TTFT/E2E | 保留；不覆盖为大模型公式，继续承担指标落地例 |
| §2.10 | 用 TTFT、TPOT、P99 变化诊断 prefill、KV、排队与长尾 | 保留并承接 §2.8 的测量顺序 |
| §2.11 | prefix cache token 条件与收益边界 | 保留；补 benchmark 冷/热缓存必须记录，避免复用输入抬高吞吐 |
| §2.12 | 训练与推理的状态、目标及 token 并行差异 | 保留 |
| §2.13 | 短聊天、RAG、代码、离线、Agent、多模态 workload 分型 | 保留；方法比较放在 roofline 主线，避免把不同方法误并成某一 workload 的“通用优化” |
| §2.14 | 零依赖 toy 指标 demo、输入与 cache 估算、失败输入边界 | 保留代码、输入集合与核心指标；重命名 proxy、cohort rate、KV envelope 并补 percentile 边界 |
| §2.15 | prefill/decode、GPU utilization、batch、分位数、量化和 cache 命中误区 | 原题全部保留 |
| §2.16 | 真实压测步骤、变量控制与记录 | 保留并加入负载模型、warm-up/cache、重复实验及测量开销 |
| §2.17 | 代码助手 workload 的容量与 SLO 决策案例 | 保留；数值仍是假设例，非目标生产容量 |
| §2.18 | 来源入口与论文/文档/实测证据边界 | 保留并加入 survey、Sarathi、Splitwise、vLLM/TensorRT benchmark 文档和 Nsight |
| §2.19 | prefill/decode、指标、cache 与排查总结 | 保留原结论并补一句权重流量随 batch 摊薄、KV 随上下文增长，两者共同决定瓶颈 |

## 数值与代码验收

- **Roofline 表：** 以十进制 GB、`F/P_peak` 与 `M/BW` 的较大值作理想时间下界。手算/脚本复核的五个 decode case 为：`B=1,T=256` 1.007 FLOPs/B、8.771 ms；`B=64,T=256` 56.020 FLOPs/B、10.092 ms；`B=128,T=256` 98.889 FLOPs/B、15.076 ms；`B=1,T=8192` 1.214 FLOPs/B、9.421 ms；`B=64,T=8192` 14.155 FLOPs/B、51.700 ms。prefill attention shares 分别是 7.12%、23.48%、55.10%。
- **指标聚合例：** 两个请求的 102 个间隔合并平均为 `(2×10+100×20)/102=19.8039 ms`；逐请求 TPOT 等权平均为 15 ms。
- **demo：** 从 Markdown 提取并用系统 Python 运行 §2.14 代码；全部 8 行输出与正文示例一致。有限 cohort 结果为 3.22 cohort requests/s、241.47 cohort output tokens/s、104.6 ms mean TTFT、2147.6 ms 插值 P95 latency、16.4 ms P95 TPOT；token envelope 为 10,200 tokens / 1,275 MiB。以上仅验证 toy 逻辑与输出一致性。
- **代码边界：** demo 内置拒绝零 input token、重复 request id、越界/boolean/非有限概率和非有限 percentile sample；并检查 singleton `p=1` 返回样本自身。
- 未运行 GPU kernel、真实模型 serving、生产 trace、论文 benchmark 或目标硬件 profiler；没有声称目标服务的吞吐、延迟、容量和成本。

## 配套同步与待验证项

- `PAPERS.md`：推理优化节加入 Splitwise 与 Roofline 来源/限制；已有 inference survey、Orca、Sarathi-Serve、DistServe 保留。
- `GLOSSARY_EN_ZH.md`：新增 Arithmetic Intensity、Roofline Ridge Point、Open-loop Load、Closed-loop Load，定义中注明证据/使用边界。
- `INTERVIEW_BANK.md`、`EXERCISES.md`：本轮不加重复概念题。已有推理系统、KV/bandwidth、压测协议题和 accelerator audit 覆盖相关问题；本章的定量 roofline 表和 demo 直接承担可复算例题。
- 本章导航已存在于第六册目录，无需重复调整。
- **未验证/风险：** 数值模型假设稠密 7B BF16、单卡权重、全 attention 与相同上下文长度；因果 kernel、MoE active weights、cache hierarchy、kernel shape、memory reuse、通信和调度都会改变真实流量及时间。官方工具文档指向 `main`/动态指南，需按运行版本固定 revision 复核。测量例和论文结果都不能替代目标硬件实测。
