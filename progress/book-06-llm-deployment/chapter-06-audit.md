# 第六册第 6 章《批处理与调度》章级审计

审计日期：2026-11-02。状态：完成本章定向扫描、正文精修、配套同步和静态门禁。保留原有章节结构，不新增、删除、拆分或合并章节。章节代码只做静态审阅，未运行；本轮没有运行测试、模型、benchmark 或 PDF。正文和维护记录分别说明代码状态、证据边界及后续复核条件。

## Scope 与章节职责

- **中心问题**：在请求长度、到达时间、服务等级和输出长度不同的情况下，怎样安排每轮 prefill/decode 工作、控制等待、KV 状态和租户服务份额，使 goodput 与 TTFT/TPOT SLO 同时可接受。
- **本章职责**：解释 static/dynamic/continuous batching、chunked prefill、请求队列、token/KV budget、优先级、多租户、长短队列、公平性、抢占和 prefill/decode 分离。使用 demo 与事故案例讲清状态变化、度量和安全边界。
- **章节边界**：第 5 章负责 runtime 选型；本章只核对调度器公开参数，不替代引擎源码解析。第 8 章负责多 GPU、拓扑与跨节点 KV 传输实现；本章用 P/D transfer 解释是否拆池的请求级账本。Agent 程序依赖调度转第 9 章，多模态 deferred prefill 转第 10 章，集群/platform scheduler 归第 23 册，engine 内部实现归第 24 册。
- 简介和目录的第 6 章摘要已加入长短请求隔离与 token 预算措辞，和正文职责一致；章节入口、顺序与文件名不变。

## 章级扫描与候选去向

- **日期、范围与资料类型**：2026-11-02。围绕 LLM serving 的 batch/step token 预算、长短 prompt 队列、P/D 资源分配、优先级方向、KV 压力下 retraction、token-level preemption、Agent program scheduling 和多模态 deferred prefill 定向核验。阅读原始论文/预印本摘要及相关方法/评测段落、USENIX 会议论文页，以及固定 vLLM v0.32.0 与访问日 SGLang 官方参数文档。扫描是本章范围内的定向研究，不声称穷尽调度论文或 runtime。
- **LAPS**：纳入 §6 与 §12.1，用于说明 P/D 拆池后，长、短 prefill 仍会在 prefill pool 内互相干扰；保留双队列、长度感知 batch 和预印本状态边界。论文报告绑定其多轮 trace、Qwen2.5-32B、H200、SGLang 和具体多实例设置，未推广成通用收益。
- **P-PAS**：纳入 §7 与 §12.1，说明 max batched tokens 对低/高调度压力可能有不同 latency 作用，固定预算不必在全部负载区间最优。保留 arXiv 预印本边界，不捏造通用 pressure 分数或参数阈值；正文把队列 prefill token、活动 decode、step budget 饱和与 TPOT 尾部作为本地回放的观测建议，明确它不是论文披露的统一信号。
- **PPD multi-turn disaggregation**：沿用已扫描来源，在 §6 用 append-prefill 与本地 KV 可用性界定“所有 prefill 都应远端处理”的过度泛化；68% Turn 2+ TTFT 等结果只归于论文指定多轮 workload/SLO，不视为 runtime 默认行为。
- **FastServe**：§12.1 使用 NSDI 2026 USENIX 正式论文页报告的 token-level preemption、skip-join MLFQ、GPU/host state offload 与最高 6.1× throughput；说明早期 arXiv 倍率版本不混用，论文数字绑定作者测试环境。
- **vLLM scheduler**：以 v0.32.0 `serve` CLI/API/source reference 描述单 step token、sequence、active admission、API in-flight request、prefill token queue、priority 方向与 KV watermark 等不同预算边界。移除本章相关部分的 `latest/stable` 泛指，并固定到 v0.32.0。
- **SGLang scheduler**：访问日（2026-11-02）官方 Server Arguments 显示默认 FCFS；priority 默认高整数先，可反转；priority preemption threshold、是否允许 priority preemption、KV 满时的 length/priority retraction 是不同控制。动态页面不替代部署 release/revision 核验。
- **Agentix**：不纳入本章完整讲解，路由第 9 章；它把程序依赖与多次 LLM call 放进 program-level scheduler 上下文，而本章中心是单请求和单轮资源调度。
- **Raincheck**：不纳入本章完整讲解，路由第 10 章；它研究多模态 deferred prefill、信息恢复与后续轮次质量/时延的权衡，不等同于一般文本调度。
- 新发现的论文/官方文档知识与来源证据、阅读位置和边界已记录于 `research/inference-serving-metrics-and-disaggregation-2026.md`；本审计只记录候选去向、章节决定和正文修改。

## 篇幅、例子与代码状态

解释性汉字静态计数采用保守口径：排除标题、表格、围栏内容、公式、行内代码、链接文本和状态标记；核心单元下限 1,800，辅助单元下限 1,000。

| 教学单元 | 汉字数 | 类型 | 下限 | 状态/计数理由 |
| --- | ---: | --- | ---: | --- |
| §§1—2 坐标系与服务目标链 | 1,048（不含章首开场段） | 辅助导入 | 1,000 | 通过；连接 batch 口径、SLO、goodput 和等待窗口，不单列为词条 |
| §§3—5 batching、请求生命周期与队列链 | 3,419 | 核心连续链 | 1,800 | 通过；从 static/dynamic/continuous 推进到请求状态、backpressure 与 HOL/分队列 |
| §6 Prefill/Decode 分离 | 1,856 | 核心 | 1,800 | 通过；另有开场计数未加入此数，按当前保守口径达到门槛 |
| §7 Token budget 调度 | 1,993 | 核心 | 1,800 | 通过；同时核算 step tokens、sequence admission、KV block、API queue 与 pressure-adaptive budget |
| §§8—10 优先级、多租户与长短请求争用链 | 2,001 | 核心连续链 | 1,800 | 通过；排序、权重份额、队列隔离与 starvation 相互递进 |
| §§11—15 观测、策略、demo、恢复正确性与高峰事故链 | 3,856 | 核心连续链 | 1,800 | 通过；同一操作闭环：看信号→选策略→走状态→处理失败→复盘 |
| §13 Python demo 周边解释 | 1,055 | 辅助代码教学 | 1,000 | 通过；代码本身不计入篇幅，手算 trace、时间单位、准入差异与能力边界均有说明 |
| §15 多租户故障案例 | 1,164 | 辅助案例 | 1,000 | 通过；作为独立复盘案例也达到下限 |
| §16 资料、§17 小结 | 不单独计数 | 导航/收束 | 不适用 | 不承担独立知识主讲 |

手算案例覆盖：60 秒内 output TPS 与 goodput 分离（100 请求、95/70 个达 SLO）；10 ms dynamic batch 等待（0/1/8 ms 到达）；dense padding 的 608/1,024；704 prompt tokens 与 600-token step budget；Little 定律和 M/M/1 的平均值边界；P/D 阶段 GPU·s、1.6 GB/20 GB/s 的理想 transfer 和 break-even；多轮 1,600-token prompt 的 1,200-token cache hit；1,024-token scheduler step 的 decode/prefill admission、16-token KV block 舍入与 2,000-token queue-cap sizing；priority/tenant debt 手算、6:3:1 服务份额、Gini 限制；demo 中每步活动集、KV token、wait/TTFT/TPOT、租户 output tokens；事故案例的反事实回放变量。

- §13 代码已静态走读并逐步手算 expected output，标为“静态审阅，未运行（2026-11-02）”。`time` 是 iteration，不是毫秒；`tool_call` 仅为普通 toy request，不模拟工具副作用；实现不含真实 GPU、chunked prefill、tenant aging/fairness、抢占、backpressure、物理 KV allocator、实时 TPOT 或 P99 验收。输出是手算预期值，不冒称执行结果。代码无外部依赖，正文已说明输入、输出、运行入口与不覆盖范围。
- 公式状态：逐节检查符号和示例输入；静态检查发现 60 个 fence marker（30 对，其中 27 对为 math fence），行内单美元标记 72 个、成对；`\operatorname`、尖括号下标未命中，math fence 花括号净差为 0。公式数值在正文手算，未用代码输出替代推导。
- 链接状态：本章 30 个 Markdown 链接，均为外链；无本地链接失效问题。关键 vLLM v0.32.0、SGLang、USENIX FastServe/Agentix、LAPS、P-PAS、Raincheck 来源已打开阅读；没有逐一在线检查本章全部外链。
- 视觉表达：保留请求生命周期状态流、预算/策略对照表和 demo trace 输出；关系可由状态流与逐步 trace 直接读取，没有新增 Mermaid，避免把示意政策画成框架实现。

## 配套同步

- 第六册 `简介.md`、`目录.md` 的第 6 章短摘要补明长短请求隔离与 token budget；章名、顺序和阅读路径不变。
- `PAPERS.md` serving scheduling 阅读线加入 LAPS、P-PAS，并保留论文实验条件边界。
- `EXERCISES.md` 新增 80I：在同一 trace 中比较固定 token budget、chunked prefill、长度分队列和压力自适应预算。
- `INTERVIEW_BANK.md` 新增 40ZZZM-B，围绕 P/D pool 内长短请求争用与 pressure-adaptive budget；既有 continuous batching/queue 问题不重抄。
- `GLOSSARY_EN_ZH.md` 添加 Scheduling Pressure、Length-Aware Prefill Scheduling、Adaptive Prefill Budget 三项边界定义。
- `KNOWLEDGE_GRAPH.md` 的 inference scheduling 子链加入调度压力、自适应预算和长度感知 prefill queue。

## 遗留边界与交接

- 本章未执行 demo、章节代码、测试、模型、benchmark 或 PDF；代码仍为静态审阅，未运行。若后续修改其可执行语句、API、输入输出约定或预期结果，按规范只复验受影响代码块；纯文字改动不触发运行。
- 外部资料链接未全量在线复查；SGLang 是 access-date 动态文档，使用前需要固定目标 deployment revision。论文/预印本的性能结果均保留作者 workload、硬件、模型、到达模式与 SLO 边界。
- P/D transfer 示例、队列容量和 budget 例子是明确标记的假设输入，不代表本地测量。P-PAS pressure signals 的本地观测建议由正文注明为本章推论，不归因成论文统一定义。
- 下一步按册内顺序处理第六册第 7 章《量化与模型压缩》，重点检查通用方法与第 15—17 章低精度格式分支的主讲边界；全套书修书 goal 继续 active。
