# 批处理与调度章节资料驱动精修记录（2026-10-26）

## 状态与研究问题

- `status`：资料驱动精修与声明范围内验收完成。
- `chapter`：`book-06-llm-deployment/chapters/06-批处理与调度.md`，§1—§17。
- `basis`：当前 `WRITING_SPEC.md` 的研究支撑、方案对照、公式手算、教学代码边界、原知识点保留和配套同步要求。
- `date_boundary`：论文及官方动态文档按 2026-10-26 可查内容记录；stable/latest/main 页面不能替代目标部署 release。

本章研究围绕一个核心问题：LLM serving 的 scheduler 到底在什么时间边界上重新选择请求，又怎样同时决定计算工作、KV 状态、队列顺序、租户份额和失败恢复？不能把 iteration scheduling、token-level preemption、chunked prefill 与 prefill/decode 分离说成四个“更大 batch”的同义词，它们介入的是不同的系统层次。

## 代表研究解决了什么

| 研究 | 原问题与核心方案 | 作者报告的结果 | 代价与证据边界 |
| --- | --- | --- | --- |
| Orca（OSDI 2022） | 生成请求在静态 batch 中运行到完成，短请求结束后留下 batch hole，新请求等整批结束。Orca 在每次模型 iteration 重排请求，并提出 selective batching。 | GPT-3 175B 特定设置下，相对 FasterTransformer 在相同 latency 报告 36.9× throughput。 | 结果是论文时点模型/系统对照；迭代级重排并不自动处理 KV 碎片、跨租户公平和多阶段池配置。 |
| FastServe（NSDI 2026 正式版） | 长生成任务导致 head-of-line blocking。允许 token 边界抢占，用 skip-join MLFQ 根据可见输入长度分配初始队列，并主动在 GPU 与 host 间 offload/upload 中间状态。 | [USENIX NSDI 2026 正式论文页](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)报告相对 vLLM 最多 6.1× throughput。 | 抢占需保存/恢复 token position、KV 与采样状态；offload/upload 消耗主机内存和链路带宽。该结果绑定论文 workload、系统和约束。[arXiv v3](https://arxiv.org/abs/2305.05920)早期摘要报告 31.4×/17.9×，与正式版数字不可混用；本章只使用正式发表数字。 |
| Sarathi-Serve（2024） | 长 prefill 一次插入会让 decode 等待；以 chunked prefill 和 stall-free schedule 在每轮预算内交错新请求与已有 decode。 | Mistral-7B 单 A100 报告 2.6× serving capacity；Yi-34B 双 A100 最高 3.7× vLLM；Falcon-180B pipeline 场景最高 5.6×。 | Chunk size、SLO、模型、GPU 数和 pipeline 改变结果；它减少同池的阶段干扰，不等同于 token 级抢占或物理资源池分离。 |
| DistServe（OSDI 2024） | Prefill 与 decode 共置造成干扰，且两阶段 GPU 资源/并行配置被绑定；将两阶段放入不同 GPU 池，按 TTFT/TPOT 约束共优化资源、并行方案和 KV/state 传输位置。 | 作者报告最多 7.4×请求量或 12.6×更紧 SLO，且超过 90% 请求在 latency 约束内。 | 需要跨池 state transfer、路由、网络 placement 和两池容量协调；流量不均时会出现一边排队、一边空闲。数字只对应论文 workload、硬件和约束。 |
| PPD Disaggregation（2026） | 多轮对话中，不是每次都需要重算完整历史；后续 append-prefill 可复用已有 KV。论文提出按 SLO 动态决定是否把后续轮次的 append-prefill 路由到 decode 节点本地处理，减少反复传 KV。 | arXiv/ICML 2026 论文报告 Turn 2+ TTFT 约下降 68%，TPOT 保持有竞争力。 | 是多轮 serving 的近期特定方案；依赖 KV 命中、跨节点带宽、负载和 SLO，不意味着固定一律本地执行或当前 runtime 已默认支持。 |

因此研究路线不是单一时间序列的替代品：Orca 改变请求重排边界；FastServe允许长任务暂停并恢复；Sarathi 调整每轮 prefill 的执行粒度；DistServe 改变计算资源池结构；PPD 针对多轮缓存复用改变请求路由。组合可能合理，但需验证状态契约、KV 迁移、优先级、公平性和端到端 goodput。

## 队列与运行时资料核对

| 来源 | 本轮确认的接口/机制 | 在正文中的解释 |
| --- | --- | --- |
| [vLLM scheduler API](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/) | `max_num_batched_tokens` 为单 iteration token 预算；`max_num_seqs` 为每 iteration sequence 数；`max_num_queued_reqs` 为等待/运行中的请求准入上限，可在 API server 拒绝超限请求；FCFS 为默认策略，priority 模式较小整数先、同值按到达时间。 | 把 step 计算预算、batch 形状和队列准入分开，防止把数值相同误认为语义相同。Dynamic docs 需锁定 release。 |
| [vLLM chunked prefill tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) | 文档说明 V1 可把长 prefill 切块并与 decode 放入预算；较小预算倾向降低 ITL、较大预算倾向改善 TTFT，实际需在目标 workload 调参。 | 用来解释 token budget 如何表达 prefill/decode 取舍；官方建议和示例不是无条件最优值。 |
| [SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) | 默认 `fcfs`；priority 模式默认较大数值先，可反转方向；另有抢占阈值、KV 满时按长度/priority retraction、`max-running-requests` 和 chunked-prefill 参数。 | 显示 priority 数值方向与 vLLM 不同，且队列排序、抢占和 KV retraction 是不同决策。 |
| [TensorRT-LLM PyTorch architecture](https://nvidia.github.io/TensorRT-LLM/latest/torch/arch_overview.html) | 当前 `PyExecutor` 包含 scheduler、model engine、decoder 等组件；scheduler 决定资源分配与本 step 请求，model engine 执行单步 forward。 | §16 更新到当前 PyTorch backend 架构；仍可访问的旧 TensorRT Build Workflow 明确标为 legacy，不作为当前执行流程。 |
| [TGI official documentation](https://huggingface.co/docs/text-generation-inference/en/index) | 文档保留 serving 能力说明并标注 maintenance mode。 | 仅用于维护状态和迁移判断，不把项目历史能力说成当前未来路线。 |

多租户 fairness 的 Gini 公式只描述一组非负同单位份额的离散程度。两租户 `80:20` 时 G=0.3，即便这是 4:1 的目标配额；公平比较因此需把实际服务量除以目标 entitlement，并一起看 SLO、最长等待与拒绝情况。该解释属于数学定义与教学手算，不构成某框架的 SLA 保证。

## 原章节覆盖图

| 原节 | 保留并深化的知识点 |
| --- | --- |
| §1—§3 | 保留 LLM 的 prefill/decode 差异、batch 三种计数单位、static/dynamic/continuous batching 与 chunked prefill；补清 continuous batching 不是公平性或 TTFT/TPOT 保证。 |
| §4—§5 | 保留请求生命周期、队列信息、Little 定律、backpressure、retry storm 和 head-of-line blocking；新增 M/M/1 直觉算例并明确其不是 LLM 容量模型。 |
| §6 | 保留 prefill/decode 的资源画像和分离利弊；补 DistServe 的资源共优化证据，以及多轮 append-prefill/PPD 的近期扩展。 |
| §7 | 保留 prefill token、active sequences、逻辑/物理 KV 三本账；映射 vLLM 当前 token/sequence/queue 配置层级。 |
| §8—§10 | 保留优先级、租户隔离、长短请求竞争和策略取舍；增加 vLLM 与 SGLang priority 整数方向、抢占、retraction 语义差异。 |
| §11—§12 | 保留调度指标、Gini 与 FIFO/priority/aging/token/length/deadline/fair share；补 Gini 目标权重反例和 Orca/FastServe/Sarathi/DistServe 的方案对照。 |
| §13 | 保留零依赖 continuous batching demo；改善 KV admission reservation，加入正限额/唯一 ID/单轮 prefill fitting/每请求 KV 上界验证，并明确其不实现 chunking。 |
| §14—§15 | 保留抢占、取消、重试、副作用边界和多租户故障复盘；公式的 `\eta` 写法更正，结合完整 state 与 idempotency 解释恢复正确性。 |
| §16—§17 | 更新来源清单并保留章节总结；历史 TensorRT build workflow 被注明为旧 backend，而不是删除为“没有发生过”。 |

本轮没有删除任何原有主线。重要旧内容继续作充分解释；研究和版本补充分别进入机制段、方法对照表、参考资料或风险边界。

## 算式、代码与验证

- Little 定律 `L=λW` 保持原义。为解释容量接近时延迟曲线变陡，增加 M/M/1 教学例：`μ=10 req/s` 时，`λ=5` 得 `W=1/(10-5)=0.2 s, L=λW=1`；`λ=9` 得 `W=1 s, L=9`。这不是 LLM serving 预测，真实工作负载要用异构请求 trace 和目标系统测量。
- Gini `G=Σ_iΣ_j|x_i-x_j|/(2nΣ_i x_i)` 对 `(80,20)` 算得 0.3、对 `(50,50)` 为 0；章节补充了按 4:1 权重时 raw Gini 不等于公平性判断。
- 从 Markdown 提取并运行 §13 标准库 demo。标准输出与正文 fenced `text` 完全一致；把 prefill token budget 改为 1,000 后，长请求由新增验证正确拒绝，而不再无限留在 waiting。运行只验证 toy 状态机和逻辑 token reservation。
- 静态检查 §1—§17 顺序、代码/文本/数学 fence 配对、math 符号和表格、相对链接。外部研究与官方页面已按原论文/primary docs核对；`git diff --check` 在最终修改后执行。

## 配套文件与开放边界

- `PAPERS.md` 补 FastServe 正式版、PPD 多轮调度和 vLLM/SGLang 最新调度接口；保留 Orca、Sarathi、DistServe 的来源线。
- `GLOSSARY_EN_ZH.md` 补 skip-join MLFQ、token-level preemption、P/D disaggregation、append-prefill、M/M/1 queue。
- `INTERVIEW_BANK.md` 增 FastServe/Sarathi/DistServe 的问题—机制—代价对照；`EXERCISES.md` 增代表路线 falsifiable benchmark 和 Little/M/M/1 数值题。
- 第六册目录原有第六章入口有效。
- 没有运行真实 vLLM/SGLang/TensorRT-LLM、GPU、集群、模型或生产 workload；未测真实调度 fairness、抢占性能、state transfer、queue stability、PPD/PD 容量或目标 SLO。论文数据绑定作者协议和系统；官方文档只证明当前公开参数/机制描述，不能替代目标 workload replay。
