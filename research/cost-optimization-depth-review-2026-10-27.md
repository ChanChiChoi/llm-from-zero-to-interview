# 第六册第十二章：成本优化资料驱动精修研究记录

- 日期：2026-10-27
- 正文：[第六册第十二章](../book-06-llm-deployment/chapters/12-成本优化.md)
- 依据：当前 [WRITING_SPEC.md](../WRITING_SPEC.md)、[progress_v2.md](../progress_v2.md)、仓库 AGENTS.md。
- 结论：保留原章从账本、GPU、路由、prompt、缓存、解码、量化/蒸馏、调度、配额到 RAG/Agent/多模态、实验案例和 demo 的主线；主要加深资源单位、成本分母、代表方案机制、盈亏线和端到端评测边界，没有用新专题替换旧知识。

## 原知识点覆盖图

| 原章范围 | 保留内容 | 本轮校正或补深 |
| --- | --- | --- |
| §1 成本账本 | 请求链路、模型/辅助模型/平台/可靠性/人工成本，直接与分摊成本 | 以资源向量和货币账本分开不同量纲；明确 API 发票与自托管资源分摊不可重复计费；修正成本/token 的可比 workload 和成功任务成本分母，补单价下降但成功率回落算例。 |
| §2 GPU 与单位经济 | GPU 小时、prefill/decode、显存/KV/并发约束 | 区分墙钟吞吐与忙碌吞吐，避免重复乘利用率；补 $2/GPU-hour、100 tok/s、50% 利用率计算；补 API vs 自托管固定/边际成本交叉点。 |
| §3 模型路由 | 任务能力契约、规则/分类器/级联、fallback、公平安全 | 用可估计的调用路径成本替代含义不清的 routing-error 惩罚；比较 FrugalGPT 级联与 RouteLLM 偏好路由，补成本、串行延迟和领域偏移边界。 |
| §4 Prompt/上下文压缩 | 可压缩对象、结构化裁剪、历史状态摘要、压缩率与质量门槛 | 对比 LLMLingua、LongLLMLingua、LLMLingua-2 的选择机制和目标；用压缩器成本、失败恢复和输入节省推导回本条件。 |
| §5 Cache | prefix、response、embedding/OCR/视觉缓存，一致性、失效和 stampede | 强调最终渲染 token 前缀；区分本地 KV 复用与供应商计费；补 vLLM 不减少 decode、缓存驻留会占 KV 容量；修正 response cache 期望成本并导出命中率回本门槛。 |
| §6 Speculative decoding | draft/target 轨迹、接受率、适用场景和测量项 | 修正期望生成长度遗漏的 target token/bonus token；补 K=4、α=.8 手算、exact rejection sampling 分布条件，并比较独立 draft、MTP、EAGLE、n-gram/suffix。 |
| §7 量化/蒸馏 | 权重/激活/KV、部署评测、生命周期训练成本/fallback | 区分 GPTQ、AWQ、SmoothQuant、KIVI 所压缩张量与误差路径；列 metadata/kernel/质量切片；用 Minitron 的 15B→8B/4B 研究解释重复训练成本的条件边界。 |
| §8 batching/队列/autoscaling | continuous batching、排队预算、worker 冷启动、扩缩容与 draining | 比较 continuous batching、Sarathi-Serve chunked prefill、DistServe P/D 分离；用 M/M/1 例子说明饱和区排队非线性；补 Kubernetes HPA 控制公式与多指标注意事项。 |
| §9 限流/多租户 | QPS/token/KV/GPU/多模态/Agent 预算，租户公平 | 保留原预算单位和治理要求；与 §1 资源向量、§8 自定义扩容指标建立连接。 |
| §10 RAG/Agent/多模态 | rerank/context、Agent step/tool/state、多模态媒体工作量 | 保留三类专项优化；新增 Agent 全历史重复发送的二次增长推导和固定摘要对比。 |
| §11 实验方法 | 假设、固定基线、回放/shadow/canary、成本/性能/质量/安全 | 明确成对 request ID、失败和人工介入统计；把成本目标与质量、SLO、安全硬门槛分开。 |
| §12 设计案例 | 客服请求分层、路由、缓存、RAG、发票和 Agent 方案 | 保留原因果链与任务专属指标，没有把 case 假设写成生产结果。 |
| §13 成本诊断 demo | 账单、缓存、压缩、路由、步骤、延迟、质量与门禁 | 修正 Agent prompt 累计与辅助成本重复乘步数；cache 需要显式 eligibility、版本化 key 和自己的成功率假设；加入每成功任务成本、baseline/candidate P95 和非法 percentile 输入拒绝；删除硬编码安全通过值，改为缺少证据时暂停。 |
| §14 常见误区 | 小模型、压缩、缓存、autoscaling、单一指标误区 | 以受约束优化式表达目标，安全门槛不可被成本收益抵消。 |
| §15–§16 来源/总结 | 官方链接、论文入口、总结主线 | 补入研究对照、证据适用条件及版本提示；结论仍按资源链路与成功任务成本组织。 |

## 代表性研究与官方资料

| 来源 | 解决的问题 / 方法机制 | 报告的证据 | 限制与本章用途 |
| --- | --- | --- | --- |
| [FrugalGPT](https://arxiv.org/abs/2305.05176) | 讨论 prompt adaptation、模型近似和 LLM cascade；级联可让低成本模型先处理容易请求，再按分数/阈值升级。 | 在 HEADLINES、OVERRULING、COQA 等实验设置里报告约 50%—98% 成本节省，结果与特定 2023 API 价格、模型和任务绑定。 | 串行调用增加时延；阈值需用当前任务校准。章节不把论文比例写成目标业务承诺。 |
| [RouteLLM](https://arxiv.org/abs/2406.18665) | 用偏好数据学习选择强模型或弱模型的 router，避免固定级联对所有请求先付弱模型调用。 | 论文报告偏好 router 的成本/质量权衡；仅用 Arena 偏好数据训练时，MMLU 上表现接近随机，加入目标 benchmark 的 gold labels 后改善。 | 偏好数据必须代表线上 query 分布；不是一个 Arena router 可泛化所有领域。正文据此要求目标切片训练/校准和路由审计。 |
| [LLMLingua](https://arxiv.org/abs/2310.05736) | coarse-to-fine token budget、重要性评分和分布对齐，用于减少冗长 prompt。 | 论文在其长 prompt 任务与配置中评估 token 压缩、质量和计算开销。 | 压缩器自己耗算力；token 删除可能损伤否定、条件、数字与出处，需固定任务样本验收。 |
| [LongLLMLingua](https://arxiv.org/abs/2310.06839) | 指出长上下文有高计算成本、性能下降和位置偏置；使用 question-aware 粗/细筛选、文档重排、动态压缩比例和 subsequence recovery。 | 在 NaturalQuestions、LongBench、ZeroSCROLLS、MuSiQue、LooGLE 等论文 benchmark 上评估。 | 适合问题已知、证据可定位的上下文压缩；论文任务不等于业务召回/引用保证。 |
| [LLMLingua-2](https://arxiv.org/abs/2403.12968) | 从 LLM 压缩结果构建蒸馏数据，训练双向编码器进行 token 分类，以提高压缩器推理速度并弱化生成式 scorer。 | 论文描述 MeetingBank 数据构建、in/out-domain benchmark 与 latency evaluation；只用 MeetingBank 训练也在 BBH/GSM8K 展示跨域实验。 | 训练语料影响 token 保留偏好；跨域不等于所有上下文/语言/指令都可靠，仍要验证关键事实保留。 |
| [Speculative Decoding 原论文](https://arxiv.org/abs/2211.17192) | draft model 先提出候选，target 批量计算并按接受/残差抽样修正，减少串行 decode 次数。 | 标准 speculative sampling 可保持 target sampling distribution；论文的加速实验绑定其模型、硬件和实现。 | 实际收益取决 draft 成本、接受率、验证 kernel、KV 更新和调度。若 verifier 有损，需明确区别精确分布保证。 |
| [vLLM speculative decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | 当前文档列 EAGLE、MTP、draft model、n-gram、suffix 等入口。 | 文档将用途定位于中低 QPS 的 memory-bound 工作负载，并提醒真实收益依赖模型家族、流量、硬件和 sampling。页面记录日为 2026-10-27。 | latest 会变；属于当前公开实现说明，不是目标 release 或目标 GPU 实测。页面标 FLy verifier 为 lossy。 |
| [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | 复用相同前缀 KV 跳过重复 prefill。 | 文档说明不降低 decode；长生成或无共享前缀时不带来该性能收益。页面记录日为 2026-10-27。 | KV 驻留会占资源；命中率不等于节省金额或端到端 latency。 |
| [OpenAI Prompt Caching 官方指南](https://developers.openai.com/api/docs/guides/prompt-caching) | 说明托管 API 上精确共享 prompt prefix、计费、最小可缓存长度和保留行为。 | 页面说明缓存单价/阈值依模型和设置变化；例子包括不同最小 prefix 与 cache read/write rate。 | 供应商 API pricing/retention 不能推广成本地 KV cache；版本与合同需当前核对。 |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 近似二阶信息与误差补偿做权重后训练量化。 | 论文给出低 bit 权重量化算法与特定模型/硬件实验。 | 不能从 bit 数推导目标 kernel 提速或所有任务无损；需核对 group、calibration、runtime。 |
| [AWQ](https://arxiv.org/abs/2306.00978) | 激活统计找显著通道，通过缩放保护量化敏感权重。 | 论文评估 weight-only 路线和设备映射。 | activation-aware 不是 activation quantization；runtime/kernel 决定收益。 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 等价通道缩放把 activation 动态范围难度迁移给 weight，启用 W8A8 矩阵乘。 | 论文在其模型、数据和推理实现上测量准确率和速度。 | 与权重专属 GPTQ/AWQ 的目标不同，需要 W8A8 kernel。 |
| [KIVI](https://arxiv.org/abs/2402.02750) | 根据 K/V 统计差异，K per-channel、V per-token；按组量化，最近不完整分组保留全精度，达到 residual length 后并入低比特缓存。 | 论文实验设 `G=32`、`R=128`；保留局部全精度窗口对 GSM8K 等较难生成任务很关键。作者报告的质量、显存和吞吐结果只绑定论文模型/任务/实现。 | 属于 KV representation，不是权重压缩；精度窗口付出额外显存。目标 runtime 支持、分组/窗口配置、长上下文和生成质量都要实测。 |
| [Minitron](https://arxiv.org/abs/2407.14679) | 结构化剪枝与蒸馏，从 Nemotron-4 15B 派生 8B/4B；迭代剪枝可保留更多能力。 | 论文报告派生模型最多少用 40× 训练 token；按 15B/8B/4B 全家族、从头训练同 token 量的假设计算约 1.8× FLOPs 节省。 | 数字取决该模型族、数据和成本模型；不意味着任意基座都能压缩到能力等价。 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | 将长 prefill 切块并与 decode 交错，针对 prefill 阻塞 decode 和吞吐/延迟矛盾。 | 论文在其模型/GPU/负载上报告吞吐与延迟结果。 | chunk size 和调度开销影响 TTFT/TPOT，不能横向裸比 tokens/s。 |
| [DistServe](https://arxiv.org/abs/2401.09670) | 将 prefill 和 decode 放到不同 GPU pool，分别配置其资源以满足 TTFT/TPOT goodput。 | 论文给出在其 workload、硬件、SLO 下的配置和 goodput 结果。 | KV handoff、资源切分与两池负载不均是额外成本；不保证所有负载都优于共置。 |
| [Kubernetes HPA 官方文档](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) | 基于期望/当前指标比值计算 replica 建议，支持多个 metrics。 | 基础公式为 ceil(currentReplicas × currentMetric / desiredMetric)，多个指标建议取最大值。 | 控制器公式不决定合适的 LLM metric；冷启动和采集延迟需要模型服务侧设计。 |
| [A Survey of LLM Inference Systems](https://arxiv.org/abs/2506.21901) | 将请求处理、执行、batch/scheduling、memory/quantization 和多副本系统做 taxonomy。 | 作为 2025 文献地图。 | 综述只帮助定位，不替代原论文或目标 release 核验。 |

## 定量核对

- 成功任务成本示例：旧方案 0.010/0.90=0.011111；新方案 0.007/0.60=0.011667，增加 5%。
- GPU 例子：2/(3,600×100×0.5)=0.00001111 美元/输出 token，即约 11.11 美元/百万 token；若 R 已是墙钟吞吐再乘 50% 利用率，会把成本翻倍。
- API/自托管例子：48,000/(0.0024−0.0008)=30M 请求/月；20M 时 API 48,000、自托管 64,000 美元；40M 时分别 96,000 与 80,000 美元。数值均为教学假设，不含实际波峰/折扣实测。
- Prompt compression：4,000 token × $2/1M=$0.008 毛节省；扣除 $0.0015 compressor 与 $0.002 期望恢复成本，净节省 $0.0045。回本条件为 aΔP>C_comp+ΔC_recovery。
- Response cache：H×0.01>0.0001+0.0002，纯经济命中率阈值为 0.03；不含错误复用的安全和质量硬门槛。
- Speculative decoding：E[L]=1+0.8+0.64+0.512+0.4096=3.3616 token/round；它不是加速比，target/draft 和系统开销需全量测量。
- M/M/1：λ=8、μ=10 时 Wq=0.4s；λ=9.5 时 Wq=1.9s。作为排队直觉，不可拿来估计变长 LLM batch 服务容量。
- HPA：4×1.8=7.2，ceil 后建议 8 副本。
- Agent 历史：P0=1,000、Δ=400、T=6 时 6,000+400×15=12,000 输入 token；每轮固定摘要 1,000 token 是 6,000，尚未计摘要生成和恢复成本。
- Minitron 的 40× 与 1.8×只复述作者模型族成本模型，不作普遍收益推断。

## Demo 运行与口径

- 从 Markdown 提取 §13 唯一 Python fence 后，用标准库 compile 并执行成功。
- 通过 stdout 与相邻 text 输出逐字比较；结果一致。
- 六条等权合成样本：baseline 成本 0.05238，candidate 成本 0.02292，表观节省 56.2%；预期成功数分别 5.52 与 5.17，每预期成功任务成本分别 0.00949 与 0.00443；候选 P95 为 5,088ms，baseline P95 为 11,743ms。
- 门禁结果：成本目标通过、假设成功率门槛通过；P95 5,000ms 门槛失败；安全证据未提供，动作 reduce_long_tail_latency 与 require_safety_evidence，最终暂停调查。
- Agent baseline 每步重发历史，输入序列为 900/1,200/1,500/1,800，共 5,400 token；candidate 为两次固定 700-token 摘要上下文，共 1,400 token。摘要生成成本未纳入独立模型调用，故不是完整生产成本对照。
- 价格、success_prob、cache success、线性延迟均为合成输入；六个样本的 P95 和期望成功数不能解释成生产统计或目标质量证据。

## 配套与验收

- PAPERS.md：新增 FrugalGPT/RouteLLM、LLMLingua 三路线与 Minitron；已有 GPTQ/AWQ/SmoothQuant/KIVI、Serving scheduling、survey 和 HPA 相关材料。
- GLOSSARY_EN_ZH.md：定义 Token Cost、Cost per Token、Cost per Successful Answer、Unit Economics、Marginal Cost、Break-even Request Volume、Cache Hit Rate 的计量口径。
- INTERVIEW_BANK.md：深化第六册推理成本题，加入 FrugalGPT/RouteLLM、压缩/量化/调度路线对比、每成功任务成本与 API/self-host break-even 推导。
- EXERCISES.md：新增九题，覆盖资源账本、成本交叉点、路由、压缩、缓存、speculative decoding、量化、排队/HPA 和 Agent 历史增长。
- 第六册目录已有第十二章入口，无导航改动。
- Python demo 执行成功，实际 stdout 与正文输出块完全一致。公式数值按上述输入手算核对；没有连接真实模型、GPU、目标 API 价格、业务数据、生产 trace 或安全评测。
- 未运行 PDF 构建或外部公式渲染。

## 未解决边界

没有目标 workload、请求分布、真实商业合同、GPU 价格、量化 runtime、压缩器部署、缓存命中 trace 或线上质量/安全证据。因此本章中的成本与性能数值只用于解释计量关系；FrugalGPT、RouteLLM、LongLLMLingua、Minitron、Sarathi-Serve、DistServe 与量化论文结果均保持原有模型、任务、硬件和评估范围。OpenAI、vLLM 与 Kubernetes 动态文档需按目标版本重新核对。
