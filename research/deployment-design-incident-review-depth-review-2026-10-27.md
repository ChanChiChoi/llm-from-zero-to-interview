# 第六册第十四章：部署方案推导与故障复盘深度精修研究记录

- 日期：2026-10-27
- 正文：[第六册第十四章](../book-06-llm-deployment/chapters/14-部署方案推导与故障复盘.md)
- 依据：[WRITING_SPEC.md](../WRITING_SPEC.md)、[progress_v2.md](../progress_v2.md) 和仓库 `AGENTS.md`。
- 范围：复读正文 §1—§16，检查部署资源推导、KV 分配、量化、服务生命周期、RAG/工具安全、成本、事故复盘和 Python 设计审计。旧主题均保留；章节没有因为增加研究专题而删掉原问题。
- 证据标签：`paper`=原始研究；`official`=官方文档/标准；`local-demo`=本地可运行教学代码；`synthetic`=人为构造的案例；`inference`=基于来源机制提出、尚待实验验证的判断。

## 本轮核心问题图谱与旧知识点去向

| 正文范围 | 原有知识点 | 精修后的解释与证据 |
| --- | --- | --- |
| §1—§2 | 70B 权重、并行、量化、QPS 与副本数 | 保留名词到架构推理、显存账本和并行选择；把 prefill、decode、活跃模型序列三种容量拆开，增加逐项可复算的副本示例和 N+1 失败域检查。 |
| §3—§5 | TTFT、continuous batching、吞吐、KV Cache 与回收 | 保留阶段分解、调度和 KV 公式；诊断按 trace span 与工作负载切片进行，不将 SLO、吞吐、质量和成本混为一个指标。 |
| §6 | PagedAttention、KV 动态分配、碎片 | 保留 block table、动态 block 与 serving 取舍；补 vAttention 的 CUDA VMM、连续虚拟 KV 布局、实现依赖和论文适用范围，并提出按瓶颈对比的实验路线。 |
| §7 | INT8/INT4、权重/激活/KV 量化、质量评估 | 保留原量化对象；增加 GPTQ、AWQ、SmoothQuant、KIVI 的问题、对象、机制、收益证据和限制，强调不能用权重压缩数字推断 KV 或 activation 收益。 |
| §8—§11 | 高并发请求、RAG、工具权限、成本 | 保留请求路径、版本 manifest、检索证据、prompt injection、工具校验和成功任务成本；把模型池 active sequence 与端到端 job 并发的区别同步到容量设计。 |
| §12 | 跨组件事故诊断清单 | 将泛化步骤扩成一份明确标注为合成的 prompt/cache/prefill/queue 案例：含时间线、用户影响、机制链、可推翻假设、系统促成因素和可验收行动项；不同于真实生产事故记录。 |
| §13—§16 | 部署设计 demo、常见误判、资料、总结 | 保留 toy 设计清单和综合小结；demo 增活跃序列容量和三维副本计算，显式限制 token-only 成本与声明型 safety signals 的意义。 |

## 代表方案：针对的问题、证据与后续改进

| 方案 / 来源 | 解决什么问题及机制 | 来源报告的证据 | 代价、边界与后续改进 |
| --- | --- | --- | --- |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 长短请求动态生成时，按最大长度连续预留导致浪费和碎片；KV 切成固定 token block，通过逻辑到物理映射按需分配，可共享前缀。 | 原论文系统评测在其模型/负载与相近延迟条件下报告相对 FasterTransformer/Orca 2–4× 吞吐。 | 这是系统级结果，不能单独归因于 block table；kernel 必须支持分页布局，block 粒度还在尾部浪费、映射和管理成本间折中。后续实验应固定 kernel、block size 和到达/长度分布，并记录 KV 利用率、P99、抢占和 OOM。 |
| [vAttention](https://arxiv.org/abs/2405.04437) | 希望按需申请物理页，同时让 attention kernel 继续看到连续虚拟 KV；通过 CUDA VMM 分离 virtual address reservation 和 physical page mapping。 | arXiv 摘要报告在指定 FlashAttention/FlashInfer PagedAttention kernel 对照中的吞吐收益最高 1.23×；论文完整评测还包含其他模型/场景和指标。 | 依赖 CUDA VMM、驱动/runtime 和映射粒度；结果不能和 PagedAttention 原论文的吞吐倍率横向排序。若 block-table kernel 改造是瓶颈，可把 VMM 作为候选；如果映射开销或环境不兼容，应保留已验证路径。该后续方案是场景化推理，不是普遍胜出结论。 |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 权重逐步舍入误差会沿层传播；用校准激活估计近似二阶信息，量化后把误差补偿回剩余权重。 | 论文在 OPT-175B WikiText-2 条件下报告 4-bit GPTQ 的困惑度增加约 0.03，而 RTN 约 2.2；作者报告 175B 量化用时约 4 GPU-hours。 | 校准分布、group/scale、精度格式和 kernel 都影响质量/速度；论文数字不代表现代目标模型。质量掉点时，先做校准域和分组/格式的配对消融，再单独检查 kernel 路径，避免把算法误差与执行回退混为一谈。 |
| [AWQ](https://arxiv.org/abs/2306.00978) | weight-only 低比特量化中，并非所有权重对输出同样重要；从激活统计识别显著通道，用保持浮点等价的缩放降低重要通道的权重量化误差，不需反传或重建优化。 | 论文将 activation-aware scaling 与 TinyChat 的 kernel fusion/platform-aware packing 分层报告；TinyChat 的速度结果不能只归因于 AWQ 算法。 | 激活统计覆盖不足时，显著通道选择可能不适配线上分布。下一步可补目标域校准流量并测通道/任务切片；量化质量与端到端速度分别报告。 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | LLM activation outlier 难以直接 INT8；利用 `Y=XW=(XD^{-1})(DW)` 的等价重参数化，把一部分量化难度从激活迁移到权重，支持 W8A8 矩阵乘。 | 论文报告其特定模型/硬件/实现设置中最高约 1.56× 延迟收益和约 2× 显存节省；该数值属于作者设置。 | 它针对 W8A8 路径，不等于 INT4 weight-only；alpha、校准和 fused INT8 kernel 决定实际收益。后续应做 alpha 消融并确认实际 kernel 命中，不能只比较理论 bit 数。 |
| [KIVI](https://arxiv.org/abs/2402.02750) | KV Cache 体积随活跃序列和上下文增长；作者在所测模型中观察 K 的 channel outlier，提出 K per-channel、V per-token 的 2-bit 非对称量化，并以 residual window 处理新 token 流入。 | 论文给出分组/残差算法和模型、任务、cache-budget 条件下的质量与容量结果；此前量化与 KV 专题记录的 Llama-2-7B/A100/ShareGPT 系统评测报告吞吐提升部分来自可容纳更大 batch。 | 不能说所有模型均有相同 outlier，也不能把 2-bit 算法等同于任意 runtime 的 FP8 接口。后续改变 group/residual 时要同时测长上下文质量、有效字节、反量化 kernel、并发和延迟。 |

以上方法改的是不同层：PagedAttention/vAttention 管地址映射；GPTQ/AWQ 管权重数值表示；SmoothQuant 改 W/A 数值分布以服务 W8A8；KIVI 压缩请求期 KV 状态。指标与方法层级不匹配时不能排名。详细量化数值与公式的既有一手研究见[量化研究记录](quantization-model-compression-depth-review-2026-10-26.md)；分页、KIVI 算法和 runtime 证据见[KV Cache 研究记录](kv-cache-memory-management-depth-review-2026-10-26.md)及[推理引擎研究记录](inference-engine-depth-review-2026-10-26.md)。

## 事故复盘来源与合成案例的因果审计

| 来源 | 支持的判断 | 不支持的判断 |
| --- | --- | --- |
| [Google SRE: Postmortem Culture](https://sre.google/sre-book/postmortem-culture/) | 复盘要记录影响、缓解/解决行动、促成原因和后续行动；重大事件复盘应经审阅，重点是减少复发概率/影响，避免把复盘当惩罚。 | 不能证明本章合成 prompt 事故曾在 Google 或任何实际服务中发生，也不能提供本章的延迟值。 |
| [Google SRE Workbook: Incident Response](https://sre.google/workbook/incident-response/) | 事故响应需明确事件角色、时间线和后续复盘；可作为从现场响应到行动项的工程流程参考。 | 组织流程案例不等于 AI serving 的具体根因或性能基准。 |
| [NIST SP 800-61 Rev. 3](https://csrc.nist.gov/pubs/sp/800/61/r3/final) | NIST 于 2025-04 发布 Rev. 3，取代 Rev. 2；建议把网络安全事件响应考虑纳入 CSF 2.0 风险管理活动，覆盖准备、检测、响应和恢复。 | 网络安全事件标准不替代在线性能事故的 SLI、模型回滚或质量评测；正文仅用它说明凭证/数据外泄事件要另走安全处置流程。 |

合成事故假设：每请求变化的租户字段被移动到稳定 prompt/tool 前缀之前，v42 canary 的前缀 token 复用率从 42% 降到 14%；TTFT P95 从 1.1s 升到 2.6s、P99 到 5.4s；queue wait 从 150ms 升至 950ms，TPOT 与 5xx 近似稳定。回滚到 v41 后复用率约 41%、TTFT P95 约 1.2s。数字均为教学构造，未从厂商事故或线上 telemetry 抄取。

因果链是一个待检验解释：动态字段前置 → 公共 token 前缀缩短 → cache saved tokens 下降 → 重复 prefill 和 KV 占用时间增长 → 长 RAG 干扰共享池 → 排队和短请求 TTFT 尾延迟变差。支持证据包括按版本 token-prefix/hash 对比、cache token reuse、prefill/queue span 与 TPOT 稳定；反证包括 cache reuse 无变化、未命中请求无 prefill 变化、GPU 水位/路由/检索长度先于模板变化恶化等。复盘不能以时间相关性代替因果验证。

行动项的验收设计是本案例的重要结论：prompt 拼接契约有 token-level regression；canary 面板按有限的 release/workload/cache cohort 展示 P95/P99、prefill 和 queue，不把 prompt/user ID 当 metric label；短/长请求混合回放评估分池、chunked prefill 和 fairness；发布清单包含模型、prompt、索引、工具 schema 与策略并做回滚演练。每项由 owner 提交复现/回放结果后关闭，创建 ticket 不代表防线已验证。

## 容量公式与手算

1. 对任务类别 `j` 到模型池 `m` 的速率 `λ_j`、模型调用次数 `s_{j,m}` 和 runtime 占槽时长 `τ_{model,j,m}`：
   `A_m≈Σ_j λ_j s_{j,m} τ_{model,j,m}`；端到端任务并发 `J_task≈Σ_j λ_j L_task,j`。
   `L_task` 含 RAG/tool/human 等等待，因此不能替代 runtime 序列占槽时间。
2. 演示输入 token：`60×1500×(1−0.25)=67,500 token/s`；输出 token：`60×200=12,000 token/s`；活跃序列：`60×1.2=72`。
3. 单副本比例为 `67,500/80,000=0.84375`、`12,000/2,500=4.8`、`72/12=6`。`γ=1.2` 是工作负载波动/运营 headroom，不包含再次计入的单副本故障余量；`ceil(7.2)=8` 个容量副本。若要求失去一副本后仍保留这份 headroom，总池需 9 个；代码逐项检查 8 个 survivors 的 prefill `640,000≥81,000`、decode `20,000≥14,400`、active sequence `96≥86.4`。只要求故障后满足原需求时，应按原始需求独立检查 survivors，不能把 `+1` 当普遍公式。
4. 70B FP16/INT4 的理论权重下界为 `70e9×16/8/2^30=130.39 GiB` 与 `70e9×4/8/2^30=32.6 GiB`；KV toy 示例为 20 GiB。它们不含完整 allocator/runtime 预算。
5. Token 费率 demo：`60 QPS × 3600 × ((1500/1000)×0.003 + (200/1000)×0.010) = $1404/hour`；假设 25% 输入 token 不产生相同边际费用得 `$1161/hour`、约 17.3% 变化。是假定的 token-level 账单，不包含 GPU 固定成本、缓存、重试、fallback、网络或人工，也不代表各 API 缓存价格。

## 代码、资料和配套文件验收

- `code_status`：提取并运行 §13 标准库 Python 示例；exit code 0、stderr 空。输出容量为 72 active sequence、8 base replicas、9 N+1 replicas；模拟移除一个副本后逐项检查 prefill/decode/sequence 三项均覆盖 `γ=1.2` 需求。Token-only baseline/cached cost 为 1404.0/1161.0，诊断清单缺 `throughput.kv_budget`。安全输出字段改为 `safety_topics_are_declared`，它只检查清单中是否写了安全主题；算术容灾检查不建模故障相关性、路由失衡、冷启动或容量非线性。
- `formula_status`：手算 3 种容量比例、`ceil(1.2×6)=8`、失一副本仍保留 8 的要求、70B FP16/INT4 权重 GiB、KV GiB 和每小时 token 费用；全都按单位与分母解释。未使用公式渲染器。
- `source_status`：查阅 GPTQ/AWQ/SmoothQuant/KIVI 原论文入口和已有专章的细读记录；核对 vAttention 原论文摘要、Google SRE 复盘文化/Workbook 事故响应页面、NIST SP 800-61 Rev.3 官方页面。厂商动态 serving 文档在目标 release 上仍需复核。
- `sync_status`：`PAPERS.md` 增事故复盘与 NIST 入口；`GLOSSARY_EN_ZH.md` 细化 active sequence 定义并新增 runtime occupancy、N+1、prefix token reuse、blameless postmortem；`INTERVIEW_BANK.md` 增容量推导、量化方案定位和证据型复盘题；`EXERCISES.md` 新增 5 项本章练习。第六册目录已包含本章，无需更新。
- `link_status`：本章目录映射已存在；本轮新增 research note、exercise、interview 和 paper source 的相对目标应在本地验证。未对全章旧外链执行 HTTP 全扫。
- `validation`：复核章节 §1—§16 连续、Python/数学/text fence 配对、demo 输出与 fenced `text` 逐字相同、chapter local links 和 `git diff --check`；未构建 PDF 或运行目标硬件基准。

## 未解决边界

本轮没有目标模型、GPU、线上 token trace、cache hit trace、真实成本表或生产事故资料。容量输入、服务吞吐和 v42 事故数值都是教学/假设数据，不能用于真实资源承诺或断言 prefix cache 是任何现网事故的原因。原论文速度、质量和显存结果绑定各自的模型、硬件、kernel、任务和测量口径；vLLM/TensorRT-LLM 等运行时支持需固定版本重验。OpenAI/供应商 token cache 的计费方式可能单独定价，demo 的 25% 节省假设不能代替真实账单。
