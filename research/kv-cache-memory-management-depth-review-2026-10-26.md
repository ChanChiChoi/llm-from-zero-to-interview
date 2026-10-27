# KV Cache 与内存管理章节：资料驱动精修记录

- `date`：2026-10-26
- `scope`：按 `WRITING_SPEC.md` 复读 `book-06-llm-deployment/chapters/03-kv-cache与内存管理.md` §3.1—§3.17，重点核查分页分配、前缀复用、量化、淘汰和显存排障。目标是补齐代表方法的输入/状态/操作/结果/边界，不把原章节改成只有几个新论文专题。
- `evidence_level`：`paper`、`official`、`local-demo`、`inference`
- `lifecycle`：`active`

## 本轮筛选出的核心问题

1. KV 的 token 位置如何经 block table 映射到物理槽位？共享的未满尾块怎样安全写入？
2. PagedAttention 与 vAttention 分别把动态映射放在哪一层，收益和系统依赖是什么？
3. FP8/INT8 的容量比例为什么只是数据面理想值？KIVI 怎样从 K/V 的误差路径推导出不同量化轴，并处理逐 token 流入？
4. 一个 prefix hit 从 hash lookup、active 引用、suffix prefill 到 reusable block 淘汰经历哪些状态？不同 runtime 的数据结构与回收策略有什么差别？
5. H₂O 和 StreamingLLM 为什么不会保留同一批历史 token？有损 eviction 与无损 offload、低精度量化、重算的边界在哪里？
6. 缓存共享如何形成多租户时序侧信道，cache salt 能解决什么、不能证明什么？

原稿已经覆盖了 KV 公式、MHA/MQA/GQA、MLA/窗口/递归状态、总显存账本、block 分配、量化评估、prefix key、active/reusable 状态、长上下文策略、OOM、多租户示例和 demo。本轮没有删掉这些主线；新增研究用于把机制和方法差异落到具体轨迹及测试条件。

## 资料阅读与证据边界

| 来源 | 本轮阅读内容及支持的结论 | 不能由该来源推出 |
| --- | --- | --- |
| [PagedAttention 原论文](https://arxiv.org/abs/2309.06180) | 阅读论文摘要、方法与系统实验相关段落。支持 KV block、logical/physical mapping、按需分配、引用计数/共享和 copy-on-write 的方法说明；作者报告 vLLM 在其 workload 上相近延迟时相对 FasterTransformer/Orca 有 2–4 倍吞吐。 | 不能把系统级吞吐归因于 PagedAttention 一个组件，也不能把论文数字外推到当前 GPU、kernel、模型、SLO 或不同 batch/workload。 |
| [vAttention 原论文](https://arxiv.org/abs/2405.04437) | 阅读架构、CUDA VMM 机制和评估章节。它保留 KV 的连续虚拟地址、按需映射物理页，以便复用连续布局 kernel；作者专门处理 CUDA 2MB 原始粒度和映射调用延迟。论文使用 Yi-6B、Llama-3-8B、Yi-34B 与 1–2 张 A100，报告特定 kernel/对照下最多 1.97 倍吞吐和 1.45 倍 TTFT speedup。 | 不能将 vAttention、PagedAttention 两篇论文各自的 `up to` 数字拼成同条件排名；论文结果不证明其他驱动、GPU 或 runtime 同样支持/获益。 |
| [KIVI 原论文](https://arxiv.org/abs/2402.02750) | 阅读 K/V 分布观察、量化轴分析、streaming algorithm 与实验。作者在所测模型中观察到 K 的固定 channel outlier，提出 K per-channel、V per-token 的 2-bit 方案；K 分组量化、近期 residual 保持高精度，V 把超出 residual 的旧 token 逐 token 量化。实验 group size 32、residual length 128。系统实验中的 Llama-2-7B、A100 80GB、ShareGPT 合成请求平均输入/输出约 161/338 token，batch 增至 OOM；作者报告相近峰值显存下最高 4 倍 batch、2.35–3.47 倍吞吐。 | “K 总有固定 channel outlier”、2-bit 必然保质或 8 倍压缩不是普遍结论；作者吞吐包含可容纳更大 batch 的容量收益，不是单请求 kernel 的倍率。 |
| [H₂O 原论文](https://arxiv.org/abs/2306.14048) | 阅读 heavy-hitter 观察与 eviction 方法。它用累计 attention score 识别 heavy hitters，并与近期 token 一起保留；论文把该选择与动态子模问题联系，在 OPT 等模型和指定任务/cache budget 下评测。 | attention 分数只是重要性代理；不能保证任务相关证据都会保留，也不能用其 benchmark 证明任意模型上的相同质量。 |
| [StreamingLLM 原论文](https://arxiv.org/abs/2309.17453) | 阅读摘要与引言：仅保留最近窗口会在超过 cache 长度时失败；作者观察初始 token 的 attention-sink 现象，采用 sink + recent window 稳定流式生成。其结果绑定论文模型及语言建模/流式设置。 | sink 稳定性不等于能随机检索任意中间文档内容，不等于通用长上下文问答质量保证。 |
| [KV cache 系统感知综述](https://arxiv.org/abs/2607.08057) | 阅读 taxonomy/摘要，将调度、跨设备放置与迁移、KV 表示压缩、保留/淘汰分开，用作检索地图。 | 综述分类不能替代代表原论文的算法、实验和边界核验；章节未照搬全部 taxonomy。 |
| [vLLM v1 Prefix Caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/) | 核对 block hash = parent hash + block tokens + extra hashes；多模态 hash、full-block cache、`cache_salt`、ref count、free queue、allocation/free/LRU 生命周期。此页为动态 `latest`，记录日为 2026-10-26。 | 文档描述的是当前公开 vLLM v1 设计，不代表其他 runtime；salt 限制 cache reuse namespace，不证明数据加密或全部旁路都消失。 |
| [vLLM FP8 KV cache 文档 v0.25.0](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/) | 核对 per-tensor / per-attention-head scale、校准和 backend 支持边界。 | 不能由 FP8 runtime contract 推断 KIVI 2-bit per-channel/per-token 算法已被支持。 |
| [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) | 核对 radix tree reuse、priority/LRU、secondary-memory offload 和 leaf-only eviction 限制。动态官网文档按记录日访问。 | 不能把文档当前接口推广到所有 manager/model/release，也没有本地 TensorRT 系统性能验证。 |
| [NDSS 2025 Prompt Leakage 论文页](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/) | 论文在特定多租户 serving 系统展示 cache sharing 可形成时序侧信道，并研究 prompt reconstruction。 | 不代表所有 cache 实现都存在相同攻击面；具体威胁取决于观测能力、命中策略、隔离配置和速率限制。 |

## 代表方案对照

| 方法 | 所解决的问题 | 机制与收益 | 主要限制/后续改进方向 |
| --- | --- | --- | --- |
| PagedAttention | 动态长短请求让连续最大长度预留浪费显存、难以共享 | block table 把逻辑连续序列映射到可分散的物理 block；按需分配/归还，支持 prefix 或分支共享；物理尾浪费受 block 粒度限制 | kernel 要能处理 block table；block size 会影响尾浪费、映射和 kernel 成本。比较 vAttention 时须同 workload 测量，不能把 2–4x 系统结果当成 kernel 单因果。 |
| vAttention | 希望动态申请物理页，同时保留连续虚拟 KV 布局以复用已有 attention kernel | CUDA VMM 分开预留 virtual range 和按需映射 physical pages；作者对小页粒度和映射延迟做优化 | 依赖 CUDA VMM 和驱动行为；粒度、映射调用、回收/重映射和部署可移植性仍要核验。论文证据不应跨硬件外推。 |
| KIVI | KV 读取/占用成为瓶颈，朴素统一量化会遇到 outlier 和误差累积 | K per-channel，V per-token；分组量化并保留高精度 residual；读取时融合反量化；可释放容量支撑更大 batch | 量化尺度、residual、metadata、kernel 和质量集都会影响实际收益；论文的低比特算法不等同于 runtime FP8 参数。 |
| H₂O | 纯 recent window 可能丢掉被反复关注的远处 token | 用累计 attention 的 heavy hitters 加 recent token，在指定预算保留非局部历史 | 注意力重要性不等于下游任务重要性；需要中段检索/指令/代码/数学切片。 |
| StreamingLLM | 长期流式生成的窗口截断导致模型状态不稳定 | 保留 attention sink 的初始 token 与最近窗口 | 主要目标是流式稳定，不保留任意中间证据；不能替代完整 cache 做任意长文档检索。 |
| vLLM v1 / TensorRT-LLM reuse | 重复 prompt prefill、跨请求保留和回收 | vLLM 公开设计使用哈希链、full blocks、ref count/free queue；TensorRT-LLM 文档描述 radix tree、priority LRU 与 host secondary tier | cache key 命名、active pin、跨租户 salt、offload 延迟和有限 attention 对淘汰的影响须绑定具体 runtime 版本验证。 |

这些方法分别优化地址分配、cache 表示、历史保留和跨请求复用。eviction 会删除可见历史，量化会改变数值精度，offload 保存完整状态但移动资源层级，recompute 用计算换存储；“压缩率”不能替代各自的质量、延迟和容量指标。

## 原知识点覆盖图

| 原位置 | 原有知识点 | 修订后的去向 |
| --- | --- | --- |
| §3.1–§3.3 | 为什么缓存 K/V、不是 logits、状态生命周期、标准 shape/字节、变长长度和 8k 算例、prefill/decode/位置状态 | 原主线保留，没有删减；精修只在分页/复用处补系统实例。 |
| §3.4–§3.5 | cache 省掉的投影与未省掉的历史读取、MHA/MQA/GQA 公式与模型取舍 | 原公式、例子和边界保留；不把 block 分配误写成减少 attention 读取。 |
| §3.6 | MLA、滑动窗口、递归 state 及混合层状态公式 | 原知识点保留；研究笔记把它们列为不同 state layout，未统一套 MHA/GQA 账本。 |
| §3.7 | GPU 总显存账、静态最大长度浪费、per-rank 分片条件 | 原账本与示例保留；新 KV methods 仍需计 metadata/scale/workspace/headroom。 |
| §3.8.1–§3.8.2 | 连续分配问题、逻辑/物理 block、尾浪费公式、block size tradeoff | 保留并原地加深；新增逻辑 token → logical block/offset → physical block → slot 地址换算。 |
| §3.8.3–§3.8.4（原稿） | 前缀 block 共享、引用计数/COW、key 和 PagedAttention 边界 | 保留原共享机制；新增 partial-tail fork 的写时复制状态表，并区分 vLLM prefix 只缓存 full blocks 的实现约束。 |
| §3.8.5（新增） | — | 增加 PagedAttention 与 vAttention 的地址管理、收益、代价、论文数据和不可直接排名边界。 |
| §3.9 | 理想量化收益、仿射量化公式、K/V 误差差异、评测矩阵 | 原内容保留；新增 KIVI 的 K/V 不对称量化轴、G/R residual 的 6→7 token 轨迹，真实系统结果及 vLLM FP8 版本对照。 |
| §3.10 | prefix workload、命中比例、缓存键、失效/回收、多租户安全 | 原 key 维度和权限论点保留；新增 block parent hash/multimodal extra hash/cache salt 证据、首个共享 prefix 生命周期 trace、NDSS 侧信道边界，以及 vLLM/TensorRT-LLM 实现对照。 |
| §3.11 | 截断/滑窗/摘要/retrieval/offload、active vs reusable、1M context 限制 | 原策略与边界保留；新增 H₂O 与 StreamingLLM 代表性方法、8-slot 教学轨迹，区分 eviction/quantization/offload/recompute。 |
| §3.12–§3.13 | OOM 阶段、排查顺序、分母、可执行 KV 成本审计 demo | OOM 诊断保留；修正 batch validation 顺序，补非法请求映射、重复 ID、空 batch 测试。输出账本数据不变。 |
| §3.14–§3.15 | 七类误解、长上下文多租户容量与 cache key 设计 | 全部保留；新增章节没有覆盖或替代其内容。 |
| §3.16–§3.17 | 来源入口、证据边界与总结 | 更新论文、官方文档和综述入口；总结保持标准 KV 公式/内存管理主线。 |

## 公式、代码、导航与配套验证

- `formula_status`：复核 block 数 `ceil(T/S)`、尾部浪费 `S*sum(N_i)-sum(T_i)`、示例 `i=9,S=4,physical block=11 -> slot=45`，以及 KIVI 教学轨迹中 `G=2,R=4` 在第 6 个 token 后 K/V 的低比特区和 residual、第 7 个 token 到来后的更新。代码 demo 的 MHA/GQA/MQA 容量、0.25/0.03125 比例、分页槽位 5,760、尾浪费 300、静态槽浪费 1,708、GQA 0.667 GiB、分页 0.703 GiB 和预算判定一致。`\operatorname` 写法已从本章公式移除；外部公式渲染未执行。
- `code_status`：从 Markdown 提取并执行 §3.13 标准库 Python demo。所有金额/比率/列表和诊断与正文相同。预期拒绝 6 类输入：零 block size、reserved 小于 active、bool token count、非 mapping request、duplicate request id、empty batch。仅是 local demo，不是 GPU allocator、真实 attention 或 serving benchmark。
- `link_status`：本轮实际打开核对 PagedAttention/vAttention/KIVI/H₂O/StreamingLLM 原论文、KV survey、NDSS 论文页、vLLM latest paged/prefix/量化页面和 TensorRT-LLM 当前 KV 文档；版本化文档明确使用 vLLM v0.25.0。后续部署须按目标 runtime release 复核动态 `latest` 文档。
- `sync_status`：`PAPERS.md` 增加 H₂O/StreamingLLM、KV 管理综述和 NDSS prompt-leakage 入口，并补全 vAttention 论文链接；`GLOSSARY_EN_ZH.md` 增 Cache Salt、Copy-on-Write KV Block、KV Cache Eviction、Attention Sink、Heavy-Hitter Token、KIVI；`INTERVIEW_BANK.md` 增 H₂O/StreamingLLM 与 KIVI 面试比较；`EXERCISES.md` 增分页/量化/淘汰方法推演；第六册目录已有本章入口，未改动。
- `open_risks`：没有在目标 GPU、目标模型或线上 workload 实测 PagedAttention/vAttention、KIVI、H₂O、StreamingLLM 或 prefix reuse；论文数字只绑定各自报告的模型、任务、kernel、硬件和 baseline。没有证明目标 runtime 实现了 KIVI 2-bit；没有证明 cache salt 可替代鉴权、加密、日志策略或数据删除；没有外部公式渲染或 PDF 构建。
- `status`：按声明范围完成本章资料驱动精修与代码/公式/链接/导航检查；全书 goal 保持进行中，下一份正文进入第六册第四章 `04-解码策略与生成控制.md`，按书系顺序继续，不回头重做前序章节。
