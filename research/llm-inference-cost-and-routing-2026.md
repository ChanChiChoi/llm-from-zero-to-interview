# 在线推理成本、缓存计费与模型路由证据（2026）

本记录保存在线推理成本与路由主题的来源事实和证据边界。动态页面访问日期为 2026-11-02；教学数值与生产账单、吞吐或质量结果无关。

## 路由评估目标可能偏离任务难度

- **来源**：[Dynamic LLM Routers are Often Misguided](https://arxiv.org/abs/2610.02762v1)，arXiv v1，2026-10-02。来源类型：研究预印本；证据等级：source-backed。阅读范围包括摘要、评测设置、主要结果与限制。
- **作者设置**：评估 6 个商业路由器、14 种设置、17 个 benchmark、8 类任务，共 800 个评估样例和 26 个候选模型。结论绑定该评测集、候选模型集合、单轮英语任务和每模型/问题仅一次生成的协议。
- **作者报告**：在其匹配成本比较中，没有被测 router 超过两个合适模型之间的随机选择基线。作者分析了 difficulty blindness、length reversal、semantic matching 和 roster suboptimality：总成本/准确率目标可能让系统把更多升级预算给中等难度问题，短输出的模型选择也可能因实际输出较短而显得更便宜；跨来源相关性可能让路由器利用数据集来源，而不是题目难度。
- **评估建议**：作者建议比较模型平均成本与按样例实际输出长度计价的差异，按难度切片报告，并检查每个数据来源内部的路由表现。其对“更难题优先获得更强模型”的偏好是设计选择，不是普适数学定理。
- **限制**：单轮英文 benchmark、有限样例、单次生成不能覆盖随机性、多轮状态、切换模型后的 KV 代价或生产任务分布；该结果不证明所有路由器都无效。未来模型专长、候选 roster 和路由训练也可能改变结论。

## OpenAI API 的 reasoning token 与 prompt-cache 账单字段

- **来源**：[Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)、[Token counting](https://developers.openai.com/api/docs/guides/token-counting) 与 [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)。来源类型：供应商官方动态 API 文档；访问日期：2026-11-02；证据等级：source-backed。阅读范围包括 usage 字段、计费类别、缓存保留与价格说明。
- **Reasoning usage**：当前 reasoning 文档说明，reasoning token 不作为用户可见文本返回，但计入输出 token 用量；usage 可通过 `output_tokens_details.reasoning_tokens` 单独查看。Token counting 文档指出输出 token 统计不只包含可见文本。结论只适用于其公开 API 合同，其他供应商和模型须检查各自 usage schema 与账单。
- **Prompt cache**：文档按模型区分缓存读取、缓存写入和普通输入；读写字段应结合总 input usage 和最终发票核对，不能把子集再次加到完整输入量。命中读取仍计入相应 token rate limit；缓存最小前缀、breakpoint 和保留时长依模型/API 而异。
- **日期化价格示例**：访问日文档对 GPT-5.6 及之后的部分模型列出默认 30 分钟写入费率为普通输入的 1.25 倍，12 小时写入费率为 2.5 倍；读取费率按模型变化，多数示例为 0.1 倍，GPT-6.1 Sol/Terra 为 0.05 倍。此处是该日期的动态合同记录，不能视为全系列、所有 endpoint 或未来价格规则。
- **解释边界**：供应商 prompt cache 影响 API usage/账单；本地 prefix KV cache 复用 prefill 计算并占据 GPU 显存。两者不能用同一命中率或账单推论互相替代。

## 推理运行时与扩容文档

- **vLLM Automatic Prefix Caching**：[官方文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)，访问于 2026-11-02。文档说明命中共享前缀可跳过重复 prefill；性能收益取决于共享前缀与 workload，KV 驻留仍占用显存。该能力不意味着供应商账单自动下降，也不保证 decode 主导负载加速。
- **vLLM Speculative Decoding**：[官方文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)，访问于 2026-11-02。文档将推测解码描述为降低 inter-token latency 的方法，收益依 draft/target、接受率、硬件和负载而变；当前指南指出其适用条件偏向中低 QPS、memory-bound 场景，不能据算法接受率单独推断端到端吞吐或单位成功成本。
- **Kubernetes HPA**：[Horizontal Pod Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)，访问于 2026-11-02。官方文档给出基于当前/目标指标比例计算期望副本数的控制关系，并说明多个指标时采用各指标建议副本数的最大值。它不自动表达 LLM 的排队年龄、待处理 token、KV 容量、模型加载时间或任务质量约束；自定义指标、稳定窗口和冷启动需结合目标服务定义。

## 2026-11-05 补充：语义缓存风险控制与多租户 KV 复用

### vCache 与 Surplus

- **来源**：[vCache: Verified Semantic Prompt Caching](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9559cd2116de7a8f5672eac3fcd232cc-Abstract-Conference.html)，ICLR 2026 正式论文页；访问日期：2026-11-05；证据等级：source-backed。阅读范围包括摘要及其所述的按缓存条目在线估计阈值方法。作者报告目标是用户指定的错误率界限，并称实验在其基准上满足该界限；这是作者对其方法和评估条件的报告，不代表任意任务、反馈标签或部署都获得无条件保证。
- **来源**：[Semantic Caches for LLM Web Services Leave Their Error Budget Unspent](https://arxiv.org/abs/2611.02498v1)，arXiv v1，2026-11-02；论文页面列明 ACM Web Conference 2027；访问日期：2026-11-05；证据等级：source-backed。阅读范围包括方法、评测、反馈信号/人工复核、serving testbed 和限制。
- **方法事实**：Surplus 在相似问题簇间共享风险评分，以校准后的风险阈值决定复用范围，并使用未命中请求获得的反馈标签和已发生错误账本应对流量漂移。作者报告五条查询流的 15 个测试单元均满足论文设定的 δ；在高负载 serving 点报告平均延迟降低 31.59%，vCache 对照为 23.67%。这些均是作者在论文数据集、反馈协议和配置下报告的结果。
- **serving 条件**：主要对照之一使用 vLLM 0.11、Qwen3-8B bf16（不启用 thinking）、单张 RTX 4090 和 2,000 条 LmArena 查询，按 Poisson 到达加载至后端闭环吞吐的 1.1 倍。数字不代表中文合同问答、其他模型、硬件或流量下的收益。
- **反馈边界**：论文说明 δ 是相对于所采用反馈信号定义的；在 SearchQueries 上，benchmark 标签比人工评审判断宽松。作者另报告 blind human review 结果，但不能据此把搜索任务的错误预算等同于合同内容、权限判断或业务损失的错误率。

### 多租户 Prefix/KV Cache 安全

- **来源**：[I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/)，NDSS 2025 官方论文页；访问日期：2026-11-05；证据等级：source-backed。阅读范围包括摘要和攻击描述。论文在其多租户框架及攻击者先验知识/观测假设下展示共享 KV 缓存侧信道与提示重建；这不证明所有当前引擎、版本、硬件或部署都可按相同方式利用。
- **来源**：[vLLM Security](https://docs.vllm.ai/en/latest/usage/security/)，官方动态文档，页面标注日期 2026-10-30；访问日期：2026-11-05；证据等级：source-backed。文档称共享同一 server process 的租户没有进程级隔离，请求共同调度并共享缓存；`cache_salt` 只能缓解特定前缀缓存风险，不能作为隔离边界。它由可信网关按隔离范围设置为不可猜的秘密，降低跨盐复用；严格隔离需由独立实例或网关/部署边界实施。盐粒度会影响前缀共享率。
- **缓存机制**：[Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)，vLLM 官方动态文档；页面标注日期 2026-11-04；访问日期：2026-11-05。命中相同前缀可复用 KV 并跳过共享部分的 prefill；文档说明它不减少生成阶段的 decode 时间，因此长输出占主导或请求间缺少共享前缀时不带来相同收益。结论限于该运行时文档描述的功能，部署表现仍需按 workload 测量。
