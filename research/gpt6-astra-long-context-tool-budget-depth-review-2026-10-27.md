# GPT-6 Astra 长上下文与工具预算：深度精修研究记录

核验日期：2026-10-27。研究对象是第六册第十八章原有的 GPT-6 Astra 长上下文、reasoning budget、工具状态、服务费率和 Agent runtime 内容。本笔记只使用 OpenAI 官方开发者文档与官方博客核验 OpenAI 产品事实；不把官方 API 文档当作论文、内部架构披露或生产实测。

## 1. 本轮要回答的问题

1. 模型页的 context、maximum input 和 maximum output 分别由什么字段支持？哪些数字只是算术推导？
2. reasoning mode、reasoning effort 与 max output 的关系是什么？为什么它们不能当作单一“模型质量档”？
3. Astra 的 short/long context、缓存读取、30 分钟/12 小时缓存写入、Standard/Batch/Flex/Fast/Ultrafast 如何共同影响模型 token 费用？
4. 一个多轮 Agent 任务怎样设置任务级预算，避免每个响应都未超限但整个 job 无限扩张？
5. tool search、configuration update、async tool、mid-turn steering、compaction、安全监控分别修改哪种状态？适用边界和故障恢复是什么？
6. 旧稿已有的 GPT-5.3 Codex、GPT-6 Sol/Luna、GPT-6.1 Sol 对照如何保留，同时纠正过期的 maximum input、endpoint 和弃用状态断言？

## 2. 阅读的官方资料与可支持结论

| 官方资料 | 本轮实际核验内容 | 证据不能外推到 |
|---|---|---|
| [GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra) | 页面列出 1,050,000 context、128,000 max output、effort 档位、模态、工具和 endpoint；没有单独列 922,000 maximum input；输入超过 272K 后整次请求 input/cache ×2、output ×1.5；列出 Standard token 价和 cache write 1.25× uncached input | 真实请求一定能用满容量、模型架构、目标账号的SLA/费用 |
| [GPT-6 family guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra) | family 级模型参数兼容矩阵、Astra 的 reasoning tool-call endpoint、configuration update 和家族部署限制 | Astra 的内部训练差异或未列出的端点行为 |
| [Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) | mode=standard/pro 与 effort 独立；推理 token 会消耗上下文并计入输出；输出上限可能在可见回答前触发；初次试验建议预留 25K | 所有任务都会用到相同 token 数或 pro 一定更准确 |
| [API pricing](https://developers.openai.com/api/docs/pricing) | Astra 的 Standard、Flex/Batch、Fast、Ultrafast 价表，short/long context 以及 30m/12h cache-write TTL；region/FedRAMP 附加费说明 | 账户折扣、目标地区合同、未来价格 |
| [Fast mode](https://developers.openai.com/api/docs/guides/fast-mode) 与 [Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode) | 两种服务档位；Ultrafast 单独限流、频繁工具回合适合持久 WebSocket 以减少往返；Astra Ultrafast 的 data residency 范围；Astra Fast 没有 latency SLA | 某个工具链的端到端加速幅度或时延保证 |
| [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) | 缓存复用依赖稳定前缀；deferred tool loading 将新工具追加到上下文；configuration update 改变后续 effort 时保持 request-level effort 不变可避免改写隐藏指令前缀 | 缓存命中保证、实际线上命中率 |
| [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state) 与 [Compaction](https://developers.openai.com/api/docs/guides/compaction) | 无状态 replay 保留 output items；server-side compaction 的 item；stateless array 与 previous_response_id 的不同裁剪方式；standalone compact 输出应原样继续使用 | compaction 的内部摘要算法、等价性保证或持久业务状态 |
| [Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search) | hosted 与 client-executed search、deferred function schema、发现步骤的收益与代价 | 应用自定义目录的准确性或权限正确性 |
| [Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling) | 应用侧慢 function/custom tool 的异步路径、task handle、原 call_id、独立 wait 工具、multi-agent 限制 | 内建 hosted tools 自动异步或外部任务天然幂等 |
| [Mid-turn steering](https://developers.openai.com/api/docs/guides/steering) | accepted 只表示排队；自动 continuation 条件；原响应可能 incomplete 或先 completed；每个响应单独计算 token/tool limits | 已发文本/副作用会被撤回 |
| [Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring) | Responses 状态保留方式决定监控/自动停止能力；Chat Completions 不覆盖；403 错误码和 stream 错误处理 | 权限、审批、工具沙箱或 verifier 可以被它替代 |
| [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) | 技能描述、最小根级路由文档、按需加载材料的公开工程建议 | Astra 内部记忆、训练机制或普遍效果提升 |

资料页在不同日期会变更；表中是 2026-10-27 的页面观察，不是固定产品合同。

## 3. 核心事实核对与推导

### 3.1 1.05M context 与 922K 的来源

Astra 当前模型页明列 context window 1,050,000 和 max output 128,000，没有单列 maximum input。减法 1,050,000−128,000=922,000 只有一个有限用途：在假设满额 output 全部留给本响应时，展示理想化算术余量。它不能改写成“OpenAI 公布 Astra 的 maximum input 是 922K”。实际 input 还受输入 item、工具 schema、历史状态、预留推理和服务端实现约束。

同样的页面口径也适用于 GPT-6 Sol、Luna、GPT-6.1 Sol 和 GPT-5.3 Codex 的当前模型页：若页面只列 context 与 max output，可以做条件化的减法推导，不能把余量提升为 provider 的独立输入限制。GPT-5.3 Codex 当前页面已标记 Deprecated；其页面同时列出 Chat Completions 与 Responses，因此“Responses-only”已过时。

2026-10-27 官方定价页还列有 GPT-6.1 Terra。第十八章仍聚焦 Astra 与旧稿中已有的 Sol/Luna/Codex 对照，不把这份比较误作完整的 GPT-6 模型目录，也不从价格条目推断 Terra 的参数或接口行为。

### 3.2 Reasoning 的三层预算

- <code>reasoning.mode</code> 选择 standard 或 pro 执行模式；<code>reasoning.effort</code> 是该模式中的投入设置。二者独立。
- <code>max_output_tokens</code> 同时约束 reasoning 与可见 output。Reasoning 不展示为普通文本，也仍占 context 和 output billing。
- “至少留 25K”是官方初次试验建议；后续要依 usage 调整，而非将 25K 写成固定模型开销。

所以正确的实验矩阵要固定模型、输入、工具和 timeout，同时分别记录 mode、effort、reasoning_tokens、可见输出 token、结果质量、延迟和成本。仅切档后看主观回答质量，无法识别收益来自更多推理、更多模型调用还是失败样本选择变化。

### 3.3 Astra 费率计算

本轮从定价页读取 Standard/Flex-Batch/Fast/Ultrafast 与 short/long 两组费率，写入正文完整表格。Long context 阈值按 input token 总量 >272K 判断；整次请求 input/cache rates 与 output rate 改变，不做超过阈值部分的边际计费。缓存的三个输入类别按互斥分解：普通 uncached 输入、cached read、新 cache write。cache-write TTL 是 30m 或 12h；推理 token 已在 output usage 中，不可二次相加。

正文 Python 教学函数在 300K 输入示例中使用普通输入 140K、cached read 100K、30m cache write 60K、output 20K。Astra Standard long-context rates 分别为 $20/M、$2/M、$25/M 和 $75/M，成本为：2.80 + 0.20 + 1.50 + 1.50 = $6.00。工具费、region/FedRAMP 附加费、账户价和 app executor 费用均排除在函数之外。

### 3.4 从单响应限制到 job budget

模型/API 限制通常按单个 response 生效，业务目标却按整项 task 生效。多轮 response、steering continuation、retry、async job、验证器和人工接管都会累积花费。正文把预算字段拆到每请求上下文、每响应输出、整个 job 模型费、工具调用数/并发/timeout、重试与恢复、verifier、deadline 与副作用审批；其中 global tool cap、job cost cap、retry policy 都需要应用 runtime 维护。

并行仅可能缩短并行子任务的 wall-clock time；并发不会自动减少 call count、模型消费工具结果的 tokens 或托管工具逐次费用。对不可逆写入，重试前需先查 executor receipt/idempotency 状态，不能把 API 或模型响应状态视为外部事务回滚证据。

## 4. 代表性运行时方案各自解决什么

| 方案 | 解决的问题 | 机制和收益 | 成本/限制 | 验收方式 |
|---|---|---|---|---|
| 长原始上下文 | 本轮仍需查阅多段原始资料 | 保留原始消息或内容供模型访问 | 提高 input tokens/成本；窗口大不保证长距离检索有效 | 固定位置、长度与证据密度测检索召回和回答正确率 |
| Eager tool schema | 工具少且高频，避免每次先发现 | 创建请求时全部提供可调用 schema | 未用工具 schema 占上下文，schema 变化可能打断 cache prefix | 统计 schema tokens、每任务工具使用率和 cache hit |
| Deferred tool search | 大工具目录中每任务只需少数工具 | 先提供目录，再按需加载 schema；可 hosted 或应用执行 discovery | 增加发现步骤；描述/搜索可能漏选；不授予权限 | 测 candidate recall、额外 token/latency、错误调用与权限拒绝 |
| configuration_update | 长对话中需改变后续 reasoning effort | 追加 item，保留顶层 effort，尝试保持缓存前缀 | 有模式/单 Agent/compaction 等兼容限制；更新历史必须留存 | replay 原更新顺序，确认后续响应设置与 cache 状态 |
| Server-side compaction | 多轮历史达到阈值，需要 API 管理压缩与续接 | 自动生成 opaque compaction item 并继续响应 | 内部表示不可读；服务端状态仍不等于业务状态 | 在 stateless replay 保留最新 item，验证任务、tool receipt 与产物 |
| Standalone compact endpoint | 应用明确控制何时压缩 stateless context | 返回 canonical next context window | 输出可能含 retained items，不能手工改写或 prune | 后续请求原样使用返回窗口并做任务状态回归 |
| 外部 job/artifact ledger | 副作用、权限、任务目标与产物需要权威来源 | runtime 独立持久化工具回执、审批、状态和文件 hash | 应用负责一致性、访问控制、保留与清理 | 故障注入检查重复执行、恢复、权限变化和最终产物 |

方法间的关键区别是状态对象不同：tool search 改 schema 是否载入；compaction 改上下文表示；configuration update 改后续 reasoning 设置；外部 ledger 保存业务真相。它们可组合，但不能互相替代。

## 5. 保留的旧正文知识点与新位置

| 原章节内容 | 本轮去向 | 修订判断 |
|---|---|---|
| context / maximum input / maximum output 三长度 | 新 §1、§10、练习 1 | 保留长度预算问题；将 Astra/Sol/Luna/6.1 Sol 与 Codex 的 922K/272K 改成条件算术余量，删除“官方单列 maximum input”的错误口径 |
| reasoning effort 与模型边界 | 新 §2、§10、面试题 | 保留；新增 mode/effort 正交、隐藏 reasoning 消耗 output cap、incomplete 诊断 |
| 模态、工具可用性与宿主权限 | 新 §3 与练习 5 | 保留；补 model proposal、policy、executor、回执、verifier 责任链 |
| 成本表与 Python estimate_cost | 新 §4 Python 估算 | 保留阈值、cached tokens、输出、费用例子；改为包含 30m/12h cache write、Batch/Flex/Fast/Ultrafast 和 long-context rates |
| Agent Runtime、跨层 trace | 新 §5 与运行时 ledger | 保留；从日志字段拓展成具有所有者和硬拒绝路径的整个 job budget |
| replay、encrypted reasoning、phase、previous_response_id | 新 §6 与练习 6 | 保留；将 final text-only replay 的状态丢失风险说清 |
| configuration_update、effort 升档、公平评测 | 新 §2、§6 | 保留；加 cache-preserving top-level effort 约束与兼容边界 |
| tool search / deferred schema | 新 §6、§9 与练习 7 | 保留；加 hosted 与 client 搜索差异、schema 可见性、发现步骤成本、授权隔离 |
| async tools 和 response lineage | 新 §7 与练习 8 | 保留；明确只针对 app-executed function/custom tool、call_id 与 task_handle、executor 与模型消费状态拆开 |
| mid-turn steering | 新 §7 与练习 9 | 保留；补 accepted 仅排队、continuation 条件、原响应两种结束路径、per-response caps |
| Skills、AGENTS.md 与 progressive disclosure | 新 §8 与练习 12 | 保留；清楚标作上下文工程建议，不声称 Astra 的内部学习机制 |
| misalignment monitoring 与禁止盲重试 | 新 §8、面试题和练习 10 | 保留；新增 Responses/WebSocket/state/Chat Completions 覆盖矩阵 |
| server-side 与 standalone compaction | 新 §9 与练习 11 | 保留；分别说明 stateless array 可删最近 compaction item 前旧项、previous_response_id 不手动 prune、standalone 输出完全不 prune |
| GPT-5.3 Codex 对照 | 新 §10 | 保留研究对照；修正为 Deprecated，修正 endpoint 列表与 input 余量口径 |
| GPT-6 Sol/Luna/GPT-6.1 Sol | 新 §10 | 保留精确 model identity；不把某 sibling 的价格、effort、评测或 API 组合迁移给 Astra |
| GPT-6 Sol 合同审计 toy | 新 §10 与 [Sol 来源记录](model-update-2026-09/gpt-6-sol-source-notes.md) | 保留旧稿脚本作为历史教学材料；明确其中 `922K` 输入上限和费用常量是 2026-09-23 Sol 合同快照，不能当作 Astra 或当前合同 |
| GPT-OSS 开放权重对比 | 新 §10 末段 | 保留预算分层；提醒权重驻留/分片和 API token 账单不是同一口径 |

## 6. 仍未验证的内容

- 未调用 GPT-6 Astra API；异步、steering、compaction、工具检索和误对齐事件只按当前官方文档核验，未实测真实事件顺序、失败率或数据保留。
- 没有目标帐号的 regional/FedRAMP 合同价、配额、延迟 SLA 或工具账单；正文价格仅为 2026-10-27 公共页快照。
- 没有目标 Agent 任务集、真实工具 schema、executor trace、任务成本样本或 SLO；job-budget 数字明确标作教学假设。
- 官方文档没有披露 compaction 算法、Astra 参数规模、内部结构或完整训练配方；不作推断。
- 第三方榜单和 DataCurve/Artificial Analysis 配置需绑定其自身观察时间与 harness；不把邻近模型成绩迁移到 Astra。

第十八章现有状态机 demo 的范围仍限于教学状态转换；它不覆盖本轮新增的定价表，也不作为 API 运行时的事实证据。
