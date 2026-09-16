# GLM-5.2 官方文档核验摘记

核验日期：2026-09-09。来源：[Z.ai GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2)。

## 已确认内容

官方将 GLM-5.2 定位为面向长任务的旗舰基础模型，页面标称文本输入/输出、1M 上下文和 128K 最大输出，并列出函数调用、MCP、上下文缓存和结构化输出能力。文档用项目级代码库接管、跨文件重构、生产标准压力测试、移动设备调试、论文复现等场景说明长任务定位。

页面称为让百万 token 上下文“可用”，模型接受了数月面向长程 Coding Agent 的专项训练，并声称使用“lossless context”。这些属于发布方叙述，尚未有独立复现；正文不应把“lossless”解释为数学意义上的绝对无损。

## 与 GLM-5.3 的关系边界

GLM-5.3 官方文档称沿用 GLM-5.2 基础模型并通过后训练改进。GLM-5.2 页面在本次读取的正文中没有出现 SAO 或 compaction 的可展开定义，因此不能从该页反推算法。GLM-5.3 所说“继承 GLM-5.2 的 SAO with compaction”必须等待 GLM-5.2 技术报告、发布博客或论文的专门来源。

GLM-5.2 页面展示的 /goal 模式、CLAUDE.md/Agent.md 约束、ADB/logcat、研究复现等提示词属于产品使用示例，不等同于模型内部训练算法。正式章节应将它们作为 Agent harness 与工作流案例讲解。

## 后续核验

## 2026-09-16 断点恢复与榜单复验

本轮按夜间 20:00—次日 09:00 可能中断的规则重新抓取两个排行榜。Artificial Analysis [`glm-5-2`](https://artificialanalysis.ai/models/glm-5-2) 和 DataCurve DeepSWE 均返回 HTTP 200；AA 快照 `/tmp/glm52-aa-20260916.out` 为 `3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；DataCurve 快照 `/tmp/glm52-ds-20260916.out` 为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。

Artificial Analysis 将其记录为 `GLM-5.2 (max)`，第三方 FAQ 约为 Intelligence Index `34`、1M context、约 `72 tokens/s` 和 `$1.40/$4.40` 每百万 input/output token；这些仍是第三方页面的配置、provider 和测量字段。DataCurve 页面记录：

- `mini_swe_agent_glm_5_2_high`：164/452，Pass@1 `36.2832%`，Pass@4 `68.1416%`，平均成本约 `$2.8355`，平均输出 `54,245.50` token，平均 Agent steps `121.88`；
- `mini_swe_agent_glm_5_2_max`：197/450，Pass@1 `43.7778%`，Pass@4 `76.9912%`，平均成本约 `$3.9199`，平均输出 `78,175.31` token，平均 Agent steps `129.13`。

这两行使用 `mini-swe-agent`、4 runs、113 tasks、91 repositories、5 languages，并受工具、环境和 verifier 影响；不能与 AA Intelligence Index 拼成裸模型排名。

## 2026-09-16 官方资料补充

新鲜获取的 [Z.ai GLM-5.2 官方文档](https://docs.z.ai/guides/llm/glm-5.2) 的 `dateModified` 为 `2026-09-03T14:49:41.429Z`，页面快照 `/tmp/glm52-zai-doc-8098-20260916.out` 为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`。官方页面明确：

- GLM-5.2 面向 long-horizon tasks，支持 1M context、128K maximum output、thinking mode、function calling、context caching、structured output 和 MCP；
- 训练与产品叙述强调 project-scale codebase、跨文件重构、需求到部署的连续工作流，并声称经过数月 Coding Agent 场景专项训练；“lossless context”按发布方产品描述记录，不解释成数学意义上的绝对无损；
- 页面把 /goal 模式、CLAUDE.md/Agent.md 约束、ADB/logcat、研究复现和分阶段的 requirements—implementation—verification 流程作为使用案例；这些属于提示词/Agent workflow 证据，不等同于内部训练算法。

本轮没有找到 GLM-5.2 独立技术报告、模型卡架构细节或 SAO/compaction 的原始算法定义。GLM-5.3 文档声称沿用 GLM-5.2 基础模型并继承 SAO with compaction，但不能据此确认 SAO 全称、损失函数或压缩实现。该缺口继续保留。

## 当前结论

GLM-5.2 已从“部分覆盖”升级为“资料级闭环”：两个排行榜的精确配置、Z.ai 官方模型文档、长上下文/Agent 工作流证据和研究笔记均已具备；暂无独立正式章节。面试主线是 1M context 的可用性工程、长任务状态连续性、工具/MCP 集成、context caching、跨文件计划—执行—验证闭环，以及榜单配置与 Agent 系统结果的分层。参数量、内部架构、SAO/compaction 原始定义、完整训练/后训练 recipe、生产 kernel、硬件 profiling 和独立 benchmark 仍待核验。
