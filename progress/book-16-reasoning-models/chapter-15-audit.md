# 第十五章审计：Preserve Thinking 与推理状态

审阅日期：2026-11-13

## 范围与章级扫描

- **范围：** 完整精修《Preserve Thinking 与推理状态》，聚焦 provider 协议状态的跨轮续接，以及它与宿主执行、权限和外部状态的分工。
- **章级扫描：** 2026-11-13 定向阅读 Anthropic Preserved thinking、OpenAI Conversation state / Preserve reasoning / Reasoning models、Google Thought signatures / Interactions API 概览，以及已有的 xAI Reasoning 文档。问题包括：服务端会话引用和客户端回放的差别、thinking/reasoning item 的前缀和模型兼容条件、stateless 重放要求、provider 状态的保留边界及工具超时后的责任归属。
- **候选主题去向：**
  - OpenAI Responses/Conversations、Anthropic Messages、Gemini Interactions 的续接方式及 replay 限制：纳入 §15.2，逐一绑定官方 API surface；Grok 4.7 Responses 的 encrypted reasoning 作为第四个简短示例。
  - 上下文改写、压缩与模型切换：纳入 §15.3；只说明官方文档记录的前缀/兼容性行为，不推断隐藏 reasoning 表示。
  - 工具调用超时、外部状态、权限重查：在 §15.4 保留识别边界所需的最小案例；工具事件生命周期和状态机详解交第十九章，checkpoint/Runtime 实现交第二十册。
  - reasoning item 的内部表示、模型完整 latent state 和跨供应商统一兼容规则：无公开证据或不存在统一合同，不作推断。
  - 数据保留、隐私和压缩摘要：纳入本章接口与宿主责任边界；不展开通用 Runtime 存储架构。
- **架构边界：** 第十四章讲离散 thinking level；本章讲 provider 状态协议；第十九章讲交错工具事件；第二十册讲 Agent Runtime 与长任务 checkpoint。无需改变第十六册骨架。

## 正文结构、篇幅与数值例子

- 原稿 15.1—15.22 中，多次重复“thinking 状态不等于公开 CoT”“完整历史不是 Runtime 状态”“超时不能直接重试”等解释；重排为八节：状态分类、provider API 合同、历史编辑、工具/外部状态分界、隐私生命周期、恢复评估、兼容性代码演示、结论与练习。
- 核心讲解构成一个连续单元；去掉标题、表格、代码、公式和围栏后，解释性汉字实计 2,358 字，高于核心单元 1,800 字下限。
- §15.6 以 100 个中断任务演算：82 个正确恢复、12 个安全暂停、6 个恢复错误，恢复率为 82/100=0.82；安全暂停率另报 12%。无中断样本时分母为零，指标记为 N/A。示例是教学输入，不是行业 SLO。
- 唯一 display-math 公式在前文定义变量、分母条件、正确恢复判定，并有完整数值代入；没有未解释的附加公式。

## 代码、证据与链接

- §15.7 Python 教学例用 SHA-256 摘要比较状态产生前的 system、tools 和 messages。正文明确它是严格的教学合同，不模拟任一 provider 的加密签名、兼容矩阵、compaction 或账户规则。
- 代码状态：已验证，日期 2026-11-13，CPython 3.12.9，仅 Python 标准库。使用命令：

~~~text
sed -n '/^~~~python$/,/^~~~$/p' 'book-16-reasoning-models/chapters/15-preserve-thinking与推理状态.md' | sed '1d;$d' | .tmp/llm-book-cpu-check/bin/python
~~~

- 实际输出与正文一致：append-only 为 compatible；编辑历史和工具 schema 变更为 prefix_mismatch；切换模型为 model_mismatch。代码内断言覆盖四种结果；没有真实 provider API 调用。
- 2026-11-13 官方来源分别支持其所在句子的 API 合同；研究来源与范围记录于[推理时计算与评估研究记录 §12.1](../../research/test-time-compute-and-evaluation-2026.md)。来源证据只支持所列 provider、model/API surface 和文档状态，不支持“公开完整 CoT”或统一隐藏状态机制。
- 本地章节链接指向第十九章和第二十册第十七章，目标均存在；本章简介、目录摘要已能准确概括修订后职责，无需同步。
- 复核 INTERVIEW_BANK.md、EXERCISES.md、GLOSSARY_EN_ZH.md 和 KNOWLEDGE_GRAPH.md 已有 preserve thinking、state replay、provider compatibility 相关题目与术语，不重复添加配套内容。
- 状态分类表和 provider 合同表承担主要视觉比较；本章不需额外 Mermaid 图。公式与代码围栏配对，git diff --check 通过。

## 遗留风险与交接

- Provider 官方文档、支持模型、账号绑定、存储期限和压缩规则会变化；真实集成仍需按准确 model ID、API surface、账号设置和访问时点复核。
- 本轮没有执行真实线上请求；文档证据不替代目标账户的 API 回归、数据治理审查或运行时故障演练。
- 第十六册第 1—15 章完成逐章审阅和精修；下一入口为第十六章《Multi-Agent 与 Ultra Test-Time Compute》，第 16—20 章和整册连续复读/最终门禁待完成。按用户要求不更新 PDF 或 plan.md。
