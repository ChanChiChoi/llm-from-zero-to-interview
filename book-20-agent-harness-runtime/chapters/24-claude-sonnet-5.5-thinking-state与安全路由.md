# Claude Sonnet 5.5：Thinking State、Agent Runtime 与安全路由

本章把 Claude Sonnet 5.5 作为官方 API/runtime 合同案例。模型来自 Artificial Analysis 榜单锚点；DataCurve DeepSWE 当前无精确 Agent 行。下面的 API 行为以 Anthropic 文档为准，能力与安全数字以 System Card 的发布方测试为准；两类证据都不揭示内部模型架构。

完整快照、来源与未核验项见[研究笔记](../../research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)。

## 24.1 从榜单配置到模型身份

Artificial Analysis 当前列出 `claude-sonnet-5-5`，及 low/medium/high/xhigh effort 变体。它们属于同一基础模型的配置；AA 主页面的 max + Default Fallback 是第三方测量配置。DataCurve 没有精确 `mini_swe_agent_claude_sonnet_5_5_*` 行，因此不要借用 Sonnet 5 或其他 Claude 版本的 coding-agent 分数。

Anthropic API model ID 是 `claude-sonnet-5-5`，官方文档列 1M context、128K max output、输入/输出 `$2/$10` 每百万 token、adaptive thinking、默认 `high` effort。Sonnet 5.5 与 Sonnet 5 的 tokenizer 相同，但 effort 行为重新校准；从旧版本迁移要重跑 workload-specific effort sweep，而不是把 `high` 当成固定 token 预算。

## 24.2 `between_tools`：关闭 upfront thinking，但不是“没有 reasoning block”

Sonnet 5.5 不接受旧字段 `thinking: {"type":"disabled"}`。关闭工具调用前的 upfront thinking 要求 `thinking: {"type":"between_tools"}`。名字容易令人以为模型在工具间绝不会输出 thinking；官方文档实际说明，较长的工具间进度仍可能作为 thinking/progress block 返回。

| 请求 | Sonnet 5.5 合同 | 宿主注意点 |
|---|---|---|
| `thinking.type=disabled` | 400 | 迁移旧客户端 |
| `thinking.type=between_tools` + low/medium/high | 接受 | 工具间仍解析/回放 progress blocks |
| `between_tools` + xhigh/max | 400 | 改用 adaptive thinking |
| `between_tools` + `display` / `budget_tokens` / `block_binding` | 400 | 该模式不接受额外 thinking 字段 |
| adaptive + 默认 display omitted | 可接受 | 可能返回空文本的 thinking block；不要只取首个 text |

如果请求没有工具，`between_tools` 会直接回答，不先进行多步思考。因此它适合降低工具型工作流中的 upfront reasoning 成本，不能代替需要先推理的数学或 JSON 问题。adaptive thinking 的 effort 也不是精确 token 预算；`max_tokens` 是思考与最终答复共享的硬上限。

## 24.3 Thinking block 是绑定的会话状态

模型返回的 thinking block 属于有类型的协议状态，不能随意删除、重排、跨模型搬运或当成 UI 文本展示。默认 display 设置下，其内部文本可能为空；同时工具间的长 progress update 也可能被封装为 block。一个只渲染 `text` 的界面会表现为“工具调用期间突然没动静”，但服务并没有报错。

回放前至少检查三种绑定：

1. **Model binding：**目标模型是否能读取 producer model 的 block。Sonnet 5.5 可以接续部分旧模型 block，但 Sonnet 5.5 自己产生的 block 不会被其他模型读取；切换后不兼容 block 可静默 drop。
2. **Prefix binding：**block 前面的 `system`、`tools` 或旧消息有没有被改写。特定新账号默认强制检查，失败可能是 400；老账号的默认行为不同，需按请求字段判断。
3. **Account binding：**Sonnet 5.5 的 block 绑定生成它的账号或关联账号；跨账号回放会被 drop。可用 beta 返回的 `input_transformations` 审计 drop 原因。

因而 thinking replay 必须原样保留 block 和对话前缀，或者在编辑历史时有意识地删掉失效 block。API 接受请求不等于它接收了所有旧 thinking 状态。模型块也不是持久 memory：目标、权限、工具副作用和 artifact 仍要写在应用状态库。

## 24.4 工具、Computer Use 与长任务合同

Sonnet 5.5 的工具调用存在实际迁移差异：

- 不支持 `tool_choice=any` 或指定单个 tool；保留 `auto`/`none`。严格参数输出用 `strict=true` 或 Structured Outputs，避免用强制 tool choice 代替 schema。
- Claude API 和 Google Cloud 的 computer use 使用 `computer_toolset_20260801`；旧 `computer_20251124` 在这两个平台会报 400。Bedrock 当前文档仍接受旧类型，因此 provider 是 capability key 的一部分。
- Advisor tool 只接受兼容的模型组合，建议结果以 `advisor_redacted_result` 加密块返回，宿主不得假设可以读取建议正文。
- `compact-2026-09-04` on-demand compaction 由客户端选择触发时机，返回签名 compaction block；保留 tail 的 thinking block 仍需满足模型/前缀/回放合同。
- `inline-tools-2026-09-15` beta 允许通过会话中的 `tool_addition` 增加完整 tool schema 或切换 server-tool 版本，避免修改顶层 tools 列表并失去 prompt cache。

这给出一个通用工程原则：模型/API 迁移不只是 endpoint/model 字符串修改，还包括 typed block、工具策略、平台能力、错误语义、缓存前缀和状态回放的兼容性矩阵。

## 24.5 安全防护与 fallback：请求可能被分流，但类别不同

System Card 报告 cyber safeguard 有三阶段：内部 activation probe、轻量 classifier、独立 LLM classifier 与 probe 结果共同决定是否 block。普通 API/产品的 cyber block 在一些配置上可能 fallback 到 Sonnet 5；但类别之间差异很大：

| 风险类别 | 报告中的处理 |
|---|---|
| Cyber misuse | 阻断；多数接口可以 fallback 到 Sonnet 5，API 需显式 opt in |
| Chemical / biological | 阻断，无 fallback |
| 窄范围 AI/frontier-LLM development | 部分请求可能 fallback 到 Sonnet 5 |
| 常规武器/高当量爆炸物 | 阻断，无 fallback |
| Distillation / reasoning extraction | 阻断，无 fallback |

API refusal 仍是正常 HTTP 200 响应，`stop_reason=refusal`，并用 `stop_details.category` 表达类别。官方 server-side fallback 仅对 cyber 与 frontier-LLM 类拒绝重试；不能将所有拒绝都交给更大模型，否则可能绕过原策略。审计 trace 应保存 `requested_model`、`actual_model`、分类器结果、fallback reason、工具权限、执行副作用和独立 verifier。

## 24.6 System Card 数字必须绑定 harness

System Card 判定 Sonnet 5.5 达到 CB-1 与 Autonomy-1 阈值、未达到 CB-2/Autonomy-2；Autonomy threat model 1 的风险评估为 low。它是 Anthropic 按 RSP 评测得出的风险结论，不等于没有风险或独立验证。

两个代表性结果展示了评测口径的重要性：

| 评测 | 发布方结果 | 必须保留的条件 |
|---|---:|---|
| Terminal-Bench 4.0 | `70.6%` | 66 tasks，Claude Code `--bare`、max effort、5 trials、无外网、预缓存资源；1.2% requests/1.5% trials 触发 fallback |
| OSWorld 2.1 | `80.1%` partial / `43.5%` strict | 108 GUI tasks、1080p、最多 500 actions、5 runs、max effort、100K 后 server-side compaction；模型 grader 使用 Opus 4.8 |

Terminal-Bench 4.0 的更新还通过更长 timeout 和部分任务 RAM/CPU 自适应，降低旧版对资源限制、CLI 内存和 compaction 策略的混杂。这是 benchmark 设计的知识点，不是“新版本分数自然更可信”：仍需注明容器提供方与候选任务约束。

安全评测也要把 fallback 写入数据模型。Gray Swan IPI 的 k=1/10/15 `0.4%/2.7%/3.4%` 包含 classifier fallback；报告称约 11% rollout 转到 Sonnet 5，而成功攻击集中在没有 fallback 的 computer-use 场景。Coding Shade 中整体 ASR 也不能解释成 Sonnet 5.5 裸模型风险率，因为一部分请求由 Sonnet 5 实际作答。CoT controllability 的“较低”只被 Anthropic解释成更易监控的弱证据；当 monitor 看不到 reasoning，SHADE/LinuxArena 的行为又不同。它不能证明公开 reasoning 忠实映射内部状态，也不能转成通用欺骗概率。

## 24.7 面试回答框架

被问到“升级到 Sonnet 5.5 最需要改什么”，先答协议合同：`disabled` 改 `between_tools`，按 block type 解析与完整回放，复核 tool choice / computer toolset / advisor 兼容，并为 effort 重新做成本-质量 sweep。再答状态与治理：模型/前缀/账号三种 binding、refusal category-specific fallback、实际执行模型记录和独立验证。最后说明 System Card 的分数绑定工具、effort、权限、网络、compaction、fallback 和任务判分，不能等同裸模型能力或生产 SLO。

研究笔记：[Claude Sonnet 5.5 来源与证据账本](../../research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)。本章没有真实 API probe，也未独立复现 benchmark、下载权重或推断参数/架构/完整训练配方。
