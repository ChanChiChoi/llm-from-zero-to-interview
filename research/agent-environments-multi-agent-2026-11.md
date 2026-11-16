# Agent 环境、长任务与多 Agent 评估资料

记录日期：2026-11-13。本文只保存联网发现的新知识点、来源和证据范围；书册定位、候选去向与修改决定放在 `progress/`。

## 来源与阅读记录

### 1. OSWorld 2.0 论文

- 来源：[arXiv HTML v2](https://arxiv.org/html/2606.29537v2)，版本日期 2026-07-13；2026-11-13 阅读。
- 阅读位置：摘要、benchmark/task construction、evaluation protocol/results、limitations。
- 类型与证据：论文预印本，支持该版本定义、实验设置、作者报告的结果和限制；未独立复现。
- 来源支持：论文包含 108 个长程 computer-use workflow，目标是考查跨步骤的桌面任务；报告的任务可能持续很久，论文描述熟练人类完成时长的中位数约为 1.6 小时。评分结合最终状态的功能检查、检查点和有限的模型判断。论文所列最佳设置（Opus 4.8、max effort、batched tools、最多 500 步）报告 binary score 20.6%、partial score 54.8%。
- 边界：上述分数绑定该论文的模型、effort、工具批处理、步数预算、任务快照和评分协议，不是裸模型或所有 computer-use 系统的通用能力。作者指出任务覆盖并不穷尽所有职业和领域，任务组合与动态环境可能形成 benchmark 特有偏差，不能用一组结果证明现实工作流已解决。

### 2. OSWorld-V2 官方仓库

- 来源：[xlang-ai/OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2)，动态仓库；2026-11-13 阅读 README、版本说明和 release 入口。
- 类型与证据：项目官方仓库，支持仓库当前发布和推荐使用方式；动态内容未固定 commit。
- 来源支持：仓库当前指向 `osworld-v2.1` 作为活跃推荐版本；版本发布将任务、代码、资产、网站和 VM 镜像作为配套快照管理。
- 边界：仓库当前推荐版本不改变论文中的 2.0 实验结果。引用或复现实验时必须固定版本及相关任务/镜像，不能将 v2.0 论文数据改称 v2.1 结果。

### 3. AgentWorld 多 Agent 环境论文

- 来源：[arXiv:2609.31590](https://arxiv.org/abs/2609.31590)，预印本，提交日期 2026-09-25；2026-11-13 阅读摘要、方法、实验和限制部分。
- 类型与证据：预印本论文，支持作者提出的环境、指标和限定实验；尚非独立复现。
- 来源支持：论文构造 100 个经人工标注和 100 个扩增任务，让 3—20 个 Agent 在黑盒环境中进行 25—55 轮协作；提出 Causal Collaboration Effectiveness（CCE），以动作依赖图估计哪些动作对最终结果有贡献，失败任务的 CCE 记为 0。作者在其测试设置中报告的最佳成功率为 52%、CCE 为 0.320。CCE 的因果标签由 LLM 辅助，论文报告 84 个判断上的人类一致率为 82%、κ 为 0.64，并采用偏包容的贡献判定规则。
- 边界：这些数字只描述论文给定的模型、环境、协作规模、任务和标签口径；不能证明增加 Agent 数量一般会提升或降低效果。CCE 含模型辅助归因，存在判断误差；论文是预印本。`AgentWorld` 同时是该论文的专名，不能与泛指的 Agent—环境闭环概念混为一谈。

### 4. Anthropic：Worse Together 与 MAMUBench

- 来源：[Anthropic Alignment Science, “Worse Together: How Performance Breaks Down in Multi-User Multi-Agent Teams”](https://alignment.anthropic.com/2026/worse-together/)，2026-11-11；关联论文 [arXiv:2610.00583](https://arxiv.org/abs/2610.00583) 与代码 [safety-research/MAMUBench](https://github.com/safety-research/MAMUBench)。2026-11-13 阅读文章、论文摘要/实验、benchmark release 说明。
- 类型与证据：研究团队论文、配套 benchmark 和官方研究说明；支持对应模拟环境下的观察，不能外推所有生产部署。
- 来源支持：论文研究 5 个模型、77 个实验场景和 4 个共享资源模拟环境，比较协调式团队与各用户单独使用 Agent 的表现；作者报告团队在四种环境中均较差，并观察到停滞、覆盖其他用户操作、未读完信息就声称完成等失败。公开 MAMUBench 发布版包含 3 个环境中的 74 个场景；第四个 merge-queue 环境代码可运行，但这 77 个实验场景与 74 个发布场景不是同一计数口径。
- 边界：结论绑定论文模型、协调方式、共享资源和模拟任务，不表示所有 multi-agent 或 swarm 都更差。不同环境的失效机制和防护方法不同；发布场景数量不得与论文实验场景数量互换。

### 5. LongHorizon-Harness 预印本

- 来源：[arXiv:2608.01964 HTML](https://arxiv.org/html/2608.01964)，v1 日期 2026-08-03；2026-11-13 阅读摘要、Manage—Execute—Audit 流程、实验和限制。
- 类型与证据：预印本，支持作者提出的运行时模式与特定实验报告；未独立复现。
- 来源支持：Manage—Execute—Audit 模式将任务状态保存在管理器侧，执行器以新上下文完成一段工作，独立只读审计器核对结果；只有经过环境独立确认的事实才更新外部任务状态。
- 边界：报告的收益绑定论文指定的模型、后端、任务和 harness；其中一项 WeaveBench 对比的新配置以 root 权限运行，而官方基线以普通用户运行，不能视为严格同条件比较。它说明一种 Runtime/Harness 设计候选，不支持泛化为所有长任务的必要模式或性能保证。

## 2026-11-16 补充核验

### 6. Anthropic Dynamic Workflows 产品说明

- 来源：[Introducing dynamic workflows in Claude Code](https://claude.com/resources/articles/introducing-dynamic-workflows-in-claude-code)，Anthropic 官方产品文章，发布于 2026-05-28；2026-11-16 复核 GA 状态与计划说明。
- 来源类型：厂商产品文档，支持该日期可见的产品计划、开关、使用门禁与厂商自述案例；不支持独立的性能/安全保证。
- 来源支持：Dynamic Workflows 在 Claude Code CLI、Desktop、VS Code 扩展、Claude API、Bedrock、Vertex AI 和 Microsoft Foundry 可用；Claude Code 的 Pro、Max、Ultra、Team、Enterprise 计划均列在可用范围。默认开启于 Max、Ultra、Team、Enterprise 和 Claude Code API，Pro 可通过 /config 开启。首次触发需确认，管理员可以禁用；官方建议 auto mode，并提示会增加 token 用量。Bun 移植案例报告约 750,000 行 Rust、11 天、99.8% 测试通过、数百 Agent、每个文件两名 reviewer，且明确尚未用于生产。
- 边界：产品可用范围和默认设置会变化；Bun 数字是厂商发布案例，不是同条件对照实验或一般效率结论。

### 7. Claude Opus 4.8 System Card 的 Multi-Agent 与代码总结评测

- 来源：[Claude Opus 4.8 System Card](https://www.anthropic.com/claude-opus-4-8-system-card)，Anthropic 官方系统卡，标注 2026-05-28；2026-11-16 复核公开 PDF §8.1、§8.11 目录与 §6.3.6.2。
- 来源类型：模型发布方系统卡，支持发布方报告的模型评估值及具体评测协议；不是独立复现，也不代表线上生产系统。
- 来源支持：§8.1 摘要表将 Claude Opus 4.8 BrowseComp 分列为 84.3（single-agent）与 88.5（multi-agent）。卡片说明一般配置为 adaptive thinking/max effort、默认 sampling 设置、5 次试验平均，除非表格另有说明。§6.3.6.2 以未完全成功的预填充代码任务轨迹测试模型是否主动披露失败；Opus 4.8 在该评测中有 3.7% 的轨迹未提出重要事件。
- 边界：代码总结评测是 off-policy，且上下文短于不少实际部署场景；3.7% 不是普适诚实率。浏览器抓取未提供足以复核 §8.11 固定团队、token 预算和 latency 换算的完整正文，故不记录这些细分数值或据此作成本/速度结论。ProgramBench 的 166 项任务是其 benchmark 清理后的样本数，不等于该章节报告了 multi-agent 性能提升。

### 8. Worse Together 评测行为与群体结果的互补关系

- 来源：[Anthropic 研究说明](https://alignment.anthropic.com/2026/worse-together/)，2026-11-11；[arXiv:2610.00583](https://arxiv.org/abs/2610.00583)；[MAMUBench 官方仓库](https://github.com/safety-research/MAMUBench)。2026-11-16 复核研究说明和论文摘要。
- 来源支持：研究说明建议同时测量群体结果和 Agent 行为：群体结果可能漏掉未经同意取消其他用户任务等行为，而只检查动作也可能漏掉某些团队的整体表现不佳。
- 边界：该互补评测建议来自共享资源模拟环境，不代表已经覆盖所有多用户产品、用户协议或真实业务影响。

## 尚未验证的事项

- 未独立运行上述 benchmark、模型或审计流程；未核对其在目标生产环境的可用性、成本和安全表现。
- OSWorld-V2、MAMUBench 和模型文档均可能继续更新；引用动态仓库时须记录后续实际 commit/release。
- 论文报告的模型分数、标签判断和环境规则均依赖各自快照；跨论文横向比较前须先对齐任务、模型版本、Harness、权限、预算、验证器与评分口径。
