# Kimi K3.1 Pro 来源与证据记录

记录日期：2026-11-13  
核验范围：Kimi K3.1 Pro 的官方发布信息、长任务/Agent 评测口径、公开模型卡和多轮工具调用合同。此记录只保存来源支持的新知识及其边界；书册映射、正文修改和验收状态记录在 `progress/`。

## 来源与阅读位置

### 1. Kimi K3.1 Pro 官方技术博客

- URL: <https://www.kimi.ai/blog/kimi-k3-1-pro>
- 发布日期：官方研究博客目录列为 2026-10-24；页面于 2026-11-13 访问。
- 阅读位置：开篇模型定位与后训练说明；Coding 部分 Kimi Code Bench 3.0 Long-horizon；Availability；Coding/Agentic benchmark footnotes。
- 来源类型/证据等级：厂商发布方技术博客，source-backed。
- 支持的事实：官方将 K3.1 Pro 描述为基于 K3 后训练的版本，并称其使用更多后训练计算、跨数百下游领域联合 RL；列举 RL framework、长上下文推理、训练效率、sandbox infrastructure、rollout scheduling 和环境质量检查作为栈级改进方向。官方报告 Kimi Code Bench 3.0 Long-horizon 内部评测为 K3.1 Pro 61.2%、K3 55.95%，并将其描述为覆盖多步执行、工具使用和端到端问题求解。博客还报告 Terminal-Bench 4.0 上 K3.1 Pro 为 65.7%，评测使用 Claude Code harness；Kimi Code Bench 的 harness 和其他模型 effort 条件见脚注。
- 证据边界：上述训练和系统描述属于发布方披露，不是完整 recipe、独立复现或生产指标。Kimi Code Bench 为内部 benchmark；公开材料没有给出足够细节以重建全部任务集、运行记录和统计不确定性。不同公开 benchmark 的模型、harness、硬件、effort、fallback、版本和分数口径并不统一，不应拼成统一排名。报告的相对改进不能推广到未测任务或所有 harness。

### 2. Kimi 官方 K3.1 Pro 模型卡

- URL: <https://huggingface.co/moonshotai/Kimi-K3.1-Pro>
- 版本/快照：模型卡 `main` 动态页面，2026-11-13 访问；未固定 commit，后续引用须重新核 revision。
- 阅读位置：Model Introduction、Model Summary、Evaluation Results、Deployment and Availability、Model Usage、License。
- 来源类型/证据等级：发布方维护的公开模型卡，公开模型配置为 source-backed。
- 支持的事实：模型卡称 K3.1 Pro 从 K3 后训练而来；披露 MoE、总参数量 2.8T、激活参数 104B、93 层、69 KDA + 24 Gated DSA、1M context、MXFP4 weights/MXFP8 activations 和量化感知训练。模型卡特别说明，相对 K3，原 Gated MLA 层被 Gated DSA 替换。模型卡给出模型/API 使用入口，并说明多轮对话和工具调用需将完整 assistant 消息原样传回 `messages`，包括 `reasoning_content` 和 `tool_calls`，不能只回传 `content`。
- 证据边界：模型卡字段是发布方披露，不代表已在本地加载、运行或复现；页面当前 `main` 未固定 revision。架构字段不支持推断未披露训练数据、完整优化目标、全部 serving kernel 或独立性能。消息回放条款是该模型公开接口合同，不应泛化成所有 provider 的协议。

### 3. Kimi Code 官方模型配置文档

- URL: <https://www.kimi.com/code/docs/en/kimi-code/models>
- 版本/快照：动态官方文档，2026-11-13 访问。
- 阅读位置：Model Overview、Usage recommendations、model ID 表格。
- 来源类型/证据等级：官方产品文档，source-backed。
- 支持的事实：文档将 `k3.1-pro` 与 `k3.1-pro-highspeed` 列为不同模型 ID；建议保持 thinking 启用，并建议在 K3.1 Pro、K3/K2.8 或 K2.7 Code 会话间切换时新建会话。文档将会话中途切换与输出质量下降、缓存命中率下降联系起来，并推荐使用其原生 API 以获得较稳定输出。
- 证据边界：这是 Kimi Code 的动态产品建议，不是独立对照实验，也不证明所有第三方 harness 都会出现相同退化。兼容层和具体 API 的行为需要按版本实测。

### 4. Kimi 官方 K3 原始仓库

- URL: <https://github.com/MoonshotAI/Kimi-K3>
- 版本/快照：官方公开仓库，页面于 2026-11-13 访问；如引用具体配置或代码必须固定 commit/revision。
- 阅读位置：README 中 Model Usage、Evaluation Notes、Deployment；`k3_tech_report.pdf` 入口。
- 来源类型/证据等级：官方模型仓库与技术报告入口，source-backed。
- 支持的事实：仓库将 K3 多轮/工具使用的历史回传合同写为保留完整 assistant 消息，包括 `reasoning_content` 与 `tool_calls`；仓库同时提供技术报告、权重和部署入口。K3.1 模型卡明确将其描述为后训练版本，并列出相对 K3 的注意力层变化。
- 证据边界：K3 仓库中的模型使用说明不自动代表 K3.1 的所有托管 API 和第三方适配器；仓库现状是动态内容。

## 结论范围与待核验项

- K3.1 Pro 可作为长任务 Agent 案例，支持讨论“后训练收益同时依赖模型、评测 harness、环境与运行协议”的证据链；发布材料没有公开完整 RL recipe。
- “61.2% vs 55.95%”仅按 Kimi 官方内部 Kimi Code Bench 3.0 Long-horizon 报告值引用；不得写成独立复现、普遍提升或跨 harness 保证。若后续需要比较，应先核任务快照、模型 revision、harness revision、预算、effort、评判器和失败处理是否一致。
- 多轮消息回放与会话切换建议都绑定 Kimi 的模型/产品合同；不将其描述成通用 reasoning 协议或所有模型都必须回传隐藏思维。
- 公开模型卡未固定读取 revision；技术报告与配置的后续引用需补固定 commit、实际阅读位置及差异。未独立执行权重加载、接口请求或 benchmark。

