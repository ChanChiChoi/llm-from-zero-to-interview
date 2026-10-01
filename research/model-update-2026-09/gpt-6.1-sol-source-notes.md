# GPT-6.1 Sol：模型合同、推理状态与长任务运行时

核验日期：2026-10-01。模型发现来自 Artificial Analysis；OpenAI 官方资料只用于核验已发现的 `gpt-6-1-sol`。

## 榜单身份

- Artificial Analysis `/zh` 经 `10.24.27.134:7890` 显式代理 HTTP 200，快照 `1,930,493` bytes / SHA-256 `4fa2574617fc125da6e0cf6a8575d5b81084baf22a134fb69a5ad450504c8c8d`，包含 `gpt-6-1-sol` 及 effort 变体。
- DataCurve DeepSWE HTTP 200，`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，没有精确 `mini_swe_agent_gpt_6_1_sol_*`，不迁移其他 GPT 的 Agent 分数。

## OpenAI 官方合同

官方模型页确认：`gpt-6.1-sol` 是 near-Astra、较低成本的复杂 coding、computer use 和 professional work 模型；`reasoning.effort` 支持 `low/medium/high/xhigh/max`，默认 `medium`，不支持 `none/minimal`；Responses API 支持 tool calling，Chat Completions 支持但不支持 tool calling；text/image input、text output；1,050,000 context、922,000 maximum input、128,000 maximum output；knowledge cutoff 2026-04-30；支持 web/file search、code interpreter、hosted shell、apply_patch、skills、computer use、MCP、tool search；EU residency 下 Fast mode 不可用。

官方 Reasoning 文档确认 GPT-6.1 Sol 支持 `standard/pro` mode 与 effort 分离；`reasoning.context=all_turns` 可复用同模型族中兼容的历史 opaque reasoning items，`current_turn` 则只让当前回合 reasoning 可用，不暴露原始 CoT。Agents 文档区分 Agents API、Agents SDK、Responses conversation 和 sandbox 资源。Compaction 文档规定 `context_management.compact_threshold` 触发 encrypted compaction item，stateless chaining 必须保留该 item，`previous_response_id` chaining 不应手工裁剪历史。

当前状态：**AA 单榜内容专题闭环 + OpenAI 官方模型/Reasoning/Agents/Compaction 合同**。未核验参数、架构、完整训练/后训练 recipe、完整权重、独立 benchmark、目标硬件、真实 endpoint probe 和生产 SLO。
