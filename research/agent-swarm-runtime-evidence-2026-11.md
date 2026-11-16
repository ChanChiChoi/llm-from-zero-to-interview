# Agent Swarm 产品与多 Agent runtime 来源记录

记录日期：2026-11-16。本文仅保存公开来源、版本/访问日期、来源支持的事实与边界；章节取舍、正文修改和进度交接只写入 `progress/`。

## 1. Moonshot：Kimi K2.6 与 Agent Swarm

### 来源和阅读范围

- [Kimi K2.6 官方博客](https://www.kimi.com/en/blog/kimi-k2-6)：Moonshot 官方发布页；2026-11-16 阅读 “Agent Swarms, Elevated”“Long-Horizon Coding” 与 footnotes。页面正文未显示可确认的发布日期，本记录固定访问日期，不推断发布日期。
- [Kimi K2.6 Hugging Face 模型卡](https://huggingface.co/moonshotai/Kimi-K2.6)：Moonshot 维护的动态模型卡；2026-11-16 阅读 `preserve_thinking` 与 multi-step tool call 说明。
- [Kimi K2.6 config.json 固定 revision](https://huggingface.co/moonshotai/Kimi-K2.6/resolve/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json)：模型仓库固定 revision `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`；该 revision 是 2026-11-16 读取 `main` 时解析到的配置文件提交。
- 类型与证据：发布方博客及发布方模型卡/config；source-backed 于公开产品描述和配置字段，不是独立复现。

### 来源支持的事实

- 官方博客把 Agent Swarm 描述为可横向扩展至 300 个 sub-agent 和 4,000 个协调步骤；同一段称其较 K2.5 的 100 个 sub-agent/1,500 步有扩展。这是发布方对产品编排能力的说明。
- 博客另外报告了两个长时程代码案例：在 Mac 上部署并优化 Qwen3.5-0.8B 的案例用了 4,000 多次 tool call、超过 12 小时和 14 次迭代；`exchange-core` 案例运行 13 小时、超过 1,000 次 tool call，并修改 4,000 多行代码。博客把这些作为案例展示，没有提供可用于控制变量比较的完整独立 benchmark 协议。
- 固定 config 披露 61 层、384 个 routed experts、每 token 选择 8 个 expert、另有 1 个 shared expert。该配置描述模型内部 MoE 字段，与官方博客描述的外部 sub-agent 协作规模属于不同层次；这是对两份公开材料的概念区分，不是模型训练或运行架构的独立复现。
- 当前模型卡写明 K2.6 支持 `preserve_thinking`，该选项默认关闭，并以 `reasoning_content` 展示跨轮思考内容保留方式。该合同只适用于所读到的模型/API说明；第三方适配器或后续服务版本需另外核对。

### 证据边界

- 300 sub-agent/4,000 coordinated steps 及长时程案例均为 Moonshot 发布方描述；不能推导所有示例都使用了最大 swarm、不能作为独立速度/质量实验，也不能外推生产完成率。
- tool call 数、运行时长和代码行数不等于 sub-agent 数或独立并行任务数；报告时分别记录其统计对象。
- config 的字段只支持固定 revision 中公开的配置值，不揭示完整训练 recipe、生产服务实现或实际激活路径。

## 2. xAI：Grok 4.20 Multi-agent runtime

### 来源和阅读范围

- [Grok 4.20 模型文档](https://docs.x.ai/developers/models/grok-4.20)：xAI 动态官方文档；2026-11-16 阅读模型 ID 与能力字段。
- [xAI Multi-agent 文档](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md)：xAI 动态官方文档；2026-11-16 阅读 beta 声明、运行方式、工具、输出行为及 agent-count 配置。
- 类型与证据：供应商公开 API/runtime 文档，source-backed 于文档所述接口；未执行 provider endpoint，也未取得 xAI 专属技术报告。

### 来源支持的事实

- 文档把 `grok-4.20-multi-agent` 描述为 Realtime Multi-agent Research beta。请求可以协调专门 Agent 搜集资料、交叉核对，再由 leader agent 汇总。该说明是 provider 提供的运行时合同，不揭示隐藏的完整执行 trace。
- 文档列出 4-agent 与 16-agent 两种设置。SDK 可直接设置 `agent_count`；OpenAI-compatible REST/SDK 中，`reasoning.effort=low/medium` 映射到 4 agents，`high/xhigh` 映射到 16 agents。这里的 effort 映射是这个 multi-agent endpoint 的编排选择，不是一般模型推理预算或 Transformer 内部计算深度。
- 默认只返回 tool call 和 leader final response；使用 `use_encrypted_content` 时才随响应包含加密的 sub-agent 状态。内置工具由服务端运行并产生额外费用。

### 证据边界

- 文档明确标记 beta，API 和行为可能变化；模型 ID、配置与输出字段须按访问日期复核。
- 文档不证明增加 Agent 数量会提高正确率、降低总体延迟或改善生产效果，也没有给出可独立复现的 Grok 4.20 专属 benchmark。
- encrypted state 的存在不代表调用方可读取、修改或重新排序隐藏 reasoning；没有实际 provider 调用，本记录不声称验证 endpoint 行为。
