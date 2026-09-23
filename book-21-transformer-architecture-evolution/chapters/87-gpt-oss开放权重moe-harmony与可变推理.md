# GPT-OSS：开放权重 MoE、Harmony 协议与可变推理

> 核验日期：2026-09-22。`gpt-oss-120b` 与 `gpt-oss-20b` 已在 Artificial Analysis 中作为 OpenAI 候选出现；本章技术事实主要来自 OpenAI Model Card、官方 gpt-oss 仓库、Harmony 仓库和 developers.openai.com Cookbook。Model Card 的 benchmark、训练时长和部署数字均保留为发布方口径。

## 87.1 先确定研究对象

本章研究的是同一 gpt-oss 家族的两个尺寸，而不是两个完全不同的模型：

| 模型 | 总参数 | active/token | 层数 | 专家数 | 每 token 专家数 |
|---|---:|---:|---:|---:|---:|
| gpt-oss-120b | 116.8B | 5.13B | 36 | 128 | 4 |
| gpt-oss-20b | 20.9B | 3.61B | 24 | 32 | 4 |

`active parameters` 只描述一个 token 当前路径实际计算的参数口径。它不能直接替代总权重显存，因为服务端仍需保存或分片管理所有专家，还要加上 KV cache、dispatch、通信 buffer、workspace 和并发请求。

Artificial Analysis 还把两个模型分别按 `high` 推理配置列出，但 `high` 不是第三个权重版本。比较时至少固定四个维度：model revision、reasoning level、tool harness 和 verifier。

## 87.2 架构：稀疏计算与全局交换的折中

### MoE 路由

对 token 表示 `x`，router 产生专家分数：

```text
s = W_router x
I = TopK(s, k=4)
p_i = softmax(s_I)_i
y = sum(i in I) p_i * Expert_i(x)
```

这个式子只表达路由语义，不代表生产实现只有这几个算子。真实服务还要处理 token-to-expert dispatch、容量限制、padding、负载倾斜、跨卡 all-to-all 和专家结果回收。

面试中应把三个问题分开回答：

1. 为什么 active compute 低：每个 token 只激活四个专家。
2. 为什么总显存不等于 active compute：所有专家权重仍属于 checkpoint 或分片存储。
3. 为什么低 active compute 不保证低延迟：router、dispatch、通信和最慢专家会决定尾延迟。

### 交替 sliding/full attention

gpt-oss 的注意力层交替使用 128-token banded/sliding-window attention 和 fully dense attention。每层有 64 个 query heads、8 个 KV heads、head dimension 64，使用 GQA；位置编码采用 RoPE，dense 层使用 YaRN 扩展到 131,072 token。

局部层的近似访问成本可写成：

```text
local layer:  O(L * w),  w = 128
dense layer:  O(L^2)
```

这不是说模型整体复杂度简单地等于两者平均值。层间表示会传播，dense 层的 KV、batch、并行度和 kernel 仍影响实际成本。更准确的工程问题是：局部层节省了多少带宽和计算，dense 层是否足以恢复远程检索，以及长上下文任务的召回是否发生层相关退化。

### learned attention bias

官方 Model Card 还描述了每个 attention head 的 learned bias：它允许 softmax 分母中的注意力质量接近“一个 token 都不看”。这可以帮助模型跳过无用历史，但不能把该描述扩写成某个外部 attention sink 论文的完整复现。

## 87.3 MXFP4：权重格式与 serving 后端共同决定可用性

OpenAI 公开的关键工程点是：MoE 权重使用 post-training MXFP4，平均约 4.25 bits/parameter；MoE 权重占总参数 90% 以上。它带来两层效果：

```text
checkpoint bytes 下降
        -> 权重驻留门槛下降
        -> 单卡部署目标成为可能
        -> 但 kernel / dequant / dispatch 成为新的性能约束
```

官方仓库同时提供非优化 PyTorch 参考实现和更接近部署的 Triton 路径。前者需要更多设备，后者利用支持 MXFP4 的 MoE kernel 并针对单个 80GB GPU 做了优化。这个差异很适合面试追问：模型“公开权重”不等于任意后端都能以相同吞吐和延迟服务。

不要把 MXFP4 描述成“先用 BF16 训练、部署时随手量化”的无损外挂。Model Card 明确发布评测也使用该量化；因此质量、显存、kernel 和 benchmark 条件需要一起记录。当前公开资料没有完整的 kernel 源码消融、跨 GPU profiling、专家负载 p99 或线上 acceptance rate。

## 87.4 Harmony：让推理、工具和可见答案成为不同 channel

普通文本 prompt 只给模型一串 token；Harmony 则把角色、channel 和 recipient 变成结构化 response format。可以把一轮 Agent 轨迹抽象为：

```text
system/developer/user
        -> assistant:analysis
        -> assistant:commentary to=tool
        -> tool result
        -> assistant:analysis
        -> assistant:final
```

关键规则包括：

- instruction hierarchy：System > Developer > User > Assistant > Tool；
- `analysis` 用于 reasoning，`commentary` 可承载工具前置说明或中间进度，`final` 是用户可见答案；
- tool recipient、函数名、参数和结果具有明确边界；
- 渲染和解析应保持 token 序列可逆；官方 Harmony 实现把核心放在 Rust，并提供 Python 绑定；
- 多轮继续时，不应把历史 assistant reasoning traces 原样持续回灌。

因此 Harmony 不是“换几个特殊分隔符”。它同时影响训练数据格式、采样停止条件、工具 parser、可见性策略和下一轮状态回放。

### 工具调用责任链

```text
model output
  -> channel/recipient parser
  -> JSON/schema validation
  -> permission and confirmation gate
  -> sandboxed executor
  -> timeout/retry/cancel handling
  -> tool result replay
  -> final artifact verifier
```

模型能生成合法 function call，不等于工具被授权、执行成功或最终 artifact 正确。这个边界适用于 browser、Python 和 developer-defined functions。

## 87.5 Variable-effort reasoning：同一权重的 test-time scaling

gpt-oss 支持 `low`、`medium`、`high` 三种 reasoning level，官方示例通过 system prompt 中的 `Reasoning: high` 等关键词配置。公开评测显示，档位升高时平均 CoT 长度通常增加，部分任务的准确率提高，代价是更多 output tokens、延迟和费用。

可以用一个教学预算模型表达方向：

```text
unit_success_cost
  = request_cost(reasoning_tokens + answer_tokens + tool_tokens)
    / verified_success
```

这个式子不是 OpenAI 的计费公式，只是提醒面试者不要只比较 accuracy。真正实验要固定 prompt、sampling、工具、硬件、最大输出和 verifier，再比较：

- accuracy / task success；
- CoT 或 reasoning token 长度；
- TTFT、TPOT、端到端延迟；
- 工具调用次数与失败率；
- 单位成功成本。

`low/medium/high` 是配置，不是三种架构；也不能从档位名称反推内部 RL 目标或固定 token budget。

### 87.5.1 Provider 兼容性：从“能生成”到“可回放”

OpenAI 的 gpt-oss 实现验证指南把 provider 验收拆成多层，而不是只检查 HTTP 200 或单轮文本质量。首先，输入必须正确渲染为 Harmony；角色、channel、recipient、stop token 或工具 schema 的小错误，可能只表现为 function calling 变差。其次，tool call 可能发生在 CoT 内部，下一次 sampling 必须收到此前 raw CoT、tool call 和 tool output；只保存用户看得到的 final 会破坏状态。

Responses API 是官方推荐的承载形状。一个 reasoning item 可以同时包含用户可展示的 summary 和不可直接展示的 raw CoT：

```json
{
  "type": "reasoning",
  "summary": [],
  "content": [
    {"type": "reasoning_text", "text": "..."}
  ]
}
```

流式 raw CoT 使用 `response.reasoning_text.delta` 和 `response.reasoning_text.done`。应用要按 `item_id`、`output_index`、`content_index` 做顺序和去重校验，再把完整 reasoning item 按协议重放到下一轮；不能把 delta 直接拼到用户可见答案中。

Chat Completions 没有公开的通用 raw CoT 字段。对于提供 gpt-oss 的兼容层，官方建议把 raw CoT 放在 `reasoning` 字段，并让流式 delta 也使用 `reasoning`；这是 provider 兼容约定，不能反写成 OpenAI 托管模型的通用 API 契约。

### 87.5.2 raw CoT 的安全与生命周期

raw CoT 可能包含有害内容或泄露 developer instructions，因此不能直接展示给终端用户。产品如果需要解释，应展示经过审查和过滤的 summary；raw CoT 只进入受控的 provider、调试、解释性研究或后续状态回放链路。

回放时还要依赖对话状态：后续 sampling 产生 `final` 后，可以丢弃之前的 `analysis`；如果 assistant 最后一条是 tool call，则应继续保留到上一个 `final` 之前的 analysis；发往 `commentary` channel 的 function call 可以按协议保留。raw CoT、tool call、tool result 和 final 应带同一轮 lineage，重试时还要防止重复执行外部副作用。

### 87.5.3 兼容性 smoke test 与质量 eval 分账

官方仓库的 `compatibility-test` Node.js 测试用于检查 Responses/Chat Completions 的 function calling 和 API shape。invalid request 数量、工具选择、参数形状、结果回灌和 streaming 事件是协议门禁；官方指南把 0 invalid requests 且 `pass@k`、`pass^k` 均超过 90% 作为“很可能正确”的信号，但明确它不是完整兼容性或精度证明。`jsonl` 响应、单用例 `-n 1` 和 debug 日志适合定位 provider 差异。

质量层再运行同一仓库的 AIME、GPQA、HealthBench harness：AIME 每题 16 次、GPQA 每题 8 次、HealthBench 每题 1 次，并固定 base URL、model、reasoning effort、tool、sampler 和 verifier。可以把证据链画成：

```text
Harmony/render + API shape
        -> tool-call smoke test
        -> raw CoT/state replay
        -> AIME/GPQA/HealthBench
        -> kernel/precision/hardware profiling
        -> production acceptance
```

因此“tool call 看起来正确”不等于 MXFP4 kernel、MoE dispatch、模型质量或生产副作用都已验收；反向地，某项 benchmark 接近发布方结果也不能证明状态回放和 API 契约正确。

## 87.6 评测归因：模型分数不是系统分数

OpenAI Model Card 报告了不同 reasoning level、with/without tools、coding harness 和 safety setting 下的结果。即使同一模型在 SWE-bench 上得到一个数字，也至少有以下变量：

```text
model revision
 + tokenizer / Harmony format
 + reasoning level
 + tool set and permissions
 + prompt / system message
 + harness and retry
 + environment / hardware
 + verifier / judge
```

因此 OpenAI Model Card 的发布方结果、Artificial Analysis Intelligence Index 和 DataCurve DeepSWE 必须分栏。当前两个榜单中 Artificial Analysis 有 gpt-oss-120b/20b 精确条目，DataCurve 没有精确 gpt-oss 行，不能借用 GPT-5.x、Codex 或其他模型的 DeepSWE 分数。

## 87.7 开放权重改变安全边界

OpenAI 将该文档称为 model card 而不是 system card，原因是公开权重可以被下游微调，系统级安全措施不再完全由发布方控制。公开安全主线包括 deliberative alignment、拒答、越狱鲁棒性、instruction hierarchy，以及默认和 adversarial fine-tuning 条件下的 Preparedness 评估。

下游系统还要自己负责：

- 微调数据的安全和污染检查；
- browser/Python/function 工具的最小权限；
- 沙箱、网络出口、文件系统和密钥隔离；
- Harmony parser、schema、超时、取消和审计；
- 最终 artifact 的独立验证。

发布方 safety result 不能替代部署方安全边界，也不能从“Apache 2.0”推导出无约束的生产适用性。

## 87.8 面试速答

**问：为什么 120B 只有 5.1B active 仍然可能需要大显存？**

答：active 只表示每个 token 的路径计算；所有专家权重仍需保存或分片，另有 KV cache、dispatch、通信 buffer、workspace 和并发请求。

**问：为什么不全用 sliding-window attention？**

答：纯局部窗口对远程精确检索和跨段信息交换不友好；交替 dense 层提供全局交换，局部层负责更低成本的连续更新，最终要用长程召回和端到端指标验证。

**问：Harmony 比 ChatML 多解决了什么？**

答：它把角色层级、可见 channel、tool recipient、函数结构和渲染/解析契约一起纳入模型训练和推理；它不是只增加几个分隔 token。

**问：`Reasoning: high` 是否等于另一个模型？**

答：不是。同一权重的推理配置，主要改变 test-time compute 和 CoT 长度；要用质量、延迟、token 和单位成功成本一起比较。

**问：模型输出工具调用后，为什么宿主还要验证？**

答：模型输出是意图，不是授权。schema、权限、确认、沙箱、超时、真实回执和 artifact verifier 必须由宿主完成。

## 87.9 证据边界与相关阅读

本章能确认 gpt-oss 的公开架构、两个尺寸、MXFP4、Harmony、可变推理档位、工具训练方向、发布方评测和开放权重安全边界；不能确认完整训练数据配比、optimizer、后训练奖励函数、全部 kernel、线上吞吐/接受率或独立 benchmark 复现。

详细快照、来源和待核验清单见 [`gpt-oss-source-notes.md`](../../research/model-update-2026-09/gpt-oss-source-notes.md)。相关基础章节包括第 5 章 MoE FFN、第 19 章滑动窗口/稀疏注意力、第 20 章 attention kernel、第 21 章 MoE 路线、第 66 章 active parameters、第 80 章 MTP，以及第 49 章 Agentic Model Architecture。
