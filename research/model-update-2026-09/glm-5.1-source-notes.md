# GLM-5.1：长周期 Agent 工程、过程质量与 DSA 配置资料摘记

首次核验日期：2026-09-20；官方博客补证日期：2026-09-21；当前时点复验日期：2026-09-22。本笔记只把已经出现在 Artificial Analysis 的 GLM-5.1 作为模型锚点，再沿 Z.ai 官方模型文档、官方模型卡/配置、发布说明和 API 能力文档追踪面试相关技术。GLM-5.1 的公开资料没有给出独立的完整技术报告；模型卡链接的是 GLM-5 技术报告，因此 GLM-5 的训练和架构结论不能自动改名为 GLM-5.1 专属事实。

## 1. 榜单锚点与证据边界

### 1.1 Artificial Analysis

- [Artificial Analysis GLM-5.1](https://artificialanalysis.ai/models/glm-5-1) 是本轮模型发现入口。2026-09-20 通过 `10.24.27.134:7890`、`10.237.126.170:1234` 和 `10.24.27.134:8098` 抓取的历史快照为 `3,935,095` bytes，SHA-256 为 `f6b1ca673b777602013f13eb348a7c48773120e13684eadcbd7220b50b6c137d`；当时页面第三方字段约为 `39.9 tokens/s` 和 `$1.20/$4.40` 每百万 input/output token。
- 2026-09-22 重新通过三条代理取得详情页，三份内容一致；当前快照 `/tmp/aa-glm51-current-1234.html` 为 `3,971,543` bytes，SHA-256 为 `c806874d4eb235658b04ba6f527e1ca1db30b2c5e149e8690cd428fd5d2614c9`。当前页面仍为 `GLM-5.1 (Reasoning)`、release `April 2026`、`200K` context；第三方当前测量为 Intelligence Index `26.0585912980095`、median output speed `37.1922485381327 tokens/s`、cost per Intelligence Index task `0.9217822401466147`。这些变化按 provider/采集时点漂移处理，不解释为模型 revision 或训练变化。
- 页面标题为 `GLM-5.1 (Reasoning)`，release 字段为 2026 年 4 月；页面同时存在 `glm-5-1-non-reasoning` 配置。它们应归并到同一个基础模型，再保留 reasoning/non-reasoning 运行配置，不应当作两个 checkpoint。
- 页面第三方字段包括约 `744B` total、`40B` active、`200K` context、Intelligence Index `26.0585912980095`、约 `39.9 tokens/s` 输出速度和约 `$1.20/$4.40` 每百万 input/output token。这些字段是 Artificial Analysis 的目录/provider 测量，不能替代官方参数、训练报告或裸模型能力结论。

### 1.2 DataCurve DeepSWE

- [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 2026-09-22 当前快照 `/tmp/dataswe-fresh-7890-20260922.html` 为 `268,571` bytes，SHA-256 为 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；与此前快照一致。
- 当前快照没有精确的 `mini_swe_agent_glm_5_1_*` 行。因此本轮不记录 GLM-5.1 的 Pass@1、Pass@4、成本、输出 token 或 Agent steps，也不迁移 GLM-5/5.2/5.3/5.3-Flash 的 DeepSWE 结果。
- 观测对象仍应写成：

```text
result = F(model_revision, effort, mini_swe_agent, tools,
           task_set, environment, verifier, timeout, provider)
```

当前状态是 **AA 单榜资料级闭环**，不是双榜 Agent 评测闭环。

## 2. 官方模型卡与配置

主要来源：

- [Z.ai GLM-5.1 模型文档](https://docs.z.ai/guides/llm/glm-5.1)：Markdown 快照 `/tmp/glm51-zai-md-20260920.out`，`19,347` bytes，SHA-256 `69958d7d95452d3853903524612a0c6a83f368c79417a6889b9c531f45ab3d01`。
- [Hugging Face GLM-5.1 README](https://huggingface.co/zai-org/GLM-5.1/raw/main/README.md)：快照 `/tmp/glm51-hf-readme-7890-20260920.out`，`10,751` bytes，SHA-256 `2cb148b732f596716f8c9366edf072e35746de6d06fe69c92301ddb492557077`。
- [Hugging Face GLM-5.1 config.json](https://huggingface.co/zai-org/GLM-5.1/raw/main/config.json)：快照 `/tmp/glm51-hf-config-7890-20260920.out`，`1,379` bytes，SHA-256 `726e45e28d1be1636a5834048a4101a2cfe384819ad2ebe08d41f6fe04656526`。
- [Z.ai 文档索引](https://docs.z.ai/llms.txt)：快照 `8,784` bytes，SHA-256 `8cabef678d8bd42f9eec02084af2b5028a62328eff8780e12b04e88ece2bdd2d`。

官方 GLM-5.1 文档公开了以下产品字段：

| 字段 | 官方公开值 | 证据边界 |
|---|---:|---|
| 定位 | flagship foundation model，面向 long-horizon tasks | Z.ai 模型文档 |
| 输入/输出 | text → text | Z.ai 模型文档 |
| context | 200K | Z.ai 模型文档 |
| 最大输出 | 128K tokens | Z.ai 模型文档 |
| API model ID | `glm-5.1` | Quick Start 示例 |
| 开源许可 | MIT | Hugging Face README front matter |
| 本地部署入口 | SGLang、vLLM、xLLM、Transformers、KTransformers | Hugging Face README |

固定配置还公开了这些实现字段：

- `architectures = ["GlmMoeDsaForCausalLM"]`，`model_type = "glm_moe_dsa"`，`dtype = "bfloat16"`。
- `num_hidden_layers = 78`，`first_k_dense_replace = 3`；配置的前 3 层使用 dense 替换路径，其余层使用 MoE 配置。这里的字段说明实现布局，不自动等于完整的发布方层排布叙述。
- `n_routed_experts = 256`、`n_shared_experts = 1`、`num_experts_per_tok = 8`；`moe_intermediate_size = 2048`。
- `hidden_size = 6144`、`num_attention_heads = 64`、`num_key_value_heads = 64`；`q_lora_rank = 2048`、`kv_lora_rank = 512`、`qk_nope_head_dim = 192`、`qk_rope_head_dim = 64`、`v_head_dim = 256`。
- DSA 实现相关索引字段为 `index_n_heads = 32`、`index_head_dim = 128`、`index_topk = 2048`，并启用 `indexer_rope_interleave`；`max_position_embeddings = 202752`、`rope_theta = 1000000`。
- 配置还有 `num_nextn_predict_layers = 1` 和 `transformers_version = "5.4.0"`。这只能说明公开实现接口包含一个 next-token prediction 相关字段，不能在没有训练/解码说明时把它扩写成完整 MTP 方案。

上述配置与 GLM-5 的 DSA/MoE 路线高度相近，但“字段相近”不是把 GLM-5 技术报告自动迁移到 GLM-5.1 的理由。GLM-5.1 自己的配置支持结构字段核验；总参数、激活参数、完整 indexer loss、训练 recipe 和生产 kernel 仍分别需要对应证据。

## 3. GLM-5.1 的新技术主线：从一次生成到长周期工程闭环

Z.ai 将 GLM-5.1 的关键进步描述为长周期 Agent 工程能力，而不是单轮代码生成能力。官方文档和模型卡反复出现以下闭环：

```text
目标/规划
  -> 分步执行与工具调用
  -> 实验、读取结果、识别阻塞
  -> 调整策略并再次执行
  -> 测试、修复、优化
  -> 交付可验收 artifact
```

官方描述称模型可以在一个任务上连续、自主工作最长约 8 小时，跨越规划、执行、测试、修复、迭代优化和交付；模型卡进一步说它能在数百轮、数千次工具调用中反复审视推理和修订策略。这里最适合的面试解释是“长任务控制与验证问题”，不是“context window 变大就自然得到 8 小时 Agent”。

要把这个能力拆开，至少需要测量：

1. **目标保持**：长轨迹中原始目标、约束和停止条件是否仍可恢复；
2. **策略更新**：实验失败后是否能改变策略，而不是重复同一动作；
3. **错误累积**：工具错误、环境变化和错误假设是否被识别并隔离；
4. **过程质量**：中间步骤是否有可验证证据，而不是只看最后一段文本；
5. **交付质量**：测试、性能指标、artifact digest 和权限审计是否通过外部 verifier。

这也解释了为什么“长上下文”不是充分条件：上下文只是承载历史，长期 Agent 还需要状态压缩/回放、预算、工具执行、失败恢复和独立验收。

## 4. 过程质量、SFT/RL 与评测口径

Z.ai 的 2026-04-07 release note 对 GLM-5.1 的描述包括：multi-turn SFT、RL 和 process-quality evaluation framework，用于提升 extended-task 的 stability、consistency 和 tool use。公开说明没有给出 RL 的具体算法、奖励模型、轨迹过滤、优势估计或 verifier 实现，因此只能记录为后训练方向，不能改写成 GRPO、PPO、RLVR 或某个具体损失函数。

模型卡/官方文档的评测数字包括：

- SWE-Bench Pro：`58.4`；
- Linux desktop 案例：官方称完成 `655` 次迭代，并将向量数据库吞吐提升到初始 production version 的 `6.9×`；
- KernelBench Level 3：官方称几何平均加速 `3.6×`，对比 `torch.compile` 的 `max-autotune` 为 `1.49×`。

这些是 Z.ai 发布方自报，必须绑定任务、工具、机器、超时、停止条件、评测脚本和 verifier。它们不能与 Artificial Analysis 的 Intelligence Index 或其他模型的 DeepSWE 行拼接成裸模型排名。尤其是“655 次迭代”和“数千次工具调用”属于长任务 harness 的轨迹统计，不是模型层数、MoE experts 或 hidden reasoning token 数。

## 5. API、thinking、工具与缓存协议

### 5.1 Thinking

[Deep Thinking 文档](https://docs.z.ai/guides/capabilities/thinking.md) 当前将 GLM-5.1 列入支持列表。对 GLM-5.1 可直接记录：

- `thinking.type = "enabled"` 是 Quick Start 的请求示例；文档说明默认启用，模型可以自动判断是否需要思考；
- `thinking.type = "disabled"` 可用于直接回答；这属于请求级模式，不是另一个 checkpoint；
- `reasoning_effort` 在当前文档中只列为 GLM-5.2 及以上支持，不能把 GLM-5.2 的 `max/high/low` 选项迁移到 GLM-5.1；
- 流式响应可分别读取 `reasoning_content` 和最终 `content`，但 API 字段不等于公开了完整内部训练过程或可编辑的隐藏思维。

### 5.2 Function calling 与 MCP

GLM-5.1 模型页列出 function calling、structured output 和 MCP。Z.ai [Function Calling 文档](https://docs.z.ai/guides/capabilities/function-calling.md) 的协议字段包括 `tools`、函数名与 JSON Schema、`tool_choice`、响应中的 `tool_calls`、唯一 `id` 和 JSON 字符串形式的 `function.arguments`。当前文档示例将 `tool_choice` 写为 `auto`；面试中应把“模型提出调用”和“宿主授权/执行/回灌”分开：

```text
model tool_call -> host schema/permission check -> executor
               -> tool result + call_id -> model continuation
               -> tests/verifier -> final artifact
```

MCP 是外部工具和数据源的集成协议能力，不等于模型内置了网络、文件系统或数据库权限。

### 5.3 Context caching

Z.ai [Context Caching](https://docs.z.ai/guides/capabilities/cache.md) 文档描述隐式缓存：服务端识别重复的 system prompt、长文档或多轮历史，返回 `usage.prompt_tokens_details.cached_tokens`，并按缓存命中 token 使用折扣价格。该缓存是 API/provider 层的上下文复用语义，不应直接写成永久 GPU KV cache；仍需考虑 TTL、内容边界、模板变化、revision、租户隔离和失败重算。

Z.ai [Pricing](https://docs.z.ai/guides/overview/pricing.md) 当前列出的 GLM-5.1 价格为 input `$1.40`、cached input `$0.26`、output `$4.40`/每百万 token；这与 Artificial Analysis 页面上的 provider/目录字段分栏记录，不能混成同一计费快照。

## 6. 论文、博客、代码与负面证据

- 模型卡 README 链接的是 [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)，引用条目标题为 *GLM-5: from Vibe Coding to Agentic Engineering*；README 没有提供 GLM-5.1 专属 arXiv 编号。
- 本轮对 arXiv `all:"GLM-5.1"` 的精确检索返回 20 个结果，但未检出 GLM-5.1 专属技术报告；结果主要是将 GLM-5.1 作为外部被测对象的论文。该结论只表示本轮公开入口未检出，不证明未来不会发布报告。
- 2026-09-20 的一次访问曾把 `https://z.ai/blog/glm-5.1` 记录为 404；2026-09-21 重新通过三条代理访问时均返回博客前端壳（598 bytes，SHA-256 `6fa12ef1d6f8bd1e834b4d8074da0035e36112641ba609d7855d31f92e2727db`）。继续抓取官方 `glm-5.1-UPT9aJ4D.js` 正文资源，三条代理逐字节一致，221,954 bytes，SHA-256 `0e2a4ae9177f44509ee54e9105127d17ff65f6d3b5f95294a7b3b3bdb125c08b`；因此旧的“博客不可得”负面证据已被新鲜官方资源修正。
- Hugging Face README 给出 SGLang、vLLM、xLLM、Transformers 和 KTransformers 的本地部署入口。它说明生态适配状态，但不等于这些后端已经对所有量化、长上下文和生产并发做出相同保证。

### 6.1 官方博客新增的长周期优化实验

官方博客标题为 *GLM-5.1: Towards Long-Horizon Tasks*，日期为 2026-04-07。它把长周期能力具体拆成三种反馈条件，而不是只给出“能工作 8 小时”的结论：

1. **VectorDBBench：有单一数值指标的外循环优化。** 在 Rust ANN 数据库骨架、SIFT-1M 数据集和 Recall ≥ 95% 约束下，模型先受 50-turn 工具预算限制；官方再用 Claude Code 外层循环，让每轮可以编辑、编译、测试、profile 后提交新版本。博客称 GLM-5.1 在 600+ iterations、6,000+ tool calls 后达到 21.5k QPS，并在约第 90/240 轮发生从扫描到 IVF/f16 压缩、再到 u8 预评分 + f16 重排的结构性策略切换。这里的 QPS、迭代次数和跃迁轨迹是发布方 harness 结果，不是模型内部 token 或训练算法证据。
2. **KernelBench Level 3：有可执行正确性和性能 verifier 的系统优化。** 评测包含 50 个完整模型问题、每题独立 Docker + 1 张 H100、最多 1,200 次工具回合；结果要通过 `atol=rtol=1e-4` 的正确性检查，并由 Claude Opus 4.6 与 GPT-5.4 分别审计 benchmark exploitation，取较低 speedup。博客报告 GLM-5.1 几何平均 3.6×，`torch.compile` max-autotune 为 1.49×；数字必须绑定这套任务、硬件、审计器和停止条件。
3. **Linux desktop：没有单一数值目标的自评闭环。** 一个简单 harness 在每轮后要求模型检查缺失功能、粗糙样式、坏交互和边界情况，再继续迭代 8 小时。它说明“无标量目标时的自我评价”是长周期 Agent 的独立难点，不能把自评写成可靠 verifier；最终仍需要 artifact、测试和人工/外部验收。

这组实验把面试问题从“窗口有多长”推进到“额外运行时间是否仍然有用”：应分别问目标/约束、反馈信号、策略切换、错误恢复、停止条件、成本和独立 verifier。博客还明确写出模型在数值指标和无指标任务上仍有局部最优、长轨迹一致性和自评可靠性缺口。

当前仍未核验：GLM-5.1 的官方总/激活参数独立披露、完整层排布叙述、DSA indexer 训练目标与召回曲线、完整预训练/后训练 recipe、奖励/过程质量 verifier、生产 CUDA kernel、硬件 profiling、线上 tool acceptance、8 小时案例的完整 harness 和独立复现。

### 6.2 2026-09-22 当前时点复验

- Artificial Analysis 中文首页当前快照 `/tmp/aa-zh-current-1234.html` 为 `1,798,627` bytes，SHA-256 为 `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`；详情页与三条代理内容一致。首页和详情页都只能作为第三方目录/provider 测量证据，不能覆盖官方模型配置。
- Z.ai GLM-5.1 文档的当前完整页面快照为 `/tmp/zai-glm51-current.md`，`509,732` bytes，SHA-256 为 `6eda05875cc6f63769e0365c1cc7ddad4b499ed7292610e25d24a1daaf7e6ab8`；按 8098 线路提取的稳定 Markdown 内容为 `19,347` bytes，SHA-256 为 `69958d7d95452d3853903524612a0c6a83f368c79417a6889b9c531f45ab3d01`。当前页面仍支持既有的 200K/128K、API ID、thinking、工具和长周期 Agent 结论，没有新增独立技术报告或完整训练 recipe。
- DataCurve 当前没有精确 `mini_swe_agent_glm_5_1_*` 行；因此仍不记录 GLM-5.1 的 Pass@1、Pass@4、成本、输出 token 或 Agent steps，也不迁移 GLM-5/5.2/5.3/5.3-Flash 的行。
- 当前状态仍为 **AA 单榜资料级闭环**。本次仅更新榜单/provider 复验和证据哈希；不新增第二十一册 Transformer 正式章节，不把当前页面测量值写成模型升级。

## 7. 面试知识映射

| 面试主题 | GLM-5.1 证据 | 回答边界 |
|---|---|---|
| 长周期 Agent | 8 小时、数百轮、数千工具调用、实验—分析—优化 | 发布方长任务描述；必须追问 harness、环境、预算、停止和 verifier |
| 过程质量 | multi-turn SFT、RL、process-quality evaluation framework | 没有公开具体 RL 算法/奖励模型，不能写成 GRPO/PPO/RLVR |
| DSA/MoE 配置 | `GlmMoeDsaForCausalLM`、78 层、256 routed/top-8/1 shared、`index_topk=2048` | 配置字段不等于完整生产 kernel、召回曲线或 GLM-5.1 独有算法 |
| 运行时推理 | `thinking.type` enabled/disabled；GLM-5.1 不迁移 `reasoning_effort` | 请求级模式不是新权重，effort 支持必须按精确版本核验 |
| 工具协议 | `tools`、`tool_calls`、JSON Schema、MCP、streaming | tool call 不是权限，也不是工具执行成功 |
| 缓存/成本 | cached token usage 与 `$0.26/M` cache input | provider cache、GPU KV cache、prompt prefix cache 分层 |
| 评测公平性 | AA 配置 vs Z.ai 自报 benchmark vs DataCurve 无精确行 | 不拼接成裸模型排序，不迁移相邻 GLM 版本成绩 |

## 8. 当前状态与后续补证

GLM-5.1 当前为 **AA 单榜资料级闭环**：已有精确榜单条目、官方模型文档、模型卡/配置、API 能力文档、release note、论文负检索和研究笔记。由于没有 GLM-5.1 精确 DataCurve 行，也没有独立 GLM-5.1 技术报告，本轮不新增第二十一册 Transformer 正式章节，内容复用已有 GLM-5 DSA/MoE、Agentic Engineering、工具协议和 serving 章节。

下一步只补以下证据缺口：GLM-5.1 生产 kernel/硬件 profiling、长周期任务的完整可复现实验 harness、过程质量 verifier 的公开定义、精确 DataCurve 行（若排行榜新增）和独立第三方复现。博客正文已取得，但它仍是发布方实验说明，不等于 GLM-5.1 专属技术报告；下一主锚点仍从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜的八家重点厂商条目选择，不从官方目录或 arXiv 外部检索另发现模型。

## 9. 来源清单

### 榜单

- [Artificial Analysis: GLM-5.1](https://artificialanalysis.ai/models/glm-5-1)
- [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/)

### Z.ai 官方资料

- [GLM-5.1 模型文档](https://docs.z.ai/guides/llm/glm-5.1)
- [Deep Thinking](https://docs.z.ai/guides/capabilities/thinking.md)
- [Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md)
- [Context Caching](https://docs.z.ai/guides/capabilities/cache.md)
- [Pricing](https://docs.z.ai/guides/overview/pricing.md)
- [New Released](https://docs.z.ai/release-notes/new-released.md)
- [GLM-5.1: Towards Long-Horizon Tasks](https://z.ai/blog/glm-5.1)
- [Z.ai 文档索引](https://docs.z.ai/llms.txt)

### 模型卡、配置与论文

- [Hugging Face GLM-5.1](https://huggingface.co/zai-org/GLM-5.1)
- [GLM-5.1 config.json](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json)
- [GLM-5 技术报告 arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
