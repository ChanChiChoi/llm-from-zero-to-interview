# OpenAI gpt-oss 官方资料摘记

初始核验日期：2026-09-18；当前时点复验：2026-09-23。本笔记只升级已经在 Artificial Analysis 出现的 `gpt-oss-120b` 与 `gpt-oss-20b`。DataCurve DeepSWE 当前快照没有精确的 `mini_swe_agent_gpt_oss_*` 行，因此不迁移其他 OpenAI 模型或其他 Agent harness 的分数。

## 1. 榜单锚点与证据分层

| 对象 | 榜单证据 | 当前状态 |
|---|---|---|
| `gpt-oss-120b (high)` | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-oss-120b)；页面标注 OpenAI、open weights、high reasoning 配置 | 精确 AA 单榜候选 |
| `gpt-oss-20b (high)` | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-oss-20b)；页面标注 OpenAI、open weights、high reasoning 配置 | 精确 AA 单榜候选 |
| DataCurve DeepSWE | [DeepSWE](https://deepswe.datacurve.ai/) 快照没有 `mini_swe_agent_gpt_oss_120b_*` 或 `mini_swe_agent_gpt_oss_20b_*` | 负证据；不迁移相邻模型结果 |

Artificial Analysis 详情快照：

- `gpt-oss-120b`：3,701,133 bytes，SHA-256 `197c0a6caa8791acd9803daf2d656596b77c84aae448ad9d788e01099d8e4940`。
- `gpt-oss-20b`：3,701,427 bytes，SHA-256 `7990041b97d0ea2f04122ca6817f9f64c6e3b756b5d30ba433e92c3fd3d630fa`。

AA 页面当前字段把 120B/20B 归为约 117B/21B 总参数、5.1B/3.6B active parameters、约 130K context、Apache 2.0 和 2025-08-05 release；这些是第三方目录/测量字段。正式架构、参数账本、许可证和训练资料以 OpenAI 的模型卡、官方仓库和固定 Hub 文件为准。AA 的 Intelligence Index、速度、价格、provider 数量和 context 字段不与模型卡的能力结论混写。

## 2. 官方资料入口与快照

| 来源 | 用途 | 本轮快照 |
|---|---|---|
| [gpt-oss Model Card, arXiv:2508.10925](https://arxiv.org/abs/2508.10925) | 架构、量化、预训练、后训练、工具、评测和安全 | HTML 321,574 bytes；PDF 3,049,591 bytes；HTML SHA-256 `902010af198553e37c044835a64e195e4fb683eea87ff977d69c44ddbc35346f`；PDF SHA-256 `8839e1efdf835be08ad1d60bfa99c8c4fdde15c7ef49eda6a8b910575fb699a0` |
| [OpenAI gpt-oss GitHub](https://github.com/openai/gpt-oss) | 参考推理实现、工具环境、Harmony 接入和 serving 入口 | README 24,453 bytes；SHA-256 `578ad0f82c1d823229f9bf2b52b3f1c55ce82772ac57f634f0ca4eb46a6370aa` |
| [gpt-oss-120b Model Card](https://huggingface.co/openai/gpt-oss-120b) | 权重、模型卡、部署入口和尺寸信息 | README 7,111 bytes；SHA-256 `92b0408cf5dce04e4c5e4e5f2be0361da710d3650baefd7d7519badd1c687919` |
| [120B config.json](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | 可机器读取的 revision 配置 | 2,089 bytes；SHA-256 `933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566` |
| [gpt-oss-20b Model Card](https://huggingface.co/openai/gpt-oss-20b) | 20B sibling 的权重和部署入口 | README 7,095 bytes；SHA-256 `03c2fcf292549176757b85c911e7dcf527aef3e4241d64b6caec94af3ecf3ac2` |
| [Harmony response format](https://github.com/openai/harmony) | 对话渲染、channel、工具调用和解析库 | README 7,462 bytes；SHA-256 `c38c6a52720adebbdd90e691d2db5fd0711952cb22c7b13246786924375975da` |
| [OpenAI gpt-oss Cookbook topic](https://developers.openai.com/cookbook/topic/gpt-oss) | 官方 Transformers、vLLM、Harmony、Agents 运行入口 | 312,379 bytes；SHA-256 `50a0bcb0c26618d7d5264aa2600a6bc243772c8abac0d1293125beb5c2f00f66` |
| [Harmony Cookbook](https://developers.openai.com/cookbook/articles/openai-harmony) | Harmony 的渲染/解析和 Responses 风格协议说明 | 422,739 bytes；SHA-256 `1853f1385ecf7a7cfaf858ddb12d2d860af163e5282af3f8688c367cb76b92da` |
| [Run with Transformers](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers) | Transformers/chat template 和本地推理 | 366,775 bytes；SHA-256 `a90cec792b208da4632eb43cddd3d590ba9c3e41efc04f5023966ceb73a0d152` |
| [Run with vLLM](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm) | vLLM、函数调用和 Agents SDK 接入 | 359,249 bytes；SHA-256 `bdc75fd543254e93806cbda66c3786e61b28994baab1f6b4d597b91ca576d193` |

OpenAI 主站发布页在本轮代理访问中返回 HTTP 403，不能把该失败解释成资料不存在；模型卡、官方仓库和 developers.openai.com 的 Cookbook 已提供可复核的一手资料。后续若主站线路恢复，再补发布页快照，不改变当前已确认结论。

## 3. 模型族身份：两个尺寸，不是两条架构路线

OpenAI Model Card 同时介绍两个 text-only open-weight autoregressive MoE Transformer：

| 字段 | `gpt-oss-120b` | `gpt-oss-20b` |
|---|---:|---:|
| 层数 | 36 | 24 |
| 总参数 | 116.8B | 20.9B |
| active parameters/token | 5.13B | 3.61B |
| MoE experts | 128 | 32 |
| top experts/token | 4 | 4 |
| checkpoint size | 60.8 GiB | 12.8 GiB |
| 目标部署 | 单个 80GB GPU 级别的高推理/生产场景 | 更低延迟、本地或专用场景 |

这里的 active parameters 是每个 token 的路径计算口径，不是总权重显存、专家通信量、KV cache 或并发工作区。总参数和 active parameters 必须分开写入 serving 账本。

官方模型卡称两个模型均采用 Apache 2.0、支持可调 reasoning effort、函数调用、浏览器、Python 和 Structured Outputs，并在 MXFP4 MoE 权重上完成后训练。模型卡/仓库把 `gpt-oss-120b` 描述为可以在一张 80GB GPU 上运行，20B 可在约 16GB memory 级别运行；这属于指定实现、量化和硬件条件下的部署目标，不是任意后端的普遍 SLO。

## 4. 架构：MoE + 交替滑动/全注意力

### 4.1 MoE 路由

每个 MoE block 由固定数量专家、线性 router 和 gated SwiGLU 组成。router 对每个 token 产生专家分数，选择 top-4 专家，再只在被选集合上做 softmax 加权。120B 使用 128 experts，20B 使用 32 experts。模型卡还明确指出 SwiGLU 实现包含 clamping 和 residual connection；不能把它简化成普通无门控 FFN。

面试时至少区分四个量：

1. 总参数：所有专家和共享组件的权重规模；
2. active parameters：当前 token 路径参与主要前向的参数口径；
3. dispatch/通信：token 到 expert 的路由、容量、padding 和 all-to-all；
4. serving 显存：权重、KV、workspace、通信 buffer 和并发请求的总账本。

### 4.2 交替注意力

两个模型交替使用 banded/sliding-window attention 和 fully dense attention。窗口带宽为 128 tokens；每层使用 64 个 query heads、head dimension 64、8 个 key-value heads，即 GQA。位置编码使用 RoPE；dense 层通过 YaRN 扩展到 131,072 token。配置快照与模型卡同时给出 `sliding_window=128`、`num_attention_heads=64`、`num_key_value_heads=8` 和 `max_position_embeddings=131072`。

因此“130K context”不能被解释成每一层都做 130K 范围的 dense attention：局部层的访问范围受窗口约束，交替的 dense 层负责跨远距离的信息交换。工程评估需要分别记录 dense/sliding 层的 FLOPs、KV 形状、长程检索召回和 TTFT/TPOT。

### 4.3 Attention bias 与 tokenizer

每个 attention head 在 softmax denominator 中带 learned bias，使 attention 可以选择不关注任何 token；这与 attention sink/off-by-one attention 的思路相关，但不能据此把模型等同于某一篇外部论文的实现。

两个模型使用 `o200k_harmony` tokenizer，是基于 OpenAI `o200k` 的 BPE，加入 Harmony chat format 所需的显式 token，总词表为 201,088。tokenizer、chat format 和权重 revision 必须一起固定，否则 prompt tokenization、reasoning channel 和 stop token 可能不兼容。

## 5. MXFP4：量化不是部署后的独立外挂

Model Card 说明 MoE 权重采用 post-training quantization 到 MXFP4，平均每参数约 4.25 bits；MoE 权重占总参数 90% 以上，因此该量化直接决定 checkpoint 和单 GPU 部署可行性。官方评测也使用同一 MXFP4 量化，不能用 BF16 运行结果替换发布方数字。

仓库把 MXFP4 作为原生支持，并提供 PyTorch/Triton 参考实现。README 说明 Triton 路径使用支持 MXFP4 的优化 MoE kernel，并在单个 80GB GPU 上运行 120B；参考 PyTorch 实现是教学/非优化路径，需要至少 4 张 H100 级别设备。这个差异体现了“模型可运行”和“生产可运行”的后端边界。

尚未确认或不能从公开资料推出：完整 kernel 的所有优化、不同 GPU/后端的实际吞吐、quantization error 的完整消融、online batch 的 p99、专家负载倾斜和跨设备通信 profiling。

## 6. 预训练与后训练：公开了方向，不等于完整 recipe

### 6.1 预训练

Model Card 公开：数据是 text-only、规模为 trillions of tokens，重点覆盖 STEM、coding 和 general knowledge；预训练对有害生物安全内容使用 CBRN 过滤；知识截止为 2024 年 6 月。训练使用 NVIDIA H100、PyTorch、expert-optimized Triton kernels 和 FlashAttention；120B 训练运行约 2.1 million H100-hours，20B 约少一个数量级。这里的时间和硬件是发布方训练设置，不是独立成本测量。

### 6.2 CoT RL 与工具使用

预训练之后，OpenAI Model Card 只把后训练概括为与 o3 类似的 CoT RL techniques：同时教模型通过 CoT 解题、使用工具和遵循安全/指令层级。公开的训练工具包括：

- 浏览工具：search/open，用于补充知识截止日期之后的信息；
- stateful Python/Jupyter 工具：进行计算和代码执行；
- developer-defined functions：在 Harmony 的 Developer message 中给出 schema，允许 CoT、函数调用、函数结果、中间可见消息和 final answer 交错出现。

不要把“类似 o3 的 CoT RL”扩写成已确认的 PPO、GRPO、奖励函数、KL 系数、rollout 数、teacher 或优化器配方。公开资料没有给出完整后训练 recipe。

## 7. Harmony：模型输出协议也是训练接口

Harmony 不是普通 prompt 模板，而是模型训练时使用的 response format。它提供：

- message boundary 与 `system/developer/user/assistant/tool` 角色；
- instruction hierarchy：System > Developer > User > Assistant > Tool；
- `analysis`、`commentary`、`final` 等 channel，区分推理、工具前置说明和用户可见答案；
- recipient/tool namespace、函数参数和 structured outputs 的明确结构；
- 可逆的渲染与解析，官方实现使用 Rust core 并通过 PyO3 提供 Python 包。

一个重要的训练/推理边界是：多轮对话中，历史 assistant 的 reasoning traces 应按模型卡要求移除；不能因为模型支持 full CoT 就把原始 reasoning 一直回灌到下一轮。生产宿主还需要做 channel、recipient、schema、权限、超时和结果回灌校验。

在 vLLM 或自写 sampler 中，如果直接调用 `model.generate`，必须使用 chat template 或 `openai-harmony` 手工渲染；否则模型可能无法正确工作。使用 Transformers chat template 时，库会自动应用 Harmony。这个格式兼容 OpenAI Responses 风格的结构，但不代表本地模型获得 OpenAI 托管 API 的权限、工具或安全过滤。

## 8. Variable-effort reasoning：推理预算的可调旋钮

模型支持 `low`、`medium`、`high` 三档 reasoning level，通过 system prompt 中的 `Reasoning: low/medium/high` 指定。Model Card 报告随着档位升高，平均 CoT 长度增加，AIME/GPQA 等任务呈现相对平滑的 test-time scaling；代价是 latency 和 token cost 上升。

这不是一个公开的“固定 token budget”保证，也不是三个基础模型。应把它记录为同一权重的配置维度，并在实验中固定模型 revision、tool availability、sampling、最大输出和 verifier，分别画质量—CoT 长度—延迟—成本曲线。

## 9. 评测：把模型、工具和 harness 分开

Model Card 的基础评测覆盖 AIME、GPQA、HLE、MMLU、Codeforces、SWE-bench Verified、tau-Bench、HealthBench 和 MMMLU；120B/high 的示例包括 AIME 2024 no-tools 95.8、with-tools 96.6、SWE-bench Verified 62.4、tau-Bench Retail 67.8、Codeforces with-tools Elo 2622。20B/high 的对应示例包括 92.1、96.0、60.7、54.8 和 2516。

这些数字都是 OpenAI Model Card 的发布方结果，并绑定 high/low/medium、是否有工具、默认 system prompt、benchmark split、采样与 harness。它们不能与 Artificial Analysis Intelligence Index 或 DataCurve Pass@1 拼成一张裸模型排名；当前 DataCurve 没有 gpt-oss 精确行。

一个适合面试的归因表是：

| 层 | 需要固定的变量 | 不能直接归因给 |
|---|---|---|
| 模型 | checkpoint/revision、tokenizer、precision、reasoning level | Agent harness |
| 协议 | Harmony channels、tool schema、消息回放 | 模型参数本身 |
| 执行器 | browser/Python/function runtime、权限、超时、沙箱 | 模型单独能力 |
| 评测 | task split、prompt、tool、verifier、retry、hardware | AA 或 DataCurve 的另一榜单 |

## 10. 安全与开放权重边界

OpenAI 明确把该资料称为 model card 而不是 system card：权重发布后，下游使用者可以微调模型，系统级安全措施不能完全由 OpenAI 撤回或统一控制。公开的安全路线包括 deliberative alignment、拒答、越狱鲁棒性、instruction hierarchy 和 Preparedness 评测。

模型卡报告默认模型没有达到其三个 tracked categories 的 High capability 指示阈值，并另外评估了 adversarial fine-tuning 后的生物/化学和 cyber 能力；这些是发布方安全评测结论，不能推导成所有下游微调和部署场景的安全保证。生产系统仍需把权重、微调数据、工具权限、沙箱、输出过滤和审计分开治理。

## 11. 面试追问

1. 120B 只有 5.1B active，为什么不能按 5.1B 估算整机显存？
   - 因为总专家权重仍要存储或分片，token dispatch、KV、workspace、通信 buffer 和并发请求另有账本；active 只描述当前 token 的主要计算路径。
2. 交替 sliding/full attention 如何兼顾长程依赖和成本？
   - sliding 层提供局部高效更新，dense 层提供跨远程 token 的信息交换；要测不同层的 FLOPs、KV、长程召回和端到端延迟。
3. MXFP4 为什么属于训练/部署联合设计？
   - MoE 权重占比很大，post-training MXFP4 改变 checkpoint 和显存边界；官方评测也在同一量化下进行，不能只在 BF16 结果上讨论质量。
4. Harmony 和普通 ChatML 的差别是什么？
   - Harmony 规定角色层级、channel、recipient、工具/函数结构以及可逆渲染解析，且模型在该格式上训练；它是协议契约，不只是字符串分隔符。
5. `Reasoning: high` 是模型的第三个版本吗？
   - 不是。同一模型的推理配置，主要改变 CoT/test-time compute 与成本延迟；必须把 effort 作为实验变量而不是模型身份。
6. gpt-oss 的工具调用为什么不能直接等同于工具执行？
   - 模型只产生 Harmony function call；宿主还要做 schema、权限、确认、沙箱、超时、结果回灌和 artifact 验证。
7. 为什么 OpenAI 的 SWE-bench 结果与 DataCurve 不能直接比较？
   - 任务集、harness、工具、verifier、effort、重试和环境不同；DataCurve 当前也没有 gpt-oss 精确行。
8. open weights 发布对安全工程带来什么额外问题？
   - 下游可以微调和改变拒答行为，发布方无法撤回每个副本；因此要把模型卡安全结论与部署方的权限、沙箱、数据和审计责任分开。

## 12. 待核验项与书系映射

待核验：完整训练数据配比、optimizer 和后训练 loss/奖励配方、MoE router 的所有负载均衡细节、MXFP4 kernel/误差/硬件 profiling、线上专家负载和 acceptance rate、完整 OpenAI 发布页正文、不同 provider 的真实 API 行为，以及独立 benchmark 复现。

本锚点新增第二十一册第 87 章，承接 MoE、sliding/full attention、MXFP4、Harmony 和 variable-effort reasoning；Agent 工具权限与评测归因映射到第十七、二十和二十四册。没有把 `gpt-oss-20b` 另建成重复架构章节。

## 13. 2026-09-22 Cookbook：实现兼容性与 raw CoT 补证

本轮继续只沿 Artificial Analysis 已发现的 `gpt-oss-120b`/`gpt-oss-20b` 推进。OpenAI Cookbook 当前专题页列出两个此前研究笔记没有展开的官方入口：`Verifying gpt-oss implementations` 与 `How to handle the raw chain of thought in gpt-oss`。它们不是新的模型发现，也不是新的权重版本，而是把“模型能运行”拆成 API 兼容、工具调用、推理质量和安全回放四层的部署验收资料。

### 13.1 快照与版本边界

| 官方资源 | 抓取结果 | 证据用途 |
|---|---:|---|
| [gpt-oss Cookbook topic](https://developers.openai.com/cookbook/topic/gpt-oss) | `312,061` bytes / SHA-256 `8027d2c25fc0607e52783669a52914524edd4ea999335b1aae98836fc67d78a0` | 当前专题资源目录；列出实现验证与 raw CoT 文章 |
| [Verifying gpt-oss implementations](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations) | `336,633` bytes / SHA-256 `4f293d2bd51666966a8d3657893eb1d1093b4b3230a405c9178cbc7302d664b8` | API 形状、兼容性 smoke test 和 AIME/GPQA/HealthBench eval 门禁 |
| [How to handle the raw chain of thought](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot) | `338,262` bytes / SHA-256 `7330273ff59d6e1a344abe3f435ae421a20ebd775bda9122d22ff606f0782072` | raw CoT 的可见性、Harmony 回放和 Responses/Chat Completions 协议 |
| [OpenAI Harmony Cookbook](https://developers.openai.com/cookbook/articles/openai-harmony) | `422,675` bytes / SHA-256 `de55dd27cd90dfce7078dfad3101b74c07522102f42b58a3b4abfad9fe25ef6c` | Harmony 渲染/解析的当前官方入口 |
| [gpt-oss GitHub commit history](https://api.github.com/repos/openai/gpt-oss/commits?per_page=10) | `44,611` bytes / SHA-256 `420dcfab2dad69aa386b5207ac3aa35f627151e292c63caced716d46c59e288c` | mutable upstream history；最近提交为 `7b583341fe16`（2026-07-24），内容是 You.com backend API key 一致性修复，不是模型/核心推理变化 |

README 当前仍为 `24,453` bytes / SHA-256 `578ad0f82c1d823229f9bf2b52b3f1c55ce82772ac57f634f0ca4eb46a6370aa`，与此前快照逐字一致。上述快照证明本轮读取到的官方页面和仓库状态，不把动态 Cookbook 页面哈希变化解释成模型 revision。

### 13.2 兼容性不是一个“能返回 200”指标

官方实现验证指南把 gpt-oss provider 的验收拆成以下层次：

1. **输入渲染**：provider 必须把请求正确映射到训练使用的 Harmony format。模板、角色层级、channel、recipient 或 stop 处理错误，会以级联方式表现为更差的 function calling，而不一定立刻报错。
2. **工具循环状态**：tool call 可能发生在 CoT 内部。下一次 sampling 需要带回此前 raw CoT、tool call 和 tool output；只保留最终可见答案会破坏后续调用的状态。
3. **Responses API 形状**：官方推荐用 Responses API 承载 raw CoT。`reasoning` item 可包含 `content` 数组，其中 raw CoT 使用 `{"type":"reasoning_text","text":"..."}`；下一轮必须把该 item 按协议重放，再进入 Harmony prompt。
4. **Chat Completions 兼容层**：公开 OpenAI Chat Completions 规范没有 raw CoT 字段。官方建议 provider 采用 `reasoning` 作为主字段，并支持流式 delta 的 `reasoning`；这属于 gpt-oss provider 的兼容约定，不应写成 OpenAI 托管模型的普遍 API 契约。

因此，API 返回成功、产生了一个看似合法的 tool call，或者完成了单轮文本生成，都不能单独证明实现兼容。至少要同时测 schema、channel、tool call、结果回灌、重复回放、取消、streaming 和下一轮 continuation。

### 13.3 Responses raw CoT 的事件和状态协议

raw CoT 应与可展示的 reasoning summary 分开保存。Responses 形状新增 `reasoning.content`，对应两个流事件：

```text
response.reasoning_text.delta
response.reasoning_text.done
```

`delta` 事件逐段携带 `item_id`、`output_index`、`content_index` 和文本增量；`done` 事件给出该 content item 的完整文本。应用层应把它们视为同一个 reasoning item 的增量/完成状态，使用 `item_id` 和 index 做去重与顺序校验，不能把 delta 拼接到用户可见消息后就算完成。

官方 raw CoT 指南同时强调安全边界：raw CoT 可能包含有害内容或泄露 developer instructions，不能直接展示给终端用户。若产品需要可见解释，应使用经过审查/过滤的 summary；raw CoT 只进入受控的 provider、调试、解释性研究或下一轮状态回放链路。

Harmony 回放规则需要按状态处理：

- 后续 sampling 产生 `final` 后，可以丢弃此前的 `analysis` 消息；
- 如果 assistant 的最后一条消息是 tool call，应在下一轮继续保留到上一个 `final` 之前的 analysis；
- 发往 `commentary` channel 的 function call 可以按协议保留；
- raw CoT、tool call、tool result 和 final 必须带同一轮的 lineage，避免重试时重复执行外部副作用。

### 13.4 两层验证：API smoke test 与模型质量 eval

官方仓库的 `compatibility-test` Node.js 测试用于快速检查 Responses/Chat Completions 的 function calling 和 API shape。结果中的 invalid request 数量、工具选择、参数形状和结果回灌应作为协议门禁；官方指南把“0 invalid requests，且 pass@k 与 pass^k 均超过 90%”作为“很可能正确”的信号，但明确这不是完整兼容性或精度证明。测试还支持 `jsonl` 逐响应审计、`DEBUG=openai-agents:openai` 请求调试、单用例 `-n 1` 和 streaming 模式。

质量层使用同一仓库的 eval harness：AIME 使用每题 16 次尝试，GPQA 使用每题 8 次，HealthBench 使用每题 1 次；Responses 与 Chat Completions 通过不同 sampler 参数运行，并固定 base URL、model、reasoning effort、工具、采样和 verifier。协议 smoke test 通过不代表 kernel、MXFP4、MoE dispatch 或 benchmark 结果正确；反过来，单项 benchmark 接近发布方数字也不能证明 API 状态回放和工具副作用安全。

这形成适合面试的证据分层：

```text
Harmony/render + API shape
        -> tool-call compatibility smoke test
        -> raw CoT/state replay
        -> AIME/GPQA/HealthBench quality eval
        -> kernel/precision/hardware profiling
        -> production acceptance and artifact verifier
```

本轮新增的是官方兼容性和 raw CoT 协议证据；没有新增模型、权重 revision 或独立 benchmark。完整训练 recipe、router 负载均衡、MXFP4 kernel 误差、真实 provider 行为、目标硬件 profiling、线上接受率和独立复现仍待核验。

## 14. 2026-09-23 7890 当前时点复验：榜单字段与负证据

本轮使用用户确认可用的 `10.24.27.134:7890` 重新验证联网状态：百度、Artificial Analysis 中文首页和 DataCurve DeepSWE 均返回 HTTP 200。当前榜单页面快照为：AA 中文首页 `1,783,626` bytes / SHA-256 `0580fad58c167fb96e87289addbabccd40cbd60302c527437805c8eb6ee7530b`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。

本次仍只沿两个榜单已经存在的 `gpt-oss-120b` 与 `gpt-oss-20b` 推进，没有从 OpenAI Cookbook、GitHub 或 Hugging Face 另发现模型。AA 详情页的当前快照和第三方观察字段如下：

| AA 条目 | 详情快照 | 当前观察字段 |
|---|---|---|
| `gpt-oss-120b (high)` | `4,078,295` bytes / SHA-256 `43e5f13a3e58976b077acac1810bce82ac60ce33a7d81476f0dc2b17175532de` | 117B total、5.1B active、131,072 context、Intelligence Index `11.6028431512592`、median output speed `196.389235173389 tokens/s`、cost per Intelligence Index task `0.10742452290394947` |
| `gpt-oss-20b (high)` | `4,083,068` bytes / SHA-256 `64d67047a5964c0109f103108f9b77c64edbaf1ee17f3fec3c82ffccd7cea2cc` | 21B total、3.6B active、131,072 context、Intelligence Index `8.9675171856126`、median output speed `185.656167974395 tokens/s`、cost per Intelligence Index task `0.012460640242179213` |

这些是 AA/provider 在采集时点的目录与测量字段，不是 OpenAI 模型卡的新架构、训练或权重证据。详情页的动态页面内容发生变化也不能单独证明模型 revision 变化。DataCurve 当前快照仍没有精确的 `mini_swe_agent_gpt_oss_120b_*` 或 `mini_swe_agent_gpt_oss_20b_*` 行，因此不迁移 GPT-5.x、Codex 或其他 OpenAI 模型的 Agent 结果；该缺失是本快照的负证据，不宣称所有历史版本都不存在。

### 14.1 官方 Cookbook 当前快照

7890 当前取得的两个官方页面为：

| 官方资料 | 当前快照 | 与上一轮的解释 |
|---|---|---|
| [Verifying gpt-oss implementations](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations) | `337,423` bytes / SHA-256 `e6500d57bc6041133a8f63acc03101260182ea58d45d2f7248806ae609c85474` | 动态正文快照更新；仍是 API shape、tool call/result、streaming、invalid request 和质量 eval 分层指南 |
| [How to handle the raw chain of thought](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot) | `339,052` bytes / SHA-256 `1c08e3fb06a129256596c1af9ba907b7b3c435be3b61cfb00a72da83b1071fe6` | 动态正文快照更新；仍确认 `reasoning_text`、raw CoT 受控回放和终端不可见边界 |

与 9 月 22 日快照相比，页面字节数/哈希发生变化，但本轮没有观察到新的模型 ID、权重 revision 或架构声明。当前 gpt-oss 状态因此更新为：**AA 当前时点复验 + 官方 Model Card/仓库/Harmony/Cookbook 证据 + provider compatibility/raw-CoT protocol evidence**；完整训练/后训练 recipe、MoE 负载均衡、MXFP4 kernel/误差、目标硬件 profiling、独立 Agent 评测和生产 acceptance 仍待核验。
