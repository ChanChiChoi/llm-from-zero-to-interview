# GLM-5 官方资料摘记

核验日期：2026-09-16

## 1. 候选来源与证据边界

Artificial Analysis 有精确的 `GLM-5 (Reasoning)` 条目，canonical slug 为 [`glm-5`](https://artificialanalysis.ai/models/glm-5)，并另列 `glm-5-non-reasoning`。本轮三条用户提供的代理均对详情页返回 HTTP 200，页面大小均为 `3,577,227` bytes，SHA-256 为 `57dbab2e4ef52e2c95c585bb1d0549f8044486903b77366fed1d5c534569c2a1`。

DataCurve DeepSWE 页面本轮重新抓取三次，均返回 HTTP 200，大小 `268,313` bytes，SHA-256 为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。页面能够检出 `mini_swe_agent_glm_5_2_max`、`mini_swe_agent_glm_5_3_max` 和 `mini_swe_agent_glm_5_3_flash_max`，没有精确的 `GLM-5` 或 `mini_swe_agent_glm_5_max` 行。因此 GLM-5 是 **Artificial Analysis 单榜发现**，不能写成两个排行榜同时确认的锚点，也不能把 GLM-5.2/5.3 的 DeepSWE 分数迁移给它。

Artificial Analysis 页面当前显示的 `744B` 总参数、`40B` active、`200K` context、约 `72.4 tokens/s` 和约 `1.34s` TTFT 是第三方目录或测量字段。它们与官方模型卡交叉一致的部分才进入规格结论；指数、价格、速度和 TTFT 仍保留为 Artificial Analysis 的配置/provider 结果。

## 2. 官方模型、模型卡与报告

Z.ai 官方 [GLM-5 技术博客](https://z.ai/blog/glm-5) 的页面正文由 JavaScript 加载，本轮取得入口页 HTTP 200；可复核的模型卡 README 同时链接该博客、Z.ai API、GitHub 仓库和论文入口。官方 [Hugging Face 模型卡](https://huggingface.co/zai-org/GLM-5) 的固定内容声明：

- GLM-5 面向 complex systems engineering 和 long-horizon agentic tasks；
- 相对 GLM-4.5，规模从 `355B total / 32B active` 增加到 `744B total / 40B active`；预训练数据从 `23T` 增加到 `28.5T` tokens；
- 集成 DeepSeek Sparse Attention（DSA），目标是在保持长上下文能力的同时降低部署成本；
- 发布方提供 [GLM-5 专属技术报告](https://arxiv.org/abs/2602.15763)，标题为 *GLM-5: from Vibe Coding to Agentic Engineering*；
- 模型卡为 MIT，提供 Transformers、vLLM、SGLang、KTransformers 和 xLLM 等部署入口。

固定模型卡配置（本轮直接取得 `config.json`，SHA-256 `308d3ceebe7bd9c2246d20550d5b793098dd03838d1e504f8236063ae152bcae`）公开了可复现的实现字段：`GlmMoeDsaForCausalLM`、`glm_moe_dsa`、78 层、隐藏维度 6144、256 routed experts、每 token top-8、1 个 shared expert、前三层 dense replacement、`q_lora_rank=2048`、`kv_lora_rank=512`、`qk_rope_head_dim=64`、`index_topk=2048`、`max_position_embeddings=202752` 和 bfloat16。配置字段是实现合同，不等于完整训练 recipe；总/激活参数仍以模型卡和官方发布说明为边界。

技术报告是 GLM-5 的专属一手研究来源，不把它与 GLM-5.3 的关联报告混用。报告中的 benchmark 表和模型卡 README 的对比表都是发布方测量，必须同时记录任务集、prompt、最大生成长度、温度、工具/harness、judge 和重复次数，不能与 Artificial Analysis 或 DataCurve 的结果拼成一个裸模型排序。

## 3. 值得面试追踪的技术

### 3.1 DSA：把长上下文注意力变成带索引的检索问题

标准全注意力在序列长度为 (L) 时需要近似 (O(L^2)) 的 token-pair 计算。DSA 的核心面试问题不是“它用了稀疏 attention”这么简单，而是：先用轻量 indexer 对历史 token 建立相关性估计，再只让 query 读取 top-k 候选，从而把主 attention 的访问范围压缩到稀疏集合。GLM-5 配置公开 `index_topk=2048`、index head 维度和 RoPE 字段，但没有公开足以复现全部训练细节的 indexer loss、召回率曲线、分层布局和生产 kernel。

因此应把收益拆成三本账：

1. 主 attention 的 FLOPs 和 KV 读取减少；
2. indexer 本身的计算、索引存储和内存带宽成本；
3. top-k 召回不足造成的质量损失，以及训练/推理稀疏模式不一致带来的风险。

面试追问可以要求候选人说明：固定 `top-k` 是否适合不同 query；长上下文中稀有但关键的信息如何避免被过滤；如何用 recall@k、needle retrieval、长程任务成功率和端到端 latency 而不是单看 attention FLOPs 验证方案。

### 3.2 MoE 扩容：总容量和单 token 计算分离

GLM-5 的 `744B total / 40B active` 体现典型 MoE 取舍：增加专家总容量和知识分工，但每个 token 只路由到少数专家，以控制单 token 计算量。配置进一步公开 256 个 routed experts、top-8 和 1 个 shared expert。面试中要区分 total parameters、active parameters、通信量和实际 wall-clock latency；active 参数不自动等于推理显存，也不包含 routing、KV cache、通信和 runtime overhead。

### 3.3 `slime`：异步 RL 基础设施

模型卡将 [`slime`](https://github.com/THUDM/slime) 描述为 asynchronous RL infrastructure，用于提升 rollout 和训练吞吐，并支持更细粒度的 post-training iterations。它解决的是训练系统瓶颈，不是一个可以从 API 字段直接推出的模型内部算法。

面试时可用生产流水线解释：rollout workers 生成多轮答案/工具轨迹，verifier 或 reward pipeline 计算任务级信号，trainer 消费样本更新策略，新的策略版本再回到 rollout。生成和训练解耦后，吞吐与资源利用率可能提升，但要管理 policy lag、off-policy 偏差、样本版本、奖励延迟、失败轨迹过滤和 checkpoint 一致性。

### 3.4 Agentic Engineering：评价完整闭环而非单轮代码

GLM-5 的定位从 vibe coding 转向 agentic engineering。对应的能力对象是规划、代码编辑、shell/工具调用、执行、测试、错误诊断、修复和长任务状态管理的循环。模型卡 README 提供的 SWE-bench、Terminal-Bench、BrowseComp、MCP-Atlas 等结果，应按各自的 OpenHands、Terminus 2、Claude Code、工具权限、上下文管理和最大生成长度读取。

这带来一个重要的评测原则：同一个模型换 harness、工具或 verifier，结果可能显著变化。DeepSWE 的统一 `mini-swe-agent` 结果属于 Agent 系统组合；Artificial Analysis 的 Intelligence Index 又是另一套 provider/workload。面试回答应先固定模型 snapshot、effort、工具集、harness、任务集、超时、上下文压缩和 verifier，再讨论成功率、成本和延迟。

### 3.5 长轨迹 RL 的信用分配

当轨迹包含几十次工具调用时，最终任务成功信号距离早期动作很远。可讨论的工程方法包括按阶段切分 reward、对工具调用和中间 artifact 做 verifier、过滤不可执行轨迹、保存环境状态，并用任务级而非文本表面相似度评价结果。GLM-5 官方资料公开了异步 RL 和 agentic 方向，但没有公开足以确认其完整 reward shaping、credit assignment 或 optimizer recipe 的细节；这些方法只能作为通用面试分析，不能冒充 GLM-5 已确认实现。

## 4. 评测证据

模型卡 README 的发布方表包含 HLE、AIME、GPQA-Diamond、SWE-bench Verified、SWE-bench Multilingual、Terminal-Bench 2.0、CyberGym、BrowseComp、MCP-Atlas、Tool-Decathlon 和 Vending Bench 2 等结果。README 明确给出不同任务的最大生成长度、温度/top-p、上下文上限、OpenHands/Terminus/Claude Code、judge、超时和重复次数。例如 HLE 使用最多 131,072 个新 token，SWE-bench 使用 OpenHands 和 200K context，Terminal-Bench 使用 Terminus 2 或 Claude Code。故这些数字是发布方在明确设置下的结果，不是跨榜单可直接比较的模型常数。

本轮不写入 GLM-5 的 DataCurve 分数：页面没有精确条目。GLM-5.2、GLM-5.3 和 GLM-5.3 Flash 的行可以用于版本演进背景，但不能当作 GLM-5 的实测替代品。

## 5. 负面证据与待核验

- 已找到 GLM-5 专属技术报告 `arXiv:2602.15763`，因此不能再写“没有 GLM-5 专属技术报告”。
- 本轮没有在 DataCurve 页面找到精确 `GLM-5` 行；这只说明当前页面快照没有该行，不否定未来补测。
- 仍待核验 DSA 的完整 indexer 训练目标、层级布局、真实 top-k recall、生产 kernel、KV/indexer bytes 和目标硬件 profiling。
- 仍待核验 `slime` 的完整异步调度、样本新鲜度控制、策略版本同步、reward/credit assignment 细节和独立复现。
- 仍待核验完整预训练/后训练 recipe、optimizer、数据处理细节、线上工具接受率和外部独立 benchmark。

## 6. 来源清单

- 榜单：[Artificial Analysis GLM-5](https://artificialanalysis.ai/models/glm-5)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/)（本轮无精确 GLM-5 行）。
- 官方：[Z.ai GLM-5 博客](https://z.ai/blog/glm-5)；[Z.ai GLM-5 API 文档](https://docs.z.ai/guides/llm/glm-5)；[Hugging Face 模型卡](https://huggingface.co/zai-org/GLM-5)；[固定 README 原文](https://huggingface.co/zai-org/GLM-5/raw/main/README.md)；[固定 config.json 原文](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)；[GLM-5 官方 GitHub](https://github.com/zai-org/GLM-5)；[`slime` GitHub](https://github.com/THUDM/slime)。
- 论文：[GLM-5 技术报告](https://arxiv.org/abs/2602.15763)。
- 本轮快照：AA `/tmp/gather-aa8098-20260916.html`；DataCurve `/tmp/gather-ds8098-20260916.html`；模型卡 `/tmp/glm5-hf-card-20260916.html`；README `/tmp/glm5-hf-readme-20260916.out`；config `/tmp/glm5-hf-config-20260916.out`；博客入口 `/tmp/glm5-zai-blog-20260916.out`。

## 7. 当前结论

GLM-5 当前为 **内容专题闭环（AA 单榜）**：Artificial Analysis 精确发现、Z.ai 官方模型卡、专属技术报告、API/部署资料、面试技术主线和正式章节已经具备；DataCurve 当前没有精确 GLM-5 配置，因此不升级为双榜锚点，也不新增 GLM-5 的 DeepSWE 分数。后续仍优先补 DSA/slime 的实现与独立复现证据。

## 8. 2026-09-20 重新联网复验

本轮重新访问两个唯一排行榜和 GLM-5 官方周边资料，结果按线路和站点分别记录：

- Artificial Analysis 详情快照 `/tmp/glm-aa-20260920.out`，`3,811,809` bytes，SHA-256 `0b9c56ff97a87b1dd0a006d1c300057f5ef20c3f65a19bcddf0f6803d8a94f8d`。当前页面标题为 `GLM-5 (Reasoning)`，并提示已有更新模型 `GLM-5.1`；因此 GLM-5 保留为历史 AA 锚点，不能把页面提示误写成 GLM-5.1 的技术证据。页面第三方字段约为 744B/40B、200K context、Intelligence Index `27.9111550817227` 和约 69 tokens/s；指数、速度、TTFT 和价格仍是第三方配置/provider 字段。
- DataCurve 快照 `/tmp/aa-check-8098-ds-20260920.out`，`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。仍未找到精确 `mini_swe_agent_glm_5_*` 行，不迁移 GLM-5.2/5.3/5.3 Flash 结果。
- Z.ai 博客入口资源 `/tmp/glm-blog-js-20260920.out`，`145,189` bytes，SHA-256 `99d27d6132c25e1b39fe26df0605ad1abd855e9b30b098996c64423093fb618f`；资源正文可检出 `GLM-5`、`DSA`、`slime`、`744B`、`40B`、`28.5T` 和 `Agentic` 等字符串，作为博客入口资源的复验，不把压缩 JavaScript 中的字符串扩写成未公开实现细节。
- `https://docs.z.ai/guides/llm/glm-5.md` 和无扩展名文档页本轮经 `8098/1234` 返回代理 HTTP 503；这只记录为本轮文档线路失败。已有 9 月 16 日官方文档/模型卡快照继续作为资料来源，不能把 503 解释为页面不存在。Hugging Face/GitHub 本轮的线路失败同样不改变已有固定快照的证据状态。

本轮新增正式章节 [`第二十一册第 89 章`](../../book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)，内容只确认：DSA 的候选召回问题、MoE 的 total/active 分账、`slime` 的异步 rollout/trainer 系统问题，以及 Agentic Engineering 的可执行验证闭环。完整 indexer loss、top-k recall、生产 kernel、异步调度、policy freshness 控制、完整训练/后训练 recipe、硬件 profiling、线上工具接受率和独立复现仍待核验。
