# DeepSeek V3.2 官方资料摘记

核验日期：2026-09-29（本轮榜单复验并视觉核对 Figures 1–7；原始模型/实现资料于 2026-09-20 起核对）。本笔记只升级已经出现在 Artificial Analysis 的 DeepSeek V3.2；本轮 DataCurve DeepSWE 快照没有精确的 V3.2 行，因此不记录或迁移相邻版本的 DeepSWE 分数。

## 榜单锚点与快照

| 来源 | 结果 | 证据边界 |
|---|---|---|
| [Artificial Analysis DeepSeek V3.2](https://artificialanalysis.ai/models/deepseek-v3-2) | 当前复取标题为 `DeepSeek V3.2 (Non-reasoning)`；页面显示 2025-12-01、128K context、Intelligence Index `16.043537719683`、37B active / 685B total 的第三方目录字段 | 这是第三方配置页；`Non-reasoning` 是运行配置，release/参数/指数不是 DeepSeek 官方发布或裸模型能力证明；约 648B 是 2026-09-20 历史快照字段 |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 当前快照没有精确 `mini_swe_agent_deepseek_v3_2_*` 行 | 不把 DeepSeek V3.1、V4 或其他模型的 Pass@1、成本和 Agent steps 迁移给 V3.2 |
| Artificial Analysis 详情页历史快照（2026-09-18） | HTTP 200；3,391,821 bytes；SHA-256 `488c2c63fdb6d1746525f317222642d12cd57c7daf0f0abeba31653a6924ab7a` | 页面版本和测量字段随榜单变化；只用于候选发现与第三方配置复现 |
| DataCurve 历史快照（2026-09-18） | HTTP 200；268,571 bytes；SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` | 页面模型集合证据；无 V3.2 精确行 |

## 官方资料

- [DeepSeek V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)：官方组织 `deepseek-ai` 的公开权重模型卡；本轮 raw README 快照 7,361 bytes，SHA-256 `22b9f57e1ece2df27f9de0e71377e19481f7a6892eca8bc23e73fcdffe1d845c`。固定官方仓库 revision 为 `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`，`lastModified=2025-12-01T11:04:59Z`。
- [固定 revision 的 `config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json)：文件 1,552 bytes，SHA-256 `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1`；确认 `q_lora_rank=1536`、`kv_lora_rank=512`、61 层、前三层 dense、256 routed experts/top-8/1 shared、64 index heads、`index_head_dim=128`、`index_topk=2048`、`max_position_embeddings=163840`、YaRN factor 40、BF16 以及 FP8 `e4m3`/`ue8m0`、128×128 block 字段。
- [DeepSeek V3.2 技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)：本轮通过 7890 代理下载成功，PDF 907,086 bytes，SHA-256 `f6fda5753db7b106baa5eeb1286877f17e6a111354762f8aa53c7e6556498df7`。本地没有常用 PDF 工具，但已用标准库解压文本流完成正文抽取；抽取文本 252,018 bytes，SHA-256 `f082910a550666ad32c914db94056daa59a13b75a7ffd2bbd548695aa5cd043c`。数学公式、图表和字体编码存在抽取失真，以下只记录能由正文段落和公式上下文交叉核对的结论。
- [DeepSeek V3.2-Exp 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)：模型卡把 V3.2 的本地运行入口和结构说明指向该仓库；仓库 README 的 GitHub raw 页面本轮 HTTP 200，6,899 bytes。
- [DeepSeek API 文档 News](https://api-docs.deepseek.com/news)：官方导航列出 `DeepSeek-V3.2 Release 2025/12/01` 与 `DeepSeek-V3.2-Exp Release 2025/09/29`。直接访问 `/news/news251201` 本轮返回 Docusaurus fallback，而不是发布正文，因此不把该次响应当成正文证据。
- [V3.2 encoding 入口（模型卡说明）](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/encoding)：固定 revision 的 `encoding/encoding_dsv32.py` 为 14,317 bytes，SHA-256 `5e068c2ba2a6e5ebe37a49bb005650c507e7935d77a32f3f7c11ee071498b370`；可核验 DSML 风格 `<｜DSML｜function_calls>`/`<｜DSML｜invoke>`、`system`/`developer`/`user`/`assistant`/`tool` role、`<think>`/`reasoning_content`、`<function_results>`/`<result>` 以及字符串与 JSON 参数的区分。该脚本是编码/解析入口，不等于生产级容错 parser 或工具权限系统。

## 公开技术主线

### 1. DSA：让长上下文注意力具有检索阶段

模型卡把 DeepSeek Sparse Attention（DSA）列为第一项技术突破，并将它描述为面向长上下文、降低注意力计算复杂度而尽量保持能力的稀疏注意力机制。面试时应把它拆成“候选检索”和“主注意力读取”两个阶段：轻量 indexer 先估计历史位置相关性，主 attention 再访问有限候选。这个解释能说明复杂度收益来自哪里，也能自然引出候选漏检、indexer 开销、top-k 排序和局部依赖等风险。

技术报告进一步给出了可核验的训练边界：DSA 基于 MLA 的 MQA 模式实例化，使每个 MLA latent key-value entry 在一个 query token 的所有 query heads 间共享；dense warm-up 阶段冻结主模型，只训练 lightning indexer。Indexer 将各 attention head 的主注意力分数沿 head 求和并沿序列做 L1 归一化，使用 KL loss 对齐 indexer 分布，warm-up 为 1,000 步、每步 16 条 128K 序列、合计约 2.1B tokens。

随后进入 sparse training：引入细粒度 token selection，同时优化主模型和 indexer；indexer 输入从主计算图 detach，indexer 只接受 indexer KL loss，主模型只接受 language-modeling loss。该阶段每个 query 选择 2,048 个 KV tokens，学习率为 `7.3e-6`，训练 15,000 步、每步 480 条 128K 序列，报告给出合计约 943.7B tokens。报告的复杂度表述是：主模型 core attention 从 `O(L^2)` 降为 `O(L * k)`，但 lightning indexer 仍为 `O(L^2)`，只是相较 V3.1-Terminus 的 MLA 需要更少计算；实际收益依赖 kernel 和长上下文场景。

官方 Exp README 还给出工程边界：indexer logit（含 paged 版本）进入 DeepGEMM，sparse attention kernel 进入 FlashMLA，研究可读实现进入 TileLang；README 记录过 indexer 内 RoPE 非 interleaved 布局与 MLA 的 interleaved 布局不一致导致的实现问题。它们证明官方提供了实现入口，不等于本地已经复现性能。完整层排布、训练召回曲线、KV/indexer 字节账本和目标硬件 profiling 仍未公开核验。

### 2. Scalable RL：把后训练计算作为可扩展资源

模型卡第二项主线是 scalable reinforcement learning framework：通过更强的 RL protocol 和扩大 post-training compute，提升复杂推理能力；模型卡还把 V3.2-Speciale 描述为面向深度推理的高计算量变体。`V3.2` 与 `V3.2-Speciale` 的公开定位不同：前者支持工具调用，后者专注深度推理且不支持 tool calling。

这里应区分三件事：推理时增加 reasoning tokens/test-time compute，训练时扩大 rollout 或更新预算，以及一个不支持工具的专项 checkpoint。技术报告明确 V3.2 仍采用 GRPO，并把 reasoning、agent 和 human alignment 合并为一个 RL stage；specialist distillation 先覆盖数学、编程、一般逻辑推理、一般 Agent、Agent coding、Agent search 六个领域，再用后续 RL 缩小与专家模型的差距。

报告还披露了四个稳定化方向：用 importance-sampling ratio 修正 unbiased KL estimate；对与当前策略显著偏离且 advantage 为负的序列做 off-policy sequence masking；在 MoE 中保留 rollout 时的 expert routing（Keep Routing）；训练时沿用旧策略采样的 top-p/top-k truncation mask（Keep Sampling Mask），避免 old/current policy 的 action space 不一致。这些是报告披露的 V3.2 训练策略，但仍不是完整奖励权重、rollout 数量、优化器和部署 recipe。

### 3. Large-scale agentic task synthesis：把工具使用变成可训练数据

模型卡第三项主线是 large-scale agentic task synthesis pipeline，用系统化方法大规模生成训练数据，将 reasoning 融入 tool-use 场景，目标是改善复杂交互环境中的泛化和指令遵循鲁棒性。对面试而言，关键不是背一句“有 Agent 数据合成”，而是问清数据闭环：任务定义、环境/工具接口、轨迹生成、结果验证、困难样本筛选、失败轨迹处理和最终后训练目标分别由谁负责。

技术报告补充了数据规模和验证闭环：code agent 24,667 个任务、search agent 50,275 个任务、general agent 4,417 个任务、code interpreter 5,908 个任务；环境可以是真实或合成，prompt 可以来自互联网抽取或合成。Search agent 使用多 Agent 搜索、异构回答模型和多轮验证；code agent 从 GitHub issue-PR 对构造可执行环境，要求 gold patch 产生非零 F2P 且零 P2F；general agent 自动合成 1,827 个“难解易验”的环境，并只保留 `pass@100` 非零样本。报告还强调 solution function 只能通过工具接口工作，不能直接访问数据库，verifier 独立检查结果。

这些数字和流程属于官方技术报告的训练数据描述，不是 DataCurve 的 Agent 评测结果，也不能证明合成数据没有污染或 shortcut。环境来源、过滤器和 verifier 的生产级完整实现仍未公开。

### 4. “Thinking with tools”是协议和训练边界的交叉面

模型卡指出 V3.2 相比前代更新 chat template，增加 revised tool-calling format 和 “thinking with tools”。它提供 `encoding` 目录帮助把 OpenAI-compatible messages 编码为模型输入并解析输出，说明模型的 reasoning 内容、工具调用和最终回答需要在模板层保持明确边界。

公开模板要点包括：

- 不提供 Jinja chat template，而是用独立 Python encoder/parser；
- 新增 `developer` role，但模型卡限定它只服务 search-agent 场景，官方 API 不接受该 role；
- V3.2-Speciale 不支持 tool calling，不能把 reasoning checkpoint 的工具协议套给 Speciale；
- 官方 parser 只处理格式良好的字符串，不能当生产级容错解析器。

这些是协议/仓库事实，不等于内部模型能稳定执行工具。生产 harness 仍须做 schema 校验、权限校验、调用超时、结果回灌、格式异常恢复和 verifier 门禁。

技术报告还给出了 reasoning context 的保留规则：如果后续轮次只追加 tool message（例如 tool output），历史 reasoning content 会继续保留；只有新的 user message 进入对话时才丢弃历史 reasoning，同时保留工具调用与结果历史。这样可以避免每次工具返回后从头重复推理，但把工具交互模拟为 user message 的 Roo Code 或 Terminus 类 harness 可能无法获得该收益，报告建议这类架构使用 non-thinking 模型。这个结论是上下文协议与 harness 的耦合边界，不是“保留思考文本就一定更强”。

## 报告中可直接用于面试的数量级

下列数字来自 DeepSeek V3.2 技术报告正文，绑定报告版本，不是 Artificial Analysis 或 DataCurve 分数：

| 项目 | 报告披露 | 解释边界 |
|---|---:|---|
| Indexer dense warm-up | 1,000 steps、约 2.1B tokens | 冻结主模型，只优化 lightning indexer |
| Sparse training | 15,000 steps、约 943.7B tokens | 每 query 选择 2,048 个 KV tokens；主模型和 indexer 分离优化 |
| Code agent | 24,667 tasks | GitHub issue/PR 对构造可执行环境，gold patch 以 F2P/P2F 检查 |
| Search agent | 50,275 tasks | 多 Agent 搜索、异构回答与多轮验证 |
| General agent | 4,417 tasks、1,827 environments | 合成环境、工具、任务和 verifier，保留非零 pass@100 |
| Code interpreter | 5,908 tasks | Jupyter Notebook 覆盖数学、逻辑和数据科学问题 |

这些是训练数据与训练过程的发布方描述；它们不能替代独立复现，也不能直接解释为线上成功率。

## 开放性与配置边界

- 模型卡和权重仓库标注 MIT License；license 结论绑定该官方仓库，不从 Artificial Analysis 的开放性字段推导。
- 固定 `config.json` 与模型 metadata 共同标注 `DeepseekV32ForCausalLM`/`deepseek_v32`、每 token 8 个 experts 和 FP8 quantization 字段；配置字段可以作为该 revision 的结构证据，但仍不自动推出完整训练 recipe、生产 kernel 或部署性能。
- 模型卡写明 V3.2 与 V3.2-Speciale 的 model structure 与 V3.2-Exp 相同；这只支持结构族关系，不证明权重、训练数据、RL 配方和能力完全相同。
- Artificial Analysis 页面把约 648B/37B 和 128K 作为第三方目录字段；模型卡没有在本轮 README 中给出对应完整参数账本，因此正式资料中将其标为第三方字段。

## 面试追问

1. DSA 为什么不是“只保留 top-k 就结束”？应回答 indexer、候选召回、主 attention、局部依赖和漏检代价是不同层；还要测 recall、延迟、KV/indexer bytes 与长上下文失败。V3.2 报告进一步给出 2,048 KV top-k、dense warm-up 与 sparse training 的分离，以及主 attention 降阶但 indexer 仍需计算的边界。
2. `thinking with tools` 与普通 function calling 的区别是什么？应回答 reasoning 状态、工具调用格式、结果回灌和最终回答共用一个模板协议；它不授予工具权限，也不替代宿主 parser。
3. 为什么 V3.2-Speciale 不能直接用于 coding agent？官方模型卡明确该 variant 专注 deep reasoning、无 tool-calling；Agent 运行时需要另外的工具协议和执行器。
4. Agentic task synthesis 最容易发生什么数据问题？任务和环境可能泄漏答案、verifier 可能奖励 shortcut、成功轨迹可能偏向单一工具路径；应做污染检查、独立 verifier、环境隔离和失败轨迹审计。
5. 为什么不能把 AA 的指数和 DataCurve 的 Pass@1 合并？前者是 Artificial Analysis 的模型配置评测，后者是 `mini-swe-agent` 加工具、环境和 verifier 的组合系统；本轮 V3.2 还没有 DataCurve 精确行。

## 待核验与书系映射

- 待核验：Figures 1–7 均已取得原始图/面板并视觉复核；arXiv v1 HTML 中全部编号公式 Eq. (1)–(9) 的 TeX annotations 已逐式文本核对，但未在 PDF 页面视觉核验；未编号行内数学表达式未做穷尽审计。完整 DSA 层排布和 kernel 源码行为、indexer 训练召回曲线、KV/indexer 字节账本、完整 RL 超参/奖励权重/rollout 配方、合成数据污染审计、线上 tool-call acceptance rate、硬件 profiling 和独立 benchmark 也仍待核验。
- 不新增 V3.2 专属独立架构册章：DSA 已由第二十一册第 19 章承接，19.28--19.34 覆盖 DSA、实现、kernel、serving、encoding 和榜单证据分层；19.35 补 Search Agent context management，19.36 补 Figure 2 的 indexer/top-k 与 Core Attention 分工，19.37 补 Figure 3 的成本曲线及部署证据边界，19.38 补 Figure 4 的 thinking-retention/harness 边界，19.39 补 Figure 5 合成 Agent 数据 RL 实验的图表证据边界，19.40 补 Figure 7 MLA MHA/MQA 两种执行形态，19.41 补 Figure 1 多指标/多评测口径边界，19.42 补 DSA Eq. (1)–(4) 的检索与 indexer 蒸馏目标，19.43 补 scalable RL Eq. (5)–(9) 的目标、KL 估计和 off-policy 稳定化边界；第二十册第 18 章保留通用折叠策略映射。
- 研究笔记和现有第 19 章专题段共同作为本轮交付；模型清单、来源索引、榜单解释、题库、练习、术语、项目、知识图谱、`plan.md` 与 `progress_v3.md` 均同步记录模型和证据边界。

## 2026-09-20 排行榜复验与下一锚点确认

本轮作为 GLM-5 之后的顺序确认，重新抓取 [Artificial Analysis DeepSeek V3.2](https://artificialanalysis.ai/models/deepseek-v3-2) 和 DataCurve DeepSWE：

- Artificial Analysis 通过 `10.24.27.134:8098` 与 `10.237.126.170:1234` 均返回 HTTP 200；两个详情快照均为 `3,625,720` bytes，SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`，逐字节一致。当前页面仍为 `DeepSeek V3.2 (Non-reasoning)`，128K context、第三方 Intelligence Index `16.043537719683`、约 648B total/37B active；这些是 AA 配置/目录字段。
- DataCurve `/tmp/deepswe-20260920-8098.out` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，与本轮排行榜首页快照一致；页面只有 `mini_swe_agent_deepseek_v4_flash_max` 和 `mini_swe_agent_deepseek_v4_pro_max` 等行，没有精确 `mini_swe_agent_deepseek_v3_2_*` 行。
- 因此 DeepSeek V3.2 被选为 GLM-5.3 之后的当前活动锚点：仍是 **AA 单榜资料级闭环**，不记录或迁移 V3.1/V4 的 DeepSWE 分数。官方模型卡、技术报告和 V3.2-Exp 入口已经完成核验；本轮已把实现证据补进第 19 章，仍待 PDF 公式/图表复核、完整 kernel/层排布、硬件 profiling、线上 tool acceptance 和独立 benchmark。

## 2026-09-20 官方实现与 serving 复核

本节把“官方论文/模型卡描述”与“官方实现代码/部署 recipe”分开记录。实现代码主要来自 `DeepSeek-V3.2-Exp` 的 inference demo、TileLang 示例、DeepGEMM/FlashMLA PR 和 vLLM recipe；它们证明了公开实现路径，不等于最终 V3.2 权重、所有生产后端或目标硬件性能已经被本地复现。

### 1. 最终 V3.2 与 V3.2-Exp 实验配置不能混写

| 对象 | 本轮可核验字段 | 证据边界 |
|---|---|---|
| 最终 V3.2 模型卡/固定公开模型 artifact | `q_lora_rank=1536`、`kv_lora_rank=512`、61 layers、前三层 dense、256 routed/top-8/1 shared、64 index heads、`index_head_dim=128`、`index_topk=2048`、163840 positions、YaRN factor 40、BF16/FP8 配置 | 固定 revision `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` 的 `config.json` 字段；不自动证明所有生产 kernel |
| V3.2-Exp inference demo | `dim=7168`、`n_layers=61`、`n_dense_layers=3`、`n_heads=128`、`n_routed_experts=256`、`n_activated_experts=8`、`q_lora_rank=1536`、`kv_lora_rank=512`、`index_n_heads=64`、`index_head_dim=128`、`index_topk=2048`、FP8/`ue8m0` | 这些字段来自实验版 inference config；本轮快照 `/tmp/deepseek-v32-infer-config-20260920.json`，SHA-256 `f9fe074cdef8fdcc0feabbe635ff7c072a65d273a42a6bc207fe425439dff835` |

本轮纠正：最终 V3.2 固定配置与 V3.2-Exp inference demo 都显示 `q_lora_rank=1536`，此前把最终字段误读为另一数值的说法不能保留。两者仍不能无条件合并：最终模型 artifact 的固定 revision/config、实验 demo 的实现快照、kernel 和 serving recipe 各自回答不同问题；字段相同不代表权重、训练数据、RL 配方和能力完全相同。

### 2. Inference demo 暴露出的 DSA 执行路径

V3.2-Exp 的 `model.py` 将 indexer 与 MLA 主路径写成可读的参考实现：

- Indexer 对 query/key cache 使用 FP8；代码明确使用 **non-interleaved RoPE**，而 MLA 的 RoPE 布局不同，不能把两者的旋转维度实现直接复用。
- indexer 通过 `fp8_index` 计算 index score，对 score 施加 causal mask，再执行 `topk(min(index_topk, end_pos))`；这里的 `topk` 是候选位置选择，不是最终 attention 输出。
- MLA 保存 `kv_lora_rank=512` 的 latent KV cache 和独立 positional cache。prefill 路径使用 MHA，decode 路径使用 MQA；实际部署使用 FP8 KV cache，demo 用量化/反量化模拟精度。
- 选出的 top-k 位置被转换成 attention mask，后续主 attention 只读取这些 KV；因此端到端链路仍是“index score -> causal/top-k -> gather/masked sparse MLA”，不是单独的排序优化。

本轮本地代码快照为 `model.py` SHA-256 `bfe5b89186b579910b6d59fa7e0cf1f7a75ceb4ec5d0f5b54f0ccbe77cdb6a63`、`kernel.py` SHA-256 `92abc46cb6457b8cbe105be66f26f9a119aeb86c872e4521d853d5177d181868`。TileLang 的研究实现把执行链明确拆成 **Lightning Indexer -> Top-k Selector -> Sparse MLA**；top-k selector 使用 radix-sort/histogram 两阶段筛选，sparse MLA gather 被选 KV，并在选中索引上执行 causal mask。pipelined 版本使用 producer/consumer warp group 和 double buffering；FP8 sparse MLA 还涉及 K-major V layout 与 shared-memory transpose。这些是源码/README 的实现描述，不是本地性能复现结果。

### 3. CUDA kernel 与 vLLM serving 证据

- [DeepGEMM PR #200](https://github.com/deepseek-ai/DeepGEMM/pull/200) 增加 FP8 MQA logits 和 paged MQA logits 路径，并覆盖 SM90/SM100；vLLM recipe 说明 DeepGEMM 同时用于 MoE 和 MQA logits，MQA logits 路径是该 recipe 的必要组件。
- [FlashMLA PR #98](https://github.com/deepseek-ai/FlashMLA/pull/98) 增加 sparse prefill、sparse FP8 decode、SM90 sparse MLA、metadata/combine、测试和量化支持。它是高性能 CUDA 实现入口，不等于所有 GPU 型号都能得到相同结果。
- [TileLang DeepSeek V3.2 示例](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32) 适合研究可读性；[V3.2-Exp README](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp#inference) 将 inference demo、DeepGEMM、FlashMLA 和 TileLang 作为不同实现层次推荐给社区。
- [vLLM DeepSeek-V3.2-Exp recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html) 推荐在 kernel 主要按 `TP=1` 优化时使用 `DP=8, EP=8, TP=1`；普通 tensor parallel 是更稳妥但性能可能较差的 fallback。recipe 默认 `max-num-seqs=1024`，遇到 FlashMLA 配置错误可试 `256` 或更小；默认偏向 FP8 KV cache，短请求可选择 BF16 KV cache。
- 同一 vLLM recipe 给出 GSM8K 5-shot `0.9591`、20-shot `0.9538` 的本地 serving/harness 结果。它们绑定 V3.2-Exp 权重、vLLM、lm-eval、prompt 和运行参数，只能记录为 recipe 结果，不能写成 AA 指数、DataCurve Pass@1 或最终 V3.2 裸模型分数。

### 4. 实现层面新增的面试边界

1. **为什么 indexer 的 RoPE 不能直接照搬 MLA？** 因为官方 demo 明确把 indexer 的 RoPE 标为 non-interleaved，而 MLA 使用另一种布局；旋转维度的数学名称相同，不代表内存/通道排列相同。复现时必须固定布局、权重切分和 cache 格式。
2. **为什么 indexer score 仍可能成为长上下文瓶颈？** DSA 把主 attention 从全量 KV 读取压到候选 KV，但 indexer 仍要对历史位置计算分数；FP8 GEMM、tile、causal mask 和 top-k selector 的成本都要算进 TTFT/TPOT。
3. **prefill MHA 与 decode MQA 的差异是什么？** prefill 一次处理多 query，参考实现保留 MHA 形态；decode 每步 query 数少，复用 latent KV 与位置 cache，以 MQA 形态访问历史。不能只用 decode 的 cache 字节数估算 prefill 峰值。
4. **为什么 radix top-k 不是普通 `torch.topk` 的同义替换？** TileLang 代码先用 histogram 找阈值桶，再对阈值桶做有限轮次 radix refinement，目标是避免对整个序列完全排序；它仍需验证 ties、causal 边界、paged KV 和不同长度 batch。
5. **EP/DP/TP 是如何由 kernel 约束决定的？** recipe 的 `DP=8, EP=8, TP=1` 是特定 kernel/硬件的部署建议，不是模型结构的唯一并行方式；TP fallback 的正确性和性能要分别测试，并记录通信、显存、warmup、batch 和故障恢复。
6. **如何给 serving benchmark 分层？** 模型 artifact、inference implementation、kernel、vLLM recipe、lm-eval harness 和指标必须逐层绑定。recipe 的 GSM8K 数字不能替换模型卡 benchmark，更不能填补 V3.2 缺失的 DataCurve 精确行。

### 5. 本轮仍未确认的内容

已确认的是公开实验实现的关键执行路径；仍未确认完整最终 V3.2 production kernel 覆盖、indexer 召回曲线、FP8 与 BF16 的端到端误差/延迟曲线、各 GPU 代际 profiling、EP/DP 在真实并发下的 p99、线上 tool-call acceptance rate 和独立 benchmark。下一轮若继续该锚点，应优先固定具体 commit、权重 artifact、GPU、batch、context、cache dtype 和 harness，再做可复现实验。

## 2026-09-21 当前榜单与官方 README 复核

本轮只复核已经由 Artificial Analysis 发现的 DeepSeek V3.2，没有从官方仓库、vLLM 文档或 HF 目录另发现模型。Artificial Analysis 详情页快照为 3,638,730 bytes、SHA-256 4dbd4f6cb11f25092a221ef0490a910f91f43885330fe34cc38c212c4d7470c3；结构化对象仍是 `deepseek-v3-2` / DeepSeek V3.2 (Non-reasoning)，release date 为 2025-12-01，third-party fields 为 685B total、37B active、128K context、Intelligence Index 16.043537719683、输入/输出每百万 token 0.28/0.42 美元。当前对象没有可用的 timescale output-speed 或 TTFT 字段，因此不补写速度和首 token 延迟。

与 2026-09-20 的 3,625,720 bytes、SHA-256 7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052 以及当时页面显示的约 648B total 相比，685B 是第三方目录/页面数据漂移。它不构成 checkpoint、训练或架构变化证据；旧快照继续保留为历史测量。DataCurve 当前快照仍为 268,571 bytes、SHA-256 67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870，并且没有精确的 `mini_swe_agent_deepseek_v3_2_*` 行，所以不迁移任何相邻 DeepSeek 版本的 Agent 分数。

V3.2-Exp 官方 README 快照为 6,899 bytes、SHA-256 dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74。README 明确 Exp 基于 V3.1-Terminus，并用对齐训练配置比较 reasoning without tool use 与 agentic tool use；表格是 DeepSeek 发布方对照，不是 DataCurve 或独立复现。README 的 2025-11-17 更新还明确指出：indexer 的 RoPE 必须使用 non-interleaved layout，MLA 使用 interleaved layout，旧 inference demo 的布局差异可能造成性能下降，更新代码已修复。

README 同时把实现入口按层次分开：TileLang 用于可读的研究 kernel，DeepGEMM PR #200 提供 indexer logits（含 paged 版本）的高性能 CUDA 路径，FlashMLA PR #98 提供 sparse attention 路径；SGLang 入口公开 dsv32、dsv32-rocm、dsv32-a2、dsv32-a3 镜像及 tp 8、dp 8、enable-dp-attention 的启动命令。这些证据支持“官方实现路径可追溯”，不支持“本机完整权重、目标 GPU 性能或线上 tool acceptance 已验收”。

README 链接的 vLLM recipe 当前通过 1234 返回 HTTP 404。该负证据只说明本次 URL/线路在当前时点不可取得，不能推导 vLLM 没有实现，也不能删除已有 recipe 证据；后续应固定可访问的 recipe revision 后再核对依赖和 benchmark。HF 当前固定文件与论文 PDF 通过同一线路出现 503 时，也只记为传输/站点访问失败，不改写已固定 revision 的历史哈希。

当前状态仍为 **AA 单榜资料级闭环**。本轮新增的是榜单目录漂移解释和 Exp README 的 RoPE/kernel/SGLang 证据分层；完整最终 production kernel、召回/误差曲线、GPU profiling、线上 tool-call acceptance、独立 benchmark 和完整 RL recipe 仍待核验。

## 2026-09-28 7890 复验与 Search Agent context management

本轮只沿 Artificial Analysis 已有的 `deepseek-v3-2` canonical 锚点补证，没有从论文或工具链发现新模型。当前工作区显式经 `10.24.27.134:7890` 访问百度、Artificial Analysis、DataCurve 和 arXiv 均成功；对这几个 URL 的可达性只作本时点记录。

- Artificial Analysis 详情页 HTTP 200，3,647,642 bytes，SHA-256 `239b8fef11d18d7b06e5e7c177ed76cbbfd29e07d795d83c5dc3e6ac8acef0b8`；当前页为 `DeepSeek V3.2 (Non-reasoning)`，2025-12-01、128K、685B total/37B active，Intelligence Index `16.043537719683`。这些均为第三方目录/provider 字段。9 月 20 日的 648B 继续作为历史页面快照，不解释为模型扩容；同日另一个同尺寸快照 hash 不同，hash 只绑定具体抓取字节。
- DataCurve DeepSWE HTTP 200，268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；当前页面没有精确 `mini_swe_agent_deepseek_v3_2_*` 行。没有迁移相邻 DeepSeek 型号的 Pass@1、成本或 Agent steps。
- [arXiv v1 HTML](https://arxiv.org/html/2512.02556v1) HTTP 200，295,170 bytes，SHA-256 `5da74d488b218a45a995838b94feb95aacc63c03b806f8462496bfd4bceb07ef`；该固定版本的 [PDF](https://arxiv.org/pdf/2512.02556v1) 为 980,616 bytes、SHA-256 `2bec0671778769c159ec389412727d1f3d4889fe1c71564b61edaa24705bd17b`。它与 HF 模型卡 `assets/paper.pdf`（907,086 bytes、SHA-256 `f6fda5753db7b106baa5eeb1286877f17e6a111354762f8aa53c7e6556498df7`）是不同抓取 artifact；本文 §4.4 结论直接绑定 versioned arXiv HTML，不宣称两个 PDF 字节相同。

### 论文 §4.4：Search Agent 的上下文管理

论文在 Search Agent 工具轨迹使用量超过 context window 80% 时触发上下文管理。对比项为：

1. `Summary`：摘要已溢出的轨迹并重新开始 rollout；
2. `Discard-75%`：丢弃轨迹中最早的 75% tool-call history；
3. `Discard-all`：重置上下文并丢弃此前全部 tool-call history；
4. `Parallel-fewest-step`：采样 N 条独立轨迹，选择步骤数最少的一条，作为并行 test-time compute 基线。

Search Agent 评估使用 standard commercial search API。作者称 V3.2 最大 context 为 128K，约 20% 以上测试案例超过此上限；无 context management 的参考 BrowseComp 分数为 51.4。论文 Table 2 将 BrowseComp 标为 Pass@1，并以星号标注使用 context management 的结果，列为 `51.4/67.6*`；§4.4 另称 `Discard-all` 得分 67.6，与 parallel scaling 可比但使用显著更少 steps。作者还报告 `Summary` 平均扩展到 364 steps，performance improvement “up to 60.2”。该句没有百分号；已视觉核对的 Figure 6 纵轴为 Browsecomp，图中 Summary 曲线约在 364 real steps 达到 60.2，因此应把 60.2 作为该图上的分数/指标值理解，而不是 `+60.2%` 的相对增幅。图中没有逐点数字标签，不从曲线像素外推精确点值。

Figure 6 视觉复核（2026-09-28）：通过 `10.24.27.134:7890` 获取 arXiv v1 的 [search.svg](https://arxiv.org/html/2512.02556v1/search.svg)，75,689 bytes、SHA-256 `e2fb1029cf88458b2881d49546c1b9db4dd426fb4d2b3337dd6897b9c6fdb7a3`；与先前经 1234 获取的 SVG 逐字节一致。渲染后确认横轴为 Real Steps、纵轴为 Browsecomp，图例含 `Summary`、`Discard-75%`、`Discard-all` 和 `Parallel-fewest-step` 四种策略。曲线定性支持作者关于串行 context management 与并行 test-time compute 的效率/扩展性权衡；精确报告值仍以正文/Table 2 为准，图中未标注的散点不作数字化录入。

面试中的可迁移结论是：上下文管理是 Agent harness 的 test-time compute 与状态保留策略，不是模型永久记忆。摘要策略保留压缩后的历史但引入摘要失真，丢弃前缀/全部工具历史省 token 却可能失去来源链和已发现证据；实际系统需在外部持久化来源 URL、关键观察、未决问题和副作用状态，并用相同任务预算对比成功率、steps、工具调用成本、证据可追溯性与 artifact 完成率。上述 51.4/67.6/364/60.2 是论文中的 Search Agent/BrowseComp harness 结果，不是 DataCurve DeepSWE、Artificial Analysis 指数、裸模型分数或通用 API 行为保证。

## 2026-09-29 7890 双榜复验、Figures 1–7 视觉核验与 DSA 公式文本审计

- 当前工作区经 `10.24.27.134:7890` 对百度与 arXiv Figure 2 均曾取得 HTTP 200；其后代理连接短暂失败，获批重试成功。该记录描述不同请求时点，不把一次成功泛化为持续可用保证。
- Artificial Analysis `/zh` 快照为 1,694,614 bytes，SHA-256 `04f62f3f39c1f65c3c1d8dd564782f9d7d20fb506fa671ee75fcfaf4e9482f9f`；相较留存基线，唯一模型路由从 55 增至 62，无移除。新增路由中，GLM-5.3-Flash、Kimi K3、Qwen3.8 2.4T A95B、Qwen3.8 27B 均已是盘点内重点 canonical；其余三项属于暂不跟踪厂商。没有新增重点 canonical 锚点。
- DataCurve DeepSWE 快照为 268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；70 个唯一 `mini_swe_agent_*` 配置 ID 与留存快照一致，仍无精确 V3.2 行，不迁移相邻模型成绩。

### Figure 2：Lightning Indexer、Top-k 与核心注意力的分工

通过 `10.24.27.134:7890` 获取 arXiv v1 的 [v32_arch.svg](https://arxiv.org/html/2512.02556v1/v32_arch.svg)，HTTP 200，269,427 bytes，SHA-256 `1e6bc6c61ea26ae2b8528edb1eab14874832470fd38f115fe9b8feb615facb9a`；渲染为 1600×833 后完成视觉复核。图中 Lightning Indexer 计算索引分数，Top-k Selector 据此筛出候选 KV entries；被选条目进入 Multi-Query Attention（Core Attention），与主注意力 query 一起产生 attention 输出。图注、§2.1 与 Eq. (2) 的叙述相互印证。面试时应明确：index score/top-k 是候选检索与筛选信号，不是 attention 输出，也不取代 MLA 主注意力。图示解释模块关系，不证明某个 production kernel 的完整行为或性能。

### Figure 3：长上下文 prefill/decode 成本曲线

从同一固定版本的 arXiv v1 页面取得 Figure 3 两个 panel：[prefilling SVG](https://arxiv.org/html/2512.02556v1/cost_prefilling.svg)，33,398 bytes、SHA-256 `b045c26eb19d92325de7e86aabec905a9ac8bdb6729db94e04c8701693b19705`；[decoding SVG](https://arxiv.org/html/2512.02556v1/cost_decoding.svg)，31,868 bytes、SHA-256 `5894e01515f7f9e9ef9045ae8e30a092f5cc1e4229936c493c66c4748127c61d`。两图均渲染为 1600×1200 并视觉核对：横轴是 Token Position，纵轴是 Cost Per Million Tokens；蓝线为 DeepSeek-V3.1-Terminus，橙线为 DeepSeek-V3.2。随着 token position 增长，V3.2 的 prefill/decode 曲线都比 V3.1 平缓，长位置上的成本差距明显；极短位置的曲线接近并有交叉，不能宣称每个长度点都更便宜。图上没有可支持逐点抄录的数字标签，不从像素估算精确数值。

报告正文称成本基于 H800 上实际部署服务的 benchmark，并按每 GPU 小时 2 美元的租赁价格估算；短序列 prefill 还特别采用 masked MHA mode 模拟 DSA。故 Figure 3 是绑定作者服务实现、H800 与该租赁假设的发布方成本测量，不是官方 API 价目、硬件无关结论或独立复现。结合主 attention 从 `O(L^2)` 到 `O(Lk)`、但 indexer 仍为 `O(L^2)` 的正文边界，面试应追问 kernel、短/长上下文路径和服务账本，而不能把复杂度式直接换算成全模型成本比例。

### Figure 4：工具调用时 thinking 的保留边界

arXiv v1 的 [Figure 4 JPEG](https://arxiv.org/html/2512.02556v1/figures/template.JPEG) 为 76,981 bytes、SHA-256 `58623875cc487b3cbbd60955c14801c07d5f526d2b2e875795e3bf5ace511733`；原图 1280×671，已目视核验。图示显示：同一 user turn 内追加 tool call/result 后，前序 `Thinking` 块继续保留并可继续生成 reasoning/tool call；形成 Answer 1 后收到新的 user message，下一轮输入保留此前工具调用、工具结果和回答，但不再包含旧 thinking 块。它与 §3.2.1 的文字规则一致：工具消息续接不清除历史 reasoning，新 user message 到来时清除 reasoning，而工具轨迹仍保留。

面试边界：这是模型上下文/消息类型驱动的回放策略，不是模型永久记忆，也不代表任意客户端都按该方式构造 history。若 harness 把工具结果包装成 user message（报告举 Roo Code、Terminus 为例），消息类别可能触发不同保留行为；应审计实际 role、上下文构造和版本，而不是仅看“工具调用”语义。

### Figure 5：合成 Agent 数据上的 RL 训练曲线

arXiv v1 的 [synthesis-rl-plot.png](https://arxiv.org/html/2512.02556v1/figures/synthesis-rl-plot.png) 经 `10.24.27.134:7890` 获取，为 222,137 bytes、SHA-256 `2107344cc2002f51bc1922df0bdcabd7afa74bb0f6cb5d3aa996f33e3f0d7cc3`，尺寸 1432×1024，已视觉复核。图例对比 `DeepSeek-V3.2-RL-Synthetic-Data`（蓝色训练曲线）、`DeepSeek-V3.2-SFT`（绿色基线）和 `DeepSeek-V3.2-Exp`（黑色基线）；横轴为 Steps，分面覆盖 Tau2-Bench Airline/Retail/Telecom/Overall、MCP-Mark Filesystem/PostgreSQL 与 MCP-Universe Financial Analysis/Location-Navigation/3D-Designing。曲线随训练步骤总体上升，但不同分面波动和幅度不同；不从像素读出未标注的精确成绩。

§4.3 将该实验限定为从 V3.2-SFT checkpoint 出发、只用合成 general-agent tasks、non-thinking mode 进行 RL，再与 SFT 及仅在 search/code environments 做 RL 的 V3.2-Exp 比较。Figure 5 支持“该发布方实验中的合成 Agent 训练在若干不同环境评测上出现提升”的观察，不证明最终 V3.2 的独立 benchmark、真实环境泛化因果、训练数据无污染或完整 RL recipe 已公开。结果属于发布方训练消融/评测，不是 AA 或 DataCurve 分数。

- Figures 1–7 已视觉复核；其余论文公式仍未全部检查。完整 production kernel、召回/误差曲线、硬件 profiling、线上 acceptance、独立 benchmark 与完整 RL recipe 继续待核验。

### Figure 7：MLA 的 MHA 与 MQA modes

从 arXiv v1 HTML 的 Appendix A 定位到 Figure 7 的真实图资源（不是猜测路径 `figure7.svg`）：[MHA panel](https://arxiv.org/html/2512.02556v1/MLA-MHA.svg)，HTTP 200，244,227 bytes，SHA-256 `afd21a50bbaef58314036862cb6ce44dca81a9d42a414c0969074fbf954b32b4`；[MQA panel](https://arxiv.org/html/2512.02556v1/MLA-MQA.svg)，HTTP 200，224,311 bytes，SHA-256 `c6aa4f3e2ea222d2a75ef000dca2f9e8c48a820116957c6a741cbd0f253d77c3`。两图均渲染后视觉核验。

图示对比 MLA 中 MHA 与 MQA 的表示：MHA 从共享 latent KV `c_t^KV` 经各头投影形成 per-head K/V；MQA 让 query heads 共享 latent KV entry 与位置 key，再产生各头 attention output。正文称 latent vector（MLA 的 KV entry）由当前 token 的 query heads 共享。图注只将 V3.1-Terminus 的 training/prefill→MHA、decode→MQA 作为特定阶段安排；V3.2-Exp inference demo 另有对应 prefill/decode 路径证据。不要把它写成所有 V3.2 serving 后端都必须遵循的合同，也不能从图直接推出实际 cache bytes、吞吐、kernel 覆盖或质量等价。此前试探 `/figures/figure7.svg` 返回 404 是路径猜错；按 HTML 中的真实对象 URL 复取两图均成功。Figures 1–7 已视觉复核，其他论文公式仍未全部检查。

### Figure 1：多指标图表中的评测协议

arXiv v1 的 [Figure 1 SVG](https://arxiv.org/html/2512.02556v1/v32_performance.svg) 经 7890 获取，HTTP 200，128,338 bytes，SHA-256 `14354740f4d54692b6af6323cc12d3a5f0e0f937bc2b7dd65021307bd7820973`；渲染后视觉核验图例、Reasoning/Agentic 分组、各 benchmark 标签及双纵轴。Codeforces Rating 使用右侧量纲，其他 accuracy/Pass@1 使用左侧百分比轴，不能跨量纲看柱高排总名次。

图注注明 HMMT 是 February 2025 赛次、HLE 是 text-only 子集。§4.1 写明评测温度 1.0、上下文 128K，数学任务使用统一的 step-by-step 模板；另以 HLE 官方模板评估 V3.2-Thinking，报告 23.9，而 Figure 1 同一模型的 HLE bar 标为 25.1。这是模板敏感性的直接例子。Tool-use 使用 standard function-call format + thinking mode；MCP-Universe/MCP-Mark 使用内部环境，作者指出与官方环境可能略有差异。故 Figure 1 是发布方、特定 task/prompt/environment/model mode 下的评测结果，不是 AA/DataCurve 数据、独立复现或统一模型排行榜。该发现映射到第二十一册第 19 章 19.41。

### Eq. (1)–(4)：index score、候选集和 indexer loss

对照 arXiv v1 HTML 的 MathML `annotation encoding="application/x-tex"` 核对了 Eq. (1)–(4)：Eq. (1) 是 indexer 多头加权 ReLU 点积的排序分数，不是 softmax 概率；Eq. (2) 按 Top-k 取 latent KV `c_s`，再由主 attention 计算 `u_t`；Eq. (3) 以 dense attention heads 求和并沿序列 L1-normalize 的 `p_{t,:}` 监督 indexer softmax；Eq. (4) 将 KL 对齐限定到 `S_t` 选中的位置。Sparse-stage `p_{t,S_t}` 是否在截取后另行归一化，展示式/该段正文没有明确写出，不补作假设。报告说明 indexer input detach、indexer 只由 `L_I` 更新、主模型只由 LM loss 更新。

经 7890 下载 arXiv v1 PDF：980,616 bytes / SHA-256 `2bec0671778769c159ec389412727d1f3d4889fe1c71564b61edaa24705bd17b`。此环境缺少可用 PDF renderer/extractor（GDK thumbnailer 不能识别 PDF），所以记录为“HTML TeX 源文本核对”，没有声称 PDF 公式视觉复核。对应第二十一册第 19 章 19.42；实现级的 loss normalization 细节仍待公开训练代码/补充材料确认。

随后经 7890 检查 [DeepSeek-V3.2-Exp 官方仓库网页](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) 和 [raw README](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/README.md)：分别 HTTP 200、295,427 bytes 与 6,899 bytes；README SHA-256 `dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74`，与既有快照一致。可见根目录列有 README、report PDF、license、cost image 和 `inference/`；README 指向 inference demo 及 TileLang/DeepGEMM/FlashMLA kernel，没有 trainer/loss 代码入口。GitHub API tree endpoint 返回 403，故结论仅限于当前可见公开 V3.2-Exp 仓库；不能外推为其他官方仓库或私有训练代码不存在。`p_{t,S_t}` 的归一化仍未由代码解决。

### Eq. (5)–(9)：GRPO token 目标与稳定化策略（2026-09-29）

本轮沿既有 DeepSeek V3.2 锚点继续核对同一份 arXiv v1 HTML；7890 返回 HTTP 200，295,170 bytes，SHA-256 5da74d488b218a45a995838b94feb95aacc63c03b806f8462496bfd4bceb07ef，与已留存快照一致。直接从每个 numbered equation 对应的 application/x-tex annotations 核对 Eq. (5)–(9)，没有把 HTML 源文本验证说成 PDF 公式视觉检查。

- Eq. (5) 的目标先对 group 中 G 条 response 求平均，再在每条 response 内按长度 |o_i| 对 token 目标取均值；token 项包含 PPO-style clipping 与 KL penalty。
- Eq. (6) 的 importance ratio 是 current/old 的逐 token 条件概率比。正文把每条 response 的 outcome reward 减去 group mean 定义为 advantage；展示式没有标准差除法，故公开公式支持“组内中心化”，不支持擅自改写成 z-score。
- Eq. (7) 用 pi_theta/pi_old importance weight 将 old-policy rollout 用于估计 current-vs-reference KL。论文具体声称该 estimator 的 gradient unbiased；作者对 K3 高噪声/无界权重及按领域调节 KL 强度的解释应保留为论文主张，而不是独立验证。
- Eq. (8) 的 mask 乘在 clipped policy 项上，KL 项不受 mask 直接屏蔽。Eq. (9) 仅在 advantage 为负、且 response 长度平均的 log(pi_old/pi_theta) 超过 delta 时置零；旧策略概率来自 inference framework 返回值。此门控按 sequence divergence 判断，不是逐 token divergence threshold。
- Keep Routing 复用 rollout 时 inference framework 的 MoE expert routes；Keep Sampling Mask 把 top-p/top-k sampling 的截断支持集同时用于 current policy。报告把二者作为稳定化策略，但没有因此公开完整 RL recipe。

面试重点：区分 policy ratio 与 KL 内部 ratio、group centering 与标准化、token-mean 与 group-mean、policy mask 与 KL penalty，以及负优势条件和序列级 divergence 门槛。待补项仍包括完整 reward/rollout/update recipe、训练代码、production acceptance、硬件 profiling 和独立复现。
