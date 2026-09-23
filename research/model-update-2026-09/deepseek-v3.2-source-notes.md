# DeepSeek V3.2 官方资料摘记

核验日期：2026-09-20。本笔记只升级已经出现在 Artificial Analysis 的 DeepSeek V3.2；本轮 DataCurve DeepSWE 快照没有精确的 V3.2 行，因此不记录或迁移相邻版本的 DeepSWE 分数。

## 榜单锚点与快照

| 来源 | 结果 | 证据边界 |
|---|---|---|
| [Artificial Analysis DeepSeek V3.2](https://artificialanalysis.ai/models/deepseek-v3-2) | 页面标题为 `DeepSeek V3.2 (Non-reasoning)`；页面显示 2025 年 12 月、128K context、Intelligence Index `16.043537719683`、约 37B active / 648B total 的第三方参数字段 | 这是第三方配置页；`Non-reasoning` 是运行配置，release/参数/指数不是 DeepSeek 官方发布或裸模型能力证明 |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 当前快照没有精确 `mini_swe_agent_deepseek_v3_2_*` 行 | 不把 DeepSeek V3.1、V4 或其他模型的 Pass@1、成本和 Agent steps 迁移给 V3.2 |
| Artificial Analysis 详情页快照 | HTTP 200；3,391,821 bytes；SHA-256 `488c2c63fdb6d1746525f317222642d12cd57c7daf0f0abeba31653a6924ab7a` | 页面版本和测量字段随榜单变化；只用于候选发现与第三方配置复现 |
| DataCurve 当前快照 | HTTP 200；268,571 bytes；SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` | 页面模型集合证据；无 V3.2 精确行 |

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

- 待核验：PDF 抽取失真的图表/公式逐页视觉复核、完整 DSA 层排布和 kernel 源码行为、indexer 训练召回曲线、KV/indexer 字节账本、完整 RL 超参/奖励权重/rollout 配方、合成数据污染审计、线上 tool-call acceptance rate、硬件 profiling 和独立 benchmark。
- 不新增 V3.2 专属独立架构册章：DSA 已有第二十一册第 19 章承接一般机制，本轮在该章新增 19.29--19.32，补齐最终模型/实验 demo、kernel、serving recipe 和 thinking-with-tools 的证据分层。
- 研究笔记和现有第 19 章专题段共同作为本轮交付；模型清单、来源索引、榜单解释、题库、练习、术语、项目、知识图谱、`plan_v2.md` 与 `progress_v2.md` 均同步记录模型和证据边界。

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
