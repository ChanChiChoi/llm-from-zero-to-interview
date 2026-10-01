# DeepSeek V4：官方 API 公告核验摘记

核验日期：2026-09-09。来源：DeepSeek 官方 API 文档与公告。

## 已确认版本

- [V4 Preview](https://api-docs.deepseek.com/news/news260424)：官方新闻索引列为 2026-04-24，详情页本轮尚未读取。
- [V4 Pro GA](https://api-docs.deepseek.com/news/news260813)：2026-08-13 公告；公告称 V4 Pro 上线，支持 low/high/max reasoning effort，原生 OpenAI Responses API，并针对 Codex 优化。API 模型名保持不变，具体快照由 API 文档解释。
- [V4 Flash Vision Experimental](https://api-docs.deepseek.com/news/news260821)：2026-08-21 公告；接受混合文本和图像输入，图像 token 计费上限 384 tokens/张，可经 Chat Completions、Messages、Responses 使用，也支持 URL、base64 和 Files API。它是实验模型，不能和 GA V4 Pro 的稳定性等同。
- [API Quick Start](https://api-docs.deepseek.com/quick_start)：2026-09-15 当前页面以 `deepseek-flash` 为主模型名，并说明旧的 `deepseek-v4-flash` 与 `deepseek-v4-flash-vision-exp` 已 retired，兼容请求由 DeepSeek-V4.1-Flash 服务。此前 2026-09-09 快照中列出的旧别名仅作为历史接口状态保留，不能当作当前后端身份。

## 能力与系统层边界

V4 Pro 公告只确认 Agent 升级、可调 reasoning effort、Responses API 和 Codex 优化，没有披露参数量、架构、训练数据或完整技术报告。不能从榜单名称 “V4 Pro” 推断 MoE、上下文长度或训练算法。

V4 Flash Vision Exp 的“接近 Opus-4.8”是发布方对多模态 Agent 基准的表述；公告没有给出完整任务、harness、硬件和原始分数，暂不把它写成可比排名。模型的视觉输入、图像 token 计费、Files API 和 Agent harness 是独立系统能力，应分别讲解。

DeepSeek API 同时提供 OpenAI/Anthropic 兼容接口、Responses API 与 reasoning effort 参数。接口兼容不是模型内部实现兼容；迁移章节需展示参数映射和不兼容错误。

## 扩写方向

1. **版本别名与快照**：别名保持不变但后端更新，如何做可复现实验、回滚和模型卡记录。
2. **推理预算**：low/high/max 对延迟、token、成功率的影响，不能跨模型直接比较档位名。
3. **多模态 Agent**：输入图片、Files API、工具调用和视觉 benchmark 的完整链路。
4. **Responses API 与 Codex**：协议层、harness、模型和工具宿主之间的边界。
5. **峰谷定价**：公告提到 V4 阵容采用峰谷价格，正式写作需另查价格页并记录时间语境。

## 待核验

V4 Preview 正文、模型卡、开源权重、技术报告、上下文窗口和参数规模尚未找到一手证据；排行榜中的 `V4 Pro 0813` 不能单独承担这些结论。

## V4 Preview 技术报告与模型卡补充

[官方公告](https://api-docs.deepseek.com/news/news260424) 链接到 [DeepSeek V4 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) 和技术报告 `arXiv:2606.19348`。模型卡明确披露：V4-Pro 总参数 1.6T、激活 49B；V4-Flash 总参数 284B、激活 13B；均为 1M context。模型卡称采用 MoE、FP4+FP8 mixed precision（专家参数 FP4，其余多数参数 FP8）。

架构披露包括 Hybrid Attention：Compressed Sparse Attention (CSA) 与 Heavily Compressed Attention (HCA)；模型卡称在百万 token 场景下，V4-Pro 单 token 推理 FLOPs 约为 V3.2 的 27%，KV cache 约为 10%。还披露 Manifold-Constrained Hyper-Connections (mHC) 和 Muon optimizer。训练部分称预训练超过 32T token，后训练采用领域专家独立培养（SFT 与 GRPO），再用 on-policy distillation 统一合并。

这些数字和方法来自发布方模型卡/技术报告入口，仍需按报告正文、实验协议和代码进一步核对；“27%/10%”不是所有上下文、硬件和实现的普遍比例。V4-Pro-Max 与 V4-Flash-Max 是 reasoning effort 配置，不应视为新的基础权重。

## 教学扩写入口

CSA/HCA 单独讲 token-wise compression、稀疏检索与压缩误差；mHC 连接第二十一册和残差稳定性；FP4/FP8 连接第五、六册低精度训练/推理；MoE 激活参数连接负载均衡和通信；SFT+GRPO+on-policy distillation 连接后训练与专家能力合并；1M context 连接评测和 KV cache 成本。每个主题都需要独立例子与公式，不用公告摘要替代正文。

## 技术报告源码核验补充

已从 `arXiv:2606.19348` 下载 LaTeX 源码并读取架构、训练和系统章节。

### CSA 与 HCA

论文把 Compressed Sparse Attention（CSA）与 Heavily Compressed Attention（HCA）交错使用。CSA 先沿序列维以每 `m` 个 token 一个条目压缩 KV，再用 DeepSeek Sparse Attention 的 indexer 对压缩条目做 top-k 选择，核心注意力只访问选中的压缩 KV。HCA 使用更大的 `m'` 做更激进压缩，但对压缩后的条目保持 dense attention；两者都保留滑动窗口分支以处理局部依赖，并明确限制因果可见范围。

论文报告 1M 场景下相对 V3.2 的单 token FLOPs 和 KV cache 比例，也另给出相对 BF16 GQA8 基线的极低 KV 比例。不同基线、精度和指标不能混成一个“压缩率”；正式章节必须把 token 压缩倍率、稀疏 top-k、KV 字节数和 FLOPs 分开计算。

### mHC

mHC 将残差映射矩阵 `B_l` 约束在双随机矩阵集合（Birkhoff polytope）中，满足行和列均为 1 且元素非负。源码指出该集合对矩阵乘法封闭，用于稳定深层信号传播；投影通过交替行归一化与列归一化近似完成。它与普通残差、Attention Residuals 都解决信号传播问题，但约束对象和机制不同，不能混为同一技术。

### 训练与后训练

模型卡提到 32T+ 预训练，论文源码确认后训练两阶段：先分别培养领域专家（SFT 后接使用领域奖励模型的 GRPO），再用 on-policy distillation 训练统一学生模型，通过 reverse-KL 方向整合教师能力。这里的“领域专家”是训练流程中的教师模型/能力来源，不等于推理时运行一个 MoE expert 路由。

### 工程系统

源码披露了 Muon 与 AdamW 的模块分工、tensor-level checkpointing、混合 ZeRO、两阶段 contextual parallelism、异构 KV cache 与磁盘共享前缀复用，以及 FP4 量化感知训练用于 MoE 专家权重和 indexer QK 路径。这些适合映射到训练、推理和 AI Infra 章节；每项都需单独解释成本和硬件假设。

## 2026-09-15 当前 API 路由复核

当前 [Quick Start](https://api-docs.deepseek.com/quick_start) 与 [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing) 将 `deepseek-flash` 的模型版本标为 `DeepSeek-V4.1-Flash`，支持 Vision、1M context 和最多 384K 输出；旧 `deepseek-v4-flash`、`deepseek-v4-flash-vision-exp` 请求被说明为由 V4.1-Flash 服务。该行为是 2026-09-15 的服务端文档状态，不能回写成 2026-08-21 实验模型本身的架构事实。

当前 [Vision guide](https://api-docs.deepseek.com/guides/vision) 支持 base64、外部 URL 和 Files API `file_id`，并按图片尺寸自动 resize；文档给出的当前上限约为每图 1024 image tokens。发布公告对 `deepseek-v4-flash-vision-exp` 的历史表述是每图最多 384 tokens、按 V4-Flash 价格计费。两个数字属于不同时间/路由/文档语境，应并列记录并在实时接入时以响应的 model、usage 和文档版本为准。

本次复核的临时快照仅用于证据识别：Vision announcement 25,247 bytes，SHA-256 `f381bb106aa6d046a7a32015715e4e283320ad1fcf6b608ad439599600cf5784`；Quick Start 46,116 bytes，SHA-256 `6e2eb037db92ebef6a8f6408d87c12318c973388d6e27321606bb0e67dd67a6c`；Vision guide 78,192 bytes，SHA-256 `d6797f08bdb139e486e9cdd846003c14d3598254a76adbc932c3217be7a2b0df`；Files API guide 61,828 bytes，SHA-256 `4a7b7a36f57f8a3f946af7307071499c02c378d28e19096d36bba5d7b9b6f8a6`；Responses guide 56,250 bytes，SHA-256 `1fbc3261d73a2954466b75c17e233c5d7b94248ba3c5888da84eb63810e34b44`；Pricing page 23,359 bytes，SHA-256 `755aa9b488d1185cba016ca4de3b3b6b8f593f5e13e5f9f961305289a5c8d242`。

## 2026-09-20：DeepSeek V4 Pro 0813 当前活动锚点

本轮把 `DeepSeek V4 Pro 0813 (Reasoning, Max Effort)` 作为新的活动锚点。模型发现仍来自两个允许的排行榜：Artificial Analysis 精确页为 [DeepSeek V4 Pro](https://artificialanalysis.ai/models/deepseek-v4-pro)，页面标题为 `DeepSeek V4 Pro 0813 (Reasoning, Max Effort)`，release date 为 `2026-08-13`，Intelligence Index 为 `35.9967791278402`，目录 context 为 `1,000,000`，total/active 参数字段为 `1.6T/49B`；页面未标记 deprecated、标记开放权重和 MIT。以上指数、目录字段和开放性标签首先属于 AA 快照，只有与官方模型卡/配置一致的字段才进入模型事实。

AA 详情快照为 3,934,926 bytes，SHA-256 `b14ce625eb337401f4faa63eed37f1b08fe109a693dcdf2a34e21924b6cddff1`。DataCurve 当前精确行是：

```text
model: deepseek-v4-pro
harness: mini-swe-agent
reasoning_effort: max
config: mini_swe_agent_deepseek_v4_pro_max
Pass@1: 62.831858%
Pass@4: 88.495575%
n_runs: 4
mean_cost_usd: 1.6660232187
mean_output_tokens: 105998.9
mean_agent_steps: 154.71
```

这组数字是 `model + max effort + mini-swe-agent + 工具 + 任务集 + 环境 + verifier` 的组合结果；它可以说明该配置在该 harness 下的观察结果，不能写成 DeepSeek V4 Pro 裸模型能力，也不能迁移给 V4 Flash、V4.1-Flash 或其他 DeepSeek 版本。

### 官方 API 与运行时证据

- [V4 Pro GA 公告](https://api-docs.deepseek.com/news/news260813)确认 2026-08-13 上线、`low/high/max` reasoning effort、原生 Responses API 和 Codex 优化。V4 Pro 的 API 模型名保持为 `deepseek-v4-pro`；effort 是请求级运行配置，不是三个基础权重。
- [DeepSeek Quick Start](https://api-docs.deepseek.com/quick_start) 的当前页面列出 `deepseek-v4-pro`；[Responses API 文档](https://api-docs.deepseek.com/guides/responses_api)明确 DeepSeek Responses 是 stateless，不支持 `previous_response_id`、`conversation`、`background` 或 `store`，支持 function tools、`apply_patch`，并行工具调用始终开启；不支持的参数可能被静默忽略。
- [Thinking 文档](https://api-docs.deepseek.com/guides/thinking)、[Tool Calls 文档](https://api-docs.deepseek.com/guides/tool_calls)和[价格页](https://api-docs.deepseek.com/quick_start/pricing)用于解释 thinking/tool loop、参数边界和成本语境。协议字段只能证明 API 行为，不能反推模型内部的推理算法。

本轮临时快照用于证据识别：GA 公告 23,248 bytes，SHA-256 `a5d1169a61c8c33e1810873818257ace891ea09e27e9457038785adec7f1a362`；官方模型卡 README 13,149 bytes，SHA-256 `c4d714818a4d3333542edc7d38ea065825a0cf7aa8fea3605bbd1d1c18e4a610`；配置 JSON SHA-256 `5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e`；技术报告摘要页 SHA-256 `fe89d8f32de038d4b0daeee279ab90d3e087bc5a1c7889aac38250dc9d9482a4`；Responses、Thinking、Pricing、Tool Calls 页面分别为 `1719ac1b05e29579acd0cbc5ba0bcdb629cf7722eb551b5cd6d6d62c271e3ca2`、`8f7d45a2a97d7a18d480130d24f26a4a4f6f4c8de2b4cc39be86438e9d6da5b3`、`2fecee48bf6ad791bce38d1d5504d8ad5c8b0fd4da93e6dc198ae88ff1a4506a`、`5ee72ac00e5594bffac058cfef8872beb121b97e122f914c114a618f6a2b4027`。

### 模型卡、配置与报告的技术边界

[官方 V4-Pro 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、[配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)和 [DeepSeek V4 技术报告](https://arxiv.org/abs/2606.19348)共同支持以下可用于面试的主线：V4-Pro 为 1.6T total/49B active、1M context 的 MoE；CSA/HCA 混合注意力压缩远程 KV，再分别做稀疏或 dense 读取；mHC 约束残差流；Muon 与 AdamW 分工；32T+ 预训练；SFT 与 GRPO 培养领域教师，再用 on-policy distillation 合并能力；FP4/FP8 与异构 KV cache 面向百万上下文 serving。

配置快照还公开了 61 层、384 routed experts、每 token 6 个 routed experts、1 个 shared expert、hidden size 7168、1M position、`q_lora_rank=1536`、`o_lora_rank=1024`、`index_topk=1024`、YaRN factor 16 和 FP8 quantization 等实现字段。配置字段描述的是该公开 artifact 的实现接口，不等于完整训练 recipe、目标硬件最优 kernel 或所有服务端快照的内部结构。

### V4 Pro 的闭环状态

当前状态为**内容专题闭环（双榜锚点）**：排行榜发现、AA/DataCurve 精确配置、DeepSeek 官方公告/API 文档、模型卡、配置、技术报告入口、研究笔记和既有 CSA/HCA、mHC、MoE、低精度与后训练章节均已具备。由于现有书系已经覆盖核心架构，不新增重复的 Transformer 正式章节；第 77 章只补充 V4 Pro 的锚点与评测分层。

仍待核验：V4 Pro 专属完整报告正文中的所有实验协议、生产 kernel 与目标硬件 profiling、线上 tool acceptance、不同 API 快照的行为差异、完整训练/后训练超参，以及独立于发布方的 benchmark 复现。特别是 `low/high/max` 的质量—延迟—成本曲线必须绑定固定模型快照、任务、harness 和 verifier，不能跨模型按档位名比较。

## 2026-09-24：DeepSeek V4 Flash 的 SGLang v0.5.20 serving 补证

本节只扩展此前已由 Artificial Analysis/DataCurve 收录的 DeepSeek V4 Flash/Pro 锚点；SGLang 版本与 PR 是 runtime 证据，不是模型发现入口。经 7890 读取 GitHub 官方 `releases/latest` API，stable tag 仍为 `v0.5.20`，发布日期 `2026-09-18T22:41:33Z`；release 响应 `87,872` bytes / SHA-256 `c75daa307a5ead07993a00b9d8367579a9146dfc63b12ccfe31ae29cd4154bdc`。

### SWA 分支点缓存：Full KV 命中不代表混合状态完整

DeepSeek V4 Flash 同时有完整/压缩注意力 KV 与 sliding-window attention（SWA）状态。chunked prefill 可释放已滑出窗口的 SWA slots 以降低内存；但在共享前缀处分叉的 sibling request，可能仍命中完整 KV，却找不到对应分支点的 SWA 状态，只能重新计算前缀。SGLang PR [#34565](https://github.com/sgl-project/sglang/pull/34565) 在统一 radix tree 中保留可复用的 SWA branch-point state：插入分支前处理 out-of-window slots，分支进入树后再清理不再需要的状态；关闭对应优化开关时不改变旧路径。它说明混合注意力 serving 不能只用一个 `prefix_hit` 或 KV 长度代表所有状态。

PR 的内部 shared-prefix workload 使用 `DeepSeek-V4-Flash-0731`、TP=2、FlashInfer MXFP4、DSpark、共享 system prompt 24,576 tokens、question 8,192 tokens、output 128 tokens、64 个请求分成 8 组并发分支；server 上限、chunk size 和缓存开关也固定在 PR 中。启用 out-of-window free 时，PR 对照数据为：token hit rate `43.81% → 60.75%`，cached tokens `939,264 → 1,302,528`，mean TTFT `1,569.93 → 1,069.58 ms`，p95 TTFT `3,427.47 → 2,372.52 ms`，input throughput `66,310.37 → 70,509.72 tokens/s`。这是 SGLang PR 的指定合成 shared-prefix serving 结果，不是独立复现或对所有硬件/流量的保证；该 PR API 响应 `58,568` bytes / SHA-256 `cf0f40241e79c9b3cf0a8798a438784118d65ac1c0d2ba0fe8af0b364e5156a3`。

### CSA/HCA kernel 与硬件分支

| 来源/目标 | runtime 变化 | 发布方测量与边界 |
|---|---|---|
| [PR #30805](https://github.com/sgl-project/sglang/pull/30805)，B200 / SM100、SM103 | 为 DeepSeek V4 的 CSA/HCA attention 接入 TRT-LLM kernel，与 FlashMLA backend 对照 | PR 的 unit-kernel 结果约为 prefill `1.2x`、decode `1.45x`；测试注明 B200、FP8、TP=1 和私有 benchmark repo，不能写成端到端 V4 Flash 或 DataCurve 提升。API 响应 `55,177` bytes / SHA-256 `8a9bdc0788755b5082d0d1f5abbb340811e9a6fc5ecaf06b5dfc5402a4b13d7f` |
| [PR #29927](https://github.com/sgl-project/sglang/pull/29927)，4× RTX PRO 6000 / SM120 | sparse-MLA indexer 改走 DeepGEMM paged-MQA logits；prefill 使用 FlashInfer paged sparse attention；启用 DeepGEMM FP4 MoE，并避免每步重排整个 SWA KV pool | PR 报告单并发 TPOT 可到 `3.4x`，但基线是当时唯一能启动的慢速 torch indexer fallback；PR 说明另一个 HC prenorm kernel 贡献了 8K/BS1 TPOT 增益的约 `3.2%`。同 PR 的 MTP accepted length 接近，故报告把提升归因于 decode path，而非更高接受长度。不能将其简化为模型本身加速。API 响应 `47,628` bytes / SHA-256 `3e968d8d1f0eb5940e8c7be7fe8556b666264596f14cd99286457b4655065819` |

面试时应先问比较的是 kernel、token-cache 命中还是端到端 Agent task；然后锁定模型 snapshot、硬件、baseline backend、并行度、batch/context、共享前缀、speculation 和 verifier。SGLang `v0.5.20` 的 V4 serving 证据是系统/内核实现，不是 CSA/HCA 的新论文算法，也不证明完整权重、目标硬件、本地 acceptance 或生产 SLO。V4.1 的 `FlashMLA` dependency bump 另见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)，不能与通用 V4/Flash 实验混作同一 checkpoint 结果。

## 2026-09-20：官方 Hugging Face revision 与 inference implementation 补证

本节把“模型卡/论文宣称的技术”与“公开 artifact 中确实存在的实现路径”分开记录。模型发现仍只来自 Artificial Analysis 与 DataCurve DeepSWE；Hugging Face 仅作为已经发现的 V4 Pro 锚点的官方实现来源。

### 固定 artifact 与权重边界

- [DeepSeek-V4-Pro HF API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro) 当前 `sha` 为 `b5968e9190ef611bbf34a7229255be88a0e937c1`，`lastModified` 为 `2026-06-22T12:12:50Z`；metadata 响应 SHA-256 为 `08c362390ff25f26f1744b695ff7ef32cc0e153f6a81545b17776b2cec24febf`。
- 该 revision 列出 64 个 safetensors 分片；safetensors metadata 的总 storage 为 `1,598,839,674,782` bytes。完整权重没有下载，也没有在本地加载 V4 Pro；metadata 不能写成“权重已获取/推理已运行”。
- 固定文件哈希如下，便于把配置、协议和 inference 代码与以后变化的 `main` 分开：

| revision 文件 | bytes | SHA-256 |
|---|---:|---|
| `README.md` | 13,149 | `c4d714818a4d3333542edc7d38ea065825a0cf7aa8fea3605bbd1d1c18e4a610` |
| `config.json` | 1,828 | `5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e` |
| `encoding/README.md` | 8,118 | `605363e9e43ee91beba88ea96c7806ce6ecdb2924e481459c9d16e1526470c10` |
| `encoding/encoding_dsv4.py` | 27,908 | `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0` |
| `encoding/test_encoding_dsv4.py` | 3,741 | `c2bc54c4c934f5c64096bd9c555efa7d1ddf179c1eff58f01ceb2dcd60adcf28` |
| `inference/README.md` | 951 | `68dba94f8676578cddff2b0e8861586ef89d1857c6ad29e40bf5f17610b03bdf` |
| `inference/config.json` | 1,070 | `a6aded1806a2dbacbbab89bae2380d0422a6d0dcc55c946b421c7f5e06ef6094` |
| `inference/kernel.py` | 22,198 | `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2` |
| `inference/model.py` | 38,632 | `ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71` |
| `generation_config.json` | 170 | `5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0` |

`config.json` 的实现字段包括 `DeepseekV4ForCausalLM`、61 层、hidden size 7168、128 attention heads/1 KV head、384 routed experts、每 token 6 个 routed experts、1 个 shared expert、`q_lora_rank=1536`、`o_lora_rank=1024`、`index_topk=1024`、1,048,576 positions、YaRN factor 16 和 FP8 `e4m3`/scale 配置。它是固定公开 artifact 的配置账本，不等于服务端每个快照都使用同一 kernel，也不等于完整训练账本。

### inference config 与模型实现

`inference/config.json` 进一步公开了参考 inference 路径的字段：`n_hash_layers=3`、`score_func=sqrtsoftplus`、`route_scale=2.5`、`swiglu_limit=10`、`window_size=128`、`index_n_heads=64`、`index_head_dim=128`、`hc_mult=4`、`hc_sinkhorn_iters=20`、`expert_dtype=fp4`，以及 128/4 交错的压缩倍率布局。README 给出 `EXPERTS=384`、`MP=8` 的转换示例，并说明可在 FP4 与 FP8 expert dtype 之间切换；`MP=8` 是该官方转换示例的并行配置，不是所有硬件的生产部署结论。

从 `inference/model.py` 可读出以下可追踪路径：

1. `Compressor` 以 gated KV pooling 生成压缩表示；ratio=4 的路径保留 overlap state，以便块边界和因果可见性连续。
2. `Indexer` 对压缩 KV 做 learned scoring、causal mask 与 top-k 选择；indexer 路径有 FP4 模拟量化，不能只按论文中的“稀疏注意力”四字推断实现细节。
3. `Attention` 组合 MLA 的低秩 Q/O 投影、128-token sliding window 和 compressed-KV sparse attention；远程压缩分支与局部窗口分支是两类状态。
4. `Gate` 前 3 层使用 token-id hash routing，后续层使用 `sqrtsoftplus` score routing；bias 只参与 expert selection，不改变最终 routing weights。
5. `MoE` 使用 top-6 routed experts 加 1 个 shared expert，专家按 tensor parallel 分片；`MTPBlock` 还公开了 multi-token prediction block。
6. `Block` 使用 Hyper-Connections，`hc_mult=4`，通过 20 轮 Sinkhorn 近似双随机混合；这与第 78 章讲的 mHC 几何约束相互对应，但 inference 代码字段不能替代训练报告的完整推导。

### kernel 与协议实现

`inference/kernel.py` 是 TileLang 参考实现，包含按 `[128,128]` block 的 FP8 activation quantization、FP4 quantization、FP8/FP4 GEMM、稀疏 attention 的 online softmax，以及 HC Sinkhorn kernel。代码还显示 FP4 权重在按 K 维打包后参与 FP8/FP4 GEMM；这证明公开 artifact 有 kernel 路径，不证明在本机或线上硬件达到论文/README 的吞吐。

独立的 `encoding/encoding_dsv4.py` 是 prompt encoder/parser，不应写成模型内部思维算法。它支持 `system/user/assistant/tool/latest_reminder/developer` 角色，其中 `developer` 只给内部 search-agent 使用；tool role 会合并为 user 的 `<tool_result>` block。工具调用使用 DSML：`<｜DSML｜tool_calls>`、`<｜DSML｜invoke>` 和 `<｜DSML｜parameter>`，字符串与 JSON 参数由 `string="true/false"` 区分。`thinking_mode="thinking"` 使用 `<think>...</think>`；`drop_thinking=True` 默认删除旧 reasoning，但有 tools 时会自动保留 reasoning。

parser 对 malformed output 采取严格失败路径，不负责恢复；它检查特殊 token、EOS、参数和 tool-call 结构。`reasoning_effort` encoder 只接受 `max`、`high` 或 `None`，这是 prompt protocol 字段，不等于公开了内部 reasoning 算法。已做 Python AST 检查和无 CUDA 的手工 encode/parse round trip；没有运行需要 CUDA/TileLang 或完整权重的 inference。

### 本轮证据结论

本轮把 V4 Pro 从“模型卡/报告的内容专题闭环”推进到“固定 revision + encoding + reference inference implementation 的实现证据补强”。仍不能宣称完整权重已下载、本地推理成功、生产 kernel 已验收或 `MP=8` 是普适部署方案。下一步若继续补证，应固定 kernel/convert commit、目标 GPU、量化误差、compression/index recall、端到端 prefill/decode profiling、线上 tool acceptance 和 API snapshot 行为；所有结果都要绑定模型 revision、effort、harness、工具、环境和 verifier。
