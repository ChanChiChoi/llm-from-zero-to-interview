# Qwen3.5-397B-A17B 官方资料摘记

核验日期：2026-09-18。本笔记只把 Artificial Analysis 和 DataCurve DeepSWE 作为候选发现入口；模型卡、配置、Qwen 官方仓库和 Qwen 官方博客用于核验已经发现的 Qwen3.5 锚点。模型卡和博客中的 benchmark、训练效率、RL 规模与能力数字均保留为 Qwen 官方自报，不等同于独立复现。

## 1. 榜单发现与证据边界

- Artificial Analysis 精确条目：[Qwen3.5 397B A17B](https://artificialanalysis.ai/models/qwen3-5-397b-a17b)。当前页面同时有 Reasoning/Non-reasoning 配置，它们按同一基础模型归并，不按 effort 或 mode 重复计数。
- 本轮详情快照：`/tmp/qwen35-397-aa.html`，3,700,616 bytes，SHA-256 `04f09d032033c2705bc9a40c649ebc226f81eeef457adf11886f644490181719`。快照字段包含 Reasoning、397B total、17B active、约 1M context 和第三方 Intelligence Index 约 19.1073；这些字段属于 Artificial Analysis，不替代官方规格。
- DataCurve DeepSWE 页面当前没有精确的 `mini_swe_agent_qwen3_5_397b_a17b_*` 行，因此不把其他 Qwen 模型的 Pass@1、成本、输出 token 或 Agent steps 迁移到 Qwen3.5-397B-A17B。
- DataCurve 页面快照：`/tmp/qwen35-ds.html`，268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。页面中没有精确模型行，负证据只说明当前榜单未提供该配置的结果。

## 2. 官方资料

- [Qwen3.5-397B-A17B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)：模型身份、公开结构字段、上下文、thinking、工具和 serving 示例。
- [Qwen3.5 配置文件](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json)：固定模型配置和视觉 encoder 字段；配置文件是实现契约，不自动证明完整训练 recipe。
- [Qwen 官方 Qwen3.8 仓库](https://github.com/QwenLM/Qwen3.8)：README 同时记录 Qwen3.5 的发布路线、官方引用和系列关系；仓库页面不是独立 Qwen3.5 技术报告。
- [Qwen3.5 官方博客](https://qwen.ai/blog?id=qwen3.5)：原生多模态 Agent、架构效率、RL 规模和全球语言覆盖的发布方说明。
- [Qwen3.5 GitHub README 原始快照](https://raw.githubusercontent.com/QwenLM/Qwen3.8/main/README.md)：包含 Qwen3.5 的官方 highlights、发布时间和引用信息。
- [Qwen3.5-Plus 托管关系](https://www.alibabacloud.com/help/en/model-studio/text-generation)：模型卡声明 Qwen3.5-Plus 是对应的 hosted version，拥有默认 1M context、内置工具和 adaptive tool use；它不是新的公开 checkpoint。

本轮保存的官方模型卡为 `/tmp/qwen35-modelcard.md`，配置为 `/tmp/qwen35-config.json`，GitHub 页面为 `/tmp/qwen35-github.html`，README 为 `/tmp/qwen35-readme.md`，博客为 `/tmp/qwen35-blog.html`。本轮没有下载几十 GB 权重；研究闭环不以本地持有权重为前提。

## 3. 模型与架构字段

模型卡明确给出 Qwen3.5-397B-A17B 是带 vision encoder 的 causal language model，397B 是 total parameters，17B 是 activated parameters。公开结构字段为：

| 字段 | 官方模型卡口径 | 面试解释边界 |
|---|---|---|
| 层数与 hidden | 60 layers，hidden size 4096 | 可用于结构和通信账本，不等于完整实现细节 |
| block layout | 15 个重复组，每组 3 个 `Gated DeltaNet -> MoE`，再接 1 个 `Gated Attention -> MoE` | 3:1 是 Qwen3.5-397B-A17B 的公开排布，不能自动迁移到其他 Qwen3.5 尺寸 |
| Gated DeltaNet | 64 个 value heads、16 个 query/key heads、head dimension 128 | 递归 state 负责低成本历史压缩，需与显式 attention 的 global retrieval 分开 |
| Gated Attention | 32 Q heads、2 KV heads、head dimension 256、RoPE dimension 64 | GQA-like KV 共享字段；不能从字段反推完整 kernel |
| MoE | 512 experts；每 token 10 routed + 1 shared；expert intermediate dimension 1024 | 17B active 不等于总权重显存、KV cache 或并发显存 |
| context | 原生 262,144 tokens，可用 YaRN 扩展到约 1,010,000 | context 上限不等于所有任务都拥有同等有效召回率 |
| multimodal | 原生 vision-language，配置中的 vision encoder 为 27 层、hidden 1152、16 heads，patch size 16、temporal patch size 2、输出 hidden 4096 | 这是公开配置字段，不等于完整视觉训练配方或视觉 benchmark 独立复现 |
| MTP | 模型卡声明 trained with multi-steps | 支持 MTP 训练/推测解码路线；不能由此推导固定 acceptance rate |

Qwen3.5 使用 Gated DeltaNet 与 sparse MoE 的混合架构。教学上可以把一个 Gated DeltaNet state 写成：

```math
\widetilde{S}_{t-1}=\alpha_t S_{t-1}
```

```math
e_t=v_t-\widetilde{S}_{t-1}^{\top}k_t
```

```math
S_t=\widetilde{S}_{t-1}+\beta_t k_t e_t^{\top}
```

```math
y_t=S_t^{\top}q_t
```

这里的 `alpha`、`beta` 和矩阵方向是教学抽象；模型卡确认的是 Gated DeltaNet 及 head layout，不是该项目内部所有实现细节。面试回答应说明：delta update 试图写入“当前 value 与已有 state 读出之间的误差”，而不是无条件累加所有 `k_t v_t^T`。

## 4. Qwen3.5 的新增技术主线

### 4.1 Native multimodal early fusion

官方博客和模型卡把 Qwen3.5 定位为 native multimodal foundation，并声明 early-fusion 训练覆盖 trillions of multimodal tokens。此处的关键面试点是统一 token 流：文本、图像和视频信息进入同一个生成/推理协议，减少“独立视觉模型 + 文本模型 + 外部胶水”的接口断裂。

但“early fusion”只证明发布方的训练方向，不能直接推出视觉 token 如何分配、vision encoder 与 language model 的具体交互、跨模态 loss 权重或生产端到端吞吐。模型卡的 benchmark 图表也必须标作官方自报。

### 4.2 Scaled RL 与 agent environments

Qwen 官方声明 RL 扩展到 million-agent environments，并使用 progressively complex task distributions、asynchronous RL framework 和大规模 agent scaffold。这提供了面试中的系统问题：rollout 产生速度、环境并发、轨迹存储、verifier、stale policy、失败重试和训练更新之间如何解耦。

可复用的系统账本至少应分开：

1. 模型推理 token 与 reasoning budget；
2. tool call、tool result 和环境状态转移；
3. verifier/reward 延迟与失败轨迹；
4. asynchronous learner 的 policy version、数据 freshness 和 replay 过滤。

官方页面没有公开完整 RL loss、rollout scheduler、奖励模型、环境协议或独立规模复现，因此不能把“million-agent environments”改写为已确认的训练 recipe。

### 4.3 Multi-step training 与 serving

模型卡明确声明 MTP trained with multi-steps，并给出 SGLang 的 `NEXTN` 与 vLLM 的 `qwen3_next_mtp` speculative decoding 示例。MTP 的工程闭环必须包含 draft、target verification、accepted tokens、rollback 和 committed KV；仅仅打开 speculative 参数不能证明速度提升。

官方模型卡还给出以下 serving 边界：

- 397B-A17B 示例建议 8-GPU tensor parallel；这是官方示例配置，不是所有硬件的最低要求。
- 可使用 SGLang、vLLM、KTransformers 和 Transformers serving；框架兼容不代表各版本 kernel、tool parser 和视觉路径完全一致。
- `--language-model-only` 可以跳过 vision encoder，以释放更多显存给 KV cache；这改变的是服务模式，不是新的模型变体。
- Qwen3.5 默认 thinking mode，使用 `<think>...</think>` 内容；soft switch、历史 thinking 保留和托管 API 字段要按对应接口文档处理。

## 5. 与 Qwen3.8 的关系

Qwen 官方 README 把 Qwen3.8 描述为 built on the architectural foundation of Qwen3.5。当前可以确认的教学关系是：Qwen3.5 是 Qwen3.8 后续架构的公开基础方向，Qwen3.8 的 GDN/Gated Attention、QSA、Gated Residual、N-gram 和 Muon 等新增技术在既有 Qwen3.5 混合架构之上继续演化。

不能反向把 Qwen3.8 的每一个细节迁移到 Qwen3.5-397B-A17B：Qwen3.5 模型卡公开的是 Gated DeltaNet、Gated Attention、MoE、vision encoder、MTP 和上下文字段；QSA、四分支 Gated Residual、N-gram table、Flash-Next 的 Muon 分工属于 Qwen3.8/Flash-Next 的公开材料，除非 Qwen3.5 专属来源明确说明，否则不写成 Qwen3.5 的技术。

| 维度 | Qwen3.5-397B-A17B | Qwen3.8 系列公开材料 |
|---|---|---|
| total/active | 397B / 17B | 27B dense、2.4T / 95B、Flash-Next 125B / 6B 等不同形态 |
| hybrid core | Gated DeltaNet + Gated Attention + sparse MoE | 延续 GDN/GA，并在 Flash-Next 引入 QSA、Gated Residual、N-gram 等路线 |
| multimodal | 原生 vision-language，vision encoder | 27B 与 Flash-Next 的视觉/模型卡边界分别核验 |
| MTP | 模型卡声明 multi-step training，给出 NEXTN/MTP serving 示例 | Flash-Next 等资料有更具体的 MTP/QSA 组合，但不可回填 |
| serving | 8-GPU TP 示例，支持 LM-only、tool parser、MTP | 不同 checkpoint/hosted revision 需要单独固定 revision 和后端 |

## 6. 待核验项

- 完整训练数据、optimizer、RL loss、奖励模型、rollout 和 asynchronous RL recipe；
- Gated DeltaNet/Gated Attention 的生产 kernel、state layout、跨卡通信和目标硬件 profiling；
- MTP acceptance length、target calls、端到端 TTFT/TPOT 和不同 batch 的吞吐；
- vision encoder 的完整预训练/对齐方式、图像视频 token 预算和独立多模态复现；
- Qwen3.5-Plus hosted version 与公开 checkpoint 在 revision、工具路由、默认 1M context 和 adaptive tool use 上的精确差异；
- DataCurve 精确 Qwen3.5-397B-A17B Agent 行是否未来出现；出现后仍需固定 harness、revision、工具、任务集和 verifier。

## 7. 闭环状态与书系落点

Qwen3.5-397B-A17B 当前为**内容专题闭环**：有两个排行榜中的 Artificial Analysis 候选、官方模型卡/配置/仓库/博客、研究笔记、第二十一册专题补充和配套面试训练同步。DataCurve 没有精确行，因此没有跨模型迁移 Agent 分数。

正式教学落点为：

- [第二十一册第 83 章](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md) 的 Qwen3.5 前置与对比小节；
- 第五册 MoE、后训练、RL 和多模态训练章节；
- 第六册/第二十四册的 MTP、混合 state/KV cache、vision encoder 和 serving 章节；
- 第十七册、第二十册的 tool contract、million-agent environment 和 verifier 账本。

## 8. 快照摘要

| 快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/qwen35-397-aa.html` | Artificial Analysis Qwen3.5 397B A17B 详情页 | 3,700,616 | `04f09d032033c2705bc9a40c649ebc226f81eeef457adf11886f644490181719` |
| `/tmp/qwen35-ds.html` | DataCurve DeepSWE 页面 | 268,571 | `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` |
| `/tmp/qwen35-modelcard.md` | Qwen 官方模型卡 | 本地快照 | 未在本轮记录哈希 |
| `/tmp/qwen35-config.json` | Qwen 官方 config.json | 本地快照 | 未在本轮记录哈希 |

