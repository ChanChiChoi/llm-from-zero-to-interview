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
