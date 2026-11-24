# 卷积序列模型在 LLM 中的角色与证据

研究日期：2026-11-23
研究问题：区分局部卷积、长卷积和混合卷积主干的机制；核对近年的 LLM 卷积实验与公开模型实例；判断现有结果对“卷积复兴”及替代 Transformer 的主张支持到哪里。
范围：因果短卷积/深度卷积、显式或隐式长卷积、卷积与 attention/状态模型的组合，以及完整序列计算和自回归生成的结构性成本。研究服务于第二十一册第四十一章；不作穷尽综述，不把 serving 系统或通用硬件调优当作本章职责。

## 检索与筛选

2026-11-23 使用网页搜索并直接阅读原始论文、会议论文库、方法作者/官方团队文章和教学材料。英文检索词包括 `Convolution for Large Language Models Qwen3`、`Simple Hardware-Efficient Long Convolutions for Sequence Modeling`、`StripedHyena-7B convolution attention`、`ShortConv tutorial`、`CS224S local context convolution`。对性能与结构事实优先回到论文或模型发布方；作者博客、课程讲义和讲解页用于理解机制及组织解释。检索结果不是系统综述，也未确认存在覆盖同一问题的独立复现实验。

## 研究发现

### 1. 局部深度卷积可以嵌入仍保留全局 attention 的 Transformer

《Convolution for Large Language Models》将 depthwise Conv1D 插入 Qwen3 block 的 17 个位置。论文的 P5 位置在 Q/K/V 投影之后、attention 之前；在其位置消融中，P5 得到最低平均 loss 和 WikiText-103 perplexity。后续模块消融选用残差 depthwise convolution、$k=3$、随机初始化，且不加额外 normalization 或 activation。作者解释该位置可先构造局部上下文化的 Q/K/V，再由全局 attention 交互；该机制解释与位置消融相符，但论文没有单独隔离“局部组合”这一中间机制。

模型从头训练 Qwen3-1.7B 和 Qwen3-4B，使用 FineWeb-100B 的 30B、60B、100B token 预算（不同模型覆盖的预算不同）；序列长度为 4096，global batch 为 256 条序列。七项下游基准的平均准确率变化为：Qwen3-1.7B/30B，52.71→54.70（+1.99）；Qwen3-4B/30B，57.13→58.70（+1.57）；Qwen3-4B/60B，58.78→60.69（+1.91）；Qwen3-4B/100B，58.65→62.41（+3.76）。七项为 ARC-Challenge、ARC-Easy、BoolQ、HellaSwag、PIQA、WinoGrande、WSC。平均分每个所测设置都上升，但若干单项下降；较大的 100B 增益仍需更多预算和重复实验，不能据此建立普遍 scaling 趋势。论文对重复 token 的表示差异只展示一个 WSC 案例，作者也指出 cosine similarity 不能独立证明语义机制。

参数比例存在来源内部不一致：摘要和图注称额外参数小于 0.01%；但 Table 1 的基线为 1720.57M、P5 为 1721.15M，差值 0.58M，按表中数值计算约为 0.034%；Table 2 的 Conv+Shortcut 为 1721.03M，差值 0.46M，约为 0.027%。论文没有解释该差异。正文可说明 depthwise 运算参数对通道宽度是线性量级，但不应无保留复述摘要中的精确比例。

### 2. 直接长卷积可在混合语言模型中承担序列混合层

Fu 等人的《Simple Hardware-Efficient Long Convolutions for Sequence Modeling》发表于 ICML 2023（PMLR 202）；研究直接学习长卷积核，并发现核平滑/稀疏化正则化对其序列建模质量有重要作用。方法作者 Hazy Research 的文章以 Long Range Arena 结果讲解：未加 Squash 的随机长核平均分 69.5，加入 Squash 后为 86.1，与该设置下 S4 的 86.1 相同。该数字是六项 LRA 特定配置，不能当成语言模型结果。

语言建模实验不是纯卷积 Transformer 替代：作者依 H3 语言模型设置保留两层 attention，并以长卷积替换模型中的 SSM 核。Pile 上 5B/10B/15B 训练 token 的 test PPL：Transformer 为 12.7/11.3/10.7，H3 为 11.8/10.7/10.2，H3+LongConvs（geometric initialization）为 11.9/10.7/10.3。该表支持长卷积可成为混合序列模型里的竞争性层；它不证明纯长卷积模型胜过 Transformer，也不能单独归因 attention、SSM 替换和其余模型因素。

### 3. StripedHyena-7B 是卷积与 attention 混合的模型实例

Together AI 团队于 2023-12-08 发布 StripedHyena-Hessian-7B base 和 StripedHyena-Nous-7B chat。作者描述其为 attention 与 Hyena operator 中 gated convolution 的混合。官方文章报告的 ZeroScrolls 零样本结果（SH 7B 对 Mistral 7B，F1）：GovReport 27.9/17.5、NarrativeQA 25.8/24.7、Qasper 28.8/30.3，三项平均 27.5/24.2。结果并非逐项领先；这是模型发布方报告，比较不能隔离卷积的因果贡献，也不等于独立复现。

同一作者文章把滤波表示按 FIR 与 IIR 描述：FIR 可用有限滑动窗口状态，IIR 可用递归更新的定长状态。该描述是 StripedHyena 的表示/推理选择示例，不能推广为任意卷积都能使用定长状态。文章所报的训练速度、cache footprint 等性能属于作者在指定模型、硬件、长度和 baseline 下的系统测量；本章无需将其泛化为卷积家族定律。

## 来源与证据边界

| 来源 | 类型、版本与阅读位置 | 可支持 | 不单独支持 |
| --- | --- | --- | --- |
| [Convolution for Large Language Models](https://arxiv.org/html/2607.18413)，Tian et al.，arXiv:2607.18413v1，2026-07-20 | 技术报告 v1；§3.1—3.4、Tables 1—4；§4 的单例 representation analysis；§5.1 实验设置与 Table 8 | Qwen3 两个规模上的 Conv1D 插入位置、选定局部模块、指定 token 预算与七项基准的报告值 | 其他模型家族的普遍收益；WSC 一个案例不证明通用变量绑定机制；摘要精确参数比例（与参数表有未解释差异） |
| [Simple Hardware-Efficient Long Convolutions for Sequence Modeling](https://proceedings.mlr.press/v202/fu23a.html)，Fu et al.，ICML 2023，PMLR 202 | 正式会议论文；§3.1、§4.1.3 及语言建模 Tables 7—8；实验结构由 arXiv HTML 版 §4.1.3 明述保留两层 attention | 长核平滑正则化、H3+LongConv 在指定数据/预算上的结果 | 纯卷积主干的普遍优越性、把结果只归因于长卷积或外推到其他硬件 |
| [Simple Long Convolutions for Sequence Modeling](https://hazyresearch.stanford.edu/blog/2023-02-15-long-convs)，Fu et al. 等，Hazy Research，2023-02-15 | 方法作者教学文章；Squash 解释、LRA 平均数和 Pile 表 | 作者对核平滑的机制解释和对应的特定实验数字；适合理解直觉 | 将 LRA 数值当成语言模型指标，或把模型级表现归因给单一卷积算子 |
| [Long Convolutions for GPT-like Models: Polynomials, Fast Fourier Transforms and Causality](https://hazyresearch.stanford.edu/blog/2023-12-11-conv-tutorial)，Ré 与 Fu，Hazy Research，2023-12-11 | 方法作者教程；卷积与多项式乘法、FFT、因果线性卷积及回绕说明 | 长卷积和 FFT 的教学解释、因果边界讲解 | 未在教程中报告的模型质量或通用硬件速度结论 |
| [StripedHyena-7B release article](https://www.together.ai/blog/stripedhyena-7b)，Together AI，2023-12-08 | 官方模型发布文章；发布段、架构混合说明、ZeroScrolls 表、FIR/IIR 段落 | 发布模型名、作者披露的混合架构、指定的作者评测和状态表示说明 | 独立验证、纯卷积因果效果、所有卷积结构的普遍 cache/速度性质 |
| [Stanford CS224S Lecture 11](https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec11.pdf)，2025 Spring | 课程 slides；“Local context: Convolution module”与总体架构图 | Conformer 语音模型中局部卷积与全局 attention 的角色分工，作为讲解组织参考 | LLM 模型证据、Qwen3 结构事实或普遍学习效果 |
| [Short Convolution (ShortConv)](https://www.sebastianraschka.com/llm-architecture-gallery/shortconv/)，Sebastian Raschka，访问于 2026-11-23 | 技术讲解页；causal depthwise equation、local receptive field、state/cost | 单通道因果深度卷积的直观说明及其在 Kimi Linear/Inkling 示例中的具体解释 | Qwen3 或 StripedHyena 的实际模块事实；不注明模型边界的家族概括 |

本研究未运行代码、模型、API、kernel 或 benchmark；实验结论均为 `source-backed`，不是本地复现。
