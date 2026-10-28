# Adam、AdamW 与学习率调度的来源证据

检索与阅读日期：2026-10-28

## Adam

- 来源：[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v9)，Kingma 和 Ba，arXiv v9（2017-01-30 修订），发表于 ICLR 2015。
- 实际阅读位置：arXiv 题名、版本和摘要（第 8–19 行）。摘要描述 Adam 为随机目标的一阶优化方法，基于低阶矩的自适应估计，并讨论带噪或稀疏梯度问题及作者报告的实验比较。
- 来源支持的事实：Adam 的方法定位、基于一阶/二阶矩自适应估计的总体思路，以及原作者报告的适用动机和实验结果。
- 证据等级：原始论文摘要及版本元数据；更新公式按论文算法定义。
- 适用边界：摘要不能证明 Adam 在所有任务、模型规模或泛化指标上都优于 SGD；原论文的实验条件不等于现代大模型训练结果。
- 未解决项：本轮按本讲公式所需范围阅读摘要及算法定义，未对论文全部实验做复现或跨任务再分析。

## AdamW 与解耦权重衰减

- 来源：[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101v3)，Loshchilov 和 Hutter，arXiv v3（2019-01-04 修订），发表于 ICLR 2019。
- 实际阅读位置：arXiv 元数据与摘要（第 8–20 行）；并对照 PyTorch [AdamW v2.14 API 与算法](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)的算法段（页面第 371–407 行）、参数组和状态加载说明（第 412–460、635–647 行）。
- 来源支持的事实：论文摘要指出，按相应学习率缩放时，L2 正则与权重衰减在标准 SGD 中等价，但在 Adam 等自适应算法中不等价；论文提出把权重衰减与损失梯度更新解耦，并报告了其特定实验结果。PyTorch v2.14 文档以更新顺序说明衰减不会累积进动量或方差，并提供参数组、optimizer state_dict 和 scheduler/optimizer 加载顺序的公开接口语义。
- 证据等级：原始论文摘要加固定版本官方 API 文档。
- 适用边界：原论文的实验结果来自其设定的任务和数据，不能直接外推为 AdamW 在所有 LLM 任务上都更优；PyTorch 文档仅支持 v2.14 的公开行为。
- 未解决项：不同框架、融合 kernel、低精度状态和分片方案的具体数值顺序/内存实现需按所用版本另行核对。

## 学习率预热的架构条件与后续研究

- 来源：[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745v2)，Xiong 等，arXiv v2（2020-06-29），发表于 ICML 2020。
- 实际阅读位置：论文摘要（第 8–24 行）。
- 来源支持的事实：作者的 mean-field 分析指出，其所研究的 Post-LN Transformer 在初始化时靠近输出层的预期梯度较大，较大学习率可能导致训练不稳定；同一论文分析和实验中，Pre-LN 初始化梯度较平稳，并报告可移除预热。
- 证据等级：原始论文摘要支持作者报告的理论结果和实验范围。
- 适用边界：结论依论文所定义的 Post-LN/Pre-LN 架构、分析与实验设置；不代表所有 Transformer 变体均无需或必须使用预热。

- 来源：[Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training](https://arxiv.org/abs/2410.23922)，Kosson、Messmer 和 Jaggi，arXiv v1（2024-10-31），页面标记为 NeurIPS 2024 接收。
- 实际阅读位置：arXiv 版本信息与摘要（第 8–19 行）。
- 来源支持的事实：该研究聚焦小规模 GPT 的 AdamW/Lion 训练，分析早期更新大小与方向变化，并报告预热可限制早期更新；经特定方式改造更新归一化后，实验中预热需求可减少或消除。
- 证据等级：原始论文摘要。
- 适用边界：小规模实验和论文提出的改造优化器，不能证明生产规模模型均可移除预热。

## 调度选择与模型配置示例

- 来源：[DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954v1)，DeepSeek-AI，arXiv v1（2024-01-05）。
- 实际阅读位置：HTML 正文引言第 111 行；`2.3 Hyperparameters`（第 142–153 行），特别是第 144–147 行。
- 来源支持的事实：报告描述其预训练使用 AdamW；第 2.3 节给出 beta 与权重衰减配置，使用 multi-step 学习率调度，并在报告的实验中称其最终表现与 cosine 调度基本一致；报告选择该调度还考虑了持续训练时复用前阶段计算的便利。
- 证据等级：作者技术报告中的实际配方和作者报告的实验结论。
- 适用边界：仅代表 DeepSeek LLM 报告中的模型、训练规模和条件，不证明 multi-step 或 cosine 在别的配置中等价或更优。

## 参数分组代码示例

- 来源：[Transformers translation training example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py)，Hugging Face 官方仓库 main 分支；2026-10-28 浏览。
- 实际阅读位置：optimizer 参数分组代码（页面第 2744–2768 行）。
- 来源支持的事实：该脚本把名称匹配 `bias`、`LayerNorm.weight` 和 `layer_norm.weight` 的参数放到 `weight_decay=0` 组，其余参数应用用户设置的权重衰减。
- 证据等级：官方源代码中的特定训练示例。
- 适用边界：main 分支会变化；这个名称名单既不构成所有模型的推荐，也不能自动覆盖 RMSNorm 或不同命名的归一化层。
- 未解决项：本轮未用某一具体 LLM 的 named_parameters 执行分组检查；生产训练应按实际参数名确认分组结果。
