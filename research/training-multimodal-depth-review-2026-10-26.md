# 第五册第十二章：多模态训练资料驱动精修记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/12-多模态训练.md`](../book-05-llm-training/chapters/12-多模态训练.md)  
依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md) 与 [`progress_v2.md`](../progress_v2.md)

## 研究目标与问题图谱

本章覆盖图文对齐、视觉语言模型、多模态指令微调、图像问答、OCR/文档理解、ASR/语音问答、视频理解与生成、diffusion、数据组织、跨模态对齐、算力、评测和安全。重审时将“训练目标假设—数据怎样生成监督—信息怎样进入模型—训练成本—能力怎样验证”作为主线，不以再列一组概念结束。

本轮把以下问题作为重点：

1. CLIP 与 SigLIP 的目标函数分别如何耦合 batch 候选；语义相关的伪负样本为什么不会被更换 loss 自动消除；SigLIP 2 的整体收益如何受组合 recipe 约束。
2. 冻结单模态模型、视觉指令微调和大规模多模态预训练分别受什么资源/数据条件限制，代表性方案能提供什么证据。
3. 文档任务中 OCR pipeline、文本/布局/图像联合预训练、OCR-free 结构生成和 screenshot parsing 的输入与监督假设为何不同。
4. CTC、RNN-T 与 autoregressive encoder-decoder 怎样处理时间对齐和输出历史；流式延迟还受哪些 encoder/endpoint 条件限制。
5. 视频变长训练怎样在批次利用率、时序信息、token 数和 attention 成本间取舍；Frame Pack 的训练效率不等于时序能力证明。
6. Diffusion 的 timestep 重加权怎样随 `x0`、`epsilon`、`v` 参数化变化；数据 mixture、显式 loss 权重和样本筛选分别改变哪一层。

## 代表性研究与证据边界

| 来源 | 研究问题与具体设计 | 本章提取的证据 | 限制与迁移边界 |
| --- | --- | --- | --- |
| [CLIP](https://arxiv.org/abs/2103.00020) | 双编码器图文预训练，以 batch 相似度矩阵和双向对比目标学习共享表示 | 为 CLIP 机制、batch negatives 与零样本迁移建立原始方法基线 | 目标主要是整图/整句匹配；不单独保证 OCR、区域定位或事实支持 |
| [SigLIP](https://arxiv.org/abs/2303.15343) 与 [Google Big Vision 实现](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py) | 将全局 softmax 分类改为图文 pair 的 sigmoid logistic loss；实现给出 scale、bias、正负标签和 log-sigmoid 路径 | 论文报告该目标在较小 batch 上有竞争力，超大 batch 收益递减；官方代码可核对标签和 loss 的具体张量关系 | Sigmoid 不消除错配标签：默认一正多负仍会惩罚未标注的同义 pair；论文 batch 结论不等于所有任务/集群都不需通信 |
| [SigLIP 2](https://arxiv.org/abs/2502.14786) | 对齐目标结合 captioning decoder、自蒸馏、masked prediction、在线筛选与可变分辨率数据 | 作者报告图文语义、检索、定位/dense features、多语言能力的组合提升 | 多个策略联合变化，不能仅凭整体结果归因某个辅助 loss；筛选器可能改变长尾覆盖 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 冻结 vision encoder 和 LLM，以两阶段 Q-Former 学会查询视觉表示并衔接生成语言模型 | 说明有限可训练参数下如何连接强单模态底座；其 VQAv2 zero-shot 与 Flamingo 对照只绑定论文任务和配置 | 不能把论文数字与其他模型、benchmark 的 headline 分数作横向排序；冻结 encoder 的域适应能力有限 |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | CLIP ViT-L/336px、MLP projector、视觉指令数据构成较简单的开放 VLM 训练基线 | 论文公开数据量、GPU/时间和多 benchmark 结果，可说明“简单 connector + 数据”在其设置下的有效性 | academic VQA 训练数据会影响评测可比性；作者报告的资源和指标依赖模型版本、硬件和训练 protocol |
| [MM1](https://arxiv.org/abs/2403.09611) | 系统消融 image encoder、分辨率、视觉 token、connector 和 caption/interleaved/text mixture | 1.2B base 消融配置中的 45% captioned image、45% interleaved image-text、10% text-only 是一个被测配置；作者认为 encoder/数据/分辨率/token 数在其 setup 影响显著 | 45/45/10 不是通用最优 mixture；论文各项消融受其数据规模、模型和评测集约束，connector 的相对影响也不是普遍定律 |
| [JEST](https://arxiv.org/abs/2406.17711) | 按联合 batch 的可学习性选择样本，而不是只独立打分单个样本 | 作者报告其多模态预训练设定下迭代数和计算量显著下降（最高约 13x 更少迭代、约 10x 更少计算） | 论文依赖参考模型/数据分布；大候选池打分本身有成本，参考分布偏差可能使稀有领域被低选；公开数字不能外推为所有训练作业收益 |
| [LayoutLMv3](https://arxiv.org/abs/2204.08387) | 将 OCR 文本/位置与图像 patch 结合，使用 MLM、MIM、word-patch alignment 等预训练目标 | 表明文本、布局与图像之间可加入明确的局部对齐监督；作者在指定任务/数据上报告相应收益 | 需要 OCR 和 box；预训练/任务分数不能替代目标业务字段关系和坐标审计 |
| [Donut](https://arxiv.org/abs/2111.15664) | 直接图像到结构化文本生成，使用合成文档数据与下游 JSON/任务目标绕过外置 OCR | 展示避免 OCR 中间错误传播的端到端路线；论文披露合成文档构造和读序列训练 | 不保留独立 OCR 中间结果会降低局部错误可见性；合成版式与真实扫描件/手写件的域差异要单测 |
| [Pix2Struct](https://arxiv.org/abs/2210.03347) | 先 warm up 识读，再进行 screenshot parsing；变分辨率 patch 将页面图像转为结构目标 | 作者消融的 base 配方为 100K 步（30K warmup + 70K parsing）；DocVQA ANLS 67.8 对比移除 parsing 的 12.2，说明页面结构目标的作用；论文还披露与 Donut 和文本基线的任务内对照 | 数字绑定原论文输入分辨率、模型、数据和任务；文本基线在论文中也很强，提醒评估需检验是否真正依赖视觉证据；高分辨率/长序列昂贵 |
| [CTC 原论文](https://www.cs.toronto.edu/~graves/icml_2006.pdf)、[RNN-T 原论文](https://arxiv.org/abs/1211.3711)、[Whisper](https://arxiv.org/abs/2212.04356) | CTC 汇总 blank/重复折叠后的单调帧路径；RNN-T 在声学时间×输出前缀格点求和；Whisper 是带任务 token 的 autoregressive encoder-decoder | 为不同对齐假设建立机制级对照；`book` 重复字符例说明 CTC 最短帧数约束，`T=160,U=20` 例说明 RNN-T 格点规模 | CTC/RNN-T 是否流式还取决于 encoder、chunking、endpoint 和搜索状态；不能从 loss 名称推断完整产品延迟 |
| [Min-SNR-γ](https://arxiv.org/abs/2303.09556) | 把扩散时间步视为共享参数任务，针对梯度冲突按 SNR 和预测参数化重加权 | 核对 `x0`、`epsilon`、`v` 三种权重公式；论文在 ImageNet 256、ViT-B 设定报告约 3.4x 收敛迭代差和 FID 2.06 | 权重乘在不同参数化的平方误差上，不能当作梯度范数横比；FID/迭代数绑定论文模型、噪声 schedule、采样器及评价协议 |
| [CogVideoX](https://arxiv.org/abs/2408.06072) | 3D causal VAE、时空注意力、progressive resolution、caption pipeline、mixed-duration/Frame Pack 与 timestep sampling | 给出变长视频、时空压缩及其 batch 处理的具体路线；论文的 rank-partitioned uniform timestep ablation 只作为该配置观察 | pack 需要正确隔离样本与时间坐标，不会消除总序列注意力成本，也不证明动作理解；所有数字留在论文训练/验证设置内 |
| [ACL 2025 MLLM 综述](https://aclanthology.org/2025.findings-acl.453/) | 按从特定任务 MLLM 到 omni-MLLM 梳理架构、训练阶段、数据和评估 | 用作问题地图与延伸阅读入口 | 综述分类不是某个算法的实验结果；机制与数字仍引用原论文 |

## 旧稿覆盖图

以下是旧章节知识入口在新章中的去向；覆盖的是问题和知识含义，不要求保留每句旧文字。

| 旧稿主题 | 本轮处理 | 修订后位置 |
| --- | --- | --- |
| 多模态主任务与数据模态分类 | 保留为全章问题地图，补训练目标差别 | §1、§2、§4—§10 |
| CLIP 图文对比、batch negative、温度、检索 | 保留并补损失、伪负例、多正例标签和数据并行边界 | §2、§2.5 |
| Vision Encoder、Connector、LLM 组合及分阶段训练 | 保留并补视觉 token 预算、代表性训练 recipe 和冻结/解冻取舍 | §3、§3.7、§14 |
| 多模态 instruction tuning、对话协议、assistant-only mask | 保留 shape、占位符、label mask 和错误诊断 | §4、§12、§15.3 |
| Caption 与 VQA | 保留任务差别、数据构造和为什么 caption loss 不代表 VQA 能力 | §5 |
| OCR 和文档训练 | 保留数据、监督和难点；新增 OCR pipeline/LayoutLMv3/Donut/Pix2Struct 机制对照及字段级评估 | §6、§6.5 |
| ASR、语音问答、识别目标和错误传播 | 保留并补 CTC/RNN-T/seq2seq 的路径、格点、定义域及流式前提 | §7、§7.4—§7.6 |
| 视频理解与视频生成、token 估算 | 保留并加入固定帧/分桶/Frame Pack 对照、成本数量例和评估边界 | §8、§8.2—§8.3 |
| Diffusion 目标、加噪、prediction parameterization 和生成条件 | 保留并补参数化相关的 Min-SNR 公式和逐步手算；既有 scheduler/CFG/latent 保留 | §9、§9.4—§9.6 |
| 多模态数据清洗、归一化、manifest、混合 | 保留并将 mixture、loss 权重与 batch 选择分开 | §10、§10.1—§10.4 |
| 跨模态对齐、显存/算力、demo | 保留作为工程闭环，扩展反事实、token 成本及数据审计案例 | §11、§12 |
| 安全、评估、训练策略、常见误判 | 保留并关联输入证据、关键字段、时间证据、幻觉和 prompt injection | §13—§15 |
| 原章节练习和进一步阅读 | 保留总结与书末资料，新增 6 题机制/方案对照练习；将专题练习同步到总练习库 | §16—§18、`EXERCISES.md` |

## 公式、教学例和代码审查

- SigLIP：核对 `q=s·cos+b` 与 `z∈{-1,+1}` logistic loss；两候选教学值 CLIP `p≈0.731, loss≈0.313`，SigLIP 错负例 `softplus(2)≈2.127`。此例仅说明错标签受新 loss 继续惩罚。
- CTC：字符级 `book` 的相邻 `o` 需要 blank 分隔，最短 5 帧；如果 tokenizer 把 `oo` 合成单 token，则长度条件依词表改变。RNN-T `T=160,U=20` 对应约 3,200 个时间/前缀内部位置，非 FLOPs 结论。
- 视频：patch 16 时 `8×224×224` 为 1,568 token，`16×448×448` 为 12,544 token；全注意力 score cells 的平方比为 64。只作形状/数量级例子，真实模型可能采用 tubelet、局部注意力或其它 token 压缩。
- Min-SNR：用 `SNR=ᾱ/(1-ᾱ)` 和原论文按参数化的权重式复算 `γ=5` 的两组样例；`SNR=0` 对 epsilon 权重按连续极限取 1。
- 数据 mixture：`p=(0.90,0.09,0.01)` 和显式权重 `(1,1,10)` 得 scalar coefficient `(0.90,0.09,0.10)`；明确这不决定最终梯度大小或方向。
- 教学 demo：标准库代码由 CLIP loss、VLM shape/mask、diffusion noising、ASR 帧数、视频 patch 成本和多模态数据审计构成，输出用来展示多个边界检查，不冒充完整多模态模型训练。

## 结构与验证状态

静态核对结果：18 个一级章节编号连续；108 个 fenced marker 成对；41 个 math fence 的花括号数一致；9 张 Markdown 表列数一致；语音小节编号 `7.1—7.6` 连续；章节和配套资料共检查 104 个本地 Markdown 链接目标，均存在。标准库教学 demo 输出与正文 expected output 逐字一致，零帧、零向量、空审计集、越界交叉熵 target、重复样本 ID 五类异常输入均被拒绝。`git diff --check` 通过。

没有运行真实模型、GPU、真实文档 OCR/ASR、多模态训练、benchmark 或 PDF/MathJax 渲染。官方 Big Vision 代码链接指向可变 `main` 分支，适合核对实现路径，不是固定版本快照；复现前应记录 commit。论文数字只复述原文设置，没有在目标系统上重测。
