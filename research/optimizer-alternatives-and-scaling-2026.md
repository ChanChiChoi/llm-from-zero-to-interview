# 优化器替代路线与比较证据

检索与阅读日期：2026-10-30

本记录保存优化器方法、学习率策略和论文比较中新增核验的事实。论文结果均为来源作者在限定模型、数据、预算、硬件和评测口径下的报告；本记录不表示本地复现，也不构成跨任务的统一排名。

## Adafactor：因子化二阶状态与配套更新规则

- 来源：[Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/html/1804.04235)，arXiv v1（2018-04-11），发表于 ICML 2018。
- 实际阅读位置：摘要（第 43–46 行）、行/列统计和算法（§3，第 85–145 行）、update clipping 与 gradient clipping 对照（§6，第 180–192 行）、递增二阶衰减（§7，第 193–290 行）、relative step size（§8；本轮未逐段复核其全部公式）。
- 来源支持：矩阵参数的二阶统计可用梯度平方的行和、列和近似重构；辅助状态由 `O(nm)` 降到 `O(n+m)`。原论文以 rank-1 非负分解推导该近似，并指出可按需加入一阶矩；论文所报告的低存储 WMT14 En–De 结果由其完整 recipe 共同产生。该 recipe 还包含更新裁剪、逐渐增大的二阶衰减、去除动量和参数尺度相关步长等选择。
- 特别边界：Adafactor 的 update clipping 截断经过预条件后的更新 RMS；它与在反向传播后裁剪梯度不是同一位置、也不具有相同上界。论文中的 WMT 结果不能单独归因于 factorization，也不能外推为现代 LLM 中必然节省端到端训练成本。
- 证据等级：原始论文算法、方法段和作者实验。

## Lion：符号动量及其学习率/权重衰减配比

- 来源：[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/html/2302.06675)，arXiv v4（2023-05-08）。
- 实际阅读位置：摘要（第 45–54 行）、Lion 伪代码（第 68–85 行）、批大小实验与超参说明（§5，第 392–414 行）。
- 来源支持：Lion 只保存动量状态，以符号给出坐标更新方向；该方法还采用标准解耦权重衰减。作者报告其更新通常需要比 AdamW 小的学习率，并因 `lr × weight_decay` 的有效衰减关系而在其设置中采用更大的 decay 系数。论文给出的 3–10 倍比例是其经验调参范围，并不是通用换算式；其 batch-size 优势结果来自特定 ViT/ImageNet 扫描，论文也指出某些任务收益很小或无统计显著性。
- 证据等级：原始论文算法和特定任务实验。
- 未解决项：若将 Lion 放入另一框架或 LLM 配方，需核对 momentum 更新次序、decoupled decay、schedule 与参数组；本轮没有执行实现或本地调参。

## Sophia：对角曲率估计与逐坐标裁剪

- 来源：[Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/html/2305.14342)，arXiv v4（2024-03-05）。
- 实际阅读位置：摘要（第 42–50 行）、方法与两类估计器（§2）、裁剪比例调节（第 229–230 行）、GPT 配置和每 `k` 步估计（第 448–452 行）、比较实验摘要。
- 来源支持：Sophia 用一阶动量除以轻量对角 Hessian 估计，并逐坐标裁剪；Hutchinson 路线与交叉熵下的 Gauss–Newton–Bartlett（GNB）路线具有不同估计假设和误差。算法 3 初始化 `h_{1-k}=0`，并在 `t mod k = 1` 时刷新曲率状态；伪代码还先做独立 weight decay，再应用预条件裁剪更新。论文建议在其调参过程中将未触发裁剪的坐标比例调至 10%–50%，并在其 GPT 配置中比较估计刷新频率，常用 `k=10`；`gamma=0.01`（Sophia-H）和 `0.05`（Sophia-G）是论文实验超参，不是默认推荐。作者在 125M–1.5B GPT 设置报告了约 2× 效率改善。
- 特别边界：估计器、刷新频率、学习率、对照调参和预算都会影响结果；作者摘要的 2× 结论不是独立复现或生产规模保证。Hessian/曲率估计的负值、非有限数和状态陈旧仍需实现显式处理。
- 证据等级：原始论文方法、配置和作者实验。

## GaLore：双侧子空间投影与全参数更新

- 来源：[GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/html/2403.03507)，arXiv v2（2024-06-02），发表于 ICML 2024。
- 实际阅读位置：摘要（第 71–79 行）、梯度低秩动机（§3.2，第 172–192 行）、正式更新定义（§3.3，第 193–205 行）、规模/内存实验（摘要及 §5；本轮未重新核对每一张性能表）。
- 来源支持：对 `G∈R^{m×n}`，方法使用 `P∈R^{m×r}`、`Q∈R^{n×r}`，把梯度投影为核心矩阵 `PᵀGQ`，在低秩坐标中更新 optimizer state，再映回 `P ρ(PᵀGQ) Qᵀ`。方法目标是全参数学习时降低状态内存；它不是固定低秩权重参数化，也不是冻结底座的 LoRA。投影基会按算法刷新，秩和刷新策略影响近似误差及计算成本。
- 特别边界：投影改变每一步的可用更新方向，不能表述为训练轨迹完全不变；论文在 LLaMA/C4、有限 token 和具体数值精度下报告的显存下降不能推广成任意模型都能在 24GB 卡训练。
- 证据等级：原始论文算法定义、实验摘要和特定实验。

## Muon 与矩阵预条件比较

- 来源：[Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982)，arXiv v1（2025-02-24）；[Fantastic Pretraining Optimizers and Where to Find Them](https://proceedings.iclr.cc/paper_files/paper/2026/file/ea4f0a6ad7088c9b93f22d74c0c9e8c2-Paper-Conference.pdf)，ICLR 2026 正式论文；[Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](https://arxiv.org/html/2512.05620)，对应 arXiv 版本。
- 实际阅读位置：Muon 报告摘要及更新 RMS、权重衰减和参数分组部分（第 71–74、128–152 行）；ICLR 2026 正式论文摘要、方法设置和结论（PDF 第 0–5 页，尤其第 15–35、131–173、245–274 行）；μP 论文摘要及矩阵预条件/扩展设置（本轮核验了论文页面，具体超参迁移表仍需按所引结论定向阅读）。
- 来源支持：Moonshot 报告提出 weight decay 和按形状校准 update RMS，并报告其 scaling-law 设置下约 2× computational efficiency；其矩阵参数与 embedding、LM head、RMSNorm 等参数采用不同优化器。ICLR 2026 比较在 Llama-2 架构的约 0.1B–1.2B 模型、C4-EN 验证目标和 1×–8× Chinchilla 数据预算下，对多种优化器进行系统调参；论文报告速度优势随规模下降，1.2B/8× 条件约为 1.1×，且学习率衰减期间中间 checkpoint 排名可能翻转。其完整实验设置使用 TPU v5 和固定混合数据；tokens-to-target-loss 与端到端 wall-clock 不是同一指标。
- 特别边界：ICLR 论文摘要称“十种”优化器，正文方法表列出包括 AdamW 在内的十一种；正文不依赖该数量。Moonshot、ICLR 与 μP 论文在架构、数据、硬件、调参和质量目标上不同，约 2×、1.1× 或其他加速值不能合并成统一排行榜。ICLR 论文中的 C4-EN loss 是主要 proxy，也同时跟踪下游指标；不能把该单一 proxy 当成全部能力。
- 证据等级：作者技术报告、正式会议论文、原始预印本；均为来源自报而非本地复现。

## SOAP 与 WSD 的比较边界

- 来源：[SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/html/2409.11321)，arXiv v2（2025-01-31）；[Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective](https://arxiv.org/html/2410.05192)，arXiv v3（2024-12-02）。
- 实际阅读位置：SOAP 摘要（第 47–54 行），说明在 Shampoo 预条件器的特征基中使用 Adam 式运行统计；WSD 论文摘要与 WSD-S 方法（第 46–53 行及目录 §5）。
- 来源支持：SOAP 原论文在 360M 与 660M 语言模型预训练中报告其大 batch 设置下相较 AdamW 的 iteration 与 wall-clock 改善，并解释低频更新特征基会带来退化；不能据此认定所有硬件、模型和 batch 上均稳定。WSD 原论文提出 warmup 后稳定学习率主干、需要候选终点时从中分支 decay；作者对 River Valley 解释和 WSD-S 的恢复方式均有各自假设与实验，不等价于所有持续训练的保证。
- 未解决项：本轮只核对 SOAP 摘要和 WSD 的版本/摘要/方法入口，正文中更细的模型预算、刷新频率和 WSD-S 结果必须保留各自论文链接并在需要精确数字时重新定位对应表格。

## 结论范围

各路线解决的资源或更新几何问题不同：Adafactor/GaLore 关注状态占用，Lion 改变方向与状态，Sophia 使用稀疏刷新曲率估计，Muon/Shampoo/SOAP 用矩阵结构处理梯度，WSD 改变学习率轨迹。论文结果应按其任务目标分别报告 token、FLOPs、step time、wall-clock、显存和质量；不同量纲和终点评估不可互换。本轮未运行章节代码、模型或 benchmark。
