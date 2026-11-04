# LLM 数据集估值：来源证据

记录日期：2026-11-03

本记录只保存来源支持的研究结论、版本和边界；书册审阅决定与修改过程写入 `progress/`。

## Convex Dataset Valuation for Post-Training

- **正式版本：** Zeng et al., [“Convex Dataset Valuation for Post-Training”](https://proceedings.mlr.press/v306/zeng26k.html)，ICML 2026 / PMLR 306；[arXiv:2605.16704v1 HTML](https://arxiv.org/html/2605.16704v1)。
- **实际阅读范围：** 摘要、§§1–4.4 的问题设定、梯度估值、KMM 方法、实验条件与分析；核对官方 PMLR 会议页面和论文 HTML。
- **来源支持：** 论文把后训练数据估值表述为给定目标任务、度量和预算下的辅助数据集选择。数据值是目标相关、可正可负、且仅在同一模型和后训练设置内有比较意义。单步梯度对齐可作为单数据集的局部信号，但其可加和假设忽略辅助数据集之间的冗余和交互；作者提出在梯度空间使用 kernel mean matching 的凸优化，把目标梯度对齐与候选数据间的冗余一并纳入估值。
- **实验范围与报告：** 主要受控实验使用 LoRA rank 8，覆盖 Aya 的多语言 SFT、Big-Math 衍生多语言 GRPO，以及数学、代码和通用指令辅助数据。评估固定 optimizer update 总数，使增添候选数据时重新分配辅助训练预算，而非单纯增加训练量。作者在多语言实验中报告 KMM 估值相对梯度单点评分通常有提升；在丹麦语目标、8 个候选语言的 56 个三源子集比较中，task-vector + KMM 与实测完整训练效用的 Spearman 相关为 0.396、`p=0.003`，其他对照相关较弱。此相关结果来自指定语言、8 源、3 源子集和训练方案。
- **证据等级：** source-backed（同行评审 ICML 2026 论文中的方法、实验和作者报告结果）。
- **适用边界：** 分数不是数据的内在价值或通用市场价格；结论限于固定的目标任务/度量、基模型、后训练算法、候选池和预算。梯度空间方法依赖局部近似、样本预览及论文中给出的正则化/任务条件；作者指出目标信号过弱时估值优势会减弱，并提醒不能不经人工判断就据此定价或排除数据。论文不取代许可、隐私、安全及用途授权审查。

## DataInf: LoRA 微调中的 influence 近似

- **版本与来源：** Kwon 等，[“DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models”](https://arxiv.org/abs/2310.00902v3)，arXiv v3，2024-03-13；ICLR 2024 [正式会议页面](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e84a0f233611a1dc8fb794dc52415a3-Abstract-Conference.html)。
- **实际阅读范围：** arXiv HTML 摘要、§3 方法、§4 实验及附录 D 中与模型和微调配置相关的说明。
- **来源支持：** DataInf 使用面向低秩参数高效微调的闭式近似来降低 influence 计算的时间和内存开销。论文的影响样本识别实验包括 LoRA 微调的 Llama-2-13B-chat；实验也覆盖 stable-diffusion-v1.5。作者报告了对误标样本识别的应用。
- **证据等级：** source-backed（论文方法和作者报告的实验结果）。
- **适用边界：** 论文指出精确 influence 的成本高；近似误差随可训练参数规模、LoRA 秩和设定变化。近似误差分析在 RoBERTa-large 上完成，不能据此声称大模型全参数预训练的逐样本贡献可被精确恢复。

## LoGra: 大模型 influence 的梯度投影

- **版本与来源：** Choe 等，[“What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions”](https://arxiv.org/abs/2405.13954v1)，arXiv v1，2024-05-22。
- **实际阅读范围：** arXiv HTML 摘要、§§2–4.2、表 1 和附录中 Llama-3-8B-Instruct / 1B-token 实验设置。
- **来源支持：** LoGra 依据反向传播中的梯度结构直接计算低秩投影梯度，减少完整梯度物化及 influence 计算的显存需求。论文在 Llama-3-8B-Instruct、1B-token 数据和一张 A100 GPU 的半精度设置中，与 EK-FAC influence 比较；用训练梯度批量 256 与测试批量 256 的 influence 计算时，作者报告约 79,004 对/秒、15 GB 显存和 3.5 TB 投影梯度存储，对照 EK-FAC 报告 12.2 对/秒、75 GB 显存和 89 GB 存储。作者将接近 6,500 倍吞吐归因于固定批量下重叠磁盘读取与计算。
- **证据等级：** source-backed（arXiv 论文中的方法、实验表格和作者报告）。
- **适用边界：** 吞吐比是指定硬件、批量、模型、数据量、实现和 EK-FAC 对照下 influence 计算阶段的结果，不代表训练吞吐或普遍加速比例；高吞吐同时伴随较大的投影梯度磁盘存储。

## PreSelect: 预测式预训练数据筛选

- **正式版本：** Shum 等，[“Predictive Data Selection: The Data That Predicts Is the Data That Teaches”](https://proceedings.mlr.press/v267/shum25a.html)，ICML 2025 / PMLR 267。
- **实际阅读范围：** PMLR 论文正式页面和作者摘要中的方法、模型规模、token 数和对照条件。
- **来源支持：** PreSelect 用 fastText scorer 进行数据选择。作者在 1B 和 3B 参数模型实验中报告，30B 个所选 token 训练的模型超过了其 300B token vanilla 对照，并报告相对的计算量降低。
- **证据等级：** source-backed（同行评审 ICML 2025 论文的正式页面和作者报告）。
- **适用边界：** 论文的预测信号基于文本域与下游目标对齐的观察；30B/300B 结果绑定论文数据、目标、模型和训练协议，不是跨任务、跨模型的普遍预算保证。

## BLISS: 双层 proxy influence 选择

- **正式版本：** Hao 等，[“BLISS: A Lightweight Bilevel Influence Scoring Method for Data Selection in Language Model Pretraining”](https://proceedings.mlr.press/v306/hao26b.html)，ICML 2026 / PMLR 306。
- **实际阅读范围：** PMLR 正式页面和作者摘要中的方法、模型系列与主要对比结果。
- **来源支持：** BLISS 用小型 proxy 的多步加权训练更新优化 score model，形成双层选择目标；方法从头运行且不依赖外部预训练 oracle。作者在 410M、1B、2.8B Pythia 和 0.5B LLaMA 的选定 C4 子集上评估，并在 1B 配置中报告相对其 state-of-the-art 比较方法达到同等表现的速度提升。
- **证据等级：** source-backed（同行评审 ICML 2026 论文的正式页面和作者报告）。
- **适用边界：** 1.7 倍速度数字限于论文指定的 1B 配置、候选子集、对照方法和评测任务；不能直接视为所有预训练管线的收益。

## OPUS: 优化器更新空间中的动态选择

- **正式版本：** Wang 等，[“OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration”](https://proceedings.mlr.press/v306/wang26gc.html)，ICML 2026 / PMLR 306。
- **实际阅读范围：** PMLR 正式页面和作者摘要中的 utility 定义、投影与采样机制、开销和训练阶段。
- **来源支持：** OPUS 将候选数据有效更新投影到由稳定、分布内 proxy 得到的目标方向；摘要列出 Ghost、CountSketch 和 Boltzmann sampling，并报告 4.7% 的额外计算开销。作者报告在从头预训练和 mid-training 设置中与既有选择方法比较，也描述了静态 toxicity 过滤的潜在用途。
- **证据等级：** source-backed（同行评审 ICML 2026 论文的正式页面和作者报告）。
- **适用边界：** 开销、收益和过滤能力绑定论文的模型、语料、优化器、目标方向和评估协议；摘要级结果不足以证明其他训练栈具有相同代价或收益。

## Mechanistic Data Attribution: 可解释单元的训练来源

- **正式版本：** Chen 等，[“Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units”](https://proceedings.mlr.press/v306/chen26cv.html)，ICML 2026 / PMLR 306。
- **实际阅读范围：** PMLR 正式页面和作者摘要中的方法、模型系列、干预和结论范围。
- **来源支持：** 作者用 influence functions 将可解释 LLM 单元追溯到训练样本，并在 Pythia 系列上报告：针对高 influence 样本进行少量定向增补或移除会改变特定 interpretable heads，随机干预没有相同效果。摘要还报告针对 induction head 的干预伴随 ICL 能力变化。
- **证据等级：** source-backed（同行评审 ICML 2026 论文的正式页面和作者报告）。
- **适用边界：** 这是特定模型家族、可解释单元和干预目标的机制级证据，不是对总体任务能力或数据源经济价值的通用因果估值。

## RISE: 输出层 readout sketch

- **版本与来源：** Ran 等，[“Sketching the Readout of Large Language Models for Scalable Data Attribution and Valuation”](https://arxiv.org/abs/2604.16197v2)，arXiv v2，2026-07-19。
- **实际阅读范围：** arXiv HTML 摘要、§§1、2.5、3.1–3.3、结论及附录 C.1 的架构/稳定性假设说明。
- **来源支持：** RISE 聚焦 LM head 的输出层影响信号，以 outer-product 结构构造 lexical residual（RH）和 semantic projected-error（GH）双通道，并用 CountSketch 压缩。论文在 Pythia（14M–6.9B）和 OLMo（1B–32B）系列的 Howdy 后门检索、Finance–Medical 域区分与 Brain Rot 数据选择三类任务上报告结果；相对 RapidIn 的最高 112 倍索引存储压缩来自 Pythia-1B 对照，OLMo-3-32B 设置中 RapidIn 与 ZO-Inf 出现内存不足。Brain Rot 闭环实验用 OLMo-3-32B 给 50K 候选排序，取 top-5K 继续训练 Pythia-1B；表 4 中 RISE 相对 TrackStar 的控制集困惑度为 2.33 对 6.63，RULER-CWE 为 4.10% 对 2.54%。
- **证据等级：** source-backed（2026-07-19 arXiv v2 预印本中的方法、实验和作者报告；非同行评审结论）。
- **适用边界与来源内差异：** 112 倍、32B 可运行性和闭环结果仅对应上述模型、候选池与任务。摘要/引言对 RULER-CWE 写“4.1×”，而 §3.3 和表 4 的 4.10% 对 2.54% 对应约 1.6×；正文引用闭环对照时采用 §3.3 与表 4 的数字，不采用摘要中的倍数表述。输出层梯度集中、CountSketch 误差与排名稳定性不等于目标系统上的干预因果证据。
