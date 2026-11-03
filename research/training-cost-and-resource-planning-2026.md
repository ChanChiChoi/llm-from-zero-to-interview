# 训练成本与资源规划：来源证据

检索与阅读日期：2026-11-02。

本记录保存成本估算、训练效率、能耗和生命周期规模研究的来源事实及边界；章节审阅与修改决定只记录在 `progress/`。

## PaLM 的 MFU/HFU 口径

- **来源与版本：** Chowdhery et al., [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/html/2204.02311v5)，arXiv v5，2022-10-05。
- **实际阅读位置：** §4 Training Efficiency、MFU/HFU 表及 Appendix B 的计算说明。
- **来源支持的事实：** PaLM 540B 使用 6,144 个 TPU v4 芯片；论文按包含自注意力的口径报告 46.2% MFU，另报告 57.8% HFU。MFU 分子按目标模型所需 FLOPs 计算，HFU 将 rematerialization 等实际执行 FLOPs 纳入，因此两个数字不是同一个分子下的“利用率”。
- **证据等级与边界：** source-backed（原论文对特定模型、TPU、编译器与并行配置的测量/分析）。不能将 PaLM 数值直接当成其他集群的目标效率。

## MegaScale 强扩展与长程恢复

- **来源与版本：** Jiang et al., [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/html/2402.15627)，arXiv v1，2024-02-23。
- **实际阅读位置：** §6.1、Table 2、§6.2—6.3 的生产训练与故障统计。
- **来源支持的事实：** 175B 模型的强扩展实验使用 2,048 序列长度；在 3,072—12,288 GPU 范围内保持 batch size 6,144，并报告训练 300B token 的时间。MegaScale MFU 从 59.1% 降至 55.2%，Megatron-LM 对照从 48.7% 降至 41.2%；Table 2 括号中的 1.21×—1.34× 是 MFU 比值。12,288 GPU 行另报告 MegaScale 1,984.0k tokens/s、Megatron-LM 1,466.8k tokens/s，按表中舍入数相除约 1.35×吞吐。
- **长程观察：** 作者另报告一个运行数周、使用超过 10,000 GPU 的专有训练作业中发生 100 余次重启；对其故障记录，超过 90% 的软件/硬件异常由框架自动检测、定位和恢复，检测及诊断平均少于 10 分钟；从最新 checkpoint 追赶回崩溃前进度少于 15 分钟，effective training time rate 超过 90%。该指标定义为迭代数乘迭代训练时间再除总训练时间。
- **证据等级与边界：** source-backed（同论文受控性能对照与作者的专有生产作业报告）。强扩展值依赖 175B 模型、batch、序列长度和论文集群；恢复率、故障率和追赶时间是团队自报，不是其他集群的保证。MFU 比值与 tokens/s 比值应分别表述，不能混称。

## Llama 2 与 OLMo 的能耗报告口径

- **来源一：** Touvron et al., [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/html/2307.09288)，§2.2.1。
- **来源支持的事实：** 作者报告 Llama 2 预训练总计 3,311,616 A100 GPU-hours，并估计排放为 539 tCO₂e。估算使用 A100 GPU 功率上限/TDP（350W 或 400W）及其采用的碳效率口径；作者明确说明没有计入互联、非 GPU 服务器和数据中心冷却功率，也没有计入硬件制造相关排放。报告还说明所估算排放由 Meta sustainability program 抵消；抵消与物理排放估算是不同事实。
- **来源二：** Groeneveld et al., [OLMo: Accelerating the Science of Language Models](https://arxiv.org/html/2402.00838v4)，Appendix B。
- **来源支持的事实：** 作者以单节点 25 ms 采样功率的整次训练平均值乘节点数，再乘 PUE=1.1；报告 7B 预训练消耗约 239 MWh。其碳估算按训练数据中心所在地的电网因子计算，并说明 operational-emissions 数字不包括 embodied carbon、设施全生命周期运行、反弹效应、水耗和采矿等影响，因而是下界性质的报告。
- **证据等级与边界：** 两者均为 source-backed（各自论文报告的估算）。Llama 2 的 GPU TDP 估算与 OLMo 的节点遥测/PUE 方法不同；模型、集群、地区、电网年份和核算范围不同，不可用其绝对数值做“哪一个更绿色”的横向排序。

## 把推理请求计入 scaling 的研究

- **来源：** Besiroglu et al., [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/html/2401.00448)，arXiv:2401.00448。
- **实际阅读位置：** 摘要、§§2—6、Appendix B 的成本情景与 Appendix C 的训练模型范围。
- **来源支持的事实：** 论文在给定预训练 loss 作为质量约束下，联合最小化训练与推理计算/成本；摘要称约 10⁹ 次推理请求的需求可能支持比 Chinchilla-optimal 更小、训练更久的模型。作者训练 47 个 MPT 风格模型，规模约 150M—6B 参数，数据预算为每参数 10—10,000 token，以检验极高 token/parameter 区间的 loss 趋势。real-world cost 情景假设每次请求 70 个输入 token、215 个输出 token，训练和输入 MFU 50%、输出 MFU 1%，并使用 A100 硬件与特定价格；论文公式简化处理了延迟约束，并假设给定质量下不同规模模型面对相同请求需求。
- **证据等级与边界：** source-backed（论文拟合、其有限规模实验和作者构造的成本情景）。约 10⁹ 请求结论取决于请求长度、质量约束、训练/推理 MFU、硬件、价格和延迟假设；不能写成“小模型总成本必然更低”。47 个中小规模实验不等于数百亿参数生产模型的直接验证。

## Kaplan 与 Chinchilla 的固定计算预算分配

- **来源：** Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/html/2203.15556) 对 Kaplan et al. scaling-law 结果的比较与自身实验。
- **来源支持的事实：** Chinchilla 论文概述，在计算预算扩大 10× 时，Kaplan 的建议约为参数扩大 5.5×、训练 token 扩大 1.8×；作者自己的实验则支持参数和训练 token 近似同比例增长。这里是在特定语言模型预训练损失、模型、数据和预算范围内比较两种 compute-optimal 分配建议。
- **证据等级与边界：** source-backed（论文方法与实验范围）。不可把这个比例直接迁移至不同架构、数据质量、多模态、SFT/RL 或服务生命周期成本。
