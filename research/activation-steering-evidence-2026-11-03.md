# Activation Steering 与 Representation Engineering 证据

检索日期：2026-11-03。主要来源为 ACL、ICML/PMLR、ICLR 正式论文页及 arXiv 原文。下述结果均绑定论文的模型、任务、指标与实现，不能直接外推为其他 checkpoint 或生产安全保证。

## 控制面与基础方法

### AI Steerability 360

- **来源：** Miehling et al., *AI Steerability 360: A Toolkit for Steering Large Language Models*, ACL 2026 System Demonstrations，[正式论文页](https://aclanthology.org/2026.acl-demo.43/)。阅读摘要和出版信息。
- **证据支持：** 论文将控制面分为 input（prompt）、structural（权重/架构）、state（激活/attention）和 output（解码/生成）；通过 steering pipeline 组合方法，并以 use-case class 和 benchmark class 定义任务和比较方法。
- **边界：** 这是 Python 工具与评估入口，支持统一组织实验；接口支持不证明某种 steering 机制有效、鲁棒或安全，也不保证新模型实现兼容。

### Contrastive Activation Addition

- **来源：** Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024，[正式论文页](https://aclanthology.org/2024.acl-long.828/)。阅读摘要。
- **证据支持：** CAA 从正、负行为样本对的 residual-stream activation 差估计 steering vector，并在推理时向 prompt 后生成 token 的激活中加入该方向。论文实验使用 Llama 2 Chat 和指定的多选题/开放生成任务。
- **边界：** 论文报告的行为变化、能力影响和实现位置依赖其模型、层位、数据和任务；不代表对任意 checkpoint 的持续或安全控制。

## 2025—2026 年 steering 评估与方法

### SteeringSafety

- **来源：** Siu et al., *SteeringSafety: Benchmarking Representation Steering in LLMs Across Safety Perspectives*, arXiv v3，2026-08-12，页面标注接收 ICML 2026，[论文](https://arxiv.org/abs/2509.13450)。阅读摘要。
- **证据支持：** 基准覆盖九种安全视角、18 个数据集；摘要列出 Gemma-2-2B、Llama-3.1-8B、Qwen-2.5-7B。作者报告方法、模型与目标视角的配对影响明显，并观察到目标间的纠缠；特定结果包括社会行为分数下降、拒答 steering 影响常识道德判断，以及幻觉 steering 对政治倾向的模型相关变化。
- **边界：** 这些是所列模型和 benchmark 下的测量，不是现实世界行为发生率；不能用拒答率或单项目标分数代替安全总体判断。

### Endogenous Steering Resistance

- **来源：** Mckenzie et al., *Endogenous Resistance to Activation Steering in Language Models*, ICML 2026, PMLR 306，[正式论文页](https://proceedings.mlr.press/v306/mckenzie26a.html)。阅读摘要。
- **证据支持：** 在 SAE latent steering 设置下，作者报告 Llama-3.3-70B 更明显地出现生成中恢复，较小 Llama-3 与 Gemma-2 模型中较少；研究还描述了相关 latent、提示自监控和自我纠正微调实验。
- **边界：** 恢复现象及其干预结果绑定作者的 steering 设定和模型。它说明需检查完整生成与多轮持续性，不能推出所有 steering 都会失效，或某个 profile 在其他模型有相同抵抗机制。

### Contrastive Covariance Steering

- **来源：** Xenos et al., *Contrastive Covariance Subspaces for Activation Steering*, arXiv v1，2026-09-29，[论文](https://arxiv.org/abs/2610.14025)。阅读摘要。
- **证据支持：** 论文提出 Contrastive Covariance Steering（CCS），以对比方差目标计算子空间，保留目标概念变化并抑制其他类别共享的变化。作者报告三种 LLM、四个 steering domain 的 12 个 model-domain 设置都优于其对照，另有 23 名评审对 296 条生成的比较及多轮实验。
- **边界：** 当前为预印本；摘要中的优胜范围、相对改进和人类比较受该模型/领域/基线与 judge 设置约束，尚不能支持跨模型迁移或生产安全优势。

### SteerEval

- **来源：** Xu et al., *How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities*, ACL 2026，[正式论文页](https://aclanthology.org/2026.acl-long.1443/)。阅读摘要与出版信息。
- **证据支持：** SteerEval 覆盖 language features、sentiment、personality 三类行为域，并按 L1“表达什么”、L2“如何表达”、L3“如何具体落实”分层；作者报告控制性会随约束变得细粒度而下降。
- **边界：** 结论适用于论文 benchmark 的行为域和任务规格；不能据此断言所有目标或生产策略都具有相同的粒度难度曲线。

### 输入扰动下的 activation-steering 鲁棒性

- **来源：** Le & Le, *Adversarial Robustness of Activation Steering in Large Language Models*, arXiv v1，2026-06-05，[论文](https://arxiv.org/abs/2606.07696)。阅读摘要。
- **证据支持：** 作者比较四种方向提取方法、三类文本扰动、六个人格和五个 1.5B—30B 模型；摘要报告方向稳健性最高下降 64%，自动选择的最佳层位在扰动后最多偏移 17 层，并提出从扰动输入重提方向只能部分恢复部分设置。
- **边界：** 这些是预印本所用攻击、模型、扰动和评分口径下的实验结果；不是线上攻击成功率或所有输入改写下的普遍退化率。

### Spherical Steering

- **来源：** You et al., *Spherical Steering: Geometry-Aware Activation Rotation for Language Models*, ICML 2026, PMLR 306，[正式论文页](https://proceedings.mlr.press/v306/you26a.html)。阅读摘要。
- **证据支持：** 方法沿 geodesic 将激活旋向目标方向，以保持激活范数，并按输入不确定性设置 confidence gate。作者报告在其多选题 benchmark 上相对加法基线有改进，同时维持所测开放生成质量。
- **边界：** 摘要中的收益限于所列 benchmark 和作者实验，不能证明范数保持会普遍避免表示退化，也不能证明对安全行为的收益。

### PID Steering

- **来源：** Nguyen et al., *Activation Steering with a Feedback Controller*, ICLR 2026，[正式论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fa5617c176e76fee83f3f9947fdf9f3f-Abstract-Conference.html)。阅读摘要。
- **证据支持：** 论文将常见激活 steering 解释为比例控制器，并提出 Proportional-Integral-Derivative（PID）steering：比例项对齐目标语义方向，积分项跨层累积误差，微分项抑制快速变化造成的过冲。
- **边界：** 摘要所称稳定性联系依赖论文的控制模型、假设和实验配置；不能直接当作任意生成过程或运行时策略的稳定保证。

## 治理框架状态

### NIST AI RMF

- **来源：** NIST, [AI Risk Management Framework 官方页面](https://www.nist.gov/itl/ai-risk-management-framework)，2026-11-03 页面访问。阅读框架概览与当前版本说明。
- **证据支持：** 页面称 AI RMF 1.0 正在修订，并说明 2026-04-07 发布了面向关键基础设施可信 AI 的 Profile concept note。该框架面向 AI 产品、服务和系统的风险管理，声明为自愿使用。
- **边界：** 这是风险治理框架的状态与用途说明，不定义 activation steering 的技术效能、统一阈值或发布认证；正式使用时需固定所引用版本及适用组织范围。
