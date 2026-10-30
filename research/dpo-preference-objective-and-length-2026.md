# DPO 的推导、长度口径与参考策略变体

检索日期：2026-10-29。

本记录只保存联网核验得到的知识点、来源版本和证据边界，不记录书籍修改过程或章节状态。

## 标准 DPO 的目标与假设

- Rafailov et al., “Direct Preference Optimization: Your Language Model is Secretly a Reward Model,” arXiv:2305.18290v3（2024-07-29），[论文 HTML](https://arxiv.org/html/2305.18290)；发表于 NeurIPS 2023。
  - 实际阅读范围：§3 KL 正则 reward maximization 与 Bradley–Terry 偏好；§4 Eq. 4–7 闭式最优策略、reward 重参数化和分类目标；§5.1 隐式 reward；附录 A 的相应推导说明。
  - 来源支持：固定 prompt 下，KL-regularized reward optimum 可写为参考策略乘以指数化 reward 再归一化；取对数后 reward 等于参考策略的 log-ratio 项加上只依赖 prompt 的归一化项。Bradley–Terry 比较同 prompt 的两个回答时，归一化项抵消，得到 DPO 的 reference-adjusted log-probability margin 与 sigmoid 分类 loss。来源亦说明标准方法训练阶段无需额外采样循环或单独拟合 reward model；偏好数据仍是静态 comparison dataset，候选可来自 SFT policy。
  - 证据等级：source-backed（论文理论推导及其指定的离线 preference training 方法）。
  - 适用边界：推导依赖正的 reference 支持、固定 reference、KL 正则化 reward objective 和 Bradley–Terry/相应偏好模型假设；参数化 policy 只是在有限数据上拟合目标，不自动等于理论最优 policy。DPO loss 降低表示对这些标签的分类拟合变化，不单独证明正确性、偏好质量或安全提升。论文特定任务实验不能外推为 DPO 在所有任务上优于 PPO-style RLHF。

## 序列 log probability 与长度口径

- Rafailov et al. 上述论文 §4 Eq. 7；Meng, Xia, Chen, “SimPO: Simple Preference Optimization with a Reference-Free Reward,” arXiv:2405.14734v3（2024-11-01；NeurIPS 2024），[论文 HTML](https://arxiv.org/html/2405.14734)。
  - 实际阅读范围：DPO §4 序列策略概率和偏好目标；SimPO §2.1–2.3 与 §4.1/4.2/4.4 的 length-normalization 分析及作者实验。
  - 来源支持：自回归回答的序列概率由逐 token 条件概率相乘，log probability 对应求和；DPO 比较 chosen/rejected 的完整回答 log-ratio。若将序列和改为按 token 平均，会改动长度不同样本的 margin 和训练目标。SimPO 提出使用平均 token log probability 作为隐式 reward、加入 target margin，并移除单独的 reference model；作者报告其在论文指定的 Mistral、Llama 3、Gemma 2 等模型及 AlpacaEval 2/Arena-Hard 等设定中的结果。
  - 证据等级：source-backed（目标定义、方法设计与论文实验）。
  - 适用边界：SimPO 的收益、长度行为和 benchmark 排名依作者所用模型、基线、超参数与评测；不能据此断言 SimPO 普遍优于 DPO，或简单的 length normalization 在任何数据上都能消除长度效应。response mask、EOS、tokenizer 和序列求和/平均约定必须按具体训练实现核对。

## 2026 年 likelihood-displacement 研究候选

- Im et al., “Normalized Rewards for Preference Optimization,” arXiv:2607.16240（2026-06-26），[论文 HTML](https://arxiv.org/html/2607.16240)。
  - 实际阅读范围：摘要和 §1 引言。
  - 来源支持：作者研究 direct alignment algorithms 中 chosen/rejected response likelihood 同时下降的 likelihood-displacement 现象，并在 DPO 与 SimPO 上测试维持长度归一化概率的附加正则；摘要报告其 Llama 3.1 8B Instruct 指定任务与一般 benchmark 结果。
  - 证据等级：source-backed（当前为 arXiv 预印本摘要/引言中的作者报告）。
  - 适用边界：预印本和有限模型/数据/benchmark 结果；摘要不能证明机制适用于全部 direct preference objectives，也不能作为通用长度或过度优化修复方案。

## RLHF 与 DPO 的 2026 年理论比较

- Shi et al., “Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO,” ICML 2026, PMLR 306, pp. 112109–112149，[会议论文页面](https://proceedings.mlr.press/v306/shi26s.html)。
  - 实际阅读范围：PMLR 官方会议论文摘要。
  - 来源支持：作者把两阶段 RLHF 与 DPO 的差别分解为精确优化下的显式表示差异和有限样本下的隐式表示差异；摘要称在不同模型失配情形下 RLHF、标准 DPO 或 online DPO 可以互相胜出，并给出一个稀疏真实 reward 构造，在该构造下 RLHF 用更少样本恢复有效 reward model。
  - 证据等级：source-backed（正式会议论文摘要的理论结论概述）。
  - 适用边界：具体优势依赖论文的 reward/policy 函数类、失配条件、样本构造和优化假设；摘要不能支持任意任务的算法排名或实际训练样本节省幅度。

## DPO 偏好数据质量（2025）

- Pan et al., “What Matters in Data for DPO?,” NeurIPS 2025 Main Conference，[官方会议摘要](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3f37b8fbd43303106dd141a602838ad5-Abstract-Conference.html)。
  - 实际阅读范围：NeurIPS 官方论文页摘要；未在本记录中复核全文理论条件和实验表格。
  - 来源支持：摘要称作者从理论和实证角度研究偏好数据分布，报告 chosen response 质量在其设定中对 DPO 目标表现起主导作用，而 rejected response 质量的影响相对较小；在线 DPO 在该分析下可化为对 chosen responses 的监督微调。
  - 证据等级：source-backed（NeurIPS 2025 同行评审论文的官方摘要）。
  - 适用边界：不表示 rejected 样本在所有训练目标或数据构造中均无用；优势依赖论文的偏好分布、模型、任务和在线数据生成设置，摘要不能支持通用标注预算分配结论。

## 固定版本训练器接口与数据契约

- Hugging Face TRL DPOTrainer v1.14.1：[DPOTrainer 文档](https://huggingface.co/docs/trl/v1.14.1/dpo_trainer)、[dataset formats](https://huggingface.co/docs/trl/v1.14.1/dataset_formats)、[RewardTrainer 文档](https://huggingface.co/docs/trl/v1.14.1/reward_trainer)。
  - 实际核对范围：DPO 的标准/对话型数据格式、DPOTrainer 参数签名、processing_class、DPOConfig 签名、参考策略说明、记录指标和 loss 类型。
  - 来源支持：显式 prompt 的偏好行使用 prompt/chosen/rejected；对话型数据由 trainer 应用 chat template；processing_class 要设置左侧 padding 并有 pad token；ref_model=None 使用训练开始前的初始 policy；日志的 logps/chosen 与 logps/rejected 是 completion token 的平均 log-probability。v1.14.1 的 DPOConfig 签名包含 max_length 与 truncation_mode，没有 max_prompt_length 参数。
  - 证据等级：source-backed（固定版本官方接口说明与签名）。
  - 适用边界：这些行为绑定 TRL v1.14.1。其他版本参数名和 adapter/reference 机制需要分别核对。Trainer 平均 log-probability与手写按 response mask 求和的序列 log-probability是不同 reduction，不可直接比较 loss 数值；文档版本也不等于已联调依赖锁。
- Transformers v5.17.0 [bitsandbytes 微调说明](https://huggingface.co/docs/transformers/v5.17.0/quantization/bitsandbytes) 与 PEFT v0.18.0 [LoRA API](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora)。
  - 来源支持：Transformers 指出训练时不必传 device_map，device_map=auto 只用于推理；NF4 用于 4-bit base model fine-tuning。PEFT 指出 GPT-2 Conv1D 采用 fan_in/fan_out 存储，LoRA 配置需设 fan_in_fan_out=True；默认初始化将 LoRA B 置零，使初始 adapter 为 no-op。
  - 证据等级：source-backed（固定版本官方文档）。
  - 适用边界：接口支持依赖相应模型实现、硬件及依赖组合；未通过文档阅读证明具体环境能加载、训练或达到某种吞吐。
- PyTorch v2.14 [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)、[logsigmoid](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.logsigmoid.html)、[log_softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.log_softmax.html) 与 [gather](https://docs.pytorch.org/docs/2.14/generated/torch.gather.html) 文档确认本章所用稳定损失与逐 token log-probability 计算接口。
  - 证据等级：source-backed（固定版本官方 API 参考）；边界是 API 文档支持局部语义，不代表本章代码已执行。
- 章节中 tiny-gpt2 数据使用 Hub commit `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` 的[固定文件树](https://huggingface.co/sshleifer/tiny-gpt2/tree/5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be)，以固定模型、tokenizer 和配置文件身份；该小模型仅作加载/shape/数据流教学样例。

## 生命周期

标准 DPO 的 reference-adjusted preference objective：active；SimPO 为对 reference 依赖和序列长度目标的一个特定方法变体。Likelihood displacement 的成因和 2026 正则化方案仍需结合论文全文、后续复现及不同 direct preference objectives 分别核验。
