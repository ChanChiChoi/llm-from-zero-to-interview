# RLHF、偏好代理与策略优化：来源证据

检索日期：2026-10-29。

本记录只保存联网核验得到的知识点、来源版本和证据范围，不记录书籍修改过程、逐讲状态或工作交接。

## PPO-style RLHF 的代表性流程

- Ouyang et al., “Training language models to follow instructions with human feedback,” arXiv:2203.02155v1（2022），[论文 HTML](https://arxiv.org/html/2203.02155)；发表于 NeurIPS 2022。
  - 实际阅读范围：摘要、§3.1 方法流程、§3.5 RM/RL 设定、Eq. 1–2、KL 说明和结果摘要。
  - 来源支持：论文从 GPT-3 开始，收集示范进行 SFT，再以同一 prompt 下的回答比较训练标量 Reward Model，之后用 PPO 优化；PPO reward 中加入来自 SFT 模型的逐 token KL penalty。论文在其 API prompt 分布的人评中报告了偏好改善，并报告真确性/毒性等特定指标变化。
  - 证据等级：source-backed（具体 GPT-3/InstructGPT 流程、标注分布和评测）。
  - 适用边界：结果依其模型、标注者、提示分布和训练配方；论文同时指出仍有编造事实、漏跟指令等错误。三阶段流程不是所有偏好优化方法的必需顺序，也不是所有“有用/诚实/安全”属性的保证。

## 成对偏好与 Reward Model 代理

- Ouyang et al. 上述论文 §3.5；Stiennon et al., “Learning to summarize from human feedback,” arXiv:2009.01325v3（2022-02-15），[论文 HTML](https://arxiv.org/html/2009.01325)，发表于 NeurIPS 2020。
  - 实际阅读范围：Ouyang §3.5 的 pairwise loss、reward 差不变性和 RL reward；Stiennon §3.4 模型、§4.3 “What happens as we optimize the reward model?”、Limitations/Discussion。
  - 来源支持：常见 Bradley–Terry 风格目标用 $\sigma(r_w-r_l)$ 表示同一 prompt 下 chosen 相对 rejected 的偏好概率，优化 log-loss；RM 分数是排序代理。Stiennon 在 TL;DR 摘要任务中报告轻度优化时人工偏好改善，继续优化一个较早 RM 后人类偏好和 RM 预测分离，极端时出现反相关。
  - 证据等级：source-backed（论文目标定义和指定摘要实验）。
  - 适用边界：RM loss 只定义其标注模型下的比较目标，不是事实准确性或绝对效用；过度优化的曲线限于该摘要任务、RM 版本和实验参数，不提供跨任务统一阈值。

## PPO 的 clipped surrogate

- Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv:1707.06347v2（2017），[论文 PDF](https://arxiv.org/pdf/1707.06347)。
  - 实际阅读范围：摘要、§2.2 trust-region 背景、§3 clipped surrogate（Eq. 6–7）及相关解释。
  - 来源支持：策略比率 $r_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$ 与优势估计进入 clipped surrogate；clip 的动机是移除比率超出范围后继续改善代理目标的激励，并以 clipped/unclipped 的较小项形成 pessimistic surrogate。原论文在机器人控制和 Atari 等一般 RL 任务中评估 PPO。
  - 证据等级：source-backed（通用 PPO 方法与原论文实验）。
  - 适用边界：clip 是 surrogate objective 的操作，不是硬性的 KL 约束，也不保证参数更新后所有概率比率都在裁剪区间内。PPO 原论文不是语言模型 RLHF 结果；InstructGPT 等工作才将其用于偏好训练。

## DPO 与固定偏好数据

- Rafailov et al., “Direct Preference Optimization: Your Language Model is Secretly a Reward Model,” arXiv:2305.18290v3（2024-07-29），[论文 HTML](https://arxiv.org/html/2305.18290)，发表于 NeurIPS 2023。
  - 实际阅读范围：摘要、§1、§4 Direct Preference Optimization、§5.1 和 §6 实验概述。
  - 来源支持：在偏好模型与 KL-regularized reward maximization 等假设下，DPO 可将目标重写为使用偏好 pair 的分类损失；固定数据集优化时不显式训练独立 RM，也不需要 PPO rollout 循环。论文 v3 §4 Eq. (7) 的目标为 `-E_D[log σ(β log(πθ(y_w|x)/πref(y_w|x)) − β log(πθ(y_l|x)/πref(y_l|x)))]`，其中 `y_w`、`y_l` 分别是 chosen 与 rejected 序列。原方法仍使用 reference policy 定义隐式 reward，并说明可重用由先前 policy 采样的离线偏好数据。
  - 证据等级：source-backed（理论重写和论文指定的情感控制、摘要、对话任务实验）。
  - 适用边界：结果限于论文的偏好假设、reference 初始化、数据和任务；“DPO 无需显式 RM/PPO rollout”描述的是算法训练阶段，不表示 preference 数据从未经模型采样，也不证明 DPO 在所有任务上优于 PPO。

## RLVR 中的 spurious reward 观察

- Shao et al., “Spurious Rewards: Rethinking Training Signals in RLVR,” ICML 2026, PMLR 306, pp. 110167–110207，[会议论文页面](https://proceedings.mlr.press/v306/shao26d.html)；arXiv:2506.10947v2，[HTML](https://arxiv.org/html/2506.10947v2)。
  - 版本与阅读范围：ICML 2026 正式会议版本；同时阅读 arXiv:2506.10947v2（2026-02-25）HTML，实际检查摘要、§§1–5、相关 Figure 2–6/Table 1–2 和附录中的机制/定性分析，并核对 PMLR 出版页面的版本信息。
  - 来源支持：论文固定在 Qwen2.5-Math 使用 GRPO 和 DeepScaleR 训练数据的设置比较 ground-truth、majority-vote、format、incorrect-label 与 random reward；在 MATH-500 报告 random reward 对 Qwen2.5-Math-7B 提升 21.4 个百分点、ground-truth reward 提升 29.1 个百分点。其他模型结果显示 spurious rewards 不稳定或不迁移。作者消融发现去除 clipping 后随机奖励不再带来稳定提升，并以 clipping bias 解释其如何放大高概率 token/行为；Qwen2.5-Math-7B 的 Python 风格 code-reasoning 比例及准确率变化是作者讨论的既有行为案例。
  - 证据等级：source-backed（同行评审论文正文及其指定实验、消融和作者解释）。
  - 适用边界：结果受模型家族、底模行为、prompt、训练数据/预算、GRPO 配置与数学 benchmark 限制；机制解释是论文的分析与假设，不能扩展成随机奖励在其他模型/任务上有效的因果定律。论文明确将随机/错误奖励作为分析工具，不是训练真实能力的建议；这也不是人类偏好 RM/PPO 的证据。

## PPO/GRPO 日志中的 KL 指标

- Hugging Face TRL, “Logging,” `main` 文档，[官方页面](https://huggingface.co/docs/trl/main/en/logging)，访问日期 2026-10-29。
  - 实际阅读范围：PPO Logging 与 GRPO Logging 小节。
  - 来源支持：该文档将 PPO 的 `objective/kl` 描述为当前 policy 与 reference policy 的平均 KL，将 `policy/approxkl_avg` 描述为相邻 PPO policy 间的近似 KL，并明确两者不是同一指标；GRPO 部分也单列 policy/reference KL 与 clipping、entropy 指标。
  - 证据等级：source-backed（框架当前文档说明）。
  - 适用边界：`main` 是随代码更新的文档分支，不是固定安装版本；实际训练应记录 TRL 版本、指标生成代码与配置，不能把字段名和归约方式外推到其他 release。

- Hugging Face TRL, “Asynchronous GRPO,” v1.16.0，[官方版本文档](https://huggingface.co/docs/trl/v1.16.0/en/async_grpo_trainer)，访问日期 2026-10-29。
  - 实际阅读范围：queue、rollout 与 sample metric 的定义。
  - 来源支持：该版本列出 `sample/staleness_mean/max`、`sample/dropped_stale_total`、queue wait/backpressure 等诊断字段，用于区分 rollout 数据相对当前 policy 的版本陈旧程度、样本等待与生成/训练侧瓶颈。
  - 证据等级：source-backed（固定文档版本的指标说明）。
  - 适用边界：这些字段和语义只对应 TRL v1.16.0 的异步 GRPO trainer；它们是训练器观测指标，不是 PPO/GRPO 理论定义或跨框架通用 API。

## 2026-10-30 补充来源

### On-policy 与 off-policy 偏好候选

- Sun et al., “Is On-Policy Data Always the Best Choice for Direct Preference Optimization-Based LM Alignment?”, ICLR 2026，[会议论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0dc678bae2ad5d8fdfcd3d0c9f75705d-Paper-Conference.pdf)。
  - 版本与阅读范围：ICLR 2026 正式会议论文；实际阅读摘要、§1、§3–4、结论及附录 C.1、C.3–C.4。
  - 来源支持：论文比较静态/off-policy 与训练期间从当前 policy 采样的 on-policy 偏好候选，在五种模型（Llama、Zephyr、Phi-2、Qwen、Pythia）和 DPO、SLiC-HF 两种方法上的实验显示，哪类候选更有效依模型而异。摘要举例报告 Llama-3 上 on-policy 效果可达静态数据的约 3 倍，而 Zephyr 上约为 0.4 倍。作者提出阶段假设：偏好注入阶段需要候选多样性，偏好微调阶段更偏向候选质量。
  - 数据与评估边界：论文从 UltraFeedback 取 prompt/候选，并用 PairRM 统一给静态与 on-policy 候选标偏好；PairRM 是偏好模型代理，不是人类标注。主要指令跟随结果使用 AlpacaEval 2.0 的长度控制胜率，其评估器为 GPT-4-1106-preview，测试集 805 个样本。不能把胜率当成人类直接标注，也不能把阶段假设写成固定训练日程或普适定律；采样策略、PairRM、基模型和评估协议均限制结论迁移。
  - 证据等级：source-backed（正式会议论文所述实验与作者解释）。

### DQO 的质量与语义多样性联合目标

- Chen et al., “Post-training Large Language Models for Diverse High-Quality Responses,” ICLR 2026，[会议论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/cfeb1fa78c5fb06be27c0980aa8ee581-Paper-Conference.pdf)。
  - 版本与阅读范围：ICLR 2026 正式会议论文；实际阅读摘要、§3.1–3.3、§4.2、图 1 与相关实验设置。
  - 来源支持：DQO 对同一 prompt 采样一组回答，将回答映射至 embedding，并以核相似矩阵行列式表示这组向量张成的体积；目标将 reward、`log det` 多样性项和 KL 项联合起来。论文在 reasoning、summarization、story-writing 和 instruction-following 等任务上评估该方法。
  - 测量边界：图 1 的质量分是单次生成的 reward；多样性分是 10 个回答上多个指标的平均，回答 temperature 为 1.0。结果依赖所选 embedding、kernel、任务、采样和 reward；不能据此断言 RL 后训练必然降低多样性，或将 DQO 视为跨任务的通用修复。
  - 证据等级：source-backed（正式会议论文中的方法和所列实验）。

### RLVR 中错误标注与 verifier 噪声

- Zhu & Kang, “Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards,” arXiv:2603.16140v2，[论文 HTML](https://arxiv.org/html/2603.16140v2)。
  - 版本与阅读范围：arXiv v2，页面版本日期 2026-07-29；实际阅读摘要、§3.2–3.3、§4–6、限制讨论及实验附录相关设置。
  - 来源支持：作者将 RLVR 噪声区分为 ground-truth annotation 错误与等价 verifier 错误。在对一组先前被称为“错误”的合成数学标注进行 20 例随机审计时，作者发现其中 8 例实际正确；该比例描述这次小样本审计，不代表普遍标注错误率。经重验的数据实验中，作者报告真正错误的数学标注训练相对干净数据在所测推理 benchmark 低约 8–10%；BIRD/Text2SQL 中真实人工标注错误的训练设置低约 5–12% 准确率。论文也报告若干算法改进未消除其特定设置中的噪声损害。
  - 证据等级：source-backed（arXiv 预印本的具体审计和实验报告，尚非同行评审结论）。
  - 适用边界：数字依赖作者重验流程、模型、噪声构造、任务、训练预算和 benchmark，不能外推为所有 verifier 或所有 RLVR 训练的固定损失。它与 Shao et al. 的“Spurious Rewards”不是同一噪声定义：后者研究特定设置下与答案弱相关或随机分配的奖励可能如何被 GRPO 放大；本预印本研究被误认为错误的标注、实际错误标注和 verifier/annotation 质量。两者不能合并成“随机奖励有益”或“脏数据总会有害”的单一结论；“训练信号需按类型单独核验”是基于两篇论文问题定义与实验范围的综合推断。

### DeepSeek-V3.2 的公开后训练描述

- DeepSeek-AI, “DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models,” arXiv:2512.02556v1，[技术报告](https://arxiv.org/html/2512.02556v1)；[DeepSeek-V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/README.md)。
  - 版本与阅读范围：报告 v1 页面日期 2025-12-02，实际阅读 §3、§3.1–3.2；模型卡 `main` 分支于 2026-10-30 访问。
  - 来源支持：报告公开描述了 specialist distillation 后的 mixed RL，以及其 GRPO 扩展中的重要性采样 KL 估计、off-policy sequence mask、Keep Routing 和 Keep Sampling Mask，并讨论面向工具使用的 thinking/context 管理和合成 agent 任务。
  - 证据等级：source-backed（厂商报告和模型卡的公开说明）。
  - 适用边界：这些是特定模型版本的公开训练与模型卡描述，不是完整可复现训练配方，也不支持推断未披露的超参数、数据或运行时行为。当前模型卡明确说明 V3.2-Speciale 面向深度推理且不支持 tool calling；该限制只适用于该变体，不可外推到 DeepSeek-V3.2 一般模型或其他工具运行时。

## 生命周期

三阶段 PPO-style RLHF、pairwise RM 目标、PPO clipping 与 DPO 关系，以及偏好候选来源、DQO、RLVR 标注噪声和 DeepSeek-V3.2 公开方案：active。反馈质量、候选生成分布、RM 校准、PPO 采样/优势估计、KL 归约、verifier 与偏好冲突依模型、任务、标注协议和实现分别核验。
