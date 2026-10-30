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
  - 来源支持：在偏好模型与 KL-regularized reward maximization 等假设下，DPO 可将目标重写为使用偏好 pair 的分类损失；固定数据集优化时不显式训练独立 RM，也不需要 PPO rollout 循环。原方法仍使用 reference policy 定义隐式 reward，并说明可重用由先前 policy 采样的离线偏好数据。
  - 证据等级：source-backed（理论重写和论文指定的情感控制、摘要、对话任务实验）。
  - 适用边界：结果限于论文的偏好假设、reference 初始化、数据和任务；“DPO 无需显式 RM/PPO rollout”描述的是算法训练阶段，不表示 preference 数据从未经模型采样，也不证明 DPO 在所有任务上优于 PPO。

## RLVR 中的 spurious reward 观察

- Shao et al., “Spurious Rewards: Rethinking Training Signals in RLVR,” ICML 2026, PMLR 306, pp. 110167–110207，[会议论文页面](https://proceedings.mlr.press/v306/shao26d.html)。
  - 实际阅读范围：PMLR 官方会议论文摘要。
  - 来源支持：作者报告，在指定 Qwen2.5-Math-7B/GRPO 设置中，随机奖励在 MATH-500 上带来 21.4 个百分点的绝对提升，接近 ground-truth reward 的 29.1 个百分点；论文将其与 GRPO clipping bias 和模型预训练中可被放大的高先验行为联系起来。摘要还指出，该现象具有模型依赖性，在 Llama 3 与 OLMo 2 上没有复现相同收益。
  - 证据等级：source-backed（正式会议论文摘要中的实验设置和作者解释）。
  - 适用边界：这是 RLVR 数学推理的特定模型、方法、奖励和评测结果，不是人类偏好 RM/PPO 的证据，也不表示随机奖励是训练建议；机制和迁移范围应按论文完整实验及目标模型验证。

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

## 生命周期

三阶段 PPO-style RLHF、pairwise RM 目标、PPO clipping 与 DPO 关系：active。反馈质量、候选生成分布、RM 校准、PPO 采样/优势估计、KL 归约和偏好冲突依模型、任务、标注协议和实现分别核验。
