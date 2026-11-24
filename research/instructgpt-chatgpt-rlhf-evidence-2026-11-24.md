# InstructGPT、ChatGPT 与 RLHF 研究记录

## 研究问题与范围

核对 InstructGPT 如何使用指令示范、偏好比较和策略优化改变 GPT-3 的回答行为；确定论文中人类偏好结果的比较条件和限制；梳理 ChatGPT 2022 年首次发布时对基座模型、训练流程和对话交互的公开说明。阅读日期：2026-11-24。该记录用于第二十一册第五十三章，范围限于历史公开证据，不试图还原未公开的训练数据、模型权重、服务系统或当前 ChatGPT 版本。

检索词包括：InstructGPT paper 85 3 API prompt distribution human preference limitations; ChatGPT launch November 30 2022 GPT-3.5 RLHF trainers both user assistant; language model preference fine tuning Ziegler 2019; human feedback summarization reward model PPO Stiennon 2020; instruction tuning versus model architecture.

## 主要发现

1. **InstructGPT 的三阶段流程分别使用不同信号。** Ouyang 等人的论文先用标注者撰写的示范回答监督微调 GPT-3，再以同一 prompt 下多个模型回答的人类排序训练奖励模型，之后使用 PPO 优化策略。论文明确将基座描述为 GPT-3 架构；该实验显示训练数据与目标可以显著改变行为，不是新的 Transformer block 提案。
2. **偏好胜率绑定测试分布和标注协议。** 论文在 API prompt 分布上比较标注者偏好，报告 $175\mathrm{B}$ InstructGPT 相较 $175\mathrm{B}$ GPT-3 的人类偏好胜率为 $85\pm3\%$；当基线换成 few-shot GPT-3 时，InstructGPT 的偏好胜率为 $71\pm4\%$。论文也报告 $1.3\mathrm{B}$ InstructGPT 在其 API prompt 评价中可被偏好于 $175\mathrm{B}$ GPT-3。以上结果属于成对回答偏好，不能当作跨任务能力排名或“较小模型普遍胜过较大模型”的证据。
3. **论文明确限制了“人类偏好”的含义。** 训练反馈来自特定标注者和研究人员群体表达的偏好，不能概括为全体人类价值。论文称模型仍会犯错并可能产生事实错误；部分公开 NLP 数据集出现回归，且偏见评测没有显示普遍消除。偏好优化的结果不能替代事实核验、任务测试或单独的安全评估。
4. **RLHF 在 InstructGPT 前已有研究。** Ziegler 等人 2019 年研究语言模型的人类偏好微调；Stiennon 等人 2020 年将人类比较、奖励模型和强化学习用于摘要。InstructGPT 的历史贡献是把这类路线系统用于更广泛的指令遵循，并公开了 GPT-3 基座上的行为比较，而不是发明所有 RLHF 组件。
5. **ChatGPT 发布说明只支持发布期的公开描述。** OpenAI 2022 年 11 月的说明称当时 ChatGPT 从 GPT-3.5 微调，采用与 InstructGPT 类似但数据收集略有差异的方法；示范训练者分别扮演用户和助手，后续包括回答排序、奖励模型和 PPO。页面展示了可追问、承认错误、拒绝不当请求等对话交互，也列出生成错误、措辞敏感和偏好较长回答等限制。页面注明产品已演进，不能用它还原当前实现。
6. **教学组织上需把行为、结构和产品分层。** InstructGPT 论文直接支持基座模型经后训练后改变回答行为；ChatGPT 发布页支持对话产品功能和发布期训练披露。两种资料都不足以从产品体验推断隐藏的模型层、推理系统或当前服务策略。SFT/RM/PPO 配方回链第五册第十一章，后续 reasoning RL/RLVR 由第二十一册第五十八章承接。

## 来源与证据边界

| 来源 | 类型与阅读位置 | 支持的内容 | 不支持的外推 |
| --- | --- | --- | --- |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)（Ouyang 等，NeurIPS 2022） | 原始论文；摘要、§§2–3、§§5–6、limitations 与 Appendix | GPT-3 基座的 SFT/RM/PPO 流程、API prompt 人类偏好比较、模型行为变化及已声明限制 | 跨分布通用能力排名、普遍价值对齐、事实正确性保证或新架构主张 |
| [Introducing ChatGPT](https://openai.com/index/chatgpt/)（OpenAI，2022-11-30） | 发布期官方说明；training method、dialogue format 与 limitations 部分 | 初始 ChatGPT 的 GPT-3.5 微调披露、标注者角色、偏好/RLHF流程、多轮交互及发布期局限 | 当前 ChatGPT 的完整架构、训练数据或服务系统 |
| [Aligning language models to follow instructions](https://openai.com/index/instruction-following/)（OpenAI，2022-01-27） | 官方项目说明；方法和 API 行为介绍 | InstructGPT 的研究背景及指令遵循问题的公开说明 | 独立于原始论文的重复验证或跨任务外推 |
| [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)（Ziegler 等，2019） | 原始论文；摘要与实验流程 | InstructGPT 前已有语言模型偏好微调研究 | 与后续 InstructGPT 在规模、数据或目标上完全相同 |
| [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)（Stiennon 等，2020） | 原始论文；摘要、方法和评估 | 摘要任务中以人类比较训练奖励模型并优化策略的先行工作 | 所有指令任务上都具有相同效果 |
| [Stanford CS224N: Instruction Tuning and RLHF](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture10-instruction-tunining-rlhf.pdf)（2025） | 课程讲义；SFT、RLHF、InstructGPT/ChatGPT 概览 | 教学顺序与概念对照 | 不能替代原始论文对数字和适用范围的支持 |
| [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)（Chip Huyen，2023-05-02） | 教学文章；RLHF 流程和对话产品区别 | 面向读者的流程解释与概念对照 | 不能替代论文或官方发布资料作为历史证据 |

## 章节设计边界

- 第二十一册第五十三章讲后训练如何改变 GPT 模型的使用行为，以及 InstructGPT/ChatGPT 的历史位置；内部 Transformer block 结构仍由第十六、十七章讲解。
- 第五册第十一章讲 SFT、奖励模型、PPO、DPO 等算法与训练流程；本章仅保留理解历史所需的阶段接口。
- 第二十一册第五十八章讲 DeepSeek reasoning RL/RLVR。本章不把可验证推理奖励回写成 2022 年 InstructGPT 的方法。
- ChatGPT 官方发布说明按 2022 年页面自身声明限定为初始版本资料，不扩展为当前产品说明。

## 未解决边界

公开论文和发布说明没有披露足以重建所有训练数据、标注者构成、优化细节和线上系统的资料。API prompt 人类偏好测量的是特定比较协议下的偏好，不足以判断所有用户、领域和风险等级中的可靠性。产品体验同时受模型、提示模板、上下文、政策和外部系统影响，不能单独识别模型内部架构或某一训练阶段的因果贡献。
