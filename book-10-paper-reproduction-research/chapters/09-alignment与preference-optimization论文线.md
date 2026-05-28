# 第九章：Alignment 与 Preference Optimization 论文线

Alignment 与 preference optimization 是大模型论文中最容易被简化的一条线。很多候选人会把它讲成：先 SFT，再 RLHF，再 DPO。但真实研究脉络更复杂：它不是单纯换 loss，而是在回答一个长期问题：如何让一个会预测文本的 base model，变成一个更有用、更诚实、更安全、更符合人类偏好的 assistant model。

本章目标不是把每个算法都推公式，而是训练读论文的能力：当论文提出一种对齐或偏好优化方法时，你要能判断它解决的是数据成本、reward model 误差、RL 训练不稳定、reference model 约束、偏好信号形式、还是安全监督瓶颈。

本章重点：InstructGPT、Constitutional AI、DPO、IPO、KTO、ORPO、RLAIF。

---

## 1. 为什么需要 alignment？

预训练语言模型的目标通常是预测下一个 token。这个目标能让模型学到语言、知识、代码模式和推理模式，但它不直接等价于“做一个好助手”。

一个 base model 可能会：

1. 不按用户指令回答。
2. 延续不合适的上下文。
3. 在不知道时编造答案。
4. 对危险请求缺少边界。
5. 输出啰嗦、无礼、格式混乱或不稳定的回答。
6. 模仿互联网文本中的偏见、攻击性或低质量模式。

Alignment 的核心问题是：模型学到的预测目标，和人类希望它表现出的行为目标，并不完全一致。

所以，alignment 论文通常不是在问“模型会不会生成流畅文本”，而是在问：如何把模型行为从“会续写”推向“会按人类意图可靠地完成任务”。

---

## 2. 小白例子：会背书的人不一定会当客服

可以把 base model 想象成一个读过海量资料的人。它知道很多内容，也能模仿很多风格，但如果直接让它当客服，它可能会出问题。

用户问一个简单问题，它可能长篇大论。

用户要求固定格式，它可能忽略格式。

用户问危险操作，它可能照着网上文本继续写。

用户问它不知道的事实，它可能编一个听起来合理的答案。

SFT 像给它看优秀客服示范，让它学会基本服务格式。RLHF 像让用户比较多个客服回答，告诉它哪个更好。DPO 等偏好优化方法则尝试直接利用“哪个回答更好”的数据，让模型更偏向好回答、更远离差回答。

这个比喻说明：alignment 不是让模型知道更多知识，而是改变模型在面对用户时的行为选择。

---

## 3. 读 alignment 论文要抓哪条主线？

读 alignment 论文时，建议先拆成六个问题。

第一，它优化什么行为？

是 helpfulness、honesty、harmlessness、instruction following、format following、reasoning quality，还是某种综合偏好？

第二，偏好信号来自哪里？

可能来自人类标注、AI feedback、规则奖励、可验证任务、用户日志、红队数据或自动 judge。

第三，数据形式是什么？

是 demonstrations、rankings、chosen/rejected pair、binary desirable/undesirable label，还是 scalar reward？

第四，优化方式是什么？

是 SFT、reward modeling + RL、direct preference optimization、online RL，还是混合方法？

第五，如何防止模型漂移？

常见方式包括 reference model、KL penalty、保留 SFT loss、约束采样分布和回归评估。

第六，评估是否可信？

要看人工偏好、自动评测、安全评测、长度偏差控制、数据污染、OOD 泛化和真实用户分布。

这六个问题比背算法名字更重要。

---

## 4. InstructGPT：现代 RLHF 路线的代表

InstructGPT 论文的历史意义在于，它系统展示了用人类反馈把 GPT-3 类模型调整成更符合用户意图的助手模型。

它要解决的问题很明确：更大的语言模型不一定更会遵循用户意图。原始 GPT-3 虽然强大，但可能不够 helpful、truthful、harmless。论文展示了一个关键结果：经过人类反馈训练的 1.3B InstructGPT，在人类偏好评估中可以优于 175B GPT-3 输出，尽管参数少很多。

这说明 alignment 不是 scaling 的自然副产品。模型更大不自动意味着更符合用户偏好。

InstructGPT 的典型流程可以概括为：

```text
收集指令和示范回答
-> SFT 训练初始助手模型
-> 对同一 prompt 生成多个回答
-> 人类标注回答排序
-> 训练 reward model
-> 用 PPO 优化策略，同时约束不要偏离初始模型太远
```

这条路线奠定了后续 RLHF 的基本结构。

---

## 5. InstructGPT 论文该怎么读？

读 InstructGPT 时不要只记 “SFT + RM + PPO”。更重要的是看它的证据链。

第一，问题定义。

论文关注的是 user intent alignment，而不是单个 NLP benchmark 刷分。

第二，数据来源。

它使用标注员写的 prompts、API 用户分布中的 prompts、示范回答和排序数据。也就是说，它把训练目标从互联网文本分布转向真实用户请求分布。

第三，评估方式。

论文强调 human preference evaluation，同时也检查 truthfulness、toxicity 和公开 NLP 任务上的能力退化。

第四，关键发现。

小参数量的 aligned model 可以在人类偏好上超过大得多的 base model。这是对“参数越大越好”的重要补充。

第五，局限。

InstructGPT 仍会犯错，reward model 不是完美人类价值函数，PPO 训练复杂且敏感，标注员偏好也不等于所有用户偏好。

面试中能讲出这些点，比只说“RLHF 用 PPO”更有深度。

---

## 6. RLHF 的核心机制和问题

RLHF 的核心思想是：用人类偏好训练一个 reward model，再用强化学习优化语言模型，使模型输出获得更高 reward，同时不要偏离参考模型太远。

它的优点是：

1. 能直接优化人类偏好信号。
2. 能比较多个回答的质量差异。
3. 可以在 SFT 基础上继续提升 helpfulness 和安全行为。
4. 可以结合在线采样，让模型探索新的回答。

但 RLHF 也有明显缺点：

1. 需要训练 reward model。
2. PPO 等 RL 训练复杂，超参数敏感。
3. Reward model 可能被过度优化。
4. 训练成本高，工程链路长。
5. 偏好数据噪声会被放大。
6. KL 约束太强学不动，太弱会模型漂移。

后来的 DPO、IPO、KTO、ORPO 等方法，大多是在回应这些问题：能不能不用完整 RL loop？能不能直接用偏好对训练？能不能减少 reference model 或 reward model 依赖？能不能适配更弱、更便宜的反馈信号？

---

## 7. Constitutional AI 与 RLAIF：从逐条人类反馈到原则监督

Constitutional AI 关注另一个瓶颈：人类偏好标注成本高，而且安全有害样本会给标注员带来负担。

它的思路不是让 AI 完全替代人类，而是让人类提供一组原则或规则，也就是 constitution，然后让模型根据这些原则进行自我批评、自我修正和偏好判断。

公开论文中的高层流程可以理解为两阶段。

第一阶段是监督学习式自我修正：

```text
模型生成初始回答
-> 根据 constitution 生成 critique
-> 根据 critique 生成 revised answer
-> 用 revised answer 做监督微调
```

第二阶段是 AI feedback 偏好优化：

```text
模型对同一输入生成多个回答
-> AI 根据 constitution 判断哪个更好
-> 用 AI preference 训练 preference model
-> 用 RL 优化模型
```

这类路线常被称为 RLAIF，也就是 Reinforcement Learning from AI Feedback。

Constitutional AI 的贡献是：把一部分监督从“人类逐样本判断”变成“人类制定原则，AI 按原则扩展监督”。它尤其适合讨论 scalable oversight，因为随着模型能力提升，人类逐条评价所有复杂输出会越来越困难。

---

## 8. Constitutional AI 解决了什么，又没解决什么？

它解决的问题包括：

1. 降低大量逐样本人类标注成本。
2. 减少人类直接接触有害内容。
3. 让安全原则更显式、更可审计。
4. 通过 critique 和 revision 训练模型解释和修正行为。
5. 训练更不机械回避的 harmless assistant。

但它没有消除 alignment 难题。

原因是：

1. Constitution 本身可能不完整。
2. 原则之间可能冲突。
3. AI judge 可能继承模型偏差。
4. 模型可能学会表面迎合原则文本。
5. 原则到具体判断之间仍然需要解释，而解释过程可能出错。

所以，Constitutional AI 不是“AI 自己监督自己就安全了”。更准确的说法是：人类通过原则提供监督框架，AI 帮助扩展和执行一部分监督过程，最终仍需要人工审计、安全评估、红队和上线治理。

---

## 9. DPO：为什么它影响这么大？

DPO 的背景是：RLHF 有效，但复杂。

传统 RLHF 需要训练 reward model，再用 PPO 等 RL 方法优化语言模型。这个流程有很多工程难点：采样、reward 估计、KL 控制、value model、训练稳定性、超参数调节。

DPO 的关键主张是：在标准 RLHF 目标下，可以通过重新参数化 reward，把偏好优化变成一个直接的分类式目标。也就是说，不需要显式训练 reward model，也不需要 PPO loop，直接用 chosen/rejected 偏好对更新语言模型。

非常直观地说，DPO 想让模型做到：

```text
相对于 reference model，提高 chosen answer 的相对概率，降低 rejected answer 的相对概率。
```

DPO 的影响力来自三个特点：

1. 工程简单。
2. 训练稳定。
3. 容易在开源模型和离线偏好数据上复现。

这让大量团队可以不用完整 RLHF 基础设施，也能做偏好对齐实验。

---

## 10. DPO 论文该怎么读？

读 DPO 时，重点不是背 loss，而是理解它的假设和边界。

第一，它仍然依赖偏好数据。

chosen/rejected 质量差，DPO 会直接学偏。它简化的是优化流程，不是降低数据质量要求。

第二，它依赖 reference model 的相对约束。

reference model 帮助控制模型不要过度偏离原有能力。reference 太弱、太强或和训练模型差异过大，都会影响效果。

第三，它是离线偏好优化。

DPO 通常不在训练过程中持续采样新回答并获取新反馈，因此它对已有偏好数据分布敏感。如果训练后模型进入数据覆盖不到的新区域，偏好优化可能不稳定。

第四，它不消除 reward hacking 或目标错配。

DPO 没有显式 reward model，但它仍然在优化偏好数据隐含的目标。如果偏好数据偏向长回答、讨好式回答、过度拒答或模板化风格，模型仍会学到这些偏差。

所以 DPO 的正确定位是：它是 RLHF 复杂流程的一种重要简化，不是 alignment 的最终答案。

---

## 11. IPO：从理论上重新审视偏好学习

IPO 通常出现在对 DPO 和 RLHF 理论基础的进一步讨论中。相关工作指出，常见 RLHF 路线依赖一些近似，例如把 pairwise preference 转成 pointwise reward，或者希望 reward model 能泛化到策略采样的新分布。

IPO 所在的论文线试图更一般地理解“从人类偏好学习”到底在优化什么。它把 RLHF、DPO 等方法放到统一框架中分析，并指出不同目标函数可能带来不同偏差。

对面试来说，你不一定要推完整理论，但要知道它出现的原因：DPO 让偏好优化变简单后，研究者开始进一步问：这个直接优化目标是否真的等价于我们想要的 preference learning？它有没有隐藏近似？在什么情况下会失败？有没有更直接处理 pairwise preference 的目标？

IPO 的价值在于提醒我们：

1. 偏好优化不是只有 DPO 一种形式。
2. 不同 loss 隐含不同偏好建模假设。
3. 理论上看似接近的方法，优化行为可能不同。
4. 经验效果必须结合数据分布和评估指标判断。

读 IPO 类论文时，重点是看它指出了前人哪个假设不够严谨，而不是只记一个新缩写。

---

## 12. KTO：当你没有成对偏好数据时怎么办？

DPO 通常需要 chosen/rejected 成对偏好数据。但现实中，很多数据并不是天然成对的。

你可能只有：

1. 这个回答是好的。
2. 这个回答是不好的。
3. 用户点赞或点踩。
4. 审核通过或不通过。
5. 某条输出是否符合要求。

KTO 的动机就是：能不能用更简单的二元反馈信号对齐模型，而不强制要求每个 prompt 都有成对比较？

KTO 借鉴 prospect theory 的思想，把模型对 desirable 和 undesirable 输出的优化设计成更贴近人类偏好感知的目标。论文强调它只需要输出是否 desirable 的二元信号，也能在一些规模上达到或超过偏好对方法。

KTO 的意义不是说它一定全面替代 DPO，而是扩展了 preference optimization 的数据形态：从“成对比较”扩展到“单样本好坏反馈”。

读 KTO 时要问：

1. 它对数据格式有什么要求？
2. 它如何处理 desirable 和 undesirable 样本不平衡？
3. 它和 DPO 相比减少了什么标注成本？
4. 它的 inductive bias 是否适合当前任务？
5. 它在什么规模和数据集上验证？

---

## 13. ORPO：把 SFT 和偏好优化合并

ORPO 的背景是另一个工程问题：很多偏好优化方法需要先 SFT，再做额外偏好优化阶段，还可能需要 reference model。这会增加训练流程复杂度和显存成本。

ORPO 提出一种 reference model-free 的单阶段偏好优化思路。它强调 SFT 本身对收敛很重要，同时在 SFT 过程中加入 odds ratio 形式的偏好惩罚，让模型提高 favored response 的概率，同时压低 disfavored response 的相对倾向。

可以粗略理解为：

```text
保留 SFT 对 chosen answer 的学习
同时加入偏好项，惩罚 rejected style
不再额外维护 reference model
```

ORPO 的吸引力在于流程简单：把指令微调和偏好对齐合在一起。但它的结论仍要结合具体数据、模型规模和评估方式看。reference model-free 不等于没有约束风险，只是约束方式变了。

读 ORPO 时，要关注：

1. 它是否真的减少训练阶段？
2. 它和标准 SFT、DPO 的公平比较如何做？
3. 它对数据质量是否更敏感？
4. 没有 reference model 时如何控制模型漂移？
5. 它在不同模型规模上是否稳定？

---

## 14. Preference optimization 方法对比

可以用下面这张文字表建立整体关系。

```text
SFT:
  数据：prompt -> ideal answer
  目标：模仿示范回答
  优点：简单稳定
  局限：不直接比较多个回答好坏

RLHF/PPO:
  数据：偏好排序 + reward model
  目标：最大化 reward，同时 KL 约束
  优点：可在线采样优化偏好
  局限：复杂、昂贵、训练敏感

DPO:
  数据：chosen/rejected pair
  目标：直接提高 chosen 相对 rejected 的偏好
  优点：简单、稳定、无需显式 reward model
  局限：依赖离线偏好数据和 reference model

IPO:
  数据：pairwise preference
  目标：从更一般理论框架理解偏好优化
  优点：揭示 DPO/RLHF 的假设和潜在问题
  局限：工程流行度和具体收益需结合场景

KTO:
  数据：desirable/undesirable 二元反馈
  目标：基于人类效用偏置优化输出
  优点：不强制要求成对偏好
  局限：依赖好坏标签质量和目标假设

ORPO:
  数据：偏好数据，通常围绕 favored/disfavored response
  目标：把 SFT 和偏好优化合并
  优点：流程简单，无需额外 reference model
  局限：控制漂移和公平比较需要仔细验证
```

这张表不是为了背，而是为了在面试中快速定位每个方法解决的瓶颈。

---

## 15. 偏好数据比算法名字更重要

Preference optimization 论文经常让人把注意力放在 loss 上，但真实效果很大程度由偏好数据决定。

偏好数据至少要检查：

1. prompt 是否覆盖真实用户分布。
2. chosen 是否真的更好，而不是只是更长。
3. rejected 是否足够有挑战性。
4. 标注标准是否一致。
5. 是否区分 helpfulness、truthfulness、harmlessness。
6. 是否有多语言、长上下文、代码、数学、工具调用等场景。
7. 是否存在过度拒答或安全边界不清的问题。
8. 是否存在模型自己生成数据带来的偏差放大。

如果 chosen/rejected 构造随意，DPO、IPO、ORPO 都可能学到错误偏好。如果二元好坏标签很粗糙，KTO 也可能被噪声带偏。

算法简化不等于数据可以粗糙。

---

## 16. 评估 preference optimization 要注意什么？

偏好优化评估比普通 benchmark 更难，因为很多指标容易被“表面优化”。

常见评估包括：

1. 人工 win rate。
2. LLM-as-a-Judge win rate。
3. MT-Bench、AlpacaEval、IFEval 等自动或半自动评测。
4. 安全评估。
5. Truthfulness 和 hallucination 评估。
6. 长度控制后的偏好评估。
7. OOD prompt 评估。
8. Regression suite。

必须特别警惕长度偏差。很多 judge 和人类标注都会偏好更长、更有结构、更自信的回答。模型可能通过变啰嗦提高 win rate，而不是实质提升质量。

也要警惕 over-refusal。安全偏好数据如果设计不好，模型可能把大量正常请求也拒绝掉，看起来更安全，实际可用性下降。

好的评估应该同时看：helpfulness 是否提升，truthfulness 是否提升，harmfulness 是否下降，over-refusal 是否可控，原有能力是否退化。

---

## 17. Preference optimization 和 safety 的关系

偏好优化可以用于 safety，但它本身不是 safety 的全部。

DPO 或 RLHF 可以让模型更倾向安全回答，但安全系统还需要：

1. 明确 policy。
2. 安全数据和边界样本。
3. 红队发现失败模式。
4. 安全评估和回归测试。
5. Prompt injection、隐私、危险能力等专项测试。
6. 上线后的监控和事故响应。
7. 必要的系统层权限控制。

不要把“做了 DPO”当成“模型安全了”。DPO 优化的是偏好数据隐含目标，如果偏好数据没有覆盖真实风险，模型仍会失败。

Constitutional AI 和 RLAIF 也类似。它们能扩展监督，但原则本身、AI judge 和评估闭环都需要持续审查。

---

## 18. 如何复现 DPO 或偏好优化论文？

资源有限时，可以做一个小规模复现实验。

推荐流程：

1. 选择一个已 SFT 的小模型。
2. 准备一个公开偏好数据子集。
3. 保持 tokenizer、prompt template 和数据截断规则一致。
4. 实现或调用 DPO loss。
5. 对比 SFT baseline 和 DPO model。
6. 用人工抽样、LLM judge 和少量任务评估 win rate。
7. 控制回答长度，避免长度偏差。
8. 检查安全、事实性和格式遵循是否退化。
9. 做 ablation：不同 beta、不同数据量、不同 rejected 质量。

一个好的复现不是只跑出 loss 下降，而是回答：偏好优化是否真的改变了行为？改变的是质量、长度、语气、安全边界，还是只是更会迎合 judge？

---

## 19. 常见错误读法

错误一：把 alignment 等同于安全拒答。

纠正：alignment 包括 helpfulness、honesty、harmlessness、指令遵循、格式控制和偏好一致性。安全只是其中一部分。

错误二：把 RLHF 简化成 PPO。

纠正：PPO 是一种常见优化方法，RLHF 的核心是从人类反馈学习 reward 或偏好目标，再优化模型行为。

错误三：认为 DPO 没有 reward model，所以没有 reward hacking。

纠正：DPO 仍优化偏好数据隐含目标，数据偏差仍会带来目标错配。

错误四：认为 DPO 一定优于 RLHF。

纠正：DPO 工程简单、离线稳定，但在线 RL 和 reward-based 方法在某些需要探索、可验证奖励或复杂任务中仍有价值。

错误五：只比较算法，不比较数据。

纠正：偏好数据质量、prompt 分布、chosen/rejected 构造和评估方式往往决定结果。

错误六：认为 RLAIF 不需要人类。

纠正：人类仍然提供原则、系统设计、模型选择、审计和最终治理。

---

## 20. 面试中如何讲这条论文线？

一个成熟回答可以这样组织：

1. 先说问题：预训练目标是 next-token prediction，不等于用户意图对齐。
2. 再说 InstructGPT：通过 SFT、reward model 和 PPO，把 GPT-3 调成更符合人类偏好的助手，证明 alignment 可以让小模型在人类偏好上超过大 base model。
3. 接着说 RLHF 的代价：reward model、PPO、KL、采样和训练稳定性都复杂。
4. 再说 DPO：通过直接偏好优化简化 RLHF，不显式训练 reward model，也不用 PPO loop。
5. 然后补充后续方法：IPO 从理论上审视偏好学习假设，KTO 支持二元好坏反馈，ORPO 尝试合并 SFT 和偏好优化并去掉 reference model。
6. 最后说 Constitutional AI/RLAIF：用原则和 AI feedback 扩展监督，缓解人工标注成本和有害样本标注负担，但仍需要人类治理和评估闭环。

这个回答能体现你理解的是研究脉络，而不是算法缩写表。

---

## 21. 典型面试题

### 问题 1：InstructGPT 的核心贡献是什么？

参考回答：

InstructGPT 的核心贡献是系统展示了如何用人类反馈把 GPT-3 类 base model 对齐到用户意图。它通过 SFT 学习示范行为，通过人类排序训练 reward model，再用 PPO 优化模型，同时约束模型不要偏离太远。重要发现是，较小的 InstructGPT 在人类偏好上可以超过更大的原始 GPT-3，说明 alignment 不是单纯 scaling 能自动解决的问题。

### 问题 2：DPO 相比 RLHF/PPO 解决了什么问题？

参考回答：

DPO 主要解决 RLHF/PPO 工程复杂和训练不稳定的问题。传统 RLHF 要训练 reward model，再用 PPO 进行采样和策略优化，还要处理 KL、value model 和超参数。DPO 通过重新参数化偏好目标，把偏好优化变成直接的分类式训练，使用 chosen/rejected 数据提高好回答相对坏回答的概率，因此更简单、稳定、便宜。但它仍依赖高质量偏好数据和 reference model，不消除目标错配。

### 问题 3：DPO 是否比 RLHF 更安全？

参考回答：

不能简单说更安全。DPO 简化了优化流程，但安全性取决于偏好数据、policy、评估和上线防护。如果偏好数据覆盖了安全边界，DPO 可以提升安全行为；如果数据有偏或只鼓励拒答，模型可能过度拒答或学到错误边界。DPO 没有显式 reward model，但仍优化偏好数据隐含目标，所以仍可能有目标错配和偏好过优化问题。

### 问题 4：Constitutional AI 和 RLHF 的区别是什么？

参考回答：

RLHF 通常依赖人类对模型输出的逐样本偏好标注。Constitutional AI 让人类先提供一组原则，再让模型根据原则生成 critique、revision 和 AI preference，用于监督学习和 RLAIF。它减少了大量逐样本人类标注，尤其降低有害内容标注负担，并让原则更显式。但它仍依赖人类制定原则、审计和评估，AI feedback 也可能继承模型偏差。

### 问题 5：KTO 和 DPO 的数据要求有什么不同？

参考回答：

DPO 通常需要同一个 prompt 下的 chosen/rejected 成对偏好数据。KTO 试图使用更简单的二元反馈，也就是某个输出是 desirable 还是 undesirable，不强制要求成对比较。因此 KTO 适合只有点赞、点踩、通过、拒绝这类单样本反馈的场景。但它仍依赖标签质量和目标假设，二元反馈噪声大时也会学偏。

### 问题 6：ORPO 的核心思路是什么？

参考回答：

ORPO 尝试把 SFT 和偏好优化合并成单阶段训练，并去掉额外 reference model。它在学习 favored response 的同时，用 odds ratio 形式惩罚 disfavored response，让模型在 SFT 过程中完成偏好对齐。它的优势是流程简单、成本较低，但没有 reference model 不代表没有漂移风险，仍要通过评估验证能力、安全和泛化。

---

## 22. 本章小结

Alignment 与 preference optimization 的主线是：预训练模型学会了语言建模，但不自动学会如何当一个符合人类意图的助手。

InstructGPT 证明了人类反馈可以显著改善用户偏好表现。Constitutional AI 和 RLAIF 试图用原则和 AI feedback 扩展监督。DPO 把复杂 RLHF 流程简化为直接偏好优化。IPO、KTO、ORPO 等后续方法则分别从理论假设、数据形式和工程流程上继续改进。

读这条论文线时，要记住三句话。

第一，alignment 不是 scaling 的自动结果。

第二，preference optimization 简化的是优化流程，不是消除数据和目标问题。

第三，真正可靠的对齐需要算法、数据、评估、安全策略和上线治理共同形成闭环。
