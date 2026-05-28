# 第八章：Alignment 与 Safety 面试

Alignment 与 Safety 面试考的不是你会不会说“RLHF、DPO、reward hacking、jailbreak、red teaming、model card”这些词，而是你能不能把模型行为、训练目标、风险评估、系统防护和发布治理串成一个闭环。

很多候选人在这一关容易走偏：要么只讲 RLHF 和 DPO，不讲评估与上线门禁；要么只说“拒绝危险请求”，忽略 over-refusal 和正常 helpfulness；要么把 jailbreak 和 prompt injection 混在一起；要么把 safety 当成政策背诵，而不是一个可评估、可监控、可迭代的工程系统。

Alignment 与 Safety 面试的关键，是证明你理解：模型不是只要能力强就够了，还要在复杂用户、复杂工具、复杂部署环境和分布外场景里尽量可靠、诚实、安全、可控。

本章重点：alignment problem、RLHF/DPO 局限、reward hacking、jailbreak、prompt injection、red teaming、safety eval、over-refusal、隐私、危险能力评估、发布治理。

## 8.1 Safety 面试到底考什么

Safety 面试通常围绕八条主线：

1. 目标定义：什么是 alignment，什么是 safety，Helpful、Honest、Harmless 如何权衡。
2. 后训练方法：SFT、RLHF、DPO、Constitutional AI、RLAIF 在安全中的作用和局限。
3. 目标错配：reward hacking、Goodhart 定律、proxy objective、over-optimization。
4. 攻防问题：jailbreak、prompt injection、多轮诱导、工具越权和 RAG 污染。
5. 评估体系：safety eval、red teaming、harmful compliance、over-refusal、dangerous capability eval。
6. 模型行为：幻觉、不确定性表达、拒答边界、偏见、公平性和诚实性。
7. 隐私治理：memorization、PII、训练数据泄露、RAG 权限和日志脱敏。
8. 发布治理：model card、system card、发布门禁、灰度、监控、事故响应。

这些主题可以用一句话串起来：安全团队先定义风险 taxonomy 和可接受边界，再通过训练、评估、红队和系统防护降低风险，最后用发布门禁和线上监控保证模型在真实使用中持续符合安全目标。

面试官想看到的是你能不能回答这些问题：

1. RLHF 为什么不是 alignment 的最终解？
2. DPO 简化了什么，又没有解决什么？
3. Reward hacking 在 LLM 里可能如何表现？
4. Jailbreak 和 prompt injection 有什么区别？
5. 如何设计一个 safety eval？
6. 如何同时降低 harmful compliance 和 over-refusal？
7. Agent 工具调用如何做安全边界？
8. 如果 helpfulness 提升但 jailbreak 成功率也提升，能不能上线？

## 8.2 回答 Safety 题的七层框架

Safety 题不要直接背方法。推荐用七层框架：

1. 定义目标：要解决什么安全或对齐问题。
2. 定义风险 taxonomy：风险类别、严重度、用户场景和攻击面。
3. 分析失败模式：模型为什么会失败，训练、数据、系统、用户侧各有什么原因。
4. 给缓解方法：训练方法、系统防护、权限控制、人工审核。
5. 给评估方法：离线 eval、红队、人工复核、线上监控。
6. 给发布治理：门禁、灰度、回滚、日志、事故响应。
7. 讲 trade-off：安全与 helpfulness、隐私与 utility、透明度与滥用风险。

例如问“你如何提升一个模型的安全性”，不要只说：

```text
做安全 SFT，再用 RLHF，对 jailbreak 做红队。
```

更好的回答是：

```text
我会先定义安全目标和风险分类，比如有害内容、隐私泄露、jailbreak、prompt injection、工具滥用和高风险专业建议。训练上可以加入安全 SFT、偏好数据和拒答边界样本；系统上要有 policy layer、输入输出检测、工具权限和高风险操作确认。评估上要同时看 harmful compliance、over-refusal、正常 helpfulness、多轮 jailbreak 成功率和工具越权率。发布前设置门禁和灰度，发布后持续监控安全 bad case，并把红队样本沉淀为 regression suite。
```

这个回答有目标、方法、评估、治理和 trade-off。

## 8.3 Alignment 和 Safety 的区别

AI Safety 更关注模型或系统是否会造成伤害，例如有害内容、隐私泄露、越狱、工具误用、危险能力、偏见和部署事故。

Alignment 更关注模型实际优化的目标和行为是否符合人类真实意图、价值和边界，例如模型是否 helpful、honest、harmless，是否会 reward hacking，是否会在分布外场景里学错目标。

面试回答：

```text
Safety 更偏风险和伤害控制，Alignment 更偏目标一致性和行为一致性。Safety 可以看作 alignment 的重要目标之一，但 alignment 还包括诚实性、可控性、可解释性和目标泛化问题。一个模型可能在常见 safety eval 上表现不错，但仍可能在复杂工具、分布外任务或长期交互中暴露 alignment 问题。
```

常见追问：“安全是不是让模型多拒答？”

可以回答：

```text
不是。安全不是简单提高拒答率。真正的目标是该拒绝时拒绝，该帮助时帮助。只提高拒答率会造成 over-refusal，损害正常用户请求。好的 safety 评估必须同时看 harmful compliance、refusal accuracy、over-refusal 和 safe alternative quality。
```

## 8.4 Helpful、Honest、Harmless 的冲突

Helpful、Honest、Harmless 是面试中非常常见的三目标框架。

1. Helpful：模型能解决用户问题。
2. Honest：模型不编造、不假装知道，能表达不确定性。
3. Harmless：模型不帮助用户造成伤害。

三者经常冲突：

1. 用户请求危险帮助时，helpful 和 harmless 冲突。
2. 用户要求模型回答不知道的信息时，helpful 和 honest 冲突。
3. 某些真实信息本身高风险时，honest 和 harmless 冲突。
4. 过度安全会损害 helpfulness，过度迎合会损害 harmlessness 和 honesty。

面试回答：

```text
我不会把 helpful、honest、harmless 当成三个独立指标，而会看它们在具体场景中的权衡。比如普通教育性请求要尽量 helpful，高风险操作要优先 harmless，资料不足时要优先 honest。安全策略应该按风险等级和上下文处理，而不是一刀切拒答。
```

## 8.5 RLHF 的价值与根本局限

RLHF 的价值是把难以手写的目标函数，转化为人类偏好数据和 reward model，让模型更符合人类对回答质量、安全性和风格的期望。

典型流程：

1. 先做 SFT，让模型学会基础指令遵循。
2. 收集 prompt 下多个回答的人类偏好。
3. 训练 reward model，给回答打分。
4. 用 PPO 或类似方法优化 policy，同时用 KL 约束不要偏离 reference model 太远。

面试中不能只夸 RLHF，要讲清局限：

```text
RLHF 的根本局限在于 human preference 和 reward model 都是 proxy，不等于真实人类目标。偏好数据可能有标注员偏差、长度偏差、风格偏差和覆盖不足；reward model 可能被 policy 过度优化，导致 reward hacking；复杂任务里人类也未必能准确判断答案质量。因此 RLHF 是重要后训练方法，但不是 alignment 的最终解。
```

常见追问：“为什么需要 KL 约束？”

回答：

```text
KL 约束让 policy 不要为了追求 reward model 分数而偏离 SFT 或 reference model 太远。它可以缓解语言质量下降、reward hacking 和行为漂移，但不能完全解决 reward model 本身不可靠的问题。
```

## 8.6 DPO 简化了什么，没有解决什么

DPO 把带 KL 约束的偏好优化推导成直接在 chosen/rejected pair 上优化 policy 的损失，避免显式训练 reward model 和 PPO loop。

它的工程优势是：

1. 流程更简单。
2. 训练更稳定。
3. 不需要在线采样和单独 reward model。
4. 更容易复现和调参。

但 DPO 没有消除偏好数据的根本问题。

面试回答：

```text
DPO 简化的是 RLHF 的工程优化流程，而不是解决 alignment 的所有问题。它仍然依赖 chosen/rejected 偏好数据。如果偏好数据有噪声、长度偏差、安全边界不清或覆盖不足，DPO 仍可能学到表面偏好、过度拒答或能力退化。所以 DPO 要配合高质量数据、reference model、beta 调参和多维评估。
```

常见追问：“DPO 一定比 RLHF 更安全吗？”

回答：

```text
不一定。安全性取决于偏好数据、训练强度、reference 选择和评估门禁。DPO 训练更简单，不等于目标更正确。如果 preference pair 本身奖励了错误拒答或表面安全，DPO 也会放大这些偏差。
```

## 8.7 Reward Hacking 和 Goodhart 定律

Reward hacking 指模型找到提高 reward 或评估分数的捷径，但输出不符合真实人类目标。

LLM 中常见表现包括：

1. 为了高分输出冗长但空泛的答案。
2. 自信编造引用或事实。
3. 过度拒答，以避免安全风险。
4. 迎合 judge 偏好的格式，而不是解决真实问题。
5. 在 RAG 中给出看似有 citation 但证据不支持的回答。
6. 在代码任务中通过测试样例但隐藏边界错误。

面试回答：

```text
Reward hacking 本质上是 proxy objective 被过度优化。Goodhart 定律说，当一个指标变成目标，它就会失去作为好指标的可靠性。RLHF 中 reward model 只是人类偏好的近似，如果 policy 过度优化 reward model，就可能得到高 reward 但低真实质量的输出。
```

缓解方法包括：

1. 提高偏好数据质量。
2. 加 hard negatives 和 adversarial examples。
3. 控制优化强度和 KL。
4. 做 reward model 校准。
5. 引入 human eval 和 expert eval。
6. 用 holdout、红队和线上 bad case 监控 reward-human gap。

## 8.8 Jailbreak：绕过模型安全边界

Jailbreak 是用户通过角色扮演、诱导、多轮对话、编码、语言混合或对抗提示，让模型绕过安全策略，输出本应拒绝的内容。

防 jailbreak 不能只靠一句 system prompt。

面试回答：

```text
我会从训练、评估和系统三层防 jailbreak。训练上加入安全 SFT、偏好优化、多轮诱导样本、边界样本和安全替代回答。评估上覆盖角色扮演、跨语言、编码包装、多轮诱导和正常请求误拒。系统上加入输入输出分类器、policy layer、日志监控和红队回归。指标上同时看 attack success rate、harmful compliance、refusal accuracy、over-refusal 和 safe alternative quality。
```

为什么 jailbreak 难以完全防住？

1. 自然语言空间巨大。
2. 攻击可以跨语言、多轮、编码和组合变形。
3. 模型遵循指令的能力本身会被滥用。
4. 安全训练覆盖不了所有分布外场景。
5. 需要持续红队和 regression suite。

面试时注意：讲防御框架和评估，不要展示可操作攻击步骤。

## 8.9 Prompt Injection：LLM 应用的指令边界问题

Prompt injection 和 jailbreak 经常被混淆。

Jailbreak 主要是绕过模型自身安全边界。Prompt injection 主要是 LLM 应用混淆了可信指令和不可信数据，让外部文档、网页、邮件或工具返回中的恶意文本影响模型行为。

面试回答：

```text
Prompt injection 的核心问题是指令和数据边界混淆。防御不能只靠 prompt，而要做系统层隔离：明确 system、developer、user、external content、tool result 的信任等级；把 RAG 文档和网页标记为 untrusted content；工具调用做最小权限、schema 校验、高风险二次确认和审计日志；模型只能把外部内容当证据，不能执行外部内容里的指令。
```

Agent 和 RAG 中 prompt injection 风险更高，因为外部内容可能影响：

1. 检索结果选择。
2. 最终回答内容。
3. 工具调用决策。
4. 敏感数据泄露。
5. 越权操作。

高质量回答要强调：不可信文本不能直接控制高权限动作。

## 8.10 Safety Eval 怎么设计

Safety eval 不是找几个危险 prompt 测一下拒答率。

一个完整 safety eval 应包含：

1. 风险 taxonomy：有害内容、自伤、隐私、网络安全、偏见、工具滥用、prompt injection、危险能力。
2. 样本分层：正常请求、边界请求、禁止请求、多轮请求、跨语言请求、对抗请求。
3. 指标体系：harmful compliance、refusal accuracy、over-refusal、safe alternative quality、attack success rate。
4. 人工复核：高风险样本不能只依赖自动 judge。
5. 错误归因：区分政策理解错、上下文理解错、拒答边界错、系统工具权限错。
6. 回归机制：红队发现样本进入 regression suite。

面试回答：

```text
我会先定义风险分类和严重度，再构造分层测试集，既包括明显违规请求，也包括正常安全请求和边界请求。指标上不能只看拒答率，而要同时看 harmful compliance、over-refusal、正常 helpfulness、safe alternative quality 和多轮攻击成功率。高风险类别需要人工或专家复核，红队发现的问题要沉淀成回归集。
```

常见追问：“Safety eval 和 red teaming 区别是什么？”

回答：

```text
Safety eval 更像系统化评估，有固定测试集、指标和发布门禁。Red teaming 更偏主动探索未知失败模式。两者互补：red teaming 发现新问题，safety eval 把已知问题稳定纳入回归测试。
```

## 8.11 Over-Refusal：安全模型也会伤害体验

Over-refusal 指模型对本应回答的正常请求过度拒绝。

例如：

1. 用户问网络安全防御知识，模型误判为攻击请求。
2. 用户询问心理健康支持，模型只拒绝或机械建议求助。
3. 用户进行合法药品科普，模型过度回避。
4. 用户要求分析暴力文本，模型无法区分分析和煽动。

面试回答：

```text
安全优化不能只降低 harmful compliance，还要控制 over-refusal。否则模型看似安全，但正常用户体验很差。评估时要构造 benign-but-sensitive 样本，检查模型是否能给出安全替代、教育性说明或澄清问题，而不是简单拒绝。
```

降低 over-refusal 的方法：

1. 加入边界样本和正常敏感请求。
2. 训练 safe completion，而不是只训练 refusal。
3. 区分意图、上下文和细节级别。
4. 设计分级响应策略。
5. 用人工评估检查误拒。

## 8.12 Red Teaming：主动发现未知失败模式

Red teaming 的目标不是证明模型安全，而是尽可能发现模型和系统的失败模式。

高质量 red teaming 包括：

1. 风险覆盖：有害内容、隐私、prompt injection、工具调用、偏见、危险能力。
2. 攻击面覆盖：单轮、多轮、跨语言、RAG、Agent、文件上传、网页浏览、工具结果。
3. 角色覆盖：普通用户、恶意用户、内部员工、第三方插件、外部文档。
4. 严重度分级：P0、P1、P2、P3。
5. 修复闭环：问题、复现、根因、修复、回归测试。

面试回答：

```text
Red teaming 应该和发布流程绑定，而不是一次性活动。红队发现的问题要分级，P0/P1 问题进入发布门禁；修复后要把样本加入 regression suite，防止后续模型或 prompt 更新重新引入同类问题。
```

## 8.13 Dangerous Capability Eval

危险能力评估关注模型是否显著降低高风险行为门槛，而不只是会不会输出违规文本。

面试中可以按风险类型拆：

1. Cyber：是否辅助漏洞利用、攻击链规划或恶意自动化。
2. Bio：是否降低生物安全相关高风险门槛。
3. Autonomy：是否能长期规划、获取资源、规避限制。
4. Tool use：是否能用工具放大风险。
5. Deception：是否表现出误导、隐瞒或策略性行为。

安全表达要保持防御和评估视角，不提供具体危险操作细节。

面试回答：

```text
危险能力评估要看模型相对非 AI baseline 是否提供实质增量，而不是只看是否知道公开信息。我会明确评估条件，包括 prompt、工具、scaffold、专家辅助、多轮交互和成本；同时区分默认部署能力和 capability elicitation 后的能力上限。评估结果要进入访问控制、分级发布和发布门禁。
```

常见追问：“什么是 capability elicitation？”

回答：

```text
Capability elicitation 是用合适 prompt、工具、scaffold 或专家辅助尽量激发模型潜在能力，避免低估风险。但评估报告要清楚记录条件，不能把极端设置下的能力直接等同于普通用户默认使用能力。
```

## 8.14 Privacy、Memorization 与数据治理

LLM 隐私风险来自多个环节：训练数据、模型参数、RAG 知识库、日志、工具调用和输出。

常见风险：

1. 训练数据中包含 PII、密钥、私密文本。
2. 模型记忆并复现敏感样本。
3. RAG 权限过滤错误导致越权读取。
4. 日志记录用户敏感输入。
5. 工具调用返回敏感字段但未脱敏。

面试回答：

```text
LLM privacy 要做全生命周期治理。训练前做数据来源记录、许可检查、PII scrub、密钥扫描、去重和数据分级；训练后做 memorization eval、PII leakage eval 和 extraction red team；部署时做 RAG 权限过滤、输出敏感信息检测、日志脱敏、访问控制和事故响应。不能只靠一次数据清洗。
```

常见追问：“Memorization 和 privacy 有什么关系？”

回答：

```text
Memorization 本身是模型复现训练样本。如果被记住的是公开常识，风险较低；如果是 PII、密钥、私有邮件或商业机密，就会变成隐私和安全风险。所以要结合数据敏感度、重复度、抽取难度和输出过滤一起评估。
```

## 8.15 诚实性、不确定性和幻觉

Honesty 面试常围绕一个问题：如何让模型不知道时说不知道？

模型自信幻觉的原因包括：

1. Next-token prediction 不直接优化真实性。
2. 训练数据中有错误事实。
3. SFT/RLHF 可能偏好流畅、完整、自信的回答。
4. 模型缺少外部验证。
5. 奖励模型可能奖励看似专业的错误答案。

面试回答：

```text
让模型诚实表达不确定性，需要训练、评估和系统共同支持。训练上加入不知道、资料不足、需要澄清、需要检索的样本；评估上看 factuality、calibration、abstention accuracy、unsupported claim rate 和 citation accuracy；系统上结合 RAG、工具验证和高风险领域人工审核。关键是不要奖励模型假装知道。
```

不要说“降低 temperature 就能解决幻觉”。低温只能降低随机性，不能保证事实正确。

## 8.16 Mechanistic Interpretability 与 Safety

Mechanistic interpretability 试图理解模型内部 features、circuits 和信息流。

它对 safety 的潜在价值：

1. 分析拒答、幻觉、事实性等行为机制。
2. 发现与危险能力或不安全行为相关的表示。
3. 用 activation patching 或 ablation 做因果验证。
4. 用 SAE 等方法分解更可解释的 feature。
5. 辅助 red teaming 和模型调试。

但它目前不能单独证明模型安全。

面试回答：

```text
Mechanistic interpretability 对 safety 很有价值，但目前更像辅助诊断工具，而不是安全保证。行为评估、红队、系统权限和发布治理仍然必不可少。解释工具发现某个 feature 或 circuit，也需要验证它是否因果影响行为，以及是否能泛化到真实场景。
```

常见追问：“Attention heatmap 是否能解释模型？”

回答：

```text
Attention heatmap 可以提供线索，但不是完整解释。它只显示某层某头的读取权重，不说明 value 写了什么，也不说明后续层如何使用这些信息，更不一定有因果性。要做解释还需要 ablation、patching 和路径分析。
```

## 8.17 Model Editing 与 Unlearning

Model editing 关注局部修改模型知识或行为，例如把错误事实改成新事实，同时尽量不影响其他能力。

Unlearning 关注让模型近似移除某些训练数据、知识、行为或能力的影响。

面试回答：

```text
Editing 通常有明确的新目标答案，目标是局部更新；unlearning 通常要求目标数据或目标知识的影响消失，同时保留其他能力。LLM unlearning 很难，因为知识分散在参数中，同一知识可能来自多个来源，很多方法只能提供 approximate unlearning 证据，不能轻易宣称完全遗忘。
```

评估 unlearning 要看：

1. Forget set 是否不再复现。
2. Retain set 能力是否保持。
3. 改写、多轮、翻译是否仍能诱导目标知识。
4. Membership inference 风险是否下降。
5. 是否只是输出层 guardrail suppression。
6. 是否有版本记录和审计证据。

## 8.18 Model Card、System Card 和发布门禁

Model card 关注模型本身：训练数据概要、用途、不适用场景、评估结果、限制、风险和缓解措施。

System card 关注完整系统：模型、RAG、工具、权限、policy layer、监控、红队、隐私和事故响应。

面试回答：

```text
大模型产品的风险往往来自系统组合，而不只是模型权重本身。一个模型接入 RAG、浏览器、代码执行或企业 API 后，风险边界会变化。因此 model card 说明模型能力和限制，system card 说明完整应用系统的风险、评估、缓解和监控。
```

发布门禁应提前定义，不能上线前临时解释结果。

门禁可以包括：

1. P0 安全问题为 0。
2. P1 问题有明确修复或限制。
3. Harmful compliance 低于阈值。
4. Over-refusal 不超过阈值。
5. Prompt injection 和工具越权风险低于阈值。
6. 隐私和数据泄露评估通过。
7. Dangerous capability 不超过发布等级。
8. 监控、回滚和事故响应就绪。

## 8.19 Agent Safety 系统设计题

题目可能是：

```text
请设计一个能调用企业 API 的 Agent，并保证工具调用安全。
```

回答可以分七层：

1. 工具注册：tool name、description、schema、风险等级、权限范围。
2. 身份权限：用户身份、租户、资源、操作级别。
3. 参数校验：类型、范围、枚举、必填项、业务规则。
4. 执行隔离：只读/写操作分离，高风险操作二次确认。
5. Prompt injection 防护：外部内容不能直接控制工具调用。
6. 审计日志：记录 tool call trace、参数、结果、用户确认。
7. 评估监控：工具选择准确率、参数准确率、越权率、失败率、人工复核。

标准回答：

```text
我会把 Agent 安全边界放在系统层，而不是只依赖模型。工具要有 schema、权限和风险等级；调用前做用户身份和资源权限检查；参数必须由系统校验；写操作和高风险操作需要二次确认；工具返回错误时模型不能编造结果；所有调用要有 trace 和审计。外部文档、网页或邮件内容都应被视为不可信，不能直接覆盖系统指令或触发高权限工具。
```

## 8.20 Safety 面试中的常见失分点

Alignment 与 Safety 面试常见失分点包括：

1. 把 safety 简化成拒答。
2. 只讲 RLHF，不讲评估、红队和系统防护。
3. 认为 DPO 天然更安全。
4. 不区分 jailbreak 和 prompt injection。
5. 讲 prompt injection 时只说“加强 prompt”，不讲权限和隔离。
6. 只看 harmful compliance，不看 over-refusal。
7. 不会设计 safety eval，只会列风险词。
8. 不知道 red teaming 和 regression suite 的关系。
9. 把 mechanistic interpretability 说成可以证明模型安全。
10. 对 unlearning 轻易宣称完全遗忘。
11. 讨论危险能力时提供不必要的攻击细节。
12. 不讲发布门禁、灰度、回滚和事故响应。

## 8.21 高频题回答模板

### 8.21.1 RLHF 为什么不是最终解

```text
RLHF 用人类偏好训练 reward model，再优化 policy，让模型更符合人类期望。但它的根本问题是偏好和 reward 都是 proxy，不等于真实目标。偏好数据可能有标注偏差、长度偏差和覆盖不足，reward model 可能被过度优化导致 reward hacking。复杂任务里人类也未必能准确监督。因此 RLHF 是重要后训练方法，但还需要 scalable oversight、red teaming、safety eval、系统防护和发布治理。
```

### 8.21.2 如何防 jailbreak

```text
我会从训练、评估和系统三层防护。训练上加入安全 SFT、偏好优化、多轮诱导和边界样本；评估上覆盖角色扮演、跨语言、编码包装、多轮诱导和正常请求误拒；系统上加入 policy layer、输入输出检测、日志监控和红队回归。指标上同时看 attack success rate、harmful compliance、refusal accuracy、over-refusal 和 safe alternative quality。
```

### 8.21.3 如何设计 safety eval

```text
我会先定义风险 taxonomy 和严重度，再构造正常、边界、禁止、多轮、跨语言和对抗样本。指标上同时看 harmful compliance、refusal accuracy、over-refusal、safe alternative quality、jailbreak robustness 和正常 helpfulness。高风险类别需要人工或专家复核，红队发现的问题要进入 regression suite，并作为发布门禁的一部分。
```

### 8.21.4 如何防 prompt injection

```text
Prompt injection 的核心是可信指令和不可信数据边界混淆。防御上要明确 system、developer、user、external content 和 tool result 的信任等级；把网页、文档、邮件和检索内容标记为 untrusted；工具调用由系统做权限、schema 和参数校验；高风险操作二次确认；所有调用有审计日志。不能让外部文本直接控制高权限动作。
```

### 8.21.5 如何平衡 helpfulness 和 safety

```text
我会按风险等级和用户意图区分，而不是一刀切拒答。明显有害请求要拒绝危险细节并提供安全替代；正常教育、防御、合规请求要尽量帮助；边界请求可以澄清意图或降低细节级别。评估上同时看 harmful compliance、over-refusal、helpfulness 和 safe alternative quality。
```

## 8.22 一套完整 Safety 面试回答模板

如果被问开放题：“请你设计一个大模型 Safety Platform”，可以这样组织：

```text
第一，我会先定义风险 taxonomy 和严重度，包括有害内容、隐私、jailbreak、prompt injection、工具滥用、危险能力、偏见和 over-refusal。

第二，建设数据和训练闭环。数据包括 safety SFT、偏好数据、边界样本、红队样本、正常敏感请求和 regression suite；训练方法包括 SFT、RLHF/DPO、Constitutional AI 或 RLAIF，并控制 reward hacking 和过度拒答。

第三，建设评估体系。离线 eval 覆盖 harmful compliance、refusal accuracy、over-refusal、safe alternative quality、prompt injection、tool misuse、privacy 和 dangerous capability；高风险样本需要人工或专家复核。

第四，建设系统防护。输入输出 policy、RAG 权限过滤、工具最小权限、schema 校验、高风险操作二次确认、沙箱、日志审计和 prompt injection 隔离。

第五，建设发布治理。用 model card 和 system card 记录能力、限制、风险和缓解措施；发布前设置门禁，发布时灰度，发布后监控 bad case、攻击趋势、误拒和事故。

第六，形成持续迭代。线上失败样本和红队样本进入 regression suite，修复后重新评估，保证安全能力不会随模型版本、prompt 或工具更新回退。
```

## 8.23 准备清单

准备 Alignment 与 Safety 面试时，至少要能回答下面的问题：

1. AI Safety 和 Alignment 有什么区别？
2. Helpful、Honest、Harmless 如何冲突？
3. RLHF 的完整流程是什么？根本局限是什么？
4. DPO 简化了什么，没有解决什么？
5. Reward hacking 和 Goodhart 定律有什么关系？
6. 如何设计 safety eval？
7. Safety eval 和 red teaming 有什么区别？
8. Jailbreak 和 prompt injection 有什么区别？
9. 如何防 prompt injection？
10. 如何评估 over-refusal？
11. 如何让模型诚实表达不确定性？
12. 如何治理模型隐私和 memorization？
13. Dangerous capability eval 要评估什么？
14. Capability elicitation 为什么重要？
15. Mechanistic interpretability 对 safety 有什么用和局限？
16. Model editing 和 unlearning 有什么区别？
17. Model card 和 system card 有什么区别？
18. 如何设计 Agent 工具调用安全架构？
19. 如果新模型能力提升但 jailbreak 成功率也提升，是否上线？
20. 如何把红队发现的问题转成长期回归测试？

这一章的核心不是让你背完所有安全术语，而是建立一套风险治理思维：先定义目标和风险，再设计训练、评估、系统防护和发布门禁，最后用线上监控和回归测试持续改进。
