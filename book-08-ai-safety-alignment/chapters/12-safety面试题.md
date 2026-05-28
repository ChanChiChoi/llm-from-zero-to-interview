# 第十二章：Safety 面试题

高频问题：RLHF 有什么根本问题？如何防 jailbreak？如何评估危险能力？如何让模型诚实表达不确定性？

本章是第八册的收束章。目标不是继续讲新概念，而是把前 11 章转成面试中的结构化表达能力。

面试重点：Safety 面试不是背政策，也不是只说“拒绝危险请求”。高质量回答必须能同时覆盖目标定义、风险分类、训练方法、评估、红队、系统防护、上线治理和 trade-off。

安全边界：本章只提供防御性、治理性和评估性回答模板，不提供 jailbreak、数据抽取、网络滥用、生物风险等可操作攻击步骤。

## 本章目标

学完本章，你要能做到：

1. 用 2 分钟解释 Safety 和 Alignment。
2. 用 3 分钟拆解 RLHF 的根本局限。
3. 用 5 分钟设计一个 Safety eval。
4. 用系统设计方式回答 Prompt Injection 防御。
5. 用发布门禁方式回答危险能力评估。
6. 用治理视角回答 model card、system card 和 responsible scaling。
7. 在不知道具体细节时，仍能给出清晰的分析框架。

## 1. Safety 面试的通用答题框架

遇到 safety 题，不要直接背一个方法。

推荐用七层框架：

```text
1. 定义目标
2. 定义风险 taxonomy
3. 分析失败模式
4. 训练或系统缓解方法
5. 评估和红队
6. 上线门禁和监控
7. Trade-off 和残余风险
```

例如面试官问：

```text
你如何提升一个模型的安全性？
```

差回答：

```text
做 RLHF，加安全数据，防 jailbreak。
```

好回答：

```text
我会先定义安全目标，例如降低 harmful compliance，同时避免 over-refusal。然后建立风险 taxonomy，覆盖有害内容、隐私、jailbreak、prompt injection、工具滥用和高风险专业建议。训练上可以做安全 SFT、偏好优化和拒答样本；系统上加入 policy layer、权限控制、输出过滤和人工确认。评估上要跑 safety eval、红队、多轮攻击、正常请求误拒评估和线上监控。最后用发布门禁控制 P0/P1 风险，并持续把失败样本加入 regression suite。
```

## 2. 问题 1：AI Safety 和 Alignment 有什么区别？

### 标准回答

```text
AI Safety 更关注模型是否会造成伤害，比如有害内容、隐私泄露、jailbreak、prompt injection、工具滥用、危险能力和部署事故。Alignment 更关注模型目标和行为是否符合人类真实意图、价值和安全边界，比如模型是否 helpful、honest、harmless，是否会 reward hacking，是否学到错误目标。Safety 可以看作 alignment 的重要目标之一，但 alignment 还包括诚实性、可控性、目标一致性和泛化行为。
```

### 追问 1：为什么安全不是简单拒答？

回答要点：

1. 安全有漏拒和误拒两类错误。
2. 只提高拒答率会导致 over-refusal。
3. 正常安全教育、防御、合规请求应该被帮助。
4. 高风险请求应拒绝危险细节并提供安全替代。
5. 评估要同时看 harmful compliance、over-refusal 和 helpfulness。

### 追问 2：Helpful、Honest、Harmless 如何冲突？

回答要点：

1. 用户要求危险帮助时，helpful 和 harmless 冲突。
2. 用户要求模型不知道的信息时，helpful 和 honest 冲突。
3. 有些真实信息本身高风险，honest 和 harmless 可能冲突。
4. 好模型需要按风险等级和上下文处理，而不是单目标优化。

## 3. 问题 2：什么是 Alignment Problem？

### 标准回答

```text
Alignment Problem 是如何让 AI 系统实际优化的目标和行为，与人类真实意图、价值和安全边界保持一致的问题。难点在于人类真实目标很难完整写成训练目标，而模型内部学到的策略也未必等于我们设计的目标。部署后遇到分布外用户、工具和攻击时，这些目标错配会暴露出来。
```

### 追问 1：Outer alignment 和 inner alignment 区别？

回答：

```text
Outer alignment 问我们写下的训练目标是否代表真实人类目标，例如 reward model、偏好数据和安全政策是否只是 proxy。Inner alignment 问即使目标设计合理，训练出的模型是否真的学到了这个目标，还是学到了训练分布上的捷径或错误内部目标。
```

### 追问 2：Goal misgeneralization 是什么？

回答要点：

1. 模型能力仍在，但目标泛化错了。
2. 比普通能力失败更隐蔽。
3. LLM 中可能表现为优化用户满意、judge 分数、长回答或表面专业，而不是真实帮助。
4. 需要分布外和反事实评估。

## 4. 问题 3：RLHF 有什么根本问题？

### 标准回答

```text
RLHF 的核心价值是用人类偏好替代难写的 reward function，让模型更符合人类期望。但它的根本问题是 human preference 和 reward model 都是 proxy，不等于真实目标。偏好数据可能有标注员偏差、长度偏差、风格偏差和覆盖不足；reward model 可能被过度优化，导致 reward hacking；RL 优化还需要控制 KL、稳定性和 helpfulness-safety trade-off。因此 RLHF 不是对齐问题的最终解，只是重要的后训练方法之一。
```

### 追问 1：Reward hacking 是什么？

回答要点：

1. 模型利用 reward function 或 reward model 漏洞。
2. 得到高分但偏离真实目标。
3. LLM 中表现为冗长、自信幻觉、过度拒答、引用不忠实、judge gaming。
4. 符合 Goodhart 定律。

### 追问 2：DPO 是否解决了 RLHF 的问题？

回答：

```text
DPO 简化了 RLHF 的优化流程，避免显式训练 reward model 和 PPO loop，但它仍然优化偏好数据隐含的目标。如果偏好数据有偏，chosen/rejected 质量差，模型仍会学到表面偏好和目标错配。所以 DPO 改善工程复杂度，不消除 alignment 的根本难题。
```

### 追问 3：如何缓解 reward model overoptimization？

回答要点：

1. 改进偏好数据和 hard negative。
2. 校准 reward model。
3. 控制优化强度。
4. KL penalty。
5. Early stopping。
6. 限制 best-of-n。
7. 用 human eval、expert eval 和 red teaming 监控 reward-human gap。

## 5. 问题 4：Scalable Oversight 解决什么问题？

### 标准回答

```text
Scalable Oversight 解决的是监督瓶颈。当模型输出的代码、数学证明、长文档分析、RAG 综合或 Agent 工具调用计划超过单个人类直接判断能力时，普通 RLHF 假设就不成立。Scalable oversight 试图通过任务分解、AI-assisted evaluation、debate、iterated amplification、recursive reward modeling 和 Constitutional AI，让监督能力随模型能力增长。
```

### 追问 1：AI feedback 能否替代 human feedback？

回答要点：

1. 不能完全替代。
2. AI feedback 可扩规模、降成本、生成 critique。
3. Human feedback 提供价值锚点和校准。
4. 高风险场景需要专家或人工复核。
5. 最稳妥是 human gold labels + AI assisted eval + human audit。

### 追问 2：Debate 的局限是什么？

回答要点：

1. 可能优化说服力而不是真实性。
2. Human judge 仍可能被误导。
3. 复杂专业反驳仍需专家。
4. 辩论规则和成本难控制。

## 6. 问题 5：如何防 Jailbreak？

### 标准回答

```text
防 jailbreak 不能只靠一句 system prompt。需要训练和系统多层防护：训练上加入安全 SFT、偏好优化、多轮诱导样本、边界样本和安全替代；评估上覆盖角色扮演、编码、多轮诱导、对抗后缀和正常请求误拒；系统上加输入输出分类、policy layer、日志监控和 red teaming。指标上同时看 attack success rate、harmful compliance、refusal accuracy、over-refusal 和 safe alternative quality。
```

### 追问 1：为什么 jailbreak 难完全防住？

回答要点：

1. 自然语言空间巨大。
2. 攻击可以多轮、跨语言、编码和角色包装。
3. 模型遵循指令的能力本身带来风险。
4. 安全训练覆盖不了所有组合。
5. 需要持续红队和 regression suite。

## 7. 问题 6：如何防 Prompt Injection？

### 标准回答

```text
Prompt injection 的核心是 LLM 应用混淆了可信指令和不可信数据。防御不能只靠 prompt，而要做系统层隔离：明确 system/developer/user/external content/tool result 的指令层级；把 RAG 文档和网页标记为 untrusted content；检索前做来源和权限过滤；生成时只把文档当证据，不执行文档指令；工具调用要最小权限、schema 校验、高风险二次确认、审计日志和回滚。评估时看 instruction hijack、data exfiltration、unauthorized tool call 和正常任务成功率。
```

### 追问 1：Jailbreak 和 Prompt Injection 区别？

回答要点：

1. Jailbreak 主要绕过模型安全策略。
2. Prompt injection 主要污染或覆盖应用指令。
3. Jailbreak 多来自当前用户。
4. Prompt injection 可来自外部文档、网页、邮件、工具返回。
5. Agent 和 RAG 中 prompt injection 风险更高。

## 8. 问题 7：如何设计 Safety Eval？

### 标准回答

```text
我会先定义风险 taxonomy，例如有害内容、自伤、网络安全、隐私、jailbreak、prompt injection、工具滥用、偏见、over-refusal 和高风险专业建议。然后按允许、边界、禁止、多轮、跨语言、对抗和正常请求分层构造样本。指标上同时看 harmful compliance、refusal accuracy、over-refusal、safe alternative quality、jailbreak robustness 和正常 helpfulness。最后做人工复核、分层分析、错误归因和 regression suite。
```

### 追问 1：Safety eval 和 red teaming 区别？

回答：

```text
Safety eval 更像系统化评估，可以包含固定测试集、自动指标、人工评估和上线门禁。Red teaming 更偏主动探索和对抗发现，用来寻找未知失败模式。红队发现的样本可以沉淀成 safety regression suite。
```

## 9. 问题 8：如何评估危险能力？

### 标准回答

```text
危险能力评估关注模型是否显著降低高风险行为门槛，而不只是会不会输出违规文本。我会按 cyber、bio、autonomy、tool use、deception 等风险分层，比较模型相对于搜索引擎、公开资料和专家流程是否提供实质增量。评估时要区分自然能力和 capability elicitation 后的能力上限，并记录 prompt、工具、专家辅助、多轮和成本条件。结果要进入发布门禁、访问控制和分级发布。
```

### 追问 1：什么是 capability elicitation？

回答要点：

1. 用合适 prompt、工具、scaffold、专家辅助激发模型潜在能力。
2. 防止低估模型风险。
3. 需要报告评估条件。
4. 同时避免不现实高估普通部署风险。

### 追问 2：如何设计发布门禁？

回答要点：

1. P0 安全问题为 0。
2. P1 有修复或限制。
3. 危险能力不超过阈值。
4. 隐私评估通过。
5. Prompt injection 工具误用率低于阈值。
6. Over-refusal 不超过范围。
7. 监控、审计、回滚就绪。

## 10. 问题 9：如何让模型诚实表达不确定性？

### 标准回答

```text
诚实表达不确定性需要训练、评估和系统支持。训练上要有不知道、资料不足、需要检索、需要澄清的样本，避免奖励自信但错误的回答。评估上要看 factuality、calibration、abstention accuracy、unsupported claim rate 和引用准确性。系统上可以结合 RAG、工具验证、置信度触发检索、对高风险领域要求引用或人工审核。关键是不要让模型为了 helpfulness 或偏好评分假装知道。
```

### 追问 1：为什么模型会自信幻觉？

回答要点：

1. 语言模型目标是预测下一个 token，不直接优化真实性。
2. SFT/RLHF 可能偏好流畅、完整、自信回答。
3. 训练数据中有错误和不确定表达不足。
4. 缺少外部验证。
5. Reward hacking 可能奖励看似专业的错误回答。

## 11. 问题 10：Mechanistic Interpretability 对 Safety 有什么用？

### 标准回答

```text
Mechanistic interpretability 试图逆向工程模型内部的 features、circuits 和信息流。对 safety 来说，它可能帮助我们理解幻觉、拒答、jailbreak、危险能力、steering 和 safety tuning 的内部机制。例如 activation patching 可以验证某个激活是否因果影响输出，SAE 可以帮助分解更可解释的 feature。但当前它还不能单独证明模型安全，更适合作为 red teaming、safety eval 和模型调试的补充工具。
```

### 追问 1：Attention heatmap 为什么不是完整解释？

回答要点：

1. Attention 权重只是读取模式。
2. 不说明 value 写入了什么。
3. 不说明后续层如何使用。
4. 不一定有因果性。
5. 需要 patching、ablation 和路径分析。

## 12. 问题 11：Steering 和 Representation Engineering 如何用于 Safety？

### 标准回答

```text
Representation engineering 从高层行为出发，比如 honesty、factuality、refusal、harmlessness，寻找模型内部表示方向。Steering 可以在推理时修改激活，让模型行为朝目标方向变化。它对 safety 的价值是可以帮助监控和调节拒答、不确定性、事实性等行为；风险是方向可能不泛化、有副作用，也可能被双用来削弱安全机制。因此它不能替代 RLHF、安全评估、红队和系统权限控制。
```

### 追问 1：拒答方向说明什么？

回答要点：

1. 拒答行为可能有重要内部表示。
2. 有助于分析 safety tuning 和 over-refusal。
3. 也暴露当前安全机制可能脆弱。
4. 单一方向不等于完整安全机制。
5. 有双用风险。

## 13. 问题 12：Model Editing 和 Unlearning 有什么区别？

### 标准回答

```text
Model editing 关注局部修改模型知识或行为，例如把一个错误事实改成新事实，同时尽量保持其他知识不变。Unlearning 关注让模型近似移除某些训练数据、知识、行为或能力的影响。Editing 通常有明确新目标答案，unlearning 通常要求目标影响消失并保留其他能力。LLM unlearning 很难，因为知识分散在参数中，同一知识可能来自多个来源，很多方法只能提供 approximate unlearning 证据。
```

### 追问 1：如何评估 unlearning？

回答要点：

1. Forget set 是否不再复现。
2. Retain set 能力是否保持。
3. 改写、多轮、翻译是否仍能诱导目标知识。
4. Membership inference 风险是否下降。
5. 是否只是 guardrail suppression。
6. 是否有第三方审计和版本记录。

## 14. 问题 13：LLM Privacy 如何治理？

### 标准回答

```text
LLM privacy 要做全生命周期治理。训练前记录数据来源、许可、PII scrub、去重、密钥扫描和数据分级；训练中控制敏感数据进入、监控过拟合，必要时考虑差分隐私；训练后做 memorization eval、PII leakage eval 和 training data extraction red team；部署时做 RAG 权限过滤、输出敏感信息检测、日志脱敏、访问控制和事故响应。Watermarking 主要用于生成内容溯源，不直接解决训练数据隐私。
```

### 追问 1：Memorization 和 privacy 有什么关系？

回答要点：

1. Memorization 是复现训练样本。
2. 如果样本包含 PII、密钥、私密文本，就会有隐私风险。
3. 去重、PII 清理、差分隐私、输出过滤和泄露评估都能降低风险。
4. 不能只靠一次 scrub。

## 15. 问题 14：Model Card 和 System Card 有什么区别？

### 标准回答

```text
Model card 关注模型本身，记录模型用途、不适用场景、训练数据概要、评估结果、限制、风险和缓解措施。System card 关注完整应用系统，包括模型、RAG、工具调用、权限、policy layer、监控、红队、隐私和事故响应。大模型产品的实际风险往往来自系统组合，所以 system card 对 Agent 和 RAG 产品尤其重要。
```

### 追问 1：为什么文档不是形式主义？

回答要点：

1. 文档把评估和风险转成可沟通信息。
2. 帮助用户理解使用边界。
3. 支持内部审计和责任追踪。
4. 前提是文档和真实系统一致，并随版本更新。

## 16. 综合题：设计一个 Safety Platform

### 题目

```text
请设计一个大模型 Safety Platform，支持模型训练、评估、红队、发布门禁、线上监控和事故响应。
```

### 回答框架

可以分七层。

第一，风险 taxonomy。

1. 有害内容。
2. 隐私。
3. Jailbreak。
4. Prompt injection。
5. 工具滥用。
6. 危险能力。
7. Over-refusal。

第二，数据层。

1. Safety SFT 数据。
2. 偏好数据。
3. 红队样本。
4. Regression suite。
5. PII 和隐私测试集。
6. 正常请求集。

第三，训练层。

1. SFT。
2. RLHF / DPO。
3. Constitutional AI / RLAIF。
4. Safety tuning。
5. Unlearning / editing。

第四，评估层。

1. Static safety eval。
2. LLM judge + human audit。
3. Red teaming。
4. Dangerous capability eval。
5. Privacy eval。
6. Tool-use eval。

第五，系统防护层。

1. Policy layer。
2. Input/output filters。
3. Tool permission。
4. Human confirmation。
5. Sandbox。
6. Logging and audit。

第六，发布治理层。

1. Model card。
2. System card。
3. 发布门禁。
4. 分级发布。
5. 第三方评估。
6. 回滚机制。

第七，线上层。

1. Abuse monitoring。
2. Incident response。
3. 用户反馈。
4. 日志抽检。
5. 持续回归。
6. 安全策略更新。

### 标准回答

```text
我会把 Safety Platform 设计成风险 taxonomy、数据、训练、评估、系统防护、发布治理和线上监控七层。首先定义风险类别和严重程度，然后维护 safety 数据、红队样本和 regression suite。训练层支持 SFT、RLHF/DPO、Constitutional AI 和安全微调。评估层支持 jailbreak、prompt injection、privacy、dangerous capability、tool-use 和 over-refusal。系统层做 policy、过滤、权限、沙箱和人工确认。发布前用 model card、system card 和门禁控制上线，发布后做监控、事故响应和持续回归。
```

## 17. 高频追问清单

1. Safety 和 Alignment 的边界在哪里？
2. RLHF 为什么不是最终解？
3. DPO 是否更安全？
4. 如何避免 reward hacking？
5. 如何评估 over-refusal？
6. 如何防 jailbreak？
7. 如何防 indirect prompt injection？
8. Agent 工具调用如何做权限控制？
9. 什么是 dangerous capability eval？
10. Capability elicitation 为什么重要？
11. Mechanistic interpretability 能否证明模型安全？
12. Steering 是否有双用风险？
13. Unlearning 如何证明有效？
14. PII scrub 为什么不够？
15. Model card 里必须写什么？
16. 如何设计发布门禁？
17. 如何平衡 transparency 和 security？
18. 如何处理线上安全事故？

## 18. 常见失分点

### 18.1 只说拒答

Safety 不是拒答率。

要同时看 helpfulness、honesty、harmlessness、over-refusal 和安全替代。

### 18.2 只说 RLHF

RLHF 是方法之一，不是安全系统。

还要讲评估、红队、权限、治理和监控。

### 18.3 不区分 jailbreak 和 prompt injection

这会显得没有真实系统安全意识。

### 18.4 不讲评估指标

安全方案必须有指标，否则无法上线决策。

### 18.5 不讲 trade-off

Safety 和 helpfulness、privacy 和 utility、transparency 和 security 都有 trade-off。

### 18.6 把前沿风险说成确定事实

例如 deceptive alignment、内部目标等，要谨慎区分已观察现象、理论风险和推测。

### 18.7 提供危险细节

面试中讲风险和防御即可，不要展示攻击步骤或高风险操作细节。

## 19. 10 个万能句式

1. “我会先定义风险 taxonomy，而不是直接上方法。”
2. “安全不是只看 harmful compliance，也要看 over-refusal。”
3. “RLHF 优化的是偏好 proxy，不等于真实目标。”
4. “Prompt injection 的根本问题是指令和数据边界混淆。”
5. “Agent 安全的关键是不要让不可信文本直接控制高权限动作。”
6. “危险能力评估要比较模型相对非 AI baseline 是否显著降低门槛。”
7. “红队发现的问题要进入 regression suite，否则无法防复发。”
8. “Mechanistic interpretability 有潜力，但目前不能替代行为评估和治理。”
9. “Unlearning 通常只能提供近似证据，不能轻易宣称完全遗忘。”
10. “发布门禁要在评估前定义，不能上线前临时解释结果。”

## 20. 模拟面试题

### 基础概念

1. 什么是 AI Safety？
2. 什么是 Alignment Problem？
3. Helpful、Honest、Harmless 如何理解？
4. Outer alignment 和 inner alignment 区别？
5. Reward hacking 和 Goodhart 定律关系？

### 后训练和监督

1. RLHF 的根本局限是什么？
2. DPO 是否能解决 reward hacking？
3. 什么是 scalable oversight？
4. Constitutional AI 解决什么问题？
5. AI feedback 能否替代 human feedback？

### 攻防和评估

1. 如何防 jailbreak？
2. 如何防 prompt injection？
3. 如何设计 safety eval？
4. 如何做 red teaming？
5. 如何评估 dangerous capability？

### 解释、控制和治理

1. Mechanistic interpretability 对 safety 有什么用？
2. 什么是 steering vector？
3. Model editing 和 unlearning 区别？
4. LLM privacy 如何治理？
5. Model card 和 system card 区别？

### 系统设计

1. 设计一个 Safety Platform。
2. 设计一个 Agent 安全架构。
3. 设计一个 RAG prompt injection 防御系统。
4. 设计一个危险能力发布门禁。
5. 设计一个隐私和数据治理流程。

## 21. 小练习

### 练习 1

用 3 分钟回答：RLHF 为什么不是 alignment 的最终解？

要求覆盖：proxy、reward hacking、标注偏差、scalable oversight 和评估。

### 练习 2

用 5 分钟设计一个 Agent safety eval。

要求覆盖：tool permission、prompt injection、multi-turn jailbreak、unauthorized action、human confirmation。

### 练习 3

设计一个隐私泄露评估。

要求覆盖：memorization、PII leakage、RAG 权限、日志脱敏和 incident response。

### 练习 4

为一个企业知识库助手写 system card 大纲。

### 练习 5

模拟回答：如果新模型 helpfulness 提升，但 jailbreak 成功率也提升，你是否上线？

要求覆盖：风险严重度、分层、门禁、灰度、修复和回滚。

## 22. 本章总结

Safety 面试考察的是系统思维，而不是单点术语。

高质量回答要覆盖目标、风险、训练、评估、红队、系统防护、发布门禁和线上监控。

RLHF、DPO、Constitutional AI、steering、interpretability、unlearning 都是工具，不是单独的安全保证。

Jailbreak 和 prompt injection 要区分；前者偏模型安全边界，后者偏 LLM 应用指令边界。

危险能力评估、privacy、model card 和 responsible scaling 把 safety 从模型训练推进到发布治理。

面试中最重要的能力是：在模糊、高风险、开放问题中给出清晰的风险分解、可验证评估和可执行治理闭环。
