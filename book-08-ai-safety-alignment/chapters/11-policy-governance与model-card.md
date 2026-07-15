# 第十一章：Policy、Governance 与 Model Card

重点：模型发布规范、model card、system card、使用政策、风险披露和治理框架。

面试重点：AI Safety 不是只做模型训练和红队测试，最终必须落实到文档、流程、责任、发布门禁、使用边界和持续监控。Model card 和 system card 是把技术评估转成治理沟通的重要工具。

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点核对了 Datasheets for Datasets、Model Cards for Model Reporting、NIST AI Risk Management Framework、NIST Generative AI Profile、OpenAI System Cards / Preparedness Framework、Anthropic Responsible Scaling Policy、EU AI Act 风险分级和透明度要求，以及前序 safety eval、red teaming、privacy、unlearning、tool safety 和 data governance 章节。

本章采用工程治理和面试表达口径：

1. policy 只写抽象类别、期望动作、责任边界和评估要求，不列可复用违规样例或规避策略。
2. model card / system card 强调能力、限制、风险、评估覆盖、缓解措施和版本责任，不写成营销材料。
3. governance 关注发布门禁、审批、审计、监控、事故响应和持续更新，不把“有文档”当成治理完成。
4. responsible scaling 只讲“能力越强、评估与控制越严格”的原则，不照搬单一机构条款作为通用法律结论。
5. 合规内容只作技术团队面试和工程治理背景，不替代法律意见。

## 本章目标

学完本章，你要能回答：

1. Policy、Governance、Model Card、System Card 分别是什么？
2. 为什么模型发布需要文档化和风险披露？
3. Model card 和 dataset datasheet 的来龙去脉是什么？
4. 一个大模型 model card 应该包含哪些内容？
5. System card 和 model card 有什么区别？
6. 使用政策如何定义允许、限制和禁止场景？
7. Responsible scaling、发布门禁和安全阈值如何结合？
8. 面试中如何设计一个完整的 AI governance 流程？

## 1. 来龙去脉：为什么需要治理文档

### 1.1 早期机器学习的发布方式

早期很多模型只是论文里的实验对象。

研究者发布模型时，通常给出：

1. 模型结构。
2. 数据集名称。
3. 指标分数。
4. 训练方法。
5. 代码或权重。

这对研究复现有帮助，但对真实世界使用不够。

一个模型在 benchmark 上表现好，不代表它适合所有场景。

### 1.2 高风险场景暴露的问题

当模型进入医疗、招聘、教育、执法、金融、客服和生产工具链时，用户需要知道：

1. 模型适合什么用途？
2. 不适合什么用途？
3. 在哪些人群或场景上表现差？
4. 训练数据有什么限制？
5. 评估覆盖了哪些风险？
6. 有哪些已知失败模式？
7. 如何反馈问题？
8. 谁对发布和维护负责？

只给一个 accuracy 或 leaderboard 分数不够。

### 1.3 Datasheets for Datasets

Datasheets for Datasets 的思想是：数据集也需要说明书。

就像电子元件有 datasheet，数据集也应该记录：

1. 为什么收集。
2. 如何收集。
3. 包含什么。
4. 不包含什么。
5. 谁被代表，谁被遗漏。
6. 有什么许可和隐私限制。
7. 推荐用途和不推荐用途。

这个思想解决的是数据透明度问题。

### 1.4 Model Cards

Model Cards for Model Reporting 进一步提出：训练好的模型也应该有报告卡。

Model card 应该说明：

1. 模型基本信息。
2. 预期用途。
3. 不适用场景。
4. 训练和评估数据。
5. 指标和分层表现。
6. 风险和限制。
7. 使用建议。

这个思想解决的是模型发布透明度问题。

### 1.5 大模型时代的 System Card

大模型不是单个模型权重。

一个真实 AI 产品可能包含：

1. Base model。
2. SFT / RLHF 后训练。
3. Safety classifier。
4. RAG。
5. Tool calling。
6. Policy layer。
7. Rate limit。
8. Monitoring。
9. Human review。

所以只写 model card 不够。

还需要 system card 来描述整个系统的能力、风险、防护和部署条件。

## 2. 小白例子：药品说明书

模型发布文档可以类比药品说明书。

药品说明书会写：

1. 适应症。
2. 禁忌症。
3. 用法用量。
4. 副作用。
5. 特殊人群注意事项。
6. 保存方式。
7. 生产批次。

Model card 类似告诉用户：

1. 这个模型适合做什么。
2. 不适合做什么。
3. 已知有什么风险。
4. 在哪些场景评估过。
5. 出问题怎么处理。

没有说明书，用户就容易把模型用到不适合的地方。

## 3. Policy 是什么

Policy 是使用规则和安全边界。

它定义：

1. 哪些用途允许。
2. 哪些用途限制。
3. 哪些用途禁止。
4. 哪些场景需要人工审核。
5. 哪些行为需要记录和告警。
6. 违反规则后如何处理。

### 3.1 使用政策

使用政策面向用户和开发者。

例如：

1. 不得用于违法行为。
2. 高风险医疗建议需要专业人士审核。
3. 不得收集或泄露个人敏感信息。
4. 不得绕过系统安全限制。
5. 不得生成欺诈或冒充内容。

### 3.2 模型行为政策

模型行为政策面向模型训练和安全策略。

例如：

1. 对危险请求拒答。
2. 对边界请求澄清。
3. 对正常安全教育请求提供高层说明。
4. 对资料不足问题表达不确定。
5. 对隐私请求保护个人信息。

### 3.3 内部发布政策

内部发布政策面向组织。

例如：

1. 什么评估通过才能上线。
2. 谁有权批准发布。
3. 哪些风险必须上报。
4. 如何灰度和回滚。
5. 安全事件如何响应。

### 3.4 Radar：risk-calibrated access 与 fallback routing

Frontier model release 的一个明显趋势是：模型越强，发布治理越不像“一个模型开给所有人”，而更像“按能力、用户、任务和风险分层开放”。

可以把它抽象成 risk-calibrated access：

1. 低风险普通任务走默认模型和默认策略。
2. 高价值复杂任务可以走 thinking / multi-agent / long-running agent 模式。
3. 高风险领域任务需要 trusted access、企业审批、专家用户、额外监控或人工复核。
4. 触发危险能力、隐私、网络攻击、化学/生物、金融或医疗高风险边界时，系统需要降级、拒答、转人工或切到更保守模型。

这会带来 fallback routing：

```text
request
  -> risk classifier
  -> capability router
  -> policy gate
  -> model / tool / agent route
  -> fallback / human review / deny
```

Fallback 不只是系统故障时切备用模型，也包括安全和治理意义上的 fallback：

1. 高风险请求从强自主 Agent 退回只读回答。
2. 不确定场景从自动执行退回人工确认。
3. 危险能力边界从开放模型退回受限模型或拒答。
4. 企业数据场景从公有工具退回租户内 connector。
5. 长周期任务从全自动退回阶段性审批。

面试中可以这样表达：

```text
前沿模型发布越来越强调 risk-calibrated access。能力越强，不代表所有用户都默认拿到全部能力。治理系统要把用户身份、任务类型、数据敏感度、工具风险、模型能力和评估证据放进同一个 routing / fallback / approval 流程。否则强模型、长上下文、记忆和 Agent 工具会把单次错误放大成真实系统风险。
```

## 4. Governance 是什么

Governance 是把 policy 落地的组织和流程。

如果 policy 是“规则”，governance 是“谁负责、怎么执行、如何审计”。

它包括：

1. 决策结构。
2. 风险评估流程。
3. 发布门禁。
4. 权限管理。
5. 文档和审计。
6. 事故响应。
7. 持续监控。
8. 第三方评估。

### 4.1 为什么只写 policy 不够

很多团队会写漂亮的原则。

但没有治理流程时，原则不会自动执行。

例如：

```text
原则：高风险能力必须严格评估。
问题：谁定义高风险？谁评估？不通过能否阻断发布？业务压力下谁拍板？
```

Governance 要回答这些问题。

## 5. Model Card 应该写什么

一个大模型 model card 可以包括以下部分。

### 5.1 模型基本信息

1. 模型名称和版本。
2. 发布时间。
3. 发布方。
4. 模型类型。
5. 参数规模或大致类别。
6. 支持语言和模态。
7. 访问方式。

### 5.2 Intended Use

说明模型适合什么。

例如：

1. 通用问答。
2. 写作辅助。
3. 代码辅助。
4. 企业知识库问答。
5. 教育辅导。
6. 低风险自动化任务。

### 5.3 Out-of-Scope Use

说明模型不适合什么。

例如：

1. 无人审核的医疗诊断。
2. 高风险法律决策。
3. 自动化金融建议。
4. 关键基础设施控制。
5. 危险能力自动化。
6. 隐私敏感身份推断。

### 5.4 Training Data

说明训练数据来源类别，而不是一定公开所有细节。

包括：

1. 数据类型。
2. 时间范围。
3. 数据清洗。
4. 去重。
5. PII 处理。
6. 许可和限制。
7. 已知偏差。

### 5.5 Evaluation

包括：

1. 通用能力评估。
2. 专项能力评估。
3. 安全评估。
4. 多语言评估。
5. 分层表现。
6. 长上下文和工具调用评估。
7. 红队和危险能力评估摘要。

### 5.6 Limitations

说明已知限制：

1. 幻觉。
2. 过时知识。
3. 多语言不均衡。
4. 长上下文遗漏。
5. 复杂推理失败。
6. 安全边界不完美。
7. 对抗输入风险。

### 5.7 Safety and Mitigations

说明防护措施：

1. Safety tuning。
2. Policy layer。
3. 内容过滤。
4. Tool 权限。
5. 监控和告警。
6. 人工审核。
7. 回滚机制。

### 5.8 Contact and Feedback

说明：

1. 如何报告问题。
2. 如何提交安全漏洞。
3. 如何请求数据删除。
4. 如何查看更新日志。

## 6. System Card 和 Model Card 的区别

Model card 关注模型本身。

System card 关注完整产品系统。

| 维度 | Model Card | System Card |
|---|---|---|
| 对象 | 模型 | 模型 + 工具 + 策略 + 部署系统 |
| 重点 | 训练、评估、限制 | 系统能力、风险、防护、发布流程 |
| 场景 | 模型发布 | 产品发布或平台发布 |
| 包含内容 | 模型指标、用途、局限 | Tool、RAG、权限、监控、红队、用户影响 |

大模型应用通常更需要 system card。

因为实际风险来自系统组合，而不只是模型权重。

## 7. 使用政策如何设计

### 7.1 风险分级

可以把用途分为：

1. 允许用途。
2. 低风险用途。
3. 需要披露或限制的用途。
4. 需要人工审核的高风险用途。
5. 禁止用途。

### 7.2 行为分级

模型响应也可以分级：

1. 正常回答。
2. 澄清意图。
3. 高层安全说明。
4. 拒绝具体危险内容。
5. 拒绝并提供安全替代。
6. 触发人工审核或告警。

### 7.3 政策冲突

Policy 经常冲突。

例如：

1. Helpfulness vs harmlessness。
2. 透明披露 vs 安全保密。
3. 用户隐私 vs 滥用监控。
4. 开放访问 vs 风险控制。

政策设计必须说明冲突时优先级。

## 8. 发布门禁

发布门禁把治理要求转成上线条件。

示例：

1. 核心能力不低于基线。
2. Safety eval 通过。
3. Red team P0 问题为 0。
4. P1 问题有修复或限制。
5. 隐私泄露评估通过。
6. 危险能力未超过阈值。
7. Tool 权限和审计就绪。
8. Model card / system card 完成。
9. 监控、回滚、事故响应就绪。

门禁必须在发布前定义。

否则业务压力会影响判断。

## 9. Responsible Scaling

Responsible Scaling 的核心思想是：

```text
模型能力越强，安全要求越高。
```

当模型只是低风险工具，普通评估和安全措施可能足够。

当模型出现更强危险能力或自主性时，需要更严格要求。

例如：

1. 更强红队。
2. 第三方评估。
3. 更严格访问控制。
4. 更高模型权重安全。
5. 更强监控。
6. 可能暂停或延迟发布。

Anthropic RSP 中的 AI Safety Levels 是一种公开示例。面试中不必背具体条款，关键是讲清能力、评估、安全措施和发布决策之间的绑定关系。

## 10. 风险披露怎么写

风险披露不能只写“模型可能出错”。

应该具体说明：

1. 哪些场景容易失败。
2. 失败后果是什么。
3. 哪些用户群体受影响。
4. 评估覆盖了什么。
5. 没覆盖什么。
6. 已采取哪些缓解。
7. 使用者应该怎么做。

### 10.1 好的风险披露

例如：

```text
模型可能在长文档中遗漏中间位置证据。因此，不建议在无人审核情况下用于合同审查或高风险法律判断。若用于辅助阅读，应启用引用核查，并由专业人士复核关键结论。
```

### 10.2 差的风险披露

例如：

```text
模型可能不准确，请谨慎使用。
```

太泛，不能指导使用。

## 11. 组织治理流程

### 11.1 角色

治理通常涉及：

1. 模型团队。
2. 安全团队。
3. 评估团队。
4. 产品团队。
5. 法务和合规。
6. 隐私团队。
7. 平台和基础设施。
8. 业务负责人。

### 11.2 流程

一个简化流程：

```text
需求定义
-> 风险分类
-> 数据和模型审查
-> 离线评估
-> 红队测试
-> 风险评审
-> 发布门禁
-> 灰度上线
-> 线上监控
-> 事故响应和复盘
```

### 11.3 审计

需要记录：

1. 模型版本。
2. 数据版本。
3. 评估结果。
4. 红队发现。
5. 修复记录。
6. 发布审批。
7. 监控事件。
8. 回滚记录。

## 12. Model Card 模板

可以用下面结构。

```text
1. Model Overview
2. Intended Use
3. Out-of-Scope Use
4. Training Data Summary
5. Evaluation Summary
6. Safety Evaluation
7. Known Limitations
8. Ethical and Social Risks
9. Mitigations
10. Deployment Conditions
11. Monitoring and Feedback
12. Version History
```

每个部分都要写具体，不要只写空话。

## 13. System Card 模板

```text
1. System Overview
2. Model Components
3. Tool and RAG Components
4. User Access and Permissions
5. Safety Policies
6. Evaluation and Red Teaming
7. Privacy and Data Handling
8. Known Risks and Limitations
9. Mitigations and Guardrails
10. Release Decision and Restrictions
11. Monitoring and Incident Response
12. Update Policy
```

System card 要把模型外部系统也写清楚。

### 13.1 关键公式与治理门禁指标速查

把一次模型发布审计写成：

$$
g_i=(m_i,s_i,p_i,e_i,r_i,a_i,o_i)
$$

其中 `m_i` 是 model card，`s_i` 是 system card，`p_i` 是 policy 覆盖，`e_i` 是评估结果，`r_i` 是风险问题，`a_i` 是审批记录，`o_i` 是上线后监控和事故响应准备状态。

model card 完整度：

$$
C_{model}=
\frac{1}{|M|}\sum_{j\in M} I[c_j=1]
$$

其中 `M` 是必填章节集合，`c_j=1` 表示第 `j` 个章节有具体内容和证据，而不是空话。

system card 完整度：

$$
C_{system}=
\frac{1}{|S|}\sum_{j\in S} I[s_j=1]
$$

policy 覆盖率：

$$
C_{policy}=
\frac{|P_{covered}|}{|P_{required}|}
$$

评估切片覆盖率：

$$
C_{eval}=
\frac{1}{|E|}\sum_{k\in E} I[e_k=1]
$$

未解决风险可以按严重度加权：

$$
R_{open}=
\frac{\sum_j w_j I[u_j=1]}{\sum_j w_j}
$$

其中 `u_j=1` 表示风险问题未解决，`w_j` 是严重度权重。P0 / P1 问题通常不应该被平均指标掩盖。

高风险缓解覆盖率：

$$
C_{mit}=
\frac{\sum_j I[h_j=1\land b_j=1]}
{\sum_j I[h_j=1]}
$$

其中 `h_j=1` 表示高风险问题，`b_j=1` 表示已经修复、限制访问、加监控或有明确缓解。

审批覆盖率：

$$
C_{approve}=
\frac{|A_{approved}|}{|A_{required}|}
$$

治理发布门禁可以写成：

$$
G_{gov}=
I[
C_{model}\ge\tau_m
\land C_{system}\ge\tau_s
\land C_{policy}\ge\tau_p
\land C_{eval}\ge\tau_e
\land R_{open}\le\tau_r
\land C_{mit}\ge\tau_h
\land C_{approve}\ge\tau_a
\land G_{release}=1
]
$$

其中 `G_release` 是发布硬检查，例如 P0 未解决数为 0、隐私门禁通过、工具权限通过、监控和回滚就绪。面试中要强调：`G_gov=1` 不是说模型没有风险，而是当前证据、文档、责任和控制条件达到可发布阈值。

## 14. 真实项目中的坑

### 14.1 只写优点，不写边界

文档如果只像营销材料，就失去治理价值。

### 14.2 只写总分，不写分层

不同语言、地区、用户群体和任务类型可能差异很大。

### 14.3 不写不适用场景

用户最需要知道的往往是模型不能用于哪里。

### 14.4 文档和实际系统不一致

如果 system card 写有人工审核，但产品实际没有，这会造成治理失败。

### 14.5 发布后不更新

模型、数据、策略和风险都会变化。

Model card 和 system card 应随版本更新。

## 15. 面向专家：治理中的几个 trade-off

### 15.1 Transparency vs Security

公开更多细节有利于透明，但可能暴露攻击面。

需要区分公开版和内部版。

### 15.2 Innovation vs Risk Control

限制太多会阻碍创新，限制太少会增加风险。

需要分级发布和风险分层。

### 15.3 Open Release vs Controlled Access

开源权重促进研究和生态，但高能力模型可能需要更谨慎访问控制。

### 15.4 User Autonomy vs Safety Guardrails

用户希望自由使用，平台需要阻止高风险滥用。

### 15.5 Documentation vs Accountability

写文档不是治理本身。

如果没有责任人、门禁和审计，文档只是形式主义。

## 16. 面试官会怎么问

### 问题 1：什么是 model card？

回答要点：

1. 模型发布文档。
2. 说明模型用途、评估、限制和风险。
3. 帮助使用者理解适用边界。
4. 支持透明和负责任发布。

标准回答：

```text
Model card 是模型发布时的说明文档，记录模型基本信息、预期用途、不适用场景、训练数据概要、评估结果、分层表现、风险限制和缓解措施。它的目的不是宣传模型，而是帮助使用者理解模型适合什么、不适合什么，以及如何安全使用。
```

### 问题 2：Model card 和 system card 有什么区别？

回答要点：

1. Model card 关注模型本身。
2. System card 关注完整应用系统。
3. 大模型产品常有 RAG、工具、权限、过滤和监控。
4. 系统风险不能只由模型指标解释。

### 问题 3：如何设计模型发布门禁？

回答要点：

1. 定义能力、评估和安全阈值。
2. 核心能力不能退化。
3. P0 安全问题阻断。
4. 危险能力不超过阈值。
5. 隐私、RAG、工具权限和监控就绪。
6. 文档和审批完成。

### 问题 4：Responsible scaling 是什么？

回答要点：

1. 模型能力越强，安全要求越高。
2. 高风险能力触发更严格评估和控制。
3. 评估结果影响发布和访问。
4. 不是固定一次性规则，而是随能力演化的治理框架。

### 问题 5：为什么文档不是形式主义？

回答要点：

1. 文档把评估和风险转成可沟通信息。
2. 帮助用户理解边界。
3. 支持内部审计和责任追踪。
4. 前提是文档和真实系统一致，并随版本更新。

## 17. 标准回答模板

面试中可以这样回答：

```text
我会把 AI governance 看成把技术评估转成发布决策和持续责任的流程。Model card 记录模型本身，包括用途、训练数据概要、评估、限制和风险；system card 记录完整系统，包括 RAG、工具、权限、监控、guardrails 和事故响应。

发布前需要定义门禁，例如核心能力不退化、P0 安全问题为 0、P1 问题有缓解、危险能力未超过阈值、隐私和 prompt injection 评估通过、工具权限和回滚机制就绪。对于能力更强的模型，还需要 responsible scaling 思路，即能力越强，安全评估、访问控制和组织审批越严格。

治理不是写文档本身，而是让 policy、评估、门禁、责任人、监控和事故响应形成闭环。
```

## 18. 常见误区

### 18.1 误区：Model card 是宣传材料

纠正：它应该披露限制、风险和不适用场景。

### 18.2 误区：有 model card 就治理完成

纠正：还需要门禁、监控、责任人和事故响应。

### 18.3 误区：只要模型安全，系统就安全

纠正：RAG、工具、权限、日志和用户交互都会引入风险。

### 18.4 误区：所有风险都应公开到最细

纠正：透明和安全要平衡，公开版和内部版可以不同。

### 18.5 误区：发布门禁上线前再定

纠正：门禁应在评估前定义，避免被结果和业务压力影响。

## 19. 最小可运行治理审计 demo

下面的 demo 不涉及真实机构政策，只模拟一个发布审计表：检查 model card / system card 是否完整，policy 是否覆盖关键风险，评估切片是否充分，未解决问题是否按严重度加权，高风险问题是否缓解，审批和发布硬检查是否达标。

```python
def ratio(values):
    return round(sum(1 for value in values if value) / len(values), 3)


model_card_sections = {
    "overview": True,
    "intended_use": True,
    "out_of_scope": True,
    "training_data": True,
    "evaluation": True,
    "safety_eval": True,
    "limitations": True,
    "ethical_risks": False,
    "mitigations": True,
    "deployment_conditions": False,
    "monitoring_feedback": True,
    "version_history": True,
}

system_card_sections = {
    "system_overview": True,
    "model_components": True,
    "tool_rag_components": True,
    "user_permissions": True,
    "safety_policies": True,
    "red_teaming": True,
    "privacy_data_handling": True,
    "known_risks": True,
    "guardrails": True,
    "release_restrictions": False,
    "monitoring_incident_response": True,
    "update_policy": False,
}

required_policy_categories = {
    "harmful_content",
    "privacy",
    "prompt_injection",
    "tool_misuse",
    "high_risk_advice",
    "copyright",
    "children_safety",
    "appeals",
}
covered_policy_categories = {
    "harmful_content",
    "privacy",
    "prompt_injection",
    "tool_misuse",
    "high_risk_advice",
    "copyright",
}

eval_slices = {
    "core_capability": True,
    "safety": True,
    "privacy": False,
    "red_team": True,
    "tool_use": True,
    "rag": True,
    "multilingual": False,
    "accessibility": False,
}

risk_issues = [
    {"id": "jailbreak_regression", "severity": "P1", "weight": 4, "resolved": True, "mitigated": True},
    {"id": "privacy_logging_gap", "severity": "P1", "weight": 4, "resolved": False, "mitigated": False},
    {"id": "tool_permission_edge", "severity": "P2", "weight": 2, "resolved": False, "mitigated": True},
    {"id": "multilingual_overrefusal", "severity": "P2", "weight": 2, "resolved": False, "mitigated": False},
    {"id": "model_card_gap", "severity": "P3", "weight": 1, "resolved": False, "mitigated": False},
]

release_checks = {
    "core_quality_not_regressed": True,
    "p0_unresolved_zero": True,
    "p1_mitigated": False,
    "privacy_gate": False,
    "tool_permission_gate": True,
    "monitoring_ready": True,
    "rollback_ready": True,
    "model_card_ready": False,
    "system_card_ready": False,
    "approval_quorum": True,
}

approval_votes = {
    "model_owner": True,
    "safety_owner": True,
    "privacy_owner": False,
    "legal_owner": True,
    "incident_owner": True,
}

model_card_completion = ratio(model_card_sections.values())
system_card_completion = ratio(system_card_sections.values())
policy_coverage = round(len(covered_policy_categories) / len(required_policy_categories), 3)
eval_coverage = ratio(eval_slices.values())
unresolved_weight = sum(issue["weight"] for issue in risk_issues if not issue["resolved"])
total_weight = sum(issue["weight"] for issue in risk_issues)
severity_weighted_unresolved = round(unresolved_weight / total_weight, 3)
high_risk = [issue for issue in risk_issues if issue["severity"] in {"P0", "P1"}]
high_risk_mitigation = round(sum(issue["mitigated"] for issue in high_risk) / len(high_risk), 3)
approval_coverage = ratio(approval_votes.values())
release_gate_pass = all(release_checks.values())

metrics = {
    "model_card_completion": model_card_completion,
    "system_card_completion": system_card_completion,
    "policy_coverage": policy_coverage,
    "eval_coverage": eval_coverage,
    "severity_weighted_unresolved": severity_weighted_unresolved,
    "high_risk_mitigation": high_risk_mitigation,
    "approval_coverage": approval_coverage,
}
missing_model_card = [name for name, ok in model_card_sections.items() if not ok]
missing_system_card = [name for name, ok in system_card_sections.items() if not ok]
missing_policy = sorted(required_policy_categories - covered_policy_categories)
missing_eval = [name for name, ok in eval_slices.items() if not ok]
unresolved_issues = [issue["id"] for issue in risk_issues if not issue["resolved"]]
failed_release_checks = [name for name, ok in release_checks.items() if not ok]
governance_gates = {
    "model_card": model_card_completion >= 0.90,
    "system_card": system_card_completion >= 0.90,
    "policy": policy_coverage >= 0.85,
    "eval": eval_coverage >= 0.80,
    "risk": severity_weighted_unresolved <= 0.20,
    "high_risk": high_risk_mitigation >= 1.0,
    "approval": approval_coverage >= 0.80,
    "release_checks": release_gate_pass,
}

print("metrics=", metrics)
print("missing_model_card=", missing_model_card)
print("missing_system_card=", missing_system_card)
print("missing_policy=", missing_policy)
print("missing_eval=", missing_eval)
print("unresolved_issues=", unresolved_issues)
print("failed_release_checks=", failed_release_checks)
print("governance_gates=", governance_gates)
print("release_ready=", all(governance_gates.values()))
```

运行后可以看到：

```text
metrics= {'model_card_completion': 0.833, 'system_card_completion': 0.833, 'policy_coverage': 0.75, 'eval_coverage': 0.625, 'severity_weighted_unresolved': 0.692, 'high_risk_mitigation': 0.5, 'approval_coverage': 0.8}
missing_model_card= ['ethical_risks', 'deployment_conditions']
missing_system_card= ['release_restrictions', 'update_policy']
missing_policy= ['appeals', 'children_safety']
missing_eval= ['privacy', 'multilingual', 'accessibility']
unresolved_issues= ['privacy_logging_gap', 'tool_permission_edge', 'multilingual_overrefusal', 'model_card_gap']
failed_release_checks= ['p1_mitigated', 'privacy_gate', 'model_card_ready', 'system_card_ready']
governance_gates= {'model_card': False, 'system_card': False, 'policy': False, 'eval': False, 'risk': False, 'high_risk': False, 'approval': True, 'release_checks': False}
release_ready= False
```

这个 demo 的关键观察：

1. 审批覆盖达标不等于可以发布；文档、评估、风险缓解和硬门禁仍可能失败。
2. model card / system card 的缺口要能映射到真实证据，例如缺少部署条件或更新策略。
3. severity-weighted unresolved risk 可以防止平均指标掩盖 P1 问题。

## 20. 小练习

### 练习 1

为一个代码助手写 model card 大纲。

要求包含：用途、不适用场景、评估、安全风险、工具权限和反馈渠道。

### 练习 2

为一个企业 RAG 助手写 system card 大纲。

要求包含：RAG 权限、数据处理、prompt injection 防护、日志、监控和事故响应。

### 练习 3

设计一个模型发布门禁。

要求覆盖：能力、safety、privacy、red teaming、dangerous capability、tool use 和 rollback。

### 练习 4

解释 Responsible Scaling 为什么不是单纯“越强越不能发布”。

### 练习 5

写一段好的风险披露，说明模型在医疗建议场景的边界。

## 21. 本章总结

Policy 定义规则，governance 负责把规则落地。

Datasheets for Datasets 解决数据透明度问题，Model Cards 解决模型发布透明度问题，System Cards 解决完整 AI 系统风险沟通问题。

Model card 应包含模型信息、用途、不适用场景、训练数据概要、评估、限制、风险和缓解措施。

System card 还应覆盖 RAG、工具、权限、监控、事故响应和发布条件。

发布门禁把评估结果转成上线决策，应该在发布前定义。

Responsible scaling 的核心是模型能力越强，安全评估、访问控制、组织审批和发布要求越严格。

治理不是写文档，而是 policy、评估、门禁、责任、监控、审计和事故响应形成闭环。
