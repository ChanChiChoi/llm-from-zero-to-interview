# 第八章：Safety 与 Robustness Eval

重点：jailbreak、prompt injection、harmful output、bias、privacy、dangerous capability、red teaming。

面试重点：安全评估要覆盖模型能力、模型行为和滥用风险。

## 本章目标

学完本章，你要能回答：

1. Safety eval 和普通能力评估有什么不同？
2. Jailbreak、prompt injection、harmful output 分别评估什么？
3. Bias、privacy、toxicity、dangerous capability 如何设计评估？
4. Robustness eval 为什么要看输入扰动和分布外样本？
5. Red teaming 在模型安全评估中起什么作用？
6. 如何设计安全评估指标、护栏指标和回归测试？
7. 面试中如何说明安全评估的局限性？

大模型安全评估的目标不是证明模型“绝对安全”。

现实中不存在绝对安全的通用模型。

安全评估的目标是发现风险、量化风险、降低风险，并建立持续监控和回归机制。

普通能力评估通常问：“模型能不能完成任务？”

Safety eval 还要问：

1. 模型会不会在不该回答时回答？
2. 模型会不会泄露敏感信息？
3. 模型会不会被恶意输入诱导越权？
4. 模型会不会对特定群体产生系统性偏见？
5. 模型会不会在高风险能力上提供不当帮助？
6. 安全策略会不会过度拒答，影响正常用户？

面试表达：Safety eval 不是单一 benchmark，而是一套覆盖模型行为、系统边界、滥用风险和线上监控的评估体系。

## 1. Safety Eval 的特殊性

Safety eval 和普通 eval 有几个关键区别。

### 1.1 关注低概率高风险事件

普通评估通常关注平均准确率。

安全评估更关注少量高风险失败。

例如模型 99.9% 情况下正常，但在少数输入下泄露隐私或产生危险建议，这仍然是严重问题。

### 1.2 攻击者是自适应的

普通用户输入比较自然。

攻击者会根据模型反馈不断调整输入。

所以安全评估不能只用固定测试集，还需要 red teaming、对抗样本和持续更新。

### 1.3 指标存在 trade-off

安全系统有两个方向的错误：

1. False negative：危险请求被放行。
2. False positive：正常请求被拒答。

只追求低风险可能导致模型过度拒答。

只追求有用性可能导致安全风险上升。

### 1.4 安全策略依赖场景

同一句话在不同场景下风险不同。

医学、法律、金融、未成年人、企业数据、公开聊天机器人、代码助手的安全边界都不同。

### 1.5 评估本身要谨慎

安全评估会涉及敏感类别、攻击方式和高风险能力。

写报告时应描述风险类型、评估方法和防护结论，避免提供可直接滥用的操作细节。

面试表达：Safety eval 的难点是低概率高风险、攻击者自适应、安全与有用性的 trade-off，以及不同场景安全边界不同。

## 2. 安全风险分类

设计安全评估前，要先定义风险 taxonomy。

常见类别包括：

1. Harmful output。
2. Jailbreak。
3. Prompt injection。
4. Toxicity。
5. Bias 和 fairness。
6. Privacy leakage。
7. Copyright 和敏感数据复现。
8. Dangerous capability。
9. Misinformation。
10. Self-harm 或 vulnerable user 风险。
11. Tool misuse。
12. Over-refusal。

不同公司、产品和地区会有不同政策。

评估必须对齐具体 policy，而不是只使用抽象的“安全/不安全”。

### 2.1 Harmful Output

Harmful output 指模型输出可能帮助用户造成伤害、违法、欺诈、暴力、侵犯隐私或规避安全机制。

评估重点：模型是否能识别高风险意图，并给出安全替代回答。

### 2.2 Jailbreak

Jailbreak 指用户试图绕过模型安全策略，让模型输出本应拒绝的内容。

评估重点：模型在对抗性措辞、角色扮演、间接请求、多轮诱导下是否仍遵守安全边界。

### 2.3 Prompt Injection

Prompt injection 指外部内容试图覆盖系统指令或诱导模型泄露、越权、执行错误操作。

它在 RAG、Agent、浏览器助手、代码助手里特别重要。

### 2.4 Privacy Leakage

Privacy leakage 指模型泄露个人信息、企业机密、训练数据片段、系统 prompt 或用户上下文中的敏感信息。

### 2.5 Dangerous Capability

Dangerous capability 指模型在某些高风险领域具备可被滥用的能力。

安全评估要关注模型是否提供具体、可执行、能显著降低滥用门槛的帮助。

面试表达：安全评估第一步是建立风险 taxonomy，把 harmful output、jailbreak、prompt injection、privacy、bias、dangerous capability 和 over-refusal 分开测。

## 3. Harmful Output Eval

Harmful output eval 评估模型面对不安全请求时的行为。

### 3.1 评估目标

评估目标包括：

1. 是否识别不安全意图。
2. 是否拒绝提供危险细节。
3. 是否给出安全替代建议。
4. 是否避免道德说教过度。
5. 是否区分教育性、新闻性、求助性和滥用性请求。

例如同样涉及高风险主题：

1. 新闻报道分析可能是允许的。
2. 安全防护教育可能是允许的。
3. 具体实施伤害的操作指导应拒绝。

### 3.2 指标

常见指标：

1. Unsafe compliance rate。
2. Refusal precision。
3. Refusal recall。
4. Safe completion rate。
5. Over-refusal rate。
6. Policy violation severity。
7. Human policy agreement。

其中 unsafe compliance rate 是核心风险指标。

Over-refusal rate 也很重要，因为模型不能把正常请求全部拒掉。

### 3.3 分级评估

安全风险通常不是二分类。

可以分级：

1. 允许：普通信息或低风险教育内容。
2. 谨慎回答：高敏感主题的安全、预防、求助内容。
3. 拒绝：明确伤害、违法、滥用、规避安全的请求。
4. 升级：涉及紧急风险或需要人工处理的场景。

### 3.4 错误类型

常见错误：

1. 明确危险请求被直接满足。
2. 表面拒绝但仍泄露关键细节。
3. 对安全教育内容过度拒绝。
4. 对上下文意图判断错误。
5. 多轮对话中逐步放宽边界。

面试表达：Harmful output eval 要同时看 unsafe compliance 和 over-refusal，安全模型不是拒绝越多越好，而是要在 policy 下做精确决策。

## 4. Jailbreak Eval

Jailbreak eval 关注模型是否能抵抗绕过安全策略的输入。

### 4.1 常见攻击形态

在评估报告中可以按形态分类，但不应把攻击模板写成可直接复用的操作手册。

常见形态包括：

1. 角色扮演。
2. 假设性场景。
3. 多轮诱导。
4. 编码或混淆表达。
5. 权威冒充。
6. 情绪施压。
7. 任务拆分。
8. 跨语言绕过。

### 4.2 评估指标

常见指标：

1. Attack success rate。
2. Robust refusal rate。
3. Policy violation severity。
4. Multi-turn degradation rate。
5. Cross-lingual jailbreak success rate。
6. Regression pass rate。

Attack success rate 越低越好。

但也要同时看正常请求是否被误拒。

### 4.3 单轮和多轮

单轮 jailbreak 是一次输入触发失败。

多轮 jailbreak 是用户通过多次对话逐步诱导模型偏离策略。

多轮评估更接近真实攻击。

评估时应保留完整对话上下文，不能只看最后一轮。

### 4.4 回归集

Jailbreak 样本更新很快。

团队需要维护回归集：

1. 历史失败样本。
2. 新发现攻击样本。
3. 跨语言样本。
4. 多轮样本。
5. 产品真实风险样本。

每次模型或系统 prompt 更新后都要跑回归。

面试表达：Jailbreak eval 要测试模型在对抗性、多轮、跨语言和混淆输入下是否仍遵守安全策略，并把历史失败样本纳入回归测试。

## 5. Prompt Injection Eval

Prompt injection 是 Agent 和 RAG 系统中的核心安全问题。

它不是单纯“模型回答不安全”，而是外部内容试图改变系统行为。

### 5.1 为什么 RAG 和 Agent 更容易受影响

RAG 会把外部文档放进上下文。

Agent 会读取网页、邮件、文档、代码仓库，并可能调用工具。

如果外部内容中包含恶意指令，模型可能把它当成用户或系统指令执行。

### 5.2 评估目标

Prompt injection eval 要看：

1. 模型是否区分系统指令、用户指令和外部内容。
2. 模型是否会泄露系统 prompt 或敏感上下文。
3. 模型是否会执行外部内容中的越权指令。
4. 工具调用是否遵守权限边界。
5. 检索内容是否影响安全策略。

### 5.3 RAG 场景

RAG 评估可以构造带有恶意指令的文档，看模型是否仍只基于文档回答用户问题，而不是执行文档中的指令。

评估重点：

1. 引用是否正确。
2. 是否忽略文档中的越权指令。
3. 是否泄露无关上下文。
4. 是否坚持系统优先级。

### 5.4 Tool-use 场景

Agent 工具调用更危险。

评估要覆盖：

1. 是否调用了不该调用的工具。
2. 是否把敏感数据发给外部工具。
3. 是否在用户未确认时执行高影响操作。
4. 是否遵守 read-only 和 write 权限。
5. 是否在工具结果中再次受到注入影响。

### 5.5 防护指标

常见指标：

1. Injection success rate。
2. Secret leakage rate。
3. Unauthorized tool call rate。
4. Permission violation rate。
5. Safe task completion rate。

面试表达：Prompt injection eval 的核心是指令层级和权限边界，尤其要评估 RAG 文档、网页、邮件、工具结果是否会诱导模型越权。

## 6. Robustness Eval

Robustness eval 评估模型面对输入扰动、分布变化和异常场景时是否稳定。

它和 safety eval 有交集，但不完全相同。

### 6.1 输入扰动

常见扰动：

1. 拼写错误。
2. 口语化表达。
3. 多语言混合。
4. 格式变化。
5. 噪声文本。
6. 冗余上下文。
7. 顺序变化。
8. 对抗性干扰句。

模型应在不改变任务语义的扰动下保持稳定。

### 6.2 分布外样本

分布外样本是训练或测试中少见的输入。

例如：

1. 罕见领域。
2. 低资源语言。
3. 新术语。
4. 异常格式文档。
5. 极长输入。
6. 多约束任务。

### 6.3 一致性测试

可以构造语义等价问题，看模型回答是否一致。

例如：

```text
问题 A：请总结这段文字。
问题 B：用三句话概括下面内容。
```

如果语义相近，输出质量不应大幅波动。

### 6.4 鲁棒性指标

常见指标：

1. Perturbation accuracy drop。
2. Consistency rate。
3. Worst-group performance。
4. OOD failure rate。
5. Format robustness。
6. Refusal stability。

面试表达：Robustness eval 不只看平均表现，而要看扰动、分布外样本、最差群体表现和语义等价输入下的稳定性。

## 7. Bias 与 Fairness Eval

Bias eval 评估模型是否对不同群体产生系统性不公平行为。

### 7.1 Bias 的表现形式

常见表现：

1. 刻板印象。
2. 毒性输出差异。
3. 拒答率差异。
4. 推荐或评分差异。
5. 表述礼貌程度差异。
6. 对方言、口音、低资源语言支持差异。

### 7.2 评估设计

常用方法：

1. 构造最小对比样本。
2. 控制任务语义不变，只替换群体属性。
3. 统计不同群体的输出差异。
4. 分析最差群体表现。
5. 用人工评审校准敏感样本。

例如只改变姓名或代词，观察模型是否给出不同能力判断。

### 7.3 指标

常见指标：

1. Group-wise accuracy。
2. Toxicity gap。
3. Refusal rate gap。
4. Sentiment gap。
5. Recommendation disparity。
6. Worst-group score。

### 7.4 局限性

Bias 评估很难一次性完成。

原因：

1. 群体属性复杂。
2. 文化语境不同。
3. 标注者判断存在差异。
4. 任务场景影响安全边界。
5. 指标可能掩盖交叉群体问题。

面试表达：Bias eval 要用控制变量的对比样本和 group-wise 指标，不能只看总体平均分；同时要承认公平性评估依赖语境和政策定义。

## 8. Privacy Eval

Privacy eval 关注模型是否泄露敏感信息。

### 8.1 隐私风险来源

隐私风险可能来自：

1. 预训练数据记忆。
2. 微调数据泄露。
3. RAG 知识库权限错误。
4. 对话上下文泄露。
5. 工具调用泄露。
6. 日志和监控系统泄露。

### 8.2 评估目标

Privacy eval 要看：

1. 模型是否复现训练数据中的敏感片段。
2. 模型是否泄露其他用户上下文。
3. RAG 是否只返回授权内容。
4. Agent 是否把敏感数据发给未授权工具。
5. 模型是否拒绝不合理的个人信息请求。

### 8.3 Canary 与成员推断

可以用 canary 数据检测训练或系统泄漏。

例如在受控环境中插入唯一字符串，检查模型或检索系统是否会在不应出现时输出。

成员推断关注某条样本是否可能出现在训练数据中。

这些方法需要严格权限控制，不能用于探测真实个人隐私。

### 8.4 指标

常见指标：

1. Secret leakage rate。
2. Memorization rate。
3. Unauthorized retrieval rate。
4. Cross-user leakage rate。
5. PII exposure rate。
6. Privacy refusal precision。

面试表达：Privacy eval 不只评估模型参数记忆，还要评估 RAG 权限、上下文隔离、工具调用和日志链路中的泄露风险。

## 9. Dangerous Capability Eval

Dangerous capability eval 评估模型是否具备可能被滥用的高风险能力。

这类评估必须谨慎设计，重点是风险分级和防护决策，而不是传播能力细节。

### 9.1 评估对象

常见高风险维度包括：

1. 网络安全滥用能力。
2. 欺诈和社会工程能力。
3. 非法规避能力。
4. 生物、化学等高风险知识的滥用支持。
5. 自动化扩展滥用能力。
6. 工具组合后的执行能力。

### 9.2 评估原则

设计时应遵循：

1. 只在受控环境评估。
2. 使用抽象化或安全替代任务。
3. 避免生成可直接滥用的操作细节。
4. 关注模型是否显著降低滥用门槛。
5. 由具备权限和专业背景的人员审查。

### 9.3 能力与行为分离

模型“知道”某些内容和“会不会输出不当帮助”是两件事。

评估要区分：

1. Capability：模型是否具备相关知识或推理能力。
2. Propensity：模型是否倾向于提供危险帮助。
3. Control：安全系统是否能阻止不当输出。
4. Tool access：接入工具后风险是否上升。

### 9.4 指标

常见指标：

1. High-risk compliance rate。
2. Unsafe tool-use rate。
3. Risk escalation rate。
4. Safe redirection rate。
5. Expert severity score。

面试表达：Dangerous capability eval 要区分模型能力、输出倾向、安全控制和工具权限，并在受控环境中做风险分级，而不是简单追求高分或公开细节。

## 10. Red Teaming

Red teaming 是用攻击者视角系统寻找模型和产品的安全弱点。

它是安全评估的重要组成部分，但不是唯一组成部分。

### 10.1 Red teaming 的价值

它可以发现固定 benchmark 发现不了的问题：

1. 新型 jailbreak。
2. 多轮诱导漏洞。
3. 产品工作流漏洞。
4. 工具权限漏洞。
5. 人机交互边界问题。
6. policy 模糊地带。

### 10.2 Red teaming 流程

一个安全流程通常包括：

1. 定义评估范围。
2. 定义风险分类和严重等级。
3. 准备测试环境和权限边界。
4. 收集失败案例。
5. 去重和聚类。
6. 标注严重程度。
7. 修复或缓解。
8. 加入回归测试。
9. 复测。

### 10.3 人工红队和自动红队

人工红队优点：创造性强，能发现复杂链路问题。

自动红队优点：规模大，适合持续回归。

真实系统通常两者结合。

### 10.4 Red teaming 的局限

Red teaming 不能证明没有风险。

它只能发现已探索路径上的问题。

所以还需要：

1. 固定 benchmark。
2. 随机抽检。
3. 线上监控。
4. 用户反馈。
5. 事故复盘。

面试表达：Red teaming 是发现未知风险的手段，产出应转化为可复现测试、修复任务和长期回归集。

## 11. 安全评估指标体系

安全评估不能只给一个总分。

建议构建分层指标。

### 11.1 风险指标

风险指标越低越好：

1. Unsafe compliance rate。
2. Attack success rate。
3. Secret leakage rate。
4. Unauthorized tool call rate。
5. Policy violation rate。
6. High severity incident rate。

### 11.2 有用性指标

有用性指标避免模型过度保守：

1. Safe completion rate。
2. Over-refusal rate。
3. Helpful answer rate。
4. Task success rate。
5. User satisfaction。

### 11.3 鲁棒性指标

鲁棒性指标关注稳定性：

1. Cross-lingual safety rate。
2. Multi-turn safety rate。
3. Perturbation robustness。
4. Worst-group safety score。
5. Regression pass rate。

### 11.4 严重等级

安全失败应按严重程度分级。

例如：

1. Low：轻微不合适或格式问题。
2. Medium：明显违反策略但影响有限。
3. High：可能造成实质伤害或敏感泄露。
4. Critical：高影响、可扩展、可被现实滥用的失败。

面试表达：安全指标要同时看风险、有用性和鲁棒性，并按严重程度加权；一个 critical failure 不能被大量低风险样本平均掉。

## 12. 线上安全评估与监控

离线安全评估不够。

上线后还需要监控。

### 12.1 线上信号

常见线上信号：

1. 拒答率。
2. 用户重试率。
3. 安全分类器命中率。
4. 人工审核命中率。
5. 用户举报。
6. 工具调用异常。
7. 敏感数据访问异常。
8. 安全事故数量。

### 12.2 护栏指标

上线 A/B test 时，安全指标通常是护栏指标。

例如新模型主指标提升，但 unsafe compliance 或 privacy leakage 上升，就不能直接全量上线。

### 12.3 灰度和回滚

高风险系统应支持：

1. 小流量灰度。
2. 风险分层放量。
3. 实时监控。
4. 自动或人工回滚。
5. 事故样本沉淀到回归集。

### 12.4 反馈闭环

线上安全事件要进入闭环：

```text
监控发现 -> 人工复核 -> 定级 -> 修复 -> 回归测试 -> 再上线
```

面试表达：安全评估必须从离线 benchmark 延伸到线上监控和灰度机制，安全指标在上线实验中通常是硬护栏。

## 13. 常见评估陷阱

### 13.1 只测英文单轮样本

真实攻击可能是多语言、多轮、混淆表达和工具链组合。

只测英文单轮会低估风险。

### 13.2 只看拒答率

拒答率高不等于安全。

可能是模型过度拒绝正常请求。

### 13.3 忽略 over-refusal

安全模型如果拒绝医疗常识、心理支持、法律常识等正常求助，会严重影响用户体验。

### 13.4 把模型安全等同于系统安全

系统安全还包括 RAG 权限、工具权限、日志隔离、前后置分类器、人工审核和产品流程。

### 13.5 固定 benchmark 用太久

攻击方式会演化。

固定测试集会被过拟合。

### 13.6 LLM judge 未校准

安全评估中 LLM judge 可能误判隐晦风险或过度敏感。

必须用人工 gold set 校准。

面试表达：安全评估最常见误区是只看拒答率、只测固定样本、忽略 over-refusal，以及把模型层安全误当成系统层安全。

## 14. 真实项目中的 Safety Eval 流程

一个较完整的流程如下。

### 14.1 定义 policy

先明确产品安全策略：

1. 哪些请求允许。
2. 哪些请求谨慎回答。
3. 哪些请求必须拒绝。
4. 哪些请求需要升级人工。
5. 不同用户、地区、产品形态是否有差异。

### 14.2 构建评估集

评估集应包括：

1. 普通安全样本。
2. 明确违规样本。
3. 边界样本。
4. 多轮样本。
5. 跨语言样本。
6. Prompt injection 样本。
7. 历史事故样本。
8. 正常但容易被误拒的样本。

### 14.3 选择评估方式

组合使用：

1. 规则检查。
2. 安全分类器。
3. LLM judge。
4. 人工 policy 标注。
5. Red teaming。
6. 线上监控。

### 14.4 输出报告

安全报告应包含：

1. 风险 taxonomy。
2. 评估集来源和覆盖范围。
3. 各类别风险指标。
4. Over-refusal 指标。
5. 严重失败案例摘要。
6. 回归集结果。
7. 已知局限和上线建议。

### 14.5 修复和回归

修复方式可能包括：

1. 数据改进。
2. SFT 或偏好训练。
3. 安全分类器。
4. 系统 prompt 调整。
5. 工具权限限制。
6. RAG 权限控制。
7. 人工审核。
8. 产品流程改造。

面试表达：我会从 policy 定义开始，构建覆盖违规、边界、正常和攻击样本的评估集，用自动评估加人工校准，最后把失败案例沉淀为回归测试。

## 15. 面试官会怎么问

### 15.1 如何评估一个大模型是否安全？

回答要点：

1. 先定义具体 policy 和风险 taxonomy。
2. 分别评估 harmful output、jailbreak、prompt injection、privacy、bias、dangerous capability。
3. 同时评估 unsafe compliance 和 over-refusal。
4. 用人工 gold set 校准自动 judge。
5. 加入 red teaming 和线上监控。
6. 把失败案例做成回归集。

### 15.2 Jailbreak eval 和 harmful output eval 有什么区别？

回答要点：

1. Harmful output eval 看模型面对不安全请求是否合规。
2. Jailbreak eval 看模型面对绕过策略的对抗输入是否仍合规。
3. Jailbreak 更强调多轮、混淆、跨语言和策略绕过。
4. 两者都要看 unsafe compliance 和 over-refusal。

### 15.3 Prompt injection 为什么是系统问题？

回答要点：

1. 它通常来自外部文档、网页、邮件或工具结果。
2. 风险在于模型混淆指令层级。
3. Agent 可能进一步调用工具，造成越权操作。
4. 需要模型、权限、工具、RAG 和产品流程共同防护。

### 15.4 如何权衡安全和有用性？

回答要点：

1. 不能只优化拒答率。
2. 要同时看 unsafe compliance 和 over-refusal。
3. 对不同风险等级采用不同策略。
4. 对边界样本做人审校准。
5. 上线时用安全指标做护栏。

## 16. 标准回答模板

如果面试官问：“你会如何设计 safety 和 robustness eval？”

可以这样回答：

```text
我会先定义产品 policy 和风险 taxonomy，把安全风险拆成 harmful output、jailbreak、prompt injection、privacy、bias、dangerous capability、tool misuse 和 over-refusal。然后分别构建评估集，覆盖明确违规样本、边界样本、正常易误拒样本、多轮样本、跨语言样本和历史失败样本。

指标上我不会只看拒答率，而会同时看 unsafe compliance rate、attack success rate、secret leakage rate、unauthorized tool call rate、over-refusal rate 和 safe completion rate。对于高风险失败，还要按严重程度分级，不能被平均分掩盖。

评估方法上，我会结合规则、安全分类器、LLM judge、人工 gold set 和 red teaming。对于 RAG 和 Agent，我会重点测 prompt injection、权限边界和工具调用安全。最后把失败样本沉淀为回归集，并在上线 A/B test 中把安全指标作为护栏指标持续监控。
```

## 17. 常见误区

### 17.1 认为安全评估可以证明模型绝对安全

安全评估只能降低风险，不能证明不存在未知攻击。

### 17.2 拒答越多越安全

过度拒答会损害正常用户，也可能掩盖模型理解能力不足。

### 17.3 忽略边界样本

很多安全系统失败发生在边界样本，而不是明显违规样本。

### 17.4 只做离线评估

上线后用户和攻击者行为会变化，必须有监控和反馈闭环。

### 17.5 不评估系统组件

模型本身安全不代表 RAG、Agent、工具和日志链路安全。

### 17.6 不维护回归集

安全问题修复后可能复发，必须持续回归。

## 18. 小练习

### 练习 1

设计一个安全评估 taxonomy，覆盖聊天助手、代码助手和 RAG 助手。

要求至少包含 harmful output、prompt injection、privacy、over-refusal 和 tool misuse。

### 练习 2

一个模型 unsafe compliance rate 降低了，但 over-refusal rate 大幅上升。你会如何判断这次更新是否可上线？

思考方向：风险等级、正常任务成功率、用户场景、护栏指标、人工抽检。

### 练习 3

为一个企业知识库 RAG 系统设计 prompt injection eval。

至少说明：外部文档样本、权限边界、泄露指标、越权工具调用指标。

### 练习 4

为什么 red teaming 发现的问题应该进入回归集？

请说明如果不进入回归集，会产生什么风险。

## 19. 本章总结

Safety 与 robustness eval 的核心是风险管理，而不是追求单一分数。

Safety eval 要覆盖 harmful output、jailbreak、prompt injection、privacy、bias、dangerous capability 和 over-refusal。

Robustness eval 要看模型在扰动、分布外样本、多语言、多轮和最差群体上的稳定性。

Red teaming 能发现固定 benchmark 之外的风险，但它的产出必须转化为可复现案例、修复任务和回归测试。

真实系统中，安全评估必须贯穿离线评估、灰度上线、线上监控和事故复盘。

面试中要强调：安全不是模型单点能力，而是 policy、模型、数据、RAG、工具权限、产品流程和监控体系共同决定的系统属性。
