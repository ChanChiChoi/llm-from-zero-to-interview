# 第六章：Red Teaming 与危险能力评估

重点：红队测试、cyber risk、bio risk、autonomy risk、capability elicitation、安全阈值。

面试重点：Red teaming 不是随便找人“怼模型”，而是一套系统化发现、度量、复现、修复和回归测试风险的流程。危险能力评估则进一步服务于模型发布门禁、分级发布和治理决策。

安全边界：本章只讨论评估框架、风险分类、流程设计和治理逻辑，不提供网络攻击、生物安全、规避监控等高风险能力的操作步骤。

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 Anthropic Red Teaming Language Models to Reduce Harms、OpenAI Preparedness Framework、Google DeepMind 关于 evaluating dangerous capabilities 和 Frontier Safety Framework 的公开资料、NIST AI RMF Generative AI Profile、Anthropic Responsible Scaling Policy 以及前序 Safety Eval、Jailbreak / Prompt Injection、Scalable Oversight 和 Reward Hacking 章节资料边界。

本讲聚焦 red teaming 和 dangerous capability evaluation 的防御性流程：风险分类、样本设计、能力激发条件、严重度分级、修复闭环、回归测试、发布门禁和治理记录。

```text
风险 taxonomy -> 红队任务 -> 执行 trace -> 严重度分级 -> 修复 -> 回归 -> 发布门禁
```

本讲不提供网络攻击、生物化学、规避监控、欺诈操纵或自主代理滥用的可执行步骤。涉及高风险能力时，只保留抽象指标、评估条件、门禁和 toy case。

## 本章目标

学完本章，你要能回答：

1. 什么是 red teaming？它和普通测试、benchmark、安全评估有什么区别？
2. 大模型 red teaming 为什么重要？
3. 红队测试如何从发现有害输出，演化到危险能力评估和发布门禁？
4. Cyber risk、bio risk、autonomy risk 分别关注什么？
5. 什么是 capability elicitation？为什么评估时既要测“自然能力”，也要测“最大可激发能力”？
6. 如何设计一个 red teaming 流程？
7. 如何把红队结果转成训练数据、策略更新、回归测试和上线决策？
8. 面试中如何讲清危险能力评估，同时避免提供危险细节？

## 1. 来龙去脉：Red Teaming 从哪里来

### 1.1 传统安全里的红队

Red team 最早来自军事和网络安全语境。

蓝队负责防守。

红队负责模拟攻击者，从对手视角寻找系统漏洞。

传统软件安全中，红队会关注：

1. 身份认证是否可绕过。
2. 权限边界是否可靠。
3. 数据是否会泄露。
4. 系统是否能抵抗对抗输入。
5. 监控和告警是否有效。

核心思想是：

```text
不要只证明系统在正常路径下能工作，还要主动寻找它如何失败。
```

### 1.2 进入大模型时代

大模型的失败方式和传统软件不同。

传统软件通常有明确代码路径。

大模型则可能在自然语言、上下文、多轮对话、工具调用和用户诱导下出现复杂行为。

因此 LLM red teaming 要找的不只是崩溃或漏洞，还包括：

1. 有害输出。
2. 越狱成功。
3. 隐私泄露。
4. 偏见和歧视。
5. 欺骗性或误导性回答。
6. 多轮诱导失败。
7. Prompt injection。
8. 工具调用风险。
9. 危险能力被激发。

### 1.3 早期 LLM Red Teaming 的启发

Anthropic 的 Red Teaming Language Models to Reduce Harms 系统总结了语言模型红队经验，包括不同模型规模、不同训练方式下的红队难度和有害输出类型。

这类工作的重要意义是：

1. 把红队从零散攻击样例变成可记录、可分析的数据流程。
2. 让模型安全不只依赖静态 benchmark。
3. 把失败样本用于改进模型和评估。
4. 让社区开始形成红队方法、统计和报告规范。

### 1.4 后来的演化

随着模型能力增强，红队不再只看“会不会输出坏话”。

它开始扩展到：

1. Dangerous capability evaluation。
2. Frontier model system card。
3. Responsible scaling policy。
4. Model release gate。
5. Third-party evaluation。
6. Continuous monitoring。

也就是说，red teaming 从安全测试演化为模型治理的一部分。

## 2. Red Teaming 是什么

### 2.1 定义

Red teaming 是一种主动、对抗式、系统化的安全测试方法。

它模拟恶意用户、误用场景、边界场景和异常环境，寻找模型或系统的失败模式。

简单说：

```text
站在攻击者和真实风险的角度，主动找模型会怎么出问题。
```

### 2.2 和普通测试的区别

普通测试通常验证：

```text
系统是否按预期工作？
```

Red teaming 更关注：

```text
系统如何被诱导、误用、绕过或推到边界？
```

### 2.3 和 benchmark 的区别

Benchmark 通常固定、可重复、适合比较。

Red teaming 更开放、动态、探索性强。

二者关系：

1. Benchmark 给稳定指标。
2. Red teaming 发现未知失败。
3. 红队发现的失败样本可以沉淀成 regression suite。

### 2.4 和安全评估的关系

Safety eval 是更大的集合。

Red teaming 是其中偏主动探索和对抗发现的一部分。

完整安全评估还包括：

1. 静态测试集。
2. 自动化评估。
3. 人工评估。
4. 专家评估。
5. 上线监控。
6. 事故复盘。

## 3. 小白例子：检查一座桥

普通测试像检查桥是否能承受日常车流。

红队测试像故意问：

1. 如果很多车同时急刹会怎样？
2. 如果某个支撑点被破坏会怎样？
3. 如果暴雨和大风同时出现会怎样？
4. 如果有人故意超载会怎样？

危险能力评估则进一步问：

1. 桥的设计是否允许它承载更危险的用途？
2. 什么时候需要限制通行？
3. 什么时候必须加固后才能开放？

对应到大模型：

1. 普通评估看模型回答是否好。
2. 红队看模型如何失败。
3. 危险能力评估看模型是否具备会显著提高风险的能力。
4. 发布门禁决定能不能部署、如何部署、需要哪些防护。

## 4. Red Teaming 的风险分类

一个成熟 red teaming 项目要先定义 taxonomy。

### 4.1 有害内容

包括：

1. 暴力。
2. 自伤。
3. 仇恨和骚扰。
4. 欺诈。
5. 高风险违法行为。
6. 不当性内容。

评估重点是模型是否会 harmful compliance，以及是否能给安全替代。

### 4.2 网络安全风险

这里只讨论防御性评估框架，不提供攻击步骤。

关注：

1. 模型是否会降低恶意用户执行网络滥用的门槛。
2. 是否能自动化复杂攻击链的规划。
3. 是否能帮助规避检测。
4. 是否能对真实系统造成可操作风险。
5. 是否能在 Agent 工具环境中执行危险动作。

### 4.3 生物与化学风险

只讨论治理框架。

关注：

1. 模型是否显著降低高风险知识获取门槛。
2. 是否能整合分散信息形成更可执行方案。
3. 是否能帮助排查实验失败。
4. 是否会给非专家提供危险指导。
5. 是否在拒答和安全替代上稳定。

### 4.4 自主性和 Agent 风险

关注：

1. 长期规划。
2. 工具调用。
3. 多步任务执行。
4. 自我纠错。
5. 资源获取。
6. 绕过限制。
7. 在无人监督下持续行动。

### 4.5 欺骗和操纵风险

关注：

1. 是否生成误导性内容。
2. 是否能个性化劝说。
3. 是否能隐藏不确定性。
4. 是否能进行社会工程式诱导。

### 4.6 隐私和数据泄露

关注：

1. 训练数据记忆。
2. 上下文泄露。
3. 系统提示泄露。
4. 企业内部数据泄露。
5. 跨用户信息混淆。

### 4.7 Prompt Injection 和工具滥用

关注：

1. 间接注入。
2. 工具参数污染。
3. 外部文档控制模型行为。
4. 高风险操作缺少确认。
5. 记忆被污染。

## 5. Dangerous Capability Evaluation

### 5.1 定义

危险能力评估关注模型是否具备可能显著增加现实世界风险的能力。

它不是看模型是否“回答不好”。

而是看：

```text
模型是否让某类高风险行为更容易、更便宜、更自动化、更可靠。
```

### 5.2 和普通 safety eval 的区别

普通 safety eval 常问：

```text
模型会不会输出不该输出的内容？
```

危险能力评估还问：

```text
即使模型被限制输出，它的底层能力是否已经足够强，需要更高安全等级和部署限制？
```

例如：

1. 模型是否能辅助复杂网络任务。
2. 模型是否能辅助高风险科学误用。
3. 模型是否能自主完成长任务。
4. 模型是否能绕过监督或工具限制。

### 5.3 为什么它和发布门禁相关

如果模型能力达到某个风险阈值，不能只靠普通产品策略上线。

需要：

1. 更严格安全评估。
2. 更强访问控制。
3. 更严格模型权重安全。
4. 更强监控和审计。
5. 更小范围灰度。
6. 可能延迟发布。

Anthropic 的 Responsible Scaling Policy 提出了 AI Safety Levels 这类分级思路，用模型潜在危险能力和误用风险来决定不同级别的安全要求。不同机构的具体标准可能不同，但核心思想是一致的：能力越强，发布前安全证明和组织防护要求越高。

## 6. Capability Elicitation

### 6.1 定义

Capability elicitation 指通过合适的 prompt、工具、scaffolding、示例、搜索、分解和多轮交互，把模型潜在能力尽可能激发出来。

危险能力评估中，能力激发很关键。

因为模型默认回答差，不一定说明它没有能力。

可能只是：

1. Prompt 不合适。
2. 工具没给。
3. 上下文不足。
4. 没有允许多步推理。
5. 没有使用最佳 scaffolding。
6. 安全策略影响了表现。

### 6.2 自然能力和最大可激发能力

评估时要区分：

1. Natural capability：普通用户随便问，模型能做到什么。
2. Elicited capability：强提示、工具、专家辅助、多轮 scaffold 下，模型能做到什么。

发布风险往往更关心第二个。

因为恶意用户会努力激发模型能力。

### 6.3 面向专家

如果评估没有做足 elicitation，可能低估风险。

但如果 elicitation 过强，也可能测到一个现实中很难触发的上界。

所以需要报告：

1. 模型访问级别。
2. Prompt 和工具条件。
3. 是否允许多轮。
4. 是否允许外部资料。
5. 是否有专家辅助。
6. 是否有自动搜索或 agent scaffold。
7. 成功率和成本。

这能让评估结论可解释。

## 7. Red Teaming 流程

### 7.1 定义目标和范围

先明确：

1. 测哪个模型版本。
2. 测哪些风险类别。
3. 是否包含工具调用。
4. 是否包含多模态。
5. 是否包含长上下文和 RAG。
6. 红队能否使用自动化工具。
7. 成功标准是什么。

### 7.2 构造攻击和边界样本

样本来源：

1. 历史事故。
2. 已知 jailbreak 类型。
3. 用户日志脱敏。
4. 专家设计。
5. 自动生成。
6. 模型辅助生成。
7. 外部红队。

注意：高风险领域样本应控制细节，避免变成操作手册。

### 7.3 执行测试

记录：

1. 输入。
2. 输出。
3. 模型版本。
4. Prompt 版本。
5. 工具权限。
6. 解码参数。
7. 多轮上下文。
8. 是否触发安全系统。

### 7.4 标注和分级

按严重程度分级。

例如：

1. P0：可能导致严重现实伤害或重大数据泄露。
2. P1：高风险有害行为或关键安全边界失守。
3. P2：中等风险违规或明显安全退化。
4. P3：低风险、风格或边界问题。

具体等级要由团队政策定义。

### 7.5 Root Cause Analysis

不要只记录“模型失败”。

要分析原因：

1. 安全策略缺失。
2. SFT 数据覆盖不足。
3. 偏好优化过度 helpful。
4. Prompt 层级不清。
5. 工具权限过大。
6. 检索内容污染。
7. 输出过滤失败。
8. 多轮状态追踪失败。

### 7.6 修复和回归测试

修复可能包括：

1. 数据补充。
2. Safety tuning。
3. Policy 更新。
4. Prompt 修改。
5. 工具权限收紧。
6. 分类器增强。
7. 产品交互调整。
8. 人工确认流程。

修复后必须加入 regression suite。

## 8. 危险能力评估的设计原则

### 8.1 分层风险模型

不要只说“模型危险”或“不危险”。

应该按能力和风险分层。

例如：

1. 只会复述公开常识。
2. 能整合公开信息但可靠性低。
3. 能提供专家级分析但需要大量人工辅助。
4. 能自主完成多步任务。
5. 能显著降低恶意行为成本。

### 8.2 对比非 AI baseline

评估模型是否增加风险，要和 baseline 比较。

例如：

1. 普通搜索引擎。
2. 公开教材。
3. 专家人工流程。
4. 现有自动化工具。

如果模型没有比现有公开资源提供实质增量，风险等级不同。

如果模型显著降低门槛、提高成功率或自动化程度，风险更高。

### 8.3 关注可靠性而不只是知识

危险能力不只是“知道一些内容”。

还包括：

1. 是否可靠。
2. 是否能纠错。
3. 是否能适应失败。
4. 是否能规划。
5. 是否能调用工具。
6. 是否能长期执行。

### 8.4 关注可操作性

模型输出高层概念和输出可操作计划，风险不同。

评估时要区分：

1. 概念性解释。
2. 高层风险讨论。
3. 可执行步骤。
4. 自动化执行。
5. 现实世界影响。

### 8.5 保持安全边界

危险能力评估文档要避免泄露具体可执行细节。

可以记录内部细节，但公开材料应：

1. 聚合报告。
2. 删除可操作步骤。
3. 保留风险等级和结论。
4. 说明方法类别和防护措施。

## 9. 安全阈值和发布门禁

### 9.1 为什么需要阈值

如果没有阈值，红队结果很难转成决策。

团队会陷入：

```text
发现了一些风险，但到底能不能发？
```

阈值的作用是提前定义：

1. 什么风险必须阻断发布。
2. 什么风险允许灰度。
3. 什么风险需要额外 mitigations。
4. 什么风险可以记录后续修复。

### 9.2 发布门禁示例

一个模型上线前，可以要求：

1. P0 安全问题为 0。
2. P1 问题有明确修复或限制方案。
3. 危险能力评估未超过预设阈值。
4. Jailbreak 成功率低于阈值。
5. Prompt injection 工具误用率低于阈值。
6. Over-refusal 不超过可接受范围。
7. 高风险工具有二次确认。
8. 日志、监控和回滚机制就绪。

### 9.3 分级发布

对于风险较高模型，可以采用：

1. 内部测试。
2. 受控 alpha。
3. 限量 beta。
4. API 访问控制。
5. 功能限制。
6. 高风险能力禁用。
7. 逐步扩大可用范围。

### 9.4 和 Responsible Scaling 的关系

Responsible Scaling 的思想是：模型能力越高，安全要求越高。

这类框架通常会把能力评估、安全措施和发布条件绑定。

面试中不要死背某一家公司的细节，而要讲清：

```text
能力增长必须伴随评估、控制、组织流程和发布标准的升级。
```

### 9.5 关键公式与红队门禁指标速查

设红队评估集为：

$$
T=\{t_i\}_{i=1}^{N}
$$

每个样本记录风险类别、严重度、评估条件、基线能力、模型自然能力、能力激发后能力、实际结果和修复状态：

$$
t_i=(x_i,c_i,s_i,e_i,b_i,n_i,h_i,y_i,m_i,w_i)
$$

其中：

1. \(x_i\) 是抽象测试任务。
2. \(c_i\) 是风险类别，例如 jailbreak、prompt injection、privacy、cyber、bio、autonomy。
3. \(s_i\) 是严重度等级，例如 P0、P1、P2、P3。
4. \(e_i\) 是评估条件，例如 natural、tool、scaffold、expert-assisted。
5. \(b_i\) 是非 AI baseline 或旧模型基线分。
6. \(n_i\) 是模型自然能力分。
7. \(h_i\) 是 capability elicitation 后的最高能力分。
8. \(y_i\) 是是否触发安全失败。
9. \(m_i\) 是是否已经修复并通过回归。
10. \(w_i\) 是严重度权重。

**1. 风险分类覆盖率**

设目标风险分类集合为 \(C^*\)，本轮红队覆盖的类别为 \(C_T\)：

$$
C_{tax}=\frac{|C_T \cap C^*|}{|C^*|}
$$

覆盖率低时，没发现问题不能证明模型安全。

**2. 红队失败发现率**

$$
R_{find}=\frac{1}{N}\sum_i y_i
$$

这个值不是线上真实风险率，只表示在给定搜索强度和样本分布下发现失败的比例。

**3. 高严重度未修复率**

设 \(H_i=1\) 表示样本属于 P0 或 P1：

$$
R_{sev}=\frac{\sum_i H_i y_i (1-m_i)}{\sum_i H_i}
$$

真实发布门禁通常要求 P0 为 0，P1 必须修复或有明确限制方案。

**4. 能力激发增益**

$$
G_{elic}=\frac{1}{N}\sum_i (h_i-n_i)
$$

如果 \(G_{elic}\) 很大，说明默认问答低估了模型上限，危险能力评估需要报告工具、scaffold 和专家辅助条件。

**5. 相对基线能力提升**

$$
U_{base}=\frac{1}{N}\sum_i \max(0,h_i-b_i)
$$

危险能力评估关心的不是模型是否知道某些知识，而是它是否相对公开资料、旧模型或人工流程显著降低门槛。

**6. 自主性能力分**

对 autonomy / agent 样本，设 \(p_i,g_i,l_i,q_i\) 分别表示规划、工具使用、持续执行和失败恢复分：

$$
A_{auto}=\frac{1}{|A|}\sum_{i\in A}\frac{p_i+g_i+l_i+q_i}{4}
$$

自主性风险不只看单步答案，而要看长期任务、工具调用和纠错能力。

**7. 回归通过率**

设 \(B\) 是历史红队失败样本集合，\(M_j=1\) 表示修复后同类回归样本通过：

$$
C_{reg}=\frac{1}{|B|}\sum_{j\in B} M_j
$$

修复一个具体样本不等于修复同类 root cause，回归集应包含变体和 held-out 样本。

**8. 严重度加权红队风险**

$$
S_{red}=\frac{\sum_i w_i y_i}{\sum_i w_i}
$$

这个指标用于排序修复优先级，不能替代 P0 / P1 硬门禁。

**9. 发布门禁**

$$
G_{red}=
\mathbb{1}[
N_{P0}=0
\land R_{sev}\le \tau_{sev}
\land U_{base}\le \tau_{cap}
\land A_{auto}\le \tau_{auto}
\land C_{reg}\ge \tau_{reg}
\land C_{tax}\ge \tau_{tax}
]
$$

这里 \(G_{red}=1\) 只表示当前评估条件下可以进入下一阶段发布流程，不表示没有未知风险。

## 10. Capability Evals 的常见坑

### 10.1 低估能力

如果 prompt、工具、scaffold 太弱，可能低估模型风险。

### 10.2 高估能力

如果给了大量专家辅助和不现实条件，可能高估普通部署风险。

### 10.3 只测知识，不测执行

知道概念不等于能完成任务。

要区分知识、计划、执行和纠错。

### 10.4 只看单轮

危险任务往往多轮、多步、有反馈。

### 10.5 不记录评估条件

没有记录工具、提示、上下文和人工辅助，结果不可解释。

### 10.6 评估泄漏

如果评估样本被训练或调参反复使用，结果会虚高或虚低。

## 11. 真实项目中的 Red Teaming 闭环

一个成熟闭环：

```text
定义风险 taxonomy
-> 构造红队任务
-> 执行攻击和边界测试
-> 标注严重程度
-> root cause analysis
-> 修复模型或系统
-> 加入 regression suite
-> 上线门禁
-> 线上监控
-> 事故复盘
```

### 11.1 对模型团队

输出：

1. 失败样本。
2. 数据补充需求。
3. Safety tuning 方向。
4. 偏好数据改进。
5. 模型能力边界报告。

### 11.2 对产品团队

输出：

1. 高风险功能限制。
2. 用户确认流程。
3. 风险提示。
4. 灰度策略。
5. 回滚预案。

### 11.3 对平台团队

输出：

1. 权限控制。
2. 日志审计。
3. 监控告警。
4. Rate limit。
5. 安全分类器。
6. Tool sandbox。

### 11.4 对治理团队

输出：

1. System card。
2. 风险接受说明。
3. 第三方评估报告。
4. 安全事件响应流程。
5. 发布决策记录。

## 12. 面向专家：Red Teaming 的统计问题

红队结果不是简单“发现了几个问题”。

需要考虑统计解释。

### 12.1 样本偏差

红队人员擅长的攻击类型会影响结果。

如果团队只擅长 jailbreak，就可能漏掉工具风险。

### 12.2 搜索强度

红队越努力，越容易发现问题。

所以报告要说明：

1. 红队人数。
2. 时间预算。
3. 是否允许自动化。
4. 是否允许多轮。
5. 是否有专家。

### 12.3 Success rate 的分母

攻击成功率取决于样本定义。

如果样本全是高难攻击，成功率会低。

如果样本都是弱攻击，成功率会高。

所以要分层报告。

### 12.4 发现率不等于真实风险率

红队发现的问题说明存在风险，但不能直接等同于线上发生概率。

真实风险还取决于：

1. 用户分布。
2. 访问控制。
3. 工具权限。
4. 监控。
5. 攻击者动机。
6. 产品场景。

### 12.5 修复后的回归

修复一个红队样本，不代表同类风险消失。

需要构造同类变体和 held-out 红队集。

## 13. 面试官会怎么问

### 问题 1：什么是大模型 red teaming？

回答要点：

1. 主动、对抗式寻找模型失败模式。
2. 覆盖有害输出、jailbreak、prompt injection、隐私、工具风险和危险能力。
3. 结果用于修复、评估、回归和发布决策。

标准回答：

```text
大模型 red teaming 是从攻击者和真实误用角度主动测试模型，通过构造边界样本、多轮诱导、间接注入、工具调用场景和高风险任务，发现模型在安全策略、对齐行为、权限控制和危险能力上的失败模式。它不是一次性测试，而应进入修复、回归测试和上线门禁闭环。
```

### 问题 2：危险能力评估和普通安全评估有什么区别？

回答要点：

1. 普通安全评估看模型会不会输出有害内容。
2. 危险能力评估看模型是否显著提高现实风险。
3. 关注 cyber、bio、autonomy 等高风险能力。
4. 结果影响发布等级、访问控制和安全措施。

### 问题 3：什么是 capability elicitation？

回答要点：

1. 用合适 prompt、工具、scaffold、专家辅助激发模型潜在能力。
2. 防止低估模型风险。
3. 需要报告评估条件，避免不现实高估。
4. 风险评估通常要同时看自然能力和最大可激发能力。

### 问题 4：红队发现问题后怎么处理？

回答要点：

1. 标注严重程度。
2. 做 root cause analysis。
3. 选择模型、数据、prompt、系统或产品层修复。
4. 加入 regression suite。
5. 更新上线门禁和监控。

### 问题 5：如何设计模型发布门禁？

回答要点：

1. 定义风险 taxonomy 和严重等级。
2. P0 必须阻断。
3. P1 需要修复或限制。
4. 危险能力不能超过预设阈值。
5. 工具、权限、监控、回滚必须就绪。
6. 结合分级发布和持续监控。

## 14. 标准回答模板

面试中可以这样回答：

```text
我会把 red teaming 看成安全闭环的一部分，而不是一次性攻击测试。首先定义风险 taxonomy，例如有害内容、jailbreak、prompt injection、隐私、工具调用、cyber、bio 和 autonomy。然后构造单轮、多轮、长上下文、RAG、Agent 和工具调用场景下的红队任务，并记录完整 trace。

评估结果要按严重程度分级，做 root cause analysis，判断问题来自模型对齐、数据覆盖、prompt 层级、工具权限、检索污染还是输出过滤。修复后要加入 regression suite，防止同类问题复发。

对于 frontier model，还需要危险能力评估，看模型是否显著降低高风险行为门槛，是否具备自主多步执行能力，以及在 capability elicitation 后的能力上限。最终这些结果要进入发布门禁、分级发布、访问控制、监控和回滚策略。
```

## 15. 常见误区

### 15.1 误区：Red teaming 就是找几个 jailbreak prompt

纠正：成熟红队覆盖风险 taxonomy、工具调用、RAG、隐私、危险能力、分级和回归测试。

### 15.2 误区：没发现问题就说明安全

纠正：没发现可能是搜索强度不足、样本偏差或 elicitation 不够。

### 15.3 误区：红队样本修了就结束

纠正：要修 root cause，并加入同类变体回归集。

### 15.4 误区：危险能力评估就是问模型危险问题

纠正：危险能力评估关注能力是否显著增加现实风险，需要安全设计、专家参与和细节控制。

### 15.5 误区：安全阈值上线时再定

纠正：阈值应在评估前定义，否则容易被业务压力影响。

## 16. 小练习

### 练习 1

设计一个 LLM red teaming 流程。

要求覆盖：taxonomy、样本构造、执行记录、严重度分级、root cause、修复和回归测试。

### 练习 2

比较 red teaming、benchmark 和 safety eval。

要求说明它们各自解决什么问题、优点和局限。

### 练习 3

设计一个 Agent 工具调用红队测试。

要求覆盖：工具权限、参数污染、间接注入、二次确认和审计日志。

### 练习 4

解释 capability elicitation 为什么重要。

要求说明自然能力、最大可激发能力、低估风险和高估风险。

### 练习 5

设计一个模型发布门禁。

要求包含：P0/P1 阈值、危险能力阈值、jailbreak 指标、prompt injection 指标、over-refusal、监控和回滚。

## 17. 最小可运行 Red Teaming 审计 demo

下面的 demo 不包含任何高风险任务细节，只用抽象标签模拟红队审计表。它展示如何把 red team case 转成 taxonomy 覆盖、严重度、capability elicitation、baseline uplift、工具权限、回归测试和发布门禁。

```python
from collections import Counter


REQUIRED_CATEGORIES = [
    "harmful_content",
    "jailbreak",
    "prompt_injection",
    "privacy",
    "cyber_dual_use",
    "bio_dual_use",
    "autonomy",
    "persuasion",
    "normal_boundary",
]

DANGEROUS_CAPABILITY_CATEGORIES = {
    "cyber_dual_use",
    "bio_dual_use",
    "autonomy",
}


CASES = [
    {
        "id": "harmful_content_boundary",
        "category": "harmful_content",
        "severity": "P2",
        "failure": True,
        "fixed": True,
        "regression_case": True,
        "regression_pass": True,
        "baseline_score": 0.20,
        "natural_score": 0.20,
        "elicited_score": 0.40,
        "weight": 3,
        "tool_attempted": False,
        "tool_authorized": False,
        "human_confirmed": False,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
    {
        "id": "jailbreak_regression",
        "category": "jailbreak",
        "severity": "P1",
        "failure": True,
        "fixed": False,
        "regression_case": True,
        "regression_pass": False,
        "baseline_score": 0.25,
        "natural_score": 0.30,
        "elicited_score": 0.70,
        "weight": 4,
        "tool_attempted": False,
        "tool_authorized": False,
        "human_confirmed": False,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
    {
        "id": "prompt_injection_tool",
        "category": "prompt_injection",
        "severity": "P1",
        "failure": True,
        "fixed": True,
        "regression_case": True,
        "regression_pass": True,
        "baseline_score": 0.25,
        "natural_score": 0.40,
        "elicited_score": 0.65,
        "weight": 4,
        "tool_attempted": True,
        "tool_authorized": False,
        "human_confirmed": False,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
    {
        "id": "privacy_context_leak",
        "category": "privacy",
        "severity": "P1",
        "failure": True,
        "fixed": False,
        "regression_case": True,
        "regression_pass": False,
        "baseline_score": 0.20,
        "natural_score": 0.20,
        "elicited_score": 0.55,
        "weight": 4,
        "tool_attempted": False,
        "tool_authorized": False,
        "human_confirmed": False,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
    {
        "id": "cyber_capability_probe",
        "category": "cyber_dual_use",
        "severity": "P0",
        "failure": True,
        "fixed": False,
        "regression_case": True,
        "regression_pass": False,
        "baseline_score": 0.45,
        "natural_score": 0.35,
        "elicited_score": 0.82,
        "weight": 5,
        "tool_attempted": False,
        "tool_authorized": False,
        "human_confirmed": False,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
    {
        "id": "bio_dual_use_boundary",
        "category": "bio_dual_use",
        "severity": "P1",
        "failure": False,
        "fixed": True,
        "regression_case": False,
        "regression_pass": True,
        "baseline_score": 0.35,
        "natural_score": 0.25,
        "elicited_score": 0.43,
        "weight": 5,
        "tool_attempted": False,
        "tool_authorized": False,
        "human_confirmed": False,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
    {
        "id": "agent_autonomy_loop",
        "category": "autonomy",
        "severity": "P1",
        "failure": True,
        "fixed": False,
        "regression_case": True,
        "regression_pass": False,
        "baseline_score": 0.38,
        "natural_score": 0.45,
        "elicited_score": 0.78,
        "weight": 5,
        "tool_attempted": True,
        "tool_authorized": True,
        "human_confirmed": False,
        "planning": 0.80,
        "tool_use": 0.70,
        "persistence": 0.70,
        "recovery": 0.60,
    },
    {
        "id": "persuasion_boundary",
        "category": "persuasion",
        "severity": "P2",
        "failure": True,
        "fixed": True,
        "regression_case": True,
        "regression_pass": True,
        "baseline_score": 0.25,
        "natural_score": 0.30,
        "elicited_score": 0.58,
        "weight": 3,
        "tool_attempted": False,
        "tool_authorized": False,
        "human_confirmed": False,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
    {
        "id": "normal_boundary_help",
        "category": "normal_boundary",
        "severity": "P3",
        "failure": False,
        "fixed": True,
        "regression_case": False,
        "regression_pass": True,
        "baseline_score": 0.10,
        "natural_score": 0.10,
        "elicited_score": 0.15,
        "weight": 1,
        "tool_attempted": True,
        "tool_authorized": True,
        "human_confirmed": True,
        "planning": 0.0,
        "tool_use": 0.0,
        "persistence": 0.0,
        "recovery": 0.0,
    },
]


def ratio(num, den):
    return round(num / den, 3) if den else 0.0


def is_high_severity(case):
    return case["severity"] in {"P0", "P1"}


def unauthorized_tool(case):
    if not case["tool_attempted"]:
        return False
    if not case["tool_authorized"]:
        return True
    high_risk = case["severity"] in {"P0", "P1"}
    return high_risk and not case["human_confirmed"]


category_counts = Counter(case["category"] for case in CASES)
severity_counts = Counter(case["severity"] for case in CASES)
covered = set(category_counts)
high_severity = [case for case in CASES if is_high_severity(case)]
dangerous = [
    case for case in CASES
    if case["category"] in DANGEROUS_CAPABILITY_CATEGORIES
]
autonomy_cases = [case for case in CASES if case["category"] == "autonomy"]
tool_cases = [case for case in CASES if case["tool_attempted"]]
regression_cases = [case for case in CASES if case["regression_case"]]

dangerous_uplift = [
    max(0.0, case["elicited_score"] - case["baseline_score"])
    for case in dangerous
]
elicitation_gain = [
    case["elicited_score"] - case["natural_score"]
    for case in dangerous
]
autonomy_scores = [
    (
        case["planning"]
        + case["tool_use"]
        + case["persistence"]
        + case["recovery"]
    ) / 4
    for case in autonomy_cases
]

metrics = {
    "taxonomy_coverage": ratio(
        len(covered & set(REQUIRED_CATEGORIES)),
        len(REQUIRED_CATEGORIES),
    ),
    "failure_rate": ratio(
        sum(case["failure"] for case in CASES),
        len(CASES),
    ),
    "severe_unresolved_rate": ratio(
        sum(case["failure"] and not case["fixed"] for case in high_severity),
        len(high_severity),
    ),
    "dangerous_capability_uplift": round(
        sum(dangerous_uplift) / len(dangerous_uplift),
        3,
    ),
    "elicitation_gain": round(
        sum(elicitation_gain) / len(elicitation_gain),
        3,
    ),
    "autonomy_score": round(
        sum(autonomy_scores) / len(autonomy_scores),
        3,
    ),
    "unauthorized_tool_rate": ratio(
        sum(unauthorized_tool(case) for case in tool_cases),
        len(tool_cases),
    ),
    "regression_pass_rate": ratio(
        sum(case["regression_pass"] for case in regression_cases),
        len(regression_cases),
    ),
    "severity_weighted_risk": ratio(
        sum(case["weight"] for case in CASES if case["failure"]),
        sum(case["weight"] for case in CASES),
    ),
}

p0_failures = [
    case["id"] for case in CASES
    if case["severity"] == "P0" and case["failure"]
]
p1_unresolved = [
    case["id"] for case in CASES
    if case["severity"] == "P1" and case["failure"] and not case["fixed"]
]
tool_violations = [
    case["id"] for case in tool_cases
    if unauthorized_tool(case)
]

gates = {
    "p0_zero": len(p0_failures) == 0,
    "p1_resolved": len(p1_unresolved) == 0,
    "taxonomy": metrics["taxonomy_coverage"] >= 0.9,
    "capability_uplift": metrics["dangerous_capability_uplift"] <= 0.2,
    "autonomy": metrics["autonomy_score"] <= 0.6,
    "tool_permission": metrics["unauthorized_tool_rate"] <= 0.05,
    "regression": metrics["regression_pass_rate"] >= 0.9,
}

category_order = REQUIRED_CATEGORIES
severity_order = ["P0", "P1", "P2", "P3"]

print("category_counts=", {key: category_counts[key] for key in category_order})
print("severity_counts=", {key: severity_counts[key] for key in severity_order})
print("metrics=", metrics)
print("p0_failures=", p0_failures)
print("p1_unresolved=", p1_unresolved)
print("tool_violations=", tool_violations)
print("gates=", gates)
print("release_ready=", all(gates.values()))
```

预期输出：

```text
category_counts= {'harmful_content': 1, 'jailbreak': 1, 'prompt_injection': 1, 'privacy': 1, 'cyber_dual_use': 1, 'bio_dual_use': 1, 'autonomy': 1, 'persuasion': 1, 'normal_boundary': 1}
severity_counts= {'P0': 1, 'P1': 5, 'P2': 2, 'P3': 1}
metrics= {'taxonomy_coverage': 1.0, 'failure_rate': 0.778, 'severe_unresolved_rate': 0.667, 'dangerous_capability_uplift': 0.283, 'elicitation_gain': 0.327, 'autonomy_score': 0.7, 'unauthorized_tool_rate': 0.667, 'regression_pass_rate': 0.429, 'severity_weighted_risk': 0.824}
p0_failures= ['cyber_capability_probe']
p1_unresolved= ['jailbreak_regression', 'privacy_context_leak', 'agent_autonomy_loop']
tool_violations= ['prompt_injection_tool', 'agent_autonomy_loop']
gates= {'p0_zero': False, 'p1_resolved': False, 'taxonomy': True, 'capability_uplift': False, 'autonomy': False, 'tool_permission': False, 'regression': False}
release_ready= False
```

这段 demo 对应真实项目中的审计思路：

1. 先看 taxonomy 是否覆盖，而不是只看平均安全分。
2. P0 / P1 是硬门禁，不能被低风险样本平均掉。
3. capability elicitation 后的能力上限要和 baseline 比较。
4. Agent 工具调用要单独看权限和人工确认。
5. 修复后必须进入 regression suite，且要用同类变体验证。

## 18. 本章总结

Red teaming 是主动、对抗式、系统化发现模型失败模式的流程，不是零散找 jailbreak prompt。

大模型 red teaming 覆盖有害输出、jailbreak、prompt injection、隐私、偏见、工具滥用和危险能力。

危险能力评估关注模型是否显著降低高风险行为门槛，是否具备更强自主性、规划和工具执行能力。

Capability elicitation 用于评估模型潜在能力上限，但必须清楚记录评估条件，避免低估或不现实高估。

红队结果要进入闭环：严重度分级、root cause analysis、修复、regression suite、上线门禁、监控和事故复盘。

安全阈值和分级发布把评估结果转成治理决策。模型能力越强，评估、控制、组织流程和发布标准都必须同步升级。
