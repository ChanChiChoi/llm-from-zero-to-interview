# 第九章：Multi-Agent

Multi-Agent，多智能体系统，是把多个 Agent 组织成协作网络来完成任务的架构。一个 Agent 负责规划，一个 Agent 负责检索，一个 Agent 负责执行，一个 Agent 负责评审，一个 Agent 负责汇总，听起来比单 Agent 更强，但它并不是“Agent 越多越智能”。

Multi-Agent 真正解决的问题是：复杂任务中，单个 Agent 的上下文、工具权限、专业视角和自我验证能力都有限。多 Agent 可以通过角色分工、并行执行、独立检查和冲突暴露提升可靠性。但它也会带来通信成本、责任不清、重复劳动、冲突处理、权限扩散和错误传播。

本章系统讲 Multi-Agent：为什么需要多 Agent，角色分工、通信协议、协调器、共享状态、辩论与评审、任务分配、冲突解决、共识与投票、成本控制、安全边界和评估指标。

## 0. 本讲范围与资料

本章参考 AutoGen、CAMEL、MetaGPT、ChatDev、AI safety via debate 和近年 LLM multi-agent survey 相关资料。这里不把任何框架写成唯一标准，也不展开特定框架 API，而是抽象出系统设计中更稳定的共性问题：

1. 角色如何定义，什么时候值得拆成多个 Agent。
2. Agent 之间应该传什么，不应该传什么。
3. Coordinator、黑板、辩论、投票和工具验证各自解决什么问题。
4. 多 Agent 是否真的比单 Agent 更好，如何用指标证明。
5. 如何限制权限、成本、错误传播和不可审计的自由聊天。

本章不提供绕过权限、规避审批、利用 Agent 间通信传播恶意指令或自动执行高风险动作的方法。涉及安全问题时，只从防御性设计、权限隔离、审计和评估验收条件角度讨论。

## 9.1 为什么需要 Multi-Agent

单 Agent 的问题是能力和状态都集中在一个执行循环里。复杂任务中，单 Agent 往往要同时承担规划、检索、写作、执行、验证、总结和风险判断，容易出现四类问题：

1. 角色冲突：既写答案又审答案，容易放过自己的错误。
2. 上下文拥挤：规划、证据、工具结果、历史状态都塞进同一个上下文。
3. 工具权限过大：为了完成所有步骤，单 Agent 往往被授予过多工具。
4. 无法并行：多个独立子任务只能串行完成。

Multi-Agent 的核心直觉是把复杂任务拆成多个受控角色，让它们专业化、互相校验或并行完成子任务。

Multi-Agent 的价值可以从一个具体问题理解。假设任务是调查供应商事故：需要有人整理时间线、有人查合同、有人分析日志、有人审查结论。让一个 Agent 同时拥有所有资料和权限，会导致上下文拥挤，也很难独立检查自己的结论；拆成多个 Agent 后，任务可以并行，权限可以按资料类型切开，审查者也可以看到生成者没有看到的证据。

但拆分带来新的系统变量：子任务之间如何传递结果，谁负责发现冲突，谁拥有最终决策权，多个 Agent 是否使用了同一条错误来源，以及并行节省的时间是否抵消了通信和验证成本。因此 Multi-Agent 的设计单位不是 Agent 数量，而是任务依赖、权限边界、验证独立性和可观测状态。

## 9.2 Multi-Agent 的基本结构

常见结构包括：

1. Centralized：一个 coordinator 分配任务、收集结果、做最终决策。
2. Decentralized：多个 Agent 彼此通信，没有单一中心节点。
3. Hierarchical：高层 Agent 规划，低层 Agent 执行。
4. Debate：多个 Agent 给出不同观点，再由 judge 或 verifier 汇总。
5. Committee：多个 Agent 独立解题后投票、加权汇总或触发复核。
6. Blackboard：所有 Agent 通过共享状态表读写任务状态和证据。

工程系统最常用的是 centralized 或 hierarchical，因为它们更容易控制流程、权限、日志和预算。完全自由聊天式 multi-agent 在 demo 中很直观，但生产系统通常更需要结构化消息、状态机和审计 trace。

## 9.3 Agent 抽象与角色集合

可以把一个 multi-agent 系统中的 Agent 集合写成：

~~~math
\mathcal{A}=\{a_1,\ldots,a_n\}
~~~

其中每个 Agent 不只是一个 prompt，而是一个带角色、工具、权限、上下文和预算的执行单元：

~~~math
a_i=(r_i,T_i,P_i,C_i,B_i)
~~~

变量含义：

1. `r_i` 是第 `i` 个 Agent 的角色，例如 planner、researcher、coder、reviewer。
2. `T_i` 是它可调用的工具集合。
3. `P_i` 是它的权限边界。
4. `C_i` 是它能看到的上下文。
5. `B_i` 是它可消耗的预算，例如 token、时间、工具调用次数或成本。

这条公式的工程含义是：不要把 Agent 只理解为“一个不同的 system prompt”。可靠的 Multi-Agent 设计必须同时定义角色、工具、权限、上下文和预算。只换 prompt、不限权限、不管 trace，通常只是把单 Agent 的风险复制了多份。

## 9.4 关键公式与 Multi-Agent 指标速查

Multi-Agent 的核心对象包括 Agent、消息、任务分配、通信图、冲突和验收条件。

### 9.4.1 消息协议

一条 Agent 间消息可以抽象为：

~~~math
m_t=(s_t,r_t,\iota_t,c_t,E_t,\gamma_t)
~~~

其中 `s_t` 是 sender，`r_t` 是 receiver，`\iota_t` 是 intent，`c_t` 是消息内容，`E_t` 是证据集合，`\gamma_t` 是置信度。

消息协议至少要回答三个问题：

1. 这条消息是谁发给谁的。
2. 它是请求、回答、证据、反驳、状态更新还是最终建议。
3. 它有没有可检查的证据和置信度。

消息有效率可以写成：

~~~math
R_{\mathrm{msg}}=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\mathrm{valid}(m_t)]
~~~

其中 `valid` 表示消息 schema 可解析、字段完整、接收方明确、意图合法。
只有实际记录了消息时，`R_msg` 才有数值；`T=0` 表示没有消息证据，
应记为 `None` 或 `unknown`，不能因为没有观察到无效消息就记成 `1`。

### 9.4.2 任务分配

一个子任务分配可以写成：

~~~math
z_j=(u_j,a(z_j),d_j,\rho_j)
~~~

其中 `u_j` 是子任务，`a(z_j)` 是被分配的 Agent，`d_j` 是 deadline 或依赖约束，`\rho_j` 是风险等级。

角色匹配率衡量任务是否分给了合适角色：

~~~math
R_{\mathrm{role}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\mathrm{role}(a(z_j))=\mathrm{role}^{\star}(z_j)]
~~~

其中 `M` 是子任务数量，`\mathrm{role}^{\star}(z_j)` 是该子任务理想角色。
只有 `M>0` 时才定义角色匹配率。空计划表示没有角色匹配证据，而不是
所有任务都分配正确。

### 9.4.3 通信图

Agent 之间的通信关系可以写成有向图：

~~~math
G_{\mathrm{comm}}=(\mathcal{A},\mathcal{E}_{\mathrm{comm}})
~~~

如果所有 Agent 都能任意互发消息，通信图接近完全图，沟通成本会迅速增加。若有 coordinator，通信图通常更接近星型或层级结构，更容易审计。

通信成本可以粗略写成：

~~~math
C_{\mathrm{comm}}=\sum_{t=1}^{T}C(m_t)
~~~

其中 `C(m_t)` 可以按 token、延迟、模型调用成本或人工审阅成本估算。
若没有消息，通信成本可以记录为观测到的零成本，但通信质量指标仍然是
`unknown`；二者不能混为一谈。

### 9.4.4 证据支持与冲突解决

证据支持率衡量消息或结论是否有可追溯依据：

~~~math
R_{\mathrm{evid}}=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\mathrm{supported}(m_t)]
~~~

这里的 `T` 是实际进入评估集的消息或结论数，`T=0` 时支持率未定义。

冲突解决率衡量系统是否识别并解决矛盾结论：

~~~math
R_{\mathrm{conf}}=\frac{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k \land \mathrm{resolved}_k]}{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k]}
~~~

注意：冲突“被解决”不等于“解决正确”。高风险任务还要看解决正确率：

~~~math
R_{\mathrm{conf\_ok}}=\frac{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k \land \mathrm{correct}_k]}{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k]}
~~~

两个冲突指标都要求至少存在一个被检测到的冲突。没有冲突样本时，
系统没有获得冲突处理能力的证据；不能把它当作解决率或正确率为 `1`。

### 9.4.5 重复劳动、收益和成本

重复劳动率衡量多个 Agent 是否在做同一件事：

~~~math
R_{\mathrm{dup}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\mathrm{duplicate}(z_j)]
~~~

Multi-Agent 相比单 Agent 的收益可以写成：

~~~math
L_{\mathrm{multi}}=S_{\mathrm{multi}}-S_{\mathrm{single}}
~~~

其中 `S_{\mathrm{multi}}` 是 multi-agent 系统任务分数，`S_{\mathrm{single}}` 是单 Agent baseline 分数。这个指标很重要，因为 Multi-Agent 的比较对象不是“空系统”，而是更简单、更便宜的单 Agent 或固定 workflow。
只有 multi-agent 与 baseline 在相同任务、输入、评估规则和可比预算下都
有观测值时，`L_multi` 才有解释力。缺少 baseline 不能默认为零收益。

总成本可以写成：

~~~math
C_{\mathrm{multi}}=\sum_{i=1}^{n}C(a_i)+\sum_{t=1}^{T}C(m_t)+C_{\mathrm{coord}}+C_{\mathrm{verify}}
~~~

其中 `C_{\mathrm{coord}}` 是协调器成本，`C_{\mathrm{verify}}` 是验证和人工升级成本。

### 9.4.6 Multi-Agent 综合交付检查

一个简化的综合交付检查条件可以形式化为：

~~~math
I_{\mathrm{multi}}=\mathbf{1}[S_{\mathrm{multi}}\ge \tau_s \land L_{\mathrm{multi}}\ge \tau_l \land R_{\mathrm{conf}}\ge \tau_c \land R_{\mathrm{dup}}\le \tau_d \land R_{\mathrm{perm}}\le \tau_p \land C_{\mathrm{multi}}\le B]
~~~

其中 `R_{\mathrm{perm}}` 是权限违规率，`B` 是成本预算，`\tau_s,\tau_l,\tau_c,\tau_d,\tau_p` 是上线阈值。

直觉：Multi-Agent 必须同时证明“做得好”“比单 Agent 有增益”“冲突能处理”“没有大量重复劳动”“权限不乱”“成本可接受”。只展示一个看起来很热闹的协作过程，不足以交付。`I_multi` 是把这些维度汇总到审计表的教学记号，不是建议把复杂产品压缩成一个布尔字段。
公式中的每个比例都需要自己的非空分母，收益需要配对 baseline，成本需要
完整记录模型、通信、协调和验证开销。任一必要指标为 `None` 或 `unknown`
时，综合结果应保持未定义，直到补齐相应证据；不能把未知当成通过。

## 9.5 角色分工

Multi-Agent 的关键是角色明确。

常见角色包括：

1. Planner：拆解任务和制定计划。
2. Researcher：检索资料、收集证据和标注来源。
3. Executor：调用工具或执行动作。
4. Coder：阅读代码、生成 patch、运行测试。
5. Reviewer：检查结果、diff、证据和测试。
6. Critic：专门找漏洞、反例和边界条件。
7. Judge：在候选答案或冲突结论之间做结构化裁决。
8. Summarizer：汇总输出，压缩 trace。
9. Coordinator：分配任务、控制流程、预算和权限。

角色不是越多越好。一个实用判断是：如果拆出某个 Agent 能降低单个上下文复杂度、提升验证独立性、限制权限范围或实现真实并行，就值得考虑；如果只是把同一份工作拆成多人聊天，通常会增加成本而不提升质量。

## 9.6 Coordinator 模式

Coordinator 是多 Agent 系统中的调度者。

它负责：

1. 理解用户目标和成功标准。
2. 拆解子任务和依赖关系。
3. 选择合适 Agent。
4. 控制每个 Agent 的上下文和工具权限。
5. 收集结果并维护全局状态。
6. 识别冲突和缺失证据。
7. 控制预算、超时和停止条件。
8. 生成最终回答或请求人工确认。

Coordinator 模式的优势是可控，适合生产系统。缺点是 coordinator 可能成为瓶颈。如果 coordinator 分错任务、漏看冲突或过早停止，整个系统会偏离方向。

工程上可以把 coordinator 设计成“状态机 + 路由器 + 审计器”，而不是一个自由生成长文本的 Agent。这样更容易做 replay、debug 和指标统计。

## 9.7 通信协议

Agent 之间不能随意聊天，最好有结构化通信协议。

一个实用消息 schema 可以包含：

1. `sender`：发送者。
2. `receiver`：接收者。
3. `task_id`：任务编号。
4. `intent`：消息目的，例如 assign、question、evidence、critique、decision。
5. `content`：正文内容。
6. `evidence`：证据、引用或工具结果。
7. `confidence`：置信度。
8. `status`：完成、失败、阻塞或需要澄清。
9. `risk_level`：是否涉及高风险动作。

结构化通信的价值：

1. 减少误解。
2. 降低上下文冗余。
3. 支持日志回放。
4. 支持自动评估。
5. 支持权限审计。
6. 支持冲突定位。

Multi-Agent 不是让多个模型互相发散聊天，而是要把通信变成可解析、可过滤、可度量的事件流。发送者、接收者、意图、证据、状态和风险字段共同决定一条消息能否进入下一个 Agent 的上下文；自由文本可以作为内容，但不能替代协议字段。

## 9.8 共享状态与 Blackboard

多 Agent 需要共享部分状态，但共享越多不一定越好。

共享状态可以包括：

1. 全局目标。
2. 当前计划。
3. 子任务状态。
4. 已收集证据。
5. 已执行动作。
6. 冲突点。
7. 风险标记。
8. 最终决策。

工程上常用 blackboard 或 task state table：

~~~text
task_id | owner | status | evidence_ids | blockers | risk_level | decision
~~~

Blackboard 的优点是状态集中、便于审计。缺点是容易成为上下文膨胀点。如果每个 Agent 都读取全部 blackboard，系统仍然会退化成一个超长上下文单 Agent。更稳的做法是：按角色、任务和权限过滤可见状态。

## 9.9 辩论、Critique 与 Judge

多 Agent 常用于辩论或互评。

典型流程：

~~~text
Agent A 给出答案
Agent B 找反例
Agent C 检查证据
Judge 汇总并选择最终结论
~~~

优势：

1. 能发现单 Agent 忽略的问题。
2. 能从不同角度评估答案。
3. 适合复杂决策、开放问题和假设比较。
4. 可以把“生成”和“检查”分离。

局限：

1. 多个 Agent 可能共享同样偏差。
2. 辩论可能变成冗长文本互相说服。
3. Judge 也可能被流畅表达误导。
4. 成本明显上升。
5. 对可执行任务，工具验证通常比纯文本辩论更可靠。

辩论不应被当成万能验证器。它适合暴露候选假设和不确定性，但最终结论最好结合证据、工具验证、测试、规则检查或人工复核。若所有参与者都读取同一份错误材料，更多发言只会把共同错误表达得更自信。

## 9.10 并行执行

Multi-Agent 可以提升并行性。

例如研究报告任务：

1. Agent A 查背景。
2. Agent B 查竞品。
3. Agent C 查技术路线。
4. Agent D 查风险。
5. Coordinator 汇总。

并行执行能节省墙钟时间，但不是总成本更低。系统还需要处理结果合并、来源去重和冲突。如果多个 Agent 得到矛盾结论，系统必须记录来源并解决冲突，而不是让 summarizer 随机选择一个更像真的说法。

## 9.11 冲突解决

多 Agent 系统一定会出现冲突。

冲突类型：

1. 事实冲突：两个 Agent 给出不同事实。
2. 方案冲突：两个 Agent 推荐不同路径。
3. 优先级冲突：速度、成本、质量、安全目标冲突。
4. 工具结果冲突：检索、测试、执行结果不一致。
5. 权限冲突：某个 Agent 试图执行超出权限的动作。
6. 资源冲突：多个 Agent 争用预算、工具或上下文窗口。

解决方式：

1. 要求提供证据。
2. 调用外部工具验证。
3. 使用 judge Agent 做结构化裁决。
4. 由 coordinator 按规则决策。
5. 触发人工确认。
6. 保留不确定性，而不是强行统一。

高风险任务中，冲突不应只由模型闭环自动决定。更合理的设计是：低风险冲突可由规则或 verifier 自动处理，高风险冲突升级给人或确定性系统。

## 9.12 任务分配与权限隔离

任务分配要同时考虑能力、上下文、工具和权限。

例如：

1. Researcher 有检索工具，但没有写权限。
2. Coder 有文件编辑权限，但不能访问无关隐私数据。
3. Reviewer 只能读 diff、测试结果和审计 trace。
4. Judge 只能做裁决，不能执行外部动作。
5. Coordinator 可以分配任务，但不直接执行高风险动作。

权限分离可以降低事故范围。不要让所有 Agent 都拥有全部工具权限。否则 Multi-Agent 不但没有增加安全性，反而把一个过大权限 Agent 复制成多个过大权限 Agent。

## 9.13 Consensus、Voting 与加权汇总

Committee 型 Multi-Agent 常用投票或共识。

常见方式：

1. 多个 Agent 独立回答，简单多数投票。
2. 按历史准确率或任务匹配度加权投票。
3. 要求每个 Agent 提供证据，再按证据质量汇总。
4. 分歧过大时不输出确定结论，触发复核。

投票适合答案空间明确、独立错误概率较低的任务。它不适合所有开放任务，因为多个 Agent 可能共享同一模型、同一训练偏差、同一错误检索来源。投票前要尽量保证候选答案独立，投票后要看证据，而不是只看票数。

## 9.14 多 Agent 的成本

Multi-Agent 成本很高。

成本来源：

1. 多个模型调用。
2. Agent 之间通信。
3. 重复检索。
4. 重复验证。
5. 上下文汇总。
6. 冲突解决。
7. 人工升级。
8. 额外 trace 存储和回放。

因此要判断任务是否值得多 Agent。简单任务用单 Agent 或固定 workflow 更合适。一个常见反模式是：一个一句话任务被 planner、researcher、writer、reviewer、summarizer 转一圈，最后质量没有提升，只是成本上升。

## 9.15 Multi-Agent 和 Workflow

Multi-Agent 和 workflow 可以结合。

一种实用架构：

~~~text
固定 workflow 控制主流程
关键步骤交给专门 Agent 执行
Coordinator 汇总结果
Verifier 做最终检查
~~~

这样既保留 workflow 的稳定性，又利用 Agent 的灵活性。生产系统中更常见的是“workflow 主控 + Agent 局部自治”，而不是“多个 Agent 完全自由聊天”。

## 9.16 安全风险

Multi-Agent 增加了安全面。

风险包括：

1. Agent 之间传播错误信息。
2. 一个 Agent 接收不可信内容后影响其他 Agent。
3. 权限边界不清。
4. 日志难以审计。
5. 冲突处理不透明。
6. 成本失控。
7. 多 Agent 互相强化错误结论。
8. 高风险动作缺少统一审批。

安全设计：

1. 最小权限。
2. 明确角色边界。
3. 结构化消息。
4. 共享状态审计。
5. 高风险动作统一由 controller 审批。
6. 不信任其他 Agent 的未验证结论。
7. 对不可信内容做来源标记和上下文隔离。
8. 保留 trace，支持 replay 和责任定位。

## 9.17 评估指标

Multi-Agent 评估可以看：

1. 任务成功率。
2. 相比单 Agent 的提升。
3. 平均成本。
4. 平均延迟。
5. 通信轮数。
6. 角色匹配率。
7. 消息 schema 有效率。
8. 证据支持率。
9. 冲突解决成功率。
10. 重复工作比例。
11. 权限违规率。
12. 不必要 multi-agent 比例。
13. 人工接管率。
14. 最终答案证据支持率。

核心问题是：多 Agent 是否真的比单 Agent 更好，还是只是更贵、更复杂。评估必须固定单 Agent 或 workflow baseline，记录相同任务、相同输入和相近预算，再比较成功率、证据质量、延迟、成本、冲突和权限事件。否则“协作提升”可能只是任务样本或预算不同造成的假象。

## 9.18 最小可运行 Multi-Agent audit demo

下面这个 demo 不调用外部模型，而是构造 4 条 toy multi-agent trace，审计角色匹配、任务成功、消息有效性、证据支持、冲突解决、重复劳动、权限违规和相对单 Agent 的收益。

它演示的问题是：Multi-Agent 不能只看最终是否成功，还要看协作过程是否可控、可解释、低重复、低权限风险，并且是否真的优于单 Agent baseline。

~~~python
import math
from collections import Counter
from dataclasses import dataclass


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


def require_bool(value, field):
    if type(value) is not bool:
        raise TypeError(f"{field} must be bool")


def require_tuple(value, field):
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise TypeError(f"{field} must be a tuple of non-empty strings")


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role: str
    tools: tuple
    permissions: tuple

    def __post_init__(self):
        require_text(self.agent_id, "agent_id")
        require_text(self.role, "role")
        require_tuple(self.tools, "tools")
        require_tuple(self.permissions, "permissions")


@dataclass(frozen=True)
class Assignment:
    task_id: str
    required_role: str
    assigned_agent: str
    success: bool
    duplicate: bool = False

    def __post_init__(self):
        require_text(self.task_id, "task_id")
        require_text(self.required_role, "required_role")
        require_text(self.assigned_agent, "assigned_agent")
        require_bool(self.success, "assignment.success")
        require_bool(self.duplicate, "assignment.duplicate")


@dataclass(frozen=True)
class Message:
    message_id: str
    sender: str
    receiver: str
    intent: str
    valid_schema: bool
    supported_by_evidence: bool

    def __post_init__(self):
        for field in ("message_id", "sender", "receiver", "intent"):
            require_text(getattr(self, field), field)
        require_bool(self.valid_schema, "message.valid_schema")
        require_bool(self.supported_by_evidence, "message.supported_by_evidence")


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    detected: bool
    resolved: bool
    resolution_correct: bool

    def __post_init__(self):
        require_text(self.conflict_id, "conflict_id")
        require_bool(self.detected, "conflict.detected")
        require_bool(self.resolved, "conflict.resolved")
        require_bool(self.resolution_correct, "conflict.resolution_correct")


@dataclass(frozen=True)
class Run:
    run_id: str
    agents: dict
    assignments: tuple
    messages: tuple
    conflicts: tuple
    final_success: bool
    single_agent_success: bool | None
    total_cost: float
    permission_violations: int = 0
    unnecessary_multi_agent: bool = False

    def __post_init__(self):
        require_text(self.run_id, "run_id")
        if not isinstance(self.agents, dict):
            raise TypeError("agents must be a dict")
        for field in ("assignments", "messages", "conflicts"):
            if not isinstance(getattr(self, field), tuple):
                raise TypeError(f"{field} must be a tuple")
        require_bool(self.final_success, "final_success")
        if self.single_agent_success is not None:
            require_bool(self.single_agent_success, "single_agent_success")
        if not isinstance(self.total_cost, (int, float)) or not math.isfinite(self.total_cost) or self.total_cost < 0:
            raise ValueError("total_cost must be finite and non-negative")
        if type(self.permission_violations) is not int or self.permission_violations < 0:
            raise ValueError("permission_violations must be a non-negative integer")
        require_bool(self.unnecessary_multi_agent, "unnecessary_multi_agent")


def validate_runs(runs):
    if not isinstance(runs, (list, tuple)):
        raise TypeError("runs must be a list or tuple")
    run_ids = set()
    for run in runs:
        if not isinstance(run, Run):
            raise TypeError("runs must contain Run values")
        if run.run_id in run_ids:
            raise ValueError(f"duplicate run_id: {run.run_id}")
        run_ids.add(run.run_id)
        if any(key != spec.agent_id for key, spec in run.agents.items()):
            raise ValueError("agent dictionary keys must match AgentSpec.agent_id")
        if len(set(run.agents)) != len(run.agents):
            raise ValueError("agent IDs must be unique")
        if run.permission_violations > len(run.assignments):
            raise ValueError("permission_violations must count assignment-level violations")

        assignment_ids = set()
        for assignment in run.assignments:
            if not isinstance(assignment, Assignment):
                raise TypeError("assignments must contain Assignment values")
            if assignment.task_id in assignment_ids:
                raise ValueError(f"duplicate task_id: {assignment.task_id}")
            assignment_ids.add(assignment.task_id)
            if assignment.assigned_agent not in run.agents:
                raise ValueError(f"unknown assigned agent: {assignment.assigned_agent}")

        message_ids = set()
        for message in run.messages:
            if not isinstance(message, Message):
                raise TypeError("messages must contain Message values")
            if message.message_id in message_ids:
                raise ValueError(f"duplicate message_id: {message.message_id}")
            message_ids.add(message.message_id)
            if message.sender not in run.agents or message.receiver not in run.agents:
                raise ValueError("message endpoints must be agents in the same run")

        conflict_ids = set()
        for conflict in run.conflicts:
            if not isinstance(conflict, Conflict):
                raise TypeError("conflicts must contain Conflict values")
            if conflict.conflict_id in conflict_ids:
                raise ValueError(f"duplicate conflict_id: {conflict.conflict_id}")
            conflict_ids.add(conflict.conflict_id)


def rate(numerator, denominator):
    """Return a bounded rate, or None when the relevant set is empty."""
    if type(numerator) is not int or type(denominator) is not int:
        raise TypeError("rate counts must be integers")
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("rate counts must satisfy 0 <= numerator <= denominator")
    return None if denominator == 0 else round(numerator / denominator, 3)


def at_least(value, threshold):
    return value is not None and value >= threshold


def at_most(value, threshold):
    return value is not None and value <= threshold


def exactly(value, expected):
    return value is not None and value == expected


def role_match(run, assignment):
    return (
        assignment.assigned_agent in run.agents
        and run.agents[assignment.assigned_agent].role == assignment.required_role
    )


def audit_metrics(runs):
    validate_runs(runs)
    total_runs = len(runs)
    all_assignments = [assignment for run in runs for assignment in run.assignments]
    all_messages = [message for run in runs for message in run.messages]
    all_conflicts = [conflict for run in runs for conflict in run.conflicts if conflict.detected]

    paired_baseline = [run for run in runs if run.single_agent_success is not None]
    return {
        "task_success_rate": rate(sum(run.final_success for run in runs), total_runs),
        "single_agent_lift_rate": rate(
            sum(run.final_success and not run.single_agent_success for run in paired_baseline),
            len(paired_baseline),
        ),
        "role_match_rate": rate(
            sum(role_match(run, assignment) for run in runs for assignment in run.assignments),
            len(all_assignments),
        ),
        "assignment_success_rate": rate(sum(a.success for a in all_assignments), len(all_assignments)),
        "message_valid_rate": rate(sum(m.valid_schema for m in all_messages), len(all_messages)),
        "evidence_support_rate": rate(sum(m.supported_by_evidence for m in all_messages), len(all_messages)),
        "conflict_resolution_rate": rate(sum(c.resolved for c in all_conflicts), len(all_conflicts)),
        "correct_resolution_rate": rate(sum(c.resolution_correct for c in all_conflicts), len(all_conflicts)),
        "duplicate_work_rate": rate(sum(a.duplicate for a in all_assignments), len(all_assignments)),
        "permission_violation_rate": rate(
            sum(run.permission_violations for run in runs), len(all_assignments)
        ),
        "unnecessary_multi_agent_rate": rate(
            sum(run.unnecessary_multi_agent for run in runs), total_runs
        ),
        "avg_cost": None if total_runs == 0 else round(sum(run.total_cost for run in runs) / total_runs, 3),
    }


runs = [
    Run(
        run_id="research_report",
        agents={
            "coord": AgentSpec("coord", "coordinator", ("route",), ("assign",)),
            "plan": AgentSpec("plan", "planner", ("outline",), ("read",)),
            "search": AgentSpec("search", "researcher", ("search",), ("read",)),
            "check": AgentSpec("check", "verifier", ("compare",), ("read",)),
            "sum": AgentSpec("sum", "summarizer", ("write",), ("read",)),
        },
        assignments=(
            Assignment("scope", "planner", "plan", True),
            Assignment("collect", "researcher", "search", True),
            Assignment("verify", "verifier", "check", True),
            Assignment("summarize", "summarizer", "sum", True),
        ),
        messages=(
            Message("m1", "coord", "plan", "assign", True, True),
            Message("m2", "plan", "search", "request_evidence", True, True),
            Message("m3", "search", "check", "provide_evidence", True, True),
            Message("m4", "check", "sum", "verified_summary", True, True),
        ),
        conflicts=(Conflict("c1", True, True, True),),
        final_success=True,
        single_agent_success=False,
        total_cost=1.40,
    ),
    Run(
        run_id="code_review_swarm",
        agents={
            "coord": AgentSpec("coord", "coordinator", ("route",), ("assign",)),
            "dev": AgentSpec("dev", "coder", ("patch", "test"), ("read", "write")),
            "rev": AgentSpec("rev", "reviewer", ("diff",), ("read",)),
            "search": AgentSpec("search", "researcher", ("grep",), ("read",)),
        },
        assignments=(
            Assignment("inspect_diff", "reviewer", "search", False),
            Assignment("run_tests", "coder", "dev", True),
            Assignment("review_security", "reviewer", "rev", True),
            Assignment("inspect_diff_dup", "reviewer", "rev", True, duplicate=True),
        ),
        messages=(
            Message("m5", "search", "coord", "claim", True, False),
            Message("m6", "dev", "rev", "test_result", True, True),
            Message("m7", "rev", "coord", "review", True, True),
        ),
        conflicts=(),
        final_success=True,
        single_agent_success=False,
        total_cost=1.55,
        permission_violations=1,
    ),
    Run(
        run_id="debate_answer",
        agents={
            "coord": AgentSpec("coord", "coordinator", ("route",), ("assign",)),
            "a": AgentSpec("a", "debater", ("reason",), ("read",)),
            "b": AgentSpec("b", "debater", ("reason",), ("read",)),
            "j": AgentSpec("j", "judge", ("score",), ("read",)),
        },
        assignments=(
            Assignment("argue_yes", "debater", "a", True),
            Assignment("argue_no", "debater", "b", True),
            Assignment("judge", "judge", "j", False),
        ),
        messages=(
            Message("m8", "a", "j", "argument", True, False),
            Message("m9", "b", "j", "counterargument", True, False),
            Message("m10", "j", "a", "clarify", True, True),
            Message("m11", "a", "j", "unsupported_claim", False, False),
            Message("m12", "j", "coord", "final_judgment", True, True),
        ),
        conflicts=(Conflict("c2", True, False, False),),
        final_success=False,
        single_agent_success=False,
        total_cost=1.10,
    ),
    Run(
        run_id="over_coordinated_small_task",
        agents={
            "coord": AgentSpec("coord", "coordinator", ("route",), ("assign",)),
            "plan": AgentSpec("plan", "planner", ("outline",), ("read",)),
            "writer": AgentSpec("writer", "writer", ("write",), ("read",)),
            "rev": AgentSpec("rev", "reviewer", ("check",), ("read",)),
        },
        assignments=(
            Assignment("plan_one_sentence", "planner", "plan", True),
            Assignment("write_one_sentence", "writer", "writer", True),
            Assignment("review_one_sentence", "reviewer", "rev", True, duplicate=True),
        ),
        messages=(
            Message("m13", "plan", "writer", "outline", True, True),
            Message("m14", "rev", "coord", "approval", True, True),
        ),
        conflicts=(),
        final_success=True,
        single_agent_success=True,
        total_cost=1.25,
        unnecessary_multi_agent=True,
    ),
]

validate_runs(runs)
metrics = audit_metrics(runs)

failure_reasons = Counter()
problem_runs = []
for run in runs:
    run_has_problem = False
    for assignment in run.assignments:
        if not role_match(run, assignment):
            failure_reasons["role_mismatch"] += 1
            run_has_problem = True
        if not assignment.success:
            failure_reasons["assignment_failed"] += 1
            run_has_problem = True
        if assignment.duplicate:
            failure_reasons["duplicate_work"] += 1
            run_has_problem = True
    for message in run.messages:
        if not message.valid_schema:
            failure_reasons["invalid_message_schema"] += 1
            run_has_problem = True
        if not message.supported_by_evidence:
            failure_reasons["unsupported_message"] += 1
            run_has_problem = True
    for conflict in run.conflicts:
        if conflict.detected and not conflict.resolved:
            failure_reasons["unresolved_conflict"] += 1
            run_has_problem = True
    if run.permission_violations:
        failure_reasons["permission_violation"] += run.permission_violations
        run_has_problem = True
    if run.unnecessary_multi_agent:
        failure_reasons["unnecessary_multi_agent"] += 1
        run_has_problem = True
    if not run.final_success:
        failure_reasons["task_failed"] += 1
        run_has_problem = True
    if run_has_problem:
        problem_runs.append(run.run_id)

checks = {
    "task_success": at_least(metrics["task_success_rate"], 0.80),
    "single_agent_lift": at_least(metrics["single_agent_lift_rate"], 0.30),
    "role_match": at_least(metrics["role_match_rate"], 0.90),
    "message_valid": at_least(metrics["message_valid_rate"], 0.95),
    "evidence_support": at_least(metrics["evidence_support_rate"], 0.80),
    "conflict_resolution": at_least(metrics["conflict_resolution_rate"], 0.80),
    "duplicate_work": at_most(metrics["duplicate_work_rate"], 0.10),
    "permission": exactly(metrics["permission_violation_rate"], 0.0),
    "unnecessary_multi_agent": at_most(metrics["unnecessary_multi_agent_rate"], 0.10),
    "cost": at_most(metrics["avg_cost"], 1.50),
}

top_failure_reasons = sorted(failure_reasons.items(), key=lambda item: (-item[1], item[0]))

# 空运行集和没有冲突的运行都不能伪装成“所有检查已通过”。
empty_metrics = audit_metrics([])
assert all(value is None for value in empty_metrics.values())
no_conflict_metrics = audit_metrics([runs[1]])
assert no_conflict_metrics["conflict_resolution_rate"] is None
no_baseline_run = Run(
    run_id="missing_single_agent_baseline",
    agents={"a": AgentSpec("a", "researcher", ("search",), ("read",))},
    assignments=(Assignment("collect", "researcher", "a", True),),
    messages=(),
    conflicts=(),
    final_success=True,
    single_agent_success=None,
    total_cost=0.2,
)
assert audit_metrics([no_baseline_run])["single_agent_lift_rate"] is None
assert not at_least(None, 0.0)
assert not at_most(None, 0.0)
assert not exactly(None, 1.0)

try:
    rate(1, float("nan"))
except TypeError:
    pass
else:
    raise AssertionError("non-integer denominator must be rejected")

try:
    audit_metrics([runs[0], runs[0]])
except ValueError:
    pass
else:
    raise AssertionError("duplicate run IDs must be rejected")

invalid_run = Run(
    run_id="invalid_agent_reference",
    agents={"coord": AgentSpec("coord", "coordinator", ("route",), ("assign",))},
    assignments=(Assignment("task", "planner", "missing", True),),
    messages=(),
    conflicts=(),
    final_success=False,
    single_agent_success=False,
    total_cost=0.1,
)
try:
    audit_metrics([invalid_run])
except ValueError:
    pass
else:
    raise AssertionError("unknown assigned agents must be rejected")

print(f"metrics={metrics}")
print(f"problem_runs={problem_runs}")
print(f"top_failure_reasons={top_failure_reasons}")
all_checks_pass = all(checks.values())
print(f"checks={checks}")
print(f"empty_metrics={empty_metrics}")
print(f"all_checks_pass={all_checks_pass}")
~~~

输出示例：

~~~text
metrics={'task_success_rate': 0.75, 'single_agent_lift_rate': 0.5, 'role_match_rate': 0.929, 'assignment_success_rate': 0.857, 'message_valid_rate': 0.929, 'evidence_support_rate': 0.714, 'conflict_resolution_rate': 0.5, 'correct_resolution_rate': 0.5, 'duplicate_work_rate': 0.143, 'permission_violation_rate': 0.071, 'unnecessary_multi_agent_rate': 0.25, 'avg_cost': 1.325}
problem_runs=['code_review_swarm', 'debate_answer', 'over_coordinated_small_task']
top_failure_reasons=[('unsupported_message', 4), ('assignment_failed', 2), ('duplicate_work', 2), ('invalid_message_schema', 1), ('permission_violation', 1), ('role_mismatch', 1), ('task_failed', 1), ('unnecessary_multi_agent', 1), ('unresolved_conflict', 1)]
checks={'task_success': False, 'single_agent_lift': True, 'role_match': True, 'message_valid': False, 'evidence_support': False, 'conflict_resolution': False, 'duplicate_work': False, 'permission': False, 'unnecessary_multi_agent': False, 'cost': True}
empty_metrics={'task_success_rate': None, 'single_agent_lift_rate': None, 'role_match_rate': None, 'assignment_success_rate': None, 'message_valid_rate': None, 'evidence_support_rate': None, 'conflict_resolution_rate': None, 'correct_resolution_rate': None, 'duplicate_work_rate': None, 'permission_violation_rate': None, 'unnecessary_multi_agent_rate': None, 'avg_cost': None}
all_checks_pass=False
~~~

这个 demo 的 `all_checks_pass=False` 不是程序错误，而是刻意暴露 multi-agent 系统的常见问题：成功率不足、消息证据不够、冲突未解决、重复劳动、权限违规和小任务过度编排。真实系统应把这些 trace 级问题作为待处理的工程发现，而不是用一个总结果掩盖具体原因。

## 9.19 Agent Swarm 的收益、成本与责任边界

一些前沿模型和产品资料把多子 Agent 协作作为长周期任务能力的一部分。无论具体产品如何命名，这类系统都可能让搜索、代码、测试、审查等子任务并行，但“子 Agent 越多越强”是错误的工程直觉。若公开资料没有披露内部调度、失败合并和成本数据，书稿只能把它作为产品形态观察，不能把宣传描述写成通用能力结论。

一个简化的总成本模型是：

~~~math
C_{\mathrm{swarm}}=\sum_{i=1}^{N}C_{\mathrm{agent},i}
+C_{\mathrm{coord}}
+C_{\mathrm{sync}}
+C_{\mathrm{verify}}
~~~

`C_sync` 包括消息、共享状态和上下文重建成本，`C_verify` 包括最终汇总、冲突解决和工具验证。并发只可能降低 wall-clock 的一部分，不能消除 token、工具和通信成本。

设计 Agent Swarm 时应明确：

1. 每个子 Agent 有独立 session、权限和上下文边界。
2. 共享内容优先是结构化 artifact 和证据引用，不是无限复制完整对话。
3. Coordinator 负责分配和停止，但不能替代最终 verifier。
4. 高风险动作必须绑定到统一 permission engine，不能因为来自“内部 Agent”就默认信任。
5. 每个子任务有 owner、deadline、checkpoint 和失败状态，避免 swarm 只产出一堆无人负责的意见。

因此，多 Agent 的增益应相对单 Agent baseline 报告：

~~~math
\mathrm{NetLift}=\Delta\mathrm{Success}
-\lambda_c\Delta\mathrm{Cost}
-\lambda_l\Delta\mathrm{Latency}
-\lambda_r\Delta\mathrm{Risk}
~~~

### 案例：Dynamic Workflows 与 System Card 多 Agent 评测

Anthropic 的 Claude Code **Dynamic Workflows** 展示了一种面向长任务的产品形态：Claude 根据 prompt 动态拆解任务、生成 orchestration scripts、并行运行 subagents，在结果汇总前检查；其他 agents 会从独立角度尝试反驳发现，workflow 迭代到结论收敛。协调状态放在对话之外并持续保存，因此中断的长任务可以从 checkpoint 恢复。这超出了普通的“同一轮多发几次 tool call”，但官方博客没有公开完整 scheduler、失败合并算法或独立成功率，不能据此推断模型内部结构。该博客是沿 Opus 4.8 锚点补充的 Anthropic 产品/harness 资料，不表示这项产品能力仅属于 Opus 4.8。

这类并行化要与安全、预算控制一起设计。Anthropic 页面提醒 workflow 比一般 Claude Code session 消耗更多 token；首次触发先展示将运行的内容并请求确认，组织管理员可以通过 managed settings 关闭。页面当前称该功能 generally available，建议开启 auto mode，并称 Max/Team/Enterprise 与 Claude Code API 默认开启、Pro 可在 `/config` 启用；具体计划权限仍受管理员设置影响。`ultracode` 是 Claude Code 专属入口：effort 设为 `xhigh`，workflow 的启动时机由 Claude 决定。Bun 从 Zig 移植到 Rust 的案例报告约 750,000 行、11 天、99.8% 测试通过、数百并行 agents 和每文件双 reviewer；文章同时注明尚未用于生产，因此应作为发布方案例，而非对照实验结论。

Opus 4.8 System Card 又提供了独立于产品 Dynamic Workflows 的多 Agent benchmark harness，面试时不要把二者混成同一系统：

| 评测与配置 | 发布方观察 | 解释边界 |
|---|---|---|
| BrowseComp，blocking orchestrator | 88.5% | 该 harness 的最高分；不能单独推出它的延迟/成本更优 |
| BrowseComp，fixed 5-agent team vs single agent | 85.4% vs 84.3%；total token limit 为 5M vs 10M，派生 latency 约为 single 的 20% | 质量略高且 latency 更低，但仍消耗较多 tokens；这是指定配置下的比较 |
| BrowseComp 难度切片 | 100% 历史通过率的易题没有明显 speedup；历史通过率低于 0.5 的 hard tail 中位约 3× | 难度由先前模型通过率代理，收益集中在慢的困难题 |
| ProgramBench，166 个 golden tasks | 三 Agent team 在 score 0.6 时约 1.8× latency improvement | 从 200 题中排除 34 个参考实现质量不足的题；分数—成本曲线依赖该筛选与 harness |

System Card 的 “latency” 是派生值：把所有 Agent 的 token 数按固定 prefill/decode rate 折算，再加工具执行时间；它不是生产环境的实际 wall-clock 或 p95 SLO。测试还分别定义 blocking、peer team 与 asynchronous subagents 的上下文、工具和预算，见 [System Card §8.11](https://www.anthropic.com/claude-opus-4-8-system-card)。因此公平的 multi-agent 面试回答至少要同时给任务成功率、全体 Agent token、串并行关键路径、工具时间、真实 wall-clock 和 verifier 质量，并按简单/困难任务切片。

System Card 还把 coding honesty 具体化为“未完成事项是否在总结中被主动披露”：预填充短上下文 coding traces 中，Opus 4.8 漏报重要失败事件为 3.7%，但该评测是 off-policy transcript，不能写作普适 honesty rate。最终仍应由测试和 artifact verifier 判断交付状态，而不是让模型自我宣布完成。

## 9.20 常见失败模式

1. 角色分工不清：每个 Agent 都在规划、执行和总结。
2. Coordinator 分配错误：任务给错角色，后面再努力也难补救。
3. 多个 Agent 重复做同一件事。
4. Agent 互相传递错误结论。
5. Judge 被流畅表达误导。
6. 冲突没有解决就输出答案。
7. 通信成本超过收益。
8. 权限边界混乱。
9. 共享 memory 或 blackboard 被污染。
10. 最终结果没人负责。
11. 简单任务过度使用多 Agent。
12. 缺少单 Agent baseline，无法证明增益。

Multi-Agent 的目标不是把系统变热闹，而是提高任务成功率、可验证性、并行效率和权限可控性。

## 9.21 何时值得拆成多个 Agent：一个决策案例

假设任务是生成一份供应商尽调报告。它包含合同条款核对、历史事故检索、财务指标整理、风险分级和最终建议。拆分成 researcher、contract reviewer、risk analyst 和 verifier 可能有价值，因为这些子任务使用不同资料和权限，也可以并行执行；但如果任务只是把一段已经给出的文字改成更短的摘要，拆成 planner、writer、reviewer 和 summarizer 只会增加通信轮数。

判断是否拆分时，可以先列出单 Agent baseline 的瓶颈：是上下文装不下，还是需要独立验证；是任务可以并行，还是权限必须隔离；是单 Agent 缺少工具，还是只是 prompt 不清楚。只有当角色分离直接解决其中一个瓶颈，Multi-Agent 才有明确的因果理由。

还要计算净收益。假设并行让墙钟时间从 120 秒降到 70 秒，但模型调用、通信和汇总成本翻倍，且冲突率增加，那么它可能只在有严格延迟目标的请求上值得使用。相反，如果 reviewer 能发现单 Agent 经常漏掉的高风险事实，质量收益可能超过额外成本，但必须用配对任务和相同预算进行验证。

最终选择可以是三种之一：单 Agent、固定 workflow，或 workflow 中局部使用多个专门 Agent。不要把架构选择变成全局开关；任务复杂度、风险、可并行性和工具权限都可能随请求变化。

## 9.22 从角色到责任：设计一个可审计的协作系统

设计时先为每个角色写一份契约，而不是先创建多个聊天窗口。契约应说明输入 schema、输出 schema、可见上下文、可调用工具、读写权限、预算、失败状态和交付对象。例如 researcher 只能提交带来源的证据包，coder 才能生成 patch，reviewer 只能读取 diff、测试结果和证据，不应直接修改代码或替代最终权限判断。

Coordinator 维护全局状态，但不应成为所有内容的无条件转发器。它要根据任务依赖选择下一个 Agent，过滤上下文，合并 artifact，发现缺失字段，处理超时和冲突，并在成本或风险超过阈值时停止。共享 blackboard 可以保存状态索引和证据引用，完整对话应留在各自 trace 中，避免每个 Agent 都读取所有历史。

消息协议要区分事实、假设、建议和请求。一个 researcher 说“文档记录了 X”与说“因此应该执行 Y”具有不同的风险；前者可以作为待验证证据，后者不能直接成为高权限动作。接收方需要检查发送者权限、来源、时间、schema 和是否支持当前子任务，而不是因为消息来自内部 Agent 就默认可信。

责任归因需要绑定 owner 和事件。每个子任务要记录分配者、执行者、输入版本、输出 artifact、验证者和后续使用者；最终结论要能追溯到支持它的消息和工具结果。发生错误时，团队才能判断是分配错误、检索错误、消息污染、验证遗漏还是协调器过早停止。

## 9.23 Debate、Verifier 与投票：三个不同的协作机制

Debate 的作用是制造对立假设和反例。它可以让一个 Agent 解释为什么支持某个结论，另一个 Agent 寻找边界条件，再由协调器决定哪些问题需要外部验证。Debate 的输出通常是候选假设和未决问题，而不是事实证明；如果参与者共享同一错误前提，辩论可能只是把错误包装得更有说服力。

Verifier 的作用是检查候选结果是否满足外部可观察条件。数学 verifier 可以重新计算，代码 verifier 可以运行测试，RAG verifier 可以检查 claim 与引用，业务 verifier 可以检查权限和对象状态。Verifier 不一定需要是一个模型，确定性规则和工具往往更适合高风险约束。

投票的作用是聚合多个候选输出。它适合答案空间相对明确、候选错误较独立的任务；若所有 Agent 使用同一模型、同一检索结果和同一提示，票数并不代表独立证据。加权投票还必须记录权重来源、校准集和失效切片，不能只按主观可信度排序。

一个稳健的组合是：先用 debate 展开候选与反例，再用 verifier 检查可验证条件，最后在仍有多个合格候选时投票或请求人工选择。顺序不能反过来把投票结果当作验证，也不能把 judge 的语言流畅度当作事实依据。

## 9.24 小练习

1. 给一个“写一份竞品调研报告”的任务，设计 4 个 Agent 角色、每个角色的工具权限和消息 schema。
2. 给一个“修复代码 bug 并提交 patch”的任务，说明为什么 coder 和 reviewer 不应该拥有完全相同的权限。
3. 设计一个冲突处理规则：当 researcher 和 verifier 对事实结论不一致时，系统应该如何升级。
4. 修改本章 demo，让 `debate_answer` 的冲突被正确解决，观察 `conflict_resolution_rate` 和 `all_checks_pass` 如何变化，并检查是否还有其他失败维度。
5. 为一个高风险外部动作设计统一 permission engine，说明为什么 coordinator、reviewer 和内部 Agent 的身份不能替代它。
6. 记录一次并行运行的 wall-clock、模型 token、消息 token、验证成本和人工升级成本，计算 NetLift，并与串行 workflow 比较。
7. 构造两个共享同一错误文档的 Agent，说明为什么多数票没有增加独立证据；再加入一个外部工具 verifier，比较冲突识别结果。
8. 比较单 Agent、固定 workflow 和 Multi-Agent 在成本、可靠性、可解释性上的差异。
9. 用同一任务复刻 BrowseComp 风格的 single、blocking、fixed-team、async 四条曲线；分别累计所有 Agent token、模型关键路径、工具时间和实际 wall-clock，说明 benchmark 的派生 latency 为什么不能当线上 SLO。
10. 为一个跨数百文件的长迁移设计 Dynamic Workflow：写出任务 DAG、独立验证与反驳回合、checkpoint/resume、取消和预算/权限门禁；用故意失败的测试验证“状态已保存”“任务已完成”和“artifact 已验收”是三个不同状态。

## 9.25 本章小结

Multi-Agent 通过角色分工、并行执行和互相检查，让 Agent 系统有机会处理更复杂任务。但它不是免费午餐，会引入通信成本、协调复杂度、冲突解决、权限管理、安全风险和评估难度。

可靠的 Multi-Agent 系统通常不是一群 Agent 自由聊天，而是有 coordinator、结构化消息、共享状态、明确权限、冲突解决和评估闭环的工程系统。下一章会进入 Agent 评估，系统讨论如何衡量 Agent 是否真的完成任务、是否安全、是否高效。

本章的代表性资料入口：

- [AutoGen](https://microsoft.github.io/autogen/)：多 Agent 编排框架和消息协作入口。
- [CAMEL](https://www.camel-ai.org/)：角色扮演和多 Agent 研究/工程入口。
- [MetaGPT](https://arxiv.org/abs/2308.00352)：以软件组织角色组织多 Agent 工作流的研究入口。
- [ChatDev](https://arxiv.org/abs/2307.07924)：软件开发场景中的多 Agent 协作研究入口。
- [AI Safety via Debate](https://arxiv.org/abs/1805.00899)：用辩论暴露和检查模型结论的研究入口。
- [LLM-based Multi-Agent Systems survey](https://arxiv.org/abs/2402.01680)：多 Agent 系统的综述入口。
- [Introducing dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)：Anthropic 对长周期动态编排、subagents、复核、恢复与使用门禁的产品描述。
- [Claude Opus 4.8 System Card](https://www.anthropic.com/claude-opus-4-8-system-card)：§8.11 的多 Agent benchmark harness、score/token/派生 latency 曲线，以及 §6.3.6 的 coding diligence 评测。

这些资料分别涉及框架、角色化工作流、软件开发协作、辩论和综述。它们可以帮助理解架构空间，却不能单独证明多 Agent 比单 Agent 更便宜、更可靠或更安全；具体结论仍需在相同任务、相近预算、明确 baseline、权限配置和失败切片下复测。
