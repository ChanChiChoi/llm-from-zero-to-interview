# 第四章：Planning 与 Task Decomposition：把复杂目标变成可验证工作

当任务只有一个明确动作时，Agent 可以直接调用工具。查询一个订单、计算一个表达式或读取一个文件，通常不需要复杂规划。但现实任务往往包含多个子目标、顺序依赖、资源限制和风险边界：先收集信息，再判断条件；先修改代码，再运行测试；先生成邮件草稿，再确认，最后才允许发送。

Planning 解决的是“接下来要完成哪些阶段、它们依赖什么、完成的证据是什么”；Task Decomposition 解决的是“如何把一个大目标拆成输入输出清楚、能够执行和验证的子任务”。两者不是生成一张漂亮的待办清单，而是建立一个可以被控制器执行、被观察结果修正、被验证器检查、在失败时回退的结构。

本章从一个代码修复任务开始，说明为什么分解需要保留原始目标和约束，再介绍子任务契约、依赖图、拓扑顺序、关键路径、并行与串行、一次性规划和动态重规划。随后讨论长期任务的 checkpoint、memory、计划版本和失败恢复，最后用一个 toy planning 审计程序检查目标覆盖、可执行性、验收标准、依赖、重规划、风险确认、重复子目标和最终任务结果。

## 0. 资料边界与方法

Plan-and-Solve Prompting 提供先规划再求解的研究背景；LLM+P 说明语言模型可以与传统规划器连接；Tree of Thoughts、ReAct 和 Reflexion 分别提供搜索、行动反馈和修正方向。OpenAI Agents SDK 的 tools、guardrails 和 tracing 文档展示了公开 runtime 中的工具、控制和轨迹概念。

这些资料并不构成统一的规划协议。论文中的计划可能是文本步骤，生产系统中的计划也可能是 DAG、状态机、队列、数据库记录或传统调度器对象。规划质量必须绑定模型、工具、环境、约束、预算和执行器版本，不能仅凭计划文字是否完整来判断。

本章中的代码修复、部署、退款制度和报告任务是教学构造。高风险步骤只讨论确认、权限、回滚和审计，不讨论规避审批或绕过安全控制的操作方法。

## 1. 为什么复杂任务需要 Planning

### 1.1 一个看似简单的请求

用户说：“分析这个项目的测试失败，修复代码，补充必要测试，并总结修改。”

这个请求至少包含：

1. 确认项目范围和可修改文件。
2. 运行测试并保存环境信息。
3. 找到失败测试和触发条件。
4. 阅读相关代码和已有测试。
5. 形成可解释的根因假设。
6. 设计最小修复。
7. 修改代码并补充回归测试。
8. 重新运行测试，检查回归。
9. 记录修改文件、验证结果和仍然存在的限制。

如果 Agent 没有规划，可能在没有测试证据时修改代码，在没有理解根因时增加补丁，忘记重新运行测试，或者只因为某一条命令返回成功就报告整个任务完成。规划的价值是把这些阶段和完成条件显式化，让每一步都有输入、输出和依赖。

### 1.2 从直观例子开始：把“大目标”变成“下一件可检查的事”

“修复项目”不是一件可以直接执行的动作；“运行相关测试并记录失败信息”才是一个可以执行和检查的子任务。完成这个子任务后，系统获得新的观察，才知道下一件事是阅读代码、检查环境还是追问用户。

好的分解不是把一句话切成很多短句，而是让每个子任务回答三个问题：现在需要什么输入，完成后会产生什么输出，怎样知道它已经完成。若这三个问题没有答案，子任务只是计划文本，不是工作单元。

### 1.3 形式化地看：规划是约束下的搜索和调度

从专家角度，规划同时包含四个问题：

1. **可达性**：从当前状态是否存在完成目标的动作路径。
2. **顺序性**：哪些子任务必须先完成，哪些可以并行。
3. **资源性**：路径需要多少 token、工具调用、时间、费用和人工确认。
4. **风险性**：哪些动作会产生副作用，失败后能否回滚。

纯文本模型可以提出候选计划，但可达性、依赖和风险最好由独立校验器、规则、调度器或执行器再次检查。规划器越接近外部副作用，越不能只凭语言流畅度判断质量。

## 2. Task Decomposition：子任务不是清单项目

### 2.1 子任务的基本契约

一个可执行子任务至少需要：

1. 子目标和作用范围。
2. 输入及其来源。
3. 预期输出。
4. 前置条件和依赖。
5. 可调用的动作或工具。
6. 验收标准。
7. 风险等级和权限要求。
8. 失败后的恢复或回退方式。
9. 资源预算和预计耗时。

例如“定位测试失败原因”可以具体写成：输入是测试失败日志和当前代码版本，输出是失败测试、触发条件、相关函数、根因假设和证据，验收标准是至少有一条可复现命令和一处代码位置支持该假设。

### 2.2 粒度过粗

“修复项目”“处理数据”“完成报告”都过于粗。它们没有说明先做什么、需要哪些资源，也无法在中途判断进展。粗粒度任务会把规划难题推回模型的下一轮，导致工具调用仍然临时发生。

### 2.3 粒度过细

“打开文件 A”“读取第 1 行”“读取第 2 行”通常过细。它们产生大量调度和日志成本，计划对环境变化非常敏感，还可能阻止 Agent 根据观察选择更有效的工具。

一个合适的粒度通常是“一次工具调用或一组紧密相关动作可以推进的子目标”：运行相关测试并记录失败、定位失败函数和边界条件、做最小修复并增加回归、重新运行测试并检查差异。这些任务既能执行，也能通过结果判断是否完成。

### 2.4 分解不能丢失原始约束

用户说“不要改无关文件”“只读检查”“不要自动提交”，这些约束必须沿着分解传播到子任务。若规划器只保留“完成报告”而丢掉“不要发信”，后续子任务可能扩大动作范围。

可以把原始目标看成不可变的约束集合，把子任务看成在约束下产生的工作集合。局部子任务完成不能覆盖全局禁止事项，局部观察也不能改变用户的授权范围。

## 3. 需求、子目标和验收标准

### 3.1 先提取需求

把用户请求拆成需求集合时，应区分：

- 必须完成的结果；
- 必须保留的证据；
- 不能做的动作；
- 资源和时间限制；
- 需要用户补充或确认的条件；
- 失败时必须说明的边界。

“补充必要测试”是结果需求，“不要改无关文件”是范围约束，“提交代码前先展示 diff”是确认条件。它们不能都被当成普通步骤。

### 3.2 子目标的验收标准

子目标“修复 bug”的验收标准可以包括：相关测试通过、隐藏或变体测试没有明显回归、补丁范围只涉及允许文件、修改理由有代码证据。子目标“收集政策资料”的验收标准可以包括来源版本、引用位置、关键条款和冲突说明。

验收标准应该尽量连接到可观察事实，而不是“看起来合理”“解释得清楚”这类主观描述。开放任务仍然需要人工判断，但应把判断对象、证据和分歧记录下来。

### 3.3 局部完成和全局完成

所有子任务都执行过，不代表全局目标完成。某个子任务可能失败后被跳过，某个输出可能没有传给下游，某个高风险动作可能没有确认。规划器应保存每个子任务的状态：`pending`、`ready`、`running`、`done`、`failed`、`blocked` 或 `cancelled`，并说明状态变更原因。

### 3.4 验收条件的传播

上游输出会成为下游输入。若“定位根因”的验收标准没有满足，就不应该把“修改代码”标记为 ready；若“验证修复”的测试失败，就不能把“总结结果”写成成功。验收条件是依赖图中的约束，不是最后才执行的一次检查。

## 4. 规划的形式化表示

### 4.1 原始需求和子任务集合

设用户目标包含需求集合：

~~~math
\mathcal{R}_g=\{r_1,r_2,\ldots,r_m\}
~~~

任务分解产生子任务集合：

~~~math
\mathcal{V}=\{v_1,v_2,\ldots,v_n\}
~~~

每个子任务可以抽象为：

~~~math
v_i=(q_i,I_i,O_i,A_i,D_i,C_i,\rho_i,b_i)
~~~

其中：

- `q_i` 是子目标描述；
- `I_i` 是输入及来源；
- `O_i` 是预期输出；
- `A_i` 是允许的动作或工具；
- `D_i` 是依赖的子任务集合；
- `C_i` 是验收条件；
- `\rho_i` 是风险级别；
- `b_i` 是该子任务的资源预算。

### 4.2 目标覆盖

令 `R(v_i)` 表示子任务覆盖的需求，目标覆盖率为：

~~~math
C_{\mathrm{goal}}=
\frac{\left|\left(\bigcup_i R(v_i)\right)\cap\mathcal{R}_g\right|}
{|\mathcal{R}_g|}
~~~

这里要求 `|R_g|>0`。它回答“分解是否覆盖了原始目标”；若没有任何
可评估的原始需求，覆盖率应为 `None`，而不是把空集合当成完全覆盖。
覆盖率高不代表任务可执行，因为需求可能由顺序错误、缺少输入或没有
验收标准的子任务覆盖。

### 4.3 依赖图

把子任务表示为有向图：

~~~math
G=(\mathcal{V},\mathcal{E})
~~~

若 `(v_i,v_j)\in\mathcal{E}`，表示 `v_j` 依赖 `v_i` 的输出。一个合法执行顺序 `\pi` 应满足：

~~~math
(v_i,v_j)\in\mathcal{E}
\Rightarrow
\pi(v_i)<\pi(v_j)
~~~

如果图存在环，说明规划包含互相等待的依赖，必须拆开、增加初始条件或承认任务不可达。拓扑排序只能检查顺序，不能证明子任务本身有意义，因此还要检查输入、输出和验收。

### 4.4 关键路径

若 `c_i` 是子任务的时间或成本，依赖图中的关键路径长度为：

~~~math
L_{\mathrm{critical}}=
\max_{p\in\mathcal{P}(G)}\sum_{v_i\in p}c_i
~~~

`\mathcal{P}(G)` 是非空图中的路径集合，`c_i` 是非负且有限的时间或
成本。空图没有可比较的关键路径，应报告 `None`；不能把长度 `0` 解读
成任务已经完成。即使所有互不依赖的子任务都能并行，任务耗时也不能
低于关键路径。若关键路径包含高风险写入或昂贵人工确认，应优先优化
这些环节，而不是盲目增加并行度。

### 4.5 规划指标的基础形式

可执行子任务比例：

~~~math
R_{\mathrm{exec}}=\frac{1}{n}\sum_{i=1}^{n}I_{\mathrm{exec}}(v_i)
~~~

验收标准覆盖率：

~~~math
R_{\mathrm{accept}}=\frac{1}{n}\sum_{i=1}^{n}I_{\mathrm{accept}}(C_i)
~~~

依赖违规率：

~~~math
R_{\mathrm{dep}}=\frac{1}{|\mathcal{E}|}
\sum_{(v_i,v_j)\in\mathcal{E}}I_{\mathrm{violation}}(v_i,v_j)
~~~

`I_exec` 表示子任务是否具备可执行输入、工具和动作，`I_accept` 表示是否有非空且可检查的验收条件，`I_violation` 表示执行顺序是否违反依赖。分母为零时应在实现中定义明确的空图策略，而不是默默把结果当成满分。
因此，`n>0` 时才计算前两个比例，`|E|>0` 时才计算依赖违规率；没有
节点、没有边或没有可评估事件时，指标值是 `None`。`None` 代表尚未
测量，不代表规划质量为好或为坏。

## 5. 计划表示：列表、DAG 和状态机

### 5.1 线性列表

线性列表最容易阅读，适合简单、顺序固定的任务。它的问题是无法表达并行、条件分支和回退。如果某一步失败，后面的步骤是等待、取消、替代还是回到上游，列表没有足够的信息。

### 5.2 依赖 DAG

DAG 用节点表示子任务，用边表示依赖，适合表达并行和汇合。它可以让调度器找出当前 ready 节点，也能在执行前检测依赖违规。DAG 仍然需要每个节点的状态、输出 schema、错误策略和资源信息，否则只是一个画出来的流程图。

### 5.3 状态机

状态机适合有明确状态和转移的流程，例如草稿、待确认、已提交、处理中、完成、失败和需要人工。它对高风险动作的边界很清楚，但面对开放探索和大量动态分支时，状态数量可能快速增长。

### 5.4 分层计划

分层计划把高层目标拆成阶段，再由低层规划器将每个阶段拆成动作。高层可以保持全局约束，低层可以根据观察灵活执行。层级之间需要明确输出契约，否则高层的“收集证据”可能无法告诉低层何时算完成。

### 5.5 选择表示方式

任务越固定，越适合 workflow 或状态机；依赖越复杂，越需要 DAG；环境越开放，越需要状态、候选动作和动态重规划。生产系统可以混合使用：关键审批和写入用状态机，资料调查用 DAG，开放检索阶段使用 Agent planner。

## 6. 计划粒度与执行边界

### 6.1 粗计划

“检查、修改、验证、总结”是合理的阶段级计划，但还不足以直接执行。它适合向用户展示整体方向，也适合在长任务中作为稳定的上层结构。

### 6.2 细计划

“运行登录测试”“读取登录函数”“检查空密码分支”“增加回归测试”更接近可执行单元。细计划应在需要执行或确认时展开，而不是从任务开始就把所有可能的文件和行号写死。

### 6.3 计划和工具调用的边界

计划节点描述要达到的子目标，动作描述当前如何推进。一个节点可能需要多个动作，也可能在第一次动作后发现无需继续。把每一个思考句子都当成节点，会让计划被无效文本占满；把高风险写入和验证合并成一个节点，会让权限和回滚边界消失。

## 7. 一次性规划、动态规划和混合规划

### 7.1 一次性规划

一次性规划先生成完整计划，再按顺序执行。优点是全局结构清楚，便于用户预览和估算成本；缺点是早期信息不足，环境变化后容易执行失效路径。它适合流程稳定、输入完整、依赖明确且副作用可控的任务。

### 7.2 动态重规划

动态规划在每次观察后重新判断后续路径。它适合调试、浏览器、开放检索和不确定环境，但会增加模型调用、状态管理和计划漂移风险。动态重规划不能变成“每轮从零开始”，否则系统会忘记已经完成的工作和原始限制。

### 7.3 混合规划

常见的工程方案是：先生成阶段级粗计划，再在阶段内部逐步决策；遇到失败、冲突、权限或预算变化时，只重规划受影响的子图。这样既保留全局方向，也避免把错误的细节执行到底。

### 7.4 计划版本和差异

每次重规划都应产生版本号和差异：保留了哪些节点，取消了哪些节点，新增了哪些节点，哪些依赖改变，以及改变原因是什么。版本差异比保存多份完整计划更容易审计，也能避免恢复时误用旧动作。

## 8. 依赖关系和拓扑执行

### 8.1 依赖的类型

子任务依赖可以分成：

1. 信息依赖：必须先获得资料或事实。
2. 资源依赖：必须先拿到文件、连接或权限。
3. 顺序依赖：动作必须按先后发生。
4. 验证依赖：必须先通过测试或检查。
5. 安全依赖：必须先预览、确认或人工批准。
6. 版本依赖：必须基于同一个资源版本执行。

依赖类型不同，失败处理也不同。信息依赖失败可能需要换来源；安全依赖失败通常不能换工具绕过；版本依赖失败需要重新读取资源，而不是继续使用旧快照。

### 8.2 Ready、Blocked 和 Cancelled

一个节点只有在所有硬依赖完成、输入可用、权限允许和预算足够时才进入 `ready`。依赖失败、权限缺失、等待确认或资源版本冲突时应是 `blocked`，而不是伪装成 `pending`。如果上游失败使节点不再有意义，可以标记为 `cancelled`，并保存取消原因。

### 8.3 依赖违规的后果

在验证完成前部署、在定位根因前修改、在确认收件人前发信，都是依赖违规。即使动作结果偶然正确，也说明系统缺少必要的约束。依赖检查应在规划阶段和执行阶段各做一次，因为计划可能在重规划后发生变化。

## 9. 并行任务和结果合并

### 9.1 哪些任务可以并行

多个只读检索、互不依赖的指标计算和不同来源的背景收集可以并行，前提是它们不会修改共享资源，不超过工具限流，且最终结果可以合并。并行节点应有独立的输入快照和结果 ID，避免一个节点悄悄改变另一个节点的环境。

### 9.2 哪些任务必须串行

共享资源写入、依赖前置验证、顺序敏感的数据库更新和需要用户确认的动作通常必须串行。两个 Agent 同时修改同一个文件或同一条业务记录，若没有版本检查和冲突合并，可能造成不可见覆盖。

### 9.3 并行的代价

并行可以降低墙钟时间，却可能增加 token、工具费用、限流、状态合并和错误传播。若某个分支结果不可信，汇总器还要决定是否等待、丢弃、重新检索或请求人工。并行度应由关键路径、合并成本和风险共同决定。

### 9.4 冲突合并

汇总阶段不能简单把所有结果拼接。对同一事实的不同来源，应记录版本、时间、来源等级和冲突；对多个代码补丁，应检查是否修改同一行和测试是否覆盖；对多个计划建议，应保留各自假设并让独立验证器选择。冲突本身是需要处理的状态，不是可以被平均掉的噪声。

### 9.5 一个并行调查例子

设任务是“比较三家供应商的交付周期和售后条款，并给出带引用的采购建议”。可以把资料收集拆成三个相互独立的只读节点：供应商 A、供应商 B、供应商 C。它们共享同一份问题 schema，但各自保存来源、抓取时间和文档版本。三条分支完成后，再进入“字段对齐”和“冲突核查”，最后才生成建议。

这个例子里，“收集三家资料”可以并行，“比较字段”必须等待所有必要资料，“生成建议”又依赖比较结果和冲突状态。如果供应商 B 的资料过期，汇总节点不能用 A、C 的结果掩盖缺失，而应把 B 标记为待补充或降低结论强度。并行的正确性来自依赖和合并契约，而不是启动了多少个子 Agent。

可以用如下结构记录合并输入：

~~~text
evidence = {
  supplier: ..., version: ..., observed_at: ...,
  source: ..., fields: ..., confidence: ...
}
~~~

合并器先按字段和版本对齐，再处理冲突。若两个来源都声称“交付 7 天”，可以合并为一致事实；若一个写 7 天、一个写 14 天，系统应保留冲突并触发补充检索或人工判断。把冲突直接交给模型用一句“综合来看”消掉，会损失重要的不确定性。

## 10. 长期任务、Checkpoint 和恢复

### 10.1 为什么一次上下文不够

持续监控、工单处理、知识库整理和实验跟踪可能持续数小时或数天。模型上下文会被压缩、服务可能重启、权限可能过期、外部资源可能改变，不能只把全部历史放在一次 prompt 中。

### 10.2 Checkpoint 内容

一个可恢复 checkpoint 至少包含：

1. 目标和约束版本。
2. 当前计划版本和节点状态。
3. 已确认事实及来源。
4. 未完成、阻塞和取消的节点。
5. 最近工具结果和外部 request id。
6. 资源版本、幂等键和副作用状态。
7. 剩余预算和权限有效期。
8. 最后一次安全可恢复位置。

Checkpoint 不是把模型最后一句话存下来。恢复程序必须验证外部状态，再决定是否可以重放动作。写操作若状态未知，应先查询而不是从旧 checkpoint 直接重发。

### 10.3 断点恢复

恢复时先加载计划和状态摘要，再验证资源版本、权限和未完成节点。若依赖图已经变化，创建新计划版本；若目标已被用户修改，暂停旧计划并请求确认。恢复成功的标准是状态一致，而不是模型继续输出。

### 10.4 持久执行与重复副作用

长期任务的调度器可能在节点执行后、状态写回前崩溃，也可能在状态写回后没有收到执行器的响应。此时不能简单依赖“每个节点只执行一次”的假设。读操作可以通过请求 ID 重放；写操作需要幂等键、外部状态查询或明确的 pending 状态。

例如，邮件服务已经接受请求，但 Agent 进程在收到响应前退出。恢复程序看到节点仍是 `running`，正确动作是查询 request id，而不是再次发送。若服务没有查询接口，系统应把节点标记为 `unknown` 并交给人工。所谓 exactly-once 往往需要执行器、存储和外部服务共同支持，不能由 planner 的文本承诺实现。

### 10.5 检查点的粒度

每个动作都保存 checkpoint 会产生大量存储和恢复成本，只在阶段结束保存又可能丢失关键外部状态。常见做法是对只读探索按若干节点保存，对有副作用的动作在执行前后分别保存，对计划版本变化立即保存。粒度选择应由重放成本、数据敏感性和副作用风险决定。

## 11. Planning 与 Memory

### 11.1 短期任务状态

短期状态保存当前目标、计划、完成节点、观察、错误、预算和确认事项。它服务当前任务，不应自动跨用户或跨项目复用。

### 11.2 长期 memory

长期 memory 可以保存用户偏好、项目结构、常见错误和成功方案，但每条记忆应有来源、时间、作用域、可信度和删除策略。历史成功方案只是候选，不是当前事实。

### 11.3 记忆污染

过期配置、别的项目路径、其他用户偏好和未经验证的模型总结，都可能污染计划。读取 memory 后应进行权限过滤、时间检查和当前环境验证。对高风险动作，长期 memory 不能作为唯一授权或资源依据。

### 11.4 Context Folding

长任务需要压缩上下文时，应优先保留原始目标、禁止事项、权限、版本、未解决风险和关键证据；可压缩重复工具输出和已经确认的低价值细节。摘要必须指向原始证据，不能让摘要成为无法追溯的唯一事实来源。

## 12. Planner 的实现方式

### 12.1 Prompt-based Planner

直接让 LLM 生成计划最灵活，适合开放任务和快速试验。它容易产生幻觉依赖、漏掉验收条件和过度承诺，需要独立解析器和计划校验器。

### 12.2 Rule-based Planner

规则和模板稳定、可测试，适合审批、数据流水线和固定运维流程。它的缺点是难以覆盖开放分支，规则数量增长后维护成本高。

### 12.3 Hybrid Planner

混合规划让规则负责目标、权限、关键顺序和停止条件，让模型负责候选子任务、查询改写和局部路径。生产系统常用这种方式，因为模型的灵活性被限制在可观察、可验证的范围内。

### 12.4 Search-based Planner

搜索规划器生成多个分解或路径，再用成本、完成条件、风险和验证器评分。它可以避免一次错误计划决定全局，但会增加计算和选择错误；若评分器偏向计划长度或格式，搜索可能稳定选择漂亮但不可执行的计划。

### 12.5 Hierarchical 和 External Planner

分层规划器处理高层目标和低层动作；传统调度器、SAT/SMT 求解器、任务规划器或业务规则引擎可以处理严格约束。LLM 不必独自承担所有规划工作，关键是定义清楚输入输出、失败状态和责任边界。

### 12.6 规划成本和收益

规划本身也消耗资源。一次请求的规划成本可以粗略拆成：

~~~math
C_{\mathrm{plan}}=C_{\mathrm{model}}+C_{\mathrm{tool}}+C_{\mathrm{verify}}+C_{\mathrm{review}}
~~~

`C_model` 是生成和更新计划的模型成本，`C_tool` 是为规划收集信息的工具成本，`C_verify` 是依赖和验收检查，`C_review` 是人工确认。加入规划后的收益不应只看任务成功率，也要看每个成功任务的额外成本和延迟。

如果 baseline 的成功率为 `A_0`，加入规划后的成功率为 `A_1`，相应成本为 `C_0` 和 `C_1`，可以用教学性的边际收益表示：

~~~math
G_{\mathrm{plan}}=\frac{A_1-A_0}{C_1-C_0}
~~~

当 `C_1=C_0` 时不能直接使用这个比值；当规划提高了高风险任务的安全性而没有提高平均成功率时，也不能仅凭平均值否定规划。应按简单任务、难任务、高风险任务和需要恢复的任务分别看收益。
只有 `C_1-C_0\ne0` 且成本口径一致时，`G_plan` 才是有定义的边际比值；
成本没有增加时，应单独报告成功率差和成本差，而不是用无穷大或零替代。
如果任一成功率来自空样本，边际收益也应保持 `None`。

规划器还可能伤害简单任务：为了生成完整计划而增加两轮模型调用，最后只执行一个查询。路由器可以根据任务复杂度选择直接动作、阶段计划或动态规划，但路由本身也要纳入评估。

## 13. 规划质量如何评估

### 13.1 覆盖和可执行性

目标覆盖率回答计划是否遗漏需求，可执行率回答节点是否具备输入、动作、权限和资源。覆盖高但可执行低，说明计划像愿望清单；可执行高但覆盖低，说明系统在高效完成错误的部分任务。

### 13.2 验收和依赖

验收覆盖率检查每个子任务是否有可检查条件，依赖违规率检查实际顺序是否满足图约束。还应检查依赖是否真实存在：过度添加依赖会降低并行度，遗漏依赖则会产生错误或副作用。

### 13.3 重规划和恢复

失败发生后，系统是否更新了受影响计划，是否保留了原始目标，是否恢复到安全状态，都是独立指标。重规划覆盖率高不一定好，因为无意义地反复重写计划也可能表示不稳定；要同时报告重规划原因、后续成功率和额外成本。

### 13.4 关键路径、成本和任务结果

计划文本质量不能替代实际任务成功。报告应同时包括关键路径、总成本、并行度、P50/P95 延迟、人工介入、失败分支和最终成功率。若规划增加了大量步骤却没有提高变体任务成功率，应考虑减少规划或改进验证器。

### 13.5 一个小型 worked example

考虑“分析登录测试失败并提交一个待确认的修复草稿”。可以定义五个节点：

| 节点 | 输出 | 依赖 | 风险 |
| --- | --- | --- | --- |
| `run_tests` | 失败测试和环境信息 | 无 | 低 |
| `inspect_failure` | 根因假设和代码证据 | `run_tests` | 低 |
| `patch_code` | 本地补丁和 diff | `inspect_failure` | 中 |
| `rerun_tests` | 回归结果 | `patch_code` | 低 |
| `draft_summary` | 修改说明和待确认草稿 | `rerun_tests` | 中 |

合法路径是 `run_tests -> inspect_failure -> patch_code -> rerun_tests -> draft_summary`。如果 `run_tests` 失败且环境不可用，`inspect_failure` 不应被标为完成；如果 `rerun_tests` 仍失败，`draft_summary` 可以生成“未完成报告”，但不能写成“修复成功”。如果用户只授权生成草稿，`draft_summary` 可以结束；真正提交补丁则需要另一个明确的确认节点。

这个例子说明规划评估至少有三个层次：图结构是否合法，节点输出是否满足验收，整个任务是否满足用户目标。只检查拓扑顺序会漏掉测试失败，只检查最终文本会漏掉中间依赖和授权范围。

## 14. 最小可运行 Planning 审计实验

下面的 demo 不访问外部 API，只用五个教学计划检查目标覆盖、可执行性、验收标准、依赖顺序、重规划、失败恢复、高风险确认、计划长度、重复子目标和任务结果。数据故意包含缺验证、依赖违规、过度分解和虽然成功但有审计问题的计划。

~~~python
from collections import Counter


MAX_STEPS = 5


PLANS = [
    {
        "id": "code_fix_good",
        "requirements": {"inspect", "test", "patch", "verify", "summarize"},
        "replanned": False,
        "final_success": True,
        "steps": [
            {"id": "run_tests", "covers": {"test"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "inspect_failure", "covers": {"inspect"}, "deps": ["run_tests"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "patch_code", "covers": {"patch"}, "deps": ["inspect_failure"], "executable": True, "acceptance": True, "risk": "medium", "confirmed": True, "status": "done", "recovered": True},
            {"id": "rerun_tests", "covers": {"verify"}, "deps": ["patch_code"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "summarize", "covers": {"summarize"}, "deps": ["rerun_tests"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "missing_verify",
        "requirements": {"inspect", "patch", "verify", "summarize"},
        "replanned": False,
        "final_success": False,
        "steps": [
            {"id": "inspect", "covers": {"inspect"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "patch", "covers": {"patch"}, "deps": ["inspect"], "executable": True, "acceptance": True, "risk": "medium", "confirmed": True, "status": "done", "recovered": True},
            {"id": "summarize", "covers": {"summarize"}, "deps": ["patch"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "deploy_dependency_violation",
        "requirements": {"collect", "evaluate", "deploy"},
        "replanned": False,
        "final_success": False,
        "steps": [
            {"id": "deploy", "covers": {"deploy"}, "deps": ["evaluate"], "executable": True, "acceptance": True, "risk": "high", "confirmed": False, "status": "blocked", "recovered": False},
            {"id": "collect_metrics", "covers": {"collect"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "evaluate", "covers": {"evaluate"}, "deps": ["collect_metrics"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "replan_success",
        "requirements": {"find_policy", "answer"},
        "replanned": True,
        "final_success": True,
        "steps": [
            {"id": "search_docs", "covers": {"find_policy"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "failed", "recovered": True},
            {"id": "alternate_search", "covers": {"find_policy"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "answer", "covers": {"answer"}, "deps": ["alternate_search"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "over_decomposed_loop",
        "requirements": {"diagnose", "answer"},
        "replanned": False,
        "final_success": False,
        "steps": [
            {"id": "think_1", "covers": set(), "deps": [], "executable": False, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "think_2", "covers": set(), "deps": ["think_1"], "executable": False, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "search_logs", "covers": {"diagnose"}, "deps": ["think_2"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "search_logs_again", "covers": {"diagnose"}, "deps": ["search_logs"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "answer_without_evidence", "covers": {"answer"}, "deps": ["search_logs_again"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "extra_check", "covers": set(), "deps": ["answer_without_evidence"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
]


def rate(num, den):
    if (
        not isinstance(num, int)
        or isinstance(num, bool)
        or not isinstance(den, int)
        or isinstance(den, bool)
    ):
        raise TypeError("rate expects integer counts")
    if den < 0 or num < 0 or num > den:
        raise ValueError("rate requires 0 <= numerator <= denominator")
    return None if den == 0 else round(num / den, 3)


def at_least(value, threshold):
    return value is not None and value >= threshold


def is_zero(value):
    return value is not None and value == 0.0


def dependency_violations(plan):
    done_or_seen = set()
    violations = []
    for step in plan["steps"]:
        for dep in step["deps"]:
            if dep not in done_or_seen:
                violations.append((step["id"], dep))
        done_or_seen.add(step["id"])
    return violations


def repeated_cover(plan):
    seen = set()
    repeats = 0
    for step in plan["steps"]:
        cover_key = tuple(sorted(step["covers"]))
        if cover_key and cover_key in seen:
            repeats += 1
        if cover_key:
            seen.add(cover_key)
    return repeats > 0


def critical_path_len(plan):
    by_id = {step["id"]: step for step in plan["steps"]}
    memo = {}
    visiting = set()

    def depth(step_id):
        if step_id in memo:
            return memo[step_id]
        if step_id in visiting:
            raise ValueError(f"dependency cycle at {step_id}")
        visiting.add(step_id)
        step = by_id[step_id]
        if not step["deps"]:
            memo[step_id] = 1
        else:
            if any(dep not in by_id for dep in step["deps"]):
                raise ValueError(f"unknown dependency in {step_id}")
            memo[step_id] = 1 + max(
                depth(dep) for dep in step["deps"]
            )
        visiting.remove(step_id)
        return memo[step_id]

    return max((depth(step["id"]) for step in plan["steps"]), default=0)


def critical_path_metric(plans):
    lengths = [critical_path_len(plan) for plan in plans if plan["steps"]]
    return max(lengths) if lengths else None


def plan_gain(success_before, success_after, cost_before, cost_after):
    if cost_after == cost_before:
        return None
    if success_before is None or success_after is None:
        return None
    return round((success_after - success_before) / (cost_after - cost_before), 3)


def audit_plans(plans):
    all_steps = [step for plan in plans for step in plan["steps"]]
    coverage_scores = []
    failed_or_blocked_plans = []
    dep_violation_plans = []
    overlong_plans = []
    repeated_plans = []
    missing_acceptance_plans = []
    for plan in plans:
        covered = set().union(*(step["covers"] for step in plan["steps"])) if plan["steps"] else set()
        coverage_scores.append(rate(len(covered & plan["requirements"]), len(plan["requirements"])))
        if any(step["status"] in {"failed", "blocked"} for step in plan["steps"]):
            failed_or_blocked_plans.append(plan)
        if dependency_violations(plan):
            dep_violation_plans.append(plan["id"])
        if len(plan["steps"]) > MAX_STEPS:
            overlong_plans.append(plan["id"])
        if repeated_cover(plan):
            repeated_plans.append(plan["id"])
        if any(not step["acceptance"] for step in plan["steps"]):
            missing_acceptance_plans.append(plan["id"])

    high_risk_steps = [step for step in all_steps if step["risk"] == "high"]
    failed_or_blocked_steps = [step for step in all_steps if step["status"] in {"failed", "blocked"}]
    metrics = {
        "goal_coverage": (
            round(sum(coverage_scores) / len(coverage_scores), 3)
            if coverage_scores and all(score is not None for score in coverage_scores)
            else None
        ),
        "executable_step_rate": rate(sum(step["executable"] for step in all_steps), len(all_steps)),
        "acceptance_coverage": rate(sum(step["acceptance"] for step in all_steps), len(all_steps)),
        "dependency_violation_rate": rate(len(dep_violation_plans), len(plans)),
        "replan_coverage": rate(sum(plan["replanned"] for plan in failed_or_blocked_plans), len(failed_or_blocked_plans)),
        "failure_recovery_rate": rate(sum(step["recovered"] for step in failed_or_blocked_steps), len(failed_or_blocked_steps)),
        "risk_confirmation_coverage": rate(sum(step["confirmed"] for step in high_risk_steps), len(high_risk_steps)),
        "overlong_plan_rate": rate(len(overlong_plans), len(plans)),
        "repeat_subgoal_rate": rate(len(repeated_plans), len(plans)),
        "task_success_rate": rate(sum(plan["final_success"] for plan in plans), len(plans)),
        "max_critical_path_len": critical_path_metric(plans),
    }
    reasons = Counter()
    for plan in plans:
        if plan["id"] in dep_violation_plans:
            reasons["dependency_violation"] += 1
        if plan["id"] in overlong_plans:
            reasons["overlong_plan"] += 1
        if plan["id"] in repeated_plans:
            reasons["repeated_subgoal"] += 1
        if plan["id"] in missing_acceptance_plans:
            reasons["missing_acceptance"] += 1
        if not plan["final_success"]:
            reasons["task_failed"] += 1
        if any(step["risk"] == "high" and not step["confirmed"] for step in plan["steps"]):
            reasons["unconfirmed_high_risk"] += 1
        covered = set().union(*(step["covers"] for step in plan["steps"])) if plan["steps"] else set()
        if covered != plan["requirements"]:
            reasons["goal_not_fully_covered"] += 1
    checks = {
        "coverage_ok": at_least(metrics["goal_coverage"], 0.90),
        "executable_ok": at_least(metrics["executable_step_rate"], 0.90),
        "acceptance_ok": at_least(metrics["acceptance_coverage"], 0.90),
        "dependency_ok": is_zero(metrics["dependency_violation_rate"]),
        "replan_ok": at_least(metrics["replan_coverage"], 0.50),
        "recovery_ok": at_least(metrics["failure_recovery_rate"], 0.80),
        "risk_ok": at_least(metrics["risk_confirmation_coverage"], 1.0),
        "length_ok": is_zero(metrics["overlong_plan_rate"]),
        "repeat_ok": is_zero(metrics["repeat_subgoal_rate"]),
        "success_ok": at_least(metrics["task_success_rate"], 0.75),
    }
    return {
        "metrics": metrics,
        "dependency_violations": {
            plan["id"]: dependency_violations(plan)
            for plan in plans
            if dependency_violations(plan)
        },
        "problem_plans": sorted(
            set(
                dep_violation_plans
                + overlong_plans
                + repeated_plans
                + missing_acceptance_plans
                + [plan["id"] for plan in plans if not plan["final_success"]]
            )
        ),
        "top_failure_reasons": reasons.most_common(),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


report = audit_plans(PLANS)
print("metrics=", report["metrics"])
print("dependency_violations=", report["dependency_violations"])
print("problem_plans=", report["problem_plans"])
print("top_failure_reasons=", report["top_failure_reasons"])
print("checks=", report["checks"])
print("all_checks_pass=", report["all_checks_pass"])
~~~

预期输出如下：

~~~text
metrics= {'goal_coverage': 0.95, 'executable_step_rate': 0.9, 'acceptance_coverage': 0.65, 'dependency_violation_rate': 0.2, 'replan_coverage': 0.5, 'failure_recovery_rate': 0.5, 'risk_confirmation_coverage': 0.0, 'overlong_plan_rate': 0.2, 'repeat_subgoal_rate': 0.4, 'task_success_rate': 0.4, 'max_critical_path_len': 6}
dependency_violations= {'deploy_dependency_violation': [('deploy', 'evaluate')]}
problem_plans= ['deploy_dependency_violation', 'missing_verify', 'over_decomposed_loop', 'replan_success']
top_failure_reasons= [('task_failed', 3), ('missing_acceptance', 2), ('repeated_subgoal', 2), ('goal_not_fully_covered', 1), ('dependency_violation', 1), ('unconfirmed_high_risk', 1), ('overlong_plan', 1)]
checks= {'coverage_ok': True, 'executable_ok': True, 'acceptance_ok': False, 'dependency_ok': False, 'replan_ok': True, 'recovery_ok': False, 'risk_ok': False, 'length_ok': False, 'repeat_ok': False, 'success_ok': False}
all_checks_pass= False
~~~

### 14.1 目标覆盖不代表计划可用

示例中目标覆盖率是 `0.95`，说明大部分需求出现在某个子任务的 `covers` 集合里。但 `missing_verify` 没有真正覆盖 `verify`，而 `over_decomposed_loop` 虽然覆盖了 `diagnose` 和 `answer`，却缺少有意义的验收标准。因此覆盖率需要和验收、可执行性、任务结果一起读。

### 14.2 依赖和高风险步骤

`deploy_dependency_violation` 先出现了依赖 `evaluate` 的 `deploy`，而且高风险步骤没有确认。即使后面补上了收集和评估，也不能把已经发生的顺序违规从报告中删除。真实执行器应在部署动作前阻断，而不是等审计程序事后发现。

### 14.3 重规划和恢复

`replan_success` 的第一次检索失败后换了来源并最终完成，说明重规划和失败恢复可以同时成功。但它仍有两个覆盖相同 `find_policy` 的子任务，因此被列入问题计划。一个任务成功不代表计划结构没有浪费或风险，质量分析要保留这些信息。

### 14.4 过度分解

`over_decomposed_loop` 把无实际输出的思考拆成两个节点，又重复搜索同一个诊断目标，还增加了没有明确验收的额外检查。关键路径长度达到 6，不表示系统真的获得了更多证据。过度分解会增加调度、上下文和失败面。

## 15. 计划失败的归因

### 15.1 覆盖失败

原始需求没有映射到任何子任务，说明目标解析或分解器遗漏了约束。先修需求 schema 和分解逻辑，不要只提高 planner 的生成预算。

### 15.2 可执行性失败

子任务没有输入、工具、权限或明确输出时，计划无法落地。规划器应在生成后运行静态检查，执行器还要在运行时再次检查当前环境。

### 15.3 依赖失败

若顺序不合法、依赖形成环或依赖的输出没有传递给下游，应修正图结构和状态映射。拓扑排序只能发现部分问题，不能代替业务语义和验收检查。

### 15.4 验收失败

子任务执行了但没有可检查的结果，通常说明计划把活动误当成产出。增加更多动作不一定有帮助，应先定义证据、测试、引用、状态或人工判断的验收方式。

### 15.5 重规划失败

失败后不断重写整个计划，可能造成成本和方向漂移；完全不重规划，可能执行失效路径。应记录触发原因、受影响子图、保留节点、取消节点和新预算，再评估新路径是否真的改善。

### 15.6 并行合并失败

并行分支的结果冲突、版本不一致或证据质量不同，说明汇总契约不足。修复重点是结果 schema、来源和版本，而不是简单增加并行 Agent 数量。

### 15.7 长期恢复失败

恢复后重复写入、使用旧权限或丢失用户限制，说明 checkpoint 不完整或恢复前没有重新验证外部状态。恢复系统必须把“可以重放”与“可以再次产生副作用”区分开。

## 16. 练习：把计划写成可执行结构

### 练习一：拆分代码修复

把“修复一个失败测试并补充回归测试”拆成子任务，给出每个节点的输入、输出、依赖、验收标准和风险。指出哪些节点可以在同一版本快照上并行，哪些必须串行。

### 练习二：找出计划漏洞

给一个包含“修改代码、运行测试、部署、总结”的计划，标出缺少的根因分析、验证依赖、确认和回滚。说明为什么“部署成功”不能替代“验证通过”。

### 练习三：构造依赖图

为“收集政策、查询订单、比较日期、生成建议”画 DAG，写出一个合法拓扑顺序，再加入“用户确认”节点，说明哪些边因此新增。

### 练习四：并行和合并

设计三个可以并行收集的资料来源，定义结果 schema、来源等级、时间戳和冲突处理规则。说明什么时候应放弃一个低可信分支，什么时候需要人工复核。

### 练习五：计划版本

为一次检索失败设计 `plan_v1` 和 `plan_v2`，记录失败观察、取消节点、新增节点、预算变化和为什么不能继续使用旧计划。

### 练习六：长期 checkpoint

为持续处理工单的 Agent 设计 checkpoint。至少包含当前工单、计划节点、权限有效期、外部 request id、未完成事项、幂等键和最后安全恢复点。

### 练习七：运行审计程序

把 `deploy_dependency_violation` 改成先收集再评估、经过确认后部署，把 `over_decomposed_loop` 合并成两个有验收标准的节点。重新运行程序，观察依赖、风险、长度、重复和任务结果指标的变化。

## 17. 本章小结

Planning 与 Task Decomposition 的目标不是生成更长的步骤清单，而是把大目标变成一组有输入、输出、依赖、验收、风险和恢复方式的子任务。计划必须保留原始目标和禁止事项，局部完成不能替代全局完成。

依赖图帮助系统区分串行和并行，关键路径决定理论上的最短耗时，结果合并需要来源、版本和冲突策略。一次性计划适合稳定流程，动态重规划适合信息不完整和环境变化的任务，混合方式通常更实用：上层保持阶段和约束，下层根据观察调整受影响子图。

长期任务还需要外部状态、计划版本、checkpoint、幂等和恢复验证。规划质量应同时看目标覆盖、可执行性、验收标准、依赖、重规划、失败恢复、风险确认、重复子目标、成本和最终任务成功率。一个计划写得完整，却不能执行、不能验证或不能安全恢复，仍然不是好计划。

下一章将进入 Memory 系统，讨论 Agent 如何保存当前任务状态、长期偏好和历史经验，以及如何处理记忆的时效、权限、污染和隐私边界。

## 18. 延伸资料与证据边界

1. Plan-and-Solve Prompting，论文：<https://arxiv.org/abs/2305.04091>。
2. LLM+P: Empowering Large Language Models with Optimal Planning Proficiency，论文：<https://arxiv.org/abs/2304.11477>。
3. Tree of Thoughts，论文：<https://arxiv.org/abs/2305.10601>。
4. ReAct，论文：<https://arxiv.org/abs/2210.03629>。
5. Reflexion，论文：<https://arxiv.org/abs/2303.11366>。
6. OpenAI Agents SDK 文档入口：<https://openai.github.io/openai-agents-python/>。

论文资料支撑规划、搜索和反馈修正的研究背景，官方文档支撑公开 Agent runtime 组件。生产系统中的计划表示、依赖检查、资源调度和安全策略可能完全不同；真实结论必须绑定自己的模型、工具、数据、环境、权限、调度器和回归集。
