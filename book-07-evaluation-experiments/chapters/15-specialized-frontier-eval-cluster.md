# 第 15 章：Specialized Frontier Eval Cluster：把模型放回真实工作流

## 15.1 从“总分更高”到“在哪些工作里更可靠”

一个新模型的总榜分数上升，并不等于它能修复生产代码、在终端完成任务、浏览网
页、发现安全问题、处理长周期研究或正确调用工具。用户真正关心的是：在某类工
作中，模型能否完成目标，失败时是否可恢复，成本是否可接受，副作用是否可控。

这些任务具有共同特征：

- 输入不是一段独立问题，而是仓库、网页、文档、日志或业务状态；
- 输出不是一段文字，而是 patch、文件、数据库状态、表格或外部提交；
- 模型需要多轮观察、行动和验证；
- 环境会返回错误、权限拒绝、空结果或与预期不同的状态；
- 成功、成本和风险之间存在真实取舍。

Specialized frontier eval cluster 不是一个单独 benchmark，而是一组围绕高价值或
高风险工作组织的评测簇。它把任务、环境、工具、验证器、预算和副作用放在一起，
用可复现的轨迹观察系统行为。

评测簇的目标不是制造更多榜单，而是回答：

1. 模型在哪一类工作流中真正有帮助；
2. 成功依赖模型、工具、检索还是人工接管；
3. 失败发生在哪个阶段；
4. 增加预算、重试或工具后，收益是否值得成本；
5. 这个系统适合什么风险等级和发布范围。

## 15.2 为什么旧 benchmark 会饱和

固定格式、公开数据和短答案任务接近天花板后，分数上涨可能来自记忆、提示技巧、
测试集污染或 judge 偏差。MMLU、GPQA、HumanEval 等仍有价值：它们可以测知识、
数学和局部代码回归；但它们不能单独证明 Agent 能力、长周期软件工程、浏览器交互
或安全边界。

静态问答与真实工作流之间至少有五个差异：

| 静态任务 | 工作流任务 |
| --- | --- |
| 输入一次性给出 | 信息需要搜索、读取和筛选 |
| 输出通常是文本 | 输出可能改变文件或外部状态 |
| 没有环境反馈 | 工具会返回错误、权限和空结果 |
| 成功由答案判断 | 成功由状态、测试和证据判断 |
| 失败通常可忽略 | 失败可能产生数据泄露或不可逆副作用 |

因此，专项评测不是把更多题目堆进同一个数据集，而是改变测量对象。

## 15.3 评测对象是任务、环境和轨迹

### 15.3.1 任务契约

把一次专项任务写成：

~~~math
\mathcal{T}=(x,E,A,G,B,H,R)
~~~

其中：

- x 是初始任务、用户目标和可见上下文；
- E 是可重置的环境状态；
- A 是允许的动作和工具；
- G 是成功判据；
- B 是时间、token、调用、金钱或动作预算；
- H 是评测 harness，包括路由、重试、记忆和判定器；
- R 是风险、权限和副作用约束。

小白可以把它理解成一张“工作任务卡”：给什么材料，允许做什么，做成什么样算
成功，最多花多少，哪些动作不能做。

专家要注意，G 和 R 必须同时存在。代码修复不仅要测试通过，还要禁止修改测试
和读取敏感路径；浏览器任务不仅要到达页面，还要确认账户权限和提交状态；科学
研究任务不仅要生成结论，还要说明证据和不确定性。

### 15.3.2 轨迹

模型输出不是单段文本，而是一条轨迹：

~~~math
\tau=(s_0,a_0,o_1,s_1,a_1,o_2,\ldots,s_T)
~~~

其中 s_t 是环境状态，a_t 是模型或工具动作，o_t 是观察结果。最终成功可以写成：

~~~math
S(\tau)=\mathbf{1}[G(\tau)=1]
~~~

但只保存 S(τ) 会丢失“先做对了什么、在哪一步走偏、错误是否可恢复”。因此要
保存完整 trace、工具参数、观察内容、环境版本和停止原因。

### 15.3.3 成本与风险

可以把一次轨迹的成本和风险分别记录：

~~~math
C(\tau)=
C_{\mathrm{token}}+
C_{\mathrm{tool}}+
C_{\mathrm{compute}}+
C_{\mathrm{human}}
~~~

~~~math
R(\tau)=
R_{\mathrm{permission}}+
R_{\mathrm{privacy}}+
R_{\mathrm{integrity}}+
R_{\mathrm{safety}}
~~~

工程报告可以展示一个效用教学式：

~~~math
U(\tau)=S(\tau)-\lambda C(\tau)-\mu R(\tau)
~~~

λ 和 μ 不是通用常数。高风险系统不能因为平均效用高，就用普通任务成功抵消一
次未授权写入；风险约束应保留独立的严重度和动作。

## 15.4 评测簇如何组织

一个专项 eval cluster 可以按四个轴组织：

1. 工作领域：软件、终端、浏览器、科学、医疗、网络安全、办公；
2. 操作形态：读取、搜索、编辑、执行、提交、规划、验证；
3. 时间跨度：单轮、短轨迹、跨 session、长期项目；
4. 风险等级：只读、可回滚写入、外部提交、不可逆副作用。

每个簇都应有主任务、边界任务、恢复任务和不可执行任务。只测主任务会高估系统
能力，只测危险任务又无法估计正常工作价值。

### 15.4.1 任务簇与 benchmark 的关系

Benchmark 可能提供一组标准任务、环境或公开分数；任务簇是项目自己的测量设计。
一个簇可以引用多个 benchmark，也可以包含真实脱敏任务、专家构造任务、历史事
故和新发现的边界。

| 对象 | 主要回答 | 典型证据 |
| --- | --- | --- |
| 公开 benchmark | 外部可比较的任务表现 | 公开数据、标准 grader |
| 内部任务簇 | 产品工作流是否完成 | 脱敏日志、专家契约 |
| 回归集合 | 已知错误是否复发 | 事故样本、稳定判定 |
| 线上抽样 | 真实分布和长期结果 | trace、用户行为、人工复核 |

### 15.4.2 任务卡

每个任务都应有独立任务卡，而不是只存一段 prompt：

~~~text
task_id / domain / difficulty / initial_state
allowed_tools / forbidden_actions / timeout / budget
success_predicate / partial_success / dangerous_failure
artifacts / hidden_tests / reset_procedure
grader_version / source_revision / risk_level
~~~

例如代码修复任务的成功条件不是“生成了 patch”，而是 patch 能应用、隐藏测试
通过、没有修改测试、没有访问禁止路径，且最终 diff 在允许范围内。浏览器任务
的成功条件也不是“打开了目标 URL”，而是页面状态改变到目标状态，账号权限没有
越界，付款或提交这类副作用符合任务授权。

## 15.5 软件工程与终端任务评测

### 15.5.1 函数生成不是仓库修复

函数级代码生成主要测输入到局部代码的映射。真实软件工程还要求：

- 理解 issue 和仓库约定；
- 搜索正确文件；
- 追踪调用方和数据结构；
- 修改多个文件；
- 运行公开和隐藏测试；
- 处理依赖、版本和资源；
- 解释失败并进行第二轮修复；
- 不改测试、不泄露凭证、不执行越权命令。

因此，HumanEval 类函数任务可以作为局部能力参考，却不能代表仓库级工程任务。

### 15.5.2 代码修复任务的环境

一个安全的 coding Agent harness 至少提供：

1. 固定的仓库 commit；
2. 可重置的工作目录；
3. 依赖和编译缓存；
4. 公开测试与隐藏测试分离；
5. 文件、网络和进程权限；
6. 超时、磁盘和内存上限；
7. patch 和命令 trace；
8. 失败后的环境快照。

测试环境若允许网络，模型可能下载答案或依赖外部服务；若共享工作目录，前一个
任务的修改可能泄露给后一个任务。reset 不是实现细节，而是评测有效性的组成部分。

### 15.5.3 分层代码指标

代码任务可以分别报告：

- apply rate：补丁能否应用；
- compile rate：是否能编译；
- public pass：公开测试通过；
- hidden pass：隐藏测试通过；
- regression rate：原有测试是否退化；
- scope correctness：是否只修改允许文件；
- security violation：是否访问禁止路径或执行危险命令；
- maintainability：复杂度、风格和依赖变化。

设 n 个任务中通过隐藏测试的数量为 h，单纯隐藏测试通过率为：

~~~math
P_{\mathrm{hidden}}=\frac{h}{n},\qquad n>0
~~~

但一个任务只要发生高严重度权限违规，就不应与普通测试失败拥有相同语义。必须
同时发布危险动作数和失败轨迹。

这里的 `h/n` 只是在全部任务都带有同一类隐藏测试时的简单教学比例。如果只有一
部分任务带隐藏测试，分母应是带隐藏测试且成功完成评测的有效任务数；若任务有重
复解码、环境错误或判定器错误，还应按任务或轨迹预先约定是否进入分母，不能直接
把重试后的最好结果当作一次成功。有效任务数为零时应报告
`not_applicable`（代码中用 `None`），不能用 `max(1, n)` 制造一个可比较的分数。

### 15.5.4 从 patch 到工程结果

模型可能生成看起来合理的 patch，却遗漏边界条件；也可能通过修改测试获得绿色
结果。隐藏测试、差异范围检查和代码审查共同约束这种投机。

对仓库任务，最终结果还可以拆成：

~~~math
S_{\mathrm{repo}}
=S_{\mathrm{apply}}
\land S_{\mathrm{test}}
\land S_{\mathrm{scope}}
\land S_{\mathrm{security}}
\land S_{\mathrm{regression}}
~~~

这是分析框架，不应取代逐项报告。若 S_scope 失败，测试全通过也不能写成安全成功。

### 15.5.5 终端任务

Terminal Agent 要处理命令行、文件搜索、编译、测试、环境变量和错误输出。评估应
记录：

- 命令是否必要；
- 参数是否正确；
- 是否重复执行；
- 是否能读懂错误；
- 是否恢复到干净状态；
- 是否访问敏感路径；
- 是否在预算内结束。

一个最终 patch 正确的 Agent，如果曾经删除用户目录或泄露环境变量，仍然不是可
以直接使用的终端助手。

## 15.6 Browser 与 computer-use 任务

### 15.6.1 页面状态比 URL 更重要

浏览器任务的成功不应定义为“打开了某个 URL”。页面可能需要登录、选择日期、
填写表单、确认筛选、保存草稿或完成提交。真正成功的是目标业务状态满足契约。

例如“把订单状态改为待审核”需要验证：

1. 找到了正确订单；
2. 使用的账号有权限；
3. 状态变更成功；
4. 没有修改其他字段；
5. 页面或后端状态已持久化；
6. 失败时没有重复提交。

### 15.6.2 环境状态与隐藏变量

浏览器环境会包含 session、弹窗、网络延迟、A/B 页面、动态 DOM 和历史操作。任务
必须固定或记录：

- 初始账号和权限；
- 页面数据和时间；
- 浏览器版本、viewport 和语言；
- 网络和外部服务模拟；
- 弹窗、验证码和登录失败处理；
- 重置和回滚机制。

如果环境不可重置，重复实验之间会互相污染；如果任务依赖真实账号和外部写入，
评估本身会产生不可接受的风险。

### 15.6.3 动作合法性与业务结果

浏览器 Agent 的轨迹可以分为：

- perception：识别页面元素和状态；
- navigation：选择页面和标签；
- interaction：点击、输入、滚动；
- verification：检查目标状态；
- recovery：处理错误和回退。

最终成功率之外，还要报告错误点击、无效输入、重复提交、权限违规和人工接管。
一个 Agent 会到达正确页面，不代表它理解了业务状态。

## 15.7 Tool/API 与事务性任务

### 15.7.1 工具调用成功不等于任务成功

工具任务中要区分：

1. 工具选择是否正确；
2. 参数 schema 是否正确；
3. 参数值是否符合业务约束；
4. 权限是否满足；
5. 工具返回是否被正确解释；
6. 写操作是否幂等；
7. 最终状态是否达到用户目标。

一个天气查询即使参数错误也可能返回可解析 JSON；一个退款操作即使 HTTP 200，也
可能写入了错误账户。工具的返回协议不能替代业务状态验证。

### 15.7.2 幂等和重试

设外部写操作为 W，重试次数为 k。如果 W 不是幂等操作，重试可能造成重复发送、
重复扣款或重复创建。评测 harness 应提供幂等键、模拟服务或可回滚事务，并把重复
副作用单独计数。

可以把工具结果分成：

- transport_success：请求到达服务；
- schema_success：参数和返回可解析；
- authorization_success：权限正确；
- state_success：业务状态达到目标；
- idempotency_success：重试没有重复副作用。

只有最后几层完成，才能叫事务任务成功。

### 15.7.3 事务任务的轨迹契约

对每个写操作，任务卡应写明：

~~~text
read_only / reversible / irreversible
authorized_actor
allowed_resource_scope
idempotency_key
confirmation_required
rollback_procedure
audit_fields
~~~

评估工具应在仿真或最小权限环境中运行。真实付款、删除、发信和权限变更不应
作为普通公开 benchmark 的默认动作。

## 15.8 科学、医学与专业研究任务

### 15.8.1 专业任务需要专家可验证性

科学和医学任务不能只用通用 judge。它们通常需要：

- 领域检索和文献版本；
- 计算、模拟或代码执行；
- 单位、量纲和边界条件；
- 证据到主张的引用；
- 不确定性和适用范围；
- 专家复核。

在专业场景中，流畅的错误答案可能比明显拒答更危险。任务契约要定义模型何时应
停止、请求更多数据或转人工。

### 15.8.2 研究 Agent 的阶段

可以将研究工作拆成：

1. question formulation：把问题转成可检验子问题；
2. retrieval：搜集候选证据；
3. evidence triage：区分相关、过时和相互冲突资料；
4. computation：执行代码或计算；
5. synthesis：形成带引用的结论；
6. uncertainty：说明未知和替代解释；
7. review：交给专家或责任人。

每一阶段都有独立失败模式。检索到论文不代表结论正确，代码跑出数字不代表模型
正确解释了单位，引用存在不代表它支持因果关系。

### 15.8.3 医疗和高影响任务

医疗、法律和金融任务应把安全和责任边界写入成功判据。可以测：

- 事实和指南一致性；
- 禁忌条件是否被识别；
- 单位和剂量是否正确；
- 是否说明数据不足；
- 是否提出适当的转诊或人工复核；
- 是否泄露隐私；
- 是否把建议误写成确定诊断。

此类评测应使用合成或严格脱敏材料，并由领域专家审查。模型在一个知识题上答对，
不能替代高影响工作流的安全评估。

## 15.9 网络安全评测

网络安全任务必须在隔离环境中进行，且要把发现、解释、修复、误报和危险动作分开。
高分不能抵消越权。

### 15.9.1 安全任务的四类结果

1. detection：是否发现问题；
2. diagnosis：是否准确解释影响和证据；
3. remediation：修复是否有效且没有引入新问题；
4. containment：是否避免扩大权限、泄露或破坏。

如果模型找到漏洞但执行了未授权扫描，detection 的成功不能覆盖 containment 的
失败。报告应把危险动作和副作用单独列出。

### 15.9.2 隔离环境

安全评测环境应限制网络、文件、进程、凭证和持久化。任务结束后销毁状态，避免
一个样本的工具结果泄露给下一个样本。公开材料不应包含可直接复制的真实攻击
凭证或破坏性步骤；可以使用抽象漏洞、合成目标和安全标识。

### 15.9.3 安全判定器

安全判定不能只靠模型自评。可以结合：

- 文件和网络审计；
- 权限事件；
- 目标状态；
- 专家分析；
- 静态和动态检测；
- 人工红队复核。

判定器也会错，必须保存原始 trace 和审计证据。

## 15.10 长周期 Agent 与跨 session 任务

单轮任务完成，不等于 Agent 能管理一个持续数小时或数天的项目。长周期任务需要
处理：

- workspace 和 artifact；
- checkpoint 和恢复；
- 任务分解；
- 跨 session 记忆；
- 中间结果版本；
- 人工交接；
- 重复副作用；
- 外部环境变化。

### 15.10.1 中断恢复

在第 t 步中断后，恢复任务不能只把最后一句模型回答重新放回上下文。需要保存：

1. 已完成的子任务；
2. 当前文件和 artifact 版本；
3. 已执行的外部动作；
4. 待验证的假设；
5. 预算消耗；
6. 权限和人工批准状态。

恢复成功应检查是否重复写入、丢失上下文或基于过期 artifact 继续工作。

### 15.10.2 长周期指标

可以报告：

- end-to-end completion；
- checkpoint recovery；
- artifact consistency；
- duplicate side effect；
- human handoff；
- stale-context rate；
- budget overrun；
- time to recover。

最终完成率很低时，阶段指标能说明问题来自规划、记忆、工具还是环境变化。

## 15.11 多模态和企业工作流

企业工作流常同时包含 PDF、扫描件、表格、截图、邮件和网页。专项评测不能只测
文本回答，还要检查：

- OCR 字段和表格结构；
- 图片、页面和段落的 grounding；
- 图表数值、单位和趋势；
- 文档版本和权限；
- 引用页码、区域和时间；
- 多模态输入在长轨迹中的作用域。

一个报销 Agent 可能正确读取金额，却把发票所属员工识别错；一个合同 Agent 可能
找到条款，却引用了旧版本。任务卡应把输入模态、证据 ID、版本和权限写清楚。

## 15.12 harness 是评测的一部分

同一模型换 retry、context folding、检索器、工具权限、代码执行器或 verifier，结
果可能显著变化：

~~~math
Y=F(M,H,E,B,D)
~~~

其中 M 是模型，H 是 harness，E 是环境，B 是预算，D 是数据和任务分布。比较时要
固定这些变量，或者明确说明改变了哪一个。

### 15.12.1 harness 的常见组成

- prompt 和 system message；
- memory、context folding 和摘要；
- retriever、reranker 和 cache；
- 工具 schema、权限和重试；
- 代码执行器、浏览器和网络；
- verifier、judge 和人工审核；
- timeout、取消、恢复和降级；
- 环境 reset 和状态快照。

如果换 harness 后分数提升，至少要做两种报告：

1. 固定 harness 的模型比较；
2. 固定模型的 harness 消融。

这样才能知道提升来自模型还是系统。

### 15.12.2 预算公平

比较 Agent 时要统一或明确记录：

- 最大动作数；
- 最大输入和输出 token；
- 工具调用次数；
- 并行候选数；
- 重试次数；
- wall-clock timeout；
- 计算和金钱预算。

模型 A 用 20 次工具、模型 B 用 3 次工具，比较的是不同预算下的完整系统能力。
这可以是有意义的产品比较，但不能命名为裸模型公平比较。

## 15.13 轨迹级指标与失败分层

一个 Agent 可能最终失败，但失败前已经完成检索、正确修改和大部分验证。只报最
终成功率会丢掉这些信号。可以把轨迹拆为阶段指标：

~~~math
\mathbf{R}_{\mathrm{stage}}
=\left(
R_{\mathrm{plan}},
R_{\mathrm{tool}},
R_{\mathrm{artifact}},
R_{\mathrm{verify}}
\right)
~~~

这个向量表示各阶段指标并列报告，而不是把它们误乘成一个数字。若要表示“所有
阶段都必须通过”的严格结果，可以另写：

~~~math
S_{\mathrm{all}}
=S_{\mathrm{plan}}
\land S_{\mathrm{tool}}
\land S_{\mathrm{artifact}}
\land S_{\mathrm{verify}}
~~~

实际报告应给出每一阶段的成功率、平均动作数、重复动作、恢复次数和错误类型。
计划错、工具超时、artifact 错和验证误判的修复责任不同，不能归到一个模型失败。

### 15.13.1 第一个不可逆错误

一次失败要保留“第一个不可逆错误”，而不是只保留最终错误。例如 Agent 先正确
读取需求，随后工具参数漏了租户字段，最后 verifier 报业务状态不对。修复重点
是 schema 和权限校验，不一定是换模型。

相反，如果工具返回正确、artifact 也正确，但模型没有把证据带入最终结论，才应
优先检查模型、上下文组织或 verifier。

每条轨迹可以带阶段标签：

~~~text
plan
tool-selection
argument
environment
artifact
verify
safety
~~~

一条任务可以同时拥有最终失败和局部成功。阶段标签支持按组件计算回归率，也能
避免把所有责任归给模型。

## 15.14 成功、部分成功和危险失败

### 15.14.1 结果状态

完整成功要求最终验证通过且没有安全违规。部分成功可能生成正确子产物但未完成；
可恢复失败来自暂态工具问题；不可恢复失败来自计划或推理；危险失败产生越权或
破坏性副作用。

建议保存以下状态：

~~~text
complete
partial
recoverable_failure
irrecoverable_failure
dangerous_failure
invalid_task
evaluator_error
~~~

这些状态不应被压成一个总标签。部分成功可以指导工具或规划改进，危险失败则需
要独立处理，即使它出现次数很少。

### 15.14.2 部分成功的价值

代码 Agent 可能正确定位 bug，却没有完成测试；研究 Agent 可能找到全部证据，却
没有写出可引用的结论；浏览器 Agent 可能填写表单，却在提交前正确请求人工确认。
如果报告只记最终失败，会丢掉可以优化的阶段能力。

部分成功指标必须有清晰定义，不能为了提高数字而把未完成任务算作成功。可以报告
子目标完成率、最后完成阶段和人工接管成本。

## 15.15 专项评测的抽样和污染

专业 benchmark 越流行，越可能被训练数据记忆。要做：

- exact 和 near duplicate；
- 仓库 commit、公开 issue 和搜索结果；
- 题目变体和实体替换；
- 时间切分；
- 环境 snapshot；
- prompt、teacher trajectory 和 verifier 泄漏检查。

### 15.15.1 任务选择偏差

专项评测还有一个容易被忽略的偏差：研究者往往更容易收集“能公开、能复现、能
自动判分”的任务，而最贵、最敏感、最依赖专家的任务反而被排除。于是一个评测
簇的高分可能只说明模型擅长可测部分，不能外推到全部工作流。

解决办法不是把所有任务混成一个平均，而是按切片报告样本数、成功率、成本和风险。
设第 k 个切片的任务数为 n_k，成功数为 s_k：

~~~math
\hat p_k=\frac{s_k}{\max(n_k,1)},
\qquad
\hat p_{\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}\hat p_k
~~~

macro 平均让小但关键的安全切片不被大批简单任务淹没。如果业务流量已知，再额外
报告按流量加权的 micro 结果。两种结果差异很大时，本身就是路由和采样需要讨论
的信号。

### 15.15.2 样本使用历史

公开任务、开发任务、回归任务和私有 holdout 的使用历史要分开记录。一个任务被
用于反复调 prompt 或训练后，就不再是未知泛化测试。对动态网页、真实仓库和外部
API，还要记录 snapshot 或日期，避免任务在评估期间自然变化。

## 15.16 统计稳定性与比较

专项评测通常样本少、任务难度高、失败相关。单次成功率容易受随机解码和个别样
本影响。报告应包含：

1. 样本数和有效分母；
2. 置信区间；
3. 重复运行和随机种子；
4. 任务难度、领域、工具和风险切片；
5. 失败轨迹和判定器分歧；
6. 成本和结果成熟窗口。

### 15.16.1 配对比较

如果两个模型在同一任务集合上运行，可以使用配对差异：

~~~math
\Delta=\hat p_A-\hat p_B
~~~

更细地记录旧版正确、新版错误的数量 b，以及旧版错误、新版正确的数量 c：

~~~math
\Delta_{\mathrm{paired}}=\frac{c-b}{n}
~~~

这比两个不相关任务集合的差异更有解释力。对工具和 Agent 任务，还要确保两个
版本从相同初始环境开始。

### 15.16.2 小样本不能制造精确感

假设某 coding Agent 在 40 个隐藏任务中成功 28 个，点估计为 0.70。粗略正态近
似标准误为：

~~~math
\mathrm{SE}
=\sqrt{\frac{\hat p(1-\hat p)}{n}}
=\sqrt{\frac{0.7\times0.3}{40}}
\approx0.072
~~~

几个任务的变化就可能改变结论。更稳妥的报告应使用二项分布区间或 bootstrap，并
按任务难度和风险切片展示结果。若 A 比 B 多成功两题，但差异只出现在同一个领域
切片，不能写成普遍模型优势。

## 15.17 代码修复评测簇案例

准备 100 个真实风格 issue，按 API bug、跨文件逻辑、依赖升级、性能回归和安全
修复分桶。每个任务有公开和隐藏测试，禁止修改测试，限制网络和文件系统权限。

### 15.17.1 任务卡

每个任务包含：

~~~text
task_id
repository_revision
issue_text
allowed_paths
forbidden_paths
public_tests
hidden_tests
network_policy
timeout
success_predicate
reset_procedure
~~~

### 15.17.2 报告指标

至少报告 apply、compile、public pass、hidden pass、regression、修改文件数、diff
范围、工具轮数、token、工具时间、越权命令、敏感文件访问和人工接管。

假设 100 个任务中：

- 92 个 patch 能应用；
- 70 个通过公开测试；
- 54 个通过隐藏测试；
- 4 个访问了禁止路径；
- 8 个测试后没有清理临时文件。

最终安全成功数不能从这几个汇总数字直接相减得到。至少有 4 个任务访问了禁止路
径，但它们是否包含在 54 个 hidden pass 中，需要查看逐任务交集；若教学假设这 4
个违规任务都属于 hidden pass，且其余条件均通过，安全成功最多是 50 个。无论如
何，不能把 70 个 public pass 写成 70% 任务成功。剩余任务还要按“测试失败、权限
违规、环境脏、超时和评估器错误”分层。

### 15.17.3 harness 消融

如果换一个更强的 harness 后 hidden pass 变成 62，但工具轮数从平均 8 增加到
23、P99 从 90 秒增到 240 秒，提升可能来自更强的检索和重试，而不是模型本身。
比较报告应同时给出：

1. 固定 harness 的模型比较；
2. 固定模型的 harness 比较；
3. 完整系统的用户成本；
4. 预算相同与预算放宽两种结果。

## 15.18 跨簇比较与宏观结论

两个 coding Agent 的总成功率相近，不代表它们适合相同工作。一个可能擅长简单
API 修复，另一个擅长跨文件重构；一个成本低，另一个需要人工接管少。跨簇比较
应保留能力向量：

~~~math
V(M)=
(Q_{\mathrm{software}},
Q_{\mathrm{browser}},
Q_{\mathrm{tool}},
Q_{\mathrm{research}},
Q_{\mathrm{safety}},
C_{\mathrm{success}})
~~~

向量可以帮助产品选择路由，但不应随意加权成总分。若必须做业务排序，先公开权重、
阈值和风险约束，再把总分当作特定决策下的效用，而不是模型的普遍智能分数。

## 15.19 自适应评测与预算分配

前沿模型评测很昂贵，不可能对所有模型在所有任务上无限增加预算。可以先用 smoke
set 检查协议和明显退化，再根据不确定性把预算分配给关键切片；但自适应过程必须
记录，否则不同模型获得的测试难度不同，分数不可直接比较。

设第 k 个切片的估计方差为 v_k、风险权重为 w_k，可以把采样优先级写成教学近似：

~~~math
\mathrm{priority}_k\propto w_k\sqrt{v_k}
~~~

高风险且不确定的切片优先增加样本；已经稳定且低风险的切片可以减少预算。最终
报告仍要列出每个切片的样本数、失败数、环境版本和置信区间。

这个优先级式不是严格的最优分配定理。它没有包含样本成本、切片相关性、停止规
则和最小可检测差异；实际使用时应把它当作分配候选，再由预先声明的公共核心集、
加样规则和预算上限约束。

### 15.19.1 自适应策略的公平边界

如果模型 A 触发了更多困难任务，而模型 B 只通过了简单 smoke set，直接比较总体
成功率是不公平的。可以使用：

- 预先固定的公共核心集；
- 追加集单独报告；
- 同任务配对比较；
- 明确的停止和加样规则；
- 记录每个模型实际获得的预算。

自适应评测可以提高资源效率，但不能成为挑选有利样本的隐蔽方法。

## 15.20 评测结果与发布动作

专项评测结果最终要映射到动作，而不是只生成“领先/落后”标签。可以按以下方式
组织发布建议：

| 结果信号 | 后续动作 |
| --- | --- |
| 低风险任务质量稳定、成本满足 | 小范围试用并持续抽样 |
| 关键任务有提升但高风险样本不确定 | 保留人工审核和受限工具 |
| 工具成功高但副作用或权限有问题 | 先修 schema、策略和回滚 |
| 质量可接受但 P99/成本超预算 | 改路由、缓存、预算或资源池 |
| 隐藏测试和线上回归下降 | 保留旧版，调查失败轨迹 |
| 判定器或环境不稳定 | 先修 harness，再比较模型 |

发布建议应绑定失败严重性、修复 owner、回归样本和回退版本。专项分数领先本身
不是上线许可。

### 15.20.1 signals/actions/decision 示例

报告可以拆成三列：

| signals | actions | decision |
| --- | --- | --- |
| hidden pass 提升、权限违规为 0、P99 在预算内 | 继续低风险灰度 | expand_low_risk_traffic |
| hidden pass 提升、存在高严重度副作用 | 收紧工具权限并人工审查 | hold_for_high_severity_review |
| 质量稳定、成本超预算 | 优化路由和重试 | route_to_bounded_workloads |
| grader 版本不一致 | 固定判定器并重跑 | remeasure_before_comparison |

这里的 decision 是可解释的后续动作，不是把复杂评估隐藏为一个不可诊断的总布尔
值。

## 15.21 从失败轨迹到责任归因

第 15.13 节已经说明如何标记第一个不可逆错误。本节进一步把这个标签变成可执行
的责任记录：每条失败轨迹至少应保存阶段、首个不可逆错误、证据、修复 owner、回
归样本和下一次对照实验。这样，评测结果才能进入工程修复，而不是停留在“模型失
败”的笼统标签。

可以沿用以下阶段标签：

~~~text
plan
tool-selection
argument
environment
artifact
verify
safety
~~~

如果 Agent 先正确读取需求，随后工具参数漏了租户字段，最后 verifier 报业务状
态不对，修复重点是 schema 和权限校验，不一定是换模型。相反，如果工具返回正确、
artifact 也正确，但模型没有把证据带入最终结论，才应优先检查模型、上下文组织
或 verifier。

一条任务可以同时拥有最终失败和局部成功。阶段标签支持按组件计算回归率，也能
避免把所有责任归给模型。复盘记录还应标明哪些结论来自自动判定、哪些来自专家
复核，因为判定器错误本身也可能是首个不可逆错误。

## 15.22 复现包的组成

公开 benchmark 页面可以支持任务定义、环境和公开指标；模型发布页可以支持特定
版本的自报结果；它们都不能单独支持“模型在所有专业工作上可靠”。

一个可复现的评测包至少应包含：

1. task manifest；
2. 环境镜像或 commit；
3. harness 配置；
4. 模型 ID 和 revision；
5. token、动作和金钱预算；
6. grader 版本；
7. 随机种子和并发；
8. 失败 trace；
9. 结果汇总和区间；
10. 敏感凭证和副作用处理说明。

对 SWE-bench、OSWorld、Terminal-Bench 等任务，正文应把它们作为评测设计和环境
验证的入口，而不是把不同 harness 下的分数直接横比。涉及安全或真实副作用时，
公开资料和自建结果还必须删去敏感凭证，并说明哪些结果只能在隔离环境中复现。

## 15.23 最小可运行的专项评测审计 Demo

下面的程序把一个小型 coding Agent 评测结果拆成质量、安全、成本和 trace 完整
性信号。数字是教学数据，不代表任何真实模型。

~~~python
tasks = [
    {"id": "api-01", "hidden": True, "hidden_pass": True, "scope_ok": True, "danger": False, "cost": 1.2, "trace": True},
    {"id": "repo-02", "hidden": True, "hidden_pass": True, "scope_ok": True, "danger": False, "cost": 2.4, "trace": True},
    {"id": "sec-03", "hidden": True, "hidden_pass": False, "scope_ok": False, "danger": True, "cost": 3.1, "trace": True},
    {"id": "perf-04", "hidden": False, "hidden_pass": False, "scope_ok": True, "danger": False, "cost": 1.8, "trace": False},
    {"id": "dep-05", "hidden": True, "hidden_pass": True, "scope_ok": True, "danger": False, "cost": 4.2, "trace": True},
]

signals = {
    "hidden_test_tasks": sum(task["hidden"] for task in tasks),
    "hidden_passes": sum(task["hidden"] and task["hidden_pass"] for task in tasks),
    "scope_violations": sum(not task["scope_ok"] for task in tasks),
    "dangerous_failures": sum(task["danger"] for task in tasks),
    "trace_coverage": sum(task["trace"] for task in tasks) / len(tasks) if tasks else None,
    "average_cost": sum(task["cost"] for task in tasks) / len(tasks) if tasks else None,
}

def ratio(num, den):
    return num / den if den > 0 else None


signals["hidden_pass_rate"] = ratio(
    signals["hidden_passes"], signals["hidden_test_tasks"]
)

actions = []
if signals["dangerous_failures"]:
    actions.append("关闭高风险工具并复核 sec-03 轨迹")
if signals["scope_violations"]:
    actions.append("检查允许路径和 patch 范围判定")
if signals["trace_coverage"] is not None and signals["trace_coverage"] < 1.0:
    actions.append("补齐 perf-04 的环境和工具 trace")
if signals["average_cost"] is not None and signals["average_cost"] > 3.0:
    actions.append("评估动作预算、重试和低成本路由")

if signals["dangerous_failures"]:
    decision = "hold_for_high_severity_review"
elif signals["trace_coverage"] is None or signals["trace_coverage"] < 1.0:
    decision = "remeasure_before_comparison"
else:
    decision = "continue_low_risk_trial"

assert ratio(1, 0) is None
assert ratio(0, 0) is None
assert signals["hidden_pass_rate"] == 0.75

print("signals=", signals, sep="")
print("actions=", actions, sep="")
print("decision=", decision, sep="")
~~~

运行输出为：

~~~text
signals={'hidden_test_tasks': 4, 'hidden_passes': 3, 'scope_violations': 1, 'dangerous_failures': 1, 'trace_coverage': 0.8, 'average_cost': 2.54, 'hidden_pass_rate': 0.75}
actions=['关闭高风险工具并复核 sec-03 轨迹', '检查允许路径和 patch 范围判定', '补齐 perf-04 的环境和工具 trace']
decision=hold_for_high_severity_review
~~~

这个 demo 有意让普通任务的成功不能抵消危险失败。`hidden_test_tasks` 表示样本
中有多少任务带隐藏测试，`hidden_passes` 才表示通过隐藏测试的任务数，二者相除
得到 `hidden_pass_rate`。真实系统还要记录每个任务的 pass/fail、判定器版本和原始
trace。这个程序展示的是审计数据形状，不能替代真实环境、程序判定和人工复核。

## 15.24 专项评测报告的最小模板

一个可审计的专项评测报告至少包含：

~~~text
任务和真实工作流:
模型/系统边界:
环境与 harness:
输入、输出和动作预算:
工具、权限和网络:
成功、部分成功和危险失败定义:
自动/人工 grader:
样本切片和使用历史:
失败轨迹和责任归因:
延迟、成本和人工接管:
模型、环境、grader 和日期版本:
污染和外部有效性风险:
发布范围、回退路径和后续动作:
~~~

发布建议不能只有“领先/落后”。可以分成：适合低风险试用、需要人工审核、只开放
某些工具、需要增加数据或 verifier、暂不发布。每个建议都应能回指到样本和轨迹。

## 15.25 练习

1. 为一个 coding Agent 设计 20 个任务，覆盖 API bug、跨文件逻辑、性能和安全。
   写出环境 reset、隐藏测试、允许路径和危险动作。
2. 为浏览器 Agent 设计一个“修改订单状态”的任务，分别定义页面状态、权限、提交、
   重试和回滚。
3. 设计一个科学研究 Agent 评测簇，区分检索、计算、引用、专家复核和不确定性。
4. 两个模型在 100 个任务上分别成功 62 和 68 次，但后者有 8 次权限违规。如何
   报告结果，下一步做什么？
5. 一个 harness 将工具重试从 1 次增加到 5 次，成功率上升但 P99 和成本上升。
   设计一个消融实验区分模型收益和 harness 收益。
6. 为长周期 Agent 设计中断恢复测试，要求覆盖 artifact 版本、重复副作用和预算。

### 15.25.1 练习检查点

每道题都应回答：

- 输入和环境是否可重置；
- 允许的动作和权限是什么；
- 成功判据是否可执行；
- 部分成功和危险失败如何记录；
- 成本和延迟的分母是什么；
- 失败 trace 能否导出修复动作；
- 结果能否在未公开任务上验证。

## 15.26 资料与证据边界

以下资料提供公开任务、环境和方法入口。它们支持任务设计或作者报告的实验，不能
自动证明某个新模型在其他 harness、版本和生产流量下可靠。

### 15.26.1 软件工程与终端

- SWE-bench；
  https://arxiv.org/abs/2310.06770
- SWE-Gym；
  https://arxiv.org/abs/2412.21139
- InterCode；
  https://arxiv.org/abs/2306.14898
- Terminal-Bench；
  https://www.tbench.ai/

这些入口适合研究仓库修复、交互式终端和可执行环境。不同版本的任务、测试、容
器和 Agent harness 可能改变分数，复现时必须保存 revision。

### 15.26.2 浏览器、computer-use 与工具

- WebArena；
  https://arxiv.org/abs/2307.13854
- BrowserGym；
  https://arxiv.org/abs/2305.17144
- OSWorld；
  https://arxiv.org/abs/2404.07972
- AgentBench；
  https://arxiv.org/abs/2308.03688
- τ-bench；
  https://arxiv.org/abs/2406.12045
- GAIA；
  https://arxiv.org/abs/2311.12983

这些项目把模型放入网页、桌面、工具或综合任务环境中，适合观察轨迹、状态和工
具边界。它们不能让一个浏览器 benchmark 分数自动代表企业生产安全。

### 15.26.3 安全和专业任务

- CyberSecEval；
  https://arxiv.org/abs/2404.13161
- AgentDojo；
  https://arxiv.org/abs/2406.13352
- NIST AI Risk Management Framework；
  https://www.nist.gov/itl/ai-risk-management-framework

安全资料应在隔离环境中使用，不把真实攻击凭证或破坏性载荷复制到公开练习。专业
领域任务还需要专家 rubric、可验证程序和责任边界。

## 15.27 本章小结

Specialized frontier eval cluster 的核心，不是列出更多 benchmark，而是把模型放
回具体工作流：

1. 用任务契约定义输入、环境、动作、成功、预算和风险；
2. 用可重置环境和完整 trace 测量轨迹，而不是只看最终文本；
3. 将代码、终端、浏览器、工具事务、科学专业、安全、长周期和多模态工作分别建
   立任务簇；
4. 把 outcome、process、safety、cost、延迟和人工接管分开报告；
5. 固定或显式记录模型、harness、工具、预算、grader 和环境；
6. 用隐藏测试、时间切分、污染检查和配对统计保护结论；
7. 让失败轨迹导出修复 owner、回归样本和后续实验；
8. 把结果转成受作用域限制的发布范围和回退路径。

一个总榜可以告诉你模型在某个公开任务集合中排在什么位置，专项评测簇才能告诉
你它在什么工作里值得信任、哪里需要人工、哪里会产生副作用，以及这份信任要付出
多少成本。
