# 第 29 章 A2A 与 MCP 的分工：Agent-to-Agent vs Agent-to-Tool

前面我们分别讲了 MCP 和 A2A。

MCP 解决的是 Agent 或 Host 如何连接外部工具、资源和提示模板。A2A 解决的是 Agent 与 Agent 之间如何发现、委派、同步状态、传递上下文和返回结果。

很多同学学到这里会产生一个问题：

> 既然 MCP 可以暴露工具，A2A 也可以让 Agent 互相调用，那到底什么时候用 MCP，什么时候用 A2A？

这个问题非常关键。如果分工不清，系统会变得很混乱：有的团队把所有工具都包装成 Agent，有的团队把复杂 Agent 当成一个普通工具，有的团队让 Agent 之间直接共享数据库连接，有的团队用 A2A 做文件读取、用 MCP 做任务委派，最后安全、审计、状态和权限都乱在一起。

本章专门讲清楚 A2A 与 MCP 的边界。

你可以先记住一句话：

> MCP 连接能力，A2A 连接协作者；MCP 面向工具和资源调用，A2A 面向任务委派和协作生命周期。

## 29.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求，先核对 A2A 官方 Protocol Specification、A2A and MCP 主题说明，以及 MCP 官方 Specification 和 Concepts 文档，再回到本项目已有章节做通用工程抽象。

资料边界要说清楚：

1. A2A 官方资料强调的是 Agent 间协作：Agent Card、Task、Message、Part、Artifact、TaskStatus、streaming / push update、身份和安全扩展等。
2. MCP 官方资料强调的是模型应用连接外部上下文和能力：Host、Client、Server、tools、resources、prompts、lifecycle、transport、authorization、roots 和安全最佳实践等。
3. A2A 官方主题说明也明确把 MCP 视为工具和资源侧协议，把 A2A 视为 Agent 协作侧协议。二者不是替代关系，而是经常组合使用。
4. 本章不实现真实 A2A server、MCP server、OAuth、SSE、Webhook、Registry 或远程调用，也不把 toy demo 的字段和阈值写成协议标准。
5. 本章只抽象稳定的系统设计边界：什么能力应该作为 MCP tool / resource / prompt 暴露，什么能力应该作为 A2A remote agent 委派，二者组合时如何治理权限、上下文、产物、trace、版本和 eval。

所以读本章时不要把结论理解成“哪个协议更高级”。更准确的理解是：MCP 负责把外部能力接入 Host，A2A 负责把远程 Agent 纳入协作生命周期；它们的边界越清楚，系统越容易做最小权限、上下文控制和审计。

## 29.1 先用一句话区分

最简单的区分是：

| 协议 | 主要关系 | 被调用方特征 | 核心动作 |
| --- | --- | --- | --- |
| MCP | Agent-to-Tool / Agent-to-Resource | 被动能力提供方 | 调工具、读资源、取提示模板 |
| A2A | Agent-to-Agent | 具备自主任务执行能力的协作者 | 委派任务、同步状态、返回产物 |

如果被调用方只是执行一个明确操作，比如读取文件、查询数据库、搜索文档、运行测试、打开网页，那更像 MCP。

如果被调用方会理解目标、规划步骤、追问信息、调用自己的工具、维护任务状态、返回结构化产物，那更像 A2A。

## 29.2 MCP 的核心抽象

MCP 的核心抽象是：

1. Tools：可调用操作。
2. Resources：可读取或引用的上下文资源。
3. Prompts：可复用提示模板或工作流入口。
4. Server：暴露这些能力的服务。
5. Client/Host：连接 Server，并决定如何把能力交给模型。

MCP 更像“外部能力接口层”。

例如：

1. `read_file(path)`。
2. `search_docs(query)`。
3. `query_orders(start_date, end_date)`。
4. `run_tests(command)`。
5. `get_current_page()`。
6. `apply_patch(diff)`。

这些能力可能非常强，但它们本身通常不会长期规划，也不会主动委派子任务。它们接收输入，执行操作，返回结果。

## 29.3 A2A 的核心抽象

A2A 的核心抽象是：

1. Agent Card：能力声明和服务发现。
2. Task：任务委派单元。
3. Message：Agent 间消息。
4. Status：任务状态。
5. Artifact：任务产物。
6. Context Policy：上下文边界。
7. Trace：跨 Agent 审计链路。

A2A 更像“协作任务协议层”。

例如，总控 Agent 委派给数据分析 Agent：

```text
请分析过去 30 天退款率升高的原因，使用聚合数据，输出业务报告和证据链。
```

数据分析 Agent 可能会：

1. 检查输入是否完整。
2. 追问指标定义。
3. 调用自己的 MCP 数据库 Server 查询数据。
4. 调用知识库 Server 获取业务定义。
5. 生成中间表格。
6. 输出根因假设。
7. 返回报告 Artifact。

这已经不是一次工具调用，而是一个任务生命周期。

## 29.4 判断标准一：被调用方是否有自主性

这是最重要的判断标准。

### 29.4.1 低自主性：更适合 MCP

如果被调用方只做明确操作，通常适合 MCP。

例如：

1. 读取某个文件。
2. 搜索某个关键词。
3. 查询一张表。
4. 运行一个测试命令。
5. 获取网页正文。
6. 调用一个业务接口。

这些能力不需要自己规划任务，也不需要追问用户。它们只需要输入输出 schema、权限、超时和错误处理。

### 29.4.2 高自主性：更适合 A2A

如果被调用方会独立完成一个目标，通常适合 A2A。

例如：

1. 评审一份合同。
2. 分析一个业务指标异常。
3. 修复一个测试失败。
4. 生成一份上线风险报告。
5. 做一次安全审计。
6. 规划一次营销活动。

这些任务不是一次函数调用能表达的。它们需要目标理解、上下文选择、状态同步、产物返回和可能的多轮协商。

## 29.5 判断标准二：是否需要任务生命周期

如果一次调用就能完成，MCP 往往足够。

例如：

```text
查询 2026-05-01 到 2026-05-29 的订单总量。
```

这可以是 MCP Tool。

如果任务需要长时间执行、阶段性进度、追问、取消、重试和部分结果，就更适合 A2A。

例如：

```text
分析过去一个月订单下降的原因，并生成给高管看的报告。
```

这个任务可能需要多个步骤和多个产物。它需要 Task 状态机，而不是一个简单 tool result。

## 29.6 判断标准三：输出是结果值还是产物

MCP Tool 常返回一个结果值或操作结果：

```json
{
  "rows": [
    {"date": "2026-05-01", "orders": 1024}
  ]
}
```

A2A Task 常返回一个或多个 Artifact：

```json
{
  "summary": "订单下降主要来自新用户渠道。",
  "artifacts": [
    "artifact://task_123/executive-report.md",
    "artifact://task_123/channel-analysis.csv"
  ],
  "evidence": [
    "trace://task_123/tool-call-7"
  ]
}
```

如果你需要的是“一个值”，MCP 更自然。如果你需要的是“一个任务成果”，A2A 更自然。

## 29.7 判断标准四：是否需要能力发现和路由

MCP 也有能力发现，但发现的是工具、资源和 prompt。

A2A 发现的是 Agent。

两者粒度不同。

例如，MCP Server 可以声明：

```text
我有 search_docs、read_doc、list_collections 这些工具。
```

A2A Agent Card 可以声明：

```text
我是合规审查 Agent，擅长支付、隐私和金融合规，可以生成审查报告，需要合同或需求文档作为输入。
```

如果路由目标是“选择哪个操作”，偏 MCP。如果路由目标是“选择哪个专家 Agent”，偏 A2A。

## 29.8 判断标准五：权限边界在哪

MCP 权限通常围绕工具和资源：

1. 能不能读这个文件？
2. 能不能查这个表？
3. 能不能执行这个命令？
4. 能不能访问这个网页？
5. 能不能应用这个补丁？

A2A 权限通常围绕任务和协作：

1. 能不能把任务委派给这个 Agent？
2. 能不能把这段上下文传给下游 Agent？
3. 下游 Agent 能不能继续委派？
4. 下游 Agent 能不能看到用户身份？
5. 下游 Agent 返回的产物能不能给其他 Agent 使用？

这也是二者分工不同的关键原因。MCP 更关注“操作权限”，A2A 更关注“协作权限”。

## 29.9 一个典型组合架构

真实系统里，A2A 和 MCP 经常组合使用。

例如：

```text
用户
  -> 总控 Agent
    -> A2A -> 数据分析 Agent
      -> MCP -> 数据库 Server
      -> MCP -> 知识库 Server
    -> A2A -> 代码修复 Agent
      -> MCP -> IDE Server
      -> MCP -> Terminal Server
      -> MCP -> Git Server
    -> A2A -> 报告写作 Agent
      -> MCP -> 文档模板 Server
```

这里的分工是：

1. 总控 Agent 用 A2A 把任务交给不同专家 Agent。
2. 专家 Agent 用 MCP 访问自己需要的工具和资源。
3. 每个 Agent 内部有自己的 Host/Runtime，控制 MCP 权限。
4. Agent 间只通过 A2A 传递任务、消息、状态和产物。
5. 不同 Agent 不直接共享对方的工具权限。

这个架构比“一个大 Agent 直连所有工具”更可治理，也比“所有工具都包装成 Agent”更清晰。

## 29.10 反例一：把工具伪装成 Agent

有些团队会把每个工具都包装成 Agent：

1. 文件读取 Agent。
2. 数据库查询 Agent。
3. 搜索 Agent。
4. 测试执行 Agent。
5. 浏览器点击 Agent。

这通常不是好设计。

如果这些“Agent”只是被动执行一个操作，没有自主规划、任务状态、追问和产物管理，那它们就是工具。把工具伪装成 Agent 会带来额外复杂度：

1. 多一层任务协议。
2. 多一层状态管理。
3. 路由更复杂。
4. 延迟更高。
5. 审计语义混乱。
6. 权限边界不清。

工程上应该保持概念诚实：工具就是工具，Agent 就是 Agent。

## 29.11 反例二：把 Agent 降级成工具

另一个反例是，把复杂 Agent 包装成一个 MCP Tool：

```json
{
  "name": "do_everything",
  "description": "Complete the whole business analysis task."
}
```

这也有问题。

如果被调用方需要长任务、追问、状态更新、多个产物和复杂失败语义，把它塞进一个 Tool 会导致：

1. 状态不可见。
2. 中途无法追问。
3. 难以取消。
4. 难以返回多个产物。
5. 错误语义不清。
6. 难以审计内部过程。

这种能力更适合作为 A2A Agent。

## 29.12 反例三：跨 Agent 共享 MCP 权限

还有一种危险设计：总控 Agent 拥有很多 MCP 工具权限，然后把这些权限直接传给下游 Agent。

例如：

```text
总控 Agent 有生产数据库权限。
它委派给报告 Agent，并把数据库查询能力也交给报告 Agent。
```

这会造成权限扩散。

更合理的方式是：

1. 报告 Agent 不直接访问生产数据库。
2. 数据 Agent 负责查询和聚合数据。
3. 报告 Agent 只接收聚合结果和证据引用。
4. 如果报告 Agent 需要更多数据，必须通过 A2A 请求上游或数据 Agent。
5. 所有跨 Agent 数据流受 context_policy 控制。

委派任务不等于转交工具权限。

## 29.13 什么时候只用 MCP 就够了

以下场景通常只用 MCP 就够：

1. 单 Agent 使用外部工具。
2. 能力都是明确工具操作。
3. 不需要多个 Agent 协作。
4. 任务生命周期短。
5. 不需要异步状态同步。
6. 不需要 Agent 能力发现。
7. 不需要复杂上下文转发。

例如，一个 IDE Coding Assistant 只需要读文件、搜索代码、应用补丁、运行测试，那么 MCP 就能覆盖大部分需求。

## 29.14 什么时候需要 A2A

以下场景更适合引入 A2A：

1. 多个 Agent 由不同团队维护。
2. 任务需要专业分工。
3. 任务需要异步执行。
4. 下游 Agent 可能追问。
5. 需要结构化状态和产物。
6. 需要 Agent 能力发现和服务发现。
7. 需要跨 Agent 权限和上下文治理。
8. 需要审计任务经过哪些 Agent。

例如，一个企业自动化系统需要销售 Agent、合同 Agent、法务 Agent、财务 Agent 和交付 Agent 协作，这就明显适合 A2A。

## 29.15 什么时候二者组合使用

最常见的生产形态是组合使用。

组合使用适合：

1. 每个 Agent 都需要自己的工具和资源。
2. 不同 Agent 有不同权限边界。
3. 总控 Agent 负责任务分解。
4. 专家 Agent 负责领域任务执行。
5. 工具访问需要独立审计。
6. Agent 之间只传递必要结果和产物。

一个简单原则是：

> Agent 内部用 MCP，Agent 之间用 A2A。

这不是绝对规则，但在大多数系统设计题里是一个很好的默认答案。

## 29.16 系统设计中的分层架构

一个清晰的分层可以是：

```text
用户界面层
  -> Orchestrator Agent
    -> A2A 协作层
      -> Specialist Agents
        -> MCP 工具资源层
          -> 文件、数据库、知识库、浏览器、终端、业务 API
```

每层职责不同：

1. 用户界面层：收集用户目标、展示状态、请求确认。
2. Orchestrator Agent：拆解任务、选择 Agent、汇总结果。
3. A2A 协作层：任务委派、消息、状态、Artifact、审计。
4. Specialist Agents：领域推理和任务执行。
5. MCP 工具资源层：具体工具、资源和外部系统连接。
6. 底层系统：文件、数据库、网页、业务服务等。

这套分层能让权限和责任边界更清晰。

## 29.17 Trace 如何串起来

组合使用 A2A 和 MCP 时，trace 非常重要。

一个完整 trace 可能长这样：

```text
trace_root_001
  A2A task: orchestrator -> data_agent
    MCP tool call: data_agent -> query_orders
    MCP resource read: data_agent -> refund_definition_doc
    artifact: refund_analysis.csv
  A2A task: orchestrator -> report_agent
    artifact read: refund_analysis.csv
    artifact: final_report.md
```

你需要能回答：

1. 用户原始任务是什么？
2. 总控 Agent 委派给了哪些 Agent？
3. 每个 Agent 调用了哪些 MCP 工具？
4. 哪些资源进入了上下文？
5. 产生了哪些 Artifact？
6. 最终答案引用了哪些证据？
7. 哪一步出现了错误或幻觉？

没有统一 trace，A2A + MCP 组合系统很难排障。

## 29.18 安全边界如何串起来

A2A 和 MCP 的安全边界也要分层。

MCP 层负责：

1. 工具权限。
2. 资源权限。
3. 本地沙箱。
4. 网络限制。
5. 命令执行限制。
6. 工具结果脱敏。

A2A 层负责：

1. Agent 调用权限。
2. 任务委派权限。
3. 上下文转发策略。
4. Agent 身份和信任。
5. Artifact 共享策略。
6. 跨 Agent 审计。

Host/Runtime 层负责把二者统一起来：

1. 当前用户是谁。
2. 当前 Agent 是谁。
3. 当前任务允许什么。
4. 当前数据能不能传给下游。
5. 当前工具结果能不能进入上下文。
6. 当前产物能不能被其他 Agent 读取。

如果只管 MCP 权限，不管 A2A 上下文转发，数据会在 Agent 间泄露。如果只管 A2A 委派，不管 MCP 工具权限，Agent 内部会越权操作。

## 29.19 面试中如何回答这类系统设计题

如果面试官问：

```text
设计一个多 Agent 企业助手，既能查知识库、跑数据分析、改代码，又能让多个 Agent 协作，你如何划分 MCP 和 A2A？
```

一个清晰回答可以是：

1. 用 A2A 做 Agent 间协作：总控 Agent 通过 Agent Card 发现数据 Agent、代码 Agent、报告 Agent，使用 Task/Status/Artifact 管理任务生命周期。
2. 用 MCP 做每个 Agent 内部的工具和资源连接：数据 Agent 通过 MCP 查数据库和知识库，代码 Agent 通过 MCP 连接 IDE、Git 和终端，报告 Agent 通过 MCP 读取模板和文档资源。
3. Agent 间不直接共享工具权限，只共享最小必要上下文、结构化结果和 Artifact 引用。
4. Trace 横跨 A2A task 和 MCP tool call，支持审计和 replay。
5. 权限分两层：MCP 控制工具资源访问，A2A 控制任务委派和上下文转发。
6. 对高风险动作使用用户确认、策略引擎和人工接管。

这类回答会比简单说“用 MCP 调工具，用 A2A 调 Agent”更有工程深度。

## 29.20 常见误区

### 29.20.1 误区一：A2A 可以替代 MCP

A2A 不是工具协议。它可以委派给一个 Agent，但不适合描述文件读取、数据库查询、浏览器点击、终端执行等具体工具能力。

### 29.20.2 误区二：MCP 可以替代 A2A

MCP 可以暴露强工具，但不适合表达多 Agent 任务生命周期、追问、状态同步、能力发现和产物协作。

### 29.20.3 误区三：所有能力都包装成 Agent

如果一个能力只是被动操作，把它包装成 Agent 会增加不必要复杂度。

### 29.20.4 误区四：所有 Agent 都共享同一批 MCP 工具

这会导致权限扩散。不同 Agent 应该有各自的工具权限和沙箱。

### 29.20.5 误区五：只做调用，不做 trace

A2A + MCP 组合系统没有 trace，就无法解释最终结果来自哪里，也无法审计工具调用和上下文流动。

## 29.21 A2A/MCP 分工审计指标与最小 demo

为了避免系统设计停留在口号上，可以把 A2A 和 MCP 的分工做成一组可审计指标。它不要求你在真实项目里照搬下面的 toy 字段，而是帮助你形成工程判断：每一个外部能力到底是工具、资源、prompt、远程 Agent，还是“Agent 内部用 MCP、Agent 之间用 A2A”的组合形态。

设第 $i$ 个能力接入样本为：

```math
b_i=(p_i,w_i,a_i,l_i,d_i,c_i,r_i,o_i,t_i,v_i,z_i)
```

其中 $p_i$ 是期望协议分类，$w_i$ 是 workload 类型，$a_i$ 是被调用方自主性，$l_i$ 是生命周期承载方式，$d_i$ 是能力发现方式，$c_i$ 是上下文所有权，$r_i$ 是权限分离情况，$o_i$ 是输出边界，$t_i$ 是 trace 链路，$v_i$ 是版本捕获，$z_i$ 是 eval 标签。

对任意检查项 $k$，统一覆盖率写成：

```math
C_k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{g_k(b_i)=1\}
```

本章建议至少看这些指标：

1. 协议分类准确率 $C_{\mathrm{proto}}$：工具、Agent 和组合能力是否被分到正确协议层。
2. 工具 / Agent 边界清晰率 $C_{\mathrm{boundary}}$：明确操作是否走 MCP，自主任务是否走 A2A。
3. 自主性匹配率 $C_{\mathrm{auto}}$：低自主性能力没有伪装成 Agent，高自主性能力没有被塞进单个 tool。
4. 生命周期放置率 $C_{\mathrm{life}}$：短操作走 tool call，长任务走 Task lifecycle，组合任务有 combined lifecycle。
5. 能力发现分离率 $C_{\mathrm{discover}}$：MCP 发现 tools / resources / prompts，A2A 发现 Agent Card / skills。
6. 上下文所有权清晰率 $C_{\mathrm{context}}$：MCP 结果进入 Host context builder，A2A 消息受 task context policy 控制。
7. 权限分离率 $C_{\mathrm{perm}}$：委派任务没有顺手转交工具权限，工具权限仍由各自 Host / Runtime 控制。
8. 结果 / 产物边界率 $C_{\mathrm{output}}$：MCP 返回 tool result / resource content，A2A 返回 artifact / structured task result。
9. Trace 串联率 $C_{\mathrm{trace}}$：A2A task span 和 MCP tool span 能在同一 trace 中串起来。
10. 版本与 eval 覆盖率 $C_{\mathrm{version}}$：记录 MCP spec / tool schema / A2A spec / Agent Card / policy / eval label。

上线门禁可以写成：

```math
G_{\mathrm{a2a\_mcp}}=
\mathbf{1}\{
C_{\mathrm{proto}}\ge \tau_{\mathrm{proto}},
C_{\mathrm{boundary}}\ge \tau_{\mathrm{boundary}},
C_{\mathrm{auto}}\ge \tau_{\mathrm{auto}},
C_{\mathrm{life}}\ge \tau_{\mathrm{life}},
C_{\mathrm{discover}}\ge \tau_{\mathrm{discover}},
C_{\mathrm{context}}\ge \tau_{\mathrm{context}},
C_{\mathrm{perm}}\ge \tau_{\mathrm{perm}},
C_{\mathrm{output}}\ge \tau_{\mathrm{output}},
C_{\mathrm{trace}}\ge \tau_{\mathrm{trace}},
C_{\mathrm{version}}\ge \tau_{\mathrm{version}}
\}
```

如果要汇总成一个分数，可以用加权平均：

```math
S_{\mathrm{a2a\_mcp}}=\frac{\sum_k \alpha_k C_k}{\sum_k \alpha_k}
```

注意：分数只是排序和回归对比用，不能替代硬门禁。比如权限分离不过线时，即使其他指标很高，也不能把系统描述成可治理。

下面是一个 0 依赖 demo。它模拟 12 个能力接入样本，故意放入把工具伪装成 Agent、把 Agent 塞成 Tool、共享工具权限、缺 Agent Card、缺 MCP registry、上下文泄露、结果和 Artifact 混淆、trace 缺失、版本和 eval 缺失等 bad case。

```python
from collections import OrderedDict


CASES = [
    {
        "id": "read_file_mcp_ok",
        "expected": "mcp",
        "actual": "mcp",
        "autonomy": "low",
        "workload": "operation",
        "lifecycle": "mcp_call",
        "discovery": {"tools_resources_prompts"},
        "context_owner": "host_controlled",
        "permission_separated": True,
        "output": "tool_result",
        "trace": {"mcp_call"},
        "versions": {"mcp_spec", "tool_schema", "permission_policy"},
        "eval_label": True,
    },
    {
        "id": "contract_review_a2a_ok",
        "expected": "a2a",
        "actual": "a2a",
        "autonomy": "high",
        "workload": "task",
        "lifecycle": "a2a_task",
        "discovery": {"agent_card"},
        "context_owner": "task_context_policy",
        "permission_separated": True,
        "output": "artifact",
        "trace": {"a2a_task"},
        "versions": {"a2a_spec", "agent_card", "permission_policy"},
        "eval_label": True,
    },
    {
        "id": "enterprise_analysis_both_ok",
        "expected": "both",
        "actual": "both",
        "autonomy": "high",
        "workload": "mixed",
        "lifecycle": "combined",
        "discovery": {"agent_card", "tools_resources_prompts"},
        "context_owner": "combined_policy",
        "permission_separated": True,
        "output": "artifact_and_tool_results",
        "trace": {"a2a_task", "mcp_call"},
        "versions": {"a2a_spec", "mcp_spec", "agent_card", "tool_schema", "permission_policy"},
        "eval_label": True,
    },
    {
        "id": "tool_as_agent_bad",
        "expected": "mcp",
        "actual": "a2a",
        "autonomy": "low",
        "workload": "operation",
        "lifecycle": "a2a_task",
        "discovery": {"agent_card"},
        "context_owner": "task_context_policy",
        "permission_separated": True,
        "output": "artifact",
        "trace": {"a2a_task"},
        "versions": {"a2a_spec", "agent_card"},
        "eval_label": True,
    },
    {
        "id": "agent_as_tool_bad",
        "expected": "a2a",
        "actual": "mcp",
        "autonomy": "high",
        "workload": "task",
        "lifecycle": "mcp_call",
        "discovery": {"tools_resources_prompts"},
        "context_owner": "host_controlled",
        "permission_separated": True,
        "output": "tool_result",
        "trace": {"mcp_call"},
        "versions": {"mcp_spec", "tool_schema"},
        "eval_label": True,
    },
    {
        "id": "shared_tool_permission_bad",
        "expected": "both",
        "actual": "both",
        "autonomy": "high",
        "workload": "mixed",
        "lifecycle": "combined",
        "discovery": {"agent_card", "tools_resources_prompts"},
        "context_owner": "combined_policy",
        "permission_separated": False,
        "output": "artifact_and_tool_results",
        "trace": {"a2a_task", "mcp_call"},
        "versions": {"a2a_spec", "mcp_spec", "agent_card", "tool_schema"},
        "eval_label": True,
    },
    {
        "id": "missing_agent_card_bad",
        "expected": "a2a",
        "actual": "a2a",
        "autonomy": "high",
        "workload": "task",
        "lifecycle": "a2a_task",
        "discovery": {"tools_resources_prompts"},
        "context_owner": "task_context_policy",
        "permission_separated": True,
        "output": "artifact",
        "trace": {"a2a_task"},
        "versions": {"a2a_spec"},
        "eval_label": True,
    },
    {
        "id": "missing_mcp_registry_bad",
        "expected": "mcp",
        "actual": "mcp",
        "autonomy": "low",
        "workload": "operation",
        "lifecycle": "mcp_call",
        "discovery": set(),
        "context_owner": "host_controlled",
        "permission_separated": True,
        "output": "tool_result",
        "trace": {"mcp_call"},
        "versions": {"mcp_spec"},
        "eval_label": True,
    },
    {
        "id": "context_leak_bad",
        "expected": "a2a",
        "actual": "a2a",
        "autonomy": "high",
        "workload": "task",
        "lifecycle": "a2a_task",
        "discovery": {"agent_card"},
        "context_owner": "shared_raw_prompt",
        "permission_separated": False,
        "output": "artifact",
        "trace": {"a2a_task"},
        "versions": {"a2a_spec", "agent_card"},
        "eval_label": True,
    },
    {
        "id": "result_artifact_confused_bad",
        "expected": "both",
        "actual": "both",
        "autonomy": "high",
        "workload": "mixed",
        "lifecycle": "combined",
        "discovery": {"agent_card", "tools_resources_prompts"},
        "context_owner": "combined_policy",
        "permission_separated": True,
        "output": "tool_result",
        "trace": {"a2a_task", "mcp_call"},
        "versions": {"a2a_spec", "mcp_spec", "agent_card", "tool_schema"},
        "eval_label": True,
    },
    {
        "id": "trace_missing_bad",
        "expected": "both",
        "actual": "both",
        "autonomy": "high",
        "workload": "mixed",
        "lifecycle": "combined",
        "discovery": {"agent_card", "tools_resources_prompts"},
        "context_owner": "combined_policy",
        "permission_separated": True,
        "output": "artifact_and_tool_results",
        "trace": {"a2a_task"},
        "versions": {"a2a_spec", "mcp_spec", "agent_card", "tool_schema"},
        "eval_label": True,
    },
    {
        "id": "version_eval_missing_bad",
        "expected": "both",
        "actual": "both",
        "autonomy": "high",
        "workload": "mixed",
        "lifecycle": "combined",
        "discovery": {"agent_card", "tools_resources_prompts"},
        "context_owner": "combined_policy",
        "permission_separated": True,
        "output": "artifact_and_tool_results",
        "trace": {"a2a_task", "mcp_call"},
        "versions": set(),
        "eval_label": False,
    },
]


def required_discovery(case):
    if case["expected"] == "mcp":
        return {"tools_resources_prompts"}
    if case["expected"] == "a2a":
        return {"agent_card"}
    return {"agent_card", "tools_resources_prompts"}


def required_trace(case):
    if case["expected"] == "mcp":
        return {"mcp_call"}
    if case["expected"] == "a2a":
        return {"a2a_task"}
    return {"a2a_task", "mcp_call"}


def required_versions(case):
    if case["expected"] == "mcp":
        return {"mcp_spec", "tool_schema"}
    if case["expected"] == "a2a":
        return {"a2a_spec", "agent_card"}
    return {"a2a_spec", "mcp_spec", "agent_card", "tool_schema"}


def check(case):
    expected_by_workload = {
        "operation": {"mcp"},
        "task": {"a2a"},
        "mixed": {"both"},
    }[case["workload"]]
    expected_lifecycle = {
        "operation": {"mcp_call", "combined"},
        "task": {"a2a_task", "combined"},
        "mixed": {"combined"},
    }[case["workload"]]
    expected_context = {
        "operation": {"host_controlled", "combined_policy"},
        "task": {"task_context_policy", "combined_policy"},
        "mixed": {"combined_policy"},
    }[case["workload"]]
    expected_output = {
        "operation": {"tool_result"},
        "task": {"artifact"},
        "mixed": {"artifact_and_tool_results"},
    }[case["workload"]]
    autonomy_ok = (
        (case["autonomy"] == "low" and case["actual"] in {"mcp", "both"})
        or (case["autonomy"] == "high" and case["actual"] in {"a2a", "both"})
    )
    return OrderedDict([
        ("protocol_classification", case["expected"] == case["actual"]),
        ("tool_agent_boundary", case["actual"] in expected_by_workload),
        ("autonomy_fit", autonomy_ok),
        ("lifecycle_placement", case["lifecycle"] in expected_lifecycle),
        ("discovery_split", required_discovery(case) <= case["discovery"]),
        ("context_ownership", case["context_owner"] in expected_context),
        ("permission_separation", case["permission_separated"]),
        ("result_artifact_boundary", case["output"] in expected_output),
        ("trace_linkage", required_trace(case) <= case["trace"]),
        ("version_eval_coverage", required_versions(case) <= case["versions"] and case["eval_label"]),
    ])


scores = [check(case) for case in CASES]
metrics = OrderedDict()
for key in scores[0]:
    metrics[key] = round(sum(score[key] for score in scores) / len(scores), 3)

thresholds = {
    "protocol_classification": 0.95,
    "tool_agent_boundary": 0.95,
    "autonomy_fit": 0.95,
    "lifecycle_placement": 0.95,
    "discovery_split": 0.95,
    "context_ownership": 0.95,
    "permission_separation": 0.95,
    "result_artifact_boundary": 0.95,
    "trace_linkage": 0.95,
    "version_eval_coverage": 0.95,
}

failed_cases = [
    case["id"]
    for case, score in zip(CASES, scores)
    if not all(score.values())
]
failed_gates = [
    name
    for name, value in metrics.items()
    if value < thresholds[name]
]

case_score = {case["id"]: score for case, score in zip(CASES, scores)}
smoke = OrderedDict([
    ("mcp_tool_kept_as_tool", all(case_score["read_file_mcp_ok"].values())),
    ("a2a_task_kept_as_agent", all(case_score["contract_review_a2a_ok"].values())),
    ("combined_architecture_detected", all(case_score["enterprise_analysis_both_ok"].values())),
    ("caught_permission_leak", not all(case_score["shared_tool_permission_bad"].values())),
])

print("smoke=", dict(smoke))
print("metrics=", dict(metrics))
print("failed_cases=", failed_cases)
print("failed_gates=", failed_gates)
print("a2a_mcp_boundary_gate_pass=", not failed_gates)
```

期望输出：

```text
smoke= {'mcp_tool_kept_as_tool': True, 'a2a_task_kept_as_agent': True, 'combined_architecture_detected': True, 'caught_permission_leak': True}
metrics= {'protocol_classification': 0.833, 'tool_agent_boundary': 0.833, 'autonomy_fit': 0.833, 'lifecycle_placement': 0.833, 'discovery_split': 0.667, 'context_ownership': 0.75, 'permission_separation': 0.833, 'result_artifact_boundary': 0.75, 'trace_linkage': 0.75, 'version_eval_coverage': 0.583}
failed_cases= ['tool_as_agent_bad', 'agent_as_tool_bad', 'shared_tool_permission_bad', 'missing_agent_card_bad', 'missing_mcp_registry_bad', 'context_leak_bad', 'result_artifact_confused_bad', 'trace_missing_bad', 'version_eval_missing_bad']
failed_gates= ['protocol_classification', 'tool_agent_boundary', 'autonomy_fit', 'lifecycle_placement', 'discovery_split', 'context_ownership', 'permission_separation', 'result_artifact_boundary', 'trace_linkage', 'version_eval_coverage']
a2a_mcp_boundary_gate_pass= False
```

这个 demo 的重点不是阈值，而是 bad case 的形状。只要你看到 `tool_as_agent_bad`、`agent_as_tool_bad`、`shared_tool_permission_bad` 这类失败，就应该意识到：协议选型错误通常会很快演变成权限扩散、状态不可见、上下文泄露和 trace 断链。

## 29.22 面试高频题

### 题 1：A2A 和 MCP 的核心区别是什么？

参考回答：

MCP 是 Agent/Host 连接工具、资源和提示模板的协议，核心是 Tool、Resource、Prompt。A2A 是 Agent 与 Agent 协作的协议，核心是 Agent Card、Task、Message、Status 和 Artifact。MCP 处理 Agent-to-Tool，A2A 处理 Agent-to-Agent。

### 题 2：什么时候用 MCP，什么时候用 A2A？

参考回答：

如果被调用方只是执行明确操作，如读文件、查数据库、跑测试、搜索文档，适合 MCP。如果被调用方需要理解目标、规划步骤、追问、维护状态、调用自己的工具并返回产物，适合 A2A。生产系统常见模式是 Agent 内部用 MCP，Agent 之间用 A2A。

### 题 3：为什么不要把所有工具包装成 Agent？

参考回答：

工具是被动操作，Agent 是自主任务执行者。把工具包装成 Agent 会引入不必要的任务状态、路由、延迟和审计复杂度，也会模糊权限边界。工具应该用 MCP，只有具备任务规划和协作生命周期的能力才适合 A2A。

### 题 4：为什么不要把复杂 Agent 包装成一个 MCP Tool？

参考回答：

复杂 Agent 可能需要长任务、状态更新、追问、取消、多个 Artifact 和复杂错误语义。用一个 MCP Tool 包起来会导致状态不可见、中途无法协商、产物难管理、失败语义不清和审计困难。更适合用 A2A 表达。

### 题 5：A2A + MCP 组合系统如何做安全？

参考回答：

MCP 层控制工具和资源权限，如文件、数据库、浏览器、终端。A2A 层控制 Agent 调用、任务委派、上下文转发和 Artifact 共享。Host/Runtime 统一用户身份、Agent 身份、任务策略和数据分类。Agent 间不直接共享工具权限，只传最小必要上下文和可审计的产物引用。

## 29.23 小练习

1. 判断以下能力应该用 MCP 还是 A2A：读取文件、合同审查、运行测试、生成上线风险报告、查询数据库、修复测试失败。
2. 设计一个“总控 Agent + 数据 Agent + 报告 Agent”的架构，标出 A2A 和 MCP 分别出现在哪里。
3. 解释为什么“报告 Agent 不应该直接拥有生产数据库权限”。
4. 画出一个 trace，包含一次 A2A task 和两次 MCP tool call。
5. 思考：如果一个 MCP Tool 内部开始自己规划、追问、异步执行并返回多个产物，它是否应该升级成 A2A Agent？为什么？

## 29.24 本章小结

本章我们专门讲清楚了 A2A 与 MCP 的分工。

MCP 面向工具和资源连接，适合明确操作、短生命周期、输入输出清晰的能力。A2A 面向 Agent 协作，适合具备自主规划、长任务、状态同步、追问、产物返回和跨 Agent 治理的能力。生产系统中最常见的模式是：Agent 之间用 A2A，Agent 内部访问工具和资源用 MCP。

你可以把本章重点记成一句话：

> 不要用 A2A 伪装工具，也不要用 MCP 隐藏复杂 Agent；协议分工越清楚，系统边界、权限和审计就越清楚。

下一章我们会继续讲 A2A 中最关键的安全主题：跨 Agent 权限、身份、审计和信任模型。
