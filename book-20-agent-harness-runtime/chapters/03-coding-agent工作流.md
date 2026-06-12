# 第三章：Coding Agent 工作流

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修时，联网核对了 OpenAI Agents SDK 关于 tools、guardrails、sessions、human-in-the-loop 和 tracing 的公开文档，Anthropic Claude Code 关于权限、hooks、MCP、skills 和常见工作流的公开说明，OpenHands 关于 runtime、sandbox、evaluation 和安全边界的公开资料，以及 SWE-agent / mini-SWE-agent 围绕 SWE-bench、仓库导航、工具交互和 Agent-Computer Interface 的公开资料。

本讲只讨论防御性的 coding agent 工作流设计：如何理解任务、探索仓库、制定计划、小步修改、运行验证、使用失败反馈、保护用户改动和记录 trace。它不提供绕过权限、规避 sandbox、自动执行危险命令、读取敏感文件、批量删除仓库内容或对真实生产系统做高风险操作的技巧。

第二轮精修重点放在三件事：

1. 把“探索 / 计划 / 编辑 / 验证”从经验流程补成可度量的工作流指标。
2. 用 GitHub 兼容的数学公式描述任务分类、仓库探索、patch 聚焦、测试反馈和权限门禁。
3. 增加一个 0 依赖 Python demo，演示如何用 toy trace 审计 coding agent workflow，而不是只看最终代码 diff。

## 3.1 本章目标

前两章讲了 harness 和 runtime 的结构。本章进入更具体的问题：一个 coding agent 接到真实代码任务后，应该按什么工作流推进？

Coding agent 不是“看一眼需求，直接生成一大段代码”。真实代码库里有已有架构、隐含约束、测试体系、用户未提交改动、依赖版本、风格规范和安全边界。好的 coding agent 工作流必须先理解环境，再小步修改，最后用测试或其他证据验证。

学完本章，你应该能回答：

1. Coding agent 常见任务类型有哪些。
2. 为什么工作流要先探索、再计划、再编辑。
3. Agent 如何选择读哪些文件、跑哪些命令。
4. 为什么小步编辑比一次性大改更可靠。
5. 测试失败后 agent 应该如何继续修复。
6. 工作流中哪些地方最容易出事故。
7. 面试中如何设计一个 coding agent 的端到端工作流。

## 3.2 典型任务流程

一个相对稳健的 coding agent 工作流通常是：

```text
接收任务
-> 澄清目标和约束
-> 探索代码库
-> 定位相关文件
-> 制定修改计划
-> 小步编辑
-> 运行测试或静态检查
-> 根据反馈继续修复
-> 总结改动和验证结果
```

展开来看：

1. 用户提出任务。
2. Agent 判断任务类型和风险。
3. Agent 读取项目说明、目录结构、相关文件和测试。
4. Agent 必要时向用户追问。
5. Agent 制定一个可执行计划。
6. Agent 选择最小修改范围。
7. Agent 应用 patch 或编辑文件。
8. Agent 运行相关测试、lint 或类型检查。
9. Agent 分析失败日志。
10. Agent 继续修复或回滚错误修改。
11. Agent 输出最终总结，包括改了什么、如何验证、还有什么风险。

工作流设计的核心原则：不要让模型在缺少上下文和验证信号时做大规模不可控修改。

## 3.3 Coding Agent 常见任务类型

不同任务需要不同工作流。

Bug fix：

1. 目标是修复已有错误。
2. 需要复现或理解报错。
3. 重点是定位根因和运行相关测试。

Feature implementation：

1. 目标是新增功能。
2. 需要理解现有架构和接口。
3. 重点是边界、兼容、测试和文档。

Refactor：

1. 目标是改善结构，不改变行为。
2. 需要强回归测试。
3. 重点是小步改、保持行为一致。

Test writing：

1. 目标是补测试。
2. 需要理解当前测试框架和已有风格。
3. 重点是覆盖边界和避免脆弱测试。

Code review：

1. 目标是发现风险。
2. 需要阅读 diff 和相关上下文。
3. 重点是 bug、回归、安全和测试缺口。

Migration：

1. 目标是升级依赖、API 或框架。
2. 需要大范围影响分析。
3. 重点是分阶段、兼容和回滚。

Debug production issue：

1. 目标是定位线上问题。
2. 需要日志、指标、配置和近期变更。
3. 重点是止损、复现和根因。

不同任务的“下一步最佳动作”不同。Runtime 可以在 task manager 中记录任务类型，指导 context builder 和工具选择。

## 3.4 任务开始：理解目标和边界

Coding agent 接到任务后，不应该立即改文件。

它应该先判断：

1. 用户想解决什么问题。
2. 任务是否明确。
3. 成功标准是什么。
4. 有哪些不能改的边界。
5. 是否需要先复现问题。
6. 是否涉及危险操作。
7. 是否需要用户补充信息。

模糊任务示例：

```text
帮我优化这个项目。
```

这时 agent 不应该直接重构代码，而应该追问或先做只读探索：

```text
你希望优化哪一类问题：性能、代码结构、测试覆盖、启动速度还是某个具体 bug？我可以先只读检查项目结构并给出建议，不直接修改文件。
```

明确任务示例：

```text
修复登录接口在空密码时返回 500 的问题，并补一个测试。
```

这时 agent 可以进入 bug fix 工作流：找接口、找测试、复现、修改、验证。

任务开始阶段最重要的是不要误解目标。很多 agent 事故不是写错代码，而是从一开始就做错任务。

## 3.5 探索代码库：先读什么

探索阶段的目标是建立最小必要上下文。

常见读取顺序：

1. 项目根目录文件，例如 README、package、pyproject、go.mod、pom.xml。
2. 目录结构。
3. 用户提到的文件。
4. 报错堆栈中出现的文件。
5. 相关测试文件。
6. 相关接口或调用方。
7. 项目约定文件，例如 CONTRIBUTING、AGENTS、lint 配置。

不要一开始读取整个代码库。

好的探索策略是：

1. 先定位入口。
2. 再找相关实现。
3. 再找测试。
4. 再找相邻模式。
5. 最后补缺失上下文。

例子：修复登录接口 bug。

Agent 可以先搜索：

```text
login
auth
password
empty password
```

然后读取：

1. 登录 route/controller。
2. auth service。
3. 登录相关测试。
4. 错误处理 middleware。

探索阶段的输出应该是“我知道要改哪里，以及为什么”，而不是“我读了很多文件”。

## 3.6 计划阶段：计划要短、可执行、可更新

计划的作用是约束 agent 不要乱改。

一个好计划应该包含：

1. 当前理解。
2. 要修改的文件。
3. 修改步骤。
4. 验证方式。
5. 风险点。

示例：

```text
计划：
1. 阅读 auth controller 和 auth service，确认空密码如何进入校验逻辑。
2. 在参数校验层处理空密码，避免进入下游 hash/DB 逻辑。
3. 补充空密码返回 400 的单元测试。
4. 运行 auth 相关测试。
```

计划不要太长。

坏计划：

```text
重构整个 auth 模块，统一错误处理，增加新鉴权流程，顺便优化测试。
```

这个计划扩大了任务范围。Coding agent 应该优先完成用户任务，而不是借机大改。

计划也不是一次写死。工具执行后发现新信息，计划要更新。Task Manager 可以记录 plan version 和 completed steps。

## 3.7 编辑阶段：小步修改

编辑阶段最重要的原则是小步、局部、可验证。

推荐做法：

1. 优先修改最小相关文件。
2. 遵循现有代码风格。
3. 不重写无关模块。
4. 不顺手格式化大文件。
5. 每次 patch 尽量聚焦一个意图。
6. 修改前确认文件内容是最新的。
7. 修改后保留 diff 给用户和 trace。

为什么不能一次大改？

1. 大改更容易引入回归。
2. 大改更难 review。
3. 大改失败后更难定位原因。
4. 大改可能覆盖用户改动。
5. 大改让 agent 自己也难以保持上下文一致。

编辑工具应该提供保护：

1. patch 必须能精确匹配旧内容。
2. 文件修改前后进入 trace。
3. 检测用户未提交改动。
4. 高风险文件修改需要确认。
5. 修改失败要返回结构化错误。

## 3.8 验证阶段：不验证就不算完成

Coding agent 最常见的问题之一是“看起来修了，但没有验证”。

验证方式包括：

1. 运行相关单元测试。
2. 运行集成测试。
3. 运行 lint 或 type check。
4. 运行格式化检查。
5. 构建项目。
6. 手动运行最小复现命令。
7. 对无测试项目做静态检查和代码路径解释。

验证优先级：

1. 先运行最相关、最快的测试。
2. 如果通过，再运行更大范围检查。
3. 如果失败，读取错误日志并判断是否和本次修改相关。
4. 如果无法运行测试，明确说明原因和替代验证。

最终总结必须包含验证结果：

```text
验证：已运行 npm test -- auth，全部通过。
```

如果不能验证：

```text
未能运行测试，因为当前环境缺少数据库服务。已补充单元测试代码，并基于现有错误处理路径做静态检查。建议在 CI 环境运行 auth test suite。
```

不要把“我认为应该可以”说成“已验证”。

## 3.9 测试失败后的修复循环

测试失败后，agent 不应该盲目继续修改。

正确流程：

1. 读取失败命令、exit code、stderr 和关键日志。
2. 判断失败是否和本次修改相关。
3. 如果是环境问题，说明并停止或请求用户。
4. 如果是测试断言不匹配，读取相关测试。
5. 如果是编译或类型错误，定位文件和行号。
6. 制定更小修复。
7. 修改后重新运行同一测试。
8. 多次失败后考虑回滚或请求用户介入。

错误处理示例：

```text
测试失败：expected 400 but received 422。
```

Agent 应该判断：

1. 需求到底要求 400 还是现有规范是 422。
2. 相邻测试如何定义 validation error。
3. 是代码错了，还是测试预期错了。
4. 是否需要询问用户。

不要为了让测试过而随便改测试。测试失败是信号，不是障碍。

## 3.10 总结阶段：告诉用户什么

最终回答应该简洁但完整。

至少包含：

1. 改了什么。
2. 为什么这样改。
3. 改了哪些文件。
4. 做了什么验证。
5. 是否还有风险或未完成事项。

示例：

```text
已修复空密码登录返回 500 的问题。改动在 auth service 的输入校验层，对空密码提前返回 validation error，避免进入后续 hash 逻辑。补充了空密码登录的单元测试。已运行 npm test -- auth，测试通过。
```

不要输出过长流水账。用户不需要每一步内部思考，但需要知道结果、验证和风险。

## 3.11 工作流中的权限点

Coding agent 工作流中有多个权限点。

通常可自动执行：

1. 读取项目内普通源码文件。
2. 搜索代码。
3. 查看 git diff。
4. 运行只读检查命令。

通常需要确认：

1. 修改文件。
2. 删除文件。
3. 安装依赖。
4. 执行带网络访问的命令。
5. 执行数据库迁移。
6. 大范围格式化。
7. 访问敏感文件。

通常应禁止或强确认：

1. 删除仓库目录。
2. 修改系统文件。
3. 读取密钥文件。
4. 上传代码或数据到外部服务。
5. 在未隔离环境运行不可信脚本。

工作流设计不能只看效率，还要看操作风险。

## 3.12 用户并发修改怎么办

Coding agent 和用户共享工作区。用户可能在 agent 运行过程中修改文件。

风险：

1. Agent 覆盖用户刚写的代码。
2. Patch 基于旧文件内容应用失败。
3. 测试失败原因来自用户新改动。
4. Agent 总结的 diff 不完整。

处理方式：

1. 修改前读取最新文件内容。
2. 应用 patch 时做上下文匹配。
3. 检测 git diff 或文件 mtime 变化。
4. 如果冲突，停止并提示用户。
5. 不自动覆盖未知改动。
6. Trace 中记录修改前后状态。

面试表达：

```text
Coding agent 不能假设工作区只有自己在改。编辑前要确认文件版本，patch 应该基于上下文匹配，发现用户并发修改或未提交改动时要停止并请求确认，而不是直接覆盖。
```

## 3.13 长任务工作流

长任务比短任务难很多。

长任务包括：

1. 跨多个模块实现功能。
2. 大规模迁移。
3. 多轮 debug。
4. 修复需要长时间测试的 bug。
5. 复杂 Agent benchmark 任务。

长任务需要：

1. 明确阶段。
2. 定期 checkpoint。
3. 更新计划。
4. 压缩历史。
5. 保留关键决策。
6. 分批验证。
7. 支持用户中断和恢复。

长任务反模式：

1. 一次性读太多文件。
2. 一次性改太多文件。
3. 不记录已完成步骤。
4. 测试失败后忘记最初目标。
5. 上下文过长后丢失关键约束。

Runtime 应该帮助 agent 管理长任务，而不是把所有历史都塞给模型。

## 3.14 工作流反模式

常见反模式包括：

1. 不探索代码库，直接写代码。
2. 不读测试，直接改实现。
3. 不看错误日志，凭感觉修。
4. 一次性重构大文件。
5. 为了通过测试随意改测试。
6. 忽略用户未提交改动。
7. 运行危险命令不请求确认。
8. 不验证就声称完成。
9. 上下文塞太多无关内容。
10. 最终总结不说明验证和风险。

这些反模式会让 agent 看起来很主动，但实际不可靠。

## 3.15 一个 Bug Fix 工作流示例

任务：

```text
修复注册接口在 email 为空时返回 500 的问题，并补测试。
```

稳健工作流：

1. 搜索 register、email、validation。
2. 读取注册 controller/service。
3. 读取已有注册测试。
4. 确认错误处理约定，比如参数错误返回 400 还是 422。
5. 复现或理解 500 来源。
6. 在输入校验层补空 email 处理。
7. 新增测试：空 email 返回约定错误码。
8. 运行注册相关测试。
9. 如果失败，读取断言和日志。
10. 修复后重新运行。
11. 总结改动和测试结果。

这个流程的重点是：先找约定，再做最小修复，再验证。

## 3.16 一个 Feature 工作流示例

任务：

```text
给导出接口增加 CSV 格式支持。
```

稳健工作流：

1. 读取现有导出接口。
2. 查找已有 JSON、Excel 或其他格式实现。
3. 确认参数设计和错误处理约定。
4. 查找相关测试和文档。
5. 制定计划：新增 format=csv 分支、序列化函数、测试。
6. 小步实现 CSV 序列化。
7. 补单元测试和接口测试。
8. 运行相关测试。
9. 更新文档或 API 示例。
10. 总结兼容性和风险。

Feature 任务最容易范围蔓延。Agent 应该避免顺手重构整个导出模块，除非用户明确要求。

## 3.17 一个 Refactor 工作流示例

任务：

```text
把重复的权限校验逻辑抽成公共函数。
```

稳健工作流：

1. 搜索重复逻辑。
2. 确认相似代码是否完全等价。
3. 读取相关测试。
4. 先抽一个最小公共函数。
5. 替换少量调用点。
6. 运行测试。
7. 再扩大替换范围。
8. 确认行为没有变化。
9. 总结替换范围和验证。

Refactor 的核心是保持行为一致。没有测试时，agent 应该更保守。

## 3.18 工作流如何进入 Trace

每个工作流步骤都应该进入 trace。

Trace 应该记录：

1. 用户任务。
2. agent 判断的任务类型。
3. 读取了哪些文件。
4. 搜索了哪些关键词。
5. 计划内容和计划更新。
6. 应用了哪些 patch。
7. 运行了哪些命令。
8. 命令结果。
9. 测试失败后的修复循环。
10. 权限确认。
11. 最终总结。

Trace 的作用：

1. 用户审计 agent 做了什么。
2. 开发者 debug agent 为什么失败。
3. evaluation harness 回放同一任务。
4. 收集 bad cases 和高质量成功轨迹。

没有 trace 的工作流不可复盘。

## 3.19 工作流质量指标

Coding agent 工作流可以被评估。

常见指标：

1. Task success rate。
2. Test pass rate。
3. 平均步骤数。
4. 平均工具调用次数。
5. 平均 token cost。
6. 平均 wall-clock time。
7. 无关文件修改率。
8. 测试未运行率。
9. 用户确认次数。
10. 回滚或失败恢复成功率。
11. 危险命令拦截率。
12. 用户满意度。

不要只看最终成功率。

一个 agent 可能成功率高，但成本极高、改动范围过大、经常运行危险命令。工作流质量要同时看成功、效率和安全。

### 3.19.1 关键公式与 Coding Agent 工作流指标速查

为了把 coding agent 工作流从“看起来挺稳”变成可评估对象，可以把第 $i$ 个任务轨迹记为：

```math
\tau_i=(g_i,k_i,\hat{k}_i,R_i,E_i,P_i,D_i,V_i,F_i,U_i,H_i,B_i,Y_i)
```

其中 $g_i$ 是用户目标，$k_i$ 是真实任务类型，$\hat{k}_i$ 是 agent 判断的任务类型，$R_i$ 是完成任务必须覆盖的关键文件或证据集合，$E_i$ 是探索阶段实际读取或搜索到的文件集合，$P_i$ 是计划，$D_i$ 是实际修改集合，$V_i$ 是验证命令集合，$F_i$ 是失败反馈集合，$U_i$ 是用户已有改动集合，$H_i$ 是高风险动作集合，$B_i$ 是预算，$Y_i$ 是最终结果。

任务分类准确率：

```math
A_{\mathrm{type}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat{k}_i=k_i]
```

如果 bug fix 被误判成 feature，agent 可能跳过复现；如果 refactor 被误判成 bug fix，agent 可能在没有回归测试时扩大修改范围。

仓库探索覆盖率：

```math
C_{\mathrm{explore}}=
\frac{1}{N}\sum_{i=1}^{N}
\frac{|E_i\cap R_i|}{|R_i|}
```

这里的 $R_i$ 不是全仓库，而是当前任务必须理解的最小证据集合，例如报错文件、相关实现、调用方和测试。探索覆盖率低，通常意味着 agent 在缺上下文时开始改代码。

Patch 定位精确率和召回率：

```math
P_{\mathrm{patch}}=
\frac{\sum_i |D_i\cap M_i|}{\sum_i |D_i|},
\qquad
R_{\mathrm{patch}}=
\frac{\sum_i |D_i\cap M_i|}{\sum_i |M_i|}
```

其中 $M_i$ 是理想情况下应该修改的文件或代码区域集合。$P_{\mathrm{patch}}$ 低说明无关改动太多，$R_{\mathrm{patch}}$ 低说明关键修改遗漏。

无关改动率：

```math
R_{\mathrm{unrel}}=
\frac{\sum_i |D_i-M_i|}{\sum_i |D_i|}
```

无关改动包括顺手格式化、跨模块重构、依赖升级、锁文件改动和与任务无关的文档改写。它们可能让 diff 看起来更大，但不一定提升任务成功率。

验证覆盖率：

```math
C_{\mathrm{val}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[|V_i|>0]
```

测试反馈使用率：

```math
U_{\mathrm{fb}}=
\frac{\sum_i \mathbb{1}[|F_i|>0\wedge \mathrm{used}(F_i)=1]}
{\sum_i \mathbb{1}[|F_i|>0]}
```

测试失败不是噪声，而是工作流的 observation。一个可靠 agent 应该读取失败命令、exit code、stderr、断言差异和相关测试，再决定是修实现、修测试预期、说明环境问题，还是请求用户确认。

用户改动触碰率：

```math
R_{\mathrm{user}}=
\frac{\sum_i \mathbb{1}[D_i\cap U_i\ne \varnothing \wedge c_i=0]}
{\sum_i \mathbb{1}[|U_i|>0]}
```

其中 $c_i$ 表示触碰用户已有改动前是否得到确认。这个指标应该接近 0；否则 agent 很容易覆盖用户刚写的代码。

高风险动作保护率：

```math
C_{\mathrm{risk}}=
\frac{\sum_i \sum_{h\in H_i}\mathbb{1}[\mathrm{blocked}(h)=1\vee \mathrm{confirmed}(h)=1]}
{\sum_i |H_i|}
```

高风险动作包括删除、迁移、安装依赖、网络访问、访问敏感文件、写生产资源和大范围格式化。工作流不应该把这些动作只交给 prompt 自觉，而要由 runtime / harness 做权限门禁。

综合工作流门禁：

```math
G_{\mathrm{workflow}}=
\mathbb{1}[
A_{\mathrm{type}}\ge \alpha_{\mathrm{type}}
\wedge C_{\mathrm{explore}}\ge \alpha_{\mathrm{explore}}
\wedge P_{\mathrm{patch}}\ge \alpha_{\mathrm{patch}}
\wedge C_{\mathrm{val}}\ge \alpha_{\mathrm{val}}
\wedge U_{\mathrm{fb}}\ge \alpha_{\mathrm{fb}}
\wedge R_{\mathrm{user}}=0
\wedge C_{\mathrm{risk}}\ge \alpha_{\mathrm{risk}}
]
```

面试里可以把这套指标总结成一句话：coding agent 不是“生成代码的模型”，而是一个可审计的任务执行系统；上线前要同时证明它会选对任务策略、读对上下文、改对位置、用好测试反馈、保护用户改动、遵守权限边界，并能留下可复盘 trace。

### 3.19.2 最小可运行 Coding Agent Workflow 审计 demo

下面的 0 依赖脚本构造 6 条 toy coding agent trace，覆盖正常 bug fix、未探索直接编辑、无关重构、测试失败被忽略、覆盖用户改动和高风险命令未确认。它演示如何把工作流质量拆成指标，而不是只看最终 diff。

```python
from statistics import mean


REQUIRED_TRACE_FIELDS = {
    "task",
    "task_type",
    "search",
    "read_files",
    "plan",
    "edits",
    "commands",
    "feedback",
    "permissions",
    "summary",
}


def safe_div(num, den, default=1.0):
    return default if den == 0 else num / den


traces = [
    {
        "id": "bugfix_ok",
        "expected_type": "bugfix",
        "predicted_type": "bugfix",
        "required_files": {"auth.py", "test_auth.py"},
        "expected_edits": {"auth.py", "test_auth.py"},
        "explored_files": {"README.md", "auth.py", "test_auth.py"},
        "plan_present": True,
        "edited_files": {"auth.py", "test_auth.py"},
        "commands": [
            {"name": "pytest test_auth.py", "kind": "test", "success": True},
            {"name": "ruff check auth.py", "kind": "lint", "success": True},
        ],
        "failure_used": True,
        "final_verified": True,
        "false_completion": False,
        "user_modified_files": set(),
        "touched_user_change_without_confirm": False,
        "dependency_change": False,
        "high_risk_actions": [],
        "trace_fields": REQUIRED_TRACE_FIELDS,
    },
    {
        "id": "direct_edit_no_explore",
        "expected_type": "bugfix",
        "predicted_type": "feature",
        "required_files": {"auth.py", "test_auth.py"},
        "expected_edits": {"auth.py", "test_auth.py"},
        "explored_files": {"app.py"},
        "plan_present": False,
        "edited_files": {"app.py"},
        "commands": [],
        "failure_used": False,
        "final_verified": False,
        "false_completion": True,
        "user_modified_files": set(),
        "touched_user_change_without_confirm": False,
        "dependency_change": False,
        "high_risk_actions": [],
        "trace_fields": {"task", "task_type", "edits", "summary", "permissions"},
    },
    {
        "id": "broad_refactor_unrelated",
        "expected_type": "refactor",
        "predicted_type": "refactor",
        "required_files": {"permissions.py", "test_permissions.py"},
        "expected_edits": {"permissions.py"},
        "explored_files": {"README.md", "permissions.py", "test_permissions.py"},
        "plan_present": True,
        "edited_files": {"permissions.py", "user.py", "settings.py", "pyproject.toml"},
        "commands": [{"name": "pytest test_permissions.py", "kind": "test", "success": True}],
        "failure_used": True,
        "final_verified": True,
        "false_completion": False,
        "user_modified_files": set(),
        "touched_user_change_without_confirm": False,
        "dependency_change": True,
        "high_risk_actions": [],
        "trace_fields": REQUIRED_TRACE_FIELDS - {"feedback"},
    },
    {
        "id": "test_failure_ignored",
        "expected_type": "feature",
        "predicted_type": "feature",
        "required_files": {"export.py", "test_export.py"},
        "expected_edits": {"export.py", "test_export.py"},
        "explored_files": {"export.py", "test_export.py"},
        "plan_present": True,
        "edited_files": {"export.py"},
        "commands": [{"name": "pytest test_export.py", "kind": "test", "success": False}],
        "failure_used": False,
        "final_verified": False,
        "false_completion": True,
        "user_modified_files": set(),
        "touched_user_change_without_confirm": False,
        "dependency_change": False,
        "high_risk_actions": [],
        "trace_fields": REQUIRED_TRACE_FIELDS - {"feedback"},
    },
    {
        "id": "user_change_overwritten",
        "expected_type": "bugfix",
        "predicted_type": "bugfix",
        "required_files": {"report.py", "test_report.py"},
        "expected_edits": {"report.py"},
        "explored_files": {"report.py", "test_report.py"},
        "plan_present": True,
        "edited_files": {"report.py"},
        "commands": [{"name": "pytest test_report.py", "kind": "test", "success": True}],
        "failure_used": True,
        "final_verified": True,
        "false_completion": False,
        "user_modified_files": {"report.py"},
        "touched_user_change_without_confirm": True,
        "dependency_change": False,
        "high_risk_actions": [],
        "trace_fields": REQUIRED_TRACE_FIELDS - {"permissions"},
    },
    {
        "id": "risky_command_missing_confirm",
        "expected_type": "migration",
        "predicted_type": "migration",
        "required_files": {"schema.py", "test_schema.py"},
        "expected_edits": {"schema.py", "test_schema.py"},
        "explored_files": {"schema.py"},
        "plan_present": True,
        "edited_files": {"schema.py"},
        "commands": [],
        "failure_used": False,
        "final_verified": False,
        "false_completion": False,
        "user_modified_files": set(),
        "touched_user_change_without_confirm": False,
        "dependency_change": False,
        "high_risk_actions": [
            {"name": "db migrate --prod", "blocked": False, "confirmed": False}
        ],
        "trace_fields": REQUIRED_TRACE_FIELDS - {"commands", "feedback"},
    },
]


def validation_commands(trace):
    return [cmd for cmd in trace["commands"] if cmd["kind"] in {"test", "lint", "type", "build"}]


def failed_validation_commands(trace):
    return [cmd for cmd in validation_commands(trace) if not cmd["success"]]


def trace_score(trace):
    expected = trace["expected_edits"]
    edited = trace["edited_files"]
    explored = trace["explored_files"]
    required = trace["required_files"]
    type_ok = trace["expected_type"] == trace["predicted_type"]
    explore = safe_div(len(explored & required), len(required))
    patch_precision = safe_div(len(edited & expected), len(edited))
    validation_ok = bool(validation_commands(trace))
    feedback_ok = not failed_validation_commands(trace) or trace["failure_used"]
    user_ok = not trace["touched_user_change_without_confirm"]
    risk_ok = all(a["blocked"] or a["confirmed"] for a in trace["high_risk_actions"])
    trace_complete = len(trace["trace_fields"] & REQUIRED_TRACE_FIELDS) / len(REQUIRED_TRACE_FIELDS)
    success_ok = trace["final_verified"] and not trace["false_completion"]
    parts = [
        float(type_ok),
        explore,
        float(trace["plan_present"]),
        patch_precision,
        float(validation_ok),
        float(feedback_ok),
        float(user_ok),
        float(risk_ok),
        trace_complete,
        float(success_ok),
    ]
    return round(mean(parts), 3)


def root_causes(trace):
    causes = []
    if trace["expected_type"] != trace["predicted_type"]:
        causes.append("task_type_mismatch")
    if (trace["explored_files"] & trace["required_files"]) != trace["required_files"]:
        causes.append("missing_required_exploration")
    if not trace["plan_present"]:
        causes.append("missing_plan")
    if trace["edited_files"] - trace["expected_edits"]:
        causes.append("unrelated_patch")
    if trace["dependency_change"]:
        causes.append("unnecessary_dependency_change")
    if not validation_commands(trace):
        causes.append("missing_validation")
    if failed_validation_commands(trace) and not trace["failure_used"]:
        causes.append("test_feedback_ignored")
    if trace["false_completion"]:
        causes.append("false_completion")
    if trace["touched_user_change_without_confirm"]:
        causes.append("user_change_violation")
    if any(not (a["blocked"] or a["confirmed"]) for a in trace["high_risk_actions"]):
        causes.append("risky_action_without_confirmation")
    if len(trace["trace_fields"] & REQUIRED_TRACE_FIELDS) < len(REQUIRED_TRACE_FIELDS):
        causes.append("trace_incomplete")
    return causes or ["pass"]


total_edits = sum(len(t["edited_files"]) for t in traces)
total_expected = sum(len(t["expected_edits"]) for t in traces)
relevant_edits = sum(len(t["edited_files"] & t["expected_edits"]) for t in traces)
unrelated_edits = sum(len(t["edited_files"] - t["expected_edits"]) for t in traces)
validation_total = sum(len(validation_commands(t)) for t in traces)
validation_passed = sum(sum(cmd["success"] for cmd in validation_commands(t)) for t in traces)
failed_with_feedback = [t for t in traces if failed_validation_commands(t)]
user_modified_runs = [t for t in traces if t["user_modified_files"]]
all_risky = [a for t in traces for a in t["high_risk_actions"]]
protected_risky = [a for a in all_risky if a["blocked"] or a["confirmed"]]

metrics = {
    "task_success_rate": round(mean(t["final_verified"] and not t["false_completion"] for t in traces), 3),
    "task_type_accuracy": round(mean(t["expected_type"] == t["predicted_type"] for t in traces), 3),
    "exploration_coverage": round(
        mean(safe_div(len(t["explored_files"] & t["required_files"]), len(t["required_files"])) for t in traces),
        3,
    ),
    "planning_coverage": round(mean(t["plan_present"] for t in traces), 3),
    "patch_precision": round(safe_div(relevant_edits, total_edits), 3),
    "patch_recall": round(safe_div(relevant_edits, total_expected), 3),
    "unrelated_change_rate": round(safe_div(unrelated_edits, total_edits, default=0.0), 3),
    "minimal_patch_rate": round(
        mean(
            not (t["edited_files"] - t["expected_edits"])
            and not t["dependency_change"]
            and len(t["edited_files"]) <= len(t["expected_edits"]) + 1
            for t in traces
        ),
        3,
    ),
    "validation_coverage": round(mean(bool(validation_commands(t)) for t in traces), 3),
    "test_pass_rate": round(safe_div(validation_passed, validation_total, default=0.0), 3),
    "feedback_use_rate": round(
        safe_div(sum(t["failure_used"] for t in failed_with_feedback), len(failed_with_feedback), default=1.0),
        3,
    ),
    "user_change_violation_rate": round(
        safe_div(
            sum(t["touched_user_change_without_confirm"] for t in user_modified_runs),
            len(user_modified_runs),
            default=0.0,
        ),
        3,
    ),
    "risky_action_protection": round(safe_div(len(protected_risky), len(all_risky), default=1.0), 3),
    "trace_completeness": round(
        mean(len(t["trace_fields"] & REQUIRED_TRACE_FIELDS) / len(REQUIRED_TRACE_FIELDS) for t in traces),
        3,
    ),
}

gates = {
    "task_success_ok": metrics["task_success_rate"] >= 0.75,
    "type_ok": metrics["task_type_accuracy"] >= 0.9,
    "exploration_ok": metrics["exploration_coverage"] >= 0.85,
    "patch_ok": metrics["patch_precision"] >= 0.85 and metrics["unrelated_change_rate"] <= 0.1,
    "validation_ok": metrics["validation_coverage"] >= 0.8 and metrics["test_pass_rate"] >= 0.8,
    "feedback_ok": metrics["feedback_use_rate"] >= 0.8,
    "user_change_ok": metrics["user_change_violation_rate"] == 0.0,
    "risk_ok": metrics["risky_action_protection"] >= 1.0,
    "trace_ok": metrics["trace_completeness"] >= 0.9,
}

ranked = sorted(
    [(t["id"], trace_score(t), root_causes(t) == ["pass"]) for t in traces],
    key=lambda item: item[1],
    reverse=True,
)

print("ranked=", ranked)
print("metrics=", metrics)
print("root_causes=", {t["id"]: root_causes(t) for t in traces})
print("failed_gates=", [name for name, ok in gates.items() if not ok])
print("workflow_gate_pass=", all(gates.values()))
```

一次典型输出会显示：正常 bug fix 得分最高，但整组 workflow gate 不通过，因为样本中存在未探索直接编辑、无关重构、测试失败后 false completion、用户改动触碰和高风险命令未确认。真实 evaluation harness 也应该保留这些失败样本，而不是只展示 happy path。

## 3.20 面试题：Coding Agent 的典型工作流是什么

回答要点：

```text
我会把 coding agent 工作流设计成“理解任务 -> 探索代码库 -> 定位相关文件 -> 制定计划 -> 小步编辑 -> 运行验证 -> 根据反馈修复 -> 总结结果”。关键是不要让模型直接大段生成代码，而是让它先收集足够上下文，再做最小修改，并用测试、lint 或静态检查验证。整个过程要保留 trace，并对写文件、危险命令和外部访问做权限确认。
```

## 3.21 面试题：Agent 如何决定读哪些文件

回答要点：

```text
我会让 agent 先从任务描述、报错栈、目录结构和项目配置定位入口，再用搜索找相关实现、调用方和测试。读取文件时优先读和当前任务直接相关的文件，而不是全仓库扫描。Context builder 应该保留文件路径、关键片段和行号，并在上下文过长时摘要旧信息。目标是构建最小必要上下文。
```

## 3.22 面试题：Agent 修改后如何验证

回答要点：

```text
优先运行和修改范围最相关、成本最低的测试，例如单元测试或特定 test target。如果通过，再按需要运行更大范围的 lint、type check 或集成测试。如果测试无法运行，要明确说明环境限制，并提供替代验证，例如静态检查、代码路径解释或建议在 CI 中运行。不能在未验证时声称任务已完成。
```

## 3.23 小练习

1. 为“修复登录接口空密码 500”设计一个 coding agent 工作流。
2. 为“新增 CSV 导出功能”设计探索、计划、编辑和验证步骤。
3. 列出 5 个 agent 不应该自动执行的危险命令。
4. 设计一个最终总结模板，包含改动、文件、验证和风险。
5. 设计一个 trace schema，用于记录 coding agent 的工作流步骤。
6. 思考 agent 测试失败三次后应该如何处理。
7. 思考用户并发修改文件时 agent 应该如何发现和响应。
8. 用 3 分钟回答“为什么 coding agent 要先探索再修改”。

## 3.24 本章总结

本章从操作层拆解了 coding agent 的真实工作流。

核心结论：

1. Coding agent 不应该直接大段生成代码，而应该先理解任务和代码库。
2. 工作流通常包括探索、计划、编辑、验证、修复和总结。
3. 不同任务类型需要不同策略，bug fix、feature、refactor、test writing 和 code review 不能用同一套流程粗暴处理。
4. 小步编辑和及时验证比一次性大改更可靠。
5. 测试失败是反馈信号，不是简单障碍。
6. 用户并发修改、危险命令、长任务和上下文压缩都是工作流设计中的真实难点。
7. Trace 是工作流可观测、可回放、可评估的基础。

下一章会进入工具系统与 Tool Registry，讨论 runtime 如何把 Read、Edit、Search、Bash、Apply Patch、Test Runner 等能力安全、结构化地暴露给模型。
