# 第十六章：Harness 实战坑与面试题

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修只基于公开资料和前文已经建立的 Agent Harness 抽象：Claude Code security / troubleshooting 文档、OpenCode troubleshooting / permissions / MCP 文档、SWE-agent trajectory / output files 文档、Aider troubleshooting / repo map 文档、MCP specification 与安全最佳实践、OpenAI Agents SDK 的 agent、tool、session、human-in-the-loop、MCP 和 tracing 公开资料，以及本册前十五章对 runtime、工具、权限、沙箱、trace、replay 和 evaluation harness 的拆解。

本章不把任何闭源 coding agent 的内部 planner、隐藏 prompt、工具排序、沙箱实现、远端 trace 存储或模型路由写成确定事实；如果需要讨论，只作为系统设计推断。本章只讨论防御性工程排查、面试表达和 toy audit demo，不提供绕过权限、规避沙箱、读取密钥、破坏工作区、攻击 MCP/A2A 服务、隐藏 trace、伪造评估通过或自动执行生产高风险动作的方法。

第二轮补强重点有三点：

1. 把“实战坑”从经验清单升级成可审计指标，避免面试回答停留在“要加权限、要加日志”的口号。
2. 用稳定 MathJax 公式定义排查覆盖率、权限安全、上下文预算、编辑安全、命令安全、注入边界、trace、replay、eval、成本循环和环境一致性。
3. 补一个 0 依赖 Python demo，输入 toy incident 表，输出 failed gates 和 root causes，帮助读者把事故复盘讲成可复现工程方法。

## 16.1 本章定位

这是第二十册的收尾章。前面我们已经系统讲过 Agent Harness、Coding Agent Runtime、工具系统、文件编辑、终端执行、上下文压缩、权限沙箱、trace/replay、evaluation harness、Claude Code、OpenCode、Codex、MCP、A2A 和系统设计。本章不再引入大量新概念，而是把这些内容转成实战排查清单和面试回答模板。

真实 coding agent 系统的问题很少是“模型完全不会写代码”。更常见的是：上下文错了、工具太多、权限太松、编辑失败、命令副作用、trace 不完整、eval 不稳定、MCP 输出污染上下文、云端沙箱和本地环境不一致。

学完本章，你应该能回答：

1. Coding agent harness 最常见的线上问题有哪些。
2. 每类问题的症状、根因、排查步骤和修复方案是什么。
3. 面试中如何从模型、上下文、工具、权限、执行、trace、eval 和产品体验几个维度回答 harness 问题。
4. 如何把第二十册内容压缩成一套高频面试题答案。

## 16.2 资料来源和可信边界

本章综合参考以下公开资料和前面章节内容：

1. Claude Code security 文档：read-only 默认、显式权限、sandboxed bash、写权限限制、prompt injection 防护、trust verification、MCP security、云端 VM 隔离和 audit logging。
2. Claude Code troubleshooting 文档：高 CPU/内存、auto-compaction thrashing、大文件和大工具输出、命令 hang、search 问题、WSL 文件系统性能问题。
3. OpenCode troubleshooting 文档：日志位置、storage、desktop server 连接、插件、缓存、provider 和 auth 问题。
4. OpenCode permissions 文档：allow/ask/deny、granular rules、external directory、`.env` 默认 deny、doom loop、agent 级权限。
5. OpenCode MCP 文档：MCP server 会增加上下文、GitHub 等 MCP server 可能消耗大量 token、local/remote MCP、OAuth、按 agent 管理 MCP。
6. SWE-agent output files 文档：trajectory 记录 thought/action/observation/state/query，是 debug 和研究复现的重要基础。
7. Aider troubleshooting 和 repo map 文档：编辑失败、模型警告、token 限制、repo map token budget 等问题。
8. MCP specification：MCP 安全原则、user consent、tool safety、data privacy、sampling controls。

不同产品的具体命令和实现会变化。本章提炼的是通用工程模式，不把某个工具的私有细节当成行业标准。

## 16.3 总排查框架

遇到 coding agent 出问题，不要一上来就说“模型不行”。先按八层排查：

```text
1. User goal: 用户目标是否清楚？
2. Context: 模型看到的信息是否正确、足够、不过载？
3. Model: 模型能力、参数、provider 是否正常？
4. Tool: 工具 schema、参数、输出是否正确？
5. Permission: 权限是否过松或过紧？
6. Execution: 文件、shell、sandbox、网络、环境是否正确？
7. State: session、diff、todo、memory、history 是否一致？
8. Observability: trace、logs、replay、eval 是否能定位问题？
```

一个好的排查顺序是：

1. 看 final diff 和用户目标是否匹配。
2. 看 trace 中模型看到的 query/context。
3. 看工具调用输入输出。
4. 看权限请求和用户批准。
5. 看 shell/test/lint 输出。
6. 看文件修改前后的 diff。
7. 看是否有上下文压缩或大输出污染。
8. 看是否能 replay。

如果没有 trace，只能猜。这就是为什么生产级 harness 必须记录足够轨迹。

## 16.4 坑一：工具权限过大

症状：

1. Agent 自动执行了删除、覆盖、部署、发邮件、写数据库等高风险动作。
2. Agent 读取了不该读的 `.env`、token、私有目录。
3. MCP server 拿到了过宽 OAuth scope。
4. 用户不知道 agent 做了哪些副作用操作。

常见根因：

1. 默认所有工具 allow。
2. 没有区分 read-only、edit、bash、network、external system。
3. 权限只靠 prompt，没有 runtime enforcement。
4. MCP/A2A 默认可信。
5. 没有路径级、参数级、agent 级权限。

修复方案：

1. 默认 read-only，写文件和 bash 需要 ask。
2. 对高风险命令 deny，例如 `rm -rf`、`git push`、生产 deploy、数据库写入。
3. 对 `.env`、secret、外部目录默认 deny 或 ask。
4. MCP server 做 allowlist，OAuth scope 最小化。
5. agent 级权限隔离：review agent 不能 edit，build agent 可以 run test 但不能 push。
6. 所有权限决策写入 trace。

面试表达：

```text
权限系统必须在 runtime 层实现。Prompt 可以告诉模型不要执行危险命令，但不能作为安全边界。真正的边界应该是 permission engine，根据工具、参数、路径、agent、用户和组织策略输出 allow、ask 或 deny。
```

## 16.5 坑二：上下文被大文件或大工具输出挤爆

症状：

1. Agent 前面理解任务，后面突然忘记目标。
2. Auto-compaction 反复触发，甚至出现 thrashing。
3. MCP 或日志输出非常长，模型开始围绕无关日志回答。
4. 读取整个 bundle、lockfile、minified 文件、超长测试日志。

常见根因：

1. 无 token budget。
2. 工具输出不截断。
3. 大文件整读。
4. MCP tools 全量加载到上下文。
5. history compaction 没有保留关键任务状态和 diff。

修复方案：

1. 大文件按范围读取。
2. 工具输出分页、摘要或落盘引用。
3. 对 MCP tools 做 tool search 或按 agent 启用。
4. 使用 repo map，而不是全仓库塞上下文。
5. compaction 时保留用户目标、当前 plan、已改 diff、失败测试、关键文件。
6. 对 node_modules、dist、build、lockfile、minified 文件默认降权。

排查问题时看：

1. 当前 context 中最大块是什么。
2. 最近一次工具输出是否超长。
3. 是否读取了无关大文件。
4. compaction 后是否丢失用户目标。

## 16.6 坑三：文件编辑覆盖用户改动

症状：

1. Agent 修改后，用户并发改动丢失。
2. Patch apply 成功但覆盖了无关代码。
3. 整文件重写导致格式和注释大面积变化。
4. Revert 不能精确撤销。

常见根因：

1. 没有 before hash。
2. 没有检测工作区 dirty state。
3. 使用 whole-file generation。
4. Patch context 太宽或太窄。
5. 没有把 FileChange 绑定到 message。

修复方案：

1. 每次 edit 前记录 file hash。
2. Apply 前检查文件是否被外部修改。
3. 优先小 patch，避免整文件重写。
4. 生成 diff 并让用户审查。
5. 支持按 message revert。
6. 对冲突重新读文件并重新规划。

面试表达：

```text
文件编辑不能只看模型输出是否合理，还要看 apply 语义。生产系统需要 before hash、patch apply、diff review、conflict detection 和 message-level revert，避免覆盖用户并发修改。
```

## 16.7 坑四：终端命令有副作用

症状：

1. Agent 执行了安装依赖、迁移数据库、删除缓存、启动长进程。
2. 命令 hang 住，session 卡死。
3. 命令在错误 cwd 执行。
4. 环境变量泄漏到日志或模型上下文。
5. 本地能跑，云端 sandbox 不一致。

常见根因：

1. bash 权限过宽。
2. 没有 timeout。
3. 没有 cwd/env 显式记录。
4. 没有区分 test/lint 和 destructive command。
5. sandbox 没有限制网络和文件系统。

修复方案：

1. 命令级 allow/ask/deny。
2. 对每个 command 设置 timeout。
3. 记录 cwd、env 摘要、exit code、stdout/stderr。
4. 长进程要求后台管理或明确终止策略。
5. 网络命令默认 ask。
6. 生产环境命令 deny。
7. 云端使用 VM/container 隔离，并限制网络。

安全原则：复杂 bash 命令应该给自然语言解释，让用户知道它要做什么。

## 16.8 坑五：Prompt Injection 通过外部内容进入系统

症状：

1. GitHub issue、网页、README、MCP 输出中写着“忽略之前指令”。
2. Agent 试图读取 secret 或执行与任务无关命令。
3. 外部文档让 agent 改权限或连接未知服务。

常见根因：

1. 把外部内容当成 trusted instruction。
2. Context builder 没有标注来源和信任级别。
3. Permission engine 被模型输出绕过。
4. Web/MCP 内容没有隔离。

修复方案：

1. 外部内容统一标注为 untrusted data。
2. 系统指令和安全策略优先级最高。
3. 高风险动作必须 permission ask。
4. Web fetch、MCP output、issue content 做 prompt injection 检测或隔离摘要。
5. 不允许外部内容修改权限策略。
6. 对命令注入做参数化和 blocklist。

面试表达：

```text
Prompt injection 不能靠让模型“更听话”解决。要把外部内容当成数据，不当成指令，并让 permission engine、sandbox、secret redaction 和工具边界承担真正防护。
```

## 16.9 坑六：Trace 不完整导致无法 Debug

症状：

1. 线上事故后不知道 agent 为什么执行某个命令。
2. 只能看到最终回答，看不到中间工具调用。
3. 无法复现同一个任务。
4. Eval 失败不知道是模型、工具、上下文还是环境问题。

常见根因：

1. 只记录聊天消息，不记录 tool call。
2. 不记录 context/query。
3. 不记录权限决策。
4. 不记录文件 diff。
5. 不记录外部系统版本和输出 hash。

修复方案：

1. 记录 model query metadata。
2. 记录 model output 和 tool call。
3. 记录 tool input/output、exit code、耗时、错误。
4. 记录 permission request 和用户响应。
5. 记录 FileChange 和 patch。
6. 记录 MCP/A2A server、版本、参数摘要、结果摘要。
7. 支持 deterministic replay 和 re-evaluation replay。

SWE-agent 的 trajectory 思想很有代表性：每一步都记录 response、thought/action、observation、state 和 query，方便研究和 debug。

## 16.10 坑七：Evaluation 不可比

症状：

1. 今天 pass rate 提升，明天下降，不知道为什么。
2. 换了 prompt、模型、工具、repo snapshot 后仍放在一起比较。
3. 外部 API 返回变化导致任务不稳定。
4. 测试本身 flaky。

常见根因：

1. 没有固定 repo snapshot。
2. 没有记录 prompt/model/tool 版本。
3. 没有隔离外部依赖。
4. 没有统计成本和失败类型。
5. benchmark 任务和真实业务分布不一致。

修复方案：

1. 固定 task set、repo snapshot、测试命令。
2. 记录 agent config、prompt version、model version、tool version。
3. 对 MCP/A2A 外部结果做 mock 或 snapshot。
4. 区分 compile fail、test fail、timeout、permission fail、wrong patch。
5. 使用公共 benchmark + 私有历史 issue benchmark。
6. 加入人工 review 和安全评分。

面试表达：

```text
Evaluation harness 的核心不是跑一批 prompt，而是固定任务、环境、工具、模型和评分函数，让不同版本 agent 的结果可比。否则 pass rate 的变化没有工程意义。
```

## 16.11 坑八：Agent 循环调用导致成本失控

症状：

1. Agent 反复 grep 同一个关键词。
2. 反复读取同一个文件。
3. 反复运行失败测试但不改变代码。
4. subagent 互相委派或重复执行。
5. token 和费用快速上升。

常见根因：

1. 没有 step limit。
2. 没有 repeated tool call detection。
3. 工具失败后没有状态更新。
4. Agent plan 不稳定。
5. 没有 stop condition。

修复方案：

1. 每个 session 设置 max steps、max cost、max wall time。
2. 对相同 tool+input 连续调用触发 doom loop guard。
3. 工具失败后要求模型解释新策略。
4. 用 todo/plan 约束任务进展。
5. subagent 设置独立预算。
6. 触发循环时总结当前状态并询问用户。

OpenCode 的 `doom_loop` 权限项就是这类保护的一个公开例子：相同工具调用重复多次时触发安全机制。

## 16.12 坑九：MCP Server 过多导致工具选择混乱

症状：

1. 模型选择了错误 MCP tool。
2. 工具 schema 占满上下文。
3. GitHub、Jira、Sentry、Docs 工具全打开，模型不知道用哪个。
4. MCP server 认证失败但错误不明显。

常见根因：

1. MCP 全局启用。
2. 工具命名和描述不清。
3. 没有 tool search 或 lazy loading。
4. 没有按 agent 管理工具。
5. OAuth 和 server 状态不可见。

修复方案：

1. MCP server 默认按需启用。
2. 按 agent 或任务类型启用工具。
3. 工具加 namespace 和清晰 description。
4. 使用 tool search，避免全量 schema 进上下文。
5. 暴露 MCP status、auth status、tool count。
6. 对大输出设置 output limit。

面试表达：

```text
MCP 扩展能力很强，但不能无限制堆工具。Harness 要把 MCP tools 纳入 capability registry、权限、上下文预算和状态监控，否则工具越多，模型越容易误选，成本也越高。
```

## 16.13 坑十：本地和云端环境不一致

症状：

1. 本地测试通过，云端失败。
2. 云端没有某个依赖、系统库或环境变量。
3. 文件路径、换行、权限、shell 不同。
4. WSL 或跨文件系统搜索变慢或漏结果。

常见根因：

1. 没有标准 workspace image。
2. 环境初始化不完整。
3. 依赖安装不确定。
4. 命令默认 shell 不同。
5. 文件系统性能和语义不同。

修复方案：

1. 使用 dev container、Docker 或固定 VM image。
2. 记录 environment fingerprint。
3. 初始化阶段运行 doctor/check。
4. 明确 shell、cwd、PATH。
5. 对 WSL/Windows/macOS 做平台适配。
6. 失败时输出环境诊断信息。

## 16.14 实战排查清单

当用户说“agent 改错了代码”，按这个顺序查：

1. 用户原始 prompt 是否清楚。
2. Agent 当时看到哪些文件和规则。
3. 是否读取了正确文件。
4. 是否有错误 repo map 或过期 summary。
5. 模型生成了什么 plan。
6. 执行了哪些 tool call。
7. 每个 edit 的 before/after diff。
8. 是否覆盖用户并发改动。
9. 是否运行了测试。
10. 最终回答是否如实说明验证情况。

当用户说“agent 卡住了”，按这个顺序查：

1. 当前是否在等待权限确认。
2. 是否有 shell 命令 hang。
3. 是否模型 API timeout 或 rate limit。
4. 是否 auto-compaction thrashing。
5. 是否 MCP server 连接或认证失败。
6. 是否 event stream 断开。
7. 是否进入重复工具调用。

当用户说“agent 太贵”，按这个顺序查：

1. 平均 steps。
2. 每步 context size。
3. 是否全量加载 MCP tools。
4. 是否读取大文件或大日志。
5. 是否频繁重新总结。
6. 是否使用过大模型处理简单任务。
7. 是否缺少缓存和 repo map。

当用户说“agent 不安全”，按这个顺序查：

1. 默认权限是否过松。
2. `.env` 是否可读。
3. 外部目录是否可写。
4. 网络请求是否默认允许。
5. MCP server 是否可信。
6. OAuth scope 是否过宽。
7. 是否有 audit log。
8. 是否有 prompt injection 防护。

## 16.15 高频面试题总表

| 问题 | 回答关键词 |
|---|---|
| Harness 是什么？ | 把模型、工具、状态、权限、执行、trace、eval 组织成 runtime |
| 为什么 coding agent 需要 runtime？ | 需要读写文件、执行命令、管理上下文、权限、安全和恢复 |
| Tool registry 怎么设计？ | schema、namespace、risk、permission、timeout、output limit、origin |
| 权限系统怎么设计？ | allow/ask/deny，按工具、参数、路径、agent、组织策略判断 |
| 如何防 prompt injection？ | 外部内容 untrusted，权限 runtime enforcement，secret redaction |
| 如何防危险命令？ | bash pattern、sandbox、timeout、blocklist、用户确认 |
| 如何管理上下文？ | repo map、检索、摘要、token budget、大输出截断 |
| 如何支持 replay？ | trace model/tool/context/diff/permission，mock 外部依赖 |
| 如何评估 agent？ | task set、repo snapshot、tests、scoring、cost、safety、human review |
| MCP 如何接入？ | MCP client、capability registry、permission、output limit、trace |
| A2A 如何接入？ | Agent Card、task delegation、streaming、trust policy、trace |
| Claude Code/OpenCode/Codex 差异？ | 产品入口、闭源/开源、配置、权限、server/SDK、生态和 workflow |

## 16.16 高频题参考答案

### 题 1：Harness 是什么？

参考回答：

```text
Harness 是把大模型包装成可运行 agent 的工程外壳。它不只是调用模型 API，而是管理上下文、工具、文件系统、终端执行、权限、安全、session 状态、trace、replay 和 evaluation。对于 coding agent，harness 决定模型能看到什么、能做什么、怎么改代码、怎么验证、出错后怎么恢复。
```

### 题 2：为什么 coding agent 不能只是一个聊天机器人？

参考回答：

```text
因为 coding agent 要在真实代码库里完成任务，需要搜索、读取、编辑、运行测试、处理错误、生成 diff、回滚修改、调用外部系统并保证安全。聊天机器人只输出文本，而 coding agent runtime 必须把模型输出转成受控动作，并记录整个过程。
```

### 题 3：如何设计 tool registry？

参考回答：

```text
Tool registry 要统一管理内置工具、custom tools、MCP tools 和 A2A remote agents。每个工具要有 namespace、schema、description、origin、risk level、permission policy、timeout、output limit、enabled agents 和 version。这样模型选择工具时有清晰描述，runtime 执行前可以检查权限，trace/replay 也知道调用了哪个版本的工具。
```

### 题 4：如何设计权限系统？

参考回答：

```text
权限系统输入用户、workspace、agent、tool、参数、路径、当前模式和组织策略，输出 allow、ask 或 deny。规则支持系统默认、组织、用户、项目、agent 和 session 临时批准多层覆盖。读普通文件可以 allow，写文件 ask，读 .env deny，外部目录 ask，git push、生产 deploy、数据库写入默认 deny。关键是权限在 runtime 层执行，而不是靠 prompt。
```

### 题 5：如何防止 agent 执行危险命令？

参考回答：

```text
我会从三层防护：第一是 permission pattern，对 bash 命令按前缀、参数和路径做 allow/ask/deny；第二是 sandbox，用容器或 VM 限制文件系统、网络、环境变量和资源；第三是用户确认和审计，对高风险命令给出自然语言解释、要求批准并记录 trace。命令还要设置 timeout，避免 hang 住。
```

### 题 6：如何处理大仓库上下文？

参考回答：

```text
不能把整个仓库塞进上下文。我会先构建 repo map，让模型看到关键文件、类、函数和符号骨架，再用 grep/glob/LSP 找相关文件，只读取关键片段。历史对话做 compaction，大工具输出做摘要或落盘引用，MCP tools 用 tool search 延迟加载。Context builder 要有 token budget，并保留用户目标、当前 plan、diff 和失败测试这些关键状态。
```

### 题 7：如何设计 trace 和 replay？

参考回答：

```text
Trace 记录每一步的用户输入、context/query、模型输出、tool call 输入输出、权限决策、文件 diff、shell exit code、MCP/A2A 调用、错误和 final answer。Replay 分两类：deterministic replay 用录制的模型和工具结果复现问题；re-evaluation replay 重新调用模型和工具比较新版本。对外部系统要记录版本、输出 hash，并在评估中尽量 mock 或 snapshot。
```

### 题 8：如何设计 evaluation harness？

参考回答：

```text
Evaluation harness 要固定 task set、repo snapshot、prompt、agent config、model config、tool version、测试命令和评分函数。指标不仅看 test pass，还要看 patch minimality、regression、安全违规、成本、延迟和人工 review。公共 benchmark 可以看通用能力，企业还需要私有历史 issue benchmark 来贴近真实业务分布。
```

### 题 9：MCP 如何接入 harness？

参考回答：

```text
MCP 接入时，harness 先从配置加载 local 或 remote MCP server，建立 MCP client，做 capability discovery，把 resources、prompts、tools 注册到 capability registry。每个 MCP tool 要绑定 namespace、权限、timeout、output limit 和 risk level。调用前经过 permission engine，调用后结果做截断、脱敏和 prompt injection 标记，并写入 trace。不能简单把 MCP 全量工具塞进上下文。
```

### 题 10：Claude Code、OpenCode、Codex 这类系统核心差异在哪里？

参考回答：

```text
核心差异不只是模型，而是 harness 和产品形态。Claude Code 更强调多入口、产品闭环、MCP、skills、hooks、subagents 和安全权限；OpenCode 更强调开源、config、server/SDK、permissions、agents、MCP、custom tools 和可扩展 runtime；Codex 代表 OpenAI 的本地和云端 coding agent 产品线，覆盖 CLI、IDE、desktop app 和 Codex Web。比较时要看上下文、工具、权限、执行、trace、评估和工作流，而不是只看 demo。
```

## 16.17 回答 Harness 问题的万能结构

面试中遇到任何 harness 问题，可以用这个结构回答：

```text
1. 先定义问题边界：产品形态、任务范围、安全要求。
2. 拆核心模块：session、context、model、tools、permission、execution、trace、eval。
3. 讲一次完整链路：用户输入 -> context -> model -> tool -> permission -> execution -> result -> trace。
4. 讲关键风险：上下文、权限、prompt injection、文件覆盖、命令副作用、成本。
5. 讲验证方式：test、diff review、trace、replay、benchmark、人工 review。
6. 讲 trade-off：自动化 vs 安全，上下文 vs 成本，通用性 vs 体验。
```

如果时间很短，可以压缩成一句：

```text
Coding agent harness 的本质是用工程 runtime 约束和增强模型，让模型在受控上下文、受控工具、受控权限、可观测 trace 和可评估环境中完成软件工程任务。
```

## 16.18 最后一张知识图

第二十册可以浓缩成这张图：

```text
User / Issue / IDE / CLI
-> Session Manager
-> Context Builder
   -> repo map / rules / history / diff / tool results
-> Model Adapter
-> Agent Orchestrator
-> Capability Registry
   -> built-in tools / custom tools / MCP / A2A / skills
-> Permission Engine
   -> allow / ask / deny
-> Execution Engine
   -> file edit / shell / sandbox / git / LSP / MCP / A2A
-> Trace Store
   -> model / tool / permission / diff / observation
-> Replay / Evaluation
   -> benchmark / private issues / regression / safety
```

只要能围绕这张图展开，绝大多数 agent harness 面试题都能答得有结构。

## 16.19 Harness 实战坑审计指标

为了把本章十类坑讲成工程闭环，可以把每个事故样本抽象成：

```math
h_i=(g_i,c_i,t_i,p_i,e_i,s_i,o_i,r_i,v_i,u_i)
```

其中 $g_i$ 是用户目标和验收标准，$c_i$ 是上下文与预算，$t_i$ 是工具和协议能力，$p_i$ 是权限决策，$e_i$ 是文件 / 终端 / sandbox 执行，$s_i$ 是 session 状态和 diff，$o_i$ 是 trace / log / artifact，$r_i$ 是 replay 条件，$v_i$ 是 eval / 版本条件，$u_i$ 是用户影响和事故分级。

常用排查指标可以写成：

```math
C_{\mathrm{triage}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\mathrm{triaged}(h_i)]
```

```math
C_{\mathrm{perm}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\mathrm{permission\_ok}(h_i)]
```

```math
C_{\mathrm{ctx}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[B_i\le L_i\land \mathrm{output\_limited}(h_i)]
```

```math
C_{\mathrm{edit}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\neg \mathrm{edit\_risk}(h_i)\lor \mathrm{edit\_checked}(h_i)]
```

```math
C_{\mathrm{cmd}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\neg \mathrm{command\_risk}(h_i)\lor \mathrm{command\_guarded}(h_i)]
```

```math
C_{\mathrm{inj}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\neg \mathrm{untrusted}(h_i)\lor \mathrm{boundary\_ok}(h_i)]
```

```math
C_{\mathrm{trace}}=\frac{1}{N}\sum_{i=1}^{N}\frac{|F_i\cap F_{\mathrm{req}}|}{|F_{\mathrm{req}}|}
```

```math
C_{\mathrm{replay}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\mathrm{replay\_ready}(h_i)\land \mathrm{version\_captured}(h_i)]
```

```math
C_{\mathrm{eval}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\mathrm{eval\_deterministic}(h_i)\land \mathrm{version\_captured}(h_i)]
```

```math
C_{\mathrm{cost}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[S_i\le S_{\max}\land M_i\le M_{\max}\land \neg \mathrm{repeat}(h_i)]
```

```math
C_{\mathrm{env}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\mathrm{env\_fingerprint}(h_i)\land \mathrm{cwd\_recorded}(h_i)]
```

最终实战坑门禁可以定义为：

```math
G_{\mathrm{pitfall}}=\mathbb{1}[
C_{\mathrm{triage}}\ge\tau_{\mathrm{triage}}
\land C_{\mathrm{perm}}\ge\tau_{\mathrm{perm}}
\land C_{\mathrm{ctx}}\ge\tau_{\mathrm{ctx}}
\land C_{\mathrm{edit}}\ge\tau_{\mathrm{edit}}
\land C_{\mathrm{cmd}}\ge\tau_{\mathrm{cmd}}
\land C_{\mathrm{inj}}\ge\tau_{\mathrm{inj}}
\land C_{\mathrm{trace}}\ge\tau_{\mathrm{trace}}
\land C_{\mathrm{replay}}\ge\tau_{\mathrm{replay}}
\land C_{\mathrm{eval}}\ge\tau_{\mathrm{eval}}
\land C_{\mathrm{cost}}\ge\tau_{\mathrm{cost}}
\land C_{\mathrm{env}}\ge\tau_{\mathrm{env}}
]
```

这些指标的价值不是替代人工复盘，而是让排查顺序更稳定：先确认事故是否被正确归因，再看权限、上下文、编辑、终端、外部内容、trace、replay、eval 和环境是否形成闭环。

### 16.19.1 最小可运行 Harness 实战坑审计 demo

下面的 demo 不调用模型、不执行命令、不访问文件系统或网络，只审计一组 toy incident。它故意构造 permission overreach、context overload、用户改动覆盖、shell hang、外部内容边界缺失、trace 缺失、eval flaky、doom loop、MCP tool overload 和环境漂移等 bad case，让 gate 失败并输出根因。

```python
incidents = [
    {
        "id": "permission_overreach",
        "root": "permission_overreach",
        "triaged": True,
        "expected_permission": "ask",
        "actual_permission": "allow",
        "high_risk": True,
        "approved": False,
        "context_tokens": 3600,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": False,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": False,
        "trace_fields": ["goal", "tool", "permission", "diff"],
        "replay_ready": False,
        "eval_deterministic": True,
        "version_captured": True,
        "steps": 4,
        "step_limit": 12,
        "cost": 0.08,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": True,
    },
    {
        "id": "context_overload",
        "root": "context_overload",
        "triaged": True,
        "expected_permission": "allow",
        "actual_permission": "allow",
        "high_risk": False,
        "approved": True,
        "context_tokens": 14500,
        "context_limit": 8000,
        "output_limited": False,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": False,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": True,
        "trace_fields": ["goal", "context", "tool", "error"],
        "replay_ready": False,
        "eval_deterministic": False,
        "version_captured": False,
        "steps": 9,
        "step_limit": 12,
        "cost": 0.41,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": True,
    },
    {
        "id": "user_edit_overwrite",
        "root": "edit_overwrite",
        "triaged": True,
        "expected_permission": "ask",
        "actual_permission": "ask",
        "high_risk": True,
        "approved": True,
        "context_tokens": 5200,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": True,
        "edit_checked": False,
        "command_risk": False,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": True,
        "trace_fields": ["goal", "diff", "permission", "error"],
        "replay_ready": True,
        "eval_deterministic": True,
        "version_captured": True,
        "steps": 5,
        "step_limit": 12,
        "cost": 0.11,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": True,
    },
    {
        "id": "shell_hang",
        "root": "shell_hang",
        "triaged": True,
        "expected_permission": "ask",
        "actual_permission": "ask",
        "high_risk": True,
        "approved": True,
        "context_tokens": 4200,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": True,
        "command_safe": False,
        "timeout": False,
        "cwd_recorded": False,
        "env_masked": False,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": True,
        "trace_fields": ["goal", "tool", "permission"],
        "replay_ready": False,
        "eval_deterministic": False,
        "version_captured": False,
        "steps": 13,
        "step_limit": 12,
        "cost": 0.22,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": False,
    },
    {
        "id": "prompt_injection_from_issue",
        "root": "prompt_injection_boundary",
        "triaged": True,
        "expected_permission": "deny",
        "actual_permission": "ask",
        "high_risk": True,
        "approved": False,
        "context_tokens": 3900,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": True,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": True,
        "trust_labeled": False,
        "high_risk_blocked": False,
        "trace_fields": ["goal", "context", "tool", "permission", "error"],
        "replay_ready": True,
        "eval_deterministic": True,
        "version_captured": True,
        "steps": 6,
        "step_limit": 12,
        "cost": 0.16,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": True,
    },
    {
        "id": "trace_missing",
        "root": "trace_missing",
        "triaged": False,
        "expected_permission": "allow",
        "actual_permission": "allow",
        "high_risk": False,
        "approved": True,
        "context_tokens": 4700,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": False,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": True,
        "trace_fields": ["goal"],
        "replay_ready": False,
        "eval_deterministic": False,
        "version_captured": False,
        "steps": 3,
        "step_limit": 12,
        "cost": 0.05,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": True,
    },
    {
        "id": "eval_flaky",
        "root": "eval_flakiness",
        "triaged": True,
        "expected_permission": "allow",
        "actual_permission": "allow",
        "high_risk": False,
        "approved": True,
        "context_tokens": 5100,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": False,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": True,
        "trace_fields": ["goal", "tool", "trace", "error"],
        "replay_ready": False,
        "eval_deterministic": False,
        "version_captured": False,
        "steps": 5,
        "step_limit": 12,
        "cost": 0.12,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": False,
    },
    {
        "id": "doom_loop",
        "root": "doom_loop",
        "triaged": True,
        "expected_permission": "allow",
        "actual_permission": "allow",
        "high_risk": False,
        "approved": True,
        "context_tokens": 7800,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": False,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": True,
        "trace_fields": ["goal", "context", "tool", "trace", "error"],
        "replay_ready": True,
        "eval_deterministic": True,
        "version_captured": True,
        "steps": 18,
        "step_limit": 12,
        "cost": 0.52,
        "cost_limit": 0.30,
        "repeated_call": True,
        "env_fingerprint": True,
    },
    {
        "id": "mcp_tool_overload",
        "root": "mcp_tool_overload",
        "triaged": True,
        "expected_permission": "ask",
        "actual_permission": "allow",
        "high_risk": True,
        "approved": False,
        "context_tokens": 9800,
        "context_limit": 8000,
        "output_limited": False,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": False,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": True,
        "env_masked": True,
        "untrusted": True,
        "trust_labeled": True,
        "high_risk_blocked": False,
        "trace_fields": ["goal", "context", "tool", "permission", "trace"],
        "replay_ready": False,
        "eval_deterministic": True,
        "version_captured": False,
        "steps": 10,
        "step_limit": 12,
        "cost": 0.36,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": True,
    },
    {
        "id": "env_drift",
        "root": "environment_drift",
        "triaged": True,
        "expected_permission": "allow",
        "actual_permission": "allow",
        "high_risk": False,
        "approved": True,
        "context_tokens": 4100,
        "context_limit": 8000,
        "output_limited": True,
        "edit_risk": False,
        "edit_checked": True,
        "command_risk": True,
        "command_safe": True,
        "timeout": True,
        "cwd_recorded": False,
        "env_masked": True,
        "untrusted": False,
        "trust_labeled": True,
        "high_risk_blocked": True,
        "trace_fields": ["goal", "tool", "permission", "trace", "version"],
        "replay_ready": False,
        "eval_deterministic": True,
        "version_captured": False,
        "steps": 7,
        "step_limit": 12,
        "cost": 0.15,
        "cost_limit": 0.30,
        "repeated_call": False,
        "env_fingerprint": False,
    },
]

required_trace = {"goal", "context", "tool", "permission", "diff", "trace", "error", "version"}


def avg(values):
    return round(sum(values) / len(values), 3)


def trace_score(case):
    return len(required_trace & set(case["trace_fields"])) / len(required_trace)


metrics = {
    "triage_coverage": avg([case["triaged"] for case in incidents]),
    "permission_safety": avg([
        case["actual_permission"] == case["expected_permission"]
        or (case["expected_permission"] == "deny" and case["actual_permission"] == "ask" and not case["approved"])
        for case in incidents
    ]),
    "context_budget_pass": avg([
        case["context_tokens"] <= case["context_limit"] and case["output_limited"]
        for case in incidents
    ]),
    "edit_safety": avg([not case["edit_risk"] or case["edit_checked"] for case in incidents]),
    "command_safety": avg([
        not case["command_risk"]
        or (case["command_safe"] and case["timeout"] and case["cwd_recorded"] and case["env_masked"])
        for case in incidents
    ]),
    "prompt_injection_boundary": avg([
        not case["untrusted"] or (case["trust_labeled"] and case["high_risk_blocked"])
        for case in incidents
    ]),
    "trace_completeness": avg([trace_score(case) for case in incidents]),
    "replay_readiness": avg([case["replay_ready"] and case["version_captured"] for case in incidents]),
    "eval_determinism": avg([case["eval_deterministic"] and case["version_captured"] for case in incidents]),
    "cost_loop_control": avg([
        case["steps"] <= case["step_limit"]
        and case["cost"] <= case["cost_limit"]
        and not case["repeated_call"]
        for case in incidents
    ]),
    "environment_parity": avg([case["env_fingerprint"] and case["cwd_recorded"] for case in incidents]),
}

thresholds = {
    "triage_coverage": 0.95,
    "permission_safety": 0.95,
    "context_budget_pass": 0.90,
    "edit_safety": 0.95,
    "command_safety": 0.95,
    "prompt_injection_boundary": 0.95,
    "trace_completeness": 0.85,
    "replay_readiness": 0.80,
    "eval_determinism": 0.80,
    "cost_loop_control": 0.90,
    "environment_parity": 0.90,
}

failed_gates = [name for name, limit in thresholds.items() if metrics[name] < limit]
root_causes = {}
for case in incidents:
    root_causes.setdefault(case["root"], []).append(case["id"])

print(f"metrics={metrics}")
print(f"failed_gates={failed_gates}")
print(f"root_causes={root_causes}")
print(f"harness_pitfall_gate_pass={not failed_gates}")
```

运行结果应类似：

```text
metrics={'triage_coverage': 0.9, 'permission_safety': 0.8, 'context_budget_pass': 0.8, 'edit_safety': 0.9, 'command_safety': 0.8, 'prompt_injection_boundary': 0.8, 'trace_completeness': 0.5, 'replay_readiness': 0.3, 'eval_determinism': 0.4, 'cost_loop_control': 0.6, 'environment_parity': 0.7}
failed_gates=['triage_coverage', 'permission_safety', 'context_budget_pass', 'edit_safety', 'command_safety', 'prompt_injection_boundary', 'trace_completeness', 'replay_readiness', 'eval_determinism', 'cost_loop_control', 'environment_parity']
root_causes={'permission_overreach': ['permission_overreach'], 'context_overload': ['context_overload'], 'edit_overwrite': ['user_edit_overwrite'], 'shell_hang': ['shell_hang'], 'prompt_injection_boundary': ['prompt_injection_from_issue'], 'trace_missing': ['trace_missing'], 'eval_flakiness': ['eval_flaky'], 'doom_loop': ['doom_loop'], 'mcp_tool_overload': ['mcp_tool_overload'], 'environment_drift': ['env_drift']}
harness_pitfall_gate_pass=False
```

面试中可以这样解释这个 demo：它不是为了证明某个真实产品不合格，而是展示 production harness 的事故复盘应该有一张统一表，把权限、上下文、编辑、终端、外部内容、trace、replay、eval、预算和环境问题放在同一个 gate 里。只看最终 diff 或最终回答，无法稳定定位这些根因。

## 16.20 小练习

1. 拿一个你使用过的 coding agent，按本章八层排查框架分析它的一次失败案例。
2. 设计一个 permission policy，要求 review agent 只读，build agent 可编辑和测试，release agent 可创建 PR 但不能 deploy。
3. 设计一个 trace viewer 页面，显示模型调用、工具调用、权限请求、diff 和测试结果。
4. 构造一个 MCP prompt injection 示例，并写出 harness 防护策略。
5. 设计一个私有 evaluation task，从历史 bug issue 到 repo snapshot、测试命令和评分函数。
6. 对比“本地 CLI agent”和“云端异步 agent”的安全威胁模型。
7. 用 3 分钟口述“设计一个 coding agent harness”的面试答案，并录音复盘是否覆盖了核心模块。

## 16.21 本章总结

本章整理了 Agent Harness 的实战坑和面试题。

核心结论：

1. 真实 coding agent 问题往往不只是模型问题，而是上下文、工具、权限、执行、状态和观测问题。
2. 排查时要按 user goal、context、model、tool、permission、execution、state、observability 八层定位。
3. 工具权限过大、大输出挤爆上下文、文件覆盖、命令副作用、prompt injection、trace 不完整、eval 不可比、agent 循环、MCP 过载、本地云端不一致，是最常见工程坑。
4. 面试回答要有结构：定义边界、拆模块、讲链路、讲风险、讲验证、讲 trade-off。
5. 一个优秀的 harness 不是让模型“想做什么就做什么”，而是让模型在可控、可审计、可恢复、可评估的工程系统里完成任务。

到这里，第二十册《Agent Harness、Coding Agent Runtime 与智能体工程框架》正文第一版完成。后续可以进入第二十一册，继续学习大模型系统架构演进与前沿工程方向。
