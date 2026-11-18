# 第七章《上下文管理与记忆压缩》精修审计

审计日期：2026-11-18。状态：完成章级扫描、正文精修、配套同步和静态文档检查；第二十册第 8—24 章待逐章处理。

## Reader path 与章节结果

- **读者前置：** 理解 LLM 的基本 prompt/生成和工具往返，能读文件路径、测试回执和简单 YAML；无需熟悉某一厂商的 memory API。
- **书册贡献：** 承接第四至第六章的工具、文件和终端事实，说明这些事实如何按任务、来源、版本、信任和 token 预算进入下一轮模型输入；补充分层任务状态、会话历史、工作区、trace 与 memory 的边界。
- **中心问题：** 下一轮模型需要哪些信息，怎样在预算内保留必要内容、来源和时效，同时避免旧摘要或外部内容误导动作？
- **预期结果：** 读者能沿一条 Coding Agent 任务选择上下文，压缩带截断标记的测试输出，辨认过期摘要，设计带 provenance 的 task state 和 context item，并按定义清楚的分母审查预算、覆盖、保真和信任边界。
- **主要任务与讲解路径：** 系统/工程机制解释。用明确标为合成的显示名修复任务串起用户约束、当前文件、相邻测试、终端失败回执、截断 artifact、当前 diff、旧版本摘要和 token 预算。章内 4/5 测试回执承接第六章的同类教学构造，但不声称是同一真实运行。

## 章级扫描与候选去向

扫描日期：2026-11-18。扫描问题：如何将会话/任务事实转换成本轮模型输入；如何压缩日志和历史而保留可回查证据；状态、记忆和 workspace 如何分工；信任标签能否被误当成安全控制；Anthropic 与 OpenAI 的产品合同是否具有不同状态所有者。资料类型：Anthropic Messages API compaction 文档、OpenAI Agents SDK Sessions 与 Agents API 文档、OWASP LLM01:2025、LongHarness Bench 预印本。

| 候选主题 | 证据与适配性 | 去向与判断 |
| --- | --- | --- |
| 上下文来源、优先级和预算 | 通用 Harness 机制；产品合同显示 session history 与本轮 model input 可以分开选择 | 保留 7.3—7.5、7.13；补充 authority、freshness、provenance 和预算边界，不设固定百分比配额 |
| 文件、搜索、终端输出和对话历史 | 与前章文件/执行回执衔接；短摘要若缺路径、版本、exit code、失败目标或截断标记会失去可恢复性 | 保留 7.6—7.9，以单一合成任务演示 snippet 回读、日志摘要与 artifact 引用 |
| Task State、Memory、过期和来源失真 | 任务状态、会话记录、跨任务记忆和持久 workspace 并非同一存储 | 保留 7.10—7.12，改为按作用范围、来源、版本、保留和删除职责讲解 |
| Prompt injection 与外部内容信任边界 | OWASP 建议分离和标记外部内容，但没有单一保证防住 prompt injection 的方法 | 保留 7.15；明确分区标记是缓解措施，权限/隔离仍在模型外执行并交第八章 |
| 上下文质量指标、旧复合门槛和 Python demo | 原稿有多项口径不一的比例、固定 alpha/beta 及约 220 行 toy auditor；程序把标注假设和任意阈值编码成统一 `G_context` | 保留预算、关键项覆盖、精确率、约束保留、摘要支持和 stale 等诊断量；删除聚合 gate 与脚本，新增一条逐步手算轨迹 |
| Anthropic Sonnet/Opus compaction 产品细节 | Anthropic 文档区分 on-demand 与 threshold compaction，thinking block 回放还受模型和前缀条件影响；支持平台和型号表会变化 | 原稿 7.24—7.25 的详细型号/平台说明与第二十四章重复且时效性强；在 7.18.3 留下简短的 API 合同对照并链接第二十四章，完整证据及版本边界记入 research |

没有新增独立章节或改变本章承诺；目录摘要仍准确，因此未改目录或简介。

## 覆盖图与原稿主题去向

| 原稿主题 | 本轮处理与正文位置 | 去向理由 |
| --- | --- | --- |
| Context Builder、来源优先级、token budget | 7.1—7.5 保留并澄清 authority、trust、relevance、freshness 不是同一排序量；删除 15/10/45 等固定比例示例 | 给读者解释每轮 input 怎样由 runtime 形成，避免把百分比记成配额 |
| 文件、搜索、终端和历史摘要 | 7.6—7.9 保留；统一显示名任务样例和 artifact/source 引用 | 让读者能判断 snippet 与完整文件、日志摘要与原始 trace 的证据差异 |
| Task State、Memory、stale summary | 7.10—7.12 重组；增加带 revision 的 YAML 概念片段和会话/项目/用户作用范围 | 澄清任务约束不自动成为长期记忆，session ID 也不等于授权 |
| Context Builder 数据模型、裁剪与注入 | 7.13—7.15 保留并强化版本/provenance 和“不可信标记不等于执行防护”边界；关联第八章 | 继续服务于下一轮决策，并把真正执行控制交回 runtime permission/sandbox |
| 质量评估公式与 Python audit demo | 7.16—7.18 重组为独立诊断指标和显示名任务手算；删除 Python demo、任意阈值和 `G_context` | 保留评估读者结果，避免把主观标注和局部目标包装成通用质量分 |
| Context Folding / Persistent Workspace | 移至 7.18.2，保留目标、约束、workspace revision、验证结果和 artifact ref 的可恢复状态示例 | 与通用 memory 定义分开，强调摘要不是唯一事实源 |
| Anthropic compaction 两个产品小节 | 合并为 7.18.3 的短版 API 对照；详细 model-specific thinking 状态由第二十四章承接 | 不在通用机制章维护重复且易变的厂商平台矩阵 |
| 面试题与练习 | 保留 7.19—7.23；同步 `INTERVIEW_BANK.md` 40I 与 `EXERCISES.md` 35K | 让共享材料移除旧 `context builder gate` 和程序实现要求，改为事件轨迹与分母审查 |

## Evidence、示例与呈现状态

- **evidence：** 2026-11-18 打开 Anthropic [on-demand compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)、[threshold compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)、[preserved thinking](https://platform.claude.com/docs/en/build-with-claude/compaction-thinking-blocks)；OpenAI [Agents SDK Sessions](https://openai.github.io/openai-agents-python/sessions/) 与 [Agents API overview](https://developers.openai.com/api/docs/guides/agents-api/overview)；[OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)；[LongHarness Bench](https://arxiv.org/abs/2609.38137)。来源事实、日期、平台/model 边界及预印本证据范围记入[长上下文 Harness 与 Agent Memory 来源记录](../../research/long-context-harness-and-agent-memory-2026.md)。
- **example_status：** 7.2、7.8、7.10、7.13 和 7.18 共用显示名任务。4/5 测试回执、rev-19、1,200-token 预算和各项 token 估算均为教学构造；7.18 手算预算利用率、关键项召回、上下文精确率、约束保留和 stale 率，并明确不是生产结果或通用阈值。
- **code_status：** 无可执行代码。保留的文本/YAML 均是解释性片段，不对应产品 API schema；未运行测试、模型或 API，也未声称合成回执来自真实运行。
- **formula_status：** 保留 5 个 MathJax 指标块，定义 `U_tok`、`R_key`、`P_ctx`、`C_cons`、`F_sum` 和 `R_stale`；显示名案例手算分子、分母和 N/A 边界。删除原固定阈值及聚合 `G_context`。
- **visual_status：** 未增加 Mermaid 或图片；以数据来源表和预算审阅表展示分类、选择和前后变化。未做 GitHub MathJax 页面渲染核验。
- **link_status：** 本章引用 7 个唯一外部页面，并含第六、八、十七、十八、二十四章 5 个本地链接；章节及配套资料的本地目标在静态检查中缺失 0。未做全站外链可达性扫描。
- **sync_status：** 将 `EXERCISES.md` 35K 从 Python auditor 改为合成上下文审阅与手算；将 `INTERVIEW_BANK.md` 40I 改为独立指标、来源和分母说明。简介与目录摘要仍符合本稿范围，故未改；未改 `plan.md` 或 PDF。

## 遗留风险与下一入口

- 没有目标读者试读；公式页面渲染及全站外链可达性未核验。
- Anthropic compaction 的 beta header、trigger、model/platform 支持会变化；正文只保留截至 2026-11-18 读取的公开合同，后续使用时须重新核对目标平台。模型特定 thinking block 细节回到第二十四章。
- 本章的 token 数、测试回执和上下文候选完全是教学构造；不能据此声称仓库运行过相关命令或系统达到任何生产阈值。
- 下一入口为[第八章《权限模型与安全沙箱》](../../book-20-agent-harness-runtime/chapters/08-权限模型与安全沙箱.md)，承接本章强调的 trust label 与独立授权边界，继续讲清 permission、approval、OS sandbox、文件/网络/secret 和执行限制各自负责什么。
