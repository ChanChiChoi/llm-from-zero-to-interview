# 第八册第五章 Jailbreak 与 Prompt Injection 正文精修记录

- `date`：2026-10-27。
- `scope`：依据当前 `WRITING_SPEC.md` 从章首完整复读 `book-08-ai-safety-alignment/chapters/05-jailbreak与prompt-injection.md`；复用旧[章级扫描](book-08-ch05-jailbreak-injection-scan-2026-10-27.md)中的候选线索，并补读官方 OWASP 分类和 WASP、StepJack、INFUSE、MemSecBench 原始论文页面。旧扫描不作为正文精修完成证据。
- `chapter_role`：承接第二章目标错配、第三章可扩展监督和第四章 reward hacking，解释模型行为层 jailbreak 与应用信任边界层 prompt injection 的失效路径。第六章负责红队协议及发布统计，第十三章负责自动化对抗评估，第十四章负责分类器，第十五章负责运行时授权，第十六章负责降级；第二十册第七章负责记忆管理工程。本章将跨会话记忆污染纳入威胁模型和生命周期评估，不重复记忆存储实现。
- `architecture_decision`：保留单篇教材，不新增章节。章节主线仍是信任边界 → Jailbreak / Prompt Injection 区分 → RAG 和 Agent 攻击面 → 多层防御 → 应用级评估 → 项目闭环；在 §6 增加跨会话记忆生命周期，在 §10 增加阶段条件指标。章首原纯文本信任流改为 Mermaid 防御闭环。

## 外部证据与边界

| 来源 | 实际阅读支持的判断 | 外推边界 |
| --- | --- | --- |
| OWASP [GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) 与 [Agentic Top 10 2026 ASI06](https://genai.owasp.org/download/52117/) | LLM Top 10 2026 保留 LLM01 Prompt Injection；Agentic Top 10 以 ASI06 讨论跨任务保留、检索和复用的 Memory & Context Poisoning，并把一次性输入提示指向其所引用的 LLM01:2025。用于区分一次请求攻击和持久状态污染。 | OWASP 是社区风险分类与控制指导，不是产品发生率或安全认证；引用时保留其版本语义，不能把 LLM01:2025 与 LLM01:2026 混为同一个版本标签。 |
| [WASP, NeurIPS 2025 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets_and_Benchmarks_Track.html) | 端到端 Web Agent benchmark 报告在特定测试案例中部分注入成功行为可达 86%，而攻击者最终目标完成率通常更低。支持分开报告中间行为、策略违规和已验证后果。 | 数值绑定该论文的任务、harness 与判分口径，“最多”不是总体发生率。 |
| [StepJack, arXiv:2608.06477 v1](https://arxiv.org/abs/2608.06477) | 480 个 CUA 样例把攻击目标分散到多页面、多步骤；固定深度下 6 个 CUA 中 3 个 ASR 上升，单模型最大差 31.2 个百分点；五个可靠跟随参考链的 CUA 均值从单步 31.3% 到三步 36.9%。支持把多步骤链加入测试矩阵。 | 只使用 12 个攻击目标、最多三步；API 成本导致完整评估只运行一轮。不能推断所有 Agent 都随步骤数增加而单调变脆弱。 |
| [INFUSE, arXiv:2610.20776 v1](https://arxiv.org/abs/2610.20776) | 40 个任务、56 个手工攻击场景、19 个模型、41,824 条对抗轨迹及 720 条 benign reference runs。全 benchmark 的 judge ASR-L 为 19.6%，heuristic ASR-H 为 15.4%；应成对报告不同判定器的结果。 | 全量网格以 Mini-SWE-Agent 为主；其它 harness 只覆盖已在主 harness 受攻击的子集。任务与 payload 为英文、攻击集静态；单一 judge 有误差，沙箱出口不等于现实副作用。数字不作生产风险率。 |
| [MemSecBench, arXiv:2607.27080 v1](https://arxiv.org/abs/2607.27080) | 310 个案例来自 48 个情境，跨 2 个 harness、4 个 memory backend、3 个 LLM backend 的 24 个配置，以 Write–Execute–Forget 和七个生命周期检查点审计。作者报告总案例恶意记忆持久化率 84.2%、完整 Write–Execute 链 50.3%；在成功投毒案例中完整 Execute 链 59.6%、选择性修复 56.1%。 | 结果绑定指定配置、isolated runtime、case rubric 和 verifier，属于 benchmark 描述性结果；不是生产损害率。各指标使用不同分母，正文已标明条件分母。 |

## 正文改动

- `trust_boundary`：将章首 text 箭头换成标准 Mermaid `flowchart TD`；节点标签加双引号，文字虚线统一写作 `-.->|标签|`。图解释了来源标记、服务端权限验证、确认、审计与修复闭环。
- `memory_lifecycle`：§6.4 新增一次性输入与跨会话持久记忆的边界，按写入、持久化、召回、采纳、副作用、选择性修复解释风险；给出邮件助手状态案例，并链接第二十册记忆工程主讲。
- `evaluation_evidence`：§10.3 加入 WASP、INFUSE、StepJack 结果，以各论文明示条件支撑“部分行为与已验证后果分开测”“多步骤要纳入评测”，同时陈列可比性和外推限制。
- `formula_status`：将原 `$$` 展示美元围栏和 `\(...\)` 行内写法统一为单美元行内公式或 `math` 围栏；将一处 `\operatorname` 改为 `\mathrm`。新记忆阶段指标显式定义 $W_i,R_i,A_i,X_i,F_i$，说明每个条件分母和 `N/A` 语义。
- `worked_example`：用 10 条合成记忆轨迹手算持久化 `7/10=0.70`、条件召回 `6/7≈0.857`、条件采纳 `4/6≈0.667`、采纳后副作用 `2/4=0.50`、端到端副作用 `2/10=0.20`、选择性修复 `5/7≈0.714`。示例数据明确不是 MemSecBench 复现。
- `demo_changes`：§16 让既有 `expected` 标签参与 `exact_action_mismatch=7/12=0.583` 的诊断并输出样本 ID，正文说明它只供人工复核，不是安全评分；增加 10 条记忆生命周期样例、分阶段聚合、链路不变量和空分母 N/A 断言。
- `references_and_sync`：正文新增 OWASP 2026、WASP、StepJack、INFUSE、MemSecBench 入口；`PAPERS.md` 新增 Prompt Injection 与记忆污染资料组，`GLOSSARY_EN_ZH.md` 增加 `Memory & Context Poisoning`，`EXERCISES.md` 新增条件分母与选择性修复演练。`INTERVIEW_BANK.md` 已有来源/taint/跨 Agent 传播和运行时门禁题，本轮复核后未重复添加。

## 验收状态

- `length_status`：按本篇单一核心讲解单元统计，排除标题、代码/图示/公式围栏、表格行、引用清单、练习和状态标记，机械统计 **13,418 个汉字**，高于 1,800 字门槛。字数不替代人工解释链复读。
- `code_status`：§16 唯一 Python 块的可执行语句本轮有改动。已用 Python 3.12.9 标准库运行文档中的 fenced code；固定 12 条安全 case、10 条记忆轨迹、阶段结果和空分母分支均通过内部断言。输出为 `persistence=0.7`、`recall_given_persistence=0.857`、`adoption_given_recall=0.667`、`side_effect_given_adoption=0.5`、`end_to_end_side_effect=0.2`、`selective_repair_given_persistence=0.714`、`exact_action_mismatch=0.583`、`empty_denominator_check=N/A preserved`，主决策为 `hold_high_risk_actions_and_retest`。命令从 Markdown 抽取并执行唯一 fenced Python block；未增加测试文件或运行测试套件。
- `formula_status`：当前 chapter5 不再保留 `$$` 或 `\(...\)`；复杂指标、signal 和授权条件使用 `math` 围栏，变量及分母均在正文定义。GitHub MathJax 页面渲染尚未确认。
- `link_status`：本地新增第二十册第七章相对链接目标存在；其余本地相对链接仍待全量解析。外链已逐个核对正文关键 OWASP / benchmark 来源；未执行所有外链的批量 HTTP 检查。
- `visual_status`：Mermaid 围栏和标签语法已静态复核；当前本地修改尚未发布到远端，GitHub 最终渲染仍待目标平台页面检查，故正文门禁未通过。
- `open_risks`：未运行真实模型、目标 memory backend、邮件系统、Agent harness 或生产工具；没有生产风险率证据。GitHub Mermaid/MathJax 页面渲染、逐链接检查及第八册顺序复读仍开放。
- `status`：本章正文已实质修改、教学代码已按声明范围运行、配套入口已同步；因目标平台渲染和全章连续复读仍未验，不能标记完整正文门禁通过。下一篇为第八册第六章。
