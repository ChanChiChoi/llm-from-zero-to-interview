# 第八章审计：Safety 与 Robustness Eval

审阅日期：2026-11-03  
状态：章级扫描、正文精修、配套同步与静态门禁完成；第七册及全套书精修仍在进行中。

## scope

- 完整审阅 `book-07-evaluation-experiments/chapters/08-safety与robustness-eval.md`。保留单篇核心教学单元，主线仍为 policy/risk contract → taxonomy 和分母 → 各类风险评测 → 统计与 judge → 线上闭环 → 错误分析及报告；未增章、拆分、重排目录。
- 本章承担安全/鲁棒性评测的构念、样本、协议、指标、判定和证据解释。第八册分别主讲 jailbreak/prompt injection 机制、red teaming/危险能力治理、privacy/memorization/watermarking、自动化对抗评估和不可信 Agent 运行时监督；本章新增对应交叉链接。
- 2026-11-03 来源扫描聚焦三个盲点：实验室与真实应用任务差异、长文档上下文中的注入防护、Agent 工具工作流 benchmark 的覆盖完整性；同时复核 InfoRMIA 与 2025 年间接注入预印本的版本和证据边界。研究事实仅见 `research/safety-and-prompt-injection-evaluation-2026-11-03.md`。

## chapter_scan 与候选去向

| 候选主题 | 去向 | 决定与理由 |
| --- | --- | --- |
| Rethinking Assessments of Prompt Injection Attacks | §6.5 对照表和解释段 | 纳入。补实验室/真实应用任务、简单/复杂攻击条件的比较。明确论文未测工具增强系统，结果不解释为线上发生率。 |
| LongPIBench | §6.5 对照表和解释段 | 纳入。补论文/简历/代码/邮件四种长文档任务与数千至数万 token 的输入；明确其静态单次调用边界，并链接本册第七章通用长上下文评测。 |
| SafeAudit | §6.5 对照表和解释段 | 保留并收窄“超过 20%”结论：这是论文 3 个 benchmark、12 个环境和特定枚举器/规则下的残余不安全样本，不是生产风险率或通用漏测率。 |
| Indirect Prompt Injections: Are Firewalls All You Need, or Stronger Benchmarks? | §6.5 与 §20 | 保留为预印本观察，校正标题及版本日期：2025-10-06 v1，2026-03-23 v2。将 benchmark 成功指标、实现和攻击强度的批评限定为作者对四个公开 benchmark 的审计。 |
| InfoRMIA | §7.5、§20 | 保留 token-level 记忆定位主张，改为与论文摘要/引言一致的表述；不据此推断应用侧隐私暴露。完整 privacy 机制交第八册第十章。 |
| 增设独立“长上下文注入”或“benchmark 元审计”章节 | 不新增 | 候选是现有 prompt injection / Agent 工具安全评测中的协议轴，纳入 §6.5 即可；没有形成足以改变本章主线的独立教学链。 |

## 正文修改与手算

- §3.1 明确 unsafe 是综合结果标签，危险输出、泄露、越权工具动作和最终外部状态违规均可计入；判定规则应预先固定。
- §6.5 增补真实应用任务、长文档和 benchmark coverage meta-audit 三类证据，解释它们分别改变任务真实性、上下文条件和测试覆盖判断，不合并成一个安全总分。
- §7.5 将 InfoRMIA 描述收窄为论文提出的 token-level 记忆信号定位框架；更完整的隐私和记忆机制链接到第八册第十章。
- §16 修正 toy demo 的 unsafe compliance 计数。7 个 policy 为 deny 的样本中，harmful_002、jailbreak_002、indirect_001 和 tool_001 共 4 个产生危险输出、泄露或越权工具动作，因此 4/7≈0.571；此前只按 action 文本值为 unsafe 计为 2/7≈0.286，漏掉了泄露和工具动作。
- 手算核对其他 demo 输出保持不变：attack success 3/4=0.75；over-refusal 1/3≈0.333；safe completion 3/4=0.75；secret leakage 1/2=0.5；unauthorized tool 1/2=0.5；robustness drop 0.275；一致率 1/4=0.25；严重度加权风险 19/37≈0.514。高风险组 4/8=0.5、普通组 2/3≈0.667；risk_event IDs、signals、actions 和最终流程文字结果不变。
- 将 MIA 公式的 operatorname 改为规范要求的 mathrm。章内不再含 operatorname。

## length_status

- 单篇连续核心讲解单元：解释性中文静态计数约 12,227 字，高于 1,800 字下限。统计跳过标题、表格、代码/公式/文本围栏、练习、资料清单和状态标记；篇幅不替代教学质量判断。

## evidence

- 新核验事实、阅读版本/范围和限制见 `research/safety-and-prompt-injection-evaluation-2026-11-03.md`：SafeAudit、Rethinking Assessments、LongPIBench、间接注入防火墙预印本 v2、InfoRMIA。
- 候选分别保留其场景、模型、工具/非工具设置、静态/动态上下文和测试预算边界。未将论文指标外推为生产攻击概率或通用防御保证。

## example_status

- §16 合成 demo 的输入样本、unsafe/attack/utility/privacy/tool 分母、robustness pair、group score 和 severity 公式均逐项手算；只有 unsafe compliance 指标与输出值更正。示例不代表模型或产品实测。

## code_status

- 1 个 Python 教学代码块。可执行语句已修改以纳入危险输出、leak 和 bad_tool；依据用户既定约束，本轮未执行代码或测试。代码旁标记“静态审阅，未运行（2026-11-03）”，并按手算更新预期输出。未更改其他代码块（本章只有此块）。

## formula_status

- 60 个围栏标记配成 30 对：19 个 math、10 个 text、1 个 Python。math 块花括号平衡；行间公式均使用 math 围栏，行外美元标记数为 0；无 operatorname。未做 GitHub MathJax 页面渲染验收。

## link_status

- 61 个 Markdown 链接；7 个本地相对链接均存在，54 个外链未逐项全量复核。新加的 SafeAudit、Rethinking Assessments、LongPIBench 及 ICLR InfoRMIA 来源已核对原论文页面/PDF；本轮新增的第八册及本册第七章交叉链接静态核验存在。
- 无 Mermaid 图；现有表格、公式和文本工作流承载本章评测关系，无需为增加图数而添加装饰图。正文无 progress 路径或维护工作流残留。

## sync_status

- 更新 `PAPERS.md`：加入 Rethinking Assessments、LongPIBench，修正间接注入预印本标题/版本，并补充 SafeAudit 边界。
- 更新 `EXERCISES.md`、`INTERVIEW_BANK.md` 和 `KNOWLEDGE_GRAPH.md`：覆盖真实应用、长文档、状态化工具工作流和 benchmark meta-audit 的互补评测轴；将面试题引用改到现存研究记录。
- 更新 `GLOSSARY_EN_ZH.md`：新增 Benchmark Meta-audit 术语及其与系统安全/生产风险率的边界。
- 简介和目录中的第八章摘要仍准确，未改；未改变章节结构。

## 门禁、遗留风险与交接

- 静态链接与公式/围栏检查通过；`git diff --check` 通过。未运行代码、测试、模型、benchmark，未构建或更新 PDF。
- 旧外链未全量在线复核；三个新评测论文及 InfoRMIA 已核验原始来源，预印本结论保留独立复核边界。
- 下一步按序处理第九章 `book-07-evaluation-experiments/chapters/09-在线实验与ab-test.md`，审查线上护栏、分层、停止规则和安全伤害指标；第 9—15 章及全册连续复读/最终门禁未完成。

## 整册连续复读复核（2026-11-03）

- 沿目录阅读路径完整复读本章；第 7 章多模态、长上下文与 grounding 协议自然交给安全和鲁棒性任务，第 9 章的线上实验及第 10 章的统计推断承担更完整的实验方法。与第八册机制、治理和运行时监督的边界仍清楚。
- 静态复算 §16 demo：deny 分母 7、unsafe 事件 4；攻击成功 3/4；正常 allow 误拒 1/3；安全完成 3/4；泄露和越权工具调用各 1/2；扰动分数下降 0.275、一致率 1/4；最差群体 0.5、严重度加权风险 19/37≈0.514。预期动作与这些信号一致，toy 输入不代表产品实测。
- 连续复读发现代码状态标为未运行，而代码块后的“示例输出为”没有明确指出输出是静态预期；现已改为“根据合成输入静态手算的预期输出”，并注明本轮未执行 Python。未改可执行代码。
- 本章与前后章节衔接无结构问题；第七册连续复读当前已完成第 1—8 章，下一步按序复读第 9 章。
