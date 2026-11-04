# 第六章审计：Reasoning、Code、Math Eval

审阅日期：2026-11-03  
状态：章级扫描、正文精修、配套同步与静态门禁完成；第七册和全套修书仍在进行中。

## 范围与章节职责

- 完整审阅 `book-07-evaluation-experiments/chapters/06-reasoning-code-math-eval.md`。本章把 reasoning、math 与 code 的任务契约、结果/过程证据、答案抽取、测试与 verifier、变体迁移和错误定位连成一个评测方法链；不重讲 Chapter 2 的 benchmark 全流程、Chapter 5 的污染检测、Chapter 10 的统计推断，也不承接 Book 17/20 的 Agent/harness 实现。
- 全章按一篇连续核心单元计数：任务契约与答案基础 → 数学答案/过程评估 → 代码测试和沙箱 → `pass@k` 与 self-consistency → verifier/过程评估 → benchmark 生命周期、报告和诊断。未新增章节、拆分、重排或更改目录入口。
- 第 12 节 demo 是上述评测链的 toy 诊断流程，不作为生产 evaluator；Python 代码未运行。

## 章级扫描与候选去向

扫描日期：2026-11-03；时间窗：2024-11 至 2026-11；资料类型：数学过程评测论文、CoT 忠实性论文、仓库级代码 benchmark 原始论文、benchmark 使用方审计、维护方版本论文和官方发布页。来源的版本、阅读范围和证据限制见 `research/reasoning-code-math-eval-depth-review-2026-11-03.md`。

| 候选方向 | 证据与去向 | 决定与边界 |
| --- | --- | --- |
| 可见 CoT 是否忠实代表内部推理 | EMNLP 2025 FUR、NeurIPS 2025 thinking-draft 反事实忠实性研究 | 纳入 §8.2 与参考资料。区分过程标签、参数忠实性和反事实影响；这些都是指定干预/任务下的探针，不是内部推理的通用读取器。 |
| SWE-bench Verified / Pro 的版本生命周期与任务质量 | OpenAI 2026-02 对 Verified 的审计和 Pro 推荐、2026-07 Pro 审计并撤回推荐；Scale AI 2026-10 Pro V2 预印本报告任务修订和 verifier 加固 | 更新 §4.8、§9.6 与资料表，明确来源立场、版本时间和任务集合。Pro V2 的修订数字是维护方自报，不能当作独立外部验证。 |
| SWE-Bench Pro Verified 变体 | 2026-09 预印本提出另一套防答案泄漏和任务修订协议 | 在 §4.8 作为独立变体点到为止，并与 Scale 的 Pro V2 区分；其模型/环境结果不可迁移到其他版本。 |
| Harness/Agent 端到端质量、轨迹归因与更长工作流 benchmark | 本册 Chapter 12/15 已有完整系统/专业任务的责任入口；实现机制仍交 Book 17/20，相关候选证据已有 benchmark lifecycle / Agent trace research 记录 | 不在本章扩成另一套 Agent 基准目录；保留本章对固定 harness、环境、预算和 verifier 的必要条件说明。 |

## 正文修改与数值例子

- 更新 §4.8：增加 SWE-Bench Pro / Pro V2 路线；保留 Verified 的 issue、patch、`FAIL_TO_PASS`、`PASS_TO_PASS` 解释，并改写后续审计时间线，指出 OpenAI 在 2026 年 7 月撤回此前推荐；说明 Pro V2 和 Pro Verified 是不同变体。
- 更新 §9.6：把 Verified/Pro 审计和 Pro V2 修订作为版本维护实例，强调固定回归、滚动新题与 private holdout 的分母不能混合。
- 更新 §8.2：加入 FUR 与 thinking-draft 反事实干预作为可见推理轨迹忠实性探针；保留结论必须绑定模型、数据和干预方式的边界。
- Self-consistency 五路径例 `42,42,40,42,41` 的分布为 `(3/5, 1/5, 1/5)`，答案熵手算 `-(3/5)ln(3/5)-2(1/5)ln(1/5)≈0.950 nat`，top-1/top-2 margin 为 `0.4`；若 gold 为 41，众数仍然错误。补充并列票的预先处理规则。
- `pass@k` 例 `n=10,c=2`：`pass@1=0.2`，`pass@5=1-C(8,5)/C(10,5)≈0.778`。将公式中的禁用 `\\operatorname` 改为 `\\mathrm`。
- 静态复核诊断 demo：手算 greedy / self-consistency accuracy 均为 0.667，平均熵 0.708，平均投票 margin 0.4，平均过程分 0.667；代码 `pass@1` 均值 0.133、`pass@5` 均值 0.472，唯一隐藏覆盖缺口为 `parse_table`。连续复读发现原实现用容差判 gold 正确、却按精确分数归并 self-consistency 答案，导致 `66.7%` 未与 `2/3` 合并。现改为 gold 比较和投票共用三位小数归一化；相应熵和 margin 已静态重算。另保留 `0.3-0.1` 的二进制浮点边界说明，生产门禁应按整数计数或显式容差处理。
- 将 demo 标题中的“可运行”移除，紧邻代码块增加“代码状态：静态审阅，未运行（2026-11-03）”；补充 `extract_number` 按类型优先级抽取导致混合数字文本误抽取的限制。没有执行代码。
- §9.4 增加指向 Chapter 5 的相对链接；更新 Chapter 6 资料列表以包含 CoT faithfulness 和 SWE-bench 各版本的 primary sources。

## 门禁与配套同步

- `length_status`：静态计数 10,511 个汉字，按全篇一篇连续核心讲解单元计，高于 1,800 字下限；标题、表格、代码/文本/数学围栏、行内代码、链接不计。计数不能替代对因果链和教学完整度的判断。
- `example_status`：pass@k、self-consistency 熵与 margin，以及代码 demo 的关键分母、切片和后续动作均按源码与玩具输入手工复算；不是模型或 benchmark 实测。
- `code_status`：1 个 Python block。整册复读修改了 `equivalent`/`canonical` 的数值分组，使 self-consistency 投票和 gold 比较共用三位小数键；只做静态审阅和手算，未运行。正文状态标记仍为“静态审阅，未运行”。
- `formula_status`：58 个围栏标记配成 29 对（14 个 math、14 个 text、1 个 Python）；数学花括号平衡，未发现 `\operatorname`，正文无行内 `$` 数学公式。
- `link_status`：35 个 Markdown 链接，其中 1 个本地 Chapter 5 目标存在、34 个外链；新增 CoT 与 SWE-bench 来源已按 ACL Anthology、NeurIPS Proceedings、arXiv 与 OpenAI 官方页面核对，历史外链未逐项全量在线复核。
- `sync_status`：更新 `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`EXERCISES.md`、`KNOWLEDGE_GRAPH.md` 和 `INTERVIEW_BANK.md`；修复 PAPERS 中指向缺失研究文件的两个链接，改到现存的 benchmark lifecycle 与 Agent trace 研究资料。目录摘要仍准确，无需结构同步。
- 本章无 Mermaid 图。正文无 `progress/` 维护路径、作者指令或未完成任务残留。`git diff --check` 通过；未运行代码、测试、模型、benchmark 或 PDF 构建。

## 遗留风险与下一步

- Python demo 未执行；尽管静态核对了目标输入与输出，运行环境兼容性仍未由本轮实测确认。
- SWE-bench Pro 的缺陷比例来自 OpenAI 指定审计流程，Pro V2 数字来自维护方预印本；均不代表所有仓库 benchmark 的普遍错误率或独立验证结论。
- CoT 忠实性方法采用不同定义和干预，不能将分数直接合并或解释成统一的内部透明度。
- 第六章逐章审阅已完成；整册连续复读按序已核到本章，下一步复读第七章《多模态与长上下文评估》。第 1—15 章逐章审阅均已完成，但第七册连续复读和最终门禁未完成；全套书修书 goal 保持 active。

## 整册连续复读复核（2026-11-03）

- 第 6 章的答案、过程、变体迁移与 verifier 主线承接第 1—5 章任务/证据/污染方法，并按目录把多模态与长上下文具体任务交给第 7 章；代码 harness 和 Agent 执行细节继续由邻册承接。
- 修正 self-consistency 中数值答案分组与 gold 正确性口径不一致的问题：本 demo 对简单数值答案统一格式化为三位小数，`2/3`、`0.6666667` 和 `66.7%` 归并为 `0.667`。手算输出相应调整为平均熵 `0.708 nat`、平均 margin `0.4`；准确率、代码指标和动作不变。
- 既有公式和逐步案例的分母、测试覆盖和证据边界与章节说明一致；本轮代码变化保持“静态审阅，未运行”，没有执行程序。第七册连续复读继续进行。
