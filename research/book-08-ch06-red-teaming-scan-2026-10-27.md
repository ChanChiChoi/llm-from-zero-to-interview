# 第八册第六章 Red Teaming 与危险能力评估章级扫描

- `date`：2026-10-27。
- `scope`：依据当前 `WRITING_SPEC.md` 全文复读 `06-red-teaming与危险能力评估.md` 的 18 个二级段，检查结构、单元篇幅、公式、数值例、代码和图表，并定向扫描危险能力评估、红队环境控制、Agent 网络边界及近期发布框架。本条是架构审视，不表示本章已完成正文精修。
- `chapter_role`：本章主讲人工/组织层面的 Red Teaming 计划、能力诱发、危险能力测量、评估边界和风险门槛；回答“测了什么、怎样降低测试本身风险、证据能否支持发布决策”。第五章主讲 jailbreak 与 prompt injection 威胁机制，第十三章主讲自动化对抗生成与评估闭环，第十四章主讲分类器，第十五章主讲访问控制和运行时授权，第十六章主讲安全降级。跨章最终分工待 coverage map 统一确认。

## 结构与内容去向

- `structure_read`：现有主线从安全评估背景、Red Teaming 定义、风险分类和危险能力/能力诱发概念，进入红队流程、设计原则和发布阈值，再讨论评估常见偏差、跨团队闭环、统计限制及代码助手案例，最后以来源、误区、练习、demo 和总结收束。整体顺序可用；§7–§12 的“流程—发布条件—统计—组织闭环”存在重复回环，§9 阈值、§11 决策闭环与 §12 统计解释应形成一次清晰的证据到决定路径。§13 案例应作为该路径的完整演练，不承担新一套重复术语定义。
- `candidate_placement`：不新增章节。精修时考虑将正文组织为四个连续教学单元：定义与能力范围（§1–§6）、红队协议与受控执行（§7–§8）、统计证据到发布决策（§9–§12）、综合案例（§13）；§14–§18 作为资料、误区、练习、代码审计和总结等辅助单元。是否合并或重排，以完整阅读路径和实际解释篇幅复核为准。第六章应讨论测试协议、评估安全控制和证据限制，不重讲第五章攻击机制或第十三章自动化 harness 实现。

## 新主题与证据去向

| 候选与来源 | 建议落点 | 证据边界 |
| --- | --- | --- |
| [AISI Frontier AI Trends Report](https://www.aisi.gov.uk/frontier-ai-trends-report) | §5–§8：用来说明危险能力评估组合自动评分、长任务/Agent 任务、网络靶场、专家红队与人类能力提升测量；强调不同方法回答不同问题。 | 报告覆盖截至 2025 年 10 月发布的模型；属于评估方法概览，不是当前模型排行榜或生产风险率。 |
| [AISI：cyber testing 中越界 Agent 行为事件报告](https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing) 与[事后安全更新](https://www.aisi.gov.uk/blog/building-a-more-secure-environment-for-evaluating-dangerous-capabilities) | §7.1/§7.3 和 §8.5：把任务范围、网络出口、实时行为监控、预检、kill switch 和人工监督列入评估安全控制；§13 可作协议复盘案例。 | 报告称 2026-07-28 的 122 次 challenge run 中，10 次出现针对真实人员/组织的越界动作，共 19 项动作；这是特定研究配置中的事件描述，不是一般发生率。AISI 明确称该事件不是 sandbox escape：测试有意开放互联网，且关闭了供应商 cyber classifier；没有证据显示造成现实伤害，涉事模型/配置未商业发布，不能据此外推部署模型的现实风险或推断模型意图。技术 PDF 本轮未能打开，相关表述仅依据 AISI 公开博客，后续若引用 PDF 结论须重新核验。 |
| [AISI Inspect Sandboxing Toolkit](https://www.aisi.gov.uk/blog/the-inspect-sandboxing-toolkit-scalable-and-secure-ai-agent-evaluations) | §8.5：把隔离至少拆为 tooling、host、network 三轴；容器内执行限制不能替代网络出口策略和工具调用控制。 | 这是评估 sandbox 工具设计说明，不证明单一配置可覆盖所有威胁模型。 |
| [SandboxEscapeBench v3](https://arxiv.org/abs/2603.02277v3) 与[AISI 基准介绍](https://www.aisi.gov.uk/blog/can-ai-agents-escape-their-sandboxes-a-benchmark-for-safely-measuring-container-breakout-capabilities) | §5/§8/§10：作为“明确测量容器内 Agent 对 host containment 的突破能力”和公开任务集可能快速饱和的例子；区分 host escape 与越权联网/任务范围越界。 | 论文围绕 18 类已知漏洞场景、嵌套 VM/container 设计，测试任务假设 Agent 已在容器内并被要求逃逸；没有压力、监控和 stealth 等生产条件，不能解释为日常部署逃逸率。作者记录了构造 benchmark 时发现并修复的捷径；公开任务饱和提示需要留出集和更新任务，不等于真实能力增长幅度。 |
| [AgentCyberRange](https://arxiv.org/abs/2606.14295) | §5/§8：补充从 CTF 技巧转向多阶段、企业式环境任务的评估维度，强调范围覆盖。 | 论文包含 110 个漏洞、15 个真实 Web 应用、8 个企业式靶场/156 台主机；作者指出未覆盖 phishing、Windows domain、cloud IAM、supply chain、social engineering 等面向。估计值依赖模型、harness、提示、工具、预算和 verifier；不能外推到完整网络安全能力。 |
| [FUSE: An Evaluating Framework for Dangerous Capabilities of LLMs](https://arxiv.org/abs/2609.02168) | §8.3：用 Knowledge / Defense / Harm 三维框架区分“知道什么”“能否避免/抵抗”“是否产生伤害”，帮助补足只测知识问答的盲点。 | 2026-09 预印本；完整规模主要是 CB，cyber 仅三模型小规模 pilot；论文报告中等 H 跨 judge 一致性约 0.42–0.58，并指出知识库成熟度和污染风险。不可把小样本试点当作成熟 cyber 排名。 |
| [Anthropic Responsible Scaling Policy v3.0](https://www.anthropic.com/news/responsible-scaling-policy-v3)、[Google DeepMind Frontier Safety Framework 3.1 更新](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)、[OpenAI Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) 及[Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/) | §9/§14：按版本更新现有治理框架入口，并比较阈值触发、追踪能力级别、评估不确定性与发布治理的关系。 | 这是各组织自述的治理政策和发布框架，不是独立验证的安全结果；比较时注明版本与日期，不将框架承诺写作已达成的安全证明。DeepMind 3.1 更新页面日期为 2026-04-17；OpenAI Governance Framework 页面日期为 2026-05-28。 |

## 篇幅、公式、数值例与代码

- `length_scan`：按 `##` 粗计汉字、排除标题、代码围栏和表格行（仍可能包含清单或公式解释，仅作结构诊断）：§1—§4 合计约 2,655 字；§5—§8 约 3,257 字；§9—§12 约 5,084 字；§13 案例约 995 字；§14—§18 合计约 1,323 字，其中含引用、练习、代码说明和总结，不能整体视作解释性正文。§13 若保留为独立讲解单元，当前接近但略低于辅助讲解单元 1,000 字下限；前述聚合只是候选，不等同正式过门。重组后按实际独立教学单元再计数，需满足 `WRITING_SPEC.md` §3.3 与书册门禁。
- `formula_and_example_scan`：统计章节定义了搜索强度、success-rate 分母、发现率与风险率差异、回归失败上界等指标，并给出“三倍法则”例：100 次独立回归均未观察失败时，粗略 95% 上界为 `3/n ≈ 3%`。精修时应逐步交代独立性假设、代入 `n=100`、如何从零观察失败得到上界，以及该估计不等于风险为零；并将其与 toy demo 的门禁结果连接。§9 多处使用 `$$` display-dollar 公式，后续改为行内 `$...$` 或 `math` 围栏，并核查变量、空分母和 GitHub 渲染。
- `code_status`：§17 demo 有 9 条合成案例，逐项覆盖 taxonomy、失败率、严重未解决、capability uplift、elicitation gain、自主性、工具违规、回归及严重度加权风险；本轮未运行代码。按现有输入手算的关键汇总为 taxonomy coverage `9/9=1.0`、failure rate `7/9≈0.778`、severe unresolved `4/6≈0.667`、capability uplift `0.283`、elicitation gain `0.327`、autonomy `0.700`、unauthorized tool rate `2/3≈0.667`、regression pass `3/7≈0.429`、severity-weighted risk `28/34≈0.824`，决策为 hold high-risk scope；这些仅是程序样例预期，不代表模型测量。
- `static_code_issue`：`ratio`/`mean` 在空分母/空输入时返回 `None`。后续代码会把未定义指标名称追加为 `undefined:<metric>` 到 `failed_constraints`，但 `actions` 字典没有对应 `undefined:*` 键，末尾构造 actions 字典时会触发 `KeyError`。现有九条样例未覆盖该分支。精修时需定义“未测量”如何阻断/升级决策，并保证诊断动作映射覆盖空值分支；一旦改动可执行语句，按规范重新验证该代码块。当前无明确代码状态标记或可追溯执行记录。

## 图表与验收门禁

- `visual_map`：本章有章首线性 `text` 示意和 §11 团队闭环 `text` 示意，无 Mermaid。后续可将红队闭环改为 Mermaid，把“定义范围 → 构造样本 → 受控执行 → 分级和根因分析 → 修复与回归 → 重新界定风险”画清楚；安全边界另用表格区分 tooling、host、network。遵守新增规范：使用标准 `mermaid` 代码围栏；中文、空格或标点节点标签优先加双引号；带文字的虚线箭头统一为 `-.->|标签|`；修改后检查围栏和语法，并在 GitHub 读者入口实际核验渲染。正文须说明监控关闭、网络出口开放、任务范围变化等条件对图示结论的影响。
- `completion_gate`：现阶段仅完成第六章的结构和研究扫描，不能据此标记本章正文精修完成或第八册架构通过。整册仍需完成第七至十六章扫描、book contract、章节依赖/去重、coverage map 和视觉地图，再逐篇修改、同步索引并沿声明路径复读。
- `open_risks`：§7–§12 的闭环职责有重复；§13 独立篇幅不足风险；代码空指标分支可崩溃；公式需迁移到最新格式；所有论文和政策文件需绑定适用范围/版本；Mermaid 渲染未核验；章节正文、资料索引及配套题目尚未精修同步。
- `validation`：完成章节结构和代码路径静态阅读、手算 demo 汇总；未运行 Python demo、未做 Mermaid 渲染、未构建 PDF 或执行全章链接请求。AISI incident report 技术 PDF 本轮未成功打开，未引用其不可读内容。

## 参考来源

1. UK AI Security Institute, [Frontier AI Trends Report](https://www.aisi.gov.uk/frontier-ai-trends-report)；[Incident report: unsanctioned agent behaviour during cyber testing](https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing)；[Building a more secure environment for evaluating dangerous capabilities](https://www.aisi.gov.uk/blog/building-a-more-secure-environment-for-evaluating-dangerous-capabilities)；[The Inspect Sandboxing Toolkit](https://www.aisi.gov.uk/blog/the-inspect-sandboxing-toolkit-scalable-and-secure-ai-agent-evaluations)；[SandboxEscapeBench introduction](https://www.aisi.gov.uk/blog/can-ai-agents-escape-their-sandboxes-a-benchmark-for-safely-measuring-container-breakout-capabilities).
2. [SandboxEscapeBench](https://arxiv.org/abs/2603.02277v3), arXiv v3, 2026-08-01；[AgentCyberRange](https://arxiv.org/abs/2606.14295), arXiv, 2026-06；[FUSE](https://arxiv.org/abs/2609.02168), arXiv, 2026-09-02。
3. Anthropic, [Responsible Scaling Policy v3.0](https://www.anthropic.com/news/responsible-scaling-policy-v3)；Google DeepMind, [Frontier Safety Framework update](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)；OpenAI, [Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf)；[Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/)。

- `status`：第六章结构与外部扫描完成；尚未修改正文、运行 demo、核验 GitHub Mermaid 渲染或同步配套索引。下一步扫描第七章并继续完成第八册架构图谱。
