# 第八册第十一章：Policy、Governance 与 Model Card 研究和精修记录

## 范围与章节决定

本轮按更新后的 `WRITING_SPEC.md` 从头复读第十一章，复查章节主线、公式、图表、代码、来源、边界和配套索引。保留单篇结构：第 1—4 节说明治理对象、报告卡、policy 和责任流程；第 5—6 节解释模型与系统层的披露；第 7—9 节连接访问边界、发布门禁和能力扩展；第 10—13 节处理风险披露、审计及工程取舍。第 14 节保留可执行的卡片结构，第 15 节保留最小审计 demo。

旧第 7 节大段讲述 risk-calibrated access 与 fallback routing，与第十五、十六章的请求级授权、运行时状态、幂等和降级路径重复。本轮把它收束为治理记录需要向系统传递的发布范围、配置审批、证据有效期与 fallback 交接；定量路由由本章给一个手算例，完整运行时机制交给第十五、十六章。未新增章节。

## 章级扫描与候选去向

扫描问题聚焦于：模型/系统报告卡如何绑定版本和证据，近期法规与机构治理框架的法律地位和更新情况，公开系统卡的数字如何保留评测条件，以及发布门禁如何处理缺测与硬约束。来源以法规原文、欧盟委员会、NIST 和模型开发机构的官方页面为主，并用既有模型卡研究补充报告框架背景。

| 候选方向 | 证据与范围 | 决定 |
| --- | --- | --- |
| GPAI 模型文档义务及自愿 Code of Practice | 欧盟委员会 GPAI obligations 页面列出技术文档、版权政策和训练内容公开摘要等责任；Code of Practice 页面说明透明度章节包含 Model Documentation Form，Safety and Security 章节针对系统性风险相关义务。法规与自愿合规工具分别表述 | 纳入 §5.3、§14.1、§18.2；不把欧盟表格写成所有 Model Card 的通用法律要求 |
| Article 50 透明义务与生成内容标记 | 欧盟委员会 FAQ 对适用角色、系统和日期作说明；生成内容标记准则属于配套实践材料 | 纳入 §10.3、§18.2；与模型身份/作者证明和 C2PA provenance 声明区分，不把标记等同来源真实性 |
| 机构前沿安全与治理框架更新 | OpenAI Preparedness Framework v2（2025-04-15）、Frontier Governance Framework（2026-05-28）、Anthropic RSP v3.4（2026-07-08 生效，页面 2026-08-14 更新）、Google DeepMind FSF v3.1（2026-04-17）及 NIST AI RMF 页面按发布方当前材料记录 | 纳入 §9.3 比较表；区分机构承诺、通用自愿风险框架、受众与版本，不视为独立评测或普遍法律标准 |
| NIST AI RMF 与 GenAI Profile | NIST 将 AI RMF 1.0 标为自愿框架并说明主框架修订状态；GenAI Profile 是 2024 年配套文件 | 纳入 §9.3、§18.2；不把框架使用写成发布认证 |
| Model Card / System Card 的版本依赖与证据账本 | Model Cards 与 Datasheets 原始论文提供报告框架；Claude Opus 5/5.5 官方 System Card 展示 snapshot、harness、fallback、任务和计时口径会改变结论范围 | 扩展 §§5—6，保留具体官方案例；厂商自报作为“发布方报告内容”的证据，不作为独立复现 |
| 尚未证实的未来模型清单 | 旧稿有未经稳定官方证据支持的预测性名称 | §12.1 改为通用证据等级与暂定命名规则，不保留猜测性名单 |

主要官方来源：

- [EU AI Act GPAI obligations](https://digital-strategy.ec.europa.eu/en/factpages/general-purpose-ai-obligations-under-ai-act)、[GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai)、[Article 50 transparency FAQ](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act)、[AI-generated content Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)、[NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- [OpenAI Preparedness Framework update](https://openai.com/index/updating-our-preparedness-framework/)、[OpenAI Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/)
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy)、[Google DeepMind Frontier Safety Framework](https://deepmind.google/frontier-safety/)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)、[Datasheets for Datasets](https://arxiv.org/abs/1803.09010)、[Claude Opus 5.5 System Card](https://www.anthropic.com/claude-opus-5-5-system-card)、[Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card)

## 正文、公式与教学例子

- §1.1、§4.3、§6.1 使用三张标准 Mermaid 图，分别说明治理生命周期、证据复评闭环和系统拓扑。节点标签加引号；反馈虚线使用 `-.->|标签|`。正文说明实线与虚线含义。
- 展示公式使用 `math` 围栏，行内公式使用单美元；移除旧式 `$$`、`\(...\)` 和 `\operatorname`。发布覆盖率、未关闭风险权重、高风险缓解和访问候选比较都保留逐步可手算输入。
- 访问候选例：只读路由 `Q=.82,C=.10,H=.02, λ=.5, μ=2` 得 `U=.73`；可写 Agent `Q=.90,C=.20,H=.24` 得 `U=.32`。只对满足授权、证据和可逆性硬条件的候选比较效用。
- 发布 demo 的固定输入手算：Model Card `8/9=.889`、System Card `8/10=.800`、Policy `4/6=.667`、Evaluation `4/7=.571`、未解决风险权重 `12/16=.750`、高风险缓解 `8/13≈.615`、审批 `4/5=.800`。六项比例门槛失败，审批门槛通过，隐私日志硬约束仍失败，最终 hold。风险账本权重不代表事故率。
- §14 模板增加证据、版本、未知状态与变更影响的填写方法；以权重不变但 tokenizer、策略和 RAG 索引变化为例，解释模型卡和系统卡分别更新什么、原评估何时变 stale。
- §15 demo 增加输入合同、分母含义、硬门禁顺序、缺测与无效输入区别及结果适用范围。Python 代码本身和代码相邻说明均未写维护文件名。

## 单元篇幅

按 §3.3 规则机械统计中文解释性正文，排除标题、围栏中的代码/公式/图示、表格单元格、行内公式与代码、引用目标及注释性状态标记；每个小节组按独立教学单元计。统计脚本保守地移除了 Markdown 链接标签和目标。结果如下：

| 单元 | 汉字数 | 门槛 |
| --- | ---: | ---: |
| §§1—4 核心 | 3,011 | 1,800 |
| §§5—6 核心 | 2,186 | 1,800 |
| §§7—9 核心 | 2,304 | 1,800 |
| §§10—13 核心 | 2,351 | 1,800 |
| §14 辅助卡片模板 | 1,139 | 1,000 |
| §15 辅助审计 demo | 1,190 | 1,000 |

篇幅达标只表示越过机械下限，不代替人工复读。§5—6 和 §7—9 已补充版本变化、跨组件行为链及候选比较的硬门槛解释，避免只靠表格或公式承载主线。

## 代码执行与边界

§15 唯一 Python 块已在 Python 3.12.9、仅使用标准库的环境中执行；Markdown 中的固定输入生成：

- signals：`model_card=.889`、`system_card=.800`、`policy=.667`、`evaluation=.571`、`open_risk=.750`、`high_risk_mitigation=.615`、`approval=.800`；
- failed thresholds：Model Card、System Card、Policy、Evaluation、open risk、high-risk mitigation；审批通过；
- failed constraints：`privacy_logging_controlled`；decision 为 `hold_and_repair_hard_constraints`；
- boundary checks：空比例、缺测比例、零权重分母和空 policy 集返回 `None`；未定义信号门禁不通过；负权重、非有限权重和非布尔比例输入被拒绝。

输出与正文表格及逐步手算一致。代码块修改后已执行；本轮后续只扩写解释文本，未改变可执行语句。示例没有连接真实模型、组织审批、法律检查、生产日志或工具系统，不能作为其发布批准证据。

## 配套资料与验证

- `PAPERS.md`：补充 OpenAI、Anthropic、Google DeepMind 的当前治理框架入口及 EU GPAI/Article 50 官方材料。
- `EXERCISES.md`：新增第八册第十一章卡片版本追踪、门禁手算、N/A 和法规/框架边界练习。
- `INTERVIEW_BANK.md`：扩展 Model Card/System Card 与治理门禁回答，纳入证据绑定、版本失效、硬约束和法律地位区分。
- `GLOSSARY_EN_ZH.md`：新增 Frontier Governance Framework、Tracked Capability Level、GPAI Code of Practice、Model Documentation Form 和 Article 50 透明义务术语。
- 章节内 Markdown 相对链接均已定位检查；Claude Opus 5 研究笔记目标存在。当前未对本章全部历史外链做 HTTP 全量扫描。
- `git diff --check` 通过；12 个三反引号围栏配对（3 Mermaid、7 math、1 text、1 Python），公式旧语法无残留，三个 Mermaid 虚线箭头标签格式正确。

## 未完成门禁

GitHub 页面上的 Mermaid/MathJax 渲染尚未复核；本章全部外链 HTTP 全量扫描和第八册第 1—11 章主路径连续复读仍开放。因此本章完成了正文精修、研究、配套同步和声明范围内代码执行，但没有标记完整章节或书册门禁通过。第八册之前章节的未关闭门禁继续有效。
