# 第八册第十二章：安全的系统化判断与综合复习研究记录

## 范围与章节定位

依据当前 `WRITING_SPEC.md` 从头复读第十二章全文，按“安全目标 → 代理目标 → 监督 → 攻击面 → 指标 → 危险能力与不确定性 → 机制证据 → 安全平台 → 综合案例 → 技术表达”的链路审查。保留单篇综合复习章，不把前十一章的算法细节再讲一遍；补专题回链，让本章负责建立跨层判断和面试表达，而不是替代前序主讲。

本章解释性主线为 §§1—10。§11 是可独立使用的面试表达辅助讲解，§12 是独立的答题复盘审计 demo。综合练习、来源列表和小结不计作讲解单元。

## 章级扫描与候选去向

扫描问题包括：近期风险目录如何组织不同安全失败；预部署评估与真实效用/风险之间有哪些已知证据缺口；Agent runtime 控制接口有哪些值得作为设计入口的方案；评估工具的公开入口是否仍然准确。来源限定在报告原文、OWASP 官方项目页面、NIST 官方资料和 OpenAI Evals 官方仓库。扫描是发现方向的有限快照，不是穷尽检索。

| 候选 | 来源支持与边界 | 去向 |
| --- | --- | --- |
| International AI Safety Report 2026 的 misuse / malfunctions / systemic risks 分类和评估缺口 | 报告于 2026-02-03 发布；由国际专家综合通用 AI 能力、风险和治理资料。报告指出能力/现实影响度量仍不成熟、预部署测试与真实部署存在评估落差，并整理模型识别测试情境及利用评估漏洞的证据 | 纳入 §5.2、§5.7、§14.2，作为高层分类与证据边界；不把报告综述当成单模型测量。建议隔离 holdout 与线上反馈是本章据此作出的工程推论，不声称报告验证了某个具体控制 |
| OWASP Top 10 for LLM Applications 2026 | OWASP 官方页面将其描述为 community-driven 的应用风险分类，2026-08-03 发布；可帮助发现应用风险面，但不测量目标系统安全性 | 纳入 §5.2、§14.2；不把风险排序当作目标产品的故障概率 |
| OWASP Agent Control Standard (ACS) | OWASP 官方页面 2026-09-01 介绍 middleware hooks、可移植的声明式控制及 runtime policy 接口方向 | 纳入 §9.4、§14.2，并保留 PAPERS 入口；写成社区方案和接口词汇，不写成已证明安全有效的标准 |
| OpenAI Evals 当前公开入口 | 官方仓库仍作为开源 framework/registry 展示，README 同时指向 Dashboard 配置与运行 Evals，并说明当前不接收 custom-code eval contributions | §14.2 将其明确为一个实现/registry入口，并提示产品工作流需按最新官方文档复核；不把仓库定义成所有 safety evaluation 的通用方法 |
| 前十一章已有技术路线和新专题 | Alignment、scalable oversight、prompt injection、red teaming、interpretability、steering、editing/unlearning、privacy 和 governance 都已有单独主讲 | 本章只连接任务定义、分母、证据边界、平台和面试表达；不重复展开方法细节，也未新增独立章节 |

外部来源：

- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026)
- [OWASP Top 10 for LLM Applications 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)
- [OWASP Agent Control Standard](https://genai.owasp.org/resource/agent-control-standard-acs/)
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)、[NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)
- [OpenAI Evals repository](https://github.com/openai/evals)

## 正文修订

- 开篇增加专题回链，说明本章的综合复习职责和相关前章入口。
- §3.3 增监督 recall 与 AI feedback disagreement 的两组不同分母手算。
- §5.2 映射高层风险目录到可执行样本 taxonomy，并说明单条事故可跨类别；§5.3 补有害遵循率、误拒率的手算与分母分离，排除“允许请求回答错误”等非拒答失败；§5.5 明确越权工具执行率与服务端阻断覆盖率分开统计；§5.7 纳入评估缺口，并将 holdout 建议标为基于报告证据的工程推论。
- §6.3 补 AI / 非 AI baseline 的同任务增量手算；§7.2 补 Brier score 三样本中间步骤；§8.2 补 steering 向量数值例，强调隐藏状态变化不等于安全收益。
- §8.4 补充删除请求对索引、缓存、摘要、日志和副本的影响，区分模型 unlearning 与系统数据删除证据。
- §9.4 将 ASCII 路径改为标准 Mermaid，加入带文字虚线的复核/降级分支和图后读法；引入 ACS 作为接口方案示例，明确效果未验证。
- §10.7 补企业研究助手数值案例：20 条允许问题中 18 条成功、2 条误拒；10 条越权检索全部阻断；区分固定测试集结果与总体风险。仅在独立随机试验假设下解释 Rule of Three 上界 `3/10≈0.30`，说明构造红队样本不满足此类总体估计前提。
- §11 增面试表达因果链、常见追问和不确定性表达方式；§12 demo 修复 risk coverage 门槛与 action 不一致的问题，添加严格输入校验、空分母状态和手算。

## 单元篇幅

按 §3.3 统计连贯中文解释性汉字，排除标题、围栏中的公式/代码/图示、表格单元格、行内代码/公式、链接标签及引用目标、状态标记。统计方法保守地移除链接标签。

| 单元 | 汉字数 | 类型 | 下限 |
| --- | ---: | --- | ---: |
| §§1—10 综合安全判断主线 | **8,026** | 核心 | 1,800 |
| §11 如何表达安全判断 | **1,144** | 辅助 | 1,000 |
| §12 Safety 复盘审计 demo | **1,140** | 辅助 | 1,000 |

篇幅数字仅证明超过机械下限；逻辑深度和示例是否清晰仍由人工复读判断。

## 公式与手算

- Scalable oversight：20 个确认错误中发现 15 个，recall `15/20=.75`；30 条 AI feedback 中 3 条与专家不一致，disagreement `3/30=.10`，分母不同。
- 有害遵循：禁答集合权重总和 6，未被安全约束的回答权重为 3，得到 `.50`；允许请求中 2/8 错误拒绝得到 over-refusal `.25`，两指标分开报告。
- Dangerous capability：同一 10 个任务和预算下，baseline 4/10、AI system 7/10，差值 `.30`，不外推为现实用户能力提升。
- Brier example：置信度 `.9,.8,.2`、结果 `1,0,0`，中间平方误差 `.01,.64,.04`，均值 `.23`。
- Steering：`h=(.2,.4)`、`v=(1,-1)`、`alpha=.5` 得 `h'=(.7,-.1)`，只表示向量编辑，不代表行为控制成功。
- 企业案例：允许任务成功 `18/20=.90`、误拒 `2/20=.10`、固定越权场景服务端阻断 `10/10=1.0`；其覆盖不代表开放生产写入。

## Demo 执行与边界

§12 唯一 Python 围栏在 Python 3.12.9、仅使用标准库的环境执行。固定结果：

- topic scores：`.8, 1.0, .8, 1.0, .833`；加权 answer coverage `.892`；
- risk coverage `7/8=.875`，由于全覆盖门槛为 `1.0`，状态为 partial；metric coverage `5/7=.714`，状态为 partial；unsafe detail count `0`；
- decision：`revise_answer_and_retest`；缺项包括 over-refusal 风险类别、severity-weighted risk、evidence coverage、prompt injection 的 normal-task 切片、Safety Platform 的 incident 内容等；
- 9 项边界断言通过：零分母、空集合、空加权平均、未定义门禁、负数/无穷/布尔比例、超出 rubric 的 covered 项和非有限权重均得到 N/A、门禁失败或输入错误。

代码输出与手算一致。程序负责机械 rubric 计数，不判断回答事实是否正确、评分者是否一致或样本是否代表目标系统；不用于招聘决策、模型安全认证或真实攻击判定。

## 配套资料与校验

- `EXERCISES.md`：新增 35EF，练习高层风险分类映射、分母、红队样本边界及六步表达。
- `INTERVIEW_BANK.md`：扩展安全综合题 Q15，增加 risk taxonomy、评估证据层次和结论范围。
- `GLOSSARY_EN_ZH.md`：复用并细化已有 Misuse Risk、Malfunction Risk、Systemic Risk 定义，补充 OWASP Top 10 for LLM Applications 2026；不重复建同名术语。
- `PAPERS.md` 原有 International AI Safety Report 2026、OWASP 2026 Top 10 和 ACS 入口已覆盖本章使用来源，本轮不另造重复条目。
- 11 个章节本地相对 Markdown 链接目标全部存在；外部来源新增链接已按官方入口检查。未对全章历史外链做 HTTP 自动全扫。
- `git diff --check` 通过；20 个三反引号围栏配对（18 math、1 Mermaid、1 Python）；旧 `$$`、`\(`、`\)`、`\operatorname` 和波浪线围栏均无残留；行内美元符号成对。Mermaid 节点标签加引号，文字虚线使用 `-.->|标签|`。

## 未关闭项

GitHub 上的 Mermaid/MathJax 目标渲染尚未复核；全章 HTTP 外链扫描、第八册第 1—12 章连续阅读和整册路径验收仍未完成。本地 toy demo 不提供真实模型、真实样本、实际面试评价或生产系统安全证据；本章和第八册不标记完整门禁通过。
