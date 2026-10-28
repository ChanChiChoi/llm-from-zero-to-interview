# 第八册第十七章：AI Control 与不可信 Agent 监督精修记录

## scope

依据更新后的 `WRITING_SPEC.md`，从正文首行复读至资料清单末尾，完成第十七章结构、论证、示例、代码、图表、来源和配套索引复核。旧扫描及本轮之前的草稿只作线索，不作为本次精修完成证据。章节面向需要评估或设计不可信 Agent 运行时监督的安全、评测和系统工程读者。

## architecture_decisions

保留单篇结构，以“控制问题与威胁边界 → 微协议 → 宏协议和评估 → 指标 → 自适应红队 → 能力校准与安全论证 → 代码 Agent 案例 → 固定输入 demo”推进。第六章主讲红队流程和危险能力，第十三章主讲自动化攻击搜索，第十四章主讲安全分类器，第十五章主讲身份授权与动作执行门禁，第十六章主讲失败恢复；本章说明这些组件如何组成控制协议，不重复完整方法。新增的代码 Agent 沙箱案例提供跨章运行时交接。

## research_scan

检索日期为 2026-10-28，时间窗为 2024-10 至 2026-10。围绕本章问题复核原始论文、ICLR 官方 proceedings、AISI ControlArena 文档与机构报告；资料支持特定协议和评估设置，不构成穷尽搜索。

- AI Control 原始工作支持把可信监控、可信编辑及其 threat model 纳入协议分析，结论绑定代码后门任务和对应模型。
- SHADE-Arena 的 17 组配对任务、sabotage score 与监控 AUC 只按论文轨迹、scratchpad、模型和评分口径叙述；不转成现实 Agent 作恶率。
- 控制协议比较论文将 `defer-to-resample`、`resample-then-defer`、`defer-on-critical` 分开，并通过迭代红蓝队评估协议知情攻击。`17%` 安全和 `100%` 安全、`53%—66%` 效用均是来源论文所测模型、阈值、SHADE 任务和事件定义下的作者报告。
- ICLR 2026 monitor-targeted injection 研究支持加入针对监控器的自适应攻击回归；不能据此断言所有监控器或协议都可被绕过。
- Weak-to-strong monitoring 工作支持分层/顺序监控和人工升级作为候选评估维度；结果受其环境与威胁条件约束。
- AISI AI Control Level 框架和 AI control safety case 论文作为能力校准与安全论证的示例；不把 ACL 当作标准认证、能力预测或统一分级。
- ControlArena 公开文档支持 settings、policy、monitor、micro/macro protocol 等工程术语和入口；文档接口会演进。
- OpenAI 2026 年 10 月训练环境红队文章作为公司自述的相邻运行时案例，不能当成独立复现或其他组织的漏洞概率。

上述来源分别纳入 §§17.1—17.10、§17.14，并在 `PAPERS.md` 建立入口。没有把搜索到的协议名称扩成独立摘要小节；未有目标部署的模型、网络或副作用 trace，故不推断生产风险率。

## changes

增加威胁模型、U/T 资源假设和外部 affordance 限制；重写微协议比较，明确候选是否提交、监控器可见的尝试和最终提交者；补充宏协议审核预算、停止/恢复流程和评估冻结后的 holdout；区分诚实任务效用与攻击样本安全性，并增加逐步手算；补上针对 monitor 的自适应红队、评测到部署迁移假设及安全论证结构；扩充代码 Agent 后门/沙箱案例，并限定 OpenAI 披露为发布方自述；增加 Python demo 输入合同和失败分支说明。旧有章节内容不迁移到其它正文。

## length_status

按解释性中文汉字计数，排除标题、代码和 math 围栏、表格、公式、引用清单、状态行与空白：核心讲解 §§17.1—17.10、§§17.12—17.13 合计 **5,555 字**，高于 1,800 字下限；作为独立辅助 demo 单元的 §17.11 为 **1,144 字**，高于 1,000 字下限。计数脚本将 Markdown 的 fenced blocks、表格行、标题、单美元行内公式和状态行排除；篇幅达标不替代连续复读。

## example_status / formula_status

§17.7 手算使用 20 个诚实主任务：$p_U=16/20=0.80$、$p_T=10/20=0.50$、$p_{\mathrm{protocol}}=13/20=0.65$，因此 $U_{\mathrm{norm}}=(0.65-0.50)/(0.80-0.50)=0.50$。100 对攻击任务中 6 个审核捕获、2 个未捕获副任务成功，故 $p_{\mathrm{audit}}=0.06$、$p_{\mathrm{attack}}=0.02$，在来源定义的重复、平稳部署假设下 $S=0.06/(0.06+0.02)=0.75$。剩余 92 对不是安全成功；代码同时不把无终止事件样本自动记为 $S=1$。正文额外说明事后审计不等于副作用提交前拦截，且非平稳、自适应或跨任务学习会破坏简单重复部署假设。上述均为教学合成值。

展示公式使用标准 `math` 围栏和行内美元公式；变量及适用条件已定义，未发现 `$$`、`\operatorname` 或不匹配的美元符号。两条公式与手算共享同一组输入。

## code_status

§17.11 唯一 Python 教学块在 2026-10-28 以 Python 3.12.9 标准库执行；从 Markdown 抽取唯一 Python 围栏并执行其代码。正文输出与实际输出逐行相同：

```text
main_success U=0.800 T=0.500 protocol=0.650 normalized_usefulness=0.500
long_run_safety audited=6/100 attack=2/100 p_audit=0.060 p_attack=0.020 safety=0.750
decision=hold reasons=safety_below_0.90,usefulness_below_0.60
```

代码内断言覆盖固定数值、$U \le T$ 的未定义基线、无终止事件、负数/布尔/非整数、空任务集、成功计数超分母及终止事件计数超任务对。代码只核对合成计数和固定公式，不实现 monitor、模型、沙箱、执行器或生产判定。

## visual_map

§17.2 使用 1 张 Mermaid 红蓝白队交互图，采用标准三反引号 `mermaid` 围栏；节点标签已加引号，图中使用带文字的实线边，没有虚线边。源码围栏、节点及边标签静态检查通过；尚未在 GitHub 页面确认实际渲染。

## link_status

正文内部链接到第六、十三、十五和十六章，目标文件存在；第十七章的论文、官方文档和报告链接均指向原始论文页、官方 proceedings、AISI 文档或发布方页面。本轮沿用资料复核阶段对对应来源页面的阅读记录；未对全章全部外链执行自动 HTTP 扫描。

## sync_status

`PAPERS.md` 新增 AI Control、SHADE-Arena、协议比较、自适应 monitor attack、weak-to-strong monitoring、AISI safety case/ACL、ControlArena 和 OpenAI 红队报告入口；`EXERCISES.md` 新增 35EJ；`INTERVIEW_BANK.md` 安全与治理新增 Q28；`GLOSSARY_EN_ZH.md` 新增 AI Control、U/T、control protocol、重采样协议、monitor red teaming、Pareto 前沿和安全论证术语。研究与配套引用均链接到本记录。

## validation / open_risks

静态结构检查确认章节围栏配对、唯一 Python 块和一张 Mermaid 图；正文和 demo 周边没有维护文件名或工作流引用。目标平台 Mermaid/MathJax 渲染、全章外链 HTTP 扫描、第八册沿阅读路径连续复读及第八册整册门禁仍开放；没有目标模型、部署 harness、真实副作用、人工审核流程或线上流量验证。论文章节中的数据、控制假设和机构自述均不能外推为一般生产安全保证。

## status

第十七章已实际复读并完成实质精修，固定 demo 输出与预期一致，篇幅、手算和配套索引已记录；完整章节及书册门禁保持开放。下一步按新版规范开展第九册的定位、简介/目录、章节地图和册级发现扫描，然后进入逐章精修。
