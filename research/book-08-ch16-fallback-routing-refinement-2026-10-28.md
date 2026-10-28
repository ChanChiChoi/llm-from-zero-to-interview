# 第八册第十六章 Fallback Routing 研究与精修记录

## Scope 与架构决定

依据 2026-10-28 当前版 `WRITING_SPEC.md` 从头复读并实质修订第十六章。按第五章精修条目的字段记录篇幅、证据、可复算例子、代码、图表、链接、配套索引和开放风险。保留单篇结构：状态恢复与安全不变量 → 失败分类和路由组件 → 有界重试、预算和幂等恢复 → 协议、证据和授权降级 → 评估与发布 → 固定状态审计 demo → Claude System Card 路由案例。第 16.20—16.22 节作为一组辅助比较案例；与第二十册第 24 章交叉链接，不重复接口迁移主讲。

## 定向研究扫描与证据边界

- Google SRE 的 [Handling Overload](https://sre.google/sre-book/handling-overload/) 与 [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/) 支持过载控制、降级、重试放大和级联故障的工程背景；不能据此推断具体 provider 的内部请求状态或 SLA。
- [RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2) 区分幂等方法并规定非幂等请求的自动重试边界。HTTP 方法语义不能代替付款、邮件或文件操作的应用层幂等键及提交查询契约。
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) 与 [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) 支持生命周期风险管理背景，不构成具体 fallback 方案的认证。
- [OWASP Top 10 for LLM Applications 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) 用作提示注入、敏感信息和工具集成风险分类入口；社区风险指南不能证明实现安全。
- Opus 5.5、Fable/Mythos 5.1 与 Sonnet 5.5 案例复用各自官方 System Card 的版本化摘录及哈希记录：[Opus 5.5](model-update-2026-09/claude-opus-5.5-source-notes.md)、[Fable/Mythos 5.1](model-update-2026-09/claude-fable-5.1-source-notes.md)、[Sonnet 5.5](model-update-2026-09/claude-sonnet-5.5-source-notes.md)。这些是发布方评测和产品策略说明；条件性 safeguard、fallback 覆盖和 benchmark 指标不等于独立复现，也不能外推为本地或生产安全率。

候选主题去向：错误分类、SRE 过载与熔断进入 §§16.3、16.5—16.6；提交状态、HTTP 幂等边界和状态查询进入 §16.7；协议兼容与上下文完整性进入 §§16.8—16.9；综合安全保持率和重复副作用率进入 §16.12；Claude 三份 System Card 作为一组具体发布策略案例放在 §§16.20—16.22。没有把模型切换写成普适安全策略，也没有把 RFC 或治理框架写成 Agent 工具合同。

## 正文改动

- 将 §16.1.3 的 ASCII 状态图和 §16.4 的路由框图改为标准 Mermaid 围栏；节点标签加引号，所有文字虚线边采用 `-.->|标签|`。逐步解释 `unknown`、`failed` 与 `committed` 对自动重试的不同影响。
- 修正 §16.5.1 重试预算变量，补上等待和执行两部分；用 `0+0.4+0.2+0.6=1.2 秒` 对照 `1.5 秒` 上限，再演示增加一次退避与调用后达到 `2.2 秒` 而应停止。
- 修正 §16.6.1 嵌套重试上界：各层重试额度为 $n_r,n_m,n_t$ 时，嵌套结构上界为 $(n_r+1)(n_m+1)(n_t+1)$；每层各重试一次得到 $2×2×2=8$ 次尝试，并注明该乘积只适用于所述调用结构。
- §16.7.3 引入 RFC 9110 方法幂等性边界；说明非幂等请求只有在确认原请求未生效或应用契约定义了幂等语义后才可重试。
- §16.9.3 修复损坏的行内数学标记；§16.14 定义联合效用公式中的变量并增加 `8-1-1-0-2=4` 手算，同时强调授权和未知状态是硬门禁。
- §16.12 用两个独立样本集手算安全保持率 `9/12=0.75` 与重复副作用率 `1/20=0.05`，说明分母条件和空样本的 `N/A` 处理。
- 扩写 §16.15 demo 的读法、权限边界和断言覆盖；将 retry exhausted 纳入九条固定轨迹，增加完整预期映射和范围断言。新增范围模型及生产状态竞态的解释。删除章节末端的工作流口吻，改为说明 System Card 发布结果与本地/生产验证证据之间的界线。
- 更新 OWASP 入口至 2026 页面；为 Opus 5.5、Fable/Mythos 5.1、Sonnet 5.5 的具体结论就近增加官方发布页或 System Card 链接，并明确评测配置、actual model、fallback 覆盖等口径。

## 长度与数值例

按解释性中文汉字计数，排除标题、代码/公式/图示围栏、表格、引用清单、行内代码/公式和状态标记：

| 讲解单元 | 汉字数 | 门槛 | 状态 |
| --- | ---: | ---: | --- |
| 核心主线 §§16.1—16.14、§§16.16—16.19 | 6,458 | 1,800 | 达标 |
| 辅助 demo §16.15 | 1,053 | 1,000 | 达标 |
| 辅助比较案例组 §§16.20—16.22 | 1,559 | 1,000 | 达标 |

核心手算使用正文定义的预算：首轮耗时 `0.4 秒`，两轮之间退避 `0.2 秒`，第二轮耗时 `0.6 秒`，总预算消耗为 `1.2 秒`；若下一轮再用 `0.4+0.6 秒`，累计为 `2.2 秒`，超过 `1.5 秒`。三层各准一次重试时，尝试上界是 `(1+1)^3=8`。由合成轨迹分别算安全保持率 `9/12=0.75` 和重复副作用率 `1/20=0.05`。联合效用示例为 `8-1-1-0-2=4`。这些是教学输入，不是生产时延、事故率或模型评测。

## 代码、公式与图表状态

- `code_status`：§16.15 唯一 Python 围栏已于 2026-10-28 在 Python 3.12.9 标准库执行。通过 Markdown 提取该围栏并运行固定输入；九条合成轨迹的输出与正文预期逐行一致。覆盖 plain timeout 重试、unknown 状态查询、策略拒绝、后端不支持工具、查询不可用、提交后撤权、撤权后只读查询、查询权限撤销和重试耗尽。代码断言完整决定/动作/scope 映射，且逐条确认允许 scope 不扩大。它不调用远端状态服务、provider、PDP、工具或持久幂等存储，不覆盖并发租约、撤权竞态和实际副作用。此次只新增解释文字，没有改变可执行语句，因此既有运行证据继续适用，无需重复执行。
- `formula_status`：展示公式使用 `math` 围栏或单美元行内式；修正嵌套重试表达式与损坏的 `(k)`，并为各符号补上语义。章节没有 `\operatorname` 残留。
- `visual_map`：两张 Mermaid 图使用标准三反引号围栏、引号节点标签和 `-.->|标签|` 文字虚线箭头；源码语法已静态核对。目标平台渲染仍未核验。
- `link_status`：正文新增 RFC 9110 §9.2.2、Google SRE、NIST、OWASP 2026 入口及三份发布方 System Card 直链；本章与本记录的本地相对链接均已检查且目标存在。定向来源页面已在本轮研究中核读或由同一册中已哈希的官方 System Card 摘录核验；未声称完成全章所有历史外链的 HTTP 扫描。

## 配套同步与尚未完成项

- `PAPERS.md` 增加第十六章 Google SRE、RFC 9110、NIST、OWASP 2026 与三份 System Card 研究记录入口。
- `EXERCISES.md` 增加 35EI，覆盖九条状态轨迹、嵌套重试、时延预算、独立指标分母、撤权与业务幂等。
- `INTERVIEW_BANK.md` 更新第八册 fallback 面试题，强调先查状态、权限不扩大、unknown 查询、幂等和实际副作用评估。
- `GLOSSARY_EN_ZH.md` 更新 Fallback Routing，增加 Safety Preservation Rate 与 Duplicate Side-Effect Rate。
- `git diff --check` 通过；章节含 32 个围栏标记（16 对），其中两个 Mermaid 和一个 Python 围栏；无 `~~~`、`$$`、`\operatorname` 或进度文件名；本地相对链接目标已核对。
- GitHub Mermaid/MathJax 页面渲染、全章历史外链 HTTP 检查和第八册连续阅读路径复读仍开放；未在真实 provider、工具、授权服务或线上流量验证。因此第十六章和整册门禁仍保持开放。

`status`：第十六章正文已实质修订，教学数值和固定代码按所述范围核对，配套索引已同步；完整章节/书册门禁未通过，下一篇按序为第十七章。
