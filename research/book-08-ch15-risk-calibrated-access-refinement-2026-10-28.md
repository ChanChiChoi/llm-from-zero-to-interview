# 第八册第十五章 Risk-Calibrated Access 研究与精修记录

## Scope 与架构决定

依据 2026-10-28 当前版 `WRITING_SPEC.md` 从头复读第十五章，并参照第五章精修条目的复核字段组织记录。早期扫描只作候选线索，不视为正文已修改或验收完成的证据。本章保留为一篇，主线为访问请求建模 → 风险、权限、证据和状态分离 → PDP/PEP 与身份委托 → 执行前状态核验 → 风险校准和系统评估。未新增章节；classifier 信号留在第十四章，运行时授权、撤销竞态和实际副作用留在本章。

## 定向研究扫描与证据边界

- NIST SP 800-207 与 800-207A 支持按资源、身份和应用上下文实施零信任访问，并说明云原生环境要同时处理用户与服务身份。它们是架构指导，不给 Agent 提供可照抄的角色表或风险阈值。
- OpenID Foundation 的 AuthZEN Authorization API 1.0 在规范状态页标明 2026 年 1 月获批为 Final Specification。其边界是 PDP/PEP 的授权请求与决定互操作；不定义内部策略语言、风险模型或实际 Agent 动作权限。状态页已核读；直接规范 URL 在本轮浏览器抓取中超时，故不把它记为完整外链 HTTP 扫描。
- OAuth 2.0 Token Exchange (RFC 8693) 定义令牌交换协议及委托/冒充、`sub`/`act` 等语义；部署信任模型、全部 token 安全属性及业务授权需要另行规定。
- DPoP (RFC 9449) 以持有证明对 OAuth token 作 sender constraint，帮助降低被盗 token 的重放风险；不授予业务权限，也不代替 HTTPS。
- OWASP Top 10 for Agentic Applications 2026 将 Identity and Privilege Abuse 列为 ASI03。该目录帮助发现继承身份、凭证泄漏和权限扩展风险，不是实现标准，也不能证明部署安全。
- Zanzibar 论文和 Cedar 文档分别作为关系型授权/一致性和具体策略引擎输入模型的案例。两者都是有边界的实现参考，不代表所有 PDP 的共同语义。

候选主题去向：NIST 资源与身份边界进入 §§15.2、15.5；AuthZEN 进入 §15.3.4；委托、冒充和令牌持有证明进入 §15.5.1；RBAC/ABAC/ReBAC 与具体实现边界进入 §15.5.2；ASI03 作为威胁类别进入 §15.8.1；版本竞态、撤销、执行器状态和指标分母进入 §§15.9—15.14。没有把 AuthZEN、Cedar 或 Zanzibar 写成 Agent 安全的统一方案。

## 正文改动

- §15.2.1 将加权风险值明确限定为组织自定义排序/分层指标，补全五个归一化因子、权重和为 1 的前提，以及生产发布 `0.59` 的逐项手算；说明它不是事故概率，并用零信任资源/身份判断作边界对照。
- §15.3.1 将文本流程改为标准 Mermaid PDP/PEP 执行闭环；§15.3.4 纳入 AuthZEN 互操作范围和限制。
- §15.5.1 区分发起用户、执行 Agent/service actor、委托 scope 与审批者，并解释 RFC 8693 和 DPoP 各自解决的问题；§15.5.2 增加 RBAC/ABAC/ReBAC 对照及 Zanzibar、Cedar 的适用边界。
- §15.8.1 把 OWASP ASI03 标为风险分类，不把风险目录等同实现证据。
- §15.10.2 明确 $r_p$、$r_o$、$r_s$ 是审批时绑定的主体、资源和策略版本，并要求 PEP 从权威状态源读取当前版本后逐项比较。
- §15.10.4 补充主体/资源/策略版本变化的合成时间线，并说明策略服务恢复后逐项重评估，不能批量回放旧批准。
- §15.12.1 区分启发式风险值 $R$ 与预测分数 $s$；§15.12.2 手算两桶 ECE；§15.12.3 定义漏放、误拒和人工复核成本。
- §15.13.4 将阶段发布链改为标准 Mermaid，说明 stage、scope、revision 和回滚边界。
- §15.14.2 分开 HighRiskAllowRate 与 UnauthorizedSideEffectRate，定义不同分母、ground truth 和空分母 N/A。
- §15.15.2 使用 total variation distance 演示两分布差异：$(0.6,0.3,0.1)$ 与 $(0.3,0.5,0.2)$ 的 TV 距离为 $(0.3+0.2+0.1)/2=0.3$，并指出分布变化不等于事故增加。
- §15.17 将教学 demo 改为服务端已核验输入合同，区分活跃主体与 grant、resource scope、revision、审批和双人审批；包含 11 条合成请求路径和范围/数值断言。状态行及代码附近说明没有内部工作流引用。

## 长度与数值例

按解释性汉字统计，排除标题、代码/公式/图示围栏、表格、引用清单、行内代码/公式与状态行：

| 讲解单元 | 汉字数 | 门槛 | 状态 |
| --- | ---: | ---: | --- |
| §§15.1—15.6 | 2,647 | 1,800 | 达标 |
| §§15.7—15.11 | 1,869 | 1,800 | 达标 |
| §§15.12—15.16 | 2,202 | 1,800 | 达标 |
| 辅助 §15.17 | 1,051 | 1,000 | 达标 |

核心手算输入沿用 demo：五项乘积 `0.15 + 0.20 + 0.10 + 0.10 + 0.04 = 0.59`。ECE 两桶权重为 `20/30` 与 `10/30`，误差为 `0.1` 与 `0.2`，所以 `ECE = (20/30)×0.1 + (10/30)×0.2 = 0.1333`。数值均为合成教学示例，不是生产事件率或校准结论。

## 代码、公式与图表状态

- `code_status`：§15.17 唯一 Python 围栏已于 2026-10-28 在 Python 3.12.9 标准库环境执行。使用 `python3 - <<'PY'` 从 Markdown 提取唯一 Python 围栏并以 `exec(compile(...))` 执行。11 条输出与正文预期逐行一致：`public_read 0.100 allow_read`、`sandbox_patch 0.435 allow_sandbox`、`production_release 0.590 review`、`production_canary 0.590 allow_canary`、`stale_approval 0.590 re_evaluate`、`missing_rollback 0.590 hold`、`high_risk_release 0.815 hold`、`high_risk_release_dual 0.815 allow_canary`、`wrong_scope_release 0.590 hold`、`ungranted_write 0.200 deny`、`revoked_write 0.200 deny`。除五项输入长度/有限性/`[0,1]` 校验外，断言还覆盖全零、全一、错误长度、布尔值、字符串、NaN、无穷、越界因子、风险分数 `0.59` 和 11 个预期决策。只验证固定合成策略，不是身份服务、PDP、PEP、生产工具或真实撤销时延验证。
- `formula_status`：展示公式使用 `math` 围栏或单美元行内式；变量在公式前定义，未发现 `\\operatorname`。版本比较的三个批准时变量已显式定义。
- `visual_map`：两张 Mermaid 均使用标准围栏、优先引用节点标签，文字虚线箭头采用 `-.->|标签|`。源码围栏与箭头语法静态检查通过；GitHub 实际页面渲染仍开放。
- `link_status`：本轮直接核读 NIST SP 800-207/207A、OpenID AuthZEN 状态页、RFC 8693/9449、OWASP Agentic Top 10、USENIX Zanzibar 和 Cedar 授权/校验文档。AuthZEN 最终规范的直接 URL 抓取超时，状态页能确认版本及用途；本轮不声称已做全章全链接 HTTP 扫描。

## 配套同步与尚未完成项

- `PAPERS.md` 增加第十五章 NIST、AuthZEN、RFC 8693、RFC 9449、OWASP ASI03、Zanzibar 和 Cedar 资料入口。
- `EXERCISES.md` 新增 35EH，练习风险分数、ECE、主体/grant/scope/revision、双人审批和两个授权指标分母。
- `INTERVIEW_BANK.md` 更新第八册第 10 题，明确区分第十四章分类信号与第十五章授权执行，并链接两篇正文。
- `GLOSSARY_EN_ZH.md` 修订 Risk-Calibrated Access 定义，补 PDP、PEP、AuthZEN 与 DPoP。
- `git diff --check` 通过；围栏配对、公式残留/变量、Mermaid 箭头、代码状态位置及章节维护文件名检查通过。未运行测试套件或 PDF 构建。
- GitHub Mermaid/MathJax 页面渲染、全章全部外链 HTTP 扫描和第八册连续阅读路径复读未完成，因此正文和书册完成门禁保持开放。未在真实模型、Agent runtime、身份服务、目标工具或生产流量上验证。

`status`：第十五章正文已实际精修，合成例和执行代码已核对，索引已同步；按序继续第十六章。完整章节/书册门禁仍开放。
