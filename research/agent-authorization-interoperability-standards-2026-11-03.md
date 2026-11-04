# Agent 授权互操作与运行时控制标准资料

检索日期：2026-11-03。本文只记录官方标准页面及规范草案支持的事实，不记录章节审阅过程或修改决定。

## AuthZEN Authorization API 1.0

- OpenID Foundation 于 2026-01-12 批准 Authorization API 1.0 为 Final Specification。该 API 为 Policy Decision Point（PDP）与 Policy Enforcement Point（PEP）交换授权请求和决定提供接口，不要求双方了解彼此内部实现。
- 请求信息模型围绕 subject、action、resource 与 context；规范提供 API 和数据交换语义，不统一部署方的内部策略语言、风险阈值或业务授权规则。
- 来源：[Final Specification 批准公告](https://openid.net/authorization-api-1-0-final-specification-approved/)；[Authorization API 1.0](https://openid.net/specs/authorization-api-1_0.html)；[AuthZEN 工作组规格状态](https://openid.net/wg/authzen/specifications/)。
- 证据边界：最终规范证明接口标准状态和公开接口内容，不证明任何具体 PDP、PEP 或 Agent 部署正确执行授权。

## AuthZEN Access Request and Approval Profile（AARP）1.0

- OpenID Foundation 于 2026-06-15 宣布 AuthZEN 工作组批准 AARP 为 Working Group Draft。草案为当前拒绝但允许补齐前置条件的访问请求定义扩展流程，包括 requestable denial、申请 task/status handle、异步满足审批等条件和重新评估。
- 草案要求拒绝提示带过期时间和可验证绑定信息（如 evaluation ID 或 binding token）；请求处理完后，策略决策点仍在执行时重新评估授权，批准本身不会把原拒绝变成 allow。
- 来源：[AuthZEN 工作组草案公告](https://openid.net/openid-foundation-advances-authorization-for-the-agent-era-with-new-authzen-working-group-drafts/)；[AARP 1.0 Draft 1](https://openid.github.io/authzen/authzen-access-request-approval-profile-1_0.html)；[AuthZEN 工作组规格状态](https://openid.net/wg/authzen/specifications/)。
- 证据边界：截至检索日仍标为工作组草案，不是 Final Specification；草案描述的互操作流程不证明任何组织的审批身份或权限配置有效。

## COAZ 与 COAZ-MCP

- COAZ（Compatible with OpenID AuthZEN）框架把具体协议/接口的信息模型映射为 AuthZEN Authorization API 请求。COAZ-MCP 是面向 Model Context Protocol（MCP）的绑定；草案描述如何从 MCP 操作及 tool schema 的映射数据构造 AuthZEN 请求并在 PEP 侧执行决定。
- OpenID Foundation 于 2026-06-15 将 COAZ / MCP 授权 profile 列为 Working Group Draft；工作组规格页将 COAZ Framework 1.0、COAZ-MCP Binding 1.0 标为草案。
- 来源：[AuthZEN 工作组草案公告](https://openid.net/openid-foundation-advances-authorization-for-the-agent-era-with-new-authzen-working-group-drafts/)；[COAZ Framework 1.0 Draft 1](https://openid.github.io/authzen/authzen-coaz-framework-1_0.html)；[COAZ-MCP Binding 1.0 Draft 1](https://openid.github.io/authzen/authzen-coaz-mcp-binding-1_0)；[AuthZEN 工作组规格状态](https://openid.net/wg/authzen/specifications/)。
- 证据边界：协议映射草案不授予业务权限。主体身份、资源 scope、参数真实性和执行点仍由部署方的信任与策略边界负责。

## OWASP Agent Control Standard（ACS）

- OWASP ACS 官方文档站在 2026-11-03 显示当前发布 `v0.1.4`、规范版本 `v0.1.0`。其 Instrument 规范定义 Agent 平台的生命周期 hook、与 Guardian 的运行时交互合同及决策 disposition；Trace 和 Inspect 另由相应规范描述。
- ACS 与 AuthZEN 的层次不同：ACS 标准化 Agent 运行时如何接入拦截和控制；它不替部署方定义具体授权规则。ACS 可以与策略引擎配合，但接口存在本身不是权限判断正确的证据。
- ACS v0.1.4 符合性资料明确说明，当前 profile 声明为自我声明，尚无独立验证方、符合登记册或随版本发布的符合性测试套件。
- 来源：[OWASP ACS 状态页](https://owaspacs.org/)；[ACS Instrument 规范](https://owaspacs.org/docs/spec/instrument/specification/)；[ACS 符合性配置文件](https://owaspacs.org/docs/spec/conformance/)；[OWASP ACS 发布公告](https://genai.owasp.org/resource/agent-control-standard-acs/)。
- 证据边界：版本、profile 和验证状态是检索日期的官方页面状态；规范文档与自我声明不证明某一实现符合、安全或适配所有 Agent 框架。
