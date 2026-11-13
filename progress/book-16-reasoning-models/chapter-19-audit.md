# 第十九章审计：交错推理与工具协议

审阅日期：2026-11-13
整册连续复读：2026-11-13

## 范围与章级扫描

- **范围：** 解释模型推理与工具调用、外部 observation、状态更新、验证和恢复如何构成一条可审计的交错循环。provider/API 消息合同、host 授权、executor 副作用和任务结果分别说明；完整 Agent Runtime 与 MCP 工具生态链接到第十七、二十、二十二册。
- **章级扫描：** 2026-11-13 定向核查 OpenAI Responses Function Calling、Anthropic Messages 工具往返与 thinking block 回放、Model Context Protocol 2026-07-28 规范及 Tasks 扩展、OWASP 2025 LLM 风险分类、HTTP 重试与幂等资料，并复看 ReAct、Toolformer 的研究定位。官方接口事实绑定文档/API 版本；来源阅读范围、支持事实与证据边界见[工具调用、安全与协议研究记录](../../research/agent-tool-calling-security-2026.md)。
- **候选主题去向：**
  - Responses 的 `function_call`/`function_call_output` 与 `call_id`、Messages 的 `tool_use`/`tool_result` 与 `tool_use_id`：纳入 §19.5 对照表，只描述各自公开消息合同，不推为跨 provider 标准或业务授权。
  - Anthropic thinking block 回放与 interleaved thinking 可用范围：纳入 provider 合同和资料边界，绑定 Messages API、模型与当前动态文档。
  - MCP 无协议会话核心、能力发现和 Tasks 扩展：纳入 §19.15，绑定 2026-07-28 版本；不声称所有 Host/SDK 均已实现。
  - 工具返回提示注入、过度代理权限、取消和 unknown：纳入 observation 信任边界及状态恢复，OWASP 分类仅作威胁模型入口，不表述为发生率。
  - HTTP 重试、幂等键、side-effect reconciliation：纳入本章事件生命周期与恢复例子；RFC/AWS 资料不替业务动作定义幂等性。
  - ReAct 与 Toolformer 作为推理—行动交错的研究谱系入口；不将论文方法等同于当前生产协议。通用工具执行、授权与长任务 Runtime 分别交第十一/十三章、第二十册和第二十二册。
- **架构决定：** 保留第十九章作为“模型输出—工具执行—观察验证—状态恢复”的连接层，不扩成通用 Agent/Harness 或完整 MCP 教程。现有章节边界和跨册交接成立，无需改变第十六册骨架。

## 正文结构、篇幅与例子

- §§19.1—19.15 构成一个连续核心讲解单元：从一次性生成转入交错循环，经状态与事件合同、observation 信任、代码和研究案例、信息增益与预算、unknown/重试/副作用/恢复，推进至评估、安全和协议边界。§19.7—19.8 是主线案例，不是独立讲义。§19.16 为可独立运行的辅助审计器示例；§19.17 是练习，未计入篇幅；§19.18 为结论与资料边界。
- 按解释性中文计数，排除标题、表格、公式、代码围栏及内容、状态标记、列表、练习和参考资料，连续复读后 §§19.1—19.15 核心正文共 **7,273 汉字**（核心下限 1,800）；§19.16 的说明性正文 **1,101 汉字**（辅助单元下限 1,000）。计数只纳入连贯散文段落，故为保守值；篇幅达标不代替机制与边界审查。
- §19.2 的合成 `refresh_config` 轨迹初始设置 8,000 token、3 次工具调用、20 秒、2 次验证和 0 次副作用预算；6 秒观察到连接数 100/上限 100 后，剩余工具预算降为 2、时间降为 14 秒。证据支持连接池假设但未证明因果，示例因此只调度只读诊断，不直接写配置。
- 预算和信息增益部分给出状态/熵、观察带来的信息增益及剩余资源向量的逐步算例；§19.11 以规范化 JSON 计算 `reserve_stock` 幂等键 SHA-256：`sha256:c3dc203a758f739a20f7240a9d6965d073d8b06b51a96f0c8b09d2d07b7e743c`。例子标明为合成数据，不把演示值当生产发生率。
- §19.14 用 50 个合成任务演示 success/call/cost 指标、成功率和单位成功成本的不同含义；成本项需使用相同币种或归一化口径，成功数为零时单位成功成本未定义。§19.16 进一步演示授权、返回 hash、验证、幂等提交、断线 unknown 和取消后的状态轨迹。

## 公式、代码、证据与链接

- **公式：** 12 个 display-math 围栏块，覆盖 observation 指标、熵/信息增益、预算、幂等 hash、恢复和评估成本等。连续复读将状态字段与转移函数、observation 检查标志、熵分布、预算分量、hash 编码规则、恢复谓词，以及成功率/有效调用率/成本项的变量域、分母和单位前置；例子继续提供数值或状态轨迹。全章 38 个围栏标记配对；`$$` 与 `\operatorname` 均为 0。GitHub MathJax 实际页面渲染尚未验收。
- **代码：** §19.16 唯一 Python 代码块在 2026-11-13 以 CPython 3.12.9 和 Python 标准库验证；从 Markdown 围栏提取后执行，实际 stdout 与正文预期逐行一致。覆盖正常提交、started 后 unknown、未授权启动、错误内容 hash、取消后提交、布尔序号、非字符串事件类型/工具名、完成后追加事件、缺少 `response_completed` 和 NaN hash。无第三方依赖，不访问模型、网络、文件或真实 executor。此次示例输入内容调整后已重跑；后续纯文字边界说明没有改动可执行语句。
- 连续复读只修改正文和公式变量说明；§19.16 可执行语句、输入输出、依赖、环境与预期结果未变化，沿用原验证，不重复运行。
- **证据：** 正文直接链接 OpenAI、Anthropic、MCP、OWASP、IETF 和 AWS 官方资料及 ReAct/Toolformer 原论文；动态 provider 合同绑定 API surface，MCP 合同绑定 2026-07-28 版。事件 hash 只验证对象编码一致，不证明来源可信或语义正确；代码不验签、不查询真实授权/业务账本，也不证明副作用发生。
- **链接：** 24 个 Markdown 行内链接中 7 个本地目标全部存在，17 个外部目标为对应论文、规范或官方资料。简介与目录中的第十九章标题、问题和交接摘要同正文一致；第九、十五、十八章及第二十册、第十一/十三/十八章交叉入口有效。

## 配套同步与遗留风险

- 新版 provider/MCP/OWASP 来源和证据边界已加入 `research/agent-tool-calling-security-2026.md`。`PAPERS.md` 已收录 ReAct、Toolformer、function calling、MCP；`INTERVIEW_BANK.md` 与 `EXERCISES.md` 已覆盖工具事件、幂等、unknown 与恢复；`GLOSSARY_EN_ZH.md` 已定义 Interleaved Thinking、Idempotency Key、Commit-Unknown 等术语；`KNOWLEDGE_GRAPH.md` 已有 executor、安全和恢复链路，因此不重复粘贴条目。
- provider/API 文档会随模型和版本变化，本章未执行真实 API 探测，也未核对每种 MCP SDK 的实现状态；部署仍须按目标版本检查。§19.16 是输入日志合同的教学审计器，不提供事件真实性、授权、分布式事务或生产安全保证。合成预算、成本和成功率不用于业务容量决策。
- 章节门禁完成。第十六册第 1—19 章现已逐章精修，整册连续复读已覆盖第 1—19 章并回看第 17 章；下一入口为第二十章《GLM-5.3 长任务环境与验证器》，之后完成整册最终门禁。按用户要求不更新 PDF 或 `plan.md`。
