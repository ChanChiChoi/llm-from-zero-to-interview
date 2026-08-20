# 第五章：企业级 LLM 应用

## 0. 本章范围与资料

本章回答一个企业落地问题：怎样把模型能力嵌入真实组织的身份、数据、流程和责任体系，而不是只搭一个能聊天的页面。正文会从典型应用进入系统边界，再说明权限、租户隔离、数据治理、审计、上线、组织协作和评估如何共同决定一个应用能否长期运行。

资料核验主要参考 OpenAI 企业隐私与安全资料入口、NIST AI Risk Management Framework 及生成式 AI Profile、OWASP LLM Top 10、Microsoft RBAC 与 Zero Trust 文档，以及可靠性工程中关于 SLI、SLO 和尾延迟的通用实践。OpenAI 企业页面本轮访问受限，因此这里只把它作为资料入口，不把未能直接读取的产品承诺写成事实；OWASP、NIST 和 Microsoft 的公开页面可支持相应的风险、治理和身份控制口径。

本章讨论的是企业级应用的系统边界，不替代企业安全架构、法务合规审查、采购合同、真实 IAM 设计或行业监管要求。隐私治理、RAG 产品实现和具体场景会在相应专题中分别展开；这里先建立一个统一判断：企业级应用不能只做聊天入口，权限、租户隔离、数据治理、审计、SLO、人审和业务指标必须共同进入上线条件。

企业级 LLM 应用和个人消费级应用有很大区别。个人用户更关注好不好用、有不有趣；企业更关注能否接入现有系统、是否符合权限和合规要求、是否能节省成本、是否能稳定服务多人协作和复杂流程。企业级应用不是简单加一个聊天框，而是把大模型嵌入真实业务流程。

本章系统讲企业级 LLM 应用：客服、知识库、代码助手、数据分析、办公自动化、行业助手等典型场景，以及企业落地中的权限、系统集成、数据治理、工作流、审计、运维、ROI 和评估。示例和评分表都是教学构造，真实上线仍需结合组织制度、合同、行业监管、数据流和生产账单审查。

## 5.1 企业级应用的特点

企业级 LLM 应用通常有这些特点：

1. 数据来自企业内部。
2. 权限和合规要求高。
3. 需要接入现有系统。
4. 用户角色复杂。
5. 任务结果影响业务流程。
6. 要有审计和监控。
7. 需要可衡量 ROI。
8. 不能只靠 demo 效果。

企业级应用的判断重点不是“模型能不能回答”，而是“谁可以在什么条件下看到什么数据、触发什么动作，以及发生错误后由谁接管”。因此，企业级设计应同时写清身份来源、数据边界、动作权限、结果验收、人工接管、审计字段和业务主指标。缺少其中任何一项，系统可能仍然适合内部探索，但不能把探索结果直接当作生产能力。

## 5.2 企业知识库问答

企业知识库是最常见的 LLM 应用。

目标：让员工快速找到内部制度、产品文档、技术文档、流程说明和历史经验。

关键能力：

1. 文档解析。
2. 权限过滤。
3. 语义检索。
4. 答案生成。
5. 引用来源。
6. 多轮追问。
7. 反馈纠错。

常见问题：

1. 文档过期。
2. 权限复杂。
3. 召回不准。
4. 引用错误。
5. 用户问法和文档表达不一致。
6. 缺少标准答案评估。

知识库问答的成败，很大程度取决于数据治理和权限设计，而不只是模型能力。一个请求至少经过“身份解析 -> 权限过滤 -> 检索 -> 证据组装 -> 生成 -> 引用展示”几个阶段。权限过滤越晚，越可能在检索缓存、日志或上下文中短暂暴露越权内容；过滤越早，又需要检索系统理解文档 ACL、部门、项目和版本信息。工程上应记录每个证据片段的来源、版本、可见范围和有效期，并把这些字段带入引用检查。

知识库项目还需要区分三类失败：没有召回所需证据，召回了错误或过期证据，以及证据正确但模型解释错误。三类问题的修复路径不同，不能都归因于“模型需要更强”。前两类主要检查解析、元数据、权限和检索；第三类才需要调整提示、模型、结构化输出或人工复核。评估集应同时包含可回答问题、无答案问题、权限边界问题和版本冲突问题。

## 5.3 智能客服

智能客服是高频场景。

价值来源：

1. 降低人工客服压力。
2. 提高自助解决率。
3. 缩短响应时间。
4. 提升回复一致性。
5. 总结用户问题。
6. 辅助人工客服。

客服场景要关注：

1. 意图识别。
2. 知识库检索。
3. 多轮澄清。
4. 转人工策略。
5. 情绪识别。
6. 合规话术。
7. 用户满意度。

客服机器人不能为了减少转人工而强行回答。无法确认时及时转人工，反而更能保护体验。转人工也不是简单的失败计数：应记录转人工原因、用户是否接受、人工接管后的处理时长、是否重复收集信息，以及机器人是否提供了足够的上下文。这样才能判断转人工率上升是模型退化，还是系统更诚实地暴露了高风险问题。

客服系统的主指标通常是“有效解决率”而不是“自动回复率”。一个自动回复但用户再次提问、投诉或最终转人工的会话，不能算成功解决。评估还要按意图、语言、客户等级、渠道和高风险主题切片，否则平均满意度可能掩盖少数严重错误。

## 5.4 客服 Copilot 和全自动客服

客服场景有两种形态。

客服 Copilot：

1. 给人工客服推荐答案。
2. 总结用户历史。
3. 提醒风险话术。
4. 自动生成工单摘要。

全自动客服：

1. 直接面对用户回答。
2. 自动处理简单问题。
3. 必要时转人工。

早期更推荐 Copilot，因为风险低、容易积累数据、人工能纠错。全自动适合边界清楚、知识稳定、风险低的问题。两者的差异不只是 UI：Copilot 的最终动作由人工完成，系统可以把建议、证据和不确定性交给工作人员判断；全自动客服则必须自己承担拒答、升级、状态更新和错误补偿。全自动化前应证明任务边界稳定、转人工链路可用、关键动作可回滚，并保留人工接管。

## 5.5 代码助手

企业代码助手可以帮助：

1. 代码补全。
2. 单元测试生成。
3. 代码解释。
4. Bug 定位。
5. 代码审查。
6. 文档生成。
7. 迁移和重构辅助。

企业代码助手要关注：

1. 代码隐私。
2. 仓库权限。
3. 许可证风险。
4. 安全漏洞。
5. 与 IDE 和 CI 集成。
6. 是否符合团队规范。

代码助手的价值指标可以是开发效率、review 通过率、测试覆盖率、缺陷减少和新人上手速度。但“生成代码行数”不是可靠的价值指标：更多代码可能意味着更多维护负担。应把建议采纳率、编译或测试通过率、回滚率、静态安全问题、许可证检查结果和人工 review 时间放在一起观察。仓库权限也要细到分支、目录和操作类型，读代码、创建补丁、提交合并请求和直接写入生产环境不是同一权限。

## 5.6 数据分析助手

数据分析助手帮助业务人员用自然语言分析数据。

能力包括：

1. 自然语言转 SQL。
2. 指标解释。
3. 图表生成。
4. 异常分析。
5. 报告生成。
6. 数据口径解释。

难点：

1. 指标口径复杂。
2. 数据权限严格。
3. SQL 生成错误可能误导决策。
4. 需要防止查询敏感数据。
5. 需要可追溯计算过程。

数据分析助手不能只生成漂亮图表，还要保证口径、权限和可验证性。自然语言转 SQL 至少要经过指标语义解析、表和字段授权、只读或写入策略、SQL 静态检查、资源限制、执行结果校验和结果解释。模型生成的 SQL 即使语法正确，也可能使用错误时间窗、重复连接或把不同粒度的数据相乘。生产系统应展示查询口径、过滤条件、执行时间和数据版本，并对大表扫描、敏感字段和写操作设置单独的阻断策略。

## 5.7 办公自动化

办公自动化场景包括：

1. 会议纪要。
2. 邮件草稿。
3. 周报生成。
4. 文档润色。
5. PPT 大纲。
6. 表格处理。
7. 日程安排。
8. 工单摘要。

这类场景通常比自动审批和自动扣款容错率高，适合作为企业大模型试点，但“容错率高”不等于可以忽略治理。会议纪要可能包含未公开的商业信息，邮件草稿可能造成错误承诺，日程安排可能暴露客户关系。试点应优先选择可编辑、可撤销、影响范围小的动作，并保留发送、删除、外部共享等不可逆操作的人工确认。

但也要注意：

1. 不要泄露会议敏感信息。
2. 邮件发送前要人工确认。
3. 生成内容要符合企业风格。
4. 结果要可编辑。

## 5.8 行业助手

行业助手针对特定行业。

例如：

1. 金融投研助手。
2. 法律合同助手。
3. 医疗病历助手。
4. 教育备课助手。
5. 制造运维助手。
6. 保险理赔助手。

行业场景价值高，但要求也高：

1. 专业知识准确。
2. 数据合规。
3. 结果可追溯。
4. 风险可控。
5. 需要专家评估。
6. 不能替代最终责任人。

行业助手通常适合先做人机协同，而不是直接全自动决策。行业专家不只是上线前签字的人，还应参与术语定义、风险分级、评估集构造、错误复盘和版本回归。对于医疗、法律、金融等领域，系统应明确它是在检索、起草、提示风险还是作出决策；“辅助”不能成为模糊责任的替代词。最终责任人、证据来源、模型版本和人工修改记录都应进入业务记录。

## 5.9 企业系统集成

企业级 LLM 应用需要接入现有系统。

常见系统：

1. SSO 和身份系统。
2. 权限系统。
3. 文档系统。
4. CRM。
5. ERP。
6. 工单系统。
7. 数据仓库。
8. BI 系统。
9. 代码仓库。
10. 审计系统。

集成难点往往比模型难点更大。没有系统集成，AI 功能很难进入真实工作流。

企业集成至少要分三层看：

1. 身份层：SSO、MFA、SCIM、用户组、服务账号和离职回收。
2. 数据层：文档系统、数据仓库、代码仓库、知识库、向量库和日志系统。
3. 工作流层：工单、审批、CRM、ERP、BI、CI/CD 和人工复核入口。

企业集成应按依赖顺序推进：先确定身份、租户和授权来源，再接入数据和知识库，之后才把模型输出接入工单、CRM、ERP 或审批流。每一个外部动作都要定义输入 schema、调用者身份、允许的资源范围、幂等键、超时、回滚或补偿方式，以及人工接管入口。没有审计、回滚和人工接管时，模型最多提供草稿或建议，不应直接驱动高风险动作。

集成还要区分控制平面和数据平面。控制平面保存模型版本、提示模板、工具注册、权限策略、评估结果和发布记录；数据平面处理用户问题、文档、工具结果和业务状态。模型输出不能直接修改控制平面配置，数据平面中的不可信文档也不能改变工具权限。这个区分能够降低“文档中的指令被模型当成管理员命令”的风险。

## 5.10 权限控制

企业应用必须做权限控制。

需要保证：

1. 用户只能访问有权限的数据。
2. RAG 检索不能越权召回。
3. Agent 工具不能越权调用。
4. 日志不能泄露敏感信息。
5. 不同租户数据隔离。
6. 管理员可审计访问记录。

权限控制不能只在前端做，必须贯穿检索、生成、工具调用和日志。

更具体地说，权限至少要穿过四个环节：

1. 检索权限：向量检索和关键词检索只能召回用户有权看的文档。
2. 生成权限：模型不能把无权信息通过总结、引用或多轮对话泄露出来。
3. 工具权限：Agent 调用工单、数据库、邮件、代码仓库等工具前要做角色和动作验收条件。
4. 日志权限：prompt、检索片段、工具返回和模型输出写入日志前要脱敏并限制查看范围。

企业权限的关键不是“模型知道用户是谁”，而是后端系统在每次检索、工具调用、写日志和展示引用时都重新执行权限判断。权限判断的输入至少包括主体、租户、资源、动作、上下文和策略版本；输出应是允许、拒绝或暂时无法判断，而不是让模型自行猜测。只靠 prompt 告诉模型“不要泄露信息”，不是可靠权限控制。

权限还有时间和状态维度。员工离职、项目结束、文档撤回、客户授权过期后，旧的 embedding、缓存、摘要和对话历史都可能仍然存在。系统需要定义权限变更的传播路径，以及传播完成前如何阻断读取；如果无法确认撤销已经完成，应进入人工处理或拒绝返回，而不是继续使用旧缓存。多租户系统还要验证错误租户 ID、空租户、跨部门共享和管理员代查等边界。

## 5.11 数据治理

企业数据常见问题：

1. 文档过期。
2. 多版本冲突。
3. 命名不统一。
4. 权限不清。
5. 格式复杂。
6. 缺少元数据。
7. 没有标准答案。
8. 敏感信息混杂。

大模型不能自动解决数据治理问题。相反，数据治理差会放大模型幻觉和错误引用。进入知识库前至少要保留来源、所有者、创建和更新时间、版本、适用范围、权限标签、保留期限和删除状态。解析失败、版本冲突、无法确认所有者或缺少 ACL 的文档不能默认为公开内容；可以进入隔离区等待处理，但不能直接用于生产回答。

数据治理要形成可追溯的生命周期：采集、授权、脱敏、解析、索引、更新、下线和删除。删除不只是从主表删一行，还要考虑向量索引、全文索引、缓存、摘要、训练样本、日志和导出的报表。企业应记录删除请求的传播状态和验证结果，无法证明删除已传播到所有副本时，系统应报告未知并限制后续使用。

## 5.12 审计与合规

企业级应用需要审计。

审计内容：

1. 谁访问了什么数据。
2. 模型回答了什么。
3. 使用了哪些文档。
4. 调用了哪些工具。
5. 是否涉及敏感信息。
6. 是否有人审。
7. 是否发生异常。

合规要求因行业不同而不同。金融、医疗、法律和政企场景通常要求更严格。审计记录不能只是“模型回答了一句话”，而应能关联一次请求的身份、权限决策、输入数据摘要、检索证据 ID 与版本、模型和提示版本、工具参数与结果、人工操作、最终状态和错误原因。敏感原文不一定应该写入普通日志，可以使用脱敏字段、哈希、受控引用和分级访问；但过度脱敏又可能让事故无法复盘，因此审计设计要同时满足最小暴露和可追溯性。

审计系统还要考虑完整性和重放。事件应有时间、序号或关联 ID，写入应尽量不可事后静默修改；对于工具写操作，需要记录提交前的预览、确认者和幂等键。日志本身也属于敏感数据，必须有保留期限、访问审计和删除策略。审计覆盖率高不等于事件内容可信，生产系统还要抽样检查字段完整性、时间顺序和跨服务关联是否成立。

## 5.13 上线策略

企业 LLM 应用不建议一次全量上线。

推荐路径：

1. 内部小范围试点。
2. 人机协同模式。
3. 收集反馈和失败样本。
4. 建立评估集。
5. 灰度扩大范围。
6. 再考虑自动化比例提升。

上线策略要和风险等级匹配。高风险场景需要更小的试点范围、更严格的人工复核、更完整的回滚和更长的观察窗口。灰度不仅按用户百分比切流，也可以按任务类型、租户、数据敏感等级、模型版本和工具权限切片。每次扩大范围前，应比较基线与候选版本的任务成功率、错误率、采用率、p95 延迟、单位成本、权限违规和人工接管率；任何关键安全或数据隔离问题都应触发暂停或回滚，而不能用平均分数掩盖。

发布还需要定义“停止”和“回滚”条件。模型版本回滚并不一定能撤回已经写入 CRM、发送的邮件或更新的业务状态，因此不可逆动作应有补偿流程。灰度期间产生的反馈、失败样本和人工修改必须带上版本与租户信息，避免把不同版本混在一个评估集里。

## 5.14 组织协作

企业级 LLM 项目需要多方协作。

参与方：

1. 业务团队。
2. 产品团队。
3. 算法团队。
4. 工程团队。
5. 数据团队。
6. 安全团队。
7. 法务合规。
8. 运维团队。

算法团队不能独立完成企业级落地。很多关键问题在数据、流程、权限和组织协作中。业务团队负责定义任务价值、责任人和不可接受的错误；产品团队负责工作流、用户反馈和采用；算法团队负责模型、检索、提示和评估；工程团队负责服务、容量、发布和回滚；数据团队负责来源、质量和生命周期；安全与法务负责威胁、合规和合同边界；运维团队负责监控、告警和事故响应。责任划分不是官僚流程，而是为了让一个失败样本有明确的归属和修复路径。

项目还应设置变更评审：模型、提示、索引、工具 schema、权限策略和业务规则任何一项变化，都可能改变行为。轻微的模板修改可能影响引用和成本，权限策略修改可能影响数据隔离，工具 schema 修改可能影响副作用。版本号和变更记录应贯穿这些对象，而不是只给模型打版本。

## 5.15 企业应用评估指标

常见指标：

1. 使用率。
2. 任务完成率。
3. 用户满意度。
4. 人工节省时间。
5. 转人工率。
6. 答案正确率。
7. 引用准确率。
8. 响应延迟。
9. 单次任务成本。
10. 安全事件数。
11. 业务指标改善。

不同应用要选不同主指标。知识库看有效解决率和引用支持率；客服看自助解决率和满意度；代码助手看建议采纳率、测试通过率和回滚率。指标必须和任务、用户、版本、租户及风险切片绑定；一个脱离分母和样本来源的百分比不能直接用于发布决策。

### 5.15.1 关键公式与企业级应用指标速查

可以把一个企业 LLM 应用样本写成：

```math
e_i=(u_i,d_i,p_i,\tau_i,w_i,a_i,r_i,l_i,m_i)
```

其中 `u_i` 是用户和角色集合，`d_i` 是接入的数据源，`p_i` 是权限策略，`\tau_i` 是租户或组织边界，`w_i` 是接入的业务工作流，`a_i` 是审计记录，`r_i` 是风险等级，`l_i` 是延迟观测，`m_i` 是业务指标。这里的集合可以为空，但空集合表示“不适用”或“没有观测对象”，不能自动解释成完美通过。

**1. 权限通过率**

企业应用最核心的检查之一，是检索、工具和日志都没有越权。设 $N_{\mathrm{perm}}$ 是实际要求同时进行三层权限检查的请求数：

```math
R_{\mathrm{perm}}=\frac{1}{N_{\mathrm{perm}}}\sum_{i=1}^{N_{\mathrm{perm}}}h_i^{\mathrm{rag}}h_i^{\mathrm{tool}}h_i^{\mathrm{log}}, \qquad N_{\mathrm{perm}}>0
```

其中 `h_i` 是 0/1 指标。`h_i^{\mathrm{rag}}=1` 表示 RAG 检索没有越权召回；`h_i^{\mathrm{tool}}=1` 表示工具调用通过角色和动作校验；`h_i^{\mathrm{log}}=1` 表示日志没有泄露超出查看者权限的数据。三层只要有一层失败，这次请求就不能算权限通过。如果没有需要三层检查的请求，指标为“不适用”；如果检查记录缺失，指标为 `unknown`，不能填入 0 或 1。

**2. 租户隔离违规率**

多租户企业产品要单独监控跨租户数据泄露。设 $N_{\mathrm{tenant}}$ 是需要租户边界检查的请求数：

```math
R_{\mathrm{viol}}=\frac{1}{N_{\mathrm{tenant}}}\sum_{i=1}^{N_{\mathrm{tenant}}}v_i, \qquad N_{\mathrm{tenant}}>0
```

其中 `v_i=1` 表示第 `i` 次请求发生跨租户、跨部门或跨项目边界的违规访问。这个指标通常应接近 0；哪怕平均任务成功率很高，只要出现租户隔离事故，也不能直接上线。如果产品确实是单租户且没有租户边界，这一指标是不适用；不能用“没有发生过事故”替代“已经验证过隔离”。

**3. 引用支持率**

企业知识库和行业助手不能只要求“回答像真的”，还要要求关键结论能被授权证据支持。设 $N_{\mathrm{claim}}$ 是评估集中明确要求引用的关键结论数：

```math
C_{\mathrm{cite}}=\frac{1}{N_{\mathrm{claim}}}\sum_{i=1}^{N_{\mathrm{claim}}}c_i, \qquad N_{\mathrm{claim}}>0
```

其中 `c_i=1` 表示第 `i` 个关键结论的引用存在、相关、足以支持结论、未过期，并且当前用户有权查看。没有需要引用的结论时，指标是不适用；有结论但引用核验字段缺失时，指标为 `unknown`。引用支持率低时，产品应该更保守地展示“不确定”“需要人工复核”或触发知识库更新。

**4. 审计覆盖率**

企业应用需要能回答“谁在什么时候用什么身份访问了什么数据、调用了什么工具、得到什么结果”。设 $N_{\mathrm{audit}}$ 是需要审计的请求或动作数：

```math
C_{\mathrm{audit}}=\frac{1}{N_{\mathrm{audit}}}\sum_{i=1}^{N_{\mathrm{audit}}}a_i, \qquad N_{\mathrm{audit}}>0
```

其中 `a_i=1` 表示请求 trace 中包含用户身份、数据来源、引用文档、工具调用、权限判断、模型版本、输出摘要和人工复核状态。没有审计覆盖，事故发生后无法复盘，也无法证明系统按权限和流程运行；没有可审计请求时不适用，缺失 trace 时为 `unknown`。

**5. 数据新鲜度通过率**

企业知识库常见问题是文档过期。设 $N_{\mathrm{doc}}$ 是本次回答实际使用且有版本信息的证据文档数，可以用文档年龄检查：

```math
C_{\mathrm{fresh}}=\frac{1}{N_{\mathrm{doc}}}\sum_{i=1}^{N_{\mathrm{doc}}}\mathbf{1}[g_i \leq A_{\max}], \qquad N_{\mathrm{doc}}>0
```

其中 `g_i` 是证据文档距当前时间的年龄，`A_{\max}` 是业务允许的最大文档年龄。不同业务阈值不同：报销制度可能按月更新，接口文档可能按版本更新，合规政策可能要求更严格的版本控制。没有使用文档时不适用，文档缺少可信更新时间时为 `unknown`，不能把未知年龄当成最新。

**6. SLO 通过率**

企业应用不仅要准，还要在业务流程可接受的时间内返回。设 $N_{\mathrm{slo}}$ 是有完整延迟记录的请求数，并明确超时请求也要进入分母：

```math
R_{\mathrm{slo}}=\frac{1}{N_{\mathrm{slo}}}\sum_{i=1}^{N_{\mathrm{slo}}}\mathbf{1}[l_i \leq L_{\mathrm{slo}}], \qquad N_{\mathrm{slo}}>0
```

其中 `l_i` 是请求延迟，`L_{\mathrm{slo}}` 是场景定义的延迟目标。客服助手、代码补全、会议纪要、合同复核的 SLO 不一样，不能共用一个平均响应时间。没有请求时不适用，存在请求但延迟记录缺失时为 `unknown`；p95 也不能用已成功返回的请求选择性计算而排除超时。

**7. 高风险人审覆盖率**

对金融、医疗、法律、合规、退款、权限变更等高风险任务，要看高风险样本是否有人审。设 $N_{\mathrm{high}}$ 是高风险任务数：

```math
C_{\mathrm{human}}=\frac{1}{N_{\mathrm{high}}}\sum_{i=1}^{N_{\mathrm{high}}}q_i, \qquad N_{\mathrm{high}}>0
```

其中 `q_i=1` 表示有人审、审批或二次确认。没有高风险任务时，指标是不适用；风险标签缺失时，不能把任务排除在分母外，而应报告 `unknown`。高风险任务不是不能用 LLM，而是不能跳过责任人、审批和审计。

**8. 企业就绪分与上线条件**

一个简化企业就绪分可以写成：

```math
S_{\mathrm{ent}}=0.25R_{\mathrm{perm}}+0.15(1-R_{\mathrm{viol}})+0.15C_{\mathrm{cite}}+0.15C_{\mathrm{audit}}+0.10C_{\mathrm{fresh}}+0.10R_{\mathrm{slo}}+0.10C_{\mathrm{human}}
```

这个分数只用于教学 demo 解释，真实项目要按行业和风险重新定权重。它不能抵消一次严重的权限或审计失败。与其把所有维度压成一个分数，不如同时保留关键条件的状态：`passed` 表示有足够证据且达到目标，`failed` 表示有证据但未达到目标，`unknown` 表示样本、字段或分母不足。

```math
G_{\mathrm{ent}}=\mathbf{1}[R_{\mathrm{perm}}\geq 0.95]\mathbf{1}[R_{\mathrm{viol}}=0]\mathbf{1}[C_{\mathrm{audit}}\geq 0.90]\mathbf{1}[R_{\mathrm{slo}}\geq 0.95]\mathbf{1}[M_{\mathrm{biz}}=1]
```

其中 `M_{\mathrm{biz}}=1` 表示已定义业务主指标。若任一关键指标为 `unknown`，整体决策也应为 `unknown`；不能把未知当作 0 或 1。企业应用不是靠一个综合分上线，而是关键权限、隔离、审计、延迟和业务指标都必须有可追溯证据。

## 5.16 常见失败模式

1. 只做聊天框，没有接入工作流。
2. 知识库数据质量差。
3. 权限控制不完整。
4. 缺少业务指标。
5. 用户不知道怎么用。
6. 模型回答无法追溯。
7. 成本没有监控。
8. 高风险场景没有人审。
9. 试点成功但无法规模化。
10. 组织协作不顺。

企业级 LLM 应用失败往往不是因为模型完全不行，而是因为没有解决企业环境中的系统问题。失败复盘应沿着“输入数据 -> 权限决策 -> 检索证据 -> 模型输出 -> 工具动作 -> 人工接管 -> 业务结果”逐段定位，而不是只看最后一句回答。每个失败都要标记影响范围、是否可逆、责任系统、补救动作和是否需要扩大评估集。

## 5.17 企业应用架构复盘

评审一个企业级应用时，可以从五条链路展开。第一条是身份链：用户、服务账号、租户和角色从哪里来，离职或授权撤销如何传播。第二条是数据链：数据来源、版本、ACL、脱敏、索引、缓存和删除如何关联。第三条是推理链：模型看到哪些经过授权的证据，输出如何引用，哪些字段必须结构化。第四条是动作链：工具允许哪些动作，是否只读，是否需要确认，失败后如何重试或补偿。第五条是责任链：谁定义业务指标，谁批准高风险动作，谁响应事故，谁维护评估集。

这五条链路应在同一个 trace ID 下关联起来。若只能看到模型文本而看不到授权、证据、工具和人工状态，就无法判断问题来自模型还是系统边界，也无法为审计和事故响应提供足够证据。

## 5.18 企业知识库落地决策

知识库项目可以按四个阶段推进。第一阶段整理来源、版本、所有者、权限和删除状态，先解决“哪些内容可以被使用”。第二阶段建立解析、分块、索引、检索和引用评估，区分召回失败、证据失效和生成错误。第三阶段在低风险任务中以 Copilot 形式试点，让人工修改和拒答样本进入评估集。第四阶段才把稳定、可回滚、边界清晰的结果接入工作流，并为每个工具动作设置授权、确认和补偿逻辑。

试点报告不应只列答案正确率，还应包括权限违规、引用支持、过期文档、无法回答率、人工接管、p95 延迟、单位任务成本和业务主指标。任何关键指标无数据时，结论应保持 `unknown`，而不是用“暂时没有投诉”替代验证。

## 5.19 最小可运行企业级应用审计 demo

下面这个 demo 用 0 依赖 Python 模拟企业级 LLM 应用审计。它不是生产治理系统，而是把身份、权限、租户隔离、RAG 权限过滤、工具权限、审计日志、PII 脱敏、数据新鲜度、引用支持、SSO、工作流、评估、反馈、SLO、业务指标和高风险人审拆成可检查字段。代码中的阈值是教学参数，不代表任何行业标准。

```python
from math import isfinite


UNKNOWN = "unknown"


REQUIRED_FIELDS = {
    "name", "scenario_type", "users", "data_sources", "permission_coverage",
    "tenant_isolation", "rag_permission_filter", "tool_permission_gate",
    "audit_log_coverage", "pii_redaction", "data_freshness", "citation_support",
    "sso_integration", "workflow_integration", "eval_ready", "feedback_loop",
    "sla_p95_latency_ok", "business_metric_defined", "human_review_coverage",
    "risk_level",
}


def mean(values):
    return UNKNOWN if not values else sum(values) / len(values)


def validate_app(app):
    missing = sorted(REQUIRED_FIELDS - app.keys())
    if missing:
        return [f"missing:{name}" for name in missing]
    errors = []
    for key in ("name", "scenario_type"):
        if not isinstance(app[key], str) or not app[key].strip():
            errors.append(f"invalid:{key}")
    for key in ("users", "data_sources"):
        value = app[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"invalid:{key}")
    for key in REQUIRED_FIELDS - {"name", "scenario_type", "users", "data_sources", "risk_level"}:
        value = app[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"not_numeric:{key}")
        elif not isfinite(value) or not 0 <= value <= 1:
            errors.append(f"out_of_range:{key}")
    if app["risk_level"] not in {"low", "medium", "high"}:
        errors.append("invalid:risk_level")
    return errors


def audit_app(app):
    errors = validate_app(app)
    if errors:
        return {
            "name": app.get("name", "<unknown>"),
            "status": UNKNOWN,
        "status": UNKNOWN,
            "errors": errors,
        }
    high_risk = app["risk_level"] == "high"
    gates = {
        "permission_coverage": app["permission_coverage"] >= 0.90,
        "tenant_isolation": app["tenant_isolation"] >= 1.00,
        "rag_permission_filter": app["rag_permission_filter"] >= 0.95,
        "tool_permission_gate": app["tool_permission_gate"] >= 0.90,
        "audit_log_coverage": app["audit_log_coverage"] >= 0.90,
        "pii_redaction": app["pii_redaction"] >= 0.85,
        "data_freshness": app["data_freshness"] >= 0.75,
        "citation_support": app["citation_support"] >= 0.80,
        "sso_integration": app["sso_integration"] >= 1.00,
        "workflow_integration": app["workflow_integration"] >= 0.75,
        "eval_ready": app["eval_ready"] >= 0.80,
        "feedback_loop": app["feedback_loop"] >= 0.75,
        "sla_p95_latency_ok": app["sla_p95_latency_ok"] >= 1.00,
        "business_metric_defined": app["business_metric_defined"] >= 1.00,
        "human_review_coverage": (not high_risk) or app["human_review_coverage"] >= 0.80,
    }

    permission_score = mean([
        app["permission_coverage"],
        app["rag_permission_filter"],
        app["tool_permission_gate"],
    ])
    governance_score = mean([
        app["tenant_isolation"],
        app["audit_log_coverage"],
        app["pii_redaction"],
        app["data_freshness"],
    ])
    evidence_score = mean([
        app["citation_support"],
        app["eval_ready"],
        app["feedback_loop"],
        app["business_metric_defined"],
    ])
    integration_score = mean([
        app["sso_integration"],
        app["workflow_integration"],
    ])
    ops_score = mean([
        app["sla_p95_latency_ok"],
        app["human_review_coverage"] if high_risk else 1.0,
    ])

    enterprise_score = (
        0.30 * permission_score
        + 0.22 * governance_score
        + 0.18 * evidence_score
        + 0.15 * integration_score
        + 0.15 * ops_score
    )
    failed_gates = [name for name, ok in gates.items() if not ok]
    status = "passed" if not failed_gates else "failed"
    return {
        "name": app["name"],
        "scenario_type": app["scenario_type"],
        "status": status,
        "enterprise_score": round(enterprise_score, 3),
        "status": status,
        "failed_gates": failed_gates,
    }


apps = [
    {
        "name": "support_kb_rag",
        "scenario_type": "enterprise_knowledge_base",
        "users": 1800,
        "data_sources": 7,
        "permission_coverage": 0.96,
        "tenant_isolation": 1.00,
        "rag_permission_filter": 0.98,
        "tool_permission_gate": 0.94,
        "audit_log_coverage": 0.93,
        "pii_redaction": 0.91,
        "data_freshness": 0.88,
        "citation_support": 0.90,
        "sso_integration": 1.00,
        "workflow_integration": 0.86,
        "eval_ready": 0.84,
        "feedback_loop": 0.78,
        "sla_p95_latency_ok": 1.00,
        "business_metric_defined": 1.00,
        "human_review_coverage": 0.82,
        "risk_level": "high",
    },
    {
        "name": "contract_copilot",
        "scenario_type": "legal_assistant",
        "users": 120,
        "data_sources": 5,
        "permission_coverage": 0.91,
        "tenant_isolation": 1.00,
        "rag_permission_filter": 0.94,
        "tool_permission_gate": 0.88,
        "audit_log_coverage": 0.90,
        "pii_redaction": 0.86,
        "data_freshness": 0.70,
        "citation_support": 0.82,
        "sso_integration": 1.00,
        "workflow_integration": 0.72,
        "eval_ready": 0.80,
        "feedback_loop": 0.62,
        "sla_p95_latency_ok": 1.00,
        "business_metric_defined": 1.00,
        "human_review_coverage": 0.71,
        "risk_level": "high",
    },
    {
        "name": "data_analyst_nl2sql",
        "scenario_type": "data_analysis",
        "users": 260,
        "data_sources": 9,
        "permission_coverage": 0.86,
        "tenant_isolation": 1.00,
        "rag_permission_filter": 0.90,
        "tool_permission_gate": 0.73,
        "audit_log_coverage": 0.82,
        "pii_redaction": 0.79,
        "data_freshness": 0.93,
        "citation_support": 0.55,
        "sso_integration": 1.00,
        "workflow_integration": 0.84,
        "eval_ready": 0.72,
        "feedback_loop": 0.64,
        "sla_p95_latency_ok": 0.00,
        "business_metric_defined": 1.00,
        "human_review_coverage": 0.76,
        "risk_level": "high",
    },
    {
        "name": "office_summarizer",
        "scenario_type": "office_automation",
        "users": 2300,
        "data_sources": 4,
        "permission_coverage": 0.78,
        "tenant_isolation": 0.95,
        "rag_permission_filter": 0.80,
        "tool_permission_gate": 0.65,
        "audit_log_coverage": 0.70,
        "pii_redaction": 0.68,
        "data_freshness": 0.82,
        "citation_support": 0.45,
        "sso_integration": 0.70,
        "workflow_integration": 0.76,
        "eval_ready": 0.62,
        "feedback_loop": 0.50,
        "sla_p95_latency_ok": 1.00,
        "business_metric_defined": 0.60,
        "human_review_coverage": 0.70,
        "risk_level": "medium",
    },
    {
        "name": "generic_chat_portal",
        "scenario_type": "generic_chatbot",
        "users": 900,
        "data_sources": 2,
        "permission_coverage": 0.52,
        "tenant_isolation": 0.70,
        "rag_permission_filter": 0.40,
        "tool_permission_gate": 0.30,
        "audit_log_coverage": 0.35,
        "pii_redaction": 0.42,
        "data_freshness": 0.50,
        "citation_support": 0.20,
        "sso_integration": 0.00,
        "workflow_integration": 0.20,
        "eval_ready": 0.30,
        "feedback_loop": 0.25,
        "sla_p95_latency_ok": 0.00,
        "business_metric_defined": 0.00,
        "human_review_coverage": 0.20,
        "risk_level": "medium",
    },
]

results = [audit_app(app) for app in apps]
ranked = sorted(
    [
        (r["name"], r.get("enterprise_score", UNKNOWN), r["status"])
        for r in results
    ],
    key=lambda item: item[1] if isinstance(item[1], (int, float)) else float("-inf"),
    reverse=True,
)
passed = [r["name"] for r in results if r["status"] == "passed"]
needs_rework = {
    r["name"]: r.get("failed_gates", r.get("errors", []))
    for r in results
    if r["status"] != "passed"
}

print("ranked=", ranked)
print("enterprise_pass=", passed)
print("needs_rework=", needs_rework)


empty_app = audit_app({})
nan_app = dict(apps[0])
nan_app["citation_support"] = float("nan")
missing_risk = dict(apps[0])
missing_risk.pop("risk_level")
assert empty_app["status"] == UNKNOWN
assert audit_app(nan_app)["status"] == UNKNOWN
assert audit_app(missing_risk)["status"] == UNKNOWN
assert all(result["status"] in {"passed", "failed"} for result in results)
print("boundary_status=", {
    "empty": empty_app["status"],
    "nan": audit_app(nan_app)["status"],
    "missing_risk": audit_app(missing_risk)["status"],
})
```

一组典型输出是：

```text
ranked= [('support_kb_rag', 0.927, 'passed'), ('contract_copilot', 0.866, 'failed'), ('data_analyst_nl2sql', 0.77, 'failed'), ('office_summarizer', 0.753, 'failed'), ('generic_chat_portal', 0.354, 'failed')]
enterprise_pass= ['support_kb_rag']
needs_rework= {'contract_copilot': ['rag_permission_filter', 'tool_permission_gate', 'data_freshness', 'workflow_integration', 'feedback_loop', 'human_review_coverage'], 'data_analyst_nl2sql': ['permission_coverage', 'rag_permission_filter', 'tool_permission_gate', 'audit_log_coverage', 'pii_redaction', 'citation_support', 'eval_ready', 'feedback_loop', 'sla_p95_latency_ok', 'human_review_coverage'], 'office_summarizer': ['permission_coverage', 'tenant_isolation', 'rag_permission_filter', 'tool_permission_gate', 'audit_log_coverage', 'pii_redaction', 'citation_support', 'sso_integration', 'eval_ready', 'feedback_loop', 'business_metric_defined'], 'generic_chat_portal': ['permission_coverage', 'tenant_isolation', 'rag_permission_filter', 'tool_permission_gate', 'audit_log_coverage', 'pii_redaction', 'data_freshness', 'citation_support', 'sso_integration', 'workflow_integration', 'eval_ready', 'feedback_loop', 'sla_p95_latency_ok', 'business_metric_defined']}
```

这个 demo 的重点不是分数本身，而是审计口径：企业级应用必须把权限、租户隔离、审计、PII、引用、SLO、业务指标和人审放进同一张表。`contract_copilot` 分数不低，但因为法律场景高风险、人审和反馈闭环不足，不能直接全自动运行；`data_analyst_nl2sql` 工作流接入不错，但 SQL 工具权限、引用支持和延迟条件仍未满足；`generic_chat_portal` 则说明“通用聊天入口”如果没有身份、权限、指标和工作流，只能算内部 demo。空应用、缺失风险标签和非有限指标都返回 `unknown`，不会被排序或评分逻辑误当作通过。

## 5.20 资料入口与证据边界

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：NIST 的风险管理框架，可支持 govern、map、measure、manage 的治理闭环；它不替代具体组织的安全验收。
- [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)：NIST 的生成式 AI 风险画像，可支持数据、生成内容、评估和治理边界；实际风险概率和损失仍需结合业务数据。
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)：社区安全项目，可支持提示注入、敏感信息泄露、过度代理和资源消耗等应用层威胁分类；它不是某个产品的安全证明。
- [Microsoft RBAC 文档](https://learn.microsoft.com/en-us/azure/role-based-access-control/overview)：官方身份与访问控制文档，可支持角色、主体、资源和动作授权的基础概念；企业实际策略仍需按租户和数据分类设计。
- [Microsoft Zero Trust 概览](https://learn.microsoft.com/en-us/security/zero-trust/zero-trust-overview)：官方安全架构入口，可支持持续验证、最小权限和假设已被攻破等原则；它不自动完成 LLM 数据流审查。
- [OpenAI 企业隐私与安全入口](https://openai.com/enterprise-privacy/)：供应商官方资料入口；本轮页面访问受限，因此正文没有把其未直接核验的产品承诺写成事实，真实采购应以当前合同和安全问卷为准。

本章的企业评分、阈值、应用字段和 Python 审计是教学构造，不能替代 IAM 测试、渗透测试、数据保护影响评估、行业合规审查或真实 SLO 报告。`passed` 只表示示例字段满足示例条件，`failed` 表示已有证据但未满足条件，`unknown` 表示数据或证据不足；未知状态必须保留到项目决策层。

## 5.21 本章小结

企业级 LLM 应用的难点在于真实环境：数据复杂、权限严格、系统多、流程长、用户角色多、合规要求高。客服、知识库、代码助手、数据分析、办公自动化和行业助手都是常见场景，但每个场景都需要结合业务流程和风险等级设计。

下一章会进入 RAG 产品落地，深入讨论企业知识库和检索增强生成系统如何从技术方案变成可用产品。
