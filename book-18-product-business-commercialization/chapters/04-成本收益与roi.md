# 第四章：成本收益与 ROI

## 0. 本章范围与资料

本章回答一个产品化问题：一个大模型功能在什么条件下能够持续创造价值，而不是只在演示中看起来有效。分析会把任务量、采用率、模型调用、检索、工具、人审、固定投入、质量变化、风险和业务收益放到同一套单位经济账中，再用试点数据和敏感性分析检查结论是否稳健。

资料核验主要参考 OpenAI 开发者文档中的 Batch 与 Prompt Caching 说明、OpenAI 的评估与延迟优化资料入口、NIST AI Risk Management Framework 及其生成式 AI 配套文档、FinOps Foundation 的 Framework，以及可靠性工程中关于服务目标和尾延迟的通用实践。这些资料用于确认成本分类、评估闭环和治理边界，不提供本章示例的业务结论。

本章不固化任何长期有效的 API 价格，也不替代企业财务模型、采购合同、税务口径或真实云账单分析。示例中的价格、任务量和人工成本都是教学参数；正式决策必须以当前供应商价目表、账单、实验数据和财务认可的收益口径为准。

大模型项目不能只问“效果好不好”，还要问“值不值得”。如果一个系统每次回答都很准确，但调用成本太高、人工审核太重、延迟太长、收益无法覆盖投入，就很难成为可持续产品。成本收益与 ROI，是大模型产品化绕不开的问题。

本章系统讲大模型项目的成本和收益：token 成本、GPU 成本、工程成本、数据成本、人工审核成本、运维成本；收益侧包括人工替代、效率提升、质量提升、收入增长和风险降低；最后讲 ROI 估算、单位经济账、成本优化和项目复盘。

## 4.1 为什么要算 ROI

ROI（Return on Investment，投资回报率）描述投入相对于成本产生了多少净回报，但它不是“项目好不好”的单一答案。一个项目可能净回报为正，却不满足延迟、质量或安全要求；也可能短期 ROI 不高，却是后续产品能力的必要基础。因此，ROI 必须和任务成功率、用户采用率、可靠性、风险以及一次性投入一起看。

一个大模型 demo 可以不算 ROI，但产品必须算。原因是：

1. 模型调用有持续成本。
2. 用户规模扩大后成本会线性甚至超线性增长。
3. 人工审核和运维不能忽略。
4. 模型效果提升不一定带来业务收益。
5. 业务方需要判断优先级。

一个可执行的判断顺序是：先定义任务和基线，再分别记录收益与成本，最后检查结论对关键假设是否敏感。只有当收益来源可以被观测、成本边界足够完整、质量和安全条件没有被破坏时，ROI 数字才具有决策价值。

## 4.2 成本不只是 Token

很多估算从供应商价目表开始，最后也停在供应商价目表。这样得到的数字通常很小，却没有回答产品真正花了多少钱。一次模型调用只是一个可见的变动成本；围绕它建立的检索、权限、监控、人工复核和故障处理，才决定了系统能否长期运行。

可以把成本分成三层：

1. **单任务变动成本**：每处理一个任务都会发生的模型、检索、工具、人工审核、重试和预期失败成本。
2. **周期固定成本**：无论当月处理多少任务，都需要支付的服务、监控、数据更新、合规和团队维护成本。
3. **一次性投入**：架构开发、数据治理、评估集建设、迁移和培训等投入。一次性投入不应重复计入每个月的固定成本，但应进入回本分析或长期总拥有成本。

完整成本清单通常包括：

1. 模型调用和 embedding 成本。
2. 自部署场景的 GPU、机房、电力和容量冗余成本。
3. RAG 的解析、分块、向量库、检索、reranker 和引用校验成本。
4. Agent 的工具调用、浏览器或代码执行、沙箱和 trace 存储成本。
5. 人工审核、质检、投诉处理和反馈标注成本。
6. 数据清洗、脱敏、更新和授权成本。
7. 工程开发、测试、发布和迁移成本。
8. 监控、日志、告警、备份和事故响应成本。
9. 安全评估、合规审查、审计和供应商管理成本。
10. 错误输出、重试、退款、业务中断和事故处置的预期成本。

成本清单的目的不是把所有数字都精确到小数点后两位，而是防止比较不同方案时漏掉某一类成本。例如 API 方案的 GPU 成本可能接近零，但它可能包含更高的单价、数据传输和供应商锁定成本；自部署方案的单次推理账面价格可能很低，却要承担闲置容量、升级和故障处理成本。只有把两边放在同一时间范围、同一任务口径中，比较才有意义。

## 4.3 Token 成本

Token 成本通常按输入 token 和输出 token 分开计价。输入中可能同时包含系统指令、用户问题、历史对话、检索片段和工具结果；输出则可能包含最终答案、结构化字段、思考过程之外的可见说明或工具参数。不同供应商对缓存命中、批处理、长上下文和工具调用的计费方式并不相同，所以账本需要记录原始 token 数和实际计费 token 数，而不能只记录“调用次数”。

影响因素包括：

1. 系统提示和固定模板的长度。
2. 用户输入和历史上下文的长度。
3. RAG 拼接文档的数量、重复率和截断策略。
4. 输出上限以及实际输出长度。
5. 重试、纠错和多样本采样次数。
6. Agent 的规划、工具调用和结果回读轮数。
7. 缓存命中率、批处理折扣和不同模型的输入输出单价。

一个可审计的估算至少要保留四个量：平均输入 token、平均输出 token、每个任务的模型调用次数，以及这些量的 p50/p95 或分位数。平均值可能掩盖长尾：大多数客服问题很短，但少量长对话或失败重试会消耗大量上下文。工程上应把“平均单次成本”和“高分位任务成本”同时监控。

例如，普通问答可能只有一次短调用；Agentic RAG 可能经历查询改写、召回、重排、阅读、工具查询和最终生成。即使每一轮单价相同，总 token 量也会随着轮数累积。降低 token 成本不能简单截断上下文，还要验证召回覆盖、引用正确率和任务成功率是否下降。

## 4.4 GPU 成本

如果自部署模型，GPU 成本不能只看一张卡每小时的租金。真正要计算的是一段时间内为目标服务保留的容量，以及这些容量在实际负载下被有效利用了多少。

需要考虑：

1. GPU 采购、租用、折旧或云实例费用。
2. 显存容量与 KV cache 对并发数的限制。
3. 预填充和解码阶段不同的吞吐特征。
4. 高峰期冗余、故障转移和低峰期空闲。
5. 量化、批处理、并行和 kernel 优化的工程投入。
6. 机房、电力、网络、镜像、监控和升级成本。
7. 推理平台、值班和模型版本维护人员的成本。

可以用一个粗略的单位成本估算帮助建立直觉：

```text
单位推理成本 = 计费周期内的总基础设施成本 / 同周期内完成的有效任务数
```

“有效任务数”必须有定义。仅仅返回了 HTTP 200 的请求，不一定是有效任务；超时、错误答案、用户拒绝或人工重做的请求，都可能需要从收益侧扣除，或者计入失败成本。自部署不一定比 API 便宜。只有当调用量、模型尺寸、并发结构和团队推理能力共同匹配时，固定基础设施成本才可能被足够多的有效任务摊薄。

## 4.5 RAG 成本

RAG 的成本包括：

1. 文档解析。
2. 分块。
3. embedding。
4. 向量库存储。
5. 检索。
6. reranking。
7. 文档更新。
8. 权限过滤。
9. 引用校验。

RAG 不是免费的“准确率插件”。离线建库会产生解析、分块、embedding、去重和索引更新成本；在线查询会产生过滤、检索、重排、上下文拼接和引用校验成本。文档更新越频繁，增量索引和版本管理越重要；权限越细，检索结果越不能先召回后随意过滤，否则可能在中间日志或缓存中留下越权数据。

因此，RAG 的收益不能只用“检索后回答更像正确答案”描述。需要同时观察召回覆盖率、引用支持率、答案任务成功率、延迟、每次查询的文档量和权限过滤失败率。减少 top-k 或缩短片段可能降低 token 成本，却也可能丢掉回答所需的证据；合并重复片段可能降低成本，却可能破坏文档版本或权限边界。RAG 优化是成本、证据和安全的联合优化。

## 4.6 Agent 成本

Agent 成本通常比普通问答高，因为它把一次生成任务变成了一个带状态的执行过程。来源包括：

1. 多轮模型调用。
2. 工具调用。
3. 搜索和检索。
4. 代码执行或浏览器操作。
5. 失败重试。
6. trace 日志存储。
7. 安全检查。
8. 人工确认。

Agent 适合需要检索、规划、执行和验证的高价值任务，不适合所有简单请求默认开启。一个成熟的路由器会先判断任务是否需要工具和多步执行，再为任务设置模型调用次数、工具预算、最长运行时间和最大输出规模。每个步骤还应记录输入、工具参数、结果、重试原因和最终状态，这样才能解释成本为什么上升。

Agent 的成本还具有相关性：某个工具失败可能触发重试，重试又会带来更多模型调用和上下文；一个错误的规划可能让后续所有步骤都失效。于是“平均每步价格”不能代替“每个完整任务的成功成本”。单位账应记录成功任务、失败任务和部分成功任务的比例，并把失败后的人工接管纳入成本。

## 4.7 人工审核成本

很多大模型产品需要人在回路，尤其是法律、金融、医疗、安全、对外发布和会改变业务状态的场景。人工并不只是一个“审核分钟数”，还包括审核规则、培训、质检和争议处理。

人工审核成本包括：

1. 审核时间。
2. 审核人员培训。
3. 质检流程。
4. 复核争议样本。
5. 处理用户投诉。
6. 标注反馈数据。

如果模型节省了 5 分钟，但审核要花 4 分钟，实际收益就很有限；如果模型把错误整理得更像正确答案，审核人员还可能需要额外核查来源。人机协同的目标不是把所有风险都交给人工，而是让系统在低风险、可验证的部分自动完成，把人工注意力集中到高风险字段、异常样本和不可逆动作。

审核成本还会随通过率变化。模型质量下降时，人工接管比例和每单审核时长可能同时上升，形成“质量下降 -> 人审增加 -> 延迟和成本上升”的反馈。估算时应分别记录平均审核时间、审核覆盖率、复核比例和人工改写比例，不能用一个固定的审核单价掩盖这些变化。

## 4.8 工程和运维成本

产品上线后需要持续维护。

工程成本包括：

1. 前端和后端开发。
2. 模型服务接入。
3. 数据管道。
4. 权限系统。
5. 监控告警。
6. 日志和审计。
7. 回归评估。
8. 灰度发布。
9. 线上问题排查。

大模型产品不是接一个 API 就结束。越接近企业级应用，工程和运维成本越重要。特别是模型、提示词、检索索引和工具接口都会变化，系统需要回归集、版本记录、灰度发布和回滚能力；否则一次看似便宜的升级可能把错误率、人工接管和投诉成本一起推高。

固定成本摊销要说明周期和归属。例如，评估集建设可以按预计使用年限摊销，也可以作为一次性投入单独进入回本期；两种做法都可以，但不能一部分放进固定成本、一部分又放进一次性投入。成本账还应标记成本负责人和业务归属，便于发现某个团队节省模型费用却把人工或故障成本转移给另一个团队的情况。

## 4.9 收益类型

大模型项目收益可以分为几类：

1. 节省人工。
2. 提升效率。
3. 提高质量。
4. 增加收入。
5. 降低风险。
6. 提升用户体验。
7. 加速知识沉淀。
8. 扩大服务覆盖。

不同收益的可量化程度不同。节省人工通常比较容易估算，但“节省时间”只有在这些时间能够转化为减少外包、承接更多任务、缩短交付或避免招聘时，才等于可兑现的经济收益。用户满意度、知识沉淀和组织学习较难直接换算成现金，却仍可用留存、投诉、复购、交付周期、知识复用率等代理指标观察。

收益计算应先写清楚基线：没有新系统时同类任务的处理时间、质量、失败率和业务结果是什么。没有基线时可以先做探索性试点，但不能把试点后的绝对值直接称为“AI 带来的提升”。如果多个项目同时上线，或者季节、人员和流程也发生变化，收益归因应使用对照组、分阶段上线或其他可解释的比较设计。

## 4.10 人工替代收益

最常见的收益是减少人工处理时间，但“减少时间”不自动等于“减少支出”。如果团队没有因此减少外包、减少招聘、承接更多任务或缩短交付周期，那么它更准确地叫作产能释放，而不是已经兑现的现金收益。

例如，一个客服团队每月处理 10,000 个问题，历史基线是每个问题 5 分钟，AI 流程覆盖其中 40%，人工时间成本为每小时 60 元。假设这些节省的时间确实可以减少外包工作量，则毛收益为：

```text
每月 10000 个客服问题
每个问题原来人工处理 5 分钟
AI 能自动解决 40%
每小时人工成本 60 元
```

粗略收益：

```text
10000 * 40% * 5 / 60 * 60 = 20000 元/月
```

这个数字还没有扣除模型、检索、人工复核和固定系统成本。更重要的是，“覆盖 40%”要有观测定义：是模型生成了答案，还是用户采纳了答案，还是人工不再需要介入？三种定义会得到不同的收益。

## 4.11 效率提升收益

有些场景不是替代人工，而是提高单位时间内完成的有效任务数。

例如：

1. 销售写方案从 2 小时降到 30 分钟。
2. 工程师查文档从 15 分钟降到 3 分钟。
3. 法务初审合同从 1 小时降到 20 分钟。
4. 数据分析师生成初稿从半天降到 1 小时。

效率提升不一定减少人数，但可以提升产能、缩短交付周期、支持更多客户。评估时要同时记录处理时长和结果质量：如果时间从 2 小时降到 30 分钟，却导致返工率上升，单看时长会高估收益。一个更完整的效率指标是“单位时间完成的合格任务数”，而不是“生成了多少份草稿”。

## 4.12 质量收益

大模型也可能提升质量，但质量收益通常不是一个可以直接从模型分数复制到财务表的数字。

例如：

1. 减少漏检风险。
2. 提高回复一致性。
3. 提升文档完整性。
4. 降低新人上手成本。
5. 让专家经验可复用。

质量收益较难量化，但可以通过错误率、投诉率、返工率、审核通过率、缺陷逃逸率和人工复核结果间接衡量。先建立基线，再估计错误率变化；如果错误率从 5% 降到 4%，这意味着错误率下降了 1 个百分点，而不是“质量提升 20%”这一可以脱离样本量使用的宣传数字。质量变化也可能为负，出现幻觉、格式错误或审核负担增加时，质量收益应该记为负值，不能用 `max(uplift, 0)` 把损失藏起来。

如果要把质量变化换算成金额，需要给出每个成功或失败结果的业务价值。例如，质量提升 0.01，意味着每个被采用的任务成功概率提高 1 个百分点；若一个成功结果相对基线的增量价值为 100 元，则该项期望收益是 1 元/任务。这个换算依赖价值估计，价值未知时应保留为 `unknown`，而不是填入零。

## 4.13 收入增长

大模型也可能带来收入增长。

例如：

1. 提高销售转化率。
2. 提升用户留存。
3. 增加付费功能。
4. 支持更多客户同时服务。
5. 提升客单价。

收入增长类项目要特别注意归因。转化率提升可能来自价格、页面、销售策略、季节性和用户结构等多个因素，不能轻易把所有增长都归因于大模型。用于 ROI 的应尽量是增量毛利或增量贡献利润，而不是未经扣除退款、履约和渠道成本的流水。缺少对照组或实验设计时，应把归因结论标为 `unknown`，而不是把同期增长全部计入收益。

## 4.14 风险降低收益

有些收益来自降低风险。

例如：

1. 合同风险提示。
2. 内容合规审核。
3. 安全告警总结。
4. 财务异常检测辅助。
5. 客诉预警。

风险降低的收益可能不高频，但单次事故代价大。可以用预期损失表示：

```text
预期风险损失 = 事件发生概率 * 事件损失
风险降低收益 = 基线预期损失 - 上线后预期损失
```

如果历史样本不足、事件损失无法估计或系统没有改变风险暴露，风险收益应为 `unknown`。不能因为增加了一个审核步骤，就默认风险已经下降；需要观察漏检率、误报率、人工覆盖率和实际事故，必要时还要让领域专家审查估计。风险降低也可能为负：新系统引入错误自动执行或泄露敏感信息时，应记录为负收益。

## 4.15 单位经济账

单位经济账把一个月度项目拆成单个任务，回答“每多处理一个任务，收益和成本如何变化”。它既能发现低价值任务，也能说明固定成本需要多大的规模才能摊薄。

例如每处理一个客服问题：

```text
模型成本：0.05 元
检索成本：0.01 元
人审成本：0.20 元
总成本：0.26 元
节省人工成本：0.80 元
净收益：0.54 元
```

这个例子中的单位净收益为 0.54 元，但它仍然只是一个毛估算。如果用户不采纳答案、人工需要重新处理，0.80 元的节省就不应完整计入；如果答案错误引发退款或投诉，还要扣除预期失败成本。任务量很大时，单次差异会被规模放大，因此必须把成功、失败和人工接管分开统计。

单位经济账还要区分两种“每单”：

1. **AI 尝试单**：系统已经消耗模型或工具资源的任务。它承担变动成本，即使最后没有被用户采用。
2. **有效采用单**：用户或业务流程真正使用了结果的任务。它才产生大部分时间、质量或收入收益。

把两者混成一个分母，会让采用率下降时收益和成本同时被错误缩小，掩盖系统其实在大量无效尝试上花钱。

## 4.16 ROI 估算模板

先约定符号。设一个月有 $N$ 个符合条件的任务，$A$ 是任务进入 AI 流程的比例，$U$ 是 AI 输出被实际采用的比例，$Q$ 是相对于基线的任务成功率变化，$T$ 是每个有效采用任务节省的时间，$W$ 是人工时间的货币价值，$V$ 是每增加一个成功结果带来的价值，$G$ 是每个有效采用任务带来的增量贡献利润，$R$ 是每个有效采用任务带来的预期风险损失下降，$C_{var}$ 是每个 AI 尝试任务的变动成本，$F$ 是月固定成本，$I$ 是一次性投入。

这里的 $A$ 和 $U$ 不能互换。一个结果可能已经生成，却没有被用户采用；模型成本通常在生成时发生，而业务收益要到结果被采用后才发生。若系统对所有任务都尝试调用模型，$A$ 应接近 1；若有路由器只把部分任务交给模型，$A$ 就反映路由比例。

```text
时间收益/有效采用任务 = T / 60 * W
质量收益/有效采用任务 = Q * V
业务与风险收益/有效采用任务 = G + R
有效采用任务数 = N * A * U

月收益 = N * A * U * (T / 60 * W + Q * V + G + R)
月成本 = N * A * C_var + F
净收益 = 月收益 - 月成本
```

如果是收入增长场景，$G$ 应尽量使用扣除履约、渠道、退款等直接成本后的增量贡献利润；如果是风险降低场景，$R$ 应使用上线前后预期风险损失的差值。若一类收益已经包含在 $G$ 中，就不要再次放进 $R$，以免重复计算。

### 4.16.1 关键公式与 ROI 指标速查

为了让公式可以审计，把候选产品化场景按任务、收益、成本和决策四类记录：

```math
b_i=(N_i,A_i,U_i,Q_i,T_i,W_i,V_i,G_i,R_i,C_i,F_i,I_i)
```

其中，$N_i \ge 0$ 是月任务量，$0 \le A_i \le 1$ 是 AI 路由比例，$0 \le U_i \le 1$ 是结果采用率，$Q_i$ 是成功率变化，通常在 $[-1,1]$ 内但必须根据指标定义检查，$T_i$ 是每个有效采用任务节省的分钟数，$W_i \ge 0$ 是每小时人工价值，$V_i \ge 0$ 是成功率变化对应的单位价值，$G_i$ 和 $R_i$ 可以为负，分别表示增量贡献利润和预期风险损失变化，$C_i \ge 0$ 是每次 AI 尝试的变动成本，$F_i \ge 0$ 是月固定成本，$I_i \ge 0$ 是一次性投入。负的 $Q_i$、$G_i$ 或 $R_i$ 表示系统造成了质量、业务或风险回退，不能被截断为零。

模型调用成本可以先按输入和输出 token 拆开：

```math
C_{\mathrm{model},i}=M_i\left(\frac{X_i p_{\mathrm{in}}(1-H_i)}{K}+\frac{Y_i p_{\mathrm{out}}}{K}\right)
```

其中，$M_i \ge 0$ 是每个 AI 尝试任务的平均模型调用次数，$X_i,Y_i \ge 0$ 分别是每次调用的平均输入和输出 token，$p_{\mathrm{in}}$ 和 $p_{\mathrm{out}}$ 是每 $K$ 个 token 的 toy 单价，$H_i \in [0,1]$ 是输入 token 的缓存折扣比例。这里用 $X_i,Y_i$ 表示 token，避免和月任务量 $N_i$ 以及一次性投入 $I_i$ 混淆。真实价格、缓存折扣、批处理折扣和不同模型的阶梯价必须从当前官方价目表或账单读取，不能把 toy 参数当作报价。

单次任务的全可变成本可以写成：

```math
C_{\mathrm{var},i}=C_{\mathrm{model},i}+C_{\mathrm{rag},i}+C_{\mathrm{tool},i}+C_{\mathrm{review},i}+C_{\mathrm{retry},i}+C_{\mathrm{risk},i}
```

其中，$C_{\mathrm{rag}}$ 包含 embedding、检索、rerank、向量库和引用校验，$C_{\mathrm{tool}}$ 是工具调用，$C_{\mathrm{review}}$ 是人工审核，$C_{\mathrm{retry}}$ 是重试，$C_{\mathrm{risk}}$ 是预期失败成本。若把失败概率 $p_f$ 和单次失败损失 $L_f$ 分开记录，则 $C_{\mathrm{risk}}=p_fL_f$；若缺少其中任一项，风险成本不是零，而是 `unknown`。

月度总成本是：

```math
C_{\mathrm{month},i}=N_iA_iC_{\mathrm{var},i}+F_i
```

这个公式假设只有进入 AI 流程的任务产生 $C_{\mathrm{var}}$；如果预热、预取、失败重试或后台索引会让未采用任务也产生成本，应把它们单独加入账本。$F_i$ 包括研发摊销、运维、监控、日志、合规、安全评估、数据更新和固定基础设施。很多 ROI 误判来自只算 $C_{\mathrm{model}}$，不算 $F_i$ 或没有把失败调用算进去。

每个有效采用任务的收益可以拆成节省时间、质量变化、收入增长和风险下降：

```math
B_{\mathrm{used},i}=\frac{T_i}{60}W_i+Q_iV_i+G_i+R_i
```

其中，$B_{\mathrm{used},i}$ 是有效采用任务的期望收益，$G_i$ 是增量贡献利润，$R_i$ 是预期风险损失下降。$Q_i$ 可以为负。$A_i$ 只影响进入 AI 流程的任务数，$U_i$ 才影响生成结果最终被采用的任务数。

月度收益是：

```math
B_{\mathrm{month},i}=N_iA_iU_iB_{\mathrm{used},i}
```

为了简化后续表达，可以定义每个 AI 尝试任务的贡献为：

```math
m_i=U_iB_{\mathrm{used},i}-C_{\mathrm{var},i}
```

于是：

```math
P_{\mathrm{net},i}=N_iA_im_i-F_i=B_{\mathrm{month},i}-C_{\mathrm{month},i}
```

收益成本比和净 ROI 分别是：

```math
R_{\mathrm{bc},i}=\frac{B_{\mathrm{month},i}}{C_{\mathrm{month},i}}
```

```math
R_{\mathrm{roi},i}=\frac{B_{\mathrm{month},i}-C_{\mathrm{month},i}}{C_{\mathrm{month},i}}
```

其中 $R_{\mathrm{bc}}$ 是收益成本比，$R_{\mathrm{roi}}$ 是扣除成本后的净 ROI。只有 $C_{\mathrm{month},i}>0$ 时这两个比率才有有限定义；成本为零时不能用任意一个极小数替代分母，也不能把结果解释成正常的商业 ROI。可以报告 `unknown`，同时说明成本记录不完整或当前模型没有可计量成本。

如果有一次性投入 $I_i$，且月净收益为正，回本周期可以写成：

```math
P_{\mathrm{back},i}=\frac{I_i}{P_{\mathrm{net},i}}, \qquad P_{\mathrm{net},i}>0
```

如果 $P_{\mathrm{net},i}\le 0$，当前假设下不存在有限回本期，应报告“不可回本”；如果收益或成本数据缺失，则报告 `unknown`。不能用 `max(P_net, epsilon)` 把负数变成极小正数，因为那会把亏损项目伪装成一个极长但仍然存在的回本期。

在 $A_i$ 和其他单位参数保持不变时，盈亏平衡任务量满足 $P_{\mathrm{net},i}=0$：

```math
N_{\mathrm{be},i}=\frac{F_i}{A_i m_i}, \qquad A_i>0,\ m_i>0
```

如果 $A_i=0$ 且 $F_i>0$，没有任务进入 AI 流程，固定成本无法摊薄；如果 $m_i\le 0$，规模越大亏得越多，不存在有限盈亏平衡任务量。此时必须先提高采用率、降低变动成本、提升有效采用后的价值或重新选择场景。若 $F_i=0$，则任务量为零也已经达到成本层面的平衡，但这不表示产品创造了价值。

敏感性分析至少要看路由比例、采用率、质量变化和变动成本。由

```math
P_{\mathrm{net}}=NA\left[U\left(\frac{T}{60}W+QV+G+R\right)-C_{\mathrm{var}}\right]-F
```

在小范围变化下，可以用一阶近似：

```math
\Delta P_{\mathrm{net}}\approx N\left[m\Delta A+A B_{\mathrm{used}}\Delta U+A U V\Delta Q-A\Delta C_{\mathrm{var}}\right]-\Delta F
```

这个近似把 $N,A,U,Q,C_{\mathrm{var}},F$ 的影响拆开：采用率下降会减少收益和变动成本，但固定成本不会同步下降；质量变化下降会直接减少每个有效采用任务的收益；重试和人审增加会提高 $C_{\mathrm{var}}$。如果变量变化幅度较大，或系统存在阈值和拥塞，应使用情景表或重新运行实测，而不是依赖线性近似。

一个项目的经济决策可以形式化为一组同时成立的条件：

```math
G_{\mathrm{roi},i}=\mathbb{1}\left(P_{\mathrm{net},i}>0\ \land\ R_{\mathrm{bc},i}\ge \tau_b\ \land\ P_{\mathrm{back},i}\le \tau_p\ \land\ m_i>0\right)
```

这里的 $\land$ 表示逻辑与，不能用逗号代替。只要任一输入缺失或比率无定义，决策结果就应为 `unknown`；未知不能自动当作通过，也不能自动当作失败。

## 4.17 成本优化手段

成本优化的目标不是把账单压到最低，而是在任务成功率、延迟、可靠性和安全约束下，降低每个有效采用任务的总成本。常见方法可以按决策顺序理解：

1. **减少不必要的调用**：用规则、缓存或轻量路由处理可确定的简单请求。
2. **匹配模型与任务**：简单任务使用小模型，复杂任务才升级到更强模型，并用回归集确认升级条件没有损害质量。
3. **减少无效上下文**：去除重复历史、压缩检索片段、控制 top-k，但同时检查召回覆盖率、引用正确率和任务成功率。
4. **控制 Agent 预算**：限制步骤数、工具次数、运行时长和输出长度，工具失败时采用明确的重试和人工接管策略。
5. **利用缓存和批处理**：高频前缀或重复结果可以缓存，允许延迟的离线任务可以批处理；两者都必须核对数据新鲜度和计费条件。
6. **优化推理基础设施**：在自部署场景使用量化、批处理、KV cache 管理和合适的并行策略，但要用有效任务吞吐而不是裸 token 吞吐衡量收益。
7. **让结构化输出减少返工**：稳定的 schema 能降低后处理和人工整理成本，但 schema 校验失败仍应计入重试和失败成本。

每一次优化都需要配套实验：记录变更前后的任务成功率、采用率、p95 延迟、单位成本、人工接管率和安全事件。平均 token 下降而采用率下降，或者 GPU 利用率上升而尾延迟恶化，都不能算作真正的成本优化。

## 4.18 ROI 的常见误区

1. 只算 API 成本，不算工程、数据、人审和失败成本。
2. 把节省的时间直接当成已经兑现的现金。
3. 把模型生成率当成用户采用率。
4. 夸大自动化比例，忽略人工接管和返工。
5. 把质量提升截断为非负数，隐藏质量回退。
6. 用同期业务增长代替因果归因。
7. 用平均成本掩盖长尾请求、重试和拥塞。
8. 用极小分母替代零成本，制造看似精确的 ROI。
9. 在没有基线、样本或风险损失估计时把未知填成零。
10. 只看财务数字，不看延迟、可靠性、安全和用户体验。

ROI 估算是决策工具，不是包装项目的数字游戏。

## 4.19 项目 ROI 复盘

一个完整的 ROI 复盘可以沿着下面的顺序展开：

1. 明确任务边界和基线：哪些请求属于目标任务，系统上线前的处理时间、质量和业务结果是多少。
2. 分开记录路由、生成和采用：多少任务进入 AI 流程，多少结果被用户或下游流程真正使用。
3. 建立成本账：模型、检索、工具、人审、重试、风险、固定成本和一次性投入分别记录。
4. 把收益拆开：时间、质量、贡献利润和风险损失下降使用不同字段，避免重复计算。
5. 先做单位经济账，再做规模预测：如果每个 AI 尝试任务的贡献为负，增加任务量不能修复问题。
6. 做情景分析：至少给出保守、基准和乐观三组采用率、质量、成本和任务量假设。
7. 用受控试点校准：比较试点前后数据，保留失败样本和未采用样本，不只挑选成功案例。
8. 给出不确定性说明：数据缺失、归因不足或风险价值无法估计时，结论应标为 `unknown`。

这套复盘方法适用于人工替代、代码助手、客服、RAG 和 Agent。场景不同，收益字段不同，但“基线 -> 任务 -> 采用 -> 成本 -> 质量 -> 归因 -> 敏感性”的推理顺序不变。

## 4.20 成本优化决策

成本优化应先定位最大成本来源，再选择最小的有效变更。若成本主要来自长上下文，优先分析历史消息和检索片段；若成本主要来自人工审核，优先改进风险分层和结构化校验；若成本主要来自 Agent 重试，优先分析工具错误和状态恢复；若成本主要来自固定 GPU 容量，优先分析并发、批处理和弹性伸缩。

一个可复用的决策记录至少包含：变更假设、受影响的任务类型、预计节省、质量和安全风险、实验样本、结果指标、回滚条件和后续监控。这样可以避免把“换成更便宜的模型”当成万能答案，也能在模型价格或业务规模变化后重新计算，而不是依赖过时的经验。

## 4.21 最小可运行 ROI / 单位经济账审计 demo

下面的 0 依赖 demo 演示一个教学版 ROI audit：输入 toy 场景的任务量、AI 路由比例、结果采用率、节省时间、质量变化、输入输出 token、RAG / 工具 / 人审 / 固定成本，输出单次可变成本、月收益、月成本、净收益、收益成本比、净 ROI、回本周期、盈亏平衡任务量、敏感性分析和三态决策结果。示例数据只是教学构造，不是任何供应商或业务的报价。

```python
from math import isfinite


UNKNOWN = "unknown"


def safe_div(num, den):
    if den == 0:
        return UNKNOWN
    return num / den


def round_or_unknown(value, digits=3):
    return value if isinstance(value, str) else round(value, digits)


REQUIRED = {
    "name", "monthly_tasks", "route_rate", "use_rate", "quality_uplift",
    "time_saved_minutes", "hourly_cost", "value_per_quality_success",
    "incremental_revenue_per_task", "risk_reduction_per_task", "model_calls",
    "input_tokens", "output_tokens", "cache_discount_rate",
    "input_price_per_1k", "output_price_per_1k", "rag_cost_per_task",
    "tool_cost_per_task", "review_minutes", "review_hourly_cost",
    "retry_cost_per_task", "risk_cost_per_task", "fixed_monthly_cost",
    "upfront_cost", "latency_ok", "quality_gate",
}


def validate(s):
    missing = sorted(REQUIRED - s.keys())
    if missing:
        return [f"missing:{key}" for key in missing]
    errors = []
    numeric = REQUIRED - {"name", "latency_ok", "quality_gate"}
    if not isinstance(s["name"], str) or not s["name"].strip():
        errors.append("invalid:name")
    for key in numeric:
        value = s[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"not_numeric:{key}")
        elif not isfinite(value):
            errors.append(f"not_finite:{key}")
    for key in ("monthly_tasks", "model_calls", "input_tokens", "output_tokens",
                "hourly_cost", "value_per_quality_success", "model_calls",
                "input_price_per_1k", "output_price_per_1k", "rag_cost_per_task",
                "tool_cost_per_task", "review_minutes", "review_hourly_cost",
                "retry_cost_per_task", "risk_cost_per_task", "fixed_monthly_cost",
                "upfront_cost", "time_saved_minutes"):
        if key in s and isinstance(s[key], (int, float)) and s[key] < 0:
            errors.append(f"negative:{key}")
    if "quality_uplift" in s and isinstance(s["quality_uplift"], (int, float)):
        if not -1 <= s["quality_uplift"] <= 1:
            errors.append("out_of_range:quality_uplift")
    for key in ("route_rate", "use_rate", "cache_discount_rate"):
        if key in s and isinstance(s[key], (int, float)) and not 0 <= s[key] <= 1:
            errors.append(f"out_of_range:{key}")
    if not isinstance(s["latency_ok"], bool):
        errors.append("not_bool:latency_ok")
    if not isinstance(s["quality_gate"], bool):
        errors.append("not_bool:quality_gate")
    return errors


SCENARIOS = [
    {
        "name": "support_rag",
        "monthly_tasks": 4200,
        "route_rate": 0.62,
        "use_rate": 0.82,
        "quality_uplift": 0.24,
        "time_saved_minutes": 4.5,
        "hourly_cost": 38,
        "value_per_quality_success": 5.0,
        "incremental_revenue_per_task": 0.8,
        "risk_reduction_per_task": 0.4,
        "model_calls": 1.2,
        "input_tokens": 1800,
        "output_tokens": 360,
        "cache_discount_rate": 0.25,
        "input_price_per_1k": 0.002,
        "output_price_per_1k": 0.008,
        "rag_cost_per_task": 0.035,
        "tool_cost_per_task": 0.0,
        "review_minutes": 0.4,
        "review_hourly_cost": 24,
        "retry_cost_per_task": 0.012,
        "risk_cost_per_task": 0.02,
        "fixed_monthly_cost": 4200,
        "upfront_cost": 9000,
        "latency_ok": True,
        "quality_gate": True,
    },
    {
        "name": "contract_review",
        "monthly_tasks": 90,
        "route_rate": 0.55,
        "use_rate": 0.78,
        "quality_uplift": 0.18,
        "time_saved_minutes": 55,
        "hourly_cost": 130,
        "value_per_quality_success": 280,
        "incremental_revenue_per_task": 0.0,
        "risk_reduction_per_task": 18.0,
        "model_calls": 2.5,
        "input_tokens": 6200,
        "output_tokens": 900,
        "cache_discount_rate": 0.15,
        "input_price_per_1k": 0.004,
        "output_price_per_1k": 0.012,
        "rag_cost_per_task": 0.18,
        "tool_cost_per_task": 0.08,
        "review_minutes": 8.0,
        "review_hourly_cost": 95,
        "retry_cost_per_task": 0.10,
        "risk_cost_per_task": 0.35,
        "fixed_monthly_cost": 8200,
        "upfront_cost": 32000,
        "latency_ok": True,
        "quality_gate": True,
    },
    {
        "name": "code_agent",
        "monthly_tasks": 650,
        "route_rate": 0.38,
        "use_rate": 0.70,
        "quality_uplift": 0.20,
        "time_saved_minutes": 22,
        "hourly_cost": 90,
        "value_per_quality_success": 40,
        "incremental_revenue_per_task": 0.0,
        "risk_reduction_per_task": 0.0,
        "model_calls": 5.0,
        "input_tokens": 4800,
        "output_tokens": 1200,
        "cache_discount_rate": 0.10,
        "input_price_per_1k": 0.004,
        "output_price_per_1k": 0.012,
        "rag_cost_per_task": 0.12,
        "tool_cost_per_task": 0.25,
        "review_minutes": 5.0,
        "review_hourly_cost": 80,
        "retry_cost_per_task": 0.35,
        "risk_cost_per_task": 0.25,
        "fixed_monthly_cost": 12000,
        "upfront_cost": 45000,
        "latency_ok": False,
        "quality_gate": False,
    },
    {
        "name": "generic_chatbot",
        "monthly_tasks": 300,
        "route_rate": 0.25,
        "use_rate": 0.45,
        "quality_uplift": 0.04,
        "time_saved_minutes": 1.5,
        "hourly_cost": 35,
        "value_per_quality_success": 2.0,
        "incremental_revenue_per_task": 0.0,
        "risk_reduction_per_task": 0.0,
        "model_calls": 1.0,
        "input_tokens": 900,
        "output_tokens": 260,
        "cache_discount_rate": 0.05,
        "input_price_per_1k": 0.002,
        "output_price_per_1k": 0.008,
        "rag_cost_per_task": 0.0,
        "tool_cost_per_task": 0.0,
        "review_minutes": 0.2,
        "review_hourly_cost": 24,
        "retry_cost_per_task": 0.01,
        "risk_cost_per_task": 0.03,
        "fixed_monthly_cost": 2500,
        "upfront_cost": 6000,
        "latency_ok": True,
        "quality_gate": False,
    },
]


def model_cost(s):
    discounted_input = s["input_tokens"] * (1.0 - s["cache_discount_rate"])
    input_cost = discounted_input * s["input_price_per_1k"] / 1000.0
    output_cost = s["output_tokens"] * s["output_price_per_1k"] / 1000.0
    return s["model_calls"] * (input_cost + output_cost)


def variable_cost(s):
    review_cost = s["review_minutes"] / 60.0 * s["review_hourly_cost"]
    return (
        model_cost(s)
        + s["rag_cost_per_task"]
        + s["tool_cost_per_task"]
        + review_cost
        + s["retry_cost_per_task"]
        + s["risk_cost_per_task"]
    )


def benefit_per_task(s, quality=None):
    quality = s["quality_uplift"] if quality is None else quality
    time_value = s["time_saved_minutes"] / 60.0 * s["hourly_cost"]
    quality_value = quality * s["value_per_quality_success"]
    return time_value + quality_value + s["incremental_revenue_per_task"] + s["risk_reduction_per_task"]


def audit(s):
    errors = validate(s)
    if errors:
        return {
            "status": UNKNOWN,
            "errors": errors,
        }
    var_cost = variable_cost(s)
    benefit_task = benefit_per_task(s)
    routed_tasks = s["monthly_tasks"] * s["route_rate"]
    used_tasks = routed_tasks * s["use_rate"]
    monthly_benefit = used_tasks * benefit_task
    monthly_cost = routed_tasks * var_cost + s["fixed_monthly_cost"]
    net_benefit = monthly_benefit - monthly_cost
    bcr = safe_div(monthly_benefit, monthly_cost)
    roi = safe_div(net_benefit, monthly_cost)
    payback = s["upfront_cost"] / net_benefit if net_benefit > 0 else "not_recoverable"
    unit_margin = s["use_rate"] * benefit_task - var_cost
    break_even_tasks = (
        s["fixed_monthly_cost"] / (s["route_rate"] * unit_margin)
        if s["route_rate"] > 0 and unit_margin > 0
        else "not_recoverable"
    )

    sensitivity = {}
    for delta in (-0.15, 0.0, 0.15):
        adjusted_use = max(0.0, min(1.0, s["use_rate"] + delta))
        adjusted_benefit = s["monthly_tasks"] * s["route_rate"] * adjusted_use * benefit_task
        adjusted_cost = routed_tasks * var_cost + s["fixed_monthly_cost"]
        sensitivity[f"use_rate_{delta:+.2f}"] = round(adjusted_benefit - adjusted_cost, 2)
    for multiplier in (0.8, 1.0, 1.2):
        adjusted_cost = routed_tasks * var_cost * multiplier + s["fixed_monthly_cost"]
        sensitivity[f"variable_cost_x{multiplier:.1f}"] = round(monthly_benefit - adjusted_cost, 2)

    failed = []
    if net_benefit <= 0:
        failed.append("net_benefit")
    if bcr == UNKNOWN:
        failed.append("benefit_cost_ratio_unknown")
    elif bcr < 1.25:
        failed.append("benefit_cost_ratio")
    if payback == "not_recoverable" or payback > 6:
        failed.append("payback")
    if unit_margin <= 0:
        failed.append("unit_margin")
    if not s["latency_ok"]:
        failed.append("latency")
    if not s["quality_gate"]:
        failed.append("quality")
    status = UNKNOWN if bcr == UNKNOWN else ("passed" if not failed else "failed")

    return {
        "status": status,
        "model_cost_per_task": round(model_cost(s), 4),
        "variable_cost_per_task": round(var_cost, 4),
        "benefit_per_task": round(benefit_task, 3),
        "unit_margin": round(unit_margin, 3),
        "monthly_benefit": round(monthly_benefit, 2),
        "monthly_cost": round(monthly_cost, 2),
        "net_benefit": round(net_benefit, 2),
        "benefit_cost_ratio": round_or_unknown(bcr),
        "roi": round_or_unknown(roi),
        "payback_months": round_or_unknown(payback),
        "break_even_tasks": round_or_unknown(break_even_tasks, 1),
        "sensitivity": sensitivity,
        "roi_gate": status,
        "failed_gates": failed,
    }


audits = {scenario["name"]: audit(scenario) for scenario in SCENARIOS}
ranked = sorted(
    (
        (name, result["net_benefit"], result["roi_gate"])
        for name, result in audits.items()
        if "net_benefit" in result
    ),
    key=lambda item: item[1],
    reverse=True,
)

print("ranked=", ranked)
print("roi_pass=", [name for name, result in audits.items() if result["roi_gate"] == "passed"])
print("needs_rework=", {
    name: result.get("failed_gates", result.get("errors", []))
    for name, result in audits.items()
    if result.get("status", result.get("roi_gate")) != "passed"
})
for name in [item[0] for item in ranked]:
    print(name, audits[name])


def copy_case(base, **updates):
    case = dict(base)
    case.update(updates)
    return case


zero_cost = copy_case(
    SCENARIOS[0],
    input_price_per_1k=0.0,
    output_price_per_1k=0.0,
    rag_cost_per_task=0.0,
    tool_cost_per_task=0.0,
    review_minutes=0.0,
    retry_cost_per_task=0.0,
    risk_cost_per_task=0.0,
    fixed_monthly_cost=0.0,
)
unprofitable = copy_case(SCENARIOS[0], monthly_tasks=1, fixed_monthly_cost=1_000_000)
nan_case = copy_case(SCENARIOS[0], quality_uplift=float("nan"))
negative_cost = copy_case(SCENARIOS[0], rag_cost_per_task=-1.0)
missing_value = dict(SCENARIOS[0])
missing_value.pop("value_per_quality_success")

boundary_results = {
    "empty": audit({}),
    "zero_cost": audit(zero_cost),
    "unprofitable": audit(unprofitable),
    "nan": audit(nan_case),
    "negative_cost": audit(negative_cost),
    "missing_value": audit(missing_value),
}
assert boundary_results["empty"]["status"] == UNKNOWN
assert boundary_results["zero_cost"]["status"] == UNKNOWN
assert boundary_results["zero_cost"]["benefit_cost_ratio"] == UNKNOWN
assert boundary_results["unprofitable"]["status"] == "failed"
assert boundary_results["unprofitable"]["payback_months"] == "not_recoverable"
assert boundary_results["nan"]["status"] == UNKNOWN
assert boundary_results["negative_cost"]["status"] == UNKNOWN
assert boundary_results["missing_value"]["status"] == UNKNOWN
print("boundary_status=", {name: result["status"] for name, result in boundary_results.items()})
```

这段 demo 的关键结论是：`support_rag` 在 toy 假设下通过经济检查；`contract_review` 单次价值高，但固定成本和人审成本太重，当前月净收益为负且不可回本；`code_agent` 即使单位贡献为正，固定成本、延迟和质量条件仍未满足；`generic_chatbot` 任务价值和采用率太低，规模也不够。边界样例进一步说明：缺失、非有限或非法成本数据返回 `unknown`，合法但亏损的场景返回 `failed`，不可回本不能被改写成一个极大的月数。

## 4.22 资料入口与证据边界

- [OpenAI Batch API 指南](https://developers.openai.com/api/docs/guides/batch)：官方开发者文档，可支持批处理适合离线任务、请求生命周期和计费条件需要单独核对的讨论；它不提供本章 toy 场景的业务收益。
- [OpenAI Prompt Caching 指南](https://developers.openai.com/api/docs/guides/prompt-caching)：官方开发者文档，可支持缓存命中、输入前缀和缓存有效期等成本变量的讨论；实际折扣和命中率必须以当前价目表和账单为准。
- [OpenAI Latency Optimization 指南](https://developers.openai.com/api/docs/guides/latency-optimization)：官方工程文档，可支持模型选择、token、并行、缓存和延迟之间的权衡；目标流量下的 p95/p99 仍需自行实测。
- [OpenAI Evals 指南](https://developers.openai.com/api/docs/guides/evals)：官方评估入口，可支持把任务、样本、指标和回归评估纳入 ROI 计算的讨论；它不能替代具体业务的基线和因果归因。
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：政府机构发布的风险管理框架，可支持把风险识别、测量和治理纳入成本收益分析；它不自动证明某个产品安全。
- [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)：NIST 的生成式 AI 风险画像，可支持生成式系统的风险、评估和治理边界；具体风险概率与损失仍需按业务数据估计。
- [FinOps Framework](https://www.finops.org/framework/)：FinOps Foundation 的成本管理框架，可支持成本可见性、责任归属、单位经济账和持续优化的讨论；它不是模型价格表，也不替代财务核算。

本章引用的官方文档说明了成本、评估和治理应如何被拆分；公式、场景参数、收益货币化和 Python demo 是教学构造。任何“通过”或“不可回本”的结论只对 demo 参数成立，不能外推到真实供应商、组织或行业。生产决策还需要真实账单、基线、受控试点、失败样本、用户采用数据、可靠性指标和领域专家审查。

## 4.23 本章小结

大模型项目要可持续，必须算成本收益。成本不仅是 token，还包括 GPU、RAG、Agent 工具、人审、数据、工程、运维、安全和失败成本。收益也不只是一句“提升效率”，而要尽量连接到人工节省、质量提升、收入增长或风险降低。

下一章会进入企业级 LLM 应用，讨论大模型在企业场景中如何接入知识库、权限系统、业务流程、审计合规和组织协作。
