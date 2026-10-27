# 第六册第九章：RAG 与 Agent 部署深度精修研究记录

- 日期：2026-10-26
- 正文：[`book-06-llm-deployment/chapters/09-rag与agent部署.md`](../book-06-llm-deployment/chapters/09-rag与agent部署.md)
- 依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)、本轮全书精修约束及 `AGENTS.md`。
- 结论：原章 1—17 节的部署链路、机制、demo、企业案例、风险和小结均保留；研究补充围绕代表方案分别解决什么、证据来自哪里、代价和限制是什么，以及如何验证下一步改进。

## 研究问题

1. 固定 top-k RAG 常见的证据缺口有哪些？Self-RAG、CRAG、GraphRAG 和多轮 Agentic RAG 各自改变了哪一段机制？
2. RAGAS、ARES 一类自动评估器在缺少唯一标准答案时能提供什么证据，哪些结论仍需人工标注和目标语料校准？
3. 当检索内容本身可包含恶意指令时，风险如何越过检索层进入工具执行？安全评测应同时观察哪些效用与安全结果？
4. 多次工具调用何时适合由模型逐轮决策、确定性 workflow 编排或程序化工具编排？
5. 原 demo 的输出是否准确说明了代码真正检查的边界？

## 代表性研究与方案

| 来源/方案 | 解决的问题与机制 | 作者证据或工程事实 | 代价、限制与适用条件 |
| --- | --- | --- | --- |
| [原始 RAG](https://arxiv.org/abs/2005.11401) | 将可更新的非参数文档检索与生成结合；论文区分整段共享 passage 与按 token 使用 passage 的形式。 | 在论文的开放域知识密集问答设置中评估。 | 召回错误仍会传给生成器；不能因接入检索就认为事实正确，需独立测召回和生成器是否使用证据。 |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 用训练得到的 reflection tokens，让模型决定是否检索、评估 passage 相关性并反思生成，针对固定 top-k 的噪声和缺证据问题。 | 作者在其 7B/13B 模型及任务中报告开放域问答、推理、事实核验和引用方面的改进。 | 需要匹配训练与推理机制；不是任意托管模型可打开的参数。自评仍是同一模型产生的判断，不是独立 verifier；应分别测漏检索、误检索和无证据回答。 |
| [CRAG](https://arxiv.org/abs/2401.15884) | 增加 retrieval evaluator，根据候选质量估计触发本地修正或外部搜索，再分解和过滤检索内容。 | 论文在四个短、长生成数据集报告改进。 | evaluator 校准差会选错回退路径；web search 增加来源质量、延迟、费用和间接注入面。需要单测 evaluator、外部来源和回退后的端到端收益。 |
| [GraphRAG](https://arxiv.org/abs/2404.16130) 与 [Microsoft 实现](https://github.com/microsoft/graphrag) | 从实体关系构图、聚类成社区并生成层级摘要，解决局部相似 chunk 不擅长回答“整个语料有哪些共同主题”的 global sensemaking 问题。 | 论文在约百万 token 规模语料的全局 sensemaking 任务上报告 comprehensiveness、diversity 优于 naive RAG。仓库注明这是 research project、处于 mostly maintenance mode，并提示 indexing 可能昂贵。 | 离线抽取、社区汇总、更新和错误传播成本较高；对普通局部问答不自动优于向量检索。适用于问题确实需要跨语料全局归纳的场景。 |
| [RAGAS](https://arxiv.org/abs/2309.15217) | 在不要求每个问题都有完整人工参考答案时，分开估计 context relevance、faithfulness 与 answer relevance。 | 降低了构造评估集的门槛。 | 许多指标依赖模型判断、分解或相似度；评审模型、提示、语言和领域变化可能改变分数，不能当成 oracle。 |
| [ARES](https://arxiv.org/abs/2311.09476) | 用合成样本训练轻量 judge，并用少量人工标注和 prediction-powered inference 校准总体估计与置信区间。 | 作者在自己的测试条件中报告，相对 RAGAS，context relevance 与 answer relevance 的系统排序 Kendall's tau 平均分别高 0.16、0.15。 | 这是论文 benchmark、标注和 judge 配置下的对比，不能推出任意企业语料上的准确性排序；仍要用目标场景人工样本检查偏差。 |
| [Agentic RAG / ReAct](https://arxiv.org/abs/2210.03629) | 根据中间观察继续拆解问题、检索和整合，补足单次查询无法预见的多跳证据缺口。 | ReAct 展示 reasoning 与环境 action 交错的控制模式。 | 多轮会引入 query drift、重复检索和错误停止判断；限制轮数和预算，以固定 query trace 测新增证据和答案质量，不以检索次数当成质量指标。 |
| [AgentDojo](https://arxiv.org/abs/2406.13352) | 提供有工具和间接提示注入的 Agent 任务，用于同时观察正常任务效用和攻击下的安全属性。 | 基准包含 97 个任务、629 个安全用例；论文显示无攻击时任务成功率也并非总是很高，攻击会破坏部分安全属性。 | 基准不等于目标系统验收。应同时切片 clean-task utility、under-attack utility、越权读取、数据外发和高风险副作用。 |
| [Overcoming the Retrieval Barrier](https://www.usenix.org/conference/usenixsecurity26/presentation/chang-hongyan) | 将 trigger fragment 和攻击指令分开，使恶意指令较短也能被检索带入上下文，针对只在生成端过滤注入的假设。 | 作者报告在其 11 个 benchmark、8 个 embedding model 上接近 100% 的恶意内容 retrieval；在一个被投毒邮件的多 Agent 场景中，GPT-4o SSH key 外泄成功率超过 80%。 | 结果绑定论文威胁模型、数据、检索器和多 Agent 环境，不代表所有 RAG 系统的攻击率。它说明检索后检测器无法保证攻击内容不进入上下文；还需治理采集来源、修订/信任标签、读取与外发权限、工具前策略和确认流程。 |
| [OpenAI Programmatic Tool Calling 官方指南](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling) | 在特定 API/runtime 中允许模型生成受限程序，循环、并行调用工具并过滤或归并中间结果。 | 官方建议用于控制流可预测、结果可由代码筛选或聚合的任务；每个返回都需模型新一轮语义判断、审批或保留原生引用时，直接调用更合适。 | 这是产品特定能力，不是通用 tool calling 协议。运行时隔离、工具 allowlist、逐次鉴权和高风险操作审批仍由应用保证。 |
| [RAG Survey](https://arxiv.org/abs/2312.10997) | 作为 RAG pipeline、retrieval、generation、augmentation 和评估路线的分类地图。 | 综述帮助定位方法类别与原始工作。 | 综述用于建立路线图，机制和指标结论需回到原论文及目标系统证据。 |

### 方案选择的因果顺序

这些方案处在不同层次，不是按新旧排序的替代品。部署时先确认正确证据是否进入候选集；若没有，改解析、过滤、召回或重排。若证据已进入上下文但回答未使用，检查 prompt、上下文位置与 claim-to-span 对齐。需要全语料主题总结时评估 GraphRAG；需要判断何时检索时才评估 Self-RAG 式训练或控制流；初始候选可能失效且外部来源可控时评估 CRAG 式回退；多跳问题确有新增证据时再加入 Agentic RAG。这个顺序是基于失败定位的部署建议，不是上述论文共同验证过的统一最佳 pipeline。

## 原章节覆盖图

| 原章节 | 本轮处理 |
| --- | --- |
| §1 RAG 生产链路 | 保留离线构建与在线查询的职责和版本边界。 |
| §2 文档解析和切分 | 保留解析、chunking、重叠与来源元数据；作为证据丢失的首个定位点。 |
| §3 Embedding 服务 | 保留模型选择、批处理和版本影响。 |
| §4 Vector database | 保留 ANN 索引机制、更新和过滤约束。 |
| §5 Hybrid retrieval 和 reranker | 保留稀疏/稠密召回、融合与重排；新增方法比较以其为基线定位召回问题。 |
| §6 Prompt assembly | 保留上下文预算和证据装配；明确 citation ID 存在不等于 claim 得到支持。 |
| §7 RAG 评估和监控 | 保留检索/生成/系统评测；新增 Recall@K 手算、Self-RAG/CRAG/GraphRAG/Agentic RAG、RAGAS/ARES 的目标、证据和限制。 |
| §8 Agent 和普通 Chat 的区别 | 保留循环、观察和状态差异。 |
| §9 Tool calling 部署 | 保留结构、校验和执行职责；补逐次调用、确定性 workflow 与特定 PTC 的比较和适用条件。 |
| §10 Agent 状态管理 | 保留状态、长任务和恢复语义。 |
| §11 权限、安全和审计 | 保留权限与审计；补 retrieval poisoning 研究与 AgentDojo 的效用/安全联合评估。 |
| §12 Agent 可观测性 | 保留 trace 与分层指标。 |
| §13 RAG 与 Agent 的部署风险 | 保留失效路径；证据链加入 source provenance/revision/trust/taint 等字段。 |
| §14 最小 Python demo | 保留 toy RAG、租户过滤、上下文、引用、工具和延迟示例；更正“引用 ID 检查”和“词项预算”的命名及解释。 |
| §15 企业知识助手端到端发布案例 | 保留权限、状态转移、幂等与发布复盘。 |
| §16 资料与证据边界 | 扩展至 22 条论文、官方文档和工程入口，给出每类证据的适用边界。 |
| §17 本章小结 | 保留原有部署要点，增加方法选型、评估校准和检索注入的结论。 |

## 公式、示例与代码核验

- Recall 示例：gold set 为 `{A,B,C}`，top-5 为 `{A,X,B,Y,Z}`，命中 2 个 gold passage，所以 `Recall@5=2/3`、`Hit@5=1`。若上下文组装阶段只留下 A，则 context recall 为 `1/3`。两个指标分别定位“候选未召回”和“已召回证据未进入上下文”。
- demo 将 token budget 改称 `word_units`：正则统计词项，不等于任何目标模型 tokenizer 的 token 数；当前固定预算示例输出 26 units。
- demo 将 `citation_ok` 改为 `citation_references_valid`：只检验引用 ID 存在，不能证明被引 span 支持回答中的 claim。缺失/未知 ID 检查通过。
- demo 的 tool checker 仅检查 required 字段、角色和写权限，不是完整 JSON Schema、业务规则或真实工具执行；未知工具和缺少 required 参数分别被拒绝。延迟数据是预设 toy 值，不是代码测量。
- 从 Markdown 提取并运行 §14 Python 块，退出码 0、stderr 为空；实际 stdout 与文中 `text` 围栏逐字一致，共 9 行。另核对租户隔离、引用已知/未知/缺失 ID、未知工具及缺参路径。
- 检查主节编号连续 §1—§17；40 个 fenced-code/math/text 标记闭合；14 张表格的每行列数一致。章节没有相对 Markdown 文件链接；`git diff --check` 通过。
- 未使用外部数学渲染器。论文的 Recall 示例为可手算例子，不是线上测量；demo 不连接 embedding、ANN、模型、Agent runtime 或事务系统。

## 配套资料同步

- `PAPERS.md`：增加 Self-RAG、CRAG、GraphRAG、RAGAS/ARES、检索注入研究和 Programmatic Tool Calling 入口。
- `GLOSSARY_EN_ZH.md`：增加 Self-RAG、CRAG、RAGAS、ARES、AgentDojo、Retrieval Poisoning 等术语。
- `INTERVIEW_BANK.md`：加入代表性 RAG 方法的故障目标/机制/代价比较、自动 judge 校准问题和检索投毒安全题。
- `EXERCISES.md`：增加方法选择对比、ARES/RAGAS 校准实验和 retrieval-poisoning 红队设计练习。
- 第六册目录已有第九章条目，无需改导航。

## 链接与证据边界

本轮重点打开核验 Self-RAG、CRAG、GraphRAG、RAGAS、ARES、AgentDojo、USENIX Security 2026 retrieval-attack 页面和 OpenAI Programmatic Tool Calling 官方指南；新加入的方法比较与数字均以原论文或官方页面为据。章节现有全部外链未做自动 HTTP 状态扫描。论文实验结果只绑定各自模型、任务、benchmark、标注与威胁模型；Microsoft GraphRAG 仓库状态及 OpenAI API 能力是会变化的当前信息，后续使用前应按版本复核。

本轮没有目标企业语料、查询 trace、目标模型、向量服务、运行时权限策略、AgentDojo 本地环境或生产工具链；没有执行真实 red team、延迟测量、成本测算或生产 benchmark。因此论文结果和 toy demo 不能证明目标系统的质量、安全或服务能力。
