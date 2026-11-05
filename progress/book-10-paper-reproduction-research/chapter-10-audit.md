# 第十章《Long Context、MoE、RAG、Agent 论文线》逐章审阅与精修

- 审阅日期：2026-11-04
- 状态：章级扫描、正文静态精修和篇幅/公式/链接门禁完成；Python demo 未运行，GitHub Mermaid/MathJax 页面渲染与历史外链全量复核留待最终验收。
- 正文：[第十章 Long Context、MoE、RAG、Agent 论文线](../../book-10-paper-reproduction-research/chapters/10-long-context-moe-rag-agent论文线.md)

## 范围与研究问题

完整审阅本章 18 个连续一级主题。章级问题是：扩展输入、稀疏模型容量、外部检索证据、工具动作与多步 Agent 分别从哪里获得能力；各自转移了哪些计算、存储、数据、权限和失败责任；组合论文是否能以公平预算和合适的分层指标识别贡献；结论可以外推到哪些模型、任务、harness 和运行环境。

本章架构审计曾指出，Long Context、MoE、RAG 和 Agent 的主题跨度较宽。逐章扫描后又将 Tool Use 作为 Agent loop 的接口层一起审阅。正文仍可沿一条共同研究主线阅读：能力来源是什么、成本与状态放在哪里、证据链是否可追踪、组合效果如何归因。MoE 处在模型内部，RAG 提供外部证据，Tool Use 暴露受权限约束的动作接口，Agent 负责跨步骤状态控制；它们不被当作同一种机制，却能在组合系统和公平实验问题上形成连贯专题。未发现需要拆章或另增章节的断层，保留现有组织。

定向扫描覆盖 2024—2026 年长上下文有效利用和 benchmark 任务边界、MoE 路由/容量/负载、RAG 检索与引用证据链、间接提示注入、Agent 记忆和工具执行、授权接口及 trace/harness 评估。阅读了相关原始论文、固定版本预印本、正式出版页与官方标准/会议资料中的机制、实验设置或限制；扫描不是对所有方法的穷尽检索。新事实和证据范围分别留在 research/，本文件记录章节候选处置、修改决定与验收状态。

相应证据记录见[长上下文评估](../../research/long-context-evaluation-and-effective-capability-2026-11-03.md)、[RAG 检索与证据链](../../research/rag-retrieval-and-evidence-pipeline-2026.md)、[Agent 工具调用安全](../../research/agent-tool-calling-security-2026.md)、[Agent 记忆](../../research/long-context-harness-and-agent-memory-2026.md)、[Agent trace 评估](../../research/agent-trace-diagnosis-and-online-evaluation-2026.md)、[授权互操作标准](../../research/agent-authorization-interoperability-standards-2026-11-03.md)和[benchmark/harness 生命周期](../../research/evaluation-benchmark-lifecycle-and-agent-harness-2026.md)。

## 候选主题与处置

| 候选主题/缺口 | 证据范围与判断 | 去向 |
| --- | --- | --- |
| 输入窗口上限能否代表长上下文理解 | NoLiMa v3 通过降低 query 与证据的词面重合来测长上下文检索；作者在 32K 设置中报告 13 个模型有 11 个低于各自短上下文基线的一半，结果绑定论文模型与任务。LongBench Pro v1 是 8K—256K token 的中英自然样本预印本；LongBench v2 使用 words 描述长度，单位不可混用。 | §§2、9 增加 attention/KV cache 账本、位置/多证据分层测试、benchmark 边界和成本解释；不把最大接口长度写成通用理解能力。 |
| MoE 的总参数和 active compute 是否可直接比较 | DeepSeek-V3 报告 671B 总参数、每 token 激活约 37B，并使用 batch load-aware routing bias 与 sequence-wise auxiliary loss；这属于该模型、训练和集群方案。容量/overflow 的实现语义会决定 dropped token 的分母和影响。 | §§3、10 增加 assignment、capacity、负载 CV、通信时间手算；以 dense/MoE 双预算边界评价，不将一个报告方案写成普遍配方。 |
| RAG 最终准确率能否说明证据完整 | Recall@k、完整证据集合召回、claim-level support、citation alignment 与文档正确/新鲜是不同对象。Self-RAG、CRAG、RAPTOR、GraphRAG 使用不同机制；USENIX Security 2026 检索增强注入结果限于其 threat model 和设置。 | §§4、7、11、14 区分机制、分母、来源血缘、生成使用和安全边界；不把“RAG 降低幻觉”作为未拆分的总 claim。 |
| 工具格式正确是否意味着动作安全 | AuthZEN Authorization API 1.0 是 PDP/PEP 的 Final 接口规范，不定义部署方业务策略或具体授权正确性；执行器还涉及错误、幂等、状态变化和副作用。 | §§5、7、14 增加 schema—授权—执行—结果使用的分层与超时重试案例；保持授权与模型计划分离。 |
| Agent 记忆和成功率是否说明可靠长程能力 | AdaMem 与 Mem2ActBench 研究特定记忆写入/动作 grounding 任务；Agentic CLEAR 做 system/trace/node 层级诊断；Harness-Bench v2 比较完整 model–harness 配置。Harness-Bench 摘要与固定正文数量信息存在差异。 | §§6—8、12—16 增加记忆来源、trace、调用预算、完整配置归因和不确定性；不采用 Harness-Bench 的规模/排名数字，不把单次 demo 成功外推为业务可靠性。 |

## 修改决定与正文结果

1. 保留本章原有 18 个一级主题及顺序，不拆分或迁移内容。开篇明确五种能力扩展的对象、来源和责任边界，并用统一研究问题把四个标题主题与 Tool Use/Agent Planning 接在一起。
2. §2 增加单 head、batch=1、16-bit 的 attention score 矩阵手算：4096 token 对应 16,777,216 个元素、约 32 MiB；8192 token 对应 67,108,864 个元素、约 128 MiB。增加 32 层、8 个 KV head、128 维、128,000 token、每元素 2 byte 的 KV cache 例，算得约 16.78 GB/15.625 GiB，并标出不含并发、allocator 和临时状态。
3. §3 修正 MoE 公式中的函数写法；用构造分配 [80, 50, 40, 30] 手算均值 50、总体标准差约 18.71、CV 约 0.374。以 100 token、top-2、4 专家、capacity factor 1.25 算得每专家容量 63、17/200 assignment overflow 为 8.5%，强调 assignment drop 不等于 token 序列失败率。添加 DeepSeek-V3 的参数规模和负载策略事实，限定为作者设置。
4. §4 将文本数据流改为 Mermaid；增加 Recall@5 与完整证据集合召回构造例（72/100=0.72，12/20=0.60），说明 Self-RAG、CRAG、RAPTOR 与 GraphRAG 的机制差异，纳入检索型间接注入研究并限定 threat model。
5. §§5—6 将工具结果和 ReAct 状态流改为 Mermaid；补 schema、AuthZEN PDP/PEP、实际授权、执行与结果使用的分层；说明写操作超时后的幂等与状态核验；补 AdaMem/Mem2ActBench 的任务范围及 $0.95^{20}\approx0.358$、$0.97^{20}\approx0.544$ 的独立性玩具计算。
6. §§7—8 增加长上下文/RAG 的 2×2 交互项构造计算，说明它不是无误差的因果结论；纳入 Agentic CLEAR 三层诊断和 Harness-Bench 的模型—harness 配置边界，不引用数量冲突和排名。
7. §9 将原有 1M 输入案例扩展为有位置桶、分母、token 单位、请求账目与 benchmark 边界的案例；标出 0.92 与 0.61 的差为 31 个百分点、多证据组合为 0.48，并解释每请求账单 6 倍与裸 token 长度比约 7.81 倍不应混为一项。
8. §§10—12 的三类系统案例分别处理 MoE、RAG 和 Agent。MoE case 区分 token 与 assignment 分母，并手算 3% overflow 及 all-to-all 占 28% 时的局部优化/step time 差异；RAG case 分开 72/100→81/100 答案准确率与 55/100 claim support；Agent case 算出 1,800/500 调用、25/7.58 次调用/成功任务和 3.3 倍差，并说明 0.42 单位成功成本不能从调用数推出。
9. §13 为标准库 Python demo 标注“静态审阅，未运行（2026-11-04；Python 3.7+，仅标准库）”。逐项手算整体/中间位置 recall、MoE CV、RAG recall/support、tool success 与 task success，并解释四条动作和最终 decision 是阈值触发的诊断提示，不是模型得分或实测输出。未运行程序或测试。
10. §14 加入首个不可恢复失败的分层诊断和两个教学构造的失败账本；说明 trace 只缩小候选根因范围，不能单独证明因果。§§15—16 将复现与报告作为一个连续核心工作流，扩展固定输入/预算、样本 ID、版本/血缘、证据等级、偏差披露、负结果与决策边界。
11. PAPERS.md 已包含 DeepSeek-V3、NoLiMa v3、LongBench Pro、RAG 家族、检索注入、Agentic CLEAR、Harness-Bench、AdaMem、Mem2ActBench 和 AuthZEN 来源入口，无需重复添加。简介与目录原摘要继续准确，无需同步；本章没有新增跨册主讲或改动章节入口。
12. DeepSeek-V3 负载均衡证据补入现有 [transformer-architecture-research-2026.md](../../research/transformer-architecture-research-2026.md)；该研究笔记仅记录来源、版本、事实和适用边界，不记章节工作日志。

## 教学单元篇幅

按解释性汉字统计，排除标题、表格、代码/图示/公式围栏、行内公式/代码、链接和状态标记。核心单元下限 1,800 汉字，辅助单元下限 1,000 汉字。开篇—§1 是定位和统一研究问题的有边界辅助单元；§§5—6 把动作协议与跨步状态循环组合为一条连续系统链；§§7—8 连接组合故障与论文评价；§§15—16 是从实验复现到报告的连续研究工作流。

| 独立学习单元 | 汉字数 | 类型与下限 | 状态 |
| --- | ---: | --- | --- |
| 开篇—§1 五种能力与共同研究问题 | 1,147 | 辅助，≥1,000 | 通过 |
| §2 Long Context 机制、成本与评估 | 2,013 | 核心，≥1,800 | 通过 |
| §3 MoE 路由、负载、容量与成本 | 2,017 | 核心，≥1,800 | 通过 |
| §4 RAG 证据链、评估与安全 | 1,978 | 核心，≥1,800 | 通过 |
| §§5—6 Tool Use 与 Agent 状态循环 | 2,387 | 核心，≥1,800 | 通过 |
| §§7—8 组合系统与论文评价 | 1,169 | 辅助，≥1,000 | 通过 |
| §9 百万上下文能力案例 | 1,148 | 辅助，≥1,000 | 通过 |
| §10 MoE 质量归因案例 | 1,308 | 辅助，≥1,000 | 通过 |
| §11 RAG 引用支持案例 | 1,079 | 辅助，≥1,000 | 通过 |
| §12 Agent 调用与成本案例 | 1,192 | 辅助，≥1,000 | 通过 |
| §13 组合审计 demo 讲解 | 1,063 | 辅助，≥1,000 | 通过 |
| §14 失败模式与诊断顺序 | 1,250 | 辅助，≥1,000 | 通过 |
| §§15—16 复现契约与研究报告 | 1,885 | 核心，≥1,800 | 通过 |
| §17 资料边界、§18 结语 | 资料入口/收束 | 不作为独立讲解单元 | — |

## 数值例子、公式、图示、代码与链接

- **数值例子：** 除各单元表中的手算外，补充 100 query 的 Recall@5、20 个双来源问题的 evidence-set recall；长上下文按相同位置桶列成功率差；Agent 20 步可靠性用独立性假设示范；§7—8 用 2×2 构造分数算交互差 2 个百分点。所有自行构造结果在正文标明为教学值或构造案例，不冒充论文/生产测量。
- **公式：** 共 17 个 math 围栏；花括号数量逐块平衡，未发现 \operatorname；208 个未转义单美元符号成对。MathJax 页面渲染未确认。
- **图示/围栏：** 18 个一级主题编号连续。共 21 个代码围栏块、42 个围栏标记，包含 17 math、3 Mermaid、1 Python；静态检查配对。三幅 Mermaid 均使用目标平台支持的 flowchart 语法和带引号中文标签；源码关系已人工检查，GitHub 实际渲染未确认。
- **代码：** 仅 §13 的纯标准库 Python demo。根据行内状态如实标为静态审阅、未运行；手算预期：long recall 0.75、middle 0.50、MoE CV 约 0.3510、RAG recall 与 support 均约 0.6667、tool success 0.75、task success 0.50；预计加入位置/多证据、router、retrieval/citation、trace/recovery/budget 四项动作，decision 为 **continue_after_system_repairs**。未执行代码、测试、模型、benchmark 或 PDF 构建。
- **链接：** 共 33 个 Markdown 链接，均为外部来源入口，本地目标数为 0；外部链接未做全量在线可达性复查。关键事实均在相邻正文链接到原始论文、官方标准或会议页面。
- **配套同步：** 章标题、职责和顺序未变化；简介与目录摘要仍准确，PAPERS 来源入口已存在。未修改题库、练习、术语表、知识图谱、plan.md 或 PDF。
- **静态检查：** git diff --check 在本审计写入后复跑并通过。

## 剩余验收与下一步

- 第十章 Python demo 的动态验证、三幅 Mermaid 与 math 页面渲染、旧外链全量巡检留待全书最终门禁；代码未变化前可沿用静态状态，若之后改动可执行语句则按规范复验。
- 第十册其余章节和全书连续复读尚未完成。本册整册门禁不因本章通过静态验收而完成。
- 下一步按目录处理第十一章《Multimodal、Diffusion、Video 论文线》；恢复工作时先重读总纲，再按阶段完整读取模块 02—05。
- 全套书精修 goal 保持 active；PDF 不更新。
