# 第十六册架构审视：Reasoning Model 专题

审视日期：2026-11-13

阶段：书册契约、简介/目录审视、二十章章节地图、2025—2026 册级扫描和骨架决策完成；保留现有二十章文件与编号，简介、目录已更新以说明主线、专题路径和跨册分工。第 1—20 章已完成逐章通读、章级定向扫描、精修和单篇门禁；整册连续复读与最终门禁待处理。按用户要求不更新 PDF。

## 书册契约（book_contract）

- **读者问题：** 推理模型如何通过训练信号、候选生成、验证、搜索和额外计算解决难题；怎样区分可见思维文本、模型推理能力、provider 接口状态与可验证任务结果；如何用质量、成本、安全和执行证据判断收益。
- **目标读者与前置：** 具备 Python、概率、优化和 Transformer 基础；面向推理、数学/代码模型、test-time compute 与 reasoning API 方向的算法和工程岗位准备者。数学薄弱者回第十三册；需要训练实现可参考第五、十四册及第三册项目。
- **能力目标：** 能解释 CoT 与轨迹忠实性的边界；设计采样、verifier、搜索和预算策略；区分 outcome/process supervision 与可验证 RL；理解数学、代码推理中的不同 verifier；设计评估、安全和污染检查；核算 effort、延迟和单位成功成本；辨析 provider 的 thinking、状态保留及工具交错协议；阅读 R1、GLM-5.3 等指定模型路线的公开证据并限定结论。
- **纳入范围：** 推理机制与提示轨迹、采样/一致性/搜索、verifier 与过程监督、推理时计算扩展、数学和代码训练与执行反馈、评估和安全、reasoning effort 与动态预算、thinking levels 与状态协议、工具交错、多 Agent/并行计算，以及 DeepSeek-R1 和 GLM-5.3 的模型案例。
- **不承担范围：** 完整预训练/RL 系统归第五册；通用评估统计归第七册；安全治理归第八册；项目交付归第三册；通用 Agent 模式归第十七册；Harness/Runtime 长任务实现归第二十册；工具协议生态归第二十二册；Serving 和生产成本归第六/二十四册。provider API 字段作为有日期的合同示例，不写成模型内部机制或跨 provider 通用标准。
- **类型与独特价值：** 机制教材，以“生成候选—验证/搜索—训练—预算控制—评估证据”为主线，并用数学、代码及公开模型案例展示可验证与不可验证推理的差别；接口专题解释思考状态和工具事件的连续性，但不替代 Agent Runtime 手册。
- **前后交接：** 承接第二册推理进阶、第十三册数学和第十四册 PyTorch；通过第三册进入项目实践；将通用训练、评估、安全和 Serving 细节分别交第五、七、八、六/二十四册；Agent 执行、Harness 与工具协议分别交第十七、二十、二十二册。

## 简介审计（intro_audit）

- 现有简介只承诺 CoT、self-consistency、verifier、搜索、数学/代码训练、评估和安全；读者建议只有第三册项目，没有表达已扩展的 13—20 章：reasoning effort、thinking levels、thinking 状态保留、interleaved thinking、多 Agent/Ultra TTC、DeepSeek-R1、adaptive thinking 和 GLM-5.3 长任务环境。
- 现有能力目标把训练数据、动态预算、接口协议和模型案例压成少数泛化目标，没有解释 visible CoT 不等于忠实内部过程，也没有说明 API thinking state 与 Agent Runtime 的边界。
- 已更新[第十六册简介](../../book-16-reasoning-models/简介.md)：说明 20 章范围、读者先修、核心路径和专题选读；为各章补一行核心问题与交接，后续逐章精修时继续校准摘要。

## 目录审计（toc_audit）

- 现有目录仅列出 20 个文件链接，没有核心问题、依赖、读者产出、阅读顺序或跨册入口；新增章节 13—20 虽在文件清单中，但无法看出训练路线、test-time compute、provider 控制协议和长任务案例的分组。
- 已更新[第十六册目录](../../book-16-reasoning-models/目录.md)：提供核心路径（1—9、17、10—12）和按需专题路径；为 20 章列出问题、依赖/产出和交接；链接对应的配套书册。保留现有文件编号，避免破坏已有章节链接。

## 章节地图（chapter_map）

架构阶段仅依据简介、目录、文件名、章节标题和已有章节入口建立地图；未把章节正文当作已审计内容。每章的论证、篇幅、数例、公式、代码、证据和视觉呈现留待逐章审阅。

| 章 | 作用 | 依赖与交接 | 架构关注点 |
|---|---|---|---|
| 1 Reasoning 总览 | 建立推理任务、机制与证据对象地图。 | 起点；交给 CoT、验证、搜索和评估。 | 定义 reasoning、能力、可见轨迹和系统行为时不可混层。 |
| 2 Chain-of-Thought | 解释显式步骤提示及其适用边界。 | 接第 1 章；交给采样、过程监督和忠实性评估。 | 可见 CoT 不作为真实内部推理的直接读数。 |
| 3 Self-Consistency 与采样 | 从单次生成扩展到多候选和聚合。 | 依赖第 2 章；交给 verifier 与搜索。 | 区分候选命中率、投票和最终选择准确率。 |
| 4 Verifier 与 Reward Model | 解释结果/步骤的判定器及其不同监督语义。 | 接第 3 章；交给过程监督、RL 和测试时选择。 | verifier 与 reward model 的训练/运行角色可能重叠，但不能直接等同。 |
| 5 Process Supervision | 说明步骤级信号及错误定位。 | 依赖第 4 章；交给数学训练和 R1 路线。 | 步骤粒度、标注可靠性与 outcome supervision 边界。 |
| 6 搜索与 Tree-of-Thought | 以状态和转移组织搜索，比较 ToT、beam、best-first/A*、MCTS/UCT 及剪枝评估。 | 接第 3—5 章；交给第 7 章预算分析与第 16 章并行推理。 | 与普通采样、完整 MCTS、Agent 运行时的边界。 |
| 7 Test-Time Compute Scaling | 比较单轨迹延长、完整候选聚合、前缀搜索和工具策略，以覆盖—选择误差、质量曲线和停止收益决定额外计算是否值得。 | 接第 3—6 章；交给第 13 章预算账本、第 18 章动态控制和第 16 章并行推理。 | 已明确本章讲策略类别和质量—成本关系；第 13 章讲计量，第 18 章讲在线控制，第 16 章讲多路径协调。 |
| 8 数学推理训练 | 拆分题目、答案、步骤和来源，构造数学数据并验证训练信号与迁移。 | 依赖第 4—5 章和第十三册数学；交给第 17 章 RLVR 案例。 | 不替代第五册完整 RL/训练系统；答案、步骤和题目有效性分别审计。 |
| 9 代码推理与执行反馈 | 以规格、运行环境、测试和修复回路解释代码推理。 | 接第 4—7 章；交给长任务案例及第三册实践。 | 执行结果依赖测试 oracle、环境、权限和资源；与 Agent Runtime 分工。 |
| 10 Reasoning 评估 | 检查任务切片、指标、复现、污染和不确定性。 | 回用前九章；交给安全和综合案例。 | 通用统计方法转第七册，保留 reasoning 专属证据。 |
| 11 Reasoning 安全与局限 | 讨论过度自信、思维链忠实性与监控脆弱性、工具权限和高风险流程。 | 接第 2、4、9、10 章；交给第八册治理。 | 将监控结果限定在任务、线索来源、模型和监控器设置，不推成普遍内部机制结论。 |
| 12 Reasoning 综合复习 | 将生成、验证、搜索、成本与安全串成面试/系统判断。 | 回用第 1—11 章；作为核心路径收束。 | 后续专题需有明确回跳，不暗示本章已覆盖第 13—20 章细节。 |
| 13 Reasoning Effort 与推理预算 | 建立额外推理计算的计量和账本。 | 接第 7 章；交给策略选择和 serving 成本核算。 | 与第 18 章拆分静态预算维度和动态策略。 |
| 14 Thinking Levels | 分析离散思考档位与接口字段。 | 接第 13、18 章；交给模型/API 迁移评估。 | provider-specific level 不等于统一算力档或质量保证。 |
| 15 Preserve Thinking 与推理状态 | 处理跨轮 thinking block/state 的保留和继续使用。 | 接第 14 章；交给工具交错与第十七/二十册状态管理。 | API 状态协议不等于公开 raw chain-of-thought，也不等于完整 Runtime。 |
| 16 Multi-Agent 与 Ultra Test-Time Compute | 分析并行候选、协调、验证和额外计算。 | 接第 6—7 章；交给第十七/二十册多 Agent 与 Runtime。 | 聚焦推理计算结构，不扩成通用 Agent 编排教材。 |
| 17 DeepSeek-R1 的 RLVR 与蒸馏路线 | 以指定模型路线解释可验证 RL 与蒸馏。 | 接第 4—5、8—9 章；交给第五册训练专题。 | 论文/作者报告只能支持特定方案，不泛化为所有 reasoning 模型。 |
| 18 Adaptive Thinking 与动态预算 | 解释基于任务和反馈选择推理投入的控制环。 | 接第 7、13 章；交给第 14 章接口档位。 | 与第 13 章区分资源账本和在线分配策略。 |
| 19 Interleaved Thinking 与工具协议 | 描述推理、工具调用、观察、验证和恢复的事件顺序。 | 接第 9、15 章；交给第十七、二十、二十二册。 | 区分模型输出协议、宿主执行、权限和外部副作用。 |
| 20 GLM-5.3 长任务环境与验证器 | 用发布方披露的长任务环境、RL 与 verifier 作为案例。 | 接第 4、9—11、13、19 章；交给第十七/二十册。 | 明确发布方证据、未公开训练细节及与 Agent/Harness 的边界。 |

## 册级扫描（discovery_scan）

- **日期与时间窗：** 2026-11-12；回顾 2025—2026 推理训练、测试时计算、轨迹忠实性、并行推理和 reasoning API 协议。此扫描按书册定位、简介、目录和章节标题进行，不代替后续章级扫描。
- **扫描问题：** 新的 test-time compute 研究是否要求新增机制主线；可见 CoT 与模型内部推理、provider 控制项和状态协议如何分界；长任务 coding verifier 与第十七/二十册如何分工；多 Agent/并行 reasoning 是否属于本册核心。
- **资料类型：** 同行评审论文、arXiv 原始论文/预印本、官方模型/API 文档及发布方技术博客；搜索结果仅作资料入口，结论按具体来源、模型和协议限定。

| 候选主题 | 来源证据 | 去向与决定 |
|---|---|---|
| RLVR、冷启动、拒绝采样和蒸馏路线 | DeepSeek-R1 arXiv v2/Nature 论文报告 R1-Zero 与多阶段 R1 的不同训练路径及蒸馏；见[RLHF/策略优化研究记录](../../research/rlhf-preference-and-policy-optimization-2026.md)与[test-time compute 记录](../../research/test-time-compute-and-evaluation-2026.md)。 | 保留第 17 章作为模型特定训练案例；通用训练算法回到第 4、5、8 章，第五册承担完整训练系统。 |
| 搜索、verifier 和推理预算的交互 | Snell et al., ICLR 2025；Setlur et al., ICML 2025；Hariri et al., arXiv:2608.04001v2 的策略、验证和评估分类。 | 不新增章节；第 3—7 章讲单候选、聚合、验证、搜索和预算。第 7/13/18 章的控制对象必须在章级复核中拆清。 |
| 并行/协作推理和潜在空间测试时计算 | PaCoRe（ACL 2026）、ParaThinker（Findings EMNLP 2026）摘要及 Geiping et al.（NeurIPS 2025）已收录在[test-time compute 记录](../../research/test-time-compute-and-evaluation-2026.md)。 | 保留第 16 章作为高级推理计算专题；并行推理不外推为通用 Agent 架构。潜在空间计算作为第 7/16 章扫描候选，不据标题先增章。 |
| 轨迹忠实性、推理接口风险与安全测量 | ACL/EMNLP 2026 的 FACT-E、TRACE 及 reasoning-trace exposure 论文见既有[test-time compute 记录](../../research/test-time-compute-and-evaluation-2026.md)。 | 保留第 2、10、11 章的边界与评估职责；论文特定干预指标不宣称读取了完整内部思维。治理措施交第八册。 |
| Effort、thinking mode、thinking-state 保留和工具交错 | OpenAI [Reasoning 指南](https://developers.openai.com/api/docs/guides/reasoning)；Anthropic [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)与[Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)。访问日为 2026-11-12；来源边界已补入[test-time compute 研究记录](../../research/test-time-compute-and-evaluation-2026.md)。 | 保留第 13—15、18—19 章，但分别定义预算、动态控制、离散接口能力、状态延续和工具事件；不得把 provider 字段当作跨模型统一机制。 |
| GLM-5.3 长周期环境与 verifier | Z.ai 官方 GLM-5.3 博客披露任务环境、可解性检查、verifier 和训练/rollout 系统；[训练与后训练研究记录](../../research/llm-training-and-post-training-systems-2026.md)注明访问范围来自官方博客索引，且缺少复现实验条件。 | 保留第 20 章作发布方技术案例；明确其主张与验证边界，不替代第九章通用代码反馈、十七册 Agent 模式或二十册 Harness 实现。 |

**扫描结论：** 现有二十章足以覆盖机制、训练、测试时计算、评估和接口案例；没有证据要求另增独立主线或删除既有主题。当前首要结构问题是简介与目录没有为 13—20 章提供读者路径；以主题分组和边界表修正入口，保留编号，逐章扫描后再决定局部调整。

## 架构决定（architecture_decisions）

1. 保留现有 20 章及文件名，不因补写目录而移动正文；简介和目录以清楚的主题路径组织非线性选读。
2. 核心路径：第 1—7 章建立推理、验证、搜索和 test-time compute；第 8—9 章落到数学与代码任务；第 17 章作为可验证后训练案例；第 10—12 章收束评估、安全和综合回答。
3. 专题路径分为：预算控制（13、18、14）、thinking 状态和工具交错（15、19）、高级并行计算（16）、长周期模型案例（20）。目录明确回跳和跨书交接。
4. 第 7 章解释策略类别、候选覆盖与选择、质量—成本曲线；第 13 章解释资源计量和成本账本，第 18 章解释按难度/反馈在线分配，第 16 章解释并行路径及协调成本；第 14 章仅比较具体 provider 的离散控制字段，不把其 level 映射成统一算力单位。
5. 第 15 章描述 provider 对跨轮 thinking block/state 的公开接口合同；第 19 章描述推理与工具事件的顺序、观察和错误恢复。工具授权/协议及完整 runtime 分别交第十七、二十、二十二册。
6. 第 16 章以多路径/并行计算和验证为中心；Agent 团队协作、长期任务管理与工作区恢复分别交第十七、二十册。
7. 第 17/20 章保留为特定模型路线案例。DeepSeek-R1 的论文和 GLM-5.3 发布方资料只支持其具体训练/环境披露，不作为普遍 recipe 或能力排名。
8. 稳定读者承诺写入[简介](../../book-16-reasoning-models/简介.md)和[目录](../../book-16-reasoning-models/目录.md)；后续逐章扫描如发现边界改变，回到本审计更新骨架。

## 视觉机会（visual_map）

- 目录用阅读路径表说明核心、专题及配套书册依赖；第 3—7 章可用候选—verifier—搜索的状态图与预算账本区分方法。
- 第 8/9/17 章可用数学答案校验与代码执行环境的并排流程，明确 oracle 差别；第 10/11 章用证据链或评估维度表解释结果边界。
- 第 13 章说明资源计量，第 18 章以预算向量、效用算例和状态机表达动态控制；第 15/19 章用状态协议和工具事件顺序说明跨轮 thinking 与 observation。第十八章源码已有决策状态图和容量数例，本轮不另加装饰图。
- 后续逐章复读时判断现有图/表是否已足够；不预先硬塞总图。若使用 Mermaid，按 GitHub 读者入口完成真实渲染检查。

## 知识去向（coverage_map）

| 知识范围 | 第十六册入口 | 深入/配套去向 | 当前决定 |
|---|---|---|---|
| reasoning 基础与提示轨迹 | 第 1—2 章 | 第二册概览；第十册论文方法 | 本册讲机制和 CoT 忠实性边界。 |
| 采样、验证、搜索和 test-time compute | 第 3—7、13、16、18 章 | 第六/二十四册部署预算；第十七册 Agent | 第 7 章比较策略与质量曲线，第 13 章核算请求资源，第 18 章讲在线控制，第 16 章讲并行协调；Serving 实测交配套册。 |
| 数学与代码推理训练 | 第 8—9、17 章 | 第三册项目；第五册规模化训练；第十三册数学 | 本册讲任务信号和验证差异，不重写训练全流程。 |
| reasoning 评估与安全 | 第 10—11 章 | 第七册通用评估；第八册安全治理 | 保留推理任务、轨迹和专项风险，不代替统计/治理体系。 |
| effort、thinking level 与动态控制 | 第 13—14、18 章 | 第六/二十四册成本、接口实测 | 记录有日期的 provider API 合同，不推断内部算法。 |
| thinking 状态与工具交错 | 第 15、19 章 | 第十七册 Agent；第二十册 Runtime；第二十二册协议生态 | 解释模型/API 侧事件边界，完整宿主和协议实现外移。 |
| 长任务 RL 环境与 verifier | 第 20 章 | 第十七册 Code Agent；第二十册 Harness | 只做发布方案例审计，保留证据限制。 |

## 未决风险（open_risks）

1. 第 7、13、18 章已精修并确立策略/质量曲线、资源计量和成本账本、在线反馈控制的分工；第 18 章效用阈值与容量参数仍须在目标业务中校准，不得把合成例子当作线上建议。
2. 第 14、15、19 章涉及 provider thinking 协议；三章均已按 2026-11-13 可访问的官方文档核对档位、回放状态、历史兼容、usage 或工具往返，并明确 API surface、版本和证据边界。动态 API 合同仍可能变化，后续引用必须按具体模型/版本复核。
3. 第十六章已按多路径计算、协调成本和 verifier 边界精修；整册复读时再检查其与第十七册多 Agent 和第二十册 Harness 的交界。
4. 第十七章已绑定 DeepSeek-R1 固定版本、评测口径和未公开边界；第二十章已按发布方、第三方、固定源码和本地教学代码分层核对。GLM-5.3 专属 compaction、训练配方与托管服务实现仍未公开，不能据通用源码补推。
5. 全书存在曾发现的失效本地研究/演示链接；逐章审核时必须修复并更新链接状态，不能以旧记录替代复核。
6. 第 1—20 章已完成逐章审阅与精修并通过单篇门禁；尚须沿简介和目录中的阅读路径连续复读全册，检查依赖、重复、链接和跨册边界，再完成整册最终门禁。

## 工作交接（work_handoff）

- **证据记录：** 当前 thinking API 合同与 CoT 忠实性研究见[test-time compute 与评估研究记录](../../research/test-time-compute-and-evaluation-2026.md)；采样、self-consistency、min-p 和 Transformers 版本化接口证据见[解码采样与多路径推理研究记录](../../research/decoding-sampling-and-test-time-compute-2026.md)；verifier 质量、代码测试集覆盖与 reward hacking 证据见[Verifier 质量研究记录](../../research/verifier-quality-and-reward-hacking-2026.md)及[Reward Model 评估与过度优化记录](../../research/reward-model-evaluation-and-overoptimization-2026.md)；改写样本与 benchmark contamination 证据见[泛化与污染研究记录](../../research/generalization-and-benchmark-contamination-evidence-2026-11-09.md)；数学合成数据、MathDebugger、OpenWebMath 和 AlphaGeometry 证据见[数学与领域数据来源记录](../../research/specialized-code-math-domain-data-2026.md)；代码执行反馈、过程奖励、性能奖励和形式 oracle 证据见[代码执行反馈研究记录](../../research/code-execution-feedback-and-formal-oracles-2026-11-12.md)；推理链忠实性与监控规避评测见[安全与对齐研究记录](../../research/ai-safety-alignment-architecture-scan-2026-11-03.md)；工具调用、当前 MCP、OWASP 分类与重试幂等证据见[Agent 工具协议研究记录](../../research/agent-tool-calling-security-2026.md)；GLM-5.3 早期发布方证据见[训练与后训练研究记录](../../research/llm-training-and-post-training-systems-2026.md)，本轮接口、榜单和固定源码证据见[GLM-5.3 来源记录](../../research/model-update-2026-09/glm-5.3-source-notes.md)。第十五章状态续接证据见[test-time compute 与评估研究记录 §12.1](../../research/test-time-compute-and-evaluation-2026.md)；第十六章并行推理证据见该记录 §9。章节去向、范围判断和维护状态只记录在本目录。
- **已完成：** 对照《BOOK_SERIES.md》《ROADMAP.md》及第十五、十七册定位；建立契约；审读现有简介和目录；按 20 个章节文件名/标题整理章节地图；围绕 RLVR、TTS、provider thinking/state 和长任务 verifier 完成定向册级扫描；决定保留现有编号。第 1—20 章已完成通读、章级扫描和精修，细节见[第一章审计](chapter-01-audit.md)、[第二章审计](chapter-02-audit.md)、[第三章审计](chapter-03-audit.md)、[第四章审计](chapter-04-audit.md)、[第五章审计](chapter-05-audit.md)、[第六章审计](chapter-06-audit.md)、[第七章审计](chapter-07-audit.md)、[第八章审计](chapter-08-audit.md)、[第九章审计](chapter-09-audit.md)、[第十章审计](chapter-10-audit.md)、[第十一章审计](chapter-11-audit.md)、[第十二章审计](chapter-12-audit.md)、[第十三章审计](chapter-13-audit.md)、[第十四章审计](chapter-14-audit.md)、[第十五章审计](chapter-15-audit.md)、[第十六章审计](chapter-16-audit.md)、[第十七章审计](chapter-17-audit.md)、[第十八章审计](chapter-18-audit.md)、[第十九章审计](chapter-19-audit.md)和[第二十章审计](chapter-20-audit.md)。
- **已同步：** 简介和目录路径已更新；两份读者导航中的全部本地链接存在，目录列出 20/20 章。
- **第二十章审计：** 逐章扫描、篇幅、数例、代码、公式、证据、链接、视觉与配套状态见[第二十章审计](chapter-20-audit.md)；GLM-5.3 发布方、榜单、固定源码和本地 toy 的来源证据见[GLM-5.3 来源记录](../../research/model-update-2026-09/glm-5.3-source-notes.md)。
- **待完成：** 第十六册整册连续复读已覆盖第 1—18 章并回看第 17 章；随后复读第 19—20 章专题。完成后核查跨章依赖、重复、术语和前后交接并执行最终门禁。阶段记录见[整册连续复读记录](continuous-read.md)。
- **下一步：** 从第 19 章《Interleaved Thinking 与工具协议》继续连续复读；整册门禁完成后按全套书进度队列进入下一册。PDF 暂不更新，审计记录不回填 `plan.md`。
