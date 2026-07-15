# 内容广度与深度审计计划

## 目标

当前项目已经完成大规模正文写作和第二轮精修，下一阶段的重点不应只是格式、链接、PDF 或进度表收尾，而是回答三个更关键的问题：

1. 知识范围是否足够广，是否覆盖当前大模型算法岗、工程岗、推理岗、Agent 岗和 AI Infra 岗的主流知识面。
2. 最新且有影响力的知识点是否已经收录，是否避免只停留在旧一代 LLM 知识体系。
3. 内容深度是否分层合理：初学者能读懂来龙去脉，资深读者也能看到适用场景、优缺点、边界、失败模式和工程 trade-off。

本计划用于组织一次“内容广度与深度审计”，优先级高于普通格式收尾。

## 当前收口状态

截至 2026-07-15，本轮内容广度与深度审计已经完成阶段性收口：

1. 第一轮覆盖审计已经完成，结论已归并到本计划和 `PROGRESS.md`。
2. 覆盖矩阵已经完成，最终状态已归并到本计划和 `PROGRESS.md`。
3. P1 小补丁已完成：SimPO / RLVR / DeepSeek-R1、vAttention / FlashInfer、GraphRAG、WebArena / Browser Agent Eval、data contamination / benchmark leakage 等已补正文或纵向入口。
4. P2 抽样深度复核已完成：Mamba / SSM / hybrid architecture、多模态实时 / speech-to-speech / video generation、AI Infra future trends 三组主题无 P0/P1 主干缺口。
5. 中间审计文件已删除；当前仅保留观察项和后续维护项，不建议继续大面积补写正文。

## Frontier Model Release Radar：用新模型发布反推前沿知识点

用户提到 GPT-5.6、Fable 5 这类名字时，核心意图不是把某个未核验型号写进正文，而是用“frontier model release”作为锚点，反向挖掘当下大模型面试会追问的新知识。

因此后续维护不能只补 benchmark 名字，也不能只抄模型发布分数。每次 OpenAI、Anthropic、Google DeepMind、Meta、DeepSeek、xAI、Mistral、Qwen 等发布新模型或技术报告时，应按下面框架抽取新增知识：

1. 架构层：是否出现新的 attention 变体、MoE 形态、MLA / GQA / MQA 取舍、SSM / hybrid architecture、长上下文位置编码、跨模态统一架构或推理模型专用结构。
2. 预训练层：是否出现新的数据配比、合成数据、代码 / 数学 / 多模态数据路线、数据去重与污染治理、scaling law 修正、optimizer 或训练稳定性技巧。
3. 后训练层：是否出现新的 SFT、DPO / SimPO / KTO / ORPO、RLHF / RLAIF、RLVR、GRPO / DAPO / DrGRPO、verifier、process supervision、distillation 或 safety tuning 路线。
4. Test-time compute 层：是否强调 long thinking、self-consistency、search、tool-augmented reasoning、dynamic compute allocation、router、verifier rerank、budget-aware reasoning。
5. 推理与 serving 层：是否带来新的 KV cache 管理、speculative decoding、prefix / prompt cache、PD 分离、batching、低延迟多模态 streaming、端侧部署或成本优化技术。
6. Agent 与工具层：是否出现新的 browser / computer use、coding agent、tool calling、MCP / A2A / plugin / skill、trace replay、sandbox、权限治理和任务成功率评估方式。
7. 多模态层：是否出现新的原生多模态、实时语音、audio codec、video understanding / generation、vision-language-action、文档 / 图表 / OCR / 视频长上下文能力。
8. 评估层：是否新增或强化 benchmark，例如 HLE、BrowseComp、SimpleQA、SWE-Lancer、DeepSWE、RULER、FRAMES、MMMU、Video-MME，以及是否说明旧 benchmark 已饱和。
9. 安全与治理层：是否出现新的 system card、model spec、安全策略、危险能力评估、privacy / memorization、jailbreak / prompt injection、model release gate、监控和回滚机制。
10. 产品与工程层：是否暴露新的应用形态，例如 coding assistant、research agent、enterprise agent、realtime assistant、deep research、computer-use worker，以及它们对成本、延迟、可靠性和权限的要求。
11. 技术生命周期层：是否出现生态位迁移、抽象替代或能力吸收。例如某一阶段 MCP、tool protocol、plugin、skill、workflow、agent runtime 都可能承担“外部能力接入”的角色；但随着模型智能、上下文长度、记忆、工具调用稳定性和内置产品能力提升，一些外部框架可能从核心卖点退化为工程实现细节，甚至变成过渡技术。

每个新知识点进入正文前必须回答五个问题：

1. 它解决什么旧问题。
2. 相比上一代方法的核心变化是什么。
3. 适用场景和反模式是什么。
4. 有哪些优点、代价、失败模式和安全边界。
5. 应落到哪本书、哪一章、哪个面试题、哪个练习和哪个术语条目。

对“新技术是否值得写入主干”，需要额外判断它处于哪个生命周期阶段：

1. Emerging：新问题刚出现，方案还未稳定。适合写入观察项或论文路线，不急着写成标准范式。
2. Ecosystem Capture：开始抢占生态位，例如某个协议、工具包、skill 体系或 agent runtime 成为主流入口。适合补概念、架构图、适用场景和面试题。
3. Standardization：接口、权限、trace、评估、部署和治理形态稳定。适合进入正文主干和系统设计章节。
4. Absorbed by Model Capability：原本需要复杂外部 scaffold 的能力，被更强模型、更长上下文、更稳定工具调用、更好记忆或内置产品能力吸收。此时正文要讲“为什么不再需要它作为独立层”，而不是继续把它当新热点推广。
5. Obsolescent / Transient：生态热度下降、被替代或只剩少数场景需要。适合保留为历史脉络、反模式或 trade-off 案例。

面试表达要能讲清这种变化：

```text
我会区分一个技术是长期基础设施、短期 scaffold，还是被模型能力吸收的过渡形态。比如工具协议、plugin、skill、agent runtime 这类生态位会随模型能力、上下文长度、记忆和产品内置能力变化而迁移。面试里不能只说“最近流行什么”，还要说明它解决了什么约束、约束是否仍存在、如果模型本身变强后这层抽象会保留、下沉还是消失。
```

维护原则：

1. 未经官方文档、论文、system card、technical report 或高可信公开资料确认的模型名，不写成事实，只作为观察项。
2. 闭源模型没有披露训练细节时，只写“公开报告能支持的推断”，不把社区猜测写成确定结论。
3. benchmark 分数只作为入口，真正要补的是分数背后的能力、机制、工程条件和评估陷阱。
4. 对 MCP、A2A、plugin、skill、workflow、agent runtime 等生态技术，既要讲机制，也要讲生态位迁移、被替代风险和被模型能力吸收的可能性。
5. 面试准备要能回答“最近模型发布体现了哪些技术趋势，以及哪些热点可能只是过渡形态”，而不是只背“某模型在某榜单多少分”。

## 当前初步判断

从现有目录和抽样关键词看，本项目的一级主题覆盖已经较宽：

1. 基础与 Transformer。
2. 预训练、SFT、后训练和对齐。
3. 推理、部署和 serving engine。
4. 评估、实验和科学方法。
5. AI Safety、Alignment 和模型行为。
6. 数据工程、数据治理和数据价值。
7. 论文精读、复现和研究方法。
8. 系统设计面试。
9. 数学基础和 PyTorch 工程。
10. 多模态、Reasoning、Agent 和工具调用。
11. 产品化、商业化和真实工程坑。
12. Agent Harness、工具协议生态、AI Infra 和 LLM Inference Engine。

真正需要审计的是二级、三级知识点是否跟上前沿，以及每个关键主题是否讲到足够深。

抽样关键词显示，`GRPO`、`DAPO`、`KTO`、`ORPO`、`vLLM`、`SGLang`、`PagedAttention`、`RadixAttention`、`MCP`、`A2A`、`EAGLE`、`Medusa`、`SWE-bench`、`LLM-as-a-judge`、`unlearning` 等已有覆盖。但 `SimPO`、`DrGRPO`、`vAttention` 等命中较少或没有命中，`data contamination` 这类英文精确词命中少，需要确认中文内容是否已经充分覆盖。

## Radar Sweep 2026-07-15：frontier release 初扫

本次按 Frontier Model Release Radar 做了一次轻量联网初扫，重点不是记录榜单分数，而是提炼新模型发布背后的技术趋势。主要参考 OpenAI GPT-5 / GPT-5.6 官方页面、Anthropic Claude 4 / Fable 5 / Mythos 5 / Skills / Connectors 页面、Google Gemini 2.5 官方发布页、Meta Llama 4 官方技术博客，以及 DeepSeek-R1 公开资料。

### 已确认的新信号

1. OpenAI GPT-5 系统化强调“fast model + thinking model + router”的统一系统形态；GPT-5.6 进一步把 Sol / Terra / Luna 分层、`max` / `ultra` reasoning、多 agent 并行、programmatic tool calling、computer use、端到端知识工作、cyber / science 专业评估和 safe-completions 放进同一个发布叙事。
2. Anthropic Claude 4 已强调 hybrid reasoning、extended thinking with tool use、parallel tool execution、memory files、Claude Code、MCP connector、code execution 和 prompt caching；Claude 4 的 SWE-bench 方法说明还显示，相比上一代已经不再需要某些 planning tool scaffold，这是“模型能力吸收外部 scaffold”的直接例子。
3. Anthropic Fable 5 / Mythos 5 页面确认第五代模型线已进入 long-running agentic work、days-long coding / knowledge work、risk-calibrated safeguards、fallback routing 和 trusted access 叙事。Fable 偏一般长周期专业工作，Mythos 偏 cyber / biology 等高风险能力并限制开放。
4. Claude Skills 与 Connectors 形成生态层：Skills 用 `SKILL.md`、reference files、scripts 等让模型学习组织流程；Connectors 通过 MCP 把外部工具、数据库和应用接入 Claude。它们当前处在 Ecosystem Capture / Standardization 之间，但随着模型记忆、上下文和内置 agent 能力增强，部分 skill / workflow 可能被吸收为产品内置能力或普通上下文能力。
5. Google Gemini 2.5 强调 thinking model、增强 base model + improved post-training、把 thinking capability 内建到更多模型，以及 GPQA、AIME、HLE、SWE-bench Verified 等评估口径。
6. Meta Llama 4 强调 open-weight native multimodal、MoE、early fusion、MetaP、FP8 训练、30T+ token、多语言、mid-training、10M context、iRoPE、轻量 SFT -> online RL -> 轻量 DPO、adaptive filtering、teacher / codistillation、异步 online RL infrastructure，以及 GOAT 这类自动化对抗评估。
7. DeepSeek-R1 仍是 RLVR / GRPO / reasoning distillation 路线的重要公开锚点；项目中已覆盖主线，但后续可继续关注其后续技术报告是否把 GRPO / DAPO / DrGRPO 等路线进一步稳定化。

### 与当前书稿的覆盖对照

当前项目已覆盖的主线：

1. RLVR、GRPO、DAPO、DrGRPO、DeepSeek-R1、SimPO、DPO / PPO / RLHF。
2. MoE、Mamba / SSM / hybrid architecture、MLA / MQA / GQA、长上下文、RoPE scaling、位置外推。
3. vLLM、SGLang、PagedAttention、RadixAttention、PD 分离、prefix cache、KV cache、continuous batching、speculative decoding。
4. Function Calling、MCP、A2A、Skill、Plugin、Tool Registry、Tool Router、trace / replay、permission gate。
5. WebArena、OSWorld、SWE-bench、GAIA、tau-bench、ToolBench、BrowseComp、DeepSWE、SimpleQA、HLE、RULER、FRAMES、MMMU、Video-MME 等评估入口。

本次初扫暴露的新增 P1 / 观察项：

1. Programmatic Tool Calling：需要作为 tool-use 生态位迁移案例补入第二十二册或第十七册。它说明工具调用不一定总是“模型看见所有工具结果再继续思考”，部分中间处理可以变成程序化 workflow，降低 token、round trip 和上下文压力。
2. Multi-agent / Ultra as Test-time Compute：需要在第十六册 test-time compute 和第十七册 multi-agent 中补充“并行 agent 不是产品噱头，而是一种 budget-aware parallel search / workflow execution”的表达，同时强调成本、协调、验证和失败合并。
3. Fable / Mythos 风险分层与 fallback routing：需要在第八册 safety / 第十八册产品化 / 第二十三册发布治理中补充“能力越强，越需要 risk-calibrated access、自动 fallback、trusted access 和 data retention / monitoring 约束”。
4. Skills / Connectors / MCP 生命周期：第二十二册已讲 Skill 和 MCP，但还需要增加“生态位迁移”段落：Skills 可能抢占 prompt / workflow / plugin 的位置，也可能被更强模型、更长上下文和记忆吸收为较薄的组织知识层。
5. MetaP、iRoPE、10M context 和 long-context mid-training：第二册/第二十一册已覆盖长上下文和位置编码主线，但 MetaP、iRoPE 作为 Llama 4 官方发布信号，可以作为观察项补一段“前沿模型如何把架构、训练 recipe 和长上下文一起发布”。
6. GOAT / 自动化对抗评估：第八册已有 red teaming 和危险能力评估，但 Meta 的 GOAT 说明自动化、多轮、动态对抗测试正在成为发布前评估基础设施，应补为 safety eval 观察项。
7. 专业 eval 新簇：GeneBench-Pro、LifeSciBench、MedChemBench、SEC-Bench Pro、ExploitBench、ExploitGym、Agents' Last Exam、Terminal-Bench 2.1、AutomationBench 等应进入第七册 benchmark 观察项，但不宜一次性写成“全量主流标准”，需要继续确认公开定义、规模和稳定性。

本次雷达修补已完成轻量落地：

1. Programmatic Tool Calling 与 Skills / MCP 生命周期已补入第二十二册工具协议生态未来演进章。
2. Multi-agent / ultra as test-time compute 已补入第十六册 Test-Time Compute Scaling 章。
3. Risk-calibrated access 与 fallback routing 已补入第八册 Policy / Governance / Model Card 章。
4. MetaP、iRoPE、10M context 和 long-context mid-training 已作为前沿发布观察项补入第二十一册位置编码章。
5. GOAT、专业 eval 新簇和自动化对抗评估已补入第七册评估总览。
6. `INTERVIEW_BANK.md`、`EXERCISES.md` 和 `GLOSSARY_EN_ZH.md` 已补充对应面试题、练习和术语入口。

### 面试可复述结论

```text
我会把最近 frontier model release 看成技术雷达，而不是榜单。趋势上，模型发布正在从单模型能力对比，转向系统能力对比：router 选择 fast / thinking model，多 agent 并行作为 test-time compute，programmatic tool calling 减少 token 和 round trip，long-running coding agent 需要 memory、trace、sandbox 和 verification，安全上用 risk-calibrated fallback 和 trusted access 控制高风险能力。与此同时，MCP、Skills、Connectors、workflow、agent runtime 这些生态技术不是线性替代关系，它们会随模型能力、上下文、记忆和产品内置能力变化而迁移：有的会标准化为基础设施，有的会被模型能力吸收，有的只是过渡 scaffold。
```

## 审计任务一：前沿主题覆盖审计

建立一张“2024-2026 高影响主题覆盖表”，逐项标注：

1. 是否已覆盖。
2. 覆盖在哪本书、哪一章、哪一个纵向文件。
3. 是简单提到，还是讲清机制。
4. 是否有公式、demo、面试题、工程 trade-off。
5. 是否需要补节、补章、补百科条目、补题库或补交叉引用。

### Reasoning 与 RLVR

重点检查：

1. DeepSeek-R1。
2. GRPO。
3. DAPO。
4. DrGRPO。
5. RL with verifiable rewards / RLVR。
6. Process Reward Model / PRM。
7. Outcome Reward Model / ORM。
8. Verifier。
9. Test-time compute scaling。
10. Self-consistency。
11. Tree-of-Thought。
12. MCTS / search。
13. Reasoning distillation。
14. 数学、代码和工具增强推理。

审计问题：

1. 是否讲清 reasoning model 从 CoT、verifier、search 到 RLVR 的演进脉络。
2. 是否区分“生成推理文本”和“真实内部推理能力”。
3. 是否讲清 RLVR 为什么在数学、代码等可验证任务上更容易落地。
4. 是否讲清 GRPO、PPO、DPO、PRM/ORM 的关系和适用边界。
5. 是否有 test-time compute 的成本、延迟、收益递减和动态路由讨论。

### Preference Optimization 与 Alignment

重点检查：

1. RLHF。
2. RLAIF。
3. PPO。
4. DPO。
5. IPO。
6. KTO。
7. ORPO。
8. SimPO。
9. Reward hacking。
10. Constitutional AI。
11. Preference data quality。
12. Safety tuning。

审计问题：

1. 是否讲清 SFT、Reward Model、PPO、DPO 和新型 preference optimization 的关系。
2. 是否说明不同算法对数据质量、chosen/rejected 构造和 reference model 的要求。
3. 是否讲清 reward hacking、过度拒答、风格迁移和能力退化风险。
4. 是否有面向面试的算法对比表和工程选择建议。

### Inference Serving 与推理框架

重点检查：

1. vLLM。
2. PagedAttention。
3. SGLang。
4. RadixAttention。
5. TensorRT-LLM。
6. FlashInfer。
7. continuous batching。
8. prefix cache / prompt cache。
9. speculative decoding。
10. EAGLE。
11. Medusa。
12. KV cache offload。
13. KV cache quantization。
14. PD 分离 / disaggregated serving。
15. chunked prefill。
16. preemption、swap 和 recompute。
17. TTFT、TPOT、吞吐、显存和 SLO。
18. OpenAI-compatible API、streaming、限流、鉴权和灰度发布。

审计问题：

1. 是否从 naive generate 讲到 production serving engine。
2. 是否讲清 prefill 和 decode 资源画像不同。
3. 是否讲清 KV cache 是推理系统的核心瓶颈。
4. 是否比较 vLLM、SGLang、TensorRT-LLM 等框架的设计重点。
5. 是否有从单机 engine 到多 worker、多 GPU、分布式 serving 的升级路径。

### Agent、Tool Use 与协议生态

重点检查：

1. Function calling。
2. Structured outputs。
3. Tool schema。
4. MCP。
5. A2A。
6. Skill / plugin。
7. ReAct。
8. planning。
9. memory。
10. agentic RAG。
11. code agent。
12. browser agent。
13. computer use。
14. multi-agent。
15. SWE-bench。
16. WebArena。
17. sandbox。
18. trace / replay。
19. permission gate。
20. prompt injection 和 tool injection。

审计问题：

1. 是否把 Agent 讲成“模型 + 工具 + 状态 + 权限 + 评估 + trace”的系统，而不是简单 prompt 模板。
2. 是否讲清工具协议和普通 API 调用的区别。
3. 是否覆盖工具权限、租户隔离、副作用确认、审计和回滚。
4. 是否有 coding agent runtime / harness 的系统视角。
5. 是否覆盖 Agent 评估和线上事故复盘。

### RAG、数据和评估

重点检查：

1. hybrid retrieval。
2. dense retrieval。
3. BM25。
4. rerank。
5. query rewrite。
6. citation。
7. GraphRAG。
8. context assembly。
9. data contamination / benchmark leakage。
10. LLM-as-a-judge。
11. pairwise eval。
12. human eval。
13. regression test。
14. bad case taxonomy。
15. statistical significance。
16. eval data governance。

审计问题：

1. 是否讲清 RAG 不是“向量库 + prompt”，而是数据、索引、召回、重排、组装、引用、权限和评估闭环。
2. 是否讲清评估集污染、benchmark leakage 和线上回归测试。
3. 是否讲清 LLM-as-a-judge 的偏差、校准和适用边界。
4. 是否有面向工程落地的 bad case 分类和修复闭环。

### 模型架构与训练系统

重点检查：

1. MHA、MQA、GQA、MLA。
2. RoPE、RoPE scaling、ALiBi。
3. MoE。
4. Mamba / SSM。
5. RWKV。
6. RetNet。
7. Hyena。
8. Linear Attention。
9. hybrid architecture。
10. optimizer：AdamW、Muon 等。
11. distributed training：DP、TP、PP、ZeRO、FSDP。
12. activation checkpointing。
13. mixed precision。
14. training stability。
15. data mixture。
16. scaling law。

审计问题：

1. 是否讲清 Transformer 为什么成为主流，以及后 Transformer 路线解决什么问题。
2. 是否讲清架构创新的收益、代价、硬件友好性和生态成熟度。
3. 是否避免把每个新架构写成“替代 Transformer”的简单叙事。
4. 是否有训练系统和模型结构之间的联动分析。

### 多模态、语音、视频和安全

重点检查：

1. CLIP。
2. VLM。
3. vision encoder。
4. multimodal projector / connector。
5. 多模态 instruction tuning。
6. diffusion。
7. Stable Diffusion / DALL-E。
8. video generation。
9. world model。
10. speech-to-text。
11. text-to-speech。
12. speech-to-speech / realtime assistant。
13. unified multimodal model。
14. multimodal eval。
15. multimodal safety。
16. watermarking。
17. unlearning。
18. privacy / memorization。
19. model card / governance。

审计问题：

1. 是否覆盖文本 LLM 到图像、视频、语音和实时多模态助手的演进。
2. 是否讲清多模态 token 成本、延迟、评估和安全问题。
3. 是否讲清生成模型和理解模型的不同训练目标与工程约束。
4. 是否覆盖多模态产品落地和系统设计题。

## 审计任务二：内容深度分层审计

每个关键主题按四层打分，避免只看“有没有提到”。

### 初学者层

检查点：

1. 是否说明这个概念为什么出现。
2. 是否讲清它要解决什么问题。
3. 是否有直觉类解释、简单例子或类比。
4. 是否避免一上来堆公式和术语。
5. 是否能让第一次接触该主题的读者读懂主线。

评分：

1. 0 分：只给术语或结论。
2. 1 分：有简短定义，但缺背景。
3. 2 分：有背景和直觉，但例子不足。
4. 3 分：背景、直觉、例子完整，小白可读。

### 机制层

检查点：

1. 是否有核心流程。
2. 是否有关键公式。
3. 是否解释变量含义。
4. 是否有最小可运行 demo 或伪代码。
5. 是否说明输入、输出、shape、状态变化或指标含义。

评分：

1. 0 分：没有机制解释。
2. 1 分：有流程描述，但不够精确。
3. 2 分：有公式或代码，但解释不足。
4. 3 分：公式、流程、代码和解释完整。

### 工程层

检查点：

1. 是否说明适用场景。
2. 是否说明不适用场景。
3. 是否给出实现坑、debug 方法或指标。
4. 是否讨论成本、延迟、显存、数据质量、稳定性或安全约束。
5. 是否能指导真实项目落地。

评分：

1. 0 分：没有工程讨论。
2. 1 分：只有泛泛而谈。
3. 2 分：有工程指标或坑，但不系统。
4. 3 分：场景、指标、坑、排查和取舍完整。

### 资深层

检查点：

1. 是否比较相邻方法。
2. 是否讲清优点、缺点和 trade-off。
3. 是否讲清失败模式和反模式。
4. 是否讨论前沿争议、资料边界或未验证说法。
5. 是否有面试深挖点和标准回答模板。

评分：

1. 0 分：没有资深视角。
2. 1 分：有优缺点，但很浅。
3. 2 分：有 trade-off 和对比，但缺失败场景。
4. 3 分：能体现方法判断力、工程判断力和面试表达能力。

## 审计任务三：补缺优先级排序

审计后不要发现一个缺口就立刻补一个缺口，而是按影响力排序。

### A 类：必须补

标准：

1. 当前大模型岗位高频。
2. 影响模型训练、后训练、推理、评估、Agent 或 AI Infra 的核心能力。
3. 已经在主流论文、框架、官方文档或真实系统中稳定出现。
4. 缺失会明显影响面试竞争力。

处理方式：

1. 补正文小节或章节。
2. 同步百科、题库、练习、术语表和项目路线。
3. 必要时补 demo。

### B 类：高级加分

标准：

1. 高频度略低，但能体现研究视野或资深工程视角。
2. 适合系统设计、论文讨论或开放研究题。
3. 有一定资料支撑，但不一定是所有岗位必问。

处理方式：

1. 优先补百科、论文路线、面试题或对比表。
2. 视情况补专题章节中的“面向专家”小节。

### C 类：趋势观察

标准：

1. 新但尚未稳定。
2. 社区讨论多，但工业落地和公开证据有限。
3. 容易过时或存在不同解释。

处理方式：

1. 写入“趋势、争议和待核验”小节。
2. 明确区分官方披露、论文结论、社区推测和个人判断。
3. 不写成确定结论。

### D 类：暂不纳入

标准：

1. 噪声热词。
2. 缺少可靠资料。
3. 和项目目标关系弱。
4. 容易让读者分散注意力。

处理方式：

1. 不纳入正文。
2. 最多在 `ideas.md` 或论文线索中保留待观察。

## 已产出材料

本轮内容审计已经产出以下正式留存材料或章节更新：

1. `plan.md`：保留本轮审计目标、执行状态、观察项和后续维护建议。
2. `PROGRESS.md`：保留本轮审计、P1/P2/P3 收口和中间文件删除记录。
3. `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 已完成必要补充。
4. 对 A/P1 缺口明显的专题章节已做最小补节，避免大面积重写。
5. 临时覆盖审计、矩阵和深度复核材料已归并后删除，不作为长期项目文件保留。

## 执行顺序与完成状态

本轮按下面顺序推进，目前均已完成：

1. 先建立前沿主题清单，联网核对高影响主题。
2. 再用关键词和目录扫描映射到现有章节。
3. 对每个主题标注覆盖等级：未覆盖、提到、讲清、讲深、有 demo、有题库。
4. 输出 A/B/C/D 补缺优先级。
5. 先补 A 类缺口，再补 P1 小补丁。
6. 对 P2 主题做抽样深度复核。
7. 最后做审计材料、进度记录、代码围栏、格式和风险关键词的轻量收尾验证。

## 当前优先审计清单状态

第一批建议审计主题的处理状态：

1. GRPO / DAPO / DrGRPO / RLVR：已补对比表，DrGRPO 保持观察项。
2. SimPO / KTO / ORPO / DPO 对比：已补 SimPO 入口和谱系定位。
3. DeepSeek-R1 类 reasoning RL 和 distillation：已补显式入口。
4. Test-time compute scaling 和动态路由：已有覆盖，本轮未发现主干缺口。
5. vLLM / SGLang / TensorRT-LLM / FlashInfer 对比：已补 FlashInfer 层次说明。
6. PagedAttention / RadixAttention / prefix cache / KV offload：已补 vAttention；KV offload / quantization 保留后续横向小表建议。
7. EAGLE / Medusa / speculative decoding：已有覆盖，本轮未列为缺口。
8. MCP / A2A / structured outputs / tool schema：已有强覆盖，后续只需跟随官方规范变化维护。
9. Computer use / browser agent / coding agent / SWE-bench：已补 WebArena / Browser Agent Eval 入口。
10. Agent safety：权限、沙箱、trace、prompt injection、tool injection：已有强覆盖。
11. GraphRAG、hybrid retrieval、rerank、citation 和 RAG eval：已补 GraphRAG 正文和系统设计交叉引用。
12. LLM-as-a-judge、benchmark leakage、data contamination：已补 contamination / leakage 英文入口。
13. MLA、MoE、Mamba/SSM、Linear Attention 和 hybrid architecture：P2 抽样确认 Mamba/SSM/hybrid 深度达标。
14. 多模态实时助手、speech-to-speech、video/world model：P2 抽样确认覆盖达标，未稳定公开的实时语音传闻列观察。
15. unlearning、watermarking、privacy、model card 和 governance：已有覆盖，未列为本轮补丁。

## 判断标准

最终目标不是让书变成“所有热词合集”，而是让读者获得稳定、可迁移、可面试、可落地的知识体系。

一个知识点是否值得纳入，应看：

1. 是否代表重要技术趋势。
2. 是否影响主流岗位面试。
3. 是否能帮助理解真实工程系统。
4. 是否已有足够可靠资料支撑。
5. 是否能被放进现有知识图谱，而不是孤立堆叠。

一个章节是否合格，应看：

1. 初学者是否能读懂它为什么出现。
2. 中级读者是否能掌握机制和实现。
3. 工程读者是否知道什么时候用、什么时候不用。
4. 资深读者是否能看到 trade-off、失败模式和面试追问。
5. 纵向文件是否同步：百科、题库、练习、术语、项目路线和论文路线。
