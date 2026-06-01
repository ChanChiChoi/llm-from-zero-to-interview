# OpenAI 大模型算法岗系统训练工程

这是一套面向大模型算法岗面试与能力建设的开源书系。

它不是零散笔记，而是一个完整训练系统：从机器学习基础、Transformer、LLM 训练、对齐、推理部署、推理框架、AI Infra、评估安全，到论文精读、项目实战、系统设计、工具协议生态和模拟面试，逐步帮助读者建立大模型算法岗所需的理论、工程、研究和表达能力。

这套书的宗旨不是罗列知识点，而是帮助读者建立一张可迁移的大模型知识图谱。重要概念会尽量讲清历史背景、提出动机、前人方法的问题、当前方法的特点、优缺点、适用边界、后来者的改进方向以及和相邻概念的关系。项目默认小白友好，但目标是面向全阶段读者：初学者能看懂来龙去脉，中级读者能串起系统框架，高阶读者能看到机制细节、工程 trade-off 和面试深挖点。

适合读者：

1. 想系统准备大模型算法岗面试的同学。
2. 想从传统算法工程师、后端工程师、数据科学家转向 LLM 方向的工程师。
3. 想补齐 LLM 原理、训练、对齐、推理、评估和工程落地体系的学习者。
4. 想建立大模型研究与工程全局视角，并逐步形成项目作品集的初学者。

## 当前进度

当前阶段：第二轮全系列精修阶段。

进度摘要：

1. 24 本主书均已完成正文第一版，已打通从基础、训练、部署、评估、安全、数据、论文、系统设计、求职、数学、PyTorch、多模态、Reasoning、Agent、产品化、实战坑、Agent Runtime、架构演进、工具协议、AI Infra 到推理框架的完整闭环。
2. 第一册 `book-01-core-30/` 已完成核心 38 讲，第二册 `book-02-advanced-100/` 已完成进阶 120 讲，第三册 `book-03-practical-handbook/` 已完成实战 72 讲。
3. 第四册到第十九册已完成正文第一版，其中第四册定位为后续持续回填和补全的概念索引。
4. 第二十册到第二十四册已完成正文第一版，分别覆盖 Agent Harness、Transformer 架构演进、工具协议生态、AI Infra 和 LLM Serving Engine。
5. `book-llm-engineer/` 已完成大模型工程师面试补充篇第一版，覆盖第 60-82 章。
6. 第二轮精修已启动，重点是公式兼容性、demo 级代码补充、联网校验、广度审计、深度增强、术语和题库同步。

详细进度见：[`PROGRESS.md`](PROGRESS.md)

## 书系总览

本项目规划 24 本书。每本书解决一个明确问题：有的用于打基础，有的用于做项目，有的用于补训练、推理、评估、Infra、Agent、工具协议和系统设计能力。

如果你刚打开这个仓库，可以先扫下面这张表，找到自己最感兴趣或最薄弱的方向，再进入对应目录阅读。

| 序号 | 书名 | 简介 | 目录 |
|---|---|---|---|
| 1 | OpenAI 大模型算法岗面试核心 38 讲 | 从零建立大模型算法岗的最小完整知识闭环，覆盖 Transformer、训练、对齐、推理、评估、多模态和模拟面试。适合第一次系统准备 LLM 面试的读者。 | [`book-01-core-30/`](book-01-core-30/) |
| 2 | OpenAI 大模型算法岗进阶 120 讲 | 在第一册基础上补齐研究深度和前沿视野，系统展开架构、训练、对齐、推理、评估、安全、Agent 和多模态等高频进阶问题。适合冲刺更高难度面试。 | [`book-02-advanced-100/`](book-02-advanced-100/) |
| 3 | 大模型算法岗实战手册 | 把理论转成可运行代码、可复现实验、可展示项目和面试表达。覆盖 PyTorch、Transformer、miniGPT、微调、偏好优化、推理优化、RAG、Agent、评估 debug、多模态实战。 | [`book-03-practical-handbook/`](book-03-practical-handbook/) |
| 4 | 大模型百科全书 | 作为全书系概念索引和速查手册，最终应覆盖其他各册出现的核心知识点，并给出简明介绍、相关概念和面试入口。它可以放到后期持续补全，不要求先于专题书完整写完。 | [`book-04-llm-encyclopedia/`](book-04-llm-encyclopedia/) |
| 5 | 大模型训练全流程 | 聚焦从数据准备、预训练、SFT、偏好优化到训练监控和 debug 的完整链路。适合想理解大模型训练工程和训练事故排查的读者。 | [`book-05-llm-training/`](book-05-llm-training/) |
| 6 | 大模型部署与推理工程 | 讲清模型从 checkpoint 到线上服务的过程，包括推理优化、量化、并发、延迟、吞吐、服务化和生产稳定性。适合准备推理部署和工程落地岗位。 | [`book-06-llm-deployment/`](book-06-llm-deployment/) |
| 7 | 大模型评估、实验与科学方法 | 系统训练评估集设计、指标选择、实验对照、误差分析、A/B 测试和上线门禁能力。适合想把“模型好不好”讲清楚的读者。 | [`book-07-evaluation-experiments/`](book-07-evaluation-experiments/) |
| 8 | AI Safety、Alignment 与模型行为 | 覆盖对齐、安全、拒答、越狱、偏见、红队、模型行为控制和风险治理。适合准备 alignment、safety 和模型行为相关面试。 | [`book-08-ai-safety-alignment/`](book-08-ai-safety-alignment/) |
| 9 | 大模型数据工程与数据智能 | 聚焦大模型数据的采集、清洗、去重、质量评估、合成数据、数据配比和数据闭环。适合想理解“数据如何决定模型能力”的读者。 | [`book-09-data-engineering/`](book-09-data-engineering/) |
| 10 | 论文精读、复现与研究方法 | 训练如何读论文、拆贡献、复现实验、做 ablation、写研究笔记和形成自己的研究判断。适合准备研究型岗位和高阶面试。 | [`book-10-paper-reproduction-research/`](book-10-paper-reproduction-research/) |
| 11 | 大模型系统设计面试 | 面向系统设计题，训练如何设计 RAG、Agent、推理服务、训练平台、评估平台、数据平台和多模态系统。适合集中突破架构表达能力。 | [`book-11-system-design-interview/`](book-11-system-design-interview/) |
| 12 | OpenAI 大模型算法岗求职与面试作战手册 | 聚焦简历、项目包装、面试节奏、英文表达、行为面、模拟面试和 offer 策略。适合临近投递和面试冲刺阶段阅读。 | [`book-12-career-interview-playbook/`](book-12-career-interview-playbook/) |
| 13 | 大模型数学基础 | 补齐线性代数、概率统计、信息论、优化、矩阵计算和深度学习常用数学工具。适合数学基础不稳、看论文容易卡住的读者。 | [`book-13-math-foundations/`](book-13-math-foundations/) |
| 14 | PyTorch 与深度学习工程 | 从 PyTorch 基础走到训练工程实践，覆盖 tensor、autograd、module、dataloader、训练循环、混合精度和工程 debug。适合补代码实现能力。 | [`book-14-pytorch-deep-learning-engineering/`](book-14-pytorch-deep-learning-engineering/) |
| 15 | 多模态与生成模型专题 | 系统覆盖 CLIP、VLM、Diffusion、语音、图像、视频和多模态评估。适合想从文本 LLM 扩展到多模态大模型的读者。 | [`book-15-multimodal-generative-models/`](book-15-multimodal-generative-models/) |
| 16 | Reasoning Model 专题 | 聚焦推理模型、CoT、verifier、search、test-time scaling、数学推理、代码推理和复杂任务求解。适合关注 reasoning 能力和新一代模型范式的读者。 | [`book-16-reasoning-models/`](book-16-reasoning-models/) |
| 17 | Agent 与工具调用专题 | 讲解 Agent loop、tool use、function calling、规划、记忆、反思、多 Agent 协作和工具调用评估。适合准备 Agent 工程和智能体应用方向。 | [`book-17-agent-tool-use/`](book-17-agent-tool-use/) |
| 18 | 大模型产品化、商业化与落地 | 从产品、业务、成本、ROI、场景选择和交付角度理解大模型落地。适合想从技术走向真实业务价值的读者。 | [`book-18-product-business-commercialization/`](book-18-product-business-commercialization/) |
| 19 | 资深大模型工程师实战宝典 | 汇总真实工程中的坑、事故、排查路径、架构取舍和资深表达方式。适合从“会做 demo”升级到“能负责生产系统”的读者。 | [`book-19-practitioner-playbook/`](book-19-practitioner-playbook/) |
| 20 | Agent Harness、Coding Agent Runtime 与智能体工程框架 | 聚焦 coding agent、harness、runtime、sandbox、任务编排、评估框架和工程约束。适合想理解 Claude Code、OpenCode、Codex 类系统背后工程设计的读者。 | [`book-20-agent-harness-runtime/`](book-20-agent-harness-runtime/) |
| 21 | Transformer 架构详解、升级变种与未来架构演进 | 系统梳理 Transformer 结构、长上下文、MoE、Mamba、SSM、RWKV、RetNet、Hyena、Linear Attention 和混合架构。适合想深入模型架构演进的读者。 | [`book-21-transformer-architecture-evolution/`](book-21-transformer-architecture-evolution/) |
| 22 | Function Calling、MCP、A2A、Skill 与工具协议生态 | 聚焦工具 schema、工具注册、MCP server、跨 Agent 协议、插件/Skill 生态、权限安全和协议层系统设计。适合关注工具协议和 Agent 生态的读者。 | [`book-22-tool-protocol-ecosystem/`](book-22-tool-protocol-ecosystem/) |
| 23 | AI Infra、大模型基础设施与平台工程 | 覆盖 GPU 集群、网络、存储、调度、训练平台、推理平台、数据平台、实验平台、可观测性和成本治理。适合准备 AI Infra 和平台工程方向。 | [`book-23-ai-infra/`](book-23-ai-infra/) |
| 24 | 大模型推理框架与 Serving Engine 实战 | 从 0 实现推理框架，并系统学习 vLLM、SGLang、PagedAttention、continuous batching、KV Cache 管理和 PD 分离。适合想深入 LLM serving engine 的读者。 | [`book-24-llm-inference-engine/`](book-24-llm-inference-engine/) |

完整书系索引见：[`BOOK_SERIES.md`](BOOK_SERIES.md)

## 怎么开始读

如果你不知道从哪里开始，按目标选择：

| 目标 | 推荐入口 |
|---|---|
| 第一次系统学习大模型面试 | [`book-01-core-30/目录.md`](book-01-core-30/目录.md) |
| 已有基础，想冲刺高难度问题 | [`book-02-advanced-100/目录.md`](book-02-advanced-100/目录.md) |
| 想做项目、写代码、准备简历作品 | [`book-03-practical-handbook/目录.md`](book-03-practical-handbook/目录.md) |
| 概念太多，想随时查术语 | [`GLOSSARY_EN_ZH.md`](GLOSSARY_EN_ZH.md) 和 [`book-04-llm-encyclopedia/目录.md`](book-04-llm-encyclopedia/目录.md) |
| 想补数学 | [`book-13-math-foundations/目录.md`](book-13-math-foundations/目录.md) |
| 想补 PyTorch 和训练代码 | [`book-14-pytorch-deep-learning-engineering/目录.md`](book-14-pytorch-deep-learning-engineering/目录.md) |
| 想准备系统设计面试 | [`book-11-system-design-interview/目录.md`](book-11-system-design-interview/目录.md) |
| 想深入推理框架、vLLM、SGLang | [`book-24-llm-inference-engine/目录.md`](book-24-llm-inference-engine/目录.md) |
| 想看完整学习路线 | [`ROADMAP.md`](ROADMAP.md) |

推荐基础阅读顺序：

1. [`background.md`](background.md)：了解项目目标和写作定位。
2. [`ROADMAP.md`](ROADMAP.md)：选择 3 个月、6 个月或 12 个月学习路线。
3. [`book-01-core-30/目录.md`](book-01-core-30/目录.md)：从核心 38 讲建立主干知识。
4. [`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)：配合每一讲做面试题训练。
5. [`EXERCISES.md`](EXERCISES.md)：用练习检验自己是否真的掌握。
6. [`book-02-advanced-100/目录.md`](book-02-advanced-100/目录.md)：进入进阶原理和前沿研究。
7. [`book-03-practical-handbook/目录.md`](book-03-practical-handbook/目录.md)：把理论转成代码、实验和项目作品集。

## 已完成重点内容

24 本主书已经完成正文第一版。第一册是核心入门书，覆盖：

1. LLM 入门与 next-token prediction。
2. 机器学习基础、交叉熵、优化器和反向传播。
3. Tokenization、Embedding、Self-Attention、MHA、Causal Mask、Transformer Block、RoPE。
4. miniGPT 实现思路。
5. 预训练、Scaling Law、分布式训练。
6. Instruction Tuning、SFT、RLHF、DPO、Reward Model。
7. Decoding、KV Cache、FlashAttention、量化与部署。
8. 幻觉、评估、安全、开放研究问题和模拟面试复盘。
9. 多模态基础，包括 CLIP、VLM、Diffusion、语音和视频生成。
10. OpenAI 风格开放题与 12 周总复习。

第二册覆盖更细的进阶主题，包括现代 Transformer 变体、长上下文、MoE、SSM/S4/Mamba、Reasoning、Agent、多模态、评估、安全和研究前沿。

第三册目标是把前两册的知识转成代码、实验、项目和面试表达。已覆盖 PyTorch 基础、Transformer 组件、miniGPT、Hugging Face 微调、偏好优化、推理优化、RAG 与 Agent、评估 debug、简历项目、面试实战训练和多模态实战。

第五册、第六册和第七册也已完成正文第一版，分别覆盖训练全流程、部署推理工程，以及大模型评估、实验设计、A/B 测试、统计显著性、error analysis 和评估面试题。

第八册到第十九册分别覆盖 AI Safety 与 Alignment、数据工程、论文复现、系统设计、求职面试、数学基础、PyTorch 工程、多模态、Reasoning、Agent、产品落地和资深工程师实战坑。

第二十册到第二十四册分别覆盖 Agent Harness 与 Coding Agent Runtime、Transformer 架构演进、Function Calling/MCP/A2A/Skill 工具协议生态、AI Infra 平台工程，以及大模型推理框架与 Serving Engine 实战。

第四册《大模型百科全书》不是普通专题书，而是全书系的概念索引。其他各册里出现的重要知识点，最终都应该在第四册中有简明介绍，但不需要和专题正文同步写到同等深度；更合理的方式是先扩写专题书，再把成熟知识点回填到百科中。

## 配套训练文件

除了 24 本书，本项目还包含一组纵向训练资料。

| 文件 | 用途 |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | 3 个月、6 个月、12 个月学习路线图。 |
| [`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md) | 大模型知识图谱，帮助理解概念依赖关系。 |
| [`PROJECTS.md`](PROJECTS.md) | 项目路线与实战项目规划，用于形成简历作品集。 |
| [`PAPERS.md`](PAPERS.md) | 论文阅读路线，帮助建立研究视角。 |
| [`INTERVIEW_BANK.md`](INTERVIEW_BANK.md) | 集中面试题库，用于自测和模拟面试。 |
| [`EXERCISES.md`](EXERCISES.md) | 练习与阶段验收体系，检查是否真正掌握。 |
| [`GLOSSARY_EN_ZH.md`](GLOSSARY_EN_ZH.md) | 中英文术语表，方便读论文和英文面试。 |
| [`ENGLISH_INTERVIEW_TEMPLATES.md`](ENGLISH_INTERVIEW_TEMPLATES.md) | 英文面试表达模板。 |
| [`WRITING_PLAN.md`](WRITING_PLAN.md) | 写作计划与扩写规范。 |
| [`PROGRESS.md`](PROGRESS.md) | 当前进度追踪。 |

## 项目结构

```text
.
├── README.md
├── BOOK_SERIES.md
├── ROADMAP.md
├── INTERVIEW_BANK.md
├── EXERCISES.md
├── PROJECTS.md
├── PAPERS.md
├── GLOSSARY_EN_ZH.md
├── PROGRESS.md
├── book-01-core-30/
├── book-02-advanced-100/
├── book-03-practical-handbook/
├── book-04-llm-encyclopedia/
├── ...
├── book-20-agent-harness-runtime/
├── book-21-transformer-architecture-evolution/
├── book-22-tool-protocol-ecosystem/
├── book-23-ai-infra/
└── book-24-llm-inference-engine/
```

## 学习方法

建议不要只读正文，而是按“章节学习 + 面试表达 + 练习验收”的方式推进。

每学一讲，建议完成：

1. 用自己的话复述核心概念。
2. 回答对应面试题。
3. 做对应练习。
4. 写下一个容易混淆的点。
5. 用 2 分钟模拟面试回答一次。

如果目标是面试，建议每周至少做一次 mock interview，并用 [`PROGRESS.md`](PROGRESS.md) 或自己的笔记记录复盘。

## 写作原则

本项目会尽量保持以下风格：

1. 从小白能理解的直觉讲起。
2. 不回避公式，但不堆公式。
3. 每个主题都连接到面试问题和工程场景。
4. 强调 trade-off、实验意识、数据意识和工程约束。
5. 同时覆盖研究视角、工程视角和面试表达。

## 后续计划

近期重点：

1. 按 `SECOND_PASS_REVIEW_PLAN.md` 推进第二轮全系列精修。
2. 从第一册开始逐章修正数学公式、变量说明和 GitHub Markdown 渲染问题。
3. 为适合代码辅助理解的核心知识点补充 demo 级 Python 示例，并验证可运行性。
4. 对高时效主题做联网校验，修正过时、片面或证据不足的表述。
5. 同步维护第四册百科、面试题库、练习体系、项目路线、论文路线、术语表和知识图谱。

## 适用边界

这个项目是学习与面试训练资料，不是 OpenAI 官方资料，也不代表任何公司的招聘标准。

大模型领域变化很快，部分内容会随着研究进展持续更新。阅读时建议结合最新论文、官方文档和真实工程实践一起理解。

## License

本项目内容免费开放，欢迎学习、阅读、引用和分享。

如果后续需要用于更正式的开源分发、转载或商业场景，建议补充标准开源协议文件，例如 `LICENSE`。
