# OpenAI 大模型算法岗系统面试二十四部曲

这个目录下规划二十四本书，并配套学习路线、知识图谱、项目路线、论文路线、面试题库、练习验收、术语表、英文面试模板和进度追踪。

目标是系统辅导算法工程师从大模型初学者成长到能面试 OpenAI 等顶级机构大模型算法岗的候选人。

## 二十四本书定位

1. `book-01-core-30/`：第一册，核心 38 讲，建立面试最小完整闭环。
2. `book-02-advanced-100/`：第二册，进阶 120 讲，补齐研究深度和前沿视野；其中第二部分显式覆盖现代 Transformer 变体、SSM/S4/Mamba、混合架构等替代路线。
3. `book-03-practical-handbook/`：第三册，实战手册，把理论转成代码和项目。
4. `book-04-llm-encyclopedia/`：第四册，大模型百科全书，用于概念速查。
5. `book-05-llm-training/`：第五册，大模型训练全流程。
6. `book-06-llm-deployment/`：第六册，大模型部署与推理工程。
7. `book-07-evaluation-experiments/`：第七册，大模型评估、实验与科学方法。
8. `book-08-ai-safety-alignment/`：第八册，AI Safety、Alignment 与模型行为。
9. `book-09-data-engineering/`：第九册，大模型数据工程与数据智能。
10. `book-10-paper-reproduction-research/`：第十册，论文精读、复现与研究方法。
11. `book-11-system-design-interview/`：第十一册，大模型系统设计面试。
12. `book-12-career-interview-playbook/`：第十二册，OpenAI 大模型算法岗求职与面试作战手册。
13. `book-13-math-foundations/`：第十三册，大模型数学基础。
14. `book-14-pytorch-deep-learning-engineering/`：第十四册，PyTorch 与深度学习工程。
15. `book-15-multimodal-generative-models/`：第十五册，多模态与生成模型专题。
16. `book-16-reasoning-models/`：第十六册，Reasoning Model 专题。
17. `book-17-agent-tool-use/`：第十七册，Agent 与工具调用专题。
18. `book-18-product-business-commercialization/`：第十八册，大模型产品化、商业化与落地。
19. `book-19-practitioner-playbook/`：第十九册，资深大模型工程师实战宝典。
20. `book-20-agent-harness-runtime/`：第二十册，Agent Harness、Coding Agent Runtime 与智能体工程框架。
21. `book-21-transformer-architecture-evolution/`：第二十一册，Transformer 架构详解、升级变种、替代路线与未来架构演进，重点覆盖 Mamba、SSM、RWKV、RetNet、Hyena、Linear Attention、MoE 和混合架构。
22. `book-22-tool-protocol-ecosystem/`：第二十二册，Function Calling、MCP、A2A、Skill 与工具协议生态，重点覆盖工具 schema、工具注册、MCP server、跨 Agent 协议、插件/Skill 生态、权限安全和协议层系统设计。
23. `book-23-ai-infra/`：第二十三册，AI Infra、大模型基础设施与平台工程，重点覆盖 GPU 集群、网络、存储、调度、训练平台、推理平台、数据与实验平台、可观测性、成本治理、安全治理和系统设计面试。
24. `book-24-llm-inference-engine/`：第二十四册，大模型推理框架与 Serving Engine 实战，重点覆盖从 0 实现推理框架、vLLM、SGLang、PagedAttention、continuous batching、KV Cache 管理、PD 分离和教学项目源码升级。

## 纵向训练系统文件

1. `ROADMAP.md`：3 个月、6 个月、12 个月学习路线。
2. `KNOWLEDGE_GRAPH.md`：核心知识依赖图谱。
3. `PROJECTS.md`：项目路线和简历产出。
4. `PAPERS.md`：论文阅读路线。
5. `INTERVIEW_BANK.md`：集中面试题库。
6. `EXERCISES.md`：练习与阶段验收体系。
7. `GLOSSARY_EN_ZH.md`：中英文术语表。
8. `ENGLISH_INTERVIEW_TEMPLATES.md`：英文面试表达模板。
9. `PROGRESS.md`：学习进度追踪。

## 推荐学习顺序

1. 先学第一册，建立完整主线。
2. 同步查第四册，用作概念词典。
3. 用第十三册补数学短板。
4. 用第十四册补 PyTorch 和工程实现短板。
5. 学第三册，把理论转成代码和项目。
6. 学第五册，补齐训练全流程和训练 debug 能力。
7. 学第六册，补齐部署、推理服务和生产系统能力。
8. 学第七册，建立评估和实验方法能力。
9. 学第九册，深入理解数据工程和数据智能。
10. 学第八册，补齐 safety、alignment 和模型行为控制。
11. 学第十五册、第十六册、第十七册，深入多模态、reasoning 和 Agent 专题。
12. 学第二十一册，系统补齐 Transformer 深度、Mamba/SSM 等后 Transformer 架构和未来架构判断能力。
13. 学第十册，训练论文精读、复现和研究能力。
14. 学第十一册，集中训练系统设计面试。
15. 学第十二册，打磨简历、项目表达和完整面试策略。
16. 用第十八册理解大模型产品化和真实落地。
17. 用第十九册补齐真实工作中的坑、事故、排查路径和资深表达。
18. 用第二十册补齐 harness、coding agent runtime、Claude Code/OpenCode/Codex 架构分析和 evaluation harness 工程能力。
19. 用第二十二册补齐 function calling、MCP、A2A、Skill、插件系统和工具协议生态能力。
20. 用第二十三册补齐 AI Infra、GPU 集群、训练平台、推理平台、可观测性和成本治理能力。
21. 用第二十四册补齐推理框架、serving engine、vLLM/SGLang 架构和 PD 分离能力。
22. 最后回到第二册，补齐研究深度和前沿视野。

## 文件维护原则

1. 每本书先维护完整大纲，再逐章扩写。
2. 每章固定包含目标、直觉、公式、代码、面试题、误区和练习。
3. 所有内容面向面试能力，而不是单纯知识罗列。
4. 后续每次扩写章节时，优先保证可理解、可复述、可实战。
5. 纵向训练系统文件用于组织学习，不替代各书正文。
