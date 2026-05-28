# 项目背景与写作规范

## 项目目标

我是一个算法工程师小白，希望系统准备 OpenAI 当前大模型算法岗面试。

本项目的目标是写出一套系统教材和训练系统，通过二十四本书和配套训练文件帮助读者从大模型初学者逐步成长为具备大模型算法岗面试竞争力的候选人。

## 书系宗旨

本书系不是简单罗列知识点，而是要帮助读者在脑中建立一张可以迁移和扩展的“大模型知识图谱”。

每个重要知识点都应尽量讲清：

1. 它是在什么历史背景下提出的。
2. 它之前的方法是什么，前人方法遇到了什么问题。
3. 它解决了哪些关键问题。
4. 它的核心直觉、机制和特点是什么。
5. 它有什么优点、缺点和适用边界。
6. 后来者如何继承、替代或改进它。
7. 未来可能还会沿着哪些方向继续演化。
8. 它和同一知识谱系中的相邻概念是什么关系。

本项目虽然默认从“小白友好”出发，但最终目标是面向全阶段读者：初学者能看懂来龙去脉，中级读者能建立系统框架，高阶读者能看到机制、trade-off、前沿争议和面试深挖点。

因此，后续整套书要持续朝这个方向完善：用历史脉络串联概念，用问题演化解释方法，用优缺点和后来者构建方法谱系，而不是写成孤立术语列表。

这套书不只是知识点汇总，而是面向真实面试能力建设：

1. 能理解核心概念。
2. 能讲清楚技术动机。
3. 能写出关键代码。
4. 能分析方法优缺点。
5. 能讨论适用场景和改进方向。
6. 能回答基础题、算法题、工程题、开放研究题和项目深挖题。

## 目标读者

本项目默认读者是算法工程师小白，可能具备一定 Python、机器学习或深度学习基础，但对当前大模型体系不完整。

写作时要遵循以下原则：

1. 不默认读者已经理解复杂背景。
2. 不直接堆公式和术语。
3. 先讲直觉，再讲公式，再讲代码，再讲面试。
4. 对每个重要概念都要解释它为什么出现、解决了什么问题。
5. 对复杂技术要尽量配合例子、图示思路或代码帮助理解。
6. 对优化器、DPO/RLHF/RLAIF、reasoning、Agent、推理引擎、主流大模型架构创新等快速变化主题，写作前要优先联网核验最新论文、官方技术报告、项目文档和高可信技术博客，避免只依赖已有记忆或旧资料。
7. 对新近模型、闭源模型和社区传闻，要区分“官方披露”“论文复现”“社区推测”和“待核验”，不能把未确认信息写成确定事实。
8. 在有网络条件时，整本书大纲设计、章节扩写和关键知识点补全可以主动联网搜索新增资料，以提升内容覆盖面、权威性和时效性。但不要求每个细节都联网，基础概念、稳定知识和已充分掌握的内容可以直接写；联网主要用于新近进展、版本变化、争议点、具体模型或工程实现细节。资料优先级应为官方论文、官方技术报告、官方 GitHub 仓库、官方文档、主流框架文档、知名研究者或工程团队博客、可信社区讨论；低可信来源只能作为线索，不能直接当作结论。
9. 使用网络资料时必须考虑可信度和投毒风险。对来源不明、营销化、标题党、无法复现、与官方资料冲突或只有单一社区帖子支持的信息，应降级为“待核验”或不写入正文。重要结论应尽量交叉验证，优先以官方材料、论文、源码和可复现实验为准。

## 二十四本书定位

### 第一册：OpenAI 大模型算法岗面试核心 38 讲

目录：`book-01-core-30/`

定位：核心主线、面试最小闭环、第一轮系统学习。

目标：帮助读者快速建立大模型算法岗的完整知识框架，覆盖语言模型、Transformer、预训练、对齐、推理优化、评估安全、多模态基础和开放题。

适合阶段：刚开始系统准备时优先阅读。

### 第二册：OpenAI 大模型算法岗进阶 120 讲

目录：`book-02-advanced-100/`

定位：深入原理、前沿研究、复杂系统、论文精读。

目标：帮助读者把回答深度提升到高级算法工程师或研究员面试水平，能够讨论架构设计、scaling law、训练稳定性、对齐算法、推理优化、Reasoning Model、Agent、多模态大模型和开放研究问题。

适合阶段：学完第一册后用于进阶拔高。

### 第三册：大模型算法岗实战手册

目录：`book-03-practical-handbook/`

定位：代码实战、项目实战、实验实战、面试实战。

目标：通过可运行代码和可展示项目，把理论知识转成实际能力。重点训练 PyTorch、Transformer 组件、小 GPT、SFT、LoRA、DPO、推理优化、RAG、Agent、多模态模型、评估和 debug。

适合阶段：学习第一册过程中可同步实践，也可在第一册后集中训练。

### 第四册：大模型百科全书

目录：`book-04-llm-encyclopedia/`

定位：概念索引、术语解释、方法对比、公式速查、面试词典。

目标：建立一个持续扩展的大模型知识库，供学习和复习时查阅。覆盖语言、多模态、语音、图像、视频、Agent、推理优化、安全评估等方向。每个条目要简洁准确，并能连接到相关概念和面试问题。

适合阶段：全程作为工具书使用。

### 第五册：大模型训练全流程

目录：`book-05-llm-training/`

定位：训练数据、tokenizer、预训练、后训练、分布式训练、训练稳定性、评估、实验管理和训练成本。

目标：帮助读者系统掌握大模型训练全流程，能够回答如何设计训练方案、如何监控训练过程、如何 debug 训练异常、如何做资源和成本规划。

适合阶段：第一册建立主线后阅读，也可以和第三册实战同步学习。

### 第六册：大模型部署与推理工程

目录：`book-06-llm-deployment/`

定位：推理基础、部署架构、KV Cache、解码策略、推理引擎、批处理调度、量化压缩、多 GPU 服务、RAG/Agent 部署、多模态部署、监控安全和成本优化。

目标：帮助读者系统掌握大模型从 checkpoint 到线上服务的完整部署链路，能够回答低延迟、高吞吐、低成本和高可靠性的生产系统设计问题。

适合阶段：掌握第一册推理优化基础后阅读。

### 第七册：大模型评估、实验与科学方法

目录：`book-07-evaluation-experiments/`

定位：benchmark、human eval、pairwise eval、LLM-as-a-judge、污染检测、reasoning/code/math eval、多模态评估、safety eval、在线实验、统计显著性、error analysis。

目标：帮助读者回答“如何知道模型真的更强”，建立科学实验和可信评估能力。

适合阶段：学完第一册核心能力后阅读，也可和训练、部署两册同步学习。

### 第八册：AI Safety、Alignment 与模型行为

目录：`book-08-ai-safety-alignment/`

定位：AI safety、alignment problem、scalable oversight、reward hacking、jailbreak、prompt injection、red teaming、interpretability、steering、model editing、unlearning、privacy、governance。

目标：帮助读者理解大模型能力提升之外的安全、可靠、可控问题。

适合阶段：学习后训练和对齐后阅读。

### 第九册：大模型数据工程与数据智能

目录：`book-09-data-engineering/`

定位：web-scale 数据采集、清洗过滤、去重、污染检测、数据配比、代码数学领域数据、合成数据、偏好数据、多模态数据、数据价值评估和数据治理。

目标：帮助读者理解数据如何决定模型能力，并能设计高质量数据体系。

适合阶段：学习预训练和训练全流程时同步阅读。

### 第十册：大模型论文精读、复现与研究方法

目录：`book-10-paper-reproduction-research/`

定位：论文阅读、贡献判断、实验复现、ablation、controlled experiment、负结果、研究假设和核心论文线。

目标：帮助读者具备研究讨论、论文复现和技术判断能力。

适合阶段：完成核心知识学习后用于进阶拔高。

### 第十一册：大模型系统设计面试

目录：`book-11-system-design-interview/`

定位：ChatGPT 服务、训练平台、推理平台、RAG、Agent、多模态助手、实时语音助手、视频生成服务、评估平台、数据标注平台、安全审核系统、模型路由和缓存系统。

目标：帮助读者系统准备大模型系统设计面试。

适合阶段：训练、部署、RAG、Agent 和多模态基础掌握后阅读。

### 第十二册：OpenAI 大模型算法岗求职与面试作战手册

目录：`book-12-career-interview-playbook/`

定位：岗位画像、简历策略、项目包装、coding interview、ML/LLM 基础面试、training 面试、deployment 面试、alignment/safety 面试、multimodal/research 面试、system design 面试、behavioral interview、mock interview 和冲刺计划。

目标：把知识、项目、表达和策略转化为真实面试竞争力。

适合阶段：全程参考，面试前重点使用。

### 第十三册：大模型数学基础

目录：`book-13-math-foundations/`

定位：线性代数、概率论、信息论、优化、低秩、统计学习、贝叶斯、强化学习数学和评估统计。

目标：补齐大模型算法岗需要的数学底层能力。

### 第十四册：PyTorch 与深度学习工程

目录：`book-14-pytorch-deep-learning-engineering/`

定位：PyTorch 张量、autograd、Module、DataLoader、训练循环、混合精度、分布式训练、debug、profiling 和 Transformer 组件实现。

目标：补齐工程实现能力，让读者能写、能跑、能调试深度学习代码。

### 第十五册：多模态与生成模型专题

目录：`book-15-multimodal-generative-models/`

定位：CLIP、VLM、vision encoder、diffusion、Stable Diffusion、DALL·E、视频生成、语音、统一多模态模型和多模态安全。

目标：系统掌握当前多模态和生成模型方向。

### 第十六册：Reasoning Model 专题

目录：`book-16-reasoning-models/`

定位：CoT、self-consistency、verifier、process supervision、search、test-time compute、数学推理、代码推理和 reasoning eval。

目标：系统理解 reasoning model 和推理时计算扩展。

### 第十七册：Agent 与工具调用专题

目录：`book-17-agent-tool-use/`

定位：tool use、function calling、ReAct、planning、memory、agentic RAG、code agent、computer use、multi-agent、Agent 评估和安全。

目标：系统理解 Agent 架构、工具调用和可靠性安全问题。

### 第十八册：大模型产品化、商业化与落地

目录：`book-18-product-business-commercialization/`

定位：产品化、用户场景、产品体验、ROI、企业应用、RAG/Agent/多模态落地、隐私合规、上线运营和反馈闭环。

目标：帮助算法工程师理解大模型如何从技术变成真实产品和业务价值。

### 第十九册：资深大模型工程师实战宝典

目录：`book-19-practitioner-playbook/`

定位：真实工作中的坑、事故、排查路径、经验法则、工程 trade-off、项目复盘和资深面试表达。

目标：帮助读者站在多年经验从业者视角理解大模型项目，而不是只掌握书本知识。

### 第二十册：Agent Harness、Coding Agent Runtime 与智能体工程框架

目录：`book-20-agent-harness-runtime/`

定位：Harness、Agent Runtime、Coding Agent 工作流、工具系统、文件编辑、终端执行、安全沙箱、trace/replay、evaluation harness、Claude Code、OpenCode、Codex 和 MCP/A2A 集成。

目标：帮助读者理解 coding agent 和智能体工程系统的底层运行框架，而不是只理解模型调用。

### 第二十一册：Transformer 架构详解、升级变种与未来架构演进

目录：`book-21-transformer-architecture-evolution/`

定位：系统讲解 Transformer 原始结构、现代 LLM 架构改造、Linear Attention、SSM/S4/Mamba、RWKV、RetNet、Hyena、MoE、混合架构和未来架构路线。

目标：帮助读者补齐 Transformer 深度、后 Transformer 架构和未来模型架构判断能力。

### 第二十二册：Function Calling、MCP、A2A、Skill 与工具协议生态

目录：`book-22-tool-protocol-ecosystem/`

定位：Function Calling、Tool Schema、MCP、A2A、Skill、Plugin、工具注册、工具协议、跨 Agent 通信、权限安全和企业工具治理。

目标：帮助读者理解大模型工具协议生态，能够设计可靠、安全、可评估、可治理的工具接入和跨 Agent 协作系统。

### 第二十三册：AI Infra、大模型基础设施与平台工程

目录：`book-23-ai-infra/`

定位：GPU 集群、网络、存储、调度、训练平台、推理平台、数据与实验平台、模型仓库、可观测性、可靠性、成本治理、安全治理和 AI Infra 系统设计。

目标：帮助读者理解支撑大模型训练、推理、评估、数据、Agent 和多模态系统运行的基础设施底座，能够从算法视角分析平台瓶颈、设计可靠系统并回答 AI Infra 面试问题。

### 第二十四册：大模型推理框架与 Serving Engine 实战

目录：`book-24-llm-inference-engine/`

定位：LLM Serving Engine、从 0 实现推理框架、vLLM、SGLang、PagedAttention、continuous batching、KV Cache 管理、scheduler、prefix/radix cache、speculative decoding、PD 分离、disaggregated serving 和教学项目源码升级。

目标：帮助读者理解推理框架内部架构，能够从 naive generate 升级到可调度、可批处理、可流式输出、可管理 KV Cache、可支持 PD 分离的 serving engine，并能回答 vLLM/SGLang 架构和推理系统设计面试题。

## 总体内容要求

二十四本书整体需要满足以下要求：

1. 覆盖大模型算法岗相关核心知识点，包括语言、多模态、语音、图像、视频、Agent、推理优化和安全评估。
2. 列出每个主题的核心议题。
3. 用通俗易懂的方式介绍概念。
4. 说明每个技术点解决了什么问题，也就是为什么提出这个技术点。
5. 说明该技术点的特点、优点、缺点、适用场景和可能改进方向。
6. 对稍微复杂的知识点提供 Python 代码。
7. 代码最好是自组织的，尽量能低依赖或 0 依赖运行。
8. 文件和目录组织结构要合理，但不因篇幅限制压缩解释；章节长度应服务于读者理解，必要时通过拆分文件改善阅读体验。
9. 内容要服务于面试表达，不只服务于阅读理解。
10. 每个重点知识点都要尽量能转化成面试问题和标准回答。

具体写作顺序、章节模板、同步维护策略和质量自查标准见 `WRITING_PLAN.md`。后续扩写任何章节前，应先检查该文件，避免偏离项目目标。

## 每章推荐结构

每一章或每一讲建议采用以下结构：

1. 本章目标
2. 问题背景
3. 这个技术解决了什么问题
4. 直觉解释
5. 数学定义或核心公式
6. 最小代码实现
7. 方法特点
8. 优点和缺点
9. 适用场景
10. 常见改进方向
11. 面试官会怎么问
12. 标准回答模板
13. 常见误区
14. 小练习
15. 本章总结

不是每一章都必须机械包含全部小节，但重要技术点至少要覆盖：背景、动机、直觉、公式或机制、代码或例子、优缺点、适用场景、面试问题。

## 百科条目推荐结构

第四册中的百科条目建议采用以下结构：

1. 一句话定义
2. 核心直觉
3. 为什么需要它
4. 关键公式或结构
5. 典型使用场景
6. 优点和局限
7. 常见误区
8. 面试追问
9. 相关概念

百科条目应简洁准确，适合快速查阅；如果某个概念需要长篇展开，应链接或指向第一册、第二册或第三册中的对应章节。

## 代码写作规范

代码是本项目的重要组成部分。复杂概念要尽量通过代码帮助理解。

### 代码依赖原则

1. 基础数学、概率、采样、tokenization 示例优先使用纯 Python。
2. 深度学习模型实现优先使用 PyTorch。
3. 微调、对齐、推理服务和多模态实战可以使用 Hugging Face Transformers、Datasets、Accelerate、PEFT、TRL、vLLM、Pillow、torchaudio 等工具。
4. 如果代码不是 0 依赖，需要在章节中明确说明依赖和运行方式。

### 代码内容原则

1. 代码要尽量短小、自包含、可运行。
2. 不写只有形式、不能帮助理解的伪代码。
3. 关键张量要标注 shape。
4. 关键步骤要有简洁注释。
5. 面试高频代码要优先手写实现，例如 attention、causal mask、top-k、top-p、KV cache、LoRA、DPO loss。
6. 工程实战代码要配合实验现象和结果分析，而不是只给训练脚本。

## 文件组织规范

当前项目采用“一本书一个文件夹”的组织方式：

1. `book-01-core-30/`
2. `book-02-advanced-100/`
3. `book-03-practical-handbook/`
4. `book-04-llm-encyclopedia/`
5. `book-05-llm-training/`
6. `book-06-llm-deployment/`
7. `book-07-evaluation-experiments/`
8. `book-08-ai-safety-alignment/`
9. `book-09-data-engineering/`
10. `book-10-paper-reproduction-research/`
11. `book-11-system-design-interview/`
12. `book-12-career-interview-playbook/`
13. `book-13-math-foundations/`
14. `book-14-pytorch-deep-learning-engineering/`
15. `book-15-multimodal-generative-models/`
16. `book-16-reasoning-models/`
17. `book-17-agent-tool-use/`
18. `book-18-product-business-commercialization/`
19. `book-19-practitioner-playbook/`
20. `book-20-agent-harness-runtime/`
21. `book-21-transformer-architecture-evolution/`
22. `book-22-tool-protocol-ecosystem/`
23. `book-23-ai-infra/`
24. `book-24-llm-inference-engine/`

每本书目录下必须包含：

1. `目录.md`：记录全书目录、阅读方式和章节文件索引。
2. `chapters/`：存放分章或分部分 Markdown 文件。

文件拆分原则：

1. 不设置单个 Markdown 文件的固定篇幅上限；以是否讲清楚、是否便于读者消化为第一标准。
2. 一个章节可以拆成一个或多个 Markdown 文件，拆分应按主题自然发生，而不是为了压缩正文。
3. 如果某章代码较多，可以拆成单独的代码讲解文件，但正文不能省略机制、例子和 debug 说明。
4. 文件名要带序号，便于排序和阅读。
5. 后续扩写时优先在对应书籍目录下新增或修改文件，不要把内容都堆到根目录。

## 推荐学习顺序

建议学习顺序如下：

1. 先学习第一册，建立大模型算法岗核心主线。
2. 学习过程中随时查第四册，把不懂的术语补齐。
3. 第一册学到 Transformer 后，同步开始第三册的代码实战。
4. 第一册学完后，集中完成第三册中的项目实战。
5. 学习第五册，补齐训练全流程、训练系统和训练 debug 能力。
6. 学习第六册，补齐部署、推理服务、生产系统和成本优化能力。
7. 学习第七册，建立评估、实验设计和 error analysis 能力。
8. 学习第九册，深入理解数据工程、数据质量和数据治理。
9. 学习第八册，补齐 AI safety、alignment 和模型行为控制能力。
10. 学习第十册，训练论文精读、复现和研究讨论能力。
11. 学习第十一册，集中训练大模型系统设计面试。
12. 学习第十二册，打磨简历、项目表达、模拟面试和求职策略。
13. 最后学习第二册，用于进阶拔高、论文精读和开放研究题训练。
14. 根据短板穿插学习第十三册数学基础和第十四册 PyTorch 工程。
15. 根据岗位方向深入第十五册多模态、第十六册 reasoning、第十七册 Agent。
16. 用第十八册理解产品化、商业化和真实落地约束。
17. 用第十九册补齐真实工作中的坑、事故、排查路径和资深表达。
18. 用第二十册补齐 harness、coding agent runtime、evaluation harness、Claude Code/OpenCode/Codex 架构分析和 MCP/A2A 集成能力。
19. 用第二十一册系统补齐 Transformer 变体、Mamba/SSM、RWKV、RetNet、Hyena、Linear Attention 和混合架构判断能力。
20. 用第二十二册系统补齐 Function Calling、MCP、A2A、Skill、插件协议和工具生态能力。
21. 用第二十三册系统补齐 AI Infra、GPU 集群、训练平台、推理平台、可观测性和成本治理能力。
22. 用第二十四册系统补齐推理框架、vLLM/SGLang 架构、KV Cache 管理、continuous batching 和 PD 分离能力。

## 面试能力目标

最终希望读者具备以下能力：

1. 能解释大语言模型到底在学什么。
2. 能解释 next-token prediction 为什么有效。
3. 能从零讲清楚 Transformer block。
4. 能手写 self-attention、causal mask 和 sampling 代码。
5. 能解释 LLM 预训练、SFT、RLHF、DPO 的流程和差异。
6. 能分析数据质量、scaling law、训练稳定性和分布式训练问题。
7. 能解释 KV Cache、FlashAttention、量化和 speculative decoding。
8. 能讨论幻觉、评估、安全、reward hacking 和 benchmark contamination。
9. 能完成 miniGPT、LoRA 微调、DPO、RAG 或推理优化等实战项目。
10. 能回答 OpenAI 风格开放题，提出假设、实验、风险和 trade-off。
11. 能把个人项目讲成一个完整的算法问题、工程问题和实验问题。
12. 能系统设计大模型训练流程并排查训练异常。
13. 能系统设计大模型部署服务并优化延迟、吞吐、显存和成本。
14. 能设计可信 benchmark，并判断评估结果是否可靠。
15. 能讨论 AI safety、alignment、jailbreak、red teaming 和模型行为风险。
16. 能设计大模型数据 pipeline，并分析数据质量和污染问题。
17. 能精读论文、复现实验、设计 ablation 并形成研究判断。
18. 能完成大模型系统设计面试。
19. 能写出有竞争力的简历，并把项目讲成高质量面试故事。
20. 能补齐数学、PyTorch、工程实现、多模态、reasoning、Agent 和产品落地等横向专题能力。
21. 能从资深工程师视角分析真实项目问题，讲清楚事故、排查、复盘和 trade-off。
22. 能设计 agent harness 和 coding agent runtime，并分析 Claude Code、OpenCode、Codex 等系统的工程架构。
23. 能设计 AI Infra 平台，分析 GPU 集群、训练平台、推理平台、可观测性、可靠性和成本治理问题。
24. 能设计 LLM serving engine，分析 vLLM/SGLang、PagedAttention、continuous batching、KV Cache 管理、prefix cache 和 PD 分离问题。

## 写作风格

本项目的写作风格应满足：

1. 先通俗，后严谨。
2. 先动机，后方法。
3. 先直觉，后公式。
4. 先最小例子，后真实系统。
5. 不回避复杂性，但要逐层拆解。
6. 不只告诉读者结论，也要告诉读者为什么。
7. 面试相关内容要明确标注，方便复习。

## 后续维护原则

后续扩写二十四本书时，应遵循以下原则：

1. 第一册和第二册正文第一版已完成，后续以校对、补图、补公式和统一风格为主。
2. 当前优先继续扩写第三册实战手册，完成第十一部分“多模态实战”。
3. 对第一册、第二册和第三册中的重要概念，同步在第四册补充百科条目。
4. 对涉及代码的章节，同步在第三册或专题书中补充实战版本。
5. 每次新增内容都要检查是否符合本文件中的写作规范。
6. 每次扩写正式章节后，要按 `WRITING_PLAN.md` 的质量自查标准检查是否合格。
7. 每次扩写核心章节后，要同步维护百科、题库、练习、术语和项目路线中相关内容。
8. 每次扩写高时效主题前，要先查最新公开资料；正文中涉及具体模型、算法版本、系统特性或工程实践时，应标注来源边界和更新时间语境。
9. 每次设计新书大纲、专题目录或关键章节时，如果网络可用，应主动搜索权威资料补充遗漏知识点，避免目录和正文只反映旧知识结构；但应控制搜索粒度，优先查会影响知识结构和结论准确性的资料，而不是机械地逐句检索。
10. 对网络资料要做可信度筛选，不能全信。若资料来源可疑、证据不足或存在冲突，应在正文中谨慎表达，必要时标注“待核验”“社区说法”或直接舍弃。
11. 每完成一本书的正文编写，必须同步检查并更新当前目录下所有根目录总控文件，包括 `README.md`、`BOOK_SERIES.md`、`ROADMAP.md`、`PROJECTS.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROGRESS.md` 等，避免书籍完成度、学习路线、题库、练习、术语和后续计划互相脱节。
