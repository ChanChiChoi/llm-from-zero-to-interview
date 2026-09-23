# OpenAI 大模型算法岗系统面试二十四部曲

这个目录下规划二十四本书，并配套学习路线、知识图谱、项目路线、论文路线、面试题库、练习验收、术语表、英文面试模板和进度追踪。

目标是系统辅导算法工程师从大模型初学者成长到能面试 OpenAI 等顶级机构大模型算法岗的候选人。

## 二十四本书定位

当前状态：24 本主书正文第一版已完成，`book-llm-engineer/` 补充篇第一版已完成。项目已进入第二轮全系列精修阶段，重点是公式修正、代码 demo 补充、联网校验、广度审计、深度增强和纵向训练文件同步；其中第二十四册 `book-24-llm-inference-engine/` 当前落盘第 1-60 章已完成第二轮精修与纵向同步。

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
24. `book-24-llm-inference-engine/`：第二十四册，大模型推理框架与 Serving Engine 实战，重点覆盖从 0 实现推理框架、vLLM、SGLang、PagedAttention、continuous batching、KV Cache 管理、PD 分离和教学项目源码升级；当前现有 60 章已完成第二轮公式、demo、资料校准和纵向文件同步。

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

1. 第一版已完成的书籍后续以精修、校验、补充和同步为主，不再按“先大纲后扩写”的第一轮节奏推进。
2. 每章应尽量包含目标、直觉、公式、代码或例子、面试题、误区和练习，但不机械套模板。
3. 所有内容面向面试能力、工程能力和研究判断，而不是单纯知识罗列。
4. 后续每次精修章节时，优先保证可理解、可复述、可实战、可被专家追问。
5. 纵向训练系统文件用于组织学习，并需要随正文第二轮精修同步更新。

## 2026-09 架构专题增补

第二十一册新增 Frontier Model 架构章节：AttnRes、KDA、CSA/HCA、mHC、Mistral Small 4 和 Step 3.5 Flash。它们分别覆盖深度方向残差选择、线性递归状态、长上下文压缩注意力、双随机残差流约束、统一推理模式、EAGLE/NVFP4、MTP-3、滑动窗口注意力和稀疏 MoE，并与第二十四册 serving engine、第五册训练、第二十三册 AI Infra 形成交叉阅读路径。

## 2026-09 新模型专题增补

第六册新增 GPT-6 Astra 长上下文、reasoning effort、工具宿主、阈值计费、异步工具、WebSocket steering、canonical compaction 和技能/`AGENTS.md` 上下文治理章节；第十六册新增 GLM-5.3 长任务环境、验证器、奖励捷径和协议迁移章节；第十七册新增 Kimi K3 发布证据、思考状态和 Harness 评测章节；第二十一册新增 Mistral Small 4、Step 3.5 Flash、DeepSeek V4.1-Flash、K2 Horizon、Qwen3.8、GLM-5.3-Flash 与 DeepSeek V4 Flash Vision 架构/多模态专题。相关章节均以官方模型页、发布文档、模型卡和论文为事实边界，并将参数规模、训练架构、完整配置、alias 路由与未公开缩写保留为待核验或带日期字段。

Anthropic 官方模型目录核验新增 Claude Opus 5：第四册记录 1M context、adaptive thinking、平台和宿主边界；第六册、第七册、第十七册和第二十册分别承接长上下文成本、固定 effort 评测、Agent 工具和状态审计。参数规模、内部架构和训练方法不从模型目录字段反推。

Anthropic 专属模型页进一步核验 Claude Fable 5.1：第四册记录 adaptive always-on、preserved thinking、beta 状态协议和长任务产品定位；第六册、第七册、第十七册和第二十册承接上下文成本、公平评测、工具进度与状态恢复。Fable 5.1 的参数规模和训练/推理内部机制仍待核验。

Anthropic 官方模型目录继续核验 Claude Sonnet 5：第四册记录 `claude-sonnet-5`、1M context、128K/300K 输出边界、Adaptive thinking、默认 high effort、Fast latency 字段、平台和成本；第六册与第二十四册承接缓存、延迟、并发和成本账本，第七册承接与 Opus/Fable 的固定条件对照。参数规模、内部架构、训练方法和独立 benchmark 复现仍待核验。

同一官方目录快照补充 Claude Haiku 4.5：第四册记录 `claude-haiku-4-5-20251001`、200K context、64K 输出、extended thinking、`fastest` 延迟字段、平台和低价位；第六册/第二十四册承接延迟、缓存与单位成功成本，第七册承接与 Sonnet/Opus/Fable 的固定 harness 对照。参数规模、内部架构、训练方法和独立 benchmark 复现仍待核验。

DeepSWE v1.1 快照作为评测方法专题接入第七册、第十七册和第二十册：第七册拆解 Pass@1、区间、成本和 Agent steps 的统计口径，第十七册审计工具回执、权限、上下文与恢复，第二十册固定 `mini-swe-agent`、verifier、重试和环境版本。排行榜行记录的是完整 Agent 系统组合，不能把分数直接归因于基础模型；原始快照与待核验字段见 [`research/model-update-2026-09/deepswe-snapshot-notes.md`](research/model-update-2026-09/deepswe-snapshot-notes.md)。

DeepSeek V4.1-Flash 接入第二十一册第 81 章：先用 CED 分开 prefill/decode，再学习 CSA2、层次化索引、FP4 global KV 和 SWA Bounded Replay，随后连接 MoE、Engram、mHC、DSpark、多模态 prompt 与协议迁移。第五册承接 45T 多模态预训练和 `SFT -> RL -> OPD`，第六册承接 HBM/SSD/replay/cache 账本，第七册、第十七册和第二十册承接 Agent harness、verifier、工具权限和恢复评估。模型卡与技术报告的自报数字必须和独立实测分栏；固定 HF 仓库已公开 reference `model.py`/TileLang `kernel.py`，但完整 production kernel、线上接受率和独立 profiling 仍待核验。
本轮再补入 vLLM `v0.30.0` stable release/source evidence：第二十一册第 81 章新增 stable registry、NVIDIA/ROCm V4.1 package、PyPI artifact 与 release-note 面试点，并明确 `release surface -> full-weight load -> target profile -> acceptance/SLO` 的证据阶梯。v0.30.0 的 registry 或 wheel 不能替代完整权重、DSpark verify/rollback、FP4 质量、EPD 和生产验收。

K2 Horizon MoVA 36B/A4B 接入第二十一册第 82 章：先学习 36B total/4B active proxy 与 dense/sparse layer 排布，再学习 MoVA value routing、FFN MoE、GQA KV cache、512K 分阶段训练和 TP/EP serving。第五册承接 0.9B 卡片披露的 MOPD 边界，第六册承接双重 dispatch、KV、workspace 和通信账本；第二十一册同章介绍 K2 7B Uno 的 LoRA diffusion draft 与 `Psi-Spec` rejection verification。Uno 是官方关联 adapter/论文技术，不是排行榜新增模型；完整训练 recipe、接受率和生产 profiling 仍待核验。

Qwen3.8 接入第二十一册第 83 章：先从 27B/A95B 的 GDN + Gated Attention 层布局进入，再学习 Flash-Next 的 QSA micro-block indexer、两阶段稀疏训练、四分支 Gated Residual、N-gram host-memory prefetch 和 Muon/AdamW 参数分工。第四册承接术语与模型家族边界，第五册承接优化器/训练消融，第六册承接长上下文 cache、带宽和参数账本，第七册承接报告自报数字与独立评测，第十六册/第十七册承接 thinking protocol、工具和长任务 harness。Qwen3.8-Max 只作为基于 A95B 的 hosted version 记录；完整 kernel、线上接受率、目标硬件 profiling 和独立 benchmark 仍待核验。

Kimi K2.6 接入第二十一册第 90 章：先学习 1T/32B MoE、MLA/YaRN、MoonViT 和 native INT4 的模型—serving 账本，再用第十七册理解 300 sub-agents/4,000 coordinated steps 的 Agent Swarm 任务图、权限和 artifact gate，用第二十四册理解 `preserve_thinking`、工具解析、KV/cache 和 Kimi Vendor Verifier 的部署验收。K2.5 架构复用只表示公开路线复用，不把 K3 的 KDA/AttnRes 或 K2.7 Code 的专属协议回写给 K2.6；DataCurve 没有精确 K2.6 行，不迁移其他 Kimi 版本的 Agent 结果。

Qwen3-Omni 接入第二十一册第 91 章：先理解 Thinker/Talker 的职责拆分，再学习 AuT 的 12.5 Hz 音频时间粒度、Qwen3-VL/SigLIP2-So400m 视觉 encoder、TM-RoPE 的 temporal/height/width 对齐、Talker 首码本 AR + residual-codebook MTP 和 Code2Wav。随后用第十五册补齐跨媒体输入与语音输出，用第十六册和第十七册审计 Thinker 后训练、RAG/function calling/safety/verifier 介入和多码本状态，用第六册/第二十四册复算异步 chunked prefill、音频 packet、取消恢复、显存和首包延迟。Instruct、Thinking、Captioner 是同家族 artifact；DataCurve 没有精确 Qwen3-Omni 行，不迁移其他 Qwen 的 Agent 分数。报告的 `234/547 ms` 首包数字和 README 的 vLLM Thinker 支持均按来源/实现边界记录，不写成生产 SLO。

K2 Horizon 3.7B 接入第二十一册第 82 章的 dense 对照小节：先固定 `K2HorizonForCausalLM`、36 层、32Q/8KV GQA、524K position 和 `num_experts=0`，再与 36B/A4B 的 MoVA value top-4、FFN top-8、shared expert 和 TP/EP dispatch 对照。第五册承接 22.9T pretraining、32K/128K/512K 分阶段训练、Math/Code/STEM-Code RL 分支与 ISO/RAM merge，第六册承接 dense GEMM、KV、parser/revision、migration 和 H200 serving 账本，第七册区分 AA、发布方 recipe benchmark 与本机实测。旧 `APPENDIX.md` 的 Xllm/FP32 字段标为 revision 冲突；K2 3.7B 是 AA 单榜资料级闭环，DataCurve 无精确 Agent 行。

Qwen3-VL-235B-A22B 接入第二十一册第 92 章：先学习 SigLIP2 vision encoder、两层 MLP merger、Qwen3 MoE decoder 的三模块数据流，再理解 Interleaved-MRoPE 如何交错 temporal/height/width 频率、DeepStack 如何把 `[8,16,24]` 中间视觉特征作为 residual 注入早期语言层，以及 Video Timestamp 如何把秒数写进视频 patch 上下文。用第五册理解 S0-S3 的 67B/1T/1T/100B curriculum、square-root normalized loss、SAPO 和 General RL；用第十五册补视觉 token、视频时间轴和多模态预算；用第十六/十七/二十册审计 Thinking with Images、tool-call reward、GUI executor、权限、回放和 verifier；用第六册/第二十四册拆开 MoE dispatch、visual prefill、KV/cache、TP/EP、目标硬件和端到端 serving gate。Instruct、Thinking、Reasoning 是同一基础模型的配置或 artifact，DataCurve 无精确 Qwen3-VL 行，不迁移其他 Qwen Agent 分数。

Qwen3.7 Plus 暂不新增第二十一册 Transformer 架构章节：Artificial Analysis 有精确榜单条目，但 DataCurve 没有精确 Agent 行，Alibaba Cloud 公开的是托管模型合同而非参数/架构/训练报告。第十五册补充图像/视频和 GUI evidence budget；第十七册补充 action proposal、权限、执行器、观察回灌、幂等和 verifier；第二十册补充 region/scope capability manifest、alias/snapshot 和状态回放；第二十四册补充 prefix/cache、长输入分档、视觉 prefill 和 tool serving gate。完整证据见 [`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)。

Claude Opus 5.5 接入既有 Anthropic/Agent 路线：第十六册承接 always-on adaptive thinking、effort、thinking block binding、token efficiency 与质量-成本曲线，第七册区分 AA `max with fallback`、Anthropic 发布方 benchmark 与 DataCurve 精确行缺失，第八册承接 Cyber/Life Sciences/Distillation safeguard 与透明 fallback，第十七册和第二十册承接长任务 coding、工具/computer-use migration、compaction、inline tools、权限、实际执行模型和 artifact verifier，第二十四册承接 cache、fast mode、工具轮次、成本、`usage.speed` 和 serving trace。由于参数、架构、训练 recipe 和独立 DataCurve Agent 行未确认，不新增第二十一册 Transformer 架构章节。完整证据见 [`claude-opus-5.5-source-notes.md`](research/model-update-2026-09/claude-opus-5.5-source-notes.md)。

OpenAI GPT-6 Sol 接入既有 GPT-6/Agent 路线：第六册补充 1.05M context、922K maximum input、128K output、272K whole-request pricing threshold 和 serving cost ledger；第十六册承接 `standard/pro` mode、`none`--`max` effort、reasoning token、`configuration_update` 和 incomplete budget；第十七册承接复杂 coding、Responses function calling、tool search、permission、executor 和 verifier；第二十册承接 Agents API/SDK/Responses 的状态所有权与 opaque compaction replay；第二十四册承接 cache、tool schema、compaction、TTFT/TPOT、Batch/Flex、Fast mode 和单位成功成本。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行，不迁移其他 GPT 的 Agent 结果；参数、架构、完整训练 recipe 和生产 acceptance 未确认，不新增第二十一册 Transformer 架构章节。完整证据见 [`gpt-6-sol-source-notes.md`](research/model-update-2026-09/gpt-6-sol-source-notes.md)。

OpenAI GPT-6 Luna 接入同一 GPT-6/Agent 路线，但保留独立模型合同：第六册记录 focused/high-volume 定位、1.05M/922K/128K、2026-05-18 knowledge cutoff、272K whole-request threshold、`$0.10/$0.50` token 价格和 serving cost ledger；第十六册承接 `standard/pro` mode、`none`--`max` effort、reasoning token、`configuration_update` 和 incomplete budget；第十七册承接 Responses function calling、工具 capability surface、permission、executor 和 verifier；第二十册承接 Agents API/SDK/Responses 的所有权与 opaque compaction replay；第二十四册承接 cache、tool schema、TTFT/TPOT、Batch/Flex、Fast mode 和单位成功成本。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_luna_*` 行，不迁移 Sol、Astra、GPT-5.6 或其他 GPT 的 Agent 结果；参数、架构、完整训练 recipe 和生产 acceptance 未确认，不新增第二十一册 Transformer 架构章节。完整证据见 [`gpt-6-luna-source-notes.md`](research/model-update-2026-09/gpt-6-luna-source-notes.md)。
