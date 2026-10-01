# 学习路线图

## 定位

本文件用于把二十四本书组织成可执行学习路径，并根据当前正文进度动态调整。

## 当前可优先阅读内容

1. 第一册 `book-01-core-30/`：正文第一版已完成，共 38 讲，适合建立核心主线。
2. 第二册 `book-02-advanced-100/`：正文第一版已完成，共 120 讲，适合进阶拔高和开放研究题训练。
3. 第三册 `book-03-practical-handbook/`：正文第一版已完成，共 72 讲，适合把理论转成代码、实验和项目。
4. 第四册到第十九册：正文第一版已完成，适合系统补齐百科索引、训练、部署、评估、安全、数据、论文、系统设计、求职、数学、PyTorch、多模态、Reasoning、Agent、产品落地和实战坑。
5. 第二十册到第二十四册：正文第一版已完成，适合补齐 Agent Harness、Transformer 架构演进、工具协议生态、AI Infra 和推理框架专题；其中第二十四册现有 60 章已完成第二轮精修，可优先作为推理框架专题阅读版本。
6. `book-llm-engineer/`：补充篇第一版已完成，适合作为大模型工程师面试和职业表达的延伸材料。

## 3 个月冲刺路线

1. 第 1-2 周：学习第一册第 1-18 讲，完成语言模型、Transformer 和 miniGPT 主线。
2. 第 3-4 周：学习第三册第 1-18 讲，完成 PyTorch、Attention、Transformer Block 和小 GPT 实战。
3. 第 5-6 周：学习第一册训练、对齐、推理部分，并配合第三册 SFT、LoRA、DPO 实战。
4. 第 7-8 周：学习第三册推理优化、RAG/Agent 和评估 Debug 实战。
5. 第 9-10 周：学习第三册项目作品集和面试实战训练，整理 miniGPT、SFT、DPO、RAG、推理服务项目讲稿。
6. 第 11-12 周：学习第三册多模态实战、第二册开放题和论文讨论，完成 mock interview 和错题复盘。

## 6 个月系统路线

1. 第 1 月：第一册 + 第十三册数学基础 + 第十四册 PyTorch 工程。
2. 第 2 月：第三册第 1-18 讲，完成 Transformer 组件和 miniGPT。
3. 第 3 月：第三册 SFT、LoRA、DPO、推理优化和 RAG/Agent 实战。
4. 第 4 月：第二册架构、训练、对齐、部署和长上下文部分。
5. 第 5 月：第三册项目作品集、面试实战、多模态实战，以及第十五册、第十六册、第十七册相关专题。
6. 第 6 月：系统设计、论文讨论、英文面试、简历项目打磨和完整 mock interview。

## 12 个月研究路线

1. 前 3 个月：完成第一册、第三册核心实战和至少 3 个可展示项目。
2. 第 4-6 个月：完成第二册进阶内容，重点突破训练、对齐、推理、评估、RAG、Agent 和 reasoning。
3. 第 7-8 个月：系统学习第五册到第九册，补齐训练、部署、评估、安全和数据工程专题。
4. 第 9-10 个月：学习第十册、第十一册、第二十一册、第二十三册、第二十四册，完成论文复现、系统设计、架构演进、AI Infra 和推理框架专题。
5. 第 11 个月：学习第十五册到第十七册、第二十二册，深入多模态、reasoning、Agent 和工具协议生态。
6. 第 12 个月：学习第十二册、第十八册、第十九册、第二十册，完成求职材料、项目复盘、系统设计和 mock interview。

## 当前维护路线

排行榜模型资料专题已完成当前一轮收口：AA/DataCurve 重点 canonical 均已登记并形成可追溯资料链，Gemini 4 Argon 的 Google 官方发布资料已接入第二十一册第 95 章。后续只在排行榜出现新重点模型或已有模型出现新的权威资料时增量更新，不把工程待验证项误写成已完成验收。

1. 第一优先级：继续推进尚未收口章节的第二轮全系列精修；第二十四册现有第 1-60 章已完成第二轮，可进入总体验收和目录/索引一致性检查。
2. 第二优先级：补充 demo 级 Python 代码，优先覆盖 cross entropy、attention、sampling、KV Cache、RAG、Agent 工具调用和推理调度等高频知识点。
3. 第三优先级：联网校验高时效主题，重点包括优化器、后训练算法、推理框架、Agent 协议、多模态、Reasoning、AI Infra 和架构演进。
4. 第四优先级：同步更新第四册百科、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`PAPERS.md`、`GLOSSARY_EN_ZH.md`、`KNOWLEDGE_GRAPH.md` 和 `PROGRESS.md`。

## 新模型专题插入路线

完成第一册 Transformer 和 KV Cache 基础后，可阅读第二十一册第 75-78 章：先学 AttnRes 与残差流，再学 KDA 递归状态，随后学 CSA/HCA 压缩注意力和 mHC 稳定化。配合 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 与 `PROJECTS.md` 完成论文、代码、面试和实验闭环。

接着阅读第二十一册第 79-80 章、第六册第 18 章、第十六册第 20 章和第十七册第 15 章：前两章训练 Mistral Small 4 的统一推理模式、MoE、EAGLE/NVFP4，以及 Step 3.5 Flash 的 MTP-3、滑动窗口/全注意力和 Context Manager；第六册训练长上下文、reasoning effort、工具宿主和单位成功成本的预算意识；第十六册训练可执行长任务、verifier 门禁、奖励捷径和 `thinking.type` 协议迁移；第十七册训练 Kimi K3 的发布证据分层、思考状态 manifest、harness-aware evaluation 和 fallback 审计。对这些模型的参数量、训练架构、完整配置及未公开机制，始终回到 `research/model-update-2026-09/` 的证据等级记录。

Claude Opus 5 作为接口核验专题接在上述路线之后：先读第四册的 1M context 与 adaptive thinking 边界，再用第六册做 KV/cache、TTFT/TPOT 与成本账本，用第七册固定平台、快照、effort 和 harness 做公平评测，最后用第十七册/第二十册审计工具宿主、压缩和状态恢复。

随后阅读 Claude Fable 5.1 专题：先区分 adaptive always-on、preserved thinking 和 beta 状态协议，再用第七册验证其长任务产品定位，最后用第十七册/第二十册检查跨轮模型切换、工具进度、上下文压缩、恢复和回滚。

最后接入 Claude Sonnet 5 接口对照：先在第四册确认 1M context、128K 普通输出、300K batch 输出、Adaptive 和 Fast latency 字段，再用第六册/第二十四册建立 KV/cache、TTFT/TPOT、并发与单位成本账本，用第七册固定 Sonnet/Opus/Fable 的 revision、平台、effort、工具和 harness，最后在第十七册/第二十册审计工具宿主与状态恢复。目录字段不用于推断 Sonnet 5 的参数量、架构或训练方法。

补充 Claude Haiku 4.5 路线：先在第四册核对 200K/64K 接口边界、extended thinking 和 `fastest` 目录字段，再用第六册/第二十四册测低成本模型的 TTFT、TPOT、缓存和单位成功成本，用第七册固定 Haiku/Sonnet/Opus/Fable 的任务、平台和 harness。官方目录字段不替代实测，也不用于推断 Haiku 4.5 的参数量或训练架构。

DeepSWE v1.1 评测专题贯穿第七册、第十七册和第二十册：先读 [`deepswe-snapshot-notes.md`](research/model-update-2026-09/deepswe-snapshot-notes.md) 的任务范围与 21 个配置，再用固定模型 revision、effort、工具、verifier、超时、重试和上下文策略复现实验。报告 Pass@1 区间、成本、输出 token、Agent steps、失败类型和最终 artifact，禁止将 `mini-swe-agent` 组合结果写成基础模型能力或与其他 harness 分数直接合并。

DeepSeek V4.1-Flash 路线接在第二十一册第 80 章之后：阅读第 81 章的 CED、CSA2、Hierarchical Sparse Indexer、FP4 KV、SWA replay、Engram、DSpark 和原生多模态，再用第六册做 cache/replay/effort 预算，用第五册和第十六册审阅预训练、RL/OPD 与可验证 Agent 数据，用第七册/第十七册/第二十册固定 harness、verifier、权限和恢复协议。固定 revision 的 reference inference tree 用于核对 runtime 账本，但 `forward_spec` 不等于生成入口已接入 speculative scheduler；官方模型卡的 8B/16B active、890 bytes/global-KV-token 与 Agent 评测数字只作为带条件参考，不替代目标硬件和任务实测。
当前 serving 顺序补充为：固定 HF reference -> vLLM main -> vLLM `v0.30.0` stable release/source -> SGLang main/stable 对照 -> 完整权重 -> 数值/召回/DSpark acceptance -> 目标硬件与 tool/verifier/SLO。v0.30.0 registry、wheel 和 release notes 只能通过 release-surface gate，不能跳过 full-weight、FP4、EPD、rollback 与生产门禁。

K2 Horizon MoVA 36B/A4B 路线接在第二十一册第 81 章之后：阅读第 82 章的 dense/sparse layer 排布、MoVA value routing、FFN top-k、GQA KV cache、512K 分阶段训练和 TP/EP serving，再用第五册区分 0.9B 的 MOPD 披露与 36B 未公开 recipe，用第六册建立双路由 dispatch/通信/显存账本。随后学习 K2 7B Uno 的冻结 AR base、LoRA diffusion draft 与 `Psi-Spec` rejection verification，并用固定 revision、采样器、硬件和 harness 测接受长度、回退和有效 token。Uno 只作为锚点周边技术，不新增模型候选。

新模型发现规则：后续只从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜发现候选；沿已选锚点追踪到的官方 adapter、论文、训练方法、推理内核和部署工具可以作为周边技术，但必须单独标注，不得绕过两个排行榜新增独立模型。

Qwen3.8 专题路线：在第二十一册第 82 章之后阅读第 83 章，先掌握 GDN/Gated Attention 的混合记忆，再做 QSA 的 block 选择与两阶段训练实验，随后连接 Gated Residual 的深度残差流、N-gram 的 host-memory 带宽账本和 Muon/AdamW 的参数分组。用第四册查术语，用第五册复盘优化器和训练稳定性，用第六册复算 cache/显存/成本，用第七册区分报告自报与独立评测，再用第十六册和第十七册练习 thinking protocol、工具调用和长任务 harness。Max 作为 A95B hosted version 学习，不当作独立 open checkpoint。

DeepSeek V4 Flash Vision 专题路线：在第二十一册第 85 章先区分 Artificial Analysis 的历史 `deepseek-v4-flash-vision` 配置、官方 `deepseek-v4-flash-vision-exp` 实验 alias 和当前 `deepseek-flash`/V4.1-Flash served model；再学习图片 detail/resize/token 预算、Files `file_id` 生命周期、Responses `function_call_output(input_image)` 和流式事件回放。用第六册建立图像/文本/工具/重试/传输的单位成功成本账本，用第十七册和第二十册审计权限、取消、恢复、artifact 和 verifier。DataCurve 没有 Vision 同名行，不迁移 V4 Flash/Pro 分数；历史 384 与当前约 1024 image-token 规则始终绑定日期、alias、detail、provider 和 response model。

Kimi K2.6 专题路线：在第二十一册第 90 章先拆开 1T/32B、MLA/YaRN、MoonViT、native INT4 和 K2.5 架构复用，再阅读第十七册的 Agent Swarm 协作协议与第二十四册的 multi-turn/tool serving。练习 `preserve_thinking`/`reasoning_content` 的状态回放、工具 schema/权限/执行回执、KV/cache/量化和 300 sub-agents 的 DAG/预算/隔离 workspace；最后用 KVV 六类测试和固定硬件/harness 验收。K2.6 当前是 AA 单榜资料级闭环，DataCurve 无精确行，不迁移 K2.7 Code/K3 的 Agent 分数。

Qwen3-Omni 专题路线：在第二十一册第 91 章先区分 Instruct、Thinking、Captioner 三个同家族 artifact，再学习 Thinker/Talker 的理解—语音生成分工、AuT 的 12.5 Hz/80 ms 时间粒度、视觉 encoder、TM-RoPE 和真实 timestamp 对齐。随后练习首码本 AR + residual-codebook MTP、Code2Wav 和多码本 latency manifest；用第十五册理解多模态输入/语音输出，用第十六册/第十七册审计 Thinker 的 RAG、function calling、policy、verifier 和取消恢复，用第六册/第二十四册测异步 chunked prefill、音频 packet、显存、TTFT/首包和并发。Qwen3-Omni 是 AA 单榜内容专题闭环，DataCurve 没有精确行；报告的 `234/547 ms` 是发布方口径，不替代目标硬件 profiling 和线上 acceptance。

K2 Horizon 3.7B 专题路线：先阅读第二十一册第 82 章新增的 dense 对照，画出 36 层 dense、32Q/8KV GQA、524K RoPE 和 `num_experts=0` 的执行图，再和 36B/A4B 的前 3 层 dense、后 45 层 MoVA/FFN 双路由对照。用第五册按阶段增量理解 22.9T/8K、32K/128K/512K midtraining、512K SFT、RL expert merge 和中间 checkpoint；用第六册计算 dense GEMM、KV、long-prefill、parser、migration 和 H200 TTFT/TPOT 账本；用第七册把 AA 指标、SGLang/vLLM 发布方结果、DataCurve 缺失和本地实测分栏。旧 APPENDIX 的 `XllmForCausalLM`/FP32 只作为版本冲突练习，Uno 与 0.9B MOPD 只作为关联资料，不新增候选模型。

Qwen3-VL-235B-A22B 专题路线：先阅读第二十一册第 92 章，画出视觉 encoder、merger、DeepStack、Qwen3 MoE 和视频 timestamp 的数据流；再用第十五册计算 patch/grid、temporal bin、视觉 token 与 262K context 的预算，用第五册按 S0-S3 分析 merger 对齐、1T multimodal pretraining、32K/262K long-context adaptation 和 square-root normalized loss。用第十六册拆 SAPO、answer/reasoning/tool reward 和视觉推理，用第十七/二十册验证 GUI action 的 schema、权限、执行、观察回灌和独立 verifier，用第六册/第二十四册建立 visual prefill、expert dispatch、KV/cache、TP/EP、TTFT/TPOT、失败恢复和目标硬件 profiling 账本。把 AA 目录字段、论文自报结果、上游 raw main 和本地实测分开记录。

Qwen3.7 Plus 专题路线：先读 [`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)，确认它是 Artificial Analysis 的 AA 单榜锚点和 Alibaba Cloud 托管 snapshot，不把 DataCurve 相邻 Qwen 行迁移过来。再用第十五册拆 text/image/video 的有效预算和媒体 revision，用第十七册练 `proposal -> permission -> executor -> observation -> verifier` 的 GUI/mobile action replay，用第二十册记录 region/scope、alias、snapshot、tool schema 和状态恢复，用第二十四册比较 prefix completion、Context Caching、GPU KV cache、长输入价格和视觉 prefill。最后用本地 toy harness 测截图 revision、坐标失效、Structured Outputs/Web Search capability 差异、缓存失效和单位成功成本；不新增重复 Transformer 章节，也不把官方产品定位写成未公开内部架构。

Claude Opus 5.5 专题路线：先读 [`claude-opus-5.5-source-notes.md`](research/model-update-2026-09/claude-opus-5.5-source-notes.md)，固定 AA canonical slug、`max with fallback` 和 DataCurve 精确行缺失；再用第十六册画 always-on thinking、effort、token、tool-call、thinking binding、fallback 和 task budget 账本。用第七册把 Anthropic 发布方 Terminal/Frontier/Cursor/OSWorld/WANDR 结果与 AA provider 结果分栏，用第八册第 11 章审计 snapshot、safeguards、fallback、分母和计时定义，并在第 16 章承接 cyber/biology safeguard、同权限 fallback、人工升级和副作用审计；用第十七/二十册做长任务 coding 的 context snapshot、patch、test receipt、tool/computer-use migration、compaction、权限和 artifact verifier 回放，用第二十四册计算 cache read、工具轮次、重试、fast mode 和单位成功成本。7890 已取得真实平台 Markdown；1234/8098 的区域重定向只作为线路边界记录，不覆盖已核验 API contract。

GPT-6 Sol 专题路线：先读 [`gpt-6-sol-source-notes.md`](research/model-update-2026-09/gpt-6-sol-source-notes.md)，固定 AA canonical slug、`max` 配置和 DataCurve 精确行缺失；再用第六册计算 1.05M/922K/128K 的长上下文、272K whole-request threshold、cache read/write、Batch/Flex、Fast mode 和工具成本。用第十六册比较 `standard/pro` mode 与 `none`--`max` effort，练习 `configuration_update`、reasoning token、incomplete response 和任务预算；用第十七/二十册建立 Responses typed items、Agents API/SDK/Responses 所有权、tool search、permission、executor、verifier 和 opaque compaction replay；用第二十四册记录 schema token、prefix/cache、compaction、TTFT/TPOT、重试和单位成功成本。模型页与通用 runtime 文档不证明参数、架构或训练 recipe，因此不新增重复 Transformer 章节，也不迁移其他 GPT 的 Agent 成绩。

GPT-6 Luna 专题路线：先读 [`gpt-6-luna-source-notes.md`](research/model-update-2026-09/gpt-6-luna-source-notes.md)，固定 AA canonical slug、`max` 配置、2026-09-22 release 和 DataCurve 精确行缺失；再用第六册比较 Luna/Sol 的 1.05M/922K/128K、2026-05-18 cutoff、272K whole-request threshold、`$0.10/$0.50` 价格、Batch/Flex、Fast mode 和工具成本。用第十六册比较 GPT-6 family 的 `standard/pro` mode、`none`--`max` effort、`configuration_update`、reasoning token 和 incomplete response；用第十七/二十册建立 tool capability、permission、executor、verifier、Agents API/SDK/Responses 所有权和 opaque compaction replay；用第二十四册记录 schema token、prefix/cache、TTFT/TPOT、重试和单位成功成本。Luna 没有公开独立架构/训练 recipe，且 DataCurve 无精确 Agent 行，因此不新增重复 Transformer 章节，也不迁移 Sol/Astra/GPT-5.6 的成绩。
