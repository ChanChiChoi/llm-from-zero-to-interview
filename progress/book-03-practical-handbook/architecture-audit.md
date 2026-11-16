# 第三册架构审视与精修记录

## book_contract

- **定位**：第三册《大模型算法岗实战手册》是项目训练教材，把前两册的原理转换成教学规模、可运行、可检查的实验与应用项目；它在书系中负责项目闭环和可复核交付，不是代码片段合集或生产部署手册。
- **目标读者与前置**：具备基础 Python 和机器学习概念，希望把大模型原理转成代码和项目成果；第一册提供共同技术主线，第十四册可补 PyTorch 工程细节；不假设具备 GPU 集群或 Agent runtime 经验。
- **读者结果**：能够检查 PyTorch 张量/梯度，训练并解释小 GPT，完成微调与偏好实验，测量推理服务，建立带权限和证据的 RAG/Agent，进行离线及线上评估，并整理包含运行与失败证据的作品集。
- **范围与类型**：16 章项目教材；第 1—10 章为模型实现、训练适配、推理、RAG/Agent、离线评估及线上验证主线；第 11—14 章为图文、Diffusion、语音与多模态交付选读支线；第 15—16 章负责项目归档和技术表达。
- **不承担的范围**：大规模训练、生产推理/Serving Engine、评估科学、数据工程、完整 Agent 方法与 runtime、工具协议、完整多模态生成等深度内容由相应专题书承担。
- **书系交接**：承接 book-01 的核心原理和 book-02 的机制判断，向 book-14 的 PyTorch 工程、book-05/06/07/09/15/17/20/22/24 等专题提供实践入口；作品集可继续交给 book-12 的职业与面试流程。

## intro_audit 与 toc_audit

- 简介和目录都将本册定义为实战项目教材，主线为实现—训练/适配—推理—RAG—Agent—评估—线上验证—作品集—面试，多模态支线单独标为选读。简介明确强调运行代码不能代替机制解释，指标只能描述有保存证据的实验范围；与本册在书系中的独特价值一致。
- 目录保留 16 个逻辑章节，并在需要多个读者结果或展开阶段较多的章节中列出连续分篇；主线 1—10、选读 11—14、收束 15—16 的顺序保留。
- 原简介没有逐章短摘要，而规范要求简介说明各章问题、作用及承接；已新增 16 条一两句摘要及章节链接。后续逐章完成时按实际精修内容复核对应摘要。
- 简介和目录的 Mermaid 图源码尚未做目标平台渲染验证；留待最终门禁。

## chapter_map

| 章节 | 功能与前置 | 交接 |
| --- | --- | --- |
| 01 PyTorch 基础实战 | 建立 tensor、loss、gradient、optimizer 的数值检查基线 | 组件实现 |
| 02 Transformer 组件实战 | 用可检查的张量和中间量构造模型部件 | 小 GPT |
| 03 从零训练小 GPT | 串联文本数据、next-token 训练、曲线和生成 | SFT |
| 04 Hugging Face 微调实战 | 定义对话样本、监督范围和适配配置 | 偏好优化 |
| 05 偏好优化实战 | 组织偏好对和优化实验，核对行为变化 | 推理服务 |
| 06 推理优化实战 | 在固定负载下测量服务成本与性能 | RAG 应用 |
| 07 RAG 项目实战 | 文档、检索、重排、引用和评估共同形成有权限的证据回答链 | Agent 只读证据服务 |
| 08 Agent 项目实战 | 约束工具提议，记录执行回执与轨迹 | 离线评估 |
| 09 评估与 Debug 实战 | 建立离线基线、失败归因和回归集 | 线上评估 |
| 10 Agent 轨迹与线上评估实战 | 连接轨迹诊断、线上实验和灰度决策 | 主线项目交付 |
| 11 图文理解与检索实战 | 选读：图文数据、表征和检索链路 | 多模态综合评估 |
| 12 Diffusion 数值实验 | 选读：从前向加噪、训练目标追到反向更新、文本引导和生成验收 | 多模态综合评估 |
| 13 语音识别与问答实战 | 选读：记录 ASR 转写并诊断误差传播 | 多模态综合评估 |
| 14 多模态评估与系统交付 | 汇总媒体输入、证据、安全和评估要求 | 作品集整理 |
| 15 作品集实战 | 归档可复核项目材料、失败和个人贡献 | 技术表达 |
| 16 技术表达与系统设计实战 | 把项目的约束、机制、结果和边界转成答辩 | 求职/专题学习 |

## discovery_scan

- **日期与范围**：2026-10-29；按项目教材的边界扫描框架、工具版本和项目证据维护候选，阅读官方发布、官方框架文档和官方可观测性规范；不据此决定增加独立章节。
- **候选 1：跨项目可复核产物**。简介已承诺配置、数据契约、运行记录、失败样例与贡献说明；逐章扫描时确认这些字段是否能沿第 1—10 章连续传递，并在第 15 章归档。优先以现有交接产物完善，不预设新增章节。
- **候选 2：GenAI trace 的敏感内容边界**。官方 OpenTelemetry 属性说明输入/输出消息、系统指令、检索查询及工具参数/结果可能含敏感信息；需重点核对第 8、10、14、15 章是否明确过滤/截断、权限和留存边界，避免把 trace 可回放误写成可默认存储原文。
- **候选 3：动态库 API 与版本锁定**。PyTorch 2.15.0 于 2026-10-28 发布；TRL `main` 文档明确它与稳定 pip 版本不同；vLLM `latest` 指标页面动态演进。逐章扫描适配代码和 benchmark 时，固定项目使用的版本、硬件/模型及测量端点；不因新版本发布而机械更新教学接口。
- **不扩展项**：当前目录已经分章覆盖训练、Serving、RAG/Agent、多模态、评估和作品集；新增完整 Agent runtime、生产 Serving、评估科学或多模态生成会与专题册重复。没有发现足以改变 16 章骨架的证据，后续章级扫描若发现结构缺口再回到架构审视。
- **来源记录**：上述新知识点及版本/敏感数据的来源边界见 `research/project-tooling-and-observability-2026.md`；本文件仅记录本册候选去向与架构决定。

## architecture_decisions

1. 保留现有 16 章和编号；主线与多模态选读支线分开，避免实战教材扩展成第二本概念百科或专题手册。
2. 保留 RAG 与 Agent 分章：前者建立证据，后者治理外部动作；随后先做离线诊断，再做轨迹与线上验收。
3. 第 15、16 章分别负责证据归档与面试表达，避免“项目已运行”直接等同于成果可信或本人贡献清楚。
4. 在简介补入逐章摘要，以满足导航和章节摘要门禁；目录、正文章节结构不因此重排。

## current_status 与交接

- 第三册的既有架构审视和本轮册级定位已完成；当前全册重写进行中，第 1—13 章正文已完成本轮重写，第 14—16 章待处理。此前的 `chapter_scan` 记录属于历史精修证据，不代表本轮正文已复核。
- 第 2 章拆为四篇、第 3 章拆为五篇、第 4 章拆为七篇、第 6 章拆为七篇、第 7 章拆为六篇、第 8 章拆为五篇；简介、目录及已触及的交叉链接已同步。各章完成后继续检查本章摘要、相关索引和链接。
- 本轮不运行书中代码、测试、模型、benchmark 或 PDF 构建。代码只有静态审阅证据时明确标注未运行；改过的 BPE 演示已更新状态日期。
- 当前下一入口：第 14 章《多模态评估与交付》。整册连续复读、Mermaid GitHub 实际渲染和外链终审留到全部章节重写后；在这些门禁关闭前不标记第三册完成。上下文压缩或新会话恢复时，先完整重读 `写作规范/总纲.md` 和当前阶段相关完整模块，再读本交接记录。
- 保留用户明确约束：不更新 PDF；不执行书中程序、模型、benchmark 或测试，仅做正文、公式、代码静态审阅和必要的文档静态检查。
- book-02 的 GitHub Mermaid 实际渲染及旧外链全量核验仍未完成；book-01 和 book-02 的整册连续复读/链接门禁也依各自进度处理。

## 历史接续状态：第 1 章完成（由下方全册重写轮次覆盖）

- 第三册现为进行中：全书架构审视已完成，第 1 章已通读并修订；其余 15 章尚待扫描和精修。
- 简介摘要、配套索引和跨章交接要在后续章节完成时复核；项目级版本锁定、指标定义、敏感 trace 处理和作品集证据连续性列为后续章级检查重点。
- 用户要求不执行书中代码、测试、模型或 benchmark，也不更新 PDF。首章代码只沿用有日期/环境的既有执行记录，之后代码未变时可依规范复用。
- 下一步从 `chapters/02-transformer组件实战.md` 进入；书内 Mermaid 和外链最终门禁尚未完成。

## chapter_scan：第 1 章《PyTorch 基础实战》

- **扫描范围与问题链**：2026-10-29 完整通读 `chapters/01-pytorch基础实战.md`（1,244 行）。围绕一次参数更新，检查线性回归与形状、分类与交叉熵、语言模型 token loss、计算图/反向传播、优化器与混合精度、学习率调度和 checkpoint 的依赖关系；按 PyTorch 2.15 官方文档复核版本敏感 API 说明。
- **候选去向**：保留当前从数值基线到训练循环的六段结构。有限差分、参数分组、梯度累积、mixed precision、恢复状态均留在本章作为训练循环组件；分布式、GPU 集群及大规模状态分片交给第五册/本册后续适配章。本章没有理由新增独立主题。
- **正文修改**：为 Adam 原论文引用补成可点击的 arXiv 链接；未改公式、示例代码或讲解结论。简介第 1 章摘要已与本章定位相符。
- **篇幅与例子**：解释性汉字 12,005，按单篇核心教学单元计，超过 1,800 下限。保留可手算的线性回归梯度/参数步进、CE 稳定计算、反向传播链、优化与累积边界；代码均未被用于替代手算。
- **代码、公式、图示与链接**：23 个 Python 代码块均带有 2026-10-19、Python 3.12.9、PyTorch 2.12.0+cpu 的既有验证说明；代码正文和输出在本轮未改变，因此按可追溯记录沿用，未执行程序。44 个 math 围栏和全章围栏配对；1 个 Mermaid 图仍待 GitHub 页面渲染。官方 PyTorch 文档在本次访问时跳转至 2.15，语义核验见 `research/project-tooling-and-observability-2026.md`；源清单中的 Adam arXiv 链接已补齐。
- **遗留项与下一步**：旧外链全量审计和 Mermaid 页面渲染仍纳入本册最终门禁；没有运行测试、模型或 PDF。下一步处理第 2 章《Transformer 组件实战》，先核对张量形状、实现接口和与本章训练组件的依赖。

## chapter_scan：第 2 章《Transformer 组件实战》

- **扫描范围与问题链**：2026-10-29 完整通读并静态审阅 `chapters/02-transformer组件实战.md`（精修后 2,137 行），检查 embedding/position、attention、multi-head shape、mask、block、RoPE、RMSNorm 和 SwiGLU 如何接成可验证的数据路径。
- **定向扫描与候选去向**：对照 `research/transformer-architecture-research-2026.md` 中 SpanNorm、PowLU、MoA、Log-Linear Attention、Gated DeltaNet-2 等已登记来源，判定它们是研究型替代方案或不同状态路线，不扩成这篇密集 Transformer block 实作的主体；相关架构比较由第二册 Transformer 进阶篇承载。GQA 仅说明 Q/KV head 数契约，并链接到本册第 6 章的 cache shape 与容量练习。本章不增加新的独立主题，保留 2.1—2.7 原顺序。
- **结构与正文修改**：将开篇流程图改成明确的 Pre-RMSNorm 代表路径，区分加性位置表示在 embedding 后加入、RoPE 在 Q/K 投影后应用、两个子层前各自归一化及残差相加；同时收窄“每个主题均有可运行代码”的过宽承诺。为 Transformer、RoPE、RMSNorm、GLU/SwiGLU 与 LLaMA 组合补充靠近结论的论文入口；删除 mask、归一化、RoPE 高频/低频及 RMSNorm/SwiGLU 讲解中重复的段落。
- **篇幅与例子**：按 2.1—2.7 作为七个核心教学单元，静态计数分别为 1,921、1,859、1,854、1,860、1,905、1,844、1,831 个解释性汉字，均超过 1,800 下限。保留 embedding/位置向量、Q/K/V 手算、attention 加权、head shape/参数量、三 token mask、残差梯度、RoPE 相对距离与共同 offset、LN/RMSNorm、SwiGLU 和参数账本例子。
- **代码、公式与图表**：22 个 Python 代码块的可执行内容未改变，均静态审阅、未运行。原状态标签指向 `.tmp/llm-book-cpu-check`，但 `progress/` 中没有逐块命令、输入与边界记录；依据当前规范，读者侧标签统一改为“静态审阅，未运行（2026-10-29）”，不保留内部临时路径。69 个 math 围栏、1 个 Mermaid 图及总计 102 对围栏闭合；新增 RoPE 同时平移位置时的内积恒等式，说明它只适用于固定向量的旋转内积，不推成整模型对 padding、mask 或窗口变化均不变。Mermaid 语法仅做静态检查，GitHub 实际渲染仍待确认。
- **证据与链接**：新增 `research/transformer-component-apis-2026.md`，记录 PyTorch 2.12 的 `padding_idx`、SDPA/mask/dropout、MultiheadAttention bool mask、LayerNorm 统计和 Dropout eval 语义。正文 API 链接固定到与现有示例状态相符的 2.12 文档，相关论文链接贴近机制说明；10 个唯一外部来源页已打开核对。本章新增到第 6 章的相对链接目标存在，简介中的第 2 章摘要仍准确。
- **同步与遗留项**：PAPERS、EXERCISES、术语表和知识图谱已有 Transformer、RoPE、GQA、RMSNorm 与 SwiGLU 入口，本轮不重复增项。没有修改目录或简介。唯一尚未关闭的本章外部门禁是 GitHub 页面实际 Mermaid 渲染；没有执行书中程序、测试、benchmark、模型或 PDF 构建。
- **下一步**：进入第 3 章《从零训练小 GPT》，检查训练数据到 next-token loss、训练曲线与生成的完整闭环；全册连续复读及 Mermaid 目标平台门禁保留在最终验收。

## chapter_scan：第 3 章《从零训练小 GPT》

- **扫描范围与问题链**：2026-10-29 完整通读并静态审阅 `chapters/03-从零训练小gpt.md`（1,940 行）。检查数据切分、tokenizer 拟合、next-token 样本、最小 GPT、训练/验证 loss、解码、checkpoint、曲线解释、BPE 迁移和实验复现如何组成闭环；本章结构与小节顺序保留。
- **章级扫描候选及去向**：先按文档/实体去重并切分数据，再仅从 train split 拟合词表、编码和采样；将按独立文档采样或显式 attention 隔离作为避免跨文档泄漏的边界说明，保留 EOS 不会自行重置 causal attention 的区别。packing/masking 的结论限定于 ACL 2024 论文条件。保留 PyTorch/Hugging Face 版本化 API 契约、EOS=PAD 下有效位置 mask 风险、checkpoint 安全加载、有效 token 预算与验证集选择边界；不新增章节。关于去重、sequence composition、复现评估、nucleus sampling 与固定版本 API 的来源事实和边界见现有 `research/` 记录。
- **正文修改**：修正 tokenizer 训练与 split 的先后关系，避免示例验证流复制训练前缀；澄清 EOS、文档边界、padding/loss mask、eval/no-grad、交叉熵与 token 计数语义。修复多前导维 logits 直接交给 `torch.multinomial` 的 shape 契约，展平为行抽样后恢复输出形状，并支持独立 generator；把 top-p 阈值示例移离浮点边界，纠正 Holtzman 论文链接。checkpoint 使用显式 `weights_only=True`，不再通过宽泛 `TypeError` 回退；统一 CSV header/row 的 elapsed 字段和示例输出目录。补充变长验证 token 加权、选模集不等于独立测试集、EOS=PAD 和 padding 有效位置的说明。
- **篇幅与例子**：3.1—3.7 七个核心单元的解释性汉字数分别为 2,138、2,075、1,972、1,867、1,862、2,048、2,061，均超过 1,800 下限。数值推演覆盖右移的输入/目标对齐、交叉熵及有效 token 分母、temperature/top-k/top-p 候选集、checkpoint 保存间隔的平均/最大重算时间、按 token 加权的验证 loss、验证选模 step 300/600 的边界、字符级到 BPE 的词表/序列账本，以及同 token 预算下不同硬件吞吐的 wall-clock 差异。训练预算示例明确 `tokens_seen` 是处理过的 token positions（含重复抽样位置），例如 `(8,32)` 批次形状跑 1,000 步处理 256,000 个位置，`(64,128)` 则为 8,192,000 个位置。
- **代码状态**：17 个 Python 块均为“静态审阅，未运行（2026-10-29）”，按位置标识为 §3.1.2、§3.1.3、§3.1.4、§3.1.5、§3.1.6、§3.1.7、§3.2.4、§3.2.6、§3.3.4、§3.3.5、§3.4.3、§3.4.4、§3.4.5、§3.5.4、§3.6.3、§3.6.4、§3.6.6。旧“已验证（2026-10-20；.tmp/llm-book-cpu-check…）”缺少 `progress/` 中逐块可复核的命令、输入、结果和边界记录，因此不能沿用；本轮不得执行书中代码、测试、模型或 benchmark。§3.3.4/3.3.5 的采样形状、generator 和边界示例及 §3.4.3/3.4.5 的加载/日志代码有改动，只完成静态审阅，不声称执行验证。
- **公式、图表与链接**：41 个 math block 和 1 个 Mermaid flowchart；全文 71 个 fenced block 均配对（142 个围栏标记），17 个 Python 块与 17 个状态标记对应。行内 `$` 分隔符计数为 0；主要 API 均链接至 PyTorch 2.12 或 Transformers v5.17.0 固定文档，Holtzman 指向 1904.09751v2；本章无相对 Markdown 文件链接。Mermaid 源码仅作静态检查，GitHub 页面实际渲染仍待确认。
- **配套同步**：PAPERS.md 补入 ACL 2024 packing/masking 论文与 Holtzman nucleus sampling 论文，并写明各自证据边界。EXERCISES.md 已覆盖 BPE、top-k/top-p；GLOSSARY_EN_ZH.md 已有 Temperature、Top-k、Top-p/Nucleus Sampling；KNOWLEDGE_GRAPH.md 已覆盖 tokenization、packed-sequence mask、cross entropy、perplexity 和 sampling subchain；INTERVIEW_BANK.md 已有 PPL、解码和 EOS/PAD 题目，因此其余索引不重复新增。本章简介摘要与目录入口仍准确，无需改目录或简介。
- **遗留项与下一步**：全册 16 章中尚余 13 章，并保留最终连续复读；本章 Mermaid GitHub 页面渲染、外链全量核验留到本册最终门禁。没有运行测试、书中程序、模型、benchmark 或 PDF。下一步处理第 4 章《Hugging Face 微调实战》。

## chapter_scan：第 4 章《Hugging Face 微调实战》

- **扫描范围与候选去向**：2026-10-29 完整通读并静态审阅 chapters/04-huggingface微调实战.md。检查 checkpoint/tokenizer 身份、chat template、assistant labels mask、全参数与 LoRA/QLoRA 状态、adapter 保存/恢复及 SFT 行为回归的连续教学链。保留 4.1—4.6 既有结构；LoFT 只作为有边界的 ICLR 2026 研究指针，不扩写成训练配方或新章节；现有简介、目录和章节摘要仍准确。
- **正文与接口修订**：将 tiny-gpt2 和 Qwen2.5-0.5B 示例固定到包含权重的 commit revision，加载 tokenizer/model/adapter 时保持 base revision 一致；移除 Qwen 示例不需要的 trust_remote_code。按 Transformers v5.17.0 更新 dtype 参数，并将 device_map=auto 明确限定为推理用途；QLoRA 单卡训练交给训练进程放置，多卡说明要求采用训练框架支持的并行策略。修正 prompt 边界跨 token 时的 offset mask 规则并解释保守丢弃的响应前缀；collator 增加右侧 padding 与 token/label ID 范围检查，同时说明 embedding 行数仍须单独核对。ToyTokenizer 对未知 token 改用 UNK，不再映射到 EOS。补足梯度裁剪、非有限 loss、调度器及评估/checkpoint 责任边界，消除重复初始化说明，并修复 2×2 LoRA 秩和参数量例子的维度/结论。
- **篇幅与数值例子**：4.1—4.6 六个核心单元的解释性汉字数分别为 1,939、2,036、1,890、1,974、1,896、1,912，均超过 1,800 下限。保留并核对 12-token prompt 加最多 8 个新 token 的上下文预算、不同长度 prompt 左 padding、prompt/response loss 均值与错误纳入 prompt loss 的对照、900+200 token 在 1,024 截断后的目标剩余量、7B 全参训练状态量级与累积批次、4096×4096 全参/LoRA 参数账本、QLoRA 7B 裸权重与量化元数据估算，以及成对胜率和安全回归计数例子；各计算均说明适用范围，不作为模型质量结论。
- **代码、公式、围栏和链接**：48 个 Python 块均标记“静态审阅，未运行（2026-10-29）”，稳定位置标识为：§4.1.2 Py1；§4.1.3 Py1–2；§4.1.4 Py1–3；§4.1.5 Py1；§4.2.1 Py1–2；§4.2.2 Py1–3；§4.2.4 Py1–2；§4.2.5 Py1；§4.2.6 Py1–2；§4.2.7 Py1；§4.2.8 Py1；§4.3.3 Py1–2；§4.3.4 Py1–2；§4.3.5 Py1；§4.3.6 Py1；§4.4.2 Py1；§4.4.3 Py1–3；§4.4.4 Py1；§4.4.5 Py1–3；§4.4.7 Py1；§4.5.2 Py1；§4.5.3 Py1–3；§4.5.4 Py1–2；§4.5.6 Py1；§4.5.7 Py1；§4.6.3 Py1；§4.6.4 Py1；§4.6.5 Py1–2；§4.6.8 Py1；§4.6.9 Py1。全章 228 个围栏标记、114 个配对代码围栏块（48 Python、40 math、25 text、1 Mermaid）；行内美元符号分隔符为 0。相对链接静态检查无失效项。Mermaid 源码作静态检查，目标平台实际渲染仍留待最终门禁。
- **证据与配套同步**：正文按 Transformers v5.17.0、PEFT v0.18.0、TRL v1.14.1、PyTorch v2.14.0 文档核对，并明确这些文档版本不是联调依赖锁；模型快照、assistant mask、device placement、PEFT 初始化/缩放、QLoRA 和 LoFT 的来源及证据边界记入 research/sft-adaptation-loss-and-forgetting-2026.md。PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY_EN_ZH 和 KNOWLEDGE_GRAPH 已有相关 SFT/LoRA/QLoRA 入口，本轮不重复新增。
- **未执行事项与交接**：未运行书中程序、测试、模型、benchmark 或 PDF。没有改目录；简介第 4 章摘要与正文一致。最终连续复读、旧外链全量核验和 Mermaid 页面渲染仍待收口。下一步处理第 5 章《偏好优化实战》。

## chapter_scan：第 5 章《偏好优化实战》

- **扫描范围与问题链**：2026-10-29 完整通读 `chapters/05-偏好优化实战.md`（约 2,900 行），检查偏好对的可比性与泄漏、Reward Model 的 pairwise 目标与池化、手写 DPO 的 response log-prob/mask/reference、TRL 训练接口、训练后行为评估之间的衔接。保留 5.1—5.6 原有结构；没有证据支持增章或拆章。
- **候选去向**：检查 DPO 序列求和与 token 均值、reference 与 beta 的假设、chosen/rejected 候选来源、奖励模型的长度捷径、2025 偏好数据结论与 2026 likelihood-displacement 研究。保留 SimPO、IPO、KTO/ORPO 等在已有比较位置作边界清楚的方法对照；2026 正则化研究仅作为预印本候选，不扩写成通用训练配方。2025 数据论文在来源记录中明确仅核对官方摘要，未据此推广 rejected 样本普遍无用。来源与阅读范围见 `research/dpo-preference-objective-and-length-2026.md`。
- **正文与接口修订**：固定 Transformers v5.17.0、TRL v1.14.1、PEFT v0.18.0、PyTorch v2.14.0 的文档语义，并说明这不是联调依赖锁；tiny-gpt2 示例固定到单一 Hub commit。对 prompt 级切分加随机种子；快速 tokenizer 的偏好预处理检查截断后仍保留 response token；Reward collator 检查 pad 与样本 token ID 范围；RewardModel 显式接收模型 revision。DPO 数据划分可复现；policy/reference 使用同一固定快照，说明 `ref_model=None` 的 TRL v1.14.1 行为。移除不受该版本 DPOConfig 支持的 `max_prompt_length` 和旧 `tokenizer=` Trainer 示例，区分本地 prompt 预算与 trainer 参数；QLoRA 单卡训练示例移除 `device_map="auto"`。配置审计明确空格分词只是占位，不能当作目标 tokenizer 计数。修正 §5.5.13 的损坏公式字符和行内数学分隔；§5.6.9 的格式判定改成恰好三条清单项，评分时跳过不适用标准并拒绝无可用标准的输入。
- **篇幅与例子**：按六个核心单元统计解释性汉字（排除标题、代码、公式、text 样例和状态行）分别为 1,830、1,824、1,861、1,911、2,050、1,827，均达到 1,800 下限。关键可复算例子包括：§5.1 同一 prompt 的 5 个候选可形成 10 个 pair，但按 prompt 留组避免近重复泄漏；§5.2 reward 分差 1.5 时 chosen 概率约 0.818、loss 约 0.201，翻转分差后 loss 约 1.704；§5.3 长 3/5 token 的序列补齐到宽度 5 时，短序列 reward 必须取索引 2 而非 padding 列索引 4，另用 100 prompts × 4 候选形成 300 pairs 对照按 prompt 留出 20 个验证 prompt；§5.4 同 policy/reference 初始化时 (z=0)、loss (ln2\approx0.693)，并补入 2-token chosen 增量和 0.30、4-token rejected 增量和 0.40 的 sum/mean 对照：总和 margin (-0.10)、(z=-0.01)、loss 约 0.698；均值 margin (+0.05)、(z=0.005)、loss 约 0.691，显示聚合口径可反转梯度方向；§5.6 40 个盲评 prompt 有 27 胜、10 负、3 平时，含平局胜率 67.5%，去平局 win rate 约 73.0%，并分别核算拒答任务的正确拒答和过度拒答。
- **代码状态**：全章 43 个 Python 块均标记“静态审阅，未运行（2026-10-29）”；稳定位置为 §5.1.2 Py1–2、§5.1.3 Py1–3、§5.1.5 Py1、§5.1.6 Py1、§5.1.7 Py1、§5.1.8 Py1、§5.1.10 Py1；§5.2.3 Py1–3、§5.2.6 Py1–2、§5.2.8 Py1；§5.3.2 Py1–2、§5.3.3 Py1、§5.3.4 Py1、§5.3.5 Py1–2；§5.4.3 Py1、§5.4.4 Py1–2、§5.4.5 Py1、§5.4.6 Py1–2、§5.4.8 Py1；§5.5.2 Py1、§5.5.3 Py1、§5.5.4 Py1、§5.5.5 Py1、§5.5.6 Py1、§5.5.7 Py1–2、§5.5.8 Py1、§5.5.9 Py1、§5.5.11 Py1；§5.6.3 Py1、§5.6.4 Py1、§5.6.8 Py1、§5.6.9 Py1。代码正文改动涉及 §5.1.10 Py1、§5.3.2 Py1–2、§5.3.3 Py1、§5.5.3 Py1、§5.5.5 Py1、§5.5.6 Py1、§5.5.9 Py1、§5.5.11 Py1、§5.6.9 Py1（共 10 块）；§5.5.7 删除了一个旧 `tokenizer=` API 代码块。其余 33 个代码正文未改。旧状态标签虽附 2026-10-20 环境摘要，但 `progress/` 无逐块命令、输入、实际/预期结果及边界记录，故本轮不沿用“已验证”；没有运行书中代码、测试、模型或 benchmark。
- **公式、图表与链接**：围栏共 125 块/250 个标记（43 Python、48 math、33 text、1 Mermaid），全部配对；行内 `\(` 与 `\)` 各 83 处，无未转义 `$`、控制字符或失效本地相对链接。Mermaid 只做源码静态检查，GitHub 页面渲染保留到目标平台门禁。章节外部 API 链接已固定到上述版本文档，DPO 原论文用 v3 链接；无动态 `main`/`stable` API 页面残留。
- **证据与配套同步**：TRL v1.14.1 的 DPO 数据/API/左 padding/reference/loss 聚合、Transformers v5.17.0 的 bitsandbytes 训练设备放置、PEFT v0.18.0 的 LoRA 初始化/Conv1D 约定、PyTorch v2.14 的损失与 log-prob API，以及固定 tiny-gpt2 commit 的来源边界已补入 `research/dpo-preference-objective-and-length-2026.md`。检查 `简介.md` 的第 5 章摘要、目录和 PAPERS/EXERCISES/INTERVIEW_BANK/GLOSSARY_EN_ZH/KNOWLEDGE_GRAPH 现有 DPO、Reward Model 与偏好数据入口，均与正文一致，不需重复增项；没有改目录。
- **静态验收与遗留项**：`git diff --check` 通过；围栏、字数、状态标记、公式分隔符、本地相对链接和控制字符静态检查通过。文档版本不等于已联调依赖锁；tiny-gpt2 只作 API/数据流示范；目标平台 Mermaid 渲染、最终整册连续复读与全章外链终审仍待本册收口。没有更新 PDF。下一步处理第 6 章《推理优化实战》。

### 初学者视角正文重写（2026-11-13）

- **修改目标与主线**：按“懂深度学习和 Python、刚接触大模型”的读者视角重写代码围栏外的讲解，保留 5.1—5.6 结构。章节开头补了阅读路线，并用“每天 20 分钟学 Python”的同一问题串起偏好对、Reward Model、DPO 和评估；首次出现的 SFT、RM、PPO、DPO、policy、reference、token 概率等概念补充直白解释，区分相对偏好、奖励分数、排序概率和回答正确性。
- **难点拆解与去重**：把 Reward Model 的分差、sigmoid 概率、loss 和梯度方向连成手算过程；DPO 部分先用 token 条件概率乘积解释整句概率，再说明 log 概率、KL 惩罚和偏好 logit，并展示 KL 正则化目标到隐式 reward 的关系。将数据审计、奖励模型实现、TRL 配置及底层细节明确标成进阶查阅，压缩重复的边界提醒和路线对比。
- **解释性篇幅**：排除标题、代码/公式围栏、text 样例、状态行、行内代码和链接后，5.1—5.6 分别为 2,099、1,947、1,814、2,033、2,060、2,058 个汉字；各节均包含能手算或对照的具体例子。
- **代码与静态检查**：相对本轮开始版本，43 个 Python 围栏、33 个 text 示例和 1 个 Mermaid 围栏逐字未变；数学围栏随讲解调整，从 46 个增至 50 个。当前共 127 个围栏块、全部配对；行内数学分隔符各 59 个，无控制字符；3 个 Markdown 链接中无失效相对链接。git diff --check 通过。
- **执行边界与遗留项**：本轮未运行书中代码、测试、模型、benchmark、构建或 PDF，也未新增联网研究。目标平台 Mermaid 渲染、最终整册连续复读和外链终审仍按本册既有门禁处理。

### GitHub 数学公式渲染修正（2026-11-13）

- 用户反馈 §5.4.1 中“回答概率按 token 条件概率相乘”的展示公式在 GitHub 报 `Extra open brace or missing close brace`，而紧邻的行内 `\(\pi(y\mid x)\)` 正常。
- 对照 Markdown 源码后确认，公式的大括号本身成对；差异在展示公式中的 `y_{<t}`：下标以 `<` 开头，可能在 Markdown/HTML 解析阶段被误当成标签，导致后续数学内容解析错位。此写法也不符合模块 04 关于避免尖括号下标的要求。
- 已改为显式前缀范围 `y_{1:t-1}`，并说明 `t=1` 时前缀为空。未改公式含义、代码或训练结论；全章仅此一处含 `_{<` 的数学写法。待用户在 GitHub 页面确认实际渲染。

### GitHub 行内数学分隔符修正（2026-11-13）

- 用户进一步指出同一段及后文叙述中的行内公式仍显示为原始文本。检查发现全章 61 对行内数学分隔符均写作 `\(...\)`，而[GitHub 官方文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)列出的行内语法为 `$...$` 或 `$` 反引号 `$` 形式；本章规范也要求使用单美元符号。
- 全章 61 对行内分隔符已统一改为 `$...$`。此前文件没有其他美元符号；展示公式、代码块和数学内容未改。尚待用户在 GitHub 页面确认行内公式渲染。

### 全章重写（2026-11-13）

- **范围与读者路径**：按用户最新指示完全替换第 5 章正文，代码也重新编写，不再沿用此前“保留代码围栏”的限制。章节由原先密集的实现片段改为一条连续路径：偏好数据怎样记录比较、Reward Model 怎样学习排序、PPO 路线怎样使用奖励、DPO 怎样从 KL 正则化目标推出、序列概率怎样落实到 token mask、TRL 怎样接训练循环、训练后怎样做独立评估。
- **教学主线**：贯穿“每天只有 20 分钟，怎么开始学 Python”的假设例子，说明 chosen/rejected 是特定标准下的相对标签。对 SFT、RM、RLHF、PPO、policy、reference、token 概率、KL 和 DPO loss 补了首次解释；加入 RM 分数差与梯度、DPO 的隐式 reward 推导、policy/reference 概率对照，以及 40 个盲评问题的胜负平分母例子。结尾要求读者留下数据、配置、评估输出和失败案例，呼应第三册实战交付目标。
- **代码与框架**：四个 Python 块均为本轮新写：偏好数据基本审计、response-mask 序列 log-probability、DPO loss、TRL v1.14.1 最小训练流程。均标为“静态审阅，未运行（2026-11-13）”；未执行书中代码、测试、模型、benchmark 或训练。
- **证据与版本**：重新核对 DPO 原论文 v3、InstructGPT 论文、PPO 原论文 v2，以及 TRL v1.14.1 DPOTrainer 和数据格式文档。章节固定引用 v1.14.1 参数，不声称文档版本等于当前最新版本或已联调依赖；source-backed 资料和边界补记于 research/dpo-preference-objective-and-length-2026.md。
- **静态验收与同步**：24 个围栏块、48 个围栏标记成对（4 Python、15 math、2 text、2 JSON、1 Mermaid）；84 个单美元行内数学分隔符成对，行内括号分隔符为 0；未发现以尖括号开头的数学下标；本章无本地相对链接。git diff --check 通过。简介和目录中第 5 章的摘要仍符合本章结果，本轮不改导航或配套题库。未更新 PDF；GitHub 上 Mermaid 和数学公式的页面渲染仍待用户实际查看。

### 补足覆盖面与推导主线（2026-11-16）

- **用户反馈与诊断**：用户认可新稿前后衔接改善，同时指出原稿有较多知识点，新稿删减过多；进一步明确不要求所有旧知识点原样保留，但主线需要更丰满。此次目标是沿用连贯主线，有选择地补回支撑读者理解、实现和评估的内容，不按原稿逐项恢复。原稿约 2,900 行中有大量重复说明、版本/API 细节和代码围栏；此前把“完全重写、忽略代码要求”误解为可以显著缩窄教学覆盖面，且没有先做旧知识点去向盘点。
- **规范调整**：此前取消核心单元统一 1,800 字下限后，篇幅检查容易被误读成鼓励压缩。本轮修订总纲、模块 02，明确“无统一字数门槛”不等于没有覆盖要求；改写前应判断旧知识点的保留、合并、进阶、迁移或删除，区分章节覆盖面与单点解释深度，并说明主线用于组织次序，不能替代本章承诺的内容。
- **选择性补充**：恢复偏好数据来源、候选覆盖、标注分歧、长度/截断审计；奖励模型的稳定 pairwise loss、margin、padding 池化、留出评估和从 batch 更新到训练/验证/checkpoint 的过程；PPO 中 RM 与 value model 的区别及裁剪目标算例；DPO 的策略梯度方向、相同起点自检、token shift/mask 对齐、求和/均值长度口径；TRL 数据契约、有效 pair batch 估算、LoRA/QLoRA 参数账本与 checkpoint/恢复区别；以及 base/SFT/DPO 分阶段对照、盲评、LLM judge 审核、指标切片和逐样本复核记录。补上从 KL 正则化目标经拉格朗日乘子推到闭式最优策略的中间步骤，并用二候选分布算例解释 beta 的作用。
- **有意不恢复的内容**：不复刻原稿 43 个 Python 代码块及各版本接口的重复演示；多种偏好优化算法的展开对照留在第一册/第二册既有章节，本章聚焦 DPO 实战与 RM/PPO 的必要关系；架构专属 target modules、硬件配方等不写成跨模型通用说明。新稿保留可复核的 RM 单步、DPO 概率/loss 和 TRL 训练器示例。
- **证据口径修正**：重新检查 TRL v1.14.1 固定版本源码，修正 `loss_type="sigmoid"` 使用每条回答 token log-probability 求和、训练指标再跨样本平均的说明；该记录已同步至 `research/dpo-preference-objective-and-length-2026.md`。TRL 文档对指标的文字描述与此处源码聚合口径存在表述差异，正文按固定版本实际计算路径说明。
- **范围与验收**：正文 1,001 行，保留 5.1—5.7 连贯顺序；旧主题均按本章目标选择性处理，不要求逐项保留。36 个 fenced blocks / 72 个围栏标记配对（5 Python、25 math、3 text、2 JSON、1 Mermaid）；5 个 Python 均为静态审阅、未运行，新增 RM 单步代码标注 2026-11-16；行内单美元分隔符 230 个、数量成对，行内括号分隔符为 0；无 operatorname 写法、尖括号开头的数学下标或本地相对链接。没有执行书中代码、测试、模型、benchmark、构建或 PDF。静态复核包括围栏/公式/链接、术语与 TRL reduction 口径及 `git diff --check`；GitHub 页面公式/Mermaid 渲染与整册连续复读仍待最终门禁。

## chapter_scan：第 6 章《推理优化实战》

- **扫描范围与主线**：2026-10-29 完整通读 `chapters/06-推理优化实战.md`，沿单个请求的状态与资源生命周期检查 KV Cache、Prefill/Decode 指标、vLLM 分页和调度、量化、speculative decoding、长上下文压测，以及最终实验报告如何衔接。维持 6.1—6.7 原有结构，没有增删章节。
- **候选主题去向**：将每层历史长度不同的 sliding/chunked/hybrid cache 作为 KV 容量公式的必要边界；KVSET 的 LRU prefix-cache 工作集与活动请求 KV payload 分开说明，Tessera 的检索驱动非前缀复用只作有限模型范围的预印本路线。Prefill/Decode 分离并入 6.2.7 的调度取舍，说明 KV 传输与 TTFT/goodput 条件，不扩成独立架构章节。UltraQuant 作为 KV 低位宽实现与硬件 kernel 协同的受限案例；LongHarness Bench 补充 needle test 外的 harness 评测对象；AgentPerfBench 用固定 v1 链接锚定多轮负载思路。上述研究结果均按对应模型、trace、硬件或论文配置限定，不作为通用性能承诺。来源事实见 `research/kv-cache-capacity-paging-and-compression-2026.md`、`research/inference-serving-metrics-and-disaggregation-2026.md`、`research/decoding-sampling-and-test-time-compute-2026.md`、`research/long-context-harness-and-agent-memory-2026.md` 和 `research/weight-quantization-formats-and-ptq-2026.md`。
- **正文与接口修订**：KV Cache 公式扩展为按层 `T_l`、`n_{kv,l}`、`d_{h,l}` 求和，并区分 full-attention、滑窗层和额外 recurrent state；多请求混合层采用 request/layer 双重账本。将 Transformers 文档固定到 v5.17.0、PyTorch 固定到 2.14.0、vLLM 固定到 v0.32.0；TPOT/ITL 指标按 v0.32.0 精确定义。注明 vLLM Paged Attention 设计页是历史说明，不能代表该 release 的当前实现。修正 QLoRA 示例：固定 Qwen2.5-0.5B commit，去除训练加载中的 `device_map="auto"`，明确片段只演示 NF4 基座加载，并链接第四章完整 QLoRA 主讲。Transformers assisted decoding 示例改用有固定 Hub commit 的同词表 Pythia 160M/70M，并说明 v5.17 默认 assistant 路线与 UAD 异词表路径的边界。
- **篇幅与可复算例子**：将 6.1—6.7 作为七个核心教学单元统计解释性汉字，分别为 2,246、1,979、1,961、1,933、2,000、1,970、1,850，均达到 1,800 下限。保留/核对的例子包括：单请求 BF16、32 层 GQA、8K 历史约 1 GiB KV；sliding layer 的 `T_l = min(T, W_l)` 容量关系；TPOT 以 39 个间隔算得约 43.6 ms/token，而 40 token/2 s 的端到端速率是 20 token/s；block size 16、1000 token 需 63 blocks、尾块浪费 8 个位置；7B FP16/INT8/INT4 裸权重与 128-group scale 账本；假设的 speculative 一轮 41 ms、3.4 输出 token 得约 12.1 ms/token；needle 位置曲线与 4 轮累计 15,512 输入 token、相对首轮约 30.3 倍的负载账本。假设值均没有写成模型或服务实测结论。
- **代码状态**：11 个 Python 代码块分别位于 §6.1.5、§6.1.6、§6.2.4、§6.2.6、§6.3.5、§6.4.5、§6.4.6、§6.5.5、§6.5.6、§6.6.6 和 §6.6.9，全部标为“静态审阅，未运行（2026-10-29）”。没有可追溯到每块命令、输入、实际输出和边界的执行记录，因此不沿用 2026-10-20 的既有运行标签；只改动 §6.4.6 和 §6.5.6 的可执行正文，其余九块正文未改。没有运行书中代码、测试、模型或 benchmark。
- **公式、图表、链接与同步**：107 个围栏块共 214 个围栏标记（11 Python、63 math、31 text、2 Mermaid），均成对；行内 `\(` 与 `\)` 各 145 处、未转义 `$` 为 0，无控制字符、失效本地路径或动态 `main/latest/stable` API 链接。新增第四章 QLoRA 交叉链接目标存在。简介第 6 章摘要和目录问题/交付物仍匹配；PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY_EN_ZH 和 KNOWLEDGE_GRAPH 已有 KV/P-D、量化、speculative 与长上下文主题入口，本轮不重复增项。GitHub Mermaid 实际渲染仍留到本册最终门禁。
- **验收与交接**：静态围栏、单元篇幅、状态标记、公式分隔符、本地链接及控制字符检查通过；`git diff --check` 通过。未更新 PDF。第三册现为前 6 章已扫描精修、后 10 章待处理；下一章是第 7 章《RAG 项目实战》。本册最终连续复读、Mermaid 页面渲染和外链终审，以及前两册未完成的目标平台/外链门禁，继续留在各册收口阶段。

## chapter_scan：第 7 章《RAG 项目实战》

- **扫描范围与主线**：2026-10-30 完整通读 `chapters/07-rag项目实战.md`，检查文档解析与 chunk 生命周期、稀疏/稠密召回、权限过滤、rerank、引用支持、选择性回答和评估能否串成一条可追溯链。保留 7.1—7.4 原结构；简介摘要和目录交付仍准确，不改导航。
- **章级候选与去向**：保留 GraphRAG 为跨文档关系/全局摘要问题的候选，不默认加入主链；完整 GraphRAG 机制继续由第十七册承接。补足 BM25 + dense hybrid 在本项目的接入和评估动作，RRF 机制链接第二册第七章，避免重复主讲。澄清 post-filter 只能在可信检索边界内完成且无权文本、元数据和分数不得进入 reranker、生成、外部 trace 或用户输出。检索文档视为不可信输入；提示注入仅补入安全边界说明，不扩写为 Agent 执行章节。评估报告的敏感原文采集按项目现有 OpenTelemetry 候选收窄。
- **正文与接口修订**：以 Microsoft 官方 GraphRAG 方法页替代根仓库入口，具体描述其标准/fast 索引流程及成本差异，并将保真检查表述为工程要求。外部检索代码按 Sentence Transformers 6.2.0 和 faiss-cpu 1.15.1 的文档接口整理；固定 `BAAI/bge-small-zh-v1.5` revision `7999e1d3359715c523056ef9478215996d62a620` 与 `BAAI/bge-reranker-base` revision `2cfc18c9415c912f9d8155881c133215df768a70`，使用 query/document 编码 API，并声明仓库锁文件没有覆盖该依赖组合、示例没有联调。收窄 CrossEncoder 对条件排序的效果承诺；说明模型分数不等于支持概率、pair 截断和多证据覆盖需单独检查。修正 eval demo 对 q2 的 first-failure 描述。RAG trace 默认保留版本、chunk ID、状态和时延等定位字段，原文、完整检索结果、prompt 与答案只进入经授权隔离的脱敏离线材料。
- **篇幅与例子**：按四个核心单元统计解释性汉字（不计标题、代码/公式围栏、表格、状态行）：7.1 为 2,288，7.2 为 1,860，7.3 为 1,929，7.4 为 1,887；均达到 1,800 下限。保留 chunk size/overlap 的 320-token 手算、三题 MRR 从约 0.528 到 0.833、40/100 候选的粗略 cross-encoder 延迟账本；增加 200-query 的候选召回/条件重排分母对照、逐 claim 的 6,200 元与 5,000 元单笔阈值核验、100 题 A/B 成对回归例子，以及 20 题 Recall/accuracy/coverage/abstention 分子分母说明。所加数字均为假设输入或教学示例，不写成模型或服务测量。
- **代码状态**：16 个 Python 块均标记“静态审阅，未运行（2026-10-30）”。位置为 §7.1.1 Py1–2、§7.1.2 Py1、§7.1.5 Py1–2、§7.1.6 Py1、§7.1.8 Py1、§7.2.2 Py1、§7.2.3 Py1、§7.3.1 Py1、§7.3.2 Py1、§7.3.4 Py1、§7.3.5 Py1、§7.4.1 Py1、§7.4.2 Py1 和 §7.4.7 Py1。旧标签仅指向 `.tmp/llm-book-cpu-check`，`progress/` 没有逐块命令、输入、预期/实际结果和边界记录；按本轮约束不沿用旧“已验证”。本轮改变 §7.1.5 Py1–2 与 §7.2.2 Py1 的可执行语句；其余代码正文未改。没有运行书中代码、测试、模型或 benchmark。
- **证据、公式和链接**：新增 `research/rag-retrieval-and-evidence-pipeline-2026.md` 记录 GraphRAG、Sentence Transformers、两份 BGE 固定 revision、FAISS、RRF 与提示注入的官方来源及动态文档边界；在 `research/project-tooling-and-observability-2026.md` 更新 OpenTelemetry GenAI 内容采集入口。围栏共 69 块/138 个标记（16 Python、33 math、19 text、1 Mermaid），全部配对；行内 `\(` / `\)` 各 46 个，未转义 `$` 为 0；本地相对链接目标均存在。Mermaid 仅完成源码静态检查，GitHub 页面渲染未确认；旧外链尚未作全章完整终审。
- **同步、风险与交接**：`简介.md` 第 7 章摘要和 `目录.md` 的问题/交付物匹配；hybrid retrieval 的基础机制已由第二册覆盖，本轮不重复改 PAPERS/EXERCISES/术语表。外部 Sentence Transformers/FAISS 示例依赖未进入根 `uv.lock`，只有 API 静态审阅；不据此声称完整环境可运行。没有执行书中程序、测试、benchmark、模型或 PDF。下一步处理第三册第 8 章《Agent 项目实战》；第三册整册连续复读、Mermaid 页面渲染和外链终审仍留在最终门禁。

## chapter_scan：第 8 章《Agent 项目实战》

- **扫描范围与主线**：2026-10-30 完整通读 `chapters/08-agent项目实战.md`，检查 Tool Calling schema/执行回执、ReAct 状态循环、风险与授权、RAG+Agent 贯通案例能否沿同一任务形成可回放闭环。维持 8.1—8.4 结构，不新增章节；第十七册和第二十册的规划、runtime 与沙箱深讲仍作为外部交接。
- **候选主题与正文决定**：将当前 provider strict schema 规则纳入 8.1，并明确示例 `TOOLS` 是内部目录而非通用 API payload；分别说明调用 ID、请求 ID、幂等键和授权的职责。高风险确认改为服务端保存绑定用户/租户、规范化参数摘要、策略版本、有效期和一次性状态的 pending record，并在执行前原子消费；把此写法标作基于 OWASP transaction authorization 指南的工程实现，而非行业统一协议。补足 loop 的 durable state、无进展判定、恢复和终态语义；把 AgentDojo/ToolSandbox 的评测构造与本地安全率边界说清。收紧生产 trace 的原文采集，并修正审计 demo：审计副本字段脱敏，授权调用仍接收业务数据。
- **篇幅与数例**：8.1—8.4 作为四个核心教学单元，解释性汉字数分别为 1,896、1,915、1,939、1,912，均超过 1,800 下限。复核 6,000 与 5,000 元的阈值差额及严格/包含边界；多轮输入/输出累计为 2,650/245 token，总耗时按模型 1,360 ms 加工具 92 ms 得 1,452 ms；40 条样例零次未授权执行明确记为 0/40，不能外推到未测输入；10 条贯通样例分开计算覆盖率 6/10、可回答集正确率 6/6、资料不足拒答率 2/2 和越权尝试未执行 0/1。
- **代码状态**：12 个 Python 块统一标为“静态审阅，未运行（2026-10-30）”，位置为 §8.1.1 Py1、§8.1.2 Py1、§8.1.3 Py1、§8.1.5 Py1、§8.1.6 Py1、§8.2.1 Py1、§8.2.2 Py1、§8.2.4 Py1、§8.3.2 Py1、§8.3.3 Py1、§8.3.8 Py1、§8.4.3 Py1。旧标记只指向临时环境，没有 `progress/` 中可逐块复核的命令、输入、预期/实际输出和边界记录；本轮未执行程序、测试或 benchmark。改动 §8.1.5、§8.1.6 和 §8.3.8 三个代码块；其余代码正文未改。
- **证据、公式和链接**：新增 `research/agent-tool-calling-security-2026.md` 记录 OpenAI Function Calling strict mode、Anthropic 工具执行方差异、Agent Evals、ReAct、AgentDojo/ToolSandbox、OWASP transaction authorization/logging 与 OpenTelemetry GenAI spans 的事实和适用边界。围栏 57 块/114 个标记（12 Python、18 math、26 text、1 Mermaid）配对；行内 `\(` 与 `\)` 各 13 处，未转义 `$` 为 0；本地 Markdown 链接静态存在性检查通过，`git diff --check` 通过。新核验的 API/安全来源可支持对应段落；全章既有外链终审和 GitHub Mermaid 页面渲染留在最终门禁。
- **同步与交接**：简介第 8 章摘要和目录的职责/交付物仍匹配，无需改导航。PAPERS 已有 ReAct、AgentDojo、ToolSandbox 和 Function Calling 资料；EXERCISES、INTERVIEW_BANK 与术语表已有工具权限、确认、幂等和安全评估入口，本轮不重复添加。未更新 PDF。下一步处理第三册第 9 章《评估与 Debug 实战》；第三册连续复读、Mermaid 渲染和外链终审仍待最终收口。

## chapter_scan：第 9 章《评估与 Debug 实战》

- **扫描范围与主线**：2026-10-30 完整通读 `chapters/09-评估与debug实战.md`（1,786 行），审查 9.1 benchmark、9.2 数据污染/评测泄漏、9.3 事实性与 claim、9.4 训练不收敛、9.5 微调后能力退化和 9.6 人工评测是否围绕“固定协议—检查边界—定位失败—比较改动—解释证据”构成离线闭环；保留六节结构。简介和目录的第 9 章承诺与正文相符。
- **候选主题与范围**：把 Data Contamination Quiz、ICLR 2026 推理模型污染检测脆弱性研究和 BeyondBench 作为 9.2 的方法案例，明确词级扰动、算法生成题等只覆盖各自协议，不扩写成污染检测综述；项目级评估科学和 benchmark 详细分类继续交给第七册。PAPERS 已有这些论文的索引，EXERCISES 已覆盖 Wilson 小样本和污染探针，故不重复新增。
- **正文与算例修订**：补上 9.1 micro/macro 平均和 4/4 Wilson 区间手算；9.2 三字符 shingle 的 Jaccard 例子；9.3 claim 支持率与必需事实完整性的不同分母及权重例子；9.4 线性分类器首步损失/梯度/更新手算，并修正原 AND 标签数据在零初始化、全批次对称计算下权重梯度相消、不能学到正例的问题，改为由第一维特征决定的线性可分标签；9.5 用 `adapter_large` 算出成本调整效用 0.8805，并说明硬门槛优先于加权排序；9.6 补 Cohen's Kappa 的 10 样本边际计数演算。收紧 runner 和人工评测表的原始输入/输出保存建议，默认最小化普通日志中的内容。
- **篇幅与状态**：六个核心单元解释性汉字分别为 2,642、2,278、2,252、1,902、2,276、2,384，均达到 1,800 下限。全章 6 个 Python 块位于 §9.1.7、§9.2.4、§9.3.5、§9.4.5、§9.5.6 和 §9.6.8，统一标记“静态审阅，未运行（2026-10-30）”；旧的 2026-10-22 标签没有 `progress/` 内逐块命令、输入、结果及边界记录。本轮修改 §9.4.5 的数据标签和首步说明；没有运行书中代码、测试、模型或 benchmark。
- **证据、公式与链接**：截至 2026-10-30，OpenAI 官方下线页列出 Evals 于 10-31 转只读、dashboard/API 于 11-30 计划关闭；使用 PyTorch 2.15 Autograd 固定文档链接。污染检测近期论文的事实与适用边界见 `research/evaluation-design-uncertainty-and-contamination-2026.md`。围栏共 75 块/150 个标记（6 Python、39 math、29 text、1 Mermaid），全部配对；行内 `\(` / `\)` 各 133 处、未转义 `$` 为 0；本地相对链接目标存在。章节中的外部来源已逐项核对；Cohen 与 Krippendorff DOI 直连在浏览工具超时，但分别由 SAGE 与 Wiley 出版页确认元数据。`git diff --check` 通过。
- **同步与遗留项**：`简介.md` 和 `目录.md` 第 9 章摘要/交付承诺准确；现有 PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY_EN_ZH 与 KNOWLEDGE_GRAPH 已覆盖污染检测、评估统计、事实性及标注一致性，不重复堆条目。Mermaid 目标平台实际渲染、第三册整册连续复读和外链最终门禁仍待收口。没有更新 PDF；下一步处理第 10 章《Agent 轨迹与线上评估实战》。

## chapter_scan：第 10 章《Agent 轨迹与线上评估实战》

- **扫描范围与主线**：2026-10-30 完整通读 `chapters/10-agent轨迹与线上评估实战.md`（573 行），检查 Agent trace 的状态契约、first critical failure、回归样例，以及线上实验的随机化单元、SRM、功效、护栏、日志和灰度/回退是否组成从离线诊断到真实流量的交付链。保留 10.1、10.2 两段主结构，简介摘要与目录交付承诺仍准确。
- **章级候选与证据边界**：将 Agentic CLEAR 更新为 ACL 2026 System Demonstrations 正式论文，不再称预印本；保留其 system/trace/node 诊断作为研究方法，并把论文结果限制在原 benchmark、agent 与 judge 协议。AgentRx 的 115 条人工标注失败轨迹按 Microsoft Research 发布方材料描述，不当成本地复现。OpenAI Agent Evals 当前 trace-to-dataset 工作流及托管 Evals 生命周期按访问日核验；新事实与来源边界见 `research/agent-trace-diagnosis-and-online-evaluation-2026.md` 和 `research/evaluation-design-uncertainty-and-contamination-2026.md`。没有发现需要改变第三册章节边界的新主题。
- **正文与例子修订**：trace 字段改为伪名化 ID、状态摘要和受控证据引用；完整提示词、生成内容、检索文本与敏感工具参数不默认写入普通 trace，并补 OpenTelemetry 动态规范的敏感数据边界。补 SRM 卡方自由度、CUPED 单协变量调整及 r=0.6 时 36% 理想方差降幅的假设推演；明确按分配单元保留主分析分母、独立记录 exposure。移除缺少样本定义的 580/890 配对胜率；改为报告两组安全事件 Wilson 区间，并说明正式非劣效要估计组间安全风险差。补算 12,000/12,080 分流的期望数和卡方统计量，以及 77.00% 到约 79.39% 的解决率差和 95% 区间；练习改为三道可复算题。
- **篇幅与例子**：按 10.1 和 10.2 两个核心教学单元统计解释性汉字（不计标题、代码/公式围栏、表格、状态行及引用清单），分别为 1,901 和 3,333，均超过 1,800 下限。数例还覆盖 20 条地址任务的提前提议/拦截/副作用分母、每组 6,736 的样本量估算、预算/护栏结果和低频安全事件的不确定性。
- **代码状态**：全章 1 个 Python 块，位于 §10.2.6，统一标记“静态审阅，未运行（2026-10-30）”。原 2026-10-22“已验证”没有可追溯至该块的命令、输入、预期/实际输出和边界记录；本轮依用户约束不运行书中程序、测试、模型或 benchmark。代码改动仅替换无样本定义的配对胜率字段，改为调用 Wilson 计算 Control/Treatment 安全事件区间；其余统计逻辑只做静态审阅。
- **公式、图示与链接**：全章 19 个 fenced block 共 38 个围栏标记（1 Python、8 math、8 text、2 Mermaid），配对闭合；行内数学分隔符分别为 \( 与 \) 各 48 处，未转义美元符号 0。两个 Mermaid 图只做源码静态审阅，GitHub 页面渲染留待最终门禁；章节不含本地 Markdown 文件链接。OpenAI、ACL Anthology、Microsoft Research、OpenTelemetry、NIST、CUPED 作者页和 Always Valid Inference 等来源已核对，托管 Evals 时间线截至 2026-10-30。
- **配套同步与遗留项**：`PAPERS.md` 将 Agentic CLEAR 链接改至 ACL Anthology；`EXERCISES.md` 增加 CUPED 方差比练习并细化 10.1/10.2 入口；`KNOWLEDGE_GRAPH.md` 增加随机化单元、SRM、CUPED、序贯推断和回退链。术语表已有 SRM/CUPED，题库已有 SRM、线上灰度与 Agent trace，不重复新增。未改简介和目录，无需回填 `plan.md`。整册连续复读、两幅 Mermaid 目标平台渲染及全章外链终审仍待最终门禁；不更新 PDF。
- **下一步**：进入第三册第 11 章《图文理解与检索实战》；全套书精修 goal 保持 active。

## chapter_scan：第 11 章《图文理解与检索实战》

- **扫描范围与定位**：2026-10-30 完整通读 `chapters/11-图文理解与检索实战.md` 原稿（1,143 行），检查共享图文表征、对比目标、VLM 输入预算、instruction 数据治理和图文 RAG 是否围绕“把图片转成可检索、可核验、受权限约束的证据”形成连续选读支线。精修后 1,150 行。简介摘要和目录承诺准确，章节仍限于图文理解与检索，不扩展到多模态生成专题。
- **候选与取舍**：把 SigLIP 的 pairwise sigmoid 目标及 SigLIP 2 的组合训练 recipe 纳入 §11.1.2，作为 CLIP 对照；配方结果不归因于单一损失，也不另增独立讲次。已有 SigLIP 论文、术语、面试题和练习索引，不重复添加；其他新模态、完整评测与系统交付仍交给第十四章和第十五册。
- **正文与算例修订**：补 CLIP 与 SigLIP 的分母/标签机制差异及 `x=1` 的单 pair 正负损失手算（约 0.313/1.313）；将 §11.3.3 改为 Qwen2.5-VL 模型专属接口例，按 `Qwen2_5_VLForConditionalGeneration` 与 `AutoProcessor.apply_chat_template()` 编排消息、图片、输入张量及新 token 解码，并说明公开图像 URL、依赖、revision 与设备边界。原稿五个“已验证”状态在 `progress/` 无逐块命令、输入、结果和边界记录，故全部六个代码块统一为“静态审阅，未运行（2026-10-30）”；不运行书中代码、模型、测试或 benchmark。
- **篇幅与例子**：五个核心单元解释性汉字分别为 2,182、1,842、1,939、1,990、1,916，均达到 1,800 下限。保留 CLIP 4×4 排名与 Recall@K/MRR、2×2 双向损失、图片 token 预算、按图片分组切分、assistant-only mask、图文分数融合及 ACL 过滤的可复算例子；数字均为手造教学输入，不是模型或服务测量。
- **代码、公式与图示**：§11.1.3、§11.2.2、§11.3.3、§11.3.5、§11.4.5、§11.5.4 共 6 个 Python 块，均只做静态审阅；Qwen 接口示例的可执行语句已改变，本轮未执行。全章 46 个 fenced block／92 个标记（6 Python、27 math、11 text、2 Mermaid），配对闭合；行内 `\\(` 与 `\\)` 各 7 处、未转义 `$` 为 0；2 个 Mermaid 仅静态检查，GitHub 页面渲染仍待最终门禁。公式定义域、损失分母、patch 近似、样本切分与分数归一化均按输入条件复核，`git diff --check` 通过。
- **来源与链接**：SigLIP/SigLIP 2 原论文、CLIP、LLaVA、BLIP-2、RAG、FAISS 官方资料、Azure AI Search 权限过滤，以及 Transformers CLIP、图像文本任务、多模态模板和 Qwen2.5-VL 文档均于 2026-10-30 核对；动态 API 的版本边界补入 `research/multimodal-systems-and-evaluation-2026.md`，其余训练目标证据见 `research/clip-contrastive-learning-and-evaluation-2026.md`，FAISS 来源见 `research/rag-retrieval-and-evidence-pipeline-2026.md`。17 个 Markdown 链接中 15 个外链引用对应 12 个唯一目标、2 个本地链接；本地目标均存在，所有外部目标均已打开核对。
- **同步、遗留项与交接**：新增知识图谱中的 CLIP/SigLIP/SigLIP 2 对照链；PAPERS、EXERCISES、INTERVIEW_BANK 和术语表已有相应入口，不重复增项。简介和目录无需修改。全章代码保持未运行；两幅 Mermaid 的目标平台渲染、第三册整册连续复读仍待后续最终门禁。不更新 PDF。下一步进入第 12 章《Diffusion 数值实验》，全套书精修 goal 保持 active。

## chapter_scan：第 12 章《Diffusion 数值实验》

- **扫描范围与主线**：2026-10-30 完整通读 `chapters/12-diffusion数值实验.md`，检查前向 Markov 加噪、闭式 `x_t`、噪声预测、DDPM 后验均值/方差、CFG、少步采样和生成评估能否组成一条“可手算状态变化—模型预测—采样/验收”的选读主线。章级扫描结合 DDPM/CFG/少步采样论文、Diffusers v0.35.1 固定版本源码、Diffusers scheduler 文档、图像生成评估原论文及截至 2026-10-30 的扩散蒸馏候选；保留一篇章和 12.1.1—12.1.7 顺序，不新增/拆分章节。
- **候选去向**：把训练末端非零 terminal SNR 与从纯 Gaussian 初始化之间的潜在边界不匹配补入 12.1.1，并限定为 schedule/checkpoint 相关；在 12.1.5 统一原论文 `w` 与差分尺度 `s=1+w`，补空条件训练、相同 `x_t/t` 的双分支比较、score 差值解释，以及 Diffusers v0.35.1 的 `guidance_scale=1`、negative prompt 分支和拼 batch 语义。DDIM/DPM-Solver 归为更改采样路径/ODE 求解；渐进蒸馏归为要训练学生模型的路线，不混作同一类 scheduler。新增的 2026 年 data-free multistep distillation 预印本仅核对摘要，留作专题候选，不据摘要扩写本章或宣称通用性能；第十五册仍承担生成系统主讲。
- **正文修订与数值核查**：修正两步手算中反向均值遗漏除以 `sqrt(alpha_2)` 的错误：分子约 `0.7729`，均值约 `0.8642`；后验方差 `0.0714`、标准差约 `0.2672` 保持不变。增加 `s=1.5` 与论文 `w=0.5` 均得 `0.8` 的 CFG 对照，并用 `s=0.5` 的 `0.4` 区分插值和外推。五步代码示例的预期累计 alpha、末步 `x_t`、反向均值和方差按公式给出为 `[0.990000, 0.970200, 0.941094, 0.903450, 0.858278]`、`[1.9658,-0.4518]`、`[1.9760,-0.3000]` 和 `0.03406`，明确是手算预期值。
- **篇幅与教学例子**：主单元 12.1 解释性汉字 4,481；12.1.5 CFG 独立核心单元 1,839；12.1.6 生成评估辅助单元 1,033，分别达到 1,800/1,800/1,000 下限。原两步 schedule `x_2≈0.5839` 的信号/噪声分量、噪声预测误差传至均值的 `-0.0845` 偏移和 `SNR≈2.57` 均保留。评估补入 20 prompts × 3 seeds 的 60 张教学样例：主体 51/60、属性 44/60、关系 30/60、全条件 24/60，并说明 seed 共享 prompt 的分母边界。
- **代码、公式、围栏与链接**：唯一 Python 块状态改为“静态审阅，未运行（2026-10-30）”；源文件原“已验证”所指逐块命令、输入、预期/实际结果与边界记录在 `progress/` 中不可追溯，因此不沿用旧状态。本轮未运行书中代码、测试、模型或 benchmark。19 个 fenced block/38 个标记（1 Python、16 math、1 text、1 Mermaid）配对；行内 `\(` / `\)` 各 46 处、未转义 `$` 为 0。20 条 Markdown 链接含 18 个外链引用（14 个唯一目标）与 2 个本地链接；两条本地链接目标存在。`git diff --check` 通过；Mermaid 源码静态检查，GitHub 页面渲染仍待最终门禁。
- **研究与配套同步**：`research/diffusion-generation-guidance-and-sampling-2026.md` 补充 FID、Inception Score、CLIPScore 原论文的定义/适用边界，CFG score 解释、Diffusers negative prompt 实现、progressive/guided distillation 来源以及 2026 预印本的摘要级证据状态。简介第 12 章摘要和目录功能/交接仍匹配；PAPERS 已有 CFG、DDIM、DPM-Solver、Progressive Distillation、FID 与文生图 benchmark；EXERCISES、INTERVIEW_BANK、GLOSSARY_EN_ZH 和 KNOWLEDGE_GRAPH 已覆盖对应手算与概念，本轮不重复增项，不改 `plan.md`。
- **遗留项与交接**：没有更新 PDF，也没有执行书中代码、模型、测试或 benchmark。第三册整册连续复读、相关 Mermaid 的 GitHub 页面渲染和全章外链最终门禁仍待收口。下一步进入第 14 章《多模态评估与交付》，全套书精修 goal 保持 active。

## chapter_scan：第 13 章《语音识别与问答实战》

- **扫描范围与主线**：2026-10-30 完整通读 `chapters/13-语音识别与问答实战.md`（466 行），检查 Whisper 识别、片段/时间戳、文本规范化、证据检索、回答及高风险动作确认是否组成一条可定位错误来源的链路。保留 13.1/13.2 两段结构；简介和目录承诺仍与正文匹配。
- **章级候选与边界**：按 Whisper 官方论文、v20250625 实现、Transformers v5.17.0 文档和 pipeline 源码复核长音频路径。明确模型自带的时间戳驱动长音频解码与通用 `chunk_length_s` 外部分块不是同一策略；外部分块/VAD 与重叠窗口保留为自定义方案案例，不据此描述 Whisper 默认行为。mu-bench 的 UER 和参考引导规范化已在第二册语音评估章节主讲，本章链接复用，不重复扩写或新增讲次。
- **正文修订与例子**：补全 log-Mel 信息流公式的变量说明、Whisper 长音频调用边界、输出 `text/chunks/timestamp` 数据形状、`ffmpeg` 前置条件和版本/revision；CER/WER 先定义中文名、分子分母及规范化边界。检查器同时接受 pipeline 的 `timestamp=(start,end)` 与归一化后的 `start/end` 字段，拒绝 `None` 未知时间；补充预期输出。为语音问答链路逐一定义 ASR 函数、片段集合、规范化函数、检索分数、TopK、上下文构造和生成函数；确认阈值使用关键槽位校准概率，并说明高风险任务仍须确认。普通 trace 改用脱敏摘要和受控内容引用。
- **篇幅与数值例子**：13.1 与 13.2 两个核心单元解释性汉字数分别为 2,469 和 2,302，均达到 1,800 下限。包括中文 CER 删除 1/11、英文 WER 删除 1/6、95 秒音频采用自定义 30 秒窗口/5 秒重叠时共处理 110 秒并产生 15 秒重复覆盖、18/95 秒的教学 RTF、关键否定词漏识别以及问答各层成功概率与成本账本；合成输入和假设不作实际服务结论。
- **代码、公式、围栏与链接**：3 个 Python 块均标为“静态审阅，未运行（2026-10-30）”，不执行书中代码、测试、模型、benchmark 或 PDF。全章 23 个 fenced block/46 个围栏标记（3 Python、10 math、8 text、2 Mermaid）配对；72 个单美元行内数学分隔符成对，括号分隔符数量均为 0；11 个 Markdown 链接中 8 个外链与 3 个本地目标均核对存在，`git diff --check` 通过。Mermaid 仅作源码静态检查，GitHub 页面渲染留待最终门禁。
- **研究与配套同步**：Whisper 论文、tagged release、固定 Transformers 文档/源码和模型 revision 的来源、阅读范围与适用边界记入 `research/speech-recognition-generation-and-realtime-2026.md`。简介摘要和目录入口准确；PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY_EN_ZH、KNOWLEDGE_GRAPH 已覆盖 ASR、WER/CER、语音链路评估及其概念，本轮不重复增项。
- **遗留项与交接**：无 PDF 更新；未运行任何书中程序或模型。第三册整册连续复读、Mermaid 目标平台渲染及最终外链门禁仍待收口。下一步处理第 14 章《多模态评估与交付》；全套书精修 goal 保持 active。


## chapter_scan：第 14 章《多模态评估与交付》

- **扫描范围与定位**：2026-10-30 完整通读并精修 chapters/14-多模态评估与交付.md，检查统一样本 schema、任务指标、版本回归、媒体处理链、权限/失败状态、成本和发布回滚能否组成同一条可复核交付链。保留 14.1/14.2 两个核心单元及原有章节边界；简介摘要和目录承诺准确，无需更改。
- **候选主题及去向**：把视频/音视频时间区间 IoU 纳入任务指标，加入 10—20 秒 gold、15—25 秒 prediction 的手算；解释 IoU 不证明事件语义正确或证据支撑，并链接第二册第十三章和 RVG-Bench 预印本。VHELM 作为多维 VLM 评估方法入口补入来源说明；不扩成 benchmark 目录，不更改章节结构。RVG-Bench 结果严格限制在论文数据、模型快照、输入采样和评分协议。
- **正文修订**：定义样本数 N、切片集合 E_c、prediction_i、m_j 和 pass_i；明确缺失输出、超时、解析失败和拒答的分母语义，并区分逐样本宏平均与语料级 WER/CER 微平均。扩充 F1、VLM、OCR、ASR、WER/CER、VAD、embedding、groundedness、LLM judge、win/loss、group bootstrap 和人工裁决等术语。增加跨感知、结构化证据、可选权限过滤检索、生成/动作门、安全结果、脱敏 trace、离线评估及发布回滚的 Mermaid 数据流图。定义 QPS_peak、A、V、T_in、T_out、TPS_demand、各阶段延迟和成功任务成本的单位及端点；说明音频 RTF、媒体 decode 与 LLM token decode 的区别，延迟只按串行关键路径求和。
- **代码、公式与图示**：14.1.3 评估器静态修改，补上 None/缺失、timeout、parse_error、missing_prediction 和 task-defined refusal 状态；缺少预测字段计解析失败，所有样本仍计入规则通过率分母。手算 ASR 示例为 2 次编辑/9 字符，CER=0.2222，预期摘要和 bad_cases 明列。14.2.5 代码未重跑，旧已验证状态无逐块可追溯记录，改为静态审阅、未运行；手算预期为 missing=[]、voiceqa_pass 相对变化 0.2459、asr_cer 相对降低 0.3889、bad_case_ready=True。两段 Python 代码均未执行。14.1/14.2 两个核心单元解释性汉字数为 2,893、2,621。全章 48 个围栏标记配对（24 个 fenced blocks，2 个 Python、2 个 Mermaid）；82 个未转义美元分隔符成对，无括号形式的数学分隔符；11 个 Markdown 链接的本地目标均存在。Mermaid 仅静态检查，GitHub 页面渲染待最终门禁。
- **证据与配套同步**：research/multimodal-systems-and-evaluation-2026.md 增加 VHELM v2 摘要的九个维度、标准化范围与论文快照边界；research/project-tooling-and-observability-2026.md 增加 2026-10-30 OpenTelemetry GenAI registry 复核。PAPERS.md 近期多模态评测入口补入 VHELM；EXERCISES.md 已有多模态交付和 temporal IoU 练习，GLOSSARY_EN_ZH.md 已有 temporal grounding/mIoU，其他配套不需重复增项。NIST 当前说明和 OpenTelemetry 动态 convention 在正文链接并限定证据范围。
- **门禁与下一步**：简介、目录未改；全仓 git diff --check 通过。章节围栏、数学分隔符和本地链接静态检查通过。未运行书中代码、测试、模型、benchmark 或 PDF，也没有更新 PDF。第三册完整连续复读、所有 Mermaid 目标平台渲染和全章外链最终复核留到收口阶段。下一步处理第 15 章《作品集实战》。

## chapter_scan：第 15 章《作品集实战》

- **扫描范围与定位**：2026-10-30 完整通读并精修 `chapters/15-作品集实战.md`，检查如何从工程主张串起固定数据与运行条件、基线比较、失败归因、复现记录和简历表达。保留单一核心单元 15.1 与既有章节顺序；第 3—14 章仍是各技术主题的完整主讲，不在作品集章重复展开。
- **章级扫描与候选去向**：核对运行清单、Model Card/实验追踪、逐样本对照、切片退化、失败轨迹和隐私边界。将与总分相容的逐题配对表作为数值例子的必要补充，并明确它是教学构造、不能由两个汇总分反推真实 win/loss。保留现有 16 章骨架，不新增技术主题；Model Cards、MLflow、NIST 去标识化和 OpenTelemetry trace 敏感内容来源及事实边界见对应官方文档与 `research/project-tooling-and-observability-2026.md`。
- **正文修订**：统一 RAG、SFT/LoRA、DPO、推理服务、SLO、manifest、revision、processor、chat template、GPU、kernel、artifact 等术语。补充 NIST 对带密钥/无密钥哈希、假名化与再识别风险的限定；明确完整失败轨迹应在授权受控环境留存，公开作品集使用合成或经过风险审查的脱敏材料。删除失效的“第 15.4 节”练习引用。
- **篇幅与可复算例子**：核心单元 15.1 解释性正文静态计数 3,436 个汉字，超过 1,800 下限。教学配对表逐行加总为旧政策 16/20 对 12/20、普通问题 56/80 对 68/80、总体 72/100 对 80/100；旧政策切片下降 20 个百分点，违反 75% 发布门槛。成本手算为基线 $1.80/72=$0.025/成功任务、候选 $2.40/80=$0.030/成功任务，即候选单位成功成本上升 20%；明确不是显著性或线上收益结论。
- **图示、代码、公式与链接**：全章没有程序代码或公式块；1 个 Mermaid 流程图、2 个配对围栏标记，源码静态检查通过，GitHub 页面渲染仍待最终门禁。13 个 Markdown 链接中 9 个本地目标均存在，4 个外部官方来源页（Hugging Face、MLflow、NIST、OpenTelemetry）已核对；未转义美元数学分隔符和括号数学分隔符均为 0。
- **证据与配套同步**：`research/project-tooling-and-observability-2026.md` 增补 NIST SP 800-188 §4.3.1–4.3.2 的直接标识符哈希边界，并记载指南的政府数据治理适用范围；既有 Model Card/MLflow 记录保留 latest 文档动态版本边界。`PAPERS.md` 增加 NIST SP 800-188 入口，`GLOSSARY_EN_ZH.md` 增加 De-identification 与 Pseudonymization。简介第 15 章摘要和目录入口仍准确；章节自带的练习已覆盖本章实践，不重复写入 EXERCISES/INTERVIEW_BANK，现有知识图谱已有数据隐私与 trace 治理入口。
- **门禁与下一步**：全仓 `git diff --check` 通过；未运行书中程序、自动化测试、模型、benchmark 或 PDF，也没有更新 PDF。第三册连续复读、Mermaid 目标平台渲染和其他章节外链终审仍待收口。下一步处理第 16 章《技术表达与系统设计实战》。

## chapter_scan：第 16 章《技术表达与系统设计实战》

- **扫描范围与定位**：2026-10-30 完整通读 `chapters/16-技术表达与系统设计实战.md`，围绕“把技术判断讲成可复核的证据链”检查项目经历答辩、开放目标到系统方案、论文主张到验证结论三个核心单元的因果与交接。保留 16.1—16.3 结构；简介和目录摘要准确，无需改动。
- **候选主题与取舍**：复核 Attention/KV Cache 的容量账本、指标归因与随机对照、硬约束下的方案选择、系统容量/SLO/权限/回滚，以及 Transformer、LoRA、DPO、FlashAttention、RAG、Mamba 的代表机制和验证方式。没有发现需新增讲次或改动全册骨架的主题；完整技术主讲仍由第 2、6—10 章及相应专题册承担，本章保留答辩迁移与证据边界。
- **正文精修**：补齐候选/基线指标符号及指标方向约定，说明单变量变化不单独证明因果；定义硬约束函数和阈值符号，补全项目到第 7—10 章的导航；为论文引用固定原论文版本，并明确作者实验与目标系统测量的区别。补充 NIST AI RMF 自愿使用、1.0 正在修订且不构成法规合规证明的边界。
- **篇幅与可复算例子**：16.1、16.2、16.3 解释性汉字静态计数分别为 2,259、3,328、3,145，均超过核心单元 1,800 下限。保留 KV Cache `512 MiB × 24 = 12 GiB`、小模型路由成本降低约 54.17%、容量估算的 QPS/Little’s Law/KV 账本，以及 FlashAttention 在线 softmax、LoRA 参数量和论文卡片覆盖率的手算；合成数据明确只作教学推演。
- **代码、公式、图示与链接**：5 个 Python 示例均静态审阅、标为“静态审阅，未运行（2026-10-30）”；原“已验证”没有本章 `progress/` 中逐块命令、输入、结果和边界记录，本轮不运行示例。全章 56 个 fenced blocks／112 个围栏标记配对（5 Python、2 Mermaid、37 math、12 text）；2 幅 Mermaid 做源码静态检查，GitHub 实际渲染留待整册门禁。26 个 Markdown 链接含 11 个外链和 15 个本地目标，本地目标全部存在；`git diff --check` 通过。
- **来源与同步**：6 篇原论文链接固定到对应 arXiv 版本；vLLM 动态指标页于 2026-10-30 复核，NIST AI RMF 当前状态及适用边界写入 `research/project-tooling-and-observability-2026.md`。`PAPERS.md` 已有 AI RMF、Google SRE 和 vLLM 指标入口，不重复增项；本章引用的本地章节链接均存在。简介和目录无需修改。
- **遗留项与交接**：没有更新 PDF，也没有执行书中代码、测试、模型或 benchmark。第三册整册连续复读、全部 Mermaid 的 GitHub 页面渲染和全章外链终审仍待最终门禁；未完成前不将第三册标记完成。全套书精修继续进行。

## 全册重写轮次（2026-11-16）

### 本轮目标与当前架构判断

- **任务**：按 2026-11-16 的最新版写作规范重新审视并重写第三册正文。目标读者是有深度学习与 Python 基础、尚未系统理解大模型工程流程的人；全册要把他们从“能运行片段”带到“能解释机制、检查结果、定位失败并交付证据”。
- **已核对的书册主线**：01 张量与更新 → 02 Transformer 信息流 → 03 小 GPT 闭环 → 04 监督适配 → 05 偏好适配 → 06 推理预算 → 07 检索证据 → 08 受控动作 → 09 离线评估与诊断 → 10 轨迹和线上验证 → 11—14 多模态选读项目 → 15 成果归档 → 16 技术说明与设计答辩。该顺序能回答“实现什么、学习目标怎样落到代码、怎样测出效果并留下证据”，暂不改编号。
- **本轮重写原则**：先按读者会遇到的具体问题开篇，再逐步给对象、术语和必要假设；把公式、代码和算例放在它们实际回答的问题之后。每章继续承载多个相连知识点，覆盖图逐项说明核心、支撑、进阶和迁出内容；旧内容不按行数配额保留，也不以“讲主线”为由删掉履行章节承诺所需的机制、推导、实现、评估和边界。跨章共用一个项目/数据/记录对象时，明确交接关系，减少每章重新起一个互不相干的例子。
- **范围审视**：第 6 章主题多但共同受“延迟、吞吐、显存、质量和任务负载”约束，先尝试用同一服务预算主线重排；若正文复读后仍需要并列讲授多个独立优化目标，再拆成连续章节。第 10 章的轨迹诊断与线上随机实验属于从离线到线上验收的连续证据链，重点补足轨迹诊断的解释与案例，暂不拆分。第 11、12、13、14 章各自为不同模态项目，保持选读边界；第 15 章篇幅和交付流程明显需要扩成多个读者可执行阶段；是否另增章节在完成定向扫描后决定，不为凑长度扩写。
- **覆盖与篇幅复核**：旧进度中按固定 1,800 汉字计数作为完成依据的写法已过时。本轮逐章记录必要知识覆盖与单点解释深度；短章重点检查承诺中的步骤、失败路径和读者产物是否缺失，长章重点检查多条主线、重复和专家细节是否打断初学者路径。
- **研究与事实边界**：既有 `research/` 中的研究记录仅作为来源线索和可复用证据，不能替代本轮针对正文改写处的定向核对。涉及时效 API、论文状态或规范时，先阅读官方文档/固定版本源码/原始论文，并将新知识及证据写入 `research/`；本文件只记本册去向、取舍、修改与状态。没有变化的旧代码不执行；新改代码最多做静态审阅并如实标注，除非用户另行要求运行。
- **执行边界**：不运行书中程序、测试、模型、benchmark 或 PDF 构建；不更新 PDF。每章完成后核对公式语法、代码围栏、图表源码、必要本地链接与正文—目录—简介一致性。图表平台渲染和外链最终核验留到末轮，并记录尚未完成项。
- **进度状态**：册级定位、目录与旧知识去向完成初审；第 1—11 章已完成本轮重写，第 12—16 章待处理。上一轮的 `chapter_scan` 是历史证据，不等同于本轮的新正文检查。每完成一章，在该章记录覆盖、保留/合并/迁移去向、解释深度、算例、公式/代码状态、同步结果和未解风险；全部章节完成后更新简介/目录并沿主路径整册复读。

### 本轮第 1 章复写

- **章节结果与主线**：本章帮助读者看懂一次参数更新：模型如何给出预测，损失如何转成梯度，梯度如何沿计算图回传，优化器和学习率怎样决定参数移动。章节承诺和目录摘要仍准确，不增章；第 2 章接续到 Transformer 的张量信息流。
- **覆盖图与知识去向**：核心内容保留回归预测/MSE/梯度、分类 logits 与 MLP、交叉熵及 [B,T,V] token loss、反向传播、SGD/Momentum/Adam/AdamW、训练循环、梯度累积、混合精度、学习率调度和 checkpoint。支撑内容保留广播错误、train/eval 与 inference_mode、有限差分、PPL 和调度器恢复诊断；XOR 与阈值选择作为表达能力/决策边界的例子保留，不另开评估讲次。没有删除需要支撑本章结果的主题；生产训练、Serving 性能和系统级评估仍在后续章节。
- **正文修改与深度**：重写开篇问题和六段承接，先用具体的四条回归样本连接形状、损失、梯度和参数变化。新增逐样本梯度贡献表，使读者可从四行误差复算到 dw=-35、db=-12；更新该代码学习率为 0.01，并手算更新后 MSE 从 41 降至 28.4532。补一阶/二阶矩和偏差修正的 Adam 单参数算例（theta_0=1、g_1=2），点明 Adam 首步约移动 0.1、同学习率 SGD 移动 0.2。扩展 AMP 说明：loss scaling 的数值目的、autocast/反向/裁剪/step 顺序、GradScaler 对非有限梯度的处理，以及 PyTorch 2.15 中旧 torch.cuda.amp.* / torch.cpu.amp.* 入口的弃用状态。
- **公式、代码和来源**：全章行内公式统一为 `$...$`，无 `\(...\)`；修订首个参数更新示例代码，手算输出由上表复核，状态标为“静态审阅，未运行（2026-11-16）”。其余 22 个 Python 代码块未改，沿用已有 2026-10-19、Python 3.12.9/PyTorch 2.12.0+cpu 验证状态；未运行书中代码。代码状态中删除内部 `.tmp` 路径。PyTorch 2.15 Autograd、CrossEntropyLoss、inference_mode、logsumexp、optimizer/scheduler 与 AMP 官方链接固定到 v2.15；混合精度接口和语义来源见 `research/project-tooling-and-observability-2026.md` 新增条目。相关页面于本轮阅读核对。
- **静态检查与未完成项**：23 个 Python 块对应 23 条状态（22 已验证沿用、1 静态审阅）；围栏标记计数为偶数，行内美元符号计数 346（成对）；未发现行内括号式公式分隔、内部临时路径、`operatorname` 或错误下标；`git diff --check` 通过。代码未执行，GitHub 页面渲染未确认；本册其他章节尚未开始本轮重写。

### 本轮第 2 章拆分与逐篇重写（2026-11-16）

- **原章审读**：完整复读原 `chapters/02-transformer组件实战.md`（2,137 行）。它把输入表示、attention/mask、block 结构和现代 decoder 组件都放在同一章。知识点依赖关系是成立的，但读者要连续跨过四类学习任务；原稿另有重复实现、重复解释和位置机制/attention 交接不够清楚的问题。
- **结构决定（复核后调整）**：第 2 章现拆为四篇，保留全册后续章节编号不变：上篇讲 tokenizer id、embedding、加性位置、padding 与输入输出权重约定；中篇讲 Q/K/V、scaled dot-product、多头、交叉注意力和各类 mask；下篇集中讲 attention 与 FFN 的分工、residual、Pre-LN/Post-LN 和基础 block；续篇讲 RoPE、RMSNorm、SwiGLU 及现代 block 组合。原下篇接近千行，承载基础 block、位置机制和现代 FFN/归一化三类讲解；拆出续篇后，基础 block 和现代组件各有独立读者结果，交接仍沿 `[B,T,D]` 不变的外部接口展开。四篇分别交付 `[B,T,D]` 输入契约、可见性受限的上下文计算、可堆叠的基础 block、可审计的现代 decoder 组件组合，并按此顺序接入小 GPT。文件分别为 `02-transformer组件实战-1-输入表示.md`、`02-transformer组件实战-2-attention与mask.md`、`02-transformer组件实战-3-decoder-block.md`、`02-transformer组件实战-4-modern-decoder.md`；数字后缀保持 PDF 构建脚本自然排序。简介、目录和第 16 章的组件回链已同步；同一章节分篇，不增加全册编号。
- **覆盖图**：上篇保留 token id 非数值特征、查表/one-hot 等价、Embedding 参数量与稀疏梯度、`padding_idx`/attention mask/loss mask 分工、learned 与 sinusoidal position、weight tying、左右 padding、position offset 和 tokenizer/checkpoint 配套。中篇保留 score 缩放、softmax key 轴、value 汇聚、多头 reshape、自注意力/交叉注意力、参数量/复杂度、GQA/MQA 引导、因果/填充/segment/prefix mask、全屏蔽行、PyTorch 布尔语义及局部回归检查。下篇保留 attention/FFN 职责、Pre-LN/Post-LN、LayerNorm/Dropout、残差梯度路径、基础 block 的参数/计算量与诊断；续篇保留 RoPE 的相对位置推导/布局/offset、RMSNorm 低精度处理、SwiGLU 参数账本、现代组件组合和组件级证据边界。没有因拆分移除核心知识点。
- **上篇重写**：开篇改为“整数 id 如何成为带位置线索的向量”，逐段解释从 tokenizer 到 `[B,T,D]` 的数据路径。明确 embedding 行不是语义标签，position ids 与 padding/cache 约定要一致，`padding_idx` 不替代 attention/loss mask。将重复的第二个 `InputEmbedding` 完整实现改为输入审计表，并补一个独立的 `padding_idx` 梯度契约短例；原有可复算向量例子保留。
- **中、下、续篇修改状态**：中篇已重写开篇、Q/K/V 角色和 attention 例子的顺序，将两 key/两 value 手算放到公式之后、实现之前；合并重复的权重解释，压缩 MHA 测试中的重复实现，整并重复的 GQA 说明；加入 PyTorch 非方形 causal mask 的左上对齐及 KV Cache 绝对位置边界。下篇聚焦基础 block，保留 Pre/Post-LN、残差梯度与数值例子、参数量和诊断路径；将重复的职责、归一化与架构对比改成两 token 数值例子、残差推演和症状—检查表。续篇承接 block 骨架，讲 RoPE、RMSNorm、SwiGLU 的定义、推导、数值例子、实现边界和现代组件组合；静态审阅时修复 SwiGLU 中间维取整和 RoPEAttention 零 head 数检查，并在组合示例中补齐 RoPE 缓存配置校验。四篇已连读复核，未发现需再拆分或删除的核心主题。
- **公式、代码与研究**：四篇行内数学分隔符分别计数为 74、224、54、114，均成对；无 `\(...\)`；fenced block 标记分别为 56、62、36、50，均成对；Python 代码块及相邻状态分别为 7/7、8/8、1/1、6/6；共 4 个 Mermaid 图块。22 个 Python 块均标为静态审阅、未运行；MHA 局部测试改为复用前文类的片段，SwiGLU 取整和现代组合代码按静态复核日期更新状态。PyTorch 2.15 的 Embedding、SDPA、MultiheadAttention、LayerNorm、Dropout 页面已于 2026-11-16 直接复查；非方形 mask 和接口语义证据补在 `research/transformer-component-apis-2026.md`。RoPE、RMSNorm、SwiGLU 和 Pre-LN 的论文边界沿用现有研究记录。
- **静态检查与同步**：简介、目录和第 16 章回链已增加现代 decoder 续篇；四篇文件无尾随空格，公式分隔、代码围栏和状态数配对，整册 21 个 Markdown 文件的相对链接目标存在，`git diff --check` 通过。没有运行书中代码、测试、模型、benchmark 或 PDF，也没有更新 PDF。4 个 Mermaid 图源码已检查，GitHub 页面实际渲染和外链最终复核仍待全册收口，因此第 2 章内容改写已完成，最终门禁未关闭。下一步进入第 3 章。

### 本轮第 6 章拆分与逐篇重写（2026-11-16）

- **结构诊断与决定**：原章约 2,052 行，包含 KV Cache、Prefill/Decode 指标、vLLM 分页/调度、量化、推测解码、长上下文压测和整体验收报告。它们共同服务于“目标负载下服务能否以可接受的质量、延迟、显存和成本完成任务”，但方法表示、优化目标与实现算法各不相同，单篇会让读者在六套机制之间频繁切换。保留第 6 章全册编号和总目标，拆为七篇：KV Cache、Prefill/Decode、Serving runtime、量化、推测解码、长上下文压测、实验报告；每篇明确读者问题与交接产物。
- **覆盖图与内容去向**：第 1 篇保留 attention K/V 状态、MHA/GQA/MQA shape、同构/混合层容量公式、Prefill/Decode cache 写入、RoPE offset、单请求与 prefix cache 区别、复用 key/租户边界及错误诊断。第 2 篇保留用户 TTFT 与模型阶段时间、TPOT/ITL/吞吐定义、CUDA 异步计时、warmup、prefill/decode 资源画像、chunked prefill 和 P/D 解耦边界。第 3 篇保留 KV block 容量与尾块碎片、前缀共享、引用计数、continuous batching、调度公平性、vLLM 参数、OpenAI 兼容语义检查和 runtime 公平比较。第 4 篇保留对称量化与 group scale、FP16/BF16/INT8/INT4/NF4/FP8 区别、权重/激活/KV 三本账、QLoRA 与推理量化边界、量化质量回归和失败归因。第 5 篇保留 greedy 接受、采样的 residual 校正、期望接受长度和成本推导、Transformers assisted generation/UAD、draft 类型及分布正确性检查。第 6 篇保留 attention/KV 复杂度、NIAH 与 Lost-in-the-Middle 的证据边界、长度×位置×任务×负载矩阵、长上下文 harness、截断与 OOM 归因。第 7 篇将前述实验条件、调用/会话指标、多轮 trace、任务成功分母、成本和回退规则合成可交付报告。没有删除核心目标所需的机制、例子、实现边界或质量评估。
- **主线、解释与去重**：开篇说明 prompt 经 tokenizer 成为 token ids，prefill 处理输入、decode 逐步生成，并将各方法定位为改变状态表示、调度或生成算法。沿一条负载证据链推进：先算状态容量，再测等待时间，再看多请求调度，再试数值压缩/推测加速，最后联合验证长上下文质量并给出决定。移除了章节尾部重复的 KV 容量、PagedAttention 尾块、量化三本账和推测解码速度演算；将重复案例移回公式后对应的小节。保留独立支线的知识深度，不为维持单文件而压成术语清单。
- **可复算例子**：保留 32 层、8 KV heads、8K token 的约 1 GiB cache，并对照 MHA 的 4 GiB；40-token 请求的 TTFT 0.30 s、E2E 2.00 s 得 TPOT 43.6 ms/token 与 E2E 20 token/s；16-token block 保存 1,000 token 需 63 个 block、尾块空 8 位；7B 参数裸权重估算 BF16 13.04 GiB、INT8 6.52 GiB、INT4 3.26 GiB，128 参数/scale 的 FP16 scale 增量约 0.102 GiB，总计约 3.36 GiB（未计其他运行时开销）；推测解码按 4 个候选、draft 16 ms、target 验证 25 ms、接受率 0.8/0.4 分别算约 12.2/24.9 ms 每输出 token；长上下文矩阵示例为 3 种长度×3 个位置×20 样本共 180 次；四轮 Agent 输入合计 15,512 token，说明首轮 512 token 会低估约 30.3 倍；200 秒 100 请求样例分别计算 80 token/s、72% 任务成功和约 $0.0056/成功任务。
- **公式、代码和图表**：七篇的行内数学统一为 `$...$`；静态扫描未发现旧行内括号式分隔、尖括号开头的数学下标或 `operatorname`。107 个 fenced blocks（11 个 Python、63 个 math、31 个 text、2 个 Mermaid；214 个围栏标记）全部配对；11 个 Python 块及紧邻状态均匹配，代码文本没有因拆分和讲解重组而改变，状态沿用“静态审阅，未运行（2026-10-29）”。行内美元符号计数 288（成对）。没有运行书中代码、模型、benchmark 或测试。两幅 Mermaid 仅完成源码静态检查，GitHub 页面渲染留到最终门禁。
- **证据与来源复核**：本轮沿用已登记的来源证据，并在 2026-11-16 定向复核 vLLM v0.32.0 metrics/design metrics、Transformers v5.17.0 assisted decoding、PyTorch 2.14.0 CUDA Event、speculative decoding 原论文和 LongHarness Bench v1 页面；未发现需要改写原结论范围的新证据，因此不新增 research 事实条目。固定版本只支持对应公开文档/API 语义，不代表依赖组合已联调或目标硬件已实测。来源边界继续由 `research/kv-cache-capacity-paging-and-compression-2026.md`、`research/inference-serving-metrics-and-disaggregation-2026.md`、`research/decoding-sampling-and-test-time-compute-2026.md`、`research/long-context-evaluation-and-effective-capability-2026-11-03.md` 和 `research/weight-quantization-formats-and-ptq-2026.md` 支持。
- **导航和静态验收**：简介/目录改为七篇阅读路径；第 2 章 GQA 回链指向 KV Cache 篇，第 15 章项目索引指向报告篇，第 16 章系统设计回链指向 runtime 与报告篇；QLoRA 链接沿用第 4 章拆分后的文件和锚点。对本轮触及的 12 个 Markdown 文件做本地相对目标扫描，失效目标为 0；`git diff --check` 通过。没有更新 PDF。GitHub 页面公式/Mermaid 实际渲染、外链终审和第三册连续复读仍待最终门禁，下一步处理第 7 章《RAG 项目实战》。

### 本轮第 7 章拆分与逐篇重写（2026-11-16）

- **原稿诊断与结构决定**：完整通读原 `chapters/07-rag项目实战.md`（1,430 行）。章节主线原本明确，但文档解析/切分、向量与关键词召回、精排、引用协议、选择性回答和评估各自承担独立任务；7.3 和 7.4 的若干关键算例还放在资料清单之后，读者需要跨过来源段才能继续看到解释。现拆为六篇，保持第七章逻辑编号：文档与切分、检索与上下文、候选重排、引用与证据、评估指标、失败归因与报告。旧单文件已由六篇连续正文替代，简介、目录和内外链接同步。
- **覆盖图与内容去向**：第一篇保留文档读取/清洗、页码/标题/表格/来源位置、chunk 元数据、版本/权限、token 窗口、overlap、chunk size 取舍和解析失败边界，并补入索引版本切换与过期证据停用。第二篇保留 embedding、归一化余弦、Sentence Transformers/FAISS、稀疏 toy demo、BM25、hybrid/RRF、query 改写、ACL/版本过滤、上下文预算和召回诊断。第三篇保留 bi-encoder/cross-encoder、分数口径、最大长度截断、候选覆盖、MRR/延迟账本、难负例和排序失败。第四篇保留引用编号、prompt 边界、结构化输出、引用合法性与支持性、claim 级证据、拒答、提示注入、版本有效期、稳定 citation map 和历史访问权限。第五篇保留评估样本契约、Hit/Recall/MRR、答案正确性/忠实性/完整性、引用支持率、拒答质量。第六篇保留失败归因、分层评估 demo、trace 隐私、单位成功成本、小样本/Judge 校准、成对回归和最终报告。GraphRAG 仍是跨文档关系与全库摘要的范围外候选，完整方法交给第十七册。
- **解释与算例调整**：把 token 化、文档结构、检索候选、证据支持和评估阶段接成一次员工制度问答。保留 320-token 文档在 120-token 窗口/20 overlap 下形成 3 块、存储 360 token 的手算；补 2 维 cosine 例子（0.96 对 0.60）和 BM25 的词频饱和/长度归一化推导；保留三题 MRR 从约 0.528 到 0.833、200-query 候选召回与精排保留率、40/100 候选延迟估算、四个 claim 的版本冲突与 7/10 支持率，以及 20 题 Recall/accuracy/coverage/abstention 分母示例。把旧稿放在资料列表之后的 citation-map 案例和评估切片、judge、A/B 例子移回相应解释段。所有实验数值均为明确的教学假设，不作为生产表现。
- **新研究与来源边界**：2026-11-16 复核 Microsoft GraphRAG 方法页、Sentence Transformers query/document 和 CrossEncoder API、FAISS 归一化内积语义、OpenAI Structured Outputs 与 prompt injection 说明、OpenTelemetry GenAI trace 敏感内容边界。补核 Lucene 10.3.1 BM25Similarity 的 IDF、词频饱和和文档长度参数，并参考 Robertson/Zaragoza 2009 原作者综述；新事实及版本/范围写入 `research/rag-retrieval-and-evidence-pipeline-2026.md`。Structured Outputs 仅约束符合条件的完整非拒答响应结构，应用仍需处理拒答/不完整响应并验证业务规则；这不证明答案事实或引用支持。
- **代码与公式**：16 个 Python fenced blocks 与原稿逐块比较，内容完全一致，状态均沿用“静态审阅，未运行（2026-10-30）”；未运行任何代码、模型、benchmark 或测试。新增两段 BM25 数学围栏；原有 33 段 math 围栏均保留。行内数学统一改为 `$...$`，没有改动代码字符串。
- **导航、链接与收尾状态**：简介第 7 章摘要和目录改为六篇路径；第 8、9、15、16 章及第 4、6、11、13 册和 INTERVIEW_BANK 的正文入口已改到对应分篇。代码、公式、链接、围栏和 Mermaid 的静态检查及 `git diff --check` 待本轮最终检查后补记；未更新 PDF。GitHub 页面 Mermaid 渲染、所有外链终审和第三册沿主路径连续复读仍留待全书重写结束。

### 本轮第 3 章拆分与逐篇重写（2026-11-16）

- **原稿诊断与结构决定**：原章将数据准备、最小 GPT、训练、解码、checkpoint/评估和 tokenizer 对照集中在一章。它们组成同一个小模型闭环，但各阶段都有独立的读者任务。现拆为五篇：token 数据契约、模型与训练、解码控制、checkpoint 与评估、tokenizer 与公平比较；五篇分别交付可审计窗口、可检查训练状态、可控生成、可复现评估和受控 tokenizer 比较，按此顺序衔接，不改变全册编号。
- **覆盖图与去向**：数据切分/去重、train-only tokenizer 拟合、文档边界与跨文档窗口、tokenizer artifact、EOS/PAD、shifted labels 与 loss mask 留在第 1 篇；模型 shape、交叉熵、优化、训练/验证、资源预算及过拟合定位在第 2 篇；temperature/top-k/top-p、停止条件、随机种子和解码行为在第 3 篇；checkpoint 原子保存/恢复状态、token 加权验证 loss、PPL、选模与评估边界在第 4 篇；字符级与 BPE 的序列长度、词表参数量、Unicode、tokenizer 配套和公平比较在第 5 篇。没有为了拆篇删除支撑“从数据到可复核模型行为”的核心主题。
- **解释顺序与细节**：第 1 篇把切分放在 tokenizer 拟合之前，使用边界窗口说明 `x/y` 右移对齐，并区分 EOS 标记与因果注意力重置；第 2 篇用一个小 batch 串起 logits、逐 token loss、有效位置分母和参数更新；第 3 篇把采样变换按 logits→temperature→候选截断→抽样的顺序解释，补足平局、无效候选、EOS、长度上限和空输出边界；第 4 篇区分 latest/best/archive 与基础推理/精确恢复，补充 optimizer、scheduler、RNG 和数据游标对恢复的影响；第 5 篇统一字符级和 BPE 的内容 token 口径，并保留规范化、往返解码、覆盖率和截断差异。
- **代码与公式**：共 17 个 Python 块，均有静态状态标记且未运行。第 5 篇 §3.6.6 BPE 演示改为比较排除 BOS/EOS 的内容 token 数；其状态已更新为“静态审阅，未运行（2026-11-16）”。其余代码块没有执行。五篇行内数学统一使用 `$...$`，静态扫描未发现 `\(...\)`、未配对美元符号、未配对代码围栏或 `_{<...}` 下标；不以代码运行结果代替例子解释。
- **导航与静态检查**：简介、目录及第 6 章 tokenizer 比较入口已指向拆分后的文件；相对链接扫描覆盖全册 31 个 Markdown 文件，未发现失效目标；`git diff --check` 通过。GitHub 页面公式/Mermaid 实际渲染、外链终审和整册连续复读留到最终门禁。没有运行书中代码、测试、模型、benchmark、构建或 PDF。

### 本轮第 4 章拆分与逐篇重写（2026-11-16）

- **原稿诊断与结构决定**：原章把基座模型/生成、SFT 数据、数据审计与全参数训练、LoRA、QLoRA、评估协议和自动化报告放在一篇。它们共享“改变回答行为并证明变化”的目标，但初学者需要分别学会数据契约、更新方式和评估流程。现拆为七篇，依次讲模型与生成、SFT 数据、数据审计与全参数 SFT、LoRA、QLoRA、评估协议与指标、自动评估与报告；简介、目录和相关章节入口已同步。
- **覆盖图与读者路径**：从固定模型/tokenizer revision 和生成基线开始；再讲 message 到 labels、padding 与 assistant-only loss；随后审计切分、截断、数据版本并建立全参数训练基线；再比较 LoRA 的低秩增量和 QLoRA 的量化基座；最后固定评估样本、指标和裁决协议，保存逐样本输出与报告。保留训练目标、有效 batch、显存/优化器状态账本、可训练参数检查、灾难性遗忘、量化误差、judge 校准及报告复核等原有重要知识；重复资源估算和重复 caveat 已合并。
- **推理与例子**：用一次 assistant 回答说明 prompt、目标 token 与 mask 的位置；以固定矩阵尺寸和 rank 说明 LoRA 可训练参数差异；以 7B 权重的 16-bit/4-bit 估算对比理想权重下界并说明 scale/metadata 开销；评估部分显式处理 tie、分母、切片退化和逐样本复核。修正公式中可能触发 Markdown/MathJax 解析问题的尖括号下标；QLoRA 内链改为拆分后的正确锚点。
- **代码与公式**：七篇共 48 个 Python 块，每块均保留“静态审阅，未运行”状态；本轮未修改代码，也未执行书中代码。行内公式采用 `$...$`；静态扫描中各文件围栏、公式分隔和状态标记数量配对，未发现旧行内括号分隔符、`operatorname` 或尖括号开头的数学下标。
- **导航与静态检查**：简介与目录的第 4 章条目链接七篇并说明顺序；第 6 章 QLoRA 回链和第 15 章相关实战入口均已更新；旧整章路径在本册 Markdown 中无残留。全册 31 个 Markdown 文件的相对链接扫描无失效目标，`git diff --check` 通过。没有运行代码、测试、模型、benchmark、构建或 PDF。目标平台公式/Mermaid 渲染、外链终审与整册连续复读仍待全书收口。

### 本轮第 8 章拆分与逐篇重写（2026-11-16）

- **原稿诊断与结构决定**：完整通读原《Agent 项目实战》（1,418 行）。原稿已包含 Tool Calling、ReAct、风险与授权、RAG+Agent 贯通案例，但知识点散布在长节和节后补充中，读者需要自行拼出“提议—校验—执行—观察—停止”的闭环。现拆为五篇：工具调用协议、ReAct 状态循环、执行安全与授权、轨迹审计与评估、贯通项目验收；简介、目录和当前交叉链接已同步。
- **覆盖与讲解顺序**：第一篇解释模型提议和真实执行的边界，保留 schema、内部注册表、计算器、参数检查、调用标识、依赖/并行、回执状态与审计；第二篇用轨迹、结构化状态、工具协议与 ReAct 的区别、预算、停止、无进展和恢复组织多步任务；第三篇覆盖最小权限、提示注入信任边界、业务校验、服务端确认、幂等/未知状态及沙箱；第四篇说明事件关联、trace 最小采集、终态、first failure stage、安全评估分母和 benchmark 边界；第五篇从只读政策助手任务契约走到 RAG/Agent 接口、版本时点、边界样例和交付证据。原稿中的重要判断均有新的上下文承接，不把“安全”“完成”留作空泛结论。
- **可复核算例**：保留 6,000 元对 5,000 元阈值及严格/包含边界；3 次模型调用输入/输出累计 2,650/245 token、模型加工具耗时约 1,452 ms；新增 120 元退款与 150 元自动额度的动作核验例，说明确认绑定、原幂等键查询和 unknown 转人工；安全样例保留 0/40 仅代表该组输入；10 条端到端样例分别报告 6/10 覆盖率、6/6 可回答集正确率、2/2 正确拒答、0/1 越权执行和 1/1 版本冲突转人工。
- **代码、公式和来源**：12 个 Python fenced blocks 与原文逐块保留，静态状态标记均沿用“静态审阅，未运行（2026-10-30）”；没有执行书中代码、测试、模型或 benchmark。原 Mermaid 图保留。公式使用 `$...$` 行内形式和 `$$` 展示形式，静态检查未发现旧行内括号式分隔、双反斜线、`operatorname` 或不合规下标；13 个 fenced blocks（12 Python、1 Mermaid；26 个围栏标记）配对。2026-11-16 复核 OpenAI Function Calling strict/default 与 parallel call 规则、Anthropic client/server tool 往返和 OpenAI Agent Evals trace grading；事实范围和动态接口边界记录于 `research/agent-tool-calling-security-2026.md`。
- **导航与静态验收**：简介和目录已列出五篇职责与交付物；第 9、15、16 章及第四册相关入口改到对应分篇。新章节本地链接目标、触及页面中的第八章入口均存在；数学分隔和代码围栏配对检查通过；`git diff --check` 通过。未更新 PDF；GitHub 页面公式/Mermaid 实际渲染和外链终审仍留到全册最终门禁。下一步处理第三册第 10 章。

### 本轮第 9 章拆分与逐篇重写（2026-11-16）

- **原稿诊断与结构决定**：原《评估与 Debug 实战》约 1,798 行，包含固定 benchmark、污染/泄漏、事实性、训练不收敛、微调后能力退化和人工评测六种独立的读者任务。它们组成离线评估闭环，但一次讲六套判断标准会让初学者频繁切换对象。现按诊断顺序拆成六篇：评测协议与 benchmark、数据污染与泄漏、事实性与 claim、训练不收敛、微调能力回归、人工评测；保留“先确保测量可信，再定位失败，最后解释证据”的关系，后续交给第十章做 Agent 轨迹与线上评估。
- **覆盖图与知识去向**：第一篇保留 benchmark 版本/运行条件、公开/业务/回归集、分层采样、development/validation/test/private holdout、样例 schema、exact/claim/schema 评分、micro/macro、空分母与 Wilson 区间手算、零依赖 runner、失败样例和证据层级。第二篇保留文档/时间切分、训练/评测/提示/检索/标注等信息泄漏路径、精确哈希与 shingle/Jaccard 算例、近重复扫描器、训练语料不可见时的调查、评测器负例测试、报告用语及 DCQ/ICLR 2026/BeyondBench 的各自边界。第三篇保留 claim—证据—世界状态区分、支持率与必需信息完整性各自的分母、开放/闭合证据世界和拒答效用、结构/claim 对齐/外部事实/稳定性四层评估、风险权重、checker 局限、失败归因与 TruthfulQA/FActScore/SelfCheckGPT 证据边界。第四篇保留目标与症状定义、数据/标签不变量、极小数据过拟合、loss/梯度/参数更新/数值精度、线性分类首步手算、诊断顺序及训练稳定性与泛化的区别。第五篇保留目标/保留能力双成绩单、协议和运行条件排除、有效训练预算、回放/减小更新/输出锚定/EWC/路由五条路线、能力退化矩阵与硬门槛优先规则、0.8805 效用算例及原因区分。第六篇保留绝对评分和成对比较、带行为锚点的量表、硬约束与加权分数、校准/盲评/仲裁、Cohen’s Kappa 手算、抽样与区间、LLM judge 校准、逐样本字段和隐私边界。
- **叙述与算例**：删除重复的“初学者视角/深入视角”模板标题，改成贴着各节问题的过渡标题；每篇新增具体问题开头和交接段。原有六个核心单元均完整保留，解释性汉字统计分别为 4,043、2,803、2,850、2,648、2,862、3,591；这些计数只作覆盖诊断，不作为篇幅门槛。练习保留 4/4 与 Wilson、部分事实覆盖、训练标签/学习率故障三个可复算任务，并改为链接到对应分篇。
- **研究和时效事实**：2026-11-16 重读 OpenAI 官方 Working with evals、Deprecations 和 Agent Evals 页面。官方指南仍称既有托管 Evals 于 2026-10-31 转为只读、计划于 2026-11-30 下线；迁移表另提供既有 hosted workflow 的 Promptfoo 入口，指南对新建/迭代提到 Datasets。正文明确这是产品服务状态，不表示评估方法或开源仓库失效，也不把不同迁移入口说成等价。动态事实与边界已更新到 research/evaluation-design-uncertainty-and-contamination-2026.md。
- **代码、公式与视觉**：全章 75 个 fenced blocks（6 Python、39 math、29 text、1 Mermaid；150 个围栏标记）按原章节内容迁移，六个 Python 块与既有状态标记均保留“静态审阅，未运行（2026-10-30）”；未运行书中代码、测试、模型或 benchmark。六篇行内数学统一为单美元分隔符，旧括号式行内分隔数为 0，美元符号成对；未发现尖括号开头下标或 operatorname。原 Mermaid 流程图保留在第一篇。
- **导航与静态检查**：简介和目录的第九章摘要改列六篇职责与交接；第八章贯通验收、第十五章作品集、第十六章技术表达、PAPERS 和 EXERCISES 的第九章入口均改到具体分篇。本章及这些入口的相对链接检查通过；代码状态、公式分隔、围栏静态计数通过，git diff --check 通过。对 PAPERS/EXERCISES 的全文件扫描另发现多条指向无关旧 research 资料路径的既有失效链接，当前第九章新增入口有效，这些未改的历史引用不属于本章内容。未更新 PDF。
- **遗留门禁与交接**：外部链接全量终审、GitHub 页面 Mermaid/公式实际渲染、第三册沿目录主路径连续复读留待全册收尾。下一篇进入第十章《Agent 轨迹与线上评估实战》。

### 本轮第 10 章拆分与逐篇重写（2026-11-16）

- **原稿诊断与结构决定**：原《Agent 轨迹与线上评估实战》有 573 行，包含轨迹失效定位、线上随机实验和灰度运行三类连续但不同的学习任务。保留“先解释单次 Agent 行为，再估计真实流量差异，最后作发布决定”的顺序，拆为三篇：轨迹诊断、线上 A/B 实验设计与统计判断、灰度发布与实验报告。这样能分别讲清过程证据、随机化推断和操作决策，同时维持离线评估到线上验收的交接。
- **覆盖图与知识去向**：第一篇保留任务状态契约、模型提议/程序 guard/工具执行/外部状态/用户答复的分层、首个不可恢复失败与根因的区别、trace 脱敏和受控证据引用、模拟回放边界、20 条地址任务中的 4/20 提前提议、3/4 拦截、1/20 真实副作用，以及任务成功、工具选择、参数契约、策略违例和失败定位的不同分母；保留 Agentic CLEAR、AgentRx 的研究角色和来源边界。第二篇保留随机化估计对象、哈希稳定分桶、主指标/护栏/诊断指标、请求/会话/用户/组织随机化的干扰关系、SRM 卡方、功效样本量、实验前协变量与 CUPED 假设、统计显著性与业务意义、固定样本停止规则以及零依赖统计 demo。第三篇保留随机分配分母与实际 exposure 的区别、日志字段和隐私保护、输出随机性/多轮状态/成本延迟/低频高后果事件、灰度阶段和可执行回退、切片、可复核报告及三道迁移练习。
- **数值与推理**：样本量例子按 77% 到 79% 的 2pp MDE 估算每组 6,736 个独立单位；CUPED 的教学假设为实验前/实验期相关系数 0.6，线性最优调整对应方差比例约 0.64（约降 36%），正文明确这不修复分桶或日志问题。代码示例中的 50:50 组数 12,000/12,080 对应期望 12,040 和 SRM χ²≈0.266；解决率从 77.00% 到约 79.39%，差约 2.39pp，区间约 1.34—3.43pp；负反馈、P95 延迟与成本分别给出变化；稀疏安全事件 6/12,000 与 7/12,080 的 Wilson 区间不足以证明非劣效，示例因此保持 hold。代码后明确组内 Wilson 区间不能替代组间安全风险差检验。
- **研究与时效事实**：2026-11-16 复核 OpenAI 官方 Agent Evals 工作流指南；trace 用于 workflow debugging，criterion 稳定后用 datasets/eval runs 做重复评测。托管 Evals 生命周期以同日官方指南和 Deprecations 页面为准，记录在 research/evaluation-design-uncertainty-and-contamination-2026.md；Agent trace 工作流来源边界新增在 research/agent-trace-diagnosis-and-online-evaluation-2026.md。OpenTelemetry GenAI span 敏感内容边界沿用 2026-11-16 的动态规范复核记录。
- **公式、图示和代码**：行内公式统一为单美元符号分隔，旧括号式行内分隔数为 0，美元符号成对；全章 19 个 fenced blocks/38 个围栏标记（1 Python、8 math、8 text、2 Mermaid），配对闭合；原有两张 Mermaid 流程图均保留。唯一 Python 块及“静态审阅，未运行（2026-10-30）”状态沿用；代码未改、未运行书中代码、测试、模型或 benchmark。
- **导航与静态检查**：简介和目录更新为三篇及其交接产物；第八章贯通项目、第九章人工评测、第十五章作品集、第十六章系统答辩、PAPERS 和 EXERCISES 的第十章入口均改到具体分篇；分篇之间链接前后承接。第十章正文和本册受影响导航入口的相对链接检查通过，公式/代码围栏静态计数通过，git diff --check 通过。未更新 PDF。
- **遗留门禁与交接**：Mermaid 和公式 GitHub 页面渲染、动态外链最终巡检、第三册沿目录主路径连续复读留待全部章节完成后。下一步进入第十二章《Diffusion 数值实验》。

### 本轮第 11 章拆分与逐篇重写（2026-11-16）

- **原稿诊断与结构决定**：重读第 11 章原稿（1,150 行）。内容包含 CLIP 检索和 zero-shot、对比损失、开源 VLM 问答、指令数据治理、图文 RAG 五种相互依赖的读者任务；原章已用“图片如何成为可检索、可回答、可审计的证据”串联，但一篇连续承载五组机制和评估，初读时要跨过多个对象。拆成五篇，按先观察图文检索、再解释训练目标、转向生成问答、组织监督数据、最后连接证据与权限的顺序展开，交接到第十四章多模态评估。
- **文件与章节结果**：新文件为 `11-图文理解与检索实战-1-图文检索.md`、`-2-对比学习损失.md`、`-3-VLM图片问答.md`、`-4-多模态指令数据.md`、`-5-图文RAG.md`；原单文件移除。第一篇交付相似度排序、zero-shot 提示、Recall@K/MRR 分母和可维护索引的读法；第二篇交付 CLIP 双向目标的张量/损失/梯度解释及多卡标签核查；第三篇追踪视觉输入、预算、生成输出和分任务评估；第四篇交付多模态样本契约、回答监督、图片级切分和泄漏检查；第五篇连接多通道候选、分数融合、授权证据和可复核回答。
- **覆盖图与知识去向**：第一篇保留双编码器、归一化与相似度矩阵的维度/零向量边界、CLIP 与 SigLIP/SigLIP 2 目标差异、prompt 影响、手造 4×4 检索与打乱反事实、索引版本、权限字段、多正样本和近重复切分。第二篇保留稳定 log-sum-exp、双向交叉熵、`N=1` 无负样本退化、温度影响、batch/全局负样本、all-gather offset、标签错位的梯度方向，以及训练目标和业务 Recall 不等价。第三篇保留 CLIP/VLM 职责、projector/Q-Former/跨注意力等信息流、patch 数与上下文预算的近似边界、Qwen2.5-VL 调用、VQA 任务切片、toy 评分、分辨率消融和各自手算。第四篇保留任务 schema/覆盖、拒答和安全样本、assistant-only mask 与有效分母、图片与近重复组切分、数据构造器、行随机切分约 58.88% 泄漏概率、监督 mask 算例及样本配比诊断。第五篇保留文本化/双索引/原生多模态架构、BM25/图像/OCR 分数口径、证据扩展、权限与版本复核、词袋融合 demo、过滤顺序导致的 top-k 差异、冲突处理和端到端错误归因。必要知识未因拆分移除；各独立讲解单元的例子与原对象保持一致。
- **过渡与解释调整**：各篇新增具体问题开头和承接/交接，区分“候选排序”“生成回答”“监督行为”和“证据授权”四种结果。第一篇先从用户检索截图的场景观察排名，再由第二篇返回训练目标；VLM 接口段说明模型专属类和 processor 输入字段；最后一篇把 ACL 逻辑边界与具体向量库过滤时机分开，解释过滤实现对召回和延迟的影响。
- **研究与事实复核**：2026-11-16 重读 Transformers 当前 Qwen2.5-VL、多模态模板和图像到文本指南，并核对固定 v5.21.0 多模态模板/模型页面；确认专用模型类、Processor 模板处理、图像输入字段和 `min_pixels`/`max_pixels` 分辨率控制的文档依据。同期复核 Azure AI Search security filter 和 vector query filter 文档，确认示例过滤标识不等于身份认证，且 pre/post/strict-post 过滤模式对召回、假阴性和延迟的语义不同。新证据和边界已补入 `research/multimodal-systems-and-evaluation-2026.md`；正文引用范围没有外推到其他服务。CLIP、SigLIP、SigLIP 2 论文结论继续按各自训练目标和配方范围表述。
- **代码、公式和图示**：6 个 Python 块按拆分前后 SHA-256 逐块核对完全一致；均沿用“静态审阅，未运行（2026-10-30）”状态，本轮没有运行书中代码、模型、测试、benchmark 或 PDF。总计 46 个 fenced blocks、92 个围栏标记（6 Python、27 math、11 text、2 Mermaid）配对；旧行内 `\(...\)` 分隔符已统一为单美元形式，行内美元符号成对。两张 Mermaid 均保留，GitHub 页面实际渲染留到整册最终门禁。
- **导航与静态检查**：简介和目录改为五篇及其职责、问题和交接物；`PROJECTS.md` 的图文 RAG 验收入口改到 CLIP 检索篇与图文 RAG 篇。五篇及简介/目录本地相对链接检查通过，相关项目链接目标存在；`git diff --check` 通过。全量扫描 `PROJECTS.md` 仍发现多条指向旧 `research/model-update-2026-09/` 的无关失效链接，本轮未改动；全书外链最终核验、GitHub 页面渲染和连续复读仍待全册完成。
- **遗留门禁与交接**：代码未执行，PDF 未更新。全册其余章节连续复读、全部 Mermaid/公式目标平台渲染、外链最终巡检留待最后门禁；下一篇进入第 12 章《Diffusion 数值实验》。

### 本轮第 12 章重写（2026-11-16）

- **结构决定与读者路径**：完整重读 343 行原章并保留单章，因为前向加噪、噪声目标、反向转移、CFG、采样器和生成验收共同回答“噪声如何变成可评估样本”，依赖关系连续。改为“前向日程 → 训练目标与反向后验 → 两步手算 → 向量公式审计 → 图像系统 → 文本引导和少步采样 → 失败诊断与指标”八段；把关键算例移到代码前，不删减原有知识范围。
- **覆盖与讲解深度**：保留 Markov 前向分布、`alpha/beta/alpha_bar` 关系、合并噪声闭式公式、SNR、终端信噪比与初始化边界；解释噪声 MSE 的张量归约、Min-SNR 时间步加权、后验均值/方差从高斯乘积配方得到的过程，以及 `epsilon`/`x0`/`v` 参数化换算。补入固定与学习反向方差的区别，并明确训练 loss、模型均值、方差配置和采样更新是不同环节。保留 MLP/图像主干、VAE latent 路径、CFG 两套参数记法及其数值对照、负向提示分支、score 直觉、DDIM/DPM-Solver/蒸馏的改动层次，以及 FID/IS/CLIPScore 的适用边界和按失败维度评估。
- **可复算例子**：两步教学 schedule `beta=[0.1,0.2]`、`x0=1`、合并噪声 `-0.5` 得 `x2≈0.583953`、反解 `x0≈1`、反向均值 `≈0.8642`、方差 `≈0.0714`、标准差 `≈0.2672`；预测误差 `+0.2` 使均值偏移 `≈-0.0845`，该步 `SNR≈2.57`。明确合并噪声不包含各中间步独立抽样信息，也不把算式自洽说成模型或图像质量验证。保留 20 prompts × 3 seeds 的 60 张教学标注及分母边界。
- **研究与事实核对**：复用研究记录中的 DDPM、Improved DDPM、Min-SNR、CFG、DDIM/DPM-Solver、指标定义及论文证据边界；2026-11-16 另核对当前 Diffusers DDPMScheduler 参数文档与固定 v0.35.1 Stable Diffusion pipeline 源码。预测目标和方差类型、经典 CFG scale 与 negative prompt 的版本边界写入 `research/diffusion-generation-guidance-and-sampling-2026.md`，正文不外推至其他 pipeline。
- **代码、公式和图示**：原有 Python 块 SHA-256 前后均为 `78aee8903171a6b6d0a4ee2cfcb466b1fd4b684442a052bf120aae52b22d84a6`，按“静态审阅，未运行（2026-10-30）”保留；没有运行书中代码、测试、模型、benchmark 或 PDF。当前共有 29 个 fenced blocks / 58 个标记（1 Python、26 math、1 text、1 Mermaid）；238 个行内美元定界符成对，未发现 `\(...\)`、`operatorname`、`_{<...}` 或未闭合围栏。图只做源码静态检查，GitHub 页面渲染留到整册门禁。
- **导航与静态检查**：简介和目录的第 12 章摘要与问题/交付已更新；本章本地链接目标均存在；`git diff --check` 通过。没有需要重复新增的 PAPERS、EXERCISES、INTERVIEW_BANK、GLOSSARY_EN_ZH 或 KNOWLEDGE_GRAPH 条目。
- **遗留门禁与交接**：外链全量终审、GitHub Mermaid/公式实际渲染和第三册目录主路径连续复读留待全书完成后。下一篇处理第 13 章《语音识别与问答实战》。

### 本轮第 13 章重写（2026-11-16）

- **范围与结构决定**：回读 466 行原章并复核已有章节扫描。保留单章与 13.1/13.2 两段结构：Whisper 识别、误差测量、检索问答和高风险确认共同回答“音频里的信息怎样成为可核验的答案或动作”；其中 13.1 建立识别与评估基线，13.2 把基线接入证据和业务状态。未为保持原章数删减主题，也未机械拆段。
- **读者路径与连续案例**：开篇用“耳机用了三天、有杂音、能否退货”引出识别、规则检索与证据解释；说明“使用三天”不等于规则的“签收七天”。补足波形、log-Mel、编码器/解码器、token、任务控制 token 的初学者解释；将同一问答贯穿 ASR 假设、字符重合 toy 检索、证据边界和最终回答。另保留“订单 712 不要取消”作为否定词反转的高风险例子。
- **覆盖与深度**：保留 Whisper 前处理、30 秒窗口、长音频时间戳解码、通用外部分块/VAD 差异、chunk 重叠与时间戳契约、模型/API 版本与 ffmpeg、WER/CER/UER、空参考、关键实体、编辑距离实现、95 秒/30 秒/5 秒重叠的 110 秒算例、RTF、宏/微聚合、置信度校准、数据修订和隐私。补充离线长音频不等于流式字幕、片段/词级时间戳粒度、专名提示可能误纠正、开发/测试录音分组、BM25/embedding 的角色与分数边界、query normalization 的语义保护、端到端分层轨迹、音频 hash 的可关联风险、音频输入不提升工具权限、确认/修改/取消/超时状态和跨层交付字段。
- **来源核验**：2026-11-16 复核 Whisper v20250625 固定源码和 Transformers v5.17.0 Whisper 文档；固定版本来源支持 16 kHz 单声道解码/重采样、30 秒窗口、未截断长音频、时间戳驱动的顺序转写、compression ratio/log probability/no-speech/temperature fallback、prompt IDs 和 token 时间戳分组。接口版本只作为可复查示例，不声称为当前最新版本或已联调环境。AudioHijack 预印本的引用只支撑威胁边界提示，不推广为目标产品攻击率。新事实及其证据范围写入 `research/speech-recognition-generation-and-realtime-2026.md`。
- **代码、公式与静态检查**：3 个 Python 代码块未改、未运行；保留 2026-10-30“静态审阅，未运行”状态。未执行书中程序、测试、模型、benchmark、构建或 PDF。待本章静态核查完成后记录围栏、数学分隔符、术语、本地链接、公式及 `git diff --check` 结果；Mermaid/公式的 GitHub 页面渲染与全册连续复读留到最终门禁。
- **导航、配套与交接**：简介和目录的第 13 章摘要同步到“ASR 分层质量—关键字段—证据问答 trace—高风险确认”；新增的 RAG/Agent 交叉链接指向拆分后的第七章证据篇和第八章执行安全篇。第二册 UER/文本规范化内容继续复用，不新增 PAPERS/EXERCISES/面试题/术语/知识图谱重复条目。下一篇进入第 14 章《多模态评估与交付》；整册最终复读、外链终审和 Mermaid 实际渲染仍待完成。
