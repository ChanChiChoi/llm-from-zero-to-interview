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
- 目录列出 16 章，回答的问题和交接产物清楚；与 `chapters/` 的 16 个正文文件逐一对应，序号和路径相符。主线 1—10、选读 11—14、收束 15—16 的顺序保留。
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
| 07 RAG 项目实战 | 用版本化、带权限的文档形成可引用证据 | Agent 动作 |
| 08 Agent 项目实战 | 约束工具提议，记录执行回执与轨迹 | 离线评估 |
| 09 评估与 Debug 实战 | 建立离线基线、失败归因和回归集 | 线上评估 |
| 10 Agent 轨迹与线上评估实战 | 连接轨迹诊断、线上实验和灰度决策 | 主线项目交付 |
| 11 图文理解与检索实战 | 选读：图文数据、表征和检索链路 | 多模态综合评估 |
| 12 Diffusion 数值实验 | 选读：手算式追踪前向加噪和反向去噪 | 多模态综合评估 |
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

- 第三册架构审视、简介/目录检查已完成；第 1—16 章已全文扫描并完成静态精修，但整册连续复读、Mermaid 目标平台渲染和外链终审尚未关闭，因此不得标记第三册完成。
- 简介已有 16 条章节摘要，均已按本轮章节审阅结论复核；目录条目和正文章节结构一致。
- 第 1—16 章均有下方 `chapter_scan` 记录；代码无可靠逐块执行证据时标记静态审阅，不沿用无法追溯的“已验证”标签。
- 当前下一入口：第三册最终门禁——按既定阅读路径连续复读、逐图核对 GitHub 渲染、终审章节外链。按 `AGENTS.md` 的模块化规则，压缩或新会话恢复时完整重读 `写作规范/总纲.md`，再读取当前阶段相关完整模块。
- 保留用户明确约束：不更新 PDF；不执行书中程序、模型、benchmark 或测试，仅做正文、公式、代码静态审阅和必要的文档静态检查。
- book-02 的 GitHub Mermaid 实际渲染及旧外链全量核验仍未完成；book-01 和 book-02 的整册连续复读/链接门禁也依各自进度处理。

## 接续状态：第 1 章完成

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
