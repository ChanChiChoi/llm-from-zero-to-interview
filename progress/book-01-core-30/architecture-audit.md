# 第一册架构审视

## book_contract

- 定位：书系主线入口，讲清大模型从输入表示、预测与训练到生成和验证的最小完整闭环。
- 目标读者：具备基本机器学习或深度学习背景，但尚未形成大模型全局图景的读者。
- 前置：理解向量、矩阵乘法、概率和梯度的基本含义；数学与 PyTorch 深入内容转接第十三、十四册。
- 能力目标：从 token 序列解释表示、Transformer 前向、训练目标、参数更新、后训练、生成与验证；能复核小例子并指出下一步专题入口。
- 范围：建立共同语言和机制地图；不承担百科全收、前沿研究深挖或生产系统实战。
- 类型：渐进式概念教材，含多模态概览和综合迁移收束。
- 书系交接：第二册承接理论和研究深度；第三册承接代码与项目；第四册承担速查；第五至第二十四册承接训练、推理、评估、安全、多模态、Agent、AI Infra 等专题。

## intro_audit 与 toc_audit

简介与目录目前相互一致：都承诺五部分阅读递进，并将文本模型主线、多模态分支和综合迁移分开。简介给出读者、前置、能力、范围和后续入口；目录给出 38 讲的分组、文件入口、阅读顺序和书系交接。初步判断现有骨架兑现了本册定位，未发现需要在册级阶段重排或增设独立章节的依据。

文件夹名 book-01-core-30 沿用旧标识，而目录、README 和实际章节均是 38 讲。当前先保留稳定路径，后续导航 QA 需确认所有对外名称持续使用“38 讲”，并把文件夹名视作历史 slug；不因路径名称触发大规模重命名。

## chapter_map

| 文件 | 讲次 | 在本册主线中的作用 | 依赖与交接 |
| --- | --- | --- | --- |
| chapters/01-基础与语言模型.md | 1–5 | 建立语言建模、next-token 目标、交叉熵、梯度和优化器基础 | 为 Transformer 前向和训练过程提供目标与优化语言 |
| chapters/02-transformer核心.md | 6–15 | 从 tokenization、表示、attention、block、位置编码走到 miniGPT | 接收训练目标，输出 logits 和自回归生成路径 |
| chapters/03-预训练与规模化.md | 16–18 | 从训练数据和预训练流程扩展至规模化与分布式基础 | 接续模型机制，交给后训练 |
| chapters/04-后训练与对齐.md | 19–23 | 解释指令微调、SFT、RLHF、DPO、奖励模型与目标错配 | 将 base model 变成可交互助手，交给推理与评估 |
| chapters/05-推理、部署与评估.md | 24–28 | 讲采样、KV cache、注意力优化、量化、幻觉、评估和安全 | 从模型回答过渡到服务约束与可信验证 |
| chapters/06-研究前沿与综合实验.md | 29–30 | 用开放问题和综合实验收束训练/推理部分，练习证据和实验判断 | 依赖前 28 讲，并将综合方法交给全册复习 |
| chapters/07-多模态理解与表示.md | 31–33 | 从文本主线扩展到视觉和跨模态表示、对齐与理解 | 多模态概览，完整机制交接第十五册 |
| chapters/08-多模态生成与交互.md | 34–36 | 概览图像生成、语音交互、视频生成与评估边界 | 延续多模态分支，交接第十五册和相关专题 |
| chapters/09-综合复习与技术表达.md | 37–38 | 把知识迁移到开放问题、实验设计和技术表达 | 汇总前八个文件，形成全册学习闭环 |

### 章节关系图

~~~mermaid
flowchart TD
    A["语言建模与优化基础：1–5"] --> B["Transformer 前向：6–15"]
    B --> C["预训练：16–18"]
    C --> D["后训练与对齐：19–23"]
    D --> E["推理、部署、评估与安全：24–28"]
    E --> F["研究问题与综合实验：29–30"]
    B --> G["多模态理解与表示：31–33"]
    G --> H["多模态生成与交互：34–36"]
    F --> I["开放问题拆解与技术表达：37–38"]
    H --> I
    J["专题书与实战书"] -.->|需要深入时转接| C
    J -.->|需要深入时转接| D
    J -.->|需要深入时转接| E
    J -.->|需要深入时转接| G
~~~

## discovery_scan

- 检索日期：2026-10-28。
- 扫描问题：第一册“输入—表示—计算—学习—生成—验证”的主线是否仍能覆盖当前读者必须具备的整体模型；近年推理时计算、实时多模态交互和评估激励是否要求独立改变书册骨架。
- 时间范围与资料类型：重点查阅 2024–2026 的原始论文、官方技术文档和系统卡；基础架构仍以经典论文和专题书作交接。
- 候选 1：推理时计算和可验证推理。来源包括 Snell 等人在 ICLR 2025 发表的 test-time compute 研究及 DeepSeek-R1 技术报告。它对本册的读者价值是解释“next-token 预测”如何在推理时通过采样、搜索、修订和 verifier 扩展；当前目录已有第 24 讲采样/Best-of-N、第 29 讲 test-time compute 研究问题及后续第十六册。初步去向：保留在已有讲次的机制桥接与专题链接，不新增独立章节；第 24、29 讲章级扫描时确认覆盖和证据边界。
- 候选 2：实时多模态交互与工具。OpenAI Realtime 文档将会话描述为有状态事件流，包含音频/文本生成、图像输入和 function calling；这是单一产品 API 的接口证据，不代表所有模型结构。当前第 31、35 讲已讨论统一模态和实时对话，第十五、十七、二十、二十二册承担专题。初步去向：保留高层概念与跨册入口；在第 31、35 讲章级扫描时检查是否把级联与原生音频路线、轮次状态和工具执行边界讲清。
- 候选 3：准确率式评估对猜测和拒答的激励。Kalai 等发表于 Nature 的 2026 研究分析了准确率中心评估可能惩罚不确定时的 abstain，并以开放 rubric 作为方向；证据范围受论文任务和构造实验限制。现有第 1、28 讲已提及该研究和拒答评估。初步去向：不新增章节，核对现有结论是否保留论文边界，并由第七册承接完整评估设计。
- 其他领域：tokenization、Transformer/位置表示、预训练、对齐、推理服务和多模态基础已有对应主线或专题落点；本轮扫描未发现足以改变五部分骨架的新独立主线。章级扫描仍须逐篇确认，不把册级扫描当作章级完成。
- 证据来源和研究事实详见 research/test-time-compute-and-evaluation-2026.md；该文件只存知识与证据，不存本册决定或修改状态。

## architecture_decisions

1. 暂时保留简介的五部分承诺、目录的 38 讲顺序及 9 个正文文件；主线从基础到生成验证，多模态作为分支，最后综合迁移，符合渐进教材定位。
2. 不把第一册扩成推理模型、Agent、多模态系统或评测专题书；新兴机制先在既有讲次提供足够的定位、最小例子和边界，再链接专题书。
3. 完成章级扫描后如发现重要机制缺少讲次或跨册交接断裂，返回本节更新册级骨架决定。
4. 目录定稿和简介稳定后，需要确认文件夹旧 slug 与 38 讲标题的导航一致；暂不重命名目录。

## visual_map

现有简介、目录与章节已使用主线流程图、数据流图和评估表。当前审视未发现必须新增章节级图示的骨架关系；逐章阶段再判断状态、资源账本、比较和时序是否需要图表，并检查 Mermaid 语法与 GitHub 渲染入口。

## coverage_map

- 输入与语言模型目标：第 1–2 讲。
- loss、梯度、优化器：第 3–5 讲。
- token 到 logits：第 6–15 讲。
- 数据、预训练与分布式：第 16–18 讲；深度转第五、九、二十三册。
- SFT/偏好/奖励：第 19–23 讲；深度转第五册。
- 解码、缓存、量化、评估与安全：第 24–28 讲；深度转第六至第八、第二十四册。
- 研究/综合实验：第 29–30 讲；研究方法转第十册。
- 多模态理解、生成与交互：第 31–36 讲；深度转第十五册。
- 全册问题拆解与迁移：第 37–38 讲；求职和面试专项转第十二册。
- reasoning、Agent、协议与工具调用：第一册只作适用时的跨册入口，完整机制分别转第十六、十七、二十、二十二册。

## chapter_scan 与精修记录：第 1 讲

- 日期与范围：2026-10-28；`chapters/01-基础与语言模型.md` 的第 1 讲；检查 next-token 目标、能力来源、记忆/泛化、事实性边界及与后续训练和评估机制的衔接。
- 扫描问题与资料范围：语言建模目标能支持哪些能力，单个示例如何区分记忆和泛化，记忆暴露有什么证据边界，以及准确率式评估怎样处理不确定回答。覆盖 2020–2026 年原始论文；包括 GPT-3 论文的引言与任务结果、Chinchilla 的固定计算预算比较、Carlini 等人的 GPT-2 训练数据提取全文、Kalai 等人的 2026 年 Nature 论文。
- 候选与去向：
  - few-shot/in-context 能力：保留在本讲作为 next-token 目标可支持的任务行为示例。GPT-3 论文报告了翻译、问答和特定算术等任务，但不同任务表现差异明显；正文保留任务与规模条件，不将结果写成普遍推理保证。
  - 模型规模与训练数据量：保留 GPT-3 与 Chinchilla 原有来源和限定语，不新增章节；Chinchilla 的比较是固定计算预算及其训练配置下的实验结论。
  - 记忆与泛化：新增对 GPT-2 训练数据提取实验的短说明，强调“只在一份文档中”不等于“只出现一次”，并链接第八册隐私专题和第九册去重/污染专题；不在第一册展开攻击方法。
  - 幻觉与评估激励：为 2026 年 Nature 研究增加贴近结论的引用，保留论文设定限制，不把准确率指标概括为必然导致所有模型幻觉；机制深度仍由第一册第 28 讲和第七册承担。
  - 未发现足以改变本册定位、边界或章节关系的独立主线；维持第一讲骨架。
- 修改决定与结果：明确 next-token 预测利用的是有助于降低预测误差的统计线索，不把正确续写归因为固定的人类式推理机制；将模型条件分布与温度/top-k/top-p 解码操作分开描述；将 bigram 改称“二元语言模型”，补上计数定义和逐步手算；补充模型特定的记忆提取例证及跨册入口。
- `length_status`：本讲修改后解释性中文 5,258 字（脚本剔除标题、代码围栏、表格、引用状态行、行内代码与公式后计汉字；保留解释性列表），核心单元，下限 1,800 字，通过。
- `evidence`：关键来源均贴近正文。新增记忆证据保存于 `research/language-model-memorization-2026.md`；评估激励来源范围和局限补充于 `research/test-time-compute-and-evaluation-2026.md`。GPT-3 与 Chinchilla 的结论按论文任务、训练预算和配置限定。
- `example_status`：bigram 例子复用代码中的五句输入；`love` 的三个后继各出现一次，分母为 3，各得 1/3；有后继的 `learning` 两次都接 `is`，所以样本频率为 2/2=1。再加入 `machine learning really works` 后，`learning` 的计数变成 `is: 2`、`really: 1`，概率为 2/3 与 1/3。明确这些值只描述小语料观察，不代表真实语言中的必然概率。
- `code_status`：代码块未改动，可执行语句、输入输出约定、依赖、运行环境及承诺输出均未改变；沿用正文现有“已验证（2026-10-07；Python 3.13.13；标准库）”标记，本轮未重跑。更早的详细命令和实际运行日志未在现有 `progress/` 找到，记录为证据缺口，不补造运行细节。
- `formula_status`：新增经验条件概率公式，变量和计数口径已定义，且与手算过程对应；将 `P_θ` 记号统一为 `$P_\theta(x_t \mid x_{1:t-1})$`，区分模型分布和解码。
- `link_status`：新增 arXiv 论文与 Nature 论文直达链接；新增的第八册、第九册相对链接及本讲原有第 28 讲链接均已确认目标文件存在，仍需纳入全库链接 QA。
- `visual_status`：本讲的概率计数由输入、分母、结果和代码输出直接对应；未额外加图。章节文件开头已有本部分训练主线流程图。
- `sync_status`：简介、目录和外部配套无需因本讲的例子与引用补充而改动；已建立到隐私/记忆和数据去重专题的正文入口。
- `open_risks`：Carlini 等人的结果仅针对 GPT-2 与特定查询/筛选流程；评估激励论文的理论与实验不代表所有准确率指标；旧代码验证记录缺少完整命令和实测输出细节。

## chapter_scan 与精修记录：第 2 讲

- 日期与范围：2026-10-28；`chapters/01-基础与语言模型.md` 的第 2 讲，从序列联合概率、自回归分解到训练批次、生成循环、mask、对话模板、练习与小结。其余讲次不在本次范围。
- 扫描问题与资料范围：核对链式法则的首位置约定、序列概率如何逐项计算、next-token 输入/目标的位置映射，以及框架内部/调用方移位责任；同时检查 teacher forcing 与推理前缀差异、因果 mask、SFT loss mask、EOS 和模型特定聊天模板。资料范围为 2015 年序列预测论文与固定版本 Transformers v5.17.0 官方文档/源码。文档页面提示 v5.21.0 已发布，但检索工具无法打开该版本，故本讲引用固定 v5.17.0 且不称其为最新版本。
- 候选主题与去向：
  - BOS/EOS 与链式分解首项：补充空历史前缀约定、BOS 上下文和 EOS 是否属于被评分序列的说明，不另设讲次。
  - Scheduled sampling / 暴露偏差：Bengio 等 2015 年来源针对循环神经网络序列预测，保留为训练/推理条件差异的简短机制线索；不展开为通用 Transformer 解决方案，不把它作为所有生成错误的解释。
  - 标签 shift 契约：保留调用方显式右移与模型 loss 内部移位两种表示，并用 Transformers v5.17.0 教程、data collator 和 GPT-2 causal LM 头交叉核对。固定版本源码区分 collator 生成同长 labels/屏蔽 padding 与模型头内部 shift；只作为该版本/模型的实现例，不外推到全部框架。
  - EOS 与 padding 的 ID 冲突：教程样例把 pad token 设为 EOS，v5.17.0 collator 源码按 `pad_token_id` 掩蔽 labels；正文将二者结合标为条件性风险，并明确这是特定配置下的推论，不是所有训练管线的事实。
  - 聊天模板和特殊 token：依据 Transformers v5.17.0 的 Mistral/Zephyr 示例，说明同一基础模型也可能采用不同控制 token；加入重复 BOS/EOS 风险提示，未展开具体模型配置。
  - SFT prompt loss mask：按监督目标区分 assistant-only 与整段对话目标；不写成所有训练都必须屏蔽 user/system prompt。
- 修改决定与结果：保留本讲结构；在链式法则后定义空前缀并补充假设条件概率乘积例；修正输入/labels形状与两种 shift 接口之间的关系，增加 `[10,20,30,40]` 逐位置对齐表；将未来 token 泄漏检查改为固定前缀与输入条件的反事实比较；收紧 prompt mask 和模板失配诊断的适用条件。
- `length_status`：本讲解释性中文 3,894 字（剔除标题、代码围栏、表格、引用状态行、行内代码与公式后计汉字；保留解释性列表），核心单元，下限 1,800 字，通过。
- `evidence`：Bengio 等论文支持特定 RNN 序列预测任务中的真实前序 token/生成前序 token 差异及误差累积动机；HF v5.17.0 文档与固定版本源码分别支持教程约定、collator 行为、GPT-2 causal LM 的 labels 内部移位、聊天模板示例和 pad/EOS 配置。由教程配置与 collator 逻辑推出的 EOS labels 遮蔽风险已标为条件性推论。来源阅读范围与证据边界见 `research/causal-lm-training-and-chat-template-contracts-2026.md`。v5.21.0 页面无法访问，未借该版本支持正文结论。
- `example_status`：链式概率例使用明确标注的教学假设值 $0.2,0.5,0.3,0.8$，逐步得 $0.1,0.03,0.024$，即四 token、未计 EOS 的序列概率为 2.4%；不是实测概率。对齐例用 token IDs `[10,20,30,40]` 展开三个预测对，并说明显式 shift 与完整序列内部 shift 的对应关系。
- `code_status`：第 2 讲 Python token-list 例子未修改，代码语句、输入输出、依赖、环境与承诺输出均未变；沿用正文“已验证（2026-10-07；Python 3.13.13；标准库）”标记，本轮未重跑。现有 progress 未找到对应的命令、完整输入范围和实际输出日志，保留为历史证据缺口，不补造细节。
- `formula_status`：链式公式的 $t=1$ 空前缀已定义；增补序列概率逐步乘法；说明完整长度 $T$ 的模型输入与显式切为 $T-1$ 输入/目标的 shape 责任。
- `link_status`：新增 arXiv Bengio 与 Hugging Face v5.17.0 文档/固定源码链接，均已在浏览中打开并核验对应段落；没有增加本地相对链接。
- `visual_status`：用概率累乘表和 token 位置映射表分别呈现运算与对齐；它们比新增流程图更直接。现有章节 Mermaid 未改动。
- `sync_status`：本讲未改目录或书册边界；正文文件还包含第 3–5 讲，待该文件完成后再决定并更新简介中的章节摘要。新事实来源仅写入 `research/`。
- `open_risks`：标签移位责任是模型/版本相关接口契约；v5.17.0 例子不可外推。v5.21.0 文档未能访问。代码旧验证的详细运行记录仍缺失。

## chapter_scan 与精修记录：第 3 讲

- 日期与范围：2026-10-28；`chapters/01-基础与语言模型.md` 的第 3 讲，从交叉熵直觉、最大似然与 KL 关系到 perplexity、单样本/批量代码、mask、指标边界、练习和小结；不含第 4 讲梯度更新。
- 扫描问题与资料范围：核对负对数似然与交叉熵的数值链、one-hot 目标的实际计算、logits 到稳定 loss 的路径、mask/reduction 对分母的影响，以及 PPL 在 tokenizer、有限上下文和窗口评估下的可比性。查阅 PyTorch `CrossEntropyLoss` v2.14 官方 API 和 Hugging Face 固定长度模型 PPL 官方指南（页面未固定版本）；范围覆盖接口定义、忽略标签、自然对数 PPL、tokenization 和 sliding-window/stride 说明。
- 候选主题与去向：
  - MLE、NLL、cross entropy 与 KL：作为同一目标链的核心机制保留；把数据集似然写作序列似然乘积，说明序列内部仍由条件概率分解；补充 KL 非对称，避免将它误称为距离。
  - PPL 的上下文窗口与切块：纳入本讲，说明相同目标 token 在不同前缀条件下有不同 NLL，并给出长度为 3 的窗口与目标 `$x_5$` 的对照；不引入跨模型榜单分数比较。
  - tokenization、有效 mask 和 reduction：用 PyTorch v2.14 文档约束 raw logits、类别索引、`ignore_index` 和聚合表述；用 NumPy helper 已对齐的 logits/labels 说明清楚，不暗示它自动处理 causal shift。
  - 固定长度 PPL 指南中的计算示例：只借用窗口上下文和目标计数方法，不沿用其示例分数。静态阅读发现，该页示例对每个窗口均用 `num_valid_tokens - batch_size` 估计 shift 后目标数；在后续重叠窗口中，上下文前缀 labels 已是 `-100`，再减一次可能少计一个有效目标。该判断是代码阅读推导，未执行；记录在研究来源边界中，不把它写成已验证的普遍缺陷。
  - 交叉熵数值稳定性：保留现有 log-sum-exp 教学代码，增加与同一 logits 输入对应的手算；代码没有改变，不扩大为框架性能比较。
- 修改决定与结果：补充样本序列乘积似然、取 log 后的 NLL 和 token 平均；加入 `$0.8,0.5,0.1$` 的手算，依次得似然 $0.04$、总 NLL $3.2189$ nats、平均 loss $1.0730$ nats 和 PPL $2.9240$；展开 one-hot CE 与 good logits 的 softmax/目标概率/loss 中间量；为 PPL 增加 causal-LM 适用边界、上下文窗口/stride/tokenizer 条件及目标计数说明；收窄 prompt loss mask 和低 PPL 的解释。
- `length_status`：本讲解释性中文 4,011 字（剔除标题、代码围栏、表格、引用状态行、行内代码与公式后计汉字；保留解释性列表），核心单元，下限 1,800 字，通过。
- `evidence`：PyTorch v2.14 官方文档支持未归一化 logits、类别索引、`ignore_index` 和 reduction 的公开口径；Hugging Face PPL 指南支持标准 PPL 定义、tokenization 和固定上下文窗口示例。PPL 指南页面未固定版本；其具体代码中目标分母的疑点以静态推论记录，未复用页面报告的分数。来源和边界见 `research/cross-entropy-mle-and-perplexity-2026.md`。
- `example_status`：三 token 条件概率 $0.8,0.5,0.1$ 逐项得到 NLL $0.2231,0.6931,2.3026$ nats；乘积似然 $0.04$，总 NLL $3.2189$，平均 $1.0730$，PPL $2.9240$。one-hot 示例从 $q(y)=0.7$ 算得 $0.3567$ nats；纯 Python logits 示例手算稳定归一化常数 $1.134149$、目标概率 $0.881718$ 和 loss $0.125883$ nats。批量 NumPy 示例保留 5 个有效目标的 token loss、均值和 PPL 输出，用于 mask 与 reduction 对照。
- `code_status`：本讲纯 Python 与 NumPy 两段可运行代码均未修改；沿用正文标记的已验证状态（2026-10-07；Python 3.13.13；标准库/NumPy 2.4.6），本轮未重跑。progress 中未发现两段对应的具体命令、完整输入边界和实测输出记录；保留为历史验证细节缺口，不补造运行日志。
- `formula_status`：核对了 $H(p,q)$、$\mathrm{KL}(p\Vert q)=H(p,q)-H(p)$、序列负对数似然和 $\mathrm{PPL}=\exp(L)$；解释自然对数对应 nats，并给出完整代入。PPL 例限定自回归口径及有效目标/窗口设置。
- `link_status`：新增 PyTorch v2.14 `CrossEntropyLoss` 与 Hugging Face fixed-length PPL 直达链接，浏览时已打开并核对相关段落；未增加本地相对链接。PPL 页面未固定版本，进度中显式保留该边界。
- `visual_status`：用三 token NLL 表、四类 one-hot 计算和短上下文窗口对照呈现数值；它们直接说明公式与协议影响，本讲未新增 Mermaid 图。
- `sync_status`：无目录或书册边界变化。该 Markdown 文件仍含第 4–5 讲，待整文件完成后更新简介的章节摘要；公式/例子的变化无需题库或术语表同步。
- `open_risks`：Hugging Face PPL 指南无固定文档 revision；其重叠窗口目标分母需谨慎，正文未引用其具体分数。旧代码已验证标记没有可追溯的详细本地运行记录。

## chapter_scan 与精修记录：第 4 讲

- 日期与范围：2026-10-28；`chapters/01-基础与语言模型.md` 的第 4 讲，从梯度方向、链式法则、梯度下降和三段教学代码，到梯度消失/爆炸、混合精度、裁剪、训练诊断、练习与小结。
- 扫描问题与资料范围：检查学习率稳定性结论是否限定在具体目标函数；链式法则是否从输入走到梯度和一次参数更新；反向传播与自动微分的关系能否由代表论文和固定框架版本支持；FP16 下溢、BF16/FP16 取舍、loss scaling 与梯度裁剪边界是否准确。阅读 Rumelhart、Hinton 和 Williams 1986 年论文的 Nature 摘要及元数据；PyTorch Autograd mechanics v2.14 的运算图记录、反向计算章节；PyTorch AMP v2.14 的 Gradient Scaling 小节。论文全文未公开访问，本讲不据摘要判断方法首创权。
- 候选主题与去向：
  - 学习率和收敛：保留现有二次目标，推导误差递推因子 $1-2\eta$ 及 $0<\eta<1$ 的收敛区间；明确这只是 $L(w)=(w-3)^2$ 的条件，不外推至大模型训练。
  - 链式法则手算：补充 $z=2w+1$、$L=(z-3)^2$ 的前向值、局部导数、梯度和一次更新，和已有标量代码思想相对应。
  - 反向传播谱系与自动微分：引用 1986 年代表性论文摘要和 PyTorch v2.14 官方文档，解释权重如何由输出误差反传、前向怎样记录图、反向怎样应用链式法则；不主张该论文首创反向传播。
  - 梯度消失/爆炸与混合精度：保留已有 Jacobian 连乘直觉，将数值格式表述限定为常见范围差异；增加 FP16 小梯度下溢、loss scaling 的作用及其不能保证消除所有溢出的边界。
  - 梯度范数和裁剪：把 gradient norm 定位为诊断信号而非能力分数；补充零范数不做除法，并解释裁剪只限制大梯度、不能修复错误标签或 mask。
  - 未增设独立讲次：上述内容均服务于“如何从损失得到可控参数更新”这一中心问题；技术专题留在后续优化器、模型训练与数值计算内容中。
- 修改决定与结果：保留本讲顺序和原有代码；补充当前二次函数的稳定区间与三个学习率行为、完整链式求导数值例、代表论文与 PyTorch 自动微分实现说明、混合精度风险边界和零范数裁剪条件；局部说明了损失、学习率、雅可比矩阵、BF16/FP16、损失缩放和梯度裁剪术语。
- `length_status`：本讲解释性中文 3,421 字（剔除标题、代码围栏、表格、引用状态行、行内代码与公式后计汉字；保留解释性列表），核心单元，下限 1,800 字，通过。
- `evidence`：Nature 页面元数据与摘要支持该论文所述的连接权重调整及隐藏表示主张，但全文未访问，故不用于历史首创判断。PyTorch Autograd v2.14（2026-08-03 更新）说明前向记录有向无环计算图、反向沿图用链式法则计算梯度；PyTorch AMP v2.14 支持 FP16 梯度下溢、loss scaling/更新前反缩放，以及部分 BF16 预训练模型可能超出 FP16 最大数值范围的警告。来源阅读位置、证据等级和边界见 `research/backprop-autograd-and-mixed-precision-2026.md`。
- `example_status`：链式例输入 $w=0$，得 $z=1,L=4$；反向为 $\mathrm{d}L/\mathrm{d}z=-4$、$\mathrm{d}z/\mathrm{d}w=2$、$\mathrm{d}L/\mathrm{d}w=-8$；取 $\eta=0.1$ 后 $w=0.8,z=2.6,L=0.16$。对二次目标，误差乘数 $1-2\eta$；$\eta=0.05,0.5,1.1$ 分别对应 $0.9,0,-1.2$，解释收缩、一步到最优点和交替发散；梯度 `(3,4)` 的范数为 5，阈值 1 时裁剪为 `(0.6,0.8)`。
- `code_status`：纯 Python 标量、NumPy 学习率对比、PyTorch autograd 三块可执行语句与依赖未变，未运行代码。正文原有验证标记（分别为 Python 3.13.13 标准库、NumPy 2.4.6、PyTorch 2.12.0+cpu）保留；本轮未复验。静态阅读未发现代码与配套文字的输入/输出约定冲突，更新后的二次函数输出通过手算核对；`progress/` 中仍无可追溯的原始命令和完整实测输出，作为历史验证记录缺口保留，不将静态核对表述为执行验证。
- `formula_status`：核对梯度下降更新式、二次函数误差递推、链式法则乘积、梯度范数裁剪式及零范数分支；新数值均已逐步代入，学习率范围明确限定函数条件。
- `link_status`：正文新增 Nature 论文、PyTorch Autograd mechanics v2.14 和 AMP v2.14 直达链接；均已打开并核对支持段落。资料入口同步替换为固定版本 Autograd 链接并增加三条来源说明；无本地相对链接变更。
- `visual_status`：本讲的核心关系可由链式求导步骤和误差递推式直接复算；已有训练 step 顺序及梯度诊断表承担整体流程/诊断导航。本轮未额外加图，避免把单变量计算图重复画成装饰图。
- `sync_status`：未改变讲次结构、章节关系或第一册能力承诺，不改目录、题库或术语表。第一正文文件还含第 5 讲及后续内容，待该文件完成时更新简介中对应章节摘要。
- `open_risks`：Nature 全文未访问，历史优先权不在本讲下结论；PyTorch v2.14 文档只支持相应版本的公开语义；特定 BF16/FP16 稳定性仍需模型、算子与硬件验证；三块旧代码标记缺少可追溯命令/完整输出记录。

## chapter_scan 与精修记录：第 5 讲

- 日期与范围：2026-10-28；`chapters/01-基础与语言模型.md` 的第 5 讲，从 SGD、Momentum、Adam/AdamW 与学习率调度，到三段教学代码、参数分组、失败模式、练习和讲次小结；章末“五讲综合”不计入本讲篇幅。
- 扫描问题与资料范围：核对 Adam 的一阶/二阶矩、偏差修正和状态成本；区分 L2 正则、普通 SGD 权重衰减和 AdamW；检查预热/衰减是否被写成跨架构通则；审查参数分组和恢复训练状态的版本边界。阅读 Adam、AdamW 原始论文元数据/摘要及算法定义，PyTorch AdamW v2.14 算法、参数组与状态加载说明，Xiong 等人的 Post-LN/Pre-LN 摘要，Kosson 等人的小规模 GPT 研究摘要，DeepSeek LLM 报告 §2.3，及 Hugging Face 当前 main 分支翻译训练示例对应参数组代码。具体阅读位置、版本和证据边界见 `research/adam-adamw-and-learning-rate-schedules-2026.md`。
- 候选主题与去向：
  - Adam 矩状态与偏差修正：作为核心机制纳入，给出 $m_t$、$v_t$、偏差修正及坐标更新的完整关系，并用状态内存估算说明代价。
  - L2 正则与权重衰减：纳入同一解释链，展示正则项如何加入梯度、普通 SGD 下的等价形式，以及为何自适应缩放会使 Adam 中的两种写法不同；以 AdamW 更新式落地。
  - AdamW 的证据与定位：作为常见比较基线介绍，但收窄其论文结论和项目实例，不写成所有 LLM 的普遍最优选择。
  - 学习率预热：纳入架构相关边界；Post-LN/Pre-LN 结论限定在 Xiong 等人所研究的架构与设定，小规模 GPT 研究也不外推为生产规模通则。
  - 衰减调度：保留常数、线性、余弦和阶梯形式，并按 DeepSeek LLM 的一项报告说明 multi-step 只是在该项目设置下的选择。
  - 参数分组与 checkpoint 恢复：纳入本讲，Hugging Face 参数名只作为可变示例；PyTorch v2.14 状态加载顺序限定为该版本公开接口。
  - Adafactor、Lion、Sophia 等优化器的机制比较不在本册主线展开，交由[第五册优化器专题](../../book-05-llm-training/chapters/06-优化器与学习率策略.md)；Muon 与 AdamW 的架构参数分工交由[第二十一册相应章节](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)。两个入口已确认文件存在。保留核心训练闭环，不把基础讲次扩成优化器目录。
- 修改决定与结果：补足 Adam 偏差修正和矩状态成本；用二次目标手算 Momentum；推导 L2/decoupled decay 区别；收窄 AdamW 基线、warmup、参数分组和恢复状态结论；添加标量 AdamW、两坐标 Adam 式 L2/AdamW 和 warmup+cosine 的手算；把重复公式段改为配置、检查点和参数组的操作衔接；新增第五册和第二十一册正文入口。简介已补上基础与语言模型部分的短摘要。三段可执行代码均未改动。
- `length_status`：本讲解释性中文 4,583 字（按剔除标题、代码围栏、表格、引用状态行、行内代码和公式后计汉字；保留解释性列表），核心单元，下限 1,800 字，通过。
- `evidence`：正文公式依据 Adam 和 AdamW 原始论文算法定义；PyTorch v2.14 支持 AdamW 公开更新和状态加载语义；Xiong 2020、Kosson 2024 限定预热机制的研究范围；DeepSeek LLM v1 §2.3 支持该报告的 AdamW/multi-step 配置与作者报告结论；HF main 示例仅支持一份具体参数分组实现。事实和版本边界见同名 research 记录；正文没有将作者报告外推为行业通则。
- `example_status`：Momentum 输入 $g_1=g_2=1$、$\beta=0.9$、$\eta=0.1$，得到 $u_1=1,u_2=1.9$，两步累计更新 0.29，对比 SGD 的 0.20。AdamW 单步例输入 $\theta_0=1,g_1=0.1,m_0=v_0=0$、$\beta_1=0.9,\beta_2=0.999,\eta=0.001,\lambda=0.01,\varepsilon=10^{-8}$，得到 $m_1=0.01,v_1=10^{-5},\hat m_1=0.1,\hat v_1=0.01,\theta_1\approx0.9989900001$。NumPy 二坐标例输入 $\theta=(1,1),g=(0.1,0.1),v=(0.01,100),\eta=\lambda=0.1$，Adam 式 L2 得约 $(0.800,0.998)$，AdamW 得约 $(0.890,0.989)$；明确它省略一阶矩/偏差修正。调度例 $T=10000,T_w=1000,\eta_{max}=3e-4,\eta_{min}=3e-5$，手算 step 500 为 $1.5e-4$，step 5000 进度 $4/9$、余弦因子约 0.586824、学习率约 $1.8844e-4$；所有数值均标为教学输入。
- `code_status`：三段代码的可执行语句、输入输出、依赖和声明环境未改变，沿用正文原有“已验证（2026-10-07；Python 3.13.13；标准库/NumPy 2.4.6）”标记，本轮未运行。现有 `progress/` 没有找到原始命令与完整实测输出，保留该历史可追溯性缺口，不将静态手算表述为执行验证。
- `formula_status`：核对 Momentum 递推、Adam 偏差修正、L2 梯度附加项、AdamW 的 decoupled update，以及 warmup/余弦调度边界；输入和中间量均在正文解释。整篇复读时再次检查了本文件公式围栏配对和讲次内关键变量衔接。
- `link_status`：新增 Adam、AdamW、PyTorch v2.14、Xiong、Kosson、DeepSeek LLM 和 Hugging Face 示例的直达来源；新增第五册和第二十一册相对链接，目标文件存在。Hugging Face `main` 可变，正文和 research 均记录了阅读日期及边界。
- `visual_status`：用二坐标更新表显示自适应缩放与解耦衰减的差异，用 warmup/cosine 表呈现若干步数的学习率；两表均在正文前后解释读法。本轮未添加装饰图。
- `sync_status`：完成第一正文文件五讲后，在 `book-01-core-30/简介.md` 补入该部分核心问题、作用和向 Transformer 核心部分的交接。目录结构未变；配套资料未因机制例子调整而需要同步。整篇主路径复读已完成。
- `open_risks`：三个旧代码块有已验证标记但缺少历史原始命令/实测输出日志；本轮没有代码变化，不复验。第五册专题详细优化器比较和第二十一册 Muon 案例尚未在其自身章节审阅范围内重新审计；本讲入口按当前仓库路径设置，后续如目录结构改变需同步。

## 第一正文文件整篇复读

- 日期：2026-10-28；沿 `chapters/01-基础与语言模型.md` 从本部分路线读到“五讲综合”、资料入口和本章小结，复读范围覆盖全文件 1–2539 行。
- 章节主线确认：第 1 讲解释预测目标及能力边界；第 2 讲将概率目标落实到标签对齐、训练与生成；第 3 讲把目标转成 loss 和 PPL；第 4 讲从损失推到梯度与诊断；第 5 讲讨论参数更新、正则和调度；章末综合用一条训练排查链收束。与目录和第一册定位相符，不需要移动、拆分或合并讲次。
- 复读时修订：把“现代大模型通常使用 AdamW”改为本讲选作代表性基线，保留具体公开配置和范围；将 warmup 与 Adam 矩估计不准的单因果说法收窄；去掉第 5 讲重复的 Adam/AdamW 公式，改为检查点、参数组和可复现配置衔接；增加第五册优化器专题及第二十一册 Muon 架构案例入口。简介已同步第一部分摘要。
- 导航与呈现：第一正文文件的 5 个相对链接均能定位到文件；Markdown 三反引号围栏计 230 个、配对；1 个 Mermaid 流程图沿用标准 `flowchart LR`、引号标签和管道箭头。`git diff --check` 通过。没有运行测试、正文代码或 PDF 构建。
- 结果：文件级复读通过；第一册仍有第 6–38 讲及整册终审未完成，不能标记书册完成。

## chapter_scan 初筛：第二正文文件（第 6–15 讲）

- 日期与范围：2026-10-28；基于 `chapters/02-transformer核心.md` 的本部分定位、十讲承接表和讲次标题，检查当前 decoder-only Transformer 主线覆盖及跨册边界；本轮尚未通读该文件正文，具体结论须随讲次精修核实。
- 扫描问题与资料范围：评估核心序列是否从 token IDs、embedding/position、attention、MHA/GQA、因果训练、Block、Norm/残差、RoPE 连到小型 GPT；并检查 GQA/KV cache、attention IO 优化、RoPE 长度边界和 Transformer 替代架构是否需要改变本册骨架。阅读 GQA、FlashAttention、SentencePiece、RoFormer、Mamba 与 Mamba-2/SSD 原始论文的摘要和相关方法段；来源版本、阅读位置和证据边界见 `research/transformer-core-mechanisms-2026.md`。
- 候选主题与暂定去向：
  - 子词模型：BPE、WordPiece、SentencePiece 需要讲清算法与工具边界；SentencePiece 论文也包含 unigram language model。作为第 6 讲中的短比较候选，不增独立讲次；在该讲通读时核对实际覆盖和 tokenizer 输入契约。
  - GQA/MQA：保留在第 10 讲主线，因为 KV head 数直接影响 KV cache 形状和读取量。原论文报告的是论文条件下的质量/速度折中，不写成跨模型保证。
  - FlashAttention/attention IO：标准 attention 的数学计算与硬件 IO 优化是不同层次；候选是第 9 讲保留简短机制定位，GPU tiling、kernel 和性能数字移交第二十一册 FlashAttention 专题，避免本册扩成系统书。
  - RoPE 与长上下文：位置旋转和相对位移机制留在第 14 讲；最大窗口、外推、缩放方法和长上下文评测的深入比较交给第二十一册 RoPE/长上下文专题。
  - Mamba、SSM 与 SSD：纳入扫描视野，确认其作为独立或混合序列架构方向的持续价值；不并入 decoder-only Transformer 基础链，不改本册骨架，转至第二十一册的 SSM、替代架构和架构演进章节。SSD 与部分 structured attention 的联系不能写成“标准 softmax attention 与任意 SSM 等价”。
  - 小型 GPT：保留第 15 讲，作为前九讲机制的程序化闭环；不以一段代码替代前文的数值推演。
- 架构决定：当前目录和简介已经承诺完整的 decoder-only 前向路径，十讲顺序形成“输入—表示—信息路由—层结构—位置—实现”递进；本轮未发现足以增章、拆分、合并或重排的证据。候选去向为初筛结果，待逐讲审阅确认。
- 跨册入口候选：第二十一册现有 MHA/MQA/GQA/MLA、FlashAttention、RoPE/长上下文、SSM 与替代架构章节；第五册承接训练架构与优化，第六册和第二十四册承接服务与推理系统。当前只确认相应文件路径存在，不将其尚未逐章审计的内容视为本册已验收。
- 逐讲精修已从第 6 讲推进至第 18 讲；第二正文文件第 6–15 讲已精修、更新简介中的九章阅读提要并完成连续复读，第三正文文件第 16–18 讲已精修并完成连续复读，简介中的第三正文文件提要已同步。

## open_risks 与下一步

- 第 1–18 讲已完成本轮定向扫描与精修；第 19–38 讲仍未完成各自的章级扫描和候选去向判断。
- 第 1 讲计数为 5,258 字，第 2 讲为 3,894 字，第 3 讲为 4,011 字，第 4 讲为 3,421 字，第 5 讲为 4,583 字，第 6 讲为 2,083 字，第 7 讲为 2,064 字，第 8 讲为 1,852 字，第 9 讲为 1,951 字，第 10 讲为 2,008 字，第 11 讲为 2,093 字，第 12 讲为 1,917 字，第 13 讲为 1,995 字，第 14 讲为 1,936 字，第 15 讲为 1,860 字，第 16 讲为 3,365 字，第 17 讲为 3,849 字，第 18 讲为 2,840 字；第 19–38 讲尚未按规范计数。
- 第 1–18 讲已逐讲记录例子、公式、代码状态、来源支持、链接和配套同步；第一、第二、第三正文文件均已完成主路径连续复读。
- 研究来源文件在 `research/`；正文链接仍需随整章和全册导航 QA 复核，不假定维护记录链接等同读者入口。
- 下一工作单元：开始 `chapters/04-后训练与对齐.md` 第 19 讲的章级扫描；第一册整册复读仍待第 19–38 讲完成后进行。

## chapter_scan 与精修记录：第 6 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 6 讲，覆盖 token/ID 关系、切分粒度、BPE/WordPiece/Unigram/SentencePiece、聊天特殊 token、输入探针、token 预算及词表/embedding 兼容；不含第 7 讲。
- 扫描问题与资料范围：检查 UTF-8 字节计数是否适用于示例字符；BPE 小例子能否手算且没有冒充真实训练流程；WordPiece、Unigram 和 SentencePiece 是否清楚区分；聊天模板、特殊 token 与输入 ID 的接口主张有无普适化；4,096-token 窗口的整数请求容量是否计算正确。阅读 RFC 3629 §3、Sennrich 等人的 BPE 论文 v5 §3.2、BERT 论文 §3、SentencePiece 论文 §2–3/结论，以及 Transformers v5.17.0 官方 tokenizer algorithms 和 chat templates 文档。新知识与证据边界见 `research/tokenization-input-contract-2026.md`。
- 候选主题与去向：
  - UTF-8 与 byte-level tokenizer：在现有例子中准确限定“模”“型”各 3 字节；说明原始字节 tokenizer 和 byte-level BPE 合并字节串后的区别，不扩成 Unicode 教程。
  - BPE、WordPiece、Unigram：在本讲保留一段机制比较及来源；SentencePiece 说明为支持 BPE/Unigram 的原始文本工具。更完整的 tokenizer 训练、语言专项和模型词表设计不扩成独立讲次。
  - 聊天模板与特殊 token：保留角色、控制流和停止 ID 作为输入契约内容；补充模板因模型而异及二次编码可能重复添加特殊 token 的限定，不把一个框架文档外推为所有 runtime 的协议。
  - token 窗口预算：保留明确标注的教学请求长度，改成“请求长度当量”并向下取整为 5/3 个完整输入，同时注明其他上下文与输出也消耗窗口。
  - 未发现需改变第 6–15 讲骨架或增设独立讲次的候选；BPE/WordPiece/Unigram 支持内容仍服务于“文本如何变成输入 ID”这一问题。
- 修改决定与结果：保留讲次结构；用 RFC 3629 限定示例中两个具体汉字的 UTF-8 字节数；补出 WordPiece 的频率比值评分思路和 SentencePiece 对 BPE/Unigram 的支持；为聊天模板补入固定版本官方文档；修正 token 窗口商数的解释及整数容量；压缩重复的输入契约总结，把收束段聚焦于更换 tokenizer 时词表与 embedding 的兼容性。
- `length_status`：修改后解释性中文 2,083 字；逐讲脚本剔除标题、表格、围栏代码和行内代码后计汉字，核心单元，下限 1,800 字，通过。
- `evidence`：RFC 3629 支持 UTF-8 的 1–4 octet 规则和 U+0800–U+FFFF 的三字节范围；BPE、WordPiece、Unigram 和 SentencePiece 的说法分别受原始论文与 Transformers v5.17.0 官方文档限定。聊天模板事实限定于 Transformers v5.17.0 文档。来源的证据等级、支持范围与未验证项见研究记录；未测量任何真实 tokenizer 或在线服务。
- `example_status`：字符级“模型”=2 个 token；两个具体汉字按原始 UTF-8 字节切分为 6 个字节 token；含“模型”的假设子词词表得到 1 个 token。BPE 小语料 `a b a b` 有 `ab` 两次、`ba` 一次，合并 `ab` 后为两个符号。ID 序列 `[12,44,9,31]` 明确是教学编号。预算例 `4096/700≈5.85`、`4096/1100≈3.72` 为长度当量；只按完整输入取整为 5/3，现实中还要扣除模板、系统消息、历史和输出预算。
- `code_status`：本讲没有可执行代码；文本围栏仅呈现 token 序列。无代码运行或 PDF 构建。
- `formula_status`：无数学公式；字节数与窗口预算为整数/除法算例，计算口径已核对。
- `link_status`：新增 RFC 3629 官方标准、Transformers v5.17.0 tokenizer/chat-template 文档直达链接；BPE、BERT、SentencePiece 论文链接均对应所述结论并已阅读相关章节。本讲没有本地相对链接。
- `visual_status`：保留切分粒度对比表和 BPE 逐轮计数表；两者对应本讲的分类与手算过程。未加独立图示；本部分总路径图已在讲前提供。
- `sync_status`：讲次边界、章节关系和目录未变；本文件第 7–15 讲完成后再更新简介摘要。无需改动题库、练习或术语表。
- `open_risks`：不同 tokenizer 的规范化、词表映射、控制 token 和窗口接口仍依模型及版本而异，本讲不声称实测或普适行为；后续讲次还需复核上下文长度及 embedding 兼容关系。

## chapter_scan 与精修记录：第 7 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 7 讲，从 embedding 查表、位置注入、padding/位置 ID、权重共享到绝对/相对位置方案和长度边界。
- 扫描问题与资料范围：核对 `[B,T]→[B,T,D]` 查表和位置相加的 shape、手算参数、0-based/1-based 位置编号、causal mask 与位置线索的区别、权重共享下梯度路径、固定正弦位置数例，以及 learned absolute/relative/ALiBi/RoPE 的作用位置和实验边界。阅读 Vaswani 等人原始 Transformer v7 §3.4–3.5、Shaw 等人相对位置论文 v2 §3.1–3.3、Press 等人 ALiBi 论文 v2 §1/§3/§4；来源阅读范围和证据边界见 `research/embedding-and-position-representations-2026.md`。
- 候选主题与去向：
  - Embedding 查表、训练梯度与输入/输出权重共享：保留为从 ID 到初始表示的核心机制；权重共享写成模型选择，并说明输入查表梯度和标准全词表 softmax 梯度的路径不同。
  - 学习式绝对、正弦、相对位置与 ALiBi：保留短比较，指出方法改变的位置路径；不把单个论文的性能或外推结论写成普遍性质。
  - RoPE：本讲只说明旋转 Q/K 的定位和阅读入口；详细推导、扩展方法与长上下文评估移交第 14 讲及第二十一册专题，不重复完整主讲。
  - 长度外推与有效上下文利用：在本讲明确分开 position-table 越界、运行时窗口和任务能力；深入评测留给第 14 讲/第二十一册，不增设本讲次。
  - 未发现需要改变第 6–15 讲骨架或独立增讲的候选；本讲继续承担从 token ID 到带位置初始表示的过渡。
- 修改决定与结果：明确查表示例的 1-based 展示次序和位置向量的 0-based 下标；删除重复的 token 交换回顾；补充 Q/K/V 的首次释义、权重共享的具体来源及未输入词表行可能从完整 softmax 得到梯度的限定；给相对位置表示和 ALiBi 增加机制区别及论文入口；将 4,096 行位置表示例改为零基索引；收窄正弦编码对外推的表述。
- `length_status`：修改后解释性中文 2,064 字；脚本剔除标题、表格、围栏代码和行内代码后计汉字，核心单元，下限 1,800 字，通过。
- `evidence`：原始 Transformer v7 §3.4–3.5支持该模型采用 embedding/softmax 权重共享、embedding 缩放及正弦位置编码加入输入的设计；固定正弦“可能外推”是论文动机，不是一般保证。Shaw v2 §3.1–3.2支持一种对 Q–K 匹配与 Value 汇总加入相对距离表示的具体实现；ALiBi v2 §1/§3支持向 Q–K score 加距离线性惩罚。实验结果限定于论文自己的 WMT 或语言模型设置。未测量任何真实模型的 position IDs、weight tying、padding 或长度能力。
- `example_status`：查表例以 `E` 的三行假设参数把 `[1,3]` 映射为 `[1,3,2]`；第二条输入使 batch shape 成 `[2,3,2]`。位置表算出三个初始向量 `(0.3,-0.1)`、`(-0.3,0.6)`、`(0.4,0.3)`；交换前两 token 后，ID 12 在位置 1 的结果 `(0.2,0.1)` 与位置 0 不同。另算 `V=100,D=2` 共 200 个 embedding 参数；正弦示例的两个位置向量分别为 `[0,1]` 与约 `[0.8415,0.5403]`。这些是教学假设数。位置表 4,096 行使用 0-based ID 时，索引 4,096 是第 4,097 个位置并越界。
- `code_status`：无可执行代码；text/math 围栏只承载张量数例和公式，未运行代码。
- `formula_status`：检查 `$E\in\mathbb{R}^{V\times D}$` 查表、位置向量逐行相加和正弦数例；`V,D,B,T` 使用本部分统一定义，位置向量 `P_i` 按零基序号定义。shape 与中间量一致。
- `link_status`：正文引用 Attention Is All You Need v7 §§3.4–3.5、Shaw 等人 v2 §§3.1–3.2、ALiBi v2 §3 的直达 HTML 章节；相关页面已打开并核验。无本地相对链接。
- `visual_status`：保留 embedding 查表表、位置向量相加表、位置方法对比表和正弦两位置算例；均直接呈现本讲机制，不需再加 Mermaid 图。
- `sync_status`：未改讲次结构、目录或书册边界；本文件第 8–15 讲完成后更新简介摘要和入口。无需同步题库或配套索引。
- `open_risks`：不同模型的 position ID、padding 处理、embedding 权重共享及位置扩展接口不统一；绝对位置表或相对位置方法不能单独证明长上下文有效利用。具体行为待对应模型版本和任务验证。

## chapter_scan 与精修记录：第 8 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 8 讲，从 Self-Attention 的 Q/K/V 直觉、加权 Value 汇总，到并行性、序列长度成本、KV Cache 前置、因果遮罩热图读法和残差流更新；不含第 9 讲的 scaled dot-product 计算推导。
- 扫描问题与资料范围：核验 attention 权重究竟表达什么、权重图能否直接作为模型解释、并行性和复杂度比较应如何限定、因果关系数与实际内存优化是否混淆，以及自回归生成、KV Cache 和残差更新的讲解边界。阅读 Vaswani 等人的 Transformer v7 §4、Dao 等人的 FlashAttention v2 §§2.2–3.3、Jain 与 Wallace 的 Attention is not Explanation v3 摘要/§§1、4.1–4.2、6，以及 Wiegreffe 与 Pinter 的 Attention is not not Explanation v2 摘要/§§1–5；具体来源位置、研究事实和证据限制见 `research/attention-mechanism-and-interpretation-2026.md`。
- 候选主题与去向：
  - Q/K/V 与按权重混合 Value：作为本讲核心，保留小向量手算并分别改变 Value 与权重，明确权重系数和被汇入内容是两条作用路径；Q/K 点积和 softmax 的完整算法移交第 9 讲。
  - Attention 权重的解释性：纳入有范围的支持性讨论。Jain/Wallace 与 Wiegreffe/Pinter 的结论存在方法论分歧，且主要证据来自 RNN 注意力任务；正文因此将单层单头权重图限定为局部路由线索，不外推为所有 Transformer/LLM 的解释结论，并建议用干预检验输入影响。
  - 并行、长度复杂度与 FlashAttention：保留 T=4、T=8 的因果关系计数及一般三角数公式；区分逻辑位置对数量、标准 dense 注意力与 exact FlashAttention 的 HBM IO 优化。GPU tiling、kernel 实现和性能测量继续交给第二十一册专题，不在本讲展开为系统分析。
  - 训练并行与自回归生成、KV Cache：保留一次前向可并行而输出逐 token 生成的区别，并给出复用过去 K/V、新 Query 仍读取历史的过渡；缓存 shape 和内存核算交第 10 讲。
  - 残差流：保留条件明确的二维玩具加法，说明注意力输出是更新量而不是替换输入；完整 Block、归一化顺序和 MLP 留给第 12–13 讲。
  - 未发现需要改变第 6–15 讲骨架或新增独立讲次的候选；第 8 讲继续承担直觉桥梁，后接第 9 讲数值算法。
- 修改决定与结果：说明 Q/K/V 是学习投影而非字面查词；用三个可复核向量例子区分固定权重改变 Value、不同权重得到相同输出以及单行权重图的解释边界；补充 RNN 注意力解释性研究的分歧和范围；按 T=4/T=8 核对因果关系数，区分逻辑算量与 IO 优化；删除重复的复杂度、权重解释和信息路径总结；加入 KV Cache 数据流提示、注意力热图轴向读法和带条件的残差加法示例。讲次结构未改。
- `length_status`：修改后解释性中文 1,852 字；剔除标题、表格、围栏内容、行内代码及链接目标后统计汉字，核心单元，下限 1,800 字，通过。
- `evidence`：Transformer v7 §4 支持原论文标准层的复杂度与顺序运算比较；FlashAttention v2 支持 exact attention 的 tiling/recomputation 与 HBM IO 优化，不支持把全连接关系数说成线性；两篇解释性论文支持其 RNN 任务和方法论范围内的争论，不能外推为所有 Transformer/LLM 的结论。未测量实际模型注意力或 kernel 性能。
- `example_status`：假设权重 `[0.2,0.3,0.5]` 与 Value `(2,0),(0,2),(1,1)` 得到贡献 `(0.4,0),(0,0.6),(0.5,0.5)`，合计 `(0.9,1.1)`；固定权重把第二个 Value 改为 `(0,4)`，输出成为 `(0.9,1.7)`。令前两个 Value 同为 `(1,0)`、第三个为 `(0,1)`，两组不同权重 `[0.2,0.3,0.5]` 和 `[0.4,0.1,0.5]` 都输出 `(0.5,0.5)`。长度 4/8 的因果位置对数分别为 10/36。残差玩具例取 `x=(1,0), o=(0.9,1.1)`，假设输出投影为单位映射并忽略归一化、dropout，得到 `(1.9,1.1)`；均在正文标为教学假设。
- `code_status`：无可执行代码；未运行代码或测试。
- `formula_status`：复核向量加权和、softmax 权重行和为 1、因果位置对数 `T(T+1)/2` 在 T=4/8 时分别为 10/36；示例中间量和残差假设已在正文说明。
- `link_status`：四篇来源的直达论文版本链接均已打开并核对支持位置；本讲无本地相对链接。
- `visual_status`：加权贡献表呈现逐项求和，复杂度段展示 T=4/T=8 关系数；热图读法明确行 Query、列 Key 和因果 mask。未新增 Mermaid 图，现有表格已直接表达核心数值与矩阵关系。
- `sync_status`：未改变讲次结构、目录、跨册分工或第一册能力承诺；第 6–15 讲完成后更新简介摘要和入口。缓存细节已有第 10 讲交接，本轮不改配套题库或索引。
- `open_risks`：注意力解释研究结论仍受 RNN 任务和诊断方式限制；真实模型的缓存接口与 fused-kernel 执行行为依实现、版本和硬件而异，需在第 10 讲及专题范围内继续核验。本讲玩具残差例省略完整 Transformer Block 的归一化、dropout 和投影差异，正文已明示假设。

## chapter_scan 与精修记录：第 9 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 9 讲，从 Q/K/V 投影 shape、scaled dot-product attention、softmax 数值计算与稳定性，到因果 mask、全屏蔽行边界、dense attention 复杂度及 FlashAttention/稀疏/linear attention 的简要对照；不含第 10 讲的多头结构展开。
- 扫描问题与资料范围：核对缩放项方差推导的假设、投影和拆头后的 shape、softmax 归一化维度、因果 mask 行为、全屏蔽行的数学定义、FlashAttention 的 exact 与 IO 含义，以及 sparse/linear 方法是否被误说成只改速度。阅读 Vaswani 等人的 Transformer v7 §§3.2.1、3.2.3，Dao 等人的 FlashAttention v2 §§2.2–3.3，Beltagy 等人的 Longformer v2 摘要/§3.1，以及 Katharopoulos 等人的 Linear Transformer v2 §3.2.1；来源事实与版本边界见 `research/scaled-dot-product-attention-2026.md`。
- 候选主题与去向：
  - Scaled dot-product 主计算链：作为本讲核心，保留投影、拆头、点积、按行 softmax、加权 V 的 shape 和数值推导；输出投影与并行头结构交第 10 讲。
  - $1/\sqrt{d_h}$ 缩放：保留独立、零均值、单位方差假设下的方差推导及 d_h=2/64 的教学例；将其写作原始论文设计动机，不泛化为真实网络分布定律。
  - Causal mask、padding 和数值稳定：保留 -∞ 的理论屏蔽、有限哨兵差别、行内减最大值的 softmax 计算以及全屏蔽行无定义边界；不假设所有框架/kernel 对无效行采取同一处理。
  - FlashAttention、local/sparse 和 linear attention：作为有边界的支持性比较，区分同一 attention 数学目标的 IO 优化、改变可见位置的稀疏结构、以及用 feature map/重排求和的线性方法；性能/质量深入比较交给第二册及第二十一册，不新增本册讲次。
  - 未发现改变第 6–15 讲骨架或新增独立讲次的候选；第 9 讲仍衔接第 8 讲的直觉例并为第 10 讲多头拆分提供算法基础。
- 修改决定与结果：明确定义 B/T/D 和标准 MHA 的投影矩阵宽度，并说明 GQA/MQA 的 KV 投影宽度差异；补全 M 可广播到 score shape 的约束；保留可复算的单 Query、双 Query 矩阵、causal mask、稳定 softmax、高维缩放及复杂度例子；纠正原例中“第二 Query 只看第一个 Key”的因果遮罩错误，改为两位置序列的第一行屏蔽未来 Key；压缩重复的双 Query 数值、全屏蔽行和有限 mask 哨兵说明；修正原论文章节锚点，加入 FlashAttention、Longformer 和 Linear Transformer 版本链接。连续复读时又发现并删除一段重复的 softmax 稳定性说明，合并精度边界；讲次结构未改。
- `length_status`：连续复读去重后解释性中文 1,951 字；计数剔除标题、表格、代码/公式围栏、行内代码/数学和链接目标后统计汉字，核心单元，下限 1,800 字，通过。
- `evidence`：Transformer v7 §§3.2.1/3.2.3 支持 scaled dot-product 定义、方差动机和非法连接设为负无穷；FlashAttention v2 §2.2 支持标准实现物化 $N\times N$ 中间矩阵，§3.1 支持 tiling/recomputation 与减少 HBM 访问；Longformer v2 §3.1 支持其窗口/全局可见模式；Linear Transformer v2 §3.2.1 支持该论文 feature map 方法及 exponential softmax kernel 的精确有限维映射限制。性能结论限于各论文条件；未运行或测量 kernel。
- `example_status`：单 Query `q=(1,0)`、`k_1=(1,0)`、`k_2=(0,1)`、`v_1=(2,0)`、`v_2=(0,4)`、$d_h=2$，点积 `[1,0]`，缩放 `[0.7071,0]`，exp 稳定化 `[1,0.4931]`，归一权重 `[0.6698,0.3302]`，输出约 `(1.3396,1.3208)`。第二 Query `(0,1)` 得权重 `[0.3302,0.6698]` 和输出 `(0.6604,2.6792)`，二者矩阵化后逐行归一化。长度 3 的位置 2 对 `[1,2,100]` 加 causal mask 后是 `[1,2,-∞]`，权重约 `[0.269,0.731,0]`；长度 2 的第一 Query 只见 `k_1`，输出 `(2,0)`。稳定 softmax `[1000,999,998]` 先减最大值得 `[0,-1,-2]`，概率约 `[0.6652,0.2447,0.0900]`。d_h=64 的分数示例 `[8,0]` softmax 约 `[0.9997,0.0003]`，除以 8 后 `[1,0]` 得 `[0.7311,0.2689]。数值均为教学输入。
- `code_status`：无可执行代码；围栏仅含公式和矩阵文本，未运行代码或测试。
- `formula_status`：核对 Q/K/V 投影与拆头 shape、$QK^\top/\sqrt{d_h}$、softmax 按 Key 轴归一化、mask 的 1-based 条件、方差假设、双 Query 输出矩阵和 $O(BT^2D)$/$O(BHT^2)$ 的适用范围；所有数例的中间步骤已展示。
- `link_status`：新增 Transformer v7 §3.2.1/§3.2.3、FlashAttention v2 §2.2/§3.1、Longformer v2 §3.1 和 Linear Transformer v2 §3.2.1 直达 HTML 链接；锚点经来源目录核对。第二册相对链接目标存在；本讲无其他本地相对链接。
- `visual_status`：保留 Q/K/V shape 表、双 Query 的 score 矩阵、4×4 因果 mask 表和 sparse/linear/IO 方法对比表。它们分别承担张量对应、按行 softmax、可见位置和方案差异说明；未加额外 Mermaid 图。
- `sync_status`：无讲次结构、目录或书册边界变化；第 6–15 讲全部完成后更新简介摘要。无需同步题库、术语表或其他索引。
- `open_risks`：全屏蔽行的处理及低精度哨兵行为依具体 kernel、dtype 和 runtime；FlashAttention 的实测速度/内存收益未在仓库复现，正文未写具体性能数字；线性/稀疏注意力的机制与质量比较留给专题书。

## chapter_scan 与精修记录：第 10 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 10 讲，覆盖多头投影、各轴 shape、输出拼接、MHA/GQA/MQA 头数关系、参数量变化、增量生成的数据流和 KV Cache 容量估算；不含第 11 讲的标签右移与因果训练目标。
- 扫描问题与资料范围：核验每头投影/输出投影维度、GQA 的组与头数关系、MQA/GQA 节省的是哪些张量、实验中质量/速度结论的适用范围，以及 KV Cache 的元素数、单位、batch 和变长序列口径。阅读 Vaswani 等人的 Transformer v7 §3.2.2、Shazeer 的 MQA v1 摘要/§§2.4、3、4.1–4.3，以及 Ainslie 等人的 GQA v1 摘要/§§2.1–2.2、3.1–3.3、5；来源与证据边界见 `research/transformer-core-mechanisms-2026.md`。
- 候选主题与去向：
  - MHA 的多组投影和输出重组：作为本讲核心，保留从 [B,T,D] 投影到头维、每头独立 attention、concat 和 $W_O$ 回到 D 的 shape 路径；不将头赋予固定语义功能。
  - MHA→MQA→GQA 的历史与结构折中：纳入共享 KV 头数的递进解释；把作者在不同模型和训练条件下观察到的质量/速度变化明确限定为论文证据，不写成通用保证。
  - KV Cache 的数据流与容量：保留“每层生成新 K/V、追加旧 cache、下步复用”的逐 token 轨迹，推导 `2×L×T×H_kv×d_h×b` 并做数值换算；分页分配、量化布局、生命周期和 serving 调度深度交第六册/第二十四册。
  - 参数量与运行时缓存：分别说明 MQA/GQA 会缩小 K/V 投影宽度和活动 cache，但不减少 Query 头计算，也不改变 MLP；端到端收益需要受控 profile。
  - 未发现改变第 6–15 讲骨架或增设新讲次的候选；本讲继续承接第 9 讲 scaled dot-product，连接第 11 讲因果训练与生成。
- 修改决定与结果：定义 H/H_q/H_kv/d_h 及标准 MHA 关系，补足每头投影和 $W_O$ shape；保留 MHA/GQA/MQA 对照表和 Mermaid 数据流；把 MQA 与 GQA 的论文结果拆开限定，补充其不同实验设置下的质量/稳定性证据；核对 D=8 的 KV 元素数和投影参数数；新增一般参数量表达式、解码 token 的 Query/KV shape 轨迹、cache 逐项容量推导和变长 batch 的活动 token 口径；压缩重复的头语义、拼接和缓存结论。讲次结构未改。
- `length_status`：修改后解释性中文 2,008 字；剔除标题、表格、围栏内容、行内代码和链接目标后统计汉字，核心单元，下限 1,800 字，通过。
- `evidence`：Transformer v7 §3.2.2 支持每头独立投影、拼接和输出投影；Shazeer 的 MQA 论文支持多个 Query 头共享一组 K/V、增量解码反复读 K/V 的带宽动机及 WMT14/Billion Word 实验结果；Ainslie 的 GQA 论文支持按组共享、MHA/MQA 两端关系及其 T5.1.1 实验和 MQA 稳定性观察。实验结论局限于各论文模型、任务和硬件；本文 cache/参数数字是按明确维度构造的教学计算，不是实测。
- `example_status`：D=8、H_q=4、d_h=2 时输入 [1,3,8]，标准 MHA 的 Q/K/V 为 [1,4,3,2]，每头输出 [1,3,2]，拼接后 [1,3,8]。T=3 时 MHA/GQA(H_kv=2)/MQA 的 K/V 总元素数为 48/24/12。忽略 bias 的 Q/K/V/O 参数数为 256/192/160；一般投影账本为 $2D^2+2D H_{kv}d_h$。新 token 的 GQA Query shape [1,4,1,2]，K/V 新增 shape [1,2,1,2]，追加后三位置 cache 变为 [1,2,4,2]，四个 Query 输出拼回 [1,1,8]。容量例 L=32、T=8192、H_q=32、H_kv=8、d_h=128、b=2，单 KV 头单位置单层是 512 B，乘得 GQA 1 GiB、MHA 4 GiB、MQA 128 MiB；均为忽略实际 allocator overhead 的教学估算。
- `code_status`：没有可执行代码；数学围栏、表格和 Mermaid 均非代码实验。未运行测试或性能 profile。
- `formula_status`：复核 MHA projection / concat / output shape、GQA/MQA 参数量表达式、`2×L×T×H_kv×d_h×b` cache 公式、48/24/12 元素计数、256/192/160 参数计数、25%/37.5% 子模块缩减率和 1 GiB/4 GiB/128 MiB 单位换算；变长 batch 用 $\sum_sT_s$ 表达逻辑 cache 载荷。
- `link_status`：新增 Transformer v7 §3.2.2、Shazeer v1 §§3/4.2、Ainslie v1 §§2.2/3.2/5 的直达版本链接；第六册和第二十四册 KV Cache 本地相对链接目标均存在。
- `visual_status`：保留头投影和 KV 分组 Mermaid flowchart、MHA/GQA/MQA 对照表及 cache 容量表；flowchart 采用标准 `flowchart LR` 和带引号节点标签，未改变图结构。
- `sync_status`：未改目录、讲次结构或书册边界；第二正文文件第 6–15 讲结束后更新简介短摘要。缓存实现细节留有第六册/第二十四册入口，无需重复到题库或索引。
- `open_risks`：参数和容量例依赖文中声明的均匀层宽、统一 dtype、无 bias/额外状态等简化假设；真实 GQA/MQA 质量、解码吞吐、缓存物理分配和 batch 调度需按模型与 serving 实现评估。


## chapter_scan 与精修记录：第 11 讲

- 日期与范围：2026-10-28；chapters/02-transformer核心.md 第 11 讲，覆盖自回归分解、标签右移、因果 attention mask、token 平均交叉熵、Teacher Forcing、训练并行与逐 token 生成、prefill/decode 与 KV Cache 前置、Causal LM/Masked LM 区别、padding/loss mask、assistant-only 监督、结束标记和变长 batch。
- 扫描问题与资料范围：核对标签移位和遮罩各自约束什么、因果 mask 如何进入 softmax、有效监督 token 的分母、Teacher Forcing 的训练/推理前缀差异、BERT 的双向 MLM 目标、assistant-only loss 的公开接口前提，以及特殊 token、停止配置和 padding 行为的适用范围。阅读 Vaswani 等人的 Transformer v7 §§3.1、3.2.3，Devlin 等人的 BERT v2 §3.1，Bengio 等人的 Scheduled Sampling v3 §§1、2.1–2.4，以及 Transformers v5.17.0 Chat Templates 和 TRL v0.29.0 SFTTrainer 官方文档。来源事实、版本和边界见 research/causal-training-and-loss-masks-2026.md。
- 候选主题与去向：
  - 标签右移和 causal mask：作为本讲核心，分别定义预测目标对齐和 query-key 可见范围；避免重复第 9 讲完整点积推导，使用最小因果约束公式承接。
  - padding attention mask 与 loss mask：保留为调试所需的支持主题，明确前者作用于输入可见性、后者选择目标标签；不假定框架布尔语义或 shift 责任相同。
  - Teacher Forcing 与训练/生成前缀差异：保留一个具体前缀轨迹和 RNN 研究入口；收窄 exposure bias 结论，不把它写成所有长回答失败的诊断，也不罗列未经本讲比较的通用补救方案。
  - BERT Masked LM：保留短对照说明其双向上下文目标不同，不在本讲扩展 MLM 训练细节。
  - assistant-only loss、EOS/结束 token 与聊天模板：保留明确的目标 mask 例子，并将 assistant_only_loss 的能力限定为 TRL v0.29.0 和支持返回 assistant mask 的模板；训练标签监督与推理停止分别讨论。
  - 变长 batch：保留 3/5 token 补齐例，明确 causal mask、padding mask、loss mask 分工及模型/API 版本边界。
  - 未发现需要改变第 6–15 讲骨架、增设讲次或调整跨册分工的候选；第 11 讲仍连接注意力机制与 Transformer Block。
- 修改决定与结果：将因果 LM 概率写为链式分解；用 [12,44,9,31] 的输入/目标对齐表说明三行来自同一次前向；补充加性因果 mask 定义，解释未来权重归零和目标泄漏条件；把 token loss 推广为带 m_i 与有效标签分母的 token mean，并逐项复核三处 NLL；压缩重复的训练/生成例子，保留 prefill/decode 时序图和 KV Cache 交接；新增 Causal/Masked LM 对照、四类对齐/遮罩分工、assistant-only 目标 mask、token/sample reduction 差别和变长 padding 排查；收窄 EOS 与 exposure bias 结论，补上固定版本和原始论文入口。讲次及目录结构未改。
- length_status：修改后解释性中文 2,093 字；计数剔除标题、表格、围栏内容、行内数学和链接目标后统计汉字；核心单元，下限 1,800 字，通过。
- evidence：Transformer v7 §§3.1、3.2.3 支持 decoder 的右移目标与未来连接遮罩；BERT v2 §3.1 支持其双向 Masked LM 目标；Scheduled Sampling v3 的训练/推理差异和错误传播讨论局限于其 RNN 序列任务；Transformers v5.17.0 与 TRL v0.29.0 的模板/API 行为限定于各自固定版本。所有新事实及来源边界在 research/causal-training-and-loss-masks-2026.md。
- example_status：输入 [12,44,9] 对齐目标 [44,9,31]，位置 1/2/3 可见前缀依次为 [12]、[12,44]、[12,44,9]。假设正确标签概率为 0.50、0.25、0.80，负对数似然约 0.6931、1.3863、0.2231，总和 2.3026，token mean 为 0.7675；均为教学输入。assistant-only 例按假设让答案及约定结束标记参与监督，prompt 保留为上下文。
- code_status：没有可执行代码；文本围栏呈现输入/目标示例，MathJax 和 Mermaid 不作为程序执行；未运行训练代码、测试或 PDF 构建。
- formula_status：复核自回归链式分解、1-based query/key 的 C_ij 条件、softmax 后未来权重为零、token mean 的符号和分母定义及 NLL 手算。未运行模型。
- link_status：新增 Transformer v7 §§3.1/3.2.3、BERT v2 §3.1、Scheduled Sampling v3 §§1/2.4、固定版本 Transformers/TRL 直达链接；第六册推理基础相对链接目标存在。来源页面已阅读；外部链接后续仍纳入全库导航 QA。
- visual_status：保留并校准输入/目标对齐表，新增不同 mask 作用对象的对照表；保留 Prefill/逐 token Decode 时序图，Mermaid 源码静态检查通过，尚未在 GitHub 页面确认渲染。
- sync_status：无讲次结构、目录或书册边界变化；第二正文文件第 6–15 讲完成后再更新简介短摘要及入口。无需同步题库、练习、术语表或其他索引。
- open_risks：实际训练框架中的 label shift、padding 语义、loss reduction、结束 token 与生成停止条件仍需按目标模型/API 版本确认；本讲无本地训练或服务实测。Mermaid GitHub 渲染待后续确认。


## chapter_scan 与精修记录：第 12 讲

- 日期与范围：2026-10-28；chapters/02-transformer核心.md 第 12 讲，覆盖 Pre-LN block 的两条残差支路、Attention 与逐位置 MLP 的信息路径、MLP 维度与参数账本、Encoder/Decoder-only/Encoder-decoder 的注意力分支，以及 Block 输出到语言模型头的接口。
- 扫描问题与资料范围：核对原始 Transformer 的子层顺序、残差宽度、MLP 是否跨 token 共享参数、标准 FFN 的维度与激活、门控 FFN 的证据范围、Pre-LN/Post-LN 记号，以及 encoder-decoder 的 Cross-Attention 查询/键值来源。阅读 Vaswani 等人的 Transformer v7 §§3.1、3.2.3、3.3，Xiong 等人的 Layer Normalization 论文 v2 摘要，以及 Shazeer 的 GLU Variants Improve Transformer v1 摘要。事实和来源边界见 research/transformer-block-and-feed-forward-2026.md；Pre-LN 梯度与 warm-up 的更完整研究事实还见 research/adam-adamw-and-learning-rate-schedules-2026.md。
- 候选主题与去向：
  - Attention 与逐位置 MLP：作为本讲核心，沿同一 residual stream 解释先跨位置汇入上下文、再逐位置做共享参数的非线性重组；不重复第 8–10 讲 attention 的内部推导。
  - MLP 中间宽度、标准参数量与门控变体：给出两层线性层的 shape 和参数账本；GLU 只作有范围的技术演进例子，不将论文特定质量结果写成普遍优势。
  - Pre-LN/Post-LN：本讲只标明公式所选变体及原论文的 Post-LN；训练梯度与稳定性由第 13 讲集中分析。
  - Encoder、Decoder-only、Encoder-decoder 与 Cross-Attention：保留短比较表，说明 Query 与 Key/Value 的来源，不扩成 seq2seq 独立章节。
  - 语言模型头：明确 block 输出是 hidden states，词表 logits 由后续 head 产生；不重复第 11 讲的训练目标推导。
  - 未发现改变第 6–15 讲骨架或增设独立讲次的候选；第 12 讲继续承接第 11 讲训练/生成协议并交给第 13 讲归一化和残差稳定性。
- 修改决定与结果：重写去除原稿重复的同一组残差手算、重复 D=8 参数账本及多次 Block 总结；定义 Pre-LN 的两个更新式，并明示省略 dropout、子层内部注意力计算和实际 Norm 数值；用两位置算例逐步计算 Attention 残差、第二 Norm 的假设输入、ReLU 中间值、MLP 更新和最终状态；区分 Attention 跨 token 路由与 MLP 对已带上下文状态的逐位置加工；新增 MLP 维度公式、512/2048 原论文配置范围、GLU 变体的证据边界、参数与中间激活的区别、三类 Transformer 的 Cross-Attention 数据流、logits 接口和 shape 审计。讲次及目录结构未改。
- length_status：修改后解释性中文 1,917 字；计数剔除标题、表格、围栏内容、行内数学和链接目标后统计汉字；核心单元，下限 1,800 字，通过。
- evidence：Transformer v7 §§3.1/3.3 支持原始 residual/LayerNorm 顺序和逐位置共享的 ReLU FFN；§3.2.3 支持 masked decoder self-attention 与 encoder-decoder attention 的 Q/K/V 来源；Xiong v2 只支持其分析设置下 Pre-LN/Post-LN 初始化梯度差异及作者的 warm-up 实验报告；Shazeer GLU v1 摘要只支持其序列到序列实验中部分门控变体相对 ReLU/GELU 的报告结果。均未外推为所有现代模型的统一结构或收益。
- example_status：B=1,T=2,D=2，输入 x_1=(1,0)、x_2=(0,1)，假设 Attention 更新 (0.5,0.25)、(0.25,0.5)，得到 x'_1=(1.5,0.25)、x'_2=(0.25,1.5)。另假设第二 Norm 输出 u_1=(1,0)、u_2=(0,1)，W_1=[[1,-1],[0,1]]、W_2=[[0.5,0.25],[-0.25,0.5]]、bias 为 0；W_1u 经 ReLU 后分别为 (1,0)、(0,1)，MLP 更新为 (0.5,-0.25)、(0.25,0.5)，最终状态为 (2,0)、(0.5,2)，shape 保持 [1,2,2]。Attention 和 Norm 输入明确为构造假设。另算 D=8、D_ff=32、B=1、T=3 的中间激活数 96、标准 MLP 投影权重 512、无 bias 标准 MHA 四个投影权重 256；计数不含 bias、Norm 或门控额外矩阵。
- code_status：无可执行代码；数学围栏、表格和 Mermaid 均非程序实验。未运行代码、测试或模型 profile。
- formula_status：复核 Pre-LN 两次残差 shape、列向量约定下 W_1/W_2 维度和 ReLU 手算、MLP 参数数 2DD_ff、标准 MHA 投影数 4D²，以及中间激活 BTD_ff；假设值与各行残差加法一致。
- link_status：新增 Transformer v7 §§3.1/3.2.3/3.3、Xiong v2 和 Shazeer GLU v1 直达链接；来源页面已阅读，论文适用边界写入 research 记录。正文没有新增本地相对链接。
- visual_status：保留并重写 Pre-LN 数据流 Mermaid 图；保留残差计算表、MLP 中间值表、架构对照表和 shape/参数说明。Mermaid 源码语法静态检查，尚未在 GitHub 页面确认渲染。
- sync_status：未改讲次结构、目录或书册边界；第二正文文件第 6–15 讲完成后更新简介短摘要和入口。无需同步题库、练习、术语或知识图谱。
- open_risks：toy Attention 和 Norm 输出不是完整前向；真实模型的 Norm 顺序、MLP 激活/门控宽度、bias/dropout、交叉注意力和参数配置依具体实现。未做性能或资源 profile；图表的 GitHub 渲染待后续确认。

## chapter_scan 与精修记录：第 13 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 13 讲，覆盖残差直通路径、LayerNorm/RMSNorm 的统计与仿射参数、Pre-LN/Post-LN 的计算顺序、数值例子和训练排查信号。
- 扫描问题与资料范围：核对 LayerNorm 相对 BatchNorm 的原始动机、每 token 特征统计口径、总体方差与 epsilon、RMSNorm 省略去均值的假设、LayerNorm/RMSNorm 对整体平移的响应、原始 Transformer 的 Norm 放置，以及 Pre-LN/Post-LN 梯度路径和 warm-up 结论范围。阅读 Ba 等人 Layer Normalization 原论文摘要、Zhang 与 Sennrich RMSNorm 论文摘要、Vaswani 等人的 Transformer v7 §3.1、Xiong 等人的论文 v2 摘要，以及 PyTorch v2.14 LayerNorm/RMSNorm 官方 API 文档。来源内容与证据边界见 `research/normalization-and-residual-2026.md`。
- 候选主题与去向：
  - 残差结构与 Jacobian 直通项：作为本讲核心，保留一个二维数值例及局部导数解释；不重复第 12 讲完整 Block 前向。
  - LayerNorm 与 RMSNorm：作为同一问题下的两种特征重标定方法，补充原论文动机、总体方差/RMS 手算、平移反事实及仿射参数的作用；具体模型选择和 kernel 性能不在本讲做普遍比较。
  - Pre-LN/Post-LN：保留计算式和链式法则路径对照，收窄 Xiong 等人的理论与实验为论文研究条件下的结论；优化器和 warm-up 的完整策略留给第五册对应章节。
  - BatchNorm 对照：只用作 LayerNorm 的历史问题动机，不扩展成归一化方法百科。
  - epsilon、归一化轴和 dtype：用 PyTorch v2.14 文档作一个明确版本的接口例子，不外推其他框架或模型。
  - 未发现改变第 6–15 讲骨架或新增独立讲次的候选；本讲承接第 12 讲 Block，交接第 14 讲位置表示。
- 修改决定与结果：去重原稿重复的局部扰动、梯度直通和归一化结论；将残差表示为主状态加更新，并解释恒等 Jacobian 路径的帮助与限制；补充 LayerNorm/RMSNorm 公式、四维向量逐步计算、整体平移反事实和 gamma/beta 代入；加入 LayerNorm 相对 BatchNorm 的来源动机、Pre-LN/Post-LN 链式法则比较、固定版本 epsilon/轴/精度约定及逐层排查量。讲次结构未改。
- `length_status`：修改后解释性中文 1,995 字；剔除标题、表格、公式围栏、行内数学、行内代码和链接目标后计汉字；核心单元，下限 1,800 字，通过。
- `evidence`：Ba 等人的论文摘要支持单样本层统计、RNN 逐时间步统计和训练/测试计算一致性；RMSNorm 摘要支持省略重新中心化的研究假设及其特定实验；Transformer v7 §3.1 支持原论文 residual 后 LayerNorm 顺序；Xiong v2 摘要支持其 mean-field 分析和作者实验中的 Pre-LN/Post-LN 结论，均不外推为所有模型规则；PyTorch v2.14 文档只支持该版本的 normalized_shape、总体方差、epsilon 和 RMSNorm 默认 eps 语义。未 profile 真实模型。
- `example_status`：残差例取 x=(1,2)、F(x)=(-0.05,0.1)，得 (0.95,2.1)。Norm 例取 [1,2,3,10]：LayerNorm 均值 4、中心化向量 [-3,-2,-1,6]、总体方差 12.5、标准差约 3.5355、输出约 [-0.8485,-0.5657,-0.2828,1.6971]；RMSNorm 均方 28.5、分母约 5.3385、输出约 [0.1873,0.3746,0.5620,1.8732]。整体加 5 后 RMSNorm 均方 93.5、分母约 9.6695、输出约 [0.6205,0.7239,0.8273,1.5513]；LayerNorm 的中心化值和结果不变。仿射例取 gamma=2、beta=0.5，首维 -0.8485 变为 -1.197。epsilon 例取常量向量 [2,2,2,2] 和教学值 1e-5，分母约 0.00316，中心化输出为零。数值均为教学输入。
- `code_status`：无可执行代码；未运行程序、模型训练、测试或 PDF 构建。
- `formula_status`：复核总体方差分母 D、RMS 分母、平移后均值/平方和/归一化结果、gamma/beta 仿射值、epsilon 零方差例，以及 Pre-LN `I+J_FJ_N` 与 Post-LN `J_N(I+J_F)` 的链式法则次序。
- `link_status`：新增 LayerNorm、RMSNorm、Transformer v7 §3.1、Xiong v2 与 PyTorch v2.14 LayerNorm/RMSNorm 直达链接；对应来源页面已阅读。无新增本地相对链接。
- `visual_status`：保留 LayerNorm/RMSNorm 的数值对照表，呈现统计量与输出；本讲未新增 Mermaid 图。
- `sync_status`：未改变讲次结构、目录或书册边界；第 6–15 讲完成后更新简介短摘要并进行整篇复读。未发现需同步题库、练习或术语表的概念变更。
- `open_risks`：实际模型的 Norm 轴、仿射参数、epsilon、低精度累加、残差缩放与放置位置依具体实现；本文训练诊断量只用于定位异常，不构成因果证明。GitHub 外部链接后续随全册导航 QA 复核。

## chapter_scan 与精修记录：第 14 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 14 讲，覆盖 RoPE 的二维旋转/相对位移推导、位置扩展例、窗口预算、有效使用能力和长上下文评测设计；最大窗口、位置扩展算法全量比较和 RAG 深入内容仍转第二十一册/第二册。
- 扫描问题与资料范围：核对 RoPE 的二维 Q/K 点积符号、不同频率维度、标准 Value 路径、相对位置性质与外推边界；核对 PI 的位置索引缩放和特定实验，YaRN 的频率插值与 logits 温度；区分位置映射、服务窗口、注意力计算成本和真实任务能力；检查证据位置、填充、上下文长度、任务类型和基座知识是否需要分别控制。阅读 RoFormer v4 §§3.1–3.3，PI v2 引言/§§2.2–2.3/§§3.3–3.5，YaRN v3 摘要/引言/§§3.1–3.3/§4.1，Lost in the Middle v3 摘要，ACL 2025 LongBench v2 与 100-LongBench 论文摘要/引言/评测说明，LongSocialBench 的 EMNLP 2026 摘要，以及 2026-05 arXiv 预印本 Positional Failures 的摘要、评测设计和结果。来源及版本边界见 `research/rope-long-context-evaluation-2026.md`。
- 候选主题与去向：
  - RoPE 相对位移机制：作为本讲核心，保留旋转矩阵推导和单位向量手算，不重复第 7 讲位置方法总览或第 9 讲注意力完整推导。
  - Position Interpolation：加一个 4K→8K 的位置索引缩放例，说明插值会改变相对距离尺度；只用 PI 作为代表机制，不外推论文模型结果。
  - YaRN 与其他 scaling：正文定位其频率分段和 Attention 温度的组合类别，完整方法、参数和实验比较留第二十一册。
  - 长上下文评估：保留位置/长度/填充受控的短协议，结合 Lost in the Middle、LongBench v2、100-LongBench 的证据；长上下文任务结构可超出单纯 token retrieval。
  - LongSocialBench 与 Positional Failures：作为近期评测候选纳入扫描；前者是具体讨论树任务，后者本轮以 arXiv 预印本为证据，均不扩成第一册独立讲次或普适能力结论。
  - FlashAttention、Local/Sparse Attention 与 RAG：保留层级对照，说明分别影响 IO、注意力图和证据选择；实现与效果比较交给相应专题册。
  - 未发现改变第 6–15 讲骨架或新增长上下文独立章节的证据；本讲继续从第 13 讲 Norm/残差交到第 15 讲小型 GPT。
- 修改决定与结果：删去重复的两次 RoPE 正交性推导、重复证据位置/长度评测清单和多段窗口上限复述；保留旋转公式并完整手算一个点积及相同相对距离的平移例；新增 PI 索引公式与 4,096→8,192 示例，说明位置距离压缩和短上下文回归风险；重组长上下文约束表，加入输入/输出预算手算及受控位置、长度、填充、基线与任务类型评测链；加入 2025–2026 benchmark 发展和研究边界。讲次结构未改。
- `length_status`：修改后解释性中文 1,936 字；剔除标题、表格、公式围栏、行内数学、行内代码和链接目标后计汉字；核心单元，下限 1,800 字，通过。
- `evidence`：RoFormer v4 支持二维旋转点积、一般频率形式与指定频率设置下的距离性质；PI v2 支持位置索引线性压缩及 LLaMA 7B–65B 等特定训练/评测范围，原窗口任务在其延展实验中出现一定退化；YaRN v3 支持按频率插值和注意力温度组合；Lost in the Middle 支持所测多文档 QA/retrieval 中的位置效应；LongBench v2 支持多任务长上下文数据设计；100-LongBench 支持固定长度与基线能力混杂是作者指出的评测问题；LongSocialBench 是讨论树专项 benchmark；2026 arXiv 预印本的三因素结果限定在论文任务、模型和填充设置。均未当成跨模型普适结论。
- `example_status`：二维例取 q=k=(1,0)、每步角 0.25 rad；位置对 (1,3) 的旋转向量约为 (0.9689,0.2474) 与 (0.7317,0.6816)，点积约 0.8776，与 cos(0.5) 一致；把相对距离相同的例子平移到 (8,10)，该单一分数仍为 0.8776。PI 示例以原长度 4,096、目标 8,192，将 0-based 索引 5,000 映为 2,500；原相距 1,000 的两个索引映射后间隔 500。窗口预算例设总预算 8,192、预留输出 1,024，输入最多 7,168；6,000-token 文档将证据放在 300/3,000/5,700 三处作为受控评测建议，不是模型实测。
- `code_status`：无可执行代码；未运行模型、benchmark、性能 profile、测试或 PDF 构建。
- `formula_status`：复核 $R(m)^TR(n)=R(n-m)$ 的矩阵关系、两组相位点积、PI 位置压缩比例、相对距离变化及输入 token 预算减法；算例输入均标为教学假设。
- `link_status`：新增 RoFormer v4 §3、PI v2 §2.3、YaRN v3 §3.3、Lost in the Middle v3、LongBench v2、100-LongBench、LongSocialBench 及 2026 arXiv 预印本直达链接；来源页面及关键章节已阅读。第二册长上下文/RAG 相对链接目标存在。
- `visual_status`：保留方法层级对照表；通过旋转向量、同距离点积和位置预算算例呈现数值关系；未新增 Mermaid。
- `sync_status`：未调整讲次结构、目录或跨册边界；第 6–15 讲完成后更新简介摘要并复读第二正文文件。保留第二册长上下文/RAG 导航入口。
- `open_risks`：未执行任何真实模型的 PI/YaRN 扩展或长上下文基准；位置效应、接口窗口、截断和预算依模型/服务版本；2026 arXiv 预印本的同行评审状态本轮未核验，正文已限定为预印本。全册链接 QA 待后续统一复核。

## chapter_scan 与精修记录：第 15 讲

- 日期与范围：2026-10-28；`chapters/02-transformer核心.md` 第 15 讲，覆盖字符级 next-character 数据、decoder-only MiniGPT 的 embedding/causal attention/Pre-LN block/MLP、训练 loss、采样生成、shape 路径及教学边界。
- 扫描问题与资料范围：核对重复数据长度与字符词表、随机窗口的 x/y 一位对齐、因果 mask 切片、Q/K/V 头维、logits 与 target 的损失接口、逐步采样 shape、生成超过 block size 后的上下文裁剪和位置编号。检查仓库 Python/PyTorch 锁定值；阅读 PyTorch 2.12 `cross_entropy` 和 `multinomial` 官方文档。外部 API 事实及版本边界见 `research/minigpt-pytorch-api-2026.md`。
- 候选主题与去向：
  - 最小端到端实现：保留本讲的字符级小模型作为第 6–14 讲机制闭环，不改成使用预训练库的高层封装。
  - 右移标签与 causal mask：在本讲的实际数据和 mask 中给逐位置例子，强调目标定义和可见输入是两个不同约束；不重复展开前文的训练目标理论。
  - 生成期 sliding window：解释超过 16 token 后丢弃早期上下文、并按当前窗口重新编号为 0–15；KV Cache、RoPE、子词 tokenizer、服务性能不纳入本讲，分别交接后续专题。
  - 训练与评估：说明 100 次循环是随机窗口更新、没有独立验证集，避免把有限 loss 或生成文本写成质量结论；真实数据与训练评估留第五册及实验章节。
  - 未发现需要改变第 6–15 讲章节骨架或新增独立讲次的候选。
- 修改决定与结果：删除重复说明 loss/生成结果不确定性的段落；补充起点 0 的 `hello transforme` → `ello transformer` 标签例、长度 4 的因果允许矩阵、PyTorch 交叉熵的形状/reduction 对照、`[1,16,13]` 到 `[1,1]` 的生成 shape 手算、窗口裁剪后的局部位置编号及 100 步训练循环含义。未改可执行代码块、接口、依赖或声明环境；未改变章节结构。
- `length_status`：修改后解释性中文 1,860 字；计数剔除标题、表格、代码围栏、行内代码/数学及链接目标后统计汉字；核心单元，下限 1,800 字，通过。
- `evidence`：PyTorch 2.12 文档支持 cross_entropy 的 logits/类别索引形状约定、默认 mean reduction，以及 multinomial 的输入约束、抽样数量和返回 shape；不支持代码运行或模型训练效果结论。来源详情见对应 research 文件。
- `example_status`：重复串每份 18 字符、64 份总长 1,152，词表 V=13；长度 16 的 batch 目标是右移一个字符，起点 0 的输入/目标为 `hello transforme` / `ello transformer`。T=4 的因果允许矩阵前两行为 `[1,0,0,0]` 与 `[1,1,0,0]`。训练 shape `[8,16,13]` 展成 `[128,13]`，target 展成 `[128]`。手算 loss 例 logits `[0,ln 2,0]` 得概率 `[0.25,0.5,0.25]`、中间目标 NLL `ln 2≈0.6931`、logits 梯度 `[0.25,-0.5,0.25]`。生成时 `[1,16]→[1,16,13]→[1,13]→[1,1]→[1,17]`；prompt 1 字符加 16 个新字符后，下一轮只保留末 16 个并映射到位置 0–15。均为代码构造或手算示例，不是训练实测。
- `code_status`：可执行脚本原有标记“静态审阅，未运行”；本轮没有更改代码，因此未触发代码复验；未运行训练、测试、性能评估或 PDF 构建。已核对锁文件声明 Python 3.13+、torch 2.12.0。
- `formula_status`：静态复核 x/y 的右移索引、causal 下三角允许关系、attention mask 广播维、Q/K/V 与 logits shape、cross_entropy 展平接口、生成采样拼接和 block-size 裁剪；手算交叉熵与梯度沿用正文明确标注的人工 logits。
- `link_status`：新增 PyTorch 2.12 `cross_entropy` 和 `multinomial` 直达文档；将本文件 RoFormer 与 Lost in the Middle 资料入口更新为正文采用的 v4/v3 版本；第五册模型架构相对链接路径已核对存在。其余全书链接仍待后续导航 QA。
- `visual_status`：保留完整 shape 表；新增位置/采样 shape 推演和小型 mask 数值行示例；未新增 Mermaid 图。
- `sync_status`：更新 `book-01-core-30/简介.md`，新增九章阅读提要，其中 Transformer 核心摘要明确本章从 token 表示走到 MiniGPT 前向与生成；没有讲次/目录或跨册边界变化。其他章节摘要将在各章精修完成后复核；无需改题库或练习。
- `open_risks`：代码没有实际执行证据，生成质量、随机可复现程度和性能均未测；示例无验证集，不代表真实训练。第二正文文件仍待连续复读；简介其余章节摘要需随章节精修复核。全库链接 QA 尚未完成。

## chapter_scan 与精修记录：第 16 讲

- 日期与范围：2026-10-28；`chapters/03-预训练与规模化.md` 第 16 讲，从预训练目标、数据来源/清洗/去重、tokenizer 与 packing，到训练监控、checkpoint 评估及 base model 到后训练的交接。本轮不含第 17–18 讲内容。
- 扫描问题与资料范围：核对 next-token 目标能支持哪些能力主张；exact/near dedup 与 benchmark overlap 分别支持什么判断；EOS 分隔符是否落实 attention 边界；packing 的 mask 与样本目标如何对应；checkpoint 的状态恢复和逐位复现是否被混为一谈；如何按分领域结果选择 checkpoint。阅读 GPT-3、Deduplicating Training Data Makes Language Models Better（ACL 2022）、Task Contamination（AAAI 2024）、Packing and Masking Strategies（ACL 2024）、InstructGPT 原始论文，以及 PyTorch 2.14 checkpoint 与 reproducibility 官方文档。知识点、原文范围和证据边界见 `research/pretraining-data-packing-checkpoints-2026.md`。
- 候选主题与去向：
  - 预训练数据治理与过滤误差：本讲增加来源/采集时间/使用限制/处理版本的记录要求，并指出语言形式过滤可能同时移除目标信号；采样配比和数据质量效应留给第 17 讲。
  - 重复数据和污染：本讲保留最小机制与训练/验证重叠风险，改掉“只要出现 benchmark 就会虚高”的绝对句；详细污染检测与 data mixture 仍在第 17 讲展开。
  - Packing 的文档边界：说明 EOS 本身不构成注意力隔离；按独立文档或连续 token stream 的目标选择 mask。ACL 2024 结果只作为特定实验例证，不外推为跨文档注意力普遍有害。
  - Checkpoint 与续训：把模型、optimizer/scheduler/global step、RNG/数据游标及实验元数据整理为状态表；区分完整状态恢复与只加载权重的 warm start，并收窄逐位复现承诺。分片状态接口交接第 18 讲。
  - Base model 与助手行为：收窄“模型会什么/怎么回答”的边界，用 InstructGPT 作为特定 prompt 分布下的训练前后案例，不将其人类偏好结果外推为普遍能力或安全保证。
- 修改决定与结果：把“学习世界规律”改写为拟合语料统计规律并需要独立评估；清洗部分加入误删检查和数据来源元数据；去重与污染结论补来源和条件限定；扩充 EOS/文档边界解释；checkpoint 清单改成状态表，修正“权重重启后的曲线变化可直接诊断损坏”的推断；把 checkpoint 选择示例改为固定评估版本、测试口径与分领域指标后再做消融；将 2048×512 的训练 token 账本改成一致的 1T 目标与 1.2T 候选语料例；改写现场故障表与复核问题，减少重复清单。讲次结构不变。
- `length_status`：修改后解释性中文 3,365 字；计数剔除标题、表格、围栏代码、引用状态行、行内代码/数学及链接目标后统计汉字，核心讲次下限 1,800 字，通过。
- `evidence`：GPT-3 原论文支持自回归目标并报告其受测 benchmark 中多数污染影响较小、少数结果需标注/不报告；Lee et al. ACL 2022 在特定语料和模型中报告近重复、记忆输出和训练/验证重叠；Li 与 Flanigan 的时间比较、样例提取和训练数据检查结论限于其模型/任务，且检测方法有低召回边界；ACL 2024 packing 论文的 mask 比较和运行开销限于其预训练设置；PyTorch 2.14 教程支持保存模型/optimizer 状态的通用建议，reproducibility 文档明确跨版本/平台不保证完全相同；InstructGPT 结果限于论文 prompt 分布和人工偏好评测。未把任何单项实验作为普遍定律。
- `example_status`：将序列长度 2,048、global batch 512 条序列且所有位置有效的假设算为每次更新 1,048,576 个目标 token；目标 1T 约需 95.4 万次更新；1.2T 候选语料不重复采样时目标约占六分之五。实际 input token 与有效 loss token 可能因 mask、padding、尾块和重复读取而不同。评估例 `validation loss 2.8→2.6`、代码通过率 `61%→58%` 明确标注为教学假设，不表示实测。
- `code_status`：packing 演示代码未改动，保留原有“已验证（2026-10-07；Python 3.13.13；标准库）”标记；本轮未运行代码、测试、训练或 PDF 构建。代码运行日志的历史证据缺口仍存在，未补造。
- `formula_status`：复核 2,048×512=1,048,576 和 1T/1,048,576≈953,674，使用约 95.4 万步表达；区分输入 token、目标 token 和有效 loss token；无新增数学围栏公式。
- `link_status`：新增 GPT-3、ACL 2022 去重论文、AAAI/arXiv task contamination、ACL 2024 packing、PyTorch 2.14 保存/加载与复现、InstructGPT 的直达来源；均在浏览中打开。仅本阶段交接相对链接未修改；全册导航 QA 尚待后续统一复核。
- `visual_status`：checkpoint 状态表呈现恢复语义，现场故障表呈现观察信号到初步核对项；未新增 Mermaid 或可执行程序示例。
- `sync_status`：无讲次、目录、书册边界或配套索引变化；章节 03 的简介提要待第 17、18 讲精修后再复核并同步。
- `open_risks`：未处理真实语料、数据许可或训练系统；未实测 packing mask 质量、checkpoint 分片恢复或训练复现。PyTorch 2.14 和各论文的细节只支持声明的版本/实验配置；完整 chapter 03 仍须处理第 17–18 讲后连续复读。

## chapter_scan 与精修记录：第 17 讲

- 日期与范围：2026-10-28；`chapters/03-预训练与规模化.md` 第 17 讲，覆盖 scaling law、compute-optimal 预算、数据质量与 mixture、合成数据、重复数据、benchmark contamination 及训练预算决策；不含第 18 讲分布式训练。
- 扫描问题与资料范围：核对 scaling law 的目标变量和测试范围；$C\approx6ND$ 的参数口径与 FLOPs 假设；Chinchilla 结论能否外推到新模型/数据；数据质量和 mixture 是否被写成普遍因果；不同合成数据类型、生成成本与预算如何比较；重复/benchmark overlap 能支持哪些记忆或成绩推断。阅读 Kaplan scaling law、Chinchilla、DoReMi、Kang 合成数据研究、Roberts 2026 预印本、Subramanyam ICLR 2026、Lee et al. ACL 2022、GPT-3 原论文和 Li 与 Flanigan AAAI 2024。具体证据和边界见 `research/scaling-laws-data-mixtures-2026.md`。
- 候选主题与去向：
  - scaling law 与预算分配：保留本讲主线，补 held-out language modeling loss 的目标边界、拟合式、计算量近似和 Chinchilla 特定设定；不将 loss scaling 当作下游能力定律。
  - 数据质量和 mixture 优化：保留任务相关性和独立评估；增加 DoReMi 作为代理优化案例，结果限于该研究的模型、数据和任务。
  - 合成数据：细分自然文本改写、教材式生成等类型，保留 Kang 的预算条件实验、Roberts 的 compute-limited 预印本拟合及 Subramanyam 的人工噪声实验；不提炼通用合成比例。
  - 重复数据与 benchmark contamination：用 Lee、GPT-3、Li 与 Flanigan 的原始研究收窄记忆、overlap、召回与误报结论。核对后将 arXiv:2312.16337 的作者归属从错误的 Sainz 改为 Li 与 Flanigan，并同步第 16 讲正文、研究记录和进度记录。
- 修改决定与结果：更新 Scaling Law 定义与引用；解释参数/token/compute 的变量和 $C\approx6ND$ 的量级估算边界；收窄 Chinchilla 和数据质量结论；补 DoReMi 以及三类合成数据研究的条件性结果；纠正污染检测证据与作者归属；将 30B/1T、10B/3T、7B/4T 三组预算逐项代入并列出 FLOPs；删去重复的简单估算小节，整理失败模式与复核问题。未改变讲次结构。
- `length_status`：修改后解释性汉字 3,849 字；计数剔除标题、表格、围栏代码、代码状态行、行内代码/数学及链接目标，核心讲次下限 1,800 字，通过。
- `evidence`：Kaplan 的幂律与约 $6N$ FLOPs/token 是指定 Transformer/语料范围内的实证和估算；Chinchilla 与 DoReMi 的配比结论限于各自模型、数据和评测；Kang 的 1/3 改写比例和 5–10 倍速度只在其较大数据预算实验中成立；Roberts 的比例为预印本拟合/外推；Subramanyam 的质量参数依赖其可控噪声定义；Lee 的记忆降低结果限于特定数据/模型；GPT-3 与 Li/Flanigan 的污染检测各有干净子集偏差、precision/recall 边界。未外推为行业通用训练配方。
- `example_status`：将 $C\approx6ND$ 用于非 embedding 参数和训练 token 量级估算：30B×1T 得 $1.8\times10^{23}$ FLOPs，10B×3T 得 $1.8\times10^{23}$ FLOPs，7B×4T 得 $1.68\times10^{23}$ FLOPs；正文逐项展示代入过程，并说明计算量相近不预测相同 loss 或任务能力。
- `code_status`：参数/token 权衡 demo 未修改，既有“已验证（2026-10-07；Python 3.13.13；标准库）”标记保留；本轮未运行程序或测试。
- `formula_status`：增加拟合形式 $L(N,D)$ 并说明参数由实验拟合；复核 $C\approx6ND$ 计算口径与三组预算的代入、数量级。未改可执行代码。
- `link_status`：新增/补充 Kaplan、Hoffmann、DoReMi、Kang、Roberts、Subramanyam、Lee、GPT-3 和 Li/Flanigan 的直达来源链接，浏览核对了摘要或相关实验段；同步修正第 16 讲污染论文作者名。全册导航 QA 待后续统一复核。
- `visual_status`：用合成数据对照表呈现类型、研究观察和边界；新增逐项代入表呈现预算和中间量；失败模式整理为控制表。未新增 Mermaid 图。
- `sync_status`：未改变讲次、目录或书册边界；章节 03 的简介提要待第 18 讲精修及整文件复读后同步。
- `open_risks`：worked example 是 FLOPs 粗估而不是训练实测；Scaling law、质量参数、配比和污染结论均依赖原论文设定。第 18 讲未精修，章节 03 和第一册仍未完成连续复读/整册验收。

## chapter_scan 与精修记录：第 18 讲

- 日期与范围：2026-10-28；`chapters/03-预训练与规模化.md` 第 18 讲，覆盖训练状态显存、Data Parallel/DDP、ZeRO、FSDP、Tensor/Pipeline/3D Parallelism、activation checkpointing、collective、设备拓扑、训练指标、profiling 与分片 checkpoint 恢复；不含具体框架配置教程或性能 benchmark。
- 扫描问题与资料范围：逐项核对参数、梯度、Adam 状态的显存口径；DDP 梯度同步与本地更新前提；ZeRO/FSDP 的分片范围、all-gather/reduce-scatter 和 API 版本差异；TP/PP 的切分位置与通信；pipeline bubble、activation recomputation、DCP 分片读取与恢复边界；并行策略和 MFU/利用率结论是否过度外推。阅读 PyTorch 2.14 DDP、FSDP1/FSDP2、Distributed Checkpoint 与 checkpoint 文档，DeepSpeed ZeRO v0.19.9，ZeRO、Megatron-LM、GPipe、Chen 等原始论文。来源事实、阅读位置和证据范围见 `research/distributed-training-parallelism-2026.md`。
- 候选主题与去向：
  - DDP 作为可放入单卡模型的基线：保留本讲，明确 rank 数据切分、反向同步和相同初始/优化器状态下的副本更新条件；loss reduction、局部 batch 和 hook 等会影响梯度聚合口径。
  - ZeRO/FSDP 状态分片：保留核心讲解，分别说明 optimizer state、gradient、parameter 的分片阶段，并标注 FSDP1/FSDP2 API 差异；具体峰值、预取与恢复拓扑依框架和配置。
  - TP、PP 与三轴组合：保留作为互补切分机制，连接矩阵布局、collective、micro-batch 与 bubble；不把某种组合写成所有大模型或前沿训练的必选配方。
  - 激活重计算与分布式 checkpoint：作为显存账本和训练恢复的重要支持机制保留；DCP 的读取和 state-dict 约束以 PyTorch 2.14 文档为边界，跨拓扑恢复仍需实际验证。
  - 未发现需要新增独立讲次或改变第 16–18 讲到后训练交接的证据；深度框架调优交给第五、九、二十三册及具体框架文档。
- 修改决定与结果：补充 7B 参数、8 rank 的训练状态逐项字节/GiB 手算与假设边界；说明数据按 rank 切分；收紧 DDP 副本一致性条件；厘清 ZeRO 阶段与 FSDP1/FSDP2 共同机制及配置差异；补 TP 矩阵切分、PP micro-batch/bubble、checkpoint 恢复约束、collective 到并行方式的对应；将 3D 并行经验句改为受模型、互联和 profiling 约束；整理故障信号表，并给通信重叠公式补输入、单位、逐步代入和理想/无重叠边界。移除 DDP 每进程绑定一张 GPU 的重复句，未改代码。
- `length_status`：第 18 讲修改后解释性中文 2,840 字；计数剔除标题、表格、代码围栏、状态行、行内代码/公式及链接目标；核心讲次下限 1,800 字，通过。
- `evidence`：ZeRO 三阶段以原始论文和 DeepSpeed 当前文档分别支持方法定义与当前接口；DDP、FSDP1/FSDP2、DCP 以 PyTorch 2.14 文档限定 API 行为；Megatron-LM 支持论文中的层内张量切分和与流水线维度组合；GPipe 支持其 micro-batch 与 bubble 分析；Chen 论文和 checkpoint 文档支持激活重计算的内存/计算折中。设备和框架版本、精度、wrap 粒度、拓扑与 collective 实现会改变峰值和性能，不外推为统一配置。
- `example_status`：显存例使用 7B 参数、bf16 参数/梯度各 2 bytes、fp32 Adam 一阶/二阶状态各 4 bytes、8 个均匀分片 rank；DDP 84×10^9 bytes≈78.2 GiB，ZeRO-1 35×10^9≈32.6 GiB，ZeRO-2 22.75×10^9≈21.2 GiB，ZeRO-3/FSDP 10.5×10^9≈9.8 GiB，明确未计激活、临时/通信 buffer 和碎片。通信例设 `t_c=40 ms`、`t_m=20 ms`、`rho=0.5`，代入得约 50 ms；无重叠约 60 ms，理想隐藏短段约 40 ms，并说明这只是简化模型。
- `code_status`：显存估算代码的可执行语句、输入输出、依赖、环境和输出约定均未改变；沿用原正文“已验证（2026-10-07；Python 3.13.13；标准库）”标记，本轮不重跑。当前进度目录未找到更早的详细运行命令和边界日志，不补造验证细节。
- `formula_status`：逐项复核 7B/8-rank 状态显存和 GB/GiB 换算；定义 `$t_c$`、`$t_m$`、`$\rho$` 后展示重叠公式的代入、结果及适用限制。
- `link_status`：核对 PyTorch 2.14 DDP/FSDP/FSDP2/DCP 与 checkpoint 文档、DeepSpeed ZeRO、ZeRO/Megatron-LM/GPipe/activation checkpointing 原始论文的直达链接；本章现有相对链接目标存在，持续纳入全书链接 QA。
- `visual_status`：用显存账本表、并行选择表和故障信号表呈现数字、策略映射与排查路径；现有流程图覆盖预训练整体链路，本讲未新增 Mermaid 图。
- `sync_status`：已将第一册简介中“预训练与规模化”的逐章摘要更新为数据治理、预算、状态分片、通信与恢复边界；目录与章节结构未变。
- `open_risks`：显存结果是理想状态量估算，不含 activation、峰值 all-gather、碎片和框架开销；代码历史详细运行日志缺失但原验证标记仍在；未运行真实多卡训练、恢复或性能剖析。章节 03 连续复读已完成，第一册整册复读仍待第 19–38 讲精修后进行。
