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
- 逐讲精修已从第 6 讲推进至第 25 讲；第二正文文件第 6–15 讲已精修、更新简介中的九章阅读提要并完成连续复读，第三正文文件第 16–18 讲已精修并完成连续复读，第四正文文件第 19–23 讲已完成，且已连续复读并更新简介摘要；第五正文文件第 24–25 讲已精修，第 26–28 讲待继续。

## open_risks 与下一步

- 第 1–25 讲已完成本轮定向扫描与精修；第 26–38 讲仍未完成各自的章级扫描和候选去向判断。
- 第 1 讲计数为 5,258 字，第 2 讲为 3,894 字，第 3 讲为 4,011 字，第 4 讲为 3,421 字，第 5 讲为 4,583 字，第 6 讲为 2,083 字，第 7 讲为 2,064 字，第 8 讲为 1,852 字，第 9 讲为 1,951 字，第 10 讲为 2,008 字，第 11 讲为 2,093 字，第 12 讲为 1,917 字，第 13 讲为 1,995 字，第 14 讲为 1,936 字，第 15 讲为 1,860 字，第 16 讲为 3,365 字，第 17 讲为 3,849 字，第 18 讲为 2,840 字，第 19 讲为 4,233 字，第 20 讲为 4,334 字，第 21 讲为 4,199 字，第 22 讲为 3,010 字，第 23 讲为 3,002 字，第 24 讲为 4,885 字，第 25 讲为 4,423 字；第 26–38 讲尚未按规范计数。
- 第 1–25 讲已逐讲记录例子、公式、代码状态、来源支持、链接和配套同步；第一至第四正文文件均已完成当前范围的主路径连续复读，第五正文文件第 24–25 讲已完成，第一册整册复读仍待第 26–38 讲完成后进行。
- 研究来源文件在 `research/`；正文链接仍需随整章和全册导航 QA 复核，不假定维护记录链接等同读者入口。
- 下一工作单元：处理 `chapters/05-推理、部署与评估.md` 第 26 讲；第一册第 26–38 讲和整册复读仍待完成。

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

## chapter_scan 与精修记录：第 19 讲

- 日期与范围：2026-10-28；`chapters/04-后训练与对齐.md` 第 19 讲，覆盖 base/instruct 行为边界、指令数据、assistant-only 监督选择、chat template、单轮/多轮、system prompt、合成指令、SFT/RLHF/DPO 关系和评估；第 20 讲承担 SFT 目标、标签移位与代码实现的主讲。
- 扫描问题与资料范围：核对指令微调是否被误写为保证用户意图/安全的机制；SFT 与 instruction tuning 的概念层次；assistant-only loss 是否被说成普遍规则以及 mask 对输入上下文的影响；模板如何序列化和追加 generation prompt；数据质量、任务覆盖、数量和多样性的证据是否支持普遍排名；合成数据生成、过滤和评估成本；system prompt 与运行时权限边界；单轮训练对多轮行为的证据。阅读 InstructGPT、FLAN、LIMA、Self-Instruct、QDIT 原始论文/会议页面，Transformers v5.17.0 与 TRL v1.16.0 官方文档，以及 2026 年后训练综述预印本和 ICLR RL compute scaling 论文摘要。来源范围和事实边界见 `research/instruction-tuning-data-and-templates-2026.md`。
- 候选主题与去向：
  - 指令微调任务与未见任务泛化：保留为本讲主线，以 FLAN 作为特定评测案例；不将任务数扩展写成单调提升规律。
  - 数据质量、覆盖和数量：保留并改为多变量判断；用 LIMA 的小规模精选样本、FLAN 的任务扩展和 QDIT 的质量/多样性权衡对照，不提炼单一数据配方。
  - 合成指令：保留 Self-Instruct 流程作为研究案例，说明生成与过滤可减少逐条人工撰写但并非零成本，且生成结果需质量检查。
  - Chat template 和 assistant-only loss：保留在本讲建立概念边界；用固定 Transformers/TRL 文档说明角色序列化和模板标记依赖，第 20 讲继续负责机制实现与训练代码。
  - System prompt 与安全：保留其作为模型上下文的作用，明确其不替代工具权限和运行时执行机制。
  - 后训练分类：引用 2026 arXiv 综述作为方法关系视角，不把 off-policy/on-policy 组织方式当作所有模型的固定流水线。
- 修改决定与结果：将“base model 不会听话”“assistant-only 是唯一正确目标”“数据质量必然胜过数量”“system prompt 可控制行为”等绝对措辞改为任务和配置相关的结论；解释输入 mask 与目标 mask 的差别，加入两目标概率为 1/2、1/4 的手算，assistant-only 平均 loss 为 1.040 nats/token、全序列四目标平均为 0.866 nats/token，说明数值变化来自分母口径而非回答概率改善；更新模板差分与 generation prompt 边界；用 FLAN/LIMA/QDIT 对照解释数据质量和多样性；加入 Self-Instruct 的种子—生成—过滤案例；重写系统提示、安全和评估边界；删除与第 20 讲重复的 assistant-only label 构造代码块，保留本讲可读的 token/label 轨迹并将实现主讲交给第 20 讲。
- `length_status`：修改后解释性中文 4,233 字；计数剔除标题、表格、围栏内容、代码状态行、行内代码/公式及链接目标；核心讲次下限 1,800 字，通过。
- `evidence`：InstructGPT 支持其 GPT-3 示范 SFT→奖励模型→PPO 路线和特定提示分布的人评结果；FLAN 支持论文任务集和模型条件下的未见任务改善；LIMA 的 1,000 样本、质量/多样性消融和多轮结果限定在其模型与小型评估；Self-Instruct 的 52,445 指令、82,439 实例及噪声抽查限定在 GPT-3 生成管线；QDIT 的质量/多样性关系限于论文实验。Transformers/TRL 文档分别支持固定版本的序列化、generation prompt、completion/assistant-only loss 与模板标记行为。2026 后训练分类来自 arXiv 预印本，RL compute 配方观察来自 ICLR 2026 论文；未外推成统一生产配方。
- `example_status`：用 `input_ids=[<user>, q, <assistant>, 4, <eos>]` 与 `labels=[-100,-100,-100,4,<eos>]` 说明上下文 token 保留、assistant 目标计算；假设两个回答目标的正确概率为 1/2 和 1/4，逐项以自然对数计算 assistant-only 平均交叉熵 1.040 nats/token；再加入两个概率均为 1/2 的上下文目标，得到全序列平均 0.866 nats/token，并解释不可把两个口径直接当作回答质量对比。
- `code_status`：删除原有空格切分的 assistant-only label 构造 demo，以避免与第 20 讲标签构造/shift 实现重复；当前第 19 讲无可执行代码，剩余 fenced text 均为对话或 token 序列示意。未执行程序；第 20 讲的 PyTorch mask demo 未改，沿用其既有验证标记。
- `formula_status`：新增以自然对数表示的两种平均交叉熵计算，说明 token 目标集合、归一化分母、舍入和 nats/token 单位；公式代入可复算。
- `link_status`：在结论附近增加 InstructGPT、FLAN、LIMA、Self-Instruct、QDIT、Transformers v5.17.0、TRL v1.16.0 和 2026 后训练综述直达来源；本章相对链接目标存在，后续继续纳入全书链接 QA。
- `visual_status`：研究对照表展示数据量/质量/多样性证据和边界；token/label 轨迹与手算表述训练目标变化；保留章节级后训练流程图，未新加 Mermaid。
- `sync_status`：讲次结构、目录和书册边界不变；简介中的“后训练与对齐”总提要继续覆盖 instruction tuning、偏好优化和代理目标，本讲不触发额外目录变更。
- `open_risks`：LIMA、FLAN、Self-Instruct 和 QDIT 的实验不可直接横向比较；本讲没有运行真实模型行为差分，评估方案仍是教学实验设计；assistant-only mask 与模板功能依赖具体框架/版本。第 20–23 讲及第四正文文件连续复读仍待处理。

## chapter_scan 与精修记录：第 20 讲

- 日期与范围：2026-10-29；`chapters/04-后训练与对齐.md` 第 20 讲 SFT，覆盖监督目标、对话格式、label mask、causal shift、packing/padding、全参微调/LoRA/QLoRA、学习率与过拟合、能力遗忘、数据质量和评估；不改变讲次、章节顺序或书册边界。
- 扫描问题与资料范围：核对 assistant-only loss 是不是通用必需设置；`input_ids`、attention mask、labels 和 causal shift 各自控制什么；`ignore_index=-100` 和 mean reduction 的分母口径；EOS 能否隔离 packed 样本；LoRA 参数量能否推出显存和质量；QLoRA 的性能对照是否覆盖 33B/65B；PEFT 是否能避免遗忘；SFT 后的任务提升和旧能力回归如何评估。阅读 PyTorch 2.14 `CrossEntropyLoss` 官方 API、LoRA 原论文 v2、QLoRA 原论文及其 Limitations/Discussion、Kalajdzievski 的遗忘研究、ACL 2024 packing 论文；另复核 TRL v1.16.0 SFTTrainer 文档。新事实和证据边界见 `research/sft-adaptation-loss-and-forgetting-2026.md`，模板接口背景见 `research/instruction-tuning-data-and-templates-2026.md`。
- 候选主题与去向：
  - SFT 目标、assistant-only 与标签/上下文边界：保留为本讲核心，说明 assistant-only 是常见可选目标，按任务选择；用 `input_ids`/labels 轨迹解释被 mask token 仍可作上下文。
  - Causal shift、ignore index 与归一化：保留实现主线，区分 attention mask、目标 mask、模型接口中的 shift 责任；补按有效目标 token 求平均的公式和手算，不把代码当作推导替代。
  - Packing：保留 padding、截断与拼接的效率动机；解释 EOS 不自动隔离 attention，并把注意力隔离策略交给样本独立性假设决定。ACL 2024 结果只作特定预训练配置证据，不外推为 SFT 的统一结论。
  - 全参、LoRA、QLoRA：保留参数更新与资源账本对照；用单线性层的 rank-8 参数数作教学例，明确参数量不等于显存、速度或质量比例；QLoRA 的大模型比较限定到论文实际验证范围。
  - 灾难性遗忘与评估：保留为新任务收益与旧分布回归的实测问题；不把数据窄、学习率或训练轮数写成已经证实的通用单因果解释，安排固定变量的消融与任务切片评估。
  - 未发现需要新增独立讲次、调整目录或改变第 19–21 讲承接的依据；偏好优化仍由后续讲次展开。
- 修改决定与结果：收窄 SFT 对正确性、意图和安全的承诺；将预训练/SFT 区别改为数据分布、上下文和目标 token 的差别；为 assistant-only 目标补充 TRL 可选接口示例；拆清输入上下文、attention mask、目标 mask 与标签右移；细化 `-100` 对目标项和 logits 梯度的含义；增加 EOS/packing 边界与检查路径；以对比表整理三种微调方式；补 LoRA 低秩式及单层参数量手算；限定 QLoRA 16-bit 性能对照不覆盖 33B/65B；将遗忘从列因果清单改写为待测风险；将“先 SFT 再偏好优化”改成常见但非强制流程，并补任务切片和旧能力评估。
- `length_status`：第 20 讲解释性中文 4,334 字，统计排除标题、表格、围栏代码、状态标记、行内代码/公式和链接目标；作为核心讲解单元，超过 1,800 字下限。篇幅达标不替代教学和证据门禁。
- `evidence`：PyTorch 2.14 文档支持类别索引 logits、`ignore_index` 和未忽略目标的 mean reduction，边界限于该 API，分布式或自定义归约须另核；LoRA 原论文支持冻结基座、低秩更新以及指定模型/任务结果，不能推出跨任务等效性；QLoRA 原论文支持冻结 4-bit 基座、adapter 训练及 NF4/double quantization/paged optimizer 机制，作者明确未证明 33B/65B 与全参微调等效；遗忘论文限于 Llama 2 7B Chat、指定数据和训练配置；ACL 2024 packing 结果限于论文的模型、语料、任务和 mask 实现。证据细节及来源版本见上述 research 文件。
- `example_status`：SFT 手算假设三个有效 assistant 目标的正确 token 概率为 1/2、1/4、1/2，逐项 NLL 为 0.6931、1.3863、0.6931 nats，总和 2.7726，除以 3 后为 0.9242 nats/token；明确 mask 只排除该 token 的目标项，不排除其作为上下文。LoRA 单层例使用 `d_in=4096`、`d_out=11008`、`r=8`：完整矩阵 45,088,768 个参数，adapter 120,832 个参数，占该单矩阵权重数约 0.268%；这是构造的参数数例，不是实测资源或性能结果。
- `code_status`：`最小 demo：用 ignore_index 只计算 assistant loss` 中的 PyTorch 代码和输入、依赖、输出承诺未改，保留原正文“已验证（2026-10-07；Python 3.13.13；PyTorch 2.12.0+cpu）”标记，按规范未重跑。当前 progress 中未找到该代码块此前的命令、输入范围、预期/实际完整输出和边界覆盖日志；此证据缺口如实保留，未补造。本轮未运行程序、训练、测试或 PDF 构建。
- `formula_status`：定义 `$m_t$` 为有效目标位置，检查 t−1 logits 对齐位置 t token 的因果语言模型约定、有效 token 分母和自然对数单位；手核 $4\ln2/3\approx0.9242$。复核 LoRA 参数式 $r(d_{in}+d_{out})$、单矩阵乘积与比例约 0.268%。
- `link_status`：正文增加 PyTorch 2.14、TRL v1.16.0、ACL 2024 packing、LoRA、QLoRA、遗忘研究和 InstructGPT 直达链接；相关来源已打开核对。全章 04 的本地相对链接目标静态检查均存在。
- `visual_status`：新增全参/LoRA/QLoRA 资源与验证边界表；保留输入/目标 token 轨迹作为监督状态表述；没有适用的额外 Mermaid 图，不新增装饰图。
- `sync_status`：无结构变化，不改 `目录.md`；`简介.md` 的后训练章节摘要待第 21–23 讲完成并连续复读章节 04 后统一核对。全局交接已推进到第 21 讲。
- `open_risks`：未运行真实 SFT、LoRA/QLoRA 或模型行为回归；具体模板、packing mask、量化内核和分布式 loss reduction 依赖实现；旧代码验证详细日志缺失但原验证标记保留。章节 04 第 21–23 讲和整文件连续复读、第一册第 1–38 讲整册复读均未完成。

## chapter_scan 与精修记录：第 21 讲

- 日期与范围：2026-10-29；`chapters/04-后训练与对齐.md` 第 21 讲，覆盖 PPO-style RLHF 的代表性流程、偏好数据和 Reward Model 的代理角色、policy/reference/value 组件、PPO clipped objective、KL 正则、rollout/credit assignment、reward hacking、over-optimization、DPO 对照及评估；不改变讲次顺序或书册边界。
- 扫描问题与资料范围：检查 InstructGPT 三阶段路线是否被误写成所有偏好优化的固定流程；chosen/rejected 和 RM 标量能否代表客观正确或跨 prompt 质量；PPO clip 是否构成硬 trust region；reference KL 能否保证旧能力/安全；逐 token log-ratio 的单样本估计是否与完整分布 forward KL 混淆；RM 过度优化证据是否被外推；DPO 的离线固定数据阶段是否被误写成数据从未由 policy 生成。阅读 InstructGPT §3/结果摘要、PPO 原论文 §2–3、Stiennon 摘要论文 §3.4/§4.3/limitations、DPO 原论文 §1/§4/§5.1；来源位置、版本和适用边界见 `research/rlhf-preference-and-policy-optimization-2026.md`。
- 候选主题与去向：
  - PPO-style SFT→RM→PPO：作为代表性流程保留，注明 InstructGPT 的任务与 prompt 分布边界；不写成所有 RLHF/DPO 配方的必要顺序。
  - Reward Model 排序概率、pairwise loss 与代码：移至第 23 讲作为唯一完整主讲；第 21 讲保留代理分数职责说明并链接至 `#pairwise-ranking-loss`，避免重复公式和代码。
  - PPO clipped surrogate：保留 token 比率、优势和目标手算；明确 clip 改变代理目标，不是硬性概率/信赖域边界。
  - Reference/KL：保留防止无约束漂移的调节作用，区分完整条件分布 KL 与 sampled token log-ratio；明确不保证能力、安全或语言质量保留。
  - Reward hacking/over-optimization：保留 Stiennon 等人在 TL;DR 摘要任务中的观察和优化强度评估，不外推统一拐点。
  - DPO：简述原始固定偏好数据微调中不显式训练独立 RM、也不运行 PPO rollout 的范围，说明 reference/隐式 reward 仍存在，且数据可以来自先前 policy。
  - 未发现需新增讲次、改变章节顺序或调整跨册边界的候选。
- 修改决定与结果：降低 RLHF 对 helpfulness、honesty、harmlessness 等属性的保证语气；把 RM 的输出限定为标注/生成分布附近的排序代理；新增 policy/reference/RM/value 组件及更新职责表；补 PPO clip 数值代入、KL 正则目标的两输出分布数值代入和序列级奖励的 value/advantage 轨迹；调整训练流程和评估说明，区分离线 preference pairs 与当前 policy rollout；在 DPO 对照中补充参考策略、数据来源和适用范围。
- `length_status`：解释性中文 4,196 字，计数保留链接文字但排除链接目标、标题、表格、代码/公式围栏、引用状态、行内代码和数学；核心讲解单元，下限 1,800 字，通过。
- `evidence`：InstructGPT 支持其 GPT-3 设定中的示范 SFT、偏好 RM、PPO 和 API prompt 分布评估，不证明通用属性提升；PPO 原论文支持 clipped surrogate 一般算法，不是 LLM RLHF 专项实证；Stiennon 的 RM 过度优化/人类偏好分离仅限 TL;DR 等论文条件；DPO 论述依其偏好模型、KL 约束假设、reference 和论文任务。完整来源及未验证边界见上述 research 文件。
- `example_status`：PPO 假设旧/新 token 概率 0.20/0.30，得比率 1.5；取 ε=0.2、A=0.5，未裁剪项 0.75、裁剪项 0.60、surrogate 取 0.60，而实际比率仍为 1.5。KL 例使用 policy (0.6,0.4)、reference (0.5,0.5)，精确 forward KL 约 0.0201 nats；两输出 RM 分数假设为 1.2/0.8，期望 reward 1.04，β=2 reward units/nat 时目标约 0.9997；sampled log-ratio 分别约 +0.1823 与 −0.2231。Delayed reward 例设序列分数 2.0、γ=1、value baseline 为 1.2/1.5/1.8，得 returns 均为 2.0、优势为 0.8/0.5/0.2。均为教学假设，不是模型实测。
- `code_status`：第 21 讲不再含可执行代码；原有 `pairwise_reward_loss` demo 连同相同输入、依赖和输出说明移入第 23 讲，执行语句及声明环境未改，沿用 2026-10-07 的 PyTorch 2.12.0+cpu 验证标记，本轮不重跑。现有 progress 中未找到其完整历史命令、输入边界和实际输出记录，作为证据缺口保留；未补造。本轮没有运行程序或测试。
- `formula_status`：复核 PPO ratio/clipped surrogate、固定 prompt 的期望 reward 与 forward KL、sampled log-ratio 估计、prompt 平均目标及 delayed-return 优势；手算结果与公式一致。clip 不解释成硬约束，sampled log-ratio 不误写成每条样本非负 KL。
- `link_status`：正文含 InstructGPT、PPO、Stiennon 和 DPO 原始来源；第 21 讲新增章节内 Pairwise Ranking Loss 锚点，目标标题存在；第四正文文件本地相对 Markdown 链接静态核对均有目标。
- `visual_status`：新增 policy/reference/RM/value 状态与职责对照表，以逐步算例呈现 PPO 和 KL 数值；未新增 Mermaid，当前内容未发现必须另绘复杂系统图的关系。
- `sync_status`：未改讲次结构、目录或书册边界；第 21–23 讲完成后再连续复读章节 04 并统一复核 `简介.md` 的后训练提要。第 23 讲承担 RM pairwise 机制、算例和代码的完整主讲。
- `open_risks`：未运行 PPO/RLHF 训练或独立模型评估；KL shaping、value/policy 参数共享和采样策略依具体实现；来源实验不能确定跨任务通用过度优化阈值。第 22–23 讲、章节 04 连续复读及第一册整册复读仍待完成。

## chapter_scan 与精修记录：第 22 讲

- 日期与范围：2026-10-29；`chapters/04-后训练与对齐.md` 第 22 讲，覆盖标准离线 DPO 的问题定义、偏好数据、reference policy、KL 正则化 reward 目标至 DPO loss 的推导、beta、SFT/RLHF 对照、序列长度口径、评估和失败模式；不改变讲次顺序或书册边界。
- 扫描问题与资料范围：核对标准 DPO 与 PPO-style RLHF 的训练阶段边界；DPO 理论等价依赖哪些 KL/偏好模型/reference 假设；$Z(x)$ 为何在同 prompt pair 中抵消；reference 是否是硬 KL 上限；beta 的理论作用与 loss 缩放有何区别；序列 log probability 求和和 token 平均是否改变 pair margin；SimPO 与 DPO 的 reference/length 选择；2026 年是否有与 likelihood displacement/长度正则相关的新证据。阅读 DPO 原论文 §§3–5.1 与 Appendix A 对应推导，SimPO 原论文 §§2.1–2.3/§4 的方法与实验分析，以及 2026 年 arXiv 预印本摘要和 §1。来源版本与证据边界见 `research/dpo-preference-objective-and-length-2026.md`。
- 候选主题与去向：
  - KL-regularized reward 闭式解和 DPO 的 Bradley–Terry 推导：作为本讲核心，逐步展示 $Z(x)$ 的消项及 reference-adjusted margin；理论关系不外推成不依赖假设的等价。
  - beta：保留为 KL 惩罚系数和 loss logit/梯度尺度的双重解释；不把其说成硬概率上限或独立的更新强度旋钮。
  - 序列 logprob 汇总：加入总和/平均导致 margin 符号翻转的教学例，明确 EOS、目标 mask 与 tokenizer 依数据契约确认。
  - reference-free 方法：以 SimPO 作一种具体变体定位，说明其 reward normalization/target margin 与标准 DPO 不同；不扩成偏好算法列表或跨任务排名。
  - 2026 年 likelihood-displacement 预印本：保留一句研究候选和预印本限定，不将作者的模型/benchmark 结果作为统一修复结论。
  - 未发现需新增长讲、目录调整或改变本册/跨册分工的主题。
- 修改决定与结果：把标准 DPO 与 PPO-style RLHF 的数据和训练环节分开；补 KL 正则目标、闭式最优策略、reward 反解、Bradley–Terry 偏好概率和数据集 loss 的完整推导；解释 reference、beta 和 policy 初始化的角色；将 “DPO 更简单稳定” 改为“固定离线环节少在线组件，稳定性和质量依条件测量”；补 DPO loss 两组反向标签手算及 scalar gradient 权重；说明序列概率求和并以均值/总和 margin 反转例解释长度口径；加入 SimPO 与 2026 预印本的范围有限引用；修正 reference-free 方法与标准 DPO 的概念边界。
- `length_status`：解释性中文 3,010 字，保留链接标签、排除链接目标、标题、表格、代码/公式围栏、引用状态、行内代码和数学；核心讲解单元，下限 1,800 字，通过。
- `evidence`：DPO 原论文支持 KL 约束最优策略、Bradley–Terry 下归一化项抵消、隐式 reward 和固定偏好数据训练，结论限于相关理论假设和论文实验；SimPO 支持平均 token logprob/target margin/reference-free 设计及论文所测模型与 benchmark，不支持通用优越结论；2026 preprint 摘要/§1 支持作者研究 DPO/SimPO likelihood displacement 与长度归一化正则的报告，尚不足以作为已确立的跨任务修复。证据详见 research 文件。
- `example_status`：DPO pair 设 policy chosen/rejected logprob 为 −2/−4、reference 为 −2.5/−3.5，得 $\Delta_\theta=2$、$\Delta_{ref}=1$、$g=1$；$\beta=0.1$ 时 logit 0.1、概率约 0.5250、loss 约 0.6444。反向 policy 得 $g=-3$、loss 约 0.8544；另一个 $g=2$ 例 loss 约 0.5981。gradient 例用 beta 0.1、g=1/2 得 margin 标量导数约 −0.0475/−0.0450。长度例 chosen per-token policy/reference log-ratio 为 (0.10,0.05)，sum/mean 为 0.15/0.075；rejected (0.05,0.04,0.04,0.04)，sum/mean 为 0.17/0.0425；两种归约的 pair margin 分别为 −0.02 和 +0.0325。全为构造输入，不是模型结果。
- `code_status`：DPO `dpo_loss` demo 的可执行语句、张量输入、依赖、运行环境与打印接口未修改，保留正文既有“已验证（2026-10-07；Python 3.13.13；PyTorch 2.12.0+cpu）”状态；未重跑。新数值解释在正文复用同一输入手算，但代码旁没有增加新的确切输出承诺。未运行训练、测试或程序。
- `formula_status`：复核 KL 目标的指数化闭式解及 $Z(x)$ 归一化、reward 的 log-ratio 重排、同 prompt 归一化项抵消、DPO reference-adjusted margin、pair loss/导数、logprob chain-rule sum 和 sum/mean length margin 例；数值与符号方向一致。
- `link_status`：新增 DPO 原论文、SimPO 原论文和 2026 likelihood-displacement 预印本直达链接；章节 04 本地相对 Markdown 目标静态检查存在。保留原有 TRL 固定 revision 文档链接作接口入口，但本轮浏览器对该 commit 页面返回 cache miss；公式依据原始 DPO 论文核验，固定页面仍待全库导航 QA。
- `visual_status`：保留文本化流程对照、偏好样例和代码 demo；公式推导和两组对照算例展示关键关系，当前没有必要新增 Mermaid 图。
- `sync_status`：未改变讲次结构、目录或跨册边界；第 23 讲完成后连续复读第四正文文件并统一复核 `简介.md` 后训练提要。无新增配套题库/术语表同步需求。
- `open_risks`：没有运行 DPO/SimPO 真实模型或 benchmark；偏好损失降低不能证明外部行为改善；2026 预印本仍需全文实验细节与后续复现；固定 revision 的 TRL 页面未能由本轮浏览器读取。第 23–38 讲、章节 04 连续复读及第一册整册复读仍待完成。

## chapter_scan 与精修记录：第 23 讲

- 日期与范围：2026-10-29；`chapters/04-后训练与对齐.md` 第 23 讲，覆盖 RM 的对象与标注目标、chosen/rejected pairwise loss、模型打分结构、RM 校准与任务切片评估、分布偏移、reward hacking、过度优化、过度拒答和独立 verifier；不调整讲次顺序或书册边界。
- 扫描问题与资料范围：核对标注来源是否被误写成一律是人类偏好，Bradley–Terry sigmoid 与 batch 平均目标是否定义完整，RM head pooling 是否被写成统一实现，整体准确率与校准/分领域/真实任务验证的区别，RewardBench 的样本生成和下游相关性限制，proxy 优化退化与 KL 的证据范围，以及过度拒答是否被写成必然结果。阅读 Ouyang §3.5、Stiennon §3.4/§4.3/limitations（此前证据记录见 RLHF research note）、Gao et al. 摘要/§1/§2.1/§3.2–3.6、RewardBench v2 摘要/§1/§5.2/附录 A；版本、结论及边界见 `research/reward-model-evaluation-and-overoptimization-2026.md`。
- 候选主题与去向：
  - Bradley–Terry pairwise loss、数值例和 PyTorch demo：作为 RM 唯一完整主讲保留在第 23 讲；与第 21 讲的高层职责介绍及章节内锚点相连，不重复推导。
  - RM 校准和评估：用 accuracy/loss、分差概率校准、任务切片、反事实扰动和外部 verifier 表呈现；RewardBench 作一个有边界的评测例，不把其准确率等同下游训练收益。
  - 分布偏移、proxy overoptimization、over-refusal：放在排序目标之后讲因果与评估，引用合成和摘要实验但不推成跨任务阈值或安全保证。
  - 专业知识、代码、数学、工具和安全判断需外部任务验证；不新增独立讲次，未发现需改变第 19–23 讲交接的结构候选。
- 修改决定与结果：把 RM 的目标限定为具体标注协议和候选分布下的相对排序；补 `σ` 定义、单 pair loss 与数据集平均的关系及 reward 平移不变性；把 pooling 明确为实现选择；增加评估信号表和 RewardBench v2 的适用边界；将“分布偏移=奇怪回答”改为训练分布之外均可能不可靠，并说明优化的选择效应；用 Gao 合成设定和 Stiennon 摘要实验限定 overoptimization 证据，指出指定 RL 配置中的 KL 结果和敏感性；将过度拒答写成可测风险而非必然因果；删除无来源直引的 Goodhart 句式和未定义的 `B_proxy` 均值公式，改为四对候选分歧率手算；增加代码/数学/事实/工具任务对应的独立 verifier 及风险控制表。
- `length_status`：解释性汉字 3,002，统计排除标题、表格、代码/公式围栏、状态标记、行内代码/公式和链接目标；作为核心讲解单元，高于 1,800 字下限。
- `evidence`：RewardBench v2 支持其 chat/reasoning/safety 与细微可核验错误样本设计及论文报告的 RM 差异；作者同时指出许多 pair 半自动构造后人工复核、某些领域可能有伪相关、下游训练相关性未解决并存在污染可能。Gao 等人用固定 gold RM 生成合成标签并比较 proxy 优化下的 gold reward 曲线；此结果不是人类评审，且依方法、数据、RM/policy 尺寸和配置。指定 RL 配置下的 KL penalty 未测得 gold-reward/KL frontier 改善，作者提示超参数敏感。Stiennon 证据限于 TL;DR 摘要设置。事实和版本详情见 research note。
- `example_status`：pairwise 例设 $r_w=3,r_l=1$，差值 2，概率约 0.8808、loss 0.1269；倒转后概率约 0.1192、loss 2.1269。玩具线性 RM 按权重算得 accurate 6.0、fluffy 8.5，而人为 `true_pref` 是 9/4，展示排序翻转。四个独立 pair 中三个外部顺序与 RM 一致、一个冲突，分歧率为 $1/4=25\%$。全部为教学构造值。
- `code_status`：`pairwise_reward_loss` PyTorch demo 与标准库 toy RM 代码的可执行语句、输入输出、依赖和声明环境未改，保留正文既有 2026-10-07 验证标记；未重跑。pairwise demo 的历史详细命令和边界日志缺口沿用第 21 讲记录，未补造。本轮没有运行程序或模型。
- `formula_status`：明确 sigmoid 定义、单 pair 负对数损失、数据集平均、分差平移不变性并手算正反排序；核算 toy RM 两个加权分数与四对比较分歧率。公式与代码中 `.mean()` 口径一致。
- `link_status`：增加 RewardBench v2 和 Gao 等人的直达来源；Ouyang、Stiennon 来源沿用前文读者入口。全篇相对 Markdown 目标静态检查均存在。
- `visual_status`：新增 RM 评估信号表和风险控制表，帮助对照指标、外部验证与边界；不需要额外 Mermaid 图。
- `sync_status`：无讲次、目录或书册边界变化；连续复读第四正文文件第 19–23 讲后更新 `简介.md` 的后训练摘要。无需同步题库或术语表。
- `open_risks`：未运行 RM 训练、policy 优化、RewardBench 或真实模型评估；玩具代码和算例不代表实测表现。RewardBench v2 的数据生成和下游相关性存在论文自述限制，Gao 结果是合成 gold 设置，外部 verifier 也只覆盖各自目标。第一册第 24–38 讲、整册主路径复读和终审仍待完成。

## chapter_scan 与精修记录：第 24 讲

- 日期与范围：2026-10-29；`chapters/05-推理、部署与评估.md` 第 24 讲，覆盖 logits 到输出选择、greedy、sampling、temperature、top-k/top-p、beam search、repetition/presence/frequency penalty、EOS/end-of-turn 与 runtime stop、幻觉边界、任务评估及 Best-of-N/self-consistency；不调整讲次顺序或书册边界。
- 扫描问题与资料范围：检查“手算例子”是否可复算，top-p 是否明确首次越阈 token 的边界，beam/temperature 是否被写成对现代聊天或事实性的普遍规律，重复惩罚和 presence/frequency 参数是否误当作跨框架统一定义，模型控制 token 是否与运行时停止条件混淆，self-consistency 与 Best-of-N 的选择机制和测试时成本是否区分，受约束解码能保证到什么范围。阅读 Holtzman et al. arXiv:1904.09751v2 摘要、引言、GPT-2/WebText 实验设置及重复/人工评估结果；Wang et al. arXiv:2203.11171v4 摘要、§3.2 和采样配置；Hugging Face Transformers GenerationConfig 与当前 logits_process.py 实现；OpenAI Chat Completions presence/frequency penalty 与 stop 文档。Snell 等人的 test-time compute 来源范围沿用 `research/test-time-compute-and-evaluation-2026.md`。版本、阅读位置及结论边界见 `research/decoding-sampling-and-test-time-compute-2026.md`。
- 候选主题与去向：
  - greedy、sampling、temperature、top-k/top-p、beam 和 repetition penalty：作为本讲核心机制，新增同输入数值手算与任务评估条件。
  - Transformers 当前列出的 min-p、top-h、typical-p、epsilon/eta cutoff：以一张支持表识别其筛选依据；只作为框架选项，不扩写成独立算法课，运行时文档和默认值按固定版本核对。
  - frequency/presence penalty：采用 OpenAI 当前文档的特定公式并给构造数例，明确不外推到其他 API。
  - stop token / stop string / max token budget：拆分模型 token 与 runtime/API 停止条件，避免统一称作 stop token。
  - Best-of-N 与 self-consistency：区分外部选择器和答案聚合；把 test-time compute 预算问题链接到第 29 讲，不新增独立讲次。
  - 受约束解码：作为格式合法性边界简述，不扩成语法解码完整主讲，也不把格式约束等同语义正确。
- 修改决定与结果：修正贪心“局部最优”说法，说明它只逐 token 选当前最大概率且不保留备选前缀；删除无来源的“现代聊天模型通常更常用 sampling”和笼统任务参数数值配方。温度公式写为 $P_T(i\mid c)=\exp(z_i/T)/\sum_j\exp(z_j/T)$，加入减最大 logit、指数权重和归一化概率表；用 $u=0.80$ 手算抽到 B，并比较 $T=0.5$、top-k=3 与 top-p=0.9。增加束宽为 2 的两步累计 logprob 算例，说明贪心与 beam 的分支选择差异；增加 self-consistency 多数票可能错误的构造反例。补 Transformers repetition penalty 和 OpenAI presence/frequency penalty 的逐步 logit/softmax 例子；把 stop 信号拆为 EOS/end-of-turn、最大长度/时间和 stop string；将“温度必然增加幻觉”收窄为按任务实测；增加任务指标表、Best-of-N/self-consistency 对比和候选成本/错误相关性边界。
- `length_status`：解释性汉字 4,885，排除标题、表格、代码/公式围栏、状态行、行内代码/公式及链接目标；本讲是单个核心讲解单元，超过 1,800 字下限。
- `evidence`：Holtzman 的开放生成结果限于 GPT-2 Large（762M）和 held-out WebText 等论文设置，不能推出当前聊天模型的事实错误率或普遍 beam/sampling 优劣；Wang 的 self-consistency 结果限于论文中的 UL2-20B、LaMDA-137B、PaLM-540B、GPT-3、算术/常识任务及每设置 40 输出、10 次运行；Snell 证据限于其数学任务、模型、verifier 和计算预算。HF 源码页面为可变 main 分支，OpenAI penalty/stop 只支持各自当前 API 合同。证据及版本边界见 research note。
- `example_status`：教学 logits [3,2,1,0.5,0] 在 T=1 减最大值后权重约 [1,.3679,.1353,.0821,.0498]、总和 1.6351、概率约 [.6116,.2251,.0828,.0502,.0304]；u=.80 落入 B 区间。T=.5 得 [.8600,.1164,.0158,.0058,.0021]；top-k=3 与 top-p=.9 当前均保留 A/B/C，重归一化 [.6652,.2448,.0900,0,0]。Beam toy 例首步 P(A/B/C)=.50/.30/.10，A 后续 x=.50、B 后续 z=.99；累计 log probability 分别 ln(.50)+ln(.50)=−1.386 与 ln(.30)+ln(.99)=−1.214，束宽 2 将 B→z 保留为高于贪心 A→x 的候选。Self-consistency 例五条路径投票为 8、8、12、12、12，若 gold 是 8 则多数票错。Transformers repetition penalty 例用 logits [2,1,0]、A 已出现、p=2，变为 [1,1,0]，A 概率约 .665→.422。OpenAI 文档定义例用 logits [2,1,0]、计数 [2,1,0]、presence=.5、frequency=.25，变为 [1,.25,0]，概率约 [.543,.257,.200]。以上是构造输入，不是实测模型结果。
- `code_status`：temperature/top-k/top-p PyTorch demo 的可执行代码、输入、输出、依赖和环境未变，沿用原正文“已验证（2026-10-07；Python 3.13.13；PyTorch 2.12.0+cpu）”标记，按规范未重跑。手算概率与原 demo 的三位小数输出一致；未运行代码、API 请求、任务 benchmark、测试或 PDF 构建。
- `formula_status`：核对 temperature softmax、减最大值不改变 softmax、top-p 第一次越阈项、top-k/top-p 重归一化、beam 累计 log probability、正负 logit repetition 变换及 OpenAI additive penalty 公式。数值近似保留 4 位；正文说明所有惩罚公式依实现/接口而定。
- `link_status`：增加 Holtzman、Wang、Snell、Transformers GenerationConfig 与源码、OpenAI penalty/stop 直达链接及第 29 讲文件入口；章节全篇相对链接目标、围栏和行内公式分隔符静态检查通过。
- `visual_status`：新增 top-k/top-p 与扩展筛选方式表、stop 条件表、任务-假设-指标表、Best-of-N/self-consistency 对照表、softmax 数值表和 beam 累计分数表；这些表与逐步算例清晰呈现机制，无需额外 Mermaid 图。
- `sync_status`：无讲次、目录、书册边界变化；添加第 29 讲相对入口。第五正文文件第 25–28 讲完成后，再沿文件路径复读并统一核对 `简介.md` 的推理/部署/评估摘要；无新题库或术语表同步要求。
- `open_risks`：未测试真实模型、API 参数兼容性或推理服务停止行为；HF main 与在线 API 文档会更新，应按实际版本复核；算例仅验证机制，不代表质量收益。第 25–38 讲、第一册整册复读和终审仍待完成。

## chapter_scan 与精修记录：第 25 讲

- 日期与范围：2026-10-29；`chapters/05-推理、部署与评估.md` 第 25 讲，覆盖自回归推理的数据流、prefill/decode、逐层 K/V 缓存、固定长度与变长 batch 的容量估算、MHA/GQA/MQA 对 KV 载荷的影响、TTFT/TPOT/吞吐、batching、continuous batching、PagedAttention、服务瓶颈和缓存压缩/前缀复用；未改变讲次或章节边界。
- 扫描问题与资料范围：核对“无 cache 时重算”的可数工作量、KV 公式对变长请求和并行分片的口径、prefill/decode 的瓶颈是否被写成定律、GQA/MQA 能否被理解为推理时可切换开关、PagedAttention 与 scheduler/cache manager 的职责、碎片收益边界、当前 vLLM 文档是否代表当前实现，以及 2026 KVSET/UltraQuant 结果的负载与硬件适用范围。阅读 Kwon et al. PagedAttention arXiv:2309.06180v1 摘要、§§1–4.6、§§6–7、§10；KVSET arXiv:2609.27746v1 摘要、§§1–3、§5 与限制说明；UltraQuant arXiv:2606.20474v3 摘要、§§2–9、附录 A/B/C/E；vLLM 当前 Paged Attention 设计页的历史文档 Warning。来源版本、阅读位置和事实边界见 `research/kv-cache-capacity-paging-and-compression-2026.md`。
- 候选主题与去向：
  - K/V 重算、prefill/decode 和缓存容量：作为第一册的核心机制保留；用一次 prompt 与两次输出的前向轨迹说明缓存复用，用 `sum(T_i)` 处理变长请求；更复杂的混合注意力、并行分片和 runtime 账本留作目标引擎容量验证。
  - MHA/GQA/MQA：只讲 `H_kv` 对公式的线性影响，并指出这是模型 checkpoint 的结构属性，不是同 checkpoint 的运行时旋钮；结构质量与模型训练沿第 10 讲和专题材料继续。
  - PagedAttention：保留逻辑/物理 block 映射、按需分配、尾部空槽和共享的基础解释；scheduler、block manager、请求生命周期与代码实现转接第二十四册第 17–23、35 讲，不把原论文系统加速数字写成当前系统保证。
  - KVSET：作为 prefix-cache 工作集容量规划的近期支持案例纳入，不替代活动请求 payload 公式；限制在目标 trace、page-granular LRU 和论文的 full-attention cache 范围。
  - UltraQuant：作为低比特 KV 必须联合评估量化误差、kernel 和 SLO goodput 的近期案例纳入；不展开量化算法或 AMD kernel 实现，不把特定 MI355X/CDNA4 replay 数字外推到其他 GPU。
  - 未发现足以增设第一册独立讲次或改变第 24–28 讲顺序的候选；推理引擎的实施细节已有第二十四册多个明确入口。
- 修改决定与结果：补充逐步 KV 复用计数（prompt `a,b,c`，生成 `x,y`），区分每层 K/V 投影位置与因果 Query-Key 配对；扩展变长 batch 的求和公式和 GQA 896 MiB 算例；收窄 MHA/GQA/MQA 的结构关系和 decode memory-bound 表述；定义可比的 TTFT、TPOT、requests/s 与 output tokens/s 口径；说明 continuous batching 的迭代边界；将 PagedAttention 写成 attention kernel、block table、cache manager 与 scheduler 的协同设计，补 block size=4 的物理映射例，并删除“分页直接提高并发”的无条件结论；校正 `max_tokens` 不等于分页缓存启动时全部预分配；区分活动请求 cache 与 prefix-cache 工作集，加入 KVSET 和 UltraQuant 的条件化结果；标注文档页面已提示其为历史实现说明，并添加第二十四册对应专题入口。
- `length_status`：修改后解释性汉字 4,423 字；剔除标题、表格、代码/数学围栏、状态标记、行内代码/数学和链接目标后计数；单个核心讲解单元，达到 1,800 字下限。
- `evidence`：PagedAttention 原论文支持其逻辑 block 到非连续物理 block 映射、按需分配、共享以及与 scheduler/cache manager 协同设计，吞吐实验只支持论文所测模型、系统、baseline 和 workload；vLLM `latest` 设计页明确警告它是历史文档，不描述当前代码。KVSET 支持对 page-granular LRU prefix-cache 做 trace 工作集分析；10,000 请求/24,000 请求内部 trace、SGLang/Mooncake、H20、GLM-5.2 W4A8、FP8 KV 及 95%/99% coverage 容量只属于作者实验，并且算法边界在 LRU 与 full-attention KV。UltraQuant v3 的 goodput 结果限于生产来源 Claude Code trace、MiniMax M2.5/Qwen3-235B、AMD MI355X、TP=2、滚动 P95 TTFT 120 秒门槛和 600 秒窗口；质量结果按模型与任务变化，短上下文低并发下相对 FP8 的优势不能保证。相关事实详见 research note；未外推到其他 GPU 或线上系统。
- `example_status`：单层单头 toy trace 中，prefill `[a,b,c]` 计算 3 个位置的 K/V 和 6 个因果配对；再处理 `x` 得到 `y` 时，使用 cache 追加 1 个位置 K/V 和 4 对，累计 4 个位置/10 对；朴素不缓存重算 4 个位置 K/V 和 10 对，累计 7 个位置/16 对。标准 KV 估算 `L=32,B=1,T=4096,H_kv=32,D_h=128,b=2` 得 2 GiB；`H_kv=8` 得 0.5 GiB，`H_kv=1` 得 0.0625 GiB；假设 6 GiB 可用时等长 MHA 请求的理想算术上限为 3。变长 GQA 请求 4,096/2,048/1,024 token，每 token 128 KiB、总计 896 MiB（不计 block 尾部空槽及元数据）。PagedAttention 教学映射取 block size 4、7 token，logical 0→physical 7、logical 1→physical 1；第 8 token 填入末槽，第 9 token 需新 block。均为构造或公式算例，不是运行时测量。
- `code_status`：KV-cache 显存 demo 的可执行语句、输入输出、依赖、环境与声明结果未变化，沿用正文“已验证（2026-10-07；Python 3.13.13；标准库）”标记；按规范本轮未重跑代码。未运行程序、模型、API、benchmark、测试或 PDF 构建。
- `formula_status`：核算 `M_kv=2×L×H_kv×D_h×b×ΣT_i`，同长度退化为 `2×L×B×T×H_kv×D_h×b`；复算 2 GiB、0.5 GiB、0.0625 GiB、128 KiB/token、896 MiB 和 causal pair counts 6/4/10。公式只表示逻辑 K/V payload，实际分片、复制、量化 scale、block 对齐及 runtime buffers 另计。
- `link_status`：逐一确认本章新增第二十四册 PagedAttention、prefix cache、prefill/decode 相对文件目标存在；打开并核对 PagedAttention、KVSET v1、UltraQuant v3 原始论文及 vLLM 设计页面。全章静态检查：fence 配对、行内数学分隔符、本地相对链接、`git diff --check` 均通过。
- `visual_status`：加入 prefill/decode 重算轨迹表、MHA/GQA/MQA 缓存关系表和服务指标表；PagedAttention 逻辑/物理 block 通过一个可复算例子解释。关系已能由表格和状态轨迹读出，本讲不另加 Mermaid 图。
- `sync_status`：讲次结构、目录、书册边界、题库或术语表未变；新增第二十四册精确专题链接。第五正文文件第 26–28 讲完成后，再沿文件路径连续复读并统一更新 `简介.md` 的推理/部署/评估摘要。
- `open_risks`：未在本地运行 serving engine、复现论文 trace、测量真实请求延迟或验证量化质量；KVSET trace 是内部 workload，UltraQuant 测量有硬件和 SLO 限制；vLLM Paged Attention 页面为历史文档。第 26–38 讲、第五正文文件连续复读及第一册整册复读/终审仍未完成。

## chapter_scan 与精修记录：第 26 讲

- 日期与范围：2026-10-29；`chapters/05-推理、部署与评估.md` 第 26 讲，覆盖 dense attention 的矩阵与 IO 账本、online softmax、tiling/recomputation、前反向存储、dense/sparse/linearized attention 区分、训练与 prefill/decode、长上下文边界及 FlashAttention-1 至 FA4 硬件演进；不改变讲次顺序、篇章边界或书册契约。
- 扫描问题与资料范围：核对 `O(T²)` 是计算还是显式中间存储；online softmax 是否给出可复算的块合并；“exact”是否误成 bitwise 精确或涵盖低精度/稀疏变体；backward 的保存与重算是什么；prefill、decode、TTFT 是否被写成普遍规律；FlashAttention-2/3/4 演进和性能数字是否受硬件、dtype、shape、runtime 与 baseline 约束；相似标题的 FP4 预印本是否容易与 FA4 主线混淆。阅读 Dao et al. arXiv:2205.14135v2 摘要、§§2.2–5、附录 B/C；Dao arXiv:2307.08691v1 摘要、§§3–5；Shah et al. arXiv:2407.08608v2 摘要、§§2.2–5 与 limitations；Zadouri et al. arXiv:2603.05451v1 摘要、§§2–5、附录 A.1；Dao-AILab 官方仓库当前 README；Hu arXiv:2609.04105v1 摘要。版本、阅读范围和证据限制见 `research/flashattention-io-softmax-and-hardware-evolution-2026.md`。
- 候选主题与去向：
  - Dense FlashAttention-1 的 IO-aware tiling、online softmax 与训练重算：保留为本讲主线，补逐步手算和内存/计算量区分。
  - FlashAttention-2/3/4：纳入一张硬件代际表，说明 work partition/occupancy、Hopper asynchronous/TMA/FP8 与 Blackwell pipeline co-design；不独立增讲，性能数字仅作为绑定设置的案例。
  - Sparse/block-sparse 与线性化 attention：用比较表解释连接图和算子是否改变，深层稀疏模式转接第二十一册 `26-稀疏与块稀疏Attention.md`；不把原论文 block-sparse extension 混成 dense exact 核心。
  - `Hardware-Aware FP4 FlashAttention-4`：这是不同作者的独立 arXiv 预印本，不属于 2026-03 FA4 主线论文/仓库版本；记录在资料边界并不纳入 FA1–4 主线，避免名称误读和把 GB200 窄实验写成通用 kernel 能力。
  - 未发现需要新增长讲、调整目录或改变第一册与第二十一/二十四册分工的主题。
- 修改决定与结果：把 dense attention FLOPs 写成 `O(BH_qT²d_h)`，把朴素 score/probability 矩阵存储写成 `O(BH_qT²)`，并说明 FlashAttention 固定 head dimension 下的逐行归一化统计量按 `B H_q T` 增长，但总模型显存不因此变线性；说明 causal mask 的实际 allocation 由实现决定。修正标准 attention score 的缩放和可选 mask/bias；用 score `[0, ln 2, ln 4]`、value `[1,3,5]` 分块计算并合并 `(m,l,u)`，得到 `27/7≈3.8571`。明确 backward 保存输入/输出所需状态与 rowwise log-sum-exp，再 tile-wise 重算 score/probability 和梯度。将“exact”限定为 dense 算法在相同 attention 语义下计算同一算子目标，区分浮点舍入、FP8 误差和改变连接图的 block-sparse；删掉 linear attention 一概等同近似的暗示。收窄训练、prefill/decode、TTFT、长上下文与速度收益断言；增加 8192-token FP16 矩阵手算、FA1–4 代际表及论文/官方实现链接和跨册入口。
- `length_status`：解释性汉字 3,622，剔除标题、表格、代码和数学围栏、状态行、行内代码/数学和链接目标；单个核心讲次，高于 1,800 字下限。
- `evidence`：FA1 v2 支持 dense exact attention、tiling/online softmax、HBM access 降低、O(N) 额外统计量和 backward 重算；同一论文将 block-sparse extension 作为改变连接模式的 approximate attention。FA2 v1 的约 2×/A100 与训练 MFU 数字绑定论文形状与 baseline。FA3 v2 面向 Hopper/H100 的异步执行并评估 FP8 数值误差。FA4 v1 报告 B200、BF16、1k–32k、固定总 token 数设置下的结果，比较 cuDNN 9.13/Triton，并说明新版 cuDNN 纳入部分技术后相近；FA3 未在 B200 上做该论文对比。官方仓库 README 为访问时的可变 `main`，不等于固定 release 合同。Hu 的 FP4 文献是独立 v1 预印本，摘要报告 GB200 窄路径收益和分布式 MXFP4 P/V trajectory divergence，不证明主线 FA4 或通用表现。
- `example_status`：online-softmax 教学输入为 scores `[0, ln2, ln4]`、values `[1,3,5]`，块 A `(m,l,u)=(ln2,1.5,3.5)`，块 B `(ln4,1,5)`；以旧块 scale `0.5` 合并为 `(ln4,1.75,6.75)`，输出 `6.75/1.75=27/7≈3.8571`，与直接权重 `[1,2,4]/7` 相同。内存例设 `B=1,H_q=32,T=8192,b=2`：`T²=67,108,864`；每 head `134,217,728 bytes=128 MiB`，32 heads 一个矩阵 `4 GiB`，S 与 P 同时物化 `8 GiB`；不包括 Q/K/V、输出、梯度和其他层状态。均为公式教学算例，不是 GPU 实测。
- `code_status`：attention matrix memory demo 的可执行语句、输入、输出、依赖和环境与 HEAD 一致，沿用正文“已验证（2026-10-07；Python 3.13.13；标准库）”标记；按规范未重跑。无其他代码块新增或修改。
- `formula_status`：核对 scaled score、`O(BH_qT²d_h)` FLOPs、朴素显式矩阵 `O(BH_qT²)`、固定 head dimension 的 row-statistics `O(BH_qT)`，以及 online-softmax 的 `m/l/u` 合并和 `27/7` 输出；复算 `8192²×2 bytes`、128 MiB/head、4 GiB scores 与 8 GiB scores+probabilities。行内公式分隔符与 math fence 配对通过。
- `link_status`：新增第一册第 25 讲、第二十一册稀疏注意力/FlashAttention、第二十四册目录及 FA1–4、官方实现、独立 FP4 预印本链接；本地相对文件目标存在，当前目录结构未改变。
- `visual_status`：用 online-softmax block 轨迹表显示归一化合并，用 dense/block-sparse/linearized 对比表呈现算子关系，用 FA1–4 表显示问题与硬件演进；没有额外 Mermaid 图的必要。
- `sync_status`：更新 `简介.md` 推理部署摘要；同步改写 `EXERCISES.md` 的注意力题为可手算容量/softmax/数值边界问题，校准 `INTERVIEW_BANK.md` 的 online softmax、复杂度、exactness 和速度回答，细化 `GLOSSARY_EN_ZH.md` 的 FlashAttention/Online Softmax 条目，给 `PAPERS.md` 增加 FA3 阅读入口，并修订第四册比较表关于精确数值和低精度的边界。目录讲次与篇章边界不变；第二十一册的深入实现章节不在本轮重写，后续随第二十一册架构审视处理。
- `open_risks`：未运行 GPU kernel、数值等价性实验、吞吐/显存基准或生产服务；速度和 FP8/FP4 误差未在本机验证。官方仓库 `main` 会变化；FA4 论文 baseline 已随 cuDNN 版本变化；独立 FP4 工作仍是单篇预印本。第五正文文件第 27–28 讲、该文件连续复读和简介摘要复核、第一册整册复读/终审及其他书册仍未完成。

## 2026-10-29：第 27 讲量化与部署

- `scope`：第一册第五正文文件 `05-推理、部署与评估.md` 第 27 讲；覆盖数值格式、权重 payload 和 scale 元数据、线性量化手算、量化粒度、weight-only、PTQ/QAT、GPTQ/AWQ、端到端部署评估；未改变讲次顺序、篇章边界或书册契约。
- `chapter_scan`：检索日期 2026-10-29；问题包括 FP8 是否可视为单一格式、FP4 与 INT4 是否混淆、权重 payload 如何换算到 GB/GiB、scale 元数据是否会改变模型容量估计、GPTQ/AWQ 的校准输入如何参与误差降低、weight bit 是否会被误解为 compute bit，以及 2026 低比特 PTQ 候选是否值得进入第一册主线。阅读 Transformer Engine 2.20.2 FP8 Current Scaling 与 NVFP4 官方文档、GPTQ arXiv v2 方法/校准/实验段落、AWQ MLSys 2024 论文及 arXiv HTML 的激活通道缩放段落、PyTorch QAT 官方博客；近期 AAAC v2、D²Quant v2 和 SliderQuant v1 仅读 arXiv 摘要。版本、具体阅读位置和证据限制见 `research/weight-quantization-formats-and-ptq-2026.md`。
- `candidates_and_disposition`：
  - FP8 E4M3/E5M2：纳入格式表，区分范围和尾数精度；FP8 scale recipe 归于版本化官方实现，不暗示编码本身决定有效张量范围。
  - NVFP4：纳入一条格式/recipe 边界，说明 E2M1 与 INT4 不同；不在第一册展开 block scaling 细节，延伸至第六册量化章节。
  - AAAC、D²Quant、SliderQuant：摘要呈现 adaptive codebook、down-projection/activation correction 与 layer sensitivity 等新方向，但本次未读正文、复现或审计 kernel。它们不改变第一册的入门概念主线，暂留第六册后续章节审视；不把作者摘要的横向性能主张写入正文。
  - GPTQ 与 AWQ：保留为代表性 PTQ 方法；此讲只解释二者不同的误差处理机制，不新增完整算法讲次。
- `modification_decisions`：格式比较改为表格，分别描述 FP32、FP16、BF16、FP8 E4M3/E5M2、INT8、INT4 与 NVFP4；将所有 payload 大小标为十进制 GB 并附 GiB 换算。以 7B 权重、group size 64、每组一份 FP16 scale 手算 metadata 下限，不计 zero-point/alignment/padding。线性量化明确变量、正 scale 和对称 `z=0` 条件；给 `[-1.0,-0.2,0.0,0.3,0.9]` 逐项算 q、反量化值及绝对误差，再以同组异常值 8.0 说明 scale 步长扩大。per-tensor/per-channel/group-wise 统一说明共享轴及元数据/kernel 取舍。weight-only 部分区分权重存储、activation、累加精度和 W4A16 kernel 执行路径；PTQ/QAT 加入 fake quantization 边界。GPTQ 改为 layer-wise reconstruction、校准输入形成的近似二阶敏感度和对未量化权重的误差补偿；AWQ 改为 activation-aware 通道选择及等价缩放，不再泛称“保护权重”。INT8/INT4 的质量比较改为条件化判断，并把 13B/16 GiB 容量例子的 GB 与 GiB 口径统一。
- `length_status`：解释性汉字 3,826；标题、表格、代码、公式围栏、状态行、行内代码/公式和链接目标不计；本讲为核心单元，超过 1,800 字门槛。
- `evidence`：TE 2.20.2 文档支持其 E4M3 最大有限值 448、E5M2 最大有限值 57,344，以及 NVFP4 的 E2M1、16-value E4M3 block scale、FP32 tensor scale 和权重 2D scale 变体；其结论只适用于该版本 recipe。GPTQ ICLR 2023/arXiv v2 支持 layer reconstruction 与 approximate second-order 信息，结果绑定作者模型、数据、位宽和 GPU。AWQ MLSys 2024 支持从 calibration activation 识别 salient channels 并作等价缩放；不据此断言普遍胜过其他方法。PyTorch QAT 博客支持在浮点训练中用 fake quant 模拟低精度数值；不证明目标 runtime 自动支持对应模型格式。近期候选论文只根据摘要记为待进一步核验。
- `example_status`：INT8 例取 `q_max=127`、`z=0`、`s=1/127`，`[-1,-.2,0,.3,.9]` 映射为 `[-127,-25,0,38,114]`，逐项反量化并列绝对误差；若同组出现 8.0，则 `s=8/127`、步长增至 8 倍，0.3 映射为 5 并反量化约 0.3150。容量例：`7e9×4/8=3.5e9 bytes=3.5 GB≈3.26 GiB`；109,375,000 个 FP16 scales 共 218,750,000 bytes，payload+scales 为 3.719 GB≈3.46 GiB（不含其他元数据）。部署例：13e9 权重参数下 FP16/INT8/INT4 payload 分别为 26/13/6.5 GB，即约 24.21/12.11/6.05 GiB，对照 16 GiB 预算并说明留给 KV、activation 和 runtime 的空间。
- `code_status`：本讲原 PyTorch demo 相较 HEAD 的 Python 代码块完全未变，依赖和输入输出未改；沿用“已验证（2026-10-07；Python 3.13.13；PyTorch 2.12.0+cpu）”正文标记。本轮未运行该 demo、测试、模型、kernel 或 benchmark。
- `formula_status`：检查通用线性量化 `q=clip(round(x/s)+z,q_min,q_max)` 与 `x_hat=s(q-z)`，明确 `s>0`；复算 INT8 各行量化和 7B/13B payload、group metadata 及 GB/GiB 换算。Math fence 配对、行内 `$` 分隔符通过。
- `link_status`：新增第一册到第六册 `07-量化与模型压缩.md` 和 `17-native-int4与原生量化部署.md` 的相对入口；新增 TE、QAT、GPTQ v2、AWQ MLSys 2024 直接来源。本地相对目标均存在。`PAPERS.md` 已有 GPTQ、AWQ、NVFP4 与 QAT 的资料入口，本轮未重复添加。
- `visual_status`：格式表呈现范围/位宽差异；量化粒度表呈现 scale 共享范围与实现代价；手算误差表和 GB/GiB ledger 展示具体数值关系。无需 Mermaid 图。
- `sync_status`：更新 `EXERCISES.md` 的 7B payload+scale 账本与 GPTQ/AWQ 机制题；更新 `INTERVIEW_BANK.md` 的量化定义、INT8/INT4、PTQ/QAT、weight-only、GPTQ/AWQ 回答；更新 `GLOSSARY_EN_ZH.md` 的 GPTQ/AWQ 定义并加入 W4A16。章节结构与简介摘要无需调整。
- `open_risks`：未在本机测量 GPU 量化精度、实际 checkpoint 文件大小、目标 kernel、质量回归或 serving 延迟；具体精度损失和速度收益不能由教学算例推断。AAAC/D²Quant/SliderQuant 正文和实现仅在后续候选审视时补验。第五正文文件第 28 讲、连续复读与简介摘要复核、第一册第 28–38 讲及整册复读/终审、其他书册仍待处理。

## chapter_scan 与精修记录：第 28 讲

- `scope`：2026-10-29；第一册 `chapters/05-推理、部署与评估.md` 第 28 讲，覆盖幻觉定义与诱因、拒答评分、评估标签噪声、benchmark contamination、LLM-as-a-Judge、安全评估、jailbreak / prompt injection、red teaming 与回归闭环。未改变讲次顺序、章节边界或书册契约。
- `chapter_scan`：核对不同论文的“幻觉”是否被混作一个标签；二元准确率是否把拒答与错误断言按同一损失计分；幻觉检测 gold label 的噪声如何影响判断；文本重叠、时间切分、私有 holdout 和黑盒行为信号各自能支持什么；污染缓解是否损害 benchmark fidelity；LLM judge 的 position/verbosity/self-preference bias 是否有校准步骤；jailbreak 与直接/间接 prompt injection 是否被误写成互斥类别；防御是否落实到宿主权限和副作用边界。
- `候选主题与去向`：准确率评分与拒答激励纳入 Helpful/Honest 数例，限制为特定评分设定；TRIVIA+ 标签噪声纳入评估标签可信度；ICLR 2024 顺序似然检验和 ICML 2025 污染缓解保真度作为检测方法边界，不新增独立讲次；judge 偏差、校准与人工/程序检查保留本讲摘要，深讲链接第七册第 3–5 章；OWASP 术语关系和应用侧安全控制纳入本讲，具体线上治理链接第六册第 11 章。没有需要调整目录或新增讲次的主题。
- `modification_decisions`：将幻觉操作化为错误断言、违背证据或无证据推断被表述为事实，并把事实正确、evidence faithfulness/grounding、引用归因、推理、工具结果编造分开记录；指令失败、格式错误、过度拒答和不安全服从归为独立行为指标。移除“安全幻觉”这一混合标签。保留 10 题准确率冲突的教学假设，说明拒答计分条件和结果并非实验数据；不把 SFT/RLHF 写成幻觉的必然原因。将污染检测方法改为信号与边界表，增加候选匹配不等于污染证明、似然检验前提、私有 holdout 仍有暴露风险，以及污染缓解与评测保真度取舍。LLM judge 加固定 rubric、盲化、顺序交换、人评/程序校准、弃判和人工升级要求。把 jailbreak 作为 OWASP LLM01:2025 分类中的 prompt injection 子类，直接/间接标注按输入来源区分；prompt 不作为安全边界，防护落到宿主权限、最小权限、模拟工具和状态校验。复核 support ratio 的断言分母、数值例和空集合 N/A 处理。
- `length_status`：解释性汉字 5,424；排除标题、表格、代码/公式围栏、状态行、链接目标、行内代码和行内数学；第 28 讲为核心单元，高于 1,800 字下限。长度是必要门槛，教学链仍按正文人工复读。
- `evidence`：研究来源、版本、阅读范围和边界见 `research/hallucination-evaluation-and-safety-2026.md`。Nature 正文链接在本轮访问时跳转身份验证，使用先前官方摘要/预览范围支持有限结论，不延伸到未核对的完整实验；ACL Anthology、ICLR、PMLR、arXiv 与 OWASP 页面可访问范围记录在研究笔记。
- `example_status`：10 题例假设 8 道有证据题均答对、2 道证据不足题；若 abstention 按精确匹配记错，A 得 `8/10=0.80`，B 在未知题上猜对一题、错一题则得 `9/10=0.90` 但制造一项错误断言。引用审计例为 5 个可核验断言中 3 个被引用来源充分支持，`A_support=3/5=0.60`；与事实正确率分开。安全评估表按高风险、正常/边界、缺证据、含不可信外部内容切片，指标需报告样本数和分母；演示文本使用合成内容，不调用真实工具或凭证。
- `code_status`：L28 Python demo 与 HEAD 相同，输入/输出/依赖未变，保留正文 2026-10-07 Python 3.13.13 标准库验证标记；未重跑代码、测试或 benchmark。全文未运行模型、攻击测试或 PDF 构建。
- `formula_status`：核对 `A_support=|C_s|/|C|` 中 `C` 为可独立核验断言、`C_s` 为被明确引用来源实际支持的断言；无断言时为 N/A。行内公式分隔符成对，math fence 成对；现有数例按教学输入复算。
- `link_status`：增加第一册到第七册人评/pairwise、LLM judge、污染检测三章及第六册监控评估安全章的本地入口；本地目标均存在。Nature、ACL、ICLR、PMLR、arXiv 与 OWASP 外部来源已核对可访问性/阅读边界；OWASP LLM01:2025 同步至 `PAPERS.md`。
- `visual_status`：使用错误标签对照表、污染检测信号/边界表、安全风险切片表帮助读者比较对象和指标；现有数值例可复算，无需新增 Mermaid。
- `sync_status`：更新 `简介.md` 中推理/部署/评估摘要；`EXERCISES.md` 区分断言级事实、证据、推理、工具错误与行为安全失败，并要求各指标定义分母；`INTERVIEW_BANK.md` 更新 contamination 定义和检测边界；`GLOSSARY_EN_ZH.md` 对齐幻觉、LLM judge、jailbreak 与 prompt injection 术语；`PAPERS.md` 增加 OWASP LLM01:2025 入口，Nature、TRIVIA+、ICLR contamination、PMLR 2025 和 judge 研究已有条目，未重复添加。
- `continuous_read`：第 28 讲后从第 24 讲连续复读本正文文件至结尾，复核 24→25 的生成/缓存交接、25→26 的服务/attention 交接、26→27 的 kernel/量化边界、27→28 的部署/可靠性衔接及简介提要；发现第 24 讲将 greedy 特点误接为 batching 收益，已改为 greedy 特点；合并重复的采样实验说明，保留过滤顺序、候选交集和结构校验边界。
- `static_validation`：第 28 讲解释性汉字 5,424；backtick fence 10 个、tilde fence 2 个，均成对；math fence 1 对，行内 `$` 数量为偶数；4 个本地 Markdown 链接全部存在；L28 Python code block 与 HEAD 相同；`git diff --check` 通过。未运行自动化测试或 PDF 命令。
- `open_risks`：未在目标模型、数据分布或生产系统上测量幻觉率、judge 校准、拒答策略、污染暴露或 prompt injection 缓解效果；Nature 正文本轮受身份验证限制，结论维持在可核验摘要范围；安全章节示例为教学级合成测试，不是上线安全认证。第 29–38 讲、第一册整册复读与终审、book-02 至 book-24 及补充篇仍待处理；下一步处理第 29 讲。

## chapter_scan 与精修记录：第 29 讲

- `scope`：2026-10-29；第一册 `chapters/06-研究前沿与综合实验.md` 第 29 讲“开放研究问题”。检查开放问题如何落到研究卡片、推理迁移和长上下文数例、研究方向范围、面试复核题及来源边界；第 30 讲不在本次修改范围。
- `chapter_scan`：检查研究方向是否能写成可证伪问题，基线和预算是否可比，指标是否包含明确分母及质量/成本/风险，研究结论能否区分机制与替代解释；检查 test-time compute、长上下文证据使用、多模态 grounding、合成数据、安全/帮助性、verifier 与 benchmark integrity 是否有清楚的交接入口。复核了 `research/test-time-compute-and-evaluation-2026.md`、`research/scaling-laws-data-mixtures-2026.md`、`research/rope-long-context-evaluation-2026.md` 中对应来源的阅读范围和结论边界。
- `candidates_and_disposition`：将推理迁移、test-time compute、长上下文、多模态 grounding、合成数据、安全与帮助性、可验证性/评测激励、benchmark integrity 纳入研究方向表；没有发现需要新增独立讲次或改变第一册边界的证据。推理时计算的收益限定到论文模型、数学任务和预算条件；Lost in the Middle 限定为特定模型/任务的位置效应；Kang 的合成数据比例与 Roberts 预印本的比例拟合均不作为通用配方；其余主题保留为实验入口，深入内容交接第七、八、十五、十六册及本册第 17 讲。
- `modification_decisions`：用研究方向地图把宽泛议题改写为问题、对照、指标/边界和跨册入口；保留长上下文研究卡片，明确按召回、定位、引用与评测链条归因。补充推理迁移和干扰鲁棒性的假设数例，加入干净版/干扰版配对四格表，说明边际准确率差与配对结构的关系。将重复的通用研究计划步骤合并为实验结果判读和停止条件；把“只报结论、只看总分、方向太空泛”等笼统提示改为具体错误推断及对应的控制方法。复核习题，使其要求样本单位、反事实/消融、分母、预算、风险和停止条件。
- `length_status`：解释性中文 1,957 字；剔除标题、表格、围栏、行内代码、公式和链接目标后计汉字。本讲作为一个核心教学单元，达到 1,800 字下限；篇幅门槛不替代论证质量判断。
- `evidence`：本讲复用现有来源记录，不新增研究事实。Snell 等人的 TTC 结论按数学推理任务、模型、verifier/修订方法与 FLOPs 口径限定；Lost in the Middle 只支持其报告模型与任务的位置效应；Kang 等人的结果绑定改写/教材文本定义、数据预算和实验设置；Roberts 等人为 2026 年 arXiv 预印本，其拟合/外推不写成已验证配方。长上下文、多模态与评测入口分别链接已有专题章节。
- `example_status`：假设输入 200 道分布内题与 200 道难度配平的语义改写/新组合题，原题 `164/200=0.82`，变体题 `122/200=0.61`，描述性迁移差 `-0.21`；明确该差异不能证明“没有推理”。干扰数例设 200 道同题配对样本：两种条件都对 140、仅干净版对 24、仅干扰版对 8、两种条件都错 28；复算得干净版 `164/200=0.82`、干扰版 `148/200=0.74`、配对净差 `(24-8)/200=0.08`。均为构造的教学数据，不是模型实测；正文要求基于配对结果报告不确定性。
- `code_status`：本讲没有可执行代码；三处 `text` 围栏用于模板/示意输入，不作为程序示例。未运行代码或测试。
- `formula_status`：推理迁移差、鲁棒差、分母及配对四格合计均已复算；L29 行内 `$` 共 22 个，成对；未新增公式围栏。
- `link_status`：检查 L29 的 17 个 Markdown 链接，所有本地目标均存在；外部链接直达 Snell、Lost in the Middle、Kang 与 Roberts 来源。暂无章节拆分，未改目录。
- `visual_status`：方向地图用比较表呈现八个议题的可检验问题、指标边界和深入入口；新增同题配对的 2×2 结果表，便于区分一致样本与条件翻转样本。没有新增 Mermaid 图的必要。
- `sync_status`：`简介.md` 的本讲提要改为强调假设、基线、反事实、指标和边界；`EXERCISES.md` 更新开放研究题的研究卡片交付要求；`INTERVIEW_BANK.md` 更新开放研究问题的推理迁移、预算比较和多模态证据回答框架。`PAPERS.md` 已包含 Snell、合成数据与长上下文入口，不重复添加。术语表、目录与书册边界无须变化。
- `static_validation`：L29 解释性中文 1,957 字；全章 Markdown 围栏 7 组并配对；L29 行内公式分隔符平衡；17 个本地/外部链接中无失效本地目标；推理迁移与配对数例复算通过；`git diff --check` 通过。未运行自动化测试、模型实验、benchmark 或 PDF 构建。
- `open_risks`：本讲示例不代表模型实测；正式实验仍须审查变体等价性、难度和污染，并报告配对不确定性。研究方向地图不是领域综述，也不宣称穷尽候选。第 30 讲、第一册整册复读与终审，以及 book-02 至 book-24 和补充篇仍待完成；下一步处理第 30 讲。

## chapter_scan 与精修记录：第 30 讲

- `scope`：2026-10-29；第一册 `chapters/06-研究前沿与综合实验.md` 第 30 讲“综合实验、失败分析与知识迁移”。覆盖实验问题定义、固定条件、基线/反事实、质量/成本/风险指标、结果解释、失败定位、上线门槛和回退；未改变讲次顺序、文件边界或书册契约。
- `chapter_scan`：检查实验步骤是否能复现且归因，质量、风险和成本是否使用明确单位与分母，聚合结果是否掩盖切片回归，失败样本能否导向最小区分实验，以及候选提升是否受硬性发布门槛约束。复读第 29–30 讲完整文件后确认开放研究方向自然交给综合实验，没有需增删讲次的主题。
- `candidates_and_disposition`：保留 FlashAttention 作为“从机制假设到目标负载测量”的例子，并在正文链接第 26 讲和 Dao 等人的原论文；新增优化器文档问答助手作为端到端教学实验，连接检索、生成、证据支持、结构化输出与发布决策。所有数字为明确标注的假设输入，不是新研究发现或生产指标；无须新增 research note。
- `modification_decisions`：把原先的六个实验动作改写为任务/失败成本、冻结条件、可证伪假设、单因素对照、分母/切片/不确定性和失败归因/回退；区分实验执行记录与对外因果解释。贯穿例固定 200 个样本与系统配置，只增加 reranker；定义有证据/缺证据情况下的不同成功条件，加入成对基线/候选通过表和样本切片表；再独立计算无依据断言、归一化每请求成本、单位成功任务成本和 P95 延迟。以预先声明的质量、风险、单位成本、P95 门槛判定候选不能发布，并指向检查新失败样本的最小归因实验。FlashAttention 机制段落链接现有原论文入口，范围与其固定设置一致。
- `length_status`：解释性中文 2,726 字；排除标题、表格、代码围栏、公式、行内代码和链接目标后计汉字。第 30 讲为核心单元，超过 1,800 字下限。
- `evidence`：新增统计数字及发布门槛是特意构造的教学样本，不能作为模型效果、成本、质量或线上收益事实。FlashAttention 的 IO-aware tiling、online softmax、dense 算子 exactness 与 FLOPs 边界引用 Dao et al. arXiv v2 原论文；来源阅读范围和硬件/负载边界见 `research/flashattention-io-softmax-and-hardware-evolution-2026.md`。
- `example_status`：同一 200 题的配对表为“两边通过 150、仅基线通过 10、仅候选通过 18、两边未过 22”，复算基线 `160/200=80%`、候选 `168/200=84%`、点差 `+4pp`。按 160 道有证据题与 40 道缺证据题切片后，基线/候选分别为 `132/160=82.5%` 与 `140/160=87.5%`、`28/40=70%` 与 `28/40=70%`。无依据断言按全部 200 个问题计为 `4/200=2%` 对 `8/200=4%`；归一化总成本 200/250 单位，单位成功成本 `1.25` 对 `250/168≈1.488`，相对增加约 `19.05%`；P95 从 `0.80s` 到 `1.00s`，增加 `25%`。质量配对区间未算，正文将该门槛标为证据不足；风险、成本、延迟门槛均未过，因此不发布。输入仅为教学假设。
- `code_status`：本讲没有可执行代码；围栏块是实验记录/失败复盘文本模板，未运行代码或测试。
- `formula_status`：L30 行内 `$` 计 14 个，成对；复算配对总数 200、样本切片总数 200、`+4pp` 质量点差、单位成功成本增幅约 `19.05%` 及 P95 `+25%`。未添加 math fence。
- `link_status`：本讲 2 个 Markdown 链接（第 26 讲本地文件、FlashAttention 原论文），本地目标存在。
- `visual_status`：使用配对成功四格表、answerable/unanswerable 切片表与预设门槛表呈现不同层级的证据；现有失败复盘模板保留为记录结构，不增 Mermaid 图。
- `sync_status`：更新 `简介.md` 的研究与综合实验摘要；在 `EXERCISES.md` 增加手算 paired lift、切片、风险率、单位成功成本和发布门槛的任务；在 `INTERVIEW_BANK.md` 增加质量提升但风险/成本/延迟恶化时的发布判断题。`PAPERS.md` 已有 FlashAttention 入口，无需重复添加；目录与术语表无须改动。
- `continuous_read`：在第 30 讲完成后，从 `chapters/06-研究前沿与综合实验.md` 开头连续复读至文件末尾；检查文件导读流程图、L29 研究地图/配对例、L30 发布决策案例及其过渡和下一部分交接。研究问题到综合实验的路径连贯；未发现需要调整正文顺序或文件分工的问题。
- `static_validation`：L30 解释性中文 2,726 字；整篇围栏 7 组并配对，L30 行内公式 14 个美元分隔符成对；本讲本地链接均存在；全部教学数例复算通过；`git diff --check` 通过。未运行测试、模型、benchmark 或 PDF 命令。
- `open_risks`：发布案例是教学数据，不是模型实验；其配对置信区间没有给出，因此正文只示范“证据不足”判定。真实候选仍须使用代表性数据、固定配置、目标成本单位和实际服务分位数测量。第一册尚未完成整册复读与终审，下一步处理第 31 讲多模态理解与表示。

## chapter_scan 与精修记录：第 31 讲

- 日期与范围：2026-10-29；`chapters/07-多模态理解与表示.md` 第 31 讲，覆盖级联、LLM 中心式融合、统一模态建模，视觉/音频/视频表示、token 预算、对齐/生成目标、证据评估、GPT-4o 公开能力边界和 patch-count demo；未改变讲次结构或本册分工。
- `chapter_scan`：检查架构路线是否被误写成穷尽分类；LLaVA、BLIP-2、Flamingo 的接入机制是否有原论文支持；Qwen3.5 early-fusion 是否限于发布方声明；图像 patch、视频 tubelet 和音频 hop 是否可手算且未冒充真实 tokenization；GPT-4o 与 OmniCapBench 的证据范围是否准确。阅读三篇原论文、Qwen3.5 官方模型卡、GPT-4o 发布说明/System Card 和 OmniCapBench arXiv 摘要；版本、阅读位置和边界见 `research/multimodal-systems-and-evaluation-2026.md`。
- `candidates_and_disposition`：将 LLaVA projector、BLIP-2 Q-Former、Flamingo Perceiver Resampler/gated cross-attention 作为设计实例，不据此穷尽分类；保留 Qwen3.5 early-fusion 声明，不推断 token 路径/recipe；OmniCapBench 只记录摘要明确的 786 视频与任务错误类别，作为限定的评测案例；原始 patch/tubelet 数量级只解释输入长度，不推断模型 token/cost。未发现应增讲或调整章节关系的候选。
- `modification_decisions`：补全 ViT、patch、latent token、neural codec、tubelet、VAE、STFT 首次释义；补 LLaVA/BLIP-2/Flamingo 机制差异和原论文入口；修复失效的 Qwen3.5 source-note 链接；逐步算出 336×336/patch14 的 576 块、16 帧/tubelet2 的 4,608 块和 20ms=0.02s 下 3,000 hops；把第 31.9 节标题改为视觉块数，收窄 OmniCapBench 与 GPT-4o 公开证据范围。
- `length_status`：解释性中文 3,442 字；剔除标题、代码/公式围栏、表格、状态行、行内代码/公式和链接目标后计汉字；核心单元超过 1,800 字下限。
- `evidence`：LLaVA、BLIP-2、Flamingo 结论限于各论文；Qwen3.5 early-fusion 是官方模型卡声明；GPT-4o 发布说明/System Card 仅支持公开输入输出和产品体验，不证明内部路由；OmniCapBench arXiv 2610.12458 于 2026-10-08 提交，本轮只读摘要，全文与复现未核验。
- `example_status`：`336/14=24`，空间块 `24×24=576`；`16/2=8` 个 tubelet，视频共 `8×24×24=4,608`，为图像 8 倍；`60s/0.02s=3,000` hops。全部为指定输入和补齐规则下的教学计算。
- `code_status`：视觉块数 Python demo 可执行语句与 HEAD 相同，保留正文 2026-10-07 Python 3.13.13/标准库验证标记，未运行。维护记录未找到历史命令和完整输出，未补造执行细节。
- `formula_status`：核对 patch、tubelet、音频 hop 公式和上述手算；全篇围栏及 L31 行内公式分隔符通过静态检查。
- `link_status`：L31 共 12 个链接，本地目标静态检查存在；新增模型论文、模型卡、System Card、OmniCapBench 和 research 入口。
- `visual_status`：沿用系统路线流程图、token 预算推导和任务评估表，无需另加图。
- `sync_status`：更新简介、`EXERCISES.md`、`INTERVIEW_BANK.md`、`GLOSSARY_EN_ZH.md`；`PAPERS.md` 已有来源，无重复增加。讲次和目录不变。
- `static_validation`：L31 解释性中文 3,442 字；整篇 33 组 Markdown 围栏配对、14 个行内 `$` 成对、12 个链接的本地目标存在；数值例与代码补齐规则一致；`git diff --check` 通过。
- `open_risks`：未运行图像/视频模型或 benchmark；OmniCapBench 全文待核；公开模型卡和产品材料不披露完整内部路径；代码历史命令记录缺失。第 32–38 讲与整册门禁待处理。

## chapter_scan 与精修记录：第 32 讲

- 日期与范围：2026-10-29；`chapters/07-多模态理解与表示.md` 第 32 讲“CLIP 与图文对齐”，涵盖双编码器、单位向量/温度、batch 对比目标、zero-shot/prompt ensemble、SigLIP、评估和代码 demo；未改变章节结构或跨册分工。
- `chapter_scan`：核对单正样本对角线是否被误解成真实语义标签；余弦、logit scale、温度是否区分；3×3 算例是否能复算；prompt ensemble 是否将 embedding average 和 logit average 混称；SigLIP loss 标签、归一化与 batch 复杂度是否准确；SigLIP 2 的结果是否被错误归因于 loss；原始 CLIP 的数据重叠/偏差研究是否被过度外推。阅读 CLIP 原论文 v1 §§2.2–2.3、3.1.4–3.1.5、5–6，SigLIP 摘要与 Google Research `big_vision` 当前 `main` trainer，SigLIP 2 arXiv v1 §§2.1–2.4、3.1–3.5；来源边界见 `research/clip-contrastive-learning-and-evaluation-2026.md`。
- `candidates_and_disposition`：保留 CLIP 双向 softmax 主线，明确 diagonal 是单正样本 batch 标签；把 logit scaling 和 prompt ensemble 讲清；SigLIP 作为成对 sigmoid 变体，强调全量 pair 仍有 `B²` 项且不自动消除伪负例；SigLIP 2 作为多目标训练配方更新，不增独立讲次或模型排行榜；原始 CLIP overlap/bias 分析只支持论文自身模型与评测。
- `modification_decisions`：解释 `α=1/τ` 与乘法 logit scale 关系，标出 3×3 输入为已缩放 logits，列出三个指数权重、`0.926` 概率和 `0.077` 行损失；补 CLIP prompt 与 overlap 原论文入口；将 SigLIP 写成显式归一化量 `Z`，说明原始全 pair 配方为 `B²` 项除以 `B`；新增 `s=1,b=0` 的正负标签手算，说明 no global softmax 不代表 pair 数线性下降；增加 SigLIP 2 配方和证据边界。
- `length_status`：解释性中文 2,950 字；排除标题、表格、代码/公式围栏、状态行、行内代码/公式及链接目标后计汉字；核心单元超过 1,800 字下限。
- `evidence`：CLIP 原论文支持双向交叉熵、WIT 数据集、prompt 与该论文 overlap/bias 分析，限定于 2021 模型/数据/检测方法。SigLIP 摘要支持无全局 softmax 的 per-pair 目标及作者实验；官方 trainer 当前 main 支持对角线正/非对角负、scale+bias 和 `sum/B` reduction，不外推到所有复现。SigLIP 2 论文支持其组合 recipe 和所报告任务结果，不能单归因 sigmoid loss。来源细节见 research note。
- `example_status`：CLIP 行权重约 `54.598+2.718+1.649=58.965`，配对概率 `54.598/58.965≈0.926`，行 CE `≈0.077`。SigLIP 单 pair `s=1,b=0` 时，正标签 loss `≈0.313`、负标签 loss `≈1.313`；均为合成教学输入。
- `code_status`：PyTorch CLIP demo 的可执行语句、输入输出、依赖和环境未变，保留正文“已验证（2026-10-07；Python 3.13.13；PyTorch 2.12.0+cpu）”标记，未重跑；progress 中无历史命令/完整实测输出，未补造。
- `formula_status`：核对归一化点积、`α=1/τ`、双向 CE、CLIP softmax 数例、SigLIP `y(s+b)` 和 full-pair `B²/B` 归一化。
- `link_status`：新增/强化 CLIP prompt 与 data-overlap 原论文链接、SigLIP 论文及官方 trainer、SigLIP 2 论文来源；全文件共 26 个链接，其中 6 个本地目标均静态存在。
- `visual_status`：沿用 3×3 相似度矩阵与损失公式，数值权重和 pair loss 已逐步解释；无须新增 Mermaid。
- `sync_status`：更新简介摘要、`EXERCISES.md` CLIP/SigLIP 手算题、`INTERVIEW_BANK.md` SigLIP 复杂度/归一化/SigLIP 2 问答和术语表 SigLIP 2 条目。`PAPERS.md` 已有入口，无重复添加。
- `static_validation`：第 32 讲解释性中文 2,950 字；整文件 33 组 Markdown 围栏配对；L31 的 14 个、L32 的 0 个行内 `$` 分隔符均平衡；全篇 26 个链接中 6 个本地目标全部存在；L32 两个 fenced code blocks 与 HEAD 完全相同。SigLIP 数值按公式手算复核；`git diff --check` 通过。未运行代码、测试、benchmark 或 PDF。
- `open_risks`：未运行 CLIP/SigLIP/SigLIP 2 checkpoint 或 benchmark；SigLIP 2 为作者报告实验；官方 trainer 指向可变 main；CLIP 历史 overlap 研究不能替代后来模型审计。第 33–38 讲、同文件连续复读和整册门禁待处理。

## chapter_scan 与精修记录：第 33 讲

- 日期与范围：2026-10-29；`chapters/07-多模态理解与表示.md` 第 33 讲，覆盖视觉/文本数据流、张量维度、projector/Q-Former/resampler/cross-attention、视觉 token 接入、视觉对齐与指令微调、assistant loss mask、高分辨率处理、证据评估、幻觉和多模态提示注入；不改变讲次结构、目录或本册/第十五册分工。
- `chapter_scan`：核对连接器类别是否互斥、维度对齐与 token 压缩是否混为一谈、BLIP-2 query 压缩是否被误当成整体推理成本比例、LLaVA 两阶段流程能否外推为通用训练配方、assistant-only SFT 的 mask/reduction 口径、TaskScore 的量纲和权重、幻觉机制是否被写成已证实的普遍因果，以及图像内注入是否由提示词标注就能防住。定向阅读 BLIP-2 原论文摘要/引言/§3.1–3.3、LLaVA 原论文摘要/§3–4、POPE 摘要/§3.1、HallusionBench 摘要/§3 与图像编辑/无图对照示例、OWASP LLM01:2025 当前页面；来源范围及未验证边界见 research note。
- `candidates_and_disposition`：不增加讲次；把 projector 的维度映射、Q-Former/Perceiver 的序列压缩、prefix/cross-attention 的读取位置拆成三条可组合接口轴；将 BLIP-2、LLaVA、Flamingo 作为具体实例；保留 SFT mask、视觉证据评分、视觉幻觉和多模态提示注入作为本讲的训练/评估/安全收束。
- `modification_decisions`：为 `W` 与 `b` 补充 shape 和广播语义；新增接口对照表；以 BLIP-2 原论文的 `257×1024` 特征和 `32×768` query 输出手算 `257/32≈8.03×`、225 个位置减少和 `87.55%` 序列缩短，明确不代表整体成本降幅；把训练阶段限定为 LLaVA 的论文设计，不写作行业通例；增加两目标 token 的 NLL 求和与 token 均值；增加合成 `TaskScore=0.63` 算例并展示低 Evidence 分项；把幻觉机制描述为可检验排查假设，补 POPE/HallusionBench 任务和边界；将提示词/来源标注定义为软提示，应用侧权限、最小授权和高风险确认作为执行控制。
- `length_status`：解释性中文 3,460 字；剔除标题、代码/公式围栏、表格、状态行、行内代码/公式和链接目标后计汉字；核心单元超过 1,800 字下限。
- `evidence`：BLIP-2 原论文支持冻结编码器/LLM、query bottleneck、32 queries 和指定 `257×1024→32×768` 表示，以及生成阶段全连接映射/视觉前缀；数字只用于序列位置手算。LLaVA 原论文支持可训练投影、论文中的两阶段训练和 GPT-4 视觉指令数据构造，合成数据质量仍待独立验证。POPE 只支持其对象存在轮询协议；HallusionBench 只支持其图像上下文诊断构造和作者评测；二者都不证明普遍错误率或某个具体失败的因果根因。OWASP LLM01:2025 是动态风险指南，不是特定模型或防护措施的独立有效性实验。证据细目见 `research/multimodal-systems-and-evaluation-2026.md`。
- `example_status`：BLIP-2 表示长度 `257/32=8.03125`，减少 `225` 个位置、`225/257≈87.55%`；NLL `-ln(0.8)-ln(0.5)≈0.9163` nat，均值 `≈0.4581` nat/token；`TaskScore=0.30×0.90+0.40×0.25+0.20×0.80+0.10×1.00=0.63`。后两项是教学假设；没有视觉模型或 benchmark 实测。
- `code_status`：形状 demo 的可执行语句、输入输出、依赖与声明环境未变，三个 fenced blocks 与 HEAD 相同；静态复核 `[2,6,12]→[2,6,16]` 与文本 `[2,8,16]` 拼接为 `[2,14,16]`。未在 progress 中找到该块的既往命令和完整执行记录，故把正文旧“已验证”状态改为“静态审阅，未运行（2026-10-29）”；本轮未运行代码。
- `formula_status`：核对 `W∈R^{d_v×d_l}`、`b∈R^{d_l}` 和 batch/token 维广播，核对 query 长度比例、sum/mean NLL 与加权 TaskScore；数值代入已复算。
- `link_status`：L33 有 9 个外部来源链接；第七正文文件共 30 个 Markdown 链接，其中 6 个本地目标全部存在。
- `visual_status`：新增连接器接口维度对照表，明确投影、压缩和读取机制可组合；现有数据流代码围栏继续说明 end-to-end 形状，无需新增 Mermaid 图。
- `sync_status`：新增/修改 `EXERCISES.md` L33 的 VLM shape/loss 手算及视觉诊断练习，校准 `INTERVIEW_BANK.md` VLM 接入/架构/assistant mask/幻觉回答，给 `GLOSSARY_EN_ZH.md` 增补 HallusionBench，给 `PAPERS.md` 多模态评测加入 HallusionBench。简介摘要已有 VLM 视觉推理概述，无需再改；讲次、目录和交叉导航均未变化。
- `static_validation`：L33 解释性中文 3,460 字；全文件 66 个围栏标记为 33 组配对，L33 9 组围栏成对；L33 行内 `$` 分隔符为 0；全文件 30 个链接中 6 个本地目标存在；L33 9 个来源链接为外链；代码块与 HEAD 相同；三段手算复核；`git diff --check` 通过。没有运行自动化测试、代码、模型、benchmark 或 PDF。
- `open_risks`：未运行视觉语言模型或 POPE/HallusionBench；评测论文结论只限其任务和数据；OWASP 页面动态更新；教学加权分数不是产品 rubric；形状代码本轮仅静态审阅。第一册第 34–38 讲、同文件连续复读与简介终审、整册门禁仍待处理。

## chapter_scan 与精修记录：第 34 讲

- 日期与范围：2026-10-29；第一册 `chapters/08-多模态生成与交互.md` 第 34 讲“Diffusion Model 与图像生成”，扫描前向/反向过程、参数化、U-Net/DiT、latent diffusion、CFG、少步 sampler、训练—推理噪声边界、与自回归生成的比较、评估和失败模式；不改变讲次结构或跨册边界。
- `chapter_scan`：核对逐步与闭式加噪公式、epsilon prediction 到反向均值的假设、训练目标与参数化的非唯一性、DiT scaling-law 的实测/外推边界、latent 标量数与端到端成本的区别、Ho–Salimans CFG 公式及软件 `guidance_scale` 映射、DDIM/DPM-Solver 与需训练学生模型的蒸馏路线、terminal SNR 结论的适用范围。检查现有数值示例能否复算及 code status 是否可追溯。论文版本、阅读位置与来源边界见 `research/diffusion-generation-guidance-and-sampling-2026.md`。
- `candidates_and_disposition`：保留本讲现有图像扩散主线，不新增讲次；把 2026 DiT scaling 结果作为前沿提示，明确 `1.5×10^21` FLOPs 点是摘要中直接实验上限的 250 倍外推；把训练噪声预测说明为常见而非唯一参数化；Latent Diffusion 的空间/标量压缩只作张量规模示例；把 CFG 的论文参数与差分系数分开定义；DDIM 与 DPM-Solver 保留为无需新训练的采样路线，distillation 留在面试题对照中。
- `modification_decisions`：弱化“扩散训练天然稳定”的开场断言；补前向一步 Gaussian 转移公式和 `β=[0.1,0.2]` 的两步加噪手算；补固定方差 DDPM epsilon 参数化下的反向均值及 MSE、`x̂_0`、`μ_θ` 数值链；收紧 DiT scaling 摘要的范围与外推倍率；新增 `512×512×3→64×64×4` 的 latent 元素数计算，明确不等于 FLOPs/延迟比例；修正 CFG 论文与差分系数的数值约定，引用 Diffusers v0.35.1 固定版本说明 `guidance_scale=1` 的 no-CFG 基线，并提醒核对具体接口；限定 DDIM/DPM-Solver 的 sampler 结论与 Lin et al. terminal SNR 分析条件。连续复读时为 34.9 示例补充 Python 3.8+ 与 PyTorch 2.x 的依赖范围和 `torch.isfinite` 官方 API 入口；未改可执行代码。
- `length_status`：解释性中文 2,823 字；剔除标题、代码/公式围栏、表格、行内代码/公式及链接目标后计汉字；核心单元超过 1,800 字下限。
- `evidence`：DDPM、CFG、DiT、Latent Diffusion、DDIM、DPM-Solver 和 terminal-SNR 结论均按原论文版本与阅读范围限定；ICLR 2026 scaling-law 仅阅读官方摘要，数字按摘要报告且预测与直接实验分开。Diffusers 接口只引用 v0.35.1 的经典 Stable Diffusion pipeline，不外推到所有 pipeline。新来源事实与未独立验证项见 research note。
- `example_status`：`ᾱ_2=0.72`，`x_2≈(0.8485,0.5292)`；预测噪声 `(0.2,0.6)` 的平均 MSE 为 `0.10`，得 `x̂_0≈(0.8753,0.2494)`、固定方差反向均值 `μ_θ≈(0.8642,0.3381)`；CFG 论文 `w=2` 与差分系数 `s=3` 均得 `1.1`；latent 标量数量 `786,432/16,384=48`。这些均为明确输入的教学手算，不是模型或服务实测。
- `code_status`：PyTorch noising/CFG demo 可执行代码块与 HEAD 相同；连续复读新增的 Python 3.8+、PyTorch 2.x 声明是建议环境范围，尚未在本地验证版本兼容。本讲状态为“静态审阅，未运行（2026-10-29）”，本轮未运行代码。
- `formula_status`：核对前向一步转移、闭式加噪、epsilon 重建式、固定方差 DDPM reverse mean、CFG 两种代数记法及 latent 计数；公式变量含义与适用假设已写明，数值由本地计算复核。
- `link_status`：连续复读后 L34 有 14 个实际 Markdown 链接，均为外部来源，没有本地相对链接；其中新增 `torch.isfinite` 官方 API 入口，原有 Diffusers v0.35.1 固定 tag 入口保留。`PAPERS.md` 已包含所需论文和 ICLR 2026 scaling-law 入口，不重复登记。
- `visual_status`：保留双链流程、latent pipeline 与 AR/扩散序列示意；新增的反向均值和 CFG 两种系数公式与文字算例相互对照，无需另加图。
- `sync_status`：`EXERCISES.md` 更新第 25 题，覆盖前向/反向手算、CFG 参数约定和 latent 比例；`INTERVIEW_BANK.md` 更新第 25–29、31 题的公式、成本和版本边界。术语表已有 Diffusion、DDIM、DPM-Solver、CFG 和 Latent Diffusion 入口；简介摘要已概括图像、语音和视频生成，无须重复改写；目录和讲次不变。
- `static_validation`：L34 解释性中文 2,823 字；Markdown backtick fence 10 个标记（5 组）配对，tilde/math fence 18 个标记（9 组）配对；行内 `$` 为 0；14 个外部链接均可识别；PyTorch demo 及其他 backtick fenced 示例与 HEAD 相同；手算复核；`git diff --check` 通过。未运行测试、模型、benchmark 或 PDF。
- `open_risks`：ICLR 2026 仅读摘要，未独立检查其拟合方法和预测误差；未运行扩散 checkpoint、sampler benchmark 或图像评估；Diffusers 文档结论仅限 v0.35.1 指定 pipeline；代码本轮仅静态审阅。第一册第 35–38 讲、多模态正文连续复读、简介提要终审和整册门禁仍待处理。

## chapter_scan 与精修记录：第 35 讲

- 日期与范围：2026-10-29；第一册 `chapters/08-多模态生成与交互.md` 第 35 讲“语音模型：Whisper、TTS 与实时对话”，覆盖波形与声学表示、codec/ASR、WER/CER、流式延迟、TTS、语音原生系统、打断/重叠和安全；未改变讲次顺序、章节边界或书册契约。
- `chapter_scan`：核对采样率、窗长、帧移和 padding 是否足以复算 log-Mel shape；CTC/RNN-T 的对齐目标与流式能力是否混同；Whisper 的训练数据数字能否相加、长音频策略是否被省略；WER/CER 是否明确规范化、分词/字符单位、宏平均与空分母；延迟账本是否适用于流式重叠；TTS 质量是否被压成单一自然度；打断时旧 chunk 是否会复活；代码名称和状态能否与实际行为、验证证据对应。
- `candidates_and_disposition`：维持语音基础、识别、生成和交互主线，不新增讲次。CTC 与 RNN-T 保留为不同序列目标，但说明 encoder 因果性/分块、lookahead、提交与 endpoint 才决定是否流式；Whisper 训练规模只按论文分类表述，不把未证明互斥的描述相加；完整评测和架构专题链接到题库、第二十一册与原论文。
- `modification_decisions`：新增 16 kHz、25 ms、10 ms 前端的手算，明确 center=False/无 padding 下 98 帧及 `[98,80]`；引用原始 CTC/RNN-T 论文，移除 CTC “适合流式”无条件表述；细化 Whisper 语料分类和 30 秒长音频滑窗边界；补 WER/CER 规范化、分词、聚合、插入与空参考规则；将串行延迟和式标为关键路径近似，区分 TTFC/客户端 TTFA 与 p50/p95/p99；扩展 TTS 评估切片、Moshi 证据边界及 `turn_id` 旧 chunk 丢弃策略；将 `audio_frames` 改为 `complete_hop_intervals` 并明确只数完整 hop 区间。连续复读时注明两个代码示例按 Python 3.6+ 标准库编写；未改可执行代码。
- `length_status`：解释性中文 2,892 字，排除标题、代码、公式、行内代码和链接目标后统计；核心讲解单元超过 1,800 字下限。
- `evidence`：Whisper、CTC、RNN-T、SoundStream、EnCodec、Moshi、VALL-E 2 的版本、阅读位置与证据边界见 `research/speech-recognition-generation-and-realtime-2026.md`。论文结论分别限于论文模型/语料/协议；研究笔记未记录章节审阅与改动过程。
- `example_status`：16 kHz 下 25 ms 窗为 400 点、10 ms hop 为 160 点；1 秒完整窗口数 `floor((16000-400)/160)+1=98`，对应 `[98,80]`。命令词例 CER `2/6≈0.333`、固定三词切分 WER `1/3≈0.333`；延迟例 `240+110+280+130+40=800 ms`，加 2 秒采集得 `2,800 ms`；hop demo 的 1/5/30 秒分别为 50/250/1500 个 320 点 hop。以上均是明确约定的教学计算，不是模型或服务实测。
- `code_status`：hop helper 的可执行语句和输出命名已调整；没有可追溯的旧运行记录，本轮按静态审阅标记，未执行。CER helper 代码未改，但找不到相应历史验证记录，也将旧“已验证”状态降为“静态审阅，未运行（2026-10-29）”。
- `formula_status`：核对 STFT 完整窗口计数、WER/CER 汇总分母、RTF 与串行延迟账本；定义均标出约定，手算已复核。流式延迟的阶段重叠不作机械求和。
- `link_status`：L35 共 8 个链接（7 个外部、1 个第二十一册本地链接）；CTC、RNN-T 和 Moshi 使用原始论文，Whisper 直链至 HTML 论文，Cross-book 目标存在。`PAPERS.md` 已有语音论文入口，不重复登记。
- `visual_status`：保留 ASR/TTS 数据流和打断状态图；补充的窗长、帧移和形状用可复算数例说明，不另加装饰图。
- `sync_status`：更新 `EXERCISES.md` 第 28 题的 padding/规范化/聚合、延迟端点和打断指标；更新 `INTERVIEW_BANK.md` 第 40–45 题；补充 `GLOSSARY_EN_ZH.md` 的 CTC、RNN-T、WER、CER 边界。`PAPERS.md` 已包含论文入口；`简介.md` 已概括语音生成、交互与评估，本讲摘要仍适用；目录未变。
- `static_validation`：L35 解释性汉字 2,892；backtick fence 12 个标记（6 组）配对，tilde/math fence 10 个标记（5 组）配对，行内 `$` 为 0；8 个 Markdown 链接中 7 个外链、1 个本地目标存在；三个手算及代码输入输出关系静态复核；`git diff --check` 通过。未执行测试、代码、模型、benchmark 或 PDF。
- `open_risks`：未运行 ASR/TTS 模型、听者评测或流式 benchmark；TTFA、barge-in 停止时间和尾延迟仍须按具体系统测量；CTC/RNN-T 不作跨配置准确率/速度排名；两个教学代码块均未运行。第一册第 36–38 讲、多模态正文连续复读、简介终审和整册门禁，以及其余书册仍待处理。

## chapter_scan 与精修记录：第 36 讲

- 日期与范围：2026-10-29；第一册章节 08《多模态生成与交互》第 36 讲，覆盖视频的时序约束、tubelet/token 预算、视频扩散架构、长视频控制、world model 判据、Sora 资料与生命周期、评测、安全及 token 估算 demo；不改讲次结构或跨册边界。
- chapter_scan：核对 raw patch 数与压缩后 latent/token 数是否混淆；图像扩散扩展、时空分解、时空级联、全片时空网络和 latent Transformer 的机制与瓶颈；复杂度例子是否逐步可算；world model 是否以动作条件与状态转移验证，而非演示观感；Sora 的历史研究披露与当前产品生命周期是否分开；WBench 的维度、案例、自动子指标及人工校验是否严格限于论文协议；视频来源标识是否被误说成真实性保证。来源版本、阅读位置和证据边界见 research/video-generation-world-modeling-and-sora-lifecycle-2026.md。
- candidates_and_disposition：不增加讲次；补足 2022–2024 的代表架构链，并在 36.3 加入四路线对照表。Video Diffusion Models/Make-A-Video 留在图像先验迁移与时空模块路线；Imagen Video 代表空间/时间超分辨率级联；Lumiere 作为全片时空建模的对照；CogVideoX 代表 3D VAE 加 Diffusion Transformer。WBench 保留为专门的多轮交互评测；World-in-Context 作为具体 v1 预印本的显式控制例子，不外推为行业共通机制。
- modification_decisions：补出时空分解 U-Net 中的空间卷积/时间注意力、图文配对先验与无配对视频运动学习、Imagen Video 七子模型及作者报告输出配置、Lumiere 的全时长 Space-Time U-Net 与剩余显存/多镜头边界，以及 CogVideoX 的时空 VAE/Transformer 路线。加入 dense attention 的 N×N 每头配对账本，明确 raw patch 计数不是实际 latent token 数。修正原有 Qwen3.5 研究摘记失效链接，改为现存多模态证据笔记。连续复读时注明视频 token demo 按 Python 3.6+ 标准库编写；没有把跨论文配置和评测当作可直接排名的数据，也未改可执行代码。
- length_status：解释性汉字 2,951 字，剔除标题、代码/公式围栏、表格、状态行、行内代码和链接目标后计数；核心单元通过 1,800 字下限。篇幅达标不替代对机制、证据和取舍的人工审阅。
- evidence：方法谱系的来源、版本及阅读范围见研究笔记；Sora 2024 研究说明只支持其公开路线并明确未披露模型/实现细节，Help Center 仅支持 2026-10-29 时的服务终止日期；WBench 的 289 案例、1,058 轮、四类交互、五维与 22 个自动子指标为作者报告，限其数据和评分协议；World-in-Context 是 2026-10-23 的 arXiv v1，控制 token 与结果绑定其模型和 WBench 协议。没有从模型演示推断生产级物理理解或未公开架构。
- example_status：tubelet 例 16/2×24×24=4,608；64/2×24×24=18,432；64/2×48×48=73,728，只代表未压缩 patch 计数。稠密注意力例中 4,096²=16,777,216、8,192²=67,108,864，序列翻倍使每头 query-key 配对数增至四倍；不含 head 总数、窗口/稀疏实现和 kernel 的显存/延迟。
- code_status：纯 Python video_tokens 函数及输入、输出语句与 HEAD 相同；连续复读时新增 Python 3.6+ 标准库环境说明，此版本范围未在本地执行验证。此前无可靠可追溯运行记录，正文状态保持“静态审阅，未运行（2026-10-29）”；本轮没有执行代码。
- formula_status：核对 tubelet 的向上取整、token 计数与密集 attention 的 N² 形状关系；变量和计数边界写明，三组 patch 数与两组平方结果已手工复算。
- link_status：L36 共有 11 个外部链接，无本地链接；整文件扫描得到 65 个实际 Markdown 链接，其中 3 个本地目标均存在。排除一处公式中方括号被简单扫描误判为 Markdown 链接。新增书目 29–32 对应 Make-A-Video、Imagen Video、Lumiere、CogVideoX；第 25 条 Qwen3.5 研究笔记现指向仓库中存在的 research/multimodal-systems-and-evaluation-2026.md。
- visual_status：在 36.3 增加四路线比较表，按代表方案、机制和边界并列；36.7 保留视频生成评估矩阵。表格只作比较索引，详细机制与例子仍由正文说明。
- sync_status：INTERVIEW_BANK.md 第 37 题更新为路线与成本对照；EXERCISES.md 第 27 题加入四类架构的显存、序列成本和一致性比较；GLOSSARY_EN_ZH.md 增加 Temporal Super-Resolution、Space-Time U-Net；PAPERS.md 视频生成索引增加 Lumiere。简介摘要已经覆盖视频生成与时序世界建模，目录与结构未变。
- static_validation：L36 解释性汉字 2,951；全文件 backtick 围栏 26 个标记（13 组）和 tilde/math 围栏 38 个标记（19 组）成对；行内美元符号数为 0；L36 的 11 个链接均为外链；全文件 3 个本地链接目标存在。patch 与注意力数例手工复算，git diff --check 通过。
- open_risks：没有运行视频生成代码、模型、视觉人工评测、WBench 或性能 benchmark；论文数字仍只支持其模型配置。未实测不同 VAE 压缩对细节/运动的影响，也未验证级联、全片生成或 DiT 在同硬件同数据下的成本排序。第一册第 37–38 讲、多模态正文连续复读与简介终审、第一册整册门禁，以及其余书册仍待完成。

## chapter_scan 与精修记录：第 37 讲

- 日期与范围：2026-10-29；第一册章节 09《综合复习与技术表达》第 37 讲“开放技术问题的系统拆解”。扫描任务契约、根因假设、实验排序、数学能力回归、污染扰动、claim 证据支持、模型信息分级、多目标取舍及实验记录；未改变讲次结构或书册边界。
- `chapter_scan`：检查信息增益公式中的假设/熵符号是否冲突；评测条件是否覆盖数据快照、模型 revision、预处理、runner、工具与解析器；二项标准误是否被误用于新旧版本差值；重复运行是否与独立题目混淆；扰动题是否保持语义和难度；污染信号是否被误写成训练来源证明；claim 缺少证据是否被等同为错误；A—D 证据等级是否被误说成行业标准；Pareto 定义是否给出可复算对照。证据版本、阅读位置与边界见 `research/evaluation-design-uncertainty-and-contamination-2026.md`。
- `candidates_and_disposition`：不增加讲次或转移主题。用主动学习文献支撑 EIG 的不确定性减少概念，同时将 `G×R/C` 明确留作未校准教学启发式；用 LLM evaluation harness 的协议敏感性作为实验复现依据；跨域引用深度 RL 不确定性论文时显式限制范围；把污染扰动题、词级 DCQ 与 ICLR 2026 检测脆弱性作为互补但不等价的风险证据。
- `modification_decisions`：将 `H(P(H))` 改为区分 `\mathcal H` 与熵函数 `\mathrm H` 的后验 EIG 表达，并手算二假设实验的先验/后验熵；补 `S_priority` 代入算例并说明它不是 VOI/EIG 估计；固定题目集、模型、prompt、预处理和评测程序版本，分开题目抽样、运行和生成随机性，增加 Bernoulli 标准误数例和 paired/cluster uncertainty 说明；把工具验证结果限定为系统端改善；要求等价改写先经题意/难度核验，污染性能差列为信号而非证明，且未检出不能证明无污染；把 SupportRate 改为带 rubric 的三分类人工/校准评估，另报矛盾率和证据不足率；声明证据 A—D 是本书工作分级；为 Pareto 前沿加配置表和支配关系推算。将本讲代码旧“已验证”标签改为静态审阅状态；可执行代码未改。
- `length_status`：解释性中文 4,472 字；计数排除标题、代码与公式围栏、表格、行内代码/公式、引用目标和代码状态行；核心单元超过 1,800 字门槛。
- `evidence`：Kirsch & Gal arXiv v2 支持信息论 EIG 与主动学习目标的关系，不支持把本讲工程评分当作论文方法；Biderman et al. arXiv v3 支持 LLM 评测设置/实现细节影响复现比较；Agarwal et al. arXiv v4 的不确定性结果限深度 RL 基准；DCQ 的原题/三种词级扰动协议限其污染信号；ICLR 2026 的检测掩蔽结论限作者报告的推理模型和训练条件，且本次仅核对官方摘要。来源细目与未解决边界见研究笔记。
- `example_status`：EIG 构造例先验熵 `0.6931 nat`、后验熵 `0.5004 nat`、信息增益 `0.1927 nat`；优先级分数 `0.90×0.75/2=0.3375≈0.338`；独立二项示例 `n=100, p̂=0.61` 得 `SE≈0.0488`（4.9 pp），不作为前表真实区间；加权 claim 例 `w=(3,2,1)` 得支持率 50%、矛盾率 16.7%、证据不足 33.3%；Pareto 教学配置 C 支配 A，B/C/D 非支配。所有数字均为明示构造值。
- `code_status`：实验优先级 Python 代码可执行语句、输入输出和依赖与 HEAD 一致；连续复读时注明该示例按 Python 3.6+ 标准库编写。`progress/` 未找到能对应旧“已验证”状态的运行命令、输入和结果，因此正文状态改为“静态审阅，未运行（2026-10-29）”；本轮没有运行该代码或任何测试。
- `formula_status`：核对二假设 EIG 后验概率和熵代入、`S_priority` 分数、二项标准误、加权支持/矛盾率及 Pareto 支配定义；手算与归一化分母一致。新旧系统差值仍要求按同题配对估计不确定性，不能由单版本 Bernoulli SE 直接推出。
- `link_status`：修复章节末尾原有失效学习科学研究笔记链接，改为引用下列 DOI 条目；讲次增加 13 个外部来源链接和 1 个本地研究笔记入口，均已做目标检查。研究笔记链接相对路径为 `../../research/evaluation-design-uncertainty-and-contamination-2026.md`。
- `visual_status`：37.9 新增四配置 Pareto 对照表，表格标出方向、单位与教学假设；不新增 Mermaid 图，现有因果链和系统分层已表达主要结构。
- `sync_status`：`EXERCISES.md` 阶段 5 新增 4B，覆盖任务契约、控制变量、配对不确定性、改写等价检查及加权 claim 标注；`INTERVIEW_BANK.md` 更新“如何评估一个 LLM”回答框架；`GLOSSARY_EN_ZH.md` 增加 EIG 并与 VOI/工程优先级评分区分；`PAPERS.md` 统计评估小节补入深度 RL 不确定性来源及其适用边界。讲次、目录和简介无结构变化。
- `static_validation`：第 37 讲解释性中文 4,472 字；全章节代码/数学围栏配对且无未闭合围栏；代码块正文与 HEAD 相同；EIG、SE、SupportRate 和 Pareto 例子手算；修复后本文件本地链接可达；`git diff --check` 通过。未运行代码、自动化测试、模型、benchmark、PDF 构建或更新。
- `open_risks`：`G×R/C` 是未校准的教学分数，真实先验/效用和实验成本需按项目定义；改写语义与难度须在具体任务中验证；当前污染检测不能证明训练来源或排除所有污染；深度 RL 统计结果不能直接推广到 LLM；ICLR 2026 此处只根据官方摘要引用；未对目标模型或服务运行实验。下一步处理第 38 讲“学习迁移与技术表达”，随后完成第七、八正文连续复读、第一册简介摘要终审和整册验收。

## chapter_scan 与精修记录：第 38 讲

- 日期与范围：2026-10-29；第一册章节 09《综合复习与技术表达》第 38 讲“学习迁移与技术表达”。审视主动检索/间隔练习证据、学习产物评分、依赖复习、12 周时间预算、项目指标、模拟交流、迁移差距、终局评估及复习调度代码；不调整章节边界和讲次关系。
- `chapter_scan`：核对两篇学习科学研究是否被外推成大模型技术学习的直接证据；检查指数衰减曲线和评分公式是否作为实测规律；为 `R(t)`、mastery score、周时间分配和 session score 补可复算输入和中间量；检查成功/失败检索是否被误解释为真实记忆参数变化；核对 `TransferGap` 的尺度和难度条件；查找代码验证的旧记录及与当前代码状态的对应。
- `candidates_and_disposition`：保留检索—推导—实现—反例的四产物主线、12 周学习路线和项目证据链，不新增方法或知识讲次。Roediger/Karpicke 的实验与 Cepeda 的 verbal-recall meta-analysis 仅支持各自材料、测验和保持间隔中的发现；技术学习计划作为待通过延迟回忆和陌生变体继续验证的设计建议。
- `modification_decisions`：在学习科学结论附近增加 DOI 引用，并明确研究没有直接测量 LLM 工程学习；将一次检索成功/失败改为间隔调度信号，而非真实 `τ` 变化或唯一根因；补指数曲线、四类掌握度评分、12 小时分配、模拟交流分数和调度状态更新的逐步代入；声明示例分数需要固定 rubric 且未经验证；说明 Python 3.7+ 标准库的代码输入范围，把旧“已验证”状态改为“静态审阅，未运行”。连续复读时将 38.11 标题和说明收窄为状态更新核心，说明熟题答对但变体失败只小幅更新强度、事件由调用者给定且不会自动建立队列；可执行代码未修改。
- `length_status`：解释性中文 3,417 字；按独立讲次统计，剔除标题、代码/公式围栏、表格、行内代码/公式、引用目标与状态行；核心单元通过 1,800 字下限。
- `evidence`：Roediger & Karpicke 2006 支持 prose-passage 实验中短延迟重复学习与较长延迟检索测试的具体差异；APS publisher 摘要通过索引读取，官方页直接打开返回 403，未读全文。Cepeda et al. 2006 PubMed 摘要支持 verbal recall 的 ISI 与 retention interval 共同影响；UC eScholarship 全文访问要求 JS/验证，本轮未读全文。证据范围和访问边界见 `research/retrieval-practice-and-spacing-2026.md`。
- `example_status`：指数模型教学例 `R_0=0.8, τ=7 天, t=7 天` 得 `0.8e^{-1}≈0.294`，不代表研究拟合；mastery 权重例合计 `0.555`；每周 6 小时、主题分配两周得 `H_k=12` 小时，五项预算 `3/3/2.4/2.4/1.2` 小时；mock session 例逐项相乘后为 `0.61`；熟悉题 80、难度匹配变体 55 得 `TransferGap=25`；调度示例中 attention 强度 `0.4→0.52→0.64`、间隔 `1→1.8→3.24` 天、下一日期 `1.8→5.24` 天，显示值四舍五入为 `5.2`。数值为教学输入。
- `code_status`：第 38.11 复习调度 Python 代码的可执行语句、输入输出和标准库依赖与 HEAD 相同。`progress/` 未找到能对应旧“已验证（2026-10-07）”的运行命令、输入与输出记录，因此正文更新为“静态审阅，未运行（2026-10-29）”；依据状态转移逐步手算预期输出，本轮未运行代码。
- `formula_status`：核对指数衰减代入、`S_mastery` 四项加权和、`H_k` 五类时间预算之和、`S_session=0.61` 及 `TransferGap=80-55=25`；公式中的单位和评分边界已说明。
- `link_status`：全章节 17 个 Markdown 链接，其中 15 个外部链接、2 个本地研究记录目标；两处本地路径均存在。此前指向不存在学习科学研究记录的链接已改为新建来源笔记。
- `visual_status`：保留第一册线性依赖链、12 周计划表、项目对照表、复盘错误分类表和综合证据表；各视觉材料均在正文解释其读法及边界，无需新增图。
- `sync_status`：`GLOSSARY_EN_ZH.md` 增加 Retrieval Practice、Distributed Practice / Spaced Practice、Transfer Gap；`EXERCISES.md` 阶段 5 新增第 50 题，要求构造检索、变体、实现与反例并记录延迟表现。`PAPERS.md` 已有两项学习科学文献入口，不重复添加；简介摘要已更新，目录不变。
- `static_validation`：L38 解释性中文 3,417 字；全章 19 组代码/数学围栏配对，行内 `$` 标记 102 个（51 对），无未闭合围栏；代码正文与 HEAD 相同；17 个链接的 2 个本地目标均存在；算例手算；`git diff --check` 通过。没有运行代码、自动化测试、模型、benchmark 或 PDF。
- `open_risks`：两篇研究均非大模型技术学习的直接实验，且本轮没有读取其全文；指数衰减、学习分数和时间权重是教学假设；真实学习迁移仍需在目标材料、间隔、反馈和评分 rubric 下测量；未运行调度代码。章节 08、09 连续复读、简介摘要与目录导航复核现已完成；GitHub Mermaid 页面渲染确认仍待完成，第一册门禁未通过。


## 整册复读与门禁状态：2026-10-29

- `scope`：依据第一册主路径复核简介、目录及九个正文文件的前后交接；在此前逐讲审阅和正文复读记录基础上，完成章节 08《多模态生成与交互》与章节 09《综合复习与技术表达》的连续复读。五部分定位、九章摘要、38 讲编号和专题入口一致，无需调整目录或章节边界。
- `continuity_review`：章节 08 从多模态理解与表示承接到图像、语音和视频生成，再以任务评估与系统边界收束；章节 09 接收前四部分，沿任务契约、区分假设、实验、证据和学习迁移形成结尾。跨讲复读补足 34.9 的 Python 3.8+/PyTorch 2.x 依赖与官方 API 链接，注明 35.10 的 Python 3.6+ 标准库、36.10 的 Python 3.6+ 标准库及 37.11 的 Python 3.6+ 标准库；将 38.11 说明收窄为状态更新核心，补出熟题与变体不同结果下的 strength/interval 更新和 attention 手算轨迹。可执行代码未改，相关代码仍为静态审阅状态。
- `whole_book_static_validation`：简介、目录与 9 个章节文件共检查 452 个 Markdown 链接，失效本地目标为 0；Markdown 围栏均配对，章节 08 行内美元符号为 0 个、章节 09 为 102 个（51 对）；38 个逐讲进度记录及篇幅计数齐全，均不低于 1,800 个解释性汉字；`git diff --check` 通过。未运行代码、自动化测试、模型、benchmark 或 PDF。
- `open_risks`：GitHub 是本项目读者入口之一。工作区编辑尚未发布，当前无法把公开 GitHub 页面的显示结果当作本地编辑版验收；Mermaid 源码已静态检查，但本地未安装 mmdc 或浏览器渲染器，web 工具直接读取章节页返回 cache miss，目标页面渲染仍未确认，因此第一册整册门禁未通过。
- `next_action`：寻找可对本地编辑版 Mermaid 源码进行目标兼容渲染的验证途径；确认通过后再关闭第一册门禁并启动 book-02。
