# 第二十一册《Transformer 架构详解、升级变种与未来架构演进》架构审视

审视日期：2026-11-20；状态更新：2026-11-24。册级架构审视完成；截至本次更新，已完成第 1—72 章正文首轮审阅与修订。第 12 章两处、第 14 章一处、第 18 章一处、第 20 章两处、第 21 章一处及第 24—27、29—33、35、37—44、52 章公式和第 53—55、61—72、88 章公式的 GitHub MathJax 实际渲染尚待验收；第 32、33、43 章 Mermaid 的 GitHub 页面渲染也待验。第 56—60 章无公式、代码或 Mermaid 图；第 73—95 章尚未逐章审阅，共 23 章，不能据目录提前判定章节内容正确或可合并。全套书精修 goal 继续 active；不更新 PDF 或 `plan.md`。

## 书册契约

- **定位与类型：** Transformer、替代序列架构和现代大模型架构的专题教材；同时包含历史演进和带版本边界的模型案例。核心内容是结构、信息流、训练/推理代价、证据边界与适用条件，不把排行榜、API 产品能力或模型服务系统误称为内部架构。
- **读者起点：** 已理解基础 Transformer 和 next-token prediction；能够阅读常见线性代数记号、模型配置字段及论文图表。关键符号应在本书首次使用处解释，不假设有训练 frontier 模型或部署大规模 serving 的经验。
- **读者结果：** 读者能复述 Transformer 的信息路由和并行优势；逐步估算 dense attention、KV cache、MoE 等结构成本；比较 Linear Attention、SSM/RNN、卷积、检索/记忆与混合结构的计算、表达、数据访问和硬件约束；分析模型论文/模型卡中的架构披露及未披露项；把结构创新与训练、serving、Agent harness 和 benchmark 结果分开归因。
- **中心问题：** 什么信息流和状态表示值得由模型内部计算，架构改变了哪些计算/内存/通信代价，所宣称的能力证据能否支持该结构结论？
- **书系边界：** 第一册负责从零实现和解释标准 Transformer；第二册给出有限篇幅的架构总览；第五册把架构选择转成训练配置、预算和稳定性回归；第六册和第二十四册负责服务与推理引擎细节；第十五册主讲多模态模型；第十六册主讲 reasoning 方法；第十七册和第二十册主讲 Agent 行为与运行时。本册保留架构与 serving/多模态/Agent 的交界案例，但不重复主讲 API、协议、产品 runtime、量化内核或统计评估。

## 目录架构地图

| 范围 | 主题职责 | 复核重点 |
| --- | --- | --- |
| 1—8 | Attention、QKV/head、FFN/norm、位置与 mask 的结构复习 | 从直觉到符号、shape/复杂度逐步推进，不重复第一册完整推导 |
| 9—15 | Transformer 的并行/表达优势与注意力、缓存、检索、流式瓶颈 | 明确任务、输入长度和硬件条件；区分模型结构、kernel、retrieval 和评测 |
| 16—23 | Decoder-only、现代 block、长上下文、经典稀疏 pattern、kernel、MoE/Dense trade-off 与小模型/端侧效率 | 第 19 章负责窗口/dilation/global/random pattern；第 21—22 章负责 MoE 路由与成本；第 23 章比较小模型设计和设备约束，避免把 serving 速度直接当成结构优越证据 |
| 24—28 | Linear Attention、核近似与稀疏/block-sparse attention 深入比较 | 第 26 章负责块布局、输入依赖路由和执行边界；与第 19 章的经典 pattern 介绍明确分工 |
| 29—36 | SSM/RNN/CNN 信息流、S4/Mamba/Mamba-2 与替代证据 | 统一状态更新与序列复杂度口径；按数据访问/精确检索、硬件扫描和生态解释限制 |
| 37—42 | RWKV、RetNet、Hyena、早期混合路线与跨架构比较 | 使用相同任务与成本维度比较，不以渐近复杂度代替实际吞吐 |
| 43—50 | Attention/SSM/卷积/retrieval/memory/MoE/Agentic 的组合与开放题 | 第 49 章区分模型参数/提示能力、外部动作闭环和 Agent 评测归因；Agent 方法、Runtime 和协议分别回链第十七、二十、二十二册；第 50 章综合层间、head 级、记忆分支和分数级组合及递归模型独立演进的证据，限制“未来以混合架构为主”的趋势判断 |
| 51—60 | GPT、训练规模、后训练、LLaMA/Mistral/Qwen/DeepSeek 谱系与综合判断 | 是按历史模型组织的教学章节；独立时间线文件应定位为索引/速查，不再整段复述章节正文 |
| 61—78 | KDA/Gated DeltaNet/Gated MLA、NoPE、Hybrid Attention、cache、mHC、CSA/HCA、AttnRes 等机制 | 明确机制所属模型、版本、来源和变量；区分论文、模型卡、源码与发布方性能报告 |
| 79—95 | 近期模型架构和系统案例：Mistral、Step、DeepSeek、Kimi、Qwen、GLM、GPT-OSS、Gemini 等 | 仅把可确认的结构创新作为本册主线；协议、Agent runtime、benchmark、硬件和 API 事实按需压缩并回链对应主讲书 |

## 首轮架构发现与处理决定

1. 目录现有 95 个连续编号章节。保留全部文件路径，先按现有顺序逐章审阅；只有正文诊断确认重复或读者结果缺失后，才考虑移动、合并或删减。
2. 目录原把第 61—95 章统一放在“Frontier Model 架构独立专题”，机制论文与近期模型专题跨度过大。现已拆为“前沿架构机制”（61—78）和“前沿模型架构专题”（79—95），保留全部章节路径和编号；具体章节边界仍须逐章核验。
3. 第 51—60 章和 `主流大模型架构创新时间线.md` 有潜在重复。前者负责按模型谱系解释机制和训练历史，后者负责短时间线、canonical 模型目录、每项关键证据和章节入口；后续精修需要按此职责审查时间线中的长篇重复讲解。
4. 独立时间线中的后续模型发现流程原指向不存在的 `model-inventory.md`。现已改为链接项目计划第 2.3 节的榜单候选发现原则，并将真实 source notes、`PAPERS.md`、`BOOK_SERIES.md`、时间线和有读者结果的正式章节接入同步流程；未创建空白模型目录。
5. 第 79—95 章标题含 API、工具、Agent swarm、安全与 benchmark 等系统材料；逐章时先提炼其可验证架构问题，无法形成架构读者结果的部分应缩减并回链第 6、8、15—17、20、24 册，不以“前沿模型”名义扩展出第二套产品综述。
6. `PAPERS.md`、`BOOK_SERIES.md` 的相关模型条目仍有跨书缺失的 source-note 链接；此前扫描已登记于[共享索引本地链接复核](../shared-index-local-links-audit-2026-11-19.md)。第二十一册处理时按对应章节来源逐项修复，不把缺失路径当作证据。

## 待逐章核验的高风险主题

- 2025—2026 闭源模型的结构通常未披露；从 API、榜单、产品 benchmark 或运行时行为推断 Dense/MoE、注意力或训练配方属于越界结论。
- 2026 开放/半开放模型专题需要固定模型 ID、模型卡/配置/权重 revision、官方报告日期，并把发布方自报、源码可见和独立复现分开。
- Linear Attention 与 SSM 的线性复杂度声明要交代状态大小、特征映射/扫描、因果性、精确性、训练和 decode 适用条件；不能把算法复杂度写成普遍的 wall-clock 性能优势。
- GQA/MLA/KV、稀疏 attention、CSA/HCA、latent/recursive cache 之间需要统一 cache 单位和比较对象；kernel/量化/provider 的性能归第二十四册及相关实测条件。
- 多模态编码器、语音流式、Agent swarm 和 safety/runtime 与其他书重叠明显；保留的是结构和信息流解释，系统合同与安全评测按证据回链。
- 现有全书审计尚未处理 95 章的链接、公式、围栏、source-note、配套题目和代码声明；由逐章精修建立证据后再做整册终验。

## 资料和基线

- 本轮完整读取《简介.md》《目录.md》和 `主流大模型架构创新时间线.md`。时间线 793 行中含有现行模型登记、历史机制教学及 2026 案例；正文核验日期和各模型来源需在逐章阶段重新对照。简介新增了从机制、替代架构、历史谱系到近期案例的阅读路线，目录将 61—78 与 79—95 分组，并把原“优先扩写”清单改为核心主线速读。
- `chapters/` 当前有 95 个 Markdown 文件，编号 1—95 无缺号。册内 98 份 Markdown 的首轮本地链接扫描发现 139 个链接实例、28 个失效实例及 20 个目标文件，集中在 12 份时间线/模型专题资料；详见[本地链接首轮扫描](local-links-audit.md)。静态围栏检查发现 2,092 个围栏标记均成对；将 10—12 章的 `x.0` 作为范围/资料前言后，显眼的 H2 序号异常是第 88 章若干追加小节次序倒置，留待该章全文审阅时修复。此扫描不代替数学、图表渲染或内容验收。
- 已发现的非引用路径 `model-inventory.md` 在书册目录中不存在；已删除这一错误流程依赖。架构初审阶段未读 95 章正文；其后逐章精修进度见当前入口。未运行代码/测试/模型/API、未构建或更新 PDF。

## 截至第 27 章完成时的入口（历史记录）

简介阅读路径、目录分组和时间线维护入口已完成结构同步。第 1—27 章的正文审阅与修订见对应 `chapter-XX-audit.md`。第 13 章审计记录长距离依赖、位置敏感行为和 LURCH/CRE 边界；第 14 章记录归纳偏置、顺序/局部性、逐位置历史与递归状态的职责及 MQAR 证据范围；第 15 章记录流式输入/输出、自回归延迟、KV 历史与 runtime 调度边界；第 16 章记录架构/目标/任务格式、通用续写接口及架构比较边界；第 17 章记录 GPT、LLaMA、Mistral 的公开 block 配置与证据边界；第 18 章记录 RoPE 位置映射、频率分配、继续训练和有效上下文；第 19 章记录 causal sliding window、dilation、Longformer global、BigBird local/random/global 与 query-conditioned sparse selection。第 19 章仅承担 attention pattern 与信息路径；第 20 章负责 dense attention kernel、decode 分段并行与分页 KV 映射，审计见[第二十章精修审计](chapter-20-audit.md)。第 26 章现负责 token mask 到 block layout、输入相关路由、选择成本及执行边界；审计见[第二十六章精修审计](chapter-26-audit.md)，研究来源见[研究记录](../../research/block-sparse-attention-layout-and-routing-2026-11-20.md)。第 27 章现主讲 Longformer、BigBird、Reformer 的专属方法机制，审计见[第二十七章精修审计](chapter-27-audit.md)，研究来源见[研究记录](../../research/longformer-bigbird-reformer-methods-2026-11-20.md)；第 89 章负责 GLM-5 DSA 模型案例。第 7 章负责位置表示分类和能力边界，第 8 章负责可见性、监督目标与样本边界，第 9 章负责并行路径、理论表达力与 scaling 证据边界，第 10 章负责 pair 计算、实现相关显存、在线 softmax 与不同阶段的成本口径，第 11 章负责逐层 KV 复用、显式 cache 容量与 serving 状态分配，第 12 章负责固定参数下的 ICL、示例信息、上下文证据使用及其机制边界，第 13 章负责远距离证据依赖和位置效应；第 14 章负责序列结构先验、顺序/局部性与状态记忆的概念桥接，不重复第 29—33 章的状态空间推导和硬件分析，也不重复第 42 章的横向架构比较；第 15 章负责输入流与输出流的架构边界及其对低延迟的影响，只用调度案例区分 runtime 缓解，不重复第六册的服务设计；第 16 章负责解释 decoder-only 如何用统一 token 接口承载通用生成，并把“主流”与“普遍最优”分开；第 17 章负责按版本比较公开 block 配置，不重复第 4、6、7、11、18、23 章的机制推导或第 51 章的谱系与 scaling/ICL 深讲；第 18 章负责位置映射、长上下文训练与任务级有效长度，不重推 RoPE 基础；第 23 章负责小模型参数预算、结构改造与端侧设备证据。第 24 章负责有限 kernel 特征分解、分子/分母前缀状态、固定特征宽度下的成本边界与状态可寻址性；第 25 章负责 Performer/FAVOR+ 的 softmax 随机特征、估计器误差和保证条件，不重复第 24 章的状态递推。第 25 章精修审计见[链接](chapter-25-audit.md)，研究来源见[研究记录](../../research/performer-favor-plus-random-features-2026-11-20.md)。SSM 动力学交第 29—30 章；第 21、22、23、24 章审计见对应链接。第 1—27 章已修订，28—95 章共 68 章待逐章精修；下一步进入第 28 章《高效 Attention 为什么没有完全替代标准 Attention》。每个章节完成后更新对应 `chapter-XX-audit.md`，事实与来源进入 `research/`，本书状态和结构决定仅留在本目录。


## 第三十六章完成时的入口（历史记录）

第二十一册第 1—36 章已完成首轮正文审阅与修订；第 37—95 章共 59 章待精修。第 29 章将固定参数线性时不变 SSM 的连续方程、离散递归、因果卷积和衰减/稳定性边界收束为五节；职责、删改和交接见[第二十九章精修审计](chapter-29-audit.md)，数学来源见[SSM 基础研究记录](../../research/state-space-model-foundations-2026-11-23.md)。第 30 章以四节比较 RNN/SSM 递归状态、CNN 感受野、causal Attention 可见历史，以及训练并行与生成状态；改写决定见[第三十章精修审计](chapter-30-audit.md)，研究来源见[序列信息流研究记录](../../research/sequence-architecture-information-flow-comparison-2026-11-23.md)。第 31 章现以五节解释 HiPPO 投影、NPLR/DPLR 坐标变换、resolvent/Woodbury/Cauchy 核生成及 S4 的卷积/递归双视图；审计见[第三十一章精修审计](chapter-31-audit.md)，事实来源见[S4/HiPPO 研究记录](../../research/s4-hippo-dplr-kernel-2026-11-23.md)。第 32 章现以五节解释内容选择缺口、输入依赖的 $\Delta_t,B_t,C_t$、仿射前缀 scan 和 Mamba block，并将合成任务证据限制在论文设置内；审计见[第三十二章精修审计](chapter-32-audit.md)，事实来源见[Mamba 选择性状态与扫描研究记录](../../research/mamba-selective-state-and-scan-2026-11-23.md)。第 33 章现以五节解释理论复杂度与阶段口径、HBM/SRAM 中的融合 scan、反向重计算、固定状态与 KV 容量，以及原论文三类效率基准的实验条件；审计见[第三十三章精修审计](chapter-33-audit.md)，研究来源见[Mamba 硬件感知扫描与效率证据](../../research/mamba-hardware-aware-scan-and-benchmarks-2026-11-23.md)。第 34 章现以五节区分检索信息路径、MQAR 条件、ICL 任务切片、规模/版本证据和 workload/生态核查；完整重写及取舍见[第三十四章精修审计](chapter-34-audit.md)，来源见[Mamba 检索与 ICL 研究记录](../../research/mamba-retrieval-icl-evidence-2026-11-23.md)。第 35 章现以六节讲清 SSD 的三层含义、Mamba-2 状态参数化、半可分离矩阵、与 softmax attention 的边界和 chunk state-passing 算法；论文的 $2$–$8$ 倍结果限定为特定 SSD 核心层对优化 selective scan 的条件性比较，见[第三十五章精修审计](chapter-35-audit.md)和[Mamba-2/SSD 研究记录](../../research/mamba-2-ssd-duality-2026-11-23.md)。第 36 章现以六节综合替代主张范围、约 8B 匹配模型证据、复制/ICL 任务差异、Mamba-3 与 hybrid 结果，并将纯模型和混合模型分开归因；详见[第三十六章精修审计](chapter-36-audit.md)及[跨架构比较研究记录](../../research/ssm-transformer-replacement-evidence-2026-11-23.md)。第 12、14、18、20—27、29—33、35 章公式的 GitHub MathJax 实际渲染仍待验收；第 32、33 章 Mermaid 页面渲染也待验。下一入口为第 37 章《RWKV：RNN 式状态与 Transformer 式训练的结合》；第 37—95 章共 59 章待精修。目标读者试读、外链最终逐条复核和第二十一册连续复读仍待全册门禁。

## 第三十七章完成时的入口（历史记录）

第二十一册第 1—37 章已完成首轮正文审阅与修订；第 38—95 章共 58 章待精修。第 37 章按六节重写 RWKV-4 的 token shift、time-mixing/channel-mixing、逐通道 WKV 归一化状态，以及训练 scan 与递归生成的关系；再区分 RWKV-5 Eagle、RWKV-6 Finch、RWKV-7 Goose 的矩阵状态、动态衰减和 delta 更新，并将 RWKV-8 的官方实验性 DeepEmbed/ROSA 资料限于版本边界。章审计见[第三十七章精修审计](chapter-37-audit.md)，来源见[RWKV 状态与训练形态研究记录](../../research/rwkv-state-training-evolution-2026-11-23.md)。

第 12、14、18、20—27、29—33、35、37 章公式的 GitHub MathJax 实际渲染仍待验收；第 32、33 章 Mermaid 页面渲染也待验。下一入口为第 38 章《RetNet：Retention 机制与并行/递推统一》；目标读者试读、外链最终逐条复核和第二十一册连续复读仍待全册门禁。

## 第五十一章开始前入口（历史记录：2026-11-23）

第二十一册第 1—50 章已完成首轮正文审阅与修订；第 51—95 章共 45 章待精修。第 41 章按六节说明局部卷积、长卷积与混合主干的不同职责，并把 Qwen3、H3+LongConvs 和 StripedHyena 证据绑定各自设置，见[第四十一章精修审计](chapter-41-audit.md)及[卷积序列模型研究记录](../../research/convolution-sequence-models-in-llms-2026-11-23.md)。第 42 章按比较层级、历史表示/查询路径、任务证据、序列复杂度、硬件与生态及条件化结论六节横向综合；删除旧稿重复的比较清单、虚构评分器和越界的 serving 操作流程，并明确与第四十三章混合架构专题的分工。第 43 章区分层间串接与层内并行融合，并将比例、层位置、任务成绩与速度预测限定在各自实验条件内；见[第四十三章精修审计](chapter-43-audit.md)及[Attention–SSM 混合架构研究记录](../../research/hybrid-attention-ssm-architectures-2026-11-23.md)。第 44 章按历史表示、读取方式与融合位置比较卷积、SSM 和 Attention，删除固定的“局部/长程/全局”职责推断及越界的状态协议、rollback 和生产验收流程；见[第四十四章精修审计](chapter-44-audit.md)及[模块路径研究记录](../../research/attention-convolution-ssm-module-roles-2026-11-23.md)。第 45 章按外部语料的信息路径、检索接入位置、失效阶段与分层评估重写，删除重复 RAG 清单、toy 检索代码及缓存/回滚/p99/生产验收流程；见[第四十五章精修审计](chapter-45-audit.md)及[检索增强架构研究记录](../../research/retrieval-augmented-architecture-evidence-2026-11-23.md)。第 46 章将模型架构记忆限定为跨片段/步骤可访问的状态，按外部 K/V 记忆、解耦读取网络、潜在情景记忆和原生递归状态比较 Memorizing Transformer、LongMem、Larimar 与 MemT-1.0；删除重复的个人记忆权限/删除运维清单、generic memory toy 代码及与第四十九章重叠的 Agent 状态内容。详见[第四十六章精修审计](chapter-46-audit.md)和[模型记忆架构研究记录](../../research/memory-augmented-transformer-architectures-2026-11-23.md)。第 47 章改为比较 Jamba 的混合主干加 FFN-MoE、BlackMamba 的 Mamba+routed MLP，以及 Routing Mamba 在 SSM 投影和 Samba 混合主干中的专家化；删除无来源通用成本模型、假设性路由代码与 serving/Agent 运维流程。详见[第四十七章精修审计](chapter-47-audit.md)和[组合架构研究记录](../../research/moe-ssm-attention-composition-2026-11-23.md)。 第 48 章将职责收窄到视觉语言与长视频模型中 Mamba 所在位置、视觉序列化/连接、跨模态对齐和文本读取路径；删除与第十五册重复的通用多模态预算、serving、安全及系统工作流。详见[第四十八章精修审计](chapter-48-audit.md)及[多模态 Mamba 架构研究记录](../../research/multimodal-mamba-architectures-2026-11-23.md)。第 49 章区分模型能力层、宿主执行层与环境结果层，比较 ReAct 提示、Toolformer/Gorilla/ToolLLM 工具能力路径及外部 Agent 闭环；以 AgentBench、OSWorld、τ-bench 说明评测成绩绑定完整交互配置，详见[第四十九章精修审计](chapter-49-audit.md)及[Agentic Model Architecture 研究记录](../../research/agentic-model-architecture-boundaries-2026-11-23.md)。第 50 章从层间、head 级、记忆分支和分数级融合综合当前架构证据，并将未来趋势保留为条件化推断；详见[第五十章精修审计](chapter-50-audit.md)及[未来架构组合研究记录](../../research/future-architecture-composition-2026-11-23.md)。

第 12、14、18、20—27、29—33、35、37—44 章公式的 GitHub MathJax 实际渲染仍待验收；第 32、33、43 章 Mermaid 页面渲染也待验。第四十五至五十章无公式、代码或 Mermaid 图。下一入口为第 51 章《GPT 系列：Decoder-only、Scaling 与 In-Context Learning》；目标读者试读、外链最终逐条复核和第二十一册连续复读仍待全册门禁。

## 第五十二章开始前入口（历史记录：2026-11-23）

第二十一册第 1—51 章已完成首轮正文审阅与修订；第 52—95 章共 44 章待精修。第五十一章把 GPT-1/2/3 的任务适配、早期 scaling 证据和 ICL 历史位置拆成五节，区分语言模型损失与下游任务指标，并把机制、架构配置、compute-optimal 训练及后训练分别交接到第十二、十六/十七、五十二和五十三章；详见[第五十一章精修审计](chapter-51-audit.md)与[研究记录](../../research/gpt-lineage-scaling-icl-evidence-2026-11-23.md)。

第 12、14、18、20—27、29—33、35、37—44 章公式的 GitHub MathJax 实际渲染仍待验收；第 32、33、43 章 Mermaid 页面渲染也待验。第四十五至五十一章无公式、代码或 Mermaid 图。下一入口为第 52 章《Chinchilla、PaLM 与 compute-optimal training》；目标读者试读、外链最终逐条复核和第二十一册连续复读仍待全册门禁。

## 第六十章完成时的入口（2026-11-24）

第二十一册第 1—57 章已完成首轮正文审阅与修订；第 58—95 章共 38 章待精修。第五十四章把原始 LLaMA 的稠密 decoder 基线、RMSNorm/RoPE/SwiGLU 组件位置、Llama 2/3 的版本演进以及 Llama 4 的 MoE/early-fusion 边界放在同一条架构历史线上；将家族级讨论与第十七章配置读取、第五十七章训练工程分开。旧稿重复讲解模块、数据、显存/服务成本和工程清单的 26 个小节已收束为五节；详见[第五十四章精修审计](chapter-54-audit.md)、[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/54-LLaMA现代开源LLM基线.md)与[研究记录](../../research/llama-family-architecture-evidence-2026-11-24.md)。

第五十五章聚焦 Mistral 7B v0.1 的 GQA/SWA 与 Mixtral 8x7B 的 FFN 稀疏专家，区分 KV 状态、直接注意力范围、总权重和激活计算，并把论文 benchmark 结论限定到原评测。旧稿 28 个标题对 GQA、滑窗、MoE 路由、资源成本和 serving 流程有重复展开；精修后分为五节，结构与职责见[第五十五章精修审计](chapter-55-audit.md)、[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/55-Mistral-Mixtral高效与MoE.md)和[研究记录](../../research/mistral-mixtral-architecture-evidence-2026-11-24.md)。

第五十六章改为固定版本的公开证据阅读：Gemini 1.0/1.5 的结构披露，GPT-4V 与 Claude 3 的视觉接口与未披露内部路径，以及产品窗口和 NIAH 实验之间的边界。旧稿 31 个 H2 反复展开通用评测、安全、权限、服务成本和工具 harness 清单，且没有对应章节大纲；精修后按五节整理公开结构、接口、产品和实验观察，并将多模态架构、长上下文评估和 Agent 执行细节回链各自主讲章节。详见[第五十六章精修审计](chapter-56-audit.md)、[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/56-Gemini-GPT4V-Claude3多模态与长上下文公开证据.md)与[研究记录](../../research/gemini-gpt4v-claude3-disclosure-evidence-2026-11-24.md)。

第五十七章以 Qwen2.5、Llama 3.1 与 Gemma 2 的固定版本报告为案例，比较公开证据能支持的数据规模与混合、tokenizer、训练阶段、基础设施和蒸馏信息，并区分开放权重、训练数据、完整配方与许可证。旧稿将多个版本及数据、后训练、评估和服务清单混在一起；精修后采用五节历史案例结构，并将训练方法、数据工程与模型架构细节交回第五册及前序章节主讲。详见[第五十七章精修审计](chapter-57-audit.md)、[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/57-Qwen2.5-Llama3.1-Gemma2训练工程.md)与[研究记录](../../research/open-weight-training-engineering-qwen25-llama31-gemma2-2026-11-24.md)。

第五十八章以固定版本报告区分 V2/V3 架构延续、MTP 训练目标和 V3-Base 上的 R1 后训练路线；旧稿 26 个小节重复展开 MLA、MoE、RL、serving 和通用评估清单，精修后按版本边界、V2 架构、V3 负载均衡、MTP、R1 多阶段后训练和稠密蒸馏证据组织。详见[第五十八章精修审计](chapter-58-audit.md)、[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/58-DeepSeek-V2-V3-R1-架构与训练路线.md)与[研究记录](../../research/deepseek-v2-v3-r1-architecture-training-2026-11-24.md)。

第五十九章把第五十一至五十八章的固定版本案例归入架构、训练/适配、接口和披露证据四组比较轴；旧稿 32 个二级标题重复通用评估、资源账本、Pareto、发布和回滚清单，现已改为六节、24 个编号标题，目录与章节大纲同步。新建本册完整当前目录索引并链接本章大纲；同步练习 31BQ、面试题 31BR、`PAPERS.md` 第 56 项和知识图谱。审计见[第五十九章精修审计](chapter-59-audit.md)、[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/59-主流模型创新点架构训练证据.md)及[研究记录](../../research/mainstream-model-architecture-training-evidence-2026-11-24.md)。

第六十章从通用 checklist 改为“读懂新架构主张”的机制与证据章节。旧稿的 27 个标题扩展到通用评测、单位成功成本、serving/回滚、安全和 Agent 运维，并与第四十二、五十、五十九章和其他册冲突；新稿按六节、十六个小节解释变化层、MLA/MoE/SSM/记忆检索/Agent 的信息路径、资源预测、任务压力点和证据归因。同步正文、完整结构大纲、读者目录、设计目录、练习 31BS、面试题 31BT、`PAPERS.md` 第 57 项及知识图谱。详见[第六十章精修审计](chapter-60-audit.md)、[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/60-新架构主张的信息路径资源与证据.md)和[研究记录](../../research/architecture-claims-mechanisms-and-evidence-2026-11-24.md)。

第 12、14、18、20—27、29—33、35、37—44、52 章公式及第五十三至五十五章公式的 GitHub MathJax 实际渲染仍待验收；第 32、33、43 章 Mermaid 页面渲染也待验。第五十六至六十章无公式、代码或 Mermaid 图。第二十一册第 1—60 章已完成首轮正文审阅与修订，第 61—95 章共 35 章待精修，下一入口为第 61 章《KDA：把历史表示成可门控的递归状态》；目标读者试读、外链最终逐条复核和第二十一册连续复读仍待全册门禁。

## 第六十一章完成时的入口（2026-11-24）

第六十一章收束到 Kimi Linear 报告中的 KDA 状态转换：逐 key 通道对角衰减、value 预测误差与标量 delta 写入，以及 chunkwise WY/UT 训练视图。章节职责与正文诊断、版本和实验边界见[第六十一章精修审计](chapter-61-audit.md)，公式来源见[KDA 研究记录](../../research/kimi-delta-attention-kda-2026-11-24.md)。第 62 章继续负责通用 Gated DeltaNet，避免重复讲 KDA 专属参数化；第 63 章负责 Gated MLA latent cache，第 88 章负责 Kimi K3 全模型案例。第六十一章公式源码与数学含义已静态核对，GitHub MathJax 页面渲染仍待验；未更新 PDF。全套书精修 goal 继续 active。

## 第六十二章完成时的入口（2026-11-24）

第六十二章聚焦通用 Gated DeltaNet 的递归状态读写：从 key-by-value 矩阵方向推导 delta correction，再说明全局衰减与定向写入的先后关系、独立 output gate、chunkwise WY/UT 和固定状态的能力边界；与 KDA、Gated DeltaNet-2 的门控维度按公式区分。旧稿 30 个二级标题重复并混合 DeltaNet 机制、相互不一致的状态方向与公式、缓存/回滚/租户隔离等通用 serving 清单；重写与编辑决定见[第六十二章精修审计](chapter-62-audit.md)，章节结构见[大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/62-Gated-DeltaNet带门控的Delta状态更新.md)。

依据 Gated Delta Networks v3、Gated DeltaNet-2 v2 原始论文核对状态更新、WY/UT 视图、GDN/KDA/GDN-2 的特例关系与作者报告的匹配实验；两份教学材料仅用于检查在线回归和 WY 的解释路径。来源版本、阅读位置和不能外推的边界见[Gated DeltaNet 研究记录](../../research/gated-delta-net-evidence-2026-11-24.md)。同步第二十一册 reader TOC 和设计 TOC、练习 31BU、面试题 31BV、`PAPERS.md` 第 59 项及 `KNOWLEDGE_GRAPH.md`；术语表已有相关词条，未重复添加。

静态验收：正文、大纲和两份目录的 16 个标题名称、层级和顺序匹配；状态递推矩阵方向、残差与在线回归梯度、二维示例、GDN-2 的 erase/write 公式已手工核对；练习和面试题编号唯一；新增本地链接目标存在。`git diff --check` 通过。正文公式源码及其含义已静态检查，GitHub MathJax 页面实际渲染仍待验收；目标读者试读、外链终审和第二十一册连续复读仍属全册门禁。未运行测试或构建，未更新 PDF/`plan.md`。第二十一册第 1—62 章已完成首轮正文修订，第 63—95 章共 33 章待精修，下一入口为第 63 章《Gated MLA：潜变量缓存与通道输出门》。全套书精修 goal 继续 active。

## 第六十三章完成时的入口（2026-11-24）

第六十三章聚焦 Kimi K3 Gated MLA：MLA 为每个历史 token 保存共享低秩 KV latent，仍通过全局 attention 按位置读取，再用输入依赖的逐通道 full-rank gate 调制 attention 输出。旧稿 34 个二级标题反复展开 latent cache、一般性压缩评估和 serving 生命周期，并把 Gated MLA 错写成显式 KV 与 latent KV 的动态选路/融合；新稿将通用 cache 保真交第 70 章、NoPE 性质交第 64 章、混合层间结构交第 65 章、Kimi K3 全模型交第 88 章。诊断和决定见[第六十三章精修审计](chapter-63-audit.md)，结构见[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/63-Gated-MLA潜变量缓存与通道输出门.md)。

依据 DeepSeek-V2 v5 与 Kimi K3 v2 原论文，及 Moonshot 官方固定 config revision，核对 MLA 的 KV latent、可吸收投影、decoupled RoPE、Kimi K3 NoPE、输出 gate 公式与参数披露。来源版本、阅读位置和不能据整模型成绩归因给 gate 的边界见[Gated MLA 研究记录](../../research/kimi-gated-mla-evidence-2026-11-24.md)。同步 reader TOC、设计 TOC、练习 31BV、面试题 31BW、`PAPERS.md` 第 60 项与 `KNOWLEDGE_GRAPH.md`；术语表已有 Gated MLA 条目，未重复添加。

静态验收：正文、大纲和两份目录的 7 个 H2、8 个 H3 名称、层级和顺序匹配；输出 gate 的作用位置、latent/位置分支维度、每 token cache 账本及三组教学数值已手工核对；新增本地链接目标存在，配套编号唯一。`git diff --check` 通过。公式源码与形状已静态核对，GitHub MathJax 实际渲染、目标读者试读、外链终审和全册连续复读仍待验收。未运行测试或构建，未更新 PDF/`plan.md`。第二十一册第 1—63 章已完成首轮正文修订，第 64—95 章共 32 章待精修，下一入口为第 64 章《NoPE：没有显式位置变换，不等于没有顺序》。全套书精修 goal 继续 active。
