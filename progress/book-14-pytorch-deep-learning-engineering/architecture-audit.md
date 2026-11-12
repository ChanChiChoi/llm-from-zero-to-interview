# 第十四册架构审视：PyTorch 与深度学习工程

审视日期：2026-11-09

阶段：册级定位、简介/目录审视、章节地图和资料扫描完成；稳定十章骨架，已补充简介和目录导航。进度更新至 2026-11-11：十章均已完成章级扫描与精修；书册完成门禁通过，下一步进入第十五册架构审视。

## 书册契约（book_contract）

- **读者问题：** 读者认识模型公式或能阅读训练代码，却不能稳定追踪 tensor shape、梯度来源、batch 组成、训练状态和故障原因。本册回答“怎样把模型与数学变成可运行、可检查、可复现的 PyTorch 实验”。
- **目标读者与前置：** 有基本 Python、机器学习或神经网络背景；需要补足 PyTorch 实现、训练脚本、调试和面试工程判断。默认会用 Python 函数/类、列表/字典，知道损失和梯度下降；不要求预先掌握完整 Transformer 或分布式系统。数学短板可回第十三册。
- **读者能力：** 能操纵并检查 tensor；解释 autograd 和模型状态；构建数据管线及训练/验证/恢复循环；分析精度、显存和分布式数据同步；用最小实验和测量验证问题假设；实现小型 decoder-only Transformer 的关键路径。
- **纳入：** PyTorch eager 基础、模型/数据/训练工程、小型实验中的 AMP/显存和单机多卡入门、debug/profiling、Transformer 组件实现和证据化工程判断。
- **不承担：** 不替代第十三册数学推导、第三册完整项目路径或第五册大规模训练系统；不在此展开多机集群调度、完整 TP/PP/FSDP 系统设计、服务部署及低层 kernel 开发。
- **书册类型与独特价值：** 面向 LLM 实现和面试的工程教材，以一条可检查的小型实验链连接张量、自动微分、模型、数据、训练循环、诊断和 Transformer 实现。
- **前后衔接：** 数学按需回第十三册；读完后进入第三册项目实践，再到第五册学习完整训练系统。第十章训练的可复现解释也可接第十二册面试表达。

## 简介审计（intro_audit）

- 原简介只有笼统的“训练脚本/调试/核心组件”目标，没有明确前置条件、范围、不承担内容和读法，亦未说明十章如何组成一个递进过程。
- 已更新[第十四册简介](../../book-14-pytorch-deep-learning-engineering/简介.md)：列出读者问题、前置、能力目标、范围边界、路径，以及十章各自的问题和前后衔接；区分数学基础、项目实践和大规模系统的职责。
- 各章短摘要依据简介、目录和章首概述整理；逐章精修完成后必须按实际正文回写，不把现摘要当作章节验收结论。

## 目录审计（toc_audit）

- 原目录只有十个文件链接，无法判断每章产物、依赖、排错入口及相邻书册边界。
- 已更新[第十四册目录](../../book-14-pytorch-deep-learning-engineering/目录.md)：增加主线路径、按读者问题选读的入口、章级核心问题/依赖/产出/交接表及第十三、三、五册链接。
- 保留顺序化目录，表格说明明确学习递进；暂不把部分章节标成选读，因为 tensor、autograd、模型、数据、训练、诊断和集成组件构成了本册声明的共同能力链。

## 章节地图（chapter_map）

册级阶段仅参考简介、目录、十个章名和章首概述，未通读章节正文。

| 章 | 作用 | 依赖与交接 | 架构关注点 |
| --- | --- | --- | --- |
| 1 PyTorch 张量基础 | 已将 token id→embedding→线性投影→next-token loss 作为贯穿链，补齐 shape contract、类别轴、device/stride 边界和手算例子。 | 起点；连接 autograd、模块与 Transformer。 | [第一章审计](chapter-01-audit.md)记录章级扫描、代码执行与遗留门禁；核心 demo 仅 CPU 验证。 |
| 2 Autograd 与反向传播 | 已补手算链式法则/VJP、梯度累积和有效 token 归一化；校准无梯度上下文、hook 与 checkpoint/DDP 边界。 | 依赖第 1 章；连接模块组织、训练循环和梯度故障诊断。 | [第二章审计](chapter-02-audit.md)记录扫描、字数、六组已验证代码和 44 个静态片段；CPU-only 验证。 |
| 3 nn.Module 与模型组织 | 已完成：沿注册树说明参数、子模块和 buffer 如何进入 optimizer、设备迁移与 checkpoint；补足额外状态、安全加载、模式与包装边界。 | 依赖第 1–2 章；交给数据和训练流程。 | [第三章审计](chapter-03-audit.md)记录三条核心教学链、可复算例子和逐块代码状态；DDP/compile 只保留版本化入口。 |
| 4 Dataset 与 DataLoader | 已精修：从样本 contract 推到 padding、shift 后有效目标、采样/分片和 worker 吞吐诊断。 | 依赖第 1–3 章；交给训练循环，rank 同步交第七章。 | [第四章审计](chapter-04-audit.md)记录三条教学链、手算例、PyTorch 2.15 API 扫描和代码状态；多 rank/CUDA 未实测。 |
| 5 训练循环工程 | 已精修：从有效 loss 分母和梯度累积/裁剪推到 scheduler 更新、token 加权验证、checkpoint 安全保存与恢复后的下一步一致性。 | 依赖第 1–4 章；交给 AMP、分布式和诊断。 | [第五章审计](chapter-05-audit.md)记录章级扫描、三条教学链、数例、39 个代码块状态和 CPU 验证边界。 |
| 6 混合精度与显存优化 | 已精修：沿 FP16/BF16 数值范围、GradScaler 更新、checkpoint 重算和累积窗口，推导显存账本并按 OOM 发生阶段选择诊断。 | 依赖第 2、5 章；交给 DDP 与性能诊断，token 分母和真实 update 约定回连第五章。 | [第六章审计](chapter-06-audit.md)记录两条核心教学链、手算例、14 个代码块状态及 CPU 验证边界；CUDA 与 PyTorch 2.15 runtime 未实测。 |
| 7 分布式训练入门 | 已沿进程组/rank、DDP 梯度、启动/分片、通信/no_sync、FSDP/checkpoint 和排错六条主线精修，补入 token 分母与 sampler 尾部数例。 | 依赖第 4–6 章；完整并行策略交第五册。 | [第七章审计](chapter-07-audit.md)记录六条核心教学链、PyTorch 2.15 API 扫描、数例、21 个代码状态和 CPU-only demo；未实测 CUDA/NCCL/多 rank。 |
| 8 Debug 与 Profiling | 已精修：从 shape/label、数值、梯度和显存诊断，推进到 profiler schedule、有效 token 吞吐、稳态测量与 `torch.compile` 正确性/重编译核验。 | 回用前七章；连接第九章实现核验、第十章判断。 | [第八章审计](chapter-08-audit.md)记录三条教学链、PyTorch 2.15 定向扫描、数值例子及逐块代码状态；仅 CPU demo/玩具编译前向已验证，CUDA/多 rank 未实测。 |
| 9 实现 Transformer 组件 | 已沿表示/Norm、attention/位置、训练目标和 KV 状态四条教学链精修；明确 PyTorch 2.15 非方阵 SDPA mask、dropout 与 API 边界。 | 回用第 1–8 章；完整 toy GPT 项目交第三册。 | [第九章审计](chapter-09-audit.md)记录候选去向、数例、篇幅、14 个代码状态及 CPU 集成 demo；PyTorch 2.12 CPU 验证，CUDA/2.15 runtime 未实测。 |
| 10 工程问题与判断 | 已将 shape/data/optimizer/checkpoint、AMP/显存、DDP reduction、故障定位与 Transformer cache 串成“条件—假设—最小实验—证据”路径，并加入可运行 audit demo。 | 回用前九章；面试表达接第十二册。 | [第十章审计](chapter-10-audit.md)记录三条核心教学链、数例、固定 PyTorch 2.15 来源、两段 CPU 验证和工程证据边界。 |

## 册级扫描（discovery_scan）

- **时间与范围：** 2026-11-09；按本册契约与十章目录扫描 PyTorch 2.15 当前官方文档/教程，聚焦张量/Autograd、AMP、分布式、Profiler、基准测量和 `torch.compile`；主要检查近年稳定方向及 2.15 已发布变化。该扫描为目录级，不替代逐章定向扫描，也不试图穷尽所有设备或 API。
- **资料类型：** PyTorch 官方 release blog、固定版本 API 文档和教程。搜索结果只作入口；对纳入判断使用了原始官方页面：`torch.compile` 教程、Profiler/benchmark 教程、DDP 教程、AMP 文档及 `fully_shard` API。
- **候选 A：PyTorch compiler 使用与诊断。** PyTorch 2.15 官方教程解释 `torch.compile` 的 graph break：不支持的代码片段会退出并在之后继续编译，带来优化机会损失；`fullgraph=True` 可暴露首个断点；收益随模型、batch 与瓶颈变化。候选去向：第八章新增/强化一条“eager 基线 → 编译 → 正确性对照 → 预热后测量 → graph-break 排查”的精修链。理由：它属于本册实践调优与 debug 能力，且不要求扩展成编译器教材；最终深度等第一至第八章正文审阅时按现有覆盖核定。事实见[PyTorch API 边界记录](../../research/pytorch-training-api-boundaries-2026.md)。
- **候选 B：测量工具的扰动与边界。** PyTorch 2.15 Profiler 可观察 CPU/device 活动、shape、stack 和内存，但 shape/stack 采集会产生开销；benchmark 教程提示需处理线程数与 CUDA 异步同步。候选去向：第八章的基准与 profile 流程需明示先形成无 profiler 的计时基线，再以 profiler 定位，不能把 trace 时延直接当端到端性能。来源同上。
- **候选 C：分布式 API 的版本变化。** PyTorch 2.15 发布说明记录默认 NCCL backend 和 collective hang flight recorder 变化，并有 FSDP2 精度/通信控制更新；DDP 教程强调各 rank 必须按同一顺序到达同步点。候选去向：第七章按版本标注一个排障/阅读入口，区分教学级 DDP、PyTorch 2.15 特定发布变化和生产集群恢复；FSDP2 与高级并行继续交第五册。事实与边界见[分布式训练研究记录](../../research/distributed-training-parallelism-2026.md)。
- **候选 D：AMP 与编译组合。** PyTorch 2.15 AMP 文档包含 compiled backward autocast 上下文的版本特定配置注意事项。第六章扫描仅确认 checkpoint 的 `preserve_rng_state` 在 `torch.compile` 下边界；编译器专属 autocast 配置留给第八章按实际使用场景核验，不在第六章主线引入内部参数。
- **扫描结论：** 现有十章覆盖核心实现链，无证据要求新增章节或改变顺序。第八章需把编译优化和测量方法纳入章级扫描；第七章需检查分布式 API/故障边界。候选均适配既有章节，未发现需要越过本册范围的新独立主线。

## 架构决定（architecture_decisions）

1. 保留十章数量和现有顺序；正文章级审视如发现依赖或边界问题再返回此处。
2. 简介强调小型/可控实验、工程证据和前置条件；不再笼统承诺“熟练”或暗示本册能完成集群级大模型训练。
3. 第八章承接 debug、profiling 与性能测量，并在正文扫描后纳入 `torch.compile` 的正确性/性能验证与 graph-break 诊断；不增加单独编译器章节。
4. 第七章主讲分布式心智模型和 DDP 入门；FSDP2/并行策略只按读者理解的必要程度设置入口，完整规模化并行交第五册。
5. 第九章可实现小型模型组件和生成路径，不把端到端项目重复到第三册，也不声称可复刻生产 Transformer 训练栈。
6. 简介和目录已按当前骨架更新；后续每章完成即同步该章短摘要，并核验相关交叉链接。
7. 第六章不改变十章骨架；把 AMP dtype/scaler 一致性、按窗口归一化与真实 update 计数、checkpoint RNG 追踪和分段 peak 测量并入现有主线。FlashAttention 仅作为显存机制的证据入口，不展开 kernel 实现。

## 知识去向（coverage_map）

| 知识范围 | 本册入口 | 深入/配套去向 | 当前决定 |
| --- | --- | --- | --- |
| 线性代数、梯度和优化器数学 | 第 1、2、5 章所需机制 | 第十三册 | 本册讲 API 和执行流，不复制数学推导。 |
| 张量、模型、数据、训练循环和基础代码 | 第 1–5 章 | 第三册项目 | 本册提供工程组件；第三册串成端到端任务。 |
| AMP、显存与单机多卡入门 | 第 6–7 章 | 第五册规模化训练 | 保留基础理解、测量和排错，生产拓扑/吞吐归第五册。 |
| Profiler、benchmark、torch.compile 与调试 | 第 8 章已完成章级扫描与精修 | 第五册系统级性能工程及框架文档 | 教会验证和定位；性能效果按配置说明。 |
| Decoder-only Transformer 组件 | 第 9 章 | 第三册 toy GPT；第五册训练和扩展 | 本册解释组件与张量流，项目/规模化训练不重复。 |
| 面试工程判断 | 第 10 章 | 第十二册面试表达 | 本册训练证据化判断，第十二册组织回答与求职呈现。 |

## 册级完成门禁（completion_gate）

复核日期：2026-11-11。按简介和目录的 1→10 主路径复读，逐章核对章节地图、正文小结、下章交接和对应章审计；十章全文精修、章级扫描及逐章审计均已有记录，本次门禁以这些完整审阅记录为内容依据，不重复章级扫描。

- **定位、承诺与书系分工：通过。** 简介列清读者前置、能力目标、范围、不承担内容和读法；十章短摘要与目录的依赖/产出/交接表一致。数学回第十三册，端到端项目交第三册，规模化训练交第五册，面试表达交第十二册。
- **主线、顺序与交接：通过。** 阅读路径从 tensor、Autograd、Module 推进到数据和训练状态，再到精度/显存、分布式、诊断、Transformer 组件及工程判断。重复主题各有层次：第三章解释模型状态，第五章解释完整训练状态，第十章综合恢复判断；第四章讲数据与 sampler，第七章讲 rank 同步，第八章讲排错；第五/六章建立训练与资源机制，第八/十章分别承担系统诊断和综合决策。未发现影响读者承诺的断层、倒序或需迁移的内容。
- **篇幅、例子与代码状态：通过。** 十份章审计逐讲记录独立教学链篇幅、可复算例子及代码状态；所有核心链均达到对应下限。已验证、静态审阅和未验证环境边界均按代码块记录，不能据 CPU demo 推出 CUDA、多卡或生产性能结论。
- **视觉、公式与导航：通过（范围受限）。** 各章审计均记录了视觉机会及其表格、张量轨迹或逐步案例表达；本册正文没有 Mermaid 图，因此不适用目标平台 Mermaid 渲染项。公式完成源码围栏/结构检查，但没有目标平台 MathJax 页面渲染记录。对本册正文和审计中的 90 个本地 Markdown 链接目标完成存在性检查，未发现断链；未执行全量外链 HTTP crawl。
- **门禁结论：通过。** 当前未发现 8.3 所列阻断缺口；未验证的设备、版本、真实 workload 和渲染范围作为下列开放风险保留，不扩张正文结论。

## 未决风险（open_risks）

1. 公式完成源码级检查，但未在目标平台实际渲染；外链没有全量 HTTP crawl。若目标发布平台出现排版或外链问题，应按页面结果修复。
2. PyTorch API 与设备支持持续演进。具体代码必须按实际声明的版本、设备和依赖核验，不以 2.15 文档代表所有未来版本。
3. 本地验证环境主要为 PyTorch 2.12.0+cpu；没有 CUDA/GPU、PyTorch 2.15 runtime、真实 NCCL/多 rank 或目标 workload 结果。第七章和相邻章节的 demo 均明确限制为 CPU、纯 Python 或静态审阅证据。
4. 第八章的 `torch.compile` 结论按 PyTorch 2.15 官方文档收窄，但本地只在 PyTorch 2.12 CPU 玩具前向上验证；第九、十章没有新增 AMP 与编译组合实现。若后续章节引入该组合，应按实际版本和配置核验。

## 工作交接（work_handoff）

- **已完成：** 依据简介、目录、BOOK_SERIES/ROADMAP 书系位置和相邻册边界形成契约；完成以 PyTorch 2.15 官方来源为主的册级扫描；决定保留十章骨架；简介/目录与各章短摘要已按逐章正文复核。第一至第十章均已完成全文审阅、章级扫描和正文精修。
- **当前处理：** 第一至第十章分别建有[第一章审计](chapter-01-audit.md)、[第二章审计](chapter-02-audit.md)、[第三章审计](chapter-03-audit.md)、[第四章审计](chapter-04-audit.md)、[第五章审计](chapter-05-audit.md)、[第六章审计](chapter-06-audit.md)、[第七章审计](chapter-07-audit.md)、[第八章审计](chapter-08-audit.md)、[第九章审计](chapter-09-audit.md)和[第十章审计](chapter-10-audit.md)。第六章 CPU BF16 demo、第七章纯 Python DDP 机制账本、第八章 CPU debug/profiler demo 与编译玩具前向、第九章 Transformer 集成 demo、第十章 layout 与工程 audit demo 已在 Python 3.12.9 / PyTorch 2.12.0+cpu CPU 环境验证；其余片段按章审计记录为静态审阅。CUDA、PyTorch 2.15 runtime、真实多 rank/NCCL 和严格多 worker 续跑未实测。
- **下一步入口：** 第十四册书册完成门禁已于 2026-11-11 通过；按全书顺序进入第十五册《多模态与生成模型专题》的架构审视。
- **执行限制：** 按用户要求不更新 PDF；不修改总体计划文件。
