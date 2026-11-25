# 第八十章精修审计

日期：2026-11-25。范围：第 80 章完整结构大纲、正文主题思想不变重写、读者/设计目录与配套索引同步；不处理 PDF。

## 旧稿诊断与写作决定

旧稿以“Agent 要快”开篇，把 MoE、3:1 注意力和 MTP 写成笼统的协同提速结论，但没有把三个设计分别映射到每 token 前馈计算、注意力可见范围和解码候选路径。3:1 仅用模 4 教学约定，容易遗漏真实布局的 leading Full 层；代码统计的是抽象四层块，无法核对发布配置的 45 层顺序。参数段混合模型卡与技术报告口径，未说明含 MTP-3 的统计。

旧稿将 MTP-3 的预测头数近似等同于一次服务轮的输出数，对目标验证、连续接受前缀和拒绝修正讲解不足；有关 vLLM 完整支持状态、专家过载被 MTP 放大、两种 KV cache 生命周期及多项部署指标的陈述没有相应的配对实验或固定实现依据。Context Manager 段则从一个模型卡协议扩展成通用 Agent 工程清单，偏离本章的架构与证据归因职责。

保留“Step 3.5 如何同时处理稀疏计算、局部/全局注意力和多 token 解码”的中心思想，按参数账本、实际 45 层布局、理论 pair 账与消融、SWA query heads/gate、MTP 训练目标、draft/verify、运行时与 harness 边界展开。正文直接走查配置层号和一个拒绝候选后的输出状态；公式只归纳已说明的注意力 pair 和验证规则。所有吞吐、消融和参数差异绑定原来源与范围，删除无证据的 MoE overflow 与 KV cache 生命周期推断。

## 研究依据与边界

技术报告 v2 支持 196B/11B backbone、含 MTP-3 的 198B/13B 表格、45 层构成、S3F1 起始顺序、窗口/头数、MTP 偏移与训练安排、30B-A3B 消融和 gate 定义。固定 Hugging Face config revision 支持 layer_types、层数、最大序列长度和 MTP 层数。模型卡另列 196.81B（196B backbone + 0.81B Head），并报告吞吐和 BrowseComp with/without Context Manager；这一参数范围与技术报告统计没有完全对账，本文保留差异。当前公开 vLLM recipe 配置一层 MTP 和一个 speculative token；模型卡仍说明 full MTP3 未可用。两者是公开文档状态，不是本地 runtime 测试。Chen 等 speculative sampling 论文支撑通用验证与残差校正机制，不把其 Chinchilla 加速数字移用于 Step 3.5。

完整检索词、来源版本、阅读位置、来源支持范围和未验证项见[第 80 章研究记录](../../research/step-3-5-flash-s3f1-mtp-evidence-2026-11-25.md)。

## 本轮改动

- 重写[第 80 章正文](../../book-21-transformer-architecture-evolution/chapters/80-step-3-5-flash的mtp与滑动窗口moe.md)，新增完整的[章节结构大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/80-Step-3.5-Flash-S3F1-MTP-3与11B激活参数.md)。
- 修正读者目录和设计目录中的章节标题，并同步 80.1–80.9 及下级标题；设计目录链接到章节大纲。
- 更新第 80 章的配套练习、面试追问、Step 3.5 知识图谱、术语口径及 PAPERS.md 来源条目。

## 静态验收与遗留项

- 正文和结构大纲均为 9 个 H2、22 个 H3；标题文字、层级和顺序一致。两份目录中的 80.1–80.9 与 80.1.1–80.7.2 标题、顺序一致。
- 正文 4 组 math 围栏成对；未使用 operatorname 命令或圆括号形式的公式分隔符。手工核对因果 pair 数：$n=4096,W=512$ 时，SWA 为 1,966,336、Full 为 8,390,656；33 SWA + 12 Full 为 165,576,960，约为全 Full 层 pair 数的 43.9%。它是未加权可见 pair 账，不是 FLOPs 或延迟。接受概率和残差分布按目标分布 $p$、草稿分布 $q$ 静态核对。
- 正文三个本地跨章链接均指向现存文件；结构大纲、研究记录和正文的导航链接已设置。git diff --check 通过。
- GitHub MathJax 页面尚未实际查看，公式视觉验收仍待完成；目标读者试读、外链终审和第二十一册沿读者路线连续复读仍属整册门禁。
- 未加入或运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建；未更新 PDF 或 plan.md。

第二十一册第 1—80 章已完成首轮正文审阅与修订，第 81—95 章共 15 章待精修。下一入口为[第八十一章](../../book-21-transformer-architecture-evolution/chapters/81-deepseek-v4.1-flash-causal-encoder-decoder.md)。全套书精修 goal 继续 active。
