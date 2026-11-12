# 第十二册第五章审计：ML 与 LLM 基础面试

审阅日期：2026-11-05  
状态：本轮章级扫描、正文精修和静态门禁完成；第十二册连续复读已通过第一至第五章。本章无可执行章节代码。Mermaid/MathJax 页面渲染与本册最终门禁仍待处理。PDF 不更新。

## Scope 与章节扫描

- 全篇审阅《从概率到模型行为：ML 与 LLM 的基础》，覆盖 §§5.1—5.24：条件概率、最大似然、交叉熵、NLL/PPL/KL、优化与泛化、Transformer/attention、位置编码、归一化与 MLP、架构类型、tokenizer、训练阶段、幻觉、scaling、综合分析、练习和证据边界。
- 章级扫描检查基础知识是否由定义推进到机制、手算例、比较条件和面试迁移；重点核对 loss 的目标位置与 reduction、PyTorch API 版本、缓存/位置/归一化数例、多语言 tokenizer 证据，以及 scaling 结论是否受实验条件约束。
- 定向核验日期：2026-11-05。研究范围包括 PyTorch 2.15 `cross_entropy`/`kl_div` 的输入与归约语义，Transformer、AdamW、RoPE、RMSNorm、GQA、ALiBi、GPT/BERT/T5、SentencePiece/WordPiece、InstructGPT、DPO、Scaling Laws/Chinchilla 原始论文，以及多语言 tokenizer 研究。知识证据与适用边界分别保存在[交叉熵、最大似然与困惑度来源记录](../../research/cross-entropy-mle-and-perplexity-2026.md)和[tokenizer 设计与训练证据记录](../../research/tokenizer-design-and-training-evidence-2026.md)。

## 候选主题与处置

| 候选或问题 | 去向与处理 | 边界 |
| --- | --- | --- |
| 完整序列 NLL、全局 token mean 与按序列 mean 容易混称 | 在 §5.3 定义目标位置集合，分开说明求和/归一化、prompt mask 与回答条件目标；加入 2-token/8-token 的手算对照 | 相同有效目标集合下，序列求和平均与 token mean 仅差归一化常数；联合正则项及不同 mask 会改变比较结论 |
| PyTorch KL 的 log-probability 输入及 `mean`/`batchmean` 语义 | 在 §5.4.4 固定 PyTorch 2.15 文档，明确三维语言模型张量还需按有效 token mask 归一 | 不将 API 的 `batchmean` 误称为有效 token 平均 |
| 多语言 tokenizer 评价不能由压缩率或单一 intrinsic 指标决定 | 在 §§5.5、5.15 增加跨 tokenizer 指标条件、多语言限制和对照入口；将 2026 年 BPB 预印本纳入来源索引 | 单一约 1.27B 模型、配方、31 种语言和固定 token 预算的 BPB 结果，不代表下游能力或因果效应；新入口已同步至 `PAPERS.md` |
| AdamW 参数排除规则常被写成数学必然 | §5.7.3 改为“部分配方会排除”，要求核对实际参数组 | 参数分组是训练配方选择，不由 AdamW 定义强制 |
| scaling law 易被写成跨规模通用的联合定律 | §5.19 收窄为固定条件下的单变量经验拟合，说明与 Chinchilla 设置的关系，加入假设曲线的代入计算 | 教学拟合值不来自论文，不可推出 benchmark 能力或跨设置最优值 |
| 已有数例及图示存在重复 | 删除 §5.10 重复 attention 数例与 §5.20 重复 Mermaid 流程图，各保留一份 | 保留原有解释链，不改章节骨架或相邻册边界 |

扫描未发现需要新增章节或改变本册范围的成熟候选。MoE、训练/部署体系、对齐安全、研究评估与工程实现保留适用的基础连接，并交给对应技术册或本册专门章节深讲；不在本章展开成第二份专题教材。

## 篇幅与教学例子

- **length_status：** 按连续论证划为五个核心教学单元。统计只计解释性汉字；排除标题、表格、围栏内容、行内代码和链接标记。计数脚本与章节文件相同版本的静态统计结果如下：

| 核心单元 | 汉字数 | 下限 | 状态 |
| --- | ---: | ---: | --- |
| §§5.1—5.5：概率、损失与评估指标 | 3,644 | 1,800 | 通过 |
| §§5.6—5.9：优化、batch 与泛化 | 1,936 | 1,800 | 通过 |
| §§5.10—5.14：Transformer 结构与位置/归一化 | 2,450 | 1,800 | 通过 |
| §§5.15—5.18：tokenizer、训练目标与行为 | 2,334 | 1,800 | 通过 |
| §§5.19—5.21：scaling 与基础分析 | 1,935 | 1,800 | 通过 |

- **example_status：** 章节含完整、可复算的教学输入和中间量：§§5.2—5.5 的序列概率、2/8-token reduction、交叉熵/PPL、padding mask 与 KL；§§5.6—5.9 的梯度更新、AdamW 一步更新、梯度累积与样本权重；§§5.10—5.14 的 attention score/输出、KV cache 容量、RoPE 旋转、LayerNorm/RMSNorm 与 MLP；§§5.15—5.18 的 tokenizer 密度、上下文预算、SFT mask 与偏好目标；§§5.19—5.21 的 6ND FLOPs/墙钟估算、假设幂律曲线代入和完整 token 数据流。所有自行构造的输入均标明为教学假设，不作为论文或生产测量。

## 公式、图示、代码与链接

- **formula_status：** 45 个 math 围栏块配对；静态逐块检查花括号平衡；未见 `\operatorname`。NLL、token reduction、KL、attention、AdamW、RoPE、KV cache、SwiGLU、scaling 和 FLOPs 均配变量解释或就近文字，并有对应数例。2026-11-06 经 MathJax 4.1.3 本地解析通过；该轮发现 RMSNorm 公式的分式缺少分母分组，已修正并在全册重渲染中通过。GitHub 当前页面尚未能核对本地未发布修改。
- **code_status：** 无 Python 或其他可执行章节代码；4 个 `text` 围栏用于文本示意，不代表程序执行。未运行章节代码、测试、模型、benchmark 或 PDF。
- **diagram_status：** 1 个 Mermaid 训练/推理数据流图；源码围栏配对，节点标签使用引号，文字箭头使用标准管道标签。做过源码静态检查，GitHub 页面渲染待本册最终门禁。
- **link_status：** 42 个 Markdown 链接，均为外部链接；本地目标数为 0，未发现本地断链。新增 tokenizer 预印本已在 `PAPERS.md` 增加入口；本章引用仍需随本册最终门禁巡检。
- **sync_status：** 章节号、顺序和本册结构不变，无需修改简介或目录。研究证据记录写入 `research/`，篇章修改与候选去向仅记于本 `progress/` 审计及架构审计。

## 遗留风险与交接

- Mermaid 与 MathJax 的 GitHub 页面渲染尚未验证；历史外链需在本册最终门禁复巡。
- 本章没有可执行代码，因此不存在代码运行验证待办；本轮未执行测试、模型、benchmark 或 PDF 构建/更新。
- **下一步入口：** 连续复读已通过本章；从第六章《Training 面试》继续，检查数据、目标、优化、实验和故障是否构成有证据、有算例、可迁移的回答链。之后完成本册 Mermaid/MathJax 渲染、历史外链巡检和最终门禁。全套书精修 goal 保持 active。

## 整册连续复读记录（2026-11-06）

- 按目录读完 §§5.1—5.24，确认本章承接第四章的程序语义、公式和 shape，转为解释这些实现背后的概率、目标、优化、架构和数据假设，并把系统训练细节交接至第六章；章节顺序、简介和目录无需调整。
- 手工核对了按 token/按序列 reduction（2.6、2 与 13）、NLL/PPL（0.6931、1.3863、2.0794、PPL≈4）、KL 两个方向（约 0.1308/0.1438）、attention 输出（约 [0.670, 0.660]）、GQA cache（512 MiB）、tokenizer 密度（0.5/0.75 token/byte）、AdamW 单步更新及 6ND 时间估算（6.94 天理想、约 23.1 天按 30% 利用率）。算例与章节公式相符，构造值边界清楚。
- 复读中修正 AdamW 手算段落的行内代码标记；未改变算术、来源、章节结构或导航。章节无可执行代码；45 个 math、4 个 text 和 1 个 Mermaid 围栏仍成对，页面渲染和 42 个历史外链巡检留待本册最终门禁。下一步从第六章继续。
