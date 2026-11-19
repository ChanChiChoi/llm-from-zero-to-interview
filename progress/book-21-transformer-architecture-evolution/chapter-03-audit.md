# 第 3 章《Q、K、V、Attention Head 与表示子空间》审阅记录

审阅日期：2026-11-19。状态：章级扫描、正文重写、相关索引同步及本地静态检查完成；GitHub MathJax 页面渲染和第二十一册整册连续复读留待后续门禁。

## 读者契约与章节职责

- **书册目标贡献：** 让读者能沿张量形状理解 Transformer 的多头注意力，分析其投影、读取、写入及 head 证据边界。
- **读者前置：** 理解第二章的单头 Q/K/V、缩放点积、softmax 和 Value 加权；能阅读常见向量、矩阵及张量形状。
- **离章结果：** 能说明每个 head 如何从完整输入学习独立投影、各 head 的输出怎样拼接并投影；用 QK/OV 视角区分读取权重与内容写入；判断 attention pattern、单头干预和联合剪枝分别支持什么结论；计算固定总隐藏宽度下 head 数与 K/V 总宽度的关系。
- **读者路径：** 第二章的单头读取 → 每头独立投影及张量形状 → QK/OV 代数视角 → 子空间与固定语义分工的区别 → head pattern、干预和剪枝证据 → head 宽度、head 数及 KV cache 的正确边界。
- **前后交接：** 第二章完整主讲单头读取；第四章完整主讲 MHA/MQA/GQA/MLA 结构演进；第十一章完整主讲 KV cache 估算和 serving 成本。第九章 §9.8 有一段重复的多头概览，后续精修时应链接本章并改正“把 hidden representation 切成子空间”的表述。

## 章级扫描与外部主题去向

- **检索日期：** 2026-11-19。通过网页检索发现原始论文、ACL Anthology 论文、NeurIPS 会议论文、arXiv 预印本和课程材料；随后打开原文核对支持范围。检索词包括：`multi-head attention head roles interpretability pruning importance prunability primary research`、`attention head stability across random initialization refits 2026 transformer`、`causal attention head gating task roles modern LLM 2025 NeurIPS`、`multi-head self attention number of heads ablation model variations original transformer`。
- **外部教学样本：** Stanford CS224N 2026 Lecture 5 官方幻灯片，重点看第 55—59 页：先展示单头步骤，再引入独立 Q/K/V 投影、逐头 attention、拼接和输出投影。该课材料证明其采用这种讲解顺序，不能证明某一种固定 head 语义或所有读者的唯一最佳顺序。
- **基础结构来源：** [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7；阅读 §§3.2.2、4.2。支持每头独立可学习投影、并行读取、拼接/输出投影及原论文特定 WMT 模型变体结果；不能支持投影彼此正交或通用最佳 head 数。
- **pattern 观察：** [What Does BERT Look at?](https://aclanthology.org/W19-4828/)，Clark 等，BlackboxNLP 2019，阅读摘要和 BERT head 分析。支持 BERT 中 delimiter、位置、广域模式及与部分句法/共指关系的相关观察；不支持这些观察等同于因果功能或跨模型稳定标签。
- **任务限定的专门化/剪枝：** [Voita et al.](https://arxiv.org/html/1905.09418v2)，阅读摘要及 §§3–6；支持作者所测机器翻译模型和数据域下的位置、句法、低频词模式及门控剪枝结果。
- **单头与联合剪枝差异：** [Michel et al.](https://arxiv.org/html/1905.10650v3)，阅读摘要、§§3.2–4.2、§5；支持所测翻译与 BERT NLI 任务中逐个屏蔽和全网联合剪枝给出的不同结果。
- **importance 与 prunability：** [Budhraja et al.](https://aclanthology.org/2020.emnlp-main.260/)，阅读摘要、论文 §§2–3；支持其模型/任务/微调协议下随机剪枝和既有 importance 指标结果，不支持“所有 head 解释均无用”。
- **现代 LLM 因果干预：** [Causal Head Gating](https://papers.nips.cc/paper_files/paper/2025/file/a7f530e11fa19e9551b7a51dbd0f336f-Paper-Conference.pdf)，Nam 等，NeurIPS 2025，阅读摘要、§§3–4 和讨论；支持四个 Llama 3 系列模型变体、句法/常识/数学数据集及多随机种子下的任务条件 head gating；作者明确该方法并不单独解释 head 的精确计算。
- **跨重训稳定性：** [Quantifying LLM Attention-Head Stability](https://arxiv.org/abs/2602.16740)，Bali 等，arXiv v1（2026-02-17），阅读摘要及重训研究方法/结果；这是预印本，结论不外推到未测架构、训练 recipe 或稳定性定义。
- **近期剪枝准则：** [Entropy Meets Importance](https://aclanthology.org/2026.findings-emnlp.2191/)，Choi 等，Findings EMNLP 2026，阅读摘要、§§4.1–4.2、§§5.1–5.3、附录 C.2、D.2、E.3。HIES 将梯度重要度与注意力熵组合，论文在多种模型/数据集和基线下评估；正文只用它说明剪枝准则是方法依赖的，不把该分数当作通用功能标签。
- **未完成全文核验的候选：** [Examining the Universality Assumption of Attention Head Pruning across Datasets](https://doi.org/10.1587/transinf.2026PAL0001)，IEICE 2026-06 advance online。出版方页面摘要报告剪枝选择具有数据集依赖；本轮无法打开 PDF 正文，因此仅在研究记录中注明，不作为正文的关键证据。
- **QK/OV 机制化视角：** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/) 提供 QK/OV 分解视角；它适合分析局部 head 的读写路径，不是完整预测的普遍因果解释。

| 外部主题 | 来源与支持内容 | 决定与去向 |
| --- | --- | --- |
| 多头注意力结构、投影与 shape | Vaswani 原论文的投影矩阵、并行 head、拼接和输出投影 | 本章主线；明确每个投影读取完整输入，区分一般 $d_k,d_v$ 与常见等宽配置 |
| QK 读取与 OV 写入 | Transformer Circuits 的代数分解及原论文注意力定义 | 本章以公式解释局部读写机制，并明确 softmax、残差路径和后续层使其不等于完整因果解释 |
| head pattern 与专门化 | Clark、Voita 等具体模型/任务的注意力观察 | 本章说明观察结果及其范围，不为所有模型定义固定 head 语义 |
| head 重要度、屏蔽与剪枝 | Michel、Budhraja、Voita、Causal Head Gating、HIES | 本章比较证据问题和实验边界；不讲解生产剪枝算法或自拟评分。HIES 的完整方法及比较留作论文入口 |
| 随机重训中的角色稳定性 | Bali 等 2026 arXiv 预印本 | 本章只作有明确预印本标记的当前研究边界 |
| head 数和 KV cache 关系 | 标准 MHA 的总投影宽度，以及第四章/第十一章缓存职责 | 本章只纠正固定 $D$ 下 head 数本身不增大 K/V 总宽度；MQA/GQA/MLA 机制迁移至第四章，第十一章展开缓存账本 |

## 正文修改与去向

1. 删除原 `3.0` 作者工作流、资料 dump、demo 写作计划及重复解释清单；重排为 3.1—3.9，沿“多路读取—投影 shape—QK/OV—功能证据—宽度与缓存—面试和练习”展开。
2. 将“把 hidden representation 切成多个 head”修正为每个 head 从完整输入应用独立可学习投影；补充小型 shape walkthrough，并区分 $d_k$、$d_v$、$D$ 和常见等宽假设。
3. 修正 QK 公式中的缩放和掩码位置；解释输出投影按 head 的行块如何组成 OV 展开，并标明该分解不等于完整预测解释。
4. 把表示子空间写成投影后的低维坐标表示，明确不保证正交、互不重叠或稳定语义分工。用 pattern、单头屏蔽、全网联合剪枝、任务/种子变化及当前方法说明证据边界。
5. 删除将 head 数增加直接等同于 KV cache 上升的错误结论。用常规缓存账本说明固定 $D=Hd_h$ 时 K/V 总宽度不变；MQA/GQA/MLA 的成本和机制交给第四章、第十一章。
6. 删除作者自创的 QK/OV 命中率、冗余率、阈值门禁和标准库模拟代码；将章节及全局练习从运行 demo 改为比较 pattern、干预和剪枝证据。
7. 将新的剪枝/稳定性来源事实及证据边界写入[注意力研究记录](../../research/attention-mechanism-and-interpretation-2026.md)，将阅读入口补入 `PAPERS.md`；同步修订 `EXERCISES.md`、`INTERVIEW_BANK.md` 和 `GLOSSARY_EN_ZH.md` 中依赖已删 demo/旧表述的条目。简介和目录不含逐章摘要，本轮无需修改。

## 本地验收与剩余项

- H2 顺序为 3.1—3.9 连续；共 14 个 Markdown 围栏标记且配对（7 个 `math` 开围栏、7 个闭围栏），没有代码块；行内数学定界符按物理行配对，公式前全角标点间距扫描无命中。
- 第三章 14 个 Markdown 链接中，3 个本地章节目标均存在；外链为原始论文、会议论文或出版方入口。`git diff --check` 通过。
- 前一章末尾的全角标点/行内公式扫描无命中。章节中的旧 QKV 审计 demo、`repeat_kv` 示例和原 3.17/3.20 小节已删除；第九章 §9.8 的重复主讲和错误切分表述留待该章精修时处理。
- 本轮没有运行章节代码、测试、模型、API、benchmark 或 PDF 构建；第三章不再承诺代码/程序操作。GitHub MathJax 页面渲染、整册连续复读及后续章节和历史链接终检留待全书门禁。
- 下一入口：第 4 章《MHA、MQA、GQA、MLA 的演进与 trade-off》。
