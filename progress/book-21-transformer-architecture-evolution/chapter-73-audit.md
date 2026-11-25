# 第二十一册第七十三章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行。旧稿把 Gated Attention 泛化为显式 attention 与递归、局部路径之间的路由，并加入缓存回滚、推测解码、安全策略、P99 运维和玩具证据选择流程；这些内容没有建立在 Qiu 等人提出的机制上，容易使读者误以为输出门会选择计算路径或跳过 attention。

重写后将对象限定为 softmax Gated Attention，主线从标准 SDPA 的 token 加权聚合出发，说明 G1 在 SDPA 后、输出投影前逐特征调节结果；再分开讨论 G1—G5 的位置、headwise/elementwise 粒度、head-specific/shared 设置及论文中的训练、sink、长上下文和成本结果。固定一层的 Q/K/V 时，G1 不会代数地改写该次 softmax 权重；训练后 attention map 的变化和实际计算节省分别作为不同问题处理。

## 结构与分工

现行结构为 7 个二级标题、21 个三级标题。第 62 章解释 Gated DeltaNet 状态更新，第 63 章解释 Gated MLA，第 65 章比较局部、全局和递归路径；第 74 章负责 CSA/HCA 历史压缩。本章只讲 SDPA 输出门及其直接证据，不承担混合路径路由或通用 serving 机制。

完整结构见[第七十三章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/73-Gated-Attention-SDPA输出门控.md)。

## 研究依据与边界

机制和主实验依据 Qiu 等人的 [Gated Attention for Large Language Models（NeurIPS 2025 正式版）](https://papers.neurips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf)。Raschka 的文章用于教学解释，Qwen3-Next 固定 revision 用于核对层布局；SinkProbe v2 的短序列实验与 Kimi K3 诊断分开看待，Spectral Spikes v1 的局部梯度分析限于其固定 attention weights/upstream gradients 条件。详细来源、版本、阅读位置、证据强度和不支持的外推见[Gated Attention 输出门研究记录](../../research/gated-attention-output-gating-evidence-2026-11-24.md)。

## 正文、配套与验收

- 已重写正文并新增完整大纲，同步第二十一册读者目录和设计目录。正文、大纲及两份目录的 7 个 H2、21 个 H3 名称、层级与顺序一致。
- 已同步练习第 18 题、面试题 31CG 及第 15 题回答、Gated Attention 术语、`PAPERS.md` 第 70 项和 `KNOWLEDGE_GRAPH.md`；另修正第 83 章把 G1 与 hybrid path routing 混为一谈的描述。
- 已静态检查正文公式源码与数学含义、表格指标和配置、标题层级、相关本地链接与新增编号；未发现 `\operatorname`。`git diff --check` 通过。
- GitHub MathJax 页面未实际查看，公式视觉渲染门禁仍待验；目标读者试读、外链终审及第二十一册连续复读留待全册门禁。
- 未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

第二十一册第 1—73 章已完成首轮正文修订，第 74—95 章共 22 章待精修。下一入口为第七十四章《CSA/HCA：压缩稀疏与高度压缩注意力》。全套书精修 goal 继续 active。
