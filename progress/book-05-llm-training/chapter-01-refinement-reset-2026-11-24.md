# 第一章精修记录（重启流程，2026-11-24）

## 范围与依据

- 本轮按书册正文及章节契约核对第一章，不使用旧 progress 记录决定审阅范围或结论。
- 对照契约 写作规范/全套书不同册的章节大纲结构/book-05-llm-training/01-训练总览.md，保持“将目标、数据、更新、系统和评估接成可追溯训练链”的中心。
- 与近期 scaling-law 研究相关的原始证据和限制见 research/training-overview-scaling-evidence-2026-11-24.md。

## 本次修改

1. 修正 §1.6 一次更新的 token 估算口径：公式改用 data-parallel world size $W_{\mathrm{DP}}$，解释 TP/PP rank 参与同一数据副本的模型计算，不应把 GPU 总数直接乘成样本副本数。
2. 明确 §1.6 与 §1.17 教学算例假设 8 张卡全部用于数据并行，以便 batch、全局吞吐和更新数三处口径一致。

## 研究与边界

- 原论文核实：Chinchilla 用 70B 参数和 1.4T tokens，在其设定下与 Gopher 训练计算预算相当；数据受限与 inference-aware scaling 分别改变重复数据价值及训练/推理生命周期成本的最优分配条件。
- Scaling-law 综述列出的 51 篇研究中，45 篇量化 power-law 关系；其中训练设置、公式变量、评估和拟合方式披露不足会影响复现和外推。
- ICLR 2026 两项研究分别指出生成式评估的 pass@k 会改变拟合与预测，文本质量干预会改变 scaling-law 系数/指数和 compute-optimal token/parameter 选择。结论限定于各论文的模型、语料和评测范围。
- 本章算例仍是教学估算；没有进行训练、评测、benchmark 或集群测量。

## 静态验收

- H2/H3 契约结构：80 项，正文 80 项，缺失 0、额外 0。
- Markdown 链接 42 个，本地目标缺失 0。
- ~~~ 围栏 46 个，全部成对；无反引号代码围栏。
- git diff --check -- book-05-llm-training/chapters/01-训练总览.md 通过；研究记录无尾随空白。
- 未运行任何正文示例、训练代码、benchmark、测试套件或 PDF 构建。

## 交接

第一章正文、研究证据和本次精修记录分别位于 book-05-llm-training/chapters/01-训练总览.md、research/training-overview-scaling-evidence-2026-11-24.md 和本记录中。后续按章节契约继续第2章，不依赖旧进度记录推断其完成状态。
