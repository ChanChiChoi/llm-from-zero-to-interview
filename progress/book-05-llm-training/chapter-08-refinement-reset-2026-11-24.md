# 第8章精修交接（本轮重做）

日期：2026-11-24
正文：`book-05-llm-training/chapters/08-显存优化与数值精度.md`
研究证据：`research/memory-precision-refinement-evidence-2026-11-24.md`

## 本轮范围

依用户要求从写作规范模块 02a 开始，跳过模块 01；本章按第五册目录顺序阅读已确认契约与完整正文，忽略既有 `progress/` 记录作为章节导航和完成状态依据。正文继续围绕“训练过程中哪些显存对象同时存活、数值格式怎样改变计算路径”展开。

## 精修内容

- 在开篇明确本章两条学习线索：按时间定位活张量峰值；把数值格式放回前向、反向、累积、optimizer state 和冻结底座存储的具体位置。
- 限定 PyTorch 2.15 `expandable_segments` 为 native caching allocator 的实验性配置，解释它处理可变分配形状的机制及不减少活张量字节数的边界。
- 按 PyTorch Adam v2.15.0 源码说明 `exp_avg`/`exp_avg_sq` 的首次状态创建，并补入 2.15 Adam 文档关于 foreach tensor-list 临时量约为参数总量的额外峰值提示；把持久状态与实现临时量分开诊断。
- 精确化 FP4 All the Way v2 的 rounding 路径：权重和前向激活用 round-to-nearest，反向/更新 GEMM 的 neural gradients 与更新 GEMM activations 用 stochastic rounding；记录约 40B token 的 QAF 及 v2 摘要“最多 200B”与 §5 报告 1T token 的内部差异。
- 精确化 Quartet v4 的 QuEST/round-to-nearest 路径、8×H100 质量实验资源与 RTX 5090 局部 layer/block 测速范围，避免将局部速度结果读作分布式预训练吞吐。
- 按 OOMB v4 区分作者单 H200 4M 主张、单 H200 dense-attention 表格的 256K 范围，以及 4×H200 TP 扩展图的范围；保留 sparse attention 和普通 checkpoint API 的边界。
- 为 DDP `no_sync()` 的 forward 作用域、PyTorch AMP/compile 和 optimizer 实现的资料边界补充固定版本官方文档引用。
- 教学代码未改，历史状态标记未重写，也未运行代码。

## 验收状态

- 正文 81 个 H2/H3 标题与第8章契约逐项一致。
- 52 个 Markdown 链接目标中有 2 个本地链接，均可解析；未发现缺失目标。
- 98 个 `~~~` 围栏标记成对；围栏外未发现奇数个未转义 `$` 定界符。
- 正文与研究证据文件未发现尾随空格；正文 `git diff --check` 通过。
- 未执行教学代码、测试、训练、模型、基准或 PDF 构建。

## 下一步

从第五册第9章契约开始，按目录顺序完整阅读契约与正文，再做研究、正文精修和新交接；不按既有进度记录跳章。
