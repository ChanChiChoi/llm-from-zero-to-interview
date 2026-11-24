# 第9章《训练稳定性与 debug》精修交接（2026-11-24）

## 本章中心

保留“按时间轴找到首次异常对象，以可证伪假设和反事实对照确认根因”的主线。正文继续区分 loss spike、NaN/Inf、梯度幅值、数据与 mask、学习率和 optimizer state、精度、分布式同步及 checkpoint 连续性；不把某个研究机制或一次缓解措施写成通用根因。

## 本轮精修

- 将 spike 研究表中的作者和论文名改为直接链接，链接 PyTorch 2.15、训练稳定性研究及架构归一化论文的延伸阅读条目。
- 区分普通梯度下降的路径平均 Hessian 精确一步条件与 Bai 等人基于局部二次近似、预条件 Hessian 的 Adam 梯度方向曲率代理；补入论文使用的预条件 Hessian 形式，并限制其预测结论范围。
- AdaGC score 明确为此前 1,000 个值滚动均值的十标准差判据；列出论文 v3 的模型、步数与 score，并指出 Mixtral 表中“52 次/36,000 步/0.0144%”相互不一致，按计数计算为 0.1444%。保留原文表列数值并标明差异。
- 明确 AdaGC 论文的逐张量范数归约在 TP 组内进行，不能将其实现等同为任意 DP/TP/PP 配置的全局裁剪。
- 补充 PyTorch DDP `no_sync()` 对同步边界的影响，以及 forward 必须置于 context 内的限制。
- 新增研究证据与来源版本记录：[training-stability-debug-refinement-evidence-2026-11-24.md](../../research/training-stability-debug-refinement-evidence-2026-11-24.md)。

## 静态验收

- 章节 H2/H3 标题与第9章契约的 54 项逐项一致。
- Markdown 链接 41 个；未发现缺失的本地链接。
- 围栏代码块成对，未闭合围栏 0；行内数学 `$` 分隔异常 0；尾随空白 0。
- `git diff --check -- book-05-llm-training/chapters/09-训练稳定性与debug.md` 通过。
- 没有执行正文示例、测试、训练或 PDF 构建。章节中的玩具代码状态仍标注为静态审阅、未运行。

## 后续顺序

下一章按仓库章节文件顺序处理第10章《checkpoint 评估与实验管理》。本交接不从既有 `progress/` 状态推导跳章或完成情况。
