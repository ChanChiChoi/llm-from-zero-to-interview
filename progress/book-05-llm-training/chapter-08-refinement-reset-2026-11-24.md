# 第五册第 8 章精修记录（2026-11-24）

## 本轮范围与依据

本轮按用户指定从写作规范模块 02a 起检查，并先完整阅读第五册第 8 章契约，再从第一行到最后一行顺读正文。本记录只使用本轮正文、章节契约和新核验的来源；不读取或沿用既有 `progress/` 记录。章节契约 8.0–8.19 的主题及小节次序均在正文中有对应展开。

## 新核验发现

- 正文主要账本、格式数值、online softmax、QLoRA、激活重算和教学 toy 的量级关系总体一致，论文结果已有明确配置边界。
- 正文仍引用 PyTorch v2.14 Adam 实现说明首次有梯度时懒初始化 m/v；本轮核对的 v2.15.0 源码仍支持该行为，可统一到 2.15 固定源码。
- Transformer Engine 引用 2.20.2 primer；当前官方 primer 为 2.21.0。通用 NVFP4 的 1D 16 元素 scale 应与 TE 训练线性权重采用的 16×16 2D scale 区分，避免把一种训练 recipe 写成通用格式规则。
- 8.3.2 一处变量下标排版为 `$M_{\\mathrm{other}} live$`，应修正为清晰的“other live”下标。
- OOMB 的 4M-token、Qwen2.5-7B、单 H200 结果应继续限定为作者系统和设置；需明确其 O(1) 指激活内存，端到端显存仍随分页 KV 状态增长，稀疏 attention 是近似路径，4 H200 TP 图表不是单卡结果。
- 其余主要技术主张按 PyTorch 2.15 文档、Transformer Engine 2.21.0、QLoRA/8-bit optimizer/FlashAttention/FP4/OOMB 原始论文核对；这些资料的证据与边界另记入 research 文件。

## 修改与验收

- 将第一次 Adam step 的状态懒初始化描述和源码链接统一到 PyTorch v2.15.0，并补明首步 OOM 也可能来自 foreach/fused 临时空间，需以分配时间线区分。
- 修正 8.3.2 变量下标排版错误。
- 将 Transformer Engine primer 的版本引用更新至 2.21.0，并补充通用 NVFP4 1D scale 与 TE 线性权重 16×16 二维 scale 的区别。
- 收紧 OOMB 结果边界：区分单 H200 的作者报告、单卡表格的 256K 测量和 4×H200 TP 的 4M–8M 扩展图；明确 O(1) 指 activation memory，稀疏 attention 属近似路径。
- 新增对应来源证据文件：`research/memory-optimization-and-precision-evidence-2026-11-24.md`。

静态验收：`git diff --check -- book-05-llm-training/chapters/08-显存优化与数值精度.md` 通过；69 个章节 heading ID 与契约顺序相符；Markdown 代码围栏配对；本地链接均可解析；正文无遗留的 PyTorch 2.14 Adam 或 Transformer Engine 2.20.2 引用。第 8.15 节 toy 保留原有 2026-10-26 验证标签，本轮未执行代码、测试或 PDF 构建。

## 后续顺序

完成第 8 章后按正文顺序进入第 9 章；不根据既有 `progress/` 记录跳章。
