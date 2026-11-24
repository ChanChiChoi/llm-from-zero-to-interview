# 第三十三章精修审计

审阅日期：2026-11-23。正文主题重写、完整章节大纲、研究记录、目录和读者配套同步已完成；公式源码、容量口径和本地链接静态复核通过。GitHub MathJax 与 Mermaid 实际页面渲染尚未验收；公式及图示门禁仍待完成。目标读者试读、外链最终逐条复核和第二十一册连续复读留待全书门禁。全套书精修 goal 继续 active；不更新 PDF 或 plan.md。

## 旧稿诊断与结构决定

旧稿有 26 个 H2，多次重复 scan 算法、training/decode 区别、state 固定容量、roofline、状态迁移、benchmark 与生产准入门槛；还放入使用任意权重和成本单位的 toy Python 比较，代码没有对应真实 Mamba kernel 或可解释的硬件测量。大量状态调度、迁移和服务指标属于第六册与第二十四册的主讲范围，通用能力和架构优劣证据属于第三十四、三十六章。

依据第二十一册总纲、第 32 章 Selective Scan 机制和后续章节分工，将内容收束为五节：理论复杂度与训练/prefill/decode 的测量对象；训练硬件感知扫描；反向阶段重计算和训练显存；固定状态与随长度增长的 KV cache；原论文算子、端到端推理、训练显存三类实验及适用边界。保留章名、文件路径和目录位置，不保留无法支撑性能结论的 toy code 或服务准入流程。

## 研究与证据

问题、检索日期、论文版本、读取位置和来源边界见[Mamba 硬件感知扫描与效率证据研究记录](../../research/mamba-hardware-aware-scan-and-benchmarks-2026-11-23.md)。以 Gu 与 Dao 的 Mamba 原论文 arXiv v2 `3.3.2、Appendix D、`4.5 和 Appendix E.5 核对 $(B,L,D,N)$ 状态展开、融合/scan/SRAM 路径、反向重计算、A100 scan 微基准、端到端 batch decode 和完整训练显存。FlashAttention 原论文 v2 只用于说明 exact attention 的 IO-aware tiling；Stanford CS277 和 DAU AIDT Lab 讲义只用于检查内存层级与实验分层的教学表达，不作为性能证据。正文数字均保留硬件、模型、batch、长度、精度和 baseline 条件；未从论文结果推导当前服务或跨 GPU 的普遍速度优势。

## 正文决定与同步

正文区分 scan 工作量随序列长度线性增长与墙钟时间；解释融合如何在 SRAM 内完成离散化、并行扫描和读出，反向重计算怎样减少扩展状态的 HBM 存取。推理部分用带变量定义的公式比较 KV 与 per-layer SSM/卷积状态容量随上下文长度的变化，并保留权重读取、内层宽度和状态维度等因素。实验表分离 scan 算子、端到端推理和完整训练显存，并将论文的 A100 80GB PCIe、BF16、batch/长度及 Hugging Face baseline 写在各自行中。

同步第二十一册目录的大纲入口、练习 31AL、面试题 31AI、PAPERS.md 的 Mamba 原论文证据边界和 KNOWLEDGE_GRAPH.md 的硬件路径。HBM、SRAM、Fused Kernel、Activation Checkpointing、FlashAttention 和 Arithmetic Intensity 等概念已有术语条目，无需新增重复定义。

## 静态检查与遗留门禁

- 正文 H2 为 33.1—33.5，与[第三十三章结构大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/33-Mamba训练推理与硬件效率.md)标题和顺序一致。
- 正文包含容量公式块、1 个 Mermaid 数据路径图和 1 张三类 benchmark 表；复核 KV/state 容量依赖、状态尺寸口径、论文测量设置与 baseline 边界。MathJax 源码静态检查通过；GitHub MathJax/Mermaid 实际视觉渲染未验收。
- 正文、目录、研究记录、审计、练习、面试和相邻章节链接目标均已检查；练习 31AL、面试 31AI 唯一。无可执行代码，未运行模型、kernel、GPU benchmark 或测试。
- `git diff --check` 通过。目标读者试读、外链最终逐条复核、GitHub 渲染及第二十一册连续复读待后续门禁；不更新 PDF 或 plan.md。

第二十一册第 1—33 章已完成首轮正文审阅与修订；第 34—95 章共 62 章待精修。下一入口为第三十四章《Mamba 的局限：精确检索、ICL、规模验证与生态》。
