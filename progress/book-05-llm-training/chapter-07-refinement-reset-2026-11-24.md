# 第7章精修交接（本轮重做）

日期：2026-11-24
正文：`book-05-llm-training/chapters/07-分布式训练.md`
研究证据：`research/distributed-training-refinement-evidence-2026-11-24.md`

## 本轮范围

按第五册目录顺序完成第7章。本轮按用户要求忽略既有 `progress/` 记录作为导航或完成状态依据；第7章按本章契约和当前正文重新核对。正文继续沿“全局更新如何跨 rank 切分、通信、同步和恢复”的中心组织。

## 精修内容

- 第7.11节对齐已确认契约的 H2/H3 标题，将内容限定为激活重算与 FSDP、TP/CP、PP 的通信和执行调度交互；把重算机制、分段估算和论文性能证据明确留给第8章。
- 使用 PyTorch FSDP2 2.15 文档限定参数聚集和 `reshard_after_forward` 的描述，并指出重算是否引发再次 all-gather 取决于区域、wrap 边界、配置和运行时。
- 第7.13.3节区分 DCP 自身 checkpoint metadata 与应用管理的同一步提交标记，按 PyTorch DCP 2.15 明确 async staging、保存完成、全局 `.metadata`、文件创建假设及跨版本兼容边界。
- 第7.14.3节把不完整恢复点表述与 DCP 保存完成及应用层标记对齐；第7.19资料边界表也改为相同语义。
- 第7.17账本按 $DP\times TP\times PP$ 的同一混合布局重算每卡状态份额：总参数先按 TP/PP 分成每 rank 的本地模型片，再按 DP 阶段估计状态；梯度 bucket 调整到本地梯度份额可容纳的教学值，修正原账本把 6B 全模型状态误记到每张混合并行 rank 的问题。
- 资源账本按公式静态复核：每 rank 0.75B 本地参数对应 DDP 约 8.4 GiB、ZeRO-3 约 1.05 GiB；1 GiB bucket 在 DP=8 ring 模型下每 rank 发送约 1.75 GiB，按 200 GB/s 理想链路约 0.0094 秒。未运行教学 Python 片段。
- 原有 DualPipe、bubble 口径、CP/Ring Attention、DynaTrain、MoE capacity 和资源 toy 数值保持在论文/文档实际范围内；教学 Python 片段仅静态审阅，未运行。

## 验收状态

- 正文 98 个 H2/H3 标题与第7章契约逐项一致。
- 29 个 Markdown 链接目标中，4 个是本地链接且均可解析；未发现缺失目标。
- 130 个 `~~~` 围栏标记成对；围栏外未发现奇数个未转义 `$` 定界符。
- 正文、研究证据和本交接文件未发现尾随空格；正文 `git diff --check` 通过。
- 未执行教学代码、测试、训练、模型、基准或 PDF 构建。

## 下一步

本章静态验收完成后，从第8章契约开始，按目录顺序阅读契约与正文，再做研究、正文修订和新交接；不要按旧进度记录跳章。
