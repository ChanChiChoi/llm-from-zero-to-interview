# 第5章 预训练目标与训练循环：本轮从头复核与精修

## 范围与依据

- 按用户本轮要求，不用既有 `progress/` 记录决定阅读顺序或章节状态；从第4章之后按目录顺序进入第5章。
- 完整读取第5章章节契约和当前正文。契约与正文的 H2/H3 标题序列均为 59 项，标题文字、级别和顺序逐项一致。
- 本章中心保持不变：跟踪一个 batch 从 causal next-token 目标、label/mask 与有效 loss 分母，经过反向、累积、梯度处理和 optimizer update，进入验证、日志与可恢复 checkpoint；其他预训练目标作为比较支线。

## 正文精修

- 在 5.7.3 保留有效 token 加权公式、$2$ 与 $4$ 个目标的窗口算例、DDP 全局分母及尾窗口处理；删除重复的二维梯度演算和第二组数值裁剪演算，集中说明逐 micro-batch 裁剪不能与全窗加权互换。
- 将 5.13 的教学代码收束到 packing、外部 label shift、有效目标过滤、简化 logits 梯度、变长累积与一次 SGD 更新；删减与“最小训练步”无关的通用输入形状/数值检查，保留必要的空目标和输入边界检查。示例对 attention mask 与 causal mask 只打印、不参与模型计算的边界没有改变。
- 精简 toy 代码后的重复解释，分开交代跨文档 `EOS → next document` 语义、token-to-logits 模型限制和断言/容差边界；明确代码与断言仍未运行。
- 为 BERT-style 双向上下文补入原始论文链接；将 5.7.3 Accelerate 变长 batch 的引用固定到 v1.15.0。公式语义未改写。

## 研究与证据

- 本轮来源、版本和外推边界记录在 `research/pretraining-objective-training-loop-evidence-2026-11-24.md`。
- 核对 PyTorch CrossEntropyLoss 与 AMP 文档、Accelerate v1.15.0 变长 batch 示例、Transformers 当前 causal shift 计数说明，以及 BERT、T5、UL2、FIM、两类 MTP 实现和 masked diffusion 的原始论文。
- 论文分数、训练设置及框架缩放规则均绑定其具体模型、版本和实现；没有将 token-level 平均、FIM、UL2、MTP 或 diffusion 结果推广为通用配方。

## 静态验收

- 契约与正文 H2/H3 标题序列：59 项，文字、级别、顺序一致。
- Markdown 链接共 34 个，其中本地链接 3 个，缺失目标 0。
- 36 组 fenced blocks 的起止标记成对；简易行级 `$` 定界符检查未发现奇数未配对行。
- `git diff --check -- book-05-llm-training/chapters/05-预训练目标与训练循环.md` 通过。
- 未运行章内 toy 代码、tests、benchmark、模型、训练或 PDF 构建；公式只作源码和来源核对，未逐式在目标阅读平台渲染。

## 交接

- 下一步按顺序进入第6章：完整读取有效章节契约与正文，再研究优化器和学习率策略；不参考历史进度记录决定阅读顺序。
