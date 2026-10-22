# 第三册第一章：PyTorch 基础实战审阅记录

## 范围

- 正文：`book-03-practical-handbook/chapters/01-pytorch基础实战.md`，覆盖 1.1—1.6。
- 配套：第三册简介、目录、`PAPERS.md` 训练论文入口。
- 环境：仓库 `.tmp/llm-book-cpu-check`，PyTorch 2.12.0 CPU build；未安装 CUDA，也未依赖 GPU。

## 来源与支持边界

本轮通过指定代理 `http://10.24.27.134:7890` 检查了下列 10 个入口，最终目标均返回 HTTP 200：

| 来源 | 支持范围 |
| --- | --- |
| [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | 计算图、梯度记录、`detach` 和 `no_grad` 的接口行为 |
| [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | logits/目标格式、类别权重、`ignore_index` 和 reduction 语义 |
| [PyTorch `logsumexp`](https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html) | 稳定 log-sum-exp API |
| [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)、[Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) 与 [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | 框架优化器参数和状态接口；稳定版页面会随版本更新 |
| [PyTorch learning-rate scheduler guide](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) | scheduler 调用顺序和状态接口 |
| [Transformers optimization schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) | Hugging Face 调度器封装语境，不支持特定模型训练效果结论 |
| [Adam](https://arxiv.org/abs/1412.6980) 与 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 原论文中的 Adam 矩估计/偏差修正与 AdamW 解耦衰减动机 |

论文入口和文档 HTTP 状态不等同于逐篇全文、附录或特定框架版本源码的完整审计。正文中的优化算法公式是解释机制的简化表达；具体默认值和边界行为应以目标 PyTorch 版本为准。

## 本地验证

- 当前正文提取出的 23 个 Python 代码块均在 CPU-only 环境中独立执行通过。
- 已将各块实际输出与正文预期逐项核对。固定种子 MLP 输出为 `[0, 0, 1, 1, 2, 2]`、loss `0.0869`、准确率 `1.0`；梯度累积示例含 5 个样本，按 4+1 两组累计并输出 2 次更新；scheduler 输出与 PyTorch 2.12.0 的记载一致。
- 随机初始化输出只绑定示例代码、版本和 CPU 环境，不作为跨版本数值保证；模型日志 demo 的随机 loss 明确为观测值，不在正文固定具体数值。
- 44 个 `math` 围栏的花括号和方括号计数平衡，代码围栏配对；章节 1.1—1.6 顺序连续，目录映射到当前章节文件正确。
- `git diff --check` 通过。PDF dry-run 未完成：环境缺少 Pandoc；未为此安装依赖。

## 修订与遗留边界

- 训练循环引导句改为完整句；梯度累积说明明确最后一个不完整组按其实际组大小归一化。
- 补充 AdamW 原论文直链，并同步到 `PAPERS.md`。
- 本地检查不覆盖 CUDA、GPU kernel、分布式训练、真实数据集、论文复现或生产训练表现；未运行外部 Markdown 数学渲染器。
