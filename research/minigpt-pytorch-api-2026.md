# PyTorch 2.12 小型语言模型 API 证据

## `torch.nn.functional.cross_entropy`

- 来源：[PyTorch 2.12 `cross_entropy` 文档](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.cross_entropy.html)，2026-10-28 阅读。
- 文档将输入说明为未归一化 logits；类别索引 target 的形状须与各样本/空间位置对应。对序列分类可将 `[B,T,V]` 重排为 `[B×T,V]`，并将目标 `[B,T]` 重排为 `[B×T]`。
- 文档列出的默认 `reduction` 是 `mean`，将逐位置损失取平均。对本例所有 target 都是有效类别索引，因此分母是 `B×T`。
- 证据边界：该文档支持 PyTorch 2.12 的接口与张量约定，不证明仓库脚本已运行，也不支持任何训练效果结论。

## `torch.multinomial`

- 来源：[PyTorch 2.12 `multinomial` 文档](https://docs.pytorch.org/docs/2.12/generated/torch.multinomial.html)，2026-10-28 阅读。
- 输入可按行提供抽样分布或权重；每行必须为有限、非负且总和非零的值。矩阵输入有 `M` 行时，抽取 `num_samples` 个索引会返回 `[M,num_samples]`；`replacement` 默认为 `False`。
- 将 softmax 概率 `[B,V]` 与 `num_samples=1` 结合，输出形状为 `[B,1]`。默认无放回在单次抽一个类别时不改变形状；后续生成步是新的 API 调用，因此不能据此禁止同一类别在不同步重复。此项是依据接口语义对示例调用的推论。
- 证据边界：API 文档不保证跨设备、版本或运行环境的精确随机序列，也不评价采样文本质量。
