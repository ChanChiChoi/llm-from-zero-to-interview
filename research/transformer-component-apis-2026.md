# Transformer 组件的 PyTorch API 语义

主要检索与阅读日期：2026-10-29；非方形 causal mask 专项复核：2026-10-30。本记录仅保存列明版本的官方接口语义与适用边界。

## PyTorch 2.12 API

- **来源与版本**：PyTorch 2.12 官方文档：[Embedding](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Embedding.html)、[scaled_dot_product_attention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)、[MultiheadAttention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html)、[LayerNorm](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html)、[Dropout](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Dropout.html)。
- **阅读位置**：各页的参数、输入/输出 shape、mask 语义、归一化统计量、dropout 行为与 SDPA 后端说明。
- **来源支持的事实**：`Embedding.padding_idx` 对应行不贡献梯度；新建模块默认将该行初始化为零。SDPA 的布尔 `attn_mask` 中 True 表示参与 attention，浮点 mask 加到 score；它总是按显式 `dropout_p` 应用 dropout，评估时需传 0；设置 `is_causal=True` 时不能同时传 `attn_mask`。`MultiheadAttention.attn_mask` 中 True 表示禁止该位置，`key_padding_mask` 中 True 表示忽略该 key。`LayerNorm` 在 `normalized_shape` 对应的末尾维度上统计，方差使用有偏估计（`correction=0`），训练与评估都使用当前输入统计量；`Dropout` 在评估模式下是恒等映射。
- **证据等级与边界**：source-backed（PyTorch 2.12 官方 API 文档）。这些结论只描述文档所对应版本的接口；它们不证明某个融合后端在指定硬件上会被选中，也不能替代代码在其他版本、设备和 dtype 下的运行验证。

## PyTorch 2.12 的非方形 causal mask

- **来源与版本**：[scaled_dot_product_attention 官方文档](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)，PyTorch 2.12。
- **阅读位置**：参数说明中 `is_causal` 的非方形 mask 对齐规则，以及 `attn_mask` 与 `is_causal` 的互斥约束。
- **来源支持的事实**：当 query 长度 L 与 key 长度 S 不同，`is_causal=True` 使用左上对齐的 causal bias；布尔 `attn_mask` 中 True 表示该位置参与 attention；`attn_mask` 与 `is_causal=True` 不能同时传入。带 KV cache 的 query 具有非零绝对位置，其可见关系通常应根据 query/key 绝对位置显式构造，满足 `key_position <= query_position`。
- **证据等级与边界**：官方文档直接支持接口对齐与互斥规则；绝对位置关系是因果解码约束的推论。此记录不推断其他 PyTorch 版本、第三方 attention API 或任一融合 kernel 的行为。

## PyTorch 2.15 API 复核

复核日期：2026-11-11。

- **来源与版本**：PyTorch 2.15 官方文档：[Embedding](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html)、[scaled_dot_product_attention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)、[causal_lower_right](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.bias.causal_lower_right.html)、[MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html)、[RMSNorm](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RMSNorm.html)、[LayerNorm](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html)。
- **阅读位置**：`Embedding.padding_idx` 的参数说明；SDPA `attn_mask`、`is_causal`、`dropout_p` 与 backend/numerical accuracy 边界；causal lower-right bias 的定义与 prototype 标记；MultiheadAttention 的 mask 语义；RMSNorm epsilon 默认值；LayerNorm 方差和统计维度。
- **来源支持的事实**：SDPA 的 bool mask 中 True 表示该元素参与 attention，而 `MultiheadAttention` 的相关 mask 中 True 表示屏蔽/忽略；SDPA 的 `dropout_p` 不跟随 module `.eval()` 自动变为 0；SDPA 的 `is_causal=True` 在非方形矩阵上使用左上对齐 causal bias，且不能同时传入显式 `attn_mask`。`causal_lower_right(L,S)` 按右下对角线对齐，等价 materialized bool mask 的对角偏移为 `S-L`，但该 bias API 标为 prototype。`Embedding.padding_idx` 行不从 lookup 梯度更新，新建层该行默认初始化为零。`RMSNorm(eps=None)` 使用 computation/opmath dtype 的 machine epsilon；fp16、bf16、fp32 输入用 fp32 epsilon，fp64 输入用 fp64 epsilon。`LayerNorm` 沿 `normalized_shape` 的末尾维度统计，使用 `correction=0` 的有偏方差，并在 train/eval 都用当前输入统计量。SDPA 不同后端可能产生略有差异的浮点结果，后端自动选择不能单独证明目标硬件上的速度收益。
- **证据等级与边界**：source-backed（固定版本官方 API 文档）。cached query 的绝对位置可见关系 `key_position <= query_position` 是因果生成定义的推论，不是 API 对所有 cache 布局的自动推断；lower-right bias 的 prototype 稳定性/后端支持须按实际运行版本和设备重新核验。以上文档语义不替代本地 PyTorch 2.15、CUDA 或融合后端运行验证。
