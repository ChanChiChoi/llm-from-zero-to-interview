# Transformer 组件的 PyTorch API 语义

检索与阅读日期：2026-10-29。本记录保存与本章示例版本对应的官方接口语义；逐章审阅、修改决定和代码状态记录在 `progress/book-03-practical-handbook/architecture-audit.md`。

## PyTorch 2.12 API

- **来源与版本**：PyTorch 2.12 官方文档：[Embedding](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Embedding.html)、[scaled_dot_product_attention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)、[MultiheadAttention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html)、[LayerNorm](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html)、[Dropout](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Dropout.html)。
- **阅读位置**：各页的参数、输入/输出 shape、mask 语义、归一化统计量、dropout 行为与 SDPA 后端说明。
- **来源支持的事实**：`Embedding.padding_idx` 对应行不贡献梯度；新建模块默认将该行初始化为零。SDPA 的布尔 `attn_mask` 中 True 表示参与 attention，浮点 mask 加到 score；它总是按显式 `dropout_p` 应用 dropout，评估时需传 0；设置 `is_causal=True` 时不能同时传 `attn_mask`。`MultiheadAttention.attn_mask` 中 True 表示禁止该位置，`key_padding_mask` 中 True 表示忽略该 key。`LayerNorm` 在 `normalized_shape` 对应的末尾维度上统计，方差使用有偏估计（`correction=0`），训练与评估都使用当前输入统计量；`Dropout` 在评估模式下是恒等映射。
- **证据等级与边界**：source-backed（PyTorch 2.12 官方 API 文档）。这些结论只描述文档所对应版本的接口；它们不证明某个融合后端在指定硬件上会被选中，也不能替代代码在其他版本、设备和 dtype 下的运行验证。
