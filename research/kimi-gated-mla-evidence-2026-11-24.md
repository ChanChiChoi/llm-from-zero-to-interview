# Kimi Gated MLA 研究记录

日期：2026-11-24

## 检索范围与来源

- 检索词：`Kimi K3 Gated MLA equation output gate`、`DeepSeek MLA low-rank KV joint compression cache`、`Kimi-K3 config kv_lora_rank mla_use_nope`。
- 一手来源：[Kimi K3 arXiv v2](https://arxiv.org/abs/2607.24653v2)（2026-07-27 提交，2026-08-07 修订）；阅读 §2.1、§2.1.2 Eq. 7。固定 HTML：[v2 正文](https://arxiv.org/html/2607.24653v2)。
- 一手来源：[DeepSeek-V2 arXiv v5](https://arxiv.org/abs/2405.04434v5)（2024-06-19 当前版本）；阅读 §2.1.2 Eqs. 9–13、§2.1.3 Eqs. 14–19 和 KV cache 对照段。固定 HTML：[v5 正文](https://arxiv.org/html/2405.04434v5)。
- 官方模型资料：[MoonshotAI/Kimi-K3 仓库](https://github.com/MoonshotAI/Kimi-K3)；官方组织下的 [固定 config revision `f831ab66814297da540d832a5235f8e904f29d06`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)。记录 `kv_lora_rank=512`、`mla_use_nope=true`、`mla_use_output_gate=true`、`num_hidden_layers=93`、`num_attention_heads=96`，以及包含 24 个索引的 `full_attn_layers` 和 `kda_layers` 列表。

## 可支持的技术事实

- DeepSeek-V2 MLA 对输入 hidden state 做 KV down-projection，形成每 token 的共享低维 `c_t^KV`；key/value 经各自上投影得到。论文说明推理时缓存 `c_t^KV`，并给出 key 上投影可折入 query projection、value 上投影可折入 output projection 的代数吸收视图。
- DeepSeek-V2 将 RoPE 与低秩 KV latent 解耦：另有逐位置的共享 RoPE key 与 query RoPE 分量。因此该版本的 cache 账本不能只计 `d_c` 个 latent 元素。
- Kimi K3 v2 §2.1.2 将 Gated MLA 定义为 MLA 加输入依赖、逐通道的 full-rank 输出门：`y_t = W_o[Sigmoid(W_g x_t) ⊙ õ_t]`。门调制的是未门控 attention output。
- Kimi K3 v2 明确说明其 MLA 层采用 NoPE，不对 query/key 施加显式位置编码；报告把位置敏感和 recency-aware mixing 归于其间交错的 KDA 层，并把 MLA 层解释为全局内容交互。此为 Kimi K3 的模型设计，不是 MLA 通用定义。
- 固定 HF config 披露 `kv_lora_rank=512`、`mla_use_nope=true`、`mla_use_output_gate=true`，并列出 93 层中的 KDA 和 full-attention/MLA 层索引。官方仓库模型摘要披露 69 KDA + 24 Gated MLA；整套 K3 层间模式留给第 65、88 章。

## 证据边界

- Kimi K3 报告和 config 支持模块公式、NoPE 设置及具体版本参数；它们不足以单独说明 output gate 对最终任务成绩或吞吐的因果贡献。本轮不以整模型结果归因给 gate。
- DeepSeek-V2 的 cache 百分比与吞吐数字是论文报告的模型级对比，未作为 Gated MLA 输出门的效果证据。
- 逐 token payload 公式是按公开张量形状计算的理想账本；不含对齐、页元数据、临时 workspace 和实际 runtime cache dtype，不是实测 serving memory。
- 未运行本地 forward、实现复现、kernel benchmark 或用户试读；Kimi K3 的 gate kernel/state layout 以公开资料实际披露为限。
