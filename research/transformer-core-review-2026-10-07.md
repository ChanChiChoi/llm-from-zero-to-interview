# 第一册第二份正文：Transformer 核心资料核验

核验日期：2026-10-07。用途：精修 `book-01-core-30/chapters/02-transformer核心.md`，不是目标模型配置或生产 benchmark 报告。

## 资料与可支持范围

| 来源 | 等级 | 本轮用于支持的范围 | 边界 |
| --- | --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | `paper` | scaled dot-product attention、`sqrt(d_k)` 缩放、多头结构、正弦位置编码、dense attention 的计算关系 | 原始实验不是现代 decoder-only LLM 的完整配置 |
| [BERT](https://arxiv.org/abs/1810.04805) | `paper` | BERT 使用 WordPiece 与双向 masked language modeling 的实例 | 不据此推断所有 WordPiece tokenizer 的训练/打分实现相同 |
| [SentencePiece](https://arxiv.org/abs/1808.06226) | `paper` | 直接处理原始文本的 tokenizer 工具；BPE、Unigram 等模型与空格边界处理 | 工具接口不等同于单一切分算法 |
| [RoFormer](https://arxiv.org/abs/2104.09864) | `paper` | RoPE 对 Q/K 的旋转以及相对位移性质 | 相对位置性质不保证任意长度外推 |
| [Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409) | `paper` | ALiBi 在 attention 分数中加入距离相关线性 bias | 长度泛化结论限于论文设置 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | `paper` | RMSNorm 的公式、提出动机和论文实验 | 不据此声称 RMSNorm 对所有模型都优于 LayerNorm |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | `paper` | Pre-LN/Post-LN 的梯度路径和训练稳定性分析 | 结论受论文架构与训练设置限制 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | `paper` | IO-aware 分块实现、保留精确 attention 数学目标、减少中间矩阵读写 | 实际速度取决于硬件、shape、dtype 和 kernel |
| [GQA](https://arxiv.org/abs/2305.13245) | `paper` | K/V head 共享与 MHA/MQA/GQA 结构、论文评测 | 不推导所有模型的质量或吞吐收益 |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | `paper` | 长上下文任务可能受证据位置影响的研究实例 | 不外推为所有模型和任务上的固定曲线 |
| [PyTorch `MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) | `official` | 张量 shape、`batch_first` 与布尔 `attn_mask=True` 表示屏蔽的 API 语义 | 稳定版文档会随 PyTorch 发布更新；本轮本地运行使用 2.12.0 CPU build |
| [SentencePiece 官方仓库](https://github.com/google/sentencepiece) | `official` | tokenizer 工具实现和模型入口 | 不把 README 介绍当作目标模型的训练配方 |

以上 arXiv 页面与 PyTorch 文档在 2026-10-07 的 HTTP 状态为 200；SentencePiece 官方仓库为 200。Hugging Face Tokenizers 文档本次无法连接，TensorFlow Text 文档返回代理 503，因此正文不引用这些页面来断言具体 WordPiece merge score；措辞限定为各实现的词表构造、评分和编码规则不同。

## 本地教学计算

- 手算单 query、两个 key 的 attention：`q=(1,0)`，`K=[(1,0),(0,1)]`，`V=[(2,0),(0,4)]`，`d_k=2`；分数约 `[0.7071,0]`，权重 `[0.6698,0.3302]`，输出约 `(1.3395,1.3208)`。
- KV Cache 估算：`L=32,T=8192,d_h=128,b=2` 字节、32 个 query heads 时，MHA/GQA(8 K/V heads)/MQA 的单序列原始 K/V 载荷分别为 4 GiB、1 GiB、128 MiB。结果是公式计算，不含分页对齐、元数据和 workspace。
- LayerNorm/RMSNorm：输入 `[1,2,3,10]`，总体方差下 LayerNorm 均值为 4、标准差约 3.5355；RMSNorm 分母约 5.3385。对应向量已写入正文，说明两者中心化行为不同。
- RoPE：二维单位向量、每步相位 0.25 rad，点积为 `cos((j-i)·0.25)`；相同相对位移的分数相同。该例不代表多头模型的完整分布。

以上数值是由定义直接计算的 `local-demo`，不构成论文复现、模型评测或服务性能声明。

## 本地代码验证

临时环境：Python 3.13.13、NumPy 2.4.6、PyTorch 2.12.0+cpu。正文 22 个 Python 代码块全部通过语法编译；21 个独立运行，训练 loop 按正文说明与前一块的 MiniGPT 定义在同一进程执行。共通过 15 个代表性边界检查，覆盖负/零位置维度、attention shape 与 NaN、空序列、非法 head 数、全屏蔽 mask、超出 block 长度、RMSNorm 非法 epsilon/shape/NaN、RoPE 奇数维、MiniGPT 越界 token/target 和负生成长度。

这些结果验证了 CPU 教学路径。未测试 CUDA kernel、GPU 数值差异、生产 serving、分布式训练或不同 PyTorch 版本。
