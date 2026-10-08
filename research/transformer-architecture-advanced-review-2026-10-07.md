# Transformer 架构进阶精修记录（2026-10-07）

## 范围与状态

- `chapter`：`book-02-advanced-100/chapters/02-transformer架构进阶.md`。
- `scope`：修订前 1,610 行，覆盖第 11—25 讲；当前 1,766 行。文末“架构选择的统一评估”是章节总结，不另计讲次。
- `evidence_level`：`paper`、`local-demo`、`inference`。
- `lifecycle`：`active`。位置缩放、稀疏路由和 serving 的具体收益继续绑定对应论文、模型、实现或评测设置。

## 主要修订

- 更正 32K 到 1M 的 attention 交互数倍率，并明确十进制与二进制 token 单位。
- 将 Encoder–Decoder 的 aligned 公式补齐分行符；为 RoPE 补上一个常见的频率定义并区分频率基数与位置角度。
- 明确 FlashAttention 是精确 dense attention 的 IO/存储优化；区分稀疏连接、线性近似与 kernel 优化。
- 为 Attention、Norm、FFN、RoPE、ALiBi、PagedAttention、MoE、SSM 等主张补上就近来源与适用边界。
- 将 MoE top-k combine 权重写为 router 概率，并说明实现可选择重新归一化；把辅助均衡损失限定为 Switch top-1 形式，明确 token 数、路由组范围及 top-k 实现差异；限定 capacity 公式为单 token 单 expert 的估算，并澄清 router z-loss 约束的量。
- 修正 SwiGLU 参数量 demo：attention 的四个投影与 MLP 的两/三矩阵分别计数；修复原示例低估 SwiGLU block 参数的问题。
- 修复第 21 讲公式分行和第 26 讲编号冲突：目录分配给本章第 11—25 讲，下一章从第 26 讲开始，因此本章末尾改为不编号的总结。
- 为教学代码补上输入/输出契约、有限值和形状检查、bool-as-int、KV head 整除、hidden/head 整除及数值溢出处理；为主要 demo 写入可复核的实际输出。

## 来源和证据范围

本章原有的 25 个 arXiv 来源 ID 已由 arXiv Atom API 核对题名；新增的 8 个来源为：

1. [Layer Normalization](https://arxiv.org/abs/1607.06450)：逐样本的层归一化定义。
2. [Transformers without Tears](https://arxiv.org/abs/1910.05895)：Pre-Norm、ScaleNorm；实验限定于论文的翻译任务和设置。
3. [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)：GeLU 定义和早期多任务实验范围。
4. [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)：特定模型上的 FFN 解释性分析，不推出事实由单神经元存储。
5. [ST-MoE](https://arxiv.org/abs/2202.08906)：稀疏专家训练稳定性与 router z-loss。
6. [PagedAttention](https://arxiv.org/abs/2309.06180)：KV block 管理及 vLLM 论文评测；论文吞吐数字不迁移到本地或其他硬件。
7. [Position Interpolation](https://arxiv.org/abs/2306.15595)：RoPE 上下文扩展及论文中的模型、微调预算和任务。
8. [YaRN](https://arxiv.org/abs/2309.00071)：RoPE 扩展方法及原论文评测范围。

另外核对了 Longformer、BigBird、Performer、FlashAttention/FlashAttention-2、Xiong 等人的 Pre-LN 分析、DeepNet、RMSNorm、GLU、RoFormer、ALiBi、MQA、GQA、Shazeer MoE、GShard、Switch、S4、Mamba、Mamba-2/SSD、BERT、GPT、T5、LLaMA 和 Chinchilla 的题名元数据。arXiv feed 的分批摘要查询中，一批返回 503；其他成功批次提供的摘要用于检查稀疏注意力、IO-aware attention、Pre-LN、DeepNorm、Performer、SSM、PI/YaRN 等概念范围。最后一次 503 不作为论文链接失效证据。

33 个唯一 arXiv ID 均通过元数据 feed 命中；GPT 论文 PDF 入口直接返回 HTTP 200。论文证据只支持论文定义与指定实验，不能证明本地代码、生产 serving、目标硬件速度或任意模型的质量收益。对没有本地目标模型、GPU 和 serving stack 的主张，正文采用条件性表述并给出验证指标。

## 本地验证

- `code_status`：15 个标准库 Python 代码块逐块独立执行，Python 3.12.9，正常输出均与正文一致。另执行 18 项非法输入/溢出检查和 2 项数值不变量检查，覆盖 NaN/Inf、ragged logits、布尔尺寸、非整除 KV heads、Norm 溢出、MoE capacity、mask 长度、状态递推和配置维度等边界。未运行 GPU kernel、真实模型、论文复现或生产服务。
- `formula_status`：53 个 `math` 围栏；花括号、环境和围栏配对检查通过，未发现 `\operatorname`、`\text{..._}` 或尖括号下标写法。复核了 attention 复杂度、KV/score 显存、LayerNorm/RMSNorm、SwiGLU 参数量、RoPE 旋转恒等式、MoE 均衡与 z-loss、容量和 SSM 状态关系。未使用外部 Markdown 数学渲染器。
- `numeric_examples`：attention 最后位置路由 `[0.2308, 0.4063, 0.3629]`、输出 `[3.3965, 6.6035]`；32K 单层 32-head BF16 dense score 上界 64 GiB、示例 KV Cache 4 GiB；32-head/8-KV-head/8192/32-layer BF16 KV Cache 1024 MiB；GeLU 与 SwiGLU block stack 估值分别约 6.442B 和 6.476B；RoPE 二维绝对/相对点积一致。
- `navigation`：第二册目录仍将本章列为第 11—25 讲，下一章第 26—40 讲；对应文件存在，讲次没有重复。知识图谱、百科、术语表和练习已覆盖 attention、Norm、RoPE、GQA/MQA、MoE、SSM、PagedAttention 等主题；本轮只向 `PAPERS.md` 增补新的论文入口，没有重复抄写题目或百科解释。
- `format_status`：`git diff --check` 通过。

## 未验证项

- 未在 CUDA/GPU、真实 Transformer checkpoint 或目标 serving engine 上验证速度、显存峰值、并行通信、长上下文质量和 MoE 尾延迟；所有代码只展示局部机制和数量关系。
- arXiv API 有一批摘要请求返回 HTTP 503；已有成功元数据查询确认来源 ID 与题名，正文将未复现的论文结果保留在原论文范围内。
- 未用 Pandoc/LaTeX 构建 PDF，也未使用外部 Markdown 数学渲染器。本章公式通过源级配对与 GitHub MathJax 写法检查。

## 结果

`source-backed`、`verified`、`local-demo`。第二册第二章完成；全书精修仍在进行，下一份正文为 `book-02-advanced-100/chapters/03-tokenization数据与预训练进阶.md`。
