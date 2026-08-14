# 第十八章：RoPE、NTK Scaling、YaRN 与长上下文位置外推

## 18.1 位置编码解决的不是“给 token 编号”

self-attention 本身对输入排列缺少顺序感。如果不加入位置信息，`A B` 和 `B A` 可能共享同一种内容集合表示。位置编码提供顺序线索，但不同方法对相对距离、外推长度、局部性和实现成本的偏好不同。

RoPE 的重要性在于，它不是简单把位置向量加到 hidden 上，而是对 query/key 做位置相关旋转，使注意力点积显式携带相对位移。现代长上下文方案又会调整位置频率、插值策略、训练长度或 attention 结构。把所有方法统称为“RoPE scaling”会掩盖它们的差异。

## 18.2 RoPE 的二维旋转

对 head 向量的每两个维度组成一个二维平面。位置 `p` 的旋转矩阵为：

~~~math
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
~~~

第 `i` 个频率的角度为：

~~~math
\theta_{p,i}=p\,\omega_i
~~~

于是：

~~~math
q'_p=R(p\omega_i)q_p,\qquad k'_r=R(r\omega_i)k_r
~~~

旋转的内积包含 `((p-r)\omega_i)` 的相对位置项。高频维度更敏感于近邻变化，低频维度提供更长周期，但超出训练范围后会遇到相位外推和混叠问题。

## 18.3 最小实现与 shape

~~~python
import torch


def apply_rope(x, position_ids, base=10_000.0):
    # x: [batch, heads, seq, head_dim], head_dim 必须为偶数
    b, h, t, d = x.shape
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device) / d))
    angles = position_ids[:, None, :, None] * inv_freq[None, None, None, :]
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


x = torch.randn(2, 4, 8, 16)
positions = torch.arange(8).repeat(2, 1)
print(apply_rope(x, positions).shape)
~~~

输出仍为 [2, 4, 8, 16]。工程代码要核对 position dtype、batch 广播、interleaved/half-split 排布、KV cache 追加位置和长位置精度。一个“能运行”的 RoPE 可能因为维度排列不同而与 checkpoint 不兼容。

## 18.4 为什么短训长测会失败

假设训练最大长度为 `L_train`，推理长度为 `L_test>L_train`。直接把 position id 递增到 `L_test` 会产生训练中没有出现的角度。常见现象包括 perplexity 上升、远处 token 相关性下降、重复、局部注意力异常和中间证据丢失。

注意，修改位置公式只能解决坐标范围的一部分问题。模型还要在长样本上学习跨段任务、正确使用长距离证据，并让训练时的 attention mask、position reset 和推理 cache 一致。一个可以处理 `L_test` 的 kernel 不代表模型已经学会 `L_test`。

## 18.5 位置插值与频率缩放

位置插值把长位置压回训练范围，例如：

~~~math
p'=p\cdot\frac{L_{train}}{L_{test}}
~~~

频率缩放则调整 `\omega_i`，降低相位变化速度。二者都在改变模型看到的相对距离分布，可能保留长程稳定性，却牺牲近距离分辨率。实践中还可能配合少量长文本继续训练、分段损失或分层频率策略。

这些公式是解释性抽象。具体 implementation 可能按维度、层或 attention 类型采用不同缩放，必须对照模型配置和代码，而不能只修改一个 `rope_theta` 就宣称复现了某论文。

## 18.6 NTK-aware scaling、YaRN 与 ALiBi 的位置

NTK-aware scaling 的直觉是让不同频率维度以不同方式调整，使核函数在更长范围内保持较平滑的相似性。YaRN 组合了位置插值、频率处理和训练/温度方面的策略，目标是以较少额外训练扩展上下文。ALiBi 不加位置 embedding，而是把与距离成比例的偏置加入 attention score：

~~~math
s_{ij}=\frac{q_i k_j^\top}{\sqrt{d}}-m_h|i-j|
~~~

ALiBi 的 recency bias 使外推行为与 RoPE 不同。它们不是简单的“谁更先进”，而是不同 inductive bias。比较时应固定训练长度、模型规模、长程任务和 kernel。

## 18.7 一个位置外推实验矩阵

最小实验至少包含四个轴：训练长度、测试长度、证据位置和任务类型。比如训练 2K，测试 2K/8K/32K；把 passkey 放在开头/中间/结尾；再加入多证据合并和数字复制。

~~~python
def scaled_position(position, train_len, test_len):
    return position * train_len / test_len


for test_len in (2_048, 8_192, 32_768):
    mapped = scaled_position(test_len - 1, 2_048, test_len)
    print(test_len, round(mapped, 2))
~~~

这段代码只是展示插值关系。真实结果要从 logits、loss、exact match、引用准确率和吞吐中读出，不能凭位置映射值估计能力。

## 18.8 位置与 KV cache 的契约

RoPE 通常在生成时作用于新 query 和新 key。若 prefill 后继续 decode，新增 token 必须使用连续的 position id；prefix cache 复用时，缓存内容和起始位置必须匹配。多轮对话压缩、滑窗截断、streaming attention sink 和文档重排都会改变位置契约。

常见 bug 是：恢复请求时 position 从 0 重新开始；对 K 做过一次旋转后又重复旋转；不同 tensor parallel worker 使用不同 position；batch 内 padding 把位置推进；或长上下文 scaling 只改了 query 没改 key。排查可以在固定短序列上比较逐 token logits，再逐步增加长度。

## 18.9 训练稳定与精度问题

长位置的 sin/cos 计算、半精度角度、超长 position id 和不同 GPU kernel 可能带来数值差异。角度过大时，低精度计算会放大误差；位置缩放后近邻频率又可能过于接近。要检查 cos²+sin²≈1、旋转前后范数、不同 dtype logits 差异和超长分桶 loss。

位置机制还与 attention temperature、softmax 稳定、长样本 curriculum 和数据 packing 相互作用。只看短序列 loss 下降，不能说明长位置训练完成。

## 18.10 位置方案的比较

| 方法 | 位置进入方式 | 主要优势 | 主要风险 |
|---|---|---|---|
| Learned absolute | 加到 embedding | 简单、训练范围内直接 | 外推弱 |
| Sinusoidal | 固定函数 | 无额外参数 | 长度和频率不匹配 |
| RoPE | 旋转 Q/K | 相对位移自然、生态成熟 | 长位置相位外推 |
| ALiBi | score 线性偏置 | 结构简单、recency 先验 | 任务/距离偏置可能过强 |
| 插值/缩放 | 改位置映射或频率 | 可扩展现有 checkpoint | 近远程分辨率 trade-off |

## 18.11 机制与边界：外推的真正问题是核和任务分布

位置编码定义了不同位置表示的相似性核，但模型能力还由 attention 内容寻址、训练分布和层间传播决定。两个方法可能有相近的短文本 perplexity，却在精确检索、跨段推理和局部代码补全上表现不同。

长上下文训练常需要同时调节：长样本比例、数据去重、loss 权重、位置分布、检索任务和 memory/compute budget。若只增加 `max_position_embeddings`，那通常只是接口层变化。评测需区分“模型在长输入上不崩”和“模型能在长输入中正确找到证据”。

## 18.12 面试追问、误区与练习

**问：RoPE 为什么能表达相对位置？**

标准回答：对 query 和 key 按各自位置旋转后，二者内积中的旋转相对角与位置差有关，因此 attention score 同时含有内容相似性和相对位移信息。

**问：把上下文窗口从 8K 改成 128K 就完成长上下文扩展了吗？**

标准回答：没有。还要核对位置缩放/插值、长文本训练、mask、KV cache、精度、有效检索和跨段评测。接口上限只是必要条件。

常见误区包括把 RoPE 的 theta 当成最大长度、混淆 interleaved 与 split-half 实现、只改 config 不改 kernel、把 YaRN/NTK scaling 当作同一算法，以及忽略缓存恢复的 position id。

练习：在同一小模型上实现直接 RoPE、线性插值和 ALiBi，对 2K 训练/8K 测试的数字复制、局部 next-token 和中间证据任务做分桶评估。

## 18.13 相位分辨率与长度外推

RoPE 的不同频率维度承担不同时间尺度。高频维度能区分近邻位置，但周期更短；低频维度变化慢，适合远距离趋势，却可能缺少局部精度。长度扩展实际上是在分配这两类分辨率。

可以把某个频率的角度写成：

~~~math
\theta_{p,i}=p\omega_i,\qquad
\Delta\theta_{p,r,i}=(p-r)\omega_i
~~~

当相位差超出训练分布，模型可能出现混叠；当所有频率都被过度压缩，远处位置难以区分。位置外推的实验要同时看近距复制和远距定位，不能只看最大长度。

## 18.14 从数学运行到任务可用

把 position id 送进 kernel 并不证明 RoPE 扩展成功。应做四层验收：旋转数值与参考实现一致，mask 和 cache offset 正确，长度曲线不出现异常断点，长程任务的证据和引用通过验收。

恢复和 prefix sharing 还要测：同一前缀一次性处理与分块处理的 logits 是否一致；同一 token 位置在不同 batch 排列中是否保持 position；模型升级后旧 cache 是否被拒绝。

## 18.15 RoPE 外推的实验设计

位置外推至少需要两个维度的对照。第一维是长度：训练长度、略超出长度和远超长度；第二维是任务：局部 next-token、远距复制、中间证据、相对距离判断和多轮 cache 恢复。只在长文本 perplexity 上变好，不能证明精确定位能力变好。

对每个长度 L，记录：

~~~math
R(L)=\Pr(\text{exact answer}\mid L),\qquad
\Delta_{\mathrm{cache}}(L)
=\lVert y_{\mathrm{full}}-y_{\mathrm{chunk}}\rVert_\infty
~~~

前者是任务能力，后者是执行一致性。若任务指标下降而 cache 一致，可能是训练外推不足；若 cache 误差先出现，优先查 position offset、旋转实现或分页恢复。

## 18.16 位置缩放与内容分辨率的交换

RoPE 通过相位差把相对位置带入 Q/K 关系。缩放位置相当于重新分配相位变化速度：远距离位置更容易保持在训练分布附近，但近邻位置的分辨率可能降低。不同频率维度的影响并不相同，不能用一个“最大长度倍数”概括所有任务。

长上下文训练还需要让数据分布真正出现远距离依赖。只修改位置函数而不提供长距离监督，模型可能接受长输入，却没有学会在远端寻找证据。这是接口上限、训练长度和有效能力之间的差别。

## 18.17 RoPE 与 cache、并行和多模态

在线服务要保存逻辑 position、segment、window offset 和模型 revision。prefix cache 命中时，复用的不只是 K/V 数值，还包括它们对应的位置语义；多模态 token 插入、图像 patch 数变化和分块视频采样都可能改变 position 序列。

因此 position bug 往往表现为偶发的长文档错误、跨 chunk 变化或多模态时间错位。最小回归应覆盖一次性运行、chunked prefill、continuous batching、prefix sharing、量化和恢复。

## 18.18 相位误差如何变成任务误差

RoPE 对第 p 个位置的每个二维子空间施加与 p 成比例的旋转。若训练最大位置为 `L_train`，测试位置达到 `L_test`，外推方法实质上是在改变角度随位置的增长速度。低频分量可以提供长距离相对位置，高频分量则更容易在长位置上产生快速相位变化和分辨率损失。

这解释了一个常见现象：模型可能在长上下文的主题总结上表现不错，却在精确位置、数字复制和两个相邻证据的顺序判断上退化。评估不能只测 perplexity，应加入位置交换、重复实体、跨段引用和数字精确匹配。

## 18.19 外推实验要同时测长度和位置

一个最小矩阵固定模型和 tokenizer，只改变最大长度、缩放策略、证据位置和任务类型：短训短测、短训长测、长训长测、随机位置、首尾位置、中间位置和跨段组合。每个格子报告 loss、needle recall、引用支持率、数字准确率、格式通过率、TTFT、KV 显存和 p99。

如果某策略在 128K 的平均分提高，却让 8K 短任务下降，说明它改变了整个位置分布而非只补长尾；如果 loss 正常但中间位置失败，问题可能是有效检索而不是单纯位置外推。发布材料中的 context claim 只能作为测试入口，不能直接转成“可进行任意长程推理”。

## 18.20 位置方案与 cache、并行和多模态

位置编码不是只存在于一次前向中。增量 decode 需要 position offset 正确，prefix cache 需要绑定 position scheme 和模板，chunked prefill 需要保持块间位置连续，tensor/pipeline parallel 需要保证分片后的 position ids 一致。多模态模型还要定义图像 patch、音频帧和文本 token 如何共享或分离位置坐标。

因此位置方案升级要做 token-level 和 logit-level golden replay。缓存命中、长输入、断点恢复、batch reorder、图片/音频占位和 speculative rollback 都要覆盖；只比较最终短文本无法发现位置偏移一位、EOS 错位或媒体 token 位置漂移。

## 18.21 位置扩展的验收条件顺序

长上下文位置方案不应一次性用大模型和长文档验收。先做数学单元测试：固定 `q`、`k` 和位置，比较参考旋转与实现旋转的最大误差；再做 cache 一致性测试：把同一输入一次性 prefill 和分块 prefill，比较每个位置的 logits；最后才做任务曲线和生产压测。

```math
e_rope = max over p,i of distance(R_impl(p,i), R_ref(p,i))
```

在分块路径中，若第一个 chunk 长度为 `u`，第二个 chunk 的 position 必须从 `u` 继续，而不是从 0 重置。prefix cache 还要绑定模型 revision、RoPE 参数、tokenizer/template 和媒体 processor；否则 cache 命中会把旧位置语义复用到新协议上。这个错误通常不会在短样本立刻暴露，却会在长文档中造成局部证据错位。

## 18.22 一个可手算的外推例子

设模型在 2K 上训练，测试长度为 8K。把一个关键数字放在 `p=1536`、`p=4096` 和 `p=7168`，其余 token 使用相同的干扰模板。直接外推、线性位置插值和频率缩放都跑三组任务：局部 next-token、精确复制和跨段引用。

局部 next-token 可能在三种方案下都正常，因为它主要使用邻近 token；精确复制更容易暴露高频相位分辨率损失；跨段引用还会受到干扰和训练分布影响。若插值方案的复制率上升但局部任务下降，说明它在长短任务之间做了交换；若只有 cache 分块路径失败，则先修 position offset，不应把问题归因于模型外推能力。

报告结果时至少分开：`R_local(L)`、`R_copy(p,L)`、`R_ground(p,L)`、cache logit error、TTFT 和峰值 KV。一个“支持 8K”的结论只有在这些指标和具体 prompt/harness 都写清楚时才有意义。

## 18.23 从相位误差到检索失败的链路

位置扩展的风险可以沿一条可测链路理解：位置公式改变角度，角度改变 query/key 的相对相位，相位改变 attention score，score 改变证据选择，最终表现为 recall、引用或多段推理下降。对某一频率 `omega_i`，位置误差 `Delta p` 造成的角度误差近似为：

~~~math
\Delta\theta_i=\omega_i\Delta p.
~~~

高频维度对位置误差更敏感，低频维度更容易覆盖长距离，但低频又可能降低近距离区分度。因而不能只测最长长度的平均 perplexity；应分别改变证据位置、长度、干扰数量和实体冲突，绘制位置—任务成功率矩阵。若中间位置和边界位置差异很大，问题可能是训练分布、attention bias、模板或 cache，而不只是 rope 参数。

## 18.24 RoPE scaling 的实现契约

长上下文配置至少包含 base/theta、频率缩放、位置插值、最大训练长度、最大推理长度、维度排列、层/维度策略和 KV cache position。把配置中的一个字段改大，不等价于复现某个 NTK-aware 或 YaRN 配方；不同实现可能同时改变频率、温度、训练数据和 attention 层行为。

发布前要做三条路径对照：完整 prefill、增量 decode 和 chunked/恢复 decode。对同一个 prefix，保存每一步 position id、旋转后的 Q/K checksum 和 logits；如果只比较最终答案，采样噪声会掩盖位置错位。多模态模型还要单独定义图像 patch、视频帧和文本位置的坐标系，不能把媒体 token 随意接在文本 position 后就认为对齐正确。

## 18.25 长上下文评估的最小矩阵

一个有辨识力的矩阵至少包含：短/长长度，证据在开头/中间/末尾，单证据/多证据，独立实体/冲突实体，直接回答/引用来源，以及 full/streaming/cache restore。可以记录：

~~~math
A_{\mathrm{exact}}(L,p),
\qquad
A_{\mathrm{cite}}(L,p),
\qquad
E_{\mathrm{logit}}(L,p),
~~~

其中 `L` 是长度，`p` 是证据相对位置。`A_exact` 只说明答案对不对，`A_cite` 还要求来源和字段正确，`E_logit` 用于发现实现路径差异。对一个 1M claim，真正重要的不是单点能否放入请求，而是这些曲线在目标长度、成本和 p99 验收条件下是否仍然可用。

## 18.26 RoPE scaling 和 NoPE 不能混为一谈

没有显式 RoPE 或其他位置变换，不等于模型没有顺序信息。顺序可能来自 causal mask、训练数据中的排列统计、卷积/状态路径、token 内容和实现中的递推步数。NoPE 需要回答的是模型如何获得、保持和泛化顺序，而不是简单得出“没有位置编码”。

因此比较 RoPE、ALiBi、NoPE 或混合位置方案时，应固定模型规模、数据、训练长度和 tokenizer，分别测试顺序交换、局部语法、远程复制、长上下文位置、cache restore 和多模态时间对齐。公开模型若没有披露完整位置实现，正文只能记录可观察行为和证据范围，不能用模型名称推断内部方案。

## 18.27 公开证据与工程边界

RoPE 原始论文、位置插值、NTK-aware scaling 和 YaRN 资料可以支持各自的公式与实验设定；具体模型的长上下文训练配方、位置实现、token budget 和 serving cache 要以该版本配置、技术报告和复现实验核对。一个位置方案在论文长度上有效，不等于在任意模型、任意任务和任意媒体输入上有效。

## 18.28 小结

RoPE 原始论文为 *RoFormer: Enhanced Transformer with Rotary Position Embedding*（https://arxiv.org/abs/2104.09864）；ALiBi 见 https://arxiv.org/abs/2108.12409。NTK scaling、YaRN 及各模型的长上下文方案应以对应论文、模型卡和实现版本为准。

RoPE 的价值是把位置和内容寻址结合得很自然，但长上下文不是调一个数字。位置公式、训练长度、有效任务和 cache 契约必须一起通过验证。
