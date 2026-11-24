# 17. LLaMA、GPT、Mistral 风格架构对比

“GPT 用什么架构？”这个问题需要先补上版本。GPT-3 的论文公开了层数、宽度和注意力模式；GPT-4 技术报告说它是预测下一个 token 的 Transformer，却没有披露内部 block 规格；OpenAI 后来公开的 gpt-oss 又采用了 MoE、GQA 和局部/全局注意力组合。它们都叫 GPT，但不能据此认为内部结构相同。

LLaMA 和 Mistral 也一样。本章不把品牌名当作架构模板，而是用公开版本说明怎样读 block 选择、配置字段和证据边界。我们要回答的是“这个版本采用什么结构”，而不是“哪个家族更强”。

## 17.1 先问“哪一个版本”，再问“什么架构”

模型家族名通常包含多个时期、规模和用途不同的模型。比较之前，先锁定模型 ID、版本，以及它是基础模型还是指令微调版本。论文描述的方法、模型卡公开的配置、固定版本配置文件记录的字段，能支持的结论也不一样。

以 GPT 为例，GPT-3 论文把 175B 模型描述为沿用 GPT-2 架构，特别指出层间交替使用密集注意力和局部稀疏注意力；报告还给出层数、隐藏宽度和注意力头数。GPT-4 技术报告只说明模型以预测文档下一个 token 为预训练目标，并能接收图像和文本输入；报告明确不提供架构、模型规模等细节，因此不能从 GPT-3 或产品表现反推出 GPT-4 的内部结构。[GPT-3 论文](https://arxiv.org/abs/2005.14165)；[GPT-4 技术报告](https://cdn.openai.com/papers/gpt-4.pdf)

gpt-oss-20b 则有不同的公开模型卡：它是 24 层的 MoE Transformer，每个 token 激活约 3.6B 参数，并采用 GQA 与局部/全局注意力交替。这个版本能用于讨论 OpenAI 公布的架构，但不能据此推断 GPT-4 或其他 API 模型也使用相同结构。[gpt-oss 模型卡](https://deploymentsafety.openai.com/gpt-oss/architecture)

因此，读到“GPT 使用某种注意力”时，要继续追问来源指的是 GPT-2、GPT-3、gpt-oss，还是一个没有公开配置的产品模型。相同做法也适用于 LLaMA 与 Mistral。

## 17.2 用同一张 decoder block 图确定比较坐标

本书第十六章已经说明，decoder-only 模型沿因果方向处理 token：当前层的表示经过注意力子层和前馈子层，再进入下一层。各版本常共享这个大体骨架，却可以替换或调整其中的部件。读模型卡时，下面这些坐标能帮助我们分清它们改了哪里：

| 比较坐标 | 读配置时要问什么 | 它描述什么 |
| --- | --- | --- |
| 位置表示 | 位置向量加在什么位置，或位置关系怎样进入注意力？ | 模型怎样获得 token 顺序信息 |
| 归一化 | 用什么归一化函数，放在子层之前还是之后？ | block 中激活值如何被规范化 |
| 注意力头 | query heads 与 key/value heads 各有多少，是否分组共享？ | 注意力通道的组织方式及 KV 状态规模 |
| 可见范围 | 每层读取全部前文，还是只读取局部窗口？不同层的规则是否不同？ | 当前表示能从哪些较早位置取得信息 |
| 前馈子层 | 是普通稠密 FFN、门控 FFN，还是 MoE 专家层？ | 注意力之后怎样变换表示、是否按 token 路由到专家 |
| 上下文相关字段 | 来源写的是训练序列长度、`context_len`、`max_position_embeddings`，还是 `sliding_window`？ | 这些字段各自描述的范围，不应被压成一个数字 |

这些字段是比较工具，不是每个模型都必须填满的检查表。来源没有公开某个部件时，标记为未知比从家族名补全更可靠。归一化和注意力机制的细节分别见[第六章](06-residual-layernorm-rmsnorm-prenorm与训练稳定性.md)和[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md)；本章只解释它们怎样帮助我们读具体版本。

## 17.3 位置、归一化与前馈子层：同一主干里的不同部件

GPT-2 的公开参考实现把 token embedding 与可学习的位置表相加，在注意力和 MLP 子层前使用 LayerNorm，MLP 使用 GELU。GPT-3 论文说明它沿用 GPT-2 架构，只有注意力模式是特别指出的例外。因此，把这些细节作为 GPT-3 公开设计基线，是由论文和参考实现共同支持的判断；它不代表后来的闭源 GPT 模型。[OpenAI GPT-2 参考实现](https://github.com/openai/gpt-2/blob/master/src/model.py)；[GPT-3 论文 §2.1](https://arxiv.org/abs/2005.14165)

原始 LLaMA 论文报告了另一组选择：在 Transformer 子层前进行预归一化，归一化函数采用 RMSNorm；前馈子层采用 SwiGLU；位置信息通过每层的旋转位置嵌入 RoPE 加入，而不是使用可学习的绝对位置表。Meta 后续的 Llama 3 技术报告称其仍是标准稠密 Transformer，并在公开的 8B 配置表中列出 SwiGLU、RoPE 与 GQA；表中 RoPE 基频为 500,000。Llama 3.1 官方模型卡另外列出 128K 上下文和 GQA。这里引用的论文表与模型卡各自对应不同的公开说明，不应把某个字段悄悄挪到所有 LLaMA 版本上。[原始 LLaMA 论文](https://arxiv.org/abs/2302.13971)；[Llama 3 技术报告](https://arxiv.org/abs/2407.21783)；[Llama 3.1 官方模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)

Mistral 7B v0.1 也采用预归一化。它的固定配置记录 `rms_norm_eps=1e-5`、`rope_theta=10000` 和 `hidden_act=silu`；与之对应的 Transformers v4.34.0 参考实现显示，block 在 attention 与 MLP 子层前使用 RMSNorm，RoPE 作用于 Q/K，MLP 将 SiLU 门控分支与另一投影逐元素相乘。这里的实现核验只解释这一固定配置，不代表后续 Mistral 版本都使用相同组件。[Mistral v0.1 固定配置](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/93bc63fc98d90b891e2ab513e1ebb7480424cec3/config.json)；[Transformers v4.34.0 Mistral 实现](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py)

gpt-oss 的模型卡则明确记录了 RMSNorm、RoPE 和带 SwiGLU 激活的 MoE 前馈层。它仍使用 Transformer decoder 主干，但前馈子层不再是每个 token 都经过同一组稠密权重：路由器为 token 选择专家。专家路由机制见[第二十一章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)，Dense 与 MoE 的训练/服务成本见[第二十二章](22-Dense与MoE训练服务trade-off.md)。

位置编码决定模型如何把顺序线索放进计算，归一化和前馈子层则决定每层表示怎样变化。它们是可分别替换的设计点，不是“GPT”“LLaMA”这些名字自带的定义。RoPE 的位置关系与扩展方法见[第七章](07-position-encoding总览-sinusoidal-learned-relative-rope-alibi.md)和[第十八章](18-RoPE与长上下文位置外推.md)。

## 17.4 MHA 与 GQA：看清 Q 头和 KV 头是否一一对应

注意力配置中，query heads 负责提出不同的查询；key 和 value heads 提供被查询的位置表示和值。MHA 为每个 query head 配有各自的 K/V head。MQA 让所有 query heads 共用一组 K/V。GQA 位于两者之间：先把 query heads 分组，每组共享一组 K/V。GQA 论文把这种中间方案作为降低 KV 状态读取量的设计，并在其特定的 T5 uptraining 实验中报告：GQA 的质量接近 MHA，速度接近 MQA。这个结果受实验配置约束，不等于所有 GQA 模型在所有硬件上都会达到同样的速度或质量。[GQA 原论文](https://arxiv.org/abs/2305.13245)

配置里的头数能让这层关系变得具体。Llama 3 8B 和 Mistral 7B v0.1 的论文都列出 32 个 query heads、8 个 KV heads；按 GQA 分组时，每组对应 4 个 query heads。gpt-oss-20b 的模型卡列出 64 个 query heads、8 个 KV heads，每组对应 8 个 query heads。KV heads 少于 query heads，意味着不同 query 组复用同一组 K/V；它直接改变注意力的结构和缓存中的 KV 数量，但模型质量、实际吞吐仍受其他因素影响。

GPT-3 175B 论文列出 96 个注意力头，并称其架构沿用 GPT-2、只有 attention pattern 例外。GPT-2 参考实现为各个头分别生成 Q、K、V，因此可以把这两个来源合起来，将 GPT-3 公开基线视为 MHA 设计。这里是对指定公开版本的来源推断；它不能用于猜测未披露 GPT 模型的 KV 头数。KV cache 的容量计算与推理阶段影响由[第十一章](11-kv-cache长上下文推理和显存增长问题.md)主讲。

## 17.5 Dense 与滑动窗口：局部窗口不等于总上下文

注意力头的数量描述了通道怎样组织，可见范围则描述每层能从哪些 token 读信息。普通因果注意力允许位置查看全部较早 token。局部窗口注意力只开放附近一段位置。即使某层只看局部范围，上一层的表示也可能已经汇集更远处的信息；层层传递后，一个 token 的影响可以跨出单层窗口。它说明“局部可见”不等于“模型永远看不到窗口外的信息”，但这种间接传递不保证模型能可靠检索任意距离的内容。

Mistral 7B v0.1 论文给出的滑动窗口宽度为 4,096，模型有 32 层；论文据此报告约 131K token 的理论 attention span，同时在表格中把 `context_len` 列为 8,192。理论 span 描述信息在层间可能传播的最远范围，不等同于该版本训练和声明的上下文窗口。固定 revision 的官方配置还把 `max_position_embeddings` 写为 32,768，并单独记录 `sliding_window=4096`。这几个值来自不同字段与材料，不能把配置上限直接解释成已训练、已验证或可靠使用的上下文长度。[Mistral 7B 论文](https://arxiv.org/abs/2310.06825)；[固定 revision 的 Mistral 配置](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/93bc63fc98d90b891e2ab513e1ebb7480424cec3/config.json)

局部与全局层也可以交替。GPT-3 论文和 gpt-oss 模型卡都报告局部窗口与密集注意力交替；gpt-oss 的模型卡把局部带宽列为 128。全局层提供跨序列读取路径，局部层则限制该层的直接可见范围。Mistral 7B v0.1 论文使用的是滑动窗口设计，并报告在特定窗口、序列长度和经过修改的 attention 实现下的速度与缓存收益。这些收益是该论文的实验结果，不是仅凭“使用 SWA”就能推出的普遍结论。

LLaMA 的长上下文版本也显示，窗口、位置机制和训练长度必须分开看。Llama 3 的技术报告描述从原始 8K 上下文继续训练到 128K；Llama 3.1 官方模型卡列出 128K context length。长上下文的 RoPE 扩展、训练适配和评测边界详见[第十八章](18-RoPE与长上下文位置外推.md)。本章只要求读者辨认来源中的字段，不把一个最大数字当成模型能可靠利用所有位置的证据。

## 17.6 用版本化对照得出有限结论

下表把论文和模型卡中可核验的版本放在一起。它用来练习定位结构与辨认未知项，不构成质量或推理速度排名。

| 指定对象 | 来源支持的结构事实 | 读取时要保留的边界 |
| --- | --- | --- |
| GPT-3 175B | 96 层、96 个注意力头；GPT-2 架构基线；层间交替 dense 与局部稀疏 attention；上下文窗口 2,048（[论文](https://arxiv.org/abs/2005.14165)） | GPT-2 参考实现可补充公开基线的可学习绝对位置、LayerNorm、GELU 与 MHA（[实现](https://github.com/openai/gpt-2/blob/master/src/model.py)）；不能推广到 GPT-4 等未披露模型 |
| GPT-4 技术报告 | 说明 Transformer、自回归 next-token 预训练、图像/文本输入（[技术报告](https://cdn.openai.com/papers/gpt-4.pdf)） | 报告未披露 block 结构或模型规模；应记录为未知 |
| gpt-oss-20b | 24 层 MoE；64 个 query heads、8 个 KV heads；RoPE；局部窗口与 dense 层交替（[模型卡](https://deploymentsafety.openai.com/gpt-oss/architecture)） | 模型卡列出 128K 上下文和 128-token 局部带宽；仅代表 gpt-oss 版本 |
| Llama 3 / 3.1 8B | 论文列出 32 层、32 个 query heads、8 个 KV heads、SwiGLU、RoPE（[Llama 3 报告](https://arxiv.org/abs/2407.21783)）；官方 3.1 卡列出 GQA 和 128K context（[模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)） | 论文表、3.1 卡和具体 config 是不同证据对象；引用时注明用的是哪一项 |
| Mistral 7B v0.1 | 32 层、32 个 query heads、8 个 KV heads、4,096 窗口；论文表列 `context_len=8,192`（[论文](https://arxiv.org/abs/2310.06825)）；固定配置与 v4.34.0 实现记录 RMSNorm、RoPE 和 SiLU 门控 MLP（[配置](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/93bc63fc98d90b891e2ab513e1ebb7480424cec3/config.json)、[实现](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py)） | 固定配置另列 `max_position_embeddings=32,768`；不能将字段合并成同一“有效上下文” |

表中一个版本比另一个版本多了局部窗口、GQA 或 MoE，只能说明这些公开配置的结构不同。若要比较能力、质量或速度，还必须对齐模型规模、tokenizer、训练数据与训练配方、长上下文训练、后训练、软件和硬件、输入负载与评测协议。模型间的 benchmark 分数或单次吞吐差异不能单独证明某个 block 设计更优。

实际阅读模型架构时，可以按这样的顺序推进：先锁定模型名、版本和来源；再把位置表示、归一化、头数、可见范围与 FFN 对齐；对每个字段确认它的定义和适用版本；最后把来源明确写出的事实与自己的推断分开。家族名帮助我们找入口，具体版本和公开证据才支持架构结论。GPT 谱系的变化见[第五十一章](51-GPT-decoder-only-scaling-ICL.md)；注意力变体、归一化、位置机制、KV cache 与 MoE 的完整解释分别见本册[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md)、[第六章](06-residual-layernorm-rmsnorm-prenorm与训练稳定性.md)、[第七章](07-position-encoding总览-sinusoidal-learned-relative-rope-alibi.md)、[第十一章](11-kv-cache长上下文推理和显存增长问题.md)、[第十八章](18-RoPE与长上下文位置外推.md)、MoE 路由和 Dense/MoE 成本的[第二十一章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)/[第二十二章](22-Dense与MoE训练服务trade-off.md)，以及小模型和端侧效率的[第二十三章](23-小模型边缘模型与高效Transformer.md)。
