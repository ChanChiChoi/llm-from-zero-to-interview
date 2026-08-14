# 第十七章：LLaMA、GPT、Mistral 风格架构对比

## 17.1 先分清“家族名”和“可观察结构”

GPT、LLaMA、Mistral 既可以指一系列模型，也可以指一组常被复用的设计习惯。面试和工程讨论中最危险的说法是“某模型就是某家族，所以所有细节都一样”。公开资料通常只披露部分层结构、训练配方和推理接口；闭源模型更不应根据输出行为反推完整架构。

更稳妥的比较方式是逐项记录：是否 decoder-only、位置机制、attention 的 Q/K/V 头数、FFN 激活、normalization、窗口或稀疏 mask、词表和 tokenizer、训练上下文、后训练模板以及 serving 支持。模型名字只是索引，结构字段才是证据。

## 17.2 三种风格的历史角色

GPT 系列把 decoder-only、自回归 next-token prediction 和规模化训练推到主流位置。它的核心价值是统一目标和大规模上下文学习；具体后续版本的内部细节不一定公开。

LLaMA 系列对开源生态的影响在于提供了一个可复现、可扩展的现代 decoder-only 基线。RoPE、RMSNorm、SwiGLU、更成熟的数据和训练 token 组合，被许多后续模型采用。

Mistral 7B 则把“同等参数下如何更高效”放到台前，公开资料明确讨论了 GQA 和 Sliding Window Attention。它说明模型能力不能只看参数量，attention 头共享、局部窗口、数据质量和训练配方都能改变实际表现。

## 17.3 核心模块的共同骨架

现代 decoder block 通常可以写成：

~~~math
h' = h + \operatorname{Attn}(\operatorname{Norm}(h))
~~~

~~~math
h_{\mathrm{out}} = h' + \operatorname{FFN}(\operatorname{Norm}(h'))
~~~

常见的 Pre-Norm 版本先做 RMSNorm，再进入 attention/FFN；部分模型的具体 residual、norm 放置和 bias 选择不同。RMSNorm 的教学形式为：

~~~math
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
~~~

它不做均值中心化，计算路径更简洁。不能由“用了 RMSNorm”直接断言一定更稳定；稳定性还取决于初始化、学习率、残差缩放、精度和训练规模。

## 17.4 MHA、GQA 与 MQA 的实际差异

设 query head 数为 `H_q`，KV head 数为 `H_{kv}`，每个 head 维度为 `d_h`。MHA 是 `H_q=H_{kv}`；MQA 近似为 `H_{kv}=1`；GQA 将多个 query head 绑定到一个 KV group：

~~~math
g(h)=\left\lfloor\frac{h}{H_q/H_{kv}}\right\rfloor
~~~

每个 KV head 的历史 cache 大小近似为：

~~~math
M_{KV}=2\cdot B\cdot T\cdot H_{kv}\cdot d_h\cdot b
~~~

其中 `2` 对应 K 和 V，`b` 是每元素字节数。GQA 在减少 cache 的同时，让不同 query head 共享历史表示，可能损失一部分表达自由度。它不是单纯“免费压缩”，需要在 perplexity、长程检索和真实 batch 上验证。

## 17.5 SwiGLU 与 FFN 预算

普通 FFN 可以写成：

~~~math
\operatorname{FFN}(x)=W_2\,\sigma(W_1x)
~~~

SwiGLU 形式常写为：

~~~math
\operatorname{SwiGLU}(x)=W_2\left(\operatorname{SiLU}(W_gx)\odot W_upx\right)
~~~

它引入 gate 分支和 up 分支，通常需要调整 hidden size，才能在参数/FLOPs 预算下公平比较。只看中间维度相同会误判成本，因为矩阵数量和投影形状都改变。

## 17.6 用配置表而不是宣传语比较

| 维度 | GPT 风格公开基线 | LLaMA 风格基线 | Mistral 风格公开基线 |
|---|---|---|---|
| 主体 | decoder-only | decoder-only | decoder-only |
| 训练目标 | causal LM | causal LM | causal LM |
| 位置 | 版本相关 | RoPE 常见 | RoPE + 局部窗口常见 |
| KV | 版本相关 | GQA 等变体 | GQA |
| FFN | 版本相关 | SwiGLU 常见 | SwiGLU 常见 |
| 长序列 | 版本和服务相关 | 需外推/训练验证 | SWA 与全局路径取舍 |
| 证据 | 官方技术资料 | 论文/模型卡 | Mistral 7B 论文 |

表中的“常见”不是所有版本的事实。实际选型应把具体 revision、tokenizer、chat template、上下文和权重路径写入实验记录。

## 17.7 一个 KV 预算计算器

~~~python
def kv_cache_mib(batch, tokens, kv_heads, head_dim, bytes_per_value=2):
    raw = 2 * batch * tokens * kv_heads * head_dim * bytes_per_value
    return raw / (1024 ** 2)


for name, kv_heads in (("MHA", 32), ("GQA", 8), ("MQA", 1)):
    print(name, round(kv_cache_mib(8, 32_000, kv_heads, 128), 1), "MiB")
~~~

这段代码只估算一个层的 K/V；完整模型还要乘以层数，并考虑分页、对齐、量化 scale、临时 workspace 和多请求共享。它适合在设计评审时快速识别“权重能放下但 KV 放不下”的情况。

## 17.8 Mistral 的滑窗和全局能力

滑窗 attention 限制当前位置直接看的历史范围。它能降低每层的 attention 计算和 cache 需求，但远处证据要通过层间传播或特殊 global token 传递。一个深度为 `L`、窗口为 `w` 的粗略直接感受野可近似为 `Lw`，但这不是精确可用上下文，因为内容选择、残差和层间混合会改变有效路径。

长文档任务如果需要随机访问任意位置，纯滑窗可能不够；如果数据高度局部、输入持续流动，滑窗可能是合理的带宽策略。评估应包括局部语言建模、远距复制和跨段问答，不能只看短 benchmark。

## 17.9 训练与推理边界

同一个架构在训练和 serving 中的瓶颈不一样。训练更关心激活、全序列 attention 和通信；推理更关心 KV cache、decode 带宽、batch、tokenizer 和模板。GQA 对推理 cache 直接有利，但训练质量可能需要更多数据或不同 head 配置；SWA 减少访问范围，却要求 kernel 和 mask 实现一致。

模型对比还要控制后训练。一个强大的 instruction tuning 或 reasoning recipe 可能掩盖 base architecture 的差异；如果要回答“架构贡献是多少”，需要同数据、同 token、同后训练预算的消融。

## 17.10 常见失败模式

最常见的错误是把“LLaMA-like”写成完整架构事实；把 Mistral 的 SWA 误写成所有版本都只看局部；把 GQA 的 KV head 数和 query head 数混淆；只比较参数量不比较 active FLOPs；把模型卡的上下文上限当作有效长程能力；以及把 API 兼容当作 tokenizer/模板兼容。

排查时先打印 config、tokenizer special tokens、position ids、attention mask 和实际 batch shape。再用同一 prompt 测 greedy logits、一致性和 cache 命中；最后才对 benchmark 差异做结构解释。

## 17.11 机制与边界：比较应落到 Pareto 前沿

架构选择可以看成多目标优化：

~~~math
\max\; Q(\text{quality}),\qquad
\min\; (M_{KV},T_{\mathrm{latency}},C_{\mathrm{train}},C_{\mathrm{serve}})
~~~

没有一个模型在所有目标上同时最优。GQA、SWA、SwiGLU、RoPE 都是具体预算下的取舍。若部署是低并发短 prompt，KV 压缩的收益可能小于 kernel 兼容性；若是高并发长对话，KV 和调度可能比单 token FLOPs 更重要。

## 17.12 面试切入点与资料范围

**问：为什么 GQA 常被认为是 MHA 和 MQA 的折中？**

标准回答：它让多个 query head 共享一个 KV group，减少 KV cache 和读取带宽，同时比单组 MQA 保留更多独立 K/V 表达；最终质量和速度要用同一模型族、同一硬件及真实 workload 验证。

**问：比较 LLaMA 和 Mistral 时最少看哪些字段？**

标准回答：decoder block、norm、RoPE、Q/K/V head 数、窗口/稀疏 mask、FFN、训练长度、tokenizer、后训练模板和 serving kernel。只看参数量或模型名字不够。

练习：选两个公开模型，读取 config 并填出上述字段；用第 17.7 节计算理论 KV，和实际显存 profile 对比，解释差值来自哪里。

LLaMA 可参考 https://arxiv.org/abs/2302.13971；Mistral 7B 的公开架构说明见 https://arxiv.org/abs/2310.06825；GPT-3 的 decoder-only 与 ICL 背景见 https://arxiv.org/abs/2005.14165。闭源 GPT 版本的未披露结构不要用家族名推断。

## 17.13 用同一资源表比较模型

比较 GPT-like、LLaMA-like 和 Mistral-like 模型时，建议把配置与系统指标放在同一张表：

~~~text
layers
hidden_size
attention_heads
kv_heads
ffn_type
position_scheme
context_training_length
total_parameters
active_parameters
kv_bytes_per_token
~~~

再追加实际的 prefill tokens/s、decode tokens/s、TTFT、TPOT、p99 和任务质量。结构上的 GQA 或滑窗只有在这些资源指标中才有可验证的含义。

## 17.14 差异来自模块还是训练配方

若两个模型配置相近，能力差异可能来自数据、token 数、去重、优化器、学习率、后训练和评测污染。要证明某个模块带来收益，需要在相同基座上做 controlled ablation，而不能把不同公司的完整模型直接当作模块实验。

一个合格的对比至少包括原始模型、替换模块、相同训练 token、相同 tokenizer、相同解码和同一任务集。否则结论应写成“模型系统表现不同”，而不是“模块 X 证明更好”。

## 17.15 配置差异如何传导到显存

对 decoder-only 模型，单层 KV cache 的粗略字节数可写成：

~~~math
M_{\mathrm{KV/layer}}
\approx 2T H_{\mathrm{kv}}d_h b
~~~

其中 T 是已处理 token 数，H_kv 是 KV head 数，d_h 是 head dimension，b 是每个元素的字节数。GQA/MQA 通过减少 H_kv 降低 cache，而不会直接减少 query head 数。滑窗则限制一部分层的有效 T，但可能需要 global 层或检索补偿。

这类公式能把架构配置和系统指标接起来。两个模型参数量相近，若 KV heads、位置策略、窗口和 dtype 不同，长上下文并发能力可能完全不同。

## 17.16 品牌比较要拆成可复现实验

GPT、LLaMA 和 Mistral 不是三个单一固定结构，而是一组随版本变化的模型家族。比较时要先锁定 checkpoint、revision、tokenizer 和 prompt template，再把 attention、FFN、position、context training 和后训练单独列出。

若使用闭源模型，公开资料通常只能支持接口、上下文声明、评测条件和产品行为，不能支持未披露的层数、专家路由或训练 recipe。技术判断应写成“公开资料显示”或“实验观察到”，不应把架构猜测当作事实。

## 17.17 从架构选择到部署选择

小模型、高吞吐和低延迟场景可能偏好滑窗、GQA、量化和成熟 kernel；复杂多证据任务可能更重视 full attention、长上下文训练和引用能力。Mistral 类高效设计的价值不只是某个窗口公式，而是把模型规模、训练配方和 serving 目标一起设计。

面试中可以用 Pareto 语言总结：固定质量验收条件后比较显存和延迟，固定成本约束后比较任务质量，再把迁移、生态和故障恢复纳入长期决策。这样不会把品牌名变成未经条件限定的排名。

## 17.18 用配置表把比较落到可计算对象

比较 LLaMA、GPT 或 Mistral 风格模型时，至少记录层数 `L`、hidden size `d`、query/KV head 数、head dimension、FFN 中间维度、词表、上下文、位置方案、归一化、激活、MoE 配置、dtype 和 tokenizer。只写家族名会把架构事实、训练配方和产品能力混在一起。

对单层 decode，KV payload 的教学估算为：

```math
M_{\mathrm{KV}}
\approx 2BTH_{\mathrm{kv}}d_hb
```

GQA/MQA 通过降低 `H_kv` 改变缓存和带宽压力；它不会自动提高知识质量。SwiGLU、层数和 hidden size 则影响参数量、FLOPs 和表达能力。比较时要把“更省 KV”与“更强模型”分开测量。

## 17.19 模块差异和训练配方如何拆开

一个模型分数领先，原因可能来自数据量、数据质量、继续预训练、指令数据、RL、长上下文 curriculum、tokenizer 或评估协议，而不只是 attention/FFN 结构。公平对照需要固定参数规模、训练 token、数据混合、优化器和上下文预算，或者明确这是端到端系统比较而不是架构消融。

对公开闭源模型，通常只能确认发布材料明确披露的结构和能力，不能把社区反推的层数、训练 token 或内部 router 当成事实。对开源模型，也要绑定 commit、config、权重 revision 和 tokenizer；同名模型的不同量化和 chat template 可能不是同一个实验对象。

## 17.20 从架构选择到部署选择

选择 GQA、MQA、滑窗或 MoE 时，服务目标同样重要。低延迟短输出可能受 decode 带宽主导，长 prompt 批处理可能受 prefill 计算主导，多租户 Agent 还要考虑 prefix cache、工具事件和 KV 增长。应把质量、显存、吞吐、p99、故障回退和模型协议一起放进决策矩阵。

一个模型若需要自定义 kernel 或特殊 tokenizer，迁移到普通 serving runtime 的成本可能超过它在单卡 benchmark 上的收益。架构比较最终要落到 Pareto 前沿：在目标硬件、任务和 SLO 下，哪些方案没有被另一方案同时在质量、成本和可靠性上压倒。

## 17.21 一个可计算的模型架构账本

比较模型家族时，最好把每个结论写成一条可复核的账。以单请求 decode 为例，先从配置得到每 token 的 KV payload：

```math
m_kv_per_token = 2 L H_kv d_h b
```

再把上下文长度、并发和可用显存代入：

```math
M_kv_total = B T m_kv_per_token + M_metadata + M_slack
```

如果 Mistral 风格模型使用滑动窗口，不能简单把 `T` 全部替换成窗口大小。需要知道哪些层是局部的、是否存在 global token、prefix cache 是否能跨窗口复用，以及工具结果是否会触发全局回看。LLaMA 风格的 GQA 可以直接减少 `H_kv`，但质量变化仍需要同一训练预算下的消融。GPT 风格的闭源模型如果没有公开 `H_kv`，只能记录未知，不能从 decode 速度反推唯一结构。

一个完整的账本还要包括 tokenizer、上下文训练长度、chat template、量化、kernel、硬件、采样和评测 harness。只要其中一项不同，最终的 tokens/s 或 benchmark 分数就不再是纯架构比较。书写时应把“可观察的配置事实”“在当前环境测到的现象”和“由结构推断的假设”分成三列。

## 17.22 从模块差异到任务失败模式

架构选择的价值最终要通过失败类型体现。GQA/MQA 主要影响 KV 读取和缓存容量，可能在长输出高并发时改善吞吐，却不一定解决复杂推理；滑动窗口主要改变远距离信息路径，可能让局部代码补全更便宜，却在跨文件引用上需要 global token 或 RAG；SwiGLU/FFN 影响 token 内变换和参数预算，不能直接解释所有长上下文能力。

因此，实验要按任务切片：局部 next-token、远距数字复制、跨段引用、工具 JSON、长输出 decode 和混合长度压测。若质量只在跨段引用下降，应优先查窗口和位置；若质量保持而 p99 下降，应查 cache/调度；若小模型在短任务胜出而长任务失败，可能是训练数据或上下文 curriculum，不应把结果简化为“某家族架构更强”。

## 17.23 迁移成本也是架构成本

一个模型在论文或单卡 benchmark 上占优，不等于它能低成本迁移到现有服务。自定义 tokenizer、chat template、RoPE 变体、KV layout、MoE dispatch、量化格式和 speculative draft 接口都可能需要 engine 适配。适配工作还包括 golden logit、prefix cache key、streaming stop reason、监控字段和回滚 artifact。

可以把迁移决策写成：

```math
U_migrate = Delta_Q - lambda_1 Delta_C_serve
             - lambda_2 C_adapter - lambda_3 R_unknown
```

`Delta_Q` 是在目标任务上的质量变化，`C_adapter` 是适配和维护成本，`R_unknown` 是公开资料不足或实现未验证带来的风险。它不是财务模型，而是防止选型只盯着参数和榜单。对闭源模型，应把协议限制、速率限制、数据保留和供应商切换成本一起记入账本。

## 17.24 小结

GPT 提供统一生成目标和规模化主线，LLaMA 形成现代开源 decoder 基线，Mistral 把高效 attention 与小模型性能推向工程实践。真正可迁移的知识不是记住品牌，而是能从 head、position、FFN、mask、训练和 serving 预算解释每个设计的收益和代价。
