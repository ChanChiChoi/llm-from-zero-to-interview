# 第 63 章 Gated MLA：潜变量缓存与通道输出门

本章解释 Kimi K3 报告中的 Gated MLA。它包含三步：为每个历史 token 保存一个低维 KV latent；用全局 attention 从这些 token 位置读取信息；再用当前输入生成的逐通道 gate 调制 attention 输出。低秩压缩改变每个位置存什么，output gate 改变读出如何传到后续层，两者作用在不同位置。

第 70 章讨论 latent cache 的通用压缩与保真，第 64 章讨论 NoPE 的顺序信息，第 65 章比较递归与全局注意力的层间组合，第 88 章讲 Kimi K3 全模型。本章只保留理解 Gated MLA 所需的基础机制。

## 63.1 Gated MLA 改变历史表示和输出调制

显式 attention cache 为每个历史 token 保留 key 和 value。即使采用 GQA 减少 KV head 数，cache 仍随历史长度增长。Multi-head Latent Attention（MLA）把同一 token 的 key/value 信息联合压到较小的 latent，再由查询和投影读取；它保留 token 位置，不会把整段历史合并成一个递归状态。

理解 Gated MLA 时，要把三个对象分开：

1. **KV latent：** 每个历史 token 的压缩表示，也是 MLA cache 的主体。
2. **全局 attention：** 当前 query 仍能对历史位置计算注意力权重。
3. **输出 gate：** attention 汇总后，按当前 token 的输入调制输出通道。

Kimi K3 v2 把 Gated MLA 作为其全局 attention 模块之一；该名称中的 gate 指输出调制，不能据此推断模型在显式 KV 与 latent KV 两条缓存路径之间选路。[Kimi K3 技术报告 v2，§2.1.2](https://arxiv.org/html/2607.24653v2)。

## 63.2 MLA 将每个 token 的 key/value 联合压到 latent

设 attention 输入为 $x_t\in\mathbb{R}^{d}$。MLA 先用下投影得到该 token 的 KV latent，再分别映射为供各 head 使用的 key 与 value：

```math
c_t^{KV}=W^{DKV}x_t\in\mathbb{R}^{d_c},
\qquad
k_t^C=W^{UK}c_t^{KV},
\qquad
v_t^C=W^{UV}c_t^{KV}.
```

$d_c$ 是 KV latent 的维度。标准显式 KV cache 为每个 token 保存多个 head 的 key 与 value；MLA 则在 cache 中保存共享的 $c_t^{KV}$，而投影矩阵负责把它映射到当前注意力计算所需的表示。这里的 $c_t^{KV}$ 是逐 token latent，不是把整段历史压成一个摘要向量。

### 63.2.1 共享 latent 与低秩 key/value 投影

MLA 的压缩发生在 head-specific key/value 展开之前。所有 query head 通过各自的 key/value 投影使用同一个 token latent，因此 cache 的理想 payload 主要取决于 $d_c$，而不是显式存储的 head 数乘 head 维度。

训练和推理的计算图可以包含下投影、latent cache、key/value 上投影和 attention；推理阶段实际物化哪些中间张量取决于实现。公式说明模型表示关系，不直接说明某个运行时 kernel 的 buffer 布局。

### 63.2.2 Decode 时吸收投影的代数视图

对某个 head，key 上投影参与 query-key 内积：

```math
(q_{t,i}^{C})^\top k_{j,i}^{C}
=(q_{t,i}^{C})^\top W_{UK}^{(i)}c_j^{KV}
=\big((W_{UK}^{(i)})^\top q_{t,i}^{C}\big)^\top c_j^{KV}.
```

因此可以把 key 上投影折入当前 query 的变换；value 上投影也可与 attention 输出投影组合。DeepSeek-V2 论文用这一代数吸收解释 decode 时可以直接围绕 KV latent 计算，而不必先恢复每个 token 的完整 key/value。位置变换可能限制这种折叠方式，见下一节。[DeepSeek-V2 v5，§2.1.2](https://arxiv.org/html/2405.04434v5)。

这是线性投影的等价重排，不代表所有实现都会使用相同的融合 kernel，也不代表低秩 latent 消除了 attention 对历史 token 的读取。

## 63.3 位置路径按模型版本区分

低秩压缩会改变 key/value 的表示方式，但模型仍须处理 token 位置。位置编码与 latent 投影怎样组合，不能只由 MLA 这个名称推断。

### 63.3.1 DeepSeek-V2 的 decoupled RoPE 分支

DeepSeek-V2 把内容 key 与位置 key 分开：内容 key 从 KV latent 投影得到，另有按位置旋转的共享 RoPE key；query 也分成内容与 RoPE 部分。每个 head 的打分结合两部分：

```math
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],
\qquad
k_{j,i}=[k_{j,i}^{C};k_j^{R}].
```

这一做法保留低秩 KV cache 的内容压缩，同时把位置相关分量放在独立路径。于是该类 cache 的理想每 token 元素量包含 $d_c$ 和共享位置 key 维度，不能只数 $c_t^{KV}$。[DeepSeek-V2 v5，§2.1.3](https://arxiv.org/html/2405.04434v5)。

### 63.3.2 Kimi K3 Gated MLA 的 NoPE 配置

Kimi K3 v2 报告说明其 MLA 层使用 No Position Encoding（NoPE），不对这些层的 query/key 施加显式位置变换；报告把位置敏感、近期性建模主要放在交错的 KDA 层，并将 MLA 层用于全局内容交互。该解释属于 Kimi K3 的整体架构设计，不能外推为所有 MLA 的位置方案。[Kimi K3 技术报告 v2，§2.1–2.1.2](https://arxiv.org/html/2607.24653v2)。

NoPE 也不表示整个网络没有顺序信息：进入 MLA 的 hidden state 已经过前序层计算，而其他注意力路径也可能携带顺序结构。NoPE 的一般性质和证据边界见[第 64 章](64-nope与隐式顺序.md)；KDA/MLA 的层间分工见[第 65 章](65-hybrid-attention局部全局与递归状态.md)。

## 63.4 Gated MLA 在 attention 输出后施加通道门

令 $\tilde{o}_t$ 为当前 token 对历史 latent 完成全局 attention 后、尚未施加输出 gate 的结果。Kimi K3 报告给出的计算为：

```math
y_t=W_o\left[\operatorname{Sigmoid}(W_gx_t)\odot\tilde{o}_t\right].
```

$W_gx_t$ 根据当前输入产生 gate，sigmoid 将各通道缩放到 $0$ 与 $1$ 之间；$\odot$ 表示逐通道相乘；随后 $W_o$ 把调制结果投影回模型表示空间。报告称 Kimi K3 使用输入依赖的 full-rank gate。[Kimi K3 技术报告 v2，Eq. 7](https://arxiv.org/html/2607.24653v2)。

### 63.4.1 输入依赖的逐通道输出公式

这个 gate 作用于已经聚合好的 attention 输出。不同输出通道可以根据当前位置的输入得到不同缩放，因此 gate 能控制哪些全局 attention 通道更强地传给后续计算。它不需要改写历史 latent，也不改变每个历史位置的 KV cache 大小。

### 63.4.2 输出门不是 latent/explicit 路由

公式中的 gate 并未在“latent 输出”与“显式 KV 输出”之间做加权混合。Kimi K3 的 Gated MLA 路径本身对 latent cache 做 attention，再调制 attention result。gate 也不直接选择历史 token、替代 softmax attention 权重或删除缓存条目。

这与 KDA 的递归状态门作用位置不同：KDA 的 decay 改变旧状态保留量，KDA 的 output gate 调制递归读出；本章的 Gated MLA output gate 调制全局 attention 输出。KDA 的状态转移见[第 61 章](61-kda递归注意力.md)。

## 63.5 K3 配置中的 cache 账本与代价边界

按每个 attention layer、每个请求计算，令历史长度为 $T$，元素字节数为 $b$，显式 KV 有 $H_{kv}$ 个 KV head、每 head 维度为 $d_h$。忽略 page metadata、padding 和临时 workspace：

```math
M_{\mathrm{GQA}}\approx 2TH_{kv}d_hb.
```

采用共享 KV latent 的 MLA，若另存一条每 token 共享的位置 key，理想 payload 近似为：

```math
M_{\mathrm{MLA}}\approx T(d_c+d_R)b,
```

其中 $d_R$ 是该位置分支的维度。若具体版本使用 NoPE 并且没有这条显式位置 cache，则 MLA 主体近似为 $Td_cb$。这些是按张量维度得到的 payload 估算，不包含模型其他层的 KDA 状态。

### 63.5.1 位置分支如何进入 per-token payload

DeepSeek-V2 风格的 decoupled RoPE MLA 每个位置还保存共享位置 key，理想元素量约为 $T(d_c+d_R)$；使用 NoPE 且无显式位置分支时约为 $Td_c$。两者都保留 token 位置，cache 随 $T$ 线性增长；查询仍对历史位置执行全局 attention。通用 cache 账本见[第 70 章](70-latent-cache压缩历史表示.md)。

### 63.5.2 Kimi K3 公布 rank 的理想估算

Moonshot 的固定 Kimi K3 config 披露 24 个 Gated MLA 层和 $d_c=512$ 的 KV latent rank。跨 MLA 层，每个历史位置需要保留的理想 latent 元素数为：

```math
N_{\mathrm{latent/token}}=N_{\mathrm{MLA}}d_c=24\times512=12{,}288.
```

若 cache 元素用 2 bytes 存储，这相当于 24,576 bytes（24 KiB）/token，尚未计入 KDA 递归状态、页元数据、对齐和临时 workspace。这个数字是按公开 rank 和层数作出的理想估算，不代表该模型的实测 cache dtype、请求峰值显存或 serving 性能。

## 63.6 Token 级 latent cache 与递归状态

MLA 将每个位置压成一个 latent，仍保留按 token 位置参与全局注意力的路径；历史长度增加时，latent cache 也随之增长。KDA 则把历史写入递归状态，不再为每个历史 token 留一个独立 latent 槽位。因此两者解决的资源问题不同：MLA 减少每位置 cache 宽度，KDA 改变历史状态随序列增长的方式。

低秩投影也不保证任意 key/value 都能无损重构。若任务依赖精确数字、实体关系或很小的差异，要检查相应模型和训练设置中的质量证据；不能从 cache 压缩比推出全任务精度或任意长上下文检索。通用压缩保真、任务切片与读放大由[第 70 章](70-latent-cache压缩历史表示.md)负责。

## 63.7 Kimi K3 证据能支持到哪里

Kimi K3 v2 报告定义了 Gated MLA 的 NoPE global-attention 路径和 Eq. 7 输出门；Moonshot 的固定模型 config revision f831ab66814297da540d832a5235f8e904f29d06 列出 kv_lora_rank=512、mla_use_nope=true、mla_use_output_gate=true，并记录了模型层配置。[固定 config](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)。

官方模型资料披露 Kimi K3 使用 69 层 KDA 与 24 层 Gated MLA；3:1 层间模式及末端全局层的完整结构由第 65 章和[第 88 章](88-kimi-k3-kda-stable-latentmoe与百万token-agent.md)展开。本章只把这些版本事实用于定位 Gated MLA 的模型背景。

论文公式支持“存在输入依赖的输出 gate”这一结构判断。整模型 benchmark 同时受 KDA、MLA、训练、数据、MoE、后训练和系统实现影响；没有对应的单独对照时，不能把整模型成绩单独归因于这个 gate。
