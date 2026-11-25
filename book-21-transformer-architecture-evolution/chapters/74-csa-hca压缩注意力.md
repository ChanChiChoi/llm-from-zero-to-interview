# 第 74 章 CSA/HCA：压缩历史与 query 读取路径

本章讲 DeepSeek-V4 技术报告里的两种长历史 attention：Compressed Sparse Attention（CSA）和 Heavily Compressed Attention（HCA）。两者都会先把多个 token 的 KV 表示压成更少的新条目。区别在后续读取：CSA 用 Lightning Indexer 为当前 query 选出少量压缩条目；HCA 压得更粗，直接读取全部已完成的压缩条目。两者还会并入近期的原始 token 窗口。模块定义依据[报告 §2.3](https://arxiv.org/html/2606.19348v1#S2.SS3)。

理解这套结构时，要把三个动作分开：压缩器决定一个历史块怎样变成新表示，indexer 决定 CSA 当前读取哪些压缩表示，core attention 再用 query 对候选 key/value 计算输出。压缩比例、索引分数和 attention 权重分别属于不同阶段。本章聚焦机制；DeepSeek-V4 的具体层表、百万 token 成本数字、低精度和参考实现放在[第 77 章](77-csa-hca从压缩kv到百万上下文.md)。

## 74.1 长上下文要同时控制历史存储与每步读取

普通自回归 attention 会为历史 token 保留可供后续 query 读取的 KV。上下文变长时，长期保存的条目增加，每次 decode 也需要处理更多历史候选。CSA/HCA 同时改变这两件事里的不同部分：压缩减少长期表示的条目数，CSA 的 query 级选择进一步限制主 attention 实际读取的压缩条目数。

### 74.1.1 传统 KV 逐 token 保存，改变宽度不等于缩短历史序列

GQA 或 MQA 主要减少每个 token 要保存的 KV 头数；MLA 主要把每个 token 的 KV 表示换成较窄的 latent。它们可以降低单条目大小，却仍然按 token 逐步积累历史。CSA/HCA 的 block compression 沿另一个轴工作：把一组位置变成一条记录，直接降低长程序列上的条目数。

这不表示前几种方法不能与压缩 attention 组合。它们描述的是不同设计选择：条目有多宽、每个 token 保存几份，以及多少个历史位置合成一条记录。

### 74.1.2 表示压缩与 query 级选择针对不同成本

表示压缩把多个 token 的内容合成更少的 KV 条目，因此它改变长期存储对象，也可能混合掉原始位置之间的差别。query 级选择从已有候选中挑一部分送入 attention，限制这次读取哪些条目；它本身不说明候选表示是否仍保存原文细节。

CSA 把两者接在一起：先形成相对细的压缩历史，再按 query 选择压缩条目。HCA 则用更大的压缩跨度缩短历史后，密集读取所有可用压缩条目。这个区别是后续理解两种数据流的坐标。

### 74.1.3 CSA/HCA 是 DeepSeek-V4 中的具体混合注意力方案

CSA 和 HCA 在这里专指 DeepSeek-V4 报告中的两个模块名称。报告把它们组合进混合 attention 架构，并分别定义压缩率、索引选择和读取路径；其他模型即使使用相同缩写，也需要按自己的论文或代码核对，不能直接套用本章公式。

下文用 $n$ 表示原始历史长度，$m$ 表示 CSA 的步幅/压缩率，$m'$ 表示 HCA 的压缩跨度，$k$ 表示 CSA 每个 query 选取的条目上限，$w$ 表示近期原始窗口的长度。这些符号帮助比较结构，具体模型值留在第 77 章。

## 74.2 Learned pooling 怎样形成新的压缩 KV

CSA 和 HCA 都使用可学习的加权 pooling。压缩器先把每个输入位置映射成内容向量，再为每个输出特征维计算块内位置权重，最后把不同位置的内容合成一个向量。它因此可以让压缩向量的不同维度分别保留不同位置的信息。逐维归一化的直观解释也可参照 [Stanford MS&E 319 Lecture 5](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf) 的 coordinatewise pooling 例子；V4 的具体双流结构以原报告为准。

### 74.2.1 content 与 gate logits 分头投影

令块内位置 $j$ 的 hidden state 为 $h_j\in\mathbb{R}^{d}$。压缩器生成内容 $u_j$ 和打分 $z_j$：

```math
u_j=h_jW_U,\qquad z_j=h_jW_Z+b_j,
\qquad u_j,z_j\in\mathbb{R}^{c}.
```

$W_U$ 决定位置能提供哪些压缩特征，$W_Z$ 决定这些特征会从哪些位置汇入。$b_j$ 是按块内相对位置索引的可学习偏置，因此压缩器能学到某个偏移位置是否更适合提供特定维度。内容投影和打分投影承担不同作用，不能把它们合并成“对 KV 做平均”。

### 74.2.2 每个特征维在块内位置间学习加权

对于块 $\mathcal{B}$ 和输出维 $r$，权重沿位置维做 softmax：

```math
\alpha_{j,r}=\frac{\exp(z_{j,r})}{\sum_{\ell\in\mathcal{B}}\exp(z_{\ell,r})},
\qquad
\bar{c}_{r}=\sum_{j\in\mathcal{B}}\alpha_{j,r}u_{j,r}.
```

每个特征维都有一组位置权重，并且该维的权重和为 $1$。因此一个压缩向量可以在第一维更多保留位置 A 的内容、在第二维更多保留位置 B 的内容；模型不是为整个 token 块选出唯一代表位置。

例如，教学用的两个位置内容设为 $u_1=(2,8)$、$u_2=(6,4)$。假设第一维权重为 $(0.75,0.25)$，第二维权重为 $(0.25,0.75)$，结果就是 $\bar{c}=(3,5)$。压缩后的向量由两个位置共同形成，但不同维度侧重不同位置。这组数字只解释逐维 pooling，不是论文测量值。

### 74.2.3 压缩条目本身参与 attention，并非原始 KV 索引

压缩器的输出就是后续 attention 读取的 KV 表示。它不只是一个指向原始 token 范围的地址，也不保证能从输出精确恢复组成它的每个原始 key/value。若压缩期间丢掉了两个 token 的差别，后续 query 只能根据压缩向量继续计算。

这一点区分了“先用摘要搜索，再回到原文 KV 读取”与 CSA/HCA 的主路径。后者将 compressed entry 作为注意力对象；在 DeepSeek-V4 的完整结构中，近期窗口另行保留一段未压缩 KV，但它不能还原任意远程压缩块的原 token。

## 74.3 CSA：重叠压缩后选择少量全局条目

CSA 的长程路径可概括为：生成重叠 compressed KV 和另一组 compressed indexer keys；当前 query 对因果可用的 indexer keys 打分；top-$k$ 选中的主 KV 条目再进入 core attention。两次压缩使用相似的块组织方式，但 indexer keys 和主 attention 的 compressed KV 是不同表示。

### 74.3.1 a/b 两条投影令压缩窗口跨越相邻块

CSA 为内容与 gate logits 各准备两条投影流，记作 $a$、$b$。对第 $i$ 个压缩条目，$a$ 流处理当前位置对应的 $m$ 个 token，$b$ 流处理前一个 $m$ 个 token。将两条流的内容向量记作 $C^a,C^b$，将拼接的两块上逐特征归一化得到的权重记作 $S^a,S^b$，则压缩结果为：

```math
\bar{c}_i^{\mathrm{CSA}}
=\sum_{j=mi}^{m(i+1)-1}S_j^a\odot C_j^a
+\sum_{j=m(i-1)}^{mi-1}S_j^b\odot C_j^b.
```

$S^a,S^b$ 由两条流的 gate logits 和位置偏置共同产生；归一化跨越这两段中的位置。[DeepSeek-V4 报告 §2.3.1](https://arxiv.org/html/2606.19348v1#S2.SS3)给出这一压缩定义。$i=0$ 时不存在前一个块，对应的 $b$ 流以零权重填充。这样的定义让相邻压缩结果可以共享边界块，同时仍然只使用当前及过去的 token。

### 74.3.2 Overlap 以 $m$ 为 stride，条目数仍约为 $n/m$

注意“一个条目查看多少位置”和“压缩序列每次前进多少位置”不是同一回事。一个 CSA 条目最多合并当前块和前块，共覆盖 $2m$ 个原始位置；下一个条目则前进 $m$ 个位置。因此输出条目数近似为 $n/m$，不是 $n/(2m)$。

用 $m=2$ 的教学位置索引看重叠关系：

| 压缩条目 | $a$ 流处理的块 | $b$ 流处理的块 |
| --- | --- | --- |
| 0 | token 0–1 | 起始 padding |
| 1 | token 2–3 | token 0–1 |
| 2 | token 4–5 | token 2–3 |

每条流都有自己的可学习内容和 pooling 权重。表中的范围只说明块怎样重叠，不代表一个 token 在两条流里使用相同数值。

### 74.3.3 Lightning Indexer 使用独立压缩 key 对候选条目打分

Indexer 为候选历史再形成 compressed indexer keys。当前 query 先产生一个低秩 query latent，再由它构造 indexer query heads；这些 heads 计算与 indexer keys 的相似度，经过 ReLU 和当前 query 生成的 head 权重合并为一个条目分数。概括其核心分数可写为：

```math
I_{t,s}=\sum_{h=1}^{n_h^I}w_{t,h}^I
\,\mathrm{ReLU}\!\left(q_{t,h}^I\cdot K_s^{I\mathrm{Comp}}\right).
```

这里 $s$ 是压缩条目的索引，$h$ 是 indexer head。Indexer keys 有独立的投影与压缩结果，不是主 attention 中 $\bar{c}^{\mathrm{CSA}}$ 的别名。论文将 indexer 计算设计为低维检索路径；第 77 章再核对配置维度和低精度实现。

### 74.3.4 top-k 选压缩条目，不等于选出正确证据

对因果上可用的分数取 top-$k$，得到主 attention 本轮要读取的压缩条目集合。假设教学分数为 $(0.2,1.4,0.7,0.9)$ 且 $k=2$，选中的是索引 $1$ 与 $3$；这只演示排序步骤，并不表示这两个块一定包含任务答案。

Indexing 与核心 attention 也不能混成一组权重：indexer 分数先决定哪些压缩条目进入候选集合；core attention 随后依据主 query/key 分数，在候选条目与近期原始 KV 之间计算注意力。top-$k$ 是模型的内部稀疏选择，不是 gold token 标签或语义证据充分性的证明。

## 74.4 HCA：粗粒度压缩后密集读取压缩序列

HCA 使用与 CSA 相同类型的逐特征 learned pooling，但用一条内容流、一个不重叠块和更大的压缩跨度 $m'$。它不建立 CSA 的 Lightning Indexer，而是把全部可用 HCA compressed entries 交给 core attention。

### 74.4.1 单 stream 与非重叠块形成每 $m'$ 一个条目

一个 HCA 条目由不重叠的 $m'$ 个 token 投影而来，gate logits 加上块内位置偏置，再为每个输出维沿块内位置归一化。用前节定义的内容向量 $u_{j,r}$ 和权重 $\alpha_{j,r}$ 表示，可写成：

```math
\bar{c}_{i,r}^{\mathrm{HCA}}
=\sum_{j=m'i}^{m'(i+1)-1}\alpha_{j,r}u_{j,r}.
```

连续块互不重叠，故长度 $n$ 的历史形成约 $n/m'$ 个长程条目。与 CSA 的双流窗口不同，当前条目的信息范围就是这个不重叠块。[报告 §2.3.2](https://arxiv.org/html/2606.19348v1#S2.SS3)给出 HCA 的单流 pooling 定义。

### 74.4.2 Dense 作用在可用压缩条目上

HCA 不用 indexer 或 top-$k$；query 对所有已经形成且因果可见的 HCA 条目做 dense attention。比如一个教学序列有 $12$ 个位置、$m'=3$，最多形成 $4$ 个压缩条目。位于尾端的 query 可以访问这 $4$ 条记录；它读取的是四个摘要向量，不是十二个原始 token 的 KV。

所以“dense HCA”准确地说是“在压缩域内 dense”。它消除了 top-$k$ 漏选这一选择步骤，却没有消除学习式压缩带来的细节混合或信息损失。

### 74.4.3 与 CSA 的差异在压缩尺度和选择阶段

CSA 保留较细的压缩粒度，先打分并选少量全局条目；HCA 以较大跨度形成更短的序列，再完整读取其中可见的压缩条目。前者用索引搜索换较细的历史单元，后者用更粗的历史单元换去掉 top-$k$ 阶段。

在 DeepSeek-V4 中，两类层按配置组合，但具体压缩率、层排布和 top-$k$ 会随 Flash、Pro 等模型版本变化。这里不把某个模型的一组数值写成 CSA/HCA 的定义；对应模型配置见[第 77 章](77-csa-hca从压缩kv到百万上下文.md)。

## 74.5 长程压缩与近期原文共同进入 attention

只有压缩器还不足以构成报告里的完整 attention 输入。CSA/HCA 都会把长程 compressed KV 与短期未压缩 KV 结合，让 query 同时有一条压缩历史路径和一段精细的近期路径。

### 74.5.1 未完成压缩块只有在其完成后才可见

压缩条目代表一个完整块；块尚未结束时，压缩器不能把含未来 token 的结果提供给较早的 query。decode 到块末尾之后，该条目才进入因果可见的 compressed history。具体边界需要与 query 的位置索引和 causal mask 一致。

### 74.5.2 局部窗口提供原始近期 token 和块边界内因果访问

模型另存长度为 $w$ 的近期原始 KV 窗口。它让 query 逐 token 读取近处信息，并覆盖尚未完成压缩的尾部位置。因此局部路径既保持短程细节，也使当前压缩块尚未完成的部分仍能按因果次序参与 attention；窗口外旧历史则主要通过压缩条目访问。

### 74.5.3 Shared-KV MQA 让各 query heads 读取同一压缩 entry

主 attention 通过多个 query heads 计算不同的查询，但 compressed KV entry 在这些 query heads 间共享，并同时作为 key 和 value。这是 shared-KV MQA 的读法：query heads 不等于 KV heads，使用多少个 query head 也不意味着要为每个 head 复制一份压缩历史。

### 74.5.4 local 与 compressed 候选共用核心 attention 归一化

CSA 将 top-$k$ 选中的压缩条目与 local KV 合并为当前 query 的候选；HCA 则把全部因果可用的压缩条目与相同类型的 local KV 合并。它们进入同一次核心 attention 计算。不能把两类候选各自做完 softmax 后再随意相加；这样会改变二者竞争同一注意力预算的方式。

KV entry 的 RoPE 位置、query/KV 归一化、attention sink 和 grouped output projection 也会影响完整 V4 block，但不决定 CSA/HCA 的定义差别。报告把这些列为核心压缩/选择之外的共同细节；模型配置和位置处理由第 77 章及本册相应位置编码章节承接。

## 74.6 CSA/HCA 混合路径的成本与误差边界

压缩倍率不能单独代表读取成本或答案质量。应该分别数出长期保存的条目、indexer 扫描的候选、core attention 读取的条目，再把压缩产生的表示损失与 top-$k$ 选择可能造成的漏选分开。

### 74.6.1 存储条目数、主读取数与 indexer 搜索不是同一成本

在长度为 $n$ 的历史、单个 decode query、忽略边界项时，条目数量可粗略表示为：

| 路径 | 长程存储条目 | 每 query 的长程主读取 | 另有的近期路径 |
| --- | ---: | ---: | ---: |
| CSA | 约 $n/m$ | 至多 $k$ 个选中条目 | 约 $w$ 个原始 KV |
| HCA | 约 $n/m'$ | 约 $n/m'$ 个已完成条目 | 约 $w$ 个原始 KV |

CSA 的主 attention 只读取 top-$k$，但 indexer 仍需给因果候选压缩 key 打分；因此 $k$ 不能代替 indexer 搜索量。表格只数 entry，不包含每个 entry 的维度、精度、索引存储、kernel 吞吐与实际 memory traffic，不能直接换算成字节或延迟。

### 74.6.2 CSA 面临压缩损失与 top-k 漏选，HCA 面临更粗的表示损失

CSA 有两个连续的误差来源：压缩器可能将区分 query 所需的特征合到同一向量；indexer 也可能没有选出包含相关信息的 compressed entry。HCA 没有稀疏选择步骤，但更大的压缩块仍可能丢失细节。两者的 local window 只保留近期 token 的精细访问，不会恢复任意远程块中的原始 token。

评估时应分别检查压缩表示是否保留任务相关信息、CSA indexer 是否把相关压缩条目送入候选、core attention 是否能用这些条目完成任务。仅报告 pooling 的压缩比例或最终平均分，无法定位误差来自哪一步。

### 74.6.3 压缩倍率、召回和延迟需在对齐条件下测量

改变 $m$、$m'$ 或 $k$ 会同时改变历史表示数量、条目粒度和 query 计算路径。比较方案时，要固定模型、训练、任务、序列长度、精度、实现和硬件；分别报告质量、indexer/search 工作、core attention 工作、KV 占用与实测延迟。理论条目数是分析起点，不是 kernel 加速倍数。

### 74.6.4 百万 token 主张属于完整 V4 配置，详数由第 77 章承担

DeepSeek-V4 报告的百万 token 效率来自具体模型的 CSA/HCA 层配置、shared-KV 与低秩路径、位置处理、滑动窗口、精度和实现共同作用。报告给出的 FLOPs/KV 比率须保留指定模型及 baseline，不能单独归因于 CSA 或 HCA 的 pooling 算子，也不保证其他任务或硬件有相同结果。第 77 章按公开版本展开这些具体证据。

研究版本和证据边界见[CSA/HCA 机制研究记录](../../research/csa-hca-mechanism-evidence-2026-11-24.md)。
