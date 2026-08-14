# 第 64 章 NoPE：没有显式位置变换，不等于没有顺序

## 64.1 一个常见但错误的推断

看到模型资料写 NoPE 或 no positional embedding，有人会立即说“模型不知道顺序”。这把一种位置注入方式误认为顺序信息本身。即使某个 attention 分支不使用 RoPE，causal mask、递归 state、局部窗口、衰减、卷积或层间 pattern 仍然可以把顺序带进计算。

正确的问题是：哪个路径没有显式位置变换？顺序从哪里进入？模型能否区分方向、距离和跨段位置？这些问题要由 config、技术报告或可重复实验回答。

## 64.2 没有位置时 attention 的性质

没有 mask 和位置信号的 self-attention 对输入排列具有 permutation equivariance：交换输入 token，输出会随之交换。因果 mask 改变了可见集合：位置 `t` 只能看到 `j\le t` 的 token。

```math
A_{t,j}=\begin{cases}q_tk_j^{\top}/\sqrt{d_h},&j\le t\\-\infty,&j>t\end{cases}
```

因此模型知道“哪些 token 已经发生”，但不一定知道两个 token 之间的精确距离。递归状态又会按 token 到达顺序更新，产生另一种隐式顺序。

## 64.3 一个交换实验

输入 `A B` 和 `B A`。在 causal attention 中，第二个位置看到的 token 集合都包括当前和历史，但第一步的 hidden 不同，第二步的 query/key/value 也不同，所以输出通常能区分顺序。

但把 `A` 和 `B` 分隔几万 token，问题就变成距离和记忆，而不是简单顺序。模型可能能判断 A 在 B 前面，却不能准确回答 A 出现在哪个版本或哪一段。顺序敏感性、距离分辨率和长期归因要分别测。

## 64.4 NoPE 与 RoPE 的取舍

RoPE 对 Q/K 做位置相关旋转，把相对位置信息注入点积：

```math
q'_p=R(p)q_p,\qquad k'_r=R(r)k_r
```

点积因此包含与 `p-r` 有关的相位关系。它表达距离很直接，但长上下文扩展需要处理频率、插值和外推。

NoPE 或部分 NoPE 可以减少显式旋转路径，适合某些递归/混合结构；代价是更多依赖 causal visibility、state update、训练数据和层 pattern。不能从“没有 RoPE”推断“不需要任何位置机制”。

## 64.5 长上下文的实际问题

如果任务要求“找出最新版本中的配置”，顺序、时间和版本都很重要。NoPE 路径可能依靠训练中形成的顺序统计，也可能依靠 global layer 或状态衰减。随着干扰长度增加，远端证据可能被压缩或遗忘。

评测要覆盖开头/中间/结尾位置、同名实体、版本冲突、顺序交换和不同距离，而不是只测一个 short prompt。

## 64.6 实现审计

审计时检查 attention mask 是否真的 causal，position id 是否仍被其他模块使用，递归 state 的 step 是否每个 request 独立，local window 是否在边界重置，packed sequence 是否正确插入 reset。源码未公开时，不要用一两个输出样例反推完整架构。

## 64.7 一个无 RoPE toy 实验

实现一个只有 token embedding、causal mask 和两层 attention 的 toy 模型。训练顺序复制、距离分类和版本选择三个任务。分别在训练长度内、两倍长度和插入干扰后测试。结果可以说明 causal mask 提供了方向，但不能说明模型拥有无限距离精度。

## 64.8 失败模式

把 NoPE 扩大为“没有任何位置机制”；padding 改变递归更新次数；跨请求 state 污染；local window 边界 off-by-one；训练和推理 position/reset 不一致；把顺序可区分误读成长上下文可检索；社区帖子成为唯一架构证据。

## 64.9 面试回答与练习

回答“NoPE 会不会失去位置信息”时，应说某个分支没有显式 position embedding/rotation，不等于计算没有顺序；causal mask、递归更新、局部/全局层和训练数据都可能提供隐式顺序。要用顺序交换、距离、长文档和 state reset 实验验证，而不能只背缩写。

练习一：构造 `A B`、`B A`、远距离 A/B 和版本冲突四组输入。

练习二：列出源码审计 NoPE 模型的五个位置相关入口。

练习三：解释为什么“顺序可区分”不等于“精确位置可检索”。

### 64.9.1 一个最小的顺序敏感性实验

准备四类输入：`A B`、`B A`、`A ... B` 和 `B ... A`。第一组只测方向，第二组逐渐增加中间干扰，第三组加入同名版本。模型输出要求包含顺序、距离桶和证据 ID。若短输入能区分方向而长输入不能，说明因果可见性提供了局部顺序，但远距离状态或位置分辨率不足。

实验还应交换 padding、packed sequence 和 request batch。若相同内容因 padding 数量不同而得到不同结果，可能是递归 state 或 position reset 实现错误，而不是 NoPE 本身的性质。

### 64.9.2 mask、state 和顺序的关系

没有显式位置变换时，causal mask 仍通过可见集合表达“过去和当前”；递归 state 通过更新顺序表达时间；local window 通过有限可见范围表达近邻。三者提供的是不同粒度的信息，不能用“有/没有位置编码”二分。

如果 state 在 batch 中被错误复用，模型会表现出不可解释的跨请求顺序；如果 packed sequence 没有 reset，第二个样本可能把第一个样本当作历史。审计 NoPE 模型时，要同时看 mask、position id、state owner 和 sequence boundary。

### 64.9.3 能力边界的一个反例

模型可以回答“句子 B 出现在句子 A 后面”，却不一定能回答“它位于第 800K 个 token 的哪份版本文档”。前一个任务只需要方向，后一个任务要求精确定位、长程记忆和证据引用。把前者成功写成“无需位置机制”是过度推断。

### 64.9.4 从置换性质看顺序来源

如果 self-attention 没有位置、mask 和递归状态，输入 token 的排列只会导致输出同步排列，模型无法知道哪个 token 先出现。causal mask 改变了每个位置可见的集合，递归更新又让状态按时间步累积，因此 NoPE 模型仍可能具有顺序敏感性。

这几种来源的区别可以用干预实验分开：保留 token embedding，分别关闭 causal mask、清空 state、打乱 position id，再比较顺序交换任务。关闭某个路径后能力下降，说明该路径承担了部分顺序信息；但实验只能说明该 toy 或该版本的行为，不能据此替代完整架构文档。

### 64.9.5 padding 与 packed sequence 的实现检查

对 batch 中长度不同的请求，padding token 是否进入 attention 和 state update 是关键问题。正确的实现应让 padding 不产生有效更新，packed sequence 应在样本边界重置 mask、position 或 state。一个最小回归是：同一请求单独运行、与一个短请求 padding 后运行、与一个长请求放入 packed batch 后运行，三者结果应在允许的数值误差内一致。

如果结果只在 batch 形态下变化，先查数据边界、mask 和 state owner；不要把这种差异解释成 NoPE 的“隐式位置能力”。

## 64.10 隐式顺序的来源分解

没有显式位置变换时，顺序信号仍可能来自 causal mask、token 到达顺序、残差写入、递归 state、特殊分隔符和训练数据中的文体规律。它们提供的不是同一种位置信息：causal mask 只说明可见性，递归更新保留处理顺序，分隔符提供离散边界，而 RoPE 才直接把位置差异放进 Q/K 点积。

因此“无位置编码”应进一步拆成几个问题：是否没有绝对 embedding，是否没有 RoPE，是否仍有 relative bias，是否使用 local window，是否由 state step 传递顺序。术语越短，越需要查看配置和源码。

## 64.11 对抗性顺序测试

顺序评估不能只交换两个短 token。可以构造三种输入：同一组事件的不同排列、同一实体的旧版和新版、相同词汇但不同段落边界。问题分别要求输出先后关系、最新版本和来源位置。

再增加重复干扰，让正确顺序信息只在一个位置发生变化。若模型只在没有干扰时成功，说明它利用了局部模式；若模型能在长干扰后区分版本但不能给出位置，说明存在顺序记忆而缺少精确位置解析。

## 64.12 与 serving 位置状态的关系

即使 backbone 不使用 RoPE，服务端仍可能需要 position、step、segment 和 reset 元数据。连续 batching 重排时，state 的 step 必须跟 request 迁移；多轮对话截断时，新的起点是否重置状态也必须明确。

把“模型内部没有显式位置”误解成“runtime 可以不传位置元数据”，会造成 cache 续接和 batch 恢复错误。位置是否进入网络和位置是否进入协议，是两个不同层次。

## 64.13 NoPE 的对照实验

研究 NoPE 或隐式顺序时，应与显式 RoPE、绝对位置和随机位置基线对照，并保持训练数据、参数和长度一致。任务包括顺序交换、相对距离判断、长距复制、冲突版本和 chunk 恢复。

如果去掉显式位置后局部任务不变、远程顺序任务下降，说明模型主要依赖 token 顺序和局部结构；如果恢复任务只在 runtime position 正确时成功，说明位置元数据仍然属于系统契约。

## 64.14 因果 mask 的顺序信息

causal mask 本身不把位置编号写进向量，却限制了每个位置能看到哪些前缀。不同前缀长度会产生不同的可见集合，残差和递归更新也会把处理顺序带入状态。可见性、到达顺序和相对距离是三个不同概念。

因此 NoPE 模型仍需要严格的 mask、segment、reset 和 padding 处理。把 mask 误写成双向，或让 padding 更新 state，会让顺序实验和线上结果失去意义。

## 64.15 NoPE 的证据边界

没有显式位置变换只能说明某条网络路径未使用该形式的位置编码，不能推出整个系统没有任何位置机制。局部窗口、relative bias、特殊分隔符、递归 step、数据顺序和 serving position 都可能提供顺序信息。

阅读新模型时应逐层查看 config、代码和技术报告；如果只看到“无 RoPE”一句话，就不能进一步推断隐式顺序的具体来源或长上下文能力。

## 64.16 从置换等变性看顺序

先考虑一个没有位置、没有 mask、没有递归状态的 self-attention。对输入序列做置换，输出只会随 token 一起置换，这叫 permutation equivariance。它可以判断 token 内容之间的关系，却不能知道“谁在前、谁在后”。

因果 mask 改变了这个性质。位置 t 只能看到 j <= t 的 token：

~~~math
A_{t,j}
=
\begin{cases}
q_t k_j^{\mathsf T}/\sqrt{d_h},&j\le t,\\
-\infty,&j>t.
\end{cases}
~~~

第一步只能看到自己，第二步能看到前两个 token，第三步能看到前三个 token。即使没有 RoPE，模型也能通过可见集合区分一部分顺序。这里的顺序信号是离散的“谁已经发生”，不是连续的“相隔多少位置”。

递归 state 又提供了第三种顺序来源：状态更新次数本身就是时间步。只要每个 token 按顺序更新 state，S_t 与 S_{t-1} 的因果关系就编码了处理顺序。若 state 被错误复用或 packed sequence 没有 reset，模型会出现跨请求的伪顺序。

## 64.17 “没有 RoPE”和“没有位置”不是同一句话

工程讨论中至少有五种不同的说法：

1. 没有绝对 position embedding。
2. 没有 RoPE 旋转。
3. 没有任何显式 position feature。
4. 某些 attention 分支没有位置变换。
5. 整个模型依赖隐式顺序或递归 state。

它们的证据要求不同。一个模型可能在 local attention 层没有 RoPE，但在 global 层使用位置 bias；也可能 backbone 的某一支没有位置旋转，但 runtime 仍然传递 position id、segment id 和 reset flag。阅读模型卡时，不能把宣传页中的短语直接提升为完整架构结论。

## 64.18 顺序、距离和地址是三个能力

给模型输入 A B 和 B A，测的是顺序可区分性；给出 A 和 B 中间不同长度的干扰，测的是相对距离和长期记忆；要求模型返回原文行号或段落，测的是显式地址和证据追溯。三者不能用一个 accuracy 代替。

一个 NoPE 路径可能很好地完成第一项，却在第二项和第三项失败。原因不是“模型完全没有顺序”，而是它获得的顺序信号粒度不足，或者状态容量无法保留精确位置。把“能判断 A 在 B 前面”写成“拥有任意长度的位置理解”，是典型过度推断。

## 64.19 一个可执行的 toy 对照实验

可以建立四个极小模型，只改变一条位置路径：

| 模型 | causal mask | position feature | recurrent state |
| --- | --- | --- | --- |
| A | 否 | 否 | 否 |
| B | 是 | 否 | 否 |
| C | 是 | RoPE/absolute | 否 |
| D | 是 | 否 | 是 |

训练三个任务：顺序二分类、距离分桶和唯一证据回读。输入长度从训练范围内逐渐增加到两倍，并随机插入干扰。若 B 能完成顺序但不能稳定完成距离，说明 mask 提供了方向而不是精确坐标；若 D 的流式任务更稳但引用失败，说明 state 提供了摘要能力却没有原文地址。

实验要保持参数量、训练 token、优化器和 batch 尽量一致。否则不能把性能差异归因给 NoPE。

## 64.20 position id 仍然可能属于协议

即使网络内部不使用显式 position 变换，runtime 仍可能需要 position metadata。原因包括：

1. prefix cache 需要知道缓存从哪个逻辑位置开始。
2. packed batch 需要区分不同样本的边界。
3. state snapshot 需要记录已经处理了多少步。
4. local/global attention 需要计算窗口和块偏移。
5. 多轮对话截断后需要决定是否 reset。

因此“NoPE”不能被用来简化 API 契约。服务端仍要传递并校验 logical position、segment id、reset、page offset 和 model revision。模型内部不读某个字段，不代表整个系统可以丢弃该字段。

## 64.21 padding 和 packed sequence 的回归

用同一请求做三次运行：

~~~text
1. 单独运行，无 padding。
2. 与较短请求组成 padded batch。
3. 与另一个请求组成 packed sequence。
~~~

在正确实现中，有效 token 的 logits 应在数值误差范围内一致。若 padded batch 结果改变，可能是 padding 参与了 state update；若 packed sequence 的第二个样本受第一个样本影响，可能缺少 reset；若只在恢复后变化，可能是 position offset 或 state checksum 错误。

这类实验比一句“没有位置编码所以顺序靠 mask”更有工程价值，因为它能定位实际系统中的边界错误。

## 64.22 NoPE 与长上下文外推

长上下文的困难至少来自四层：

1. API 层：请求是否被截断，tokenizer 是否与 max context 一致。
2. 访问层：mask、window、global layer 或 state 是否能到达远端证据。
3. 表示层：状态或 hidden 是否保留足够的顺序和版本信息。
4. 训练层：模型是否见过真实的跨段依赖，而不是只见过拼接的无关文本。

NoPE 只描述其中一条架构路径，不能单独解释长上下文成功或失败。一个长上下文模型的 needle recall 下降，可能是 position 外推、state 压缩、attention budget、数据分布或 serving offset 造成的。

## 64.23 顺序交换与干扰实验的统计设计

每个样本至少生成四个版本：原顺序、交换顺序、增加短干扰、增加长干扰。对多个随机种子和多个实体重复，报告准确率的均值和置信区间。若只测一组 A B，模型可能利用词汇偏置而不是顺序信息。

还应加入同名实体和冲突版本。例如文档先说“服务端口为 8080”，后文更新为“端口改为 9090”，问题要求输出最新值及其来源。这个任务同时测试顺序、版本覆盖和证据定位，比单纯顺序分类更接近实际使用。

## 64.24 对 NoPE 的因果干预

要判断顺序信号来自哪里，可以做四组干预：

1. 保留 token embedding，关闭 causal mask。
2. 保留 mask，清空 recurrent state。
3. 保留 mask 和 state，打乱或固定 position metadata。
4. 保留所有路径，只改变 padding 和 reset。

如果关闭 mask 后顺序能力消失，说明可见集合承担了主要作用；如果清空 state 后长程任务下降，说明递归状态参与了记忆；如果只改变 position metadata 就出现 batch 差异，说明 runtime 契约有问题。干预结果只能说明具体版本和 toy 模型的因果贡献，不能据此替代官方架构说明。

## 64.25 读者容易犯的三个错误

第一，把 NoPE 当成“模型天然拥有相对位置”。没有显式位置不代表距离表达无损，必须用距离分桶测试。

第二，把 causal mask 当成完整位置编码。mask 只改变可见性，不能直接告诉模型两个 token 相差 100 还是 10000。

第三，把模型行为当作内部结构证据。一个模型能回答顺序问题，可能来自训练语料、特殊分隔符、外部检索或 harness，而不一定来自某个 NoPE 设计。

## 64.26 面向 serving 的检查表

部署含有 NoPE 或隐式顺序路径的模型时，至少检查：

~~~text
tokenizer and special-token revision
causal/local/global mask
position id and block offset
state owner and reset boundary
padding and packed-sequence isolation
prefix-cache compatibility
preemption restore
speculative decoding position advance
model revision and cache checksum
~~~

推测解码尤其容易出错：draft model 产生多个 token 后，target model 和 state path 必须以相同 logical position 校验；拒绝部分 token 时，state 只能提交被接受的前缀。输出文本看似正常，并不能证明位置状态正确。

## 64.27 “没有显式位置”不等于“没有顺序信息”

自回归 mask、递归状态更新、卷积扫描、token 顺序和训练数据都可能提供顺序信号。移除显式位置参数后，模型仍然可能利用因果可见性判断先后；但这不意味着它天然知道任意距离的绝对位置。NoPE 的真正问题是：顺序信息从哪里来、在长序列和 batch 重排后是否仍然一致。

可以做一个位置置换实验：保持 token 内容不变，只交换位置、插入等价 padding、改变 packed boundary，再比较 logits。如果输出变化符合预期，说明模型依赖了某种顺序来源；如果变化只在特定长度出现，可能是训练分布或 kernel 的偶然偏差。

## 64.28 NoPE 与位置外推的边界

没有 RoPE 并不会自动消除长上下文外推问题。模型仍可能受训练长度、attention 衰减、状态容量、数据分布和数值精度影响。评估应包含长度扩展、middle retrieval、相对顺序判断、绝对位置任务和 packed sequence，不能只测一个 needle。

## 64.29 serving 中的隐式状态也要版本化

如果顺序依赖 causal mask、position offset 或递归 state，runtime 仍需保存 start position、reset 边界、padding mask 和 batch slot。把“没有 position embedding”误读成“不需要 position metadata”，会导致 prefix cache 复用和请求迁移时产生错位。

## 64.30 小结与资料边界

NoPE 是一种不在某个路径显式施加位置变换的设计选择，不是“没有顺序”。位置、mask、state、训练和层 schedule 共同决定模型能否使用顺序与距离。具体模型是否完全 NoPE、哪些层仍有位置处理，以官方 config、技术报告和代码为准。

本文的置换性质、因果 mask 和 toy 实验是通用教学分析；它们不能替代 Kimi、North Mini Code 或其他具体版本的官方模型卡。对于闭源模型，只有公开披露的接口和结构信号可以写成事实，其余应保留为待核验假设。
