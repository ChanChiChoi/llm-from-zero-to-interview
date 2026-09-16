# DeepSeek V4.1-Flash：CED、CSA2、缓存重算与原生多模态

> 本章核验日期：2026-09-14。模型事实主要来自 [DeepSeek-V4.1-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)、固定 revision `dba1be0a40aa45a94ad051997016db3960a90277` 的 `config.json`、[DeepSeek API 发布页](https://api-docs.deepseek.com/news/news260910) 和已逐页读取的 [技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)。报告中的内部实验、部署数字和 kernel 数量仍是发布方自报；文中把模型事实、报告设置、教学抽象和目标系统实测严格分开。

## 81.1 先从一个百万 token Agent 任务开始

假设一个代码 Agent 接到这样的任务：读取一个大型单体仓库，追踪过去几个月的迁移记录，检查几十个配置文件，运行测试，然后提交一个补丁。输入可能接近百万 token，输出却只有几千 token。系统的成本结构与普通聊天不同：

- 输入侧需要一次很重的 prefill，但之后可能复用很长的上下文；
- 输出侧每个 token 都要经过一次 decode，工具调用会把这个循环重复很多次；
- 历史状态既要能被检索，又不能把 HBM、SSD 和跨卡通信全部耗尽；
- 代码、日志和截图可能同时进入上下文，文本 token 数不再是全部输入成本；
- 任务成功还依赖工具、权限、verifier 和恢复策略，不能只看语言模型的单轮答案。

这类工作负载解释了 DeepSeek V4.1-Flash 公开架构中几个看似分散的设计：CED 把输入和输出的主干计算做成不对称路径，CSA2 压缩并选择远端 KV，SWA Bounded Replay 用有限重算换持久化空间，FP4 缓存降低每个历史 token 的字节数，DSpark 试图减少逐 token 等待，视觉编码器让图像进入同一个模型上下文。

但“有这些组件”不等于“任何 Agent 都会更快”。真正的服务指标仍由模型 revision、后端 kernel、上下文长度、batch、硬件、工具和 harness 共同决定。本章的第一条原则是：先把每个成本账本单独定义，再谈总收益。

## 81.2 证据分层和模型边界

当前可以直接引用的证据分成四层：

1. **官方发布页**支持页面日期、API 模型名、兼容 alias 和发布方产品层描述。
2. **模型卡正文**支持 552B backbone、1M context、CED、CSA2、FP4 KV、Engram、DSpark、多模态训练和评测协议等高层事实。
3. **固定 revision 的配置**支持 hidden size、层数、专家数、窗口字段、indexer 字段和视觉配置等机器可读快照。
4. **本地实验**只能支持教学实现或目标部署的实测，不能把 toy 代码的结果写成模型卡结果。

模型卡 revision 是权重与仓库的版本锚点。页面上的 `main` 内容、API 当前路由、榜单记录和本地代码实验都可能在之后变化。写报告时至少记录：

```text
model = DeepSeek-V4.1-Flash
revision = dba1be0a40aa45a94ad051997016db3960a90277
source_date = 2026-09-14
backend = <后端版本>
hardware = <硬件与并行方式>
harness = <工具、verifier、超时与重试>
```

“552B”是模型卡的 backbone 参数口径；“8B prefill / 16B decode”是每 token 激活参数口径；“890 bytes/token”是 global KV 的公开口径。三者回答不同问题，不能相加、相除或直接换算成单卡显存。

## 81.3 CED：为什么输入与输出可以使用不同计算账本

### 81.3.1 普通 decoder-only 的基线

在常见 decoder-only Transformer 中，输入 token 经过所有层的 causal self-attention，得到每个位置的 KV。生成第一个输出 token 后，后续每一步读取历史 KV，再只计算新 token 的 query、key、value 和 FFN。

若把每 token 的主干计算粗略记作 `C`，输入长度为 `N_in`、输出长度为 `N_out`，一个极粗的计算预算可以写成：

```math
C_{total} \approx N_{in} C_{prefill} + N_{out} C_{decode}.
```

这个式子只展示 token 数和路径的分离，不是 FLOPs 公式。真实成本还包括 attention 访存、MoE dispatch、通信、量化解码、kernel launch、padding 和工具等待。

普通结构往往用相同的主干深度处理输入和输出。对于“输入百万 token、输出很短”的 Agent，输入侧大量计算可能只服务于一次上下文建立；对于“输出很长”的任务，decode 的逐 token 成本又会成为瓶颈。

### 81.3.2 V4.1-Flash 的公开 CED 描述

模型卡将语言主干描述为 Causal Encoder-Decoder：40 层由 20 层 causal encoder 和 20 层 decoder 组成。它不是传统的双向 encoder，因为 encoder 仍保留 causal 约束；也不能仅凭名称把它等同于机器翻译中的 encoder-decoder。

模型卡进一步说明 decoder 的 global KV cache 从最终 encoder hidden states 投影得到，而不是由每个 decoder 层自己的 hidden states 逐层产生。公开资料因此支持一个教学级结构图：

```text
causal input tokens
        |
  20-layer causal encoder  ----> final encoder states
        |                                  |
        |                                  +--> projected global KV
        v
  20-layer decoder <---- local/SWA state + global compressed state
        |
  autoregressive output
```

这里的箭头是帮助理解数据依赖的简化图，不是对未公开 kernel、cross-attention 矩阵或每层投影实现的完整复原。

技术报告把这条路径写得更具体：global attention 的 decoder 层从第 `L/2` 层 hidden state 通过 layer-dependent projection 得到 global KV 与压缩权重；SWA 则仍从每层自己的 hidden state 生成局部 KV。因而，在 `N` 远大于 `n_win` 时，报告给出的教学级 prefill 复杂度近似为 `O(NL/2 + n_win*L/2)`。这解释了为什么 CED 能减少接近一半的主干 prefill，但不能把该复杂度式直接当作端到端延迟保证。

### 81.3.3 8B 与 16B 应该怎样读

模型卡给出 prefill 每 token 激活约 8B、decode 每 token 激活约 16B。对输入密集型任务，可以用激活参数作为一个粗略的工作量代理：

```math
A_{proxy}=8N_{in}+16N_{out}.
```

例如 `N_in=1,000,000`、`N_out=4,000` 时，代理账本是 `8,064,000` billion-parameter-token units；如果把一个普通 decoder-only 路径粗略记作每个阶段 16B，则输入侧代理会是 `16,000,000`。这只是说明不对称路径可能减少输入侧主干激活，不能直接宣称实际 FLOPs 减半，因为：

- 一个“激活参数”不等于一次浮点运算；
- encoder 与 decoder 的算子形状、专家路由和通信不同；
- global KV 投影、压缩、indexer 和量化会产生额外成本；
- batch、padding、并行度和显存带宽会改变有效吞吐。

工程评估应同时记录 `prefill_time`、`decode_time`、输入/输出 token、专家通信、峰值 HBM、persistent cache、TTFT 和 TPOT。只记录一个总延迟，无法判断 CED 是否解决了真正的瓶颈。

### 81.3.4 CED 的缓存含义

CED 把“能否重建 global KV”和“是否持久化每一层原始 KV”分开。一个服务系统至少要区分：

| 状态 | 用途 | 可能的生命周期 |
|---|---|---|
| encoder hidden states 或其投影 | 重建/访问 global context | HBM、SSD 或共享前缀存储 |
| global compressed KV | 远端注意力访问 | 持久化或按层共享 |
| SWA 最近窗口 | 局部连续性 | HBM 环形缓冲区，必要时重算 |
| indexer K/Top-K 结果 | 远端候选选择 | 与对应 revision 和上下文版本绑定 |
| 未完成压缩块 | 防止因果边界错误 | 当前请求结束或块封存时更新 |

如果服务端把所有状态都叫作“KV cache”，就容易误判命中、淘汰和恢复成本。CED 的价值不能脱离这些状态的物理布局来评价。

## 81.4 SWA Bounded Replay：用有限重算换持久化空间

### 81.4.1 滑动窗口为什么仍需要状态

滑动窗口注意力只让当前位置访问最近 `w` 个 token。窗口可以降低局部 attention 的访问范围，但窗口内的 K/V 仍要在增量生成和恢复时可用。最直接的方法是把每个窗口层的 KV 持久化到 HBM 或 SSD。

当上下文很长、租户很多、prefix cache 需要跨请求复用时，持久化全部 SWA KV 会产生三个问题：

1. HBM 容量不够，频繁换入换出增加尾延迟；
2. SSD 写入和读取放大，缓存命中不再等价于低延迟；
3. 不同层、不同窗口和不同压缩块的状态难以保持一致。

### 81.4.2 Bounded Replay 的核心交换

模型卡把 SWA Bounded Replay 描述为：需要恢复时，只重放最近的 `n_win` 个 token，重建缺失的 SWA KV，而不是把所有 SWA KV 写入 SSD。它把资源交换写成：

```math
\text{persistent bytes} \downarrow
\quad\Longleftrightarrow\quad
\text{replay compute} \uparrow.
```

如果窗口注意力严格只依赖窗口内状态，而且重放使用相同的模型 revision、tokenization、位置 id、精度和 kernel 语义，那么重建可以接近原路径。若窗口状态受到更远的递归摘要、随机算子或动态路由影响，重放就必须把这些依赖也纳入恢复协议。

“bounded”指重放范围有上界，不表示重放免费，也不表示任何状态都能被有限窗口准确恢复。系统应记录：

- `replay_tokens` 和 `replay_time`；
- 触发原因，是 SSD 缺失、cache eviction 还是跨 worker 迁移；
- replay 前后的 hidden/KV 校验摘要；
- 恢复时的模型 revision、tokenizer 和位置 offset；
- 重放失败后是否降级为完整 prefill。

### 81.4.3 HBM、SSD 与重算的统一账本

可用一个简化的请求账本描述选择：

```math
T_{restore}=T_{read}+T_{dequant}+T_{replay}+T_{verify},
```

其中 `T_read` 是从持久层读取的时间，`T_dequant` 是恢复量化状态的时间，`T_replay` 是重算时间，`T_verify` 是一致性检查时间。直接持久化时 `T_replay` 可能下降，但 `T_read` 和 SSD 空间上升；Bounded Replay 则相反。

策略不是固定选“缓存”或“重算”，而是按请求生命周期选择：热 prefix 可能保留 global KV，冷的 SWA 状态只保留恢复所需的 token 和 metadata；当 GPU 正在处理高并发 decode 时，延迟敏感请求可以少重算，离线任务可以多重算。

### 81.4.4 常见失败模式

- **错误窗口边界**：重放少了一个 token，局部状态从该点开始全部漂移。
- **位置 id 不一致**：同一 token 在一次性 prefill 和恢复路径使用了不同 offset。
- **模型版本混用**：旧 revision 的 SWA 状态被新 revision 读取，表面上能解码，语义已经不一致。
- **压缩块未封存**：正在写入的尾部块被当作完整块复用，造成因果泄漏或重复 token。
- **只验最终答案**：采样噪声掩盖了恢复路径与完整 prefill 的差异。

最小回归应该逐步比较 attention 输入、位置、dequant 后 KV 和 logits，而不是只比较最终字符串。

## 81.5 CSA2：压缩、索引和跨层复用

### 81.5.1 先把全注意力拆成两件事

全注意力的一个 query 通常要面对很多历史 KV。长上下文优化可以从两个方向下手：

- **表示压缩**：把多个 token 的历史表示汇聚成更少的条目；
- **候选选择**：从压缩后的条目中只读取可能相关的一小部分。

前者可能损失细节，后者可能漏掉相关块。若二者都被称作“稀疏注意力”，调试时就无法知道错误来自压缩还是 top-k。

### 81.5.2 CSA2 的三种静态层模式

模型卡公开 CSA2 为每个 attention layer 分配三种静态模式：

| 模式 | 教学化理解 | 主要成本/风险 |
|---|---|---|
| `Full` | 建立或完整访问所需的 main KV/indexer K | 成本较高，但可提供候选基准 |
| `Reindex` | 重新计算 indexer 相关状态 | 可适应当前层，但有额外索引计算 |
| `Reuse` | 复用已有 main KV、index 和 Top-K 结果 | 便宜，但依赖版本、位置和候选稳定 |

“静态”说明层模式在配置中预先安排，不等于请求中的每个 token 都访问相同数量的实际历史。模型卡明确把这些模式用于跨层共享 main KV 与 indexer K，并复用 Top-K 稀疏注意力索引。

一个简化的层间依赖可以画成：

```text
Full layer:    main KV + indexer K + Top-K candidates
                  |          |
Reindex layer:  shared KV   refresh indexer
                  |          |
Reuse layer:    shared KV   reuse Top-K
```

真实实现还需要处理因果可见性、压缩块边界、SWA 局部路径和跨卡分片。本图不表示每种模式只需一次矩阵乘。

### 81.5.3 Hierarchical Sparse Indexer

如果每个深层 indexer 都在一百万 token 上重新排序，indexer 本身可能成为瓶颈。模型卡给出的层次化方案是：先由第一个 Full Mode layer 构造候选池，后续 indexing layer 只在该候选池内继续索引。

令完整上下文压缩块数为 `B`，第一层选出的候选池大小为 `C`，深层最终 Top-K 为 `K`，可将两条路径粗略写成：

```math
\text{flat index cost} \propto B,
\qquad
\text{hierarchical later-layer cost} \propto C,
\quad C \ll B.
```

这里的“与上下文长度无关”应谨慎理解：它表示后续 indexer 的候选范围由 `C` 约束，不能把它解释为整个模型的所有成本都与长度无关。建立候选池、维护压缩条目、写入 global cache、读取局部窗口和跨层通信仍可能随上下文增长。

技术报告给出了这个候选池的具体构造：decoder 第一个 `Full` 层先对全部因果可见的 main KV 位置打分，再按 block 内最大分数选择最多 2,048 个 block，每个 block 含 8 个位置，得到最多 16,384 个候选位置。后续 `Reindex` 层只在这个池内重新打分和选择 Top-512；`Reuse` 层不再进行新的 indexing。第一层的全范围扫描仍然存在，所以这里降低的是后续 indexer 的搜索成本，不是把整个 decode 变成常数成本。

漏检风险也变成了两级风险：第一层没有把相关块放进候选池，后续层就没有机会找回；后续层的 Top-K 又可能在候选池内继续漏检。因此评估至少要拆成：

```math
R_{end-to-end}=R_{candidate\ pool}\times R_{top-k\mid candidate\ pool}.
```

若两个召回率都接近 0.9，端到端上界只有约 0.81。这个乘法只是独立近似下的教学解释，正式测量应使用带 gold evidence 的任务和实际相关块。

### 81.5.4 配置快照提供了什么

该 revision 的 `config.json` 给出 `index_n_heads=32`、`index_head_dim=128`、`index_topk=512`、`candidate_topk_blocks=2048`、`candidate_block_size=8`，并给出 index source layer ids。它们是可复现配置字段，适合用于 cache layout 和实验参数记录。

技术报告的训练排布还明确了这些模式如何组合：encoder 前两层为纯 SWA，余下 18 层按三个六层组使用压缩率 `m=2` 的 `Full + 5 Reuse`；decoder 的 20 层按五个四层组使用压缩率 `m=1`，第一组为 `Full + 3 Reuse`，其余四组为 `Reindex + 3 Reuse`。这是该报告的模型设置，不能从 `Full/Reindex/Reuse` 三个名称推导出其他 revision 采用同一排布。

配置字段不自动说明：

- 每个字段在所有后端中如何映射到 kernel；
- Top-K 的评分函数和 tie-breaking 是否完全稳定；
- indexer 结果是否跨请求、跨 batch 复用；
- 2048 个 block 是否就是每个 query 的最终读取量；
- 这些值在未来 revision 是否保持不变。

因此实验报告应把 `config_revision` 与实测代码版本一起写入，而不是只在标题中写“CSA2”。

## 81.6 FP4 main KV：字节减少与误差账本

### 81.6.1 为什么 KV 量化不是只改 dtype

KV cache 在 decode 中被反复读取。把 BF16 改成 FP4 可以减少存储和带宽，但量化误差会进入每一步 attention score 和 value 聚合。长上下文还会放大量化误差影响，因为某个远端 token 可能在许多层和许多次检索中被读取。

模型卡公开 main KV 使用 E2M1 FP4，并为每 16 个 channel 使用一个 E4M3 scale。教学上可以写成：

```math
\hat{x}_i=s_g q_i,
\qquad
g=\left\lfloor i/16\right\rfloor,
```

其中 `q_i` 是 E2M1 可表示的有限集合中的值，`s_g` 是第 `g` 组的 scale。真正的编码、舍入、异常值和 kernel 解码规则以实现为准；上式只说明为什么 scale 粒度影响误差。

技术报告进一步说明，main KV 的量化在 post-training QAT 中加入，位置选择在 RoPE 之后，attention 前再反量化；SWA KV 保留 FP8。报告还说明该 E2M1 配置沿用 NVFP4 的每 16 channel scale 组织，但省略第二级 global scale，以简化 cache layout。它把这些作为该模型的精度选择，不是所有 FP4 cache 的通用实现。

组越大，scale 元数据越少，但同一组内大值可能挤压小值；组越小，误差可以更局部，但 metadata 和解码工作增加。E4M3 scale 也不是“无限精确”的浮点数，scale 本身需要量化和边界处理。

### 81.6.2 890 bytes/token 的正确口径

模型卡称 global KV cache footprint 为每 token 890 bytes，约为 DeepSeek-V4-Flash 的四分之一。这个数字至少要附带以下标签：

```text
metric = global KV cache footprint
unit = bytes / token
precision = FP4 main KV + documented scale scheme
model = DeepSeek-V4.1-Flash
revision = dba1be...
```

它不等于：

- 所有 attention layer 的总 cache；
- SWA 最近窗口、indexer、未完成尾部和 metadata 的总和；
- 一个 batch 的 HBM 占用；
- 权重、专家、通信 buffer 或 runtime workspace；
- 任意硬件和 batch 下的实际显存。

一个多请求 cache 账本可以写成：

```math
M_{HBM}=M_{weights}+M_{global\ KV}+M_{SWA}+M_{index}+M_{workspace}+M_{comm}.
```

只有 `M_global KV` 的某个组成部分有 890 bytes/token 的公开参考值，其他项必须实测或从配置和实现单独推导。

### 81.6.3 误差如何被评估

只比较平均 perplexity 不足以评价 KV 量化。至少应加入：

1. 短上下文复制和局部代码补全，检查局部细节是否被损坏；
2. 长上下文单证据检索，检查远端 evidence 是否还能进入候选池；
3. 多证据合并，检查多个压缩块同时被访问时的累积误差；
4. Agent 工具循环，检查一次错误 cache 是否影响后续计划和最终 artifact；
5. 完整 prefill、cache hit、eviction 后 replay 三条路径的 logits 对照。

报告应同时给出召回、答案准确率、TTFT、TPOT、cache bytes、dequant 时间和单位成功成本。降低字节数但使 verifier 失败率上升，不是无条件的优化。

## 81.7 MoE、Engram、mHC 与 DSpark 要分开理解

### 81.7.1 MoE 的三本账

配置快照给出每个 MoE 层 1 个 shared expert、384 个 routed experts，每 token 选择 6 个 routed experts。至少要分开记录：

- **总参数**：决定权重规模、分片和存储压力；
- **激活参数**：近似一次 token 参与主干矩阵计算的参数量；
- **通信账本**：决定 token dispatch、all-to-all、容量溢出和尾延迟。

可以用一个简化的每层路由表示：

```math
y_t=y_t^{shared}+\sum_{e\in TopK(t)}p_{t,e}f_e(x_t),
\qquad |TopK(t)|=6.
```

这不是该模型完整的 router 公式，只说明 shared expert 与 routed experts 的角色不同。若一个 batch 中大量 token 选择同一专家，6 个 active experts 并不意味着通信成本固定；还要看容量因子、专家并行拓扑和 overflow 策略。

### 81.7.2 Engram conditional memory

模型卡把 Engram 描述为 196B 参数的 conditional memory，并通过 token-based lookup 稀疏访问。它与普通 dense FFN 的差异在于：不是每个 token 都激活全部记忆参数，而是由 token 相关的 lookup 选择一部分条件记忆。

从系统角度看，Engram 具有类似“模型内条件查表”的性质，但不能简单等同于 RAG：

| 机制 | 访问对象 | 更新方式 | 主要风险 |
|---|---|---|---|
| RAG | 外部文档/向量库 | 文档和索引可独立更新 | 权限、时效、引用和注入 |
| Engram | 模型内条件记忆 | 随模型版本发布 | lookup 冲突、容量和版本固定 |
| KV cache | 当前上下文状态 | 请求生命周期内写入 | 位置、淘汰和恢复一致性 |

公开资料没有给出 Engram 的完整哈希、参数分片和训练损失，本章不补写这些实现细节。部署时应记录 lookup hit、miss、额外存储、跨卡访问和与主干输出的消融结果。

技术报告已公开一部分更具体的工程设置：196B Engram 参数均分到两个模块，n-gram 阶数为 `{2, 3, 4}`，每阶使用 8 个 hash heads 和 2048 的总 embedding dimension，表项约 16M 且表大小取不同素数；embedding 与 key/value projection 使用 FP8，模块放在 zero-indexed 的第 1、14 层。报告还说明省略 short causal convolution，并以 host memory RDMA 预取 embedding。哈希细节、分片方式和实际 hit/miss 曲线仍不能由这些字段补出。

### 81.7.3 Single-Pass mHC

模型卡列出 Single-Pass mHC 和 Mega-mHC kernel。mHC 的主题是残差流混合与稳定性，和本册第 78 章的双随机矩阵约束相连；它不是 CSA2，也不是沿深度选择历史表示的 Attention Residuals。

比较三个机制时要问不同的问题：

- AttnRes：当前层如何选择历史深度表示？
- mHC：多路残差映射如何避免深层连乘失控？
- CSA2：历史 token 的 KV 如何压缩、索引和复用？

它们可以出现在同一个模型中，但优化对象、状态生命周期和失败模式不同。没有 Mega-mHC 的实现与 profiling，不能由 “Single-Pass” 直接推出固定吞吐提升。

技术报告给出了 Single-Pass 的机制边界：把当前 block 使用的 input-mixing 系数从 `A_l` 移到前一 block 产生的 `A_{l-1}`，从而消除等待当前 hidden-dimension reduction 的依赖；预训练仍使用原来的多 kernel 路径，部署使用 Mega-mHC 融合 residual update、input mixing、coefficient prediction、pre-norm 和 FP8 conversion。报告称 activation traffic 从原 mHC 实现约 `(4n+4)d` 降到 Single-Pass 的 `(2n+2)d`，并观察到性能损失可忽略；这仍是报告设置下的结果。

### 81.7.4 DSpark 与推测解码

DSpark 被模型卡描述为半自回归草稿生成与按置信度调度验证。它的基本目标仍是减少目标模型逐 token 的等待：草稿路径提出多个候选，目标路径验证一个连续前缀，错误位置之后回退。

技术报告补充了草稿器结构：它包含 3 个 Transformer blocks、128 token 的 sliding window，一次前向并行产生 5 个 draft positions；Markov head 建模草稿 token 之间的依赖，confidence head 预测逐位置的条件接受概率，scheduler 再结合当前 engine 的吞吐曲线决定验证长度。DSpark 在 backbone 预训练后单独训练，后训练时与 backbone 一起更新但不把 DSpark objective 的梯度传回 backbone；报告同时说明 backbone 预训练阶段不使用 MTP。

若草稿提出长度为 `m` 的序列，目标模型平均接受 `A` 个连续 token，则一次目标调用的有效 token 代理为：

```math
T_{effective}=1+E[A].
```

`E[A]` 受草稿质量、置信度阈值、验证开销、KV 状态维护和 batch 调度影响。它不是“3 层预测所以三倍吞吐”。配置中 `num_nextn_predict_layers=3` 和目标层字段只能说明该 revision 的实现快照，不能替代后端测量。

需要特别区分：

- **模型头存在**：权重和配置提供草稿预测路径；
- **后端支持**：推理框架能正确调度草稿与目标验证；
- **有效收益**：在固定硬件、batch、任务和接受率下，TPOT 或吞吐确实改善。

## 81.8 原生多模态路径和 token 预算

模型卡称视觉编码器为从头训练的 DeepSeek-ViT，使用 2D-RoPE 和 3x3 pixel-unshuffle 下采样，再通过两层 MLP projector 产生视觉 embedding，与文本 embedding 一起参与语言模型预训练。

配置快照给出视觉侧的一组字段：32 层、hidden size 1024、16 个 attention heads、patch size 14、downsample ratio 3、最多 1024 个 image tokens。它们可以用于初始化显存和序列预算，但不能把每张图简单换算成固定文本 token：分辨率、裁剪、纵横比、处理器策略和模型版本都会改变视觉 token 数。

“原生多模态”在工程上至少涉及三个接口：

1. **输入编码**：图像在 prompt 中用 image marker 占位，媒体数据按出现顺序传给视觉处理器。
2. **序列对齐**：文本、图像 patch 和位置坐标要有明确顺序；图像 token 变化会改变后续 position id。
3. **输出与工具**：模型可以读图并生成文本，但图像读取、文件权限、外部检索和工具执行仍由宿主系统管理。

多模态 Agent 还会产生媒体 prompt injection。图像中的文字、网页截图和文档表格不应自动获得 system 指令地位；verifier 要把“模型看到了什么”与“系统允许执行什么”分开。

## 81.9 预训练、后训练与 reasoning effort

### 81.9.1 长上下文训练不是接口字段

模型卡称该模型在包含 45T tokens 的多模态语料上从头训练；稀疏 attention 在 64K sequence length 上训练，并使用 34T tokens 把上下文扩展到 1M。这里至少有三种长度：

```text
训练 sequence length：模型在哪些长度分布上优化过
上下文配置上限：实现允许输入的最大 position
有效评测长度：任务在多长输入下仍能正确检索和推理
```

不能从 `max_position_embeddings=1048576` 推出任意一百万 token 文档都能准确使用。评测需按长度分桶，同时报告证据位置、检索距离、压缩路径、cache 命中和失败类型。

技术报告把训练设置写得更具体：固定 batch 为约 100.6M tokens，学习率前 2,000 steps 线性 warmup，在 28T tokens 前保持 `2.6e-4`，28T--40T 按 cosine 衰减到 `2.6e-5`，并保持到 45T；稀疏 attention 从 64K sequence length 起训，在 34T tokens 扩展到 1M。视觉编码器另有约 47B alt-text image-text pairs 的 224x224 对比预训练，以及接入 4B MoE LLM、使用 236B tokens 和 544x544--1344x1344 分辨率的自回归微调。这些是报告中的训练配置，不是服务 API 的输入限制。

### 81.9.2 SFT -> RL -> OPD

模型卡把后训练总体流程写为 `SFT -> RL -> on-policy distillation (OPD)`，并称主要变化在 Agent 任务、环境和 rollout 的自动合成与渐进扩展数据管线。官方发布页则以较高层级提到新的预训练方法和更大规模 RL。

对这两种表述，正确做法是保留粒度差异：

- 可以写流程顺序和数据管线方向；
- 不能写未公开的奖励权重、教师结构、具体 RL 算法或 loss 系数；报告明确披露的“40 多个 teacher models”只能作为该报告的训练设置记录；
- Agent benchmark 结果需要绑定环境、工具、verifier、超时、重试和 context policy；
- OPD 的“统一能力”不等于推理时同时运行多个领域专家。

技术报告把任务抽象成 `(problem, environment, verification system)` 三元组，并把任务难度与正确性作为质量维度；报告还称最后的全词表 OPD 使用 40 多个 teacher models。它同时说明主要变化在自动合成任务、构造环境、过滤和课程平衡，而不是新的后训练算法，因此这些数量和流程仍应标为该报告的训练设置。

### 81.9.3 数值 reasoning effort

V4.1 的编码说明把 `reasoning_effort` 公开为 1--100 的整数预算，`low=50`、`high=75`、`max=100`，默认 `high=75`。它更像一个连续控制面，而不是五个独立模型。

技术报告还说明 RL 训练时把 effort 写入 system prompt，并在相同 `(prompt, effort)` 子组内做 reward mean-centering；长度惩罚系数随 effort 指数衰减。报告观察到从 effort 25 提升到 100 时，八个 reasoning benchmarks 的平均 Pass@1 从 67.1% 到 76.3%，DeepSWE v1.1 从 66.0% 到 74.2%，Terminal-Bench 2.1 从 82.4% 到 90.6%，输出 token 约增加 2.5 倍。这里的数字是报告自报的曲线摘要，不是本项目实测，且不能据此保证每个任务点都单调提升。

如果把质量记作 `Q(e)`、推理 token 记作 `R(e)`、工具调用数记作 `U(e)`，一个请求级决策可以用：

```math
J(e)=Q(e)-\lambda_R R(e)-\lambda_U U(e)-\lambda_T T(e),
```

其中 `e` 是 effort，`lambda` 是业务对成本、工具和延迟的权重。教学公式不主张 `Q` 一定单调，也不表示平台对所有 effort 都线性分配 token。应在固定任务和固定 harness 下测质量曲线、p95 延迟、成本、超时和权限事件。

## 81.10 Prompt 协议不是普通 chat template 的小改动

该 revision 没有 Jinja chat template，而是提供独立的 `encoding/encoding.py` 和测试。工程接入不能只把旧模型名替换成新模型名。

### 81.10.1 DSML 标签变化

V4.1 的工具调用标签带前导空格，例如 `<｜DSML｜ calls>`、`<｜DSML｜ invoke>` 和 `<｜DSML｜ parameter>`；V4 使用的无空格形式不能原样复用。一个不可见空格就可能让 parser 把工具调用当作普通文本。

### 81.10.2 effort 只在正确位置渲染

编码说明称 effort 前缀只在 thinking 模式、对话开头渲染。中途 system message 使用 `<｜System｜>`，并影响 assistant generation header。适配器需要保存原始消息序列、thinking mode、effort、图像顺序和工具 schema，不能只保存拼好的字符串。

### 81.10.3 协议、传输和执行器三层分离

`deepseek-recipe` 可以把 Messages、Chat Completions 和 Responses 转换成模型 prompt，解析 thinking、工具调用、图像和流式响应。但它不负责：

- 发起 HTTP 请求；
- 执行 shell、文件或网络工具；
- 授予模型宿主权限；
- 处理业务审批、超时、回滚和审计。

正确的请求链是：

```text
application messages
  -> protocol encoder
  -> model generation
  -> parser
  -> policy / permission check
  -> tool executor
  -> verifier and audit log
```

## 81.11 API alias 与版本迁移

官方 API 发布页标明 V4.1-Flash 的 API 模型名为 `deepseek-flash`。页面还写明：

- `deepseek-v4-flash` 与 `deepseek-v4-flash-vision-exp` 在兼容期暂时路由到 V4.1-Flash；
- 从 2026-09-14 04:00 UTC 起，`deepseek-v4-pro` 请求路由到 V4.1-Flash，直到 V4.1-Pro 发布；
- V4.1-Flash 已在 DeepSeek API 提供原生多模态能力。

alias 路由属于服务端行为，不能当作权重身份。生产客户端应记录请求 alias、响应中的实际 model/version、请求时间、协议版本和成本。回归测试至少覆盖：

1. 纯文本 thinking 与非 thinking；
2. 数值 effort 以及 `low/high/max` 别名；
3. 中途 system message；
4. 工具调用的 DSML 标签；
5. 多图交错文本；
6. cache hit、cache miss、SWA replay 和错误恢复。

## 81.12 官方自报评测：先看协议再看数字

模型卡 base 表给出 MMLU-Pro 74.1、HumanEval 79.4、GSM8K 93.0、MMMU-Pro 56.5 和 DocVQA 95.6。其 instruct/Agent 表给出 Terminal-Bench 2.1 Pass@1 90.6、DeepSWE v1.1 Resolved 74.2、AutomationBench 54.8 和 Agent's Last Exam 31.8。

这些是发布方模型卡中的结果，不是本项目独立复现。模型卡还绑定了关键条件：instruct 使用 `reasoning_effort=100`、`temperature=1.0`、`top_p=0.95`；代码 Agent 使用 DeepSeek Harness Minimal mode 和 1M context，DeepSWE 对齐 `mini-swe-agent`，视觉 Agent 使用 Claude Code harness 和 512K context，其他 benchmark 使用自己的官方 scaffold。

因此正确的记录格式应是：

```text
score = 74.2
benchmark = DeepSWE v1.1 Resolved
model_revision = dba1be...
reasoning_effort = 100
harness = mini-swe-agent / DeepSeek evaluation instructions
temperature = 1.0
top_p = 0.95
context = 1M
source = official model card
```

DeepSWE 的 74.2 观测的是模型、harness、工具、容器、verifier、超时和重试的组合。它不能直接和 Artificial Analysis 的指数、另一个模型的不同 harness 或普通单轮 HumanEval 分数拼在一起。

技术报告的 Table 1 还给出内部 base-model 对照，例如 DeepSeek-V4.1-Flash-Base 的 Codeforces rating 为 3471、LongBench-V2 为 45.2、CVBench 为 77.9、RefCOCO 平均值为 86.0；Table 3 则补充 Terminal-Bench 3.0/4.0、CyberGym、SEC-Bench Pro、ExploitGym 和视觉 Agent 的结果。它们都受报告的内部评测框架、指定 scaffold、采样设置、上下文、容器和 verifier 约束，应与模型卡表格分栏引用。

## 81.13 Serving engine 应该维护哪些状态

### 81.13.1 请求级 manifest

一个可恢复的请求至少需要：

```text
request_id
model_revision
tokenizer_revision
prompt_encoding_revision
position_offset
media_manifest
cache_segments
CSA2 mode and index metadata
SWA replay metadata
reasoning_effort
tool schema and permission policy
harness / verifier revision
```

没有这些字段，cache 命中后的错误无法区分为模型漂移、协议漂移、位置错位、媒体顺序错乱还是工具执行失败。

### 81.13.2 分层 cache layout

建议把缓存分为逻辑 segment，而不是一个连续巨型数组：

```text
prefix manifest
  ├── encoder/global KV segments
  ├── compressed KV and quantization scales
  ├── indexer K and candidate pool
  ├── recent SWA ring buffer
  ├── unsealed tail block
  └── replay metadata and checksums
```

每个 segment 都要带 model revision、位置范围、精度、压缩倍率、layer ids 和生命周期。Prefix reuse 只能复用满足这些条件的 segment；“文本前缀相同”不够。

技术报告描述的部署实现还把 EPD（Encoder--Prefill--Decode）作为解耦边界，使视觉编码、prefill 和 decode 可以独立扩缩并重叠。它不再把 SWA KV 放入持久 KV cache，而是放入每台机器约 10% host DRAM 的短 TTL 分布式池；global KV 保留在持久 cache 中，报告给出的配置生命周期至少为 72 小时。命中 global KV 但缺失 SWA KV 时，Encoder SWA Bounded Replay 只重放最近 `n_win` token；decoder replay 也只对 prompt 尾部做有界重算。这些是报告描述的服务实现，不能直接当作所有后端的默认布局。

### 81.13.3 资源门禁

在调度前估算：

```math
M_{request}=M_{global}+M_{index}+M_{SWA}+M_{tail}+M_{workspace}+M_{comm}.
```

在恢复时估算：

```math
T_{request}=T_{read}+T_{replay}+T_{verify}+T_{decode}+T_{tool}.
```

只有当容量、延迟、权限和成功率门禁都通过，才允许把“cache 节省”写成发布结论。一个部署实验至少输出 HBM peak、SSD bytes、replay tokens、index hit、candidate recall、acceptance length、TTFT、TPOT、p95、tool failure 和单位成功成本。

## 81.14 一个可运行的教学实验

下面的代码不加载模型、不实现真实 FP4 kernel，也不声称复现 V4.1。它把本章的四个账本缩小为可检查的标准库示例：CED 激活代理、global KV 字节、E2M1 风格教学量化和草稿接受前缀。

```python
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ServingLedger:
    prefill_active_b: float
    decode_active_b: float
    global_kv_bytes: int
    persistent_hbm_ratio: float
    persistent_ssd_ratio: float


def require_positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def require_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def activation_proxy(ledger, input_tokens, output_tokens):
    if isinstance(input_tokens, bool) or isinstance(output_tokens, bool):
        raise ValueError("token counts must be integers")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise ValueError("token counts must be integers")
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return (
        input_tokens * ledger.prefill_active_b
        + output_tokens * ledger.decode_active_b
    )


def e2m1_like(values, scales, group_size=16):
    if not isinstance(scales, (list, tuple)) or not scales:
        raise ValueError("scales must be a non-empty sequence")
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        raise ValueError("group_size must be an integer")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    expected_scales = (len(values) + group_size - 1) // group_size
    if len(scales) != expected_scales:
        raise ValueError("one scale is required for each value group")
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    quantized = []
    for index, value in enumerate(values):
        require_finite(value, "value")
        scale = scales[index // group_size]
        require_positive_finite(scale, "scale")
        magnitude = abs(value) / scale
        nearest = min(levels, key=lambda level: abs(level - magnitude))
        quantized.append((1.0 if value >= 0 else -1.0) * nearest)
    return quantized


def accepted_prefix(draft, target):
    accepted = 0
    for draft_token, target_token in zip(draft, target):
        if draft_token != target_token:
            break
        accepted += 1
    return accepted


def main():
    ledger = ServingLedger(8.0, 16.0, 890, 0.25, 0.125)
    proxy = activation_proxy(ledger, input_tokens=1000, output_tokens=4)
    teaching_values = e2m1_like([0.25, 1.1, -3.2, 5.9], scales=[1.0])
    accepted = accepted_prefix([11, 12, 13], [11, 12, 99])
    print("CED active proxy:", proxy)
    print("global KV bytes/token:", ledger.global_kv_bytes)
    print("relative persistent HBM/SSD:", ledger.persistent_hbm_ratio,
          ledger.persistent_ssd_ratio)
    print("E2M1-like values:", teaching_values)
    print("accepted draft tokens:", accepted)


if __name__ == "__main__":
    main()
```

这个 demo 有三个刻意的简化：`e2m1_like` 没有编码真实 bit pattern，`activation_proxy` 不是 FLOPs，`accepted_prefix` 没有维护草稿/目标 KV 状态。真实系统必须加入向量维度、group scale、causal mask、cache invalidation、后端 kernel 和错误恢复。

## 81.15 容易被问到的误区

### 误区一：CED 就是普通 encoder-decoder

不准确。公开描述强调 encoder 仍是 causal，decoder global KV 从最终 encoder states 投影得到；不能把它自动等同于双向 BERT encoder 或传统翻译 Transformer。

### 误区二：8B active 意味着只需要 8B 参数显存

不准确。总权重、专家分片、KV/cache、量化 scale、通信 buffer、workspace 和运行时并发仍要计入。8B/16B 是激活参数口径。

### 误区三：890 bytes/token 就是每 token 的全部显存

不准确。这是 global KV footprint 的公开指标，SWA、indexer、尾部、workspace 和权重另算。

### 误区四：CSA2 让所有注意力都是 O(1)

不准确。hierarchical indexer 限制后续候选范围，但建立候选池、维护压缩、局部窗口和 global attention 仍有成本；召回也会受压缩和候选漏检影响。

### 误区五：Bounded Replay 没有 SSD 就没有恢复成本

不准确。它把部分持久化成本换成 replay compute、校验和可能的恢复延迟。

### 误区六：DSpark 的三层预测等于三倍吞吐

不准确。有效收益取决于 acceptance length、验证成本、状态维护、batch 和后端支持。

### 误区七：原生多模态意味着图像拥有工具权限

不准确。模型可以接受图像输入；文件读取、网络、工具、权限和审批仍由宿主系统控制。

### 误区八：模型卡 Agent 分数就是基础模型分数

不准确。Agent 分数绑定 harness、环境、工具、verifier、effort、上下文和超时策略，应作为组合系统结果记录。

## 81.16 面试追问

1. **为什么输入侧是 8B active、decode 侧是 16B active？**

   先说这是模型卡公开的激活参数口径，再解释 CED 对输入和输出使用不对称路径的动机；最后强调它不是直接的 FLOPs 或延迟保证。

2. **SWA Bounded Replay 与把 KV 全部写 SSD 的取舍是什么？**

   比较 persistent bytes、SSD IO、重放 token、恢复延迟和一致性校验；指出窗口边界、位置 offset 和 revision 是恢复正确性的必要条件。

3. **CSA2 的 Full、Reindex、Reuse 分别解决什么问题？**

   Full 建立可靠基准，Reindex 刷新索引，Reuse 共享已有状态；再说明跨层复用需要绑定位置、压缩块和模型版本。

4. **Hierarchical Sparse Indexer 为什么可能降低长上下文成本？**

   因为后续 indexer 在第一层形成的候选池上工作，候选规模可以小于完整压缩块数；但候选池漏检会给后续层带来不可恢复的召回损失。

5. **FP4 KV 如何评估是否值得？**

   同时测 bytes、bandwidth、dequant time、长上下文 evidence recall、局部复制、Agent 成功率和单位成功成本，不能只看 cache 大小。

6. **Engram 与 RAG 有什么不同？**

   Engram 是模型内条件 lookup，随模型 revision 发布；RAG 读取外部文档，可独立更新，但需要权限、引用、时效和注入控制。

7. **reasoning_effort=100 是新模型吗？**

   不是。它是同一模型的请求级预算配置。比较 effort 时要固定 revision、任务、harness、工具和输出上限，并报告质量、token、延迟和成本。

8. **为什么不能把 DeepSWE 74.2 和另一个榜单分数直接排序？**

   因为 DeepSWE 观测的是模型 revision 加 harness、工具、环境、verifier、超时、重试和 context policy 的组合；不同协议没有可比的单一分母。

## 81.17 小练习与实验设计

1. 实现一个 CED 与 decoder-only 的 token 账本，分别改变输入/输出长度，报告 prefill/decode proxy；再说明为什么 proxy 不能替代 profiler。
2. 给一个窗口层 cache 加入 replay metadata，故意改变 position offset、revision 和窗口边界，验证恢复门禁能否拒绝不一致状态。
3. 用二维向量实现压缩块，分开测压缩误差、候选池召回和 Top-K 召回；不要把两个错误率合成一个未经定义的“稀疏损失”。
4. 模拟 `Full -> Reindex -> Reuse` 三种层模式，比较 indexer 次数、候选池大小、cache bytes 和 evidence recall。
5. 对比 FP4-like、INT8 和 BF16 教学缓存，在长度桶、远端检索和 Agent 工具循环上报告质量—内存—延迟曲线。
6. 模拟 DSpark 的草稿接受率为 0.2、0.5、0.8，加入验证成本和状态回退，测量有效 token、目标调用数和 p95 延迟。
7. 设计多模态 prompt injection 测试：同一张图分别包含普通说明、伪 system 指令和恶意工具参数，验证模型输入、策略层和执行器的边界。
8. 固定 `reasoning_effort`、工具、超时和 verifier，复现一个短任务与长任务矩阵，并记录 `not_applicable` 的空任务集，不用默认分母伪造成功率。

## 81.18 来源与进一步阅读

- [DeepSeek-V4.1-Flash Hugging Face 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)：架构、训练、评测、prompt encoding 和许可证。
- [DeepSeek-V4.1-Flash `config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)：本章配置字段快照。
- [DeepSeek-V4.1-Flash API 发布页](https://api-docs.deepseek.com/news/news260910)：API alias、兼容路由和服务端发布说明。
- [DeepSeek-V4.1-Flash 技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)：51 页报告已逐页读取；页码对应的层排布、训练/部署设置和评测边界见本章相关小节。完整 kernel source、线上接受率和独立 profiling 仍待核验。
- [DeepSeek V4 技术报告](https://arxiv.org/abs/2606.19348)：CSA/HCA、mHC、训练与部署背景；不能用 V4 报告替代 V4.1-Flash 模型卡的具体字段。
- [第二十一册第 77 章：CSA/HCA 从压缩 KV 到百万上下文](77-csa-hca从压缩kv到百万上下文.md)：压缩注意力先修。
- [第二十一册第 78 章：mHC](78-mhc双随机残差连接.md)：残差流稳定性先修。
- [第二十一册第 79 章：Mistral Small 4](79-mistral-small-4混合推理与eagle量化.md)：统一 reasoning effort 与推测部署对照。
- [第二十一册第 80 章：Step 3.5 Flash](80-step-3-5-flash的mtp与滑动窗口moe.md)：MTP、SWA/full attention 和 Context Manager 对照。

本章的结论可以压缩成一句话：DeepSeek V4.1-Flash 把长上下文的主要问题拆成输入/输出计算不对称、远端表示压缩、候选索引、持久化与重算、低精度缓存、条件记忆、推测验证和多模态协议等多个账本；只有在固定 revision 与固定 harness 下分别测量，再把账本合并，才能判断架构是否真的改善了生产任务。
