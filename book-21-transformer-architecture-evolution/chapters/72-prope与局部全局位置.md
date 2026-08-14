# 第 72 章 p-RoPE：局部与全局位置处理如何配合

## 72.1 为什么不同层可能需要不同位置尺度

局部窗口关心相邻 token 的精细距离，全局层关心跨段关系。把所有层、所有 head 使用同一套 RoPE 频率，会让局部精度和长距离外推互相牵制。p-RoPE 这类模型特定术语可以作为一个入口，讨论位置处理是否应该按路径、层或 head 分工。

这里先限定证据范围：如果公开资料只给出名称或产品描述，就不能据此推出完整公式、层表和训练配方。本章先讲局部窗口、全局路径和位置连续性这些可迁移的问题；涉及某个具体模型的缩写展开和参数时，读者应把它们视为待核验信息，直到模型卡、技术报告或代码给出直接证据。

## 72.2 RoPE 与局部窗口是两件事

RoPE 对位置 `p` 的 Q/K 做旋转：

```math
q'_p=R(p)q_p,\qquad k'_r=R(r)k_r
```

局部 attention 则通过 mask 限制可见范围：

```math
M_{p,r}=\mathbf{1}[|p-r|\le w]
```

改变 RoPE 不会自动改变窗口，改变窗口也不会自动解决远距离位置表示。两者叠加时，需要同时检查 position id、mask、cache page 和窗口边界。

## 72.3 分层位置抽象

可以用一个教学化 schedule 表示：

```math
\theta_l=\theta_{\mathrm{local}}\quad(l\in L_{\mathrm{local}}),\qquad
\theta_l=\theta_{\mathrm{global}}\quad(l\in L_{\mathrm{global}})
```

局部层可以保留较细的短距离频率，全局层可以使用更适合外推的尺度。真实 p-RoPE 参数化可能完全不同，不能把这个分段式当作公开实现。

## 72.4 一个跨窗口例子

代码文件在前半段定义函数，后半段调用它。局部窗口太小而没有 global 连接时，模型只能通过递归 state 或中间摘要传递函数签名；如果位置 id 在每个窗口重新从 0 开始，跨窗口的相对关系也可能被破坏。

训练/推理都要测局部括号、跨窗口调用、长文档引用和 checkpoint resume。正确的局部路径不等于正确的全局路径。

## 72.5 评估位置策略

固定模型和 prompt，只改变局部窗口、global layer 位置和 position 参数。测试窗口内复制、跨窗口复制、顺序交换、位置中间检索、packed batch 和恢复。记录 recall、短任务 loss、KV/state memory、TTFT、TPOT 和 p99。

如果窗口内任务不变而跨窗口任务下降，优先查 global path、position continuity 和训练数据；如果只在 batch 中失败，查 page offset、padding mask 和请求重排。

## 72.6 工程实现的边界

local/global layer 的 cache layout 可能不同。allocator 要知道哪些层保留完整 KV，哪些层只保留窗口或 state；scheduler 要在请求增长时更新活跃页；prefix cache 共享要绑定 position schedule 和模型 revision。

窗口边界的 off-by-one 会导致重复 token、漏读一项或周期性质量下降。用短的人工构造序列和逐层 logits 比较，通常比直接跑大 benchmark 更容易定位。

## 72.7 面试回答与练习

回答“p-RoPE 解决什么问题”时，应说明它指向一种按局部/全局路径处理位置尺度的思路，位置编码和 attention mask 必须一起分析；通过窗口、位置、层 schedule、长程引用和恢复实验验证，不根据名字猜完整实现。

练习一：为 24 层模型设计 local/global schedule，并标注每层的 cache 类型。

练习二：构造一个能发现窗口边界 off-by-one 的输入。

练习三：解释为什么改变 RoPE 不等于改变可见范围。

### 72.7.1 局部与全局位置的分工

如果局部路径只看窗口 `w` 内的 token，全局路径看更宽的历史，位置处理可以针对两条路径使用不同尺度。教学上可以写成：

```math
\theta_{\mathrm{local}}(p)=f_{\mathrm{local}}(p),\qquad
\theta_{\mathrm{global}}(p)=f_{\mathrm{global}}(p)
```

但位置函数只改变表示，不能单独扩大 mask 的可见范围。若 global path 的 mask 仍限制在局部窗口，换一个 RoPE 也无法读取窗口外 token。

### 72.7.2 边界实验

把答案证据放在局部窗口边缘两侧，轮换窗口起点，并加入一个跨窗口的版本冲突。若模型在窗口内正确、跨边界错误，优先检查 window、reset、global layer 和 position id；若只改变 position scaling 就出现结果变化，说明位置数值与访问路径共同影响能力。

### 72.7.3 serving 和 checkpoint

局部/全局位置方案需要保存 layer schedule、窗口 offset、global position、segment reset 和 cache block 的起始位置。chunked prefill 或 context parallelism 改变块处理顺序时，必须保证逻辑 position 连续。恢复错误一个 offset，可能只影响远程引用，难以通过普通短 prompt 健康检查发现。

### 72.7.4 位置与 mask 的联合消融

可以固定 token 和位置函数，只改变 local/global mask；也可以固定 mask，只改变 local/global 的位置尺度。若仅改变 mask 就能让远程证据可见，说明此前瓶颈是访问范围；若远程 token 可见但引用仍错，才需要继续分析位置表示、训练分布和状态压缩。

三种常见错误要分别测试：local window 的边界 off-by-one、global layer 的 position reset、packed sequence 的跨样本泄漏。每种错误都可能在短文本上不明显，却在长文档的中间位置产生系统性失败。

### 72.7.5 资源取舍

增加 global 层可以提高跨段信息流，却增加显式 KV 和 prefill；缩小 local window 可以节省计算，却可能让局部实体和代码结构断裂；扩大 recursive state 可以保留更多历史，却增加 state bytes 和更新成本。schedule 选择应绑定任务：代码跨文件、文档审计、长日志和连续流的最优路径可能不同。

## 72.8 局部全局位置的训练课程

如果训练一开始就把所有层分成固定 local/global，模型可能只依赖 global 层；如果 global 路径太晚出现，模型又可能学不会跨窗口组合。一个可行的研究流程是先用全局路径建立任务能力，再逐步引入局部限制和位置尺度差异，最后用真实 mask、cache 和 batch 进行验证。

训练样本要覆盖窗口内任务、跨窗口任务、窗口边界任务和长程冲突任务。仅用随机长文本会让模型学到长度统计，却不一定学会跨窗口引用。

## 72.9 位置尺度的数值测试

对每个位置尺度记录旋转角、相位差、attention logits 和不同 dtype 下的误差。位置函数在很长索引下可能出现精度损失，即使 mask 正确，logits 也可能漂移。

可以把位置数值误差写成：

~~~math
E_{\mathrm{pos}}
=\left\|R_{\mathrm{fp}}(p)-R_{\mathrm{ref}}(p)\right\|
~~~

这只是诊断指标，不等于任务错误。最终仍要把数值误差和 evidence recall、引用、短任务回归联系起来。

## 72.10 position id 的连续性实验

构造一个固定 prefix，把它切成三个 chunk，分别用一次性 prefill、连续 chunked prefill 和恢复后的 decode 处理。记录每个 chunk 的起始 position、窗口起点、global layer 的可见范围和最终 logits。

随后故意在第二个 chunk 重置 position id，或把窗口起点多加一，观察错误是否只出现在远程引用。这个实验能把模型位置外推问题与 runtime offset bug 分开。

## 72.11 schedule、位置和 cache 的联合契约

局部/全局位置机制需要一个版本化契约：

~~~text
position_scheme
layer_schedule
window_size
global_layers
segment_reset
cache_layout
model_revision
~~~

prefix cache 命中、checkpoint restore 和 context parallelism 都必须验证这组字段。只要其中一个变化，旧 cache 可能不能复用，即使 token 前缀完全相同。

## 72.12 局部和全局位置的资源分工

局部位置处理可以让多数层关注相对近邻，少数全局通道负责跨段对齐。它降低了每一层都保存完整历史的压力，却把全局通道的层位置、容量和证据传播变成关键设计。

若全局层太少，远程证据可能没有足够的重写机会；若全局层太多，资源收益消失。应固定全局层数量，改变其位置和局部窗口，测局部语法、远距复制、冲突版本、引用支持和 p99。

## 72.13 位置机制与多模态时间轴

对视频和音频，局部/全局位置不只对应 token index，还对应时间戳、帧采样和 chunk offset。文本插入媒体 token 后，position scheme 必须说明跨模态位置是否共享、是否重置、是否使用独立时间轴。

改变采样率或帧数会改变位置序列，即使原始媒体相同也不能直接复用 cache。评估应检查时间定位、跨帧事件和文本问题与媒体证据的对齐。

## 72.14 p-RoPE 的运行时检查条件

运行时需要保存 position scheme、window 起点、global layer、segment reset、cache layout 和 model revision。prefix sharing、context parallel、抢占恢复和量化都必须验证这些字段。

位置公式正确而协议字段错误，模型仍会出现静默错位。因而 p-RoPE 的工程验证不应只测试旋转数值，还要测试 batch 重排、chunk continuation、媒体缓存复用和旧状态拒绝。

## 72.15 局部位置压缩的能力边界

局部位置机制减少了每个层都处理全局距离的压力，但它不能自动恢复被窗口隔开的精确关系。对实体版本、坐标、时间戳和代码标识符，应分别测局部窗口内定位、跨窗口定位、全局层介入和原文回读。

如果全局通道只保存摘要，主题可能仍然正确，数字和限定条件却可能丢失。高风险任务要把关键字段作为显式证据，不能只凭位置模块的平均语言指标。

## 72.16 从位置机制走向局部/全局系统

p-RoPE 的可迁移知识是位置尺度、局部窗口、全局通道和状态连续性之间的关系。真正的系统问题是：窗口边界之外的证据在哪里保存，哪个层负责跨窗口重写，global path 的位置是否与 local path 对齐，以及请求切块、恢复和 prefix sharing 时是否仍然使用同一语义。

可以设计一个小型层表消融：固定总层数，分别改变 global layer 的数量和位置；固定窗口大小，再改变 position reset 与连续 offset；最后加入长文档、版本冲突、视频时间定位和工具参数任务。记录 local exact recall、跨窗口 exact recall、引用支持、global path 使用率、KV/state bytes 和 p99。若全局层增加才恢复远程证据，说明收益来自全局信息路径，而不能简单归因于 RoPE 公式。

运行时还要把 `position_scheme`、`layer_schedule`、`window_size`、`global_layers`、`segment_reset`、`cache_layout` 和 `model_revision` 作为一个版本化契约。具体缩写展开、参数和层表必须由官方模型卡、技术报告或代码支持；缺少一手资料时只讨论这些可测试的通用机制。

## 72.17 局部位置和全局位置的分工

当模型同时使用 local 与 global attention 时，位置机制也可能分层处理。local path 关注窗口内部的相对距离，global path 需要跨窗口的逻辑位置和块偏移。两者不能简单共享一个 position id 而不说明语义。

局部位置处理可以降低每层都表示超长距离的压力，但会引入边界：窗口外的两个 token 只有在某个 global layer、递归 state 或外部 retrieval 中再次相遇，才能建立直接关系。位置机制的目标因此变成“在什么范围内精确、在什么范围内摘要、什么时候回读”。

## 72.18 一个局部窗口的数学模型

设窗口大小为 w，位置 t 的 local 可见集合是：

~~~math
\mathcal V_t^{\mathrm{local}}
=\{j\mid \max(0,t-w+1)\le j\le t\}.
~~~

如果每个 token 只在 local window 内使用 RoPE，距离大于 w 的 token 不会在该层直接比较。global path 可以通过块摘要或稀疏索引扩大可见集合，但必须记录它读取的是原文、latent 还是 summary。

一个常见误区是把 local window 的总上下文长度当成单层的直接可见长度。模型可以经过多层传播跨越窗口，但传播次数、状态容量和误差累积决定最终效果。

## 72.19 p-RoPE 的证据层级

遇到 p-RoPE 这类模型特定名字，资料可分为三层：

1. 官方明确的全称、作用层和配置。
2. 技术报告给出的频率、窗口或训练实验。
3. 根据 local/global 设计推导出的通用教学模型。

第三层可以帮助读者理解，但不能被写成第二层的事实。若只知道模型使用 local/global attention，而不知道 p-RoPE 的具体公式，可以讨论位置尺度和窗口边界，不能编造某个频率分段。

## 72.20 位置偏移和 cache page

服务端的局部窗口经常与 page/block cache 结合。逻辑位置 t、窗口起点、page offset 和本地 page index 需要保持一致：

~~~math
\mathrm{local\_index}
=t-\mathrm{window\_start},
\qquad
\mathrm{page\_index}
=\left\lfloor\frac{t}{P}\right\rfloor.
~~~

若 window_start 在滑动时更新了，却没有同步 position metadata，模型可能在输出上保持流畅，却把跨窗口引用错位。回归测试需要在窗口刚好跨 page、跨 chunk 和跨 preemption 边界的位置放置证据。

## 72.21 全局层的证据预算

全局路径不一定保存所有 token。它可能只保留少量 global token、块摘要或候选索引。设每个 chunk 有 n 个 token，global budget 为 g，则压缩比约为 g/n；但有效能力取决于被选中的内容：

~~~math
P(\mathrm{hit})
=P(\mathrm{evidence\ selected})
\times P(\mathrm{evidence\ preserved})
\times P(\mathrm{query\ reads}).
~~~

任何一个概率低，最终引用都会失败。全局预算应按任务分配，数字、版本、权限和代码标识符需要比一般背景更高的保留优先级。

## 72.22 局部/全局对照实验

建立四个 baseline：全局 RoPE、全局无位置、局部位置加 global summary、局部位置加显式回读。任务包括：

1. 窗口内语法。
2. 跨窗口实体。
3. 多版本冲突。
4. 精确位置和引用。
5. padding、packed batch 和 snapshot restore。

报告窗口大小、global budget、position scheme、证据位置、citation support、TTFT、TPOT 和 page miss。否则“局部位置更快”可能只是使用了更小的输入或更低的输出预算。

## 72.23 长度扩展的失败诊断

如果窗口内任务正常、跨窗口任务下降，先查 global budget 和摘要保真；如果长到某一位置突然崩溃，查 position frequency、page offset 和训练长度；如果只在 batch 中崩溃，查 packed boundary 和 state owner；如果答案正确但位置错误，查 provenance 和全局索引。

这套分层诊断比把所有问题归因于 RoPE 外推更有效，因为同样的表现可能来自模型路径、数据和 runtime。

## 72.24 位置 metadata 的可观测性

位置错误往往不会让服务直接报错，而是让模型在某些长度或窗口边界上悄悄退化。每个请求的 trace 应记录 position scheme、logical offset、window start、global block、page offset、segment/reset、tokenizer/template revision 和 cache schema。

如果输出变化，只看最终文本无法判断是模型质量、position id、page table 还是窗口路由问题。可以对同一输入做 full recompute、cache decode、chunked prefill、batch reorder 和 snapshot restore 五路回放，比较 token-level logits 和位置 metadata。

## 72.25 多模态时间轴不能套文本位置

图像 patch 有二维坐标，音频有采样时间，视频还有帧率、时间戳和镜头边界。把它们简单展平为文本 token 后，模型仍需要知道哪些 token 属于同一帧、同一空间区域或相邻时间段。local/global position 的窗口也应按媒体语义定义，而不是只按 token 数切割。

评估时改变分辨率、帧率、patch 顺序和缺失帧，测空间关系、时间顺序、跨帧引用和长视频中间证据。媒体 processor、位置编码和 cache layout 必须绑定同一个版本，否则“同一视频”可能生成不同的有效位置序列。

## 72.26 位置升级的发布与回滚

位置方案或窗口策略升级时，不能复用旧 prefix/KV/state cache，除非 manifest 明确证明 position、template、dtype、layer schedule 和 schema 兼容。灰度应先跑短文本、长文本、窗口边界、packed batch、工具 JSON 和多模态样本，再开放高风险任务。

若新路径只在少数长度失败，回滚时要切换 position adapter、cache schema、router 和评测验收条件，而不是只恢复权重。旧请求若携带新位置状态，应排空、重算或明确返回不可恢复。

## 72.27 位置机制的最终实验清单

一个可以写入模型卡的长上下文位置声明，至少应提供训练长度、测试长度、位置方案、证据位置分桶、任务类型、tokenizer/template、硬件/runtime、质量指标、KV/状态成本和失败边界。没有这些条件，“支持百万上下文”只能说明接口上限或单项压力测试。

## 72.28 p-RoPE 的工程问题是“谁使用哪种位置”

局部层、全局层和多模态时间轴可能需要不同的位置范围。实现时要明确每个 head/layer 的 position policy、window offset、跨 chunk 规则和 cache metadata；否则同一个 token 在不同路径中可能得到不同旋转。

## 72.29 位置策略的反事实实验

固定模型和训练数据，只替换局部/全局位置分配，比较短文、长文、中间检索、跨窗口复制、视频帧顺序和 cache continuation。若只在长文和多媒体任务改善，不能把它描述成所有任务的通用质量提升；若位置变化影响短文，检查是否破坏了原有局部归纳偏置。

## 72.30 位置失败的诊断信号

position id 不连续、chunk 重置、padding 参与旋转、跨模态时间单位错误和 cache page 复用错误，都会产生类似的输出退化。日志中应保存 position range、mask hash、modality timestamp、layer policy 和 cache key，支持按请求重放。

## 72.31 小结

p-RoPE 的通用学习价值在于理解位置尺度与局部/全局访问的联合设计。局部窗口降低成本，全局路径保留远程通道，显式回读提供证据和版本保证；三者的预算和边界必须在模型、训练与 serving 中一致。具体缩写和参数化仍以公开一手资料为准。
