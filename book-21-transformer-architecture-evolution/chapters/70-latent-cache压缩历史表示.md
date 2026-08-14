# 第 70 章 Latent Cache：压缩历史表示的收益与信息瓶颈

## 70.1 为什么 cache 可以不保存完整 K/V

完整 K/V 能保留每个 token 的高维表示，却占用大量显存。latent cache 把历史投影到较低维 `c_t`，运行时通过投影或专用 attention 使用它。它和 prefix cache 的“复用相同 K/V”不同，重点是改变历史表示本身。

压缩换来的不是免费空间。低维表示可能无法同时保留实体、数字、顺序、引用和局部语法；读取阶段还要花 projection、解码或特殊 kernel。真正的 trade-off 是“少保存多少”与“丢失哪些可验证细节”。

## 70.2 压缩公式

一个简单的线性压缩/恢复模型为：

```math
c_t=W_dh_t,\qquad
\hat k_t=W_kc_t,\qquad
\hat v_t=W_vc_t
```

显式 cache 每 token 的理想元素量约为 `2H_{kv}d_h`，latent cache 约为 `d_c`。理想 payload 比例：

```math
\rho=\frac{d_c}{2H_{kv}d_h}
```

如果 `H_{kv}=8,d_h=128,d_c=512`，则 `\rho=512/(2\times8\times128)=0.25`，理想 payload 约减少 75%。真实实现还要加 scale、metadata、对齐、projection workspace 和可能保留的高精度路径。

## 70.3 表示压缩与访问压缩

表示压缩改变每个历史 token 的表示维度；访问压缩则只读取一部分 token、摘要块或候选页。前者可能造成重建误差，后者可能漏掉正确证据。两者可以组合，但评测要能区分“证据被存坏了”和“证据没有被访问”。

## 70.4 信息瓶颈的直觉

一段长文本中，主题摘要可能只需要少量维度；但精确数字和版本差异需要保留更多细节。latent dimension `d_c` 太小，模型可能把“错误率 2%”和“错误率 20%”压成相似状态；太大则显存收益下降。

因此选择 latent dimension 不能只看平均 perplexity。要看数字准确率、代码测试、证据召回、引用支持和冲突识别。

## 70.5 一个四段文档实验

文档分为背景、实验设置、结果表和结论。问题分别要求主题摘要、找结果数字、比较两版本和引用表格行。对 full KV、latent 256、latent 512、latent 1024 做对照。

如果主题摘要都正确，但 256 在数字和版本比较上明显下降，说明压缩保留了语义主题却丢了高精度关系。此时可以让结果表和 global evidence 走显式路径，而不是把所有 latent 维度翻倍。

## 70.6 读取与 kernel 成本

latent cache 的服务路径是：

```text
encode -> compress -> store -> read/project -> attend
```

压缩减少了存储和带宽，却可能增加 projection 和 kernel launch。若每次 decode 都把 latent 恢复成完整 K/V，部分收益会被恢复成本抵消；如果直接在 latent 空间 attention，模型表达和 kernel 需要重新设计。

压测要分别记录 prefill compression、decode projection、memory bandwidth、kernel occupancy、TTFT、TPOT 和并发，而不是只看 cache 文件大小。

## 70.7 cache 的版本和回滚

latent cache 需要保存 projection revision、dtype、scale、layout、position range 和 token range。speculative decode 产生临时 latent 时，只在 target 接受后 commit；拒绝分支必须释放，不能因为内容“只是压缩向量”就忽略状态污染。

跨模型版本不能因为 shape 相同就复用旧 latent。投影矩阵、位置处理和归一化变化都会改变语义。

## 70.8 量化的区别

KV quantization 仍保存每个 token 的 K/V，只改变精度；latent cache 改变表示维度和读取路径。两者可以叠加，但误差来源不同。实验应分别比较 FP16 latent、低精度 latent、FP16 projection 和低精度 projection，确定退化来自哪里。

## 70.9 评估与诊断

按 token 类型做 sensitivity test：替换数字、实体、表格列名、代码标识符和工具参数，检查答案和引用变化。还要把证据放在上下文开头、中间和结尾，观察压缩是否放大 lost-in-the-middle。

如果平均 loss 好而 citation support 下降，说明压缩保留了可生成的主题，却损失了可归因的细节。生产系统应允许回退到显式检索或高精度 cache。

## 70.10 常见失败

把 latent cache 当 prefix cache；只比较存储体积；压缩丢数字和来源；prefix hash 没有 projection 版本；跨版本复用；量化与压缩误差叠加；speculative reject 不回滚；fallback 时没有清空旧 latent。

## 70.11 面试回答与练习

回答“latent cache 和 KV cache 的区别”时，应说 KV 保存可直接读取的 K/V，latent cache 保存压缩历史并在读取时投影或用专用 attention；latent 省状态但有信息瓶颈和额外计算。要按精确任务和真实 kernel 评估，而不是只看理论压缩比。

练习一：实现一个线性 encoder/decoder toy cache，比较维度、重建误差和 attention 输出。

练习二：设计一个能发现数字信息损失的测试集。

练习三：列出 latent cache snapshot/restore 必须保存的字段。

### 70.11.1 压缩率和重构误差的关系

设原始历史表示为 `H`，压缩器为 `C`，重构或读取映射为 `D`，则：

```math
\hat H=D(C(H)),\qquad
E_{\mathrm{repr}}=\lVert H-\hat H\rVert
```

表示误差小不等于任务误差小。一个小数字、版本标识或引用边界可能在整体 MSE 中占比很小，却决定最终回答是否正确。评估应把重构误差与任务级 evidence recall、citation support 和结构化字段准确率联系起来。

### 70.11.2 latent 读取的额外成本

显式 KV 可以直接按 query 读取；latent cache 可能需要投影、解码或专用 kernel。若压缩节省的显存带宽小于投影和通信成本，TPOT 不一定改善。latent state 还可能不适合随意拼接、分割或跨 request 共享，allocator 需要理解其 shape 和生命周期。

### 70.11.3 质量消融

按 latent dimension、量化精度、压缩频率和读取路径做消融。任务至少包含局部问答、远程数字、跨段冲突、代码长文件和工具历史。若只有高维 latent 才保持引用能力，系统可以对 global 层使用高维、对局部层使用低维，而不是统一设置。

### 70.11.4 snapshot 的一致性

snapshot 必须与 position、层编号、模型 revision、压缩器版本、dtype 和 request owner 绑定。恢复时先验证 checksum 和 shape，再把 state 重新挂接到 scheduler；压缩器升级不能静默读取旧 latent。失败时应能重算或回退显式路径。

## 70.12 压缩率和任务保真不是同一个坐标

压缩率只描述状态大小变化，不能描述信息价值变化。两个压缩器都把历史缩小四倍，一个可能保留数字和引用，另一个可能只保留主题；二者的产品价值完全不同。

可以绘制二维曲线：横轴是状态 bytes 或压缩 ratio，纵轴分别画主题质量、exact match、evidence recall、citation support 和 p99。若只画一个总分，关键任务的损失可能被普通摘要样本掩盖。

## 70.13 选择性保真

实际系统可以把历史分成普通背景、结构化字段、最新事件和高风险证据。背景使用高压缩 latent，结构化字段保留精确表示，最新窗口保留完整 KV，高风险查询触发原文 retrieval。

一个简单的路由函数为：

~~~math
p_t=\mathrm{Route}(value(t),risk,query)
~~~

value 不应只由模型自己决定。可以利用文档类型、字段、用户权限、任务标签和历史失败率，避免模型为了节省成本把重要证据压缩掉。

## 70.14 压缩器的训练目标

如果压缩器只用重构误差训练，可能保留平均语义而删除稀有但关键的字段。训练数据应加入数字、否定、版本、表格、代码和冲突样本，并对这些字段提高权重。

评估时要把压缩器和主模型分开消融：原文直接 attention、固定摘要、学习 latent、latent 加原文回读。这样才能判断问题来自压缩器、主模型还是回读策略。

## 70.15 压缩 cache 的错误预算

可以把 latent cache 的总错误拆为四类：投影误差、量化误差、访问误差和协议误差。它们的表现不同：投影误差常集中在高频细节，量化误差可能随长度累积，访问误差表现为漏读，协议误差表现为错位或跨版本污染。

~~~math
E_{\mathrm{total}}
\approx E_{\mathrm{projection}}
+E_{\mathrm{quantization}}
+E_{\mathrm{access}}
+E_{\mathrm{protocol}}
~~~

这个式子用于排查，不是严格可加定理。先分别运行高精度全显式、低精度全显式、高精度 latent 和低精度 latent，才能判断是哪种误差主导。

## 70.16 长文档的双通道读取

一个实用流程是先用 latent 读取完成粗筛，再由 query 生成候选来源范围，最后用显式原文或 RAG 回读高价值片段。粗筛阶段追求覆盖和低成本，回读阶段追求 exactness 和 provenance。

回读结果应包含文档 revision、段落范围、原始 token 或字符偏移，以及压缩表示中的候选分数。没有这些字段，最终答案即使正确，也无法判断它是否真正支持了引用。

## 70.17 压缩比不是唯一目标

设原始 cache 为 B_full，latent cache 为 B_latent，压缩比可以写成：

~~~math
\rho=\frac{B_{\mathrm{full}}}{B_{\mathrm{latent}}}
~~~

但真正的系统收益还要扣除投影、临时 buffer、索引、provenance 和回读成本。一个压缩比很高的实现，如果 projection kernel 读放大严重，TPOT 可能反而上升。

评估应画 memory、latency、exact recall、citation support 和单位成功成本的多维曲线，而不是只报一个 rho。

## 70.18 关键字段的压缩验收条件

数字、否定、版本、时间、权限和代码标识符通常比主题摘要更脆弱。训练或评估压缩器时，应对这些字段建立专项样本，并要求 latent 粗筛能够触发原文回读。

若压缩后只保留“文档讲了什么”，却丢掉“哪个版本、何时生效、例外条件是什么”，系统会产生流畅但不可审计的答案。字段级 recall 比整体相似度更适合高风险任务。

## 70.19 压缩 cache 的回滚和迁移

推测解码、请求分叉和跨 worker 迁移都要求 latent 与投影状态一致。候选 token 被拒绝时，要恢复压缩表示、position、projection workspace 和 provenance；迁移时要确认设备、dtype、压缩器 revision 和 layout。

可以保留一段显式原文作为安全窗口，其余历史使用 latent。这样提高了回滚和回读可靠性，却会减少可压缩比例，最终仍需用任务和容量曲线做选择。

## 70.20 从压缩表示走向服务回退

Latent cache 是表示压缩与 serving 设计的结合。它可以降低历史状态，却把压缩误差、projection kernel、版本和回滚引入系统。上线时不应只有“latent 可用/不可用”两个状态，而应定义逐请求回退策略：低风险普通背景可以继续使用 latent；数字、否定、版本、权限和工具参数等高价值字段触发原文回读；checksum、revision 或 shape 不一致时直接重算或走显式 KV。

一个简单的回退判据可以写成：

```math
G_{\mathrm{latent}}
=G_{\mathrm{version}}
\land G_{\mathrm{checksum}}
\land G_{\mathrm{evidence\_recall}}
\land G_{\mathrm{latency}}
```

其中 `G_evidence_recall` 不是模型自评，而是通过带来源的字段测试或 verifier 检查。回退本身也要计费：projection、原文读取、额外模型调用和重试可能让单位成功成本上升。压测应比较全显式、全 latent、按风险回退和 cache 失效四种模式，报告 memory、TPOT、引用支持、回退率、恢复成功率和 p99。具体 latent layout 和是否恢复 K/V，以对应模型和引擎实现为准。

## 70.21 latent cache 的信息瓶颈

设原始历史表示为 H，压缩器为 C，latent 为 Z=C(H)，读取器为 R。系统输出可以抽象成：

~~~math
Z=C(H),\qquad
\hat H=R(Z),\qquad
y=f(q,\hat H).
~~~

只要 Z 的维度或数量小于 H，系统就存在信息瓶颈。目标不是重构每个 token，而是在给定任务集合上保留足够的信息：

~~~math
\min_C\ 
L_{\mathrm{task}}(f(q,C(H)))
+\lambda\operatorname{cost}(C(H)).
~~~

如果只用平均语言 loss 训练或评估，压缩器可能保留流畅主题，却删除数字、否定和版本条件。高风险场景必须把这些字段单独纳入质量验收条件。

## 70.22 token latent 与 summary latent

latent cache 至少有两种形态：

1. token latent：每个历史 token 都有一个较小 latent，保留相对位置和较细粒度寻址。
2. summary latent：多个 token 聚合成一个摘要块，缓存更小但地址和细节更粗。

token latent 的压缩比可能没有 summary 高，但更容易做局部回读；summary latent 适合重复背景和主题摘要，却不适合直接承载精确表格。系统可以把两者组合：近期 token 保留 token latent，远端重复背景使用 summary latent，关键字段单独保存原文索引。

## 70.23 压缩比和有效能力

定义表示压缩比：

~~~math
\rho=\frac{\mathrm{bytes}(Z)}{\mathrm{bytes}(H)}.
~~~

rho 越小，理论缓存收益越大，但质量曲线不一定单调。常见情况是压缩比从 1 降到 0.5 时几乎不损失质量，继续降到 0.1 后数字和冲突任务突然崩溃。这说明信息损失可能集中在少数关键字段，而不是平均发生。

因此报告应画 quality-versus-rho 曲线，并按任务类型和证据位置分桶。只报一个压缩比会把“能压缩主题摘要”和“能压缩法律证据”混为一谈。

## 70.24 provenance 不能被压缩掉

如果 latent 只保存向量，没有来源范围，后续模型即使生成了正确结论，也无法说明它来自哪份文档。每个 latent block 至少应绑定：

~~~text
source document
source span or token range
document revision
position range
compressor revision
dtype and quantization
checksum
~~~

provenance 不一定进入模型 hidden，但必须存在于 serving metadata。压缩器返回候选时，retrieval 或 verifier 可以沿来源回读原文；没有来源，系统只能把语义相似当作证据。

## 70.25 动态压缩比

所有请求使用相同压缩比通常不是最优。可以按 query 风险、任务类型和证据要求选择：

| 任务 | 压缩策略 |
| --- | --- |
| 普通主题摘要 | 较高压缩比 |
| 多文档比较 | 中等压缩比并保留版本 |
| 代码 patch | 关键文件显式或低压缩 |
| 安全/权限审计 | 原文回读和字段验收条件 |
| 长期对话偏好 | summary latent + 用户可见 memory |

动态策略会增加路由和 batch 复杂度。若压缩比不同导致 kernel shape 碎片化，平均显存可能下降而吞吐下降。需要把 quality、cache bytes、batch 分裂和 p99 一起测。

## 70.26 latent 与位置的关系

压缩器必须明确位置处理发生在压缩前还是压缩后。若 hidden 已经含有位置相关信息，直接跨位置复用 latent 可能改变语义；若位置单独保存，则读取器需要在正确的 position 上恢复。prefix cache 的 key 因而至少包含 token prefix、position scheme、model revision 和 adapter。

长上下文中，位置误差可能表现成“latent 记忆差”。诊断时应先做同一内容、不同起始 position 的对照，再做不同内容、同一 position 的对照。前者变化大，说明位置路径或 cache offset 有问题；后者变化大，才更可能是内容压缩能力问题。

## 70.27 量化顺序

压缩和量化有两种顺序：

~~~text
H -> compress -> quantize latent
H -> quantize hidden -> compress
~~~

两者误差分布不同。先压缩再量化可以让 scale 适应 latent 分布，但可能放大压缩后少数重要通道；先量化 hidden 再压缩，压缩器接收的信号已经有量化噪声。部署时应固定顺序并用同一 calibration set 比较，而不是只看最终 bit width。

## 70.28 回读和 fallback

latent 路径应能回答“何时不信任压缩”。可用信号包括：

1. query 涉及数字、版本、否定、权限或引用。
2. latent 检索得分低或多个候选分歧大。
3. verifier 发现 claim 没有 source span。
4. compress revision 与当前模型不匹配。
5. latent decode 产生异常 norm 或 NaN。

fallback 可以扩大 latent、保留更近窗口、读取显式 KV、调用外部 RAG 或直接拒答。高风险任务宁愿增加一次回读，也不能把不可追溯的摘要当作完整证据。

## 70.29 snapshot 与 speculative 的事务

推测解码会先让 draft 路径产生候选。若候选被 target 拒绝，latent cache 也必须回滚到被接受前缀。可以把每次追加标为 transaction：

~~~text
begin(prefix_revision)
append tentative latent
verify target tokens
commit accepted prefix
rollback rejected suffix
~~~

snapshot 不仅保存 latent 数组，还要保存 logical position、source range、量化 scale、压缩器 revision 和 provenance。只恢复数组而不恢复 metadata，会产生难以发现的引用错位。

## 70.30 端到端评测

latent cache 评测至少包含 full KV、latent-only、latent-plus-explicit verification 和外部 RAG 四条路径。指标包括：

~~~text
cache bytes per token
peak GPU memory
projection FLOPs
TTFT
TPOT
P95/P99
evidence recall
exact numeric accuracy
citation support
fallback rate
restore error
unit successful-task cost
~~~

如果 latent-only 质量高但 citation support 低，应增加 provenance 和回读，而不是继续追求压缩；如果质量不变但 p99 变差，应查投影 kernel 和不规则访问；如果显存没有下降，应查临时 buffer 和 metadata。

## 70.31 latent cache 的压缩比不是唯一目标

压缩比高但读取投影昂贵、量化误差大或无法共享时，端到端收益可能为负。除了 `M_latent/M_kv`，还要报告每 token 读取 FLOPs、内存带宽、cache hit、恢复时间和任务质量。对低并发短请求，固定的投影开销可能比省下的 cache 更大。

## 70.32 压缩表示的可逆性和证据边界

latent 通常不是原文的可逆编码，不能把它当成审计证据或删除原始数据后的等价备份。需要引用、删除和人工复核的系统应保留受控原文或可验证索引；latent 只作为模型计算状态。租户删除时，latent cache、投影临时 buffer 和相关日志都要进入删除清单。

## 70.33 量化、迁移和 speculative 的组合测试

低精度 latent 在普通 decode 上可能稳定，在长序列迁移或拒绝候选回滚时放大误差。测试应覆盖量化前后、snapshot/restore、worker migration、accepted/rejected speculative prefix 和 batch 重排，并比较 state checksum、logits 和最终协议事件。

## 70.34 小结

Latent cache 的本质是有约束的信息压缩。它的价值不在于把数字变小，而在于在质量验收条件下减少显存、带宽和调度压力。有效实现必须同时解决压缩误差、位置、量化、来源、回滚和 fallback；任何只展示压缩率的方案，都没有证明它能用于真实长上下文任务。
