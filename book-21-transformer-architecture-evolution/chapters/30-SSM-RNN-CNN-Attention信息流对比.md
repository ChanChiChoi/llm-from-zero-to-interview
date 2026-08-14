# 第三十章：SSM、RNN、CNN、Attention 的信息流对比

## 30.1 比较架构要比较信息流

只背“RNN 线性、CNN 局部、Attention 二次、SSM 线性”不够。更有用的问题是：历史在哪里保存；当前 query 能否按内容任意访问；远程信息需要多少层/步传播；训练能否并行；推理状态如何增长；硬件是否有成熟 kernel。

不同架构可能具有相同的 Big-O，却有完全不同的能力和系统表现。比较时应先画出信息流，再写复杂度。

## 30.2 四种基本更新形式

RNN：

~~~math
s_t=f(s_{t-1},x_t),\qquad y_t=g(s_t)
~~~

CNN：

~~~math
y_t=\sum_{j=0}^{w-1}W_jx_{t-j}
~~~

Attention：

~~~math
y_t=\sum_{i\le t}a_{ti}v_i,\qquad
a_{ti}\propto\exp(q_t^\top k_i)
~~~

SSM：

~~~math
s_t=\bar A s_{t-1}+\bar B x_t,\qquad
y_t=C s_t+D x_t
~~~

RNN/SSM 主要维护递归状态，CNN 维护固定窗口，attention 保留可寻址历史。Mamba 等模型让 SSM 的部分参数输入依赖，缩小了递归状态与内容选择之间的差距。

## 30.3 小白直觉：四种查资料方式

RNN 像每读一页就写一份摘要；CNN 像只看最近几页；Attention 像保留整本书并按问题翻页；SSM 像用多个时间尺度的统计滤波器持续更新摘要。

如果问题是“最近十秒的信号是否异常”，CNN/SSM 可能高效；如果问题是“全文第 18 页的数字与第 70 页的条款是否冲突”，attention 或外部检索更自然；如果问题是持续预测，固定 state 可能更适合。

## 30.4 远程路径长度

在宽度为 `w` 的卷积中，距离 `d` 的信息至少需要约 `ceil(d/w)` 层（忽略 dilation）；RNN/SSM 的递归路径每步都有一条状态边，但信息要经过状态混合；full attention 在单层就可建立远程边：

~~~math
\ell_{\mathrm{path}}^{\mathrm{CNN}}\approx\left\lceil\frac{d}{w}\right\rceil,\quad
\ell_{\mathrm{path}}^{\mathrm{Attention}}\approx1
~~~

路径短不等于可用。attention 的边由内容和位置决定，SSM 的 state 可能在每一步衰减，CNN 的共享局部模式可能对噪声更稳。还要测信号是否在表示中保留。

## 30.5 计算和状态表

| 架构 | 训练主要路径 | 推理状态 | 内容寻址 | 长度成本 |
|---|---|---|---|---|
| RNN | scan/并行化较难 | 固定向量 | 受 state 限制 | 线性 |
| CNN | 并行卷积 | 窗口/缓存 | 固定局部 | 近线性 |
| SSM | 卷积或 scan | 固定结构 state | 输入依赖时增强 | 线性/近线性 |
| Attention | 大矩阵并行 | KV 随长度 | 强 | full 近二次 |

表格是结构层抽象，实际还要加入 batch、head、kernel、通信和 cache。

## 30.6 一个合成信息流实验

构造三类任务：局部模式检测、远程 key-value 复制、连续流中的异常状态。让每个模型拥有接近参数量和相同训练 token，再改变序列长度与噪声。

~~~python
def make_stream(length, key_at, value):
    stream = [0] * length
    stream[key_at] = value
    stream[-1] = 1  # 查询标记
    return stream


for position in (0, 10, 100, 1000):
    print(position, make_stream(1024, position, 7)[-4:])
~~~

这个 demo 只是生成探针；真正实验需让模型输出 value，并测 exact recall、state norm、GPU time 和恢复一致性。不要用一个局部分类任务代表所有长程能力。

## 30.7 训练并行与推理吞吐

Transformer 在训练时可以并行处理所有位置，适合大矩阵乘；RNN/SSM 的递归形式存在时间依赖，需要卷积、scan、chunk 或硬件专用实现。推理时情况反过来：Transformer 要读取增长的 KV，SSM/RNN 只更新 state，但小状态更新可能不能充分利用 GPU。

因此“训练快”和“推理快”可能属于不同架构。音频实时流的 batch 小、状态持续；离线 pretraining 的 batch 大、序列齐，结论不能互换。

## 30.8 状态压缩和精确检索

设历史 `H` 被递归模型压成 `S=f(H)`。如果存在不同历史 `H_1,H_2` 使 `f(H_1)\approx f(H_2)`，而未来 query 需要区分它们，就会发生信息碰撞。attention 的显式 K/V 降低了这种碰撞概率，却承担更大内存。

可通过历史替换实验测量：保持主题相同，只交换一个数字或实体，观察最终 query 能否区分；再增加干扰 token、冲突版本和多轮状态恢复。

## 30.9 混合架构的现实

混合架构让 attention 负责可寻址 global 证据，SSM/CNN 负责局部和流式处理。层级、head 或 token 级路由可以不同，但需要明确状态接口和训练目标。混合并不是把两个模块简单串联：不同路径的尺度、归一化、位置和 residual 要匹配。

系统上要管理两类 cache：显式 KV 和递归 state；scheduler 还要知道哪些层需要多少状态、如何 snapshot、如何 rollback。混合提高表达和选择空间，也提高验证成本。

## 30.10 常见失败模式

错误包括用理论复杂度替代硬件 profile；用 RNN 的“固定 state”描述所有 SSM；把 CNN 的窗口误说成不会遗忘；把 attention 权重当真实证据使用；训练时 full attention、推理时 local/state；以及 padding 或 chunk 边界污染状态。

排查应分三层：数学更新是否正确，mask/state contract 是否一致，kernel/scheduler 是否实现了声明的路径。每层都要有 reference test 和长序列 regression。

## 30.11 机制与边界：用任务的未来查询分布选架构

若任务未来 query 主要询问近邻、趋势或状态，局部/递归先验能提高数据和算力效率；若 query 会随机指向历史细节，显式 attention 或外部 retrieval 更适合。架构没有脱离 query 分布的“最优”。

实验应报告不同任务的 Pareto 前沿，而不是一个总分。尤其要把长程精确 recall、工具参数、引用正确率和 safety edge case 从平均语言 loss 中分离出来。

## 30.12 面试追问、误区与练习

**问：SSM 和 attention 的最根本区别是什么？**

标准回答：attention 在当前 query 到来时对显式历史做内容寻址；SSM 通过递归状态压缩历史，再从 state 读取。前者访问灵活但状态随长度增长，后者成本稳定但有信息压缩边界。

**问：CNN 为什么仍可能用于序列模型？**

标准回答：局部、规则、硬件友好的归纳偏置对音频、局部模式和低延迟流有价值；远程依赖可以通过 dilation、层叠、attention 或外部记忆补充。

常见误区包括只背复杂度、把状态固定当作能力固定，以及忽略训练和推理的不同硬件路径。

练习：为同一流式异常检测任务画出 RNN、CNN、SSM、attention 的状态图，列出每一种的故障恢复、长程能力和 serving 指标。

## 30.13 用查询类型选择信息流

局部语法、连续信号和固定窗口预测更适合卷积或 SSM；需要当前 query 临时选择远处 token 的任务更适合 attention；需要跨 chunk 延续状态的任务还要检查 state lifecycle。

可以把查询类型分成 local、sequential、random-access 和 external-evidence 四类，再为每类记录所需的历史访问能力。架构比较不应只问“谁的复杂度低”，而应问“它是否提供任务真正需要的访问原语”。

## 30.14 混合系统的验证顺序

先测试每条路径的单独能力，再测试两条路径的接口，最后测试完整路由和回退。若一开始只跑完整模型，某条路径的 bug 会被其他路径掩盖，出现偶发错误时很难归因。

接口测试包括 shape、位置、状态 reset、padding、batch 重排、量化和 speculative 回滚。信息流不同，runtime contract 也不同。

## 30.15 四类架构的状态图

可以用同一个序列任务画四张状态图。RNN/SSM 在每个时间步把输入写入状态；CNN 保存窗口或通过层叠扩大感受野；attention 保存显式历史并在 query 到来时重新计算权重；retrieval 把历史放在模型外部，按查询返回原文或结构化证据。

这四种状态图对应四种故障模式：递归状态可能串租户或恢复丢失，窗口可能发生边界错位，KV 可能耗尽显存，外部索引可能版本过期或权限错误。架构比较如果只写理论复杂度，就看不到这些线上故障。

## 30.16 复杂度表还不够：写出状态预算

设序列长度为 T、hidden 维度为 d、state 维度为 s、窗口为 w，粗略的单请求历史存储可写成：

~~~math
M_{\mathrm{RNN/SSM}}=O(s),\qquad
M_{\mathrm{CNN}}=O(wd),\qquad
M_{\mathrm{Attention}}=O(Td)
~~~

这只是激活或状态的数量级，实际还要加层数、KV heads、dtype、分页碎片、workspace 和并发数。SSM 的 s 如果很大，或者每层都有多个矩阵状态，固定状态也可能成为显存瓶颈；CNN 的窗口如果由长卷积或多阶结构实现，也不一定只占 wd。

因此容量实验应同时报告每请求 bytes、batch 下的总 bytes、state update 带宽和恢复大小。用 O(1) 代替真实资源表，会让读者误以为所有递归模型都天然适合高并发。

## 30.17 选择架构的实际决策顺序

先判断任务需要的是局部模式、持续状态、随机历史访问还是有来源的外部证据；再判断请求是离线长 prefill、短交互 decode 还是连续流；最后才比较模型质量、kernel 和生态。若任务包含多个阶段，可以让不同路径承担不同阶段，而不是强行选择单一主干。

面试回答中，最有说服力的结论通常是条件句：“在固定训练预算和硬件下，若查询以趋势和近邻为主，我会优先评估 state/conv；若需要任意实体引用和多证据比较，我会保留 attention/retrieval；上线前用 exact recall、p99、state bytes、恢复一致性和单位成功成本做验收条件。”

## 30.18 用查询类型选择信息流

可以先按未来查询划分任务，再选架构：局部模式查询偏向 CNN/局部 attention，长流趋势和有限状态查询偏向 SSM/RNN，任意历史精确检索偏向 full attention 或 RAG，跨模态对齐可能需要显式 cross-attention。这个顺序比先选一个流行模块再寻找适用任务更可靠。

对同一数据集，至少构造四个 probe：局部邻域、远距精确复制、状态重置和多证据合并。测每种路径的远程距离、状态大小、训练吞吐、decode 吞吐、p99 和恢复成本。平均分类准确率无法说明模型是否真的具备任意历史访问。

## 30.19 混合架构的接口优先

混合模型不是把几种层堆在一起就完成了。层之间要定义 hidden shape、position、mask、state/cache、residual scale、chunk 边界和训练/推理切换。Serving 还要管理显式 KV、递归 state、latent cache、prefix sharing 和 preemption。

调试时先验证接口不变量，再分析模块能力：同一输入在 full、chunk、step 和恢复路径的 logits 是否一致；padding 是否更新 state；batch reorder 是否保留 request owner；cache key 是否包含模型和位置 revision。混合层名字不能替代这些契约。

## 30.20 复杂度表之外的实际决策

理论表可以写出 `O(T^2)`、`O(T)` 或固定 state，但实际系统还受常数、内存层次、kernel 成熟度、batch、长度分布和协议影响。一个线性算法在小 batch 上可能因为 kernel 启动和状态读写不如 dense；一个二次算法通过 FlashAttention 和高效 KV 管理可能满足当前 SLO。

决策应以目标 workload 的质量—成本—延迟 Pareto 曲线为准，并保留 fallback 和回滚路径。架构研究的结论应说清“在哪个任务、长度、硬件和 runtime 条件下成立”，而不是把复杂度记号当作生产收益。

## 30.21 同一任务的四路信息账本

为了让架构对比不止停留在表格，可以为一个长日志任务建立四路账本：RNN 记录固定 state，CNN 记录感受野和边界，SSM 记录 state transition 与 scan，attention 记录 KV、可见连接和 query-to-evidence 路径。每一步写入的对象、读取的对象和丢失的对象都要明确。

例如，日志中第 3,000 个事件定义了一个错误码，第 50,000 个事件要求引用它。RNN/SSM 可能以固定 state 携带摘要，CNN 需要经过多层扩大感受野，attention 可以直接访问 KV，RAG 则先从索引召回原文。若答案只要求判断是否出现过，摘要足够；若要求返回精确错误码和来源，必须增加可寻址路径或外部证据。

这个例子还显示，系统可以组合信息流：SSM 负责扫描全部日志并触发异常候选，retrieval 返回原始片段，少量 attention 负责版本和引用对齐。组合的代价是接口、state snapshot、权限、回滚和 trace；因此混合架构的收益必须减去这些治理成本。

## 30.22 一个可复现的信息访问实验

可以用同一份长日志构造四个 probe：局部邻域预测、远距数字复制、状态 reset 后继续、三段证据合并。对 RNN/SSM、CNN、attention 和 retrieval 分别记录它们保存的状态、可访问的历史、计算成本和证据位置。

实验不必先训练四个完整大模型。教学阶段可以用可控的 toy state、局部卷积、显式 KV 和外部索引验证访问原语；生产阶段再用相同任务切片和真实 kernel 重跑。关键是保持输入、输出预算、长度、硬件和 verifier 一致。

## 30.23 worked example：复杂度相同也可能体验不同

假设两个方案都标注为线性历史成本。方案 A 的 state 更新每 token 需要一次高带宽读写；方案 B 使用局部块 kernel，能在 batch 中合并访问。短请求下 A 可能更快，长流和高并发下 B 可能更稳定。反过来，一个 dense attention 方案如果 prefix cache 命中率高，也可能在真实工作负载中胜过理论更省的方案。

因此报告要同时给出 prefill、decode、KV/state bytes、kernel fallback、batch 利用率、p95/p99、恢复时间和单位成功成本。复杂度只说明随长度增长的趋势，不说明常数、内存层次和协议开销。

## 30.24 访问原语决定回退策略

当 state 路径无法支持精确引用时，回退可以是扩大显式窗口、触发 retrieval、请求用户提供证据或转到更高成本模型。回退条件应由任务风险和证据类型决定：普通摘要可以接受压缩近似，法律、财务、代码发布和权限判断需要原文回读与引用支持。

回退也必须进入容量规划。若所有长请求同时转 retrieval 或 attention，系统可能出现新的队列和 KV 峰值；因此要设定 fallback rate、额外 token、p99 和成本预算，不能把回退当成无限资源。

## 30.25 四种信息流的选择条件

SSM 可参考 S4（https://arxiv.org/abs/2111.00396），attention 可参考 Transformer 原论文（https://arxiv.org/abs/1706.03762）。表格和公式是统一分析框架，具体模型需按实现核验。

四类架构的差异可以归结为“历史以什么形式保存、未来如何访问”。沿着信息流回答问题，比背一组复杂度数字更能支撑工程和面试判断。

## 30.26 四种信息流的状态不变量

比较架构时，可以先为每种路径写出状态不变量。RNN/SSM 的不变量是：每个时间点只有固定大小 state 和当前输入；CNN 的不变量是：输出只依赖有限感受野或若干层扩张后的邻域；attention 的不变量是：查询可以按照 mask 读取允许的历史表示；RAG 的不变量是：答案只能在检索候选与模型先验的组合中生成，且外部来源可以独立审计。

这些不变量决定了错误的样子。state 路径容易出现遗忘、覆盖和恢复不一致；CNN 容易出现边界和感受野不足；attention 容易出现 KV 成本、lost-in-the-middle 和提示注入；RAG 容易出现召回漏失、权限错误和引用不支持。用不变量组织调试，比把所有错误都称为“模型幻觉”更有用。

## 30.27 一个统一的历史访问公式

设任务在时刻 `t` 需要访问历史证据 `e`，可以把成功概率粗略拆成：

```math
P_{\mathrm{success}}(e,t)
=P_{\mathrm{reachable}}
 P_{\mathrm{preserved}}
 P_{\mathrm{addressed}}
 P_{\mathrm{verified}}.
```

对于 RNN/SSM，`reachable` 通常由 state transition 决定，`preserved` 受衰减和容量影响，`addressed` 受读出层影响；对于 attention，`reachable` 由 mask 和长度决定，`preserved` 受层间混合影响，`addressed` 受 query/heads 影响；对于 RAG，前三项分别对应索引召回、候选保留和查询匹配，`verified` 还包括引用/权限检查。

这个分解不是统计独立性假设，而是一个错误归因框架。若证据根本不可达，增加 decoder 层数可能无效；若可达但读错，应该检查表示、query 或训练；若答案正确但引用无效，问题在 provenance 和后处理。

## 30.28 何时采用混合信息流

混合系统适合任务本身包含不同阶段：先对长流做低成本扫描，再对候选片段做精确比较；先用局部卷积提取视觉/语音短程模式，再用 attention 做跨模态对齐；先用状态模型预测下一步，再用工具或检索验证事实。这样每个模块承担自己擅长的信息访问。

混合的代价是状态和协议变多。每次交接要定义 hidden shape、position、mask、state reset、cache owner、错误码和回退；训练要决定交接是否可学习，服务要决定交接失败时重算哪一段。若交接没有明确契约，系统可能在平均 benchmark 上看似提升，却在 chunk 边界、取消和工具调用时失效。

可以用接口预算表达：

```math
C_{\mathrm{hybrid}}
=\sum_i C_i+\sum_{(i,j)}C_{i\to j}
 +C_{\mathrm{state}}+C_{\mathrm{governance}}.
```

只有质量收益能够覆盖额外交接成本，混合架构才值得采用。

## 30.29 结论要绑定未来查询分布

同一训练语料可以被不同任务查询出完全不同的能力排序。若未来查询主要是局部预测，CNN/局部 attention 的感受野可能足够；若查询是趋势、累计和有限状态，SSM/RNN 的固定 state 可能更省；若查询随机指向文档中的任意实体，full attention 或 RAG 更有优势；若查询混合存在，路由或混合架构比单一结论更合理。

因此架构报告应先公布 query distribution：查询距离、证据数量、是否需要原文引用、是否跨模态、是否有外部工具、输入输出长度和风险等级。再在相同分布下比较质量、吞吐、状态/ KV、p99、恢复和成本。离开未来查询分布谈“哪个架构更先进”，本质上是在用一个任务的结果替代所有任务。
