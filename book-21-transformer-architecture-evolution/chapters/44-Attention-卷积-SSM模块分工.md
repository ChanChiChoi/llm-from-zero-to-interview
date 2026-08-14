# 第四十四章：Attention + Convolution + SSM 的模块分工

## 44.1 三种模块分别擅长什么

Attention 擅长内容条件的全局对齐；卷积擅长局部、规则和高吞吐混合；SSM 擅长按时间维护压缩状态和长流扫描。把三者放到同一个系统中，关键是定义接口和职责，而不是堆叠名字。

一个合理的假设是：卷积处理短程模式，SSM 处理持续趋势和流式记忆，attention 处理稀疏高价值证据。这个假设必须通过消融验证。

## 44.2 信息流公式

局部卷积：

~~~math
c_t=\sum_{j=0}^{w-1}W_jx_{t-j}
~~~

状态更新：

~~~math
s_t=F(s_{t-1},x_t)
~~~

全局 attention：

~~~math
a_t=\operatorname{Attn}(q_t,K_{\le t},V_{\le t})
~~~

融合可以是串联：

~~~math
h_t=\operatorname{Attn}(q_t,K(s,c),V(s,c))
~~~

也可以是并联加权：

~~~math
y_t=W_c c_t+W_s s_t+W_a a_t
~~~

不同融合方式改变梯度和 cache，不应只按模块集合比较。

## 44.3 小白直觉：三层阅读

卷积像读相邻几句话，发现词法和局部模式；SSM 像持续记住文章的总体状态和重要趋势；attention 像在需要时打开全文搜索某条证据。三者一起工作时，要避免每层都重复做全文搜索。

## 44.4 局部卷积的价值

卷积能把相邻 token 的混合交给规则 kernel，减少 attention 处理短程关系的压力。音频和视频中，局部连续性更明显；代码中，括号、缩进和局部语法也可能受益。

卷积窗口太小会漏掉局部结构，太大则增加成本；causal padding 和 position 要一致。卷积不是只适合非语言模态，关键在于任务局部性。

## 44.5 SSM 的价值

SSM 通过 state 维护多时间尺度历史，适合持续输入和长流。它不需要存所有 token，但状态摘要可能丢掉精确实体。将它放在 attention 之前，可以先做低成本信息过滤；放在 attention 之后，可以保存已对齐的全局表示；位置不同会产生不同能力。

## 44.6 Attention 的价值

attention 层可以提供显式 global path，让末端 query 直接访问历史 K/V。少量 attention 可能足以恢复关键 evidence，但仍要管理 KV cache。对文档、工具返回、跨模态对齐和多实体冲突，attention 的直接路径尤其有价值。

## 44.7 一个职责消融设计

~~~python
def ablation_plan():
    return {
        "all": ["conv", "state", "attention"],
        "no_conv": ["state", "attention"],
        "no_state": ["conv", "attention"],
        "no_attention": ["conv", "state"],
        "conv_only": ["conv"],
        "state_only": ["state"],
        "attention_only": ["attention"],
    }


for name, modules in ablation_plan().items():
    print(name, "+".join(modules))
~~~

对每种配置测局部模式、长流分类、精确复制、多证据和 serving。只在总 benchmark 上做消融会掩盖模块专长。

## 44.8 Cache 和状态协议

三种模块的状态不同：卷积需要最近窗口或 overlap buffer；SSM 需要递归 state；attention 需要 KV block。请求状态必须带 schema/version，否则模型升级时旧状态可能被误读。

取消/回滚时要同时处理：卷积缓存、SSM state、attention KV、position 和工具 observation。若其中一个路径仍更新，后续输出会出现难以复现的偏差。

## 44.9 系统资源画像

卷积往往计算规则、state 更新偏 memory-bound、attention prefill 偏 compute-heavy、decode 偏 KV-bandwidth-heavy。混合调度可以把不同阶段放在不同 worker，但跨池传输和状态同步有代价。

应按模块记录 FLOPs、HBM read/write、kernel time、workspace、state bytes 和通信。不能用单一层的理论复杂度估计整套模型。

## 44.10 常见失败模式

包括卷积和 SSM 处理相同信号导致冗余、attention 层位置不当、gate 让某一路径塌缩、三个模块的 norm/scale 不匹配、状态协议不完整和不同 kernel 的 dtype 差异。

排查先关闭模块、观察输出 norm/梯度；再做固定输入路径 trace；最后压测真实请求。对每个模块保存 reference logits 和 state checksum。

## 44.11 机制与边界：职责分工是 inductive bias 设计

串联三模块等于给模型一组时间和空间尺度先验。局部卷积提供短程平移共享，SSM 提供低维时间状态，attention 提供稀疏内容寻址。若任务查询分布与这些先验一致，数据效率和系统成本可能改善；若不一致，额外结构会限制表达。

一个值得研究的方向是按 token/层动态分配路径，但动态路由会破坏规则 kernel。固定层 schedule 适合部署，输入条件 gate 适合内容，二者需在可解释和性能之间平衡。

## 44.12 面试追问、误区与练习

**问：为什么三种模块不能简单说成“局部、长程、全局”？**

标准回答：这是有用的初始直觉，但实际感受野、状态压缩、gate 和 layer placement 会改变信息流；要通过模块消融、证据任务和 profile 验证。

**问：混合模型最难的系统问题是什么？**

标准回答：多类状态的生命周期、position/rollback 一致性、动态 batch、kernel 组合和故障恢复；不是只把三个 forward 拼起来。

常见误区包括忽略 cache schema、把 attention 层当无限 global memory，以及只看模块 FLOPs。

练习：为一个视频问答服务设计 conv/SSM/attention 分工，明确每类 token 的状态、传输和回退。

## 44.13 三种模块的接口契约

attention 通常输出 query 相关的动态聚合，卷积输出局部混合，SSM 输出递归状态变换。模块组合时要明确输入 shape、时间可见范围、状态是否持久、padding 语义和 position 字段。

如果模块接口只约定 hidden tensor 而没有声明 state 和 mask，线上 batch 重排、chunk continuation 和恢复很容易出错。

## 44.14 任务驱动的模块选择

局部代码语法、音频帧和连续日志适合卷积或 SSM；实体跨段比较、精确数字和工具参数需要 attention 或 retrieval；高并发系统可能采用三者混合，并让风险较高的 query 走显式路径。

## 44.15 模态和任务会改变模块分工

文本中的实体和代码依赖可能需要 attention；音频的局部频谱和视频的时间连续性可能更适合卷积或 SSM；跨模态问题则需要 projector 和 global alignment。模块分工不是固定的“attention 做全局、卷积做局部”口诀，而是由输入结构和查询类型决定。

## 44.16 组合模块的可测试接口

为每个模块设计输入、输出、mask、state、position 和 reset 的 golden test。再用同一输入比较单模块、两模块串联和完整模型，确认 shape、时间因果、梯度和状态恢复都没有隐式变化。

## 44.17 用信息访问原语而不是标签划分模块

“局部、长程、全局”是有用的入门分类，但更精确的分类是信息访问原语。卷积提供规则的邻域混合，SSM 提供持续的压缩更新，attention 提供基于当前 query 的内容寻址，retrieval 提供可回到原始证据的外部索引。它们的区别不只在感受野，也在是否能重新选择历史、是否保留来源和是否可以回滚。

对每条路径都可以问四个问题：历史以什么格式保存，当前 query 能否改变读取权重，读取结果能否定位回原文，失败后能否重新访问未压缩证据。这个框架比给模块贴“局部/全局”标签更能解释为什么混合架构仍需要 retrieval 或少量显式 attention。

## 44.18 混合架构的调度和成本归因

假设一个请求经过卷积路径、state 路径和 global attention，端到端延迟可以粗略拆成：

~~~math
T_{\mathrm{e2e}}
=T_{\mathrm{queue}}+T_{\mathrm{conv}}+T_{\mathrm{state}}
+T_{\mathrm{attn}}+T_{\mathrm{sync}}+T_{\mathrm{post}}
~~~

其中同步和状态传输经常被忽略。即使三个模块各自很快，跨 kernel 的 layout 转换、不同 batch 的 padding 和跨 GPU state 迁移也可能成为 p99 尾部。性能分析需要把时间归因到模块、状态读写、通信和 fallback，而不是只比较 FLOPs。

质量侧同样要做归因。若关闭 attention 后引用正确率下降，说明它承担证据对齐；若关闭 SSM 后长流恢复下降，说明 state path 有独立价值；若所有关闭实验都没有变化，可能是 gate 没有学出分工。

## 44.19 一个面向生产的最小状态协议

混合模型的每个请求状态可以抽象为：

~~~text
StateEnvelope {
  request_id
  model_revision
  position
  attention_kv
  recurrent_state
  convolution_overlap
  dtype_and_layout
  checksum
}
~~~

这不是某个框架的固定 API，而是一个审计清单。snapshot、迁移、恢复和 rollback 都必须明确哪些字段被保存、哪些字段可以重算、哪些字段不能跨 revision 复用。若只保存 hidden tensor，调度器无法判断它是否包含完整历史状态。

## 44.20 模块分工需要反事实测试

如果完整模型在一个任务上答对，仍不能知道答案来自哪条路径。可以分别遮挡局部卷积、递归 state、global attention 和 retrieval，再比较答案、引用和中间状态。对多模态输入，还要分别遮挡视觉、音频和文本证据，检查模型是否靠错误模态猜测。

反事实测试的关键是保持其他条件不变，并把“拒答”“引用缺失”“答案错误”和“延迟回退”分开记录。这样才能把模块贡献从总分中分离出来。

## 44.21 不同阶段的最优路径可能不同

长 prompt 的 prefill 更适合批量矩阵和并行卷积，逐 token decode 更容易受 KV 或 state bandwidth 影响，工具返回后的少量证据又可能值得打开 global attention。相同模型不应使用一个固定的性能结论覆盖所有阶段。

一个阶段化成本模型可以写成：

~~~math
C_{\mathrm{request}}
=C_{\mathrm{prefill}}(T_{\mathrm{in}})
+C_{\mathrm{decode}}(T_{\mathrm{out}})
+C_{\mathrm{state\ transfer}}
+C_{\mathrm{verification}}
~~~

容量规划和路由都要使用真实输入/输出长度分布。若只用平均长度，长任务和高风险回退的峰值会被掩盖。

## 44.22 维护混合模型的最小回归集

每次 kernel、gate、position、cache 或 scheduler 改动，都应运行四类回归：full 与 streaming 一致性，chunk 与 snapshot restore 一致性，多租户 state 隔离，以及精确证据任务的引用支持。性能回归还要检查 fallback、state bytes 和 p99。

这些测试把“混合架构能工作”具体化为一组长期契约。没有回归集时，新增一个高效路径可能悄悄破坏旧路径的状态语义。

## 44.23 从频率和查询角度理解模块互补

卷积擅长局部平移模式，SSM/递归状态擅长连续、多尺度和在线压缩，attention 擅长按 query 选择任意历史 token。真实任务往往同时包含三种信息：局部语法、长期趋势和少数精确证据。模块分工应由查询分布决定，而不是由“哪种模块最新”决定。

可以把任务拆成 local、compressible、addressable 三类需求，分别测局部预测、长流状态、任意位置召回和多证据合并。若某一模块承担了它不擅长的需求，混合结构应提供显式回退或外部检索。

## 44.24 组合模块的接口测试

接口测试包括 hidden shape、dtype、position、mask、residual scale、state reset、cache ownership 和 chunk continuation。对同一输入比较完整前向、分块前向、单步 decode 和恢复后 decode；对每个模块做 ablation，确认输出差异符合预期。

服务侧还要测不同模块的资源叠加：attention 的 KV、SSM 的 state、卷积的历史 buffer 和临时 workspace。总显存不是各个理论最小值的简单相加，因为 batch、对齐、通信和 fallback 会产生额外峰值。

## 44.25 按查询形态分配模块

可以把一个混合模型的输入拆成三类查询：邻域查询、压缩查询和精确寻址查询。邻域查询需要相邻 token 的稳定模式，卷积或局部 attention 更自然；压缩查询关注长流趋势和状态，SSM/retention 更合适；精确寻址查询要找一个特定数字、版本或工具结果，full attention 或 retrieval 更容易验证。

一个任务可能同时包含三类查询。例如语音助手用卷积处理局部声学帧，用 state 保持对话连续性，用 attention 对齐用户刚刚提到的实体。评测应分别改变邻域大小、流长度、实体数量和证据位置，观察哪条路径在何处失败。模块分工只有能解释失败，才不是一张漂亮的架构图。

成本账本还应包含状态交接、回读和验证。若每次精确查询都触发昂贵的 attention，混合路径可能不如直接使用成熟 dense baseline；若精确查询很少且压缩路径稳定，混合设计才可能形成 Pareto 优势。

## 44.26 一个模块分工的带数字例子

假设长日志任务中 80% 的 token 只需要局部模式，15% 需要压缩历史，5% 需要精确查找版本和错误码。让全部 token 经过 full attention 可能浪费资源；让全部 token 经过固定 state 又可能损失那 5% 的原子证据。一个候选混合路由可以先用卷积/SSM 处理主路径，再对高风险 query 打开 attention 或 retrieval，并用固定回归集检查是否漏掉关键证据。

路由收益不能只按 token 比例估算，还要加入状态交接、gate、fallback 和 p99。若 5% 的精确查询触发了频繁跨 GPU 迁移，混合方案可能比 dense baseline 更慢；若精确查询稀疏且状态路径可稳定复用，才可能形成端到端 Pareto 优势。

## 44.27 模块分工的工程验证

模块路线可参考 S4（https://arxiv.org/abs/2111.00396）、Mamba（https://arxiv.org/abs/2312.00752）、Hyena（https://arxiv.org/abs/2302.10866）和 Transformer（https://arxiv.org/abs/1706.03762）。不同模型的具体模块分工不能从论文家族名推断。

Attention、卷积和 SSM 的组合价值在于把不同信息流分工，工程难点在于让三种状态和计算路径真正协同。

## 44.28 三种模块的边界条件

卷积的优势依赖局部平移结构，SSM 的优势依赖可压缩和可递推的历史，attention 的优势依赖有足够预算保存和查询显式表示。任务中若不存在这些结构，模块替换可能只增加复杂度。比如随机指向任意版本号的查询难以由固定感受野或单一状态无损回答，通常需要显式寻址或外部检索。

因此，模块选择要先写 query distribution：邻域距离、历史长度、实体数、是否需要原文、是否有流式约束和是否允许回读。然后才能讨论 layer ratio、kernel 和容量。

## 44.29 模块互补的评估闭环

一条可复查的混合架构证据链是：局部 probe 证明卷积路径有效，长流 probe 证明 state 稳定，精确引用 probe 证明 attention/retrieval 可寻址，端到端压测证明交接成本可接受，故障演练证明 state/cache 可恢复。

如果某项失败，要保留失败样本和路径 trace。混合结构的价值不是每个模块都在总体分数上贡献正值，而是每个模块承担了可解释的失败类型，且新增交接成本没有超过质量和资源收益。
