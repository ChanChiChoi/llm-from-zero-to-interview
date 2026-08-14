# 第 63 章 Gated MLA：潜变量 cache 与显式注意力如何组合

## 63.1 MLA 先解决什么问题

多头 attention 的 K/V cache 维度随 head 数增长。长上下文推理时，历史 token 的 K/V 可能比模型权重更快成为显存瓶颈。MLA（Multi-head Latent Attention）的基本思路是把历史表示压缩到低维 latent，再在需要时投影或直接在 latent 空间参与 attention。

Gated MLA 进一步讨论路径选择：模型可以根据 token、channel、head 或 layer 决定更依赖 latent attention，还是保留显式/局部 attention。Kimi 等公开资料提供了这一方向的模型锚点，但具体 gate 位置和计算图必须以公开实现为准。

## 63.2 潜变量压缩的数学直觉

设输入或历史 hidden 为 `x_t`，先降维：

```math
c_t=W_{\mathrm{down}}x_t
```

再根据需要产生 key/value：

```math
\hat k_t=W_{\mathrm{up}}^kc_t,\qquad \hat v_t=W_{\mathrm{up}}^vc_t
```

如果 cache 保存 `c_t` 而不是完整的 `k_t,v_t`，单 token 的理想 payload 可以从 `2H_{kv}d_h` 降到 `d_c`。读取阶段需要 projection 或特殊 kernel，因此节省的是历史存储和带宽，不是无条件减少所有 FLOPs。

## 63.3 gate 为什么有用

latent 是压缩表示，不可能无损保留所有细节。某些任务需要精确数字、实体名或最近工具结果，显式路径更可靠；文档背景和重复日志则可以用 latent 表示。一个抽象融合为：

```math
y_t=g_t\odot y_t^{\mathrm{latent}}+(1-g_t)\odot y_t^{\mathrm{explicit}}
```

如果 `g_t` 总是接近 1 或 0，说明模型可能退化为单一路径；如果 gate 在每个 token 上剧烈切换，kernel 分支和延迟可能不稳定。gate 还不是证据选择器，不能直接解释成“它找到的重要 token”。

## 63.4 显式、latent 和局部 cache 的组合

一个长文档系统可以把旧正文保存成 latent，把最近窗口保存成显式 KV，把标注过的 global evidence 单独保留高精度。查询先经过 hybrid attention，必要时再回读原文。

假设文档主体占 95%，关键表格和最近工具结果占 5%。全显式方案精确但状态大；全 latent 方案省显存却可能损失表格数字；混合方案让 5% 的高价值内容保留精确表示。是否值得取决于证据召回和系统成本，而不是压缩比例本身。

## 63.5 资源估算

显式 cache 的理想 payload：

```math
M_{\mathrm{explicit}}\propto2BLTH_{kv}d_hb
```

latent cache 的理想 payload：

```math
M_{\mathrm{latent}}\propto BLTd_cb_c
```

理想压缩比：

```math
\rho=\frac{d_cb_c}{2H_{kv}d_hb}
```

真实值要加 projection buffer、scale、page metadata、对齐和临时恢复空间。若 latent kernel 带宽效率低，显存省下来的空间可能换来更高延迟。

## 63.6 长文档问答的 worked example

准备四类问题：主题摘要、精确数字、跨段实体比较和工具参数引用。分别测试只显式、只 latent 和 gated hybrid。若三者主题摘要相近，但 latent 在数字和引用上下降，说明压缩误差集中在高精度任务；可以增加 global evidence 或提高相应 token 的 cache 精度，而不是全局扩大 latent 维度。

## 63.7 serving 的状态生命周期

prefill 阶段生成 latent，decode 读取并做 projection；continuous batching 需要按请求维护 layout；prefix cache 共享时要绑定 model revision、projection revision 和 tokenizer/template；preemption 需要 snapshot；speculative 需要临时 latent 和 commit/rollback。

状态对象至少要记录 dtype、scale、latent dimension、position、page id 和 checksum。shape 相同不代表不同模型版本的 latent 可以互用。

## 63.8 信息瓶颈诊断

不能只用平均 perplexity 测 latent 质量。对文档中的数字、实体、表格列、代码标识符和工具参数做 sensitivity test，观察替换一个关键 token 是否影响答案和引用。还要检查压缩后是否仍能区分新旧版本和互相冲突的事实。

如果平均 loss 很好而 citation support 下降，说明 latent 表示保留了主题，却没有保留可归因的细节。可以按 token 价值动态选择显式/latent 路径，但要把策略写入评测协议。

## 63.9 量化和 speculative 的组合

latent cache 可以再量化，但量化误差和投影误差会叠加；要分开做 FP16 latent、低精度 latent、低精度 projection 的消融。speculative decode 中，candidate latent 只能在 target 接受后提交；拒绝分支释放临时 block，parser 和 stream 不得读到它。

## 63.10 常见失败

把 latent cache 当 prefix cache；只比较文件大小，不测读取 kernel；压缩丢失数字和来源；prefix hash 没包含 projection 版本；跨版本复用 latent；gate 量化后饱和；speculative reject 不回滚；把“有 MLA”写成“所有 KV 都消失”。

## 63.11 面试回答与练习

回答“latent cache 和 KV cache 有什么区别”时，应说 KV 保存可直接读取的历史 K/V，latent cache 保存压缩历史并在读取时投影或用专用 attention；前者精确、状态大，后者状态小、存在信息瓶颈和额外计算。要按数字、代码、引用和长文档任务测质量，并测真实 kernel 和并发。

练习一：给定 `H_kv=32,d_h=128,d_c=1024`，比较理想 payload。

练习二：设计一个能发现数字信息损失的测试集。

练习三：列出 latent cache snapshot/restore 必须保存的字段。

### 63.11.1 latent cache 的计算路径

显式 KV 通常保存 `K_t,V_t`。潜变量路径可以先把历史表示压到较小维度：

```math
z_t=C(h_t),\qquad
K_t=D_K(z_t),\qquad V_t=D_V(z_t)
```

读取时使用 `z_t` 或重新投影的 K/V。若 `d_c` 小于完整 K/V 维度，状态 payload 可能下降；但投影、量化、跨设备读取和恢复元数据会增加新的成本。

### 63.11.2 gate 的角色

门控 MLA 可以把显式路径和 latent 路径组合：

```math
o_t=g_t\odot o_t^{\mathrm{explicit}}
 +(1-g_t)\odot o_t^{\mathrm{latent}}
```

gate 可以按层、head、token 或 query 改变。它不是“模型解释了为什么选择证据”，只是参数化中的路径权重。要研究 gate 是否真的有用，需要遮挡显式或 latent 路径，观察任务结果和资源变化。

### 63.11.3 worked example：状态大小的理想比较

设完整 KV 每 token 的维度为 `2H_kv d_h=8192`，latent state 维度为 `d_c=2048`，元素都用 2 bytes。忽略 scale 和元数据，latent payload 是完整 KV 的四分之一。若长序列任务的 citation support 明显下降，这个压缩比就不能直接转成产品收益；可以只对低风险局部层使用 latent，对全局证据层保留显式路径。

### 63.11.4 cache manifest 和回滚

latent cache 需要保存压缩投影版本、gate 配置、dtype、位置状态、层 ID、request owner 和模型 revision。不能用“相同 token 前缀”直接共享不同 latent layout 的缓存。模型升级、batch 重排和 speculative rejection 都要验证 latent state 是否仍对应同一 token 前缀。

## 63.12 latent 维度如何影响任务

latent 维度 d_c 不是越大越好。维度增加会降低压缩误差，却提高 cache bytes、投影计算和跨设备带宽；维度减少会节省状态，却可能让数字、代码标识符和多实体关系发生碰撞。

应画出质量—资源曲线，而不是只比较一个压缩比。固定模型和任务，逐步改变 d_c，记录 citation support、exact match、长程 recall、cache GiB、projection time 和 p99。若质量在某个维度之后不再提升，才有理由把额外容量用于别的路径。

不同层的 latent 价值也可能不同。靠近输入的层处理局部词法，靠近输出的层负责任务组合；统一压缩比例可能浪费容量。按层消融能判断哪些层更需要显式 K/V 或更高 latent precision。

## 63.13 latent attention 的读放大

压缩 cache 省的是历史存储，但读取时可能发生读放大。若每个 query 要把 latent 重新投影为多个 head 的 key/value，投影 FLOPs 和临时 buffer 会随 head 数增长。可以把一轮 decode 的成本写成：

~~~math
C_{\mathrm{decode}}
=C_{\mathrm{latent\_read}}
+C_{\mathrm{projection}}
+C_{\mathrm{attention}}
+C_{\mathrm{output}}
~~~

当 batch 很小、projection kernel 不成熟时，显存下降可能换来更高 TPOT；当 batch 较大、latent 读取连续时，带宽收益才可能显现。压测要分别记录 projection kernel 和 attention kernel，而不是只看总 wall-clock。

## 63.14 latent cache 的安全和可追溯性

latent 表示通常不可直接供人审阅。高风险任务必须保留来源范围、原文索引或候选证据，不要把不可解释的 latent 当审计记录。缓存共享还要绑定租户、权限过滤和文档版本，否则一个用户的 latent 可能携带另一个用户不应看到的信息。

当模型用 latent 做初筛，再回读原文时，回读范围应由 provenance 驱动，而不是只依赖相似度。若压缩结果没有记录原始 token 范围，后续即使答案正确也无法可靠生成引用。

## 63.15 latent 表示与显式证据的分工

latent cache 适合保存供后续计算使用的紧凑表示，显式 K/V 或原文适合可寻址、可审计和精确回读。二者不是简单的高低精度版本：latent 可能已经改变了表示空间，无法直接恢复每个历史 token。

一个可靠架构可以先用 latent 做大范围扫描，再对高价值位置保留显式 K/V 或回读原文。这个双通道设计把成本和精确性分开，但需要保存候选来源、位置、revision 和触发原因。

## 63.16 gate 的功能验证

Gated MLA 中的 gate 可能控制压缩路径、显式路径、写入强度或输出融合。要证明 gate 有功能，至少要做四组反事实：保留证据并开 gate、删除证据并开 gate、保留证据并关 gate、删除证据并关 gate。

若删除证据后 gate 仍然高，gate 可能只响应长度或 token 类型；若开关 gate 对答案没有影响，可能存在冗余路径；只有证据存在且 gate 开启时质量改善，才支持它参与证据选择。还应记录 gate norm、分布、量化后变化和 p99。

## 63.17 latent cache 的版本和回滚

潜变量 cache 必须绑定压缩器版本、投影权重、模型 revision、dtype、position 和原始来源。更新压缩器后旧 latent 不能默认复用；推测解码拒绝候选时，投影中间量和 latent 写入也必须回滚。

服务层应为 latent 建立 checksum 和 provenance。没有来源范围的压缩表示无法生成可信引用；没有版本字段的 latent 可能在模型升级后产生静默错误。

## 63.18 从压缩公式走向服务契约

Gated MLA 关注的是“如何压缩历史表示，并让显式与潜变量路径按需组合”。一旦进入 serving，latent 不能只被当作一个更小的张量，而要成为有版本和所有权的状态对象。可以把一份可恢复状态写成：

```math
\Sigma_r=(z_r,\mathrm{source\_span},\mathrm{position},
\mathrm{projector\_rev},\mathrm{dtype},\mathrm{model\_rev},
\mathrm{checksum},\mathrm{fallback\_mode})
```

写入新 token 时，latent、显式 KV、gate 统计和 position 必须一起提交；speculative candidate 被拒绝时，它们也必须一起回滚。压缩器更新、量化切换、模型灰度或请求迁移时，先验证 revision、shape 和 checksum，失败就重算 prefix 或走显式路径，不能静默读取旧 latent。

上线前至少测四条路径：全显式 baseline、latent 正常路径、latent 误差超阈值回退、snapshot/restore 后继续生成。指标除了显存和 TPOT，还要看数字/版本 exact match、引用 provenance、回退率、恢复误差和租户隔离。具体 latent 定义、gate 位置和层表以官方技术资料为准；上述状态契约是部署上必须显式解决的通用问题。

## 63.19 为什么要把 KV 压到 latent

增量解码时，模型每生成一个 token，都要把 query 与历史 K 做匹配，再读取历史 V。标准多头注意力的缓存规模近似为：

~~~math
M_{\mathrm{KV}}
=L\times B\times H_{\mathrm{kv}}\times d_h
\times 2\times s,
~~~

其中 L 是历史长度，B 是 batch，H_kv 是 KV head 数，d_h 是 head dimension，乘 2 表示 key 和 value，s 是每个元素的字节数。即使使用 GQA/MQA 降低 H_kv，长上下文和高并发仍会让缓存占据显存与带宽。

MLA 的基本思路是先把历史 token 的 K/V 信息投影到较小的 latent：

~~~math
c_t = W_{\mathrm{down}} h_t,
\qquad
k_t = W_{\mathrm{up}}^k c_t,
\qquad
v_t = W_{\mathrm{up}}^v c_t.
~~~

服务端主要保存 c_t 或其进一步压缩的表示，读取时再恢复与当前 query 交互所需的 K/V。DeepSeek-V2 论文是公开讨论 MLA 及其 KV cache 优势的重要来源；具体模型如何组合 gate、position 和 latent，需要回到相应模型卡或技术报告。

这里有一个关键认识：latent cache 不是“免费地保存完整 K/V”。它把缓存空间从显式 token 表示换成低维坐标，代价是投影误差、重构限制和额外计算。若某个任务依赖一个低频但关键的字段，平均重构误差很小也可能掩盖该字段已经丢失。

## 63.20 Gated MLA 的门控位置

门控可以出现在多个位置，含义并不相同。一个教学抽象是把 latent 路径和显式路径融合：

~~~math
y_t
=g_t\odot y_t^{\mathrm{latent}}
+(1-g_t)\odot y_t^{\mathrm{explicit}}.
~~~

也可以在 query、latent K、latent V 或 residual 输出上加 gate。阅读模型资料时应逐项确认：

1. gate 是按层、head、channel 还是 token 生成的。
2. gate 控制的是计算路径、表示尺度还是残差融合。
3. gate 是否依赖当前 query，还是只依赖历史 token。
4. gate 的训练目标是否有稀疏、稳定或负载约束。

小白可以把 latent 路径看成压缩目录，把显式路径看成原始档案。大多数问题先查目录，遇到关键字段再访问档案。深入分析时要注意，gate 高并不等于模型找到了正确证据；它可能只反映两条路径的数值尺度不同。必须用证据遮挡、路径消融和输出变化判断 gate 是否真正有功能作用。

## 63.21 显式 KV、latent cache 和 state cache 的区别

三种历史表示经常被混在一起：

| 表示 | 保存什么 | 强项 | 典型风险 |
| --- | --- | --- | --- |
| 显式 KV | 每个 token 的 key/value | 精确寻址、实现直观 | 长度线性增长 |
| latent cache | 每个 token 的低维压缩表示 | 降低缓存大小 | 信息重构和投影开销 |
| recursive state | 聚合后的固定/受控状态 | 低内存、流式 | 失去 token 地址和细节 |

MLA 通常仍保留 token 级 latent，只是每个 token 的表示更小；递归 state 则进一步把多个 token 汇总到固定状态。前者更容易沿 token 位置回读，后者更依赖训练出来的状态更新。一个系统可以同时使用三者，但容量公式和恢复协议必须分开记录。

## 63.22 一个缓存账本例子

设 L=128000、B=8、H_kv=8、d_h=128、BF16 每元素 2 字节。显式 KV 的原始元素量为：

~~~math
128000\times8\times8\times128\times2
\times2
=4,194,304,000\ \text{bytes},
~~~

约为 3.91 GiB，尚未计入 page metadata、对齐和临时 workspace。若 latent dimension 从 H_kv x d_h = 1024 压到 256，理想化的 token 表示部分约降到四分之一，但还要加上恢复投影、scale、位置分量和 kernel buffer。实际收益应以峰值显存和端到端 p99 测量，而不是只看 latent 维度比例。

如果 gate 让只有 20% 的请求走显式路径，平均成本不等于显式路径成本乘 0.2。边界请求可能触发回读、重算或 verifier，显式路径还会造成 batch 分裂。容量规划需要模拟请求混合，而不是使用单一平均比例。

## 63.23 位置编码与 latent 的耦合

压缩 K/V 时不能忘记位置。若位置旋转作用在 K 上，服务端需要明确：

1. position 是在压缩前进入 K，还是在恢复后进入 K。
2. latent 是否包含与位置相关的分量。
3. prefix cache 命中时起始 position 是否一致。
4. sliding window、context parallel 和 preemption 是否能恢复同一位置坐标。

一个常见错误是把相同文本的 latent 在不同起始位置直接复用。文本内容相同不代表经过位置变换后的 K 相同。若模型把位置和内容分成不同路径，缓存 key 还要绑定位置处理版本。

## 63.24 量化与投影误差的组合

latent cache 的量化通常比显式 K/V 更难直觉判断。压缩已经改变表示分布，再量化可能放大少数通道的误差。一个简单的总误差账本可以写成：

~~~math
E_{\mathrm{total}}
\approx E_{\mathrm{projection}}
+E_{\mathrm{quant}}
+E_{\mathrm{position}}
+E_{\mathrm{kernel}}
+E_{\mathrm{protocol}}.
~~~

这不是可直接相加的严格误差定理，而是排查顺序。若 FP16 latent 已经丢失关键证据，换成更高精度量化不会解决投影瓶颈；若未量化路径正确、量化后只在长上下文失败，应检查 scale 粒度、异常值、累积精度和 dequant 位置。

数字、代码、版本号和权限条件应单独建立回归集。embedding cosine 相似度高，只能说明平均方向相近，不能证明每个数字和否定条件都保留。

## 63.25 显式路径的 fallback 设计

门控 latent 路径必须有明确的 fallback，否则它只是一个不可解释的压缩黑箱。适合触发显式回读的信号包括：

1. gate 处在不确定区间。
2. query 包含数字、版本、权限或精确引用要求。
3. latent 与显式候选的 logits 分歧超过阈值。
4. verifier 发现答案 claim 没有充分证据。
5. cache revision、position 或 dtype 不匹配。

fallback 可以是读取更大的 latent、打开 local/global attention、从外部索引回读原文，或直接拒答/转人工。不同业务风险要有不同策略：摘要可以接受更高压缩比，支付、法律和安全审计不能把“看起来合理”当作通过。

## 63.26 从 query 到 latent 的读取路径

一个可读的 serving 路径如下：

~~~text
query
  -> query projection
  -> latent candidate scoring
  -> optional explicit KV verification
  -> gated fusion
  -> output projection
~~~

每一步都可能成为瓶颈。candidate scoring 如果需要扫描所有 latent，压缩缓存可能只节省显存、不节省带宽；explicit verification 如果频繁触发，平均延迟会被尾部请求主导；gated fusion 如果引入不规则分支，kernel 利用率可能下降。

因此 benchmark 至少要区分 latent-only、latent-plus-verification 和显式 KV 三条路径，并分别测短请求、长请求、不同 batch 和不同 gate 分布。

## 63.27 训练目标不能只看语言模型 loss

若训练只优化 next-token loss，模型可能学会用 latent 保持一般语义，却没有学会保存可回读的证据。可以按任务补充辅助约束或评估：

~~~math
L
=L_{\mathrm{LM}}
+\lambda_1 L_{\mathrm{retrieval}}
+\lambda_2 L_{\mathrm{citation}}
+\lambda_3 L_{\mathrm{conflict}}
+\lambda_4 L_{\mathrm{gate}}.
~~~

这些项不一定都加入真实训练；它们首先是评估和分析维度。L_retrieval 检查证据是否可找回，L_citation 检查引用是否支持 claim，L_conflict 检查旧新版本能否区分，L_gate 可以约束门控不要出现全开或全关的退化。

专家还要观察潜变量利用率。如果所有 token 的 latent 都趋近相同，压缩比看起来很高但信息容量已经塌缩；如果 gate 长期只使用显式路径，latent 模块没有产生实际收益。

## 63.28 最小伪代码和 shape 检查

~~~python
latent = down_projection(hidden)       # [batch, length, latent_dim]
cached = quantize(latent)              # [batch, length, latent_dim]
query = query_projection(current)      # [batch, 1, query_dim]
latent_k = up_k(cached)                # [batch, length, key_dim]
latent_v = up_v(cached)                # [batch, length, value_dim]
score = query @ latent_k.transpose(-1, -2)
latent_out = softmax(score, dim=-1) @ latent_v
explicit_out = read_explicit_kv(query)
gate = sigmoid(gate_projection(current))
output = gate * latent_out + (1 - gate) * explicit_out
~~~

这段代码只展示数据流，不能直接代表高性能实现。实际 serving 需要 page layout、异步预取、量化 scale、position metadata 和 batch 调度。教学 demo 的价值在于让读者先核对 shape，再理解为什么压缩并不等于删除所有投影计算。

## 63.29 评测 latent cache 的四个维度

评测要同时回答四个问题：

1. 节省了多少峰值显存？
2. 是否降低了 prefill/decode 延迟和尾延迟？
3. 哪类证据在压缩后消失？
4. fallback、重试和 verifier 是否把成本加回来？

推荐报告：

~~~text
peak memory
cache bytes per token
TTFT
TPOT
P50/P95/P99
evidence recall
exact numeric accuracy
citation support
gate activation rate
fallback rate
unit successful-task cost
~~~

如果 cache bytes 降低 4 倍、p99 只改善 5%，可能是 attention kernel、网络或调度主导；如果质量下降只发生在中间位置，应检查位置分量和长上下文训练；如果 fallback rate 随长度急升，说明 latent 路径的有效能力边界低于 API context window。

## 63.30 latent 压缩的误差从哪里来

把历史映射到 latent `c_t` 能减少 cache 存储，但查询时需要重新投影或解码。误差至少有三类：压缩维度不足造成的信息丢失，低精度量化造成的数值误差，以及 gate/位置变换不一致造成的系统误差。不能只用 latent 的字节数宣称“省了显存”，还要测读取时的投影计算、带宽和质量变化。

设每个 token 的显式 KV 为 `b_{kv}` 字节，latent 为 `b_c` 字节，长度为 `T`，则粗略存储差额为：

```math
\Delta M=T(b_{kv}-b_c)-M_{\mathrm{metadata}}-M_{\mathrm{projection}}.
```

当 batch 很小或序列很短时，投影和 metadata 可能吃掉收益；当长上下文和高并发时，latent 的存储优势才更明显。

## 63.31 latent cache 的读放大与调度

显式 KV 可以直接按 page 读取，latent cache 可能需要把一个 latent 解码成 query 所需的 K/V。多个请求共享前缀时，解码 kernel 可以合并；请求长度和访问位置分散时，读放大可能抵消压缩收益。调度器因此需要知道 cache 的表示类型，而不能把所有 cache 当成同一种 block。

压测至少分短请求、高并发长请求、prefix sharing 和随机恢复四类。记录 cache bytes、latent decode time、memory bandwidth、TTFT、TPOT 和 p99。平均 tokens/s 上升但 p99 恶化，仍可能不满足线上 SLO。

## 63.32 gate、位置和量化的联合消融

Gated MLA 的 gate 可能位于 latent 写入、读取或输出融合路径，不同位置会改变误差传播。评测应固定其中两个，只替换第三个：无 gate/有 gate、FP16/低精度、短位置/长位置。这样才能知道收益来自压缩、门控还是位置训练，而不是把多个变更归到同一个模型名称下。

## 63.33 latent cache 的回退条件

当 latent projection kernel 不可用、量化误差超过阈值、cache revision 不匹配或特定任务需要精确引用时，可以回退到显式 KV 或 RAG。回退必须在请求开始前或安全的事务边界发生，并清理临时 latent page。不能在已经混用两种表示后静默切换，否则 state 和位置可能不一致。

## 63.34 小结

Gated MLA 的价值可以概括为：把历史 K/V 的存储与读取从“每个 token 保存完整表示”改成“保存 latent，并在需要时恢复或融合显式路径”。它同时改变了模型结构、缓存账本、位置协议、量化路径和评测方法。

理解时要把三件事分开：MLA 的 latent 压缩、gate 的路径选择、serving 的 cache 生命周期。公开资料没有给出的 gate 位置、latent layout 和层表，不能由名称推断。只有在质量验收条件、显存、p99、fallback 和版本恢复都通过后，latent cache 的理论优势才算转化为工程收益。
