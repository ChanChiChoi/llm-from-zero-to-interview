# 第十章：Checkpoint、评估与实验管理：让训练结论可恢复、可比较

一次大模型训练结束时，真正留下的不是一条下降的 loss 曲线，而是一组可以被别人重新解释的证据：模型参数来自哪个数据版本，训练停在什么 token 位置，验证集和下游任务表现如何，是否有能力回归，最终选择的 checkpoint 为什么不是另一个版本。

初学者可以把 checkpoint 理解成训练过程的“存档”。只保存模型权重，能够让模型再次做推理；要从中断处继续训练，还要保存优化器、学习率进度、随机状态和数据位置。专家则会继续追问：这个存档是否原子写完？分片布局是否匹配？恢复后的一步更新是否与未中断轨迹连续？

评估也不是训练结束后的装饰。validation loss 只反映某种 token 预测目标，下游 benchmark 只覆盖有限能力，安全和回归评估还可能揭示平均分掩盖的退化。实验管理把代码、数据、配置、环境、artifact 和评估结果绑在一起，才能知道一次提升究竟来自什么。

本章沿着一条具体链路展开：先定义训练状态，再讨论保存频率和成本；接着验证恢复连续性，拆开 validation、per-domain、benchmark、regression 和 safety 证据；最后说明如何管理实验版本、公平比较和选择 checkpoint。面试场景放在章节后部，用来检验前面的工程理解，而不是替代前面的解释。

## 为什么 checkpoint 和实验管理很重要

大模型训练成本很高，一次训练可能持续数天、数周甚至更久。

如果没有可靠 checkpoint，任何机器故障、网络故障、代码 bug 或 NaN 都可能让训练前功尽弃。

如果没有实验管理，即使跑出一个好模型，也很难回答：

1. 这个模型是用哪批数据训练的。
2. 训练了多少 token。
3. 用了什么学习率和 batch size。
4. 哪个 checkpoint 最好。
5. 相比上一个版本到底提升在哪里。
6. 是否有能力退化或安全风险。

训练工程不只是把 loss 跑下来，还要保证过程可恢复、结果可比较、结论可复现。下面的内容会同时照顾两种读者：初学者先建立字段和指标的直觉，专家再处理分片、统计分母、回归风险和多目标选择的边界。

### Checkpoint 与实验管理的资料边界

本章参考了 PyTorch 保存/加载通用 checkpoint、PyTorch Distributed Checkpoint、Hugging Face Transformers `Trainer` checkpoint 与 best model 配置、perplexity 评估口径，以及实验追踪和 artifact versioning 的官方实践。

本章聚焦训练工程中最小可落地的闭环：checkpoint 保存内容、恢复连续性、存储保留策略、validation / benchmark / regression 评估、实验记录、版本管理和可复现性。它不展开生产级对象存储、跨机房灾备、完整 MLOps 平台、权限审计系统和所有评测框架 API。

### 实验对象与证据链

一次训练实验可以抽象成：

```math
e=
(c,d,m,h,a,r)
```

其中 `c` 是代码版本，`d` 是数据版本，`m` 是模型和 tokenizer 配置，`h` 是训练超参和硬件环境，`a` 是 checkpoint artifact，`r` 是评估结果。实验结论是否值得采信，取决于这些对象之间能否追溯和对齐，而不是取决于一个单独的总分。

```math
E_evidence=(E_ckpt,E_eval,E_repro,E_compare,E_safety)
```

这五个分量分别描述 checkpoint 完整性、评估覆盖、记录可复现性、对比公平性以及安全/回归风险。某一项不足时，结论的适用范围应随之收窄；例如 benchmark 分数可以报告，但不能把它解释成完整能力提升。

## 1. Checkpoint 应该保存什么

一个完整训练 checkpoint 不只是模型权重。

通常至少包含：

| 内容 | 作用 |
| --- | --- |
| model weights | 模型参数 |
| optimizer state | AdamW 的 m/v 等状态 |
| scheduler state | learning rate 进度 |
| scaler state | FP16 mixed precision 的 loss scale |
| RNG state | Python、NumPy、PyTorch、CUDA 随机状态 |
| global step | 当前训练步数 |
| consumed tokens | 已训练 token 数 |
| data state | dataloader 或 shard 位置 |
| model config | 架构配置 |
| tokenizer | tokenizer 文件和 special token |
| training config | batch、lr、并行、精度等配置 |

并不是每个字段在每种精度和训练框架下都同样存在。例如 BF16 训练通常不需要 FP16 `GradScaler`，某些优化器不会保存独立的 master weights，某些数据管道也会把 sampler 状态放在单独的文件中。表格表达的是恢复时要回答的问题，而不是一个所有框架都必须采用的字典键名。

只保存 model weights 可以用于推理，但通常不能完整恢复训练。即使张量形状能够匹配，也要核对 model config、词表和 special token；错误 tokenizer 可能让模型“成功加载”却在输入切分、EOS 处理或输出解码上产生完全不同的行为。

如果 optimizer 或 scheduler 没恢复，继续训练时 loss 可能跳变。

随机状态也不是装饰字段。数据 shuffle、dropout、采样和某些 CUDA kernel 都可能读取随机数；恢复 Python、NumPy、CPU、CUDA 以及分布式 sampler 的状态，才能缩小恢复前后的随机差异。它仍然不保证跨硬件、跨 kernel 或跨版本 bit-level 一致，这个边界应当在实验记录中说清楚。

恢复训练所需字段可以写成一个集合：

```math
C_{\mathrm{resume}}=
\{model,opt,sched,scaler,rng,data,step,tok,config,tokenizer\}
```

恢复时要把缺失或版本不匹配的字段显式列出来。设 `available(u)` 表示当前 checkpoint 的字段 `u` 存在、版本匹配且能被训练脚本读取，则缺失集合为：

```math
M_{\mathrm{missing}}=
\{u\in C_{\mathrm{resume}}\mid
\neg\operatorname{available}(u)\}
```

`M_missing` 为空时，只能说明恢复字段在文件层面齐全；仍需验证分片布局、数值有限性、数据 cursor 和恢复后的一步轨迹。若集合非空，就应明确这是“部分状态 checkpoint”，不能把它直接描述成可无缝恢复训练的版本。

一个实用的 manifest 还应记录每个文件的大小、哈希、dtype、张量形状、写入版本和完成状态。这样加载失败时，工程师可以区分“代码不认识这个字段”“文件没有写完”和“文件内容已被破坏”，而不是把所有错误都归因于模型本身。

## 2. 保存频率怎么定

Checkpoint 保存太少，故障时损失大；保存太频繁，会浪费存储和训练时间。

需要权衡：

1. 训练成本。
2. 故障概率。
3. checkpoint 大小。
4. 存储带宽。
5. 恢复时间。
6. 评估频率。

常见策略：

```text
每 N steps 保存一次 latest checkpoint
每 M tokens 保存一个里程碑 checkpoint
保留最近 K 个 checkpoint
对关键 checkpoint 做长期归档
```

例如：

```text
latest: 用于故障恢复，只保留最近几个
milestone: 用于评估和对比，按 token 数保存
best: 根据评估指标选择，用于后训练或发布
```

如果每 `f` 个 step 保存一次，总训练步数为 `S`，保存次数约为：

```math
N_{\mathrm{save}}=
\left\lceil \frac{S}{f}\right\rceil
```

一次故障最多会损失最近一次保存后的 token：

```math
T_{\mathrm{lost,max}}=
fB_{\mathrm{tok}}
```

其中 `B_tok` 是每 step 消耗的 token 数。保存开销占比可以粗略估成：

```math
R_{\mathrm{save}}=
\frac{C_{\mathrm{save}}}{fT_{\mathrm{step}}}
```

这里把每 step 的 token 数近似成常数；变长 batch 或动态 packing 时，更准确的做法是对故障窗口内实际消费的 token 求和。`f` 越小，容错越好，但 `R_save` 和存储压力越高。

这里 `C_save` 是一次保存占用的墙钟时间，`T_step` 是一个训练 step 的平均时间；若保存与计算完全串行，`R_save` 近似表示保存带来的时间占比。异步写盘可以降低前台阻塞，但会增加显存/主机内存中的待写队列和“最新 checkpoint 尚未真正落盘”的窗口，因此还要记录发布状态，而不能只看保存函数返回。

`latest`、`milestone` 和 `best` 是三种不同语义：latest 追求故障恢复距离短，milestone 追求按 consumed tokens 做公平纵向比较，best 追求某组评估目标下的候选选择。一个 checkpoint 可以同时属于多个集合，但保留策略不能把它们当成同一个对象。

保存过程还要处理“半成品可见”的问题。常见做法是先写入临时目录，完成所有 shard、索引和 manifest 后，再通过原子重命名或完成标记发布版本；清理任务只删除明确标记为完整且不再被引用的旧版本。这里的原子性不是说多节点写入真的在同一时刻发生，而是说读取端不会把一个尚未完成的集合误认成可恢复版本。

因此，checkpoint 策略不能只回答“每隔多少 step 保存一次”。更完整的设计会把 latest、milestone、best 分开，并同时说明故障恢复距离、评估纵向比较的锚点和总存储预算。只有这样，保存频率才是一个有上下文的工程决策。

## 3. Checkpoint 存储成本

大模型 checkpoint 非常大。

如果是 70B 模型，单模型权重就可能上百 GB。若再保存 optimizer state，体积可能数倍增加。

因此要区分两类 checkpoint：

1. 训练恢复 checkpoint：包含 optimizer、scheduler、scaler 等完整状态。
2. 推理/评估 checkpoint：只包含模型权重、config 和 tokenizer。

训练恢复 checkpoint 更大，但能继续训练。

推理 checkpoint 更小，适合评估、部署和归档。

如果使用 FSDP 或 ZeRO，checkpoint 还可能是 sharded 格式，需要专门的保存和加载逻辑。

“sharded”描述的是存储布局，不等于“不能换并行规模”。有的格式保存了全局参数名和分片元数据，允许加载端重新切分；有的格式与当前 world size 或参数分片方式绑定，换机器数后需要转换。评估一个格式时，要把保存端和加载端的 world size、参数命名、dtype、词表大小以及 optimizer state 的布局一起记录。

设模型参数量为 `P`，权重每个参数占 `b_w` 字节，则推理权重大小近似为：

```math
M_{\mathrm{eval}}=
Pb_w+M_{\mathrm{meta}}
```

如果训练恢复还保存梯度、AdamW 一阶/二阶状态和 master weights，恢复 checkpoint 可以粗略估成：

```math
M_{\mathrm{resume}}
\approx
P(b_w+b_g+b_m+b_v+b_{\mathrm{master}})
+M_{\mathrm{meta}}
```

保留策略的总存储可以写成：

```math
M_{\mathrm{retain}}=
K_{\mathrm{latest}}M_{\mathrm{resume}}
+K_{\mathrm{best}}M_{\mathrm{eval}}
+K_{\mathrm{mile}}M_{\mathrm{eval}}
```

这里 `latest` 偏容错，`best` 偏下游使用，`mile` 偏实验复盘。真实项目还要考虑 sharded checkpoint 合并、对象存储带宽和跨节点读取失败。

上式只是容量预算，不应被误读为精确的文件大小。压缩、稀疏格式、共享参数、元数据、索引、对齐填充和分片副本都会改变实际占用；优化器状态的 dtype 也可能与模型权重不同。工程上最好分别测量“写入前的逻辑大小”“对象存储实际大小”和“加载时的临时峰值”，因为 OOM 可能发生在恢复合并阶段，而不是发生在最终 artifact 的静态大小上。

## 4. 训练恢复和容错

恢复训练时，首先要区分“文件能被加载”和“训练状态真的接上了”。前者只说明序列化格式、路径和当前代码能够读出对象；后者还要求优化器、学习率调度、数据游标、随机状态以及分布式分片共同落在正确的位置。

最直观的连续性检查包括三件事：

1. 能否成功加载。
2. 恢复后 loss 是否连续。
3. 恢复后学习率和 step 是否正确。

一个健康恢复应该表现为：

```text
恢复前后 loss 曲线连续
learning rate 不从头开始
optimizer 动量状态存在
global step 和 token count 正确
```

常见恢复错误：

1. 只加载 model weights。
2. scheduler 从 0 开始。
3. optimizer state 缺失。
4. tokenizer 版本不一致。
5. 分布式 shard 数变化导致加载失败。
6. 数据迭代位置变化，重复或跳过大量数据。

恢复后要做连续性检查。设恢复前最后一个健康日志为 `-`，恢复后第一个日志为 `+`：

```math
\delta_L=
\frac{|L^+-L^-|}{L^-+\epsilon}
```

```math
\delta_{\eta}=
\frac{|\eta^+-\eta^-|}{\eta^-+\epsilon}
```

```math
\delta_{\mathrm{step}}=
\left|(s^+-s^-)-1\right|
```

如果探针只跨越一个 optimizer step，还应检查 consumed tokens 的增量。设 `B_{\mathrm{tok}}^+` 是恢复后这一步按实际 batch 计算的预期 token 数，则：

```math
\delta_{\mathrm{tok}}=
\frac{\left|(T^+-T^-)-B_{\mathrm{tok}}^+\right|}
{B_{\mathrm{tok}}^++\epsilon}
```

因此，恢复探针更适合记录为偏差向量：

```math
R_{\mathrm{resume}}=
(\delta_L,\delta_{\eta},\delta_{\mathrm{step}},\delta_{\mathrm{tok}})
```

这个向量比一个真假值更有信息。`\delta_L` 较大，可能指向数据 batch、loss mask 或 optimizer state；`\delta_{\eta}` 异常，优先检查 scheduler 的 step 口径；`\delta_{\mathrm{step}}` 或 `\delta_{\mathrm{tok}}` 异常，则要检查 global step、gradient accumulation 和 data cursor。阈值 `\tau_L`、`\tau_{\eta}` 只能作为当前实验的比较尺度，不能被当成所有模型都适用的固定常数。

还要注意，token count 连续并不能证明样本顺序连续。一个数据管道可能正好消费了相同数量的 token，却从另一个 shard 开始；因此 checkpoint 中的 data state 至少应包含数据版本、epoch 或 shard 顺序、当前 shard、样本/packed sequence 游标，以及必要时的 sampler 状态。恢复探针应把这些字段和日志里的 token 数放在同一张记录中。

分布式训练还多一层风险：某个 rank 的 shard 写完，不代表整个 checkpoint 已经可读。发布一个 checkpoint 前，应确认各 rank 的文件、索引和元数据都已落盘，并让读取端能够发现“未完成写入”的状态，而不是把半套 shard 当成最新版本。对象存储上的临时前缀、完成标记或 manifest 都是实现手段；关键是读取者能区分正在写入和完整发布的 artifact。

因此，面试中回答 checkpoint 恢复时，不能只说“调用 `load_state_dict`”。完整回答应说明：先验证 artifact 和版本，再恢复所有训练状态，最后用 loss、学习率、step、token 增量和数据游标做一个短探针；如果某一项偏离，就根据偏差类型定位，而不是盲目继续训练。

## 5. 为什么不能只看最后一个 checkpoint

很多初学者默认最后一步最好，但真实训练不一定。

可能出现：

1. 后期过拟合某些数据。
2. 某些能力退化。
3. 安全拒答变差。
4. checkpoint 附近发生 loss spike。
5. 后训练中风格过拟合。

所以 checkpoint 选择是多目标决策，不是只看 step 最大。

更准确地说，checkpoint 选择是在“训练进度”和“模型表现”之间找一个合适的候选。两个 checkpoint 如果训练 token 数不同，就不能把较晚版本的优势简单归因于某个超参；如果 token 数相同而一个版本在代码 domain 退化，也不能因为总分略高就忽略它。选择前先对齐训练量和评估口径，往往比设计一个更复杂的总分更重要。

需要比较：

1. validation loss。
2. per-domain loss。
3. 下游 benchmark。
4. 人工评测。
5. safety eval。
6. regression eval。
7. 推理延迟和成本。

## 6. Validation loss 和 perplexity

预训练阶段最基础的指标是 validation loss。

Perplexity 可以理解为 cross entropy loss 的指数形式：

```text
perplexity = exp(loss)
```

对 causal LM，带 mask 的 validation loss 可以写成：

```math
L_{\mathrm{val}}=
-
\frac{
\sum_i\sum_t m_{i,t}\log p_{\theta}(x_{i,t}\mid x_{i,<t})
}{
\sum_i\sum_t m_{i,t}
}
```

对应 perplexity 为：

```math
PPL=
\exp(L_{\mathrm{val}})
```

loss 越低，perplexity 越低，说明模型对验证集 token 的预测越好。这里的“更好”只针对给定 tokenizer、给定数据分布和给定预测目标；它不是对模型全部能力的排序。

还要留意统计分母。若一个 batch 中有 padding、prompt-only token 或被排除的标签，不能把总序列长度当作有效 token 数。对比不同 checkpoint 时，最好同时保存有效 token 数、每个 domain 的 token 数和聚合方式；否则同一个 loss 数字可能来自不同的样本覆盖。

但要注意：

1. validation set 必须干净且不泄漏。
2. 不同 tokenizer 的 loss/perplexity 不宜直接比较。
3. 总 loss 可能掩盖某些 domain 退化。
4. loss 下降不代表所有能力提升。

Perplexity 还受 tokenizer 的切分粒度影响。同一段文本被切成更多 token 时，token-level loss 的统计对象已经变了；因此跨 tokenizer 比较时，不能把 PPL 当成语言能力的绝对尺度。若必须比较，应同时报告 tokenizer、有效 token 数、评估文本和聚合方法，或者改用与 tokenizer 无关的下游任务指标。

因此，validation loss 更像训练目标分布上的温度计：它能告诉我们模型在这类 token 上的预测误差如何变化，却不能单独告诉我们代码、数学、工具调用或安全行为是否变好。

## 7. Per-domain loss

只看整体 validation loss 不够。

应该按 domain 分开看：

| Domain | 可能观察 |
| --- | --- |
| general web | 通用语言能力 |
| code | 代码建模能力 |
| math | 数学符号和推理模式 |
| zh | 中文能力 |
| en | 英文能力 |
| academic | 论文和专业文本 |
| dialogue | 对话格式能力 |

例如整体 loss 下降，但代码 loss 上升，说明模型可能在代码能力上退化。

这种情况在数据配比调整后很常见。

domain 划分也要保持稳定。若一次实验把“代码注释”放进 general，另一次把它放进 code，曲线变化可能只是分类规则变化。比较时应保存 domain 定义、样本清单或稳定的哈希抽样规则，并记录每个桶的有效 token 数。

设验证集按 domain 分成 `D_k`，每个 domain 的 loss 为：

```math
L_k=
-
\frac{
\sum_{(i,t)\in D_k}m_{i,t}\log p_{\theta}(x_{i,t}\mid x_{i,<t})
}{
\sum_{(i,t)\in D_k}m_{i,t}
}
```

整体 loss 可以看成加权平均：

```math
L_{\mathrm{all}}=
\sum_k w_kL_k
```

其中 `w_k` 来自各 domain 的有效 token 占比。checkpoint 对比时，不仅要看 `L_all` 是否下降，也要看是否存在：

```math
L_k^{new}-L_k^{old}>\tau_k
```

这类 domain regression。

per-domain loss 的价值在于把“平均变好”拆成多个可解释的变化。它还会提醒我们检查 domain 的采样比例和有效 token 数：如果某个 domain 很小，均值波动可能很大；如果 domain 定义或 tokenizer 发生变化，跨版本的数值也不能直接排成一条曲线。

## 8. 下游 benchmark

Base model 训练阶段也需要下游 benchmark。

常见评估维度：

1. 知识问答。
2. 数学推理。
3. 代码生成。
4. 阅读理解。
5. 多语言能力。
6. 长上下文能力。
7. 安全和毒性。

但 benchmark 有风险：

1. 可能被污染。
2. 可能和真实目标不一致。
3. 可能只反映某一类能力。
4. 分数提升可能没有统计显著性。

对于准确率类指标，样本数为 `n`、观测准确率为 `p` 时，独立同分布的粗略标准误可以写成：

```math
\operatorname{SE}(p)\approx
\sqrt{\frac{p(1-p)}{n}}
```

它不是所有评测的完整置信区间，尤其不适用于有相关样本、生成式评分或复杂 judge 的场景，但足以提醒我们：几十道题上的几个百分点变化，未必比评测噪声大。更可靠的比较通常保留逐样本结果，使用配对样本分析，并固定 prompt、解码参数、评判模型和随机种子。

所以 benchmark 要和 validation loss、人工评测、回归测试一起看。

## 9. Regression eval

模型新版本不能只看新能力是否提升，还要看旧能力是否退化。

Regression eval 的目标是发现：

```text
旧模型答对，新模型答错
```

这类样本可以叫 lost cases 或回归样本。

设旧模型正确标记为 `z_i^{old}`，新模型正确标记为 `z_i^{new}`，回归率可以写成：

```math
R_{\mathrm{reg}}=
\frac{
\sum_i \mathbf{1}[z_i^{old}=1]\mathbf{1}[z_i^{new}=0]
}{
\sum_i \mathbf{1}[z_i^{old}=1]+\epsilon
}
```

如果样本带业务权重 `w_i`，高风险回归率可以写成：

```math
R_{\mathrm{reg,w}}=
\frac{
\sum_i w_i\mathbf{1}[z_i^{old}=1]\mathbf{1}[z_i^{new}=0]
}{
\sum_i w_i\mathbf{1}[z_i^{old}=1]+\epsilon
}
```

常见回归维度：

1. 格式遵循。
2. 数学和代码。
3. 安全拒答。
4. 多语言。
5. 企业业务规则。
6. 长上下文引用。

回归率的分母也决定了它回答什么问题。上式回答的是“旧模型答对的样本中，有多少被新模型弄错”；它没有衡量新模型在旧模型答错样本上的新增正确，也没有区分评判器噪声。对开放式生成任务，应保存旧模型和新模型的原始输出、评分依据及评判器版本，必要时进行人工复核，否则一个 judge 的偶然变化可能被误报成模型回归。

模型评估不能只看平均分提升。对实际系统而言，一次高风险场景的退化可能比多个低风险题目的小幅提升更重要，所以回归样本、业务规则和安全样本应当保留原始输出、评判依据和模型版本，而不是只保存一个汇总比例。

## 10. 实验追踪应该记录什么

一个训练实验至少要记录：

| 类别 | 内容 |
| --- | --- |
| 代码 | git commit、分支、diff |
| 数据 | 数据版本、配比、过滤规则、token 数 |
| 模型 | 架构、参数量、tokenizer、context length |
| 训练 | lr、batch、optimizer、schedule、precision、parallelism |
| 环境 | GPU 类型、数量、驱动、框架版本 |
| 日志 | loss、grad norm、lr、tokens/sec、OOM/NaN |
| checkpoint | 保存路径、step、token count、评估结果 |
| 结论 | 提升、退化、异常、下一步 |

这些字段还应尽量有可验证的引用：数据和 checkpoint 保存内容摘要或哈希，代码保存 commit 与未提交 diff，环境保存依赖锁文件和 CUDA/驱动信息，评估保存脚本、prompt、解码参数和逐样本结果。没有这些记录，实验也许能够再次运行，却很难确认运行的是同一个实验。

实验记录可以抽象为：

```math
R_{\mathrm{run}}=
(v_{\mathrm{code}},v_{\mathrm{data}},v_{\mathrm{tok}},v_{\mathrm{cfg}},h,s,A,M)
```

其中 `v_code` 是代码版本，`v_data` 是数据版本，`v_tok` 是 tokenizer 版本，`v_cfg` 是模型和训练配置版本，`h` 是硬件和依赖环境，`s` 是随机种子，`A` 是 artifact 列表，`M` 是指标表。这里的 `R_run` 是一份可追溯记录，不是把实验压缩成 run name；例如只记录一个 git commit，却没有保存数据过滤规则和评估脚本版本，仍然无法解释结果。

可以先衡量记录层面的缺失情况。设 `U_repro` 是复现所需的字段集合，`M_repro` 是缺失、不可解析或无法取得原始 artifact 的字段集合：

```math
M_{\mathrm{repro}}=
\{u\in U_{\mathrm{repro}}\mid
\operatorname{missing}(u)\lor\operatorname{unresolvable}(u)\}
```

```math
C_{\mathrm{repro}}=
1-\frac{|M_{\mathrm{repro}}|}{|U_{\mathrm{repro}}|}
```

`C_repro` 只是记录覆盖率：它等于 1，仍不等于重新运行后每个数字都相同。真正做复现时，还要比较原实验与复现实验的指标偏差。例如对第 `j` 个指标，可以写成：

```math
\Delta_{j}=
\left|q_{j}^{\mathrm{rerun}}-q_{j}^{\mathrm{original}}\right|
```

对于 loss、吞吐、benchmark 分数等指标，允许的偏差应分别定义，并说明数据顺序、硬件、随机性和评估脚本是否一致。这样，“可复现”才同时包含可取得的材料和重新运行后的证据，而不是一个没有定义的完整标记。

实验管理的核心也由此变得清楚：让别人能够取得同一组输入和 artifact，知道每次变化来自哪一个版本，并能判断复现实验与原实验的差异是否在合理范围内。

## 11. 实验命名和版本管理

实验命名要能表达关键信息。

坏命名：

```text
run1
test_new
final_v2
```

更好的命名：

```text
7b_pretrain_lr3e-4_bs4m_tok2t_data-v5_bf16_fsdp
```

不一定要很长，但至少应该能看出：

1. 模型规模。
2. 训练阶段。
3. 数据版本。
4. 关键超参。
5. 时间或 run id。

命名只解决“人能不能快速找到实验”，不能承担完整版本管理。超参数很多时，名称应保持稳定的短格式，把完整配置放进不可变的 manifest；否则为了改一个字段就不断出现 `final_v2_final`，而名称和真实配置逐渐脱节。

版本管理包括：

1. 代码版本。
2. 数据版本。
3. tokenizer 版本。
4. config 版本。
5. checkpoint 版本。

其中数据版本尤其重要。很多训练差异来自数据变化，而不是模型或优化器变化。

因此，一个 checkpoint 的目录或 artifact manifest 通常应同时指向代码、数据、tokenizer、训练配置和评估结果，而不是只依赖文件名。保留旧版本时也要保留这些引用；只归档一份权重，却删除了它对应的数据配比和评估脚本，未来仍然无法解释“为什么它最好”。

## 12. 可复现性

大模型训练完全 bit-level 复现很难，但工程上至少要做到结论可复现。

影响复现的因素：

1. 随机种子。
2. 数据顺序。
3. 分布式并行策略。
4. kernel 非确定性。
5. mixed precision。
6. checkpoint 恢复状态。
7. 依赖库版本。

可复现性最低要求：

```text
同样代码 + 同样数据 + 同样配置，可以得到相近 loss 曲线和相近评估结论。
```

这里至少有三种不同强度的复现目标：

1. **字节级复现**：输出文件或参数逐字节相同，通常只在严格固定硬件、软件、随机状态和执行顺序的窄环境中可行。
2. **轨迹级复现**：loss、学习率、吞吐和 token 进度在容差内相近，适合定位恢复和训练稳定性问题。
3. **结论级复现**：关键评估结论、排序或业务决策保持一致，适合大规模分布式训练的实际对比。

不要承诺所有大规模分布式训练都能逐 bit 复现；但也不能用这个困难掩盖缺少数据版本、评估脚本或 checkpoint manifest 的记录问题。应当在实验开始前写清楚本次实验追求哪一种复现强度。

## 13. 实验对比的基本原则

比较两个实验时，要尽量做到单变量控制。

例如你想比较学习率：

```text
只改 learning rate
其他数据、模型、batch、seed、训练 token 尽量一致
```

如果同时改了数据、学习率和 tokenizer，就很难判断提升来自哪里。

常见错误：

1. 不同训练 token 数直接比较。
2. 不同 tokenizer 直接比较 loss。
3. 不同数据版本却归因到 optimizer。
4. 只看一个 benchmark。
5. 没有统计波动和多次实验。

如果两个实验配置分别是 `h_A` 和 `h_B`，想验证的变量集合是 `V`，公平对比要求除 `V` 外的关键字段一致：

```math
D_{\mathrm{diff}}=
\{u\mid h_A(u)\ne h_B(u),u\notin V\}
```

```math
\Delta_{\mathrm{compare}}=
\left(D_{\mathrm{diff}},\left|T_A-T_B\right|,
\mathbf{1}[E_A\ne E_B]\right)
```

其中 `T_A/T_B` 是训练 token 数，`E_A/E_B` 是评估集和评估脚本版本。`\Delta_compare` 把对比中最容易被忽略的三类差异显式列出来：未列入研究变量的配置差异、训练量差异和评估定义差异。真实实验中很难做到所有变量完全一致，但至少要把差异记录清楚，不能把多变量变化包装成单变量结论。比如数据版本变化通常比学习率变化更可能改变结果，却最容易被一个“lr sweep”标题掩盖。

所以，比较两个实验时，最重要的不是把结果表格做得很大，而是先写清楚反事实问题：如果只改变 `V`，结果是否仍然发生同样方向的变化？对照越接近这个问题，结论越有解释力。

## 14. Checkpoint 选择流程

一个实用选择流程：

```text
1. 过滤掉训练异常 checkpoint。
2. 按 validation loss 选候选。
3. 看 per-domain loss，排除明显偏科。
4. 跑核心 benchmark。
5. 做 regression eval。
6. 做 safety eval。
7. 抽样人工评测。
8. 评估推理成本。
9. 选择进入后训练或部署的 checkpoint。
```

可以把 checkpoint 选择写成多目标打分：

```math
J(c)=
\alpha S_{\mathrm{val}}(c)
+\beta S_{\mathrm{bench}}(c)
+\gamma S_{\mathrm{safety}}(c)
-\lambda R_{\mathrm{reg}}(c)
-\mu \widetilde{C}_{\mathrm{infer}}(c)
```

这里 `S_val`、`S_bench` 和 `S_safety` 应先按项目定义归一化，`R_reg` 是回归风险，`\widetilde{C}_{infer}` 是归一化后的推理成本。不能直接把 nats、准确率、百分比和毫秒相加，否则权重没有可解释的尺度。权重也不是固定真理，而是要和项目目标绑定：base model 可能更重视 validation / benchmark，面向企业部署的模型可能更重视 regression / safety / latency。

总分只是方便排序的工具，不应替代约束和人工判断。比如一个模型即使总分最高，只要它在必须遵守的格式或安全场景上明显退化，就不应因为其他分数高而掩盖问题。实践中可以先排除不可接受的候选，再在剩余版本中比较总分；也可以保留 Pareto 前沿，把“更低成本但略低分”和“更高能力但更贵”的候选交给业务目标决定。

这比“最后一个 checkpoint”稳健很多。

## 15. 最小可运行的 checkpoint 与实验管理 demo

下面的 demo 用标准库模拟一个小型训练记录。输入是 3 个 checkpoint、一次恢复探针、一个实验记录和两组对照实验；输出 checkpoint 完整性、PPL、domain regression、候选打分、保留存储、恢复连续性、实验记录完整性和对照差异。它不是生产级 checkpoint 管理器，而是把本章的字段和计算口径压缩成一段可运行的最小例子。

```python
import math
from statistics import mean

required_resume_fields = [
    "model", "optimizer", "scheduler", "scaler", "rng", "data_position",
    "global_step", "consumed_tokens", "config", "tokenizer",
]

checkpoints = [
    {
        "id": "ckpt_1000", "step": 1000, "tokens_b": 52.4, "train_loss": 2.44, "val_loss": 2.31,
        "domain_loss": {"general": 2.20, "code": 2.90, "math": 2.70},
        "bench": {"qa": 0.52, "math": 0.31, "code": 0.28},
        "safety": 0.95, "regression_failures": 5, "stable": True, "size_gib": 180,
        "resume_fields": {name: True for name in required_resume_fields},
    },
    {
        "id": "ckpt_2000", "step": 2000, "tokens_b": 104.9, "train_loss": 2.29, "val_loss": 2.18,
        "domain_loss": {"general": 2.08, "code": 2.74, "math": 2.52},
        "bench": {"qa": 0.56, "math": 0.38, "code": 0.34},
        "safety": 0.94, "regression_failures": 4, "stable": True, "size_gib": 180,
        "resume_fields": {name: True for name in required_resume_fields},
    },
    {
        "id": "ckpt_3000", "step": 3000, "tokens_b": 157.3, "train_loss": 2.20, "val_loss": 2.16,
        "domain_loss": {"general": 2.05, "code": 2.98, "math": 2.41},
        "bench": {"qa": 0.57, "math": 0.40, "code": 0.29},
        "safety": 0.89, "regression_failures": 11, "stable": False, "size_gib": 180,
        "resume_fields": {**{name: True for name in required_resume_fields}, "scaler": False, "data_position": False},
    },
]

resume_probe = {
    "checkpoint_id": "ckpt_2000",
    "loss_before": 2.19,
    "loss_after": 2.20,
    "lr_before": 2.8e-4,
    "lr_after": 2.8e-4,
    "step_before": 2000,
    "step_after": 2001,
    "tokens_before_b": 104.90,
    "tokens_after_b": 104.95,
}

run_manifest = {
    "git_commit": "abc1234", "data_version": "data_v5", "tokenizer_version": "tok_v2",
    "model_config": "7b_gqa_rope", "train_config": "lr3e-4_bs4m_bf16_fsdp",
    "env": "64xa100_cuda12", "seed": 42, "eval_script": "eval_v3.py",
    "checkpoint_id": "ckpt_2000", "metrics": {"val_loss": 2.18}, "notes": "lr sweep winner",
}

run_a = {"data_version": "data_v5", "tokenizer": "tok_v2", "tokens_b": 104.9, "lr": 3e-4, "batch_tokens": 4_000_000, "seed": 42}
run_b = {"data_version": "data_v5", "tokenizer": "tok_v2", "tokens_b": 104.9, "lr": 2e-4, "batch_tokens": 4_000_000, "seed": 42}
run_bad = {"data_version": "data_v6", "tokenizer": "tok_v2", "tokens_b": 157.3, "lr": 2e-4, "batch_tokens": 4_000_000, "seed": 42}

base = checkpoints[0]
base_val_loss = base["val_loss"]
storage_budget_gib = 400
latest_k = 2
save_interval_steps = 1000
step_seconds = 8
checkpoint_write_seconds = 60
expected_step_tokens_b = 0.05

def missing_fields(ckpt):
    return [name for name in required_resume_fields if not ckpt["resume_fields"].get(name, False)]

def perplexity(loss):
    return round(math.exp(loss), 2)

def domain_regressions(ckpt, threshold=0.05):
    return [
        domain
        for domain, loss in ckpt["domain_loss"].items()
        if loss - base["domain_loss"][domain] > threshold
    ]

def score_checkpoint(ckpt):
    bench_avg = mean(ckpt["bench"].values())
    domain_penalty = 0.10 * len(domain_regressions(ckpt))
    regression_penalty = 0.02 * ckpt["regression_failures"]
    stability_penalty = 0 if ckpt["stable"] else 0.50
    score = (base_val_loss - ckpt["val_loss"]) + bench_avg + ckpt["safety"]
    score -= domain_penalty + regression_penalty + stability_penalty
    return round(score, 4)

def changed_keys(a, b, ignore=("lr",)):
    return [key for key in a if key in b and a[key] != b[key] and key not in ignore]

completeness = {ckpt["id"]: missing_fields(ckpt) for ckpt in checkpoints}
ppl = {ckpt["id"]: perplexity(ckpt["val_loss"]) for ckpt in checkpoints}
domain_report = {ckpt["id"]: domain_regressions(ckpt) for ckpt in checkpoints}
score_table = [
    {"id": ckpt["id"], "score": score_checkpoint(ckpt), "stable": ckpt["stable"]}
    for ckpt in checkpoints
]
eligible_checkpoints = [
    ckpt
    for ckpt in checkpoints
    if not missing_fields(ckpt) and ckpt["stable"]
]
best = max(eligible_checkpoints, key=score_checkpoint)

latest_retained = sorted(checkpoints, key=lambda ckpt: ckpt["step"], reverse=True)[:latest_k]
retained_ids = sorted({best["id"], *(ckpt["id"] for ckpt in latest_retained)})
retained_storage_gib = sum(ckpt["size_gib"] for ckpt in checkpoints if ckpt["id"] in retained_ids)
save_overhead = round(checkpoint_write_seconds / (save_interval_steps * step_seconds), 4)
tokens_per_step_b = (
    (checkpoints[1]["tokens_b"] - checkpoints[0]["tokens_b"])
    / (checkpoints[1]["step"] - checkpoints[0]["step"])
)
max_lost_tokens_b = round(save_interval_steps * tokens_per_step_b, 1)

loss_jump = abs(resume_probe["loss_after"] - resume_probe["loss_before"]) / resume_probe["loss_before"]
token_jump = abs(
    (resume_probe["tokens_after_b"] - resume_probe["tokens_before_b"])
    - expected_step_tokens_b
)
resume_diagnostics = {
    "loss_continuous": loss_jump < 0.02,
    "lr_continuous": resume_probe["lr_before"] == resume_probe["lr_after"],
    "step_continuous": resume_probe["step_after"] == resume_probe["step_before"] + 1,
    "tokens_continuous": token_jump < 0.001,
}

manifest_required = [
    "git_commit", "data_version", "tokenizer_version", "model_config", "train_config",
    "env", "seed", "eval_script", "checkpoint_id", "metrics", "notes",
]
manifest_missing = [name for name in manifest_required if name not in run_manifest]
fair_lr_compare = changed_keys(run_a, run_b) == []
bad_compare_changed = changed_keys(run_a, run_bad)

diagnostics = {
    "best_complete": completeness[best["id"]] == [],
    "best_stable": best["stable"],
    "resume_continuous": all(resume_diagnostics.values()),
    "storage_under_budget": retained_storage_gib <= storage_budget_gib,
    "manifest_complete": manifest_missing == [],
    "fair_lr_compare": fair_lr_compare,
    "bad_compare_detected": bad_compare_changed == ["data_version", "tokens_b"],
    "safety_ok": best["safety"] >= 0.92,
    "regression_ok": best["regression_failures"] <= 5,
}

print("completeness=", completeness)
print("ppl=", ppl)
print("domain_report=", domain_report)
print("score_table=", score_table)
print("best_checkpoint=", best["id"])
print("retained_ids=", retained_ids)
print("retained_storage_gib=", retained_storage_gib)
print("save_overhead=", save_overhead)
print("max_lost_tokens_b=", max_lost_tokens_b)
print("resume_diagnostics=", resume_diagnostics)
print("manifest_missing=", manifest_missing)
print("fair_lr_compare=", fair_lr_compare)
print("bad_compare_changed=", bad_compare_changed)
print("diagnostics=", diagnostics)
```

期望输出：

```text
completeness= {'ckpt_1000': [], 'ckpt_2000': [], 'ckpt_3000': ['scaler', 'data_position']}
ppl= {'ckpt_1000': 10.07, 'ckpt_2000': 8.85, 'ckpt_3000': 8.67}
domain_report= {'ckpt_1000': [], 'ckpt_2000': [], 'ckpt_3000': ['code']}
score_table= [{'id': 'ckpt_1000', 'score': 1.22, 'stable': True}, {'id': 'ckpt_2000', 'score': 1.4167, 'stable': True}, {'id': 'ckpt_3000', 'score': 0.64, 'stable': False}]
best_checkpoint= ckpt_2000
retained_ids= ['ckpt_2000', 'ckpt_3000']
retained_storage_gib= 360
save_overhead= 0.0075
max_lost_tokens_b= 52.5
resume_diagnostics= {'loss_continuous': True, 'lr_continuous': True, 'step_continuous': True, 'tokens_continuous': True}
manifest_missing= []
fair_lr_compare= True
bad_compare_changed= ['data_version', 'tokens_b']
diagnostics= {'best_complete': True, 'best_stable': True, 'resume_continuous': True, 'storage_under_budget': True, 'manifest_complete': True, 'fair_lr_compare': True, 'bad_compare_detected': True, 'safety_ok': True, 'regression_ok': True}
```

这个 demo 的关键是把“选哪个 checkpoint”拆成可解释的观察：先列出恢复字段缺失，再把不稳定版本排除在候选之外；随后比较 validation / PPL / domain / benchmark / safety / regression，最后查看实验记录、对照差异和保留存储。输出的每一项都对应正文中的一个问题，读者可以把 toy 数据替换成真实训练日志，而不必依赖一个隐藏的总判断。

## 16. 面试官会怎么问

这一节不是把前面的内容再压缩成背诵答案，而是展示如何把工程判断讲出因果关系。面试官真正想知道的，通常不是你能否背出字段名称，而是你能否说明“缺少这个字段会破坏什么证据”。

### 问法 1：checkpoint 里应该保存什么？

回答时先区分用途。如果只做推理，model weights、model config 和 tokenizer 通常已经足够；如果要从中断处继续训练，还必须保存 optimizer state、scheduler state、必要的 mixed-precision scaler、global step、consumed tokens、random state、data state 和分布式分片信息。少了 optimizer 动量，参数虽然能加载，下一次更新却不再是原来的更新；少了 data state，训练可能重复或跳过样本；少了 tokenizer 版本，输入和标签的 token 边界也可能改变。

更完整的回答还会补一句：字段齐全只是文件层面的条件，恢复后仍要用 loss、学习率、step、token 增量和数据游标做短程连续性检查，并核对 shard、manifest 和版本。

### 问法 2：怎么选择最好的 checkpoint？

不能只选最后一步。我会先排除缺失关键状态或出现明显数值异常的版本，然后按相同的训练 token 数和相同评估口径比较 validation loss、per-domain loss、目标 benchmark、regression 和 safety。对部署候选，还要加入推理延迟、显存占用和成本；对开放式生成，则保留人工抽样和原始输出，避免总分掩盖关键退化。

如果多个目标互相冲突，我不会把一个未经归一化的总分当成客观真理，而会说明项目优先级，或者给出 Pareto 候选。这样“最好”才有明确的使用场景。

### 问法 3：为什么 train loss 降低不代表模型更好？

train loss 降低只说明模型更好地拟合了当前训练目标。数据重复、训练集泄漏、过拟合，或者数据配比变得更容易，都可能让它下降，却不带来泛化能力、安全性或目标任务的提升。因此我会同时看干净的 validation、per-domain 分布、下游任务、回归样本和安全评测，并核对评估集是否被训练数据污染。

如果只报告一个总 loss，面试官无法判断模型究竟变好了，还是只在高频、低难度的数据上变好了。

### 问法 4：如何保证实验可复现？

我会把代码 commit 和未提交 diff、数据版本与过滤规则、tokenizer、模型和训练配置、依赖及硬件环境、随机种子、checkpoint manifest、评估脚本、prompt、解码参数和逐样本结果放进同一份实验记录。大规模分布式训练不一定能 bit-level 复现，所以还要先定义目标：是复现轨迹，还是复现最终结论，并为 loss 和评估分数分别设置合理容差。

另一个关键点是记录“拿不到什么”。如果数据对象过期、artifact 哈希无法解析或评估脚本没有版本，应该把它作为复现限制写在结论旁边，而不是用一个看起来完整的 run name 掩盖缺口。

### 问法 5：两个实验怎么公平比较？

先写清楚要验证的变量，例如只验证 learning rate，然后保持模型、数据版本、tokenizer、训练 token、有效 batch、精度、并行策略、seed、评估集和评估脚本尽量一致。之后对配置做差异清单；如果 data version 或训练 token 也变了，就应把它们作为混杂因素报告，不能继续把结果称为单变量学习率实验。

最后不要只看一个平均分。逐 domain 结果、回归样本、安全指标和统计波动，往往能解释两个看似相近的总分为什么会导向不同的工程选择。

### 问法 6：恢复成功后为什么还要跑探针？

因为 `load_state_dict` 成功只验证了当前代码能读出张量，并没有验证 scheduler 的计数、optimizer 的内部状态、数据游标或 rank 间的 shard 是否处在正确位置。我会保存中断前后的 loss、学习率、step、consumed tokens 和数据 cursor，先运行一小段固定长度的恢复探针，再与未中断或参考轨迹比较。探针发现的是偏差，偏差的类型才是定位问题的入口。

## 17. 本章小结

本章核心结论：

1. Checkpoint 是训练容错、评估和版本管理的核心产物。
2. 恢复训练需要 model、optimizer、scheduler、scaler、随机状态和数据状态。
3. 保存频率要在容错、评估和存储成本之间平衡。
4. 不能默认最后一个 checkpoint 最好。
5. Validation loss 是基础指标，但不是完整能力评估。
6. Per-domain loss 能发现模型偏科和能力退化。
7. Benchmark 要结合污染检测、回归测试和人工评测使用。
8. 实验追踪要记录代码、数据、模型、训练配置、环境和结论。
9. 可复现性要求结论可复现，不一定要求 bit-level 完全一致。
10. 面试中要把 checkpoint、评估和实验管理讲成一个长期训练系统。

真正成熟的训练记录，应该让一个没有参与原实验的人回答三个问题：这份参数从哪里来，为什么选择它，以及如果恢复或复现失败，缺少哪一段证据。checkpoint、评估和实验管理并不是三个互不相干的后台任务，而是同一个训练结论的三个观察面。

## 18. 资料与进一步阅读

本章涉及的框架行为优先以官方文档为准，机制和方法论再参考原始论文：

- [PyTorch Saving and Loading a General Checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)：说明如何保存模型、优化器、epoch/step 等恢复状态；具体字段仍需按训练框架补充。
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)：说明分布式 checkpoint 的保存、加载和 state-dict 抽象，适合核对 sharded artifact 的接口边界。
- [PyTorch Reproducibility Notes](https://pytorch.org/docs/stable/notes/randomness.html)：说明随机种子、确定性算法和跨平台复现的限制；官方明确提醒完全复现并非所有环境都能保证。
- [Hugging Face Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)：说明 causal language model 在固定长度窗口下计算 perplexity 时的 token shift、stride 和统计口径。
- [Hugging Face Transformers Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)：说明 `save_strategy`、`resume_from_checkpoint`、`load_best_model_at_end` 和 `metric_for_best_model` 等训练管理参数的实际语义。
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)：说明记录参数、指标和 artifacts，以及比较 runs 的实验追踪思路；它是工具实践，不是对任何项目的唯一架构要求。
- [PyTorch Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html)：说明参数、梯度和 optimizer state 在 FSDP 下的分片边界。
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)：说明 optimizer、gradient 和 parameter 分片如何降低训练内存，论文中的实现和硬件假设不能直接等同于每个 runtime 的 checkpoint 格式。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 的原始论文；本章引用其训练状态和模型配置的背景，不把论文中的实验设置当成现代训练系统的默认配置。

官方文档适合核对 API 和版本行为，论文适合核对机制与方法来源，实验平台文档适合了解记录方式。至于保存频率、恢复偏差阈值、domain 划分、评估权重和复现容差，都必须结合目标数据、模型规模、硬件和业务风险重新测量。
