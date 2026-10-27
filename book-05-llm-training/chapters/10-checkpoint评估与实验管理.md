# 第十章：Checkpoint、评估与实验管理：让训练结论可恢复、可比较

一次大模型训练结束时，真正留下的不是一条下降的 loss 曲线，而是一组可以被别人重新解释的证据：模型参数来自哪个数据版本，训练停在什么 token 位置，验证集和下游任务表现如何，是否有能力回归，最终选择的 checkpoint 为什么不是另一个版本。

初学者可以把 checkpoint 理解成训练过程的“存档”。只保存模型权重，能够让模型再次做推理；要从中断处继续训练，还要保存优化器、学习率进度、随机状态和数据位置。专家则会继续追问：这个存档是否原子写完？分片布局是否匹配？恢复后的一步更新是否与未中断轨迹连续？

评估也不是训练结束后的装饰。validation loss 只反映某种 token 预测目标，下游 benchmark 只覆盖有限能力，安全和回归评估还可能揭示平均分掩盖的退化。实验管理把代码、数据、配置、环境、artifact 和评估结果绑在一起，才能知道一次提升究竟来自什么。

本章沿着一条具体链路展开：先定义训练状态，再讨论保存频率和成本；接着验证恢复连续性，拆开 validation、per-domain、benchmark、regression 和 safety 证据；最后说明如何管理实验版本、公平比较和选择 checkpoint。章节后部把这些概念放进几个故障和决策场景，帮助读者把抽象字段还原成真实训练系统中的判断。

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
\neg\mathrm{available}(u)\}
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

这些量的有效域是 S>=0、f 为正整数、B_tok>=0、C_save>=0、T_step>0。训练尚未执行任何 step 时不应报告平均保存开销；若异步保存还未提交 manifest，已发起的写入也不能被计入可恢复版本数。

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
\frac{|L^+-L^-|}{L^-}
```

```math
\delta_{\eta}=
\frac{|\eta^+-\eta^-|}{\eta^-}
```

```math
\delta_{\mathrm{step}}=
\left|(s^+-s^-)-1\right|
```

如果探针只跨越一个 optimizer step，还应检查 consumed tokens 的增量。设 `B_{\mathrm{tok}}^+` 是恢复后这一步按实际 batch 计算的预期 token 数，则：

```math
\delta_{\mathrm{tok}}=
\frac{\left|(T^+-T^-)-B_{\mathrm{tok}}^+\right|}
{B_{\mathrm{tok}}^+}
```

因此，恢复探针更适合记录为偏差向量：

```math
R_{\mathrm{resume}}=
(\delta_L,\delta_{\eta},\delta_{\mathrm{step}},\delta_{\mathrm{tok}})
```

这个向量比一个真假值更有信息。`\delta_L` 较大，可能指向数据 batch、loss mask 或 optimizer state；`\delta_{\eta}` 异常，优先检查 scheduler 的 step 口径；`\delta_{\mathrm{step}}` 或 `\delta_{\mathrm{tok}}` 异常，则要检查 global step、gradient accumulation 和 data cursor。阈值 `\tau_L`、`\tau_{\eta}` 只能作为当前实验的比较尺度，不能被当成所有模型都适用的固定常数。

相对偏差要求用于分母的 L^-、eta^- 和 B_tok^+ 都严格大于 0，且比较值有限。loss 恰为 0、初始学习率为 0 或恢复后没有有效 token 的情形，应分别报告绝对差、调度阶段或 `not_applicable`，而不是在分母加 epsilon 把“没有可比基线”伪装成一个稳定的小偏差。

还要注意，token count 连续并不能证明样本顺序连续。一个数据管道可能正好消费了相同数量的 token，却从另一个 shard 开始；因此 checkpoint 中的 data state 至少应包含数据版本、epoch 或 shard 顺序、当前 shard、样本/packed sequence 游标，以及必要时的 sampler 状态。恢复探针应把这些字段和日志里的 token 数放在同一张记录中。

分布式训练还多一层风险：某个 rank 的 shard 写完，不代表整个 checkpoint 已经可读。发布一个 checkpoint 前，应确认各 rank 的文件、索引和元数据都已落盘，并让读取端能够发现“未完成写入”的状态，而不是把半套 shard 当成最新版本。对象存储上的临时前缀、完成标记或 manifest 都是实现手段；关键是读取者能区分正在写入和完整发布的 artifact。

以 PyTorch 2.14 的 Distributed Checkpoint（DCP）为例，分布式 API 可以让多个 rank 按 state dict 保存和读取状态，并依据加载端的分片布局重新分配张量；这解决的是“分布式状态如何写读和重分片”的问题。DCP 的协调写入流程会在各 rank 数据写完后写入 metadata，并标记 checkpoint 成功；应用仍要决定何时发布自己的版本指针、如何校验 manifest，以及存储后端是否支持所需的可见性和原子性。异步保存返回的 future 要完成后，才能把新版本公布给读取者。异步也有代价：staging 可能复制每个 rank 的 checkpoint 状态，抬高 CPU 或 pinned-memory 峰值；同时排队多个未完成写入会继续增加内存压力。DCP 文档还明确指出，保存的 state dict 不保证跨 PyTorch 版本向后兼容，因此切换训练环境时必须执行真实恢复演练。

以 PyTorch Distributed Checkpoint（DCP）为例，分布式 API 可以让多个 rank 按 state dict 保存和读取状态，并依据加载端的分片布局重新分配张量；这解决的是“分布式状态如何写读和重分片”的问题。应用仍要定义一份 checkpoint 何时算完整：所有 rank 的写入是否成功、索引是否可读、manifest 和校验信息是否齐全、恢复程序是否能解释这些字段。异步保存还要等待写入 future 完成后再发布版本指针。DCP 的 API 成功返回不等价于跨对象存储事务提交，也不等价于跨 PyTorch 版本的永久兼容；这些边界要按使用的 PyTorch 版本和存储后端核对。

因此，说明 checkpoint 恢复时，不能只停留在“调用 `load_state_dict`”。完整流程应先验证 artifact 和版本，再恢复所有训练状态，最后用 loss、学习率、step、token 增量和数据游标做一个短探针；如果某一项偏离，就根据偏差类型定位，而不是盲目继续训练。

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

这里的有效 token 分母必须大于 0，L_val 必须是有限实数。PPL 是指数变换：很大的 loss 可能使有限精度实现溢出为 Inf，因此日志应保存原始 loss，并把 Inf PPL 标记为数值范围问题而非一个可排序的有限分数。

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

每个 domain 的有效 token 分母都必须大于 0，权重满足 w_k>=0 且 sum_k w_k=1。空 domain 应从本次聚合中排除并记录为未评估，而不是以零 loss 参与平均；否则只改变切分或过滤规则就能人为降低总 loss。

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

### 8.1 Benchmark 分数由一份评测协议产生

一个分数不只由模型权重决定。评测至少要固定模型 checkpoint 与 tokenizer、chat template、benchmark 数据和 split、few-shot 示例及顺序、prompt、生成参数、答案抽取和评分器、评测框架版本。可以把一次评测记成协议对象：

```math
E=(c,v_{tok},v_{template},D_{task},S_{split},P,F,G,M,H)
```

这里 `c` 是 checkpoint，`D_task` 和 `S_split` 是任务数据及划分，`P` 是 prompt/few-shot 配置，`F` 是输入格式，`G` 是生成设置，`M` 是 metric/scorer，`H` 是 harness 及其版本。只有关键字段一致，两个分数才是在同一把尺子上测得。实际报告还要保留逐样本输入、模型输出、解析结果和评分，便于重算指标与定位异常。

代表性评测研究和工具分别处理不同层次的问题：

| 方案 | 针对的问题与核心做法 | 证据和限制 | 可继续改进的方向 |
| --- | --- | --- | --- |
| HELM | 过去模型评测覆盖的场景不一致，且常只报准确率。HELM 先按场景和指标组织评测，再在固定条件下统一运行；其核心集合覆盖 16 个场景、7 类指标，并发布原始 prompts/completions。 | 论文报告，在其选定的 core scenarios 上，模型平均覆盖率由此前 17.9% 提高到 96.0%。这是 2022 年论文所定义的场景集合和评测设置，不代表覆盖了所有真实用例，也不意味着多指标可无损压成一个总排名。 | 保留共同核心套件以便横向比较，同时增加目标领域、地区/语言和新出现任务的独立切片；分别报告每种指标，避免综合分数掩盖取舍。 |
| lm-evaluation-harness（lm-eval） | 不同研究把任务实现、提示、答案解析和模型接入散落在不同代码中。lm-eval 用统一接口和 YAML/Python task 把数据、prompt、请求类型、后处理和指标封装起来，目标是可复用地执行各种评测。 | “Lessons from the Trenches”把它定位为非规定 benchmark 的 orchestration library：执行过程更容易复核，但它不会替使用者决定 benchmark 是否有代表性；任务定义和 harness 版本变化仍会改变分数。 | 报告精确 task/version、代码 commit、模型/tokenizer、few-shot 与 generation 参数；保存样本级 outputs，避免只给一个可以被不同实现重新解释的总分。 |
| Dror et al. 的 NLP 显著性检验指南（ACL 2018） | 同一批题同时交给两个 checkpoint，按题目配对比较，减少题目难度差异对比较的干扰；按任务和 metric 选择 McNemar、bootstrap、随机化等检验，而不是对所有指标套同一个公式。 | 统计检验回答“观察到的差异是否可能由抽样波动造成”，不回答提升是否对业务重要。有限测试题的不确定性也不涵盖训练 seed 方差、prompt 选择或 scorer 偏差。 | 同时报差值、区间和预先声明的最小有意义提升；对多个任务/指标控制多重比较；必要时增加独立训练 seed，而不是只扩大同一测试集。 |

这几类工作不能互相替代：HELM 主要规定“看哪些场景和维度”，lm-eval 主要解决“如何以可复核方式执行任务”，统计检验回答“观察到的差距有多不确定”。它们都不能单独证明某个 benchmark 完整代表了目标用户的真实能力。

对于准确率类指标，样本数为 `n`、观测准确率为 `p` 时，独立同分布的粗略标准误可以写成：

```math
\mathrm{SE}(p)\approx
\sqrt{\frac{p(1-p)}{n}}
```

它不是所有评测的完整置信区间，尤其不适用于有相关样本、生成式评分或复杂 judge 的场景，但足以提醒我们：几十道题上的几个百分点变化，未必比评测噪声大。更可靠的比较通常保留逐样本结果，使用配对样本分析，并固定 prompt、解码参数、评判模型和随机种子。

这个近似要求 n>0、0<=p<=1，并把样本近似视作独立同分布的伯努利观测；p=0 或 p=1 时标准误公式给出 0，却不代表没有不确定性，尤其当样本量很小或评测集经过筛选时。

比较两个模型时，两个独立准确率的标准误并没有利用“同一道题同时测两个模型”的配对信息。令 `b` 是旧模型答对而新模型答错的样本数，`c` 是旧模型答错而新模型答对的样本数；新模型准确率减旧模型准确率为 `(c-b)/n`，McNemar 检验等配对检验主要看这两类不一致样本，而不是把两组总错误数当成互不相关。例如两模型在 1000 道题上各对 900 题，若 `b=c=30`，整体准确率相同；若 `b=10,c=30`，新模型净多答对 20 题，差值为 2 个百分点，但这是否稳定仍要结合不一致样本数量和检验区间判断。

对连续分数或样本级奖励，可对同一题的 `(score_old, score_new)` 成对重采样，得到差值分布；不能分别独立重采样后再假装两组样本没有对应关系。还有两种不确定性要分开报告：在固定 checkpoint 上重采样测试题，估计题目样本的不确定性；改变训练 seed 或 few-shot 选择重复评测，观察训练和提示选择带来的波动。lm-eval 论文明确把 bootstrap 的题目分布波动与模型/提示随机性区分开。若同时筛查很多 benchmark、domain 和阈值，还会出现多重比较：应先声明主要指标和决策门槛，或对探索性比较做相应校正，避免从一长串结果里只挑显著项。检验选择应参照任务和指标性质，例如 [Dror et al. 的指南](https://aclanthology.org/P18-1128/)及其[常见 NLP 检验附录](https://arxiv.org/abs/1809.01448)；不能把 McNemar 或 paired bootstrap 当成所有生成式评测的通用按钮。

以上配对统计方法是 NLP 评测中的工具，不是自动的“晋级门槛”。评测题目可能同源或重复，judge 评分可能不稳定，重复训练也可能成本很高。至少应公开效应大小和区间、样本/seed 数、主要检验及多重比较策略；在决策上还需把统计差异与实际影响阈值分开。

### 8.2 Benchmark 污染：字符串重叠不是完整答案

当 benchmark 的题干、答案或近似改写出现在训练数据里，模型可能记住评测内容而不是泛化出解决任务的能力。对开放数据，精确匹配和 n-gram overlap 可以筛出明显重复；但 paraphrase、翻译、模板变体或任务级泄漏可能绕过字面匹配。另一方面，一段短重叠也不必然说明模型记住答案，规则会产生误报。因此“没有检出字符串重叠”不能推出“确定无污染”。

有代表性的研究采用了不同定义与访问条件：

| 研究 | 解决什么问题、用了什么证据 | 主要限制 | 实际使用方式 |
| --- | --- | --- | --- |
| Oren et al., *Proving Test Set Contamination in Black-Box Language Models*（ICLR 2024） | 当闭源模型训练语料不可见时，比较基准 canonical 顺序与随机打乱顺序的序列似然。若模型显著偏好原始顺序，可在 exchangeability 等前提下按论文方法检测并给出精确假阳性保证；论文报告可检测小到 1.4B 模型、1000 题测试集和语料中少量出现的任务。 | 它检测的是模型行为中与特定记忆机制相符的信号，不是通用“污染证明器”。需要可取得序列似然及可交换题目假设；没检出不能证明训练数据干净，也不适用于所有打分接口。 | 将行为检测作为闭源模型审计的一条证据；记录 benchmark 的排序、可交换性和似然接口条件，并与已知语料扫描分开报告。 |
| ConStat（NeurIPS 2024） | 把污染操作化为“主要 benchmark 上异常升高、但不能泛化到重述样本、同分布合成样本或同任务参考基准”的表现差距，并相对参考 benchmark 与模型量化该差距。 | 它检验的是表现的非泛化异常，不等同于证明原始样本确实进入训练集；判断依赖参考任务/参考模型的选择与可比性。 | 用重述/新采样/同任务参考集检查提升能否迁移；把结果写成污染风险或非泛化表现证据，不写成二元的“干净/不干净”认证。 |

比较稳妥的做法是多证据并用：训练语料可访问时做精确与近似去重、来源时间核验和切分审计；语料不可访问时使用适合接口的行为检查，并准备新写、受控或隐藏的留出样本；对 flagged 样本回看原始输出和来源。每种方法的覆盖层级不同，必须报告访问权限、阈值、误报/漏报风险和未检查的污染类型。新 benchmark 也会被后续爬取进入训练数据，因此数据版本和评测集版本要一起管理，必要时轮换保留集。

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
\sum_i \mathbf{1}[z_i^{old}=1]
}
```

如果样本带业务权重 `w_i`，高风险回归率可以写成：

```math
R_{\mathrm{reg,w}}=
\frac{
\sum_i w_i\mathbf{1}[z_i^{old}=1]\mathbf{1}[z_i^{new}=0]
}{
\sum_i w_i\mathbf{1}[z_i^{old}=1]
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

R_reg 的分母要求至少存在一个旧模型答对的样本；加权版本还要求这些样本的总正权重大于 0，且 w_i 非负。若旧模型没有任何正确样本，回归率应为 `not_applicable`，不能因分母加 epsilon 而显示为 0；同时应另报新模型的新增正确和总体表现。

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
\mathrm{missing}(u)\lor\mathrm{unresolvable}(u)\}
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

该覆盖率要求 U_repro 非空，且 M_repro 是它的子集；空需求集合不应自动得到 100% 复现率。Delta_j 也只有在同一指标定义、同一单位和两个有限数值下才可比较，缺失 artifact 或不可比评估脚本应作为限制单列，而不是赋予零偏差。

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

下面的 demo 用标准库模拟一个小型训练记录。输入是 3 个 checkpoint、一次恢复探针、一个实验记录和两组对照实验；输出 checkpoint 完整性、PPL、domain regression、归一化打分、保留存储、恢复连续性、实验记录完整性和对照差异。这里故意保留一个明确的 toy 选择政策：先过稳定性、恢复完整性、安全、回归和领域门槛，再用事先给定的 validation loss 上限归一化，并将所有分量压到 0 到 1 后加权。这个 profile 假设 scaler-enabled FP16 训练，因此把 `scaler` 列为必需状态；若训练采用 BF16 或不使用 scaler，应按 run 配置把它标为 `not_applicable` 或从该 profile 的必需字段中移除。它不是生产级 checkpoint 管理器，也不代表行业通用权重。

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
        "safety": 0.95, "regression_failures": 5, "regression_cases": 100, "stable": True, "size_gib": 180,
        "resume_fields": {name: True for name in required_resume_fields},
    },
    {
        "id": "ckpt_2000", "step": 2000, "tokens_b": 104.9, "train_loss": 2.29, "val_loss": 2.18,
        "domain_loss": {"general": 2.08, "code": 2.74, "math": 2.52},
        "bench": {"qa": 0.56, "math": 0.38, "code": 0.34},
        "safety": 0.94, "regression_failures": 4, "regression_cases": 100, "stable": True, "size_gib": 180,
        "resume_fields": {name: True for name in required_resume_fields},
    },
    {
        "id": "ckpt_3000", "step": 3000, "tokens_b": 157.3, "train_loss": 2.20, "val_loss": 2.16,
        "domain_loss": {"general": 2.05, "code": 2.98, "math": 2.41},
        "bench": {"qa": 0.57, "math": 0.40, "code": 0.29},
        "safety": 0.89, "regression_failures": 11, "regression_cases": 100, "stable": False, "size_gib": 180,
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

if not checkpoints:
    raise ValueError("at least one checkpoint is required")
base = checkpoints[0]
storage_budget_gib = 400
latest_k = 2
save_interval_steps = 1000
step_seconds = 8
checkpoint_write_seconds = 60
expected_step_tokens_b = 0.05
val_loss_cap = 4.0
safety_gate = 0.92
max_regression_failures = 5
score_weights = {"val": 0.40, "benchmark": 0.30, "safety": 0.15, "regression": 0.15}


def finite_number(name, value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def probability(name, value):
    value = finite_number(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def validate_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a mapping")
    if not isinstance(ckpt.get("id"), str) or not ckpt["id"]:
        raise ValueError("checkpoint id must be a nonempty string")
    positive_int("step", ckpt.get("step"))
    for name in ("tokens_b", "train_loss", "val_loss", "size_gib"):
        finite_number(name, ckpt.get(name), nonnegative=True)
    if not isinstance(ckpt.get("stable"), bool):
        raise ValueError("stable must be a boolean")
    failures = ckpt.get("regression_failures")
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        raise ValueError("regression_failures must be a nonnegative integer")
    positive_int("regression_cases", ckpt.get("regression_cases"))
    if failures > ckpt["regression_cases"]:
        raise ValueError("regression_failures cannot exceed regression_cases")
    probability("safety", ckpt.get("safety"))
    for name in ("domain_loss", "bench", "resume_fields"):
        if not isinstance(ckpt.get(name), dict) or not ckpt[name]:
            raise ValueError(f"{name} must be a nonempty mapping")
    for domain, loss in ckpt["domain_loss"].items():
        if not isinstance(domain, str) or not domain:
            raise ValueError("domain names must be nonempty strings")
        finite_number("domain loss", loss, nonnegative=True)
    for name, score in ckpt["bench"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("benchmark names must be nonempty strings")
        probability("benchmark score", score)
    for name in required_resume_fields:
        if not isinstance(ckpt["resume_fields"].get(name), bool):
            raise ValueError(f"resume field {name} must be a boolean")


def validate_resume_probe(probe):
    if not isinstance(probe, dict) or not isinstance(probe.get("checkpoint_id"), str):
        raise ValueError("resume probe must identify a checkpoint")
    for name in ("loss_before", "loss_after", "lr_before", "lr_after"):
        finite_number(name, probe.get(name), positive=True)
    for name in ("step_before", "step_after"):
        positive_int(name, probe.get(name))
    for name in ("tokens_before_b", "tokens_after_b"):
        finite_number(name, probe.get(name), nonnegative=True)


for checkpoint in checkpoints:
    validate_checkpoint(checkpoint)
if len({checkpoint["id"] for checkpoint in checkpoints}) != len(checkpoints):
    raise ValueError("checkpoint ids must be unique")
for previous, current in zip(checkpoints, checkpoints[1:]):
    if current["step"] <= previous["step"]:
        raise ValueError("checkpoints must be ordered by strictly increasing step")
    if current["tokens_b"] <= previous["tokens_b"]:
        raise ValueError("consumed tokens must increase with training step")
benchmark_keys = set(base["bench"])
for checkpoint in checkpoints:
    if set(checkpoint["bench"]) != benchmark_keys:
        raise ValueError("benchmark keys must match across checkpoints")
validate_resume_probe(resume_probe)
if resume_probe["checkpoint_id"] not in {ckpt["id"] for ckpt in checkpoints}:
    raise ValueError("resume probe checkpoint_id must exist")
positive_int("latest_k", latest_k)
positive_int("save_interval_steps", save_interval_steps)
finite_number("storage_budget_gib", storage_budget_gib, positive=True)
finite_number("step_seconds", step_seconds, positive=True)
finite_number("checkpoint_write_seconds", checkpoint_write_seconds, nonnegative=True)
finite_number("expected_step_tokens_b", expected_step_tokens_b, positive=True)
finite_number("val_loss_cap", val_loss_cap, positive=True)
probability("safety_gate", safety_gate)
positive_int("max_regression_failures", max_regression_failures)
if set(score_weights) != {"val", "benchmark", "safety", "regression"}:
    raise ValueError("score_weights must define every score component exactly once")
for name, weight in score_weights.items():
    finite_number(f"score weight {name}", weight, nonnegative=True)
if not math.isclose(sum(score_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError("score weights must sum to 1")


def missing_fields(ckpt):
    return [name for name in required_resume_fields if not ckpt["resume_fields"].get(name, False)]


def perplexity(loss):
    loss = finite_number("loss", loss, nonnegative=True)
    try:
        return round(math.exp(loss), 2)
    except OverflowError as exc:
        raise ValueError("loss is too large for finite perplexity") from exc


def domain_regressions(ckpt, threshold=0.05):
    threshold = finite_number("threshold", threshold, nonnegative=True)
    if set(ckpt["domain_loss"]) != set(base["domain_loss"]):
        raise ValueError("domain keys must match the baseline")
    return [
        domain
        for domain, loss in ckpt["domain_loss"].items()
        if loss - base["domain_loss"][domain] > threshold
    ]


def score_components(ckpt):
    # The fixed cap is a project policy chosen before examining the candidates.
    val_score = max(0.0, min(1.0, 1.0 - ckpt["val_loss"] / val_loss_cap))
    benchmark_score = mean(ckpt["bench"].values())
    regression_score = 1.0 - ckpt["regression_failures"] / ckpt["regression_cases"]
    return {
        "val": val_score,
        "benchmark": benchmark_score,
        "safety": ckpt["safety"],
        "regression": regression_score,
    }


def score_checkpoint(ckpt):
    components = score_components(ckpt)
    score = sum(score_weights[name] * value for name, value in components.items())
    return round(score, 4)

def changed_keys(a, b, ignore=("lr",)):
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("run configs must be mappings")
    keys = set(a) | set(b)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("run config keys must be nonempty strings")
    return sorted(
        key
        for key in keys
        if key not in ignore
        and (key not in a or key not in b or a[key] != b[key])
    )

completeness = {ckpt["id"]: missing_fields(ckpt) for ckpt in checkpoints}
ppl = {ckpt["id"]: perplexity(ckpt["val_loss"]) for ckpt in checkpoints}
domain_report = {ckpt["id"]: domain_regressions(ckpt) for ckpt in checkpoints}
score_table = [
    {
        "id": ckpt["id"],
        "components": {key: round(value, 4) for key, value in score_components(ckpt).items()},
        "score": score_checkpoint(ckpt),
        "stable": ckpt["stable"],
    }
    for ckpt in checkpoints
]
eligible_checkpoints = [
    ckpt
    for ckpt in checkpoints
    if not missing_fields(ckpt)
    and ckpt["stable"]
    and ckpt["safety"] >= safety_gate
    and ckpt["regression_failures"] <= max_regression_failures
    and not domain_regressions(ckpt)
]
best = max(eligible_checkpoints, key=score_checkpoint) if eligible_checkpoints else None

latest_retained = sorted(checkpoints, key=lambda ckpt: ckpt["step"], reverse=True)[:latest_k]
retained_ids = {ckpt["id"] for ckpt in latest_retained}
if best is not None:
    retained_ids.add(best["id"])
retained_ids = sorted(retained_ids)
retained_storage_gib = sum(ckpt["size_gib"] for ckpt in checkpoints if ckpt["id"] in retained_ids)
save_overhead = round(checkpoint_write_seconds / (save_interval_steps * step_seconds), 4)
if len(checkpoints) >= 2:
    tokens_per_step_b = (
        (checkpoints[1]["tokens_b"] - checkpoints[0]["tokens_b"])
        / (checkpoints[1]["step"] - checkpoints[0]["step"])
    )
    max_lost_tokens_b = round(save_interval_steps * tokens_per_step_b, 1)
else:
    tokens_per_step_b = None
    max_lost_tokens_b = None

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


def manifest_value_present(value):
    if value is None or value == "":
        return False
    if isinstance(value, (dict, list, tuple, set)) and not value:
        return False
    return True


manifest_missing = [
    name
    for name in manifest_required
    if name not in run_manifest or not manifest_value_present(run_manifest[name])
]
fair_lr_compare = changed_keys(run_a, run_b) == []
bad_compare_changed = changed_keys(run_a, run_bad)

diagnostics = {
    "best_exists": best is not None,
    "best_complete": best is not None and completeness[best["id"]] == [],
    "best_stable": best is not None and best["stable"],
    "resume_continuous": all(resume_diagnostics.values()),
    "storage_under_budget": retained_storage_gib <= storage_budget_gib,
    "manifest_complete": manifest_missing == [],
    "fair_lr_compare": fair_lr_compare,
    "bad_compare_detected": bad_compare_changed == ["data_version", "tokens_b"],
    "safety_ok": best is not None and best["safety"] >= safety_gate,
    "regression_ok": best is not None and best["regression_failures"] <= max_regression_failures,
}

print("completeness=", completeness)
print("ppl=", ppl)
print("domain_report=", domain_report)
print("score_table=", score_table)
print("best_checkpoint=", None if best is None else best["id"])
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
score_table= [{'id': 'ckpt_1000', 'components': {'val': 0.4225, 'benchmark': 0.37, 'safety': 0.95, 'regression': 0.95}, 'score': 0.565, 'stable': True}, {'id': 'ckpt_2000', 'components': {'val': 0.455, 'benchmark': 0.4267, 'safety': 0.94, 'regression': 0.96}, 'score': 0.595, 'stable': True}, {'id': 'ckpt_3000', 'components': {'val': 0.46, 'benchmark': 0.42, 'safety': 0.89, 'regression': 0.89}, 'score': 0.577, 'stable': False}]
best_checkpoint= ckpt_2000
retained_ids= ['ckpt_2000', 'ckpt_3000']
retained_storage_gib= 360
save_overhead= 0.0075
max_lost_tokens_b= 52.5
resume_diagnostics= {'loss_continuous': True, 'lr_continuous': True, 'step_continuous': True, 'tokens_continuous': True}
manifest_missing= []
fair_lr_compare= True
bad_compare_changed= ['data_version', 'tokens_b']
diagnostics= {'best_exists': True, 'best_complete': True, 'best_stable': True, 'resume_continuous': True, 'storage_under_budget': True, 'manifest_complete': True, 'fair_lr_compare': True, 'bad_compare_detected': True, 'safety_ok': True, 'regression_ok': True}
```

这里 validation loss 用预先固定的 `val_loss_cap=4.0` 映射到 `[0,1]`，benchmark 和 safety 本身处于 `[0,1]`，regression 用失败数除以明确的样本数得到失败率后再转成得分；权重和为 1。先执行硬门槛，再计算加权分，所以较高平均分不能抵消缺状态、训练不稳定、安全失败或明显 domain regression。更换 loss cap、权重或门槛会改变选择结果，故这些配置必须作为实验记录的一部分，不能在看完分数后再挑一组有利权重。

这个 demo 还刻意把几种常见边界显式处理：空 checkpoint 列表、重复/倒序 step、非递增 token 数、benchmark 键集合不一致会报错；无合格候选会返回 `None`；缺失配置字段会作为差异报告，不会被静默漏掉。示例使用三份结构完整的 checkpoint，因此正常路径会选择 `ckpt_2000`；如果把三个 checkpoint 的 `stable` 都设为 `False`，`best_checkpoint` 会返回 `None`，latest 保留策略仍可独立计算。

这个 demo 的作用是把“选哪个 checkpoint”拆成可解释的观察：先列出恢复字段缺失，再按硬门槛筛候选；随后比较 validation / PPL / domain / benchmark / safety / regression，最后查看实验记录、对照差异和保留存储。它只模拟记录与决策，不实现模型训练、分布式 checkpoint、统计检验或生产存储事务。

## 16. 常见误判与检查路径

前面的章节给出了字段、公式和协议，实际排障时还需要把它们串成检查路径。下面的场景来自训练系统中最容易被误解的几类现象：文件可以加载但轨迹不能接上，平均指标变好但关键领域退化，实验名称看似清楚但无法找到真实输入。每个场景都先说明现象，再给出应该验证的证据和不能直接推出的结论。

### 16.1 把“能加载”误当成“可恢复”

推理加载成功只证明当前代码能够读取一组参数。训练恢复还要证明 optimizer 的动量、scheduler 的位置、精度状态、随机状态、数据 cursor 和分片元数据都被正确解释。少了 optimizer 动量，下一次更新就不是原轨迹上的更新；少了 data state，训练可能重复或跳过样本；少了 tokenizer 版本，输入和标签的 token 边界也可能改变。

检查时先读 manifest，再检查文件哈希、参数形状、dtype、分片索引和版本引用，最后用 loss、学习率、step、token 增量和数据游标做短程恢复探针。字段齐全只是文件层面的条件；探针偏离时，应该根据偏差类型回到 scheduler、数据管道或 optimizer 状态，而不是把异常归咎于“随机性”。

### 16.2 把最后一个 checkpoint 误当成最佳版本

时间顺序只说明哪个版本保存得更晚，不说明它在目标任务上更合适。应先排除缺失关键状态、出现非有限值或违反硬约束的版本，再按相同训练 token 数和相同评估口径比较 validation loss、per-domain loss、目标 benchmark、regression 和 safety。部署候选还要加入推理延迟、显存占用和成本；开放式生成则必须保留原始输出，避免总分掩盖关键退化。

多个目标互相冲突时，不能把未经归一化的总分当成客观真理。先应用安全、格式、稳定性和资源等硬约束，再在可用候选中说明权重，或者保留 Pareto 候选。这样“最好”才有明确的使用场景，选择过程也能在数据或目标变化后重新复盘。

### 16.3 把训练 loss 下降误当成能力全面提升

train loss 降低只说明模型更好地拟合了当前训练目标。数据重复、训练集泄漏、过拟合或数据配比变得更容易，都可能让它下降，却不带来泛化能力、安全性或目标任务的提升。应该同时查看干净的 validation、per-domain 分布、下游任务、回归样本和安全评测，并核对评估集是否被训练数据污染。

如果只报告一个总 loss，读者无法判断模型究竟变好了，还是只在高频、低难度的数据上变好了。有效 token 数、领域配比和逐样本结果是解释这条曲线的必要上下文。

### 16.4 把随机种子误当成完整复现方案

随机种子只能固定随机数生成器的一部分状态，不能固定数据对象、通信规约顺序、GPU kernel、依赖版本或 checkpoint 内容。完整记录应包含代码 commit 和未提交 diff、数据版本与过滤规则、tokenizer、模型和训练配置、依赖及硬件环境、随机状态、checkpoint manifest、评估脚本、prompt、解码参数和逐样本结果。

还要先定义复现目标：是字节级、轨迹级，还是结论级，并为 loss、吞吐和评估分数分别设置容差。如果数据对象过期、artifact 哈希无法解析或评估脚本没有版本，应把它作为复现限制写在结论旁边，而不是用一个看起来完整的 run name 掩盖缺口。

### 16.5 把“配置名相似”误当成公平对照

先写清楚要验证的变量，例如只验证 learning rate，然后保持模型、数据版本、tokenizer、训练 token、有效 batch、精度、并行策略、seed、评估集和评估脚本尽量一致。随后对最终解析配置做差异清单；如果 data version 或训练 token 也变了，就应把它们作为混杂因素报告，不能继续把结果称为单变量学习率实验。

最后不要只看一个平均分。逐 domain 结果、回归样本、安全指标、有效 token 分母和统计波动，往往能解释两个看似相近的总分为什么会导向不同的工程选择。比较结果应保留原始输出和评估脚本版本，以便区分模型变化与评估器变化。

### 16.6 把 `load_state_dict` 成功误当成恢复验证完成

`load_state_dict` 成功只验证了当前代码能读出张量，并没有验证 scheduler 的计数、optimizer 的内部状态、数据游标或 rank 间的 shard 是否处在正确位置。恢复验证应保存中断前后的 loss、学习率、step、consumed tokens 和数据 cursor，先运行一小段固定长度的探针，再与未中断或参考轨迹比较。探针发现的是偏差，偏差的类型才是定位问题的入口；在探针完成前，不应把恢复后的训练日志与原轨迹直接拼接成一条连续曲线。

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
10. checkpoint、评估和实验管理共同决定训练结论能否被解释、恢复和复核。

真正成熟的训练记录，应该让一个没有参与原实验的人回答三个问题：这份参数从哪里来，为什么选择它，以及如果恢复或复现失败，缺少哪一段证据。checkpoint、评估和实验管理并不是三个互不相干的后台任务，而是同一个训练结论的三个观察面。

## 18. 章节练习：把评测分数还原成可核查的证据

### 练习一：固定评测协议

选两个 checkpoint 和一个生成式 benchmark，列出 `E` 中每个字段的实际值：checkpoint revision、tokenizer/chat template、数据与 split、few-shot 样本及顺序、prompt、generation 参数、答案抽取器、metric 和 harness commit。故意只更换 chat template 或 harness 版本重跑一小批样本，保存逐样本 outputs，判断分数变化来自模型还是评测协议。

### 练习二：同题配对比较

构造 100 道题的两个模型结果，分别统计两者都答对、都答错、只有旧模型答对和只有新模型答对的数量。计算准确率差；再说明为什么只知道两个模型各自的准确率无法恢复 discordant-pair 信息，以及 paired bootstrap 或 McNemar 检验可以回答什么、不能回答什么。补充一个实际最小提升阈值，区分统计显著与决策有意义。

### 练习三：区分题目波动、训练 seed 和提示波动

固定一个 checkpoint 做题目 bootstrap；固定题目并更换训练 seed；固定模型并更换 few-shot 顺序。分别记录三类结果分布，说明每一类重采样在估计哪种不确定性，以及为什么只扩大同一个 benchmark 的题目数不能替代多 seed 训练。

### 练习四：污染审计方案

分别为公开训练语料可访问和闭源模型不可访问两种情形，设计一套 contamination audit。至少覆盖精确/近似重叠、时间切分或新写留出题、样本级输出复核，以及 Oren 顺序似然检验或 ConStat 式泛化差距中的一种。列出每种证据的访问条件、误报/漏报方式，并写出为什么审计结论应报告风险层级而不是“绝对干净”。

### 练习五：解释多目标 checkpoint 选择

对照本章 demo 的三个 checkpoint，逐项计算归一化 validation、benchmark、safety 和 regression 分量；先应用稳定性、恢复字段、安全和 domain 门槛，再比较剩余候选。改变 validation loss 的归一化上限或各项权重，观察最佳候选是否变化，并说明这些变化体现了目标偏好，而不是模型事实本身改变。

## 19. 资料与进一步阅读

本章涉及的框架行为优先以官方文档为准，机制和方法论再参考原始论文：

- [PyTorch Saving and Loading a General Checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)：说明如何保存模型、优化器、epoch/step 等恢复状态；具体字段仍需按训练框架补充。
- [PyTorch Distributed Checkpoint 2.14](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)：说明分布式 checkpoint 的保存、加载和 state-dict 抽象，适合核对 sharded artifact 与 load-time resharding 的接口边界；事务式发布和跨版本兼容仍需应用验证。
- [Asynchronous Saving with Distributed Checkpoint 2.14](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)：说明异步 staging/write future 路径以及保存与训练计算重叠时需要等待完成的生命周期。
- [PyTorch Reproducibility Notes 2.14](https://docs.pytorch.org/docs/2.14/notes/randomness.html)：说明随机种子、确定性算法和跨平台复现的限制；官方明确提醒完全复现并非所有环境都能保证。
- [Hugging Face Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)：说明 causal language model 在固定长度窗口下计算 perplexity 时的 token shift、stride 和统计口径。
- [Hugging Face Transformers Trainer v5.19.0](https://huggingface.co/docs/transformers/v5.19.0/main_classes/trainer)：说明 `save_strategy`、`resume_from_checkpoint`、`load_best_model_at_end` 和 `metric_for_best_model` 等训练管理参数的实际语义；该页提示 v5.20.2 已发布，本章固定到实际打开核验的 v5.19.0 页面，目标环境升级后应再复核。
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)：说明记录参数、指标和 artifacts，以及比较 runs 的实验追踪思路；它是工具实践，不是对任何项目的唯一架构要求。
- [PyTorch Fully Sharded Data Parallel 2.14](https://docs.pytorch.org/docs/2.14/fsdp.html)：说明参数、梯度和 optimizer state 在 FSDP 下的分片边界。
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)：说明 optimizer、gradient 和 parameter 分片如何降低训练内存，论文中的实现和硬件假设不能直接等同于每个 runtime 的 checkpoint 格式。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 的原始论文；本章引用其训练状态和模型配置的背景，不把论文中的实验设置当成现代训练系统的默认配置。
- [Holistic Evaluation of Language Models（HELM）](https://arxiv.org/abs/2211.09110)：比较统一场景覆盖、多指标评估和原始输出发布；论文中的场景集合是覆盖取舍，不是所有真实任务的全集。
- [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)：总结 lm-eval 作为可扩展评测执行框架的设计、prompt/版本记录、原始输出和统计不确定性实践；框架本身不决定 benchmark 是否代表目标能力。
- [The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing](https://aclanthology.org/P18-1128/) 与 [Recommended Statistical Significance Tests for NLP Tasks](https://arxiv.org/abs/1809.01448)：提供 NLP 任务成对统计检验与多重比较的入口；具体方法应按 metric、样本依赖和研究设计选择。
- [Proving Test Set Contamination in Black-Box Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html)：研究通过 canonical/shuffled benchmark 顺序似然检测闭源模型的污染信号；只在其假设与可访问接口条件下成立。
- [ConStat: Performance-Based Contamination Detection in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a7f89793b9e6f8c6568dbbb6ff727b9b-Abstract-Conference.html)：以异常且不能泛化的 benchmark 表现差距定义污染信号；不等价于训练语料中样本包含关系的直接证明。

官方文档适合核对 API 和版本行为，论文适合核对机制与方法来源，实验平台文档适合了解记录方式。至于保存频率、恢复偏差阈值、domain 划分、评估权重和复现容差，都必须结合目标数据、模型规模、硬件和业务风险重新测量。
