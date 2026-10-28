# 练习与验收体系

## GPT-6.1 Sol：长任务状态账本

设计一个 replay manifest，分别记录 `model_id`、`reasoning.mode`、`reasoning.effort`、`reasoning.context`、tool call/result、compaction item、executor receipt 和 verifier receipt。用合成输入比较 `current_turn` 与 `all_turns`，再模拟 stateless input-array chaining 和 `previous_response_id` chaining；不得调用真实 API，也不得把 toy 结果写成模型质量结论。

## 阶段 1：基础验收

1. 3 句话讲清楚 LLM 到底在学什么。
2. 5 分钟讲清楚 next-token prediction。
3. 举例说明为什么预测下一个 token 需要世界知识。
4. 举例说明为什么 next-token prediction 可能导致 hallucination。
5. 用自己的话解释“大模型既会记忆，也会泛化”。
6. 手写一个 bigram next-token predictor。
7. 写出 `P(x_1, x_2, x_3, x_4)` 的自回归分解。
8. 给定 token 序列 `[10, 20, 30, 40]`，写出 input 和 label 如何错位。
9. 解释 teacher forcing 的优点和缺点。
10. 用纯 Python 写一个函数，把 token id 序列转成 next-token prediction 的 input 和 labels。
11. 手写 cross entropy。
12. 解释 perplexity。
13. 如果真实 token 概率是 0.8，计算单 token loss。
14. 如果平均 loss 是 2.0，计算 perplexity。
15. 用自己的话解释最大似然和交叉熵的关系。
16. 解释为什么 loss 低不一定代表模型真实能力强。
17. 回答 10 道 ML 基础题。
18. 对 `L(w) = (w - 5)^2` 求梯度，并手写 5 步梯度下降。
19. 修改梯度下降代码，比较 learning rate 为 0.01、0.1、1.0 的表现。
20. 解释为什么 PyTorch 需要 `zero_grad()`。
21. 用自己的话解释梯度消失和梯度爆炸。
22. 列出 loss 不下降时你会优先排查的 5 个方向。
23. 用自己的话解释 Adam 的一阶矩和二阶矩。
24. 解释 AdamW 为什么叫 decoupled weight decay。
25. 写一个 warmup + cosine decay 的学习率函数。
26. 列出训练 loss spike 时与 optimizer 相关的排查清单。
27. 解释为什么 LayerNorm 或 bias 参数通常不做 weight decay。
28. 用自己的话解释线性代数在 embedding、linear layer、attention、LoRA 和 RAG 检索中的作用。
29. 给定 `X [B,T,d_in]` 和 `W [d_in,d_out]`，推导输出 shape 并计算参数量。
30. 手算两个三维向量的 dot product、L2 norm、cosine similarity 和归一化后的 dot product。
31. 推导 `Q,K,V [B,H,T,d_h]` 下 attention score、attention weight 和 output 的 shape。
32. 给定 `d_in=d_out=4096`、LoRA rank 为 8 和 16，分别计算可训练参数量和相对全量更新比例。
33. 写一个 0 依赖 Python demo，同时验证归一化 dot product 等于 cosine、attention weight 行和为 1、LoRA 参数量小于全量参数量。
34. 用自己的话解释为什么下一个 token 可以看成随机变量。
35. 写出长度为 5 的 token 序列的概率链式分解，并说明 GPT 如何用 next-token prediction 建模它。
36. 给定 logits `[2.0, 1.1, 0.8, 0.1]`，手算或写代码计算 softmax、真实 token NLL 和 entropy。
37. 写一个 0 依赖 Python demo，计算 bigram 条件概率、序列 log probability、平均 NLL 和 perplexity。
38. 给一个 Bayes 公式例子，说明先验、证据似然和后验如何变化。
39. 比较 greedy、sampling、temperature、top-k 和 top-p，并说明为什么它们不能根治幻觉。
40. 给定分布 `p=[0.7,0.2,0.1]` 和 `q=[0.6,0.25,0.15]`，计算熵、交叉熵和 KL，并验证 `H(p,q)=H(p)+KL(p||q)`。
41. 写一个 0 依赖 Python demo，同时验证均匀分布熵高于尖锐分布、one-hot 交叉熵等于真实类别 NLL、`PPL=exp(avg NLL)`。
42. 构造一个 toy 联合分布，计算互信息，并解释它如何类比图文对齐或 RAG 证据质量。
43. 用自己的话解释为什么 KL penalty 只能约束 policy 不要远离 reference，不能保证每个输出都真实、安全或最优。
44. 比较两个 tokenizer 下同一段文本的 token 数变化，并解释为什么 PPL 不能脱离 tokenizer 口径比较。
45. 用纯 Python 对 `L(w)=(w-5)^2` 跑 6 步梯度下降，比较 `lr=0.1` 和 `lr=1.1` 的 loss 变化。
46. 手写 Momentum 更新，解释为什么它可能更快接近目标，也可能越过最优点。
47. 给定梯度 `[3,4,12]` 和 max norm `5`，计算 global norm、缩放系数和裁剪后的梯度。
48. 写一个 warmup + cosine decay 函数，输出 10 步学习率并解释 warmup peak 和最终 decay。
49. 给定 `micro_batch=2`、`gradient_accumulation=4`、`data_parallel=8`，计算 global batch，并解释它为什么影响 learning rate。
50. 用 0 依赖 Python demo 比较 Adam + L2 和 AdamW 首步更新，说明 decoupled weight decay 的差异。
51. 列出 checkpoint resume 后 loss spike 的优化状态排查清单。
52. 给定奇异值 `[5,2,0.5]`，计算 rank-1 和 rank-2 近似的 Frobenius 误差以及 rank-2 能量保留率。
53. 用 0 依赖 Python demo 计算一个二维 toy 数据集的协方差矩阵、特征值和第一主成分 explained variance ratio。
54. 写一个 0 依赖 Python demo，验证 `Delta W=A B` 的 rank 不超过 LoRA rank `r`。
55. 给定 `d_in=d_out=4096`、LoRA rank 为 8，计算全量参数、LoRA 参数和相对比例。
56. 解释 LoRA、低秩压缩和 QLoRA 的区别，并分别说一个失败场景。
57. 设计一个 LoRA rank 消融实验，至少包括 rank、目标模块、训练成本、验证集、人评和能力回归指标。
58. 写出真实风险、经验风险、ERM 和泛化 gap 的公式，并说明每个符号含义。
59. 给定三组分数：train 高 validation 低、train 低 validation 低、train 高 validation 高但 production 低，分别判断最可能的问题。
60. 用 0 依赖 Python 写一个泛化审计 demo，比较 memorizer、rule-based model 和 underfit baseline 的 train / validation / test accuracy。
61. 给定 checkpoint 的 train loss 与 validation loss，选择 early stopping step，并解释为什么不能用 final test set 做选择。
62. 用 3 个 seed 的 toy prediction 计算 bias^2、variance、noise 和 expected error。
63. 构造一个 distribution shift 切片，让验证集 accuracy 为 1.0、线上切片 accuracy 明显下降，并解释原因。
64. 设计一个训练-评估 exact overlap 检查，输出 overlap 样本和 overlap rate。
65. 写出 Bayes rule，并用一个“RAG 强证据提高答案可靠性后验”的例子解释 prior、likelihood、evidence 和 posterior。
66. 比较 MLE 和 MAP，说明为什么零均值高斯先验会带来类似 L2 正则的目标。
67. 用 0 依赖 Python 写一个 Beta-Bernoulli 更新 demo，输入成功 / 失败次数，输出 posterior mean 和 MAP。
68. 给定 token 分布 `[0.7,0.2,0.1]`，计算 entropy，并说明为什么低 entropy 不等于事实正确。
69. 给定 10 条 confidence / correctness 记录，计算 ECE、Brier score 和每个置信度桶的 gap。
70. 比较 aleatoric uncertainty 和 epistemic uncertainty，各举 2 个 LLM 场景和对应处理方式。
71. 设计一个 answer / verify / abstain 路由规则，至少包含 posterior reliability、risk level、evidence support 和 human review。
72. 设计一个 LLM judge 校准流程，覆盖 human gold set、position bias、length bias、切片 ECE 和高风险人工复核。
73. 把一个三 token 生成过程写成 MDP，标出 state、action、transition、reward、policy 和 trajectory。
74. 给定 rewards `[0.0,0.2,1.0]` 和 `gamma=0.9`，手算每个时间步的 discounted return。
75. 给定 returns 和 value baseline，计算 advantage，并解释正负 advantage 分别如何影响 policy 更新。
76. 给定 old probabilities `[0.5,0.4,0.25]`、new probabilities `[0.6,0.3,0.4]` 和 advantages，计算 PPO ratio、clipped surrogate 和平均 objective。
77. 给定 policy 分布和 reference 分布，计算 `KL(policy || reference)` 以及 `reward - beta * KL`。
78. 给定 chosen score 和 rejected score，计算 reward model pairwise loss。
79. 给定 policy / reference 对 chosen / rejected 的 log probability，计算 DPO margin 和 DPO loss。
80. 设计一个 RLHF 数学审计 demo，至少输出 return、advantage、PPO objective、KL penalty、RM loss、DPO loss 和检查项。
81. 给定 12 条 0/1 correctness，计算样本均值、样本方差、标准误和正态近似 95% 置信区间。
82. 给定 old/new 同题评估结果，写出 paired diff，计算 `mean(diff)`、`SE(diff)`，并判断置信区间是否跨过 0。
83. 用 0 依赖 Python 写 paired bootstrap，固定随机种子，输出新旧模型差异的 percentile CI。
84. 给定 old/new 二分类结果，统计 McNemar 的 `b` 和 `c`，计算连续性校正统计量，并解释为什么只看 discordant pairs。
85. 给定 `p=0.5`、MDE `0.05`、`z_alpha=1.96`、`z_power=0.84`，估算双比例实验每组样本量。
86. 给定 5 个 p-value，分别用 Bonferroni 和 Benjamini-Hochberg 判断哪些发现保留。
87. 设计一个 LLM A/B test 门禁，至少包含主指标、置信区间、样本量、latency、cost、safety regression 和分层结果。
87A. Treatment 组随机开放了模型建议，但 70% 的用户实际看到建议、50% 采纳；control 组主指标成功率为 54%，treatment ITT 成功率为 60%。计算 ITT，并解释为什么不能只比较采纳者和 control；若要估计实际使用效果，随机开放权限作为工具变量还需要哪些假设？
87B. 对照 QJE 的客服助手分阶段部署、三家公司的代码助手随机试验、肯尼亚基层医疗的 LLM 集群随机试验。分别列出随机/分批单位、treatment、主要结局、结果窗口、最重要的外推边界，并说明它们回答的不是同一个“LLM 提升率”。
87C. 为一项允许中途查看结果的 LLM 上线实验设计停止规则：区分事故触发回滚与统计停止，写出为何普通固定终点 p 值不能支持任意 optional stopping，并说明多主指标如何控制错误发现。
87D. 两个模型在同一批题上出现 `b=1` 道“旧对新错”和 `c=8` 道“旧错新对”。计算连续性校正 McNemar 统计量与双侧精确二项 p 值；说明为什么精确 p 值不是新模型更好的后验概率，并列出仍需检查的抽样单位与 holdout 条件。
87E. 一次评估同时比较 20 个语言/任务切片，切片共享题目和用户。说明 Bonferroni、Holm、BH 与 BY 各控制什么错误；若目标是少数预设安全门禁与大量探索性切片，分别如何安排检验家族、校正和独立复验？
87F. 一个 RAG 系统在 50 条已失败 trace 中，向模型提供核验过的目标文档后有 31 条答对。写出至少三个竞争根因假设，设计只改变单一组件的复现实验；说明 31/50 能支持什么、不能支持什么，并给出未见公司/年份的 holdout 和需要保留的中间证据字段。
87G. 对照 CheckList、The ML Test Score、HELM、EvalPlus 和 Agent trace grading：逐项说明它们补的是行为覆盖、系统工程、横向比较、程序 oracle 还是工作流可观测性；为一个 Agentic RAG 产品选择组合方案，并写出每种方法不能证明的结论。
88. 设计一个最小评估统计审计 demo，输出 accuracy、paired diff、bootstrap CI、McNemar p-value、sample size、multiple comparison 和上线决策。
89. 给第十三册 9 类数学主题建立 required topic set，并列出每类至少 2 个必须会写的公式。
90. 用 0 依赖 Python 写一个公式审计 demo，验证 CE/KL 分解、softmax gradient、LoRA 参数量、DPO loss 和 paired eval lift。
91. 设计一个数学面试回答 rubric，至少包含 formula、intuition、LLM scenario、caveat 和 demo 五项。
92. 给 6 道 mock math interview answers 打分，输出 topic coverage、formula accuracy、demo coverage、weak questions 和 revision plan。
93. 选择一题你最薄弱的数学面试题，写出“直觉 -> 公式 -> LLM 场景 -> 常见误区 -> 最小代码”的完整回答。
94. 给定 `input_ids [B,T]`、embedding table `[V,d]`、LM head `[d,V]`，推导 hidden states、logits、shift logits 和 shift labels 的 shape。
95. 写一个 PyTorch demo，打印 tensor 的 shape、dtype、device、stride、is_contiguous 和 finite 检查结果。
96. 用 PyTorch 构造 `[B,T,d] + [d]` 的 broadcasting 示例，并解释 `[B,T,1]` mask 为什么可能语义错误。
97. 用 PyTorch 比较 `matmul` 和 `einsum` 计算 attention scores，确认输出都是 `[B,H,T,T]`。
98. 构造一个 `transpose` 后 `view` 失败的例子，并用 `contiguous().view(...)` 或 `reshape(...)` 修复。
99. 用 PyTorch 构造 padding mask `[B,T]` 和 causal mask `[T,T]`，合成 attention mask `[B,1,T,T]`。
100. 设计一个最小 PyTorch tensor audit demo，至少输出 dtype/device、broadcast shape、attention score shape、mask shape、view 是否失败、LM loss flatten 是否对齐。
101. 用 PyTorch 对 `y=x**3+2*x` 在 `x=2` 时调用 backward，验证梯度为 14。
102. 构造向量输出 `y=x**2`，给 `backward()` 传入外部梯度，解释 vector-Jacobian product。
103. 构造 leaf tensor 和 non-leaf tensor，观察 `.grad`，再用 `retain_grad()` 保存中间 tensor 梯度。
104. 连续两次 backward 展示梯度累积，再用 `grad=None` 或 `optimizer.zero_grad(set_to_none=True)` 清空。
105. 写一个误用 `detach()` 导致某个分支不反传的例子，并解释哪些参数会有梯度。
106. 比较 `model.eval()`、`torch.no_grad()` 和 `torch.inference_mode()` 的作用差异。
107. 构造一个 in-place 操作导致 backward 风险的例子，并说明如何改成非 in-place 写法。
108. 写一个最小 autograd audit demo，输出 scalar grad、VJP grad、leaf/non-leaf grad、grad accumulation、detach branch、no_grad validation 和 missing grad 参数。
109. 写一个 `ResidualMLP`，包含 Linear、LayerNorm、GELU、Dropout 和 residual connection，并确认所有可训练层都在 `__init__` 中创建。
110. 分别用普通 list 和 `nn.ModuleList` 堆叠 2 个子模块，打印 `named_parameters()`，解释为什么普通 list 中的层没有注册。
111. 定义一个 `nn.Parameter` 和一个普通 `requires_grad=True` tensor，比较它们是否出现在 `model.named_parameters()` 和 `state_dict()` 中。
112. 用 `register_buffer` 注册固定 mask 或统计量，再设置一个 `persistent=False` buffer，比较两者在 `named_buffers()` 和 `state_dict()` 中的差异。
113. 保存一个模型的 `state_dict`，重新构造同结构模型加载，并在 `eval()` + `inference_mode()` 下比较同一输入 logits 是否一致。
114. 删除 checkpoint 中分类头的 key，用 `load_state_dict(strict=False)` 加载，打印 missing / unexpected keys，并判断这些 key 是否符合预期。
115. 冻结 encoder，只训练 head，重新创建 optimizer，并核对 optimizer 参数量是否等于 `requires_grad=True` 的参数量。
116. 写一个最小 Module audit demo，输出普通 list 注册失败、ModuleList 注册成功、Parameter / buffer / state_dict 检查、train/eval 递归状态、冻结后 optimizer 参数量和保存加载一致性。
117. 写一个 map-style `Dataset`，实现 `__len__` 和 `__getitem__`，返回变长 `input_ids`、`labels` 和 `length`。
118. 写一个 `collate_fn`，把变长 causal LM 样本 pad 成 `[B,T]`，同时构造 `attention_mask`，并把 padding label 设为 `-100`。
119. 给定 batch 内样本长度 `[2,6,3]`，计算 `T_b`、valid token ratio 和 padding waste。
120. 用 PyTorch `DataLoader` 固定 `torch.Generator` seed，验证 shuffle 后第一个 batch 的 shape 和样本长度可复现。
121. 写一个 length bucket 小函数，比较 bucket 前后 padding waste 的变化。
122. 用 `DistributedSampler(num_replicas=2, rank=0/1)` 模拟两卡索引切分，验证两个 rank 的索引集合是否重叠。
123. 解释 `num_workers`、`pin_memory`、`persistent_workers` 和 `prefetch_factor` 分别影响什么，以及为什么不是越大越好。
124. 写一个最小 DataLoader audit demo，输出 batch shape、attention mask、ignore index 检查、shift logits / labels shape、padding waste 和 distributed overlap。
125. 写一个最小 causal LM training step，包含 shift logits、shift labels、`ignore_index=-100` 和 `optimizer.step()`。
126. 给定 `accum_steps=4`，解释为什么每个 micro-step 的 loss 要除以 4，并写出平均梯度公式。
127. 写一个 PyTorch demo，比较 scheduler 每个 micro-step 更新和每个 optimizer step 更新时的学习率序列差异。
128. 给训练循环加入 `clip_grad_norm_`，打印裁剪前总范数，并说明它应该放在 backward 之后、optimizer step 之前。
129. 写一个 token-weighted validation loop，使用 `model.eval()` 和 `torch.inference_mode()`，验证结束后恢复原训练模式。
130. 保存完整 checkpoint，字段至少包含 model、optimizer、scheduler、global_step、config 和 RNG state。
131. 从 checkpoint 恢复模型、optimizer 和 scheduler，验证恢复前后 validation loss 是否一致。
132. 写一个最小 Training Loop audit demo，输出 raw loss、grad norm、optimizer step 数、scheduler step 数、final lr、validation loss、checkpoint resume 一致性和非有限 loss 检查。
133. 给定 100 万参数和 BF16 参数 / 梯度、FP32 AdamW 一阶矩 / 二阶矩，估算 params、grads、optimizer states 各占多少 MiB。
134. 写出训练显存粗略分解公式，至少包含参数、梯度、optimizer state、activation 和临时 buffer。
135. 比较 FP16 和 BF16 的指数范围、尾数精度、数值稳定性和硬件依赖。
136. 写一个 CPU 可运行的 `torch.amp.autocast("cpu", dtype=torch.bfloat16)` demo，打印 autocast 后线性层输出 dtype。
137. 用 `torch.amp.GradScaler` 写一个最小 loss scaling / unscale / clip / step demo，并解释为什么裁剪前要先 unscale。
138. 写一个 activation checkpointing demo，统计同一个 block 在不用 checkpoint 和使用 checkpoint 时 forward 调用次数。
139. 设计一个 OOM audit checklist，覆盖训练 / 验证阶段、batch size、sequence length、AMP、gradient accumulation、activation checkpointing、optimizer state 和带图 tensor 缓存。
140. 写一个最小 AMP memory audit demo，输出 dtype 显存估算、autocast dtype、GradScaler scale、checkpoint recompute 和 `torch.cuda.is_available()`。
141. 写出 `rank`、`local_rank` 和 `world_size` 的区别，并说明它们在 `torchrun` 脚本中分别用于什么。
142. 给定 `per_device_batch_size=2`、`world_size=8`、`accum_steps=4`，计算 global batch size，并解释学习率和 warmup 为什么可能需要调整。
143. 用纯 Python 模拟 4 个 rank 的数据索引切分，验证 rank 间索引集合没有 overlap，且覆盖完整 epoch 样本。
144. 给定 4 个 rank 的本地梯度 `[0.2,0.4,-0.1,0.3]`，计算 all-reduce 平均梯度，并比较同步更新和各自本地更新后参数是否会漂移。
145. 解释 DDP backward 中 gradient bucket 和 communication overlap 的直觉。
146. 写一个梯度累积下 `no_sync()` 的伪代码，并计算 6 个 micro-step、`accum_steps=3` 时同步次数从多少降到多少。
147. 设计一个分布式训练 deadlock audit 表，至少覆盖缺失 collective、rank 条件分支、不同 batch 数、rank 0 文件依赖和未使用参数。
148. 写一个最小 Distributed Training audit demo，输出 global batch、rank 数据切分、all-reduce 平均梯度、参数一致性、`no_sync` 同步次数、缺失 collective rank 和 rank 0 checkpoint 写入者。
148A. 写一个 0 依赖 Python 分布式事故审计 demo，输入 4 个 toy rank 的 step、phase、token 数、step/data/comm time、collective 序列、checkpoint shard、resume step / lr 和 pipeline stage 耗时，输出 rank step alignment、collective mismatch、token shard imbalance、straggler ratio、communication ratio、checkpoint shard coverage、resume continuity、pipeline bubble ratio、failed gates 和 gate pass 结论。
148B. TorchTitan 的一个 8×H100 教学案例中，Llama3 8B（batch=1、sequence=8192、defer=8）把 offload budget 从 10 GB 提到 12 GB，额外释放约 0.46 GiB、tokens/s 减少 60—70。计算区间平均每 GiB 的吞吐代价；再比较保留、重计算和 CPU 卸载各消耗的资源，并设计需要固定的 token、模型、NUMA、峰值显存和 loss 测量条件。解释为什么这个拐点不能外推成通用配置。
148C. 一个 8-rank mesh 设 DP=2、TP=4。权重 `W` 的全局形状为 `[16,64]`，沿输出维做 TP 切分、再沿行做 DP/FSDP 切分，dtype 为 BF16。逐步计算全局权重、每 rank 常驻分片和 FSDP all-gather 后的本地 TP 权重元素数与字节数。说明 DTensor 持久化表示和 SPMD Types 计算表示改变了什么，又有哪些张量形状、collective 和 checkpoint 性质仍需一致。
149. 写一个 `debug_tensor` 函数，输出 tensor 的 shape、dtype、device、requires_grad 和 contiguous 状态。
150. 给定 logits `[B,T,V]` 和 labels `[B,T]`，写出 causal LM shift 后展平维度，并验证 logits / labels 的第 0 维是否一致。
151. 写一个 `finite_status` 函数，检查输入、logits、loss 和 grad 是否包含 NaN / Inf。
152. 构造一个 forward hook，记录某层输出 shape；运行一次 forward 后调用 `remove()`，并验证 hook 已移除。
153. 打印所有 `requires_grad=True` 参数的 grad norm，并检查这些参数是否都进入 optimizer。
154. 设计一个 OOM audit 表，区分峰值过高和 step 后显存持续增长两类问题。
155. 写一个 step timing demo，分别说明 CPU 计时和 CUDA 计时为什么不同，CUDA 计时为什么需要 `torch.cuda.synchronize()`。
156. 写一个最小 Debug / Profiling audit demo，输出 tensor metadata、valid label 数、loss、activation shape、grad norm、non-finite report、step timing 和 profiler 可选入口。
157. 写一个 `TransformerConfig`，并检查 `hidden_size % num_heads == 0`。
158. 用 PyTorch 实现 token embedding 和 LM head，验证 `input_ids [B,T] -> hidden [B,T,d] -> logits [B,T,V]`。
159. 手写 RMSNorm，并验证输出 shape 不变、最后一维均方根接近 1。
160. 写一个支持 `past_key_values_length` 的 causal mask，分别验证 prefill 和 decode 场景。
161. 手写 scaled dot-product attention，输出 context shape、attention row sum 和 future weight max。
162. 手写 Multi-Head Self-Attention，打印 Q/K/V reshape、score、context 和合并后的 shape。
163. 实现 SwiGLU MLP 和 Pre-Norm DecoderBlock，确认 residual 相加前后 shape 一致。
164. 实现 RoPE 的 `build_rope_cache` 和 `apply_rope`，验证旋转后范数保持。
165. 写一个最小 decoder-only LM，验证 shift logits / labels、`ignore_index=-100` 和 weight tying。
166. 写一个最小 Transformer component audit demo，输出 shape、mask、RoPE、loss、cache length 和 gate pass。
167. 给第十四册工程面试建立 required topic set，至少包含 tensor、autograd、Module、DataLoader、training loop、AMP、DDP、debug/profiling 和 Transformer components。
168. 写出 causal LM shift rows、global batch size、training memory、DDP average gradient、attention score shape 和 KV Cache memory 六个公式。
169. 用 0 依赖 Python 写一个 PyTorch engineering interview readiness demo，输出 topic coverage、missing topics、formula checks、red flags、revision plan 和 gate pass。
170. 对“loss 不下降”“NaN/Inf”“OOM”“GPU 利用率低”四类问题分别写出 5 步排查顺序。
171. 录一段 3 分钟 PyTorch 工程面试回答，按 mechanism、shape/formula、pitfall、debug path、demo evidence 五项自评。
172. 选择一个最薄弱的 PyTorch 工程主题，补一个最小 demo，并说明它如何支持面试回答。
173. 写一个 0 依赖 Python 训练稳定性审计 demo，输入 toy step 日志，输出 first bad step、loss spike steps、non-finite steps、grad explosion steps、AMP overflow steps、有效 label token 异常、LR resume jump、rank loss skew、failed gates 和 gate pass 结论。
174. 对照 Molybog 的 Adam instability 与 adaptive-preconditioner decoupling 研究，画出各自的因果链、观测量和证据边界，并设计一次固定 checkpoint 的反事实重放。
175. 为 global norm clipping、降低 Adam β₂、SPAM 和 GradientStabilizer 设计同口径对照；报告机制、状态改动、额外超参、吞吐、目标质量和失败条件。
176. 用逐层 activation RMS、gradient RMS、残差流 RMS 和 update/parameter ratio 比较 Post-LN、Pre-LN、DeepNorm、Peri-LN 的训练信号；说明为何架构改动应从初始化开始验证。
177. 为两个 checkpoint 冻结完整 evaluation contract，至少记录 tokenizer/template、task/split、prompt/few-shot、generation/scorer 和 harness revision；对照一次只变 template 的小规模复测，保存逐样本输出并归因分数变化。
178. 构造两个模型在同一批 100 道题上的 paired outcomes，统计两边都对/都错和各自独有正确的样本数；说明为何独立准确率看不出 discordant pairs，并比较 McNemar、paired bootstrap 与多训练 seed 分别估计什么。
179. 分别为可访问训练语料的开源模型和不可访问语料的闭源模型设计 benchmark contamination 审计；组合 overlap scan、新写/隐藏留出集、样本输出检查以及 Oren 顺序似然检验或 ConStat 泛化差距，列出访问条件、误报与漏报边界。
180. 用固定 loss cap 和权重实现多指标 checkpoint 选择 demo：先应用 stability、resume completeness、safety 与 domain 硬门槛，再归一化 validation、benchmark、safety 和 regression 分量；测试无合格候选、权重和不为 1、配置字段缺失及 benchmark 键不一致。

## 阶段 2：Transformer 验收

1. 用自己的话解释 token、vocabulary、token id。
2. 比较 word-level、character-level、subword tokenization 的优缺点。
3. 手动模拟一次 BPE 合并过程。
4. 解释为什么中文 tokenizer 压缩率会影响上下文长度。
5. 解释新增 `<tool>` token 后需要修改哪些模型组件。
6. 假设 `vocab_size=50000`，`hidden_size=4096`，计算 token embedding 参数量。
7. 写一个 PyTorch `nn.Embedding` 示例，输入 `[B, T]`，输出 `[B, T, d]`。
8. 解释为什么 token id 不能直接当连续数值输入。
9. 举例说明为什么顺序对语言理解重要。
10. 用自己的话解释 token embedding 和 position embedding 的区别。
11. 用自己的话解释 self-attention。
12. 举一个例子说明同一个 token 在不同上下文中含义不同。
13. 用搜索引擎类比解释 Q、K、V。
14. 比较 RNN 和 self-attention 在长距离依赖上的差异。
15. 解释为什么标准 attention 的复杂度是 `O(T^2)`。
16. 推导 `Q [B,T,d]` 和 `K [B,T,d]` 相乘后 scores 的 shape。
17. 用 PyTorch 实现 causal scaled dot-product attention。
18. 解释为什么 attention scores 需要 mask 后再 softmax。
19. 比较 full attention、local attention、linear attention 的优缺点。
20. 用自己的话解释 FlashAttention 为什么是系统优化而不是建模近似。
21. 手写 self-attention。
22. 实现 causal mask。
23. 解释 MHA、MQA、GQA。
24. 假设 `d_model=4096`，`num_heads=32`，计算 `head_dim`。
25. 推导 MHA 中 scores 的 shape。
26. 估算忽略 bias 时 MHA 的参数量。
27. 比较 MHA、MQA、GQA 对 KV Cache 的影响。
28. 修改 MHA PyTorch 代码，加入 causal mask。
29. 写出长度为 5 的 causal mask。
30. 用 PyTorch 生成 `[1,1,T,T]` 形状的 causal mask。
31. 解释为什么 mask 要在 softmax 前加入。
31A. 用自己的话解释 QK circuit 和 OV circuit 的区别，并各举一个可能失败的例子。
31B. 运行第 21 册第 3 章的 QKV / Head 子空间审计 demo，解释 head redundancy 和 KV head ratio 分别对应什么工程问题。
31C. 运行第 21 册第 4 章的 MHA / GQA / MQA / MLA 成本审计 demo，解释为什么 cache 最省不等于默认最适合上线。
31D. 运行第 21 册第 5 章的 FFN / MoE 成本审计 demo，解释 dense 4D FFN、SwiGLU 近似等参配置、MoE total parameters 和 active parameters 的差异。
31E. 给定 12 个 token、8 个 experts、top-2 routing 和 capacity factor 1.25，手算 balanced routing 与 collapsed routing 的 expert counts、capacity、overflow 和 load ratio。
31F. 运行第 21 册第 6 章的 Residual / Norm 稳定性审计 demo，解释 `plain_no_residual`、`residual_no_norm`、`post_layernorm` 和 `pre_rmsnorm_scaled` 的 gate 结果。
31G. 用自己的话写出 Pre-LN、Post-LN、LayerNorm 和 RMSNorm 的公式，并说明 warmup 降低 early update pressure 的直觉。
31H. 运行第 21 册第 7 章的 Position Encoding 审计 demo，解释为什么 `score_2_5` 和 `score_10_13` 相等、`key_2` 的 ALiBi bias 更负，以及 position gate 为什么失败。
31I. 给定 `cache_len=128`、新 token position ids `[0,1,2]`、训练上下文 4096、目标上下文 32768，写出 position id audit 应该报出的风险，并说明如何修正。
31J. 运行第 21 册第 8 章的 Mask 可见性审计 demo，解释 `prefix_counts`、`plain_causal_packing_leak`、`block_diag_packing_leak` 和 `mask_gate_pass` 分别说明什么。
31K. 手画 prefix 长度为 2、target 长度为 4 的 Prefix LM mask，并说明哪些位置如果错误可见会造成未来泄漏。
31L. 运行第 21 册第 9 章的并行与 Scaling 审计 demo，解释 `arch_cost`、`loss_best`、`ready_best` 和 `scaling_gate_pass` 分别说明什么。
31M. 给定固定训练 FLOPs 预算，列出 3 个不同参数量 / token 数候选，并判断哪个候选可能 undertrained、哪个 serving pressure 最高。
31N. 运行第 21 册第 10 章的 Attention 成本审计 demo，解释 `short_vs_long_pair_multiplier`、`full_32k_score_tensor_gib`、`prefill_32k_tflops_per_layer`、`decode_one_step_32k_tflops_per_layer` 和 `attention_cost_gate_pass` 分别说明什么。
31O. 给定 `B=1`、`H=32`、`T=32768`、`D_h=128` 和 fp16，手算 full score tensor、score+prob tensor、causal visible score tensor 的显存，并解释为什么 causal mask 不能根治平方瓶颈。
31P. 运行第 21 册第 11 章的 KV Cache 成本审计 demo，解释 `mha_8k_cache_gib`、`gqa_8k_cache_gib`、`gqa_cache_per_token_kib`、`decode_read_gib_per_step`、`paged_waste_ratio` 和 `kv_cache_gate_pass` 分别说明什么。
31Q. 给定 `L=32`、`B=16`、`T=8192`、`H_q=32`、`H_kv=8`、`D_h=128`、fp16，手算 GQA KV cache 显存，并说明把 `H_kv` 降到 4 可能带来哪些质量和工程风险。
32. 比较 causal mask 和 padding mask。
33. 用自己的话解释 GPT 和 BERT 的训练目标区别。
34. 画出 decoder-only Transformer block 的数据流。
35. 写出 Pre-LN block 的两行核心公式。
36. 解释 attention 和 MLP 的分工。
37. 比较 GPT block、BERT block、seq2seq decoder block。
38. 修改 Transformer block 代码，把 GeLU 换成 SiLU，并观察输出 shape 是否变化。
39. 对一个长度为 4 的向量手算 LayerNorm。
40. 对同一个向量手算 RMSNorm。
41. 比较 LayerNorm 和 BatchNorm 的统计维度。
42. 解释为什么 norm 参数通常不做 weight decay。
43. 写出 Pre-LN Transformer block 的公式。
44. 用自己的话解释 RoPE 为什么作用在 Q/K 上。
45. 写一个简化函数，对二维向量做旋转。
46. 解释为什么标准 attention 的长上下文成本高。
47. 比较 RoPE scaling 和长上下文继续训练的优缺点。
48. 设计一个长上下文评估集，至少包含 3 类任务。
49. 给定 token 序列 `[1,2,3,4,5]`，构造 block_size=4 的 input 和 target。
50. 修改 miniGPT 的 `generate`，实现 greedy decoding。
51. 给 miniGPT 加 dropout。
52. 设计把 position embedding 替换成 RoPE 的接口。
53. 打印 miniGPT 每一层输出 shape，确认数据流。
54. 训练一个 miniGPT。
54A. 写一个 0 依赖 Transformer 架构选择审计 demo，输入 toy architectures 的 route steps、training parallelism、content routing、next-token prediction alignment、scaling evidence、ecosystem readiness、hardware fit、ICL support、long-context efficiency、KV efficiency 和 streaming state，输出 ranked architectures、foundation score、deployment pressure、failed gates 和 weakest architecture；要求解释为什么 Transformer 综合过线但仍有长上下文和 KV cache 压力。
54B. 写一个 0 依赖 Attention 路由审计 demo，输入 toy cases 的 query、keys、visible mask、expected route、interpretation support，输出每个 case 的 top route、route hit、future mass、raw entropy、scaled entropy、row sum、route hit rate、future leak rate、avg entropy gain、interpretation risk rate、failed cases、failed gates 和 attention routing gate；要求至少构造一个 causal mask 正确但 route miss 的样本。
54C. 写一个 0 依赖 ICL / token retrieval 审计 demo，输入 few-shot 示例、query、expected label、task labels、示例位置和噪声标记，输出 label space coverage、format consistency、top example、predicted label、retrieval label match、middle relevant examples、conflicting relevant examples、middle evidence risk、conflict rate、failed gates 和 ICL gate；要求至少构造一个相关示例在中间位置的样本和一个近尾冲突示例覆盖正确标签的样本。

## 阶段 3：训练与对齐验收

1. 画出 LLM 预训练从数据到 checkpoint 的流程图。
2. 讲清楚预训练 pipeline。
3. 列出 5 类预训练数据来源。
4. 列出数据清洗中至少 5 个要处理的问题。
5. 解释为什么 checkpoint 要保存 optimizer state。
6. 用自己的话解释 base model 和 instruct model 的区别。
7. 用自己的话解释 scaling law。
8. 解释为什么固定 compute 下不能只增大模型参数量。
9. 设计一个数据 mixture ablation 实验。
10. 列出 5 种低质量数据对模型的负面影响。
11. 解释 benchmark contamination 为什么会让评估失真。
12. 列出训练显存的 5 个主要组成部分。
13. 画出 DDP 的梯度同步流程。
14. 比较 ZeRO-1、ZeRO-2、ZeRO-3。
14A. rank 0 有 100 个有效 token、mean loss 为 1，rank 1 有 10 个有效 token、mean loss 为 2。比较 rank mean 与全局 token mean，并推导 DDP 默认按 rank 平均梯度时每个 rank 应怎样缩放本地 loss sum。
15. 用自己的话解释 Tensor Parallel 和 Pipeline Parallel。
16. 思考为什么通信开销会限制多 GPU 训练加速。
17. 用自己的话解释 base model 和 instruct model 的区别。
18. 写一个单轮 instruction tuning 样本。
19. 写一个多轮 chat template 示例。
20. 解释为什么 user token 不应该参与 SFT loss。
21. 列出 5 类高质量指令数据应覆盖的任务。
22. 写一个 `messages` 格式的 SFT 样本。
23. 给定一段 user/assistant token 序列，手写 labels 和 `-100` mask。
24. 比较全参 SFT、LoRA、QLoRA 的优缺点。
25. 列出 SFT 数据清洗的 5 个检查项。
26. 设计一个 SFT 评估表，覆盖指令遵循、代码、数学、安全和多轮对话。
27. 画出 RLHF 从 SFT model 到 policy model 的完整流程。
28. 写一个 chosen/rejected 偏好数据样本。
29. 用自己的话解释 Reward Model 学的是什么。
30. 解释为什么 PPO 训练中需要 reference model。
31. 列出 5 种 reward hacking 可能表现。
32. 写一个 DPO 使用的 prompt/chosen/rejected 样本。
33. 用自己的话解释 DPO 和 RLHF 的区别。
34. 解释 reference model 在 DPO 中的作用。
35. 思考 chosen 比 rejected 更长时可能带来什么问题。
36. 设计一个 DPO 后的评估表。
37. 写一个 reward model 训练用的 chosen/rejected 样本。
38. 用自己的话解释 pairwise ranking loss。
39. 举 3 个 reward hacking 的具体例子。
40. 设计一个 Reward Model 评估表，包含整体和分领域指标。
41. 分析 over-refusal 为什么会损害用户体验。
42. 实现 SFT label mask。
43. 手写 DPO loss。
43A. 写一个 0 依赖 Python SFT / 对齐训练事故审计 demo，输入 toy SFT 样本的 role token 数和 label 数、base / after 各能力分数、误拒 / 漏拒案例、chosen/rejected 偏好 pair、reward 样本、DPO policy / reference log probability 和工具 schema，输出 assistant mask coverage、prompt loss leak、PAD loss leak、EOS coverage、capability regression、over-refusal rate、unsafe leak rate、preference margin、reward-human gap、reward length bias、DPO negative margins、tool schema drift、failed gates 和 gate pass 结论。
43B. 比较 DPO、KTO、ORPO 和 SimPO，要求写出每种方法是否需要 reference model、是否需要成对偏好、最容易出现的偏差和上线前必须看的回归指标。
43C. 设计一个 RLVR / DeepSeek-R1 风格 reasoning 后训练审计表，字段包含任务是否可验证、reward 来源、rollout 数、采样策略、错误答案处理、过程质量、人审切片、安全回归、蒸馏目标和污染标记。
43D. 同一 prompt 的 reward 为 [1, 0]，response 长度为 20 和 100，generation budget 为 100。分别计算 group-standardized GRPO advantage 与只中心化的 DrGRPO advantage；对照逐 response 长度 mean 和固定预算分母时两条回答的总 token 权重。再构造全同 reward group，说明 DrGRPO 为什么仍没有相对信号，DAPO dynamic sampling 和 DCPO 的历史 SAS 分别改变什么。
43E. 设计 GRPO、DAPO、DrGRPO、GSPO、DCPO 的同条件消融表，固定 base checkpoint、题目/模板、verifier、reward、训练 token、rollout 数、最大长度、采样参数和硬件。至少报告正确率及 Avg@1/Avg@32、长度切片、非零 advantage/RUR、token clipping rate、保留/过滤 group、wall-clock、单位成功成本、通用能力和安全回归；解释哪些指标仍然不可互换。
43F. 为一个 RLVR 代码修复任务设计 verifier 红队评估：给出已知正确/错误 patch、可被变形的输入、与规格冲突的测试、可注入的 mutation 和独立 artifact grader。比较固定样例通过率、变形一致率、错误 mutation 检出率和独立正确率；再在相同底模、数据、训练预算与随机种子下增加随机/打乱 reward 对照，说明哪些结果可以归因于 verifier，哪些只能归因于训练流程或底模先验。
43G. 三条同任务轨迹到达同一个环境状态：动作 A 两步后成功，动作 B 一步后成功，动作 C 失败。取 `gamma=0.9`，计算各自 discounted return 和在 `F_norm=1` 时的 GiGPO step-level relative advantage；说明它还要与 episode-level advantage 组合。再构造一个遗漏关键环境字段的 state key，解释为何错误合并状态会污染 credit。
44. 分析一次训练异常。
45. 用自己的话解释 AI Safety 和 Alignment 的区别，要求覆盖目标、风险、训练、评估和部署。
46. 给 8 条 toy 请求设计 risk taxonomy，至少包含 harmful content、privacy、jailbreak、prompt injection、tool misuse、high risk domain 和 benign help；再把 misuse、malfunction、systemic risk 作为影响来源/范围的另一条分类轴，说明一条轨迹为什么可以同时拥有多种标签。
47. 写一个 0 依赖 Python demo，计算 unsafe compliance、refusal accuracy、over-refusal、attack success、unauthorized tool call、safe completion quality 和 severity-weighted risk。
48. 设计一个 safety gate，说明哪些指标是硬门禁，哪些指标可以作为灰度观察项。
49. 选 3 个 helpful、honest、harmless 冲突案例，写出模型应该拒绝、澄清、部分回答或安全替代的理由。
50. 为一个带工具调用的助手设计安全审计表，字段包含工具权限、参数校验、二次确认、审计日志、回滚和人工接管。
51. 用一个客服模型例子解释 Alignment Problem，要求区分真实意图、目标规范、训练 proxy、模型实际行为和部署风险。
52. 比较 outer alignment 和 inner alignment，要求各举 2 个 LLM 场景例子。
53. 设计一个 goal misgeneralization 评估，让训练分布中的两个目标在测试分布中分离。
54. 列出 5 个 specification gaming 例子，至少覆盖 reward model、benchmark、judge、引用和安全关键词。
54A. 对照 HarmBench、StrongREJECT、JailbreakBench：分别说明它们怎样构造行为、怎样判定回答、怎样固定攻击成本，以及各自没有证明什么。
54B. 为一个只读日历摘要 Agent 设计抽象的间接注入评估，分别定义 benign utility、utility under attack、targeted ASR、工具状态断言和零攻击样本时的报告状态；不要写可复用攻击载荷。
54C. 将四种隐私事件分别归入训练数据抽取、成员推断、属性推断或应用权限越权：说明攻击者目标、可观察证据、适用指标和不能推出的结论。
54D. 某固定测试集做了 100 次独立有效尝试，观察到 0 次失败。用 95% 单侧零事件上界计算失败率上界；再讨论样本相关或没有有效尝试时为什么不能照搬该结果。
54E. 为一条安全策略设计一个 MFT、一个语义保持的 INV 和一个 DIR。说明由谁确认语义关系，并分别检查 clean correctness、perturbed correctness 和 action consistency。
55. 写一个 0 依赖 Python demo，计算 outer mismatch、behavior mismatch、proxy follow、Goodhart gap、goal misgeneralization 和 alignment gate。
56. 用 3 分钟回答 deceptive alignment，要求说明它是潜在研究风险、需要什么证据、为什么不能当作当前事实断言。
57. 用自己的话解释 Scalable Oversight，要求覆盖人类监督瓶颈、AI feedback、verifier、人审和上线门禁。
58. 比较 Debate、Iterated Amplification、Recursive Reward Modeling 和 Constitutional AI 的核心假设与失败模式。
59. 为一个 RAG 长文档问答系统设计 AI-assisted evaluation 流程，字段包含 claim、evidence、citation、judge、human audit 和 verdict。
60. 为一个 coding agent 设计 scalable oversight 表，字段包含 tool trace、unit test、static check、LLM review、permission gate 和 human escalation。
61. 写一个 0 依赖 Python demo，计算 direct coverage、AI feedback accuracy、verifier coverage、process step accuracy、evidence support、high-risk audit coverage、cost saving 和 oversight gate。
62. 用 3 分钟回答“为什么 AI feedback 不能无审计替代 human feedback”。
63. 用考试刷分例子解释 reward hacking、Goodhart 定律和 proxy objective 的关系。
64. 设计一个 reward overoptimization 实验，比较不同优化强度下的 proxy reward、human eval、输出长度和失败切片。
65. 写一个 0 依赖 Python demo，计算 proxy mismatch、reward hacking rate、reward-human gap、length bias、high-reward-low-quality ratio 和 reward hacking gate。
66. 解释为什么 best-of-N、LLM judge reranking 和自动数据筛选也可能放大 reward model 漏洞。
67. 为一个 RAG 系统设计 reward hacking 防护表，覆盖 citation presence、citation accuracy、unsupported claim、faithfulness、人工抽检和回归集。
68. 用 3 分钟回答“为什么 KL penalty 有用但不能根治 reward hacking”。
69. 写一个 0 依赖 Python tokenizer / 格式事故审计 demo，输入一条 system/user/assistant 多轮样本，输出 token ids、decode 前缀、assistant-only label 覆盖率、prompt loss 泄漏率、PAD loss 泄漏率、EOS 是否参与 label、训练 / 推理 template 是否一致、截断是否丢关键边界、多模态 placeholder 是否匹配和 gate pass 结论。

### 训练成本：最优点、扩展效率与能耗账本

配套正文：[第五册第十三章](book-05-llm-training/chapters/13-训练成本与资源规划.md)。

1. 对照 Kaplan、Chinchilla 和 inference-aware scaling 的优化目标、代表性方法和限制；给定服务请求量，列出选择模型规模前仍需采集的质量、长度、硬件和价格变量。
2. 某训练作业需要 `1T` 有效 token，deadline 60 天、非训练时间 20%，单卡长期吞吐 4,000 token/s，128 卡扩展效率 0.55。复算所需吞吐、集群吞吐和剩余裕量；再说明将 GPU 数翻倍时应实测哪些通信与收敛指标。
3. 计算 0.4 kW 平均设备功率、100,000 GPU-hours、PUE 1.2、电网碳强度 0.5 kgCO₂e/kWh 的设施用电与运行排放；分别列出 TDP、设施电表、PUE、电网因子和 embodied carbon 口径。
4. 从一个大规模训练案例中分别记录 MFU/吞吐、长程故障次数、自动恢复比例和有效时间率；说明为何单步 benchmark 不足以证明数周训练的可用率。
5. 比较 1 小时与 4 小时 checkpoint 间隔的平均丢失窗口、写入开销和恢复窗口；列出要从目标集群测量的保存时长、带宽、故障分布与数据游标状态。

### On-Policy Distillation：目标、时间信用与行业证据

配套正文：[第五册第十七章](book-05-llm-training/chapters/17-on-policy-distillation.md)。

1. 令学生分布为 `p_S=(0.5,0.5)`、教师分布为 `p_T=(0.8,0.2)`。计算两个动作的 `log p_T-log p_S`、学生期望、reverse KL 以及以动作无关 baseline 校正后的优势；说明为什么逐样本 `-r` 不能直接当监督标签，且教师偏好不等于任务真值。
2. 对照 MiniLLM 与 GKD：画出状态来源、监督目标和梯度路径，指出哪个方法显式处理未来 token 对早期动作的信用，哪个方法把学生 rollout 比例与条件 divergence 分开；列出一项公平比较必须固定的变量。
3. 读 Qwen3 Technical Report 表 21，计算 RL 与 OPD 的 AIME'24、AIME'25、LiveCodeBench v5 分差，以及 GPU-hours 比例。分别写一条“数据直接支持的结论”和一条“不能从该表推出的结论”，再设计多 seed 的同起点复现协议。
4. 为 REOPOLD、EOPD、vOPD、RG-OPD、MOPD 设计故障到机制诊断表：每个方法对应一个待解决症状、改变的数学/采样环节、作者报告的证据、可能的新偏差或成本，以及一个能证伪该方法有效性的消融。
5. 依据 TRL DistillationTrainer/GKDTrainer 动态文档做版本化实现审计，记录 rollout 来源、`lmbda`/`beta` 的代码语义、loss 方向与 reduction、teacher/student tokenizer、生成和 teacher compute、tool token mask。没有实际运行时不得填造性能数值。

### 训练系统综合设计与故障诊断

配套正文：[第五册第十四章](book-05-llm-training/chapters/14-训练系统综合设计与故障诊断.md)。

1. 设计一个 7B 训练账本，明确目标、数据版本、有效 token、模型、优化器、并行拓扑、评测协议、风险状态和证据。让一个配置故意缺失评估或污染状态，说明审计为什么应失败。
2. 对照默认 mixture 与 DoReMi 风格 proxy mixture。固定总 token、模型和评测；记录域权重变化、代理成本、逐域 loss、下游任务和误差范围，并说明 proxy 权重为何不能代替许可、质量和污染门禁。
3. 为一条 loss spike 设计可证伪排查：从异常前 checkpoint 重放固定 batch，区分数据/mask、数值/kernel、optimizer state 和恢复游标；分别列出能支持与不能支持的结论。
4. 对一个 benchmark 同时做语料近似匹配、PaCoST 类改写题置信度检查和一组新鲜隐藏题。分开记录覆盖率、候选命中、人工确认、行为信号和新题结果，解释为什么它们不能折成一个污染率。

### 领域专家 SFT 与 RL：适配信号、行业证据与评估

配套正文：[第五册第十五章](book-05-llm-training/chapters/15-领域专家sft与rl.md)。

1. 对术语 loss 高、答案格式错、财报版本过期、专家偏好不一致、数学答案错误和越权工具调用六种故障，分别选择 continued pretraining、RAG、SFT、偏好优化、RLVR 或独立 policy；说明为什么其余信号不能直接解决该故障。
2. 对照 FinGPT 的 LoRA/市场弱标签、TIMER 的时间戳指令 SFT 和 DeepSeekMath 的可验证奖励 RL。逐项写出研究问题、监督信号、指标、改善证据、混杂/失败条件，并说明哪些论文结果不能横向比较。
3. 为纵向医疗记录设计 Recent-Events、Timeline-Extremes 和 Full-Timeline 三种训练切分；固定底模、样本数与 token budget，按病人隔离训练和测试，报告 correctness、completeness、关键事件召回、时间窗错误和亚群指标。
4. 评审一条“新闻后股价上涨，因此新闻情感为正”的标签链。画出新闻、市场事件、价格窗口、交易成本和输出标签之间可能的混杂，设计时间切分、行业/市场基线、负控制以及专家标签对照。
5. 依据 FinBen 或 MedAlign 做一页评测计划：区分基准覆盖、模型能力、训练收益和部署安全；列出任务分桶、来源版本、时间/实体泄漏检查、自动 verifier、专家抽检及未覆盖项。

## 阶段 4：部署与系统验收

1. 解释 prefill/decode。
2. 估算 KV Cache 显存。
3. 部署一个小模型服务。
4. 设计一个 RAG 系统。
5. 用自己的话解释 greedy decoding 和 sampling 的区别。
6. 手算一个 temperature 降低后分布变尖锐的例子。
7. 给定 token 概率，分别选出 top-k 和 top-p 候选集合。
7A. 对 `p=(0.40,0.30,0.20,0.10)` 手算 entropy 和 locally typical 的 surprisal 距离顺序，再比较 `top-p=0.40`、`typical_p=0.40`、`min_p=0.30` 保留的 token；解释为何候选集合不同本身不能证明输出质量高低。
8. 为事实问答、代码生成、创意写作分别设计 decoding 参数。
8A. 令 target `p=(0.50,0.30,0.20)`、draft `q=(0.20,0.50,0.30)`，逐项计算接受概率和拒绝后的 residual distribution，并验证最终 A/B/C 边际概率；说明 exact distribution guarantee 与同 seed 重放的区别，以及 Medusa typical acceptance 为什么不能直接称为严格分布保持。
9. 解释为什么 repetition penalty 过强可能伤害代码生成。
10. 用自己的话解释 KV Cache 为什么能加速推理。
11. 比较 prefill 和 decode 阶段的瓶颈。
12. 用公式估算一个 32 层模型在 4096 token 下的 KV Cache 显存。
13. 解释为什么 GQA/MQA 能降低 KV Cache 显存。
14. 说明 PagedAttention 为什么适合高并发 LLM serving。
15. 写出标准 attention 需要显式构造的两个大矩阵。
16. 解释为什么 `[T, T]` attention matrix 在长上下文下很贵。
17. 用自己的话解释 IO-aware optimization。
18. 比较 FlashAttention、sparse attention、linear attention。
19. 说明 FlashAttention 在 prefill 和 decode 阶段的作用差异。
20. 估算 7B 模型 FP16、INT8、INT4 权重显存。
21. 解释 weight-only quantization 为什么常用于 LLM 推理。
22. 比较 PTQ 和 QAT。
23. 用自己的话解释 GPTQ 和 AWQ 的区别。
24. 设计一个量化模型上线前的质量和性能评估清单。
25. 解释连续分配 KV Cache 为什么会产生内部碎片和外部碎片。
26. 用操作系统分页类比解释 PagedAttention。
27. 画出 `logical_block_id -> physical_block_id` 的 block table 示例。
28. 比较 block size 过大和过小分别会带来什么问题。
29. 说明 PagedAttention 为什么有助于 Continuous Batching。
29A. 比较 PagedAttention、vAttention 和普通连续 KV 分配，要求从显存碎片、动态加入退出、系统复杂度、kernel 适配、可观测性和故障边界六个角度回答。
29B. 用 20 个 token 和 8 个 KV 槽作教学例子，对照最近窗口、H₂O 的 heavy hitters + recent tokens、StreamingLLM 的 attention sinks + recent window 各自会保留哪些 token；说明这些有损 eviction 与完整 KV offload、KV 量化的差别，并设计能发现中段证据丢失的评测切片。
29C. 用 KIVI 教学参数 `G=2`、`R=4` 手推 6 个 token 后和第 7 个 token 到来后的 K/V 量化状态；解释 K per-channel 与 V per-token 的选择理由、scale/zero-point/residual 对实际显存的影响，以及论文 cache 量化算法和 runtime FP8 配置为什么不能视为同一方案。
30. 写一个 0 依赖 Python 推理性能事故审计 demo，输入 toy 请求 trace，输出 P95 TTFT、P95 TPOT、P99 E2E、P95 queue、KV pressure、cache hit rate、prompt cost drift、错误 / 取消率、失败门禁和 gate pass。
31. 给定一组请求日志，区分 TTFT regression 和 TPOT regression 的根因，并说明哪些证据能排除 GPU kernel 本身是瓶颈。
32. 设计一个 prompt cost drift 仪表盘，至少拆分 system prompt、RAG chunk、tool schema、多轮历史、输入 token、prefill 时间、KV Cache 和单位成功任务成本。
30. 设计一个压测实验，观察并发数、上下文长度和 KV Cache 显存占用的关系。
31. 解释为什么静态 batching 不适合变长自回归生成。
32. 画出一个 Continuous Batching 中请求加入、完成、退出的时间线。
33. 比较 dynamic batching 和 Continuous Batching。
34. 写一个简化调度器伪代码，包含 waiting queue、running batch、decode step 和完成释放。
35. 解释长 prompt prefill 为什么会影响已有请求的流式输出速度。
36. 说明 chunked prefill 的收益和代价。
37. 设计一个 token budget 策略，平衡新请求 TTFT 和老请求 TPOT。
38. 列出线上排查“tokens/s 高但用户觉得卡”的 5 个方向。
39. 画出 Speculative Decoding 中 draft model 和 target model 的交互流程。
40. 解释为什么给定候选 token 后，target model 可以并行验证多个位置。
41. 用自己的话解释接受概率 `min(1, p(y)/q(y))`。
42. 分析 draft 模型太弱、太慢、太大分别会带来什么问题。
43. 比较 greedy speculative decoding 和 sampling speculative decoding。
44. 设计一个实验，按任务类型统计 speculative decoding 的接受率和加速比。
45. 说明 Speculative Decoding 和模型蒸馏的区别。
46. 列出 Speculative Decoding 在线上落地需要关注的 5 个工程问题。
47. 解释 Medusa heads 如何从当前 hidden state 预测未来多个 token。
48. 说明为什么 Medusa 需要候选树和原模型验证。
49. 比较 Medusa、EAGLE 和普通 draft model speculative decoding。
50. 用自己的话解释 EAGLE 为什么在 feature 层面预测未来状态。
51. 设计一个实验，比较不同候选树大小下的接受率、TPOT 和显存占用。
52. 分析多 token 预测在高温采样下为什么可能收益下降。
53. 列出 Medusa/EAGLE 上线时需要检查的 5 个系统指标。
54. 用 2 分钟回答“多 token 预测为什么不是简单一次输出多个 token”。
55. 写出量化和反量化公式，并说明 scale 与 zero-point 的作用。
56. 比较 per-tensor、per-channel 和 group-wise quantization。
57. 解释为什么 LLM 推理中常见 weight-only quantization。
58. 比较 PTQ 和 QAT 的成本、质量和适用场景。
59. 用自己的话解释 GPTQ 如何利用校准激活做误差补偿。
60. 用自己的话解释 AWQ 为什么要保护激活敏感的重要权重。
61. 设计一个 INT4 量化模型上线前的评估 checklist。
62. 分析为什么量化后 perplexity 变化很小，但 JSON 输出稳定性可能变差。
63. 列出 INT4 模型显存降低但速度没有提升的 5 个可能原因。
64. 设计一个混合精度量化方案，说明哪些层可能保留更高精度。
65. 解释 KV Cache 量化和权重量化的区别。
66. 比较 K cache 量化误差和 V cache 量化误差的影响。
67. 设计一个 per-token KV quantization 的 scale 存储方案，并说明元数据开销。
68. 比较 INT8 KV Cache 和 INT4 KV Cache 的收益与风险。
69. 设计一个长上下文 KV Cache 量化评估集，至少包含 4 类任务。
70. 分析为什么 KV Cache 显存下降不一定带来 TPOT 下降。
71. 设计一个“最近 token FP16、远端 token INT8”的混合精度 KV 策略。
72. 列出 KV Cache 量化上线前需要监控的质量和系统指标。
73. 画出一个 LLM serving 系统架构图，包含 gateway、router、scheduler、GPU worker 和 streamer。
74. 设计一个按 token 负载而不是 request 数负载均衡的 router。
75. 设计一个 scheduler，说明如何同时控制 token budget 和 KV Cache budget。
76. 列出 TTFT 变高、TPOT 变高和 OOM 增加时各自的排查清单。
77. 设计一个长请求和短请求隔离方案。
78. 设计一个 LLM rate limit 方案，覆盖 request、input token、output token 和并发。
79. 设计一个 LLM serving 监控 dashboard，包含 P50/P95/P99、tokens/s、KV blocks、error rate。
80. 用系统设计面试方式回答“如何设计一个 ChatGPT 类推理服务”。
80A. 对照 Orca、PagedAttention/vLLM、Sarathi-Serve、SGLang 和 vAttention：分别写出它们针对的系统瓶颈、机制、作者报告的模型/硬件/指标、主要限制及后来者改进方向；明确为什么论文中的 36.9×、2–4×、2.6×、6.4× 和 1.23×不能组成一张横向性能排行榜。
80B. 为同一模型设计两个引擎压测：一个固定并发闭环，一个固定 RPS 开环；分别控制请求长度、burstiness、prefix cache 冷热和重复 prompt。定义 TTFT、TPOT、ITL、E2E 和满足质量/SLO 的 goodput，并说明哪个结果能回答容量、交互延迟和过载问题。
80C. 比较 Orca、FastServe、Sarathi-Serve 和 DistServe：把它们分别映射到 iteration 级重排、长请求抢占、prefill 分块、prefill/decode 资源隔离；针对每种方法写出问题、核心状态、资源代价、失败边界和一项可以证伪其收益的 trace/benchmark。
80D. 单服务台的平均服务率为 10 req/s。用 Little 定律和 M/M/1 直觉模型分别估算到达率 5 req/s 与 9 req/s 时的平均系统在途请求数和平均停留时间；说明为什么这组数不能直接作为异构 LLM serving 集群容量预测，并设计需要采集的多资源 trace。
80E. 为共置 + chunked prefill、DistServe、Mooncake、TaiChi 和 SmartGen 设计一组可证伪的 P/D serving 对照：固定模型 revision、真实输入/输出长度和 prefix reuse trace、GPU/网络、并发及 TTFT/TPOT SLO，分别记录 P/D 算力分配、KV payload、transfer overlap、destination KV reservation、TTST、TPOT、goodput 和成本。再解释 SpectrumKV 的 PPL 与这些延迟指标为什么不能合成一个排行榜，并列出会让 P/D 解耦退化为反模式的网络或 workload 条件。
81. 用 `6 * P * T` 估算 7B 模型训练 1T token 的训练 FLOPs。
82. 解释为什么真实训练成本不等于 final run 成本。
83. 设计一个推理成本估算表，包含 input token、output token、KV Cache、GPU 小时和系统 overhead。
84. 比较 input token 和 output token 的成本差异。
85. 给定 prefill tokens/s、decode tokens/s 和业务 token 流量，估算所需 GPU 数。
86. 列出 10 个推理降本策略，并说明每个策略的质量或延迟风险。
87. 设计一个 LLM cost dashboard，按模型、业务线、输入 token 和输出 token 拆分成本。
88. 比较 API 调用和自部署模型在小流量、大流量、强合规场景下的成本结构。
89. 用单位有效任务成本解释为什么小模型不一定更划算。
90. 设计一个控制 output token 成本的产品策略，例如 max_tokens、stop、摘要和缓存。
91. 比较端侧部署和云端部署在质量、隐私、延迟、成本和更新上的差异。
92. 列出 10 个适合端侧小模型处理的任务。
93. 设计一个端侧 LLM 评估表，包含冷启动、tokens/s、内存、功耗、发热和崩溃率。
94. 比较 INT8 和 INT4 在端侧部署中的收益和风险。
95. 设计一个端云协同路由策略，覆盖隐私、网络、任务难度和设备能力。
96. 分析端侧模型更新和回滚为什么比云端更难。
97. 设计一个保护隐私的端侧文档助手。
98. 设计一个弱网可用的端云混合 AI 助手。
99. 列出端侧部署上线前需要做的设备兼容性测试。
100. 用 3 分钟回答“如何把一个小语言模型部署到手机端”。
101. 解释长上下文能力的三个层次：能放进去、能看得见、能用得好。
102. 比较原生长窗口、RAG、memory 和 Agent 工作流四条长上下文路线。
103. 解释为什么不能只修改 `max_position_embeddings` 来获得长上下文能力。
104. 用自己的话解释 Position Interpolation 和 RoPE scaling。
105. 设计一个长上下文 continued pretraining 数据混合方案。
106. 设计一个检测 lost in the middle 的评估集。
107. 列出把 4k 模型扩展到 128k 的完整步骤。
108. 分析长上下文训练对 attention 计算、显存和 KV Cache 的影响。
109. 设计一个长上下文 SFT 样本，要求答案必须依赖远距离证据。
110. 用 3 分钟回答“长上下文和 RAG 是什么关系”。
111. 解释为什么最大 context window 不能代表真实长上下文能力。
112. 设计一个 needle-in-a-haystack 测试，并写出它不能覆盖的 3 类真实任务。
113. 设计一个 lost in the middle 评估，按证据位置输出准确率表。
114. 构造一个需要 3 条分散证据才能回答的长文档 QA 样本。
115. 构造一个包含相似错误证据的抗干扰长上下文样本。
116. 设计一个引用准确性评估规则，检查引用是否支持答案。
117. 列出长上下文模型必须保留的 5 类短任务回归测试。
118. 设计一个 RAG error attribution 表，区分检索、rerank、reader 和 citation 错误。
119. 设计一个长上下文质量-成本 dashboard，包含准确率、TTFT、TPOT、KV Cache 和 P99。
120. 用 3 分钟回答“如何设计可信的长上下文评估体系”。
121. 画出 RAG 离线索引链路和在线查询链路。
122. 比较固定 token chunk、按标题 chunk、语义 chunk 和滑动窗口 chunk。
123. 设计一个企业文档 metadata schema，包含权限、版本和来源信息。
124. 解释 hybrid retrieval 为什么常比单纯向量检索更稳。
125. 设计一个 retriever top-100 到 reranker top-10 的 RAG 流程。
126. 写一个要求引用和资料不足时拒答的 RAG prompt 模板。
127. 设计一个 RAG error attribution 表，覆盖解析、chunk、retrieval、rerank、context、generation、citation、permission。
128. 设计一个企业 RAG 权限控制方案，说明过滤应发生在哪些环节。
129. 设计一个 RAG 评估集，包含 retrieval、generation 和 system 三类指标。
129A. 用纯 Python 写一个 RAG 事故审计 demo，输入 toy corpus、expected evidence、retrieved、context、claims、ACL 和 staleness，输出 retrieval recall、MRR、context recall / precision、citation accuracy、unsupported claim rate、permission leak rate、stale evidence rate 和 failed gates。
129B. 构造 5 个 RAG bad case：retrieval miss、context drop、旧版本引用、越权证据进入上下文、证据不足却未拒答，并为每个样本写出 root cause 和修复优先级。
129C. 设计一个 GraphRAG 小型案例：给 8 段跨文档材料抽取 entity、relation 和 community summary，并比较 GraphRAG、向量 RAG、Agentic RAG 在全局问题、多跳问题和局部事实问题上的适用边界。
129D. 对照原始固定 top-k RAG、Self-RAG、CRAG、GraphRAG 和 Agentic RAG：为“局部事实漏召回、检索候选不可靠、语料全局总结、多跳证据缺口”分别选方案，列出核心机制、运行/索引成本、失效条件、需要的评估集和一项下一步改进假设。
129E. 用同一组 query、context、reference 和人工标签比较 RAGAS 与 ARES 风格 judge：记录 faithfulness、context relevance、answer relevance、人工一致率和置信区间；构造一个跨领域分布变化样本，说明自动指标为什么需要重新校准。
129F. 设计检索投毒红队样本：攻击者可向一个公开知识源写入被优化的恶意文档，正常用户查询触发召回，Agent 持有读邮件与外发工具。分别验证 ingestion provenance、retrieval hit、source taint、least privilege、pre-tool policy、人工确认和 clean-task utility；说明为何仅依赖提示注入分类器无法证明系统安全。
130. 用 5 分钟回答“如何设计一个企业知识库 RAG 问答系统”。
131. 画出 embedding retrieval 的离线索引和在线查询流程。
132. 用自己的话解释双塔检索为什么适合大规模召回。
133. 写出 cosine similarity 公式，并说明和 dot product 的关系。
134. 解释 in-batch negatives、hard negatives 和 false negatives。
135. 构造一个 query、positive document、hard negative document 的训练样本。
136. 比较 HNSW、IVF、PQ 的直觉、优点和代价。
137. 设计一个向量检索评估集，包含 Recall@k、Hit@k、Precision@k、MRR、nDCG；至少构造一个包含多个 gold 文档的 query，说明 Recall@k 与 Hit@k 为什么会给出不同结果，并处理无 gold 的 query。
138. 设计一个 hybrid retrieval 方案，说明如何合并 BM25 和向量检索结果。
139. 分析 embedding 模型升级为什么通常需要重建索引。
140. 用 3 分钟回答“RAG 检索不到正确文档时如何排查”。
141. 解释为什么 embedding retriever 后面常接 reranker。
142. 比较 bi-encoder 和 cross-encoder 在速度、质量和成本上的差异。
143. 构造一个 query、positive chunk、hard negative chunk 的 reranker 训练样本。
144. 比较 pointwise、pairwise、listwise reranker 训练方式。
145. 设计一个 reranker 评估表，包含 MRR、nDCG、precision@k、latency 和下游答案质量。
146. 设计一个 retriever top-100、reranker top-10、LLM context top-5 的流程。
147. 设计一个去重和多样性控制策略，避免重复 chunk 占满 prompt。
148. 分析 reranker 引入后对端到端延迟和成本的影响。
149. 设计一个 metadata-aware reranking 方案，考虑权限、版本和更新时间。
150. 用 3 分钟回答“检索到了相关文档，但 RAG 仍然答错，如何排查”。
151. 举 3 个 RAG 仍然会 hallucinate 的例子，并标注根因。
152. 比较 correctness、faithfulness、groundedness 和 attribution。
153. 把一个 RAG 回答拆成 atomic claims，并为每个 claim 找证据。
154. 设计一个 citation accuracy 评估规则，要求检查引用是否支持答案。
155. 构造一个资料不足时必须拒答的 RAG 测试样本。
156. 设计一个 unsupported claim rate 的计算流程。
157. 写一个 LLM-as-a-judge prompt，用于判断回答是否被 context 支持。
158. 设计一个 RAG hallucination error attribution 表。
159. 构造一个包含冲突证据的新旧版本 RAG 测试样本。
160. 用 3 分钟回答“如何治理 RAG 系统中的幻觉”。
161. 用 3 分钟回答“Agent 和普通 LLM 应用有什么区别”，要求覆盖 goal、state、action、observation、controller 和 trace。
162. 画出一个 Agent loop，包含目标读取、状态检查、动作选择、权限校验、工具执行、观察解析、状态更新和停止判断。
163. 设计一个 Agent trace 日志 schema，字段包含 task id、goal、state diff、action、tool、arguments、permission result、observation、budget、stop reason 和 final status。
164. 写一个 0 依赖 Agent trace 审计 demo，输入 toy trace，输出 task success rate、tool selection accuracy、argument validity、observation use rate、budget overrun rate、unauthorized action rate、stop correctness 和 agent gate。
165. 给定 3 条失败 Agent trace，分别归因到目标理解、工具选择、参数、observation 使用、state update、budget、permission 或 stop condition。
166. 设计一个天气查询工具 schema，包含城市和日期字段。
167. 设计一个订单查询工具 schema，并说明权限检查在哪里做。
168. 给定 5 个用户问题，判断是否需要调用工具，并写出工具名和参数。
169. 构造一个工具参数缺失时需要追问用户的样本。
170. 构造一个工具返回错误时模型不编造结果的回答样本。
171. 设计一个高风险工具调用的二次确认流程，例如转账、删除数据或发邮件。
172. 设计一个 function calling 评估表，包含工具选择、参数、执行、最终回答和安全。
173. 设计一个 tool call trace 日志 schema。
174. 分析 prompt injection 如何诱导工具越权调用，并给出 3 个防护策略，要求包含 tool result injection、参数污染、二次确认和 audit trace。
175. 用 3 分钟回答“如何设计一个能调用企业 API 的 LLM 助手”。
176. 写一个 0 依赖 function calling 审计 demo，输入 toy tool schema、toy tool calls、期望工具、期望参数、权限确认状态和工具返回文本，输出 schema valid rate、tool selection accuracy、argument exact match、unauthorized block rate、tool result injection rate、failed gates 和 tool calling gate pass。
176A. 写一个 0 依赖 function calling 协议完整性审计 demo，输入 toy message traces，覆盖缺 assistant tool call、tool result id 错误、finish reason 不一致、streaming 半截参数被执行、parallel result 错配和副作用工具重复执行，输出 protocol chain validity、tool result ID match rate、argument parse rate、finish reason consistency、streaming safety rate、parallel alignment rate、idempotency protection rate 和 protocol gate pass。
176B. 写一个 0 依赖 Tool Schema 参数约束审计 demo，输入 toy schemas 和 toy tool calls，覆盖缺 required、类型错误、enum 错误、pattern 错误、数值越界、extra field、可安全修复参数和 schema 合法但业务规则失败的样本，输出 schema valid rate、required field pass rate、type valid rate、enum valid rate、pattern valid rate、range valid rate、additional properties block rate、business rule pass rate、schema repair success rate、failed gates 和 schema gate pass。
176C. 写一个 0 依赖 Tool Choice 策略审计 demo，输入 toy policy、allowed tools、tool calls、tool results、确认状态、成本预算、最大并发和 loop 状态，覆盖纯知识问题误调用工具、required 实时查询、forced 缺参编造、并行结果 ID 错配、依赖工具错误并行、过量并发、高风险未确认和重复搜索超限，输出 candidate coverage、tool choice mode accuracy、no-tool clarification block rate、forced missing argument block rate、parallel safety rate、parallel ID alignment rate、rate limit pass rate、confirmation enforcement rate、cost budget pass rate、loop control rate、failed gates 和 tool choice gate pass。
177. 给一个工具 schema 增加 `additionalProperties=false`，并设计 3 个 extra field 测试样本。
178. 构造一个“参数格式合法但业务语义错误”的订单查询样本，说明 schema validation 为什么不能替代权限检查。
179. 设计一个 tool result injection 防御性评估样本，要求包含不可信 observation、后续 action、permission result 和 final answer。
180. 设计一个高风险写入工具的二次确认流程，字段包含 role、risk level、confirmed、audit id 和 rollback plan。
181. 给 5 条 tool call trace，分别归因到 wrong tool、bad argument、schema failure、permission failure、tool timeout、observation ignored 或 injection violation。
182. 比较 tool registry 和 tool executor 的职责，并说明哪个模块负责版本、权限、执行和日志。
183. 用 3 分钟回答“为什么 function calling 不只是输出 JSON”，要求覆盖 schema、executor、permission、observation、trace 和 eval。
184. 把一个 ReAct trace 写成 `goal -> thought -> action -> observation -> state update -> final` 的结构。
185. 设计一个 Plan-Act-Observe trace，要求包含 initial plan、plan update、action、observation、stop reason 和 final status。
186. 写一个 0 依赖 ReAct / PAO trace 审计 demo，输出 action accuracy、plan adherence rate、observation use rate、state update coverage、repeat action rate、budget overrun rate、premature final rate 和 gate pass。
187. 构造一个重复 action 失败样本，并设计 duplicate action guard。
188. 构造一个 observation ignored 样本，说明为什么最终答案看似合理也不能通过验收。
189. 构造一个 premature final 样本，要求说明缺失的 success criteria。
190. 设计一个 blocked action recovery 流程，高风险动作被拦截后应转为 draft、ask confirmation、降级或 stop。
191. 比较 ReAct 与 Plan-Act-Observe 的区别，并说明什么时候应该先规划、什么时候应该边做边改。
192. 给 3 条 ReAct trace，分别归因到 plan drift、parse failure、bad stop、budget overrun 或 blocked not recovered。
193. 把“修复一个失败测试并补充回归测试”拆成子任务、依赖图和验收标准。
194. 设计一个 planning audit demo，输出 goal coverage、executable step rate、acceptance coverage、dependency violation rate、replan coverage、failure recovery rate、risk confirmation coverage 和 gate pass。
195. 构造一个依赖顺序错误的计划，例如未评估就部署，并写出正确拓扑顺序。
196. 构造一个过度分解计划，说明哪些步骤不可执行、缺少验收或重复覆盖同一子目标。
197. 设计一个 dynamic replanning 样本，要求记录 failure observation、plan update reason 和新的子目标。
198. 设计一个高风险子任务规划表，字段包含 risk level、permission、confirmation、rollback plan 和 audit id。
199. 计算一个 toy dependency graph 的 critical path length，并说明哪些任务可以并行。
200. 用 3 分钟回答“Agent 如何做任务分解，以及如何评估规划质量”。
201. 把 Agent memory 分成 short-term、semantic、episodic、procedural 和 preference 五类，并各写 2 条 toy memory。
202. 设计一条 memory schema，字段包含 user、project、kind、key、value、source、timestamp、importance、confidence、sensitivity、scope、expires_at 和 deleted。
203. 写一个 0 依赖 memory retrieval demo，用词集合相似度、recency decay、importance、confidence 和 stale penalty 输出 top-k memory。
204. 设计一个 memory permission gate，要求检查 user namespace、project namespace、private scope、sensitivity、deleted flag 和 expiry。
205. 构造 3 条 memory conflict 样本，例如同一项目 Python 版本冲突，并写出覆盖、降权、保留版本或请求确认的处理策略。
206. 设计一个 memory write gate，拦截 one-off instruction、tool result injection、敏感信息、低置信模型推断和低未来价值候选。
207. 写一个 0 依赖 memory audit demo，输出 retrieval precision、retrieval recall、stale use rate、blocked memory count、conflict count、unsafe write block rate、deleted memory returned 和 gate pass。
208. 用 3 分钟回答“为什么 Agent memory 不是把历史聊天都塞进 prompt”，要求覆盖 context、state、RAG、写入、检索、权限、过期和删除。
209. 把一个复杂 RAG 问题拆成 4 个 retrieval subquestions，并说明每个 query 要找什么 evidence gap。
210. 设计一个 Agentic RAG trace schema，字段包含 goal、round id、query、retrieval tool、filters、returned docs、selected evidence、evidence gap、stop reason 和 final claims。
211. 写一个 0 依赖 Agentic RAG audit demo，输出 context precision、context recall、facet coverage、citation accuracy、stale evidence rate、conflict count、blocked injection count、blocked unauthorized count、avg new evidence gain 和 gate pass。
212. 构造一个 query drift 样本，说明 query rewrite 如何丢失原始约束，并写出防漂移检查规则。
213. 构造一个 citation hallucination 样本，要求文档真实存在但不支持 claim，并计算 citation accuracy。
214. 设计一个多轮检索停止条件，要求结合 evidence support、新证据增益、冲突数量、预算和不可回答判断。
215. 构造一个 RAG prompt injection 检索样本，要求说明外部文档为什么只能作为 evidence，不能作为指令。
216. 用 3 分钟回答“Agentic RAG 为什么不是检索越多越好”，要求覆盖成本、query drift、context precision、引用错误和停止条件。
217. 设计一个 Code Agent trace schema，字段包含 task id、required files、search actions、read files、edits、commands、tests、final status 和 validation summary。
218. 写一个 0 依赖 Code Agent audit demo，输出 task success rate、test pass rate、validation coverage、patch localization precision / recall、unrelated change rate、user change violation rate、dependency change rate、command success rate、repeat command rate、unsafe command block rate 和 gate pass。
219. 构造一个最小 patch 样本：用户只要求修一个边界条件，要求写出允许修改文件、禁止修改文件和验证命令。
220. 构造一个无关改动样本，计算 unrelated change rate，并说明为什么看似合理的重构也可能不该通过。
221. 构造一个触碰用户已有改动的样本，说明 Code Agent 应该如何请求确认或改用更小 patch。
222. 设计一个 Code Agent 测试策略，要求包含相关测试、失败日志分析、回归测试、全量测试未运行说明和验证摘要。
223. 设计一个 code sandbox policy，覆盖文件写入范围、命令超时、依赖变更、敏感文件、生成文件和高风险动作确认。
224. 用 3 分钟回答“Code Agent 的输出为什么不是一段代码，而是 diff、测试结果和审计 trace”。
225. 设计一个 Browser Agent trace schema，字段包含 task id、URL、screenshot summary、accessibility nodes、action、target、coordinates、form value、risk level、confirmation、observation 和 final verification。
225A. 比较 WebArena、OSWorld、SWE-bench、AgentBench 和 GAIA：分别写出任务环境、主要动作空间、成功标准、可复现性风险、常见失败模式和适合检验的 Agent 能力。
226. 写一个 0 依赖 Computer-Use audit demo，输出 task success rate、final verification rate、action accuracy、misclick rate、form accuracy、state observation coverage、high-risk protection rate、prompt injection block rate、failure recovery rate、repeat action rate 和 gate pass。
227. 构造一个误点击样本，说明为什么 action accuracy 之外还要单独统计 misclick rate。
228. 构造一个表单填写错误样本，要求记录字段、期望值、实际值、提交前复核和恢复策略。
229. 设计一个高风险 UI action 确认流程，覆盖支付、删除、发送邮件、提交申请和权限修改。
230. 构造一个网页 prompt injection 样本，要求说明网页内容为什么只能作为不可信 evidence，不能作为系统指令。
231. 比较 screenshot、DOM 和 accessibility tree 在 UI Agent 中的优缺点。
232. 用 3 分钟回答“为什么 UI Agent 的难点不是能点鼠标，而是状态理解、风险控制和失败恢复”。
233. 设计一个 Multi-Agent trace schema，字段包含 run id、agent role、task assignment、message intent、evidence、conflict、permission violation、cost 和 final status。
234. 写一个 0 依赖 Multi-Agent audit demo，输出 task success rate、single-agent lift rate、role match rate、message valid rate、evidence support rate、conflict resolution rate、duplicate work rate、permission violation rate、unnecessary multi-agent rate 和 gate pass。
235. 给一个研究报告任务设计 planner、researcher、verifier、summarizer 和 coordinator 的权限边界，说明哪些信息应该进入 blackboard。
236. 构造一个角色分配错误样本，计算 role match rate，并说明 coordinator 如何修复。
237. 构造一个冲突样本，要求包含两个 Agent 的证据、冲突类型、自动验证结果和人工升级条件。
238. 构造一个重复劳动样本，计算 duplicate work rate，并说明共享状态如何减少重复。
239. 比较 debate、judge、verifier 和 voting 在开放问答、代码修复、事实核验和高风险决策中的适用边界。
240. 用 3 分钟回答“为什么 Multi-Agent 不是 Agent 越多越好”，要求覆盖单 Agent baseline、通信成本、冲突、权限和评估指标。
241. 设计一个 Agent eval sample schema，字段包含 goal、initial state、tools、permission policy、verifier、risk level、weight 和 reset rule。
242. 写一个 0 依赖 Agent evaluation audit demo，输出 task success rate、avg partial score、tool selection accuracy、argument valid rate、observation use rate、state update coverage、summary faithfulness、claim support rate、recovery success rate、repeat action rate、unauthorized action rate、high-risk confirmation rate、avg cost、P95 latency 和 gate pass。
243. 构造一个最终总结不忠实的 trace 样本，要求列出 summary claim、trace evidence 和 unsupported claim。
244. 设计一个错误恢复评估样本，覆盖工具超时、权限不足、测试失败、搜索无结果和页面状态变化。
245. 设计一个 Agent benchmark 的 sandbox reset 机制，说明如何固定初始状态、工具版本、权限和验收脚本。
246. 比较自动评估、人工 rubric 和 LLM judge 在 Code Agent、Browser Agent、RAG Agent 和数据分析 Agent 中的适用边界。
247. 构造一个 Agent 成本评估表，包含模型调用、token、工具调用、重试、人工审阅、P95 延迟和每成功任务成本。
248. 用 3 分钟回答“为什么 Agent eval 必须看 trace、成本和安全，而不能只看任务成功率”。
249. 设计一个 Agent tool permission matrix，覆盖 read-only、write、external transfer、high-risk、scope、TTL 和 approval 字段。
250. 写一个 0 依赖 Agent safety audit demo，输出 unauthorized attempt block rate、untrusted instruction block rate、tool output block rate、sensitive data block rate、external transfer block rate、high-risk protection rate、dry-run coverage、sandbox block rate、memory pollution block rate、audit completeness 和 gate pass。
251. 构造一个不可信内容隔离样本，只记录 source、trust level、expected policy action 和 metric，不写可复用注入文本。
252. 设计一个 external transfer gate，说明内部数据传给外部工具前要检查哪些权限、脱敏和审计字段。
253. 设计一个 high-risk confirmation 页面，要求展示动作、对象、参数摘要、影响范围、dry-run 结果、取消按钮和审计 id。
254. 构造一个 memory pollution 风险样本，说明为什么不可信来源不能写入长期规则。
255. 设计一个 sandbox policy，覆盖文件系统、网络、CPU、内存、子进程、环境变量、命令白名单和工作目录。
256. 用 3 分钟回答“为什么 Agent 安全是系统架构问题，而不是 prompt 问题”。
257. 设计一个 Agent interview readiness rubric，字段包含 concept coverage、formula coverage、demo coverage、trace metric coverage、safety coverage、evaluation coverage、project evidence、trade-off depth 和 red flags。
258. 写一个 0 依赖 Agent interview readiness demo，输入 5 道 mock interview 回答记录，输出 question scores、overall coverage、red flags、weak questions、average score、readiness gate 和 revision plan。
259. 把 12 道 Agent 高频题按 Agent loop、tool calling、ReAct / PAO、planning、memory、Agentic RAG、Code Agent、UI Agent、Multi-Agent、evaluation、safety 和 project 分桶。
260. 给每个 Agent 弱题绑定一个公式、一个 demo、一个 bad case 和一个 3 分钟回答模板。
261. 设计一个 Agent 项目证据表，字段包含 baseline、offline metrics、trace sample、tests、bad cases、personal ownership、trade-off 和 next iteration。
262. 构造 3 条 Agent 面试红旗回答，并分别改写成覆盖 controller、trace、评估和安全边界的版本。
263. 给一轮 Agent mock interview 计算平均分、最低单题分、安全覆盖率、红旗数量和 readiness gate。
264. 用 3 分钟回答“为什么 Agent 面试准备不是背框架名，而是证明你能建立可评估、可控制的任务执行系统”。

## 阶段 5：研究与面试验收

1. 精读 10 篇论文。
2. 完成一个项目讲稿。
3. 完成 3 次 mock interview。
4. 能回答开放研究题。
5. 给出 5 个不同类型的幻觉例子。
6. 设计一个事实性评估集，说明数据来源和评分方式。
6A. 写一个 0 依赖评估指标事故审计 demo，输入 toy baseline / candidate 评估样本、切片、污染标记、人工偏好、judge 偏好、输出长度、成本、延迟、线上反馈和安全状态，输出 paired lift、bootstrap CI、slice regression、clean eval lift、judge-human agreement、judge length bias、cost ratio、latency delta、failed gates 和 gate pass。
6B. 做一张模型发布 benchmark 审计表，至少覆盖 MMLU / MMLU-Pro、GPQA、AIME、HLE、HumanEval、LiveCodeBench、SWE-bench、SWE-Lancer、DeepSWE、WebArena、OSWorld、GAIA、tau-bench、BrowseComp、LongBench、RULER、FRAMES、SimpleQA、TruthfulQA、MMMU、MathVista、OCRBench 和 Video-MME；每行写清数据规模、主要能力、评分方式、仿写样例、污染风险、适用场景和不能代表什么。
6C. 做一次 frontier release radar 复盘：任选一个近期模型发布报告，按架构、预训练、后训练、test-time compute、serving、Agent/tool、多模态、评估、安全治理、产品工程和技术生命周期 11 个维度提取新知识；至少写出 3 个已稳定进入主干的知识点、3 个 P1 观察项、3 个可能被模型能力吸收或淘汰的过渡技术，并说明应同步到哪本书、哪道面试题和哪个术语条目。
6D. 以客服 RAG 的模型升级为例，分别选用 HELM 式覆盖矩阵、lm-eval 式固定 runner、human/pairwise、ConStat 式改写题检查和线上实验；逐项写清输入对象、能支持的判断、主要限制及下一步补证，说明为什么不能由一张总分表替代整条评估链。
6E. 为一个新发布的科学问答助手设计 benchmark，比较 GPQA 式专家构题、IFEval 式可程序核验约束、LiveBench 式滚动新题和选项/评分协议扰动审计；说明每种设计验证什么、不能证明什么、需要保存哪些版本字段，以及如何把固定回归集与新鲜题集同时维护。
6F. 对一组 1,000 个 pairwise 评审（新模型胜 420、旧模型胜 350、平局 230）计算折半平局偏好率、单条得分方差和独立样本近似区间；比较错误地套用 Bernoulli 方差会得到什么；再按目标最小差异规划样本量，并设计 prompt/user 聚类 bootstrap。最后说明 Arena 的自适应对战抽样、Elo/Bradley--Terry 聚合、MT-Bench 专家偏好和 ICLR 2026 最坏情况删点审计各自解决什么问题、不能支持什么结论；再对照 Zhao 等人基于旧 Arena 快照注入随机/定向低质量票的压力测试，说明模拟比例不等于真实坏票率，并列出人评报告需保留的抽样、评审、质量控制和不确定性字段。
6G. 为同一个客服问答任务比较 G-Eval、Prometheus 2 和 SAJA：分别画出一次样本从输入、评分到汇总的路径，指出它们依赖 token 概率、专用 evaluator 训练数据或人工校准标签中的哪些资源；分别选择标签一致、模型排序、绝对分数校准或自动放行指标。再设计一组仅改变答案事实、仅反转 rubric 的反事实样本，并用独立 holdout 检查候选方法在目标数据上的边界；解释论文 headline score 为什么不能直接横向排名。
6H. 为同一个评估计划比较 GSM8K、GSM-Symbolic/GSM-NoOp、ORM/PRM/ProcessBench、EvalPlus、LiveCodeBench 和 SWE-bench Verified：每条路线说明测量对象、关键操作、论文或官方报告支持的结论、作者设置中的限制，以及针对该限制的下一步补证。为一项数学任务和一项代码助手任务各选一条主路径，设计固定回归集、滚动新题和独立 holdout 的分工；说明为何这些分数不能合成一个总分。
6I. 为一个“多页合同 + 图表附件 + 会议录音”的助手设计评估组合。比较 DocVQA/ANLS、MMLongBench-Doc、ChartQA、AudioBench、Video-MME/MLVU，以及 NIAH、Lost in the Middle、RULER、NoLiMa、LongBench v2 各自能测的失败；选出主任务基准和补充探针，写出各自的样本构造、指标、答案/证据格式、提示与预处理版本、一个重要限制，以及针对该限制的私有对照。最后说明为什么不能把这些分数平均成一个“多模态长上下文总分”。
6J. 为一个声明支持 1M context 的研究助手设计一套评测证据链。分别说明 NIAH、Lost in the Middle、RULER、NoLiMa、LongBench v1/v2、LongBench Pro 和 MRCR v2 各自补上什么缺口、使用什么长度单位、最容易误读的结论是什么；组合成“输入/位置冒烟 → 受控复杂度 → 自然任务覆盖 → 1M 目标语料”的评测矩阵。固定 tokenizer、模板、工具权限、推理预算和文档来源，设计至少一个冲突归因与一个不可回答对照；按任务和长度报告正确率、证据召回、引用支持、P95、单位成功成本与不确定性，并指出哪些结果仍不能支持“模型有效使用任意 1M 输入”。
6K. 选择一个刚发布的 frontier model claim，先拆成模型身份、架构/接口、任务能力和产品效果等最小命题，再比较 Model Card、Datasheet、HELM、CheckList、G-Eval 与 Evaluation Card 分别解决什么问题。为每种方法写出输入、操作机制、它能支持的结论、作者结果或证据、主要限制及下一步补证；再用 Claim Ledger 记录来源、revision、harness、预算、独立性和未知项。最后选一项厂商 benchmark、一项第三方 provider/config 测量和一项固定 harness 的独立评测，说明为何缺失榜单行不等于零分，也不能借用相邻模型的结果。
6L. 为一个要自动处理真实工作流的 Agent 设计评测组合，比较 SWE-bench、SWE-Gym、Terminal-Bench 2.0、WebArena、OSWorld、BrowserGym、τ-bench、ToolSandbox、AgentDojo、GAIA、ScienceAgentBench、PaperBench、METR 和 Gaia2。每种选用的方法都写清任务对象、机制、判定器、作者证据、限制和下一步补证；明确区分 benchmark、实验框架、训练资源、难度标尺与动态环境。再设计工具合同审计，验证工具前置条件、状态变化和 grader 实际读取字段，并说明为什么“潜在缺陷”不能直接推出 Agent 曾触发或榜单结论已改变。
7. 解释 benchmark contamination 为什么会误导模型比较。
7A. 用英文术语 data contamination、benchmark leakage 和 train-test leakage 分别写一个 LLM 评估风险例子，并说明如何用 exact match、near duplicate、canary、时间切分和私有动态集降低风险。
7B. 为同一个公开 benchmark 比较语料 exact/near overlap 检查、Oren 顺序似然检验、ConStat、PaCoST、发布前 benchmark watermark 和 AntiLeakBench 式新鲜知识集。逐项写明它直接检验的命题、所需数据/API、核心假设、能支持的结论、失败条件和补证手段；再为“只有文本生成 API”“有 token log-prob 但无训练语料”“即将发布新 benchmark”三种条件选方案。区分数据重合、行为记忆信号与分数不泛化，不把异质结果合成一个污染率。
8. 写 3 个防御性 jailbreak / prompt injection 抽象测试样本，只记录风险类别、期望策略动作和评估指标，不写可复用攻击提示词。
9. 设计一个包含 helpfulness、honesty、harmlessness 的评估表。
10. 用自己的话回答“如何判断模型是否真的具备推理能力”。
11. 设计一个多模态模型的研究 roadmap。
12. 给定 3 个分辨率和 2 个 patch size，手算每张图片进入 VLM 后的视觉 token 数，并说明对 OCR、图表和延迟的影响。
13. 写一个 0 依赖 Python demo，输入文本 token、图片分辨率、视频帧数、音频时长和上下文上限，输出多模态总 token、over budget 样本和审计门禁。
14. 用 3 分钟回答“为什么多模态模型设计必须先算视觉 token 和上下文预算，而不是只比较模型榜单分数”。
15. 给 3 对 toy image/text embedding 手写 CLIP 相似度矩阵、image-to-text loss、text-to-image loss 和平均 loss。
16. 写一个 0 依赖 Python demo，计算 CLIP 风格的 L2 normalize、`N x N` 相似度矩阵、Recall@1、MRR 和 zero-shot prompt ensemble 预测。
17. 固定一组 toy logits，分别用 3 个 temperature 计算 softmax 置信度，并解释 `logit_scale` 为什么会影响训练梯度和检索排序置信度。
18. 用 3 分钟回答“CLIP 为什么能做 zero-shot 分类，但不能等价于视觉问答模型”。
19. 给定 `B=2,C=3,H=W=336,P=14,d_v=1024`，手算 patch grid、patch token 数、加 CLS 后的 position embedding shape 和 patch embedding 参数量。
20. 写一个 0 依赖 Python demo，输入分辨率、patch size、vision hidden size 和 LLM hidden size，输出 patch tokens、attention cell、projector 参数量和 over-budget 标记。
21. 对比 CNN residual block 和 ViT patch embedding 的直觉，说明为什么 CNN 更有局部归纳偏置，而 ViT 更容易和 LLM 架构统一。
22. 构造一个 OCR / 图表 bad case，说明该如何选择高分辨率、切图、patch size 或外部 OCR 工具。
23. 写一个 0 依赖 VLM connector / prompt shape demo，检查 `[B,N_v,d_v] -> [B,N_v,d_l]` projector shape、`<image>` 占位符数量、多图 token budget、resampler 压缩、cross-attention cell 数和 assistant-only label 数。
24. 写一个 0 依赖多模态 SFT 数据审计 demo，检查 placeholder、assistant-only label 数、任务覆盖、evidence support、missing refusal、上下文预算和拒绝进入训练的坏样本。
25. 写一个 0 依赖 diffusion 加噪 / 去噪审计 demo，手算 `alpha_bar`、`x_t`、noise MSE、`x0_hat`、DDPM reverse mean、CFG 和 latent cost ratio。
26. 写一个 0 依赖文生图 pipeline 审计 demo，输入 prompt、negative prompt、分辨率、VAE scale、采样步数、CFG scale、ControlNet 条件通道数和 DALL-E image token 网格，输出 latent shape、压缩比、U-Net 调用次数、cross-attention cells、image-to-image 起始步、自回归图像 token 数和 gate pass。
27. 写一个 0 依赖视频 token / 时序一致性审计 demo，输入帧数、分辨率、spatial patch、temporal patch、latent 压缩倍数、采样步数、CFG scale 和 toy object trace，输出 framewise tokens、spatiotemporal tokens、latent shape、denoiser calls、cross-attention cells、flicker、identity stability、motion smoothness、object permanence、world rollout MAE 和 gate pass。
28. 写一个 0 依赖音频 token / WER / codec 审计 demo，输入采样率、音频时长、window / hop、mel bins、reference / hypothesis、codec 帧率、codebooks、级联延迟和说话人授权状态，输出采样点数、log-mel shape、WER、CER、codec token 数、压缩比、TTS token 数、级联延迟和 gate pass。
29. 写一个 0 依赖统一多模态样本审计 demo，输入文本 token、图片分辨率、音频时长、视频帧数、输出模态、上下文上限、loss weights 和安全标记，输出各模态 token、输入 / 输出总 token、attention cells、路由模块、loss mixture、风险标记和 gate pass。
30. 写一个 0 依赖多模态评估与安全审计 demo，输入 toy VQA、OCR / ASR、图表、grounding、claim support 和安全样本，输出 VQA accuracy、WER / CER、chart relaxed accuracy、IoU、hallucination rate、policy accuracy、risk counts、latency gate 和 gate pass。
31. 写一个 0 依赖多模态面试复盘 demo，输入 mock interview 回答记录、required topics、required formulas、required demos、risk set 和 trade-off set，输出 topic coverage、formula coverage、demo coverage、risk coverage、weak questions、missing formulas、revision plan 和 interview ready。
32. 写一个 0 依赖 reasoning 成本 / 正确率审计 demo，输入 toy reasoning 题、候选答案、verifier 分数、步骤正确标签和 token 成本，输出 greedy accuracy、self-consistency accuracy、verifier accuracy、pass@k、process step accuracy、cost per verified correct 和 gate pass。
33. 写一个 0 依赖 CoT 质量 / 成本审计 demo，输入 toy 题、direct 答案、CoT 答案、步骤正确标签、是否含 unsupported step、token 成本和可见解释边界，输出 direct accuracy、CoT accuracy、routed accuracy、step accuracy、CoT regression、unsupported step、cost per routed correct 和 gate pass。
34. 写一个 0 依赖 self-consistency 采样审计 demo，输入 toy 候选答案、verifier 分数和 token 成本，输出 normalized answers、majority vote、weighted vote、pass@1、pass@2、candidate diversity、majority failure、weighted rescue 和 cost per correct。
35. 写一个 0 依赖 verifier rerank 审计 demo，输入 toy 候选答案、RM 分数、步骤正确标签、programmatic pass、hard negative 标记和 token 成本，输出 greedy accuracy、RM rerank accuracy、hybrid verifier accuracy、pairwise accuracy、hard negative accuracy、ECE、RM failure、hybrid rescue 和 gate pass。
36. 写一个 0 依赖 process supervision / PRM 审计 demo，输入 toy 推理样本的最终答案、步骤标签、第一处错误、相关步骤、自动标注标记、人工标注成本和搜索候选，输出 outcome accuracy、step accuracy、first-error accuracy、auto label coverage、human label cost、outcome blind spots、redundant steps、search failures 和 process supervision gate。
37. 写一个 0 依赖 search / Tree-of-Thought / MCTS 审计 demo，输入 toy 推理任务的 greedy 答案、搜索分支、verifier score、MCTS value、visit count 和 token cost，输出 greedy accuracy、beam accuracy、MCTS accuracy、candidate diversity、nodes expanded、total tokens、cost per correct、pruned correct paths、MCTS rescues 和 search gate。
38. 写一个 0 依赖 test-time compute scaling 审计 demo，输入 toy 请求的 difficulty、value、verifiable、direct / self-consistency / verifier / search 成本和正确性，输出 fixed strategy accuracy、total cost、cost per correct、P95 latency、adaptive routing records、marginal accuracy per cost、wasted high compute 和 TTC gate。
39. 写一个 0 依赖数学推理训练数据审计 demo，输入 toy 数学样本的题型、难度、模板、答案正确性、verifier 结果、步骤标签、自动标注、人审成本和污染标记，输出 answer accuracy、step accuracy、first-error accuracy、auto label coverage、topic / difficulty slice accuracy、curriculum order、training mix、contaminated samples、verified-wrong samples、repeated templates 和 math training gate。
40. 写一个 0 依赖代码推理 / 执行反馈审计 demo，输入 toy 代码任务的候选代码、公开测试、隐藏测试和静态沙箱规则，输出 greedy accuracy、public rerank hidden accuracy、public-hidden gap、pass@1、pass@2、repair success rate、blocked candidates、hidden failures after public pass 和 code reasoning gate。
41. 写一个 0 依赖 reasoning 评估审计 demo，输入 toy 样本的 baseline / candidate 正确性、variant 正确性、过程步骤、第一处错误、污染标记、切片和成本，输出 paired lift、bootstrap CI、robustness drop、process step accuracy、first-error accuracy、contamination rate、cost per correct、slice accuracy、regressions、variant failures 和 reasoning eval gate。
42. 写一个 0 依赖 reasoning 安全审计 demo，输入 toy 样本的风险域、答案正确性、过程支持、置信度、工具动作、权限、人审、CoT 暴露和严重度，输出 pseudo reasoning rate、overconfident error rate、unsafe compliance rate、tool misuse rate、hidden CoT exposure rate、high-risk review coverage、over-refusal rate、severity-weighted risk 和 reasoning safety gate。
43. 写一个 0 依赖 reasoning 面试复盘 demo，输入 mock interview 回答记录、required topics、required formulas、required demos、risk set 和 trade-off set，输出 topic coverage、formula coverage、demo coverage、risk coverage、weak questions、revision plan 和 interview ready。
44. 说出 3 个平衡 safety 和 helpfulness 的具体策略。
45. 解释 test-time compute 为什么会成为重要研究方向。
46. 讨论合成数据的一个优势和一个风险。
47. 录音模拟一次 3 分钟自我介绍，并写下 3 个可改进点。
48. 选一个项目，用“问题、数据、方法、实验、失败、改进”六段式讲一遍。
49. 从题库中随机抽 10 道题，要求每题用 2 分钟回答。
45. 对一次 mock interview 做复盘，记录原题、原回答、错误原因和修正答案。
46. 准备 10 个必须能讲清的问题，并逐个写出 5 句话版本答案。
47. 做一次 45 分钟完整模拟面试：自我介绍、项目、基础、系统、开放题各一轮。
48. 写一个 0 依赖事故排查复盘 demo，输入 toy incident 记录，输出 evidence coverage、repro rate、root cause rate、priority、postmortem completeness、needs rework 和 debug gate。
49. 用 3 分钟讲一次大模型项目异常排查经历，必须覆盖背景、现象、影响范围、证据链、最小复现、baseline、根因、修复、回滚和复盘沉淀。

## 阶段 5B：产品化与商业化验收

1. 为 4 个大模型候选场景设计产品化审计表，字段包含 user pain、monthly tasks、baseline success、LLM success、adoption、value per success、P95 latency、model cost、ops cost、risk cost、fixed cost、safety、privacy、workflow、eval 和 feedback。
2. 写一个 0 依赖产品化审计 demo，输入 toy 场景，输出 success uplift、monthly benefit、monthly cost、net benefit、benefit cost ratio、ROI、payback months、rejected reasons 和 productization gate。
3. 把一个 RAG 项目的 `recall@k`、citation accuracy、answer correctness、自助解决率和客服成本下降串成一条指标链。
4. 把一个 Code Agent 项目的测试通过率、修复成功率、采纳率、节省开发时间和单位任务成本串成一条指标链。
5. 构造一个“demo 看起来很好但不适合产品化”的场景，说明它在哪些门禁上失败。
6. 设计一个大模型产品上线门禁，至少覆盖 quality、unit economics、latency、risk、workflow、eval、feedback 和 rollback。
7. 用 3 分钟回答“为什么大模型产品化不是接一个模型 API，而是建立可评估、可运营、可控风险的产品系统”。
8. 为自己的一个项目补充产品化表达：目标用户、痛点、baseline、技术指标、产品指标、ROI 粗算、风险门禁、反馈闭环和下一步迭代。
9. 为 5 个候选大模型场景设计 scenario selection audit 表，字段至少包含 user persona、stakeholder、task journey、real demand、monthly frequency、value per task、LLM fit、data readiness、verifiability、risk control、workflow fit、automation level 和 implementation difficulty。
10. 构造 2 个真实需求和 2 个伪需求，分别说明当前替代方案、用户痛点、任务频率、可衡量指标和为什么适合或不适合进入试点。
11. 写一个 0 依赖场景选择 demo，输入 toy 场景，输出 monthly benefit、LLM fit、priority score、recommendation、rejected reasons 和 pilot candidates。
12. 对同一个高价值但高风险场景，分别设计 assist、review、approval 和 full automation 四种产品形态，说明每一层需要怎样的数据、验证、权限和人审门禁。
13. 为 4 个大模型产品形态设计 experience audit 表，字段至少包含 task success、adoption、P95 latency、latency SLO、citation support、structured output stability、controllability、failure recovery、confidence calibration、over-refusal 和 unsafe output。
14. 写一个 0 依赖能力到体验 demo，输入 toy 产品指标，输出 UX score、experience gate、failed gates、ship candidates 和 needs rework。
15. 把一个高分模型但体验差的案例拆成至少 6 个失败原因，例如尾延迟、格式漂移、引用不支持、用户不可控、错误不可恢复和过度拒答。
16. 设计一个 RAG 产品的引用支持率评估规则，要求按 claim 检查证据是否存在、相关、足以支持、未过期且有权限展示。
17. 为一个 Code Assistant 设计体验门禁，至少覆盖补全延迟、采纳率、测试通过率、格式稳定、误改风险、撤销/恢复和用户反馈。
18. 为 4 个大模型场景设计 ROI audit 表，字段至少包含 monthly tasks、adoption rate、quality uplift、time saved、hourly cost、input / output tokens、model calls、cache discount、RAG cost、tool cost、review minutes、retry cost、risk cost、fixed monthly cost 和 upfront cost。
19. 写一个 0 依赖 ROI 审计 demo，输出 model cost per task、variable cost per task、benefit per task、monthly benefit、monthly cost、net benefit、benefit-cost ratio、ROI、payback months、break-even tasks、sensitivity 和 ROI gate。
20. 对同一个场景做敏感性分析，分别把采用率降低 15%、可变成本提高 20%、人审时间翻倍，观察净收益和回本周期如何变化。
21. 构造一个“用户量越大亏越多”的负单位经济账案例，说明为什么单次毛利为负时不能靠规模化解决。
22. 比较 API 调用和自部署两种成本模型，至少覆盖 token 价格、GPU 利用率、固定运维、人力、低延迟、合规、模型升级和故障风险。
23. 为 5 个企业级 LLM 应用设计 enterprise audit 表，字段至少包含 scenario type、users、data sources、permission coverage、tenant isolation、RAG permission filter、tool permission gate、audit log coverage、PII redaction、data freshness、citation support、SSO integration、workflow integration、eval ready、feedback loop、SLO pass、business metric 和 human review coverage。
24. 写一个 0 依赖企业级应用审计 demo，输入 toy 应用，输出 enterprise score、enterprise gate、failed gates、enterprise pass candidates 和 needs rework。
25. 对一个企业知识库 RAG 项目画出权限链路，要求覆盖身份登录、用户组、文档 ACL、检索过滤、rerank、上下文拼接、引用展示、日志脱敏和审计查询。
26. 选一个高风险企业场景，分别设计 copilot、review、approval 和 full automation 四种上线形态，并说明每一层需要哪些权限、SLO、评估、人审和审计门禁。
27. 为 4 个 RAG 产品设计 RAG product audit 表，字段至少包含 retrieved relevant、relevant total、context relevant、context total、supported claims、total claims、citation supported、citation total、correct abstentions、expected abstentions、permission filter rate、unauthorized hits、stale evidence rate、P95 latency、cost per answer、eval ready、feedback loop 和 business metric。
28. 写一个 0 依赖 RAG 产品审计 demo，输出 retrieval recall、context precision、evidence support、citation accuracy、abstention accuracy、RAG score、RAG gate、failed gates 和 needs rework。
29. 把一个 RAG 回答拆成 atomic claims，并逐条标注引用是否存在、相关、足以支持、版本正确、权限正确和能打开原文。
30. 构造 3 个 RAG 应该拒答的样本，分别覆盖无证据、权限不足和文档过期，并说明如何避免过度拒答。
31. 为 4 个 Agent 产品设计 agent product audit 表，字段至少包含 tasks total、tasks completed、tool calls、tool successes、high-risk actions、high-risk confirmed、recoveries needed、recoveries succeeded、state update required、state updates、observations total、observations used、trace complete、unauthorized actions、budget overruns、P95 latency、unit cost、eval ready、feedback loop 和 business metric。
32. 写一个 0 依赖 Agent 产品审计 demo，输出 task success、tool success、confirmation coverage、recovery rate、state update coverage、observation use rate、trace coverage、unauthorized rate、budget overrun rate、agent score、agent gate、failed gates 和 needs rework。
33. 对同一个企业任务分别设计 suggest、draft、semi-auto、approval 和 auto 五种自动化层级，说明每一层需要怎样的权限、验收器、人审、回滚和审计。
34. 画出一个 workflow-agent hybrid 产品流程，要求 workflow 负责接收、审批、提交、审计和回滚，Agent 负责分类、检索、草稿、诊断和异常恢复。
35. 构造 3 条 Agent 产品失败 trace，分别覆盖工具参数错误、未授权动作和 observation 未使用，并给出应触发的恢复策略或降级策略。
35A. 用纯 Python 写一个 Agent 落地事故审计 demo，输入 toy trace、expected tools、plan steps、tool args、permission、tool status、observation use、state update、confirmation、budget 和 backend status，输出 task success、plan feasibility、tool selection accuracy、argument validity、tool execution success、false completion、unauthorized action、budget overrun、trace completeness 和 failed gates。
35B. 构造 5 个 Agent bad case：工具失败但最终声称完成、参数结构合法但业务归属错误、高风险动作缺少确认、工具结果不可信边界失败、循环检索超预算，并为每个样本写出 root cause 和降级策略。
35C. 为一个最小 Coding Agent Harness 设计运行审计表，字段至少包含 task、context、model output、action、arguments、permission、confirmed、executed、execution result、observation used、state updated、trace fields、budget、final status 和 final verified。
35D. 写一个 0 依赖 harness loop 审计 demo，输出 parse valid rate、tool execution success、unauthorized execution rate、confirmation coverage、observation use rate、state update coverage、trace completeness、budget overrun rate、task success rate、root causes、failed gates 和 harness gate。
35E. 为 Agent Runtime 设计状态机，至少包含 RECEIVED_TASK、BUILDING_CONTEXT、CALLING_MODEL、PARSING_ACTION、WAITING_PERMISSION、EXECUTING_ACTION、OBSERVING_RESULT、UPDATING_STATE、COMPLETED、FAILED 和 CANCELLED，并标出非法状态转移。
35F. 写一个 0 依赖 runtime step 审计 demo，输入 toy step 的 session、task、phase、next phase、context tokens、context relevance、model adapter status、parse repair、execution result fields、error recovery、checkpoint、cancel / timeout 和 trace fields，输出 session isolation、phase transition valid、context budget ok、adapter success、parse recovery、execution standardized、error recovery、checkpoint coverage、cancel timeout handled、trace completeness、root causes、failed gates 和 runtime gate。
35G. 写一个 0 依赖 coding agent workflow 审计 demo，输入 toy trace 的 expected type、predicted type、required files、explored files、plan、edited files、validation commands、failed feedback、user modified files、high-risk actions 和 trace fields，输出 task type accuracy、exploration coverage、patch precision / recall、unrelated change rate、validation coverage、feedback use rate、user change violation rate、risky action protection、trace completeness、root causes、failed gates 和 workflow gate。
35H. 写一个 0 依赖 Tool Registry 审计 demo，输入 toy registry items 的 namespace、name、description flags、input schema、output schema、additional properties policy、executor、timeout、retry、concurrency、permission、data scope、risk level、side effect、confirmation、idempotency、version、schema hash、changelog、trace version captured、lifecycle、model visible、runtime executable、owner、SLO、eval dataset、regression result、provider projection 和 audit fields；输出 identity coverage、description quality、schema contract coverage、runtime metadata coverage、permission binding coverage、risk annotation coverage、version trace coverage、lifecycle policy pass rate、owner SLO coverage、eval binding coverage、provider projection readiness、registry audit completeness、failed tools、failed gates 和 tool registry gate；要求覆盖高风险邮件工具缺确认、重复泛化工具名 `search`、草稿占位工具 `query`、MCP 工具缺 namespace、deprecated 工具仍暴露给模型、内部定义绑定单一 provider 格式等 bad case。
35I. 写一个 0 依赖文件编辑审计 demo，输入 toy workspace、read/search/edit/apply patch 操作、workspace root、secret pattern、file type、generated marker、old context、read hash、user modified flag、diff summary 和 validation result，输出 workspace containment、secret file block rate、read truncation clarity、patch context match、patch apply success、concurrent edit protection、whole-file rewrite rejection、unrelated diff rate、rollback readiness、diff faithfulness、failed gates 和 file edit gate。
35J. 写一个 0 依赖命令安全审计 demo，输入 toy command requests 的 command、working directory、timeout、reason、declared expected effect、detected risk、network flag、permission decision、confirmation、exit code、stdout / stderr、timeout / cancelled、truncated、artifact ref、masked、observation used 和 trace fields，输出 command risk accuracy、auto execution precision、dangerous command block rate、confirmation coverage、effect consistency、network access control、timeout cancellation coverage、command output truncation、secret masking coverage、command trace completeness、unsafe executed、root causes、failed gates 和 command safety gate。
35K. 写一个 0 依赖上下文压缩审计 demo，输入 toy context candidates 的 source、trust、tokens、relevance、required、constraint、included、summary、supported、stale、compressed tool、keeps signal、artifact ref、untrusted boundary 和 memory write candidates，输出 token budget utilization、key context recall、context precision、constraint retention、task state completeness、summary faithfulness、stale summary rate、tool output compression fidelity、current diff coverage、trust boundary coverage、memory write safety、root causes、memory causes、failed gates 和 context builder gate。
35L. 写一个 0 依赖权限沙箱审计 demo，输入 toy permission requests 的 user、task、action、resource、risk、side effect、expected policy、actual decision、task scope、permission matrix row、secret、network、allowed domain、confirmation、dry run、sandbox、irreversible、untrusted source 和 trace fields，输出 least privilege coverage、permission matrix coverage、unauthorized action block rate、high-risk confirmation coverage、secret access block rate、network egress control、sandbox enforcement coverage、dry run coverage、audit log completeness、irreversible action protection、unsafe allowed、root causes、failed gates 和 permission sandbox gate。
35M. 写一个 0 依赖 Trace / Replay 审计 demo，输入 toy traces 的 trace id、session、task、workflow、status、started / ended、spans、parent span、span type、input / output summary、artifact refs、truncated flag、error type、root cause、model / prompt / tool / git / sandbox / permission versions、replay env snapshot、deterministic tools、sandbox flag、metrics、eval labels 和 validation result，输出 trace schema completeness、span schema completeness、span tree validity、timeline validity、artifact reference coverage、version capture coverage、replay readiness rate、privacy masking coverage、error attribution coverage、final status consistency、metric export coverage、eval export coverage、root causes、failed gates 和 trace replay gate。
35N. 写一个 0 依赖 Evaluation Harness 审计 demo，输入 toy eval runs 的 task id、bucket、weight、environment ready、checksum、sandbox reset、validators、success、partial score、tests passed、expected files、modified files、trace completeness、failure taxonomy、baseline success、baseline / candidate budget、permission / tools fairness、unsafe attempt / executed、cost、repeated outcomes、dataset / agent / model / prompt / tool / env / permission versions 和 report fields，输出 dataset bucket coverage、environment reproducibility、validator coverage、weighted task success、partial success、diff scope safety、trace coverage、baseline fairness、regression pass rate、unsafe execution rate、cost budget pass rate、flaky task rate、version capture、report completeness、root causes、failed gates 和 evaluation harness gate。
35O. 写一个 0 依赖 Claude Code 架构审计 demo，输入 toy architecture components 的 name、group、documented、present、governed、required 和 risk 字段，输出 architecture evidence coverage、required module coverage、access governance、core loop governance、permission control、state recovery、extension governance、observability、architecture eval readiness、high-risk surface governance、root causes、failed gates 和 Claude Code architecture gate；要求显式区分公开证据、系统设计推断和未经确认的内部实现猜测。
35P. 写一个 0 依赖 OpenCode 架构审计 demo，输入 toy runtime capabilities 的 name、group、documented、covered、governed、stateful、required 和 risk 字段，输出 public docs coverage、open runtime module coverage、config governance、agent permission isolation、tool permission binding、extension governance、server API governance、snapshot recovery、architecture eval readiness、high-risk surface governance、root causes、failed gates 和 OpenCode architecture gate；要求解释为什么开放 runtime 的 MCP、custom tools、plugins、server、SDK 和 remote config 必须同时审计扩展收益和风险边界。
35Q. 写一个 0 依赖主流 Coding Agent 横向对比审计 demo，输入 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent 和 OpenHands 的 toy capability table，字段至少包含 product surfaces、context sources、tools、permissions、edit / recovery、eval readiness、governance 和 risk surfaces；按 daily IDE、enterprise platform、research benchmark 三种场景输出 product surface coverage、context source coverage、tool execution coverage、permission governance coverage、edit recovery coverage、agent eval readiness、enterprise governance coverage、cross-agent risk governance、ranked systems、weak spots 和 coding agent comparison gate；要求解释为什么同一个系统在不同场景下权重和门禁会不同。
35R. 写一个 0 依赖 MCP/A2A 协议集成审计 demo，输入 toy capability table，字段至少包含 name、protocol、kind、connected、namespaced、schema_valid、permission、risk、approval、auth_scoped、context_tokens、output_limited、trust_boundary、A2A lifecycle states、trace fields、replay mode、version_captured、can_cancel 和 error_mapped；输出 capability discovery coverage、namespace isolation coverage、protocol schema validity、protocol permission binding、high-risk approval coverage、context output budget coverage、external trust boundary coverage、A2A lifecycle coverage、protocol trace coverage、protocol replay readiness、protocol version capture、failure handling、root causes、failed gates 和 protocol integration gate；要求解释为什么 MCP/A2A 接入必须同时治理互联、权限、上下文、trace、replay 和版本。
35S. 写一个 0 依赖 Agent Harness 系统设计审计 demo，输入 toy module table，字段至少包含 name、group、present、contract、stateful、permission、context_budget、isolated、trace、replay、eval、recovery、version 和 governance；输出 runtime module coverage、interface contract coverage、agent state machine coverage、permission integration coverage、context control coverage、execution isolation coverage、system trace coverage、system replay readiness、system eval readiness、recovery coverage、harness version capture、enterprise governance readiness、missing required modules、root causes、failed gates 和 harness system gate；要求解释为什么模块图完整不代表生产级可上线。
35T. 写一个 0 依赖 Harness 实战坑审计 demo，输入 toy incidents，字段至少包含 id、root cause、triaged、expected / actual permission、high risk、approval、context tokens / limit、output limited、edit risk / checked、command risk / timeout / cwd / env、untrusted boundary、trace fields、replay ready、eval deterministic、version captured、steps、step limit、cost、cost limit、repeated call 和 environment fingerprint；输出 triage coverage、permission safety、context budget pass、edit safety、command safety、prompt injection boundary、trace completeness、replay readiness、eval determinism、cost loop control、environment parity、failed gates、root causes 和 harness pitfall gate；要求解释为什么 Coding Agent 实战坑必须放进统一事故审计表。
35U. 写一个 0 依赖工具参数校验与修复审计 demo，输入 toy tool calls 的 raw arguments、schema、business rules、evidence map、model repair candidates、confirmation、side effect 和 idempotency key，覆盖 JSON parse 失败、schema 类型 / 范围 / 额外字段错误、低风险 normalization、业务时间错误、关键参数缺 evidence、高风险金额 / 路径阻断、模型自修复成功、模型自修复耗尽、缺幂等键；输出 parse success rate、schema pass rate、business pass rate、argument evidence coverage、safe normalization rate、repair success rate、clarification coverage、unsafe argument block rate、retry budget pass rate、idempotency key coverage、repair trace completeness、failed cases、failed gates 和 argument repair gate；要求解释为什么参数修复不能自动改金额、收件人、路径和权限范围。
35V. 写一个 0 依赖 Tool Result 上下文审计 demo，输入 toy tool results 的 expected tool_call_id、actual tool_call_id、status、rendered status、source metadata、trust level、raw fields、projected fields、sensitive fields、external instruction flag、token budget、compressed summary、key facts、conflict flag、citation support、expires_at / freshness、memory write decision；输出 result ID alignment rate、safe projection rate、redaction coverage、context budget pass rate、source metadata coverage、injection containment rate、error status fidelity、compression fidelity、conflict labeling rate、citation support rate、freshness pass rate、memory boundary pass rate、failed cases、failed gates 和 tool result context gate；要求解释为什么工具结果不能原样塞回上下文，且 timeout 不能被当成 empty result。
35W. 写一个 0 依赖工具失败恢复审计 demo，输入 toy tool failures 的 user request、tool name、side effect、error code、expected class、actual class、retry attempts、retry budget、idempotency key、unknown state、fallback、circuit state、timeout cancelled、handoff context、user-visible report 和 trace fields；输出 failure detection coverage、error classification accuracy、retry policy precision、retry budget pass rate、idempotency protection rate、unknown state escalation rate、fallback honesty rate、circuit breaker containment、timeout cancellation coverage、human handoff readiness、user-visible error clarity、failure trace completeness、failed cases、failed gates 和 tool failure recovery gate；要求覆盖邮件超时未知状态盲目重试、缓存降级假装实时、超时任务未取消、失败后编造答案和 trace 缺字段等 bad case。
35X. 写一个 0 依赖工具调用评估审计 demo，输入 toy traces 的 user request、should call、expected tools、actual tools、expected args、actual args、argument source、execution status、observation used、task state、safety block、cost / latency budget 和 regression label；输出 tool call recall、tool call precision、tool selection accuracy、tool set precision / recall、argument completeness、argument value accuracy、argument source coverage、execution success rate、observation use correctness、task success rate、safe failure rate、unsafe failure rate、safety block rate、cost latency budget pass rate、regression pass rate、failed cases、failed gates 和 tool calling eval gate；要求覆盖私有数据漏调用、知识问题误调用、高风险退款缺确认、日期参数错误、内部政策工具选错、工具超时安全失败、工具结果被忽略、prompt injection 被拦截和成本延迟超预算等 bad case。
35Y. 写一个 0 依赖 Tool Router 审计 demo，输入 toy router decisions 的 user request、scenario、expected tools、actual candidate tools、forbidden tools、max tools、expected / actual tool choice mode、expected / actual forced tool、expected / actual clarification decision、expected / actual parallel policy、provider supported modes、cost budget 和 trace fields；输出 scenario filter pass rate、permission filter pass rate、risk filter pass rate、candidate recall、candidate precision、candidate size pass rate、clarification accuracy、tool choice mode accuracy、forced tool accuracy、router parallel safety、provider capability compatibility、cost budget pass rate、router trace completeness、failed cases、failed gates 和 tool router gate；要求覆盖高风险退款工具过早暴露、普通用户导出全部客户、正确工具未进入候选、候选集过宽、缺参时错误 forced tool、provider 不支持 required 却仍输出 required 等 bad case。
35Z. 写一个 0 依赖 Tool Executor 审计 demo，输入 toy executor traces 的 expected accept、actual accept、schema result、permission result、expected / actual execution mode、async job tracking、timeout / cancelled、side effect、confirmation、idempotency key、duplicate prevention、unknown state、retry policy、structured result、error mapping 和 trace fields；输出 execution request validity、schema validation pass rate、permission enforcement pass rate、execution mode accuracy、async completion tracking、timeout cancellation coverage、idempotency protection rate、unknown state escalation rate、retry safety rate、side effect confirmation coverage、structured result coverage、executor trace completeness、failed cases、failed gates 和 tool executor gate；要求覆盖超时未取消、有副作用缺幂等、未知状态盲目重试、异步任务缺状态查询、结果未结构化和 trace 缺字段等 bad case。
35AA. 写一个 0 依赖工具权限模型审计 demo，输入 toy permission traces 的 expected / actual allow、trusted context、tenant isolation、tool permission、object permission、returned fields、allowed fields、sensitive fields、action context、prompt injection、least privilege、token audience、cache revocation、error disclosure 和 audit fields；输出 authorization decision accuracy、trusted context injection、tenant isolation pass rate、tool permission enforcement、object permission accuracy、field projection safety、action context policy accuracy、prompt injection block rate、least privilege coverage、token audience validation、revocation cache safety、error disclosure safety、permission audit completeness、failed cases、failed gates 和 tool permission gate；要求覆盖相信模型传入 `tenant_id`、敏感字段泄露、缺确认退款、prompt injection 触发管理员工具、服务账号过宽、MCP token audience 错误、权限撤销后缓存仍放行、资源枚举错误泄露和审计字段缺失等 bad case。
35AB. 写一个 0 依赖工具安全审计 demo，输入 toy tool security traces 的 prompt injection、untrusted content、unauthorized attempt、sensitive data、external transfer、dangerous action、SSRF、SQL、path、shell、data flow、audit fields、alert 和 regression label；输出 prompt injection containment、untrusted content isolation、unauthorized access block rate、sensitive data protection、external transfer gate、dangerous action confirmation、SSRF block rate、SQL risk block rate、path traversal block rate、shell sandbox enforcement、data flow policy pass rate、audit completeness、alert completeness、safety eval regression pass rate、failed cases、failed gates 和 tool security gate；要求覆盖间接网页注入放行、跨租户导出放行、敏感字段外发、metadata SSRF 放行、SQL 敏感读、路径逃逸、shell 读取 secret、未确认删除和审计字段缺失等 bad case。
35AC. 写一个 0 依赖工具 trace / replay 审计 demo，输入 toy traces 的 trace fields、ID integrity、span tree、version fields、argument lineage、permission fields、tool result fields、PII masking、audit fields、replay readiness、side-effect live replay block、metric export、alert owner 和 eval linkage；输出 trace schema completeness、ID tree integrity、version capture coverage、argument lineage coverage、permission trace completeness、tool result trace completeness、privacy masking coverage、audit event completeness、replay readiness rate、side effect replay safety、metric export coverage、alert owner coverage、eval linkage coverage、failed cases、failed gates 和 trace replay gate；要求覆盖 span tree 断链、trace id 不一致、版本缺失、参数 lineage 缺失、权限缺 reason、结果未投影、PII 未脱敏、审计缺 actor、有副作用工具 live replay、指标缺失、告警无 owner 和 eval 链接缺失等 bad case。
35AD. 写一个 0 依赖工具版本发布审计 demo，输入 toy release candidates 的 spec fields、schema compatibility、description risk、output compatibility、permission policy change、version matrix、offline eval、canary routing、quality guard、safety guard、cost latency、rollback fields 和 lifecycle fields；输出 spec lint pass rate、schema backward compatibility、description behavior guard、output compatibility、permission policy compatibility、version matrix capture、offline eval pass rate、canary routing stability、canary quality guard、canary safety guard、cost latency guard、rollback readiness、lifecycle coverage、failed cases、failed gates 和 tool release gate；要求覆盖新增可选字段被过度填充、新增必填字段破坏兼容、description 扩大边界、输出字段重命名、权限收紧缺通知、版本矩阵缺失、离线评估失败、灰度路由不 sticky、延迟成本回归、高风险动作回归、回滚缺旧 spec、状态迁移缺策略和 deprecated 无 sunset 等 bad case。
35AE. 写一个 0 依赖企业工具平台审计 demo，输入 toy platform cases 的 module、interface contract、registry snapshot、router policy、executor timeout / idempotency / sandbox、permission fail closed、safety guard、trace / audit、eval golden traces、release rollback、tenant isolation、provider / MCP adapter、HA signed snapshot 和 admin ops 权限；输出 platform module coverage、platform interface contract coverage、registry readiness、router governance coverage、executor safety coverage、platform permission integration coverage、platform safety guard coverage、platform trace audit coverage、platform eval service coverage、platform release governance coverage、platform tenant isolation coverage、platform provider adapter coverage、high availability readiness、admin ops governance coverage、failed cases、failed gates 和 enterprise tool platform gate；要求覆盖 MCP 工具未审核、provider capability 未建模、后台权限过宽、trace / audit 不完整、cache 缺租户 key、发布无回滚、eval 缺 golden trace 和 Registry 快照未签名等 bad case。
35AF. 写一个 0 依赖 MCP 背景审计 demo，输入 toy MCP background cases 的 host 数、external system 数、capability types、是否覆盖 tools / resources / prompts、discovery、schema contract、host / client / server 边界、本地上下文控制、跨客户端复用、企业治理接入、trace / eval readiness 和 MCP / A2A 区分；输出 direct integration count、MCP integration count、integration reduction、capability model coverage、context object coverage、discovery standardization、MCP schema contract coverage、host server boundary clarity、local context control、cross client reuse、governance boundary clarity、MCP trace eval readiness、MCP A2A distinction、failed cases、failed gates 和 MCP background gate；要求覆盖只做 function calling、只包装 HTTP API、平台插件锁定、server 直接连模型、缺 resources / prompts、本地上下文无权限、把 MCP 说成 A2A、无 trace / eval 等 bad case。
35AG. 写一个 0 依赖 MCP 基本概念审计 demo，输入 toy MCP concept cases 的 host policy、client / server boundary、server capabilities、tool schema / result、resource URI / metadata、prompt arguments / review、lifecycle negotiation、transport policy、roots boundary、sampling control、host filtering、trace / eval mapping 和 context budget；输出 host policy ownership、MCP client server boundary、server capability declaration、MCP tool schema and result、MCP resource URI metadata、MCP prompt argument review、MCP lifecycle negotiation、MCP transport policy、MCP roots boundary、MCP sampling control、host capability filtering、MCP context budget、failed cases、failed gates 和 MCP concept gate；要求覆盖 Client 被当成 LLM、Server 直接连模型、tool 缺 schema、resource 被设计成任意动作、prompt 自动可信、缺 lifecycle、transport 无认证、roots 过宽、sampling 无审查、Host 不过滤能力等 bad case。
35AH. 写一个 0 依赖 MCP Server 最小实现审计 demo，输入 toy server cases 的 metadata、capabilities、tool registry、strict input schema、handler、argument validation、structured result、structured error、resource scope、prompt review、transport policy、Host connection、safety baseline 和 trace fields；输出 metadata readiness、capability declaration、tool registry readiness、strict schema coverage、handler execution coverage、argument validation coverage、structured result coverage、structured error coverage、resource scope coverage、prompt template coverage、transport policy coverage、host connection readiness、safety baseline coverage、trace readiness、failed cases、failed gates 和 MCP server gate；要求覆盖缺 schema、允许额外字段、错误未结构化、resource 路径逃逸、prompt 自动可信、远程 server 无认证、trace 缺失等 bad case，并说明为什么最小 MCP Server 不是裸 handler。
35AI. 写一个 0 依赖 MCP Tool / Function Calling 对比审计 demo，输入 toy compare cases 的 protocol layer、tool discovery、capability scope、execution boundary、projection mapping、adapter separation、lifecycle/version、governance registry、security boundary、use-case selection、error surface 和 latency/availability trade-off；输出 protocol layer clarity、tool discovery boundary、MCP capability scope coverage、MCP execution boundary clarity、MCP projection mapping coverage、adapter separation coverage、MCP lifecycle version awareness、MCP governance registry import、MCP security boundary enforcement、MCP use case selection fit、MCP error surface separation、MCP latency availability tradeoff、failed cases、failed gates 和 MCP function calling gate；要求覆盖把 MCP 等同于 Function Calling、认为 MCP 替代 Function Calling、Server 直接连模型、发现工具后自动信任、忽略 resources/prompts、缺 Registry 导入、adapter 混杂、忽略生命周期和忽略延迟可用性等 bad case。
35AJ. 写一个 0 依赖 MCP Resources 审计 demo，输入 toy resources 和 toy cases，字段至少包含 URI、metadata、kind、tenant、allowed roots、sensitivity、trust level、last modified、etag、size、allowed fields、content、citation、template、subscription 和 trace fields；实现 `resources/list`、`resources/read`、`resources/templates/list` 和最小 subscription smoke test；输出 resource URI validity、resource metadata completeness、resource list filtering boundary、resource read scope enforcement、MCP roots containment、MCP resource permission enforcement、MCP resource field projection safety、resource context budget control、resource citation traceability、untrusted resource labeling、resource freshness version awareness、resource template boundary、resource subscription awareness、resource trace eval readiness、failed cases、failed gates 和 MCP resource gate；要求覆盖路径逃逸、密钥文件、跨租户数据库记录、字段泄露、大资源超预算、metadata 缺失、网页注入未标注、过期资源和缺 citation 等 bad case。
35AK. 写一个 0 依赖 MCP Prompts 审计 demo，输入 toy prompt server、prompt definitions 和 toy cases，字段至少包含 prompts capability、name、title、description、arguments、message role、template、dependencies、version、owner、risk、approval、eval dataset、allowed roles、list changed 和 trace fields；实现 `prompts/list`、`prompts/get`、required argument validation、参数转义包装、role boundary、permission filtering 和 list changed smoke test；输出 prompt capability declaration、prompt discovery coverage、prompt argument schema coverage、prompt required argument validation、prompt rendering safety、prompt role boundary enforcement、prompt dependency alignment、prompt version governance、prompt eval binding、prompt permission enforcement、prompt injection containment、prompt user control、prompt list change awareness、prompt trace readiness、failed cases、failed gates 和 MCP prompt gate；要求覆盖缺 prompts capability、prompt 不可发现、缺必填参数、role 提权、参数注入未转义、依赖缺失、未版本化、无 eval、未授权可见、用户不可见来源、忽略 list changed 和 trace 缺失等 bad case。
35AL. 写一个 0 依赖 MCP 权限、安全和本地沙箱审计 demo，输入 toy server registry、token、roots、file request、network request、shell request、data flow、tenant context、confirmation 和 trace fields；实现 server allowlist / 签名检查、token audience / scope 检查、scope 最小化检查、roots containment、敏感文件阻断、SSRF 阻断、shell sandbox 检查、prompt injection data boundary、high risk confirmation、sensitive data flow control、local credential isolation、tenant isolation、server supply chain governance、trace 和安全 eval 覆盖率；输出 server connection governance、authorization token binding、scope minimization、roots sandbox containment、file secret blocking、network SSRF protection、MCP shell sandbox enforcement、prompt injection data boundary、high risk confirmation、sensitive data flow control、local credential isolation、MCP tenant isolation、server supply chain governance、MCP security trace readiness、MCP security eval coverage、failed cases、failed gates 和 MCP security gate；要求覆盖未审核 Server、token audience 错误、scope 过宽、路径逃逸、密钥文件、SSRF、无沙箱 shell、prompt injection 触发危险工具、敏感数据外发、跨租户读取、trace 缺失和安全 eval 缺失等 bad case。
35AM. 写一个 0 依赖 MCP 与 IDE、知识库、数据库、浏览器、终端集成审计 demo，输入 toy capability registry、namespace、IDE context candidates、knowledge resources、database query tools、browser actions、terminal tools、cross-server data flows、approval records、output projection、trace fields 和 eval labels；实现 capability registration、namespace isolation、IDE context routing、knowledge citation traceability、database read-only / parameterized / field projection 检查、browser untrusted resource / action preview / confirmation 检查、terminal allowlist / sandbox / env filter 检查、context budget、cross-server data flow、high-risk approval、output projection、trace 和 eval 覆盖率；输出 MCP capability registration coverage、MCP namespace isolation coverage、MCP IDE context routing、MCP knowledge citation traceability、MCP database query governance、MCP browser action governance、MCP terminal sandbox governance、MCP integration context budget control、MCP cross server data flow control、MCP high risk approval coverage、MCP output projection coverage、MCP integration trace readiness、MCP integration eval coverage、failed cases、failed gates 和 MCP integration gate；要求覆盖未注册能力、namespace 冲突、IDE patch 未预览、知识库缺 citation、数据库自由 SQL、浏览器高风险提交缺确认、终端无沙箱、上下文超预算、跨 Server 外发、输出未投影、trace 缺失和 eval 缺失等 bad case。
35AN. 写一个 0 依赖 A2A 背景审计 demo，输入 toy Agent Card、remote agent discovery、task delegation cases、TaskState 序列、Message parts、Artifact refs、context items、auth / permission、failure policy、trace fields 和 eval labels；实现 agent card completeness、agent discovery readiness、task delegation contract、A2A task lifecycle coverage、A2A message structure coverage、A2A artifact reference coverage、A2A context boundary control、A2A permission boundary、A2A MCP distinction、A2A failure handling coverage、A2A trace readiness 和 A2A eval coverage；输出 failed cases、failed gates 和 A2A background gate；要求覆盖未知 Agent、能力不匹配、缺 task id、状态跳跃、input_required 缺消息轮次、Artifact 内联、上下文过度共享、缺认证、把 A2A 混成 MCP 工具调用、不能取消、trace 缺字段和 eval 缺标签等 bad case。
35AO. 写一个 0 依赖 Agent Card 服务发现审计 demo，输入 toy Agent Cards、skills、supported interfaces、security requirements、public / extended card 标记、version / cache、discovery cases、routing expectations、trace fields 和 eval labels；实现 agent card field completeness、agent skill declaration quality、supported interface readiness、agent card security coverage、agent card version cache readiness、agent discovery match quality、agent routing decision quality、extended agent card control、agent card trace readiness 和 agent card eval coverage；输出 selected candidates、failed cases、failed gates 和 agent card gate；要求覆盖字段缺失、skill 缺 examples、HTTP interface、不支持目标 output mode、scope 不足、未知 skill、extended card 误披露、版本缓存缺失和 trace 缺字段等 bad case。
35AP. 写一个 0 依赖 A2A 任务委派生命周期审计 demo，输入 toy Task、Message、Artifact、TaskState 序列、error、retry、cancel、permission scope、parallel child results、trace fields 和 eval labels；实现 A2A task contract coverage、state transition validity、input-required handling、message structure coverage、artifact metadata coverage、error semantics coverage、retry idempotency coverage、cancellation coverage、delegation permission boundary、parallel aggregation readiness、task trace readiness 和 task eval coverage；输出 smoke、metrics、failed cases、failed gates 和 A2A task gate；要求覆盖 happy path、状态跳跃、input-required 未处理、Artifact 缺 hash / classification、failed 缺结构化 error、越权委派、写操作重试不幂等、取消未进入 canceled、并行子任务未完成、并行结果冲突、trace 缺字段和 eval 缺标签等 bad case。
35AQ. 写一个 0 依赖 A2A 消息边界审计 demo，输入 toy Message、Part、metadata、source / trust label、context policy、recipient、claim、summary、trace fields 和 eval labels；实现 message contract coverage、part typing coverage、source trust labeling、instruction / data separation、minimal context coverage、reference over copy coverage、context policy enforcement、sensitive redaction coverage、claim grounding coverage、summary constraint retention、message trace readiness 和 message eval coverage；输出 smoke、metrics、failed cases、failed gates 和 A2A message gate；要求覆盖 happy path、缺 `contextId`、Part 缺语义类型、缺 source、外部网页伪装成 instruction、system prompt 被复制、敏感大对象未引用、禁止转发却继续转发、收件人不在 allowed recipients、hypothesis 被升级成 fact、摘要丢约束、上下文超预算、trace 缺字段和 eval 缺标签等 bad case。
35AR. 写一个 0 依赖 A2A/MCP 分工审计 demo，输入 toy capability cases 的 expected protocol、actual protocol、autonomy、workload、lifecycle、discovery、context owner、permission separation、output boundary、trace fields、version fields 和 eval label；实现 protocol classification、tool agent boundary、autonomy fit、lifecycle placement、discovery split、context ownership、permission separation、result artifact boundary、trace linkage 和 version eval coverage；输出 smoke、metrics、failed cases、failed gates 和 A2A MCP boundary gate；要求覆盖把工具伪装成 Agent、把复杂 Agent 塞成 MCP Tool、跨 Agent 共享 MCP 工具权限、缺 Agent Card、缺 MCP registry、上下文所有权错误、Tool Result / Artifact 混淆、trace 缺字段和版本 eval 缺失等 bad case。
35AS. 写一个 0 依赖跨 Agent 安全审计 demo，输入 toy A2A task traces 的 user identity、tenant、caller agent、assignee agent、service id、OBO scopes、task scopes、requested scopes、Agent allowlist、Agent Card signature、service verification、data classification、detail level、context policy、forwarded target、redelegation approval、high-risk flag、confirmation、MCP tool requirements、result / artifact policy、audit fields、trust level 和 eval label；实现 identity chain coverage、OBO scope binding、delegation allowlist、permission attenuation、context policy enforcement、redelegation control、high-risk confirmation、MCP tool permission binding、result release control、audit trace completeness、trust evidence verification 和 tenant isolation；输出 smoke、metrics、failed cases、failed gates 和 cross-agent security gate；要求覆盖身份缺失、OBO scope 扩大、第三方 Agent 未授权、权限衰减失败、上下文转发失控、继续委派未审批、高风险写操作缺确认、MCP 工具越权、Artifact 泄露、audit 缺字段、低信任结果未验证和跨租户泄露等 bad case。
35AT. 写一个 0 依赖多 Agent 失败模式审计 demo，输入 toy root task traces 的 root goal、final goal、hard constraints、delegation graph、max depth、visited agents、conflict / arbitration、claim type、evidence、confidence、limitations、duplicate tool calls、cost budget、state events、artifact writers、trace fields、data policy、agent count、termination condition 和 eval label；实现 delegation loop control、role conflict arbitration、hallucination propagation containment、context drift control、duplicate work budget、state consistency、artifact conflict control、accountability trace、policy chain enforcement、collaboration fit、termination handoff readiness 和 failure eval coverage；输出 smoke、metrics、failed cases、failed gates 和 multi-agent failure gate；要求覆盖循环委派、深度溢出、角色冲突无仲裁、hypothesis 被升级为 fact、上下文目标漂移、重复查询成本超预算、状态事件乱序、多 Agent 抢写 Artifact、trace 缺字段、安全策略绕过、简单任务过度协作和 eval 标签缺失等 bad case。
35AU. 写一个 0 依赖 A2A 系统设计审计 demo，输入 toy design answers 的 requirements、modules、protocol models、Task fields、Agent Card fields、Message fields、Artifact fields、discovery filters、policy-before-route、states、event seq、context controls、permission model、MCP/A2A boundary、artifact governance、trace fields、failure controls、evals、scale controls 和 eval label；实现 requirement clarification、architecture module coverage、protocol contract coverage、discovery routing governance、runtime state readiness、context boundary control、permission model coverage、MCP/A2A boundary clarity、artifact governance coverage、trace audit completeness、failure handling coverage、eval observability coverage 和 scalability idempotency readiness；输出 smoke、metrics、failed designs、failed gates 和 A2A system design gate；要求覆盖缺需求澄清、只画 Agent、不讲协议对象、LLM 直接选 Agent、旧状态机、上下文转发失控、权限模型缺失、A2A/MCP 混淆、Artifact 内联、trace/audit 缺字段、失败处理缺失、eval 缺失和扩展性 / 幂等缺失等 bad case。
35AV. 写一个 0 依赖 Skill 定义审计 demo，输入 toy skill manifests 的 manifest fields、task goal、scope、components、tool count、task strategy、instructions、resources、prompts、workflow steps、permissions、config keys、eval spec、versioning、lifecycle、boundary、installable、auditable、progressive loading 和 eval label；实现 manifest metadata、task goal clarity、bundle completeness、tool skill boundary、instruction strategy、resource prompt grounding、workflow readiness、permission safety、configuration reuse、eval coverage、lifecycle governance、product install governance 和 progressive disclosure；输出 smoke、metrics、failed skills、failed gates 和 skill definition gate；要求覆盖缺 manifest、万能办公 Skill、单个 read_file 伪装成 Skill、只有工具列表、缺说明、缺资源和 prompt、缺 workflow、隐藏权限、缺配置、缺 eval、缺生命周期、不可安装不可审计、一次性加载全部资料等 bad case。
35AW. 写一个 0 依赖五层工具生态边界审计 demo，输入 toy capability items 的 expected abstraction、actual abstraction、fields、granularity、permission layer、eval layer、audit fields、schema、workflow、task goal、install、high risk、approval、alias documented、lifecycle 和 eval label；实现 required field coverage、abstraction classification、granularity boundary、action trace、tool contract、workflow control、skill bundle、plugin packaging、permission layering、eval layering、audit trace layering、naming alias documentation、governance lifecycle、high risk approval 和 eval ready；输出 smoke、metrics、failed items、failed gates 和 ecosystem boundary gate；要求覆盖把 Tool 误叫 Action、Tool 缺 schema、Workflow 隐藏删除动作、单个 API 伪装成 Skill、Plugin 缺安装治理、权限层混淆、评估层混淆、别名未说明、审计字段缺失和 eval 缺失等 bad case。
35AX. 写一个 0 依赖 Skill Manifest 审计 demo，输入 toy manifests 的 identity、description、capabilities、inputs、outputs、tools、resources、prompts、workflow、permissions、configuration、safety、eval、examples、lifecycle、audit 和 high-risk 标记；实现 identity metadata、description quality、capability granularity、input contract、output contract、tool dependency purpose、resource prompt binding、workflow readiness、permission least privilege、configuration schema、safety policy、eval gate、examples trigger coverage、version lifecycle 和 audit readiness；输出 smoke、metrics、failed manifests、failed gates 和 skill manifest gate；要求覆盖缺身份、描述过泛、capability 太泛、缺 required input、输出不稳定、tool 缺 purpose、资源 / prompt 缺失、workflow 缺审批、权限过宽、配置无类型、安全策略缺失、eval 缺失、examples 缺负例、生命周期和审计字段缺失等 bad case。
35AY. 写一个 0 依赖 Skill 生命周期审计 demo，输入 toy lifecycle cases 的 review、install、enabled scope、permission delta、configuration、running task policy、compatibility、rollout、metrics、rollback、dependencies、audit、emergency suspend、uninstall retention、eval 和 update policy；实现 review before publish、install approval、enable scope control、permission reapproval、configuration versioning、running task policy、compatibility check、rollout guard、rollback readiness、dependency versioning、audit log completeness、emergency suspend readiness、uninstall retention、eval monitoring 和 update policy accuracy；输出 smoke、metrics、failed cases、failed gates 和 skill lifecycle gate；要求覆盖发布未审、安全权限未审批、启用范围过宽、新权限未重审、配置未版本化、禁用无运行任务策略、输出破坏兼容、无灰度全量发布、灰度安全回归、回滚计划缺失、依赖未 pin、审计字段缺失、紧急下架能力不足、卸载删除 audit、major 版本自动更新等 bad case。
35AZ. 写一个 0 依赖 Skill Marketplace 审计 demo，输入 toy listings 的 catalog、search index、detail page、review steps、permissions、risk level、install approval、rating signals、operations metrics、duplicate policy、admin view、developer console、security center、recommendation policy、lifecycle fields、owner maintenance 和 audit events；实现 catalog metadata、search discovery、detail page completeness、review workflow、permission transparency、install approval flow、rating quality balance、operations metrics、duplicate capability governance、admin permission view、developer console readiness、security center readiness、recommendation policy safety、lifecycle visibility、owner maintenance 和 portal audit trace；输出 smoke、metrics、failed listings、failed gates 和 skill marketplace gate；要求覆盖目录缺 owner、搜索缺 examples、详情页过泛、审核缺 security、权限 reason 隐藏、高风险自助安装、只看星级、运营指标缺失、重复能力无治理、管理员视图缺外部传输、开发者控制台缺 eval、安全中心缺紧急下架、不安全推荐、owner 失效和审计事件缺失等 bad case。
35BA. 写一个 0 依赖 Tool / Skill 质量安全审计 demo，输入 toy audit cases 的 tool schema、argument checks、execution controls、output contract、side effect controls、skill eval、task success、citation accuracy、unsupported claim、completeness、format compliance、offline eval、online monitoring、permission review、data security、prompt injection tests、high-risk controls、supply chain、human review、regression gate 和 audit fields；实现 tool schema clarity、argument validation、execution reliability、output stability、side effect control、skill task quality、factual grounding、completeness format、offline eval coverage、online monitoring、permission least privilege、data security review、prompt injection resilience、high risk action control、supply chain governance、human review readiness、regression release gate 和 audit trace readiness；输出 smoke、metrics、failed cases、failed gates 和 quality safety gate；要求覆盖 schema 模糊、业务校验缺失、无 timeout、输出无 version、副作用无 idempotency / dry run、任务成功率低、引用缺失、格式不完整、离线 eval 缺 adversarial / regression、线上监控缺安全拦截、权限过宽、数据泄露、prompt injection、高风险动作缺确认、依赖未 pin、人工审核缺失、回归门禁缺阻断和审计字段缺失等 bad case。
35BB. 写一个 0 依赖 Tool / Skill 开发者体验与文档规范审计 demo，输入 toy developer docs 的 quickstart、hello world、scaffold、tests、eval stub、input / output / error schema、happy / error / permission examples、local run / lint / eval、trace / replay、review feedback、安全文档、migration docs、CLI / SDK 一致性、文档版本、owner、support channel 和 portal path；实现 quickstart path clarity、scaffold completeness、schema contract、documentation example coverage、local debug readiness、trace replay readiness、review feedback actionability、security documentation coverage、migration documentation coverage、CLI SDK consistency、documentation freshness、owner maintenance 和 portal path clarity；输出 smoke、metrics、failed docs、failed gates 和 DX documentation gate；要求覆盖 quickstart 抽象不可运行、脚手架缺测试、schema 模糊、只给 happy path、本地调试缺失、trace / replay 缺失、审核反馈不可定位、安全文档缺失、迁移文档缺失、CLI / SDK 与文档漂移、文档过期、owner 缺失和门户路径混乱等 bad case。
35BC. 写一个 0 依赖可组合 Workflow 审计 demo，输入 toy workflow definitions 的 steps、dependencies、conditions、inputs、outputs、workflow states、permissions、data policy、side effects、idempotency keys、retry policy、compensation、approval points、cancellation policy、trace fields、evals 和 version compatibility；实现 workflow graph validity、dependency acyclicity、workflow IO contract、condition determinism、workflow state coverage、workflow permission boundary、workflow data flow policy、workflow idempotency coverage、retry safety、compensation readiness、workflow human approval coverage、workflow trace replay readiness、workflow eval coverage 和 version compatibility；输出 smoke、metrics、failed workflows、failed gates 和 workflow gate；要求覆盖未知依赖、循环依赖、输入输出缺失、条件分支写成模型自行决定、缺 cancelled 状态、权限过宽、confidential artifact 被转发、有副作用步骤缺 idempotency、补偿缺失、高风险发送缺人工审批、取消策略缺失、trace 字段缺失、eval 缺回归 / 安全样本和版本兼容缺失等 bad case。
35BD. 写一个 0 依赖 Prompt Injection 防御审计 demo，输入 toy protocol traces 的 user task、untrusted content、source metadata、trust level、taint、candidate action、tool risk、policy gate、sensitive fields、confirmation、sandbox、projected output、RAG source policy、multi-agent derived_from、trace fields 和 eval labels；实现 instruction data separation、source trust labeling、taint propagation、policy pre-tool gate、risky tool isolation、untrusted action blocking、sensitive data control、high-risk confirmation、sandbox enforcement、output redaction projection、RAG source boundary、multi-agent taint propagation、trace audit readiness 和 regression eval coverage；输出 smoke、metrics、unsafe allowed rate、failed cases、failed gates 和 prompt injection defense gate；要求覆盖直接注入被当成指令、工具结果缺来源、摘要后 taint 丢失、模型自判安全、高风险工具进入普通候选集、不可信网页触发外发、敏感字段外传、缺高风险确认、shell 无沙箱、输出未脱敏投影、RAG 文档改变引用策略、跨 Agent taint 丢失、trace 缺字段和 eval 缺间接注入 / 工具结果注入 / 回归样本等 bad case。
35BE. 写一个 0 依赖工具输出可信度审计 demo，输入 toy tool results、RAG chunks、artifacts、claims、citations、source metadata、trust level、freshness、access scope、evidence chain、conflicts、summaries、multi-agent messages、trace fields 和 eval labels；实现 source metadata coverage、trust level coverage、freshness version coverage、access scope disclosure、citation binding、citation support accuracy、evidence chain completeness、claim type calibration、confidence limitation disclosure、conflict resolution readiness、summary provenance retention、RAG chunk citation accuracy、multi-agent evidence propagation、provenance trace replay、citation forgery block 和 tool output trust eval coverage；输出 smoke、metrics、bad answer allowed rate、failed cases、failed gates 和 tool output trust gate；要求覆盖来源元数据缺失、可信级别缺失、过期版本未说明、权限范围隐藏、citation_id 伪造、引用不支持 claim、证据链缺失、hypothesis 写成 fact、confidence 缺 limitations、冲突来源未处理、摘要丢 citation、RAG chunk 弱相关、跨 Agent evidence 丢失、trace 缺 citation map / claim map 和 eval 缺引用准确率 / 无证据 claim / 过期来源 / 冲突来源 / 权限范围样本等 bad case。
35BF. 写一个 0 依赖 RAG + Tool + Agent + Memory 组合审计 demo，输入 toy integration traces 的 user goal、RAG query / docs / citations、tool calls / observations、Agent plan / state / stop reason、Memory reads / write candidates、context priority、context budget、evidence chain、conflicts、security flags、trace fields 和 eval labels；实现 capability boundary clarity、orchestration mode fit、context priority enforcement、context budget allocation、evidence preservation、tool observation use、agent state update discipline、memory read relevance、memory write gate、memory scope permission、conflict resolution policy、injection propagation control、sensitive data memory block、trace linkage coverage 和 layered eval coverage；输出 smoke、metrics、unsafe integration rate、failed cases、failed gates 和 RAG Tool Agent Memory integration gate；要求覆盖边界混淆、编排模式错误、上下文优先级错误、预算缺失、证据链丢失、工具 observation 被忽略、Agent 状态缺失、Memory 读取无关、Memory 写入未确认、Memory 权限缺失、Memory 与 RAG 冲突未处理、RAG 注入触发工具、敏感工具结果写入长期 Memory、trace 链路缺失和分层 eval 缺失等 bad case。
35BG. 写一个 0 依赖工具调用成本、延迟和并发审计 demo，输入 toy tool traces 的 cost components、latency components、budget、timeout、retry policy、idempotency key、concurrency limits、queue priority、rate limit、cache key、batch result、result tokens、degradation、router decision、trace fields 和 eval labels；实现 cost attribution coverage、latency breakdown coverage、tool call budget enforcement、timeout deadline coverage、retry classification accuracy、idempotent retry safety、concurrency limit enforcement、queue priority fairness、rate limit quota handling、cache safety correctness、batching partial success readiness、result trimming budget control、degradation transparency、router performance awareness、performance trace readiness 和 cost latency eval coverage；输出 smoke、metrics、budget or latency overrun rate、failed cases、failed gates 和 tool performance gate；要求覆盖缺成本归因、缺延迟分解、预算超限仍放行、缺 timeout、权限错误仍重试、非幂等重试、并发无上限、队列无优先级、限流无 retry-after、跨用户缓存、批处理无 partial success、大结果直塞上下文、静默降级、router 忽略成本延迟、trace 缺字段和 eval 缺预算 / 超时 / 缓存 / 降级样本等 bad case。
35BH. 写一个 0 依赖 Tool-use eval benchmark 审计 demo，输入 toy eval samples 的 user input、available tools、gold tool calls、pred tool calls、gold / pred args、argument sources、tool results、final answer、safety policy、trace fields、eval slices、cost、latency 和 labels；实现 tool need accuracy、no-tool overcall control、tool selection accuracy、tool set precision、tool set recall、argument schema validity、argument semantic accuracy、argument source coverage、sequence order accuracy、observation grounding accuracy、error recovery readiness、safe failure handling、safety policy compliance、tool simulator determinism、trace replay coverage、cost latency regression control、benchmark slice coverage 和 regression gate readiness；输出 smoke、metrics、safety violation rate、failed cases、failed gates 和 tool use eval benchmark gate；要求覆盖 no-tool 过度调用、应该调用却未调用、工具选错、工具集合多调、工具集合漏调、参数 schema 不合法、相对时间语义错误、参数来源缺失、多工具顺序错误、工具 observation 被忽略、权限错误未恢复、超时后编造结果、prompt injection 未拦截、模拟器不确定、trace replay 字段缺失、成本延迟回归、benchmark 切片缺失和 regression gate 缺失等 bad case。
35BI. 写一个 0 依赖 Function Calling / MCP / A2A 横向对比审计 demo，输入 toy protocol designs 的 scenario、function calling 使用方式、MCP 集成方式、A2A 协作方式、Host runtime 责任、context policy、permission policy、trace fields 和 eval gates；实现 layer boundary clarity、function calling fit、MCP integration fit、A2A delegation fit、object contract coverage、capability discovery fit、lifecycle state alignment、context transfer boundary、permission governance split、Host runtime ownership、trace chain continuity、eval gate linkage 和 overengineering control；输出 smoke、metrics、severe protocol misuse rate、failed cases、failed gates 和 protocol composition gate；要求覆盖把 MCP 当模型 tool_call 格式、用 Function Calling 做企业工具发现、绕过 MCP 直连数据库、用 A2A 包装 read_file、对象契约缺失、能力发现缺失、A2A 旧状态机、跨 Agent 全量上下文转发、权限写进 prompt、Host runtime 缺失、trace 断链、eval gate 未串联和简单 API 过度工程等 bad case。
35BJ. 写一个 0 依赖主流平台工具协议迁移审计 demo，输入 toy provider / framework cases 的 platform、platform type、internal schema、projected schema、tool choice、tool call、tool result、streaming、parallel、built-in tools、error、adapter、framework、RAG、permission、eval、trace 和 lock-in 字段；实现 provider framework type clarity、tool schema projection、tool choice mapping、tool result round trip、streaming event assembly、parallel call alignment、built-in tool boundary、error normalization、provider adapter isolation、framework escape hatch、RAG tool traceability、permission safety consistency、migration eval coverage、provider trace replay readiness 和 vendor lock-in control；输出 smoke、metrics、severe migration regression rate、failed cases、failed gates 和 provider runtime gate；要求覆盖把 framework 当 provider、schema 投影丢 required、force tool 退化成 auto、tool result id 丢失、streaming JSON 增量拼接错误、parallel result 错配、内置工具边界缺失、原始错误未归一化、provider 字段泄漏到业务、framework 无 escape hatch、RAG citation / query trace 丢失、权限只写进 prompt、迁移 eval 缺失、trace replay 缺失和 vendor lock-in 等 bad case。
35BK. 写一个 0 依赖企业 MCP 工具平台审计 demo，输入 toy platform designs 的 MCP Gateway、Registry metadata、capability namespace、tool / resource / prompt contract、Host / Client / Server boundary、OBO authorization、scope binding、tenant isolation、Policy Engine、roots sandbox、Context Manager、prompt injection taint、trace / audit / replay、cost latency quota、developer portal、release lifecycle、provider adapter、eval regression 和 HA 字段；实现 mcp gateway readiness、registry metadata completeness、capability namespace isolation、tool resource prompt contract、host client server boundary clarity、authorization obo scope binding、tenant isolation enforcement、policy engine coverage、sandbox roots containment、context output projection、prompt injection taint propagation、trace audit replay continuity、cost latency quota control、developer portal review readiness、release lifecycle governance、provider adapter compatibility、eval regression coverage 和 high availability readiness；输出 smoke、metrics、tenant leak rate、hard blocker count、failed cases、failed gates 和 enterprise MCP platform gate；要求覆盖 Gateway 缺失、Registry 元数据缺失、namespace 跨租户、tool/resource/prompt 契约缺失、Host/Server 边界混淆、OBO scope 丢失、租户泄露、Policy Engine 被绕过、roots 沙箱缺失、原始工具结果直接进上下文、taint 丢失、trace replay 断链、quota 缺失、开发者审核缺失、发布治理缺失、provider adapter 缺失、eval regression 缺失和 HA 缺失等 bad case。
35BL. 写一个 0 依赖跨 Agent 协作系统审计 demo，输入 toy collaboration designs 的 root goal、requirements、modules、Agent Card、Registry routing、task graph、A2A lifecycle、context package、OBO scopes、permission attenuation、MCP tool boundary、Artifact evidence、conflict policy、claim types、delegation graph、human handoff、trace / audit / replay、eval baseline、budget、idempotency 和 collaboration fit 字段；实现 requirement goal clarity、collaboration module coverage、agent registry routing governance、task graph validity、A2A lifecycle alignment、context minimization enforcement、permission attenuation binding、MCP tool boundary governance、artifact evidence grounding、conflict arbitration readiness、hallucination propagation control、delegation loop containment、human handoff approval readiness、trace audit replay coverage、eval baseline regression coverage、cost latency budget control、scalability idempotency readiness 和 collaboration fit overengineering control；输出 smoke、metrics、unsafe collaboration rate、unnecessary multi-agent rate、hard blocker count、failed cases、failed gates 和 cross-agent collaboration gate；要求覆盖需求缺失、模块缺失、Registry 路由失控、任务图有环、A2A 生命周期不对齐、上下文整包广播、权限放大、MCP 工具权限泄露、Artifact 缺证据、冲突无仲裁、幻觉传播、循环委派、人审缺失、trace replay 缺失、eval baseline 缺失、预算无上限、幂等缺失和简单任务过度多 Agent 化等 bad case。
35BM. 写一个 0 依赖工具协议生态未来演进审计 demo，输入 toy future protocol answers 的 layer boundary、standardization surface、long task lifecycle、capability package、agent autonomy governance、safety metadata、provenance、marketplace governance、provider adapter migration、auto generated tool review、workflow runtime、behavior eval、human / model documentation split、autonomy risk tiering、responsibility trace、natural language API boundary、operating layer governance、engineer readiness 和 unsupported speculation 字段；实现 layered protocol boundary clarity、standardization extension balance、long task lifecycle readiness、capability package completeness、agent-centric autonomy governance、tool safety metadata coverage、provenance verifiability、marketplace governance readiness、provider adapter migration control、auto generated tool review gate、workflow runtime integration、behavior eval coverage、human model documentation split、autonomy risk tiering、responsibility attribution trace、natural language API boundary、agent operating layer governance 和 engineer readiness coverage；输出 smoke、metrics、unsupported speculation rate、hard blocker count、failed cases、failed gates 和 tool protocol future gate；要求覆盖只预测单个 API 字段、认为一个协议解决一切、缺长任务状态、缺能力包、Agent 自主权无边界、安全 metadata 缺失、provenance 缺失、Marketplace 只是列表、vendor lock-in、自动生成工具未审核、workflow 与 Agent 边界混淆、缺行为评估、工具描述混用人和模型读者、缺风险分级、责任 trace 缺失、认为自然语言替代 API、Agent 操作层缺治理和无依据确定性预测等 bad case。
35BN. 写一个 0 依赖 AI Infra 总览审计 demo，输入 toy infra cases 的 compute accelerator、network communication、storage checkpoint、scheduler governance、training reproducibility、inference SLO、data lineage、artifact registry、eval tracking、observability signals、security governance、cost capacity、developer self-service、AI Infra boundary、MLOps / LLMOps / Platform Engineering boundary 和 algorithm infra collaboration 字段；实现 compute accelerator readiness、network communication readiness、storage data checkpoint readiness、scheduler resource governance、training platform reproducibility、inference platform SLO readiness、data platform lineage quality、model artifact registry governance、eval experiment tracking coverage、observability signal coverage、security governance coverage、cost capacity governance、developer self-service readiness、AI Infra boundary clarity、MLOps LLMOps platform boundary clarity 和 algorithm infra collaboration readiness；输出 smoke、metrics、hard blocker count、failed cases、failed gates 和 AI Infra overview gate；要求覆盖把 AI Infra 等同于 Kubernetes、只买 GPU 不管网络存储、训练不可复现、推理无 SLO、数据无 lineage、artifact 无版本、评估实验不可追踪、缺 metrics / logs / traces、权限审计缺失、成本容量无归因、没有开发者自助和算法平台团队割裂等 bad case。
35BO. 写一个 0 依赖 AI Infra / MLOps / LLMOps / Platform Engineering 边界审计 demo，输入 toy boundary cases 的 user scenario、expected owner、actual owner、AI Infra scope、MLOps lifecycle、LLMOps application lifecycle、Platform Engineering developer experience、DevOps / SRE boundary、Data Platform boundary、Model Platform boundary、primary owner、interface contract、artifact lineage handoff、observability SLO handoff、security cost governance、lifecycle stage、incident route 和 collaboration handoff 字段；实现 AI Infra scope accuracy、MLOps lifecycle accuracy、LLMOps application accuracy、Platform Engineering DX accuracy、DevOps SRE boundary accuracy、Data Platform boundary accuracy、Model Platform boundary accuracy、primary owner clarity、interface contract coverage、artifact lineage handoff、observability SLO handoff、security cost governance handoff、lifecycle stage mapping、anti tool name confusion、incident routing accuracy 和 collaboration handoff readiness；输出 smoke、metrics、route error rate、hard blocker count、misrouted cases、failed cases、failed gates 和 boundary gate；要求覆盖 GPU 队列被误归为 MLOps、模型 registry 没有生命周期、LLMOps 只剩 prompt、Platform Engineering 被说成 DevOps 改名、RAG 文档过期误路由、模型权重只是对象存储路径、trace 与 SLO 断链、成本安全无 owner、事故路由错误和团队交接缺失等 bad case。
35BP. 写一个 0 依赖 AI 加速器选型审计 demo，输入 toy accelerator cases 的 workload、model size、batch / concurrency、context length、runtime target、peak compute、memory capacity、memory bandwidth、interconnect bandwidth、low precision、software stack、kernel library、distributed communication、training memory budget、inference KV cache budget、cloud / self-build decision、cost / power / capacity、profiling observability、fallback portability 和 risk governance 字段；实现 peak compute fit、memory capacity fit、memory bandwidth fit、interconnect bandwidth fit、low precision support、software stack maturity、kernel library readiness、distributed communication readiness、training memory budget、inference KV cache budget、workload hardware fit、cloud self build decision、cost power capacity awareness、profiling observability readiness、fallback portability plan 和 selection risk governance；输出 70B BF16 权重显存、70B AdamW 训练状态显存、长上下文 KV cache 显存、roofline bound、smoke、metrics、hard blocker count、failed cases、failed gates 和 accelerator gate；要求覆盖只看峰值 TFLOPS、显存太小、decode 带宽被忽略、多卡互联被忽略、低精度支持缺失、软件栈不成熟、kernel 库缺口、collective 通信缺口、训练状态 OOM、KV cache OOM、工作负载不匹配、云上 / 自建无依据、成本电力忽略、无 profiler 指标、无迁移预案和厂商锁定无人负责等 bad case。
35BQ. 写一个 0 依赖显存和带宽瓶颈审计 demo，输入 toy bandwidth cases 的 VRAM objects、HBM bandwidth、PCIe copy、NVLink topology、NVSwitch all-to-all、inter-node network、KV cache growth、training state memory、communication volume、parallel group topology、dataloader / storage I/O、checkpoint I/O、offload penalty、overlap / fusion、observability metrics 和 bandwidth gate 字段；实现 vram capacity accounting、HBM bandwidth model、PCIe transfer awareness、NVLink topology awareness、NVSwitch all-to-all awareness、inter-node network awareness、KV cache growth accounting、training state memory accounting、communication volume accounting、topology-aware parallel group、dataloader storage I/O awareness、checkpoint I/O awareness、offload penalty awareness、overlap fusion optimization、observability metric coverage 和 bandwidth bottleneck gate；输出 70B BF16 权重显存、70B AdamW 训练状态显存、KV cache 显存、HBM 读 140 GiB 时间、PCIe 拷贝 16 GiB 时间、NVLink 与 PCIe ring all-reduce 时间、8 卡扩展效率、smoke、metrics、hard blocker count、failed cases、failed gates 和 bandwidth gate；要求覆盖显存对象漏算、HBM 带宽被忽略、PCIe 拷贝隐藏、NVLink 拓扑未知、误以为一定有 NVSwitch、机间网络忽略、KV cache 增长漏算、训练状态低估、all-reduce 通信量缺失、parallel group 不拓扑感知、dataloader / storage 未测、checkpoint 阻塞、offload 经 PCIe 无上限、无 overlap / fusion 方案、关键指标缺失和没有带宽门禁等 bad case。
35BR. 写一个 0 依赖训练效率审计 demo，输入 toy training profiles 的 global tokens、step time breakdown、GPU utilization、model FLOPs、hardware executed FLOPs、hardware peak、communication time、dataloader time、H2D copy、checkpoint time、rank step times、single GPU tokens/s、multi GPU tokens/s、padding waste、recompute overhead、loss / grad / numerics status 和 efficiency gate 字段；实现 tokens throughput accounting、step time breakdown coverage、GPU utilization interpretation、MFU estimation、HFU estimation、model FLOPs accounting、hardware peak accounting、communication ratio tracking、I/O dataloader tracking、checkpoint overhead tracking、rank skew detection、scaling efficiency tracking、padding waste awareness、recompute overhead awareness、loss correctness coupling 和 efficiency gate；输出 tokens/s、model step PFLOPs、MFU、HFU、communication ratio、I/O ratio、rank skew、scaling efficiency、smoke、metrics、hard blocker count、failed cases、failed gates 和 efficiency gate；要求覆盖只报 GPU utilization、未统计 tokens/s、没有 step time breakdown、未估算 MFU、HFU 与 MFU 不区分、model FLOPs 算错、hardware peak 口径未知、通信占比缺失、dataloader / I/O 未测、checkpoint spike 忽略、rank skew 隐藏、scaling efficiency 缺失、padding waste 忽略、recompute overhead 忽略、训练很快但 loss 错和没有效率门禁等 bad case。
35BS. 写一个 0 依赖大模型任务画像审计 demo，输入 toy workload profiles 的 workload type、stage、runtime、compute、memory、network、data / storage、latency / SLO、observability、cost、risk、pretraining、SFT、RLHF、evaluation、serving、RAG、Agent、multimodal、scheduler、lineage、governance 和 task profile gate 字段；实现 workload type classification、resource shape completeness、pretraining profile accuracy、SFT iteration profile accuracy、RLHF pipeline profile accuracy、evaluation reproducibility profile、serving SLO profile、RAG freshness retrieval profile、Agent tool runtime profile、multimodal resource profile、scheduler policy fit、observability metric fit、cost model fit、artifact lineage fit、safety governance fit 和 task profile gate；输出 batch job time、serving SLO pass rate、Agent cost、smoke、metrics、hard blocker count、failed cases、failed gates 和 task profile gate；要求覆盖 generic AI job、资源向量缺维度、预训练缺 checkpoint、SFT 缺 dataset version、RLHF 忽略 rollout、评估不可复现、推理缺 TTFT / TPOT、RAG 忽略 freshness、Agent 缺 tool trace、多模态 payload 无上限、调度只有一个 FIFO 队列、只看 GPU utilization、成本只按 token 估算、artifact lineage 缺失、安全治理缺失和没有任务画像门禁等 bad case。
35BT. 写一个 0 依赖 GPU 集群拓扑审计 demo，输入 toy cluster cases 的 nodes、gpus per node、HBM per GPU、scale-up domain、PCIe / NUMA、NVLink / NVSwitch、GPU-NIC affinity、inter-node fabric、rack locality、network oversubscription、collective communication、parallel group placement、storage / checkpoint locality、fault domain、power / cooling、resource pools、scheduler、observability 和 GPU cluster gate 字段；实现 scale-up domain fit、PCIe NUMA locality、NVLink NVSwitch locality、GPU NIC affinity、inter-node fabric readiness、rack locality awareness、oversubscription awareness、collective communication fit、parallel group placement、storage checkpoint locality、fault domain isolation、power cooling capacity fit、resource pool isolation、topology-aware scheduling、observability topology coverage 和 GPU cluster gate；输出 total GPUs、total HBM、scale-up / scale-out all-reduce time、rack fault blast radius、oversubscription ratio、smoke、metrics、hard blocker count、failed cases、failed gates 和 GPU cluster gate；要求覆盖 scale-up 域缺失、PCIe / NUMA 未知、NVLink / NVSwitch 拓扑忽略、GPU-NIC 亲和性缺失、机间 fabric 未就绪、rack locality 缺失、oversubscription 未知、collective 未建模、parallel group 随机放置、checkpoint 远端路径、单 rack 故障域过大、电力散热超配、训练推理混池、拓扑感知调度关闭、缺拓扑观测指标和没有 GPU 集群门禁等 bad case。
35BU. 写一个 0 依赖网络通信审计 demo，输入 toy network cases 的 bandwidth、latency、jitter、RDMA、GPUDirect RDMA、InfiniBand、RoCE、Ethernet、collectives、NCCL、AllReduce、AllGather、ReduceScatter、rank times、packet errors、retransmits、topology congestion、scheduler locality 和 network communication gate 字段；实现 bandwidth unit accounting、latency jitter tracking、RDMA capability fit、GPUDirect RDMA path、InfiniBand fabric readiness、RoCE congestion losslessness、Ethernet fallback scope、collective operation modeling、NCCL topology runtime fit、AllReduce cost estimation、AllGather ReduceScatter cost、rank straggler detection、packet error retransmit tracking、topology congestion awareness、scheduler network locality 和 network communication gate；输出 Gbps 到 GiB/s 换算、16 GiB 传输耗时、64 rank ring all-reduce 耗时、all-gather 耗时、jitter ratio、retransmit rate、smoke、metrics、hard blocker count、failed cases、failed gates 和 network communication gate；要求覆盖带宽单位混淆、延迟抖动忽略、RDMA 缺失、GDR 路径缺失、InfiniBand fabric 未验证、RoCE 缺 PFC / ECN / 拥塞控制、普通以太网承载大规模预训练、collective 未建模、NCCL 拓扑未知、AllReduce 成本缺失、AllGather / ReduceScatter 忽略、rank straggler 隐藏、packet error / retransmit 未追踪、拓扑拥塞忽略、调度不看网络 locality 和没有网络通信门禁等 bad case。
35BV. 写一个 0 依赖存储体系审计 demo，输入 toy storage cases 的 capacity、throughput、IOPS、local NVMe cache、shared filesystem metadata、object store、data lake、training format、small files、dataloader、checkpoint、model load、artifact、commit/checksum、security、lifecycle 和 storage system gate 字段；实现 capacity tier fit、throughput IOPS fit、local NVMe cache fit、shared FS metadata fit、object store authority fit、data lake governance fit、training shard format fit、small file amplification control、dataloader cache hit tracking、checkpoint write recovery fit、model weight load cache fit、artifact lineage metadata fit、consistency commit integrity、security compliance fit、lifecycle cost governance 和 storage system gate；输出数据扫描耗时、checkpoint 写入和恢复耗时、metadata amplification、cache hit rate、月度存储成本、smoke、metrics、hard blocker count、failed cases、failed gates 和 storage system gate；要求覆盖容量不分层、吞吐 / IOPS 未实测、本地缓存缺失、共享文件系统 metadata 过载、对象存储被当 POSIX 高频训练源、数据湖治理缺失、训练格式仍是 raw JSON、小文件未 shard、cache hit 未追踪、checkpoint 未做恢复测试、模型权重冷启动无缓存、artifact 血缘缺失、commit / checksum 缺失、安全审计缺失、生命周期策略缺失和没有存储门禁等 bad case。
35BW. 写一个 0 依赖 checkpoint 生命周期审计 demo，输入 toy checkpoint cases 的 training state object、shard layout、async save、write path、metadata commit、checksum、restore replay、RNG / dataloader state、distributed rank state、retention、cost、replication、security、resume SLO、failure drill 和 checkpoint lifecycle gate 字段；实现 checkpoint object completeness、shard layout fit、async save overlap、write bandwidth fit、metadata commit integrity、checksum manifest validation、restore replay readiness、dataloader RNG state capture、distributed rank state capture、retention policy fit、lifecycle cost governance、cross-region replication fit、security access control、resume SLO tracking、failure drill coverage 和 checkpoint lifecycle gate；输出 shard size、保存耗时、异步可见阻塞、恢复耗时、最大丢失进度、月度保留成本、smoke、metrics、hard blocker count、failed cases、failed gates 和 checkpoint lifecycle gate；要求覆盖只保存模型权重、单大文件、同步阻塞保存、写入带宽未知、metadata 先标 committed、checksum 缺失、从未恢复演练、RNG 状态缺失、rank state 缺失、无限保留、冷归档不用、无跨区副本、权限开放、resume SLO 缺失、故障演练缺失和没有 checkpoint 生命周期门禁等 bad case。
35BX. 写一个 0 依赖容器环境审计 demo，输入 toy container cases 的 image digest、base layer cache、CUDA / driver / framework、GPU runtime、dependency lock、training / serving image split、image size、startup profile、build smoke tests、security scan / signing / SBOM、secret / data exclusion、runtime hardening、registry access、environment metadata、multi-tenant mounts、NCCL / RDMA 和 container environment gate 字段；实现 image digest reproducibility、base layer cache fit、CUDA driver framework compatibility、GPU runtime visibility、dependency lock coverage、training serving image separation、image size startup fit、build pipeline smoke test、security scan signing、secret data exclusion、runtime hardening fit、registry access governance、environment metadata capture、multi-tenant mount isolation、NCCL RDMA runtime fit 和 container environment gate；输出 layer cache hit、镜像拉取耗时、冷启动耗时、依赖锁定覆盖率、driver margin、critical 漏洞数、smoke、metrics、hard blocker count、failed cases、failed gates 和 container environment gate；要求覆盖只记录 latest tag、layer cache 缺失、宿主机 driver 太旧、GPU runtime 缺失、依赖未锁定、训练镜像直接上推理、镜像过大、缺 smoke test、镜像未签名且有 critical 漏洞、密钥打进镜像、root / privileged 运行、registry 公开、metadata 缺失、wildcard hostPath、RDMA 不可见和没有容器环境门禁等 bad case。
35BY. 写一个 0 依赖 Kubernetes GPU 资源管理审计 demo，输入 toy K8s GPU cases 的 device plugin、GPU extended resource、request / limit、gang scheduling、fragmentation、topology、node label / affinity、taint / toleration、MIG / time slicing、namespace / quota、training operator、inference service、GPU monitoring、multi-tenant isolation、Pending troubleshooting 和 Kubernetes GPU gate 字段；实现 device plugin readiness、GPU extended resource fit、GPU request limit integrity、gang scheduling readiness、fragmentation control、topology aware placement、node label affinity fit、taint toleration fit、MIG sharing policy fit、quota namespace governance、training operator readiness、inference service readiness、GPU monitoring mapping、multi-tenant isolation、pending troubleshooting coverage 和 Kubernetes GPU gate；输出 GPU 分配率、碎片化比例、gang 可行性、租户配额占用率、拓扑得分、smoke、metrics、hard blocker count、failed cases、failed gates 和 Kubernetes GPU gate；要求覆盖 device plugin 缺失、未声明 `nvidia.com/gpu`、request / limit 不一致、没有 gang scheduling、碎片化严重、拓扑被忽略、label / affinity 缺失、taint 无 toleration、预训练误用 time slicing、quota 缺失、训练 operator 缺失、推理 SLO 缺失、GPU 监控无法映射到 Pod、多租户隔离缺失、Pending 原因不可解释和没有 Kubernetes GPU 门禁等 bad case。
35BZ. 写一个 0 依赖训练调度治理审计 demo，输入 toy scheduler cases 的 workload resource shape、queue policy、priority governance、quota usage、fair sharing、gang scheduling、backfilling、preemption、checkpoint、fragmentation、topology、cost attribution、observability、failure requeue、anti-starvation 和 training scheduler gate 字段；实现 workload resource shape、queue policy coverage、priority governance、quota usage control、fair share accounting、gang scheduling fit、backfilling safety、preemption checkpoint safety、fragmentation control、topology aware scheduling、checkpoint scheduler coupling、cost attribution control、scheduler observability、failure requeue readiness、anti starvation control 和 training scheduler gate；输出 queue pending P95、quota ratio、dominant share、weighted share ratio、preemption loss、backfill safety、fragmentation ratio、topology score、smoke、metrics、hard blocker count、failed cases、failed gates 和 training scheduler gate；要求覆盖资源画像缺网络或 checkpoint、队列策略缺失、优先级无边界、配额超限、公平性未统计、gang disabled、backfilling 不安全、抢占无 checkpoint、碎片治理缺失、拓扑被忽略、checkpoint 未耦合调度器、成本不可归因、调度观测缺失、失败不能重排、长期饥饿无保护和没有训练调度门禁等 bad case。
35CA. 写一个 0 依赖多租户隔离审计 demo，输入 toy tenant cases 的 resource quota、identity namespace binding、RBAC / ABAC、data access、data classification lineage、network policy、runtime security、image supply chain、secret scope、logs / trace、cost attribution、prod / experiment separation、cross tenant sharing、audit evidence、blast radius 和 multi-tenant isolation gate 字段；实现 resource quota isolation、identity namespace binding、RBAC ABAC permission fit、data access boundary、data classification lineage、network policy isolation、runtime security boundary、image supply chain boundary、secret scope rotation、logs trace redaction、cost attribution coverage、prod experiment separation、cross tenant sharing governance、audit evidence readiness、blast radius control 和 multi-tenant isolation gate；输出 quota ratio、unauthorized block rate、artifact level propagation、secret exposure rate、cost attribution rate、blast radius ratio、smoke、metrics、hard blocker count、failed cases、failed gates 和 multi-tenant gate；要求覆盖 namespace 无 quota、身份和 namespace 不一致、权限通配、跨租户数据访问未阻断、restricted 数据产物被降级、网络默认互通、runtime privileged / hostPath / root、未签名镜像放行、长期共享高权限密钥、日志 / trace 未脱敏、成本不可归因、生产和实验混池、共享资源无版本和 owner、审计日志缺失、单事故影响多个租户和没有多租户隔离门禁等 bad case。
35CB. 写一个 0 依赖集群容量规划审计 demo，输入 toy capacity cases 的 workload forecast、monthly GPU-hours、target effective utilization、GPU count、GPU type mix、training queue SLO、serving peak QPS、input / output token、instance token throughput、network bandwidth、storage throughput、checkpoint capacity、lifecycle policy、utilization headroom、failure redundancy、growth forecast、cost budget、pool separation、quota / burst、observability feedback 和 capacity planning gate 字段；实现 workload forecast coverage、GPU count capacity fit、GPU type mix fit、training queue SLO fit、serving peak SLO fit、network bandwidth capacity fit、storage throughput capacity fit、storage capacity lifecycle fit、utilization headroom fit、failure redundancy fit、growth forecast fit、cost budget fit、pool separation fit、quota burst governance、observability forecast feedback 和 cluster capacity planning gate；输出训练 GPU 需求、推理实例需求、网络余量、存储余量、checkpoint 容量、smoke、metrics、hard blocker count、failed cases、failed gates 和 cluster capacity planning gate；要求覆盖任务画像缺失、GPU 数低于需求、GPU 型号不匹配、训练队列超 SLO、推理峰值被忽略、网络带宽不足、存储吞吐未实测、checkpoint 生命周期容量不足、目标利用率过高、无故障冗余、增长预测缺失、成本超预算、训练推理混池、quota / burst 无治理、observability 缺预测回填和没有容量规划门禁等 bad case。
35CC. 写一个 0 依赖训练平台生命周期审计 demo，输入 toy TrainingJob cases 的 owner、project、image digest、code commit、entrypoint、config version、dataset version、resources、distributed config、checkpoint config、queue、priority、config validation、dependency lock、environment metadata、dataset lineage、data permission、observability、experiment tracking、artifact registry、failure policy、security audit、cost attribution、self-service、lifecycle transitions 和 training platform gate 字段；实现 training job contract、config validation、resource quota binding、image code reproducibility、dataset lineage permission、launcher distributed fit、checkpoint resume policy、observability events metrics、experiment tracking lineage、artifact registry linkage、failure recovery classification、security audit control、cost attribution control、developer self service、lifecycle state machine 和 training platform gate；输出 reproducibility field coverage、lifecycle event coverage、max lost steps、artifact lineage links、cost attribution rate、smoke、metrics、hard blocker count、failed cases、failed gates 和 training platform gate；要求覆盖 TrainingJob 契约缺字段、未 dry run、quota 超额、镜像只用 latest、数据权限缺失、launcher 进程数和 GPU 数不匹配、checkpoint 未验证恢复、events 缺失、experiment tracking 缺 seed、artifact 未注册、失败盲目重试、security audit 缺失、成本未归因、self-service 缺状态页、状态机非法跳转和没有平台门禁等 bad case。
35CD. 写一个 0 依赖训练任务提交系统审计 demo，输入 toy submission cases 的 metadata、schema version、image reference / digest、code repo / commit / diff hash、dataset id / version / permission、command / args、final config snapshot、resources、distributed launcher、checkpoint URI、logs / metrics destination、queue / priority、admission dry run、idempotency、template version、state transitions 和 audit trace 字段；实现 trainingjob schema completeness、required field coverage、image digest binding、code commit diff binding、dataset version permission、structured command safety、final config snapshot、distributed resource consistency、checkpoint URI resume validation、logging metrics destination、queue priority policy binding、quota dry run admission、idempotency duplicate control、template version governance、submit state machine validity、submission audit trace 和 training submission gate；输出 required field coverage、immutable binding coverage、duplicate created count、metrics、hard blocker count、failed cases、failed gates 和 training submission gate；要求覆盖 schema 缺失、owner 缺失、镜像只用 latest、代码只写 branch、数据 latest / 权限缺失、shell 字符串命令、最终配置未快照、分布式 shape 不一致、checkpoint 不可写、日志指标缺失、priority 未审批、quota / dry run 失败、缺 client request id、模板 latest、状态机非法跳转、audit trace 缺失和没有提交门禁等 bad case。
35CE. 写一个 0 依赖分布式训练启动器审计 demo，输入 toy launcher cases 的 launcher、adapter、nodes、gpus per node、nproc per node、allocated world size、world size、ranks、local ranks、CUDA visible devices、master addr / port、torchrun args、DeepSpeed config、Megatron parallel config、Ray workers、NCCL network、rank logs、failure stage、elastic config、checkpoint、scheduler handoff 和 audit trace 字段；实现 resource world size consistency、rank local rank mapping、rendezvous endpoint readiness、GPU binding visibility、launcher adapter fit、torchrun argument fit、deepspeed config fit、megatron parallel consistency、ray runtime fit、network NCCL readiness、log rank aggregation、failure stage classification、elastic training safety、checkpoint launcher coupling、scheduler launcher handoff、launcher audit trace 和 distributed launcher gate；输出 world size、Megatron world size、global batch、smoke、metrics、hard blocker count、failed cases、failed gates 和 distributed launcher gate；要求覆盖 world size 不一致、rank 重复、rendezvous 缺失、GPU 绑定不足、adapter 选错、torchrun 缺 node rank、DeepSpeed config 错误、Megatron 并行维度不一致、Ray scaling config 缺失、NCCL 网卡缺失、只采 rank 0 日志、失败阶段未分类、elastic 缺 checkpoint 安全、checkpoint 未绑定 launcher 状态、scheduler handoff 缺 node rank、audit trace 缺失和没有 launcher gate 等 bad case。
35CF. 写一个 0 依赖训练配置管理审计 demo，输入 toy config cases 的 sources、override order、final config snapshot、model、data、optimizer、batch、parallel、precision、checkpoint、environment、reproducibility seeds、sweep、experiment tracking、release config、permissions、config diff、audit trace 和 training config gate 字段；实现 config source order defined、final config snapshot frozen、required config field coverage、config schema type validation、batch parallel consistency、immutable version binding、environment version capture、reproducibility seed boundary、experiment tracking linkage、hyperparameter sweep trace、checkpoint resume compatibility、release config coupling、config permission approval、config diff readiness、config audit trace 和 training config gate；输出 effective batch、world size、required field coverage、reproducibility field coverage、final config hash、smoke、metrics、hard blocker count、failed cases、failed gates 和 training config gate；要求覆盖来源顺序缺失、最终配置未冻结、必填字段缺失、learning rate 类型或范围错误、global batch 与 data parallel 不一致、dataset latest / 镜像 tag / 代码 branch 漂移、环境版本缺失、随机性边界缺失、experiment 未记录最终配置、sweep 缺父配置、checkpoint resume 不兼容、发布配置未关联、审批缺失、config diff 缺失、audit trace hash 不一致和没有配置门禁等 bad case。
35CG. 写一个 0 依赖训练可观测性审计 demo，输入 toy observability cases 的 log fields、ranks、logged ranks、error ranks、metric groups、metric dimensions、events、state events、attempt id、retry policy、trace spans、dashboard sections、alert rules、anomaly rules、experiment links、retention policy、privacy controls、cost fields 和 observability gate 字段；实现 structured log context、rank log coverage、metric taxonomy coverage、metric dimension scope、event schema lifecycle、state machine validity、attempt retry traceability、trace phase coverage、dashboard diagnostic readiness、alert rule readiness、anomaly detection rules、experiment tracking linkage、retention cost governance、privacy redaction access、cost attribution signal 和 training observability gate；输出 rank log coverage、trace phase coverage、state transitions valid、step time breakdown、checkpoint age、smoke、metrics、hard blocker count、failed cases、failed gates 和 training observability gate；要求覆盖日志上下文缺失、只采 rank 0 日志、指标分类缺失、指标维度缺失、事件非结构化、状态机非法、attempt id 缺失、trace 阶段缺失、dashboard 只有 loss、告警缺失、异常检测缺失、experiment tracking 未链接、留存策略缺失、日志脱敏缺失、成本归因缺失和没有观测门禁等 bad case。
35CH. 写一个 0 依赖 checkpoint 策略审计 demo，输入 toy checkpoint strategy cases 的 RPO、RTO、step interval、step seconds、time interval、restore minutes、main loop block seconds、save triggers、checkpoint scopes、async controls、manifest fields、shard metadata、restore policy、preemption policy、checkpoint age、eval links、release links、retention items、storage budget、monitoring、restore drill、permissions 和 strategy gate 字段；实现 rpo rto budget defined、save trigger policy、checkpoint scope separation、async snapshot consistency、manifest commit integrity、shard metadata resharding、restore selection policy、preemption checkpoint coupling、evaluation best linkage、release checkpoint readiness、retention tier lifecycle、storage cost budget、checkpoint monitoring alerts、restore drill coverage、permission release protection 和 checkpoint strategy gate；输出 max lost minutes、restore minutes、blocking ratio、retention cost、retention kinds、smoke、metrics、hard blocker count、failed cases、failed gates 和 checkpoint strategy gate；要求覆盖 RPO/RTO 缺失、保存触发缺失、resume/eval/release 用途混淆、异步 snapshot 缺失、manifest commit 缺失、分片元数据缺失、恢复策略只读 latest、抢占不看 checkpoint 新鲜度、eval best 未链接、release checkpoint 未就绪、只保留 latest、存储成本超预算、监控告警缺失、恢复演练缺失、发布权限开放和没有策略门禁等 bad case。
35CI. 写一个 0 依赖训练容错审计 demo，输入 toy fault tolerance cases 的 fault taxonomy、fault type、stage、retryable、retryable errors、non-retryable errors、attempt、max attempts、backoff、checkpoint manifest、resume fields、node reschedule、GPU health、rank heartbeat、NCCL logs、OOM peak / limit、data permission、checkpoint committed、numeric instability、diagnosis report、blacklist、user action 和 training fault tolerance gate 字段；实现 fault taxonomy defined、retryability classification、retry budget bounded、backoff policy defined、checkpoint resume readiness、node failure reschedule、GPU health isolation、communication hang detection、OOM handling policy、data permission fast fail、checkpoint commit safety、numeric instability stop rollback、diagnosis report evidence、blacklist precision、user action clarity 和 training fault tolerance gate；输出 lost GPU-hours、retry success rate、waste retry rate、OOM headroom、heartbeat age、resume continuity、smoke、metrics、hard blocker count、failed cases、failed gates 和 training fault tolerance gate；要求覆盖故障分类缺失、retryability 错误、重试预算缺失、backoff 缺失、checkpoint resume 字段缺失、节点失败未重调度、GPU 健康未隔离、NCCL hang 未分类、OOM 原样重试、数据权限错误被重试、恢复 uncommitted checkpoint、NaN 盲目重试、诊断报告缺失、黑名单误伤、用户动作不清晰和没有容错门禁等 bad case。
35CJ. 写一个 0 依赖训练数据供给审计 demo，输入 toy data supply cases 的 manifest、dataset version、preprocessing version、shards、format、checksum coverage、small file count、streaming、prefetch、retry、local cache、distributed cache、shuffle seed、shuffle buffer、rank shard assignment、data locality、DataLoader workers、prefetch factor、CPU preprocess、H2D copy、bad samples、observability metrics、checkpoint dataloader state、permission version 和 training data supply gate 字段；实现 dataset manifest integrity、shard format size fit、small file amplification control、streaming prefetch retry、local cache policy、distributed cache policy、shuffle reproducibility、rank shard assignment、data locality fit、dataloader parallelism fit、H2D copy readiness、bad sample threshold、data supply observability、checkpoint dataloader state、permission version governance 和 training data supply gate；输出 data tokens/s、observed data MiB/s、data wait ratio、cache hit rate、bad sample rate、rank balance、smoke、metrics、hard blocker count、failed cases、failed gates 和 training data supply gate；要求覆盖 manifest 缺失、shard 格式不适合、小文件放大、流式读取无预取和重试、本地缓存无版本隔离、分布式缓存不一致、shuffle 不可复现、rank 读重或漏读、跨区域读取、DataLoader worker 不足、H2D copy 阻塞、坏样本超过阈值、观测指标缺失、checkpoint 不保存 dataloader state、权限版本未绑定和没有数据供给门禁等 bad case。
35CK. 写一个 0 依赖训练平台安全审计 demo，输入 toy security cases 的 identity、runtime service account、RBAC roles、ABAC attributes、dataset permission、purpose、artifact classification、resource quota、priority approval、image registry、digest、signature、vulnerability scan、provenance、runtime privileged / hostPath / root、secret manager、short-lived credential、network egress、log redaction、model artifact permission、high risk approval、audit events、audit retention、incident response 和 training platform security gate 字段；实现 identity runtime binding、RBAC ABAC policy fit、dataset permission purpose、artifact classification propagation、resource quota priority governance、image supply chain trust、runtime code isolation、secret scope rotation、network egress control、log redaction access、model artifact permission、high risk approval、audit log completeness、audit integrity retention、incident response readiness 和 training platform security gate；输出 unauthorized block rate、overprivilege rate、artifact level propagation、secret exposure rate、audit coverage、approval coverage、smoke、metrics、hard blocker count、failed cases、failed gates 和 training platform security gate；要求覆盖共享管理员身份、RBAC 通配 / 过度授权、数据用途未授权、restricted 产物降级、高优先级资源未审批、镜像无 digest / 签名、runtime privileged、密钥长期且泄露、默认允许外联、日志脱敏缺失、模型权重下载越权、高风险审批缺失、审计日志缺失、审计可篡改或留存不足、事件响应缺失和没有安全门禁等 bad case。
35CL. 写一个 0 依赖训练平台系统设计审计 demo，输入 toy system design cases 的 requirement clarification、TrainingJob contract、lifecycle state machine、admission quota policy、scheduler resource fit、distributed launcher fit、data config lineage、checkpoint recovery coupling、observability experiment linkage、fault recovery flow、security audit governance、cost capacity governance、developer self service、interface artifact contract、tradeoff boundary reasoning 和 training platform design gate 字段；实现 requirement clarification、trainingjob contract、lifecycle state machine、admission quota policy、scheduler resource fit、distributed launcher fit、data config lineage、checkpoint recovery coupling、observability experiment linkage、fault recovery flow、security audit governance、cost capacity governance、developer self service、interface artifact contract、tradeoff boundary reasoning 和 training platform design gate；输出 world size、effective batch、queue P95 minutes、max lost GPU-hours、trace coverage、cost attribution rate、smoke、metrics、hard blocker count、failed cases、failed gates 和 training platform design gate；要求覆盖需求澄清缺恢复 SLO、TrainingJob 缺 final config hash、状态机缺 retrying、准入缺 quota dry run、调度缺 gang / topology、launcher world size 不一致、数据和配置血缘缺 dataset version、checkpoint 未做 restore drill、observability 未链接 traces、容错盲目重试、安全审计缺失、成本无人归因、开发者自助缺状态页、artifact registry 契约缺失、trade-off 边界缺失和没有最终设计门禁等 bad case。
35CM. 写一个 0 依赖推理平台总览审计 demo，输入 toy inference platform cases 的 request lifecycle、SLO metric contract、model registry、model router、runtime prefill / decode、continuous batching、KV cache capacity、autoscaling warm pool、release rollback、rate limit degrade、safety governance、cost attribution、observability trace、multi-model governance、developer API self-service 和 inference platform gate 字段；实现 request lifecycle coverage、SLO metric contract、model registry readiness、model router policy fit、runtime prefill decode fit、continuous batching readiness、KV cache capacity governance、autoscaling warm pool fit、release rollback control、rate limit degrade protection、safety governance fit、cost attribution optimization、observability trace coverage、multi model governance、developer API self service 和 inference platform gate；输出 TTFT P95、TPOT P95、E2E P99、tokens/s、output tokens/s、KV cache MiB、KV pressure、cost per 1k tokens、smoke、metrics、hard blocker count、failed cases、failed gates 和 inference platform gate；要求覆盖请求生命周期缺 audit、SLO 文档缺失、Model Registry 缺 rollback version、Router 缺 cost budget、runtime 不区分 prefill / decode、continuous batching 不能动态退出、KV cache 缺 admission control、扩缩容缺 warm pool、发布缺 rollback、保护机制缺 circuit breaker、安全治理缺日志脱敏、成本归因缺 model 维度、observability 缺 decode span、多模型治理缺 quota policy、开发者自助缺 usage dashboard 和没有最终推理平台门禁等 bad case。
35CN. 写一个 0 依赖模型服务运行时选型审计 demo，输入 toy runtime profiles 的 model / hardware、runtime boundary、model format / tokenizer、prefill / decode scheduler、continuous batching、KV cache manager、streaming API、quantization、distributed inference、observability metrics、benchmark method、deployment rollback、ecosystem maintenance、cost capacity、self-development threshold 和 runtime selection gate 字段；实现 model hardware fit、runtime boundary clarity、model format tokenizer fit、prefill decode scheduler fit、continuous batching fit、KV cache memory manager fit、streaming API fit、quantization accuracy fit、distributed inference fit、observability metrics fit、benchmarking method fit、deployment rollback fit、ecosystem maintenance fit、cost capacity fit、self development threshold 和 runtime selection gate；输出 TTFT P95、TPOT P95、E2E P99、KV cache MiB、KV concurrency limit、feature score、cost per 1k tokens、smoke、metrics、hard blocker count、failed cases、failed gates 和 runtime selection gate；要求覆盖模型硬件缺失、平台和 runtime 边界混乱、chat template 缺失、prefill / decode phase metrics 缺失、continuous batching 不能动态退出、KV cache 缺 admission control、streaming 无 backpressure、量化缺质量回归、分布式拓扑未知、observability 缺 decode 指标、benchmark 缺输出长度分布、发布无 rollback、维护状态未知、成本容量缺 KV 预算、自研 runtime 只有模糊规模收益和没有最终选型门禁等 bad case。
35CO. 写一个 0 依赖 Prefill / Decode / KV 资源画像审计 demo，输入 toy resource profiles 的 request token profile、prefill phase、decode phase、SLO、KV cache、paged block、prefix cache、long context、tenant、continuous batching、PD disaggregation、streaming、metrics、cost 和 resource profile gate 字段；实现 request token profile、prefill phase accounting、decode phase accounting、TTFT TPOT contract、KV cache formula fit、KV capacity admission、paged block management、prefix cache reuse、long context policy、tenant KV isolation、continuous batching policy、PD disaggregation fit、streaming backpressure fit、observability phase metrics、cost capacity model 和 resource profile gate；输出 TTFT P95、TPOT P95、E2E P99、prefill tokens/s、decode output tokens/s、KV cache MiB、KV pressure、per-token KV KiB、KV concurrency limit、prefix saved prefill ms、cost per 1k tokens、smoke、metrics、hard blocker count、failed cases、failed gates 和 resource profile gate；要求覆盖 token 分布缺失、prefill 未计量、decode 未计量、SLO 缺 TPOT、KV 公式缺 KV heads、KV admission 缺失、paged block table 缺失、prefix cache 缺租户隔离、long context 无上限、tenant quota 缺失、continuous batching 不能动态退出、PD 分离未估 KV transfer、streaming 无 backpressure、observability 缺 prefill 指标、成本模型缺 KV MiB 和没有资源画像门禁等 bad case。
35CP. 写一个 0 依赖 Continuous Batching / PagedAttention 调度审计 demo，输入 toy scheduler profiles 的 request arrival profile、batching、token budget、phase policy、paged KV、admission、long-short isolation、tenant fairness、cancel cleanup、streaming、metrics 和 scheduler gate 字段；实现 request arrival profile、continuous batching core、token budget policy、prefill decode balance、paged KV block management、KV capacity admission、long short isolation、tenant priority fairness、cancellation cleanup、streaming backpressure、observability metrics 和 scheduler gate；再模拟 5 个 toy 请求的动态到达、chunked prefill、decode step、running set 退出和 KV block 释放；输出 finished order、avg queue wait、P95 TTFT steps、P95 active sequences、max KV blocks used、max KV waste ratio、long prompt chunks、trace tail、smoke、metrics、hard blocker count、failed cases、failed gates 和 scheduler gate；要求覆盖 arrival profile 缺失、静态 batch 不能动态退出、token budget 缺失、prefill/decode 不平衡、paged KV 无 free list、KV admission 缺水位、长短请求混跑、租户公平缺失、取消后泄漏 blocks、streaming 无 backpressure、调度指标缺失和没有最终门禁等 bad case。
35CQ. 写一个 0 依赖模型路由审计 demo，输入 toy requests 和 model registry 的 request intent profile、model capability profile、tenant permission、task capability、input / output tokens、cost budget、latency SLO、quality score、safety score、fallback chain、canary weight、health / load、route trace、degrade policy、config version 和 model routing gate 字段；实现 request intent profile、model capability profile、permission candidate filter、cost budget policy、latency SLO policy、quality safety policy、fallback chain governance、canary routing stability、realtime health load、route trace coverage、degrade policy control、route config versioning 和 model routing gate；输出 routing decisions、fallback chains、degraded requests、estimated costs、estimated latency、route trace coverage、metrics、hard blocker count、failed cases、failed gates 和 model routing gate；要求覆盖请求画像缺失、模型能力画像缺失、权限过滤缺失、成本预算缺失、延迟 SLO 缺失、质量安全缺失、fallback chain 无界、灰度分桶不稳定、健康负载被忽略、route trace 缺失、降级策略不受控、配置版本缺失和没有最终路由门禁等 bad case。
35CR. 写一个 0 依赖推理缓存体系审计 demo，输入 toy requests、result cache、prefix cache 和 semantic cache 的 request cache profile、cache layer boundary、key version fingerprint、prompt prefix reuse、KV lifecycle management、semantic similarity guard、result cache determinism、tenant permission isolation、TTL staleness control、eviction quota policy、streaming cache policy、cache observability metrics、cost latency savings 和 cache governance gate 字段；实现 result cache hit、prefix cache hit、semantic cache hit、跨租户语义命中阻断、过期数据版本阻断、非确定性采样结果缓存禁用、saved tokens、saved prefill ms、net saved cost、trace coverage 和 cache governance gate；要求覆盖请求画像缺失、缓存层边界混淆、key 缺版本、prefix 复用缺失、KV 生命周期泄漏、语义阈值缺失、随机生成被结果缓存、租户隔离缺失、TTL / 过期控制缺失、驱逐配额缺失、streaming 策略缺失、缓存指标缺失、收益模型缺失和没有最终缓存门禁等 bad case。
35CS. 写一个 0 依赖推理自动扩缩容审计 demo，输入 toy autoscaling windows 的 traffic token profile、SLO latency contract、queue backlog signal、GPU / KV capacity signal、cold start lead time、warm pool readiness、multi metric recommendation、scale up/down policy、draining streaming safety、router admission coupling、tenant quota priority、cost budget guard、autoscaling trace coverage 和 autoscaling gate 字段；按 QPS、input tokens/s、output tokens/s、queue wait、TTFT、TPOT、KV pressure 和 GPU utilization 分别计算建议副本数，取最大值后再受 min/max replicas、budget max replicas、warm pool 和 cold start 约束；输出 autoscaling recommendations、每类 metric replicas、draining decision、warm pool ready、cold start seconds、metrics、hard blocker count、failed cases、failed gates 和 autoscaling gate；要求覆盖流量 token 画像缺失、SLO 缺失、队列信号缺失、GPU / KV 信号缺失、冷启动忽略、warm pool 缺失、单指标 HPA、扩缩容策略缺失、draining 缺失、router / admission 未联动、租户配额缺失、成本保护缺失、autoscaling trace 缺失和没有最终扩缩容门禁等 bad case。
35CT. 写一个 0 依赖推理保护策略审计 demo，输入 toy requests、tenant limits 和 endpoint health 的 request token rate limit、concurrency quota control、admission capacity guard、circuit breaker state machine、retry error classification、retry budget bound、idempotency stage guard、timeout budget allocation、degradation policy control、priority tenant fairness、streaming cancellation cleanup、observability trace coverage、cost budget guard 和 protection gate 字段；按租户请求数、input / output tokens、上下文长度、并发、成本、队列、KV pressure、错误类型、执行阶段、partial output、retry attempt 和剩余 timeout 依次输出 admit、rate limited、circuit open、retry scheduled、no retry、fallback 或 degrade；要求覆盖请求 / token 限流缺失、并发配额缺失、admission 缺容量信号、熔断状态机缺失、错误分类缺失、retry budget 缺失、幂等和执行阶段缺失、timeout budget 缺失、降级策略缺失、租户公平缺失、streaming 取消未清理、保护 trace 缺失、成本预算缺失和没有最终保护门禁等 bad case。
35CU. 写一个 0 依赖推理发布治理审计 demo，输入 toy release bundle、canary steps、A/B counts、guardrail thresholds 和 rollback target 的 version bundle trace、model artifact integrity、tokenizer prompt runtime binding、offline eval quality gate、safety eval gate、contract test compatibility、stable bucket assignment、canary ramp control、AB experiment design、sample ratio health、guardrail metric control、rollback readiness、release audit trace、post release monitoring 和 release governance gate 字段；实现 release bundle completeness、stable hash bucket、canary promote / rollback 决策、二比例 A/B 近似检验、sample ratio mismatch 检查、rollback readiness 和 release gate；要求覆盖版本包缺字段、artifact checksum 缺失、tokenizer / prompt / runtime 未绑定、离线 eval 缺失、安全 eval 缺失、契约测试缺失、随机分桶、canary 放量失控、A/B 设计缺失、sample ratio 未检查、guardrail 缺失、rollback target 缺失、release trace 缺失、上线后监控缺失和没有最终发布门禁等 bad case。
35CV. 写一个 0 依赖推理平台系统设计审计 demo，输入 toy design answers 的 requirement clarification、request lifecycle、SLO token contract、model registry release、router policy、runtime prefill / decode、KV scheduler capacity、cache isolation、autoscaling capacity plan、protection policy、release governance、tenant security、observability trace、cost capacity、trade-off 和 inference platform design gate 字段；实现需求覆盖率、端到端延迟拆分、decode 时间近似、token 负载、KV 单请求显存、KV 容量门禁、多指标副本需求和最终 design gate；输出 latency summary、capacity plan、KV gate、SLO pass、metrics、hard blocker count、failed case sample、failed gate sample 和 inference platform design gate；要求覆盖需求澄清缺 token 分布、请求链路缺 trace、SLO 只写 QPS、Model Registry 缺 rollback、Router 不看权限 / 成本 / SLO、runtime 不区分 prefill / decode、KV 容量不估算、缓存不做租户隔离、扩缩容只看 GPU、保护机制只会重试、发布无 guardrail、多租户安全缺审计、观测缺 prefill / decode span、成本不可归因、trade-off 边界缺失和没有最终系统设计门禁等 bad case。
35CW. 写一个 0 依赖数据平台供给链路审计 demo，输入 toy data platform cases 的 sources、raw lake、cleaning、dedup、PII、quality、builder、manifest、shards、reader、permissions、lineage、observability 和 data platform supply gate 字段；实现 source registry coverage、raw data lake integrity、cleaning policy versioning、dedup contamination guard、PII redaction coverage、quality scoring calibration、dataset builder recipe、dataset manifest completeness、dataset version immutability、streaming shard readiness、permission purpose binding、data lineage coverage、data platform supply observability、read throughput fit、data wait fit 和 data platform supply gate；输出 source coverage、manifest completeness、keep rate、duplicate rate、PII leak rate、lineage coverage、read tokens/s、data wait ratio、permission allowed / denied count、metrics、hard blocker count、failed cases、failed gates 和 data platform gate；要求覆盖来源字段缺失、raw lake 可覆盖、清洗策略未版本化、train / eval overlap、PII 未阻断、质量分无校准、builder 配方缺失、manifest 不完整、dataset version 可变、streaming shard 未就绪、权限用途不绑定、血缘覆盖不足、观测指标缺失、读取吞吐不足、data wait 过高和没有最终数据平台供给门禁等 bad case。
35CX. 写一个 0 依赖数据版本、血缘和质量监控审计 demo，输入 toy data governance cases 的 dataset version、manifest、diff report、lineage nodes / edges / events、impact analysis、quality metrics、alerts、TrainingJob links、ModelVersion links 和 data version quality gate 字段；实现 dataset version contract、manifest checksum integrity、immutable snapshot policy、delta version diff、lineage graph completeness、impact analysis readiness、schema quality gate、content quality gate、data safety compliance gate、distribution drift monitoring、train/eval leakage monitoring、annotation quality monitoring、synthetic data monitoring、quality alert routing、training model version linkage 和 data version quality gate；输出 manifest completeness、checksum coverage、added / removed sample ratio、lineage coverage、distribution drift TV、schema valid rate、duplicate rate、PII open rate、train/eval overlap rate、annotation agreement、synthetic ratio、impacted models、metrics、hard blocker count、failed cases、failed gates 和 data version quality gate；要求覆盖 version contract 缺失、manifest checksum 缺失、snapshot 被覆盖、diff report 缺失、lineage 断边、影响分析不可用、schema 失败、重复率升高、PII 未阻断、分布漂移过大、eval 泄漏、标注一致性低、合成比例过高、告警无 owner、ModelVersion 缺数据版本血缘和没有最终数据版本质量门禁等 bad case。
35CY. 写一个 0 依赖模型仓库审计 demo，输入 toy model registry cases 的 model version、weights、tokenizer / config、adapter、merge、quantization、runtime、eval reports、security permission、release、lineage、lifecycle 和 model registry gate 字段；实现 model version contract、weight manifest integrity、safe weight format、tokenizer config binding、adapter base compatibility、merge lineage completeness、quantization eval gate、runtime compatibility fit、eval report linkage、safety permission gate、release status governance、rollback alias readiness、artifact lineage completeness、load cache readiness、lifecycle retention policy 和 model registry gate；输出 artifact metadata completeness、weight checksum coverage、weight total size、tokenizer config binding、adapter rank ratio、quant quality drop、quant latency speedup、estimated load time、lineage node count、release / rollback alias、metrics、hard blocker count、failed cases、failed gates 和 model registry gate；要求覆盖 ModelVersion 契约缺失、权重 checksum 缺失、不安全权重格式、tokenizer hash 错配、adapter base model 错配、merge lineage 缺失、量化质量退化超阈值、runtime 不兼容、eval report 缺失、权限安全不合格、发布审批不足、回滚版本缺失、artifact lineage 断边、加载缓存未准备、生命周期策略缺失和没有最终模型仓库门禁等 bad case。
35CZ. 写一个 0 依赖 Artifact 管理审计 demo，输入 toy artifact cases 的 artifact metadata、dataset manifest、checkpoint、eval report、deployment package、release manifest、integrity、lineage、store split、access policy、lifecycle、promotion、experiment tracking、rollback、delete plan 和 artifact management gate 字段；实现 artifact metadata contract、dataset artifact manifest、checkpoint artifact recoverability、eval report reproducibility、deployment package completeness、release manifest integrity、checksum integrity gate、artifact lineage completeness、artifact store metadata split、permission access control、lifecycle retention policy、promotion gate readiness、experiment tracking linkage、rollback artifact readiness、deletion dependency safety 和 artifact management gate；输出 metadata completeness、dataset manifest completeness、checksum coverage、checkpoint recoverable、eval metric count、deployment package completeness、release ref coverage、lineage node count、monthly storage cost、candidate delete safe、artifact size、metrics、hard blocker count、failed cases、failed gates 和 artifact management gate；要求覆盖 artifact metadata 缺失、dataset manifest 样本数不一致、checkpoint 恢复状态缺失、eval report prompt 版本缺失、deployment package 缺 health check、release manifest 可变、checksum 未验证、artifact lineage 断边、metadata DB 混入大 blob、权限公开、retention policy 缺依赖检查、promotion 缺 safety、experiment outputs 缺失、rollback artifact 缺失、删除仍有下游依赖和没有最终 artifact 管理门禁等 bad case。
35DA. 写一个 0 依赖实验追踪审计 demo，输入 toy experiment runs 的 run metadata、params、metrics、logs、sample results、code、data、model、environment、prompt、evaluation、artifacts、cost、state transitions、search index、lineage 和 decision 字段；实现 run metadata contract、parameter config capture、metric curve step capture、log run linkage、sample result diagnosis、code version reproducibility、data version reproducibility、model checkpoint linkage、environment capture、prompt eval config capture、artifact output linkage、cost attribution、run state transition、search index readiness、lineage graph readiness、decision note capture 和 experiment tracking gate；输出 metadata completeness、parameter coverage、metric step coverage、sample result coverage、environment coverage、sample error types、success rate、total cost、cost per success、lineage node count、reproducibility ready、metrics、hard blocker count、failed cases、failed gates 和 experiment tracking gate；要求覆盖 metadata 缺 owner、参数缺 seed、metric 缺 step、日志 run id 不匹配、样本缺 error type、代码 dirty、数据 manifest 缺失、checkpoint linkage 缺失、环境缺 NCCL、prompt / eval 配置缺 metric definition、artifact 缺 checksum、成本不可归因、状态跳转非法、搜索索引缺 metric、lineage 断边和实验结论缺失等 bad case。
35DB. 写一个 0 依赖评估平台审计 demo，输入 toy eval platform cases 的 dataset、offline config、metrics、judge、human review、pairwise、online、release gate、report、slice metrics、regression suite、scheduler、cache、observability、artifacts 和 platform gate 字段；实现 eval dataset contract、offline batch reproducibility、metric definition versioning、LLM judge calibration、human review quality control、pairwise blinding randomization、online eval guardrail、release gate readiness、eval report completeness、slice regression detection、regression suite coverage、eval job scheduler reliability、eval cache key integrity、eval observability、artifact tracking linkage 和 evaluation platform gate；输出 dataset coverage、contamination rate、offline accuracy、pairwise win rate、human gold accuracy、human pair agreement、judge-human agreement、online guardrail pass、release ready、cache hit rate、report field count、lineage edge count、estimated eval cost、metrics、hard blocker count、failed cases、failed gates 和 evaluation platform gate；要求覆盖 dataset 缺版本或切片、离线配置缺评估代码版本、metric 无版本、judge 缺人类校准、人评 gold set 低、pairwise 未盲评、在线安全事件、release gate 缺 rollback、报告缺样本结果、切片退化、回归集覆盖不足、调度缺重试、缓存 key 缺 prompt、观测缺 judge error、artifact lineage 缺失和没有最终平台门禁等 bad case。

评估平台追问（承接 35DB）：为一次样本运行设计 run_id、sample_id、attempt_id、score_id 与 trace/artifact 引用；分别定义 execution、scoring、maturity 状态。模拟请求已发送但客户端超时，说明何时重试、如何保留原 attempt、如何把失败计入系统可用性/成本，以及哪些情况下工具副作用使盲目重试不安全。

35DC. 写一个 0 依赖特征向量索引基础设施审计 demo，输入 toy feature embedding index cases 的 feature definition、offline online sample、point-in-time join、embedding records、chunk lineage、vector index、ANN quality、metadata permission filter、shadow index、retrieval traces、lineage graph、tenant isolation、quality monitoring、cost capacity、RAG / Agent integration 和 infrastructure gate 字段；实现 feature definition contract、offline online consistency、point in time correctness、embedding version contract、chunk embedding lineage、vector index build readiness、ANN quality latency gate、metadata permission filter、shadow index switch readiness、retrieval trace completeness、feature embedding lineage graph、multi tenant isolation、quality monitoring metrics、cost capacity governance、RAG agent integration 和 feature embedding index gate；输出 feature contract coverage、offline online skew rate、point in time leak rate、embedding dimension、embedding norm ok、chunk embedding coverage、bruteforce top3、permission filtered top3、cross tenant blocked、ANN recall、retrieval P95、vector memory、shadow index ready、lineage edge count、estimated build cost、metrics、hard blocker count、failed cases、failed gates 和 feature embedding index gate；要求覆盖特征定义缺字段、offline / online transform 不一致、未来特征泄漏、embedding version 缺失、chunk lineage 缺 offset / checksum、索引构建不就绪、ANN recall 低、metadata filter 缺 tenant / ACL、shadow index 缺 rollback、retrieval trace 缺 index version、lineage graph 断边、跨租户命中未阻断、质量监控缺 permission leak、成本容量无预算、RAG / Agent 引用缺 source pointer 和没有最终基础设施门禁等 bad case。
35DD. 写一个 0 依赖 RAG / Agent 平台存储审计 demo，输入 toy RAG agent storage cases 的 knowledge base、documents、chunks、sync job、user、retrieval trace、prompt trace、citations、agent definition、tools、tool policy、tool traces、execution trace、replay、memory、trace privacy、costs、audit logs 和 platform gate 字段；实现 knowledge base contract、document chunk version contract、sync delete propagation、ACL permission enforcement、retrieval trace completeness、prompt assembly trace、citation version binding、agent definition versioning、tool definition contract、tool permission gate、tool call trace completeness、execution trace replay readiness、memory privacy lifecycle、trace privacy retention、cost attribution governance 和 RAG agent storage gate；输出 kb contract coverage、document version coverage、chunk citation coverage、permission filtered chunks、retrieval trace complete、prompt hash ready、citation version bound、tool schema coverage、dangerous tool blocked、tool trace complete、trace replay ready、memory private blocked、trace redaction coverage、estimated run cost、audit event count、metrics、hard blocker count、failed cases、failed gates 和 RAG agent storage gate；要求覆盖知识库缺 owner、chunk 缺 checksum、删除未传播、ACL filter 缺失、retrieval trace 缺 embedding version、prompt hash 缺失、citation 缺 document version、agent definition 未版本化、tool schema 缺失、高风险工具绕过权限、tool call trace 缺权限决策、execution replay 缺 tool output、敏感 memory 未阻断、trace 无脱敏 / TTL、成本缺 tenant 和没有最终 storage gate 等 bad case。
35DE. 写一个 0 依赖 AI Infra 可观测性审计 demo，输入 toy telemetry cases 的 signals、metrics、SLO、latency、traces、logs、events、correlation、labels、training、inference、data quality、RAG / Agent、cost、privacy、alerts 和 platform gate 字段；实现 signal inventory coverage、metric contract completeness、SLO error budget readiness、latency quantile guard、trace span coverage、log event structure、correlation ID coverage、cardinality budget control、training observability、inference observability、data quality observability、RAG agent observability、cost observability、privacy redaction retention、alert actionability 和 observability platform gate；输出 signal coverage、metric contract coverage、availability、error budget remaining、latency p99、trace span coverage、log structure coverage、event timeline coverage、correlation ID coverage、cardinality estimate、training tokens/s、inference TTFT p95、RAG agent trace ready、cost per 1k tokens、redaction coverage、actionable alerts、metrics、hard blocker count、failed cases、failed gates 和 observability gate；要求覆盖信号清单缺 events、metric 缺 owner、SLO 错误预算烧穿、p99 超阈值、trace 缺 decode span、日志缺 trace id、关联 ID 覆盖不足、request id 进入 metrics label、训练 data wait 过高、推理 KV cache 压力过高、数据质量门禁失败、RAG / Agent tool trace 缺失、成本不可按 tenant 归因、日志 / trace 脱敏和 TTL 不合格、告警无行动性和没有最终可观测性门禁等 bad case。
35DF. 写一个 0 依赖训练故障定位审计 demo，输入 toy training incident cases 的 evidence、lifecycle、ranks、loss、NaN / Inf、update、hang、memory、communication、data I/O、checkpoint、resume、data fault、diff、repro 和 platform gate 字段；实现 training fault evidence coverage、lifecycle event timeline、rank log completeness、loss anomaly diagnosis、NaN Inf guard、update health check、hang straggler detection、OOM phase memory evidence、communication bottleneck diagnosis、IO data bottleneck diagnosis、checkpoint integrity、resume continuity、data fault isolation、config code diff readiness、minimal reproduction readiness 和 training fault diagnosis gate；输出 evidence coverage、timeline events、rank log coverage、loss spike、NaN / Inf rate、nonzero grad ratio、hang detected、OOM phase、memory peak、communication ratio、straggler ratio、data wait ratio、cache hit rate、checkpoint integrity、resume ready、data fault rate、config diff keys、minimal repro ready、metrics、hard blocker count、failed cases、failed gates 和 training fault diagnosis gate；要求覆盖故障证据缺失、生命周期事件乱序、rank 日志缺失、loss anomaly 缺异常 batch、NaN / Inf finite check 缺失、梯度更新不健康、hang 无可疑 rank、OOM 阶段未知、通信慢缺 NCCL 日志、I/O 慢缺 worker queue 证据、checkpoint checksum 缺失、resume step 不连续、数据异常缺 diff report、配置 diff 缺失、最小复现缺单机试验和没有最终训练故障定位门禁等 bad case。
35DG. 写一个 0 依赖推理故障定位审计 demo，输入 toy inference incident cases 的 evidence、scope、trace、decode、latency、traffic、errors、KV cache、model runtime、streaming、cache、route、tool、quality、release diff 和 platform gate 字段；实现 inference fault evidence coverage、request scope slice、trace stage coverage、TTFT decomposition、TPOT decode health、tail latency attribution、throughput token capacity、error taxonomy timeout stage、KV cache pressure guard、model artifact load readiness、streaming reliability、cache hit key governance、route trace correctness、tool dependency trace、quality release diff readiness 和 inference fault diagnosis gate；输出 evidence coverage、scope coverage、stage coverage、TTFT、TPOT、P99、tail amplification、tokens/s、per GPU output tokens/s、error rate、timeout stage coverage、KV pressure、model load ready、stream abort rate、cache hit rate、route correct、tool failure rate、quality delta、metrics、hard blocker count、failed cases、failed gates 和 inference fault diagnosis gate；要求覆盖推理证据缺失、影响范围切片缺 runtime、trace 缺 downstream span、TTFT 超 SLO 且无根因、TPOT 超 SLO 且无 decode 根因、p99 尾延迟无慢请求样本、token 吞吐不足、错误 taxonomy 缺失、KV pressure 超阈值、模型权重 checksum 缺失、streaming cancellation cleanup 缺失、cache key 未绑定版本、路由选错模型、tool trace 缺失、质量下降超过阈值和没有最终推理故障定位门禁等 bad case。
35DH. 写一个 0 依赖 SLO 值班体系审计 demo，输入 toy SLO / on-call cases 的 SLI contract、SLO target、SLA boundary、inference、training、data eval、error budget、burn alerts、alerts、oncall、severity、runbook、change events、mitigation、postmortem 和 cost tradeoff 字段；实现 sli contract completeness、slo target measurability、sla boundary clarity、inference SLO coverage、training SLO coverage、data eval SLO coverage、error budget accounting、burn rate alerting、alert actionability、oncall ownership escalation、incident severity routing、runbook executability、change event linkage、mitigation rollback authority、postmortem action closure 和 SLO cost tradeoff gate；输出 availability、latency pass rate、bad allowed、budget used、budget remaining、fast burn、slow burn、actionable alerts、ACK SLO、SEV1 response、runbook count、change fields、mitigation ready、action item closure、value cost ratio、metrics、hard blocker count、failed cases、failed gates 和 SLO oncall gate；要求覆盖 SLI 契约缺坏事件口径、SLO 不可测、SLA 边界不清、推理 SLO 缺 TPOT、训练不区分平台错误、数据 / 评估 SLO 缺失、错误预算耗尽、burn rate 告警缺多窗口、告警无行动性、on-call 缺升级、事故分级错误、runbook 不可执行、变更时间线缺 runtime、值班无回滚权限、复盘行动项关闭率低和 SLO 成本收益不过线等 bad case。
35DI. 写一个 0 依赖成本治理审计 demo，输入 toy cost cases 的 usage metering、labels、GPU、training waste、inference、cache、storage、network、artifact delete、budget、alerts、chargeback、dashboard、recommendation、tradeoff、unit economics 和 platform gate 字段；实现 usage metering coverage、cost attribution labels、GPU cost efficiency、training waste control、inference unit cost、cache savings accounting、storage lifecycle governance、network egress governance、artifact dependency safety、budget quota enforcement、cost anomaly alerting、tenant model chargeback、dashboard drilldown readiness、optimization recommendation trace、SLO quality cost tradeoff 和 cost governance gate；输出 GPU job cost、tokens per GPU hour、failed job cost、cost per 1k tokens、cache saved cost、storage monthly cost、network egress cost、budget used、idle GPU hours、unit margin、cost center count、metrics、hard blocker count、failed cases、failed gates 和 cost governance gate；要求覆盖 usage 缺 token 计量、成本标签缺 cost center、GPU idle 过高、失败训练任务成本过高、推理单位成本超阈值、缓存权限不安全、存储无 retention policy、跨地域网络成本过高、artifact 删除仍有回滚依赖、预算超阈值、成本告警无 owner、chargeback 缺模型版本、dashboard 不能下钻 request slice、优化建议未评估质量 / SLO 风险、降本伤害质量和没有最终成本治理门禁等 bad case。
35DJ. 写一个 0 依赖资源利用率治理审计 demo，输入 toy resource utilization cases 的 metrics、capacity、utilization records、effective output、fragmentation、topology、colocation、preemption、low priority、elastic、inference、warm pool、fairness、scheduler、dashboard、reliability、tradeoff 和 platform gate 字段；实现 resource metric contract、effective utilization accounting、fragmentation control、topology binpacking readiness、colocation SLO guard、preemption checkpoint safety、low priority reclaim policy、elastic training safety、inference capacity utilization、warm pool headroom policy、quota fairness borrowing、scheduler observability、dashboard drilldown readiness、reliability headroom guard、cost SLO tradeoff control 和 resource utilization gate；输出 allocation rate、active utilization、effective utilization、fragmentation ratio、colocation latency delta、colocation step delta、preemption loss GPU-min、global batch、adjusted grad accumulation、good output tokens/s、warm pool cost、cold start penalty、headroom、dominant share、borrowed GPU、metrics、hard blocker count、failed cases、failed gates 和 resource utilization gate；要求覆盖指标契约缺 fragmentation、有效产出被高估、资源碎片失控、拓扑装箱缺 score trace、混部 p99 超阈值、抢占无 checkpoint、低优先级队列不可回收、弹性训练 batch 漂移、推理 p99 超 SLO、warm pool 策略缺失、配额借用不可归还、调度事件不可观测、dashboard 不能下钻 endpoint、headroom 不足、优化建议伤害 SLO 和没有最终资源利用率门禁等 bad case。
35DK. 写一个 0 依赖安全治理审计 demo，输入 toy security cases 的 identity、permission、secret、data、PII、tenant、model access、output、RAG、Agent、supply chain、artifact、runtime、logs、audit response 和 platform gate 字段；实现 identity workload binding、RBAC ABAC policy fit、secret lifecycle management、data compliance classification、PII sensitive data control、tenant isolation boundary、model access control、model output safety gate、RAG permission enforcement、agent tool safety gate、supply chain integrity、model artifact integrity、runtime security boundary、log trace privacy governance、audit incident response readiness 和 security governance gate；输出 identity coverage、overprivilege rate、secret exposure rate、secret rotation coverage、compliance coverage、PII redaction coverage、sensitive block rate、tenant leak rate、output safety pass rate、RAG block rate、tool confirmation coverage、supply chain coverage、runtime hardening coverage、log redaction coverage、audit coverage、incident response ready、metrics、hard blocker count、failed cases、failed gates 和 security governance gate；要求覆盖 workload 共享管理员身份、RBAC / ABAC 策略缺默认拒绝、密钥泄露、数据训练授权缺失、PII 脱敏不足、跨租户命中、模型下载权限缺失、高风险输出未人审、RAG 未阻断无权限 chunk、Agent 工具越权、供应链 artifact 未签名、模型 artifact 签名缺失、runtime root 风险、日志 trace 脱敏不足、事件响应 runbook 缺失和没有最终安全治理门禁等 bad case。
35DL. 写一个 0 依赖审计变更治理 demo，输入 toy audit / retention / incident / change cases 的 audit schema、critical operations、audit integrity、retention、redaction、trace、timeline、incident metrics、root cause、actions、change、high risk change、rollout、freeze budget、linkage 和 platform gate 字段；实现 audit event schema completeness、critical operation audit coverage、audit log integrity retention、log retention policy fit、log minimization redaction、trace privacy sampling、incident timeline completeness、impact detection response metrics、root cause systemic analysis、postmortem action item closure、change record completeness、high risk change approval、rollout rollback readiness、change freeze error budget policy、audit postmortem change linkage 和 governance loop gate；输出 audit schema coverage、critical audit coverage、immutable coverage、retention policy fit、redaction coverage、log minimization rate、trace sample rate、timeline coverage、MTTD、MTTA、MTTR、root cause coverage、action closure rate、change record coverage、freeze execution rate、loop coverage、metrics、hard blocker count、failed cases、failed gates 和 governance loop gate；要求覆盖 audit schema 缺 trace id、关键操作审计不足、审计日志不可防篡改、留存策略缺删除请求、日志最小化不足、Trace 敏感信息未降采样、事故时间线不完整、响应指标超阈值、根因停在个人失误、行动项关闭率低、变更记录缺 rollback、高风险变更审批不足、灰度回滚不就绪、冻结期仍放行高风险变更、audit / postmortem / action item 无链路和没有最终治理闭环门禁等 bad case。
35DM. 写一个 0 依赖企业级 LLMOps 平台系统设计审计 demo，输入 toy applications、release manifest、prompt、RAG candidates、tool calls、trace runs、feedback、cost records 和 design cases 字段；实现 lifecycle boundary clarity、application resource model、model gateway policy、prompt registry versioning、RAG KB permission governance、tool agent runtime control、eval feedback loop、release manifest governance、trace observability readiness、cost budget governance、security audit policy、multi-environment promotion、developer interface readiness、production monitoring guardrail、trade-off boundary reasoning 和 LLMOps platform gate；输出 lifecycle summary、LLMOps examples、cost summary、metrics、hard blocker count、failed case sample、failed gate sample 和 LLMOps platform gate；要求覆盖生命周期边界不清、资源模型缺失、模型网关策略缺失、prompt registry 缺版本、RAG 权限缺失、Agent 工具控制缺失、评估反馈不闭环、release manifest 缺失、trace 观测缺失、成本预算缺失、安全审计缺失、多环境靠手工提升、开发者接口缺失、生产 guardrail 缺失、trade-off 边界缺失和没有最终平台门禁等 bad case。
35DN. 写一个 0 依赖多租户 GPU 集群调度系统设计审计 demo，输入 toy queues、nodes、resource snapshot、preemption victims 和 design cases 字段；实现 tenant queue contract、quota borrowing policy、dominant fairness accounting、gang scheduling fit、topology aware placement、fragmentation control、preemption checkpoint safety、low priority reclaim、inference SLO isolation、heterogeneous GPU policy、resource snapshot health、placement explainability、cost utilization attribution、scheduler observability audit、trade-off boundary reasoning 和 multi tenant GPU scheduler gate；输出 quota usage、borrowed GPUs、dominant shares、fairness gap、gang feasible、topology score、fragmentation ratio、preemption loss GPU-min、effective utilization、metrics、hard blocker count、failed case count、failed gate count 和 multi tenant GPU scheduler gate；要求覆盖队列契约缺失、quota borrowing 不可回收、公平性只看 FIFO、gang 任务半启动、topology ignored、碎片失控、抢占无 checkpoint、低优先级不可回收、推理 SLO 被训练混部影响、异构 GPU 无策略、resource snapshot 陈旧、placement 无解释、成本利用率不可归因、调度观测缺失、trade-off 边界缺失和没有最终调度门禁等 bad case。
35DO. 写一个 0 依赖 AI Infra 面试准备度审计 demo，输入 toy mock interview answers 的 topics、formulas、demos、risks、tradeoffs、score、red flags 和 revision plan 字段；实现 scope boundary clarity、training platform lifecycle、GPU scheduler quota fairness、training debug evidence、inference platform token SLO、TTFT / TPOT / KV formula、RAG Agent permission trace、model registry release gate、eval platform reproducibility、observability correlation、SLO error budget readiness、cost governance、security multitenancy、incident postmortem、trade-off boundary reasoning 和 interview revision gate；输出 topic coverage、formula coverage、demo evidence coverage、risk coverage、trade-off coverage、average score、red flag rate、metrics、hard blocker count、failed case count、failed gate count 和 AI Infra interview gate；要求覆盖把 AI Infra 说成工具名、TrainingJob 只说 YAML、GPU 调度只说 FIFO、训练故障缺证据、推理只看 QPS、TTFT / TPOT / KV 公式缺失、RAG / Agent 权限 trace 缺失、模型发布门禁缺失、评估不可复现、可观测性只有 dashboard、SLO 没有错误预算、成本没有单位 token、跨租户风险缺失、事故无复盘、trade-off 边界缺失和没有修复计划等 bad case。
35DP. 写一个 0 依赖 LLM Serving Engine 总览审计 demo，输入 toy serving engine cases 的 lifecycle、requests、phase contract、model config、KV manager、scheduler、streaming、metrics、runtime modules、traffic、cost 和 final gate 字段；实现 request lifecycle coverage、prefill decode contract、KV cache formula fit、scheduler batching readiness、streaming state management、metrics observability、runtime boundary clarity、cost capacity model 和 serving engine gate；输出 TTFT P95、TPOT P95、E2E P99、input tokens/s、output tokens/s、KV cache MiB、KV pressure、cost per 1k tokens、continuous batching gain、metrics、hard blocker count、failed case count、failed gate count 和 remediation sample；要求覆盖 lifecycle 缺 cleanup、prefill / decode 指标缺失、KV admission 缺失、静态 batching、streaming backpressure 缺失、metrics 缺 TPOT、runtime 边界不清、成本容量缺失和没有最终 serving engine 门禁等 bad case。
35DQ. 写一个 0 依赖从零实现最小推理框架 demo，包含 toy tokenizer、toy next-token model、RequestState、waiting queue、running set、finished set、KV token cache、scheduler、streaming chunks、event trace 和 metrics；先实现单请求 naive generate，再实现 `MiniFromScratchServingEngine`，输出 naive output、每个请求的 stream、TTFT steps、TPOT steps、decode steps、engine metrics 和 minimal engine gate；要求覆盖 request state machine、prefill / decode metrics、streaming chunks、KV cleanup、event trace 五个门禁。
35DR. 写一个 0 依赖推理请求生命周期审计 demo，输入 3 个 toy requests，分别覆盖正常完成、客户端取消和排队超时；实现 request object、RECEIVED / VALIDATED / TOKENIZED / WAITING / PREFILLING / DECODING / FINISHED / ABORTED / TIMEOUT 状态机、waiting queue、running set、KV token cache、streaming chunks、finish reason、queue steps、TTFT steps、TPOT steps、event trace 和 cleanup；输出每个请求的 state、reason、queue steps、TTFT、TPOT、prefill steps、decode steps、stream、trace，以及 terminal / finished / aborted / timeout 计数、avg queue steps、p95 TTFT、p95 TPOT、max KV tokens、KV tokens after cleanup 和 request lifecycle gate；要求验证正常和异常路径都能释放 KV 并留下 finish reason。
35DS. 写一个 0 依赖 Prefill / Decode / KV Cache / Token Streaming 阶段审计 demo，输入 3 个 toy requests，字段包含 prompt tokens、output tokens、client read frequency 和 stream buffer capacity；实现 prefill steps、decode steps、TTFT steps、TPOT steps、KV MiB、KV pressure、stream chunks、stream backlog、backpressure events、finish reason 和 phase gate；要求输出每个请求的 prompt / output tokens、prefill / decode steps、TTFT、TPOT、KV MiB、stream chunks、max stream backlog、backpressure events，以及全局 total prompt tokens、total output tokens、sum prefill steps、sum decode steps、p95 TTFT、p95 TPOT、total KV MiB、KV pressure、stream events、event tail，并验证 prefill accounting、decode accounting、KV budget、streaming chunks 和 cleanup finish reason 五个门禁。
35DT. 写一个 0 依赖 LLM serving 指标和成本审计 demo，输入 toy workload，每个请求包含 input tokens、output tokens、queue ms、tokenize ms、prefill ms、first flush ms、decode ms、final flush ms 和 timeout；实现 TTFT、TPOT、E2E、KV MiB、input tokens/s、output tokens/s、total tokens/s、TTFT p95 / p99、TPOT p95、E2E p99、KV pressure、active sequence capacity、cost per 1k tokens 和 timeout rate；输出 per-request 指标、summary 和 metric cost gate，并要求至少构造一个 TTFT p95 或 E2E p99 不过线的样本，说明吞吐、KV 和成本过线不等于用户体验过线。
35DU. 写一个 0 依赖推理框架 / 推理平台 / AI Infra 边界审计 demo，输入 toy capabilities 和 incidents，capabilities 至少包含 engine 的 request queue / scheduler / prefill decode / KV / streaming / metrics，platform 的 API gateway / auth quota / routing rollout / SLO / cost / observability，infra 的 GPU cluster / runtime / driver / pod scheduling / network storage / node health / capacity；实现 layer coverage、interface contract coverage、metric handoff coverage、misroute rate、role gate、interface gate、metric gate、incident gate、cost gate 和 serving boundary gate；输出 layer counts、missing by layer、metrics summary、misrouted incidents、gates 和 final boundary gate，并至少构造一个 TTFT p99 事故只查 engine、真实 root layer 在 platform / infra 的 bad case，说明职责名词完整不等于事故归层正确。
35DV. 写一个 0 依赖最小 tokenizer、model wrapper 和 generate loop demo，包含 `ToyTokenizer.encode/decode`、`ToyModelWrapper.forward`、`greedy_select` 和 `generate`；输入 prompt，输出 generated text、token ids、每轮 trace、prompt tokens、generated tokens、forward calls、naive token work、KV-like work、duplicate work 和 minimal generate gate；要求至少验证 tokenizer contract、model wrapper contract、generate loop trace、EOS stop condition 和 naive recompute visible 五个门禁，并解释为什么这个 toy loop 是理解后续 KV cache、batching、scheduler 和 streaming 的起点。
35DW. 写一个 0 依赖 greedy、temperature、top-k、top-p sampling demo，输入 toy logits 和 vocab，实现 stable softmax、temperature scaling、top-k filter、top-p nucleus filter、概率重新归一化、seeded multinomial sample 和 sampling gate；输出每种策略的 selected token、candidate list、candidate count、entropy、filtered probabilities、softmax normalized、temperature entropy changed、top-k reduced candidates、top-p adaptive candidates、seed reproducible 和 sampling gate；要求至少构造 greedy、temperature 低温、高温、top-k、top-p 和组合过滤六种 case，并说明为什么 sampling 会影响输出长度、停止条件、TPOT 和成本。
35DX. 写一个 0 依赖最小 KV Cache demo，输入 toy prompt tokens 和 generated tokens，用单头 causal attention 同时实现完整重算路径和 cached decode 路径；prefill 阶段缓存 prompt 的 key / value，decode 阶段每步只为新 token 计算 Q/K/V 并追加 cache；输出每步 token、seq_len、完整重算 hidden、cache hidden、equivalence check、attention last prob、final cache length、naive project calls、kv project calls、saved project calls、toy KV bytes 和 KV cache gate；要求说明 KV Cache 为什么能减少重复计算、为什么会增加显存，以及 attention_mask / position_ids 错误会如何破坏输出正确性。
35DY. 写一个 0 依赖 batched prefill demo，输入 3 个不同长度 toy prompts，实现 tokenizer、left padding、right padding、attention mask、toy model logits、最后真实 token logits gather 和 greedy 首 token 选择；输出 lengths、left input ids、left attention mask、left next tokens、right naive next tokens、right last indices、right gather next tokens、padding waste ratio、real KV bytes、padded KV bytes 和 batched prefill gate；要求验证左 padding 下取最后位置 logits 可用、右 padding 下 naive last logits 会错、按 attention mask gather 后结果恢复正确，并说明 padding waste、prefill token budget 和 batch KV cache shape 的工程意义。
35DZ. 写一个 0 依赖 batched decode demo，输入 3 个已完成 prefill 的 toy requests，每个请求包含 request id、prompt length、首 token、目标输出序列和 max new tokens；实现 finished mask、逐请求停止条件、position id / attention mask length 增长、batch row -> request id -> cache slot 映射和 active batch compaction；输出每轮 active ids、emitted tokens、position ids、attention mask lens、row to cache slot、alignment ok、每个请求最终 output / finish reason / cache len、fixed rows、compact rows、saved rows 和 batched decode gate；要求至少构造一个短请求先 EOS、一个中等请求后 EOS、一个长请求最后完成的样本，说明 fixed batch 和 compact batch 的代价差异。
35E0. 写一个 0 依赖请求队列和简单 scheduler demo，输入 4 个动态到达的 toy requests，每个请求包含 request id、arrival time、prompt token 数、planned output tokens 和 max new tokens；实现 FIFO waiting queue、running set、finished set、decode-first 调度、max active sequences、max batched tokens、prefill admission、deferred reason、cleanup 和 trace；输出 finished order、queue wait、TTFT steps、finish reasons、budget trace、max running、deferred count、trace tail 和 simple scheduler gate；要求至少构造一个长 prompt 因 token budget 延迟、一个请求与 decode 同轮 prefill、一个请求 prefill 后立即 EOS，并说明 TTFT 与 TPOT trade-off。
35E1. 写一个 0 依赖 token streaming demo，输入 4 个 toy requests，分别覆盖正常 EOS、跨 token stop sequence、客户端取消和慢客户端 backpressure；实现 StreamingEvent、TextStreamer、完整 decode 差分、每请求 stream queue、finish event、stop sequence holdback、client_cancelled finish reason 和 backpressure 检测；输出 stream summary、trace tail、first token streamed、full text reconstructed、finish event present、stop sequence hidden、client cancelled、backpressure detected 和 streaming gate；要求说明为什么 streaming 不是单 token `print`，以及 TTFT、TPOT、finish event、取消清理和背压指标如何进入 serving engine 验收。
35E2. 写一个 0 依赖最小 HTTP API demo，输入 toy JSON payloads，分别覆盖同步 `/generate`、SSE 风格 streaming、空 prompt 400、队列满 429 和客户端取消；实现 GenerateRequest、RequestState、ToyEngine、MinimalHTTPAPI、request validation gate、queue admission control、sync response、SSE frame format、streaming finish event、HTTP cancellation cleanup 和 HTTP API gate；输出 sync response、stream frame count、stream first frame、invalid response、busy response、cancel finish reason、KV released request ids 和 gates；要求说明为什么 HTTP handler 不能直接调用 `model.generate()`，以及 API 层如何和 engine 的 scheduler、stream queue、KV cleanup 解耦。
35E3. 写一个 0 依赖 LLM serving 压测指标 demo，输入 4 条 toy request trace，至少覆盖短请求、长 prompt、burst 和 cleanup；实现 TTFT、TPOT、E2E latency、queue wait、input tokens/s、output tokens/s、queue length p95、active request p95、KV memory peak、cleanup 后 KV 回落、SLO pass 和 bottleneck classification；输出 per-request metrics、benchmark summary、slo pass、bottleneck 和 benchmark gate；要求说明为什么只看 QPS 会误判，以及 TTFT 高但 TPOT 正常时应优先排查 queue、prefill、长 prompt 和 scheduler。
35E4. 写一个 0 依赖 vLLM 动机审计 demo，输入 4 个 toy request profiles，每个请求包含 prompt tokens、max new tokens 和 actual new tokens；实现 naive KV reservation、paged KV block allocation、KV waste ratio、logical to physical block mapping、finished / cancelled block cleanup、block reuse、static decode rows、continuous decode rows、row saving ratio 和 vLLM motivation gate；输出 request memory rows、memory summary、batch summary 和 gates；要求说明为什么 vLLM 不是单个 PagedAttention kernel，而是 KV block 管理、continuous batching、cleanup 和 metrics 共同组成的 serving engine。
35E5. 写一个 0 依赖 PagedAttention 核心 demo，输入 toy physical block ids 和两个共享 prefix 的请求；实现 RequestKV、ToyPagedKVCache、block table、token position 到 logical block / offset / physical block 的地址翻译、append token 时按 block 分配、prefix block ref count、release request、freed block reuse、internal block waste 和 paged attention gate；输出 paged KV summary、trace tail、address translation、shared ref counts、released blocks、ref counts after release、free list head、waste 和 gates；要求说明 PagedAttention 为什么是分页式 KV cache 管理，而不只是更快的 attention kernel。
35E6. 写一个 0 依赖 KV Cache Block Manager demo，输入 toy block pool、block size、prefill / decode / fork / release / admission failure 操作序列；实现 SequenceState、ToyBlockManager、required blocks、free list、ref count、prefill block admission、decode block extension、fork prefix、release、block reuse、allocation failure、metrics 和 block manager gate；输出 block manager summary、trace tail、free / used / shared blocks、allocation failures、blocks per sequence 和 gates；要求说明 scheduler 为什么必须查询 block manager，而不能只按请求数做 admission control。
35E7. 写一个 0 依赖 Continuous Batching demo，输入 5 个动态到达的 toy requests，至少覆盖同一时刻到达、中途到达、短请求先完成、长 prompt 被 token budget 延迟、KV block 不足被延迟和等待中取消；实现 Request、ToyBlockManager、ToyContinuousBatcher、iteration-level scheduling、decode-first、prefill admission、token budget、KV block budget、deferred reason、cleanup、TTFT steps、static batch hole 和 continuous batching gate；输出 finished order、cancelled ids、queue wait steps、TTFT steps、static decode rows、continuous output rows、saved rows、max KV blocks used、deferred reasons、trace tail 和 gates；要求说明 continuous batching 不是把 batch size 开大，而是每轮重组 execution batch，并用 token / KV / cleanup / metrics 共同验收。
35E8. 写一个 0 依赖 vLLM-like 请求调度流程 demo，输入 4 个 toy requests，分别覆盖正常 EOS、stop token / stop string 结束、`max_tokens=1` 首 token 后结束和 running 后客户端 abort；实现 RequestSpec、RequestState、ToyKVBlockManager、ToyRequestFlowEngine、input processor、waiting queue、scheduler output metadata、model runner execution metadata、slot mapping、output processor、abort cleanup、queue wait、TTFT、E2E 和 request flow gate；输出 finished order、aborted ids、finish reasons、queue wait steps、TTFT steps、E2E steps、metadata rows、max KV blocks used、cleanup 后 KV blocks、state transitions、trace tail 和 gates；要求说明 scheduler output 不能只是请求列表，而必须包含 block table、slot mapping、positions、phase 和 finish / cleanup 证据。
35E9. 写一个 0 依赖 vLLM memory management demo，输入 toy block pool、block size、两个共享 prefix 的请求、一个 extra hash 不同的请求和一个触发 preemption / recompute 的请求；实现 KVBlock、ToyKVMemoryManager、block pool、free queue、request block table、prefix cache hash、parent hash、extra hash、cached free block、ref count、LRU eviction、allocation failure、preemption recompute path、prefix hit rate、memory metrics 和 memory management gate；输出 A/B/C/D block tables、shared ref counts、release 后 ref counts、cached free blocks、metrics、trace tail 和 gates；要求说明 free 不等于 evict、cached-but-free 可以被后续 prefix 命中但也可被 LRU 覆盖，以及频繁 preemption 说明 KV budget 或并发参数需要调优。
35E10. 写一个 0 依赖 vLLM worker / executor / engine 架构 demo，输入 toy scheduler output 和 TP=2 worker group；实现 SchedulerOutput、WorkerInput、ToyModelRunner、ToyWorker、ToyExecutor、ToyEngineCore、engine core dispatch loop、executor dispatch layer、worker rank / local rank / device mapping、model runner metadata gate、tensor parallel shard merge、CPU process budget、worker failure recovery 和 executor architecture gate；输出 expected worker count、actual worker count、process budget、CPU budget、first outputs、failure outputs、request status、executor trace、engine trace 和 gates；要求说明 scheduler 决定谁跑，executor 决定如何派发，worker 负责设备和模型执行，model runner 负责把 request-level plan 转成 tensor-level metadata。
35E11. 写一个 0 依赖 Prefix Caching 与 Prompt Cache demo，输入 toy block pool、block size、共享前缀请求、cache salt 不同请求、LoRA / media hash 不同请求和触发 LRU eviction 的请求；实现 KVBlock、ToyPrefixCacheManager、full block split、parent hash chain、extra hashes、cached-free block、touch、ref count、free queue、LRU eviction、saved prefill tokens、prefix cache hit rate、cached-free block count 和 prefix cache gate；输出 A/B/C/D block tables、B 命中 blocks、saved prefill tokens、run prefill tokens、touch 后 free queue、salt / LoRA / media miss、evicted blocks、cached-free blocks、metrics 和 gates；要求说明 prompt cache 是平台层概念，prefix caching 是 runtime KV full block 复用机制，不能跨模型、tokenizer、LoRA、多模态输入或租户隔离边界误复用。
35E12. 写一个 0 依赖 TP / PP / DP / EP serving 并行审计 demo，输入 7B 单卡多副本、70B 单节点 TP、跨节点 TP 反例、TP+PP 两节点、TP+DP 扩吞吐和 MoE EP 六类 toy 配置；实现 ParallelCase、ToyServingParallelAuditor、总 GPU 数、worker group size、per-rank weight / KV / buffer 显存估算、TP 跨节点检测、TP 通信估算、PP bubble、DP throughput、prefix cache locality、EP expert imbalance 和 parallel serving gate；输出每类配置的 memory fit、topology ok、TPOT、bubble ratio、throughput QPS、prefix locality、expert imbalance、summary 和 gates；要求说明 TP/PP 解决单副本放不下，DP 解决横向吞吐，EP 解决 MoE expert 容量，跨节点 TP 和 DP 后 cache locality 都必须显式验收。
35E13. 写一个 0 依赖 vLLM 性能调优审计 demo，输入 toy request traces 和 serving config，至少覆盖短请求、长 RAG queue / prefill、高 TPOT 长输出、CPU / streaming 慢、DP 路由打散 prefix cache 和跨节点 TP 风险；实现 RequestTrace、ServingConfig、ToyVLLMTuningAuditor、TTFT / TPOT / E2E、queue p95、output tokens/s、KV pressure、preemption rate、prefix hit rate、saved prefill ratio、scattered prefix routes、root cause diagnosis、config trade-off recommendations、rollback policy 和 vLLM tuning gate；输出 per-request metrics、summary、root causes、recommendations、rollback policy 和 gates；要求说明 `max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`max_model_len`、chunked prefill、prefix cache、TP / DP 调整分别改善什么指标、牺牲什么指标，并说明为什么调优必须用真实流量 replay 和灰度回滚验证。
35E14. 写一个 0 依赖 SGLang 动机审计 demo，输入一个 toy complex LLM program，至少覆盖长文档 root、字段抽取、A / B 两个条款分支、分支内法规检查和最终 JSON 报告；实现 ProgramCall、ToyRadixPrefixCache、ToySGLangMotivationAudit、longest prefix matching、naive prefill tokens、radix prefill tokens、saved prefill tokens、reuse ratio、branch sharing summary、structured decoding retry saving、grammar overhead、naive work tokens、SGLang work tokens、work reduction 和 SGLang motivation gate；输出 reuse rows、branch summary、summary 和 gates；要求说明 SGLang 的价值来自 frontend language 暴露复杂程序结构与 runtime 复用 prefix / 约束 decoding 的协同，而不是简单替换一个 OpenAI-compatible endpoint。
35E15. 写一个 0 依赖 SGLang Runtime 总览审计 demo，输入 OpenAI-compatible API、native `/generate`、frontend program 和 offline engine 四类 toy 请求；实现 RuntimeRequest、ToyRadixCache、ToyMemoryPool、ToySGLangRuntimeAudit、entrypoint unification、request state、longest prefix lookup、run prefill tokens、KV slot allocation / release、prefill token budget scheduling、continuous decode loop、grammar mask step、streaming event trace、summary metrics 和 SGLang runtime gate；输出 request rows、prefill rounds、decode trace、runtime summary 和 gates；要求说明 SGLang Runtime 不是 HTTP wrapper，也不是单独的 RadixAttention，而是入口、状态、调度、cache、memory pool、model runner、sampler、grammar backend 和 streaming 共同组成的执行系统。
35E16. 写一个 0 依赖 RadixAttention 与 prefix sharing 审计 demo，输入 4 条共享 root / 分支 / 深层 continuation 的 toy token 序列；实现 RadixNode、ToyRadixAttentionCache、compressed prefix tree、longest prefix match、page aligned prefix hit、insert、edge split、KV token accounting、pin path / ref count、leaf LRU eviction、cached token summary、scheduler suffix cost 和 RadixAttention gate；输出每个请求的 input tokens、token hit、page hit、hit owner、run prefill tokens、new cached tokens、split count、cached tokens before / after eviction、pinned nodes、evicted leaf nodes、page examples、summary 和 gates；要求说明 RadixAttention 不是 attention kernel，主要优化 prefill / TTFT，且必须证明 token prefix 完全匹配、page / block 完整命中、active path 不被淘汰和 scheduler 使用 suffix cost。
35E17. 写一个 0 依赖 SGLang scheduler 审计 demo，输入 running streaming 请求、waiting cache hit 长 prompt、等待很久的 cache miss 请求、structured output 请求和 aborted 请求；实现 Request、CacheNode、ToyMemoryPool、ToyRadixScheduler、decode-first、cache-aware admission、suffix cost scheduling、token budget、sequence budget、KV slot admission、safe cache eviction、pin matched radix prefix、aging fairness、chunked prefill、grammar step accounting、abort cleanup、deferred reason 和 SGLang scheduler gate；输出 scheduled decode ids、scheduled prefill rows、scheduled tokens、scheduled sequences、prefill suffix tokens、naive prompt tokens、grammar steps、cleaned aborts、evicted nodes、pinned nodes、KV free after scheduling、deferred reasons、trace 和 gates；要求说明 scheduler 不能只看 prompt length 或 batch size，而要同时证明 TTFT / TPOT、KV capacity、prefix reuse、公平性、structured output 和 cleanup 都可观测。
35E18. 写一个 0 依赖 structured generation / constrained decoding 审计 demo，输入 toy logits 和 toy grammar states，至少覆盖 JSON schema-like object、regex / choices 分类、EBNF-like sequence、structural tag tool call、空合法 token 集合和 streaming partial JSON；实现 TinyJSONGrammar、ChoiceGrammar、SequenceGrammar、valid token mask、constrained argmax、grammar state accept、mask calls、blocked high-score invalid tokens、parseability check、fact check、stream chunk parseability 和 structured generation gate；输出 constrained JSON text、choice text、EBNF text、structural tag text、mask calls、blocked invalid tokens、empty valid detection、stream partial parseability、format-not-fact 证据和 gates；要求说明 constrained decoding 能保证格式和部分类型约束，但不能保证事实正确、业务规则正确或工具安全，streaming 中间 chunk 也不能当完整对象解析。
35E19. 写一个 0 依赖 speculative decoding 审计 demo，输入高接受率代码请求、低接受率开放写作请求和带 grammar 约束的 JSON tool 请求；实现 ToyRequest、ToySpeculativeDecoder、draft window、target verify、accepted prefix、fallback token、grammar-blocked draft、baseline target calls、spec target calls、drafted / accepted / rejected token 统计、temporary KV cleanup、acceptance rate、tokens per target call、target call reduction、latency speedup estimate 和 speculative decoding gate；输出每个请求的 match target、baseline calls、spec calls、accepted / drafted / fallback / grammar blocked、trace tail、全局 summary 和 gates；要求说明 speculative decoding 主要优化 decode / TPOT，收益取决于接受率和 draft overhead，rejected token 不能进入正式 KV 或 RadixAttention prefix cache，structured output 会影响 draft 候选和 grammar state 回滚。
35E20. 写一个 0 依赖 multi-turn / tool use / agent serving 审计 demo，输入同一 session 的 weather tool use 和 follow-up 请求，以及一个需要确认但未确认的高风险 email tool 请求；实现 ToolSpec、AgentTask、ToyReplica、ToyAgentServingAudit、session-aware routing、prefix cache lookup / insert、tool call streaming fragments 拼接、tool parser、schema / permission validator、tool execution、tool result backfill、tool wait GPU slot release、agent round budget、task-level metrics 和 agent serving gate；输出每个 task 的 replica、prompt length、hit length、run prefill、tool name、valid / reason，全局 tasks、model calls、tool calls requested / executed / blocked、parse success、validation failures、tool latency、naive / run / saved prefill tokens、reuse ratio、model calls per task、blocked tool rate、session affinity、replica model calls 和 gates；要求说明 agent serving 的单位是 task / session，不是单次 completion，工具等待不能占用 GPU decode slot，高风险工具阻断是安全证据，性能要同时看模型、cache、工具、scheduler 和 agent loop。
35E21. 写一个 0 依赖 SGLang vs vLLM 架构对比审计 demo，输入 independent chat、shared RAG document、structured JSON extraction 和 agent branch tree 四类 toy workloads；实现 Workload、ToyArchitectureComparator、full block prefix hit、vLLM-like saved prefill、SGLang-like radix / branch saved prefill、workload fit score、recommended backend、saved ratio、SGLang extra saved 和 architecture comparison gate；输出每类 workload 的 calls、naive prefill、vLLM saved / run prefill、SGLang saved / run prefill、vLLM fit、SGLang fit、recommended，全局 prompt tokens、saved prefill、run prefill、vLLM wins、SGLang wins、saved ratios 和 gates；要求说明 PagedAttention 关注 KV block / 显存布局，RadixAttention 关注 token prefix / program trajectory 复用，vLLM 没有被 SGLang 替代，复杂 program 也不是普通 chat QPS 能概括，平台层应按 workload 做 runtime router。
35E22. 写一个 0 依赖 mini-sglang 源码路径审计 demo，输入源码阅读模块笔记和实验 trace；实现 ModuleNote、ExperimentTrace、ToySourcePathAuditor、required module coverage、request lifecycle order check、source resource coverage、source experiment coverage、experiments touched runtime module coverage、observable signal count、missing modules / resources / experiments 和 source path gate；模块至少覆盖 api server、tokenizer、request state、scheduler、KV cache、radix cache、engine、attention backend、sampler、detokenizer、streaming、cleanup；实验至少覆盖 minimal generate、shared prefix、radix split、mixed prefill/decode、max tokens finish、streaming、structured output 和 abort cleanup；要求说明读源码不是背类名，而是用实验和指标证明 request lifecycle、KV lifecycle、RadixAttention、scheduler、sampler、streaming 和 cleanup 都可复盘。
35E23. 写一个 0 依赖 Prefill / Decode 资源画像审计 demo，输入 RAG 长 prompt、短 chat、代码长输出和 agent 上下文增长四类 toy request；实现 RequestProfile、ToyPrefillDecodeProfiler、run prefill tokens、prefill compute units、prefill KV write MiB、decode KV read MiB、TTFT / TPOT 估算、prefill / decode token ratio、long prefill stall、chunked prefill stall、decode-first prefill wait、PD KV transfer cost 和 prefill decode profile gate；输出每个请求的 run prefill、decode steps、prefill / decode compute、KV write / read、TTFT、TPOT，全局 prefill compute、decode KV read、stall / wait / transfer 摘要和 gates；要求说明 Prefill / Decode 资源画像不能只背 compute-bound / memory-bound，而要用 token 分布、prefix hit、TTFT、TPOT、长 prefill 干扰、decode-first 饥饿和 KV transfer 共同证明 PD 分离动机。
35E24. 写一个 0 依赖 PD 分离动机审计 demo，输入 RAG 长 prompt、短 chat、代码长输出和 agent 上下文增长四类 workload；实现 Workload、ToyPDDisaggregationAuditor、unified engine prefill / decode time、unified decode jitter p99、unified prefill wait p99、PD prefill pool / decode pool time、P/D 独立 worker 配比、KV transfer MiB、largest request transfer time、interference saving、slow transfer counterexample 和 PD motivation gate；输出 workload rows、unified summary、PD summary、motivation summary 和 gates；要求说明 PD 分离不是一拆就更快，而是只有在 unified 干扰明显、jitter / starvation 下降、独立扩缩容有意义且 KV transfer 成本可控时才值得引入。
35E25. 写一个 0 依赖 PD 系统架构审计 demo，输入 prefill workers、decode workers 和正常请求、KV transfer 失败请求、client abort 请求；实现 Worker、PDRequest、ToyPDArchitectureAuditor、PD router worker selection、decode reservation、prefill running、KV transfer metadata、streaming finished path、transfer failure cleanup、client abort cleanup、model version compatibility check、metrics summary 和 PD architecture gate；输出每个请求的 states、prefill worker、decode worker、KV metadata、stream events、cleanup 标记，以及全局 finished、client aborted、transfer failed、transfer sessions、total transfer MiB、cleanup events、max decode running 和 gates；要求说明 PD 架构不是普通 LB 加两个 worker，而是跨 router、P/D worker、transfer backend 的可清理分布式状态机。
35E26. 写一个 0 依赖 KV Cache 迁移、共享和路由审计 demo，输入 3 个 decode worker candidate 和 long RAG、short chat、cross tenant 三类请求；实现 KVRequest、DecodeWorkerCandidate、ToyKVTransferRouter、KV metadata compatibility、decode capacity reservation、KV routing cost score、prefix reuse、tenant cache isolation、short prompt recompute fallback、transfer failure plan、skipped incompatible 统计和 KV transfer routing gate；输出每个请求的 decision、decode worker、required blocks、reserved blocks、KV transfer MiB、transfer ms、score、prefix reuse、tenant cache blocked、failure plan，以及全局 migrate、recompute、route_failed、total transfer MiB、reserved blocks、skipped incompatible 和 gates；要求说明 KV transfer 不是复制 bytes，而是 metadata、layout、容量、路由、隔离、重算兜底和失败清理共同治理的运行时能力。
35E27. 写一个 0 依赖 Chunked / Disaggregated Prefill 审计 demo，输入 long RAG、short chat 和 aborting agent long 三类请求；实现 PrefillJob、ToyChunkedDisaggPrefillAuditor、run prefill tokens、prefill chunk count、full prefill summary、chunked schedule、decode interleaving、prefill token budget、position continuity、short request fairness、PD transfer pipeline、backpressure cleanup 和 chunked disagg prefill gate；输出 chunked rows、full summary、chunked summary、PD transfer summary 和 gates；要求说明 Chunked Prefill 是 how to run a long prefill，Disaggregated Prefill 是 where to run prefill，二者结合时必须同时验收调度、公平性、position / KV 连续、partial transfer、backpressure 和 abort cleanup。
35E28. 写一个 0 依赖多级 KV Cache 审计 demo，输入 GPU active blocks、CPU idle block、remote reusable block、other tenant remote block、SSD cold block 和 resume / cross tenant / cold doc 三类请求；实现 KVBlock、KVRequest、ToyMultiLevelKVCacheAuditor、KV residency level、active GPU block protection、CPU KV promote、remote KV fetch、recompute fallback、KV tenant isolation、KV demotion policy、residency metrics 和 multi level KV gate；输出每个 block 的 action、cost、transfer MiB、demoted victims，以及全局 GPU hit、CPU promote、remote fetch、SSD fetch、recompute、tenant block、demotion、GPU / CPU / remote residency、total transfer MiB、estimated latency 和 gates；要求说明多级 KV Cache 不能只看命中率，而要同时看 residency、移动成本、重算成本、TPOT 关键路径、正确性 key 和租户隔离。
35E29. 写一个 0 依赖跨节点 serving 网络瓶颈审计 demo，输入 same-node、same-rack RDMA、cross-rack 和 TCP fallback 四类链路，以及两个 decode worker candidate；实现 LinkProfile、DecodeWorker、ToyCrossNodeNetworkAuditor、TP cross-node collective cost、PP activation cost、PD KV transfer cost、remote fetch vs recompute、topology-aware routing、pending transfer backpressure、control/data plane separation 和 cross node network gate；输出 worker scores、TP same-node / cross-rack TPOT overhead、PP activation ms、PD KV transfer MiB / ms、remote hot/cold decision、router choice、pending transfer MiB、backpressure events、data/control plane MiB 和 gates；要求说明跨节点 serving 不能只看 GPU 数量，而要按 DP / TP / PP / PD / remote KV 的通信频率、关键路径、拓扑距离、p99 拥塞和 backpressure 分别治理。
35E30. 写一个 0 依赖 PD 分离收益/代价审计 demo，输入 enterprise RAG、short QA、code generation、cross-node bad network 和 long prompt no ops 五类 case；实现 PDCase、ToyPDTradeoffAuditor、PD benefit cost score、PD positive case、short request anti pattern、decode bottleneck anti pattern、slow transfer anti pattern、ops readiness gate、PD alternative path 和 PD tradeoff gate；输出每个 case 的 score、decision、reasons、alternatives，以及全局 cases、adopt_pd、rejected、anti_patterns、positive_cases、blocked_cases 和 gates；要求说明 PD 分离不是默认终态，只有 P/D 干扰收益大于 KV transfer 和系统复杂度成本，并且 metrics / cleanup / backpressure 齐备时才值得做。
35E31. 写一个 0 依赖单机 Serving Engine 到 PD 分离迁移审计 demo，输入 explicit request stage、model runner interface split、KV metadata contract、scheduler split、router state machine、same-node PD prototype、cleanup cancel paths、metrics backpressure fallback 和 cross-node PD readiness 九个 migration step；实现 MigrationStep、ToyPDMigrationAuditor、dependency readiness、capability coverage、same-node-before-cross-node 检查、interface readiness、cleanup paths、observability fallback readiness 和 PD migration gate；输出 migration rows、summary、gates；要求说明升级顺序必须是先接口和状态，再 scheduler / router，再 same-node 原型和 cleanup / fallback，最后才 cross-node PD，不能把 prefill / decode 直接拆成两个远程服务。
35E32. 写一个 0 依赖 nano-vLLM 源码学习审计 demo，输入 example、LLM、sampling params、Sequence、LLMEngine、Scheduler、BlockManager、ModelRunner、Attention、Sampler 和模型结构的模块笔记，以及 trace generate、prefill/decode step、scheduler budget、KV block reuse、prefix cache hit、decode finish cleanup 和 sampler params 七个实验 trace；实现 ModuleNote、ExperimentTrace、ToyNanoVLLMSourceAuditor、module coverage、resource coverage、experiment coverage、lifecycle order、runtime module touch、observed signal count 和 nano-vLLM source gate；输出 source summary、gates、missing modules / resources / experiments；要求说明源码学习不能只背文件名，而要证明 generate path、engine step、sequence state、scheduler decision、KV block lifecycle、model runner batch contract 和实验信号都可观察。
35E33. 写一个 0 依赖 tiny-LLM 学习路线审计 demo，输入 attention、RoPE、GQA、RMSNorm/MLP、model load、generate、sampling、KV cache、quantized matmul、FlashAttention、continuous batching、chunked prefill 和 paged attention 十三个 learning unit；实现 LearningUnit、ToyTinyLLMLearningAuditor、concept coverage、experiment coverage、dependency order、serving units、operator to model gate、generate sampling gate、KV cache equivalence gate、serving optimization gate、metrics experiments gate 和 tiny-LLM learning gate；输出 rows、summary、gates；要求说明 tiny-llm 不是只跑通文本生成，而是要证明 attention shape、RoPE position、GQA KV heads、generate/sampling、KV cache 等价性、TTFT/TPOT、batching trace、chunk position continuity 和 block table trace 都可验证。
35E34. 写一个 0 依赖 mini-sglang 学习路线审计 demo，输入 runtime request state、Radix Cache、chunked prefill、overlap scheduling、online serving、interactive shell、tensor parallel、kernel backend、structured generation boundary、tool/agent boundary 和 production gap map 十一个 runtime capability；实现 RuntimeCapability、ToyMiniSGLangLearningAuditor、capability coverage、experiment coverage、evidence count、runtime-specific count、dependency order、radix cache gate、chunked prefill gate、overlap scheduling gate、online serving gate、structured/tool boundary gate、abort cleanup gate、production gap gate 和 mini-sglang learning gate；输出 rows、summary、gates；要求说明 mini-sglang 的学习重点不是普通 scheduler/KV cache，而是 Radix Cache、chunked prefill、overlap scheduling、online serving、structured/tool 边界、abort cleanup 和完整 SGLang 差距。
35E35. 写一个 0 依赖教学项目核心模块抽象审计 demo，输入 Request、Scheduler、KVCacheManager、BatchBuilder、ModelRunner、Sampler、OutputProcessor、EngineLoop 和 Metrics 九个核心模块；实现 CoreModule、ToyCoreModuleAuditor、module coverage、state coverage、boundary coverage、dependency readiness、replaceable module count、scheduler/model runner boundary、KV manager ownership、BatchBuilder metadata contract、output processor cleanup boundary、metrics observability contract 和 core module gate；输出 rows、summary、gates；要求说明从 tiny-llm、nano-vLLM、mini-sglang 抽象自己的 mini engine 时，重点不是照抄目录，而是证明 Request 状态、waiting/running、KV blocks、batch metadata、logits、output tokens、TTFT/TPOT 的 ownership 和后续升级边界清楚。
35E36. 写一个 0 依赖 Naive Scheduler 到 Continuous Batching 升级审计 demo，输入 A/B/C 初始请求、X waiting 取消请求、D/E 中途到达请求、token budget、max prefill tokens、max running requests 和 KV block pool；实现 Request、ScheduleItem、ToyKVManager、ToyNaiveToContinuousBatcher、request-level batch lock、iteration boundary scheduling、dynamic request admission、dynamic request exit、decode-first、bounded prefill、KV capacity admission、cleanup 和 scheduler upgrade gate；输出 finished order、cancelled ids、queue wait steps、TTFT steps、static decode rows、continuous output rows、saved rows、max running、max KV blocks used、deferred reasons 和 trace tail；要求说明 continuous batching 不是简单增大 batch size，而是把 request lifetime 和 execution batch lifetime 解耦，并用 token/KV 预算与 cleanup 证明升级安全。
35E37. 写一个 0 依赖 Paged KV Cache 升级审计 demo，输入 A 的 chunked prefill / decode token、B 的 abort 请求、C 的复用请求、D 的 admission failure 请求、num blocks 和 block size；实现 RequestState、BatchItem、ToyPagedKVBlockManager、ToyBatchBuilder、required_blocks、can_allocate_until、allocate_until、physical_slot、fragmentation ratio、idempotent free、double free guard 和 paged KV upgrade gate；输出 A 的 block table 增长、decode extension block、B freed blocks、C reused blocks、D admission result、slot mapping、block tables、fragmentation before cleanup、allocation failures、free blocks after cleanup、idempotent free count 和 trace tail；要求说明 paged KV cache 不是只换存储结构，而是把 KV cache 变成 scheduler 可预算、BatchBuilder 可寻址、OutputProcessor 可释放和 metrics 可观测的资源。
35E38. 写一个 0 依赖 Prefix / Prompt Cache 升级审计 demo，输入 A 的可缓存 full prompt blocks、B 的同前缀不同 suffix 请求、C 的不同 tenant salt 请求、D 的不同 LoRA 请求、E 的 full prompt hit 请求、num blocks 和 block size；实现 KVCacheBlock、RequestState、ToyPrefixPromptCacheManager、full block split、parent hash chain、extra hashes、lookup and attach、suffix prefill start、cached-free touch、active/cached ref count、evict cached-free block、token hit ratio 和 prefix prompt cache upgrade gate；输出 A cached tables、B hit blocks、saved prefill tokens、run prefill tokens、suffix start、suffix positions、suffix slot mapping、touch 后 free queue、ref counts after attach/free、salt miss、LoRA miss、full prompt hit fallback、cached free before/after eviction、metrics 和 trace tail；要求说明 prompt cache 是平台层抽象，prefix cache 是 runtime KV full block 复用，不能跨 model/tokenizer/LoRA/tenant/media 边界误复用。
35E39. 写一个 0 依赖 Preemption / Recompute / Swap 升级审计 demo，输入 low/high running 请求、burst waiting 请求、cached-free block、GPU block pool、CPU block pool 和 block size；实现 RequestState、GPUBlockManager、CPUBlockManager、KV pressure trigger、cached block eviction before preemption、victim selection policy、preempt_by_recompute、resume_by_recompute、swap_out、swap_in、swap failure rollback 和 preemption upgrade gate；输出 free before、required blocks、cached block、evicted cache blocks、victims、freed by preemption、burst blocks、low recompute context、resume blocks、preempted count、swap out/in plan、swap in failure result、CPU table after failed swap in、metrics 和 gates；要求说明 preemption 是防 OOM 兜底机制，recompute 必须重算 prompt + output，swap in 失败不能丢 CPU KV 状态。
35E40. 写一个 0 依赖统一调度循环审计 demo，输入 running streaming 请求、低优先级 running 请求、prefix cache 命中 waiting 请求、prefix cache miss waiting 请求、block size、free blocks 和 cached-free blocks；实现 RequestState、PrefillItem、DecodeItem、SchedulePlan、ToyUnifiedBlockManager、ToyUnifiedScheduler、ToyBatchBuilder、ToyOutputProcessor、prefix lookup once、decode-first、suffix prefill、token/KV budget、cached-free eviction、preemption、commit plan、positions、slot mapping、block tables、output state update 和 unified scheduler loop gate；输出 prefix hits、decode items、prefill items、free before、required new blocks、evicted blocks、preempted requests、commit allocations、metadata positions、slot mapping、block tables、stream output tokens、computed tokens、free after、invariants 和 gates；要求说明完整 engine step 的验收不是只看生成 token，而是证明队列、KV blocks、prefix cache、preemption、BatchBuilder metadata、OutputProcessor 状态和不变量一起闭环。
35E41. 写一个 0 依赖 Serving Benchmark Framework 审计 demo，输入 baseline 和 candidate 两组 toy request traces、engine step traces、benchmark config 和 seed；实现 BenchmarkConfig、RequestTrace、ToyServingBenchmarkFramework、benchmark workload coverage、benchmark experiment fingerprint、request trace summary、engine step trace summary、SLO regression gate、throughput regression check、KV cleanup check、prefix effect check、preemption risk check、benchmark decision gate 和 serving benchmark framework gate；输出 baseline / candidate summary、candidate request rows、TTFT / TPOT / E2E p95 delta、output tokens/s delta、preemption delta、bottleneck change、decision 和 gates；要求说明 serving benchmark 不是只看 QPS，而是用 workload、trace、SLO、吞吐、KV、cache、preemption 和可复现实验指纹共同判断一次调参是否可信。
35E42. 写一个 0 依赖异步 Serving 架构审计 demo，输入 r1/r2/r3/r4 四个 raw requests、tokenizer queue、engine input queue、engine output queue、per-client stream queue 和 cancel queue；实现 BoundedQueue、RawRequest、TokenizedRequest、EngineOutput、RequestState、ClientStream、ToyAsyncServingRuntime、async boundary design、API admission boundary、tokenizer worker queue、engine input queue、bounded engine drain、engine output queue、per-client stream queue、stream queue backpressure、cancel signal queue、engine state ownership、idempotent async cleanup 和 async serving architecture gate；输出 API admitted / rejected、tokenized、first drain added、engine input backlog、finished、cancelled、client streams、queues empty、cleanup events、metrics 和 gates；要求说明异步 serving 不是给同步 loop 套 `async`，而是通过有界队列、状态所有权、backpressure、cancel signal 和幂等 cleanup 让 GPU engine core 不被 API、tokenizer、慢 client 或网络发送阻塞。
35E43. 写一个 0 依赖 Multi Worker Router 审计 demo，输入 chat/code worker、GPU group、worker load、prefix hot set、loaded adapters、r1/r2/r3/r4 四个 tokenized requests 和 heartbeat timestamp；实现 WorkerLoad、WorkerHandle、TokenizedRequest、RoutedRequestState、RouterConfig、ToyMultiWorkerRouter、router worker capability filter、request cost block estimate、load aware worker score、sticky route fallback、global router admission、worker heartbeat timeout、safe retry boundary、request worker map、worker selection imbalance 和 multi worker router gate；输出 round-robin choice、load-aware choice、sticky preferred、actual worker、big request admitted、code worker、unhealthy after timeout、state rows、selection counts、selection imbalance、metrics 和 gates；要求说明多 worker router 不能只做 round-robin，而要同时证明能力过滤、KV 容量、负载、prefix locality、过载 fallback、全局拒绝、heartbeat failure、未输出重试和已 streaming 显式失败。
35E44. 写一个 0 依赖 Distributed Parallel KV 审计 demo，输入 ParallelConfig、RequestProfile、TP ranks、PP stages、block size、link bandwidth 和 latency；实现 TPRankKV、PPStageKV、ToyDistributedParallelKVAuditor、parallel group boundary、TP head shard assignment、TP block table consistency、PP layer ownership、distributed KV byte accounting、collective communication cost、pipeline bubble ratio、KV migration decision、cross rank cleanup 和 distributed parallel KV gate；输出 worker unit、TP head ranges、head coverage、logical blocks、TP block table consistency、PP layer ranges、layer coverage、KV total MiB、per TP rank MiB、per PP stage MiB、communication report、pipeline bubble ratio、migration decision、cleanup events 和 gates；要求说明多 GPU inference 不是只看 GPU 数量，而要证明 TP / PP 的 worker 边界、KV 分片、逻辑 block 一致性、collective 成本、pipeline bubble、迁移代价和跨 rank / stage 清理都闭环。
35E45. 比较两种 toy serving 配置的 raw output tokens/s 与请求级 SLO goodput；输入每请求 TTFT、request-level TPOT、E2E、完成/取消状态、输出 token 数和测量窗口，指定 SLO 后分别计算 request goodput 与 token goodput，并展示“吞吐上升但达标请求减少”的反例；讨论请求被中途丢弃时 hard-threshold goodput 的偏差，以及 smooth-goodput 研究口径的目的与未标准化边界。
35E46. 写一个 0 依赖 OpenAI Compatible API 审计 demo，输入 bearer token、tenant model permission、chat completion request、unsupported field、long context request、rate-limited tenant 和 busy tenant；实现 APITenant、EngineRequest、ToyOpenAICompatibleAPI、API compatibility contract、chat request schema validation、SSE stream frame contract、OpenAI style error shape、bearer auth gate、model permission gate、token rate limit gate、concurrency limit gate、usage accounting check、API privacy log check 和 OpenAI compatible API gate；输出 non-stream response、stream chunk count、first delta、last `[DONE]`、usage、auth error、unsupported field error、permission error、context error、RPM error、concurrency error、metrics 和 gates；要求说明 OpenAI-compatible API 不是简单把 engine 包成 HTTP，而要证明字段契约、流式帧、错误形状、鉴权、模型权限、多维限流、usage 和隐私日志都可验证。
35E47. 写一个 0 依赖 Production Deployment 审计 demo，输入 image artifact、model artifact、deployment config、old/new workers、canary users 和 canary metrics；实现 ImageArtifact、ModelArtifact、WorkerState、DeploymentConfig、CanaryMetrics、ToyProductionDeploymentAuditor、runtime compatibility matrix、model artifact manifest、startup readiness gate、warmup probe check、worker registration gate、drain completion gate、rolling update capacity gate、canary traffic split、canary regression gate、rollback readiness gate 和 production deployment gate；输出 runtime compatible、artifact ready、startup phases、probe report、registered workers、capacity before/after drain、drain events、canary routes、canary report、rollback ready、observability logs、metrics 和 gates；要求说明 LLM serving 生产部署不能只看容器进程是否启动，而要证明 runtime、模型 manifest、readiness、滚动容量、drain、灰度、回滚和 revision 级观测都闭环。
35E48. 写一个 0 依赖 Capacity SLO Fault Drill 审计 demo，输入 workloads、worker benchmark、cost config、SLO target、incoming burst 和 fault scenarios；实现 WorkloadProfile、WorkerBenchmark、CostConfig、SLOTarget、FaultScenario、ToyCapacitySLOFaultDrillAuditor、workload token profile、benchmark stable capacity、GPU count estimate、KV concurrency limit、cost attribution model、SLO error budget、admission overload policy、fault drill scenario、runbook coverage check 和 capacity SLO fault drill gate；输出 traffic profile、required workers by request/prefill/decode/KV、base/safe/N+1/release workers、planned GPUs、cost report、SLO report、admission decision、fault drill rows 和 gates；要求说明 LLM serving 容量规划不能只看 QPS，而要把 token 压力、KV 并发、稳定压测容量、利用率、成本、error budget、admission 和故障演练放在一起复算。
35E49. 写一个 0 依赖 Inference Engine Interview Readiness 审计 demo，输入 question rubrics、candidate answers 和 project evidence；实现 QuestionRubric、CandidateAnswer、ProjectEvidence、ToyInferenceEngineInterviewAuditor、interview question rubric、concept coverage score、request lifecycle answer、KV cache answer evidence、scheduler answer evidence、serving metrics coverage、ordered debug path、production governance answer、project evidence portfolio、tradeoff coverage 和 inference engine interview gate；输出 concept rows、metric gate、debug gate、tradeoff gate、project gate、weak questions、revision plan 和 gates；要求说明推理框架面试不能只背框架名，而要证明生命周期、KV/PagedAttention、调度、指标、debug、生产治理、项目证据和 trade-off 都能闭环表达。
36. 为 5 个多模态产品设计 multimodal product audit 表，字段至少包含 modalities、samples total、task success、quality pass、supported claims、OCR CER、ASR WER、generation adopted、safety pass、privacy pass、copyright pass、high-risk reviewed、token equivalent、P95 latency、unit cost、eval ready、feedback loop 和 business metric。
37. 写一个 0 依赖多模态产品审计 demo，输出 task success、input quality、evidence support、OCR quality、ASR quality、generation adoption、safety pass、privacy pass、copyright pass、multimodal score、multimodal gate、failed gates 和 needs rework。
38. 构造 3 个输入质量不足样本，分别覆盖模糊文档、噪声音频和长视频关键片段缺失，并说明产品应如何引导重传、裁剪、澄清或转人工。
39. 把一个图像问答或文档问答案例拆成 atomic claims，并逐条标注是否被图片区域、OCR 文本、表格、文档页或引用证据支持。
39A. 用纯 Python 写一个多模态项目事故审计 demo，输入 toy media cases、expected evidence、observed evidence、OCR / ASR 错误、critical fields、video events、grounding IoU、claim support、safety flags、confirmation 和 budget，输出 input fidelity、evidence recall、OCR / ASR accuracy、temporal evidence recall、evidence support、hallucination rate、root causes 和 failed gates。
39B. 构造 6 个多模态 bad case：长截图裁剪丢证据、OCR 读错金额、ASR 读错关键字段、视频关键事件未抽到、grounding 区域不支持 claim、不可信媒体内容安全边界失败，并为每个样本写出修复优先级。
40. 为一个生成式多模态产品设计安全和版权审计表，覆盖输入素材授权、参考图边界、输出使用范围、内容溯源、深伪风险、品牌安全和人工复核。
41. 为 4 个企业大模型产品设计 privacy governance audit 表，字段至少包含 PII detected、PII redacted、sensitive events、sensitive blocked、access checks、unauthorized hits、logs total、logs redacted、training candidates、training consent ok、retention items、retention ok、deletion requests、deletion SLA met、external transfers、external approved、audit events、audit complete、high-risk reviewed、DPA ready、incident response ready 和 business metric。
42. 写一个 0 依赖隐私治理审计 demo，输出 PII 脱敏覆盖率、敏感数据阻断率、越权访问率、日志脱敏覆盖率、训练授权覆盖率、保留合规率、删除 SLA 通过率、外部传输审批覆盖率、审计覆盖率、privacy score、privacy gate、failed gates 和 needs rework。
43. 构造 3 个隐私治理失败案例，分别覆盖 RAG 越权召回、排障日志原文泄露和外部模型传输未审批，并说明应触发的阻断、降级、通知、修复和复测流程。
43A. 用纯 Python 写一个安全合规事故审计 demo，输入 toy cases 的 expected action、actual action、PII、sensitive events、access checks、high-risk confirmation、log redaction、training consent、retention、deletion、external transfer、audit、DPA 和 incident response，输出 unsafe pass rate、over-refusal rate、PII redaction coverage、sensitive block rate、unauthorized access rate、tool confirmation coverage、log redaction coverage、training consent coverage、retention / deletion、external approval、audit coverage、root causes、failed gates 和 gate pass。
43B. 构造 6 个安全合规 bad case：正常合规问题被误拒、RAG 越权上下文放行、工具写操作缺少确认、日志保存敏感原文、反馈数据未经授权进入训练候选、外部传输缺审批，并为每个样本写出阻断、降级、通知、修复和复测动作。
44. 为 4 个大模型产品设计上线运营审计表，字段至少包含 eligible、rollout、tasks、success、adopted、feedback、feedback actioned、bad cases、bad triaged、regression pass、slice covered、P95 latency、unit cost、incidents、trace coverage、rollback ready 和 business metric。
45. 写一个 0 依赖上线运营审计 demo，输出 ranked、ops pass、live success、adoption、feedback action rate、bad case triage coverage、regression pass rate、trace coverage、needs rework 和 top failure reasons。
46. 构造 3 个上线后必须暂停或回滚的案例，分别覆盖高风险事故未处理、反馈没有进入回归集、trace 缺失导致无法复现，并说明如何通过 feature flag、灰度降级、人工兜底和事故复盘恢复。
47. 为 8 道技术产品面试题设计复盘表，字段至少包含 question、topics、formulas、demos、risks、tradeoffs、score、weak reason 和 revision plan。
48. 写一个 0 依赖技术产品面试复盘 demo，输出 topic coverage、formula coverage、demo evidence coverage、risk coverage、trade-off coverage、weak questions、interview ready 和 revision plan。
49. 为 5 个跨团队大模型项目设计 project collaboration audit 表，字段至少包含 objective clear、metric tree、baseline frozen、experiment record、version trace、RACI、change control、risk escalation、rollout、rollback、bad case regression、postmortem、action items、P95 latency 和 cost ratio。
50. 写一个 0 依赖项目协作事故审计 demo，输出 objective clarity rate、metric tree coverage、baseline coverage、experiment reproducibility rate、version trace coverage、RACI coverage、change control coverage、risk escalation coverage、rollout readiness、rollback readiness、bad case regression coverage、postmortem completeness、action item closure、root causes、failed gates 和 project collaboration gate。
51. 构造 4 个项目协作 bad case：目标模糊导致实验发散、离线指标上涨但护栏缺失、prompt 热修无版本和回归、RACI 不清导致事故无人负责，并为每个样本写出阻断、降级、补证据、复盘和防复发动作。
52. 为 5 个实验或事故复盘样本设计 retrospective audit 表，字段至少包含 hypothesis、baseline、version trace、slice analysis、bad case taxonomy、evidence、timeline、impact、mitigation、root cause、prevention、action items、regression verification、review、detect time、recover time 和 failed deployment。
53. 写一个 0 依赖实验复盘与事故报告审计 demo，输出 hypothesis coverage、baseline coverage、version trace coverage、slice analysis coverage、badcase taxonomy coverage、evidence coverage、timeline coverage、impact quantification rate、root cause rate、prevention coverage、action item closure、regression verification rate、change failure rate、failed gates 和 retrospective gate。
54. 构造 4 个复盘 bad case：实验没有可证伪假设、prompt 热修无 baseline、Agent 事故缺少时间线、修复没有回归验证，并为每个样本写出补证据、止损、根因、行动项、验收标准和防复发动作。
55. 为 6 道资深工程师面试回答设计 practitioner interview audit 表，字段至少包含 question、type、goal、baseline、metrics、experiment、bad cases、debug path、trade-offs、production、safety、cost、reflection、STAR fields、follow-up readiness、English clarity、red flags 和 revision plan。
56. 写一个 0 依赖资深工程师面试表达审计 demo，输出 project evidence score、STAR reflection score、trade-off depth、expert follow-up readiness、English clarity、red flag count、weak questions、revision plan 和 practitioner interview gate。
57. 选一个自己最熟悉的大模型项目，把“为什么不用更大模型”“上线后效果变差怎么办”“最大的失败实验是什么”“如果重做会怎么改”四个追问分别写成 baseline、metric、failure、trade-off 和 boundary 五段式回答。

## 阶段 6：数据工程与治理验收

1. 设计一个 web-scale 数据采集 source registry，字段至少包含 source id、license、ToS、robots、access mode、owner、risk level 和 delete policy。
2. 为 8 个逻辑 URL 设计响应事件账本，分别记录 2xx、404、429、503 和重试；计算尝试级 2xx 比率与最终 URL 2xx 比率，并解释分母差异。
3. 比较 WARC、WAT 和 WET 的证据能力；针对同一个双栏 PDF 或 HTML 页面设计从原始载荷到解析工件的定位字段。
4. 用 3 分钟回答“为什么 HTTP 200、robots 允许和许可授权是三个不同判断”。
5. 画出来源准入、scheduler、collector、raw snapshot、parser、抽样复核和下游质量处理的链路；为每条边标明对象 ID、版本或失败状态。
6. 为 12 条 toy 文本设计质量特征表，至少包含长度、unique token ratio、重复率、符号比例、乱码比例、boilerplate / promo 命中和风险标记。
7. 写一个 0 依赖 Python demo，给 8 条含人工标签和质量分的样本比较两个阈值，输出 TP / FP / FN / TN、precision、recall、文档留存率与加权误删/漏删代价；覆盖空审计集和没有正类的切片，缺少分母时输出 `N/A`。
8. 扩展上一题的人工审计集，按语言 / 领域分层计算误删率、漏删率和 token 保留率；说明过采样稀有切片时怎样按抽样权重估计总体结果，并区分文档留存率与 token 留存率。
9. 解释为什么代码、数学、低资源语言和专业文档不能共用普通网页文本的质量阈值。
10. 设计一次小模型 ablation：比较清洗前、规则清洗后、质量分类器过滤后三个数据版本的验证 loss、下游指标和风险命中率。
11. 给 10 条 toy 文档计算规范化文本 hash，找出 exact duplicate group，并说明为何需保留原始对象、规范化函数版本和足够长的摘要；分别计算文档留存率与 token 留存率，再按来源、解析完整性和用途选代表。
12. 对 `A={a,b,c,d}`、`B={b,c,d,e}` 手算 Jaccard；给出四个随机排列下的 MinHash 最小项并计算估计误差，再用 `b=4,r=2,s=0.6` 手算 LSH 候选概率；解释候选概率不等于候选为真的概率。
13. 写一个 0 依赖 Python demo，对 toy 文档计算 exact hash、Jaccard、MinHash 估计和 SimHash 汉明距离；将候选边与人工合并决定分开，统计 false merge / missed duplicate。额外加入一条相似度链 A–B、B–C、A–C 很低的样例，说明为什么不能未经检查直接用连通分量删除。
14. 构造 8 条 toy benchmark 评估题，其中 3 条有题目/答案/解析强重叠、2 条只有弱主题相似、3 条无检索命中。计算候选命中率 `5/8`、确认暴露率 `3/8`、弱信号率 `2/8`；若强重叠组答对 `2/3`、其他题答对 `3/5`，说明这些样本能支持什么、不能得出什么因果结论。
14A. 比较训练语料可访问时的精确/近似检索、黑盒 Data Contamination Quiz、受控注入对和时间后新题；针对 SFT、RL 与推理模型分别说明探针的覆盖假设、可能漏检和结论边界。
15. 设计一份去重后 data mixture 审计表，比较去重前后的语言、领域、来源、token 数和质量分分布。
16. 给 8 个 toy 数据池设计 data mixture 配置，区分池级预算权重、样本级选择概率、训练阶段顺序和日志中的实际曝光；字段至少包含 clean tokens、quality score、risk score、能力标签、planned tokens、effective epoch 和采样器实际 token 数，并写清分母。
17. 写一个 0 依赖 Python demo，比较 natural sampling、temperature sampling、质量加权和样本级选择的配比变化；检查空池、非有限权重、计划 token 四舍五入后总量不守恒及去重后自然比例变化。
18. 计算每个数据池的 effective epoch，并标出可能带来记忆或过拟合风险的上采样数据池。
19. 设计一次 mixture ablation：分别提高代码、数学、多语言和合成数据比例，并比较至少两种样本排序/调度；固定总 token、训练预算、tokenizer、数据快照和随机种子，另留未参与调参的验证切片，报告目标能力、非目标回归、重复暴露和数据风险。
20. 用 3 分钟回答“为什么 data mixture 是隐式目标函数设计，而不是数据拼接配置”。
21. 给 12 条 toy code / math / domain 样本设计专项审计表，区分可校准的类型质量分与不能被质量分抵消的授权、secret、PII 和 benchmark 隔离硬门禁；字段包含 test pass rate、answer verification、authority、recency、citation、token 数和拒绝原因。
22. 写一个 0 依赖 Python demo，分别计算代码、数学和领域数据的质量分、拒绝原因、分类型 token 保留率和最终 mixture；手算一个保留样本和一个被硬门禁拦截的样本，并覆盖无测试、空输入、重复 ID、NaN 分数和非法类型。
23. 设计一个代码数据清洗 checklist，覆盖仓库/文件来源、fork、vendor、自动生成文件、license、secret、解析、测试和 benchmark 污染；说明怎样把问题、实现、测试与运行环境作为同一验证记录关联。
24. 设计一个数学数据 verifier 审计流程，区分题目覆盖、答案正确性和过程/错误类型；说明如何检查难度、重复题和评测污染，并按错误类型报告 verifier 的 precision/recall，避免把未验证答案当作正向监督。
25. 设计一个专业领域数据治理方案，覆盖来源授权与权威性、版本时间、引用、PII、专家抽检和 RAG 元数据；设计记忆审计以区分有用知识回忆、无信息量模板复述和潜在敏感记录泄露，并说明何时选择继续预训练、SFT、RAG 或独立工具验证。
26. 设计一个 synthetic / distillation data 记录 schema，分别保存 `origin`、`generator`、`teacher`、student objective、seed 来源与授权、prompt/sampling 版本、质量分、验证证据、重复/污染/隐私/残余风险标记；说明为何“合成”和“蒸馏”不能压成互斥标签。
27. 写一个 0 依赖 Python 审计 demo，按授权、答案验证、去重、污染、安全、PII、自然数据锚点、多样性与合成比例输出 `kept`、拒绝原因、来源/任务/teacher token mix 和门禁；质量与比例门槛按未舍入数值判断，并覆盖空输入、全拒绝、重复 ID、布尔 token 数、NaN、非法 origin/标签和比例刚好等于或略超上限。
28. 比较 Self-Instruct、Evol-Instruct、Orca/解释轨迹蒸馏与 soft-label distillation 的输入、训练目标、教师信号、过滤和验证；为数学、代码、事实与安全样本分别选一种独立验证证据，说明 teacher 输出为何不是 gold label。
29. 设计一次合成比例消融：比较 10%、30%、50%、70% 的合成/蒸馏 token 比例，同时保留自然数据锚点；固定 student、tokenizer、总训练 token、数据快照和训练 recipe，用多个种子及未参与调参的 holdout 比较目标能力、通用回归、覆盖/重复、事实、安全和风格。
30. 对递归“替换自然数据”“与已有数据累积”“保留自然数据并加入目标生成数据”三种流程列出可证伪假设，手算一组覆盖/重复/风险指标并设计至少两个训练代数的评估；说明为何不能从单个论文比例推出通用阈值或断言合成数据必然导致 model collapse。
31. 设计 10 条 prompt/chosen/rejected 偏好样本，记录任务与 rubric 版本、chosen/rejected 理由、语言和标注群体；至少加入平局或“信息不足”案例，说明分歧是标签错误、任务欠说明、事实证据冲突还是合理的风格差异。
32. 写一个 0 依赖 Python demo，手算并计算 rubric 加权分与 preference margin，分开报告两位标注者的原始一致率、长度异常比例、目标动作保留率和风险切片覆盖；空分母须为 `N/A`，且不得把数据留存率称为模型误拒率或漏拒率。
33. 构造一份安全数据 schema，将 intent/answerability、后果严重度、domain、dialogue context 和 expected action 作为独立字段；分别覆盖 answer、clarify、refuse、safe alternative 和紧急支持动作，解释这些轴为何不能压成互斥的“风险类别”。
34. 设计一个红队回归数据版本报告，记录最小必要上下文、风险切片、失败模式、安全目标动作、模型与 policy 版本、来源权限、访问级别、训练/评估隔离、修复状态和复测结果；说明哪些完整载荷应留在受限评估集而不是普通训练集。
34A. 为 8 组偏好比较分别运行原顺序和交换顺序的 LLM judge，再用独立人工标注切片校准；计算 judge 的人类一致率、位置一致率、弃判率和各风险/语言切片差异，并设计发生分歧时升级人工复核的规则。
34B. 比较“不清洗、单 judge 过滤、多 judge 过滤、翻转可疑标签”四种偏好数据处理方式；固定样本、过滤比例、训练目标和 holdout，比较 win-tie-loss、事实性、安全动作错误与跨模型泛化，并说明为什么不能把某个基准中的最佳处理法当成通用规则。
35. 写七个防御性安全审计 demo：一个 red teaming demo，输入 toy case 的风险类别、严重度、baseline、natural score、elicited score、工具确认和回归结果，输出 taxonomy coverage、P0/P1 unresolved、dangerous capability uplift、autonomy score、regression pass rate 和 release gate；手算至少一个比例，并让没有 autonomy 样本、没有工具尝试和空回归集明确输出 `N/A`，且不能把缺测当作通过；一个 mechanistic interpretability demo，输入 toy clean / corrupted logit delta、patching 结果、ablation 结果、SAE feature 激活和标签，输出 patch recovery、ablation effect、reconstruction fidelity、avg active features、feature purity 和 interpretability gate；一个 steering demo，输入正负 toy 激活、alpha 网格、target gain、safe gain、side loss 和 over-refusal delta，输出 steering vector、projection shift、target uplift、safe gain、side-effect drop、over-refusal delta 和 steering gate；一个 model editing / unlearning demo，输入 toy edit cases、paraphrase checks、locality checks、forget set、retain set、改写 / 多轮泄露、合法上下文效用和 membership inference 风险，输出 edit success、generalization、locality、retain drop、forget leak、robust leak、contextual utility、membership risk drop、editing gate 和 unlearning gate；可选增加 representation residual audit，但不得把内部可解码直接当成外部泄露；一个 privacy / watermarking demo，输入 toy PII / secret / RAG / log / membership / watermark cases，输出 PII recall、secret recall、memorization rate、output leak rate、RAG unauthorized retrieval、raw log rate、membership advantage、watermark z-score、generated recall、false positive rate、robust recall、privacy gate 和 watermark gate；手算 DP-SGD 的逐样本裁剪、噪声和 batch mean，以及一个 TPR/FPR/advantage 和一个 watermark z-score；覆盖空分母、零权重、无效 token / green 数、误报、未定义门槛状态，缺测不得当成通过；一个 governance / model card demo，输入 toy model card sections、system card sections、policy categories、eval slices、risk issues、approval votes 和 release checks，输出 model card completion、system card completion、policy coverage、eval coverage、severity-weighted unresolved risk、high-risk mitigation、approval coverage、governance gate 和 release ready；一个 safety interview retrospective demo，输入 toy answer records、risk set、metric set、gate set 和 trade-off set，输出 answer coverage、risk coverage、metric coverage、gate coverage、trade-off coverage、unsafe detail count、readiness gates 和 revision plan。
35EA. 为记忆污染评测构造 10 个合成样本，分别记录持久化、后续召回、采纳、副作用、污染清除和良性记忆保留；逐步计算阶段条件率、端到端副作用率和选择性修复率，明确每项分母，并让空阶段输出 `N/A`。
35EB. 使用第八册第七章的六个合成向量，手算一个样本的平方重构误差、六样本 SSE、SST、重构保真度、平均激活 feature 数和 top-2 purity；再用 clean margin 2.0、corrupted margin -1.0、patched margin 1.4 计算 patch recovery。解释为什么这些数值不能合并为安全分数，并比较 SAEBench、MIB 和随机基线研究各自检验的问题。让空样本、少于 top-k 的样本和近零分母显式得到 N/A。
35EC. 使用第八册第八章的五个 steering toy case，手算目标提升、安全收益、正常能力损失、误拒增量和加权候选分数。将高风险请求与正常请求分开作为分母；分别删除一个切片并让对应指标显示 N/A，不允许用总体均值代替缺失数据。再设计逐 token / 多轮持久性和输入扰动 holdout，说明 Endogenous Steering Resistance、SteerEval 与 SteeringSafety 分别帮助检查什么、又不能证明什么。
35ED. 手算第八册第九章四条编辑记录的加权目标成功率、先按编辑求均值再按权重汇总的改写率与 locality、保留能力差值及冲突数；再用三条 forget case 算 exact 与 robust leakage。说明为什么 MIA 下降和 retain 保持不能抵消泄露门槛失败，并为合法上下文效用设计独立切片。代码边界需覆盖空编辑、空 forget、空嵌套样本、零总权重和缺字段，缺测不得当成通过。
35EE. 按第八册第十一章设计一份有版本绑定的 Model Card 与 System Card：为每条用途、评测和缓解声明指定证据、适用条件、责任人、复核日期和证据状态；模拟仅更新 tokenizer、RAG 索引或工具权限时哪些结论会变旧。再用 demo 的合成输入手算 Model Card 8/9、System Card 8/10、未解决风险权重 12/16 和高风险缓解 8/13，说明为何缺测为 N/A、覆盖率不能抵销硬性隐私约束，并区分机构 Responsible Scaling policy、NIST 自愿框架、GPAI Code of Practice 与适用法规义务。
35EF. 以企业研究助手为例，把一个失败轨迹分别映射到 misuse、malfunctions、systemic risks 和 OWASP LLM 应用风险目录；区分可复用分类与具体测试样本。用 20 条允许任务中的 18 次成功、2 次误拒计算任务成功率和 over-refusal；再为 10 条越权检索样本设计分母、服务端阻断记录与人工复核，说明为什么这些定向红队样本不能直接估计总体风险。最后按定义、机制、责任层、指标、控制和适用边界组织一段面试回答。
35EG. 按第八册第十三章的六条合成轨迹，分别计算风险任务的实际副作用率、工具未授权尝试率、未授权调用的执行比例、Judge precision/recall 和正常任务成功率；手算 1,000 个候选、960 个有效运行、50 个自动警报、20 个确认发现的比例，以及 200 次工具调用中 48 次未授权、40 次阻断、8 次执行时的三个不同分母。比较 GOAT 的自适应多轮生成与 REDAgentBench 的状态回执评估；设计空分母、轨迹/状态判定不一致、被阻断动作和误拒负对照，说明何时停止、复现、修复与回归。
35EH. 按第八册第十五章的合成生产发布输入手算风险分数 $R=0.59$，逐项列出五个加权乘积，并解释它不是事故概率；再用两桶样本复算 ECE $=0.1333$，指出它需要已定义事件和完成标注。为“主体有效但没有 grant”“已撤销主体”“资源 scope 错误”“审批后 revision 改变”“高风险但缺少双人审批”设计正反例。分别计算 HighRiskAllowRate 与 UnauthorizedSideEffectRate，记录分子、分母以及无尝试时的 N/A；比较 AuthZEN、RFC 8693 与 DPoP 分别解决的接口、委托和令牌重放问题，并说明它们都不能替代本地授权与 PEP 状态验证。
35EI. 按第八册第十六章构造九条 fallback 轨迹，覆盖未产生副作用的超时、未知付款、策略拒绝、兼容工具缺失、撤权后已提交、状态查询被拒和重试耗尽；为每条给出状态、允许范围、下一步与用户说明。手算三层各允许一次重试时的上界 $(1+1)^3=8$，以及 `0+0.4+0.2+0.6=1.2` 秒如何适配 1.5 秒预算。再用 12 个受限样本中 9 个保持约束、20 个有副作用可能的任务中 1 次重复提交计算安全保持率和重复副作用率；设计空分母、已提交后撤权、读状态权限仍有效和权限也撤销四种分支，并区分 RFC 9110 方法幂等与业务幂等键。

36. 用 3 分钟回答“为什么高质量安全模型不是拒答率最高的模型”。
37. 给 12 条 toy 多模态样本设计审计表，覆盖 image-text、OCR、audio-transcript、video-caption 和 multimodal instruction 五类样本。
38. 为每条多模态样本标注 media quality、alignment score、annotation quality、license、privacy、contamination、time error、grounded answer 和 tokens。
39. 写一个 0 依赖 Python demo，按媒体质量、图文 / 音文 / 视频对齐、隐私、来源权限、评测污染、时间错位和 grounded answer 输出保留样本与全部隔离原因；让软质量分使用未舍入值做门槛判断，并使多个并存问题都能被报告。
40. 计算多模态数据的总 token 保留率、分模态 token 保留率、保留池 mixture、按样本或时长统计的对齐摘要，并说明各自分母；空分母输出 `N/A`，有分母但零保留输出 `0.0`。
41. 设计一个 OCR / 文档图像数据 schema，字段包含 page id、recognized text、bounding box、reading order、confidence、language、table region、PII flag 和 source license。
42. 设计一个音频 / 视频时间对齐抽检流程，说明如何用 WER / CER、subtitle shift、VAD segment、clip boundary 和人工抽样判断样本是否可进入训练。
43. 用 3 分钟回答“为什么多模态数据工程的核心不是增加模态数量，而是证明模态之间真的对齐”。
43A. 为一条 8 秒视频设计 frame/shot/event 三级标注，逐项计算 1 FPS 采样帧数、字幕—事件时间差和关键事件召回；比较固定窗口、镜头切分和事件切分的误差，并说明长视频数据集的标注结果为何不能直接当成目标语料保证。
43B. 以多语言长音频为例，计算一个 WER 和平均边界时间误差；设计人工参考样本、语言/口音切片和权限检查，说明论文报告的语料规模或微调收益为何不能替代项目级授权与验证。
44. 给 8 个 toy 数据源设计 valuation 表，字段包含 tokens、quality、coverage、risk、cost、license、contamination、目标能力收益向量和数据版本；先把许可未获准、隐私风险和评估污染定义为独立准入/隔离门槛，再对通过门槛的样本计算质量价值，说明为什么加权高分不能抵消硬性风险。
45. 写一个 0 依赖 Python demo，计算每个数据源的加权目标收益、风险 / 成本修正价值、梯度相似 attribution proxy 和 token budget 下的选择结果。
46. 对 3 个 toy 数据源手算一次小规模 Shapley value，说明平均边际贡献为什么会受到数据交互影响。
47. 构造一个“看起来提升 benchmark 但其实是污染”的数据源，并说明为什么高 attribution / 高相似度不能覆盖污染门禁。
48. 设计一次源级 ablation 实验：分别删除、降权、上采样某个数据源，记录通用、代码、数学、安全、误拒、成本和人工质量指标。
48. 构造 5 条负价值数据样本，覆盖错误标注、过时专业知识、低质合成、隐私风险和导致过度拒答的数据，并写出重洗 / 降权 / 删除策略。
49. 设计一个主动学习标注优先级表，字段包含 uncertainty、model disagreement、用户频率、风险等级、目标能力覆盖、多样性和标注成本。
50. 设计一个 dataset manifest，字段包含 dataset version、parent version、shard id、checksum、sample count、token count、language mix、license mix、risk summary 和 pipeline version。
51. 给 8 条 toy 样本设计 lineage 表，记录 source id、raw hash、processing steps、dedup cluster、dataset version、training run id 和 deletion status。
52. 写一个 0 依赖 Python demo，生成 shard checksum、manifest、lineage coverage、license 分布、可训练样本列表和治理门禁。
53. 构造 3 个删除请求，说明如何从 source id / hash 定位 raw、clean、shard、索引、旧版本和训练日志中的受影响数据。
54. 设计一个训练数据权限矩阵，覆盖 public、internal、restricted、quarantine、benchmark 和 user data，并给出访问日志审计规则。
55. 写一份 datasheet / dataset card 大纲，至少包含 purpose、composition、collection、processing、license、PII、bias、risk、maintenance 和 deletion policy。
56. 设计一个 model card 的数据输入清单，说明训练数据概述、评估数据版本、适用范围、已知偏差、隐私风险和删除策略分别依赖哪些数据治理证据。
57. 用 3 分钟回答“为什么 dataset versioning 不是文件名加日期，而是数据治理控制面”。
58. 选 6 道数据工程高频题，为每题写出目标、来源、处理、配比、评估、治理六段式回答提纲。
59. 给一次数据工程 mock interview 设计自评表，字段包含 answer coverage、metric coverage、evidence、risk coverage、trade-off depth、red flags 和 time budget。
60. 写一个 0 依赖 Python demo，输入 4 道 mock interview 回答记录，输出每题得分、平均分、缺失框架、缺失风险和 revision plan。
61. 用 3 分钟回答“为什么数据工程面试不是背数据集比例，而是证明你能建立可审计的数据系统”。
62. 写一个 0 依赖 Python 数据事故审计 demo，输入 12 条 toy 样本，输出 exact duplicate group、near duplicate pair、benchmark contamination、secret / license 风险、清洗后 actual mixture、mixture shift、lineage coverage、failed gates 和 gate pass 结论。

## 阶段 2026-08：Frontier Model、Agent Runtime 与 Serving

1. 写出 `S_t=alpha_t*S_{t-1}+u_t*v_t^T` 的递归状态教学公式，并说明它和显式 KV cache 的成本差异。
2. 比较 KDA、Gated DeltaNet、MLA、GQA、local/global attention 和 NoPE 的历史表示方式、优缺点与失败模式。
3. 给 6 个模型卡字段标注证据等级：官方论文、官方 model card、一方产品页、厂商自报、待核验。
4. 设计 `R=F(M,H,E,B,D)` 的 Agent benchmark 记录表，至少包含模型 revision、harness、环境、工具、effort、memory 和 trace。
5. 为 Qwen-AgentWorld 设计一个 task environment schema，字段包含 reset、observation、action、artifact、tool permission、checkpoint 和 success verifier。
6. 写出 Agent Swarm 的成本函数，解释为什么并发降低 wall-clock 不等于降低 token、通信和验证成本。
7. 比较 standalone draft、EAGLE、MTP、NEXTN 和 DSpark speculative decoding，并设计 acceptance length 与 fallback 监控表。
8. 解释 FP4/MXFP4/NVFP4、FP8 KV cache、native INT4 的作用层次，并写出量化后的显存估算公式。
9. 设计一个 Responses API 与 OpenAI-compatible API 的 capability matrix，覆盖 reasoning item、tool call id、streaming、template、encoding 和 replay。
10. 给 GPT-5.5、GPT-5.6、Claude、Gemini、DeepSeek-V4、Qwen3.5/3.6、Kimi K2/K3 设计一次 effort/level 公平比较实验。
11. 设计多模态长上下文评测，分别测 ingest、retrieve、reason、action、robust 五个门禁，并包含图片/音频/视频 token budget。
12. 设计 Shieldstral 风格的自然语言 policy 输入和连续安全分 toy demo，加入 false refusal、tool violation、fallback 和 human review 指标；另用同一图像在两种相反政策下构造配对样本，比较 F1 与 PSS。

13. 设计一个 Adaptive Thinking 控制器实验，比较固定 level、规则升档和 verifier 驱动升档，输出 reasoning/tool/verify/recovery budget、p95 成本和漏升级率。
14. 写一条包含工具超时、客户端重连和权限变化的 Interleaved Thinking trace，标记 `tool_call_id`、幂等检查、rollback 和可持久化字段。
15. 为一个 long-running coding task 设计 Persistent Workspace manifest，覆盖 commit、diff、artifact hash、环境、权限、待执行动作和外部副作用回执。
16. 设计 fallback routing capability matrix，比较主模型、只读模型、人工审核和拒答路径的工具权限、状态兼容、成本和风险。
17. 为 p-RoPE 设计 local/global layer schedule 审计，测试顺序交换、跨窗口引用、position id、packed batch 和 checkpoint resume。
18. 做 Gated Attention ablation，比较 `g=0`、`g=1` 和真实 gate 在局部检索、远距离引用、TPOT、显存和 gate 分布上的差异。
19. 为 CSA/HCA 设计压缩注意力评测，比较原始 KV、只局部、只压缩和混合路径的检索召回、压缩误差、TTFT、TPOT 和单位成功成本。
20. 用统一的模型 revision、校准样本和 workload 比较 GPTQ、AWQ、SmoothQuant、FP8 KV、KIVI 或 TurboQuant 中至少四条路线：先说明各自量化对象与误差机制，再记录 effective bits（含 scale）、质量切片、kernel 路径、峰值显存、TTFT、TPOT、吞吐和并发；明确标注不同论文数字不可直接排名，并设计一个能区分“算法误差”与“kernel fallback”的消融。
21. 对权重 [-0.9, -0.2, 0.3, 1.1] 使用有符号 INT4 对称码本 [-7, 7] 手算 scale、整数码和反量化结果；再加入一个 outlier，说明它如何改变 group scale、普通值的分辨率和 clipping 的代价。
22. 给定一个 2×2 的假设逆 Hessian和一次权重舍入误差，按 GPTQ 更新式计算剩余坐标的补偿量。解释该 Hessian为何来自校准激活的层重建目标而不是整个模型任务 loss，并对照 RTN、GPTQ 在误差处理和转换成本上的差别。
23. 从 Y=XW 推导 AWQ 的等价缩放 X D_s^-1 与 D_s W；用一个输入通道说明放大权重、缩小激活如何影响量化步长。解释 alpha 搜索和 clipping 的冲突、为何 AWQ 主方法不需要保留 1% FP16 权重，以及校准分布变化后仍需验证什么。
24. 对照 RTN、GPTQ、AWQ、LLM-QAT 和 MARLIN：逐个写出问题、机制、实验指标/设置、主要代价和适用边界。明确区分量化算法、训练方法、artifact 和执行 kernel；为“质量退化”“加载成功但数值错误”“质量通过但速度未提升”分别设计下一轮定位实验。
25. 100 个原始请求合计花费 120 个成本单位、最终 90 个成功，计算单位成功成本；设计 trace 字段分别记录首次推理、验证、重试、fallback 和人工审核成本，说明怎样避免分子重复计算，以及怎样避免把每次 attempt 当成独立请求。

## 2026-09 Frontier Architecture Updates

1. 手算三层 Full AttnRes 的 softmax 权重，并实现 Block AttnRes 的 partial sum。
2. 用零依赖 Python 实现 KDA 递推，比较逐通道 `alpha` 与标量衰减的状态轨迹。
3. 写一个二维 CSA/HCA toy cache，分别统计压缩误差、top-k 召回和滑动窗口补偿。
4. 实现 Sinkhorn 迭代，记录双随机误差随迭代轮数的变化，并比较负值输入与正值 logits 的差异。
5. 设计同预算架构消融，比较标准 residual、AttnRes、KDA+MLA 和 mHC 的 loss、梯度、显存、通信与成功检索率。

## 2026-09 Mistral Small 4 与 Step 3.5 Flash

1. 为 Mistral Small 4 的 `none/high` 模式写一份固定任务、工具、硬件和输出上限的对照实验表，报告质量、TTFT、TPOT、p95 和单位成功成本。
2. 估算 119B BF16 与 NVFP4 权重的理论存储，并把专家分片、scale、KV cache、路由 buffer 和 workspace 单独列账。
3. 写一个 toy EAGLE 推测解码模拟器，改变平均接受长度和草稿成本，输出目标模型调用次数、有效 token 和回退比例。
4. 实现 Step 3.5 的 3:1 SWA/full causal pair 统计，比较窗口大小和 full 层周期对长程访问预算的影响。
5. 为 MTP-3 加入 top-8 MoE 路由容量因子，记录专家负载方差、overflow、all-to-all token 数和验证阶段峰值。
6. 扩展 BrowseComp toy harness：在工具轨迹 token 使用量超过 80% 时分别实现 `Summary`、`Discard-75%`、`Discard-all`，并加入 N 条轨迹选最少 steps 的并行基线；区分模型失败、工具失败、上下文重启、超时和最终 artifact 未完成，比较 Pass@1、steps、工具成本、证据可追溯率与 artifact 完成率。把 DeepSeek V3.2 报告的 `51.4/67.6*` 标为 Search Agent/BrowseComp harness 参考值，不作为 toy 的预期分数或裸模型基线。

## 2026-09 Grok 4.6 接口与 Harness 审计

1. 为 500K prompt 设计 token/KV/cache/workspace 并发账本，明确 context、输入上限和输出上限不能直接相加。
2. 设计 `low/medium/high/xhigh` 的同任务对照实验，固定工具、超时、输出上限和 harness，输出成功率、reasoning token、TTFT、TPOT、p95 成本和失败类型。
3. 为 function calling、Web Search 和 X Search 写权限矩阵，记录模型输出、宿主授权、网络范围、超时、回执、重试、幂等和回滚。

## 2026-09 Claude Opus 5 长上下文与 adaptive effort 验收

1. 为 1M context 设计 token/KV/cache/workspace 并发账本，解释为什么不能由上下文上限直接推出并发数。
2. 设计 `adaptive` 与固定 effort 的对照实验，固定模型快照、平台、任务、工具和输出上限，输出成功率、推理 token、TTFT、TPOT、p95 和单位成功成本。
3. 画出 Claude API、Bedrock、Vertex AI、Foundry 的模型接口、宿主工具、沙箱、权限和审计边界，列出跨平台不可比因素。
4. 给一个长周期 coding task 设计 runtime manifest，至少记录模型 ID、平台、effort、输入/输出 token、缓存、工具 call、压缩、错误码、artifact hash 和回滚状态。
5. 阅读 Anthropic 官方模型目录，分别列出“页面明确写出”“可由接口实验验证”“当前不能确认”的字段，禁止从 `adaptive` 或模型分类反推参数规模和训练方法。

6. 构造 Opus 5 的 capability matrix，覆盖 `thinking.display: "omitted"`、thinking block/signature 回放、`thinking disabled` 与 `xhigh/max`、中途工具/effort 变更和 refusal/fallback；输出 capability error、状态失效、降级路径和是否允许重试。
7. 为一个不支持 web fetch 的长任务 Agent 设计宿主工具账本，记录搜索/抓取权限、SSRF 防护、超时、内容 provenance、缓存、重试、工具回执和最终 verifier，区分“模型未联网”和“宿主抓取失败”。
8. 实现 toy fallback ledger：同一任务在 refusal、server-side fallback、缓存命中和子 Agent 自验证下分别记录实际模型、effort、thinking token、工具轮次、成本、延迟和最终 artifact，禁止将 fallback 成功归因给 Opus 5。

## 2026-09 Claude Fable 5.1 长任务与状态协议验收

1. 为 Fable 5.1 的 1M context 和 128K 输出设计 token/KV/cache/workspace 账本，区分接口上限、有效能力和并发资源。
2. 设计 Opus 5 与 Fable 5.1 的固定任务对照，分别记录 adaptive thinking、effort、推理 token、TTFT、TPOT、工具轮次、恢复成功率和单位成功成本。
3. 实现 toy `preserved thinking` manifest，覆盖模型产生者、目标模型、消息版本、thinking block hash、权限、压缩、编辑历史和恢复结果；加入不兼容与篡改失败样例。
4. 设计 per-message effort、turn-scoped system message 和 `display: "updates"` 的 schema 迁移与回滚检查，区分 beta 协议错误和模型能力失败。
5. 将 Fable 5.1 页面自述的长任务、研究和文档工作优势拆成可复现任务切片，禁止把产品定位直接写成 benchmark 结论。
6. 实现一个 Fable 5.1 capability matrix：分别注入 forced tool、旧模型读取 Fable 5.1 thinking block、编辑历史 turn、per-message effort 和 turn-scoped system message，输出 capability error、状态失效、降级路径与是否允许重试。
7. 为 `display: "updates"` 设计进度事件与真实 tool result 的双账本，注入进度已发送但工具超时、工具已执行但 verifier 拒绝、重试后重复副作用等案例，检查 tool-call ID、幂等键和最终 artifact 是否一致。
8. 为 content provenance 设计来源链审计：记录来源、生成片段、引用覆盖、工具回执和独立 verifier，构造“有 provenance 但来源不支持结论”和“工具成功但 artifact 错误”两类失败，说明 provenance 不是事实正确性证明。
9. 将 Fable 5.1 与 Mythos 5.1 的同底模/不同 safeguards 建成 source-aware manifest，固定 revision、effort、平台、工具、拒答策略和 verifier，对比拒答率、工具可用性、任务成功率与单位成本，禁止把策略差异写成架构差异。
10. 为 DataCurve 精确行缺失写一个评测门禁：当只有 `claude-fable-5` 而没有 `mini_swe_agent_claude_fable_5_1_*` 时拒绝导入相邻版本 Pass@1、成本和 Agent steps，并输出清晰的 `not_applicable` 原因。
11. 根据 System Card 建立 Fable/Mythos 条件矩阵：分别记录最终 snapshot、helpful-only、关闭 safeguards、cyber/biology fallback、effort、工具权限和实际执行模型；检查同一任务在不同配置下是否仍可比较。
12. 运行 [`claude_fable51_state_protocol_audit.py`](research/model-update-2026-09/code/claude_fable51_state_protocol_audit.py)，解释为什么 progress update 不是 tool result，为什么 prefix mismatch 必须显式拒绝或 drop 并记录 `input_transformations`，以及为什么 provenance 仍需独立 artifact verifier；把脚本的 `local_protocol_toy` 证据等级与真实 API/模型质量分开。
12. 复现一个安全评测 manifest，输入 IPI、Shade coding、browser-use 和 sandbox 事件，输出 attack success、fallback 覆盖率、`actual_model`、`fallback_reason`、safeguard state 和 verifier；禁止把不同 harness 的数字合成一条安全率。
13. 将 Terminal-Bench、ProgramBench、OSWorld partial/strict 和 AA Intelligence Index 放入三本独立账本，构造缺失 snapshot、工具或成功定义的样本，要求系统返回 `not_comparable` 而不是自动排序。
14. 设计 RSP 阈值练习：解释 CB-1/CB-2 与 autonomy threat model 1/2 的定义、证据和不确定性；分别写出“未达阈值”“没有能力”“生产 safeguards 已阻止”三句话为什么不能互换。
15. 为 Claude Fable 5.1 与 Opus 5.5 建立有方向的 thinking-block compatibility matrix：验证 Claude API 与其他托管 surface 的差别、model_binding_mismatch 与 prefix_binding_mismatch 的不同处理、旧/新账户的 prefix 检查门槛，以及 fallback 是否真正可用。不得把“能读取”直接当成默认 fallback 或跨平台保证。

## 2026-09 Claude Sonnet 5 接口与平台对照验收

1. 为 Sonnet 5 的 context、普通输出和 batch 输出画出请求/批处理 token 账本，并标出缓存、KV、权重和 workspace 资源。
2. 设计 Sonnet 5、Opus 5、Fable 5.1 的同任务 `Adaptive`/effort 对照，输出质量、推理 token、TTFT、TPOT、p95 和单位成功成本。
3. 列出 Claude API、Bedrock、Google Cloud、Foundry 和 AWS 平台的模型 ID、协议、限流、工具和审计差异，标出不可直接合并的指标。
4. 为 `claude-sonnet-5` 建立发布门禁：模型 revision、价格/延迟页面快照、上下文行为、工具 schema、回滚和独立复现证据必须分栏记录。
5. 实现一个 compaction trace checker，分别验证 on-demand `compact-2026-09-04` 和 threshold `compact-2026-01-12` 的 beta/header/参数组合、block 位置与历史裁剪；拒绝混用两种接口、重复或被修改的 signed block，以及未闭合的 tool call。
6. 对 background keep-tail 构造“压缩请求发出 → 新轮次 append → 摘要返回 → 前缀替换”的并发轨迹，检查只替换已提交前缀、保留 tail、保留 context headroom、无并行 compaction，并把工具副作用与应用状态分别验收。
7. 比较 compaction 前后的 prompt-cache breakpoint：在 system prompt 尾部与 compaction block 分别设置 `cache_control`，估算 system cache 保留、summary cache 写入及旧前缀失效；不得把 cache miss 当成模型质量变化。
8. 为 keep-thinking 测试建立证据矩阵：分别验证原样回放/错误排序/删改 block、模型是否能读取 block、prefix check 是否启用；明确 Sonnet 5 不运行 Fable 5.1 起的 prefix mismatch check，测试输出只能记为文档合同或 local harness 结果，不能伪装成线上 API probe。

## 2026-09 DeepSWE v1.1 Harness 复现验收

1. 从 `research/model-update-2026-09/deepswe-snapshot-notes.md` 抄录 21 个配置，分别标注模型 ID、effort、Pass@1、区间、成本、输出 token 和 Agent steps，并解释为什么它们不是 21 个独立基础模型。
2. 为一个长周期软件工程任务建立 runtime manifest，记录任务/仓库版本、模型 revision、`mini-swe-agent`、工具 schema、verifier、超时、重试、上下文压缩、trace 和 artifact hash。
3. 设计固定模型换 harness、固定 harness 换模型、固定模型与 harness 换 effort 的三组实验，分别报告成功率、工具错误、verifier 拒绝、恢复率、p95 延迟、成本和失败类型。
4. 用 bootstrap 或 Wilson 区间重算 toy Pass@1，比较区间重叠、绝对百分点差、相对提升和单位成功成本；禁止只按榜单排序下结论。
5. 构造 no-op、unsolved-state、测试绕过、工具超时和最终 artifact 缺失的 verifier 反例，统计误通过率与误拒率，并说明 benchmark 设计声明不能替代独立污染审计。

## 2026-09 Claude Haiku 4.5 接口与低成本评测验收

1. 为 200K context、64K output 的请求建立 token 账本，说明输入、输出、缓存和思考 token 的边界。
2. 固定同一任务集、平台、并发和输出预算，对比 Haiku 4.5 的目录 `fastest` 字段与实测 TTFT、TPOT、p95 和单位成功成本。
3. 设计 Haiku 4.5、Sonnet 5、Opus 5 的质量/延迟/成本 Pareto 表，明确哪些字段来自官方目录、哪些来自独立实验。

## 2026-09 DeepSeek-R1-0528 发布与兼容性验收

1. 根据官方发布页设计 JSON output 和 function calling 的协议回归，覆盖合法 schema、拒绝路径、流式事件和错误码。
2. 为开源权重入口建立 revision、许可证、参数、硬件和 tokenizer 的核验清单；在资料缺失时输出 `待核验` 而不是推断。
3. 复现 benchmark 图片前，补齐任务集、采样、硬件、输出预算和统计区间字段，说明为什么发布页摘要不能替代独立复现。

## 2026-09 GPT-6 Astra 与运行时预算验收

1. 给定 context window 为 1,050,000、max output 为 128,000，说明为什么官方没有单列的 922,000 maximum input；将 922,000 标为预留满额输出后的算术余量，并画出输入、reasoning、工具状态和输出预算。
2. 用零依赖 Python 实现 GPT-6 Astra 的 token 费用估算，分别计量普通输入、缓存读取、30 分钟/12 小时缓存写入和输出；覆盖恰好 272K 与超过 272K、Standard/Batch/Flex/Fast/Ultrafast，并在 token 费用之外单列工具、重试和验证器成本。
3. 设计 `standard/pro` 与 `low/high/max` 的组合评测，固定 prompt、工具、超时和输出上限，分别记录 mode、effort、reasoning token、TTFT、TPOT、成功率、p95 成本和单位成功成本。
4. 为 hosted shell、MCP、apply patch 和 computer use 画出模型、宿主执行器、沙箱、策略、审批和审计日志的责任边界。
5. 给一个长周期 coding task 设计 runtime manifest，记录 model ID、mode/effort、输入/输出 token、cache read/write TTL、每类工具的调用/并发/timeout、全局 cost cap、重试/幂等、权限决策、compaction/steering lineage、错误码、artifact hash 和回滚状态。
6. 阅读 GPT-6 Astra 官方模型页，分别列出“页面明确写出”“可由接口实验验证”“当前不能确认”的字段，禁止把工具名称当作架构证据。
7. 根据 OpenAI 官方 Reasoning 与 Conversation state 文档，构造一组完整 output item、仅最终文本和 `previous_response_id` 的 replay 对照，检查 reasoning/phase、工具调用、计费和恢复结果。
8. 根据 Tool search 与 Compaction 文档，设计 deferred schema、namespace 版本、权限变更和 canonical context 的回归用例，覆盖工具移除、手工 prune、重复执行和缓存失配。
9. 根据 Async tool calling 文档实现一个零依赖 toy job registry，覆盖原始 `call_id`、唯一 task handle、重复结果、超时、失败重试和最新 response lineage；输出“工具已完成”和“模型已消费结果”两个独立状态。
10. 根据 Mid-turn steering 文档画出 WebSocket 事件状态机，覆盖 `response.steer.accepted`、`incomplete_details.reason=steered`、自动 continuation、已发送输出和已启动副作用；为不可回滚动作设计补偿门禁。
11. 将一份过长且互相冲突的 skill/`AGENTS.md` 说明重构为最小路由器与渐进披露目录，比较无关上下文 token、工具选择、停止原因、压缩次数和任务成功率；把误报/漏报的安全监控单独统计。
12. 扩展 [`gpt6_agent_protocol_demo.py`](research/model-update-2026-09/code/gpt6_agent_protocol_demo.py) 的负向测试：重复 task handle/call ID、错误 call ID、pending job 消费、重复 steering、旧 WebSocket steering 在新连接上的误重放，以及 `misalignment_policy_violation` 后的自动 retry；输出副作用保留和人工补偿所需的审计字段。
13. 对比 2026-09-21 Astra 专属 guide 与 2026-09-23 GPT-6 family guide，制作 capability matrix：`none` effort、Responses/Chat Completions function calling、EU data residency、unsupported parameters 和 cache-preserving `configuration_update`；说明为什么文档路由变化不能直接当作模型 revision。

## 2026-09 Kimi K3 发布证据与 Harness 验收

1. 阅读 Kimi K3 发布文章、技术报告和固定 config，把总参数、视觉、长上下文、KDA、AttnRes、Stable LatentMoE、量化、评测和权重状态分别标成“发布披露 / 论文机制 / 已核验配置 / 待核验实现”。
2. 为跨模型代码任务写一个 state manifest，覆盖目标、事实、假设、diff、工具 observation、权限、版本、预算、artifact hash 和未完成副作用。
3. 设计固定模型换 harness、固定 harness 换模型、固定模型与 harness 换 effort 的三组实验，并分别报告成功率、工具轮数、恢复成功率、延迟、成本和失败类型。
4. 比较 KDA、全注意力和混合路径，固定模型规模、训练 token、硬件和任务集，测精确检索、TTFT、TPOT、cache bytes 和长任务恢复。
5. 给出一组带 fallback 的基线/候选分数，计算绝对提升、相对提升和单位成功成本，指出哪些结果不可比。
6. 根据固定 HF `config.json` 写一份 K3 identity/config manifest，至少包含 revision、93 layers、69 KDA/24 Gated MLA、`q_lora_rank`、`kv_lora_rank`、experts、context 和“完整权重未下载”状态；禁止把 metadata 写成本地加载成功。
7. 根据 FlashKDA README 设计 H20/GB200 benchmark 复现实验，固定 `T/H/D`、warmup、iters、repeats、baseline、CUDA/PyTorch/driver 和 backend，报告 latency、speedup、失败环境与“不能外推端到端吞吐”的理由。
8. 设计 K3 hybrid serving recovery 测试：分别注入 MLA cache 丢失、KDA recurrent state 丢失、prefix-match unit 不一致、KV dtype 变化、模型 revision 变化和 tool-call parser 漂移，要求系统在 schema/权限/executor/verifier 层给出可审计失败，而不是静默继续。

## 2026-09 GLM-5.3 长任务与验证器验收

1. 为“修复吞吐回归且保持输出顺序”写一份任务契约，列出初始状态、允许动作、成功条件和终止失败。
2. 运行第十六册第 20 章的验证器 demo，解释 initial、completed 和 shortcut 三个 artifact 分别触发 no-op、成功和奖励捷径门禁的原因。
3. 给一个代码 Agent 环境设计 oracle、no-op、unsolved-state、隐藏测试和权限隔离五项质量检查。
4. 设计完整 trace 与压缩 trace 的对照实验，测关键状态召回、重复工具调用、恢复成功率、任务成功率和 token 成本；不要把 SAO 当作已知算法。
5. 写一份 GLM-5.3 迁移清单，分别覆盖 `thinking.type`、`reasoning_effort`、响应/流式协议、工具生命周期、错误处理、版本和回滚。

## 2026-09 GLM-5.3 官方博客与评测脚注验收

1. 为 CyberGym、ExploitBench、ExploitGym 画出发现、验证、利用推理和时间归一化的能力阶梯，说明为什么三个分数不能相加或直接排序。
2. 写一份 GLM-5.3 benchmark manifest，至少包括模型 revision、effort、harness、temperature、top-p、context、max output、turn、timeout、容器、Tool Search、域名白名单、verifier 和统计聚合。
3. 用 toy 任务模拟 Z.ai Code Bench 的 completion 与 checklist 两个指标，再加入 output-token cost，比较“完成率提升”和“单位成功成本”的关系。
4. 为一个漏洞发现任务分别实现发现/触发验证、深度利用推理和时间预算三种 verifier，记录假阳性、假阴性、超时、权限拒绝和人工复核。
5. 解释 269 个项目、2,436 个漏洞和 1,097 个中高危发现为什么不能写成模型单独的独立 benchmark；设计一张合作统计与模型评测的证据分层表。

## 2026-09 GLM-5.3 关联框架：Straw checkpoint 与 archive 恢复

1. 阅读固定 slime `checkpoint.py`，画出 actor/critic、rollout manager、queue/builder snapshot 到 `committed_<step>.json` 的提交顺序；注入缺 optimizer、缺 queue state、文件 size 变化和 metadata hash 变化，说明每种情况应允许 resume、拒绝 resume 还是进入 empty queue。
2. 从 step 1 回滚并创建 branch，断言旧 branch 的模型、queue snapshot 和 archive byte-for-byte 不变；检查新 branch 的 queue/task authority 不复用旧 receipt，同时仍能读取共享 immutable payload。
3. 为 `.straw.json` archive 实现 sample key/task key 查询，分别测试 `close()`、`release()`、多个 owner、活跃 reader 与 online GC；证明删除 index 不能代替释放 storage ownership。
4. 对照 `.straw.json` 与导出的 `.pt`：移除原 Straw pool 后，前者应明确失败，后者仍能读取；记录 tensor materialization 的内存峰值和导出成本。
5. 构造 fully async rollout 的 completed/partial/fresh 队列，改变 serving weight version，报告 staleness 分布、排队顺序、policy-lag 与 verified success；验证排序不会自动丢弃 stale sample。
6. 将上游 Qwen2.5-0.5B 四卡测试改写成可执行验收 manifest，标出模型、GPU 拓扑、queue pool、commit step、数据集、环境版本和预期断言；在没有实际执行前只能记录为 test design/source evidence。

## 2026-09 DeepSeek V4.1-Flash 架构、缓存与协议验收

1. 构造输入 1M token、输出 4K token 的 CED/decoder-only 计算代理账本，分别报告 prefill、decode、active-parameter proxy、TTFT 和 TPOT；在报告中说明代理量不能替代 FLOPs profiler。
2. 实现 SWA cache 的 `persist`/`replay` 两种恢复路径，注入错误 position offset、窗口边界、模型 revision 和未封存尾部，要求门禁返回失败而不是继续解码。
3. 用 toy block 构造 `Full -> Reindex -> Reuse` 层序列，统计 candidate recall、Top-K recall、indexer 次数、cache bytes 和跨层复用命中；可先运行 [`deepseek_v41_cache_demo.py`](research/model-update-2026-09/code/deepseek_v41_cache_demo.py)，再把两级漏检分开报告。
4. 对二维 KV 向量实现 E2M1-like FP4、INT8 和 BF16 教学表示，按 group size 测量量化误差、scale metadata 和检索排序变化；不要把 toy 数字写成 V4.1 kernel 性能。脚本中的 E2M1-like codebook 仅作起点，不能替代真实 E2M1 packing 和 dequant kernel。
5. 模拟 3-token DSpark 草稿，改变 acceptance length 和验证成本，比较目标调用数、有效 token、回退次数和 p95；加入草稿/目标 cache 不一致的恢复测试。
6. 根据独立 `encoding.py` 设计协议回归，覆盖 `reasoning_effort` 整数和别名、DSML 前导空格标签、中途 system message、交错图片、流式工具调用和 malformed output。
7. 设计多模态 Agent 的安全评测：图像中分别放普通文字、伪 system 指令、越权工具参数和恶意链接，记录模型识别、策略拒绝、工具执行和审计日志四层结果。
8. 复读模型卡 Agent 评测表，为 Terminal-Bench、DeepSWE、AutomationBench 和 Agent's Last Exam 建立条件表，标明哪些数字是发布方自报，哪些字段还没有独立复现。

9. 下载固定 revision 的文本源码（不下载权重），对 `inference/` Python 文件做 AST/编译检查，并把 `model.py` 的 `forward_spec` 调用链与 `generate.py` 的普通自回归调用链画成两张图；说明为什么这不能推出 DSpark 吞吐。
10. 根据 `inference/config.json` 建立 reference-runtime manifest，分别记录 `index_topk`、候选池、Engram、mHC/Sinkhorn、MTP/DSpark target 和量化 dtype；故意混入模型卡的 global KV 字段，设计 schema 检查拒绝未经来源标注的合并配置。
11. 为 EPD + SWA Bounded Replay 写一个恢复模拟：分别让 global KV 命中/SWA 缺失、候选池缺失、源码 revision 不一致和 DSML 版本不一致，输出 replay token、拒绝原因、恢复延迟和 `not_applicable` 指标。

12. 读取 vLLM `v0.30.0` 的 release API、registry、NVIDIA/ROCm V4.1 package 和 PyPI metadata，生成 source-aware manifest；把 `stable release surface`、`wheel available`、`full-weight load`、`numerical check`、`target profile` 和 `production SLO` 设为独立状态，禁止由前一项自动填充后一项。
13. 对照 vLLM `v0.29.0`、`main` 和 `v0.30.0`，画出 `DeepseekV41ForCausalLM`/`DSparkV41DraftModel` 的 registry 时间线；验证“main 有类”“stable tag 有类”“wheel 能安装”“draft/target verify 能运行”四个断言不能互相替代。
14. 从 v0.30.0 release notes 抽取 FlashMLA V4.1 MXFP8 whole-KV、Mega-mHC、Engram async prefetch、DSpark state folding 和 XGrammar strict tool parameters，分别写出 runtime integration、模型机制、质量 benchmark 三栏，并把没有直接证据的栏标为 `unverified`。

## 2026-09 K2 Horizon MoVA 与 Uno 验收

1. 根据 K2-Horizon-MoVA-36B-A4B 的配置，画出前 3 个 dense layer 与后 45 个 MoVA + MoE layer 的执行图，分别标注 32Q/8KV GQA、64 value experts/top-4、100 routed FFN experts/top-8 和 1 个 shared expert。
2. 用零依赖 Python 实现 sigmoid routing、selection-only bias、top-k 原始分数归一化和 `scaling_factor=2.5`，加入错误实现对照，验证两者的 selected experts、权重和与输出差异。
3. 为 `B * T` 个 token 建立 MoVA/FFN assignment、hidden payload、GQA KV cache、权重、workspace 和通信 buffer 账本；改变 EP 设备数、负载倾斜和 padding，解释为什么 4B active 不能直接换算显存或吞吐。
4. 用 8K、32K、128K 和 512K 四个阶段设计长上下文实验，记录 position 配置、prefill 峰值、KV bytes、TTFT、TPOT、召回和失败类型；不得把 `max_position_embeddings=524288` 当作长任务成功率。
5. 把 0.9B 卡片的领域专家合并/MOPD 与 36B 的运行时 MoVA routing 做对照表，分别写清优化对象、发生阶段、checkpoint 形态、部署影响和可核验来源。
6. 设计 Uno adapter 的 speculative decoding toy：冻结 AR base，加入 LoRA diffusion draft，比较 draft 接受率、rejection verification、回退比例、有效 token/target call 和质量；固定 base/adapter revision、采样器、batch、硬件和 harness。
7. 写一份 K2 迁移清单，至少覆盖固定 revision、tokenizer/chat template、reasoning/tool parser、TP/EP 拓扑、FlashAttention-3、路由 GEMM override、流式工具调用、错误恢复和许可证；将模型卡声明与本地实验结果分栏。

## 2026-09 Qwen3.8 架构与 Serving 验收

1. 用零依赖 Python 实现 QSA 的 micro-block 评分、block-causal top-k、token 展开和 causal tail 保留；分别测试预算足够、不足以容纳 tail、query 位于完整 block 末尾和空输入的边界。
2. 比较先 pooling 后 partial RoPE 与先 RoPE 后 pooling 的 toy 表示，记录位置相位抵消、block recall 和排序变化；明确 toy 结果不能当作 Flash-Next kernel 性能。
3. 为 QSA 设计 dense distillation、sparse-only 和两阶段训练三组消融，固定 teacher、block size、top-k、tail、tokenizer 和 revision，报告长上下文召回、loss、indexer 成本和失败类型。
4. 实现标准 residual、四分支无 gate、逐元素 read gate/branch scalar write gate 三种 toy 结构，记录激活范数、梯度代理、residual bytes、读写带宽和下游检索结果。
5. 建立 N-gram table 账本，改变 slot 数、n-gram 命中率、host bandwidth、prefetch latency 和 out-of-domain PPL，同时与 RAG、KV cache 的存储/请求时机做对照。
6. 实现 Muon/AdamW 参数分组审计，比较全 AdamW、按语义拆分、错误地对 fused QKV 直接正交化三种配置，检查 router overflow、loss spike、step time 和 checkpoint resume。
7. 为 27B、A95B、Flash-Next、Max 建立接口对照表，分别记录榜单来源、模型卡、total/active 参数、模态、thinking、hosted/open 状态和待核验字段；不要把 effort 行计为新 checkpoint。
8. 静态审计 vLLM `v0.30.0` 的 QSA dtype gate：将权重 dtype、QSA 激活/QKV、主 KV 与压缩 indexer-key cache 分成四列，验证 BF16/FP8 E4M3 组合边界；不得把源码读测写成 GPU 运行通过。
9. 对照 SGLang `v0.5.20` 的 stream 依赖图，标出 indexer、当前 stream 的 Q/K/V 准备与最终 attention launch；用 event trace 设计一个可证伪的 overlap profile，并注明报告中静态源码核对不证明 kernel 实际重叠或端到端加速。
10. 为 51B PLE/N-gram table 建立 device、pinned-host/UVA 和 file-backed mmap 三种容量/延迟账本；分开引用 SGLang 注释的约 47.7 GiB FP8 table 与 vLLM recipe 的 H100 余量提醒，再设计包含 page fault、预取命中、RSS、带宽和 TTFT 的硬件实验。

## 2026-09 GLM-5.3-Flash 混合注意力与视觉闭环验收

1. 根据固定 `config.json` 画出 45 层中 34 个 `linear_attention` 与 11 个 `deepseek_sparse_attention` 的路径，分别标记递归 state、显式 KV、indexer、top-k 和 causal tail；用 toy 输入验证候选池漏检与最终 top-k 漏检是两个错误来源。
2. 实现 4 个 indexer key 的加权 pooling 和 block top-k，改变 pool size、block size、tail 规则与 `index_topk`，分别报告候选召回、最终 token 召回、排序成本和 buffer 字节；不要把 toy 数字写成 GLM 生产 kernel 性能。
3. 实现标准 residual 与 Sinkhorn 双随机 residual mixing，比较深度增加时的行和、列和、激活范数代理和数值误差；说明 mHC 约束不等于范数恒等式或零开销稳定性证明。
4. 构造 Encode–Prefill–Decode 的多模态请求 manifest，注入 image encode 超时、representation revision 不一致、取消、decode 重算、工具权限不足和视觉 diff 变差，要求系统拒绝错误 artifact 并保留 trace。
5. 用相同 toy 任务对比“只编译通过”“只截图相似”“编译 + 行为测试 + 视觉 diff + artifact validator”四种完成门禁，计算误通过率；把模型 self-judgment 与宿主 verifier 分栏。

## 2026-09 GLM-5.3-Flash runtime 双状态与 MTP 验收

1. 为同一 session 实现 paged KV pool 与 KDA state pool 两个 allocator，注入 KV 命中但 KDA state 缺失、KDA state 恢复但 sparse indexer metadata 缺失、dtype/backend 不一致三类故障；输出恢复状态、重算范围、并发占用和最终 artifact gate。
2. 模拟 SGLang 的 MTP `5/1/6` 低延迟配置和 high-throughput 关闭 speculative 两条路径，改变 draft 成本、接受长度、回滚位置、tool-call boundary 和 batch；记录 accepted/rollback tokens、target calls、工具 parser 错误、TTFT/TPOT 和 task success。
3. 建立 KV dtype/DSA backend capability matrix，至少覆盖 FP8 KV + TRT-LLM DSA、BF16 KV + TileLang DSA、无效的 FP8 KV + TileLang DSA；在启动前拒绝非法组合，并在 toy long-context 任务中比较召回、显存和 p99。
4. 将 EPD/PD 部署拆成 wiring、dummy-weight、full-weight load、numerical correctness、state recovery、target profiling、tool/verifier acceptance 七道 gate；分别注入视觉 encode 超时、representation revision 不匹配、取消、TP 数值偏差和 speculative 不支持。
5. 对同一 toy 模型分别记录 Transformers 基础接入、vLLM recipe、SGLang recipe 和本机 runtime 验收，输出 capability、依赖、硬件、权重、cache/state recovery 和 SLO 的证据矩阵；禁止用框架类名替代端到端通过。

6. 建立 GLM-5.3-Flash 的 source-aware capability matrix：分别登记 SGLang `v0.5.20` tag、SGLang `main`、vLLM `v0.29.0`、`v0.30.0` tag、固定 main commit 和 recipe。逐文件比对 stable/main 的目录、blob 与关键 diff（Indexer/Tail cache、KDA、MTP、RoPE、Quark loading、KPool stride），区分“目录项存在”“文件实现已逐项确认”“本地运行”“目标硬件通过”；为不同 tail-cache stride/layout 构造数值与边界测试，但在测试前不预判故障归属。
7. 构造 indexer pool 边界故障：让 `index_kpool=4` 的历史在最后一个 pool 未填满时切换 prefill/decode/PD，分别恢复只完成 pool、pool+tail、pool+tail+KDA 三种 manifest，比较可见 token、重算范围和错误类型。
8. 对 MTP 的 top-k 复用做正确性实验：固定 draft/target logits，分别启用/关闭 sparse index top-k reuse、slot compact 和 local argmax，注入拒绝、工具调用边界和 batch reorder，记录 accepted/rollback token、index recall、全词表通信量和最终 task result。
9. 为 stable/main/recipe 证据设计审计报告：保留 vLLM `v0.29.0` 无专属路径、`v0.30.0` 有 `glm5next` 目录项、main 有详细文件的时间线；目标 wheel、逐文件比对、权重 revision 和未完成 gate 必须独立核验，禁止从目录存在自动生成“stable runtime 已验收”结论。

## 2026-09 GLM-5.2 IndexShare、MTP 与长轨迹 RL 验收

1. 用四层 toy DSA 实现独立 indexer 与 IndexShare 两种路径，改变层间 query 相似度和 `top_k`，分别报告 index recall、最终 token recall、indexer 次数、排序成本和复用 buffer；不能把共享 top-k 视为无损。
2. 模拟 MTP speculative decoding，改变 draft 成本、接受长度、verify 成本和拒绝位置，输出 target calls、有效 token、rejection correction、KV commit 和回退比例；加入 draft/target cache 不一致测试。
3. 为 compaction 后的多段轨迹构造 group-wise 与 critic-based 两种 advantage 计算，注入不同 segment 数、长度和 bootstrap 边界，检查 token mask、KL 归属、长度归一化和空 segment 的 `not_applicable` 语义。
4. 构造 coding-agent anti-hack toy 环境，分别注入读取隐藏文件、下载参考答案和访问上游提交三类 shortcut；比较规则过滤器、模型分类器和联合检测的误报/漏报，并验证违规后继续轨迹与环境隔离策略。
5. 设计 1M context serving 压测矩阵，固定 revision、硬件和并发，改变 cache 命中、prefill/decode 组织、CPU cache transfer 和请求长度；记录 TTFT、TPOT、p99、显存峰值、传输字节、失败重算和单位成功成本。

## 2026-09 DeepSeek V3.2 DSA、工具推理与 Agent 数据验收

1. 用同一组长上下文样本实现 dense attention、轻量 indexer + top-k 候选、候选漏检注入三条 toy 路径，分别记录 index recall、最终 token recall、排序成本、KV/indexer bytes、TTFT/TPOT 和长程失败；明确 toy 结果不能替代 V3.2 生产 kernel。
2. 设计 `thinking with tools` 的消息状态机，覆盖 reasoning、tool call、tool result、final answer、格式错误、重复 call、超时和取消；输出 parser 成功率、schema 错误率、权限拒绝率、未执行 call 率和 trace 完整率。
3. 构造 agentic task synthesis 数据流水线，输入任务契约、环境、工具轨迹、verifier、失败原因和污染标签，输出 shortcut rate、verifier false-positive/false-negative、失败轨迹覆盖、环境泄漏率和训练样本准入门禁。
4. 对比 group-wise advantage、critic/token-level advantage 和不做长度归一化三种教学方案，改变 rollout 数、轨迹长度、segment mask 和 reward 稀疏性；再加入 V3.2 报告中的 unbiased KL、negative off-policy sequence masking、Keep Routing 和 Keep Sampling Mask 开关，报告 advantage 方差、空 segment、KL 归属和 credit-assignment 失败。不要把 toy 结果写成 V3.2 的完整 RL recipe。
5. 做 V3.2 与 V3.2-Speciale 的协议兼容性审计：同一工具任务分别记录 tool-call support、parser 状态、权限门禁、最终 artifact 和不适用指标；对 Speciale 的 tool calling 指标返回 `not_applicable`，而不是记为 0。

6. 读取 V3.2-Exp inference demo，分别实现 indexer 的 non-interleaved RoPE 和 MLA 的另一种 RoPE layout；用相同 hidden state 与位置输入检查 layout 混用时的 score 偏差、top-k 变化和最终 attention 输出差异。
7. 用 toy FP8 query/key cache 实现 `fp8_index` 的分块矩阵乘、per-head ReLU/max、权重乘法和 head reduce-sum；对比 BF16 reference，报告 index score 的误差、top-k overlap、causal 边界错误和不同 `index_topk` 的成本。
8. 实现 prefill MHA/decode MQA 两条 sparse MLA 路径，分别统计 latent KV cache、positional cache、FP8/BF16 bytes、TTFT、TPOT 和 batch 维度；禁止用 decode 的单步数字代表 prefill 峰值。
9. 实现两阶段 radix/histogram top-k selector，与完整排序和 `torch.topk` 对比；覆盖 ties、`topk > end_pos`、paged KV、不同序列长度、causal mask 和无效位置，输出准确性、访存量和选择延迟。
10. 复现 serving 并行账本：输入 GPU 数、专家数、`DP/EP/TP`、batch、KV dtype 和拓扑，比较 `DP=8, EP=8, TP=1` 与 TP fallback 的专家通信、KV/权重显存、warmup、p99、失败恢复和单位成功成本。
11. 将 vLLM recipe 的 GSM8K 5-shot/20-shot 结果做证据分层表，绑定 V3.2-Exp revision、vLLM/DeepGEMM/FlashMLA 版本、lm-eval prompt、硬件和并发；输出 `recipe_result`，不得输出 `base_model_score`。

12. 按报告 Eq. (5)–(9) 实现一个可检查的 GRPO toy：分别记录组均值、response token mean、current/old importance ratio、reference KL、clipped objective 和 sequence mask。构造正/负 advantage、长短 response、训练/推理概率不一致的样本，验证 Eq. (9) 的门控只屏蔽负 advantage 且高序列级 divergence 的 policy 项，而 KL 项仍保留；比较组内中心化与额外 z-score 作为显式教学变体，不能把变体说成报告实现。

## 2026-09 DeepSeek V3.2 当前快照与实现证据分层

1. 读取两个 AA 页面快照，比较 9 月 20 日的 648B 与 9 月 21 日的 685B，设计一个只输出 directory/provider drift 的解析器；若输入没有 output-speed/TTFT，则保持 null，不能用其他模型补值。
2. 读取 V3.2-Exp README 的 benchmark 表、RoPE 修复说明和 TileLang/DeepGEMM/FlashMLA/SGLang 入口，建立 release description、reference implementation、CUDA kernel、serving recipe 四列证据表；验证每列都不能自动升级为 full-weight load、hardware profiling 或 tool acceptance。
3. 模拟 vLLM recipe URL 返回 404、HF/PDF 返回 503 和固定 revision 历史快照同时存在的情况，输出 access boundary、resource identity 和 historical evidence 三种状态；禁止把线路失败写成资源不存在。

## 2026-09 Qwen3.8 Max (0902) Revision 与缓存协议验收

1. 为 1M context、991K 普通输入、983K thinking 输入和 131K 输出建立 token/KV/cache/workspace 账本，改变 reasoning effort、工具结果长度和并发，输出 TTFT、TPOT、p95、峰值内存、失败重算和单位成功成本；不要把四个上限直接相加。
2. 实现客户端 schema gate：`reasoning_effort` 与 `thinking_budget` 同时出现时拒绝；thinking + forced `tool_choice` 时拒绝或显式切换到非 thinking 路径；记录协议错误与模型质量失败的区别。
3. 构造 `explicit`、`implicit`、`session` 三类 toy cache，模拟 1,024-token 资格阈值（不等于保证命中）、`ephemeral` marker（最多 4 个且只取最后 4 个）、20 content-block lookback 与独立的 `other messages≤20` follow-up 规则、命中/失效/5 分钟 reset 和租户隔离。分别解析 Responses `usage.input_tokens_details.cached_tokens` 与 Chat Completions `usage.prompt_tokens_details.cached_tokens`；标记文档中 `1,024` 资格表述与 explicit-cache 请求示例 `>1,024` 的边界差异，不能声称 toy 已证明线上 hit。
4. 为 Qwen3.8 Max 0902 设计多工具 Agent manifest，记录 alias、resolved revision、mode、effort、工具 schema、权限决定、tool call、执行回执、超时、取消、重试和最终 artifact；把模型 call、宿主接受和真实副作用三种状态分开。
5. 对比 Qwen3.8 Max 泛化 `qwen3_8_max_xhigh` 与一个固定 revision 的 toy harness，报告模型/版本、harness、工具、环境、verifier、Pass@1、区间、成本和 output tokens；验证不能把泛化 DataCurve 行迁移成 0902 独立结果。
6. 模拟 QwenCloud endpoint 从 `dashscope-intl.aliyuncs.com` 迁移到 `maas.qwencloudapi.com`，加入 account+model 聚合、workspace override、月度 soft TPM、429、`Retry-After`、queue latency 和 observed TPM；报告 provider quota、模型 tokens/s、GPU capacity 与单位成功成本的差异，验证 endpoint/限流变化不能被写成模型升级。
7. 运行 [`qwen_max0902_cache_contract_audit.py`](research/model-update-2026-09/code/qwen_max0902_cache_contract_audit.py)，确认 thinking + forced tool、`reasoning_effort`/`thinking_budget` 冲突、两种 20 上限、endpoint-specific usage 解析、implicit 非保证命中和 `previous_response_id` session lineage 的边界；将结果标为 `local_protocol_toy`，并说明未请求真实 endpoint。

## 2026-09 GPT-5.3 Codex Responses 与 Agent harness 验收

1. 为 GPT-5.3 Codex 建立 400K context、272K maximum input、128K maximum output 的请求账本，额外记录 reasoning tokens、visible output、工具 schema/results、workspace 和 compaction item；验证不能把三个上限直接相加推导并发。
2. 构造 Responses replay 对照：完整 output items、只保留最终文本、使用 `previous_response_id` 三种方式，检查加密 reasoning item、assistant `phase`、tool call、tool result 和下一轮恢复差异。
3. 实现 Codex-style tool gate，分离模型提出 call、schema/parser 校验、宿主权限/审批、sandbox executor、工具回执和最终 artifact；覆盖 shell、patch、skill、超时、取消、重试和幂等副作用。
4. 模拟 server-side compaction 与 standalone canonical context，注入手工删除前置 item、工具集合变化、权限变化、重复执行和 workspace artifact 缺失，输出恢复成功率、重复副作用率、cache hit 变化和失败原因。
5. 对比 GPT-5.3 Codex 的 `medium/high/xhigh` 与一个固定 toy harness，报告 task success、model calls per task、工具轮数、TTFT、TPOT、p95、compaction recovery、prefix saved tokens 和单位成功成本；禁止使用其他 GPT/Codex 的 DataCurve 行作为 Codex 结果。
6. 阅读 Artificial Analysis、OpenAI 模型页和 Codex Prompting Guide，分别列出“榜单字段”“官方 API/harness 字段”“当前未公开的架构/训练字段”，验证不从 hosted shell、skills 或 phase 反推模型内部结构。

## 2026-09 GPT-OSS MoE、量化与 Harmony 验收

1. 为 `gpt-oss-120b` 和 `gpt-oss-20b` 建立 total parameters、active parameters、专家权重、KV cache、dispatch、通信 buffer、workspace 和并发显存账本；改变 top-4 路由、batch 和专家负载倾斜，说明 active 参数为什么不能直接换算显存或吞吐。
2. 实现一个零依赖的交替 sliding/full attention toy，输入层数、窗口、dense 层间隔和上下文长度，分别输出局部层 FLOPs 代理、dense 层全局访问、KV bytes 和 long-range needle recall；明确 toy 结果不代表 gpt-oss kernel 性能。
3. 实现 MXFP4-like 分组量化与 BF16/INT8 对照，改变 group size、scale、专家权重分布和 GEMM batch，报告量化误差、路由排序变化、scale metadata、显存代理和输出一致性；不要把 toy 结果写成官方量化误差。
4. 为 Harmony 构造 `system/developer/user/assistant/tool`、`analysis/commentary/final`、recipient、function call/result 和 structured output 的渲染/解析回归；注入错层级、错误 channel、非法 schema、重复 tool result 和被回灌的历史 reasoning，检查 parser、权限和 replay 门禁。
5. 对同一固定 revision 比较 low/medium/high reasoning effort，控制 prompt、工具、sampling、最大输出和 verifier，记录 CoT/visible token、成功率、TTFT、TPOT、成本和失败类型；把 effort 配置与模型身份分开。
6. 实现 Responses raw CoT 回放器：解析 `reasoning.content[].reasoning_text`、`response.reasoning_text.delta/done`、`item_id` 和 index，测试乱序、重复 delta、缺失 done、tool result 重试和 final 后 analysis 清理。
7. 实现 Chat Completions 兼容层的 `reasoning`/delta 约定，并与 Responses 形状做同一工具任务对照；标记哪些字段是 provider 约定，哪些是模型训练所需的 Harmony 状态。
8. 运行官方 compatibility-test 的 smoke test，再按 AIME 16 次/题、GPQA 8 次/题、HealthBench 1 次/题运行质量 eval；将 invalid requests、pass@k/pass^k、tool-call 正确率、模型质量和 kernel/hardware 门禁分开报告。

## 2026-09 Claude Opus 4.6 长任务协议审计

1. 构造一个包含 thinking block/signature、两次 tool call、tool result 和 compaction block 的多轮 trace；分别测试完整 replay、只保留可见文本、篡改 signature 三种恢复路径，报告恢复成功率、重复副作用和丢失状态字段。
2. 实现 `defer_loading` 工具目录，比较全部 schema 注入、regex 搜索、BM25 搜索三种路径的上下文 token、选择错误、权限拒绝和工具执行成功率；将 tool search 与 permission gate 分开计分。
3. 模拟 `computer_20251124` 执行器，加入域名 allowlist、人工确认、截图 prompt injection、取消和超时；验证模型动作提案、宿主接受、真实副作用和最终 artifact 四种状态不能合并。
4. 为 `effort=low/high/max` 建立质量—延迟—成本账本，固定模型、任务、工具、harness 和 verifier；不要把 effort 行记录成三个 checkpoint，也不要用 `max_tokens` 代替真实 reasoning budget。
5. 将 AA 配置字段、Anthropic 发布方 benchmark 和无精确 DataCurve 行写入同一审计报告，验证报告拒绝迁移 Opus 4.8/5 的 Agent 分数。

## 2026-09 Claude Opus 4.7 预算与视觉契约验收

1. 为同一组长任务分别模拟 `effort=high/xhigh/max`、task budget 和 `max_tokens`，记录 step、loop、request 三本账，验证三者不能相加推导并发。
2. 构造包含 thinking、tool call、tool result、compaction 和 final output 的 Agent trace，比较完整 replay、只保存可见文本和 compaction 后错误重置预算三种恢复路径。
3. 用同一文档集对比 `1568 px/1568 visual tokens` 与 `2576 px/4784 visual tokens`，记录图表/坐标召回、输入 token、TTFT、缓存命中、坐标映射错误和单位成功成本；不要把 toy 结果写成视觉编码器性能。
4. 构造 tokenizer 迁移账本，固定内容类型，比较旧版本与 Opus 4.7 的输入 token、缓存前缀、thinking/output 和重试成本，验证 `1.0-1.35x` 只是发布方经验范围而非每个请求的保证。
5. 构造 cyber safeguard 测试矩阵，分别记录模型能力、实时策略拦截、组织授权、沙箱、网络隔离、人工升级和最终 artifact；将误报/漏报与工具执行副作用分栏。

## 2026-09 Claude Opus 4.8 Dynamic Workflows 与多 Agent 评测

1. 为同一个 BrowseComp-style 任务实现 single-agent、blocking orchestrator、fixed peer team 和 async-subagent 四种 harness；明确每个 Agent 的任务视图、工具、上下文、消息协议、token cap 和结束条件。
2. 按 System Card 的报告方式分别计算 score、跨全部 Agent 的 total tokens 与 derived latency；再记录真实 wall-clock/p50/p95。解释 blocking orchestrator 88.5%、fixed 5-agent 85.4% vs single 84.3%（5M vs 10M token limit、约 20% 派生 latency）为什么不是模型裸能力或生产 SLO。
3. 将任务按预先固定的难度切片，比较易题和 hard tail 的多 Agent 净收益；加入协调成本、重复劳动、token 成本和 verifier 错误率，检查 Agent 数量增加是否真的提升 NetLift。
4. 设计可恢复的长迁移 workflow：持久化任务 DAG、checkpoint、子任务 owner、重试/取消、permission manifest 与最终 artifact digest；让 reviewer 进行反驳，再由独立测试/verifier 验收，覆盖中断恢复和重复副作用。
5. 复现“总结不完整 coding transcript”的 honesty 测试，分别标记未完成特性、失败测试和未批准设计；把模型是否主动披露与外部测试结果分开计分，并说明发布方短上下文 off-policy `3.7%` 漏报率不是通用诚信率。

## 2026-09 Kimi K3

1. 为 3:1 KDA/Gated MLA 画出递归状态、全局 attention 和 output gate 的数据流，并比较全 MLA 的 KV/吞吐账本。
2. 模拟 896 experts、top-16、冻结 expert bias，测量均匀路由、热点路由和 overflow 对 all-to-all 与 p99 的影响。
3. 构造 XTM `think -> tool call -> tool result -> response` trace，删除 channel 或 tool/index 后验证 replay 门禁。
4. 固定 model revision、harness、hardware、effort、task set 和 verifier，分别记录 AA、DataCurve、官方 benchmark 与 toy 结果，禁止横向拼接。

## 2026-09 Qwen3.5-397B-A17B 验收

1. 为 397B total/17B active、512 experts、10 routed + 1 shared 建立 total/active/resident weights/state/KV/通信/workspace 显存账本，验证 active 参数不能直接换算并发容量。
2. 实现一个零依赖 Gated DeltaNet state toy，输入 `alpha`、`beta`、`q/k/v`，输出 state 更新；与 full-attention KV cache 对比 state bytes、历史检索能力假设和长序列成本边界。
3. 画出 15 组 `3 x Gated DeltaNet + 1 x Gated Attention` 的层路径，分别估算递归层和显式 attention 层的计算、state/KV 和通信账本。
4. 模拟 MTP draft/verify/rollback/commit，改变 accepted length、draft steps、batch 和 verifier 延迟，输出 target calls、committed tokens、TTFT、TPOT 和失败重算；不要把结果写成官方线上性能。
5. 模拟 multimodal mode 与 `language-model-only` mode，记录 vision encoder、视觉 token、KV cache、显存峰值和请求协议差异，明确两者是同一模型的服务模式。
6. 设计 million-agent asynchronous RL toy，记录 rollout queue、policy version、trajectory freshness、verifier latency、stale ratio 和单位成功成本；把 Qwen 官方声明与本地 toy 结果分栏。

## 2026-09 GLM-5 DSA、异步 RL 与 Agentic Engineering 验收

1. 为 `744B total / 40B active`、256 routed experts、top-8 和 1 shared expert 建立 total/active/resident weights/通信/KV/indexer/workspace 六本账，说明哪些字段不是模型卡直接给出的。
2. 实现 dense attention 与 indexer+top-k 的 toy 对照，改变 `top_k`、关键 token 位置和尾部保留策略，分别报告 index recall、最终 evidence recall、attention 计算代理值和任务成功率；不得把 `index_topk=2048` 写成无条件最终可见 token 数。
3. 模拟 `slime` 的 rollout/verifier/trainer 异步队列，改变 policy lag、样本 TTL、verifier 延迟和失败重试，输出有效样本率、stale ratio、trainer 空转、吞吐和 reward 偏差。
4. 构造 coding Agent 的 artifact gate：模型提出 patch、shell 和测试，宿主独立校验权限、执行回执、测试结果、artifact digest 和未解决失败；比较“模型声称完成”和“验收通过”的误差。
5. 写一份证据分层报告，分别列 Artificial Analysis、Z.ai 发布方 benchmark、DataCurve（精确行缺失）和本地 toy 结果；禁止迁移 GLM-5.2/5.3 的 DeepSWE 数字。

## 2026-09 Gemini 3.5 Flash-Lite Thinking、视频与证据边界验收

1. 为 `minimal/low/medium/high` 建立请求级预算实验，固定 model code、prompt、工具、输出上限和 verifier，输出 reasoning token、可见输出、TTFT、TPOT、工具轮数、成功率和单位成功成本；验证 level 不是四个 checkpoint，也不是严格 token 上限。
2. 用同一组长视频对比 static 约 1 FPS 取帧和 agentic 时间轴浏览，注入目标事件位于开头、中部、末尾、音频 transcript 才包含关键信息以及错误时间轴请求；输出证据召回、`processing_call/result` 数量、总 token、延迟、超时和端到端成功率。
3. 构造视频 Agent 状态机，覆盖 file handle 过期、租户权限拒绝、processing timeout、取消、重复 `call_id`、部分 transcript 返回和工具副作用；把模型提案、宿主授权、处理回执和最终 verifier 结果分开记录。
4. 建立 1M context 长上下文账本，分别改变输入长度、媒体 token、缓存命中、thinking level、工具结果和并发，记录峰值内存、TTFT、TPOT、有效 needle recall、失败重算和单位成功成本；不要从 1M 上限直接推导均匀记忆能力。
5. 读取 Lite Model Card 与其引用的 3.1 Flash-Lite Model Card，做一张“Lite 自身字段 / 官方前代依赖 / 发布方评测 / 尚未公开”表；验证不能把 3.1 架构、训练数据或硬件资料写成 3.5 Lite 独有创新，也不能迁移 Gemini 3.5 Flash 的默认 `medium`。

## 2026-09 Kimi K2.6 Native Multimodal、Agent Swarm 与推理验收

1. 根据 K2.6 配置建立 total/active/resident/通信/KV/cache/workspace 账本，覆盖 1T/32B、384 experts、top-8、1 shared expert，并说明哪些数值不是配置直接给出的。
2. 用 toy MLA/cache 模型改变 `q_lora_rank`、`kv_lora_rank`、上下文长度、视觉 token、工具结果和并发，输出 cache bytes、prefill/decode 代理成本、TTFT/TPOT、p99 和有效检索率；不得宣称复现生产 kernel。
3. 模拟 native INT4：加入 group size 32、scale、未量化模块、反量化 dtype、视觉路径和量化误差，比较启动成功、长输出、JSON tool call 和多模态质量；把本地 toy 结果与官方配置分栏。
4. 设计 300 sub-agent/4,000-step Agent Swarm toy，加入任务 DAG、共享只读证据、隔离 workspace、权限、预算、取消、未知工具状态、重试和 artifact owner，比较串行、并行和 verifier 门禁的单位成功成本。
5. 实现 `preserve_thinking`/`reasoning_content` 状态回放检查器，注入遗漏 reasoning、错误 tool index、schema 版本变化、重复副作用和 compaction，输出 replay completeness、cache miss、权限拒绝和恢复结果。
6. 按 KVV 六类检查设计部署验收报告：pre-flight、OCRBench、MMMU-Pro、AIME2025、K2VV ToolCall 和 SWE-Bench；明确每项故障应归因于模型、量化、kernel、parser、harness 还是 verifier。
7. 写证据分层表：Artificial Analysis 第三方字段、Kimi 官方模型卡/博客 benchmark、KVV 验收、DataCurve 精确行缺失和本地 toy；禁止迁移 K2.7 Code/K3 的 DeepSWE 结果，也禁止把 Agent Swarm 数量写成 MoE expert 数量。

## 2026-09 Kimi K2.7 Code：INT4、长周期工具状态与视频 token 预算

1. 根据固定 K2.7 Code config 写权重内存账本，区分 total 1T、active 32B、驻留权重、4-bit pack、group size 32、量化忽略模块和运行时开销；全量 INT4 只计算理论原始字节下限，不宣称 checkpoint 或 GPU 显存大小。
2. 用 kv_lora_rank=512、qk_rope_head_dim=64、61 层和 BF16 构造 MLA cache shape 估算，分别改变 context、dtype 与并发；明确该算式是假设代理，不能替代 K2.7 的目标 backend profile。
3. 运行 kimi_k27_contract_budget_toy.py，注入 disabled thinking、temperature/top_p 偏离、tool_choice=required、错配 tool_call_id 和遗漏 reasoning_content；区分硬错误与文档建议产生的连续性 warning。
4. 设计长周期 coding Agent 的工具回放账本：记录 proposal、schema validation、permission decision、执行 receipt、幂等键、工具版本、reasoning state 与 verifier；模拟执行成功但响应丢失，证明重试不能无条件重复副作用。
5. 用同一视频样本比较不同抽帧、片段长度和分辨率的 token estimate、TTFT、tool calls、成本和任务成功率；包含上传大小限制、URL 图片拒绝、file handle 过期及时间段越权。
6. 复核 Kimi Code Bench v2、Program-Bench、MLS Bench Lite、Kimi Claw 24/7、MCP Atlas、MCP Mark Verified 的官方结果；记录 snapshot、CLI/harness、参数、context、工具预算和 verifier，禁止与 DataCurve 或 AA 直接合并排序。
7. 把 Kimi K2.7 Code Highspeed 作为同模型服务路由变量设计 profile；若比较 180/260 tokens/s 发布方声明，需同时固定输入/输出长度、并发、TTFT/TPOT、供给和错误率，不新增模型候选。
8. 固定 KTransformers commit `c40722bf04c494f2492b7eb9e86ef01a4ede45b3`，逐项对照 K2.7 deployment guidance、K2.5 serving guide 和 Native Precision 支持矩阵。输出 `publisher_command`、`generic_matrix_entry`、`runtime_version`、`weight_load`、`numerical_test` 五列，缺证项标为 `unverified`；不得因矩阵未列就判定不支持，也不得因命令存在就判定已兼容。
9. 用 `threshold=400` 写出 dual prefill 的边界表：399、400、401 tokens 分别对应哪个模式；标注这是 pinned tutorial 的文档语义。设计目标 GPU 测试，记录 CPU/GPU expert placement、NUMA/threadpool、prefill peak VRAM、TTFT 和吞吐，并检查 K2.7 示例未显式传入 dynamic expert update 开关。
10. 对比 K2.7 发布指南 LoRA SFT 段、K2.5 旧 source-install 教程与 `0.7.0.post4` release tutorial/YAML。标记版本后，分别记录 base checkpoint、权重后端、`bf16` 训练字段、LoRA target modules、sequence length、optimizer 和硬件；不得把旧文档的 BF16 转换或新 K2.5 配方外推为 K2.7 事实。若无 K2.7 原始材料，就写出需要的权重/训练 probe manifest，不下载权重或伪造复现结果。
11. 根据 K2.5 release tutorial/YAML 画出 `base RAWINT4 -> attention/expert LoRA training -> resumable checkpoint -> SGLang adapter conversion -> serving` 数据流；标注普通/expert adapter 与 optimizer/RNG/scheduler 状态的保存边界，并列出 exact-resume 比对项。把该流程限定在 K2.5 `0.7.0.post4`，不要改写成 K2.7 已验证能力。

## 2026-09 Kimi K3 固定 manifest 与 hybrid cache

1. 下载固定 revision 的 `config.json` 和 `model.safetensors.index.json`（只下载 metadata），运行 [`kimi_k3_manifest_audit.py`](research/model-update-2026-09/code/kimi_k3_manifest_audit.py)，解释 96 个分片、497,220 个 tensor、93 层、92 个 MoE layer 和 247,296 对 packed/scale tensor 的关系。
2. 比较 index 的 packed `metadata.total_size` 与 HF API 的 U8/BF16/F32 参数统计，写出两个字段的适用场景，并证明不能把它们相加成“模型大小”。
3. 根据 `KimiDynamicCache` 画出 MLA `key/value` cache、KDA `conv/recurrent` state、prefill `chunk_kda` 和单 token `fused_recurrent_kda` 的状态转移图。
4. 注入 MLA cache 丢失、KDA state 丢失、prefix-match unit 改变、KV dtype/backend 不一致、expert scale 缺失和 revision 错配，设计拒绝/恢复策略；要求 schema、权限、幂等和 verifier 仍在外部副作用之前。
5. 固定 FlashKDA 的 `T/H/D`、warmup、iters、repeats、GPU、CUDA、driver 和 baseline，说明局部 kernel latency 为什么不能替代 hybrid cache 正确性、端到端 TPOT 或线上 Agent acceptance。

## 2026-09 GPT-5.4 mini/nano 路由、能力矩阵与成本验收

1. 建立 `gpt-5.4-mini-2026-03-17`、`gpt-5.4-nano-2026-03-17` 和 GPT-5.4 base 的 serving manifest，分别记录 400K/272K/128K 与 1.05M/272K/128K 的上下文/输入/输出边界；验证不能把 base 的 context 或工具清单复制给 sibling。
2. 写一个 task-shape router toy：输入任务的歧义、规划深度、工具数量、失败代价、模态和输出 schema，输出 nano、mini 或 base 路由；加入 verifier、升级、重试和 abstain，比较任务成功率、延迟、调用轮数和单位成功成本。
3. 用固定 prompt 对比隐式指令与显式 Prompt Contract，至少覆盖目标、前置依赖、工具顺序、schema、停止条件、工具失败恢复和缺失输入；记录 reasoning、visible output、tool/result、retry、cache 和 executor wait 账本。
4. 实现按 model ID 的 capability probe：让 mini 尝试 `tool_search`/`computer_use`，让 nano 进入不支持这两项的降级路径；将模型提案、宿主授权、沙箱执行、工具回执和 verifier 结果分开记录，不把 tool call 当作执行成功。
5. 写证据分层报告：Artificial Analysis mini/nano 指数与价格、OpenAI 官方模型页/指南、DataCurve base 行和本地 toy 结果分别列账；明确不能迁移 GPT-5.4 base 的 Pass@1、成本、Agent steps，也不能从 mini/nano 名称推断内部架构或训练 recipe。

## 2026-09 DeepSeek V4 Pro CSA/HCA、Responses 与 harness 审计

1. 为 V4 Pro 建立三层 manifest：AA 的 `deepseek-v4-pro` 配置、官方 `deepseek-v4-pro` API/配置、DataCurve 的 `mini_swe_agent_deepseek_v4_pro_max`；分别记录 release date、model ID、snapshot、effort、harness、工具、环境和 verifier，禁止把三层字段合并成一个模型结论。
2. 扩展 CSA/HCA toy：输入 block compression stride、CSA top-k、HCA dense read、滑动窗口、尾部 token 和因果 mask，输出压缩倍率、候选召回、最终 evidence recall、KV bytes、indexer bytes 和 attention FLOPs 代理值；明确 toy 不复现生产 kernel。
3. 建立 1.6T/49B MoE serving 账本，加入 384 routed experts、6 selected、1 shared、resident weights、FP4/FP8 scale、dispatch/combine、KV/indexer cache、workspace 和 batch；比较专家热点、通信量、p99 和单位成功成本，不能用 49B 直接当完整显存。
4. 写 stateless Responses replay toy：模拟 `function_call`、`apply_patch`、并行工具、工具回执、失败重试和宿主持久化；注入 `previous_response_id`、`conversation`、`background`、`store` 等不支持字段，验证 capability probe、静默忽略检测、权限和幂等门禁。
5. 对 `low/high/max` 固定任务、工具、revision 和 verifier，报告 reasoning/visible output、TTFT、TPOT、工具轮数、失败类型、Pass@1、成本和 steps；说明 effort 行不是三个 checkpoint，并把 DataCurve 数字与本地 toy 结果分栏。

## 2026-10 DeepSeek V4 后训练：多教师全词表 OPD

配套正文：[第五册第十八章](book-05-llm-training/chapters/18-deepseek-v4领域专家与统一蒸馏.md)。

1. 给定 `p_S=(0.60,0.30,0.10)`、`p_T=(0.70,0.20,0.10)`，手算 `D_KL(p_S||p_T)`，列出三种 sampled log-ratio 及其学生采样概率，算出单样本估计的标准差；解释 full-vocabulary 求和消除哪类方差、仍保留哪类状态采样。
2. 设两个教师在同一前缀上给出 `p_A=(0.9,0.1)`、`p_B=(0.2,0.8)`。推导等权 reverse-KL 最优学生分布的归一化几何平均，与算术平均比较；改变两个教师权重，并说明什么任务路由错误会导致这种合并目标不合适。
3. 按 `token_positions × vocabulary_size × bytes_per_logit × teacher_count` 估计 BF16 logits 缓冲。对 8,192 个 token 位置、100,000 词表项和 10 个教师计算 GiB；再设计一个只保留 teacher hidden states、按教师索引加载输出头的调度流程，列出它省下与仍保留的成本。
4. 比较 Model Soups、固定教师答案的离线 SFT、GKD、MiniLLM 与 V4 报告中的 full-vocabulary multi-teacher OPD。分别说明它们合并的是参数、token 序列还是条件分布；为一项公平消融固定底模、学生 rollout、训练 token、教师前向预算、数据、评测集与随机种子。
5. 审核“V4 用 OPD 取代 mixed RL，所以 V4 后训练没有 RL”这句话。画出领域 SFT/GRPO 专家与后续 OPD 的顺序，再列出 GRM、verifier、rubric 和独立评估中仍未由公开报告说明的字段。

## 2026-09 DeepSeek V4 Flash SGLang：分支缓存与 kernel 证据审计

1. 写一个不依赖 SGLang/CUDA 的混合状态 cache toy：记录 Full/Compressed KV、SWA slots、共享 prefix node、分支点 state、eviction 与 restore。先模拟 chunked prefill 释放滑出窗口 slots，再让 sibling requests 在共享前缀分叉；比较“仅 Full KV 命中”与“Full KV + 对应 SWA branch state 均可复用”的判定，并注入缺失状态、过期 branch key 和不一致恢复点。
2. 为 PR #34565 的 workload 建立可复现实验 manifest，固定 `DeepSeek-V4-Flash-0731`、TP=2、FlashInfer MXFP4、DSpark、24,576 shared-prefix tokens、8,192 question tokens、128 output tokens、64 requests/8 groups、chunk size、缓存开关和测量区间。分别记录 token hit rate、cached tokens、mean/p95 TTFT、input throughput；标注这些数字是 PR 自报而非本地复现。
3. 设计三列 kernel 证据账本：B200/SM100-103 的 FP8/TP=1 CSA/HCA unit kernel、4× RTX PRO 6000/SM120 的 paged-MQA indexer/sparse prefill/FP4 MoE、端到端共享前缀 Agent workload。每列分别锁定 baseline、batch/context、精度、并行度、warmup、吞吐/延迟定义与质量指标；禁止把 `1.2x`、`1.45x` 和 `3.4x` 组合成同一个 V4 加速结论。
4. 写 serving release gate：固定 SGLang tag/commit、权重 revision、GPU/driver/kernel/backend，覆盖 cache-hit 正确性、SWA 分支恢复、稀疏 index recall、量化误差、长上下文 TTFT/TPOT、并发 p95/p99、工具执行与 verifier、单位成功成本。把源码存在、PR benchmark、本机复现和生产 acceptance 分列记录。
5. 为 SGLang PR #39313 设计 main / fusion-index / fused-kernel / fused-kernel-plus-fork 四臂实验，固定 DeepSeek-V4-Flash-0731、混合 FP8/MXFP4 权重与 scale layout、SM100 门禁、CUDA Graph capture 和 alternate-stream 依赖。分别测 kernel 数、stream overlap、prefill/decode 与端到端吞吐、greedy/sample quality 和逐请求 answer flips；注入不满足 gate 的配置验证 fallback。把发布方 4×B300 表与自己的复现分栏，不能迁移成 V4.1 成绩。

## 2026-09 GLM-5.1 长周期 Agent 与过程质量审计

构建一个不下载 GLM-5.1 权重、也不调用真实付费 API 的教学审计器，模拟 `glm-5.1` 的 `thinking.type`、78 层/256 routed/top-8/1 shared 配置账本、DSA indexer top-k、长周期 Agent loop、工具协议、context cache 和外部 verifier。输入模型配置、thinking mode、上下文历史、目标、实验结果、工具 schema、权限、预算、policy version、缓存命中和 artifact，输出目标保持率、策略改变率、错误恢复、index/evidence recall、tool-call replay、cached tokens、TTFT/TPOT、任务成功和单位成功成本。

实验至少覆盖：

1. 200K context 与 8 小时长任务的区别：注入历史增长、摘要/压缩、目标漂移、工具失败、环境变化和恢复点，比较目标保持与最终 artifact，而不是只比较最后文本；
2. `experiment -> analyze -> optimize` 闭环：让 Agent 运行基准、读取结果、识别瓶颈、改代码并再次运行，注入重复策略、错误指标、超时、部分执行和未知副作用；
3. multi-turn SFT/RL/process-quality 的证据边界：把发布方自报描述与本地 toy 的过程 verifier 分栏，不把 toy 结果写成 Z.ai 的训练 recipe；
4. DSA toy：对比 dense、indexer top-k 和带尾部保留的稀疏路径，分别输出 index recall、evidence recall、KV/indexer bytes、FLOPs 代理值和任务成功率；
5. `thinking.type=enabled/disabled` 与 `reasoning_effort` 的 capability probe，验证 GLM-5.1 不应接受从 GLM-5.2 迁移的 effort 选项；
6. Function Calling/MCP/cache：注入 schema 版本变化、权限拒绝、工具执行超时、重复 call、cache TTL 失效、租户隔离错误和 `cached_tokens` 账单差异；
7. 证据分层报告：Artificial Analysis 第三方字段、Z.ai 发布方 SWE-Bench/KernelBench/长任务数字、官方 config、DataCurve 精确行缺失和本地 toy 结果分别列账，禁止迁移其他 GLM 版本的 Agent 分数。

教学实现不能宣称复现 Z.ai 的 8 小时、655 次迭代、6.9× 吞吐或 3.6× KernelBench 结果；所有 toy 数字必须绑定代码版本、输入、工具、环境、verifier 和停止条件。

## 2026-09 Grok 4.20 Multi-agent、Compaction 与工具审计

构建一个不调用真实付费 API 的教学审计器，模拟 `grok-4.20-0309-reasoning`、`grok-4.20-multi-agent-0309` 的服务 manifest、4/16 Agent 协作、leader 汇总、opaque encrypted state、context compaction、prompt caching、server/client tools 和 Remote MCP allowlist。报告必须把 Artificial Analysis 的 `grok-4-20`、xAI 官方模型页和 DataCurve 精确行缺失分栏，不能把 AA 的 2M 与官方 1M 合并成无条件能力，也不能迁移 Grok 4.5/4.6 的 Agent 评测。

实验至少覆盖：

1. 建立 source-aware capability manifest，分别记录 `model_id`、alias、snapshot、endpoint、区域、context source、1M/2M 字段、限流、价格、工具和核验时间；用实际长度探测验证服务上限，并把探测结果与目录字段分开。
2. 对同一研究任务做 4-agent/16-agent 消融，固定来源、工具、预算、prompt 和 verifier，记录子任务覆盖、证据/引用支持率、重复检索、错误传播、leader 汇总、TTFT/TPOT、p95 和单位成功成本；加入单 Agent baseline，验证多 Agent 是否真的带来增益。
3. 实现 leader/sub-agent trace schema：保存 role、任务分片、prompt revision、tool call/result、source、call ID、失败/重试、汇总依据和 verifier；分别测试明文中间状态、opaque encrypted state 缺失、乱序和重复回放。
4. 实现 compaction replay toy：将历史压成单个 opaque `compaction` item，注入手工裁剪、重排、重复插入、过期状态和 context 已超限场景，检查恢复成功率、信息损失、重复工具调用、压缩成本和 cache 命中；禁止解析或伪造内部 blob。
5. 实现混合工具状态机：server-side web/X/code/collections tool 自动执行，client-side function call 进入宿主授权、幂等执行、结果回灌和 artifact verifier；注入超时、未知副作用、重复 call、工具 schema 版本变化和 `max_turns` 跨请求重置。
6. 实现 Remote MCP 最小权限实验：比较全量工具 schema 与 `allowed_tools` allowlist 的上下文 token、工具选择、误调用、prompt-injection 和权限拒绝；验证 allowlist 不能替代租户、资源、参数和高风险动作授权。
7. 写最终证据报告：AA 指数/速度/价格、页面 snapshot 日期、指标 workload 与 benchmark freshness、xAI API 字段、DataCurve 行缺失、本地 toy 结果和待核验的架构/训练/kernel 分开列账。模拟 AA 标注“deprecated；仅默认 10K input workload 继续更新，其他 workload 为历史数据”的情况，不得把页面抓取时间冒充每项指标的测试时间，也不得声称复现 xAI 的生产多 Agent 性能、参数规模或 Grok 4.20 专属论文结论。

8. 加入 multi-agent API contract gate：拒绝 Chat Completions、`max_tokens` 和 client-side/custom function tools；允许 Responses `previous_response_id`、内置工具与 Remote MCP。模拟 beta contract 变更、encrypted sub-agent state 未携带、token/tool usage 只记录 leader 的错误账本；最终报告应汇总所有 Agent 的 input/output/reasoning tokens、server-side tool calls、continuation success、wall-clock 与单位成功成本，不调用真实 API。

## 2026-09 Gemini 3.8 Flash Thinking、Interactions 与 Computer Use 审计

配套正式章节：[Gemini Interactions：跨轮状态、签名回放与工具责任链](book-20-agent-harness-runtime/chapters/23-gemini-interactions与thought-state回放.md)。

构建一个不调用真实付费 API 的教学审计器，模拟 `gemini-3.8-flash` 的 `low/medium/high` thinking、共同 output budget、thought summary/signature、Interactions state、Search/URL/File/Code/function 工具、Computer Use、structured output、1M context、implicit caching 和外部 verifier。报告必须把 AA/DataCurve、Google 官方文档和本地 toy 分栏。

实验至少覆盖：

1. thinking 消融：固定任务、模型 snapshot、工具和 verifier，对比 low/medium/high 的思考预算、可见输出、截断、TTFT/TPOT、工具轮数、成功率、总 token 和单位成功成本；验证 `minimal` 对 Gemini 3.8 是非法能力，而不是默认低档位。
2. stateful/stateless replay：模拟 thought block、summary、opaque signature、`previous_interaction_id` 和手工回放；注入 signature 丢失、summary 缺失、乱序、重复和跨模型回放，检查状态连续性与安全失败。
3. SSE trace：记录 `interaction.created`、step start/delta/stop、tool call/result 和 completed 事件，验证事件重连、重复消费、工具超时、取消和最终 artifact 的幂等。
4. 工具组合对照：比较 Search grounding、URL Context、File Search、Code Execution、function calling 的检索/执行/结果回灌链路，记录 citation coverage、chunk recall、执行错误、重试和业务校验失败；明确工具不等于模型权重或永久记忆。
5. Computer Use 安全：用截图和归一化 `1000x1000` 坐标模拟 action intent，注入 UI 漂移、遮挡、敏感操作、域名越权、重复点击和状态竞争，分别评估模型提案、宿主审批、执行回执和 verifier。
6. Agentic Video 对照：固定视频 revision、问题位置和模型配置，比较默认 static 1 FPS 与按需 agentic transcript/frame/audio 读取；用合成 Interaction steps 覆盖 `processing_call.id` → `processing_result.call_id` 的配对、断线重放、超时/取消和媒体权限拒绝。记录关键事件召回、时间定位、processing 次数、thought/tool-use tokens、TTFT、端到端成功和单位成功成本；不要把文档中的最高 88% token 减少/约 7% 质量提升当成本地复现结果。
6. 长上下文与缓存：比较 128K/512K/1M 输入、needle 位置、文档数量、重复前缀和 implicit-cache hit/miss，分开记录有效召回、TTFT、计费 token、KV/cache 代理和后续轮 TPOT，不能把服务侧缓存写成 GPU KV cache。
7. 评测 manifest：重现 DataCurve high 的 `n_runs=4`、`n_attempted=447`、Pass@1/4、平均成本和 steps 字段；所有本地数字绑定任务、工具、环境、verifier 和代码版本，不宣称复现 Google 或 DataCurve 生产结果。
8. 直接运行 [`gemini_interactions_replay_demo.py`](research/model-update-2026-09/code/gemini_interactions_replay_demo.py)，检查 stateful continuation 只继承 history、stateless replay 保留 signature、`function_call.id` → `function_result.call_id` 对齐、SSE completed/done 顺序、store/delete 和 `55/1` retention toy 结果；记录 `network_called=false`。
9. 为同一份 Interaction history 生成两个 manifest：一个把 `tools`/`generation_config` 错误地当作跨 turn 继承，另一个按 interaction-scoped 重新指定；比较 schema、tool choice、thinking level 和 verifier 结果，说明为什么 `previous_interaction_id` 不是完整 runtime snapshot。
10. 建立 signature 兼容性对照：比较 Thinking prose、Tool combination prose、当前 Interactions API reference 与固定 SDK schema。记录 Tool combination 的 `function_response.id` 与正式 reference/SDK 的 `function_result.call_id` 命名差异；记录 custom function signature 在 prose 与 typed schema 的差异，以及 `ThoughtStep.signature` prose 必需、schema optional 的差异。让 custom function signatures 缺省/存在两种 history 都能由本地 toy 保真 round-trip；对照 `extra="allow"`、`UnknownStep.raw`。结论必须区分 schema 声明和 endpoint 行为：不伪造、不静默删除实际 opaque fields；没有授权/API key 时不声称完成 capability probe。
11. 审计固定 SDK `6d012889752f65c1a51d0ad6e5970fc97d19c4ca` 的 Interactions 类型路径：区分 `_gaos/types/interactions/*` schema 与普通 `_common.BaseModel`，核对 `_gaos` 的 `extra="allow"`、thought optional signature、custom function 未声明 signature 和 lenient `UnknownStep.raw`。扩展 demo 测试：schema profile 允许省略 thought signature，而默认宿主 replay gate 仍可以拒绝该 history；明确这是两种不同证据层，不声称 endpoint 真实拒绝。

## 2026-09 DeepSeek V4 Pro reference implementation 审计练习

1. 读取固定 revision 的 `config.json` 与 `inference/config.json`，生成一份 manifest；把模型配置、inference 配置、encoding revision、kernel 文件哈希、权重 metadata 和“未下载权重”状态分别列出。
2. 实现 CPU 教学版 `compress -> causal score -> top-k -> local window` 流程，分别输出 compression error、candidate recall、evidence recall、局部窗口命中率和 KV bytes 代理值；禁止用单个 sparsity 数字替代这些指标。
3. 用 toy router 比较前三层 token-id hash routing 与后续 `sqrtsoftplus` score routing；加入 selection bias，验证 bias 改变选择但不改变 routing weight 的假设，并记录 hash collision、expert load 和 dispatch bytes。
4. 对 FP4/FP8 block quantization 写无 CUDA 的误差实验，比较 block size `[128,128]`、scale dtype、FP4 packed weight 和 BF16 reference；明确 toy 不等于 TileLang kernel benchmark。
5. 为 DSML parser 构造合法调用、string/JSON 参数、缺失 EOS、未知参数、重复工具调用和 `<think>` 保留测试；将 parser、schema、权限、executor receipt、幂等重试和 artifact verifier 分层。
6. 构造 `low/high/max` 的固定任务 manifest，记录 visible output、tool calls、retry、TTFT/TPOT、cost 和 verifier result；不要把 max 当作新 checkpoint，也不要把 DataCurve 的 `62.831858%/88.495575%` 当作 toy 或裸模型分数。

## 2026-09 DeepSeek V4.1-Flash `deepseek-recipe` 协议验收

1. 固定 `deepseek-recipe` commit `8cadfede7063c896b944e7bae05daa3549ae97ea`，把协议请求转换成 `ConversationRequest`，记录 inference/parsing/model/stream 字段；不要把转换成功写成模型已推理。
2. 把 DSML、`<think>`、JSON fence 和 stop sequence 分别拆成跨 chunk 的输入，验证 state machine 不提前输出不完整 marker；再注入流尾截断、重复结束标签和未知工具参数。
3. 比较“渲染 prompt 后附 tokenizer”和“直接把 token-ID chunk 交给无 tokenizer processor”两条路径，输出 token-level golden trace 与明确错误；记录 tokenizer revision，不使用默认 special-token 注入。
4. 对 URL、data URL 和 in-memory bytes 做图像 quota 实验，覆盖 600 张、32 MiB 单图、64 MiB 总量、8 并发、重定向、超时和 SSRF/private-address；把 parser、fetcher、preprocessor、权限和 verifier 分栏。
5. 用 mock inference 连接 `server-py`/`server-rs` 示例，证明它只验证协议事件接线；单独列出完整权重、真实 backend、HTTP 鉴权、工具副作用、CUDA/TileLang 和线上 SLO 的未验收门禁。

## 2026-09 Kimi K3 vLLM upstream/backend capability matrix

1. 把 vLLM stable supported-models、stable K3 API、main `registry.py`、K3 package `__init__.py`、recipe YAML 和 FlashKDA Atom 写入一张 capability matrix，列出 source branch、更新时间、revision/hash、平台分支、依赖、状态和证据层级。
2. 对每一行填写 `observed / inferred / unverified`：页面类名只能填 `observed` 的文档入口；PyPI `vllm 0.29.0` 与 v0.29.0 tag 可填 `observed` 的 stable artifact/source entry；recipe 的 `Pre-release` 必须保持为预发布优化路径，不能推导目标硬件 runtime acceptance。
3. 为 NVIDIA 与 ROCm 各画一条从 wheel -> model loader -> KDA/MLA backend -> hybrid cache -> TP/TEP/DEP/PP -> tool parser 的路径；为 TPU 路径标注“不会主动加载 GPU implementation”，再列出仍需实测的门禁。
4. 用 toy manifest 注入 stable API 与 main registry 版本不一致、K3 recipe 依赖 nightly、revision 错配、完整权重未加载、MLA cache 丢失、KDA recurrent state 丢失、prefix-match unit 改变和 tool-call parser schema 改版，输出拒绝原因。
5. 把 FlashKDA 的 H20/GB200 局部 kernel benchmark 与端到端 K3 TTFT/TPOT、MoE 通信、cache recovery、tool-call acceptance 分成不同指标；报告哪些数字可以比较，哪些必须绑定硬件、backend、warmup/iters/repeats 和 verifier。
6. 最后写一段面试回答：分别解释“stable docs/API 已列出”“vLLM 0.29.0 stable source 有 K3 entry”“存在 nightly 优化 recipe”“stable wheel 在目标硬件生产 serving 已验收”四句话的证据差异。

## 2026-09 GPT-5.6 Luna 服务档位与状态账本

1. 写一份 source-aware manifest，分别记录 `gpt-5.6-luna`、AA `max`、DataCurve `mini_swe_agent_gpt_5_6_luna_max`、snapshot、mode、effort、provider、harness、task set、tools、timeout、retry 和 verifier；禁止把两个排行榜的分数合成一个裸模型分数。
2. 实现一个零依赖成本计算器，分别输入 1.05M total context、922K maximum input、128K maximum output、272K whole-request threshold、reasoning tokens、uncached/cached input、cache write、output 和 tool cost；覆盖阈值前后、缓存命中、compaction 后首次请求和 incomplete output。
3. 模拟 Responses 多工具回放：构造 user -> opaque reasoning item -> function call -> function output -> reasoning continuation -> final output，分别删除 reasoning item、tool result、call id 和 compaction item，输出 replay rejection、重复副作用和恢复结果。
4. 设计 deferred tool search/Programmatic Tool Calling 实验，比较全量 schema、延迟加载和批量编排的 prompt token、选择错误、往返次数、权限拒绝、超时、幂等和 verifier 通过率；不能把 schema 省略写成权限放开。
5. 写一份网络故障证据报告：AA/DataCurve 当前快照可写入“新鲜榜单证据”，OpenAI 403/超时/DNS 写入“访问边界”，旧官方页面写入“历史快照”；禁止把任一线路结果升级成模型不存在或官方接口改变。
6. 运行 [`gpt56_luna_state_replay_audit.py`](research/model-update-2026-09/code/gpt56_luna_state_replay_audit.py)，解释为什么 `current_turn/all_turns`、compaction canonical window、cache prefix、tool-search `call_id` 和 artifact verifier 必须分别验收；再人为改变 function output 或重复 idempotency key，记录拒绝原因。报告只标为 `local_protocol_toy`。

## 2026-09 Claude Sonnet 5：System Card 与长任务回放实验

1. 写一个 source-aware manifest，分别记录 `claude-sonnet-5`、AA `max`、DataCurve 五档 effort 和 Anthropic System Card benchmark；加入 provider、harness、tools、safeguards、trials、task set、verifier、compaction trigger 和快照日期，禁止合并三类分数。
2. 用零依赖代码模拟 adaptive thinking：让 `effort` 影响升级/工具策略，让 `max_tokens` 作为硬上限，再加入 task budget；覆盖“effort 高但没有新证据”“工具状态 unknown”“verifier 连续失败”和安全停止。
3. 构造含 thinking block/signature、tool call/result、compaction item 和未完成副作用的多轮 trace。分别删除每一类状态，输出 replay rejection、重复副作用风险和可恢复状态；最终检查 artifact verifier 而不是只比最终文本。
4. 设计 System Card 安全评测复现表：区分 Claude Code 恶意请求、computer-use 恶意任务、Gray Swan IPI 场景和 cyber benchmark，分别记录 safeguards、工具、权限、任务环境、拒答/执行/误拒答和 verifier；不要把拒答率当作裸模型安全分数。

## 2026-09-22 Grok 4.7：encrypted state 与长轨迹 Agent 评测

1. 为 `grok-4.7` 建立 source-aware manifest，记录 AA `xhigh` 条目、DataCurve 精确行缺失、xAI model/news 文档、provider、effort、工具、compaction 版本和 arXiv 负检索；禁止迁移 Grok 4.6 Agent 分数。
2. 用零依赖 toy harness 模拟 Responses trace：插入 encrypted reasoning item、tool call/result、权限决定、未完成副作用和 opaque compaction item；测试删除、重排、重复回放时的拒绝与幂等行为。
3. 固定同一代码任务和环境，比较 low/medium/high/xhigh；同时输出任务成功、artifact verifier、工具轮数、reasoning/output token、墙钟、成本、失败恢复和副作用，而不是只比较最终文本或 token 数。
4. 实现 `allowed_tools` 与宿主权限策略的消融：比较全量 schema、最小工具集合、恶意/越权工具请求，分别记录上下文 token、工具选择错误、拒绝、实际执行和 verifier 结果。
5. 将 xAI 发布方 DeepSWE/Terminal-Bench/安全数字、AA provider 字段和本地 toy 结果放入三份账本，写出不能互相校准的原因。
6. 模拟 `reasoning.encrypted_content`、服务端工具加密输出、`response.reasoning_text.delta`/`response.reasoning_summary_text.delta` 和最终文本并行到达的 Responses stream；检查 summary 可观察但 encrypted state 只能原样回放。
7. 构造 `store=true/false` 与 `previous_response_id` 的状态矩阵，比较服务端状态引用、客户端完整 input replay、删除 response 后的恢复失败和成本账本；不得把 opaque state 当作永久应用 memory。
8. 为 Remote MCP 实现 Responses/原生 SDK capability adapter，覆盖 `allowed_tools` ↔ `allowed_tool_names`、`headers` ↔ `extra_headers`、Streaming HTTP/SSE、未设置 allowlist 时全量 schema 注入，以及 unsupported `require_approval`/`connector_id` 的显式拒绝。

## Qwen3-Omni：多模态时间与流式语音验收

1. 写一个零依赖时间轴实验：把 audio timestamp、video timestamp 和 image spatial coordinate 分开编码，模拟 80 ms temporal bin 的 jitter、丢帧、重复帧和跨 chunk replay。
2. 设计 Thinker-Talker trace manifest，至少保存 model/variant、媒体时间戳、Thinker chunk、RAG/tool/policy/verifier 事件、Talker 首码本/残差码本、Code2Wav chunk、audio packet sequence 和取消状态。
3. 用合成数据比较“只用 Thinker 文本给 Talker”和“使用多模态条件给 Talker”两条 toy pipeline；分别报告工具介入后文本/语音状态是否可恢复，不把 toy 结果写成模型能力。
4. 模拟首码本 AR + residual-codebook MTP，分开测串行步数、code rate、packet 首包和 waveform 拼接错误；再加上网络延迟，说明论文首包数字为什么不能直接当 SLO。
5. 设计四级部署门禁：Thinker inference、Talker code generation、Code2Wav waveform、目标硬件/streaming/tool/verifier acceptance；每级记录失败、重试、取消和未知外部状态。

## 2026-09 K2 Horizon 3.7B Dense 对照与迁移验收

1. 根据当前 `K2HorizonForCausalLM` 配置，画出 36 层 dense decoder 的 Q/K/V、32Q/8KV GQA、RoPE、FFN 和 KV cache 路径；再画出 36B/A4B 后 45 层的 MoVA value top-4、FFN top-8、shared expert 和 EP dispatch，列出新增的 router/通信/workspace 成本。
2. 用 `22.9T/8K -> 1.1T/32K -> 498B/128K -> 110B/512K -> 199B/512K -> 199B/512K SFT -> 50B/512K SFT` 做阶段账本，说明哪些是增量 token、哪些 checkpoint 继承，为什么不能把总和称为一份训练数据集。
3. 为 Math、Code、STEM-Code RL 分支设计 ISO merge/RAM merge 的 toy manifest；对比训练分支合并、0.9B MOPD 和 forward-time MoE/MoVA routing 的优化对象、checkpoint 形态和部署影响。
4. 解析 `/tmp/k2-migration-fixed-20260922.out`，验证 source/target model type、target architecture、copy、`weights_reencoded=false`、dtype、shard/tensor 数；增加 tokenizer/chat template/parser/revision 不一致时的拒绝门禁。
5. 对 vLLM H200 recipe 与 SGLang PR #37654 设计 source-aware benchmark 表，分别记录 TTFT、TPOT、吞吐、GSM8K、硬件、输入输出长度和 parser；明确发布方数字、本地 toy 和真实 full-weight profiling 不能互换。

## 2026-09 Qwen3-VL：视觉位置、DeepStack 与 GUI Agent 验收

1. 实现一个零依赖 temporal/height/width position toy：输入视频 frame grid 和 timestamp，分别输出普通序列位置、三轴 M-RoPE 位置和 `<seconds>` 文本锚点；用不同采样率、丢帧、重复帧和跨 chunk replay 检查时间 drift。
2. 模拟 DeepStack：构造三层视觉特征、三组 projection 和早期语言 hidden states，比较 residual 注入与只使用最后视觉层的维度、激活和伪质量账本；明确“不增加视觉 token”不等于“零额外计算”。
3. 为 S0-S3 建立 training manifest，记录 67B/1T/1T/100B tokens、8K/8K/32K/262K context、merger/full-parameter/long-context/ultra-long 目标、数据源归一化和 checkpoint 继承；区分公开阶段和未公开完整 recipe。
4. 构造 Thinking with Images reward toy：分别计算 answer accuracy、multi-turn reasoning 和 tool-call reward，加入“所有任务只调用一次工具”的 shortcut，验证第三类 reward 如何发现策略问题。
5. 设计 GUI Agent replay：固定 screenshot hash、窗口尺寸、设备像素比、坐标、动作序号、权限决定、executor receipt、重试/幂等和独立 verifier；测试页面 revision 变化后拒绝旧坐标，而不是静默点击。
6. 设计 Qwen3-VL serving gate：processor、视觉数值正确性、full-weight load、M-RoPE/DeepStack/timestamp replay、TP/EP、KV/cache、目标硬件 profile、GUI/tool acceptance；分别报告 AA 字段、论文自报、raw main 和本地实测。

## 2026-09 Qwen3.7 Plus：交互式混合 Agent 与区域合同验收

1. 读取 [`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)，为 `qwen3.7-plus-2026-05-26` 写一份 capability manifest；至少记录 region、scope、endpoint、input/output modality、context/input/output limit、thinking limit、Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching、Batch 和 Fine-tuning。
2. 实现零依赖 GUI replay toy：输入 screenshot hash、窗口尺寸、设备像素比、页面 revision 和模型 action proposal；页面 revision 改变时拒绝旧坐标，检查 permission、idempotency、timeout、retry，并输出 observation 和 verifier 结果。
3. 设计视觉预算实验：固定文本和任务，改变图片 resize、视频采样率、帧数、工具历史和 thinking/output reservation；报告总预算、关键帧覆盖、未覆盖时间区间、TTFT、成本和任务正确率，不能只报告 1M context。
4. 设计 region/scope capability 对照：比较北京、Virginia Global 和 Virginia US 的 Structured Outputs/Web Search/Batch 行为；把 endpoint、API key region、scope、schema validation 和真实 executor 分开，模拟 401、403、unsupported capability 和 timeout。
5. 设计 cache/prefix replay：改变 tool schema hash、媒体 revision、region、compaction 和 prefix breakpoint，记录 cache create/read/miss；明确 Context Caching、GPU KV cache 和应用 memory 的不同失效条件。
6. 写一份 source-aware 评测报告：AA 指标、Alibaba 产品/API 字段、本地 GUI harness 和 verifier 分栏；说明 DataCurve 没有精确 Qwen3.7 Plus 行，禁止迁移其他 Qwen 的 Agent 成绩。

## 2026-09 GLM-5.3 标准 DSA runtime 验收

1. 读取 `GLM-5.3` 的固定 config，画出 78 层、前三层 dense、256 routed/top-8/1 shared、21 个 Full indexer 和 57 个 Shared indexer 的结构账本；明确哪些字段不是完整参数或训练 recipe。
2. 实现一个零依赖 Full/Shared DSA toy：Full 层计算 causal top-k，Shared 层复用前一层索引；分别报告 index recall、evidence recall、显存/排序代理成本和任务成功率，并加入候选漏检案例。
3. 对照 Transformers、vLLM `v0.29.0`/main 和 SGLang `v0.5.20`/main，制作 source-aware capability matrix；把 stable entry、mutable main、完整权重、本地数值、目标硬件和生产 SLO 编成不同状态。
4. 设计 cache recovery 测试：分别打乱 MLA latent cache、interleaved RoPE offset、Full/Shared top-k、block table、MTP iteration 和 batch reorder，检查服务是拒绝、回退还是产生错误结果。
5. 写一份标准版/Flash 版边界报告：把标准 `glm_moe_dsa` 与 Flash `glm5_next` 的 attention、视觉、state pool、MTP 和 EPD 逐项分栏，禁止把 Flash 的 KDA 或视觉证据迁移给标准版。

## 2026-09 Claude Opus 5.5：token efficiency 与 fallback 审计

1. 为同一组 coding、research 和 computer-use toy tasks 构造 low/medium/high/xhigh/max effort manifest，固定 provider、工具、环境、timeout、retry 和 verifier，输出成功率、output/reasoning token、tool calls、steps、TTFT、单位成功成本和 P95。
2. 实现一个零依赖 `FallbackLedger`，输入 primary model、risk category、policy、fallback target、permissions、cache state、retry 和 artifact verifier；要求区分拒答、路由、工具失败、未知外部状态和最终通过，禁止将 fallback 结果归因给 primary。
3. 设计长任务 coding replay：模型先生成 context snapshot，再提交 patch、运行测试、读取失败、修补并由 verifier 验收；对比“少工具调用但少测试”和“更多调用但通过验收”两种反例，计算 `verified_success / total_cost`。
4. 解析 Anthropic 发布方 benchmark 的 effort、工具、任务、safeguard 和成本条件，与 AA `max with fallback` 目录字段分账；DataCurve 没有精确 Opus 5.5 行时输出 `not_applicable`，不得借用 Opus 5。
5. 写一个不调用真实 API 的 `Opus55CompatibilityHarness`：对 `thinking` disabled/manual budget、`tool_choice` any/tool、旧 computer tool 和错误 block 顺序生成请求，按官方契约输出预期 400、迁移建议和 response parser 测试结果。
6. 实现 thinking-binding/compaction replay：记录 `model_id`、prefix hash、tools hash、thinking block、signed compaction block、inline tool schema 和 cache key，分别模拟 append-only、prefix mismatch、drop block 和 tool upgrade，检查状态是否被错误归因于模型质量。
7. 给同一 toy task 跑 standard/fast 两个 serving profile，记录 TTFT、output tokens/s、`usage.speed`、429/529、input/output/cache/tool cost 和 verified artifact，画出速度-成本曲线并说明 fast mode 不是新权重。
8. 根据当前 Opus 5.5 model page 实现输出预算与缓存成本计算器：输入同步 Messages 与 Batch endpoint、是否带 `output-300k-2026-03-24` beta header、输入/输出 token、cacheable prompt 长度、cache read/write TTL；拒绝把 Batch 300K 当作同步上限，校验 512-token cache minimum，并分别计算 `$4/$20` 基础输入/输出、`$0.20` cache read、5m `$5`/1h `$8` cache write 每 MTok。结果标为服务合同演算，不得据此推断模型内部推理 token 或真实账单。

8. 运行 [`claude_opus55_protocol_audit.py`](research/model-update-2026-09/code/claude_opus55_protocol_audit.py)，解释为什么请求 capability error、thinking state invalidation、tool execution、fallback 和 verifier failure 必须分开计数；将 toy 输出与 Anthropic 官方 API contract 分栏，不把 `local_protocol_toy` 当作真实 API 验收。

9. 为 Opus 5.5 实现 on-demand/threshold compaction compatibility matrix：分别校验 beta header、请求字段、平台可用性、签名 block 续接、服务端旧前缀裁剪和混用时的拒绝路径；加入 Bedrock 仅 threshold 列 beta 的负例。
10. 构造相同长任务的 keep-tail replay：对照 on-demand 条件满足、on-demand tail 被编辑、threshold tail 含旧 `thinking`、threshold tail 已剥除 reasoning、`drop_block` 五种情况；检查摘要、普通 text/tool blocks、reasoning binding 和 side-effect lineage，不以“请求成功”推断原 reasoning 被接受。
11. 为 2026-08-31 00:00 UTC 前后创建的账号分别模拟 prefix mismatch：区分新账号默认 `error`、显式 `drop_block` 与旧账号 opt-in；同时加入目标模型无法读取 producer block 的 model-binding mismatch，确保两种不兼容原因分开记账。输出标为文档驱动的 local toy，不调用真实 Messages API。

## 2026-09 GPT-6 Sol：预算、状态与工具 runtime 验收

1. 为 `gpt-6-sol` 建立 source-aware manifest，记录 AA `max` 配置、DataCurve 精确行缺失、model page、Reasoning、Agents、Tools、Compaction 页面和快照哈希；禁止迁移 GPT-6 Astra/GPT-5.6 的 Agent 分数。
2. 用零依赖代码计算 `1,050,000` context、`922,000` maximum input、`128,000` maximum output 的预算，加入 reasoning tokens、工具 schema、工具结果、历史 output 和可见文本；覆盖 `incomplete` 与 25K reasoning/output reserve。
3. 模拟 `standard/pro` mode 与 `none`--`max` effort 的正交矩阵，插入合法 `configuration_update`、相邻 update、自动 compaction 和 automatic truncation，输出接受/拒绝原因；另验证 update 只改变 effective effort 并持续到覆盖、response 中的 `reasoning.effort` 仍是 request-level 值、稳定 prompt prefix digest 不变。将 prefix 保留标记为 cache-friendly 设计，不得声称 toy 验证了真实 cache hit。
4. 构造 Responses item replay：保留 reasoning summary、encrypted/opaque item、tool call/result、permission、executor receipt、compaction item 和 artifact verifier；分别删除、重排或重复 item，检查恢复、重复副作用和 lineage。
5. 设计 Agents API、Agents SDK、Responses API ownership 对照，分别记录谁保存 session/history、谁运行 loop、谁做 approval、谁执行工具、谁负责 sandbox 和 verifier。
6. 模拟 tool search 的 namespace、`defer_loading`、schema hash、search call、loaded tool、permission deny 和 cache prefix；比较全量 schema 与按需加载的 token、延迟、错误和复用。
7. 实现 272K whole-request threshold、input/cache 2x、output 1.5x、Batch/Flex 50%、Fast mode 2x 的成本账本，再加入 tool、retry、compaction 和 verifier 失败成本。
8. 运行 [`gpt6_sol_contract_audit.py`](research/model-update-2026-09/code/gpt6_sol_contract_audit.py)，解释为什么 mode/effort、configuration update、automatic compaction/truncation、standalone compact 和 explicit compaction 必须分支；记录每个拒绝原因。
9. 将 toy 的 permission/executor/verifier 和 idempotency receipt 映射到一次真实 coding-agent trace，分别统计 proposal、authorized、executed、verified、duplicate 和 incomplete，禁止用最终文本替代 artifact 验收。
10. 用 272,000 与 272,001、compaction 前后 prefix、1,024 与 1,023 cache tokens 做边界测试；说明为什么这些边界测试不等于真实 provider billing、cache hit rate 或 GPT-6 Sol 质量。
11. 为 GPT-6 Sol/Luna 增加独立 residency capability gate：验证 EU data residency 仅明确覆盖 Standard processing 的 Responses/Chat Completions；将 `processing_mode` 与 `reasoning.mode` 分栏，分别对 regional storage、regional processing、Fast/Batch/Flex、system data、Remote MCP、retention controls 和 10% regional uplift 输出 eligible / not_established / external_policy，并写明这不是法律合规保证。

## 2026-09 GPT-6 Luna：sibling routing 与服务合同验收

1. 为 `gpt-6-luna` 建立 source-aware manifest：记录 AA canonical slug、max 配置、详情页哈希、DataCurve 精确行缺失、OpenAI model ID、snapshot、effort、mode、provider 和 endpoint；将 Luna 与 Sol 分开。
2. 实现 task-shape router：在 focused/high-volume、长推理、工具调用和高失败代价任务之间切换 Luna/Sol 的候选配置；输出选择依据、可观测 capability、`not_applicable` 评测字段和回滚条件，禁止用“高效率”猜测参数或架构。
3. 复算 Luna 的 1.05M/922K/128K budget、reasoning/output/tool/schema/compaction token 账本和 272K whole-request threshold；分别计算 standard、Batch/Flex、Fast mode、retry、tool 和 verifier 失败后的单位成功成本。
4. 构造 Responses/Agents runtime replay：保留 `configuration_update`、typed items、tool call/result、permission、executor receipt、opaque compaction item 和 verifier；测试 model ID 替换、跨 sibling 迁移和缺失 DataCurve 行时的拒绝策略。另分别验证自动 compaction/truncation 与 standalone `/responses/compact` 对 update 历史的拒绝，以及显式 `compaction_trigger` 后、下一条用户消息前重插 update 的可恢复路径。
5. 扩展 capability manifest 加入 `endpoint`、`region`、`processing_mode`、data-residency eligibility、retention control 与第三方 MCP policy。验证 GPT-6 Sol/Luna 的 EU residency 只对 Standard processing 的 Responses/Chat Completions 有官方支持；把它与 reasoning `standard/pro` 模式区分。注入 Fast/Flex、Batch、未获 retention approval、system metadata 和远端 MCP 请求，分别输出 eligible / not_established / external-policy 状态，并把适用地域溢价纳入单位成功成本。实验不得声称模型 API 或本地 toy 自动提供法律合规保证。

## 2026-09 DeepSeek V4.1-Flash：API contract 与媒体/工具状态验收

1. 建立 `deepseek-flash` capability manifest：记录 canonical model、requested/served model、API snapshot、1M/384K/2500 concurrency、价格/限流/错误版本；alias 变化时必须拒绝静默合并。
2. 实现零依赖 semantic SSE parser：处理递增 `sequence_number`、跨 chunk 的 response item、tool call/result、图像 `function_call_output`，并验证 completed/incomplete/failed 终态和无 `[DONE]` 的 EOF 行为。
3. 实现 Vision/Files 预算 toy：覆盖 URL 8192 字符、60 秒、32 MiB、`file_id` 64 MiB、48 MiB request、600 图、25 GiB/10,000 文件和到期/删除；输出实际媒体证据覆盖率，不把接受请求当作模型质量。
4. 对比 `/beta strict=true`、普通 schema 和 host-side verifier：注入缺字段、额外字段、权限越界、重复 `call_id`、超时重试和业务错误，分别报告 schema valid、authorized、executed、verified。
5. 把 Responses stateless 与本地 Agent state 分开回放：删除 `previous_response_id`、重排 item、重复 tool result、丢失文件和 alias 路由变化，检查 parser 是否拒绝、重试、回退或产生重复副作用。

## 2026-09 DeepSeek V4.1-Flash：Harness provider、MCP 与 webhook 审计

1. 运行 [`deepseek_harness_protocol_audit.py`](research/model-update-2026-09/code/deepseek_harness_protocol_audit.py)，核对 provider credential 脱敏、稳定 provider ID、session model 固化、插件 cleanup、MCP reconnect budget 和 webhook `202`；将输出证据等级标为 `local_protocol_toy`。
2. 为 provider/session 建立 recovery manifest，字段至少包含 `provider_id`、requested/served model、session/workspace revision、credential reference、protocol claim、tool schema hash 和实际 capability probe；故意修改 provider ID 或模型后拒绝静默迁移已有 session。
3. 模拟 MCP server 断线、工具 schema 变化和重连预算耗尽，分别输出 reconnect、rediscovery、namespace、registry unregister、permission recheck 和人工恢复状态；禁止复用断线前的工具集合。
4. 模拟 GitHub review webhook 的签名校验、重复 delivery、异步 admission、workspace/session 创建、出站权限和最终 artifact verifier；比较 HTTP `202`、executor receipt、测试通过和 verifier 通过四个状态。
5. 把 `model_intent -> admitted -> authorized -> executed -> verified` 做成 trace 状态机，注入错误 signature、缺少幂等键、重复 call、未清理 plugin effect 和越权出站请求，输出 failed gates 和 root causes。

## 2026-09 Kimi K3：当前榜单复验与评测边界

1. 写一张 K3 双榜证据表：记录 AA 详情页的 `max`、Intelligence Index、速度、成本、1M context，以及 DataCurve `mini_swe_agent_kimi_k3_max` 的 `309/451`、Pass@1/Pass@4、token、steps、runs/tasks；每个字段标明 provider/configuration 或 harness/system 来源。
2. 设计同条件复现实验，固定 hardware、backend、序列长度、warmup/iters、harness、task set、tool version、effort 和 verifier，再判断发布方“约 `2.5x scaling efficiency`”是否能支持独立结论；缺字段时输出 `not_comparable`。
3. 构造 preserved thinking history 的 replay manifest，至少保存 channel、tool/index、工具 schema、reasoning effort、工具结果、权限决定、KDA/MLA 状态引用和 revision；删除任意一项后给出拒绝或降级原因。
4. 模拟其他模型会话中途切换到 K3，分别注入 schema 版本变化、thinking history 缺失、MLA cache/KDA state 不匹配和 tool-call parser 漂移；要求系统拒绝静默继续，并生成可审计的迁移结果。
5. 为 excessive proactiveness 设计四道门：schema、permission、executor、verifier。用一个合法但超范围的工具提案测试“可解析”与“可执行/可接受”的差异，并记录幂等重试和副作用账本。
6. 对照 Kimi Code、Claude Code、Codex 与 H20/H100/compaction 条件，编写 benchmark comparability checklist；禁止把不同 harness 的分数合成单一排名。

## 2026-09 GLM-5.3：SAO 与长轨迹 RL toy 验收

1. 构造一个无网络、无模型权重的异步 rollout queue：为同一 prompt 生成长度不同的 action/observation 轨迹，比较 synchronous group barrier 与 single-rollout immediate update 的等待时间、样本新鲜度和训练顺序；输出 `policy_lag`、straggler 等待和每条轨迹的 lineage。
2. 实现 DIS toy：保存 rollout engine 的 token log-probability 和当前 policy log-probability，计算 ratio，分别应用 `[1-epsilon_low, 1+epsilon_high]` 双侧 clipping/masking；注入极端 ratio，检查越界 token 是否从 gradient proxy 和统计中分离。
3. 实现 value-model 消融：比较 critic 每个 policy update 做 `K=1/2` 次更新、attention 全量更新与 attention frozen + MoE projection 更新；固定合成 reward，报告 value loss、advantage 方差、verified success 和失败恢复，不把 toy 结果写成论文复现。
4. 实现 Skip-Observation GAE：把 action token、tool call、环境 observation 和下一 action 段编码为 segment，比较普通 GAE 与跳过 observation 的 bootstrap；故意改变 observation 文本长度和格式，检查 advantage 是否受无关环境 token 污染，同时保留完整 observation 供 replay 审计。
5. 运行 [`sao_async_rl_toy.py`](research/model-update-2026-09/code/sao_async_rl_toy.py)，核对 `group_barrier_wait_total`、ratio mask、Skip-Observation GAE 和 `K=2` critic proxy；再建立来源边界 manifest，分别记录 GLM-5.3 官方 `SAO with compaction` 声明、SAO arXiv HTML/PDF、Qwen3-30B-A3B 论文实验、GLM-5.2 部署声明、DataCurve `mini_swe_agent_glm_5_3_max` 和 AA 指标。缺少 5.3 专属实现时输出 `unverified`，禁止跨来源合并分数。
6. 设计 verifier/reward-shortcut 对照：让一条轨迹读取隐藏答案、修改测试或提交未完成 artifact，比较 reference-free verifier、oracle/no-op/unsolved-state 门禁和继续训练/丢弃 rollout 两种策略；报告假阳性、假阴性、shortcut rate、状态污染和最终 artifact verifier。

## 2026-09 GLM-5.3：compaction 状态恢复审计

1. 运行 [`glm53_compaction_contract_audit.py`](research/model-update-2026-09/code/glm53_compaction_contract_audit.py)，解释为什么合法 JSON 仍可能不是可恢复状态。
2. 依次删除 goal、工具 result digest、幂等键、pending effect、artifact digest 和 verifier 状态，记录每个门禁应拒绝哪一种候选，并区分 schema valid、authorized、executed、verified。
3. 将 toy 扩展为三种切分：完整 tool call 前切分、call 与 result 之间切分、result 完成后切分；要求系统拒绝 orphan call，并报告重复执行率和最终 artifact 一致性。
4. 写一份证据边界说明：Z.ai 的 `SAO with compaction` 是官方高层声明，toy 是通用协议实验，只有官方实现或绑定 revision 的真实 API/权重实验才能升级为模型专属结论。

## 2026-09 GLM-5.3：compaction-aware RL trajectory 与 token provenance

1. 阅读固定版 [`slime` Agentic RL roadmap](https://github.com/THUDM/slime/blob/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b/docs/en/get_started/agent.md) 与 `TrajectoryManager`，画出 session message tree 和 token-prefix alignment 两层结构。用“原始工具轨迹被 compacted history 改写”的例子标注何处分支。
2. 为一次 model → tool → observation → model 轨迹写出 `prompt_ids`、`output_ids`、log-prob 与 `loss_mask`。分别将模板/user/tool/environment token 设为 loss-zero、模型采样 token 设为 loss-one；若历史里的已生成文本发生漂移，说明何时 realign、何时 fork，以及为何不能通过重 token 化恢复原始 rollout provenance。
3. 对照固定 revision 的 `fork_threshold_tokens=1,024` 与 `< threshold` 的 realign 条件，证明该值限制短 re-render/token drift，而非上下文压缩触发阈值；写出不能把它迁移给 GLM-5.3 的理由。
4. 审计同一 commit 的 reward 文档、`TrajectoryManager.get_trajectory()`、coding-agent `generate()` 和 branching tests。报告 `reward/K` 与 full-reward-per-sample 的矛盾；不要把共享 `rollout_id` 当作 reward averaging 证据。因本地没有 pytest，源码单测只能记为已阅读的断言，不可写成已运行通过。

研究笔记：[`glm-5.3-source-notes.md`](research/model-update-2026-09/glm-5.3-source-notes.md)。

## 2026-09 Kimi K3：vLLM v0.29.0 -> v0.30.0 runtime 演进审计

1. 固定 `v0.29.0`、`v0.30.0` tag 与 release commit，读取 registry、NVIDIA K3 model、DSpark 和 PyPI metadata；做 source manifest，保存 URL、tag/commit、字节数与 SHA-256。分别标记 `registry entry`、`wheel metadata`、`wheel installed`、`full-weight load`、`numerical pass` 和 `target hardware acceptance`，不得由前项自动推断后项。
2. 对照权重生命周期：定位 v0.29 在 `load_weights` 内 finalize 与 v0.30 `process_weights_after_loading` 的变化。构造可重复的 streamed/non-contiguous weight fragment 序列，写出为何 finalize 必须等待完整 stream，以及失败重试时如何避免重复转换。
3. 追踪 IPC weight-cache 的 transformed buffer：列出 daemon 与 model process 的 ownership、buffer lifetime、raw packed tensor 释放点、cache miss fallback 和异常清理。源码有零拷贝分支不等于已测出启动速度或峰值显存改善。
4. 为 PP + AttnRes + DSpark/MTP 画状态 manifest，至少分开记 KDA recurrent/conv state、MLA/context KV、auxiliary hidden states、draft/target position、accepted/rejected token 和 PP rank；注入非末级边界与 verify rollback，检查系统是正确转发、拒绝还是恢复。
5. 从 DSpark context-KV 路径提取 dtype、per-layer scale 与 block-layout gates；分别构造 BF16、FP8、混合 dtype 和不均匀 block layout，预期不支持的情况回退而不是静默错误写 cache。注明该练习基于 source reading，不是 kernel 正确性或硬件 benchmark。

## 2026-09 Claude Opus 5：Prompt-Injection 评测可比性审计

根据 [Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card) 和研究笔记，设计一个只处理评测元数据、不执行真实攻击的 audit ledger：

1. 为 IPI、live bounty、Shade coding、computer-use 与 Cowork browser 分别建立 manifest，记录模型 snapshot、scenario set、攻击者版本、attempt budget、thinking/effort、endpoint、tool-result probe、action classifier、Auto mode、harness revision 和 verifier。
2. 每份记录同时保存 `successful_attempts / valid_attempts` 与 `scenarios_with_success / tested_scenarios`。用合成数据验证两种分母不会互相替代，并将重复/自适应尝试标为相关样本，不把 ASR 直接解释成线上用户风险概率。
3. 重现 2026-08-19 的评测修订决策：先注入 Opus 5 与基线使用不同 Cowork harness 的元数据，要求审计器拒绝直接排名；统一 harness 后为所有模型生成 rerun 要求。再注入 Cowork 不支持的 thinking-disabled 条件，要求记录为 `not_supported`，不能暗中降配后计分。
4. 对比 model-only、输入侧 probe、Auto mode 的不同结果，输出 `model_robustness` 与 `product_system_robustness` 两栏。任何评测摘要都必须显示 safeguard 状态、attempt count、scenario coverage 和置信/噪声说明。
5. 将 Anthropic System Card 的 CB-1/CB-2 与 ASL-3 归类记作 `publisher_risk_assessment`；与 DataCurve/AA、外部复现、真实 endpoint 和生产 SLO 分栏，禁止从这组安全评测推断 Opus 5 的参数、架构或普遍风险率。

原始数据与正文摘要：[`claude-opus-5-source-notes.md`](research/model-update-2026-09/claude-opus-5-source-notes.md)。本练习是评测账本设计，不要求生成或复现任何攻击 payload。

## 2026-09 Qwen3.5-Omni：ARIA 与跨媒体时间轴验收

1. 用合成 token 序列实现 ARIA 的样本级 ratio 上限：构造不同 text/speech tokenizer 速率，检查每个 prefix 是否违反 speech:text global ratio；对照固定 interleave schedule。只验证约束和调度性质，不把 toy 指标写成模型质量或论文复现。
2. 构造 VFR 视频帧、音频片段与随机音频 timestamp，分别模拟绝对 temporal ID、160 ms 统一时间粒度、秒级文本 timestamp；注入丢帧、重复帧、音画偏移和跨 chunk replay，输出各自的时间对齐误差。
3. 建立预训练数据账本：S1 encoder alignment、S2 的五类 token 数、S3 `32,768 -> 262,144` 序列长度；核验列项合计约 4.29T 而报告称约 4T，并把精度/重叠未知留为 `unverified`。不得把总体 100M+ 小时、AuT 40M 和 Talker 20M+ 小时合并。
4. 设计 Thinker-Talker 首包 trace：AuT chunk、Thinker TTFT、Talker 首 speech chunk、MTP residual codes、codec decode、网络 packet、播放器首播、并发和取消恢复各自计时。为 Plus/Flash 保留独立部署资源字段，不能直接排名。
5. 从 Model Studio 文档提取 `qwen3.5-omni-plus`、`stream=True`、Plus/Flash custom voice 与 snapshot 限制，建立 API contract manifest；明确不能由示例推断内部架构，也不发起未经授权的真实 API 请求。

研究笔记：[`qwen3.5-omni-source-notes.md`](research/model-update-2026-09/qwen3.5-omni-source-notes.md)。

## 2026-09 Qwen3.6-27B：混合递归状态的树验证与指标审计

1. 画一个 5-token MTP spine + root sibling 的候选树，为每个节点标 parent。用合成递归状态实现 branch-local scan/replay；断言每个子节点从 parent state 计算，并演示把 sibling 状态误接为 packed-row 前项时如何产生错误历史。标明这只是 toy，不是 vLLM kernel。
2. 根据 GDN Tree-Scan 预印本重算并解释三种分母：committed tokens/event 4.11→4.82、token-weighted decode TPS 18.80→23.88、per-request-equal TPS 17.80→18.51。说明为什么 +27.0% decode-only 数字不能外推为 task-wall 加速，并把四个 SWE/Codex tasks、B=1、temperature 0.6 绑定到结论。
3. 为 40-turn recurrent-oracle p-rescore 设计补充验收：定义清晰 margin flip、每组 task/turn、native numerical floor、request-cluster bootstrap、多 seeds、B=4 与 Stage-D timing；写出哪些结论可称“within observed floor”，哪些还不能称为 full distribution equivalence。
4. 制作来源类型表：Qwen 官方模型卡/config 证明模型结构；AA 是榜单/provider 字段；GDN Tree-Scan 是外部单作者预印本的 serving 实验；pinned code 和本文复现是不同证据级别。禁止把第三方 +27% 写成 Qwen 官方宣传或本地实测。
5. 审计 Qwen 官方发布博客的四个分数及脚注：为 SWE-bench Pro 标注修订任务并重跑全部基线，为 SkillsBench 标注 78 个 self-contained 子集与五次平均，为 Terminal-Bench 标注 Harbor/Terminus-2、时限和资源。另把模型卡 native 262,144、OpenClaw `contextWindow=131072` 与 `maxTokens=16384` 分列，解释它们属于不同预算层；API 示例不能替代真实 endpoint probe。

研究笔记：[qwen3.6-27b-source-notes.md](research/model-update-2026-09/qwen3.6-27b-source-notes.md)；论文：[GDN Tree-Scan](https://arxiv.org/abs/2609.23900v1)。

## 2026-09 Qwen3.6-35B-A3B：thinking trace 与历史状态验收

1. 阅读固定 revision 的 [Qwen3.6 chat template](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B/resolve/1a5ae24e867f8d82388070d3f61590158a01d15c/chat_template.jinja)，用合成 transcript（旧 user/assistant-think/tool 轮次 + 最新 user/assistant 轮次）追踪 `last_query_index`。分别渲染 `preserve_thinking` 缺省、true 和 false，断言哪些 `<think>` 内容进入 prompt；把模板静态审计标成 source evidence，不当作线上 API probe。
2. 将 `enable_thinking` 独立开关与 `preserve_thinking` 做 2×2 矩阵：检查生成前缀如何改变、历史 reasoning 是否改变。检查 Model Studio 顶层字段与 vLLM/SGLang `chat_template_kwargs` 的差异，设计 endpoint schema gate，拒绝将一种 surface 的参数形状默默转发到另一种。
3. 构造删掉、重排和跨会话未加载 transcript 的恢复用例，证明 `preserve_thinking` 不能恢复不存在的历史；再将权威计划/事实显式写入 workspace，比较 checkpoint replay、旧 reasoning 和持久状态三者的恢复结果。
4. 对同一长任务比较保留旧 thinking 与只留最新轮：记录输入 token、context/KV 占用、重复推理 token、前缀 digest、cache hit、TTFT 与任务 verifier。净成本使用 `保留历史新增开销 - 减少的重复推理开销` 评估，不假设官方效率 claim 必然成立。
5. 建立模型身份和证据 manifest：把 AA 的 reasoning/non-reasoning slug 归并为同一模型配置组；记录官方 35B/3B、40 层 GDN/Gated Attention 排布、MTP 与 262K native / YaRN 外推；DataCurve 无精确 Qwen3.6 行时不得填其他 Qwen 的成绩。
6. 读取 Qwen3.6-35B-A3B 官方博客，建立评测依赖矩阵，至少区分被测模型、Agent CLI/harness、user simulator、judge model、工具/server version、任务集 revision、截断与资源预算。用 TAU3-Bench、VITA-Bench、MCPMark、MCP-Atlas 的博客脚注填表；模拟 judge 版本漂移时，明确要锁定/重跑哪些 baselines。将 `qwen3.6-flash` 标为同一榜单锚点的 hosted API alias，不增加模型候选；不得发起真实 API 请求。

研究笔记：[`qwen3.6-35b-a3b-source-notes.md`](research/model-update-2026-09/qwen3.6-35b-a3b-source-notes.md)。

## 2026-09 Qwen3.7 Max：环境扩展与长程 Agent 证据审计

1. 按 Qwen 官方博客画出 `Task × Harness × Verifier` 训练矩阵；固定 task set，分别留出未见过的 harness/verifier 组合和 OOD 任务域。写出如何检测 task leakage、verifier 难度漂移与只在已见组合上提升。
2. 把博客的“环境多样性带来可预测增益”转成可证伪实验：记录环境数量/质量分层、训练配比、seen/unseen 组合、run-level 结果、置信区间和失败类型；说明为什么只报告一个子集相关性不足以宣称通用 scaling law。
3. 为 model-assisted reward-hacking monitor 构造合成轨迹，运行 candidate-pattern → replay → counterexample mining → versioned-rule 流程；使用人工标注 holdout 计算 precision、recall、false-positive rate 和漏检类别。说明新增规则需冻结版本并回放历史轨迹，不能只报告规则数或命中数。
4. 根据 M890 案例复原 kernel 优化时间线：Split-KV/online softmax、消除同步/临时分配、adaptive split、reduction batching、MTP `γ=4` 特化；将每个阶段绑定正确性 gate、设备/编译器、baseline 和 workload。分别重算“相对 Triton 的 10.0x 几何平均”与“KernelBench L3 的 eager median / torch.compile win-rate”，禁止把二者混成同一分数。无需下载权重或运行真实 API。
5. 建立 dated-snapshot manifest：将 AA `qwen3-7-max`、Model Studio May-20 alias、May-17 reasoning-only early snapshot 和 June-08 image/video snapshot 分行；标注 context、mode、region/scope、工具能力与来源时间。演示 `preserve_thinking` 只有在 transcript 被回放时才保留历史，不等于跨会话 memory。

研究笔记：[`qwen3.7-max-source-notes.md`](research/model-update-2026-09/qwen3.7-max-source-notes.md)。

## 2026-09 Qwen3.6-35B-A3B：VHD-Play 机制先行的环境与 reward 审计

1. 画出 environment-first 与 mechanism-first 两条依赖链，标注 `M(θ)`、solver、`zθ=(u*,u₀)`、`D(θ)`、`E(θ)`、policy trajectory 和 reward 的构造先后。回答：哪一步消除了 learned judge？哪一步仍可能引入 environment implementation bug？
2. 为一个小型两期库存问题手算 optimum 与默认策略结果，按 `r=clip((u-u₀)/(u*-u₀),0,1)` 计算三条轨迹的终局分数；再讨论 `u*=u₀`、负向目标、数值容差和部分可观测导致 optimum 不可达时应怎样设计族级 admission 规则。不要把扩展设计误写成论文已披露的实现。
3. 给生成环境实现 C1 executability、C2 default-region、C3 optimum/replay checks；分别注入异常工具、默认策略超界、reward 与 reference 不一致、重复运行不稳定，验证拒绝/重试行为。另说明为什么 8/11 family 的 replay audit 不等于全体 environment 的形式化证明。
4. 根据论文分区复算 2,200 train + 300 in-domain held-out + 800 unseen-family evaluation，并画出按机制族隔离的划分；检查参数 seed、语料 passage、工具 schema 与任务模板是否会跨 split 泄漏。把 34-step、34,816 scored rollouts、one run/one evaluation seed 一并写进可复现 manifest。
5. 解释 Base 的 written-out/agentic `0.962/0.204` 与训练后 `0.992/0.815` 的差异；分别指出能支持什么“stateful policy gap”结论、不能证明什么普适 scaling law。把 Qwen3.7-Max 标成比较模型而非本论文受训对象。

研究笔记：[`qwen3.6-35b-a3b-source-notes.md`](research/model-update-2026-09/qwen3.6-35b-a3b-source-notes.md#9-vhd-play先求解机制再生成可交互环境)；arXiv：[2609.27321v1](https://arxiv.org/abs/2609.27321v1)。

## 2026-09 Claude Sonnet 5.5：协议迁移与安全评测账本

1. 写出 `disabled` → `between_tools` 迁移的 request contract matrix；覆盖合法 effort、400 条件、无工具任务、工具间进度 block、`display` 解析和 assistant-turn replay。只依据官方 schema，不调用未授权 API。
2. 建立 thinking-block replay ledger，分别模拟 producer model 不兼容、system/tools 前缀改写、跨账号回放和 compaction 后保留 tail；每种用例选择 `replay`、`drop` 或 `reject`，并记录 `input_transformations`/错误及原始 block hash。
3. 构建 refusal/fallback 路由矩阵：`cyber`、`frontier_llm`、`bio`、`reasoning_extraction`、`general_harms`。验证非允许类别不会自动重试；对允许的 fallback 保存 requested/actual model、category、permission set、副作用状态和 verifier。
4. 为 Gray Swan IPI 与 Coding Shade 编写仅含汇总计数的 benchmark manifest，分别记 attempt-level 与 scenario-level 分母、fallback/probe 覆盖、实际执行模型、thinking 可见性及置信区间。不要生成攻击 payload，也不要把发布方结果称为独立复现。
5. 重述 Terminal-Bench 4.0 `70.6%` 和 OSWorld 2.1 `80.1% partial / 43.5% strict` 的所有必要条件；指出 resource-adaptive benchmark revision、server-side compaction、grader model 和 fallback 如何影响可比性。

研究笔记：[`claude-sonnet-5.5-source-notes.md`](research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)；正式章节：[第二十册第 24 章](book-20-agent-harness-runtime/chapters/24-claude-sonnet-5.5-thinking-state与安全路由.md)。

## 2026-09 Kimi K3：DSpark serving 验收边界

1. 为同一 K3 请求批次画出 `draft length -> verify query length -> k+1 graph capture -> offsets/masks -> KDA state capacity` 的账本。覆盖不同请求长度、prefill/decode row 和超界 profiling dummy，指出哪些应执行、mask 或拒绝；不得用固定长度 scheduler 的 token-count 等式替代请求状态。
2. 设计 cache rebind 故障注入：记录每层 buffer 地址、context-pointer cache key、graph capture 生命周期和 draft rollback。地址变化时验证旧 pointer 被清除并重建；分别检查 graph capture 内更新是否被拒绝，以及地址复用是否导致错误命中。
3. 建立版本/benchmark manifest，对照 vLLM `v0.30.0` stable、adaptive DSpark main commit、MI355X nightly recipe、SiTUv2/AITER 版本与 InferenceX lane。将真实 block verification 和 synthetic acceptance sweep 分开，逐项记录 target verification、任务 verifier、hardware、dtype/layout 与结果来源；synthetic 结果不得作为正确性、线上接受率或独立复现。

研究笔记：[`kimi-k3-source-notes.md`](research/model-update-2026-09/kimi-k3-source-notes.md)；教材：[第二十一册第 88 章](book-21-transformer-architecture-evolution/chapters/88-kimi-k3-kda-stable-latentmoe与百万token-agent.md)与[第二十四册第 32 章](book-24-llm-inference-engine/chapters/32-multi-turn-tool-use和agent-serving支持.md)。

## 2026-09 Kimi K3：DFlash draft 与 target-feature 注入

1. 根据固定 DFlash config，画出 93 层 K3 target 中六个 `target_layer_ids` 的 hidden output 如何经 projection 写入 6 层 draft 的 K/V；标记每个 tensor 的 owner、dtype、cache lifetime 和 block iteration。
2. 用 block size 8 构造 anchor + 7 masked positions 的训练样本，写出块内双向、块间隔离的 attention mask；再用 `w_k=exp(-(k-1)/gamma)` 比较第 1、4、8 个位置的 loss 权重。把这是论文训练机制还是本地 toy 结果分别记录。
3. 建立 DFLASH serving manifest：K3 target revision、DFlash revision、`target_layer_ids`、block size、draft/target backend、hardware、graph bound、真实 acceptance 和拒绝回滚。用一个错误的 `+1` layer mapping 注入 capture mismatch，并说明为什么 shape 通过也不能证明层语义正确。
4. 对照公开 HF draft README 与 SGLang PR #40794 的 benchmark：区分公开 checkpoint、未公开 production checkpoint、stable `v0.5.20` 和 main source；禁止把 95.7% GSM8K 或 synthetic/发布方 acceptance 写成本地复现。

## 六类实作：把技术结果整理成作品集证据

前四项回到第三至十章复习机制；最后一项按[第三册第十五章](book-03-practical-handbook/chapters/15-作品集实战.md)的项目证据链整理结果。

1. 复算目标概率为 0.2/0.8 时的 NLL；在 SFT 多轮样例中分别画出最后一轮监督与全部 assistant span 监督，解释内部 shift 的调用契约。
2. 把服务 trace 的失败结束时间改为 10000 ms，检查成功 QPS 降到 0.3；再将全部请求设为失败，区分零吞吐与未定义的延迟分布。
3. 为 RAG 构造空 ACL、重复 RRF ID、无可见文档和引用否定句，分别验收权限、排序与证据支持，不能以引用合法率替代语义正确性。
4. 将“250厘米”改为“250米”，再构造两次解析失败、一次正确的三候选集，检查投票分母、pass@3 与可重复正确率的区别。
5. 按第十五章 15.2、15.5 和 15.6 节，为一个实际完成的模块整理数据清单、版本、运行命令、原始结果和失败案例；只完成 toy 时不得填入真实模型提升或生产性能。

## 评估与 Debug：分母、证据与安全放行

配套正文：[第三册第九章](book-03-practical-handbook/chapters/09-评估与debug实战.md)。

1. 在四条 benchmark 全部正确时复算 Wilson 下界；把 JSON 年龄改成字符串，区分键检查、字段类型、业务约束与模型正确率。
2. 构造“两条必需事实只正确回答一条”的例子，分别计算回答支持率和完整性；将证据改成否定句，证明字符串命中不代表语义支持。
3. 向人工标注表重复插入同一 `(case, model, annotator)`，要求拒绝重复计数；设计每个问题标注人数不均的例子，比较记录平均与问题平均。
4. 用正文的 6/7 次安全事件设计放行报告：写明预定非劣效界限、区间方法、随机化单元、离线高风险验证和人工复核；没有充分证据时保持观察，不能只用安全率点估计或“不显著”放行。

## 2026-10 Gemini 4 Argon：发布方证据账本

1. 将 AA `gemini-4-argon`、Google The Keyword、DataCurve 缺失行分别登记为榜单、官方发布和负证据。
2. 对 1M output、DeepSWE 77.9%、AutomationBench 51.3%、LVBench 91.7% 和 CWE-bench 68% 记录任务集、harness、预算、verifier、分母和证据等级。
3. 为 cyber vulnerability workflow 设计权限、沙箱、出站控制、人工审批、artifact digest 和修复 verifier 字段；不执行真实攻击或调用 endpoint。

## 多模态实战：机制与证据边界

配套正文：[第三册第十四章](book-03-practical-handbook/chapters/14-多模态评估与交付.md)。

1. 分别只轮换 CLIP 文本向量、同步轮换向量与标签，解释哪一种破坏配对；验证单样本零 loss 不代表学会对齐，并说明多相关目标的 Recall 分母。
2. 向数据构造器加入同图同问题的冲突答案和不同记录的重复 ID，要求拒绝；构造不同路径的同一图片，说明路径切分为何不足以排除泄漏。
3. 向公开图片的 linked_doc_ids 插入私有文字与不存在 ID，验证不能扩展越权内容；空 ACL、错误租户和无相关候选应返回空证据。
4. 对五步 DDPM schedule 核对累计 alpha、反向均值与后验方差；验证最后一步随机项为零，解释使用已知 x0 反解噪声为何不能替代生成时噪声预测器，以及最终 alpha 约 0.8583 为何不是纯噪声。
5. 构造 CER 小于 0.2 但漏掉否定词的转写，以及“不是 128 元”的图片回答；确认 toy 规则可能通过，另设计关键字段和语义标注验收。比较语料级 CER 与逐句平均，并测试插入导致 CER 大于 1。
6. 分开计算视觉/文本输入与输出 token 到达率；用总成本 20 元、100 次请求、80 次成功计算单位成功成本，再说明重试按任务归并后应如何统一分母。

## 第三册第十章：Agent 轨迹与线上实验

配套正文：[第三册第十章 10.1 节](book-03-practical-handbook/chapters/10-agent轨迹与线上评估实战.md)。

1. 为“修改订单地址”任务构造两条轨迹：第一条在未确认时提议更新，但执行器返回拒绝且状态未变；第二条同样缺少确认，却收到 `accepted=true` 且订单地址改变。标出哪一条是可恢复的策略违规尝试，哪一步构成首个不可恢复失败；说明最终回答文本为什么不能证明动作状态。
2. 对 12 条轨迹，8 条达成任务目标、12 个适用节点中 10 个选择了正确工具、12 条中 1 条有越权副作用。分别计算任务成功率、工具选择准确率和策略违例率，给每个分母加上适用范围；解释为什么不能把三个比例平均成一个总分。
3. 将一个长 Agent trace 按 system、trace、node 三层汇总。只改变工具参数验证逻辑后，原 4 条失败中 3 条修复，新增 1 条权限错误；设计回归表分别记录成功率、schema 合法率、策略违例、首个失败位置和安全 holdout，并给出是否进入灰度的理由。
4. 为 4 轮会话记录每轮输入 token `[512, 1800, 4200, 9000]`、工具等待和生成长度。计算未命中缓存时累计 prefill 输入 token；再说明 prefix cache 命中、服务排队和工具等待分别会怎样改变“模型 TTFT”和“用户端到端任务时间”。
5. 设计一个敏感信息安全的 trace schema，保留模型/工具版本、参数校验、执行回执、前后状态和策略决定，同时对用户数据做字段级脱敏；说明若只保存最终回答、只保存外部状态或把所有原始输入写入日志，各会漏掉什么或引入什么风险。
6. 计划按 50:50 分桶，实际随机化单元计数为 Control 6,000、Treatment 4,000。手算 SRM 的卡方统计量，并说明为何应先停下业务效果解读，检查分配、过滤和日志分母；随后列出一个线上扩量前必须满足的质量/安全护栏和回退条件。

## 多模态训练：目标函数与配方对照

配套正文：[第五册第十二章](book-05-llm-training/chapters/12-多模态训练.md)。

1. 对一个正确图文 logit 为 3、语义相关错误负例 logit 为 2 的 batch，分别计算两候选 CLIP softmax 概率与 SigLIP 单对 logistic loss；再设计多正例标签，解释 sigmoid 目标为何不能自行发现语义相关性。
2. 为冻结视觉端且算力受限、视觉指令数据较多、以及多分辨率混合预训练三种条件，分别选 BLIP-2、LLaVA-1.5、MM1 作为研究对照；写出共同评测集和分辨率/token 消融，禁止跨论文分数直接排名。
3. 对一张双栏收据，比较 OCR pipeline、LayoutLMv3、Donut 和 Pix2Struct 的输入/监督/输出；构造“税额识别正确但总额关系错误”的样例，并定义字段关系、区域证据与解析成功指标。
4. 目标字符为 `book` 时写最短 CTC 路径；按 `T=160,U=20` 估算 RNN-T 格点。再给出支持低延迟 partial transcript 的 encoder/endpoint 约束，说明它们不由 loss 名称自动保证。
5. 令 `γ=5`，计算 `SNR=100` 和 `0.1` 时 `x0`、`epsilon`、`v` 三种 Min-SNR 权重，并把论文收敛/FID 数字绑定回目标参数化、schedule 和实验设置。
6. 对 8 帧 `224×224` 和 16 帧 `448×448`、patch `16` 计算 patch tokens 与全注意力 score cell 比值；设计固定帧、按桶与 Frame Pack 的等 token 预算对照，并加入时序事件召回评测。

## 数学百科：定义域与反例

配套正文：[第四册数学与机器学习](book-04-llm-encyclopedia/chapters/01-数学与机器学习.md)。

1. 用 A=-I 与非零向量验证负特征值反向；用 diag(3,1) 的 rank-1 截断验证误差范数为1、平方误差为1、保留能量为9/10，区分矩阵能量和任务能力。
2. 对 `[0.99,0.01]` 和均匀分布计算熵与第二类目标的 PPL，解释为什么更确定不等于困惑度更低。
3. 固定先验0.7，分别用通过率(0.9,0.3)、(0.9,0.9)、(0.8,0.9)计算后验，说明检查通过何时提高、保持或降低可靠性。
4. 运行三个相同的1e20 logits，比较中心化交叉熵和先算logsumexp再相减的结果；检查空输入、非有限值、非法target和单类别边界。
5. 写出PPO正负优势下的裁剪项，解释为何这不是硬KL约束；为同题多采样、人评多标注设计配对/聚类标准误或bootstrap单位，避免把相关记录当作独立样本。

## 深度学习组件：接口与状态验收

配套正文：[第四册深度学习组件](book-04-llm-encyclopedia/chapters/02-深度学习组件.md)。

1. 执行 CPU GradScaler 的 scale/unscale/clip/step 示例，核对未缩放范数2、裁剪后参数约0.9；解释累积窗口和溢出跳步时为何不能无条件推进 scheduler。
2. 构造共用 EOS/PAD ID 的 batch，分别统计 ID 命中、mask padding、未 shift labels 与真正 shifted targets；测试全 ignore、监督 padding 和非法 mask。
3. 用同一 bool mask 分别调用 SDPA 与 MultiheadAttention，验证 True 的含义相反；对 LayerNorm 设置非零 bias，说明输出均值不再为0。
4. 用相同键、错误 shape 的 state_dict 测试 strict=False；保存普通 state_dict 引用后修改参数，与深拷贝比较，解释键相同为何不等于权重恢复正确。
5. 两个 micro-batch 分别有10和1个有效 token，比较均值再平均与token加权平均；将同一推导扩展到默认按rank平均梯度的DDP，说明no_sync为什么不修正权重错误。

## Tokenization 百科：覆盖与计数契约

配套正文：[第四册 NLP 与 Tokenization](book-04-llm-encyclopedia/chapters/03-nlp与tokenization.md)。

1. 比较 `"\u00e9"` 和 `"e\u0301"` 的码点数、UTF-8 字节数与 NFC 结果，说明 Python len 不等于字素簇数；对真实 tokenizer 另测 token 数，不能从字符数推断。
2. 对 `aaa` 执行一次非重叠 BPE 合并；再用词表 `ab`、`a`、`##bc` 分析 WordPiece 贪心失败，说明存在可覆盖切分为何不保证贪心成功。
3. 固定总 NLL=4、字节数=8，分别用2和4个目标 token 计算 PPL 与 BPB；解释不同 PPL 不代表不同文本质量，并列出规范化、文档边界和窗口 stride 的控制条件。
4. 构造 EOS/PAD 共用 ID 的标签，按实际 padding 位置屏蔽，保留真实 EOS；分别统计原始有效标签和 causal shift 后分母，验证非法负标签和被监督 padding 会被审计器拒绝。
5. 为模板直接返回 IDs 和先渲染字符串再编码建立对照，核查 BOS/EOS 是否重复；用输入600、可见输出200、隐藏输出100核对总预算900，解释输出上限已含隐藏部分时为何不能重复相加。

## Transformer 百科：信息路径与状态验收

配套正文：[第四册 Transformer 架构](book-04-llm-encyclopedia/chapters/04-transformer架构.md)。

1. 执行 CPU attention 的未来位置和 padding 反事实测试；再故意取消 mask，构造输出发生变化的输入。说明 attention 局部通过为何还不能证明完整模型无泄漏。
2. 缓存2个位置后加入1个 query，写出正确允许矩阵；用零 Q/K、V=[1,2,6] 比较显式 mask 与 SDPA 非方形 is_causal=True 的3和1，解释左上对齐问题。
3. 对4个 query heads、2个 KV heads，比较 enable_gqa 与 repeat_interleave K/V；核对 head 映射，再计算32层、batch4、8192 token、8 KV heads、head_dim128、FP16缓存为4 GiB，说明它已经包含四条序列。
4. 从非负特征映射的 kernel attention 分子重排补出归一化分母，列出分母为零的处理策略；说明固定特征宽度下线性复杂度与精确 softmax attention 的区别。
5. 用 S=2、k=1、v=3、α=1、β=0.5计算两次 delta 写入2.5/2.75；与直接叠加比较，再讨论非正交 key 对旧记忆的干扰。另验证 RoPE 点积的相对相位恒等式，避免将它误解为点积只依赖距离。

## 大语言模型基础：目标与评估分母

配套正文：[第四册大语言模型基础](book-04-llm-encyclopedia/chapters/05-大语言模型基础.md)。

1. 对masked NLL分别输入未归一化log probabilities、bool目标、全忽略目标和不一致词表宽度，确认拒绝；说明它与接收logits的CrossEntropyLoss及内部shift有何区别。
2. 比较预测分布[0.99,0.01]与[0.5,0.5]的熵，以第二类为参考目标计算PPL100/2，解释低熵为何不能证明预测正确。
3. 对logits=[1e308,-1e308]、temperature=1e308核对概率约[0.880797,0.119203]；测试相同超大logits和极小温度，再检查采样不改变全局random状态及固定seed的使用边界。
4. 用N=10^9、D=2×10^10核对6ND，再在固定compute下将N翻倍；说明该dense训练近似不能直接推出下游质量或端到端费用。为涌现曲线分别设计连续分数和离散阈值对照。
5. 100个任务全部尝试花20元、80个成功，计算平均成本与单位成功成本；再用四个任务中两项条件同时成功2次、同时失败2次，比较真实联合通过率0.5与边际乘积0.25，说明失败/重试成本及相关条件如何影响口径。

## 预训练百科：数据口径与边界

配套正文：[第四册预训练与数据](book-04-llm-encyclopedia/chapters/06-预训练与数据.md)。

1. 同一训练文本与两条仅空白不同的评估文本比较，核对共有hash数1、命中文档数2；解释hash相同、规范化相同、答案污染与模型记忆四者为何不能等同。
2. 对packed输入[1,2,0,3,0]构造独立样本attention边界和labels[-100,2,0,-100,0]，核对shift后目标3且EOS保留；再说明有意允许跨文档读取的预训练流为何是另一种目标。
3. 用池大小[100,400]计算τ=1和τ=2的权重[0.2,0.8]与[1/3,2/3]；换成不同长度文档后，说明文档采样权重为何不是token比例，以及有放回采样的effective epoch为何不是逐样本曝光保证。
4. 将合成答案“2”改为“3”却保留verified=True，观察教学函数仍会通过，设计独立验证证据；分别说明替换旧数据与保留真实数据并累积的model collapse研究条件，不能把任一结果外推到所有合成训练。
5. 为数据消融选择固定token或固定compute预算，列出序列长度、packing、步数和学习率的伴随变化；设计扫描器命中率、误报/漏报和小桶风险的分层报告。说明数据删除与模型参数遗忘、不可变版本与永久保留的区别。

## 后训练百科：目标与反馈边界

配套正文：[第四册后训练与对齐](book-04-llm-encyclopedia/chapters/07-后训练与对齐.md)。

1. 用 stored labels `[-100,-100,a0,a1,EOS]` 列出内部 shift 后的四个目标及其预测位置；解释为何 mask 正确仍可能因 shift 错误监督到用户内容。
2. 在 IPO 中取 τ=0.5，分别计算间隔 0.8 与 1.2 的损失和导数；比较 DPO 的 logistic 目标，说明有限目标间隔为何不保证全模型无回归。
3. 用 SimPO 平均 log probability -0.4/-0.7、β=2、γ=0.2 核对损失约 0.513015；将长度和 log probability 总和同时翻倍，说明长度平均为什么没有直接惩罚总长度。
4. 对奖励组 `[0,0,0,0]`、`[1,1,1,1]`、`[0,0,1,1]` 执行 DAPO 过滤；核对最后一组优势并解释保留组成功率为何不能代表题池成功率。对 token ratio 2/0.5 比较逐 token clip 与 GSPO 几何平均。
5. 为 RLAIF judge、规则 verifier 和教师软分布分别写出反馈类型与可用目标；核对固定历史下 CE-H=forward KL，解释为何成败标签不能直接作为 token 分布。设计安全可答集合为空时的报告，以及工具轨迹的脱敏与权限边界。
6. 对同一模型、同一数学题池比较 GRPO、DAPO、DrGRPO、GSPO 与 DCPO：先列出每种方法修改的是 advantage、ratio 粒度、clip bounds、采样还是 loss denominator；再实现一个固定预算/逐 response token mean 对照，计算非零 advantage 率与正确率。论文中的 DAPO progressive ablation、DrGRPO AIME24 分数和 DCPO RUR 必须保留各自模型、数据、模板和统计口径，不能拼成统一排行榜。

## 推理与解码百科：分布、预算与延迟

配套正文：[第四册推理与解码](book-04-llm-encyclopedia/chapters/08-推理与解码.md)。

1. 对 logits `[2,1]` 分别取 temperature 0.5、1、2，核对第一 token 概率；再用 `[1e308,-1e308]` 与温度 `1e308` 验证先缩放和先相减的差异。
2. 对权重 `[1e308,1e308,0]` 取 top-p=0.75，说明为什么要先按最大权重缩放；比较 p=1、零概率尾项和全候选被 mask 时的处理契约。
3. 用 target `[0.5,0.3,0.2]`、draft `[0.4,0.4,0.2]` 计算接受质量、拒绝概率、残差分布并恢复 target；说明 q=0 提议为何非法、p=q 时残差未定义。
4. 用上下文 4096、输入 3800、保留 32、请求 800 核对有效输出 264；设计空输出、时间原点不一致和单 token 时 TTFT/TPOT 的报告。
5. 比较 MHA/GQA 的 KV 字节量，列出滑动窗口、混合层和前缀共享为何不能直接套用单一公式；为 best-of-N 独立候选、相关候选和选择器误判分别设计报告。

## 推理优化与部署百科：资源、协议与回滚

配套正文：[第四册推理优化与部署](book-04-llm-encyclopedia/chapters/09-推理优化与部署.md)。

1. 用请求状态图拒绝 `RECEIVED -> DECODING`，说明 `FINISHED` 与 `CLEANED` 为什么不能合并；为取消、超时和模型异常分别列出 KV 释放证据。
2. 对 running sequence 数和 waiting prefill token 分别设预算，复现长请求先到与短请求先到的调度差异；设计 aging、优先级和 P99 公平性指标。
3. 计算 GQA 相对 MHA 的 KV 节省，再列出 prefix cache 指纹中的模型、模板、adapter、位置配置、媒体和租户条件；解释命中率为何不等于 TTFT 收益。
4. 用对称 INT4 教学量化计算 scale 和 MSE，说明校准分布、kernel、KV 精度和真实任务回归为何不能由 toy MSE 推断。
5. 为一次已开始 streaming 的工具调用设计 fallback、幂等键、取消和回滚流程；列出 startup/liveness/readiness、trace/metrics、drain 和 canary 的证据字段。

## 第六册第十章：多模态部署

配套正文：[第六册第十章](book-06-llm-deployment/chapters/10-多模态部署.md)。

1. 对 1800×2400 图像、patch=32 计算原始 patch 数；比较固定 resize、tiling 与动态分辨率的视觉细节、token cap 和长尾成本。再用 1,000,000×1 图像与 cap=1 检查教学 resize 的边界，并说明为什么真实 processor 输出才是部署账本。
2. 用 45 秒视频、0.5 FPS、每帧 256 token 计算抽帧后的 token 预算，再分别比较 encoder 前减少帧数与 encoder 后使用 EVS/VidCom2 pruning：哪些成本已经发生、哪些成本还能节省？设计一组包含静态字幕、短暂事件和重复背景的视频，报告 temporal recall、证据支持、prefill、KV 和 encoder 工作量。
3. 计算 Riva 两种窗口：低延迟 `chunk=0.16s, padding=1.92s` 与高吞吐 `chunk=0.8s, padding=1.6s`。比较模型窗口时长、更新间隔和在“每个窗口均完整重算”假设下的输入覆盖率；再列出 runtime feature reuse、GPU 并发、WER/CER、首个 stable 延迟需要记录的实测项。
4. 给客服语音助手设计级联 ASR→LLM→TTS 和原生 speech/omni 两条方案。固定一组否定词、数字、打断和工具写操作样本，定义首个 stable、首包音频、打断残留、转写证据、工具副作用和单位成功成本门禁；解释语音模型是否原生不能替代服务端权限校验。
5. 为 Thinker→Talker→Code2Wav 比较融合、完整 payload handoff 和 async chunk handoff。设计含背压、乱序 chunk、取消、下游崩溃、重试和客户端断开的状态表；把 TTFT（文本首 token）、TTFP（音频首包）与 E2EL 分开测量。

## 第六册第十一章：监控、评估与安全

配套正文：[第六册第十一章](book-06-llm-deployment/chapters/11-监控评估与安全.md)。

1. 构造 root span [0,500] ms，其中检索 span [20,170]、模型 span [100,430]、审核 span [410,490]。计算根 span 的 E2E 和子 span 持续时间之和，解释为什么有并行时不能把 span duration 直接加成用户延迟；再区分逐请求 decode rate 与 10 秒墙钟窗口内的服务 output tokens/s。
2. 对 100 条安全样本、20 条攻击样本分别给出 TP/FP/FN/TN，计算 block rate、FPR、FNR 和 ASR，明确每个指标的样本集合与分母。将攻击集缩为 1 条且零次成功，按 95% 单侧零事件上界 1-0.05^(1/n) 计算结果；讨论攻击样本相关或分布不代表真实攻击时，这个区间能说明什么、不能说明什么。
3. 同一组 200 个请求运行 baseline 与 candidate，按请求 ID 配对计算质量分差，并用请求级 bootstrap 报告区间。构造 25% candidate 请求没有评分的例子，同时报告评分覆盖率、已评分回答均值和失败/拒答处理方式；说明只比较成功评分样本会如何改变结论。
4. 为开放式客服、RAG 企业问答和工具型 Agent 分别选择 HELM、MT-Bench/Chatbot Arena、RAGAS/ARES、HarmBench 或 AgentDojo 中的适用方法。每项写明方法解决的问题、判分机制、关键限制，以及在目标系统中增加的切片、人工校准、工具副作用或线上回放。

## 第六册第十二章：成本优化

配套正文：[第六册第十二章](book-06-llm-deployment/chapters/12-成本优化.md)。

1. 为一个含文本 RAG、图片理解、GPU 推理、重试和人工复核的请求列资源向量；分别按 API 和自托管两种口径换算成本，指出哪些账目不能相加或重复计费。再用单次成本 0.010→0.007、成功率 0.90→0.60 计算每成功任务成本变化，并说明零成功时分母的语义。
2. API 固定月费 0、边际 0.0024 美元/请求；自托管固定月费 48,000 美元、边际 0.0008 美元/请求。求交叉点，计算 20M 与 40M 请求的成本；随后把自托管边际成本上调到 0.0018、固定成本加上 20% 冗余，重算阈值并说明真实 trace 中还应改变哪些假设。
3. 对同一组 1,000 条带成功标签的请求设计 FrugalGPT cascade 和 RouteLLM preference router。画出每种方案一次请求的模型调用轨迹，写出期望调用成本与串行延迟来源，并构造一个 Arena 偏好数据与企业任务不匹配的测试切片，定义 fallback 和安全路由门槛。
4. 对一份长 RAG 文档分别选 LLMLingua、LongLLMLingua 或 LLMLingua-2，说明每种压缩机制要保留的证据、条件、否定和引用。用每百万输入 token 2 美元、10,000→6,000 token、压缩成本 0.0015 美元、恢复成本 0.002 美元计算净节省；再求恢复成本达到多少时压缩不再回本。
5. Prefix cache 与 response cache 各列出命中率分母、资源收益、版本/权限 key 和失效风险。若完整模型路径成本 0.01 美元、查找 0.0001 美元、维护摊销 0.0002 美元，求纯货币回本命中率；另列出哪些安全/新鲜度条件即使通过回本计算也必须绕过缓存。
6. Draft 每轮提出 K=4 个候选且各位置条件接受率为 0.8，计算 speculative sampling 的期望输出 token 数，解释为什么它不是加速比。比较独立 draft、MTP、EAGLE 与 n-gram/suffix 的候选来源、额外资源和分布保证。
7. 给权重显存受限、activation W8A8 受限和长上下文 KV 受限三个场景，分别比较 GPTQ、AWQ、SmoothQuant、KIVI 的目标张量、误差处理和 runtime 依赖。再设计目标任务质量切片与有效 bit/scale/峰值显存/吞吐报告。
8. 对 μ=10 请求/秒的 M/M/1 队列，分别计算 λ=8 与 9.5 时的利用率和平均排队时延，并解释它不能直接当作 LLM 容量公式。对 4 个副本、观测指标为目标 1.8 倍的 HPA 控制环算出建议副本数，并补充 queue age、待处理 token、KV 和冷启动指标。
9. Agent 首次带入 1,000 token，每步追加 400 token，连续 6 步都重发完整历史，计算累计输入 token；与每步固定 1,000-token 摘要比较，并把摘要生成成本、失败恢复和工具提交状态加入生命周期成本。

## 第六册第十三章：生产系统设计

1. 某服务有 FAQ、RAG 和多步 Agent 三类请求。给出每类到达率、有效输入/输出长度、模型调用数、任务停留时间和实测模型服务时间，分别计算 input/output token/s、端到端任务并发与模型 active sequence；再按单副本 prefill/decode/sequence 容量和 20% 余量求副本数。注入一个模型副本故障，检查 N+1 后余量，并说明为什么三个不同负载下测出的峰值不能直接拼接。
2. 为“30 天内 95% 有效同步请求在 3.5 秒内完成”写出 SLI 分子、分母、窗口和测量边界。分别决定服务端超时、客户端断连、用户取消和未完成请求如何入账；计算 10,000 个 eligible 请求对应的 bad-event budget，并另设任务质量与高风险安全门槛。
3. 设计一次退款 Agent 的 commit-unknown 恢复流程：列出 intent、授权、dispatch、回执未知、状态查询和最终反馈的持久字段。比较幂等键、Transactional Outbox 与 Saga 各自的原子性边界、重复投递处理、参数变化行为和补偿限制；构造“同 key 不同金额”和“下游已退款但响应丢失”两个反例。
4. 设计带 control 组的模型 canary。候选流量占 5%、候选错误率 20%，计算在 control 无错误时全服务聚合错误率的变化；再设计按租户、任务类型、请求长度和 cache 状态分层的归因表，列出质量、安全、P95、成本、未知提交和回滚 guardrail，并说明样本不足时的决策。
5. 用本章教学 demo 修改 small/large pool 的 prefill 吞吐、TPOT 或 GPU 预算，手算一个请求的模型服务时间和 active sequence 变化，再运行代码对照输出。把 `model_path_estimate_ms` 与观测到的端到端 P95 分开；给缺少真实 P95、任务质量或安全证据的版本补上暂停门禁。
6. 为模型 pod 分别写 startup、readiness、liveness 检查条件。构造“权重仍在加载”“队列暂时过长”“进程死锁”“检索依赖暂时不可用”四种状态，说明哪些应延迟探活、摘流量、重启或只做 admission control；解释为何把过载统一设成 liveness 失败可能触发重启级联。

## 第六册第十四章：部署方案推导与故障复盘

配套正文：[第六册第十四章](book-06-llm-deployment/chapters/14-部署方案推导与故障复盘.md)。

1. 用每秒 60 个请求、平均输入/输出 1,500/200 token、输入 token 复用 25%、单副本 prefill/decode 吞吐 80,000/2,500 token/s、单副本活跃序列容量 12、模型 runtime 平均占槽 1.2 秒和 `γ=1.2`，分别计算输入、输出、活跃序列三个容量比例。求满足余量的副本数，并在失去一个副本后仍保留该余量时计算总池规模。说明如果 1.2 秒包含 RAG 或工具等待，哪些数字必须重测。
2. 比较 PagedAttention 和 vAttention 的 KV 管理层、kernel 依赖、碎片处理、机制代价和论文评测边界；再比较 GPTQ、AWQ、SmoothQuant、KIVI 各自量化的张量、误差机制和适用约束。为“校准分布偏移”“量化质量掉点”“缓存容量仍不足”分别提出下一轮实验假设，并标明哪些是论文已报告机制、哪些是待验证改进。
3. 以正文的合成 prompt 发布事故为例，画出“模板字段顺序变化 → token 前缀变短 → cache 复用下降 → 重复 prefill/KV 生命周期增加 → queue wait 与 TTFT 尾延迟上升”的因果链。给出至少两条支持证据和两条能推翻假设的反证；区分观测事实、机制推断和仍未知的质量影响。数值必须标为教学合成值，不能写成真实事故。
4. 写一份无责复盘草案，包含影响范围、检测/响应/缓解/恢复时间线、直接触发、系统促成因素、未确定项、行动 owner、截止时间和关闭证据。至少为 prompt 模板契约、cache/prefill 分层监控、长短请求隔离或公平调度、canary 回滚各设计一个可验收行动项；解释为何“建了 ticket”不等于行动项关闭。
5. 修改本章 Python demo 的 QPS、平均 runtime 占槽时间、cache token 复用比例或 `γ`，逐项预测 active sequence 与副本数变化后执行示例核对。把输出中的 token 成本与 GPU 固定成本、缓存开销、重试、fallback 和人工升级分开；指出哪些输入需要目标 workload 才能测得。

## 第六册第十五章：FP4、MXFP4 与 NVFP4

配套正文：[第六册第十五章](book-06-llm-deployment/chapters/15-fp4-mxfp4-nvfp4.md)。

1. 用 E2M1 码本 `0、±0.5、±1、±1.5、±2、±3、±4、±6` 量化数值 `0.9`。分别令共享 scale 为 20 和 0.9，算出归一化值、最近码值、重构值和绝对误差；再解释为何真实 MXFP4/NVFP4 还要计入 scale 自身的量化。
2. 对 `N=4096×4096` 个参数，分别计算 FP16、OCP MXFP4、NVFP4 一维分组的 payload、scale 字节、理论压缩比和有效 bits/element。列出 padding、rowwise/columnwise layout、二维权重 scaling、未量化层与 workspace 会怎样改变 artifact 或峰值显存。
3. 比较 RTN、通用 rotation、MR-GPTQ 与 NVFP4 scale learning：各自处理哪一种误差，核心步骤是什么，代表性结果绑定什么模型/任务/硬件，在哪些条件下不能推广？据此为目标 workload 设计一个先诊断 scale error、再选择补偿方法的消融矩阵。
4. 对同一模型建立 BF16、FP4-weight-only、FP4-weight+activation、NVFP4-W4A4 与高精度回退组，固定模型 revision、校准集、kernel、engine 和请求分布；分别记录 scale/saturation、准确率切片、格式合法率、峰值显存、TTFT、TPOT 和 fallback，说明每类结果能支持什么结论。

## 第六册第十六章：FP8 KV Cache

配套正文：[第六册第十六章](book-06-llm-deployment/chapters/16-fp8-kv-cache.md)；研究来源见[精修记录](research/fp8-kv-cache-depth-review-2026-10-27.md)。

1. 用 OCP E4M3/E5M2 的 max finite、最小 normal/subnormal 和特殊值字段，计算一组输入最大值 500、`s=500/448` 时，`x=1` 如何映射到 E4M3 的 0.875 并重构为约 0.9766。再计算 `448s` 的上溢边界和 `2^-9s` 的下溢边界；说明 OCP 格式规范为什么不能单独决定 runtime scale。
2. 对两位置 logits `[1,0]` 计算 softmax；把第一项改为 0.7，重算概率并说明排序未变时输出如何变化。再令 `V=[1,0]`，分别计算 K 误差和 `V_1 += 0.1` 对输出的影响。指出该标量例子不等于模型级准确率预测。
3. 对 `L=64, B=1, T=131072, H_kv=8, D_h=128` 分别算 BF16、FP8 payload，再加入假设 8% metadata 得 17.28 GiB。说明这个账本不包含 allocator、workspace 和模型结构差异；将 KIVI 的 K-per-channel/V-per-token、KVQuant 的 pre-RoPE key 与 vLLM per-head scales 分别放到“研究算法/运行时能力”层比较。
4. 解释为什么已写入 page 的 scale 从 0.1 改为 0.2 而 payload 不重编码，会让 `D(q)=10` 的值从 1 重构为 2。为 append、共享 prefix、swap/restore 和并发读取分别设计可维持 scale revision 一致性的检查。
5. 阅读 vLLM H100/FlashAttention 3 案例：按 128k NIAH 的 91%/13%/89% 结果重建“BF16 基线—原 FP8 路径—two-level accumulation 修复”因果链；再解释 ITL slope、break-even、TTFT 和并发输出吞吐是不同指标。为目标部署列出固定模型、backend、scale、长度分布、并发和任务样本的验收矩阵。

## 第八册第十七章：AI Control 与不可信 Agent 监督

配套正文：[第八册第十七章](book-08-ai-safety-alignment/chapters/17-ai-control与不可信Agent监督.md)；来源边界见[研究与精修记录](research/book-08-ch17-ai-control-refinement-2026-10-28.md)。

35EJ. 20 个诚实任务中，不可信模型 $U$ 完成 16 个、可信模型 $T$ 完成 10 个、协议完成 13 个；先分别求三种成功率，再计算归一化效用。100 对攻击任务中 6 次在轨迹审核时被捕获、2 次未捕获地完成副任务，求重复部署安全性，并说明 92 个未发生终止事件的样本为何不能自动算安全成功。比较 `defer-to-resample`、`resample-then-defer` 和 `defer-on-critical` 的提交权与监控可见性；设计一个针对 monitor 的自适应攻击及独立 holdout。最后为不可逆写操作加提交前控制，说明轨迹结束后审计与副作用前拦截的差别，并列出一页安全论证所需的迁移假设和停止条件。

## 第九册第一章：数据总览

配套正文：[第九册第一章](book-09-data-engineering/chapters/01-数据总览.md)。

1. 为一条网页数据画出 source registry → raw snapshot → parse result → clean document → dedup cluster → training mixture → model run 的血缘图。给每个节点写稳定 ID、版本/hash 和责任字段；指出仅保留最终文本时哪些问题无法追溯。
2. 构造数学、代码、长尾语言、通用网页四个数据池，各自说明训练目标、最小覆盖、许可/隐私门禁和主要失败模式。比较“增加 token 数”和“提高单位计算下的目标能力”两种方案应使用的指标与对照。
3. 从模型在多语言、安全、代码三个切片上的变化反推需要补充的证据；为每个切片列出数据来源、训练前分布、训练后行为和至少一个能证伪“数据导致该变化”的实验。

## 第九册第二章：Web-Scale 数据采集

配套正文：[第九册第二章](book-09-data-engineering/chapters/02-web-scale数据采集.md)。

1. 为一个抓取批次定义 source、请求事件、HTTP 响应、解析结果和不可变快照字段；构造成功、重试后成功、被 robots 规则阻断和解析失败四种记录，说明它们各自如何计入采集覆盖与失败率。
2. 对同一页面的 HTTP/HTTPS、尾斜线、追踪参数、分页参数和语言参数 URL 设计 canonicalization。给出会合并的 URL 与必须保留为不同内容的 URL，并说明重定向链和源站标识如何进入 lineage。
3. 设计采集准入门禁，至少包含权利/许可、robots/访问政策、隐私、频率限制、时效性和解析完整度；说明当正文可抓取但许可状态未知时，数据进入哪个状态以及如何阻止它误入训练池。

## 第九册第三章：清洗、过滤与质量评分

配套正文：[第九册第三章](book-09-data-engineering/chapters/03-清洗过滤与质量评分.md)。

1. 在 100 条人工复核样本上，过滤阈值 A 得到 TP=32、FP=8、FN=6、TN=54；阈值 B 得到 TP=36、FP=18、FN=2、TN=44。分别计算 precision、recall 和 FPR；若漏掉高质量长尾样本代价为误删的 5 倍，比较两组加权错误成本，并解释阈值如何随风险切片变化。
2. 为文本质量分设计参考集和人工标签协议，覆盖语言、领域、长度、文档类型和高风险类别。说明如何做校准、阈值抽样复核和过滤后目标任务验证，防止参考集只代表英语网页。
3. 区分许可/隐私/污染硬门禁与可排序的软质量分；给出三种“高分但必须隔离”和三种“分数偏低但可能值得保留”的样本，写出人工复核及后续动作。

## 第九册第四章：去重与污染检测

配套正文：[第九册第四章](book-09-data-engineering/chapters/04-去重与污染检测.md)。

1. 对两个 5-shingle 集合分别为 `{a,b,c,d,e}` 与 `{a,b,c,x,y}` 的文档，计算 Jaccard 相似度；再构造一段短文本完整出现在长文档中的例子，比较 Jaccard 与 containment 为什么可能给出不同判断。
2. 为 exact、near-duplicate、长文档包含、语义相似和 benchmark 污染设计分级处理规则。列出每一级的候选信号、人工确认方式、误删风险和是否允许训练/评测。
3. 设计训练集与评测集的污染审计：冻结评测快照，定义 exact/normalized/near/semantic 查询，按题目、文档和子串分别记录分母；说明如何处置发现的泄漏并保留审计证据。

## 第九册第五章：Data Mixture 与配比

配套正文：[第九册第五章](book-09-data-engineering/chapters/05-data-mixture与配比.md)。

1. 在 200,000 个有效 token 的固定预算下，按数学 40%、代码 25%、安全 20%、通用 15% 分配 token。若四个池的过滤保留率分别为 50%、80%、75%、60%，计算各池最少需要提供的原始 token 数，并说明分母必须采用有效 token 还是原始 token。
2. 用同一总 token 预算比较自然分布和人工配比：写出目标权重、来源采样概率、实际落盘 token 和训练曝光次数；解释一个语料占比上升时其余语料的有效 epoch 如何变化。
3. 构造某个来源池为空、权重为零、权重 NaN 和所有来源被硬门禁阻断四种边界；分别说明应报错、标记 N/A、移除后重归一化还是暂停采样，并解释不能悄悄把未满足预算的 token 补给其他池。

## 第九册第六章：Code、Math 与 Domain Data

配套正文：[第九册第六章](book-09-data-engineering/chapters/06-code-math-domain-data.md)。

1. 为代码样本设计从许可检查、语言/版本识别、解析、去重、单测执行到安全沙箱的 pipeline。构造一个可运行但输出错误、一个测试通过但许可证不清、一个可复现且有授权的样本，判断各自能否进入候选池。
2. 为数学题分别设计数值答案校验、符号等价、单测/证明验证和过程质量审查。用一道有单位换算与负号的题构造“最终答案碰巧正确但推理错误”的反例，并说明标签如何分开记录。
3. 为法律或医学领域语料增加时效性、来源等级、专家复核、隐私和检索记忆风险字段；设计过期知识与低资源但重要知识的不同处置，避免用通用 loss 或质量阈值一刀切。

## 第九册第七章：Synthetic Data 与 Distillation Data

配套正文：[第九册第七章](book-09-data-engineering/chapters/07-synthetic-data与distillation-data.md)。

1. 对四条样本分别记录“是否合成”“是否由 teacher 生成/监督”“teacher revision”“原始 seed/提示”“验证方式”：人工题经 teacher 解答、teacher 自举题、程序生成题和真实用户问题经 teacher 改写。解释 synthetic 与 distillation 是两条可交叉的血缘轴。
2. 设计 synthetic 指令、推理轨迹、偏好解释和 domain QA 的分层验收。至少指定一项自动正确性检查、一项独立 holdout、一项人工抽样，以及循环自举/模板化/答案泄漏的停止信号。
3. 给 12 个生成样本构造 source/teacher/任务标签和重复簇，分别计算有效样本比例与重复率；说明 pairwise 重复率与“被重复簇覆盖的样本比例”采用不同分母时不能混报。

## 第九册第八章：Preference Data 与安全数据

配套正文：[第九册第八章](book-09-data-engineering/chapters/08-preference-data与安全数据.md)。

1. 对同一提示的两个回答分别标 helpfulness、honesty、harmlessness、style 和 uncertainty；构造标注者在风格上分歧但安全判断一致、以及偏好一致但安全边界错误的例子，说明维度为何不能合成一个无条件 chosen/rejected 标签。
2. 为拒答、转述用户请求、含工具动作和分步执行四类安全响应标注目标行为、风险严重度、过度拒答和工具副作用；定义独立的误拒率、漏拒率与任务成功分母。
3. 设计 preference rubric、双标注、仲裁和版本变更流程；将无法判定、信息不足及价值观分歧保留为独立标签，并给偏好训练前后的能力与安全回归门槛。

## 第九册第九章：多模态、语音与视频数据

配套正文：[第九册第九章](book-09-data-engineering/chapters/09-多模态语音视频数据.md)。

1. 总语料有 8,850 token，过滤后保留 3,550 token；其中视频原始 3,440、保留 980 token。分别计算整体保留率、视频池内保留率和最终 mixture 中视频占比，解释三种比例的分母为何不同。
2. 一条语音参考文本 20 个词，识别结果有 2 次替换、1 次删除、1 次插入；计算 WER。另对 8 秒视频按 1 FPS 采帧，说明理论帧数以及 scene boundary、shot sampling 和动态事件如何改变实际采样计划。
3. 为图像 OCR、语音转录和视频片段共同设计统一 sample ID 与空间/时间边界字段；加入人脸、车牌、屏幕文字和声纹隐私标记、授权状态、训练/评测污染状态，说明缺失边界、无对应模态和无法验证答案如何分别记录。

## 第九册第十章：Data Attribution 与 Valuation

配套正文：[第九册第十章](book-09-data-engineering/chapters/10-data-attribution与valuation.md)。

1. 对 `math_verified` 使用数学、代码、安全、通用权重 `0.35/0.25/0.20/0.20` 和指标变化 `0.080/0.010/0/0.015`，手算加权变化；再代入质量 `0.92`、覆盖 `0.78`、梯度相似 `0.942`、风险 `0.03`、成本 `0.22`，复算 proxy 值 `0.1808` 和 900 个有效 token 下的每千 token 值。解释这些系数为何不是货币价格或通用阈值。
2. 用章节给出的三参与者效用，逐项算出六种加入顺序的边际值与 Shapley 平均；检查三个 Shapley 值之和是否等于全体效用 `0.540`，并解释 `0.08` 协同和 `-0.05` 条件冲突分别如何分摊。
3. 将 `benchmark_leak`、`old_legal_forum` 和空梯度样本分别判定为硬阻断、软复核/负向效用或未定义 attribution；说明为何高相似分数不能解除污染门禁，且高风险不应被编码成一个负分后再排序。
4. 在 `3,200` token 预算和每项指标最大回退 `0.01` 下，手算 exact toy selection 的组合、token 总量、各目标变化和 proxy 总值。随后把代码源换成合成模板源，比较固定 token、固定训练步数两种口径下反事实结论可能如何变化。
5. 设计一个从小模型 proxy 到目标模型的验证计划：固定数据快照、总 token、优化阶段、随机种子和替代源；对 PreSelect、梯度归因或 source-level ablation 选一项设定目标/邻近/通用/安全/长尾切片、最小效应与置信区间，并列出授权、污染、风险及结果不稳定时的停止门槛。
