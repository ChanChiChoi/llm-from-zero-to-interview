# 练习与验收体系

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
44. 分析一次训练异常。
45. 用自己的话解释 AI Safety 和 Alignment 的区别，要求覆盖目标、风险、训练、评估和部署。
46. 给 8 条 toy 请求设计 risk taxonomy，至少包含 harmful content、privacy、jailbreak、prompt injection、tool misuse、high risk domain 和 benign help。
47. 写一个 0 依赖 Python demo，计算 unsafe compliance、refusal accuracy、over-refusal、attack success、unauthorized tool call、safe completion quality 和 severity-weighted risk。
48. 设计一个 safety gate，说明哪些指标是硬门禁，哪些指标可以作为灰度观察项。
49. 选 3 个 helpful、honest、harmless 冲突案例，写出模型应该拒绝、澄清、部分回答或安全替代的理由。
50. 为一个带工具调用的助手设计安全审计表，字段包含工具权限、参数校验、二次确认、审计日志、回滚和人工接管。
51. 用一个客服模型例子解释 Alignment Problem，要求区分真实意图、目标规范、训练 proxy、模型实际行为和部署风险。
52. 比较 outer alignment 和 inner alignment，要求各举 2 个 LLM 场景例子。
53. 设计一个 goal misgeneralization 评估，让训练分布中的两个目标在测试分布中分离。
54. 列出 5 个 specification gaming 例子，至少覆盖 reward model、benchmark、judge、引用和安全关键词。
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

## 阶段 4：部署与系统验收

1. 解释 prefill/decode。
2. 估算 KV Cache 显存。
3. 部署一个小模型服务。
4. 设计一个 RAG 系统。
5. 用自己的话解释 greedy decoding 和 sampling 的区别。
6. 手算一个 temperature 降低后分布变尖锐的例子。
7. 给定 token 概率，分别选出 top-k 和 top-p 候选集合。
8. 为事实问答、代码生成、创意写作分别设计 decoding 参数。
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
130. 用 5 分钟回答“如何设计一个企业知识库 RAG 问答系统”。
131. 画出 embedding retrieval 的离线索引和在线查询流程。
132. 用自己的话解释双塔检索为什么适合大规模召回。
133. 写出 cosine similarity 公式，并说明和 dot product 的关系。
134. 解释 in-batch negatives、hard negatives 和 false negatives。
135. 构造一个 query、positive document、hard negative document 的训练样本。
136. 比较 HNSW、IVF、PQ 的直觉、优点和代价。
137. 设计一个向量检索评估集，包含 recall@k、precision@k、MRR、nDCG。
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
161. 设计一个天气查询工具 schema，包含城市和日期字段。
162. 设计一个订单查询工具 schema，并说明权限检查在哪里做。
163. 给定 5 个用户问题，判断是否需要调用工具，并写出工具名和参数。
164. 构造一个工具参数缺失时需要追问用户的样本。
165. 构造一个工具返回错误时模型不编造结果的回答样本。
166. 设计一个高风险工具调用的二次确认流程，例如转账、删除数据或发邮件。
167. 设计一个 function calling 评估表，包含工具选择、参数、执行、最终回答和安全。
168. 设计一个 tool call trace 日志 schema。
169. 分析 prompt injection 如何诱导工具越权调用，并给出 3 个防护策略，要求包含 tool result injection、参数污染、二次确认和 audit trace。
170. 用 3 分钟回答“如何设计一个能调用企业 API 的 LLM 助手”。

## 阶段 5：研究与面试验收

1. 精读 10 篇论文。
2. 完成一个项目讲稿。
3. 完成 3 次 mock interview。
4. 能回答开放研究题。
5. 给出 5 个不同类型的幻觉例子。
6. 设计一个事实性评估集，说明数据来源和评分方式。
7. 解释 benchmark contamination 为什么会误导模型比较。
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
31. 说出 3 个平衡 safety 和 helpfulness 的具体策略。
32. 解释 test-time compute 为什么会成为重要研究方向。
33. 讨论合成数据的一个优势和一个风险。
34. 录音模拟一次 3 分钟自我介绍，并写下 3 个可改进点。
35. 选一个项目，用“问题、数据、方法、实验、失败、改进”六段式讲一遍。
36. 从题库中随机抽 10 道题，要求每题用 2 分钟回答。
37. 对一次 mock interview 做复盘，记录原题、原回答、错误原因和修正答案。
38. 准备 10 个必须能讲清的问题，并逐个写出 5 句话版本答案。
39. 做一次 45 分钟完整模拟面试：自我介绍、项目、基础、系统、开放题各一轮。

## 阶段 6：数据工程与治理验收

1. 设计一个 web-scale 数据采集 source registry，字段至少包含 source id、license、ToS、robots、access mode、owner、risk level 和 delete policy。
2. 给 8 条 toy HTML 样本设计采集审计表，标注 policy block、PII / secret、benchmark contamination、low quality、exact duplicate 和 kept。
3. 写一个 0 依赖 Python demo，计算采集后的保留率、语言配比、领域配比、去重率和风险命中率。
4. 用 3 分钟回答“为什么技术可访问不等于可进入训练数据”。
5. 画出从公开 crawl dump 到训练集版本的 pipeline，包含 raw storage、parser、quality filter、PII scanner、dedup、dataset builder 和 version registry。
6. 为 12 条 toy 文本设计质量特征表，至少包含长度、unique token ratio、重复率、符号比例、乱码比例、boilerplate / promo 命中和风险标记。
7. 写一个 0 依赖 Python demo，按质量分、PII / secret、安全风险、benchmark contamination 和 exact duplicate 输出 `kept` 与拒绝原因。
8. 构造一个人工审计集，计算过滤系统的误删率、漏删率、token 保留率和每个语言 / 领域桶的保留率。
9. 解释为什么代码、数学、低资源语言和专业文档不能共用普通网页文本的质量阈值。
10. 设计一次小模型 ablation：比较清洗前、规则清洗后、质量分类器过滤后三个数据版本的验证 loss、下游指标和风险命中率。
11. 给 10 条 toy 文档计算规范化 hash，找出 exact duplicate group，并说明保留哪一条以及为什么。
12. 用 3-gram shingle 手算两段文本的 Jaccard 相似度，再解释 MinHash 为什么可以近似这个相似度。
13. 写一个 0 依赖 Python demo，输出 near duplicate pairs、SimHash 汉明距离、聚类结果和每个簇的代表样本。
14. 构造 3 条 toy benchmark 样本，检测训练语料中的题目泄漏、答案泄漏和 canary 命中，并输出隔离清单。
15. 设计一份去重后 data mixture 审计表，比较去重前后的语言、领域、来源、token 数和质量分分布。
16. 给 8 个 toy 数据池设计 data mixture 配置，字段至少包含 clean tokens、quality score、risk score、能力标签、采样权重和 planned tokens。
17. 写一个 0 依赖 Python demo，比较 natural sampling、temperature sampling 和质量加权采样下的语言 / 领域配比变化。
18. 计算每个数据池的 effective epoch，并标出可能带来记忆或过拟合风险的上采样数据池。
19. 设计一次 mixture ablation：分别提高代码、数学、多语言和合成数据比例，并说明要看哪些评估指标。
20. 用 3 分钟回答“为什么 data mixture 是隐式目标函数设计，而不是数据拼接配置”。
21. 给 12 条 toy code / math / domain 样本设计专项审计表，字段包含 license、secret、test pass rate、answer verification、authority、PII 和 contamination。
22. 写一个 0 依赖 Python demo，分别计算代码、数学和领域数据的质量分、拒绝原因、分类型保留率和最终 mixture。
23. 设计一个代码数据清洗 checklist，覆盖 fork、vendor、自动生成文件、license、secret、单测和 benchmark 污染。
24. 设计一个数学数据 verifier 审计流程，说明如何检查答案、步骤、难度、重复题和评测污染。
25. 设计一个专业领域数据治理方案，覆盖来源权威性、时间戳、引用、PII、专家抽检、RAG 元数据和安全边界。
26. 设计一个 synthetic / distillation data 生成表，字段包含目标能力、seed、teacher、prompt version、sampling 参数、quality score、validation、contamination、PII 和 license。
27. 写一个 0 依赖 Python demo，按授权、验证、去重、污染、安全、PII 和合成比例门禁输出 `kept`、拒绝原因、origin mix、task mix 和 teacher mix。
28. 比较 Self-Instruct、Evol-Instruct 和 teacher-student distillation 的目标、生成方式、过滤重点和风险。
29. 设计一次 synthetic ratio ablation：分别测试 10%、30%、50%、70% 合成数据占比，并说明要观察哪些能力、风格和风险指标。
30. 用 3 分钟回答“为什么合成数据是训练分布编辑，而不是免费扩容 token”。
31. 设计 10 条 prompt/chosen/rejected 偏好样本，标注 helpful、honest、harmless、长度、语言、风险类别和选择理由。
32. 写一个 0 依赖 Python demo，计算偏好样本的标注一致性、平均 margin、长度偏置、风险覆盖、误拒修复样本数和漏拒修复样本数。
33. 构造一份安全数据分层表，至少包含明确安全、边界允许、高风险、隐私、专业高风险和多语言样本。
34. 设计一个红队回归数据版本报告，字段包含风险类别、失败模式、安全响应、模型版本、policy 版本、修复状态和复测结果。
35. 写七个防御性安全审计 demo：一个 red teaming demo，输入 toy case 的风险类别、严重度、baseline、natural score、elicited score、工具确认和回归结果，输出 taxonomy coverage、P0/P1 unresolved、dangerous capability uplift、autonomy score、regression pass rate 和 release gate；一个 mechanistic interpretability demo，输入 toy clean / corrupted logit delta、patching 结果、ablation 结果、SAE feature 激活和标签，输出 patch recovery、ablation effect、reconstruction fidelity、avg active features、feature purity 和 interpretability gate；一个 steering demo，输入正负 toy 激活、alpha 网格、target gain、safe gain、side loss 和 over-refusal delta，输出 steering vector、projection shift、target uplift、safe gain、side-effect drop、over-refusal delta 和 steering gate；一个 model editing / unlearning demo，输入 toy edit cases、paraphrase checks、locality checks、forget set、retain set、改写 / 多轮泄露和 membership inference 风险，输出 edit success、generalization、locality、retain drop、forget leak、robust leak、membership risk drop、editing gate 和 unlearning gate；一个 privacy / watermarking demo，输入 toy PII / secret / RAG / log / membership / watermark cases，输出 PII recall、secret recall、memorization rate、output leak rate、RAG unauthorized retrieval、raw log rate、membership advantage、watermark z-score、generated recall、false positive rate、robust recall、privacy gate 和 watermark gate；一个 governance / model card demo，输入 toy model card sections、system card sections、policy categories、eval slices、risk issues、approval votes 和 release checks，输出 model card completion、system card completion、policy coverage、eval coverage、severity-weighted unresolved risk、high-risk mitigation、approval coverage、governance gate 和 release ready；一个 safety interview retrospective demo，输入 toy answer records、risk set、metric set、gate set 和 trade-off set，输出 answer coverage、risk coverage、metric coverage、gate coverage、trade-off coverage、unsafe detail count、readiness gates 和 revision plan。
36. 用 3 分钟回答“为什么高质量安全模型不是拒答率最高的模型”。
37. 给 12 条 toy 多模态样本设计审计表，覆盖 image-text、OCR、audio-transcript、video-caption 和 multimodal instruction 五类样本。
38. 为每条多模态样本标注 media quality、alignment score、annotation quality、license、privacy、contamination、time error、grounded answer 和 tokens。
39. 写一个 0 依赖 Python demo，按媒体质量、图文 / 音文 / 视频对齐、隐私、版权、评测污染、时间错位和 grounded answer 输出 `kept` 与拒绝原因。
40. 计算多模态数据的整体 token 保留率、分模态保留率、最终 mixture、平均对齐分和拒绝原因分布。
41. 设计一个 OCR / 文档图像数据 schema，字段包含 page id、recognized text、bounding box、reading order、confidence、language、table region、PII flag 和 source license。
42. 设计一个音频 / 视频时间对齐抽检流程，说明如何用 WER / CER、subtitle shift、VAD segment、clip boundary 和人工抽样判断样本是否可进入训练。
43. 用 3 分钟回答“为什么多模态数据工程的核心不是增加模态数量，而是证明模态之间真的对齐”。
44. 给 8 个 toy 数据源设计 valuation 表，字段包含 tokens、quality、coverage、risk、cost、license、contamination、目标能力收益向量和数据版本。
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
