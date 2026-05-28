# 面试题库

## ML 与数学基础

1. 为什么语言模型用交叉熵损失？
   考察点：最大似然、负对数似然、one-hot 标签、next-token prediction。
   回答框架：最大化训练文本似然等价于最小化 NLL；在 one-hot 标签下 NLL 就是 cross entropy。

2. 交叉熵和 KL 散度有什么关系？
   考察点：`KL(p || q) = H(p, q) - H(p)`。
   回答框架：真实分布固定时 `H(p)` 是常数，所以最小化交叉熵等价于最小化 KL。

3. Perplexity 是什么？
   考察点：平均交叉熵、指数形式、语言模型评估。
   回答框架：如果 loss 用自然对数，perplexity = exp(loss)，可以直觉理解为平均候选困惑度。

4. 为什么 loss 降低不一定代表模型更好？
   考察点：分布差异、评估污染、平均指标、真实性和安全性。
   回答框架：loss 衡量 token prediction，不直接衡量用户偏好、事实性、安全性或某些特定任务能力。

5. Adam 和 AdamW 的区别是什么？
   考察点：decoupled weight decay、L2 regularization、自适应学习率。
   回答框架：AdamW 将 weight decay 从 Adam 的梯度更新中解耦，避免 weight decay 被自适应学习率扭曲。

6. 为什么大模型训练需要 warmup？
   考察点：训练初期不稳定、Adam 矩估计、loss spike、NaN。
   回答框架：warmup 让学习率从小逐步增大，降低训练初期发散风险。

7. SGD、Momentum、Adam 的区别是什么？
   考察点：当前梯度、历史梯度、一阶矩、二阶矩、自适应学习率。
   回答框架：SGD 直接用梯度，Momentum 加历史方向，Adam 同时使用一阶和二阶矩调整更新。

8. 训练 loss spike 时如何从优化器和学习率角度排查？
   考察点：learning rate、warmup、gradient norm、weight decay、scheduler state。
   回答框架：先看学习率曲线和 warmup，再看 gradient norm、weight decay 配置和 optimizer/scheduler 恢复状态。

9. 梯度表示什么？
   考察点：loss 对参数的敏感程度、负梯度方向、局部下降。
   回答框架：梯度指向 loss 增大最快方向，梯度下降沿负梯度方向更新参数。

10. 反向传播如何工作？
   考察点：计算图、链式法则、forward、backward、参数梯度。
   回答框架：前向计算 loss，反向从 loss 出发用链式法则计算每个参数梯度。

11. loss 不下降可能有哪些原因？
   考察点：数据、label、loss mask、学习率、梯度、优化器、数值精度、分布式。
   回答框架：先查数据和 loss，再查学习率和梯度，再查优化器、精度和工程实现。

12. 为什么需要 gradient clipping？
   考察点：梯度爆炸、训练稳定性、gradient norm。
   回答框架：限制梯度范数，避免异常大梯度导致参数更新过大和 NaN。

## Transformer 与 LLM

1. LLM 到底在学什么？
   考察点：token 序列、条件概率分布、next-token prediction、记忆与泛化。
   回答框架：LLM 学习 `P(x_t | x_<t)`，通过海量文本学习语言、知识、任务模式和推理线索；它既可能记忆，也可能泛化。
   常见追问：为什么这不是简单背诵？为什么这不是搜索引擎？

2. 为什么 next-token prediction 能产生看似智能的能力？
   考察点：自监督学习、文本中的知识压缩、上下文建模、规模化训练。
   回答框架：文本包含人类知识和推理过程，预测下一个 token 需要建模语法、语义、事实、任务意图和推理模式。
   常见追问：它是否能保证真实性？为什么会 hallucinate？

3. 大模型是背诵还是泛化？
   考察点：memorization、generalization、训练数据重复、组合泛化。
   回答框架：不是二选一。大模型会记忆训练数据中的部分内容，也会学习可泛化模式。判断要看新分布任务、污染检测和错误分析。

4. LLM 和搜索引擎有什么区别？
   考察点：生成模型 vs 检索系统、参数知识、RAG。
   回答框架：LLM 根据上下文生成 token，搜索引擎检索已有文档；RAG 是把检索和生成结合。

5. 什么是语言模型？
   考察点：序列概率、联合概率、条件概率分解。
   回答框架：语言模型为 token 序列分配概率；自回归模型用链式法则分解为 `∏ P(x_t | x_<t)`。

6. 为什么 GPT 使用 next-token prediction？
   考察点：自监督、训练目标和生成过程一致、可扩展性。
   回答框架：文本天然提供标签，目标简单统一，适合大规模预训练，并且和自回归生成一致。

7. 训练和推理有什么区别？
   考察点：label、teacher forcing、并行训练、逐 token 生成、decoding。
   回答框架：训练有真实 label 且可并行预测；推理没有 label，需要模型逐步生成，生成结果会成为后续上下文。

8. Teacher forcing 是什么？
   考察点：训练稳定性、真实历史 token、推理时错误累积。
   回答框架：训练时使用真实历史 token 作为上下文，而不是模型自己生成的 token。

9. Self-Attention 的公式是什么？
10. 为什么要除以 sqrt(d_k)？
11. GPT 和 BERT 的核心区别是什么？
12. 什么是 tokenization？
   考察点：token、vocabulary、token id、模型输入空间。
   回答框架：tokenization 把文本转成 token 序列和 token id，是模型处理文本的入口。

13. BPE 的核心思想是什么？
   考察点：高频 pair 合并、subword、OOV、词表大小。
   回答框架：从小单位开始，不断合并高频相邻片段，得到子词词表。

14. Tokenizer 如何影响上下文长度和成本？
   考察点：token 压缩率、context window、attention 成本、KV Cache。
   回答框架：模型窗口按 token 计算，token 数越多，计算和显存成本越高。

15. 新增特殊 token 要注意什么？
   考察点：vocabulary、embedding resize、chat template、loss mask。
   回答框架：更新 tokenizer，扩展 embedding，保证训练推理模板一致，并检查特殊 token 是否参与 loss。

16. Embedding 层做了什么？
   考察点：token id、embedding matrix、shape、查表。
   回答框架：Embedding 把离散 token id 映射成连续向量，矩阵形状是 `[vocab_size, hidden_size]`。

17. 为什么 Transformer 需要位置编码？
   考察点：self-attention permutation-invariant、顺序信息、position embedding。
   回答框架：attention 本身不天然知道 token 顺序，需要位置编码注入顺序信息。

18. Token embedding 和 contextual representation 有什么区别？
   考察点：静态表示、上下文动态表示、Transformer 层。
   回答框架：embedding 是查表得到的初始向量，contextual representation 是结合上下文后的动态表示。

19. 新增 token 后模型侧要做什么？
   考察点：resize embedding、初始化、weight tying、tokenizer 一致性。
   回答框架：更新 tokenizer 后要扩展 embedding 和可能的输出层，并保证训练推理一致。

20. Self-attention 解决了什么问题？
   考察点：上下文相关表示、长距离依赖、并行训练。
   回答框架：让每个 token 动态关注其他 token，生成上下文相关表示，并支持任意 token 直接交互。

21. Query、Key、Value 的直觉是什么？
   考察点：信息检索类比、匹配权重、Value 汇总。
   回答框架：Query 是查询，Key 是索引，Value 是内容；QK 匹配得到权重，再加权汇总 V。

22. Attention 和 RNN 相比有什么优势？
   考察点：并行性、长距离依赖、路径长度、复杂度。
   回答框架：attention 可并行，任意 token 直接交互；代价是 `O(T^2)`。

23. Attention 权重能否解释模型？
   考察点：可解释性、因果解释、多层网络。
   回答框架：attention 权重可提供线索，但不是完整因果解释。

24. 写出 scaled dot-product attention 公式。
   考察点：Q、K、V、softmax、scale、shape。
   回答框架：`Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`，并解释每一步含义。

25. 为什么 attention 要除以 `sqrt(d_k)`？
   考察点：点积分数方差、softmax 饱和、梯度稳定性。
   回答框架：`d_k` 大时点积尺度变大，缩放可以稳定 softmax 输入。

26. Full attention 和 linear attention 的区别是什么？
   考察点：`O(T^2)`、特征映射、近似、质量和效率 trade-off。
   回答框架：full attention 显式计算所有 token pair，linear attention 避免显式 `T*T` 矩阵但可能改变或近似 attention。

27. FlashAttention 和 sparse attention 的区别是什么？
   考察点：系统优化 vs 结构稀疏。
   回答框架：FlashAttention 优化精确 attention 的 IO 和 kernel，sparse attention 改变连接模式只算部分 token pair。

28. 为什么需要 Multi-Head Attention？
   考察点：多个子空间、不同关系模式、信息路由。
   回答框架：多个 head 可以并行关注不同类型的信息关系，比单头更灵活。

29. MHA 的 shape 如何变化？
   考察点：`[B,T,d_model]`、`[B,h,T,d_head]`、scores `[B,h,T,T]`。
   回答框架：Q/K/V 投影后 reshape 成多头，计算 scores，再 concat 回 `d_model`。

30. MHA、MQA、GQA 有什么区别？
   考察点：Q heads、K/V heads、KV Cache、推理效率。
   回答框架：MHA 每个 Q head 有独立 K/V，MQA 共享一组 K/V，GQA 分组共享 K/V。

31. 增加 head 数会增加参数量吗？
   考察点：`d_model` 固定、`head_dim`、Q/K/V/O 参数量。
   回答框架：`d_model` 固定时总投影矩阵通常不变，head 数主要改变子空间划分。

32. 什么是 causal mask？
   考察点：下三角 mask、未来 token、attention scores、softmax。
   回答框架：causal mask 防止当前位置看到未来 token，通常在 softmax 前把未来位置置为 `-inf`。

33. 训练时为什么也要 causal mask？
   考察点：完整序列输入、信息泄漏、loss 虚低、自回归约束。
   回答框架：训练时并行输入完整序列，如果不 mask，模型会看到未来，破坏 next-token prediction。

34. Causal LM 和 Masked LM 有什么区别？
   考察点：GPT、BERT、单向上下文、双向上下文、生成 vs 理解。
   回答框架：Causal LM 从左到右预测下一个 token，Masked LM 用双向上下文预测被 mask token。

35. Teacher forcing 和 causal mask 是一回事吗？
   考察点：真实历史 token、未来可见性、自回归训练。
   回答框架：不是。Teacher forcing 使用真实历史，causal mask 禁止看到未来。

36. 一个 Transformer block 包含什么？
   考察点：attention、MLP、normalization、residual、decoder-only。
   回答框架：现代 decoder-only block 通常是 `x = x + Attention(Norm(x))`，再 `x = x + MLP(Norm(x))`。

37. Attention 和 MLP 分别起什么作用？
   考察点：跨 token 信息交互、逐位置非线性变换。
   回答框架：attention 负责 token 间通信，MLP 负责每个 token 表示的非线性加工。

38. Pre-LN 和 Post-LN 有什么区别？
   考察点：norm 位置、训练稳定性、深层网络。
   回答框架：Pre-LN 在子层前 norm，更利于深层训练；Post-LN 在残差后 norm。

39. 为什么残差连接重要？
   考察点：信息保留、梯度传播、深层训练。
   回答框架：残差让每层学习增量修改，并为梯度提供更直接路径。

40. LayerNorm 和 BatchNorm 有什么区别？
   考察点：统计维度、batch 依赖、训练推理一致性、Transformer。
   回答框架：BatchNorm 依赖 batch 统计，LayerNorm 在 hidden dimension 上归一化，不依赖 batch size，更适合 Transformer。

41. RMSNorm 和 LayerNorm 有什么区别？
   考察点：mean centering、RMS、现代 LLM。
   回答框架：LayerNorm 减均值并除标准差，RMSNorm 不减均值，只按 RMS 缩放。

42. 为什么现代 LLM 常用 Pre-LN？
   考察点：深层训练稳定性、残差路径、梯度传播。
   回答框架：Pre-LN 让子层输入尺度稳定，并保留更直接的残差梯度路径。

43. 为什么 norm 参数通常不做 weight decay？
   考察点：scale 参数、训练稳定性、正则化配置。
   回答框架：norm 参数用于控制激活尺度，对其衰减可能破坏归一化效果和训练稳定性。

44. RoPE 的核心思想是什么？
   考察点：旋转位置编码、Q/K、相对位置。
   回答框架：RoPE 按位置旋转 Q/K，使 attention score 自然包含相对位置信息。

45. RoPE 为什么具有相对位置性质？
   考察点：旋转角度差、点积、`i-j`。
   回答框架：Q/K 分别按位置旋转，旋转后点积与角度差有关，角度差对应相对距离。

46. 长上下文为什么困难？
   考察点：`O(T^2)`、KV Cache、位置外推、训练数据、评估。
   回答框架：长上下文同时带来计算、显存、建模、数据和评估挑战。

47. 如何评估长上下文模型？
   考察点：needle、多文档、摘要、代码、位置敏感、成本。
   回答框架：不能只测 needle，还要测真实任务、不同位置和长度，并评估延迟显存成本。

48. 从零实现 GPT 的核心模块有哪些？
   考察点：embedding、causal attention、MLP、norm、residual、lm head、loss。
   回答框架：decoder-only GPT 由 token/position embedding、多个 causal Transformer block、final norm 和 lm head 组成。

49. Causal LM 的 input 和 label 如何构造？
   考察点：右移一位、next-token prediction。
   回答框架：input 是 `x_1..x_T`，target 是 `x_2..x_{T+1}`，每个位置预测下一个 token。

50. generate 函数的核心逻辑是什么？
   考察点：自回归生成、last logits、sampling、context append。
   回答框架：取上下文 forward，取最后位置 logits，解码出下一个 token，拼接后重复。

51. 教学 miniGPT 和真实 LLM 差距在哪里？
   考察点：tokenizer、RoPE、RMSNorm、SwiGLU、FlashAttention、KV Cache、分布式训练。
   回答框架：miniGPT 用于理解结构，真实 LLM 需要更复杂架构和大规模工程系统。

## 训练

1. 如何训练一个 7B 模型？
2. loss spike 怎么 debug？
3. 如何检测 benchmark contamination？
4. 如果让你训练一个 LLM，你会怎么设计完整流程？
   考察点：目标、数据、tokenizer、模型、优化器、分布式、监控、checkpoint、评估、后训练。
   回答框架：先明确目标，再构建数据和 tokenizer，设计模型和训练系统，监控指标，保存评估 checkpoint，最后进入 SFT/RLHF/DPO。

5. 为什么数据清洗和去重重要？
   考察点：数据质量、重复、记忆、污染、泛化。
   回答框架：清洗降低噪声，去重减少过度记忆和训练浪费，污染检测保证评估可信。

6. Checkpoint 应该保存什么？
   考察点：model、optimizer、scheduler、RNG、step、config。
   回答框架：不仅保存模型权重，还要保存完整训练状态，否则恢复训练可能异常。

7. Base model 和 instruct model 有什么区别？
   考察点：预训练、指令微调、对齐、聊天能力。
   回答框架：base model 学文本续写，instruct model 经过后训练，更会遵循用户意图和安全规范。

8. 什么是 scaling law？
   考察点：参数量、数据量、计算量、loss、资源规划。
   回答框架：scaling law 描述模型性能随规模变量变化的规律，常用于训练预算和模型规模规划。

9. Chinchilla scaling law 的核心启发是什么？
   考察点：compute-optimal、参数量、训练 token 数。
   回答框架：固定 compute 下要平衡模型大小和 token 数，很多大模型训练 token 不足。

10. 为什么数据质量重要？
   考察点：有效信号、噪声、重复、污染、能力分布。
   回答框架：高质量数据提升有效学习信号，低质量和重复数据会导致噪声、记忆和评估失真。

11. 如果模型数学能力差，你会从数据角度怎么改？
   考察点：数据配比、推理数据、合成数据、评估、ablation。
   回答框架：检查数学数据比例和质量，加入高质量推理数据，做 math eval 和 mixture ablation。

12. 为什么大模型训练需要分布式？
   考察点：参数、梯度、optimizer state、激活、计算速度。
   回答框架：单卡显存和速度不足，需要数据并行、参数分片、张量并行和流水线并行。

13. DDP 如何工作？
   考察点：模型副本、不同 batch、all-reduce、梯度同步。
   回答框架：每卡完整模型处理不同 batch，backward 后 all-reduce 梯度，保持参数一致。

14. ZeRO/FSDP 解决什么问题？
   考察点：训练状态冗余、optimizer states、gradients、parameters。
   回答框架：通过分片训练状态降低单卡显存，代价是更多通信。

15. Tensor Parallel 和 Pipeline Parallel 区别是什么？
   考察点：层内切分、层间切分、通信、pipeline bubble。
   回答框架：TP 切矩阵计算，PP 切模型层；TP 通信频繁，PP 有流水线空泡。

## 对齐

1. SFT、RLHF、DPO 的区别是什么？
2. reward hacking 是什么？
3. DPO 为什么不需要显式 reward model？
4. 为什么 base model 需要 instruction tuning？
   考察点：next-token prediction、文本续写、用户意图、助手角色、后训练。
   回答框架：base model 会续写但不天然会遵循指令，需要指令-回答数据把行为调整成助手。

5. Instruction Tuning 和 SFT 有什么关系？
   考察点：目标和方法的区别、监督学习、指令遵循。
   回答框架：Instruction Tuning 是让模型学会听指令的目标，SFT 是常用监督训练方法。

6. 为什么 SFT 通常只对 assistant response 计算 loss？
   考察点：条件上下文、label mask、ignore index、生成目标。
   回答框架：user/system 是输入条件，目标是生成 assistant，训练 user token 会偏离目标。

7. Chat template 为什么重要？
   考察点：角色 token、轮次、特殊 token、训练推理一致性。
   回答框架：模型只看 token 序列，template 决定对话结构编码，不一致会导致分布偏移。

8. SFT 的训练目标是什么？
   考察点：条件概率、next-token prediction、assistant response、cross entropy。
   回答框架：最大化给定指令和上下文下目标回答的概率，通常只对 assistant token 算交叉熵。

9. SFT 数据怎么构造成 input 和 labels？
   考察点：messages、chat template、tokenization、label mask、ignore index。
   回答框架：序列化对话，tokenize，labels 复制 input_ids，再把 system/user/padding token 置为 `-100`。

10. LoRA 和全参 SFT 有什么区别？
   考察点：参数更新范围、显存、成本、能力上限、部署。
   回答框架：全参更新所有权重，LoRA 冻结基座只训低秩 adapter，QLoRA 进一步量化基座省显存。

11. SFT 后模型能力下降怎么排查？
   考察点：灾难性遗忘、数据分布、学习率、epoch、label mask、评估。
   回答框架：查数据和训练强度，验证 template/mask，做 base capability eval，必要时混入保能力数据。

12. RLHF 的完整流程是什么？
   考察点：SFT、偏好数据、reward model、PPO、reference model、KL penalty。
   回答框架：SFT 得到初始模型，收集 chosen/rejected 偏好数据，训练 reward model，再用 PPO 优化 policy，并用 KL 约束接近 reference。

13. Reward Model 怎么训练？
   考察点：prompt+response、标量 reward、chosen/rejected、pairwise ranking loss。
   回答框架：输入回答输出分数，用 pairwise loss 让 chosen 分数高于 rejected。

14. 为什么 RLHF 需要 KL penalty？
   考察点：reference model、分布约束、reward hacking、语言质量。
   回答框架：防止 policy 为了高 reward 偏离 SFT 模型太远，降低 reward model 被 exploit 的风险。

15. RLHF 有哪些风险？
   考察点：偏好噪声、reward model 泛化、reward hacking、over-optimization、能力退化。
   回答框架：reward model 只是偏好近似，过度优化会带来高 reward 低人类偏好，需要人工和安全评估。

16. DPO 为什么不需要显式 reward model？
   考察点：preference pair、log probability、KL 约束、直接优化 policy。
   回答框架：DPO 将带 KL 约束的偏好优化转成 chosen/rejected log probability 的直接损失，因此不单独训练 reward model。

17. DPO 和 RLHF 有什么区别？
   考察点：reward model、PPO、在线采样、离线偏好优化、稳定性。
   回答框架：RLHF 是偏好数据训练 RM 再 PPO，DPO 是直接用 preference pair 优化 policy。

18. DPO 中 reference model 有什么作用？
   考察点：SFT baseline、隐式 KL、分布约束、过拟合。
   回答框架：reference 提供原模型偏好基线，约束 policy 不要在偏好数据上偏离太远。

19. DPO 有哪些风险？
   考察点：偏好数据噪声、长度偏差、reference 选择、over-refusal、能力回归。
   回答框架：DPO 工程简单但依赖数据质量，需要评估偏好、安全、事实性和原能力。

20. Reward Model 学的是什么？
   考察点：prompt+response、标量 reward、偏好排序、chosen/rejected。
   回答框架：RM 学偏好数据中的相对排序信号，近似人类偏好，但不是绝对质量或事实判断器。

21. 什么是 reward hacking？
   考察点：代理目标、RM 漏洞、Goodhart's Law、真实偏好。
   回答框架：模型找到提高 RM 分数的捷径，但输出不符合真实人类意图。

22. 如何评估 Reward Model？
   考察点：pairwise accuracy、ranking loss、margin、分领域、人工评估、分布偏移。
   回答框架：除了验证集排序准确率，还要看分领域、人类一致性、校准和 adversarial 测试。

23. 如何降低 reward hacking？
   考察点：数据质量、KL 约束、holdout、adversarial eval、人工评估。
   回答框架：用多样高质量偏好数据和约束训练，不能只看 reward score，要结合人工和安全评估。

## 部署

1. KV Cache 是什么？
   考察点：Key/Value、历史 token、自回归推理、重复计算、decode。
   回答框架：缓存每层历史 token 的 K/V，新 token 复用历史 K/V，避免每步重算完整上下文。

2. 如何降低首 token 延迟？
   考察点：prefill、prompt length、prompt cache、attention kernel、batching。
   回答框架：缩短 prompt、优化 prefill、使用 prompt cache、更快 attention kernel、合理调度。

3. vLLM 的 PagedAttention 解决了什么问题？
   考察点：KV Cache、显存碎片、block/page、continuous batching。
   回答框架：用分页思想管理不同长度请求的 KV Cache，降低碎片，提高并发和 batching 效率。

4. Temperature 是什么？
   考察点：logits scaling、分布尖锐程度、稳定性、多样性。
   回答框架：temperature 缩放 logits，低温更确定，高温更多样但风险更高。

5. Top-k 和 Top-p 有什么区别？
   考察点：固定数量、累计概率、nucleus sampling、自适应候选集合。
   回答框架：top-k 保留前 k 个，top-p 保留累计概率达到 p 的最小集合。

6. 为什么开放式对话不一定适合 beam search？
   考察点：高概率序列、低多样性、平庸回答、开放式生成。
   回答框架：beam search 偏向高概率文本，但开放式对话需要多样性和用户偏好，不只是最高概率。

7. 采样参数能解决幻觉吗？
   考察点：随机性、事实性、模型知识、RAG、工具。
   回答框架：低温能减少随机性，但不能保证事实正确，幻觉还需要检索、工具和评估治理。

8. Prefill 和 decode 有什么区别？
   考察点：prompt 处理、逐 token 生成、并行性、TTFT、TPOT。
   回答框架：prefill 并行处理完整 prompt 建 cache，decode 逐 token 串行生成并复用 cache。

9. KV Cache 显存怎么估算？
   考察点：layers、batch、seq length、kv heads、head dim、precision。
   回答框架：粗略为 `2 * L * B * T * H_kv * D_head * bytes`，长上下文和高并发时会成为瓶颈。

10. FlashAttention 解决了什么问题？
   考察点：attention matrix、HBM IO、tiling、online softmax、显存。
   回答框架：避免显式物化 `T*T` scores/probabilities，减少 HBM 读写和中间存储。

11. FlashAttention 是近似 attention 吗？
   考察点：exact attention、softmax、attention pattern、sparse/linear 对比。
   回答框架：不是，它仍计算标准 softmax attention，只改变计算顺序和内存访问。

12. 为什么 FlashAttention 更快？
   考察点：IO-aware、SRAM 复用、fused kernel、memory bandwidth。
   回答框架：通过 tiling 和融合计算减少 HBM 读写，提高片上内存复用。

13. FlashAttention 和 KV Cache 的关系是什么？
   考察点：prefill、decode、attention IO、历史 K/V。
   回答框架：FlashAttention 优化 attention kernel，KV Cache 避免 decode 重算历史 K/V；两者解决不同瓶颈。

14. 量化是什么？
   考察点：低精度、显存、带宽、数值误差、kernel。
   回答框架：用 INT8/INT4 等低精度表示权重或激活，降低显存和带宽，但可能影响质量且依赖硬件支持。

15. INT8 和 INT4 有什么区别？
   考察点：参数字节数、显存、质量损失、速度、硬件。
   回答框架：INT4 更省显存但误差更大，INT8 更稳；实际速度取决于 kernel 和系统瓶颈。

16. GPTQ 和 AWQ 的直觉是什么？
   考察点：PTQ、二阶信息、activation-aware、校准数据。
   回答框架：GPTQ 用近似二阶信息补偿量化误差，AWQ 用激活信息保护重要权重。

17. 量化后如何评估？
    考察点：perplexity、benchmark、业务评估、TTFT、TPOT、P95/P99。
    回答框架：同时评估质量、格式稳定性、安全、显存、吞吐和尾延迟，不能只看模型能否加载。

18. 为什么 KV Cache 会成为高并发推理服务的瓶颈？
    考察点：运行时状态、上下文长度、并发、显存容量、显存带宽。
    回答框架：KV Cache 是每个活跃请求独有的状态，大小随层数、上下文长度、KV head 数、head dim、精度和并发数增长；长上下文和高并发下会同时消耗显存容量和 decode 带宽。

19. 连续分配 KV Cache 有什么问题？
    考察点：内部碎片、外部碎片、变长请求、动态生成、OOM。
    回答框架：按最大长度预留会浪费未使用空间，请求动态开始结束会造成不连续空洞；即使总空闲显存足够，也可能因为连续空间不足或碎片严重影响并发。

20. PagedAttention 是新的 attention 算法吗？
    考察点：KV Cache、分页管理、block table、标准 attention、复杂度。
    回答框架：不是新的建模方法，也不改变标准 attention 数学定义；它借鉴分页思想，把逻辑连续的 KV Cache 映射到物理不连续的 blocks，主要解决 serving 中的显存碎片和动态调度问题。

21. PagedAttention 和 Continuous Batching 的关系是什么？
    考察点：动态加入、动态退出、block 分配、KV Cache 释放、调度。
    回答框架：Continuous Batching 需要请求在 decode step 间动态加入和退出，PagedAttention 提供 block 级 KV Cache 分配、追加和释放能力，是高效动态调度的重要基础。

22. block size 在 PagedAttention 中如何权衡？
    考察点：内部碎片、block table、访存效率、kernel、管理开销。
    回答框架：小 block 降低碎片但增加映射和管理开销，也可能让访存更碎；大 block 管理更简单、访存更友好，但短请求和尾部浪费更大。

23. 为什么 LLM serving 需要 Continuous Batching？
    考察点：变长输出、自回归生成、decode iteration、GPU 利用率、排队延迟。
    回答框架：LLM 请求输出长度不同，静态 batch 会让短请求等长请求，新请求也要等当前 batch 完成；Continuous Batching 在每个 decode iteration 动态移除完成请求并加入新请求，提高 GPU 利用率并减少排队浪费。

24. Continuous Batching 和普通 dynamic batching 有什么区别？
    考察点：请求开始前 batching、生成中 batching、iteration-level、in-flight batching。
    回答框架：普通 dynamic batching 多是在请求进入模型前凑批；Continuous Batching 是在自回归生成过程中持续更新 batch，允许请求在 decode 过程中加入和退出。

25. 为什么 prefill 和 decode 混合调度很难？
    考察点：compute-bound、memory-bandwidth-bound、TTFT、TPOT、token budget。
    回答框架：prefill 输入 token 多、偏计算密集，decode 单步小、偏带宽和 KV Cache 访问；过多 prefill 会让已有流式输出变慢，过度优先 decode 又会让新请求 TTFT 变差。

26. Chunked prefill 解决什么问题？
    考察点：长 prompt、调度粒度、decode 卡顿、token budget、TTFT。
    回答框架：把超长 prompt 的 prefill 拆成多个 chunk，穿插在 decode iteration 中执行，避免长 prefill 独占 GPU，降低对已有请求 TPOT 的冲击。

27. LLM serving 调度器应该控制哪些预算？
    考察点：token budget、KV Cache budget、running batch size、优先级、SLA。
    回答框架：不能只限制请求数，还要控制每轮 token 数、KV Cache block 占用、活跃 batch 大小和请求优先级，综合优化 TTFT、TPOT、吞吐和尾延迟。

28. Speculative Decoding 的核心思想是什么？
    考察点：draft model、target model、候选 token、批量验证、decode 加速。
    回答框架：用小而快的 draft model 先生成多个候选 token，再用 target model 一次 forward 验证；如果候选被接受，一次大模型调用可以推进多个 token。

29. 为什么 target model 可以一次验证多个候选 token？
    考察点：teacher forcing、causal mask、候选序列、并行 logits。
    回答框架：候选 token 已经给定后，可以把它们作为输入序列送入 target model，在 causal mask 下并行计算每个位置 logits，不违反自回归约束。

30. Speculative Decoding 会改变大模型输出分布吗？
    考察点：speculative sampling、接受-拒绝采样、修正分布、无偏。
    回答框架：严格实现不会改变 target model 分布；它用 `min(1, p/q)` 类接受规则和修正分布保证最终采样等价于直接从 target model 采样。

31. Speculative Decoding 的加速比由什么决定？
    考察点：接受率、draft 速度、draft 长度、target 验证效率、任务分布。
    回答框架：接受率越高、draft 越快、target 一次验证越高效，收益越大；如果 draft 太慢、太弱或拒绝率高，可能没有收益。

32. Speculative Decoding 和模型蒸馏有什么区别？
    考察点：小模型替代、提案模型、质量上限、target 分布。
    回答框架：蒸馏是用小模型近似或替代大模型；Speculative Decoding 中小模型只提案，最终由 target model 验证和修正，目标是保持大模型分布同时加速。

33. Medusa 的核心思想是什么？
    考察点：multi-token prediction、Medusa heads、候选树、target 验证。
    回答框架：在原 LLM 顶部增加多个轻量 prediction heads，从当前 hidden state 预测未来多个 token 候选，再通过候选树和原模型验证接受其中一段。

34. 为什么 Medusa 不能直接接受多个 head 的输出？
    考察点：自回归依赖、未来 token 条件化、一致性、验证。
    回答框架：未来 token 之间有条件依赖，第 2、第 3 个 token 应该依赖前面实际接受的 token；多个 head 的输出只是候选，直接拼接可能不符合原模型分布。

35. EAGLE 和 Medusa 的区别是什么？
    考察点：token-level proposal、feature-level proposal、hidden state、接受率。
    回答框架：Medusa 更像用多个 heads 直接预测未来 token；EAGLE 更强调预测未来 hidden state 或 feature，再映射到 token 候选，希望 proposal 更接近原模型内部状态演化。

36. Medusa/EAGLE 和 Speculative Decoding 的关系是什么？
    考察点：proposal、verification、draft model、heads、feature predictor。
    回答框架：它们都可以放在 speculative decoding 的 proposal-verification 框架下；普通方法用独立 draft model，Medusa 用额外 heads，EAGLE 用特征预测模块，最终仍需要 target model 验证。

37. 多 token 预测上线时为什么不能只看接受率？
    考察点：proposal 成本、tree attention、KV Cache、batching、尾延迟。
    回答框架：接受率只表示候选质量，还要看候选生成成本、验证成本、显存占用、batch 调度、P95/P99 延迟和不同任务分布下的稳定性。

38. PTQ 和 QAT 有什么区别？
    考察点：训练后量化、量化感知训练、校准、fake quantization、低 bit。
    回答框架：PTQ 在训练后用校准数据量化，成本低但低 bit 风险大；QAT 在训练或微调中模拟量化误差，让模型适应低精度，成本更高但质量更稳。

39. 为什么 LLM 推理常用 weight-only quantization？
    考察点：权重显存、显存带宽、decode、激活量化风险、kernel。
    回答框架：decode 阶段频繁读取权重，weight-only 能降低权重显存和带宽，同时保留激活为 FP16/BF16，减少激活量化导致的质量风险。

40. GPTQ 的直觉是什么？
    考察点：PTQ、校准激活、近似二阶信息、误差补偿、线性层输出。
    回答框架：GPTQ 利用校准激活和近似二阶信息，按顺序量化权重并补偿误差，目标是让量化后线性层输出接近原输出。

41. AWQ 的直觉是什么？
    考察点：activation-aware、重要权重、outlier、通道保护。
    回答框架：AWQ 认为与大激活相乘、对输出影响大的权重更重要，因此利用激活统计保护重要权重或通道，降低量化误差。

42. INT4 模型一定比 FP16 快吗？
    考察点：显存、带宽、kernel、反量化、硬件支持、瓶颈分析。
    回答框架：不一定。INT4 通常省显存，但速度取决于硬件和 kernel 是否高效、反量化开销以及系统瓶颈是否在权重带宽。

43. 量化模型上线前如何评估？
    考察点：质量、格式、安全、业务样本、TTFT、TPOT、P95/P99、分桶。
    回答框架：和 FP16 baseline 对比，评估 perplexity、benchmark、业务样本、格式稳定性、安全、显存、吞吐、TTFT、TPOT 和尾延迟，并按任务类型分桶。

44. KV Cache 量化和权重量化有什么区别？
    考察点：静态参数、动态运行时状态、attention、PagedAttention、长上下文。
    回答框架：权重是静态参数，可离线量化；KV Cache 是每个请求动态生成并随上下文增长的运行时状态，量化误差会直接影响 attention 和后续 decode。

45. K cache 和 V cache 的量化误差分别影响什么？
    考察点：QK score、attention 分布、value 聚合、内容表示。
    回答框架：K 误差会改变 attention score 和关注位置，可能找错历史 token；V 误差会影响被聚合的内容表示，可能取回的信息更粗糙。

46. KV Cache 量化为什么对长上下文更敏感？
    考察点：活跃 token、远距离检索、attention 排序、误差累积。
    回答框架：长上下文中 attention 候选更多，远距离关键信息更难检索，K/V 小误差可能影响关键 token 排序和后续生成，短上下文评估无法覆盖这些风险。

47. KV Cache 量化粒度如何权衡？
    考察点：per-token、per-channel、per-head、group-wise、scale 元数据、kernel。
    回答框架：细粒度 scale 通常误差更小，但元数据、访存和 kernel 复杂度更高；group-wise 或 per-head 常作为质量与效率之间的折中。

48. KV Cache INT4 可以直接上线吗？
    考察点：INT8、INT4、长上下文、RAG、代码、数学、质量风险。
    回答框架：不能默认可以。INT4 显存收益大但误差更高，必须和 FP16/INT8 baseline 在长上下文、RAG、代码、数学、多轮对话和尾延迟上做分桶评估。

49. 设计 LLM serving 系统时核心模块有哪些？
    考察点：gateway、router、scheduler、GPU worker、KV manager、streamer、observability。
    回答框架：核心模块包括 API gateway、鉴权限流、router、scheduler、tokenizer、GPU workers、KV Cache manager、batching manager、streamer、监控和安全治理。

50. 为什么 LLM 负载均衡不能只按请求数？
    考察点：input tokens、output tokens、KV Cache、长上下文、资源估算。
    回答框架：LLM 请求成本由 token 数、上下文长度、输出长度和 KV Cache 占用决定；两个请求成本可能差几个数量级，所以路由要按 token 和资源预算。

51. LLM scheduler 要优化哪些指标？
    考察点：TTFT、TPOT、吞吐、P95/P99、GPU 利用率、KV budget、公平性。
    回答框架：scheduler 要在低 TTFT、低 TPOT、高吞吐、低尾延迟、GPU 利用率、KV Cache 容量、公平性和优先级之间权衡。

52. 线上 TTFT 变高怎么排查？
    考察点：queue、prefill、prompt length、tokenizer、router、prefix cache。
    回答框架：先看排队和路由负载，再看 prompt 长度、prefill tokens/s、tokenizer CPU、prefix cache 命中率和 prefill token budget。

53. tokens/s 高但用户觉得卡，可能是什么原因？
    考察点：平均值、P99、TTFT、流式稳定性、网络、客户端。
    回答框架：平均 tokens/s 高不代表体验好，可能 TTFT 高、P99 差、stream token 间隔不稳定、长请求阻塞短请求、网络 backpressure 或客户端渲染慢。

54. LLM serving 如何做容量规划？
    考察点：token 分布、并发、TTFT、TPOT、KV Cache、GPU 容量、冗余。
    回答框架：统计 input/output token 分布和峰值并发，结合目标 TTFT/TPOT、KV Cache 显存、每 GPU tokens/s 和最大并发估算 GPU 数，并加冗余和故障切换容量。

55. 训练成本怎么粗略估算？
    考察点：参数量、训练 token、FLOPs、MFU、GPU 小时、实验成本。
    回答框架：decoder-only Transformer 训练 FLOPs 可粗略估算为 `6 * 参数量 * 训练 token 数`，再除以集群有效算力得到时间；真实成本还包括实验、失败重跑、评估、数据和人力。

56. 推理成本由哪些因素决定？
    考察点：input token、output token、模型大小、上下文、并发、KV Cache、SLA。
    回答框架：推理成本由输入输出 token、模型规模、上下文长度、并发、TTFT/TPOT SLA、KV Cache、硬件、推理引擎、量化和冗余共同决定。

57. 为什么 output token 通常比 input token 贵？
    考察点：prefill、decode、自回归、GPU 占用、KV Cache。
    回答框架：input token 可在 prefill 阶段并行处理；output token 需要逐 token decode，每个 token 都要占用模型 forward、读取权重和 KV Cache，因此边际成本更高。

58. 如何降低 LLM 推理成本？
    考察点：小模型、量化、batching、cache、prompt 压缩、模型路由、容量规划。
    回答框架：可从模型、引擎、产品和系统四层优化：小模型/蒸馏/量化，PagedAttention/continuous batching/speculative decoding，压缩 prompt/限制 max_tokens/cache/模型路由，峰谷调度和容量规划。

59. 为什么最便宜模型不一定最划算？
    考察点：成功率、重试、人工修正、业务价值、单位有效任务成本。
    回答框架：要看 cost per successful answer。如果便宜模型成功率低、重试或人工介入多，最终单位有效任务成本可能更高。

60. 什么时候自部署比调用 API 更划算？
    考察点：流量规模、合规、延迟、定制化、运维能力、API 单价。
    回答框架：小流量和快速试错通常 API 更划算；大流量、任务稳定、强合规、低延迟或深度定制时，自部署可能更划算，但要计入研发和运维成本。

61. 端侧部署和云端部署有什么区别？
    考察点：算力、内存、网络、隐私、成本、更新、质量。
    回答框架：云端算力强、模型大、统一更新方便，但依赖网络且有隐私和持续成本；端侧资源受限，但低延迟、可离线、隐私更好、云端边际成本低。

62. 为什么小模型仍然重要？
    考察点：高频简单任务、成本、隐私、离线、前置模块。
    回答框架：很多任务不需要大模型，例如意图识别、分类、安全过滤和工具路由；小模型延迟低、成本低、可离线，适合作为端侧能力或大模型前置模块。

63. 端侧部署最重要的约束有哪些？
    考察点：内存、算力、功耗、发热、包体、硬件支持、系统碎片化。
    回答框架：端侧要同时考虑内存、算力、功耗、发热、模型包体、冷启动、硬件算子支持、设备碎片化、崩溃率和更新难度。

64. 端侧模型常见压缩手段有哪些？
    考察点：量化、蒸馏、剪枝、LoRA merge、高效架构。
    回答框架：常见方法包括 INT8/INT4 量化、知识蒸馏、结构化剪枝、LoRA 合并和选择更小更高效的架构，最终要结合端侧 runtime 和质量评估。

65. 什么是端云协同？
    考察点：任务路由、隐私、网络、设备能力、成本、质量。
    回答框架：端云协同是简单、隐私、低延迟任务端侧处理，复杂高质量任务云端处理；路由根据任务难度、隐私等级、网络状态、设备能力、成本和置信度决定。

66. 端侧模型上线前要评估什么？
    考察点：设备分层、冷启动、tokens/s、内存、功耗、发热、崩溃率、更新。
    回答框架：要按设备档位评估质量、冷启动、首 token、tokens/s、峰值内存、电量、发热、降频、崩溃率、离线可用、模型更新和回滚。

## 长上下文、RAG 与 Agent

1. 长上下文能力为什么不只是把窗口调大？
   考察点：位置编码、训练数据、attention、KV Cache、评估。
   回答框架：调大窗口只能让输入进来，不代表模型会用；还需要位置外推、长上下文训练、远距离依赖数据、推理系统和长上下文评估共同支持。

2. RoPE scaling 解决什么问题？
   考察点：RoPE、位置外推、频率、长位置、继续训练。
   回答框架：通过调整位置映射或频率，让超过训练窗口的位置模式更平滑、更接近训练分布，缓解 RoPE 外推失败。

3. 什么是 lost in the middle？
   考察点：长上下文、位置偏置、中间信息、检索鲁棒性。
   回答框架：模型在长上下文中更容易使用开头和结尾信息，忽略中间位置的信息，说明长窗口不等于均匀使用所有位置。

4. 如何把 4k context 模型扩展到 128k？
   考察点：RoPE scaling、continued pretraining、长数据、SFT、评估、KV Cache。
   回答框架：选择位置扩展方法，构造长上下文数据分阶段继续训练，混合短文本避免退化，再做长上下文 SFT 和分桶评估，部署时处理 KV Cache、PagedAttention 和成本。

5. 长上下文和 RAG 是替代关系吗？
   考察点：检索、context window、知识库、综合推理。
   回答框架：不是。RAG 从大规模知识库中召回相关内容，长上下文模型读入更多候选并综合推理，二者通常互补。

6. 长上下文训练数据应该怎么构造？
   考察点：长文档、远距离依赖、多证据、代码仓库、合成数据。
   回答框架：需要有结构和任务信号的数据，如书籍、论文、代码仓库、多文档问答、长对话和合成远距离依赖任务，不能只是无关短文本拼接。

7. 为什么长上下文评估不能只看最大窗口？
   考察点：context length、位置鲁棒性、任务类型、成本、延迟。
   回答框架：最大窗口只说明能接收这么长输入，不代表能可靠使用所有位置的信息；还要按证据位置、任务类型、干扰项、短任务回归和成本延迟评估。

8. Needle-in-a-haystack 有什么价值和局限？
   考察点：sanity check、单事实检索、干扰、多证据、RAG。
   回答框架：needle 简单可控，适合测试基本检索和位置鲁棒性；但它通常只测单事实，干扰弱，不能代表多证据综合、RAG、代码和真实文档理解。

9. 如何评估 lost in the middle？
   考察点：证据位置、分桶、准确率曲线、位置偏置。
   回答框架：把同一证据放在开头、25%、50%、75%、结尾等位置，比较准确率；如果中间明显更差，就说明存在 lost in the middle。

10. 长上下文评估为什么要测短任务回归？
    考察点：RoPE scaling、继续训练、短任务退化、产品能力。
    回答框架：长上下文扩展可能改变位置分布和训练数据分布，导致聊天、代码、数学、安全和格式输出退化，所以必须保留短任务回归。

11. RAG 长上下文系统如何做错误归因？
    考察点：retrieval、rerank、context、reader、citation、hallucination。
    回答框架：答案错时要判断是检索未召回、reranker 排错、context 拼接问题、reader 没用好上下文、生成幻觉还是引用错误。

12. 如何设计可信长上下文 benchmark？
    考察点：长度分桶、位置分桶、任务分桶、干扰、引用、成本、污染。
    回答框架：覆盖不同长度、证据位置、任务类型、干扰项、多证据综合、引用检查、短任务回归和成本延迟，并使用私有或动态样本降低污染。

13. RAG 的核心思想是什么？
    考察点：retrieval、generation、grounding、外部知识、幻觉。
    回答框架：先从外部知识库检索相关资料，再让模型基于资料回答，把生成 grounding 到可更新、可追溯的证据上。

14. 一个 RAG 系统包含哪些模块？
    考察点：解析、chunking、metadata、embedding、retrieval、rerank、context、citation。
    回答框架：离线侧包括文档解析、清洗、chunking、metadata、embedding 和索引；在线侧包括 query rewrite、retrieval、rerank、context construction、generation、citation 和评估。

15. Chunking 为什么重要？
    考察点：检索粒度、语义完整、噪声、跨 chunk、token budget。
    回答框架：chunk 决定检索粒度，太大噪声多且成本高，太小语义不完整；好的 chunk 要语义完整且大小适中。

16. 为什么 RAG 不等于向量数据库加 LLM？
    考察点：完整链路、权限、hybrid retrieval、reranker、context、评估。
    回答框架：向量库只是检索组件，RAG 还包括文档处理、chunking、metadata、权限、重排、上下文构造、引用、评估和错误归因。

17. RAG 答案错了怎么排查？
    考察点：文档库、解析、chunk、retrieval、rerank、context、generation、citation。
    回答框架：先确认正确文档是否在库里，再看 chunk、召回、重排、context、生成和引用各环节，定位是检索问题还是生成问题。

18. 企业 RAG 最容易出什么事故？
    考察点：权限、过期文档、表格解析、引用、幻觉、成本延迟。
    回答框架：常见事故包括权限泄露、过期文档引用、解析错误、检索漏召回、引用编造、无依据回答和 latency/cost 超标。

19. Embedding model 在 RAG 中起什么作用？
    考察点：query、document chunk、向量空间、语义相似、召回。
    回答框架：把 query 和 document chunk 映射到同一向量空间，使语义相关内容距离更近，决定第一阶段能否召回相关证据。

20. 双塔检索和 reranker 有什么区别？
    考察点：bi-encoder、cross-encoder、离线预计算、召回、精排。
    回答框架：双塔分别编码 query 和 document，文档向量可离线预计算，速度快适合召回；reranker 让 query 和 document 交互，排序更准但更慢。

21. In-batch negatives 是什么？
    考察点：contrastive learning、负样本、batch、false negative。
    回答框架：一个 batch 中其他 query 的正文档可作为当前 query 的负样本，能高效构造负样本，但要小心其实相关的 false negatives。

22. 为什么向量检索需要 ANN？
    考察点：大规模向量库、暴力搜索、HNSW、IVF、PQ、召回延迟权衡。
    回答框架：大规模向量库暴力相似度搜索成本高，ANN 用近似索引在召回率、延迟和内存之间折中。

23. 如何评估向量检索质量？
    考察点：recall@k、precision@k、MRR、nDCG、latency、下游质量。
    回答框架：用标注 query-doc 数据评估 recall@k、precision@k、MRR、nDCG，同时看检索延迟、索引成本和下游 RAG 答案质量。

24. RAG 检索不到正确文档怎么排查？
    考察点：入库、chunk、metadata、embedding、top-k、ANN、BM25、query rewrite。
    回答框架：先看文档是否入库、chunk 是否合理、metadata 是否误过滤，再看 embedding 模型、top-k、ANN 参数、领域适配、BM25 和 query rewrite。

25. 为什么 RAG 需要 reranker？
    考察点：embedding 召回、cross-encoder、精排、context precision、噪声。
    回答框架：embedding retriever 快但交互弱，可能召回相似但不回答问题的 chunk；reranker 用更强交互模型精排，提高最终 context 质量。

26. Bi-encoder 和 cross-encoder 有什么区别？
    考察点：离线预计算、token-level 交互、召回、精排、延迟。
    回答框架：bi-encoder 分别编码 query/doc，快且可预计算；cross-encoder 拼接 query/doc 共同编码，交互充分、排序更准但更慢。

27. Reranker 怎么训练？
    考察点：pointwise、pairwise、listwise、hard negatives、相关性标注。
    回答框架：用 query-document 相关性数据训练，可做 pointwise 打分、pairwise 偏好排序或 listwise 排序，hard negatives 对区分相似干扰文档很重要。

28. Reranker 怎么评估？
    考察点：MRR、nDCG、precision@k、hit@k、answer quality、latency。
    回答框架：看排序指标和下游 RAG 指标，包括 MRR、nDCG@k、precision@k、hit@k、答案正确率、引用准确率、延迟和成本。

29. RAG 的 top-k 应该怎么选？
    考察点：retriever top-k、reranker top-k、召回、噪声、成本、token budget。
    回答框架：retriever top-k 要保证召回，reranker final top-k 要控制 prompt 噪声和成本，需要结合 recall、precision、答案质量和延迟调参。

30. 检索到了文档但回答仍然错，怎么排查？
    考察点：rerank、context construction、去重、prompt、reader、citation。
    回答框架：看正确 chunk 是否排到前面、最终 context 是否包含完整证据、是否被噪声淹没、prompt 是否约束基于证据回答、模型是否忽略证据或引用错误。

31. RAG 能彻底解决幻觉吗？
    考察点：retrieval、context、generation、citation、grounding。
    回答框架：不能。RAG 降低幻觉，但检索没召回、上下文构造错误、模型不使用证据或引用错误时仍会幻觉。

32. Faithfulness 和 correctness 有什么区别？
    考察点：事实正确、context 支持、groundedness、可追溯。
    回答框架：correctness 看答案是否事实正确，faithfulness 看答案是否被给定 context 支持；RAG 中答案即使碰巧正确，但没有证据支持也不 faithful。

33. Attribution 在 RAG 中是什么意思？
    考察点：claim、evidence、citation、unsupported claim。
    回答框架：attribution 是把回答中的关键声明对应到具体证据来源，确保每个重要 claim 都能被引用片段支持。

34. 如何评估 citation accuracy？
    考察点：引用存在、引用相关、引用支持、claim-level 检查。
    回答框架：先检查引用是否真实存在，再检查引用片段是否包含相关证据，最后判断引用是否足以支持回答中的关键声明。

35. 资料不足时 RAG 应该怎么做？
    考察点：abstention、unanswerable、拒答、风险控制。
    回答框架：应该明确说明根据现有资料无法确定，而不是凭常识补全；评估中要包含 unanswerable questions 检查 abstention accuracy。

36. 如何治理 RAG 系统中的幻觉？
    考察点：召回、rerank、prompt、引用、faithfulness、反馈闭环。
    回答框架：提高检索召回和 context precision，要求基于证据回答和资料不足拒答，做 claim-level attribution 检查，并用失败样本反馈优化检索、重排和 prompt。

37. Tool use 和 function calling 的核心思想是什么？
    考察点：工具调用、结构化输出、系统执行、外部能力、安全边界。
    回答框架：模型在需要外部信息或能力时输出结构化工具调用，由系统执行真实工具，再把结果返回模型生成最终答案。

38. 工具 schema 为什么重要？
    考察点：工具名、description、参数类型、必填字段、枚举、格式。
    回答框架：schema 告诉模型有哪些工具和参数，清晰 schema 能降低选错工具、漏参数、编造参数和格式错误。

39. 为什么 function calling 不只是输出 JSON？
    考察点：参数校验、权限、执行、错误处理、结果回填、审计。
    回答框架：JSON 只是交互格式，完整系统还要做参数校验、权限检查、工具执行、错误处理、结果回填、日志审计和最终回答。

40. 工具调用如何保证安全？
    考察点：权限控制、参数校验、二次确认、只读/写操作、审计、prompt injection。
    回答框架：系统要做工具级和参数级权限控制，高风险操作二次确认，区分只读和写操作，限制频率，记录审计，并防 prompt injection 越权调用。

41. 如何评估 function calling？
    考察点：tool selection、argument accuracy、execution success、final answer、safety。
    回答框架：评估工具选择准确率、参数准确率、执行成功率、最终答案正确率、安全违规率、漏调用/误调用率、多步成功率、延迟和成本。

42. 工具调用失败时模型应该怎么做？
    考察点：结构化错误、重试、追问、失败解释、禁止编造。
    回答框架：系统返回结构化错误，模型应解释失败、请求补充信息或建议重试，不能编造工具结果。

## 多模态

1. CLIP 如何训练？
2. VLM 如何把图像接入 LLM？
3. Diffusion Model 的基本原理是什么？

## 开放研究题

1. 如何判断模型是否真的具备推理能力？
2. 如何设计下一代多模态模型？
3. 如何平衡 safety 和 helpfulness？
4. 大模型为什么会幻觉？
   考察点：next-token prediction、数据错误、知识过时、prompt 不足、解码、缺少验证。
   回答框架：语言建模目标不是事实验证，模型会生成高概率文本而不保证真实。

5. 如何评估一个 LLM？
   考察点：benchmark、human eval、LLM judge、事实性、安全、污染检测。
   回答框架：多维度评估知识、推理、代码、长上下文、指令遵循、事实性和安全，结合自动和人工评估。

6. 如何降低幻觉？
   考察点：数据质量、RAG、工具、verifier、grounding、不确定性表达。
   回答框架：不能只调 decoding，要结合训练、检索、工具、验证器、引用和专门评估。

7. Jailbreak 和 prompt injection 有什么区别？
   考察点：用户提示、外部内容、RAG、工具、安全边界。
   回答框架：jailbreak 是直接绕过安全提示，prompt injection 多来自不可信外部内容中的恶意指令。

8. 如何判断模型是否真的具备推理能力？
   考察点：泛化、扰动、组合测试、CoT、verifier、污染。
   回答框架：不能只看 benchmark，要看分布外、扰动、组合泛化和中间过程可验证性。

9. 如何设计下一代多模态模型？
   考察点：跨模态对齐、统一表示、数据、评估、工具、安全。
   回答框架：从架构、数据、训练、评估和安全五个维度设计，而不是只拼接图像 token。

10. Test-time compute 为什么重要？
   考察点：self-consistency、verifier、search、动态预算、成本。
   回答框架：推理时增加搜索和验证可以提升复杂任务表现，但要权衡延迟和成本。

## 模拟面试与复盘

1. 请用 3 分钟介绍你自己，并说明为什么适合大模型算法岗。
   考察点：背景总结、岗位匹配、表达结构、重点选择。
   回答框架：背景、相关经历、核心能力、短板补齐、岗位动机。

2. 讲一个你最熟悉的大模型相关项目。
   考察点：问题定义、数据、方法、实验、失败分析。
   回答框架：先讲目标和约束，再讲方案、结果、失败案例和下一步改进。

3. 面试官连续追问你不确定的问题时，你怎么回答？
   考察点：诚实性、推理能力、边界意识。
   回答框架：承认不确定，回到机制分析，给出可能假设和验证方法，不硬编。

4. 如何复盘一次失败的面试？
   考察点：错误分类、知识补齐、表达修正、下一轮验证。
   回答框架：记录原题、原回答、错误原因、正确答案、背后概念和下一次改法。

5. 如何判断自己是否准备好面试？
   考察点：自测体系、mock interview、项目深挖、开放题表达。
   回答框架：能稳定讲核心概念、项目、系统 trade-off 和开放研究题，并能应对追问。
