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
169. 分析 prompt injection 如何诱导工具越权调用，并给出 3 个防护策略。
170. 用 3 分钟回答“如何设计一个能调用企业 API 的 LLM 助手”。

## 阶段 5：研究与面试验收

1. 精读 10 篇论文。
2. 完成一个项目讲稿。
3. 完成 3 次 mock interview。
4. 能回答开放研究题。
5. 给出 5 个不同类型的幻觉例子。
6. 设计一个事实性评估集，说明数据来源和评分方式。
7. 解释 benchmark contamination 为什么会误导模型比较。
8. 写 3 个 jailbreak 测试样例。
9. 设计一个包含 helpfulness、honesty、harmlessness 的评估表。
10. 用自己的话回答“如何判断模型是否真的具备推理能力”。
11. 设计一个多模态模型的研究 roadmap。
12. 说出 3 个平衡 safety 和 helpfulness 的具体策略。
13. 解释 test-time compute 为什么会成为重要研究方向。
14. 讨论合成数据的一个优势和一个风险。
15. 录音模拟一次 3 分钟自我介绍，并写下 3 个可改进点。
16. 选一个项目，用“问题、数据、方法、实验、失败、改进”六段式讲一遍。
17. 从题库中随机抽 10 道题，要求每题用 2 分钟回答。
18. 对一次 mock interview 做复盘，记录原题、原回答、错误原因和修正答案。
19. 准备 10 个必须能讲清的问题，并逐个写出 5 句话版本答案。
20. 做一次 45 分钟完整模拟面试：自我介绍、项目、基础、系统、开放题各一轮。
