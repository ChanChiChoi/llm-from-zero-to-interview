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

13. 线性代数在大模型里主要出现在哪里？
   考察点：embedding、linear layer、attention、LoRA、RAG 检索、gradient norm。
   回答框架：token hidden state 是向量，linear layer 是矩阵乘法，attention 是 `QK^T` 相似度和加权求和，LoRA 是低秩更新，RAG 检索是向量空间最近邻。

14. Dot product、cosine similarity 和 L2 distance 有什么关系？
   考察点：方向、长度、L2 normalize、归一化后排序。
   回答框架：dot product 同时受方向和长度影响；cosine 除以范数后只看方向；向量归一化后 dot product 等于 cosine，归一化 L2 距离也和 cosine 单调相关。

15. 为什么 attention score 的后两维是 `T x T`？
   考察点：`Q [B,H,T,d_h]`、`K^T [B,H,d_h,T]`、矩阵乘法 shape。
   回答框架：每个 query token 都和每个 key token 做一次内积，所以 `QK^T` 的后两维是 query length by key length，也就是 `T x T`。

16. SVD 和低秩近似如何帮助理解 LoRA？
   考察点：`W=U Sigma V^T`、rank-r 近似、`Delta W=AB`、参数量。
   回答框架：SVD 说明矩阵信息可能集中在少数重要方向；LoRA 假设微调增量位于低秩子空间，用两个小矩阵表示，参数量从 `d_in*d_out` 变为 `r*(d_in+d_out)`。

17. 为什么 embedding 检索常做 L2 normalize？
   考察点：向量方向、向量长度、cosine similarity、检索排序。
   回答框架：如果不归一化，向量长度可能主导 dot product；归一化后 dot product 等于 cosine，更关注语义方向，检索排序也更稳定。

18. 看到一个线性层 `X [B,T,d_in] @ W [d_in,d_out]`，你怎么推 shape 和参数量？
   考察点：矩阵乘法中间维度、batch/sequence 广播、bias、参数量。
   回答框架：最后一维 `d_in` 和权重第一维匹配，输出是 `[B,T,d_out]`；忽略 bias 时权重参数量是 `d_in*d_out`，有 bias 再加 `d_out`。

19. 为什么语言模型本质上是在做条件概率建模？
   考察点：随机变量、上下文、词表分布、`P(x_t | x_{1:t-1})`。
   回答框架：给定前文后，下一个 token 是离散随机变量，模型输出整个词表上的条件概率分布；训练就是提高真实 token 的条件概率。

20. 概率链式法则和 next-token prediction 有什么关系？
   考察点：联合概率分解、自回归建模、teacher forcing。
   回答框架：任意序列概率都能拆成逐步条件概率乘积；GPT 从左到右建模每一步 `P(x_t | x_{1:t-1})`，所以 next-token prediction 可以学习整段文本分布。

21. 为什么训练时优化 log probability 而不是原始概率乘积？
   考察点：概率连乘下溢、log sum、NLL、cross entropy。
   回答框架：长序列概率连乘会非常小，数值不稳定；取 log 后乘积变求和，最大化 log likelihood 等价于最小化 NLL，在 one-hot 标签下就是交叉熵。

22. Perplexity 和 NLL 有什么关系？
   考察点：平均 token NLL、自然对数、指数形式、评估局限。
   回答框架：如果 loss 是平均 token NLL 且使用自然对数，`PPL=exp(loss)`；PPL 衡量模型给真实文本的概率，不直接代表事实性、安全性或用户偏好。

23. Temperature、top-k 和 top-p 分别在概率分布上做了什么？
   考察点：softmax 温度、候选截断、累计概率、采样多样性。
   回答框架：temperature 调整分布尖锐程度；top-k 固定保留概率最高的 k 个候选；top-p 保留累计概率达到阈值的最小候选集合，然后在候选集合内重新归一化采样。

24. 为什么概率高不等于事实正确？
   考察点：训练分布、语言似然、truthfulness、RAG / verifier。
   回答框架：语言模型优化的是 token 条件概率，不是事实验证；训练数据错误、上下文不足或偏好优化都可能让流畅但错误的文本概率较高，需要检索、工具验证、引用约束和不确定性表达。

25. 熵、交叉熵和 KL 散度分别回答什么问题？
   考察点：分布不确定性、编码代价、分布差异。
   回答框架：熵衡量真实分布自身的不确定性；交叉熵衡量用模型分布解释真实分布的平均代价；KL 衡量用模型分布替代真实分布多付出的额外代价。

26. 为什么 `H(p,q)=H(p)+KL(p||q)` 能解释语言模型训练目标？
   考察点：真实分布固定、交叉熵最小化、KL 最小化。
   回答框架：训练数据分布固定时 `H(p)` 对模型是常数，最小化交叉熵就等价于最小化真实分布到模型分布的 KL，让模型更接近数据分布。

27. Perplexity 为什么不能跨 tokenizer 随便比较？
   考察点：token 粒度、平均 token NLL、验证集和截断方式。
   回答框架：PPL 是平均 token NLL 的指数，tokenizer 不同会改变 token 数和 token 难度；只有 tokenizer、语料、上下文截断和 loss 口径一致时，PPL 才更可比。

28. KL 散度为什么不是普通距离？
   考察点：非对称、方向含义、support mismatch。
   回答框架：KL 非负但不对称，`KL(p||q)` 和 `KL(q||p)` 优化偏好不同；如果 `p` 有概率而 `q` 给 0，KL 会发散，所以它不是普通几何距离。

29. 互信息如何帮助理解 RAG 和多模态对齐？
   考察点：条件熵、共享信息、证据质量。
   回答框架：互信息衡量知道一个变量后能减少另一个变量多少不确定性；RAG 的好证据应降低答案不确定性，多模态对齐希望图像和文本表示共享语义信息。

30. 为什么 DPO 的 reference model 可以理解成一种分布锚点？
   考察点：chosen/rejected、log probability ratio、隐式 KL 约束。
   回答框架：DPO 比较 policy 相对 reference 对 chosen 和 rejected 的偏好变化，reference 提供“偏离原模型多少”的参照，避免 policy 只靠偏好数据无限漂移。

31. Adam 的一阶矩和二阶矩分别解决什么问题？
   考察点：momentum、自适应缩放、bias correction。
   回答框架：一阶矩是梯度移动平均，用来平滑更新方向；二阶矩是梯度平方移动平均，用来估计每个方向的梯度尺度并调整步长。

32. AdamW 为什么比 Adam 加 L2 更适合做 weight decay？
   考察点：decoupled weight decay、自适应学习率缩放、正则化语义。
   回答框架：Adam + L2 会把衰减项加进梯度并被 Adam 的二阶矩缩放；AdamW 把衰减项单独作用在参数上，避免 weight decay 被自适应缩放扭曲。

33. Warmup 和 cosine decay 分别解决什么问题？
   考察点：训练早期不稳定、Adam 状态、后期收敛。
   回答框架：warmup 让学习率从小升到峰值，缓解早期激活、梯度和优化器状态不稳定；cosine decay 后期平滑降低学习率，减少震荡并利于收敛。

34. Global batch size 如何计算，为什么它影响优化？
   考察点：micro batch、gradient accumulation、data parallel、梯度噪声。
   回答框架：`global_batch = micro_batch * accumulation_steps * data_parallel_size`；它影响梯度噪声、更新步数、吞吐和 learning rate 设置。

35. Gradient clipping 的公式直觉是什么，为什么它不能根治 loss spike？
   考察点：global norm、缩放系数、根因排查。
   回答框架：如果梯度范数超过阈值，就按比例缩小所有梯度，使全局范数不超过阈值；它只能限制异常更新，不能修复错误数据、loss mask、过大学习率或 optimizer state 恢复错误。

36. Hessian 和曲率直觉如何解释学习率敏感性？
   考察点：二阶近似、高曲率方向、震荡。
   回答框架：Hessian 描述 loss surface 的曲率；高曲率方向对参数变化更敏感，如果学习率过大，更新容易在这些方向来回震荡甚至发散。

37. Checkpoint resume 后出现 loss spike，你会优先检查什么？
   考察点：optimizer state、scheduler state、scaler、random state、数据 batch。
   回答框架：先确认 model weights、AdamW 一二阶矩、scheduler step、mixed precision scaler、gradient accumulation、random seed 和 dataloader 状态是否完整恢复，再看 spike step 的 batch 和 grad norm。

38. Rank-r SVD 近似的误差怎么看？
   考察点：奇异值、Frobenius norm、能量保留率。
   回答框架：保留前 `r` 个最大奇异值时，Frobenius 误差来自被丢弃奇异值平方和；能量保留率越高说明矩阵主要结构越集中，但下游任务仍要实测。

39. PCA 和 SVD 有什么关系？
   考察点：中心化数据、协方差矩阵、主成分、explained variance。
   回答框架：PCA 先中心化数据，寻找方差最大的方向；可以通过协方差矩阵特征分解或对中心化数据矩阵做 SVD 得到主方向。

40. LoRA 和低秩压缩有什么区别？
   考察点：原始权重、增量权重、质量回归。
   回答框架：低秩压缩用低秩因子近似原始 `W`，可能损伤原模型能力；LoRA 通常冻结 `W`，只把微调增量 `Delta W` 参数化为低秩。

41. LoRA rank 如何影响参数量和效果？
   考察点：`r(d_in+d_out)`、表达能力、过拟合和成本。
   回答框架：rank 越大可训练参数越多、表达能力更强，但显存和过拟合风险更高；rank 太小可能表达不足，需要用验证集、人评和能力回归做 sweep。

42. QLoRA 为什么能省显存，它的风险是什么？
   考察点：base model 量化、LoRA adapter、量化误差、计算 dtype。
   回答框架：QLoRA 量化存储 base model，只训练小的 LoRA adapter，因此显存显著下降；风险是量化误差、kernel / dtype 复杂度、activation 峰值和任务质量回归。

43. 经验风险和真实风险有什么区别？
   考察点：ERM、训练分布、目标分布、泛化 gap。
   回答框架：真实风险是目标分布上的期望损失，经验风险是训练集上的平均损失。训练时只能用样本近似真实分布，因此要用验证集、测试集和线上样本检查经验风险最小化是否真的泛化。

44. 如何用 train / validation / test 判断过拟合和欠拟合？
   考察点：训练分数、验证分数、gap、loss 曲线。
   回答框架：训练好验证差通常是过拟合；训练和验证都差通常是欠拟合或训练配置错误。对 LLM 还要看人评、能力回归、安全拒答、真实用户切片和数据污染。

45. Bias-variance 分解在大模型面试中怎么讲？
   考察点：bias、variance、噪声、现代深度学习边界。
   回答框架：平方损失下误差可拆成 bias^2、variance 和不可约噪声。Bias 高说明模型或训练不足，variance 高说明对训练集过敏。LLM 中不能机械套传统小模型直觉，但它仍能帮助定位欠拟合、过拟合、seed 波动和数据不足。

46. 正则化为什么能缓解过拟合？LLM 中有哪些正则化？
   考察点：L2、weight decay、dropout、early stopping、数据正则、KL。
   回答框架：正则化通过限制参数、训练行为或数据暴露，降低模型记住噪声和偏离原能力的风险。LLM 中包括 AdamW weight decay、dropout、early stopping、数据去重、通用数据混合、LoRA rank 约束和 RLHF/DPO 中的 KL 约束。

47. 为什么不能用测试集选 checkpoint？
   考察点：validation vs test、评估污染、early stopping。
   回答框架：验证集用于调参与 early stopping，测试集用于最终一次性评估。如果反复用测试集选模型、调 prompt 或调超参，测试集就变成开发集，分数会偏乐观，无法代表真实泛化。

48. 验证集高分但线上效果差，你会怎么排查？
   考察点：distribution shift、metric mismatch、污染、系统链路、切片。
   回答框架：先确认数据切分和污染，再看验证集是否代表线上用户、领域、时间和输入格式；继续检查指标是否匹配用户价值、线上 RAG / tool / safety 链路是否一致，最后做切片分析、bad case 和线上抽样。

49. Benchmark contamination 和普通过拟合有什么关系？
   考察点：训练-评估重叠、公开题库、prompt 迭代、泛化虚高。
   回答框架：contamination 是评估信息泄漏导致的特殊过拟合。模型或团队可能见过题目、答案、相似模板或通过反复调参用穿测试集，因此 benchmark 分数不能直接等同真实泛化。

50. 如何设计一个最小泛化审计 demo？
   考察点：memorizer、rule model、underfit、分布偏移、overlap。
   回答框架：构造 train / validation / test 三份 toy 数据，比较 memorizer、可迁移规则模型和 underfit baseline 的分数；再加入线上新切片、checkpoint val loss 和 exact overlap 检查，输出 gap、worst slice、污染率和 early stopping 结论。

51. 贝叶斯公式在 LLM 系统里怎么理解？
   考察点：prior、likelihood、posterior、RAG 证据、工具验证。
   回答框架：先验是模型或历史评估对答案可靠性的初始判断，RAG 文档、工具结果和引用检查是证据，后验是结合证据后的可靠性判断。可靠系统不应只看模型语言先验，而要用证据更新是否回答。

52. MLE 和 MAP 有什么区别？为什么 MAP 能解释正则化？
   考察点：likelihood、prior、posterior、L2。
   回答框架：MLE 只最大化数据似然；MAP 最大化后验，相当于同时考虑似然和先验。取负对数后，先验项会变成正则项；零均值高斯先验对应类似 L2 正则。

53. Aleatoric uncertainty 和 epistemic uncertainty 有什么区别？
   考察点：数据噪声、知识不足、处理策略。
   回答框架：Aleatoric 来自输入或任务本身的噪声和歧义，常用澄清或条件化回答处理；epistemic 来自模型知识不足或分布外，常用检索、工具、人审、补数据或拒答处理。

54. 为什么 token probability 高不代表答案事实正确？
   考察点：语言似然、事实验证、分布先验、RAG。
   回答框架：token probability 衡量文本在模型分布下是否顺，但模型分布可能包含错误、过时或缺证据的模式。事实正确性要看外部证据、工具验证、引用支持和任务评估。

55. Calibration、ECE 和 Brier score 分别看什么？
   考察点：置信度、真实正确率、分桶误差、概率均方误差。
   回答框架：Calibration 看置信度是否匹配真实正确率；ECE 按置信度分桶计算平均置信度和准确率的差距；Brier score 计算概率预测与 0/1 标签的均方误差。

56. 如何让模型“知道自己不知道”？
   考察点：abstention、资料不足训练、RAG、verifier、高风险人审。
   回答框架：不能只靠一句 prompt。训练上加入资料不足、澄清和拒答案例；系统上接 RAG、工具和 verifier；评估上看 unsupported claim、abstention accuracy、over-refusal 和高风险人审。

57. 高风险场景为什么不能只按置信度自动回答？
   考察点：risk tier、verification、human review、audit log。
   回答框架：高置信也可能来自错误先验或污染证据。医疗、法律、金融、安全和权限操作要结合证据等级、工具验证、人工复核和审计日志；低置信应澄清或拒答，高风险即使高置信也可能需要 verify。

58. 如何设计一个最小贝叶斯与校准审计 demo？
   考察点：Bayes update、MAP、entropy、self-consistency、ECE、abstention。
   回答框架：用 toy prior 和 evidence likelihood 计算强 / 弱证据后验；用 Beta-Bernoulli 估计可靠率；比较 MLE 和 MAP；计算 token entropy、self-consistency、ECE、Brier，并按风险输出 answer / verify / abstain。

59. 如何把 LLM 生成过程映射成 MDP？
   考察点：state、action、transition、reward、policy。
   回答框架：状态是 prompt 加已生成 token，动作是下一个 token，转移是追加 token，policy 是语言模型条件分布，reward 通常是完整回答后 reward model 或人工偏好给出的分数。

60. Return、value、Q function 和 advantage 分别是什么？
   考察点：discounted return、baseline、action value、advantage。
   回答框架：Return 是累计折扣奖励；value 是状态下期望 return；Q 是状态动作对的期望 return；advantage 是 `Q - V` 或估计 return 减 value baseline，表示比预期好多少。

61. Policy gradient 为什么会提高高 reward 行为的概率？
   考察点：log probability trick、advantage、方差降低。
   回答框架：policy gradient 目标中有 `grad log pi(a|s) * advantage`。正 advantage 会提高对应动作 log probability，负 advantage 会降低；value baseline 不改变期望方向，但能降低方差。

62. PPO clipped objective 在 RLHF 里解决什么问题？
   考察点：新旧 policy ratio、clip、训练稳定性。
   回答框架：PPO 用 `pi_new / pi_old` 衡量更新幅度，并把 ratio clip 到 `1 +/- epsilon` 附近，避免一次更新让语言模型行为大幅漂移。

63. RLHF 为什么需要 KL penalty？
   考察点：reference model、reward hacking、语言能力保持、beta。
   回答框架：reward model 是 proxy，policy 无约束优化可能钻漏洞或语言退化。KL penalty 限制 policy 接近 reference；`beta` 太小容易跑偏，太大又学不动。

64. Reward model pairwise loss 怎么写？
   考察点：chosen、rejected、sigmoid ranking loss。
   回答框架：常见形式是 `-log sigmoid(r_chosen - r_rejected)`，鼓励 chosen 得分高于 rejected。它学的是偏好代理，不是真实目标本身。

65. DPO 和 PPO 式 RLHF 的数学联系是什么？
   考察点：preference pair、reference、隐式 reward、KL 约束。
   回答框架：DPO 可看作把带 KL 约束的偏好优化改写成直接用 chosen / rejected log probability 的监督式目标。它省掉 RM 和 PPO loop，但仍通过 reference 和 beta 控制偏离。

66. 如何设计一个最小 RLHF 数学审计 demo？
   考察点：return、advantage、PPO ratio、KL、RM loss、DPO loss。
   回答框架：构造 toy rewards 和 values 计算 discounted returns / advantages；给 old/new policy prob 计算 PPO ratio 和 clipped surrogate；给 policy/reference 分布计算 KL penalty；再计算 reward model pairwise loss 和 DPO loss。

67. p-value 是什么，为什么不能解释成“模型无效的概率”？
   考察点：零假设、极端结果、条件概率、常见误解。
   回答框架：p-value 是在零假设成立时观察到当前或更极端结果的概率；它不直接给出新模型无效或有效的概率。解释时要结合 effect size、置信区间、样本量和实验设计。

68. 为什么评估结果要报告置信区间？
   考察点：点估计、不确定性、标准误、上线决策。
   回答框架：置信区间表达效果大小的不确定范围。单点 lift 可能为正，但区间很宽或跨过 0 时结论不稳定；上线还要看成本、延迟、安全和分层结果。

69. paired evaluation 为什么比非配对比较更适合模型对比？
   考察点：同题比较、样本难度、方差降低、paired diff。
   回答框架：两个模型在同一批样本上比较，可以控制样本难度差异，直接统计每条样本上的差异 `d_i`。Pairwise human eval 和同题 accuracy 对比都适合 paired 设计。

70. McNemar test 在模型评估中解决什么问题？
   考察点：paired binary result、discordant pair、b/c counts。
   回答框架：McNemar 用于同一批二分类样本上比较两个模型，只看旧错新对的 `b` 和旧对新错的 `c`。两个模型都对或都错不能说明差异方向，样本小时可用 exact binomial 口径。

71. Bootstrap 为什么适合 LLM 复杂指标？
   考察点：重采样、指标分布、复杂指标、样本偏差。
   回答框架：Bootstrap 对评估样本有放回重采样，重复计算指标，得到经验分布和置信区间。它适合 win rate、judge score、RAG faithfulness 和人工 rubric，但不能修复样本不代表真实分布的问题。

72. 样本量、MDE 和 statistical power 的关系是什么？
   考察点：minimum detectable effect、显著性水平、功效、方差。
   回答框架：MDE 越小、指标方差越大、想要的功效越高，需要的样本量越大。样本量太小导致“没显著”不等于“没效果”，而可能只是实验没有足够 power。

73. 多重比较为什么会制造偶然显著？
   考察点：多指标、多 prompt、多分层、中途查看、校正。
   回答框架：同时尝试很多模型、prompt、指标和分层，总会有一部分偶然显著。要预注册主指标，区分探索性和确认性分析，用独立 holdout 复验，并考虑 Bonferroni 或 Benjamini-Hochberg 校正。

74. 如何设计一个最小评估统计审计 demo？
   考察点：mean、variance、SE、CI、bootstrap、McNemar、sample size、multiple comparison、guardrail。
   回答框架：构造同一批样本的 old/new 0/1 结果，计算 accuracy、paired diff、标准误和正态 CI；用 paired bootstrap 得到差异 CI；统计 McNemar 的 `b/c` 和 exact p-value；估算样本量；比较 Bonferroni/BH；最后用主指标和 latency/cost/safety guardrail 决定是否上线。

75. 数学面试题的标准回答结构是什么？
   考察点：直觉、公式、变量、LLM 场景、边界。
   回答框架：先讲直觉，再写核心公式并解释变量，接着连接到 pretraining / SFT / RLHF / RAG / evaluation，最后补一个常见误区或工程边界。

76. 如何快速复盘自己是否真的掌握第十三册数学基础？
   考察点：topic coverage、formula accuracy、demo coverage、weak questions。
   回答框架：把线性代数、概率、信息论、优化、低秩、泛化、贝叶斯、RLHF 数学和评估统计列成 required topics；每道 mock answer 按公式、直觉、LLM 场景、边界、demo 打分，输出缺失主题和修订计划。

77. 为什么数学面试不能只背公式？
   考察点：适用条件、工程现象、追问、误区。
   回答框架：公式正确只是底线。面试官更关注是否知道公式何时成立、变量代表什么、在 LLM 中解释什么现象、有哪些失败场景，以及能不能写最小代码验证。

78. 如何设计一个最小数学面试复盘 demo？
   考察点：CE/KL、PPL、softmax gradient、LoRA、DPO、paired eval、readiness rubric。
   回答框架：先用 toy 分布计算 entropy、cross entropy、KL、PPL 和 `p-y` 梯度；再算 gradient clipping、LoRA 参数比例、DPO margin/loss、paired eval lift；最后对 mock answers 统计 topic coverage、formula accuracy、demo coverage、weak questions 和 readiness gate。

79. 面试中如何解释“公式、代码、实验”三者的关系？
   考察点：公式抽象、代码验证、实验决策。
   回答框架：公式给出机制和变量关系，代码把机制变成可执行计算，实验判断这个机制在真实任务上是否有用。比如 CE/KL 可手算，loss demo 可运行，但模型是否更好仍要看验证集、置信区间和错误分析。

80. DPO beta、PPO clipping、KL penalty 这类超参为什么不能只背一句话？
   考察点：推导约定、实现差异、reference、稳定性、验证。
   回答框架：这些超参都在控制优化强度和偏离程度，但具体解释依赖目标函数和实现约定。成熟回答要说明它们如何影响 policy/reference 距离、训练稳定性、reward hacking 风险，并强调要用验证集、KL、质量和安全指标调参。

## PyTorch 与工程基础

1. PyTorch tensor 除了数据本身还包含哪些关键信息？
   考察点：shape、dtype、device、stride、requires_grad、contiguous。
   回答框架：tensor 至少要看数据、shape、dtype、device、stride、是否 contiguous 以及是否参与梯度。大模型调试通常先查这些元信息，再看具体数值。

2. 语言模型训练中常见 tensor shape 如何推导？
   考察点：`input_ids [B,T]`、hidden states、logits、labels、loss flatten。
   回答框架：token id 是 `[B,T]`，embedding 后是 `[B,T,d]`，LM head 后是 `[B,T,V]`。做 next-token loss 时 `logits[:,:-1,:]` 展平成 `[B(T-1),V]`，labels `[:,1:]` 展平成 `[B(T-1)]`。

3. dtype 和 device 常见错误分别是什么？
   考察点：Embedding long input、float/bfloat16、CPU/GPU mismatch。
   回答框架：Embedding 输入通常必须是 `torch.long` token id；mask 和激活 dtype 要符合算子要求；参与同一次计算的 tensor 必须在同一 device 上，否则会报 device mismatch。

4. PyTorch broadcasting 规则是什么？为什么 mask 容易错？
   考察点：右对齐、维度相等或为 1、语义方向。
   回答框架：broadcast 从右往左对齐，每维相等或其中一个为 1 才能扩展。Attention mask 常能广播但语义方向错，所以要确认 batch、head、query、key 维分别对应什么。

5. `view`、`reshape` 和 `contiguous` 的关系是什么？
   考察点：stride、layout、copy、transpose 后 view 报错。
   回答框架：`view` 要求 stride 与目标 shape 兼容；`reshape` 如果不能返回 view 可能创建拷贝；`transpose/permute` 后常非 contiguous，所以常见写法是 `transpose(...).contiguous().view(...)`。

6. `mm`、`bmm`、`matmul` 和 `einsum` 有什么区别？
   考察点：二维、三维 batch、batch broadcasting、显式维度标记。
   回答框架：`mm` 只做二维矩阵乘法；`bmm` 做三维 batch 矩阵乘法且不广播 batch；`matmul` 更通用，最后两维做矩阵乘法并广播前置维；`einsum` 用字符显式说明维度关系。

7. Attention score 的 PyTorch shape 流程怎么写？
   考察点：Q/K/V projection、reshape、transpose、matmul、mask。
   回答框架：从 `[B,T,d]` 投影到 Q/K/V，再 reshape 为 `[B,T,H,d_h]` 并 transpose 成 `[B,H,T,d_h]`；`q @ k.transpose(-2,-1)` 得到 `[B,H,T,T]`；padding / causal mask broadcast 后填充无效位置。

8. 如何设计一个最小 PyTorch tensor audit demo？
   考察点：metadata、broadcast、matmul/einsum、mask、contiguous、LM loss。
   回答框架：构造 `input_ids [B,T]`，查 dtype/device；embedding 得到 `[B,T,d]`；测试 `[d]` bias broadcasting；计算 attention scores；比较 matmul/einsum；构造 padding+causal mask；验证 transpose 后 view 报错并用 contiguous 修复；最后把 logits/labels 展平算 cross entropy。

9. PyTorch autograd 是怎么工作的？
   考察点：动态计算图、grad_fn、链式法则、leaf tensor、backward。
   回答框架：PyTorch 在 forward 时动态构建计算图，可求导操作会记录 backward 函数。`loss.backward()` 从 loss 出发按链式法则反向遍历，把梯度累积到 leaf 参数的 `.grad`。

10. 为什么每次训练 step 前要 `zero_grad()`？
    考察点：梯度累积、optimizer step、set_to_none。
    回答框架：PyTorch 的 `.grad` 默认累积，不会自动清零。普通训练如果不清梯度，会把多个 batch 的梯度叠加；主动做梯度累积时也要按 micro-step 控制 loss 缩放和 step 频率。

11. leaf tensor 和 non-leaf tensor 的 `.grad` 有什么区别？
    考察点：模型参数、retain_grad、显存。
    回答框架：模型参数通常是 leaf tensor，backward 后 `.grad` 会保留；中间 tensor 虽然有梯度流过，但默认不保留 `.grad`，需要调试时调用 `retain_grad()` 或注册 hook。

12. `detach()`、`no_grad()` 和 `inference_mode()` 有什么区别？
    考察点：切断图、上下文关闭建图、推理模式。
    回答框架：`detach()` 是对某个 tensor 切断当前图连接；`no_grad()` 是让一段代码不记录新图；`inference_mode()` 更激进，适合纯推理但不适合后续还要参与 autograd 的张量。

13. 为什么验证时要同时用 `model.eval()` 和 `torch.no_grad()`？
    考察点：模块行为、梯度记录、显存。
    回答框架：`eval()` 改变 dropout / batch norm 等模块行为，`no_grad()` 关闭 autograd 图记录。两者解决的问题不同，验证和推理通常都需要。

14. `retain_graph=True` 应该什么时候用，为什么不能滥用？
    考察点：图释放、多次 backward、显存增长。
    回答框架：默认 backward 后计算图释放。确实需要从同一张图多次反传时才考虑 `retain_graph=True`，普通训练更应该重新 forward 或把多个 loss 合并一次 backward。

15. in-place 操作为什么可能破坏反向传播？
    考察点：版本计数、保存中间值、masked_fill_。
    回答框架：反向传播可能依赖 forward 保存的中间值；in-place 改写这些值会让梯度不可靠，PyTorch 会用版本计数检测并报错。模型 forward 中优先使用非 in-place 写法。

16. 如何设计一个最小 autograd audit demo？
    考察点：scalar grad、VJP、retain_grad、grad accumulation、detach、no_grad、missing grad。
    回答框架：构造标量函数验证梯度；对向量输出传外部梯度；对中间 tensor 调 `retain_grad()`；连续 backward 展示梯度累积；用 detach 验证分支断图；用 `no_grad()` 做验证；最后遍历 `named_parameters()` 输出 missing grad 和 grad norm。

17. `nn.Module` 在 PyTorch 工程中到底负责什么？
    考察点：注册机制、递归管理、optimizer、checkpoint、DDP。
    回答框架：`nn.Module` 不只是封装 `forward`，还负责注册 `Parameter`、子模块和 buffer，并提供 `parameters()`、`state_dict()`、`to(device)`、`train/eval`、hook 和保存加载等统一接口。

18. `nn.Parameter` 和普通 `requires_grad=True` tensor 有什么区别？
    考察点：注册参数、optimizer、state_dict。
    回答框架：`Parameter` 被赋值为 Module 属性时会注册到参数树，出现在 `model.parameters()` 和 `state_dict()` 中；普通 tensor 即使需要梯度，也通常不会被 optimizer 自动更新。

19. 为什么子模块不能放普通 list / dict？
    考察点：ModuleList、ModuleDict、递归注册。
    回答框架：普通容器里的 Module 不会被注册，导致参数不出现在 `parameters()`、不随 `to(device)` 移动、不进 `state_dict()`、DDP 也不会同步。多层堆叠用 `ModuleList`，按名称选择用 `ModuleDict`。

20. Parameter 和 buffer 的区别是什么？
    考察点：是否训练、是否保存、是否迁移 device。
    回答框架：Parameter 是可训练权重，通常会被 optimizer 更新；buffer 是不训练但属于模型状态的 tensor，会随 `model.to(device)` 移动，默认进入 `state_dict()`，但不出现在 `parameters()` 中。

21. `state_dict` 保存什么，不保存什么？
    考察点：参数、持久化 buffer、Python 代码、optimizer。
    回答框架：模型 `state_dict` 保存注册参数和持久化 buffer，不保存 Python 类定义、`forward` 代码、普通属性、未注册 tensor、optimizer 状态和 `persistent=False` buffer。

22. `load_state_dict(strict=False)` 应该怎么安全使用？
    考察点：missing key、unexpected key、部分加载。
    回答框架：`strict=False` 可以用于加载 backbone、LoRA 或结构变化后的部分权重，但必须打印并解释 missing / unexpected keys。分类头 missing 合理，大量 block missing 通常说明 key 对不上。

23. 冻结参数后为什么常常要重新创建 optimizer？
    考察点：requires_grad、param_groups、可训练参数量。
    回答框架：optimizer 创建时拿到的是当时的参数列表。冻结或新增参数后，旧 optimizer 的 param_groups 不会自动重新筛选；应重新创建 optimizer 或显式调整参数组，并核对 optimizer 参数量。

24. 如何设计一个最小 Module audit demo？
    考察点：注册树、buffer、state_dict、train/eval、save/load、optimizer。
    回答框架：构造普通 list 和 `ModuleList` 对比参数注册；定义 `Parameter` 和 buffer；检查 `named_parameters()`、`named_buffers()`、`state_dict().keys()`；跑一次训练；冻结部分参数并重建 optimizer；保存加载后在 eval 模式比较 logits；用 `strict=False` 验证 missing keys 是否符合预期。

25. Dataset 和 DataLoader 的职责边界是什么？
    考察点：单样本读取、batch 组装、shuffle、sampler、worker。
    回答框架：Dataset 负责定义单个样本如何读取和轻量预处理；DataLoader 负责按 batch 取样、可选 shuffle、多进程加载、调用 `collate_fn` 和配合 sampler 控制顺序。

26. 为什么变长文本样本通常在 `collate_fn` 里 padding？
    考察点：batch max length、padding waste、attention_mask、labels ignore。
    回答框架：pad 到多长取决于当前 batch 的最长样本，所以 padding 是 batch 级逻辑。放在 `collate_fn` 里可以按 batch 动态 padding，同时构造 `attention_mask` 并把 padding label 设为 `-100`。

27. causal LM 数据 batch 的核心 shape 怎么检查？
    考察点：`input_ids [B,T]`、`labels [B,T]`、`attention_mask [B,T]`、shift。
    回答框架：先看 `input_ids`、`labels` 和 `attention_mask` 是否都是 `[B,T]`；再看 padding 位置 label 是否为 ignore index；最后确认 next-token loss 使用 `logits[:, :-1, :]` 和 `labels[:, 1:]`。

28. `shuffle`、`sampler` 和 `batch_sampler` 有什么区别？
    考察点：样本顺序、索引控制、batch 组成控制。
    回答框架：`shuffle=True` 是简单随机打乱；sampler 控制单个样本索引顺序；batch sampler 直接产出一批索引，适合 length bucket 或复杂 batch 组成策略。

29. 为什么分布式训练要用 `DistributedSampler`，还要调用 `set_epoch`？
    考察点：rank 切分、重复样本、epoch shuffle。
    回答框架：`DistributedSampler` 按 rank 切分数据，避免每张卡都训练完整数据集。每个 epoch 调用 `set_epoch(epoch)` 可以刷新 shuffle 顺序，否则不同 epoch 的随机顺序可能不变。

30. `num_workers`、`pin_memory`、`persistent_workers` 和 `prefetch_factor` 分别解决什么问题？
    考察点：CPU 预处理、CPU-GPU 拷贝、worker 启动成本、预取。
    回答框架：`num_workers` 控制并行加载进程数；`pin_memory` 有助于 CPU 到 GPU 异步拷贝；`persistent_workers` 减少多 epoch 反复启动 worker 的开销；`prefetch_factor` 控制预取 batch 数。它们影响吞吐和内存，不改变训练目标。

31. DataLoader 相关 bug 通常怎么排查？
    考察点：样本结构、batch shape、mask、worker、sampler。
    回答框架：先打印单样本，再打印一个 batch；检查字段、shape、padding、ignore index 和 shift；再看 shuffle / sampler / rank 切分；最后排查 `num_workers`、IO、解码和 `collate_fn` 是否成为瓶颈。

32. 如何设计一个最小 DataLoader audit demo？
    考察点：Dataset、collate、padding waste、DistributedSampler、reproducibility。
    回答框架：构造变长 toy token 序列；Dataset 返回单样本；`collate_fn` 动态 padding 并设置 `labels=-100`；DataLoader 固定 generator seed；输出 batch shape、attention mask、valid token ratio、shift shape、length bucket 前后 padding waste，并用 `DistributedSampler` 验证 rank 间索引是否重叠。

33. 一个标准 PyTorch training step 的顺序是什么？
    考察点：zero_grad、forward、loss、backward、clip、optimizer、scheduler。
    回答框架：常见顺序是 `zero_grad`、forward、loss、`backward()`、可选 gradient clipping、`optimizer.step()`、`scheduler.step()` 和日志记录。梯度裁剪必须在 backward 后、optimizer step 前。

34. 梯度累积为什么要缩放 loss？
    考察点：micro-batch、平均梯度、隐式学习率。
    回答框架：PyTorch 梯度默认累积。累积 `K` 个 micro-batch 时，如果每个 loss 不除以 `K`，最终梯度尺度会约等于平均梯度的 `K` 倍，相当于隐式放大学习率。

35. scheduler 应该按 micro-step 还是 optimizer step 更新？
    考察点：真实参数更新、梯度累积、warmup steps。
    回答框架：多数按 step 更新的 scheduler 应跟随真实 `optimizer.step()`，不是每个 micro-step。否则使用梯度累积时学习率会过快 warmup 或 decay。

36. 语言模型训练循环里 shift loss 怎么检查？
    考察点：`logits[:, :-1, :]`、`labels[:, 1:]`、`ignore_index`。
    回答框架：因果 LM 用当前位置预测下一个 token，所以 logits 去掉最后一位，labels 去掉第一位；展平后用 cross entropy，并用 `ignore_index=-100` 跳过 padding 或非监督位置。

37. validation loop 为什么要同时用 `eval()` 和 `inference_mode()`？
    考察点：模块行为、关闭梯度、显存、恢复训练模式。
    回答框架：`eval()` 改变 dropout / batch norm 等模块行为，`inference_mode()` 关闭梯度记录并降低推理开销。它们解决的问题不同；验证结束后要恢复 `model.train()`。

38. 完整训练 checkpoint 应该保存什么？
    考察点：model、optimizer、scheduler、step、config、rng。
    回答框架：恢复训练需要保存模型权重、optimizer state、scheduler state、epoch / global step、config 和随机状态。只保存模型会丢失动量、二阶矩和学习率位置，恢复后训练轨迹会变化。

39. 训练中出现 non-finite loss 怎么排查？
    考察点：NaN / Inf、batch、lr、grad norm、mask、AMP。
    回答框架：先确认 loss 还是梯度非有限，再看当前 batch、学习率、grad norm、混合精度溢出、mask 是否全忽略或全 `-inf`、输入是否有异常值，并保存问题 batch 做复现。

40. 如何设计一个最小 Training Loop audit demo？
    考察点：shift loss、accumulation、clip、scheduler、eval、checkpoint。
    回答框架：构造 tiny causal LM 和 toy batches；训练时输出 raw loss、grad norm、optimizer step 数、scheduler last epoch 和 final lr；验证使用 token-weighted loss；保存并恢复完整 checkpoint，确认恢复后验证 loss 一致。

41. FP16 和 BF16 的核心区别是什么？
    考察点：指数范围、尾数精度、动态范围、训练稳定性。
    回答框架：两者都是 16 位浮点，但 BF16 指数位更多，动态范围更接近 FP32，通常更不容易溢出；FP16 精度和硬件效率也常见，但更依赖 loss scaling 和数值检查。实际选择要看硬件、算子支持和训练稳定性。

42. `autocast` 和 `GradScaler` 分别解决什么问题？
    考察点：算子 dtype 选择、FP16 下溢、loss scaling。
    回答框架：`autocast` 按算子策略自动选择低精度或高精度执行，减少手动 dtype 管理；`GradScaler` 主要用于 FP16，把 loss 放大后反传，再在 optimizer step 前把梯度缩回真实尺度，以降低小梯度下溢风险。

43. 为什么梯度裁剪前要先 `unscale_`？
    考察点：scaled gradient、真实梯度范数、clip 顺序。
    回答框架：loss scaling 后 `.grad` 里保存的是放大后的梯度。如果直接裁剪，裁剪的是 scale 后的范数，会改变真实优化方向和裁剪阈值含义。正确顺序是 backward、`scaler.unscale_(optimizer)`、clip、`scaler.step()`、`scaler.update()`。

44. Activation checkpointing 为什么能省显存，代价是什么？
    考察点：保存激活、重算 forward、计算换显存。
    回答框架：checkpointing 不保存某些中间激活，而是在 backward 时重新执行对应 forward 来恢复所需中间值，因此降低 activation 显存。代价是 backward 计算更慢，并且要注意随机层、非纯 forward 副作用和 checkpoint API 参数。

45. 训练显存通常由哪些部分组成？
    考察点：参数、梯度、optimizer state、activation、temporary buffer。
    回答框架：训练显存粗略由参数、梯度、优化器状态、前向激活和临时 workspace / kernel buffer 组成。大模型里 AdamW 的一阶矩和二阶矩、长序列激活经常比参数本身更容易造成 OOM。

46. 为什么 AdamW optimizer state 会成为显存大头？
    考察点：一阶矩、二阶矩、FP32 state、参数规模。
    回答框架：AdamW 通常为每个参数保存一阶矩和二阶矩，很多实现还用 FP32 存 optimizer state。即使参数和梯度用 BF16，每个参数也可能额外占约 8 bytes 的 state，模型越大越明显。

47. 训练 OOM 时你会按什么顺序排查？
    考察点：batch、sequence length、precision、activation、optimizer state、泄漏。
    回答框架：先区分 OOM 出现在训练、验证还是保存 checkpoint；再缩小 batch size 和 sequence length 判断是否由 activation 引起；随后检查 AMP / BF16、gradient accumulation、checkpointing、optimizer state、是否缓存带图 tensor、验证是否关闭梯度，以及 `memory_allocated` / `memory_reserved` / peak memory 变化。

48. 如何设计一个最小 AMP memory audit demo？
    考察点：显存估算、autocast dtype、GradScaler、checkpoint 重算、CUDA 状态。
    回答框架：构造 tiny MLP，先用参数量和 activation 元素数估算参数、梯度、AdamW state 和 activation 显存；用 `torch.amp.autocast` 检查输出 dtype；用 `GradScaler` 跑 scale、unscale、step、update；再用一个计数 block 对比普通 forward 和 checkpoint backward 的 forward 调用次数，最后打印 CUDA 是否可用和显存统计字段。

49. DDP 的工作原理是什么？
    考察点：数据并行、模型副本、backward、all-reduce、参数一致性。
    回答框架：每个 rank 持有一份模型副本并处理不同 mini-batch；backward 时 DDP 通过 autograd hook 同步梯度 bucket，all-reduce 后每个 rank 拿到一致的平均梯度，再各自执行 optimizer step，因此参数保持一致。

50. rank、local rank 和 world size 分别是什么？
    考察点：全局进程编号、本机 GPU 编号、总进程数。
    回答框架：rank 是全局进程编号，local rank 是当前机器内的进程或 GPU 编号，world size 是总训练进程数。通常用 local rank 绑定 GPU，用 rank 0 控制日志和 checkpoint，用 world size 计算 global batch 和通信规模。

51. 为什么 DDP 要配合 DistributedSampler？
    考察点：数据切分、重复样本、set_epoch、shuffle。
    回答框架：DDP 不会自动切分 Dataset。每个 rank 应该处理不同样本子集，否则多卡重复训练同一批数据；DistributedSampler 按 rank / world size 切分索引，并通过 `set_epoch(epoch)` 改变每个 epoch 的 shuffle 顺序。

52. global batch size 怎么计算，为什么它重要？
    考察点：per-device batch、world size、gradient accumulation、学习率。
    回答框架：`B_global = B_device * world_size * accum_steps`。从单卡切到多卡时，如果每卡 batch 不变，global batch 会变大，优化动态、学习率、warmup、训练步数和日志指标都可能需要重新校准。

53. all-reduce 在 DDP 中起什么作用？
    考察点：collective、梯度求和 / 平均、同步代价。
    回答框架：all-reduce 会把所有 rank 上的梯度归约并把结果返回给所有 rank。DDP 用它让每张卡拿到相同平均梯度，代价是引入通信开销，受参数量、梯度 dtype、网络带宽和同步频率影响。

54. 梯度累积时为什么要用 DDP 的 `no_sync()`？
    考察点：micro-step、同步频率、通信浪费、最后一步同步。
    回答框架：如果每个 micro-step 都同步梯度，通信次数会放大。`no_sync()` 可以让前几个 accumulation step 只本地累积梯度，最后一个 micro-step 再同步并 optimizer step，从而减少通信。

55. DDP 和 FSDP 的区别是什么？
    考察点：完整模型副本、参数 / 梯度 / optimizer state 分片、显存。
    回答框架：DDP 每个 rank 通常持有完整参数、梯度和 optimizer state，主要解决吞吐扩展；FSDP 会分片参数、梯度和优化器状态，降低单卡显存，但通信、wrap policy、checkpoint 和 debug 更复杂。

56. 如何设计一个最小 Distributed Training audit demo？
    考察点：rank 切分、global batch、all-reduce、no_sync、collective、rank0 checkpoint。
    回答框架：用 toy world size 构造每个 rank 的数据索引，检查 overlap 和覆盖；计算 global batch；用几个本地梯度模拟 all-reduce 平均和参数一致性；比较 `no_sync` 前后通信次数；构造某个 rank 缺失 all-reduce 的 trace 来验证能发现 collective 不一致；最后检查只有 rank 0 写 checkpoint。

56A. 分布式训练卡死、吞吐低或 checkpoint 恢复异常时，如何设计 distributed incident gate？
    考察点：rank step alignment、collective mismatch、token shard imbalance、straggler ratio、communication ratio、checkpoint shard coverage、resume continuity、pipeline bubble。
    回答框架：先要求每个 rank 输出 step、phase、batch id、token 数、step/data/comm time、collective 序列、loss 和 checkpoint 状态；再检查所有 rank 是否在同一 step、collective 是否一致、token shard 是否均衡、最慢 rank 是否超过典型 rank、通信占比是否过高、checkpoint shard 是否完整、resume 后 global step 和 lr 是否连续，以及 pipeline micro-batch 是否足以压低 bubble。门禁输出 failed gates 和对应 rank / shard / stage，避免只凭 rank 0 日志猜测。

57. loss 不下降时你怎么排查？
    考察点：数据、label、shape、loss、梯度、optimizer、学习率。
    回答框架：先打印原始样本和 batch，确认 input / label / mask 正确；再检查 logits 和 labels 的 shift / flatten 是否对齐，loss 是否有限且依赖参数；随后看梯度是否为 None、0、NaN 或异常大，参数是否进入 optimizer，最后再看学习率、初始化、模型容量和数据分布。

58. 训练出现 NaN / Inf 时怎么定位？
    考察点：输入、logits、loss、grad、参数、首次出现位置。
    回答框架：按输入、logits、loss、梯度、参数的顺序做 finite check，定位非有限值首次出现的位置。常见原因包括学习率过大、fp16 溢出、mask 全 `-inf`、loss 分母为 0、异常数据和梯度爆炸。

59. shape / dtype / device debug 最先看什么？
    考察点：tensor metadata、Embedding long、device mismatch、contiguous。
    回答框架：先打印每个关键 tensor 的 shape、dtype、device、requires_grad 和 contiguous 状态。Embedding 输入通常要 `long`，参与同一计算的 tensor 要在同一 device，transpose 后 `view` 前要确认 contiguous。

60. 梯度为 None、全 0 或异常大分别意味着什么？
    考察点：断图、无信号、梯度爆炸、optimizer 覆盖。
    回答框架：grad 为 None 常见于参数未参与 loss、`requires_grad=False`、中间 detach 或没进 forward；全 0 可能是 mask 全忽略或激活饱和；异常大可能是学习率、初始化、异常 batch 或 loss scale 问题，还要检查参数是否进入 optimizer。

61. OOM 怎么区分峰值过高和显存泄漏？
    考察点：memory allocated / reserved、step 后趋势、带图 tensor。
    回答框架：如果某一步峰值超限，优先看 batch、sequence length、activation、optimizer state、AMP 和 checkpointing；如果每步结束后显存持续增长，重点查是否把带图 tensor 存进 list、验证没关梯度、评估缓存 logits / hidden states 或 hook 没释放。

62. GPU 利用率低怎么排查？
    考察点：DataLoader、CPU-GPU copy、forward/backward、同步、通信。
    回答框架：先单独计时 DataLoader 迭代，再分段计时拷贝、forward、backward 和 optimizer step；CUDA 计时要 synchronize；最后用 profiler 看 CPU/CUDA 时间分布，区分数据瓶颈、小 batch、频繁同步和分布式通信瓶颈。

63. `torch.profiler` 能看什么，使用时有什么边界？
    考察点：CPU/CUDA 时间、shape、memory、调用次数、开销。
    回答框架：profiler 可以看到算子耗时、调用次数、shape 和内存分配，适合定位瓶颈算子、DataLoader / copy / forward / backward 占比。它本身有开销，通常只开少量 step，并结合 schedule 和 trace 使用。

64. 如何设计一个最小 Debug / Profiling audit demo？
    考察点：metadata、finite、shift loss、hook、grad norm、timing、profiler。
    回答框架：构造 tiny LM 和 toy batch，输出 input / logits / shifted tensors 的 metadata；检查 shift flatten 维度和有效 label 数；计算 loss 并做 finite check；注册 forward hook 记录激活 shape 后 remove；打印所有参数 grad norm 和 optimizer 覆盖；做基础 step timing，并把 profiler 做成可选小范围入口。

65. 手写 decoder-only Transformer 组件时，最小模块清单是什么？
    考察点：embedding、RMSNorm、causal attention、MLP、block、LM head、loss。
    回答框架：最小闭环包括 token embedding、RMSNorm 或 LayerNorm、causal scaled dot-product attention、multi-head self-attention、MLP / SwiGLU、Pre-Norm decoder block、final norm、LM head 和 causal LM shift loss；每个模块都要能说明输入输出 shape。

66. `input_ids [B,T]` 到 `logits [B,T,V]` 的 shape 主线怎么推导？
    考察点：embedding、block、final norm、LM head、vocab。
    回答框架：`input_ids [B,T]` 查 embedding table `[V,d]` 得到 `[B,T,d]`，经过若干 decoder block 后 shape 仍是 `[B,T,d]`，final norm 不改 shape，LM head 把最后一维从 `d` 投到 `V`，输出 `[B,T,V]`。

67. 手写 Multi-Head Self-Attention 最容易错在哪里？
    考察点：`hidden_size % num_heads`、reshape、transpose、mask、contiguous。
    回答框架：常见错误包括 `hidden_size` 不能整除 head 数、Q/K/V reshape 维度顺序错、mask 广播能跑但语义方向错、transpose 后直接 `view` 报错，以及 softmax 前没有正确屏蔽未来 token。

68. 为什么 KV Cache decode 的 causal mask 不能只按 `[T,T]` 写？
    考察点：prefill、decode、query_len、key_len、past length。
    回答框架：训练或 prefill 时 query 和 key 长度通常相同；decode 时 query 可能只有当前 token，但 key/value 包含完整历史 cache。mask 要考虑 `past_key_values_length`，否则当前 query 可能只能看见 key 的第一个位置，而不是全部历史。

69. LM head weight tying 如何检查？
    考察点：参数共享、embedding、lm_head、checkpoint。
    回答框架：检查 `lm_head.weight` 和 `embed_tokens.weight` 是否指向同一块参数存储，例如比较 `data_ptr()`。权重绑定能减少参数量并统一输入输出 token 空间，但扩词表和加载 checkpoint 时要保持共享关系。

70. RoPE 最小实现要验证什么？
    考察点：Q/K、二维旋转、范数保持、位置 cache。
    回答框架：RoPE 通常作用在 Q/K 上，把偶数维和奇数维组成二维子空间做位置相关旋转。最小 demo 应验证输入输出 shape 不变、每个二维子向量范数近似保持、cos/sin cache 与序列长度和 head_dim 对齐。

71. 如何设计一个 Transformer component audit demo？
    考察点：shape、mask、softmax、RoPE、loss、cache。
    回答框架：用 tiny config 串起 RMSNorm、causal mask、scaled attention、MHA、SwiGLU、decoder block、RoPE、MiniDecoderLM 和 CachedSelfAttention；输出 shape、attention 行和、future weight、RoPE norm、logits / loss、weight tying、cache length 和 gate checks。

72. 手写 Transformer 组件和调用 `nn.Transformer` 的面试价值有什么区别？
    考察点：机制理解、shape debug、工程接口。
    回答框架：调用高层 API 能快速搭系统，但会隐藏 Q/K/V reshape、mask 方向、loss shift、cache 拼接等面试高频机制。手写最小组件的价值是证明自己能定位底层 shape 和语义错误；真实工程再根据需要使用高层 API 或优化 kernel。

73. PyTorch 工程面试回答的标准结构是什么？
    考察点：机制、shape、公式、坑、排查路径。
    回答框架：先给一句定义，再讲核心机制；随后写出关键 shape 或公式，说明常见失败场景，最后给出 debug 顺序或最小 demo。只背 API 名称通常不够。

74. 如何用公式快速检查工程回答是否靠谱？
    考察点：shift rows、global batch、training memory、DDP average、KV Cache。
    回答框架：用 `B(T-1)` 检查 causal LM shift 展平行数，用 `B_device * world_size * accum_steps` 检查 global batch，用参数 / 梯度 / optimizer state / activation 拆显存，用 all-reduce 平均梯度解释 DDP，用 KV Cache 公式解释长上下文显存。

75. 如何设计一个 PyTorch 工程面试复盘 demo？
    考察点：topic coverage、formula coverage、debug coverage、red flags、revision plan。
    回答框架：建立 required topics，包括 tensor、autograd、module、dataloader、training loop、AMP、DDP、debug 和 Transformer；对 mock answers 做关键词和风险表述扫描；同时计算 global batch、shift rows、训练显存、DDP 平均梯度、attention shape 和 KV Cache；最后输出 gate 和修订计划。

76. 面试中怎么回答 loss 不下降、NaN、OOM、GPU 利用率低这类综合问题？
    考察点：排查顺序、证据链、复现、指标。
    回答框架：先把现象定位到数据、shape / mask、loss、梯度、optimizer、精度、显存、DataLoader 或分布式通信；再给最小复现和日志字段，比如 batch 样本、loss、lr、grad norm、finite check、memory peak、step timing 和 profiler。

77. 为什么 PyTorch 工程面试不能只说“我会调包”？
    考察点：封装边界、底层机制、debug 能力、生产风险。
    回答框架：高层 API 能提高效率，但面试追问常落到 tensor shape、autograd、mask、loss shift、optimizer state、DDP 同步和显存。能解释底层机制和写最小 demo，才说明遇到异常时能定位问题。

78. 如何设计训练稳定性门禁？
    考察点：first bad step、loss spike、NaN / Inf、grad norm、AMP overflow、有效 label token、LR continuity、rank loss skew、checkpoint resume。
    回答框架：每 step 记录 loss、真实 lr、grad norm、overflow、有效 label token、batch id、rank loss 和 resume 状态；门禁约束 non-finite 为 0、spike 和 grad norm 不超阈值、overflow 率可控、有效 label token 不异常、resume 后 lr 连续、rank loss skew 不过大。失败时输出 first bad step 和 failed gates，再保存 batch 做单卡 / fp32 / 小 batch 最小复现。

## Transformer 与 LLM

1. LLM 到底在学什么？
   考察点：token 序列、条件概率分布、next-token prediction、记忆与泛化。
   回答框架：LLM 学习 `P(x_t | x_{1:t-1})`，通过海量文本学习语言、知识、任务模式和推理线索；它既可能记忆，也可能泛化。
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
   回答框架：语言模型为 token 序列分配概率；自回归模型用链式法则分解为 `∏ P(x_t | x_{1:t-1})`。

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

31A. Transformer 为什么能成为大模型基础架构，而不是 RNN、CNN 或 SSM 直接胜出？
   考察点：route score、training parallelism、content routing、next-token prediction、scaling evidence、ecosystem readiness、deployment pressure。
   回答框架：先承认没有单一架构在所有指标上最优。RNN/LSTM 有状态和流式优势，但训练并行性弱、远距离路径长；CNN 并行和硬件友好，但内容相关路由不如 attention；SSM/Mamba-like 路线长上下文和状态效率强，但大规模通用 LLM、ICL、生态和迁移证据仍需积累。Transformer 胜在 self-attention 一跳内容路由、训练时可并行、decoder-only + next-token 目标统一、scaling law / GPT-3 / Chinchilla / LLaMA 等证据充分、训练推理生态成熟。代价是 `O(T^2)` attention、KV cache 和流式状态压力，所以后续才有 GQA/MLA、FlashAttention、PagedAttention、sliding window、SSM 和 hybrid 架构。

31B. 如何从信息路由角度审计 Self-Attention 是否可靠？
   考察点：Q/K/V、attention score、softmax、causal mask、route hit、future leak、lost-in-the-middle、interpretation boundary。
   回答框架：先把 Self-Attention 定义成内容相关的信息路由：Query 提出需求，Key 做匹配索引，Value 提供内容，softmax 权重决定局部信息混合。审计时看五类指标：route hit rate 是否命中期望证据，future leak rate 是否为 0，softmax 每行是否归一化，除以 `sqrt(d_k)` 后分布是否更稳定，以及 attention weight 是否被过度解释。要强调 mask 和 softmax 正确不代表任务一定正确，lost-in-the-middle、干扰 token、位置偏置和多层残差都会让可见信息没有被稳定利用。

31C. 如何解释 QK Circuit、OV Circuit 和 attention head 子空间？
   考察点：QK 读哪里、OV 写什么、head subspace、head redundancy、KV head ratio。
   回答框架：一个 attention head 不能只看 attention map。QK circuit 由 Query / Key 投影决定，负责从哪些 token 读取信息；OV circuit 由 Value 投影和输出投影决定，负责把读到的信息如何写回 residual stream。多头让模型在多个低维子空间中并行路由，但不同 head 可能功能重叠，训练后可以出现冗余。MHA / GQA / MQA 的系统差异主要体现在 Query head 和 K/V head 的数量关系上，减少 `H_kv` 可以降低 KV cache 和内存带宽压力。

31D. 如何系统比较 MHA、MQA、GQA 和 MLA？
   考察点：KV head 数、latent KV cache、质量门禁、cache saving、FlashAttention / PagedAttention 边界。
   回答框架：MHA 每个 query head 有独立 K/V，表达最完整但 KV cache 最大；MQA 保留多个 Q heads 但共享 1 组 K/V，cache 最省但可能损失质量；GQA 让一组 Q heads 共享 K/V，是质量和推理成本的折中；MLA 进一步把 K/V 信息压缩到 latent cache，减少的是缓存表示维度而不只是 KV head 数。还要区分层面：FlashAttention 是 kernel / IO 优化，PagedAttention 是 serving memory manager，MQA / GQA / MLA 是模型架构。

31E. 如何系统解释 FFN、SwiGLU 和 MoE FFN 的关系？
   考察点：position-wise FFN、SwiGLU 参数量、gated MLP、top-k routing、active parameters、expert capacity、load balancing。
   回答框架：FFN 是 Transformer block 中对每个 token 独立应用的通道变换模块，attention 负责 token 间路由，FFN 负责 token 内非线性加工和容量扩展。普通 4D FFN 参数约为 `2dm`，SwiGLU 有 gate/up/down 三组矩阵，参数约为 `3dm`，所以常把 intermediate size 调到接近 `8d/3` 做近似等参比较。MoE FFN 则把一个 dense FFN 换成多个 experts，由 router 做 top-k routing，每个 token 只激活少数 experts；解释 MoE 时必须区分 total parameters 和 active parameters，并说明 capacity overflow、load balance、all-to-all 通信和 serving 延迟是核心工程难点。

31F. 如何从 residual stream 和 norm placement 角度解释 Transformer 训练稳定性？
   考察点：residual stream、LayerNorm、RMSNorm、Pre-LN、Post-LN、warmup、residual scaling、activation norm、update ratio。
   回答框架：Transformer 的 hidden state 可以看成 residual stream，attention 和 FFN 都是在这条主路径上写入增量。LayerNorm / RMSNorm 控制每个 token hidden 维的尺度，norm placement 决定残差主路径是否被 norm 包住。Post-LN 是 `Norm(x + F(x))`，每层输出尺度统一，但深层训练更依赖 warmup；Pre-LN 是 `x + F(Norm(x))`，identity path 更干净，梯度更容易直通，现代 decoder-only LLM 常见。RMSNorm 去掉 re-centering、保留 re-scaling，常和 Pre-Norm、final norm、residual scaling、warmup、AdamW、gradient clipping 和 mixed precision 监控一起使用。

31G. 如何系统比较 Sinusoidal、Learned Absolute、Relative Bias、RoPE 和 ALiBi？
   考察点：注入位置、绝对/相对、长度外推、KV cache、工程复杂度。
   回答框架：Sinusoidal 和 learned absolute 通常加到输入 embedding，前者固定多频率、后者学习绝对位置；relative representation / bias 更贴合 attention pairwise 距离，其中 bias 更轻量；RoPE 旋转 Q/K，用绝对 position id 实现但点积体现相对位置；ALiBi 直接给 attention score 加线性距离惩罚。比较时不能只说“谁更先进”，要结合训练长度、目标上下文、position id、KV cache、短上下文退化和长文任务评估。

31H. 线上排查 RoPE / position id 异常时先看什么？
   考察点：KV cache、position id 连续性、packed sequence、prefix cache、long-context scaling。
   回答框架：先确认继续生成时新 token 的 position id 是否接在 cache length 之后，再看 prefix cache / sliding window 是否做了正确偏移，packed batch 中不同样本是否被 attention mask 隔离，目标上下文超过训练长度时是否配置 RoPE scaling、Position Interpolation、YaRN 或继续训练。很多问题不是 RoPE 公式错，而是工程路径中 position id 与真实 token 位置不一致。

31I. 如何系统解释 attention mask、padding mask 和 loss mask 的区别？
   考察点：可见性、padding、训练目标、SFT、信息泄漏。
   回答框架：attention mask 控制 query 能从哪些 key/value 位置读信息；padding mask 标记补齐 token，通常要阻止真实 token 读取 pad，也不让 pad 参与 loss；loss mask 控制哪些位置计算训练损失。SFT 中 user token 通常不算 loss，但 assistant 仍要能看见 user prompt，所以不能把 loss mask 直接当 attention mask。排查时要同时看 decoded text、attention mask、labels、position ids、sample boundary 和 shift。

31J. Prefix LM、bidirectional attention、causal attention 和 packed causal mask 分别解决什么问题？
   考察点：可见性矩阵、理解/生成、跨样本泄漏、kernel fast path。
   回答框架：bidirectional attention 允许非 padding token 互看，适合理解表征；causal attention 只看过去和自己，保证 next-token prediction 不看未来；Prefix LM 让 prefix 内部双向可见、target 因果可见并读取 prefix，适合条件生成；packed causal mask 则在每个样本内部 causal、样本之间完全不可见，防止 packing 造成跨样本泄漏。工程上还要检查全 mask 行和复杂 mask 是否导致 kernel 退化。

31K. 如何从并行性、路径长度和硬件友好性解释 Transformer 的优势？
   考察点：training parallelism、parallel path length、attention pair count、GPU/TPU、矩阵乘。
   回答框架：RNN 的序列内部有时间步递推依赖，远距离信息路径随序列长度增长；小卷积核 CNN 虽然并行，但远距离依赖需要多层扩大感受野。Transformer 在训练时一次性计算 Q/K/V 和所有可见 pair 的 attention score，可见 token 一层内直接交互，大部分计算是规则 dense matmul，适合 GPU/TPU 和分布式系统。代价是 full attention 有 `n^2` pair，长上下文成本会成为瓶颈。

31L. Scaling law、Chinchilla 和 serving pressure 放在一起应该怎么讲？
   考察点：scaling evidence、compute-optimal training、tokens per parameter、undertraining、serving pressure、scaling gate。
   回答框架：Scaling law 说明在特定模型族、数据和训练 recipe 下，loss 随参数、数据和 compute 扩大通常有相对可预测的下降趋势。Chinchilla 修正了“只堆参数”的直觉，强调固定 compute 下参数量和训练 token 数要平衡，很多大模型可能 undertrained。但训练 loss 不是唯一目标，模型还要满足数据质量、训练稳定性、推理成本、KV cache、延迟和安全评估，所以 scaling gate 应同时看 compute balance 与 serving pressure。

31M. 如何系统审计 O(n^2) Attention 的计算与显存瓶颈？
   考察点：attention pair count、score/probability memory、prefill、decode、FlashAttention、PagedAttention、GQA、sliding window。
   回答框架：先写出 `S` 和 `A` 的 shape 是 `[B,H,T_q,T_k]`，full self-attention 中 `T_q=T_k=T`，所以 pair count 是 `BHT^2`，QK^T 和 AV 都随 `T^2` 增长。训练和 prefill 都可能有平方级成本；decode 单步是 `T_q=1,T_k=t`，单步线性但会受 KV cache 和内存带宽限制。显存上要估算单个 score tensor 和 score+prob tensor 的大小，并说明 causal mask 只降常数、不改阶数。FlashAttention 是 exact attention 的 kernel / IO 优化，不是线性 attention；PagedAttention 是 serving KV cache memory manager；GQA / MQA 是模型结构层面的 KV head 压缩。最后用 attention cost gate 同时看 pair count、prefill FLOPs、score memory、window ratio、质量风险和上线 SLO。

31N. 如何系统审计 KV Cache、长上下文推理和显存增长？
   考察点：KV cache formula、per-token cache、prefill/decode、decode bandwidth、MQA/GQA/MLA、PagedAttention、prefix sharing、cache gate。
   回答框架：先写出 KV cache 公式 `L * B * T * H_kv * D_h * 2 * bytes`，强调它是推理时每个活跃请求的运行时状态，不是权重显存。Prefill 中 `T_q=T_k=n`，负责构建 prompt cache 并影响 TTFT；decode 中 `T_q=1,T_k=past+1`，单步计算线性但要读取历史 K/V，常受显存容量和带宽限制。MHA、MQA、GQA 的核心差异是 `H_kv`，GQA/MQA 降低 cache 和读取带宽；MLA 压缩 KV 表示本身。PagedAttention 不是新 attention 公式，而是用 block/page 管理变长请求、碎片、prefix sharing 和 continuous batching。最后用 KV cache gate 同时看 cache GiB、每 token cache 增长、decode 读取量、block 浪费、质量回归和 TTFT/TPOT。

31O. 如何系统审计 In-Context Learning 和显式 token 检索能力？
   考察点：ICL、few-shot demonstrations、label space、format consistency、context evidence use、middle evidence risk、token retrieval audit。
   回答框架：先强调 ICL 不更新参数，而是把任务说明、示例、标签和证据作为上下文条件进入固定参数前向推理。Few-shot 示例不只提供正确答案，还提供标签空间、输入分布、输出格式和任务 framing；因此要检查 label space coverage、示例格式一致性和输入输出映射。显式 token 检索能力要看模型能否定位并使用上下文中的相关示例、文档证据、工具字段或变量定义，而不是只靠参数先验。长上下文还要做位置分桶，特别检查中间证据、首尾偏置、冲突相似示例和噪声 chunk。最后用 ICL gate 把标签空间、格式、retrieval label match、中间证据风险、conflict rate 和验证集准确率组合起来，而不是只凭单个 prompt demo 判断有效。

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

10. SFT 后模型复述用户输入或不停生成，如何从 tokenizer 和格式排查？
    考察点：chat template、special token、assistant-only mask、prompt loss leak、PAD loss leak、EOS、截断、多模态 placeholder。
    回答框架：先打印 messages 渲染结果、token ids、decode 文本、role span、labels 和 attention mask；确认训练与推理模板一致，system/user/control/PAD 都为 ignore index，assistant 与必要 EOS 参与 loss，`eos_token_id` 和 `pad_token_id` 不混，截断没有切掉 assistant/EOS，多模态 placeholder 数量和视觉特征数量一致。

11. LoRA 和全参 SFT 有什么区别？
   考察点：参数更新范围、显存、成本、能力上限、部署。
   回答框架：全参更新所有权重，LoRA 冻结基座只训低秩 adapter，QLoRA 进一步量化基座省显存。

12. SFT 后模型能力下降怎么排查？
   考察点：灾难性遗忘、数据分布、学习率、epoch、label mask、评估。
   回答框架：查数据和训练强度，验证 template/mask，做 base capability eval，必要时混入保能力数据。

13. RLHF 的完整流程是什么？
   考察点：SFT、偏好数据、reward model、PPO、reference model、KL penalty。
   回答框架：SFT 得到初始模型，收集 chosen/rejected 偏好数据，训练 reward model，再用 PPO 优化 policy，并用 KL 约束接近 reference。

14. Reward Model 怎么训练？
   考察点：prompt+response、标量 reward、chosen/rejected、pairwise ranking loss。
   回答框架：输入回答输出分数，用 pairwise loss 让 chosen 分数高于 rejected。

15. 为什么 RLHF 需要 KL penalty？
   考察点：reference model、分布约束、reward hacking、语言质量。
   回答框架：防止 policy 为了高 reward 偏离 SFT 模型太远，降低 reward model 被 exploit 的风险。

16. RLHF 有哪些风险？
   考察点：偏好噪声、reward model 泛化、reward hacking、over-optimization、能力退化。
   回答框架：reward model 只是偏好近似，过度优化会带来高 reward 低人类偏好，需要人工和安全评估。

17. DPO 为什么不需要显式 reward model？
   考察点：preference pair、log probability、KL 约束、直接优化 policy。
   回答框架：DPO 将带 KL 约束的偏好优化转成 chosen/rejected log probability 的直接损失，因此不单独训练 reward model。

18. DPO 和 RLHF 有什么区别？
   考察点：reward model、PPO、在线采样、离线偏好优化、稳定性。
   回答框架：RLHF 是偏好数据训练 RM 再 PPO，DPO 是直接用 preference pair 优化 policy。

19. DPO 中 reference model 有什么作用？
   考察点：SFT baseline、隐式 KL、分布约束、过拟合。
   回答框架：reference 提供原模型偏好基线，约束 policy 不要在偏好数据上偏离太远。

20. DPO 有哪些风险？
   考察点：偏好数据噪声、长度偏差、reference 选择、over-refusal、能力回归。
   回答框架：DPO 工程简单但依赖数据质量，需要评估偏好、安全、事实性和原能力。

20A. SFT / RLHF / DPO 后模型行为异常，如何设计 alignment training gate？
    考察点：assistant mask coverage、prompt loss leak、PAD loss leak、EOS coverage、capability regression、false refusal、unsafe leak、preference margin、reward-human gap、reward length bias、DPO reference、tool schema drift。
    回答框架：先打印 SFT 样本的渲染文本、role span、labels 和 EOS/PAD，确认只对 assistant 回答计算 loss；再比较 base / SFT / aligned model 的数学、代码、安全、工具和真实任务切片；随后统计误拒率和漏拒率，抽查 chosen/rejected 的偏好间隔和标注一致性，检查 reward 分数是否偏好长答案或高 reward 低质量样本；最后确认 DPO reference、beta、tokenizer、chat template 和线上工具 schema 一致。门禁要输出 failed gates，而不是只说 reward 或 DPO loss 下降。

21. Reward Model 学的是什么？
   考察点：prompt+response、标量 reward、偏好排序、chosen/rejected。
   回答框架：RM 学偏好数据中的相对排序信号，近似人类偏好，但不是绝对质量或事实判断器。

22. 什么是 reward hacking？
   考察点：代理目标、RM 漏洞、Goodhart's Law、真实偏好。
   回答框架：模型找到提高 RM 分数的捷径，但输出不符合真实人类意图。

23. 如何评估 Reward Model？
   考察点：pairwise accuracy、ranking loss、margin、分领域、人工评估、分布偏移。
   回答框架：除了验证集排序准确率，还要看分领域、人类一致性、校准和 adversarial 测试。

24. 如何降低 reward hacking？
   考察点：数据质量、KL 约束、holdout、adversarial eval、人工评估。
   回答框架：用多样高质量偏好数据和约束训练，不能只看 reward score，要结合人工和安全评估。

25. 什么是 Alignment Problem？
   考察点：真实意图、目标规范、代理指标、模型行为、部署分布。
   回答框架：Alignment Problem 是让模型实际目标和行为符合人类真实意图、安全边界和产品价值的问题，难点是写下的目标、训练目标、模型学到的策略和部署行为可能不一致。

26. Outer alignment 和 inner alignment 有什么区别？
   考察点：目标写对了吗、模型学对了吗、proxy objective、目标泛化。
   回答框架：outer alignment 关注训练目标是否代表真实意图；inner alignment 关注即使目标设计合理，模型是否学到了正确目标，而不是训练分布上的捷径。

27. Goal misgeneralization 和普通泛化失败有什么区别？
   考察点：能力保留、目标错误、分布外、隐蔽风险。
   回答框架：普通泛化失败是模型不会做，goal misgeneralization 是模型仍有能力但追求了错误目标，因此更难通过能力分数发现。

28. Specification gaming 和 reward hacking 有什么关系？
   考察点：规则漏洞、指标漏洞、reward model、Goodhart's Law。
   回答框架：reward hacking 是钻 reward 或 reward model 的漏洞，specification gaming 范围更广，包括钻规则、benchmark、环境和工具流程漏洞。

29. 面试中如何谨慎表述 deceptive alignment？
   考察点：潜在风险、证据强度、训练/评估/部署差异、避免断言。
   回答框架：把它作为安全研究中的潜在高风险失败模式，强调需要分布外、长期、多轮、工具使用和监督强弱变化下的证据，不能轻率说当前模型已强形式发生。

30. 如何设计一个目标错配评估？
   考察点：真实最优行为、proxy 选中行为、模型实际行为、Goodhart gap、goal misgeneralization。
   回答框架：为每个场景标注真实目标、proxy 分数和模型行为，统计 outer mismatch、behavior mismatch、proxy follow、Goodhart gap、分布外目标误泛化和高严重度失败。

31. 为什么只优化用户满意度可能带来 alignment 风险？
   考察点：短期偏好、过度承诺、事实性、安全边界、长期价值。
   回答框架：用户满意度是 proxy，模型可能学到迎合、过度自信、冗长或越过安全边界的行为，需要同时看事实性、安全、长期指标和人工复核。

32. 什么是 Scalable Oversight？
   考察点：监督瓶颈、复杂任务、AI-assisted eval、verifier、human audit。
   回答框架：当模型输出复杂到单个人类难以直接判断时，通过任务分解、AI critique、Debate、verifier、过程监督和人工审计构造更可靠的监督信号。

33. 为什么 RLHF 会遇到 scalable oversight 问题？
   考察点：偏好标注、人类能力、长上下文、代码数学、专业领域、Agent trace。
   回答框架：RLHF 依赖人类判断输出好坏；当任务需要专家、工具或长时间审查时，普通偏好标注可能只奖励表面流畅性，无法可靠监督真实正确性和安全性。

34. Debate、Iterated Amplification 和 Recursive Reward Modeling 有什么区别？
   考察点：辩论、任务分解、stronger overseer、reward model、错误传播。
   回答框架：Debate 用模型互相指出错误帮助人类判断；Iterated Amplification 用人类加模型助手分解复杂任务形成更强监督者；Recursive Reward Modeling 递归训练子任务评估器来监督复杂任务。

35. AI feedback 能否替代 human feedback？
   考察点：规模、成本、偏差、gold set、校准、高风险审核。
   回答框架：AI feedback 可以扩展监督规模和做初筛，但不能替代人类价值锚点；需要 human gold labels、专家抽检、工具验证、切片校准和高风险人审。

36. 如何设计一个 scalable oversight gate？
   考察点：direct coverage、AI feedback accuracy、verifier coverage、process accuracy、evidence support、high-risk audit、cost。
   回答框架：先定义复杂任务 gold set，再统计人类直接覆盖、AI feedback 校准、verifier 覆盖、过程步骤准确、证据支持、高风险人审覆盖、严重度加权错误和成本节省，任何高风险门禁失败都不能上线。

37. Scalable oversight 和 LLM-as-a-Judge 有什么关系？
   考察点：AI feedback、judge bias、meta-eval、human calibration、工具验证。
   回答框架：LLM-as-a-Judge 是 AI feedback 的一种工程形式；scalable oversight 更广，还包括分解、debate、verifier、process supervision、人工审计和监督质量门禁。

38. 什么是 reward hacking？
   考察点：proxy objective、reward model 漏洞、Goodhart、真实目标。
   回答框架：模型优化奖励或 judge 分数时找到 proxy 的漏洞，得到高分但没有提升真实 helpful、honest、harmless 或任务质量。

39. Reward-human gap 如何发现 reward overoptimization？
   考察点：proxy reward、human eval、held-out gold set、输出分布、error analysis。
   回答框架：如果 reward score 持续上升，但人工偏好、专家评分、事实性、安全性或 citation accuracy 下降，说明优化已经偏离真实目标。

40. 为什么 best-of-N 也会产生 reward hacking？
   考察点：候选数量、极端高分样本、reward model 误判、reranking。
   回答框架：候选越多，越容易采到 reward model 误判的高分低质样本；按最高 reward 选择会放大误判，所以不做 RL 也会过度优化 proxy。

41. KL penalty 为什么只能缓解 reward hacking，不能根治？
   考察点：reference model、分布约束、行为级风险、beta 权衡。
   回答框架：KL 限制 policy 不要远离 reference，降低 reward model 分布外失效，但 reference 也不完美，KL 是整体分布约束，不能保证每个具体行为符合真实目标。

42. 如何设计 reward hacking gate？
   考察点：proxy mismatch、reward hacking rate、reward-human gap、length bias、高风险切片、人工抽检。
   回答框架：为候选回答标注 proxy reward 和 gold quality，统计 proxy mismatch、高 reward 低质量、长度偏置、严重度加权失败和 held-out human win rate；高风险门禁失败就停止上线或降低优化强度。

43. DPO 是否避免了 reward hacking？
   考察点：无显式 reward model、偏好数据隐含目标、chosen/rejected 偏差、分布外。
   回答框架：DPO 避免了显式 RM + PPO 的链路，但仍在优化偏好数据隐含的 proxy；如果偏好数据有长度、风格、安全或领域偏差，仍会学到目标错配。

## 安全与治理

1. AI Safety 和 Alignment 有什么区别？
   考察点：伤害风险、人类意图、HHH、训练目标、部署治理。
   回答框架：Safety 更关注模型是否造成伤害，Alignment 更关注模型行为目标是否符合人类意图和价值；二者重叠但不等价，真实系统要同时处理 helpful、honest、harmless、评估和治理。

2. 为什么安全不是简单拒答？
   考察点：漏拒、误拒、safe completion、over-refusal、正常帮助。
   回答框架：拒答过少会造成 unsafe compliance，拒答过多会损害 helpfulness。好的安全模型要区分明确危险、边界允许和正常请求，并在拒绝危险细节时提供安全替代。

3. 如何设计一个 safety eval？
   考察点：risk taxonomy、样本分层、jailbreak、prompt injection、工具滥用、误拒 / 漏拒。
   回答框架：先定义风险分类和策略期望动作，再构造高风险、正常、边界、多轮、对抗、跨语言和工具调用样本；指标同时看 unsafe compliance、refusal accuracy、over-refusal、attack success、unauthorized tool call、safe completion quality 和 severity-weighted risk。

4. Helpful、Honest、Harmless 如何冲突？
   考察点：多目标权衡、不确定性、危险请求、高风险事实。
   回答框架：Helpful 要尽量解决问题，Honest 要避免编造和表达不确定性，Harmless 要避免促成伤害。冲突时按风险等级和策略边界处理，高风险请求限制粒度或拒绝，正常请求避免过度拒答。

5. 为什么模型能力越强，安全评估越重要？
   考察点：能力放大、危险能力、工具调用、多步规划、分级发布。
   回答框架：更强模型更能理解规则，也更能执行复杂危险任务和使用工具。能力提升必须同步提升红队、安全评估、权限控制、上线门禁、监控和事故响应。

6. Safety gate 应该包含哪些硬门禁？
   考察点：unsafe compliance、attack success、over-refusal、tool safety、coverage、model card。
   回答框架：上线不能只看平均质量分，应设置漏拒、误拒、对抗成功、工具越权、安全替代质量、风险覆盖、严重度加权风险和文档治理门禁；任何高严重度门禁失败都应阻断发布或灰度。

7. 如何设计一次大模型 red teaming 和危险能力评估？
   考察点：risk taxonomy、搜索强度、capability elicitation、baseline、P0/P1、回归测试、发布门禁。
   回答框架：先定义风险分类和成功标准，再设计单轮、多轮、RAG、Agent、工具和高风险能力的抽象评估任务；记录模型版本、prompt、工具条件、是否专家辅助和成本。指标上看 taxonomy coverage、failure rate、P0/P1 unresolved、dangerous capability uplift、elicitation gain、autonomy score、regression pass rate 和 release gate，不把高风险细节写进公开报告。

8. Capability elicitation 为什么会影响危险能力评估结论？
   考察点：自然能力、最大可激发能力、工具、scaffold、专家辅助、低估 / 高估风险。
   回答框架：模型默认表现差可能只是提示、工具或上下文不足。危险能力评估要同时报告 natural capability 和 elicited capability，并说明工具、scaffold、多轮、专家辅助和搜索强度，否则既可能低估风险，也可能用不现实条件高估普通部署风险。

9. Interpretability 和 mechanistic interpretability 有什么区别？
   考察点：输入归因、probing、attention heatmap、activation patching、circuits、因果验证。
   回答框架：普通可解释性常回答哪些输入、token 或激活与输出相关；mechanistic interpretability 更像逆向工程模型内部程序，寻找 features、circuits 和信息流，并用 activation patching、ablation、path patching 等干预验证机制是否真的影响行为。

10. SAE 和 activation patching 如何服务模型安全？
    考察点：superposition、polysemanticity、feature purity、reconstruction fidelity、局部机制证据、安全监控。
    回答框架：SAE 用稀疏重构把混合激活分解为更可解释的 feature 候选，缓解单个 neuron 多语义问题；activation patching / ablation 用来验证这些 feature 或组件是否对拒答、幻觉、jailbreak 等行为有因果作用。上线使用前要看 reconstruction fidelity、sparsity、feature purity、patch recovery、ablation effect 和副作用，不能把一个 feature 名字直接当作安全保证。

11. Steering 和 representation engineering 能否替代 RLHF 或安全系统？
    考察点：steering vector、CAA、activation intervention、目标收益、副作用、over-refusal、双用风险。
    回答框架：不能替代。Steering 更适合局部、可开关的行为控制和研究调试；representation engineering 从高层行为构造内部表示方向，例如 honesty、factuality、refusal。上线前要扫描强度，评估 target uplift、safe gain、side-effect drop、over-refusal delta、jailbreak / prompt injection 回归和 steering gate。真实系统仍需要 RLHF/DPO、安全评估、红队、权限控制、监控和回滚。

12. Model editing 和 machine unlearning 分别解决什么问题，如何评估？
    考察点：ROME、MEMIT、MEND、RAG、fine-tuning、forget set、retain set、locality、membership inference、guardrail 边界。
    回答框架：model editing 是局部修改模型知识或行为，重点看 edit success、改写泛化、specificity / locality、retain drop 和 editing gate；unlearning 是近似移除某些数据、知识或能力影响，重点看 forget leak、改写 / 多轮鲁棒泄露、membership inference 风险下降、retain set 能力保留和 unlearning gate。两者都不能只看一个目标 prompt 成功，guardrail 抑制也不能等同于参数级遗忘。

13. 如何设计 LLM privacy、memorization 和 watermarking 的评估治理？
    考察点：PII / secret、training data extraction、memorization、membership inference、RAG 权限、日志脱敏、watermark z-score、误报 / 漏报、privacy gate。
    回答框架：先按生命周期拆风险：训练数据、微调数据、用户日志、RAG、工具、输出和人工标注。指标上看 PII recall、secret recall、memorization rate、output leak rate、RAG unauthorized retrieval、raw log rate 和 membership advantage，并把它们放进 privacy gate。Watermarking 主要服务内容溯源，要看 generated recall、false positive、改写 / 翻译鲁棒性和质量下降，不能把水印当成隐私保护本身。

14. Model card、system card 和 governance gate 应该如何设计？
    考察点：Datasheets、Model Cards、System Card、intended use、out-of-scope use、risk disclosure、release gate、responsible scaling、approval、monitoring。
    回答框架：model card 记录模型本身的用途、训练数据概要、评估、限制、风险和缓解；system card 记录完整系统，包括 RAG、工具、权限、policy layer、guardrail、监控和事故响应。governance gate 要把 model card 完整度、system card 完整度、policy 覆盖、评估切片覆盖、P0/P1 风险、审批覆盖、监控和回滚准备度转成发布前硬门禁。文档不是宣传材料，必须和真实系统一致并随版本更新。

15. Safety 综合面试题应该如何组织回答？
    考察点：定义目标、risk taxonomy、failure mode、mitigation、eval metrics、red teaming、release gate、trade-off、safe boundary。
    回答框架：先定义目标和风险边界，再列 risk taxonomy；随后分析失败模式，说明训练和系统缓解；再给评估指标和红队方案，最后落到上线门禁、监控、回滚和残余风险。回答中必须避免可复用攻击步骤，并同时覆盖 harmful compliance、over-refusal、privacy、tool misuse、dangerous capability 和 governance。

16. 如何设计大模型安全合规事故审计表？
    考察点：policy action、unsafe pass、over-refusal、PII、permission、tool confirmation、log、training consent、DPA、incident response。
    回答框架：先把样本写成 expected action 和 actual action，再沿输入、RAG、工具、输出、日志、缓存、训练反馈、外部传输和审计链路统计 unsafe pass rate、over-refusal rate、PII redaction coverage、sensitive data block rate、unauthorized access rate、tool confirmation coverage、log redaction coverage、training consent coverage、retention / deletion、external approval、audit coverage、DPA ready 和 incident response ready。任何严重越权、敏感日志、无授权训练或外部传输未审批都应阻断发布。

17. 为什么合规不能只靠免责声明、政策文档或最终输出审核？
    考察点：data flow、system control、audit evidence、legal / policy boundary、model outside system。
    回答框架：免责声明和政策文档只能表达意图，不能证明系统真的执行。真实合规要落在数据最小化、权限过滤、日志脱敏、训练授权、保留删除、外部传输审批、审计记录、系统卡、模型卡和事故响应上。最终输出安全也不能证明中间链路没有越权检索、工具误执行或原始日志泄露。

18. 安全策略同时出现高风险漏拦和正常请求误拒时怎么排查？
    考察点：risk taxonomy、threshold、slice eval、policy version、human review、rollback。
    回答框架：先按风险分类和策略版本拆样本，分清 unsafe pass 和 over-refusal；再看分类器阈值、策略动作映射、RAG / 工具权限、日志和人工复核结果；随后按业务、语言、地区、用户类型和高风险场景切片评估。修复时不要只调一个拒绝阈值，而要补策略分层、正负样本、人工复核、灰度回放和回归门禁。

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

54A. 如何设计一次推理性能事故审计？
    考察点：trace 拆解、TTFT、TPOT、P99、KV pressure、cache、prompt 成本、质量回归。
    回答框架：先把请求 trace 拆成 queue、tokenize、prefill、decode、streaming、postprocess 和客户端接收；再按业务线、输入长度、输出长度、模型版本和调度配置分桶；最后同时检查 P95 TTFT、P95 TPOT、P99 E2E、P95 queue、KV pressure、cache hit token、prompt cost drift、错误 / 取消率和关键切片质量回归，任何硬门禁失败都不能只凭平均 tokens/s 上线。

54B. TTFT regression 和 TPOT regression 分别怎么定位？
    考察点：首响、持续输出、prefill、decode、streaming、三端打点。
    回答框架：TTFT regression 先看 queue、router、prompt 长度、tokenizer、prefill tokens/s、prefix cache 和 RAG / tool 前置耗时；TPOT regression 先看 decode step time、running batch size、KV Cache 访存、显存带宽、stream chunk 发送和客户端接收。两者不能混成一个平均 latency。

54C. cache hit rate 很高就一定省成本吗？
    考察点：命中 token、节省 prefill、cache key、权限、错误复用、内存占用。
    回答框架：不一定。要看命中的 token 数和节省的 prefill 时间，而不是只看请求命中次数；还要检查 cache key 是否包含模型、tokenizer、prompt 版本、租户、权限和文档版本，避免错误复用。缓存占用大量显存但命中 token 很少时，反而可能挤压 KV Cache。

54D. prompt cost drift 为什么会造成推理事故？
    考察点：system prompt、RAG chunk、tool schema、多轮历史、TTFT、KV、成本。
    回答框架：输入 token 增长会同时增加 prefill 时间、TTFT、KV Cache、token 成本和无关上下文干扰。排查时比较新版输入 token 总量和 baseline，按 system prompt、RAG、工具 schema 和历史对话拆分，治理手段包括按需注入、摘要、rerank、context precision、max input token 和业务线预算。

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

37. Agent 和普通 LLM 应用有什么区别？
    考察点：目标驱动、多步执行、状态、工具、观察、控制器、评估。
    回答框架：普通 LLM 应用多是输入到输出的生成系统；Agent 是目标驱动的多步任务执行系统，会维护状态、选择动作、调用工具、读取观察、更新状态并在预算和权限约束下停止。评估时不能只看最终回答，还要看 tool trace、state update、预算、权限和停止条件。

38. Agent、RAG 和 workflow 的边界是什么？
    考察点：知识获取、多步行动、预定义流程、动态决策。
    回答框架：RAG 主要解决从外部知识库取证据并 grounded 生成；workflow 是预定义流程；Agent 关注根据状态和观察动态选择下一步。工程上常把 RAG 作为 Agent 工具，把关键步骤 workflow 化，用 Agent 处理开放分支。

39. Agent 系统有哪些核心模块？
    考察点：goal、state、planner、tool registry、executor、observation handler、memory、controller、logger。
    回答框架：goal 定义目标和验收；state 记录进展；planner 决定下一步；tool registry 描述工具；executor 执行工具；observation handler 解析反馈；memory 处理历史；controller 控制预算、权限和停止；logger / trace 用于调试、审计和评估。

40. 为什么 Agent 需要 controller？
    考察点：预算、权限、超时、重试、停止、人工接管、trace。
    回答框架：模型可以提出动作，但不能无条件执行动作。Controller 负责最大步数、工具调用数、成本、超时、权限检查、二次确认、停止条件和审计日志，防止无限循环、越权动作和成本失控。

41. 如何评估一个 Agent 是否可靠？
    考察点：task success、tool selection、argument validity、observation use、state update、budget、permission、stop。
    回答框架：要记录端到端任务集和 action trace，同时看任务成功率、工具选择准确率、参数合法率、执行成功率、观察使用率、状态更新覆盖率、预算超限率、未授权动作率、停止正确率和 trace 完整性。

42. Agent 常见失败模式有哪些？
    考察点：目标理解、计划、工具、参数、观察、状态、预算、权限、注入、停止。
    回答框架：常见失败包括目标理解错误、计划不合理、工具选错、参数错误、忽略 observation、状态污染、旧记忆误用、无限循环、预算超限、未授权动作、工具结果注入和最终答案与执行结果不一致。

43. Tool use 和 function calling 的核心思想是什么？
    考察点：工具调用、结构化输出、系统执行、外部能力、安全边界。
    回答框架：模型在需要外部信息或能力时输出结构化工具调用，由系统执行真实工具，再把结果返回模型生成最终答案。

44. 工具 schema 为什么重要？
    考察点：工具名、description、参数类型、必填字段、枚举、格式。
    回答框架：schema 告诉模型有哪些工具和参数，清晰 schema 能降低选错工具、漏参数、编造参数和格式错误。

45. 为什么 function calling 不只是输出 JSON？
    考察点：参数校验、权限、执行、错误处理、结果回填、审计。
    回答框架：JSON 只是交互格式，完整系统还要做参数校验、权限检查、工具执行、错误处理、结果回填、日志审计和最终回答。

46. 工具调用如何保证安全？
    考察点：权限控制、参数校验、二次确认、只读/写操作、审计、tool result injection、prompt injection。
    回答框架：系统要做工具级和参数级权限控制，高风险操作二次确认，区分只读和写操作，限制频率，记录审计，并把工具返回标记为不可信 observation，避免外部内容直接驱动高权限工具或污染参数。

47. 如何评估 function calling？
    考察点：tool selection、argument accuracy、execution success、final answer、safety。
    回答框架：评估工具选择准确率、参数准确率、执行成功率、最终答案正确率、安全违规率、漏调用/误调用率、多步成功率、延迟和成本。

48. 工具调用失败时模型应该怎么做？
    考察点：结构化错误、重试、追问、失败解释、禁止编造。
    回答框架：系统返回结构化错误，模型应解释失败、请求补充信息或建议重试，不能编造工具结果。

49. 严格 schema 是否等于工具调用安全？
    考察点：schema validation、业务语义、权限、二次确认、tool result injection。
    回答框架：严格 schema 只能提高字段和类型合法率，不能保证工具选择正确、参数来源可信、用户有权限或高风险动作安全。系统还要做参数来源检查、权限检查、风险策略、二次确认、沙箱、审计 trace 和注入防护。

50. 如何评估一个 function calling 系统的中间过程？
    考察点：tool selection、argument exact match、schema valid、execution success、observation use、recovery、safety。
    回答框架：要记录 tool call trace，并分别统计工具选择准确率、参数 exact match、schema 合法率、执行成功率、observation 使用率、错误恢复率、不必要工具调用率、未授权工具尝试率、tool result injection 违规率、延迟和成本。

50A. Prompt tool use、JSON mode、structured outputs 和 structured function calling 如何区分？
     考察点：自由文本约定、JSON 语法、schema 输出、动作协议、runtime 执行。
     回答框架：prompt tool use 是在自然语言里约定模型输出某种工具调用文本，解析脆弱；JSON mode 主要保证输出可解析为 JSON；structured outputs 让模型按 schema 输出结构化数据，但不一定执行外部动作；structured function calling 则让模型在工具 schema 下生成结构化调用请求，由 runtime 做 schema validation、权限判断、工具执行、tool result 回填和 trace 审计。

50B. 如何设计一个最小 function calling 审计 demo？
     考察点：toy tool schema、tool call trace、schema valid、tool selection、argument、permission、tool result injection。
     回答框架：构造 `get_weather`、`search_docs` 和 `delete_file` 三类 toy 工具，样本覆盖正常调用、选错工具、缺 required 参数、enum 错误、高风险动作未确认和工具结果含不可信指令。脚本输出 schema valid rate、tool selection accuracy、argument exact match、unauthorized block rate、tool result injection rate、failed gates 和 tool calling gate，用来说明严格 schema 不是完整安全系统。

50C. Function calling 输入输出协议层最容易出哪些 bug？如何审计？
     考察点：messages、assistant tool call、tool_call_id、tool result、finish reason、streaming、parallel calls、idempotency。
     回答框架：协议层常见问题包括漏掉 assistant tool call、tool result id 不匹配、finish reason 和 tool_calls 字段不一致、arguments 半截 streaming 就执行、parallel tool results 按数组顺序错配、provider 原始格式泄漏到业务代码，以及有副作用工具重试重复执行。审计时要统计 protocol chain validity、tool result ID match rate、argument parse rate、finish reason consistency、streaming safety rate、parallel alignment rate、idempotency protection rate 和 protocol gate。

50D. 如何设计 Tool Schema 的参数约束审计？
     考察点：JSON Schema 子集、required、type、enum、pattern、range、additionalProperties、business validation、repair、gate。
     回答框架：先把工具 schema 限定在 provider 支持稳定且模型容易生成的子集，再由 runtime 用确定性 validator 检查 required、type、enum、pattern、range 和 additionalProperties。指标上不要只看 schema valid rate，还要拆出 required field pass rate、type valid rate、enum valid rate、pattern valid rate、range valid rate、additional properties block rate、business rule pass rate 和 schema repair success rate。最后用 schema gate 区分格式错误、可安全修复错误和 schema 合法但业务不允许执行的样本。

50E. 如何设计 Tool Choice、强制调用和并行工具调用策略？
     考察点：候选工具过滤、auto/none/required/forced、allowed tools、parallel safety、tool_call_id、confirmation、rate limit、cost、loop。
     回答框架：先由 runtime policy 根据用户权限、页面上下文、租户配置、意图、风险和 workflow 状态过滤候选工具，再决定 `none`、`auto`、`required`、`forced` 或澄清分支。并行只允许独立、低风险、可幂等工具，结果必须按 tool_call_id 对齐；高风险工具必须确认，缺参 forced tool 必须澄清；同时设置 max parallel、cost budget、max steps 和重复调用检测。审计指标包括 candidate coverage、tool choice mode accuracy、no-tool clarification block rate、forced missing argument block rate、parallel safety rate、parallel ID alignment、rate limit pass rate、confirmation enforcement rate、cost budget pass rate、loop control rate 和 tool choice gate。

51. Tool result injection 和普通 prompt injection 有什么区别？
    考察点：不可信 observation、外部网页、RAG 文档、邮件、第三方 API、权限动作。
    回答框架：普通 prompt injection 多来自用户输入或外部文本进入 prompt；tool result injection 来自工具返回，被模型当作 observation 读入。防护要把工具输出标成不可信数据，不允许它覆盖系统规则，并对高风险动作做权限、参数和二次确认检查。

52. 参数格式合法但业务语义错误怎么办？
    考察点：schema 局限、参数来源、所有权、业务校验、澄清问题。
    回答框架：schema 只能检查字段形状。系统还要验证参数是否来自用户输入或可信状态、是否属于当前用户、是否满足业务规则；缺少关键信息时让模型追问，不允许编造参数。

53. Tool registry 和 tool executor 的职责如何区分？
    考察点：工具目录、schema、权限、执行器、超时、日志。
    回答框架：tool registry 管工具定义，包括名称、描述、schema、返回格式、权限、风险、版本和负责人；tool executor 负责运行时校验、授权、执行、处理错误、返回 observation 和记录日志。

54. 为什么最终答案正确也可能不是好的工具调用？
    考察点：过程安全、权限违规、成本、可审计性、过度调用。
    回答框架：工具调用是过程型系统。即使最后答对，如果中间调用了不必要工具、越权访问、忽略工具错误、被工具结果注入误导或成本过高，也不能认为系统可靠。

55. ReAct 解决什么问题？
    考察点：reasoning、acting、observation、外部反馈、多步任务。
    回答框架：ReAct 把 reasoning 和 action 交替组织起来，让模型每一步根据当前状态选择工具或动作，再根据 observation 更新下一步。它适合搜索、调试、浏览器和多步工具任务，但需要循环控制、权限、预算和失败恢复。

56. ReAct 是不是等于展示 chain-of-thought？
    考察点：内部推理、用户可见解释、trace、action 审计。
    回答框架：不是。ReAct 的核心是 reasoning 和 action 的闭环，不是向用户展示完整内部推理。生产系统可以保存决策摘要、action、observation 和 trace，用户侧只展示必要证据、最终答案和失败边界。

57. Plan-Act-Observe 如何落地？
    考察点：plan、act、observe、state update、plan update、controller。
    回答框架：先根据目标生成可修改计划，每次执行一个或少量 action，工具返回 observation 后更新 state 和 plan，再判断继续、final、ask user、blocked 或 stop。工程上需要 action schema、executor、observation parser、预算、权限和 trace。

58. ReAct 和 Plan-Act-Observe 有什么区别？
    考察点：reasoning/action 交替、计划管理、状态更新、组合使用。
    回答框架：ReAct 更强调推理和行动交替，PAO 更强调先计划、执行、观察和更新。真实系统常先生成计划，再用 ReAct 风格逐步执行和读取反馈。

59. 如何评估 ReAct / PAO 循环是否可靠？
    考察点：action accuracy、observation use、state update、repeat action、premature final、stop correctness。
    回答框架：要记录 ReAct trace，统计任务成功率、action accuracy、参数合法率、plan adherence、plan update coverage、observation use rate、state update coverage、parse failure rate、repeat action rate、budget overrun rate、premature final rate、stop correctness 和 blocked recovery rate。

60. Agent 为什么会陷入重复 action 循环？
    考察点：observation ignored、state 不更新、计划不变、缺少停止条件、预算控制。
    回答框架：常见原因是模型没有利用 observation、state 没有记录已尝试动作、计划没有更新、失败工具被反复调用，或缺少 max steps / retry cap / duplicate action guard。工程上要记录 action signature，并在重复时改为换工具、追问用户或停止。

61. 什么是 premature final？
    考察点：停止条件、验收标准、证据、测试通过、权限结果。
    回答框架：premature final 是任务尚未满足验收条件时过早给最终答案，例如测试失败却说修好了、检索无证据却给结论、权限被拒后编造执行成功。防护是把 success criteria 和 final gate 写清楚。

62. Agent 如何做任务分解？
    考察点：目标、约束、子目标、依赖、验收标准、风险。
    回答框架：先明确目标、约束和完成标准，再拆成可执行、可验证的子目标。每个子目标要有输入输出、验收标准和风险级别，然后根据依赖关系安排顺序，执行中根据 observation 更新计划。

63. 如何判断一个计划质量好不好？
    考察点：goal coverage、executable step、acceptance、dependency、replan、risk。
    回答框架：不能只看计划文本是否完整，要看目标覆盖率、可执行步骤比例、验收标准覆盖率、依赖违规率、动态重规划覆盖率、失败恢复率、高风险确认覆盖率、重复子目标率和最终任务成功率。

64. 为什么任务分解需要验收标准？
    考察点：acceptance criteria、success criteria、premature final、验证。
    回答框架：没有验收标准，Agent 可能完成了步骤但没有完成目标。例如“修复 bug”的验收应该包含相关测试通过和无明显回归，而不是只修改代码。

65. 一次性规划和动态重规划如何取舍？
    考察点：流程稳定性、信息完整性、成本、反馈、状态管理。
    回答框架：一次性规划适合流程稳定、目标明确、需要用户确认的任务；动态重规划适合信息不完整、环境反馈强、执行结果决定下一步的任务。实际系统通常先生成粗计划，再根据 observation 修正。

66. 依赖图在 Agent planning 中有什么用？
    考察点：dependency graph、topological order、critical path、parallelism。
    回答框架：依赖图能检查子任务顺序是否合法，决定哪些任务必须串行、哪些可以并行，并计算关键路径。它能避免未验证就部署、未定位失败就修改代码这类错误。

67. Agent 计划为什么会过度分解？
    考察点：粒度、成本、重复子目标、不可执行步骤。
    回答框架：模型可能把“思考、继续分析、查看每一行”拆成大量不可验证步骤，导致成本高、状态冗长和计划僵化。应使用 executable step rate、acceptance coverage、overlong plan rate 和 repeat subgoal rate 审计。

68. 高风险子任务在规划中如何处理？
    考察点：risk level、permission、confirmation、rollback、audit。
    回答框架：高风险步骤要在计划中标注风险级别、权限需求、二次确认、回滚方案和审计记录。即使前置依赖满足，也不能让模型自行决定执行高风险动作。

69. Agent memory 和 context window 有什么区别？
    考察点：context、state、short-term memory、long-term memory、RAG。
    回答框架：context 是这次推理喂给模型的输入窗口，state 是当前任务进展，memory 是跨步骤、跨 session 或跨任务保存并可检索的信息系统。Memory 不能等同于把所有历史塞进 prompt，它需要写入、检索、过滤、更新、删除和权限治理。

70. 如何设计一个 Agent memory 系统？
    考察点：短期记忆、长期记忆、情景记忆、语义记忆、偏好记忆、程序性记忆。
    回答框架：先把短期任务状态和长期记忆分开，再按 semantic / episodic / procedural / preference 分类；每条 memory 记录 user、project、type、key、value、source、timestamp、confidence、sensitivity 和 scope；检索时做相关性、时间、重要性、置信度、权限和过期过滤；写入时做稳定性、价值、来源、隐私、冲突和用户确认检查。

71. Memory 写入为什么比检索更危险？
    考察点：长期影响、来源可信度、一次性指令、工具输出、敏感信息、prompt injection。
    回答框架：写入会改变未来任务状态，一条错误或恶意 memory 可能跨 session 持续影响 Agent。写入门禁要检查 future value、稳定性、source trust、confidence、sensitivity、one-off instruction、injection 和冲突，高影响记忆要用户确认。

72. 只用 embedding top-k 做 memory 检索有什么问题？
    考察点：相似不等于相关、过期、权限、冲突、敏感信息。
    回答框架：embedding 只能给候选相关性，不能判断 memory 是否仍然有效、是否有权限、是否敏感、是否和新事实冲突。生产系统要加 namespace filter、permission gate、recency decay、confidence、importance、staleness penalty、conflict detection 和压缩注入。

73. 如何评估 memory 系统是否可靠？
    考察点：retrieval precision / recall、stale use、unauthorized retrieval、pollution、delete。
    回答框架：既要看 memory 是否提升任务成功率，也要看 retrieval precision、retrieval recall、stale use rate、unauthorized retrieval rate、conflict count、unsafe write block rate、deleted memory returned 和隐私违规率。只看“召回了多少记忆”不够。

74. 如何防止 memory pollution？
    考察点：工具输出、外部文档、模型推断、来源、确认、删除。
    回答框架：把工具和外部文档输出标记为不可信数据，不允许直接变成长期规则；候选 memory 先经过写入门禁；高影响偏好、项目事实和安全规则要用户确认；memory 记录 provenance、timestamp、confidence 和 scope，并支持查看、撤销、删除和审计。

75. Memory 和 RAG 在 Agent 系统中如何配合？
    考察点：外部知识、用户历史、项目事实、向量库复用、治理差异。
    回答框架：RAG 更偏外部知识和证据引用，memory 更偏用户、项目、任务轨迹和偏好。二者都可用向量检索，但 memory 额外需要写入策略、权限隔离、过期删除、用户可控和污染防护。Agent 可以用 RAG 查知识，用 memory 保持长期协作连续性。

76. Agentic RAG 和普通 RAG 有什么区别？
    考察点：固定流程、主动检索、query planning、多轮检索、证据验证、停止条件。
    回答框架：普通 RAG 通常是一次检索加生成；Agentic RAG 把检索放入 Agent 循环，会判断是否检索、拆子问题、改写查询、选择检索工具、多轮检索、验证证据、处理冲突并决定何时停止。它适合复杂、多跳和高可靠场景，但成本和控制难度更高。

77. 如何设计 Agentic RAG 的 retrieval controller？
    考察点：goal、evidence state、query plan、tool selection、budget、stop。
    回答框架：controller 读取原始 goal 和当前 evidence state，识别 evidence gap，生成下一轮 query，选择 dense / sparse / web / SQL / code search 等工具，应用权限和 metadata filter，读取 observation 后更新证据状态，并根据证据覆盖、新证据增益、冲突和预算决定继续或停止。

78. Query rewrite 在 Agentic RAG 中有什么风险？
    考察点：query drift、约束丢失、权限范围、错误假设。
    回答框架：query rewrite 能提升召回，但可能丢失原问题约束、扩大权限范围或引入错误假设。每轮 query 要保留 original goal、must-have constraints、时间范围、用户 / 项目权限和当前 evidence gap。

79. Agentic RAG 如何评估引用是否可靠？
    考察点：claim、evidence、citation accuracy、support rate、旧版本。
    回答框架：把答案拆成 atomic claims，逐条检查引用文档是否真实支持 claim、是否是最新或适用版本、是否有权限展示、是否存在冲突证据。指标包括 evidence support rate、citation accuracy、unsupported claim rate 和 stale citation rate。

80. 多轮检索什么时候应该停止？
    考察点：evidence coverage、新证据增益、冲突、预算、不可回答。
    回答框架：当关键 claim 都有足够证据支持，新证据增益低于阈值，没有未处理冲突，并且预算允许时停止；如果证据不足但继续检索无新增信息，应表达不确定或请求澄清；如果遇到权限不足或高风险冲突，应转人工确认。

81. Agentic RAG 如何防 prompt injection 和越权证据？
    考察点：untrusted content、permission filter、tool output injection、citation permission。
    回答框架：检索前做用户 / 项目 / 文档权限过滤，检索后把外部文档标为不可信 evidence，不允许文档指令覆盖系统规则；对注入模式、工具返回和引用权限做检查，高风险证据隔离并记录 retrieval trace。

82. 如何评估 Agentic RAG 是否值得上线？
    考察点：context precision / recall、support、citation、query drift、cost、latency。
    回答框架：除了最终答案正确率，还要看 context precision、context recall、evidence support rate、citation accuracy、query drift、新证据增益、conflict count、blocked injection、unauthorized evidence return、P95 延迟和单位正确答案成本。只有证明复杂任务质量收益超过成本和风险，才值得开启。

83. Code Agent 和普通代码补全有什么区别？
    考察点：仓库上下文、工具执行、patch、测试反馈、trace。
    回答框架：代码补全主要基于局部上下文生成下一段代码；Code Agent 是仓库级任务执行系统，会搜索、读文件、定位问题、生成 patch、运行测试、根据反馈迭代并总结验证结果。它需要权限、沙箱、状态和 trace。

84. 如何设计可靠的 Code Agent？
    考察点：repo understanding、search、read、patch、test、sandbox、review。
    回答框架：先理解任务和仓库结构，再定位相关文件，修改前读取上下文，遵循最小 patch 原则；修改后运行相关测试、lint 或 build，根据失败反馈迭代。系统层面要有文件编辑工具、命令沙箱、权限、超时、trace 和高风险动作拦截。

85. Code Agent 为什么要遵守最小修改原则？
    考察点：回归风险、review、无关改动、测试。
    回答框架：最小修改降低回归风险，便于 review 和定位问题。它不是不补测试，而是在满足目标和验证的前提下避免无关重构、格式化、依赖变更和跨模块漂移。

86. 如何评估 Code Agent 的 patch 质量？
    考察点：patch localization、unrelated change、test pass、user changes。
    回答框架：看 patch localization precision / recall、changed lines、unrelated change rate、dependency change rate、user change violation rate、测试通过率、验证覆盖和 review 通过率。只看生成代码是否像样不够。

87. Code Agent 如何保护用户已有改动？
    考察点：dirty worktree、diff、read before edit、no rollback、confirmation。
    回答框架：修改前读取文件和 diff 状态，区分用户已有改动和 Agent 本轮改动；不回滚用户未要求的改动；如果必须触碰已有用户改动，应请求确认或说明原因。可用 user change violation rate 做审计指标。

88. Code Agent 的测试策略应该怎么设计？
    考察点：相关测试、全量测试、失败日志、验证覆盖。
    回答框架：先运行与任务相关的小范围测试，修复后再运行更大范围测试或静态检查；如果全量测试太慢或环境不支持，要说明未运行原因。失败时分析第一处关键错误，不重复无效命令。

89. Code Agent 为什么需要 sandbox 和权限控制？
    考察点：文件写入、shell、依赖、敏感文件、超时、trace。
    回答框架：Code Agent 有文件写入和命令执行能力，必须由 controller 检查 action schema、文件范围、权限、预算、超时和风险。高风险命令、依赖安装、锁文件、权限配置和敏感文件都要有额外确认或拦截，并记录 trace。

90. Browser agent 和 API tool use 如何取舍？
    考察点：结构化接口、UI 脆弱性、权限、验证、成本。
    回答框架：有稳定、安全、权限清晰的 API 时优先 API，因为结构化、可验证、鲁棒性更好。Browser agent 适合没有 API 或必须操作现有网页的任务，但要面对页面变化、弹窗、登录、注入和高风险确认。

91. Computer use agent 和 browser agent 有什么区别？
    考察点：浏览器、桌面、截图、窗口、文件系统、剪贴板。
    回答框架：Browser agent 主要操作网页，computer use agent 能操作更通用的 GUI / 桌面环境，包括多窗口、文件管理器、表格软件和终端。后者能力更广，风险也更高，需要更强沙箱、权限和屏幕区域限制。

92. UI Agent 的 observation 应该包含什么？
    考察点：screenshot、DOM、accessibility tree、URL、focus、form state。
    回答框架：观察不应只靠截图，最好组合 DOM、accessibility tree、截图、URL、窗口标题、焦点元素、表单状态、弹窗和错误提示。动作后要重新观察，不能假设页面一定按预期变化。

93. 如何评估 Browser / Computer Use Agent？
    考察点：任务成功、动作准确、误点击、表单、验证、恢复、安全。
    回答框架：除了任务成功率，还要看 final verification rate、action accuracy、misclick rate、form accuracy、state observation coverage、failure recovery rate、high-risk action protection rate、prompt injection block rate、repeat action rate 和人工接管比例。

94. UI Agent 如何处理高风险操作？
    考察点：支付、删除、发送、提交、权限修改、确认、草稿。
    回答框架：高风险动作应先生成摘要和预期后果，请求用户确认；能草稿化就先保存草稿；不能确认时应阻断或降级。系统要记录 trace，并提供取消、回滚或补救路径。

95. 网页 prompt injection 对 Browser Agent 为什么更危险？
    考察点：不可信网页内容、真实动作、工具输出、权限。
    回答框架：网页内容不仅会影响回答，还可能诱导 Agent 点击、填写或提交。系统必须把网页文本视为不可信 evidence，不能让页面指令覆盖系统规则或用户目标，高风险动作仍需权限和确认。

96. UI Agent 为什么不能死记坐标？
    考察点：页面改版、布局变化、弹窗、可访问性树、状态重观测。
    回答框架：坐标会受布局、分辨率、弹窗和 A/B 测试影响。更稳的做法是结合 accessibility tree、DOM、元素语义、截图和状态重观测；动作失败后重新观察，而不是重复点击同一坐标。

97. 什么时候才值得使用 Multi-Agent？
    考察点：复杂任务、专业角色、并行、独立验证、baseline、成本。
    回答框架：当任务需要多个专业角色、可以并行、需要独立审查或需要暴露冲突时才值得使用 Multi-Agent。简单任务应优先单 Agent 或固定 workflow；是否采用要和单 Agent baseline 比较任务成功率、质量、延迟、成本和安全性。

98. 如何设计一个 Multi-Agent coordinator？
    考察点：任务拆解、路由、状态、冲突、预算、停止条件。
    回答框架：coordinator 负责理解目标、拆分子任务、选择角色、控制上下文和工具权限、维护 blackboard、处理冲突、控制预算并生成最终结果。工程上最好把它设计成状态机、路由器和审计器，而不是任意聊天者。

99. Multi-Agent 的通信协议应该包含哪些字段？
    考察点：sender、receiver、task id、intent、evidence、confidence、status。
    回答框架：消息至少包含 sender、receiver、task_id、intent、content、evidence、confidence、status 和 risk_level。结构化消息能支持 trace replay、自动评估、证据检查、权限审计和冲突定位。

100. Multi-Agent 如何处理冲突？
     考察点：事实冲突、方案冲突、工具结果、judge、verifier、人工升级。
     回答框架：先记录冲突来源和证据，再按风险等级处理。低风险冲突可用证据、工具验证、规则或 judge 裁决；高风险冲突应升级给人工或确定性系统，不应让模型闭环强行统一。

101. Debate、judge、verifier 和 voting 有什么区别？
     考察点：自然语言反驳、裁决、工具验证、聚合、适用场景。
     回答框架：debate 用多个视角暴露假设和反例；judge 做结构化裁决；verifier 用规则、工具、测试或证据检查结论；voting 聚合多个相对独立候选答案。工程上高风险任务不能只靠 debate 或投票，最好结合 verifier 和人工复核。

102. 如何评估 Multi-Agent 系统？
     考察点：任务成功、单 Agent 增益、角色匹配、消息、证据、冲突、重复、权限、成本。
     回答框架：除了任务成功率，还要看 single-agent lift、role match rate、message valid rate、evidence support rate、conflict resolution rate、duplicate work rate、permission violation rate、unnecessary multi-agent rate、平均成本、延迟和人工接管率。

103. Multi-Agent 为什么更需要权限隔离？
     考察点：最小权限、工具范围、上下文隔离、高风险动作、trace。
     回答框架：Multi-Agent 会扩大工具和通信面，如果所有 Agent 都有全量权限，就等于复制多个高权限 Agent。更稳的做法是按角色限制工具、上下文和动作，高风险动作统一由 controller 审批并记录 trace。

104. 什么是不必要的 Multi-Agent？
     考察点：小任务、过度编排、重复劳动、成本收益。
     回答框架：如果任务简单、单 Agent 已能稳定完成，拆成 planner、writer、reviewer 只增加 token、延迟和冲突，就是不必要 Multi-Agent。判断标准是它是否带来可度量的质量、验证、并行或安全收益。

105. Agent 系统应该如何评估？
     考察点：任务成功、部分成功、工具、trace、恢复、安全、成本。
     回答框架：先定义任务验收标准，再记录完整 action / observation / state trace。可验证任务用自动验收，开放任务用人工 rubric 和 LLM judge 辅助，同时报告 task success、partial score、tool accuracy、argument validity、observation use、state update、summary faithfulness、recovery、安全违规、成本和延迟。

106. 为什么 Agent 不能只看最终答案？
     考察点：执行过程、工具调用、测试、权限、summary 忠实性。
     回答框架：最终答案可能掩盖未执行、未验证、越权、忽略错误或重复动作。Agent 的价值和风险都在 trace 里，所以必须审计每一步 action、参数、observation、状态更新和最终总结是否忠实于实际执行。

107. 如何设计 Agent benchmark？
     考察点：初始状态、工具、权限、验收器、日志、重置。
     回答框架：benchmark 要固定任务分布、初始环境、工具集合、权限策略、验收器、日志格式和重置机制。还要覆盖代码、浏览器、RAG、数据分析、多工具、失败恢复、安全边界和多轮回归，避免只测 happy path。

108. 什么是 trace faithfulness / summary faithfulness？
     考察点：最终声明、trace 支持、工具结果、测试记录。
     回答框架：它衡量最终总结中的 claim 是否被执行轨迹和工具结果支持。比如声称测试通过，就必须在 trace 中看到测试命令和通过结果；没有证据的 claim 要计为不忠实。

109. Agent 的错误恢复能力如何评估？
     考察点：工具失败、权限不足、页面变化、测试失败、机械重试。
     回答框架：评估集要主动包含失败路径，检查 Agent 是否识别错误类型、更新状态、调整策略并完成恢复。只在没有失败的样本上成功，不能证明恢复能力。

110. 自动评估、人工评估和 LLM judge 如何组合？
     考察点：程序化验收、rubric、judge bias、开放任务。
     回答框架：可执行任务优先程序化验收，例如测试、页面状态或文件检查；开放任务用人工 rubric；LLM judge 可做辅助分类、完整性和证据检查，但不能作为唯一裁判，尤其不能替代安全审计和工具验证。

111. Agent 成本和延迟为什么必须进评估报告？
     考察点：token、工具调用、重试、人工审阅、P95、单位成功成本。
     回答框架：Agent 往往多轮调用模型和工具，成功率提升可能伴随巨大成本和延迟。评估应同时报告平均成本、P95 延迟、工具调用次数、重试成本和每成功任务成本。

112. Agent 回归集应该包含什么？
     考察点：历史成功、历史失败、安全边界、工具异常、长任务、多轮。
     回答框架：回归集要覆盖常见成功任务、历史失败任务、安全边界、工具异常、长上下文、多轮任务、高成本任务和环境变化任务。每次更新 prompt、工具 schema、模型、controller 或权限策略都应重跑。

113. Agent 安全应该如何设计？
     考察点：最小权限、沙箱、数据流、审计、人工确认。
     回答框架：从权限、执行环境、数据流和审计四层设计。权限上区分只读、写入、外部传输和高风险工具；执行上使用沙箱；数据流上把网页、文档和工具返回视为不可信数据；审计上记录工具调用、权限检查、阻断和用户确认。

114. 为什么 Agent 安全不能只靠 prompt？
     考察点：系统层门禁、工具执行器、权限检查、模型不可靠。
     回答框架：模型可能误解、遗忘或被不可信内容影响。安全边界必须由 controller、tool executor、permission gate、sandbox、data flow guard 和 audit log 强制执行，prompt 只能作为行为引导，不能作为唯一防线。

115. 如何做 Agent 的最小权限？
     考察点：用户身份、任务范围、工具风险、时效、作用域。
     回答框架：按用户、任务、工具、数据范围和风险等级生成权限矩阵。默认只给只读和必要工具，写入、外部传输和不可逆动作需要额外授权、确认和审计。

116. Agent 如何处理不可信内容？
     考察点：网页、文档、邮件、工具返回、指令层级、隔离。
     回答框架：外部内容只能作为 evidence，不能覆盖系统规则或用户目标。系统要给内容打来源、权限和风险标签，高风险动作必须重新经过权限检查、数据流检查和用户确认。

117. 高风险动作确认界面应该展示什么？
     考察点：动作、对象、参数、影响范围、dry-run、取消。
     回答框架：确认界面应展示即将执行的动作、影响对象、参数摘要、风险说明、dry-run 或预览结果、可取消选项和审计记录，而不是只问“是否继续”。

118. Agent 如何防止数据泄露？
     考察点：租户隔离、敏感数据过滤、外部传输、日志、memory。
     回答框架：先做用户和租户权限隔离，再在工具输入、工具输出、日志、memory 和最终回答中做敏感数据检测、脱敏和阻断。内部数据传给外部工具前必须经过 external transfer gate。

119. Agent 安全评估应该看哪些指标？
     考察点：越权、注入、不可信内容、敏感数据、高风险、沙箱、审计。
     回答框架：看 unauthorized attempt block rate、untrusted instruction block rate、tool output block rate、sensitive data block rate、external transfer block rate、high-risk protection rate、dry-run coverage、sandbox block rate、memory pollution block rate 和 audit completeness。

120. Multi-Agent 安全为什么更复杂？
     考察点：跨 Agent 传播、共享状态、权限混乱、责任不清。
     回答框架：一个 Agent 接触不可信内容后可能影响其他 Agent，共享 blackboard 和 memory 也可能传播错误。每个 Agent 都应最小权限，跨 Agent 消息要带来源和风险标签，高风险动作统一由 controller 审批。

121. 一道 Agent 面试题怎样才算回答完整？
     考察点：概念、公式、demo、trace、评估、安全、项目证据、trade-off。
     回答框架：完整回答应先给定义和边界，再讲系统模块、关键指标、一个最小 demo 或 trace 数据结构、评估方式、安全门禁、项目经验和取舍。只讲术语或只讲最终效果都不完整。

122. 如何设计 Agent interview readiness rubric？
     考察点：answer coverage、formula coverage、demo coverage、safety coverage、evaluation coverage、project evidence、red flags。
     回答框架：把每道题拆成概念覆盖、公式指标、demo、trace 指标、安全边界、评估口径、项目证据和取舍深度几个维度，给出加权分。平均分、最低单题分、安全覆盖和红旗数量共同决定是否准备好。

123. Agent 面试中最常见的红旗是什么？
     考察点：浅定义、缺 controller、缺 trace、缺评估、缺安全、项目夸大。
     回答框架：常见红旗包括把 Agent 简化成会调用工具的模型、把 function calling 当安全机制、只讲 ReAct 不讲停止条件、只报最终成功率、项目没有 baseline / bad case / 个人贡献、安全只靠 prompt。

124. 如何把 Agent 项目讲得可信？
     考察点：baseline、metrics、trace、tests、bad case、ownership、trade-off。
     回答框架：先说明业务目标和约束，再给 baseline、核心模块、评估指标、trace 样例、测试或验收、失败样本、个人负责部分和下一步改进。重点是证据链，而不是堆框架名。

125. 为什么 Agent 面试必须会讲公式和指标？
     考察点：task success、tool accuracy、argument validity、observation use、summary faithfulness、safety gate。
     回答框架：Agent 是过程型系统，指标能把失败拆开。任务成功率告诉最终结果，工具和参数指标看动作质量，observation / state 指标看反馈闭环，summary faithfulness 看总结是否忠实，安全门禁看权限和数据流风险。

126. 如何复盘一轮 Agent mock interview？
     考察点：weak question、missing concept、missing formula、missing demo、revision plan。
     回答框架：先按 Agent 总览、tool use、ReAct、planning、memory、Agentic RAG、Code Agent、UI Agent、Multi-Agent、evaluation 和 safety 分类；再为每题记录缺失概念、缺失指标、缺失 demo、红旗和修复动作；最后每个弱题绑定一个 3 分钟回答模板。

127. Agent interview gate 应该怎么设？
     考察点：平均分、最低单题分、安全覆盖、红旗、项目证据。
     回答框架：可以要求平均回答分达到阈值，最低单题分不过低，安全覆盖达到阈值，红旗数量为 0，并且至少一个项目能提供 baseline、指标、bad case、测试和个人贡献证据。

128. Agent 面试最后一轮总复盘应该覆盖什么？
     考察点：Agent loop、tool calling、planning、memory、RAG、code、browser、multi-agent、eval、safety、project。
     回答框架：最后复盘应覆盖 Agent loop、tool registry / executor、ReAct / PAO、任务分解、memory、Agentic RAG、Code Agent、Browser / Computer Use、Multi-Agent、Agent evaluation、Agent safety 和项目深挖，确保每个方向都有公式、demo、评估和 trade-off。

## 数据工程

1. 如何从零构建 web-scale 预训练数据采集 pipeline？
   考察点：source registry、license / ToS / robots、raw storage、parser、quality filter、PII scanner、dedup、versioning。
   回答框架：先定义模型目标和数据需求，再建立合法数据源 registry；采集后保留原始数据、来源、hash 和时间，随后做解析、正文抽取、语言 / 领域识别、质量过滤、PII / 密钥 / 安全扫描、污染检测、去重、配比和版本审计。

2. 为什么 web-scale 数据采集不是简单写爬虫？
   考察点：合法来源、访问边界、版权、隐私、数据质量、评估污染、可追溯。
   回答框架：技术可访问不等于可训练；采集系统要证明数据来源、许可、访问策略、质量、风险和版本都可审计，否则只是拿到文本而不是构建训练数据。

3. source registry 应该记录哪些字段？
   考察点：source id、source type、license、ToS、robots、access mode、risk level、owner、crawl version、delete policy。
   回答框架：registry 要能回答“这个数据能不能用、从哪里来、什么时候进来的、进入了哪个版本、未来如何删除或回溯”。

4. 如何解释 robots、ToS、license 和 provenance 的关系？
   考察点：访问策略、网站条款、使用许可、样本级血缘。
   回答框架：robots 和访问控制约束采集行为，ToS 和 license 约束使用方式，provenance 保证后续审计、删除、污染排查和模型版本追踪。

5. Web 数据进入训练前要做哪些风险扫描？
   考察点：低质量、重复、PII、密钥、毒性、有害内容、benchmark contamination、版权风险。
   回答框架：先做元数据和合规门禁，再做解析质量、PII / secret、安全和污染扫描，最后用 exact / near dedup、抽样审计和版本报告确认可进入训练候选集。

6. 如何设计训练数据的质量评分？
   考察点：长度、unique ratio、语言 / 领域、重复率、乱码、boilerplate、PPL、质量分类器。
   回答框架：先按语言、领域和来源分桶，再融合规则特征、统计特征和模型分数；质量分不能单独决定去留，还要叠加隐私、安全、污染、去重和人工抽检门禁。

7. 数据过滤中的误删和漏删分别是什么？
   考察点：false positive、false negative、人工审计集、阈值校准、覆盖率。
   回答框架：过滤系统的误删是好数据被删掉，漏删是坏数据进入训练；阈值不能只追求保留率，要在人工审计集上看误删率、漏删率和不同语言 / 领域桶的覆盖损失。

8. 为什么困惑度过滤不能用一个全局阈值？
   考察点：语言差异、代码 / 数学 / 专业文本、参考模型偏差、桶内校准。
   回答框架：PPL 受参考模型和文本类型影响很大，代码、数学、低资源语言可能天然 PPL 高；更稳妥的是在同语言、同领域、同来源桶内看异常值，并把 PPL 当作质量特征而不是唯一裁判。

9. PII、secret 和安全过滤应该如何落到工程流程？
   考察点：正则、NER、secret scanner、分类器、隔离池、审计日志、合规参与。
   回答框架：PII / secret 通常先用规则和 scanner 做高召回拦截，再用上下文模型和抽样审计降低误伤；安全风险内容要区分有害生成样本和正当安全讨论，必要时进入隔离或专门标注数据池。

10. 如何验证清洗过滤真的提升了训练数据？
    考察点：token 保留率、质量分分布、风险命中率、语言 / 领域配比、小模型 ablation、人工审计。
    回答框架：不能只看删了多少，要比较过滤前后的质量分、重复率、PII / secret / 污染命中率和分桶覆盖；最终用抽样人工审计和小模型预实验确认过滤策略没有牺牲关键能力。

11. exact dedup 和 near dedup 的区别是什么？
    考察点：规范化 hash、Jaccard、MinHash、SimHash、转载 / 改写 / 模板差异。
    回答框架：exact dedup 用规范化文本 hash 找完全重复，便宜且可解释；near dedup 处理轻微改写、转载、镜像、代码 fork 和模板差异，需要 n-gram、MinHash/LSH、SimHash 或 embedding 辅助，并用抽样审计控制误删。

12. MinHash 和 LSH 在大规模去重里分别解决什么问题？
    考察点：shingle 集合、Jaccard 估计、signature、候选召回、避免全量两两比较。
    回答框架：MinHash 把文档 shingle 集合压成签名，用签名相同率估计 Jaccard；LSH 把相似签名放进候选桶，减少比较量。LSH 只负责召回，最终仍要计算相似度和做保留策略。

13. SimHash 和 MinHash 怎么选？
    考察点：汉明距离、网页近重复、Jaccard、短文本、代码 / 多语言、参数审计。
    回答框架：MinHash 更直接对应集合 Jaccard，适合 n-gram near dedup；SimHash 指纹短、速度快，适合快速网页近重复检测。两者都需要按语言、领域、长度和误删风险调阈值。

14. benchmark contamination 应该怎么检测？
    考察点：train-eval overlap、题目 / 答案 / 解析、代码签名、near duplicate、embedding 检索、时间切分。
    回答框架：把评测集题目、选项、答案、解析、函数签名和测试样例作为 query，在训练语料中做 exact、n-gram、MinHash/LSH、embedding 和人工审计；高风险命中要隔离或在评测报告中标注。

15. canary 在数据工程里有什么用？
    考察点：受控合成字符串、记忆风险、出现次数、隐私边界、禁止真实敏感信息。
    回答框架：canary 是受控探针，用合成罕见字符串评估模型记忆和泄漏风险；它不能使用真实用户隐私或真实凭据，目标是验证去重、隐私过滤和训练策略是否降低复述风险。

16. 去重为什么不是越彻底越好？
    考察点：data mixture、常见表达、代码 API、数学题型、多样性、误删。
    回答框架：去重是在控制不合理重复和记忆风险，不是消灭所有相似内容。过度去重会削弱常见模式、代码 API 用法、数学题型和用户意图多样性，因此要按来源、质量、语言和目标能力做保留或降权。

17. 什么是 data mixture，为什么它不是简单拼数据？
    考察点：训练分布、隐式目标函数、来源 / 语言 / 领域 / 能力维度、token budget。
    回答框架：data mixture 决定 next-token loss 在哪些数据分布上求期望，因此会影响模型能力、风格和风险。它不是把文件拼起来，而是在有限 token budget 下设计模型读什么、读多少和以什么权重读。

18. 多语言或多领域训练为什么要做 temperature sampling？
    考察点：高资源语言、低资源语言、平滑采样、过拟合、重复 exposure。
    回答框架：按原始 token 数采样会让大语种或大来源主导，均匀采样又可能让小数据池过拟合。temperature / smoothing sampling 在自然分布和均匀分布之间折中，提高低资源或高价值池的曝光，同时控制 effective epoch。

19. 如何给数据池设置采样权重？
    考察点：清洗后 token 数、质量分、目标能力、风险分、许可证、安全、采样预算。
    回答框架：先按清洗后 token 数得到自然配比，再叠加质量、能力目标和风险惩罚，得到可审计的采样分数；最终还要检查 token 保留率、effective epoch、风险率和能力覆盖。

20. effective epoch 为什么重要？
    考察点：上采样、小数据池、记忆风险、题库污染、过拟合。
    回答框架：effective epoch 表示一个数据池在计划训练预算中被重复使用了多少轮。小而高价值的数据池可以上采样，但 effective epoch 过高会增加记忆和污染风险，需要用去重、扩增、降权或隔离控制。

21. data mixture ablation 应该怎么设计？
    考察点：baseline mixture、单变量变化、小模型预实验、多维评估、放大验证。
    回答框架：先固定数据版本和训练 recipe，设计 baseline mixture，再单独调整代码、数学、多语言、高质量数据或合成数据比例；用通用、代码、数学、多语言、安全、事实性和人工样例评估 trade-off，最后在更大规模复验。

22. 合成数据配比过高有什么风险？
    考察点：同质化、教师偏差、模板化、事实错误、真实分布偏移。
    回答框架：合成数据适合补稀缺能力，但比例过高会让模型拟合生成器风格和偏差，降低自然性和多样性。需要单独标记、质量过滤、去重、执行验证和 ablation，而不是无追踪混入自然语料。

23. 代码数据治理和普通文本治理有什么不同？
    考察点：license、secret、fork、vendor、自动生成文件、语法解析、测试、benchmark 污染。
    回答框架：代码数据天然符号多、重复多、许可证复杂，不能套普通文本规则；要按仓库 / 文件 / 函数 / 测试粒度做解析、secret 扫描、license 审查、fork 去重和评测题隔离。

24. 为什么单元测试对代码数据特别重要？
    考察点：functional correctness、输入输出约束、边界条件、测试生成、HumanEval。
    回答框架：代码生成不能只看文本相似度，单测提供可执行约束，能验证功能正确性。高质量 code data 应包含需求、实现、测试和文档的对齐样本。

25. 数学数据为什么需要答案验证和过程验证？
    考察点：final answer、step-by-step、verifier、错误步骤、题库污染。
    回答框架：数学样本答案对但过程可能错，过程顺但答案可能错；高质量数学数据要同时检查题目、步骤、答案和难度，并隔离 benchmark 原题、答案和解析。

26. verifier 数据在数学推理里有什么价值？
    考察点：候选解、正确性标签、多候选采样、rerank、GSM8K。
    回答框架：verifier 数据让模型或独立 verifier 学会判断候选推理是否可靠。它能配合多候选生成提升数学题求解，而不是只学习单一标准答案。

27. 专业领域数据为什么不能只靠继续预训练解决？
    考察点：时效性、引用、隐私、专家审核、RAG、SFT、风险边界。
    回答框架：继续预训练能增强领域语言和术语，但高风险领域需要可更新知识、出处和权限控制。医学、法律、金融通常要结合 RAG、SFT、工具、专家审计和安全拒答边界。

28. 如何审计 code / math / domain 三类专项数据是否可进入训练？
    考察点：类型专属质量分、风险门禁、token 保留率、effective epoch、评估矩阵。
    回答框架：代码看 license、secret、测试和去重；数学看答案、过程、verifier 和污染；领域看权威性、时效、引用、PII 和专家审计。最后统计分类型保留率、采样权重和训练 / 评估副作用。

29. synthetic data 和 distillation data 有什么区别？
    考察点：生成来源、teacher-student、规则生成、LLM 生成、训练目标。
    回答框架：synthetic data 强调数据由规则、程序、模拟器或模型生成；distillation data 强调 teacher 输出被 student 学习。两者可以重叠，例如 teacher 生成的指令回答既是合成数据也是蒸馏数据。

30. Self-Instruct 的核心不是“让模型多生成”，而是什么？
    考察点：seed task、instruction generation、filtering、dedup、diversity、evaluation。
    回答框架：Self-Instruct 的关键是自举生成后过滤无效、重复和相似样本，再通过训练评估验证收益。生成数量不是核心，质量控制和任务覆盖才是核心。

31. Evol-Instruct 为什么有用，风险是什么？
    考察点：复杂指令、约束增加、多步骤任务、伪复杂、约束冲突。
    回答框架：Evol-Instruct 用改写和演化提高指令复杂度，补普通合成指令偏简单的问题；但复杂不等于高质量，需要检查指令是否可回答、约束是否一致、答案是否满足要求。

32. 如何建设一个合成数据生成与审计 pipeline？
    考察点：目标能力、seed、teacher 授权、prompt registry、sampling、验证、去重、污染、安全、配比、ablation。
    回答框架：先定义要补的能力和 seed 分布，再选择授权 teacher 或规则生成，记录 prompt / sampling 参数；生成后做格式、质量、正确性、安全、PII、污染和去重审计，控制 synthetic ratio，并用小模型 ablation 验证收益和副作用。

33. 蒸馏数据为什么不能直接当 gold label？
    考察点：teacher hallucination、teacher bias、授权、验证、student 容量、评估。
    回答框架：teacher 只是更强或更稳定的生成器，不是真值来源。它的输出要经过正确性验证、授权检查、污染隔离和人工抽检，否则 student 会继承错误、偏差和风格。

34. model collapse 和合成数据有什么关系？
    考察点：generated data recursion、自然数据锚点、长尾丢失、同质化、比例门禁。
    回答框架：如果模型生成数据逐代替代自然数据，生成器偏差和错误会被放大，长尾信息会被压缩。防范方法是保留高质量自然数据、控制合成比例、做多样性审计、验证和真实分布评估。

35. preference data 和普通 SFT 数据有什么不同？
    考察点：prompt/chosen/rejected、比较信号、helpful / honest / harmless、reward model、DPO。
    回答框架：SFT 给单个理想回答，preference data 给多个回答之间的相对偏好。它适合训练 reward model 或 DPO，让模型学会在多个可行回答中选择更符合行为准则的输出。

36. 如何审计 chosen / rejected 偏好数据质量？
    考察点：标注一致性、preference margin、rejected 难度、长度偏置、rubric、隐私和污染。
    回答框架：先检查 prompt 分布和 rubric，再看标注者一致率、chosen/rejected 分差、rejected 是否太弱、chosen 是否更安全真实、是否存在长回答偏置、PII 和 benchmark contamination。

37. 安全数据为什么不能只有拒答样本？
    考察点：over-refusal、boundary allowed、安全可答、误拒修复、helpfulness。
    回答框架：只有拒答会让模型看到风险关键词就拒绝，损害正常教育、防御和专业科普场景。安全数据要同时包含高风险拒答、边界允许、安全替代建议和误拒修复样本。

38. 如何同时衡量误拒和漏拒？
    考察点：over-refusal rate、unsafe leak rate、高风险请求、安全请求、双指标门禁。
    回答框架：在高风险样本上看应拒未拒的漏拒率，在安全可答样本上看误拒率。上线不能只优化拒答率，要同时约束漏拒、误拒、拒答质量和正常帮助能力。

39. 红队数据应该如何进入训练数据治理？
    考察点：failure mode、risk label、安全响应、回归测试、防御性使用、避免传播细节。
    回答框架：红队发现要转成风险类别、失败模式、安全响应和回归集，用于补安全训练和上线门禁；进入训练时保留必要上下文和安全标签，不传播可操作风险细节。

40. LLM judge 能否替代偏好标注？
    考察点：成本、初筛、偏差、长度偏好、同源模型偏袒、高风险复核。
    回答框架：LLM judge 可做初筛、辅助解释和低风险批量标注，但不能无审计替代人类。边界样本、高风险、专业领域和关键评测需要人工或专家复核。

41. 多模态数据和纯文本数据治理最大的区别是什么？
    考察点：媒体质量、跨模态对齐、时间戳、隐私、版权、评测污染、任务配比。
    回答框架：纯文本主要治理 token 序列，多模态还要证明图像、OCR、音频、视频和文本彼此对应；任何媒体质量、对齐、时序、隐私、来源或污染环节失败，都会让模型学到错误的感知-语言关系。

42. 如何审计 image-text pair 的质量？
    考察点：caption / alt text、CLIP 相似度、OCR、装饰图、人工抽样、license。
    回答框架：先记录来源、license、图像 hash 和文本来源，再用图文相似度、OCR 命中、对象 / 属性检查和人工抽样判断文本是否真的解释图片；相似度分只能辅助过滤，不能替代任务级验证。

43. OCR 和文档图像数据为什么不能只保存纯文本？
    考察点：bounding box、阅读顺序、版面、表格、票据、截图、PII。
    回答框架：文档理解依赖空间结构，纯文本会丢失列关系、表格结构、标题层级和图文位置；高质量 OCR 数据要保留文字、位置框、页码、置信度、阅读顺序和脱敏标记。

44. 语音转录数据应该如何做质量审计？
    考察点：WER / CER、采样率、噪声、VAD、时间戳、多语言、口音、隐私。
    回答框架：音频侧看清晰度、噪声、采样率和切片，文本侧看转录准确率、WER / CER 和语言覆盖；同时检查 transcript 和音频时间对齐、说话人隐私、评测污染和授权来源。

45. 视频文本数据为什么特别强调 temporal alignment？
    考察点：字幕延迟、片段切分、镜头、事件时间范围、视频标题泛化、版权。
    回答框架：视频是时间序列，caption 或字幕必须对应具体片段和事件。字幕错位、标题泛化到整段视频或片段切分不当，会让模型把错误画面、声音和文本绑定在一起。

46. 多模态指令数据如何避免视觉幻觉和无依据回答？
    考察点：grounded answer、证据支持、OCR / 视频证据、unsupported claim、人工抽检。
    回答框架：问题要能由媒体内容回答，答案要被图像、OCR、音频或视频片段支持；对无法从媒体中验证的回答标记为 hallucinated / unsupported，并用 grounded QA、引用检查、人工抽样和回归评测做门禁。

47. 多模态数据里的隐私和版权风险为什么更复杂？
    考察点：人脸、车牌、证件、屏幕截图、声纹、OCR PII、license、删除追溯。
    回答框架：敏感信息可能藏在像素、背景、OCR 和声音里，文本侧没有 PII 不代表媒体安全；数据 pipeline 要在媒体解析阶段做视觉 / 音频隐私标记、许可证审查、脱敏、删除策略和版本追溯。

48. data attribution 和 data valuation 有什么区别？
    考察点：归因、估值、错误诊断、资源决策、数据源、目标指标。
    回答框架：attribution 问模型行为或错误可能来自哪些训练数据，偏诊断和审计；valuation 问数据对目标指标、风险和成本的决策价值，偏数据选择、采购、标注和配比。

49. 为什么大模型训练数据很难做精确估值？
    考察点：训练成本、非凸优化、数据交互、多目标指标、阶段依赖、配比耦合。
    回答框架：逐条删数据重训不可行，单条样本影响很小且数据之间有交互；价值还依赖模型规模、训练阶段、目标评估集、现有分布、风险和成本，所以只能用近似证据链。

50. Influence functions 在数据归因中的核心思想和局限是什么？
    考察点：训练点权重扰动、测试 loss、Hessian inverse、非凸、LLM 规模。
    回答框架：它估计某训练样本权重微小变化对测试样本 loss 的影响；经典形式可解释，但大模型中 Hessian 相关计算昂贵且近似不稳定，通常只作为小模型或局部诊断工具。

51. Data Shapley 为什么有吸引力，为什么不能直接用于全量 LLM 预训练数据？
    考察点：平均边际贡献、数据交互、公平性、子集枚举、近似。
    回答框架：Shapley 用不同子集中的平均边际收益衡量数据价值，理论性质好；但精确计算需要大量训练或近似训练，大模型全量 token 上不可直接使用，更适合小数据池、源级估值或估值思想。

52. 如何实际评估一个数据源是否值得保留或上采样？
    考察点：目标效用、质量、风险、源级 ablation、小模型 proxy、副作用、成本。
    回答框架：先定义目标指标和风险约束，记录来源元数据；再看质量、污染、隐私和 license；用小模型 proxy、源级 ablation、上/下采样实验和人工审计看收益与副作用，最后结合成本形成版本化决策。

53. 梯度相似或 embedding 相似为什么不能单独决定数据选择？
    考察点：弱信号、评测过拟合、benchmark contamination、隐私风险、多样性。
    回答框架：相似度只能说明数据和目标任务接近，不说明它合法、安全、无污染或能泛化。高相似 benchmark 泄漏样本必须阻断，过度选择相似数据也会牺牲多样性和长尾能力。

54. 什么是负价值数据，如何发现它？
    考察点：错误标注、过时知识、低质合成、安全误拒、污染、隐私、ablation。
    回答框架：负价值数据会降低目标效用或提高风险。可以通过异常 loss、质量分、错误归因、人工审计、源级 ablation、安全回归、污染检测和版本对比定位，再重洗、降权、隔离或删除。

55. 主动学习在大模型数据工程中如何落地？
    考察点：标注预算、不确定性、模型分歧、风险、覆盖、成本、专家标注。
    回答框架：先定义要补的能力和风险边界，再用模型不确定性、分歧、用户频率、风险等级、代表性、多样性和标注成本给样本排序；常用于 SFT、偏好、安全、领域和多模态专家标注。

56. 为什么大模型训练数据需要 dataset versioning？
    考察点：不可变快照、manifest、pipeline 版本、采样配置、复现、审计。
    回答框架：训练结果依赖具体数据快照、过滤规则、去重参数、采样权重和 tokenizer。没有版本管理，模型不可复现，问题不可追溯，删除请求和污染排查也无法可靠执行。

57. 一个训练数据版本应该包含哪些内容？
    考察点：数据快照、manifest、shard hash、schema、pipeline、mixture、审计记录。
    回答框架：不能只记录数据目录；要记录原始/清洗数据快照、source registry、manifest、shard checksum、处理代码和配置版本、schema、过滤器、采样权重、风险统计和审计结果。

58. data lineage 在大模型数据治理里解决什么问题？
    考察点：source id、raw hash、processing steps、dataset version、training run、删除追溯。
    回答框架：lineage 让样本能追溯到来源和处理路径，也能从来源追踪到哪些版本和模型使用过。它支撑污染排查、删除请求、模型行为归因和版本回滚。

59. 删除请求为什么不是简单删除当前目录？
    考察点：旧版本、缓存、shard、索引、embedding、训练日志、模型处理。
    回答框架：数据可能存在于 raw、clean、shard、缓存、索引、embedding、旧版本和已训练模型里；需要 source id、hash、lineage 和版本清单定位影响范围，并记录执行证明和后续训练策略。

60. Datasheet、dataset card 和 model card 有什么关系？
    考察点：数据说明、模型说明、用途限制、风险、训练数据事实基础。
    回答框架：datasheet / dataset card 说明数据集的来源、组成、处理、用途、限制和风险；model card 说明模型用途、评估和风险。model card 的训练数据与风险描述需要数据版本和治理记录支撑。

61. 如何设计训练数据权限控制？
    考察点：数据敏感度分级、最小权限、访问日志、评测集保护、审批、审计。
    回答框架：按公开、授权、内部、用户、敏感、高风险和评测集分级；最小权限访问，高风险审批，记录访问日志，限制导出，定期审计，并重点保护评测集避免训练污染。

62. 数据 schema 演进为什么必须版本化？
    考察点：字段默认值、回填、标签体系、quality score 版本、sampler 依赖。
    回答框架：字段新增或含义变化会影响过滤、采样和评估分析。比如 quality score v1/v2 模型不同，不能混用；schema 版本要记录默认值、回填策略和下游兼容性。

63. 数据治理有哪些可量化指标？
    考察点：lineage 覆盖率、license 完整率、删除请求时延、权限审计、文档完整率、复现率。
    回答框架：可以看数据版本可复现率、lineage 覆盖率、license 记录完整率、PII/污染扫描覆盖率、删除请求处理时延、权限审计通过率、datasheet 完成度和高风险数据隔离率。

64. 数据工程面试中如何把回答从“流程描述”升级成“可量化证据”？
    考察点：目标澄清、指标、实验、治理门禁、trade-off、复盘。
    回答框架：先用目标、来源、处理、配比、评估、治理搭框架，再补 token 保留率、风险命中率、污染命中率、effective epoch、lineage 覆盖率等指标；最后说明小模型 ablation、人工抽样审计、版本 manifest 和安全治理边界。

65. 遇到没准备过的大模型数据题，如何组织 3 分钟回答？
    考察点：通用模板、目标相关性、数据分池、门禁、实验闭环、边界。
    回答框架：先问模型目标和评估指标，再把数据按来源、语言、领域、模态、质量和风险分池；说明清洗、PII/安全、去重、污染检测、mixture、ablation 和版本治理，最后给出可能取舍和残余风险。

66. 如何复盘一次数据工程 mock interview？
    考察点：answer coverage、metric coverage、evidence、risk coverage、错误表述、超时。
    回答框架：把每题拆成框架覆盖、指标覆盖、工程证据、风险边界、trade-off 深度和扣分项；低分题要记录缺少哪些框架字段、是否有错误说法、是否缺评估证据，并形成下一轮修正计划。

67. 如何排查一次模型效果异常背后的数据事故？
    考察点：exact / near duplicate、benchmark contamination、quality filter、secret / license / PII、mixture shift、lineage、版本回滚。
    回答框架：先确认异常切片和最近数据版本 diff，再抽样看原始样本；随后量化重复率、近重复、评估污染、风险命中、合成数据比例、清洗后实际 mixture 和 lineage 覆盖率。若门禁失败，要隔离问题桶、回滚或重建数据版本，并用干净评估集和小规模 ablation 复验修复。

## 多模态

1. CLIP 如何训练？
   考察点：双塔 encoder、归一化 embedding、相似度矩阵、temperature、batch 内负样本、双向 InfoNCE。
   回答框架：图像 encoder 和文本 encoder 分别输出 embedding，归一化后计算 `N x N` 图文相似度矩阵；对角线是真实配对，非对角线是 batch 内负样本；训练目标让正确图文对相似度高、错误配对相似度低，从而支持检索和 zero-shot 分类。

2. VLM 如何把图像接入 LLM？
   考察点：vision encoder、visual tokens、projector / connector、LLM hidden size、chat template、assistant-only loss。
   回答框架：先用 ViT 或 CLIP vision encoder 提取视觉 token，再用 connector 把 `[B,N,d_v]` 映射到 LLM hidden size `[B,N,d_l]`，然后与文本 token embedding 拼接或插入，通过多模态指令微调让模型学会基于视觉证据回答。

3. Diffusion Model 的基本原理是什么？
   考察点：前向加噪、反向去噪、噪声预测、条件控制、latent diffusion。
   回答框架：训练时把真实样本按噪声日程加噪，让模型在给定时间步和条件下预测噪声或恢复干净样本；生成时从噪声开始逐步去噪。文生图会把文本作为条件，Stable Diffusion 类方法把扩散过程放到 latent space 降低计算成本。

4. 如何估算一张图片进入 VLM 后会占用多少视觉 token？
   考察点：分辨率、patch size、CLS / global token、token compression、上下文预算。
   回答框架：先用 `ceil(H / P) * ceil(W / P)` 估算 patch token，再加上 CLS、全局 token 或 resampler 输出 token；如果后面直接拼进 LLM，还要把这些视觉 token 加到文本 token 里一起算上下文和 attention 成本。

5. 为什么视频和长音频会显著增加多模态成本？
   考察点：帧采样、视频 token、音频帧移、时序长度、延迟。
   回答框架：视频 token 近似等于采样帧数乘以单帧 patch token，音频通常按固定 frame shift 产生长序列；这些 token 如果进入 LLM 上下文，会增加显存、延迟和 attention 成本，所以必须做采样、压缩、分段或粗到细处理。

6. 如何设计一个多模态上线前成本与安全审计？
   考察点：context budget、p95 latency、hallucination rate、OCR、prompt injection、golden set。
   回答框架：先统计文本、图像、视频、音频 token 和 connector shape，再设定上下文、显存、延迟、幻觉率、OCR 准确率、不可信媒体指令触发率等门禁；用场景 golden set 回归，只有质量、安全和成本同时过线才上线。

7. CLIP 的 `logit_scale` / temperature 起什么作用？
   考察点：L2 normalize、cosine similarity、softmax 尖锐程度、过大/过小的影响。
   回答框架：CLIP 先把图像和文本 embedding 归一化，再用 `logit_scale` 乘以点积得到相似度 logits；它等价于 `1 / tau`，控制 softmax 分布尖锐程度。温度太高区分弱，温度太低容易过度自信，工程实现通常把 logit scale 作为参数学习或设定约束。

8. 如何手写 CLIP 双向 loss？
   考察点：`N x N` 相似度矩阵、对角线 label、image-to-text、text-to-image、cross entropy。
   回答框架：对一个 batch 的 `N` 对图文，先归一化图像向量和文本向量，计算 `S = image @ text.T / tau`；image-to-text 对每一行做 cross entropy，label 是行号；text-to-image 对每一列做 cross entropy，也就是对 `S.T` 做同样计算；最后两个方向平均。

9. 如何评估 CLIP 图文检索项目？
   考察点：Recall@K、MRR、Top-1、正负样本、bad case、数据噪声。
   回答框架：先固定图文配对评测集，离线编码 image/text embedding，按相似度排序；报告 text-to-image 和 image-to-text 两个方向的 Recall@1/5/10、MRR、Top-1，并按 OCR、细粒度属性、空间关系、中文 prompt、错误 caption 和近重复图分类分析 bad case。

10. CLIP 的 zero-shot 分类为什么会受 prompt 影响？
    考察点：类别 prompt、text embedding、prompt ensemble、开放词表、类别粒度。
    回答框架：CLIP 的类别向量来自文本 prompt，不同模板会产生不同 text embedding，因此 `dog`、`a photo of a dog`、`a blurry photo of a dog` 的相似度可能不同；常见改进是为每个类别构造多个模板并平均 embedding，但仍会受训练数据、类别描述和语言分布影响。

11. Vision encoder 在 VLM 中到底决定了什么？
    考察点：视觉可见性、patch tokens、全局 embedding、OCR / grounding、projector 前置条件。
    回答框架：vision encoder 决定模型能从图像里提取哪些视觉证据，输出可以是全局 image embedding 或 patch-level visual tokens；如果 vision encoder 分辨率低、patch 太大或预训练目标不覆盖 OCR / 图表，后面的 LLM 和指令微调也很难凭空恢复细节。

12. CNN 和 ViT 作为视觉编码器的核心区别是什么？
    考察点：卷积局部归纳偏置、残差连接、patch token、self-attention、数据规模。
    回答框架：CNN 用局部卷积、权重共享和层级特征建模图像，ResNet 通过 residual learning 让深层 CNN 更易训练；ViT 把图像切成 patch token，用 Transformer 做全局建模，更容易和 LLM 架构统一，但通常更依赖大规模预训练和正确的分辨率 / patch size 设计。

13. 如何检查一个 ViT patch embedding 的 shape 是否正确？
    考察点：`[B,C,H,W]`、patch size、patch grid、`[B,N,d_v]`、CLS、position embedding。
    回答框架：先算 `N_h=ceil(H/P)`、`N_w=ceil(W/P)` 和 `N=N_h*N_w`；Conv2d patch embedding 的权重应类似 `[d_v,C,P,P]`，输出展平成 `[B,N,d_v]`；如果加 CLS，position embedding 长度通常是 `N+1`。shape 不一致往往说明预处理分辨率、padding/crop 或 checkpoint 配置不匹配。

14. 为什么 OCR / 图表题常常需要更高分辨率或切图？
    考察点：小字、patch size、token 压缩、位置 embedding、上下文成本。
    回答框架：OCR 和图表依赖细粒度局部证据，小字在 resize 和 patch embedding 后可能只占少数 patch，甚至被压缩掉；提升分辨率或切图能保留细节，但视觉 token 和 attention 成本会快速增加，所以要结合 token budget、延迟和任务风险做取舍。

15. LLaVA、Flamingo 和 BLIP-2 的 VLM 架构差异是什么？
    考察点：projector 拼接、gated cross-attention、Perceiver Resampler、Q-Former、冻结策略。
    回答框架：LLaVA 路线通常用 vision encoder 提取 visual tokens，再用 projector 映射到 LLM hidden size 并拼进主序列；Flamingo 路线用 resampler 压缩视觉特征，并在 LLM 层中插入视觉 cross-attention；BLIP-2 用 Q-Former 的 learnable queries 查询冻结 vision encoder，再把少量 query 输出接到冻结或部分冻结的 LLM。三者核心 trade-off 是实现简单性、视觉 token 成本、融合深度和训练复杂度。

16. 为什么 VLM 的 `<image>` 占位符数量必须和图片数量一致？
    考察点：chat template、visual token 插入位置、多图引用、batch collator、label mask。
    回答框架：`<image>` 是文本模板和视觉 token 插入位置之间的锚点。占位符少于图片数会导致图片被忽略或插入位置不确定，占位符多于图片数会让模型以为空缺位置也有视觉证据，多图时还会造成引用混乱；插入视觉 token 后 labels 也要同步对齐，避免 loss mask 错位。

17. VLM 为什么常用 assistant-only loss mask？
    考察点：next-token loss、user/system mask、visual tokens、label shift、多模态 SFT。
    回答框架：VLM 的生成训练通常仍是 next-token prediction，但只希望模型学习 assistant 回答，而不是复述用户问题、system prompt 或 `<image>` 控制 token。做法是把 assistant answer 的 mask 设为 1，其他位置设为 0；有效 label 数应该等于回答 token 数，这能检查数据 collator 和 label shift 是否正确。

18. 多图 VLM 输入如何做 token budget 审计？
    考察点：图片数、visual tokens、resampler、context limit、attention 二次成本。
    回答框架：直接拼接时先算 `T_total = T_text + I * N_v + N_special`，再和上下文上限、显存和延迟预算比较；如果超预算，可以降低分辨率、减少图片数、切换到 Q-Former / Perceiver Resampler、做区域裁剪或分阶段粗到细推理。审计要同时看是否牺牲 OCR、计数、图表等细粒度证据。

19. Projector、Q-Former、Perceiver Resampler 和 cross-attention adapter 怎么选？
    考察点：维度对齐、token 压缩、细粒度信息、LLM 改造、训练数据规模。
    回答框架：线性或 MLP projector 简单便宜，适合低成本 instruction tuning，但不压缩 token；Q-Former 用 learnable queries 提取语言相关视觉信息，适合冻结强 vision encoder 和 LLM；Perceiver Resampler 更像通用 latent 压缩器，适合多图、多帧和可变长度视觉输入；cross-attention adapter 融合更深，但改造和训练成本更高。

20. 多模态 instruction tuning 数据训练前要审计什么？
    考察点：样本 schema、placeholder、assistant-only labels、任务覆盖、证据支持、拒答边界。
    回答框架：先检查图片数量和 `<image>` 占位符数量是否一致，再检查 visual tokens 加文本后是否超上下文、labels 是否只覆盖 assistant 回答；然后按 caption、VQA、OCR、chart、grounding、多轮和 refusal 统计 task coverage，抽检答案是否被图片证据支持，资料不足样本是否正确拒答，最后把不合格样本拒绝进入训练。

21. Chat template 在多模态 SFT 中为什么重要？
    考察点：role boundary、image placeholder、assistant 起始标记、训练推理一致性、label shift。
    回答框架：chat template 决定 system/user/assistant、image placeholder 和分隔符如何拼成模型输入。训练和推理模板不一致会让模型找不到图片位置或角色边界，labels 也可能错位。多模态里 `<image>` 不是普通文本，它是视觉 token 插入和 label mask 对齐的锚点。

22. 如何设计多模态 SFT 的任务覆盖？
    考察点：caption、VQA、OCR、chart、grounding、多轮、refusal、mixture。
    回答框架：先定义目标场景 required task set，再统计训练集中每类任务的样本数、token 数、质量分和风险分。caption 数据只能训练粗粒度描述，不能替代 OCR、图表、计数、空间关系、grounding 和拒答样本；覆盖率不足时，训练 loss 低也不代表业务能力足够。

23. 多模态 SFT 中如何控制视觉幻觉？
    考察点：evidence support、资料不足拒答、grounding、claim-level 检查、bad case。
    回答框架：数据侧要保证回答被视觉证据支持，加入 grounding、OCR、计数、图表和资料不足拒答样本；训练侧保证 assistant-only loss mask 正确；评估侧按 claim 检查图片证据，统计 unsupported answer rate、missing refusal rate 和 hallucination bad case，而不是只看回答是否流畅。

24. 多模态 SFT 为什么要保留拒答和资料不足样本？
    考察点：blurry image、遮挡、小字、不可见信息、高风险场景、over-refusal。
    回答框架：VLM 很容易在看不清、证据不足或用户问图片中不存在的信息时硬答。拒答样本教模型表达不确定性，减少视觉幻觉；但拒答不能过强，否则会 over-refusal，所以要同时评估该答时能答、该拒时能拒。

25. DDPM 的 forward diffusion 公式怎么写？
    考察点：`beta_t`、`alpha_t`、`alpha_bar_t`、闭式加噪、噪声强度。
    回答框架：先定义 `alpha_t=1-beta_t` 和累计信号保留 `alpha_bar_t=prod alpha_s`；然后 `x_t=sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) epsilon`。这说明任意时间步的 noisy sample 都能由干净样本和标准高斯噪声直接构造，训练时不必逐步加噪。

26. Diffusion 为什么常训练模型预测噪声？
    考察点：监督信号、MSE、`x0_hat`、score 关系。
    回答框架：训练时加入的噪声 `epsilon` 是已知的，所以可以用 MSE 训练 `epsilon_theta(x_t,t,c)` 预测噪声。预测噪声后能估计干净样本 `x0`，也能构造 DDPM 反向均值；从 score-based 视角看，噪声预测和 score 只是参数化方式不同。

27. Noise scheduler 对 diffusion 有什么影响？
    考察点：训练加噪、采样路径、步数、质量速度 trade-off。
    回答框架：scheduler 定义每个时间步的噪声强度、如何从 `x0` 得到 `x_t`，以及推理时如何从当前 noisy sample 更新到下一步。采样步数、DDPM / DDIM / Euler / DPM-Solver 等选择都会影响速度、质量和稳定性。

28. Classifier-Free Guidance 的公式和风险是什么？
    考察点：conditional prediction、unconditional prediction、guidance scale、prompt 遵循、多样性。
    回答框架：CFG 计算 `eps_uncond + w * (eps_cond - eps_uncond)`，放大条件方向，让结果更符合 prompt。`w` 太小可能不跟 prompt，太大可能过饱和、失真、细节异常和多样性下降，所以它是条件遵循和质量之间的折中旋钮。

29. Latent Diffusion 为什么比像素空间 diffusion 更高效？
    考察点：VAE / autoencoder、latent space、空间压缩、计算成本。
    回答框架：像素空间维度高，直接去噪成本大。Latent diffusion 先把图像压缩到低分辨率 latent，在 latent 中完成多步去噪，最后再解码回图像；如果空间压缩倍数是 `f`，spatial cell 数大约降到 `1/f^2`，因此更适合高分辨率生成。

30. Stable Diffusion 的 pipeline 如何拆解？
    考察点：VAE、text encoder、U-Net / DiT、scheduler、latent denoising。
    回答框架：prompt 先经过 tokenizer 和 text encoder 得到条件表示；随机 latent 在 scheduler 控制下由 U-Net 或 DiT 多步去噪；最后 VAE decoder 把 clean latent 解码成图片。VAE 负责降低空间成本，text encoder 负责语义条件，denoiser 负责生成主干，scheduler 决定采样路径。

31. Negative prompt 在 CFG 里到底怎么起作用？
    考察点：正条件、负条件、guidance scale、软约束、风险。
    回答框架：negative prompt 通常作为负条件参与 `eps_neg + w * (eps_pos - eps_neg)` 的方向计算，让模型远离负条件语义。它不是硬约束，只是改变去噪方向；`w` 太小 prompt 遵循弱，太大可能过饱和、失真和多样性下降。

32. ControlNet 和普通 prompt 控制有什么区别？
    考察点：结构条件、边缘 / 深度 / 姿态 / 分割、残差注入、可控性。
    回答框架：prompt 主要描述语义，ControlNet 额外输入结构条件，并通过条件分支向 diffusion backbone 注入残差，让模型按给定边缘、深度、姿态或分割布局生成。它解决的是纯文本很难精确控制空间结构的问题。

33. DALL-E 早期自回归图像 token 路线和 diffusion 路线怎么比较？
    考察点：离散图像 token、next-token prediction、序列长度、连续去噪、编辑生态。
    回答框架：早期 DALL-E 把图像离散成 tokens，再用 Transformer 根据文本和前序图像 token 做 next-token prediction；diffusion 从噪声开始逐步去噪，天然适合连续图像生成和可控编辑。自回归路线目标统一但高分辨率 token 序列长，diffusion 质量和编辑生态成熟但采样通常要多步。

34. 文生图 pipeline 上线前应该审计哪些数字？
    考察点：latent shape、VAE 压缩比、prompt token、U-Net 调用次数、ControlNet shape、安全门禁。
    回答框架：先检查分辨率和 VAE scale 得到的 latent shape，再算 prompt / negative prompt token 是否超上限；根据采样步数和 CFG 判断 denoiser 调用次数；如果有 image-to-image、inpainting 或 ControlNet，要检查 mask / 条件图 shape 与图像对齐；最后把 seed、scheduler、安全过滤、版权和水印策略纳入回归。

35. 视频生成为什么比图像生成难？
    考察点：时间维度、帧间一致性、身份保持、运动连续、物理合理性。
    回答框架：图像主要看单帧质量和文本一致性，视频还要在时间上保持主体身份、纹理、背景、动作、镜头和物理关系连续。很多失败不是某一帧很差，而是闪烁、身份漂移、动作跳变、遮挡后物体变化或物理关系不可信。

36. Spatiotemporal patch 的 token 数怎么估算？
    考察点：`T,H,W`、temporal patch、spatial patch、token budget。
    回答框架：逐帧 patch token 约为 `T * ceil(H/P) * ceil(W/P)`；时空 patch 约为 `ceil(T/P_t) * ceil(H/P) * ceil(W/P)`。时间 patch 能降低 token 数并注入短期运动信息，但太大会损失快速运动、小目标和局部细节。

37. Video diffusion 和 image diffusion 的核心区别是什么？
    考察点：视频 latent、时间维度、时空 denoiser、CFG 调用成本。
    回答框架：两者都可以用加噪 / 去噪和噪声预测目标，但 video diffusion 的 `z_t` 多了时间维度，denoiser 要同时建模空间结构和时间变化。工程上还要考虑帧数、时空 token、采样步数、CFG 带来的 denoiser 调用次数和 cross-attention 成本。

38. 为什么说视频生成模型可能是 world model 的候选，但不能直接等同于 world model？
    考察点：状态转移、物体持久性、因果后果、可交互、可规划、可验证。
    回答框架：视频生成要求模型预测世界随时间变化，因此可能学到物体持久性、运动、遮挡和部分物理规律。但真正的 world model 需要能在状态和动作条件下预测后果，并支持规划、交互和任务评估。生成看起来合理的视频只是证据之一，不等于可控可靠。

39. 如何设计一个视频生成上线前审计？
    考察点：token / latent shape、时序一致性、FVD / 人评、安全、deepfake。
    回答框架：先算帧数、分辨率、spatiotemporal tokens、latent shape、采样步数和 denoiser 调用次数；再评估 flicker、identity drift、motion smoothness、object permanence、物理失败和 prompt 一致性；最后结合 FVD / 任务型指标 / 人工偏好，并加入 deepfake、名人肖像、虚假新闻、版权和 provenance 门禁。

40. ASR、TTS 和 speech-to-speech 的区别是什么？
    考察点：语音转文本、文本转语音、语音到语音、cascade、端到端。
    回答框架：ASR 把语音转成文本，TTS 把文本合成语音，speech-to-speech 是语音输入并输出语音。工程上可以用 `ASR -> LLM -> TTS` 级联，也可以用 speech tokens 端到端建模；前者易调试，后者更自然但训练和安全评估更难。

41. Whisper 的基本结构和输入输出是什么？
    考察点：log-mel spectrogram、audio encoder、text decoder、自回归、弱监督多语言数据。
    回答框架：Whisper 先把音频转换成 log-mel spectrogram，再由 audio encoder 编码声学特征，text decoder 自回归生成 transcript token。它的优势来自大规模多语言弱监督数据，适合 ASR、翻译和一定噪声鲁棒场景。

42. WER / CER 怎么计算？为什么不能只看一个平均值？
    考察点：substitution、deletion、insertion、词级 / 字符级、切片分析。
    回答框架：WER 是 `(S + D + I) / N_ref`，CER 是字符级同类公式。平均 WER 可能掩盖口音、噪声、专有名词、多说话人、短命令和低资源语言问题，所以要按场景分桶看 bad case。

43. Audio codec 和 codec language model 为什么重要？
    考察点：连续音频离散化、codebooks、token rate、next-token prediction、TTS / speech-to-speech。
    回答框架：audio codec 把 waveform 压缩成离散 audio tokens，并能解码回语音；codec LM 把这些 tokens 当成语言模型序列生成。它让 TTS 和 speech-to-speech 能复用 Transformer / next-token 框架，但也带来序列长、实时推理慢、码本建模复杂和音质依赖 codec 的问题。

44. 语音助手上线前应该审计哪些指标？
    考察点：采样率、log-mel shape、WER / CER、latency、codec tokens、授权、安全。
    回答框架：先检查采样率、切片、VAD、ASR WER / CER 和噪声场景；再算 codec token rate、上下文长度、ASR / LLM / TTS 级联延迟和首包延迟；如果支持 voice cloning，要检查说话人授权、水印、深伪检测、隐私脱敏和高风险请求拒绝。

45. 什么是统一多模态模型？统一可以发生在哪些层次？
    考察点：接口统一、表示统一、架构 / 目标统一、工程系统。
    回答框架：统一多模态模型或系统希望处理文本、图像、音频、视频等多种输入输出。接口统一是多个模块对用户呈现一个助手；表示统一是映射到共享 embedding 空间；架构 / 目标统一是把多模态 token 放进统一 Transformer 或统一训练目标。真实产品常是三者混合。

46. Any-to-Any 多模态为什么难？
    考察点：输入输出模态、decoder 差异、数据配比、延迟、评估。
    回答框架：Any-to-Any 要支持任意输入模态到任意输出模态，例如 text -> image、speech -> speech、video -> text。难点在于各模态 token rate、训练数据规模、输出 decoder、质量指标和安全风险都不同，很难用单一目标和单一评估覆盖。

47. 统一 token 化为什么是机会也是瓶颈？
    考察点：文本 token、image patch / image token、audio codec token、video spatiotemporal token、context budget。
    回答框架：统一 token 化让 Transformer 可以用类似 next-token prediction 的方式处理多模态；但图像、音频、视频 token 数远多于文本，多图、长音频、多帧视频会快速推高上下文、attention 成本和延迟。tokenizer / codec 质量也会限制细节、音质和运动信息。

48. 中心 LLM 模块化系统和 early-fusion 统一模型怎么取舍？
    考察点：LLM router、专用 encoder / decoder、信息瓶颈、端到端优化、安全。
    回答框架：中心 LLM 系统复用 ASR、OCR、vision encoder、TTS、image generator 和工具，易调试、易做权限和安全控制，但可能有误差传播和信息压缩。Early-fusion 统一模型跨模态交互更自然，但训练成本高、可解释性弱、token 序列长、上线安全评估更难。

49. 统一多模态上线前如何做审计？
    考察点：各模态 token、路由、loss mixture、modality conflict、安全门禁。
    回答框架：先算文本、图像、音频、视频和 special token 总预算，再算输出文本 / audio / image / tool token；检查 OCR、ASR、video sampler、TTS、image generator 和工具路由是否匹配任务；训练侧看 loss mixture 是否覆盖目标能力；安全侧隔离不可信媒体指令、未授权 voice cloning、未审核生成输出和工具越权。

50. 如何系统评估一个多模态模型？
    考察点：VQA、OCR、chart、grounding、hallucination、generation、安全、回归。
    回答框架：先按任务拆成 VQA / OCR / 图表 / grounding / 视频 / 语音 / 生成和安全切片；再分别看 VQA accuracy、WER / CER、chart relaxed accuracy、IoU、faithfulness、人工偏好和安全门禁；最后做 bad case attribution，区分感知、OCR、grounding、推理、表达和拒答错误，不能只引用一个公开榜单均分。

51. 多模态 hallucination rate 应该怎么定义？
    考察点：atomic claim、media evidence、unsupported claim、拒答边界。
    回答框架：把模型回答拆成 atomic claims，逐条检查是否被图片、音频、视频或文档证据支持；unsupported claims 占比就是多模态幻觉率。它和普通 accuracy 不同，因为模型可能最终答案看似接近，但中间解释包含不存在物体、读错文字或编造不可见信息。

52. Grounding 在多模态安全和评估中有什么作用？
    考察点：bbox / mask / point、IoU、证据定位、GUI agent、幻觉治理。
    回答框架：Grounding 把语言 claim 对应到图片区域、框、点或 mask，可以用 IoU、pointing accuracy 或 evidence support rate 评估。它能帮助发现模型是否真的看见证据，也能约束 GUI agent 点击区域、文档审核和高风险视觉判断。

53. 多模态 prompt injection 和普通 prompt injection 有什么不同？
    考察点：OCR、截图、文档、音频、视频、不可信内容边界、工具权限。
    回答框架：普通 prompt injection 多来自用户文本或外部文档文本，多模态 prompt injection 可以隐藏在图片、网页截图、PDF、音频或视频中，再经 OCR / ASR / caption 被模型读入。防护要标记媒体内容来源，外部内容只能作为数据，不能覆盖系统指令；工具调用还要做权限、参数、二次确认和 audit trace。

54. 多模态安全上线门禁应该看哪些指标？
    考察点：clean success、attack success、hallucination、privacy、identity、over-refusal、latency。
    回答框架：门禁要同时看正常任务成功率、幻觉率、不可信媒体指令触发率、隐私泄露率、身份识别 / voice cloning 高风险拦截、生成内容审核通过率、过度拒答率和 P95 延迟。高质量安全不是拒答率最高，而是在正常任务可用的前提下正确拦截高风险样本。

54A. 多模态项目答案错了如何做事故归因？
    考察点：input fidelity、OCR、ASR、video sampling、grounding、safety、budget。
    回答框架：先保存原始媒体和模型实际看到的输入，检查 resize、crop、压缩、抽帧和降噪是否保留关键证据；再看 OCR / ASR / caption / metadata 中间结果，尤其是金额、日期、错误码、姓名和时间戳等关键字段；随后检查证据是否进入上下文、claim 是否被区域 / OCR / 时间戳支持，最后看安全过滤、关键字段确认、延迟和成本门禁。这样才能区分输入保真、模态转换、grounding、生成和安全问题。

54B. 如何设计一个多模态项目事故审计 demo？
    考察点：toy media cases、critical fields、temporal recall、evidence support、gate。
    回答框架：可以构造 6 类样本：正常商品图、长截图裁剪丢底部证据、发票金额 OCR 错、视频关键事件抽帧漏掉、噪声音频金额 ASR 错且未确认、不可信媒体内容安全边界失败。脚本输出 input fidelity、evidence recall、OCR / ASR accuracy、critical field accuracy、temporal evidence recall、grounding IoU、evidence support、hallucination rate、safety block、confirmation coverage、budget overrun、root cause 和 failed gates。

55. 如何准备一轮完整的多模态面试复盘？
    考察点：topic coverage、formula coverage、demo coverage、risk coverage、project story、revision plan。
    回答框架：先列出 CLIP、vision encoder、VLM connector、多模态 SFT、diffusion、文生图、视频、语音、统一多模态、评估安全和项目讲述这些 required topics；再检查每类是否能写公式、讲 shape、给 demo、说 bad case 和 trade-off；最后把低分题、缺失公式和缺失 demo 写成下一轮 revision plan，而不是只说“再复习一下”。

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

5A. Benchmark 总分提升但线上效果变差，如何排查评估指标事故？
   考察点：aggregate score trap、slice regression、clean eval lift、contamination、online feedback、cost latency。
   回答框架：先确认评估集版本、prompt、解码和线上配置是否一致；再按任务、语言、难度、安全、RAG、工具和真实用户场景切片看 `candidate - baseline`，检查公开 benchmark 污染和干净集提升；随后用 paired evaluation 和 bootstrap CI 看提升是否稳定，最后把线上任务完成率、用户反馈、成本、P95 延迟和安全失败样本纳入 evaluation gate。

5B. 如何校准 LLM-as-a-judge，避免 judge 偏差误导上线？
   考察点：judge-human agreement、length bias、position bias、rubric、pairwise eval、人审抽样。
   回答框架：先设计明确 rubric，并随机化答案顺序、隐藏模型名；再用人工标注或程序验证样本计算 judge-human agreement，按长度、位置、任务类型和模型来源切片分析错误；如果 judge 偏好长答案、格式漂亮但事实错的回答，就只能作为初筛，关键场景必须保留人工复核和硬验证器。

5C. 一份可信的模型评估发布报告应该包含哪些门禁？
   考察点：paired lift、bootstrap confidence interval、slice regression、contamination rate、judge calibration、cost、latency、safety、online feedback。
   回答框架：报告不能只给一个平均分。至少要记录评估集版本、样本 schema、模型 / prompt / 解码配置、总体与切片指标、污染检测、成对提升和置信区间、judge 校准、人审 bad case、成本延迟、安全误拒 / 漏拒、线上灰度反馈和 failed gates。任何关键门禁失败，都应先定位指标事故而不是发布。

6. 如何降低幻觉？
   考察点：数据质量、RAG、工具、verifier、grounding、不确定性表达。
   回答框架：不能只调 decoding，要结合训练、检索、工具、验证器、引用和专门评估。

7. Jailbreak 和 prompt injection 有什么区别？如何设计防护评估？
   考察点：用户提示、外部内容、instruction hierarchy、RAG、工具、数据外泄、未授权工具调用、误拒。
   回答框架：jailbreak 主要是当前用户直接诱导模型绕过安全策略，prompt injection 是不可信内容污染应用指令边界，尤其出现在 RAG、网页、邮件、工具返回和 Agent 场景。防御要做指令层级、不可信内容标记、权限隔离、工具参数校验、二次确认和审计；评估不能只看 attack success，还要看 data leakage rate、unauthorized tool call rate、attack task success、clean task success、over-refusal 和 prompt injection gate。

8. 如何判断模型是否真的具备推理能力？
   考察点：泛化、扰动、组合测试、CoT、verifier、污染。
   回答框架：不能只看 benchmark，要看分布外、扰动、组合泛化和中间过程可验证性。

9. 如何设计下一代多模态模型？
   考察点：跨模态对齐、统一表示、数据、评估、工具、安全。
   回答框架：从架构、数据、训练、评估和安全五个维度设计，而不是只拼接图像 token。

10. Test-time compute 为什么重要？
   考察点：self-consistency、verifier、search、动态预算、成本。
   回答框架：推理时增加搜索和验证可以提升复杂任务表现，但要权衡延迟和成本。

11. 如何设计一个 reasoning 系统的上线前审计？
   考察点：greedy、self-consistency、verifier、pass@k、process step、成本、安全。
   回答框架：先在固定 reasoning eval set 上比较 greedy accuracy、self-consistency accuracy、verifier reranking accuracy 和 pass@k；再统计过程步骤准确率、候选数、token 成本、P95 延迟、工具执行失败和污染风险；最后用门禁约束正确率、过程质量、预算和安全风险，不能只看回答变长或单一 benchmark 分数。

12. CoT 为什么不能在所有请求上无脑开启？
   考察点：CoT regression、step accuracy、simple task、visible explanation、成本、安全边界。
   回答框架：CoT 能给复杂题更多中间计算空间，但简单事实题可能被过度推理带偏，安全场景也不适合展示完整内部推理。上线前要比较 direct / CoT / routed CoT，单独统计回归样本、unsupported step、token 成本和可见解释策略。

13. Self-consistency、pass@k 和 verifier reranking 有什么区别？
   考察点：多路径采样、答案标准化、majority vote、pass@k、weighted vote、成本。
   回答框架：self-consistency 看多个候选经过投票后的最终准确率；pass@k 看候选集合里是否至少存在一个正确解；verifier reranking 看选择器能否把正确候选挑出来。三者都依赖答案标准化和候选质量，但评估对象不同，不能只汇报一个指标。

14. 如何评估一个 verifier 或 reward model 是否真的有用？
   考察点：rerank accuracy、pairwise accuracy、hard negatives、calibration、programmatic verifier、下游提升。
   回答框架：先看 verifier 在真实候选分布上的 top-1 selection accuracy 和 pairwise accuracy，再看 hard negative accuracy、ECE / Brier 等校准指标和切片偏差；最后必须比较加入 verifier 后的下游 reasoning accuracy、成本和失败样本。代码和数学任务应优先接入 programmatic verifier 或 hybrid verifier。

15. 如何设计 process supervision / PRM 的上线前审计？
   考察点：step label、first-error detection、auto label coverage、human label cost、PRM search、outcome blind spot、hard negative。
   回答框架：先定义步骤粒度和 step label，区分步骤正确性、相关性和错误类型；再统计 outcome accuracy、step accuracy、first-error accuracy、自动标注覆盖率和人工标注成本；随后把 PRM 接入搜索，检查 search top-1 accuracy、剪枝失败和 hard negative；最后单独列出最终答案正确但过程错误、过程正确但最终格式错误、冗余步骤和高分错误路径，用门禁约束过程质量、搜索效果、成本和人工复核策略。

16. Tree-of-Thought、beam search 和 MCTS 在 reasoning 系统里怎么评估？
   考察点：search state、action、beam size、UCT、candidate diversity、pruned correct path、budget、hard negative。
   回答框架：先把任务定义成状态、动作、转移、评分和停止条件；再比较 greedy、beam search、best-first、ToT / MCTS 的最终准确率和 pass@k；然后统计候选多样性、节点数、token 成本、verifier / tool 调用次数、剪枝假阴性和 hard negative 失败；最后说明 beam 简单可控但容易早剪正确路径，MCTS 能平衡探索和利用但依赖 reward 质量，不能只报平均准确率。

17. 如何评估 test-time compute scaling 是否值得上线？
   考察点：compute budget、adaptive routing、accuracy-cost curve、cost per correct、P95 latency、wasted high compute、hard slices。
   回答框架：先定义 direct、self-consistency、verifier、search 等预算档位，再在同一评测集上画准确率、总成本、单位正确样本成本和 P95 延迟曲线；然后设计 budget router，根据任务难度、价值和可验证性做 adaptive compute；最后检查边际收益、低价值请求预算浪费、高预算仍失败样本、hard negative 和安全风险。上线门禁不能只看最高预算准确率，要证明 adaptive 策略在质量、成本和延迟之间更优。

18. 如何设计数学推理训练数据的上线前审计？
   考察点：math sample、answer supervision、step label、first-error detection、synthetic data、contamination、template diversity、hard slice。
   回答框架：先定义数学样本 schema，包含题目、标准答案、解题步骤、步骤标签、题型、难度、来源、模板和污染标记；再分别统计答案准确率、步骤准确率、第一处错误定位、自动验证覆盖率、自动标注覆盖率和人审成本；随后检查合成题模板多样性、题型 / 难度覆盖、评测污染、verifier 误判和 hard proof / geometry 等难切片；最后用 math training gate 约束平均分、过程质量、污染率、模板多样性和 hard-slice 表现，不能只看最终答案或单一数学 benchmark。

19. 如何设计代码推理与执行反馈系统的上线前审计？
   考察点：functional correctness、public / hidden tests、pass@k、self-debug、sandbox、public-hidden gap、repair success。
   回答框架：先定义代码任务 schema，包括需求、函数签名、公开测试、隐藏测试、候选代码、执行环境和预算；再比较 greedy、public-test rerank、pass@k 和 self-debug 修复后的隐藏测试准确率；随后统计 public-hidden gap、首次公开测试通过轮数、修复成功率、超时 / 内存 / 静态检查失败和沙箱拦截；最后用 code reasoning gate 约束隐藏泛化、公开测试过拟合、沙箱安全和执行成本，不能只看公开样例或代码文本质量。

20. 一份可信的 reasoning 评估报告应该包含什么？
   考察点：benchmark version、contamination、variant、process eval、paired eval、bootstrap CI、TTC cost、slice。
   回答框架：先记录评测集版本、样本 schema、prompt、解码参数、候选数、verifier / tool 使用和随机种子；再报告最终答案准确率、切片准确率、过程步骤准确率、第一处错误定位、original / variant 鲁棒性、污染率和失败样本；然后用 paired evaluation 比较 baseline 与 candidate，并给出 bootstrap confidence interval 或其他统计不确定性；最后报告 token、候选数、verifier 调用、工具调用、延迟、cost per correct 和上线门禁。只报一个 benchmark 平均分不够。

21. 如何设计 reasoning 安全与局限的上线前审计？
   考察点：伪推理、过度自信、hidden CoT、工具误用、高风险人审、过度拒答、严重度加权风险。
   回答框架：先定义 reasoning safety 样本 schema，记录风险域、期望安全动作、推理过程是否支持答案、置信度、工具动作、权限判定、人工审核和严重度；再统计 pseudo reasoning rate、overconfident error rate、unsafe compliance rate、tool misuse rate、hidden CoT exposure rate、high-risk review coverage、over-refusal rate 和 severity-weighted risk；最后设置 safety gate。高风险不当服从、工具越权、隐藏 CoT 暴露或人审覆盖不过线时，即使 reasoning benchmark 提升，也不能直接上线。

## 产品化、商业化与落地

1. 大模型从 demo 到产品差在哪里？
   考察点：真实用户、长尾输入、稳定性、成本、安全、合规、监控、反馈闭环。
   回答框架：demo 证明技术能跑通，产品要长期稳定创造用户价值。产品化要面对真实输入、并发、失败恢复、权限、成本、延迟、安全合规、灰度发布、监控和反馈闭环，不能只看演示样例。

2. 如何判断一个大模型场景值不值得做？
   考察点：痛点、频次、价值、大模型适配度、成功指标、风险、成本。
   回答框架：先看用户痛点是否明确且高频或高价值，再看大模型是否比规则、搜索或传统模型更适合；随后定义任务成功、采用率、节省时间、成本下降等指标，并评估延迟、隐私、幻觉、人审和单位经济账。

3. 如何把技术指标连接到产品指标？
   考察点：技术指标、任务指标、用户指标、业务指标。
   回答框架：要建立 `技术指标 -> 任务指标 -> 用户行为 -> 业务结果` 的链路。例如 RAG 召回率提升应继续证明答案正确率、引用准确率、自助解决率和客服成本下降，而不是只停留在 recall@k。

4. 大模型产品的 ROI 怎么估算？
   考察点：收益、成本、采用率、成功率提升、固定成本、风险成本。
   回答框架：收益可以粗略写成任务频次乘采用率乘成功率提升乘单次成功价值；成本要包含模型、检索、工具、人审、运维、数据治理和风险成本。可以报告收益成本比 `benefit / cost`，也可以报告净 ROI `(benefit - cost) / cost`，但要说明口径。

5. 为什么大模型产品不能只算 token 成本？
   考察点：RAG、Agent、人工审核、日志监控、运维、治理、风险。
   回答框架：token 只是模型调用成本的一部分。真实系统还有检索、rerank、工具调用、重试、人审、监控、日志存储、权限、合规、数据治理和事故风险。尤其 Agent 和企业 RAG 的总成本可能远高于单次生成价格。

6. 产品化上线门禁应该包含什么？
   考察点：quality gate、unit economics、latency、risk、workflow、eval、feedback。
   回答框架：上线门禁至少包含任务质量提升、单位经济账过线、P95 延迟可接受、安全隐私合规、工作流适配、可重复离线评估、灰度与线上监控、用户反馈和失败样本回流。任何一个关键门禁不过线，都不应直接全量上线。

7. 为什么上线不是大模型产品的终点？
   考察点：用户长尾、知识更新、模型版本、成本漂移、安全事件、反馈数据。
   回答框架：上线后真实用户会暴露离线评估没覆盖的长尾，知识库、模型、prompt 和业务流程也会变化。必须持续监控质量、成本、延迟、安全和反馈，把失败样本沉淀为回归集或优化数据。

8. 如何用一个项目证明自己懂产品化？
   考察点：用户、痛点、baseline、指标、ROI、风险、反馈、个人贡献。
   回答框架：项目表达要说明目标用户和痛点、baseline、技术方案、任务指标、产品指标、成本收益、风险控制、上线监控、失败样本和个人贡献。只讲模型结构或 demo 效果不够。

9. 如何判断一个大模型需求是真实需求还是伪需求？
   考察点：当前替代方案、频次、成本、指标、工作流、组织承诺。
   回答框架：真实需求通常已经有人在花时间、预算或流程解决，且有明确用户、任务频次、当前痛点、成功指标和工作流接入意愿。伪需求通常只是“想做 AI”、没有当前成本、没有可衡量指标，或者规则和搜索已经足够。

10. 如何选择第一个大模型试点场景？
    考察点：用户画像、任务链路、数据、验证、风险、短周期价值。
    回答框架：先画出用户和利益相关方，再拆完整任务链路，找高痛点、高频或高价值、数据可得、结果可验证、错误可控、能接入现有流程的环节。高风险大场景不直接全自动，先拆成 assist、review 或 approval 试点。

11. 什么情况下不应该用大模型？
    考察点：规则、搜索、模板、传统模型、成本、不确定性。
    回答框架：如果规则稳定、搜索可以直接解决、输出格式完全确定、传统分类模型足够，或者错误成本高但缺少验证和人审闭环，就不应该为了使用 LLM 而使用 LLM。面试中要说明 LLM fit，而不是默认所有智能需求都上大模型。

12. 如何量化大模型场景优先级？
    考察点：痛点、频次、单次价值、LLM fit、数据准备、可验证性、风险、工作流、难度。
    回答框架：可以先估算月收益 `monthly_frequency * (time_saved_hours * hourly_cost + value_per_task * quality_uplift)`，再把用户痛点、LLM fit、数据准备度、可验证性、工作流适配、自动化层级、落地难度和剩余风险合成优先级分。分数不是绝对真理，作用是让隐含判断可讨论、可复盘。

13. 如何把模型能力转成产品体验？
    考察点：能力映射、任务成功、延迟、引用、格式、可控、恢复、信任校准。
    回答框架：先把模型能力拆成理解、生成、检索、推理、工具调用、结构化输出和安全拒答，再映射到用户可感知指标：任务完成率、P95 延迟、引用支持率、结构化输出稳定率、用户可控性、错误恢复率、采纳率和过度拒答率。最后设置 experience gate，而不是只看 benchmark。

14. 为什么高分模型的产品体验可能不好？
    考察点：离线榜单、真实工作流、尾延迟、格式、引用、恢复、安全。
    回答框架：公开 benchmark 只覆盖部分能力和输入分布，产品体验还受真实用户输入、工作流接入、尾延迟、流式反馈、引用校验、格式稳定、拒答策略、成本和错误恢复影响。高分模型如果不能稳定输出可用结果，用户仍然不会采纳。

15. 大模型产品的体验门禁应该包含哪些指标？
    考察点：success、latency、citation、format、control、recovery、calibration、risk。
    回答框架：至少包含任务成功率、首次成功率、P95/P99 延迟、引用支持率、结构化输出稳定率、可控性满意度、错误恢复率、信任校准、过度拒答率、安全违规率和人工接管率。不同产品权重不同，但不能只看平均响应时间或单一准确率。

16. 如何设计大模型产品中的信任校准？
    考察点：不确定性、证据、事实/推断、高风险复核、过度自信。
    回答框架：对事实型回答展示来源和版本，对推断型回答标注假设和不确定点，对高风险结论要求复核或人审；同时用置信度桶检查预测置信度和实际正确率差距，避免模型自信编造，也避免过度保守导致可用性下降。

17. 结构化输出为什么是产品体验问题，不只是工程细节？
    考察点：工作流、解析、审批、批处理、稳定性。
    回答框架：很多企业场景的输出会进入表单、工单、审批流、报表或下游工具。文本看起来对但字段缺失、JSON 无法解析或格式漂移，会让工作流中断。结构化输出稳定率应该进体验和上线门禁。

18. 如何估算大模型项目的完整 ROI？
    考察点：收益拆解、全成本、单位经济账、回本周期、门禁。
    回答框架：先明确收益来自节省人工、质量提升、收入增长还是风险下降，再用任务量、采用率、单次节省时间、质量提升和单次价值估算月收益；成本要包含模型 token、RAG、工具、人审、重试、数据治理、研发摊销、运维监控、合规和预期风险。最后报告净收益、收益成本比、净 ROI、回本周期和 ROI gate。

19. 为什么不能只看 token 单价判断大模型产品成本？
    考察点：上下文、输出、重试、RAG、工具、人审、固定成本。
    回答框架：token 单价只是模型调用成本的一部分。真实成本还取决于 prompt 长度、RAG 拼接、输出长度、调用次数、缓存命中、重试率、Agent 步数、检索 / rerank、工具调用、人审、监控日志和固定研发运维。很多项目模型成本很低，但人审和固定成本把 ROI 打穿。

20. 如何判断 API 调用和自部署哪个更划算？
    考察点：调用量、利用率、团队能力、延迟、合规、固定成本。
    回答框架：小流量、快速试点和模型频繁变化时 API 通常更灵活；大流量、任务稳定、低延迟、强合规或深度定制时自部署可能更划算。判断时要把 GPU 租用或折旧、利用率、推理优化、人力运维、故障风险和模型升级成本都算进去。

21. ROI 敏感性分析要看哪些变量？
    考察点：采用率、任务量、质量提升、重试、人审、固定成本、价格变化。
    回答框架：至少看采用率下降、任务量不达预期、质量提升低于假设、输出变长、重试率升高、人审时间增加、固定成本超支和单价变化。ROI 是假设驱动的决策工具，敏感性分析能暴露哪个变量最容易让项目从盈利变亏损。

22. 什么情况下用户量增长反而会让大模型项目亏得更多？
    考察点：单次毛利、可变成本、固定成本摊销、负单位经济账。
    回答框架：如果单次任务收益低于可变成本，例如大模型多轮调用、RAG、工具和人审成本超过用户愿意支付或节省的价值，那么规模越大亏损越大。必须先让单次毛利转正，再谈固定成本摊销和规模化。

23. 如何降低大模型产品成本但不伤害质量？
    考察点：路由、缓存、上下文、批处理、RAG top-k、Agent budget、评估门禁。
    回答框架：可以用规则或小模型处理简单任务，大模型只处理高价值复杂任务；对高频请求做缓存和 prompt caching；控制 RAG top-k 和上下文长度；离线任务用批处理；限制 Agent 步数、重试和工具调用；同时用回归评估、体验门禁和安全门禁确认质量没有明显下降。

24. 企业级 LLM 应用和普通消费级应用最大的区别是什么？
    考察点：身份、权限、数据、工作流、审计、SLO、ROI。
    回答框架：企业级应用不是给用户一个聊天入口，而是把模型嵌入企业流程。它要接入 SSO / IAM / RBAC，保证 RAG、生成、工具调用和日志都不越权；要处理租户隔离、数据治理、审计、SLO、反馈和业务指标；还要证明 ROI 和风险门禁过线。普通应用可以更关注单次体验，企业应用必须可治理、可追溯、可运营。

25. 企业知识库 RAG 为什么必须做权限感知？
    考察点：检索过滤、上下文拼接、引用展示、日志、租户隔离。
    回答框架：企业知识库里的文档通常有部门、项目、客户和密级边界。如果只在前端隐藏无权文档，向量检索仍可能召回无权片段，模型也可能把其中信息总结出来。权限感知 RAG 要在检索、rerank、上下文拼接、引用展示和日志写入中都执行权限过滤，并记录审计 trace。

26. 如何设计企业 LLM 应用的工具权限门禁？
    考察点：用户身份、角色、动作、资源、风险等级、二次确认。
    回答框架：先把工具按读、写、外发、删除、权限变更和资金动作分级，再用用户身份、角色、租户、资源范围和动作类型做授权判断。高风险工具要加入 dry run、人工确认、人审、限流、回滚和审计日志。不能只靠 prompt 让模型“不要越权调用”。

27. 企业 LLM 应用上线前应该看哪些治理指标？
    考察点：permission、tenant、citation、audit、freshness、SLO、human review、business metric。
    回答框架：至少看权限通过率、租户隔离违规率、引用支持率、审计覆盖率、数据新鲜度、P95/P99 延迟 SLO 通过率、高风险人审覆盖率、PII 脱敏率、反馈闭环覆盖率和业务主指标是否定义。企业门禁是硬约束集合，不是一个平均分好看就能上线。

28. 企业 LLM 项目为什么经常试点成功但规模化失败？
    考察点：数据治理、权限复杂、组织协作、工作流、运营、成本。
    回答框架：试点范围小、样本可控、人工兜底多，容易看起来成功；规模化后会暴露权限边界、文档过期、多租户隔离、系统集成、长尾输入、审计、SLO、用户培训和成本问题。要从试点阶段就建设评估集、权限模型、反馈闭环和工作流接入，而不是等全量上线后再补治理。

29. 如何判断企业 LLM 应用应该做人机协同还是全自动？
    考察点：风险等级、可验证性、权限、收益、人审成本、责任边界。
    回答框架：低风险、高频、可验证、错误可恢复的任务可以逐步提高自动化比例；金融、医疗、法律、合规、退款、权限变更等高风险任务应优先做 copilot、review 或 approval 形态。判断时要同时看质量、权限、审计、人审覆盖、SLO、单位经济账和责任归属。

30. RAG demo 和 RAG 产品差在哪里？
    考察点：文档治理、权限、版本、引用、评估、反馈、SLO。
    回答框架：RAG demo 通常只验证几个文档的切块、向量检索和生成；RAG 产品要面对多来源文档、版本冲突、权限过滤、增量更新、引用可信、拒答、评估集、反馈闭环、成本、延迟和知识运营。真正难点不是“能检索”，而是能否基于正确、最新、有权限、可追溯的证据稳定回答。

31. 如何设计企业 RAG 产品的评估指标？
    考察点：retrieval、context、generation、citation、abstention、product。
    回答框架：检索层看 retrieval recall、hit@k、MRR、nDCG 和 rerank 命中；上下文层看 context precision / recall 和过期证据率；生成层看 answer correctness、evidence support、faithfulness 和 citation accuracy；系统层看权限过滤、拒答准确率、P95 延迟、单位成本；产品层看自助解决率、满意度、反馈闭环和业务指标。

32. RAG 引用可信应该怎么评估？
    考察点：atomic claims、证据支持、版本、权限、可打开原文。
    回答框架：不能只看答案末尾有没有链接，要把回答拆成 atomic claims，逐条检查引用是否存在、相关、足以支持 claim、版本未过期、当前用户有权限查看，并能打开原文。答案碰巧正确但引用不支持，在企业 RAG 中仍然是失败样本。

33. RAG 应该什么时候拒答或澄清？
    考察点：无证据、低召回、冲突、权限不足、过期、问题不清。
    回答框架：当检索不到足够证据、证据冲突、用户无权查看来源、文档过期、问题超出知识库范围或用户问题缺少关键信息时，RAG 应该拒答、澄清、提示已检索范围或转人工。拒答准确率和过度拒答率都要评估，不能靠一句 prompt 保证。

34. RAG 产品上线门禁应该包含什么？
    考察点：retrieval recall、context precision、evidence support、citation accuracy、permission、freshness、latency、cost、feedback。
    回答框架：上线门禁至少包含 retrieval recall、context precision、evidence support、citation accuracy、abstention accuracy、permission filter pass rate、stale evidence rate、P95 latency、unit cost、eval ready、feedback loop 和 business metric。任何关键门禁不过线，都应先拆试点、补文档治理、补权限或改成 copilot，而不是直接全量上线。

34A. RAG 答案错误时如何做事故归因？
    考察点：retrieval miss、context drop、citation、permission、freshness、root cause。
    回答框架：先找标准证据，再沿链路看正确证据是否入库、是否被 retriever 召回、是否被 reranker 保留、是否进入 final context、是否支持答案 claim、引用是否对应当前版本、用户是否有权限。如果召回有但上下文没有，是 context drop；如果上下文有但 claim 不被支持，是 grounding / citation 失败；如果证据无权或过期，则要归到权限或 freshness，而不是泛泛说模型幻觉。

34B. 如何设计一个 RAG 事故审计 demo？
    考察点：toy corpus、expected evidence、context、claims、ACL、staleness、gate。
    回答框架：可以构造小型 corpus 和 5 类 bad case：召回不到正确文档、召回后上下文丢证据、引用旧版本、无权文档进入上下文、证据不足却未拒答。脚本输出 retrieval recall、MRR、context recall / precision、citation accuracy、unsupported claim rate、permission leak rate、stale evidence rate、abstention accuracy、P95 延迟、单位成本、root cause 和 failed gates。

35. Agent demo 和 Agent 产品差在哪里？
    考察点：task boundary、tool permission、human confirmation、trace、budget、eval、business metric。
    回答框架：Agent demo 主要证明模型能规划和调用工具；Agent 产品要证明任务边界清楚、自动化层级合适、工具权限最小化、高风险动作有人审、失败可恢复、trace 可审计、成本和延迟可控，并且能用业务指标持续运营。产品化重点不是让 Agent 更自由，而是让它在受控范围内可靠完成可验证任务。

36. Agent 产品为什么不应该一开始全自动？
    考察点：automation level、risk、verification、rollback、trust。
    回答框架：自动化层级应从 suggest、draft、semi-auto、approval 到 auto 逐步提高。高风险、不可逆、验收器不足或责任边界不清的任务，应先做 copilot / review / approval；只有任务成功率、权限、人审、恢复、trace、预算和回归评估稳定后，才考虑自动执行。

37. Agent 产品评估应该看哪些指标？
    考察点：task success、tool execution、confirmation、recovery、state、observation、trace、unauthorized、budget。
    回答框架：至少看任务成功率、工具执行成功率、高风险确认覆盖率、失败恢复率、状态更新覆盖率、观察使用率、trace 完整率、未授权动作率、预算超限率、P95 延迟、单位成本、评估覆盖、反馈闭环和业务指标。最终回答正确但越权调用、忽略 observation 或成本失控，仍然不能上线。

38. 如何设计 Agent 产品的工具权限和人工确认？
    考察点：least privilege、permission checker、risk level、dry run、audit。
    回答框架：工具按只读、低风险写入、中风险业务操作和高风险外部副作用分层；controller 在模型外做用户权限、任务权限、参数校验和风险检查。删除、支付、发送、改权限、生产变更等动作必须展示对象、参数、原因、影响范围和取消方式，并记录审计日志。

39. 为什么生产系统常用 workflow-agent hybrid？
    考察点：deterministic workflow、open branch、control、recoverability。
    回答框架：很多关键流程可以用确定性 workflow 固定，例如接收、审批、提交、回滚和审计；Agent 更适合处理分类、检索、草稿、诊断和异常分支。混合架构既保留 Agent 对开放问题的灵活性，又降低完全自主执行带来的不可控风险。

40. Agent 产品上线门禁应该包含什么？
    考察点：task、tool、confirmation、recovery、trace、unauthorized、budget、latency、cost、feedback。
    回答框架：上线门禁至少包含任务成功率、工具执行成功率、高风险确认覆盖率、失败恢复率、状态更新覆盖率、观察使用率、trace 覆盖率、未授权动作率、预算超限率、P95 延迟、单位成本、eval ready、feedback loop 和 business metric。越权、trace 缺失或高风险确认不足应直接降级为建议、草稿、review 或 approval。

40A. Agent 声称任务已完成但后台没有完成，如何排查？
    考察点：final status、backend status、tool result、observation、state、truthfulness。
    回答框架：先对齐 Agent 最终回答、工具返回状态和业务系统状态。如果工具返回 failed / pending / blocked，但最终回答写 completed，就是 false completion。继续看模型是否读取了 tool error，state 是否记录失败，是否触发恢复或人工接管，prompt 是否禁止失败时声称完成。修复要把成功回答绑定到工具 success 和业务验收器，而不是绑定到模型自然语言。

40B. 如何设计一个 Agent 落地事故审计 demo？
    考察点：trace、tool schema、permission、confirmation、budget、injection boundary、gate。
    回答框架：可以构造小型 Agent trace 集，覆盖正常任务、工具失败但声称完成、参数结构合法但业务归属错误、无权高风险工具、工具结果携带不可信指令、循环检索超预算和 trace 缺失。脚本输出 task success、plan feasibility、tool selection accuracy、argument validity、tool execution success、observation use、state update、confirmation coverage、false completion、unauthorized action、budget overrun、trace completeness、tool result injection block、root cause 和 failed gates。

40C. 如何设计一个最小 Coding Agent Harness？
    考察点：runtime、action parser、tool registry、permission gate、sandbox、trace、replay、evaluation。
    回答框架：先定义 harness 是模型外层的运行和控制框架，而不是单个 prompt 或模型 API。最小闭环包括 context builder、model adapter、action parser、tool registry、execution engine、permission gate、state store、trace logger 和 evaluator；最小工具包括 read file、search、apply patch 和 run tests。上线门禁要看 parse valid rate、tool execution success、unauthorized execution rate、confirmation coverage、observation use、state update、trace completeness、budget overrun 和 task success。核心观点是 coding agent harness 不能只看最终 diff，还要证明动作受控、反馈被使用、状态可恢复、trace 可 replay。

40D. 如何设计一个可恢复的 Agent Runtime？
    考察点：session、task state、context、adapter、parser、error handler、checkpoint、cancel / timeout、trace。
    回答框架：先区分 harness 和 runtime：harness 是完整系统边界，runtime 是执行内核。Runtime 要维护 session isolation、task phase、context budget、model adapter、action parser、execution result、state store、trace logger 和 error handler。关键门禁包括 session isolation、合法状态转移、context budget / relevance、adapter success、执行结果结构化、错误恢复成功率、checkpoint coverage、cancel / timeout handling 和 trace completeness。回答时要说明失败恢复策略：parse error 可以修复或重问模型，permission error 要拒绝或请求用户，patch conflict 要重新读取文件，timeout / cancel 要处理子进程、部分输出、checkpoint 和 trace 状态。

40E. 如何审计一个 Coding Agent Workflow 是否可靠？
     考察点：task classification、repository exploration、plan、minimal patch、validation、feedback、user change、risk permission、trace。
     回答框架：把 workflow 拆成任务分类、仓库探索、计划、小步编辑、验证、失败反馈使用、用户改动保护、高风险动作确认和最终总结。指标上看 task type accuracy、repository exploration coverage、patch precision / recall、unrelated change rate、validation coverage、test pass rate、feedback use rate、user change violation rate、risky action protection 和 trace completeness。关键观点是不能只看最终 diff，要证明 agent 读对上下文、改对范围、测试失败后会修、不会覆盖用户改动，也不会绕过权限边界。

40F. 如何设计和审计一个 Tool Registry？
     考察点：identity、description、input / output schema、runtime metadata、permission、risk、version、lifecycle、owner、eval、provider projection、audit。
     回答框架：Tool Registry 不是 prompt 里的工具说明，而是企业工具能力的 source of truth。每个工具要定义 namespace / name、description、input schema、output schema、executor、timeout、retry、concurrency、required permissions、data scope、risk level、side effect、confirmation、idempotency、version、schema hash、lifecycle、owner、eval dataset、provider projection 和 registry audit。模型可见字段只包含 name / description / schema 的投影，runtime 字段必须由系统强制执行。审计指标包括 tool registry identity coverage、tool description quality、schema contract coverage、runtime metadata coverage、tool permission binding coverage、risk annotation coverage、version trace coverage、lifecycle policy pass rate、owner SLO coverage、tool eval binding coverage、provider projection readiness、registry audit completeness 和 tool registry gate。核心观点是工具越多，越不能靠散落函数或单一 provider 格式管理，必须把工具变成可发现、可授权、可评估、可回放和可下线的企业资产。

40G. 如何设计和审计 Coding Agent 的文件编辑系统？
     考察点：workspace root、read/search、apply patch、context match、secret block、concurrent edit、diff、rollback。
     回答框架：先把文件系统能力分成只读和写入：list / search / read / diff 默认只读，apply patch / create / delete / move 属于写入。所有路径必须基于 workspace root 做规范化和 containment 校验，路径穿越、workspace 外路径、secret 文件、二进制文件和 generated 文件直改要默认阻断。编辑上优先 apply patch 和唯一 old context，避免 whole-file rewrite；编辑前校验读取 hash 或 mtime，发现用户并发修改就重新读取或请求确认。审计指标包括 workspace containment、secret file block rate、read truncation clarity、patch context match、patch apply success、concurrent edit protection、unrelated diff rate、rollback readiness、diff faithfulness 和 file edit gate。

40H. 如何设计和审计 Coding Agent 的终端执行系统？
     考察点：bash tool、risk level、allowlist、denylist、confirmation、network、timeout、output、secret masking、trace。
     回答框架：终端工具不能把 shell 裸暴露给模型，而要把 command、working directory、timeout、reason、expected effect、requires network 和 risk hint 作为结构化输入。Runtime 执行前要做 working directory containment、命令风险分级、allowlist / denylist、用户确认和 sandbox 策略；低风险只读命令、最小范围测试、lint 和 type check 可自动执行，高风险删除、安装依赖、网络执行、数据库、读取 secret、git reset / clean / push 必须确认或阻断。执行中要设置 timeout、取消子进程和资源限制；执行后要结构化返回 exit code、stdout / stderr 摘要、truncated、artifact ref、masked、error type 和 command trace。审计指标包括 command risk accuracy、auto execution precision、dangerous command block rate、confirmation coverage、network access control、timeout cancellation coverage、command output truncation、secret masking coverage、command trace completeness 和 command safety gate。

40I. 如何设计和审计 Coding Agent 的上下文管理系统？
     考察点：context builder、token budget、key context、constraint、task state、summary、stale、trust boundary、memory。
     回答框架：上下文管理不是把历史全部拼进 prompt，而是由 context builder 从用户目标、系统规则、task state、相关文件、搜索结果、终端输出、当前 diff、trace 和 memory 中选择下一步决策最需要的信息。用户最新指令、安全规则、当前编辑文件、相邻测试、最新失败信号和当前 diff 要优先保留；长日志、旧对话、搜索 snippet 和历史工具输出要压缩并标记来源。摘要必须区分事实、假设和待验证，关键决策前要回到原始文件或 trace；不可信 README、issue、网页和工具输出只能作为 data-only；长期 memory 写入要检查来源、敏感性、验证状态、作用域和过期策略。审计指标包括 context budget utilization、key context recall、context precision、constraint retention、task state completeness、summary faithfulness、stale summary rate、tool output compression fidelity、current diff coverage、trust boundary coverage、memory write safety 和 context builder gate。

40J. 如何设计和审计 Coding Agent 的权限模型与安全沙箱？
     考察点：least privilege、permission matrix、secret、network egress、sandbox、confirmation、dry run、audit trace、prompt injection。
     回答框架：权限模型要放在模型外的 runtime/controller 中执行，而不是只写进 prompt。先按用户、任务、动作、资源、风险和副作用设计权限请求与权限矩阵：只读源码、搜索、git status 可低风险自动执行；写文件要绑定 diff / dry run；install、网络外发、数据库、Git push 和不可逆动作要确认或拒绝；secret、workspace 外路径和不可信内容诱导的高权限动作默认阻断。沙箱用于限制文件系统、网络、环境变量、CPU/内存、时间和子进程，但不能替代权限矩阵、参数校验、secret 管理和审计。审计指标包括 least privilege coverage、permission matrix coverage、unauthorized action block rate、high-risk confirmation coverage、secret access block rate、network egress control、sandbox enforcement coverage、dry run coverage、audit log completeness、irreversible action protection 和 permission sandbox gate。

40K. 如何设计和审计 Coding Agent 的 Trace、日志、回放与可观测性？
     考察点：trace schema、span tree、artifact、version、replay、privacy masking、error attribution、final consistency、metrics、eval export。
     回答框架：Trace 要记录一次 agent run 的完整链路，而不是只保存最终回答。任务级字段包括 trace id、session、task、workflow、状态、开始/结束时间和环境摘要；span 级字段包括 span id、parent、类型、时间、状态、输入/输出摘要、artifact ref、错误类型和 root cause；artifact 保存 diff、截断命令输出、截图、长工具结果等大对象；版本字段要覆盖模型、prompt、工具 registry、git commit、sandbox 镜像和权限策略。Replay 分只读、模拟和真实三层，真实 replay 需要环境快照、确定性工具结果、沙箱和权限策略。审计指标包括 trace schema completeness、span schema completeness、span tree validity、timeline validity、artifact reference coverage、version capture coverage、replay readiness rate、privacy masking coverage、error attribution coverage、final status consistency、metric export coverage、eval export coverage 和 trace replay gate。

40L. 如何设计和审计 Coding Agent 的 Evaluation Harness？
     考察点：dataset bucket、environment reset、validator、trajectory、baseline fairness、regression、safety、cost、flaky、report。
     回答框架：Evaluation Harness 不是只跑 benchmark，而是自动加载任务集、初始化 repo / sandbox 环境、运行 agent、收集 trace / artifacts、调用验收器、计算指标、保存结果、对比 baseline 并生成回归报告。Agent eval 的任务 schema 要包含 task type、初始环境、允许工具、权限策略、成功条件、timeout、风险标签和样本权重；成功条件不能只看最终回答，要看测试、diff scope、guardrails、trace 和最终总结是否一致。审计指标包括 eval dataset bucket coverage、environment reproducibility、validator coverage、weighted task success、partial success score、diff scope safety、trace coverage、baseline fairness、regression pass rate、unsafe execution rate、cost budget pass rate、flaky task rate、eval version capture、eval report completeness 和 evaluation harness gate。

40M. 如何从公开资料分析 Claude Code 类 Coding Agent 架构？
     考察点：资料边界、CLI 控制面、session、context、工具、权限、sandbox、memory、hooks、MCP、trace、eval readiness。
     回答框架：先声明边界：只把官方文档和可观察产品行为当作确定证据，闭源内部 planner、prompt、排序策略和隐藏状态只能作为系统设计推断，不能说成官方实现。再按 terminal coding agent 的 harness 拆：CLI / slash command 是产品控制面，session manager 管长任务状态，context builder 选择用户目标、规则、文件、diff、工具结果和 memory，model adapter 调用模型，action parser / tool registry / execution engine 把模型动作变成受控文件、搜索、编辑和终端工具，permission / sandbox 负责最小权限、secret、网络、不可逆动作和人工确认，trace / local transcript / eval harness 负责复盘和回归。审计指标可以看 architecture evidence coverage、required module coverage、access governance、core loop governance、permission control、state recovery、extension governance、observability、architecture eval readiness、high-risk surface governance 和 Claude Code architecture gate。核心观点是：分析真实 coding agent 产品时，不能只讲“模型会写代码”，要证明公开证据、runtime 边界、安全治理、trace 和评估都成立。

40N. 如何从 OpenCode 分析开源 Coding Agent Runtime？
     考察点：open runtime、config、agents、tools、permissions、MCP、custom tools、server、SDK、snapshot、compaction、eval readiness。
     回答框架：先把 OpenCode 看成开放 coding agent runtime，而不是普通 CLI。公开模块可以拆成多客户端入口、server / SDK、session manager、agent manager、provider adapter、tool registry、permission engine、execution engine、file / shell / search / LSP / MCP / custom tools、snapshot / diff / revert、event stream 和 config 控制面。配置系统是 runtime 控制面，管理模型、agents、tools、permissions、MCP、plugins、instructions、compaction 和组织级策略；agent 系统体现权限隔离，Plan / Build / subagent / custom agent 应有不同工具和 step budget；工具和 MCP 体现扩展性，但必须绑定 allow / ask / deny、schema、namespace、鉴权、上下文成本和 trace；server / SDK 让 runtime 可编程，也必须讨论认证、网络边界和审计。审计指标包括 open runtime module coverage、config governance、agent permission isolation、tool permission binding、MCP namespace coverage、custom tool schema coverage、server API governance、snapshot recovery、architecture eval readiness、high-risk surface governance 和 OpenCode architecture gate。

40O. 如何横向比较 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent 和 OpenHands？
     考察点：资料边界、产品入口、上下文来源、工具执行、权限治理、编辑恢复、评估准备度、企业治理、高风险能力面。
     回答框架：先声明边界：只能把官方文档、官方仓库和可观察产品形态当作确定证据，闭源内部 planner、prompt、工具排序、trace 存储和模型路由不能说成事实。然后按场景比较：日常 IDE 场景提高 selection、diagnostics、rules、IDE apply 和 review 权重；企业平台提高 RBAC、audit、managed settings、network control、server / SDK、MCP / custom tools 和安全评估权重；研究 benchmark 提高 issue、trajectory、batch、environment、test runner 和 regression 权重。指标可以拆成 product surface coverage、context source coverage、tool execution coverage、permission governance coverage、edit recovery coverage、agent eval readiness、enterprise governance coverage 和 cross-agent risk governance。最后用 coding agent comparison gate 检查 permission、eval 和 risk 三个硬门禁，避免因为单次 demo 顺滑或模型强就忽略治理缺口。

40P. 如何把 MCP 和 A2A 集成进 Agent Harness？
     考察点：MCP Host / Client / Server、resources / prompts / tools、A2A Agent Card、task lifecycle、capability registry、namespace、schema、permission、context budget、trace、replay、version。
     回答框架：先区分协议边界：MCP 是 agent 连接工具、资源和 prompt 的协议，A2A 是 agentic application 之间发现、委派任务、streaming 和交换 artifact 的协议。接入 harness 时不要把它们简单塞进 prompt，而要先做 capability discovery，把 MCP tools / resources / prompts 和 A2A remote agents 注册进 protocol capability registry，绑定 namespace、schema、风险等级、认证 scope、权限策略、timeout、输出限制和 trace 字段。MCP tool 调用和 A2A task delegation 都应经过 permission engine，高风险写外部系统、敏感读取、部署和跨 agent 委派必须 ask 或 deny；外部输出进入 context builder 时要标注 trust boundary、做脱敏、截断、摘要和 artifact 引用；trace 要记录协议版本、能力 id、输入输出摘要、权限结果、错误、耗时和版本；evaluation / replay 要使用 mock、snapshot 或 dry run，避免真实外部系统变化破坏复现。审计指标包括 capability discovery coverage、namespace isolation coverage、protocol schema validity、protocol permission binding、high-risk approval coverage、context output budget coverage、external trust boundary coverage、A2A lifecycle coverage、protocol trace coverage、protocol replay readiness、protocol version capture 和 protocol integration gate。

40Q. 如何系统设计一个生产级 Agent Harness？
     考察点：需求澄清、模块边界、session、orchestrator、context builder、model adapter、capability registry、permission engine、execution engine、sandbox、diff/revert、trace、replay、eval、MCP/A2A、server/SDK、RBAC、治理。
     回答框架：先澄清产品形态、任务范围、执行权限、外部系统、安全等级、评估和多租户要求。高层架构可以拆成 client / server、session manager、agent orchestrator、context builder、model adapter、capability registry、permission engine、execution engine、sandbox / diff / git / LSP / MCP / A2A、trace store、replay engine、evaluation runner 和 config / policy / auth / RBAC。核心数据模型包括 Session、Message、ToolCall、PermissionRequest、FileChange、TraceEvent 和 EvalRun。运行链路是：用户消息进入 session，context builder 构建预算受控上下文，model adapter 调模型，orchestrator 解析动作，permission engine 做 allow / ask / deny，execution engine 在沙箱和 workspace 边界内执行，结果进入 trace 和下一轮上下文，最终生成 diff、测试结果和总结。生产级设计要补充系统指标：runtime module coverage、interface contract coverage、agent state machine coverage、permission integration coverage、context control coverage、execution isolation coverage、system trace coverage、system replay readiness、system eval readiness、recovery coverage、harness version capture、enterprise governance readiness 和 harness system gate。

40R. 如何排查和回答 Agent Harness 实战坑？
     考察点：triage、permission、context、edit、command、prompt injection boundary、trace、replay、eval、doom loop、environment parity。
     回答框架：先不要把问题归因成“模型不行”，而是按 user goal、context、model、tool、permission、execution、state、observability 八层排查。典型坑包括权限过大、大输出挤爆上下文、覆盖用户改动、shell hang、外部内容边界失效、trace 缺失、eval 不可比、重复工具循环、MCP tool overload 和本地 / 云端环境漂移。工程上可以定义 harness pitfall gate，指标包括 triage coverage、permission safety、context budget pass、edit safety、command safety、prompt injection boundary、trace completeness、replay readiness、eval determinism、cost loop control 和 environment parity。面试时要把每类坑讲成症状、根因、排查字段、修复策略和防复发门禁，而不是只背工具名。

40S. 如何设计和审计工具调用中的参数校验与修复？
     考察点：parse、schema、normalization、business validation、evidence、repair、clarification、unsafe block、idempotency、trace。
     回答框架：先把参数处理拆成 raw arguments、JSON parse、schema validation、low-risk normalization、business validation、permission / safety check、evidence map、repair / clarification / reject 和 execution。低风险大小写、空格、枚举别名可以白名单修复；类型、范围和额外字段错误可以把结构化 validation error 回填给模型自修复；缺少订单号、收件人、金额、路径和多个候选对象时要澄清；越权、高风险无确认、来源不可信或会改变业务含义的修复必须拒绝或转人工。审计指标包括 parse success rate、schema pass rate、business pass rate、argument evidence coverage、safe normalization rate、repair success rate、clarification coverage、unsafe argument block rate、retry budget pass rate、idempotency key coverage、repair trace completeness 和 argument repair gate。核心观点是参数修复不是让系统更“会猜”，而是让每次执行前的参数来源、风险和修改过程都可控可审计。

40T. 工具结果如何安全进入模型上下文？
     考察点：tool_call_id、tool result projection、source metadata、redaction、context budget、trust boundary、error fidelity、citation、freshness、memory boundary。
     回答框架：先声明 tool result 是 observation，不是 system / developer / user instruction。Runtime 应用内部 ToolResult 对象承接 provider 差异，再做 ID 对齐、status 区分、字段投影、敏感字段脱敏、来源和时间标注、trust level 标注、上下文预算控制、长结果压缩、冲突标注和引用保留。外部网页、邮件、文档和搜索结果要按不可信 data-only 内容处理，不能让其中的指令触发高风险工具；timeout、rate limit、permission denied 不能伪装成 empty result；天气、价格、库存等短期事实不能随意写入长期 memory。审计指标包括 result ID alignment rate、safe projection rate、redaction coverage、context budget pass rate、source metadata coverage、injection containment rate、error status fidelity、compression fidelity、conflict labeling rate、citation support rate、freshness pass rate、memory boundary pass rate 和 tool result context gate。核心观点是工具执行成功不代表结果可以原样进入上下文。

40U. 如何设计工具调用失败、重试、降级和人工接管？
     考察点：structured tool error、retryable / non-retryable、retry budget、idempotency、unknown execution state、fallback honesty、circuit breaker、timeout cancellation、human handoff、trace。
     回答框架：先把失败恢复拆成 Detect、Classify、Recover、Report。Detect 要覆盖 schema error、permission denied、timeout、rate limit、下游 5xx、空结果和重复循环；Classify 要区分模型错误、runtime 错误、工具错误、retryable / non-retryable、只读失败和有副作用未知状态；Recover 时只对临时、幂等、预算内的失败做指数退避和 jitter，权限错误直接拒绝或引导申请权限，参数错误修复或澄清，工具不可用时用缓存、备用工具或部分结果降级，高风险未知状态查询状态或转人工；Report 要给模型结构化错误，给用户清楚说明失败、缓存、未知状态和下一步，给运维完整 trace。审计指标包括 failure detection coverage、error classification accuracy、retry policy precision、retry budget pass rate、idempotency protection rate、unknown state escalation rate、fallback honesty rate、circuit breaker containment、timeout cancellation coverage、human handoff readiness、user-visible error clarity、failure trace completeness 和 tool failure recovery gate。核心观点是生产级工具调用不是永远不失败，而是失败时不重复危险动作、不编造结果、不隐藏不确定性。

40V. 如何评估 function calling / tool use 系统？
     考察点：tool call recall / precision、tool selection、argument correctness、argument source、execution、observation use、task success、safe / unsafe failure、cost、regression。
     回答框架：不要只看最终回答，要基于 trace 拆成七层：调用决策、工具选择、参数生成、runtime 执行、结果使用、任务成功、安全治理。离线 eval 集要包含应该调用、不该调用、多工具混淆、缺参、参数歧义、高风险工具、prompt injection、工具失败、多轮和成本边界；结构化指标包括 tool call recall、tool call precision、tool selection accuracy、tool set precision / recall、argument completeness、argument value accuracy、argument source coverage、execution success rate、observation use correctness、safe failure rate、unsafe failure rate、cost latency budget pass rate 和 regression pass rate。LLM judge 只适合辅助判断自然语言质量，工具名、参数、权限、执行和 trace 应优先由程序检查。核心观点是工具调用评估对象不是一句回答，而是从模型决策到工具执行再到最终回答的完整 trace。

40W. 如何设计 Tool Router，决定每轮给模型哪些工具？
     考察点：Registry、scenario filter、permission filter、risk filter、intent routing、candidate recall / precision、tool choice、parallel、provider capability、router trace。
     回答框架：Tool Router 位于 Registry 和模型 tool choice 之间，先从 Registry 取 active tools，再做场景、权限、租户、风险和 workflow 状态等硬过滤；随后用规则、embedding 或 LLM router 做语义召回和重排；最后输出 candidate tools、blocked tools、tool choice mode、forced tool、parallel policy、max steps、成本预算和 router trace。安全和权限必须用确定性策略，LLM router 只能辅助意图理解，不能绕过 Executor 的最终权限检查。审计指标包括 scenario filter pass rate、permission filter pass rate、risk filter pass rate、candidate recall、candidate precision、candidate size pass rate、clarification accuracy、tool choice mode accuracy、forced tool accuracy、router parallel safety、provider capability compatibility、cost budget pass rate、router trace completeness 和 tool router gate。核心观点是不要把所有工具直接交给模型，而要让模型只在相关、授权、安全、低成本、可解释的候选集合中选择。

40X. 如何设计 Tool Executor，处理同步、异步、超时和幂等？
     考察点：ToolCall / ToolResult、schema validation、business validation、permission、confirmation、sync / async、timeout、cancellation、idempotency、unknown state、structured result、executor trace。
     回答框架：Tool Executor 位于模型 tool call 和真实外部系统之间，先把 provider 原始 tool call 转成内部 ToolCall，再按工具存在性、schema、业务规则、权限、风险确认、限流、幂等和执行模式做硬校验。低延迟只读工具可以同步执行，长任务应异步返回 job id 并支持状态查询、取消、过期和通知；所有工具都要有分层 timeout，取消用户等待不等于取消下游执行。有副作用工具必须绑定 confirmation token、idempotency key、执行记录和 UNKNOWN 状态接管，超时后不能盲目重试或声称失败。结果侧要做 output schema validation、字段投影、脱敏、错误规范化和 trace。审计指标包括 execution request validity、schema validation pass rate、permission enforcement pass rate、execution mode accuracy、async completion tracking、timeout cancellation coverage、idempotency protection rate、unknown state escalation rate、retry safety rate、side effect confirmation coverage、structured result coverage、executor trace completeness 和 tool executor gate。核心观点是模型可以提出动作，但 Executor 必须负责把动作变成安全、幂等、可取消、可追踪、可审计的真实执行。

40Y. 如何设计工具权限模型，防止越权和跨租户泄露？
     考察点：trusted context、tenant isolation、RBAC / ABAC、tool permission、object permission、field projection、action context policy、prompt injection、least privilege、token audience、revocation cache、audit。
     回答框架：工具权限模型回答 who can do what on which resource under what context。第一层是可信身份上下文，`user_id`、`tenant_id`、roles、scope 和 confirmation 必须由 runtime / auth 注入，不能相信模型参数。第二层是权限维度：工具级权限控制能不能调用工具，对象级权限控制能不能访问具体订单 / 客户 / 文件，字段级权限控制哪些字段能进入 tool result，动作级和上下文权限控制退款、删除、转账、权限修改等高风险动作是否需要确认、二次认证或审批。第三层是纵深防御：Router 只缩小工具候选，Executor 做强制权限，下游服务兜底，结果投影做脱敏，错误返回避免泄露资源存在性，审计日志记录 user、tenant、tool、action、resource、decision、reason 和 trace。审计指标包括 authorization decision accuracy、trusted context injection、tenant isolation pass rate、tool permission enforcement、object permission accuracy、field projection safety、action context policy accuracy、prompt injection block rate、least privilege coverage、token audience validation、revocation cache safety、error disclosure safety、permission audit completeness 和 tool permission gate。核心观点是不要让模型“自觉不越权”，而要让 runtime 在可信上下文下强制决定什么能执行、什么数据能返回。

40Z. 如何设计工具安全体系，防越权、注入、数据泄露和危险动作？
     考察点：prompt injection、untrusted content、authorization、DLP、external transfer、confirmation、SSRF、SQL、path traversal、shell sandbox、data flow、audit、alert、safety eval。
     回答框架：先把工具安全拆成四类风险：不可信内容驱动工具、越权访问和跨租户泄露、敏感数据进入错误 sink、高风险工具产生外部副作用。架构上要让 Router 缩小候选工具，Executor 做强制权限、参数校验、DLP、确认和幂等，下游服务兜底 ACL，URL / SQL / file / shell 工具各自进入 allowlist、只读、workspace 和 sandbox 边界，工具结果只作为 observation 进入上下文。外部发送、删除、支付、发布、权限修改和生产配置变更必须先草稿 / dry run / 用户确认，再用 runtime 生成的 confirmation token 执行。审计指标包括 prompt injection containment、untrusted content isolation、unauthorized access block rate、sensitive data protection、external transfer gate、dangerous action confirmation、SSRF block rate、SQL risk block rate、path traversal block rate、shell sandbox enforcement、data flow policy pass rate、tool security audit completeness、security alert completeness、safety eval regression pass rate 和 tool security gate。核心观点是工具安全不是提示词安全，而是把模型输出限制在权限、数据流、沙箱、确认、审计和回归评估组成的工程边界里。

40ZA. 如何设计工具调用 trace、replay 和审计体系？
      考察点：trace schema、span tree、ID propagation、version capture、argument lineage、permission trace、tool result trace、privacy masking、audit event、replay readiness、side-effect replay safety、metrics、alerts、eval linkage。
      回答框架：先把 trace 定义成一次工具调用的事实链，而不是普通文本日志。链路要覆盖用户请求、模型输入输出、Router 候选和过滤原因、tool call、raw / parsed / normalized / final arguments、权限决策、Executor 执行、tool result 投影、最终回答和用户可见错误。每个 span 要有稳定 ID、parent、时间、状态、输入输出摘要和 artifact 引用；版本要记录 model、prompt、tool schema、registry、router policy 和 adapter。Replay 分 trace-only、model replay 和 tool / sandbox replay，有副作用动作默认禁止 live replay。审计事件要记录 actor、tenant、tool、action、resource、decision、confirmation、timestamp、trace id 和 outcome。指标包括 trace schema completeness、ID tree integrity、version capture coverage、argument lineage coverage、permission trace completeness、tool result trace completeness、privacy masking coverage、audit event completeness、replay readiness rate、side effect replay safety、metric export coverage、alert owner coverage、eval linkage coverage 和 trace replay gate。核心观点是没有 trace 就没有可靠评估、调试、回放和安全追责。

40ZB. 如何设计工具版本管理、灰度发布和回滚门禁？
      考察点：SemVer、schema diff、description diff、output compatibility、permission policy、version matrix、offline eval、canary routing、quality / safety / cost guard、rollback、sunset。
      回答框架：先把工具发布对象定义成完整 spec，而不是后端代码版本；版本矩阵要记录 model、prompt、tool schema、executor、adapter、router policy、permission policy 和 eval dataset。发布前做 spec lint、schema diff、description 行为风险评估、output compatibility、权限策略审查和 offline eval；新增可选字段、description 扩大边界、权限收紧都要进入评估。灰度时按内部租户、小流量、低风险场景逐步扩大，并用 conversation / job sticky routing；监控 task success、tool selection、argument validation、error、latency、cost、permission denied、unsafe action、trace missing。回滚要同时恢复 schema、description、executor、router、provider adapter、permission policy 和状态兼容策略，旧版本进入 deprecated / sunset / archived 生命周期。审计指标包括 spec lint pass rate、schema backward compatibility、description behavior guard、output compatibility、permission policy compatibility、version matrix capture、offline eval pass rate、canary routing stability、canary quality guard、canary safety guard、cost latency guard、tool release rollback readiness、lifecycle coverage 和 tool release gate。核心观点是工具版本发布不是部署代码，而是治理模型可见契约、runtime 实现、权限、安全、eval、trace、灰度和回滚的一致性。

40ZC. 如何设计企业级 Agent 工具平台？
      考察点：Registry、Router、Executor、Permission、Safety、Trace / Audit、Eval、Release / Governance、Admin Console、多租户、Provider Adapter、MCP、HA、feature flag / canary、平台上线门禁。
      回答框架：先澄清工具数量、租户隔离、高风险动作、内部系统类型、多 provider、MCP 接入和审计合规要求。总体架构分成管理面和运行面：管理面负责 Tool Registry、schema、owner、风险标签、审核、eval、版本、灰度、回滚和后台审批；运行面负责工具候选路由、provider adapter、模型 tool call、Executor 强制校验、Permission Service、Safety Guard、Tool Result 投影、trace / audit 和指标上报。安全边界上，模型输出、用户输入、工具结果和外部工具都不可信，runtime 身份上下文、permission decision、executor sandbox 和 audit 才是强控制点；MCP tools 必须进入 Registry，不能绕过权限、安全和审计。可靠性上，Registry 用签名快照，Router 可降级，Permission fail closed，Executor 有 timeout、circuit breaker、idempotency 和 unknown state 接管，Trace 可异步但高风险 audit 不能丢。审计指标包括 platform module coverage、platform interface contract coverage、registry readiness、router governance coverage、executor safety coverage、platform permission integration coverage、platform safety guard coverage、platform trace audit coverage、platform eval service coverage、platform release governance coverage、platform tenant isolation coverage、platform provider adapter coverage、high availability readiness、admin ops governance coverage 和 enterprise tool platform gate。核心观点是企业工具平台不是 function calling demo，而是把企业工具能力放进可注册、可路由、可执行、可授权、可审计、可评估、可灰度和可回滚的治理闭环。

40ZD. MCP 为什么出现？它解决了什么标准化连接问题？
      考察点：Function Calling 边界、tools / resources / prompts、Host / Client / Server、集成爆炸、本地上下文、HTTP API 与插件局限、企业治理、MCP 与 A2A 区别。
      回答框架：先讲背景：模型应用不只需要函数调用，还需要访问文件、数据库、网页、IDE、Git、知识库和可复用 prompts；如果每个 host application 分别适配每个外部系统和每类能力，集成数量近似是 `|H| * |S| * |K|`，会产生重复开发、权限不一致和审计困难。MCP 的核心是让 host 侧实现 MCP Client，让外部系统暴露 MCP Server，把 tools、resources、prompts、能力发现、schema、错误和生命周期放进统一协议，近似把适配数量降到 `|H| + |S|`。再讲边界：MCP 连接的是 Host / Client 和 Server，不是 server 直接连模型；它不是 HTTP API 的替代，而是模型上下文连接层；它也不是完整企业治理平台，仍需要 Registry、权限、安全、trace、eval、灰度和回滚。审计指标包括 direct integration count、MCP integration count、integration reduction、capability model coverage、context object coverage、discovery standardization、MCP schema contract coverage、host server boundary clarity、local context control、cross client reuse、governance boundary clarity、MCP trace eval readiness、MCP A2A distinction 和 MCP background gate。核心观点是 MCP 的出现，是把模型应用接入外部上下文的方式从各自为战的插件集成推进到统一协议和可复用生态。

40ZE. MCP 里的 Host、Client、Server、Tools、Resources、Prompts 分别是什么？
      考察点：Host policy、MCP Client、MCP Server、tool schema、resource URI、prompt arguments、transport、lifecycle、roots、sampling、context budget、trace/eval。
      回答框架：Host 是模型应用，负责用户会话、调用 LLM、管理 MCP Client、决定连接哪些 server、过滤哪些能力暴露给模型、做权限确认和结果回填。MCP Client 是 Host 内的协议组件，负责连接 Server、initialize、能力发现、列出 tools / resources / prompts、调用工具、读取资源和获取 prompt。MCP Server 是暴露外部能力的一端，可以连接文件系统、Git、数据库、知识库、浏览器或企业系统，并声明 tools、resources、prompts 等能力。Tools 是可执行动作，要有 name、description、input schema、结果结构和错误语义；Resources 是可寻址上下文，要有 URI、metadata、范围和内容；Prompts 是可复用提示模板，要有参数、版本和审查。Transport 决定通信方式，Lifecycle 负责初始化和能力协商，Roots 限制本地资源范围，Sampling 是 server 请求 Host 调模型的高级能力，必须谨慎控制。审计指标包括 host policy ownership、MCP client server boundary、server capability declaration、MCP tool schema and result、MCP resource URI metadata、MCP prompt argument review、MCP lifecycle negotiation、MCP transport policy、MCP roots boundary、MCP sampling control、host capability filtering、MCP context budget 和 MCP concept gate。核心观点是 MCP 基本概念是一组职责边界，不是简单的名词列表。

40ZF. 如何实现和审计一个最小 MCP Server？
      考察点：metadata、capabilities、tools/list、tools/call、input schema、handler、structured result、structured error、resources、prompts、transport、Host 接入、safety baseline、trace。
      回答框架：先说明最小 MCP Server 不是一个裸函数，而是一个可被 Client 初始化、发现能力、列出工具、调用工具、返回结果或错误的协议服务。实现上先定义 server name / version / capabilities；再注册 tools，每个 tool 有 name、description、strict input schema 和 handler；调用时先检查工具名，再校验 arguments，再执行 handler；成功返回可回填的结构化 result，失败映射成 unknown tool、invalid arguments、timeout、permission denied、upstream error 或 internal error 等结构化错误。可选 resources 要有 list/read 和 roots / URI 范围限制，prompts 要有 list/get、参数和审查状态；stdio 适合本地进程边界，远程 HTTP 类 server 要补认证、授权、TLS、限流和审计。审计指标包括 MCP server metadata readiness、capability declaration、tool registry readiness、strict schema coverage、handler execution coverage、argument validation coverage、structured result coverage、structured error coverage、resource scope coverage、prompt template coverage、transport policy coverage、host connection readiness、safety baseline coverage、trace readiness 和 MCP server gate。核心观点是最小实现要证明“可发现、可调用、可校验、可失败、可治理”，而不是只证明函数能返回天气。

40ZG. MCP Tool 和传统 Function Calling 到底有什么区别？
      考察点：protocol layer、tool discovery、tools/resources/prompts、execution boundary、projection mapping、provider adapter、MCP adapter、lifecycle、registry、security、latency trade-off。
      回答框架：先分层：Function Calling 是 Host 与模型 provider 之间的模型 API 表达，解决模型如何用结构化 tool call 表达调用意图；MCP Tool 是 MCP Server 暴露给 MCP Client 的能力对象，解决外部工具、resources、prompts 如何被模型应用发现、连接和复用。再讲组合链路：MCP Server 暴露 tools/list，Host 通过 MCP Client 发现能力，经过 MCP Adapter 导入内部 Tool Registry，再由 Provider Adapter 投影成模型 API 的 tool schema；模型返回 provider tool call 后，Host 映射回 MCP `tools/call` 并执行权限、确认、trace 和结果回填。差异包括工具来源、发现机制、执行边界、是否原生包含 resources/prompts、生命周期和连接治理、安全面、错误面、延迟可用性。选型上，小系统和单应用工具直接 function calling 更简单；跨客户端复用、本地上下文、独立工具团队、resources/prompts 和插件生态更适合 MCP。审计指标包括 protocol layer clarity、tool discovery boundary、MCP capability scope coverage、MCP execution boundary clarity、MCP projection mapping coverage、adapter separation coverage、MCP lifecycle version awareness、MCP governance registry import、MCP security boundary enforcement、MCP use case selection fit、MCP error surface separation、MCP latency availability tradeoff 和 MCP function calling gate。核心观点是 MCP 不是 Function Calling 的同义词，也不是替代品；MCP 常常是 Function Calling 上游的能力连接层。

40ZH. 如何设计和审计 MCP Resources？
      考察点：resource URI、metadata、resources/list、resources/read、roots、tenant、field projection、context budget、citation、untrusted content、freshness、templates、subscriptions、trace/eval。
      回答框架：先区分 Resource 和 Tool：Tool 是执行动作，Resource 是可读取上下文。设计资源时用稳定 URI 标识文件、数据库记录、网页、日志或 RAG chunk，并带 name、title、mimeType、size、sensitivity、trust level、last modified、etag/version 等 metadata。`resources/list` 要按租户、权限、roots、分页和标签过滤，避免泄露资源存在性；`resources/read` 要检查 URI、allowed roots、对象权限、字段权限、大小、mimeType 和敏感级别。文件资源要做 canonical path 和 roots containment；数据库资源要用逻辑 URI、tenant filter、对象权限和字段投影；网页资源要标注 untrusted content，不能把页面指令当系统指令；日志资源要限制窗口、脱敏并保留 trace id。Host 决定哪些片段进入模型上下文，必须做 token budget、chunk / summary / top K、citation 和 freshness 检查。Resource templates 要有参数边界和权限校验，动态资源要处理 version、etag、list changed 或 subscription。审计指标包括 resource URI validity、resource metadata completeness、resource list filtering boundary、resource read scope enforcement、MCP roots containment、MCP resource permission enforcement、MCP resource field projection safety、resource context budget control、resource citation traceability、untrusted resource labeling、resource freshness version awareness、resource template boundary、resource subscription awareness、resource trace eval readiness 和 MCP resource gate。核心观点是 MCP Resources 不是“把外部文本塞给模型”，而是把上下文变成可寻址、可授权、可压缩、可引用、可防注入、可追踪的资源对象。

40ZI. 如何设计和审计 MCP Prompts？
      考察点：prompts capability、prompts/list、prompts/get、arguments、PromptMessage、role boundary、tools/resources dependency、version、eval、permission、injection、user display、list changed、trace。
      回答框架：先说明 Prompt 与 Tool / Resource 的区别：Tool 是动作，Resource 是上下文，Prompt 是可复用的任务模板和工作流入口。MCP Server 需要声明 prompts capability，Host 通过 `prompts/list` 发现 prompt metadata，通过 `prompts/get` 传入字符串参数并拿到 messages。设计时要让 prompt 有 name、title、description、arguments、owner、risk、version、approval、eval dataset 和依赖的 tools/resources；Host 要按租户、角色、场景过滤可见 prompt，并把来源 server、owner、风险和审核状态展示给用户。`prompts/get` 要验证 required arguments、类型和边界，把用户参数作为数据转义和包装，不能拼进高优先级指令；MCP Prompt 返回的 role 不能覆盖 Host system policy，Host 仍然控制最终上下文拼接、工具候选、资源读取和安全策略。版本治理上要记录 template hash、arguments schema、changelog、approval、eval 和回滚；动态列表要处理 `notifications/prompts/list_changed` 和缓存失效；trace 要记录 prompt name、version、template hash、arguments hash、rendered messages、依赖工具资源和最终输出。审计指标包括 prompt capability declaration、prompt discovery coverage、prompt argument schema coverage、prompt required argument validation、prompt rendering safety、prompt role boundary enforcement、prompt dependency alignment、prompt version governance、prompt eval binding、prompt permission enforcement、prompt injection containment、prompt user control、prompt list change awareness、prompt trace readiness 和 MCP prompt gate。核心观点是 MCP Prompts 不是给外部 Server system prompt 权限，而是把领域 SOP 和提示模板变成可发现、可参数化、可降权、可评估、可授权、可追踪的协议能力。

40ZJ. MCP 权限、安全和本地沙箱怎么设计？
      考察点：server allowlist、签名、OAuth / token audience、scope 最小化、roots、文件 denylist、SSRF、shell sandbox、prompt injection、confirmation、data flow、local credentials、tenant isolation、supply chain、trace/eval。
      回答框架：先把 MCP 安全边界放在 Host / runtime，而不是模型。第一层是连接治理：Server 要经过 allowlist、签名、版本固定、发布者校验和能力展示，用户或管理员明确批准后才连接。第二层是授权：远程 Server 要校验 token audience、scope、session 和过期时间，禁止 token passthrough，scope 只给当前能力需要的最小范围。第三层是本地资源沙箱：文件系统必须有 roots、canonical path、symlink 防护、敏感文件 denylist、大小限制；网络工具要阻断 localhost、私有网段、metadata 地址、非 allowlist 域名和异常重定向；shell / code execution 必须在非 root、限 CPU/内存/磁盘/网络、限时、临时可销毁的沙箱里跑。第四层是工具权限和数据流：高风险写操作、外发、生产变更、shell 执行要确认或审批；敏感资源不能流向 external sink，外发前做 DLP 和摘要展示。第五层是 prompt injection 防御：resource、tool result、web page 和第三方 prompt 都是 untrusted data，不能覆盖 Host policy 或自动触发危险工具。第六层是多租户和供应链：tenant 来自可信 auth context，缓存和日志按租户隔离，Server 包、版本、hash、自动更新和能力变化纳入供应链治理。审计指标包括 server connection governance、authorization token binding、scope minimization、roots sandbox containment、file secret blocking、network SSRF protection、MCP shell sandbox enforcement、prompt injection data boundary、high risk confirmation、sensitive data flow control、local credential isolation、MCP tenant isolation、server supply chain governance、MCP security trace readiness、MCP security eval coverage 和 MCP security gate。核心观点是 MCP 安全不是提示词安全，而是 Host 强策略、Server 最小权限、沙箱隔离、数据流控制、确认、trace 和安全 eval 的工程闭环。

40ZK. 如何设计和审计 MCP 与 IDE、知识库、数据库、浏览器、终端的集成？
      考察点：capability registry、namespace、IDE context routing、knowledge citation、database read-only query、browser untrusted content、terminal sandbox、context budget、cross-server data flow、approval、output projection、trace/eval。
      回答框架：先说明 MCP 集成不是把多个 API 直接接给模型，而是 Host 通过 MCP Client 发现多个 Server 的 tools / resources / prompts，再导入 capability registry，统一绑定 namespace、owner、version、risk、permission、timeout、context budget 和 trace 字段。IDE 场景要暴露当前文件、选区、diff、诊断、符号和测试结果，并让写操作走 patch preview；知识库场景要用 search tool + resource URI + citation + freshness + 用户权限；数据库场景不要裸露任意 SQL，要用 schema resource、只读参数化查询和业务语义工具，字段投影、超时、成本和审计必备；浏览器场景要把页面标成 untrusted resource，对 click/type/submit/download/upload 做 action preview、域名策略和确认；终端场景要拆成 run_tests、run_build、run_formatter 等语义工具，并受 cwd、文件系统、网络、env、超时、资源配额和输出脱敏限制。多 Server 难点在跨系统数据流：数据库结果、知识库机密、网页内容和终端输出不能随意流向浏览器、外部 sink 或代码补丁。审计指标包括 MCP capability registration coverage、MCP namespace isolation coverage、MCP IDE context routing、MCP knowledge citation traceability、MCP database query governance、MCP browser action governance、MCP terminal sandbox governance、MCP integration context budget control、MCP cross server data flow control、MCP high risk approval coverage、MCP output projection coverage、MCP integration trace readiness、MCP integration eval coverage 和 MCP integration gate。核心观点是能连上多个 MCP Server 不等于可上线；可上线要证明 Host 对上下文、权限、数据流、审批、输出和 trace 有强治理。

40ZL. 为什么 Agent 之间需要 A2A 协议？
      考察点：Agent Card、discovery、task delegation、TaskState、Message、Artifact、context boundary、auth / permission、failure handling、trace / eval、A2A 与 MCP 区分。
      回答框架：先讲背景：多 Agent 协作不是两个模型互相发自然语言消息，而是跨系统、跨团队、跨权限边界的任务委派。自然语言聊天缺少能力发现、任务状态、结构化结果、上下文权限和失败语义；普通 HTTP API 可以承载通信，但通常缺少 Agent 特有的长任务、input_required、streaming / push update、Artifact 引用、取消和责任边界。A2A 的核心价值是把 remote agent 变成可发现、可委派、可追踪、可治理的协作对象：Agent Card 声明能力和接口，Task 表达目标、上下文、截止时间和幂等，Message 承载多轮协商，Artifact 引用结果产物，TaskState 表达 submitted / working / input_required / completed / failed / canceled 等状态。风险上要控制上下文最小化、身份和授权、权限传递、错误传播、循环委派、产物验证和审计。审计指标包括 agent card completeness、agent discovery readiness、task delegation contract、A2A task lifecycle coverage、A2A message structure coverage、A2A artifact reference coverage、A2A context boundary control、A2A permission boundary、A2A MCP distinction、A2A failure handling coverage、A2A trace readiness、A2A eval coverage 和 A2A background gate。核心观点是 A2A 不是让 Agent 更会聊天，而是让 Agent 之间的能力、任务、状态、上下文、产物和信任关系能被系统管理。

40ZM. Agent Card 应该如何设计和审计？
      考察点：public card、extended card、supported interfaces、skills、input / output modes、security schemes、security requirements、discovery、version / cache、signatures、routing、trace / eval。
      回答框架：先把 Agent Card 定义成远程 Agent 的结构化能力契约，而不是宣传页。它至少要声明身份、描述、版本、owner、supported interfaces、capabilities、skills、默认输入输出模式、security schemes、security requirements、文档、provider 和缓存 / 签名信息。服务发现可以走 well-known URI、registry / catalog、direct config 或定制发现机制；well-known 或 public card 只应披露粗粒度能力和认证入口，敏感 skill、内部 endpoint、租户策略、scope 和 SLA 应放到认证授权后的 extended card。路由时不能只看语义相似度，而要先用 skill tags / input output modes / tenant / health / version 做结构化过滤，再做权限校验、成本延迟排序和模型辅助选择。治理上要记录 card version、etag / hash、发现来源、候选集合、过滤原因、选中 skill、权限决策和最终结果。审计指标包括 agent card field completeness、agent skill declaration quality、supported interface readiness、agent card security coverage、agent card version cache readiness、agent discovery match quality、agent routing decision quality、extended agent card control、agent card trace readiness、agent card eval coverage 和 agent card gate。核心观点是：能发现 Agent 不等于能调用 Agent；只有 Card、发现、权限、版本、trace 和 eval 都过线，远程 Agent 才是可治理的系统组件。

40ZN. 如何设计和审计 A2A 任务委派、状态同步和结果返回？
      考察点：Task、contextId、Message、Part、Artifact、TaskState、input-required、auth-required、failed、canceled、retry、idempotency、parallel aggregation、permission boundary、trace / eval。
      回答框架：先把 A2A 任务委派定义成跨 Agent 的可追踪任务生命周期，而不是普通聊天消息。Task 至少要有 `id`、`contextId`、`status`、`message`、`artifacts`、`history` 和 `metadata`；metadata 或内部任务模型要记录 requester、assignee、parent task、deadline、idempotency key、permission scope、data policy 和 callback / event channel。状态机要使用 submitted、working、input-required、auth-required、completed、failed、canceled、rejected 等稳定语义，内部 accepted / queued / running / expired 可以存在，但必须映射回协议状态。`input-required` 要带结构化问题和回复路径，`failed` 要带 code、category、retryable、details 和 trace id，`completed` 要返回结构化结果和 Artifact 引用，取消要进入 `canceled` 或明确失败。安全上强调委派不等于转授权，传给下游的上下文和 scope 要最小化，并行委派要汇总状态、证据、冲突和部分失败。审计指标包括 A2A task contract coverage、A2A state transition validity、A2A input required handling、A2A message structure coverage、A2A artifact metadata coverage、A2A error semantics coverage、A2A retry idempotency coverage、A2A cancellation coverage、A2A delegation permission boundary、A2A parallel aggregation readiness、A2A task trace readiness、A2A task eval coverage 和 A2A task gate。核心观点是：A2A 能力不是最终答案像样，而是任务、状态、产物、权限和审计链路都可治理。

40ZO. Multi-Agent 协作中的消息格式和上下文边界怎么设计？
      考察点：Message、Part、messageId、taskId、contextId、role、metadata、source trust、instruction / data separation、context policy、minimal context、resource / artifact ref、claim grounding、summary constraints、trace / eval。
      回答框架：先说明多 Agent 消息不能是纯文本转发，而要能让 Runtime 判断来源、角色、意图、权限和数据边界。协议层 Message 应包含 `messageId`、`taskId`、`contextId`、`role`、`parts` 和 metadata；工程 metadata 可记录 sender、recipient、intent、context policy、trace id、source、trust、classification 和 forwarding rule。Part 要区分 text / data / file 等协议类型，并在业务层标注 instruction、resource_ref、artifact_ref、claim、summary、log_excerpt、policy_ref 等语义。安全上坚持最小上下文原则，敏感或大对象优先传引用而不是复制；网页、日志、文档、数据库内容都应标成 untrusted data，不能升级成 instruction；system prompt 和内部策略只能传 policy_ref 或约束摘要，不能全量转发。Agent 输出要标注 claim_type、confidence、evidence 和 limitations，防止 hypothesis 被下游写成 fact；摘要要保留原始目标、限制、不确定性、证据和过期信息。审计指标包括 A2A message contract coverage、A2A part typing coverage、A2A source trust labeling、A2A instruction data separation、A2A minimal context coverage、A2A reference over copy coverage、A2A context policy enforcement、A2A sensitive redaction coverage、A2A claim grounding coverage、A2A summary constraint retention、A2A message trace readiness、A2A message eval coverage 和 A2A message gate。核心观点是：Multi-Agent 的主要风险之一是上下文在 Agent 之间流动时被误传、污染、泄露或错误升级。

40ZP. A2A 与 MCP 在系统设计里怎么划分？
      考察点：Agent-to-Agent、Agent-to-Tool、tools/resources/prompts、Agent Card、Task lifecycle、artifact、tool result、context ownership、permission boundary、trace / eval。
      回答框架：先给一句话：MCP 连接能力，A2A 连接协作者。MCP 面向 Host / Client / Server 之间的 tools、resources、prompts，适合读文件、查库、跑测试、搜索文档、浏览器动作、终端动作等明确操作；A2A 面向远程 Agent 的 Agent Card、Task、Message、Status、Artifact 和协作生命周期，适合合同审查、指标分析、代码修复、报告生成等需要自主规划、追问、状态同步和产物返回的任务。生产系统常见模式是 Agent 内部用 MCP 访问工具和资源，Agent 之间用 A2A 委派任务。反例有三个：把所有工具包装成 Agent，会制造额外状态和权限混乱；把复杂 Agent 包成一个 `do_everything` Tool，会让状态、追问、取消、Artifact 和失败语义不可见；把上游 MCP 工具权限转交给下游 Agent，会造成权限扩散。审计指标包括 A2A MCP protocol classification、tool agent boundary、autonomy fit、lifecycle placement、discovery split、context ownership、permission separation、result artifact boundary、trace linkage、version eval coverage 和 A2A MCP boundary gate。核心观点是：协议分工不是风格问题，而是权限、上下文、状态、产物和责任边界的工程控制面。

40ZQ. 跨 Agent 权限、身份、审计和信任模型怎么设计？
      考察点：user identity、agent identity、service identity、OBO、permission attenuation、context policy、redelegation、high-risk confirmation、MCP tool permission、audit / trace、trust evidence、tenant isolation。
      回答框架：先分三层身份：用户身份决定代表谁和哪个租户，Agent 身份决定哪个智能体发起或接收任务，服务身份决定实际运行实例是否可信。授权上使用 on-behalf-of，把用户 scope、Agent scope、任务 scope、数据分类和时间边界绑定在同一个授权上下文里；下游权限必须衰减，不能从聚合扩大到明细、从只读扩大到写、从摘要扩大到全文。委派前检查 caller 是否能委派、assignee 是否在 allowlist、Agent Card 是否签名、服务身份是否可信、任务类型是否匹配、上下文是否允许转发；接收时检查调用方和任务授权；下游调 MCP 工具前仍要检查用户 scope、任务 scope、租户和工具权限；结果返回前检查 Artifact、证据引用、脱敏和保留期。继续委派默认禁止，高风险动作要人工确认。Trace 用于排障，Audit 用于合规和责任链，必须记录 trace id、task id、用户、调用 Agent、接收 Agent、策略、决策、数据分类、context refs、tool call、Artifact 和时间。信任模型要区分 Agent 可信、数据可信、方法可信、结果可信和任务可信；低信任 Agent 输出需要证据、人审或 verifier。审计指标包括 identity chain coverage、OBO scope binding、delegation allowlist、permission attenuation、context policy enforcement、redelegation control、high-risk confirmation、MCP tool permission binding、result release control、audit trace completeness、trust evidence verification、tenant isolation 和 cross-agent security gate。核心观点是：多 Agent 安全不是让模型自觉，而是让身份、权限、上下文、工具、产物和审计链都能被系统强制和证明。

40ZR. 多 Agent 系统的失败模式如何检测和治理？
      考察点：delegation loop、role conflict、hallucination propagation、context drift、duplicate work、state consistency、artifact conflict、accountability trace、policy chain、collaboration fit、termination / handoff、eval coverage。
      回答框架：先说明多 Agent 失败更像分布式系统故障，不是单个回答错了，而是错误会沿委派、摘要、工具调用和产物链路传播。典型失败包括循环委派、角色冲突、假设被升级为事实、原始约束丢失、重复查询导致成本爆炸、状态事件乱序、多 Agent 抢写同一 Artifact、责任链断裂、安全策略只在第一跳生效，以及简单任务被过度拆分。治理上要把每个 root task 建成可审计对象：委派图必须无环并有最大深度，冲突要有风险优先 / 领域权威 / 人工复核，claim 要保留 type、evidence、confidence 和 limitations，root goal 与 hard constraints 每次转发都要保留，工具调用要有预算和缓存，状态事件要有 sequence number 和 reconcile，Artifact 写入要用单写者、锁或 patch 合并，trace / audit 要记录 caller、assignee、decision、evidence、policy、status 和 cost，策略链要贯穿上下文转发和最终输出，简单任务应单 Agent 处理，高风险循环或冲突要有终止和人工接管。审计指标包括 delegation loop control、role conflict arbitration、hallucination propagation containment、context drift control、duplicate work budget、state consistency、artifact conflict control、accountability trace、policy chain enforcement、collaboration fit、termination handoff readiness、failure eval coverage 和 multi-agent failure gate。核心观点是：多 Agent 可靠性不能只看最终答案，而要验证协作链内部没有循环、漂移、冲突、证据丢失和安全策略绕过。

40ZS. A2A 企业多 Agent 协作平台系统设计怎么答？
      考察点：requirements、Task API、Orchestrator、Agent Registry、A2A Runtime、Policy Engine、Context Manager、Artifact Store、Trace / Audit、Agent Card、Task、Message、Artifact、A2A/MCP boundary、state machine、failure handling、eval、scalability。
      回答框架：先澄清平台面向内部还是第三方 Agent，任务是短任务还是长任务，是否需要异步状态、MCP 工具、多租户、敏感数据、人工审批和成功指标。架构上分为 Task API、Orchestrator、Agent Registry / Agent Card Store、A2A Runtime、Policy Engine、Context Manager、Artifact Store、Trace / Audit 和 Specialist Agents。Task API 创建 root task，Orchestrator 拆解任务和选择 Agent，Registry 按 capability、domain、version、tenant、auth scope、health / SLO 召回候选，Policy Engine 在模型路由前做权限和信任过滤，A2A Runtime 负责委派、状态、事件、幂等、超时、取消和重试，Context Manager 做最小上下文、引用、脱敏和 source label，Artifact Store 存报告、表格、patch 和证据，Trace / Audit 串联任务、策略、工具、产物和成本。协议对象要覆盖 Agent Card、Task、Message、Artifact；状态机使用 submitted、working、input-required、auth-required、completed、failed、canceled、rejected，内部 queued / running / expired 可映射到协议语义。安全上区分用户身份、Agent 身份和服务身份，使用 OBO 授权和权限衰减，高风险动作人工确认。A2A 管 Agent 间任务协作，MCP 管 Agent 内部工具 / 资源 / prompt，不能把上游 MCP 工具权限转交给下游 Agent。失败处理覆盖循环检测、状态乱序、retryable error、input-required、冲突仲裁、claim verification、预算和 human handoff。评估覆盖 offline eval、trace replay、single agent eval、collaboration eval、safety eval、cost latency eval 和 regression。核心观点是：A2A 系统设计不是把 Agent 连起来，而是把发现、委派、状态、上下文、权限、产物、审计、失败恢复、评估和扩展性做成平台能力。

40ZT. Skill 是可复用能力包还是工具集合？
      考察点：manifest、task goal、instructions、tools、resources、prompts、workflow、permissions、configuration、eval、lifecycle、plugin / tool / workflow / agent boundary、progressive disclosure。
      回答框架：先给定义：Skill 是围绕某类任务目标封装的可复用能力包，不是单个工具，也不只是工具列表。一个成熟 Skill 通常包含 manifest 元信息、使用说明、工具集合、参考资源、prompt 模板、workflow、配置、权限和 eval；复杂 Skill 还要支持版本、安装、启用、禁用、升级、回滚、下架和审计。Tool 解决“能做一个动作”，例如 read_file、query_database；Skill 解决“能完成一类任务”，例如会议总结、合同审查、代码评审。Plugin 更偏扩展载体和安装入口，一个 Plugin 可以包含多个 Skill；Workflow 更偏控制流，是 Skill 的一部分；Agent 是执行主体，可以安装或调用多个 Skill。Skill 边界要适中，不能大到“万能办公助手”，也不能小到单个 API。权限上要声明读写、外部调用、数据分类和人工确认；eval 要覆盖 task success、grounding、format、safety、regression 和 cost；progressive disclosure 要让 Skill 按需加载说明、脚本、参考资料和资产，避免上下文膨胀。核心观点是：Skill 是工具生态从工程接口走向产品化能力的抽象，让能力可以被安装、授权、复用、评估和治理。

40ZU. Plugin、Action、Tool、Skill、Workflow 怎么区分？
      考察点：execution instance、callable interface、process orchestration、task capability package、extension package、permission layer、eval layer、trace / audit、lifecycle。
      回答框架：先给五层定义：Action 是一次具体执行事实，回答“这次调用发生了什么”；Tool 是 runtime 可调用接口，回答“系统能调用什么操作”；Workflow 是多步骤控制流，回答“任务按什么流程推进”；Skill 是围绕任务目标的可复用能力包，回答“某类任务能力如何被安装、调用、评估和治理”；Plugin 是扩展交付载体，回答“能力如何安装、认证、版本化和分发”。工程上不要按产品命名硬套，而要按粒度、字段和治理责任判断：Action 要有 action id、input、output、caller、status、trace id；Tool 要有 input / output schema、权限、超时和错误语义；Workflow 要有 steps、dependencies、conditions、retry、approval 和 state store；Skill 要有 manifest、task goal、instructions、tools、resources、prompts、workflow、permissions 和 eval；Plugin 要有 install、auth、version、entrypoints、dependencies 和 contained capabilities。权限分层上，Plugin 管安装授权，Skill 管任务能力授权，Workflow 管步骤审批，Tool 管操作权限，Action 管执行审计。评估分层上，Action 看执行成功，Tool 看调用质量，Workflow 看流程成功，Skill 看任务成功，Plugin 看包健康度。核心观点是：这些概念不是谁包含谁的单链条，而是执行、接口、流程、任务能力和扩展载体五个不同治理边界。

40ZV. Skill Manifest 应该怎么设计，如何判断它能不能上线？
      考察点：identity、description、capabilities、inputs、outputs、tools、resources、prompts、workflow、permissions、configuration、safety、eval、examples、version lifecycle、audit readiness。
      回答框架：先说明 Manifest 不是好看的说明文案，而是平台、Agent、管理员和 eval 系统共同消费的结构化能力声明。一个可上线 Manifest 至少要覆盖稳定 id、name、version、owner、category、status；description 要具体说明任务对象、处理动作、输出形式、证据要求和边界；capabilities 要保持任务级粒度，不能大到万能助手，也不能小到单个 API；inputs / outputs 要有必需输入、可选输入、输出类型、schema、citation 和 artifact 约束；tools 要写 purpose，resources 和 prompts 要有用途和版本；workflow 要说明步骤和高风险 approval points；permissions 要最小化并给出 reason；configuration 要有类型、默认值和枚举；safety 要声明外部分享、人工审批、禁止动作、数据分类和脱敏；eval 要有 metrics、golden sets 和 minimum quality gate；examples 要包含触发和不触发样例；version lifecycle 要支持 changelog、rollback、兼容性和下架；audit 字段要能追踪 owner、version 和 review status。核心观点是：Skill Manifest Gate 过不了时，Skill 也许能在 demo 中运行，但还不能进入可安装、可授权、可评估和可治理的平台生态。

40ZW. Skill 的发布、安装、启用、升级、禁用和回滚怎么治理？
      考察点：publish review、install approval、enable scope、permission delta、config versioning、running task policy、compatibility、canary、rollback、dependency pinning、audit、emergency suspend、uninstall retention、eval monitoring、update policy。
      回答框架：先区分四个动作：发布是进入平台目录，安装是加入租户能力集合，启用是对某个范围开放调用，使用是一次具体执行。生命周期治理要覆盖三条线：发布审核线、使用管理线和版本治理线。发布前检查 manifest、security、eval 和 owner；安装时展示 manifest 摘要、权限、数据范围、风险等级、审批人和安装记录；启用时按 tenant、team、role、agent 或任务类型控制范围；运行时仍要检查权限、配置、版本和上下文策略。升级时要比较 manifest、prompt、workflow、tool/resource 依赖、权限、配置 schema、安全策略和 eval，新增权限、扩大数据范围、改变输出契约、高风险 workflow 或外部共享策略都要重新审批。灰度要按团队、租户、Agent 或流量比例逐步放量，并监控成功率、错误率、安全、延迟、成本和用户反馈。回滚前要确认旧版本可用、配置兼容、prompt/resource pinned、迁移可逆和计划明确。禁用和紧急下架要阻止新任务、处理运行中任务、标记受影响 artifact、通知用户并保留审计证据。卸载不能删除 trace/audit 和必要 artifact。核心观点是：Skill 生命周期治理不是后台按钮，而是企业平台证明能力可控、可升级、可回滚和可追责的控制面。

40ZX. Skill Marketplace 与企业内部门户怎么设计和治理？
      考察点：catalog、search discovery、detail page、review workflow、permission transparency、install approval、rating quality balance、operations metrics、duplicate governance、admin permission view、developer console、security center、recommendation policy、lifecycle visibility、owner maintenance、audit trace。
      回答框架：先说明 Marketplace 不是 Skill 列表页，而是能力发现、安装审批、质量运营、安全治理和开发者生态的统一入口。架构上至少包含 Skill Catalog、Search & Discovery、Detail Page、Review & Approval、Installation Manager、Permission Center、Quality Dashboard、Security Center、Developer Console 和 Operation Console。Catalog 要有 id、name、version、description、category、tags、owner、status、permissions、risk level、last updated 和 visibility；搜索要结合名称、描述、标签、examples、capabilities、质量、安全、权限可用性和维护状态；详情页要展示能做什么、不适合什么、输入输出、权限 reason、版本历史、质量指标、安全状态和已知限制。上架审核覆盖 manifest、permission、security、quality 和 maintenance；安装审批按 low / medium / high / critical 风险分级，高风险必须走安全、IT 或 data owner 审批。评分不能只看用户星级或安装量，还要看 task success、eval、安全、可靠性、维护状态和留存。运营指标包括 Skill 总数、上架通过率、安装量、启用量、活跃数、成功率、平均权限风险、安全事件、更新频率、重复比例和无人维护数。管理员视图要能看见已安装 Skill、启用范围、高风险调用、外部传输、敏感数据、陈旧版本和 owner 缺失；安全中心要支持权限巡检、异常检测、数据泄露告警、供应链风险、紧急下架、漏洞通知和受影响用户通知。核心观点是：企业 Skill 门户的高分设计不是“能搜到”，而是能证明每个 Skill 可理解、可审批、可安装、可运营、可阻断、可下架和可追责。

40ZY. 工具和 Skill 的质量评估与安全审核怎么做？
      考察点：tool schema clarity、argument validation、execution reliability、output stability、side effect control、skill task quality、factual grounding、offline eval、online monitoring、least privilege、data security、prompt injection、high-risk action control、supply chain、human review、regression gate、audit trace。
      回答框架：先区分 Tool 评估和 Skill 评估：Tool 关注单个操作能否被正确调用，包括 schema 清晰度、参数校验、执行可靠性、输出稳定性、错误语义、延迟和副作用控制；Skill 关注一类任务能否安全稳定完成，包括任务成功率、事实准确性、引用准确性、完整性、格式合规、用户价值和成本。上架前要做离线 eval，覆盖 golden set、scenario set、adversarial set、regression set 和明确阈值；升级前要比较 prompt、workflow、tool schema、资源、模型版本和权限变更，任何质量或安全回退都要阻断自动发布。安全审核要覆盖最小权限、permission reason、数据分类、脱敏、保留、租户隔离、外部传输、训练使用策略、prompt injection、防止不可信内容变成指令、高风险写操作 confirmation / approval / audit、人类 reviewer、供应链依赖 pin、漏洞扫描、代码签名和网络 allowlist。上线后继续监控成功率、错误类别、延迟、成本、用户反馈、安全拦截、输出解析失败、人工接管和分布漂移。核心观点是：Quality Safety Gate 不是一个分数，而是一组硬门禁；质量高但泄露数据、会越权写入、绕过确认或无法审计的 Skill 不能进入企业 Marketplace。

40ZZ. 从单工具到可组合工作流怎么设计和审计？
      考察点：Tool Chain vs Workflow、DAG / graph、dependencies、conditions、IO contract、state、permissions、data flow、idempotency、retry、compensation、approval、cancellation、trace、eval、version compatibility、Workflow Gate。
      回答框架：先说明单 Tool 解决一个动作，Tool Chain 只是临时串联，Workflow 才是有明确步骤、依赖、状态、输入输出、失败恢复、审批和审计的任务编排。设计时先画 DAG 或 graph，检查所有依赖引用合法且无环；每个 step 要有输入输出契约、错误语义、权限 scope、timeout、retry policy 和 trace 字段；条件分支要可复现，不应写成“模型自行决定”；中间 artifact 要标注分类、转发策略、是否保存和是否可跨租户 / 跨 Agent 传递；有副作用步骤必须有 idempotency key、补偿策略、审计和必要的人审确认；取消和超时要能停止下游并让状态收敛。评估上不能只看端到端成功率，还要看 graph validity、dependency acyclicity、workflow IO contract、condition determinism、state coverage、permission boundary、data flow policy、idempotency coverage、retry safety、compensation readiness、human approval coverage、trace replay readiness、eval coverage 和 version compatibility。核心观点是：可组合 workflow 能被 Skill、MCP、A2A 和 Marketplace 复用的前提，是它通过 Workflow Gate，而不是画出一条 happy path。

40ZZA. 工具协议中的 prompt injection 怎么防御和审计？
       考察点：instruction / data separation、source trust labeling、taint propagation、policy pre-tool gate、risky tool isolation、untrusted action blocking、sensitive data control、high-risk confirmation、sandbox、output projection、RAG source boundary、multi-agent taint、trace、eval、Prompt Injection Defense Gate。
       回答框架：先说明工具协议里的 prompt injection 风险主要来自不可信网页、文档、邮件、Issue、日志、数据库字段、RAG chunk、tool result 和其他 Agent 输出，这些内容可以作为证据或数据，但不能升级为上层指令。防御要分层：消息层做 instruction / data separation；资源和工具结果带 source、trust level、taint、derived_from 和 allowed uses；上下文压缩、摘要和跨 Agent 传递时继续保留 taint；工具调用前由 runtime / policy engine 做权限、风险、敏感数据、外发目标和确认状态检查；高风险工具从普通 auto 候选集中隔离；外发、删除、shell、写生产等动作需要确认、审批、幂等和审计；浏览器、shell、代码执行和文件工具要进 sandbox；工具输出进入上下文前做投影、脱敏和引用保留；RAG 文档只能改变证据，不能改变系统策略、引用要求或工具权限。评估上看 instruction data separation、source trust labeling、taint propagation、policy pre-tool gate、risky tool isolation、untrusted action blocking、sensitive data control、high-risk confirmation、sandbox enforcement、output redaction projection、RAG source boundary、multi-agent taint propagation、trace readiness 和 regression eval coverage。核心观点是：Prompt Injection Defense Gate 通过，才说明不可信数据没有获得指令权限，也不能直接驱动高风险工具。

40ZZB. 工具输出可信度、数据来源和引用机制怎么设计？
       考察点：source metadata、trust level、freshness / version、access scope、citation binding、citation support、evidence chain、claim type、confidence、limitations、conflict resolution、RAG chunk citation、multi-agent evidence、trace replay、citation forgery block、Tool Output Trust Gate。
       回答框架：先说明工具输出不是天然事实，而是证据。设计时每条 tool result、RAG chunk、artifact、数据库结果、网页内容和跨 Agent 输出都要带 source id、URI、标题、owner、retrieved_at、data_as_of、版本、trust level、data classification、access scope 和 limitations。引用不能让模型自由编造，应该由系统生成 citation_id / source_id，绑定 source_uri、span、chunk_id 和标题；最终回答前校验 citation 是否存在、是否支持对应 claim。关键结论要拆成 fact、observation、inference、hypothesis、recommendation，分别带 confidence 和 limitations；摘要、压缩和跨 Agent 传递时保留 evidence、produced_by、derived_from 和限制说明。多个来源冲突时比较 trust level、权威 owner、版本、新鲜度和权限范围，高风险结论要保留冲突说明或升级人工确认。评估上看 source metadata coverage、trust level coverage、freshness version coverage、access scope disclosure、citation binding、citation support accuracy、evidence chain completeness、claim type calibration、confidence limitation disclosure、conflict resolution readiness、summary provenance retention、RAG chunk citation accuracy、multi-agent evidence propagation、provenance trace replay、citation forgery block 和 eval coverage。核心观点是：Tool Output Trust Gate 通过，才说明系统能证明“这个结论来自哪里、为什么可信、有什么限制”。

40ZZC. RAG、Tool、Agent、Memory 如何组合和评估？
       考察点：capability boundary、orchestration mode、context priority、context budget、evidence preservation、tool observation use、agent state、memory read relevance、memory write gate、memory scope permission、conflict resolution、injection propagation、sensitive data memory block、trace linkage、layered eval。
       回答框架：先区分职责：RAG 提供知识和引用，Tool 执行外部查询或动作，Agent 组织目标、计划、状态和失败恢复，Memory 保留长期偏好和历史状态。组合设计要先按任务选择 RAG-first、Tool-first、Memory-first、固定 Workflow 或 Agent-planned 模式；再定义上下文优先级，system policy 和用户当前任务最高，权威工具结果和 verified RAG 高于旧 Memory，外部网页和未验证 Agent 输出不能触发高风险动作；上下文预算要给当前任务、关键证据、工具 observation、相关 Memory 和 artifact 引用留空间。Memory 读要按作用域和相关性过滤，写要经过长期价值、用户确认、敏感数据、可信来源、TTL / 删除策略门禁；RAG / Tool / Memory 冲突时按权威性、新鲜度、权限和用户确认处理。安全上要阻断 RAG 注入触发工具、敏感 Tool result 写入 Memory、外部内容污染长期状态。评估不能只看最终答案，要分层看 RAG 召回和引用、Tool 调用和参数、Agent 状态和恢复、Memory 读写质量，以及 end-to-end 成功率、安全和成本。核心观点是：RAG Tool Agent Memory Integration Gate 通过，才说明这些能力不是概念堆砌，而是可追踪、可恢复、可审计的组合系统。

40ZZD. 工具调用成本、延迟和并发怎么控制？
       考察点：cost attribution、latency breakdown、budget、timeout、retry classification、idempotency、concurrency limit、queue priority、rate limit、cache isolation、batch partial success、result trimming、degradation transparency、router performance、trace、eval、Tool Performance Gate。
       回答框架：先把成本拆开：模型 token、工具 schema、工具结果 token、外部 API、检索 / rerank、数据库、浏览器 / 沙箱、Agent 多轮规划、trace、重试和人工确认；再把延迟拆成模型决策、队列等待、权限策略、工具执行、后处理和模型继续生成。生产系统要给每个任务设置 max tool calls、max rounds、max cost、max latency、max retries 和并发预算；单工具、workflow 和用户请求都要有 timeout / deadline。重试必须先做错误分类，临时网络、限流和可恢复超时可重试，权限、参数、策略拒绝和用户取消不重试；有副作用动作要有幂等键、dry run、补偿或人工确认。并发要按全局、租户、用户、工具和下游服务限额，并进入支持 priority、deadline 和 fairness 的队列。限流要尊重 retry-after，配额耗尽时排队、降级、换工具或请求确认。缓存要有 tenant / user / permission-aware key、TTL、数据分类和失效策略，不能跨用户复用权限相关结果。批处理要支持逐项权限和 partial success，大结果要分页、top-k、字段选择、摘要、截断说明和 artifact 引用。降级要向用户透明说明数据时间和限制。Tool Router 要用成本、延迟、成功率、负载、数据新鲜度和用户预算做选择。评估上看 cost attribution coverage、latency breakdown coverage、budget enforcement、timeout coverage、retry accuracy、idempotent retry safety、concurrency limit、queue fairness、rate limit handling、cache safety、batch partial success、result trimming、degradation transparency、router performance、trace readiness 和 eval coverage。核心观点是：Tool Performance Gate 通过，才说明系统不只是能调工具，而是在成本、延迟、并发和失败恢复上可控。

40ZZE. Tool-use eval benchmark 怎么设计？
       考察点：tool need、no-tool overcall、tool selection、tool set precision / recall、argument schema validity、argument semantic accuracy、argument source coverage、sequence order、observation grounding、error recovery、safe failure、safety policy、simulator determinism、trace replay、cost latency regression、benchmark slice、regression gate。
       回答框架：先说明 Tool-use eval 不能只看最终答案，要拆成“该不该调、调哪个、参数是否合法和语义正确、多工具顺序是否合理、工具结果是否被忠实使用、失败后是否恢复、安全策略是否执行”。样本结构要包含 user input、available tools、gold tool calls、pred tool calls、gold / pred args、argument source、tool results、final answer、safety policy、cost、latency、trace fields 和 slice labels。数据集要覆盖 no-tool、单工具、多工具、参数陷阱、权限错误、超时、限流、部分失败、prompt injection、高风险确认和长上下文。评分上同时看 tool need accuracy、no-tool overcall control、tool selection accuracy、tool set precision / recall、argument schema validity、argument semantic accuracy、sequence order accuracy、observation grounding、error recovery、safe failure、safety compliance、simulator determinism、trace replay coverage、cost latency regression control 和 slice coverage。工具模拟器要固定版本、可复现、可模拟错误和超时；trace replay 要能用真实线上轨迹固定工具结果后重放新模型、新 prompt 或新 schema。上线前把这些指标放进 regression gate，安全违规、trace 缺失、高风险动作未确认等必须是硬阻断。核心观点是：好的 benchmark 要告诉你工具使用到底错在“不该调、漏调、调错、参数错、顺序错、结果用错、失败不会恢复还是安全越界”。

40ZZF. Function Calling、MCP、A2A 怎么横向对比和组合？
       考察点：layer boundary、function calling fit、MCP integration fit、A2A delegation fit、object contract、capability discovery、lifecycle state、context transfer、permission governance、Host runtime、trace chain、eval gate、overengineering control。
       回答框架：先给分层结论：Function Calling 是模型到 Host 的结构化工具调用意图，解决模型如何选工具和填参数；MCP 是 Host/Client 到外部 tools / resources / prompts Server 的连接层，解决工具资源如何被发现、接入和受 Host 管理；A2A 是 Agent 到 Agent 的任务协作层，解决 Agent Card 发现、任务委派、状态同步、input-required / auth-required、Artifact 返回和跨 Agent 审计。三者不是三选一，常见组合是 Agent 内部用 Function Calling 表达工具调用，Host 把调用路由到 MCP Server；多个 Agent 之间用 A2A 委派任务。系统设计要看层级边界、对象契约、能力发现、生命周期、上下文传递、权限治理和 trace 链路。Function Calling 不负责工具安装和资源暴露，MCP 不等于模型 provider 的 tool_call 格式，A2A 不适合包装 read_file 这类简单工具。Host / runtime 是核心控制点，负责 MCP 能力投影、provider adapter、上下文预算、权限门禁、工具结果回填、A2A 上下文转发和 trace 串联。上线评估要把 tool eval、MCP 权限 / 资源 eval、A2A task eval 和 release gate 连起来。核心观点是：Function Calling 是模型调用语言，MCP 是工具资源连接层，A2A 是 Agent 协作层，真正的系统能力来自分层组合和可审计边界。

40ZZG. 主流平台工具协议迁移和适配怎么做？
       考察点：provider / framework boundary、schema projection、tool choice mapping、tool result round trip、streaming assembly、parallel call alignment、built-in tool boundary、error normalization、provider adapter isolation、framework escape hatch、RAG traceability、permission consistency、migration eval、trace replay、vendor lock-in。
       回答框架：先区分 provider 和 framework：OpenAI、Anthropic、Google 是模型 provider，提供原生 tools / tool_use / function calling 等接口；LangChain、LlamaIndex 是应用框架，负责工具封装、Agent 编排、RAG、callback / trace 和多 provider 适配。迁移不能只改 endpoint，要先定义内部 Tool Runtime 对象，包括 tool id、schema、tool choice policy、tool call、tool result、error、trace、permission 和 eval label；再写 Provider Adapter / Framework Adapter，把内部对象投影到各平台。重点检查 schema 投影是否丢 required / enum / additionalProperties，tool_choice 的 auto / none / required / force specific / allowlist 是否等价，tool result id 是否能 round trip，streaming 参数是否能完整拼接，parallel tool calls 的 result id 和合并顺序是否正确，内置工具的权限、成本和可观测性边界是否清楚。框架层要保留 escape hatch，能拿到底层 provider 原始事件；RAG-heavy 应用要保留 citation、source metadata 和 query engine trace。错误要归一化成内部 error code、category、retryable 和 raw error。权限必须由 runtime policy 执行，不能只靠 prompt。上线前跑 migration eval、trace replay、slice report 和成本延迟回归。核心观点是：多平台工具调用的关键不是字段翻译，而是让 schema、调用策略、结果回传、安全、trace 和 eval 在不同 provider / framework 下语义一致。

40ZZH. 企业 MCP 工具平台怎么设计？
       考察点：MCP Gateway、Registry、Tool Router、Policy Engine、Context Manager、MCP Server Runtime、Host / Client / Server boundary、OBO authorization、scope binding、tenant isolation、roots sandbox、prompt injection taint、trace / audit / replay、eval regression、developer portal、release lifecycle、provider adapter、HA。
       回答框架：先给总体分层：AI App / Agent / IDE Host 通过 MCP Gateway 访问企业工具能力，Gateway 做认证、租户、限流、路由、参数校验、错误归一化和 trace；Registry 维护 MCP Server、tools、resources、prompts、owner、version、scope、risk、status 和 lifecycle；Tool Router 结合能力、租户、权限、健康、版本、成本和延迟选择目标 Server；Policy Engine 用用户身份、Agent 身份、Host 身份、租户、数据分类、任务目的、确认状态和外发目标做 OBO 授权与最小权限判断；Context Manager 把 tool / resource 输出投影、脱敏、摘要、引用、artifact 化并控制上下文预算；Server Runtime 托管知识库、数据库、代码仓库、浏览器、终端和业务 API 等 MCP Server。安全上要强调 Host 拥有最终策略控制，MCP Server 暴露能力但不能自行升级权限；租户、artifact、trace、quota 和配置必须隔离；IDE、终端、浏览器和代码执行能力要有 roots、网络 allowlist、secret 隔离、shell 白名单和资源限制；不可信 tool result、网页、文档和数据库字段要带 source / trust / taint，不能直接触发高风险工具。平台治理上要有 trace、audit、replay、tool simulator、regression eval、canary、rollback、emergency disable、开发者门户、manifest lint、review console 和 provider adapter 测试。核心观点是：企业 MCP 平台不是普通 API Gateway，而是可注册、可发现、可授权、可隔离、可审计、可回放、可评估、可运营的工具和上下文接入层。

40ZZI. 跨 Agent 协作系统怎么设计？
       考察点：goal clarification、orchestrator、planner、Agent Registry、A2A Runtime、Task graph、Agent Card、Task / Message / Artifact / TaskState、context minimization、permission attenuation、MCP tool boundary、evidence grounding、conflict arbitration、delegation loop、human handoff、trace / audit / replay、baseline eval、cost latency budget、idempotency、collaboration fit。
       回答框架：先说明跨 Agent 协作不是让多个 Agent 聊天，而是让不同能力边界的 Agent 在任务图、上下文边界、权限边界和证据链下协同完成复杂目标。架构上分为入口 / Task API、Orchestrator、Planner、Agent Registry、A2A Runtime、Context Manager、Policy Engine、MCP Gateway、Artifact Store、Trace / Audit / Replay、Eval 和 Human Review Console。Planner 把 root task 拆成有依赖的 DAG，Registry 用 Agent Card 的 capability、input/output、tenant、scope、health、version、cost 和 latency 选择 Agent；A2A Runtime 管 Task、Message、Artifact 和 TaskState，内部 created / planned / assigned / running 状态要映射到 submitted、working、input-required、auth-required、completed、failed、canceled、rejected 等稳定语义。Context Manager 只给下游 Agent 最小、脱敏、有 source / trust / taint / forwarding policy 的上下文；Policy Engine 用 OBO、权限衰减、租户策略、任务 scope 和高风险确认防止权限放大；Agent 内部访问数据库、代码、浏览器或终端时走 MCP Gateway，不能把上游工具凭据转交给下游 Agent。结果聚合时要用 Artifact、source refs、claim type、confidence 和 limitations，冲突要比较证据、领域权威和新鲜度，必要时让 verifier 或人工复核。失败治理覆盖循环委派、上下文漂移、幻觉传播、状态乱序、重复工作、Artifact 抢写、预算超限和人审接管。评估上要比较单 Agent / 固定 workflow baseline，覆盖任务拆解、Agent 选择、上下文裁剪、权限、安全、证据、冲突、成本延迟和回归切片。核心观点是：跨 Agent 系统上线前必须证明它比更简单方案更合适，并且可授权、可追踪、可恢复、可评估、成本可控。

40ZZJ. 工具协议生态未来会怎么演进？
       考察点：layered protocols、Function Calling、MCP、A2A、Skill / Plugin、Policy / Audit / Eval / Marketplace、long task lifecycle、capability package、safety metadata、provenance、provider adapter、workflow runtime、behavior eval、autonomy risk tiering、natural language API boundary。
       回答框架：先说明未来不会是某个协议吞掉所有协议，而更可能是分层共存。Function Calling 是模型侧结构化调用语言，MCP 是 Host / Agent 连接 tools、resources 和 prompts 的工具资源接入层，A2A 是 Agent Card、Task、Message、Artifact 和 TaskState 组成的跨 Agent 协作层，Skill / Plugin 是可安装、可授权、可评估的能力产品化层，Policy / Audit / Eval / Marketplace 是企业治理层。演进方向包括：从单次工具调用走向长期任务执行，Task ID、状态机、暂停、恢复、取消、Artifact 和全链路 trace 会变重要；从单个 tool schema 走向能力包，tools、resources、prompts、memory scope、workflow、permissions、eval 和 owner 要一起声明；从模型中心走向 Agent 中心，但 Agent 自主权必须按风险分级，低风险可自动化，高风险要确认，极高风险要专家审核；工具安全 metadata、side effect、data classification、allowed caller、retention 和 audit 会成为平台壁垒；工具结果要有 provenance、source、query time、data version、permission scope、evidence refs 和 limitations；Marketplace 会从工具列表变成治理入口；标准化与厂商扩展会长期共存，所以要用 provider adapter 控制 schema、tool choice、streaming、parallel、built-in tools、trace 和迁移 eval；自动生成工具可以从 OpenAPI、数据库或代码库提升效率，但必须经过审核、最小权限和回归评估；Agent 和 workflow 会融合，确定性流程、审批和补偿由 workflow runtime 控制，Agent 负责理解、摘要、分析和生成；eval 会从回答质量走向行为质量。核心观点是：未来工具协议生态的竞争力不是“能接多少工具”，而是能否安全、可控、可迁移、可观测、可评估地让 Agent 使用外部能力完成任务。

40ZZK. AI Infra 总览怎么讲？
       考察点：compute accelerator、network communication、storage checkpoint、scheduler governance、training platform、inference SLO、data lineage、artifact registry、eval tracking、observability、security、cost capacity、developer self-service、AI Infra / MLOps / LLMOps / Platform Engineering boundary、algorithm infra collaboration。
       回答框架：先给定义：AI Infra 是支撑大模型数据、训练、评估、部署和在线服务稳定运行的基础设施与平台工程体系，不等于 Kubernetes、Slurm、GPU 采购、监控工具或某个 MLOps 平台。再按层展开：底层是 GPU / 加速器、网络、存储和调度，决定训不训得动、扩展效率和资源公平性；平台层是训练平台、推理平台、数据平台、模型 artifact 仓库、评估和实验追踪，决定可复现、可发布、可回滚、可评估；横向层是 metrics / logs / traces、权限审计、安全治理、成本治理、容量规划和开发者自助。和 MLOps / LLMOps 的关系是：MLOps / LLMOps 更偏生命周期和应用运维，AI Infra 更偏底座与平台能力，Platform Engineering 是把这些能力产品化给内部用户。最后用门禁指标收束：看 compute、network、storage、scheduler、training、inference、data、artifact、eval、observability、security、cost、self-service、boundary clarity 和 algorithm infra collaboration 是否达标。核心观点是：算法岗懂 AI Infra，不是为了替代 SRE，而是为了能把模型方案、训练效率、推理 SLO、成本和故障定位放到同一张工程约束表里讨论。

40ZZL. AI Infra、MLOps、LLMOps、Platform Engineering 的边界怎么讲？
       考察点：AI Infra scope、MLOps lifecycle、LLMOps application lifecycle、Platform Engineering DX、DevOps / SRE、Data Platform、Model Platform、primary owner、interface contract、artifact lineage、observability SLO、security cost governance、incident routing、handoff readiness。
       回答框架：先说明这些词有交集但不是同义词，不能按工具名死分，而要按“解决什么问题、谁主责、交付什么、用什么门禁”来分。AI Infra 主责资源和运行底座，包括 GPU 集群、网络、存储、调度、训练平台、推理平台、可观测性、安全和成本治理；MLOps 主责机器学习生命周期，包括数据版本、训练 pipeline、实验追踪、模型注册、评估、部署、监控、回滚和再训练；LLMOps 主责大模型应用生命周期，包括 prompt、RAG、Agent trace、工具调用、模型路由、LLM eval、安全策略、provider 管理和 token 成本；Platform Engineering 主责把这些能力做成内部开发者可自助使用的 Portal、CLI、SDK、模板、golden path 和审批流。DevOps / SRE 提供自动化交付、SLO、错误预算、告警和事故响应思想；Data Platform 管数据底座和血缘；Model Platform 管权重、tokenizer、adapter、量化版本、评估报告和发布状态。最后用事故路由举例：GPU 排队是 AI Infra，prompt 回归是 LLMOps，模型版本不可回滚是 MLOps / Model Platform，RAG 文档过期是 Data Platform / LLMOps，开发者不会提交训练任务是 Platform Engineering。核心观点是：边界题的高分回答不是画组织墙，而是讲清主责、协作、接口、指标和 Boundary Gate。

40ZZM. GPU、NPU、TPU 与 AI 加速器怎么选？
       考察点：peak compute、memory capacity、memory bandwidth、interconnect bandwidth、low precision、software stack、kernel library、distributed communication、training memory、KV cache、cloud / self-build、cost power capacity、portability、risk governance。
       回答框架：先说明不能只看峰值 TFLOPS，要从工作负载、模型规模、batch / 并发、context length、训练或推理目标、显存容量、显存带宽、互联、低精度支持、软件生态、成本和迁移风险综合判断。训练要算权重、梯度、optimizer state、activation 和临时 buffer，推理要算权重、KV cache、prefill / decode 吞吐和 P95/P99 延迟；长上下文和高并发常被 KV cache 卡住，decode 常被显存带宽卡住，多机训练常被互联和通信库卡住。GPU 优势是通用性、生态和调优工具成熟；TPU / NPU / ASIC 优势可能是特定矩阵单元、能效或云上规模，但要看框架、kernel、通信、profiler、供应和迁移。最后用 Accelerator Selection Gate 收束：Peak Compute Fit、Memory Capacity Fit、Memory Bandwidth Fit、Interconnect Bandwidth Fit、Low Precision Support、Software Stack Maturity、Training Memory Budget、Inference KV Cache Budget、Cost Power Capacity Awareness 和 Fallback Portability Plan 都过线，才算不是纸面选型。

40ZZN. 显存、HBM、PCIe、NVLink、NVSwitch 与带宽瓶颈怎么排查？
       考察点：VRAM capacity、HBM bandwidth、PCIe transfer、NVLink topology、NVSwitch all-to-all、inter-node network、KV cache growth、training state memory、communication volume、topology-aware parallel group、dataloader / storage I/O、checkpoint I/O、offload penalty、overlap / fusion、observability。
       回答框架：先区分容量和带宽：显存容量决定权重、梯度、optimizer state、activation、KV cache、临时 buffer 和通信 buffer 能不能放下；HBM 带宽决定这些数据能否及时喂给计算单元。再按链路排查：decode 或小 batch 算子可能是 HBM / KV cache 读写瓶颈；CPU-GPU 拷贝、offload 和数据加载可能是 PCIe 瓶颈；tensor parallel、AllReduce、AllGather 可能受 NVLink / NVSwitch 或 PCIe 拓扑影响；多机训练还要看 InfiniBand / RoCE、网卡亲和性、NCCL 路由和网络拥塞；checkpoint 和数据读取会形成存储 I/O 瓶颈。指标上看 step time 分解、HBM throughput、PCIe RX/TX、NCCL time、rank skew、effective bandwidth、checkpoint time 和 dataloader wait。最后用 Bandwidth Bottleneck Gate 收束：容量预算完整、通信和 PCIe 暴露比例可接受、拓扑分组合理、关键链路都有观测指标，才能说瓶颈被定位和治理。

40ZZO. GPU 利用率、MFU、HFU 与训练效率指标怎么讲？
       考察点：tokens/s、step time breakdown、GPU utilization、MFU、HFU、model FLOPs、hardware peak FLOPs、communication ratio、I/O ratio、checkpoint overhead、rank skew、scaling efficiency、padding waste、recompute overhead、loss correctness。
       回答框架：先说明 GPU utilization 只是入口指标，表示采样窗口内 GPU 是否有 kernel 在跑，不等于训练效率高。高分回答要按证据链展开：端到端先看 tokens/s、samples/s 和 step time；效率口径看 MFU，也就是模型有效 FLOPs / 理论峰值 FLOPs；硬件口径看 HFU，也就是硬件实际执行 FLOPs / 理论峰值 FLOPs，HFU 高但 MFU 低可能说明 padding、重计算、低效 kernel 或额外格式转换很多。然后拆 step time：dataloader、H2D、forward、backward、communication、optimizer、checkpoint、logging；分布式侧看 communication ratio、NCCL time、rank skew、scaling efficiency 和 topology；I/O 侧看 dataloader time、storage throughput、cache hit、checkpoint spike。最后强调快但 loss 错没有意义，所以训练效率门禁必须同时看 throughput、MFU/HFU、通信/I/O、扩展效率和 loss / grad / numerics 正确性。

40ZZP. 预训练、SFT、RLHF、评估、推理、RAG、Agent 的 AI Infra 任务画像怎么区分？
       考察点：workload type、resource shape、pretraining checkpoint、SFT dataset/template version、RLHF rollout、evaluation reproducibility、serving TTFT/TPOT、RAG freshness/retrieval、Agent trace/tool runtime、multimodal payload、scheduler policy、observability、cost model、artifact lineage、safety governance、task profile gate。
       回答框架：先说明 AI Infra 服务的是不同工作负载，不是抽象模型。预训练是长周期大规模训练，重点是 tokens/s、MFU、通信、数据供给、checkpoint 和容错；SFT 规模小但迭代频繁，重点是数据版本、template、adapter artifact、自动评估和可复现；RLHF/RLAIF 同时有 SFT、reward、policy、reference、rollout 和偏好数据，难点是多阶段编排和追踪；评估要管理 benchmark、模型版本、样本级输出、judge 口径和可复现性；推理是在线服务，重点是 TTFT、TPOT、p95/p99、KV cache、batching、扩缩容和单位 token 成本；RAG 还要看文档更新、embedding、索引、retrieval、rerank、权限和引用；Agent 还要看工具、状态、trace、步数预算、权限、人审和失败恢复。最后用 Task Profile Gate 收束：任务类型清楚、资源向量完整、调度策略匹配、观测指标有效、成本模型分项、artifact lineage 和安全治理过线，才算画像可用于平台设计。

40ZZQ. 如何设计和审计一个支撑大模型训练与推理的 GPU 集群？
       考察点：scale-up domain、PCIe/NUMA、NVLink/NVSwitch、GPU-NIC affinity、inter-node fabric、rack locality、oversubscription、collective communication、parallel group placement、storage/checkpoint locality、fault domain、power/cooling、resource pool isolation、topology-aware scheduling、observability topology coverage。
       回答框架：先按任务画像拆资源池：预训练需要大规模连续 GPU、高速 fabric、checkpoint 和拓扑感知调度；推理需要独立资源池、低延迟、高可用和扩缩容；SFT/评估可以用配额、批处理或可抢占资源。底层先区分 scale-up 和 scale-out：单机内看 PCIe Root Complex、NUMA、NVLink/NVSwitch、GPU 到 NIC 的亲和性和本地 NVMe；多机看 InfiniBand/RoCE/Ethernet fabric、NCCL collective、rack locality、oversubscription 和拥塞；机柜级还要看电力、散热、故障域和维护。并行组放置上，tensor parallel / expert parallel 这类高频通信优先放在同机或高速互联域内，data parallel 可以跨节点扩展但要看网络和 checkpoint。最后用 GPU Cluster Gate 收束：scale-up 域匹配、PCIe/NUMA 和 GPU-NIC 清楚、fabric 和 collective 已验证、资源池隔离、故障域和电力散热可控、调度器拓扑感知、拓扑观测指标齐全，才算不是简单堆卡。

40ZZR. InfiniBand、RoCE、以太网和 NCCL 集合通信怎么排查？
       考察点：bandwidth unit、effective bandwidth、latency/jitter、RDMA、GPUDirect RDMA、InfiniBand fabric、RoCE PFC/ECN/congestion control、Ethernet scope、AllReduce、AllGather、ReduceScatter、NCCL topology、rank straggler、packet error/retransmit、topology congestion、scheduler network locality。
       回答框架：先把网络当训练主路径，而不是数据加载辅助链路。第一步看 step time 分解和 communication ratio，确认 AllReduce、AllGather 还是 ReduceScatter 慢；第二步核对带宽单位和有效带宽，避免把 Gbps 当 GB/s；第三步看 latency、p99 jitter、rank straggler、错误包和重传；第四步检查 RDMA / GPUDirect RDMA 是否生效、GPU-NIC 亲和性是否正确；第五步区分 fabric：InfiniBand 看子网、链路和 RDMA 健康，RoCE 看 PFC/ECN/拥塞控制和低丢包，普通以太网通常只适合小规模或低通信任务；第六步看 NCCL 日志、拓扑识别、ring/tree 和预期 NIC；最后看调度是否把通信密集 group 放到同 rack / 同 fabric / 快速互联域内。用 Network Communication Gate 收束：带宽、延迟、RDMA/GDR、collective 成本、NCCL 拓扑、错误重传、拥塞路径和调度 locality 都过线，才算网络证据链完整。

40ZZS. 大模型训练平台的存储体系怎么设计和审计？
       考察点：capacity tier、throughput/IOPS、local NVMe cache、shared filesystem metadata、object store authority、data lake governance、training shard format、small file amplification、dataloader cache hit、checkpoint write/recovery、model weight load cache、artifact lineage、consistency commit/checksum、security compliance、lifecycle cost。
       回答框架：先按访问模式分层：对象存储或数据湖做权威源和长期保存，管理原始数据、清洗数据、checkpoint、模型权重和 artifact；训练前把数据离线清洗、tokenize、packing，并转成 WebDataset / Parquet / Arrow / shard，减少小文件和在线解析；训练节点用本地 NVMe 或分布式缓存提升热数据读取，记录 cache hit 和 dataloader wait；共享文件系统适合易用和多节点共享，但要重点治理 metadata、小文件和并发；checkpoint 要分片、异步、checksum、commit metadata，并做恢复演练；模型加载要做本地缓存、预热和完整性校验；artifact 要记录 dataset version、config、model、eval report 和 deployment package 血缘；最后用 Storage System Gate 收束：吞吐、IOPS、cache、checkpoint restore、一致性、安全、生命周期和成本都可审计，才算不是只把文件堆到一个目录里。

40ZZT. 大模型训练平台的 checkpoint 生命周期怎么设计和审计？
       考察点：training state、model weights、shard layout、async save、manifest、checksum、commit、restore replay、RNG / dataloader state、rank state、retention、cost、replication、security、resume SLO、failure drill、checkpoint lifecycle gate。
       回答框架：先区分模型权重 checkpoint 和训练状态 checkpoint：前者用于评估、发布和下游微调，后者用于断点续训，需要保存模型、优化器、scheduler、global step、AMP scaler、RNG、dataloader 位置、分布式 rank / partition metadata、代码版本、数据版本和训练配置。保存路径采用分片 checkpoint，每个 shard 有 manifest 记录大小、checksum、并行配置和状态；写入时先到临时前缀，所有 shard 校验成功后再标记 committed，恢复只读取 committed 版本。异步保存要证明 snapshot 一致、后台 I/O 不拖垮训练、可见阻塞下降且队列不积压。恢复侧要定期做 replay drill，验证 step、lr、loss、optimizer、sampler 和 rank state 连续。生命周期侧保留 recent、milestone、best eval、release 和临时失败版本，冷归档和跨区副本要有恢复测试，权限、加密、审计和成本归因要进入门禁。最后用 Checkpoint Lifecycle Gate 收束：对象完整、分片一致、保存/恢复 SLO、最大丢失进度、权限成本和故障演练都过线，才算 checkpoint 可靠。

40ZZU. 大模型训练和推理平台的容器镜像体系怎么设计和审计？
       考察点：image digest、base layer cache、CUDA / driver / framework compatibility、GPU runtime、dependency lock、training / serving image split、image size、cold start、build smoke test、security scan、signing、SBOM、secret exclusion、runtime hardening、registry RBAC、metadata、multi-tenant mount、NCCL / RDMA。
       回答框架：先说明容器化的目标不是“会用 Docker”，而是让训练和推理环境可复现、可分发、可调度、可审计。镜像层面要分基础镜像、训练镜像和推理镜像，基础镜像锁 OS、CUDA、cuDNN、NCCL、PyTorch 等稳定组合；业务镜像只加项目依赖和代码；推理镜像要更精简、安全、启动快。复现时必须记录 digest、driver、CUDA、Python、框架、NCCL、命令、环境变量、代码 commit 和数据版本，不能只记录 tag。GPU 容器仍依赖宿主机 driver、runtime / CDI、device plugin、driver capabilities 和设备挂载，所以要做 GPU smoke test、NCCL / RDMA smoke test 和兼容矩阵。构建流水线要有依赖锁定、SBOM、漏洞扫描、签名、registry RBAC、非 root / 非 privileged、禁止密钥和模型权重打进镜像。最后用 Container Environment Gate 收束：digest 可复现、CUDA 兼容、GPU 可见、依赖锁定、冷启动、扫描签名、metadata、多租户挂载和 NCCL / RDMA 都过线，才算容器环境可靠。

40ZZV. Kubernetes 如何管理大模型训练和推理的 GPU 资源？
       考察点：device plugin、extended resource、`nvidia.com/gpu`、request / limit、GPU Operator、MIG、time slicing、gang scheduling、fragmentation、topology-aware placement、node label、affinity、taint / toleration、ResourceQuota、training CRD、inference SLO、DCGM、Pod-GPU mapping、Pending troubleshooting。
       回答框架：先说明 Kubernetes 通过 NVIDIA device plugin 等机制把 GPU 注册成 `nvidia.com/gpu` 这类 extended resource，Pod / Job / CRD 通过 resource limit 申请整数 GPU；GPU Operator 可以帮助管理 driver、container toolkit、device plugin、DCGM exporter、MIG manager 和 GPU feature discovery。然后强调默认 K8s 不是完整 AI 调度系统：分布式训练还需要 gang scheduling，避免部分 worker 占住 GPU 但任务无法启动；还要治理 GPU 碎片化、node label / affinity / taint / toleration、NVLink / GPU-NIC / rack / storage topology、MIG / time slicing 的工作负载边界、namespace / ResourceQuota / 成本归因和多租户隔离。训练侧可以用 PyTorchJob、MPIJob、RayJob 或自研 TrainingJob CRD 表达 rank、service discovery、checkpoint 和 worker 生命周期；推理侧要关注 Deployment / StatefulSet / InferenceService、自动扩缩容、冷启动、健康检查和 SLO 指标。最后用 Kubernetes GPU Gate 收束：device plugin ready、GPU extended resource 正确、request / limit 一致、gang 就绪、碎片和拓扑可控、quota 不越界、Pod 到 GPU 监控映射清楚、Pending 原因可定位，才算 GPU 资源管理可靠。

40ZZW. 如何设计大模型训练任务调度系统？
       考察点：TrainingJob resource shape、queue、priority、quota、fair sharing、dominant share、gang scheduling、backfilling、preemption、checkpoint、fragmentation、topology-aware scheduling、cost attribution、scheduler observability、failure requeue、anti-starvation。
       回答框架：先说明训练调度不是 FIFO，也不是单纯追 GPU utilization，而是在稀缺 GPU 上平衡吞吐、等待时间、公平性、优先级、成本和任务成功率。用户提交 TrainingJob 时应声明 GPU 类型、GPU 数、每节点 GPU、网络、存储、队列、优先级、是否可抢占、预计运行时长和 checkpoint 路径。调度层用 queue 管理业务类型和团队资源，每个队列有配额、并发上限、优先级策略、借用 / 出借规则和抢占策略；优先级必须有审批和审计，配额可以允许借用空闲资源，但超额资源要可抢占。分布式训练用 gang scheduling 保证 worker 整体 admission；短任务可以 backfilling，但必须不推迟保留的大任务；抢占只抢低优先级、可抢占、最近有 checkpoint 且恢复成本可控的任务。资源选择要考虑 GPU 碎片、NVLink / NVSwitch、GPU-NIC、rack、存储 locality 和故障域。最后用 Training Scheduler Gate 收束：资源画像完整、队列策略明确、fair share 可解释、gang / backfill / preemption / checkpoint / topology / cost / observability / requeue / anti-starvation 都过线，才算训练调度系统可靠。

40ZZX. 如何设计企业 AI Infra 的多租户隔离？
       考察点：namespace、ResourceQuota、LimitRange、queue quota、identity binding、RBAC / ABAC、data access boundary、data classification lineage、NetworkPolicy、Pod Security、image supply chain、Secret Manager、logs / trace redaction、cost attribution、prod experiment separation、cross tenant sharing、audit evidence、blast radius。
       回答框架：先说明 namespace 只是入口，不是完整隔离边界。多租户隔离要分层设计：资源层用 namespace、ResourceQuota、LimitRange、queue quota、资源池、GPU 型号配额和成本预算限制租户用量；身份层把 IdP group、tenant label、namespace、service account 和 workload identity 绑定；权限层用 RBAC + ABAC 控制 submit job、read dataset、read logs、deploy model、access secret、change quota 等动作，并避免 wildcard 权限。数据层按数据分类、项目、用途和租户授权，训练数据、checkpoint、模型权重、embedding index、eval report、RAG 文档、prompt、日志和 trace 都要继承权限；restricted 数据训练出的 artifact 不能自动降级。网络层默认拒绝，按需开放租户访问、egress、生产数据 endpoint 和调试端口；运行时限制 privileged、hostPath、root、共享 GPU 和本地缓存越界。镜像和密钥层要求可信 registry、digest、扫描签名、Secret Manager、短期凭证、轮换和日志脱敏。最后用 Multi-Tenant Isolation Gate 收束：资源、身份、权限、数据、网络、运行时、镜像、密钥、日志、成本、生产/实验隔离、共享治理、审计证据和故障域都过线，才算企业 AI 平台可多租户运行。

40ZZY. 如何做企业大模型 GPU 集群容量规划？
       考察点：workload forecast、GPU-hours、GPU type mix、training queue SLO、serving peak SLO、TTFT / TPOT、token throughput、network bandwidth、storage throughput、checkpoint lifecycle、utilization headroom、N+1、growth forecast、cost budget、pool separation、quota / burst、observability feedback。
       回答框架：先从任务画像和 SLO 出发，把需求拆成预训练、SFT、RLHF / RLAIF、评估、在线推理、RAG、Agent、多模态和交互式开发。训练侧用 `monthly GPU-hours = job count * GPUs per job * hours per job * retry factor` 估算，再除以每张 GPU 每月有效小时数，并乘上峰值、故障、碎片和增长余量，得到所需 GPU 数；推理侧用峰值 QPS、P95 输入输出 token、单实例 token/s、TTFT / TPOT 和 P99 延迟目标估算实例数。然后按 GPU 型号分池：高端大显存资源给预训练、大模型 SFT 和长上下文推理，中端资源给评估、embedding、rerank 和小模型推理，低优资源给可抢占探索任务。同时规划网络和存储：网络看 collective 通信量、RDMA、GPU-NIC、rack oversubscription、重传和 rank skew；存储看训练数据读取、checkpoint 写入 / 恢复、模型权重加载、日志和 trace 增长。最后用 Cluster Capacity Planning Gate 收束：任务画像、GPU 数、GPU 型号、训练队列、推理峰值、网络、存储、生命周期、利用率余量、故障冗余、增长预测、成本、资源池隔离、quota / burst 和 observability feedback 都过线，容量计划才可靠。

40ZZZ. 如何设计一个大模型训练平台？
       考察点：TrainingJob、config validation、queue / quota、image digest、code commit、dataset lineage、launcher、checkpoint、logs / metrics / events、experiment tracking、artifact registry、failure recovery、security audit、cost attribution、self-service、state machine。
       回答框架：先说明训练平台不是脚本启动器，而是训练任务生命周期管理系统。用户通过 Portal / CLI / SDK 提交 TrainingJob，声明 owner、project、image digest、code commit、entrypoint、config version、dataset version、resources、distributed config、checkpoint、queue 和 priority。平台先做 schema、镜像、代码、数据权限、配额、checkpoint 路径和 launcher dry run 校验，再交给调度层处理 queue、quota、priority、gang scheduling、topology 和 admission；launcher 层封装 torchrun、DeepSpeed、Megatron、Ray 或 MPI 的 master、rank、world size、端口和 worker 生命周期。运行中采集 rank 日志、loss、tokens/s、step time、GPU、通信、dataloader、checkpoint 和生命周期事件；Checkpoint Manager 管保存、manifest、恢复演练和最大丢失 step；Experiment Tracker 记录参数、指标、seed、代码、数据、环境和 eval report；Artifact Registry 注册 final / best checkpoint、tokenizer、config、adapter、eval report 和权限状态。最后用 Training Platform Gate 收束：TrainingJob 契约、配置校验、资源配额、环境复现、数据血缘、分布式启动、checkpoint 恢复、可观测性、实验追踪、artifact、失败分类、权限审计、成本归因、开发者自助和状态机都过线，才算平台可靠。

40ZZZA. 如何设计和审计一个大模型训练任务提交系统？
       考察点：TrainingJob schema、required fields、image digest、code commit / diff hash、dataset version / permission、structured command、final config snapshot、distributed resource consistency、checkpoint URI、logs / metrics destination、queue / priority、quota / dry run admission、idempotency、template version、state machine、audit trace、submission gate。
       回答框架：先说明训练任务提交系统不是把命令转发到集群，而是训练平台的控制面入口。入口可以是 Portal / CLI / SDK / API，但最终都应落到统一 TrainingJob schema，字段至少包含 metadata、image、code、data、command、config、resources、distributed、checkpoint、observability、queue、admission、idempotency、template、state 和 audit。然后讲校验顺序：本地先做 schema、必填字段、命令结构和分布式 shape 快速检查；服务端做镜像 digest / 安全扫描、代码 commit / diff hash、数据版本权限、ResourceQuota、PriorityClass / queue policy、checkpoint output 可写、final config snapshot 和 admission dry run。可靠性上要支持 client_request_id 幂等，网络重试返回同一个 job id；模板必须版本化，不能用 latest；状态机要区分 validating、rejected、queued、scheduled、starting、running、failed、retrying 和 completed；审计 trace 要记录 actor、validator version、final spec hash 和 admission decision。最后用 Training Submission Gate 收束：必填字段覆盖、不可变输入绑定、重复提交率为 0、分布式资源一致、队列权限和审计 trace 全部过线，才算提交系统可靠。

40ZZZB. 如何设计和审计分布式训练启动器？
       考察点：launcher interface、torchrun、DeepSpeed、Megatron、Ray、rank、local rank、world size、rendezvous、GPU binding、NCCL、logs、failure stage、elastic training、checkpoint、scheduler handoff、launcher audit trace、launcher gate。
       回答框架：先说明分布式启动器不是多开几个进程，而是把调度系统分配的节点、GPU、网络和挂载信息转换成训练框架能理解的进程拓扑和通信环境。平台应提供统一 launcher interface，由 torchrun、DeepSpeed、Megatron、Ray 或 MPI adapter 生成实际命令和环境变量。基础校验包括 `world_size = node_count * nproc_per_node`、rank 连续唯一、local rank 与 GPU 绑定一致、master address / port 可达、`CUDA_VISIBLE_DEVICES` 和 NCCL 网卡配置正确。框架层要分别检查 torchrun 的 `nnodes` / `nproc_per_node` / `node_rank`，DeepSpeed config / hostfile / ZeRO 配置，Megatron 的 tensor parallel、pipeline parallel、data parallel 和 global batch 公式，Ray 的 worker 数、每 worker GPU 和 scaling config。运行治理上要采集所有 rank 日志，区分 preflight、container start、distributed init 和 training runtime 失败；elastic training 必须绑定 checkpoint、数据重分片和恢复策略；调度系统要把 node list、node rank 和 GPU 绑定完整交给 launcher。最后用 Distributed Launcher Gate 收束：资源、rank、rendezvous、GPU、adapter、框架配置、网络、日志、失败恢复、checkpoint、调度交接和 audit trace 全部可见，才算训练平台真正支持分布式启动。

40ZZZC. 如何设计和审计大模型训练配置管理？
       考察点：config source order、final config snapshot、field source、config hash、schema validation、batch parallel consistency、immutable binding、environment capture、seed boundary、experiment tracking、hyperparameter sweep、checkpoint resume、release config、permission approval、config diff、audit trace、config gate。
       回答框架：先说明训练配置管理不是保存 YAML，而是冻结一次训练真正生效的全部条件。配置可能来自 default、template、YAML、CLI override、environment、配置中心和 platform policy，所以平台必须定义覆盖顺序，在启动前生成 final config snapshot，记录每个字段的值、来源、merge rule version 和 config hash。校验分四层：schema 校验字段类型、枚举、范围和必填项；一致性校验 global batch、world size、tensor / pipeline / data parallel、checkpoint resume 和 tokenizer；资源校验 batch、sequence length、precision、GPU 型号和显存；策略校验高敏数据、生产资源池、高优先级、关闭日志、禁用 checkpoint 和未审核镜像。复现上要绑定 image digest、code commit、dataset version、tokenizer version、Python / CUDA / PyTorch / NCCL / driver、随机种子和非确定性边界，并写入 experiment tracking 的 run。恢复 checkpoint 时按 immutable、resumable-compatible、runtime-only 三类字段检查模型结构、optimizer、scheduler、RNG 和 dataloader cursor；实验对比要提供 config diff；模型发布要把 tokenizer、generation config、eval config 和 serving config 与训练配置关联。最后用 Training Config Gate 收束：最终配置、版本绑定、校验、复现证据、resume 兼容、审批、diff 和 audit trace 全部过线，配置管理才可靠。

40ZZZD. 如何设计和审计大模型训练可观测性？
       考察点：structured logs、rank log coverage、metric taxonomy、metric dimensions、events、state machine、attempt / retry、trace spans、dashboard、alerts、anomaly detection、experiment tracking linkage、retention cost、privacy redaction、cost attribution、observability gate。
       回答框架：先说明训练可观测性的目标不是多收日志，而是让平台回答任务在哪、是否健康、为什么慢、为什么失败、是否可恢复、成本花在哪里。设计上把 signals 分成 logs、metrics、events、traces 和 experiment tracking：logs 必须结构化，带 job、attempt、node、pod、rank、timestamp、level 和 component；rank 日志要能覆盖所有 rank，并能定位第一个失败 rank 和慢 rank；metrics 分模型、性能、资源和平台四类，维度至少支持 job、attempt、rank、node、GPU、step 和 timestamp；events 记录 submitted、validated、queued、scheduled、started、checkpoint saved、worker failed、retry started、completed；状态机要保证合法迁移，attempt_id 要区分多次运行和 checkpoint 恢复。Trace 要覆盖 submit、validate、queue、schedule、image pull、startup、distributed init、training、checkpoint、eval 和 artifact registration，Dashboard 要能从 loss 下钻到 step time、GPU、通信、dataloader、checkpoint、rank skew 和成本。治理上补 loss NaN、GPU 低利用、step time spike、checkpoint stale、worker failed 等告警，配合滑动窗口或同类任务基线做异常检测；experiment run 要能跳到日志、指标、checkpoint 和 eval report；日志要做 RBAC、脱敏、导出审计和留存策略；成本要归到 owner、project、queue 和 run。最后用 Training Observability Gate 收束：日志上下文、rank 覆盖、指标维度、事件状态机、trace 阶段、告警、脱敏、留存和成本归因都过线，才算训练平台可排障、可复盘、可治理。

41. 多模态模型 demo 和多模态产品差在哪里？
    考察点：真实输入分布、input quality、evidence、privacy、copyright、latency、cost。
    回答框架：demo 常用清晰图片、干净语音或高质量生成样例，产品要面对模糊、遮挡、噪声、小字、长视频、文档版面、隐私、授权和安全审核。多模态产品要用真实输入集评估任务成功、输入质量、证据支持、OCR / ASR、生成采纳、安全隐私版权、延迟成本和反馈闭环。

42. 多模态产品为什么要单独看输入质量？
    考察点：blur、noise、occlusion、OCR、ASR、reupload、human review。
    回答框架：多模态失败经常来自输入本身，例如图片模糊、光照差、小字被压缩、语音噪声、多说话人、视频关键帧缺失。系统应先判断 input quality pass rate，不足时引导重新拍摄、裁剪、重新上传、澄清或转人工，而不是让模型自信猜测。

43. 多模态理解产品如何评估答案是否可信？
    考察点：grounded claims、OCR/ASR、引用、人工复核、切片。
    回答框架：把回答拆成关键断言，检查每个断言是否被图片、OCR、ASR、视频帧、文档页、表格或引用证据支持；同时按 OCR、小目标、图表、噪声音频、长视频和高风险行业切片看 bad case。回答流畅但媒体证据不支持，不能算可信。

44. 图像 / 视频生成产品应该看哪些产品指标？
    考察点：adoption、edit、controllability、brand、copyright、safety。
    回答框架：除了主观质量和人工偏好，还要看 generation adoption rate、编辑次数、品牌一致性、控制条件遵循、版权通过率、安全审核通过率、单位生成成本、P95 延迟和用户是否把结果用于真实草稿。生成产品通常先做辅助创意和分镜，而不是直接替代最终交付。

45. 语音助手产品上线前应该看什么？
    考察点：ASR、TTS、TTFA、barge-in、privacy、confirmation。
    回答框架：至少看 ASR WER / CER、专有名词召回、噪声和多人说话切片、首字延迟、打断处理、TTS 自然度、隐私脱敏、录音保存策略、高风险动作复述确认和人工接管。语音链路有 ASR -> LLM -> TTS 误差传播，不能只看最终回复是否自然。

46. 多模态产品上线门禁应该包含什么？
    考察点：task、quality、evidence、OCR/ASR、adoption、safety、privacy、copyright、latency、cost。
    回答框架：门禁至少包含任务成功率、输入质量通过率、证据支持率、OCR / ASR 质量、生成采纳率、安全通过率、隐私通过率、版权通过率、人审覆盖、上下文预算、P95 延迟、单位成本、eval ready、feedback loop 和 business metric。任何高风险隐私、版权或不可信媒体指令不过线，都应先降级或限制场景。

47. 大模型产品的隐私治理门禁应该包含什么？
    考察点：data flow、minimization、PII、sensitive data、permission、log、consent、retention、audit、DPA。
    回答框架：先画清数据流，再设置硬门禁：数据最小化率、PII 脱敏覆盖率、敏感数据阻断率、越权访问率、日志脱敏覆盖率、训练使用授权覆盖率、保留合规率、删除 SLA 通过率、外部传输审批覆盖率、审计覆盖率、DPA ready、incident response ready 和 business metric。隐私治理不能只靠提示词或免责声明，严重越权、无授权训练、原始日志泄露或外部传输未审批都应阻断上线。

48. 企业客户问“我们的数据会不会被拿去训练”，你怎么回答？
    考察点：contract、data control、training consent、lineage、deletion、third-party。
    回答框架：不能只口头保证，要说明产品条款、控制面和数据 pipeline 是否一致：哪些数据会被记录，是否默认进入训练或评估，训练候选数据是否有明确授权和脱敏，是否能追踪来源和删除，是否会传给第三方模型或标注方，以及审计日志如何证明这些承诺被执行。

49. 企业 RAG / Agent 为什么需要把日志治理当成隐私能力？
    考察点：raw log、tool args、retrieved docs、retention、access control、audit。
    回答框架：日志里可能包含用户输入、检索文档、工具参数、模型输出、错误栈和人工反馈，本身就是敏感数据集合。日志治理要做脱敏、截断、摘要化、加密、访问控制、保留周期、删除覆盖和审计查询；否则即使模型输出安全，排障日志也可能成为泄露通道。

50. 如何设计大模型产品上线运营与反馈闭环审计？
    考察点：canary、online monitoring、feedback action、bad case triage、regression、trace、rollback、ops gate。
    回答框架：先按灰度流量记录任务成功率、采纳率、P95/P99 延迟、单位成本、安全事故和业务指标；再把用户反馈、人工审核和失败样本转成可复现 bad case、评估集和回归报告；随后检查回归通过率、切片覆盖率、trace 覆盖率、事故处理覆盖率和回滚预案。只有质量、反馈行动、回归、成本、延迟、安全、trace 和 rollback 都过门禁，才适合扩大灰度。

51. 如何复盘一轮技术产品类大模型面试？
    考察点：topic coverage、formula coverage、demo evidence、risk、trade-off、weak questions。
    回答框架：把每道题拆成用户价值、技术方案、指标公式、demo 或项目证据、风险边界和 trade-off 六部分；统计是否覆盖 demo vs product、ROI、RAG、Agent、多模态、隐私治理和上线运营；检查是否能写出 monthly benefit、unit cost、ROI、retrieval recall、citation accuracy、task success、ops gate 等公式；最后列出弱题和修复计划。复盘目标不是背更多术语，而是让回答有可验证证据。

52. 如何设计工具 / Skill 生态的开发者体验和文档治理？
    考察点：quickstart、scaffold、schema、examples、local debug、trace、review feedback、security docs、migration、DX gate。
    回答框架：先说明 DX 不是文档美化，而是质量治理前置条件；再把开发者路径拆成快速开始、脚手架、schema 规范、好坏示例、本地调试、模拟模型调用、trace / replay、lint、SDK / CLI、审核反馈和迁移文档；最后用 DX documentation gate 检查 quickstart path clarity、scaffold completeness、documentation example coverage、local debug readiness、trace replay readiness、review feedback actionability、security documentation coverage 和 migration documentation coverage。

## 模拟面试与复盘

1. 请用 3 分钟介绍你自己，并说明为什么适合大模型算法岗。
   考察点：背景总结、岗位匹配、表达结构、重点选择。
   回答框架：背景、相关经历、核心能力、短板补齐、岗位动机。

2. 讲一个你最熟悉的大模型相关项目。
   考察点：问题定义、数据、方法、实验、失败分析。
   回答框架：先讲目标和约束，再讲方案、结果、失败案例和下一步改进。

2A. 讲一次真实大模型项目事故或异常排查经历。
    考察点：边界定位、证据覆盖、最小复现、baseline、根因、修复、复盘。
    回答框架：按“背景 -> 现象 -> 影响范围 -> 初步假设 -> 证据链 -> 最小复现 -> 根因 -> 修复与回滚 -> 复盘沉淀”讲。重点不是渲染事故，而是证明你会先看数据、日志、trace、评估样例和版本变更，用 evidence coverage、root cause rate、postmortem completeness 这类口径把经验沉淀成 checklist 和回归样本。

2B. 如何管理一个跨团队大模型项目，避免 demo 很好但上线失败？
    考察点：目标清晰度、指标树、baseline、版本 trace、RACI、变更控制、灰度回滚、bad case 回归、复盘行动项。
    回答框架：先把业务问题写成用户、场景、主指标、护栏指标和失败边界，再冻结 baseline 和评估集。实验阶段记录 hypothesis、数据、模型、prompt、检索、工具、配置和 bad case；上线阶段用灰度、feature flag、监控、回滚和降级控制风险；事故后用 project collaboration gate 检查 objective clarity、metric tree、baseline coverage、experiment reproducibility、version trace、RACI、change control、risk escalation、rollout、rollback、postmortem 和 action item closure。

2C. 如何把一次失败实验或线上事故复盘成可展示的工程能力？
    考察点：hypothesis、baseline、版本 trace、分桶分析、证据链、时间线、影响量化、根因、止损、预防、行动项关闭、回归验证。
    回答框架：先区分实验复盘和事故报告。实验复盘要讲 hypothesis、baseline、controlled change、分桶指标、bad case taxonomy 和 decision；事故报告要讲 impact、timeline、mitigation、direct cause、system root cause 和 prevention。最后用 retrospective gate 检查 hypothesis coverage、baseline coverage、version trace、slice analysis、evidence coverage、timeline coverage、root cause rate、action item closure、regression verification rate、change failure rate 和 recovery time，证明这次失败已经沉淀为回归样本、监控告警、上线门禁或团队规范。

2D. 如何把一个大模型项目讲出资深工程师的可信度？
    考察点：项目证据、STAR / 复盘、trade-off、专家追问、防夸大。
    回答框架：先用目标、baseline、指标、实验、失败、debug、上线、安全、成本和复盘组成项目证据链；行为面用 STAR 加 Reflection，但 Reflection 要落到评估集、门禁、版本治理、监控或 action item 关闭；trade-off 要同时讲收益、代价、最终决策和验证方式；专家追问时回到 baseline、分桶指标、bad case、上线边界和证据边界。可以用 practitioner interview gate 检查 project evidence score、trade-off depth、STAR reflection score、expert follow-up readiness、red flags 和 weak questions，而不是只看表达是否流畅。

3. 面试官连续追问你不确定的问题时，你怎么回答？
   考察点：诚实性、推理能力、边界意识。
   回答框架：承认不确定，回到机制分析，给出可能假设和验证方法，不硬编。

4. 如何复盘一次失败的面试？
   考察点：错误分类、知识补齐、表达修正、下一轮验证。
   回答框架：记录原题、原回答、错误原因、正确答案、背后概念和下一次改法。

5. 如何判断自己是否准备好面试？
   考察点：自测体系、mock interview、项目深挖、开放题表达。
   回答框架：能稳定讲核心概念、项目、系统 trade-off 和开放研究题，并能应对追问。

6. 如何复盘一轮 reasoning 专题 mock interview？
   考察点：topic coverage、formula coverage、demo coverage、risk coverage、trade-off coverage、revision plan。
   回答框架：先把题目按 CoT、self-consistency、verifier、process supervision、search、test-time compute、math / code training、eval、safety 和 system design 分类；再检查每题是否覆盖目标、机制、公式、demo、评估和风险；最后把弱题绑定到一个公式、一个 toy demo、一个失败案例和下一轮回答模板，而不是只说“继续复习”。
