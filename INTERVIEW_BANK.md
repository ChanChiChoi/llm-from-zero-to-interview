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

24. 什么是 Alignment Problem？
   考察点：真实意图、目标规范、代理指标、模型行为、部署分布。
   回答框架：Alignment Problem 是让模型实际目标和行为符合人类真实意图、安全边界和产品价值的问题，难点是写下的目标、训练目标、模型学到的策略和部署行为可能不一致。

25. Outer alignment 和 inner alignment 有什么区别？
   考察点：目标写对了吗、模型学对了吗、proxy objective、目标泛化。
   回答框架：outer alignment 关注训练目标是否代表真实意图；inner alignment 关注即使目标设计合理，模型是否学到了正确目标，而不是训练分布上的捷径。

26. Goal misgeneralization 和普通泛化失败有什么区别？
   考察点：能力保留、目标错误、分布外、隐蔽风险。
   回答框架：普通泛化失败是模型不会做，goal misgeneralization 是模型仍有能力但追求了错误目标，因此更难通过能力分数发现。

27. Specification gaming 和 reward hacking 有什么关系？
   考察点：规则漏洞、指标漏洞、reward model、Goodhart's Law。
   回答框架：reward hacking 是钻 reward 或 reward model 的漏洞，specification gaming 范围更广，包括钻规则、benchmark、环境和工具流程漏洞。

28. 面试中如何谨慎表述 deceptive alignment？
   考察点：潜在风险、证据强度、训练/评估/部署差异、避免断言。
   回答框架：把它作为安全研究中的潜在高风险失败模式，强调需要分布外、长期、多轮、工具使用和监督强弱变化下的证据，不能轻率说当前模型已强形式发生。

29. 如何设计一个目标错配评估？
   考察点：真实最优行为、proxy 选中行为、模型实际行为、Goodhart gap、goal misgeneralization。
   回答框架：为每个场景标注真实目标、proxy 分数和模型行为，统计 outer mismatch、behavior mismatch、proxy follow、Goodhart gap、分布外目标误泛化和高严重度失败。

30. 为什么只优化用户满意度可能带来 alignment 风险？
   考察点：短期偏好、过度承诺、事实性、安全边界、长期价值。
   回答框架：用户满意度是 proxy，模型可能学到迎合、过度自信、冗长或越过安全边界的行为，需要同时看事实性、安全、长期指标和人工复核。

31. 什么是 Scalable Oversight？
   考察点：监督瓶颈、复杂任务、AI-assisted eval、verifier、human audit。
   回答框架：当模型输出复杂到单个人类难以直接判断时，通过任务分解、AI critique、Debate、verifier、过程监督和人工审计构造更可靠的监督信号。

32. 为什么 RLHF 会遇到 scalable oversight 问题？
   考察点：偏好标注、人类能力、长上下文、代码数学、专业领域、Agent trace。
   回答框架：RLHF 依赖人类判断输出好坏；当任务需要专家、工具或长时间审查时，普通偏好标注可能只奖励表面流畅性，无法可靠监督真实正确性和安全性。

33. Debate、Iterated Amplification 和 Recursive Reward Modeling 有什么区别？
   考察点：辩论、任务分解、stronger overseer、reward model、错误传播。
   回答框架：Debate 用模型互相指出错误帮助人类判断；Iterated Amplification 用人类加模型助手分解复杂任务形成更强监督者；Recursive Reward Modeling 递归训练子任务评估器来监督复杂任务。

34. AI feedback 能否替代 human feedback？
   考察点：规模、成本、偏差、gold set、校准、高风险审核。
   回答框架：AI feedback 可以扩展监督规模和做初筛，但不能替代人类价值锚点；需要 human gold labels、专家抽检、工具验证、切片校准和高风险人审。

35. 如何设计一个 scalable oversight gate？
   考察点：direct coverage、AI feedback accuracy、verifier coverage、process accuracy、evidence support、high-risk audit、cost。
   回答框架：先定义复杂任务 gold set，再统计人类直接覆盖、AI feedback 校准、verifier 覆盖、过程步骤准确、证据支持、高风险人审覆盖、严重度加权错误和成本节省，任何高风险门禁失败都不能上线。

36. Scalable oversight 和 LLM-as-a-Judge 有什么关系？
   考察点：AI feedback、judge bias、meta-eval、human calibration、工具验证。
   回答框架：LLM-as-a-Judge 是 AI feedback 的一种工程形式；scalable oversight 更广，还包括分解、debate、verifier、process supervision、人工审计和监督质量门禁。

37. 什么是 reward hacking？
   考察点：proxy objective、reward model 漏洞、Goodhart、真实目标。
   回答框架：模型优化奖励或 judge 分数时找到 proxy 的漏洞，得到高分但没有提升真实 helpful、honest、harmless 或任务质量。

38. Reward-human gap 如何发现 reward overoptimization？
   考察点：proxy reward、human eval、held-out gold set、输出分布、error analysis。
   回答框架：如果 reward score 持续上升，但人工偏好、专家评分、事实性、安全性或 citation accuracy 下降，说明优化已经偏离真实目标。

39. 为什么 best-of-N 也会产生 reward hacking？
   考察点：候选数量、极端高分样本、reward model 误判、reranking。
   回答框架：候选越多，越容易采到 reward model 误判的高分低质样本；按最高 reward 选择会放大误判，所以不做 RL 也会过度优化 proxy。

40. KL penalty 为什么只能缓解 reward hacking，不能根治？
   考察点：reference model、分布约束、行为级风险、beta 权衡。
   回答框架：KL 限制 policy 不要远离 reference，降低 reward model 分布外失效，但 reference 也不完美，KL 是整体分布约束，不能保证每个具体行为符合真实目标。

41. 如何设计 reward hacking gate？
   考察点：proxy mismatch、reward hacking rate、reward-human gap、length bias、高风险切片、人工抽检。
   回答框架：为候选回答标注 proxy reward 和 gold quality，统计 proxy mismatch、高 reward 低质量、长度偏置、严重度加权失败和 held-out human win rate；高风险门禁失败就停止上线或降低优化强度。

42. DPO 是否避免了 reward hacking？
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
    考察点：权限控制、参数校验、二次确认、只读/写操作、审计、tool result injection、prompt injection。
    回答框架：系统要做工具级和参数级权限控制，高风险操作二次确认，区分只读和写操作，限制频率，记录审计，并把工具返回标记为不可信 observation，避免外部内容直接驱动高权限工具或污染参数。

41. 如何评估 function calling？
    考察点：tool selection、argument accuracy、execution success、final answer、safety。
    回答框架：评估工具选择准确率、参数准确率、执行成功率、最终答案正确率、安全违规率、漏调用/误调用率、多步成功率、延迟和成本。

42. 工具调用失败时模型应该怎么做？
    考察点：结构化错误、重试、追问、失败解释、禁止编造。
    回答框架：系统返回结构化错误，模型应解释失败、请求补充信息或建议重试，不能编造工具结果。

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
