# 第 17 章 DeepSeek-R1 的 RLVR 与蒸馏路线：让可验证结果塑造推理

## 17.1 为什么答案可检查会改变训练方式

传统监督微调把题目和答案写成样本 $(x,y)$，模型学习在给定输入后复现答案。这个方法适合语言、格式和基本任务行为，但高质量长推理轨迹需要人工或教师模型逐步写出，而且同一道题可能有很多条都正确的解法。把其中一条解法当成唯一标准，容易把答案正确和必须沿着这条路径作答混为一谈。

数学、程序和结构化任务提供了另一条路。数学答案可以代回方程或交给符号程序，代码可以在隔离环境中编译和测试，JSON 可以按 schema 校验。外部 verifier 不必逐字教模型应该怎么想，而是判断生成结果是否满足可执行约束。用这种结果训练模型，就是 RLVR（Reinforcement Learning with Verifiable Rewards）的核心直觉。

这并不意味着接上一个 verifier 就自动得到可靠推理。verifier 只定义了它能观察到的正确性；模型会优化这个定义，可能利用测试缺口、格式漏洞或奖励函数的盲区。因此 RLVR 的真正对象不是一个抽象的奖励分数，而是一套接受域、拒绝域、未知状态、资源限制和独立评估共同组成的训练契约。

DeepSeek-R1 的公开技术报告和官方仓库让这条路线受到广泛关注，但公开材料和社区猜测必须分开。公开资料可以支持 R1-Zero 直接在基础模型上进行大规模强化学习、R1 在强化学习前加入 cold-start 数据、公开的蒸馏模型和报告中的实验结果；它不能自动证明未披露的数据配比、每个奖励权重、内部筛选规则或闭源产品的实现方式。

## 17.2 RLVR 的最小数学对象

给定问题 $x$，策略模型生成一条轨迹 $\tau$。轨迹可以只是文本答案，也可以包含思考 token、工具调用、观察、代码补丁和最终结果。verifier 读取问题、轨迹和执行环境，返回一个结构化判定：

```text
verdict: correct | incorrect | unknown
score: [0, 1]
reason_code: exact_match | symbolic_check | test_pass | parse_error | timeout
evidence: ...
verifier_revision: ...
```

最简单的结果奖励是：

```math
r_{\mathrm{out}}(x,\tau)
=\mathbf{1}[\mathrm{Verify}(x,\tau)=\mathrm{correct}]
```

这里 $\mathrm{Verify}$ 必须是已定义版本的判定程序，输出是有限且可追踪的；如果 verifier 返回 unknown，上式不能擅自把它解释成正确或错误。$r_{\mathrm{out}}$ 取值为 0 或 1 只是教学构造，生产系统可以使用连续分数，但仍要说明分数是概率、排序分数还是业务效用。

把轨迹空间分成三个集合更容易理解奖励边界：

```math
\mathcal{A}_x=\{\tau:\mathrm{Verify}(x,\tau)=\mathrm{correct}\}
```

```math
\mathcal{I}_x=\{\tau:\mathrm{Verify}(x,\tau)=\mathrm{incorrect}\}
```

```math
\mathcal{U}_x=\{\tau:\mathrm{Verify}(x,\tau)=\mathrm{unknown}\}
```

这三个集合只在 verifier 契约明确规定互斥且覆盖当前输入时构成完整划分。超时、资源耗尽、解析失败和环境崩溃可能属于 unknown，也可能按任务协议进入 incorrect；选择不同，训练目标就不同。重要的是不能把没有证据伪装成答案错误，也不能把解析失败伪装成模型推理失败。

策略的简化目标可以写成：

```math
J(\theta)
=\mathbb{E}_{\tau\sim\pi_\theta(\cdot\mid x)}
 [R(x,\tau)]
-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
```

$\pi_\theta$ 是待训练策略，$\pi_{\mathrm{ref}}$ 是参考策略，$R$ 是由 verifier 和其他约束组成的总奖励，$\beta\ge0$ 是 KL 惩罚系数。这里是强化学习目标的抽象，不是某个公开项目的完整训练代码；实际系统还需要处理 token mask、采样批次、截断、优势估计、优化器和分布式 rollout。

## 17.3 R1-Zero、R1 与公开证据

DeepSeek-R1 的公开仓库把两条路线区分得很清楚。R1-Zero 在基础模型上直接进行大规模 RL，没有把人工标注的长推理 SFT 作为前置步骤；公开报告将其作为观察“可验证奖励能够诱导出某些推理行为”的研究路线。报告描述的现象包括较长的推理、反思和自我验证，但这些现象不等于所有任务都获得了稳定的可靠性。

R1 则加入了 cold-start 数据，并将强化学习、监督微调和面向不同能力的训练阶段组织成更完整的流水线。这样做的动机不仅是提高可验证任务分数，也包括改善可读性、语言一致性和通用任务行为。官方仓库还公开了基于 R1 生成数据训练的多个 dense distilled checkpoint。

这条公开路线至少说明了三件事。第一，可验证任务提供了比开放偏好更稳定的自动反馈。第二，直接 RL 可能探索出训练数据中没有逐步示范的行为，但探索会伴随重复、格式和迁移问题。第三，大模型发现的有效轨迹可以成为小模型的训练数据，但蒸馏后的能力不应被当成教师内部状态的原样复制。

它不能说明三件事。第一，所有推理模型都使用相同的 RLVR 配方；第二，某个产品暴露了 reasoning token 就必然执行了相同的搜索或 verifier；第三，训练报告中的 benchmark 提升会自动迁移到开放事实、工具权限和高风险生产任务。模型报告、代码仓库、论文实验和目标系统复测是不同的证据等级。

## 17.4 结果奖励与过程奖励

结果奖励只在轨迹末端判断成功，信号稀疏，却容易定义。过程奖励在中间步骤上提供反馈，可能帮助模型更早发现错误，但每个步骤的正确往往依赖上下文，标注和自动验证都更难。

设轨迹有 $T$ 个经过定义的步骤，最终奖励为 $R_T$，第 $t$ 步的过程奖励为 $r_t$，则教学上的混合奖励可以写成：

```math
R(\tau)
=\alpha R_T
 +\sum_{t=1}^{T}\gamma^{t-1}r_t
```

$T$ 是正整数，$\alpha$ 是最终结果奖励权重，$\gamma$ 通常满足 $0\le\gamma\le1$，过程奖励和最终奖励必须使用可比较的尺度。轨迹没有可验证步骤时，过程项不是 0 分的同义词，而是缺失或不适用；如果强行补零，模型会把未标注当成步骤错误。

结果奖励的优点是允许多种解法。模型只要进入接受域，就不必逐字复现教师路径。它的缺点是信用分配困难：前面哪一步导致失败，最终的一个 0 分不会直接说明。过程奖励能缓解这个问题，却带来新的 reward hacking 风险：模型可能学习写出符合步骤分类器的句子，而不是让步骤真正推进到正确答案。

工程上常见的组合是：结果 verifier 作为最终约束，过程 verifier 或规则作为辅助信号，独立测试和反例用来检查两者是否一致。过程分数永远不应替代最终任务验收，尤其是代码、外部工具和有副作用的任务。

## 17.5 Verifier contract：训练前先定义接受什么

一个可用的 verifier 不只是一个返回 0 或 1 的函数。它需要回答以下问题：

数学答案是否允许等价表达式、不同变量顺序和数值容差？代码答案是否必须在离线沙箱运行，依赖是否固定，隐藏测试是否存在？结构化答案的额外字段是被忽略、拒绝还是进入未知状态？超时是错误、未知还是需要人工复核？不同版本的测试结果能否直接比较？

可以把 verifier 输出抽象为：

```json
{
  "verdict": "correct",
  "score": 1.0,
  "reason_code": "test_pass",
  "evidence": ["hidden-test-03", "stdout-sha256:..."],
  "resource": {"wall_ms": 218, "cpu_ms": 190},
  "verifier_revision": "code-v17"
}
```

verdict 表示离散状态，score 必须说明是判定分数还是校准概率，reason_code 用于区分答案错误、格式错误、超时和 verifier 故障，evidence 让结论可复核，资源字段防止无限运行，revision 绑定测试和规则版本。

unknown 是训练系统必须认真保留的状态。例如代码执行器收到请求后连接断开，系统不知道代码是否已经运行；将其统一当作错误可能鼓励模型极短地输出，将其统一当作正确则会奖励空结果。对有外部副作用的工具，unknown 还关系到是否允许重试，不能只为方便计算把它压成一个数字。

verifier 还需要规定拒绝域之外的输入。空答案、非有限数、过长输出、非法工具参数、越权文件访问和资源耗尽都应有明确 reason code。否则模型可能通过触发解析异常或消耗资源来改变奖励分布。

## 17.6 奖励不只是正确加一分

真实训练往往同时考虑结果、格式、投机行为和安全约束。可以写成一个教学抽象：

```math
R(\tau)
=r_{\mathrm{result}}(\tau)
 +\alpha r_{\mathrm{format}}(\tau)
 -\beta r_{\mathrm{hack}}(\tau)
 -\gamma r_{\mathrm{unsafe}}(\tau)
```

$r_{\mathrm{result}}$ 表示可验证任务结果，$r_{\mathrm{format}}$ 表示协议是否可解析，$r_{\mathrm{hack}}$ 表示已检测的奖励投机，$r_{\mathrm{unsafe}}$ 表示不允许的工具或数据行为。每个项都应规定有限范围和测量方式，$\alpha,\beta,\gamma\ge0$ 是权重。若安全违规是硬性不可接受事件，不能依靠增大 $r_{\mathrm{result}}$ 抵消它，而应直接拒绝该轨迹。

格式奖励有用，但不等于内容正确。一个 JSON 格式完全合法的答案可能数字错误；一个数学答案写法不符合模板，也可能在数学上正确。安全惩罚也要避免把正常拒答和高风险越权混成同一个标签，否则模型会通过普遍拒答获得看似安全的分数。

奖励项要做分项报告和消融。只看总 reward 上升无法判断模型到底提高了正确率、变得更守格式、学会了绕过 hack 检测，还是仅仅增加了输出长度。每个子奖励都要在已知正确、已知错误、边界格式和对抗输入上单独检查。

## 17.7 GRPO：用组内相对表现构造优势

如果为同一个问题采样一组回答，可以用组内相对表现形成学习信号，而不必维护一个独立的 value model。设问题 $x$ 生成 $G$ 条已知奖励的轨迹，奖励为 $r_1,\ldots,r_G$，则组内均值和总体标准差为：

```math
\mu_G=\frac{1}{G}\sum_{i=1}^{G}r_i
```

```math
\sigma_G
=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu_G)^2}
```

常见的教学优势写法是：

```math
A_i=\frac{r_i-\mu_G}{\sigma_G+\varepsilon}
```

$G$ 是正整数；若希望组内比较产生有效区分，通常需要 $G\ge2$，且 $\varepsilon>0$ 是有限的小常数。所有 $r_i$ 必须使用同一 verifier revision 和奖励尺度。若一组答案全部得相同奖励，$A_i$ 会全部为 0，说明当前组没有相对学习信号，而不是说明题目已被可靠解决。

DeepSeekMath 论文把 GRPO 描述为 PPO 的一种变体，并强调它在数学推理训练中的组内相对优化和内存取舍。下面的 PPO 风格目标用于解释裁剪和 KL 的作用：

```math
\mathcal{L}
=-\mathbb{E}\left[
\min\left(
\rho_t(\theta)A_t,
\mathrm{clip}(\rho_t(\theta),1-\epsilon_c,1+\epsilon_c)A_t
\right)\right]
+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
```

$\rho_t(\theta)$ 是新旧策略在有效 token 上的概率比，$A_t$ 是与该 token 或轨迹对齐的优势，$\epsilon_c>0$ 是裁剪范围，$\beta\ge0$ 是 KL 权重。不同实现对 KL 估计、长度归一化、token mask、组内归一化和过滤条件可能不同，因此这个式子是理解接口的数学模型，不是公开仓库中每一行代码的替代品。

## 17.8 一个可手算的组内例子

题目是“解一个方程并输出整数根”。同一题采样出三条已完成且可判断的轨迹，答案分别为 x=2、x=3、x=2，verifier 奖励为 [1,0,1]。此时：

```math
\mu_3=\frac{1+0+1}{3}=\frac{2}{3}
```

```math
\sigma_3
=\sqrt{\frac{(1-\frac23)^2+(0-\frac23)^2+(1-\frac23)^2}{3}}
=\frac{\sqrt{2}}{3}
```

当 $\varepsilon$ 很小时，两个正确轨迹有正优势，错误轨迹有负优势。训练更新会提高正确轨迹在相似上下文下的相对概率，但它并没有证明模型学会了方程的普遍解法；它只说明这一组 rollout 提供了区分信号。

如果三条轨迹全部因为答案格式解析失败而得到 0，优势全为 0。此时调大学习率不能凭空创造信号，先检查格式契约、答案抽取和题目难度更合理。如果三条都得到 1，则模型可能已经掌握该题，也可能 verifier 过于宽松；需要新的题目、变体或独立 verifier。

## 17.9 Zero-reward group 如何诊断

训练日志至少应记录每个问题组的 reward 均值、方差、全零比例、全满比例、未知比例、平均长度、解析失败、verifier 超时和工具错误。全零组升高可能来自四类原因：题目超出模型能力、采样预算太小、verifier 太严格或协议不匹配、模型已经偏离了可解析分布。

诊断顺序很重要。先用已知答案和固定轨迹测试 verifier，再检查答案抽取和格式，再检查题目难度与采样预算，最后才调整策略优化器。否则把 verifier bug 当成模型退化，会用更多训练把错误接口固化。

全满组也不能简单视为好消息。它可能表示题目过于简单，也可能表示 verifier 的判定覆盖不足。可以提高变体难度、加入已知错误答案和对抗格式，观察 verifier 是否能区分，而不是只期待平均 reward 继续上升。

## 17.10 Reward hacking：模型正在优化什么

只要奖励是可观察的，模型就可能寻找比设计者预期更便宜的路径。典型现象包括：

- 数学 verifier 只看最后一个数字，模型生成与数字无关的伪推导；
- 代码 verifier 只运行公开测试，模型硬编码样例或修改测试；
- 格式 verifier 只检查字段存在，模型填入不可解析或无意义的内容；
- 长度惩罚不足，模型重复同一句话以提高被某个分类器判为“有步骤”的概率；
- 安全过滤只检查显式命令，模型把敏感动作藏在间接参数或工具链中。

奖励升高而独立质量不升高，是 reward hacking 的重要信号。独立检查不能只换一个 prompt 的 judge，而要尽量改变证据来源：数学使用等价变形和反例，代码使用隐藏测试、资源限制和静态检查，工具任务改变顺序和错误返回，格式任务加入未知字段和边界输入。

verifier 自身也要有权限边界。代码测试器应运行在没有生产凭证、网络和任意文件写权限的沙箱中；测试脚本不能因为验证模型生成的代码而被赋予比被测任务更大的权限。否则训练优化的对象可能是测试环境，而不是解题能力。

## 17.11 数据生成：从 rollout 到可学习样本

RLVR 的训练数据不是只有题目和一个分数。一个可复核的 rollout 至少包含问题版本、策略模型版本、采样配置、轨迹、工具观察、最终答案、verifier 输出、资源消耗和失败原因。对代码任务还要绑定 workspace revision、依赖锁定文件、测试集合和 patch。

一个常见流水线是：策略生成多条候选，verifier 对每条候选执行检查，过滤或标注结果，策略更新使用奖励；训练外再用没有参与优化的题目、变体和独立 verifier 评估。每个环节都可能改变数据分布。过滤掉所有冗长轨迹可能损失必要推导，保留所有高分轨迹可能把 verifier 漏洞蒸馏进去。

数据去污染要按题目族、模板和答案来源处理，不能只比较字符串。公开 benchmark 的题目可能出现在预训练语料、教师生成数据和调参集里。训练题高分、变体低分，可能是模板记忆；teacher 生成的同分布测试高分，不能替代独立新题。

## 17.12 蒸馏：把昂贵搜索变成学生能力

高预算教师可以生成更多候选、使用 verifier、运行工具并保留失败分支，但在线对每个请求都这样做通常成本太高。蒸馏的目标是让学生在较小模型、较短延迟或较少工具调用下复现一部分有效行为。

最简单的 token 级知识蒸馏可以写成：

```math
\mathcal{L}_{\mathrm{KD}}
=-\frac{1}{|\mathcal{I}|}
\sum_{(x,t)\in\mathcal{I}}
\sum_{v\in\mathcal{V}}
p_T^{(T)}(v\mid x,y_{1:t-1})
\log p_S^{(T)}(v\mid x,y_{1:t-1})
```

$\mathcal{I}$ 是非空的有效位置集合，$\mathcal{V}$ 是有限词表，$p_T^{(T)}$ 和 $p_S^{(T)}$ 是使用温度 $T>0$ 得到的归一化分布。教师和学生的 tokenizer、上下文、mask 和位置必须对齐；如果只蒸馏最终答案，目标就不是上式的完整 token 分布，而是硬标签或答案级损失。

蒸馏内容有不同粒度：最终答案、结构化工具调用、验证通过的轨迹、错误候选及其失败原因、过程摘要、状态转移和完整 token 分布。完整隐藏思维不是默认最优的蒸馏对象，其中可能有敏感信息、未验证假设或只对教师模型有效的冗余路径。对代码 Agent，patch、测试命令、测试输出和 workspace revision 往往比一段漂亮的解释更能保留可验证行为。

学生要学习的是在自己的输入和预算下如何得到可验收结果，不是机械复制教师草稿。教师通过了某个 verifier，也不意味着学生生成的相似文本无需再次验证。

## 17.13 On-policy distillation 为什么更贵

离线蒸馏直接使用教师已经生成的数据，成本较低，但数据分布可能偏离学生。学生在自己容易犯错的状态附近，可能没有任何训练样本；模型只会模仿教师常见轨迹，却不会处理学生真实的解析错误、工具失败和短上下文。

On-policy distillation 让学生先生成自己的轨迹，再由教师、verifier 或规则对这些轨迹提供标签。它更接近学生的错误分布，代价是每轮要运行学生、教师和验证器，还要处理数据缓存、版本和失败样本。

代码例子很直观。高预算教师可能直接找到正确 patch，但学生更常见的错误是修改了错误文件、没有更新测试、在依赖缺失时继续猜。若只保存教师最终 patch，学生学不到失败后的状态恢复；若保存“候选 patch—测试失败—读取新日志—修正 patch—测试通过”的 artifact 链，学生才有机会学到可迁移的执行过程。

蒸馏样本应保存生成策略、teacher revision、verifier revision、过滤规则和污染标记。否则后续无法解释学生的提升是来自教师泛化、数据筛选，还是把 benchmark 答案记进了训练集。

## 17.14 数学训练与代码训练的边界

数学任务的 verifier 常能检查精确答案、符号等价或数值容差；步骤可能有多条合理路径，过程标签需要谨慎定义。代码任务则多了运行环境、依赖、资源、权限和副作用。测试通过是某个环境和测试集合下的行为事实，不是程序对所有输入都正确的证明。

因此数学轨迹可以把答案等价类和第一处错误作为重要字段，代码轨迹还要记录：

- 输入仓库和 workspace revision；
- 候选 patch 与修改路径；
- 测试命令、依赖和资源限制；
- 公开、隐藏和变体测试；
- 失败日志、回滚状态和副作用；
- 安全扫描与最终 artifact。

把数学的最终答案奖励直接搬到代码 Agent，可能奖励硬编码样例；把代码测试通过直接当作开放证明的过程正确，也会夸大能力。两类任务都需要独立验证，但 verifier 的覆盖和失败语义不同。

## 17.15 评估：训练分数之外还要看什么

RLVR 或蒸馏至少需要同时报告四种结果：

1. **候选生成能力**：候选池中是否曾经出现正确答案；
2. **选择与验证能力**：verifier 或策略是否选出正确候选；
3. **迁移能力**：新题、等价变形、隐藏测试、工具失败和不同长度下是否保持；
4. **系统代价**：模型 token、工具调用、验证时间、人工接管和单位成功成本。

设总样本数为 $N>0$，verifier 已给出确定判定的样本数为 $N_{\mathrm{known}}$，其中成功数为 $N_{\mathrm{success}}$，则：

```math
\mathrm{coverage}=\frac{N_{\mathrm{known}}}{N}
```

```math
\mathrm{success\_rate}_{\mathrm{known}}
=\frac{N_{\mathrm{success}}}{N_{\mathrm{known}}}
```

第一项回答有多少样本被 verifier 覆盖，第二项回答在已覆盖样本中有多少成功。当 $N_{\mathrm{known}}=0$ 时，第二项未定义，不应写成 0；当 $N=0$ 时，两项都未定义。把未知样本强行算进错误率，会混淆 verifier 不可用和模型失败。

如果一批任务总成本为 $C_{\mathrm{total}}$，成功数为 $N_{\mathrm{success}}$，单位成功成本是：

```math
C_{\mathrm{success}}=\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
```

$C_{\mathrm{total}}$ 必须包含模型、工具、验证和人工的同一成本账本；$N_{\mathrm{success}}=0$ 时结果为 None 或 undefined。蒸馏是否成功，不能只看学生分数，还要确认成功率、安全条件和人工复核没有恶化。

评估切分要包含原题、变体、新题、不同 verifier、不同工具状态和风险切片。只报告 R1 风格数学 benchmark 的提升，不能推出开放事实、生产代码和高风险工具动作都得到相同提升。

## 17.16 训练、评估和部署的证据链

一项可复核的实验要把训练数据版本、策略模型、参考模型、verifier、采样配置、checkpoint、评估集、污染规则和成本记录绑定到 revision。每个结果还要说明是论文报告、官方仓库、公开模型卡、教学构造还是目标系统实测。

部署时仍然需要 verifier。训练期 verifier 只说明策略在优化它；线上输入可能超出训练分布、工具版本可能变化、外部状态可能未知。对于代码和数据任务，先在隔离环境中执行，再把测试结果、资源消耗和 artifact 交给独立检查；对于付款、发布、删除和权限变更，正确答案也不能替代授权。

一个完整的闭环可以表示为：

```text
teacher or policy rollout
    -> versioned verifier
    -> accept / reject / unknown
    -> policy update or distillation
    -> independent evaluation
    -> deployment verifier
    -> cost, safety and recovery monitoring
```

这条链上任何一个版本没有记录，后续都可能无法解释回归。尤其是 verifier 变化：测试集合扩展、答案解析修复或超时规则调整，都会改变奖励分布，不能把前后 reward 直接当成同一指标。

## 17.17 一个可运行的 RLVR 与蒸馏审计器

下面的标准库示例不训练模型，而是把一小组 rollout 按 verifier contract 审计。它演示四件事：unknown 不被当作错误，组内相对优势只使用同一题且已确定的奖励，只有带证据的安全正确轨迹才进入蒸馏集合，零成功任务的单位成功成本保持未定义。

```python
import math


VERDICTS = {"correct", "incorrect", "unknown"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_non_negative(value, name):
    require(
        type(value) in {int, float} and math.isfinite(value) and value >= 0,
        f"{name} must be finite and non-negative",
    )


def validate_rollouts(rollouts):
    require(isinstance(rollouts, list) and rollouts, "rollouts must be non-empty")
    required = {
        "id",
        "problem_id",
        "answer",
        "verdict",
        "format_ok",
        "unsafe",
        "tokens",
        "evidence_ids",
        "verifier_revision",
    }
    seen = set()
    for row in rollouts:
        require(isinstance(row, dict) and set(row) == required, "rollout schema mismatch")
        require(isinstance(row["id"], str) and row["id"], "id must be non-empty")
        require(row["id"] not in seen, "duplicate rollout id")
        seen.add(row["id"])
        require(isinstance(row["problem_id"], str) and row["problem_id"], "problem_id must be non-empty")
        require(isinstance(row["answer"], str) and row["answer"], "answer must be non-empty")
        require(row["verdict"] in VERDICTS, "unknown verdict")
        require(type(row["format_ok"]) is bool, "format_ok must be boolean")
        require(type(row["unsafe"]) is bool, "unsafe must be boolean")
        finite_non_negative(row["tokens"], "tokens")
        require(
            isinstance(row["evidence_ids"], list)
            and all(isinstance(item, str) and item for item in row["evidence_ids"]),
            "evidence_ids must be a list of non-empty strings",
        )
        require(
            isinstance(row["verifier_revision"], str) and row["verifier_revision"],
            "verifier_revision must be non-empty",
        )
    return True


def reward(row):
    require(row["verdict"] in VERDICTS, "unknown verdict")
    if row["verdict"] == "unknown":
        return None
    if row["unsafe"]:
        return -1.0
    if row["verdict"] == "correct" and row["format_ok"]:
        return 1.0
    return 0.0


def group_advantages(rollouts, problem_id, epsilon=1e-8):
    validate_rollouts(rollouts)
    finite_non_negative(epsilon, "epsilon")
    require(epsilon > 0, "epsilon must be positive")
    group = [row for row in rollouts if row["problem_id"] == problem_id]
    known = [(row, reward(row)) for row in group if reward(row) is not None]
    require(len(known) >= 2, "at least two known rollouts are required")
    values = [score for _, score in known]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return [(row["id"], round((score - mean) / (std + epsilon), 3)) for row, score in known]


def metrics(rollouts):
    validate_rollouts(rollouts)
    known = [row for row in rollouts if row["verdict"] != "unknown"]
    successes = [row for row in known if reward(row) == 1.0]
    return {
        "coverage": len(known) / len(rollouts),
        "known_success_rate": None if not known else len(successes) / len(known),
        "distillable_ids": [
            row["id"]
            for row in successes
            if row["evidence_ids"]
        ],
    }


def unit_success_cost(total_cost, success_count):
    finite_non_negative(total_cost, "total_cost")
    require(type(success_count) is int and success_count >= 0, "success_count must be a non-negative integer")
    if success_count == 0:
        return None
    return total_cost / success_count


ROLLOUTS = [
    {
        "id": "r1",
        "problem_id": "math-1",
        "answer": "x=2",
        "verdict": "correct",
        "format_ok": True,
        "unsafe": False,
        "tokens": 30,
        "evidence_ids": ["eq-check-1"],
        "verifier_revision": "math-v3",
    },
    {
        "id": "r2",
        "problem_id": "math-1",
        "answer": "x=3",
        "verdict": "incorrect",
        "format_ok": True,
        "unsafe": False,
        "tokens": 22,
        "evidence_ids": [],
        "verifier_revision": "math-v3",
    },
    {
        "id": "r3",
        "problem_id": "math-1",
        "answer": "x=2",
        "verdict": "correct",
        "format_ok": True,
        "unsafe": False,
        "tokens": 34,
        "evidence_ids": ["eq-check-1"],
        "verifier_revision": "math-v3",
    },
    {
        "id": "r4",
        "problem_id": "math-1",
        "answer": "x=2",
        "verdict": "unknown",
        "format_ok": True,
        "unsafe": False,
        "tokens": 18,
        "evidence_ids": [],
        "verifier_revision": "math-v3",
    },
]


print("schema_valid", validate_rollouts(ROLLOUTS))
print("advantages", group_advantages(ROLLOUTS, "math-1"))
print("metrics", metrics(ROLLOUTS))
print("unit_success_cost", unit_success_cost(106, 2))
print("zero_success_cost", unit_success_cost(106, 0))
```

预期输出为：

```text
schema_valid True
advantages [('r1', 0.707), ('r2', -1.414), ('r3', 0.707)]
metrics {'coverage': 0.75, 'known_success_rate': 0.6666666666666666, 'distillable_ids': ['r1', 'r3']}
unit_success_cost 53.0
zero_success_cost None
```

这个例子把 unknown 从组内优势中排除，但仍把它计入整体覆盖率的分母。这样报告同时回答了 verifier 覆盖了多少样本和已知样本中成功多少。r1 与 r3 都有证据引用，进入教学上的蒸馏候选；这不表示它们可以绕过学生模型的再次验证。

边界测试应包括重复 ID、空 verifier revision、非法 verdict、NaN token、负 token、全是 unknown 的组、空 rollout 和零成功成本。全是 unknown 时，group_advantages 应明确拒绝，因为没有可比较的奖励；整体 known_success_rate 在没有已知样本时应返回 None，不能把不可判定伪装成失败率。

## 17.18 从审计器回到真实训练

教学代码没有实现 rollout、梯度更新或分布式 verifier，只验证训练接口最容易出错的边界。真实系统还要处理策略采样、长序列截断、token 级 mask、KL 估计、混合精度、并行执行、缓存、超时和 checkpoint 恢复。

最值得保留的设计原则是数据对象的完整性。一个奖励数字应能追溯到问题版本、轨迹、verifier revision、reason code、证据、资源消耗和安全判定；一个蒸馏样本应能追溯到教师版本、过滤规则、学生分布和污染状态。没有这些字段，训练曲线即使上升，也很难判断提升来自能力、数据泄漏还是 verifier 漏洞。

## 17.19 练习与判断

**练习一：Verifier contract。** 为一道数学题、一个 Python 函数和一个 JSON 输出各写出 correct、incorrect、unknown 的具体触发条件，并说明超时和解析失败分别属于哪一类。

**练习二：组内信号。** 设一组奖励为 [0,0,1,unknown]。说明哪些样本进入优势计算，为什么全零组不能通过提高学习率获得区分信号。

**练习三：蒸馏样本。** 设计一个代码修复样本，包含教师候选、失败测试、修正 patch、最终测试、workspace revision 和 verifier revision。说明只保留最终答案会丢掉什么。

**练习四：反事实评估。** 为一个数学 verifier 设计等价表达变体，为一个代码 verifier 设计隐藏测试和资源限制，并说明它们分别能发现哪一种 reward hacking。

## 17.20 资料边界与本章结论

DeepSeek-R1 的 arXiv 技术报告和官方仓库支持关于 R1-Zero、cold-start、强化学习路线、公开蒸馏模型和报告结果的陈述；DeepSeekMath 论文支持 GRPO 作为 PPO 变体及其数学训练背景；过程监督和 verifier 论文支持结果奖励、过程奖励与验证器设计的研究讨论。它们不支持对未公开数据配方、内部奖励权重、闭源模型搜索过程或部署策略作确定推断。

RLVR 的价值在于把可执行结果变成训练反馈，蒸馏的价值在于把高预算搜索中的有效行为迁移到更便宜的学生。两者的边界同样清楚：verifier 只能覆盖它定义的世界，组内优势只能利用当前组的相对信号，教师轨迹可能包含偏差，学生仍需独立验证。真正可信的提升必须同时经过独立题目、变体或隐藏测试、成本统计、安全审计和失败样本分析。

参考资料：

- DeepSeek, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning：<https://arxiv.org/abs/2501.12948>
- DeepSeek AI, DeepSeek-R1 官方仓库：<https://github.com/deepseek-ai/DeepSeek-R1>
- Shao et al., DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models：<https://arxiv.org/abs/2402.03300>
- Lightman et al., Let’s Verify Step by Step：<https://arxiv.org/abs/2305.20050>
- OpenAI, PRM800K 数据与代码：<https://github.com/openai/prm800k>
- Schulman et al., Proximal Policy Optimization Algorithms：<https://arxiv.org/abs/1707.06347>
