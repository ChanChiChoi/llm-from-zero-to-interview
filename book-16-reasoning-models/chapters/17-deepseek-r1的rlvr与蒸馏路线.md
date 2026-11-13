# 第 17 章 DeepSeek-R1 的 RLVR 与蒸馏路线：让可验证结果塑造推理

## 17.1 为什么答案可检查会改变训练方式

传统监督微调把题目和答案写成样本 $(x,y)$，模型学习在给定输入后复现答案。这个方法适合语言、格式和基本任务行为，但高质量长推理轨迹需要人工或教师模型逐步写出，而且同一道题可能有很多条都正确的解法。把其中一条解法当成唯一标准，容易把答案正确和必须沿着这条路径作答混为一谈。

数学、程序和结构化任务提供了另一条路。数学答案可以代回方程或交给符号程序，代码可以在隔离环境中编译和测试，JSON 可以按 JSON Schema 校验。外部验证器（verifier）不必逐字教模型应该怎么想，而是判断生成结果是否满足可执行约束。用这种结果训练模型，就是可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards，RLVR）的核心直觉。

这并不意味着接上一个验证器就自动得到可靠推理。验证器只定义了它能观察到的正确性；模型会优化这个定义，可能利用测试缺口、格式漏洞或奖励函数的盲区。因此 RLVR 的真正对象不是一个抽象的奖励分数，而是一套接受域、拒绝域、未知状态、资源限制和独立评估共同组成的训练契约。

DeepSeek-R1 的公开技术报告和官方仓库让这条路线受到广泛关注，但公开材料和社区猜测必须分开。公开资料可以支持 R1-Zero 直接在基础模型上进行大规模强化学习、R1 在强化学习前加入冷启动（cold-start）数据、公开的蒸馏模型和报告中的实验结果；它不能自动证明未披露的数据配比、每个奖励权重、内部筛选规则或闭源产品的实现方式。

## 17.2 RLVR 的最小数学对象

给定问题 $x$，策略模型生成一条轨迹 $\tau$。轨迹可以只是文本答案，也可以包含思考 token、工具调用、观察、代码补丁和最终结果。验证器读取问题、轨迹和执行环境，返回一个结构化判定：

```text
verdict（判定状态）: correct | incorrect | unknown
score: [0, 1]
reason_code: exact_match | symbolic_check | test_pass | parse_error | timeout
evidence: ...
verifier_revision: ...
```

最简单的结果奖励是：

```math
r_{\mathrm{out}}(x,\tau)=
\begin{cases}
1,&\mathrm{Verify}(x,\tau)=\mathrm{correct}\\
0,&\mathrm{Verify}(x,\tau)=\mathrm{incorrect}\\
\bot,&\mathrm{Verify}(x,\tau)=\mathrm{unknown}
\end{cases}
```

这里 $\mathrm{Verify}$ 必须绑定已定义的判定程序版本，且每条轨迹只落入一个 verdict。$\bot$ 表示没有可用于该任务的数值奖励，不能当作 0；教学例中，正确、错误、超时三条轨迹分别得到 $1$、$0$、$\bot$。未知结果应保留原因，并按契约重试、复核或暂缓更新。若把未知样本从组内优势中排除，还要同时报告覆盖率，因为超时与题目难度相关时，简单排除可能造成选择偏差。$r_{\mathrm{out}}$ 取值为 0 或 1 只是教学构造，其他系统可用连续分数，但必须说明它是概率、排序分数还是业务效用。

把本题当前纳入审计的有限轨迹集 $\mathcal{T}_x$ 按验证器状态分为三个集合：

```math
\begin{aligned}
\mathcal{A}_x&=\{\tau\in\mathcal{T}_x:\mathrm{Verify}(x,\tau)=\mathrm{correct}\},\\
\mathcal{I}_x&=\{\tau\in\mathcal{T}_x:\mathrm{Verify}(x,\tau)=\mathrm{incorrect}\},\\
\mathcal{U}_x&=\{\tau\in\mathcal{T}_x:\mathrm{Verify}(x,\tau)=\mathrm{unknown}\},\\
\mathcal{T}_x&=\mathcal{A}_x\,\dot\cup\,\mathcal{I}_x\,\dot\cup\,\mathcal{U}_x
\end{aligned}
```

其中 $\mathcal{A}_x$、$\mathcal{I}_x$、$\mathcal{U}_x$ 分别收纳正确、错误和未知轨迹，$\dot\cup$ 表示两两不相交的并集。以三条合成轨迹为例，若 $\tau_1$ 答对、$\tau_2$ 答错、$\tau_3$ 执行超时，则 $\mathcal{A}_x=\{\tau_1\}$、$\mathcal{I}_x=\{\tau_2\}$、$\mathcal{U}_x=\{\tau_3\}$，三者合起来正好覆盖这三条观测。超时、资源耗尽、解析失败和环境崩溃究竟属于 unknown 还是 incorrect，必须由任务协议决定；不能把没有证据伪装成答案错误，也不能把解析失败伪装成模型推理失败。

令 `\pi_\theta` 为待训练策略，`\pi_{\mathrm{ref}}` 为参考策略，`R(x,\tau)` 为验证器与其他约束合成的有限数值奖励，`\beta` 为有限非负 KL 散度惩罚系数；参考策略在当前策略的概率支持上取正值，以保持 KL 有限。期望对 `\tau\sim\pi_\theta(\cdot\mid x)` 下的轨迹取值；没有数值奖励的 unknown 轨迹应按契约排除或延迟处理，不能把 `\bot` 代入期望。策略的简化目标可以写成：

```math
J(\theta)
=\mathbb{E}_{\tau\sim\pi_\theta(\cdot\mid x)}
 [R(x,\tau)]
-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
```

这里是强化学习目标的抽象，不是某个公开项目的完整训练代码；实际系统还需要处理 token mask、采样批次、截断、优势估计、优化器和分布式采样轨迹（rollout）。

例如一组教学轨迹的期望任务奖励为 $0.75$，策略与参考策略之间的 KL 为 $0.10$，惩罚系数 $\beta=0.20$，则 $J=0.75-0.20\times0.10=0.73$。这说明优化器会在任务收益与偏离参考策略之间折中；三个数都是合成输入，不能读作 DeepSeek-R1 的真实训练指标。

## 17.3 R1-Zero、R1 与公开证据

DeepSeek-R1 论文把 R1-Zero 与 R1 分成两条训练路线。R1-Zero 从 DeepSeek-V3-Base 出发，在没有先做监督微调（supervised fine-tuning，SFT）的设定下直接进行大规模强化学习（reinforcement learning，RL）；数学、代码等可验证任务使用规则奖励组合来检查答案与格式。作者报告训练中出现更长的推理、反思和自我检查，同时也记录了可读性差、重复和语言混杂等问题。这些是模型特定的作者观察，不证明可见推理文本忠实呈现内部过程，也不代表所有任务都得到稳定提升。[论文 v2](https://arxiv.org/abs/2501.12948v2)

作者在冷启动附录还说明，第一人称表达和更自然的对话风格是人工整理与数据生成的设计选择，不是模型获得人类式认知或自主解题能力的证据。[冷启动数据说明](https://arxiv.org/abs/2501.12948v2)

R1 是在这一探索基础上增加冷启动数据和多阶段训练。把论文披露的顺序画成表，比笼统说“结合 RL 和 SFT”更容易看到每阶段改变了什么：

| 阶段 | 公开披露的输入与训练信号 | 主要目的 |
| --- | --- | --- |
| 冷启动 SFT | 数千条对话式、人工对齐的思考样本 | 提高可读性并提供初始推理格式 |
| 第一阶段 RL | 推理任务的规则奖励，并加入语言一致性信号 | 提升推理行为并减少语言混杂 |
| 拒绝采样与 SFT | 从策略采样、筛选推理样本，并与非推理数据一起监督微调 | 扩充推理轨迹，同时恢复写作和通用任务能力 |
| 第二阶段 RL | 推理任务规则奖励；一般任务使用有帮助性与安全性相关的模型奖励 | 继续改进推理，并调整有帮助性与安全性 |

论文表 3 报告最终 R1 在 AIME 2024 的 Pass@1（单次采样正确率估计）为 79.8%、MATH-500 基准测试为 97.3%、LiveCodeBench 思维链（chain-of-thought，CoT）格式下的 Pass@1 为 65.9%。这些是作者在其评测协议下报告的结果：每题最多生成 32,768 tokens，采用采样温度 0.6、核采样 top-p 0.95 的非零温采样估计 Pass@1，AIME 每题采样 64 个回答；它们不是独立复现或生产任务的成功率。[评测协议与结果表](https://arxiv.org/abs/2501.12948v2)

这条表格概括论文公开的流水线，不是完整复现配方；数据配比、筛选阈值和许多内部实现并未全部公开。R1-Zero 的探索路线说明规则反馈能否推动能力变化；R1 的阶段组合则显示，推理正确率、输出可读性、通用行为和安全目标需要不同信号协调。更完整的训练系统见第五册[RLVR 与可验证奖励](../../book-05-llm-training/chapters/16-rlvr与可验证奖励.md)和[在线策略蒸馏](../../book-05-llm-training/chapters/17-on-policy-distillation.md)；数学、代码任务的推理细节分别回看[第八章](08-数学推理训练.md)和[第九章](09-代码推理与执行反馈.md)。

R1 的蒸馏路线也要和策略优化区分。论文报告用 R1 生成的约 80 万条样本微调 Qwen、Llama 等开源底模；蒸馏模型只做监督微调，没有接着进行强化学习。官方仓库公开了六个 1.5B 至 70B 的稠密模型检查点。[论文附录 F](https://arxiv.org/abs/2501.12948v2)与[官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)支持这些特定披露。这里的“蒸馏”指从教师模型生成样本学习，不表示学生模型复制教师权重、隐藏状态或完整推理过程。

这组材料不能推出所有推理模型都使用同一配方，也不能把特定训练报告的基准测试结果外推到开放事实、任意代码库、工具权限或高风险生产任务。论文、官方仓库、教学构造和目标系统实测分别是不同的证据等级。

## 17.4 结果奖励与过程奖励

结果监督（outcome supervision）只在轨迹末端判断成功，信号稀疏，却容易定义。过程监督（process supervision）由过程奖励模型（process reward model，PRM）等机制在中间步骤上提供反馈，可能帮助模型更早发现错误，但每个步骤的正确往往依赖上下文，标注和自动验证都更难。

设轨迹有 `T` 个经过定义的步骤，`T` 为正整数；最终奖励 `R_T` 和各步过程奖励 `r_t` 是有限数值；权重 `\alpha` 为有限非负数、折扣 `0\le\gamma\le1`，且结果与过程奖励已校准到可比较的尺度，则教学上的混合奖励可以写成：

```math
R(\tau)
=\alpha R_T
 +\sum_{t=1}^{T}\gamma^{t-1}r_t
```

轨迹没有可验证步骤时，过程项不是 0 分的同义词，而是缺失或不适用；如果强行补零，模型会把未标注当成步骤错误。

例如两步推导的最终奖励 $R_T=1$，过程分数为 $r_1=1$、$r_2=0.5$，取 $\alpha=1$、$\gamma=0.5$，则混合奖励为 $1+1+0.5\times0.5=2.25$。这是展示权重如何传播的合成例；真实系统还要把结果分与步骤分校准到有意义的共同尺度。

结果奖励的优点是允许多种解法。模型只要进入接受域，就不必逐字复现教师路径。它的缺点是信用分配困难：前面哪一步导致失败，最终的一个 0 分不会直接说明。过程奖励能缓解这个问题，却带来新的奖励投机（reward hacking）风险：模型可能学习写出符合步骤分类器的句子，而不是让步骤真正推进到正确答案。

工程上常见的组合是：结果验证器作为最终约束，过程验证器或规则作为辅助信号，独立测试和反例用来检查两者是否一致。过程分数永远不应替代最终任务验收，尤其是代码、外部工具和有副作用的任务。

过程监督的效果依赖任务和实现。Lightman 等人在 MATH 任务上报告，过程监督模型在其代表性测试子集上解出 78.2% 的题目，并发布含 80 万条步骤级人工标签的 PRM800K；论文也报告主动学习提高了该实验中的标注效率。[论文](https://arxiv.org/abs/2305.20050v1)、[PRM800K 数据与代码](https://github.com/openai/prm800k) DeepSeek-R1 作者在自己的大规模 RL 实验中则报告，细粒度步骤难以统一定义和标注，基于模型的 PRM 还增加奖励投机风险与训练开销；他们认为 PRM 在候选重排（rerank）或搜索中有用，但在该 RL 配置中的收益不足以抵消额外成本。[R1 论文附录 G.2](https://arxiv.org/abs/2501.12948v2) 两种结果评估的问题和系统条件不同，不能简化成“PRM 一定更好”或“PRM 不值得用”；步骤监督的详细机制见[第五章](05-process-supervision.md)。

## 17.5 验证器契约（verifier contract）：训练前先定义接受什么

一个可用的验证器不只是一个返回 0 或 1 的函数。它需要回答以下问题：

数学答案是否允许等价表达式、不同变量顺序和数值容差？代码答案是否必须在离线沙箱运行，依赖是否固定，隐藏测试是否存在？结构化答案的额外字段是被忽略、拒绝还是进入未知状态？超时是错误、未知还是需要人工复核？不同版本的测试结果能否直接比较？

可以把验证器输出抽象为：

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

判定状态（verdict）表示离散结果，score 必须说明是判定分数还是校准概率，reason_code 用于区分答案错误、格式错误、超时和验证器故障，evidence 让结论可复核，资源字段防止无限运行，revision 绑定测试和规则版本。

unknown 是训练系统必须认真保留的状态。例如代码执行器收到请求后连接断开，系统不知道代码是否已经运行；将其统一当作错误可能鼓励模型极短地输出，将其统一当作正确则会奖励空结果。对有外部副作用的工具，unknown 还关系到是否允许重试，不能只为方便计算把它压成一个数字。

验证器还需要规定拒绝域之外的输入。空答案、非有限数、过长输出、非法工具参数、越权文件访问和资源耗尽都应有明确的 reason_code。否则模型可能通过触发解析异常或消耗资源来改变奖励分布。一般验证器和奖励模型（reward model）的机制见[第四章](04-verifier与reward-model.md)，本章聚焦它们如何进入 R1 的可验证训练路线。

## 17.6 奖励不只是正确加一分

令 `r_{\mathrm{result}}`、`r_{\mathrm{format}}`、`r_{\mathrm{hack}}`、`r_{\mathrm{unsafe}}` 分别为结果、格式、已检测投机和不安全行为的有限数值信号；`\alpha`、`\beta`、`\gamma` 是有限非负权重。软惩罚下的教学总奖励可写为：

```math
R(\tau)
=r_{\mathrm{result}}(\tau)
 +\alpha r_{\mathrm{format}}(\tau)
 -\beta r_{\mathrm{hack}}(\tau)
 -\gamma r_{\mathrm{unsafe}}(\tau)
```

每个信号都应规定有限范围和测量方式。若安全违规是硬性不可接受事件，不能依靠提高结果奖励抵消它，而应直接拒绝该轨迹。

以合成分数 $r_{\mathrm{result}}=1$、$r_{\mathrm{format}}=0.5$、$r_{\mathrm{hack}}=0$、$r_{\mathrm{unsafe}}=0$，以及 $\alpha=0.2$、$\beta=1$、$\gamma=2$ 为例，总分为 $1+0.2\times0.5=1.1$。若同一轨迹触发硬性安全规则，系统应直接拒绝它，而不是拿这个加权总分与正确奖励抵消。

格式奖励有用，但不等于内容正确。一个 JSON 格式完全合法的答案可能数字错误；一个数学答案写法不符合模板，也可能在数学上正确。安全惩罚也要避免把正常拒答和高风险越权混成同一个标签，否则模型会通过普遍拒答获得看似安全的分数。

奖励项要做分项报告和消融。只看总奖励上升无法判断模型到底提高了正确率、变得更守格式、学会了绕过投机检测，还是仅仅增加了输出长度。每个子奖励都要在已知正确、已知错误、边界格式和对抗输入上单独检查。

## 17.7 组相对策略优化（Group Relative Policy Optimization，GRPO）：用组内相对表现构造优势

如果为同一个问题采样一组回答，可以用组内相对表现形成学习信号，而不必维护一个独立的价值模型（value model）。设问题 `x` 生成 `G` 条已知奖励的轨迹，`G` 是正整数，奖励 `r_1,\ldots,r_G` 是同一验证器版本与奖励尺度下的有限数值，则组内均值和总体标准差为：

```math
\mu_G=\frac{1}{G}\sum_{i=1}^{G}r_i
```

```math
\sigma_G
=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu_G)^2}
```

若要计算标准化优势，要求 `G\ge2`，并令有限常数 `\varepsilon>0` 用于防止标准差为零时除零；常见的教学优势写法是：

```math
A_i=\frac{r_i-\mu_G}{\sigma_G+\varepsilon}
```

若一组答案全部得相同奖励，$A_i$ 会全部为 0，说明当前组没有相对学习信号，而不是说明题目已被可靠解决。

DeepSeekMath 论文把 GRPO 描述为近端策略优化（Proximal Policy Optimization，PPO）的一个变体，并强调它在数学推理训练中的组内相对优化和内存取舍。令 `\pi_{\mathrm{old}}` 为生成轨迹的旧策略，`\rho_{i,t}(\theta)` 为第 `i` 条轨迹第 `t` 个有效 token 上新旧策略概率之比，且旧策略概率为正；`\hat A_{i,t}` 为与该 token 对齐的有限优势，`\epsilon_c>0` 为有限裁剪范围，`\beta` 为有限非负 KL 权重。裁剪函数把概率比限制在 `[1-\epsilon_c,1+\epsilon_c]`，期望在有效采样 token 上计算；参考策略在当前策略的概率支持上取正值，以保持 KL 有限。下面的 PPO 风格目标用于解释裁剪和 KL 的作用：

```math
\mathcal{L}
=-\mathbb{E}\left[
\min\left(
\rho_{i,t}(\theta)\hat A_{i,t},
\mathrm{clip}(\rho_{i,t}(\theta),1-\epsilon_c,1+\epsilon_c)\hat A_{i,t}
\right)\right]
+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
```

结果监督常把组内轨迹优势用于该轨迹的有效 token，过程监督则可使用步骤对齐的优势。不同实现对 KL 估计、长度归一化、token 掩码、组内归一化和过滤条件可能不同，因此这个式子是理解接口的数学模型，不是公开仓库中每一行代码的替代品。

看第 $i$ 条轨迹的一个 token：若 $\rho_{i,t}=1.5$、$\hat A_{i,t}=0.5$、$\epsilon_c=0.2$，裁剪后的比率为 $1.2$，未裁剪项为 $1.5\times0.5=0.75$，裁剪项为 $1.2\times0.5=0.60$，取较小值 $0.60$。若本例 KL 为 $0.10$ 且 $\beta=0.02$，则该 token 对损失的贡献是 $-0.60+0.02\times0.10=-0.598$。这只解释一个 token 的 surrogate 计算，不能代表整批更新。

DeepSeek-R1 的作者还在附录记录过蒙特卡洛树搜索（Monte Carlo tree search，MCTS）训练尝试：token 搜索空间增长快，节点扩展限制可能困在局部路径，而搜索又依赖难以训练的细粒度价值模型（value model）。作者同时指出，预训练价值模型配合 MCTS 仍可能改善推理时搜索；这不是对 MCTS 一概无效的结论。完整搜索机制见[第六章](06-search与tree-of-thought.md)，并行与协调成本见[第十六章](16-multi-agent与ultra-test-time-compute.md)。[R1 论文附录 G.2](https://arxiv.org/abs/2501.12948v2)

## 17.8 一个可手算的组内例子

题目是“解一个方程并输出整数根”。同一题采样出三条已完成且可判断的轨迹，答案分别为 x=2、x=3、x=2，验证器奖励为 [1,0,1]。此时：

```math
\mu_3=\frac{1+0+1}{3}=\frac{2}{3}
```

```math
\sigma_3
=\sqrt{\frac{(1-\frac23)^2+(0-\frac23)^2+(1-\frac23)^2}{3}}
=\frac{\sqrt{2}}{3}
```

因此 $\sigma_3\approx0.4714$。令 $\varepsilon=10^{-8}$，优势依次为 $(1-2/3)/(0.4714+10^{-8})\approx0.707$、$(0-2/3)/(0.4714+10^{-8})\approx-1.414$、$0.707$。两个通过验证器的回答获得正优势，错误回答获得负优势；如果三条都是同一分数，则标准差为 0，加入 $\varepsilon$ 后优势仍全为 0，组内没有可用区分信号。

训练更新会提高正确轨迹在相似上下文下的相对概率，但它并没有证明模型学会了方程的普遍解法；它只说明这一组 rollout 提供了区分信号。

如果三条轨迹全部因为答案格式解析失败而得到 0，优势全为 0。此时调大学习率不能凭空创造信号，先检查格式契约、答案抽取和题目难度更合理。如果三条都得到 1，则模型可能已经掌握该题，也可能 verifier 过于宽松；需要新的题目、变体或独立 verifier。

## 17.9 全零奖励组（zero-reward group）如何诊断

训练日志至少应记录每个问题组的奖励均值、方差、全零比例、全满比例、未知比例、平均长度、解析失败、验证器超时和工具错误。全零组升高可能来自四类原因：题目超出模型能力、采样预算太小、验证器过严或协议不匹配、模型已经偏离了可解析分布。

诊断顺序很重要。先用已知答案和固定轨迹测试验证器，再检查答案抽取和格式，再检查题目难度与采样预算，最后才调整策略优化器。否则把验证器缺陷当成模型退化，会用更多训练把错误接口固化。

全满组也不能简单视为好消息。它可能表示题目过于简单，也可能表示验证器的判定覆盖不足。可以提高变体难度、加入已知错误答案和对抗格式，观察验证器是否能区分，而不是只期待平均奖励继续上升。

## 17.10 奖励投机（reward hacking）：模型正在优化什么

只要奖励是可观察的，模型就可能寻找比设计者预期更便宜的路径。典型现象包括：

- 数学验证器只看最后一个数字，模型生成与数字无关的伪推导；
- 代码验证器只运行公开测试，模型硬编码样例或修改测试；
- 格式验证器只检查字段存在，模型填入不可解析或无意义的内容；
- 长度惩罚不足，模型重复同一句话以提高被某个分类器判为“有步骤”的概率；
- 安全过滤只检查显式命令，模型把敏感动作藏在间接参数或工具链中。

奖励升高而独立质量不升高，是奖励投机的重要信号。独立检查不能只换一个提示词的评审模型（judge），而要尽量改变证据来源：数学使用等价变形和反例，代码使用隐藏测试、资源限制和静态检查，工具任务改变顺序和错误返回，格式任务加入未知字段和边界输入。

“奖励信号不可靠”也不是一种单一噪声。Shao 等人在 Qwen2.5-Math-7B、GRPO 和 MATH-500 的特定设置中报告，随机分配奖励后分数提高 21.4 个百分点，接近使用真实答案奖励时的 29.1 个百分点；作者把这一反常现象与 GRPO 裁剪偏差放大底模已有高先验行为联系起来，并强调效果依赖模型家族，不能当作训练建议。[ICML 2026 论文](https://proceedings.mlr.press/v306/shao26d.html) Zhu 与 Kang 研究的则是错误答案标注和验证误差：他们从先前研究标作错误的数学标注中随机抽查 20 例，发现 8 例实际正确；重新核验后，纯错误标注训练在其数学评测中比干净数据低约 8%—10%，BIRD Text2SQL 的真实标注错误设置低约 5%—12%。这些数字绑定论文的数据清理流程、模型、算法和任务。[论文 v2](https://arxiv.org/html/2603.16140v2)

两项研究考察的信号并不相同：前者随机化策略生成轨迹获得的奖励，后者检查与目标答案及验证器有关的标注错误。它们不能合并成“随机奖励有效”或“错误标注都能被算法吸收”；实际训练仍需分别审计答案真值、验证覆盖、奖励分布和模型迁移。

验证器自身也要有权限边界。代码测试器应运行在没有生产凭证、网络和任意文件写权限的沙箱中；测试脚本不能因为验证模型生成的代码而被赋予比被测任务更大的权限。否则训练优化的对象可能是测试环境，而不是解题能力。

## 17.11 数据生成：从 rollout 到可学习样本

RLVR 的训练数据不是只有题目和一个分数。一个可复核的采样轨迹至少包含问题版本、策略模型版本、采样配置、轨迹、工具观察、最终答案、验证器输出、资源消耗和失败原因。对代码任务还要绑定工作区版本、依赖锁定文件、测试集合和补丁。

一个常见流水线是：策略生成多条候选，验证器逐条检查，系统过滤或标注结果，再把奖励交给策略更新；训练外还要用没有参与优化的题目、变体和独立验证器评估。每个环节都可能改变数据分布。过滤掉所有冗长轨迹可能损失必要推导，保留所有高分轨迹可能把验证器漏洞蒸馏进去。

数据去污染要按题目族、模板和答案来源处理，不能只比较字符串。公开基准的题目可能出现在预训练语料、教师生成数据和调参集里。训练题高分、变体低分，可能是模板记忆；教师生成的同分布测试高分，不能替代独立新题。

## 17.12 知识蒸馏（knowledge distillation，KD）：把昂贵搜索变成学生能力

高预算教师模型（teacher）可以生成更多候选、使用验证器、运行工具并保留失败分支，但在线对每个请求都这样做通常成本太高。知识蒸馏的目标是让学生模型（student）在较小模型、较短延迟或较少工具调用下复现一部分有效行为。

令 `I` 为非空有效位置集合，`V` 为有限词表；`x` 是每个样本的输入，`y_{1:t-1}` 是当前预测位置之前的目标 token 前缀；`p_T` 和 `p_S` 分别是教师与学生在温度 `tau>0` 下的归一化分布，并要求学生在教师概率非零的位置也赋予正概率。教师和学生的 tokenizer、上下文、mask 与位置必须对齐。最简单的 token 级知识蒸馏可以写成：

```math
\mathcal{L}_{\mathrm{KD}}
=-\frac{1}{|\mathcal{I}|}
\sum_{(x,t)\in\mathcal{I}}
\sum_{v\in\mathcal{V}}
p_T^{(\tau)}(v\mid x,y_{1:t-1})
\log p_S^{(\tau)}(v\mid x,y_{1:t-1})
```

如果只蒸馏最终答案，目标就不是上式的完整 token 分布，而是硬标签或答案级损失。

用一个只有两个 token 的词表演算一个有效位置：教师分布为 $(0.75,0.25)$，学生分布为 $(0.5,0.5)$，温度 $\tau=1$。该位置的蒸馏交叉熵为 $-[0.75\ln0.5+0.25\ln0.5]=\ln2\approx0.693$ 纳特（nat）。这个手算只覆盖一个 token 位置；实际训练还要在所有有效位置和词表上求和或平均，并按实现选择是否乘温度平方。

蒸馏内容有不同粒度：最终答案、结构化工具调用、验证通过的轨迹、错误候选及其失败原因、过程摘要、状态转移和完整 token 分布。完整隐藏思维不是默认最优的蒸馏对象，其中可能有敏感信息、未验证假设或只对教师模型有效的冗余路径。对代码智能体（agent），补丁、测试命令、测试输出和工作区版本往往比一段漂亮的解释更能保留可验证行为。

学生要学习的是在自己的输入和预算下如何得到可验收结果，不是机械复制教师草稿。教师通过了某个 verifier，也不意味着学生生成的相似文本无需再次验证。

## 17.13 在线策略蒸馏（on-policy distillation）为什么更贵

离线蒸馏直接使用教师已经生成的数据，成本较低，但数据分布可能偏离学生。学生在自己容易犯错的状态附近，可能没有任何训练样本；模型只会模仿教师常见轨迹，却不会处理学生真实的解析错误、工具失败和短上下文。

在线策略蒸馏让学生先生成自己的轨迹，再由教师、验证器或规则对这些轨迹提供标签。它更接近学生的错误分布，代价是每轮要运行学生、教师和验证器，还要处理数据缓存、版本和失败样本。

代码例子很直观。高预算教师可能直接找到正确补丁，但学生更常见的错误是修改了错误文件、没有更新测试、在依赖缺失时继续猜。若只保存教师最终补丁，学生学不到失败后的状态恢复；若保存“候选补丁—测试失败—读取新日志—修正补丁—测试通过”的产物链，学生才有机会学到可迁移的执行过程。

蒸馏样本应保存生成策略、教师模型版本、验证器版本、过滤规则和污染标记。否则后续无法解释学生的提升是来自教师泛化、数据筛选，还是把基准答案记进了训练集。

## 17.14 数学训练与代码训练的边界

数学任务的验证器常能检查精确答案、符号等价或数值容差；步骤可能有多条合理路径，过程标签需要谨慎定义。代码任务则多了运行环境、依赖、资源、权限和副作用。测试通过是某个环境和测试集合下的行为事实，不是程序对所有输入都正确的证明。

因此数学轨迹可以把答案等价类和第一处错误作为重要字段，代码轨迹还要记录：

- 输入仓库和工作区版本；
- 候选补丁与修改路径；
- 测试命令、依赖和资源限制；
- 公开、隐藏和变体测试；
- 失败日志、回滚状态和副作用；
- 安全扫描与最终 artifact。

把数学的最终答案奖励直接搬到代码智能体，可能奖励硬编码样例；把代码测试通过直接当作开放证明的过程正确，也会夸大能力。两类任务都需要独立验证，但验证器的覆盖和失败语义不同。

## 17.15 评估：训练分数之外还要看什么

RLVR 或蒸馏至少需要同时报告四种结果：

1. **候选生成能力**：候选池中是否曾经出现正确答案；
2. **选择与验证能力**：verifier 或策略是否选出正确候选；
3. **迁移能力**：新题、等价变形、隐藏测试、工具失败和不同长度下是否保持；
4. **系统代价**：模型 token、工具调用、验证时间、人工接管和单位成功成本。

设总样本数 `N` 为正整数，verifier 已给出确定判定的样本数 `N_known` 满足 `0\le N_known\le N`，其中成功数 `N_success` 满足 `0\le N_success\le N_known`，则覆盖率为：

```math
\mathrm{coverage}=\frac{N_{\mathrm{known}}}{N}
```

当 `N_known>0` 时，已知样本成功率为：

```math
\mathrm{success\_rate}_{\mathrm{known}}
=\frac{N_{\mathrm{success}}}{N_{\mathrm{known}}}
```

第一项回答有多少样本被 verifier 覆盖，第二项回答在已覆盖样本中有多少成功。当 $N_{\mathrm{known}}=0$ 时，第二项未定义，不应写成 0；当 $N=0$ 时，两项都未定义。把未知样本强行算进错误率，会混淆 verifier 不可用和模型失败。

例如 $N=4$ 条轨迹里有 3 条得到确定 verdict，其中 2 条成功，则覆盖率为 $3/4=0.75$，已知样本成功率为 $2/3\approx0.667$。两者回答不同问题：前者衡量判定器能否给出结论，后者衡量有结论部分的任务成功率。

若 `C_total` 是有限非负的统一成本账本，成功数 `N_success` 是正整数，则单位成功成本为：

```math
C_{\mathrm{success}}=\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
```

成本账本必须按同一口径包含模型、工具、验证和人工；当 `N_success=0` 时结果为 None 或 undefined。蒸馏是否成功，不能只看学生分数，还要确认成功率、安全条件和人工复核没有恶化。

若这批任务总成本为 106 个教学 credits，成功数为 2，则单位成功成本为 $106/2=53$ credits。credits 是示例中的统一账本单位；真实比较必须固定计价、硬件/服务、任务集合和人工计入规则。

评估切分要包含原题、变体、新题、不同验证器、不同工具状态和风险切片。只报告 R1 风格数学基准的提升，不能推出开放事实、生产代码和高风险工具动作都得到相同提升。

## 17.16 训练、评估和部署的证据链

一项可复核的实验要把训练数据版本、策略模型、参考模型、验证器、采样配置、模型检查点（checkpoint）、评估集、污染规则和成本记录绑定到各自版本。每个结果还要说明是论文报告、官方仓库、公开模型卡、教学构造还是目标系统实测。

部署时仍然需要验证器。训练期的验证器只说明策略在优化它定义的判定；线上输入可能超出训练分布、工具版本可能变化、外部状态可能未知。对于代码和数据任务，先在隔离环境中执行，再把测试结果、资源消耗和最终产物交给独立检查；对于付款、发布、删除和权限变更，正确答案也不能替代授权。

一个完整的闭环可以表示为：

```text
教师或策略采样轨迹
    -> 版本化验证器
    -> 接受 / 拒绝 / 未知
    -> 策略更新或蒸馏
    -> 独立评估
    -> 部署验证器
    -> 成本、安全与恢复监测
```

这条链上任何一个版本没有记录，后续都可能无法解释回归。尤其是验证器变化：测试集合扩展、答案解析修复或超时规则调整，都会改变奖励分布，不能把前后奖励直接当成同一指标。

## 17.17 一个可运行的 RLVR 与蒸馏审计器

下面的标准库示例不训练模型，而是把一组采样轨迹（rollout）按验证器契约审核。它演示四件事：unknown 不被当作错误，组内相对优势只使用同一题且已确定的奖励，只有带证据的安全正确轨迹才进入蒸馏集合，零成功任务的单位成功成本保持未定义。

> 示例代码：已验证（2026-11-13，CPython 3.12.9；仅使用标准库）。

```python
import math


VERDICTS = {"correct", "incorrect", "unknown"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_non_negative(value, name):
    valid = type(value) in {int, float}
    if valid:
        try:
            valid = math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
    require(valid, f"{name} must be finite and non-negative")


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
        require(
            isinstance(row["verdict"], str) and row["verdict"] in VERDICTS,
            "verdict must be correct, incorrect, or unknown",
        )
        require(type(row["format_ok"]) is bool, "format_ok must be boolean")
        require(type(row["unsafe"]) is bool, "unsafe must be boolean")
        require(type(row["tokens"]) is int and row["tokens"] >= 0, "tokens must be a non-negative integer")
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
    require(
        isinstance(row["verdict"], str) and row["verdict"] in VERDICTS,
        "verdict must be correct, incorrect, or unknown",
    )
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
    require(group, "problem_id has no rollouts")
    revisions = {row["verifier_revision"] for row in group}
    require(len(revisions) == 1, "group must use one verifier revision")
    known = []
    for row in group:
        score = reward(row)
        if score is not None:
            known.append((row, score))
    require(len(known) >= 2, "at least two known rollouts are required")
    values = [score for _, score in known]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return [(row["id"], round((score - mean) / (std + epsilon), 3)) for row, score in known]


def metrics(rollouts):
    validate_rollouts(rollouts)
    revisions = {row["verifier_revision"] for row in rollouts}
    require(len(revisions) == 1, "metrics must use one verifier revision")
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

边界检查应覆盖重复 ID、空验证器版本、非字符串判定状态、NaN/负数/布尔/浮点 token 数、同组或总体指标混用验证器版本、全是未知状态的组、未知 problem_id、空采样轨迹集合、NaN/负数/超大整数成本、布尔或负成功数及零成功成本。全是未知状态时，group_advantages 应明确拒绝，因为没有可比较的奖励；整体成功率在没有已知样本时应返回 None，不能把不可判定伪装成失败率。

## 17.18 从审计器回到真实训练

教学代码没有实现采样轨迹生成、梯度更新或分布式验证器，只验证训练接口最容易出错的边界。真实系统还要处理策略采样、长序列截断、token 级掩码、KL 估计、混合精度、并行执行、缓存、超时和检查点恢复。

最值得保留的设计原则是数据对象的完整性。一个奖励数字应能追溯到问题版本、轨迹、verifier revision、reason code、证据、资源消耗和安全判定；一个蒸馏样本应能追溯到教师版本、过滤规则、学生分布和污染状态。没有这些字段，训练曲线即使上升，也很难判断提升来自能力、数据泄漏还是 verifier 漏洞。

## 17.19 练习与判断

**练习一：Verifier contract。** 为一道数学题、一个 Python 函数和一个 JSON 输出各写出 correct、incorrect、unknown 的具体触发条件，并说明超时和解析失败分别属于哪一类。

**练习二：组内信号。** 设一组奖励为 [0,0,1,unknown]。说明哪些样本进入优势计算，为什么全零组不能通过提高学习率获得区分信号。

**练习三：蒸馏样本。** 设计一个代码修复样本，包含教师候选、失败测试、修正 patch、最终测试、workspace revision 和 verifier revision。说明只保留最终答案会丢掉什么。

**练习四：反事实评估。** 为一个数学 verifier 设计等价表达变体，为一个代码 verifier 设计隐藏测试和资源限制，并说明它们分别能发现哪一种 reward hacking。

## 17.20 资料边界与本章结论

DeepSeek-R1 的 arXiv 技术报告和官方仓库支持关于 R1-Zero、冷启动、强化学习路线、公开蒸馏模型和报告结果的陈述；DeepSeekMath 论文支持 GRPO 作为 PPO 变体及其数学训练背景；过程监督和验证器研究支持结果奖励、过程奖励与验证器设计的讨论。它们不支持对未公开数据配方、内部奖励权重、闭源模型搜索过程或部署策略作确定推断。

RLVR 的价值在于把可执行结果变成训练反馈，蒸馏的价值在于把高预算搜索中的有效行为迁移到更便宜的学生模型。两者的边界同样清楚：验证器只能覆盖它定义的判定范围，组内优势只能利用当前组的相对信号，教师轨迹可能包含偏差，学生仍需独立验证。真正可信的提升必须同时经过独立题目、变体或隐藏测试、成本统计、安全审计和失败样本分析。

参考资料：

- DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, arXiv v2：<https://arxiv.org/abs/2501.12948v2>
- DeepSeek AI, DeepSeek-R1 官方仓库：<https://github.com/deepseek-ai/DeepSeek-R1>
- Shao et al., DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv v3：<https://arxiv.org/abs/2402.03300v3>
- Lightman et al., Let’s Verify Step by Step, arXiv v1：<https://arxiv.org/abs/2305.20050v1>
- OpenAI, PRM800K 数据与代码：<https://github.com/openai/prm800k>
- Schulman et al., Proximal Policy Optimization Algorithms：<https://arxiv.org/abs/1707.06347>
- Shao et al., Spurious Rewards: Rethinking Training Signals in RLVR, ICML 2026：<https://proceedings.mlr.press/v306/shao26d.html>
- Zhu and Kang, Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards, arXiv v2：<https://arxiv.org/abs/2603.16140v2>
