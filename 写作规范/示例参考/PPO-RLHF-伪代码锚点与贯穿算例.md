# PPO-RLHF 伪代码锚点与贯穿算例（参考范例）

> 用途：作为《写作规范/02-章节编写与正文教学.md》中“伪代码先搭路线，贯穿算例再讲细节”可选模板的完整参考，方便新会话或上下文压缩后快速恢复这种讲解方式。
>
> 来源：第五册第十一章《后训练与对齐训练》§3「PPO-RLHF」。以下保留该节当前正文；若参考稿与章节原文不一致，以章节原文为准，并同步更新本稿。
>
> 这是教学示例，不是必须照抄的模板，也不是实际训练配置。模型名、算术题、token 切分及演算数值均按正文说明作为教学设定。文中“前文 §1.3”等交叉引用指向来源章节。

## 3. PPO-RLHF

**背景（PPO 算法）：** 在强化学习里，策略（policy）是模型在当前状态下选择下一动作的概率分布；对语言模型来说，状态是 prompt 加回答前缀，动作就是下一个 token。普通策略梯度可以按采样结果更新这组概率，但如果拿同一批回答反复更新，策略可能一下子改变太多。信赖域策略优化（TRPO）通过限制新旧策略的差异来控制更新，却要近似求解带约束的优化问题，实现更复杂。PPO 于 2017 年提出，常见的裁剪形式用代理目标减少策略比率越过范围后仍继续推动更新的动力，同时允许对同一批样本做多轮小批量更新；原论文在机器人运动和 Atari 等强化学习任务中评估了它。[PPO 原论文](https://arxiv.org/abs/1707.06347)

**背景（语言模型偏好）：** “预测下一个 token”和“按用户意图给出有用、安全的回答”不是同一个目标。监督微调（SFT）教模型模仿示范回答，但许多问题没有唯一的标准措辞。拿本节的题目来说，标注者可以比较“答案是 $1$。”和“答案是 $2$。”，指出哪个回答更合适；reward model 从许多这样的比较中学习给完整回答打分。早期工作已把人类偏好学习用于自然语言任务；InstructGPT 后来采用“示范做 SFT、偏好比较训练 reward model、再用 PPO 优化策略”的流程。[Ziegler 等人的语言偏好学习研究](https://arxiv.org/abs/1909.08593)，[InstructGPT 技术报告](https://arxiv.org/abs/2203.02155)

**目的：** PPO-RLHF 要用偏好信号继续调整 SFT policy，让它更常生成被人偏好的回答，同时用 reference KL 控制它偏离 SFT 的程度。在这道题里，偏好样本告诉训练器“答案是 $1$。”优于“答案是 $2$。”；RM 再给 policy 实际生成的完整回答打分，作为优化信号。

**解决的问题：** PPO 解决策略怎样利用采样回答进行多轮更新、又减少更新过大的推动；RLHF 则把人对回答的比较转成可用于训练生成策略的奖励。clipping 约束 PPO 新旧策略比率的优化压力，reference KL 约束当前 policy 与 SFT reference 的偏离，二者作用不同。它们都不能证明本例的答案在数学上正确，正确性仍需单独检查。

**意义：** 把这几种作用分开，读者就能看懂为什么同一条回答会经过 RM、reference、critic 和 policy 的不同计算，也不会把 RM 高分误当成事实正确。本节沿这条链展开：`SFT → 偏好数据训练 reward model → policy 采样 → reward 加 KL → critic 算 advantage → PPO 更新 → 评估并重复`。下面用虚构的约 320 亿参数模型“云舟-32B”把模型来源和一轮训练串起来。假设预训练已完成；题目和分数都是教学设定，不是实际训练配方。伪代码只保留主要动作，具体计算会在后文用同一条回答展开。

```python
# 云舟-32B-Base 经指令示范数据做完 SFT，得到这个 checkpoint。
sft_32b = load("云舟-32B-SFT")

# 从同一份 SFT 权重分出 PPO 所需的四个角色。
policy_model = copy(sft_32b.model)                  # πθ：继续训练
reference_model = freeze(copy(sft_32b.model))       # πref：固定不变

preference_data = [
    ("3/4+1/4=?", "答案是 1。", "答案是 2。"),  # prompt, chosen, rejected
    ...
]
H = sft_32b.hidden_size
reward_model = copy(sft_32b.model)
reward_model.score_head = Linear(H, 1)  # 回答末尾 hidden state [H] → 整条回答一个分数
train_reward_model(reward_model, preference_data)
freeze(reward_model)

critic_model = copy(sft_32b.model)
critic_model.value_head = Linear(H, 1)  # 每个生成位置的 hidden state [H] → V(prefix)

# 每轮：采样回答 → 计算训练信号 → 更新 policy 和 critic。
for prompts in prompt_batches:
    rollout = policy_model.sample_and_record(prompts)  # 回答 + 这批的旧概率 πold
    # 本批可能采到：“分母相同，分子相加，得到 4/4=1。”
    rewards = shape_rewards(reward_model, reference_model, rollout, beta)  # 回答打分，再扣 KL 代价
    advantages, returns = GAE(rewards, critic_model.value(rollout))
    update_policy_with_PPO(policy_model, rollout, advantages)
    update_critic(critic_model, rollout, returns)
    evaluate(policy_model)
```

上面的伪代码把模型怎样来、每轮怎样接力都列出来了。先对齐各模型的来源：`云舟-32B-Base` 用示范问答做 SFT，得到 `云舟-32B-SFT`；policy 和 reference 都从这份 SFT 权重复制。policy 会继续更新，reference 冻结，作为“不要离 SFT 行为太远”的比较基准。reward model 也从 SFT 权重复制，接上 `score_head`，用同一问题下的 chosen/rejected 回答训练：例如 prompt 是 $3/4+1/4=?$，chosen 可以是“答案是 $1$。”，rejected 可以是“答案是 $2$。”，训练目标是让前者得分更高；完成后 RM 冻结。到了 PPO rollout，RM 读取 prompt 和一条完整回答，只给这一整条回答一个分数。critic 则从 SFT 权重复制并接上 `value_head`；对后面这条六 token rollout，它会在六个生成位置各输出一个前缀价值预测，学习“从这个前缀继续生成，预计能得到多少回报”。这个 value 不是 token 的偏好分，也不是 RM 对整条回答的分数。为便于看清职责，伪代码把这些模型分开写；实际实现可以共享骨干。

现在沿 `for` 循环逐行走一遍，看看一条回答怎样变成 policy 和 critic 的训练信号。

**步骤 1：`sample_and_record` 让 policy 生成回答，并记下当时的概率。** 输入是这一批 prompt，例如 $3/4+1/4=?$。policy 生成“分母相同，分子相加，得到 $4/4=1$。”为把后面的计算串起来，假设这次回答在教学算例中切成下面六个 token 片段：

| 位置 $t$ | 本例假设的 token 片段 $y_t$ | 采样时记录的 $\log\pi_{\mathrm{old}}(y_t\mid s_t)$ |
|---:|---|---:|
| 1 | `分母` | $-0.30$ |
| 2 | `相同，` | $-0.24$ |
| 3 | `分子` | $-0.35$ |
| 4 | `相加，` | $-0.28$ |
| 5 | `得到` | $-0.18$ |
| 6 | `4/4=1。` | $-0.25$ |

这六段是为了演算而设定的 token 切分，不代表某个真实 tokenizer 一定如此；实际训练按 tokenizer 产出的 token 和有效位置 mask 计算。每个 log probability 都是在当时的前缀 $s_t$ 下，policy 选中对应 token $y_t$ 的概率取对数。生成这批回答的 policy 快照记作 $\pi_{\mathrm{old}}$。它不是额外训练的第五个模型：训练器保留这些旧 log probability，后面更新 policy 时就能和生成回答时的概率比较；后续各步也沿用同一条回答。

**步骤 2：`shape_rewards` 让训练器评估偏好，并计算 policy 偏离 reference 的代价。** reward model 读取 prompt 和完整回答“分母相同，分子相加，得到 $4/4=1$。”，只输出一个序列分数。假设它给这条回答 $1.50$ 个教学奖励单位。它的训练数据告诉它哪个回答更受偏好，但这个分数不是数学正确性的证明，也不是给六个 token 各打一个偏好分。reference model 则对 rollout 中相同的六个 token、相同的前缀计算 log probability。对第 $t$ 个 token，令 $s_t$ 表示生成它之前的 prompt 和回答前缀，采样回答的最后一个有效 token 位置记作 $T$：

```math
d_t=
\log \pi_{\mathrm{old}}(y_t\mid s_t)
-
\log \pi_{\mathrm{ref}}(y_t\mid s_t),
\qquad
r_t=-\beta d_t+\mathbb{1}[t=T]r_{\phi}(x,y)
```

本例的 reference log probability 依次假设为 $[-0.32,-0.25,-0.34,-0.31,-0.17,-0.27]$，所以记录的旧策略与 reference 的差为 $[0.02,0.01,-0.01,0.03,-0.01,0.02]$。取教学用的 $\beta=0.5$，逐 token 加入的 KL 项和最终 reward 如下：

| 位置 $t$ | $d_t$ | $-\beta d_t$ | 加上回答分后的 $r_t$ |
|---:|---:|---:|---:|
| 1 | $0.02$ | $-0.010$ | $-0.010$ |
| 2 | $0.01$ | $-0.005$ | $-0.005$ |
| 3 | $-0.01$ | $+0.005$ | $+0.005$ |
| 4 | $0.03$ | $-0.015$ | $-0.015$ |
| 5 | $-0.01$ | $+0.005$ | $+0.005$ |
| 6 | $0.02$ | $-0.010$ | $1.490$ |

最后一个 token 的 reward 是回答分 $1.50$ 加上该位置的 KL 项 $-0.010$；前五个位置只有各自的 KL 项。六个 $d_t$ 之和是 $0.06$，所以本例整条回答的 KL shaping 合计为 $-0.5\times0.06=-0.03$，六个位置的 reward 合计为 $1.47$。这是单条 rollout 上的 log-ratio 计算；单个位置乃至一条回答的总估计都可能为负，不能把它当作必定非负的单条 KL 值。对 policy 采样的许多回答取期望后，这些 log-ratio 才对应常说的 policy/reference KL。若只追着 reward model 分数走，policy 可能学会迎合评分器而产出人不想要的回答；增大 $\beta$ 会加强对 SFT reference 的约束，但也可能压住有用的新行为。reward 分数的尺度取决于训练数据和任务，不能把某个模型上的 $\beta$ 直接照搬；评估时还要看独立偏好和原有能力。

**步骤 3：`critic_model.value` 和 `GAE` 把回答末尾的反馈传回前面。** critic 的 `value_head` 针对每个生成位置对应的前缀 $s_t$ 输出一个标量 $V_{\mathrm{old}}(s_t)$。这里 $s_t$ 是生成 $y_t$ 之前的上下文：prompt 加上已经生成的 $y_1$ 到 $y_{t-1}$。例如 $V_{\mathrm{old}}(s_4)$ 是 critic 看过 prompt、“分母”“相同，”和“分子”之后，对从这个前缀继续生成预计能获得多少后续 reward 的预测。它不是给 token “相加，”评偏好，也不是 reward model 给整条回答的分数。`GAE` 把每个位置的 reward 和相邻前缀的 value 预测合起来，先算一步误差，再从回答末尾向前递推：

```math
\delta_t=r_t+\gamma m_tV_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t),
\qquad
A_t=\delta_t+\gamma\lambda m_tA_{t+1}
```

沿用前面六个 token 的 reward，假设 critic 在生成每个 token 之前的六个前缀上的预测值依次为 $V_{\mathrm{old}}(s_t)=[0.10,0.18,0.26,0.34,0.42,0.60]$。取 $\gamma=1,\;\lambda=0.5$，回答在 $y_6$ 后正常结束，所以终止 mask 为 $m_1=\cdots=m_5=1,\;m_6=0$。按公式逐位置得到 $\delta_t=[0.070,0.075,0.085,0.065,0.185,0.890]$。例如末尾位置 $\delta_6=1.490-0.600=0.890$；前一位置 $\delta_5=0.005+0.600-0.420=0.185$。

GAE 再从回答末尾向前递推：

- 末尾 token $y_6$ 的 advantage 为 $A_6=\delta_6=0.890$。
- token $y_5$ 的 advantage 为 $A_5=0.185+0.5\times0.890=0.630$。
- token $y_4$ 的 advantage 为 $A_4=0.065+0.5\times0.630=0.380$。
- token $y_3$ 的 advantage 为 $A_3=0.085+0.5\times0.380=0.275$。
- token $y_2$ 的 advantage 为 $A_2=0.075+0.5\times0.275=0.2125$。
- token $y_1$ 的 advantage 为 $A_1=0.070+0.5\times0.2125=0.17625$。

因此，按本例这组 reward 和 value 预测，GAE 得到的六个 advantage 都为正；policy 更新会倾向于提高这些已采样 token 的概率。critic 的训练目标 $returns_t=V_{\mathrm{old}}(s_t)+A_t$ 依次为 $[0.27625,0.3925,0.535,0.720,1.050,1.490]$，会把对应前缀上的 value 预测往这些目标拉近。这里的数值只为演示 GAE 如何把回答末尾的偏好分和逐 token KL 信号传回前面，不代表 reward model 的通用分数或真实训练配方。

$\delta_t$ 是这一步收到的 reward 加上下一状态的预期价值，再减去 critic 原先的预测； $A_t$（advantage，优势）表示这次采样得到的后续结果比原先预期好多少。例子里 $\lambda=0.5$，所以每一步只把后续 advantage 的一半加回来；真实终止处不再向后 bootstrap，padding 位置由 mask 排除。长度上限造成的截断不一定代表任务结束，不能一律把后续价值清零。 $\gamma$ 控制远处回报的权重， $\lambda$ 控制 TD 误差信号向前累积多少。真实训练中逐 token KL 项也会进入各位置的 $r_t$；因此末尾分数会影响前面的 advantage，但每个位置的 advantage 还会受该处 reward 和 critic 预测影响。

**步骤 4：`update_policy_with_PPO` 用 advantage 调整 policy。** policy 重新给同一批已采样 token 计算概率，再和步骤 1 留下的 $\pi_{\mathrm{old}}$ 概率比较。对 token $y_t$，训练器比较的 ratio 是 $\rho_t=\frac{\pi_{\theta}(y_t\mid s_t)}{\pi_{\mathrm{old}}(y_t\mid s_t)}$：刚采完样本、尚未更新时它等于 $1$。以第四个 token “相加，”为例，旧 log probability 是 $-0.28$；假设某次更新后当前 policy 在相同前缀上的 log probability 变为 $-0.22$，那么 $\rho_4=\exp(-0.22-(-0.28))\approx1.06$，表示它现在生成这个 token 的概率约为采样时的 $1.06$ 倍。由于本例 $A_4=0.38$ 为正，这个方向符合 PPO 的优化倾向。取 $\epsilon=0.2$ 时，$1.06$ 仍在 $[0.8,1.2]$ 内，clipping 尚未触发；它限制样本推动更新的幅度，不是把 token 概率硬截到某个区间，完整的裁剪目标和边界例子见前文 §1.3。这里有两个容易混的比较：步骤 2 中 policy/reference 的差异约束当前策略相对 SFT reference 的漂移；PPO 的新旧 policy ratio 则衡量当前更新相对这批回答的采样策略走了多远。它们服务于不同步骤，应分开看。对同一批 rollout 做有限轮 minibatch 更新时，$\pi_{\mathrm{old}}$ 的记录保持不变；复用过多轮会让新旧概率差距变大。

**步骤 5：`update_critic` 让 critic 学会预测 returns。** 对同一批回答前缀，critic 再算一次 value，并用有效 token 位置上的平方误差把新预测拉向刚才的 `returns`。例如第四个 token 之前，旧预测 $V_{\mathrm{old}}(s_4)=0.34$，GAE 目标是 $returns_4=0.72$；单看这个位置的未归一化平方误差为 $(0.34-0.72)^2=0.1444$，训练会把该前缀的 value 往 $0.72$ 拉近。RM 是从 chosen/rejected 学会给完整回答打一个偏好分；critic 是从 PPO 生成的回答和逐位置 returns 学会估计每个前缀的后续回报。RM 在这一流程中冻结，critic 则随 PPO 训练。若 critic 预测长期偏高或跟不上 policy，advantage 也会不准；检查时可以对照 value 预测和 returns，而不是只看 reward model 分数。

**步骤 6：`evaluate` 检查这轮更新值不值得继续。** 更新后仍用留出的偏好题、安全题和原有能力题检查 policy：例如确认它在这道分数题上仍能给出正确推理，同时在独立偏好评估中没有只学会迎合 RM。RM 给出的 $1.50$ 只是一个代理分数，不能替代这些评估。若偏好有改善且原有能力没有明显退化，再让更新后的 policy 生成下一批回答；新回答及其采样概率成为下一轮 rollout 和新的 $\pi_{\mathrm{old}}$。最初从 SFT 复制的 reference 仍保持冻结，RM 也继续冻结。于是循环每一轮都会把 policy 新生成的回答交给 RM/reference 打分，再用 GAE 和 PPO 更新 policy 与 critic。

**适用场景：** 从这道分数题可以看出，PPO-RLHF 适合能收集回答偏好、能让当前 policy 反复生成新回答并由 reward model 评分，而且在线采样成本可以接受的任务，例如评价回答是否有帮助、完整或符合指定风格。若任务有可验证的客观答案，例如本例的算术正确性，偏好分数仍不能代替独立校验。

**优缺点：**

- **优点：** 在 SFT 模仿示范的基础上，policy 还能根据偏好分数调整自己当前实际生成的回答；PPO 允许对同一批 rollout 做多轮小批量更新，并用 clipping 控制代理目标对更新的推动。
- **缺点：** 每轮都要在线生成回答，并让 policy、reference、reward model 和 critic 分别处理同一批 rollout；即使实现共享部分权重，也会增加前向计算和工程复杂度。
- **缺点：** reward model 只是有限偏好数据的代理，policy 可能学会迎合它；critic 估值不准会让 advantage 失真。clipping 也不是硬性概率边界，reference KL 系数、奖励尺度和回答长度等设置都会影响更新。

若手头主要是固定的 chosen/rejected 偏好对、暂时不需要在线采样更新，可以接着看下一节的 DPO；方法选择取决于训练信号、采样需求和资源条件，PPO 并非所有偏好优化任务的默认答案。
