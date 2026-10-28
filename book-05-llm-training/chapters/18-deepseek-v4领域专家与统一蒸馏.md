# DeepSeek V4 后训练流水线：领域专家培养与统一蒸馏

> 资料来源：[DeepSeek V4 技术报告](https://arxiv.org/abs/2606.19348) §5.1–5.2、[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)（核验日期 2026-10-26）。报告明确披露了“领域专家训练 → 多教师 OPD”的两阶段路线，并进一步说明采用全词表 logit distillation 和按需教师调度。本章把论文披露、通用算法解释和教学设计建议分别标出；没有披露的路由权重、完整 loss 配置和训练数据不作猜测。

## 一个模型为什么需要“先分开练，再合在一起”

想象一个新人同时学习数学证明、代码调试、写作和网络安全。让他每天把所有任务混在一张试卷上，确实简单，却可能出现两个问题：他在某一领域还没形成稳定策略，就被其他领域的反馈打断；不同领域的成功标准也可能互相冲突。另一种办法是先让他在每个领域分别跟一位教练训练，形成几种相对成熟的能力，再让一位总教练把这些能力合并成一套可部署的工作方式。

DeepSeek V4 技术报告把后训练描述为两阶段流程：先针对数学、代码、Agent、指令跟随等目标训练领域专家，再训练一个统一学生，让它在自己生成的状态上向多个教师学习。报告说第二阶段有 10 个以上教师，并以全词表 reverse-KL 蒸馏合并能力。[技术报告 §5.1–5.2](https://arxiv.org/html/2606.19348) 没有公开每个专家的训练数据配比、教师权重、精确路由算法、完整 loss 配置，也没有给出一个可单独归因于 OPD 的消融结果；因此读者能看懂公开方案，但不能仅凭最终 benchmark 断言 OPD 单独带来了多少提升。

“专家”在这条流水线里至少指三种不同对象：

| 名称 | 所在阶段 | 具体是什么 | 不能混淆成什么 |
|---|---|---|---|
| 领域专家模型 | 后训练第一阶段 | 针对某类任务经 SFT、领域 RL 训练出的完整教师模型 | 不是推理时按 token 路由的单个 FFN expert |
| MoE routed expert | 模型内部、训练与推理时 | 最终 V4 模型 MoE 层中的参数子网络，由 token router 选择 | 不是一份数学教师 checkpoint 或代码教师 checkpoint |
| 统一学生模型 | 后训练第二阶段之后 | 用多个领域教师的信号训练出的单一部署模型；其自身架构仍可包含 MoE | 不表示把多份教师参数直接平均成一份权重 |

因此，V4 的“统一”说的是领域教师的行为信号被蒸馏进最终模型参数；它不意味着最终模型变成 dense，也不意味着模型服务端继续逐请求加载 10 多个领域教师。模型卡明确把 V4-Pro 和 V4-Flash 都标为 MoE，这与训练阶段的领域教师模型是两条不同的“专家”轴。[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

## 第一阶段：领域专家培养

设领域集合为 `D={d_1,...,d_K}`。对领域 `d_k`，先使用领域数据和示范做监督微调，再用该领域的 prompts 与 reward signals 做强化学习。V4 报告称其专家训练沿用 V3.2 后训练流程，并在 RL 阶段使用 GRPO；报告列举数学、代码、Agent 和指令跟随作为目标领域示例，但没有列出所有专家 checkpoint 的名称和完整训练配方。报告还说最终多教师蒸馏阶段使用 10 个以上领域教师。[技术报告 §5.1.1–5.1.2](https://arxiv.org/html/2606.19348#S5.SS1)

对于领域 `d_k`，一个带目标掩码的 SFT loss 可以写成：

```math
\mathcal{L}_{\mathrm{SFT}}^{(k)}
=-\frac{1}{M_k}\sum_t m_t\log p_\theta(y_t\mid x,y_{1:t-1}),
\qquad M_k=\sum_t m_t>0.
```

`x` 是 prompt/上下文，`y_t` 是第 `t` 个目标 token，`m_t∈{0,1}` 标记该位置是否计入 loss，`M_k` 是本批次参与监督的 token 数。若 prompt 和回答都应学习，掩码可以覆盖两者；instruction tuning 常只监督 assistant 目标区间。用有效 token 数归一化，可以避免不同长度样本的 loss 尺度随 batch 总 token 数任意变化。

随后在有领域 reward signal 的环境中做 GRPO（Group Relative Policy Optimization）。通常做法是对同一个 prompt 采样一组回答，用组内相对奖励估计优势，减少对独立 value model 的依赖。下面公式是常见的教学化组内标准化写法，**不是** V4 报告公开的精确 advantage、归一化或 clipping 配置：

```math
A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}
{\mathrm{std}(r_{1:G})+\epsilon},
```

其中 `G` 是同一 prompt 的回答数，`r_i` 是 verifier、奖励模型或规则组合产生的 reward，`A_i` 是组内相对优势，`std` 这里使用组内总体标准差。实际 GRPO 还涉及 policy ratio、clip、KL 项、token mask、长度处理和 rollout 过滤；V4 报告只说超参数与其既有研究接近，没有公开本章公式中的具体配置。GRPO 的通用推导和 V4 报告不能互相替代，前者帮助理解算法，后者支持关于 V4 的事实。

组内归一化能减少不同 prompt 的绝对奖励尺度差异，但它不是奖励正确性的证明。如果一组回答全部利用了 verifier 漏洞，最高的那个仍可能只是“漏洞利用得最好”。因此每个领域都需要独立检查器、反作弊测试、隐藏任务和人工抽样。

下面的零依赖代码只演示组内标准化，不包含策略梯度、clipping、KL 约束或 rollout 服务。它的用途是让读者看到：优势表示的是同一任务组内“相对好多少”，不是跨任务可直接比较的绝对分数。

> **代码状态：已验证（2026-10-26，Python 标准库；解释器小版本未记录）。** 两个 Python 示例均已执行；组内数值和 11 类非法输入/边界检查符合研究记录，不代表复现 DeepSeek V4。

```python
import math


def group_advantages(rewards, eps=1e-8):
    if not rewards:
        raise ValueError("a group must contain at least one reward")
    if any(
        isinstance(r, bool)
        or not isinstance(r, (int, float))
        or not math.isfinite(r)
        for r in rewards
    ):
        raise ValueError("rewards must be finite real numbers")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance)
    return [(r - mean) / (std + eps) for r in rewards]


group = [0.2, 0.8, 0.5, 0.1]
advantages = group_advantages(group)
print("advantages:", [round(x, 3) for x in advantages])
print("mean advantage:", round(sum(advantages) / len(advantages), 6))
```

当组内奖励不全相同，优势的平均值应接近 0。若所有奖励都相同，标准差为 0，代码仍会返回全 0；这表示当前 verifier 没有提供组内区分度，而不是模型已经学会任务。生产训练还要单独记录 reward variance、全同组比例、verifier unknown 比例和过滤率。V4 报告没有公开这些诊断项的统计，因此它们是复现或设计训练时应保留的账本。

### 可验证任务与生成式奖励模型

奖励不是所有领域都能用同一种检查器得到。代码可运行测试，数学可核对最终答案或证明约束；开放式写作、复杂 Agent 轨迹和多维任务质量通常没有一个可靠的布尔判定。DeepSeek V4 报告把两类问题分开：容易验证的任务可用规则 verifier 或测试；难验证任务使用 rubric-guided RL 数据，并用 Generative Reward Model（GRM）评价策略轨迹。报告还称，actor 网络本身兼任 GRM，并对它进行 RL 优化，让生成能力与评价能力一同训练。[技术报告 §5.1.1](https://arxiv.org/html/2606.19348#S5.SS1)

这个设计试图缓解“训练奖励需要大量人工标注”和“固定标量 reward 难以覆盖复杂质量标准”的问题。但要把它变成可审计流程，至少还要问清楚：rubric 从何而来、GRM 输出如何转换为策略更新所需信号、actor 给自己或同系列模型打分时如何避免偏袒、是否有独立人类评审校准。论文报告没有披露这些接口的完整 schema、独立校准结果和全部 reward 细节；“少量多样人工标注下更稳健”是作者报告的主张，不能当作已独立验证的普遍结论。

工程上仍应保留一组不参与训练的独立评估：人工盲评、隐藏测试、反 rubric 套路样本和跨领域负例。若 GRM 分数升高而任务完成率、事实正确率或安全指标不升，优化的可能是评分器偏好，而非真实能力。

## 为什么领域奖励要和权限分开

代码领域的测试通过、数学领域的答案正确、网络安全领域的漏洞发现都可以成为质量信号，但它们不能自动授予工具权限。一个模型可能生成了通过单元测试的 patch，却试图读取不必要的 secret；一个安全研究任务可能在模拟环境中允许 exploit，但生产系统不允许同样的动作。

训练记录应将结果质量和策略约束分开：

```math
S=C_{\mathrm{correct}}\cdot C_{\mathrm{format}}
\cdot P_{\mathrm{permission}}\cdot B_{\mathrm{budget}}.
```

`C_correct` 和 `C_format` 分别表示结果正确、输出格式通过的指示量；`P_permission` 和 `B_budget` 分别表示权限合规、预算未超限的指示量。这里的乘法表达“任一硬条件失败都不能称为完整成功”的语义。若权限或预算状态未知，不能强行填 0 或 1；应该保留 `unknown`，修复环境或隔离该轨迹。奖励高不能抵消越权、隐私泄露或未授权副作用。

这是面向 Agent 训练的一般安全设计要求，不是 DeepSeek V4 报告披露的某个权限乘子或奖励实现。评估器给出“任务完成”与执行器允许“继续调用工具”是两个不同决定。

## 第二阶段：统一模型的 on-policy distillation

领域专家拥有不同的分布和行为风格。若只把教师预先写好的答案做离线 SFT，学生主要在固定示范前缀上学习；上线后，学生自己生成的错误前缀、错误工具参数和失败后的 observation 可能没有出现在示范里。[GKD 论文](https://arxiv.org/abs/2306.13649)把这类训练/生成分布错位作为动机，并让学生在自生成序列上获得教师反馈；它在摘要、翻译、算术推理和 instruction tuning 设置中报告了实验，不能直接外推为 DeepSeek V4 的复现结果。

统一蒸馏让学生先 rollout，再让一个或多个领域教师在这些状态上提供 token 分布、下一步动作、错误批注、候选答案或最终 artifact。为了说明 Agent 场景中“学生状态”包含什么，可以把训练记录设计成：

```math
u=(x,s_t,a_t,o_{t+1},y,u_T,v,\rho_e),
```

其中 `s_t` 是学生在第 `t` 步看到的完整状态，`a_t` 是 token 或工具动作，`o_{t+1}` 是环境 observation，`y` 是最终结果，`u_T` 是教师信号，`v` 是 verifier，`\rho_e` 是环境和协议版本。这是用于复现和审计的推荐 schema，不是 DeepSeek 报告披露的内部日志格式。若只记录用户 prompt 和最终文本，研究者就无法重建学生在哪个错误前缀上得到什么指导。

DeepSeek V4 报告将多教师目标写成加权 reverse KL。把它展开到学生 rollout 访问到的前缀上，可以写为：

```math
\mathcal{L}_{\mathrm{OPD}}(\theta)
=\mathbb{E}_{s\sim d_{\pi_\theta}}
\left[\sum_{i=1}^{N}w_i\,
D_{\mathrm{KL}}\left(
\pi_\theta(\cdot\mid s)\|\pi_{E_i}(\cdot\mid s)
\right)\right].
```

`d_{\pi_\theta}` 表示由学生策略访问到的状态分布，`\pi_{E_i}` 是第 `i` 个专家教师，`w_i` 是报告所说的教师相对权重。报告解释，学生应在数学状态上向数学教师学习、在代码状态上向代码教师学习；但它没有公开 `w_i` 是固定的、按样本指定的还是由何种路由器计算。因此公式中的状态期望用于解释 on-policy，不能被读成完整复现配方。论文给出的核心形式是 `\sum_i w_i D_{KL}(\pi_\theta\|\pi_{E_i})`，而不是本文为读者补上的状态期望记法。[技术报告 §5.1.2](https://arxiv.org/html/2606.19348#S5.SS1)

在一个固定前缀 `s` 上，若学生分布为 `p_S`、教师分布为 `p_T`，reverse KL 为：

```math
D_{\mathrm{KL}}(p_S\|p_T)
=\sum_v p_S(v\mid s)\log\frac{p_S(v\mid s)}{p_T(v\mid s)}.
```

权重方向很重要：每个 token 的差异由学生概率 `p_S(v|s)` 加权，所以学生已经几乎不考虑的 token，在 reverse KL 中通常贡献很小；forward KL 则是 `D_KL(p_T||p_S)=Σ_v p_T(v|s)log(p_T(v|s)/p_S(v|s))`，由教师概率加权，更关注覆盖教师有质量的区域。reverse KL 因而常表现出偏向学生当前主要模式的性质，但它可能漏掉学生尚未探索到的教师模式。对词表做完整求和消除了动作抽样误差，不会自动解决 student support、教师覆盖或状态分布的问题。

学生先按 `p_S` 采样 token 时，可以用下式构造一个**采样估计量**：

```math
\hat A_t^{\mathrm{KD}}
=\operatorname{sg}\!\left[
\log\frac{p_T(a_t\mid s_t)}{p_S(a_t\mid s_t)}
\right],
\qquad a_t\sim p_S(\cdot\mid s_t).
```

`\operatorname{sg}` 表示 stop-gradient。这个 log-ratio 不是一条可以直接当作监督标签的 reward；在 sampled-token 方案里，它作为 policy-gradient surrogate 的 advantage 信号，仍要乘上学生策略的梯度。学生采到一个教师也认为更好的 token 时，log-ratio 为正；反之为负。[MiniLLM](https://arxiv.org/abs/2306.08543) 也研究了 reverse-KL 与 on-policy 优化，但它的序列级目标、信用分配方法和模型规模实验属于另一套具体算法，不能把“都用了 reverse KL”理解成完全相同的实现。

Teacher 和 student 至少要在同一 tokenization、词表映射、prompt 模板、状态版本和动作定义下比较分布。否则“同一个 token ID”或“同一位置的 log-prob”未必表示同一个动作。服务失败、教师拒答、词表对不齐或版本不匹配时，不能把缺失 log-prob 当成极低概率。

### 为什么 V4 报告选择全词表 logit distillation

只抽一个学生采样 token 来估计某个前缀上的 reverse-KL，估计器无偏并不代表单次信号稳定。下面用三 token 词表手算：

| token | 学生 `p_S` | 教师 `p_T` | `ln(p_S/p_T)` | 从学生采到该 token 的概率 |
|---|---:|---:|---:|---:|
| A | 0.60 | 0.70 | -0.154 | 0.60 |
| B | 0.30 | 0.20 | 0.405 | 0.30 |
| C | 0.10 | 0.10 | 0.000 | 0.10 |

精确求和得到 `D_KL(p_S || p_T) = 0.60×(-0.154) + 0.30×0.405 + 0.10×0 ≈ 0.029` nat。若每次只抽一个 token，样本可能是 `-0.154`，也可能是 `0.405`；当真实 KL 只有 `0.029` 时，单样本波动比均值大得多。把所有词表项的概率都放进求和，可在这个前缀上直接算出 KL 及其对 logits 的梯度，消除“只抽到哪个 token”引入的采样噪声。它仍然要采样学生前缀，所以序列/状态分布的波动并没有消失。

下面的 Python 3 标准库示例输入两个已归一化的离散分布，输出精确 reverse KL，以及“只采一个 student token 来估计 KL”时单样本 log-ratio 的标准差。它只验证固定前缀上的概率账本，不实现模型梯度或完整 OPD：

> **代码状态：已验证（2026-10-26，Python 标准库；解释器小版本未记录）。** 同上，仅验证局部概率账本和输入边界，不运行教师模型或训练。

```python
import math


def reverse_kl_and_sample_std(student, teacher):
    if (
        not isinstance(student, (list, tuple))
        or not isinstance(teacher, (list, tuple))
        or not student
        or len(student) != len(teacher)
    ):
        raise ValueError("distributions must be non-empty and have equal length")
    probabilities = list(student) + list(teacher)
    if any(
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or p < 0
        for p in probabilities
    ):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(student), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("student probabilities must sum to one")
    if not math.isclose(sum(teacher), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("teacher probabilities must sum to one")

    terms = []
    for p_s, p_t in zip(student, teacher):
        if p_s == 0:
            continue
        if p_t == 0:
            raise ValueError("teacher support must include every student-supported token")
        terms.append((p_s, math.log(p_s / p_t)))

    kl = sum(p_s * log_ratio for p_s, log_ratio in terms)
    variance = sum(p_s * (log_ratio - kl) ** 2 for p_s, log_ratio in terms)
    variance = max(0.0, variance)
    return kl, math.sqrt(variance)


kl, sample_std = reverse_kl_and_sample_std(
    [0.60, 0.30, 0.10],
    [0.70, 0.20, 0.10],
)
print(f"reverse KL: {kl:.5f} nat")
print(f"one-token estimator std: {sample_std:.5f} nat")
```

预期输出约为 `reverse KL: 0.02915 nat`、`one-token estimator std: 0.25046 nat`。方差来自该示例的单个采样 token；它不是训练梯度标准差，也不能代表 V4 实测稳定性。

DeepSeek V4 报告指出，已有实现常把 full-vocabulary KL 化为逐 token sampled estimate，再复用 RL policy loss；作者认为这种做法梯度方差较高、训练容易不稳定，于是选择 full-vocabulary logit distillation。应把结论说准确：这是作者报告的训练设计与动机，不是所有 token-level OPD 必然不稳定，也不说明 V4 的总训练梯度没有任何方差。报告没有提供让 sampled 与 full-vocabulary 两种 OPD 在其余条件完全相同时的公开消融表。[技术报告 §5.1.2](https://arxiv.org/html/2606.19348#S5.SS1)

### 多教师冲突不是简单“取平均”

报告把多个教师的 reverse KL 按相对权重求和。在固定状态、不加额外约束且所有相关 token 上教师概率非零时，最小化该目标的学生分布满足：

```math
p^*(v\mid s)
=\frac{\prod_i p_{E_i}(v\mid s)^{w_i/W}}
{\sum_u\prod_i p_{E_i}(u\mid s)^{w_i/W}},
\qquad W=\sum_i w_i.
```

也就是说，它是教师分布的归一化加权几何平均。若同一状态上两位教师等权，且它们给出 `p_A=(0.9,0.1)` 与 `p_B=(0.2,0.8)`，使目标最小的分布是 `(sqrt(0.18),sqrt(0.08))` 归一化后得到 `(0.6,0.4)`。直接对概率向量取算术平均则是 `(0.55,0.45)`；两者不是同一个目标。

这个两动作例子说明：当任务相关性不一样时，教师权重会改变学生最终偏好的分布；如果把所有教师不加区分地作用到所有状态，互相冲突的偏好可能被折中掉。V4 报告说明学生会按当前任务上下文选择相关能力，但没有披露具体路由器或冲突裁决规则。下节的 task label、verifier 和人工升级规则是工程设计建议，不是 V4 的已披露实现。

## 教师选择与冲突：报告披露了什么

报告说明，学生应结合当前任务上下文向相关教师学习；目标领域例子包括数学、代码、Agent 和指令跟随，OPD 阶段使用 10 个以上教师。公式中的 `w_i` 表示各教师的相对重要性。但报告没有给出可复现的教师路由器、具体权重表、权重是否随状态变化或无匹配教师时的回退规则。因此不能把下面的建议表说成 DeepSeek V4 的实现。[技术报告 §5.1.2](https://arxiv.org/html/2606.19348#S5.SS1)

如果工程团队要实现自己的多教师路由，可以把路由条件和可审计信号显式化：

| 训练状态 | 可选择的教师信号 | 需要检查的风险 |
|---|---|---|
| 明确代码任务 | 代码教师的 logits、patch、测试诊断 | 测试过拟合、忽略副作用、环境状态错配 |
| 数学/推理任务 | 数学教师分布、可验证答案或步骤批注 | 过程错而答案偶然正确、教师不确定性未表达 |
| 多工具长任务 | Agent 教师的下一步动作、状态摘要、artifact | 过期 observation、重复副作用、工具权限越界 |
| 跨领域任务 | 多教师候选、按状态赋权、独立 verifier | 错误路由、信号冲突、推理与存储成本上升 |
| 无匹配教师或教师失败 | 延后监督、通用教师或人工升级 | 错误迁移；若静默跳过则形成训练盲区 |

教师不能只根据用户最后一句话路由。相同文本可能处在不同 workspace、权限和工具状态；这些外部状态属于训练单元的一部分。训练日志应存储候选教师、最终权重、选择理由、模型 revision、教师失败原因和冲突记录。跨领域冲突应保留多个候选及其理由，再由独立 verifier、规则或人工裁决，而不是静默覆盖。

## 全词表蒸馏需要怎样的教师调度

全词表目标不是只保存“教师选中的下一个 token”。对每个学生前缀，训练要拿到教师对整个词表的分布；多教师还会把教师权重、教师推理和 logits 缓冲的成本乘上去。可以先估算 logits 本身的体积：假设一批有 8,192 个 token 位置、词表 100,000 项、每个 logit 用 BF16 两字节表示，那么单个教师 logits 是

```math
8{,}192\times100{,}000\times2
=1{,}638{,}400{,}000\text{ bytes}\approx1.53\text{ GiB}.
```

10 个教师同时保存就约 15.3 GiB；这还不包括教师权重、前向激活、学生梯度、优化器状态或轨迹数据。这是一个透明的内存账本示例，不是 V4 报告中的 batch、词表 dtype 或实测配置。

为什么缓存 hidden state 而不是 logits？对一个前缀 `s`，教师末层表示 `h_T(s)∈R^d` 经过词表输出头变成 logits：

```math
z_T(s)=W_{\mathrm{out},T}h_T(s)+b_T,
\qquad W_{\mathrm{out},T}\in\mathbb{R}^{|V|\times d}.
```

`h_T` 只有 `d` 个数，logits 有 `|V|` 个数。若二者采用同样 dtype 且 `|V|>d`，暂存 hidden state 比暂存 full-vocabulary logits 更省空间；代价是训练时必须重新加载输出头并计算完整 logits。因此该做法把“大量持久化 logits”的内存/存储压力转成“按需读输出头并重建”的计算和 I/O。注意 `z_T` 仍是未归一化分数，形成 KL 前要经 `softmax` 得到 `p_T`；学生侧的 logits 也要按同一词表位置形成 `p_S`。

DeepSeek V4 报告披露了针对这一瓶颈的系统设计：教师权重放在集中式分布存储中，做教师前向时再按需加载，并使用类似 ZeRO 的参数分片（把参数切片分散到多设备）降低单机 I/O 和 DRAM 压力；不把所有教师的 full-vocabulary logits 长期物化，而是集中缓存教师最后一层 hidden states，在学生训练时加载对应教师的输出头并即时重建 logits。数据派发按 teacher index 排序，使一个 mini-batch 只加载对应教师头，并让 GPU 上同时最多驻留一个教师输出头。权重和 hidden states 的加载/卸载异步执行，专用 TileLang GPU kernel 计算全词表 exact KL，并减少动态内存分配。报告还称其框架面向“有效无上限”的教师数设计，单个教师可能达到 trillion-parameter 规模；这是基础设施支持能力的描述，不能反推 V4 实际用了无上限数量或每位教师都达到该规模。[技术报告 §5.2.2](https://arxiv.org/html/2606.19348#S5.SS2)

这组优化分别打在三类瓶颈上：参数分片与按需读取处理“教师权重太大”；hidden-state 缓存替代长时间保留 `[token, vocab]` logits；按教师排序和异步传输减少反复换头与关键路径 I/O。它们让 full-vocabulary 多教师 OPD 可执行，但不消除教师前向 FLOPs、分布式存储吞吐、教师版本管理和采样状态成本。报告没有给出可用于通用容量规划的峰值显存或端到端加速比，实际系统仍应测教师调用量、logits 重建时间、I/O 等待和每个成功样本成本。

## 一个可审计的训练循环

下图把报告披露的两阶段关系与本章建议的审计点放在一起。教师路由、日志字段和评估门禁是实现建议；论文没有公开可逐步复现的全部训练脚本。

```text
领域任务/奖励 -> 专家 SFT -> 领域 rollout -> GRPO -> 多个领域教师
                                                   |
统一学生 -> 自己 rollout -> 学生前缀 -> 选择/加权相关教师 -> 全词表 logits
                                                   |               |
                                            加权 reverse KL <- exact KL kernel
                                                   |
                                             更新统一学生
                                                   |
                         独立的领域、跨领域、安全、恢复与成本评估
```

每一轮都应保存模型 revision、tokenizer、模板、环境镜像、工具 schema、奖励版本、verifier 版本、教师候选/权重、采样参数和资源消耗。否则统一模型分数变化时，无法判断是能力合并有效，还是任务、工具、教师路由或判定器发生了变化。

学生 rollout 不能只保留最终答案。至少要保留错误动作、工具 observation、工作区 diff、测试结果、权限决策和恢复动作。教师的最终 patch 可以作为结果监督，但没有失败上下文，学生仍然学不会如何从错误状态恢复。

## 工程成本与失败模式

两阶段流程会增加教师推理、轨迹存储、验证器执行和多领域调度成本。对多教师 full-vocabulary OPD，单位训练任务成本至少要拆开学生 rollout、每个教师的前向、logit 重建与 KL、验证和 I/O：

```math
C_{\mathrm{task}}
=C_{\mathrm{rollout}}
+\sum_{i=1}^{N}q_iC_{\mathrm{teacher},i}
+C_{\mathrm{logit/KL}}+C_{\mathrm{verifier}}+C_{\mathrm{I/O}}+C_{\mathrm{storage}}.
```

令每一项都按同一个“每训练任务”的成本单位计量：`C_rollout` 是学生生成轨迹的成本，`C_teacher,i` 是教师 `i` 每处理一个 token/状态的前向成本，`C_logit/KL` 是输出头重建与 exact KL 计算成本，`C_verifier` 是验证成本，`C_I/O` 是参数/hidden-state/轨迹传输成本，`C_storage` 是持久化成本。`q_i` 是一个训练样本或 batch 中教师 `i` 实际处理的 token/状态数；它由教师覆盖、采样策略和数据路由决定。这个表达提醒我们：全词表方案省去的主要是 sampled-token 近似带来的动作采样噪声，并没有省去教师前向、输出头计算或分布式读写。

若实现团队需要降低成本，可把“按需调用教师”“先查轻量 verifier”“仅在失败/低置信状态请求更强教师”作为实验假设。但这会改变哪些学生状态获得蒸馏监督，必须同时测能力和覆盖率；不能只看教师 token 数下降就断言效率提高。

常见失败模式包括：

- **领域隔离过强**：专家在自己的 benchmark 很强，跨领域任务互相干扰。
- **教师冲突**：不同教师对同一状态给出不同动作；加权 reverse-KL 会形成按权重决定的分布目标，教师/学生/验证器可能都不再偏好最终折中。
- **验证器漏洞**：GRPO 优化了检查器捷径，而不是任务目标。
- **状态版本漂移**：教师和学生看到的模板、工具或 workspace 不一致，KL 信号失去语义。
- **教师覆盖过宽**：多余教师调用增加前向与 I/O 成本；如果能力提升没有跟上，单位有效样本成本会上升。
- **长轨迹信用分配困难**：只在结尾给 reward，早期错误很难定位；过程奖励又可能引入捷径。

## 代表性方案解决了什么，各自牺牲什么

“把多个专家合成一个模型”不是一个单一算法。下面按发生层次比较几种代表方案，避免把参数合并、离线模仿、在线蒸馏和推理时路由混为一谈：

| 方案 | 它解决的问题与机制 | 证据中实际改善了什么 | 主要限制，以及适用条件 |
|---|---|---|---|
| 保留多个模型，推理时路由/集成 | 每个领域保留各自 checkpoint，运行时按任务选择或组合 | 避免训练时把领域参数提前压到一份模型里；领域行为可独立升级 | 需要在线路由、多个模型常驻或重复推理，增加延迟、显存、故障面；如果选错教师，结果仍会错。它是 serving 架构选项，不是 V4 报告披露的后训练做法 |
| 权重平均 / Model Soups | 对兼容模型的参数做平均，目标是在不增加推理模型数的情况下得到一个权重集 | Model Soups 在共享预训练模型的 fine-tuning 实验中报告准确率与分布外鲁棒性改善，且没有额外推理/内存成本 | 原论文研究的是特定预训练模型的 fine-tuned checkpoints；参数坐标和低损失区域条件很重要。不同领域 RL 专家可能存在功能冲突，不能把该结果当作 V4 多专家合并已成功的证据 |
| 离线 response distillation / SFT | 教师先为固定 prompts 生成答案，学生对答案做行为克隆 | 数据易缓存、训练可批量化，教师不必在每个学生状态上实时响应 | 学生偏离示范前缀后可能暴露分布错位；教师回答也只告诉学生一个/少数目标序列，不一定保留完整词表排序 |
| GKD / 学生状态上的条件蒸馏 | 学生先生成序列，再在自身前缀上让教师提供反馈；可选择 forward KL、reverse KL 或 JSD 等局部目标 | GKD 在摘要、翻译、算术推理和 instruction distillation 任务中报告 on-policy 数据改善；具体收益绑定其 T5 模型和任务设置 | 任务、学生容量和 divergence 会影响结果；局部条件分布拟合不自动等于完整序列目标，也不证明多教师超大模型系统可扩展 |
| MiniLLM / sequence-level reverse KL | 用学生序列分布拟合教师，并用 on-policy 优化处理生成分布与 exposure bias | 作者报告在其 120M–13B 指令跟随模型设置中改善输出质量、长文本和校准等指标 | 序列级估计涉及长轨迹信用分配与采样方差；实验范围不是 V4 的 10+ 专家、多万亿参数教师配置 |
| DeepSeek V4 报告中的 full-vocabulary multi-teacher OPD | 学生在自身前缀上同时向相关专家教师拟合完整词表分布；用 reverse KL 把教师行为信号写入统一学生 | 报告称使用 10 个以上领域教师，选择 full-vocabulary logit distillation，并公开了教师权重分片、hidden-state 缓存和输出头调度设计 | 教师路由/权重细节和与 sampled OPD 的受控消融未公开；教师前向与 full-vocabulary KL 成本仍高。报告支持“采用了这套方法”，尚不足以单独归因最终 benchmark 增益 |

这几种方法可以串联：离线 SFT 可作初始化，领域 GRPO 先提高教师专长，OPD 再训练学生处理自己的状态；推理时仍可用模型内部 MoE 承担 token 级稀疏计算。真正要比较时，应固定底模和数据，至少做领域 SFT、SFT+GRPO、response distillation、sampled-token OPD、full-vocabulary OPD 的消融，并同时报告各域质量、跨域任务、能力回归、每个有效样本所需教师 FLOPs、I/O、显存和失败恢复。以上是建议的实验矩阵，不是 V4 报告已经完成的全部对照。

根据这些限制，可以提出下面的改进实验。它们是从失败机制推导的研究假设，不能说成 DeepSeek 已采用或已验证的后续方案：

| 已识别瓶颈 | 可检验的改进假设 | 至少要测什么 | 新风险 |
|---|---|---|---|
| 多教师偏好冲突、路由算法未公开 | 用任务标签/状态分类器生成稀疏教师权重；教师不确定或相互冲突时转 verifier/人工 | 路由混淆矩阵、分域成绩、跨域失败、教师覆盖率 | router 错误会把学生带向错误 teacher；人工样本会改变数据分布 |
| 全词表教师 forward 与读写成本高 | 仅在学生失败、低置信或高风险状态调用强教师；预取下一 mini-batch 所需教师参数，复用带 revision key 的 hidden states | 每个有效任务的教师 FLOPs、I/O 等待、显存峰值、成功率和分域覆盖 | 节省 token/算力可能减少长尾状态监督；缓存旧 revision 会造成错监督 |
| GRM 与 rubric 可能相互强化偏差 | 对 GRM 做独立人评校准、隐藏 rubric 变体、对抗式评分与跨领域负例 | 人评相关性/一致性、排序稳定性、分数涨而事实/任务正确性下降的比例 | 评测集泄漏、GRM 与人评共享偏差、把评分协议过拟合成新的 reward hack |
| 学生 rollout 来自旧 checkpoint 或工具环境 | 绑定 student revision / tokenizer / tool schema，限制 policy lag 并按版本分层复用 rollout | 行为策略与训练策略 log-ratio、陈旧轨迹比例、工具 observation 回放差异 | 更频繁重采样提高成本；过严过滤会降低有效数据量 |
| 只拟合教师分布并不保证真实任务结果 | 让 OPD 与独立 outcome/verifier 指标共同进入阶段性 gate，并做“仅 OPD / 仅 outcome RL / 组合”消融 | 结果正确、恢复、安全、跨域泛化、单位成功成本 | 多目标梯度冲突；verifier 错误重新进入优化环 |

代表来源：[GKD](https://arxiv.org/abs/2306.13649)、[MiniLLM](https://arxiv.org/abs/2306.08543)、[Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html) 与 [DeepSeek V4 技术报告](https://arxiv.org/abs/2606.19348)。这些实验来自不同模型、数据和任务，表中比较的是机制与证据范围，不是横向分数排名。

## 面试追问

**问：为什么不直接把所有领域数据混合训练？** 如果不同领域的 reward、工具状态和成功标准发生冲突，统一混训可能相互干扰；先培养领域能力再蒸馏是一种处理路线，但报告没有给出与“全程混合训练”完全受控的对照，因此不能断言它一定更好。

**问：GRPO 的组内优势解决什么问题？** 它用同一任务的一组样本做相对比较，降低跨 prompt 奖励尺度差异，并可减少独立 value model 依赖；它仍然依赖可靠 reward 和 verifier。

**问：on-policy distillation 和 RL 有什么区别？** RL 直接根据奖励更新策略；on-policy distillation 让学生在自己的状态上匹配教师分布或动作。两者都需要 rollout，但优化信号来源不同，可以串联使用。

**问：领域专家会在推理时一起加载吗？** 报告中的领域专家是后训练阶段的完整 teacher checkpoints；它们不同于 MoE 层内的 routed experts。V4 最终模型仍是 MoE，但不能据此推断推理服务还要逐请求加载 10 多个领域教师。

**问：如何证明统一模型真的学会了能力？** 使用未见过的跨领域任务和真实环境回放，固定模型与工具版本，分别测结果正确性、权限合规、恢复能力、token/工具成本和教师调用率。

**问：V4 报告说 OPD 替代了 mixed RL，是否意味着后训练不再用 RL？** 不是。领域教师先用 SFT 和 GRPO 训练；OPD 替换的是后续统一学生阶段的 mixed-RL 路线。

**问：为什么采用 full-vocabulary KL？它解决了什么、没有解决什么？** 它在每个学生前缀上对完整词表求和，避免只抽一个 token 造成的局部估计噪声；学生前缀仍靠 rollout 采样，教师质量、state coverage、路由和教师前向成本仍需单独处理。

**问：如何在 10 多个大教师之间调度？** V4 报告描述按需加载分片参数、集中缓存末层 hidden state、训练时加载对应输出头、按 teacher index 排序和异步读写；这些是作者公开的基础设施措施，不是通用成本保证。

## 小练习

1. 为代码、数学和安全三个领域设计独立 task schema、verifier 和权限策略。
2. 生成同一个学生状态下的两个教师分布，计算 forward KL 与 reverse KL 的差异。
3. 实现一个组内标准化 reward，加入 verifier unknown 和超时样本的隔离统计。
4. 设计领域专家到统一学生的 on-policy trace schema，至少记录状态、动作、工具 observation、教师版本和 verifier 版本。
5. 做消融：仅领域 SFT、SFT+GRPO、离线蒸馏、on-policy distillation，在跨领域任务和单位成功成本上比较。
6. 用 `p_S=(0.60,0.30,0.10)` 与 `p_T=(0.70,0.20,0.10)`，手算 full-vocabulary reverse KL 和单 token log-ratio 的标准差；指出精确求和消除的误差来源及仍存在的 rollout 状态差异。
7. 对两位教师分布推导 weighted reverse-KL 的归一化几何平均，并设计一个能检测错误 teacher routing 的跨领域测试。
8. 估算全词表 logits 缓冲，按 V4 报告思路画出“教师参数/hidden state 读取 → 输出头重建 → exact KL”的数据流，再注明报告没有公开的峰值显存和 I/O 指标。
9. 比较 Model Soups、GKD、MiniLLM 与 V4 OPD 的合并对象和成立条件，列一项原论文没有验证、但下一步值得做的控制变量实验。
