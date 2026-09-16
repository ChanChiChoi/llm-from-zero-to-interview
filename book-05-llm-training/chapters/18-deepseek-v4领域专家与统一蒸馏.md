# DeepSeek V4 后训练流水线：领域专家培养与统一蒸馏

> 资料来源：DeepSeek V4 官方模型卡与技术报告 `arXiv:2606.19348`（核验日期 2026-09-09）。报告公开描述的流程是：先独立培养领域专家，再通过 on-policy distillation 将不同能力合并到统一模型。本章解释这个流程为什么存在、如何设计训练契约，以及哪些结论不能从发布摘要直接推出。

## 一个模型为什么需要“先分开练，再合在一起”

想象一个新人同时学习数学证明、代码调试、写作和网络安全。让他每天把所有任务混在一张试卷上，确实简单，却可能出现两个问题：他在某一领域还没形成稳定策略，就被其他领域的反馈打断；不同领域的成功标准也可能互相冲突。另一种办法是先让他在每个领域分别跟一位教练训练，形成几种相对成熟的能力，再让一位总教练把这些能力合并成一套可部署的工作方式。

DeepSeek V4 技术报告把后训练描述为类似的两阶段流程：第一阶段对领域专家分别进行 SFT 和使用领域奖励的 GRPO；第二阶段训练一个统一学生模型，让它在自己的 rollout 状态上学习这些教师的行为。这种描述解释了“为什么要拆分能力来源”，但不等于推理时会加载多个领域模型，也不等于报告公开了所有数据配比和路由细节。

## 第一阶段：领域专家培养

设领域集合为 `D={d_1,...,d_K}`。每个领域都有自己的任务分布、数据、工具和 verifier。对于领域 `d_k`，先用监督微调教会模型基本协议和示范流程：

```math
\mathcal{L}_{\mathrm{SFT}}^{(k)}
=-\frac{1}{M_k}\sum_t m_t\log p_\theta(y_t\mid x,y_{1:t-1}),
\qquad M_k=\sum_t m_t>0.
```

随后在可验证或可排序的领域环境中进行 GRPO。GRPO（Group Relative Policy Optimization）通常对同一个 prompt 采样一组回答，利用组内相对奖励构造优势，减少对独立 value model 的依赖。教学化地写成：

```math
A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}
{\mathrm{std}(r_{1:G})+\epsilon},
```

其中 `G` 是同一任务采样的回答数，`r_i` 是 verifier、奖励模型或规则组合得到的奖励，`A_i` 是组内相对优势。真实实现还包含 clipping、KL 约束、mask、长度处理和 rollout 过滤；上式只说明“与同组样本比较”的核心直觉。

组内归一化能减少不同 prompt 的绝对奖励尺度差异，但它不是奖励正确性的证明。如果一组回答全部利用了 verifier 漏洞，最高的那个仍可能只是“漏洞利用得最好”。因此每个领域都需要独立检查器、反作弊测试、隐藏任务和人工抽样。

下面的零依赖代码只演示组内标准化，不包含策略梯度、clipping、KL 约束或 rollout 服务。它的用途是让读者看到：优势表示的是同一任务组内“相对好多少”，不是跨任务可直接比较的绝对分数。

```python
import math


def group_advantages(rewards, eps=1e-8):
    if not rewards:
        raise ValueError("a group must contain at least one reward")
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance)
    return [(r - mean) / (std + eps) for r in rewards]


group = [0.2, 0.8, 0.5, 0.1]
advantages = group_advantages(group)
print("advantages:", [round(x, 3) for x in advantages])
print("mean advantage:", round(sum(advantages) / len(advantages), 6))
```

当组内奖励不全相同，优势的平均值应接近 0。若所有奖励都相同，标准差为 0，代码仍会返回全 0；这表示当前 verifier 没有提供组内区分度，而不是模型已经学会任务。生产训练还要单独记录 reward variance、全同组比例、verifier unknown 比例和过滤率。

## 为什么领域奖励要和权限分开

代码领域的测试通过、数学领域的答案正确、网络安全领域的漏洞发现都可以成为质量信号，但它们不能自动授予工具权限。一个模型可能生成了通过单元测试的 patch，却试图读取不必要的 secret；一个安全研究任务可能在模拟环境中允许 exploit，但生产系统不允许同样的动作。

训练记录应将结果质量和策略约束分开：

```math
S=C_{\mathrm{correct}}\cdot C_{\mathrm{format}}
\cdot P_{\mathrm{permission}}\cdot B_{\mathrm{budget}}.
```

这里的乘法表达“任一硬条件失败都不能称为完整成功”的语义。若权限或预算状态未知，不能强行填 0 或 1；应该保留 `unknown`，修复环境或隔离该轨迹。奖励高不能抵消越权、隐私泄露或未授权副作用。

## 第二阶段：统一模型的 on-policy distillation

领域专家拥有不同的分布和行为风格。直接把它们的离线答案拼起来，统一学生只会在教师走过的干净前缀上模仿；学生上线时产生的错误状态和工具 observation 可能完全不同。

统一蒸馏让学生先 rollout，再让一个或多个领域教师在这些状态上提供 token 分布、下一步动作、错误批注、候选答案或最终 artifact。一个训练单元可以记录为：

```math
u=(x,s_t,a_t,o_{t+1},y,u_T,v,\rho_e),
```

其中 `s_t` 是学生状态，`a_t` 是动作，`o_{t+1}` 是环境 observation，`y` 是最终结果，`u_T` 是教师信号，`v` 是 verifier，`\rho_e` 是环境和协议版本。

报告源码称统一学生通过 on-policy distillation 使用 reverse KL 方向整合教师能力。对于学生分布 `p_S` 和教师分布 `p_T`：

```math
D_{\mathrm{KL}}(p_S\|p_T)
=\sum_v p_S(v\mid s)\log\frac{p_S(v\mid s)}{p_T(v\mid s)}.
```

如果学生在自己的状态上采样了动作 `a_t`，一个 token 级的教师差异信号可以写成：

```math
r_t^{\mathrm{KD}}
=\log p_T(a_t\mid s_t)-\log p_S(a_t\mid s_t).
```

这不是报告中完整的训练损失，而是帮助理解 reverse-KL 方向的教学表达。它要求教师和学生在同一状态、动作定义、tokenizer 和模板下计算概率；服务失败、教师拒答或版本不匹配时，不能把缺失 log-prob 当成极低奖励。

## 多个领域教师如何路由

统一学生遇到一个状态时，可能有多个领域教师都能提供信号，也可能没有一个教师真正适合。可以按任务标签、工具类型、verifier 失败类别和不确定性触发教师选择，但路由本身需要记录版本和理由：

| 路由情况 | 可用信号 | 主要风险 |
|---|---|---|
| 明确代码任务 | 代码教师 logits、patch、测试诊断 | 过拟合公开测试或忽略副作用 |
| 数学/推理任务 | 可验证答案、步骤批注 | 教师过程错误、答案格式偏置 |
| 多工具长任务 | 下一步动作、状态摘要、artifact | 工具状态过期、重复副作用 |
| 跨领域任务 | 多教师候选与独立 verifier | 冲突、重复调用和成本爆炸 |
| 无匹配教师 | 通用教师或人工升级 | 错误迁移、覆盖不足 |

教师不能只根据用户最后一句话路由。相同文本可能处在不同 workspace、权限和工具状态；这些外部状态属于训练单元的一部分。跨领域冲突应保留多个候选和冲突原因，再由 verifier、规则或人工裁决，而不是悄悄覆盖。

## 一个可审计的训练循环

```text
领域数据 -> SFT 专家 -> 领域 rollout -> verifier/奖励 -> GRPO 专家
                                                   |
学生 rollout -> 选择教师 -> 教师信号 -> verifier -> on-policy distillation
                                                   |
                                   独立跨领域评估与回归
```

每一轮都应保存模型 revision、tokenizer、模板、环境镜像、工具 schema、奖励版本、verifier 版本、采样参数和资源消耗。否则统一模型分数变化时，无法判断是能力合并有效，还是任务、工具或判定器发生了变化。

学生 rollout 不能只保留最终答案。至少要保留错误动作、工具 observation、工作区 diff、测试结果、权限决策和恢复动作。教师的最终 patch 可以作为结果监督，但没有失败上下文，学生仍然学不会如何从错误状态恢复。

## 工程成本与失败模式

两阶段流程会增加教师推理、轨迹存储、验证器执行和多领域调度成本。粗略地，单位任务成本可以写成：

```math
C_{\mathrm{task}}
=C_{\mathrm{student}}+p_T C_{\mathrm{teacher}}
+C_{\mathrm{verifier}}+C_{\mathrm{storage}}.
```

`p_T` 是触发教师调用的比例。不是每个状态都值得调用最昂贵的教师：低风险、已通过验证的状态可以只保留学生监督；连续失败、工具冲突、高风险动作或跨领域切换更值得获得教师信号。

常见失败模式包括：

- **领域隔离过强**：专家在自己的 benchmark 很强，跨领域任务互相干扰。
- **教师冲突**：不同教师对同一状态给出不同动作，统一学生学到不可预测的折中。
- **验证器漏洞**：GRPO 优化了检查器捷径，而不是任务目标。
- **状态版本漂移**：教师和学生看到的模板、工具或 workspace 不一致，KL 信号失去语义。
- **教师调用过多**：训练质量提高但单位成功成本和延迟不可接受。
- **长轨迹信用分配困难**：只在结尾给 reward，早期错误很难定位；过程奖励又可能引入捷径。

## 与已有训练方法的关系

- **领域 SFT** 教基本语言、格式和工作流。
- **RLVR/GRPO** 在可验证环境中优化领域结果和行为。
- **on-policy distillation** 让统一学生在自己的状态分布上吸收教师信号。
- **普通离线蒸馏** 更适合固定参考数据和低成本批量训练，但无法充分覆盖学生偏离状态。

它们不是互斥选项：领域专家可以先 SFT 再 GRPO，统一学生再做 on-policy distillation，最后还需要独立评估和安全回归。DeepSeek V4 报告披露了这样的高层流程，但没有公开每个领域、奖励权重、教师数量和完整 loss，不能擅自补齐。

## 面试追问

**问：为什么不直接把所有领域数据混合训练？** 混合训练简单，但不同领域的状态分布、成功标准和工具约束可能相互干扰；先培养领域能力，再在学生自身状态上统一蒸馏，有机会减少冲突，但成本和教师质量控制更复杂。

**问：GRPO 的组内优势解决什么问题？** 它用同一任务的一组样本做相对比较，降低跨 prompt 奖励尺度差异，并可减少独立 value model 依赖；它仍然依赖可靠 reward 和 verifier。

**问：on-policy distillation 和 RL 有什么区别？** RL 直接根据奖励更新策略；on-policy distillation 让学生在自己的状态上匹配教师分布或动作。两者都需要 rollout，但优化信号来源不同，可以串联使用。

**问：领域专家会在推理时一起加载吗？** 报告的“专家”指后训练阶段的领域能力模型/教师，不能据此推断部署时采用多模型路由或 MoE expert。

**问：如何证明统一模型真的学会了能力？** 使用未见过的跨领域任务和真实环境回放，固定模型与工具版本，分别测结果正确性、权限合规、恢复能力、token/工具成本和教师调用率。

## 小练习

1. 为代码、数学和安全三个领域设计独立 task schema、verifier 和权限策略。
2. 生成同一个学生状态下的两个教师分布，计算 forward KL 与 reverse KL 的差异。
3. 实现一个组内标准化 reward，加入 verifier unknown 和超时样本的隔离统计。
4. 设计领域专家到统一学生的 on-policy trace schema，至少记录状态、动作、工具 observation、教师版本和 verifier 版本。
5. 做消融：仅领域 SFT、SFT+GRPO、离线蒸馏、on-policy distillation，在跨领域任务和单位成功成本上比较。
