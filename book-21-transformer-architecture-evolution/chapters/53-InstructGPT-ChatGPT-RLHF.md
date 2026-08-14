# 第 53 章 InstructGPT、ChatGPT 与 RLHF：后训练如何改变模型的使用方式

## 53.1 预训练模型为什么还不是助手

预训练目标是预测互联网或其他语料中的下一个 token。它会让模型学习语言、事实、代码和对话形式，但不会自动告诉模型：用户真正的问题是什么、什么时候应该拒答、如何遵守格式、如何承认不确定、怎样调用工具或怎样把答案写得适合人类阅读。

一个 base model 可能续写一段问题，也可能给出多个互相矛盾的答案；它优化的是数据分布中的 token 概率，不是“让这个用户满意且安全地完成任务”。InstructGPT 的重要影响，是把“指令遵循和人类偏好”变成一个系统性的后训练问题。

## 53.2 三阶段的基本流程

经典 RLHF 流程可以分成三步。

第一步是监督微调。收集人工编写或筛选的指令—回答样本，用监督损失让预训练模型学会回答格式和任务行为。

第二步是训练奖励模型。对同一个 prompt 生成多个回答，让标注者比较偏好顺序，训练奖励模型使更受偏好的回答得到更高分。

第三步是强化学习。把语言模型作为策略，用奖励模型的分数和约束优化生成策略，常见方法是 PPO。

流程可以写成：

~~~text
pretrained model
  -> supervised fine-tuning
  -> preference data and reward model
  -> policy optimization
  -> safety and capability evaluation
~~~

这不是唯一的后训练方法，但它解释了为什么同一组预训练权重经过后训练后，交互行为会发生明显变化。

## 53.3 SFT 的目标

对 prompt x 和示范回答 y，SFT 通常最小化回答 token 的条件负对数似然：

~~~math
\mathcal{L}_{\mathrm{SFT}}
=-\sum_{t=1}^{T_y}
\log\pi_\theta(y_t\mid x,y_{1:t-1})
~~~

工程上常只对 assistant 部分计算 loss，把用户问题、系统消息和分隔符作为条件。模板、角色标记、EOS 处理和 loss mask 必须一致；否则模型可能学会复述 prompt、错误停止或把工具结果当作用户指令。

SFT 的优势是稳定、容易调试、可以直接注入格式；局限是示范数据覆盖有限，模型可能过拟合标注者风格，也可能在没有示范的边界场景中表现不稳。

## 53.4 偏好数据和奖励模型

给定 prompt x、两个回答 y+ 和 y−，奖励模型 rφ 可以使用 Bradley-Terry 形式：

~~~math
\mathcal{L}_{\mathrm{RM}}
=-\log\sigma\left(r_\phi(x,y^+)-r_\phi(x,y^-)\right)
~~~

它学习的是相对偏好，不是绝对真值。标注者可能更喜欢流畅而不是事实正确的答案，也可能受到回答长度、语气和格式的影响。奖励模型因此会形成自己的偏差和盲点。

奖励模型训练应保存标注协议、样本来源、比较难度、标注一致性和领域覆盖。只报告一个 reward accuracy，不能证明它能判断代码正确性、复杂推理或安全边界。

## 53.5 PPO 的约束直觉

策略优化希望提高高奖励回答的概率，但不能让策略偏离参考模型太远。一个简化目标可以写成：

~~~math
\max_\theta\;
\mathbb{E}_{y\sim\pi_\theta}
\left[r_\phi(x,y)
-\beta\,\mathrm{KL}
\left(\pi_\theta\Vert\pi_{\mathrm{ref}}\right)\right]
~~~

KL 惩罚用来抑制策略为了奖励模型而产生剧烈漂移。真实 PPO 还包含 value model、advantage、clipping、batch 和多轮更新。阅读论文时要把教学式 KL 目标和完整实现区分开。

如果奖励模型存在漏洞，策略可能找到高分但不符合人类真正目标的输出，这就是 reward hacking 或 specification gaming 的一个来源。

## 53.6 ChatGPT 让系统能力成为产品的一部分

ChatGPT 的影响不只来自基础模型架构，还来自对话模板、上下文管理、拒答策略、内容安全、流式输出、工具和产品反馈。一个用户看到的是系统，而不是一组未经包装的模型权重。

因此比较 chat model 时要区分：

1. base model 的语言和推理能力。
2. instruct/SFT 后的格式和遵循能力。
3. preference/RL 后的帮助性和拒答风格。
4. 产品系统的检索、工具、上下文、限流和安全层。

如果只比较最终网页体验，不能反推内部架构；如果只比较裸模型 benchmark，也不能说明实际助手体验。

## 53.7 RLHF 的奖励错配

设真实任务效用为 U，奖励模型给出的分数为 R。理想情况下二者相关，但实际可能存在：

~~~math
R(y)=U(y)+\epsilon_{\mathrm{label}}(y)
\qquad\text{且}\qquad
\epsilon_{\mathrm{label}}\not\perp y
~~~

误差不是纯随机噪声，而可能偏爱更长、更自信、更礼貌或更像标注样本的回答。策略优化会主动放大可利用的偏差。

常见表现包括：回答越来越冗长但信息密度下降；模型过度拒答；模型为了显得确定而减少不确定性表达；安全规则覆盖的场景变多，但边界外的真实风险没有改善。

## 53.8 直接偏好优化和 RLHF 的关系

DPO 等方法直接使用偏好对训练策略，不显式训练在线奖励模型和 PPO 循环。一个常见教学目标是比较策略与参考策略在 chosen/rejected 回答上的 log-prob 差异：

~~~math
\mathcal{L}_{\mathrm{DPO}}
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}
{\pi_{\mathrm{ref}}(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}
{\pi_{\mathrm{ref}}(y^-\mid x)}
\right]\right)
~~~

DPO 简化了工程流程，但没有消除偏好数据质量、分布外任务和奖励代理的问题。不同后训练方法的比较应固定基座、数据、训练 token 和评估集。

## 53.9 推理时的对齐约束

对齐不仅发生在训练中。推理系统还可能使用 system prompt、内容分类器、工具权限、输出过滤、置信度验收条件和人工审核。模型说“我已经完成”并不代表外部动作已经执行；安全系统要验证事实。

可以把最终动作是否允许写成：

~~~math
\mathrm{Allow}(a)
=\mathrm{Policy}(identity,resource,action,risk)
\land \mathrm{SchemaValid}(a)
\land \mathrm{Verified}(a)
~~~

这说明模型对齐、执行器安全和产品政策是不同层次。只依赖语言模型自身拒答，无法处理所有副作用和权限问题。

## 53.10 一个最小偏好训练实验

下面的代码仅演示如何计算两个回答的 log probability 差异，帮助理解偏好损失的数据接口：

~~~python
import math


def sequence_logprob(token_logprobs):
    return sum(token_logprobs)


chosen = sequence_logprob([-0.2, -0.4, -0.1])
rejected = sequence_logprob([-0.8, -0.7, -0.5])
margin = chosen - rejected
preference_score = 1.0 / (1.0 + math.exp(-margin))
print({"margin": margin, "score": preference_score})
~~~

真实训练还要考虑 response mask、长度归一化、reference model、padding、batch 和数值稳定。这个例子不能替代完整 DPO 或 PPO 实现。

## 53.11 如何评估后训练是否真的有用

评估至少包括：

- 指令遵循：是否理解任务目标和输出格式。
- 事实与推理：答案是否正确，是否会承认不确定。
- 有害请求：拒答是否准确，安全建议是否仍然有用。
- 过度拒答：合法低风险请求是否被错误拒绝。
- 偏好稳定性：长度、语气和格式变化是否改变判断。
- 工具行为：参数是否正确，失败后是否停止或恢复。
- 回归：SFT/RL 后的代码、数学、多语言和长上下文能力是否下降。

最好保存 base、SFT、preference 和 final policy 的并列结果。只给最终模型一个总分，无法知道哪一阶段带来了收益或退化。

## 53.12 常见失败模式

第一，SFT 数据中 system/user/assistant 模板不一致。第二，奖励模型把长度和礼貌当成质量。第三，PPO 更新过大造成能力回退。第四，偏好数据覆盖不了高风险边界。第五，模型学会说“已完成”但没有实际执行。第六，安全拒答过宽，正常用户任务被阻断。第七，离线评估通过，线上真实分布却触发奖励漏洞。

这些失败要求同时检查数据、目标函数、策略更新、执行器和线上反馈，不能只调整 temperature。

## 53.13 为什么偏好优化不能替代事实评估

RLHF 或其他偏好优化提高的通常是人类偏好代理，例如帮助性、清晰度、格式和拒答风格。事实正确、代码可执行、引用支持和工具副作用需要独立评估。一个更流畅的回答可能更受标注者喜欢，却不一定更真实。

训练后应保留 capability、truthfulness、safety、format 和 latency 五类验收条件。若只看 reward 或人工总体满意度，模型可能通过变长、套话、过度拒答或迎合用户来提高代理分数。

## 53.14 SFT、RM 和策略优化的接口

三个阶段的数据契约不同。SFT 需要高质量示范和 assistant loss mask；RM 需要同 prompt 下可比较的回答、标注协议和一致性；策略优化需要 rollout、奖励、参考策略、KL 或其他稳定性约束。数据格式不一致时，训练异常可能被误判为算法问题。

可把总行为变化拆成：

~~~math
\Delta \mathrm{Behavior}
\approx \Delta_{\mathrm{SFT}}
\,+\Delta_{\mathrm{Preference}}
\,+\Delta_{\mathrm{Safety}}
\,+\Delta_{\mathrm{Serving}}
~~~

这不是严格可加分解，而是排查框架。模板、system policy 和 runtime 工具也会改变最终行为，不能把所有变化归因于 PPO。

## 53.15 RLHF 的历史位置与后来者

RLHF 证明了“让模型遵循人类意图”可以成为系统化训练流程，也暴露了奖励建模、标注成本、训练不稳定和 reward hacking。DPO、IPO、KTO、ORPO、SimPO 等方法尝试减少显式在线 RL 或调整偏好目标，但它们仍依赖偏好数据质量和独立安全评估。

理解演进时要问：方法是否需要 reference model、reward model、在线 rollout、chosen/rejected 对，是否容易控制 KL，是否能处理多目标和安全边界。算法名字不是选择理由，数据与工程约束才是。

## 53.16 RLHF 是一条数据和模型的闭环

经典 RLHF 路线可以拆成监督微调、偏好数据与 reward model、策略优化、独立评估和安全回归。每一步都可能改变行为：SFT 提供任务格式，reward model 学人类偏好，RL 优化可获得的奖励，评估检查帮助性、事实性和安全是否失衡。

偏好奖励不是事实验证器。人类可能偏好流畅、详细和自信的回答，但这些属性不保证正确；reward model 还可能被冗长、格式和迎合行为欺骗。训练中需要保留拒答、事实性、长度、工具和高风险任务的独立切片。

## 53.17 对齐训练与产品协议

Instruct 模型输出自然语言，ChatGPT 类系统还需要 system/developer/user 角色、工具、记忆、流式事件和安全策略。对齐数据若没有覆盖真实协议，模型可能在单轮问答上很好，却在工具参数、拒答边界、长会话和多租户场景出错。

评估应包含 pairwise preference、任务成功、引用支持、格式通过、工具副作用、越权和用户纠正后的恢复。产品行为不能只由 reward model 的平均分决定。

## 53.18 资料范围与演进判断

InstructGPT 论文公开了基于人类反馈的指令微调、奖励模型和 PPO 流程；ChatGPT 是产品系统，公开信息不足以还原全部内部训练和服务细节。DPO、RLAIF、Constitutional AI 等方法属于后续相关路线，不能把它们混写成同一种 RLHF 实现。

后训练改变的是模型的行为分布和使用接口，而不是简单增加参数。理解它的关键，是把示范、偏好、奖励代理、策略优化、拒答政策和执行验证放在同一条证据链上。

## 53.19 面试问题与练习

**问：为什么 SFT 后还要做 RLHF？**

SFT 让模型模仿示范，RLHF 用大量相对偏好约束回答风格、帮助性和安全边界；但 RLHF 依赖奖励代理，可能产生奖励错配和过度拒答，需要独立评估。

**问：为什么不能把 reward model 当作真理？**

它从有限标注和偏好协议中学习，可能偏爱长度、语气或表面格式；策略还会主动寻找它的漏洞。代码测试、事实核验和安全红队需要独立于奖励模型。

**练习：**为一个客服助手设计 base/SFT/preference/final 四阶段评估表，至少包含指令遵循、事实性、拒答、过度拒答和工具参数正确性。

资料入口：

- InstructGPT: https://arxiv.org/abs/2203.02155
- DPO: https://arxiv.org/abs/2305.18290
- Constitutional AI: https://arxiv.org/abs/2212.08073

## 53.20 后训练的代理目标链

SFT、reward model 和 policy optimization 各自只观察到行为的一部分。SFT 看到示范，RM 看到偏好对，策略优化看到奖励和 KL 约束，产品评估还要看到事实、工具、副作用和安全。若把某一层的分数当成最终目标，模型就可能通过长度、语气、拒答或格式等表面信号获得奖励。

可以为每个阶段保存同一组 golden prompts，并追踪 helpfulness、factuality、over-refusal、tool correctness、safety 和 cost。若 SFT 提升格式但损害事实，或 RLHF 提升偏好却增加过度拒答，应保留阶段性 checkpoint 和失败样本，不能只发布最终模型。独立 verifier、代码测试和红队数据是防止 reward hacking 的必要补偿。

## 53.21 一个从 base 到 policy 的阶段性审计

后训练最容易被误判的地方，是只比较最终模型。更有信息量的做法是保存 base、SFT、preference 或 RM、policy 四个阶段，并在同一批 prompt 上重复评测。每条样本同时记录回答质量、事实支持、格式通过、拒答边界、工具参数、长度、延迟和成本。

设某阶段 `k` 的任务成功率为 `A_k`，事实支持率为 `F_k`，过度拒答率为 `O_k`，危险请求漏放率为 `U_k`，则阶段变化可以写成：

```math
\Delta_k=(A_k-A_{k-1},\;F_k-F_{k-1},\;O_k-O_{k-1},\;U_k-U_{k-1}).
```

这个向量比一个 reward 平均值更有诊断价值。例如 SFT 可能提高格式但降低长答案事实性；偏好优化可能提高礼貌和帮助性，却提高过度拒答；策略优化可能提高某个 reward 集，却使工具参数错误率上升。阶段性 checkpoint 让团队知道变化发生在哪里。

## 53.22 偏好数据不是“多数票真理”

偏好标签回答的是“在当前标注说明和比较条件下，哪个回答更符合偏好”。它不是一个自动事实检查器，也不是跨领域稳定的质量尺度。标注者可能因为回答更长、更有条理、更自信而选择它；当任务需要代码执行、数字核验或引用支持时，表面质量与真实质量可能分离。

因此偏好数据应按任务风险分层。低风险写作可以主要观察帮助性和风格；代码需要测试通过率；数学需要独立 verifier；知识问答需要引用支持；安全任务需要漏放率、过度拒答率和攻击变体覆盖。每个偏好对都应保留 prompt、chosen/rejected、标注说明、领域、难度、长度和是否有外部真值。

还要检查长度偏差。若奖励模型倾向于选择更长答案，可以把同一语义压缩成不同长度的 pair，测得长度条件下的选择率；也可以在 RM 输入中显式加入长度分桶，避免把冗长当成质量。发现偏差后，不能只在 PPO 中加惩罚，还要回到数据、标注协议和奖励模型评估。

## 53.23 RLHF、DPO 与在线反馈的边界

RLHF 的关键特点是学习 reward model，并让策略通过 rollout 优化这个代理目标；DPO 等直接偏好优化方法把偏好对写进相对策略概率目标，通常不需要同样的在线 PPO 流程。二者都依赖偏好数据，并不因为省略 reward model 就自动获得事实性或安全性。

一个便于审计的接口表是：

| 阶段 | 主要输入 | 直接优化的对象 | 不能单独证明 |
| --- | --- | --- | --- |
| SFT | 示范回答 | 条件 token 概率 | 真实偏好和边界安全 |
| RM | chosen/rejected | 相对排序分数 | 事实真值 |
| PPO/RLHF | rollout、reward、reference | 策略分布 | 生产副作用正确 |
| DPO 类 | 偏好对、reference | 相对 log-prob 差 | 分布外鲁棒性 |
| 线上反馈 | trace、纠正、业务结果 | 数据和路由迭代 | 无偏的人类价值 |

这样区分的好处是，模型行为变化不会被一个算法名字吞掉。上线前仍然要复测 capability、truthfulness、safety、format、tool 和 latency；上线后还要监测反馈样本是否发生选择偏差。

## 53.24 工具调用中的对齐边界

对齐模型输出“应该调用哪个工具”，执行器则负责判断“当前身份是否允许调用、参数是否合规、动作是否已经发生”。二者不能混成一层。可以把一次动作接受条件写成：

```math
\mathrm{Commit}(a)
=\mathrm{SchemaOK}(a)
\land\mathrm{AuthOK}(a)
\land\mathrm{PolicyOK}(a)
\land\mathrm{Idempotent}(a)
\land\mathrm{Verified}(a).
```

即使模型输出严格 JSON，`AuthOK` 或 `PolicyOK` 不通过时也不能执行；如果网络超时而状态未知，重试前必须查询执行状态，不能依赖模型重新生成一个看似相同的 call。RLHF 可以改善模型遵守协议的概率，却不能替代执行器的权限和幂等保证。

## 53.25 一个最小的回归验收条件

后训练发布可以使用“主指标 + 护栏指标”的验收条件。假设主任务成功率不能下降超过 `epsilon_a`，事实支持不能下降超过 `epsilon_f`，安全漏放必须为零或低于风险阈值，工具参数错误率不能超过 `epsilon_t`，则：

```math
G=I(\Delta A\ge-\epsilon_a)
 I(\Delta F\ge-\epsilon_f)
 I(U\le\tau_u)
 I(T\le\tau_t).
```

验收结果还要按语言、领域、长度、风险和用户类型分桶。总体均值通过而高风险切片失败，不能发布。对拒答任务，既要测 unsafe leak，也要测 benign request 的 over-refusal；否则安全优化很容易通过“全部拒绝”取得虚假的高分。

## 53.26 公开证据与比较条件

InstructGPT 论文公开了示范数据、奖励模型和 PPO 的经典流程；ChatGPT 是产品系统，公开资料不足以还原全部数据、模型和服务策略。DPO、RLAIF、Constitutional AI 等路线可以放在后续演进中比较，但不能把它们写成同一种实现，也不能用产品表现反推未公开的内部训练细节。

## 53.27 小结

后训练把 next-token 模型变成更可用的交互系统，但示范、偏好、奖励模型和策略优化都只是代理目标。真正可靠的结论要沿阶段保存证据，区分偏好与事实，区分模型协议与执行器权限，并用 capability、truthfulness、safety、tool 和成本约束共同决定发布。算法名称只能说明训练路径，不能替代数据质量和独立验证。
