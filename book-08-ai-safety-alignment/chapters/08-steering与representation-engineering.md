# 第八章：Steering 与 Representation Engineering

重点：steering vectors、activation intervention、拒答方向、风格控制、行为控制。

面试重点：Steering 不是只改 prompt，而是从输入层、表示层、激活层、参数层和系统层多种方式控制模型行为。Representation Engineering 则把高层行为对应到模型内部表示，用表示监控和激活干预来理解或调节模型。

安全边界：本章只讲控制方法的原理、用途、风险和防御性评估，不提供禁用安全机制或绕过拒答的操作步骤。

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 Representation Engineering: A Top-Down Approach to AI Transparency、Activation Addition / ActAdd、Contrastive Activation Addition、Inference-Time Intervention、refusal direction / representation 相关公开研究、前序 Mechanistic Interpretability、Jailbreak / Prompt Injection、Red Teaming 与 Safety Eval 章节资料边界。

本讲聚焦 steering 和 representation engineering 的防御性工程口径：如何构造行为方向、如何做推理时激活干预、如何扫描强度、如何评估目标效果和副作用，以及为什么这种方法不能替代对齐训练、红队和系统权限控制。

```text
正负行为样本 -> 激活差分 -> steering vector -> 强度扫描 -> 行为变化与副作用评估
```

本讲不提供禁用安全机制、削弱拒答机制或规避安全边界的操作流程。涉及拒答方向时，只讨论安全分析、over-refusal 调试、表示监控和上线评估。

## 本章目标

学完本章，你要能回答：

1. 什么是 steering？它和 prompt engineering、fine-tuning、RLHF 有什么关系？
2. 什么是 representation engineering？为什么它被称为 top-down transparency？
3. Steering vector 和 activation intervention 的核心直觉是什么？
4. Contrastive Activation Addition 大致怎么工作？
5. 拒答方向这类发现对 safety 有什么双重含义？
6. Steering 有哪些应用、风险和评估方法？
7. 为什么 steering 不能替代对齐训练、红队和系统权限控制？

## 1. 来龙去脉：从 Prompt 控制到表示层控制

### 1.1 最早的控制方式：Prompt

最自然的模型控制方式是写 prompt。

例如：

1. “请用简洁风格回答。”
2. “请逐步推理。”
3. “请扮演严谨审稿人。”
4. “请只输出 JSON。”

Prompt engineering 的优点是简单、无需训练、无需改模型。

缺点也明显：

1. 不稳定。
2. 容易受上下文干扰。
3. 很难精确控制强度。
4. 对复杂行为控制有限。
5. 容易被 prompt injection 影响。

### 1.2 第二阶段：训练层控制

为了更稳定控制模型行为，人们使用：

1. SFT。
2. RLHF。
3. DPO。
4. RLAIF。
5. Safety tuning。
6. Domain fine-tuning。

训练层控制更稳定，但代价更高。

问题包括：

1. 需要数据。
2. 训练成本高。
3. 容易产生副作用。
4. 很难快速开关某个行为。
5. 很难解释内部发生了什么。

### 1.3 第三阶段：表示层和激活层控制

后来研究者开始问：

```text
如果某种行为在模型内部对应某个方向或特征，我们能不能直接监控或干预这个方向？
```

例如：

1. 诚实 vs 不诚实。
2. 有害 vs 无害。
3. 拒答 vs 不拒答。
4. 幻觉 vs 事实性。
5. 某种写作风格。
6. 某种角色倾向。

这就进入了 steering 和 representation engineering。

### 1.4 方法谱系

可以把控制方法看成一条谱系：

```text
Prompt -> Decoding control -> Fine-tuning -> RLHF/DPO -> Activation steering -> Model editing -> System-level control
```

每种方法控制层级不同。

1. Prompt 控制输入。
2. Decoding 控制采样。
3. Fine-tuning 控制参数。
4. RLHF/DPO 控制偏好行为。
5. Activation steering 控制中间表示。
6. Model editing 修改局部知识或行为。
7. System-level control 控制权限、工具和产品边界。

Steering 不是替代其他方法，而是补充工具。

## 2. 小白例子：开车和方向盘

把模型想象成一辆车。

Prompt 像告诉司机：

```text
请开稳一点。
```

Fine-tuning 像重新训练司机。

RLHF 像给司机长期评分，让他学会更符合乘客偏好。

Steering vector 像在行驶过程中轻微转动方向盘。

System-level control 像道路限速、刹车系统、安全带和交通规则。

只靠一句“请开稳”不够。

只靠方向盘也不够。

真实系统需要多层控制。

## 3. 什么是 Steering

Steering 指在不完整重训模型的情况下，用某种方式引导模型行为朝目标方向变化。

常见 steering 目标：

1. 更诚实。
2. 更少幻觉。
3. 更少有害内容。
4. 更愿意拒绝危险请求。
5. 更符合某种语气或格式。
6. 更保守或更开放。
7. 更强调引用证据。

Steering 的形式包括：

1. Prompt steering。
2. Decoding steering。
3. Activation steering。
4. Feature steering。
5. Representation monitoring。
6. Model editing。

本章重点是表示层和激活层 steering。

## 4. Representation Engineering

### 4.1 定义

Representation Engineering 关注模型内部高层表示。

它不是从单个 neuron 或 circuit 逐底层拆解，而是从高层行为出发，寻找模型内部是否存在可监控、可干预的表示方向。

这就是为什么它常被称为 top-down approach。

Mechanistic interpretability 更像：

```text
从底层组件往上理解机制。
```

Representation engineering 更像：

```text
从高层行为往下寻找表示。
```

### 4.2 它解决什么问题

传统机制可解释性非常精细，但成本高。

如果我们只是想监控或调节某个高层行为，例如 honesty、harmlessness、power-seeking，逐个 circuit 逆向可能太慢。

Representation engineering 试图用更工程化的方式：

1. 构造正负样本。
2. 提取内部激活。
3. 找到行为相关方向。
4. 用这个方向监控或干预模型。

### 4.3 优点

1. 比完整机制逆向更快。
2. 适合高层行为监控。
3. 可以用于 steering。
4. 能和 safety eval 结合。
5. 对大模型可能更容易扩展。

### 4.4 缺点

1. 方向可能不是因果机制本身。
2. 正负样本构造会影响结果。
3. 高层行为可能不是单一线性方向。
4. 干预可能有副作用。
5. 解释粒度不如完整 mechanistic analysis。

## 5. Steering Vector 的核心直觉

### 5.1 表示空间中的方向

假设模型内部激活空间里存在一些和行为相关的方向。

例如：

```text
更诚实的回答 - 更不诚实的回答 = honesty direction
拒答样本 - 正常回答样本 = refusal direction
事实回答 - 幻觉回答 = factuality direction
```

这不是说行为一定真的只由一个方向决定。

它只是一个实用近似：很多高层行为可能在某些层的表示空间里有可线性分离的成分。

### 5.2 如何构造

高层步骤：

1. 构造正例和负例。
2. 跑模型，记录某层激活。
3. 计算正例平均激活和负例平均激活的差。
4. 得到一个 steering vector。
5. 推理时把这个向量加到某些层或位置。
6. 观察行为是否朝目标方向变化。

### 5.3 小白例子

想象你有很多人的照片。

你取“微笑照片”的平均特征，再减去“不微笑照片”的平均特征。

得到一个“微笑方向”。

如果把这个方向加到某张照片的表示上，可能让照片更像在微笑。

Steering vector 对模型行为做类似事情。

## 6. Contrastive Activation Addition

### 6.1 来龙去脉

Contrastive Activation Addition 这类方法来自一个直觉：

如果模型在“有某种行为”和“没有某种行为”的样本上，内部激活有稳定差异，那么这个差异可以作为控制方向。

它不需要重新训练整个模型。

只需要在推理时修改激活。

### 6.2 核心流程

高层流程：

```text
positive examples -> activations_pos
negative examples -> activations_neg
steering_vector = mean(activations_pos) - mean(activations_neg)
inference activation = activation + alpha * steering_vector
```

其中 `alpha` 控制 steering 强度。

### 6.3 它解决前人什么问题

相比 prompt：

1. 更直接作用于模型内部表示。
2. 可能更稳定。
3. 可调节强度。

相比 fine-tuning：

1. 不需要更新参数。
2. 可快速开关。
3. 适合实验分析。

相比完整 mechanistic interpretability：

1. 不需要完全理解 circuit。
2. 更偏工程可用。

### 6.4 局限

1. 需要白盒访问或至少能访问激活。
2. 方向选择依赖数据。
3. 不同层、位置、强度效果不同。
4. 可能损伤其他能力。
5. 可能只在某些模型或任务上稳定。
6. 可能被滥用来削弱安全行为。

## 7. 拒答方向的双重含义

### 7.1 发现的意义

一些研究观察到，在多个开源 chat model 中，拒答行为可能和 residual stream 中某个方向高度相关。

高层理解：

```text
模型是否拒答，可能部分由内部某类拒答表示调节。
```

这很有价值，因为它帮助我们理解 safety tuning 可能如何改变模型内部行为。

### 7.2 安全价值

它可以帮助：

1. 分析拒答机制。
2. 研究 over-refusal。
3. 监控模型是否进入拒答状态。
4. 理解 jailbreak 为什么成功。
5. 设计更稳健的安全训练。

### 7.3 风险

同样的理解也可能被滥用。

如果某个方向控制拒答，那么恶意方可能尝试削弱它。

因此这类研究有双重用途。

本书只讨论安全含义，不给出禁用拒答或绕过安全机制的操作细节。

### 7.4 面向专家

“单一方向”不应被过度理解成完整机制。

它可能是拒答行为的一个重要中介表示，但模型拒答还涉及：

1. 输入风险识别。
2. 安全策略表示。
3. 多层信息传播。
4. 输出模板生成。
5. 解码和系统策略。

所以更稳妥的说法是：拒答方向是理解和干预拒答行为的重要线索，不是完整 safety 机制。

## 8. Steering 的应用

### 8.1 风格控制

例如：

1. 更简洁。
2. 更正式。
3. 更教学化。
4. 更谨慎。

风险：风格改变可能掩盖事实性问题。

### 8.2 Factuality 控制

目标：减少幻觉，提高承认不确定性的倾向。

风险：可能导致模型过度保守，或在该回答时拒答。

### 8.3 Safety 控制

目标：增强拒绝危险请求、降低有害输出。

风险：可能导致 over-refusal，或损害正常安全教育请求。

### 8.4 Persona 控制

目标：改变角色、语气、专业程度。

风险：persona 可能和安全边界冲突。

### 8.5 Tool-use 控制

目标：更谨慎或更主动地调用工具。

风险：过度调用工具增加成本和安全面；调用不足降低任务完成率。

## 9. Steering 的评估

### 9.1 行为指标

看目标行为是否改变：

1. Factuality。
2. Refusal rate。
3. Helpfulness。
4. Harmlessness。
5. Conciseness。
6. Citation accuracy。
7. Tool-call accuracy。

### 9.2 副作用指标

必须看副作用：

1. 通用能力是否下降。
2. 输出是否变得奇怪。
3. 是否增加 hallucination。
4. 是否增加 over-refusal。
5. 是否影响多语言。
6. 是否影响长上下文。

### 9.3 鲁棒性指标

测试：

1. 不同 prompt。
2. 不同任务。
3. 不同领域。
4. 多轮对话。
5. Jailbreak。
6. RAG 和 Agent 场景。

### 9.4 表示指标

检查 steering 是否真的影响目标表示。

例如：

1. 目标方向激活变化。
2. 相关 feature 激活变化。
3. 下游行为变化。
4. 因果干预验证。

### 9.5 关键公式与 steering 评估指标速查

设第 \(\ell\) 层、第 \(p\) 个位置的激活为：

$$
h_{\ell,p}\in \mathbb{R}^{d}
$$

**1. 正负样本激活均值**

设正例集合为 \(P\)，负例集合为 \(N\)，某层聚合后的激活为 \(h_i^\ell\)：

$$
\mu_+^\ell=\frac{1}{|P|}\sum_{i\in P} h_i^\ell
$$

$$
\mu_-^\ell=\frac{1}{|N|}\sum_{i\in N} h_i^\ell
$$

**2. Steering vector**

最常见的差分方向：

$$
v^\ell=\mu_+^\ell-\mu_-^\ell
$$

归一化后：

$$
\bar v^\ell=\frac{v^\ell}{\|v^\ell\|_2+\epsilon}
$$

这个方向只是一种行为相关表示，不应直接解释为完整机制。

**3. Activation intervention**

推理时在指定层和位置加入 steering：

$$
\tilde h_{\ell,p}=h_{\ell,p}+\alpha \bar v^\ell
$$

其中 \(\alpha\) 是强度系数。真实实验通常要扫描多个 \(\alpha\)，而不是只试一个值。

**4. 表示投影分数**

可以用投影衡量样本是否更接近目标方向：

$$
s_i=\frac{\langle h_i^\ell,\bar v^\ell\rangle}{\|h_i^\ell\|_2+\epsilon}
$$

干预前后投影变化：

$$
\Delta s_i=s_i(\tilde h)-s_i(h)
$$

**5. 目标行为提升**

设 \(q_i^{base}\) 是未干预时目标行为得分，\(q_i(\alpha)\) 是强度为 \(\alpha\) 时的得分：

$$
U_{target}(\alpha)=\frac{1}{M}\sum_i [q_i(\alpha)-q_i^{base}]
$$

**6. 副作用下降**

设 \(g_i^{base}\) 是通用质量、helpfulness 或任务成功得分，\(g_i(\alpha)\) 是干预后得分：

$$
D_{side}(\alpha)=\frac{1}{M}\sum_i \max(0,g_i^{base}-g_i(\alpha))
$$

Steering 不能只看目标行为增强，还要看有没有损伤正常能力。

**7. 误拒增量**

设 \(R_{over}^{base}\) 是正常请求误拒率，\(R_{over}(\alpha)\) 是干预后误拒率：

$$
\Delta R_{over}(\alpha)=R_{over}(\alpha)-R_{over}^{base}
$$

安全 steering 很容易把模型推向过度保守，因此这个指标很重要。

**8. 安全收益**

设 \(R_{unsafe}^{base}\) 是高风险请求漏拒率，\(R_{unsafe}(\alpha)\) 是干预后漏拒率：

$$
U_{safe}(\alpha)=R_{unsafe}^{base}-R_{unsafe}(\alpha)
$$

**9. 强度选择**

可以把目标收益和副作用写成一个简单选择问题：

$$
\alpha^*=\arg\max_{\alpha\in A}
[U_{target}(\alpha)+\lambda_s U_{safe}(\alpha)-\lambda_d D_{side}(\alpha)-\lambda_o \Delta R_{over}(\alpha)]
$$

这里的权重应由产品、安全和评估目标决定，不能只按离线分数临时拍。

**10. Steering 上线门禁**

$$
G_{steer}=
\mathbb{1}[
U_{target}\ge \tau_{target}
\land U_{safe}\ge \tau_{safe}
\land D_{side}\le \tau_{side}
\land \Delta R_{over}\le \tau_{over}
\land R_{jail}\le \tau_{jail}
]
$$

\(G_{steer}=1\) 只表示当前强度、任务分布和安全评估下可以进入下一轮实验或灰度，不表示方向完全可靠。

## 10. Steering 和其他方法的关系

### 10.1 和 Prompt Engineering

Prompt engineering 更容易用，但更软。

Steering 更直接，但需要模型内部访问。

### 10.2 和 Fine-tuning

Fine-tuning 改参数，steering 改推理时激活。

Fine-tuning 更持久，steering 更可开关。

### 10.3 和 RLHF / DPO

RLHF/DPO 通过偏好数据改变整体行为。

Steering 可以做局部行为控制或研究工具。

### 10.4 和 Mechanistic Interpretability

Mechanistic interpretability 试图理解机制。

Steering 试图利用表示控制行为。

二者互相促进：理解帮助控制，控制实验帮助验证理解。

### 10.5 和 Model Editing

Model editing 通常修改参数或知识。

Steering 通常在推理时修改激活。

Editing 更持久，steering 更临时。

## 11. 真实项目中的使用方式

### 11.1 研究和调试

Steering 很适合用于研究：

1. 某种行为是否有内部方向。
2. 增强或减弱该方向会发生什么。
3. 该行为和哪些能力有 trade-off。
4. 安全训练是否改变了表示。

### 11.2 安全监控

可以把表示方向作为监控信号之一。

例如：

1. 检测模型是否进入高风险回答状态。
2. 检测拒答状态。
3. 检测不确定性状态。
4. 检测工具调用风险状态。

但不能只靠它做最终决策。

### 11.3 产品控制

实际产品中，steering 可以作为辅助层。

例如在不同模式下调节：

1. 更保守。
2. 更简洁。
3. 更教学。
4. 更严格引用。

但上线前必须做系统评估。

## 12. 风险和局限

### 12.1 可解释性风险

一个方向有效，不代表我们完整理解了机制。

### 12.2 泛化风险

在一个数据集上有效，不代表跨任务有效。

### 12.3 副作用风险

控制一个行为可能影响其他行为。

### 12.4 安全双用风险

能增强安全，也可能被用于削弱安全。

### 12.5 工程复杂度

需要访问内部激活，部署复杂度高于 prompt。

### 12.6 评估难度

需要同时评估目标行为、副作用、鲁棒性和安全边界。

## 13. 未来可能演化

### 13.1 从手工方向到自动发现

未来可能自动发现大量行为方向和安全相关 feature。

### 13.2 从单方向到多维控制

复杂行为可能不是单一方向，而是多个 feature 和 circuit 共同作用。

### 13.3 和 SAE 结合

SAE 可以提供更细粒度 feature。

Steering 可以从粗方向转向 feature-level control。

### 13.4 和安全监控结合

内部表示监控可能成为未来 safety monitor 的一部分。

### 13.5 和系统治理结合

最终 steering 可能作为模型治理工具之一，但仍需要权限、红队、评估和审计配合。

## 14. 面试官会怎么问

### 问题 1：什么是 steering vector？

回答要点：

1. 模型激活空间中和某种行为相关的方向。
2. 通常通过正负样本激活差异得到。
3. 推理时加入该方向，引导模型行为。
4. 需要评估目标效果和副作用。

标准回答：

```text
Steering vector 是模型内部表示空间中和某种高层行为相关的方向。通常可以用正负样本在某层激活上的平均差得到，比如更诚实和更不诚实回答的差。推理时把这个方向加到激活上，可以调节模型输出倾向。但它只是行为控制工具，必须验证泛化、副作用和安全风险。
```

### 问题 2：Representation Engineering 和 Mechanistic Interpretability 有什么区别？

回答要点：

1. Mechanistic interpretability 更偏 bottom-up，理解组件和 circuit。
2. Representation engineering 更偏 top-down，从高层行为找内部表示。
3. 前者追求机制解释，后者更偏监控和控制。
4. 二者互补。

### 问题 3：Contrastive Activation Addition 怎么理解？

回答要点：

1. 构造正负行为样本。
2. 取中间激活差异。
3. 得到 steering vector。
4. 推理时加到激活中。
5. 通过强度系数控制行为变化。

### 问题 4：拒答方向说明了什么？

回答要点：

1. 拒答行为可能有重要内部表示方向。
2. 有助于理解 safety tuning。
3. 可以帮助分析 over-refusal 和 jailbreak。
4. 也有双用风险，因此不能只把它当防御工具。
5. 单一方向不等于完整安全机制。

### 问题 5：Steering 能替代 RLHF 吗？

回答要点：

1. 不能简单替代。
2. Steering 更适合局部、可开关、研究和辅助控制。
3. RLHF/DPO 更适合整体行为塑造。
4. 真实系统需要训练、steering、评估和系统防护结合。

## 15. 标准回答模板

面试中可以这样回答：

```text
Steering 是在不完整重训模型的情况下控制模型行为的一类方法。最简单的是 prompt steering，更深入的是 activation steering，也就是在模型中间激活上加入或修改某些行为相关方向。

Representation engineering 通常从高层行为出发，比如 honesty、harmlessness、refusal、factuality，构造正负样本，提取模型内部表示差异，得到可监控或可干预的方向。Contrastive Activation Addition 就是这类方法的代表：用正负样本激活均值差作为 steering vector，在推理时按强度加入激活。

它的优势是无需重新训练、可开关、能帮助理解模型内部表示；风险是方向可能不泛化，可能影响其他能力，也有安全双用风险。尤其拒答方向这类发现既能帮助理解安全机制，也可能说明当前安全微调存在脆弱性。因此真实系统不能只靠 steering，而要结合 RLHF/DPO、安全评估、红队、权限控制和监控。
```

## 16. 常见误区

### 16.1 误区：Steering 就是 prompt engineering

纠正：Prompt 是输入层控制，activation steering 是表示层控制。

### 16.2 误区：找到方向就等于理解机制

纠正：方向可能有效，但不代表完整解释了内部 circuit。

### 16.3 误区：Steering 没有副作用

纠正：它可能改变其他能力、风格、安全边界和泛化表现。

### 16.4 误区：拒答方向说明安全很容易解决

纠正：它说明某些模型拒答行为可能有简洁表示，但也说明安全机制可能脆弱。

### 16.5 误区：Steering 可以替代系统安全

纠正：Steering 是模型内部控制工具，不能替代权限、沙箱、审计和发布门禁。

## 17. 小练习

### 练习 1

用自己的话解释 prompt steering、fine-tuning 和 activation steering 的区别。

### 练习 2

设计一个 steering vector 实验，用于增强模型回答中的事实性。

要求说明正负样本、激活层选择、强度系数、评估指标和副作用检查。

### 练习 3

解释为什么拒答方向这类发现同时有安全价值和双用风险。

### 练习 4

比较 representation engineering 和 mechanistic interpretability。

要求说明 top-down 与 bottom-up 的区别。

### 练习 5

为一个企业助手设计 steering 上线评估。

要求覆盖：helpfulness、factuality、refusal、over-refusal、latency、jailbreak 和人工审核。

## 18. 最小可运行 Steering 审计 demo

下面的 demo 不需要真实模型，只用 toy 激活和 toy 评估表演示 steering 的核心审计流程。真实项目中，`positive_activations`、`negative_activations` 和 `case["activation"]` 应来自模型 forward hook。

```python
from math import sqrt


def mean_vector(rows):
    dims = len(rows[0])
    return [sum(row[j] for row in rows) / len(rows) for j in range(dims)]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(alpha, v):
    return [alpha * x for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return sqrt(dot(v, v))


def normalize(v, eps=1e-12):
    length = norm(v) + eps
    return [x / length for x in v]


def projection_score(h, v):
    return dot(h, v) / (norm(h) + 1e-12)


def round_vector(v):
    return [round(x, 3) for x in v]


positive_activations = [
    [1.2, 0.8, -0.1],
    [1.0, 0.7, 0.0],
    [1.1, 0.9, 0.1],
]
negative_activations = [
    [0.1, -0.2, 0.8],
    [0.0, -0.1, 0.7],
    [0.2, -0.3, 0.9],
]

mu_pos = mean_vector(positive_activations)
mu_neg = mean_vector(negative_activations)
steering_vector = subtract(mu_pos, mu_neg)
unit_vector = normalize(steering_vector)

cases = [
    {
        "id": "factual_qa",
        "activation": [0.4, 0.2, 0.4],
        "target_gain": 0.30,
        "safe_gain": 0.00,
        "side_loss": 0.03,
        "over_refusal": 0.00,
    },
    {
        "id": "citation_answer",
        "activation": [0.5, 0.3, 0.3],
        "target_gain": 0.24,
        "safe_gain": 0.00,
        "side_loss": 0.02,
        "over_refusal": 0.00,
    },
    {
        "id": "boundary_safety",
        "activation": [0.2, 0.0, 0.7],
        "target_gain": 0.20,
        "safe_gain": 0.38,
        "side_loss": 0.05,
        "over_refusal": 0.05,
    },
    {
        "id": "normal_help",
        "activation": [0.3, 0.1, 0.2],
        "target_gain": 0.05,
        "safe_gain": 0.00,
        "side_loss": 0.08,
        "over_refusal": 0.08,
    },
    {
        "id": "tool_caution",
        "activation": [0.1, 0.2, 0.5],
        "target_gain": 0.18,
        "safe_gain": 0.22,
        "side_loss": 0.04,
        "over_refusal": 0.03,
    },
]


def evaluate_alpha(alpha):
    projection_shift = []
    target_uplift = []
    safe_gain = []
    side_drop = []
    over_delta = []

    for case in cases:
        before = projection_score(case["activation"], unit_vector)
        steered = add(case["activation"], scale(alpha, unit_vector))
        after = projection_score(steered, unit_vector)
        projection_shift.append(after - before)
        target_uplift.append(alpha * case["target_gain"])
        safe_gain.append(alpha * case["safe_gain"])
        side_drop.append(alpha * case["side_loss"])
        over_delta.append(alpha * case["over_refusal"])

    metrics = {
        "alpha": alpha,
        "avg_projection_shift": round(sum(projection_shift) / len(cases), 3),
        "target_uplift": round(sum(target_uplift) / len(cases), 3),
        "safe_gain": round(sum(safe_gain) / len(cases), 3),
        "side_drop": round(sum(side_drop) / len(cases), 3),
        "over_refusal_delta": round(sum(over_delta) / len(cases), 3),
    }
    metrics["objective"] = round(
        metrics["target_uplift"]
        + metrics["safe_gain"]
        - 0.8 * metrics["side_drop"]
        - 1.2 * metrics["over_refusal_delta"],
        3,
    )
    return metrics


alpha_grid = [0.0, 0.4, 0.8, 1.2]
scan = [evaluate_alpha(alpha) for alpha in alpha_grid]
best = max(scan, key=lambda item: item["objective"])

gates = {
    "target": best["target_uplift"] >= 0.10,
    "safe": best["safe_gain"] >= 0.08,
    "side_effect": best["side_drop"] <= 0.06,
    "over_refusal": best["over_refusal_delta"] <= 0.05,
    "projection": best["avg_projection_shift"] >= 0.20,
}

print("mu_pos=", round_vector(mu_pos))
print("mu_neg=", round_vector(mu_neg))
print("steering_vector=", round_vector(steering_vector))
print("unit_vector=", round_vector(unit_vector))
print("scan=", scan)
print("best=", best)
print("gates=", gates)
print("steering_ready=", all(gates.values()))
```

预期输出：

```text
mu_pos= [1.1, 0.8, 0.0]
mu_neg= [0.1, -0.2, 0.8]
steering_vector= [1.0, 1.0, -0.8]
unit_vector= [0.615, 0.615, -0.492]
scan= [{'alpha': 0.0, 'avg_projection_shift': 0.0, 'target_uplift': 0.0, 'safe_gain': 0.0, 'side_drop': 0.0, 'over_refusal_delta': 0.0, 'objective': 0.0}, {'alpha': 0.4, 'avg_projection_shift': 0.468, 'target_uplift': 0.078, 'safe_gain': 0.048, 'side_drop': 0.018, 'over_refusal_delta': 0.013, 'objective': 0.096}, {'alpha': 0.8, 'avg_projection_shift': 0.671, 'target_uplift': 0.155, 'safe_gain': 0.096, 'side_drop': 0.035, 'over_refusal_delta': 0.026, 'objective': 0.192}, {'alpha': 1.2, 'avg_projection_shift': 0.752, 'target_uplift': 0.233, 'safe_gain': 0.144, 'side_drop': 0.053, 'over_refusal_delta': 0.038, 'objective': 0.289}]
best= {'alpha': 1.2, 'avg_projection_shift': 0.752, 'target_uplift': 0.233, 'safe_gain': 0.144, 'side_drop': 0.053, 'over_refusal_delta': 0.038, 'objective': 0.289}
gates= {'target': True, 'safe': True, 'side_effect': True, 'over_refusal': True, 'projection': True}
steering_ready= True
```

这个 demo 对应真实项目中的关键点：

1. Steering vector 来自正负行为样本激活差，而不是手写规则。
2. 强度 \(\alpha\) 要扫描，不能凭直觉选。
3. 目标收益、安全收益、副作用和误拒增量要同时看。
4. `steering_ready=True` 只说明 toy gate 通过，不代表真实模型可以直接上线。

## 19. 本章总结

Steering 是控制模型行为的一类方法，从 prompt 控制、解码控制，到激活层和表示层控制都有不同形式。

Representation Engineering 从高层行为出发，寻找模型内部可监控、可干预的表示方向，是一种 top-down transparency 思路。

Steering vector 通常通过正负样本激活差异得到，推理时加入模型激活以调节行为。

Contrastive Activation Addition 是 activation steering 的代表方法之一。

拒答方向说明模型安全行为可能有可分析的内部表示，但也暴露了安全机制的潜在脆弱性和双用风险。

Steering 的价值在于快速、可开关、适合研究和辅助控制；局限在于泛化、副作用、部署复杂度和安全双用风险。

真实系统中，steering 应和对齐训练、安全评估、红队、权限控制、系统监控和治理流程结合，而不是单独承担安全保证。
