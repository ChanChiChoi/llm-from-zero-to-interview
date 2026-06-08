# 第九章：Model Editing 与 Unlearning

重点：知识编辑、遗忘训练、版权数据移除、局部修改与副作用。

面试重点：Model Editing 关心“如何局部修改模型知识或行为”，Unlearning 关心“如何让模型近似忘掉某些数据、知识或能力”。二者都不是简单删一行数据库记录，而是在神经网络参数和行为分布中做受控改变。

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点参考了 ROME、MEMIT、MEND、SERAC、CounterFact / ZsRE editing benchmark、TOFU、WMDP、MUSE、LLM unlearning survey 和机器遗忘治理相关论文与技术报告。

为了避免误导，本章只讲防御性、治理性和评估性的 model editing / unlearning：

1. model editing 关注局部知识或行为修正，但不能保证所有等价问法、相关事实和下游任务都自动一致。
2. unlearning 在 LLM 中通常只能给出近似、行为层和统计层证据，不能轻易宣称“完全删除模型记忆”。
3. guardrail、拒答策略和输出过滤可以降低泄露风险，但它们不是参数级遗忘。
4. 版权、隐私和危险能力场景必须配合数据治理、训练记录、评估集、红队、审计日志和版本回滚。
5. 本章 demo 只做 toy 级指标审计，不提供植入错误事实、规避安全策略、提取隐私记忆或削弱防护能力的操作流程。

## 本章目标

学完本章，你要能回答：

1. 什么是 model editing？它和 fine-tuning、RAG 有什么区别？
2. 为什么大模型知识更新不能只靠继续训练？
3. ROME、MEMIT 这类方法的核心直觉是什么？
4. 什么是 machine unlearning？为什么 LLM unlearning 很难？
5. 删除数据、修改知识、抑制输出、遗忘能力之间有什么区别？
6. 如何评估 model editing 的成功、泛化、局部性和副作用？
7. 如何评估 unlearning 是否真的有效？
8. 这些技术在版权、隐私、安全和模型治理中有什么价值和风险？

## 1. 来龙去脉：为什么需要 Model Editing

### 1.1 最早的知识更新方式：重新训练

如果一个模型知道了错误事实，最直接的想法是重新训练。

例如模型认为：

```text
某公司 CEO 是 A
```

但现实已经变成：

```text
某公司 CEO 是 B
```

最彻底的方法是更新训练数据，然后重新训练或继续训练模型。

问题是：

1. 成本太高。
2. 周期太长。
3. 很难只改一个事实。
4. 可能破坏其他能力。
5. 很难处理频繁更新的知识。

### 1.2 RAG 解决了一部分问题

RAG 的思路是：不要把所有新知识都写进参数，而是在推理时检索最新资料。

优点：

1. 更新快。
2. 可追溯。
3. 适合企业知识库。
4. 不必改模型参数。

但 RAG 也有局限：

1. 模型参数里的旧知识仍然可能干扰。
2. 检索失败时模型仍会凭记忆回答。
3. 对常识性事实，用户未必提供检索上下文。
4. 复杂行为和风格问题不一定能靠检索解决。

### 1.3 Model Editing 的问题定义

Model Editing 要解决的问题是：

```text
能否在不完整重训模型的情况下，局部修改模型中某个事实、关联或行为，同时尽量不影响其他知识和能力？
```

它不是 RAG。

RAG 是外部知识增强。

Model editing 是参数或内部机制层面的修改。

### 1.4 后来为什么出现 Unlearning

Unlearning 的动机更复杂。

它不只是“把错事实改对”。

它可能来自：

1. 版权数据移除。
2. 隐私数据删除。
3. 法规中的删除请求。
4. 移除有害能力。
5. 去除错误知识。
6. 降低模型对某些训练样本的记忆。

问题是，大模型不是数据库。

训练样本的影响被分散到大量参数中。

所以 unlearning 通常只能做到近似、可评估、可审计，而很难证明“完全遗忘”。

## 2. 小白例子：改书、贴便签、擦记忆

假设一本百科书里有错误。

你有几种处理方式。

第一，重新印整本书。

这像重新训练模型。

第二，在书旁边贴一张最新资料卡。

这像 RAG。

第三，直接把某一页上的错误句子改掉。

这像 model editing。

第四，要求读者以后再也想不起某个故事。

这像 unlearning。

显然，第四件事比第三件事难得多。

因为“记忆”不是只存在于某一行文字里，而是可能影响很多关联概念。

## 3. Model Editing 是什么

### 3.1 定义

Model editing 是对模型参数或内部表示做局部修改，使模型在目标事实、知识或行为上发生预期变化，同时尽量保持其他能力不变。

典型目标：

1. 修改事实关联。
2. 添加新知识。
3. 删除错误知识。
4. 修复特定行为。
5. 改变某类回答倾向。

### 3.2 和 Fine-tuning 的区别

Fine-tuning 通常是全局训练。

Model editing 更强调局部性。

对比：

| 维度 | Fine-tuning | Model Editing |
|---|---|---|
| 目标 | 整体行为调整 | 局部知识或行为修改 |
| 数据量 | 通常较多 | 可以少量编辑样本 |
| 成本 | 较高 | 相对较低 |
| 风险 | 全局漂移 | 局部副作用 |
| 评估重点 | 总体能力 | 成功率、泛化、局部性 |

### 3.3 和 RAG 的区别

RAG 不修改参数。

Model editing 修改模型内部。

如果知识经常变化，RAG 更合适。

如果希望模型在无检索时也改掉某个稳定错误，editing 更有吸引力。

但真实系统常常两者结合。

## 4. ROME 的核心直觉

### 4.1 来龙去脉

ROME 相关工作研究了 GPT 类模型中的事实关联存储。

它提出一个重要观察：某些事实关联可能在中间层 MLP 模块中以相对局部的方式被调用和编辑。

例如：

```text
subject -> relation -> object
```

模型回答事实问题时，可能在处理 subject token 时，从中间层 FFN 中激活与 object 相关的关联。

### 4.2 核心思想

ROME 的高层直觉是：

1. 先定位哪些层和激活对某个事实预测关键。
2. 再对相关 MLP 权重做低秩更新。
3. 让模型把旧事实关联改成新事实关联。
4. 同时尽量不影响其他事实。

这里不需要记具体公式。

面试中更重要的是讲清：它把 factual association 当作可定位、可编辑的内部计算。

### 4.3 它解决前人什么问题

前人方法可能需要 fine-tuning 多步更新，容易影响其他知识。

ROME 试图更精准地编辑某个事实关联。

### 4.4 优点

1. 局部编辑成本低。
2. 对某些事实关联有效。
3. 和 mechanistic interpretability 有联系。
4. 强调定位和编辑结合。

### 4.5 局限

1. 更适合事实关联，不一定适合复杂能力。
2. 编辑可能不完全泛化到所有表达方式。
3. 多次编辑可能相互干扰。
4. 局部性难保证。
5. 对不同模型结构和规模可能表现不同。

## 5. MEMIT：从单点编辑到批量编辑

### 5.1 为什么需要 MEMIT

如果只改一个事实，ROME 这类方法有吸引力。

但真实需求常常是批量编辑。

例如：

1. 更新一批过时事实。
2. 添加某个领域的大量知识。
3. 修改成千上万条关联。

逐条编辑可能效率低，且编辑之间互相干扰。

### 5.2 MEMIT 的核心目标

MEMIT 试图把 model editing 扩展到大量 memory edits。

高层理解：

```text
ROME 更像单点手术。
MEMIT 更像批量更新。
```

### 5.3 优点

1. 支持更大规模编辑。
2. 比完整重训更便宜。
3. 对模型记忆更新有工程意义。

### 5.4 风险

1. 批量编辑更容易产生副作用。
2. 编辑之间可能冲突。
3. 难评估所有等价表达。
4. 对模型整体知识结构影响更复杂。

## 6. Model Editing 的评估指标

Model editing 不能只看目标样本是否答对。

至少要看四类指标。

### 6.1 Edit success

目标问题是否按编辑后知识回答。

例如原来答 A，编辑后应答 B。

### 6.2 Generalization

同一个事实换不同问法是否也生效。

例如：

1. 直接问。
2. 改写问。
3. 多语言问。
4. 放进上下文问。
5. 间接推理问。

### 6.3 Specificity / Locality

不相关事实是否保持不变。

例如改某公司 CEO，不应该影响同名实体、相关公司或无关事实。

### 6.4 Fluency and capability retention

模型语言质量和通用能力是否保持。

编辑后不能变得：

1. 语法异常。
2. 回答模板化。
3. 推理能力下降。
4. 其他 benchmark 大幅退化。

## 7. 什么是 Unlearning

### 7.1 定义

Machine unlearning 是让模型移除某些训练数据、知识、关联或行为影响的一类方法。

在 LLM 中，它可能指：

1. 忘掉某个版权语料。
2. 忘掉个人隐私信息。
3. 忘掉某类错误知识。
4. 降低某类危险能力。
5. 让模型不再复现特定文本。

### 7.2 和删除数据库记录不同

数据库中删除一行，记录就没了。

神经网络中，一个样本影响大量参数。

同一个知识也可能来自多个来源。

例如模型知道某个角色，不一定只因为读过一本书。

它可能来自：

1. 原书内容。
2. 书评。
3. 百科条目。
4. 论坛讨论。
5. 影视资料。
6. 二次创作。

所以 unlearning 很难定义清楚。

### 7.3 Approximate Unlearning

很多 LLM unlearning 方法只能做到 approximate unlearning。

意思是：

```text
在一组评估任务上，模型表现得像没有学过目标数据或目标知识。
```

这不等于数学上证明完全删除。

面试中要明确这一点。

## 8. Unlearning 的目标类型

### 8.1 Data unlearning

目标是移除某些训练样本影响。

例如用户要求删除个人数据。

难点：很难证明样本影响完全消失。

### 8.2 Knowledge unlearning

目标是让模型不再知道某些知识。

例如不再复述某个版权作品细节。

难点：知识边界很模糊。

### 8.3 Behavior unlearning

目标是去除某类行为。

例如降低模型输出某类有害内容的倾向。

难点：行为可能和正常能力共享表示。

### 8.4 Capability unlearning

目标是降低某类能力。

例如让模型不再支持某些高风险能力。

难点：能力常常和正常知识、推理、代码能力交织。

## 9. Unlearning 方法谱系

### 9.1 重新训练

最干净但最贵。

从去除目标数据后的语料重新训练模型。

优点：概念上最明确。

缺点：成本不可接受，尤其是大模型。

### 9.2 Fine-tuning based unlearning

通过构造替代目标、反向训练或对比训练，让模型降低目标知识输出概率。

优点：成本较低。

缺点：可能只是抑制输出，不是真正删除内部知识。

### 9.3 Model editing based unlearning

定位目标知识相关参数或表示，做局部修改。

优点：局部性强。

缺点：目标知识复杂时难定位。

### 9.4 Guardrail based suppression

通过策略层、分类器、RAG 规则或输出过滤禁止输出。

优点：工程上实用。

缺点：不是模型真的忘了，只是被拦截。

### 9.5 Data governance

从源头减少问题：

1. 训练前数据过滤。
2. 版权许可管理。
3. PII 清理。
4. 数据溯源。
5. 训练数据版本化。

这是 unlearning 的前置治理。

## 10. Unlearning 的评估难点

### 10.1 Forget set

目标数据或知识是否真的被遗忘。

指标可能包括：

1. 目标内容复现率。
2. 问答准确率下降。
3. Completion likelihood 下降。
4. Membership inference 风险下降。

### 10.2 Retain set

不该忘的知识是否保持。

例如删除某本书内容，不应该破坏一般语言能力、常识能力和无关文学知识。

### 10.3 Generalization of forgetting

遗忘是否泛化到改写、翻译、间接问题和多轮追问。

### 10.4 Robustness

用户是否能通过提示、上下文、少量线索重新诱导模型输出目标内容。

### 10.5 Verifiability

如何证明模型“真的忘了”？

这很难。

实际中通常只能提供证据：

1. 多维评估下降。
2. 攻击式提问失败。
3. 保留集能力不下降。
4. 第三方审计。
5. 数据和训练流程记录。

### 10.6 关键公式与 editing / unlearning 评估指标速查

把一个编辑请求写成：

$$
e_i=(x_i,y_i^{old},y_i^{new},P_i,L_i,w_i)
$$

其中 `x_i` 是目标提示或事实查询，`y_i^{old}` 是编辑前错误或旧答案，`y_i^{new}` 是编辑后目标答案，`P_i` 是等价改写集合，`L_i` 是局部性检查集合，`w_i` 是重要性或风险权重。

编辑后的模型记为 `M'`。如果目标问题回答正确，令 `a_i^{target}=1`，否则为 0。加权 edit success 为：

$$
S_{edit}=
\frac{\sum_i w_i a_i^{target}}{\sum_i w_i}
$$

泛化要看同一编辑在改写问法上是否生效。若第 `i` 个编辑有 `m_i` 个改写，`a_{i,j}^{para}` 表示第 `j` 个改写是否回答到新目标：

$$
S_{gen}=
\frac{\sum_i w_i \left(\frac{1}{m_i}\sum_{j=1}^{m_i} a_{i,j}^{para}\right)}
{\sum_i w_i}
$$

局部性要看不该改变的样本是否保持。若 `b_{i,k}^{loc}=1` 表示第 `k` 个局部性样本保持原行为：

$$
S_{loc}=
\frac{\sum_i w_i \left(\frac{1}{n_i}\sum_{k=1}^{n_i} b_{i,k}^{loc}\right)}
{\sum_i w_i}
$$

通用能力保留可以用 retain set 上的编辑前后差值表示：

$$
D_{ret}=A_{ret}^{before}-A_{ret}^{after}
$$

很多参数级 editing 方法可以抽象为受约束优化：既要让新答案概率上升，又要控制参数变化和保留集损失。

$$
L_{edit}(\theta')=
-\frac{1}{N}\sum_i \log p_{\theta'}(y_i^{new}\mid x_i)
+\lambda_1 \lVert \theta'-\theta\rVert_2^2
+\lambda_2 D_{ret}
$$

ROME / MEMIT 一类方法常被面试概括成“定位关键表示，再对相关权重做低秩或批量更新”。最简抽象是：

$$
W'=W+\Delta W,\quad \Delta W=u v^\top
$$

其中 `W` 是被编辑的模块权重，`u v^\top` 是低秩更新；真正方法还要估计 key / value、层位置和多编辑冲突，这里只保留面试层抽象。

unlearning 请求可以写成：

$$
u_i=(z_i,F_i,R_i,Q_i,w_i)
$$

其中 `z_i` 是删除或遗忘目标，`F_i` 是 forget set，`R_i` 是 retain set，`Q_i` 是改写、多轮、翻译或诱导式鲁棒性检查集合，`w_i` 是风险权重。

forget set 泄露率：

$$
R_{forget}=
\frac{\sum_i w_i a_i^{leak}}{\sum_i w_i}
$$

retain set 能力保留：

$$
A_{retain}=
\frac{1}{|R|}\sum_{r\in R} s_r
$$

遗忘泛化泄露率：

$$
R_{robust}=
\frac{\sum_i w_i \left(\frac{1}{q_i}\sum_{j=1}^{q_i} a_{i,j}^{robust}\right)}
{\sum_i w_i}
$$

membership inference 风险下降：

$$
D_{mia}=R_{mia}^{before}-R_{mia}^{after}
$$

unlearning 的上线门禁可以写成：

$$
G_{unlearn}=
I[
R_{forget}\le\tau_f
\land R_{robust}\le\tau_r
\land D_{mia}\ge\tau_m
\land D_{ret}\le\tau_d
\land A_{retain}\ge\tau_a
]
$$

面试中要强调：`G_unlearn=1` 只说明当前评估集和攻击强度下证据达标，不等于模型在数学上已经完全遗忘。

## 11. Model Editing 与 Unlearning 的关系

Model editing 是“改”。

Unlearning 是“忘”。

二者有交集。

例如：

1. 把错误事实 A 改成 B，是 editing。
2. 让模型不再回答 A，是 unlearning 或 suppression。
3. 移除某个作品中的细节记忆，是 unlearning。
4. 修改某个安全策略行为，可能既是 editing 也是 behavior unlearning。

关键区别：

```text
Editing 通常有新的目标答案。
Unlearning 通常要求目标影响消失，同时保留其他能力。
```

## 12. 安全和治理价值

### 12.1 版权治理

Unlearning 可用于响应版权争议或数据删除要求。

但要谨慎：近似遗忘不等于法律上一定充分。

### 12.2 隐私保护

如果模型记住个人信息，unlearning 可能帮助降低泄露风险。

但更重要的是训练前 PII 清理和数据治理。

### 12.3 安全能力降低

对于高风险能力，可以尝试 behavior 或 capability unlearning。

但风险是：

1. 可能损伤正常能力。
2. 可能只抑制表面输出。
3. 可能被 prompt 绕过。
4. 很难验证彻底性。

### 12.4 知识更新

Model editing 可用于过时事实更新。

但对动态知识，RAG 往往更可控。

## 13. 真实项目中的取舍

### 13.1 什么时候用 RAG

适合：

1. 企业知识库。
2. 频繁更新事实。
3. 需要引用和追溯。
4. 权限敏感数据。

### 13.2 什么时候用 Model Editing

适合：

1. 少量稳定事实修正。
2. 模型无检索时也必须修正。
3. 研究模型内部知识存储。
4. 需要低成本局部修改。

### 13.3 什么时候用 Unlearning

适合：

1. 数据删除请求。
2. 版权争议。
3. 隐私泄露风险。
4. 高风险能力治理。

### 13.4 什么时候用 Guardrail

适合：

1. 高风险输出拦截。
2. 法规和政策约束。
3. 需要快速上线防护。
4. 不能修改底层模型时。

### 13.5 常见组合

真实系统常用组合：

```text
data governance + RAG + safety policy + model editing / unlearning + eval + audit
```

## 14. 风险和副作用

### 14.1 局部性失败

改一个事实影响相关或无关事实。

### 14.2 泛化不足

目标问法改写后编辑不生效。

### 14.3 能力退化

Unlearning 可能破坏通用能力。

### 14.4 虚假安全感

模型不回答目标问题，不代表内部知识完全消失。

### 14.5 评估污染

如果只针对测试题做 unlearning，模型可能只是过拟合评估。

### 14.6 双用风险

Model editing 也可能被恶意用于植入错误事实或后门行为。

所以需要权限、审计和版本管理。

## 15. 面向专家：几个关键判断

### 15.1 编辑的是知识还是表达？

模型回答变化可能只是表面输出变化，不一定内部知识真的改了。

需要用多问法、多上下文和因果分析验证。

### 15.2 遗忘的是数据还是概念？

忘掉训练样本和忘掉由样本支持的概念不同。

版权、隐私和安全场景中目标不同。

### 15.3 抑制输出不等于 unlearning

策略层拒答可以降低泄露，但不是参数级遗忘。

### 15.4 局部编辑和全局一致性冲突

模型知识是关联网络。

改一条事实可能需要更新很多推论，否则会出现不一致。

### 15.5 评估必须区分白盒和黑盒

黑盒只能看行为。

白盒可以看概率、激活、参数变化和 membership inference 风险。

## 16. 面试官会怎么问

### 问题 1：什么是 model editing？

回答要点：

1. 不完整重训模型。
2. 局部修改知识或行为。
3. 尽量保持其他能力不变。
4. 评估 edit success、generalization、specificity 和能力保留。

标准回答：

```text
Model editing 是在不重新训练整个模型的情况下，对模型参数或内部机制做局部修改，让它在目标事实或行为上产生预期变化，同时尽量不影响其他知识和能力。它和 RAG 不同，RAG 是外部检索增强，editing 是模型内部修改。
```

### 问题 2：ROME 的核心思想是什么？

回答要点：

1. 事实关联可能在中间层 MLP 中有相对局部机制。
2. 先定位影响事实预测的关键位置。
3. 再做低秩参数更新。
4. 目标是修改特定 factual association。

### 问题 3：Model editing 如何评估？

回答要点：

1. Edit success：目标事实是否改对。
2. Generalization：改写问法是否生效。
3. Specificity/locality：无关知识是否不变。
4. Fluency/capability retention：语言和能力是否保持。

### 问题 4：什么是 LLM unlearning？

回答要点：

1. 让模型移除某些数据、知识或行为影响。
2. 不是删除数据库记录。
3. 通常只能做到 approximate unlearning。
4. 要同时看 forget set、retain set、泛化和鲁棒性。

### 问题 5：Unlearning 和 guardrail 有什么区别？

回答要点：

1. Unlearning 试图改变模型内部参数或行为分布。
2. Guardrail 是外部拦截或策略控制。
3. Guardrail 快速实用，但不代表模型忘了。
4. 真实系统通常二者结合。

## 17. 标准回答模板

面试中可以这样回答：

```text
Model editing 解决的是大模型知识或行为局部更新问题。重新训练成本太高，RAG 又只是外部检索，不能改变模型参数里的旧知识，所以出现了 ROME、MEMIT 这类直接编辑模型内部事实关联的方法。评估时不能只看目标问题答对，还要看改写泛化、无关知识是否保持、语言质量和通用能力是否退化。

Unlearning 更难，它不是把数据库中一行删掉，而是让模型近似移除某些训练数据、知识或能力的影响。LLM 中一个知识可能来自多个来源，也可能和其他能力共享表示，所以通常只能做 approximate unlearning。评估上要同时看 forget set、retain set、改写和多轮鲁棒性、membership inference 风险以及人工红队。

真实系统里，model editing、unlearning、RAG、guardrail 和数据治理各有位置。动态知识优先 RAG，少量稳定错误可以考虑 editing，版权隐私和危险能力治理可以考虑 unlearning，但必须配合评估、审计和版本管理。
```

## 18. 常见误区

### 18.1 误区：Model editing 等于 fine-tuning

纠正：Editing 更强调局部修改和保持其他知识不变。

### 18.2 误区：RAG 可以替代所有知识更新

纠正：RAG 很重要，但参数记忆仍可能干扰。

### 18.3 误区：Unlearning 能证明模型完全忘记

纠正：大多数 LLM unlearning 是近似行为证据，不是完整数学证明。

### 18.4 误区：不输出就代表忘了

纠正：可能只是 guardrail 或输出抑制，内部知识仍在。

### 18.5 误区：编辑越局部越安全

纠正：局部编辑也可能破坏相关知识或引入不一致。

## 19. 最小可运行 Model Editing / Unlearning 审计 demo

下面的 demo 不修改真实模型，只模拟评估审计表：editing 看目标成功、改写泛化、局部性、能力保留和冲突；unlearning 看 forget set 泄露、改写 / 多轮鲁棒泄露、membership inference 风险下降和 retain set 能力保留。

```python
def weighted_average(items, value_key):
    total_weight = sum(item["weight"] for item in items)
    return round(sum(item[value_key] * item["weight"] for item in items) / total_weight, 3)


def weighted_rate(items, flag_key):
    total_weight = sum(item["weight"] for item in items)
    return round(sum((1 if item[flag_key] else 0) * item["weight"] for item in items) / total_weight, 3)


def weighted_list_rate(items, values_key):
    total_weight = sum(item["weight"] for item in items)
    score = 0.0
    for item in items:
        values = item[values_key]
        score += item["weight"] * (sum(values) / len(values))
    return round(score / total_weight, 3)


edit_cases = [
    {
        "id": "ceo_fact",
        "weight": 3,
        "target_ok": True,
        "paraphrase_ok": [1, 1, 0],
        "locality_ok": [1, 1, 1],
        "retain_before": 0.91,
        "retain_after": 0.90,
    },
    {
        "id": "product_fact",
        "weight": 2,
        "target_ok": True,
        "paraphrase_ok": [1, 1, 1],
        "locality_ok": [1, 0, 1],
        "retain_before": 0.88,
        "retain_after": 0.86,
    },
    {
        "id": "city_fact",
        "weight": 1,
        "target_ok": False,
        "paraphrase_ok": [0, 0, 0],
        "locality_ok": [1, 1],
        "retain_before": 0.90,
        "retain_after": 0.89,
    },
    {
        "id": "safety_policy",
        "weight": 4,
        "target_ok": True,
        "paraphrase_ok": [1, 0, 1],
        "locality_ok": [1, 1, 0],
        "retain_before": 0.86,
        "retain_after": 0.80,
    },
]

editing_metrics = {
    "target_success": weighted_rate(edit_cases, "target_ok"),
    "generalization": weighted_list_rate(edit_cases, "paraphrase_ok"),
    "locality": weighted_list_rate(edit_cases, "locality_ok"),
    "retain_before": weighted_average(edit_cases, "retain_before"),
    "retain_after": weighted_average(edit_cases, "retain_after"),
}
editing_metrics["retain_drop"] = round(
    editing_metrics["retain_before"] - editing_metrics["retain_after"], 3
)
edit_conflicts = [
    case["id"]
    for case in edit_cases
    if case["target_ok"] and sum(case["locality_ok"]) < len(case["locality_ok"])
]
editing_metrics["conflict_rate"] = round(len(edit_conflicts) / len(edit_cases), 3)
edit_gates = {
    "success": editing_metrics["target_success"] >= 0.80,
    "generalization": editing_metrics["generalization"] >= 0.65,
    "locality": editing_metrics["locality"] >= 0.75,
    "retain": editing_metrics["retain_drop"] <= 0.05,
    "conflict": editing_metrics["conflict_rate"] <= 0.25,
}

forget_cases = [
    {
        "id": "copyright_profile",
        "weight": 3,
        "leak_before": True,
        "exact_after": False,
        "paraphrase_after": False,
        "multi_turn_after": False,
        "mia_before": 0.82,
        "mia_after": 0.31,
    },
    {
        "id": "private_note",
        "weight": 4,
        "leak_before": True,
        "exact_after": False,
        "paraphrase_after": True,
        "multi_turn_after": False,
        "mia_before": 0.91,
        "mia_after": 0.42,
    },
    {
        "id": "fictional_story",
        "weight": 2,
        "leak_before": True,
        "exact_after": False,
        "paraphrase_after": False,
        "multi_turn_after": True,
        "mia_before": 0.75,
        "mia_after": 0.38,
    },
    {
        "id": "capability_pattern",
        "weight": 5,
        "leak_before": True,
        "exact_after": True,
        "paraphrase_after": True,
        "multi_turn_after": True,
        "mia_before": 0.88,
        "mia_after": 0.67,
    },
]

retain_tasks = [
    {"id": "general_qa", "weight": 3, "before": 0.88, "after": 0.86},
    {"id": "safe_policy", "weight": 4, "before": 0.90, "after": 0.87},
    {"id": "math", "weight": 2, "before": 0.78, "after": 0.77},
    {"id": "code", "weight": 2, "before": 0.74, "after": 0.70},
    {"id": "domain", "weight": 3, "before": 0.83, "after": 0.79},
]

for case in forget_cases:
    case["robust_after"] = (
        case["exact_after"] or case["paraphrase_after"] or case["multi_turn_after"]
    )

unlearning_metrics = {
    "forget_leak_before": weighted_rate(forget_cases, "leak_before"),
    "forget_exact_after": weighted_rate(forget_cases, "exact_after"),
    "paraphrase_leak_after": weighted_rate(forget_cases, "paraphrase_after"),
    "multi_turn_leak_after": weighted_rate(forget_cases, "multi_turn_after"),
    "robust_leak_after": weighted_rate(forget_cases, "robust_after"),
    "mia_before": weighted_average(forget_cases, "mia_before"),
    "mia_after": weighted_average(forget_cases, "mia_after"),
    "retain_before": weighted_average(retain_tasks, "before"),
    "retain_after": weighted_average(retain_tasks, "after"),
}
unlearning_metrics["forget_reduction"] = round(
    unlearning_metrics["forget_leak_before"] - unlearning_metrics["forget_exact_after"], 3
)
unlearning_metrics["membership_risk_drop"] = round(
    unlearning_metrics["mia_before"] - unlearning_metrics["mia_after"], 3
)
unlearning_metrics["retain_drop"] = round(
    unlearning_metrics["retain_before"] - unlearning_metrics["retain_after"], 3
)
unlearning_metrics["severity_weighted_forget_risk"] = round(
    0.5 * unlearning_metrics["forget_exact_after"]
    + 0.3 * unlearning_metrics["paraphrase_leak_after"]
    + 0.2 * unlearning_metrics["multi_turn_leak_after"],
    3,
)
unlearning_risks = [case["id"] for case in forget_cases if case["robust_after"]]
unlearn_gates = {
    "exact_forget": unlearning_metrics["forget_exact_after"] <= 0.10,
    "robust_forget": unlearning_metrics["robust_leak_after"] <= 0.20,
    "membership": unlearning_metrics["membership_risk_drop"] >= 0.30,
    "retain_drop": unlearning_metrics["retain_drop"] <= 0.05,
    "retain_quality": unlearning_metrics["retain_after"] >= 0.75,
}

print("editing_metrics=", editing_metrics)
print("edit_conflicts=", edit_conflicts)
print("edit_gates=", edit_gates)
print("edit_ready=", all(edit_gates.values()))
print("unlearning_metrics=", unlearning_metrics)
print("unlearning_risks=", unlearning_risks)
print("unlearn_gates=", unlearn_gates)
print("unlearning_ready=", all(unlearn_gates.values()))
print("overall_ready=", all(edit_gates.values()) and all(unlearn_gates.values()))
```

运行后可以看到：

```text
editing_metrics= {'target_success': 0.9, 'generalization': 0.667, 'locality': 0.8, 'retain_before': 0.883, 'retain_after': 0.851, 'retain_drop': 0.032, 'conflict_rate': 0.5}
edit_conflicts= ['product_fact', 'safety_policy']
edit_gates= {'success': True, 'generalization': True, 'locality': True, 'retain': True, 'conflict': False}
edit_ready= False
unlearning_metrics= {'forget_leak_before': 1.0, 'forget_exact_after': 0.357, 'paraphrase_leak_after': 0.643, 'multi_turn_leak_after': 0.5, 'robust_leak_after': 0.786, 'mia_before': 0.857, 'mia_after': 0.48, 'retain_before': 0.841, 'retain_after': 0.812, 'forget_reduction': 0.643, 'membership_risk_drop': 0.377, 'retain_drop': 0.029, 'severity_weighted_forget_risk': 0.471}
unlearning_risks= ['private_note', 'fictional_story', 'capability_pattern']
unlearn_gates= {'exact_forget': False, 'robust_forget': False, 'membership': True, 'retain_drop': True, 'retain_quality': True}
unlearning_ready= False
overall_ready= False
```

这个 demo 的重点不是“把指标调到全绿”，而是让你看到真实审计要同时检查两类失败：

1. editing 虽然目标样本大多改对，但仍可能引入局部冲突。
2. unlearning 虽然 exact leak 下降、membership risk 下降，但改写和多轮检查仍可能暴露目标影响。

## 20. 小练习

### 练习 1

比较 RAG、fine-tuning 和 model editing。

要求说明适用场景、优点、缺点和评估重点。

### 练习 2

设计一个 model editing 评估集。

要求包含：目标问题、改写问题、相关但不该改变的问题、无关问题和通用能力测试。

### 练习 3

解释为什么 unlearning 比改一个事实更难。

### 练习 4

为版权数据移除设计一个 unlearning 评估方案。

要求包含：forget set、retain set、改写攻击、多轮追问和人工审核。

### 练习 5

讨论 guardrail suppression 和真正 unlearning 的区别。

## 21. 本章总结

Model Editing 关注在不完整重训模型的情况下局部修改模型知识或行为。

RAG 是外部知识增强，fine-tuning 是全局训练，model editing 是内部局部修改。

ROME 强调定位和编辑 factual association，MEMIT 进一步尝试批量编辑模型记忆。

Unlearning 关注让模型近似移除某些数据、知识、行为或能力影响。

LLM unlearning 很难，因为知识分散在参数中，且同一知识可能来自多个来源。

评估 model editing 要看 success、generalization、specificity 和能力保留。

评估 unlearning 要看 forget set、retain set、遗忘泛化、鲁棒性和可审计证据。

真实治理中，model editing、unlearning、RAG、guardrail、数据治理和审计通常需要组合使用。
