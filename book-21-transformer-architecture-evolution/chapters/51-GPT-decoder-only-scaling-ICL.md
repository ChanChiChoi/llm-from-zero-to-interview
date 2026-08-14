# 第 51 章 GPT 系列：Decoder-only、Scaling 与 In-Context Learning

## 51.1 GPT 改变了什么

GPT 的重要性不只是“使用了 Transformer decoder”。它把预训练语言模型明确成一个统一的自回归接口：给定前缀，预测下一个 token；同一套参数可以通过文本形式承载问答、翻译、摘要、分类和代码任务。

这个接口让模型规模、数据规模和任务形式可以在一个目标下扩展。早期自然语言处理经常为不同任务设计不同的模型头和监督信号；GPT 路线把大量任务转化成序列建模问题，再通过 prompt 或示例告诉模型当前任务。

## 51.2 Decoder-only 的概率目标

给定 token 序列 x_1 到 x_T，因果语言模型分解联合概率：

~~~math
P(x_1,\ldots,x_T)
=\prod_{t=1}^{T}P_\theta(x_t\mid x_{1:t-1})
~~~

训练时最大化对数似然，等价于最小化 next-token cross entropy：

~~~math
\mathcal{L}_{\mathrm{NLL}}
=-\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{1:t-1})
~~~

因果 mask 保证位置 t 不能读取未来 token。训练中可以并行计算所有位置的 logits，因为每个位置的可见范围由 mask 一次性确定；推理中却要逐 token 生成，这就是训练并行和 decode 串行的基本差异。

## 51.3 为什么不是 Encoder-only 或 Encoder-Decoder

Encoder-only 模型适合双向理解和表示学习，但生成长文本时需要额外的 decoder 或任务头。Encoder-decoder 将输入和输出分开，适合翻译、摘要和条件生成，但每个任务都要明确输入输出边界。

Decoder-only 把一切都串成一个序列：

~~~text
instruction + context + answer
~~~

它牺牲了对输入的双向自由度，却获得了统一的扩展接口和大规模数据利用方式。任务格式可以通过分隔符、角色标记和模板表达，工具调用、代码、对话和多轮状态也能放进同一 token 流。

这不是说 decoder-only 在所有任务上理论最优。理解型分类、长输入编码和严格的输入输出分离可能仍适合 encoder 或 encoder-decoder。GPT 的胜利更多来自可扩展性、数据规模、生态和接口统一。

## 51.4 Causal attention 的信息路由

在第 l 层，隐藏状态可以写成：

~~~math
h_t^{(l)}
=\mathrm{Block}^{(l)}
\left(h_{1:t}^{(l-1)}\right)
~~~

attention score 只在 i 小于等于 t 的位置计算：

~~~math
A_{t,i}=
\begin{cases}
\dfrac{q_t^\top k_i}{\sqrt{d_k}}, & i\le t,\\
-\infty, & i>t.
\end{cases}
~~~

这条可见性规则使生成过程保持因果性，也让模型学习“前缀如何决定后续”。但它不保证模型会正确使用所有前缀信息；上下文长度、位置编码、训练数据和干扰都会影响有效检索。

## 51.5 Scaling 的三条轴

GPT 时代最具影响力的经验是，模型能力会随参数、训练 token 和计算预算增加而改善，但增长不是无限线性。常被观察到的近似形式是幂律：

~~~math
L(N)\approx L_\infty+aN^{-\alpha},\qquad
L(D)\approx L_\infty+bD^{-\beta}
~~~

N 表示参数规模，D 表示训练数据量，L 表示验证损失或相关能力代理。不同数据、架构、优化器和评估任务会产生不同指数，公式用于建立趋势，不是跨项目的精确保证。

训练计算可以粗略写成：

~~~math
C\approx kND
~~~

当模型变大但训练 token 不够时，它可能参数很多却没有充分学会；当数据很多但模型太小，模型容量又可能成为瓶颈。后续 Chinchilla 工作正是对参数和数据配比做了更系统的校准。

## 51.6 GPT-3 与 in-context learning

GPT-3 的关键观察是，同一个大模型可以通过上下文中的示例完成新任务，而不必为每个任务更新参数。一个 few-shot prompt 可以写成：

~~~text
input_1 -> output_1
input_2 -> output_2
input_3 -> output_3
input_4 ->
~~~

模型根据前面示例推断任务格式，并生成 output_4。这看起来像学习，但通常不是梯度更新，而是前向计算中利用上下文完成条件预测，所以称为 in-context learning。

ICL 依赖多种因素：示例是否代表目标分布，标签格式是否一致，示例顺序是否影响结果，token 预算是否足够，模型是否真正识别了任务规则。它不是“把几个例子放进去就一定有效”。

## 51.7 ICL 的一个教学实验

用简单的字符串映射构造 four-shot 任务，比较示例顺序、标签格式和干扰：

~~~python
def make_prompt(examples, query, style="plain"):
    if style == "plain":
        rows = [f"{x} -> {y}" for x, y in examples]
        rows.append(f"{query} ->")
    else:
        rows = [f"input={x}; output={y}" for x, y in examples]
        rows.append(f"input={query}; output=")
    return "\n".join(rows)


examples = [("red", "rouge"), ("blue", "bleu")]
print(make_prompt(examples, "green"))
~~~

真实模型实验应固定模型、解码参数和 token 预算，改变示例顺序、标签语言、干扰样本和 query 位置，记录准确率和格式错误。这样才能区分任务推断、位置偏差和模板问题。

## 51.8 ICL 与参数记忆的区别

预训练参数中保存的是跨样本统计规律；上下文提供的是当前请求的临时条件。若同一问题的答案在参数和 prompt 中冲突，模型可能受到位置、表述和训练分布影响。ICL 也可能表现为表面模式匹配，而不是稳定的抽象规则学习。

可以用三类测试区分能力：

1. 规则改变：示例中的映射与模型常识相反。
2. 组合泛化：示例覆盖组件，但 query 要求新组合。
3. 反事实：只改变 prompt 中的规则，观察输出是否随之改变。

如果模型在规则改变后仍坚持训练先验，说明它没有完全把上下文当作当前任务定义。

## 51.9 长上下文和 ICL 的关系

上下文窗口变长只提供了放入更多 token 的可能性，不代表示例都能被使用。示例之间可能互相冲突，重要示例可能落在中间，冗余示例可能挤占输出预算。

对 m 个示例和每个示例平均长度 s，prompt token 近似为：

~~~math
T_{\mathrm{prompt}}\approx T_{\mathrm{instruction}}+ms
~~~

当 T_prompt 增大时，prefill 计算和输入延迟增加；decode 的每步 attention 还要访问更长历史。应同时测任务成功、位置敏感性、TTFT、TPOT 和成本。

## 51.10 现代 GPT-like block 的工程改造

今天的 decoder-only 模型通常不等同于原始 GPT。常见改造包括 RMSNorm 或 Pre-Norm 提高训练稳定性，SwiGLU 提升 FFN 表达，RoPE 提供相对位置结构，GQA/MQA 降低 KV cache，FlashAttention 降低内存读写，MoE 增加总容量，长上下文继续训练改善目标长度能力。

这些改造不是独立徽章。GQA 改变 KV 头数，可能影响质量和 cache；RoPE 扩展需要数据和位置策略；MoE 引入路由和通信；attention kernel 改善系统效率但不改变模型的信息容量。阅读 GPT-like 模型时，要把 block、训练 recipe 和 serving 作为一个整体。

## 51.11 Decoder-only 的边界

第一，逐 token decode 在交互场景中受串行限制。第二，输入越长，prefill 和 KV 成本越高。第三，ICL 对示例选择、顺序和格式敏感。第四，模型可能在训练数据中记忆而不是理解，导致污染和隐私风险。第五，自回归目标不自动提供事实更新、工具验证或安全约束。

因此 GPT 架构是一个强大的预测器，不是完整的知识库、数据库或 Agent 执行器。需要外部检索、工具和验证时，应把它们显式接入系统。

## 51.12 In-Context Learning 不等于参数更新

上下文学习看起来像“模型在 prompt 中学会了任务”，但参数并没有被更新。模型通过 attention 和 residual stream 在当前前缀中组合示例，再把这种临时状态用于后续 token。它更接近条件推断或临时程序执行，而不是传统意义上的训练。

要区分这两种情况，可以比较三组实验：改变示例顺序、删除示例、在请求结束后重新提问。如果能力只在同一请求的上下文中存在，且请求之间不会保存参数或状态，说明它是 in-context adaptation；如果服务层保存了 memory 或 prefix cache，还要进一步区分模型内部上下文和外部系统状态。

## 51.13 Scaling 曲线如何变成工程决策

预训练规模的收益通常呈现边际递减。设增加一单位计算带来的损失下降为：

~~~math
\frac{\partial L}{\partial C}
=\frac{\partial L}{\partial N}\frac{\partial N}{\partial C}
+\frac{\partial L}{\partial D}\frac{\partial D}{\partial C}
~~~

这说明扩大参数、增加数据、改进数据质量和改变训练目标之间存在机会成本。产品还要加入后训练、评测、部署和用户反馈，不能把验证损失的下降直接当成线上价值。

## 51.14 Decoder-only 的系统边界

统一 token 流让 GPT 路线容易接入代码、工具和多模态占位符，但也把很多责任转移给协议层。角色优先级、工具结果可信度、结构化输出、停止条件和上下文裁剪都要由系统明确约束。

因此理解 GPT 的重要结论不是“decoder-only 永远最好”，而是它提供了一个可扩展的概率接口。后续的 RAG、RLHF、function calling、memory 和 serving 技术，都是围绕这个接口补足真实性、可控性和成本。

## 51.15 Scaling 的两条曲线

扩大模型、数据和训练计算通常改变参数记忆与表示能力；增加测试时计算、候选、验证和工具则改变单个请求的求解过程。两者不能用同一条“模型越大越强”的曲线替代。简单问答可能主要受参数规模影响，难数学/代码任务可能受 test-time compute 和 verifier 影响。

评估时要固定或报告总预算：预训练 token、参数、推理 token、候选数、工具时间和 verifier 次数。若一个模型用十倍 reasoning budget 赢得 benchmark，结论应写成系统级比较，而不是单纯架构胜负。

## 51.16 In-context learning 的条件边界

ICL 不是把示例永久写入权重，而是在当前上下文中根据示例推断任务规则、标签映射或风格。它受示例顺序、格式一致性、干扰、位置、输入长度和标签分布影响。模型能模仿示例不代表它理解了抽象规则；应加入反事实示例、顺序置换和新标签测试。

生产系统还要防止示例中混入未授权指令、个人数据或训练集污染。context 里的 demonstration 是外部输入，不能自动越过 system policy；长示例还会占用 KV 和推理预算。

## 51.17 从 GPT 风格骨架到现代协议

decoder-only 的统一 token 接口为指令、代码、多模态 placeholder 和工具调用提供了共同载体，但真正服务仍需要 chat template、special token、grammar、stream event、usage、权限和 state。模型“会续写”只是概率层能力，产品“能执行”还需要协议和执行器。

## 51.18 资料范围与演进判断

GPT-1、GPT-2 和 GPT-3 论文共同展示了 decoder-only 预训练和规模化路线；GPT-3 论文对 few-shot 和 in-context learning 的影响尤其重要。论文中的 benchmark 结果不能直接推断今天所有 GPT-like 模型都具有同样行为，模型版本、数据和后训练都会改变能力。

理解 GPT 的核心是三句话：用 causal next-token objective 统一生成接口；用规模、数据和计算推动能力增长；用上下文提供任务条件，但要通过控制实验判断它是规则学习还是表面匹配。

## 51.19 面试问题与练习

**问：Decoder-only 为什么能做分类？**

把分类任务改写成条件生成，例如在输入后生成标签 token；模型仍然只做 next-token prediction，分类由输出格式和标签概率决定。

**问：ICL 是不是参数更新？**

通常不是。标准 ICL 在推理前向中利用上下文示例改变条件分布，不执行梯度更新；它可能实现任务规则推断，也可能只是匹配表面模式，需要反事实实验区分。

**练习：**设计一个四组 ICL 实验，分别控制示例顺序、标签格式、规则冲突和干扰长度，并说明哪些结果能支持“模型学会了规则”。

资料入口：

- GPT-1: https://openai.com/research/language-unsupervised
- GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT-3: https://arxiv.org/abs/2005.14165
- Scaling laws: https://arxiv.org/abs/2001.08361

## 51.20 ICL 是否真的学到了规则

一个 few-shot 例子让模型答对，并不能证明它学会了任务规则。可以构造四组对照：交换示例顺序，替换无关词面，加入与规则冲突的示例，以及把答案标签换成新符号。若模型只在原始顺序和原始词面下成功，可能是模式匹配；若能在反事实和新标签下保持一致，才更支持规则条件化。

还要把 ICL 与参数记忆分开。删除示例、保留问题，或提供与预训练常识冲突的示例，观察输出变化；如果示例无效而模型坚持旧答案，说明上下文没有覆盖参数先验。长上下文 ICL 还要加入干扰和位置旋转，测的是示例检索、规则组合和输出协议的联合能力。

## 51.21 一个可复现的 ICL 对照实验

要判断模型是否根据示例推断了规则，不能只在一个 prompt 上看答案。可以先构造一个模型没有理由预先知道的符号映射任务。例如随机选择一个排列，把 `A、B、C、D` 映射到四个新标签，再把其中两到四个映射作为 demonstration，最后询问未出现过的组合或新标签。这样做的价值是把“模型记得常识”与“模型读取当前示例”分开。

实验至少要有五个条件：

1. **原始条件**：示例顺序和格式保持不变。
2. **顺序置换**：只交换 demonstration 的顺序，检查位置偏差。
3. **格式置换**：把 `A -> z1` 改成自然语言或 JSON，检查模板依赖。
4. **规则冲突**：故意让上下文映射与常识或预训练先验相反，检查条件覆盖能力。
5. **新标签**：使用模型在上下文外没有语义含义的标签，检查它是否真的使用示例。

对每个条件都固定模型 revision、temperature、最大输出 token、system message 和 stop rule。若只改变一个变量，结果差异才有可归因性。可以把每个 query 的结果记录为：答案是否正确、是否符合格式、是否引用了正确示例、是否在规则冲突时跟随上下文。

设第 `j` 个条件的任务准确率为 `A_j`，格式通过率为 `F_j`，上下文规则冲突时的遵循率为 `C_j`，则一个教学用综合分可以写成：

```math
S_{mathrm{ICL}}
=w_A A_j+w_F F_j+w_C C_j,
\qquad
w_A+w_F+w_C=1.
```

这个分数不是新的通用 benchmark，而是提醒实验者不要把“字面答案对”当成唯一结果。对高风险任务，`C_j` 可能比平均准确率更重要；对结构化抽取，`F_j` 不通过就不能交给下游执行器。

一个最小的评测 harness 可以先只负责生成对照数据和保存条件，模型调用留在外部：

```python
from dataclasses import dataclass
from random import Random


@dataclass
class Case:
    condition: str
    examples: list[tuple[str, str]]
    query: str
    expected: str


def make_cases(seed=7):
    rng = Random(seed)
    labels = ["z1", "z2", "z3", "z4"]
    rng.shuffle(labels)
    mapping = dict(zip("ABCD", labels))
    examples = [(key, mapping[key]) for key in "ABC"]
    return [
        Case("original", examples, "D", mapping["D"]),
        Case("reversed", list(reversed(examples)), "D", mapping["D"]),
        Case("new_label", [(k, "label_" + v[1:]) for k, v in examples],
             "D", "label_" + mapping["D"][1:]),
    ]


for case in make_cases():
    print(case.condition, case.examples, case.query, case.expected)
```

代码本身不会证明模型有 ICL；它只保证不同条件的输入是可审计的。真正的实验还要把每个模型输出与 `expected` 对齐，并报告置信区间、失败样例和 token 成本。若顺序置换导致性能剧烈波动，结论应写成“该 harness 下存在顺序敏感性”，而不是直接写成模型没有推理能力。

## 51.22 Scaling law 如何落到预算选择

幂律关系的工程用途不是预测一个神奇的最终分数，而是比较在固定预算下把资源投入到哪里。设验证损失的近似为：

```math
L(N,D,C)=L_\infty+aN^{-\alpha}+bD^{-\beta}+cC^{-\gamma}.
```

其中 `N` 是参数量，`D` 是训练 token 数，`C` 是训练计算，`a、b、c` 和指数由数据与训练 recipe 决定。因为这些变量并不独立，不能直接把三个下降项相加后当作精确预测；它更适合用来提出预算实验。

例如有两种方案：方案 A 使用更大的模型但较少 token，方案 B 使用较小模型但更多 token。若线上任务主要是低延迟问答，A 还要支付权重加载、显存和 decode 成本；若任务是离线批量评分，B 的吞吐和总 token 预算可能更有优势。预训练损失相同也不能消除后训练数据、工具调用、长上下文和安全回归的差异。

因此发布模型时至少要并列记录：预训练 token、有效 batch、总训练 FLOPs、参数量、active parameters、后训练样本、推理 token 预算、最大并发和单位成功成本。对 reasoning model，若一个模型用更多 test-time token 才获得更高正确率，比较对象应是“模型加预算的系统”，而不是只写“decoder-only 架构更强”。

## 51.23 ICL 的机制解释必须保持条件化

从表示角度，attention 可能让当前 query 读取 demonstration，残差流再把示例中的标签关系传递到输出位置；从行为角度，这可以解释为临时任务程序、条件分布更新或表面模板匹配。但仅凭一个行为结果，不能断言模型内部一定存在一个可解释的“算法模块”。mechanistic interpretation 需要激活分析、对照干预或至少多个行为 probe 支持。

尤其要区分三种状态：

1. **模型内部上下文**：只在当前 forward 的 token 流中存在，请求结束后消失。
2. **服务层状态**：prefix cache、会话历史或外部 memory 在请求之间保存，但参数没有改变。
3. **参数更新**：通过梯度训练或 adapter 写入权重，之后的新请求也能使用。

三者都可能让用户感觉“模型学会了”，但清除上下文、重启服务和加载新 checkpoint 后的行为不同。实验报告如果不写清状态边界，就可能把缓存命中或会话历史误报成 ICL。

## 51.24 公开证据与最终判断

GPT-3 论文支持的事实是：在其公开实验中，模型可以不做梯度更新、仅通过文本任务描述和 few-shot demonstrations 完成多类任务；Scaling Laws 论文支持的是特定训练条件下损失与规模、数据和计算之间的经验关系。它们都不等于“所有 GPT-like 模型都具有相同的 ICL 机制”，也不等于 benchmark 分数可以直接转成产品成功率。

对现代模型，还要额外核对 chat template、后训练、工具协议、上下文裁剪、推理预算和数据污染。只有在这些条件固定后，才可以把结论写成“在某任务和预算下观察到的能力”。

## 51.25 小结

GPT 风格 decoder-only 模型用 causal next-token objective 提供统一生成接口，规模、数据和计算推动能力扩展，ICL 则把任务条件放进当前上下文。真正严谨的学习结论必须通过顺序、格式、冲突和新标签对照来验证，并把模型上下文、服务状态和参数更新分开；真正的工程决策还要把训练预算、推理预算、协议和单位成功成本一起纳入。
