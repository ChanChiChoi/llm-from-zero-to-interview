# 第八部分：Reasoning Model 与 Test-Time Compute

## 第 86 讲：Chain-of-Thought 的机制与争议

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Chain-of-Thought 想解决什么问题。
2. 为什么让模型写中间推理步骤可能提升复杂任务表现。
3. CoT、scratchpad、rationale、hidden reasoning 有什么区别。
4. CoT 为什么存在忠实性、可解释性和安全争议。
5. 如何评估 CoT 是否真的带来推理能力提升。
6. 面试中如何回答“CoT 是真实推理还是语言模式模仿”。

从这一讲开始，我们进入第八部分：Reasoning Model 与 Test-Time Compute。

前面第七部分讲长上下文、RAG 和 Agent。

这些能力让模型能读更多资料、检索外部知识、调用工具和执行多步任务。

但还有一类问题，不只是“有没有信息”，而是：

```text
模型能不能在已有信息上做多步推理。
```

例如：

1. 数学应用题。
2. 逻辑推理题。
3. 代码执行和 bug 定位。
4. 多跳问答。
5. 复杂规划。
6. 科学假设分析。

Chain-of-Thought，简称 CoT，就是最早被广泛讨论的推理增强方法之一。

它的基本想法很简单：

```text
不要让模型直接给答案，而是让模型先写出中间推理步骤，再给最终答案。
```

---

### 一、为什么需要 Chain-of-Thought

普通 prompting 常常要求模型直接回答。

例如：

```text
小明有 3 个苹果，又买了 5 个，吃掉 2 个，还剩几个？
```

直接回答是：

```text
6 个。
```

但对更复杂问题，直接输出答案容易错。

原因是模型需要隐式完成多个中间步骤：

1. 理解题意。
2. 找出变量。
3. 分解子问题。
4. 执行计算。
5. 检查约束。
6. 合成答案。

如果只要求最终答案，模型可能跳步。

CoT 的直觉是：

```text
把隐式推理过程显式写出来，让模型有更多 token 表达中间状态，从而降低一步到位的难度。
```

这类似人类做题时写草稿。

不是因为草稿本身神奇，而是草稿提供了中间表示和检查点。

---

### 二、CoT 的基本形式

最简单的 CoT prompt 是：

```text
Let's think step by step.
```

中文可以写成：

```text
我们一步一步分析。
```

例如：

```text
问题：一个班有 24 人，其中 1/3 是女生，女生有多少人？

推理：总人数是 24，女生占 1/3，所以女生人数是 24 * 1/3 = 8。
答案：8 人。
```

CoT 输出通常包含：

1. 问题重述。
2. 条件提取。
3. 分步计算。
4. 中间结论。
5. 最终答案。

在实际产品中，不一定要把完整 CoT 展示给用户。

可以只让模型内部使用中间推理，再输出简洁答案和必要解释。

---

### 三、为什么 CoT 可能有效

CoT 有几个可能有效的原因。

#### 1. 增加计算 token

自回归模型每生成一个 token 都是在做一次条件计算。

让模型生成中间步骤，相当于给模型更多 test-time compute。

这和第 90 讲要讲的 test-time compute scaling 有直接关系。

#### 2. 降低任务难度

复杂问题被拆成多个简单步骤。

模型不用一步从问题跳到答案。

#### 3. 提供中间状态

中间推理文本可以作为后续 token 的上下文。

例如前面算出 `24 * 1/3 = 8`，后面就可以基于 8 继续推理。

#### 4. 激活训练中见过的解题模式

训练数据中有大量教程、题解、证明、代码注释和解题过程。

CoT prompt 可能激活这些模式。

#### 5. 便于自我检查

有中间步骤后，模型或外部 verifier 更容易检查哪一步错。

但这些解释只是可能机制。

CoT 到底是不是“真实推理”，仍有争议。

---

### 四、CoT、Scratchpad、Rationale 的区别

几个术语容易混。

#### 1. Chain-of-Thought

CoT 通常指模型生成自然语言中间推理步骤。

重点是分步推理。

#### 2. Scratchpad

Scratchpad 是草稿空间。

它可以是自然语言、公式、表格、代码或中间变量。

Scratchpad 不一定给用户看。

#### 3. Rationale

Rationale 是解释或理由。

它可能是模型真实使用的推理过程，也可能只是事后解释。

#### 4. Hidden reasoning

Hidden reasoning 是模型内部或系统内部的推理过程，不直接展示给用户。

它可以用于提升答案质量，同时避免暴露冗长、不稳定或不安全的中间文本。

可以粗略理解为：

```text
CoT = 分步推理文本
scratchpad = 草稿空间
rationale = 给出的理由或解释
hidden reasoning = 不展示的内部推理
```

面试中要小心：

```text
模型输出的 rationale 不一定等于模型真实决策原因。
```

---

### 五、Few-shot CoT 与 Zero-shot CoT

CoT 常见有两种用法。

#### 1. Few-shot CoT

Prompt 中给几个带推理步骤的示例。

例如：

```text
Q: 小红有 2 支笔，又买了 3 支，一共有几支？
A: 她原来有 2 支，又买了 3 支，所以 2 + 3 = 5。答案是 5。

Q: 小明有 10 元，花了 4 元，还剩多少？
A: 他原来有 10 元，花掉 4 元，所以 10 - 4 = 6。答案是 6。

Q: ...
```

Few-shot CoT 的作用是给模型展示解题格式和推理模式。

#### 2. Zero-shot CoT

不提供示例，只加一句：

```text
Let's think step by step.
```

Zero-shot CoT 简单，但不一定稳定。

复杂任务中，示例质量、任务相似性和输出格式约束都很重要。

---

### 六、CoT 的适用场景

CoT 更适合需要多步推理的任务。

例如：

1. 数学题。
2. 符号推理。
3. 逻辑判断。
4. 多跳问答。
5. 复杂条件分析。
6. 代码 reasoning。
7. 规划任务。

CoT 不一定适合所有任务。

例如：

1. 简单事实问答。
2. 翻译。
3. 分类。
4. 情感判断。
5. 简单抽取。

对于简单任务，CoT 可能增加成本，还可能引入多余错误。

一个实用判断是：

```text
如果任务需要多个中间步骤、约束检查或计算，CoT 可能有帮助；如果任务本身很简单，直接回答更好。
```

---

### 七、CoT 的忠实性争议

CoT 最大争议之一是 faithful reasoning。

问题是：

```text
模型写出来的推理步骤，是否真的是它得到答案的原因？
```

有时模型先“知道”答案，再编一段看起来合理的解释。

这叫 post-hoc rationalization。

例如模型给出正确答案，但推理步骤中有错误计算。

或者推理步骤看起来合理，最终答案却不匹配。

这说明：

```text
CoT 可以提高可读性，但不必然等于真实可解释性。
```

评估 CoT 忠实性可以看：

1. 修改中间步骤是否会改变答案。
2. 中间步骤是否可被 verifier 检查。
3. 推理步骤和最终答案是否一致。
4. 模型是否会隐藏关键依据。
5. 生成的理由是否只是模板化解释。

面试中不要把 CoT 简单说成“模型可解释性解决方案”。

更准确说法是：

```text
CoT 提供了可检查的中间文本，但它的忠实性需要额外验证。
```

---

### 八、CoT 的安全争议

CoT 还涉及安全问题。

#### 1. 暴露不稳定推理

模型中间推理可能包含错误假设、偏见或敏感内容。

直接展示给用户会降低可信度。

#### 2. 暴露策略细节

在安全、风控、审核场景中，完整推理可能泄露检测规则。

攻击者可以据此绕过系统。

#### 3. 增加越狱攻击面

用户可能诱导模型暴露内部推理、系统提示或安全策略。

#### 4. 误导用户

流畅的 CoT 可能让错误答案看起来更可信。

#### 5. 训练数据风险

如果用不可靠 CoT 数据训练模型，可能强化错误推理模板。

因此很多系统会选择：

```text
内部使用 reasoning，外部只展示简洁解释或答案依据。
```

这不是否定 CoT，而是区分内部计算和用户可见解释。

---

### 九、CoT 和 Verifier

CoT 的一个重要价值是方便验证。

如果模型只输出答案：

```text
答案：42
```

很难判断错在哪里。

如果模型输出步骤：

```text
第一步...
第二步...
第三步...
答案：42
```

就可以让 verifier 检查每一步。

Verifier 可以是：

1. 规则程序。
2. 单元测试。
3. 数学检查器。
4. 另一个模型。
5. 人工标注。

第 88 讲会详细讲 verifier、process reward model 和 outcome reward model。

本讲先记住一个核心关系：

```text
CoT 生成中间过程，verifier 检查中间过程或最终结果。
```

Reasoning model 的很多后续方法都围绕这条线展开。

---

### 十、CoT 和 Self-Consistency

单条 CoT 可能错。

一个自然想法是：

```text
让模型采样多条推理路径，再选择多数答案或最可信答案。
```

这就是 self-consistency 的核心直觉。

例如同一道数学题采样 10 条 CoT。

如果 7 条得到答案 A，2 条得到答案 B，1 条得到答案 C。

可以选择 A。

这说明 CoT 不只是 prompt 技巧，也可以和采样、搜索、验证结合。

第 87 讲会专门讲 self-consistency。

---

### 十一、CoT 的评估方法

评估 CoT 不能只看推理文字是否漂亮。

要看几个层面。

#### 1. Final answer accuracy

最终答案是否正确。

#### 2. Reasoning step validity

中间步骤是否正确。

#### 3. Faithfulness

中间步骤是否真的影响最终答案。

#### 4. Robustness

换 prompt、换示例、换表达后是否稳定。

#### 5. Efficiency

CoT 增加了多少 token 和延迟，收益是否值得。

#### 6. Error localization

能否通过中间步骤定位错误。

#### 7. Safety

是否泄露敏感推理、策略或有害内容。

一个好的评估应该比较：

```text
direct answer vs CoT vs CoT + self-consistency vs CoT + verifier
```

同时看准确率、成本、延迟和安全风险。

---

### 十二、真实项目中的坑

#### 1. 所有任务都强制 CoT

简单任务强制分步，会增加成本并引入噪声。

#### 2. 只看 CoT 是否流畅

流畅推理不等于正确推理。

#### 3. 把 CoT 当可靠解释

模型可能事后编理由。

#### 4. 直接展示完整内部推理

可能泄露策略、增加误导和安全风险。

#### 5. 示例质量差

Few-shot CoT 示例如果有错误，模型会模仿错误模式。

#### 6. 不做成本评估

CoT 增加输出 token，延迟和费用都会上升。

#### 7. 忽略 final answer extraction

有些模型推理写对了，但最终答案格式不稳定。

生产系统要明确最终答案字段。

---

### 十三、面试问答

#### 问题 1：Chain-of-Thought 的核心思想是什么？

可以这样回答：

```text
CoT 让模型在给最终答案前生成中间推理步骤，把复杂任务分解成多个简单步骤，并用生成的中间文本作为后续推理上下文，从而提升多步推理任务表现。
```

#### 问题 2：为什么 CoT 能提升推理效果？

可以这样回答：

```text
可能原因包括增加 test-time compute、降低一步到位的难度、提供中间状态、激活训练数据中的解题模式，并方便后续 verifier 检查。但它不是所有任务都有效。
```

#### 问题 3：CoT 是真实推理还是模式模仿？

可以这样回答：

```text
不能简单二选一。CoT 可能确实提供了有用的中间计算，也可能包含训练数据中学到的解题模板或事后解释。关键要通过答案准确率、步骤有效性、忠实性和干预实验来判断。
```

#### 问题 4：CoT 有什么争议？

可以这样回答：

```text
主要争议是忠实性和安全性。模型写出的推理不一定是得到答案的真实原因，可能是事后合理化；完整展示 CoT 还可能暴露错误假设、敏感策略或增加攻击面。
```

#### 问题 5：CoT 和 verifier 有什么关系？

可以这样回答：

```text
CoT 生成中间推理过程，verifier 可以检查中间步骤或最终答案。只有 CoT 没有验证仍可能流畅但错误，结合 verifier 可以提高可靠性。
```

#### 问题 6：生产系统应该展示完整 CoT 吗？

可以这样回答：

```text
不一定。生产系统可以内部使用 reasoning 来提升质量，但对用户展示简洁解释、证据或关键步骤。完整 CoT 可能冗长、不稳定，并带来安全和误导风险。
```

---

### 十四、常见误区

1. 误区：CoT 一定提高所有任务表现。
   纠正：CoT 主要适合多步推理，简单任务可能没收益甚至变差。

2. 误区：写得越长越会推理。
   纠正：长推理可能只是冗余或错误累积。

3. 误区：CoT 就是可解释性。
   纠正：CoT 是可见中间文本，不保证忠实反映模型真实原因。

4. 误区：最终答案对，CoT 就一定对。
   纠正：答案可能碰巧对，中间步骤可能错。

5. 误区：完整展示 CoT 最透明。
   纠正：完整展示可能带来安全、隐私和误导风险。

6. 误区：CoT 只是 prompt 技巧。
   纠正：CoT 也连接 test-time compute、self-consistency、verifier 和 reasoning model 训练。

---

### 十五、小练习

1. 给一个数学题分别写 direct answer prompt 和 CoT prompt。
2. 构造一个 CoT 中间步骤错误但最终答案正确的例子。
3. 构造一个 CoT 看起来流畅但最终答案错误的例子。
4. 比较 CoT、scratchpad、rationale 和 hidden reasoning。
5. 设计一个评估 CoT 忠实性的实验。
6. 设计一个 CoT + verifier 的数学题检查流程。
7. 分析为什么简单分类任务不一定需要 CoT。
8. 设计一个生产系统中“内部推理、外部简洁解释”的输出格式。
9. 比较 direct answer、CoT、CoT + self-consistency 的成本和收益。
10. 用 3 分钟回答：“CoT 是真实推理还是语言模式模仿？”

### 本讲总结

本讲最重要的结论：

1. CoT 让模型在最终答案前生成中间推理步骤，适合多步推理任务。
2. CoT 可能通过增加 test-time compute、降低任务难度和提供中间状态提升表现。
3. CoT、scratchpad、rationale 和 hidden reasoning 不是同一个概念。
4. CoT 的核心争议是忠实性：模型写出的理由不一定是真实决策原因。
5. CoT 还有安全争议，完整展示内部推理可能泄露策略、增加误导和攻击面。
6. CoT 最好和 self-consistency、verifier、搜索或过程监督结合，而不是单独依赖。
7. 评估 CoT 要同时看最终答案、中间步骤、忠实性、鲁棒性、成本和安全。
8. 面试中要把 CoT 讲成 reasoning 和 test-time compute 的入口，而不是简单一句 prompt 技巧。

## 第 87 讲：Self-Consistency 与采样增强推理

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Self-consistency 想解决 CoT 的什么问题。
2. 为什么多条推理路径投票可能提升推理准确率。
3. Self-consistency 和 temperature、top-p、采样次数有什么关系。
4. Majority vote、weighted vote、verifier selection 有什么区别。
5. Self-consistency 的成本、失败模式和适用边界是什么。
6. 面试中如何设计一个 CoT + self-consistency 的数学推理系统。

第 86 讲讲了 Chain-of-Thought。

CoT 的核心是让模型写出中间推理步骤，再给最终答案。

但单条 CoT 有一个明显问题：

```text
模型可能沿着一条错误推理路径走到底。
```

如果只采样一次，结果高度依赖这一次生成。

Self-consistency 的想法是：

```text
不要只相信一条推理链，而是采样多条推理链，再从多个答案中选出最一致的答案。
```

这类似人类做复杂题时用多种方法验算。

如果不同路径都得到同一个答案，这个答案更可信。

---

### 一、Self-Consistency 的基本直觉

同一个问题可能有多条推理路径。

例如一个数学题可以：

1. 用代数方法解。
2. 用画图方法解。
3. 用枚举方法解。
4. 用逆向验证解。

单次 CoT 只探索一条路径。

如果这条路径错了，最终答案很可能错。

Self-consistency 通过随机采样探索多条路径。

流程是：

```text
问题
-> 采样 N 条 CoT
-> 提取每条 CoT 的最终答案
-> 对答案投票或打分
-> 选择最一致的答案
```

例如采样 5 条：

```text
路径 1 -> 答案 A
路径 2 -> 答案 A
路径 3 -> 答案 B
路径 4 -> 答案 A
路径 5 -> 答案 C
```

最终选择 A。

核心假设是：

```text
正确答案更可能被多条独立或半独立推理路径得到。
```

这个假设不总是成立，但在很多数学和符号推理任务中有效。

---

### 二、为什么不是贪心解码

普通 greedy decoding 每一步选概率最高的 token。

它输出稳定，但缺少探索。

Self-consistency 通常需要非贪心采样。

例如：

1. 设置 temperature > 0。
2. 使用 top-p 或 top-k sampling。
3. 生成多条不同 CoT。

如果每次都 greedy，采样 10 次也可能完全一样。

那就没有 self-consistency 的意义。

所以 self-consistency 的关键不是“重复问 10 次”。

而是：

```text
通过采样得到多样化推理路径，再利用答案一致性聚合结果。
```

---

### 三、Self-Consistency 的算法流程

一个最小流程如下：

```text
Input: question q, model M, sample count N

for i in 1..N:
    reasoning_i = M.generate(q, prompt="think step by step", temperature=t)
    answer_i = extract_final_answer(reasoning_i)

final_answer = majority_vote(answer_1, ..., answer_N)
return final_answer
```

关键模块有三个。

#### 1. 采样

生成多条 reasoning path。

#### 2. 答案抽取

从每条 CoT 中抽取最终答案。

#### 3. 聚合

对答案做 majority vote、weighted vote 或 verifier selection。

这三个模块任何一个做不好，效果都会下降。

---

### 四、答案抽取为什么重要

Self-consistency 依赖最终答案聚合。

但模型输出可能不规范。

例如：

```text
所以答案应该是 12。
```

```text
最终结果：12 个。
```

```text
答案是十二。
```

```text
因此 x = 12。
```

这些都应该归一化成同一个答案。

答案抽取常见问题：

1. 输出多个候选答案。
2. 单位不同。
3. 小数和分数等价。
4. 中文数字和阿拉伯数字混用。
5. 最终答案格式不固定。
6. 推理中间出现多个数字，误抽取中间值。

因此生产系统通常会要求模型输出结构化字段：

```json
{
  "reasoning_summary": "...",
  "final_answer": "12"
}
```

或者在推理后再调用一个 answer extractor。

---

### 五、Majority Vote

最简单聚合方式是 majority vote。

也就是选择出现次数最多的答案。

例如：

```text
A: 7 次
B: 2 次
C: 1 次
```

选择 A。

优点：

1. 简单。
2. 不需要额外模型。
3. 对随机错误有一定鲁棒性。

缺点：

1. 如果多数路径共享同一个错误偏差，会选错。
2. 不能区分推理质量。
3. 对开放答案、长文本答案不容易投票。
4. 答案归一化困难。

Majority vote 最适合答案空间比较明确的任务。

例如数学题、选择题、短答案题。

---

### 六、Weighted Vote

Weighted vote 会给不同答案或路径不同权重。

权重可以来自：

1. 模型生成概率。
2. 答案置信度。
3. 推理步骤质量分。
4. Verifier 分数。
5. 路径长度惩罚。
6. 工具验证结果。

例如：

```text
答案 A: 路径 1 分数 0.8 + 路径 2 分数 0.7 = 1.5
答案 B: 路径 3 分数 0.95 = 0.95
```

选择 A。

Weighted vote 比 majority vote 更灵活。

但也更依赖评分是否可靠。

如果模型自己的 confidence 不校准，weighted vote 可能被高置信错误路径误导。

---

### 七、Verifier Selection

更强的方法是引入 verifier。

流程是：

```text
采样多条 CoT
-> 提取候选答案
-> 用 verifier 检查每条推理或答案
-> 选择 verifier 分数最高的结果
```

Verifier 可以检查：

1. 数学步骤是否成立。
2. 代码是否通过测试。
3. 答案是否满足约束。
4. 推理是否引用了正确证据。
5. 最终答案是否和题目条件一致。

Self-consistency 和 verifier 的区别是：

```text
self-consistency 依赖多条路径的一致性，verifier 依赖外部或额外模型判断质量。
```

两者可以结合。

例如先投票得到 top candidates，再用 verifier 选择最终答案。

第 88 讲会专门展开 verifier。

---

### 八、采样参数怎么选

Self-consistency 的效果和采样参数关系很大。

#### 1. Temperature

Temperature 控制随机性。

太低：路径太相似，缺少多样性。

太高：路径太随机，错误增多。

#### 2. Top-p

Top-p 控制候选 token 累积概率范围。

较高 top-p 增加多样性，但也可能增加噪声。

#### 3. Sample count N

N 越大，覆盖路径越多。

但成本线性增加。

常见实验会试：

```text
N = 5, 10, 20, 40
```

但生产系统要结合延迟和成本。

#### 4. Max tokens

CoT 太长会增加成本，也可能跑偏。

需要限制最大推理长度。

一个实用原则是：

```text
采样要足够多样，但不能让路径完全随机；N 要足够提升准确率，但不能让成本失控。
```

---

### 九、为什么 Self-Consistency 有效

可以从几个角度理解。

#### 1. 错误路径不完全一致

如果错误是随机的，不同路径会错到不同答案。

正确答案更可能集中。

#### 2. 多路径探索

复杂问题可能有多个解法。

一次采样没找到正确路径，多次采样可能找到。

#### 3. 集成效应

Self-consistency 类似 ensemble。

多个样本聚合通常比单个样本更稳。

#### 4. 增加 test-time compute

采样 N 条路径相当于把推理时计算扩大 N 倍。

这本质上是一种 test-time compute scaling。

但要注意：

```text
如果模型系统性误解题目，多采样只会更稳定地错。
```

---

### 十、适用场景

Self-consistency 适合：

1. 数学题。
2. 逻辑题。
3. 选择题。
4. 短答案多跳问答。
5. 可自动抽取答案的任务。
6. 有 verifier 的代码或工具任务。

不太适合：

1. 开放式创作。
2. 长篇总结。
3. 主观评价。
4. 答案难以归一化的任务。
5. 对低延迟要求很高的在线场景。

例如客服简单问答不适合每次采样 20 条。

但高价值数学评测、代码修复、复杂决策辅助可能值得。

---

### 十一、成本和延迟

Self-consistency 最大代价是成本。

如果采样 N 条，每条 CoT 平均 T 个 token。

输出成本大约增加为：

```text
N * T
```

延迟取决于是否并行。

如果串行采样，延迟也约增加 N 倍。

如果并行采样，延迟接近最长那条路径，但吞吐和费用仍增加。

生产系统常用优化：

1. 只对困难问题启用。
2. 先用小 N，不确定时再加采样。
3. 并行采样。
4. 提前停止：某答案已经明显领先。
5. 对简单任务直接回答。
6. 用 verifier 减少无效采样。

Self-consistency 是准确率和成本之间的 trade-off。

---

### 十二、失败模式

#### 1. 多数一致但都错

如果模型有系统性偏差，多数投票会强化错误。

#### 2. 答案抽取错误

推理正确，但 extractor 抽错最终答案。

#### 3. 归一化失败

`1/2`、`0.5`、`50%` 被当成不同答案。

#### 4. 多样性不足

采样参数太保守，N 条路径几乎一样。

#### 5. 随机性过强

temperature 太高，推理质量下降。

#### 6. 开放答案无法投票

长文本答案难以做 majority vote。

#### 7. 成本失控

对所有请求都多采样，延迟和费用不可接受。

---

### 十三、评估方法

评估 self-consistency 至少要比较四组。

```text
direct answer
single CoT
CoT + self-consistency
CoT + self-consistency + verifier
```

指标包括：

1. Accuracy。
2. Pass@k 或 solve rate。
3. Majority confidence。
4. Answer extraction accuracy。
5. Cost per solved problem。
6. Latency。
7. Token usage。
8. Robustness across prompts。
9. Calibration。

还要画成本收益曲线。

例如：

```text
N=1  accuracy=70%, cost=1x
N=5  accuracy=78%, cost=5x
N=20 accuracy=82%, cost=20x
```

如果 N 从 5 到 20 只提升 4%，但成本变 4 倍，生产中未必值得。

---

### 十四、真实项目中的坑

#### 1. 以为多采样一定更好

如果模型系统性错，多采样只会更贵地错。

#### 2. 不做答案归一化

等价答案被拆成多个类别，投票失效。

#### 3. 忽略推理路径质量

多数答案可能来自低质量推理。

#### 4. 对所有任务启用

简单任务不需要 self-consistency。

#### 5. 只报告最高准确率

不报告 token、延迟和成本，评估不完整。

#### 6. 采样路径不独立

prompt 和参数导致路径高度相似，投票没有意义。

#### 7. 最终答案格式不固定

导致 extractor 不稳定，线上结果抖动。

---

### 十五、面试问答

#### 问题 1：Self-consistency 的核心思想是什么？

可以这样回答：

```text
Self-consistency 不是只采样一条 CoT，而是采样多条推理路径，抽取每条路径的最终答案，再通过投票或 verifier 选择最一致或最可信的答案。
```

#### 问题 2：为什么 self-consistency 能提升推理准确率？

可以这样回答：

```text
因为复杂问题可能有多条推理路径，单条路径容易偶然出错。多路径采样能探索不同解法，如果错误较分散而正确答案更一致，投票就能提升准确率。
```

#### 问题 3：Self-consistency 和 temperature 有什么关系？

可以这样回答：

```text
Self-consistency 需要一定随机性来生成多样化推理路径。temperature 太低路径相似，太高推理变噪声，所以要在多样性和质量之间调参。
```

#### 问题 4：Majority vote 和 verifier selection 有什么区别？

可以这样回答：

```text
Majority vote 选择出现次数最多的答案，不判断每条推理质量；verifier selection 用规则、模型或工具检查候选路径或答案，选择验证分数最高的结果。两者可以结合。
```

#### 问题 5：Self-consistency 的主要缺点是什么？

可以这样回答：

```text
主要缺点是成本和延迟高，答案抽取和归一化困难，多数路径可能共享系统性错误，对开放式长答案不容易投票，也不适合所有低延迟场景。
```

#### 问题 6：如何设计一个 CoT + self-consistency 数学推理系统？

可以这样回答：

```text
先用 CoT prompt 并行采样多条推理路径，再用结构化格式抽取 final answer，对等价答案做归一化，然后 majority vote 或 verifier selection，最后输出最终答案和简洁解释，同时记录 N、成本、延迟和正确率。
```

---

### 十六、常见误区

1. 误区：Self-consistency 就是重复问模型多次。
   纠正：关键是采样多样化推理路径，并对最终答案做聚合。

2. 误区：采样越多越好。
   纠正：采样越多成本越高，收益会递减。

3. 误区：多数投票一定正确。
   纠正：模型系统性错误时，多数也会错。

4. 误区：不需要答案抽取。
   纠正：答案抽取和归一化是 self-consistency 成败关键。

5. 误区：Self-consistency 适合所有任务。
   纠正：它更适合答案可归一化的推理任务。

6. 误区：只看准确率提升。
   纠正：还要看 token、延迟、成本和线上可用性。

---

### 十七、小练习

1. 用伪代码写出 self-consistency 推理流程。
2. 给一个数学题设计 N=5 的 CoT 采样和投票示例。
3. 设计一个 final answer extractor，处理中文数字、单位和分数。
4. 比较 majority vote、weighted vote 和 verifier selection。
5. 分析 temperature 太低和太高分别会怎样影响 self-consistency。
6. 设计一个提前停止策略，减少采样成本。
7. 构造一个多数一致但答案错误的案例。
8. 设计一个成本收益实验，比较 N=1、5、10、20。
9. 分析为什么长篇开放问答不适合简单 majority vote。
10. 用 3 分钟回答：“Self-consistency 为什么能提升 reasoning？”

### 本讲总结

本讲最重要的结论：

1. Self-consistency 用多条 CoT 采样和答案聚合提升推理稳定性。
2. 它的核心假设是正确答案更可能被多条不同推理路径一致得到。
3. Self-consistency 需要适当随机性，greedy 重复生成没有意义。
4. 答案抽取和归一化是系统实现中的关键细节。
5. Majority vote 简单有效，但不能判断推理质量；verifier selection 更强但更复杂。
6. Self-consistency 是一种 test-time compute scaling，用更多推理时计算换准确率。
7. 它的主要代价是 token、延迟和成本，并且会受到系统性错误影响。
8. 面试中要强调准确率收益和工程成本之间的 trade-off。

## 第 88 讲：Verifier、Process Reward Model 与 Outcome Reward Model

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Verifier 在 reasoning model 中解决什么问题。
2. Outcome Reward Model 和 Process Reward Model 的区别是什么。
3. 为什么只看最终答案不够，为什么过程监督更难也更有价值。
4. Verifier 如何和 CoT、self-consistency、search 结合。
5. PRM/ORM 的数据、训练、评估和失败模式有哪些。
6. 面试中如何设计一个数学推理或代码推理的 verifier 系统。

第 86 讲讲 CoT，第 87 讲讲 self-consistency。

它们都在做一件事：

```text
生成更多推理过程，然后从中选出更好的答案。
```

但关键问题是：

```text
怎么知道哪条推理更好？
```

如果只靠 majority vote，多数答案也可能错。

如果只靠模型自信，模型可能高置信地错。

Verifier 就是为了解决这个问题：

```text
给候选答案或推理过程打分，判断它是否正确、可信、满足约束。
```

在 reasoning model 里，verifier 往往和 CoT、self-consistency、search、test-time compute scaling 一起使用。

---

### 一、什么是 Verifier

Verifier 可以理解为验证器。

它的输入通常是：

1. 原始问题。
2. 模型生成的答案。
3. 可选的中间推理步骤。
4. 可选的工具执行结果。
5. 可选的参考答案或约束。

输出通常是：

1. 正确或错误。
2. 一个分数。
3. 哪一步有问题。
4. 是否满足约束。
5. 候选之间的排序。

例如数学题：

```text
问题：解方程 2x + 3 = 11。
候选推理：2x = 8，所以 x = 4。
Verifier 输出：正确，score=0.98。
```

例如代码题：

```text
候选代码通过 18/20 个测试。
Verifier 输出：部分正确，score=0.72，失败用例是边界条件。
```

Verifier 不一定是神经网络。

它可以是规则、程序、单元测试、符号检查器、另一个模型或人工标注。

---

### 二、为什么需要 Verifier

只让模型生成答案有几个问题。

#### 1. 模型会流畅地错

错误推理可能写得很像真的。

#### 2. 多条 CoT 不知道选哪条

Self-consistency 可以投票，但不能判断少数路径是否其实更正确。

#### 3. 最终答案对不代表过程对

模型可能碰巧得到正确答案，但中间逻辑错误。

#### 4. 最终答案错不代表全程错

模型可能前面都对，最后一步计算错。

如果能定位错误步骤，就更容易改进。

#### 5. Search 需要评价函数

Tree-of-Thought、MCTS 等搜索方法需要评估中间状态好坏。

Verifier 可以提供评价信号。

因此 verifier 是 reasoning 系统中的“判题器”或“评审器”。

---

### 三、Outcome Reward Model

Outcome Reward Model，简称 ORM。

它评价的是最终结果。

输入可以是：

```text
question + final answer
```

也可以包含推理过程：

```text
question + reasoning + final answer
```

但监督信号通常来自最终答案是否正确。

例如：

```text
问题：24 的 1/3 是多少？
答案：8
label: correct
```

ORM 的优点：

1. 标注相对简单。
2. 很多任务天然有最终答案。
3. 适合对多个候选答案排序。
4. 可以和 self-consistency 结合。

ORM 的缺点：

1. 不知道哪一步错。
2. 对过程质量监督弱。
3. 可能奖励碰巧正确的错误推理。
4. 对开放题最终正确性难判断。

ORM 适合回答：

```text
这个候选最终结果是否好？
```

但不擅长回答：

```text
这个推理过程每一步是否对？
```

---

### 四、Process Reward Model

Process Reward Model，简称 PRM。

它评价的是推理过程中的每一步。

例如一个推理过程：

```text
Step 1: 2x + 3 = 11
Step 2: 2x = 8
Step 3: x = 4
```

PRM 可以给每一步打分：

```text
Step 1: correct
Step 2: correct
Step 3: correct
```

如果推理是：

```text
Step 1: 2x + 3 = 11
Step 2: 2x = 14
Step 3: x = 7
```

PRM 应该指出 Step 2 错。

PRM 的优点：

1. 能定位错误步骤。
2. 能指导模型改进过程。
3. 适合搜索中评估中间状态。
4. 比只看最终答案更细粒度。
5. 有助于训练更稳定的 reasoning。

PRM 的缺点：

1. 标注成本高。
2. 需要定义“步骤”粒度。
3. 有些推理步骤很难判断对错。
4. 标注者之间可能不一致。
5. 模型可能学会迎合 PRM，而不是真正解决问题。

PRM 适合回答：

```text
这一步推理是否合理？下一步是否值得继续？
```

---

### 五、ORM 和 PRM 对比

可以用一张表理解。

| 维度 | ORM | PRM |
|---|---|---|
| 监督对象 | 最终答案 | 推理步骤 |
| 标注成本 | 较低 | 较高 |
| 错误定位 | 弱 | 强 |
| 是否奖励好过程 | 不一定 | 是 |
| 适合搜索 | 可用于终局评价 | 可用于中间状态评价 |
| 风险 | 奖励碰巧正确 | 步骤标注主观、成本高 |

一句话总结：

```text
ORM 看结果，PRM 看过程。
```

但真实系统不一定二选一。

可以同时使用：

1. PRM 评估中间步骤。
2. ORM 评估最终答案。
3. 工具或规则做硬验证。

---

### 六、Verifier 的类型

Verifier 可以分成几类。

#### 1. Rule-based verifier

用规则检查答案格式、约束、单位、范围。

例如答案必须是整数，必须在 0 到 1 之间。

#### 2. Programmatic verifier

用程序执行检查。

例如代码题跑单元测试，数学题用符号计算验证。

#### 3. Neural verifier

训练一个模型判断答案或过程是否正确。

ORM 和 PRM 通常属于这一类。

#### 4. LLM-as-a-verifier

用另一个 LLM 判断候选答案。

优点是通用。

缺点是也会错，需要校准。

#### 5. Human verifier

人工标注或审核。

成本高，但适合高风险和校准集。

实际系统常用混合方案：

```text
规则过滤 + 程序验证 + 神经 verifier + 人工抽查
```

---

### 七、Verifier 和 CoT 怎么结合

CoT 生成过程，verifier 检查过程。

最简单流程：

```text
生成 CoT
-> verifier 检查最终答案
-> 如果不通过，重新生成
```

更强流程：

```text
生成多条 CoT
-> verifier 给每条打分
-> 选择最高分
```

更细粒度流程：

```text
每生成一步
-> PRM 打分
-> 分数低则剪枝或回退
-> 分数高则继续
```

这就接近搜索方法。

第 89 讲会讲 Tree-of-Thought 和 MCTS。

本讲先记住：

```text
Verifier 可以把“生成很多答案”变成“生成、筛选、改进答案”。
```

---

### 八、Verifier 和 Self-Consistency 怎么结合

第 87 讲讲了 self-consistency。

它默认多数答案更可信。

但有时少数答案才对。

因此可以加入 verifier。

流程一：先投票，再验证。

```text
采样 N 条 CoT
-> majority vote 得到 top answer
-> verifier 检查 top answer
```

流程二：先验证，再投票。

```text
采样 N 条 CoT
-> verifier 过滤低质量路径
-> 对剩余答案投票
```

流程三：直接按 verifier 分数选。

```text
采样 N 条 CoT
-> verifier 给每条路径打分
-> 选择最高分路径
```

工程上通常要比较这几种策略的准确率、成本和稳定性。

---

### 九、PRM 数据怎么标注

PRM 的难点是过程标注。

一个样本通常包含：

```text
问题
推理步骤 1
步骤 1 标签
推理步骤 2
步骤 2 标签
...
最终答案
```

标签可以是：

1. correct / incorrect。
2. 分数 0-1。
3. 哪一步开始错误。
4. 错误类型。
5. 是否可继续。

标注来源包括：

1. 人工专家标注。
2. 程序自动验证。
3. LLM 辅助标注。
4. 从正确解答中自动构造。
5. 从错误模型输出中挖掘负样本。

PRM 数据要覆盖错误步骤。

如果训练集只有完美推理，PRM 学不会识别错误。

---

### 十、ORM 数据怎么标注

ORM 数据相对简单。

样本通常是：

```text
问题 + 候选答案 + label/score
```

标注来源包括：

1. 有标准答案的数据集。
2. 单元测试通过率。
3. 人工偏好比较。
4. LLM judge。
5. 用户反馈。

ORM 可以做二分类：

```text
correct vs incorrect
```

也可以做 pairwise ranking：

```text
候选 A 比候选 B 更好
```

或者直接回归分数。

ORM 的关键是负样本质量。

太容易的负样本没用。

需要 hard negatives，例如：

1. 最终答案差一点。
2. 推理看起来合理但有隐藏错误。
3. 格式正确但约束不满足。
4. 常见误解导致的错误。

---

### 十一、训练目标

Verifier 的训练目标取决于任务。

#### 1. Classification

判断正确或错误。

```text
P(correct | question, answer)
```

#### 2. Regression

预测一个连续分数。

例如 0 到 1 的质量分。

#### 3. Ranking

学习候选之间的相对好坏。

例如 pairwise loss：

```text
score(good) > score(bad)
```

#### 4. Step-wise classification

PRM 对每一步判断是否正确。

#### 5. Value function

在搜索中预测当前 partial solution 最终成功概率。

面试中不必背复杂公式。

重点是讲清：

```text
Verifier 本质上是在学习一个评价函数，用来判断候选推理或答案的质量。
```

---

### 十二、Verifier 的评估

Verifier 本身也要评估。

不能因为它叫 verifier 就默认可靠。

指标包括：

1. Accuracy。
2. Precision / recall。
3. AUC。
4. Pairwise ranking accuracy。
5. Calibration。
6. 对 hard negatives 的识别率。
7. 和人工标注一致性。
8. 对下游 answer accuracy 的提升。
9. 成本和延迟。

特别重要的是 calibration。

如果 verifier 给错误答案高分，就会误导整个系统。

还要做 end-to-end 评估：

```text
没有 verifier 的准确率
vs 有 verifier rerank 的准确率
vs verifier + search 的准确率
```

Verifier 的价值最终要体现在下游任务上。

---

### 十三、失败模式

#### 1. Verifier 被候选答案欺骗

候选推理写得很像正确，verifier 打高分。

#### 2. Reward hacking

生成模型学会迎合 verifier 的偏好，而不是解决问题。

例如写更长、更像数学证明的文本骗分。

#### 3. PRM 标注不一致

不同标注者对某一步是否合理看法不同。

#### 4. ORM 奖励碰巧正确

推理错误但答案对，ORM 仍给高分。

#### 5. 对分布外问题失效

Verifier 在训练分布上好，遇到新题型不可靠。

#### 6. 过度惩罚非标准解法

有些正确推理路径和训练样本不同，被 PRM 打低分。

#### 7. 计算成本高

每个候选都调用 verifier，成本和延迟增加。

---

### 十四、真实项目中的坑

#### 1. 把 LLM judge 当绝对真理

LLM verifier 也会幻觉，也会偏。

#### 2. 只训练 ORM，不看过程

最终答案正确率提升了，但模型过程可能更不可靠。

#### 3. PRM 步骤粒度混乱

一步太大无法定位错误，一步太小标注成本爆炸。

#### 4. 没有 hard negatives

Verifier 只会区分明显错误，区分不了高迷惑错误。

#### 5. 不评估下游收益

Verifier 离线 AUC 高，不代表能提升最终系统准确率。

#### 6. 忽略成本

Verifier 让准确率提升 1%，但延迟翻倍，线上未必可接受。

#### 7. 让生成模型和 verifier 共同过拟合

长期优化同一个 verifier，可能出现 reward hacking。

---

### 十五、面试问答

#### 问题 1：Verifier 在 reasoning model 中的作用是什么？

可以这样回答：

```text
Verifier 是评价候选答案或推理过程质量的模块。它可以检查最终答案、推理步骤或中间状态，帮助从多条 CoT、搜索路径或候选答案中选择更可靠的结果。
```

#### 问题 2：ORM 和 PRM 的区别是什么？

可以这样回答：

```text
ORM 评价最终答案，看结果是否正确；PRM 评价推理过程中的每一步，看每一步是否合理。ORM 标注较容易但不能定位错误，PRM 标注更贵但能提供细粒度过程监督。
```

#### 问题 3：为什么只看最终答案不够？

可以这样回答：

```text
因为最终答案可能碰巧正确，也可能答案错但前面大部分推理正确。只看 outcome 无法定位错误步骤，也难以指导搜索和过程改进，所以复杂 reasoning 中需要过程监督或 verifier。
```

#### 问题 4：PRM 为什么难？

可以这样回答：

```text
PRM 需要对每个推理步骤标注质量，成本高，而且要定义步骤粒度、处理多种正确解法、保证标注一致性，并避免模型学会迎合 PRM 而不是真正推理。
```

#### 问题 5：Verifier 如何和 self-consistency 结合？

可以这样回答：

```text
可以先采样多条 CoT，再用 majority vote 得到候选，也可以让 verifier 给每条路径打分后选择最高分，或者先过滤低质量路径再投票。Verifier 弥补了多数投票无法判断推理质量的问题。
```

#### 问题 6：如何评估 verifier？

可以这样回答：

```text
既要评估 verifier 自身的 accuracy、AUC、ranking accuracy、calibration 和 hard negative 识别能力，也要评估它是否提升下游任务准确率、稳定性和成本收益。
```

---

### 十六、常见误区

1. 误区：Verifier 一定比 generator 更可靠。
   纠正：Verifier 也会错，也要评估和校准。

2. 误区：ORM 足够解决 reasoning。
   纠正：ORM 只看结果，不能定位过程错误。

3. 误区：PRM 标注越细越好。
   纠正：过细会增加成本和噪声，要选择合适步骤粒度。

4. 误区：Verifier 分数高就一定正确。
   纠正：可能存在 reward hacking 或分布外失效。

5. 误区：LLM-as-a-verifier 不需要人工校准。
   纠正：LLM judge 需要标准集、人工抽查和一致性评估。

6. 误区：只看 verifier 离线指标。
   纠正：最终要看 end-to-end 任务收益和成本。

---

### 十七、小练习

1. 给一个数学题设计 ORM 输入输出样本。
2. 给一个三步推理过程设计 PRM step labels。
3. 构造一个最终答案正确但中间推理错误的例子。
4. 构造一个最终答案错误但前两步正确的例子。
5. 设计一个 CoT + verifier rerank 流程。
6. 比较 rule-based verifier、programmatic verifier 和 neural verifier。
7. 设计一组 hard negatives，用于训练数学 verifier。
8. 设计 verifier 的离线指标和端到端评估指标。
9. 分析 reward hacking 在 verifier 系统中如何发生。
10. 用 3 分钟回答：“ORM 和 PRM 的区别与取舍是什么？”

### 本讲总结

本讲最重要的结论：

1. Verifier 用来评价候选答案、推理过程或中间状态的质量。
2. ORM 看最终结果，PRM 看推理过程。
3. ORM 标注更容易，但不能定位过程错误；PRM 更细粒度，但标注更难。
4. Verifier 可以和 CoT、self-consistency、search 结合，用于筛选和改进候选推理。
5. PRM 的关键难点是步骤粒度、过程标注、标注一致性和 reward hacking。
6. Verifier 本身也需要评估，包括准确率、排序能力、校准和下游收益。
7. 真实系统常用规则、程序、神经模型和人工审核的混合 verifier。
8. 面试中要强调 verifier 是 reasoning system 的评价函数，而不是天然可靠的真理源。

## 第 89 讲：Search、Tree-of-Thought 与 MCTS

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 为什么 reasoning 不只是生成一条 CoT，还可以做搜索。
2. Tree-of-Thought 想解决什么问题，和 Chain-of-Thought 有什么区别。
3. Search 中的 state、action、value、policy 分别是什么。
4. BFS、DFS、beam search、MCTS 在 reasoning 中如何理解。
5. Verifier/PRM 如何作为搜索中的评价函数。
6. 面试中如何设计一个用搜索增强数学、代码或规划推理的系统。

第 86 讲讲 CoT：生成一条推理链。

第 87 讲讲 self-consistency：采样多条推理链后投票。

第 88 讲讲 verifier：判断候选答案或推理过程质量。

这一讲进一步问：

```text
能不能把推理过程显式组织成搜索？
```

人类解决难题时经常不是一条路走到底。

我们会尝试多个思路，发现不对就回退，比较不同路径，再继续深入。

Search、Tree-of-Thought 和 MCTS 就是在大模型 reasoning 中引入这种“探索、评价、选择、回退”的思想。

---

### 一、为什么需要搜索

单条 CoT 的问题是：

```text
一旦早期步骤错了，后面可能沿着错误方向继续生成。
```

Self-consistency 虽然采样多条完整路径，但它通常是“先生成完整答案，再比较”。

它没有在中间步骤主动剪枝或调整。

搜索的目标是：

```text
把推理看成一系列状态转移，在中间阶段评估哪些路径更有希望，然后优先探索好路径，剪掉差路径。
```

例如解题时：

1. 先列出几个可能解法。
2. 对每个解法走几步。
3. 发现某条路矛盾，就停止。
4. 发现某条路有希望，就继续。
5. 最后选择最好的完整解。

这比盲目生成 N 条完整 CoT 更有结构。

---

### 二、把推理建模成搜索问题

搜索问题通常有几个要素。

#### 1. State

当前状态。

在 reasoning 中，state 可以是当前已经生成的部分推理。

例如：

```text
已知条件：...
Step 1: ...
Step 2: ...
```

#### 2. Action

下一步动作。

在 LLM reasoning 中，action 可以是生成下一步思路、调用工具、选择公式、执行代码。

#### 3. Transition

从一个状态到下一个状态。

例如模型生成下一步推理后，状态更新。

#### 4. Value

当前状态有多好。

可以由 verifier、PRM、规则、工具执行结果或模型自评给出。

#### 5. Policy

选择下一步动作的策略。

通常由 LLM 生成候选动作。

#### 6. Terminal state

推理结束状态。

例如得到最终答案、通过测试、满足目标或达到最大步数。

用公式化语言说：

```text
reasoning = 在状态空间中搜索一条从问题到答案的高质量路径
```

---

### 三、Chain-of-Thought 和 Tree-of-Thought

CoT 是一条链。

结构是：

```text
Step 1 -> Step 2 -> Step 3 -> Answer
```

Tree-of-Thought，简称 ToT，是一棵树。

结构是：

```text
                 Problem
              /     |     \
          Thought A B C
          /   \       \
       A1    A2       C1
       |              |
     Answer        Answer
```

CoT 一次只保留一条路径。

ToT 同时维护多个候选思路。

ToT 的核心思想是：

```text
把中间 thought 当成搜索节点，让模型生成多个候选 thought，再评价和选择哪些 thought 继续展开。
```

这让模型可以：

1. 同时探索多个方向。
2. 中途比较路径质量。
3. 剪掉明显错误的分支。
4. 对困难问题做更系统的推理。

---

### 四、Tree-of-Thought 基本流程

一个简化 ToT 流程是：

```text
Input question
Initialize root state

for depth in 1..D:
    Expand: 对每个当前状态生成 K 个候选 thought
    Evaluate: 给每个新状态打分
    Select: 保留最好的 B 个状态

Return best final answer
```

其中：

1. `D` 是最大深度。
2. `K` 是每个节点展开多少个候选。
3. `B` 是每层保留多少个状态。

这个流程很像 beam search。

例如在数学题中：

1. 第 1 层生成不同解法方向。
2. 第 2 层对每个方向继续推导。
3. 第 3 层检查是否得到答案。
4. Verifier 给每条路径打分。
5. 选择最好答案。

---

### 五、BFS、DFS 和 Beam Search

搜索策略决定如何探索树。

#### 1. BFS

BFS 是广度优先。

它按层展开。

优点是覆盖广。

缺点是节点数量增长快。

在 ToT 中，BFS 适合先探索多个不同思路。

#### 2. DFS

DFS 是深度优先。

它沿着一条路径走到底，再回退。

优点是内存少。

缺点是容易在坏路径上走太深。

在 reasoning 中，DFS 类似“先尝试一个完整解法”。

#### 3. Beam Search

Beam search 每层只保留 top-B 个候选。

它是质量和成本的折中。

在 LLM reasoning 中很常用：

```text
每一步生成多个候选 -> 用 verifier 打分 -> 保留前 B 个继续
```

缺点是如果早期打分不准，正确路径可能被剪掉。

---

### 六、MCTS 的直觉

MCTS 是 Monte Carlo Tree Search，蒙特卡洛树搜索。

它在围棋、游戏 AI 和规划中很常见。

MCTS 的核心是平衡：

```text
探索还不确定但可能好的分支，利用已经看起来不错的分支。
```

经典 MCTS 包含四步：

#### 1. Selection

从根节点出发，选择一个值得继续探索的节点。

#### 2. Expansion

展开这个节点，生成新的候选动作或 thought。

#### 3. Simulation / Rollout

从新节点继续模拟到终局，得到一个结果。

#### 4. Backpropagation

把结果分数回传，更新路径上节点的价值估计。

在 LLM reasoning 中，可以理解为：

1. 选择一个有希望的 partial reasoning。
2. 让模型生成下一步。
3. 继续生成到答案或用 verifier 估计成功率。
4. 把答案质量回传给前面的 thought。

---

### 七、MCTS 中的 UCB 直觉

MCTS 常用 UCB 类公式选择节点。

不用背公式，但要理解 trade-off。

一个节点被选择，取决于：

1. 它过去表现好不好。
2. 它探索次数够不够。

如果只利用高分节点，可能错过潜在好路径。

如果只探索新节点，成本会很高。

所以 MCTS 做的是：

```text
在 exploitation 和 exploration 之间平衡。
```

在 reasoning 中，这对应：

1. 继续深入当前看起来最好的思路。
2. 也给其他可能思路一些探索机会。

这比单纯 beam search 更灵活。

---

### 八、LLM Search 中的 Policy 和 Value

搜索需要两个能力。

#### 1. Policy：生成候选

LLM 可以作为 policy。

给定当前状态，让模型生成下一步 thought。

例如：

```text
基于当前推理，请给出 3 个可能的下一步。
```

#### 2. Value：评价候选

Verifier、PRM、ORM、规则、工具执行结果可以作为 value。

例如：

```text
这个 partial solution 最终成功概率是多少？
```

好的 reasoning search 不是只靠生成。

它需要：

```text
强 policy 生成好候选 + 强 value 识别好候选
```

如果 policy 差，根本生成不到正确路径。

如果 value 差，正确路径会被剪掉。

---

### 九、Search 和 Verifier 的关系

第 88 讲说过，verifier 是评价函数。

Search 离不开评价函数。

Verifier 可以用于：

1. 给完整答案打分。
2. 给中间状态打分。
3. 剪掉错误路径。
4. 选择最优候选。
5. 指导下一步搜索。

如果有 PRM，可以每一步评价。

如果只有 ORM，只能在生成完整答案后评价。

所以 PRM 更适合搜索。

但 PRM 训练更难。

这就是 trade-off。

---

### 十、Search 的适用场景

Search 适合：

1. 数学证明。
2. 复杂逻辑题。
3. 代码生成和修复。
4. 规划任务。
5. 多步工具调用。
6. 策略游戏。
7. 需要显式试错的问题。

Search 不适合所有任务。

不适合：

1. 简单问答。
2. 低延迟聊天。
3. 答案主观、无法评价的问题。
4. 搜索空间极大但没有好 verifier 的问题。

没有好的评价函数时，search 可能只是更贵的随机游走。

---

### 十一、代码推理中的搜索

代码任务很适合 search，因为可以执行测试。

流程可以是：

```text
生成多个候选修复
-> 运行单元测试
-> 根据失败信息继续修改
-> 保留通过更多测试的候选
-> 重复直到通过或达到预算
```

这里的 verifier 是测试。

测试结果提供很强反馈。

例如：

1. 编译是否通过。
2. 单元测试通过多少。
3. 哪个边界用例失败。
4. 性能是否超时。

这也是为什么代码 agent 常常比纯自然语言推理更容易做闭环。

因为代码可以运行验证。

---

### 十二、数学推理中的搜索

数学推理也适合 search，但 verifier 更难。

简单算术可以用程序验证。

但复杂证明、几何题、开放式推导很难自动验证。

常见方案包括：

1. 用符号计算器检查代数步骤。
2. 用 PRM 判断推理步骤。
3. 让多个模型互相检查。
4. 对最终答案做代入验证。
5. 使用形式化证明系统。

数学 search 的关键是：

```text
搜索空间大，必须有足够可靠的中间评价。
```

否则会生成大量看起来像证明但实际错误的路径。

---

### 十三、成本和预算控制

Search 比 self-consistency 更容易成本失控。

因为节点数可能指数增长。

如果每个节点展开 K 个候选，深度 D：

```text
节点数约为 K^D
```

所以必须设置预算：

1. 最大深度。
2. 最大节点数。
3. 最大 token。
4. 最大工具调用次数。
5. 最大运行时间。
6. 提前停止条件。

Beam search 用 beam size 控制宽度。

MCTS 用 simulation budget 控制探索次数。

生产系统必须把搜索预算作为一等公民。

---

### 十四、失败模式

#### 1. Search space explosion

分支太多，成本爆炸。

#### 2. Bad value function

Verifier 评分不准，剪掉正确路径。

#### 3. Goodhart / reward hacking

模型生成迎合 verifier 的路径，而不是真正解决问题。

#### 4. Lack of diversity

展开的候选都很相似，搜索没有意义。

#### 5. Premature pruning

早期看起来分数低但后面会变好的路径被剪掉。

#### 6. Looping

搜索反复生成相同或等价状态。

#### 7. No reliable terminal check

不知道什么时候已经得到正确答案。

---

### 十五、评估方法

Search-based reasoning 要评估：

1. 最终准确率。
2. Solve rate。
3. 平均节点数。
4. 平均 token。
5. 平均工具调用次数。
6. 平均延迟。
7. 每题成本。
8. Search depth。
9. Verifier 剪枝准确率。
10. 不同预算下的收益曲线。

要比较：

```text
single CoT
self-consistency
beam search
MCTS
search + verifier
```

重点不是 search 一定更强。

重点是看：

```text
多花的 test-time compute 是否换来了足够收益。
```

---

### 十六、真实项目中的坑

#### 1. 没有 verifier 就上 search

没有可靠评价函数，search 只是放大随机性。

#### 2. 展开粒度太细

每个 token 都作为搜索动作，成本太高。

通常按 thought、step、function call 粒度更合理。

#### 3. 展开粒度太粗

一步生成完整答案，又退回 self-consistency。

#### 4. 不去重

大量等价 thought 重复展开。

#### 5. 不设预算

线上延迟和费用不可控。

#### 6. 只报告最优结果

不报告搜索成本，容易夸大方法价值。

#### 7. Verifier 和 generator 互相过拟合

长期优化同一个 verifier，模型可能学会骗分。

---

### 十七、面试问答

#### 问题 1：Tree-of-Thought 和 Chain-of-Thought 有什么区别？

可以这样回答：

```text
Chain-of-Thought 是一条线性推理链，Tree-of-Thought 把中间 thought 组织成树，允许模型同时探索多个推理分支，并通过 verifier 或 value function 选择哪些分支继续展开。
```

#### 问题 2：为什么 reasoning 需要 search？

可以这样回答：

```text
复杂问题可能有多条解法，单条 CoT 容易早期走错。Search 可以在中间阶段探索多个方向、评价 partial solution、剪枝差路径，并把 test-time compute 用在更有希望的路径上。
```

#### 问题 3：MCTS 的核心思想是什么？

可以这样回答：

```text
MCTS 通过 selection、expansion、simulation 和 backpropagation 在搜索树上迭代探索，核心是平衡 exploitation 和 exploration，也就是既深入好路径，也探索不确定但可能有潜力的路径。
```

#### 问题 4：Verifier 在 search 中起什么作用？

可以这样回答：

```text
Verifier 或 PRM 可以作为 value function，给中间状态或完整答案打分，帮助选择、剪枝和排序候选路径。没有可靠 verifier，search 很容易变成高成本随机探索。
```

#### 问题 5：Search-based reasoning 的主要代价是什么？

可以这样回答：

```text
主要代价是节点数、token、工具调用、延迟和费用增加，而且如果评价函数不准，可能剪掉正确路径或强化错误路径。因此必须设置预算并画成本收益曲线。
```

#### 问题 6：如何设计一个代码修复 search 系统？

可以这样回答：

```text
让 LLM 生成多个候选修复，运行测试作为 verifier，根据失败信息继续展开或修改候选，用通过测试数和错误类型评分，保留 top candidates，设置最大轮数、时间和工具调用预算，最后输出通过测试且改动最小的修复。
```

---

### 十八、常见误区

1. 误区：Search 一定比 CoT 强。
   纠正：没有好的评价函数和预算控制，search 可能只是更贵。

2. 误区：Tree-of-Thought 就是多生成几条 CoT。
   纠正：ToT 的重点是中间 thought 节点的展开、评价和选择。

3. 误区：Beam search 总能保留正确路径。
   纠正：如果早期评分不准，正确路径可能被剪掉。

4. 误区：MCTS 只适合游戏。
   纠正：只要能定义状态、动作和价值估计，就可以借鉴 MCTS 思想。

5. 误区：搜索深度越深越好。
   纠正：深度越大成本越高，错误也可能累积。

6. 误区：搜索结果只看准确率。
   纠正：必须同时看 token、延迟、节点数、工具调用和成本。

---

### 十九、小练习

1. 把一个数学题的求解过程建模成 state、action、value、terminal state。
2. 画出一个三层 Tree-of-Thought 示例。
3. 比较 CoT、self-consistency、beam search 和 MCTS。
4. 设计一个 beam search reasoning 流程，包含 beam size 和评分函数。
5. 设计一个 MCTS reasoning 流程，包含 selection、expansion、rollout 和 backpropagation。
6. 为代码修复任务设计 search + test verifier 系统。
7. 构造一个 verifier 错误剪枝导致失败的案例。
8. 设计一个搜索预算，包括最大深度、节点数、token 和时间。
9. 画一条不同 search budget 下准确率和成本的曲线。
10. 用 3 分钟回答：“Tree-of-Thought 和 MCTS 如何增强 LLM reasoning？”

### 本讲总结

本讲最重要的结论：

1. Search 把 reasoning 看成在状态空间中寻找高质量推理路径。
2. CoT 是一条链，Tree-of-Thought 是多分支推理树。
3. Search 需要 state、action、transition、value、policy 和 terminal state。
4. Beam search 保留每层 top candidates，MCTS 在探索和利用之间平衡。
5. LLM 可以作为 policy 生成候选 thought，verifier/PRM 可以作为 value 评价候选。
6. Search 特别适合数学、代码、规划等可验证或可分解任务。
7. Search 的主要风险是成本爆炸、评价函数错误、过早剪枝和 reward hacking。
8. 面试中要强调：search 是 test-time compute scaling 的结构化形式，关键在评价函数和预算控制。

## 第 90 讲：Test-Time Compute Scaling

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Test-time compute scaling 是什么，和训练时 scaling 有什么区别。
2. 为什么更多推理时计算可能提升 reasoning 能力。
3. CoT、self-consistency、verifier、search 如何共同构成 test-time compute scaling。
4. 哪些任务适合花更多推理时计算，哪些任务不适合。
5. 如何评估 test-time compute 的成本收益曲线。
6. 面试中如何设计一个按任务难度动态分配推理预算的系统。

前面几讲其实都在讲 test-time compute。

第 86 讲 CoT：让模型生成更多中间 token。

第 87 讲 self-consistency：采样多条推理路径。

第 88 讲 verifier：对候选答案和过程打分。

第 89 讲 search：把推理组织成树搜索。

这些方法共同指向一个趋势：

```text
不只靠更大的模型和更多训练数据，也可以在推理时投入更多计算来换取更强表现。
```

这就是 test-time compute scaling。

---

### 一、什么是 Test-Time Compute Scaling

Test-time compute scaling 指的是：

```text
在模型参数固定的情况下，推理阶段投入更多计算、更多 token、更多采样、更多验证或更多搜索，以提升最终答案质量。
```

它和训练时 scaling 不同。

训练时 scaling 关注：

1. 更多参数。
2. 更多训练 token。
3. 更多训练 FLOPs。
4. 更长训练时间。

Test-time scaling 关注：

1. 推理时生成多少 token。
2. 采样多少条路径。
3. 是否调用 verifier。
4. 是否做 search。
5. 是否调用工具。
6. 是否反思、修正、重试。

一句话：

```text
训练时 scaling 是把能力压进模型参数里；test-time scaling 是在使用模型时花更多计算把能力释放出来。
```

---

### 二、为什么推理时计算有用

推理时计算有用，主要有几个原因。

#### 1. 给模型更多中间状态

CoT 让模型写中间步骤。

中间步骤成为后续 token 的上下文。

#### 2. 探索多个候选

Self-consistency 和 search 不只走一条路径。

多个候选可以降低偶然错误。

#### 3. 引入选择机制

Verifier 可以从多个候选中挑更好的。

#### 4. 允许试错和回退

Search 让模型发现某条路径不好后换路径。

#### 5. 使用外部反馈

代码执行、工具调用、检索结果、单元测试都能提供额外信号。

#### 6. 任务难度不均匀

简单问题不需要很多计算，难题可能需要更多计算。

动态分配计算可以更高效。

---

### 三、几种常见形式

Test-time compute scaling 有多种形式。

#### 1. Longer reasoning

生成更长的 reasoning trace。

例如 CoT、scratchpad、step-by-step solving。

#### 2. Multiple samples

采样多条答案或推理路径。

例如 self-consistency。

#### 3. Reranking

生成多个候选，再用 verifier 或 reward model 排序。

#### 4. Search

把推理过程组织成树或图。

例如 Tree-of-Thought、beam search、MCTS。

#### 5. Tool feedback

调用工具验证中间结果。

例如代码执行、计算器、检索、数据库查询。

#### 6. Iterative refinement

生成答案后检查、修正、再生成。

例如 draft -> critique -> revise。

#### 7. Debate or multi-agent

多个模型或多个角色互相挑战、验证、改进答案。

这些方法本质上都在增加推理时计算。

区别是计算花在不同位置。

---

### 四、Scaling 的对象是什么

Test-time compute 可以扩展多个维度。

#### 1. Token budget

允许模型生成更多中间推理 token。

#### 2. Sample budget

允许生成更多候选。

#### 3. Search budget

允许展开更多节点、更深层级。

#### 4. Verification budget

允许 verifier 检查更多候选或更多步骤。

#### 5. Tool budget

允许更多工具调用。

#### 6. Time budget

允许请求等待更久。

#### 7. Money budget

允许单个问题花更多推理成本。

不同任务的瓶颈不同。

数学题可能需要 sample/search budget。

代码题可能需要 tool/test budget。

RAG 问答可能需要 retrieval/rerank/verifier budget。

---

### 五、为什么不是越多越好

Test-time compute 有收益递减。

例如：

```text
1x compute -> 70% accuracy
2x compute -> 76% accuracy
5x compute -> 82% accuracy
20x compute -> 85% accuracy
```

越往后，每增加一单位计算带来的提升越少。

而成本、延迟和资源占用持续增加。

此外，更多计算也可能带来负面效果：

1. 更长 CoT 引入错误。
2. 更多采样产生更多噪声。
3. Search 被错误 verifier 误导。
4. 工具调用越多，失败点越多。
5. 用户等待时间变长。
6. 系统吞吐下降。

所以重点不是“尽量多算”。

重点是：

```text
在合适任务上，以合适预算，获得足够收益。
```

---

### 六、动态预算分配

生产系统不应该所有请求都用同样预算。

更合理的是动态分配。

例如按任务难度分层。

#### 1. Easy

简单事实问答、格式转换、普通分类。

策略：直接回答。

#### 2. Medium

需要两三步推理。

策略：单条 CoT 或短 scratchpad。

#### 3. Hard

数学、代码、多约束规划。

策略：多采样、verifier、search。

#### 4. High-risk

医疗、法律、金融、生产操作。

策略：工具验证、引用、人工确认、保守输出。

一个系统可以先做 difficulty estimation。

然后决定：

```text
直接回答
短 CoT
CoT + self-consistency
CoT + verifier
search + verifier
human review
```

---

### 七、难度估计怎么做

动态预算需要判断问题难不难。

难度估计可以来自：

1. 问题长度。
2. 是否包含数学、代码、逻辑、多约束。
3. 模型初次回答的置信度。
4. 多个候选答案是否一致。
5. Verifier 分数。
6. 检索证据是否充分。
7. 用户场景风险等级。
8. 历史错误率。

例如：

```text
如果单次回答置信度低，或 verifier 分数低，就自动增加采样或 search。
```

也可以采用 cascade：

```text
低成本解法先跑
-> 如果不确定，再升级到高成本解法
-> 如果仍不确定，拒答或人工审核
```

这比一开始就使用最高预算更高效。

---

### 八、Compute Allocation 策略

常见计算分配策略包括：

#### 1. Fixed budget

每个请求固定 N 条采样或固定最大 token。

简单但浪费。

#### 2. Adaptive budget

根据难度动态调整。

更适合生产。

#### 3. Early stopping

如果多个候选已经高度一致，提前停止。

例如 5 条里 4 条答案一致，就不再采样到 20 条。

#### 4. Escalation

先用便宜方法，不够再升级。

例如 direct answer -> CoT -> self-consistency -> search。

#### 5. Budget-aware search

Search 中根据剩余预算决定是否继续展开。

#### 6. Risk-aware budget

高风险任务增加验证和确认预算。

例如金融建议必须引用来源和人工审核。

---

### 九、成本收益曲线

评估 test-time scaling 必须画成本收益曲线。

横轴可以是：

1. token 数。
2. 采样条数。
3. search 节点数。
4. 工具调用次数。
5. 延迟。
6. 每题成本。

纵轴可以是：

1. accuracy。
2. solve rate。
3. pass@k。
4. human preference。
5. safety pass rate。
6. business success rate。

一个好的实验不是只报告最高分。

而是回答：

```text
多花 2 倍、5 倍、10 倍计算分别带来多少收益？
```

面试中如果能主动讲成本收益曲线，会显得很工程化。

---

### 十、和训练时 Scaling 的关系

训练时 scaling 和 test-time scaling 是互补关系。

训练更强的模型，可以让单次推理更准。

推理时投入更多计算，可以让固定模型在难题上表现更好。

两者之间有 trade-off：

```text
更大模型单次推理 vs 较小模型多次推理
```

例如：

1. 一个大模型单次回答。
2. 一个小模型采样 20 次加 verifier。

哪个更好，要看准确率、延迟、成本和部署约束。

未来系统可能不是单一模型，而是：

```text
模型大小选择 + 推理预算选择 + 工具验证 + 搜索策略
```

共同决定最终质量。

---

### 十一、Reasoning Model 的特殊性

Reasoning model 往往更强调推理时计算。

它们可能具备：

1. 更强的长推理能力。
2. 更好的中间步骤质量。
3. 更稳定的自我检查。
4. 更适合 verifier 或 PRM。
5. 更能从额外 token 中获益。

普通 chat model 可能生成长 CoT 后跑偏。

Reasoning model 的目标之一是：

```text
让额外推理 token 真正转化为更高质量，而不是只是更长文本。
```

因此评估 reasoning model 时，不仅看单次答案，还要看：

```text
随着 test-time compute 增加，性能是否持续提升。
```

---

### 十二、适用场景

适合 test-time compute scaling 的任务：

1. 数学推理。
2. 代码生成和修复。
3. 复杂规划。
4. 多步工具任务。
5. 科学推理。
6. 高价值决策辅助。
7. 需要严格验证的任务。

不适合或收益较低的任务：

1. 简单事实问答。
2. 翻译。
3. 简单摘要。
4. 低风险闲聊。
5. 高实时性场景。
6. 答案主观且难验证的任务。

判断标准是：

```text
任务是否有多步推理空间，是否能验证，是否值得等待和付费。
```

---

### 十三、系统设计示例

假设设计一个数学推理服务。

可以分层：

```text
Level 0: direct answer
Level 1: single CoT
Level 2: self-consistency N=5
Level 3: self-consistency N=20 + verifier
Level 4: search + PRM + symbolic checker
```

流程：

1. 先判断题目难度。
2. 简单题用 Level 0 或 1。
3. 中等题用 Level 2。
4. 难题用 Level 3。
5. 高价值难题用 Level 4。
6. 如果 verifier 不通过，拒答或请求人工。

系统记录：

1. 使用了哪个 level。
2. 消耗 token。
3. 采样次数。
4. verifier 分数。
5. 最终是否正确。

这些日志可以反过来优化预算策略。

---

### 十四、真实项目中的坑

#### 1. 只追求最高准确率

不看成本和延迟，方法无法上线。

#### 2. 所有请求都用高预算

简单问题浪费大量计算。

#### 3. 难度估计不准

简单题被高预算处理，难题却低预算回答。

#### 4. Verifier 不可靠

更多 compute 被错误评价函数引导，结果更差。

#### 5. 没有 early stopping

答案已经一致还继续采样，浪费成本。

#### 6. 忽略用户体验

用户不一定愿意等 30 秒得到稍微更好的答案。

#### 7. 不做分场景策略

同一套预算策略用于闲聊、数学、代码和高风险业务，效果会很差。

---

### 十五、评估指标

评估 test-time compute scaling 要看：

1. Accuracy / solve rate。
2. Pass@k。
3. Cost per query。
4. Cost per solved task。
5. Latency p50/p95/p99。
6. Token usage。
7. Tool call count。
8. Verifier pass rate。
9. User satisfaction。
10. Safety violation rate。

还要按任务难度分桶：

```text
easy / medium / hard / high-risk
```

因为平均指标会掩盖问题。

Test-time scaling 的收益通常主要来自 hard subset。

如果只看全量平均，可能低估或高估它的价值。

---

### 十六、面试问答

#### 问题 1：Test-time compute scaling 是什么？

可以这样回答：

```text
它是在模型参数固定的情况下，在推理阶段投入更多计算，例如更长推理、多采样、verifier、search、工具验证和迭代修正，以提升最终答案质量。
```

#### 问题 2：它和训练时 scaling 有什么区别？

可以这样回答：

```text
训练时 scaling 是用更多参数、数据和训练 FLOPs 把能力压进模型；test-time scaling 是在使用模型时增加 token、采样、搜索和验证，把固定模型的能力更充分释放出来。
```

#### 问题 3：为什么 test-time compute 对 reasoning 有用？

可以这样回答：

```text
Reasoning 问题通常需要多步探索和检查。更多推理时计算可以提供中间状态、探索多条路径、使用 verifier 选择更好答案，并允许试错和回退。
```

#### 问题 4：如何动态分配推理预算？

可以这样回答：

```text
先估计任务难度和风险，简单任务直接回答，中等任务用短 CoT，难题用 self-consistency 或 verifier，高价值高风险任务用 search、工具验证或人工审核。预算应根据置信度和 verifier 分数动态升级。
```

#### 问题 5：如何评估 test-time scaling 是否值得？

可以这样回答：

```text
要画成本收益曲线，比较不同 token、采样数、search 节点和工具调用预算下的准确率、solve rate、延迟、成本和安全指标，而不是只报告最高准确率。
```

#### 问题 6：Test-time scaling 的主要风险是什么？

可以这样回答：

```text
主要风险是成本和延迟上升、收益递减、难度估计不准、verifier 错误引导、搜索空间爆炸、用户体验下降，以及在不适合的任务上浪费计算。
```

---

### 十七、常见误区

1. 误区：推理时算得越多越好。
   纠正：收益递减，而且成本和延迟会上升。

2. 误区：所有任务都应该用 reasoning 模式。
   纠正：简单任务直接回答更好。

3. 误区：多采样就是 test-time scaling 的全部。
   纠正：还包括长推理、verifier、search、工具反馈和迭代修正。

4. 误区：只看 accuracy。
   纠正：还要看 cost、latency、token、工具调用和安全。

5. 误区：大模型不需要 test-time compute。
   纠正：大模型也可能通过额外推理计算在难题上继续提升。

6. 误区：小模型多采样一定能超过大模型。
   纠正：取决于任务、模型能力、verifier 和成本。

---

### 十八、小练习

1. 比较训练时 scaling 和 test-time scaling。
2. 设计一个 direct answer、CoT、self-consistency、search 的四级推理预算系统。
3. 为数学题设计 N=1、5、20 的成本收益实验。
4. 设计一个 early stopping 规则，当答案一致时停止采样。
5. 构造一个更多 CoT token 反而导致错误的例子。
6. 设计一个难度估计器，判断请求应该用低预算还是高预算。
7. 设计一个代码修复任务的 test-time compute 分配策略。
8. 画出 cost per solved task 随采样数变化的曲线。
9. 分析高风险医疗问答为什么不能只靠多采样。
10. 用 3 分钟回答：“如何在生产系统中动态分配 reasoning compute？”

### 本讲总结

本讲最重要的结论：

1. Test-time compute scaling 是在推理阶段投入更多计算来提升答案质量。
2. 它包括长推理、多采样、reranking、verifier、search、工具反馈和迭代修正。
3. 它和训练时 scaling 互补，一个提升模型内化能力，一个提升使用时求解能力。
4. Reasoning 任务更容易从 test-time compute 中受益，因为它们需要探索、检查和纠错。
5. Test-time compute 不是越多越好，存在收益递减、成本、延迟和用户体验问题。
6. 生产系统应按任务难度、风险、置信度和 verifier 结果动态分配预算。
7. 评估必须看成本收益曲线，而不是只看最高准确率。
8. 面试中要强调：test-time compute scaling 的核心是把额外计算花在值得花的问题上。

## 第 91 讲：数学推理模型训练

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 数学推理模型训练和普通指令微调有什么区别。
2. 数学数据为什么重要，常见数据类型有哪些。
3. CoT、过程监督、结果监督、合成数据在数学训练中分别起什么作用。
4. 如何训练和使用数学 verifier 或 reward model。
5. 数学推理训练中的数据污染、过拟合和评估陷阱有哪些。
6. 面试中如何设计一个提升 LLM 数学能力的训练方案。

数学推理是 reasoning model 最典型的训练场景之一。

原因很直接：

1. 数学题需要多步推理。
2. 很多数学题有明确答案。
3. 解题过程可以拆成步骤。
4. 部分结果可以自动验证。
5. 有大量 benchmark 可以评估。

因此，很多 reasoning 方法都会先在数学任务上验证。

但数学推理模型训练不是简单收集一堆题目和答案做 SFT。

真正难的是：

```text
让模型学会可靠地产生、检查和改进多步推理过程。
```

---

### 一、数学推理训练想解决什么

普通语言模型可能知道很多数学文本。

但它常见问题是：

1. 题意理解错。
2. 中间步骤跳步。
3. 算术错误。
4. 符号变换错误。
5. 最终答案格式错误。
6. 看起来会推理，但本质是在模仿题解模板。

数学推理训练的目标是：

```text
让模型在面对新题时，能分解问题、生成有效中间步骤、执行计算、检查约束，并得到正确答案。
```

这里不只是提升最终准确率。

还要提升：

1. 步骤正确性。
2. 推理鲁棒性。
3. 自我检查能力。
4. 与 verifier/search 的配合能力。
5. 对新题型的泛化能力。

---

### 二、数学数据类型

数学训练数据可以分成几类。

#### 1. Question-answer data

只有题目和最终答案。

例如：

```text
Q: 24 的 1/3 是多少？
A: 8
```

优点是容易收集。

缺点是缺少过程监督。

#### 2. Step-by-step solution data

包含完整解题过程。

例如：

```text
Q: 2x + 3 = 11，求 x。
Solution: 两边减 3，得到 2x = 8；两边除以 2，得到 x = 4。
A: 4
```

这是 CoT SFT 的核心数据。

#### 3. Process-labeled data

每一步都有正确/错误标签。

用于训练 PRM。

#### 4. Preference data

同一道题的两个解法，标注哪个更好。

用于训练 reward model 或做 DPO/RLHF 类优化。

#### 5. Synthetic data

由模型或程序生成的新题、新解法、新错误样本。

#### 6. Tool-verified data

用计算器、符号系统、单元测试或 proof checker 验证过的数据。

数学训练通常需要混合这些数据。

只有 QA 数据不够。

只有漂亮题解也不够。

还需要错误样本和验证信号。

---

### 三、CoT SFT

最基础的数学训练方法是 CoT supervised fine-tuning。

训练样本是：

```text
题目 -> 分步解法 -> 最终答案
```

模型学习：

1. 如何读题。
2. 如何分解步骤。
3. 如何写中间推导。
4. 如何给最终答案。

优点：

1. 实现简单。
2. 能显著提升模型按步骤解题的格式能力。
3. 有助于让模型输出可检查的推理过程。

缺点：

1. 依赖题解质量。
2. 容易学到模板。
3. 不保证步骤忠实。
4. 不能直接惩罚错误推理。
5. 对分布外题型泛化有限。

CoT SFT 是起点，不是终点。

---

### 四、结果监督 Outcome Supervision

结果监督只看最终答案是否正确。

例如模型生成 10 个解法。

只要最终答案对，就给正反馈。

结果监督可以用于：

1. 训练 ORM。
2. 选择候选答案。
3. 做 rejection sampling。
4. 做 RL 优化。

优点：

1. 标注便宜。
2. 很多数学题有标准答案。
3. 可以自动判分。

缺点：

1. 不知道哪一步错。
2. 可能奖励错误过程但答案碰巧正确。
3. 对证明题和开放题不够。
4. 难以指导模型改进中间步骤。

结果监督适合提高最终正确率。

但如果目标是训练稳定 reasoning，过程监督更重要。

---

### 五、过程监督 Process Supervision

过程监督检查每一步推理是否正确。

例如：

```text
Step 1: 2x + 3 = 11
Step 2: 2x = 8       correct
Step 3: x = 5        incorrect
```

过程监督可以用于训练 PRM。

PRM 训练好后，可以：

1. 给 CoT 每一步打分。
2. 搜索时选择更好的中间状态。
3. 发现错误步骤。
4. 过滤坏样本。
5. 指导模型重新生成。

过程监督的优点：

1. 更细粒度。
2. 更适合搜索。
3. 能提升可诊断性。
4. 对长推理更有帮助。

缺点：

1. 标注贵。
2. 步骤粒度难定义。
3. 对复杂证明很难判断。
4. 容易有标注不一致。

面试中可以这样总结：

```text
结果监督告诉模型答案对不对，过程监督告诉模型推理哪里对、哪里错。
```

---

### 六、Rejection Sampling

Rejection sampling 是数学训练中常见方法。

流程：

```text
对每道题采样多个解法
-> 用答案匹配、程序验证或 verifier 判断正确性
-> 保留正确或高分解法
-> 用这些解法做 SFT
```

它的好处是：

1. 可以自动扩大高质量 CoT 数据。
2. 利用模型自己生成多样解法。
3. 结合 verifier 过滤错误。

但也有风险：

1. Verifier 错会保留坏样本。
2. 只保留模型已经会的题，难题仍然缺数据。
3. 生成解法可能同质化。
4. 错误但答案碰巧正确的过程可能混入。

所以 rejection sampling 最好结合过程检查或人工抽查。

---

### 七、合成数学数据

合成数据是提升数学能力的重要手段。

来源包括：

1. 模型生成新题。
2. 模型生成多种解法。
3. 程序生成可验证题。
4. 从简单题变换出复杂题。
5. 生成错误解法作为负样本。
6. 从真实题抽取模板再替换变量。

合成数据的关键不是数量，而是质量和覆盖。

要关注：

1. 题目是否正确。
2. 答案是否可验证。
3. 难度是否分布合理。
4. 是否覆盖不同题型。
5. 是否和评估集污染。
6. 解法是否多样。

如果合成数据质量差，模型会学到错误推理。

如果合成数据太模板化，模型会过拟合模板。

---

### 八、数学 Verifier 训练

数学 verifier 可以是 ORM，也可以是 PRM。

#### 1. ORM for math

输入题目和最终答案，判断答案是否正确。

适合短答案题。

#### 2. PRM for math

输入题目和推理步骤，判断每一步是否合理。

适合长推理和搜索。

训练数据需要正负样本。

正样本来自正确题解。

负样本可以来自：

1. 模型错误解法。
2. 人工构造错误。
3. 随机替换中间数字。
4. 常见代数错误。
5. 单位或符号错误。

好的负样本应该是 hard negative。

也就是看起来像对，但实际错。

否则 verifier 只会识别低级错误。

---

### 九、训练流程示例

一个完整数学推理训练流程可以是：

```text
1. 收集数学题和标准答案
2. 收集或生成 step-by-step solutions
3. 用 CoT SFT 训练基础推理模型
4. 采样多个候选解法
5. 用答案匹配、程序验证和人工抽查过滤
6. 构造 ORM/PRM 数据
7. 训练 verifier 或 reward model
8. 用 verifier 做 rejection sampling 或 reranking
9. 可选：做 RL 或 DPO 类偏好优化
10. 在独立 benchmark 上评估
```

这个流程中最容易出问题的是数据质量和评估污染。

数学 benchmark 很容易被训练数据污染。

所以必须做去重和污染检测。

---

### 十、训练目标怎么选

不同阶段目标不同。

#### 1. SFT

目标是模仿高质量解题过程。

适合打基础。

#### 2. Rejection sampling SFT

目标是从模型生成中筛出好解法再训练。

适合扩大数据。

#### 3. ORM/PRM training

目标是学会评价答案或过程。

适合 test-time rerank/search。

#### 4. RL

目标是直接优化正确性或 reward。

但更难稳定，容易 reward hacking。

#### 5. DPO / preference optimization

目标是偏好更好的解法。

比 RL 简化，但仍依赖偏好数据质量。

不要把所有问题都归结为“上 RL”。

很多时候，数据清洗、SFT、verifier 和推理时搜索已经能带来主要提升。

---

### 十一、评估数学能力

数学推理评估要看多个层面。

#### 1. Final answer accuracy

最终答案是否正确。

#### 2. Step correctness

推理步骤是否正确。

#### 3. Robustness

换表达、换数字、换题型是否稳定。

#### 4. Generalization

是否能解训练中没见过的新题型。

#### 5. Difficulty breakdown

按难度分桶看表现。

#### 6. Contamination check

评估题是否出现在训练数据或合成数据中。

#### 7. Test-time compute curve

随着采样数、搜索预算增加，性能如何变化。

#### 8. Error analysis

错误来自读题、建模、计算、符号、最终格式还是 verifier。

只报告一个总 accuracy 不够。

要知道模型到底哪里不会。

---

### 十二、常见错误类型

数学模型常见错误包括：

1. 读题错误。
2. 漏条件。
3. 单位转换错误。
4. 算术错误。
5. 代数变形错误。
6. 变量定义混乱。
7. 逻辑跳步。
8. 最终答案格式错误。
9. 证明中使用未证明结论。
10. 过度依赖模板。

做 error analysis 时，要把错误分类型。

不同错误需要不同修复方法。

算术错误可以接计算器。

读题错误需要更好的数据和 prompt。

证明错误可能需要 PRM 或 formal verifier。

---

### 十三、数据污染和过拟合

数学 benchmark 污染很常见。

例如训练数据里包含评估题原题、改写题或题解。

污染会导致模型看起来数学能力很强，但其实是记忆。

防护方法：

1. 对训练题和评估题做 exact match 去重。
2. 做 fuzzy matching。
3. 对题干、答案、题解分别去重。
4. 检查合成数据是否从评估题改写。
5. 使用时间切分的新数据。
6. 做人工抽查。

过拟合也很常见。

模型可能学会某个 benchmark 的题型模板。

因此要用多个 benchmark 和自建 holdout 集评估。

---

### 十四、真实项目中的坑

#### 1. 只收集最终答案数据

模型学不到稳定步骤。

#### 2. CoT 数据质量差

漂亮但错误的题解会污染模型。

#### 3. 合成数据太模板化

模型只会解同类模板题。

#### 4. Verifier 太弱

错误解法被保留下来继续训练。

#### 5. 只看 benchmark 分数

不做污染检测和错误分析，容易误判能力。

#### 6. 忽略格式问题

模型会推理但 final answer 抽取失败。

#### 7. 直接上 RL

没有好的 reward 和数据基础，训练不稳定且容易 reward hacking。

---

### 十五、面试问答

#### 问题 1：数学推理模型训练和普通 SFT 有什么区别？

可以这样回答：

```text
普通 SFT 更关注指令遵循和回答格式，数学推理训练更关注多步推理、步骤正确性、最终答案验证、过程监督和 test-time search/verifier 配合。
```

#### 问题 2：为什么数学训练需要 CoT 数据？

可以这样回答：

```text
因为数学题通常需要多步推导。CoT 数据让模型学习如何分解问题、写中间步骤、执行计算并给出最终答案，比只有 question-answer 更能训练推理过程。
```

#### 问题 3：结果监督和过程监督有什么区别？

可以这样回答：

```text
结果监督只看最终答案是否正确，标注便宜但无法定位错误；过程监督检查每一步推理是否正确，标注更贵但能训练 PRM，帮助搜索和错误定位。
```

#### 问题 4：如何构造数学 verifier 数据？

可以这样回答：

```text
正样本来自正确题解，负样本来自模型错误解法、常见代数错误、随机替换中间步骤和 hard negatives。ORM 标注最终答案，PRM 标注每一步是否正确。
```

#### 问题 5：数学推理评估要注意什么？

可以这样回答：

```text
除了 final answer accuracy，还要看步骤正确性、难度分桶、鲁棒性、泛化、污染检测、test-time compute 曲线和错误类型分析。
```

#### 问题 6：如果让你提升一个模型的数学能力，你会怎么做？

可以这样回答：

```text
先做数据审计和 benchmark，收集高质量题目、答案和 CoT，做 CoT SFT；再采样候选解法，用答案验证和人工抽查过滤，训练 ORM/PRM；最后结合 self-consistency、verifier 或 search 做推理时增强，并持续做污染检测和错误分析。
```

---

### 十六、常见误区

1. 误区：数学能力就是背题库。
   纠正：真正能力要看新题型和变体上的泛化。

2. 误区：只有最终答案对就够。
   纠正：过程错误会影响泛化和可靠性。

3. 误区：CoT 越长越好。
   纠正：长但错误的推理会误导模型。

4. 误区：合成数据越多越好。
   纠正：低质量和模板化合成数据会造成过拟合。

5. 误区：Verifier 能自动解决所有错误。
   纠正：Verifier 也需要训练、校准和评估。

6. 误区：数学 benchmark 高分就说明 reasoning 强。
   纠正：还要排查污染、模板过拟合和 test-time compute 成本。

---

### 十七、小练习

1. 为一道代数题写 QA 数据、CoT 数据和 PRM 数据三种格式。
2. 构造一个最终答案正确但过程错误的数学样本。
3. 构造 5 个 hard negative，用于训练数学 verifier。
4. 设计一个 rejection sampling SFT 流程。
5. 设计一个数学合成数据生成和过滤流程。
6. 设计一个数学评估集污染检测流程。
7. 按读题、建模、计算、格式四类分析 20 个错误样本。
8. 比较 CoT SFT、ORM、PRM、RL 在数学训练中的作用。
9. 设计一个数学推理 test-time compute 曲线实验。
10. 用 3 分钟回答：“如何系统提升 LLM 的数学推理能力？”

### 本讲总结

本讲最重要的结论：

1. 数学推理训练的目标是提升多步推理、步骤正确性和最终答案可靠性。
2. 数学数据包括 QA、CoT、process-labeled、preference、synthetic 和 tool-verified data。
3. CoT SFT 是基础，但不足以保证过程正确和泛化。
4. 结果监督看最终答案，过程监督看每一步推理。
5. Rejection sampling 和合成数据可以扩展训练集，但必须依赖可靠过滤。
6. 数学 verifier/PRM 对搜索和推理时增强很重要，但本身也会出错。
7. 数学评估必须做污染检测、难度分桶、错误分析和 test-time compute 曲线。
8. 面试中要把数学训练讲成“数据 + 过程 + 验证 + 推理时增强 + 评估”的完整闭环。

## 第 92 讲：代码推理与执行反馈

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 代码推理为什么是 reasoning model 的重要场景。
2. 执行反馈和普通文本 verifier 有什么区别。
3. 代码生成、代码修复、debug、单元测试在训练和推理中如何形成闭环。
4. 如何用 pass@k、测试通过率、执行结果训练或筛选代码模型。
5. 代码推理中的安全、沙箱、数据污染和评估陷阱有哪些。
6. 面试中如何设计一个利用执行反馈提升代码能力的系统。

代码推理和数学推理很像。

它们都需要多步 reasoning。

但代码有一个非常重要的优势：

```text
代码可以运行。
```

这意味着模型生成的答案不只是靠人类或 LLM judge 判断。

很多情况下可以直接执行测试，得到明确反馈：

1. 编译是否通过。
2. 单元测试是否通过。
3. 哪个用例失败。
4. 报错信息是什么。
5. 性能是否超时。
6. 输出是否符合预期。

因此，代码任务天然适合 execution feedback。

这也是为什么代码能力和 agent 能力关系很紧密。

---

### 一、代码推理包含哪些任务

代码 reasoning 不只是“写代码”。

常见任务包括：

#### 1. Code generation

根据题目或需求生成代码。

例如 LeetCode、HumanEval、业务函数实现。

#### 2. Code completion

补全函数、类、测试或配置。

#### 3. Code repair

根据错误信息修复 bug。

#### 4. Debugging

理解错误原因，定位代码问题。

#### 5. Test generation

为已有代码生成单元测试。

#### 6. Code review

发现 bug、风险、性能问题和安全问题。

#### 7. Repository-level reasoning

跨文件理解项目结构、调用链、依赖和测试。

这些任务都需要模型理解代码语义，而不只是生成语法正确的文本。

---

### 二、为什么执行反馈重要

普通文本任务很难自动判断答案对错。

但代码可以运行。

例如模型生成一个函数：

```python
def add(a, b):
    return a - b
```

单元测试可以立刻发现错误：

```text
assert add(2, 3) == 5
实际输出：-1
```

执行反馈的价值是：

1. 客观。
2. 可重复。
3. 可自动化。
4. 能定位部分错误。
5. 能用于训练、筛选和搜索。

相比 LLM judge，执行反馈更硬。

但也不是完美的。

测试覆盖不足时，代码可能通过测试但仍有 bug。

---

### 三、代码推理闭环

一个典型执行反馈闭环是：

```text
生成代码
-> 运行测试
-> 读取错误信息
-> 分析失败原因
-> 修改代码
-> 再次运行测试
-> 直到通过或达到预算
```

这和人类写代码很像。

模型不是一次生成就结束。

而是在执行环境中迭代。

这个闭环可以用于推理时：

1. 多次尝试修复。
2. 根据测试反馈改进。
3. 用通过测试数排序候选。

也可以用于训练时：

1. 生成候选代码。
2. 执行测试筛选正确样本。
3. 用正确修复轨迹做 SFT。
4. 用测试通过率训练 reward model。

---

### 四、Pass@k

代码模型常用指标是 pass@k。

它衡量：

```text
模型生成 k 个候选中，至少有一个通过测试的概率。
```

例如 pass@1 是单次生成通过率。

pass@10 是生成 10 个候选时至少一个通过的概率。

Pass@k 很适合代码任务，因为：

1. 代码可以自动测试。
2. 多候选生成很常见。
3. 实际系统可以采样多个候选再筛选。

但 pass@k 也有局限：

1. 依赖测试质量。
2. 不反映代码可读性。
3. 不反映安全性。
4. 不反映性能。
5. 不反映修改范围是否合理。

所以生产系统不能只看 pass@k。

---

### 五、执行反馈作为 Verifier

在代码任务中，测试就是一种强 verifier。

Verifier 输入是候选代码。

输出是：

1. 是否编译。
2. 通过多少测试。
3. 哪些测试失败。
4. 错误栈是什么。
5. 是否超时。
6. 资源占用如何。

例如：

```json
{
  "compile": "success",
  "passed": 18,
  "total": 20,
  "failed_tests": ["test_empty_input", "test_large_n"],
  "error": "IndexError: list index out of range"
}
```

这个反馈比一句“答案不对”更有用。

模型可以根据失败用例修复代码。

---

### 六、训练代码模型的数据

代码模型训练数据可以分几类。

#### 1. Code corpus

大量开源代码。

用于预训练，让模型学习语法、API、模式和项目结构。

#### 2. Instruction-code data

需求描述到代码实现。

用于指令微调。

#### 3. Code explanation data

代码到解释，或解释到代码。

提升代码理解能力。

#### 4. Bug-fix data

错误代码、错误信息、修复补丁。

用于训练 debug 和 repair。

#### 5. Test data

函数、测试用例、期望输出。

用于训练模型理解测试和生成测试。

#### 6. Execution trace data

模型尝试、报错、修复、再测试的轨迹。

这类数据对 agentic coding 特别重要。

---

### 七、Rejection Sampling for Code

代码任务很适合 rejection sampling。

流程：

```text
给定题目
-> 采样 N 个代码候选
-> 运行测试
-> 保留通过测试的候选
-> 用通过候选做 SFT 或偏好数据
```

如果有多个通过候选，可以再按：

1. 简洁性。
2. 时间复杂度。
3. 内存复杂度。
4. 可读性。
5. 安全性。
6. 是否使用允许的库。

进一步排序。

风险是：

1. 测试覆盖不全。
2. 代码 hardcode 测试。
3. 生成不安全代码。
4. 候选通过测试但不可维护。

所以测试只是第一层过滤。

---

### 八、从失败中学习

代码执行反馈最有价值的地方是失败信息。

失败可以告诉模型：

1. 哪个输入出错。
2. 期望输出是什么。
3. 实际输出是什么。
4. 栈跟踪在哪里。
5. 是语法、类型、边界、性能还是逻辑问题。

训练数据可以包含：

```text
题目
错误代码
测试失败信息
修复分析
修复后代码
```

这比只有“正确代码”更接近真实开发。

模型学到的是 debug workflow。

例如：

```text
看到 IndexError -> 检查空列表和边界条件 -> 添加保护逻辑 -> 重新测试
```

---

### 九、代码搜索与执行反馈

代码任务可以把第 89 讲的 search 用起来。

一个简单 beam search repair：

```text
初始 bug 代码
-> 生成 K 个修复候选
-> 运行测试打分
-> 保留 top-B
-> 根据失败信息继续修改
-> 重复直到通过
```

Value function 可以是：

1. 测试通过率。
2. 是否编译通过。
3. 失败测试数量。
4. diff 大小。
5. 静态分析结果。
6. 安全扫描结果。

这比纯自然语言 search 更可靠。

因为执行环境提供硬反馈。

---

### 十、Sandbox 和安全

执行模型生成的代码有风险。

代码可能：

1. 删除文件。
2. 访问网络。
3. 读取环境变量和 secret。
4. 无限循环。
5. 消耗大量 CPU/内存。
6. 执行恶意命令。

所以必须使用 sandbox。

基本要求：

1. 文件系统隔离。
2. 禁止或限制网络。
3. CPU 和内存限制。
4. 执行时间限制。
5. 环境变量脱敏。
6. 最小权限运行。
7. 容器或虚拟机隔离。
8. 审计执行日志。

执行反馈系统如果没有 sandbox，不能上线。

---

### 十一、代码评估指标

代码能力评估不只看 pass@k。

常见指标包括：

1. pass@1。
2. pass@k。
3. compile rate。
4. test pass rate。
5. repair success rate。
6. average attempts to pass。
7. time to solve。
8. token cost。
9. runtime performance。
10. memory usage。
11. security violation rate。
12. diff minimality。

对于真实代码库任务，还要看：

1. 是否通过现有测试。
2. 是否新增测试。
3. 是否破坏其他模块。
4. 是否符合代码风格。
5. 是否解决根因而不是硬编码。

---

### 十二、数据污染问题

代码 benchmark 很容易污染。

例如 HumanEval、MBPP、LeetCode 题目和解法在网上广泛存在。

模型可能记住题解。

防护方法：

1. 训练数据和评估题 exact match 去重。
2. 函数签名、题干、测试用例去重。
3. 检查近似题目和改写题。
4. 使用时间切分的新题。
5. 自建私有评估集。
6. 使用真实 repo issue。

代码评估要特别注意：

```text
模型会不会只是见过这道题。
```

---

### 十三、真实项目中的坑

#### 1. 测试覆盖不足

代码通过测试但线上失败。

#### 2. 模型 hardcode 测试

模型写出只针对测试用例的代码。

#### 3. 执行环境不一致

本地通过，线上依赖版本不同失败。

#### 4. 没有 sandbox

执行模型代码造成安全风险。

#### 5. 只优化 pass@k

代码不可读、不安全、性能差。

#### 6. 错误反馈太长

完整日志塞回模型，噪声大且成本高。

#### 7. 修复范围失控

模型为通过测试大改无关代码。

---

### 十四、面试问答

#### 问题 1：为什么代码推理适合执行反馈？

可以这样回答：

```text
因为代码可以运行，编译结果、单元测试、错误栈、超时和输出差异都能提供客观反馈。这比只靠文本 judge 更硬，也更适合形成生成、执行、修复的闭环。
```

#### 问题 2：pass@k 是什么？

可以这样回答：

```text
pass@k 衡量模型生成 k 个候选中至少有一个通过测试的概率。它适合评估代码生成的多候选能力，但依赖测试质量，也不能完全反映可维护性和安全性。
```

#### 问题 3：如何用执行反馈训练代码模型？

可以这样回答：

```text
可以对每个题目采样多个候选，运行测试筛选通过样本做 rejection sampling SFT；也可以保存失败代码、错误信息、修复分析和最终补丁，训练模型根据执行反馈迭代修复。
```

#### 问题 4：代码执行反馈有哪些安全风险？

可以这样回答：

```text
模型生成代码可能访问文件、网络、secret，执行恶意命令或无限循环，因此必须用沙箱隔离、限制网络和资源、脱敏环境变量，并记录执行日志。
```

#### 问题 5：为什么通过测试不等于代码正确？

可以这样回答：

```text
测试覆盖可能不足，模型可能 hardcode 测试，代码可能有性能、安全、可读性或边界问题。因此通过测试只是必要条件，不是充分条件。
```

#### 问题 6：如何设计一个代码修复 Agent？

可以这样回答：

```text
输入 issue、代码和测试，模型生成候选补丁，在沙箱中运行测试，解析失败日志，继续生成修复；用测试通过率、diff 大小、安全扫描和代码风格评分排序候选，并设置最大轮数、时间和资源预算。
```

---

### 十五、常见误区

1. 误区：代码通过测试就一定正确。
   纠正：测试覆盖有限，还要看边界、性能、安全和可维护性。

2. 误区：执行反馈不需要 verifier。
   纠正：测试本身就是 verifier，但也需要静态分析、安全扫描和人工 review 补充。

3. 误区：pass@k 越高模型越适合生产。
   纠正：生产还要看延迟、成本、修复质量和安全。

4. 误区：可以直接执行模型生成代码。
   纠正：必须用 sandbox 和资源限制。

5. 误区：完整错误日志都塞给模型最好。
   纠正：应该提取关键错误、失败用例和栈信息，避免噪声。

6. 误区：代码能力只靠预训练代码语料。
   纠正：执行反馈、测试、修复轨迹和 repo-level 数据同样重要。

---

### 十六、小练习

1. 为一个简单函数设计题目、参考答案和 5 个单元测试。
2. 构造一个通过弱测试但实际错误的代码样本。
3. 设计一个代码 rejection sampling SFT 流程。
4. 设计一个执行反馈 JSON schema，包含编译、测试、错误和超时。
5. 为代码执行设计一个 sandbox 策略。
6. 设计一个代码修复 search 流程，包含候选生成、测试和 rerank。
7. 比较 pass@1、pass@k 和 repair success rate。
8. 分析 HumanEval 类 benchmark 的污染风险。
9. 设计一个错误日志压缩 prompt，只保留关键信息。
10. 用 3 分钟回答：“如何利用执行反馈提升代码模型能力？”

### 本讲总结

本讲最重要的结论：

1. 代码推理是 reasoning model 的重要场景，因为代码可以执行验证。
2. 执行反馈提供编译、测试、错误栈、超时和输出差异等硬信号。
3. 代码任务可以形成生成、执行、反馈、修复、再执行的闭环。
4. pass@k 衡量多候选中至少一个通过测试的概率，但不等于生产质量。
5. 执行反馈可用于 rejection sampling、reward model、search 和 agentic repair。
6. 运行模型生成代码必须使用 sandbox 和资源限制。
7. 代码评估要关注测试覆盖、污染、性能、安全、diff 范围和 repo-level 影响。
8. 面试中要把代码推理讲成“模型生成 + 执行环境 + 测试 verifier + 反馈修复”的系统。

## 第 93 讲：自我改进与合成推理数据

### 本讲目标

学完本讲，你应该能回答六个问题：

1. 为什么 reasoning model 需要合成推理数据。
2. Self-training、self-improvement、bootstrapping、distillation 分别是什么。
3. 如何生成、过滤、验证和迭代合成推理数据。
4. 合成数据为什么可能提升能力，也为什么可能导致模型退化。
5. 自我改进中的数据污染、模式坍塌、reward hacking 风险有哪些。
6. 面试中如何设计一个安全可控的合成推理数据 pipeline。

前面几讲分别讲了数学推理、代码执行反馈、verifier 和 test-time compute。

这些能力可以用来做一件很重要的事：

```text
让模型或更强系统生成新的推理数据，再用这些数据训练更强模型。
```

这就是自我改进和合成推理数据的核心。

大模型时代，数据不再只来自人工标注和互联网文本。

模型本身、工具、verifier、搜索系统都可以参与数据生产。

但这件事有双刃剑属性。

高质量合成数据能提升模型。

低质量合成数据会放大错误、污染分布、造成过拟合和能力退化。

---

### 一、为什么需要合成推理数据

推理数据很贵。

尤其是高质量 step-by-step reasoning 数据。

人工标注需要专家，成本高，速度慢。

而 reasoning model 需要大量数据覆盖：

1. 数学题。
2. 代码题。
3. 逻辑题。
4. 多跳问答。
5. 复杂规划。
6. 工具调用。
7. 错误修复。
8. 过程监督。

互联网文本中的推理过程质量参差不齐。

因此自然会想到：

```text
能不能用模型自己生成题目、解法、错误样本和验证数据？
```

合成推理数据的目标是：

1. 扩大数据规模。
2. 覆盖更多难度和题型。
3. 生成过程监督数据。
4. 生成 hard negatives。
5. 支持特定领域定制。
6. 支持持续迭代。

---

### 二、几个相关概念

#### 1. Synthetic data

由模型、程序或规则生成的数据。

例如模型生成数学题和解答。

#### 2. Self-training

用模型给未标注数据生成标签，再用高置信标签训练模型。

#### 3. Self-improvement

模型利用自己的生成、反馈、验证和修正，不断产生更好训练数据或更好策略。

#### 4. Bootstrapping

用已有模型或少量数据启动一个迭代流程，逐步扩大能力和数据。

#### 5. Distillation

用强模型或强推理系统生成数据，训练较小或更便宜的模型。

例如：

```text
强模型 + search + verifier -> 高质量解法 -> 训练小模型单次生成
```

这些概念有重叠。

面试中不必纠结名词边界。

重点是讲清数据生成、过滤、训练和评估闭环。

---

### 三、合成推理数据的类型

合成推理数据可以包括很多类型。

#### 1. 合成题目

生成新的数学题、逻辑题、代码题。

#### 2. 合成解法

为已有题目生成多种解法。

#### 3. 合成 CoT

生成 step-by-step reasoning。

#### 4. 合成错误样本

生成看起来合理但实际错误的解法。

用于训练 verifier 或做对比学习。

#### 5. 合成偏好对

同一道题生成好解法和坏解法，标注偏好。

#### 6. 合成工具轨迹

生成工具调用、执行反馈、修复过程。

#### 7. 合成评估样本

构造测试集或 red-team 样本。

不同类型数据用途不同。

训练 generator 需要好解法。

训练 verifier 需要正负样本。

训练 agent 需要轨迹数据。

评估需要独立且未污染的数据。

---

### 四、基本 Pipeline

一个合成推理数据 pipeline 通常包括：

```text
Seed tasks
-> Data generation
-> Verification / filtering
-> Deduplication
-> Difficulty and diversity balancing
-> Training
-> Evaluation
-> Error analysis
-> Next iteration
```

#### 1. Seed tasks

从少量真实题、领域任务或模板开始。

#### 2. Data generation

用模型、规则或程序生成题目和解法。

#### 3. Verification

用答案匹配、程序执行、verifier、人工抽查过滤。

#### 4. Deduplication

去掉重复题、近似题和评估集污染。

#### 5. Balancing

控制难度、题型、领域和语言分布。

#### 6. Training

用过滤后的数据做 SFT、DPO、reward model 或 RL。

#### 7. Evaluation

在独立 holdout 和真实任务上评估。

#### 8. Iteration

根据错误分析生成下一轮数据。

---

### 五、生成题目

生成题目时要控制几个维度。

1. 领域。
2. 难度。
3. 题型。
4. 所需推理步数。
5. 是否可自动验证。
6. 是否和已有数据重复。

例如数学题生成 prompt 可以要求：

```text
生成 20 道初中代数应用题，每题需要 3-5 步推理，有唯一整数答案，并给出标准答案和分步解法。
```

但模型生成的题目可能有问题：

1. 条件矛盾。
2. 没有唯一答案。
3. 答案算错。
4. 难度不符合要求。
5. 题目太模板化。

所以题目生成后必须验证。

---

### 六、生成解法

对已有题目，可以生成多种解法。

例如：

1. 代数解法。
2. 枚举解法。
3. 逆向验证。
4. 图形直觉。
5. 代码求解。

多解法有两个好处。

第一，可以增加 reasoning diversity。

第二，可以让模型不只记一种模板。

但多解法也可能引入错误。

尤其是模型为了多样性而编出不成立的方法。

因此每个解法都要检查：

1. 中间步骤是否成立。
2. 最终答案是否一致。
3. 是否使用题目没有给出的假设。
4. 是否过度跳步。

---

### 七、生成错误样本

错误样本对 verifier 很重要。

如果 verifier 只见过正确解法和明显错误，它很难识别高迷惑错误。

错误样本可以包括：

1. 算术错误。
2. 代数变形错误。
3. 漏条件。
4. 单位错误。
5. 变量混淆。
6. 逻辑跳步。
7. 代码边界条件错误。
8. 通过部分测试但失败隐藏测试。

Hard negative 的特点是：

```text
表面上很像正确推理，但有关键错误。
```

例如：

```text
2x + 3 = 11
2x = 14
x = 7
```

格式很像正确解法，但第二步错。

训练 verifier 时，hard negatives 比随机错误更有价值。

---

### 八、过滤与验证

合成数据最重要的是过滤。

常见过滤信号包括：

1. 答案匹配。
2. 程序执行。
3. 单元测试。
4. 符号计算。
5. Verifier 分数。
6. 多模型一致性。
7. Self-consistency。
8. 人工抽查。
9. 格式规则。
10. 去重和污染检测。

一个高质量 pipeline 通常不是单一过滤器。

而是多层过滤：

```text
格式过滤 -> 自动验证 -> verifier 打分 -> 去重 -> 人工抽查
```

过滤标准要根据用途调整。

训练 generator 的数据要尽量正确。

训练 verifier 的数据要包含高质量负样本。

评估数据必须尽量干净且独立。

---

### 九、Self-Improvement Loop

一个自我改进循环可以这样设计：

```text
当前模型 M_t
-> 生成候选题目和解法
-> verifier / tools 过滤
-> 得到高质量数据 D_t
-> 训练新模型 M_{t+1}
-> 在独立评估集测试
-> 分析错误
-> 生成下一轮针对性数据
```

关键是每一轮都要有独立评估。

否则模型可能只是越来越擅长自己生成的数据。

自我改进的风险是闭环污染：

```text
模型生成的数据带有模型自己的偏差，训练后偏差被放大，下一轮生成更偏。
```

所以需要外部锚点：

1. 人类数据。
2. 程序验证。
3. 强 verifier。
4. 独立 benchmark。
5. 真实用户任务。

---

### 十、Distillation from Strong Reasoner

一种常见做法是从强推理系统蒸馏。

强系统可以是：

```text
大模型 + CoT + self-consistency + verifier + search + tools
```

它生成高质量推理轨迹。

然后训练较小模型模仿。

目标是：

```text
把昂贵 test-time compute 的结果压缩到便宜模型里。
```

例如：

1. 强系统花 30 秒解题。
2. 生成高质量解法。
3. 小模型用这些解法 SFT。
4. 小模型未来单次生成就能接近强系统部分能力。

风险是：

1. 小模型只能模仿表面过程。
2. 强系统错误会被蒸馏。
3. 蒸馏数据分布太窄。
4. 小模型容量不足。

---

### 十一、数据多样性

合成数据容易模式化。

例如所有数学题都长得像：

```text
小明有 x 个苹果...
```

模型会学会模板，而不是泛化推理。

提高多样性的方法：

1. 控制题型分布。
2. 控制难度分布。
3. 多种生成 prompt。
4. 多模型生成。
5. 从真实错误中生成数据。
6. 引入不同领域和语言。
7. 聚类后采样，避免重复。
8. 针对薄弱点生成数据。

多样性不是越随机越好。

要在覆盖和质量之间平衡。

---

### 十二、数据污染风险

合成数据很容易污染评估集。

例如模型见过 benchmark 题，生成了改写版。

如果这些改写题进入训练，评估分数会虚高。

防护方法：

1. 生成前排除评估题。
2. 生成后做 exact/fuzzy dedup。
3. 对题干、答案、解法分别查重。
4. 用 embedding 检测近似题。
5. 保留生成来源和 lineage。
6. 使用时间更新的 holdout。
7. 对高分样本人工抽查。

Lineage 很重要。

要知道每条合成数据来自哪个 seed、哪个模型、哪个 prompt、哪个 verifier。

否则出了污染很难追踪。

---

### 十三、质量评估

合成数据质量要从多个维度评估。

1. Correctness。
2. Step validity。
3. Difficulty。
4. Diversity。
5. Novelty。
6. Verifiability。
7. Format consistency。
8. Contamination risk。
9. Downstream improvement。
10. Human audit pass rate。

最终最重要的是 downstream improvement。

也就是：

```text
加入这批合成数据后，模型在干净评估集和真实任务上是否变好。
```

如果数据看起来漂亮，但训练后没有提升，甚至退化，就要回头看分布和质量。

---

### 十四、真实项目中的坑

#### 1. 只看合成数据数量

数量大但错误多，会损害模型。

#### 2. 没有强过滤

模型生成的错误推理直接进训练集。

#### 3. 合成数据分布太窄

模型在模板题上提升，在真实题上不提升。

#### 4. 评估集污染

训练数据包含 benchmark 改写题，分数虚高。

#### 5. 自我循环放大偏差

模型生成自己的偏好数据，再训练自己，偏差越来越强。

#### 6. 忽略负样本

只生成正确解法，verifier 学不到区分高迷惑错误。

#### 7. 没有数据 lineage

无法追踪错误数据来自哪里。

---

### 十五、面试问答

#### 问题 1：为什么需要合成推理数据？

可以这样回答：

```text
高质量人工推理数据昂贵且覆盖有限。合成数据可以扩大题型、难度和过程监督覆盖，用于训练 generator、verifier、reward model 和 agent 轨迹，但必须经过严格验证和去重。
```

#### 问题 2：Self-improvement 的基本流程是什么？

可以这样回答：

```text
当前模型生成题目、解法或轨迹，再用工具、verifier、自一致性和人工抽查过滤，得到高质量数据后训练下一版模型，并在独立评估集上验证，再根据错误分析进入下一轮。
```

#### 问题 3：合成数据有什么风险？

可以这样回答：

```text
风险包括错误推理混入、数据分布单一、模式坍塌、评估集污染、模型偏差自我放大、reward hacking 和缺少 lineage 导致无法追踪问题来源。
```

#### 问题 4：如何过滤合成推理数据？

可以这样回答：

```text
可以用格式规则、答案匹配、程序执行、符号验证、单元测试、verifier 打分、多模型一致性、去重、污染检测和人工抽查组成多层过滤 pipeline。
```

#### 问题 5：Distillation from strong reasoner 是什么？

可以这样回答：

```text
用更强但更贵的推理系统，例如大模型加 search、verifier 和工具，生成高质量推理轨迹，再训练较小模型模仿，希望把昂贵 test-time compute 的能力部分压缩到模型参数中。
```

#### 问题 6：如何判断合成数据真的有用？

可以这样回答：

```text
不能只看数据量和表面质量，要看加入数据后模型在独立、未污染的评估集和真实任务上的提升，并按题型、难度和错误类型做 ablation 和 error analysis。
```

---

### 十六、常见误区

1. 误区：合成数据越多越好。
   纠正：质量、覆盖、验证和去重比数量更重要。

2. 误区：模型生成的数据可以直接训练。
   纠正：必须经过过滤、验证和污染检测。

3. 误区：自我改进可以完全不需要外部信号。
   纠正：需要工具、verifier、人类数据或独立评估作为锚点。

4. 误区：合成数据提升 benchmark 就说明有效。
   纠正：可能是污染或模板过拟合，要看干净 holdout 和真实任务。

5. 误区：只生成正确样本就够。
   纠正：训练 verifier 和鲁棒模型还需要高质量负样本。

6. 误区：强模型蒸馏一定能让小模型学会推理。
   纠正：小模型可能只学到表面格式，容量和数据分布都会限制效果。

---

### 十七、小练习

1. 设计一个数学合成数据 pipeline，包含生成、验证、去重和训练。
2. 为代码修复任务设计一个合成错误样本生成流程。
3. 构造 5 条 hard negative reasoning 数据。
4. 设计一个数据 lineage schema，记录 seed、prompt、model、verifier 和过滤结果。
5. 设计一个合成数据污染检测流程。
6. 设计一个 ablation，比较人工数据、合成数据和混合数据。
7. 分析 self-improvement 中偏差自我放大的原因。
8. 设计一个 distillation from strong reasoner 的训练方案。
9. 为合成数据质量设计 10 个评估指标。
10. 用 3 分钟回答：“如何安全地用合成推理数据提升 reasoning model？”

### 本讲总结

本讲最重要的结论：

1. 合成推理数据用于扩大题型、难度、过程监督和错误样本覆盖。
2. Self-improvement 的核心是生成、过滤、训练、评估和迭代闭环。
3. 合成数据可以包括题目、解法、CoT、错误样本、偏好对、工具轨迹和评估样本。
4. 过滤和验证是合成数据 pipeline 的核心，不能直接相信模型生成内容。
5. Distillation 可以把强推理系统的昂贵输出压缩到较小模型中，但会受数据质量和模型容量限制。
6. 合成数据的主要风险是错误放大、分布单一、污染、模式坍塌和 reward hacking。
7. 判断合成数据是否有用，要看独立评估集和真实任务的 downstream improvement。
8. 面试中要强调：合成数据不是免费午餐，而是一个需要强验证和强评估的数据工程系统。

## 第 94 讲：Reasoning Model 评估

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Reasoning model 评估和普通 chat model 评估有什么区别。
2. 数学、代码、逻辑、多跳问答、规划分别怎么评估。
3. 为什么只看 final answer accuracy 不够。
4. 如何评估过程正确性、test-time compute、鲁棒性和污染风险。
5. LLM-as-a-judge、verifier、工具评估、人类评估各有什么优缺点。
6. 面试中如何设计一个可信的 reasoning model 评估体系。

Reasoning model 的目标不是“回答更像人”。

它更强调：

```text
能否在复杂任务中稳定地分解、推理、验证和得到正确结论。
```

因此 reasoning model 评估不能只看通用聊天满意度。

它要回答：

1. 答案是否正确。
2. 推理过程是否可靠。
3. 错误是否可定位。
4. 增加 test-time compute 是否真的带来收益。
5. 模型是否只是记住 benchmark。
6. 在真实复杂任务中是否有用。

---

### 一、Reasoning 评估的特殊性

普通 chat model 评估常看：

1. 指令遵循。
2. 有用性。
3. 流畅度。
4. 安全性。
5. 用户偏好。

Reasoning model 还要看：

1. 多步推理正确性。
2. 过程一致性。
3. 约束满足。
4. 可验证结果。
5. 难题上的 solve rate。
6. 随推理预算增长的性能曲线。
7. 对变体题和分布外题的泛化。

一个答案写得很流畅，不代表 reasoning 好。

一个推理过程很长，也不代表 reasoning 好。

Reasoning 评估必须更硬、更细、更可复现。

---

### 二、Final Answer Accuracy

最基础指标是 final answer accuracy。

例如数学题最终答案是否正确。

代码题是否通过测试。

选择题是否选对。

优点：

1. 简单。
2. 可量化。
3. 易比较模型。
4. 适合有标准答案的任务。

缺点：

1. 不知道过程是否正确。
2. 答案可能碰巧对。
3. 错误难定位。
4. 不适合开放题。
5. 无法评价成本和推理预算。

所以 final answer accuracy 是必要指标，但不是充分指标。

---

### 三、过程正确性评估

Reasoning model 的核心是过程。

过程评估关注：

1. 每一步是否成立。
2. 是否有跳步。
3. 是否使用题目没有给出的假设。
4. 中间变量是否一致。
5. 最终答案是否由前面步骤推出。

方法包括：

1. 人工标注步骤。
2. PRM 打分。
3. LLM-as-a-judge 判断步骤。
4. 符号工具检查。
5. 单元测试或执行验证。
6. 对推理过程做 contradiction check。

过程评估很贵。

但它能发现 final answer accuracy 看不到的问题。

例如：

```text
答案对，但推理过程错。
```

这种模型在新题上可能不可靠。

---

### 四、数学推理评估

数学评估常见指标：

1. final answer accuracy。
2. step correctness。
3. solve rate by difficulty。
4. self-consistency gain。
5. verifier rerank gain。
6. symbolic check pass rate。
7. contamination rate。
8. error type distribution。

常见任务包括：

1. 小学应用题。
2. 代数。
3. 几何。
4. 数论。
5. 概率组合。
6. 竞赛题。

数学评估要特别注意：

```text
答案格式归一化。
```

例如 `1/2`、`0.5`、`50%` 可能等价。

如果答案抽取和归一化做不好，评估会失真。

---

### 五、代码推理评估

代码评估有更硬的执行信号。

常见指标：

1. pass@1。
2. pass@k。
3. compile rate。
4. test pass rate。
5. repair success rate。
6. average attempts。
7. runtime。
8. memory usage。
9. security violation rate。
10. diff minimality。

代码评估要注意：

1. 测试覆盖不足。
2. hidden tests。
3. hardcode 测试。
4. 依赖环境差异。
5. 代码安全。
6. repo-level side effects。

代码模型通过 benchmark 不代表能改真实仓库。

真实仓库任务还要评估：

1. 是否理解上下文。
2. 是否改对文件。
3. 是否新增测试。
4. 是否破坏现有行为。
5. 是否符合项目风格。

---

### 六、逻辑和多跳问答评估

逻辑和多跳问答通常不像数学那样容易自动验证。

常见指标：

1. answer correctness。
2. evidence recall。
3. reasoning path correctness。
4. supporting facts accuracy。
5. contradiction rate。
6. robustness to distractors。

例如多跳问答需要模型：

```text
找到证据 A -> 根据 A 找证据 B -> 合成答案
```

评估时不能只看最终答案。

还要看：

1. 是否找到了正确中间证据。
2. 是否忽略干扰证据。
3. 是否正确连接多跳关系。
4. 是否引用支持答案的来源。

这和 RAG 的 attribution 评估有重叠。

---

### 七、规划和 Agent 推理评估

规划任务评估更复杂。

例如：

```text
帮我规划一个三天学习计划，并根据每天反馈调整。
```

评估指标包括：

1. task success rate。
2. plan validity。
3. constraint satisfaction。
4. tool call accuracy。
5. recovery rate。
6. safety violation rate。
7. user confirmation correctness。
8. trace quality。
9. cost and latency。

Agent 推理不能只看最终回答。

因为 Agent 可能最终回答看起来对，但中间越权调用、泄露数据或浪费大量工具调用。

所以要评估完整 trajectory。

---

### 八、Test-Time Compute 曲线

Reasoning model 的一个关键评估是：

```text
随着推理预算增加，性能是否提升？
```

可以比较：

1. direct answer。
2. single CoT。
3. self-consistency N=5。
4. self-consistency N=20。
5. verifier rerank。
6. search。

记录：

1. accuracy。
2. solve rate。
3. token cost。
4. latency。
5. tool calls。
6. cost per solved task。

理想模型应该在难题上能从更多 compute 中获益。

如果增加 20 倍 compute 只提升 1%，说明收益很低。

如果更多 compute 反而降低准确率，说明推理过程可能不稳定。

---

### 九、鲁棒性评估

Reasoning model 可能对题目表述很敏感。

鲁棒性评估包括：

1. 改写题干。
2. 打乱无关信息顺序。
3. 加入 distractors。
4. 替换数字。
5. 改变单位。
6. 改变语言。
7. 加入无关上下文。
8. 测试同构题。

如果模型只在原题上对，改写后错，说明泛化不足。

数学和代码任务尤其要做变体测试。

因为很多模型可能记住模板。

---

### 十、数据污染检测

Reasoning benchmark 常被污染。

污染来源包括：

1. 预训练语料包含 benchmark。
2. SFT 数据包含题目或题解。
3. 合成数据从 benchmark 改写。
4. 评估 prompt 泄露答案。
5. 公开 leaderboard 被反复优化。

检测方法：

1. Exact match。
2. Fuzzy match。
3. Embedding similarity。
4. 题干、答案、解法分别查重。
5. 时间切分。
6. 新构造私有测试集。
7. 观察异常高分样本。

污染检测不是可选项。

没有污染检测的 reasoning 分数可信度很低。

---

### 十一、LLM-as-a-Judge 的使用边界

LLM judge 可以评估开放推理。

但它有风险：

1. 偏好流畅答案。
2. 被错误 CoT 说服。
3. 漏看长推理中的错误。
4. 对数学和代码细节不可靠。
5. 不稳定。
6. 和被评模型同源时有偏。

适合 LLM judge 的场景：

1. 开放答案初筛。
2. 解释质量评估。
3. 人工评估辅助。
4. 错误类型归类。

不适合完全依赖 LLM judge 的场景：

1. 数学最终判分。
2. 代码正确性。
3. 高风险安全决策。
4. 精确事实验证。

最好组合：

```text
工具验证 + verifier + LLM judge + 人工抽查
```

---

### 十二、错误分析

Reasoning 评估必须做 error analysis。

错误类型可以包括：

1. 读题错误。
2. 条件遗漏。
3. 错误建模。
4. 中间计算错误。
5. 逻辑跳步。
6. 工具调用错误。
7. Verifier 误判。
8. 答案抽取错误。
9. 格式错误。
10. 安全或权限错误。

错误分析的价值是指导下一步优化。

如果主要错在计算，可以接工具。

如果主要错在读题，需要数据和 prompt。

如果主要错在搜索剪枝，需要改 verifier。

如果主要错在污染，需要重建评估集。

---

### 十三、评估体系设计

一个可信 reasoning 评估体系可以分层：

#### 1. Unit eval

单题、单函数、单步骤评估。

#### 2. Benchmark eval

数学、代码、逻辑、多跳问答等标准集。

#### 3. Robustness eval

改写、扰动、distractor、同构题。

#### 4. Test-time scaling eval

不同预算下的性能曲线。

#### 5. Safety eval

越权、危险工具、敏感信息、错误自信。

#### 6. Human eval

复杂开放题和真实任务人工评估。

#### 7. Regression eval

历史失败样本回归测试。

#### 8. Production eval

真实用户任务、延迟、成本、满意度和事故率。

这种体系比单一 benchmark 分数可靠得多。

---

### 十四、真实项目中的坑

#### 1. 只看总分

平均分掩盖难题和高风险场景退化。

#### 2. 忽略成本

模型靠 50 倍采样拿高分，线上不可用。

#### 3. 不做污染检测

分数虚高。

#### 4. LLM judge 直接当真值

Judge 本身会错。

#### 5. 不评估过程

答案对但推理过程不可靠。

#### 6. 不做错误归因

只知道错了，不知道该改数据、模型、verifier 还是工具。

#### 7. Benchmark 和业务无关

数学题高分不代表企业 Agent 任务成功。

---

### 十五、面试问答

#### 问题 1：Reasoning model 评估和普通 chat model 评估有什么区别？

可以这样回答：

```text
普通 chat model 更多看有用性、流畅度和指令遵循；reasoning model 还要看多步推理正确性、过程可靠性、可验证结果、test-time compute 曲线、鲁棒性和污染风险。
```

#### 问题 2：为什么 final answer accuracy 不够？

可以这样回答：

```text
因为最终答案可能碰巧正确，过程可能错误；也可能答案错但前面步骤大部分正确。只看 final answer 无法定位错误，也无法评估过程可靠性和泛化能力。
```

#### 问题 3：如何评估 test-time compute scaling？

可以这样回答：

```text
比较 direct answer、CoT、多采样、verifier、search 等不同预算下的准确率、solve rate、token、延迟、工具调用和 cost per solved task，画成本收益曲线。
```

#### 问题 4：Reasoning benchmark 为什么容易污染？

可以这样回答：

```text
因为数学、代码和逻辑题常在网上公开，预训练、SFT 或合成数据都可能包含原题、题解或改写题。需要 exact/fuzzy/embedding 去重和私有 holdout。
```

#### 问题 5：LLM-as-a-judge 能不能评估 reasoning？

可以这样回答：

```text
可以作为辅助，但不能完全依赖。LLM judge 可能偏好流畅解释、被错误推理说服，也可能漏看细节。数学和代码最好用工具、测试或符号验证，开放题再结合人工和 judge。
```

#### 问题 6：如何设计一个 reasoning model 评估体系？

可以这样回答：

```text
我会分层设计：标准 benchmark、过程正确性、鲁棒性、污染检测、test-time compute 曲线、安全评估、错误分析、回归测试和真实任务评估，并同时报告准确率、成本、延迟和错误类型。
```

---

### 十六、常见误区

1. 误区：Reasoning 模型只要数学分高就行。
   纠正：还要看代码、逻辑、规划、真实任务和安全。

2. 误区：答案对就说明推理对。
   纠正：答案可能碰巧对，过程仍可能错。

3. 误区：更长 CoT 就代表更强 reasoning。
   纠正：长推理可能只是冗余或错误累积。

4. 误区：Benchmark 分数可直接代表生产效果。
   纠正：生产还要看延迟、成本、工具、用户场景和风险。

5. 误区：LLM judge 足够评估 reasoning。
   纠正：要结合工具验证、人工抽查和标准答案。

6. 误区：只需要一次离线评估。
   纠正：需要持续回归测试和线上监控。

---

### 十七、小练习

1. 为一个数学 reasoning benchmark 设计评估指标。
2. 为一个代码 benchmark 设计 pass@k、执行反馈和安全指标。
3. 构造一个答案正确但推理过程错误的样本，并设计评估方法。
4. 设计一个 test-time compute 曲线实验。
5. 设计一个 reasoning benchmark 污染检测流程。
6. 比较 LLM judge、PRM、程序验证和人工评估。
7. 为多跳问答设计 evidence 和 attribution 评估。
8. 为 Agent planning 设计 trajectory-level 评估。
9. 对 20 个 reasoning 错误样本做错误类型归因。
10. 用 3 分钟回答：“如何可信评估一个 reasoning model？”

### 本讲总结

本讲最重要的结论：

1. Reasoning model 评估要看最终答案、推理过程、鲁棒性、成本和安全。
2. Final answer accuracy 必要但不充分。
3. 数学、代码、逻辑、多跳问答和规划任务需要不同评估方法。
4. Test-time compute 曲线是 reasoning model 的关键评估维度。
5. 数据污染会让 reasoning benchmark 分数严重虚高。
6. LLM-as-a-judge 可以辅助评估，但不能替代工具验证和人工校准。
7. 错误分析比单一分数更能指导模型、数据、verifier 和工具改进。
8. 面试中要把 reasoning 评估讲成多层体系，而不是只报一个 benchmark 分数。

## 第 95 讲：从 Chat Model 到 Reasoning Model

### 本讲目标

学完本讲，你应该能回答六个问题：

1. Chat model 和 reasoning model 的核心区别是什么。
2. 为什么单纯指令微调不足以得到强 reasoning model。
3. Reasoning model 在数据、训练、推理和评估上有哪些变化。
4. CoT、verifier、search、test-time compute 如何共同推动 reasoning model。
5. 从产品和系统角度，reasoning model 带来哪些新设计问题。
6. 面试中如何完整回答“如何从 chat model 走向 reasoning model”。

第八部分从 CoT 讲起，依次讲了 self-consistency、verifier、search、test-time compute、数学训练、代码执行反馈、合成推理数据和 reasoning 评估。

这一讲做一个总收束：

```text
Chat model 如何演化为 reasoning model？
```

Chat model 的核心能力是对话、指令遵循和通用问答。

Reasoning model 的核心能力是面对复杂任务时进行多步思考、验证、搜索和纠错。

两者不是完全割裂。

Reasoning model 仍然需要聊天能力。

但它在训练目标、数据结构、推理方式和评估体系上都发生了变化。

---

### 一、Chat Model 的典型目标

Chat model 主要解决：

1. 理解用户指令。
2. 给出有帮助的回答。
3. 保持对话自然。
4. 遵守安全规范。
5. 适配多种通用场景。

训练上常见流程是：

```text
预训练 -> 指令微调 -> 偏好对齐 -> 安全对齐
```

评估上常看：

1. helpfulness。
2. harmlessness。
3. instruction following。
4. fluency。
5. human preference。

Chat model 很适合：

1. 问答。
2. 总结。
3. 写作。
4. 翻译。
5. 头脑风暴。
6. 普通助手任务。

但遇到复杂数学、代码、规划和多步工具任务时，普通 chat model 容易出现不稳定推理。

---

### 二、Reasoning Model 的典型目标

Reasoning model 更关注：

1. 多步问题分解。
2. 中间状态维护。
3. 逻辑一致性。
4. 错误检查。
5. 试错和回退。
6. 使用 verifier 和工具。
7. 随 test-time compute 增加而提升。

它适合：

1. 数学推理。
2. 代码推理。
3. 科学问题。
4. 复杂规划。
5. 多跳问答。
6. Agent 任务。
7. 高价值决策辅助。

一句话：

```text
Chat model 更像会沟通的通用助手，reasoning model 更像会解题、会验证、会试错的求解器。
```

---

### 三、为什么指令微调不够

指令微调可以教模型“怎么回答”。

但 reasoning 需要模型“怎么思考”。

普通 SFT 数据可能包含大量问答，但缺少：

1. 高质量多步推理过程。
2. 错误步骤标注。
3. 失败后修正轨迹。
4. verifier 反馈。
5. 搜索和多候选选择。
6. 难题上的过程监督。

所以只做指令微调，模型可能学会：

```text
看起来像在推理。
```

但未必学会：

```text
可靠地推理。
```

这就是 CoT 忠实性、过程监督和 verifier 重要的原因。

---

### 四、数据上的变化

从 chat model 到 reasoning model，数据形态发生变化。

Chat 数据常见是：

```text
用户问题 -> 助手回答
```

Reasoning 数据更像：

```text
问题 -> 分步推理 -> 中间检查 -> 最终答案
```

或者：

```text
问题 -> 多个候选解法 -> verifier 分数 -> 选择结果
```

或者：

```text
任务 -> 工具调用 -> observation -> 修正 -> 完成
```

需要的数据包括：

1. CoT 解法。
2. 过程标注。
3. 正负样本。
4. hard negatives。
5. 工具执行轨迹。
6. 代码测试反馈。
7. 数学 verifier 数据。
8. 合成推理数据。
9. 失败修复轨迹。

Reasoning 数据更贵，也更需要验证。

---

### 五、训练上的变化

Reasoning model 训练不只是普通 SFT。

常见训练组件包括：

#### 1. CoT SFT

让模型学习分步解题格式和基本推理。

#### 2. Rejection sampling SFT

采样多条解法，用 verifier 或工具过滤，保留高质量样本训练。

#### 3. ORM / PRM

训练结果奖励模型或过程奖励模型。

#### 4. Preference optimization

让模型偏好更正确、更简洁、更稳定的解法。

#### 5. RL

用可验证 reward 直接优化推理成功率。

但风险更高，需要强 reward 和防 reward hacking。

#### 6. Distillation

从强推理系统蒸馏能力到更便宜模型。

这些训练方法的共同目标是：

```text
不只让模型生成答案，而是让模型生成更可靠的求解过程。
```

---

### 六、推理方式上的变化

Chat model 通常一次生成答案。

Reasoning model 常常采用更复杂的推理流程。

例如：

1. 先分析问题。
2. 生成中间步骤。
3. 采样多个候选。
4. 用 verifier 检查。
5. 必要时 search。
6. 调用工具。
7. 修正错误。
8. 输出最终答案。

推理流程可以是：

```text
direct answer
CoT
self-consistency
verifier rerank
search
tool execution
human review
```

并且可以根据任务难度动态选择。

这就是第 90 讲讲的 test-time compute scaling。

---

### 七、评估方式上的变化

Chat model 可以用人类偏好评估很多场景。

Reasoning model 需要更硬的评估。

包括：

1. final answer accuracy。
2. process correctness。
3. pass@k。
4. solve rate。
5. verifier gain。
6. test-time compute curve。
7. contamination check。
8. robustness。
9. tool trajectory quality。
10. cost per solved task。

Reasoning model 的分数如果不报告推理预算，意义不完整。

例如：

```text
模型 A accuracy 90%，但每题采样 100 次。
模型 B accuracy 85%，但单次回答。
```

这两个结果不能简单比较。

必须同时报告成本、延迟和预算。

---

### 八、系统架构上的变化

Reasoning model 不一定只是一个模型。

它经常是一个系统。

系统组件包括：

1. Generator。
2. Verifier。
3. Reward model。
4. Search controller。
5. Tool executor。
6. Memory/state manager。
7. Budget allocator。
8. Safety guardrail。
9. Trace logger。

可以理解为：

```text
Reasoning system = 模型生成 + 评价函数 + 搜索策略 + 工具反馈 + 预算控制 + 安全审计
```

这比单个 chat model 复杂得多。

但也更适合复杂任务。

---

### 九、产品形态上的变化

Chat model 产品通常追求：

1. 快速响应。
2. 自然对话。
3. 用户体验流畅。
4. 覆盖广泛任务。

Reasoning model 产品还要设计：

1. 是否显示思考过程。
2. 用户是否愿意等待。
3. 任务是否需要高预算。
4. 什么时候拒答。
5. 什么时候请求确认。
6. 如何展示证据和结论。
7. 如何解释不确定性。

例如：

```text
快速模式：直接回答，低成本。
深度思考模式：多步推理和验证，高成本。
```

用户体验上要明确：

1. 为什么需要等待。
2. 结果可信度如何。
3. 是否用了工具。
4. 哪些地方不确定。

---

### 十、Hidden Reasoning 与可见解释

Reasoning model 常有内部 reasoning。

但内部 reasoning 不一定全部展示给用户。

原因包括：

1. 中间过程可能冗长。
2. 中间过程可能不稳定。
3. 可能包含错误假设。
4. 可能泄露安全策略。
5. 可能让用户过度相信错误推理。

生产系统常见做法是：

```text
内部使用长推理，外部展示简洁解释、关键步骤、证据和最终答案。
```

这不是不透明。

而是区分：

1. 内部计算轨迹。
2. 用户可读解释。
3. 可审计 trace。

面试中要避免说“完整 CoT 必须展示才可信”。

更合理的是提供可验证证据和必要解释。

---

### 十一、从 Chat 到 Reasoning 的实现路径

如果已有一个 chat model，要增强 reasoning，可以分阶段。

#### 阶段 1：数据增强

收集数学、代码、逻辑、规划等高质量 CoT 数据。

#### 阶段 2：CoT SFT

让模型学会基本分步推理。

#### 阶段 3：Verifier

训练或接入答案验证、过程验证、代码测试和工具检查。

#### 阶段 4：多候选与 rerank

使用 self-consistency 和 verifier reranking。

#### 阶段 5：Search

对高难任务引入 beam search、ToT 或 MCTS。

#### 阶段 6：合成数据闭环

用强推理系统生成数据，再过滤、训练和评估。

#### 阶段 7：动态预算和产品化

根据任务难度选择 direct、CoT、search、tool 或人工审核。

这条路径比“一次训练一个 reasoning model”更工程化。

---

### 十二、真实项目中的坑

#### 1. 把长回答当 reasoning

长不等于对。

#### 2. 只做 CoT SFT

没有 verifier 和评估，模型可能只是模仿推理格式。

#### 3. 只看 benchmark 分数

忽略污染、成本、延迟和真实任务。

#### 4. 没有预算控制

高推理预算导致线上不可用。

#### 5. Verifier 太弱

错误路径被高分选择。

#### 6. 忽略安全

Reasoning + tool use 可能造成真实副作用。

#### 7. 误把 reasoning model 当万能模型

简单任务不需要深度推理。

---

### 十三、面试问答

#### 问题 1：Chat model 和 reasoning model 的区别是什么？

可以这样回答：

```text
Chat model 更强调对话、指令遵循和通用帮助；reasoning model 更强调多步推理、验证、搜索、纠错和随 test-time compute 增加而提升的能力。Reasoning model 往往不只是一个模型，而是模型、verifier、工具和预算策略组成的系统。
```

#### 问题 2：为什么普通 SFT 不够训练 reasoning model？

可以这样回答：

```text
普通 SFT 多是问题到回答，缺少高质量中间步骤、过程监督、错误样本、执行反馈和 verifier 信号。模型可能学会推理格式，但不一定学会可靠推理。
```

#### 问题 3：从 chat model 到 reasoning model 需要哪些关键组件？

可以这样回答：

```text
需要高质量 CoT 和过程数据、结果和过程 verifier、多候选采样、search、test-time compute budget、工具执行反馈、合成数据闭环以及更严格的 reasoning 评估体系。
```

#### 问题 4：Reasoning model 是否应该展示完整 CoT？

可以这样回答：

```text
不一定。系统可以内部使用长推理提升质量，但对用户展示简洁解释、关键证据和最终答案。完整 CoT 可能冗长、不稳定，也可能带来安全和误导风险。
```

#### 问题 5：如何比较两个 reasoning model？

可以这样回答：

```text
不能只看准确率，还要看推理预算、token、延迟、成本、test-time compute 曲线、污染检测、鲁棒性、过程正确性和真实任务表现。
```

#### 问题 6：如何把 reasoning model 产品化？

可以这样回答：

```text
要做任务难度识别和动态预算分配，简单任务直接回答，复杂任务启用推理、验证、工具或 search；同时记录 trace，控制成本和延迟，处理安全确认，并向用户展示关键证据和不确定性。
```

---

### 十四、常见误区

1. 误区：Reasoning model 就是会输出很长 CoT 的模型。
   纠正：核心是可靠求解、验证和纠错，不是文本长度。

2. 误区：Chat model 加一句“逐步思考”就是 reasoning model。
   纠正：还需要训练数据、verifier、评估和推理时策略。

3. 误区：Reasoning model 一定适合所有任务。
   纠正：简单任务直接回答更快更便宜。

4. 误区：Benchmark 高分就代表 reasoning 强。
   纠正：要看污染、鲁棒性、真实任务和成本。

5. 误区：Verifier 可以完全替代模型能力。
   纠正：Verifier 只能筛选和指导，generator 仍要能产生好候选。

6. 误区：Test-time compute 可以无限提升能力。
   纠正：存在收益递减和评价函数错误的问题。

---

### 十五、小练习

1. 对比 chat model 和 reasoning model 的训练目标。
2. 设计一个从 chat model 升级为 reasoning model 的三阶段路线。
3. 为数学任务设计 CoT SFT + verifier + self-consistency 方案。
4. 为代码任务设计执行反馈增强方案。
5. 设计一个 reasoning system 架构图，包含 generator、verifier、search、tool 和 budget allocator。
6. 设计一个用户可见解释格式，不暴露完整内部 CoT。
7. 分析为什么长 CoT 不等于强 reasoning。
8. 比较单模型 reasoning 和系统级 reasoning 的优缺点。
9. 设计一个 dynamic reasoning mode：快速模式、深度模式、高风险模式。
10. 用 3 分钟回答：“如何从 Chat Model 走向 Reasoning Model？”

### 本讲总结

本讲最重要的结论：

1. Chat model 强调对话和指令遵循，reasoning model 强调复杂任务求解、验证、搜索和纠错。
2. 普通 SFT 不足以训练强 reasoning，需要 CoT、过程监督、verifier、执行反馈和合成推理数据。
3. Reasoning model 的数据形态从“问题-回答”扩展为“问题-过程-验证-修正-答案”。
4. Reasoning 推理方式从一次生成扩展为多候选、rerank、search、tool feedback 和动态预算。
5. Reasoning 评估必须同时看答案、过程、鲁棒性、污染、成本和 test-time compute 曲线。
6. Reasoning model 往往是系统能力，不只是单个模型能力。
7. 产品化时要在准确率、延迟、成本、安全和用户体验之间做权衡。
8. 面试中要把从 chat 到 reasoning 讲成“数据、训练、推理、验证、评估、产品化”的完整演进。

第八部分到这里结束。下一部分进入论文精读与开放研究题。
