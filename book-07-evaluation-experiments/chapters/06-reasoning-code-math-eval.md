# 第六章：Reasoning、Code、Math Eval

重点：数学推理、代码生成、单元测试、过程评估、verifier、pass@k、self-consistency。

面试重点：推理能力评估要区分最终答案、推理过程和泛化难度。

## 本章目标

学完本章，你要能回答：

1. Reasoning、Code、Math 任务为什么不能只看最终答案？
2. 数学推理评估常见指标有哪些？
3. 代码生成为什么常用单元测试和 pass@k？
4. `pass@k` 的直觉和计算方式是什么？
5. 什么是 self-consistency，它如何用于评估？
6. Verifier 和 process reward model 在评估中有什么作用？
7. 如何设计可信的 reasoning/code/math eval？

Reasoning、Code、Math 是大模型能力评估中最容易被误读的三类任务。

因为它们看起来有明确答案：数学题对不对、代码能不能跑、逻辑结论是否成立。

但真实评估并没有这么简单。

一个模型可能最终答案对了，但推理过程胡说。

一个模型可能代码通过了公开样例，但隐藏测试失败。

一个模型可能在经典 benchmark 上分数很高，但只是见过题目或记住了解法。

所以这类评估必须同时关注：最终结果、推理过程、测试覆盖、泛化难度和污染风险。

面试表达：Reasoning、Code、Math eval 的核心不是“算一个准确率”，而是判断模型是否真的具备可泛化的问题求解能力。

## 本章资料边界

本章第二轮精修参考了 [HumanEval / Codex](https://arxiv.org/abs/2107.03374)、[GSM8K / Training Verifiers](https://arxiv.org/abs/2110.14168)、[MATH](https://arxiv.org/abs/2103.03874)、[Self-Consistency](https://arxiv.org/abs/2203.11171)、[APPS](https://arxiv.org/abs/2105.09938)、[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 和前序污染检测 / LLM-as-a-Judge 章节的资料边界。

本章聚焦 reasoning、code、math eval 的评估工程口径：答案抽取、答案归一化、最终答案 accuracy、单元测试、隐藏测试、`pass@k`、self-consistency、过程评估、verifier、错误分类、污染风险和评估报告。不展开完整代码沙箱实现、形式化证明系统、竞赛级题库构造、复杂 verifier 训练或在线 Agent 长链路评估平台。

## 本章核心公式

一个 reasoning / code / math 评估样本可以抽象成：

```math
e_i=(x_i,y_i,r_i,z_i)
```

其中 `x_i` 是题目或任务描述，`y_i` 是标准答案、单元测试或期望行为，`r_i` 是可选 rubric、过程标注或参考解法，`z_i` 是任务类型、难度、来源、时间和污染风险等元数据。

数学最终答案评估通常先抽取和归一化：

```math
\hat{a}_i=\nu(\mathrm{extract}(\hat{y}_i))
```

其中 `\nu` 表示答案规范化，例如分数、小数、百分数、集合顺序和单位处理。最终答案 accuracy 为：

```math
A_{\mathrm{ans}}
=
\frac{1}{N}
\sum_{i=1}^{N}
I(\hat{a}_i=a_i)
```

如果输出无法抽取答案，应该单独计入解析失败率：

```math
R_{\mathrm{parseFail}}
=
\frac{N_{\mathrm{parseFail}}}{N}
```

代码任务中，候选程序 `g_{ij}` 在第 `m` 个测试上是否通过可以写成：

```math
z_{ijm}=I(T_{im}(g_{ij})=1)
```

单个候选通过全部测试的标记为：

```math
p_{ij}=\prod_{m=1}^{M_i}z_{ijm}
```

如果同一道题采样 `n` 个候选，其中 `c_i` 个候选通过隐藏测试，则 HumanEval / Codex 常用的 `pass@k` 无偏估计为：

```math
\widehat{\mathrm{pass@}k}_i
=
1-
\frac{\binom{n-c_i}{k}}{\binom{n}{k}}
```

当 `n-c_i<k` 时，`pass@k` 记为 1，因为任意抽 `k` 个候选都会包含至少一个通过者。

Self-consistency 对同一题采样 `K` 个推理路径并投票：

```math
\hat{a}_{\mathrm{sc}}
=
\mathrm{mode}(\hat{a}^{(1)},\ldots,\hat{a}^{(K)})
```

答案分布的不确定性可以用 entropy 表示：

```math
H_{\mathrm{ans}}
=
-
\sum_a p(a)\log p(a)
```

过程评估可以把第 `l` 步是否正确记为 `v_{il}`：

```math
S_{\mathrm{proc},i}
=
\frac{1}{L_i}
\sum_{l=1}^{L_i}v_{il}
```

最终 reasoning / code / math 评估门禁可以写成：

```math
G_{\mathrm{rcm}}
=
G_{\mathrm{answer}}
\land G_{\mathrm{parse}}
\land G_{\mathrm{test}}
\land G_{\mathrm{passK}}
\land G_{\mathrm{process}}
\land G_{\mathrm{contam}}
\land G_{\mathrm{report}}
```

也就是最终答案可信、解析失败率可控、单元测试和隐藏测试有效、`pass@k` 设置清楚、过程评估覆盖关键样本、污染风险已审计、报告说明采样和解码参数。

## 1. 为什么单纯 accuracy 不够

最直观的评估方法是 accuracy。

给模型一道题，看最终答案是否正确。

例如：

```text
Question: If x + 3 = 10, what is x?
Gold answer: 7
Model answer: x = 7
```

这类题可以直接判对。

但复杂推理任务里，accuracy 有几个问题。

### 1.1 答案格式可能不同

数学题中，下面答案可能等价：

```text
1/2
0.5
50%
\frac{1}{2}
```

如果只做字符串匹配，会把正确答案判错。

代码题中，模型可能写出不同实现，只要功能正确就应该算对。

### 1.2 最终答案正确不代表过程正确

模型可能通过猜测、模式匹配或泄漏记忆得到正确答案。

例如：

```text
题目：某经典竞赛题。
模型：推导过程错误，但最后写出正确答案。
```

如果只看最终答案，会高估推理能力。

### 1.3 最终答案错误不代表完全没有能力

复杂问题可能有多个步骤。

模型前 8 步正确，最后一步算术错误。

另一个模型第一步就理解错题意。

两者最终都错，但能力差异很大。

过程评估可以区分这种情况。

### 1.4 benchmark 容易被过拟合

经典数学、代码和推理 benchmark 公开传播广，容易进入训练数据。

如果模型见过题，accuracy 会虚高。

### 1.5 测试集覆盖有限

代码生成尤其明显。

如果单元测试只覆盖 happy path，错误代码也可能通过。

面试表达：accuracy 是必要指标，但不能单独代表 reasoning 能力，还要看答案规范化、过程正确性、测试覆盖、泛化难度和污染风险。

## 2. Reasoning Eval 的任务类型

Reasoning eval 不是单一任务。

常见类型包括：

1. 数学推理。
2. 逻辑推理。
3. 常识推理。
4. 多跳问答。
5. 规划任务。
6. 工具调用推理。
7. 代码推理。
8. 长上下文推理。

不同任务的评估重点不同。

### 2.1 数学推理

数学推理通常有明确答案。

评估重点：

1. 是否理解题意。
2. 推导是否正确。
3. 计算是否准确。
4. 答案格式是否可解析。
5. 是否能泛化到新题。

### 2.2 逻辑推理

逻辑推理关注条件、约束和结论。

例如：

```text
所有 A 都是 B。
所有 B 都是 C。
是否能推出所有 A 都是 C？
```

评估重点是是否遵守逻辑规则，而不是语言是否流畅。

### 2.3 多跳问答

多跳问答要求模型整合多个事实。

例如先找到人物，再找人物所属机构，再回答机构所在地。

评估重点：

1. 中间事实是否正确。
2. 检索证据是否支持答案。
3. 是否出现 unsupported reasoning。

### 2.4 规划任务

规划任务关注目标分解和步骤安排。

例如 Agent 要完成订票、查天气、安排行程。

评估不能只看最终文本，还要看工具调用顺序和状态更新。

面试表达：Reasoning eval 要先明确任务类型，因为数学、逻辑、多跳问答、规划和代码推理的错误模式不同。

## 3. 数学推理评估

数学推理是 reasoning eval 的核心场景之一。

它的优点是答案相对客观。

它的难点是过程复杂、格式多样、题目容易污染。

### 3.1 最终答案评估

最终答案评估的基本流程：

1. 抽取模型最终答案。
2. 规范化答案格式。
3. 与标准答案比较。
4. 对无法自动判断的样本做人审或 judge。

例如模型输出：

```text
After solving the equation, we get x = 7. Therefore the answer is 7.
```

评估脚本需要抽取 `7`，而不是直接比较整段文本。

### 3.2 答案规范化

数学答案规范化常见处理：

1. 去掉空格和无关文字。
2. 统一分数、小数、百分数。
3. 解析 LaTeX 表达式。
4. 对等价表达式做符号化比较。
5. 对浮点答案设置容忍误差。
6. 对集合、区间、向量等结构化答案排序或解析。

例如：

```text
Gold: 1/2
Pred: 0.5000
```

如果题目允许小数，两者应判为等价。

### 3.3 exact match 的局限

Exact match 简单可复现，但很脆弱。

它适合：

1. 单选题。
2. 整数答案题。
3. 格式严格的填空题。

不适合：

1. 多种等价表达式。
2. 证明题。
3. 开放式推导。
4. 需要解释的数学题。

### 3.4 Step-level 评估

Step-level 评估关注每一步推理是否正确。

例如：

```text
Step 1: x + 3 = 10
Step 2: x = 10 - 3
Step 3: x = 7
```

可以评估：

1. 每步是否由前一步推出。
2. 每步是否违反题目条件。
3. 中间变量是否一致。
4. 最终答案是否由过程得到。

这种评估比最终答案更细，但标注成本更高。

### 3.5 Proof-level 评估

证明题更难。

它不一定有唯一推理路径。

常见做法：

1. 人工专家评审。
2. 使用形式化证明系统验证。
3. 用 LLM judge 辅助，但需要人工校准。
4. 把证明拆成 claim-level 检查。

面试表达：数学 eval 可以用最终答案 accuracy 做主指标，但必须配合答案规范化、过程检查和污染检测，否则很容易误判模型能力。

## 4. Code Eval 的核心思想

代码评估和文本评估不同。

代码的核心不是“看起来像正确答案”，而是能否在测试下执行正确。

因此代码生成评估通常使用单元测试。

基本流程：

1. 给模型题目描述和函数签名。
2. 模型生成代码。
3. 把代码放到沙箱中运行。
4. 执行公开测试和隐藏测试。
5. 根据测试通过情况计算指标。

例如：

```python
def add(a: int, b: int) -> int:
    return a + b
```

只要通过测试，就可以认为功能正确。

### 4.1 为什么不能只做人眼评估

代码看起来合理，不代表能跑。

常见问题包括：

1. 语法错误。
2. 边界条件错误。
3. 时间复杂度过高。
4. 依赖不存在。
5. 输入输出格式错误。
6. 安全风险。

单元测试能发现很多人眼容易漏掉的问题。

### 4.2 公开测试和隐藏测试

公开测试用于帮助模型或开发者理解题目。

隐藏测试用于真正评估泛化。

如果只用公开测试，模型可能写出 hard-code 代码。

例如：

```python
def solve(x):
    if x == 1:
        return 2
    if x == 2:
        return 4
```

这类代码能过样例，但不能解决真实问题。

### 4.3 测试覆盖率

测试质量决定 code eval 的可信度。

好的测试应覆盖：

1. 正常输入。
2. 边界输入。
3. 极端规模。
4. 随机输入。
5. 异常输入。
6. 性能约束。
7. 多种等价场景。

如果测试太弱，分数会虚高。

### 4.4 执行安全

代码评估必须在沙箱里运行。

因为模型生成代码可能包含：

1. 删除文件。
2. 访问网络。
3. 读取环境变量。
4. 死循环。
5. 大量占用内存。
6. 执行系统命令。

工程上要限制：

1. CPU 时间。
2. 内存。
3. 文件系统权限。
4. 网络访问。
5. 进程数量。
6. 可导入库范围。

面试表达：Code eval 的核心是“可执行验证”，但测试覆盖、隐藏测试和沙箱安全决定了评估可信度。

## 5. pass@k

`pass@k` 是代码生成评估中最常见的指标之一。

它回答的问题是：如果模型为同一道题生成 k 个候选答案，至少有一个通过测试的概率是多少？

### 5.1 为什么需要 pass@k

代码生成通常有采样随机性。

同一个模型生成一次可能失败，生成多次可能有一个成功。

这类似真实使用场景：

1. 用户可以让模型重试。
2. Agent 可以生成多个候选并运行测试。
3. 系统可以用 verifier 选择更好的答案。

所以只看 pass@1 不完整。

### 5.2 pass@1 和 pass@k 的区别

`pass@1`：只生成一个答案，看是否通过。

`pass@k`：生成 k 个答案，只要有一个通过就算成功。

通常：

```text
pass@1 <= pass@5 <= pass@10
```

但 pass@k 越高，不代表单次输出越可靠。

它可能只是采样次数更多。

### 5.3 pass@k 的估计公式

实际评估中，通常为每道题采样 n 个候选，其中 c 个通过测试。

要估计从 n 个候选中随机抽 k 个，至少一个通过的概率。

常用无偏估计：

```math
\widehat{\mathrm{pass@}k}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
```

其中：

1. `n` 是采样总数。
2. `c` 是正确候选数。
3. `k` 是允许尝试次数。
4. `binom(a,b)` 是组合数。

如果 `n - c < k`，说明任意抽 k 个都至少包含一个正确答案，`pass@k = 1`。

### 5.4 最小代码示例

下面是一个简单的 `pass@k` 计算函数。

```python
from math import comb


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k from n samples with c correct samples."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


print(estimate_pass_at_k(n=10, c=2, k=1))
print(estimate_pass_at_k(n=10, c=2, k=5))
```

输入：采样数量 `n`、正确数量 `c`、尝试次数 `k`。

输出：估计的 `pass@k`。

这个公式的直觉是：先算 k 个候选全都错误的概率，再用 1 减掉它。

### 5.5 pass@k 的坑

使用 `pass@k` 时要注意：

1. 不同 temperature 会影响结果。
2. 不同采样数量 n 会影响估计方差。
3. 测试越弱，pass@k 越容易虚高。
4. k 越大，越接近“搜索能力”而不是单次生成能力。
5. 如果候选高度相似，多采样收益可能很小。
6. 如果用公开测试筛选候选，会引入额外变量。

面试表达：pass@k 衡量多次采样下至少一次成功的概率，适合代码生成，但必须报告采样设置、temperature、测试强度和 k 值，否则不可比较。

## 6. Self-Consistency

Self-consistency 是 reasoning eval 和 reasoning inference 中常见方法。

核心思想：对同一道题采样多个推理路径，再对最终答案投票。

如果多数推理路径得到同一个答案，该答案更可能正确。

### 6.1 基本流程

Self-consistency 的流程：

1. 用较高 temperature 生成多个推理过程。
2. 抽取每个推理过程的最终答案。
3. 对最终答案做归一化。
4. 多数投票或加权投票。
5. 输出得票最高的答案。

例如：

```text
Sample 1: answer = 42
Sample 2: answer = 42
Sample 3: answer = 40
Sample 4: answer = 42
Sample 5: answer = 41
Final: 42
```

### 6.2 为什么有效

单条 chain-of-thought 可能偶然走错。

多条推理路径可以降低单次采样错误的影响。

如果问题确实有稳定解，正确答案可能在多次采样中更频繁出现。

### 6.3 作为评估指标

Self-consistency 可以作为推理时增强方法，也可以作为评估分析工具。

例如同时报告：

1. greedy accuracy。
2. self-consistency accuracy。
3. answer entropy。
4. majority margin。

其中 answer entropy 可以衡量模型不确定性。

如果多个答案分布很分散，说明模型不稳定。

投票结果可以写成：

```math
\hat{a}_{\mathrm{sc}}
=
\mathrm{mode}(\hat{a}^{(1)},\ldots,\hat{a}^{(K)})
```

多数票优势可以写成：

```math
M_{\mathrm{vote}}
=
p_{\mathrm{top1}}-p_{\mathrm{top2}}
```

`M_vote` 越小，说明前两名答案接近，模型对这道题并不稳定。answer entropy 则能把分布是否分散量化：

```math
H_{\mathrm{ans}}
=
-
\sum_a p(a)\log p(a)
```

### 6.4 局限性

Self-consistency 不是万能的。

它的问题包括：

1. 成本更高。
2. 多数答案可能共同错误。
3. 对开放式任务投票困难。
4. 对答案抽取和规范化敏感。
5. 如果题目被污染，多次采样会强化记忆答案。

面试表达：self-consistency 通过多路径采样和投票提升推理稳定性，评估时可以反映模型答案分布，但它增加成本，也不能替代题目难度和过程正确性分析。

## 7. Verifier

Verifier 是用于判断候选答案或推理过程是否正确的模型或规则系统。

在 reasoning/code/math eval 中，verifier 很重要。

因为生成模型负责提出答案，verifier 负责检查答案。

### 7.1 Outcome Verifier

Outcome verifier 只判断最终答案是否正确。

例如数学题中，输入题目、模型答案和标准答案，输出正确或错误。

代码题中，单元测试就是一种强 verifier。

优点：

1. 简单。
2. 成本低。
3. 容易聚合成指标。

缺点：

1. 看不到中间过程。
2. 无法区分错误发生在哪一步。
3. 可能奖励错误推理加正确答案。

### 7.2 Process Verifier

Process verifier 判断每一步推理是否合理。

例如：

```text
Step 1 correct: yes
Step 2 correct: yes
Step 3 correct: no
```

它可以定位错误步骤。

这对训练 process reward model、改进 reasoning 数据和 debug 模型很有价值。

### 7.3 Rule-based Verifier

有些任务可以用规则验证。

例如：

1. 数学表达式求值。
2. 代码单元测试。
3. JSON schema 校验。
4. 形式化逻辑验证。
5. 检查答案是否满足约束。

规则 verifier 的优点是客观、可复现。

缺点是覆盖有限。

### 7.4 LLM-based Verifier

LLM verifier 可以评估更开放的推理过程。

但它也会犯错。

使用时要注意：

1. 用人工 gold set 校准。
2. 设计明确 rubric。
3. 避免让 verifier 看到无关干扰信息。
4. 评估 verifier 自身准确率。
5. 分析 false positive 和 false negative。

面试表达：verifier 把“生成”和“检查”分离，代码任务中单元测试是强 verifier，数学和开放推理中可以结合规则、人审和 LLM verifier。

## 8. 过程评估与结果评估

Reasoning eval 可以分为 outcome evaluation 和 process evaluation。

### 8.1 Outcome Evaluation

Outcome evaluation 只看最终结果。

适合：

1. 答案唯一的数学题。
2. 单元测试充分的代码题。
3. 选择题。
4. 格式化任务。

优点：

1. 成本低。
2. 可自动化。
3. 容易横向比较。

缺点：

1. 看不到推理质量。
2. 无法定位错误步骤。
3. 容易被污染或投机策略影响。

### 8.2 Process Evaluation

Process evaluation 评估推理过程。

适合：

1. 多步数学题。
2. 复杂证明。
3. Agent 规划。
4. 多工具调用。
5. 教学解释任务。

优点：

1. 可解释性更强。
2. 能定位错误。
3. 能发现“答案对但过程错”。
4. 可以为训练提供细粒度信号。

缺点：

1. 标注成本高。
2. 多条正确路径难以统一。
3. LLM judge 容易偏好看起来流畅的过程。
4. 不一定和最终任务成功完全一致。

### 8.3 两者如何结合

真实评估中通常要组合使用：

1. 用 outcome metric 做主分数。
2. 用 process eval 分析错误原因。
3. 用 verifier 检查关键步骤。
4. 用人工样本校准自动评估。
5. 用 hidden set 防止过拟合。

面试表达：结果评估适合规模化比较，过程评估适合理解模型为什么对或错；两者不是替代关系，而是互补关系。

## 9. Benchmark 设计要点

Reasoning、Code、Math benchmark 的设计比普通问答更难。

关键是要让分数反映真实泛化能力。

### 9.1 难度分层

评估集应有难度分层。

例如数学题：

1. 基础算术。
2. 代数变形。
3. 几何推理。
4. 组合计数。
5. 竞赛难题。
6. 证明题。

代码题：

1. 字符串处理。
2. 数组哈希。
3. 动态规划。
4. 图算法。
5. 并发或系统设计。
6. 大型代码库修改。

只报告总分会掩盖能力结构。

### 9.2 泛化测试

好的 benchmark 不应只包含经典题。

可以加入：

1. 新生成题。
2. 参数化变体。
3. 反模板题。
4. 对抗样本。
5. 时间切分后的新题。
6. 私有 holdout。

### 9.3 防止模板捷径

如果题目由固定模板生成，模型可能学到模板捷径。

例如：

```text
Alice has N apples and gives M to Bob. How many are left?
```

模型不一定真正推理，只是套公式。

需要设计更丰富的语言、结构和干扰信息。

### 9.4 报告子指标

建议报告：

1. Overall accuracy。
2. 按难度分层的 accuracy。
3. 按题型分层的 accuracy。
4. pass@k。
5. self-consistency accuracy。
6. 过程错误率。
7. 格式错误率。
8. 执行错误率。
9. 污染风险分析。

面试表达：Reasoning benchmark 要有难度分层、隐藏测试、泛化样本和错误类型分析，只报告 overall score 很容易误导。

## 10. 常见错误类型

做 reasoning/code/math eval 时，要把错误拆开看。

### 10.1 数学错误类型

常见数学错误：

1. 题意理解错误。
2. 变量定义错误。
3. 公式选择错误。
4. 代数变形错误。
5. 算术计算错误。
6. 单位错误。
7. 边界条件遗漏。
8. 最终答案格式错误。

### 10.2 逻辑推理错误类型

常见逻辑错误：

1. 混淆充分条件和必要条件。
2. 忽略否定词。
3. 引入题目没有给出的假设。
4. 多跳过程中丢失约束。
5. 把相关性当因果性。

### 10.3 代码错误类型

常见代码错误：

1. 语法错误。
2. 运行时错误。
3. 输入输出格式错误。
4. 边界条件错误。
5. 算法复杂度不达标。
6. 状态污染。
7. 并发或异步错误。
8. 依赖和环境错误。
9. 安全风险。

### 10.4 评估系统错误

有时不是模型错，而是评估系统错。

例如：

1. 标准答案错误。
2. 测试用例错误。
3. 答案解析失败。
4. 沙箱环境不一致。
5. 超时设置不合理。
6. judge prompt 有偏差。

面试表达：高质量 eval 不只输出分数，还要输出错误分类，因为错误分类决定下一步是改数据、改模型、改 prompt、改 verifier 还是改测试集。

## 11. 真实项目中的评估流程

一个相对完整的 reasoning/code/math eval 流程如下。

### 11.1 离线评估

离线阶段：

1. 准备公开 benchmark。
2. 准备内部私有 benchmark。
3. 做污染检测。
4. 固定 decoding 参数。
5. 运行多次采样。
6. 计算主指标和子指标。
7. 做错误分类。
8. 抽样人工复核。

### 11.2 回归测试

每次模型、prompt 或系统改动后，跑固定回归集。

重点看：

1. 总分是否下降。
2. 哪些题型下降。
3. 之前修复的 bad case 是否复发。
4. 格式错误是否增加。
5. 成本和延迟是否变化。

### 11.3 线上评估

线上不能直接依赖 benchmark。

可以看：

1. 用户任务成功率。
2. 代码执行成功率。
3. 用户重试率。
4. 人工接管率。
5. 投诉和负反馈。
6. 长链路 Agent 成功率。

### 11.4 闭环改进

评估不是一次性动作。

它应该形成闭环：

```text
评估 -> 错误分析 -> 数据或方法改进 -> 回归测试 -> 新评估
```

面试表达：真实项目里，我会把公开 benchmark、私有 holdout、回归集、错误分析和线上指标结合起来，而不是只看单一排行榜分数。

## 12. 最小 reasoning / code / math eval 审计 demo

下面这个 demo 不调用模型、不执行外部代码，只用 toy 输出审计一组 reasoning / code / math 评估结果。它覆盖答案抽取与归一化、greedy accuracy、self-consistency 投票、answer entropy、过程分、代码候选的 `pass@k`、公开测试和隐藏测试缺口、污染风险以及最终门禁。

```python
from collections import Counter
from fractions import Fraction
from math import comb, log
from pprint import pprint
import re


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def extract_number(text):
    lowered = text.lower()
    frac = re.findall(r"-?\d+\s*/\s*-?\d+", lowered)
    if frac:
        return Fraction(frac[-1].replace(" ", ""))
    pct = re.findall(r"-?\d+(?:\.\d+)?\s*%", lowered)
    if pct:
        value = float(pct[-1].replace("%", ""))
        return Fraction(value / 100).limit_denominator(1000)
    nums = re.findall(r"-?\d+(?:\.\d+)?", lowered)
    if nums:
        return Fraction(float(nums[-1])).limit_denominator(1000)
    return None


def equivalent(pred, gold, tolerance=1e-3):
    pred_value = extract_number(pred)
    gold_value = extract_number(gold)
    if pred_value is None or gold_value is None:
        return False
    return abs(float(pred_value - gold_value)) <= tolerance


def canonical(text):
    value = extract_number(text)
    if value is None:
        return "unparsed"
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * log(count / total) for count in counts.values())


math_cases = [
    {
        "id": "math_frac",
        "type": "fraction",
        "gold": "2/3",
        "greedy": "The answer is 0.6666667.",
        "samples": ["2/3", "0.6666667", "66.7%", "1/2", "2/3"],
        "process_steps": [True, True, False],
        "contamination_risk": False,
    },
    {
        "id": "math_algebra",
        "type": "algebra",
        "gold": "7",
        "greedy": "x = 7",
        "samples": ["7", "7", "8", "7", "no final answer"],
        "process_steps": [True, True, True],
        "contamination_risk": False,
    },
    {
        "id": "math_distractor",
        "type": "distractor",
        "gold": "18",
        "greedy": "The final answer is 20.",
        "samples": ["20", "18", "20", "18", "20"],
        "process_steps": [True, False, False],
        "contamination_risk": True,
    },
]

math_reports = []
for case in math_cases:
    sample_labels = [canonical(sample) for sample in case["samples"]]
    counts = Counter(sample_labels)
    top_two = counts.most_common(2)
    top_label, top_count = top_two[0]
    second_count = top_two[1][1] if len(top_two) > 1 else 0
    vote_margin = (top_count - second_count) / len(sample_labels)
    math_reports.append(
        {
            "id": case["id"],
            "greedy_correct": equivalent(case["greedy"], case["gold"]),
            "sc_answer": top_label,
            "sc_correct": equivalent(top_label, case["gold"]),
            "entropy": round(entropy(sample_labels), 3),
            "vote_margin": round(vote_margin, 3),
            "process_score": round(sum(case["process_steps"]) / len(case["process_steps"]), 3),
            "parse_failed": canonical(case["greedy"]) == "unparsed",
            "contamination_risk": case["contamination_risk"],
        }
    )

code_tasks = [
    {
        "id": "reverse_words",
        "n": 10,
        "correct": 3,
        "public_pass": 4,
        "hidden_pass": 3,
        "tests": {"public": 3, "hidden": 8},
        "error": "edge_case",
    },
    {
        "id": "two_sum",
        "n": 10,
        "correct": 1,
        "public_pass": 3,
        "hidden_pass": 1,
        "tests": {"public": 2, "hidden": 10},
        "error": "complexity",
    },
    {
        "id": "parse_table",
        "n": 10,
        "correct": 0,
        "public_pass": 2,
        "hidden_pass": 0,
        "tests": {"public": 2, "hidden": 6},
        "error": "runtime",
    },
]

code_reports = []
for task in code_tasks:
    code_reports.append(
        {
            "id": task["id"],
            "pass@1": round(pass_at_k(task["n"], task["correct"], 1), 3),
            "pass@5": round(pass_at_k(task["n"], task["correct"], 5), 3),
            "public_hidden_gap": task["public_pass"] - task["hidden_pass"],
            "hidden_tests": task["tests"]["hidden"],
            "error": task["error"],
        }
    )

greedy_accuracy = sum(report["greedy_correct"] for report in math_reports) / len(math_reports)
sc_accuracy = sum(report["sc_correct"] for report in math_reports) / len(math_reports)
parse_fail_rate = sum(report["parse_failed"] for report in math_reports) / len(math_reports)
avg_entropy = sum(report["entropy"] for report in math_reports) / len(math_reports)
avg_vote_margin = sum(report["vote_margin"] for report in math_reports) / len(math_reports)
avg_process_score = sum(report["process_score"] for report in math_reports) / len(math_reports)
avg_pass1 = sum(report["pass@1"] for report in code_reports) / len(code_reports)
avg_pass5 = sum(report["pass@5"] for report in code_reports) / len(code_reports)
contamination_flags = [
    report["id"]
    for report in math_reports
    if report["contamination_risk"]
]
weak_hidden_tests = [
    report["id"]
    for report in code_reports
    if report["hidden_tests"] < 8
]
public_hidden_gaps = [
    (report["id"], report["public_hidden_gap"])
    for report in code_reports
    if report["public_hidden_gap"] >= 2
]
error_counts = Counter(report["error"] for report in code_reports)

gates = {
    "answer_accuracy": greedy_accuracy >= 0.60,
    "parse": parse_fail_rate <= 0.05,
    "self_consistency": sc_accuracy >= greedy_accuracy,
    "code_pass1": avg_pass1 >= 0.20,
    "hidden_tests": not weak_hidden_tests,
    "process": avg_process_score >= 0.70,
    "contamination": not contamination_flags,
}

summary = {
    "math_summary": {
        "greedy_accuracy": round(greedy_accuracy, 3),
        "self_consistency_accuracy": round(sc_accuracy, 3),
        "parse_fail_rate": round(parse_fail_rate, 3),
        "avg_answer_entropy": round(avg_entropy, 3),
        "avg_vote_margin": round(avg_vote_margin, 3),
        "avg_process_score": round(avg_process_score, 3),
    },
    "math_reports": math_reports,
    "code_reports": code_reports,
    "code_summary": {
        "avg_pass@1": round(avg_pass1, 3),
        "avg_pass@5": round(avg_pass5, 3),
        "weak_hidden_tests": weak_hidden_tests,
        "public_hidden_gaps": public_hidden_gaps,
        "error_counts": dict(error_counts),
    },
    "contamination_flags": contamination_flags,
    "gates": gates,
    "gate_pass": all(gates.values()),
}

pprint(summary, sort_dicts=False)
```

一组可复现输出如下：

```text
{'math_summary': {'greedy_accuracy': 0.667,
                  'self_consistency_accuracy': 0.667,
                  'parse_fail_rate': 0.0,
                  'avg_answer_entropy': 0.858,
                  'avg_vote_margin': 0.333,
                  'avg_process_score': 0.667},
 'math_reports': [{'id': 'math_frac',
                   'greedy_correct': True,
                   'sc_answer': '2/3',
                   'sc_correct': True,
                   'entropy': 0.95,
                   'vote_margin': 0.4,
                   'process_score': 0.667,
                   'parse_failed': False,
                   'contamination_risk': False},
                  {'id': 'math_algebra',
                   'greedy_correct': True,
                   'sc_answer': '7',
                   'sc_correct': True,
                   'entropy': 0.95,
                   'vote_margin': 0.4,
                   'process_score': 1.0,
                   'parse_failed': False,
                   'contamination_risk': False},
                  {'id': 'math_distractor',
                   'greedy_correct': False,
                   'sc_answer': '20',
                   'sc_correct': False,
                   'entropy': 0.673,
                   'vote_margin': 0.2,
                   'process_score': 0.333,
                   'parse_failed': False,
                   'contamination_risk': True}],
 'code_reports': [{'id': 'reverse_words',
                   'pass@1': 0.3,
                   'pass@5': 0.917,
                   'public_hidden_gap': 1,
                   'hidden_tests': 8,
                   'error': 'edge_case'},
                  {'id': 'two_sum',
                   'pass@1': 0.1,
                   'pass@5': 0.5,
                   'public_hidden_gap': 2,
                   'hidden_tests': 10,
                   'error': 'complexity'},
                  {'id': 'parse_table',
                   'pass@1': 0.0,
                   'pass@5': 0.0,
                   'public_hidden_gap': 2,
                   'hidden_tests': 6,
                   'error': 'runtime'}],
 'code_summary': {'avg_pass@1': 0.133,
                  'avg_pass@5': 0.472,
                  'weak_hidden_tests': ['parse_table'],
                  'public_hidden_gaps': [('two_sum', 2), ('parse_table', 2)],
                  'error_counts': {'edge_case': 1, 'complexity': 1, 'runtime': 1}},
 'contamination_flags': ['math_distractor'],
 'gates': {'answer_accuracy': True,
           'parse': True,
           'self_consistency': True,
           'code_pass1': False,
           'hidden_tests': False,
           'process': False,
           'contamination': False},
 'gate_pass': False}
```

这个输出体现了本章的核心判断：总 accuracy 过线并不代表评估可信。代码单次通过率低、隐藏测试不足、过程分偏低、公开测试和隐藏测试存在缺口，并且还有污染风险样本，所以不能直接用总分做上线结论。

## 13. 面试官会怎么问

### 13.1 如何评估一个数学推理模型？

回答要点：

1. 先明确任务类型和难度分层。
2. 用最终答案 accuracy 做基础指标。
3. 做答案抽取和等价归一化。
4. 对复杂题加入 step-level 或 proof-level 检查。
5. 用 self-consistency 分析多采样稳定性。
6. 做污染检测和私有 holdout。
7. 输出错误类型分析。

### 13.2 为什么代码生成评估常用 pass@k？

回答要点：

1. 代码生成有随机性。
2. 真实系统可以多次采样或重试。
3. pass@k 衡量 k 个候选中至少一个通过测试的概率。
4. 但必须报告采样数量、temperature、测试强度和 k。
5. pass@k 高不代表单次输出稳定。

### 13.3 如何避免 reasoning benchmark 被刷分？

回答要点：

1. 做训练数据和测试集污染检测。
2. 使用时间切分和私有 holdout。
3. 加入动态生成题和变体题。
4. 报告子任务和错误类型，而不是只看总分。
5. 控制 prompt 调参对测试集的过拟合。
6. 定期更新评估集。

### 13.4 Verifier 有什么价值？

回答要点：

1. Verifier 把生成和验证分开。
2. 它可以用于候选筛选、错误定位和训练信号。
3. 代码中单元测试是强 verifier。
4. 数学中可以用规则、符号计算、人审和 LLM verifier。
5. LLM verifier 自身也要校准。

## 14. 标准回答模板

如果面试官问：“你会如何设计 reasoning/code/math eval？”

可以这样回答：

```text
我会先区分任务类型。数学题可以以最终答案 accuracy 为主，但要做答案抽取、格式归一化和等价判断；复杂题还要抽样做过程评估。代码题我会用单元测试和隐藏测试做可执行验证，并报告 pass@1、pass@k、运行错误率和超时率。推理任务还要看 self-consistency、多采样稳定性和错误类型。

同时我不会只看总分。我会按难度、题型和错误类别拆分结果，检查 benchmark contamination，用私有 holdout 或时间切分样本验证泛化。对于开放或复杂过程，可以引入 verifier 或 LLM judge，但需要用人工 gold set 校准。最终目标不是刷 benchmark，而是判断模型是否真的具备可泛化的问题求解能力。
```

## 15. 常见误区

### 15.1 只看排行榜分数

排行榜只能说明模型在某些公开 benchmark 上表现好。

它不能直接证明真实业务任务也好。

### 15.2 把 CoT 写得长当作推理强

长推理不一定正确。

有些模型会生成看似合理但实际错误的解释。

### 15.3 忽略答案解析

很多数学评估误差来自答案抽取失败，而不是模型能力。

### 15.4 隐藏测试太弱

代码题如果测试太弱，模型会通过错误解法。

### 15.5 不报告 decoding 参数

`temperature`、`top_p`、采样次数和最大 token 都会影响结果。

不报告这些参数，结果不可复现。

### 15.6 用 LLM judge 直接替代验证

对于代码和数学，能用规则或执行验证时，应优先用规则或执行验证。

LLM judge 更适合辅助开放式过程分析。

## 16. 小练习

### 练习 1

给定一个数学 benchmark，你会如何处理这些答案等价问题？

```text
Gold: 2/3
Pred A: 0.6666667
Pred B: \frac{2}{3}
Pred C: 66.7%
```

要求说明哪些可以判对，取决于什么条件。

### 练习 2

一个代码模型在 `pass@1` 上很低，但 `pass@10` 很高。这说明什么？

思考方向：

1. 单次生成稳定性。
2. 多采样搜索能力。
3. verifier 或测试筛选是否能利用多个候选。
4. 成本是否可接受。

### 练习 3

设计一个错误分类表，用于分析数学推理模型失败原因。

至少包含：题意理解、公式选择、推导、计算、格式、污染风险。

### 练习 4

如果一个模型在公开数学 benchmark 上很强，但在你新写的变体题上明显下降，你会如何排查？

参考方向：

1. benchmark contamination。
2. 模板过拟合。
3. 题目难度变化。
4. 答案解析差异。
5. prompt 和 decoding 参数。

## 17. 本章总结

Reasoning、Code、Math eval 的关键不是把所有任务都压成一个 accuracy。

数学任务要处理答案等价、过程正确性和污染风险。

代码任务要依赖单元测试、隐藏测试、沙箱执行和 `pass@k`。

推理任务要区分最终答案和推理过程，并用 self-consistency、verifier 和错误分析理解模型能力。

面试中要强调：可信评估不是追求更复杂的指标，而是让指标真正对应你关心的能力。

如果评估目标是单次用户回答，就重点看 pass@1、格式稳定性和错误率。

如果评估目标是 Agent 多次尝试解决任务，就可以看 pass@k、verifier 选择能力和端到端任务成功率。

如果评估目标是研究 reasoning 能力，就要看难度分层、过程评估、泛化题和污染检测。

最终，好的评估应该能回答三个问题：模型会不会做，为什么会错，下一步该怎么改。
