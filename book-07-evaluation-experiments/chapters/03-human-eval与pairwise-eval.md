# 第三章：Human Eval 与 Pairwise Eval

重点：人工标注规范、pairwise comparison、Elo、Arena、标注一致性、偏见控制。

面试重点：开放式生成任务往往比选择题更需要人工评估。

## 本章目标

学完本章，你要能回答：

1. 为什么开放式生成任务需要 human eval？
2. 绝对打分和 pairwise comparison 有什么区别？
3. 如何设计人工标注规范和 rubric？
4. Elo、Arena 这类成对比较系统如何工作？
5. 如何控制标注一致性、偏见和成本？

Human eval 是大模型评估里最重要也最容易被低估的部分。

很多开放式任务没有唯一标准答案，自动指标很难判断“哪个回答更有用”。

这时需要人类评审。

## 本章资料边界

本章第二轮精修参考了 [InstructGPT](https://arxiv.org/abs/2203.02155)、[Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)、[HELM / Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)、[MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)、[Chatbot Arena](https://arxiv.org/abs/2403.04132) 和 [OpenAI Evals](https://github.com/openai/evals) 的资料边界。

本章聚焦人工评估与成对偏好评估的面试和工程口径：标注协议、rubric、绝对评分、pairwise comparison、win rate、Bradley-Terry / Elo 直觉、Arena 排名、一致性、偏见控制、成本和上线门禁。不展开 RLHF 训练算法、完整标注平台、众包治理、复杂 IRT 模型或在线实验统计推导。

## 本章核心公式

一个 human eval 样本可以抽象成：

```math
h_i=(x_i,a_i,b_i,z_i)
```

其中 `x_i` 是输入 prompt，`a_i` 和 `b_i` 是两个匿名候选回答，`z_i` 是任务、语言、风险、难度、来源、长度和版本等元数据。

绝对打分里，标注员 `j` 对样本 `i` 的某个维度给分 `s_{ij}`。带权平均分可以写成：

```math
S_{\mathrm{abs}}
=
\frac{\sum_i\sum_j w_i s_{ij}}
{\sum_i\sum_j w_i}
```

其中 `w_i` 是样本权重。这个分数适合描述总体质量，但会受标注员尺度漂移影响。

Pairwise eval 中，可以把新模型相对旧模型的单次比较结果写成：

```math
o_i\in\{1,0,-1\}
```

`1` 表示新模型赢，`0` 表示 tie，`-1` 表示旧模型赢。把 tie 记半分时，pairwise win rate 为：

```math
\hat{p}_{\mathrm{win}}
=
\frac{N_{\mathrm{win}}+0.5N_{\mathrm{tie}}}{N}
```

粗略置信区间可以用二项近似估计：

```math
\mathrm{SE}
=
\sqrt{
\frac{\hat{p}_{\mathrm{win}}(1-\hat{p}_{\mathrm{win}})}{N}
}
```

```math
\mathrm{CI}_{95}
\approx
\hat{p}_{\mathrm{win}}\pm 1.96\mathrm{SE}
```

样本数较小、tie 很多或分层抽样很复杂时，最好用 bootstrap 或分层统计，而不是只套这个近似公式。

Bradley-Terry 模型把两个模型的相对强弱写成：

```math
P(A\succ B)
=
\frac{\exp(r_A)}
{\exp(r_A)+\exp(r_B)}
=
\frac{1}{1+\exp(-(r_A-r_B))}
```

其中 `r_A` 和 `r_B` 是两个模型的 latent rating。这个公式解释了为什么大量 pairwise 结果可以聚合成相对排名。

两名标注员的一致率可以先用：

```math
A_{\mathrm{agree}}
=
\frac{1}{N}
\sum_{i=1}^{N}
I(l_i^{(1)}=l_i^{(2)})
```

如果要扣除随机一致，可以用 Cohen kappa 的口径：

```math
\kappa
=
\frac{p_o-p_e}{1-p_e}
```

其中 `p_o` 是观测一致率，`p_e` 是按边际分布估计的随机一致率。

位置偏见可以用左侧答案胜出比例偏离 0.5 的幅度粗略监控：

```math
B_{\mathrm{pos}}
=
\left|
\frac{N_{\mathrm{left\_win}}}
{N_{\mathrm{left\_win}}+N_{\mathrm{right\_win}}}
-0.5
\right|
```

Human eval 上线门禁可以写成：

```math
G_{\mathrm{human}}
=
G_{\mathrm{rubric}}
\land G_{\mathrm{blind}}
\land G_{\mathrm{agree}}
\land G_{\mathrm{win}}
\land G_{\mathrm{slice}}
\land G_{\mathrm{bias}}
\land G_{\mathrm{risk}}
```

也就是 rubric 清楚、匿名化、标注一致性达标、总体胜率达标、关键切片不回退、偏见可控、高风险样本不出问题。

## 1. 为什么需要 Human Eval

大模型输出通常是开放式的。

例如：

```text
问题：给我解释一下 Transformer 中的 attention，并举一个直观例子。
```

模型 A 可能回答得短而准确。

模型 B 可能回答得长而全面。

模型 C 可能公式正确但不适合初学者。

这三个回答很难用 exact match 或 accuracy 评价。

Human eval 的价值在于：

1. 判断开放式答案质量。
2. 比较多个可接受答案。
3. 发现自动指标看不出的缺陷。
4. 评估用户感知质量。
5. 建立 gold set 校准自动评测。
6. 为 LLM-as-a-Judge 提供校准基准。

面试表达：开放式生成任务往往没有唯一答案，human eval 能评估 correctness、helpfulness、faithfulness、style 和 user preference，是自动指标的重要补充。

## 2. Human Eval 的常见形式

Human eval 不是一种固定方法。

常见形式包括：

1. 绝对打分。
2. Pairwise comparison。
3. 排序评估。
4. 错误类型标注。
5. 是否通过任务验收。
6. 专家审查。
7. 用户反馈。

不同形式适合不同问题。

### 2.1 绝对打分

绝对打分是让标注员给一个回答打分。

例如 1 到 5 分。

优点：

1. 结果直观。
2. 可以按维度评分。
3. 易于聚合。

缺点：

1. 标注员尺度不一致。
2. 评分标准容易漂移。
3. 细小差异不容易判断。

### 2.2 Pairwise Comparison

Pairwise comparison 是让标注员在两个回答中选更好的一方。

例如：

```text
给定同一个问题和两个匿名回答 A/B，选择哪个回答更好，或选择 tie。
```

优点：

1. 比绝对打分更容易。
2. 标注员尺度差异更小。
3. 适合比较两个模型版本。
4. 能转化为 win rate、Elo 或 Bradley-Terry 分数。

缺点：

1. 标注量随模型数量增长。
2. 只能比较相对优劣。
3. tie 规则会影响结果。

### 2.3 排序评估

排序评估是让标注员对多个回答排序。

优点是一次能比较多个模型。

缺点是认知负担更高。

当候选回答超过 3 个时，排序质量通常会下降。

### 2.4 错误类型标注

错误类型标注不是问哪个更好，而是问错在哪里。

常见错误类型包括：

1. 事实错误。
2. 幻觉。
3. 答非所问。
4. 格式错误。
5. 引用错误。
6. 安全违规。
7. 过度拒答。
8. 逻辑错误。
9. 工具调用错误。

这种方式适合 error analysis 和回归测试。

面试表达：Human eval 可以做绝对打分、成对比较、排序和错误归因；选择哪种方式取决于评估目标。

## 3. Absolute Rating

绝对打分适合需要多维质量画像的场景。

例如评估一个客服回答，可以从这些维度打分：

1. 正确性。
2. 完整性。
3. 有用性。
4. 简洁性。
5. 语气。
6. 事实依据。
7. 安全性。

### 3.1 Rubric 示例

以 5 分制为例：

```text
5 分：完全正确，解决用户问题，表达清晰，无额外风险。
4 分：基本正确，有轻微遗漏，但不影响主要结论。
3 分：部分正确，但遗漏关键条件或解释不够清楚。
2 分：存在明显错误，用户可能被误导。
1 分：主要错误或答非所问。
0 分：安全违规、严重幻觉或完全不可用。
```

### 3.2 多维评分

有些任务不适合一个总分。

可以拆成多个维度：

```text
Correctness：0-5
Helpfulness：0-5
Faithfulness：0-5
Style：0-5
Safety：pass/fail
```

多维评分的好处是能解释问题。

例如模型总体分高，但 faithfulness 低，说明它可能回答流畅但不够基于证据。

### 3.3 绝对打分的风险

主要风险是标注尺度漂移。

不同标注员对 4 分和 5 分的理解可能不同。

同一个标注员在不同时间也可能变严或变松。

解决方法包括：

1. 给出正反例。
2. 做标注员培训。
3. 插入校准样本。
4. 多人标注取平均。
5. 定期检查评分分布。

面试表达：绝对打分适合多维质量分析，但要用清晰 rubric、示例和校准样本控制评分尺度漂移。

## 4. Pairwise Comparison

Pairwise comparison 是大模型评估中非常常用的方法。

它回答的问题是：同一个输入下，回答 A 和回答 B 哪个更好？

### 4.1 基本流程

流程如下：

```text
收集 prompts
  -> 用模型 A 和模型 B 生成回答
  -> 随机打乱回答顺序
  -> 标注员比较 A/B
  -> 记录 win / lose / tie
  -> 聚合 win rate
```

### 4.2 为什么 pairwise 更稳定

相比绝对打分，pairwise 的判断更简单。

人类更擅长回答“哪个更好”，而不是回答“这个值几分”。

例如两个答案都不错时，标注员可能难以决定给 4 分还是 5 分，但通常能判断哪个更适合用户。

### 4.3 Win Rate

最简单的结果是 win rate。

如果模型 B 对模型 A：

```text
win = 420
lose = 350
tie = 230
```

一种常见计算方式是：

```math
\hat{p}_{\mathrm{win}}
=
\frac{420+0.5\times 230}{1000}
=
0.535
```

如果置信区间显示显著高于 50%，可以认为 B 优于 A。

### 4.4 Tie 规则

Tie 很重要。

如果强迫标注员必须二选一，会增加噪声。

但如果 tie 选项过宽，又会降低区分度。

通常需要明确：

1. 两个回答都正确且差异很小时选 tie。
2. 一个更完整或更有用时不要选 tie。
3. 一个有事实错误时不能因为更长而选 tie。

面试表达：pairwise eval 更适合模型版本比较，常用 win rate 聚合，但要处理匿名化、顺序随机、tie 规则和统计显著性。

## 5. Elo 与 Arena

当模型数量很多时，两两比较会变复杂。

这时可以用 Elo 或 Arena 系统。

### 5.1 Elo 的直觉

Elo 最初来自棋类评分。

如果高分模型赢了低分模型，分数变化小。

如果低分模型赢了高分模型，分数变化大。

它把很多 pairwise 比较聚合成一个相对排名。

一个常见写法是先用两个 rating 估计 A 赢 B 的概率：

```math
E_A
=
\frac{1}{1+10^{(R_B-R_A)/400}}
```

如果真实结果是 `S_A`，例如赢为 1、平为 0.5、输为 0，则 rating 更新可以写成：

```math
R_A'
=
R_A+K(S_A-E_A)
```

这解释了为什么“低分模型爆冷赢高分模型”会带来更大的分数变化。

### 5.2 Bradley-Terry 的排序直觉

在大模型 Arena 排名里，Elo 不是唯一选择。

很多系统会把 pairwise 结果拟合成 Bradley-Terry 这类偏好模型：

```math
P(m_i\succ m_j)
=
\frac{1}{1+\exp(-(r_i-r_j))}
```

这里 `r_i` 和 `r_j` 是模型的隐含强度分数。与单次 Elo 更新相比，Bradley-Terry 更像是用全量对战记录一起估计一组相对分数。

面试中不需要把优化过程推完，但要能说明：pairwise 数据并不只能算简单 win rate，也可以进一步拟合相对 ranking；ranking 的可靠性取决于样本量、对战覆盖、prompt 分布和偏见控制。

### 5.3 Arena 的基本思想

Arena 通常让用户或标注员在两个匿名模型输出中选择更好回答。

系统收集大量对战结果，再估计模型排名。

典型流程：

```text
用户输入 prompt
  -> 随机选择两个模型
  -> 生成匿名回答
  -> 用户选择 A / B / tie
  -> 更新模型相对分数
```

### 5.4 Arena 的优点

1. 可以持续收集真实偏好。
2. 支持多个模型排名。
3. 更接近开放式用户体验。
4. 能发现 leaderboard 难覆盖的样本。

### 5.5 Arena 的问题

1. 用户 prompt 分布不可控。
2. 评审标准不稳定。
3. 容易受回答长度影响。
4. 不同模型被比较的样本不完全相同。
5. 不一定覆盖安全和专业场景。

面试表达：Elo/Arena 能把大量 pairwise 偏好聚合成模型相对排名，但排名受流量分布、评审偏差、样本覆盖和统计不确定性影响。

## 6. 标注规范设计

Human eval 的质量高度依赖标注规范。

好的标注规范应该让不同标注员在同一个样本上尽量给出一致判断。

### 6.1 标注规范包含什么

至少包括：

1. 任务说明。
2. 输入字段解释。
3. 输出字段解释。
4. 评分维度。
5. 正例和反例。
6. 边界案例规则。
7. tie 规则。
8. 安全优先级。
9. 是否允许使用外部知识。
10. 标注界面操作说明。

### 6.2 优先级规则

当多个维度冲突时，要有优先级。

例如：

```text
安全违规 > 事实错误 > 未遵循指令 > 不完整 > 风格问题
```

如果一个回答很流畅但包含严重事实错误，不能因为表达好而给高分。

如果一个回答有安全违规，通常直接判 fail。

### 6.3 边界案例

边界案例要提前写清楚。

例如：

1. 两个回答都正确，但一个更简洁，如何选？
2. 一个回答更长但包含无关内容，如何选？
3. 一个回答拒答但其实可以回答，如何判？
4. 一个回答引用正确但措辞不友好，如何判？
5. 一个回答部分正确但缺少关键风险提示，如何判？

面试表达：标注规范要包括任务说明、rubric、正反例、边界规则和维度优先级，否则 human eval 会变成主观投票。

## 7. 标注一致性

Human eval 最大的问题之一是标注一致性。

### 7.1 为什么一致性重要

如果不同标注员对同一回答判断完全不同，说明评估不稳定。

这种结果不能作为上线依据。

### 7.2 一致性指标

常见方法包括：

1. agreement rate。
2. Cohen's kappa。
3. Fleiss' kappa。
4. Krippendorff's alpha。
5. pairwise agreement。

面试中不一定要推公式，但要知道它们用于衡量标注员一致性。

最小可解释口径是简单一致率：

```math
A_{\mathrm{agree}}
=
\frac{N_{\mathrm{same}}}{N_{\mathrm{double}}}
```

如果两个标注员都大量选择同一个类别，简单一致率可能虚高。Cohen kappa 会扣除随机一致：

```math
\kappa
=
\frac{p_o-p_e}{1-p_e}
```

这里 `p_o` 是观测一致率，`p_e` 是按两个标注员各自标签分布估计的期望随机一致率。面试里讲到这一层，已经能说明你知道 human eval 不是简单多数投票。

### 7.3 提升一致性的方法

1. 标注前培训。
2. 提供 gold examples。
3. 小批量试标。
4. 发现分歧后更新规范。
5. 多人标注。
6. 仲裁机制。
7. 定期插入校准样本。

### 7.4 仲裁机制

如果两个标注员分歧，可以让第三个更资深标注员仲裁。

对于高风险样本，最好由领域专家或安全专家最终确认。

面试表达：Human eval 要报告标注一致性；如果一致性低，要先修 rubric 和培训流程，而不是直接相信分数。

## 8. 偏见控制

人工评估并不天然客观。

常见偏见包括：

1. 长度偏见。
2. 位置偏见。
3. 格式偏见。
4. 品牌偏见。
5. 熟悉度偏见。
6. 语言偏见。
7. 过度偏好自信语气。

### 8.1 长度偏见

更长的回答看起来更努力，但不一定更好。

标注规范要提醒：长不等于正确，完整不等于啰嗦。

### 8.2 位置偏见

在 A/B 比较中，标注员可能更常选左边或第一个。

解决方法是随机打乱回答顺序，并记录位置。

### 8.3 品牌偏见

不能让标注员看到模型名字。

否则可能因为品牌印象影响判断。

### 8.4 格式偏见

列表、加粗、结构化回答可能更讨喜。

但如果内容错误，格式好不能弥补事实错误。

### 8.5 偏见监控指标

偏见控制不能只靠口头提醒，最好在结果表里留下可审计字段。

例如位置偏见：

```math
B_{\mathrm{pos}}
=
\left|
\frac{N_{\mathrm{left\_win}}}
{N_{\mathrm{left\_win}}+N_{\mathrm{right\_win}}}
-0.5
\right|
```

例如长度偏见，可以记录更长回答胜出的比例：

```math
B_{\mathrm{len}}
=
\frac{N_{\mathrm{longer\_win}}}{N_{\mathrm{non\_tie}}}
```

如果 `B_len` 长期明显高于 0.5，要检查 rubric 是否把“完整”和“啰嗦”混在一起，也要检查回答长度是否泄露了模型身份。

面试表达：Human eval 要匿名化模型、随机化顺序、控制长度和格式偏见，并用清晰 rubric 强调内容优先。

## 9. 成本和效率

Human eval 成本高。

要设计抽样和流程。

### 9.1 降低成本的方法

1. 先用自动评估粗筛。
2. 只人工评估关键样本。
3. 对高分歧样本重点复审。
4. 使用 pairwise 代替复杂多维评分。
5. 对稳定指标用程序化测试。
6. 用 LLM judge 做辅助，但保留人工 gold set。

### 9.2 样本量设计

样本量太小，结果不稳定。

样本量太大，成本过高。

需要结合：

1. 预期提升幅度。
2. 结果方差。
3. 统计显著性要求。
4. 任务切片数量。
5. 标注预算。

如果要比较两个模型 1% 的小提升，需要比比较 10% 的大提升更多样本。

面试表达：Human eval 要在可信度和成本之间权衡，通常用自动评估粗筛、人工评估关键切片、gold set 校准 judge。

## 10. Human Eval Pipeline

一个生产化 human eval pipeline 可以这样设计：

```text
Sample Selection
  -> Model Generation
  -> Anonymization
  -> Annotation Assignment
  -> Quality Control
  -> Aggregation
  -> Slice Analysis
  -> Report / Decision
```

### 10.1 Sample Selection

选择代表性样本、困难样本、bad case 和高风险样本。

### 10.2 Model Generation

固定 prompt、模型版本、解码参数和工具环境。

### 10.3 Anonymization

隐藏模型名称、版本和输出顺序。

### 10.4 Annotation Assignment

给每个样本分配多个标注员。

对专业任务分配领域专家。

### 10.5 Quality Control

插入 gold samples、重复样本和一致性检查。

### 10.6 Aggregation

聚合分数、win rate、tie rate 和置信区间。

### 10.7 Slice Analysis

按任务、难度、语言、长度、安全类别和用户场景切片。

### 10.8 Report / Decision

输出是否上线、是否灰度、哪些场景回退、哪些问题需要修复。

面试表达：Human eval 要平台化，包含采样、生成、匿名化、分配、质控、聚合、切片和决策报告。

## 11. Pairwise Eval 示例

假设要比较新旧两个客服模型。

可以这样设计：

### 11.1 样本

1. 500 个真实线上问题。
2. 200 个历史 bad case。
3. 100 个政策边界问题。
4. 100 个无答案应拒答问题。
5. 100 个高价值用户问题。

### 11.2 标注方式

每个样本展示同一问题下两个匿名回答。

标注员选择：

1. A 明显更好。
2. A 略好。
3. Tie。
4. B 略好。
5. B 明显更好。

### 11.3 判断维度

优先级：

```text
安全性 > 正确性 > 基于证据 > 完整性 > 简洁性 > 语气
```

### 11.4 上线门禁

```text
总体 win rate >= 53%，且置信区间下界 > 50%。
P0/P1 bad case 不允许回退。
无答案问题误答率不能上升。
政策边界问题不能下降。
人工分歧样本必须复审。
```

这种设计比只看“平均分提升”更可靠。

可以把门禁写成：

```math
G_{\mathrm{release}}
=
I(\hat{p}_{\mathrm{win}}\ge 0.53)
\land I(\mathrm{CI}_{\mathrm{low}}>0.50)
\land G_{\mathrm{p0}}
\land G_{\mathrm{risk}}
\land G_{\mathrm{review}}
```

其中 `CI_low` 是胜率置信区间下界，`G_p0` 是 P0 / P1 bad case 不回退，`G_risk` 是安全和政策边界不下降，`G_review` 是高分歧样本已复审。

## 12. 最小 Human Eval 审计 demo

下面这个 demo 不调用模型，只审计一个 toy human eval 结果是否足以上线：它同时检查绝对分、pairwise win rate、切片胜率、标注一致性、位置偏见、长度偏见、高风险样本和评估协议完整性。

```python
from pprint import pprint


rubric = {
    "dimensions": ["safety", "correctness", "evidence", "helpfulness", "style"],
    "priority_order": ["safety", "correctness", "evidence", "helpfulness", "style"],
    "tie_rule": True,
    "anonymized": True,
    "position_randomized": True,
    "min_annotators": 2,
    "gold_samples": 2,
}

items = {
    "qa_001": {"task": "qa", "risk": "normal", "old_len": 90, "new_len": 96},
    "rag_001": {"task": "rag", "risk": "high", "old_len": 130, "new_len": 150},
    "code_001": {"task": "code", "risk": "normal", "old_len": 100, "new_len": 105},
    "safe_001": {"task": "safety", "risk": "high", "old_len": 130, "new_len": 125},
    "summary_001": {"task": "summary", "risk": "normal", "old_len": 80, "new_len": 220},
    "math_001": {"task": "math", "risk": "normal", "old_len": 90, "new_len": 115},
}

pairwise = [
    {"item": "qa_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "qa_001", "annotator": "ann_b", "winner": "new", "new_pos": "right"},
    {"item": "rag_001", "annotator": "ann_a", "winner": "old", "new_pos": "left"},
    {"item": "rag_001", "annotator": "ann_b", "winner": "old", "new_pos": "right"},
    {"item": "code_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "code_001", "annotator": "ann_b", "winner": "tie", "new_pos": "right"},
    {"item": "safe_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "safe_001", "annotator": "ann_b", "winner": "new", "new_pos": "left"},
    {"item": "summary_001", "annotator": "ann_a", "winner": "new", "new_pos": "left"},
    {"item": "summary_001", "annotator": "ann_b", "winner": "new", "new_pos": "left"},
    {"item": "math_001", "annotator": "ann_a", "winner": "old", "new_pos": "right"},
    {"item": "math_001", "annotator": "ann_b", "winner": "new", "new_pos": "left"},
]

ratings = [
    ("qa_001", "old", 3), ("qa_001", "old", 3), ("qa_001", "new", 4), ("qa_001", "new", 4),
    ("rag_001", "old", 4), ("rag_001", "old", 4), ("rag_001", "new", 3), ("rag_001", "new", 3),
    ("code_001", "old", 3), ("code_001", "old", 3), ("code_001", "new", 4), ("code_001", "new", 3),
    ("safe_001", "old", 2), ("safe_001", "old", 3), ("safe_001", "new", 4), ("safe_001", "new", 4),
    ("summary_001", "old", 3), ("summary_001", "old", 3), ("summary_001", "new", 4), ("summary_001", "new", 4),
    ("math_001", "old", 4), ("math_001", "old", 3), ("math_001", "new", 4), ("math_001", "new", 4),
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


rating_by_model = {"old": [], "new": []}
for _, model, score in ratings:
    rating_by_model[model].append(score)

rating_summary = {
    "old": round(mean(rating_by_model["old"]), 3),
    "new": round(mean(rating_by_model["new"]), 3),
}
rating_summary["delta"] = round(rating_summary["new"] - rating_summary["old"], 3)

new_wins = sum(row["winner"] == "new" for row in pairwise)
old_wins = sum(row["winner"] == "old" for row in pairwise)
ties = sum(row["winner"] == "tie" for row in pairwise)
total = len(pairwise)
win_rate = (new_wins + 0.5 * ties) / total

scores_by_item = {}
labels_by_item = {}
for row in pairwise:
    item_id = row["item"]
    score = {"new": 1.0, "tie": 0.5, "old": 0.0}[row["winner"]]
    scores_by_item.setdefault(item_id, []).append(score)
    labels_by_item.setdefault(item_id, []).append(row["winner"])

item_scores = {item_id: mean(scores) for item_id, scores in scores_by_item.items()}

slice_scores = {}
for item_id, score in item_scores.items():
    task = items[item_id]["task"]
    slice_scores.setdefault(task, []).append(score)
slice_win_rate = {
    task: round(mean(scores), 3)
    for task, scores in slice_scores.items()
}

agreements = [
    len(set(labels)) == 1
    for labels in labels_by_item.values()
    if len(labels) >= 2
]
agreement_rate = mean(agreements)

left_wins = 0
right_wins = 0
longer_wins = 0
non_tie = 0
for row in pairwise:
    if row["winner"] == "tie":
        continue
    item = items[row["item"]]
    non_tie += 1
    if row["winner"] == "new":
        winner_pos = row["new_pos"]
        winner_len = item["new_len"]
        loser_len = item["old_len"]
    else:
        winner_pos = "right" if row["new_pos"] == "left" else "left"
        winner_len = item["old_len"]
        loser_len = item["new_len"]
    left_wins += winner_pos == "left"
    right_wins += winner_pos == "right"
    longer_wins += winner_len > loser_len

left_win_rate = left_wins / non_tie
position_bias = abs(left_win_rate - 0.5)
longer_win_rate = longer_wins / non_tie

high_risk_failures = [
    item_id
    for item_id, item in items.items()
    if item["risk"] == "high" and item_scores[item_id] < 0.5
]

required_rubric_fields = {
    "dimensions",
    "priority_order",
    "tie_rule",
    "anonymized",
    "position_randomized",
    "min_annotators",
    "gold_samples",
}
rubric_missing = sorted(required_rubric_fields - set(rubric))

gates = {
    "rubric": not rubric_missing and rubric["tie_rule"],
    "sample_count": len(items) >= 5,
    "agreement": agreement_rate >= 0.70,
    "win_rate": win_rate >= 0.53,
    "high_risk": not high_risk_failures,
    "position_bias": position_bias <= 0.20,
    "length_bias": longer_win_rate <= 0.70,
    "gold_samples": rubric["gold_samples"] >= 2,
}

summary = {
    "rating_summary": rating_summary,
    "pairwise": {
        "new_wins": new_wins,
        "old_wins": old_wins,
        "ties": ties,
        "win_rate": round(win_rate, 3),
        "tie_rate": round(ties / total, 3),
    },
    "slice_win_rate": slice_win_rate,
    "item_agreement": round(agreement_rate, 3),
    "position_bias": {
        "left_win_rate": round(left_win_rate, 3),
        "bias": round(position_bias, 3),
    },
    "longer_win_rate": round(longer_win_rate, 3),
    "high_risk_failures": high_risk_failures,
    "rubric_missing": rubric_missing,
    "gates": gates,
    "gate_pass": all(gates.values()),
}

pprint(summary, sort_dicts=False)
```

一组可复现输出如下：

```text
{'rating_summary': {'old': 3.167, 'new': 3.75, 'delta': 0.583},
 'pairwise': {'new_wins': 8,
              'old_wins': 3,
              'ties': 1,
              'win_rate': 0.708,
              'tie_rate': 0.083},
 'slice_win_rate': {'qa': 1.0,
                    'rag': 0.0,
                    'code': 0.75,
                    'safety': 1.0,
                    'summary': 1.0,
                    'math': 0.5},
 'item_agreement': 0.667,
 'position_bias': {'left_win_rate': 0.818, 'bias': 0.318},
 'longer_win_rate': 0.545,
 'high_risk_failures': ['rag_001'],
 'rubric_missing': [],
 'gates': {'rubric': True,
           'sample_count': True,
           'agreement': False,
           'win_rate': True,
           'high_risk': False,
           'position_bias': False,
           'length_bias': True,
           'gold_samples': True},
 'gate_pass': False}
```

这个输出说明：新模型总体 win rate 和绝对分看起来更高，但高风险 RAG 样本回退、标注一致率偏低、左侧位置胜出偏高，所以不能直接上线。真实评估报告里，这类失败门禁比“总分上涨”更重要。

## 13. 和 LLM-as-a-Judge 的关系

Human eval 和 LLM-as-a-Judge 不是互斥关系。

更实际的做法是组合使用。

Human eval 用于：

1. 建立 gold set。
2. 校准 judge。
3. 评估高风险和复杂样本。
4. 仲裁争议样本。

LLM-as-a-Judge 用于：

1. 扩大评估规模。
2. 快速回归测试。
3. 初筛明显好坏样本。
4. 发现需要人工复审的样本。

面试表达：LLM judge 可以提高效率，但 human eval 是校准和高风险决策的基准。

## 14. 面试回答模板

如果面试官问：

```text
你会如何设计一个 human eval 或 pairwise eval？
```

可以这样答：

```text
我会先明确评估目标，是比较两个模型版本、评估开放式质量，还是做上线门禁。对于开放式生成任务，我倾向使用 pairwise comparison，因为人类更容易判断两个回答哪个更好，而不是给绝对分数。流程上，我会从真实线上样本、困难样本、bad case 和高风险样本中采样，用固定 prompt、模型版本和解码参数生成回答，然后匿名化模型名称并随机打乱 A/B 顺序。

标注规范上，我会设计清晰 rubric 和优先级，例如安全性、正确性、事实依据、完整性、简洁性和风格，并给出正反例、tie 规则和边界案例。每个样本至少多人标注，高分歧样本进入仲裁。聚合时看 win rate、tie rate、置信区间和任务切片，同时计算标注一致性。为了控制偏见，我会做匿名化、顺序随机化、长度偏见提醒和 gold sample 质控。最终不只看总体 win rate，还会看关键场景是否回退，特别是安全、长上下文、无答案拒答和历史 bad case。
```

## 15. 常见误区

### 15.1 认为人工评估一定客观

人工评估也会有偏见和噪声。

要用 rubric、匿名化、随机化和一致性检查控制。

### 15.2 只找一个人打分

单人标注无法估计一致性。

关键评估要多人标注和仲裁。

### 15.3 只看总体 win rate

总体 win rate 可能掩盖关键场景回退。

要做切片分析。

### 15.4 忽视 tie 规则

Tie 规则会显著影响 win rate。

必须明确什么时候可以选 tie。

### 15.5 让标注员看到模型名称

这会引入品牌偏见。

要匿名化。

### 15.6 没有 gold sample 质控

没有质控，无法发现标注员漂移、误解规范或低质量标注。

## 16. 练习题

1. 为一个 RAG 问答系统设计 pairwise eval，包括采样、rubric、tie 规则和上线门禁。
2. 为什么 pairwise comparison 往往比 1-5 分绝对打分更稳定？
3. 如果两个标注员对同一批样本一致性很低，你会如何处理？
4. Human eval 中有哪些常见偏见？如何控制？
5. 如何把 human eval 和 LLM-as-a-Judge 结合起来降低成本？

## 17. 本章小结

本章讲了 Human Eval 与 Pairwise Eval。

核心结论：

1. 开放式生成任务没有唯一答案，因此需要 human eval。
2. Human eval 可以做绝对打分、pairwise comparison、排序和错误类型标注。
3. 绝对打分适合多维质量分析，但容易受标注尺度漂移影响。
4. Pairwise comparison 更适合比较模型版本，常用 win rate 聚合。
5. Elo 和 Arena 可以把大量 pairwise 偏好转成模型相对排名。
6. 标注规范要包含 rubric、正反例、tie 规则、边界案例和优先级。
7. 标注一致性是 human eval 可信度的重要指标。
8. 人工评估要控制长度偏见、位置偏见、格式偏见和品牌偏见。
9. 生产化 human eval pipeline 要包含采样、生成、匿名化、分配、质控、聚合、切片和决策。
10. Human eval 和 LLM-as-a-Judge 应该组合使用，人工评估负责校准和高风险决策，LLM judge 负责规模化和快速回归。
