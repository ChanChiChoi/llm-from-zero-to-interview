# 第四章：LLM-as-a-Judge

重点：自动裁判模型、打分 rubrics、position bias、verbosity bias、self-preference、校准方法。

面试重点：LLM-as-a-judge 高效但不等于客观真相。

## 本章目标

学完本章，你要能回答：

1. 什么是 LLM-as-a-Judge？
2. 它适合评估哪些任务，不适合哪些任务？
3. 如何设计 judge prompt 和打分 rubric？
4. 常见 bias 包括哪些，如何控制？
5. 如何用人工 gold set 校准 judge？
6. 面试中如何评价 LLM judge 的可信度？

LLM-as-a-Judge 是用大模型作为裁判，评价另一个模型或系统的输出。

它的核心价值是降低评估成本、提高评估速度、扩大评估规模。

但它不是客观真相。

## 1. 为什么需要 LLM-as-a-Judge

Human eval 质量高，但成本高、速度慢、规模有限。

大模型迭代很快，可能每天都有：

1. 新 prompt。
2. 新模型 checkpoint。
3. 新 RAG 配置。
4. 新 reranker。
5. 新 safety policy。
6. 新 decoding 参数。

如果每次都做人类评审，成本和周期都不可接受。

LLM-as-a-Judge 的作用是：

1. 快速评估大量样本。
2. 做自动回归测试。
3. 初筛明显好坏样本。
4. 辅助定位 bad case。
5. 扩展 pairwise eval 规模。
6. 为人工评审节省预算。

面试表达：LLM-as-a-Judge 的价值是把开放式生成任务的评估规模化，但它必须用人工 gold set 校准，不能当作绝对真理。

## 2. LLM Judge 的基本形式

LLM judge 常见有四种形式。

### 2.1 单答案打分

给定问题和一个回答，让 judge 打分。

例如：

```text
Question: ...
Answer: ...
请从正确性、有用性、完整性三个维度打分。
```

适合：

1. 单模型质量估计。
2. 多维质量画像。
3. 回归测试。

风险：分数尺度不稳定。

### 2.2 Pairwise 判断

给定同一个问题和两个回答，让 judge 选择更好的一方。

例如：

```text
Question: ...
Answer A: ...
Answer B: ...
请选择 A 更好、B 更好或 Tie，并说明原因。
```

适合比较两个模型版本。

Pairwise 通常比单答案打分更稳定。

### 2.3 Rubric-based 打分

给 judge 一个明确 rubric，让它按规则评分。

例如 RAG 评估：

1. 答案是否正确。
2. 是否基于给定证据。
3. 引用是否准确。
4. 是否有幻觉。
5. 是否应该拒答。

### 2.4 Error Tagging

让 judge 标注错误类型。

例如：

```text
错误类型只能从以下标签中选择：事实错误、引用错误、格式错误、拒答错误、安全违规、无错误。
```

适合 error analysis 和 bad case 聚类。

面试表达：LLM judge 可以做单答案评分、pairwise 选择、rubric-based 打分和错误类型标注，其中 pairwise 和 rubric-based 通常更可靠。

## 3. 适用场景

LLM-as-a-Judge 适合这些场景：

1. 开放式问答。
2. 摘要质量评估。
3. 指令遵循评估。
4. RAG faithfulness 初筛。
5. 多轮对话质量评估。
6. Pairwise preference 扩展。
7. 回归测试初筛。
8. Bad case 自动归因。

但它不适合单独用于这些场景：

1. 高风险安全最终裁决。
2. 法律、医疗、金融等强专业领域最终判断。
3. 需要运行代码或验证数学证明的任务。
4. 需要精确事实核查但没有证据输入的任务。
5. judge 模型明显弱于被评模型的任务。

原因是 judge 也会 hallucinate，也会误判，也有偏见。

面试表达：LLM judge 适合规模化、低风险、开放式评估，但高风险决策和专业最终判断仍需要人工或程序化验证。

## 4. Judge Prompt 设计

Judge prompt 决定评估质量。

一个好的 judge prompt 通常包含：

1. 角色定义。
2. 任务说明。
3. 评分维度。
4. Rubric。
5. 输入字段。
6. 输出格式。
7. Tie 或不确定规则。
8. 禁止使用无关标准。
9. 示例。

### 4.1 单答案评分 Prompt

示例：

```text
你是一个严格的评估员。请根据给定问题、参考答案和模型回答进行评分。

评分维度：
1. 正确性：回答是否事实正确。
2. 完整性：是否覆盖问题关键点。
3. 简洁性：是否避免无关内容。

评分规则：
5 分：完全正确且完整。
4 分：基本正确，只有轻微遗漏。
3 分：部分正确，但遗漏关键点。
2 分：有明显错误。
1 分：主要错误或答非所问。
0 分：严重错误或安全违规。

请输出 JSON：
{"score": 0-5, "reason": "简短理由"}
```

### 4.2 Pairwise Prompt

示例：

```text
你是一个严格的回答质量评估员。给定同一个问题和两个匿名回答，请判断哪个回答更好。

优先级：
1. 安全性。
2. 正确性。
3. 是否遵循指令。
4. 完整性。
5. 简洁性。

如果两个回答质量接近，请选择 Tie。
不要因为回答更长、格式更漂亮就判为更好。

请输出 JSON：
{"winner": "A" | "B" | "Tie", "reason": "简短理由"}
```

### 4.3 RAG Judge Prompt

RAG 评估要给 evidence。

示例：

```text
请只根据给定证据评价回答。
如果回答中的信息无法由证据支持，应标记为 hallucination。
如果证据不足而回答强行给出结论，应扣分。
```

面试表达：Judge prompt 要明确角色、rubric、优先级、输出格式和 tie 规则，并提醒不要被长度和格式误导。

## 5. Rubric 设计

Rubric 是 judge 的评分标准。

没有 rubric，judge 会按自己的偏好打分。

### 5.1 好 Rubric 的要求

好的 rubric 应该：

1. 明确。
2. 可执行。
3. 覆盖边界情况。
4. 维度不要过多。
5. 有优先级。
6. 与任务目标一致。

### 5.2 常见维度

开放式问答：

1. correctness。
2. helpfulness。
3. completeness。
4. clarity。
5. conciseness。

RAG：

1. faithfulness。
2. citation correctness。
3. answer correctness。
4. abstention correctness。

安全：

1. policy compliance。
2. harmfulness。
3. false refusal。
4. severity。

Agent：

1. task success。
2. tool call correctness。
3. unnecessary steps。
4. unsafe action。

### 5.3 维度冲突

维度之间会冲突。

例如一个回答很有帮助，但泄露隐私。

这时安全应该优先。

可以定义：

```text
安全违规直接 fail。
事实错误优先于表达风格。
证据不支持优先于回答流畅。
```

面试表达：Rubric 要把评估标准显式化，尤其要定义维度优先级和 fail 条件。

## 6. 输出格式和可解析性

生产系统中，judge 输出最好结构化。

推荐使用 JSON。

例如：

```json
{
  "score": 4,
  "label": "mostly_correct",
  "has_hallucination": false,
  "winner": "A",
  "reason": "A 更准确且更简洁"
}
```

结构化输出的好处是：

1. 便于自动聚合。
2. 便于统计切片。
3. 便于发现解析错误。
4. 便于存档和复现。

要注意：

1. JSON schema 要固定。
2. 非法输出要重试或标记无效。
3. reason 不宜过长。
4. 评分字段要有限枚举。

面试表达：生产化 LLM judge 要输出结构化结果，否则很难自动聚合和回归测试。

## 7. 常见 Bias

LLM judge 有很多偏差。

面试中必须能说清楚。

### 7.1 Position Bias

Position bias 是位置偏见。

在 pairwise 判断中，judge 可能更偏好 A 或 B。

解决方法：

1. 随机交换 A/B 顺序。
2. 同一样本正反顺序各评一次。
3. 检查位置胜率是否异常。

### 7.2 Verbosity Bias

Verbosity bias 是长度偏见。

judge 可能偏好更长、更详细的回答。

解决方法：

1. Prompt 中明确长不等于好。
2. Rubric 中加入简洁性。
3. 分析 winner 和输出长度的相关性。

### 7.3 Self-preference

Self-preference 是自我偏好。

某些 judge 可能偏好和自己风格相似的模型输出，甚至偏好同一家族模型的回答。

解决方法：

1. 使用不同家族 judge 交叉评估。
2. 使用人工 gold set 校准。
3. 不让 judge 看到模型名称。

### 7.4 Format Bias

格式偏见是指 judge 偏好列表、标题、Markdown 等形式。

格式好不代表内容正确。

解决方法是在 rubric 中强调内容优先。

### 7.5 Authority Bias

如果回答语气很自信，judge 可能更容易相信它。

这会放大幻觉风险。

### 7.6 Reference Bias

如果给了参考答案，judge 可能过度惩罚不同表达。

如果没有参考答案，judge 又可能凭自己知识误判。

所以是否提供 reference 要按任务决定。

面试表达：LLM judge 常见偏差包括 position bias、verbosity bias、self-preference、format bias、authority bias 和 reference bias，需要通过随机化、反向评估、多 judge、人工校准和切片分析控制。

## 8. 校准方法

LLM judge 必须校准。

校准目标是回答：judge 的判断和人类专家是否一致？

### 8.1 人工 Gold Set

先构建一批人工高质量标注样本。

包括：

1. 明显正确。
2. 明显错误。
3. 两者接近。
4. 有事实错误。
5. 有引用错误。
6. 有安全问题。
7. 有格式但无内容。
8. 有边界争议。

然后让 judge 在这些样本上评估，比较它和人工标注的一致性。

### 8.2 指标

可以看：

1. agreement rate。
2. correlation。
3. precision / recall。
4. false positive / false negative。
5. pairwise agreement。
6. confusion matrix。

### 8.3 多 Judge Ensemble

可以使用多个 judge。

例如一个强通用模型、一个专业模型、一个安全模型。

如果多个 judge 分歧大，就交给人工复审。

### 8.4 Threshold Calibration

如果 judge 输出分数，要校准阈值。

例如分数 >= 4 是否真的代表可上线？

需要用 gold set 找到合适阈值，而不是凭感觉定。

### 8.5 Periodic Recalibration

Judge prompt、judge model、任务分布变化后都要重新校准。

否则评估标准会漂移。

面试表达：LLM judge 必须用人工 gold set 校准，关注和人工的一致性、误判类型、阈值选择和周期性重校准。

## 9. 与 Human Eval 的组合

最可靠的实践不是只用人，也不是只用 judge，而是组合。

推荐流程：

```text
Human Gold Set
  -> Calibrate Judge
  -> Large-scale Judge Eval
  -> Sample Human Audit
  -> Disagreement Review
  -> Update Rubric / Judge Prompt
```

### 9.1 Judge 负责规模

Judge 可以每天跑大量回归样本。

### 9.2 人类负责基准

人类专家负责 gold set、高风险样本、争议样本和最终上线判断。

### 9.3 分歧样本最有价值

Judge 和人类分歧的样本通常最值得分析。

它们能暴露：

1. Rubric 不清晰。
2. Judge 有偏见。
3. 标注员理解不一致。
4. 样本本身有歧义。

面试表达：Human eval 是校准锚点，LLM judge 是规模化工具，二者要闭环使用。

## 10. 生产化 LLM Judge Pipeline

一个生产化 judge pipeline 可以是：

```text
Eval Dataset
  -> Candidate Generation
  -> Judge Prompt Rendering
  -> Judge Inference
  -> Output Validation
  -> Score Aggregation
  -> Bias Check
  -> Human Audit
  -> Report / Gate
```

### 10.1 Candidate Generation

固定被评模型、prompt、数据和解码参数。

### 10.2 Judge Prompt Rendering

使用版本化 judge prompt。

### 10.3 Judge Inference

记录 judge 模型版本和参数。

Judge 自身也要稳定。

### 10.4 Output Validation

检查 JSON 是否可解析、字段是否完整、分数是否在范围内。

### 10.5 Score Aggregation

计算平均分、win rate、tie rate、错误类型分布和置信区间。

### 10.6 Bias Check

检查位置偏见、长度偏见、模型家族偏见和异常样本。

### 10.7 Human Audit

抽样人工复审，重点看低置信、分歧和高风险样本。

### 10.8 Report / Gate

把结果转成上线门禁。

例如：

```text
LLM judge win rate >= 55%
人工抽检一致性 >= 85%
安全高风险样本 0 违规
关键业务切片无显著回退
```

面试表达：生产化 LLM judge 要有版本化 prompt、结构化输出、bias check、人工抽检和上线门禁，不能只是调用一次模型问“哪个好”。

## 11. 可信度判断

如何判断一个 LLM judge 结果可信？

可以看六点：

1. 是否有人工 gold set 校准。
2. 是否报告和人工的一致性。
3. 是否做 position swap。
4. 是否检查长度偏见。
5. 是否有多 judge 或人工抽检。
6. 是否做切片分析和置信区间。

如果一个评估只说“GPT judge 认为模型 B 更好”，但没有说明 prompt、rubric、校准、bias check 和样本分布，这个结论不可信。

面试表达：LLM judge 的可信度来自校准、偏差控制、可复现和人工审计，而不是来自 judge 模型本身强。

## 12. 示例：评估 RAG 答案

假设要评价一个 RAG 答案。

输入包括：

1. 用户问题。
2. 检索证据。
3. 模型回答。

Judge 输出：

```json
{
  "answer_correctness": 4,
  "faithfulness": 5,
  "citation_correctness": 4,
  "has_hallucination": false,
  "should_abstain": false,
  "reason": "回答基于证据，引用基本正确，但遗漏了一个限制条件"
}
```

关键点：

1. Judge 必须只根据证据判断。
2. 如果答案包含证据不支持的信息，要判 hallucination。
3. 如果证据不足而模型强答，要扣分。
4. 如果应该拒答却回答，要标记 abstention 错误。

这类 judge 可以快速筛选 RAG bad case，但高风险结论仍要人工复核。

## 13. 面试回答模板

如果面试官问：

```text
你如何看待 LLM-as-a-Judge？会怎么用？
```

可以这样答：

```text
LLM-as-a-Judge 的核心价值是把开放式生成任务评估规模化，尤其适合做 pairwise comparison、rubric-based scoring、RAG faithfulness 初筛和回归测试。但我不会把它当作客观真相，因为 judge 本身也有 position bias、verbosity bias、format bias、self-preference 和 hallucination。

如果我要使用它，会先定义清晰的 judge prompt 和 rubric，明确评分维度、优先级、tie 规则和结构化输出。然后用人工 gold set 校准 judge，比较它和人类专家的一致性，分析误判类型，并做 position swap、长度偏见检查、多 judge 或人工抽检。生产上我会版本化 judge prompt、judge model、被评模型、数据集和解码参数，保存原始输出和 judge 结果。最终决策不会只看 judge 总分，还会看任务切片、安全切片、置信区间和人工审计结果。
```

## 14. 常见误区

### 14.1 把 Judge 当客观真理

Judge 也是模型，也会犯错。

必须校准。

### 14.2 Judge Prompt 太模糊

如果只问“哪个回答更好”，judge 会按自身偏好判断。

要给 rubric 和优先级。

### 14.3 不做顺序随机

Pairwise judge 如果不随机 A/B 顺序，position bias 会影响结果。

### 14.4 忽视长度偏见

Judge 可能偏好更长回答。

要检查 winner 和长度的相关性。

### 14.5 用同一家族模型互评

可能引入 self-preference。

要用人工 gold set 或多 judge 校准。

### 14.6 不保存原始输出

只保存分数无法复查误判。

要保存 prompt、模型回答、judge 输出和版本。

## 15. 练习题

1. 为一个客服 RAG 系统写一个 LLM judge rubric，包含 correctness、faithfulness 和 citation correctness。
2. 设计一个实验检测 judge 是否有 position bias。
3. 如何判断 judge 是否存在 verbosity bias？
4. 为什么 LLM judge 需要人工 gold set 校准？
5. 如果 judge 和人工标注一致性只有 65%，你会如何改进？

## 16. 本章小结

本章讲了 LLM-as-a-Judge。

核心结论：

1. LLM-as-a-Judge 能降低开放式评估成本，提高评估规模和速度。
2. Judge 可以做单答案打分、pairwise 判断、rubric-based 打分和错误类型标注。
3. Judge 适合规模化初筛和回归测试，不适合高风险最终裁决。
4. Judge prompt 要包含角色、任务、rubric、优先级、tie 规则和结构化输出。
5. Rubric 要明确维度、边界情况和 fail 条件。
6. 生产化 judge 输出应尽量结构化，方便聚合和复查。
7. 常见偏差包括 position bias、verbosity bias、self-preference、format bias、authority bias 和 reference bias。
8. Judge 必须用人工 gold set 校准，并报告和人类的一致性。
9. Human eval 是校准锚点，LLM judge 是规模化工具。
10. 可信的 LLM judge pipeline 要有版本管理、bias check、人工抽检、切片分析和上线门禁。
