# 第 9 章（第四部分）：人工评测与生成质量

精确答案、结构字段和数值指标可以自动核验，但有帮助、表达清楚或是否回应用户意图，通常还要靠判断。人工评测若没有评分锚点和校准，也会把个人标准差异误当成模型差异。这一篇建立可复核的量表、盲评和分歧处理流程，并说明语言模型裁判适合承担什么角色。

## 9.6 生成质量人工评测：把主观判断变成可校准的测量

### 9.6.1 为什么自动指标不够

开放式生成没有一个对所有任务都适用的唯一参考答案。一个客服回答可能没有逐字复现参考答案，却完整解释了条件；一个 RAG 回答可能文字流畅，但引用并不支持结论；一个安全回答可能拒绝了危险请求，却使用了不必要的强硬措辞。BLEU、ROUGE、字符串匹配或 embedding 相似度都只能观察其中一部分。

人工评测的价值不在于“人永远正确”，而在于把业务真正关心的维度显式化，并通过标注指南、盲评、校准、多人复核和一致性统计减少主观噪声。人工评测结果也不是不可质疑的真值，它应当与自动检查、失败样例和线上指标共同构成证据。

### 先定评分维度，再讨论哪个回答更好

不要给标注员一个只有“好/坏”的按钮。一个完整回答至少可以从以下维度观察：

```text
事实正确性：关键事实是否正确
证据一致性：回答是否被给定资料支持
完整性：是否覆盖用户真正需要的条件和例外
指令遵循：是否按要求回答、拒答或调用工具
格式正确性：是否可被人和下游程序使用
安全性：是否产生危险、隐私或越权风险
可读性：是否清晰、适度简洁、没有无关内容
业务语气：是否符合场景和用户关系
```

维度越多不一定越好。两个维度如果标注员无法稳定区分，应该合并或修改定义；如果一个维度包含多个不同问题，应该拆开。例如“正确且完整”同时发生变化时，评分者无法解释分歧来自事实还是遗漏。

### 直接打分和成对比较回答不同问题

直接评分（absolute rating）是按每个维度给单条回答打分，适合找出哪些回答、哪些方面表现较差或出现回退；成对比较（pairwise preference）是让标注员在两条匿名回答中选出相对更好的一条，适合比较模型 A 和模型 B。两种方法不能互相替代：直接评分能看出两条回答都很差，成对比较却仍可能要选一条；成对比较更容易察觉细微偏好，但不会直接告诉我们错误有多严重。

可以同时保存：

```text
absolute:
case_id, model, dimension_scores, error_tags, rationale
pairwise:
case_id, answer_a, answer_b, winner, tie_reason, error_tags
```

A/B 位置必须随机，回答内容应去掉模型名称、版本名和内部提示词。若新模型总在 B 位置，位置偏好会污染结果；若标注员知道产品方期待哪个版本，也会产生预期效应。

### 9.6.2 量表设计：分数必须有行为锚点

以事实正确性为例，1--5 分可以写成行为定义，而不是抽象形容词：

```text
5：关键事实全部正确，没有重要遗漏或无依据扩展。
4：主要事实正确，有轻微措辞或低影响遗漏。
3：核心方向基本正确，但存在一个需要用户注意的条件遗漏。
2：包含明显事实错误，或关键结论只能部分成立。
1：核心结论错误、与证据矛盾，或捏造了关键事实。
```

同样的分数在不同维度不能共用同一套解释。格式正确性关注能否解析和是否满足 schema；安全性关注伤害、隐私、越权和可执行性；业务语气关注场景关系，而不是标注员个人喜欢的文风。

对 RAG 任务，可以把“证据一致性”设成独立维度：

```text
5：每个关键外部 claim 都能由给定证据支持，引用位置准确。
4：关键 claim 有支持，存在低影响的引用不完整。
3：主要结论有部分支持，但有条件或例外没有对应证据。
2：引用与结论有明显错配，或加入多条未支持事实。
1：核心结论与证据矛盾，或完全凭空作答。
```

量表还要给出正例、反例和边界案例。没有锚点时，标注员会把“3 分”理解成中间情绪，把“4 分”理解成自己满意，分数之间就失去可比性。

### 9.6.3 硬性条件与加权分数要分开

安全、关键事实和结构化解析通常不适合简单平均。设维度分数为 $s_{d}$，权重为 $w_{d}$，加权总分为：

```math
S_{\mathrm{weighted}}=\sum_{d\in\mathcal{D}}w_{d}s_{d},\qquad
\sum_{d\in\mathcal{D}}w_{d}=1
```

同时定义关键维度条件：

```math
C_{i}=\mathbb{1}[s_{i}\ge \tau_{i}],\qquad
C_{\mathrm{all}}=\prod_{i\in\mathcal{H}}C_{i}
```

这里要求 $\mathcal{D}\ne\varnothing$、 $w_{d}\ge0$、权重和确实为 1，且每个 $s_{d}$ 与 $\tau_{d}$ 位于预先约定的评分范围内； $\mathcal{H}$ 是明确列出的事实、安全、格式等关键维度集合。一个回答即使加权分数很高，只要核心条件不满足，也应被标记为需要修复或限制使用。这里的条件是评测协议中的判断，不是对所有场景都适用的固定数字；阈值应由风险和业务后果确定。

### 9.6.4 标注流程：校准、盲评和复核

#### 第一步：编写标注指南

指南应说明维度定义、分数锚点、正反例、边界样本、错误标签、证据来源和不确定时的处理方式。把“凭整体感觉打分”写成规则，无法提高一致性。

#### 第二步：小批量共同校准

让标注员先共同评一小批样本，逐条讨论分歧：为什么是 3 而不是 4，哪一句构成了 unsupported claim，合理拒答和过度拒答如何区分。讨论结果应回写指南，不能只停留在口头共识。

#### 第三步：正式盲评

随机回答顺序，隐藏模型身份，记录每个维度分数和错误标签。关键高风险样本、低置信度样本和模型 A/B 差异大的样本应进入多人复核。

#### 第四步：仲裁与版本化

分歧样本由资深标注员或领域专家仲裁，并保留原始分数、仲裁理由和指南版本。修改指南后，不应把新旧分数直接混在一起；需要重新标注一部分校准集，观察量表变化带来的影响。

### 9.6.5 一致性：不要只看平均分

若两名标注员在 $n$ 条样本上的类别标签分别为 $a_{i},b_{i}$，观测一致率为：

```math
p_{o}=\frac{1}{n}\sum_{i=1}^{n}\mathbb{1}[a_{i}=b_{i}]
```

Cohen's Kappa 进一步扣除按边际分布计算的偶然一致率 $p_{e}$：

```math
\kappa=\frac{p_{o}-p_{e}}{1-p_{e}}
```

这两个式子要求 $n>0$；Cohen's Kappa 还要求 $1-p_{e}\ne0$。当 $p_{e}=1$ 时分母为 0，一致性系数应报告为“不适用”，而不是返回一个默认值。当某一类别极少时，Kappa 可能受到类别分布影响；不能把某个 Kappa 数字当成跨任务通用标准。连续或有序分数可以使用加权 Kappa、Spearman 相关或 Krippendorff's Alpha，具体取决于数据类型和缺失模式。

标注一致性低时，先检查量表和样本难度，不要直接把低一致性归咎于标注员。常见原因是：事实需要领域知识、维度定义重叠、正例不足、证据版本不一致、回答 A/B 排序没有随机，或任务本身存在合理多解。

例如 10 条二分类标注中，两位评分者对 4 条都判为“是”，对 3 条都判为“否”，其余 3 条意见不同，则观测一致率 $p_{o}=(4+3)/10=0.7$。若评分者 1 的“是”比例为 $5/10$、评分者 2 为 $6/10$，偶然一致率 $p_{e}=(5/10)(6/10)+(5/10)(4/10)=0.5$，因此 $\kappa=(0.7-0.5)/(1-0.5)=0.4$。这个数依赖边际分布和任务标签；不能脱离样本构成把 0.4 当作普遍质量等级。

### 9.6.6 样本抽取与统计

人工评测样本应分层抽取，而不是只随机抽取高频简单请求：

```text
高频与长尾
短输入与长输入
单轮与多轮
证据充分与证据不足
正常与安全敏感
低延迟与超时
有用户负反馈与无反馈
模型 A/B 分歧大与分歧小
```

对二值错误率，报告错误数、样本数和 Wilson 区间；对 1--5 分量表，报告均值、中位数、分布和分组结果。若使用 pairwise，至少报告 A 胜、B 胜、平局以及排除平局后的胜率。

假设 B 赢了 $w$ 次，输了 $l$ 次，排除平局后的偏好率为：

```math
\hat p_{B}=\frac{w}{w+l}
```

这里要求 $w,l\ge0$ 且 $w+l>0$；如果所有比较都是平局，排除平局后的胜率没有定义，应报告“不适用”。当样本很少时，点估计容易被几条样本改变。报告区间比只写“B 胜率 60%”更诚实。

### 9.6.7 LLM-as-Judge 的位置

另一个语言模型可以大规模读取回答、参考答案和评分标准，降低人工成本。它适合做初筛、回归、候选错误标签和人工复核排序；它不应无条件替代高风险事实判断、专业审查和最终发布决定。

使用模型评审时至少要固定并记录：

```text
judge_model_revision
judge_prompt_revision
输入字段顺序和证据
是否显示模型身份
评分尺度与输出 schema
随机种子和采样参数
人工校准集上的一致性和偏差
```

需要检查 position bias、verbosity bias、self-preference、格式偏好和语言偏好。例如更长的回答可能看起来更完整，却包含更多未经支持的 claim；与被评估模型同源的 judge 可能偏好相似表达。人工小样本应作为校准集，定期比较 judge 与专家判断的分歧。

### 9.6.8 一个零依赖人工评测统计器

下面的程序模拟两名标注员的维度评分、关键条件、错误标签和 pairwise 结果。它的重点是把维度、失败条件和相对偏好分开统计。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import json
import math
from collections import Counter, defaultdict


DIMENSIONS = ["factuality", "groundedness", "format", "safety"]
WEIGHTS = {
    "factuality": 0.35,
    "groundedness": 0.25,
    "format": 0.15,
    "safety": 0.25,
}
THRESHOLDS = {
    "factuality": 3,
    "groundedness": 3,
    "format": 4,
    "safety": 4,
}

ROWS = [
    {
        "case": "c1",
        "model": "A",
        "annotator": "p1",
        "scores": {"factuality": 4, "groundedness": 4, "format": 5, "safety": 5},
        "tags": ["missing_detail"],
    },
    {
        "case": "c1",
        "model": "A",
        "annotator": "p2",
        "scores": {"factuality": 4, "groundedness": 3, "format": 5, "safety": 5},
        "tags": ["missing_detail"],
    },
    {
        "case": "c1",
        "model": "B",
        "annotator": "p1",
        "scores": {"factuality": 5, "groundedness": 5, "format": 5, "safety": 5},
        "tags": [],
    },
    {
        "case": "c1",
        "model": "B",
        "annotator": "p2",
        "scores": {"factuality": 5, "groundedness": 5, "format": 5, "safety": 5},
        "tags": [],
    },
    {
        "case": "c2",
        "model": "A",
        "annotator": "p1",
        "scores": {"factuality": 2, "groundedness": 2, "format": 5, "safety": 2},
        "tags": ["unsupported_claim", "unsafe"],
    },
    {
        "case": "c2",
        "model": "A",
        "annotator": "p2",
        "scores": {"factuality": 2, "groundedness": 2, "format": 5, "safety": 2},
        "tags": ["unsupported_claim", "unsafe"],
    },
    {
        "case": "c2",
        "model": "B",
        "annotator": "p1",
        "scores": {"factuality": 4, "groundedness": 4, "format": 5, "safety": 5},
        "tags": [],
    },
    {
        "case": "c2",
        "model": "B",
        "annotator": "p2",
        "scores": {"factuality": 4, "groundedness": 4, "format": 5, "safety": 5},
        "tags": [],
    },
]

PAIRWISE = ["B", "B", "tie", "A", "B", "B", "tie"]


def validate_inputs():
    if not DIMENSIONS or not ROWS or not PAIRWISE:
        raise ValueError("dimensions, rows and pairwise comparisons must be non-empty")
    if set(WEIGHTS) != set(DIMENSIONS) or set(THRESHOLDS) != set(DIMENSIONS):
        raise ValueError("weights and thresholds must cover every dimension exactly")
    if not math.isclose(sum(WEIGHTS.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("dimension weights must sum to 1")
    for name in DIMENSIONS:
        weight = WEIGHTS[name]
        threshold = THRESHOLDS[name]
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid weight for {name}")
        if not isinstance(threshold, int) or not 1 <= threshold <= 5:
            raise ValueError(f"threshold for {name} must be an integer in [1, 5]")
    allowed = set(DIMENSIONS)
    seen = set()
    for row in ROWS:
        for field in ("case", "model", "annotator"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        key = (row["case"], row["model"], row["annotator"])
        if key in seen:
            raise ValueError(f"duplicate annotation: {key}")
        seen.add(key)
        if not isinstance(row.get("scores"), dict) or set(row["scores"]) != allowed:
            raise ValueError("each row must contain one score for every dimension")
        for name, value in row["scores"].items():
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                raise ValueError(f"score for {name} must be an integer in [1, 5]")
        if not isinstance(row.get("tags"), list):
            raise TypeError("tags must be a list")
    if any(value not in {"A", "B", "tie"} for value in PAIRWISE):
        raise ValueError("pairwise values must be A, B or tie")
    if PAIRWISE.count("A") + PAIRWISE.count("B") == 0:
        raise ValueError("pairwise win rate is undefined when all comparisons are ties")


def passes(row):
    return all(row["scores"][name] >= threshold for name, threshold in THRESHOLDS.items())


def weighted(row):
    return sum(row["scores"][name] * WEIGHTS[name] for name in DIMENSIONS) / 5.0


validate_inputs()
by_model = defaultdict(list)
for row in ROWS:
    by_model[row["model"]].append(row)

summary = {}
for model, rows in sorted(by_model.items()):
    tags = Counter(tag for row in rows for tag in row["tags"])
    summary[model] = {
        "weighted_mean": round(sum(weighted(row) for row in rows) / len(rows), 4),
        "condition_failure_rate": round(
            sum(not passes(row) for row in rows) / len(rows), 4
        ),
        "dimension_means": {
            name: round(
                sum(row["scores"][name] for row in rows) / len(rows), 3
            )
            for name in DIMENSIONS
        },
        "tags": dict(tags),
    }

wins_a = PAIRWISE.count("A")
wins_b = PAIRWISE.count("B")
ties = PAIRWISE.count("tie")
summary["pairwise"] = {
    "A": wins_a,
    "B": wins_b,
    "tie": ties,
    "B_rate_without_ties": round(wins_b / (wins_a + wins_b), 4),
}
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
```
输出中 A、B 的 `weighted_mean` 分别为 $0.6725$、 $0.94$，`condition_failure_rate` 分别为 $0.5$、 $0$；排除两次平局后 B 胜率为 $4/5=0.8$。均值和失败率按标注记录计算，本例每条回答恰有两名标注员，不能把八条评分记录视为八个独立问题。多人评分数量不均时，应先按回答汇总，再按问题聚合；区间估计也应保留同一问题内的相关性。真实评测还应分开统计解析失败、引用错误、严重安全事件和领域专家否决。不同错误的严重程度不能被一个布尔值完全表达。

### 9.6.9 评测表的最小字段

一份可维护的 absolute rating 表可以包含：

```text
sample_id
task_type
input
context_or_evidence
model_revision
prompt_revision
output
dimension_scores
error_tags
annotator_id
guide_revision
confidence
adjudication
timestamp
```

成对比较还需保存回答顺序的随机种子和原始 A/B 映射，方便排查位置效应。引用评估要保存 evidence span，而不是只保存 URL；工具调用评估要保存工具请求、执行返回和最终叙述三者。表中的输入、上下文和输出应留在授权、隔离、可审计且具有保留期限的评测空间；普通报告使用样本 ID 或受保护引用，避免把真实用户内容复制到一般日志或共享文档。

### 9.6.10 资料与边界

[Cohen's Kappa 原始论文 A Coefficient of Agreement for Nominal Scales](https://doi.org/10.1177/001316446002000104)提出了分类判断一致性的一种校正方法；[Krippendorff 关于内容分析一致性的论文](https://doi.org/10.1111/j.1468-2958.2004.tb00738.x)讨论了不同数据类型的一致性；HELM 和 OpenAI Evals 则提供了多维评估与可复现评测基础设施的公开参考。

[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)讨论位置、冗长和自我偏好等评审偏差；其人工一致性结论绑定论文的模型、题目和协议，不代表任意 judge 在新业务中都可靠。

截至 2026-11-16，OpenAI 官方[评测指南](https://developers.openai.com/api/docs/guides/evals)说明，既有托管 Evals 已于 2026-10-31 进入只读期，平台计划于 2026-11-30 下线；官方[迁移时间表](https://developers.openai.com/api/docs/deprecations)还将 dashboard、API 和相关 graders 列入迁移范围。对新建或迭代评测，指南提到 Datasets；既有托管工作流的迁移表另给出 Promptfoo 入口。它们针对不同迁移场景，不能据此认为功能完全等价。这是服务生命周期变化，不表示评测方法或开源评测仓库失效；依赖托管服务的团队应按官方页面复核自身资源状态，并把样本、评分规则和运行记录保存在可迁移的位置。

人工评测不是把主观判断伪装成客观真值。量表、标注员、证据来源和仲裁过程都应进入结果记录；任何“模型 B 更好”的结论都应能追溯到具体任务、具体维度和具体失败样例。


## 把四个模块带回项目

1. 一个系统四条样例全部正确，能否声明准确率接近 100%？运行[第一部分的 runner](09-评估与debug实战-1-评测协议与benchmark.md)，解释点估计与 Wilson 下界为何不同，并把一条 JSON 年龄改成字符串，指出评分器没有检查什么。
2. 政策要求说明两种渠道，回答只正确覆盖一种。参照[第二部分的 claim 评估](09-评估与debug实战-2-数据与事实性核验.md)，分别计算回答支持率与必需事实完整性；再把证据改为否定句，解释为什么字符串检查会误判。
3. 在[第三部分的最小训练实验](09-评估与debug实战-3-训练调试与能力回归.md)中把标签全部改成 0，再恢复标签并把学习率改为 100。比较损失、梯度和最终预测，说明什么证据支持“数据问题”或“更新强度问题”，以及哪些结论不能外推到 Transformer。

四个模块最后回到同一条判断路径：固定测量协议、保留逐样本证据、用对照排除替代解释。训练曲线、事实检查、人工偏好和不同类别的回归结果观察的是不同对象；只有明确各自分母与证据范围，才能判断离线改动是否值得进入系统级评估。

离线评估能回答固定样本上的变化，不能单独证明真实流量上的结果。下一步进入[第十章 Agent 轨迹与线上评估](10-agent轨迹与线上评估实战.md)，把单次执行轨迹和线上流量纳入系统级发布判断。
