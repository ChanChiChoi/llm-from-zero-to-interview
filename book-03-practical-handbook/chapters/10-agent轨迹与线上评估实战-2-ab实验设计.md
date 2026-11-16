# 第 10 章（2）：线上 A/B 实验设计与统计判断

[第一篇的 Agent 轨迹](10-agent轨迹与线上评估实战-1-轨迹诊断.md)能指出失败发生在哪一步，固定回归集能验证修复是否重复有效；它们仍不能告诉我们新版本在真实用户流量里会带来多少收益。这一篇设计一次线上 A/B 实验：先选稳定的随机化单元和主要指标，再检查分桶、估计样本量，最后读区间和结果。

先区分两个阶段：分桶和指标定义决定比较是否可信，样本量和不确定性决定我们能从数据中下多强的结论。代码示例会把这两类条件放在同一份报告里，但不会代替实验设计。

## 10.2 线上 A/B：在真实流量中验证收益和代价

### 10.2.1 离线变好，为什么还要线上实验

离线评估集可以控制输入和参考答案，却不能完整模拟真实用户。线上系统还有多轮上下文、缓存、检索版本、工具失败、网络延迟、用户中途退出、重复追问、成本预算和安全举报。一个新模型在离线问答集上提高 3%，可能因为输出更长而让 P95 延迟增加 40%；也可能因为更积极地调用工具而提高解决率，却增加了错误动作。

A/B 实验把真实流量随机分成对照组和实验组，在相同业务环境中比较预先定义的指标。它不是把一个版本“盖章”为永远正确，而是在一段时间和一组流量条件下估计变化，并持续观察风险。

### 先固定实验单元，再谈模型差异

如果同一个用户今天使用 A，下一轮又使用 B，他可能会把两种模型的答案混在同一段对话中。对单轮独立请求，这种随机方式有时可以接受；对多轮客服、Agent、个性化推荐和长期留存，应优先使用会话级或用户级分配，让实验单元在实验期间保持稳定。

```text
Control：旧模型或当前线上版本
Treatment：新模型、提示词或系统版本
实验单元：用户、会话、请求或组织
主指标：希望改善的业务结果
护栏指标：不能出现不可接受恶化的安全、质量、成本和稳定性指标
诊断指标：帮助解释变化来自哪里
```

### 随机分桶究竟在估计什么？

令实验单元为 $u$，实验名为 $e$，哈希函数把它映射到 $M$ 个桶，Treatment 比例为 $\rho$：

```math
r(u)=\frac{h(e,u)\bmod M}{M},\qquad
b(u)=\mathbb{1}[r(u)<\rho]
```

这里要求 $M$ 是正整数，且 $0\le\rho\le1$；在真正比较两组时还应有 $0<\rho<1$，否则某一组没有样本。哈希结果和分桶规则必须保持稳定。同一 $e,u$ 必须得到稳定的 $b(u)$。实验名进入哈希键，可以避免同一用户在不同实验中意外共享分桶结果；组织级产品还要考虑同一企业内多个用户互相影响，不能只看个人随机化。

对二值指标，Control 组成功数和样本数为 $(x_A,n_A)$，Treatment 组为 $(x_B,n_B)$：

```math
\hat p_A=\frac{x_A}{n_A},\qquad
\hat p_B=\frac{x_B}{n_B},\qquad
\Delta=\hat p_B-\hat p_A
```

提升量的近似标准误为：

```math
\mathrm{SE}_{\Delta}=
\sqrt{\frac{\hat p_A(1-\hat p_A)}{n_A}
+\frac{\hat p_B(1-\hat p_B)}{n_B}}
```

这些比例要求 $n_A,n_B>0$，并且 $0\le x_A\le n_A$、$0\le x_B\le n_B$。95% 近似区间为 $\Delta\pm1.96\mathrm{SE}_{\Delta}$。检验两组是否来自同一成功率时，常用合并比例：

```math
\hat p=\frac{x_A+x_B}{n_A+n_B}
```

```math
z=\frac{\hat p_B-\hat p_A}
{\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)}}
```

合并比例还要求 $n_A+n_B>0$，而 z 统计量的分母必须大于 0；如果两组成功率都处在完全相同的 0 或 1 边界，普通 z 值未定义，应使用适合边界比例的区间或只报告点估计。这些公式依赖独立样本、合理的随机化和足够的近似条件。多轮会话、同一用户多次请求和用户之间的社交影响都会让样本相关，不能机械地把每个请求当成独立观测。

### 10.2.2 先定义主指标、护栏和诊断指标

主指标应在实验开始前确定，例如客服场景的问题解决率、用户明确满意率、人工转接率或有效任务完成率。主指标不能在实验结束后从十几个指标中挑一个看起来最好的数字。

护栏指标观察不应被牺牲的维度：

```text
安全违规率和高风险错误率
事实错误或用户投诉率
请求失败率和工具执行失败率
P50、P95、P99 延迟以及首 token 延迟
输入输出 token、单次成本和日预算
格式解析失败率、过度拒答率
核心用户群和关键业务场景的退化
```

诊断指标帮助解释主指标变化：

```text
平均输出长度
检索命中与引用覆盖率
工具调用次数、成功率和重试次数
二次追问率与会话轮数
不同 query 长度、语言、渠道和任务类型的切片
缓存命中率、队列等待和模型服务错误码
```

一个完整的结论不应只有“解决率提高”，而应说明提高是否伴随更长回答、更高成本、更少人工转接，或者只是某一类简单问题占比改变。

### 10.2.3 实验单元的选择

#### 请求级随机

每个请求独立分配。它实现简单、样本增长快，适合单轮、无状态、结果互不影响的任务。缺点是同一用户或会话会在 A/B 之间切换，无法评价连贯的多轮体验。

#### 会话级随机

同一 session 固定模型。它适合客服、RAG 多轮问答和 Agent 任务，可以测量一次完整会话的解决率、工具成功率和用户是否继续追问。需要稳定保存 session 与实验组映射，跨设备或登录状态变化时要有明确规则。

#### 用户级随机

同一用户在实验周期内固定模型。它适合留存、长期满意度、复购和个性化产品，能减少跨组污染，但需要更长实验周期和更大样本量。

#### 组织级随机

企业客户常有共享知识库、团队策略和预算。如果同一组织内不同用户被分到不同模型，模型结果可能相互影响；对于组织级策略，应考虑按 tenant 或工作空间随机，同时关注组织间规模差异。

选择原则不是“粒度越细越科学”，而是选择能够保持因果链和用户体验稳定的最小实验单元。

### 10.2.4 Sample Ratio Mismatch：先检查分桶是否真的生效

假设计划按 50:50 分组，但实际日志中 Control 有 6,000 条，Treatment 有 4,000 条。直接计算满意率没有意义，因为流量分配、过滤、日志丢失或缓存路由可能已经破坏了随机化。

对 $k$ 个分组，观测数为 $o_j$，期望数为 $e_j$，可以计算：

```math
\chi^2_{\mathrm{SRM}}=\sum_{j=1}^{k}\frac{(o_j-e_j)^2}{e_j}
```

该式要求每个期望计数 $e_j>0$，且总样本数大于 0；如果某个分组的计划比例为 0，卡方统计量不适用。卡方近似还需要足够的期望计数，稀疏分桶应使用适合小样本的方法；原假设下将统计量与自由度为 $k-1$ 的卡方分布比较。计数应对应分配的随机化单元；按用户分桶时，不能直接用受到 treatment 影响的请求数检验原定用户比例。SRM 异常可能来自：

```text
实验路由条件与统计过滤条件不一致
某组请求更容易超时或被丢弃
缓存 key 没有包含实验组
登录、设备或渠道字段在一组缺失
重复请求和重试只在一组被计数
实验配置没有在所有服务实例同步
日志采集延迟或字段解析错误
```

SRM 本身不解释业务指标，但它是随机化有效性的前置检查。发现异常时，应先修复或隔离实验数据，而不是继续解释谁更好。

### 10.2.5 样本量、显著性和实际意义

如果基线成功率为 $p_1$，希望检测到 $p_2$，两组近似等量时，常见的样本量估计如下。这里通常要求 $0<p_1,p_2<1$、$p_1\ne p_2$，并预先给定显著性水平 $\alpha$ 与检验功效 $1-\beta$：

```math
n\approx
\frac{\left(
z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}
+z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}
\right)^2}{(p_2-p_1)^2},
\qquad
\bar p=\frac{p_1+p_2}{2}
```

这里 $z_q$ 为标准正态分布的第 $q$ 分位数，$n$ 是每组样本量。例如 $p_1=0.77,p_2=0.79,\alpha=0.05$、功效 0.8 时，取 $z_{0.975}\approx1.96,z_{0.8}\approx0.842$，每组约需 6,736 个独立单元（将近似值 6,735.15 向上取整）。预期提升越小，所需样本量按提升量平方的倒数增长。用户反馈有延迟、指标方差很大或用户级随机化时，有效样本量还会进一步下降。

若存在可靠的实验前协变量，还可以用 CUPED（Controlled-experiment Using Pre-Experiment Data）调整结果、降低方差。对随机化单元 $i$，令 $Y_i$ 为实验期指标、$X_i$ 为实验开始前测得的协变量、$\theta$ 为协变量系数、$\bar X$ 为其均值：

```math
Y_i^*=Y_i-\theta(X_i-\bar X),\qquad
\theta=\frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(X)}
```

在单协变量的线性近似中，若 $X$ 与 $Y$ 的相关系数为 $r=0.6$，最优线性调整后的方差比例约为 $1-r^2=0.64$，即这个假设例子中方差约降低 36%。实际收益取决于实验前指标的预测力、缺失情况和随机化单位；CUPED 降低测量噪声，不能修复分桶错误或结果日志缺失。[原始 CUPED 论文](https://exp-platform.com/cuped/)报告的线上收益绑定其具体实验条件。

统计显著不等于业务值得。一个 0.1 个百分点的提升可能在百万请求下显著，却无法覆盖新增 token 成本；一个高风险错误率的微小上升可能未达到显著，却仍值得暂停扩量。报告要同时给出点估计、置信区间、样本量、成本和风险后果。

实验期间频繁查看结果并在某个好看的时刻提前结束，会增加假阳性。若业务必须连续监控，应使用预先设计的 sequential testing 或明确的停止规则，不要把普通固定样本检验反复使用。

### 10.2.6 一个零依赖的 A/B 统计 demo

下面的程序演示稳定分桶、SRM、二比例差异、Wilson 区间、延迟和成本护栏。数字是教学数据，不能代表任何真实服务。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import hashlib
import json
import math


def assign_bucket(unit_id, experiment, treatment_ratio):
    if not isinstance(unit_id, str) or not unit_id.strip():
        raise ValueError("unit_id must be a non-empty string")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment must be a non-empty string")
    if not isinstance(treatment_ratio, (int, float)) or isinstance(treatment_ratio, bool):
        raise TypeError("treatment_ratio must be numeric")
    if not math.isfinite(treatment_ratio) or not 0 <= treatment_ratio <= 1:
        raise ValueError("treatment_ratio must be finite and in [0, 1]")
    key = f"{experiment}:{unit_id}".encode("utf-8")
    value = int(hashlib.sha256(key).hexdigest(), 16) % 10000
    return "treatment" if value / 10000 < treatment_ratio else "control"


def normal_cdf(value):
    if not math.isfinite(value):
        raise ValueError("normal_cdf input must be finite")
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def two_rate_delta(success_a, total_a, success_b, total_b):
    for name, successes, total in (
        ("control", success_a, total_a),
        ("treatment", success_b, total_b),
    ):
        if not isinstance(successes, int) or isinstance(successes, bool):
            raise TypeError(f"{name} successes must be an integer")
        if not isinstance(total, int) or isinstance(total, bool):
            raise TypeError(f"{name} total must be an integer")
        if total <= 0 or not 0 <= successes <= total:
            raise ValueError(f"{name} must satisfy total > 0 and 0 <= successes <= total")
    rate_a = success_a / total_a
    rate_b = success_b / total_b
    delta = rate_b - rate_a
    pooled = (success_a + success_b) / (total_a + total_b)
    pooled_se = math.sqrt(
        pooled * (1 - pooled) * (1 / total_a + 1 / total_b)
    )
    z_score = delta / pooled_se if pooled_se > 0 else None
    p_value = (
        2 * (1 - normal_cdf(abs(z_score)))
        if z_score is not None else None
    )
    unpooled_se = math.sqrt(
        rate_a * (1 - rate_a) / total_a
        + rate_b * (1 - rate_b) / total_b
    )
    ci95 = (
        [
            round(delta - 1.96 * unpooled_se, 4),
            round(delta + 1.96 * unpooled_se, 4),
        ]
        if unpooled_se > 0 else None
    )
    return {
        "control_rate": round(rate_a, 4),
        "treatment_rate": round(rate_b, 4),
        "delta": round(delta, 4),
        "z": round(z_score, 4) if z_score is not None else None,
        "p_value": round(p_value, 6) if p_value is not None else None,
        "ci95": ci95,
    }


def wilson(successes, total, z=1.96):
    if not isinstance(successes, int) or isinstance(successes, bool):
        raise TypeError("successes must be an integer")
    if not isinstance(total, int) or isinstance(total, bool):
        raise TypeError("total must be an integer")
    if total <= 0:
        raise ValueError("Wilson interval is undefined when total <= 0")
    if not 0 <= successes <= total:
        raise ValueError("successes must satisfy 0 <= successes <= total")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a finite positive number")
    p_hat = successes / total
    denominator = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        p_hat * (1 - p_hat) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(center - margin, 4), round(center + margin, 4)]


def srm(control_count, treatment_count, expected_treatment_ratio=0.5):
    if not isinstance(control_count, int) or isinstance(control_count, bool):
        raise TypeError("control_count must be an integer")
    if not isinstance(treatment_count, int) or isinstance(treatment_count, bool):
        raise TypeError("treatment_count must be an integer")
    if control_count < 0 or treatment_count < 0:
        raise ValueError("group counts cannot be negative")
    if not isinstance(expected_treatment_ratio, (int, float)) or isinstance(expected_treatment_ratio, bool):
        raise TypeError("expected_treatment_ratio must be numeric")
    if not math.isfinite(expected_treatment_ratio) or not 0 < expected_treatment_ratio < 1:
        raise ValueError("expected_treatment_ratio must be in (0, 1)")
    total = control_count + treatment_count
    if total <= 0:
        raise ValueError("SRM is undefined when total count is zero")
    expected_treatment = total * expected_treatment_ratio
    expected_control = total - expected_treatment
    statistic = (
        (control_count - expected_control) ** 2 / expected_control
        + (treatment_count - expected_treatment) ** 2 / expected_treatment
    )
    # For one degree of freedom, this is the upper-tail probability.
    p_value = math.erfc(math.sqrt(statistic / 2))
    return {
        "chi_square": round(statistic, 4),
        "p_value": round(p_value, 6),
        "no_srm_detected": p_value >= 0.001,
    }


control = {
    "sessions": 12000,
    "resolved": 9240,
    "negative_feedback": 960,
    "safety_events": 6,
    "p95_latency_ms": 1800,
    "avg_cost": 0.0032,
}
treatment = {
    "sessions": 12080,
    "resolved": 9590,
    "negative_feedback": 890,
    "safety_events": 7,
    "p95_latency_ms": 1950,
    "avg_cost": 0.0037,
}

for group_name, group in (("control", control), ("treatment", treatment)):
    sessions = group["sessions"]
    if not isinstance(sessions, int) or isinstance(sessions, bool) or sessions <= 0:
        raise ValueError(f"{group_name} sessions must be a positive integer")
    for metric in ("resolved", "negative_feedback", "safety_events"):
        count = group[metric]
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(f"{group_name} {metric} must be an integer")
        if not 0 <= count <= sessions:
            raise ValueError(f"{group_name} {metric} must be in [0, sessions]")
    if not math.isfinite(group["p95_latency_ms"]) or group["p95_latency_ms"] <= 0:
        raise ValueError(f"{group_name} latency must be finite and positive")
    if not math.isfinite(group["avg_cost"]) or group["avg_cost"] <= 0:
        raise ValueError(f"{group_name} cost must be finite and positive")

resolution = two_rate_delta(
    control["resolved"],
    control["sessions"],
    treatment["resolved"],
    treatment["sessions"],
)
negative_feedback = two_rate_delta(
    control["negative_feedback"],
    control["sessions"],
    treatment["negative_feedback"],
    treatment["sessions"],
)

latency_change = treatment["p95_latency_ms"] / control["p95_latency_ms"] - 1
cost_change = treatment["avg_cost"] / control["avg_cost"] - 1
safety_change = (
    treatment["safety_events"] / treatment["sessions"]
    - control["safety_events"] / control["sessions"]
)

conditions = {
    "srm_ok": srm(control["sessions"], treatment["sessions"])["no_srm_detected"],
    "resolution_lower_bound_above_1pt": (
        resolution["ci95"] is not None and resolution["ci95"][0] > 0.01
    ),
    "negative_feedback_upper_bound_below_0.1pt": (
        negative_feedback["ci95"] is not None
        and negative_feedback["ci95"][1] <= 0.001
    ),
    "p95_latency_increase_under_15pct": latency_change <= 0.15,
    "cost_increase_under_20pct": cost_change <= 0.20,
    "safety_review_passed": False,
}

report = {
    "bucket_examples": {
        user_id: assign_bucket(user_id, "assistant_v2", 0.5)
        for user_id in ["u001", "u002", "u003", "u004"]
    },
    "srm": srm(control["sessions"], treatment["sessions"]),
    "resolution": resolution,
    "negative_feedback": negative_feedback,
    "control_safety_ci95": wilson(control["safety_events"], control["sessions"]),
    "treatment_safety_ci95": wilson(treatment["safety_events"], treatment["sessions"]),
    "latency_change": round(latency_change, 4),
    "cost_change": round(cost_change, 4),
    "safety_rate_change": round(safety_change, 6),
    "conditions": conditions,
    "decision": "expand_to_25_percent" if all(conditions.values()) else "hold_and_investigate",
}
print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
```
本例把每个会话当作随机化单元，每个安全计数代表至少发生一次事件的会话数。50:50 分流时，两组期望数都是 12,040；实际为 12,000 和 12,080，故 $\chi^2=2\times40^2/12{,}040\approx0.266$，没有触发代码预设的 $p<0.001$ SRM 阈值。解决率分别为 $9{,}240/12{,}000=77.00\%$ 和 $9{,}590/12{,}080\approx79.39\%$，提升约 2.39 个百分点；按未合并标准误计算的 95% 区间约为 1.34 至 3.43 个百分点，超过示例中的 1 个百分点门槛。负反馈率从 8.00% 降到约 7.37%，其区间上界也低于预设的 0.10 个百分点非劣界限。P95 延迟增加约 8.33%，成本增加约 15.63%。

安全事件分别是 $6/12{,}000=0.050\%$ 和 $7/12{,}080\approx0.058\%$；代码输出两组 Wilson 区间，但这些稀疏事件仍不足以证明已满足安全非劣效要求。因此 `safety_review_passed` 显式为 False，最终决策是 `hold_and_investigate`。真实放行应预先定义安全容忍界限，并结合适合稀疏事件的区间、定向离线测试和人工复核；不能把布尔值直接改成 True 来替代证据。

`no_srm_detected` 只表示在本 demo 预设的 $p<0.001$ 阈值下没有检出分配比例异常；该阈值是教学设置，不是通用标准，也不证明所有随机化条件成立。`two_rate_delta` 的普通正态区间用于本例成功/失败数充足的指标；稀疏或边界比例应改用专门方法。即使 resolution 的区间为正，如果成本、延迟或安全条件不满足，也不应只凭一个正向数字扩大流量。真实分析还要考虑多重比较、集群随机化、分层协变量、缺失数据和实验周期。

组别各自的 Wilson 区间只描述各组比例；正式安全非劣效检验还应围绕预先设定的组间风险差界限构造不确定区间，或使用适合稀疏计数的方法。两组比例区间是否重叠不能替代这项检验。

得到估计结果后，还要问它怎样进入实际发布流程。下一篇继续检查日志和实验变量，并把主指标、护栏及不确定性转成灰度、暂停和回退决定。见[灰度发布、护栏与实验报告](10-agent轨迹与线上评估实战-3-灰度发布.md)。
