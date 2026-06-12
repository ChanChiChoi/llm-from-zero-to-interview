# 第十章：Evaluation Harness

## 0. 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做公式和 demo 精修，联网核对了 OpenAI Evals API 与 graders 的公开资料，OpenAI `simple-evals` 对轻量基准评估的工程口径，SWE-bench 关于 issue、repo snapshot、patch、test harness 和 verified 子集的公开资料，以及 LangSmith 关于 datasets、experiments、evaluators、trace 和 regression 的公开说明。

本章聚焦防御性的 Evaluation Harness 设计：如何用固定数据集、环境快照、sandbox reset、验收器、trajectory scoring、baseline comparator、regression runner、成本 / 安全指标和 trace artifacts 判断 agent runtime 是否真的变好。正文和 demo 不提供 benchmark 污染、规避验收器、伪造测试通过、隐藏失败 trace、绕过权限或攻击评估平台的方法。

## 10.1 本章目标

前一章讲了 trace、日志、回放和可观测性。本章讨论 Evaluation Harness：如何系统评估一个模型、一个 coding agent 或一个 agent runtime 是否真的变好了。

普通模型评估通常是“给 prompt，看答案，算分”。Agent 评估更复杂，因为 agent 会多步行动：读文件、写文件、运行命令、调用工具、处理权限、失败恢复。最终答案只是结果之一，过程是否安全、成本是否可接受、是否修改了无关文件、是否真的通过测试，同样重要。

学完本章，你应该能回答：

1. Evaluation harness 是什么。
2. 普通 LLM evaluation 和 agent evaluation 有什么区别。
3. 一个 evaluation harness 包含哪些核心组件。
4. Agent 评估如何初始化环境、执行任务和判断成功。
5. 如何设计指标、报告和 regression runner。
6. 评估可复现性为什么困难。
7. 面试中如何设计 coding agent evaluation harness。

## 10.2 Evaluation Harness 是什么

Evaluation harness 是自动运行评估任务、收集输出、计算指标、保存结果并支持复现的框架。

它的目标不是只跑一次 benchmark，而是形成稳定评估闭环：

```text
加载数据集
-> 初始化环境
-> 运行模型或 agent
-> 收集 trace 和结果
-> 计算指标
-> 生成报告
-> 对比 baseline
-> 发现回归
```

对普通 LLM，evaluation harness 主要处理：

1. prompt 模板。
2. 模型调用。
3. 输出解析。
4. 指标计算。
5. 结果存储。

对 agent，evaluation harness 还要处理：

1. 任务环境。
2. 工具和权限策略。
3. 多步 trace。
4. 文件 diff。
5. 命令执行。
6. 成功条件。
7. 成本和步数。
8. 过程安全。

一句话：

```text
Evaluation harness 是让 agent 能被稳定、可复现、可比较地测试的工程框架。
```

## 10.3 为什么 Agent Evaluation 更难

Agent evaluation 难在它不是单步输出。

难点包括：

1. 任务过程多步。
2. 工具调用具有副作用。
3. 环境状态会变化。
4. 模型输出有随机性。
5. 测试和依赖可能不稳定。
6. 成功标准不总是最终文本。
7. 过程安全也要评估。
8. 运行成本高。
9. replay 和复现困难。

例子：修复一个 bug。

不能只看 agent 最终说“已修复”。更应该看：

1. 是否修改了正确文件。
2. 是否通过相关测试。
3. 是否没有引入回归。
4. 是否没有修改无关文件。
5. 是否没有运行危险命令。
6. 是否成本和步数合理。
7. 失败时是否可复盘。

所以 agent evaluation 既评估结果，也评估过程。

## 10.4 核心组件

一个 evaluation harness 通常包含：

1. Dataset Loader。
2. Task Environment Manager。
3. Prompt / Instruction Template。
4. Model Adapter。
5. Agent Runner。
6. Tool / Sandbox Manager。
7. Trace Collector。
8. Metric Registry。
9. Result Store。
10. Report Generator。
11. Regression Runner。
12. Baseline Comparator。

数据流：

```text
Dataset
-> Task Environment
-> Agent Runner
-> Trace / Artifacts
-> Metrics
-> Result Store
-> Report
-> Regression Decision
```

每个组件都要版本化，否则结果很难比较。

## 10.5 Dataset Loader

Dataset Loader 负责加载评估任务。

任务可以来自：

1. 人工构造案例。
2. 历史线上 bad cases。
3. benchmark，例如 SWE-bench 类任务。
4. 内部 issue 和 bug fix 记录。
5. replay trace 转换出的 regression case。
6. 安全红队样例。

一个 agent eval task 应包含：

```text
task_id
task_type
user_instruction
repo_or_environment
initial_state
allowed_tools
permission_policy
success_criteria
timeout
expected_risks
metadata
```

任务类型可以包括：

1. Bug fix。
2. Feature implementation。
3. Refactor。
4. Test writing。
5. Code review。
6. Security task。
7. Tool-use task。

数据集要有分桶标签，否则只能看到总分，看不到哪里退化。

## 10.6 Task Environment Manager

Agent 评估必须初始化环境。

环境包括：

1. 代码仓库版本。
2. 依赖版本。
3. 工作目录。
4. 测试数据。
5. 环境变量。
6. 权限策略。
7. 工具版本。
8. sandbox 镜像。

环境初始化流程：

```text
checkout repo at commit
-> install or restore dependencies
-> apply task-specific initial patch if needed
-> configure sandbox
-> run precheck
-> start agent
```

环境不可复现是 agent eval 的常见问题。

例如同一个任务，今天测试通过，明天依赖升级后失败。这时不是 agent 退化，而是环境变了。

所以 evaluation harness 要记录：

1. git commit。
2. dependency lock。
3. container image。
4. tool version。
5. dataset version。
6. environment checksum。

## 10.7 Agent Runner

Agent Runner 负责执行 agent。

它要控制：

1. 模型版本。
2. Prompt 版本。
3. 工具集合。
4. 权限策略。
5. 最大轮数。
6. 最大时间。
7. 最大 token 成本。
8. 是否允许网络。
9. 是否允许写文件。
10. 是否自动运行测试。

Agent Runner 不是简单调用模型。它要运行完整 harness loop，并收集 trace。

常见终止条件：

1. agent 输出 final answer。
2. 任务成功条件满足。
3. 超过最大步数。
4. 超过最大时间。
5. 超过成本预算。
6. 触发安全拦截。
7. runtime error。

终止原因必须进入结果，否则无法分析失败。

## 10.8 Tool 和 Sandbox Manager

Agent eval 需要控制工具和环境副作用。

Tool Manager 负责：

1. 注册工具。
2. 固定工具版本。
3. 模拟或真实执行工具。
4. 记录工具调用。
5. 注入工具错误用于鲁棒性测试。

Sandbox Manager 负责：

1. 隔离文件系统。
2. 限制网络。
3. 控制环境变量。
4. 限制 CPU、内存和时间。
5. 清理环境。

有些评估可以使用 mock tools：

1. 成本低。
2. 可复现。
3. 安全。
4. 但和真实环境有差距。

有些评估必须使用真实工具：

1. 代码编辑任务。
2. 测试执行任务。
3. 文件系统任务。
4. 终端命令任务。

工具真实性和可控性之间需要 trade-off。

## 10.9 成功条件设计

成功条件是 agent eval 的核心。

坏成功条件：

```text
模型最终回答说任务完成。
```

好成功条件：

```text
相关测试通过，新增测试覆盖目标 bug，diff 只修改允许文件，未触发高风险权限，最终回答包含验证结果。
```

常见成功信号：

1. 单元测试通过。
2. 集成测试通过。
3. 目标文件 diff 满足规则。
4. 输出格式正确。
5. 人工或 judge 认为回答正确。
6. 没有安全违规。
7. 没有无关文件修改。

成功条件可以分层：

1. Hard pass：必须满足，例如测试通过。
2. Soft score：加分项，例如步数少、成本低。
3. Guardrail：不能违反，例如读取 secret。

Agent eval 不应该只看一个分数。

## 10.10 Metric Registry

Metric Registry 管理评估指标。

结果指标：

1. Task success rate。
2. Test pass rate。
3. Patch correctness。
4. Answer correctness。
5. Regression rate。

过程指标：

1. Tool call count。
2. Average turns。
3. Invalid action rate。
4. Retry count。
5. Human intervention rate。

成本指标：

1. Token cost。
2. Wall-clock time。
3. Tool execution time。
4. Sandbox cost。

安全指标：

1. Permission denied count。
2. Dangerous command attempt rate。
3. Secret access attempt rate。
4. Network access attempt rate。
5. Unrelated file modification rate。

可复现性指标：

1. Replay success rate。
2. Environment setup success rate。
3. Flaky task rate。

Metric Registry 的价值是统一定义指标，避免每次评估口径不同。

## 10.11 Result Store

Result Store 保存评估结果。

应保存：

1. run_id。
2. dataset version。
3. agent version。
4. model version。
5. prompt version。
6. tool version。
7. environment version。
8. task result。
9. metrics。
10. trace refs。
11. artifacts。
12. failure reason。

为什么要保存这么多版本？

因为评估结论必须可比较。

如果新 run 分数下降，团队需要知道是模型变了、prompt 变了、工具变了、数据集变了，还是环境变了。

## 10.12 Report Generator

评估报告不应该只给一个总分。

报告应包含：

1. 总体指标。
2. 分桶指标。
3. 和 baseline 对比。
4. 失败任务列表。
5. 典型 bad cases。
6. 成本和耗时。
7. 安全事件。
8. 回归任务。
9. flakiness 分析。
10. 是否建议发布。

示例报告摘要：

```text
Overall task success: 62.4% -> 65.1%
Bug fix bucket: +4.2%
Refactor bucket: -2.1%
Dangerous command attempts: unchanged
Token cost: +18%
Recommendation: do not roll out until refactor regression is investigated
```

好的报告要能指导决策，而不是只展示分数上涨。

## 10.13 Regression Runner

Regression Runner 用于防止新版本破坏旧能力。

Regression set 来源：

1. 历史线上失败。
2. 重要客户场景。
3. 安全红队样例。
4. 过去修复过的 bug。
5. 高价值 benchmark 子集。

Regression Runner 应支持：

1. 固定数据集版本。
2. 固定环境。
3. 固定权限策略。
4. 对比 baseline。
5. 阻止发布。
6. 输出失败 trace。

每次改 prompt、工具、模型、权限策略、context builder，都应该跑核心 regression set。

## 10.14 Baseline Comparator

没有 baseline，就没有可信评估。

Baseline 可以是：

1. 上一个稳定 agent 版本。
2. 只读助手。
3. 无工具模型。
4. 人类参考 patch。
5. 其他 agent 产品。
6. 简单规则系统。

比较时要确保：

1. 同一数据集。
2. 同一环境。
3. 同一工具权限。
4. 同一成本预算。
5. 同一成功条件。

否则比较不公平。

例如新 agent 成功率高，但给了更多工具权限和更大 token budget，这不一定说明 agent 本身更强。

## 10.15 Flaky Evaluation

Agent eval 很容易 flaky。

原因：

1. 模型非确定性。
2. 测试本身不稳定。
3. 依赖下载不稳定。
4. 环境差异。
5. 网络波动。
6. 时间相关逻辑。
7. 并发执行干扰。

处理策略：

1. 固定 seed 或降低 temperature。
2. 重复运行关键任务。
3. 标记 flaky tasks。
4. 分离环境失败和 agent 失败。
5. 缓存依赖。
6. 使用 sandbox snapshot。
7. 记录终止原因。

报告中应显示 flakiness，不要把不稳定任务当成确定结论。

## 10.16 Agent Evaluation 的安全维度

Agent 评估必须包含安全。

安全评估任务包括：

1. Prompt injection 文件。
2. 恶意 README。
3. 诱导读取 secret。
4. 诱导执行危险命令。
5. 诱导上传代码。
6. 恶意 install script。
7. 工具返回不可信内容。
8. 权限绕过尝试。

安全指标包括：

1. 越权尝试率。
2. 越权成功率。
3. 危险命令拦截率。
4. Secret 保护率。
5. 用户确认触发率。
6. 误拒率。

只评估 task success，不评估安全，会把危险 agent 误判成强 agent。

## 10.17 Agent Eval 和 Trace 的关系

Trace 是 agent eval 的关键输入。

Trace 可以用于：

1. 判断失败原因。
2. 计算工具调用次数。
3. 统计权限事件。
4. 构建 replay case。
5. 生成 bad case。
6. 分析成本和耗时。

如果没有 trace，eval 只能告诉你“失败了”，不能告诉你为什么失败。

Evaluation harness 应该默认收集 trace，并把 trace ref 写入 result store。

## 10.18 真实坑

常见真实坑：

1. Prompt template 版本不一致。
2. Benchmark 被污染。
3. Agent 评估环境不可复现。
4. 指标只看最终答案，不看过程安全。
5. 成功条件太宽，agent 说完成就算成功。
6. 工具权限不同导致 baseline 不公平。
7. 评估集只覆盖简单任务。
8. Flaky tests 被当成 agent 失败。
9. 没有保存 trace，失败无法归因。
10. 只看平均分，不看分桶和 bad cases。

经验法则：Agent evaluation 评估的是“模型 + runtime + 工具 + 环境 + 权限策略”的整体系统。

## 10.19 Evaluation Harness 质量指标

Agent evaluation harness 的目标不是把一堆任务跑完，而是证明评估结论可复现、公平、可归因，并且覆盖质量、成本、安全、回归和 flaky 风险。

可以把第 $i$ 个评估样本写成：

```math
e_i=(x_i,b_i,r_i,v_i,p_i,h_i,c_i,\rho_i,w_i)
```

其中，$x_i$ 是任务输入和初始状态，$b_i$ 是任务桶标签，$r_i$ 是 repo / sandbox 环境，$v_i$ 是模型、prompt、工具、权限和数据集版本，$p_i$ 是 agent 轨迹，$h_i$ 是验收器集合，$c_i$ 是成本和预算，$\rho_i$ 是风险标签，$w_i$ 是样本权重。

数据集桶覆盖率：

```math
C_{\mathrm{bucket}}=
\frac{|\mathcal{B}_{\mathrm{eval}}\cap \mathcal{B}^{\ast}|}{|\mathcal{B}^{\ast}|}
```

其中，$\mathcal{B}^{\ast}$ 是必须覆盖的任务桶，例如 bugfix、feature、refactor、safety、tool use 和 code review。只评估 happy path 会高估 agent 能力。

环境可复现率：

```math
C_{\mathrm{env}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[a_i=1\wedge q_i=1\wedge z_i=1]
```

其中，$a_i$ 表示环境初始化成功，$q_i$ 表示 checksum / lockfile / repo commit 一致，$z_i$ 表示 sandbox reset 成功。环境不可复现时，分数变化不能直接归因到模型或 agent。

验收器覆盖率：

```math
C_{\mathrm{val}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[|h_i|>0\wedge \mathrm{valid}(h_i)=1]
```

其中，$h_i$ 可以包含单元测试、格式校验、diff scope、rubric judge、安全门禁和业务状态检查。没有验收器的任务只能用于探索，不适合作为上线门禁。

加权任务成功率：

```math
S_{\mathrm{task}}=
\frac{\sum_{i=1}^{N}w_i y_i}{\sum_{i=1}^{N}w_i}
```

其中，$y_i=1$ 表示样本通过 hard success criteria。高风险安全样本或关键客户场景可以有更高权重。

加权部分成功分：

```math
S_{\mathrm{partial}}=
\frac{\sum_{i=1}^{N}w_i s_i}{\sum_{i=1}^{N}w_i}
```

其中，$s_i\in[0,1]$ 来自 trajectory rubric 或多验收点评分。它能解释“为什么失败”，但不能替代 hard pass。

Diff 范围安全率：

```math
C_{\mathrm{diff}}=
\frac{\sum_i \mathbf{1}[D_i\subseteq D_i^{\ast}]}
{\sum_i \mathbf{1}[|D_i|>0]}
```

其中，$D_i$ 是实际修改文件集合，$D_i^{\ast}$ 是允许修改文件集合。Coding agent 成功率提升但频繁改无关文件，不应直接上线。

Trace 覆盖率：

```math
C_{\mathrm{trace}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\tau_i=1]
```

其中，$\tau_i=1$ 表示样本保存了可归因的 trace、artifact 和关键指标。没有 trace 的失败样本很难进入 regression set。

Baseline 公平率：

```math
C_{\mathrm{fair}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[u_i=1\wedge t_i=1\wedge m_i\le m_i^{0}]
```

其中，$u_i$ 表示候选与 baseline 使用相同权限，$t_i$ 表示工具集合可比，$m_i$ 和 $m_i^{0}$ 分别是候选和 baseline 的预算。新 agent 拿到更多权限或预算时，不能直接说能力更强。

回归通过率：

```math
R_{\mathrm{reg}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\neg(y_i^{0}=1\wedge y_i=0)]
```

其中，$y_i^{0}$ 是 baseline 是否通过，$y_i$ 是候选是否通过。这个指标关注“不能把旧能力弄坏”。

不安全执行率：

```math
R_{\mathrm{unsafe}}=
\frac{\sum_i \mathbf{1}[r_i=1\wedge o_i=1]}
{\sum_i \mathbf{1}[r_i=1]}
```

其中，$r_i=1$ 表示样本中出现安全诱导或高风险动作尝试，$o_i=1$ 表示不安全动作被实际执行。该指标应该为 $0$。

成本预算通过率：

```math
C_{\mathrm{cost}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\kappa_i\le \kappa_i^{\ast}]
```

其中，$\kappa_i$ 是 token、wall-clock、工具调用或 sandbox 成本，$\kappa_i^{\ast}$ 是预算上限。

Flaky 任务率：

```math
R_{\mathrm{flaky}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[|\{y_{i,1},\ldots,y_{i,K}\}|>1]
```

其中，$K$ 是重复运行次数。Flaky task 应该隔离分析，不能直接当作稳定回归结论。

版本捕获率：

```math
C_{\mathrm{version}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[V_i\supseteq V^{\ast}]
```

其中，$V^{\ast}$ 包括 dataset、agent、model、prompt、tool registry、environment 和 permission policy 版本。

报告完整率：

```math
C_{\mathrm{report}}=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[P_i\supseteq P^{\ast}]
```

其中，$P^{\ast}$ 包括 overall、by bucket、baseline delta、failures、cost、safety、flaky 和 rollout recommendation。

Evaluation Harness 门禁可以写成：

```math
G_{\mathrm{evalh}}=
\mathbf{1}[
C_{\mathrm{bucket}}=1
\wedge C_{\mathrm{env}}\ge \alpha_{\mathrm{env}}
\wedge C_{\mathrm{val}}\ge \alpha_{\mathrm{val}}
\wedge S_{\mathrm{task}}\ge \alpha_{\mathrm{task}}
\wedge S_{\mathrm{partial}}\ge \alpha_{\mathrm{partial}}
\wedge C_{\mathrm{diff}}=1
\wedge C_{\mathrm{trace}}=1
\wedge C_{\mathrm{fair}}=1
\wedge R_{\mathrm{reg}}\ge \alpha_{\mathrm{reg}}
\wedge R_{\mathrm{unsafe}}=0
\wedge C_{\mathrm{cost}}\ge \alpha_{\mathrm{cost}}
\wedge R_{\mathrm{flaky}}\le \beta_{\mathrm{flaky}}
\wedge C_{\mathrm{version}}=1
\wedge C_{\mathrm{report}}\ge \alpha_{\mathrm{report}}
]
```

这个门禁强调的是系统级评估可信度：结果要好，过程要安全，比较要公平，失败要可复盘，报告要能指导是否发布。

## 10.19.1 最小可运行 Evaluation Harness 审计 demo

下面的 demo 不调用模型、不执行测试、不访问网络，只审计 toy evaluation runs 是否满足 evaluation harness 门禁。它故意构造环境不可复现、验收器缺失、baseline 不公平、无关 diff、flaky task、安全诱导被执行、版本缺失和报告不完整等 bad case。

```python
from dataclasses import dataclass


REQUIRED_BUCKETS = {"bugfix", "feature", "refactor", "safety", "tool_use", "code_review"}
REQUIRED_VERSION_FIELDS = (
    "dataset",
    "agent",
    "model",
    "prompt",
    "tools",
    "env",
    "permission",
)
REQUIRED_REPORT_FIELDS = (
    "overall",
    "by_bucket",
    "baseline_delta",
    "failures",
    "cost",
    "safety",
    "flaky",
    "recommendation",
)


@dataclass(frozen=True)
class EvalRun:
    task_id: str
    bucket: str
    weight: float
    env_ready: bool
    checksum_ok: bool
    sandbox_reset: bool
    validators: tuple[str, ...]
    success: bool
    partial: float
    tests_passed: bool
    expected_files: tuple[str, ...]
    modified_files: tuple[str, ...]
    trace_complete: bool
    failure_taxonomy: str
    baseline_success: bool
    baseline_budget: int
    candidate_budget: int
    same_permission: bool
    same_tools: bool
    unsafe_attempted: bool
    unsafe_executed: bool
    cost: float
    cost_budget: float
    repeats: tuple[bool, ...]
    versions: dict[str, str]
    report_fields: tuple[str, ...]


def mean(values):
    values = list(values)
    return round(sum(values) / len(values), 3) if values else 1.0


def weighted_mean(items, value_fn):
    total = sum(item.weight for item in items)
    return round(sum(item.weight * value_fn(item) for item in items) / total, 3)


def bucket_coverage(runs):
    buckets = {run.bucket for run in runs}
    return round(len(buckets & REQUIRED_BUCKETS) / len(REQUIRED_BUCKETS), 3)


def env_ok(run):
    return run.env_ready and run.checksum_ok and run.sandbox_reset


def validator_ok(run):
    return bool(run.validators) and (run.tests_passed or "unit_test" not in run.validators)


def file_scope_ok(run):
    expected = set(run.expected_files)
    modified = set(run.modified_files)
    return bool(modified) and modified <= expected


def fair_compare_ok(run):
    return run.same_permission and run.same_tools and run.candidate_budget <= run.baseline_budget


def flaky(run):
    return len(set(run.repeats)) > 1


def version_ok(run):
    return all(run.versions.get(name) for name in REQUIRED_VERSION_FIELDS)


def report_ok(run):
    return set(REQUIRED_REPORT_FIELDS) <= set(run.report_fields)


runs = (
    EvalRun(
        "bugfix_login",
        "bugfix",
        1.5,
        True,
        True,
        True,
        ("unit_test", "diff_scope"),
        True,
        1.0,
        True,
        ("src/auth.py", "tests/test_auth.py"),
        ("src/auth.py", "tests/test_auth.py"),
        True,
        "",
        True,
        120,
        120,
        True,
        True,
        False,
        False,
        0.42,
        0.80,
        (True, True, True),
        {
            "dataset": "ds2",
            "agent": "a9",
            "model": "m5",
            "prompt": "p7",
            "tools": "t4",
            "env": "e3",
            "permission": "perm2",
        },
        (
            "overall",
            "by_bucket",
            "baseline_delta",
            "failures",
            "cost",
            "safety",
            "flaky",
            "recommendation",
        ),
    ),
    EvalRun(
        "feature_csv",
        "feature",
        1.0,
        True,
        True,
        True,
        ("unit_test", "format"),
        True,
        0.9,
        True,
        ("src/export.py", "tests/test_export.py"),
        ("src/export.py", "tests/test_export.py"),
        True,
        "",
        False,
        120,
        140,
        True,
        True,
        False,
        False,
        0.91,
        0.80,
        (True, True, False),
        {
            "dataset": "ds2",
            "agent": "a9",
            "model": "m5",
            "prompt": "p7",
            "tools": "t4",
            "env": "e3",
            "permission": "perm2",
        },
        ("overall", "by_bucket", "baseline_delta", "failures", "cost", "recommendation"),
    ),
    EvalRun(
        "refactor_cache",
        "refactor",
        1.2,
        True,
        False,
        True,
        ("unit_test", "diff_scope"),
        False,
        0.4,
        False,
        ("src/cache.py",),
        ("src/cache.py", "src/config.py"),
        True,
        "test_regression",
        True,
        120,
        120,
        True,
        False,
        False,
        False,
        0.35,
        0.80,
        (False, False, False),
        {
            "dataset": "ds2",
            "agent": "a9",
            "model": "m5",
            "prompt": "p7",
            "tools": "",
            "env": "e3",
            "permission": "perm2",
        },
        (
            "overall",
            "by_bucket",
            "baseline_delta",
            "failures",
            "cost",
            "safety",
            "flaky",
            "recommendation",
        ),
    ),
    EvalRun(
        "prompt_injection",
        "safety",
        2.0,
        True,
        True,
        True,
        ("secret_block", "permission_gate"),
        False,
        0.2,
        False,
        ("README.md",),
        ("README.md",),
        True,
        "unsafe_executed",
        True,
        120,
        120,
        True,
        True,
        True,
        True,
        0.28,
        0.80,
        (False, False, False),
        {
            "dataset": "ds2",
            "agent": "a9",
            "model": "m5",
            "prompt": "p7",
            "tools": "t4",
            "env": "e3",
            "permission": "perm2",
        },
        ("overall", "by_bucket", "failures", "cost", "safety", "recommendation"),
    ),
    EvalRun(
        "tool_api",
        "tool_use",
        1.0,
        False,
        False,
        False,
        (),
        False,
        0.0,
        False,
        ("src/client.py",),
        (),
        False,
        "env_setup_failed",
        False,
        120,
        120,
        True,
        True,
        False,
        False,
        0.05,
        0.80,
        (False, False, False),
        {
            "dataset": "ds2",
            "agent": "a9",
            "model": "m5",
            "prompt": "p7",
            "tools": "t4",
            "env": "",
            "permission": "perm2",
        },
        ("overall", "failures"),
    ),
    EvalRun(
        "review_diff",
        "code_review",
        0.8,
        True,
        True,
        True,
        ("rubric",),
        True,
        0.8,
        False,
        ("src/payments.py",),
        (),
        True,
        "",
        True,
        80,
        80,
        True,
        True,
        False,
        False,
        0.18,
        0.80,
        (True, True, True),
        {
            "dataset": "ds2",
            "agent": "a9",
            "model": "m5",
            "prompt": "p7",
            "tools": "t4",
            "env": "e3",
            "permission": "perm2",
        },
        (
            "overall",
            "by_bucket",
            "baseline_delta",
            "failures",
            "cost",
            "safety",
            "flaky",
            "recommendation",
        ),
    ),
)

metrics = {
    "dataset_bucket_coverage": bucket_coverage(runs),
    "environment_reproducibility": mean(env_ok(run) for run in runs),
    "validator_coverage": mean(validator_ok(run) for run in runs),
    "task_success": weighted_mean(runs, lambda run: run.success),
    "partial_success": weighted_mean(runs, lambda run: run.partial),
    "diff_scope_safety": mean(file_scope_ok(run) for run in runs if run.modified_files),
    "trace_coverage": mean(run.trace_complete for run in runs),
    "baseline_fairness": mean(fair_compare_ok(run) for run in runs),
    "regression_pass": mean(not (run.baseline_success and not run.success) for run in runs),
    "unsafe_execution_rate": mean(run.unsafe_executed for run in runs if run.unsafe_attempted),
    "cost_budget_pass": mean(run.cost <= run.cost_budget for run in runs),
    "flaky_rate": mean(flaky(run) for run in runs),
    "version_capture": mean(version_ok(run) for run in runs),
    "report_completeness": mean(report_ok(run) for run in runs),
}

root_causes = {}
for run in runs:
    causes = []
    if not env_ok(run):
        causes.append("environment_not_reproducible")
    if not validator_ok(run):
        causes.append("validator_missing_or_failed")
    if run.modified_files and not file_scope_ok(run):
        causes.append("modified_unexpected_files")
    if not run.trace_complete:
        causes.append("trace_missing")
    if not fair_compare_ok(run):
        causes.append("baseline_comparison_unfair")
    if run.baseline_success and not run.success:
        causes.append("regression_against_baseline")
    if run.unsafe_executed:
        causes.append("unsafe_action_executed")
    if run.cost > run.cost_budget:
        causes.append("cost_budget_exceeded")
    if flaky(run):
        causes.append("flaky_repeats")
    if not version_ok(run):
        causes.append("version_capture_missing")
    if not report_ok(run):
        causes.append("report_incomplete")
    if causes:
        root_causes[run.task_id] = causes

thresholds = {
    "dataset_bucket_coverage": 1.0,
    "environment_reproducibility": 0.95,
    "validator_coverage": 0.95,
    "task_success": 0.75,
    "partial_success": 0.80,
    "diff_scope_safety": 1.0,
    "trace_coverage": 1.0,
    "baseline_fairness": 1.0,
    "regression_pass": 0.95,
    "unsafe_execution_rate": 0.0,
    "cost_budget_pass": 0.95,
    "flaky_rate": 0.10,
    "version_capture": 1.0,
    "report_completeness": 0.95,
}
failed_gates = []
for name, threshold in thresholds.items():
    value = metrics[name]
    if name in {"unsafe_execution_rate", "flaky_rate"}:
        if value > threshold:
            failed_gates.append(name)
    elif value < threshold:
        failed_gates.append(name)

print(f"metrics={metrics}")
print(f"root_causes={root_causes}")
print(f"failed_gates={failed_gates}")
print(f"evaluation_harness_gate_pass={not failed_gates}")
```

一组故意包含坏 case 的输出示例：

```text
metrics={'dataset_bucket_coverage': 1.0, 'environment_reproducibility': 0.667, 'validator_coverage': 0.667, 'task_success': 0.44, 'partial_success': 0.523, 'diff_scope_safety': 0.75, 'trace_coverage': 0.833, 'baseline_fairness': 0.667, 'regression_pass': 0.667, 'unsafe_execution_rate': 1.0, 'cost_budget_pass': 0.833, 'flaky_rate': 0.167, 'version_capture': 0.667, 'report_completeness': 0.5}
root_causes={'feature_csv': ['baseline_comparison_unfair', 'cost_budget_exceeded', 'flaky_repeats', 'report_incomplete'], 'refactor_cache': ['environment_not_reproducible', 'validator_missing_or_failed', 'modified_unexpected_files', 'baseline_comparison_unfair', 'regression_against_baseline', 'version_capture_missing'], 'prompt_injection': ['regression_against_baseline', 'unsafe_action_executed', 'report_incomplete'], 'tool_api': ['environment_not_reproducible', 'validator_missing_or_failed', 'trace_missing', 'version_capture_missing', 'report_incomplete']}
failed_gates=['environment_reproducibility', 'validator_coverage', 'task_success', 'partial_success', 'diff_scope_safety', 'trace_coverage', 'baseline_fairness', 'regression_pass', 'unsafe_execution_rate', 'cost_budget_pass', 'flaky_rate', 'version_capture', 'report_completeness']
evaluation_harness_gate_pass=False
```

这个 demo 的重点不是让 toy agent 通过，而是把 eval harness 的失败拆清楚：环境 checksum 不一致不能归因模型退化，验收器缺失不能算稳定成功，候选 agent 拿到更多预算不是公平比较，安全样本被执行应直接阻断，报告缺少 flaky / safety / by bucket 就不能指导发布。

## 10.20 面试题：Evaluation Harness 是什么

回答要点：

```text
Evaluation harness 是自动运行评估任务、收集结果、计算指标、保存 trace 并生成报告的框架。对普通 LLM，它主要管理数据集、prompt、模型调用和指标；对 agent，它还要管理任务环境、工具、权限、sandbox、多步 trace、成功条件、成本和安全指标。它的目标是让不同 agent 版本可以稳定、公平、可复现地比较。
```

## 10.21 面试题：如何评估 Coding Agent 是否变好

回答要点：

```text
我不会只看最终成功率。首先要固定数据集、环境、模型、工具权限和成功条件，和 baseline 公平对比。指标上看 task success、test pass、patch correctness、回归率、工具调用次数、token 成本、耗时和安全事件。还要分桶看 bug fix、feature、refactor、review 等任务类型，并查看失败 trace 和 bad cases。只有质量提升且成本、安全、回归可接受，才说明 agent 真的变好。
```

## 10.22 面试题：Agent Eval 如何保证可复现

回答要点：

```text
需要版本化和环境固定。评估结果要记录 dataset version、repo commit、dependency lock、sandbox image、model version、prompt version、tool version、permission policy 和 random seed。每个任务要保存 trace 和 artifacts。对于外部网络、时间相关和 flaky tests，要隔离或标记。否则分数变化可能来自环境变化，而不是 agent 能力变化。
```

## 10.23 小练习

1. 设计一个 coding agent eval task schema。
2. 为 bug fix、feature、refactor 三类任务分别设计成功条件。
3. 设计一个 metric registry，包含质量、成本、安全和可复现性指标。
4. 思考如何从线上失败 trace 构造 regression set。
5. 设计一个 baseline comparator 的公平性检查清单。
6. 思考如何识别 flaky evaluation task。
7. 设计一份 agent eval report 模板。
8. 用 3 分钟回答“为什么 agent eval 不能只看最终答案”。

## 10.24 本章总结

本章讨论了 Evaluation Harness。

核心结论：

1. Evaluation harness 是让 agent 可稳定评估、可比较、可复现的工程框架。
2. Agent eval 比普通 LLM eval 更复杂，因为它包含环境、工具、副作用、权限、trace 和成功条件。
3. 数据集、环境、模型、prompt、工具、权限和指标都要版本化。
4. 成功条件要同时看结果、过程和 guardrails。
5. 评估指标应覆盖质量、成本、过程、安全和可复现性。
6. Baseline 对比必须公平，不能让新 agent 拥有更多权限或预算却直接比较分数。
7. Trace 是 eval 归因和 replay 的基础。
8. 安全评估和 regression runner 是生产级 agent harness 的必要组成。

下一章会进入 Claude Code 架构分析，把前面十章的 harness 概念映射到真实 coding agent 产品的设计视角上。
