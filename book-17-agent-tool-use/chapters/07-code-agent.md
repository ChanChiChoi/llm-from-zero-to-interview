# 第七章：Code Agent

Code Agent 是 Agent 最重要的落地形态之一。它不只是生成一段代码，而是能阅读仓库、理解任务、定位文件、生成 patch、运行测试、根据错误反馈继续调试，并最终给出可验证结果。

相比普通代码补全，Code Agent 更像一个受控的初级工程师：能操作工具，但必须遵守仓库边界、权限边界、最小修改原则和验证要求。它的质量不能只看代码“看起来对不对”，而要看任务是否真的完成、测试是否通过、改动是否聚焦、是否保护用户已有修改、是否可审计。

本章系统讲 Code Agent：仓库理解、代码定位、文件编辑、patch 生成、测试执行、调试闭环、最小修改原则、上下文管理、依赖变更、权限与沙箱、安全风险、评估指标，以及一个 0 依赖 Python demo，用来审计 toy Code Agent 轨迹。

## 0. 本讲范围与资料

本章参考了 SWE-bench、SWE-agent、OpenAI Codex CLI / local shell / apply patch 相关公开文档、Claude Code 概览和代码任务常见评估口径。

本章采用以下口径：

1. Code Agent 是“代码仓库中的任务执行系统”，不是普通代码补全。
2. 可靠 Code Agent 必须同时具备仓库理解、文件定位、patch 编辑、命令执行、测试反馈、状态记忆、权限控制和 trace 审计。
3. Code Agent 的核心目标是用最小必要修改完成用户目标，并用测试、构建、类型检查、lint 或人工可审 diff 验证结果。
4. 高风险命令、依赖变更、锁文件变更、权限配置、用户未提交改动和敏感文件都必须受控。
5. 本章只讨论防御性工程设计、质量评估和教学 demo，不提供绕过沙箱、规避权限、破坏文件或执行高风险操作的方法。

## 7.1 Code Agent 是什么

Code Agent 是围绕代码任务执行的 Agent。

它通常能做：

1. 阅读文件。
2. 搜索符号。
3. 理解项目结构。
4. 修改代码。
5. 运行测试。
6. 分析报错。
7. 迭代修复。
8. 总结改动。

Code Agent 的核心不是“会写代码”，而是能把自然语言任务转成一组可验证的仓库状态变化。它要先确定任务边界，再建立对仓库结构、入口、约定和当前工作区的认识；随后提出一个小补丁，运行与任务相关的验证，并根据真实 observation 判断是否需要继续。最后交付的对象不是一段孤立代码，而是 diff、测试结果、未解决风险和环境限制组成的结果包。

可以把一次代码任务看成一个状态转移：

~~~text
task -> repository_observation -> hypothesis -> patch -> execution_feedback
     -> revised_hypothesis -> validated_diff -> handoff
~~~

这里的 `repository_observation` 不只是目录列表，还包括目标文件当前内容、版本控制状态、测试入口、依赖约束和用户已经留下的改动；`execution_feedback` 包括测试失败、类型错误、运行时输出、退出码和副作用。没有这些中间状态，模型即使生成了语法正确的代码，也无法知道它是否真正满足用户目标。

## 7.2 Code Agent 和代码补全的区别

代码补全通常是局部生成：

~~~text
当前文件上下文 -> 补全下一段代码
~~~

Code Agent 是任务执行：

~~~text
用户目标 -> 理解仓库 -> 定位问题 -> 修改文件 -> 运行测试 -> 修复反馈 -> 总结
~~~

核心区别：

1. 代码补全关注局部上下文，Code Agent 关注仓库级任务。
2. 代码补全通常不执行工具，Code Agent 会运行命令和测试。
3. 代码补全生成代码，Code Agent 管理任务状态。
4. Code Agent 需要安全边界、权限控制和 trace。
5. Code Agent 的输出不是代码文本，而是“经过验证的 diff + 结果说明”。

## 7.3 关键公式与 Code Agent 指标速查

设用户任务为 `g`，仓库初始状态为 `R_0`，Code Agent 的执行轨迹可以写成：

~~~math
\tau=(g,R_0,s_0,a_1,o_1,s_1,\ldots,a_T,o_T,s_T,\Delta,\hat y)
~~~

其中 `s_t` 是任务状态，`a_t` 是第 `t` 步动作，`o_t` 是工具 observation，`\Delta` 是最终代码 diff，`\hat y` 是最终总结。

一次代码动作可以抽象为：

~~~math
a_t=(u_t,n_t,\alpha_t,\rho_t)
~~~

其中 `u_t` 是动作类型，例如 `search`、`read`、`patch`、`test`、`ask`、`final`；`n_t` 是工具名；`\alpha_t` 是参数；`\rho_t` 是风险级别。

执行动作前需要沙箱和权限检查：

~~~math
I_{\mathrm{cmd}}(a_t,s_t)=
I_{\mathrm{schema}}(a_t)\cdot
I_{\mathrm{scope}}(a_t,s_t)\cdot
I_{\mathrm{perm}}(a_t,s_t)\cdot
I_{\mathrm{budget}}(a_t,s_t)\cdot
I_{\mathrm{risk}}(a_t,s_t)
~~~

每个检查结果都可能是 `1`、`0` 或 `unknown`。只有所有检查已经完成
且为 `1` 时，`I_cmd=1`，动作才进入执行器；任何已测量的 `0` 都应
拒绝执行；存在 `unknown` 时保持未知，不能默认放行。高风险动作应被
拦截、降级为只读操作或请求用户确认；模型生成了一个合法的 shell
字符串，并不意味着它已经获得执行授权。

目标需求集合：

~~~math
\mathcal{R}_g=\{r_1,\ldots,r_m\}
~~~

Code Agent 最终修改的文件集合：

~~~math
\mathcal{F}_{\Delta}=\{f_1,\ldots,f_k\}
~~~

任务相关文件集合：

~~~math
\mathcal{F}_{\mathrm{rel}}=\{f:f\ \mathrm{is\ relevant\ to}\ g\}
~~~

Patch 定位 precision：

~~~math
P_{\mathrm{loc}}=
\frac{|\mathcal{F}_{\Delta}\cap\mathcal{F}_{\mathrm{rel}}|}
{|\mathcal{F}_{\Delta}|}
~~~

Patch 定位 recall：

~~~math
R_{\mathrm{loc}}=
\frac{|\mathcal{F}_{\Delta}\cap\mathcal{F}_{\mathrm{rel}}|}
{|\mathcal{F}_{\mathrm{rel}}|}
~~~

无关改动率：

~~~math
R_{\mathrm{unrel}}=
\frac{|\mathcal{F}_{\Delta}\setminus\mathcal{F}_{\mathrm{rel}}|}
{|\mathcal{F}_{\Delta}|}
~~~

测试通过率（要求至少运行一条测试）：

~~~math
R_{\mathrm{test}}=
\frac{\sum_i \mathbf{1}[\mathrm{test}_i\ \mathrm{passed}]}
{\sum_i \mathbf{1}[\mathrm{test}_i\ \mathrm{run}]}
~~~

验证覆盖率（要求任务集合非空）：

~~~math
R_{\mathrm{val}}=
\frac{\sum_j \mathbf{1}[\mathrm{task}_j\ \mathrm{has\ relevant\ validation}]}
{N}
~~~

用户改动触碰率：

~~~math
R_{\mathrm{user}}=
\frac{\sum_i \mathbf{1}[f_i\in\mathcal{F}_{\Delta}\land f_i\ \mathrm{has\ user\ changes}]}
{|\mathcal{F}_{\Delta}|}
~~~

重复命令率：

~~~math
R_{\mathrm{repeat}}=
\frac{\sum_t \mathbf{1}[a_t\ \mathrm{repeats\ prior\ failed\ command}]}
{T}
~~~

其中 `T>0`；没有命令步骤时，重复命令率为 `None`。同理，`P_loc` 和
`R_unrel` 要求修改文件集合非空，`R_user` 要求修改集合非空，
`R_test` 要求确实运行了测试，`R_val` 要求任务集合 `N>0`。没有
测量到的指标必须保持 `None`，不能用 0 或 1 伪造安全或质量结论。

一个简化的交付检查指标：

~~~math
I_{\mathrm{code}}=
\mathbf{1}[
R_{\mathrm{task}}\ge\tau_{\mathrm{task}}
\land R_{\mathrm{test}}\ge\tau_{\mathrm{test}}
\land R_{\mathrm{val}}\ge\tau_{\mathrm{val}}
\land P_{\mathrm{loc}}\ge\tau_{\mathrm{loc}}
\land R_{\mathrm{unrel}}\le\tau_{\mathrm{unrel}}
\land R_{\mathrm{user}}=0
\land R_{\mathrm{unsafe}}=0
]
~~~

这组条件回答：Code Agent 是否真的完成任务、是否验证、是否改动聚焦、是否保护用户改动、是否遵守安全边界。它不是产品必须采用的一个二值字段，而是把任务成功、验证、范围和安全因素放在同一张审计表中。实际系统可以根据任务风险采用不同阈值：文档格式修复与生产数据库迁移不能共用一套放行逻辑。
这里 `R_task` 表示任务成功率，`R_unsafe` 表示未授权或高风险动作的比例，`\tau` 项是随任务风险设定的阈值。它们应从 trace 和外部验证结果计算，而不是由模型在最终总结中自行填写。
若任一组成指标是 `unknown`/`None`，`I_code` 也应保持未定义；只有
完成全部必要测量后才允许给出通过或不通过的二值判断。

## 7.4 仓库理解

Code Agent 的第一步通常不是改代码，而是理解仓库。

需要了解：

1. 项目语言和框架。
2. 目录结构。
3. 构建和测试命令。
4. 入口文件。
5. 关键模块。
6. 代码风格。
7. 依赖管理方式。
8. 现有测试覆盖。

仓库理解不足时，Agent 容易改错文件、重复实现已有逻辑，或者运行错误命令。

一个可靠的流程是：

~~~text
读目录 -> 搜索相关符号 -> 读目标文件 -> 读相关测试 -> 再决定 patch
~~~

不要在没有读取目标文件当前内容的情况下直接生成 patch。

## 7.5 搜索和定位

Code Agent 必须会搜索代码。

常见搜索目标：

1. 函数定义。
2. 类定义。
3. 错误信息。
4. 配置项。
5. 测试名称。
6. API 调用点。
7. TODO 或注释。
8. 相关文档。

搜索不是越多越好。好的 Agent 会根据任务逐步缩小范围，而不是把全仓库塞进上下文。

定位能力可以用 localization precision / recall 评估：修改的文件是否真的相关，相关文件是否被覆盖。SWE-bench 这类 benchmark 的难点之一就是需要理解真实 issue、跨文件定位和执行反馈，而不是只生成一个函数。

## 7.6 上下文管理

代码仓库通常大于模型上下文窗口。因此 Code Agent 需要选择上下文。

常见策略：

1. 先看目录和关键文件。
2. 根据搜索结果读取相关片段。
3. 保留函数或类的完整上下文。
4. 对无关文件只保留摘要。
5. 修改前重新读取目标文件。
6. 保留测试失败信息。
7. 保留用户明确约束和任务验收标准。

上下文不足会导致误改；上下文过多会导致模型被噪声干扰。上下文管理的目标不是“看得越多越好”，而是让 patch 所需的证据足够完整。

## 7.7 最小修改原则

Code Agent 应该优先做最小正确修改。

原因：

1. 降低引入回归的风险。
2. 便于 review。
3. 便于定位问题。
4. 避免重构超出任务范围。
5. 更符合真实工程协作。

坏做法：用户只要求修一个 bug，Agent 顺手重构半个模块。

好做法：先修复根因，保留现有结构，只在必要时补测试或小范围清理。

最小修改不是“不补测试”。如果 bug 可复现且项目有测试体系，补充回归测试通常是必要的最小验证。

## 7.8 文件编辑和 Patch 生成

文件编辑需要精确。

Code Agent 应避免：

1. 覆盖用户未要求改动。
2. 删除无关代码。
3. 大面积格式化。
4. 改动生成文件。
5. 修改锁文件但不说明原因。
6. 引入未使用依赖。

编辑前要读取当前文件，编辑后最好查看 diff 或运行检查。多 Agent 或用户同时工作时，不能随意回滚别人改动。

Patch 生成时应保留：

1. 修改文件。
2. 修改行数。
3. 修改原因。
4. 对应需求。
5. 是否触碰用户改动。
6. 是否触碰依赖或配置。
7. 验证命令。

这样才能在 trace 中复盘：这次改动是必要修复、测试补充、依赖变更，还是无关漂移。

## 7.9 测试执行

测试是 Code Agent 的核心 verifier。

常见验证：

1. 单元测试。
2. 集成测试。
3. 类型检查。
4. lint。
5. build。
6. 格式检查。
7. 目标命令的 dry run 或小样本检查。

测试策略：

1. 先运行与任务相关的小范围测试。
2. 修复后再运行更大范围测试。
3. 如果全量测试太慢，说明未运行原因。
4. 报错时分析第一处关键失败。

不能只说“应该能通过”，要尽量实际运行可行的验证命令。

## 7.10 Debug 闭环

Code Agent 的典型 debug 闭环：

~~~text
运行测试
读取失败
定位相关代码
提出根因假设
做最小修改
再次运行测试
~~~

关键能力：

1. 从长错误日志中抓关键错误。
2. 区分根因和连锁失败。
3. 不重复无效尝试。
4. 保留失败历史。
5. 根据反馈修正假设。
6. 在无法继续时明确说明阻塞条件。

如果 Agent 每次失败后都随机改代码，就不是可靠 debug。

## 7.11 代码生成

Code Agent 也会生成新代码。

生成时要注意：

1. 遵循项目风格。
2. 复用现有工具函数。
3. 处理边界条件。
4. 保持接口兼容。
5. 补充必要错误处理。
6. 不引入不必要依赖。
7. 保证可测试性。

生成代码前应先检查仓库里是否已有类似实现。重复造轮子会降低可维护性。

## 7.12 代码修改安全

Code Agent 具备写文件和执行命令能力，因此安全很重要。

风险包括：

1. 删除重要文件。
2. 执行高风险命令。
3. 泄露密钥。
4. 修改权限配置。
5. 引入供应链风险。
6. 运行不可信脚本。
7. 破坏用户未提交改动。

安全策略：

1. 高风险命令默认拦截或要求确认。
2. 不读取或输出密钥。
3. 不回滚用户改动。
4. 高风险文件修改需谨慎。
5. 命令执行设置超时。
6. 记录所有工具调用。
7. 对依赖安装、锁文件和生成文件变更做额外说明。

Code Agent 的 shell 能力必须被 controller 管住。模型可以提出动作，但系统负责判断能否执行。

## 7.13 依赖管理

依赖变更是 Code Agent 常见风险点。

增加依赖前要问：

1. 是否已有依赖可以复用。
2. 是否真的需要新库。
3. 是否影响包体积。
4. 是否有安全风险。
5. 是否需要更新锁文件。
6. 是否符合项目技术栈。

依赖选择应该是一个可解释的工程决策，而不是 Agent 遇到缺少 API 就自动安装库。先检查项目已有依赖和标准库能力，再判断新增依赖是否真正减少复杂度；如果必须增加，还要记录版本、许可证、漏洞扫描、锁文件变化、构建影响和回滚方式。一个只改业务逻辑的任务，若最终出现大幅锁文件变化，应该被当成需要复核的异常信号。

依赖变更应该进入单独审计指标，例如 dependency change rate 和 dependency justification coverage。

## 7.14 多文件修改

复杂任务可能需要多文件修改。

例如：

1. 修改接口实现。
2. 更新调用方。
3. 修改测试。
4. 更新类型定义。
5. 更新文档。

多文件修改要保持一致性。Agent 需要追踪哪些文件已改、为什么改、是否还有调用点遗漏。

一个可靠 Code Agent 不应只看最终 diff，还要记录“这个文件为什么在 diff 里”。如果文件和任务没有关系，应该视为风险。

## 7.15 Code Agent 的 Memory

Code Agent 需要记住：

1. 当前任务目标。
2. 相关文件。
3. 已运行命令。
4. 测试结果。
5. 已做修改。
6. 失败尝试。
7. 项目约定。
8. 用户偏好。

这些 memory 主要是短期任务状态。长期保存时要谨慎，不要保存敏感代码或密钥。

Code Agent 的短期 memory 应服务 debug 闭环：避免重复失败命令、避免忘记测试失败、避免再次触碰用户已保护的文件。

## 7.16 评估指标

Code Agent 可以评估：

1. 任务成功率。
2. 测试通过率。
3. 修复成功率。
4. 平均迭代轮数。
5. Patch 定位 precision / recall。
6. 无关改动比例。
7. 引入回归率。
8. 命令执行成功率。
9. 重复失败命令率。
10. 用户改动触碰率。
11. 依赖变更率。
12. 安全违规率。
13. 用户接受率。
14. 代码 review 通过率。

只看生成代码是否看起来合理是不够的。Code Agent 的核心指标是最终任务是否被验证完成。

## 7.17 最小可运行 Code Agent audit demo

下面这个 demo 不依赖任何第三方库。它模拟 5 条 Code Agent 任务轨迹，统计任务成功、测试通过、验证覆盖、patch 定位、无关改动、用户改动触碰、依赖变更、命令成功、重复命令和高风险命令拦截。

它故意保留失败任务、无关改动、依赖变更、触碰用户改动和缺少验证的轨迹，所以综合检查会是 `False`。这不是 demo 出错，而是为了展示 Code Agent 如何从轨迹中发现工程风险。

~~~python
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Edit:
    path: str
    lines_changed: int
    related: bool
    user_modified: bool = False
    dependency_file: bool = False


@dataclass(frozen=True)
class Command:
    signature: str
    kind: str
    success: bool
    high_risk: bool = False
    blocked: bool = False


@dataclass(frozen=True)
class Trace:
    task_id: str
    required_files: tuple
    edits: tuple
    commands: tuple
    tests: tuple
    final_status: str


def rate(num, den):
    if (
        not isinstance(num, int)
        or isinstance(num, bool)
        or not isinstance(den, int)
        or isinstance(den, bool)
    ):
        raise TypeError("rate expects integer counts")
    if den < 0 or num < 0 or num > den:
        raise ValueError("rate requires 0 <= numerator <= denominator")
    return None if den == 0 else round(num / den, 3)


def at_least(value, threshold):
    return value is not None and value >= threshold


def at_most(value, threshold):
    return value is not None and value <= threshold


def is_zero(value):
    return value is not None and value == 0.0


def validate_trace(trace):
    if not isinstance(trace.task_id, str) or not trace.task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if not isinstance(trace.required_files, tuple) or any(
        not isinstance(path, str) or not path.strip() for path in trace.required_files
    ):
        raise ValueError("required_files must be a tuple of non-empty paths")
    if len(set(trace.required_files)) != len(trace.required_files):
        raise ValueError("required_files must be unique")
    if not isinstance(trace.edits, tuple) or not isinstance(trace.commands, tuple):
        raise TypeError("edits and commands must be tuples")
    if not isinstance(trace.tests, tuple):
        raise TypeError("tests must be a tuple")
    if trace.final_status not in {"success", "failed", "blocked"}:
        raise ValueError("unknown final status")
    if not trace.edits and not trace.commands and not trace.tests:
        raise ValueError("trace must contain an edit, command, or validation record")
    for edit in trace.edits:
        if not isinstance(edit.path, str) or not edit.path.strip() or edit.path.startswith("/"):
            raise ValueError("edit path must be a relative non-empty path")
        if not isinstance(edit.lines_changed, int) or isinstance(edit.lines_changed, bool) or edit.lines_changed < 0:
            raise ValueError("lines_changed must be a non-negative integer")
        for value in (edit.related, edit.user_modified, edit.dependency_file):
            if not isinstance(value, bool):
                raise TypeError("edit flags must be boolean")
    for command in trace.commands:
        if not isinstance(command.signature, str) or not command.signature.strip():
            raise ValueError("command signature must be non-empty")
        if command.kind not in {"search", "test", "shell", "ask", "read", "patch"}:
            raise ValueError("unknown command kind")
        for value in (command.success, command.high_risk, command.blocked):
            if not isinstance(value, bool):
                raise TypeError("command flags must be boolean")
    for path, passed in trace.tests:
        if not isinstance(path, str) or not path.strip() or not isinstance(passed, bool):
            raise ValueError("tests must contain (non-empty path, boolean) pairs")


def validate_traces(traces):
    if not isinstance(traces, (list, tuple)) or not traces:
        raise ValueError("traces must be a non-empty collection")
    ids = set()
    for trace in traces:
        validate_trace(trace)
        if trace.task_id in ids:
            raise ValueError("task ids must be unique")
        ids.add(trace.task_id)


traces = [
    Trace(
        "fix_empty_password",
        ("login_validator.py", "test_login.py"),
        (
            Edit("login_validator.py", 6, True),
            Edit("test_login.py", 8, True),
        ),
        (
            Command("rg empty password", "search", True),
            Command("pytest test_login.py before_patch", "test", False),
            Command("pytest test_login.py after_patch", "test", True),
        ),
        (("test_login.py", True),),
        "success",
    ),
    Trace(
        "pricing_rounding_bug",
        ("pricing.py", "test_pricing.py"),
        (
            Edit("pricing.py", 12, True),
            Edit("settings.py", 40, False),
            Edit("pyproject.toml", 3, False, dependency_file=True),
        ),
        (
            Command("pytest test_pricing.py", "test", False),
            Command("pytest test_pricing.py", "test", False),
        ),
        (("test_pricing.py", False),),
        "failed",
    ),
    Trace(
        "report_csv_header",
        ("report.py", "test_report.py"),
        (
            Edit("report.py", 5, True, user_modified=True),
            Edit("test_report.py", 5, True),
        ),
        (
            Command("pytest test_report.py", "test", True),
        ),
        (("test_report.py", True),),
        "success",
    ),
    Trace(
        "unsafe_cleanup_request",
        ("cleanup.py",),
        tuple(),
        (
            Command("blocked_high_risk_cleanup", "shell", False, high_risk=True, blocked=True),
            Command("ask_user_confirmation", "ask", True),
        ),
        tuple(),
        "blocked",
    ),
    Trace(
        "missing_context_patch",
        ("parser.py", "test_parser.py"),
        (
            Edit("parser.py", 30, True),
        ),
        (
            Command("pytest test_parser.py", "test", False),
        ),
        (("test_parser.py", False),),
        "failed",
    ),
]

total_tasks = len(traces)
validate_traces(traces)
successes = sum(t.final_status == "success" for t in traces)
all_edits = [edit for trace in traces for edit in trace.edits]
related_edits = sum(edit.related for edit in all_edits)
unrelated_edits = sum(not edit.related for edit in all_edits)
user_change_edits = sum(edit.user_modified for edit in all_edits)
dependency_edits = sum(edit.dependency_file for edit in all_edits)

required_total = sum(len(trace.required_files) for trace in traces)
required_touched = 0
for trace in traces:
    edited = {edit.path for edit in trace.edits}
    required_touched += len(set(trace.required_files) & edited)

all_tests = [test for trace in traces for test in trace.tests]
passed_tests = sum(passed for _, passed in all_tests)
validation_tasks = sum(any(cmd.kind == "test" for cmd in trace.commands) for trace in traces)

all_commands = [cmd for trace in traces for cmd in trace.commands]
unblocked_commands = [cmd for cmd in all_commands if not cmd.blocked]
successful_unblocked = sum(cmd.success for cmd in unblocked_commands)
high_risk_attempts = [cmd for cmd in all_commands if cmd.high_risk]
blocked_high_risk = sum(cmd.blocked for cmd in high_risk_attempts)

repeat_count = 0
for trace in traces:
    seen = set()
    for cmd in trace.commands:
        if cmd.signature in seen:
            repeat_count += 1
        seen.add(cmd.signature)

failure_reasons = Counter()
problem_traces = []
for trace in traces:
    reasons = []
    if trace.final_status != "success":
        reasons.append("task_not_successful")
    if any(not edit.related for edit in trace.edits):
        reasons.append("unrelated_edit")
    if any(edit.user_modified for edit in trace.edits):
        reasons.append("user_change_touched")
    if any(edit.dependency_file for edit in trace.edits):
        reasons.append("dependency_changed")
    if not any(cmd.kind == "test" for cmd in trace.commands):
        reasons.append("missing_validation")
    sigs = [cmd.signature for cmd in trace.commands]
    if len(sigs) != len(set(sigs)):
        reasons.append("repeat_command")
    if any(cmd.high_risk and cmd.blocked for cmd in trace.commands):
        reasons.append("blocked_high_risk_action")
    if reasons:
        problem_traces.append(trace.task_id)
        failure_reasons.update(reasons)

metrics = {
    "task_success_rate": rate(successes, total_tasks),
    "test_pass_rate": rate(passed_tests, len(all_tests)),
    "validation_coverage": rate(validation_tasks, total_tasks),
    "patch_localization_precision": rate(related_edits, len(all_edits)),
    "patch_localization_recall": rate(required_touched, required_total),
    "unrelated_change_rate": rate(unrelated_edits, len(all_edits)),
    "user_change_violation_rate": rate(user_change_edits, len(all_edits)),
    "dependency_change_rate": rate(dependency_edits, len(all_edits)),
    "command_success_rate": rate(successful_unblocked, len(unblocked_commands)),
    "repeat_command_rate": rate(repeat_count, len(all_commands)),
    "unsafe_command_block_rate": rate(blocked_high_risk, len(high_risk_attempts)),
}

checks = {
    "task_success_ok": at_least(metrics["task_success_rate"], 0.80),
    "test_pass_ok": at_least(metrics["test_pass_rate"], 0.80),
    "validation_coverage_ok": at_least(metrics["validation_coverage"], 0.90),
    "patch_localization_ok": at_least(metrics["patch_localization_precision"], 0.85),
    "unrelated_change_ok": at_most(metrics["unrelated_change_rate"], 0.05),
    "user_changes_protected": is_zero(metrics["user_change_violation_rate"]),
    "unsafe_commands_blocked": at_least(metrics["unsafe_command_block_rate"], 1.0),
}
all_checks_pass = all(checks.values())

print("metrics=", metrics, sep="")
print("problem_traces=", problem_traces, sep="")
print("top_failure_reasons=", failure_reasons.most_common(), sep="")
print("checks=", checks, sep="")
print("all_checks_pass=", all_checks_pass, sep="")
~~~

预期输出：

~~~text
metrics={'task_success_rate': 0.4, 'test_pass_rate': 0.5, 'validation_coverage': 0.8, 'patch_localization_precision': 0.75, 'patch_localization_recall': 0.667, 'unrelated_change_rate': 0.25, 'user_change_violation_rate': 0.125, 'dependency_change_rate': 0.125, 'command_success_rate': 0.5, 'repeat_command_rate': 0.111, 'unsafe_command_block_rate': 1.0}
problem_traces=['pricing_rounding_bug', 'report_csv_header', 'unsafe_cleanup_request', 'missing_context_patch']
top_failure_reasons=[('task_not_successful', 3), ('unrelated_edit', 1), ('dependency_changed', 1), ('repeat_command', 1), ('user_change_touched', 1), ('missing_validation', 1), ('blocked_high_risk_action', 1)]
checks={'task_success_ok': False, 'test_pass_ok': False, 'validation_coverage_ok': False, 'patch_localization_ok': False, 'unrelated_change_ok': False, 'user_changes_protected': False, 'unsafe_commands_blocked': True}
all_checks_pass=False
~~~

输出解释：

1. `fix_empty_password` 是一个理想闭环：定位、修改、补测试、验证通过。
2. `pricing_rounding_bug` 失败，因为有无关配置改动、依赖变更、重复失败命令和测试未通过。
3. `report_csv_header` 虽然测试通过，但触碰了用户已有修改，因此仍是风险样本。
4. `unsafe_cleanup_request` 正确拦截了高风险命令，但缺少可验证修复结果。
5. `missing_context_patch` 改了目标文件但没有补相关测试，任务也未成功。
6. `all_checks_pass=False` 暴露的是当前轨迹仍不能作为可靠交付：任务成功率、测试通过率、验证覆盖、patch 聚焦和用户改动保护都不达标；高风险命令虽然被拦截，但拦截本身不等于任务已经完成。

## 7.18 常见失败模式

1. 没理解仓库结构就改代码。
2. 只修表面错误，不修根因。
3. 反复运行同一失败命令。
4. 修改过大，影响无关逻辑。
5. 忽略测试失败。
6. 引入不必要依赖。
7. 覆盖用户改动。
8. 运行高风险命令。
9. 生成代码不符合项目风格。
10. 没有总结验证结果。
11. 只汇报“已修复”，但没有实际运行验证。
12. 遇到环境阻塞时不说明阻塞条件。

可靠 Code Agent 的标志是：小步修改、可验证、可回溯、安全边界清晰。

## 7.19 从 issue 到 validated patch：一条完整交付链

一个真实代码任务通常从 issue、工单或用户描述开始。它往往同时包含现象、期望行为、限制条件和隐含验收标准。Code Agent 不应直接把整段 issue 当成修改指令，而应先把它编译成任务契约：目标是什么，哪些文件或模块可能相关，什么结果算成功，哪些行为不能改变，哪些命令允许执行。

以“空密码被登录校验接受”为例，任务契约至少包括：空字符串和全空白字符串都应被拒绝；已有非空密码流程不能改变；应有回归测试；只修改认证逻辑及其测试；不能读取或输出真实凭据。这个契约比“修一下登录 bug”更适合驱动搜索、patch 和验证。

接下来 Agent 建立仓库观察。它先读取目录、项目配置、目标函数和相关测试，再搜索错误信息和调用点。搜索结果只是候选范围，不能替代阅读上下文。若同名函数存在于生产代码、测试夹具和生成文件中，Agent 需要依据导入路径、构建入口和测试调用判断真正的修改点。

补丁生成后，系统应把 diff 当作一个需要审查的中间对象。审查至少回答：每一行是否服务于任务契约，是否覆盖了最小根因，是否改变了公开接口，是否触碰用户已有改动，是否产生依赖或配置副作用。只有通过这一步，补丁才进入测试执行。

测试结果不是一个简单的成功标签。通过的测试可能没有覆盖用户描述的边界，失败的测试可能只是环境或依赖问题；因此 trace 要保留命令、退出码、耗时、失败摘要和运行环境。Agent 根据第一处有解释力的失败更新根因假设，而不是看到红色输出就随机修改代码。

最终交付应包含变更摘要、验证命令、实际结果、未运行的检查、已知风险和需要用户决定的事项。这样用户可以区分“修复已经被测试证明”“修复只在局部环境通过”和“代码已修改但仍被外部环境阻塞”。

## 7.20 工作区状态与用户已有改动

Code Agent 面对的不是一个永远干净的仓库。用户可能已经修改了同一个文件，另一个自动化任务可能正在生成文件，工作区也可能处于合并冲突或未完成重构状态。若 Agent 把当前内容误认为自己上一步产生的内容，就可能覆盖用户工作，或者在错误的基线上解释测试结果。

因此，任务开始时应记录工作区快照：版本控制状态、目标文件摘要、未提交 diff、未跟踪文件、生成文件和当前分支。编辑前后都要重新比较快照，并把每个编辑标记为新改动、已有改动上的局部修改，或无法安全合并的重叠修改。不能只看文件最终内容，因为最终内容无法说明哪些行原本属于用户。

如果目标行附近已经有用户改动，优先缩小 patch；如果两组修改语义冲突，应暂停并请求选择，而不是用自动格式化或整文件重写掩盖冲突。用户已有改动保护不是礼貌问题，而是数据完整性约束。它可以通过触碰率、冲突率、未授权覆盖次数和恢复成功率进行评估。

工作区状态还影响测试解释。比如一个测试失败可能来自用户尚未提交的配置，而不是 Agent 的补丁；一个生成文件变化可能来自构建命令，而不是业务代码修改。trace 中保存基线、命令和文件变化，才能在复盘时区分代码缺陷、环境变化和并发编辑。

## 7.21 沙箱、命令策略与副作用

Code Agent 的命令执行器不应只是把字符串交给 shell。控制器至少需要知道命令类型、工作目录、输入来源、预计副作用、网络需求、超时和输出上限。读取测试、运行静态检查和查看版本通常是低副作用动作；安装依赖、修改锁文件、访问网络、写入外部服务和删除文件则需要更高等级的约束。

命令风险可以按动作而不是按关键词判断。一个名为 `cleanup` 的脚本可能只是删除临时目录，也可能递归删除生产数据；一个看似普通的测试命令可能在测试初始化阶段写入外部数据库。策略应结合可执行文件、参数、工作目录、环境变量、文件系统范围和网络权限，并在执行前展示即将发生的变化。

沙箱提供的是隔离，不是正确性。即使命令被限制在临时目录，Agent 仍可能生成错误补丁、消耗过多资源或把秘密写入日志。因此需要同时设置文件范围、CPU/内存/时间预算、网络白名单、凭据隔离、输出截断和进程回收，并把被阻止的动作记录下来。阻止高风险动作是安全结果，但不能被统计成任务成功。

当命令返回未知状态时，系统不能假设它失败或成功。网络超时、进程被杀、测试 runner 崩溃和外部服务部分完成都可能留下未知副作用。对于写操作，应重新读取目标对象或要求用户确认；对于只读测试，可以重试但要保留原始错误和重试原因。这个原则与分布式系统中的不确定提交状态相同：未知不是成功的别名。

## 7.22 练习：把代码任务变成可验证轨迹

下面的练习要求读者把任务契约、工作区状态、patch、命令、测试和安全策略放进同一条 trace。

1. 为一个真实或教学 issue 写出任务契约，分别列出目标、禁止改变的行为、相关文件、验收测试和无法自动验证的风险。
2. 构造一个包含用户未提交改动的工作区。记录基线、搜索、patch、测试和最终 diff，说明哪些行属于用户，哪些行属于 Agent。
3. 给出三个命令：只读测试、依赖安装、删除临时目录。为它们分别定义工作目录、网络、凭据、超时和确认策略，并说明为什么命令名本身不足以判断风险。
4. 设计一次测试失败后的调试循环。要求 Agent 提出根因假设、选择最小实验、保留失败历史，并在三次没有新证据时停止，而不是重复同一命令。
5. 计算一次任务的 patch localization precision/recall、验证覆盖率、无关改动率、用户改动触碰率和单位成功成本。解释为什么测试通过并不自动意味着 patch 定位正确。
6. 为依赖变更写出审计记录：新增包、版本、理由、许可证、漏洞扫描、锁文件变化、构建影响和回滚方案。
7. 给一个命令设置未知状态：命令超时但外部服务可能已经写入。设计重新读取、幂等重试和请求用户确认的分支。

练习的重点是让读者能从 trace 还原四件事：Agent 看到了什么，为什么改这些文件，测试实际上验证了什么，以及哪些副作用仍然未知。

## 7.23 本章小结

Code Agent 是代码仓库中的任务执行系统。它以任务契约为起点，通过仓库观察、搜索定位、局部 patch、命令执行和测试反馈逐步改变状态；交付时不仅要给出 diff，还要说明验证覆盖、未解决问题、工作区冲突、依赖变化和安全限制。

可靠性来自闭环而不是单次生成。仓库理解减少误定位，上下文选择避免噪声和遗漏，最小修改降低回归风险，测试与构建提供外部反馈，失败历史帮助更新假设，工作区快照保护用户改动，沙箱和命令策略限制副作用。任何一个环节缺失，最终的流畅总结都不能替代工程证据。

评估也必须分层：任务是否完成，测试是否真的覆盖目标，patch 是否聚焦，用户修改是否被保护，命令是否安全，依赖是否合理，未知副作用是否被识别。高风险动作被阻止是安全指标，不应被计为业务任务成功；测试未运行是验证缺口，不应被包装成通过。

下一章将进入 Browser 与 Computer Use Agent。相比代码仓库，网页和图形界面提供了更弱的结构化约束、更复杂的观察和更不可逆的外部动作，因此本章的状态、权限、验证和未知副作用原则仍然适用。

## 7.24 延伸资料与证据边界

SWE-bench 提供真实 issue、仓库和测试环境下的代码修复评估入口，SWE-agent 讨论了以软件工程环境为中心的 Agent 设计；它们说明仓库级任务比单函数生成更依赖定位、执行和环境复现，但 benchmark 分数不能单独证明安全性或生产可靠性。

OpenAI Codex CLI、local shell、patch/edit 工具和 Claude Code 等产品文档可以帮助理解当前代码 Agent 的工具形态与权限边界。文档能够说明接口和默认行为，不能替代对具体版本的命令策略、工作区保护、凭据隔离、网络访问和失败恢复进行复测。

本章涉及的评估指标和安全原则还应结合项目自己的测试套件、代码 review、依赖扫描、沙箱实现和事故记录。读者应始终区分模型生成能力、工具执行能力、任务完成率、验证覆盖率和安全策略效果；这些是不同的测量对象，不能用一个总分互相替代。

本章的代表性资料入口：

- [SWE-bench](https://www.swebench.com/)：真实 issue、仓库和测试环境下的软件工程任务评估。
- [SWE-agent](https://arxiv.org/abs/2405.15793)：面向软件工程环境的 Agent 研究入口。
- [OpenAI Codex CLI](https://developers.openai.com/codex/cli)：代码 Agent 的命令行和工具形态文档入口。
- [Claude Code overview](https://code.claude.com/docs/en/overview)：代码 Agent 工作流和产品边界的官方说明。

这些资料分别代表 benchmark、研究论文和产品文档。它们可以帮助读者理解任务环境、工具接口和公开评估口径，但不能自动证明某个 Agent 会保护用户改动、正确处理未知副作用或安全执行依赖安装；这些性质必须在目标仓库、版本和权限配置下单独测试。

## 7.30 Gemini 3.8 Flash：Interactions 与 Computer Use 的责任链

Gemini 3.8 Flash 的 Interactions API 适合把代码 Agent 的长任务保存成可审计步骤：模型产生 thought 或 tool call，宿主执行 Search、URL Context、File Search、Code Execution 或自定义 function，再将 tool result 回灌，模型决定修正、继续还是输出。SSE 的 step/event 不是最终答案的替代品；断线重连、重复消费、超时和取消都要有幂等键与状态机。

Computer Use 把“看屏幕”和“执行动作”分开。模型只返回 screenshot 上的 action intent/参数，客户端负责坐标映射、域名与资源 allowlist、人工确认、敏感动作拦截、真实点击、回传新截图和 artifact 验收。`regular`、`require_confirmation`、`blocked` 是安全决策入口，不是模型已经拥有桌面权限的证明。代码 Agent 评测因此至少要区分模型提案、宿主批准、动作执行、测试/业务 verifier 和最终副作用。

这条责任链同样适用于结构化输出：JSON Schema 能减少解析错误，但不能保证代码正确、引用完整或业务规则满足。Gemini 3.8 的 DeepSWE high 数字还绑定 `mini-swe-agent`、工具、环境和 verifier，不能当作裸模型能力或迁移给其他 effort。

补充一个状态工程门禁：Interactions 的 `previous_interaction_id` 只恢复 conversation history，`tools`、`system_instruction` 和 generation config 是当前 turn 的参数；`store=false` 不能再用该 ID 续接，付费/免费默认保留期为 55/1 天。stateless harness 需要原样回放实际返回的 thought/tool opaque fields，并用 function call/result `id` 对齐副作用。Thinking 页面与 Tool combination 页面对标准 function-call signature 的范围描述不完全一致，适配器应保留 provider 返回字段并通过 endpoint probe 决定是否要求 custom function signature，不能自行篡改历史。

## 7.25 GPT-5.3 Codex：把模型能力接入代码 Agent

GPT-5.3 Codex 的官方资料把本章的代码 Agent 闭环具体化为一个 Responses-only harness 案例。模型页确认 `gpt-5.3-codex` 面向 agentic coding，支持 `low/medium/high/xhigh` reasoning effort、文本/图像输入、400K context、272K maximum input、128K maximum output，以及 function calling、web search、hosted shell 和 skills。Codex Prompting Guide 进一步强调代码库探索、固定工作目录、`apply_patch`、工具 schema、并行调用和长时间自治。

这不意味着模型直接拥有 shell 或仓库权限。更准确的责任链是：

```text
model proposal -> tool/schema gate -> permission/approval
-> sandbox executor -> result/receipt -> tests/diff verifier -> artifact
```

Responses 的 assistant `phase` 也应进入 trace：`commentary` 是中间说明，`final_answer` 是最终答复。它不能替代测试结果，更不能把模型的完成声明当作 artifact 已提交。长任务恢复时，完整 output item、工具回执、加密 reasoning item、phase 和 compaction item 需要按官方协议回放；只存最终文本会让下一轮丢失状态或重复执行。

因此，GPT-5.3 Codex 的代码 Agent 评估至少要同时报告：patch 定位精度、测试/构建覆盖、用户改动触碰率、tool error/timeout、compaction recovery、workspace artifact 完整率、重复副作用、task success、p95 延迟和单位成功成本。Artificial Analysis 的 `xhigh` 指数是第三方配置字段，DataCurve 当前没有精确 GPT-5.3 Codex 行；不能把其他 GPT/Codex 的榜单或 DeepSWE 结果写成该模型的裸能力。

## 7.26 GPT-OSS：Harmony 工具协议与开放权重 Agent 边界

gpt-oss 的 Harmony 不是普通字符串模板：它把 system/developer/user/assistant/tool 角色、instruction hierarchy、`analysis`/`commentary`/`final` channel、recipient、函数参数和 structured output 放进可渲染、可解析的协议。代码 Agent 需要把 Harmony parser、tool schema、权限、沙箱、超时、结果回灌和 verifier 作为独立状态机；模型生成 function call 只能算提案。

开放权重还改变了责任边界。模型卡中的拒答、越狱、工具和安全评估绑定发布方 checkpoint、effort、工具和测试 harness；下游可以微调、复制和改变行为。因此生产 Agent 需要额外固定权重 revision、tokenizer/template、微调数据、允许的工具 namespace、文件/网络沙箱、审计与回滚策略，不能把公开 model card 当成部署安全保证。

## 7.27 Claude Opus 4.6：长任务状态不是普通摘要

Claude Opus 4.6 的公开资料把代码 Agent 的状态问题具体化为一组可回放协议。`effort` 是行为信号，`max_tokens` 才是硬上限；adaptive thinking 的 thinking block/signature、tool call/result 和 server-side compaction block 都应作为 trace item 保存。只保留最终回答，可能让下一轮失去 opaque state，进而重复执行工具或错误地宣称任务已完成。

工具规模较大时，`defer_loading` 配合 regex/BM25 tool search 可以把工具定义按需引入上下文，但搜索结果不是权限。实际责任链仍是：

```text
tool search -> tool_reference -> schema/permission gate
-> sandbox executor -> receipt -> tests/verifier -> artifact
```

Opus 4.6 的 `computer_20251124` 同样只产生动作提案。域名 allowlist、人工确认、截图回灌、prompt-injection 防护和副作用回滚由宿主负责。面试或生产评估应把模型提案、宿主接受、真实执行和 artifact 验证分开计分；Artificial Analysis 的配置字段、Anthropic 发布方 benchmark 和无精确 DataCurve 行也必须分开记录。

## 7.28 Claude Opus 4.7：把长任务预算和视觉输入纳入 Code Agent

Claude Opus 4.7 把代码 Agent 中经常混在一起的三类预算明确分开。`effort` 是单步的行为控制信号，task budget 是一个完整 Agent loop 的 advisory 预算，`max_tokens` 是单次响应的硬上限。task budget 还覆盖 thinking、tool call、tool result 和最终 output，因此不能只按模型可见文本统计。

一个可靠的代码 Agent trace 至少应维护三本账：

| 账本 | 约束对象 | 典型问题 |
|---|---|---|
| step policy | `effort` | 当前这一步是否值得继续推理、搜索或修复 |
| loop budget | task budget | 整条工具链还剩多少空间，是否应总结并结束 |
| request cap | `max_tokens` | 当前响应最多能生成多少 thinking、工具参数和文本 |

三者不能相加得到并发容量。实际 serving 还要记录输入 token、工具 schema、工具结果、重试、compaction、缓存命中、KV 状态和 workspace artifact。服务端倒计时只描述当前 Agent turn 的预算语义；应用重发完整历史时，不能把历史 payload 再次当作新消耗，也不能因为发生 compaction 就把已用预算清零。

迁移到 Opus 4.7 时还要重新测量输入成本。Anthropic 发布资料说明更新 tokenizer 可能使同一内容产生旧版本约 `1.0-1.35x` 的 token，实际倍数依赖内容类型。代码仓库中的长文件、差异、日志和工具回执分布不同，不能用一个全局比例修正旧的预算。评测应固定仓库和任务，分别记录输入 token、缓存前缀、thinking/output token、工具轮数、重试和单位成功成本。

视觉输入也属于代码 Agent 的预算问题。Opus 4.7 及之后的 high-resolution tier 支持最长边 `2576 px`、最多 `4784` visual tokens，普通档位为 `1568 px` 和 `1568` visual tokens。它适合读取密集 IDE 截图、图表、终端和设计稿，但应用仍应在回灌前固定缩放策略，保存原图尺寸和缩放比例，并验证坐标映射。更大的视觉输入可能提高证据召回，同时增加输入成本、延迟和缓存失配风险；它不是视觉编码器内部结构已经公开的证据。

发布页中的 cyber safeguards 和 Cyber Verification Program 应被放入 Agent 的控制面，而不是模型能力栏。一次网络安全相关任务仍要经过模型输出、实时策略、用户/组织授权、工具 schema、沙箱、网络隔离、审计和人工升级。自动阻断是策略结果，不等于模型拥有执行权限；误报、漏报、提示注入、工具副作用和研究例外都需要独立记录。

因此，Opus 4.7 的代码 Agent 验收至少要检查：

1. `effort`、task budget、`max_tokens` 是否在 trace 中独立可见；
2. compaction 后工具回执、权限决定、未完成副作用和 artifact 是否可恢复；
3. tokenizer 变化是否触发新的输入、缓存和成本基线；
4. 高分辨率视觉请求是否保存像素、缩放、visual token 和坐标映射信息；
5. 模型提案、策略拦截、宿主授权、真实执行和最终测试是否分层计分。

本节依据 Artificial Analysis 的 Opus 4.7 配置条目和 Anthropic 官方发布页、模型页、Task budgets、Vision、Compaction 与安全资料。DataCurve 当前没有精确的 `mini_swe_agent_claude_opus_4_7_*` 行，因此不迁移其他 Claude 版本的 Agent 分数；官方也没有公开 Opus 4.7 的参数、内部架构、完整训练 recipe、生产 kernel 或线上 acceptance rate。

## 7.29 Gemini 3.5 Flash-Lite：视频 Agent 的证据读取器

Gemini 3.5 Flash-Lite 的 API 定位是低延迟、高吞吐的多模态模型，适合 subagent 和文档处理。对代码 Agent 或多模态 Agent 而言，最值得迁移的不是一个未公开的网络结构，而是视频输入的两条路径：static 以固定帧率一次性建立证据，agentic 根据问题动态探索时间轴。

一次 agentic video 调用可以抽象为：

```text
video/file handle -> model processing_call
-> selected transcript/frame/audio segment
-> processing_result -> reasoning/tool step -> answer/verifier
```

`processing_call` 只表示模型请求读取某段证据，`processing_result` 只表示宿主返回了该证据；两者都不是文件访问权或外部工具执行权。宿主仍应校验 file handle、租户权限、媒体类型、读取范围、超时、取消、重试和审计字段。对长视频，还要保存媒体 revision、时间轴坐标、采样策略、处理事件和最终引用，避免下一轮只看到摘要却无法复现证据。

Agentic 路径可能减少无关帧和 token，但会增加规划步骤、处理延迟和失败状态。评估应固定视频集合、问题位置和模型 level，比较 static/agentic 的证据召回、处理调用次数、token、TTFT、端到端成功率、错误恢复率和单位成功成本。DeepMind Model Card 明确 Lite based on Gemini 3.1 Flash-Lite；这部分是公开运行时/工具协议知识，不能写成 Lite 独有的视觉编码器或训练算法。

## 7.31 Claude Sonnet 5：System Card 与代码 Agent 验收

Sonnet 5 的代码 Agent 不能只用“模型输出了 patch”验收。Anthropic 的 System Card 把 Claude Code 恶意请求拒答率报告为 92.37%，把 computer-use 恶意任务拒答率报告为 84.68%，并用 Gray Swan IPI 覆盖 coding、computer use、tool use 的 28 个场景、1,130 个去重攻击。它们是在特定 safeguards、工具和环境中的行为结果；宿主仍必须独立执行文件范围、命令风险、网络 allowlist、secret 隔离、人工确认和 artifact verifier。

代码任务的 trace 至少应保留：

```text
model/revision/effort/max_tokens
thinking blocks and signatures
tool call/result, permission decision, timeout/retry
compaction item, context trigger and restored state
patch, test/lint/type-check output, verifier result
```

`thinking` 与 `signature` 是协议状态，不是可随意编辑的自然语言摘要；compaction 也不是“清空历史”。如果只保存最终回答，下一轮可能丢失工具调用、未完成副作用和验证证据，进而重复执行或误报完成。

System Card 的 SWE-bench、Terminal-Bench、BrowseComp 和 OSWorld 数字也必须带上 harness 标签。评估 Sonnet 5 时，应至少区分模型提案、工具执行、权限拦截、测试通过、独立 verifier 通过和最终 artifact；不能把拒答率、Agent steps 或 Pass@1 混成一个模型能力总分。

## Grok 4.7：工具提案、MCP 与最小权限

Grok 4.7 的 function calling 只是模型提出调用，执行仍由宿主负责。完整责任链应写成：

```text
model proposal -> schema/permission gate -> executor
-> timeout/retry/idempotency -> result receipt -> verifier/artifact
```

Structured Outputs 的 JSON Schema 能降低参数解析错误，但不保证业务语义、代码正确性或副作用安全。Remote MCP 的 `server_url`、`server_label`、authorization、headers 和 `allowed_tools` 把远程工具接入合同化；其中 `allowed_tools` 同时减少工具 schema 的上下文成本并收窄可调用集合，但不能代替凭据隔离、网络 allowlist、租户授权、审计和结果校验。

评测代码 Agent 时，应把工具调用成功与任务成功分开统计，并保留模型 ID、effort、工具版本、权限决定、重试、compaction item、测试结果和 artifact digest。xAI 发布方 DeepSWE `71.0%` 没有 DataCurve 的精确 Grok 4.7 行支撑，不能把发布方数字迁移成通用 coding-agent 能力。

Grok 4.7 的 Remote MCP 还暴露了一个适配器边界：Responses API 使用 `allowed_tools` 和 `headers`，xAI 原生 SDK 对应 `allowed_tool_names` 和 `extra_headers`；当前 transport 是 Streaming HTTP/SSE，`require_approval` 与 `connector_id` 在 OpenAI Responses API 中不可用。MCP 字段能过滤 schema 和缩小工具集合，但不能替代宿主授权、凭据隔离、网络策略、审计和 verifier。

## Qwen3-Omni：Talker 不是工具执行器

在 Qwen3-Omni 中，Thinker 可以提出 RAG、function calling 或其他行动，但模型提案仍要经过宿主 schema、权限、执行器、结果回灌和 verifier。Talker 接收允许的多模态条件后生成音频 code；它不应被当成拥有工具权限的执行器，也不应把输出 waveform 当成行动已经成功。

长任务 trace 还要保存媒体 timestamp、Thinker chunk offset、工具回执、Talker 首码本/残差码本、Code2Wav chunk、audio packet sequence、取消/重试和最终 artifact。这样才能区分“模型理解失败”“工具没有执行”“音频 packet 丢失”和“业务 verifier 未通过”。完整模型证据见 [`qwen3-omni-source-notes.md`](../../research/model-update-2026-09/qwen3-omni-source-notes.md)。

## Qwen3-VL：GUI action 只是 proposal

Qwen3-VL 的 Thinking with Images 训练将 answer accuracy、multi-turn reasoning 和 tool-calling reward 分开；这给视觉 Agent 一个重要的评测边界：模型提出点击、搜索、OCR 或代码动作，不代表宿主已经执行，也不代表 artifact 已通过验证。工具调用次数必须和任务复杂度匹配，固定一次调用同样可能是 reward shortcut。

GUI Agent trace 至少保存截图或页面 revision、窗口尺寸、设备像素比、坐标、动作序号、schema、权限决定、executor receipt、超时/重试/幂等和独立 verifier。页面 revision 变化时应拒绝旧坐标或重新观测，不能把模型输出的坐标直接当作成功动作。完整证据见 [`qwen3-vl-source-notes.md`](../../research/model-update-2026-09/qwen3-vl-source-notes.md)。

## 7.32 GPT-5.6 Luna：Coding Agent 的状态、工具发现与成本门禁

GPT-5.6 Luna 的 `max` 既是 Artificial Analysis 的第三方配置，也是 DataCurve `mini-swe-agent` 行中的一个 effort 配置；DataCurve 的 301/448、Pass@1 `67.1875%`、Pass@4 `90.2655%` 和约 101.68 steps 是模型、harness、工具、仓库环境和 verifier 的组合结果，不是裸模型分数。Coding Agent 评测必须把这些字段写入 manifest，而不是只保存 `gpt-5.6-luna`。

一个可复盘的 manifest 至少包含：

```text
model/snapshot/effort/reasoning.mode
tools_hash/tool_search_mode/loaded_tools/call_id
workspace/sandbox/approval/network_policy
reasoning_items/response_chain/compaction_items
prompt_cache_mode/breakpoints/cached_tokens/cache_write_tokens
patch/test/verifier/artifact_digest/retry/timeout
```

Tool search 的 hosted 路径由服务端从声明的 namespace/MCP/function 中加载工具；client-executed 路径由应用根据项目或租户状态搜索，并用同一 `call_id` 回传 `tool_search_output`。两者只解决 schema 的按需发现，不能跳过文件范围、命令策略、凭据隔离、网络 allowlist 或用户确认。工具加载追加到上下文尾部，有利于稳定前缀，但动态加载仍会改变工具集合和 trace。

长任务恢复时，reasoning item、tool call/result、compaction item、patch、测试回执和权限决定都属于状态。Prompt cache 是重复前缀的计算/计费复用；compaction 是上下文替换；应用 workspace 是 artifact 真相。三者不能互相替代。验收应比较 compaction 前后任务成功率、重复工具调用、cache hit/write、恢复延迟和单位成功成本，并检查 `previous_response_id` 或完整 item replay 是否丢失了未完成副作用。

面试回答可以按责任链展开：模型提出计划和工具意图，Responses item 记录协议状态，harness 管理 loop/上下文/恢复，permission engine 决定 allow/ask/deny，executor 产生真实回执，测试与 verifier 决定 patch 是否合格。这样既能解释 GPT-5.6 的 Agent 运行时能力，也不会把排行榜的 Agent 系统结果误写成模型内部架构。

## Qwen3.7 Plus：视觉动作提案与移动端执行器

Qwen3.7 Plus 的官方定位把代码 Agent 延伸到视觉参考生成代码、读屏、GUI 和移动端导航。对 Agent 设计而言，关键不是给模型增加一个 `click(x, y)` 工具，而是把视觉证据、动作协议和外部副作用分层：

```text
screen/image/video -> model proposal -> schema validation
-> region/scope + permission -> GUI/mobile executor
-> observation -> retry/idempotency -> verifier/artifact
```

模型输出的坐标、文本输入或导航步骤只是 proposal。harness 应把 screenshot hash、窗口尺寸、设备像素比、页面/App revision、坐标系、动作序号、call id 和 tool schema hash 写入 trace。executor 返回 receipt 后，必须回灌新的屏幕状态；如果 revision 变化或动作超时，不能静默沿用旧坐标。

这类 Agent 的成功指标至少拆成四层：proposal schema 解析率、宿主授权率、真实执行成功率和独立 verifier 通过率。最终回答正确但产生了越权副作用，不是成功；一次点击后页面恰好改变而没有可回放 receipt，也不能当作可靠成功。`Structured Outputs` 只减少格式错误，不能保证动作语义；`Function Calling` 只承载调用合同，不能授予手机、桌面或网络权限。

Qwen3.7 Plus 当前只有 AA 精确榜单条目，DataCurve 没有精确 `mini_swe_agent_qwen3_7_plus_*` 行。因此本章中的 GUI/mobile harness 是教学和系统设计闭环，不把其他 Qwen 的 Agent 分数迁移过来，也不把官方产品描述写成真实设备 acceptance 或公开的视觉 policy 架构。

## GPT-6 Sol：复杂 coding 与 Agent 工具边界

OpenAI 官方将 GPT-6 Sol 定位为复杂 coding 与 agentic workflows，并在模型页列出 Responses API 下的 web search、file search、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search。这里要先做一层拆分：模型页的工具列表是 capability surface；工具真正能访问什么、是否需要审批、在哪个 sandbox 执行以及副作用是否成功，仍由宿主 runtime 决定。

一个可靠的 GPT-6 Sol coding agent 可以按下面的链路审计：

```text
model proposal -> typed response item -> tool/schema validation
-> permission and sandbox -> executor receipt
-> patch/test/artifact -> independent verifier
```

Responses API 适合承载 built-in tools 和 function calling；模型页同时保留 Chat Completions，但对 GPT-6 Sol 的 function calling 只在 `reasoning_effort=none` 时支持。迁移时不能只替换 model ID，还要检查 endpoint、reasoning effort、tool schema、response item、错误和回放语义。一个能生成 shell 命令的模型不等于已经获得 shell 权限。

Tool search 的工程价值在于延迟暴露大型工具库的 schema：namespace 描述和 `defer_loading` 可以让模型先发现，再加载需要的工具定义。它减少初始上下文，但增加了 tool registry、schema hash、search call、权限校验和 cache prefix 的状态。因此 trace 至少需要保存 `tool_search_call`、加载的 schema、原始 `call_id`、权限决定、执行 receipt 和 verifier 结果。

GPT-6 Sol 的长上下文也不能替代 coding agent 的状态设计。1,050,000 context、922,000 maximum input 和 128,000 maximum output 要与 reasoning tokens、工具输出、patch、测试日志和 compaction item 一起算账。若 compaction 或预算耗尽，harness 必须明确是继续、重试、缩小任务、转人工还是停止，不能把空响应当作代码正确。

资料边界：GPT-6 Sol 当前是 AA 单榜资料级闭环，DataCurve 没有精确 Agent 行。上面的工具和 verifier 逻辑是官方 API/runtime 合同与工程设计，不是 GPT-6 Sol 的内部训练方法、参数规模或架构披露。

## 7.24 Claude Opus 5.5：长任务 coding 的效率验收

Opus 5.5 的发布方描述把长任务 coding 的优势落在一次收集足够上下文、完成更大 patch、少重复尝试和少工具轮次。实现 coding agent 时，可以把这组描述转成四个可验证事件：`context_snapshot`、`patch_set`、`test_receipt` 和 `artifact_verifier`。只有测试与最终 artifact 通过，才把任务记为成功；少调用命令本身不是质量证明。

成本还要包含 cache read/write、可见输出、工具执行、失败重试和 fallback。若 Cyber safeguard 将请求路由到其他模型，trace 中必须保留 `actual_model`、`fallback_reason`、权限与工具版本。Artificial Analysis 的 `max with fallback` 和 Anthropic 的发布方 coding benchmark 都不能替代固定仓库、固定工具、固定 verifier 下的本地回放。

迁移到 Opus 5.5 时，agent loop 还要通过 API contract 门禁：thinking 不能关闭或改成手工 budget；`tool_choice` 不能使用 `any` 或指定工具强制调用；工具间进度可能以默认隐藏文本的 thinking block 返回；Claude API/Google Cloud 的旧 `computer_20251124` 要迁移为 `computer_toolset_20260801`。客户端应按 content block 的 `type` 解析并原样回传可保留的 thinking block，而不是假设每一轮只有 text -> tool_use -> text。

对长任务，on-demand compaction 会返回带签名的摘要 block，inline tools 可以在 mid-conversation system message 中增加或升级工具定义。两者都应写入 trace、cache key 和回放测试；否则看似减少了上下文或工具 schema 的 token，实际可能破坏 reasoning state 或 prompt cache。详见 [What's new in Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md)。

System Card 的 coding/GUI 结果进一步说明，Agent 成绩要写成条件事件：Terminal-Bench 4.0 `66.36%` 使用 xhigh、Claude Code `--bare`、5 trials；OSWorld 2.0 partial/strict 为 `81.8%/48.7%`，超过 100K tokens 后由 server-side compaction 维持轨迹；五 Agent team 的约 `2.7x` 和 DRACO 的约 `2.8x` 是 derived-latency 结果。Agent trace 还应记录 screenshot、context token、tool time、handoff、fallback 和 verifier，不能把更少步骤直接当作更可靠代码。

System Card 同时暴露了长轨迹、多 Agent、语言差异和模拟环境真实性的盲点。对 coding agent 来说，最小验收链仍是 `context_snapshot -> patch_set -> test_receipt -> artifact_verifier`；模型声称完成、工具返回成功、独立 verifier 通过必须分别计数。

## GPT-6 Luna：高吞吐 coding agent 的合同边界

Luna 官方定位为 focused/high-volume，模型页列出 Responses 的 hosted shell、apply patch、computer use、MCP 和 tool search；这些是 capability surface，不是宿主权限。coding agent 仍需把模型意图、permission、schema、executor、sandbox、副作用回执、artifact 和 verifier 分开记录，并固定 `gpt-6-luna` 的 effort、工具和 harness。

DataCurve 当前没有精确 `mini_swe_agent_gpt_6_luna_*` 行，不能把 GPT-6 Sol/Astra/GPT-5.6 的 Agent 分数迁移过来。Luna 的 1.05M/922K/128K 与 reasoning/tool/compaction tokens 一起进入预算；超过 272K 是整次请求的 input/cache 2x、output 1.5x。详见 [`gpt-6-luna-source-notes.md`](../../research/model-update-2026-09/gpt-6-luna-source-notes.md)。
