# DeepSWE v1.1：公开排行榜快照与证据边界

核验日期：2026-09-14。来源为 DeepSWE 公共页面 [deepswe.datacurve.ai](https://deepswe.datacurve.ai/) 的本地 HTML 快照 `/tmp/deepswe.html`；页面自身标注排行榜更新时间为 2026-09-03。当前网络无法复访该域名，因此下面的数值只代表这份页面快照，不称为 2026-09-14 实时榜单。

## 页面明确写出的评测范围

- 页面版本为 `v1.1`，显示 113 个任务、91 个仓库、5 种语言；页面的模型筛选器显示 28 个模型，当前快照的主表显示 21 个可见配置。
- 页面数据对象记录排行榜生成时间 `2026-09-03T22:24:37.984682+00:00`，并把 GPT-6 Astra 的最近一次任务作业标为 `20260901-deep-swe-1-1-gpt-6-astra`（完成于 `2026-09-01T07:35:13Z`）。这些是快照元数据，不是所有配置共享同一运行时间。
- 所有模型均使用 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 运行，以保持 harness 一致。页面将 DeepSWE 描述为长周期软件工程 benchmark：任务从头编写、覆盖多仓库和多语言，并由手写 verifier 检查软件行为。
- 页面主表字段为 `Pass@1`、平均每任务成本、输出 token 和 Agent steps。分数带有页面给出的不确定性区间；成本和 token 受供应商、模型配置、工具预算与运行时影响。
- 任务不是从已有 commit 或 pull request 改写而来；这只是页面对污染控制的声明，不能单独证明所有训练污染风险为零。

## v1.1 页面快照中的 21 个配置

下表逐字整理页面主表的配置字段。它们是模型、推理档位和统一 Agent harness 的组合，不是 21 个独立基础模型。

| 页面配置 | effort | Pass@1 | 区间 | 平均成本/任务 | 输出 token | Agent steps |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-6-astra` | xhigh | 74% | ±3% | $6.52 | 30k | 29 |
| `gemini-3.8-flash` | high | 74% | ±1% | $2.36 | 143k | 166 |
| `claude-opus-5` | max | 74% | ±4% | $11.84 | 118k | 99 |
| `gpt-5.6-sol` | max | 73% | ±3% | $6.46 | 60k | 61 |
| `claude-fable-5` | xhigh | 70% | ±3% | $13.41 | 80k | 68 |
| `glm-5.3` | max | 69% | ±3% | $3.99 | 80k | 124 |
| `kimi-k3` | max | 69% | ±5% | $4.65 | 81k | 98 |
| `grok-4.6` | medium | 67% | ±2% | $3.45 | 50k | 70 |
| `gpt-5.6-luna` | max | 67% | ±4% | $0.61 | 73k | 102 |
| `gpt-5.5` | xhigh | 67% | ±6% | $7.23 | 46k | 82 |
| `gemini-3.7-flash` | medium | 65% | ±3% | $2.03 | 94k | 117 |
| `glm-5.3-flash` | max | 63% | ±4% | $0.24 | 73k | 123 |
| `deepseek-v4-pro` | max | 63% | ±6% | $1.67 | 106k | 155 |
| `claude-opus-4.8` | max | 59% | ±2% | $13.22 | 135k | 120 |
| `qwen3.8-max` | xhigh | 57% | ±3% | $3.73 | 95k | 111 |
| `muse-spark-1.2` | xhigh | 55% | ±2% | $3.70 | 99k | 101 |
| `claude-sonnet-5` | max | 54% | ±4% | $26.40 | 214k | 268 |
| `deepseek-v4-flash` | max | 53% | ±4% | $0.46 | 108k | 153 |
| `gemini-3.6-flash` | high | 47% | ±4% | $2.21 | 96k | 117 |
| `glm-5.2` | max | 44% | ±2% | $3.92 | 78k | 129 |
| `gemini-3.5-flash` | high | 36% | ±4% | $3.45 | 76k | 105 |

## Grok 4.6 的嵌入配置明细

主表展示的是页面默认可见配置；同一 HTML 快照的嵌入数据还包含 Grok 4.6 的四档 effort。原始配置字段如下，仍然属于 `mini-swe-agent` + 工具 + verifier 的系统结果：

| 配置 | effort | 通过/尝试 | Pass@1 | Pass@4 | 平均成本 | 平均输出 token | 平均 Agent steps |
|---|---|---:|---:|---:|---:|---:|---:|
| `mini_swe_agent_grok_4_6_low` | low | 187/449 | 41.648% | 69.027% | `$1.0424` | 16,458 | 44.19 |
| `mini_swe_agent_grok_4_6_medium` | medium | 305/452 | 67.478% | 84.071% | `$3.4490` | 49,764 | 70.29 |
| `mini_swe_agent_grok_4_6_high` | high | 294/451 | 65.188% | 84.956% | `$4.3849` | 61,161 | 78.96 |
| `mini_swe_agent_grok_4_6_xhigh` | xhigh | 301/451 | 66.741% | 84.956% | `$5.4977` | 71,404 | 87.22 |

这些原始行来自 2026-09-03 页面数据对象；它们不应与 xAI 2026-08-12 发布页的四舍五入 DeepSWE 1.1 图表值直接合并。详情与快照哈希见 [`grok-4.6-source-notes.md`](grok-4.6-source-notes.md)。

## 正确的比较口径

DeepSWE 的观测对象应写成：

```text
result = F(base_model, model_revision, effort, mini_swe_agent, tools,
           task_set, verifier, timeout, retry, context_policy, provider)
```

因此，表中的 `claude-sonnet-5`、`gpt-6-astra` 或 `kimi-k3` 分数不能直接作为基础模型排名；它们还包含 effort、工具宿主、上下文策略、超时、重试和 verifier。即使页面统一使用 `mini-swe-agent`，供应商 API、模型版本、限流和工具返回仍可能不同。

成本、输出 token 和 Agent steps 也不能单独解释质量。较高 token 可能换来更高成功率，也可能表示循环、重试或上下文管理效率较差。评测报告应同时保留成功/失败样本、工具错误、超时、verifier 拒绝和最终 artifact。

## 局限与待核验

- 页面快照没有给出每个配置的完整模型 revision、供应商端点、系统提示、工具 schema、重试策略和硬件清单；这些字段需要在目标环境中复测。
- `Pass@1` 和区间是页面发布方的评测结果，不是本项目独立复现；不能与 Artificial Analysis、SWE-bench 或其他 harness 的分数直接拼接。
- 快照内部还保留了更高精度的运行字段。例如 `kimi-k3` 记录为 309/451 次通过（Pass@1≈68.5%，Pass@4≈89.4%，4 次整套重复运行，95% run-to-run 区间约 ±4.5%）；表格中的 69% 是展示四舍五入。引用时应优先保留分子/分母、重复次数和区间，避免把四舍五入后的展示值当作精确测量。
- 页面关于任务原创性和污染控制的说明属于 benchmark 设计声明，仍需检查数据发布、训练语料和时间切分才能讨论污染风险。
- `claude-fable-5`、`claude-opus-4.8` 等页面配置与本项目其他来源中的模型 ID/版本命名可能不同；正式写作必须保留原始字符串并注明来源，不自动映射成相近名称。

## 书系映射

- 第七册：Pass@1、区间、成本/质量曲线与固定评测协议。
- 第十七册与第二十册：统一 Agent harness、工具回执、上下文压缩、重试和 verifier。
- 第六册与第二十四册：输出 token、TTFT/TPOT、KV/cache 和单位成功成本。
- 第四册及 `inventory-interpretation.md`：基础模型、推理配置与 Agent 系统三层记录。
