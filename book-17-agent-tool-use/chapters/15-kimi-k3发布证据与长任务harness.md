# 第 15 章 Kimi K3：模型发布信号、长任务 Harness 与证据边界

> 资料边界：本章依据 [Kimi 官方 K3 发布文章](https://www.kimi.com/en/blog/kimi-k3)、其官方博客索引以及 Kimi Linear 与 Attention Residuals 论文。发布文章中的参数、能力和评测数字属于厂商公开披露；除非有模型卡、权重或技术报告交叉确认，否则不把它们写成独立复现实验结论。

## 15.1 为什么模型发布文章不能直接当作架构报告

一篇模型发布文章通常同时承担三种任务：告诉用户模型能做什么，告诉工程师怎样接入，告诉研究者团队认为哪些技术值得关注。这三种信息的证据强度不同。

Kimi K3 发布文章提到总参数、原生视觉、百万 token 上下文、Kimi Delta Attention（KDA）、Attention Residuals（AttnRes）、Stable LatentMoE、量化感知训练和长任务 Agent 限制。这些信息足以建立研究路线，却不足以让我们计算每层张量形状、复现完整训练配方或断言线上所有实现细节。

阅读时应先建立“字段—证据”表：

| 字段 | 当前可说的内容 | 仍需什么证据 |
| --- | --- | --- |
| 模型名称与发布文章 | Kimi K3 是文章明确讨论的版本 | 固定 API/权重版本 |
| KDA、AttnRes | 发布文章提到，论文可分别解释一般机制 | K3 的层配置和实现 |
| Stable LatentMoE、量化、SiTU 等 | 文章给出的技术信号 | 完整报告、配置、代码和消融 |
| 长上下文与视觉 | 文章公开的产品能力描述 | 任务级有效能力和多模态 token 预算 |
| DeepSWE 分数 | 绑定文章脚注中的 harness 和时间语境 | 同版本、同 harness 的独立复测 |
| 权重与许可证 | 文章曾给出发布承诺 | 实际仓库、许可证和可下载 artifact |

这张表的作用是防止两个常见错误：把“提到某技术”写成“所有层都采用该技术”，把“某次 benchmark 得分”写成模型在任何环境下的稳定能力。

## 15.2 把长任务看成模型、Harness 和环境的合成

代码 Agent 的结果不是模型单独决定的。更准确的教学抽象是：

```math
R=F(M,H,E,B,D),
```

其中 `M` 是模型版本，`H` 是 harness（提示、工具包装器、记忆、验证器和协议），`E` 是环境，`B` 是 token、时间和工具预算，`D` 是任务分布。只要其中一项变化，成功率、成本和错误类型都可能变化。

例如同一个 K3 请求，在一个保留完整思考历史的客户端中继续执行，和从另一个模型的中间会话直接切换，可能拥有不同的状态信息。发布文章特别提醒了思考历史回传和跨模型切换的影响；这不是“上下文窗口够大”就能自动解决的协议问题。

一个可恢复的 harness 至少要保存：

1. 不可变的用户目标、禁止事项和权限；
2. 模型 ID、effort、prompt/template 和工具 schema；
3. 当前 workspace、diff、测试结果和 artifact hash；
4. 每次工具调用的 call ID、授权、开始/结束状态和幂等键；
5. 压缩前后的状态摘要及其原始 trace 引用；
6. 失败原因、剩余预算、人工接管和最终提交状态。

这些字段比“模型说它已经完成”更接近可审计事实。

## 15.3 保留思考历史是协议契约，不是把隐藏思维全部暴露

“保留思考历史”容易被误解成把模型内部的全部 chain-of-thought 原样发送给下一轮。工程上更稳妥的定义是：保留下一步决策所需的结构化状态，例如已经确认的事实、未解决假设、工具 observation、代码 diff、验证结果和待执行动作。

可以将一轮状态表示为：

```math
s_t=(g_t,f_t,h_t,p_t,b_t,r_t),
```

`g_t` 是目标与约束，`f_t` 是已确认事实，`h_t` 是假设，`p_t` 是计划，`b_t` 是预算，`r_t` 是权限和版本。上下文压缩的目标不是让摘要最短，而是让下一步在关键事实上与完整历史保持一致。

跨模型切换时至少要检查：

- 新模型是否理解旧模型的 role、tool call 和 observation schema；
- reasoning item 是否需要回传，哪些字段属于可见协议；
- tokenizer、chat template、工具名称和参数是否兼容；
- 旧 trace 中的未完成副作用是否已查询状态；
- 新模型的主动性是否会触发额外工具调用或越权风险。

如果这些契约没有检查，切换失败很难归因于模型能力还是状态丢失。

## 15.4 从 KDA/AttnRes 论文学习机制，但不移植 K3 配置

Kimi Linear 论文给出了 KDA 的递归状态和与全局 MLA 的混合思路；Attention Residuals 论文给出了沿深度选择历史表示的机制。它们能帮助我们理解 K3 发布文章中的技术名词，但论文实验的 head dimension、层比例、初始化、kernel 和训练规模不能自动移植到 K3。

对读者而言，可以把两条轴分开：

- KDA 沿**序列时间轴**维护固定大小的 key-value 状态，降低解码时对完整历史的访问；
- AttnRes 沿**网络深度轴**选择历史层表示，改变残差流的贡献分配。

两者都可能降低某类重复计算，却不解决同一个问题。KDA 仍可能遗忘远程精确 token，AttnRes 仍需要保存或传输深度表示。真正的系统评估要把状态大小、精确检索、prefill、decode、通信和质量一起测量。

## 15.5 评测数字怎样读才不夸大

发布文章中的分数至少要绑定五类条件：模型 revision、harness、环境硬件、推理档位和任务版本。若文章脚注说明不同测试混用了 H100/H20 或发生 fallback，就不能把所有分数拼成同条件排行榜。

建议把一次评测记录成结构化行：

```text
model_revision, harness_revision, environment, effort,
task_set, input_tokens, output_tokens, tool_calls,
success, failure_reason, latency, cost, fallback
```

评测报告还应区分绝对成功率和相对提升。若基线成功率是 20%，候选是 30%，相对提升是 50%，绝对提升却只有 10 个百分点。长任务中，工具次数、恢复成功率和单位成功成本常常比单一平均分更能解释上线价值。

发布文章还给出了一组产品接入快照：Kimi Work 桌面端要求 3.1.0 或更高版本，Kimi Code 通过 `/model` 选择 K3，API 模型名为 `kimi-k3`；文章列出的价格是 cache-hit 输入 $0.30/MTok、cache-miss 输入 $3.00/MTok、输出 $15.00/MTok，并自报 Mooncake 分离式推理架构在编码工作负载上的缓存命中率超过 90%。这些字段必须绑定文章版本、账户平台、缓存策略和采集日期；它们不是长期费率、普遍命中率或独立性能保证。

同一脚注规定文章中的 K3 benchmark 使用 `reasoning effort=max`、`temperature=1.0`、`top-p=1.0`，且不同基准分别采用 Kimi Code、Claude Code 或 Codex harness。没有这些条件时，只能引用“发布方报告了某个分数”，不能重算或横向合并。

下面的零依赖 demo 用三条 toy 记录检查一个常见错误：把不同 harness 或硬件下的分数放进同一张排名表。只有比较键完全一致时，程序才把结果标为可比。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalRow:
    model: str
    harness: str
    hardware: str
    effort: str
    task_set: str
    score: float


def comparable(left, right):
    keys = ("harness", "hardware", "effort", "task_set")
    return all(getattr(left, key) == getattr(right, key) for key in keys)


baseline = EvalRow("baseline", "mini-swe-agent-v1", "H100", "max", "tasks-v2", 0.20)
same_setup = EvalRow("k3", "mini-swe-agent-v1", "H100", "max", "tasks-v2", 0.30)
different_setup = EvalRow("k3", "kimi-harness-v2", "H20", "max", "tasks-v2", 0.34)

assert comparable(baseline, same_setup)
assert not comparable(baseline, different_setup)
absolute_lift = same_setup.score - baseline.score
relative_lift = absolute_lift / baseline.score
print(round(absolute_lift, 2), round(relative_lift, 2), comparable(baseline, different_setup))
```

输出 `0.1 0.5 False`：同条件下是 10 个百分点的绝对提升、50% 的相对提升；不同 harness 与硬件的 34% 分数不能直接加入这组比较。真实评测还应把工具调用、输入/输出 token、延迟、成本、fallback 和失败原因加入比较键或结果分层。

## 15.6 常见误区与面试追问

**误区一：发布文章列出的每个技术都已在 K3 的所有层实现。** 文章是技术披露入口，不是完整配置文件；应等待权重、模型卡或技术报告。

**误区二：百万 token 上下文等于能可靠检索百万 token。** 接口容量、有效检索、压缩误差、位置偏差和 serving cache 是不同指标。

**误区三：保留思考历史就是暴露全部隐藏推理。** 工程需要的是可验证的结构化状态和协议兼容，不是无条件复制隐藏 token。

**问：为什么 K3 的 DeepSWE 分数不能直接和榜单快照比较？**

答：两者可能使用不同模型 revision、harness、硬件、effort、任务版本和 fallback。应先锁定这些条件，再比较绝对成功率、失败类型、工具成本和恢复行为。

**问：KDA 和 AttnRes 是否解决同一个问题？**

答：不是。KDA 沿序列维护递归状态，AttnRes 沿网络深度选择历史表示；一个主要影响长序列状态访问，一个主要影响残差流和深度信息路由。两者可以组合，但需要独立消融。

## 15.7 小练习

1. 为一个跨模型迁移的代码任务写 state manifest，标出哪些字段可压缩、哪些字段必须保留原始引用。
2. 设计 KDA、全注意力和混合路径的同预算评测，加入精确检索、TTFT、TPOT、缓存字节和恢复成功率。
3. 给一组带 fallback 的 benchmark 结果，分别计算绝对提升、相对提升和单位成功成本，并说明哪些比较不可比。
4. 阅读 Kimi K3 发布文章，把每个技术字段标成“发布文章披露”“论文机制”“待核验配置”三类。

## 15.8 本章总结

Kimi K3 的公开资料最适合作为一张研究地图：它把 KDA、AttnRes、稀疏专家、低精度训练和长任务 Agent 放在同一个系统叙事中。学习时应沿论文和工程文档分别展开机制，再用版本化 harness、状态 manifest 和条件化评测把模型能力与运行时能力分开。这样既能理解前沿技术的方向，也能避免把发布信号、榜单行和内部实现混成一个未经验证的结论。
