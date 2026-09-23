# 第 15 章 Kimi K3：模型发布信号、长任务 Harness 与证据边界

> 资料边界：本章依据 [Kimi 官方 K3 发布文章](https://www.kimi.com/en/blog/kimi-k3)、[Kimi K3 官方仓库](https://github.com/MoonshotAI/Kimi-K3)、固定 Hugging Face revision/config、`k3_tech_report.pdf`、Kimi K3 License、FlashKDA 和 vLLM recipe，以及 Kimi Linear 与 Attention Residuals 论文。2026-09-20 已补充权重仓库 metadata、KDA kernel 和 serving recipe；榜单分数、发布方 benchmark、实际权重加载和独立复现仍按各自证据边界记录。

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

**误区一：发布文章列出的每个技术都已在 K3 的所有层实现。** 现在 K3 技术报告已经给出 3:1 KDA/Gated MLA、Block AttnRes 和 Stable LatentMoE 的具体配置，但仍不能由报告补写完整训练 recipe、所有 kernel 版本或独立线上复现。

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

Kimi K3 的公开资料把 KDA、AttnRes、稀疏专家、低精度训练和长任务 Agent 放在同一个系统叙事中。2026-09-18 的技术报告已经让 3:1 KDA/Gated MLA、8 个 Block AttnRes、Stable LatentMoE 路由和 XTM channel 协议进入“官方报告明确披露”层；它仍没有公开完整训练数据、所有 kernel 版本和独立线上复现。学习时应沿报告、论文和工程仓库分别展开机制，再用版本化 harness、状态 manifest 和条件化评测把模型能力与运行时能力分开。

## 15.9 技术报告补证：协议不是隐藏状态的字符串拼接

K3 技术报告把工具交互写成 XTM 风格的消息结构，使用 `[open]`、`[sep]`、`[close]` 和 `[end_of_msg]` 等特殊 token。全局 option 可以声明工具和 `reasoning_effort`，一次请求的 option 可以声明 `tool_choice` 与 `response_format`。工具集还可以通过后续 `tool-declare` message 动态扩展，这使得工具 schema 的加载成为上下文协议的一部分，而不是初始化时一次性拼接的字符串。

assistant 消息分为 `think`、`response` 和 `tool` channel。thinking channel 即使为空也保留结构；tool call 和 tool result 通过 `tool/index` 配对。对 Agent harness 来说，最小可恢复状态至少应包括 channel、tool index、工具声明版本、reasoning effort、工具回执和权限决定。只保存最终可见回答，会导致重试时重复调用、跨模型切换时丢失工具归属，或者把旧工具结果错误地回灌给新请求。

报告还披露 K3 的长轨迹训练包含 web search、专业知识工作、软件工程与 kernel 优化、vision-in-the-loop、持久助手、web development 和 autonomous execution；轨迹可能包含数百或数千次工具调用和百万级上下文 token。训练基础设施使用 partial rollout、外部 KV-cache retention、adaptive throttling 和可恢复 microVM sandbox。这里能支持“长任务 harness 是模型能力的一部分”的面试论点，但不能把这些环境描述推导成公开的完整 RL loss、数据配比或线上成功率。

对于面试题“为什么 K3 的技术报告比发布文章更重要”，准确回答应是：发布文章给出方向和产品边界，报告给出可核验的配置与设计动机，官方实现仓库给出代码入口，榜单只给出配置级外部观察；四者不能互相替代。K3 License 已覆盖权重、参数、配置、代码和文档，但 README 的“完整权重已发布”仍不等于我们已经确认具体文件和 revision。

## 15.10 从状态 manifest 到 hybrid serving manifest

2026-09-20 的官方实现补证让 K3 的 Agent 问题多了一层容易漏掉的部署状态。Hugging Face API 已固定 revision `f831ab66814297da540d832a5235f8e904f29d06`，并列出 96 个 safetensors 分片；这解决了“模型身份/文件是否存在”的证据问题，但没有证明本地已经加载成功。

K3 的 serving manifest 还必须保存两类不同状态：

1. 模型/协议状态：model revision、reasoning effort、XTM channel、tool/index、schema version、权限决定和未完成副作用。
2. 运行时状态：KDA recurrent state、MLA attention cache、prefix-cache block、DCP/TP/TEP/DEP/PP 拓扑、KV dtype、backend 和 GPU/driver。

vLLM recipe 明确 K3 是 hybrid 模型：MLA attention 与 KDA recurrent state 由两个 KV-cache group 管理；Blackwell 通过 `--prefix-match-unit 128` 调整粗粒度 prefix hit，decode-heavy 场景可用 DCP 跨 TP rank 分片 decode cache。它还提示 K3 偶尔产生 parser 不期望的 tool-call 格式，因此“模型输出了 tool call”仍必须经过 schema validation、retry/idempotency 和 verifier。

面试中可以用下面的因果链回答长任务故障：

```text
模型 revision/config
    -> KDA state + MLA cache
    -> prefix/DCP/parallel topology
    -> tool parser/schema/permission
    -> execution receipt
    -> verifier/artifact
```

如果只保存最终文本，可能丢失 reasoning/tool 状态；如果只保存 GPU KV state，可能无法重放权限和工具副作用；如果只验证 parser 成功，又不能证明工具真的执行或 artifact 正确。K3 的可恢复 Agent 必须同时审计协议状态、缓存状态和外部执行状态。

## 15.11 用固定 manifest 审计长任务恢复

K3 的固定权重 index 让“恢复一个 Agent 会话”可以拆成两个互不替代的检查。第一检查是模型 artifact：revision、config、分片总数、tensor key、packed weight/scale 配对和量化格式；第二检查是运行时状态：XTM channel、工具 schema、权限决定、KDA recurrent state、MLA cache、prefix-cache block 和 sandbox checkpoint。

零依赖脚本 [`kimi_k3_manifest_audit.py`](../../research/model-update-2026-09/code/kimi_k3_manifest_audit.py) 只读取 `config.json` 与 `model.safetensors.index.json`，不会触碰 TB 级权重。它通过了 96 个连续分片、497,220 个 tensor、93 层、92 个 MoE layer 和 247,296 对 packed/scale tensor 的一致性门禁。这个结果只能证明 metadata 自洽，不能证明 GPU kernel、完整加载或线上服务已验收。

恢复故障的测试应分别注入：

1. index 与 config revision 不一致；
2. 某个 expert shard 或量化 scale 缺失；
3. MLA cache 存在但 KDA recurrent/conv state 缺失；
4. KDA state 存在但 prefix-match unit、KV dtype 或 backend 不一致；
5. tool/index 被重排、schema 版本变化或未完成副作用未知；
6. 模型输出可解析但 artifact verifier 失败。

前四类属于模型/serving 状态门禁，后两类属于 Agent 协议/执行门禁。把它们都归结为“模型上下文丢了”会让故障无法归因，也容易导致系统在外部副作用未知时错误重试。

## 15.12 2026-09-23 当前榜单复验：指标、限制与可比性

本轮 K3 的 AA 与 DataCurve 页面重新抓取后没有新增 canonical 模型，也没有发现官方 artifact revision 变化。AA `max` 的 Intelligence Index、速度和成本是第三方/provider 观察字段；DataCurve 的 `309/451`、Pass@1/Pass@4、平均 token、steps 和成本则属于 `mini-swe-agent + tools + environment + verifier` 系统。面试回答中应把它们写成两本不同的账，不能把 Pass@4 写成裸模型准确率。

官方发布材料中的“约 `2.5x scaling efficiency`”同样应标为发布方声明。若没有相同硬件、序列长度、backend、warmup/迭代、harness、任务集和 verifier 的复现，就不能把它当成普遍吞吐或质量提升。发布评测混用了 Kimi Code、Claude Code、Codex，以及 H20/H100 和 compaction 条件，跨表格拼接排名会把 harness 差异误当成模型差异。

K3 还给出了三个直接影响长任务设计的限制：preserved thinking history 需要按协议原样保留结构化思考/工具历史；中途从其他模型切换到 K3 可能造成质量不稳定；模型可能 excessive proactive。第一项进入 replay/schema 账本，第二项进入跨模型迁移门禁，第三项进入权限、预算和 verifier，而不是简单归咎于“上下文不够长”。

面试中可以用一句完整结论收口：**K3 当前是榜单观察稳定、官方 revision 未漂移、内容专题已闭环；它仍不是完整权重、目标硬件、双状态恢复或生产 Agent acceptance 已证明。**
