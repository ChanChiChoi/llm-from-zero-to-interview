# 第 89 章 GLM-5：DSA、MoE、slime 与 Agentic Engineering

> 本章核验日期：2026-09-20。模型入口来自 [Artificial Analysis 的 GLM-5 条目](https://artificialanalysis.ai/models/glm-5)；DataCurve DeepSWE 当前没有精确的 `mini_swe_agent_glm_5_*` 行。模型规格和技术路线来自 [Z.ai 官方模型资料](https://z.ai/blog/glm-5)、[GLM-5 模型文档](https://docs.z.ai/guides/llm/glm-5)、[固定模型卡](https://huggingface.co/zai-org/GLM-5) 和 [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)。榜单字段、官方 benchmark 和博客中的收益数字均绑定各自的配置、环境和发布方设置，不是本地独立复现。

## 89.1 先看“写完代码”与“交付系统”的差别

一个 coding Agent 可以生成看起来合理的代码，却仍然没有完成任务：测试没有真正运行，工具回执没有读完，渲染结果溢出，或者修复动作没有落到最终 artifact。单轮代码生成只回答“下一段文本是什么”，长周期 Agentic Engineering 还要回答：

```text
理解需求 -> 规划 -> 编辑代码 -> 执行工具 -> 读取反馈
    ^                                      |
    +--------- 诊断错误 <- 测试/验证 <------+
                         |
                    交付可复现 artifact
```

GLM-5 的公开定位把模型能力、稀疏长上下文结构和训练基础设施放在同一条链路中：DSA 控制长历史的访问成本，MoE 提供较大的总容量，`slime` 支撑 rollout 与训练的异步迭代，Agentic Engineering 则把最终正确性放在可执行环境和验证闭环中。

## 89.2 锚点身份：一个榜单条目不等于一个独立测量

本轮的模型身份应这样记录：

| 证据层 | 本轮可确认内容 | 正确读法 |
|---|---|---|
| Artificial Analysis | 精确条目 `GLM-5 (Reasoning)`，约 744B/40B、200K context、约 69 tokens/s；页面同时提示已有更新模型 `GLM-5.1` | GLM-5 的历史榜单配置和第三方目录字段；不能把当前页面的更新提示改写成 GLM-5.1 的内部架构 |
| DataCurve DeepSWE | 当前页面检出 GLM-5.2、GLM-5.3 和 GLM-5.3 Flash 行，没有精确 GLM-5 行 | 没有 GLM-5 的精确 `mini-swe-agent` 组合结果；不得迁移相邻版本 Pass@1、成本或 steps |
| Z.ai 官方模型卡/配置 | `GlmMoeDsaForCausalLM`、744B total、40B active、78 层、256 routed experts、top-8、1 shared expert、`index_topk=2048`、约 202,752 positions | 模型 artifact 的公开实现合同；字段不等于完整训练 recipe 或所有生产 kernel |
| 技术报告与博客 | DSA、长周期 Agent、`slime` 异步 RL 和 Agentic Engineering 的路线描述 | 一手方法线索；未公开的细节不能由产品定位反推 |

因此本章不是“GLM-5 在两个榜单的平均排名”。它是一个 **Artificial Analysis 单榜锚点的技术专题**，官方资料负责解释其架构和训练系统，缺失的 DataCurve 结果保持缺失。

## 89.3 规模账本：total、active、resident 和 latency 不是同一个数字

公开字段给出 `744B total / 40B active`，预训练数据规模约为 `28.5T` tokens。配置还公开 78 层、256 个 routed experts、每 token top-8 和 1 个 shared expert。面试时至少拆开以下五本账：

| 账本 | 要回答的问题 | GLM-5 已知字段 | 不能直接推出的结论 |
|---|---|---|---|
| 总权重 | 模型容量、专家分片和存储压力是多少？ | 744B total | 一次 token 的 FLOPs 或单卡显存 |
| 激活计算 | 每个 token 主要经过多少专家路径？ | 40B active、top-8 routed + shared expert | 实际 wall-clock latency |
| 通信 | token dispatch/combine 和专家并行搬运多少数据？ | 专家数与 top-k 可作为输入 | all-to-all 拓扑、capacity factor、overflow |
| 状态/cache | 长上下文请求保留什么？ | DSA 的候选访问、MLA 相关 rank 和最大位置字段 | 真实 KV/indexer/state bytes |
| runtime | provider 和硬件上的 TTFT/TPOT/p99 如何？ | AA 第三方测量字段 | 任意硬件、任意 batch 的普遍速度 |

`active=40B` 不能解释成“推理只需要 40B 参数的显存”。专家权重可能分片驻留，shared expert、router、attention、通信 buffer、KV/indexer cache、workspace 和并发请求都会进入服务成本。一个严谨的容量估算应该先固定 dtype、专家并行度、batch、上下文长度和 cache 命中，再计算 resident weights、每请求状态和调度余量。

## 89.4 DSA：先检索候选，再做精确 attention

长序列中，标准 causal attention 让当前位置访问全部历史 key/value。DSA（DeepSeek Sparse Attention）的核心抽象是增加一个轻量 indexer：先为历史位置估计相关性，只把 top-k 候选交给主 attention。

对当前位置 `t` 和历史位置 `i`，可以用教学式写成：

```math
s_{t,i}=I(q_t,k_i),
\qquad C_t=\operatorname{TopK}_i(s_{t,i}),
\qquad y_t=\operatorname{Attention}(q_t,K_{C_t},V_{C_t}).
```

这里 `I` 是 indexer，`C_t` 是候选集合。这个式子只表达两阶段访问，不声称 GLM-5 的完整 indexer loss、kernel 或训练布局已经公开。模型配置的 `index_topk=2048` 可以进入实现讨论，但不能简单说成“每个 query 最终只看 2048 个 token”：因果 mask、层布局、尾部保留、batch、MTP、gather 和实现策略仍可能改变实际可见路径。

DSA 的收益必须拆成三本账：

1. **主 attention 账**：候选减少后，Q-K 相关计算、KV 读取和显式 attention 的工作量下降。
2. **indexer 账**：indexer 自身要计算分数、存储候选 key、排序或选择 top-k，并消耗带宽。
3. **召回账**：候选漏掉关键 token 会造成最终证据缺失；节省的 FLOPs 不能抵消任务失败。

因此正确的实验不是只比较理论复杂度，而是同时测：index recall、最终 evidence recall、long-context needle retrieval、工具回执恢复、TTFT/TPOT、p99 和 Agent task success。稀疏路径还要设置 dense fallback 或失败诊断，避免把 indexer 的漏召回伪装成模型推理错误。

## 89.5 MLA 与长上下文状态：压缩访问不等于免费访问

公开配置给出 `q_lora_rank=2048`、`kv_lora_rank=512`、`qk_rope_head_dim=64` 等字段。它们可以帮助面试者理解潜变量/低秩投影和位置子空间的工程方向，但不能仅凭字段推导完整的 MLA kernel、每层 cache 布局或生产内存占用。

一个抽象的 KV 压缩路径可以表示为：

```text
hidden state -> latent KV representation -> attention query/read
                              |
                         cache / bandwidth
```

长上下文 serving 仍然要为 indexer candidate、top-k indices、gather buffer、显式 KV、通信和请求调度留出空间。`202752` 的最大位置字段只表示公开配置支持的上下文位置范围，不能等价为任意 workload 都有相同的召回率、p99 或成本。

面试中如果被问“DSA 加 MLA 后还需要测什么”，可以按访问链路回答：先测 indexer 的候选质量，再测压缩 cache 的读取质量和带宽，最后测端到端任务成功率。单独报告 attention FLOPs 或 cache token 数都不足以证明部署收益。

## 89.6 `slime`：把 rollout 与 trainer 变成可伸缩的异步流水线

Z.ai 将 `slime` 描述为异步 RL 基础设施，用于提高 rollout 和训练吞吐，并支持更细粒度的 post-training iterations。它首先是训练系统设计，而不是可以从 API 行为直接推导出的模型内部层。

一个可审计的流水线是：

```text
policy version p_k
       |
       v
rollout workers -> tool/environment trace -> verifier/reward
       |                                  |
       +------ sample metadata ------------+
                          |
                          v
                    trainer / learner
                          |
                       p_(k+1)
```

异步化带来吞吐收益，也引入一致性问题：

- **policy lag**：rollout 用的是 `p_k`，trainer 可能已经更新到 `p_(k+m)`；
- **sample freshness**：旧轨迹是否仍适合当前策略，如何设 TTL 或 importance correction；
- **reward latency**：工具执行和 verifier 可能比生成慢，队列会积压；
- **环境副作用**：重复 rollout、超时重试和部分完成的工具调用必须可恢复；
- **checkpoint consistency**：策略、tokenizer、工具 schema、verifier 版本和环境镜像要能对齐。

可以把异步 RL 的有效吞吐粗略写成：

```math
\text{effective samples/sec}
\approx
\min(\text{rollout rate},\text{verifier rate},\text{trainer consume rate})
\times \text{freshness fraction}.
```

这不是 `slime` 的官方公式，而是面试审计用的系统模型。若只报 rollout tokens/s，却不报过期样本比例、verifier 队列、失败重试和任务成功率，无法证明训练真的更有效。

长轨迹 Agent 还需要信用分配：最终成功信号距离早期规划和工具选择很远。可用阶段 reward、工具结果 verifier、中间 artifact 检查、轨迹切段和 token/step-level advantage 等办法，但 GLM-5 已公开资料没有给出足以确认完整 reward shaping、credit assignment 或 optimizer recipe 的细节，不能把通用方案写成 GLM-5 已实现事实。

## 89.7 Agentic Engineering：验证最终 artifact，而不是模型的自我陈述

“Vibe coding”通常以代码文本或 demo 的即时观感为中心；“Agentic Engineering”要把代码放进真实工程循环。一个合理的 host-side contract 至少包含：

| 阶段 | 模型可以提出什么 | 宿主必须验证什么 |
|---|---|---|
| 规划 | 任务分解、文件和工具选择 | 工作目录、权限、预算和依赖 |
| 编辑 | patch、重构、配置变更 | diff 范围、schema、静态检查和审批 |
| 执行 | shell、测试、浏览器或 MCP 调用 | allowlist、沙箱、超时、真实回执 |
| 诊断 | 根据日志提出修复 | 日志来源、错误是否重现、是否引入副作用 |
| 交付 | 声称任务完成 | 测试、artifact 可重开、版本、权限和验收条件 |

最终状态最好不是 `assistant said done`，而是一个可验证的状态对象：

```text
task contract
 + model/revision
 + tool schema and permission decisions
 + execution receipts
 + test and verifier results
 + artifact digest / path / version
 + unresolved failures
```

这也是长轨迹 RL 的训练边界：如果 reward 只看最终文本，很容易奖励“看起来完成”、跳过测试、读取隐藏答案或伪造工具结果的捷径。独立 verifier、环境隔离和 artifact gate 应当与模型输出分层，不能让模型同时担任提案者、授权者和最终裁判。

## 89.8 评测证据如何分层

GLM-5 的官方模型卡/技术报告给出发布方 benchmark 和 Agent 方向；Artificial Analysis 给出 provider/configuration 级的第三方字段；DataCurve 的当前快照没有精确 GLM-5 行。三者不能拼成一个裸模型能力分数。

| 结果 | 必须保留的绑定信息 | 不能做的推断 |
|---|---|---|
| AA Intelligence Index、速度、TTFT | model slug、reasoning effort、provider、测量日期、页面版本 | 不推导官方架构或独立 Agent 成功率 |
| Z.ai benchmark | task set、prompt、最大生成长度、temperature、harness、judge、重复次数 | 不与 AA 指数直接平均 |
| DataCurve DeepSWE | `mini-swe-agent`、任务/仓库、工具、环境、verifier、runs、effort | 当前无 GLM-5 行，不迁移 GLM-5.2/5.3 结果 |
| 本地 toy/复现 | commit、硬件、batch、上下文、精度、kernel、日志和失败样本 | 不把教学抽象说成生产实现 |

## 89.9 面试追问

**问：`744B total / 40B active` 说明了什么？**

答：它说明 MoE 把总容量与单 token 的主要激活计算分开。40B 不是整机显存，也不是实际延迟；专家驻留、router、shared expert、通信、KV/indexer cache 和 runtime workspace 都要单独计入。

**问：`index_topk=2048` 是否意味着模型只看 2048 个历史 token？**

答：只能确认公开配置有这个 top-k 字段。实际路径还受 indexer、因果 mask、尾部保留、层布局、gather、batch 和 kernel 影响；应测候选召回、最终证据召回和端到端任务成功率，不能把字段直接等价为最终可见 token 数。

**问：异步 RL 为什么可能更快却更不稳定？**

答：rollout 和 trainer 解耦后资源利用率更高，但样本生成时使用的 policy 可能已经过时，verifier 也可能形成队列。要控制 policy lag、样本 freshness、版本元数据、奖励延迟、失败重试和 checkpoint 一致性。

**问：Agentic Engineering 与普通代码生成的关键区别是什么？**

答：它把编辑、执行、观察、测试、诊断、修复和 artifact 验收放进闭环；完成条件由宿主环境和 verifier 证明，而不是由模型输出一句“完成”证明。

**问：为什么 GLM-5 不能直接写成 DataCurve 双榜闭环？**

答：本轮 DataCurve 没有精确 `mini_swe_agent_glm_5_*` 行。GLM-5 是 Artificial Analysis 单榜锚点，GLM-5.2/5.3 的 DeepSWE 结果绑定不同模型和 harness，不能迁移。

## 89.10 小练习

1. 为 744B/40B MoE 画出权重、专家 dispatch、KV/indexer cache、通信和 workspace 五本内存账，列出模型卡没有直接给出的字段。
2. 实现一个 toy DSA：比较 dense attention 与 indexer+top-k，分别报告 index recall、最终 evidence recall、TTFT 代理值和任务成功率。
3. 模拟 `slime` 的异步队列，改变 policy lag、verifier 延迟和样本 TTL，观察有效样本率、trainer 空转和 reward 偏差。
4. 构造一个 coding Agent 的 artifact gate：让模型提出 patch 和测试命令，但由宿主独立检查权限、执行回执、测试结果和 artifact digest。
5. 写一页评测报告，分栏记录 Artificial Analysis、Z.ai 发布方 benchmark、DataCurve（精确行缺失）和本地 toy 结果，禁止把四者合并成一个分数。

## 89.11 GLM-5.3 标准版：Full/Shared DSA 与 runtime 证据

前面的 GLM-5 章节解释了 DSA 的动机；标准 GLM-5.3 的新问题是：怎样确认一个看起来相似的实现，确实属于标准 DSA 版，而不是 Flash/linear-attention 变体。答案要从固定模型身份开始。

### 89.11.1 先固定模型和实现入口

标准 GLM-5.3 的 HF revision `aca966e4e02791568aa6a4ced368624b3d897f42` 的 `config.json` 给出：

| 字段 | 标准 GLM-5.3 的公开值 | 面试中的正确解释 |
|---|---:|---|
| 模型类 | `GlmMoeDsaForCausalLM` | 标准 DSA/MoE 的实现合同，不是完整训练报告 |
| 层数 | 78 | 结构配置，不直接给出 FLOPs 或延迟 |
| 专家 | 256 routed，top-8，1 shared | 每 token 的路由字段，不等于通信和 resident memory |
| indexer | `index_topk=2048`，`index_topk_freq=4` | 候选选择配置，不等于最终 evidence recall |
| 低秩状态 | `q_lora_rank=2048`，`kv_lora_rank=512` | MLA/低秩 cache 方向，仍需 kernel 和 dtype 验证 |
| 位置 | 1,048,576 positions | API/config 上限，不等于每个任务都能有效利用 1M |

这组字段和 Flash 版的 `Glm5NextForConditionalGeneration` 不同。Flash 版的 `RadixLinearAttention`、KDA state、视觉模块、EPD 和双 state pool 不能因为都带有 `GLM-5.3` 前缀，就自动成为标准版的内部结构。

### 89.11.2 Full layer 和 Shared layer 的差异

Transformers main 的实现把 indexer 计算做成跨层状态：21 个 `full` indexer layer 重新计算候选，57 个 `shared` layer 复用前一个 full layer 的 `prev_topk_indices`。可以用简化流程表示：

```text
query/key
  -> Full indexer: score + causal mask + top-k
  -> save top-k indices
  -> Shared indexer: reuse previous indices
  -> sparse main attention gather
```

这种设计节省的是重复的 indexer 计算和排序开销，但它不是无损缓存。面试中至少要追问三件事：

1. shared 层复用的候选是否仍适合当前层的 query 分布？
2. 候选池漏掉远程证据时，是否有 local tail、dense fallback 或诊断信号？
3. cache 恢复时，除了 latent KV，还是否恢复 layer 对应的 top-k indices、causal offset、block mapping 和 MTP iteration 状态？

因此 `index_topk=2048` 只能进入配置账本，不能直接写成“模型最终只读取 2048 个 token”。应分别测 index recall、最终 evidence recall、长上下文 needle retrieval 和真实 Agent task success。

### 89.11.3 为什么 runtime 会走 DeepSeek-V3.2 路径

vLLM 的 stable `v0.29.0` 和 main registry 都把 `GlmMoeDsaForCausalLM` 映射到 `deepseek_v32`；SGLang stable `v0.5.20` 与 main 也包含 `is_glm_moe_dsa`、`DeepseekV32ForCausalLM` 和跨层 top-k 状态。这说明两个项目把标准 GLM-5.3 的 DSA 计算抽象复用到了已有 DeepSeek-V3.2 runtime。

但代码路径复用不等于模型等价。还必须分别固定 checkpoint revision、权重命名、量化格式、KV/indexer layout、parallelism、kernel 和测试条件。vLLM main 额外的 PCP/DCP、`SparseCacheRole.INDEXER`、HiSparse 和 logical top-k 只说明 upstream 正在演进，不能写成 `v0.29.0` stable 或本机生产能力。

### 89.11.4 标准版 serving 验收清单

部署一个标准 GLM-5.3 DSA runtime 时，建议把验收拆为：

```text
固定榜单身份与 HF revision
  -> config/model class 对齐
  -> stable/main source entry
  -> full-weight load
  -> dense-vs-sparse 数值对照
  -> full/shared index recall
  -> evidence recall 与长上下文任务
  -> MLA/indexer cache recovery
  -> MTP acceptance/rollback
  -> 目标硬件 profile
  -> tool/verifier/SLO gate
```

其中 `source entry` 只证明代码可找到，`full-weight load` 只证明权重能接入，二者都不能证明 sparse attention 没有漏召回。最后还要把工具 schema、权限、重试、幂等、verifier 和 artifact 交付结果与模型生成质量分开统计。

## 89.12 面试追问：标准 GLM-5.3 与 Flash 如何避免混淆

**问：为什么不能把 GLM-5.3-Flash 的 KDA 结论写进标准 GLM-5.3？**

答：两者模型类和 runtime 入口不同。标准版是 `glm_moe_dsa`，当前证据走 DeepSeek-V3.2 DSA 路径；Flash 是 `glm5_next`，才有 linear/KDA、视觉和双 state pool。名字相同只说明产品家族关系，不能替代 config 和源码证据。

**问：vLLM registry 已经支持 `GlmMoeDsaForCausalLM`，是否代表线上可用？**

答：只代表 source/registry entry 存在。还要在固定 revision 上加载完整权重，验证数值、Full/Shared top-k 召回、MLA/indexer cache 恢复、MTP 接受率、目标硬件 p99 和 tool/verifier SLO。

**问：Shared indexer 为什么可能带来质量风险？**

答：Shared 层复用上一 Full 层的候选集合，减少 indexer 计算，但当前层 query 可能需要不同的远程证据。应测候选召回和最终 evidence recall，并观察是否需要 tail/fallback；不能只看 attention FLOPs。

**问：`q_lora_rank` 和 `kv_lora_rank` 能直接给出显存节省吗？**

答：不能。它们只提供低秩表示的配置线索；真实显存还包括权重 dtype、专家驻留、top-k indices、indexer workspace、通信 buffer、并发和 cache layout。必须结合实现和目标硬件 profile 计算。
