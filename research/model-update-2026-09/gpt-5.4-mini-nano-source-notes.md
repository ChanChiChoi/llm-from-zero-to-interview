# GPT-5.4 mini / nano 官方资料摘记

核验日期：2026-09-20。本笔记只升级已经出现在 Artificial Analysis 候选表中的 `GPT-5.4 mini` 与 `GPT-5.4 nano`。它们是 GPT-5.4 产品家族的关联档位；本轮没有把它们伪装成公开权重、独立架构或新的技术报告。

## 1. 榜单锚点与证据边界

| 对象 | Artificial Analysis | DataCurve DeepSWE | 当前归因 |
|---|---|---|---|
| GPT-5.4 mini | [`gpt-5-4-mini`](https://artificialanalysis.ai/models/gpt-5-4-mini)，当前详情标题为 `GPT-5.4 mini (xhigh)` | 当前快照只有 `mini_swe_agent_gpt_5_4_xhigh`，没有 mini 精确行 | AA 单榜；不把 base GPT-5.4 的 Agent 结果迁移给 mini |
| GPT-5.4 nano | [`gpt-5-4-nano`](https://artificialanalysis.ai/models/gpt-5-4-nano)，当前详情标题为 `GPT-5.4 nano (xhigh)` | 当前快照没有 nano 精确行 | AA 单榜；不把 base GPT-5.4 或 mini 的 Agent 结果迁移给 nano |

本轮通过 `10.24.27.134:7890` 重新抓取两个详情页，均返回 HTTP 200：

- mini：3,857,350 bytes，SHA-256 `a01a6da4a38077bc6309333d509bf303f8b1beb36729fd94f79fe7d3a8ffe0ee`；release date `2026-03-17`，AA Intelligence Index `24.0682169231964`，约 `212 tokens/s`，400,000 context，`parameters=null`，proprietary。
- nano：3,859,704 bytes，SHA-256 `2e6ddfefc9a3483437f0fe2648ce23ff42fe4f8614cec0c365dfc69777a166b4`；release date `2026-03-17`，AA Intelligence Index `20.7197352504557`，约 `167 tokens/s`，400,000 context，`parameters=null`，proprietary。

两页都将当前配置标记为 deprecated，并分别指向 GPT-5.6 Terra/Luna 的对应 effort 档位。deprecated、指数、速度、价格和 context 都是 Artificial Analysis 的第三方配置/目录字段，不是 OpenAI 的参数披露，也不等于 API 端点已经对所有账户下线。

DataCurve 首页快照 `/tmp/deepswe-current-7890.html` 为 268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；精确检索只有 `mini_swe_agent_gpt_5_4_xhigh`。因此不能把 GPT-5.4 base 的 234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、成本、输出 token 或 Agent steps 写成 mini/nano 成绩。

## 2. OpenAI 官方身份与接口字段

本轮通过 `10.24.27.134:7890` 获取官方 Markdown 页面：

| 来源 | 快照 | SHA-256 |
|---|---:|---|
| [GPT-5.4 mini 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-mini.md) | 3,781 bytes | `97dbda29af4009ed275112e92037a5b46f313caf0d1232ff9b54b899b4fb65a7` |
| [GPT-5.4 nano 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-nano.md) | 3,766 bytes | `3dd20e2c95f09f3efb19c6255387a4e07d4d7c213367cca0a015e162a8f5b4e0` |
| [GPT-5.4 模型页](https://developers.openai.com/api/docs/models/gpt-5.4.md) | 4,196 bytes | `c30e86b38bc6ceacb3d6db269aa6ba4a09c3c5e1322bfaf90f924fddce4013a5` |
| [Using GPT-5.4](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | 64,172 bytes | `ce273b29db0bd213d19c2fc2015f46a9921bbbed8e522d1abfface1037bc54f8` |
| [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md) | 70,253 bytes | `91604df954335250d16e33f3b07ebbfa3e6e2b7f821f0722ad7a69b9d586719a` |

官方模型页确认：

- `gpt-5.4-mini` 的默认 snapshot 是 `gpt-5.4-mini-2026-03-17`；`gpt-5.4-nano` 的默认 snapshot 是 `gpt-5.4-nano-2026-03-17`。
- 两者都是 text/image 输入、text 输出，400,000 context、272,000 maximum input、128,000 maximum output，知识截止为 2025-08-31，并支持 `none`（默认）、`low`、`medium`、`high`、`xhigh` reasoning effort。
- 两者支持 Chat Completions、Responses 和 Batch；不应把 GPT-5.4 base 的 1,050,000 context 或其超过 272K 的整次请求计费规则自动写到 mini/nano。400K 是这两个小档位自己的公开窗口字段。
- mini 的官方定位是高吞吐 coding、computer use 和仍需要较强 reasoning 的 Agent workflow；nano 的官方定位是速度/成本优先的 classification、data extraction、ranking 和 sub-agent。
- 官方文本 token 价格为：mini input `$0.75/M`、cached input `$0.075/M`、output `$4.50/M`；nano input `$0.20/M`、cached input `$0.02/M`、output `$1.25/M`。区域处理的 10% uplift 也属于官方计费条件。价格不是 Artificial Analysis 的 benchmark 结果。

### 能力清单不能从家族名自动继承

两页都列出 streaming、structured outputs、function calling、file search、file uploads、image input、web search 和 prompt caching。具体 Responses tool 清单存在差异：

- mini 页面明确列出 `tool_search`、`image_generation`、`code_interpreter`、`hosted_shell`、`apply_patch`、`skills`、`computer_use` 和 MCP。
- nano 页面列出 function calling、web search、file search、image generation、code interpreter、hosted shell、apply patch、skills 和 MCP，但当前页面没有把 `tool_search` 或 `computer_use` 列入其清单。

这说明生产 router 必须按精确 `model_id + snapshot + endpoint` 做 capability probe，不能看到“GPT-5.4 家族”就把 base 或 mini 的工具表复制给 nano。页面清单是 API 支持契约；真实执行仍由宿主授权、沙箱、审批、超时、审计和 verifier 负责。

## 3. 小模型最值得面试的技术点

### 3.1 模型路由应按任务形状，而不是只按价格

官方 GPT-5.4 指南把 mini/nano 放在不同的工作负载位置：mini 适合结构清晰的 coding、computer use 和 Agent workflow；nano 适合边界明确、吞吐量大的窄任务。可以把路由写成：

```text
request -> task-shape classifier -> capability probe -> mini/nano/base route
        -> fixed effort + tool contract -> verifier -> escalation/retry
```

真实 router 至少要观察输入长度、模态、是否要调用工具、歧义程度、规划深度、失败代价和单位成功成本。nano 便宜不代表适合把隐含规划、复杂冲突消解或开放式研究全部压给它；mini 也不应因为支持更多工具就绕过权限和结果校验。

### 3.2 小模型需要更显式的 prompt contract

官方指南明确说 mini/nano 比大型模型更不容易自动补齐缺失步骤、隐含消歧或按预期包装输出。mini 更 literal、少做假设，可能在没有明确禁止时继续追问；nano 应只用于窄而明确的任务，优先输出 label、enum、短 JSON 或固定模板，复杂规划应路由到更强模型。

因此 prompt 不是一句“请认真完成”就够，而应明确：

1. 任务目标与关键规则；
2. 工具调用和副作用的完整顺序；
3. 何时询问、继续或 abstain；
4. 输入缺失和工具失败如何恢复；
5. 输出 schema、长度和是否允许追问；
6. 一个正确示例与停止条件。

这是模型行为与 harness 共同形成的工程方法，不是 mini/nano 内部训练 loss 的公开证明。

### 3.3 reasoning effort 仍不是严格 token 预算

两者都支持 `none` 到 `xhigh` 的 effort，但 effort 是行为/推理投入旋钮，不是可以直接拿来做并发容量的硬上限。容量账本仍需拆开 input、reasoning、visible output、tool schema、tool result、retry、cache hit 和 executor wait。`max_output_tokens`、context window 与 effort 也不能相加成一个“总智力预算”。

对于 nano，若把 effort 直接调高来弥补开放任务的规划不足，可能得到更慢、更贵但仍不稳定的系统；应先收窄任务、补齐结构化输入和 verifier，再比较 effort 曲线。

### 3.4 400K context 与 GPT-5.4 base 的 1.05M 分开记录

家族指南讲的是 GPT-5.4 系列新能力，包括长上下文、tool search、computer use 和 compaction；mini/nano 的模型页则给出自己的 400K context 和工具清单。面试中应主动说出“家族指南 ≠ 每个 sibling 的相同端点/窗口”。任何迁移都要固定 model ID、snapshot、tool catalog、reasoning effort、compaction、harness、provider 和 verifier。

## 4. 评测与负面证据

应分别记录：

```text
model_id + snapshot + effort + endpoint + tool capability
+ prompt contract + harness + task set + verifier
+ input/reasoning/output tokens + cache + latency + unit-success cost
```

Artificial Analysis 的指数、速度、deprecated 标志和目录参数是第三方配置字段；DataCurve 的 xhigh 行是 GPT-5.4 base + `mini-swe-agent` 系统结果。二者都不能证明 mini/nano 的裸模型能力。

本轮没有找到 GPT-5.4 mini/nano 专属参数、层数、稠密/MoE 结构、attention 变体、预训练数据、完整 post-training recipe、system card、技术报告、公开权重或独立 benchmark。官方资料公开的是 API snapshot、能力矩阵、预算和小模型 prompting guidance；不能从 “mini/nano” 名称推断内部缩放方法。

## 5. 面试追问

1. **为什么 nano 不应默认承担开放式 Agent 规划？** 因为官方定位是窄、边界明确、高吞吐任务；规划、消歧和隐含步骤推断弱于大模型，应该由 router 升级或由更强模型规划、nano 执行固定子任务。
2. **mini 和 nano 是不同 reasoning checkpoint 吗？** 公开资料只支持两个独立 API model ID/snapshot 与不同能力/价格契约；没有公开训练或权重报告，不能进一步断言内部结构。
3. **能否把 GPT-5.4 base 的 tool_search 迁移到 nano？** 不能直接迁移。当前 nano 模型页工具清单没有列出 `tool_search`；应按精确 model page 做 capability probe，并在不支持时采用显式、受控的工具子集。
4. **为什么 mini/nano 都是 400K，却不能共用所有 serving 配置？** context 容量相同不代表工具、限流、价格、kernel、provider、输出行为和 SLO 相同；manifest 必须按 snapshot 与 endpoint 分开。
5. **DataCurve 的 GPT-5.4 xhigh 分数能否作为 mini/nano 成绩？** 不能。它绑定 base GPT-5.4、xhigh、`mini-swe-agent`、工具/环境和 verifier，当前没有 mini/nano 精确行。
6. **更高 effort 能否解决 nano 的所有问题？** 不能。effort 不能替代任务边界、清晰 schema、权限门禁、工具回执和独立 verifier。

## 6. 闭环状态与书系映射

当前状态：**资料级闭环（AA 单榜关联档位）**。理由是两个对象均有 Artificial Analysis 锚点、OpenAI 官方模型页、研究笔记、评测边界和书系配套映射；没有精确 DataCurve 行、独立架构/训练报告或裸模型 benchmark，因此不新增 mini/nano 专属 Transformer 章节。

映射位置：

- 第四册：模型家族、snapshot、能力矩阵与第三方榜单归因；
- 第十六册：effort、reasoning/output/context 预算；
- 第十七册：tool capability probe、router、tool contract 和 verifier；
- 第二十四册：小模型 serving manifest、400K context、缓存、路由、工具等待和单位成功成本。

完整资料来源见 [GPT-5.4 source notes](gpt-5.4-source-notes.md) 与本文件的链接清单。

