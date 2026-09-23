# GLM-5.3：官方文档核验与技术扩写入口

核验日期：2026-09-20；2026-09-21 补充官方博客正文资源和评测脚注。研究底稿；正式面试落点复用第十六册长任务/验证器章节，并把训练/serving 证据映射到既有书系，不新增重复的基础架构章节。本轮补充的 benchmark 数值来自 Z.ai 官方文档、官方博客资源、Hugging Face 模型卡/config 和排行榜快照，均保留发布方与配置级边界。

## 来源与确认范围

已读取 [Z.ai 官方 GLM-5.3 文档](https://docs.z.ai/guides/llm/glm-5.3)。2026-09-20 首轮只读到博客前端壳，2026-09-21 通过官方正文资源补齐博客内容。Artificial Analysis 页面记录日期为 2026-08-18；该日期仍待独立发布记录交叉确认。

官方文档和 Hugging Face 模型卡均称 GLM-5.3 沿用 GLM-5.2 的基础模型，改进全部来自后训练。应据此组织“后训练如何改变 Agent 能力”的知识链，不能因为型号升级而推断它更换了基础架构。本轮已核验 GLM-5.3 的官方模型卡、`config.json` 和许可证名称字段；它们公开了推理/实现配置，但仍不等于完整训练数据规模、损失配方或生产 profiling。

本地保存的 Z.AI `llms.txt` 文档索引（`/tmp/zai-llms.txt`，读取日期 2026-09-14）单独列出 [GLM-5.3-Flash 文档入口](https://docs.z.ai/guides/vlm/glm-5.3-flash.md)、GLM-5.3 迁移指南、Thinking/Deep Thinking、流式、函数调用、缓存和结构化输出等页面，并把 GLM-5.3/GLM-5.3-FLASH 描述为可接入 Claude Code、Kilo Code、Cline、OpenCode 等 coding-agent 工具。该索引只能确认文档入口和官方产品接入范围；GLM-5.3-Flash 的参数、模态、发布日期、训练方法和独立评测仍需读取专属页面或模型卡后才能确认。

## 可直接追踪的版本差异

官方文档标明仅支持文本输入，上下文窗口为 1M token，最大输出为 128K token。这是接口标称容量，不能作为长上下文有效利用率的证明，也不意味着输入与输出预算可以无条件相加。

推理始终开启，`thinking.type` 只支持 `enabled`，`reasoning_effort` 支持 `low`、`high`、`max`，默认 `max`。文档明确提示：使用 `thinking.type: disabled` 的旧应用迁移前必须调整，否则请求失败。迁移示例应独立说明参数语义、错误路径和预算取舍，而不是只替换模型名称。

页面列出多种兼容协议，但其端点表与 Quick Start 示例使用的路径存在差异，并且提到部分订阅历史影响可用协议。因此还需核验接口文档和账户适用条件，不直接把页面示例当成已实测的通用配置。

## 后训练资料中的关键线索

官方介绍将训练环境扩展至生产工程与研究工作流：任务可能要求诊断训练系统瓶颈、修改实现、运行实验，并在保持正确性的前提下交付可测量的性能提升。这为讲解环境设计提供了具体场景，但不证明任意现实任务都可以自动构建。

环境生成管线由研究 Agent 收集工作模式并生成可运行环境，另由 judge agent 尝试求解以检查可解性。文档称部分任务的奖励也由管线生成；验证器生成时不访问参考解，求解轨迹用于发现奖励捷径。验证器需通过 oracle、no-op 和 unsolved-state 检查。这里应特别区分“一个解能通过”与“所有通过的解都正确”：前者不能证明验证器完备。

文档称继承 GLM-5.2 的 `SAO with compaction` 策略。现在已有 [SAO 论文](https://arxiv.org/abs/2607.07508) 对 Single-Rollout Asynchronous Optimization 的公开算法定义，但论文的直接训练对象和实验设置不能迁移成 GLM-5.3 专属结果；论文也没有公开 Z.ai 产品里的完整 compaction 实现。官方仍明确说环境生成和验证需要显著人工参与。

## 分章扩写方向

| 独立知识主题 | 小白入口与实验方向 | 专家深挖 | 书册 |
|---|---|---|---|
| 可执行训练环境 | 用修复小型程序的任务说明初始状态、动作、反馈和终止条件 | 状态隔离、可复现性、长程依赖和环境多样性 | 05、17、20 |
| 验证器与奖励捷径 | 比较只检查输出文本与检查真实程序行为的奖励 | 假阳性、假阴性、隐藏状态、对抗轨迹与污染 | 07、08、16 |
| 环境生成质量门禁 | 展示参考解通过、空操作失败、未完成状态失败的检查 | 这些检查的必要性与不充分性、人工审计 | 07、09、20 |
| 长任务上下文压缩 | 用多步调试过程解释丢失状态为何改变后续行动 | 状态保真度、压缩策略与训练分布一致性；SAO 公开算法已核验，5.3 专属 compaction 待核验 | 16、17、20 |
| 推理档位迁移 | 展示旧请求为何报错以及低档位的配置 | 预算、延迟、成功率及协议兼容性 | 06、22 |

以上表格是后续章节的入口，每个主题需要独立讲解、例子和实验，不直接以表格替代正文。

## 评测解读边界

官方文档页面快照（本地保存的 `glm-doc.html`，读取日期 2026-09-09）给出了公开长任务基准的前后数值：Terminal-Bench 3.0 从 4.6 提升到 28.3，DeepSWE v1.1 从 46.2 提升到 66.9，Agents' Last Exam 从 23.8 提升到 28.5。页面把这些变化归因于沿用 GLM-5.2 的 RL 策略（包括 “SAO with compaction”）以及环境/验证器管线；它没有在该段落给出完整模型 revision、推理档位、工具 schema、超时、重试、硬件或统计不确定性。因此这些数字只能记录为发布方自报的页面快照，不能当作独立复现、基础模型单独增益或跨 benchmark 的统一提升率。

同一页面还介绍私有 Z.ai Code Bench：在复杂本地开发环境中，按不同 effort 档位同时评估端到端任务完成率与细粒度 checklist 准确率，并称私有测试可降低公开测试集污染风险。页面未公开该基准完整任务集、样本数、harness、模型快照和原始分布；“更贴近真实用户体验”属于发布方定位，不等于外部验证结论。

网络安全能力部分涉及漏洞发现与利用的不同基准；后续仅据明确评测定义讨论能力边界与防御评估，不混淆发现漏洞、验证故障与完整利用链的指标。发布方自报结果尚未独立复现。

## 待办

1. 已找到 SAO 原始论文并核验其公开算法；GLM-5.3 专属的 compaction 状态表示、切分规则和训练实现仍待核验。
2. 模型卡、许可证名称和 config 已核验；独立 GLM-5.3 技术报告、完整权重加载和生产 profiling 仍待核验。
3. 2026-09-21 已从官方博客脚注补充 benchmark 的任务预算、harness、隔离、超时和 verifier 条件；独立复现仍待核验。
4. 验证器教学例子、正式章节及百科、题库、练习、术语、项目、论文路线和知识图谱已同步；后续只补新的一手实现或独立评测证据。

## 2026-09-20：榜单、官方文档、模型卡与训练代码复验

### 1. 两个排行榜的精确锚点

- Artificial Analysis 精确条目为 [GLM-5.3 (max)](https://artificialanalysis.ai/models/glm-5-3)，本轮三条代理响应逐字节一致：`3,928,329` bytes，SHA-256 `090279e870a3ebb72c7a69f24963f87fe10d63595642a4c61959b911f2a3c3a2`。页面字段为 release `2026-08-18`、Intelligence Index `44.777392385614`、约 `72.1152 tokens/s`、TTFT 约 `2.99s`（Z.ai API 测量）、1M context 和 `$1.40/$4.40` input/output；这些是 AA/provider 配置字段。
- DataCurve 当前快照三条代理逐字节一致：`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。精确对象为 `model=glm-5-3`、`harness=mini-swe-agent`、`reasoning_effort=max`、`config=mini_swe_agent_glm_5_3_max`：`n_runs=4`、`n_attempted=451`、Pass@1 `0.6895787139689579`、Pass@4 `0.8761061946902655`、`n_passed=311`、平均成本 `$3.9933584893126386`、平均输出 `80435.60975609756` token、平均输入 `13272344.266075388` token、平均缓存 `13106008.833702883` token、平均 `124.47228381374723` Agent steps、median `114`。
- DataCurve 数字绑定模型配置、`mini-swe-agent`、工具、任务集、环境和 verifier，不能与 AA Intelligence Index 拼成裸模型能力，也不能迁移 GLM-5.3-Flash 或 GLM-5.2 的行。

### 2. 官方产品/API 契约

[Z.ai GLM-5.3 文档](https://docs.z.ai/guides/llm/glm-5.3)与 [迁移指南](https://docs.z.ai/guides/overview/migrate-to-glm-new)确认：文本输入、1M context、128K 最大输出；thinking 强制 enabled；`reasoning_effort` 只有 `low/high/max`，默认 `max`；旧客户端发送 `thinking.type=disabled` 会失败，迁移前应改成 enabled 并显式选择预算。迁移指南还新增 `tool_stream=true`：与 `stream=true` 配合，把工具参数按 delta 逐步拼接，而不是等完整 JSON 一次返回。

Function Calling 文档把 `tools`、`tool_choice=auto`、`tool_calls`、函数名/JSON arguments 和 call `id` 分开；Chat Completion schema 还列出 function、retrieval、web search tool 类型与 `tool_stream`。Structured Output 是 JSON mode/结构约束，不等于业务正确性。Context Caching 使用隐式重复内容识别，并在 `usage.prompt_tokens_details.cached_tokens` 暴露命中 token；文档提醒格式变化和缓存过期会降低命中，缓存 token 按折扣价计费。

### 3. 后训练、环境和验证器：可迁移的系统知识

Z.ai 官方文档/博客称 GLM-5.3 复用 GLM-5.2 基座，全部提升来自 post-training。最有面试价值的不是“模型更大”，而是训练任务环境被做成可执行、可验证、接近专业工作的系统：研究 Agent 从真实工作模式合成长周期环境，环境含多步依赖和 hidden state；judge agent 先检查任务可解；verifier 在看不到 reference solution 的条件下生成；solver trajectory 用于发现 reward shortcut；通过 oracle、no-op、unsolved-state 检查后才把 binary reward 用于训练。

官方页面把训练场景举成 ML infrastructure 工作：访问 compute cluster、storage、内部文档、代码和实验结果，诊断训练瓶颈、改实现、运行实验并交付可测量的端到端加速，同时保持正确性。这是发布方对 environment scaling 的描述，不证明所有训练任务都使用同一 pipeline，也不公开 verifier 的完备性。

页面称 GLM-5.3 继承 GLM-5.2 的 `SAO with compaction`；[SAO 论文](https://arxiv.org/abs/2607.07508)的直接对象是单 rollout asynchronous RL，并明确讨论 off-policy、single-rollout sampling、value-model design 与 double-side token-level clipping，论文摘要称其用于 GLM-5.2（750B-A40B）。因此本项目可以把 SAO 的公开算法当作关联技术背景，但不能把论文结果改写为 GLM-5.3 独有的已证实训练实现。

### 4. 训练—rollout 一致性与 slime

官方 GLM-5.3 博客资源还说明，Z.ai 在 [slime](https://github.com/THUDM/slime) 上扩展 RL scaling：Megatron 负责训练、SGLang 负责 rollout，training、rollout 和 data buffer 处于同一 dataflow，数学、代码、sandbox、verifier 和 long-horizon environment 作为数据生成组件接入，而不是每增加一种环境就重写训练循环。

本轮博客明确列出 top-p mask、top-k/full-vocabulary OPD、R3-style 配置和 training/rollout full numerical alignment；官方自报 logprob 平均差控制到 `1e-7`，相对旧 setup 降低超过 `99.99%`。另称 workload-aware prefill/decode ratio、concurrency 等配置让长周期 coding RL 端到端训练 throughput 提升超过 `2.3x`。这些是 Z.ai 发布方的系统描述和测量，不能当作 GLM-5.3 权重内置的算法，也不能脱离硬件、batch、rollout 长度和环境分布复现。

关联论文 [IndexCache](https://arxiv.org/abs/2603.12201) 讨论 DSA lightning indexer 的跨层 top-k 复用：少数 Full layers 运行 indexer，多数 Shared layers 复用邻近 Full layer 的选择；论文在 30B DSA 实验中报告最多移除 75% indexer computation、最高 1.82x prefill/1.48x decode，并有 GLM-5 生产规模初步实验。它是稀疏注意力 serving 的关联证据，不是 GLM-5.3 官方架构变更声明。

### 5. 模型卡/config 与评测边界

官方 [Hugging Face 模型卡](https://huggingface.co/zai-org/GLM-5.3)与 [config.json](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json)本轮可访问。config 公开 `GlmMoeDsaForCausalLM`、`glm_moe_dsa`、78 层、前三层 dense、256 routed / top-8 / 1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、`index_topk_freq=4`、`index_share_for_mtp_iteration=true`、1,048,576 positions、`bfloat16` 和 `transformers_version=5.15.0`。其中 `indexer_types` 具体层模式是实现配置，不提供完整训练/参数账本；不能把 GLM-5.3 “同基座后训练”与 config 字段矛盾地写成新架构。

模型卡 benchmark 表的 Terminal Bench 3.0 `28.3`、DeepSWE `66.9`、CyberGym `84.5`、ExploitBench `54.4`、ExploitGym `105/130` 等，都需要连同官方脚注的 harness、temperature/top-p、max tokens、context、timeout、rollout 次数、domain whitelist 和 verifier 读取。比如 DeepSWE 使用 `mini-swe-agent`、temperature `0.95`、top-p `1.0`、6h timeout、400K context；Terminal Bench 3.0 使用 Claude Code harness、avg@3、600 turns/10h；CyberGym 是 Claude Code max、无 web、1507 tasks 的 single-run Pass@1。它们不能替换 DataCurve 的精确行，也不能与 AA 分数直接合并。

### 6. 当前闭环判断

GLM-5.3 已从“初步官方文档摘记”升级为**双榜资料级闭环（具有模型卡/config、官方博客训练线索、关联论文与代码）**：榜单精确对象、API 迁移、后训练环境/验证器、thinking/tool/cache、模型卡/config、IndexCache/SAO/slime 关联证据均已入库。仍待核验的内容包括：GLM-5.3 独立技术报告、完整 post-training recipe、5.3 专属 compaction 实现、production kernel、硬件 profiling、独立 benchmark 和线上 tool acceptance；不新增重复基础架构章节。

## 2026-09-21：官方博客正文资源与评测脚注补证

本轮通过官方博客前端壳和正文资源固定了可复核证据：

- `https://z.ai/blog/glm-5.3` 壳页面 598 bytes，SHA-256 `240cedb6d23b13b8bdd177e51410dbe1c7783fbd0cfca98be1e0af26688878c0`。
- [GLM-5.3 正文资源](https://z.ai/blog/assets/glm-5.3-BIDw01m9.js) 30,414 bytes，SHA-256 `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3`。
- [博客 source bundle](https://z.ai/blog/assets/src-GO5ZQO2t.js) 245,425 bytes，SHA-256 `e431f2c5e2590dd61672b259c32f49a81d6a0a0304e1311ea29b4b5a1e7b7d69`。

### 新增发布方能力披露

博客把 GLM-5.3 的叙事明确为“同一 GLM-5.2 基座上的 post-training scaling”，并称其在内部 Z.ai Code Bench 上相对 GLM-5.2 提升 50%。博客还给出 token efficiency 对照：Max effort 下 GLM-5.3 为 34.5% completion、约 75K output tokens/task，GLM-5.2 为 23.4%、约 96K；High effort 下 GLM-5.3 为 31.4%、约 50K，博客同时列出闭源基线作为参照。这里的 completion、checklist、effort 和 token 统计都属于发布方私有 benchmark 及其评测协议，不能写成独立复现或裸模型能力。

环境 scaling 的描述比前一轮更具体：部分任务模拟 ML infrastructure 工程师，提供 compute cluster、storage、内部文档、代码和实验结果；Agent 需要诊断训练瓶颈、修改实现、运行实验并交付可测量的端到端加速，同时保持正确性。研究 Agent 生成长周期、多步依赖和 hidden state 的 runnable environment，judge agent 先尝试求解；verifier 不读取 reference solution，solver trajectory 用于发现 reward shortcut；通过 oracle、no-op、unsolved-state 后才生成可直接训练的 binary reward。官方仍明确说环境生成和验证需要 meaningful human-in-the-loop。

### 网络安全能力的分层证据

博客把漏洞分析拆成不同阶段，不能把三个 benchmark 合成一个“利用能力”：

| 阶段 | 官方博客披露 | 面试解释 |
|---|---|---|
| 发现与验证 | CyberGym：从 white-box source code 出发，通过触发故障判断漏洞；GLM-5.3 `84.5%`，GLM-5.2 `77.2%` | 重点是定位和验证故障，不等于完整 exploit chain |
| 更深的利用推理 | ExploitBench：GLM-5.3 `54.4%`，GLM-5.2 `24.4%` | 评测对象更靠近真实漏洞利用推理，不能与 CyberGym 的分母和含义互换 |
| 时间归一化的利用任务 | ExploitGym：2 小时/6 小时分别完成 `105/130` 个任务；GLM-5.2 为 `29/39` | 预算按模型 TPS 归一化，必须保留时间、TPS、任务集和 harness |

博客还称，与中国多家安全团队合作，经专家复核、筛选和去重后，在 269 个项目中识别出 2,436 个漏洞，其中 1,097 个为中高危，最早缺陷约在 40 年前引入。该数字是发布方披露的真实代码库合作统计，不等于模型单独完成、全部漏洞已公开或完整利用链已验证；公开 ledger 的披露状态、CVE、严重度和代码存续时间也应与 benchmark 分开记录。

### 评测脚注与 harness manifest

官方正文资源补充了可复现条件，至少应把下面字段纳入 manifest：

| Benchmark | 关键条件 |
|---|---|
| NL2Repo | temperature `1.0`、top-p `1.0`、max new tokens `64K`、1M context；同时用规则与 LLM judge 防止未授权 `pip`/`curl` 等作弊 |
| DeepSWE | `mini-swe-agent`、temperature `0.95`、top-p `1.0`、400K context、6 小时 timeout |
| Terminal-Bench 3.0 | Claude Code 2.1.207 harness、max effort、400K context、128K max output、avg@3；官方 task image 的隔离 container、600 turns、10 小时 timeout、Tool Search disabled、官方独立 verifier |
| Agents' Last Exam | 官方 ALE protocol、Claude Code max、1M context、64K max output、105 个隔离 Docker task；默认 4 小时，任务卡可放宽到 8 小时，Tool Search disabled |
| CyberGym | Claude Code 2.1.207、max、无 web、temperature/top-p `1.0`、128K max new tokens、1,507 tasks、single-run Pass@1、task container、移除 Git 信息和域名白名单 |
| ExploitGym | Claude Code 2.1.207、max、无 web、temperature/top-p `1.0`、128K max new tokens、869 tasks、2 小时/6 小时单次 Pass@1；使用 Artificial Analysis TPS 做时间归一化，且有域名白名单 |
| ExploitBench | Claude Code 2.1.207、max、无 web、temperature/top-p `1.0`、128K max new tokens、最多 300 interaction rounds、41 tasks/3 revisions、域名白名单 |

这些条件说明“GLM-5.3 的官方 benchmark 分数”并不是一个单一对象；模型版本、harness、工具、环境、超时、隔离、域名策略、verifier 和统计聚合必须一起读取。它们也不能替换 DataCurve 的 `mini_swe_agent_glm_5_3_max` 行，更不能与 Artificial Analysis Intelligence Index 拼接。

### API 和开放权重声明的时间边界

博客正文再次确认 GLM-5.3 只支持开启 thinking，`reasoning_effort` 为 `low/high/max`；旧的 `thinking.type: "disabled"` 请求必须在切换模型 ID 前迁移。博客还写有“模型权重将在约两周后公开”的发布时承诺；这是带日期语境的 roadmap 文案，不等于本地已经下载、加载或完成目标硬件验收。本项目继续以 Hugging Face 模型卡/config 作为配置证据，不把页面承诺写成权重加载事实。

本轮新增内容适合补强既有长任务、验证器、网络安全评测和训练—rollout 一致性章节；不新建重复 Transformer 架构章节。当前状态仍为**双榜资料级闭环**，待核验项不变：独立 GLM-5.3 技术报告、完整 post-training recipe、5.3 上 SAO/compaction 的精确实现、production kernel、硬件 profiling、独立 benchmark 和线上 tool acceptance。

本地 [`sao_async_rl_toy.py`](code/sao_async_rl_toy.py) 已完成 SAO 公开机制的 protocol-level 教学验证，但只标为 `local_protocol_toy`：它不能提高 GLM-5.3 的模型、训练、权重、硬件或生产验收证据等级。

## 2026-09-22：标准 GLM-5.3 fixed revision 与 upstream runtime 对照

本轮继续只核验已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `GLM-5.3 (max)`；没有从官方仓库或框架源码另发现模型。该段专门区分标准 DSA 版与 `GLM-5.3-Flash` 的 `glm5_next`/linear-attention 路线。

### 1. 榜单与固定模型 artifact

- 2026-09-22 三条代理取得逐字节一致的 Artificial Analysis 中文首页 `1,762,420` bytes、SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`，以及 DataCurve DeepSWE `268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。重点厂商 canonical 模型集合没有新增。
- 标准版的精确对象仍是 AA `glm-5-3` 与 DataCurve `mini_swe_agent_glm_5_3_max`。DataCurve 的 311/451、Pass@1 `68.957871%`、Pass@4 `87.610619%`、平均成本约 `$3.9934` 和约 `124.47` Agent steps 绑定 `max`、`mini-swe-agent`、任务集、工具、环境和 verifier；不能迁移给 Flash，也不能与 AA 指数拼成裸模型能力。
- Hugging Face API 当前模型 revision 为 `aca966e4e02791568aa6a4ced368624b3d897f42`，`lastModified=2026-09-04T06:41:23Z`。固定 revision 的 `config.json` 为 `29,464` bytes、SHA-256 `3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69`，Git object `f4dd8fe8be2a6fee923d5ecc8de0a14892631b61`；README 为 `14,209` bytes、SHA-256 `ed1c0a4563c437a32f8953d637a1db9bd831de1b24bd3062a5ace9e04340b1cf`。
- config 公开 `model_type=glm_moe_dsa`、`GlmMoeDsaForCausalLM`、78 层、hidden size 6144、256 routed experts、top-8、1 shared expert、前 3 层 dense/其余 75 层 sparse、64 heads、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、`index_topk_freq=4`、`index_skip_topk_offset=3` 和 1,048,576 positions。`indexer_types` 是 21 个 `full` + 57 个 `shared`，即大多数层复用前一个 full layer 的 top-k；这些是 checkpoint/config 与实现合同，不等于真实 cache bytes、召回率或生产性能。

### 2. Transformers main：标准版的可读参考实现

- Transformers main branch ref 为 `0bc252863a4e5c0e709893664cc57369ebcd7353`。`configuration_glm_moe_dsa.py` 为 `7,918` bytes、blob `9759092416021956de13239d9c8e858d22a31b8d`、SHA-256 `1afa39ac6d0b6b374c973ef32c3c318d10c6c957916531e15a286eb5713acaae`；`modeling_glm_moe_dsa.py` 为 `37,868` bytes、blob `922ec9ee0e18cd6c3f8acca93f49465d35ae86f7`、SHA-256 `1494135a8b7416aa60dc7a9c194152cc5eedf80dff4a888076aa83eae146d155`。
- 参考实现明确把 `glm_moe_dsa` 作为 DSA 模型：indexer 对 query/key 使用 interleaved RoPE，full 层计算 top-k，shared 层通过 `prev_topk_indices` 复用上一个 full 层的选择，再构造 sparse attention mask；MoE 采用 top-k routed experts 并加 shared experts。这里的 `torch.topk`/eager 路径用于解释语义，不代表生产 CUDA kernel、吞吐或数值 acceptance。
- 这一实现足以支持一个重要面试区分：标准 GLM-5.3 的 top-k 复用来自 `glm_moe_dsa`/DSA 路径；Flash 的 `glm5_next.py` 使用 `RadixLinearAttention`、视觉模块和另一套 config，不能把 Flash 的 KDA/state-pool 结论迁移到标准版。

### 3. vLLM：v0.29.0 stable 与 main 的标准 DSA 入口

- vLLM main ref 为 `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`，v0.29.0 tag ref 为 `98dff2a81d747d1dba01a47f939f48c3526d4206`。两者的 registry 都把 `GlmMoeDsaForCausalLM` 路由到 `vllm.models.deepseek_v32`；main 另有独立的 `Glm5NextForCausalLM`，这是 Flash 路线的分离入口。
- 共用模型执行文件 `vllm/model_executor/models/deepseek_v2.py`：main 为 `78,620` bytes、blob `ed702400be6c21bb2ee63a20ff438eeebad6ab3e`、SHA-256 `53272b6643e7b8d2cf86fdf6bad44d157d89edb8cda5fe0e08364911c3c1f1c2`；v0.29.0 为 `77,732` bytes、blob `bc99509c7f45cb20739a215f5b32d7e7fb0af69a`、SHA-256 `8f34352a6a86da98a727c6c6c734dfb4d439fdb5f5f30e2a1472b54ca81f7a23`。两者都定义 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`，并按 `config.model_type == "glm_moe_dsa"` 进入 DSA/MLA 分支。
- `vllm/models/deepseek_v32/__init__.py` 两个版本均为 `1,463` bytes、SHA-256 `be2bdc7f98691848500c532bb25784badab16adeb16313e5c66800910f49b981`。该入口在 CUDA 上把 `GlmMoeDsaForCausalLM` 绑定到 DSA CUDA module；非 CUDA 平台回落到通用模型执行路径。
- `vllm/models/deepseek_v32/attention.py` main 为 `23,892` bytes、blob `b8e70e4ad3ef0b392b78b114de5f2069b3ad7947`、SHA-256 `9bbeeb3696c9848427bdd7c3573df15cdfa67931147ff673be12ed694c5cd326`；v0.29.0 为 `21,765` bytes、blob `6a0d207ff56895213ae41b8b4312ac3d4a2bc753`、SHA-256 `bba67788421a33a6d7db8cb7a44f91c90d07cd348fb36887026662c5523c424f`。两者都明确 indexer cache、`index_topk`、按 `index_topk_freq`/pattern 跳过或复用 top-k，并将稀疏 MLA 与 compressed latent KV 接到 attention backend。
- main 相对 v0.29.0 新增/强化了 PCP/DCP 相关 `index_group_builder`、`SparseCacheRole.INDEXER`、HiSparse cache 接口和 logical-top-k readiness 等 runtime 协作点。这是 mutable upstream 与 stable tag 的实现差异，不等于每个硬件后端都已经完成 profiling 或生产验收。

### 4. SGLang：通过 DeepSeek-V3.2 共用 DSA 路径

- SGLang `v0.5.20` tag 的 `deepseek_v2.py` 为 `130,545` bytes、blob `bae009ce8fb3f51cc00c635089624fe54ae65c0b`、SHA-256 `1578dd16dd71b78b95d4a5625f22a152fbdb6822ac6b38a4f24d106464c2831f`；main 为 `141,378` bytes、blob `c2356c907b37ff47fc6b228363685e5b3d2b88f3`、SHA-256 `9ffa2913974db31151c18e4931cebceca27c64c734be59a67540d4567e5d5a9f`。两者都检查 `is_glm_moe_dsa(config)`、暴露 `DeepseekV32ForCausalLM`，并维护 `IndexTopKShareState`/`prev_topk_indices` 这类跨层 top-k 状态。
- SGLang stable/main 的 source entry 证明标准版可以沿 DSA/MLA runtime 路线接入；它不证明当前 wheel 已加载完整 GLM-5.3 权重，也不证明目标 GPU 上的 index recall、端到端 token/s、MTP acceptance、tool-call 或 verifier 通过。
- SGLang `glm5_next.py` stable `v0.5.20` 为 Flash 专属 `Glm5NextForConditionalGeneration`，文件 `61,466` bytes、SHA-256 `12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2`；它不能作为标准 GLM-5.3 的 DSA runtime 证据。

### 5. 当前闭环与剩余门禁

标准 GLM-5.3 仍标为**双榜资料级闭环 + stable/main runtime source evidence**：排行榜锚点、官方文档/模型卡、研究笔记、既有长任务/验证器章节和配套资料均具备，标准 DSA/MLA 的 fixed config、Transformers、vLLM、SGLang 入口也已固定。仍未证明：完整权重下载与加载、stable wheel 在目标环境运行、真实 index/evidence recall、FP8/量化误差、MTP 接受率、目标硬件 profiling、PD/PP/DCP recovery、线上 tool acceptance、完整训练 recipe 和独立 benchmark。所有这些 gate 必须绑定模型 revision、框架 commit、硬件、batch、context、effort、harness、工具、环境和 verifier。

## 2026-09-23：compaction 合同审计与本机验收边界

本轮重新尝试通过 `10.237.126.170:1234`、`10.24.27.134:7890` 和 `10.24.27.134:8098` 获取 Z.ai GLM-5.3 文档/博客以及百度诊断页，三条线路都在连接阶段失败（curl HTTP `000`）。这只能说明当前时点代理不可达，不能说明官方页面或模型不存在。此前已保存的官方 Markdown（`23,446` bytes、SHA-256 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`）和博客正文资源（`30,414` bytes、SHA-256 `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3`）仍只公开“继承 SAO with compaction”，没有状态 schema、序列化格式、切分阈值、压缩质量门禁或 5.3 专属 post-training recipe。

本机门禁检查没有发现 GLM 权重文件；`torch`、`transformers`、`vllm`、`sglang` 和 `safetensors` 均不可导入，`nvidia-smi` 无法连接驱动。因此本轮不能把 fixed config、source entry 或官方文档升级成完整权重、kernel、数值、硬件 profiling 或生产 acceptance 证据。

新增 [`glm53_compaction_contract_audit.py`](code/glm53_compaction_contract_audit.py) 作为零依赖的 `local_protocol_toy`。它用合成状态检查确定性 JSON 序列化回环、任务目标/计划、工具 call-result lineage、permission scope、幂等键、待执行副作用、artifact digest、verifier 状态、预算单调性和切分标记；故意丢弃这些字段的候选会被拒绝。脚本已运行通过，但这只是“如何验收 compaction”的教学协议，不是 Z.ai 内部实现、真实 GLM-5.3 行为或 SAO 论文复现。

当前结论保持不变：**GLM-5.3 双榜资料级闭环 + SAO 关联论文算法证据 + stable/main runtime source evidence**；5.3 专属 compaction、完整 recipe、完整权重、目标硬件和线上 tool/verifier acceptance 仍为 `unverified`。网络恢复后先复抓两个排行榜和官方页面，再按 pinned revision 继续门禁；不从代码仓库或文档另发现模型。
