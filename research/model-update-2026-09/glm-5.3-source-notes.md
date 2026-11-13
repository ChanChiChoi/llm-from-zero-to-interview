# GLM-5.3 来源记录：官方接口、长任务环境与评测边界

本文件记录可核验的公开事实、来源版本和证据范围，不记录书籍审阅进度或正文修改历史。

## 2026-11-13 可访问资料复核

### 官方模型文档与迁移合同

- [Z.ai GLM-5.3 文档](https://docs.z.ai/guides/llm/glm-5.3)，动态在线文档，核验日期 2026-11-13。阅读 Overview、Feature Changes、Model API、Key Advancements 与 Stronger Coding。页面说明 GLM-5.3 沿用 GLM-5.2 基础模型、改进来自 post-training；输入为文本，接口标称 1M context、128K 最大输出，`thinking.type` 只支持 `enabled`，`reasoning_effort` 为 `low/high/max`，默认 `max`。发布方描述 research agent 生成可运行长任务环境、judge agent 检查可解性、reference-free verifier、solver trajectory 查找 reward shortcut，以及 oracle/no-op/unsolved-state 检查；同时说明人仍参与环境生成和验证。页面称 GLM-5.3 继承 GLM-5.2 的 SAO with compaction，但未披露 5.3 专属状态 schema、序列化、触发阈值、压缩保真度或完整训练配方。
- [Migrate to GLM-5.3](https://docs.z.ai/guides/overview/migrate-to-glm-new)，动态在线迁移指南，核验日期 2026-11-13。阅读迁移 checklist、reasoning 参数和 streaming tool call 示例。工具参数流需要同时启用 `stream=true` 与 `tool_stream=true`；示例按 `tool_call.index` 拼接 `delta.tool_calls[*].function.arguments`。文档展示 `name`、`call_id` 字段，但未要求它们像 arguments 一样逐片拼接。GLM Coding Plan 订阅端点的可用协议可能少于通用模型文档列出的协议，应按具体产品/API 计划区分。
- 以上是发布方接口与流程说明；不能单独验证其 verifier 的假阳性率、所有环境的可解性、长任务成功率或内部训练实现。

### 公开模型仓库与报告标记

- [Hugging Face `zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) 与 [固定 config revision](https://huggingface.co/zai-org/GLM-5.3/blob/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json)。2026-11-13 查询模型 API：仓库为公开、非 gated；revision SHA 为 `ba32f8c19c8738e2c57c71798c601d477a7bf844`，`lastModified=2026-10-14T12:22:48Z`。模型卡和配置现可用于检查公开权重与架构配置字段；这不等于完整训练配方、托管模型服务实现、目标硬件 kernel 或真实 serving 性能。
- 仓库标签列出 [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)。该技术报告讨论 GLM-5 路线，并非 GLM-5.3 专属完整报告；不得把论文实验或模型卡标签误写成 5.3 专有训练记录。

### 当前榜单页面与时间口径

- [Artificial Analysis GLM-5.3 release 页面](https://artificialanalysis.ai/models/releases/glm-5-3)，2026-11-13 访问。页面显示 GLM-5.3 Max 的 Intelligence Index 为 32，口径为 Index v5.0.1、包含 9 项评测；还显示约 72 tokens/s、63.55 秒 time to first answer token、`$1.40/$4.40` 输入/输出价格和 `$5.01` 每 Index task 加权成本。这些是第三方评测与 provider 测量，绑定页面时间、模型配置和指标定义。
- [Artificial Analysis Index v5 变更说明](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5/)说明 v5 调整/替换评测集合。故 2026-09-20 留存的旧版 GLM-5.3 页面值 `44.777392385614`（详情快照 3,928,329 bytes，SHA-256 `090279e870a3ebb72c7a69f24963f87fe10d63595642a4c61959b911f2a3c3a2`）与当前 `32` 不能解释成模型能力回退或拼成一条纵向趋势。
- [DeepSWE / DataCurve](https://deepswe.datacurve.ai/)，2026-11-13 访问。页面选择 v1.1，说明 113 tasks、更新时间 2026-09-22；`glm-5.3[max]` 汇总显示 `69%±3%`、平均 `$3.99`、约 80K output tokens、124 agent steps，并说明所有模型使用 `mini-swe-agent`。页面介绍任务覆盖 91 个仓库、5 种语言，使用手写行为 verifier。
- 2026-09 留存的精确配置 `mini_swe_agent_glm_5_3_max` 明细是 4 runs、451 attempted、311 passed、Pass@1 `68.9579%`、Pass@4 `87.6106%`、平均成本 `$3.9934`、平均输出 80.4K tokens、平均 124.47 steps；页面当前首页只显示四舍五入汇总，没有展示 Pass@4 和 attempted 细节。精确数值是 `mini-swe-agent + model revision/effort + tools + task repositories + environment + timeout + verifier` 的系统结果，不是裸模型准确率。

### SAO、IndexCache 与 slime 固定代码

- [SAO 论文](https://arxiv.org/abs/2607.07508)的公开算法讨论 single-rollout asynchronous agentic RL、off-policy correction、value-model 设计、双侧 token clipping 与 Skip-Observation GAE；公开实验主干使用 Qwen3-30B-A3B，摘要另称部署到 GLM-5.2（750B-A40B）Agent RL pipeline。它支持算法背景和 GLM-5.2 部署声明，不证明 GLM-5.3 专属实现。
- [IndexCache](https://arxiv.org/abs/2603.12201)讨论 DSA lightning indexer 的跨层 top-k 复用，是关联 serving 技术证据，不证明 GLM-5.3 实际生产 kernel 或索引召回表现。
- [`THUDM/slime` pinned revision `8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b`](https://github.com/THUDM/slime/tree/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b)中的 Agentic RL roadmap、customization、coding-agent example、`TrajectoryManager` 和 branching tests 支持 token provenance、message tree、prefix alignment、loss mask、样本分叉与固定 revision 行为的讨论。`fork_threshold_tokens=1,024` 只属于该实现快照，不是 GLM API 或当前 main 的固定保证。
- 2026-09-29 公开提交 [`8088a4b`](https://github.com/THUDM/slime/commit/8088a4b)增加 Straw checkpoint rollback 与 indexed rollout archives；[`6857224`](https://github.com/THUDM/slime/commit/6857224)支持 distributed fully async rollout。它们是通用训练系统提交，晚于 GLM-5.3 对外发布资料，不能倒推为模型内部使用的实现。上游集成测试描述其测试模型、GPU 数和覆盖范围，只能作为源码测试设计，不是本地执行结果或 GLM benchmark。

## 2026-09-28 当前榜单复验与 slime 的 compaction-aware trajectory

固定源码阅读重点是重建不同训练对象的边界：

| 对象 | 固定实现/来源给出的信息 | 适用边界 |
| --- | --- | --- |
| rollout 采样 token | trajectory snapshot 保留 `prompt_ids`、`output_ids` 和每 token log-prob；解码文本不用于重建原始训练目标 | 保持采样 token 与 log-prob provenance；不是 GLM-5.3 服务端接口 |
| user/tool/environment/template 文本 | 可进入上下文但在相应分支标为 `loss_mask=0` | 仅限 pinned slime adapter/trajectory 行为，不能推广为所有 RL 系统 |
| 已生成 assistant 输出 | 仅在 token 来源可确认时作为 loss-bearing output | 上下文被重渲染后需基于 token prefix 判别，不能仅凭可读文本认为 token 相同 |
| 分支共享前缀 | routing tree 保留不同 history；共享 assistant 前缀只在一个训练路径计入，其余作为 loss-zero context | 固定分支实现的样本构造策略 |
| reward 与 segment | roadmap/customization 有按 K 分配 reward 的说明；该 revision 中轨迹组装和测试路径呈现不同处理 | 不把文档惯例推成代码实际 reducer 或 GLM 训练规则；须按训练/advantage 归约路径核实 |

固定源码：[`trajectory.py`](https://github.com/THUDM/slime/blob/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b/slime/agent/trajectory.py)、[trajectory branching tests](https://github.com/THUDM/slime/blob/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b/tests/test_agent/test_trajectory_manager_branching.py)、[Agentic RL roadmap](https://github.com/THUDM/slime/blob/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b/docs/en/get_started/agent.md)、[customization guide](https://github.com/THUDM/slime/blob/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b/docs/en/get_started/customization.md) 和 [coding-agent example](https://github.com/THUDM/slime/blob/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b/examples/coding_agent_rl/README.md)。

证据状态：source-backed（官方文档、模型仓库、评测页面和固定源码）；local-demo（仓库内的 verifier/compaction toy 与 SAO mechanism toy，仅验证其自身合成输入）。仍未验证 GLM-5.3 专属 compaction schema、完整 post-training recipe、托管服务内部、生产硬件性能和独立模型复现。

## 本地教学脚本观察

- [`glm53_compaction_contract_audit.py`](code/glm53_compaction_contract_audit.py)：2026-11-13 以 CPython 3.12.9 和标准库运行。构造 tool-call 前、call/result 之间、result 完成后三个状态；三者通过 schema/lineage 审计。删除未完成 call 的 pending effect 会被拒绝，孤立 pending effect 也会被拒绝。输出标明 `generic_protocol_toy_not_glm53_implementation`。
- [`sao_async_rl_toy.py`](code/sao_async_rl_toy.py)：同日以 CPython 3.12.9 和标准库运行。合成 rollout latency `[1,3,2]` 的 group barrier wait 为 3 秒，完成顺序为 `r1,r3,r2`；四个 importance ratio 为 `[1.0202,0.7408,1.2840,1.0000]`，clip mask 为 `[true,false,false,true]`；action-only GAE 为 `[0.5920,0.6000]`，合成 observation 列表项从 1 增至 100 不改变该结果；标量 critic 两次更新的平方误差为 `[1.0000,0.6400]`。这些结果只验证合成机制 toy，不复现 SAO 论文或 GLM-5.3。
