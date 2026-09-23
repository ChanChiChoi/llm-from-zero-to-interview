# GLM-5.2 官方文档核验摘记

核验日期：2026-09-17（初始文档核验为 2026-09-09）。来源：[Z.ai GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2)。

## 已确认内容

官方将 GLM-5.2 定位为面向长任务的旗舰基础模型，页面标称文本输入/输出、1M 上下文和 128K 最大输出，并列出函数调用、MCP、上下文缓存和结构化输出能力。文档用项目级代码库接管、跨文件重构、生产标准压力测试、移动设备调试、论文复现等场景说明长任务定位。

页面称为让百万 token 上下文“可用”，模型接受了数月面向长程 Coding Agent 的专项训练，并声称使用“lossless context”。这些属于发布方叙述，尚未有独立复现；正文不应把“lossless”解释为数学意义上的绝对无损。

## 与 GLM-5.3 的关系边界

GLM-5.3 官方文档称沿用 GLM-5.2 基础模型并通过后训练改进。GLM-5.2 页面本身没有展开 SAO 或 compaction 的数学定义，但 [SAO 论文](https://arxiv.org/abs/2607.07508) 已提供公开的 Single-Rollout Asynchronous Optimization 定义，并明确说该路线部署到 GLM-5.2（750B-A40B）的 Agent RL pipeline。论文不是 GLM-5.3 专属报告，也没有公开产品侧 compaction 的完整实现，所以仍需把“SAO 算法”与“5.3 继承关系/具体状态压缩实现”分账。

GLM-5.2 页面展示的 /goal 模式、CLAUDE.md/Agent.md 约束、ADB/logcat、研究复现等提示词属于产品使用示例，不等同于模型内部训练算法。正式章节应将它们作为 Agent harness 与工作流案例讲解。

## 后续核验

## 2026-09-16 断点恢复与榜单复验

本轮按夜间 20:00—次日 09:00 可能中断的规则重新抓取两个排行榜。Artificial Analysis [`glm-5-2`](https://artificialanalysis.ai/models/glm-5-2) 和 DataCurve DeepSWE 均返回 HTTP 200；AA 快照 `/tmp/glm52-aa-20260916.out` 为 `3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；DataCurve 快照 `/tmp/glm52-ds-20260916.out` 为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。

Artificial Analysis 将其记录为 `GLM-5.2 (max)`，第三方 FAQ 约为 Intelligence Index `34`、1M context、约 `72 tokens/s` 和 `$1.40/$4.40` 每百万 input/output token；这些仍是第三方页面的配置、provider 和测量字段。DataCurve 页面记录：

- `mini_swe_agent_glm_5_2_high`：164/452，Pass@1 `36.2832%`，Pass@4 `68.1416%`，平均成本约 `$2.8355`，平均输出 `54,245.50` token，平均 Agent steps `121.88`；
- `mini_swe_agent_glm_5_2_max`：197/450，Pass@1 `43.7778%`，Pass@4 `76.9912%`，平均成本约 `$3.9199`，平均输出 `78,175.31` token，平均 Agent steps `129.13`。

这两行使用 `mini-swe-agent`、4 runs、113 tasks、91 repositories、5 languages，并受工具、环境和 verifier 影响；不能与 AA Intelligence Index 拼成裸模型排名。

## 2026-09-16 官方资料补充

新鲜获取的 [Z.ai GLM-5.2 官方文档](https://docs.z.ai/guides/llm/glm-5.2) 的 `dateModified` 为 `2026-09-03T14:49:41.429Z`，页面快照 `/tmp/glm52-zai-doc-8098-20260916.out` 为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`。官方页面明确：

- GLM-5.2 面向 long-horizon tasks，支持 1M context、128K maximum output、thinking mode、function calling、context caching、structured output 和 MCP；
- 训练与产品叙述强调 project-scale codebase、跨文件重构、需求到部署的连续工作流，并声称经过数月 Coding Agent 场景专项训练；“lossless context”按发布方产品描述记录，不解释成数学意义上的绝对无损；
- 页面把 /goal 模式、CLAUDE.md/Agent.md 约束、ADB/logcat、研究复现和分阶段的 requirements—implementation—verification 流程作为使用案例；这些属于提示词/Agent workflow 证据，不等同于内部训练算法。

本轮仍没有找到 GLM-5.2 独立完整技术报告、模型卡架构细节或产品 compaction 的完整实现；但已找到 SAO 原始论文并核验其全称、token-level importance sampling、single-rollout、value-model 设计和实验边界。GLM-5.3 文档声称继承 `SAO with compaction`，这只支持关联技术归因，不支持把论文实验或未公开的 compaction 细节写成 GLM-5.3 独有事实。

## 当前结论

## 2026-09-17 官方博客正文补证

此前抓取的 `https://z.ai/blog/glm-5-2` 返回 404；正确路由是 [Z.ai GLM-5.2 博客](https://z.ai/blog/glm-5.2)。页面通过 JavaScript 加载正文，本轮三条代理均返回 HTTP 200。正文资源为 `https://z.ai/blog/assets/glm-5.2-UFbrCk0E.js`，大小 `46,166` bytes，三条代理 SHA-256 一致：`c29e51551a1fb0100c681267da8330fa7a29df4e4160382b1b38cd8e7dc594db`。

博客新增了此前文档页没有披露的技术线索：

- `IndexShare`：每四层共享一个轻量 DSA indexer，由第一层计算 top-k indices 并供四层使用，目标是减少 indexer dot product 与 top-k 操作；博客称从 128K mid-training 开始引入，并引用 [arXiv:2603.12201](https://arxiv.org/abs/2603.12201)。
- MTP speculative decoding：博客描述降低 MTP draft 成本、提高 acceptance length，并通过共享 index/top-k、参数共享、rejection sampling 与 end-to-end TV loss 改善训练—推理一致性；发布方在其消融中称最终 MTP acceptance length 提高 20%。该数字仍是发布方自报，不能当作普遍收益。
- 长上下文 serving：博客把瓶颈拆为 KV-cache 容量、随上下文增长的 kernel、CPU cache 管理和调度；公开描述了更细粒度 cache 管理、prefill/decode 与 cache transfer 协同、CPU 调度优化和 PD 组织，但没有给出可独立复现的 kernel 或硬件 profiling。
- Agentic RL：博客称 `slime` 支持 white-box/black-box rollout、compact trajectory 和 sub-agent workflow，并用于并行 OPD；同时声称合并十多个 expert models 的 OPD 过程约两天。这是发布方训练基础设施描述，不足以确认完整 optimizer、数据配方或并行一致性协议。
- 长轨迹优化：针对 compaction 后产生数量和长度不稳定的 sub-traces，博客描述从 group-wise optimization 转向 critic-based PPO，以 token-level advantage 处理长度差异，并把 compacted sub-traces 纳入训练。
- Coding-agent anti-hack：博客描述规则过滤器加模型分类器的两阶段检测，先识别读取受保护评测文件、下载参考解或绕过任务的 shortcut，再对违规动作返回负反馈并继续轨迹，而不是直接丢弃整条 rollout。该机制说明 reward-hacking 防护方向，不等于公开了完整 detector、阈值或训练配方。

这些内容足以新增第二十一册第 86 章，但不会改变 GLM-5.2 的“资料级闭环”状态：博客仍未公开完整参数架构、训练 recipe、生产 kernel、硬件 profiling 或独立复现。

GLM-5.2 已从“部分覆盖”升级为“资料级闭环”：两个排行榜的精确配置、Z.ai 官方模型文档、官方博客、SAO 原始论文、长上下文/Agent 工作流证据和研究笔记均已具备；正式专题见第二十一册第 86 章。面试主线是 IndexShare 的共享 indexer、MTP speculative decoding、1M context 的 serving 账本、SAO 的 single-rollout/DIS/critic/Skip-Observation GAE、compaction 后的 critic-based PPO、anti-hack verifier、工具/MCP 集成、context caching，以及榜单配置与 Agent 系统结果的分层。参数量、内部架构、5.3 专属 compaction 状态、完整训练/后训练 recipe、生产 kernel、硬件 profiling 和独立 benchmark 仍待核验；SAO 公开论文算法本身不替代这些门禁。

## 2026-09-23：SAO 原始论文补证

- [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)（arXiv:2607.07508，Zhenyu Hou、Yujiang Li、Jie Tang、Yuxiao Dong）摘要明确：SAO 用每个 prompt 一条 rollout 替代 GRPO 的 group-wise sampling，使用 rollout engine 的 log-probability 做 token-level importance sampling，并采用严格的双侧 token clipping；论文摘要称该路线部署到 GLM-5.2（750B-A40B）Agent RL pipeline。
- 本地 HTML 正文 `/tmp/sao-html-20260923-1234.out`：`190,454` bytes，SHA-256 `953b8968fa30d5f579a9650cc17d95521515ccac6cc7408b2a45c8126651822d`；PDF `/tmp/sao-pdf-7890-20260923.out`：`664,828` bytes，SHA-256 `44c695be0428c666d06c914ba76c037e3ac77eeb5db0a81bbe239719c21bda48`。正文以 arXiv HTML 为主要可读证据，PDF 因 object streams 不被现有抽取器完整解析。
- 论文公开的关键设计包括：用 rollout engine 的 log-probability 直接计算当前策略与 rollout 策略的比值，不依赖历史 old-policy checkpoint ensemble；把比例限制在 `[1-epsilon_low, 1+epsilon_high]`，越界 token 从梯度中 mask；rollout 完成后立即训练，消除 group barrier 和长尾 straggler；critic 每次 policy update 做 `K=2` 次更新，并冻结 value model 的 attention、只优化 MoE projections。
- 对 compaction 后的 observation token，论文给出 Skip-Observation token-level GAE：跨过环境 observation，从 action token 的 value bootstrap 到下一段 action；这解决的是 credit assignment 的 token mask 问题，不等于公开了 Z.ai 产品的 compaction 序列化格式。
- 论文实验主干使用 Qwen3-30B-A3B；设置包含 batch `128`、single-rollout/group size `1`、max length `128K`、policy learning rate `1e-6` 和 value learning rate `5e-6`。这些是论文实验条件，不能写成 GLM-5.3 或 DataCurve 的运行条件。

### 归因边界

可以写：GLM-5.3 官方文档明确说继承 GLM-5.2 的 `SAO with compaction`；SAO 论文解释了公开的单 rollout 异步 RL、DIS、critic 和 Skip-Observation GAE 路线。不能写：SAO 是 GLM-5.3 独有发明、论文结果就是 GLM-5.3 结果，或论文已经公开 5.3 的 compaction 实现。当前 GLM-5.2 状态为**双榜资料级闭环 + SAO 原始论文算法证据**。

### 本地 SAO protocol toy

- 新增 [`sao_async_rl_toy.py`](code/sao_async_rl_toy.py)，只使用 Python 标准库和合成 rollout/value/reward/log-probability，不联网、不调用付费 API、不加载 GLM 权重。
- 运行结果：`group_barrier_wait_total=11`；single-rollout 的合成等待为 `0`；ratio `0.1` 和 `5.0` 被 DIS toy mask，`1.0` 和 `1.1` 保留；1 个 observation token 与 3 个 observation token 的 Skip-Observation GAE 均为 `a0=1.0603`，普通 token GAE 的首段结果则从约 `1.04276` 变为 `0.99734`；critic proxy 的 `K=2` 最终平方误差 `0.0625`，低于 `K=1` 的 `0.25`。
- 证据等级为 **local protocol toy**。这些数值只说明教学状态机和公式的内部一致性，不是 SAO 论文 benchmark、GLM-5.2/5.3 训练结果、产品 compaction、完整权重、目标硬件或生产 verifier 证据。
