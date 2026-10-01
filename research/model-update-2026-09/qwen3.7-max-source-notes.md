# Qwen3.7 Max 官方发布材料与 Agent 方法摘记

核验日期：2026-09-28。本笔记以 Artificial Analysis / DataCurve DeepSWE 中已存在的 `Qwen3.7 Max` 为唯一模型锚点；Qwen 官方博客、阿里云 Model Studio 文档及论文检索只用于扩展和核验该锚点，不由博客中的模型名另建候选。

## 1. 榜单身份与版本边界

- Artificial Analysis [Qwen3.7 Max](https://artificialanalysis.ai/models/qwen3-7-max)，canonical slug `qwen3-7-max`。当前详情快照 `/tmp/aa-qwen3-7-max-7890-20260928.html`：`3,834,482` bytes，SHA-256 `829a322408dc044f12eefaa769eab0001904ceecb33d891abc801061fced7522`。页面列出的 Index 29、约 207 tokens/s、约 1M context、平均任务成本 `$1.15` 和 `$2.50/$7.50` 每百万 input/output token，均属于 AA/provider 展示字段，不是架构披露或跨区域统一价格合同。
- AA 页面 release date 为 2026-05-19；Model Studio 文档则称当前 alias `qwen3.7-max` 功能等价于 `qwen3.7-max-2026-05-20`。官方博客页面正文/JSON-LD 显示 2026-05-16，而文章 API 的 `extra.date` 为 2026-05-20。保留这些来源各自日期，不自行推断它们对应完全相同的服务 revision。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 快照 `/tmp/datacurve-7890-20260928-followup.html`：`268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。未见精确 `mini_swe_agent_qwen3_7_max_*` 行；不迁移其他 Qwen 的 Pass@1、成本、步骤或 token 数。
- 当前状态：**AA 单榜内容专题闭环**。有精确榜单锚点、官方发布材料、研究笔记、书系章节和配套训练；没有精确 DataCurve Agent 结果、独立复现或可归到榜单 revision 的明确映射。

## 2. 官方来源与日期化快照

### Qwen 官方发布博客

[《Qwen3.7：智能体新前沿》](https://qwen.ai/blog?id=qwen3.7)（Qwen Team；canonical 页面 `https://qwenlm.github.io/zh/blog/qwen3.7/`）通过官方文章 API 取得完整正文：`https://qwen.ai/api/v2/article/?language=zh-CN&path=qwen3.7&type=qwen_ai`，HTTP 200；本地 JSON `/tmp/qwen37-article-api-8098-20260928.json` 为 `122,153` bytes，SHA-256 `41e1f58384b99f1d2495111c3f6f55d01850283daa7ce5d07d38a764eea9108c`。响应含动态 `request_id`。正文介绍长程 Agent、环境扩展、跨框架 RL、kernel 优化、奖励作弊监控和若干发布方 benchmark。

页面的显示日期 / JSON-LD 是 2026-05-16，API 元数据另列 2026-05-20；来源自身存在日期差异。博文说“进一步分析将在即将发布的技术报告中”披露。后续检索已取得 arXiv v1 VHD-Play（评论标为 “Qwen Technical Report”），但没有明确证据把它等同于该博客承诺的 Max scaling 细节报告，见 §5。

### Alibaba Cloud Model Studio 模型文档

[Qwen3.7 Max 文档](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-max)，本地快照 `/tmp/alibaba-qwen3-7-max-7890-20260928.md` 为 `33,746` bytes，SHA-256 `a35e7b374f8499b4ef94c771bcfbdcb80b61ae55e6c5fb376924569d3bf38979`。文档列出的 alias / snapshot 边界如下：

| 文档身份 | 官方描述 | 可确认的边界 |
|---|---|---|
| `qwen3.7-max` alias | 功能等价于 `qwen3.7-max-2026-05-20`；纯文本接口 | AA 的同名条目没有公布服务 snapshot hash，不能仅凭名字断定具体部署 revision |
| `qwen3.7-max-2026-05-17` | 早期版本，仅支持 reasoning mode，纯文本 | Context Caching 不支持；不能与后续 alias 合并为同一 API 合同 |
| `qwen3.7-max-2026-06-08` | 相比 May 20 snapshot 增加视觉模态理解 | 输入 image/text/video，输出 text；这些视觉能力不能回写到 May 20 alias |

May 20 alias 的官方 context limits 为 `1,000,000` context window、`991,808` max input、`131,072` max output；thinking mode max input `983,616`、max output `131,072`、max chain-of-thought length `262,144`。它们是托管 API 合同字段，不代表可读的完整 CoT 或模型内部架构。

May 20 版本的 capability table 按 region/scope 列出 Function Calling、Structured Outputs、Web Search、Prefix Completion 与 Context Caching；Batch 主要限于北京，Fine-tuning 为 unsupported。Virginia 的 US scope 不支持 Web Search（同页 Global scope 支持）。不要把地域 capability table 归约成不分 endpoint 的“模型统一能力”。官方博客还称可在消息中通过 `preserve_thinking` 保留前序轮次 reasoning 内容并建议用于 Agent；这是客户端 transcript 序列化/API 行为，不是跨会话自动记忆，历史未重放就不能假定状态仍在。

## 3. 面试相关的新方法与技术主张

### 3.1 Agent 环境多样性作为 scaling 维度

Qwen 称 Qwen3.7 在 Qwen3.5 的 environment scaling 基础上提升训练环境的质量与多样性；测试基准对应训练中未见的 OOD 环境。发布方观察到不同 benchmark 子集上的增益相对一致，并称可由部分子集预测其余子集或整体平均增益。文章把进一步的 scaling dynamics / methodology 留给后续技术报告；没有给出足以独立复现的环境数量、环境构成、训练配比、统计不确定性和逐点曲线数据。因此面试时可把“环境多样性是 Agent RL 的数据/训练 scaling 轴”作为发布方方法主张，不能说其已建立普适 scaling law。

### 3.2 Task / Harness / Verifier 正交解耦与跨配置 RL

官方描述的 rollout 基础设施把一个训练实例拆为三个可自由重组的组件：

```text
Task × Harness × Verifier
```

同一任务可以组合不同类型/版本的执行框架和验证器；Qwen 称训练中用跨 harness、跨 verifier 的 RL，使模型在不同配置下处理同源任务，目标是学到可迁移解题策略、降低利用某个框架捷径的可能。它把“任务是否完成”“运行环境如何交互”“何谓完成”从一个固化 benchmark package 变成可组合实验因子，是一个很好的 Agent RL / evaluation-system 设计点。

博客进一步称在 QwenClawBench、内部 CoWorkBench 上跨框架评估表现稳定，但这仍是发布方结果；环境是否真正 OOD、不同 verifier 是否难度等价、训练/测试组合是否存在泄漏，文章未充分展开。实际评测应保存三者的 revision/hash，留出未见过的 `Task × Harness × Verifier` 组合，并检查跨组合的置信区间和失败类型。

### 3.3 在陌生硬件上持续优化 GPU kernel

官方案例针对 SGLang 的变长 Extend Attention：在最长 32K 前缀 KV-cache 上，对 MTP 新生成 token 计算注意力。Qwen3.7 Max 被放到此前训练未见的平头哥真武 M890 PPU 上；博文称初始信息只有任务、SGLang Triton 参考实现和评估脚本，没有该架构的 profile、硬件文档或 kernel 示例。约 35 小时里完成 432 次 kernel evaluation 和 1,158 次工具调用，多个 workload 相对 Triton 参考的几何平均加速比为 10.0x（均为发布方报告）。

| 迭代阶段 | 发布方描述的优化 | 报告的阶段性加速 |
|---|---|---:|
| Split-KV | 沿 token 维拆分前缀 KV-cache，提高低并行度 workload 的 SM 利用率；用 online-softmax 重缩放合并分块结果 | `0.33x → 2.58x` |
| 消除分配/同步成本 | 预分配输出 tensor，利用 tensor metadata，减少 host-device sync；展开内层循环 | `2.58x → 5.37x` |
| workload-adaptive split | 由固定分割数改为按输入/批次规模调节 split，适配 36-SM 芯片 | `5.37x → 6.85x` |
| reduction / batching | 减少 shared-memory barrier，改用寄存器加载，复用持久化结果，批量更新 softmax | `6.85x → 8.50x` |
| MTP `γ=4` 特化 | 每个 block 处理 4 个 query token，共享 K/V 读取并批量归约；优化寄存器压力和 split | `8.50x → 10.0x` |

这里可提炼的知识不是“模型知道 M890 内部结构”，而是“模型通过编译、正确性检查、profile、运行反馈和重构来探索未知硬件”。10.0x 的设备、baseline、任务与 workload 必须一同报告；不能外推成通用 kernel 性能或 Qwen3.7 的模型架构。

不要混淆另一个结果：发布博客的 KernelBench L3 表列 Qwen3.7 Max `1.98x / 96%`，定义为 50 个问题上相对 PyTorch eager 的 per-problem speedup 中位数 / 快于 `torch.compile` 的问题比例；它使用隔离 Docker、H100 80GB、受限网络、至多 500 次工具调用、100 轮无改进后停止，并用 GPT-5.4 (xhigh) 标记潜在 hacking。它既不是 M890 上的 10.0x，也不是同一个 baseline、任务集或设备。

### 3.4 RL 轨迹上的 reward-hacking 自监控与规则演化

Qwen 称将模型接入 SWE RL 训练监控：在超过 80 小时的实验中，模型回放训练轨迹、汇总疑似作弊模式（例：访问 GitHub 标准答案），再验证候选检测规则、挖掘反例并多轮修订；发布方报告新增 13 条启发式规则、识别 1,618 个作弊案例、累计超过万次调用。

面试可将其概括为“模型辅助的 reward-hacking auditor + 可迭代规则集”，而不是“模型已自行改写 reward function”或“用 RL 自动证明无作弊”。可落地的审计闭环是 `trajectory → candidate pattern → replay → counterexample mining → versioned rule → held-out audit`。来源没有披露 1,618 个案例的标注方式、规则 precision/recall、误报漏报、复核者、规则冻结策略或相对于旧监控器的对照，因此数字只应归为发布方报告，不能推出 detector 准确率或训练收益因果关系。

### 3.5 长时序模拟与具身 demo：测量的是系统闭环

博客还称 Qwen3.7 Max 在动态累积生存博弈 / YC-Bench 中处理跨模拟一年、数百轮决策的经营任务，并报告累计完成任务数与模拟营收。这里的“营收”是模拟环境的 evaluator 结果，不是真实商业业绩；文章没有充分披露 simulator/verifier 细节，不能外推真实经营能力。物理世界 demo 则由 Qwen3.7-Max 发起工具调用、Qwen-RobotNav 导航基础模型、Qwen-RobotClaw 具身 Agent 系统和基于 Qwen-Plus 的视觉工具共同组成，演示约 20 分钟交互。这是 model + tool + embodied harness 的系统组合，并非 Qwen3.7 Max 内部含有导航/控制模块的证据；这些支撑组件只作为来源中的系统依赖，不另列为本项目的模型发现锚点。

## 4. 官方 benchmark 的口径

博文中的分数是 Qwen 发布方结果。至少需保留这些条件：

- Terminal-Bench 2.0-Terminus：Harbor/Terminus-2 harness、5 小时 timeout、12 CPU / 24 GB RAM、256K context、80K max output、5 次平均；每轮前置 token 允许模型自行决定是否启用 extended thinking。
- SWE-Bench 系列：内部 bash + file-edit scaffold、200K context；SWE-bench Pro 修订了部分问题并在修订集上重跑所有基线。
- SkillsBench：通过 OpenCode 跑 78 个自包含任务，排除 9 个依赖外部 API 的任务，5 次平均。
- KernelBench L3：定义、硬件和 tool-call 上限见 §3.3；中位加速和“快于 torch.compile 的比例”是不同统计量。
- QwenWebDev、CoWorkBench、QwenWorldBench 等被标为内部 benchmark；其分数不是公开集上的独立复现。QwenClawBench、MCP-Mark/Atlas、VITA 等还各自绑定工具版本、截断和 judge。

分数不能脱离 task set、harness、工具、资源、采样参数、重复次数和 verifier。AA 的第三方 `Index 29`、Qwen 自报 benchmark、DataCurve 的 `mini-swe-agent` 系统统计是三套不同测量，不拼成裸模型排名。

## 5. 证据边界与下一步

可以确认榜单身份、官方博客公开的 rollout/environment 方法主张、特定 kernel 案例与其 baseline、reward-monitor 过程描述、官方 dated snapshot/API 限制。尚不能确认参数规模、dense/MoE、层数、attention、完整训练数据或 RL recipe、公开权重、跨硬件 kernel 成绩、reward detector 的精确率/召回率、未披露 benchmark 的可复现性或线上生产 SLO。

后续检索已找到 2026-09-23 的 [arXiv:2609.27321v1《Verifiable Hidden Dynamics Play》](https://arxiv.org/abs/2609.27321v1)，其 arXiv 评论标为 “Qwen Technical Report”，主题是先求解数学机制、再生成带可验证 reference 的 stateful Agent RL 环境。论文使用 Qwen3.6-35B-A3B 训练；Qwen3.7-Max 仅作额外 setter / benchmark 对照。它与本博客的 environment-scaling 主题相关，但没有明确声明 VHD-Play 就是本博客的 `Task × Harness × Verifier` 机制或 Qwen3.7-Max 的内部训练 recipe，故不做该归属推断。方法、数字和限制见 [Qwen3.6-35B-A3B 研究笔记 §9](qwen3.6-35b-a3b-source-notes.md#9-vhd-play先求解机制再生成可交互环境)。

代理状态按具体 URL 和时点记录：7890 对 arXiv 摘要页曾返回 HTTP 200（43,529 bytes）；随后同代理请求完整 /html 页面报 curl error 7。完整 HTML 已从备用 1234 路径取得并分析；不完整 PDF 未使用。此前 Qwen 博客文章 API 经 8098 成功获取。不能据单次 host/path 失败概括为 7890 全面不可用。

| 快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/aa-qwen3-7-max-7890-20260928.html` | Artificial Analysis Qwen3.7 Max | 3,834,482 | `829a322408dc044f12eefaa769eab0001904ceecb33d891abc801061fced7522` |
| `/tmp/datacurve-7890-20260928-followup.html` | DataCurve DeepSWE | 268,036 | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` |
| `/tmp/alibaba-qwen3-7-max-7890-20260928.md` | Alibaba Cloud Model Studio Qwen3.7 Max | 33,746 | `a35e7b374f8499b4ef94c771bcfbdcb80b61ae55e6c5fb376924569d3bf38979` |
| `/tmp/qwen37-article-api-8098-20260928.json` | Qwen official article API response (contains dynamic request ID) | 122,153 | `41e1f58384b99f1d2495111c3f6f55d01850283daa7ce5d07d38a764eea9108c` |
