# 第 6 章 大模型任务画像：预训练、SFT、RLHF、评估、推理和 Agent

前面几章讲了 AI Infra 的总览、概念边界、AI 加速器、显存带宽和训练效率指标。本章要补一个非常重要的视角：不同大模型任务，对基础设施的要求完全不同。

预训练、SFT、RLHF、评估、推理、RAG、Agent 看起来都在“用大模型”，但它们的资源画像、瓶颈、调度方式、监控指标和故障模式差异很大。

先记住一句话：

> AI Infra 不是为“模型”服务，而是为不同 AI 工作负载服务；不同工作负载的计算、显存、网络、存储、延迟和可靠性要求完全不同。

## 6.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径按“稳定工作负载画像”处理，而不是按某个厂商平台、某个云产品 SKU 或某个调度器字段来写死。预训练部分参考 Megatron-LM 等大规模语言模型训练公开资料中对数据并行、张量并行、流水并行、GPU 集群和通信瓶颈的描述；SFT / RLHF 部分参考 InstructGPT 一类对齐流程中 SFT、reward model、policy 优化和 rollout 的多阶段结构；评估部分参考 OpenAI Evals 这类把模型或系统行为做成可复现实验的开源评估框架；推理部分参考 vLLM / PagedAttention 一类推理系统中 KV cache、continuous batching 和请求级延迟的稳定工程抽象。

需要注意三点：

1. 本章不把某个训练框架、推理框架、Agent runtime 或向量数据库写成唯一标准，只抽象它们共同暴露出的资源、调度、观测和治理问题。
2. RAG 和 Agent 的任务画像依赖业务链路，不能只按 LLM 推理成本估算；检索、工具、权限、trace、人工接管和失败重试都可能成为主瓶颈。
3. 任务画像的目标不是给每类任务贴标签，而是把“该配什么资源、用什么队列、看什么指标、怎么算成本、失败后怎么恢复”变成可审计的输入。

## 6.1 什么是任务画像

任务画像就是描述一个任务对基础设施的需求特征。

可以从这些维度看：

1. 任务目标：训练、微调、评估、推理还是 Agent 执行。
2. 运行时长：分钟级、小时级、天级还是周级。
3. 资源规模：单卡、多卡、单机多卡、多机多卡。
4. 计算强度：矩阵计算多不多。
5. 显存压力：参数、activation、KV cache 占用如何。
6. 网络压力：是否需要大量跨卡通信。
7. 存储压力：是否频繁读数据、写 checkpoint。
8. 延迟要求：离线任务还是在线服务。
9. 容错要求：失败能不能重试，是否允许中断。
10. 成本敏感度：是否需要极致压低单位成本。

理解任务画像后，你才能设计合理的平台和调度策略。

## 6.2 预训练任务画像

预训练是从大量通用数据中训练基础模型。

它的特点是：

1. 数据规模极大。
2. 训练时间长。
3. GPU 数量多。
4. 分布式训练复杂。
5. checkpoint 体积大。
6. 成本高。
7. 容错非常重要。

### 6.2.1 资源特征

预训练通常需要：

1. 大规模 GPU 集群。
2. 高速互联网络。
3. 高吞吐数据读取。
4. 高可靠 checkpoint 存储。
5. 稳定的调度和运行环境。

预训练任务可能连续运行几天、几周甚至更久。一次节点故障、网络抖动或存储异常，都可能造成巨大成本浪费。

### 6.2.2 主要瓶颈

预训练常见瓶颈：

1. GPU 利用率低。
2. 多机通信慢。
3. 数据加载跟不上。
4. checkpoint 保存慢。
5. 节点故障导致训练中断。
6. loss 异常或训练发散。
7. 集群排队时间长。

### 6.2.3 平台诉求

预训练平台需要重点支持：

1. 大规模分布式启动。
2. gang scheduling。
3. 拓扑感知调度。
4. checkpoint 自动保存和恢复。
5. 节点故障检测。
6. 训练指标和日志聚合。
7. 数据版本和实验追踪。
8. 成本统计。

预训练是 AI Infra 能力的压力测试。

## 6.3 SFT 任务画像

SFT 是 Supervised Fine-Tuning，也就是监督微调。

它通常在基础模型上，用指令数据、对话数据或任务数据做微调。

相比预训练，SFT 通常：

1. 数据规模更小。
2. 训练时间更短。
3. 资源规模更小。
4. 实验迭代更频繁。
5. 更关注数据质量和效果评估。

### 6.3.1 资源特征

SFT 可能使用：

1. 单机多卡。
2. 少量多机。
3. LoRA / QLoRA 等参数高效微调。
4. 多实验并行。

它不像预训练那样一定追求超大规模，但更强调快速迭代。

### 6.3.2 主要瓶颈

SFT 常见瓶颈：

1. 小实验太多，调度排队影响效率。
2. 数据版本混乱。
3. prompt / template 改动不可追踪。
4. 显存不够导致 batch 很小。
5. 多个实验结果难比较。
6. 评估集污染或不稳定。

### 6.3.3 平台诉求

SFT 平台更需要：

1. 快速任务提交。
2. 实验追踪。
3. 数据集版本管理。
4. 参数和配置管理。
5. 模型 artifact 管理。
6. 自动评估。
7. LoRA / adapter 管理。
8. 低优先级资源池。

SFT 的核心不是单次训练最大规模，而是高质量、高频率、可复现的迭代。

## 6.4 RLHF / RLAIF 任务画像

RLHF 是 Reinforcement Learning from Human Feedback，RLAIF 是 Reinforcement Learning from AI Feedback。

它们用于偏好对齐和行为优化。

这类任务比 SFT 更复杂，因为它通常涉及多个模型和多个阶段。

例如 RLHF 可能包含：

1. SFT 模型。
2. Reward Model。
3. Policy Model。
4. Reference Model。
5. 采样生成。
6. 偏好数据。
7. PPO、DPO 或其他优化算法。

### 6.4.1 资源特征

RLHF / RLAIF 的资源画像比较特殊：

1. 同时有训练和推理。
2. 需要大量 rollout 或样本生成。
3. 需要 reward 评分。
4. 可能需要多个模型同时加载。
5. pipeline 更复杂。
6. 对实验追踪要求高。

和预训练相比，它不一定 GPU 总量最大，但编排复杂度更高。

### 6.4.2 主要瓶颈

常见瓶颈：

1. rollout 生成慢。
2. reward model 评分慢。
3. 多模型显存占用高。
4. pipeline 中某个阶段成为瓶颈。
5. 偏好数据质量不稳定。
6. 训练过程指标难解释。
7. 实验不可复现。

### 6.4.3 平台诉求

RLHF / RLAIF 平台需要：

1. 多阶段 pipeline 编排。
2. 训练和推理资源联合调度。
3. 数据、模型、reward、policy 版本追踪。
4. rollout 样本管理。
5. 人工或 AI 反馈数据管理。
6. 安全和偏好评估。
7. 复杂实验追踪。

这类任务是 MLOps、LLMOps 和 AI Infra 的交叉区域。

## 6.5 评估任务画像

评估任务用于判断模型是否真的变好。

评估可以分为：

1. 离线自动评估。
2. 人工评估。
3. LLM-as-judge。
4. 安全评估。
5. 回归评估。
6. 在线 A/B 测试。

### 6.5.1 资源特征

评估任务通常：

1. 推理请求多。
2. 样本集多。
3. 模型版本多。
4. 并发可控。
5. 对可复现性要求高。
6. 对结果存储和分析要求高。

评估不一定需要训练级大规模 GPU，但会消耗大量推理资源。

### 6.5.2 主要瓶颈

评估常见瓶颈：

1. 推理成本高。
2. 多模型评测排队。
3. benchmark 版本混乱。
4. 评测结果不可复现。
5. judge 模型不稳定。
6. 评估样本污染。
7. 人工评估周期长。

### 6.5.3 平台诉求

评估平台需要：

1. benchmark 管理。
2. eval job 调度。
3. 多模型批量推理。
4. 评估结果存储。
5. 指标对比和可视化。
6. 样本级错误分析。
7. 安全评估和回归测试。
8. 成本预算控制。

评估任务的关键不是“跑完”，而是“可比较、可复现、可解释”。

## 6.6 推理任务画像

推理是把模型服务给用户或业务系统。

它和训练最大的区别是：推理通常在线服务，有明确延迟和可用性要求。

### 6.6.1 资源特征

推理任务关注：

1. TTFT。
2. TPOT。
3. QPS。
4. 并发。
5. KV cache。
6. 显存带宽。
7. batching。
8. 自动扩缩容。
9. 单位 token 成本。

推理还有明显流量波动。白天和夜间、工作日和节假日、活动期间和普通期间，负载可能差异很大。

### 6.6.2 主要瓶颈

推理常见瓶颈：

1. 队列等待时间长。
2. TTFT 高。
3. decode 慢。
4. KV cache 爆显存。
5. batch 调度不合理。
6. 热点模型资源不足。
7. 长上下文请求拖慢短请求。
8. 扩容太慢。
9. 成本过高。

### 6.6.3 平台诉求

推理平台需要：

1. 模型服务 runtime。
2. continuous batching。
3. KV cache 管理。
4. 模型路由。
5. 限流、熔断和降级。
6. 自动扩缩容。
7. 灰度发布和回滚。
8. 请求级 trace。
9. 成本统计。
10. SLO 监控。

推理任务的核心是质量、延迟、吞吐、可用性和成本之间的平衡。

## 6.7 RAG 任务画像

RAG 是 Retrieval-Augmented Generation，检索增强生成。

RAG 不只是模型推理，还包含知识库、embedding、向量检索、rerank 和上下文拼接。

### 6.7.1 资源特征

RAG 涉及：

1. 文档解析。
2. 文本切分。
3. embedding 生成。
4. 向量索引构建。
5. 在线检索。
6. rerank。
7. LLM 生成。
8. 引用和溯源。

因此 RAG 的瓶颈可能不在 LLM，而在检索链路。

### 6.7.2 主要瓶颈

RAG 常见瓶颈：

1. 文档更新延迟。
2. embedding 生成慢。
3. 向量索引构建慢。
4. 检索召回差。
5. rerank 延迟高。
6. 上下文过长。
7. 引用来源不准确。
8. 权限过滤复杂。

### 6.7.3 平台诉求

RAG 平台需要：

1. 知识库管理。
2. 文档解析 pipeline。
3. embedding 服务。
4. 向量索引服务。
5. 检索质量评估。
6. 权限过滤。
7. 引用追踪。
8. RAG trace。
9. 数据更新监控。

RAG 是 LLMOps 和 Data Platform 深度交叉的典型场景。

## 6.8 Agent 任务画像

Agent 任务比普通推理更复杂，因为它可能包含多轮思考、工具调用、记忆读写、RAG、代码执行、浏览器操作和多 Agent 协作。

### 6.8.1 资源特征

Agent 任务通常：

1. 请求链路长。
2. 工具调用多。
3. 延迟不稳定。
4. token 消耗难预测。
5. 需要 trace。
6. 需要权限控制。
7. 失败模式复杂。
8. 可能需要人工接管。

普通推理是“一次模型调用”，Agent 可能是“一串模型调用和工具调用”。

### 6.8.2 主要瓶颈

Agent 常见瓶颈：

1. 工具调用延迟高。
2. 多轮模型调用成本高。
3. 上下文越来越长。
4. 任务循环。
5. 工具失败后不会恢复。
6. 权限检查复杂。
7. trace 数据量大。
8. 结果不可复现。

### 6.8.3 平台诉求

Agent 平台需要：

1. Agent Runtime。
2. Tool / MCP Gateway。
3. Memory Store。
4. Trace Store。
5. 权限和审计。
6. 步数和成本预算。
7. 任务状态机。
8. 人工接管。
9. Agent eval。

Agent 工作负载对 AI Infra 的要求不仅是算力，还包括状态、工具、权限、trace 和治理。

## 6.9 多模态任务画像

多模态任务包括图像、音频、视频、文本的联合训练和推理。

它的特点是数据更重、预处理更复杂、存储和带宽压力更大。

常见场景：

1. 图文理解。
2. OCR。
3. 图像生成。
4. 视频理解。
5. 视频生成。
6. 语音识别。
7. 语音合成。

多模态任务的基础设施压力：

1. 原始数据体积大。
2. 解码和预处理重。
3. 存储吞吐要求高。
4. GPU 显存压力大。
5. batch 组装复杂。
6. 评估更难自动化。

例如视频任务可能不是 GPU 算不动，而是视频解码、帧采样和数据读取拖慢训练。

## 6.10 不同任务的资源画像对比

可以用下面的表总结：

| 任务 | 计算压力 | 显存压力 | 网络压力 | 存储压力 | 延迟要求 | 关键指标 |
| --- | --- | --- | --- | --- | --- | --- |
| 预训练 | 很高 | 很高 | 很高 | 很高 | 低 | tokens/s、MFU、loss、checkpoint |
| SFT | 中到高 | 中到高 | 中 | 中 | 低 | 迭代速度、eval、可复现性 |
| RLHF/RLAIF | 高 | 高 | 中到高 | 中 | 中 | rollout 吞吐、reward 质量、稳定性 |
| 评估 | 中 | 中 | 低到中 | 中 | 低到中 | 评测吞吐、成本、可复现性 |
| 推理 | 中到高 | 高 | 中 | 低到中 | 很高 | TTFT、TPOT、QPS、成本 |
| RAG | 中 | 中 | 低到中 | 高 | 高 | 检索延迟、召回、引用准确性 |
| Agent | 不稳定 | 不稳定 | 中 | 中 | 中到高 | 成功率、步数、成本、trace |
| 多模态 | 高 | 高 | 中 | 很高 | 视场景而定 | 数据吞吐、预处理、质量指标 |

这张表不是绝对标准，但能帮助你在面试中快速建立分析框架。

## 6.11 调度策略如何随任务变化

不同任务应该使用不同调度策略。

预训练：

1. 需要大规模连续资源。
2. 适合 gang scheduling。
3. 需要高优先级和稳定资源。
4. 需要拓扑感知。

SFT：

1. 实验多、规模小到中等。
2. 适合队列化和配额管理。
3. 可以使用低优先级或可抢占资源。
4. 需要快速启动。

评估：

1. 可以批量排队。
2. 可以使用低峰资源。
3. 需要成本预算。
4. 需要结果可复现。

推理：

1. 需要在线 SLO。
2. 需要弹性扩缩容。
3. 不能随意抢占。
4. 需要隔离高优先级流量。

Agent：

1. 需要控制最大步数。
2. 需要工具调用预算。
3. 需要任务状态和超时。
4. 高风险动作需要人工确认。

调度系统如果不区分任务画像，很容易出现资源错配。

## 6.12 监控指标如何随任务变化

预训练要重点看：

1. loss。
2. tokens/s。
3. MFU。
4. step time。
5. 通信占比。
6. checkpoint 状态。
7. 节点健康。

SFT 要重点看：

1. eval score。
2. loss。
3. 训练配置。
4. 数据版本。
5. 实验对比。

评估要重点看：

1. benchmark 版本。
2. 模型版本。
3. 样本级结果。
4. 评测成本。
5. judge 一致性。

推理要重点看：

1. QPS。
2. TTFT。
3. TPOT。
4. p95 / p99 延迟。
5. 错误率。
6. 队列长度。
7. GPU 利用率。
8. KV cache 使用率。

Agent 要重点看：

1. 任务成功率。
2. 平均步数。
3. 工具调用次数。
4. token 成本。
5. 失败原因。
6. trace 完整性。
7. 人工接管率。

指标设计必须服务于任务画像。

## 6.13 成本模型如何随任务变化

预训练成本主要来自：

1. GPU 小时。
2. 网络和存储。
3. 失败重跑。
4. checkpoint 存储。

SFT 成本主要来自：

1. 多实验并行。
2. 低利用率小任务。
3. 重复训练。
4. 评估调用。

评估成本主要来自：

1. 批量推理。
2. judge 模型调用。
3. 人工评估。
4. 结果存储。

推理成本主要来自：

1. GPU 常驻。
2. token 生成。
3. KV cache 显存。
4. 峰值容量冗余。
5. 多模型部署。

Agent 成本主要来自：

1. 多轮模型调用。
2. 工具调用。
3. RAG 检索。
4. 长上下文。
5. 失败重试和循环。

成本治理必须按任务拆分，否则很难找到真正的浪费来源。

## 6.14 面试中如何回答任务画像问题

如果面试官问：

```text
预训练、SFT、推理和 Agent 对基础设施的要求有什么不同？
```

可以这样回答：

```text
我会从任务画像分析，包括计算、显存、网络、存储、延迟、容错和成本。

预训练是长周期、大规模、多机多卡任务，重点是 GPU 吞吐、MFU、网络通信、数据供给、checkpoint 和容错。

SFT 规模通常小一些，但实验频繁，重点是快速提交、数据版本、实验追踪、模型 artifact 和自动评估。

推理是在线服务，重点是 TTFT、TPOT、QPS、KV cache、batching、自动扩缩容、限流降级和单位 token 成本。

Agent 是多轮模型调用加工具调用，重点不只是算力，还包括任务状态、工具权限、trace、步数预算、成本控制和失败恢复。

所以 AI Infra 不能用一种调度和监控方式服务所有任务，需要按任务画像设计资源池、调度策略、监控指标和成本模型。
```

## 6.15 大模型任务画像审计指标与最小 demo

前面的章节讲的是定性画像。真实平台设计里，还要把任务画像转成可检查的数据结构，否则很容易停留在“预训练很重、推理要低延迟、Agent 很复杂”这种口号层面。

可以把第 `i` 个工作负载样本写成：

```math
a_i=(w_i,s_i,r_i,c_i,m_i,n_i,d_i,l_i,o_i,g_i,z_i)
```

其中，`w_i` 是 workload type，`s_i` 是阶段，`r_i` 是运行时长或生命周期，`c_i` 是计算需求，`m_i` 是显存和内存需求，`n_i` 是网络需求，`d_i` 是数据和存储需求，`l_i` 是延迟或 SLO，`o_i` 是观测信号，`g_i` 是成本和治理，`z_i` 是风险。

更工程化一点，可以把资源画像写成向量：

```math
\mathbf{v}_i=(C_i,M_i,N_i,D_i,L_i,R_i,K_i,S_i)
```

其中，`C_i` 是 compute，`M_i` 是 memory，`N_i` 是 network，`D_i` 是 data / storage，`L_i` 是 latency，`R_i` 是 reliability，`K_i` 是 cost，`S_i` 是 security / governance。

离线批任务的总耗时可以粗略拆成：

```math
T_i=T_{\mathrm{queue},i}+T_{\mathrm{data},i}+T_{\mathrm{compute},i}+T_{\mathrm{comm},i}+T_{\mathrm{io},i}+T_{\mathrm{eval},i}
```

这个式子提醒你：预训练、SFT、评估这类任务不是只看 GPU compute。排队、数据读取、通信、checkpoint I/O 和自动评估都会进入端到端时间。

在线推理或 Agent 链路可以用 SLO 通过率表达：

```math
R_{\mathrm{slo}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[TTFT_i\le \tau_{\mathrm{ttft}}\land TPOT_i\le \tau_{\mathrm{tpot}}\land E_i=0]
```

其中，`TTFT_i` 是第 `i` 个请求的首 token 延迟，`TPOT_i` 是每输出 token 延迟，`E_i` 是错误标记。只看平均延迟通常不够，最好同时看 p95 / p99、错误率、队列等待和请求 trace。

成本也要按任务拆开：

```math
K_i=K_{\mathrm{gpu},i}+K_{\mathrm{storage},i}+K_{\mathrm{network},i}+K_{\mathrm{human},i}+K_{\mathrm{api},i}
```

预训练的 `K_gpu` 可能最大；评估和 RLHF 可能有 `K_human` 或 judge / reward 调用；Agent 和 RAG 可能因为工具调用、检索、长上下文和重试让 `K_api` 上升。

最后，可以把任务画像门禁写成：

```math
G_{\mathrm{task}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{slo}}\ge \tau_{\mathrm{slo}} \land P_0=0\right]
```

其中，`C_j` 是第 `j` 个审计指标覆盖率，`tau_j` 是最低覆盖率阈值，`P_0` 是 P0 级风险数量。这个门禁不是让所有任务都满足同一套资源指标，而是要求每类任务必须有匹配自己的资源、调度、监控、成本和风险说明。

下面这个 0 依赖 demo 演示如何把任务画像写成可审计规则。它故意构造 1 个完整样本和 16 个坏样本，让每个审计维度各失败一次。

```python
import copy


METRICS = [
    "workload_type_classification",
    "resource_shape_completeness",
    "pretraining_profile_accuracy",
    "sft_iteration_profile_accuracy",
    "rlhf_pipeline_profile_accuracy",
    "evaluation_reproducibility_profile",
    "serving_slo_profile",
    "rag_freshness_retrieval_profile",
    "agent_tool_runtime_profile",
    "multimodal_resource_profile",
    "scheduler_policy_fit",
    "observability_metric_fit",
    "cost_model_fit",
    "artifact_lineage_fit",
    "safety_governance_fit",
    "task_profile_gate",
]


def slo_pass_rate(requests, ttft_limit_ms, tpot_limit_ms):
    passed = 0
    for req in requests:
        ok = (
            req["ttft_ms"] <= ttft_limit_ms
            and req["tpot_ms"] <= tpot_limit_ms
            and req["errors"] == 0
        )
        passed += int(ok)
    return passed / len(requests)


def batch_job_time_s(parts):
    return sum(parts.values())


def request_cost_usd(prompt_tokens, output_tokens, input_per_m, output_per_m, tool_cost):
    return (
        prompt_tokens * input_per_m / 1_000_000
        + output_tokens * output_per_m / 1_000_000
        + tool_cost
    )


def build_profiles():
    complete = {
        "name": "complete",
        "workload_type": "mixed_llm_platform",
        "stages": ["pretraining", "sft", "rlhf", "eval", "serving", "rag", "agent"],
        "resources": {"C": 8, "M": 8, "N": 7, "D": 8, "L": 7, "R": 8, "K": 7, "S": 8},
        "pretraining": {
            "tokens_per_second": 100_000,
            "mfu": 0.45,
            "checkpoint_minutes": 30,
            "data_version": "pretrain-v42",
            "topology_aware": True,
        },
        "sft": {
            "dataset_version": "sft-v7",
            "template_version": "chat-template-v4",
            "eval_suite": "sft-regression-v3",
            "adapter_artifacts": True,
        },
        "rlhf": {
            "sft_model": "sft-ckpt-7",
            "reward_model": "rm-3",
            "policy_model": "policy-9",
            "reference_model": "ref-9",
            "rollout_samples": 50_000,
            "preference_data_version": "pref-v5",
            "safety_eval": True,
        },
        "evaluation": {
            "benchmark_version": "eval-v12",
            "model_version": "model-v8",
            "sample_outputs_stored": True,
            "judge_seed": 123,
            "replayable": True,
        },
        "serving": {
            "requests": [
                {"ttft_ms": 420, "tpot_ms": 31, "errors": 0},
                {"ttft_ms": 610, "tpot_ms": 42, "errors": 0},
                {"ttft_ms": 780, "tpot_ms": 55, "errors": 0},
                {"ttft_ms": 930, "tpot_ms": 62, "errors": 0},
            ],
            "ttft_limit_ms": 1_000,
            "tpot_limit_ms": 80,
            "min_slo_rate": 0.95,
            "kv_cache_tracked": True,
            "autoscaling": True,
        },
        "rag": {
            "index_version": "kb-v12",
            "embedding_model": "embed-v3",
            "document_lag_minutes": 12,
            "citation_trace": True,
            "permission_filter": True,
        },
        "agent": {
            "trace_id": "trace-001",
            "max_steps": 12,
            "tool_budget_usd": 0.05,
            "state_machine": True,
            "human_escalation": True,
            "tool_spans": 5,
        },
        "multimodal": {
            "modalities": ["text", "image", "audio"],
            "max_payload_mb": 64,
            "preprocess_queue": True,
            "quality_eval": True,
        },
        "scheduler": {
            "policy_by_workload": {
                "pretraining": "gang",
                "sft": "quota",
                "eval": "batch",
                "serving": "slo",
                "agent": "deadline",
            },
            "quota": True,
        },
        "observability": {
            "metrics": ["tokens/s", "mfu", "ttft", "tpot", "tool_latency", "cost"],
            "logs": True,
            "traces": True,
            "events": True,
        },
        "cost": {"gpu": 1500, "storage": 80, "network": 35, "human": 200, "api": 120, "budget": 2200},
        "lineage": {
            "dataset": "dataset-v7",
            "code": "git-sha",
            "image": "train-image-v4",
            "model": "model-v8",
            "eval_report": "eval-report-v12",
        },
        "governance": {
            "owner": "ai-platform",
            "risk_review": True,
            "pii_policy": True,
            "access_control": True,
        },
        "gate": {"enabled": True},
    }

    def bad_case(name, mutator):
        profile = copy.deepcopy(complete)
        profile["name"] = name
        mutator(profile)
        return profile

    bad_cases = [
        bad_case("generic_ai_job_bad", lambda p: p.update({"workload_type": "generic"})),
        bad_case("resource_vector_incomplete_bad", lambda p: p["resources"].pop("S")),
        bad_case("pretraining_no_checkpoint_bad", lambda p: p["pretraining"].update({"checkpoint_minutes": None})),
        bad_case("sft_no_dataset_version_bad", lambda p: p["sft"].update({"dataset_version": ""})),
        bad_case("rlhf_rollout_ignored_bad", lambda p: p["rlhf"].update({"rollout_samples": 0})),
        bad_case("eval_not_reproducible_bad", lambda p: p["evaluation"].update({"replayable": False})),
        bad_case(
            "serving_no_ttft_tpot_bad",
            lambda p: p["serving"].update({"requests": [{"ttft_ms": 1400, "tpot_ms": 120, "errors": 0}]}),
        ),
        bad_case("rag_freshness_ignored_bad", lambda p: p["rag"].update({"document_lag_minutes": 720})),
        bad_case("agent_tool_trace_missing_bad", lambda p: p["agent"].update({"trace_id": ""})),
        bad_case("multimodal_payload_unbounded_bad", lambda p: p["multimodal"].update({"max_payload_mb": 2048})),
        bad_case("scheduler_one_queue_bad", lambda p: p["scheduler"].update({"policy_by_workload": {"default": "fifo"}})),
        bad_case("observability_generic_gpu_only_bad", lambda p: p["observability"].update({"metrics": ["gpu_utilization"]})),
        bad_case("cost_token_only_bad", lambda p: p["cost"].pop("human")),
        bad_case("artifact_lineage_missing_bad", lambda p: p["lineage"].update({"eval_report": ""})),
        bad_case("safety_governance_missing_bad", lambda p: p["governance"].update({"risk_review": False})),
        bad_case("task_profile_gate_missing_bad", lambda p: p["gate"].update({"enabled": False})),
    ]
    return [complete] + bad_cases


def check_workload_type(profile):
    allowed = {"pretraining", "sft", "rlhf", "eval", "serving", "rag", "agent", "multimodal", "mixed_llm_platform"}
    return profile.get("workload_type") in allowed


def check_resource_shape(profile):
    return all(k in profile["resources"] for k in ["C", "M", "N", "D", "L", "R", "K", "S"])


def check_pretraining(profile):
    p = profile["pretraining"]
    return bool(p["tokens_per_second"] > 0 and p["mfu"] > 0 and p["checkpoint_minutes"] and p["topology_aware"])


def check_sft(profile):
    p = profile["sft"]
    return bool(p["dataset_version"] and p["template_version"] and p["eval_suite"] and p["adapter_artifacts"])


def check_rlhf(profile):
    p = profile["rlhf"]
    required = ["sft_model", "reward_model", "policy_model", "reference_model", "preference_data_version"]
    return all(p[k] for k in required) and p["rollout_samples"] > 0 and p["safety_eval"]


def check_evaluation(profile):
    p = profile["evaluation"]
    return bool(p["benchmark_version"] and p["model_version"] and p["sample_outputs_stored"] and p["replayable"])


def check_serving(profile):
    p = profile["serving"]
    rate = slo_pass_rate(p["requests"], p["ttft_limit_ms"], p["tpot_limit_ms"])
    return rate >= p["min_slo_rate"] and p["kv_cache_tracked"] and p["autoscaling"]


def check_rag(profile):
    p = profile["rag"]
    return bool(p["document_lag_minutes"] <= 60 and p["index_version"] and p["citation_trace"] and p["permission_filter"])


def check_agent(profile):
    p = profile["agent"]
    return bool(p["trace_id"] and p["max_steps"] > 0 and p["tool_budget_usd"] > 0 and p["state_machine"] and p["tool_spans"] > 0)


def check_multimodal(profile):
    p = profile["multimodal"]
    return bool(p["modalities"] and p["max_payload_mb"] <= 128 and p["preprocess_queue"] and p["quality_eval"])


def check_scheduler(profile):
    policies = profile["scheduler"]["policy_by_workload"]
    return all(k in policies for k in ["pretraining", "sft", "eval", "serving", "agent"]) and profile["scheduler"]["quota"]


def check_observability(profile):
    o = profile["observability"]
    needed = {"tokens/s", "mfu", "ttft", "tpot", "tool_latency", "cost"}
    return needed.issubset(set(o["metrics"])) and o["logs"] and o["traces"] and o["events"]


def check_cost(profile):
    c = profile["cost"]
    needed = ["gpu", "storage", "network", "human", "api", "budget"]
    return all(k in c for k in needed) and sum(c[k] for k in needed[:-1]) <= c["budget"]


def check_lineage(profile):
    l = profile["lineage"]
    return all(l[k] for k in ["dataset", "code", "image", "model", "eval_report"])


def check_governance(profile):
    g = profile["governance"]
    return bool(g["owner"] and g["risk_review"] and g["pii_policy"] and g["access_control"])


def check_gate(profile):
    return profile["gate"]["enabled"]


CHECKS = {
    "workload_type_classification": check_workload_type,
    "resource_shape_completeness": check_resource_shape,
    "pretraining_profile_accuracy": check_pretraining,
    "sft_iteration_profile_accuracy": check_sft,
    "rlhf_pipeline_profile_accuracy": check_rlhf,
    "evaluation_reproducibility_profile": check_evaluation,
    "serving_slo_profile": check_serving,
    "rag_freshness_retrieval_profile": check_rag,
    "agent_tool_runtime_profile": check_agent,
    "multimodal_resource_profile": check_multimodal,
    "scheduler_policy_fit": check_scheduler,
    "observability_metric_fit": check_observability,
    "cost_model_fit": check_cost,
    "artifact_lineage_fit": check_lineage,
    "safety_governance_fit": check_governance,
    "task_profile_gate": check_gate,
}


def audit_task_profile(profiles):
    case_failures = {}
    for profile in profiles:
        failures = [name for name, check in CHECKS.items() if not check(profile)]
        case_failures[profile["name"]] = failures

    metrics = {}
    for name, check in CHECKS.items():
        metrics[name] = round(sum(int(check(p)) for p in profiles) / len(profiles), 3)

    failed_cases = [name for name, failures in case_failures.items() if failures]
    return {
        "metrics": metrics,
        "hard_blocker_count": len(failed_cases),
        "failed_cases": failed_cases,
        "task_profile_gate_pass": not failed_cases and min(metrics.values()) >= 0.95,
    }


profiles = build_profiles()
profile_by_name = {profile["name"]: profile for profile in profiles}
audit = audit_task_profile(profiles)

task_profile_examples = {
    "pretraining_time_h": round(batch_job_time_s({
        "queue": 240,
        "data": 900,
        "compute": 3600,
        "comm": 600,
        "io": 180,
        "eval": 120,
    }) / 3600, 2),
    "sft_time_h": round(batch_job_time_s({
        "queue": 60,
        "data": 180,
        "compute": 900,
        "comm": 60,
        "io": 60,
        "eval": 90,
    }) / 3600, 2),
    "serving_slo_pass_rate": round(slo_pass_rate([
        {"ttft_ms": 500, "tpot_ms": 40, "errors": 0},
        {"ttft_ms": 700, "tpot_ms": 60, "errors": 0},
        {"ttft_ms": 1200, "tpot_ms": 70, "errors": 0},
        {"ttft_ms": 800, "tpot_ms": 95, "errors": 0},
    ], 1000, 80), 2),
    "agent_cost_usd": round(request_cost_usd(12_000, 4_000, 1.5, 6.0, 0.02), 3),
    "complete_cost_usd": sum(profile_by_name["complete"]["cost"][k] for k in ["gpu", "storage", "network", "human", "api"]),
}

smoke = {
    "complete_case_passes": all(check(profile_by_name["complete"]) for check in CHECKS.values()),
    "caught_pretraining_checkpoint_gap": not check_pretraining(profile_by_name["pretraining_no_checkpoint_bad"]),
    "caught_rlhf_rollout_gap": not check_rlhf(profile_by_name["rlhf_rollout_ignored_bad"]),
    "caught_serving_slo_gap": not check_serving(profile_by_name["serving_no_ttft_tpot_bad"]),
    "caught_agent_trace_gap": not check_agent(profile_by_name["agent_tool_trace_missing_bad"]),
}

print(f"task_profile_examples={task_profile_examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"task_profile_gate_pass={audit['task_profile_gate_pass']}")
```

一组典型输出是：

```text
task_profile_examples={'pretraining_time_h': 1.57, 'sft_time_h': 0.38, 'serving_slo_pass_rate': 0.5, 'agent_cost_usd': 0.062, 'complete_cost_usd': 1935}
smoke={'complete_case_passes': True, 'caught_pretraining_checkpoint_gap': True, 'caught_rlhf_rollout_gap': True, 'caught_serving_slo_gap': True, 'caught_agent_trace_gap': True}
metrics={'workload_type_classification': 0.941, 'resource_shape_completeness': 0.941, 'pretraining_profile_accuracy': 0.941, 'sft_iteration_profile_accuracy': 0.941, 'rlhf_pipeline_profile_accuracy': 0.941, 'evaluation_reproducibility_profile': 0.941, 'serving_slo_profile': 0.941, 'rag_freshness_retrieval_profile': 0.941, 'agent_tool_runtime_profile': 0.941, 'multimodal_resource_profile': 0.941, 'scheduler_policy_fit': 0.941, 'observability_metric_fit': 0.941, 'cost_model_fit': 0.941, 'artifact_lineage_fit': 0.941, 'safety_governance_fit': 0.941, 'task_profile_gate': 0.941}
hard_blocker_count=16
failed_cases=['generic_ai_job_bad', 'resource_vector_incomplete_bad', 'pretraining_no_checkpoint_bad', 'sft_no_dataset_version_bad', 'rlhf_rollout_ignored_bad', 'eval_not_reproducible_bad', 'serving_no_ttft_tpot_bad', 'rag_freshness_ignored_bad', 'agent_tool_trace_missing_bad', 'multimodal_payload_unbounded_bad', 'scheduler_one_queue_bad', 'observability_generic_gpu_only_bad', 'cost_token_only_bad', 'artifact_lineage_missing_bad', 'safety_governance_missing_bad', 'task_profile_gate_missing_bad']
task_profile_gate_pass=False
```

这个 demo 的重点不在规则本身，而在思维方式：平台团队要把“任务不同”落到数据结构和门禁上。预训练缺 checkpoint 不能过；RLHF 不统计 rollout 不能过；评估不可复现不能过；推理没有 TTFT / TPOT 不能过；Agent 没 trace 不能过；成本只按 token 估算也不能过。面试中如果能把这些差异说成审计指标，而不是只背概念表，会更接近真实平台设计。

## 6.16 常见误区

误区一：所有大模型任务都按训练任务设计。

推理和 Agent 有在线延迟、状态和权限要求，不能用纯训练平台思路解决。

误区二：SFT 只是小规模预训练。

SFT 更强调实验迭代、数据质量、评估和可复现性，不只是资源规模变小。

误区三：评估任务不重要，可以随便跑。

评估结果直接影响模型选择和上线决策，必须管理 benchmark、模型版本、结果和成本。

误区四：推理只要部署模型就行。

推理还要处理 batching、KV cache、限流、扩缩容、灰度、回滚、监控和成本。

误区五：Agent 只是多调用几次模型。

Agent 还涉及工具、权限、状态、trace、失败恢复和人工接管。

## 6.17 面试题

### 题 1：为什么预训练对 AI Infra 要求最高？

答：预训练通常数据规模大、训练时间长、GPU 数量多、通信复杂、checkpoint 体积大、成本高。任何节点、网络、存储或数据问题都可能导致巨大损失，所以需要强调分布式调度、拓扑感知、数据供给、checkpoint 和容错。

### 题 2：SFT 和预训练在平台诉求上有什么不同？

答：预训练更关注大规模稳定运行和吞吐效率，SFT 更关注快速实验迭代、数据版本、配置管理、模型 artifact、自动评估和可复现性。SFT 任务规模通常较小，但数量更多、变化更快。

### 题 3：推理任务为什么和训练任务的基础设施要求不同？

答：推理是在线服务，关注 TTFT、TPOT、QPS、p95/p99 延迟、KV cache、batching、自动扩缩容、限流降级和可用性。训练更关注吞吐、通信、checkpoint 和收敛稳定性。

### 题 4：RAG 的瓶颈为什么不一定在 LLM？

答：RAG 包含文档解析、embedding、向量索引、检索、rerank、权限过滤和上下文拼接。瓶颈可能出现在文档更新、索引构建、检索延迟、rerank 延迟或引用质量，而不仅是 LLM 生成。

### 题 5：Agent 工作负载为什么难以做成本治理？

答：Agent 的调用链不固定，可能多轮调用模型、检索、工具、代码执行和其他 Agent。步数、上下文长度、失败重试和工具延迟都不稳定，所以需要设置最大步数、token 预算、工具调用预算、超时和 trace。

## 6.18 小练习

练习一：为预训练、SFT、推理、RAG、Agent 分别设计 5 个关键监控指标。

练习二：给一个企业 AI 平台设计资源池。

要求：至少包含预训练资源池、SFT 资源池、评估资源池、推理资源池和低优先级资源池，并说明调度策略。

练习三：分析一个 Agent 成本失控问题。

假设某个 Agent 平均每个任务调用模型 30 次、工具 15 次，成本过高。请从任务拆解、步数限制、模型路由、缓存、工具失败和上下文裁剪角度分析。

练习四：设计一个评估平台任务画像。

要求：说明它需要哪些资源、如何调度、如何记录模型版本和 benchmark 版本、如何控制成本。

## 6.19 本章小结

本章讲了大模型任务画像。

你需要掌握：

1. AI Infra 服务的是不同 AI 工作负载，而不是抽象的“模型”。
2. 预训练是长周期、大规模、高成本任务，重点是吞吐、通信、数据供给、checkpoint 和容错。
3. SFT 更关注高频实验、数据版本、配置管理、评估和可复现性。
4. RLHF / RLAIF 同时包含训练、推理、reward 和 rollout，编排复杂度高。
5. 评估任务需要可复现、可比较、可解释和成本可控。
6. 推理任务关注在线延迟、吞吐、KV cache、扩缩容和单位 token 成本。
7. RAG 任务把 LLM、检索、embedding、索引、权限和引用串在一起。
8. Agent 任务关注状态、工具、权限、trace、步数预算和失败恢复。
9. 调度策略、监控指标和成本模型都要按任务画像设计。

下一章开始进入第二部分，讲 GPU 集群架构：单机多卡、多机多卡和机柜级互联。
