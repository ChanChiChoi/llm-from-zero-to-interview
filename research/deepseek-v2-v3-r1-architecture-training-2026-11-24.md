# DeepSeek-V2、V3 与 R1：架构和训练路线证据

## 研究问题与边界

- 日期：2026-11-24。
- 问题：固定版本的 V2、V3、R1 公开资料分别支持哪些架构、MoE 负载均衡、MTP 与后训练主张？哪些结构或训练机制被误并成一个“DeepSeek 配方”？
- 范围：DeepSeek-V2 技术报告 v5、DeepSeek-V3 技术报告 v2、DeepSeek-R1 论文 v2，以及 DeepSeek 官方 R1 仓库；关注公开架构和训练阶段关系，不评估当前 hosted API、排行榜、production runtime 或目标硬件性能。
- 搜索方法：2026-11-24 搜索网页并直接打开 arXiv 固定 HTML 版本、官方 GitHub 仓库、DeepSpeed Course 与 Hugging Face Course。检索词包括 DeepSeek V3 Technical Report auxiliary-loss-free load balancing MTP modules、DeepSeek R1 paper V3-Base cold start rejection sampling pipeline、DeepSeek MLA tutorial decoupled RoPE key latent cache；另用 MLA、DeepSeekMoE、MTP、R1-Zero、V3-Base、rejection sampling、distillation 在原文定位段落。
- 结论状态：以下是来源支持的版本化陈述，不代表本地复现。没有运行模型、API、benchmark 或性能测试。

## 来源、支持范围与限制

### 主要技术来源

1. DeepSeek-AI, [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, arXiv:2405.04434v5](https://arxiv.org/html/2405.04434v5), 2024-06-19。阅读摘要、§2.1 的 low-rank KV joint compression 与 decoupled RoPE、§2.2 的 DeepSeekMoE、§3.1–3.2 的模型设置和扩展上下文、附录 D 的注意力消融。支持 V2 的 236B total / 21B active / 128K context，MLA latent KV 表示加独立 RoPE key，以及每个 MoE 层 2 个 shared experts、160 个 routed experts、每 token 激活 6 个 routed experts 等报告字段。8.1T token 和 benchmark/成本结果均属于该报告自己的数据与评估配置；不能据此推断跨模型质量、通用 cache 节省率或线上吞吐。
2. DeepSeek-AI, [DeepSeek-V3 Technical Report, arXiv:2412.19437v2](https://arxiv.org/html/2412.19437v2), v2。阅读摘要、§2.1 architecture、§2.2 MTP、§3 infrastructure、§4.2 hyper-parameters 和 §4.5 ablations。支持 V3 延续 MLA/DeepSeekMoE、671B total / 37B active、辅助损失免除型 batch-level routing bias 及补充的 sequence-wise balance loss、报告所述训练/推理无 token dropping、MTP 顺序模块和其推理可丢弃或复用于 speculative decoding。报告还描述 FP8、DualPipe 等系统协同，但不支持将其训练时设置外推为所有硬件上的效率结论。
3. DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, arXiv:2501.12948v2](https://arxiv.org/html/2501.12948v2), v2（2026-01-04 修订）。阅读 §§2–3、Appendix A.1、B.3、B.5–B.6、F.1 和限制讨论。支持 R1-Zero/R1 以 V3-Base 为基础、R1-Zero 无预先 SFT 的 RL 路线、规则型准确性/格式奖励，以及 R1 的冷启动、第一轮 RL、rejection sampling、混合推理/非推理 SFT、第二轮 RL 等阶段关系；也支持蒸馏模型使用 R1 生成数据。作者关于 reasoning 行为、benchmark 和限制的结论绑定论文里的模型、提示、训练和评测设置，不证明所有任务均可使用相同 recipe。
4. DeepSeek AI, [DeepSeek-R1 官方仓库 README](https://github.com/deepseek-ai/DeepSeek-R1)。阅读模型摘要、参数表和 distill 模型列表。明确列出 R1/R1-Zero 基于 DeepSeek-V3-Base，671B total / 37B active，并列出基于 Qwen2.5/Llama 的稠密蒸馏 checkpoint。README 能确认发布方给出的模型身份和基座，不能支持未披露的内部实现或独立评测结论。

### 教学与组织材料

1. [DeepSeek MLA, Built From the Paper — DeepSpeed Course](https://yiqiao-yin.github.io/deepspeed-course/docs/tutorials/llms/deepseek-mla)，阅读 MLA 的 KV cache 问题、decoupled RoPE、矩阵吸收和逐步 cache 解释。用于观察“状态成本—latent 表示—位置分支—执行”的教学组织；其中 GLM-5 的 cache 数值和本地代码测试只属于该材料的配置/实现，不移植到 V2。
2. [Understanding the DeepSeek R1 Paper — Hugging Face Course](https://huggingface.co/docs/course/en/chapter12/3)，阅读 R1 路线、冷启动、rejection sampling、SFT 和 RL 段落。用于观察训练阶段的读者组织；阶段事实仍回到 R1 论文核验。
3. [Build a DeepSeek Model from Scratch, Chapter 3 overview — Manning](https://livebook.manning.com/learn/build-a-deepseek-model-from-scratch/chapter-3)，用于观察 MLA 缓存取舍、数据流、cache 与计算的组织方式；不作为 V2/V3 事实来源。
4. Ruan, [Stanford CS336 lecture notes, lec04](https://github.com/QihongRuan/stanford-cs336-notes/blob/main/lec04.html)，用于检查 MoE/MLA/MTP 的课程组织和术语联系。笔记中把 V3 的 routing bias 调整描述为在线梯度方案的说法未采用；负载 bias 的更新方式以 V3 报告 §2.1 为准。

## 关键发现与证据边界

| 主题 | 来源直接支持 | 证据边界 |
|---|---|---|
| V2 MLA | 共享低秩 KV latent 与单独携带 RoPE 的 key；报告给出固定模型的 cache/性能实验 | 不能据此说“没有 KV cache”，也不能承诺任意 serving backend 的速度或固定压缩倍率 |
| V2 DeepSeekMoE | shared/routed expert 组成与配置可在 V2 报告查到 | active parameters 不等同于权重常驻量、通信量或请求成本 |
| V3 load balancing | top-k 选择使用每专家 bias；输出 gate 仍取原 affinity；step 末按专家超载/不足调整 bias；另有 sequence-wise 辅助损失 | 报告将 bias 描述为按负载反馈调整；不称所有数据、推理引擎都无 token drop |
| V3 MTP | 顺序预测额外未来 token，保留预测深度间因果链；推理可丢弃模块或复用于 speculative decoding | 报告的消融和复用提案不等同于任意后端的 runtime 测量或接受长度保证 |
| R1-Zero/R1 | 都基于 V3-Base；Zero 无预先 SFT，R1 增加冷启动和多阶段训练 | 论文中的 reasoning、benchmark 和“反思”描述不证明通用能力、人类式认知或跨任务因果 |
| R1 distill | Qwen2.5/Llama 基座用 R1 生成数据微调；论文对蒸馏 checkpoint 描述为 SFT 路线 | 不把学生架构与参数归为 R1 主模型 |

## 未由这些来源解决的问题

- V2 与 V3 的参数、性能或效率数字来自不同报告与设置，不能拼成受控版本对比或统一排名。
- 公开报告没有提供所有生产 kernel、服务配置、硬件负载和线上质量/延迟结果；本轮未测量这些对象。
- MTP 的报告内消融和 speculative-decoding 提案不等于本地实现或任意引擎接受长度；R1 的作者评测不外推至开放任务或其他 harness。
