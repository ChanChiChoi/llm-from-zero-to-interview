# DeepSeek V4 书系映射与同步清单

核验日期：2026-09-09。本文是编辑清单，不替代正式章节。

## 核心技术到书册

| 技术 | 先修概念 | 首要书册 | 配套文件 | 正文必须回答的问题 |
|---|---|---|---|---|
| CSA | KV cache、稀疏注意力、压缩 | 21、24 | 百科、题库、练习、术语 | 压缩后如何检索，top-k 错过相关块怎么办？ |
| HCA | KV cache、MQA、低秩表示 | 06、21、24 | 百科、题库、练习 | 为什么更激进压缩仍能保留 dense attention？ |
| CSA/HCA 混合 | 稀疏与密集注意力 trade-off | 21、24 | 知识图谱、项目路线 | 哪些层适合稀疏，哪些场景需要密集？ |
| mHC | 残差、Hyper-Connections、双随机矩阵 | 13、21、23 | 百科、题库、练习 | 双随机约束怎样改善深层信号传播？ |
| Muon + AdamW | 优化器、矩阵更新、ZeRO | 05、13、23 | 术语、题库、论文路线 | 为什么不同模块采用不同优化器？全梯度矩阵带来什么系统成本？ |
| FP4/FP8 QAT | 量化、QAT、MoE | 05、06、23、24 | 百科、练习、项目路线 | 低精度误差如何进入训练和 KV/cache 系统？ |
| GRPO 专家培养 | RL、奖励模型、SFT | 05、16 | 题库、练习、论文路线 | 领域专家如何训练，奖励捷径如何控制？ |
| on-policy distillation | 知识蒸馏、reverse KL、模型合并 | 05、08、16 | 百科、题库、论文路线 | 为什么将多个领域能力统一到一个学生模型？ |
| 异构 KV cache | KV cache、分页、前缀复用 | 06、23、24 | 项目路线、知识图谱 | 不同注意力层的 cache 形状和淘汰策略如何共存？ |
| 1M context 评估 | 长上下文、污染、检索评估 | 07、21、24 | 练习、面试题库 | 1M 窗口如何证明“有效可用”，而非只看接口数字？ |

## 正文写作顺序

1. 从一个百万 token 项目问答的显存和延迟预算开始，解释为什么单纯扩大窗口不够。
2. 单独介绍原始 KV cache，再分别推导 CSA 的压缩与稀疏选择、HCA 的重压缩密集注意力。
3. 用小矩阵手算一个压缩块和 top-k 选择，提供零依赖 Python 示例；再说明生产实现的 indexer、MQA、滑动窗口和因果边界。
4. 用残差流的多通道例子讲 Hyper-Connections，再定义双随机矩阵和 Sinkhorn 型投影；区分 mHC 与 Attention Residuals。
5. 分别讲 Muon、FP4/FP8 QAT、GRPO 和 on-policy distillation，避免将架构、优化器、量化和后训练混在一节。
6. 最后把这些组件放回 DeepSeek V4 serving engine，解释 cache layout、prefix reuse、并行和硬件假设。

## 证据与边界

- 参数、上下文、架构名称、训练 token 和后训练流程：来自 DeepSeek V4 模型卡与技术报告源码。
- FLOPs/KV cache 比例：必须和报告中的基线、精度、上下文长度一起引用。
- V4-Pro-Max、V4-Flash-Max：推理 effort 配置，不是新基础权重。
- 任何未读源码的实现细节、具体压缩窗口 `m`、`m'`、top-k 数量和硬件吞吐不得先写成普适结论。

## 完成标准

每个技术主题写成独立小节或章节，至少包含：问题场景、旧方法瓶颈、直觉例子、定义和公式、可运行教学代码、优点、局限、适用边界、工程实现、面试追问和来源日期。完成正文后再同步配套文件并做广告注入、公式渲染和链接检查。

## DeepSeek V4.1-Flash 增补映射

| V4.1 主题 | 首要书册 | 需要固定的证据/实验字段 |
|---|---|---|
| CED 与 8B/16B prefill/decode 激活账本 | 21、06 | 输入/输出 token、模型 revision、backend、prefill/decode 时间；active parameter 不是 FLOPs |
| SWA Bounded Replay | 21、06、23、24 | 窗口边界、position offset、replay token、恢复时间、cache checksum、HBM/SSD |
| CSA2 `Full/Reindex/Reuse` | 21、24 | 层模式、main KV/indexer K 复用、压缩块、Top-K、因果边界 |
| Hierarchical Sparse Indexer | 21、07、24 | candidate pool recall、Top-K recall、候选建立成本和漏检类型 |
| FP4 E2M1 main KV | 05、06、21、24 | group scale、量化误差、dequant、bytes/token、检索和 Agent 回归 |
| Engram conditional memory | 21、05、07 | lookup hit/miss、参数/存储口径、消融、版本和访问延迟 |
| DSpark speculative decoding | 06、21、24 | 草稿接受长度、验证成本、目标调用数、回退、后端支持 |
| DeepSeek-ViT 原生多模态 | 15、21、17 | patch/图像 token、媒体顺序、位置、prompt injection、权限和工具边界 |
| `SFT -> RL -> OPD` 与 Agent 数据管线 | 05、16、17 | 任务/环境/rollout 版本、verifier、奖励、教师和独立复现边界 |
| 数值 `reasoning_effort` | 06、16、17、20 | 1--100、`low/high/max` 映射、推理 token、质量、延迟和成本 |

V4.1 正式专题见第二十一册第 81 章 [`81-deepseek-v4.1-flash-causal-encoder-decoder.md`](../../book-21-transformer-architecture-evolution/chapters/81-deepseek-v4.1-flash-causal-encoder-decoder.md)。模型卡中的 552B、890 bytes/global-KV-token、45T、1M 和 Agent benchmark 数字必须分别标注来源和条件，不能与 V4-Pro/Flash 或 Artificial Analysis 配置字段直接合并。
