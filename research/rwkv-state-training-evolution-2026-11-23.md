# RWKV 状态与训练形态演进

## 研究问题与范围

- 日期：2026-11-23。
- 研究问题：RWKV-4 的 time-mixing、WKV 归一化递归和 channel-mixing 分别如何工作；RWKV-5/6/7 如何改变状态表示或更新；“并行训练、递归推理、固定状态”各自准确指什么；截至检索日期官方资料如何描述 RWKV-8。
- 范围：模型内部序列信息流、状态方程、版本边界和对应证据。未研究部署系统、服务状态恢复、横向性能排名或独立检索基准。
- 检索平台：网络搜索；原始论文 HTML、COLM/OpenReview 正式论文、RWKV 官方 wiki 与作者 GitHub、作者参与的解释文章和大学课程讲义。
- 实际检索词：
  - RWKV-4 RNN Transformer parallel training WKV recurrence arXiv 2305.13048
  - “Eagle and Finch” RWKV matrix-valued states dynamic recurrence COLM 2024
  - RWKV-7 Goose Expressive Dynamic State Evolution paper COLM 2025
  - site:rwkv.cn RWKV-8 experimental DeepEmbed ROSA 2026
  - site:mobitec.cuhk.edu.hk/ierg5050Fall2025 RWKV Beyond Transformers slides
  - RWKV architecture course lecture notes RWKV-4 time mixing channel mixing
  - RWKV official RWKV-8 Heron experimental current paper 2026
  - RWKV-7 “replacement key” “k̃”
  - RWKV-7 “v_t^T” “k_t” WKV state update replacement key
  - site:github.com/BlinkDL/RWKV-LM “kk” “kk” wkv7 state update

## 核心发现与证据边界

### RWKV-4

Peng 等人的原始论文在 §3.1.1 将 time-mixing 与 channel-mixing 投影写成当前表示和上一位置表示的线性混合；§3.1.2 给出逐通道衰减、key 权重、当前位置 bonus 和归一化分母共同构成的 WKV；§3.1.3 给出 receptance 门控和 squared-ReLU channel-mixing。附录 D 将 WKV 写成 numerator/denominator 递归，并指出稳定实现还维护用于指数重标度的辅助状态。论文 §3.2—3.3 同时说明，位置投影可按整段输入计算，而 WKV 有时间依赖；训练时可使用相应序列计算，生成时按 token 递归更新。由此只能说序列计算可并行化，不能说各时间步彼此独立。

对每个通道，历史项权重为 $\exp(-(t-1-i)w+k_i)$，当前项在读出中使用 $\exp(u+k_t)$。递归状态以未加 bonus 的 $\exp(k_t)$ 项更新，供后续位置使用。WKV 分子、分母和读出均为逐通道运算。原论文附录 D 还列出每层状态中的两个 shift 输入、numerator、denominator 与稳定化指数状态，支持“状态维度固定但组成依版本变化”，不支持“所有 RWKV 都只有两个统计向量”。

### RWKV-5 Eagle 与 RWKV-6 Finch

Peng 等人的 COLM 2024 论文将 Eagle/RWKV-5 描述为多头矩阵状态、固定 learned decay、改造 receptance 和门控；Finch/RWKV-6 在此基础上引入依赖输入的 token shift 与逐位置衰减。每头 WKV 矩阵形状为 $(D/h)\times(D/h)$。论文附录给出 recurrent inference 形式，且展示 time dimension 的并行计算方式。该架构来源支持版本机制；其模型训练规模和 benchmark 不在本章被转换成一般性能胜负。

### RWKV-7 Goose

Peng 等人的 COLM 2025 论文 §4 定义逐头矩阵状态、输入相关的衰减、normalized removal key、in-context learning rate 和独立的 replacement key，并给出状态递归与并行展开形式。更新由旧状态的衰减/方向性移除和新 value-key 外积写入组成。论文的 regular-language/state-tracking 结论依赖其形式化模型与假设；它不等同于一般语言建模能力的普遍优越性，也不证明所有检索 workload 的行为。正文仅用递归方程说明机制。

### 训练与推理的关系

RWKV-4 原论文 §3.2 明确指出 WKV 位置有 serial scan 依赖，矩阵投影是主要并行计算；长序列可以采用 scan 类方法。v5—v7 论文也给出递归与序列并行表示。因而“Transformer 式训练”是对整段已知输入执行可并行化的序列算子，不能写成 full attention、没有因果依赖或位置完全独立。

### RWKV-8（截至 2026-11-23）

RWKV 官方 wiki 将 RWKV-8/Heron 标为实验阶段，并分别列出 DeepEmbed 与 ROSA：前者是按 token index 读取的 FFN embedding 组件，后者是后缀自动机式检索/信息传播组件。作者 GitHub 的 RWKV-8 页面标题为 “Improving RNNs (RWKV-8 and beyond)”，呈现的是探索框架。此次检索未发现一篇统一的同行评审 RWKV-8 论文取代 v4—v7 的状态更新方程；因此这些组件应与 WKV 递归方程分开叙述。官方 wiki 对 DeepEmbed/ROSA 的能力描述属于项目方材料，不作为独立验证或通用性能证据。

## 来源与阅读位置

| 来源 | 类型与版本 | 阅读位置 | 支持与不支持 |
| --- | --- | --- | --- |
| [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/html/2305.13048) | 原始论文；Findings of EMNLP 2023 | §§3.1.1–3.3、Appendix D | 支持 RWKV-4 的 token shift、WKV、门控、递归与 time-parallel 关系及状态组成；不支持后续版本共用同一公式。 |
| [How the RWKV language model works](https://johanwind.github.io/2023/03/23/rwkv_details.html) | 作者之一 Johan Wind 的解释文章，2023-03-23 | time-mixing、channel-mixing 与加权平均递归段落 | 提供 RWKV-4 的可读机制映射；最小讲解代码不等于所有版本的正式实现或生产 kernel。 |
| [Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence](https://openreview.net/pdf?id=soz1SEiPeq) | COLM 2024 正式论文 | §§3、4、Eagle/Finch recurrence 与附录实现 | 支持 v5 矩阵状态、v6 动态 shift/decay 及论文所述实验条件；不外推到所有 checkpoint 或系统性能。 |
| [RWKV-7 “Goose” with Expressive Dynamic State Evolution](https://openreview.net/pdf/74011cffc1082ac9394afde92802c810c51baabf.pdf) | COLM 2025 正式论文 | §4、式 (15)—(22)、相关附录 | 支持 v7 的状态递归、移除/替换 key 定义、并行展开和形式化结果范围；不支持家族级通用能力结论。 |
| [RWKV 架构及历史](https://www.rwkv.cn/docs/RWKV-Wiki/RWKV-Architecture) | 项目官方 wiki；2026-11-23 在线读取 | RWKV-7 与 RWKV-8/Heron 部分 | 支持官方当前对 v8 实验状态、DeepEmbed/ROSA 的公开描述；页面的能力主张没有据此视为独立复现。 |
| [Improving RNNs (RWKV-8 and beyond)](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md) | 作者 GitHub 当前 main 页面，滚动版本 | “Larger State” | 支持作者将其列为探索中的后续方向；main 页面会变化，不视为固定论文规格。 |
| [RWKV: architecture overview](https://mobitec.ie.cuhk.edu.hk/ierg5050Fall2025/static_files/slides/Beyond-Transformers.pdf) | CUHK IERG5050 Fall 2025 课程讲义 | RWKV 定义、WKV 方程、Time Mixing/Channel Mixing 图 | 用于观察教学图示如何把组件和残差路径组织起来；不作为方法定义的唯一依据，也不证明某种教学组织对所有读者更有效。 |

## 未支持或待核范围

- 不以 state shape 单独断言 RWKV 在精确复制、ICL 或多条目检索上必然失败或成功；这些需要任务与模型版本匹配的实证。
- 不把原论文的并行训练表述写成各时间步独立，也不把整段训练描述成 softmax self-attention。
- 不把 RWKV-4 的向量 numerator/denominator 方程、RWKV-5/6 的每头矩阵 state 和 RWKV-7 delta 更新混为一条公式。
- 不从官方 wiki 对 RWKV-8 组件的描述推出其已经形成统一稳定的架构定义，或推断其普遍质量、速度和检索效果。
- 不把 fixed state memory 表述成整个模型显存、计算量或成本都固定；模型宽度、层数、头维度、batch 和实现都会改变实际占用。
