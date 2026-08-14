# 第 58 章 DeepSeek-V2、V3 与 R1：MLA、MoE、MTP 和推理强化

## 58.1 DeepSeek 系列为什么值得单独学习

DeepSeek 系列把几个通常分开讨论的方向放在了同一条路线中：注意力的 KV 压缩、稀疏专家、训练效率、多 token prediction 和可验证推理强化。它的研究价值不在于提供一个单点技巧，而在于展示架构、训练和推理成本如何协同设计。

阅读时必须按版本区分。DeepSeek-V2、V3 和 R1 的模型规模、训练配方、后训练目标和公开资料不同；不能把其中一个版本的结构直接套到另一个版本，也不能把社区对未公开版本的猜测写成事实。

## 58.2 MLA 解决 KV cache 的哪一部分

普通 attention 在 decode 时为每个历史 token 保存 key 和 value。MLA 的基本直觉是保存更紧凑的 latent 表示，再在计算时恢复或在 latent 空间完成必要的投影。

设隐藏表示为 h_t，压缩 latent 为 c_t：

~~~math
c_t=W_{\mathrm{down}}h_t,\qquad
\hat k_t=W_{\mathrm{up}}^kc_t,\qquad
\hat v_t=W_{\mathrm{up}}^vc_t
~~~

如果 cache 保存 c_t，理想 payload 可能低于完整 K/V；真实实现还要考虑位置部分、投影计算、量化、对齐和 kernel。MLA 的目标不是删除 attention，而是把长上下文 serving 的状态和带宽问题纳入架构。

## 58.3 latent cache 的信息瓶颈

压缩维度越小，存储越省，但不同历史可能映射到相似 latent。对主题摘要，压缩可能足够；对数字、代码标识符和版本冲突，细节损失会更明显。

可以把压缩决策写成：

~~~math
\min_C\;
L_{\mathrm{task}}(C(H))
+\lambda L_{\mathrm{cache}}(C(H))
~~~

其中 task loss 应包含精确引用和关键字段，而不只是 perplexity。若只优化语言流畅度，latent 可能保留主题却删除改变结论的例外条件。

## 58.4 DeepSeekMoE 的容量分配

MoE 把一个大的 FFN 拆成多个专家，每个 token 只使用少数专家。DeepSeekMoE 相关公开资料强调更细粒度专家和共享专家等设计方向，目标是改善专家利用率和参数效率。

抽象表示为：

~~~math
y_t=f_{\mathrm{shared}}(x_t)
+\sum_{e\in\mathrm{TopK}(r(x_t))}
g_{t,e}f_e(x_t)
~~~

共享路径提供所有 token 都需要的通用变换，路由专家承载更细的分工。这里是教学抽象，具体专家数量、共享方式、路由和损失要以对应技术报告为准。

## 58.5 MoE 的训练和通信成本

MoE 每 token 的矩阵乘可以减少，但路由和通信不能忽略：

~~~math
C_{\mathrm{MoE}}
\approx C_{\mathrm{expert\_matmul}}
+C_{\mathrm{router}}
+C_{\mathrm{dispatch}}
+C_{\mathrm{alltoall}}
~~~

在多节点训练中，all-to-all 的带宽、拓扑和拥塞可能决定吞吐。训练稳定性还依赖专家负载、容量、token drop、路由 logits 和 auxiliary loss。

因此报告 MoE 模型时，同时给出 total parameters、active parameters、专家数、top-k、通信拓扑和实际 tokens/s，比只给一个总参数更有意义。

## 58.6 V3 的系统化优化视角

DeepSeek-V3 相关报告展示了大规模 MoE 训练的系统和配方问题。学习重点包括：如何让大量专家稳定训练，如何控制通信和显存，如何利用低精度，如何做数据和训练阶段安排，以及如何通过多 token 目标改善训练信号。

这些优化互相耦合。低精度可能降低成本，却增加路由、归一化和梯度稳定要求；专家数量增加容量，却增加通信和负载均衡；更长上下文提升任务能力，却提高 cache 和训练资源。

## 58.7 Multi-Token Prediction 的目标

标准 next-token prediction 在位置 t 预测 x_{t+1}。MTP 还让模型预测更远的 token，例如：

~~~math
\mathcal{L}_{\mathrm{MTP}}
=\sum_{k=1}^{K}\lambda_k
\left[-\sum_t
\log P_\theta(x_{t+k}\mid x_{1:t})\right]
~~~

它提供更丰富的训练信号，也可能为推理时的多 token 草稿或 speculative decoding 提供结构基础。不同实现可能使用独立预测头、共享 trunk 或其他形式，不能把这个公式当作所有模型的完整实现。

MTP 的质量要看近 token 和远 token 的损失、生成一致性、额外参数、训练成本以及推理接受长度。预测得更远不等于目标模型会接受草稿。

## 58.8 Reasoning RL 和可验证奖励

R1 相关路线把强化学习用于数学、代码等可以自动验证的推理任务。设结果验证器为 V(y,x)，奖励可以抽象成：

~~~math
r(x,y)=\mathbf{1}[\mathrm{Verify}(x,y)]
-\lambda\mathrm{FormatError}(y)
~~~

可验证奖励的优势是减少主观偏好代理，让正确答案、编译测试或证明检查成为反馈来源；局限是验证器覆盖有限，模型可能优化格式、钻验证器漏洞或在不可验证任务上迁移不稳。

推理模型还要考虑 test-time compute。增加思考 token、采样候选、投票或工具调用会提高成功率的可能性，也会提高延迟和成本。

## 58.9 推理成本的分解

一个 reasoning 请求的单位成本可以写成：

~~~math
C_{\mathrm{reason}}
=C_{\mathrm{prefill}}
+C_{\mathrm{think\_decode}}
+C_{\mathrm{tool}}
+C_{\mathrm{verify}}
+C_{\mathrm{retry}}
~~~

如果只看最终答案质量，可能忽略思考长度过长、重复采样或失败重试带来的成本。系统应记录 visible output、hidden/reasoning token（若接口提供统计）、工具时间、验证次数和总 p95。

## 58.10 一个简化的验证式 RL 实验

下面代码展示“生成—验证—记录奖励”的最小闭环：

~~~python
def verify_answer(answer, gold):
    normalized = answer.strip().lower()
    return int(normalized == gold.strip().lower())


def score_candidates(candidates, gold):
    return [
        {"answer": item, "reward": verify_answer(item, gold)}
        for item in candidates
    ]


print(score_candidates(["42", "41", "42"], "42"))
~~~

真实 RL 还要处理轨迹采样、优势估计、KL、批次、奖励稀疏、格式约束和验证器可靠性。这个例子的作用是说明“奖励来自可观察结果”，而不是说明已经实现 R1。

## 58.11 MLA、MoE、MTP 和 RL 的组合

四条路线分别作用于不同层面：

| 方向 | 主要目标 | 主要成本 |
| --- | --- | --- |
| MLA | 压缩历史 attention 状态 | latent 投影、信息瓶颈、kernel |
| MoE | 增加总容量并控制激活计算 | 路由、通信、负载均衡 |
| MTP | 增强训练/草稿预测信号 | 额外头、训练目标和一致性 |
| Reasoning RL | 提升可验证任务推理 | rollout、验证、长输出成本 |

组合后不能简单把每个收益相加。MLA 改变 cache，MTP 影响 speculative，MoE 影响每一步的执行图，RL 改变输出长度和使用工具的行为。最终必须做端到端消融。

## 58.12 评估 DeepSeek 路线

架构侧测试 MLA cache bytes、长上下文 recall、latent 量化和投影时间；MoE 侧测试专家负载、通信、active FLOPs、token drop 和 p99；MTP 侧测试 acceptance length、额外开销和错误回退；reasoning 侧测试 verifier accuracy、思考长度、答案正确率和单位成功成本。

还要做组合消融：

~~~text
baseline dense attention
  -> + MLA
  -> + MoE
  -> + MTP
  -> + reasoning post-training
~~~

每次只改变一个主要因素，才能知道收益来自哪里。

## 58.13 常见误区

第一，把 MLA 写成“没有 KV cache”。第二，用 total parameters 宣称每 token 成本。第三，把 MTP 直接等同于 speculative decoding。第四，把 R1 的 reasoning RL 写成通用任务都能提升。第五，把可验证奖励的成功推广到开放式主观任务。第六，忽略模型版本和公开资料的边界。

## 58.14 MLA、MoE、MTP 与 RL 的耦合

MLA 主要改变历史表示和 KV 资源，MoE 改变条件容量和通信，MTP 改变训练/草稿预测目标，R1 路线改变后训练和 test-time compute。它们可以互相增强，也会互相引入约束：MLA 的投影影响 speculative kernel，MoE 的路由影响草稿一致性，MTP 的远 token 目标影响生成分布，RL 又可能改变输出长度和工具行为。

评估不能把这些优化的收益简单相加。应分别做 MLA 开关、MoE 路由、MTP head、RL 后训练和 serving kernel 的消融，并记录质量、显存、通信、接受长度和单位成功成本。

## 58.15 推理强化的可验证边界

数学和代码任务有 verifier，可以把结果正确性转成奖励；开放问答、复杂工具和现实世界任务的奖励更难可靠定义。R1 式 reasoning RL 的公开证据不能自动说明所有任务都能用同样的 RLVR 配方。

对 reasoning 模型要区分：思考 token 数、最终答案正确率、过程是否可验证、工具调用是否正确和用户可接受延迟。更长的思考链可能提高质量，也可能增加错误搜索、成本和暴露敏感轨迹的风险。

## 58.16 公开模型阅读的证据卡

每个版本都应建立一张证据卡：公开结构、训练目标、参数口径、上下文、量化、benchmark 条件、未知字段和可复现实验。V2、V3、R1 的内容必须分开登记，不能用系列名替代版本。

如果资料只支持“采用某类路线”，正文就只写到该层；只有技术报告或代码明确给出层表、路由和训练配方时，才可写成具体事实。这种克制是研究阅读的一部分。

## 58.17 MLA、MoE 和训练路线要分开讲

DeepSeek-V2/V3/R1 公开材料常被放在一条“更高效、更会推理”的叙事里，但架构、训练和后训练是不同层。MLA 主要改变 KV/latent 表示和 serving 资源，MoE 改变参数激活与通信，R1 路线强调可验证奖励、推理和蒸馏；不能把三者的效果混成一个神奇模块。

阅读时建立配置—资源—能力映射：MLA 需要测 KV 与 decode 带宽，MoE 需要测 expert load 和通信，RLVR 需要测 verifier、rollout、推理长度和泛化。每项都要绑定公开报告和实验条件。

## 58.18 公开技术报告的可验证范围

技术报告可以支持公开写出的结构、训练阶段、公开 benchmark 和数据规模。它通常不能支持每个路由阈值、全部数据筛选、内部失败轨迹、闭源服务的动态预算和真实产品 harness。读者应为每条结论标记证据类型：

| 结论 | 可以直接写 | 仍需限定 |
| --- | --- | --- |
| MLA/latent 路线 | 报告明确描述的结构和缓存目标 | 具体 kernel、量化误差和服务吞吐 |
| MoE 训练 | 公开的专家、路由和训练阶段 | 全部通信优化、溢出处理和失败 run |
| reasoning RL | 公开的奖励、rollout 或蒸馏流程 | 是否迁移到开放任务、真实预算和隐藏 verifier |
| benchmark | 给定版本、模板和预算下的自报结果 | 跨 harness 的普遍能力和单位成功成本 |

例如“R1 类路线提高了数学推理”可以由公开评测在相应条件下支持；“相同 RL 配方适合所有开放任务”则超出了证据范围。严谨的文章要同时说明已知和未知：未知不是空白占位，而是决定下一步实验、复现和面试回答语气的边界。

如果多个版本共享系列名，还要按版本建立 evidence card，记录 model id、训练阶段、输出协议、推理预算、工具、评估集和日期。这样才能避免把 V2 的 MLA、V3 的系统优化和 R1 的后训练路线拼成一个并不存在的单一模型。

## 58.19 R1 类推理路线的工程代价

推理能力提升可能带来更长 reasoning token、更高 KV/state 占用、更大尾延迟和更复杂的安全/协议状态。部署时应把 reasoning budget、工具、verifier、stream event、cost per successful task 和高风险拒答一起评估；模型分数提高不等于产品单位任务成本降低。

## 58.20 资料范围与训练判断

DeepSeek-V2 技术报告是 MLA 和 DeepSeekMoE 的主要公开来源；DeepSeek-V3 报告讨论了大规模 MoE 训练和工程优化；DeepSeek-R1 报告讨论了 reasoning RL 和蒸馏路线。不同版本和接口的细节应以对应官方资料为准。

DeepSeek 路线的核心启示是：模型创新正在把网络结构、KV 状态、训练目标、验证器和推理调度放在一起优化。读者应学会画出完整资源和证据链，而不是只记住四个缩写。

## 58.21 面试问题与练习

**问：MLA 和 GQA 都在优化 KV cache，有什么不同？**

GQA 通过减少 KV head 数共享历史表示；MLA 进一步把历史表示压到 latent 空间并在计算时恢复或使用，压缩机制和信息瓶颈不同。

**问：MTP 是否等于 speculative decoding？**

不是。MTP 是训练或模型预测目标，speculative decoding 是用 draft 预测并由 target 验证的推理算法；MTP 可能提供草稿信号，但还需要接受/拒绝协议。

**练习：**设计 DeepSeek 路线的四阶段消融表，包含 cache、专家负载、接受长度、推理 token、答案正确率和单位成功成本。

资料入口：

- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

## 58.22 结构、训练和推理解耦再耦合

MLA 主要改变历史表示和 KV 资源，MoE 改变参数激活与通信，MTP 改变预测目标，RL/验证器改变推理行为；它们可以分别研究，却会在完整系统里重新耦合。MLA 的 cache 节省可能提高并发，MTP 的草稿质量可能改变 speculative acceptance，reasoning token 增长又会放大 KV、调度和单位成功成本。

一个严谨的消融按层次展开：dense/MHA baseline，替换 MLA，加入 MoE，加入 MTP，再加入 RL/蒸馏；每步固定主要训练和解码条件，报告 cache bytes、expert load、acceptance length、reasoning tokens、答案质量、p99 和成功成本。这样才能避免把整套系统的收益归因给其中一个缩写。

## 58.23 训练路线的阶段性结论

DeepSeek 路线的学习重点是把 MLA/MoE、训练效率、reasoning、蒸馏、验证器、KV/state 和 serving 调度放进同一条证据链。架构、训练和推理成本相互耦合，公开报告支持到的范围之外不能用产品名称补全未知细节。

## 58.24 四条证据链的连接方式

阅读 DeepSeek 路线时，可以分别建立架构、训练、推理和任务四条链。架构链记录 MLA/MoE 的信息与资源路径；训练链记录 data、SFT、RL、蒸馏和 verifier；推理链记录 reasoning token、MTP/speculative、KV 和调度；任务链记录数学、代码、工具、安全、延迟和成本。

四条链只有通过 controlled experiment 才能连接。例如 reasoning 质量提高可能来自 RL、蒸馏、更多 test-time budget 或更好的 verifier；MLA 的 cache 节省可能被更长 reasoning token 抵消。若没有分阶段消融，不能把系统总收益归给单一组件。

## 58.25 版本、口径和未知项

模型家族名不能替代版本。总参数与 active parameters、训练上下文与服务上限、模型卡 benchmark 与独立 harness、MTP 训练目标与 speculative runtime 都应分开记录。公开技术报告支持的内容要绑定 revision 和发布日期；产品页信号若缺少技术报告，只能作为一方观察。

一个可靠的模型卡阅读表包含：claim、来源、版本、条件、可复现实验、未知和失效日期。这样才能在新版本发布后只更新受影响的命题，不把旧数字或内部推测扩散到整个家族。

## 58.26 DeepSeek 路线的工程取舍

这条路线的可迁移启发是把稀疏参数、压缩历史、可验证奖励和推理服务共同建模，而不是直接复刻某个模块。部署者要问：active compute 是否真的降低 GPU/通信成本，MLA/其他 cache 是否兼容目标 engine，reasoning budget 是否满足 p99，RLVR/verifier 是否覆盖业务任务，蒸馏是否保持安全和工具协议。

最终用 target workload 做 gate：质量、引用/代码/工具成功率、KV/通信、TTFT/TPOT、p99、单位成功成本、回退和治理都通过后，才把公开路线转成生产设计。否则它仍然是重要的研究锚点，而不是无条件的架构推荐。
