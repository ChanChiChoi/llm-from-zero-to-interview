# 第 52 章 Chinchilla、PaLM 与 Compute-Optimal Training

## 52.1 为什么“参数越大越好”不完整

早期大模型比较经常把参数量当成最醒目的指标：模型越大，似乎能力越强。但训练一个模型同时消耗参数计算和数据计算。如果参数增长很快，训练 token 没有同步增长，模型可能处在欠训练状态；如果数据增长很快，模型容量又可能不足。

真正要优化的是固定计算预算下的最终能力，或者固定目标损失下的训练成本。Chinchilla 的影响在于把讨论从“最大参数”推进到“参数和 token 如何匹配”。PaLM 则展示了大规模路径、数据和系统扩展如何共同支撑能力。

## 52.2 一个最小的计算模型

令模型参数量为 N，训练 token 数为 D，每个 token 的训练计算近似与 N 成正比，则总训练计算可以写成：

~~~math
C_{\mathrm{train}}\approx kND
~~~

这不是所有系统的精确 FLOPs 公式。激活函数、序列长度、MoE 路由、通信、重计算和硬件利用率都会改变真实成本。但它足以说明一个事实：只增加 N 或只增加 D 都不能独立决定最优点。

假设验证损失由模型误差和数据误差共同决定：

~~~math
L(N,D)\approx L_\infty
+aN^{-\alpha}
+bD^{-\beta}
~~~

在计算约束 ND=C/k 下，最优 N 和 D 取决于 α、β、常数和训练阶段。不同数据分布和目标任务会导致不同最优比例。

## 52.3 Chinchilla 的核心启示

Chinchilla 通过一组受控规模实验拟合参数和数据的关系，结论之一是，很多此前的大模型训练 token 数偏少。与其训练一个参数极大的欠训练模型，不如在相同计算预算下使用较小模型并喂更多高质量 token，可能得到更低的损失和更好的下游能力。

常见的教学近似会把 Chinchilla 说成“每个参数约 20 个 token”。这个说法只能作为某一研究设定下的记忆锚点，不能当作所有模型、所有数据和所有后训练阶段的固定定律。模型架构、tokenizer、数据质量、去重、训练目标和推理成本都可能改变决策。

更准确的表述是：在给定的 compute-optimal 研究范围内，参数和训练 token 都应随计算预算增长，而不应只追求参数规模。

## 52.4 PaLM 的系统视角

PaLM 关注的不只是一个规模曲线，还包括大规模训练系统、数据混合、模型结构和并行。它展示了在大规模计算基础设施上训练高容量语言模型的路径，说明模型能力提升依赖软件、硬件、通信和数据工程的共同成熟。

PaLM 类工作给工程师的启示是，训练规模不是把 GPU 数量乘大那么简单。随着规模增加，数据加载、梯度同步、激活通信、故障恢复、checkpoint 和评估都会成为一等问题。

## 52.5 从损失到产品能力

验证损失是重要指标，但不是产品能力的完整代理。对于一个助手，还要关心指令遵循、事实性、代码执行、安全拒答、工具调用和延迟。compute-optimal pretraining 只解决预训练阶段的一部分资源分配，后训练和 serving 需要单独优化。

可以把总成本分成：

~~~math
C_{\mathrm{total}}
=C_{\mathrm{pretrain}}
+C_{\mathrm{posttrain}}
+C_{\mathrm{eval}}
+C_{\mathrm{serve}}
+C_{\mathrm{data}}
~~~

一个预训练损失更低的模型，如果推理显存、延迟或后训练成本高很多，未必是具体产品的最优选择。

## 52.6 数据质量改变 token 的价值

两个数据集即使 token 数相同，信息价值也可能不同。重复网页、模板噪声和错误代码不能与新颖、可靠、任务相关的数据等价计算。可以把有效 token 近似写成：

~~~math
D_{\mathrm{effective}}
=\sum_{i=1}^{D}q_i
~~~

其中 q_i 是样本质量或有效信息权重，实际不一定能被准确估计。去重、来源治理、语言比例、污染检测和高价值域采样都可能改变 compute-optimal 选择。

数据重复还会影响模型记忆和评估污染。更大 D 如果只是重复同一分布，可能带来边际收益下降，甚至加重隐私和版权风险。

## 52.7 一个可复现的 scaling 实验设计

不要直接训练一个巨型模型再猜规律。可以训练一组小模型，控制数据配比和训练步数，拟合损失曲线：

~~~python
import math


def power_law(n, limit, coefficient, exponent):
    return limit + coefficient * n ** (-exponent)


points = [(1e8, 3.10), (3e8, 2.82), (1e9, 2.55)]
for n, loss in points:
    print(math.log(n), loss)
~~~

这个片段只展示拟合输入的形状，不代表真实实验已经足够。正式实验要记录模型参数、训练 token、有效 batch、数据版本、tokenizer、硬件 FLOPs、wall-clock、验证集和随机种子，并在独立规模上验证外推。

## 52.8 计算预算如何拆成决策

给定预算后，先确定目标：最低验证损失、最快达到某能力验收条件，还是最低单位任务成本。然后做四步：

1. 估计可用训练 FLOPs 与硬件有效利用率。
2. 用 scaling curve 选择若干 N、D 候选。
3. 把数据质量、上下文长度、tokenizer 和后训练预算纳入比较。
4. 在中等规模模型上验证曲线和目标任务，再决定最终配方。

如果只把预算投到参数，可能得到大但欠训练的模型；如果只投到 token，可能得到小模型无法承载目标能力。工程决策应把训练和部署的总拥有成本一起看。

## 52.9 训练 token 与上下文长度不是同一件事

训练 token 总数 D 是整个训练过程看到的 token 数；上下文长度 T 是一次样本或一次 batch 中的序列长度。增加上下文长度会改变 attention 计算、位置分布、样本 packing 和长程学习信号，但不等于自动增加有效数据量。

例如，同样一亿 token，如果全部由短重复样本组成，模型不一定学到长距离依赖；如果用长文档 packing，可能增强跨段训练，却增加 O(T^2) 或 cache 相关成本。长度策略应与任务目标分开记录。

## 52.10 MoE 下的 compute-optimal 需要更谨慎

MoE 有总参数量 N_total 和每 token 激活参数 N_active。训练计算更接近 active 参数，但通信、路由和存储会受 total 参数影响：

~~~math
C_{\mathrm{token}}
\approx kN_{\mathrm{active}}+C_{\mathrm{route}}+C_{\mathrm{comm}},
\qquad
M_{\mathrm{weights}}\propto N_{\mathrm{total}}
~~~

因此不能把 dense scaling 的 N 直接替换成 MoE 总参数。专家数量、top-k、负载均衡、容量因子、通信拓扑和专家利用率都需要报告。

## 52.11 训练预算和推理预算的冲突

更大模型可能在少量提示下更强，但每个 token 的推理成本更高；更小模型可能需要更多 reasoning token、检索或工具调用才能完成同一任务。产品目标可能是最低交互延迟、最低美元成本或最高复杂任务成功率。

单位成功任务成本可以写成：

~~~math
C_{\mathrm{success}}
=\frac{C_{\mathrm{inference}}+C_{\mathrm{tool}}+C_{\mathrm{human}}}
{\Pr(\mathrm{task\ success})}
~~~

这个指标提醒我们：单 token 价格或单次 benchmark 分数都不是最终答案。模型选择应基于任务成功、失败恢复和服务 SLO。

## 52.12 常见误区

第一，把 Chinchilla 的比例当成永恒公式。第二，把训练 token 当作同质量信息。第三，只看参数量而忽略 active 参数和通信。第四，把预训练损失直接等同于对话质量。第五，忽略 tokenizer 差异，直接比较不同模型的 token 数。第六，外推小模型 scaling curve 时没有独立验证。第七，忘记为长上下文、后训练、评估和部署留预算。

## 52.13 从损失最优到推理最优

Chinchilla 讨论的是预训练计算预算下的损失最优，产品选择还要考虑推理阶段。一个更大的模型可能减少调用次数或提高任务成功率，却增加每次请求的权重、KV 和延迟；一个较小但训练更充分的模型可能更适合高并发。

可以把单位成功成本写成：

~~~math
C_{\mathrm{success}}
=\frac{C_{\mathrm{train\ amortized}}+C_{\mathrm{inference}}
+C_{\mathrm{retry}}+C_{\mathrm{verification}}}
{\Pr(\mathrm{task\ success})}
~~~

不同业务对训练摊销、质量、延迟和重试的权重不同。compute-optimal 不是一条跨阶段的固定比例，而是一个需要绑定目标函数的资源分配问题。

## 52.14 数据质量和模型规模的替代关系

增加低质量 token 可能带来较小收益，甚至加重重复、污染和版权风险；提高数据质量则可能让同样的训练预算产生更低损失。数据质量不是一个单一标量，领域覆盖、事实可靠性、代码可执行性、语言平衡和去重都会影响不同能力。

扩展实验可以固定模型和计算，替换数据 mixture；也可以固定 token 数，增加高价值域比例。必须保留独立验证集，防止训练数据质量提高只是因为评测集泄漏。

## 52.15 PaLM 带来的系统性教训

大规模训练的瓶颈会从矩阵乘扩展到数据加载、网络、梯度同步、checkpoint、故障恢复和评估。集群规模增大后，有效计算比例可以写成：

~~~math
\eta_{\mathrm{cluster}}
=\frac{C_{\mathrm{useful}}}
{C_{\mathrm{compute}}+C_{\mathrm{communication}}+C_{\mathrm{idle}}}
~~~

模型规模只有在有效利用率、数据质量和可恢复性一起改善时才有意义。PaLM 的历史价值因此不仅是参数数字，而是把模型、数据和训练系统作为同一个扩展问题来处理。

## 52.16 Compute-optimal 不是一个永恒比例

训练 compute-optimal 的结论依赖目标 loss、模型族、数据质量、优化器、硬件和训练 regime。增加数据 token 与增加参数的边际收益会随规模、数据重复和任务变化；长上下文、代码、数学和多模态任务还会改变 token 的有效价值。

因此阅读 scaling law 时要看拟合范围、数据去重、验证集、compute 定义和外推误差。一个在预训练 loss 上最优的配比，不一定在 instruction following、代码、事实性或推理任务上最优。

## 52.17 训练最优与服务最优不同

训练时更大模型可能在固定 token 预算下更划算，服务时却因为显存、并发、p99 和成本不合适。推理模型还会把一部分计算转移到 test time；需要比较模型参数、激活、reasoning token、工具时间和单位成功成本。

可以把系统目标写成：

```math
J=\lambda_q Q-\lambda_c C-\lambda_t T-\lambda_r R
```

权重随业务变化。低风险批处理可以更看重成本，高风险工具动作要把风险作为硬约束。只用 pretraining loss 决定部署模型，会漏掉服务阶段的真实瓶颈。

## 52.18 数据质量和重复是隐藏变量

更多 token 如果重复、污染或质量低，可能增加训练成本却不增加有效信息；高质量代码、数学、领域和多模态数据可能比同量普通网页贡献更大。训练报告应记录 dedup、mixture、数据来源和污染检查，不能只报告总 token。

## 52.19 资料范围与证据边界

Kaplan 等人的 scaling law、Chinchilla 的 compute-optimal 研究和 PaLM 的大规模语言模型报告分别提供了趋势、配比和系统规模的原始证据。它们的数字来自特定实验范围，阅读时应关注数据、模型和计算条件。

Compute-optimal training 的核心不是背一个 token/parameter 数，而是学会在预算约束下同时优化模型容量、有效数据、训练稳定性、后训练和部署成本。

## 52.20 面试问题与练习

**问：为什么更大的模型可能不如更小但训练充分的模型？**

因为在固定计算预算下，大模型如果 token 不足会欠训练；较小模型可以把预算用于更多有效数据，可能获得更低损失和更好的下游能力。

**问：MoE 的 scaling 应该看 total 还是 active parameters？**

计算和每 token 激活更接近 active parameters，但权重存储、通信、路由和容量规划受 total parameters 影响，必须同时报告。

**练习：**给定三个候选模型，分别列出参数、训练 token、active 参数、训练 FLOPs、推理显存和单位成功成本，设计一个不只看 benchmark 分数的选择表。

资料入口：

- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- PaLM: https://arxiv.org/abs/2204.02311
- Gopher: https://arxiv.org/abs/2112.11446

## 52.21 一个训练预算的 worked example

假设固定训练预算 `C`，候选方案 A 使用更多参数但较少 token，方案 B 使用较少参数但更多有效 token。先用教学模型估算训练量，再把数据质量、去重、污染、后训练和部署成本分开。A 可能在训练 loss 初期下降更快，却因 token 不足欠训练；B 可能在相同计算下获得更低 loss，但推理权重和延迟不一定符合产品预算。

比较时至少记录参数、有效训练 token、active parameters、训练 FLOPs、权重显存、KV/cache、下游质量、p95 和单位成功成本。MoE 还要把 total parameters 和 expert communication 加进账本。所谓 compute-optimal 只是某一目标和约束下的最优，不是所有产品的永恒参数/token 比。

## 52.22 把 compute-optimal 变成预算决策

给定训练预算时，先把模型参数、有效 token、训练 FLOPs、数据质量和后训练成本分别列账。一个教学化估算可以写成：

```math
C_{\mathrm{total}}
=C_{\mathrm{train}}(N,D)
 +C_{\mathrm{post\text{-}train}}
 +C_{\mathrm{eval}}
 +C_{\mathrm{serve}}.
```

在固定预训练预算下，增加参数可能减少欠拟合，但也可能减少可见 token；在产品预算下，增加参数可能提高质量，却增加权重显存、KV、p99 和重试成本。所谓最优必须绑定目标函数、数据分布和时间范围，不能把某篇论文的比例直接当成新项目配置。

## 52.23 一个反事实训练计划

假设方案 A 是 13B/1T token，方案 B 是 7B/2T token，二者预训练计算近似相当。A 可能在容量充足的任务上有优势，B 可能因为训练更充分而在验证 loss 和短任务上更好；如果 B 的 tokenizer 对代码更低效，代码有效 token 还要重新计算。最终选择应加入后训练、量化、服务并发和单位成功成本，而不是只看预训练 loss。

实验上可以先用小规模模型拟合数据/参数趋势，再在目标规模做少量验证；同时保留数据质量和去重的对照。小模型趋势不能保证大模型外推，尤其是推理、长上下文和多模态任务。

## 52.24 从损失模型推导参数与数据的趋势

前面的损失式可以进一步说明，为什么“每个参数固定配多少 token”不是一个普适定律。把训练计算约束写成 `C=kND`，于是 `D=C/(kN)`。代入：

~~~math
L(N,D)-L_\infty
=AN^{-\alpha}+BD^{-\beta}
=AN^{-\alpha}+B\left(\frac{kN}{C}\right)^\beta.
~~~

对 `N` 求导并令其为零：

~~~math
-\alpha A N^{-\alpha-1}
 +\beta B\left(\frac{k}{C}\right)^\beta N^{\beta-1}=0.
~~~

因此在这个教学模型下：

~~~math
N^*\propto C^{\frac{\beta}{\alpha+\beta}},
\qquad
D^*\propto C^{\frac{\alpha}{\alpha+\beta}}.
~~~

只有当 `alpha` 和 `beta` 以及数据、模型族都处在相近的实验条件时，才可能得到看起来稳定的参数/token 比。换数据 mixture、换 tokenizer、换训练目标、加入长上下文或把目标从预训练 loss 改为代码修复，都会改变拟合系数。这个推导的价值不是让人手算最终规模，而是让人知道“比例”来自目标函数和实验范围，而不是自然常数。

## 52.25 一个带数字的预算反事实

取一个归一化预算 `C/k=10^6`，并假设 `alpha=beta=0.5`、`A=B=1`。损失中的两项分别为 `N^-0.5` 和 `D^-0.5`，在 `ND=10^6` 下，教学模型的平衡点是 `N=1000,D=1000`。候选 A 使用 `N=2000,D=500`，候选 B 使用 `N=500,D=2000`；两者计算预算相同，但 A 的数据项更大，B 的模型项更大，平衡点通常更优。

如果数据清洗让数据误差系数从 `B=1` 降到 `B=0.5`，相同预算下最优点会向更大的模型和更少的 token 移动。对 `alpha=beta` 的这个简化例子，比例近似满足：

~~~math
\frac{N^*}{D^*}\approx\frac{A}{B}.
~~~

高质量数据使 `B` 变小，并不意味着“数据不重要”，而是表示每个 token 的边际收益变高，预算分配需要重新平衡。真实项目不应把这个数字例子当配置建议；它只展示了数据质量、模型容量和计算预算之间的反事实关系。

## 52.26 有效 token 需要经过质量、重复和 tokenizer 校正

训练报告里的 `D` 通常是 tokenizer 之后的 token 数，但不同 tokenizer 对同一段代码、中文、数学公式和多模态离散码的切分不同。直接比较两个项目的 token 数，可能把“token 更碎”误读成“看到了更多信息”。更有用的账本至少包含：原始字节或文档数、token 数、去重后 token 数、领域比例、重复率、污染率和验证集覆盖。

可以用一个教学化的有效数据量表示质量折扣：

~~~math
D_{\mathrm{effective}}
=\sum_{i=1}^{D}q_i(1-r_i)(1-p_i),
~~~

其中 `q_i` 表示任务相关性和可靠性，`r_i` 表示重复影响，`p_i` 表示污染或不可用风险。它不是可直接测得的真实定律，却提醒我们：十亿个重复 token 不等于十亿个独立信息单位。评估集去重和污染检查尤其重要，否则 scaling curve 可能只是记忆曲线。

## 52.27 预训练最优不等于全生命周期最优

预训练阶段的目标可能是最小化验证 loss，但一个上线模型还要经过后训练、对齐、评估、量化、部署和长期服务。更大模型可能减少单次回答的错误和重试，也可能增加权重显存、KV cache、通信、p99 和每次请求成本；更小模型可能需要更多 reasoning token、工具调用或 verifier。

因此决策表应把阶段分开：

~~~math
C_{\mathrm{lifecycle}}
=C_{\mathrm{data}}+C_{\mathrm{pretrain}}
 +C_{\mathrm{posttrain}}+C_{\mathrm{eval}}
 +C_{\mathrm{serve}}+C_{\mathrm{failure}}.
~~~

如果目标是离线批处理，训练和吞吐可能占主导；如果目标是交互式 Agent，工具时间、失败恢复和单位成功成本更重要；如果目标是边缘设备，模型权重、激活和峰值功耗可能比预训练 loss 更硬。所谓 compute-optimal 必须明确“对哪个阶段、哪个目标、哪个约束最优”。

## 52.28 Scaling law 实验如何避免自我欺骗

一组曲线看起来平滑，不代表它能外推到目标模型。实验至少要保留三类对照：不同模型规模、不同 token 预算和不同数据 mixture；同时固定 tokenizer、优化器、训练步数定义、验证集和随机种子范围。拟合后要在没有参与拟合的规模上做预测，再检查误差是否随长度、领域或模型族系统性变化。

报告结果时不要只给一条最优曲线，还要给残差和不确定性。可以记录：

~~~math
e_j=L_{\mathrm{observed},j}-L_{\mathrm{predicted},j},
\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_j e_j^2}.
~~~

如果通用验证集上的 RMSE 很小，但代码、数学或长上下文子集的误差很大，平均 loss 正在掩盖目标能力。加入后训练后，最好重新测一次，因为 SFT、偏好优化和 reasoning RL 会改变 loss 与产品质量之间的映射。

## 52.29 小结

Compute-optimal 的核心是把参数、有效数据和训练计算放进同一个预算模型，同时承认数据质量、重复、污染、后训练和部署成本会改变实际最优点。论文给出的比例属于特定实验条件，工程选型必须用自己的 workload、成本和质量验收条件复测。
