# 第 55 章 Mistral/Mixtral：GQA、Sliding Window Attention 与稀疏 MoE

## 55.1 两个固定版本提出的架构问题

Mistral 与 Mixtral 常被连在一起讲，但它们展示的是两种不同的资源取舍。本章以 2023 年发表论文的 Mistral 7B v0.1，以及 2023 年 12 月发布权重、2024 年 1 月发表论文的 Mixtral 8x7B 为对象：前者把 grouped-query attention（GQA）与 sliding-window attention（SWA）组合进一个稠密 decoder；后者保留 Mistral 7B 主干，在前馈层使用稀疏专家。结论只适用于这两个论文版本，不能由系列名推断后续模型配置。

[Mistral 7B 论文](https://arxiv.org/abs/2310.06825)报告，基础版在其测试的各项基准中超过 Llama 2 13B，并在推理、数学和代码任务上超过 LLaMA 1 34B；指令版在论文报告的人类和自动评估中超过 Llama 2 13B Chat。[Mixtral 论文](https://arxiv.org/abs/2401.04088)的摘要概括其在所测基准上与 Llama 2 70B 和 GPT-3.5 持平或更好，正文则将总体结果描述为多数指标持平或更高。各模型和论文评测条件不同，这些结果不构成统一排名。

## 55.2 GQA 与滑窗分别改变什么

GQA 改变注意力头之间的 K/V 共享方式：多个 query heads 使用较少的 key/value heads，从而减少每个历史 token 需要保存和读取的 K/V 状态。query heads 的数量并未因此直接减少，所以它不是“缩小整个 attention 计算”的同义词；质量和速度也要按模型与负载检验。机制和 cache 账本分别见[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md)、[第十一章](11-kv-cache长上下文推理和显存增长问题.md)及[第十七章](17-LLaMA-GPT-Mistral架构对比.md)。

SWA 改变每层 attention 可直接读取的历史范围。Mistral 7B v0.1 采用宽度为 $4{,}096$ token 的因果滑窗；单层只读取窗口内的历史，但多层表示可以逐层传递更远信息。论文中的理论传播范围、论文表的上下文字段与模型对远距离证据的实际利用不是同一个量。固定版本字段的逐项差异见[第十七章](17-LLaMA-GPT-Mistral架构对比.md)，滑窗可见性与跨层传播见[第十九章](19-Sliding-Window与稀疏注意力.md)。

| 设计 | 直接改变的对象 | 主要影响 | 不能单独推出 |
| --- | --- | --- | --- |
| GQA | query heads 与 K/V heads 的共享关系 | KV 状态大小和读取量 | 所有阶段的 attention 计算都会按同一比例下降 |
| SWA | 单层可见的历史 token 范围 | 局部 attention 连接与窗口内 KV | 模型能可靠使用与层数乘窗口相等的上下文 |

两者可以出现在同一模型里，但解决的不是同一个瓶颈：GQA 改变每层如何表示和缓存 K/V；SWA 改变当前位置能直接看到哪些历史。局部窗口以外的信息是否能完成精确引用，还需要对应任务的评测，而不能仅从理论传播路径断定。

## 55.3 Mixtral 8x7B：在前馈层使用稀疏专家

Mixtral 论文将其描述为与 Mistral 7B 同一主干的 sparse mixture-of-experts 模型，差异在于每层前馈模块由八个专家组成。路由器对每个 token 在每一层选择两个专家，并组合它们的输出；不同 token 可以选到不同专家。因此稀疏路由发生在 FFN 位置，不能与 GQA 的 K/V 共享或 SWA 的 attention 可见范围混为一种“稀疏”。

论文将 Mixtral 概括为约 $47\mathrm{B}$ 总参数、每 token 激活约 $13\mathrm{B}$ 参数，并报告其训练上下文为 $32\mathrm{k}$ token；[官方发布页](https://mistral.ai/news/mixtral-of-experts/)给出的总参数和每 token 激活参数分别为 $46.7\mathrm{B}$ 与 $12.9\mathrm{B}$。模型总权重与每 token 的前馈计算是两本不同的账。激活参数较少不代表只需保存激活专家的权重，也不代表路由、分布式通信或负载波动没有成本。MoE 的路由路径与 Mixtral 在 MoE 谱系中的位置见[第二十一章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)，权重、通信和服务代价见[第二十二章](22-Dense与MoE训练服务trade-off.md)。

## 55.4 论文结果如何与结构证据对应

Mistral 7B 论文把 GQA 与 SWA 作为模型设计的一部分，并用论文自己的评估流程重新运行若干 LLaMA 基准；Mixtral 论文则研究在 Mistral 主干上把稠密 FFN 换成 top-2 的八专家层。阅读这些结果时，应将结构事实、作者在指定评测中的表现和对其他版本的推断分开：

| 论文对象 | 公开结构或比较 | 证据范围 |
| --- | --- | --- |
| Mistral 7B v0.1 | GQA 与 SWA 的稠密 decoder；基础版和指令版分别参加任务评测 | 只支持论文模型、论文基准和比较配置；不能据此断言所有 7B 模型或后续 Mistral 版本都有同样优势 |
| Mixtral 8x7B | Mistral 主干 + 每层八个 FFN 专家 + 每 token 路由两个专家 | 论文摘要给出总体比较，正文表格呈现逐项差异；不能据此把总参数与激活参数混成一个规模指标，也不能推导其他 MoE 的服务表现 |

论文报告的 7B、13B、34B、70B 等比较还受到训练数据、指令版本、提示方式、解码设置和评估集影响。Mixtral 的主表中也有单项分数低于对照模型的情况，因此摘要的总体描述不能改写为“每项均胜出”。一个论文在某组任务上胜出，能支持该实验条件下的表现结论；如果要判断 GQA、SWA 或 MoE 中某个部件的独立因果贡献，需要匹配其他条件的消融，而不能把整模型成绩直接归因给单个模块。

## 55.5 把“效率”落回具体成本路径

Mistral 7B 与 Mixtral 8x7B 展示了不同的架构杠杆：GQA 减少 K/V 状态规模，SWA 限制单层直接读取的历史范围，MoE 将总专家容量与每 token 激活计算部分分开。它们的收益与代价应分别观察，不能汇总成一个通用的稀疏率或“更省资源”结论。

选择某个设计时，先明确任务是否受 KV 读写、局部历史窗口、参数容量或专家路由影响，再核对所用 checkpoint 的版本、license 和模型卡。Mistral 7B v0.1 与 Mixtral 权重按 Apache 2.0 发布；这说明已发布权重的许可，不代表完整训练数据与 recipe 都已公开。Mixtral 发布资料只概述了预训练数据来自开放 Web，数据规模、构成和预处理细节未公开（可参阅 [Hugging Face 技术说明](https://huggingface.co/blog/mixtral)）。需要更细的结构字段时回看第十七章；需要解释局部路径和 MoE 成本时分别回看第十九、二十一、二十二章。Mistral 后续模型有各自配置，本册第七十九章另以指定的 Mistral Small 4 版本为案例，不能从 Mistral 7B v0.1 推断。
