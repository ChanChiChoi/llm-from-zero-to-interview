# 第五册第五章：预训练目标与训练循环研究笔记（2026-10-26）

## 本轮研究问题

1. “预训练目标”是否应只解释 decoder-only causal next-token，还是要比较仍具代表性的目标路线？
2. T5 span corruption、UL2 Mixture-of-Denoisers 和 Fill-in-the-Middle（FIM）各改变了什么训练信号，论文证据分别支持到哪里？
3. 变长 micro-batch、label shift 和 DDP rank 数不同后，何时简单 `loss / accumulation_steps` 不再对应全局 token mean？
4. 当前 toy 训练步是否用可核对的数据展示了正确与错误的梯度累积口径？
5. 章节原有 label shift、mask、packing、loss、validation、精度、日志、checkpoint、故障排查和练习去向是否完整保留？

## 资料阅读与证据对照

| 来源 | 实际核对内容 | 对正文支持的判断 | 证据边界 |
| --- | --- | --- | --- |
| [T5 原论文](https://arxiv.org/abs/1910.10683)，§3.3、表 3/4/7 | 目标输入/目标样本、15% corruption、连续 span 与 sentinel、平均 span 长度消融 | span corruption 让模型从未破坏上下文重建被移除片段；目标比重建整篇原文短。平均 span 长度 3 在作者的多数非翻译任务上相对 i.i.d. corruption 有小幅且统计显著的优势 | T5/C4 和论文中的模型、任务、训练预算；不代表每类模型、语种或规模都胜过 causal LM |
| [UL2 原论文](https://arxiv.org/abs/2205.05131)，§3.1、§4.2—4.5、表 1/3/4 | R/S/X denoiser 定义、7 种任务设置、C4 训练配置、与 T5/GPT-like 的聚合比较和模式提示消融 | R 近似 T5 短 span corruption；S 是 prefix-LM 顺序预测；X 是高破坏率/长 span 恢复。论文用混合训练目标把不同输入—目标形态放在一个模型中，并区分 objective 和 architecture | 主表基础模型约 167M decoder / 335M encoder-decoder、约 32B tokens、C4、64–128 TPUv4。表中 +43.6%（相对 T5 ED）和 +76.1%（相对 GPT-like decoder）是跨任务归一化聚合收益，不是单一 benchmark 准确率；不可直接外推到 frontier 训练 |
| [FIM 原论文](https://arxiv.org/abs/2207.14255)，§3、§4.2—4.5、附录 | PSM/SPM 变换、三段损失、文档/上下文级变换、FIM rate、left-to-right 与 infilling 评测 | FIM 重排部分训练样本，仍沿自回归序列计算 loss；训练时保留 prefix/suffix/middle 学习信号。6 个大模型的 50B-token rate 消融中，最高 90% FIM 没有明显损害 left-to-right 能力，100% 有自回归 loss 退化；FIM pass rate 比 FIM PPL 更能显示不同 rate 的收益 | 特定 OpenAI 代码/自然语言语料、模型和评测；FIM pass rate 受任务、采样温度和 verifier 影响，不是通用质量数字 |
| [T5 Google Research repository](https://github.com/google-research/text-to-text-transfer-transformer) 与 [SeqIO](https://github.com/google/seqio) | 官方 T5 研究代码和 sequence task / preprocessing 组件入口 | 用于从论文结论继续核对目标构造和任务数据管线，而不是只读摘要 | 本轮未 clone、运行或逐行审计仓库；分支可变，不能声称复现论文 |
| [Hugging Face Accelerate v1.15.0 accumulation guide](https://huggingface.co/docs/accelerate/v1.15.0/usage_guides/gradient_accumulation) | accumulation、变量样本 token-level loss、跨进程 token 总数和 DDP 缩放示例 | token-level causal loss 应按 accumulation window 内所有非 padding 有效目标总数归一化；框架处理 accumulation 和 DDP 时会有额外分母因子 | 官方指南的代码属于该版本/框架 API；不能直接照搬到自定义 loss 或其他 backend |
| [Transformers gradient accumulation guide](https://huggingface.co/docs/transformers/grad_accumulation) | `num_items_in_batch`、causal LM 内部 shift 后的预测目标计数、自定义 loss 的缩放责任 | 损失分母必须与真实 shift 后目标集合一致；`labels` 张量中可见的非 `-100` 数不一定就是模型实际预测目标数 | 当前官方文档会随版本变化，使用时须锁定安装版本；本轮未运行 Trainer |
| [Accelerate autoregressive example](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/gradient_accumulation_for_autoregressive_models.py) | 固定 global token denominator、`labels[...,1:]`、all-rank gather、`num_items_in_batch`、`world_size × accumulation_steps` 调整 | 说明“先定全局有效目标总数，再按框架内部分母约定反传”是可实现的工程路线 | 阅读的是 GitHub `main` 页面，不是固定 commit；本轮未运行脚本；其 validation loss 聚合实现不作为本章 valid-token weighted validation 的依据 |
| [PyTorch DDP notes](https://docs.pytorch.org/docs/stable/notes/ddp.html) | DDP reducer 对各 rank 梯度 all-reduce 并取 mean 的机制说明 | 每 rank 先各自做 token mean 再平均梯度，会在每 rank 有效 token 数不同的情况下让短 rank 权重偏大 | 这里只用于解释梯度缩放；实际扩展还涉及并行组、loss API、空 rank 和通信同步 |

## 代表性方法解决的问题

| 路线 | 输入可见范围 | 被监督的目标 | 适用问题 | 主要限制/评价要求 |
| --- | --- | --- | --- | --- |
| Causal LM | 左侧前缀 | 每个有效位置的下一个 token | 开放式续写、统一生成接口、few-shot prompting | 无法直接条件化于目标右侧文本；通常看 token NLL/PPL，并补任务评测 |
| BERT-style MLM | 遮挡位置周围左右文 | 选中位置的原 token | 双向编码表征、分类/抽取 | 稀疏监督且 corruption 与生成时分布不同；不能直接把 MLM loss 当作自回归序列似然 |
| T5 span corruption | 所有未遮挡 token，可包含目标左右两侧证据 | 被移除的 span 与 sentinel | 条件补全/文本变换；短 target 降低 decoder 预测长度 | 通常依赖 encoder-decoder；目标重构口径不同，跨目标不能裸比 PPL |
| UL2 MoD | 由 R/S/X 任务及模式决定 | 短 span、长后缀或大量缺失片段 | 单一模型混合不同 denoising 和生成任务 | recipe、任务配比、mode prompt 复杂；证据来自受限规模和实验任务 |
| FIM | 重排后的 token 前缀；序列中保留 suffix 作为 middle 的条件 | 重排序列的自回归下一个 token | code/text 中间补全 | 特殊 token、顺序、结束标记和 context packing 决定样本质量；需采样通过率等非 PPL 指标 |

这些方案的演进不是简单替换：T5 用 span-level target 缩短 decoder 侧重建序列；UL2 将多个监督形态组合，针对单目标覆盖不足；FIM 则保留 causal decoder 和 next-token loss，靠数据重排补 middle infilling。目标选择应由输入可见范围、输出需求、模型架构、预算和目标任务评估共同决定。正文没有把任何一篇论文的榜单结果写成当前普适结论。

## 旧稿知识点覆盖图

| 旧稿位置/知识点 | 处理方式 | 当前去向 |
| --- | --- | --- |
| 5.1 自回归概率分解、teacher forcing、causal mask | 保留并深化到训练目标的上下文约束 | `05-预训练目标与训练循环.md` §5.1.1—5.1.3 |
| causal LM 之外的主要预训练目标 | 原文以 causal LM 为主，补入核心对照并保留 MLM 的历史/编码位置 | §5.1.4：Causal LM、MLM、T5、UL2、FIM 方法表、具体例子、实验和选择边界 |
| label shift、EOS、边界和 shape | 保留 | §5.2；同时提醒内部 shift 与外部 shift 只能择一 |
| cross entropy、有效 label、`ignore_index`、PPL | 保留，强调分母语义 | §5.3 |
| causal/padding/label mask | 保留 | §5.4 |
| packing、EOS 不自动隔离、利用率和截断 | 保留；连接到 context-level FIM 的边界问题 | §5.5、§5.1.4 |
| optimizer step、clipping、FP16/BF16、loss scaling | 保留 | §5.6、§5.8 |
| equal-token accumulation 与 variable-token normalization | 保留并补足按 token 权重推导、DDP world-size 关系和 framework scaling 风险 | §5.7.1—5.7.3 |
| validation token-weighted aggregation | 保留并修正伪代码计数必须遵守外部/内部 shift | §5.9 |
| logs、checkpoint state、resume、checkpoint selection | 保留 | §5.10—5.12 |
| token-to-logits code、异常输入防护、一次 SGD 更新 | 保留；累积切分改成 2+4 有效 token，并显式并排显示正确加权与错误等权 | §5.13 |
| 故障排查、7 项练习和小结 | 保留；新增目标选型练习，并更新小结口径 | §5.14—5.17 |

本轮没有删除原有 causal LM 训练链。增加的目标族用于回答“为什么当前常见 decoder-only 训练之外还存在这些目标”，没有把 encoder-only 或 encoder-decoder 细节扩展成架构主讲章节。

## 手算与代码口径

Toy stream 在 `block_size=5` 时形成两个 block：`[the, cat, sat, eos, the]` 与 `[dog, ran, eos, pad, pad]`。外部 label shift 后得到 6 个有效 pair：

```text
(the,cat), (cat,sat), (sat,eos), (eos,the), (dog,ran), (ran,eos)
```

设每个有效 pair 的未平均梯度为 `u_i`，完整 batch 的 mean gradient 为 `sum_i u_i / 6`。分成 2-token 与 4-token micro-batch 后，令其各自 mean gradient 为 `g_1` 和 `g_2`：

```text
正确 token 权重：2/6 * g_1 + 4/6 * g_2
错误 batch 等权：1/2 * g_1 + 1/2 * g_2
```

该推导由当前示例的 pair 数量和 loss reduction 直接得到。代码诊断应显示 weighted gradient 和 full batch gradient 数值一致，而等权方案不一致；代码本轮只做手算与静态逻辑审阅，依当前执行要求未运行 Python。

## 本轮变化与待办

- 增加目标函数问题地图和 T5/UL2/FIM 的方案对照、具体构造、代表性结果、方法限制与评估选择。
- 明确 FIM 是 token 重排后的 causal loss，不是双向 encoder；补齐标记 token、packing 层级和 pass-rate 评测边界。
- 在 variable-token accumulation 中加入梯度按有效 token 加权的推导、2+4 token 实例和 DDP world-size 简化公式；提醒不要与框架内置 scaling 重复。
- 修改 validation 伪代码的目标数函数名，显式要求按模型实际 label shift 计算 denominator。
- 修改 toy 累积切分为 2+4；同时计算正确 token-weighted gradient 和故意错误的等 micro-batch weighting，新增对应差异诊断。
- `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md` 补充本章相关来源、术语和面试追问。
- 静态检查标题、公式与 code fence 配对、练习编号、research note coverage map、链接目标和 `git diff --check`；不运行 Python 或训练代码。

## 风险与边界

- T5/UL2 的 benchmark 聚合值依论文的 C4、模型结构/规模、任务和计算预算；不是当前 decoder-only frontier 模型的直接比较。
- FIM 的 rate、PSM/SPM、document/context transform 和 infilling pass rate 是一套相互影响的设计，不能由单一 token loss 外推完整 IDE 体验。
- DDP 的 `P/N` 公式只适用于明示的 DDP mean-gradient 和无框架额外 loss scaling 假设。Trainer、Accelerate、DeepSpeed、FSDP、pipeline 或 custom reducer 的实现需逐版本检查实际 label shift、token count、reduction 和 accumulation scaling。
- 本章 toy 的有效梯度关系经过手算与静态核对，尚未 runtime 执行；assert 输出尚未观测。真实 Transformer、混合精度、分布式通信和 checkpoint 恢复都不由 toy 验证。
- 本轮没有做论文独立复现、真实预训练、外部公式渲染或 PDF 构建。
