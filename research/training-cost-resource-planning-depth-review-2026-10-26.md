# 第五册第十三章：训练成本与资源规划研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/13-训练成本与资源规划.md`](../book-05-llm-training/chapters/13-训练成本与资源规划.md)  
依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)、`progress_v2.md`

## 核心问题图谱

本章旧稿已覆盖 `6ND`、长序列项、MoE 激活参数、吞吐、GPU 数倒推、MFU/HFU、显存、数据 token、checkpoint、故障重跑、资源规划工作流和成本审计 demo。复读后保留这条计算链，针对代表性研究较弱的部分增加实证对照：

1. **固定预训练 FLOPs 下“最优模型”是什么**：Kaplan 与 Chinchilla 在训练规模分配上结论不同，差异来自拟合方法、模型/数据范围和目标定义；二者都不能简单变成所有任务的固定 tokens-per-parameter 规则。
2. **训练成本是否应该包含部署需求**：传统 compute-optimal scaling 优化预训练预算下的损失；inference-aware work 把未来请求量、serving cost 纳入目标，可能改变模型大小和训练 token 的选择。
3. **峰值算力、模型有效计算和长期训练效率如何连接**：PaLM 定义 MFU/HFU，MegaScale 展示大规模强扩展、通信重叠、诊断和故障恢复，说明单 step 算力和数周任务可用率属于不同证据层。
4. **训练成本估计还漏了什么**：公开 Llama 2/OLMo 研究中的能源/碳核算说明硬件功率、PUE、电网碳强度、冷却和 embodied carbon 的统计边界不同；GPU-hours、美元成本和碳排不能互相替代。

## 主要来源与结论边界

| 来源 | 解决的问题 / 方法 | 可写入正文的证据 | 限制与解释边界 |
| --- | --- | --- | --- |
| [Kaplan et al., Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 对模型规模、数据规模、训练计算与 loss 进行经验幂律拟合 | Chinchilla 论文引用其固定计算预算下参数和 token 分配估计，用作对照 | 它的实验范围、拟合目标和数据条件决定最优分配；不能脱离后续不同拟合与实验直接当作物理规律 |
| [Hoffmann et al., Chinchilla](https://arxiv.org/abs/2203.15556) | 使用 IsoFLOP 等方法重估给定计算预算的模型/数据分配，并训练验证模型 | 论文报告分析了 400 多个 70M 到超过 16B 的模型、5B—500B tokens；Gopher 同预算下训练 Chinchilla 70B/1.4T tokens，并在多项任务胜过 280B/300B-token Gopher | 关键数字是论文自身的架构、语料、optimizer、tokenizer 和目标；作者也指出极远尺度外推不确定。训练 compute-optimal 不等于包含未来 serving、SFT/RL 或多模态阶段的全生命周期最优 |
| [Sardana et al., Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) | 将推理请求量与部署成本加入训练参数/数据分配 | 作者称在约 1B 请求假设下，更小、更长训练可能降低全生命周期 cost，并训练 47 个不同规模模型验证部分拟合；覆盖最高 10,000 tokens/parameter 的研究范围 | “请求量”是目标情景变量，不是所有模型都会遇到的常数；模型、质量目标、成本单价和推理长度设定变化会改变最优点；极端 token ratio 结论受论文模型规模与语料范围限定 |
| [PaLM](https://arxiv.org/abs/2204.02311) | 提出更便于跨实现理解的 MFU，区分目标模型所需算术与实际硬件执行算术 | PaLM 540B 在 6,144 TPU v4 报告 MFU 46.2%、HFU 57.8%；MFU 不把 activation rematerialization 当有效模型进展 | 数字依赖论文架构、精度、计数口径和 TPU 设置；MFU/HFU 不是质量指标，不能替代 wall-clock、tokens/s、能源或训练收敛对照 |
| [MegaScale](https://arxiv.org/abs/2402.15627) | 以算法/系统协同优化解决 10K+ GPU 的通信、straggler、观测性和长程故障稳定性 | 其 175B、序列长 2,048、固定 batch=6,144 的强扩展表在 3,072→12,288 GPU 区间报告 MFU 59.1%→55.2%，基线 Megatron-LM 48.7%→41.2%；同论文生产任务报告数周内 >100 次 restart，>90% software/hardware faults 自动识别恢复 | 性能表是论文模型、集群与并行配置的系统比较，不应与 PaLM/别的硬件 MFU 直接排名；生产模型专有，公开细节不足以复现；auto recovery 与“零失败”不是一回事 |
| [Llama 2 report](https://arxiv.org/abs/2307.09288) | 公开预训练 GPU-hours 与碳排估算 | 报告约 3.3M A100-80GB GPU-hours、以设备 TDP 350/400W 估算，给出合计 539 tCO₂e | 原作者明确指出真实功耗可能低于/不同于 TDP，估算未含互联/非 GPU 服务器功耗、冷却和硬件制造碳排；是估算披露，不是计量电表读数 |
| [OLMo](https://arxiv.org/abs/2402.00838) | 公开训练、数据、评测工件，提供能源/碳核算的另一个透明案例 | 报告中 A100-40GB 路线估算 104 MWh、70 tCO₂e；电力来源强度影响结果，使用水电的 MI250X 路线估算接近零但作者注释指出另一估计因子会得到非零排放 | 此处的 7B 训练、区域电网强度、PUE 与核算边界不能外推成通用 LLM 碳因子；低 operational carbon 不代表没有 embodied carbon 或其它系统影响 |

## 对照分析：模型/数据分配

| 方案 | 优化目标 | 关键方法与发现 | 适合本章怎样使用 | 边界 / 后续修正 |
| --- | --- | --- | --- | --- |
| Kaplan scaling | 用固定训练计算提高预训练性能 | 经验幂律拟合并据此分配模型规模与训练时长；后来工作指出它对模型/数据相对比例的建议偏向更大模型 | 作为历史基线说明“scale up model”不等同“预算最优” | 论文范围、训练 setup 与拟合方法限制外推 |
| Chinchilla | 固定预训练 FLOPs 下改善训练 loss/下游结果 | 更密集的 IsoFLOP 研究后主张模型参数与数据近似同比例增长；70B/1.4T 与 280B/300B 同预算比较是案例证据 | 用来讲“模型大小和数据 token 一起规划”，并展示同预算如何做对照 | 目标是训练 compute，而非推理流量/全生命周期 TCO；可用数据质量、重复、下游目标会改变实践方案 |
| Inference-aware scaling | 以训练与给定推理需求总成本完成目标质量 | 把预计 request/serving load 放进参数与数据选择，作者在特定高请求情景发现更小、训练更久可划算 | 用于反驳“compute-optimal 就是总成本最优”，要求读者先给请求量/输出分布 | 预算需求曲线、质量门槛、硬件和服务效率未知时不能替目标项目选模型；若需求很低或模型很快淘汰，方向可能变化 |

将三者迁移到项目时，应先把优化目标写成“给定训练 FLOPs 的 loss”“给定墙钟的任务质量”或“给定寿命期请求量/延迟门槛的总成本”；再明确质量门槛、数据复用和部署负载。模型/数据的解析公式是候选方案生成器，目标验证集和真实服务压测负责选点。

## 对照分析：单步效率与长程可用率

PaLM 适合解释 MFU/HFU 的计数契约；MegaScale 适合展示 system-level bottleneck 与故障恢复。二者之间不应比较 MFU 排名。MegaScale 175B 表的同论文 Megatron-LM baseline 和 MegaScale 同硬件/批量对照，才有较强的相对解释力；该论文还观察到 GPU 数增加时固定 batch 的 MFU 下降，原因是计算/通信比例变化，而不是吞吐随 GPU 数自动线性。

用研究例建立工程账本时至少分开：

- **模型有效进展**：global valid tokens/s、loss/task quality、MFU 的模型 FLOPs 计数；
- **硬件忙碌**：HFU、kernel time、重计算与通信；
- **墙钟可用率**：有用训练迭代时间/总时间、checkpoint/validation/recovery 占用、resume loss window；
- **货币成本**：实际 GPU-hours × 对应时段/区域价格，再加存储、网络、数据及人力；
- **能源/碳**：测量或估算的 kWh × 时间/地点电网碳强度，并把 PUE、机房冷却、其它服务器功耗、embodied carbon 的纳入范围写清。

## 旧稿覆盖和正文变更

保留原有四类成本和总成本式、成本分母、`6ND` 的推导与边界、长序列与 MoE 激活计算、tokens/s、峰值与稳态差别、显存账本、GPU 数量反推、MFU/HFU、Chinchilla 的数据匹配、checkpoint 容量/带宽、故障期望损失、资源计划工作流、toy demo 与常见误判。

已落实：

1. 在 MFU/稳态吞吐内容加入 PaLM 与 MegaScale 的问题、配置、机制、报告证据和边界对照，分开单步效率与长任务可用率。
2. 在数据/token 规划处加入 Kaplan→Chinchilla→inference-aware scaling 对照，解释优化目标变化时模型规模和训练 token 选择为什么会变。
3. 在成本构成中加入电力/碳估算账本，给出量纲正确的 kWh/CO₂e 关系和 Llama 2/OLMo 核算范围，未把 TDP 估计写成实测。
4. 更新论文目录、成本面试题、术语与练习，没有重复复制其它章节的 scaling-law 通用题。

## 本地验收

- 1—15 编号章节连续；82 个 fenced marker 成对，31 个 `math` block 的花括号平衡，表格列数一致。
- 标准库 demo 执行成功，输出与正文示例逐行一致；此前验证的 5 类非法输入均触发拒绝。
- 章节内 Markdown 本地目标全部存在；第五册目录已有第十三章映射；`git diff --check` 通过。
- 未执行 GPU、目标集群、云价格、功率计、真实故障恢复、MathJax 或 PDF 构建；这些限制不作为代码或公式验收结果。

本轮不需要调用实际云价格 API；正文明确不硬编码供应商价格。没有目标集群、账号账单、实际电表、云实例和可恢复训练 job 时，不会声称实测成本或训练恢复性能。
