# 第六册第十七章：Native INT4 与原生量化部署深度精修记录

- 复核日期：2026-10-27
- 正文：book-06-llm-deployment/chapters/17-native-int4与原生量化部署.md
- 研究范围：全章 17.1—17.18；重点补强 INT4 数学、代表性 PTQ/QAT 方法、低比特执行 kernel、单位成功成本和并发账本。
- 取舍原则：保留章节原有的核心知识链，不因补充 GPTQ/AWQ/MARLIN 而移除 artifact、校准、质量评估、运行时兼容、灰度回滚、多模态和 adapter 等主题。

## 研究问题

1. 有符号 INT4 的表示范围、对称码本和量化误差如何区分，能否给出可手算例子？
2. GPTQ 的近似 Hessian 是什么，误差如何补偿到剩余权重，它与模型任务损失 Hessian 是否相同？
3. AWQ 的显著通道在部署时怎样处理，是否需要保留 1% FP16 权重？
4. GPTQ、AWQ、QAT 和 MARLIN 分别在哪一层解决问题，各自证据和限制是什么？
5. 如何不把量化质量、artifact 兼容、kernel 性能和端到端成本混成一个未经验证的“INT4 收益”？
6. 单位成功任务成本和并发内存估算的分子、分母是否成立？

## 阅读来源与证据范围

| 来源 | 实际阅读内容 | 支持的正文判断 | 证据边界 |
| --- | --- | --- | --- |
| [A Survey on Model Compression for Large Language Models, TACL 2024](https://aclanthology.org/2024.tacl-1.85/) | 量化分类、weight-only / weight-activation / KV-cache 分层、指标和综述表 | 用作方法地图；综述表说明模型规模、bit width、任务和速度口径需一起读 | 不是算法原始证据。其 AWQ 段把方法概括成保留 top-1% 高精度，与 AWQ 原论文主方法表述不同；以原论文为机制依据 |
| [GPTQ 原论文，arXiv:2210.17323](https://arxiv.org/html/2210.17323v2) | 算法 §3—§4、实验设置、OPT/BLOOM 表格、速度与 limitations | 层输出重建目标、H≈2XXᵀ+λI、逆 Hessian 误差补偿、共享列顺序、分块更新与 Cholesky 稳定化；128 段×2048 token C4 校准、单张 80GB A100、OPT-175B 约 4.2 小时；OPT-175B WikiText-2 的 FP16/GPTQ-4bit/RTN-4bit PPL 为 8.34/8.37/10.54 | 论文作者配置与结果；C4 校准、语言建模 PPL 和论文 kernel 不能代表目标任务、目标 runtime 或线上 SLO。作者指出 weight-only 减少内存移动，不减少乘加计算 |
| [AWQ 原论文，arXiv:2306.00978](https://arxiv.org/html/2306.00978v6) | 显著通道、等价缩放、s_X^α scale search、clipping、数据效率/分布偏移、TinyChat 与实验表 | XW=(XD_s⁻¹)(D_sW)；用激活通道幅值确定候选 scale，搜索 α∈[0,1] 并可优化 clipping；通常把反向缩放折入前置算子；部署主方法保持全低比特权重，不必保留显著权重 FP16。OPT-6.7B INT3-g128 WikiText-2 PPL：FP16 10.86、RTN 23.54、AWQ 11.39；作者 PubMed/Enron 互换实验中 AWQ 增量 0.5–0.6、GPTQ 2.3–4.9 | 质量和 calibration mismatch 数字绑定论文模型、bit/group、样本与评测。TinyChat 速度包括 fused dequant、平台相关 pack 和 kernel fusion，不能都归因于 AWQ 算法 |
| [LLM-QAT, Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.26/) | data-free distillation、量化训练和 zero-shot 表格 | 从教师模型采样生成文本，用教师 logits 作为软标签训练量化学生；论文涉及 weights、activations、KV cache 同时量化。LLaMA-7B W4A8KV4 下常识任务平均分从 RTN 41.2 到 LLM-QAT 60.7 | 绑定 LLaMA、教师生成数据、任务与训练配置；不能证明任意模型适合 QAT，也不能证明目标 serving kernel 支持相同布局 |
| [MARLIN, arXiv:2408.11743](https://arxiv.org/html/2408.11743) | kernel 动机、异步访存/调度/流水线、层级和 vLLM 端到端结果 | 解决 batched decode 中 W4A16 的权重读取、kernel scheduling 和 compute pipeline；A10 大矩阵层级 batch 16–32 约 3.9×、batch 128 约 1.5×；全模型 vLLM 收益更低，额外 attention/serving 开销可见 | 绑定 A10、形状、batch、频率和实现。MARLIN 不提高量化质量，不能与 GPTQ/AWQ 的 PPL 数字比较 |
| [vLLM Quantization 文档](https://docs.vllm.ai/en/latest/features/quantization/) | 格式支持、linear backend 选择和按量化方案设置 backend | 量化算法/artifact 与 runtime backend 是独立选择，可有全局和 per-quantization backend 覆盖 | latest 是滚动文档；支持矩阵和硬件范围会变，不能代替锁定目标 release、模型 shape 和实际 kernel trace 验收 |
| [PyTorch FakeQuantize 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html) | fake quant forward 公式与 observer 状态 | QAT 前向模拟 quantize/dequantize；scale/zero-point 和 observer 状态要与导出 artifact 区分 | API 文档不提供某个 LLM 的完整 QAT recipe |

## 代表方案对照

| 方案 | 解决的问题 | 核心做法 | 证据 | 限制与后续验证 |
| --- | --- | --- | --- | --- |
| RTN | 形成容易实现的转换基线 | 逐项舍入到最近量化格点，不根据输入激活补偿输出误差 | GPTQ 论文中 OPT-175B 4-bit WikiText-2 PPL 10.54 | 极低 bit 时误差会累积；用作相同 grid 下的 baseline，再按质量收益和转换成本决定是否采用更复杂 PTQ |
| GPTQ | 大模型 PTQ 需低质量损失，逐权重优化又难以扩展 | 对校准激活构造局部 Hessian；逐列量化；用逆 Hessian 补偿未量化权重；分块更新并用 Cholesky 稳定数值过程 | OPT-175B 4-bit PPL 8.37，对比 FP16 8.34、RTN 10.54；单 A100 约 4.2 小时 | 校准与分组依赖；局部重建目标不等于端到端任务目标；weight-only 不减 MAC。目标域校准、任务回归和新 kernel 是不同改进路径 |
| AWQ | 激活敏感通道在低 bit 时会放大量化误差，混合精度权重又不利于高效执行 | 用 activation magnitude 做等价通道缩放，搜索 alpha 与 clipping，导出全低比特权重 | OPT-6.7B INT3-g128 PPL 11.39，优于同表 RTN 23.54；小样本和分布互换结果是作者报告 | 不代表不需要校准；scale/clipping 可能牺牲同 group 内其他权重；目标模型需按真实 workload 复测 |
| LLM-QAT | PTQ 进入更低精度、权重/激活/KV 联合量化时质量不足 | 教师自生成数据与 logits 蒸馏，让量化学生训练适应噪声 | LLaMA-7B W4A8KV4 常识平均分作者报告 41.2→60.7 | 训练成本和数据分布风险；训练结果与部署格式/内核仍是不同验收对象 |
| MARLIN | W4A16 artifact 在并发 decode 中受权重读取、解码和矩阵核效率限制 | 异步访存、分块调度、流水线、量化专用布局和混合精度执行 | A10 layer benchmark 与 vLLM 端到端表均显示 batch 增大时收益下降 | 不改量化算法；只在兼容 GPU/shape/layout/backend 下测；高 batch compute-bound 后可能考虑激活量化或其他执行策略 |

这些方案跨越量化基线、PTQ/QAT 算法、artifact/layout、runtime kernel。质量回归优先定位算法与校准，静默数值错误优先检查 artifact，性能未提升先看 kernel 和 trace。该定位流程是本项目依据来源作出的工程建议，不是这些论文共同给出的唯一发布顺序。

## 需要纠正的资料简化

TACL 综述的 AWQ 简述写成把最敏感约 1% 权重存成高精度。AWQ 原论文确实先用 FP16 保留显著权重的实验说明敏感性，随后提出等价缩放路线，目的是避免硬件效率较差的混合精度权重执行。正文因此区分“动机/对照”与“AWQ 主算法”。保留综述作为地图，但具体算法以一手来源为准。

MARLIN 的结果说明低 bit 算法结果与 kernel 结果不应混排。论文将 GPTQ 生成的 INT4 权重交给 MARLIN kernel，表明算法和执行实现可以组合；质量、artifact 与 kernel 对端到端结果的贡献仍要分开测量。

## 手算与定义修订

### INT4 码本

二进制补码可表示 [-8,7]；对称量化器常选 [-7,7] 以保持正负幅值对称，二者不是同一个“范围”概念。教学值 [-0.9,-0.2,0.3,1.1] 在 s=1.1/7≈0.15714 下编码为 [-6,-1,2,7]，重构约 [-0.9429,-0.1571,0.3143,1.1000]。例子显示舍入误差与 group outlier 影响，不把码本选择当成所有 artifact 的格式约定。

### GPTQ 局部误差补偿

对假设的 H⁻¹=[[1,0.8],[0.8,1]]，若 w₁=0.62 量化成 q₁=0.42，则 e₁=0.20。第二个坐标更新为 w₂←0.50−0.20×0.8/1=0.34。这个例子演示局部二次重建补偿，不承诺所有输入输出完全相同。

### QAT 和成本分母

QAT 的教学式前向是“量化码舍入/clip 后再反量化”，高精度副本保留为可训练参数；梯度近似取决于 recipe。单位成功成本定义为“每个原始请求全流程期望成本 / 原始请求成功率”。分子已包括 retries、verification、fallback 与 human review，不能重复计算；成功率不按 attempt 计数。

## 原章节覆盖映射

| 原章节位置 | 原知识点 | 本轮处理 |
| --- | --- | --- |
| 17.1 | native 术语、格式/文件/执行的差异、小白与专家视角 | 保留；与 GPTQ/AWQ、artifact、kernel 分层关系相互支撑 |
| 17.2 | 权重与 KV 显存、总账本、卡数与显存收益边界 | 保留原数值账本；17.10 并发内存公式更正为固定权重扣除后再除每请求增量 |
| 17.3 | 对称/非对称量化、scale、zero point、group size、pack | 保留；明确可表示 INT4 与对称码本范围差异，新增四值手算 |
| 17.4 | PTQ、GPTQ、AWQ、QAT、artifact 的区别 | 全部保留；展开 Hessian/补偿、等价缩放、fake quant/LLM-QAT，加入 RTN/GPTQ/AWQ/MARLIN 跨层比较 |
| 17.5 | manifest、artifact lineage、加载/数值/任务兼容 | 保留主体，增加算法与执行边界呼应 |
| 17.6 | 先解包路径、融合 W4A16、W4A8、kernel 覆盖与 fallback | 全部保留；MARLIN 作为执行 kernel 代表 |
| 17.7 | scale 开销、TP/PP 切分、pack ABI | 全部保留 |
| 17.8 | 校准覆盖、层敏感性、高精度保留模块 | 全部保留；补层敏感实验按指标方向解释 |
| 17.9 | 同请求基线、任务指标与分母、失败归因、单位成功成本 | 全部保留；明确 retries/review 已计入期望成本并修正数值例子 |
| 17.10 | 容量、并发、prefill/decode、最低卡数 | 全部保留；修正内存并发关系并说明需 runtime profile |
| 17.11 | KV、speculative decoding、多模态、LoRA/adapter | 全部保留 |
| 17.12 | 启动兼容检查、运行时 fallback、trace 字段 | 全部保留 |
| 17.13 | 离线 replay、灰度、回滚、版本组合 | 全部保留 |
| 17.14 | 标准库教学 demo、显存下界、任务质量回归 | 保留字段与输出；容量和质量是假设输入，不是 runtime 测量 |
| 17.15 | 企业代码助手容量/质量决策 | 保留路径和风险维度 |
| 17.16 | 常见误区、症状诊断路径 | 保留 |
| 17.17 | 来源类别、证据范围、延伸阅读 | 保留；新增 TACL、LLM-QAT、MARLIN、PyTorch FakeQuantize 入口并记录来源简化 |
| 17.18 | 本章总结 | 保留完整章节入口 |

## 配套同步

- EXERCISES.md：新增手算 INT4/GPTQ、拆分 AWQ 与 MARLIN 层次、设计目标 workload 对照和核算单位成功成本的题目。
- INTERVIEW_BANK.md：新增第六册第十七章五个深答问题，涉及 GPTQ 补偿、AWQ scale、来源冲突、算法/artifact/kernel 分层、失败归因与单位成功成本。
- PAPERS.md：新增 LLM-QAT、MARLIN 和固定 PyTorch FakeQuantize 文档入口；沿用已有 GPTQ/AWQ 与 TACL 条目。
- GLOSSARY_EN_ZH.md：将 Native INT4 改写为非标准化部署称谓并区分 weight/KV；补 MARLIN kernel 定义。
- book-06-llm-deployment/目录.md：第十七章路径已存在，无需改目录。

## 验证与开放边界

- 检查 17.1—17.18 主标题连续；新增 17.4.5 不改变章节顺序。
- 数值需核对：INT4 码值/反量化、GPTQ 二坐标更新、AWQ D_s 变换、QAT W4A8KV4 表值、成本 120/90=1.33。
- 并发公式只是线性教学估算；workspace、KV 与 batch 的真实关系需目标 runtime profiling。
- demo 的 400/115/165/450 GiB、85% usable 和质量分数是合成输入；不实现 INT4 pack、真实模型、目标 GPU kernel 或线上负载。
- 论文指标不能外推为目标业务质量和性能。vLLM latest 兼容矩阵滚动更新，部署时必须锁定 release 和 GPU。
- 没有目标 checkpoint、目标 GPU、真实请求 trace、engine profile 或 production SLO 实测，因此不声称目标环境兼容、速度收益、容量或成本下降。
- 本章最终结构、公式、围栏、演示运行与链接检查状态记录在 progress_v2.md。
