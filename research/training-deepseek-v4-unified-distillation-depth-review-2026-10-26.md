# DeepSeek V4 领域专家与统一蒸馏：深度精修研究记录

核验日期：2026-10-26

## 范围与问题地图

本轮精修对象为 [`book-05-llm-training/chapters/18-deepseek-v4领域专家与统一蒸馏.md`](../book-05-llm-training/chapters/18-deepseek-v4领域专家与统一蒸馏.md)。章节沿用原有主题：领域专家、SFT/GRPO、权限边界、多教师 on-policy distillation、教师选择、训练循环、成本/失败、方法比较、面试与练习。本轮把 V4 报告中原稿没有展开的 full-vocabulary reverse-KL 与教师调度作为重点，不把整章泛化成 OPD 入门概述。

## 官方与论文证据

| 来源 | 本轮读取并确认的内容 | 适用范围与限制 |
|---|---|---|
| [DeepSeek V4 Technical Report](https://arxiv.org/html/2606.19348)，§5.1.1 | 领域专家沿用 V3.2 后训练路线，先 fine-tuning、再用领域 prompts 和 reward signals 做 RL；RL 使用 GRPO。容易验证的任务可用规则 verifier/tests；难验证任务使用 rubric-guided RL 数据和 Generative Reward Model（GRM）。报告称 actor network 兼任 GRM，并以 RL 优化。 | 报告没有公开全部专家 checkpoint、每领域数据规模、GRM 输出/打分 schema、独立校准、完整 reward 和 GRPO loss 配置。作者对少量多样人工标注的效果表述不能外推为普遍结论。 |
| [DeepSeek V4 Technical Report](https://arxiv.org/html/2606.19348)，§5.1.2 | OPD 用于把多个领域教师的能力整合到一个学生；给出 `Σ_i w_i D_KL(πθ || π_Ei)`，说明学生在自生成 trajectories 上学习；写明超过 10 个教师、按任务上下文向相关教师学习。 | 未披露 `w_i` 的具体表、是否按状态动态变化、教师选择代码或与其它方案的受控消融；只能支持公开方法描述，不能单独归因最终 benchmark 的全部提升。 |
| [DeepSeek V4 Technical Report](https://arxiv.org/html/2606.19348)，§5.1.2 | 作者称常见 sampled-token KL 估计被接入 RL policy loss 后梯度方差高、训练不稳，故采用 full-vocabulary logit distillation 计算 reverse KL。 | 这是作者的方案动机和经验陈述；公开版本未列 sampled-vs-full-vocabulary 其它条件相同的消融，也不意味着所有 OPD sampled estimator 都不稳定。 |
| [DeepSeek V4 Technical Report](https://arxiv.org/html/2606.19348)，§5.2.2 | 教师权重放在 centralized distributed storage，教师前向时按需读取、用 ZeRO-like parameter sharding；缓存教师末层 hidden states，在训练时加载对应 prediction head 重建 logits；dispatch 按 teacher index 排序，使一个 mini-batch 只加载对应输出头且 GPU 同时最多驻留一个 teacher head；load/offload 异步；TileLang kernel 计算 exact KL。论文称框架面向“有效无上限”的教师数、单个教师可能达 trillion-parameter 规模。 | “有效无上限/可到 trillion 参数”描述基础设施支持能力，不是 V4 实际教师数或教师参数统计；报告未给出可泛化的峰值显存、存储带宽、端到端吞吐/成本或目标集群复现数据。系统设计不消除 teacher forward 和 full-vocabulary 计算。 |
| [DeepSeek V4 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | V4-Pro/Flash 是 MoE 模型；模型卡摘要描述领域专家 SFT/GRPO 后接 OPD 统一蒸馏。 | 只用来区分最终模型架构的 routed experts 与训练阶段完整领域 teacher checkpoints；不据模型卡补写内部训练实现。 |
| [GKD: On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) | 摘要指出自回归 KD 的训练/生成序列分布错位，GKD 使用学生自生成序列并接受教师反馈；报告摘要、翻译、算术推理和 instruction tuning 实验。 | 论文实验绑定其 T5-family 模型与任务；不等同于 V4 的多教师万亿参数全词表实现。 |
| [MiniLLM](https://arxiv.org/abs/2306.08543) 与第五册第十七章研究笔记 | reverse-KL 与学生 on-policy 优化；历史版本研究记录含其序列级估计、teacher mixing/length normalization 等全文机制梳理。 | 与 V4 使用 reverse KL 不等于其训练目标、信用分配和系统实现完全相同。实验规模与任务不能外推成 V4 证据。 |
| [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html) | 本轮读取 PMLR 原文 PDF 首页和方法介绍：对预训练模型 fine-tuning checkpoints 做权重平均；论文指出低损失 basin 情况下可改善精度/鲁棒性且不增加推理模型成本。 | 论文主要实验条件与 V4 多领域 RL 专家有差异；这里只作合并对象/成立条件对照，不把它当作 V4 的可替代配方。 |

## 方法比较的推理边界

1. **SFT/离线 response distillation**：固定 teacher answers 作监督，缓存与批处理方便；风险是学生自生成前缀与示范前缀错位。GKD 原论文将 student-generated states 作为对这一错位的处理方向。
2. **GKD/MiniLLM**：都属于 on-policy distillation 研究，但 divergence、轨迹信用分配和可调的生成数据占比各有具体定义；不能仅按 `OPD` 或 `reverse KL` 标签把目标和结果等同。
3. **Model Soups**：平均模型参数，依赖参数坐标/低损失区域条件；不是比较 logits，也不让学生在自身错误状态上问教师。
4. **Runtime routing/ensemble**：保留各专家推理能力，但在线多模型资源/延迟和 router 是成本；这与训练期参数融合是不同决策。最终 V4 仍然具有 MoE 架构，不应写成 OPD 删除了模型内部 routed experts。
5. **V4 multi-teacher full-vocabulary OPD**：以学生状态为采样分布、对多个专家输出分布做 reverse-KL 对齐，并用按需加载/hidden-state 缓存/输出头复用支持多教师。其关键取舍是降低 sampled-action 估计噪声、保留完整词表监督，同时承担教师推理与 logits 重建成本。

## 本轮新增的数学核对

- 固定状态分布：`p_S=(0.60,0.30,0.10)`、`p_T=(0.70,0.20,0.10)`，`D_KL(p_S||p_T)=Σ p_S ln(p_S/p_T)≈0.029149` nat。
- 单 token log-ratio 随机变量取值约 `[-0.154151, 0.405465, 0]`，按 `p_S` 加权的标准差约 `0.2505` nat；该量是 toy 单样本 KL 估计的标准差，不是策略梯度方差，也不是 V4 实测。
- 固定状态的加权 reverse-KL 目标 `Σ_i w_i D_KL(p||p_i)` 的无约束最优解为 `p*(v) ∝ Π_i p_i(v)^(w_i/Σ_j w_j)`；教师支持为零时需单独检查退化/不可行情况。
- 两教师等权 `p_A=(0.9,0.1)`、`p_B=(0.2,0.8)` 时，加权几何平均为 `(0.6,0.4)`，算术平均为 `(0.55,0.45)`。
- logits 账本示例：`8,192 × 100,000 × 2 = 1,638,400,000` bytes ≈ `1.526 GiB`/teacher；10 份约 `15.26 GiB`，只计 BF16 logits buffer，不含权重、激活、gradient、优化器或 I/O。

## 原知识点覆盖图

| 原稿问题/内容 | 处理层级 | 修订后位置 | 处理说明 |
|---|---|---|---|
| 为什么先分域训练再合并 | 核心 | §“一个模型为什么需要……” | 保留问题动机，增加报告实际披露和归因边界。 |
| 领域 SFT 与 GRPO | 核心 | §“第一阶段” | 保留原公式与组内标准化代码；明确这是通用教学示意，V4 未披露完整 advantage/clipping 配置。 |
| 领域 reward 与权限 | 核心 | §“为什么领域奖励要和权限分开” | 保留硬权限不得被任务奖励抵消；明示这是通用安全设计建议，不是 V4 披露的 reward 公式。 |
| OPD 与学生 rollout | 核心 | §“第二阶段” | 保留 student state、动作、observation、verifier、环境版本，明确 trace schema 为推荐审计契约。 |
| Reverse-KL 与 sampled log-ratio | 核心并加深 | §“第二阶段”及“为什么 V4 报告选择全词表” | 保留 token loss 直觉；把 log-ratio 明确标为 sampled policy-gradient surrogate，不当普通 reward；新增 KL 手算及代码。 |
| 多教师选择与冲突 | 核心 | §“多教师冲突…”、§“教师选择与冲突…” | 保留原路由/风险问题；增加 weighted geometric mean 数学推导，并把建议路由与 V4 已披露事实分开。 |
| 审计训练循环 | 核心 | §“一个可审计的训练循环” | 保留版本、状态、失败动作和环境账本；把全词表教师流程展开。 |
| 成本、失败模式 | 核心 | §“工程成本与失败模式” | 保留冲突、verifier 漏洞、状态漂移、成本和长轨迹信用；新增 teacher forward/logit/I/O 成本拆分。 |
| 与其它方法关系 | 核心研究比较 | §“代表性方案解决了什么…” | 保留领域 SFT、GRPO、离线 KD、OPD；增加 Model Soups/GKD/MiniLLM/推理路由比较与证据边界。 |
| 面试与练习 | 支撑 | §“面试追问”、`EXERCISES.md` | 保留原五题和练习；增加 mixed-RL 边界、MoE 专家歧义、全词表方差、调度和方法比较。 |

## 校验记录

- `evidence_level`：DeepSeek-V4 报告与模型卡为 `official/paper`；GKD、MiniLLM、Model Soups 为 `paper`；KL/显存 hand calculations 为 `local-demo`。
- `lifecycle`：当前公开模型/论文资料按 `active` 处理；OPD 方法结论绑定作者报告，未标为生产复现。
- `code_status`：提取并独立运行两个标准库 Python 围栏，输出与正文一致。另检查空 reward、NaN、bool、非正 epsilon、错误分布长度、负/非有限概率、归一化错误、teacher support 缺失；11 个非法输入均按约定拒绝，零学生概率/单 token support 示例通过。只验证本地教学代码，不代表 GRPO 训练器、V4、GPU 或生产 OPD。
- `formula_status`：检查 11 个 display-math 围栏成对且花括号平衡；复核 SFT/GRPO 定义域、固定前缀 reverse KL、sampled log-ratio 方向、多教师 geometric-mean 最优解、`logits = output_head(hidden_state)` 维度关系、显存账本和成本项；手算 `D_KL≈0.029149` nat、单-token estimator std `≈0.25046` nat、等权教师目标 `(0.6,0.4)`，以及 BF16 logits `1.526 GiB/teacher`。未使用外部 LaTeX/MathJax 渲染器。
- `link_status`：打开核对 DeepSeek V4 arXiv HTML §5.1.1/5.1.2/5.2.2、官方 Hugging Face 模型卡、GKD arXiv、MiniLLM arXiv 与 PMLR Model Soups 官方页/论文；新增章节/研究笔记/题库/练习内部相对路径均存在。章节当前映射已在第五册目录中，无需改目录。
- `sync_status`：`PAPERS.md` 增 DeepSeek V4 全词表 OPD 与 Model Soups 比较入口；`GLOSSARY_EN_ZH.md` 增 Domain Specialist Model、GRM、Full-Vocabulary Logit Distillation、Teacher Scheduling；`INTERVIEW_BANK.md` 增 V4 mixed-RL、full-vocabulary、MoE 专家歧义、teacher scheduler 和 merge 方法比较题；`EXERCISES.md` 增 5 项多教师 OPD 练习。
- `validation`：逐行复读修订后的章节；检查 Markdown fences 成对、11 个数学围栏花括号平衡、5 张表格列数一致、研究/题库/练习本地链接存在；运行两个 Python demo 和上述非法输入/边界样本；`git diff --check` 通过。第五册目录已有该章节，不改导航。未运行真实 V4/GRM/OPD、GPU、独立 benchmark、外部公式渲染或 PDF 构建。
- `open_risks`：没有在 V4 权重或目标集群上训练/推理；没有独立复现 GRM/OPD；报告没有披露具体路由表、完整 reward/loss 配置、sampled-vs-full-vocabulary 受控消融、峰值显存或可泛化端到端容量数据；Model Soups、GKD、MiniLLM 的论文实验与 V4 训练数据/模型/硬件不具备直接分数可比性。
