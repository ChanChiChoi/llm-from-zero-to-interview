# 第五册第十一章：后训练与对齐训练深度精修研究记录

日期：2026-10-26  
正文：[`book-05-llm-training/chapters/11-后训练与对齐训练.md`](../book-05-llm-training/chapters/11-后训练与对齐训练.md)  
进度依据：[`progress_v2.md`](../progress_v2.md) 中第五册第十章的断点；本轮从第十一章开始，没有重做更早的完成项。

## 研究问题与旧知识点覆盖图

基于当前 `WRITING_SPEC.md`，本轮把后训练从方法名称清单改成可追踪的问题链：反馈从哪里来、policy 如何更新、loss 的归一化对长短回答意味着什么、研究证据能支持多大范围的结论。修订前 1,472 行，修订后 1,581 行。

| 原有主题 | 重要性判断 | 本轮去向与处理 |
|---|---|---|
| SFT、response mask、Reward Model、RLHF、PPO | 核心：后训练反馈和策略更新的基础路线 | §1—3 保留。§3 将 π_old rollout snapshot、π_ref reference、RM 和 critic 分开，解释各自参与哪个计算，避免都叫参考模型。 |
| DPO 推导、beta、chosen/rejected、reference policy | 核心：常见离线偏好优化方法，必须能从目标推到 loss | §4 保留旧目标并补 KL-constrained reward 的闭式解、Bradley–Terry 下 prompt partition function 如何抵消、一个可手算的 logit/loss 例子，并限定原论文实验结论。 |
| IPO、KTO、ORPO、SimPO | 核心/支持：经常被拿来与 DPO 比较，但数据结构和目标不同 | §4.1 加数据形态、reference 依赖、目标、动机和证据边界对照。IPO、KTO、ORPO 没有被压成同义词；SimPO 保留独立展开。 |
| GRPO、组内均值/方差、critic 依赖 | 核心：理解 RLVR reasoning RL 的基础更新方法 | §6 保留并补 token 级 ratio 与 response 级 advantage 的维度、mask 和 reduction；用 [1, 0.5, 0] 手算组内均值、总体标准差和标准化 advantage。全同 reward 时零信号仍明确保留。 |
| GSPO 的序列级 ratio 和 clipping | 核心热点/争议：sequence 级与 token 级更新单位不同 | §7 保留并添加两 token 的几何平均数值例子，说明它不是普通整句概率乘积 ratio；将论文作者对长序列不稳定的解释标为论文诊断，并与 DCPO 的后续批评并列。 |
| DAPO 组合配方、动态采样、Clip-Higher、token loss、overlong | 核心热点：多个组件分别处理稀疏 reward、clip、长度和低效 rollout | §8 保留并拆开解释四项机制；加 token-level global denominator。记录 Qwen2.5-32B / AIME24 progressive ablation 30→36→38→41→42→50、Avg@32 和采样设置，明确这是逐步累积配方，不是独立随机组件消融。 |
| DrGRPO | 核心比较项：直接挑战按 response 长度和 reward 标准差归一化的偏置 | §11 增加机制、教学公式、长度权重例子和实验条件。说明 unbiased 限于论文的 policy-gradient 推导与采样假设；同值 reward 组仍是零 advantage。 |
| DCPO | 新近方法/高阶比较项：动态 token clip、跨步 reward 统计和利用率 | §9 收回到该论文实际贡献：概率相关 clipping bounds、SAS、OTM、RUR 和受控实验数据。旧有安全主题保留在 §1.6、§15.6；误放在 DCPO 中的通用约束清单删除为重复内容。 |
| DeepSeek-R1 / R1-Zero / RLVR | 核心案例：把方法、数据、阶段训练和蒸馏连到具体公开方案 | §10 细化 R1-Zero 的 base、无 SFT、GRPO、规则 accuracy/format reward；R1 的 cold start、RL、rejection sampling、SFT、二次 RL 和 distillation；强调最终增益不能归因单一 loss。 |
| 领域训练、on-policy distillation、公开模型接口 | 核心/支持：连接算法与真实后训练系统，同时标明公开材料的限制 | §12 保留领域专家和 OPD。旧泛化厂商表改为 DeepSeek-R1、GLM-5.3、gpt-oss、Grok 4.7 的可核对披露。把 thinking/reasoning effort 定义压缩为运行时控制面边界；它不是训练算法证据。 |
| 数据质量、reward hacking、length bias、KL drift、遗忘、安全回归 | 核心：reward 和平均 benchmark 不能代表整体质量 | §14—15 保留风险主题。原 W_pref 与代码只比较新旧模型同题得分，改名为 paired score gain；注明它不是人工偏好胜率，安全召回与过拒指标定义域仍分开。 |
| pipeline、训练日志、诊断、分析代码 | 核心：把概念落到实际数据字段与排查路径 | §16—18 保留。demo 改为显式区分任务是否需要在线探索、rollout 能否提供、verifier / 可信反馈是否存在、离线偏好数据是否存在；不再从数据的 reasoning category 推断算法。 |
| preserve/interleaved/adaptive thinking 等跨厂商字段 | 支持/低优先级：名称随产品和版本变化，不能解释后训练目标 | 保留为运行时字段边界，不再为未经统一定义的字段写跨厂商通用机制。 |
| 通用安全约束被放在 DCPO 技术段 | 重复/错位 | 从 §9 移出；安全与多目标评估继续在安全 alignment 和风险评估位置讲。没有删除原本有用的安全知识问题。 |

## 代表性方案对照与研究结论

### 离线偏好目标不是同一种标签或同一种目标

- [Direct Preference Optimization (DPO)](https://arxiv.org/html/2305.18290v3)：阅读 KL-regularized objective、optimal-policy 闭式解、隐式 reward 重写和 preference loss 推导。相同 prompt 下 chosen/rejected 的 Bradley–Terry 比较抵消只依赖 prompt 的 partition term，因此可直接训练 policy/reference log-ratio。这个推导解释它如何省去独立 RM 与在线 RL 环节，不代表没有 reward 概念。原论文结果限定在相应情感控制、摘要和单轮对话实验，不能据此断言所有规模、任务都比 PPO 更稳定或更好。
- [IPO](https://arxiv.org/html/2310.12036)：核对平方目标和有限 margin。它针对偏好建模假设与继续扩大 margin 的风险提出另一目标；论文作者明确把更大规模/复杂语言模型实验留作后续，不能写成大模型普适胜出。
- [KTO](https://arxiv.org/abs/2402.01306)：核对 desirability / undesirability 单样本反馈与 prospect-theoretic objective。它允许不同于 pair 的反馈形态，但依赖 reference；作者实验规模和任务不代表所有偏好数据的优势。
- [ORPO](https://arxiv.org/abs/2403.07691)：核对单阶段 SFT + odds-ratio preference objective 与 reference-free 设置。论文测试的模型规模/任务范围限制了对更大模型或复杂偏好任务的外推。
- [SimPO](https://arxiv.org/abs/2405.14734)：保留原章已有机制，重点是平均 log-prob 隐式 reward 与 target margin。免 reference 降低训练中的一份模型负担，也拿掉一个 policy drift 锚点；仍须观察长度和回归。
- [A Survey of Post-Training Scaling in Large Language Models](https://aclanthology.org/2025.acl-long.140/)：用于后训练分类与 SFT、反馈式 RL、TTC 问题图谱；综述建立地图，不替代算法原论文和具体实验表。

### RLVR 的组更新、归一化与裁剪不是一个旋钮

- [DeepSeekMath / GRPO](https://arxiv.org/html/2402.03300v3)：核对 group-relative baseline、reward standardization 和减少 critic 依赖的设定。
- [DAPO](https://arxiv.org/html/2503.14476v2)：阅读四组件定义、dynamic sampling 与 progressive ablation。动态采样只保留二元 reward 中正确数 0<correct<G 的 group，会改变进入更新的题目分布；token loss 的全局分母提高长回答 token 的总权重。论文 no-KL 是其 reasoning 配方，不能概括为所有 RLVR 不需要 KL。
- [Understanding R1-Zero-Like Training / DrGRPO](https://arxiv.org/html/2503.20783v2)：阅读长度 bias 解释、移除 1/|o_i| 与 group-std normalization、固定 generation-budget denominator 的推导和 base-policy/template 分析。论文报告 Qwen2.5-Math-7B、MATH levels 3–5、Qwen-Math template、8 张 A100/约 27 小时的 recipe，在其 AIME24 设置报告 43.3%。其 unbiased 是论文推导口径，不是对所有裁剪/实现/任务的无条件保证。该工作还提醒，特定 base model 与 prompt 上的长推理行为可能先于 RL 出现；行为检测与 template 条件不能省略。
- [GSPO](https://arxiv.org/html/2507.18071v2)：阅读 sequence likelihood ratio 与 sequence clipping 方案及作者对长 response / MoE 稳定问题的诊断。正文用两 token 几何平均手算帮助看出其长度归一化效果，也说明它不同于整句 ratio 的乘积。
- [DCPO](https://arxiv.org/html/2509.02333v2)：阅读动态 clip bounds 推导、Smooth Advantage Standardization、OTM、AIME 表和 RUR。论文公式随 old-token 概率 q(x) 扩展 bounds；低概率 token 获得更宽窗口。SAS 组合当前与累计历史 reward 标准化值，RUR 只统计非零 advantage response 占比。论文 Qwen2.5-Math-7B / AIME24 Avg@1/32 报告 GRPO 36.7/32.1、DAPO 36.7/31.6、GSPO 40.0/34.9、DCPO 46.7/38.8；四模型/400 steps RUR 平均 GRPO 43.8%、GSPO 45.6%、DCPO 71.8%。所有数字均绑定论文数学数据、Qwen-Math template、温度/top-p、模型和评测口径；RUR 不是正确率，也不与 DAPO dynamic-sampling 过滤量形成同分母比较。

这些论文修改不同的项：advantage 的缩放方式、ratio 和 clipping 的粒度、clip bounds、group 保留规则或 loss denominator。因此新分数不能直接跨论文排列。迁移时需固定 base、训练题池、prompt template、verifier、reward、采样预算、max length、更新步数/训练 token 和评测解码，再同时对照准确率、长度/能力切片、clip fraction、非零 advantage、过滤率、wall-clock 与单位成功成本。

### 公开模型案例：官方披露能证明到哪一步

- [DeepSeek-R1](https://arxiv.org/html/2501.12948)：阅读 R1-Zero §§2.1–2.3 的 GRPO、规则 reward 和训练说明；R1 §3 的多阶段 pipeline；附录 B.3 的 cold-start、监督样本与 rejection sampling。论文明确写出 R1-Zero 不先做 SFT，以 DeepSeek-V3-Base 和规则 accuracy/format reward 训练；R1 再加入冷启动、推理/非推理混合 SFT、二次 RL 和蒸馏。作者也提醒表达形式与人为设计的启发式不等于人类式智能。正文用它说明配方而非单 loss 归因。
- [GLM-5.3 官方博客](https://z.ai/blog/glm-5.3)：官网发布页说明同 GLM-5.2 base 的 post-training scaling、长周期 runnable environments、solvability judge、reference-free verifier、oracle/no-op/unsolved checks、solver trajectory reward-shortcut audit，以及 slime 的 Megatron/SGLang/data-buffer dataflow。博客报告训练/rollout 平均 logprob 差约 1e-7 和长周期 coding RL throughput 超 2.3×，只按发布方测量引用。当前网页抽取器直接打开返回 0 行，本轮搜索索引呈现官方正文；因此这些技术数字在研究记录里保留为 publisher-reported，未声称独立获取原始实验 artifact。
- [OpenAI Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)：正文 Post-training 披露 SFT + high-compute RL、Model Spec/CoT/tool-use 目标，CoT 段明确表示未直接监督 CoT。没有公开足以推断 optimizer、reward、KL、rollout scale 的细节。
- [xAI Introducing Grok 4.7](https://x.ai/news/grok-4-7)：官方说明更大的 base model、更长 RL run、更难且偏多小时完成的任务，并提自验证和长上下文。文章没有算法、reward、样本规模与消融；base model 同时变化，不能把 benchmark 增益全归因于 RL 时长。

## 公式、代码、修改与验证范围

- §4 DPO：逐步说明 KL 约束最优策略、隐式 reward 和 pairwise loss 的关联；附概率 margin 手算例子。reference 是相对锚点，不是硬约束，也不阻止代理 reward 过优化。
- §6 GRPO：把 response advantage 广播到 response token 维；损失 mask 和 reduction 显式呈现；二元/三元 reward 例子说明中心化与标准差归一。
- §7 GSPO：用 [2, 0.5] 与 [1.2, 1.2] 说明几何平均 ratio 的手算结果，标明公式教学化和论文诊断边界。
- §8 DAPO：说明 dynamic sample distribution shift、全局 token denominator 的长度权重、overlong filter 与软惩罚的区别；记录论文渐进配置数据和 KL scope。
- §9 DCPO：补动态上下界 0.5+0.5√max(1−4ε_low/q,0) 与 0.5+0.5√(1+4ε_high/q)；解释低概率 token 的窗口变化、SAS、OTM、RUR 和对照条件。把泛安全约束内容回归其安全小节。
- §10 DeepSeek-R1：补 reward 组成、各训练阶段与可读性/通用性问题；引出 base-policy 研究对 reasoning emergence 的因果限制。
- §11 DrGRPO：用 response 长度 2/100、固定预算 100 的系数例子展示分母如何改变权重；另用 reward [1,0] 区分中心化与标准差归一，并解释全同 reward 情形。
- §12：替换无来源/过期字段堆叠，改为四个有官方来源的案例，保留运行时 reasoning budget 与 training algorithm 的边界。
- §13、§16、§18、§19：同步算法选择表、pipeline、诊断和总结，加入 DrGRPO/DCPO。
- §15：将新旧同题分数比较改名为 R_score_gain / paired score gain，不冒称偏好 win rate；安全召回、过拒、能力回归分别定义分母。
- §17 demo：序列平均 logp 字段改为 old_mean_logp/new_mean_logp；ratio、clipped surrogate、最大偏离都标为 normalized sequence proxy。方法建议读取任务是否需要在线探索、rollout 可用性、verifier/可信在线反馈与 preference data，不再从数据 category 猜 GRPO/DPO。执行后的关键结果为 assistant serialized-token share 0.7177、RM loss 0.3749、DPO loss 0.6786、平均 chosen-minus-rejected 长度 -3.33、paired score gain rate 0.75、safety recall 1.0、over-refusal 0.3333、候选方法家族 RLVR_candidates: GRPO/DAPO/DrGRPO/GSPO/DCPO。代码仅是标准库分析 demo，不进行参数更新，也不代表真实 GRPO/GSPO/DCPO 训练。

## 配套同步

- `PAPERS.md`：alignment 论文路线补入 Survey、IPO、KTO、ORPO、GSPO、DCPO、reward overoptimization 和三个官方模型披露案例，标明模型发布披露不等于算法证据。
- `GLOSSARY_EN_ZH.md`：补 IPO、KTO、ORPO、GRPO、DAPO、DrGRPO、GSPO、DCPO、SAS、OTM、RUR；未重复已有 RLVR、SimPO、DeepSeek-R1。
- `INTERVIEW_BANK.md`：增加 20D，要求按机制、证据口径、边界和同条件实验比较五类 RLVR 更新方法。
- `EXERCISES.md`：新增 43D/43E 与后训练百科第 6 题，覆盖分母手算、全同 reward、RUR 和公平对照协议。
- `book-05-llm-training/目录.md`：已有第十一章入口，本轮没有改动该导航。

## 静态/代码验收与未完成项

- §17 的单个 Python fence 已从正文提取并在 Python 运行，退出码为 0，标准输出与文中期望输出完全一致；未增加项目测试或依赖。
- 静态核对通过：一级标题 1—20 连续；134 个 fenced marker 成对；35 个 `math` fence 的花括号平衡；10 张 Markdown 表的列数一致；4 个相对链接目标存在；`git diff --check` 通过。
- 公式渲染器、PDF 和真实训练均未运行；上面的公式检查只覆盖数学围栏的括号配对，不等同于外部渲染或公式证明。
- 未运行真实模型、RLVR benchmark、verifier、GPU、多节点、长周期环境或第三方复现；无法据本轮 toy output 宣称模型训练结论。
- 原论文结论受各自模型、数据、模板、reward、loss reduction、采样、硬件和评测协议约束。GLM-5.3 公开博客网页当前无法由本轮 HTML 抽取器完整展开；应保留来源可读性边界。

## 验收状态

正文、研究来源和配套索引已完成本轮主要修订；示例主路径、静态结构/公式围栏/表格/链接核对均通过，且已将本轮记录同步到 `progress_v2.md`。未运行真实模型、benchmark、训练栈、PDF 或外部公式渲染；这些限制不属于本轮已验证范围。
