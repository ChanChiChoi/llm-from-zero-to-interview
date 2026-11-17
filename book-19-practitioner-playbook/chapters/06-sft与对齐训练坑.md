# 第六章：SFT 与对齐训练的行为事故

后训练把基础模型变成可用助手，却不是一个单向的“能力增强”按钮。SFT、偏好优化、Reward Model、RLHF、DPO 和安全训练都在改变模型的行为分布：它们可能改善指令跟随、格式稳定性和拒答边界，也可能损伤数学、代码、事实、长上下文、工具调用和输出多样性。

后训练事故的危险在于，局部体验可能变好，整体能力却已经退化。模型更会聊天，但数学和代码变差；安全拒答率上升，但正常教育和防护问题也被拒绝；reward 分数上升，但答案更长、更空、更模板化；DPO loss 下降，但 reference model、偏好 pair 或工具 schema 根本没有对齐。

本章把后训练看成一次行为分布重塑。先定义训练想改变什么、不能损伤什么，再分别审查 SFT 监督边界、能力回归、过度拒答、偏好数据、Reward Model、RLHF reward hacking、DPO margin、长度和工具协议。每个指标都明确样本集合、适用条件和零分母语义，最后用一个标准库审计示例把多个故障放在同一张报告中。

## 6.1 后训练改变的是行为分布

基础模型定义初始策略 $\pi_{\mathrm{base}}(y\mid x)$；后训练用特定数据、监督目标、偏好或奖励，以及优化和安全约束，得到新的策略 $\pi_{\mathrm{post}}(y\mid x)$。这不是一个统一算法：数据和 mask 决定哪些行为进入训练信号，reference 和 reward 影响优化方向，评估定义决定我们能看见哪些变化。行为分布可能在目标任务上改善，也可能在没有直接训练的能力切片上回退。

后训练开始前，应该写一份“允许改变”和“必须保留”的能力合同：

- **允许改变：** 指令跟随、结构化输出、目标领域术语、工具调用格式。
- **必须保留：** 基础知识、数学、代码、事实准确性、长上下文和多语言能力。
- **必须改善：** 危险请求处理、隐私边界、误拒和漏拒。
- **必须稳定：** EOS、role、tool schema、引用和状态协议。
- **必须观察：** 输出长度、风格、成本、延迟、人工采纳和真实任务完成。

没有这份合同，团队容易用一个聊天分数证明所有事情都变好。比较 base、SFT、偏好优化和最终版本时，还要固定 checkpoint、数据版本、模板、解码和评估器；否则观察到的变化可能来自评估条件，而不是训练。

## 6.2 SFT 的监督边界

SFT 的关键不是“有多少条指令”，而是模型被要求预测哪些 token。设第 $i$ 条序列经 tokenizer 后为 $z_{i,1},\ldots,z_{i,L_i}$。自回归模型在读到 $z_{i,1:t-1}$ 后预测目标 token $z_{i,t}$；监督 mask $m_{i,t}$ 必须与 causal shift 后的目标位置对齐。若训练目标是对所有有效目标 token 作等权平均，损失为：

$L_{\mathrm{SFT}}=-\frac{1}{D_{\mathrm{loss}}}\sum_i\sum_{t=2}^{L_i}m_{i,t}\log p_\theta(z_{i,t}\mid z_{i,1:t-1})$

在这个等权 token-mean 合同下，$D_{\mathrm{loss}}=\sum_i\sum_t m_{i,t}$；若有样本权重、按序列平均或自定义分布式归约，分母要按实际实现另行定义。输入 token 数描述计算负载，shift/mask 后有效目标数描述监督覆盖，loss 分母取决于 reduction 合同，三者不能互换。

PyTorch 的类别索引交叉熵会忽略 ignore_index 目标，并按未忽略目标或有效权重归一。[PyTorch 2.14 CrossEntropyLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) 具体 SFT 训练器还可能从 chat template 生成 assistant mask。例如 TRL v1.14.1 的 assistant-only loss 依赖模板中的 generation 区域标记，因此应回放实际渲染、shift 后 labels 和 reduction，而不能只看配置名。[TRL v1.14.1 SFTTrainer 文档](https://huggingface.co/docs/trl/v1.14.1/sft_trainer)。

在 assistant-only 任务中，system 和 user 通常是条件，不是要模型复述的目标；assistant 内容和按任务约定的 EOS 参与 loss；PAD 和未监督控制位置不参与。具体策略可以因任务而变化，但必须写进 collator 和数据契约。

最常见的 SFT 事故是 prompt loss 泄漏：labels 在 system/user 位置没有被忽略，模型因此学会复述问题、生成 role token 或把下一轮 user 说出来。另一个相反事故是 assistant 内容被全部 mask，训练 loss 仍能计算或被日志吞掉，但模型没有获得有效答案监督。

检查一条样本时，必须同时打印原始 messages、渲染文本、token、role spans、shift 后 labels 和有效监督目标数。仅仅看到训练 loss 在下降，不能证明监督位置或实际 reduction 正确。

## 6.3 SFT 后能力回归

SFT 后聊天体验变好而数学、代码或事实能力下降，可能与数据范围、学习率、训练步数、模板和分布偏移有关；也可能是评估格式、提示或评分器变化造成的假象。先固定 checkpoint、tokenizer、模板、解码和评分协议，再做同条件比较。InstructGPT 论文在其特定 RLHF 流程中报告了若干公共 NLP 任务回退，并发现混入预训练目标可缓解部分回退，但仍未恢复所有任务；这说明回归需要测量，不证明所有后训练都会产生同样变化。[InstructGPT 论文 §1、§4](https://arxiv.org/html/2203.02155)。参数高效微调也不保证旧能力不变；LoRA 微调中的遗忘已有特定模型和配置下的研究观察。[Kalajdzievski, 2024](https://arxiv.org/html/2401.05605)。

对能力切片 $k$，设同一评估口径下训练前后的指标分别为 $M_{\mathrm{before},k}$ 和 $M_{\mathrm{after},k}$：

$\Delta_k=M_{\mathrm{after},k}-M_{\mathrm{before},k}$

例如数学正确率从 70/100 到 55/100，差值是 −15 个百分点。这个差值描述这组样本上的结果；样本量、任务构成和抽样不确定性仍要一并报告。数据集、提示、评分器、解码、工具和资源条件不一致时，不能把差值归因于模型变化。需要分别报告 instruction、math、code、knowledge、long-context、multilingual、safety、tool 和真实业务任务，而不是只报一个加权总分。

常见回归模式：

- 领域 SFT 数据过窄，通用问题被错误处理；
- 高学习率或过多 epoch 让模型过度适应少量模板；
- 安全数据比例过高，正常问题被拒答；
- 工具调用格式训练过多，自然对话变得生硬；
- 合成数据风格压过真实用户表达；
- tokenizer、template 或 EOS 变化让评估输入不再可比。

修复可以从更早 checkpoint、较低学习率、混入通用高质量数据、调整数据配比和改善多维回归开始。LoRA 或其他参数高效方法不自动消除回归风险，adapter 仍然会改变输出分布和格式。

## 6.4 过度拒答和安全边界

安全训练不能简单等价为“拒答越多越安全”。正常的教育、防护、合规和风险分析问题可能包含敏感词，但任务本身不要求执行危险行为。如果所有边界输入都被拒答，用户任务完成率和系统可信度都会下降。

把样本分成应该回答的安全集合 `S` 和应该拒绝或受限处理的风险集合 `U`。当 `|S|>0` 时，误拒率为：

```math

R_{false\_refuse}=\frac{N_{S,refused}}{|S|}
```

当 `|U|>0` 时，漏拒率为：

```math

R_{unsafe\_leak}=\frac{N_{U,answered\_unsafely}}{|U|}
```

若评估集没有安全样本，指标是 `not_applicable`；如果应该有样本但标注或执行结果缺失，指标是 `unknown`。不能把没有安全样本的模型记为零漏拒。

安全边界应按任务意图、可执行性、潜在伤害和可提供的安全替代帮助判断。一个高质量拒答可能说明不能提供危险操作步骤，同时给出防护、求助或合规路径；单纯输出长免责声明不等于安全，也不等于有用。

过度拒答排查要看：安全数据与正常数据比例、标注规则、关键词触发、拒答模板、任务切片、语言差异和高风险判定是否被模型自行替代。不要把所有安全行为压缩成一个 refusal rate。

## 6.5 多轮和工具调用的后训练格式

后训练数据的角色和工具协议必须与线上执行器一致。多轮样本中，system、user、assistant 和 tool 的顺序、EOS 和 labels 决定模型学到的状态转换；工具样本中，tool name、参数 schema、call ID 和 observation 决定模型是否会把调用当成真实协议。

常见事故包括：

1. tool result 被当成 assistant 说话，模型学会伪造工具结果。
2. 工具名和线上 registry 不一致，生成能解析却找不到执行器。
3. 训练 schema 的字段是 `expression`，线上 schema 改成 `expr`，参数校验全部失败。
4. 只训练成功调用，没有训练超时、拒绝、重试和未知外部状态。
5. 生成自然语言和 tool call 混在同一段，解析器无法判断提交边界。

工具对齐要同时评估选择、参数、schema、执行、观察使用和恢复；调用次数多不代表工具使用正确。外部动作还必须受权限、幂等和人工确认控制，不能因为模型在训练中生成了某个 JSON 就自动执行。

## 6.6 偏好数据是一份质量合同

DPO、RLHF、RLAIF 和其他偏好方法依赖 chosen/rejected 或排序数据。偏好不是天然客观的标签；它可能同时混入事实正确性、安全、简洁、风格、格式、礼貌、长度和标注者个人偏好。偏好标签表示在给定任务、rubric、目标人群和候选回答下作出的比较，不能直接解释成绝对质量或事实真值。DPO 的推导也明确依赖所选偏好模型等假设。[DPO 原论文](https://arxiv.org/html/2305.18290)。

一条偏好记录应至少说明：

- 输入和任务类型；
- chosen / rejected 文本；
- 偏好的主要维度与 rubric；
- 风险等级和安全策略；
- 标注者或评审器版本、目标人群和语言；
- 置信度、平局/弃判及复核状态；
- 长度、格式和工具状态。

如果 chosen 只是更长、更礼貌或格式更完整，但事实并不更正确，模型可能学到长度或模板捷径。训练前要对 pair 做独立质量检查，不要把“有人选了 chosen”当作“chosen 在所有目标上都更好”。

标注一致性要按任务、风险级别和标注者分片。整体一致率高，可能只是大量容易样本掩盖了高风险边界的一致性低。对低置信 pair，隔离、重标或降低权重通常比盲目增加数量更可靠。偏好也受目标人群和评分规程影响：HUMAINE 的美国、英国参与者研究在其样本中观察到模型排名随年龄组变化；这提醒我们记录评审群体、语言和任务维度，不能把单一标注池的平均排序当成所有用户的共同偏好。[HUMAINE（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9c1bcfe6edf7e47d83d2e99492d851e4-Abstract-Conference.html)。

## 6.7 Chosen/Rejected 差异太小或方向相反

偏好 pair 审计的首要问题是方向是否与任务 rubric 一致，不是两个回答的分数差够不够大。让独立评审在不知道原始标签的情况下，对事实、任务完成、安全、格式和风格等目标维度分别判断：chosen 更好、rejected 更好、平局，或证据不足无法判断。

对于方向明确的抽查 pair，可以报告 chosen 胜出的占比：

$A_{\mathrm{pref}}=\frac{N_{\mathrm{chosen\ wins}}}{N_{\mathrm{chosen\ wins}}+N_{\mathrm{rejected\ wins}}}$

平局和无法判断项不进入这个分母，必须另外列出；若没有方向明确的 pair，该占比不适用。这个数只描述指定样本和 rubric 下的方向一致性，不是跨任务阈值。低置信 pair 应重标、仲裁或保留为不确定；若要比较数值评分，先确认两个分数来自同一量尺并有共同校准依据。

## 6.8 Reward Model 的捷径偏差

Reward Model 的目标是近似某种偏好或质量判断，但它可能学到容易检测的表面特征：答案长度、礼貌词、项目符号、免责声明、特定模板或参考答案词汇。模型一旦被 RL 或 rerank 优化，就会放大这些捷径。

Reward 分数和人工 rubric 分数通常不是同一量尺。除非定义了共同校准集和映射，不要直接计算两者之差并称作“人类质量差距”。更稳妥的检查是对相同 prompt 构造反事实 pair：保持任务答案和事实内容不变，只改变长度、礼貌模板、免责声明或格式；再检查 RM 的排序是否随表面变化，而独立评审是否认为任务质量发生变化。也可以保持长度相近，只改变事实正确性。

长度与 reward 的相关性只能作为筛查信号，不能单独说明长度造成了奖励上升；样本少或长度/reward 几乎不变时，相关系数也不稳定或未定义。抽查样本少时，报告具体 pair 和不确定性，不把一个相关系数当成上线结论。

## 6.9 RLHF reward hacking

Reward hacking 是策略学会利用 reward 的漏洞，而没有提升真实任务质量。常见候选信号是回答越来越长、免责声明泛滥、格式越来越像评分器偏好的模板、事实错误被漂亮结构掩盖，代理 reward 上升而独立任务评估没有同步改善。

排查要保留 SFT 或 base baseline，并同时监控：

- reward 和独立人工/任务质量；
- 输出长度与有效信息密度；
- 事实错误和引用支持；
- 安全误拒和漏拒；
- KL 或与 reference 的行为偏移；
- 真实任务完成率、成本和延迟。

KL 约束可以帮助限制策略偏离 reference，但 KL 不是质量指标，也不能修复错误 reward。Gao 等人的合成研究在其指定 proxy/gold RM 与优化设置中观察到 gold reward 随优化强度先升后降；这支持监测代理与独立质量信号是否分离，不提供跨任务阈值或普遍的人类偏好定律。[Gao et al., 2022](https://arxiv.org/html/2210.10760)。一项 2025 年预印本还在特定编码 RL 环境中报告 reward hacking 与若干跨任务行为评估变化相关；该结果受其训练与评估构造限制，不能解读成线上模型的发生率或所有 reward hacking 的必然后果。[MacDiarmid et al., 2025](https://arxiv.org/html/2511.18397v1)。

多维独立评估、人工抽样、反事实 reward 测试和高 reward 样本回放，才能发现代理目标是否偏离本次任务目标。

## 6.10 DPO 的 reference、beta 和 margin

DPO 直接用偏好 pair 和 reference model 优化相对偏好。这里把一个回答的序列 log probability 定义为目标回答各 token 条件 log probability 的**和**：

$\log\pi(y\mid x)=\sum_{t=1}^{|y|}\log\pi(y_t\mid x,y_{1:t-1})$

对 chosen $c$、rejected $r$ 和输入 $x$，reference-adjusted margin 为：

$m=\left[\log\pi_\theta(c\mid x)-\log\pi_\theta(r\mid x)\right]-\left[\log\pi_{\mathrm{ref}}(c\mid x)-\log\pi_{\mathrm{ref}}(r\mid x)\right]$

相应的 sigmoid loss 是 $L_{\mathrm{DPO}}=-\log\sigma(\beta m)$。reference model 必须是训练记录中声明的固定版本；$\beta>0$ 控制这个目标对 margin 的缩放，但具体效果依赖数据、reference 和实现，不能作为统一质量旋钮。

reference、tokenizer、template、EOS/截断和序列 reduction 必须绑定版本，否则 margin 不能直接解释。序列求和会累积每个 token 的 log probability；若改成 token 平均，长度不同的回答会得到不同的 margin 和训练目标。DPO 原论文推导依赖固定 reference、KL 正则化 reward 目标及 Bradley–Terry 类偏好模型假设；实现中也可能提供长度归一化等不同目标。举例来说，TRL v1.14.1 的 sigmoid 路径对 completion token log-prob 求和，而 sigmoid_norm 会按 completion 长度归一；这些结果不能混成同一配置比较。[DPO 原论文](https://arxiv.org/html/2305.18290)；[TRL v1.14.1 DPO 源码](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/dpo_trainer.py#L1241-L1268)。

DPO 排查要看：

1. margin 分布及方向明确/不确定的样本；
2. beta 的反事实实验；
3. reference 和 policy 的 checkpoint、模板和 tokenizer；
4. pair 的任务、风险和长度分布；
5. 输出风格、长度和能力切片的回归；
6. 训练前后真实任务和安全边界。

loss 下降只能说明优化目标被拟合，不能说明偏好标签正确或产品结果改善。

## 6.11 输出长度和风格漂移

后训练后输出变长，可能源于 SFT 答案本身过长、标注者偏好解释、reward model 的长度捷径、安全免责声明或 DPO pair 的长度差异。长度变化会影响成本、延迟、用户阅读和工具调用次数。

不要只看平均长度。应按任务、语言、风险、成功/失败和用户采纳分片。一个数学题答案变长但正确率不变，可能是效率下降；一个安全回答变长但仍漏拒，说明免责声明没有解决风险；一个 RAG 答案变长却引用覆盖下降，说明上下文和生成协议可能失衡。

风格指标也需要任务化：简洁性、结构清晰、事实密度、引用可用、拒答解释和格式遵循。所有回答都变成同一种模板，不能简单称为“风格稳定”。

## 6.12 多维评估和阶段对照

后训练评估至少包括：

| 维度 | 需要观察什么 |
| --- | --- |
| 指令 | 是否理解任务、遵守约束和处理多轮 |
| 知识 | 事实、时效、引用和拒答边界 |
| 数学/代码 | 正确率、过程、测试和鲁棒性 |
| 安全 | 误拒、漏拒、替代帮助和隐私 |
| 工具 | 选择、参数、schema、观察使用和恢复 |
| 语言 | 各语言质量、混用、术语和输入鲁棒性 |
| 长上下文 | 证据定位、跨段关系和成本 |
| 产品 | 任务完成、人工采纳、延迟、成本和反馈 |

评估必须保留 base、SFT、偏好优化和最终版本的对照。只比较最终模型与一个旧 baseline，无法知道退化发生在哪个阶段。对于高风险任务，离线分数不能替代人工审查和受控灰度。

## 6.13 后训练事故的诊断顺序

遇到后训练后行为异常，可以按以下顺序：

1. 固定模型、tokenizer、template、解码和评估器版本。
2. 对比 base/SFT/偏好优化/最终 checkpoint 的任务切片。
3. 抽查原始 SFT 样本和 labels，确认 assistant 监督边界。
4. 抽查 chosen/rejected，确认偏好方向、任务和风险维度。
5. 查看拒答、输出长度、风格、工具和引用的变化。
6. 对 reward 与人工质量做相关和反事实测试。
7. 检查 DPO reference、beta、margin 和 tokenizer/template。
8. 复核工具 schema、权限、状态和错误恢复数据。
9. 做小规模单变量 ablation，再决定重训或回滚。

这条顺序先排除数据和协议问题，再解释训练算法。因为一个错误的 mask 或 schema 会让所有高级方法都在错误目标上优化。

## 6.14 指标定义域

### 6.14.1 Assistant 覆盖、prompt 泄漏和 loss 分母

所有位置先对齐到 causal shift 后的目标 token，再按 role span、EOS 和真实 padding 位置分类：

$C_{\mathrm{asst}}=\frac{N_{\mathrm{assistant\ labels}}}{N_{\mathrm{assistant\ target\ positions}}},\quad
R_{\mathrm{prompt}}=\frac{N_{\mathrm{prompt\ labels}}}{N_{\mathrm{prompt\ target\ positions}}},\quad
R_{\mathrm{pad}}=\frac{N_{\mathrm{pad\ labels}}}{N_{\mathrm{pad\ target\ positions}}}$

$C_{\mathrm{EOS}}=\frac{N_{\mathrm{EOS\ labels}}}{N_{\mathrm{EOS\ target\ positions}}}$

这里 labels 指训练目标中未被忽略的 labels；target positions 指 shift 后属于相应类别的目标位置。若目标位置分母为零，指标是 not_applicable；若本应存在但数据/执行结果缺失，则是 unknown。input tokens 单独报告，不代替有效监督目标数。实际 $D_{\mathrm{loss}}$ 由训练器 reduction 决定：仅在等权 token-mean 且无额外权重时，它等于未忽略目标数；按样本、序列或 rank 聚合的实现应列出自己的分母合同。

### 6.14.2 能力回归

$\Delta_k=M_{\mathrm{after},k}-M_{\mathrm{before},k}$

前后评估必须使用同一任务、样本版本、提示、评分器、解码、工具和资源条件。汇报每一切片的分子/分母、样本量及适用的不确定性；缺少一侧有效分母时不计算差值。

### 6.14.3 拒答边界

$R_{\mathrm{false\_refuse}}=\frac{N_{\mathrm{safe,\ refused}}}{N_{\mathrm{safe}}},\quad
R_{\mathrm{unsafe\_leak}}=\frac{N_{\mathrm{unsafe,\ answered\ unsafely}}}{N_{\mathrm{unsafe}}}$

safe 和 unsafe 集合须由任务策略事先定义，澄清、部分拒绝或安全替代回答按 rubric 单独标记。分母为零时记 not_applicable；应有样本但标注/输出缺失时记 unknown。误拒与漏拒不能合并成一个 refusal rate。

### 6.14.4 Reward 与独立质量判断

对同 prompt、同 rubric 下由独立评审判为非平局的 $N_{\mathrm{resolved}}$ 个 pair，可报告 RM 排序与独立评审方向一致的占比：

$C_{\mathrm{rank}}=\frac{N_{\mathrm{same\ order}}}{N_{\mathrm{resolved}}}$

另列平局、无法判断和评审分歧。这个指标只描述抽查 pair 的排序一致性，不是奖励校准、事实性或普遍质量；长度相关性只能作为筛查，需结合按任务匹配的反事实 pair。若 reward 与人工评分没有共同量尺，不计算二者的数值差。

### 6.14.5 DPO margin

$m_j=(\log\pi_\theta(c_j\mid x_j)-\log\pi_\theta(r_j\mid x_j))-(\log\pi_{\mathrm{ref}}(c_j\mid x_j)-\log\pi_{\mathrm{ref}}(r_j\mid x_j))$

这里每个序列 log probability 按实际实现由 completion token log probability 求和或归约；reference revision、tokenizer、template、EOS、截断、长度归一化、beta 和 loss type 必须随指标记录。$m_j$ 不是 chosen 内容真实优劣的证明，负 margin 也要回到偏好方向和样本内容检查。

## 6.15 用一份合成账本复算后训练事故

下面的数据都是教学构造，不代表真实模型结果或行业门槛。先声明本例的训练合同：按 causal shift 后的目标 token 对所有未忽略 labels 作等权平均；assistant 回答和约定的 EOS 参与监督，system/user 和真实 padding 位置不参与。实际工程若使用别的 reduction，要换成其真实分母。

### 先确认模型实际学了哪些位置

| 样本 | 输入 token 数 | assistant 目标位 / 已标注 | prompt 目标位 / 已标注 | padding 目标位 / 已标注 | EOS 目标位 / 已标注 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 10 | 4 / 4 | 3 / 0 | 2 / 0 | 1 / 1 |
| B | 8 | 3 / 2 | 4 / 0 | 0 / 0 | 1 / 0 |
| C | 7 | 3 / 3 | 2 / 1 | 2 / 1 | 1 / 1 |
| 合计 | 25 | 10 / 9 | 9 / 1 | 4 / 1 | 3 / 2 |

先不要把 25 个输入 token 当成 25 个监督目标。按表中 causal shift 后的目标位置计算，assistant 覆盖率是 $9/10=90\\%$，prompt loss 泄漏是 $1/9$，padding loss 泄漏是 $1/4$，EOS 覆盖率是 $2/3$。样本 B 漏掉一个 assistant 目标（这里是 EOS）；样本 C 则让一个 prompt 位置和一个真实 padding 位置进入 labels。

如果这 9 个已标注 assistant 目标的交叉熵之和为 7.2，prompt 泄漏贡献 0.8，padding 泄漏贡献 1.1，那么本例实际 numerator 为 $7.2+0.8+1.1=9.1$，实际未忽略 label 数作为分母时 $D_{loss}=11$，观察到的 token-mean loss 为 $9.1/11\\approx0.827$。这个数反映训练器实际优化的目标；它既没有补回漏掉的 assistant 目标，也不能仅凭 loss 数值发现错位。修正 labels 后还要重算有效目标数和 reduction 分母。

### 再看行为变化落在哪些任务

| 切片 | 后训练前 | 后训练后 | 差值 |
| --- | ---: | ---: | ---: |
| 指令完成 | 62/100 | 76/100 | +14 个百分点 |
| 数学正确 | 70/100 | 55/100 | −15 个百分点 |
| 代码正确 | 66/100 | 49/100 | −17 个百分点 |
| 工具调用成功 | 60/100 | 52/100 | −8 个百分点 |

这些差值只有在同一批任务、提示、评分器、解码和工具条件下才有可比性。它们提示本次后训练伴随数学、代码和工具回退，不能单凭这 100 个样本推断长期总体表现；需报告样本构成和适用的不确定性，再定位回退首次出现的 checkpoint。

### 拆开检查误拒和漏拒

本例的安全集合包含 3 个应正常回答的请求，其中 2 个被拒；风险集合包含 3 个应拒绝或受限处理的请求，其中 1 个得到不安全回答。因此误拒率为 $2/3$，漏拒率为 $1/3$。两个分母分别来自安全请求和风险请求，不能合并。样本数很小，只能指出值得复查的具体案例，不能当作稳定发生率。

### 复核偏好方向，而不是制造通用 margin

对 4 个 pair 做一次盲审：p1 的 chosen 更符合事实和任务；p2 的 rejected 更好；p3 在目标 rubric 上打平；p4 缺少判断领域事实所需的来源。方向明确的 pair 中，chosen 胜出占比为 $1/(1+1)=1/2$；另有 1 个平局和 1 个无法判断。这个小账本暴露出一条反向标签，但不能估计整个数据集的质量。应按事实、安全、完整性和风格拆分原因，重标或保留不确定样本。

### 用反事实 pair 检查 reward 是否追逐表面形式

固定同一 prompt 和正确答案，只给短答案增加重复铺垫：短版 60 token，RM 分数 0.42；冗长版 240 token，RM 分数 0.88。独立 rubric 认为事实与任务完成相同，并更偏好简洁表达。这里可以直接比较同一个 RM 在配对变体上的排序；不能把 0.88 与另一套 1–5 人工分数相减。一个 pair 只是需要扩大抽查的线索，不是 reward 长度偏差的稳定估计。

### 手算 DPO，再检查 reference 和工具协议

设同一 prompt 下 policy 对 chosen/rejected 的序列 log probability 分别为 −1.2、−1.9，reference 对应值为 −1.4、−1.6。policy gap 是 $-1.2-(-1.9)=0.7$，reference gap 是 $-1.4-(-1.6)=0.2$，所以 $m=0.7-0.2=0.5$。若本次 $\beta=0.8$，则 $\beta m=0.4$，损失为 $-\log\sigma(0.4)=\log(1+e^{-0.4})\\approx0.513$。这个正 margin 只说明该 policy 相对 reference 更偏向这个 chosen；它不证明 chosen 事实正确。

本例的 log probability 是按 completion token 求和。若记录显示这些 reference log probability 来自 sft-v2，而训练配置声明 sft-v3，则这组 margin 不符合声明的 DPO 合同，必须用实际 reference 重算。若将序列求和改为 token 平均，长度不同的回答会改变 margin 和训练目标，不是只改日志展示方式。

训练 schema 若登记 search(query)、calculator(expression)，线上执行器却要求 web_search(query)、calculator(expr)，则工具名和参数字段都存在协议差异。模型能生成 JSON 不代表执行器能接受；应按版本化 schema 精确回放调用和结果状态。

### 把报告分成能采取行动的几类

- **可观测变化：** 指令切片上升，数学/代码/工具切片下降；小样本拒答统计、偏好盲审和 reward 反事实各自给出了具体异常候选。
- **已确认的协议问题：** assistant 目标缺失，prompt 与 padding 泄漏；DPO reference 版本不匹配；训练和线上工具 schema 不一致。
- **运行策略违规：** 本例没有提供本次任务预先声明的发布策略，因此不能判定策略违规，也不能临时发明统一门槛。
- **未知：** 这些样本是否代表目标人群、差值的不确定性、未覆盖的高风险行为，以及独立评审在更大样本上的一致性。

工程上的下一步是先修复 labels、reference 元数据和工具 schema，再用冻结的任务集重跑各切片；扩充低置信偏好与高 reward 反事实抽查，并记录实际 loss contract。最终是否继续训练或发布，由本次任务的风险负责人依据事先声明的策略判断，而不是把不同量纲的信号合成为一个默认通过分数。

## 6.16 后训练事故的恢复路径

修复后训练事故时，不要直接从最终 checkpoint 继续优化。建议分层处理：

1. 保存 base、SFT、偏好优化和最终版本，冻结受影响评估结果。
2. 重新抽样 SFT labels、偏好 pair、拒答边界和工具样本；标明 shift 后目标和 reduction contract。
3. 先修复数据、template、mask、reference 版本和 schema，再讨论训练超参。
4. 对 reward 做反事实测试和独立人工审查，隔离高 reward 但不符合任务 rubric 的样本。
5. 对 DPO/RLHF 做 reference、beta、KL、长度和多维任务对照。
6. 在固定能力回归集上选择 checkpoint，并报告各切片结果和不确定性，不只看训练 loss。
7. 为本次具体任务预先声明风险、可接受回退和硬性停止条件；将策略违规与观测指标分开报告。
8. 通过真实任务、人工抽检和受控灰度观察成本、延迟、采用和安全。

最佳 checkpoint 可能在训练中间。训练更久可能继续放大偏好噪声、reward shortcut 或领域过拟合，但不应把训练时长本身当成故障判据。恢复决策必须回到本次任务合同；没有声明策略时，不能从一组指标自动推出通用“通过”结论。

## 6.17 常见但危险的处理方式

### 只看聊天体验

聊天自然不代表数学、代码、事实、工具和安全边界正确。后训练前后必须有固定多维回归。

### 安全训练只追求更高拒答率

拒答率上升可能来自误拒增加。安全评估必须同时覆盖安全回答、危险拒绝、替代帮助、语言切片和任务完成。

### 偏好 pair 越多越好

低质量、低间隔和方向错误的 pair 会增加噪声。先做 pair 质量和标注一致性，再扩大数量。

### Reward 越高越好

reward 可能偏好长度、模板和免责声明。高 reward 样本必须做人审和反事实检查。

### DPO loss 下降就算成功

DPO loss 只反映相对 reference 的优化目标。reference、tokenizer、template、pair 和 beta 错了，loss 仍可能下降。

### 训练工具调用不做线上 schema 回放

工具名、字段、枚举、call ID 和 observation 任何一个不一致，模型可能生成可解析但不可执行的请求。训练和执行器必须共享版本化 schema。

## 6.18 资料与证据边界

本章参考以下资料：

- InstructGPT 论文用于说明 SFT、奖励模型和人类反馈的训练链路；论文结果不能直接推出当前应用的安全或业务效果。
- PPO、DPO 及其后续偏好优化论文用于说明 reference、偏好 pair、策略变化和优化目标；具体实现需要绑定框架版本和损失细节。
- Hugging Face Transformers chat template、TRL SFT/DPO/GRPO 文档用于公开 API、数据字段和训练接口边界；默认 collator 和版本行为应通过样本回放验证。
- Reward Model、RLHF reward hacking 和 process preference 的研究用于解释奖励捷径、长度偏差和人工质量差距；任何阈值都需要在目标任务上校准。
- OpenAI Evals、NIST AI RMF 和 OWASP 资料用于多维评估、风险和持续治理的框架背景，不替代人工安全审查或目标系统实测。

闭源模型的内部后训练配方通常没有公开。公开 API 的 `reasoning`、`effort`、`refusal` 或工具字段不能用来反推训练数据、reward model、RL 算法或内部 verifier。正文只写论文、官方文档、模型卡和本地实验能够支持的内容。

## 6.19 本章小结

后训练不是把基础模型统一推向“更好”，而是在新的数据、偏好和约束下重塑行为。SFT 需要先保证 labels 和模板正确；能力评估需要发现数学、代码、知识、长上下文和工具回归；安全训练要同时控制误拒和漏拒；偏好数据需要方向、间隔和标注一致性；Reward Model 要接受人工质量和反事实挑战；RLHF 要防止 reward hacking；DPO 要保证 reference、tokenizer、template、beta 和 pair 的一致。

当后训练报告能同时回答“模型学了什么、忘了什么、为什么拒绝、为什么变长、reward 是否可信、工具是否仍能执行、哪个版本引入了变化”，训练结果才具有工程意义。任何单一 loss、reward 或聊天分数，都不能替代这条行为证据链。
