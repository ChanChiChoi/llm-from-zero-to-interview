# 第七章：Synthetic Data 与 Distillation Data

前面几章讨论的大多是自然产生的数据：网页、代码仓库、数学题、专业文档、论坛、书籍和论文。本章讨论另一类越来越重要的数据：合成数据和蒸馏数据。

合成数据是由规则、程序、模拟器或模型生成的数据。蒸馏数据通常是强模型作为 teacher，为弱模型或小模型生成回答、解释、偏好或推理轨迹，让 student 模型学习 teacher 的行为。

这类数据在今天的大模型训练中非常常见。指令微调、数学推理、代码训练、安全训练、多语言增强、领域适配、工具调用训练，都大量使用合成数据或 teacher-generated data。

但合成数据不是免费午餐。它可以快速扩充能力，也可能制造同质化、错误放大、评估污染、teacher 偏差继承和数据退化。

本章讨论合成数据和蒸馏数据的训练、评估、治理和风险控制。合成数据不是绕过模型服务条款、复制专有模型能力、生成有害内容或规避安全策略的方法；任何 teacher 输出都必须有明确授权、可追踪来源和适用范围。

## 0. 合成数据是可审计的分布编辑

合成数据工程的核心不是一次生成多少文本，而是把目标能力、seed、生成器、验证器、配比和训练结果连成一条可回放的链路。可以把这条链路写成：

~~~text
目标能力 -> seed 数据 -> teacher/规则生成 -> 验证 -> 去重 -> 安全与污染扫描 -> 配比 -> 小规模训练实验 -> 版本审计
~~~

Self-Instruct、WizardLM/Evol-Instruct、phi-1、Orca、经典 knowledge distillation、Distilling Step-by-Step 以及 model collapse 相关研究提供了不同证据层面的入口。本章会区分这些论文真正展示的实验结果与工程上的教学抽象，并进一步讨论 teacher 授权、prompt/sampling 记录、正确性验证、自然数据锚点和数据退化。

---

## 1. 先建立直觉：为什么要合成数据？

自然数据有两个问题：不够可控，也不一定覆盖你想要的能力。

例如，你想训练模型学会复杂指令跟随。互联网上当然有问答和教程，但不一定有大量“清晰指令 -> 高质量回答”的样本。你想训练数学分步推理，网页上有题目和答案，但步骤可能缺失、错误或格式混乱。你想训练安全拒答，自然语料里可能有风险内容，但不一定有安全、合规、边界清晰的回答。

合成数据的价值在于：它可以按目标能力主动构造训练样本。

可以把合成数据想成“定制练习册”。自然数据像从世界各地收来的书，有价值但混杂；合成数据像老师根据学生短板编的练习，目标明确、格式统一、覆盖可控。

但练习册质量取决于出题老师。如果老师水平高、覆盖广、验证严格，练习册很有用。如果老师重复、偏科、答案错误，学生会被带偏。

---

## 2. 来龙去脉：从数据增强到大模型自举

合成数据不是大模型时代才有。传统机器学习里早就有数据增强：图像旋转裁剪、语音加噪、机器翻译回译、规则生成样本等。这些方法的目标是增加数据量、提升鲁棒性、覆盖边界情况。

大模型时代，合成数据的形态发生了变化。模型本身可以生成高质量文本、代码、题目、解释和对话，于是数据生成不再只是简单扰动，而是能力自举。

Self-Instruct 是一个代表性工作。它用模型生成 instructions、inputs 和 outputs，过滤无效或相似样本，再用这些合成指令数据微调模型，从而提升 instruction-following 能力。它说明：即使缺少大量人工指令数据，也可以通过模型生成和过滤构造可用训练数据。

WizardLM/Evol-Instruct 进一步强调复杂指令生成。它从初始指令出发，通过逐步改写生成更复杂的指令，用于增强模型处理复杂任务的能力。

phi-1 则展示了合成教材和练习在代码模型上的价值。它使用 textbook-quality web 数据和 GPT-3.5 生成的合成教材/练习，让小规模代码模型获得较强能力。这说明合成数据不仅能补数量，也能塑造能力结构。

---

## 3. Synthetic data 和 distillation data 的区别

这两个概念有重叠，但关注点不同。

Synthetic data 强调数据是人工或模型生成的，不是自然采集来的。生成方式可以是规则、程序、模拟器或 LLM。

Distillation data 强调 teacher-student 关系。强模型生成回答、解释、偏好或分数，弱模型通过这些数据学习 teacher 的行为。

例子：

1. 用规则生成 10 万道算术题：synthetic data，但不一定是 distillation。
2. 用 GPT-4 给指令生成高质量答案：既是 synthetic data，也是 distillation data。
3. 用模拟器生成机器人状态轨迹：synthetic data。
4. 用强代码模型为问题生成参考实现和解释：distillation data。
5. 用 teacher 模型给多个回答排序：偏好蒸馏数据。

合成数据关注“数据来源是否生成”，蒸馏数据关注“是否存在 teacher 到 student 的能力迁移”；两者可以重叠，但不应混作同一个数据标签。

### 3.1 关键公式与审计指标

训练数据可以分成自然数据、合成数据和蒸馏数据：

~~~math
D = D_nat union D_syn union D_distill
~~~

一条合成或蒸馏样本可以表示为：

~~~math
s_i = (x_i, y_i, a_i, t_i, p_i, q_i, r_i, z_i)
~~~

其中 `x_i` 是指令或上下文，`y_i` 是生成答案或轨迹，`a_i` 是目标能力标签，`t_i` 是 teacher / generator 标识，`p_i` 是 prompt 和采样参数版本，`q_i` 是质量分，`r_i` 是风险分，`z_i` 是授权、验证、去重、污染和版本元数据。

如果是普通 SFT 合成样本，目标仍是条件语言建模：

~~~math
L_sft = -(1 / N_tok) * sum_i(w_i * sum_j(log p_theta(y_i_j | x_i, y_i_<j)))
~~~

其中 `w_i` 是样本权重，`N_tok` 是有效训练 token 数。`w_i` 不应该只由 teacher 分数决定，还要叠加验证、风险、重复和多样性审计。

如果 teacher 提供 soft distribution，蒸馏损失常写成 KL 形式：

~~~math
L_kd = tau^2 * sum_i(KL(p_T_tau(. | c_i) || p_S_tau(. | c_i)))
~~~

其中 `p_T` 是 teacher 分布，`p_S` 是 student 分布，`tau` 是 distillation temperature，`c_i` 是上下文。如果只有 teacher 的 hard answer，则退化为对 teacher output 做 supervised learning。

合成样本质量分可以写成可审计的加权形式：

~~~math
q_i = w_v*V_i + w_e*E_i + w_d*D_i + w_s*S_i - lambda_h*H_i - lambda_u*U_i - lambda_c*C_i
~~~

其中 `V_i` 是正确性验证，`E_i` 是证据或测试支持，`D_i` 是多样性贡献，`S_i` 是安全边界通过情况，`H_i` 是 hallucination 风险，`U_i` 是近重复或模板化风险，`C_i` 是评测污染风险。

合成数据样本的状态不宜压缩成一个准入开关，可以保留并列的检查量：

~~~math
checks_i = (quality_ok_i, authorized_i, verified_i, dedup_ok_i, safe_i, contamination_clear_i, privacy_clear_i)
~~~

这里 `authorized_i` 表示 teacher 输出或生成规则的使用授权成立；`verified_i` 表示答案、代码、引用、安全边界或工具参数通过验证；`contamination_clear_i` 表示没有命中评测污染。不同失败原因对应隔离、重生成、人工复核、脱敏或删除等不同动作。

合成和蒸馏数据在训练集合中的 token 占比为：

~~~math
R_syn = sum_{i:o_i in {syn, distill}}(keep_i * T_i) / sum_i(keep_i * T_i)
~~~

其中 `o_i` 是样本来源类型，`T_i` 是 token 数。`R_syn` 过高时，要警惕同质化、teacher 偏差和 model collapse 风险。

这些比例的定义域必须写清楚：只有保留集合中的 token 总数大于零时，`R_syn` 才有定义；只有目标标签集合非空时，`C_cover` 才有定义；只有存在样本时，`R_dup` 才有定义。没有保留样本时应记录为 `undefined`，而不是把“没有可测分母”写成合成占比 `0` 或覆盖率 `0`。如果原始数据存在但经过过滤后一个样本也没有，保留率可以是 `0`，同时还必须触发补充或恢复数据的动作。

能力覆盖可以按标签集合计算：

~~~math
C_cover = |A_target intersect A_kept| / |A_target|
~~~

同质化或近重复率可以写成：

~~~math
R_dup = sum_i(I(max_{j<i}(sim(s_i, s_j)) > tau_sim)) / n
~~~

最终训练前可以保留一个配比和风险向量：

~~~math
C_syn = (R_syn, C_cover, R_dup, R_risk)
~~~

它分别描述合成/蒸馏 token 占比、目标标签覆盖、重复率和风险率。合成数据不是“便宜 token”，而是对训练分布的主动编辑；因此必须同时记录 teacher、prompt、sampling、验证、过滤、配比和训练效果。

---

## 4. 合成数据常见类型

大模型训练中的合成数据主要有以下类型：

1. 合成指令数据：instruction、input、output 三元组。
2. 合成对话数据：多轮用户和助手对话。
3. 合成数学数据：题目、解题步骤、答案、验证标签。
4. 合成代码数据：需求、代码、测试、解释、bug fix。
5. 合成领域数据：医学、法律、金融等场景问答和文档问答。
6. 合成安全数据：风险识别、拒答、边界解释、合规替代建议。
7. 合成工具调用数据：用户意图、工具选择、参数填充、结果总结。
8. 合成多语言数据：翻译、跨语言问答、低资源语言增强。
9. 合成偏好数据：chosen/rejected 回答对。
10. 合成评估数据：用于内部评测和回归测试的任务集。

不同类型的数据目标不同，生成和验证方法也不同。

---

## 5. 合成指令数据

指令数据是 instruction tuning 的核心。它让模型从“续写文本”转向“理解用户要求并完成任务”。

一条典型指令样本包括：

1. instruction：用户要求。
2. input：可选上下文。
3. output：理想回答。

合成指令数据的优势：

1. 可以快速扩大任务覆盖。
2. 可以补充长尾任务。
3. 可以控制格式和风格。
4. 可以构造不同难度。
5. 可以多语言扩展。

风险：

1. 指令类型重复。
2. 回答风格同质化。
3. teacher 幻觉被 student 学到。
4. 难度分布不真实。
5. 过度训练后模型变得模板化。

Self-Instruct 的关键启发不是“让模型随便生成数据”，而是“生成后必须过滤无效、重复和相似样本，并用评估验证收益”。

---

## 6. Evol-Instruct：让指令变复杂

普通合成指令容易停留在简单任务，例如总结、翻译、改写、列要点。模型如果只训练这些数据，面对复杂约束、多步骤任务、组合任务时容易失败。

Evol-Instruct 的思想是从简单指令出发，通过改写逐步提高复杂度，例如增加限制条件、增加推理步骤、增加输入复杂度、要求多角度分析或引入更具体场景。

它解决的问题是指令复杂度不足。

但复杂不等于高质量。复杂指令也可能不合理、不可回答、约束冲突或答案质量差。因此需要过滤和审计：

1. 指令是否清楚。
2. 约束是否一致。
3. 答案是否满足约束。
4. 难度是否真实有用。
5. 是否覆盖多种能力，而不是只会堆限制。

---

## 7. 推理轨迹数据

推理轨迹数据包含中间思路、步骤、证明、计算过程或解释。它在数学、代码、科学问答、规划和复杂决策中很重要。

推理轨迹的价值：

1. 给模型提供分步解决问题的模式。
2. 提升复杂任务可解释性。
3. 让模型学会检查中间结果。
4. 支持 verifier 或 judge 训练。
5. 可用于生成过程监督数据。

但推理轨迹也有明显风险：

1. 错误步骤会被学习。
2. 模型可能学会“看起来合理”的伪推理。
3. 轨迹可能过长，挤占有效上下文。
4. 不同任务不一定需要显式长推理。
5. 如果来自 teacher，student 可能模仿 teacher 的偏差和风格。

所以推理轨迹数据一定要重视正确性验证。数学题可以校验最终答案，代码题可以运行测试，领域题需要引用依据或专家抽检。

---

## 8. Teacher-student 蒸馏

蒸馏的基本思想是：用强模型教弱模型。

在大模型数据工程中，teacher 可以提供：

1. 高质量回答。
2. 分步解释。
3. 多候选答案。
4. 答案评分。
5. chosen/rejected 偏好对。
6. 错误分析。
7. 任务分解。
8. 工具调用轨迹。

student 通过监督微调或偏好训练学习这些数据。

蒸馏的好处：

1. 降低人工标注成本。
2. 把强模型行为迁移到小模型。
3. 构造更稳定的回答格式。
4. 补足长尾任务。
5. 支持私有模型能力定制。

蒸馏的风险：

1. teacher 的错误被继承。
2. teacher 的风格被过度模仿。
3. student 难以超过 teacher 的知识边界。
4. 可能违反 teacher 服务或数据使用条款。
5. 多代蒸馏可能导致数据分布退化。

工程上必须确认 teacher 输出的使用授权和合规边界。

---

## 9. Distillation data 不等于复制能力

蒸馏不是简单“复制某个闭源模型”。合规项目要关注授权、数据使用条款、输出归属和安全限制；技术上 student 学到的是经过筛选的行为信号，而不是 teacher 的参数或全部能力。

从技术角度看，蒸馏也不是直接复制参数。student 学到的是训练数据中的行为模式，受 student 容量、训练策略、数据覆盖、优化目标和评估体系限制。

合理的蒸馏场景包括：

1. 用自有 teacher 为自有 student 生成训练数据。
2. 使用授权允许的数据和模型输出。
3. 用专家审核后的 teacher 数据增强内部模型。
4. 用 teacher 辅助标注、评分和质量过滤。

不合理的做法包括无授权大规模复制专有模型输出、绕过限制获取数据或生成违反安全边界的内容。本书不讨论这类做法。

---

## 10. 合成数据生成 pipeline

一个成熟的合成数据 pipeline 通常包括：

1. 目标定义：要补什么能力。
2. 种子数据：初始任务、领域文档、题型、用户场景。
3. 生成策略：prompt、模板、规则、模型、采样参数。
4. 多样性控制：任务类型、难度、语言、领域、格式。
5. 初步过滤：去重、长度、格式、无效样本、明显错误。
6. 质量评分：LLM judge、规则校验、模型评分、人工抽样。
7. 正确性验证：测试、计算、引用、事实核验、专家审计。
8. 安全过滤：风险分类、PII、政策边界、拒答质量。
9. 配比控制：不要让合成数据淹没自然数据。
10. 训练实验：小规模 ablation 验证收益和副作用。
11. 版本化：记录 teacher、prompt、参数、过滤器和数据版本。

这条链路的核心不是“生成很多”，而是“生成、过滤、验证、评估、迭代”。

---

## 11. 质量验证：合成数据最关键的一步

合成数据的问题不是不能生成，而是太容易生成。真正难的是验证。

不同任务的验证方法不同：

1. 数学题：校验答案、步骤、单位、边界条件。
2. 代码题：运行单测、静态检查、编译、lint。
3. 事实问答：检索证据、引用来源、人工抽检。
4. 翻译：双向一致性、人工评估、术语一致性。
5. 安全回答：边界是否正确，是否误拒或漏拒。
6. 工具调用：参数是否正确，工具结果是否被正确使用。
7. 领域问答：专家审计和权威资料对照。

如果验证缺失，合成数据很容易把 hallucination 包装成训练信号。

---

## 12. 去重和多样性控制

合成数据经常高度重复。模型会生成相似指令、相似开头、相似答案结构和相似解释模板。

常见重复包括：

1. 指令重复。
2. 任务类型重复。
3. 回答模板重复。
4. 推理步骤重复。
5. 领域场景重复。
6. 安全拒答话术重复。

控制方法包括：

1. embedding 聚类去重。
2. n-gram overlap 过滤。
3. 按任务类型分桶采样。
4. 控制难度分布。
5. 多 teacher 或多 prompt 生成。
6. 人工审计高频模式。
7. 对同质模板降权。

多样性不是为了好看，而是为了避免 student 学到单一风格。

---

## 13. 数据退化风险

数据退化指模型生成的数据逐渐替代真实数据后，训练分布变窄、错误累积、长尾信息消失、语言风格同质化。

如果一代模型用自然数据训练，第二代大量用第一代生成数据，第三代再用第二代生成数据，可能出现信息损失。真实世界的复杂性被逐步压缩成模型喜欢生成的模式。

表现包括：

1. 语言更模板化。
2. 观点更平均。
3. 长尾知识减少。
4. 错误事实被重复强化。
5. 难题覆盖下降。
6. 安全边界变得机械。
7. 多语言和小众表达被削弱。

防止退化的关键是保留高质量自然数据、人工数据、真实用户分布、权威来源和严格验证。合成数据应是补充和定向增强，而不是无节制替代真实数据。

---

## 14. 合成数据在不同训练阶段的用法

### 14.1 预训练

预训练阶段可以少量使用高质量合成教材、代码练习、数学题和领域文本，但要谨慎控制比例，避免合成分布主导 base model。

### 14.2 继续预训练

继续预训练适合用合成数据补目标领域或能力短板，例如代码教材、数学练习、多语言问答。

### 14.3 SFT

SFT 是合成指令数据最常用的阶段。instruction-response、multi-turn dialogue、tool use、format following 都可以合成。

### 14.4 偏好训练

可以用 teacher 或 judge 生成 chosen/rejected 数据，但偏好质量必须审计，否则会让模型学到表面偏好。

### 14.5 安全训练

安全合成数据可以覆盖风险类别、拒答边界、误拒案例和安全替代建议。此类数据必须保持防御和合规导向。

---

## 15. 合成数据和真实用户数据

真实用户数据反映真实需求、真实表达和真实错误，但涉及隐私、授权和安全治理。合成数据可控、便宜、可扩展，但可能不真实、不多样。

两者不是替代关系。

真实用户数据适合发现真实分布和产品问题；合成数据适合针对性补齐覆盖不足、构造安全边界、生成难例和统一格式。

成熟做法是：用真实数据发现任务分布，用合成数据扩展和补齐，再用人工和在线反馈验证。

---

## 16. 合成数据配比

合成数据占比没有标准答案。不同阶段、任务和质量下差异很大。

判断合成数据比例时要看：

1. 目标能力是否稀缺。
2. 合成数据质量是否可验证。
3. 是否覆盖真实用户分布。
4. 是否和自然数据重复。
5. 是否导致风格同质化。
6. 是否提升目标指标但损害通用能力。
7. 是否增加幻觉、安全或记忆风险。

稳妥策略是从小比例开始，通过 ablation 逐步提高，观察收益曲线和副作用，而不是一次性大规模混入。

---

## 17. 机制与边界：合成数据是分布编辑

从机制上看，合成数据不是单纯增加样本，而是在编辑训练分布。

自然数据来自世界分布，合成数据来自生成器分布。生成器分布由 teacher 能力、prompt、采样参数、过滤器、任务模板和审计标准共同决定。

因此，合成数据引入的是一种新的偏置。如果这种偏置与目标能力一致，模型会受益；如果偏置过强，模型会被塑造成“像生成器一样说话”。

专家级问题包括：

1. 生成器分布和真实用户分布差多少？
2. 合成数据是否覆盖目标任务的长尾？
3. teacher 错误如何被发现和阻断？
4. student 是否只是模仿格式，而没有获得能力？
5. 合成数据提升 benchmark 是否来自污染或模板匹配？
6. 多代合成是否导致数据退化？

---

## 18. 一个可落地的合成与蒸馏数据方案

一个合成与蒸馏数据系统首先要回答“补什么能力”。指令跟随、数学、代码、工具调用、多语言、安全和领域问答需要不同的 seed、生成方式和验证器；如果目标没有拆开，后面的质量分数只能把互不相同的样本混在一起。

seed 可以来自真实任务、公开高质量数据、专家模板、领域文档或人工设计任务。规则和程序适合生成可计算、可执行、可验证的样本；LLM 适合生成开放式指令、解释和对话；teacher-student 蒸馏适合迁移回答格式、候选排序和任务分解。每条记录都要绑定 teacher 或 generator、prompt、采样参数和授权版本。

生成之后先控制多样性和结构，再做格式、长度、重复、污染和明显错误过滤。数学用答案和步骤校验，代码用编译与测试，事实用检索和引用，领域样本用专家抽检，安全样本用策略和误拒/漏拒评估。一个 LLM judge 可以作为候选信号，但不能单独充当真值。

保留样本按任务、难度、风险、语言、领域、teacher 和 prompt 版本分桶。小规模训练需要比较不同合成比例、teacher、过滤阈值和自然数据锚点，观察目标收益是否伴随同质化、幻觉、污染、风格偏移或通用能力下降。最终版本必须保存生成日志、拒绝原因、授权记录、评估结果和训练效果，让数据问题能够回放。

### 18.1 最小可运行合成与蒸馏数据审计 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy natural / synthetic / distillation 样本；输出包括保留样本、拒绝原因、来源配比、任务配比、teacher 配比、多样性覆盖和验收结果。

它演示的是合成数据治理机制，不是生产级合规审查、LLM judge、数学 verifier、代码沙箱、安全分类器或版权系统。真实系统要接入授权审查、teacher 版本管理、prompt registry、测试执行、检索证据、人工抽样、污染检测和训练 ablation。

~~~python
from collections import Counter, defaultdict


samples = [
    {"id": "seed_user_math", "origin": "natural", "task": "math", "tokens": 420, "quality": 0.86, "validated": True, "diversity": {"math", "seed"}, "teacher": None, "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": True},
    {"id": "seed_user_domain", "origin": "natural", "task": "domain", "tokens": 1200, "quality": 0.84, "validated": True, "diversity": {"domain", "seed"}, "teacher": None, "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": True},
    {"id": "syn_math_verified", "origin": "synthetic", "task": "math", "tokens": 620, "quality": 0.88, "validated": True, "diversity": {"math", "reasoning"}, "teacher": "rule_solver", "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "syn_math_wrong", "origin": "synthetic", "task": "math", "tokens": 560, "quality": 0.83, "validated": False, "diversity": {"math"}, "teacher": "rule_solver", "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "distill_tool_trace", "origin": "distill", "task": "tool", "tokens": 780, "quality": 0.91, "validated": True, "diversity": {"tool", "multi_step"}, "teacher": "owned_teacher", "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "distill_unauthorized", "origin": "distill", "task": "general", "tokens": 700, "quality": 0.90, "validated": True, "diversity": {"general"}, "teacher": "restricted_api", "authorized": False, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "syn_code_tests", "origin": "synthetic", "task": "code", "tokens": 900, "quality": 0.87, "validated": True, "diversity": {"code", "tests"}, "teacher": "owned_teacher", "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "syn_code_duplicate", "origin": "synthetic", "task": "code", "tokens": 880, "quality": 0.86, "validated": True, "diversity": {"code", "tests"}, "teacher": "owned_teacher", "authorized": True, "duplicate": True, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "syn_benchmark_leak", "origin": "synthetic", "task": "eval_like", "tokens": 500, "quality": 0.84, "validated": True, "diversity": {"eval"}, "teacher": "owned_teacher", "authorized": True, "duplicate": False, "contam": True, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "syn_safety_refusal", "origin": "synthetic", "task": "safety", "tokens": 640, "quality": 0.82, "validated": True, "diversity": {"safety", "boundary"}, "teacher": "policy_template", "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": False, "natural_anchor": False},
    {"id": "syn_unsafe_answer", "origin": "synthetic", "task": "safety", "tokens": 610, "quality": 0.74, "validated": True, "diversity": {"safety"}, "teacher": "weak_teacher", "authorized": True, "duplicate": False, "contam": False, "safety_ok": False, "pii": False, "natural_anchor": False},
    {"id": "real_user_private", "origin": "natural", "task": "domain", "tokens": 540, "quality": 0.78, "validated": True, "diversity": {"domain"}, "teacher": None, "authorized": True, "duplicate": False, "contam": False, "safety_ok": True, "pii": True, "natural_anchor": True},
]

TARGET_TAGS = {"math", "reasoning", "tool", "multi_step", "code", "tests", "safety", "boundary", "domain", "seed"}
MIN_QUALITY = 0.80
MAX_SYN_RATIO = 0.72


def safe_ratio(numerator, denominator):
    return round(numerator / denominator, 3) if denominator else None


def reject_reason(item):
    if not item["authorized"]:
        return "unauthorized_teacher"
    if item["pii"]:
        return "privacy_or_pii"
    if item["contam"]:
        return "eval_contamination"
    if item["duplicate"]:
        return "near_duplicate"
    if not item["safety_ok"]:
        return "unsafe_or_policy_fail"
    if not item["validated"]:
        return "unverified_output"
    if item["quality"] < MIN_QUALITY:
        return "low_quality"
    return None


kept, rejected = [], {}
for item in samples:
    reason = reject_reason(item)
    if reason:
        rejected[item["id"]] = reason
    else:
        kept.append(item)

raw_tokens = sum(item["tokens"] for item in samples)
kept_tokens = sum(item["tokens"] for item in kept)
origin_tokens = defaultdict(int)
task_tokens = defaultdict(int)
teacher_tokens = defaultdict(int)
covered_tags = set()

for item in kept:
    origin_tokens[item["origin"]] += item["tokens"]
    task_tokens[item["task"]] += item["tokens"]
    teacher_tokens[item["teacher"] or "human_or_seed"] += item["tokens"]
    covered_tags.update(item["diversity"])

synthetic_like_tokens = origin_tokens["synthetic"] + origin_tokens["distill"]
report = {
    "kept_ids": [item["id"] for item in kept],
    "rejected": dict(sorted(rejected.items())),
    "reason_counts": dict(sorted(Counter(rejected.values()).items())),
    "retention": safe_ratio(kept_tokens, raw_tokens),
    "origin_mix": {k: safe_ratio(origin_tokens[k], kept_tokens) for k in sorted(origin_tokens)},
    "task_mix": {k: safe_ratio(task_tokens[k], kept_tokens) for k in sorted(task_tokens)},
    "teacher_mix": {k: safe_ratio(teacher_tokens[k], kept_tokens) for k in sorted(teacher_tokens)},
    "synthetic_like_ratio": safe_ratio(synthetic_like_tokens, kept_tokens),
    "diversity_coverage": safe_ratio(
        len(covered_tags & TARGET_TAGS), len(TARGET_TAGS)
    ),
}

checks = {
    "unauthorized_teacher_blocked": "unauthorized_teacher" in report["reason_counts"],
    "unverified_output_blocked": "unverified_output" in report["reason_counts"],
    "contamination_blocked": "eval_contamination" in report["reason_counts"],
    "unsafe_output_blocked": "unsafe_or_policy_fail" in report["reason_counts"],
    "privacy_blocked": "privacy_or_pii" in report["reason_counts"],
    "diversity_ok": report["diversity_coverage"] >= 0.85,
    "synthetic_ratio_ok": (
        report["synthetic_like_ratio"] is not None
        and report["synthetic_like_ratio"] <= MAX_SYN_RATIO
    ),
    "natural_anchor_present": any(
        item["origin"] == "natural" and item["natural_anchor"] for item in kept
    ),
}
signals = {
    "unauthorized_teacher_not_blocked": not checks["unauthorized_teacher_blocked"],
    "unverified_output_not_blocked": not checks["unverified_output_blocked"],
    "contamination_not_blocked": not checks["contamination_blocked"],
    "unsafe_output_not_blocked": not checks["unsafe_output_blocked"],
    "privacy_not_blocked": not checks["privacy_blocked"],
    "diversity_gap": not checks["diversity_ok"],
    "synthetic_ratio_too_high": not checks["synthetic_ratio_ok"],
    "natural_anchor_missing": not checks["natural_anchor_present"],
    "no_retained_tokens": kept_tokens == 0,
}
actions = []
if signals["unauthorized_teacher_not_blocked"]:
    actions.append("stop_and_review_teacher_authorization")
if signals["unverified_output_not_blocked"]:
    actions.append("quarantine_unverified_generation")
if signals["contamination_not_blocked"]:
    actions.append("quarantine_eval_overlap_and_retest")
if signals["unsafe_output_not_blocked"]:
    actions.append("remove_policy_violating_samples")
if signals["privacy_not_blocked"]:
    actions.append("deidentify_or_remove_sensitive_samples")
if signals["diversity_gap"]:
    actions.append("add_task_and_style_diversity")
if signals["synthetic_ratio_too_high"]:
    actions.append("reduce_generated_ratio_and_add_natural_anchors")
if signals["natural_anchor_missing"]:
    actions.append("restore_natural_data_anchor")
if signals["no_retained_tokens"]:
    actions.append("restore_or_collect_generation_records")
decision = "continue_to_ablation" if not actions else "hold_for_repair"
report["checks"] = checks
report["signals"] = signals
report["actions"] = actions
report["decision"] = decision

for key, value in report.items():
    print(f"{key}=", value)

assert report["kept_ids"] == [
    "seed_user_math",
    "seed_user_domain",
    "syn_math_verified",
    "distill_tool_trace",
    "syn_code_tests",
    "syn_safety_refusal",
]
assert report["reason_counts"] == {
    "eval_contamination": 1,
    "near_duplicate": 1,
    "privacy_or_pii": 1,
    "unauthorized_teacher": 1,
    "unsafe_or_policy_fail": 1,
    "unverified_output": 1,
}
assert report["retention"] == 0.546
assert report["origin_mix"] == {"distill": 0.171, "natural": 0.355, "synthetic": 0.474}
assert report["synthetic_like_ratio"] == 0.645
assert report["diversity_coverage"] == 1.0
assert all(report["checks"].values())
assert not any(report["signals"].values())
assert report["actions"] == []
assert report["decision"] == "continue_to_ablation"
assert safe_ratio(0, 0) is None
assert safe_ratio(0, 100) == 0.0
assert report["checks"]["natural_anchor_present"] is True
~~~

运行后会看到类似输出：

~~~text
kept_ids= ['seed_user_math', 'seed_user_domain', 'syn_math_verified', 'distill_tool_trace', 'syn_code_tests', 'syn_safety_refusal']
rejected= {'distill_unauthorized': 'unauthorized_teacher', 'real_user_private': 'privacy_or_pii', 'syn_benchmark_leak': 'eval_contamination', 'syn_code_duplicate': 'near_duplicate', 'syn_math_wrong': 'unverified_output', 'syn_unsafe_answer': 'unsafe_or_policy_fail'}
reason_counts= {'eval_contamination': 1, 'near_duplicate': 1, 'privacy_or_pii': 1, 'unauthorized_teacher': 1, 'unsafe_or_policy_fail': 1, 'unverified_output': 1}
retention= 0.546
origin_mix= {'distill': 0.171, 'natural': 0.355, 'synthetic': 0.474}
task_mix= {'code': 0.197, 'domain': 0.263, 'math': 0.228, 'safety': 0.14, 'tool': 0.171}
teacher_mix= {'human_or_seed': 0.355, 'owned_teacher': 0.368, 'policy_template': 0.14, 'rule_solver': 0.136}
synthetic_like_ratio= 0.645
diversity_coverage= 1.0
checks= {'unauthorized_teacher_blocked': True, 'unverified_output_blocked': True, 'contamination_blocked': True, 'unsafe_output_blocked': True, 'privacy_blocked': True, 'diversity_ok': True, 'synthetic_ratio_ok': True, 'natural_anchor_present': True}
signals= {'unauthorized_teacher_not_blocked': False, 'unverified_output_not_blocked': False, 'contamination_not_blocked': False, 'unsafe_output_not_blocked': False, 'privacy_not_blocked': False, 'diversity_gap': False, 'synthetic_ratio_too_high': False, 'natural_anchor_missing': False}
actions= []
decision= continue_to_ablation
~~~

这个 demo 的重点是：合成数据进入训练前必须被当成可审计数据产品，而不是 teacher 随手生成的文本。它要能证明授权成立、错误被验证拦截、污染被隔离、PII 被过滤、近重复被降掉，并且合成 / 蒸馏 token 没有压过自然数据锚点。`safe_ratio(0, 0)` 返回 `None`，表示没有可解释的 token 分母；如果原始数据存在但最终一个样本也没有，保留率才是 `0.0`，并且系统应要求恢复或补充生成数据。

---

## 19. 决策边界：合成和蒸馏数据的几个判断

### 19.1 “生成”与“蒸馏”描述的是不同维度

synthetic data 描述来源：数据由规则、程序、模拟器或模型生成；distillation data 描述关系：teacher 产生行为信号，student 学习这些信号。一份 teacher 生成的答案可以同时属于两者，但规则生成的算术题只属于前者。数据 schema 应分别保存 origin、teacher 和 generator，而不是用一个标签覆盖两个维度。

### 19.2 Self-Instruct 的关键不在于生成数量

Self-Instruct 的工程价值在于从 seed 指令扩展任务，再过滤无效、重复和相似样本，最后用训练实验验证 instruction-following 是否改善。若只有生成步骤而没有过滤、验证和自然数据对照，得到的只是大量未经审计的文本。

### 19.3 生成器越强，验证越不能省略

强 teacher 可以提高平均质量，也会把错误、偏见、风格和不确定性以更有说服力的形式传播。数学、代码、事实、领域和安全样本应分别接入验证器；LLM judge 只能作为一个带误差的信号，不能替代独立证据。

### 19.4 合成比例如何判断

没有跨任务固定比例。需要同时观察合成/蒸馏 token 占比、自然数据锚点、目标能力覆盖、重复率、事实性、安全、改写题和人工样例。若某项 benchmark 提升伴随模板化或污染命中，比例增加并不代表真实能力增加。

### 19.5 蒸馏的能力边界

student 通常受 teacher 数据覆盖、student 容量和训练目标限制，很难在 teacher 的全部能力范围内全面超过 teacher。但在明确任务分布、严格筛选数据、加入额外自然数据或更合适的验证目标后，student 可能在局部指标上超过 teacher。这个结论必须绑定任务、评估集和训练条件。

### 19.6 为什么必须保留自然数据锚点

自然数据提供真实用户表达、长尾现象和生成器没有见过的分布。合成数据可以补齐短板，却不能独立代表现实；没有自然锚点时，模型可能越来越像生成器，而不是更像用户和世界。

### 19.7 授权、隐私和安全是数据字段而不是附注

teacher 版本、输出使用条款、隐私处理、风险类别和安全策略需要进入每条记录的血缘。把这些信息只写在项目说明里，无法在过滤、抽样、删除请求和版本回放时定位具体样本。

---

## 20. 常见误区

误区一：合成数据越多越好。

合成数据容易生成，但不一定高质量。过多会导致同质化和数据退化。

误区二：teacher 强，生成数据就一定好。

teacher 也会 hallucinate，也有偏差。生成数据必须验证。

误区三：蒸馏就是复制模型。

蒸馏是通过数据学习 teacher 行为，不等于复制参数或完整能力，还必须遵守授权和使用边界。

误区四：合成数据可以替代真实数据。

合成数据适合补齐和增强，真实数据仍然提供真实分布、长尾表达和实际用户需求。

误区五：只要 benchmark 提升就说明合成数据有效。

还要检查污染、泛化、风格、事实性、安全和人工样例。

误区六：复杂指令一定更好。

复杂但不合理、不可验证或答案低质的指令会伤害训练。

---

## 21. 资料与证据边界

1. [Self-Instruct](https://arxiv.org/abs/2212.10560)：展示用模型生成 instruction/input/output 并过滤后进行指令微调的路线；论文结果绑定其 seed、生成器和过滤策略。
2. [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)：支持复杂指令演化的公开案例，不证明指令越复杂就越有训练价值。
3. [Textbooks Are All You Need（phi-1）](https://arxiv.org/abs/2306.11644)：支持 textbook-quality 与合成练习对小型代码模型的研究案例；不构成跨模型的合成数据比例结论。
4. [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707)：支持解释轨迹蒸馏的公开案例；生成解释仍需要正确性、偏差和授权审计。
5. [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301)：讨论用推理过程进行知识蒸馏；过程文本不是自动可验证的因果推理证明。
6. [On the Out-of-Distribution Robustness of Large Language Models](https://arxiv.org/abs/2308.04190) 与 [The Curse of Recursion](https://arxiv.org/abs/2305.17493)：支持合成数据递归、分布收窄和模型退化风险的讨论；不同研究的设置不能直接外推成所有合成数据都会 collapse。
7. [Knowledge Distillation: A Good Teacher is All You Need?](https://arxiv.org/abs/2006.05950)：提供经典 teacher-student 蒸馏背景，说明温度、软标签和 student 目标之间的关系。

这些论文支持合成指令、复杂解释、教材数据和 teacher-student 蒸馏的特定实验事实；它们不能替代授权文件、隐私审计、领域专家验证、污染检查或生产评估。本文的质量函数、检查向量和 demo 是教学抽象，不能被误读为某个模型的公开训练配方。

## 22. 结语

Synthetic Data 与 Distillation Data 让数据工程从“收集世界”走向“主动构造训练分布”。它们可以补齐自然数据稀缺的任务，迁移 teacher 的回答和验证模式，也可以构造安全边界、工具轨迹和可验证难例。

但生成器同时也是偏置来源。只有当目标明确、授权清楚、生成参数可追踪、样本经过任务专属验证、合成比例受到自然数据锚定并且训练收益经过改写题和真实样例复核时，合成数据才是能力增强，而不是把错误和模板放大。
