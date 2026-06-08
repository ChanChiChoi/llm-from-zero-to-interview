# 第七章：Synthetic Data 与 Distillation Data

前面几章讨论的大多是自然产生的数据：网页、代码仓库、数学题、专业文档、论坛、书籍和论文。本章讨论另一类越来越重要的数据：合成数据和蒸馏数据。

合成数据是由规则、程序、模拟器或模型生成的数据。蒸馏数据通常是强模型作为 teacher，为弱模型或小模型生成回答、解释、偏好或推理轨迹，让 student 模型学习 teacher 的行为。

这类数据在今天的大模型训练中非常常见。指令微调、数学推理、代码训练、安全训练、多语言增强、领域适配、工具调用训练，都大量使用合成数据或 teacher-generated data。

但合成数据不是免费午餐。它可以快速扩充能力，也可能制造同质化、错误放大、评估污染、teacher 偏差继承和数据退化。

本章重点：合成指令数据、推理轨迹、self-instruct、teacher model distillation、数据退化风险。

合规边界：本章讨论合成数据的训练、评估、治理和风险控制，不提供绕过模型服务条款、复制专有模型能力、生成有害内容或规避安全策略的方法。

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 Self-Instruct、WizardLM / Evol-Instruct、phi-1 / Textbooks Are All You Need、Orca、经典 knowledge distillation、Distilling Step-by-Step 和 model collapse / generated data recursion 等公开论文资料。

本讲聚焦 synthetic data 与 distillation data 的防御性数据工程闭环：目标能力定义、seed 设计、teacher 授权、prompt / sampling 记录、生成后过滤、正确性验证、去重和污染隔离、合成比例控制、自然数据锚点、训练 ablation 与版本审计。

```text
目标能力 -> seed 数据 -> teacher / 规则生成 -> 验证 -> 去重 -> 安全与污染扫描 -> 配比 -> 小规模训练实验 -> 版本审计
```

本讲不讨论绕过模型服务条款、无授权复制专有模型输出、生成有害样本、规避安全策略或用合成数据替代真实合规审计。合成数据只能作为可验证、可追踪、可配比的训练信号。

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

面试中可以这样说：合成数据关注“数据来源是否生成”，蒸馏数据关注“是否把 teacher 能力迁移给 student”。

### 3.1 关键公式与审计指标

训练数据可以分成自然数据、合成数据和蒸馏数据：

```math
D=D_{\mathrm{nat}}\cup D_{\mathrm{syn}}\cup D_{\mathrm{distill}}
```

一条合成或蒸馏样本可以表示为：

```math
s_i=(x_i,y_i,a_i,t_i,p_i,q_i,r_i,z_i)
```

其中 `x_i` 是指令或上下文，`y_i` 是生成答案或轨迹，`a_i` 是目标能力标签，`t_i` 是 teacher / generator 标识，`p_i` 是 prompt 和采样参数版本，`q_i` 是质量分，`r_i` 是风险分，`z_i` 是授权、验证、去重、污染和版本元数据。

如果是普通 SFT 合成样本，目标仍是条件语言建模：

```math
L_{\mathrm{sft}}=-\frac{1}{N_{\mathrm{tok}}}\sum_i w_i\sum_j \log p_{\theta}(y_{i,j}\mid x_i,y_{i,<j})
```

其中 `w_i` 是样本权重，`N_tok` 是有效训练 token 数。`w_i` 不应该只由 teacher 分数决定，还要叠加验证、风险、重复和多样性审计。

如果 teacher 提供 soft distribution，蒸馏损失常写成 KL 形式：

```math
L_{\mathrm{kd}}=\tau^2\sum_i \mathrm{KL}(p_T^{\tau}(\cdot\mid c_i)\|p_S^{\tau}(\cdot\mid c_i))
```

其中 `p_T` 是 teacher 分布，`p_S` 是 student 分布，`\tau` 是 distillation temperature，`c_i` 是上下文。如果只有 teacher 的 hard answer，则退化为对 teacher output 做 supervised learning。

合成样本质量分可以写成可审计的加权形式：

```math
q_i=w_vV_i+w_eE_i+w_dD_i+w_sS_i-\lambda_hH_i-\lambda_uU_i-\lambda_cC_i
```

其中 `V_i` 是正确性验证，`E_i` 是证据或测试支持，`D_i` 是多样性贡献，`S_i` 是安全边界通过情况，`H_i` 是 hallucination 风险，`U_i` 是近重复或模板化风险，`C_i` 是评测污染风险。

合成数据样本门禁可以写成：

```math
G_i=I(q_i\ge \tau_q)I_{\mathrm{auth},i}I_{\mathrm{verify},i}I_{\mathrm{dedup},i}I_{\mathrm{safe},i}I_{\mathrm{contam},i}I_{\mathrm{privacy},i}
```

这里 `I_auth` 表示 teacher 输出或生成规则的使用授权成立；`I_verify` 表示答案、代码、引用、安全边界或工具参数通过验证；`I_contam` 表示没有命中评测污染。

合成和蒸馏数据在训练集合中的 token 占比为：

```math
R_{\mathrm{syn}}=\frac{\sum_{i:o_i\in\{\mathrm{syn},\mathrm{distill}\}}G_iT_i}{\sum_iG_iT_i}
```

其中 `o_i` 是样本来源类型，`T_i` 是 token 数。`R_syn` 过高时，要警惕同质化、teacher 偏差和 model collapse 风险。

能力覆盖可以按标签集合计算：

```math
C_{\mathrm{cover}}=\frac{|A_{\mathrm{target}}\cap A_{\mathrm{kept}}|}{|A_{\mathrm{target}}|}
```

同质化或近重复率可以写成：

```math
R_{\mathrm{dup}}=\frac{\sum_i I(\max_{j<i}\mathrm{sim}(s_i,s_j)>\tau_{\mathrm{sim}})}{n}
```

最终训练前门禁可以写成：

```math
G_{\mathrm{syn}}=I(R_{\mathrm{syn}}\le r_{\max})I(C_{\mathrm{cover}}\ge c_{\min})I(R_{\mathrm{dup}}\le d_{\max})I(R_{\mathrm{risk}}\le \rho_{\max})
```

面试里要强调：合成数据不是“便宜 token”，而是对训练分布的主动编辑。必须同时记录 teacher、prompt、sampling、验证、过滤、配比和训练效果。

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

面试中需要注意一个边界：蒸馏不是简单“复制某个闭源模型”。合规项目要关注授权、数据使用条款、输出归属和安全限制。

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

## 17. 面向专家：合成数据是分布编辑

从专家视角看，合成数据不是单纯增加样本，而是在编辑训练分布。

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

如果面试官问：“如何建设 synthetic data 和 distillation data？”可以按下面回答。

第一步，定义目标能力。明确要补指令跟随、数学、代码、工具调用、多语言、安全还是领域问答。

第二步，准备种子数据。来自真实任务、公开高质量数据、专家模板、领域文档或人工设计任务。

第三步，选择生成方式。规则生成适合可验证任务，LLM 生成适合开放任务，teacher-student 适合蒸馏行为和格式。

第四步，生成多样样本。控制任务类型、难度、语言、领域、输出格式和边界场景。

第五步，过滤和去重。去掉无效、重复、低质量、过短、过长、格式错、相似度过高的样本。

第六步，验证正确性。数学用答案校验，代码用测试，事实用检索和引用，领域用专家审计，安全用策略检查。

第七步，质量评分和分桶。按任务、难度、风险、语言、领域、teacher、prompt 版本打标签。

第八步，做小规模训练实验。比较不同合成比例、不同 teacher、不同过滤阈值的收益和副作用。

第九步，和自然数据混合。控制合成数据比例，避免同质化和分布退化。

第十步，版本化和合规审计。记录 teacher、prompt、采样参数、过滤规则、授权信息、质量评估和训练效果。

### 18.1 最小可运行合成与蒸馏数据审计 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy natural / synthetic / distillation 样本；输出包括保留样本、拒绝原因、来源配比、任务配比、teacher 配比、多样性覆盖和门禁结果。

它演示的是合成数据治理机制，不是生产级合规审查、LLM judge、数学 verifier、代码沙箱、安全分类器或版权系统。真实系统要接入授权审查、teacher 版本管理、prompt registry、测试执行、检索证据、人工抽样、污染检测和训练 ablation。

```python
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
    "retention": round(kept_tokens / raw_tokens, 3),
    "origin_mix": {k: round(origin_tokens[k] / kept_tokens, 3) for k in sorted(origin_tokens)},
    "task_mix": {k: round(task_tokens[k] / kept_tokens, 3) for k in sorted(task_tokens)},
    "teacher_mix": {k: round(teacher_tokens[k] / kept_tokens, 3) for k in sorted(teacher_tokens)},
    "synthetic_like_ratio": round(synthetic_like_tokens / kept_tokens, 3),
    "diversity_coverage": round(len(covered_tags & TARGET_TAGS) / len(TARGET_TAGS), 3),
}

gates = {
    "authorization": "unauthorized_teacher" in report["reason_counts"],
    "validation": "unverified_output" in report["reason_counts"],
    "contamination": "eval_contamination" in report["reason_counts"],
    "safety": "unsafe_or_policy_fail" in report["reason_counts"],
    "privacy": "privacy_or_pii" in report["reason_counts"],
    "diversity": report["diversity_coverage"] >= 0.85,
    "synthetic_ratio": report["synthetic_like_ratio"] <= MAX_SYN_RATIO,
    "natural_anchor": origin_tokens["natural"] > 0,
}
report["gates"] = gates
report["gate_pass"] = all(gates.values())

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
assert report["gate_pass"] is True
```

运行后会看到类似输出：

```text
kept_ids= ['seed_user_math', 'seed_user_domain', 'syn_math_verified', 'distill_tool_trace', 'syn_code_tests', 'syn_safety_refusal']
rejected= {'distill_unauthorized': 'unauthorized_teacher', 'real_user_private': 'privacy_or_pii', 'syn_benchmark_leak': 'eval_contamination', 'syn_code_duplicate': 'near_duplicate', 'syn_math_wrong': 'unverified_output', 'syn_unsafe_answer': 'unsafe_or_policy_fail'}
reason_counts= {'eval_contamination': 1, 'near_duplicate': 1, 'privacy_or_pii': 1, 'unauthorized_teacher': 1, 'unsafe_or_policy_fail': 1, 'unverified_output': 1}
retention= 0.546
origin_mix= {'distill': 0.171, 'natural': 0.355, 'synthetic': 0.474}
task_mix= {'code': 0.197, 'domain': 0.263, 'math': 0.228, 'safety': 0.14, 'tool': 0.171}
teacher_mix= {'human_or_seed': 0.355, 'owned_teacher': 0.368, 'policy_template': 0.14, 'rule_solver': 0.136}
synthetic_like_ratio= 0.645
diversity_coverage= 1.0
gates= {'authorization': True, 'validation': True, 'contamination': True, 'safety': True, 'privacy': True, 'diversity': True, 'synthetic_ratio': True, 'natural_anchor': True}
gate_pass= True
```

这个 demo 的重点是：合成数据进入训练前必须被当成可审计数据产品，而不是 teacher 随手生成的文本。它要能证明授权成立、错误被验证拦截、污染被隔离、PII 被过滤、近重复被降掉，并且合成 / 蒸馏 token 没有压过自然数据锚点。

---

## 19. 常见面试题

### 19.1 什么是 synthetic data？

synthetic data 是由规则、程序、模拟器或模型生成的数据，不是直接从自然环境采集的数据。在大模型中常用于指令数据、数学题、代码任务、安全样本、工具调用和领域问答。

### 19.2 什么是 distillation data？

distillation data 是 teacher model 生成或标注的数据，用于训练 student model 学习 teacher 的回答、推理、偏好、评分或格式。它关注能力迁移，而不仅是数据是否人工生成。

### 19.3 Self-Instruct 的核心思想是什么？

用模型自举生成 instruction、input 和 output 样本，过滤无效或相似样本，再用于 instruction tuning。核心不是无脑生成，而是生成后过滤和评估。

### 19.4 合成数据有什么风险？

主要风险包括错误放大、teacher 偏差继承、同质化、数据退化、评估污染、过拟合、合规问题和模型学会模板而不是真能力。

### 19.5 如何验证合成数据质量？

按任务选择验证方法：数学校验答案，代码运行测试，事实问答检索证据，领域问答专家审计，安全数据检查边界，工具调用验证参数和结果使用。还要做去重、人工抽样和训练 ablation。

### 19.6 合成数据比例应该是多少？

没有固定比例。应从目标能力、数据质量、验证可靠性、自然数据覆盖、训练阶段和副作用出发，通过小规模 ablation 逐步确定。

### 19.7 蒸馏能不能让小模型超过 teacher？

通常 student 受 teacher 数据、模型容量和训练目标限制，很难在 teacher 全能力范围内全面超过 teacher。但在特定任务、特定分布、经过高质量筛选和额外数据增强后，student 可以在局部指标上接近甚至超过 teacher。

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

## 21. 本章小结

Synthetic Data 与 Distillation Data 是大模型数据工程从“收集世界”走向“主动构造训练分布”的关键工具。

本章要记住几句话：

1. 合成数据用于补齐自然数据覆盖不到或难以标注的能力。
2. 蒸馏数据用于把 teacher 的回答、推理、偏好和格式迁移给 student。
3. Self-Instruct 和 Evol-Instruct 的核心启发是生成后必须过滤、去重和评估。
4. phi-1 说明高质量合成教材和练习可以显著提升特定能力。
5. 合成数据最大的风险是错误放大、同质化、teacher 偏差和数据退化。
6. 合成数据不是越多越好，而要有目标、验证、配比、审计和版本化。

如果面试中被问到合成数据，最好的回答不是“用强模型生成一批数据”，而是讲完整闭环：目标定义、种子设计、生成、多样性控制、过滤、正确性验证、安全审计、配比实验、自然数据混合和版本治理。
