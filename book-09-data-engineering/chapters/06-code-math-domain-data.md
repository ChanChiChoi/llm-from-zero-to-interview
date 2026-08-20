# 第六章：Code、Math 与 Domain Data

前一章讨论了 data mixture 与配比。本章进一步拆开三类在大模型训练中特别重要的数据：代码数据、数学数据和专业领域数据。

这三类数据有一个共同点：它们的规模通常不如通用网页文本大，但对模型能力的边际价值很高。代码数据能显著增强程序合成、结构化推理和工具使用能力；数学数据能增强多步推理、符号操作和验证意识；专业领域数据能让模型理解医学、法律、金融、科研、工程等高价值场景。

但它们也有共同风险：采集难、清洗难、授权难、污染风险高、质量判断成本高、过度上采样容易记忆和过拟合。

本章围绕代码、数学和专业领域数据的共同工程难题展开：它们对目标能力的边际价值很高，却往往规模有限、结构复杂、授权要求严格，也更容易因为题库、答案、私有信息和重复样本而失去训练或评估价值。讨论始终停留在数据治理、质量控制、授权合规、安全评估和污染检测层面，不把漏洞利用、凭据提取、真实隐私复原或专业决策建议当作训练方法。

## 0. 为什么专项数据需要独立治理

代码、数学和专业数据不能简单套用普通网页的过滤器。代码需要语法、依赖、测试和许可证信息；数学需要题目、过程、答案和验证关系；专业资料需要来源等级、时效、引用、权限和专家审计。三类数据都要在进入 mixture、继续预训练、SFT 或 RAG 之前保留这些结构。

可以把它们的共同链路写成：

~~~text
专项数据池 -> 授权/来源 -> 结构解析 -> 质量验证 -> 风险扫描 -> 污染隔离 -> 采样配比 -> 评估闭环 -> 版本审计
~~~

本章会分别展开代码仓库和测试、数学推理和 verifier、医学/法律/金融等专业数据，并在最后比较不同训练路径的边界。Codex/HumanEval、GSM8K、MATH、The Stack、StarCoder、Med-PaLM、PubMedQA、FinGPT 和 LegalBench 提供了公开研究入口；它们支持特定任务或数据构造的事实，不构成通用数据比例、生产安全性或专业决策可靠性的证明。

---

## 1. 先建立直觉：为什么这三类数据值得单独讲？

通用 web 数据像大海，覆盖广但质量不均。代码、数学和专业领域数据更像高浓度营养液。它们规模未必最大，但会强烈影响模型的某些核心能力。

代码数据让模型学习形式语言、抽象接口、变量绑定、长程依赖、测试驱动和机器可执行逻辑。数学数据让模型学习约束、步骤、推理链、答案校验和符号结构。专业领域数据让模型学习术语、概念体系、规范表达、领域证据和行业场景。

如果训练一个面向算法岗、工程岗或行业落地的大模型，这三类数据不能只是“顺便混一点”。它们必须被单独建池、单独清洗、单独评估、单独做污染检测。

---

## 2. 来龙去脉：从通用语言建模到能力定向数据

早期语言模型主要依赖新闻、百科、书籍和网页文本。模型重点学习自然语言流畅性和通用知识。随着 GPT-3 展示 few-shot 能力，行业意识到大规模预训练可以带来跨任务迁移。

但很快大家发现：通用语言能力不等于所有能力。模型能写文章，不代表会写正确代码；能解释概念，不代表能做多步数学；能给出医疗术语，不代表能在高风险专业场景中可靠。

Codex 的出现强化了代码数据的重要性。`Evaluating Large Language Models Trained on Code` 介绍了在公开 GitHub 代码上微调的 Codex，并用 HumanEval 评估从 docstring 合成程序的功能正确性。结果显示，代码数据和代码专项评估可以显著改变模型能力。

GSM8K 则提醒大家数学推理需要高质量问题和验证机制。`Training Verifiers to Solve Math Word Problems` 提出 GSM8K，并展示训练 verifier 对候选解排序可以改善数学 word problem 表现。这说明数学能力不只是背题，而涉及生成、验证和选择。

专业领域模型的发展也说明：通用语料覆盖不到的行业知识、术语和规范，需要专门的数据治理。医学、法律、金融等领域尤其不能只靠 web 噪声训练，因为错误回答可能带来现实风险。

---

## 3. 三类数据的共同特征

代码、数学和专业领域数据虽然内容不同，但工程问题相似。

共同特征包括：

1. 高价值：对特定能力提升明显。
2. 高稀缺：高质量数据远少于普通网页。
3. 高结构：代码有语法树，数学有步骤和公式，领域文本有术语和规范。
4. 高污染风险：很多 benchmark、题库、参考答案和标准文档公开可见。
5. 高合规要求：代码有 license，领域数据有隐私、版权和行业监管。
6. 高评估成本：需要功能测试、答案验证、专家审核或领域 benchmark。
7. 高过拟合风险：小数据池被上采样后容易记忆。

因此，这三类数据不能按普通网页处理。

### 3.1 关键公式与审计指标

设专项数据池由代码、数学和领域数据组成：

~~~math
D_spec = D_code union D_math union D_domain
~~~

每个样本可以表示为：

~~~math
x_i = (m_i, T_i, q_i, r_i, z_i)
~~~

其中 `m_i` 是数据类型，取值为 code、math 或 domain；`T_i` 是 token 数；`q_i` 是质量分；`r_i` 是风险分；`z_i` 是来源、许可证、时间戳、去重簇、污染标记等元数据。

代码数据常见的功能测试通过率为：

~~~math
R_test_i =
\begin{cases}
n_pass_i / n_test_i, & n_test_i > 0 \\
undefined, & n_test_i = 0
\end{cases}
~~~

这里的 `undefined` 不是“测试全部失败”，而是“没有可执行测试证据”。如果把没有测试的样本直接记成 `0`，数据表会无法区分未测试、测试失败和测试通过率为零的情况。质量分可以对缺少测试的样本采取保守分数，但必须同时保留 `test_evidence = false`，再由策略决定补测、隔离还是允许进入特定数据池。

一个可解释的代码样本质量分可以写成：

~~~math
q_code_i = w_p * I_parse_i + w_t * R_test_i + w_d * d_i - lambda_g * g_i - lambda_u * u_i
~~~

其中 `I_parse` 表示语法或解析检查是否通过，`d_i` 是文档 / 注释 / 测试配套分，`g_i` 是生成文件、vendor、bundle 等低价值标记，`u_i` 是重复或 fork 风险。

数学样本可以同时看答案验证和过程质量：

~~~math
q_math_i = w_a * I_ans_i + w_s * s_i + w_h * h_i - lambda_c * c_i
~~~

其中 `I_ans` 表示答案可验证，`s_i` 是过程完整性分，`h_i` 是难度或覆盖价值，`c_i` 是 benchmark / 题库污染风险。

领域样本更强调权威性、时效性、引用和隐私风险：

~~~math
q_domain_i = w_A * A_i + w_R * recency_i + w_C * I_cite_i - lambda_P * P_i - lambda_S * stale_i - lambda_V * V_i
~~~

其中 `A_i` 是来源权威性，`recency_i` 是时效性，`I_cite` 表示是否保留出处或引用，`P_i` 是 PII 风险，`stale_i` 是过期风险，`V_i` 是高风险建议或未经验证观点风险。

专项数据的状态不应被压缩成一个不可解释的总开关。可以保留一组并列的状态量：

~~~math
checks_i = (quality_ok_i, license_ok_i, secret_clear_i, contamination_clear_i, privacy_clear_i, verified_i)
~~~

这里不同类型的数据使用不同质量阈值 `tau_m` 和验证字段。代码要重视 license、secret、测试和 fork；数学要重视答案、过程和题库污染；领域数据要重视来源、时间、PII、引用和专家审计。某项检查失败时，动作可能是删除、隔离、脱敏、人工复核或改放到专门数据池，而不必把所有情况混成同一个标签。

按类型计算 token 保留率：

~~~math
R_keep(m) = sum_{i:m_i=m}(keep_i * T_i) / sum_{i:m_i=m}(T_i)
~~~

按类型计算风险命中率：

~~~math
R_risk(m) = sum_{i:m_i=m}(I(r_i > 0)) / sum_i(I(m_i = m))
~~~

`R_keep(m)` 的分母是该类型的原始 token 数，`R_risk(m)` 的分母是该类型的样本数，两者不能互换。如果某类型没有样本，或者该类型没有可计量的 token，结果应记为 `undefined`，而不是写成 `0`；`0` 表示确实存在分母但没有保留或没有命中，`undefined` 表示当前没有可解释的测量对象。

如果专项数据计划采样 token 数为 `b_m`，清洗后可用 token 数为 `N_m`，则 effective epoch 为：

~~~math
e_m = b_m / N_m
~~~

只有 `N_m > 0` 时，`e_m` 才有定义。空数据池不能因为采样器给了一个名义权重就被当成可训练数据。

对于代码、数学、专业文档和合成推理数据，`e_m` 过高通常意味着更高的记忆和污染风险，需要降权、扩充数据、增强去重或重新设计采样策略。

专项数据的整体状态也应保留为可解释的向量：

~~~math
C_spec(m) = (R_keep(m), R_risk(m), e_m)
~~~

它分别反映有效 token 保留、风险命中和重复暴露。代码、数学和领域数据不是“多多益善”，而是要用类型专属质量指标、风险指标和评估矩阵共同判断；这些数值只能说明数据处理状态，不能单独证明模型能力或专业可靠性。

---

## 4. Code data：代码数据为什么重要？

代码是一种非常特殊的语言。它既像自然语言，又不是自然语言。它有严格语法、可执行语义、依赖关系、模块结构、类型约束和测试反馈。

代码数据可以训练模型学习：

1. 语法结构。
2. 变量绑定。
3. 函数抽象。
4. API 调用。
5. 模块组织。
6. 错误处理。
7. 测试用例。
8. 注释和实现的对应关系。
9. 需求到代码的映射。
10. 代码到解释的映射。

即使目标不是代码助手，代码数据也可能帮助模型形成更强的结构化输出能力。例如 JSON、SQL、工具调用、配置文件、算法步骤和形式化推理，都和代码训练有关系。

---

## 5. 代码数据从哪里来？

常见代码数据来源包括：

1. 开源代码仓库。
2. README、文档和教程。
3. issue、pull request 和 commit message。
4. 单元测试和集成测试。
5. API reference。
6. 编程问答社区。
7. 竞赛题、题解和教学材料。
8. notebook。
9. 配置文件和脚本。

不同来源价值不同。

源代码提供实现模式，但缺少自然语言意图。README 和文档提供意图、用法和解释。测试提供可验证行为。issue 和 PR 提供 bug、需求、修复过程和真实工程语境。问答数据提供问题到解决方案的映射。

如果只训练纯代码，模型可能会写语法正确但不理解用户需求的代码。如果加入文档、测试和问答，模型更容易学习“需求 -> 实现 -> 验证”的完整链路。

---

## 6. 代码数据清洗

代码数据清洗不能套普通文本规则。代码天然符号多、缩进多、短行多、重复模式多。普通规则可能把高质量代码当成异常文本删掉。

代码清洗常见步骤：

1. 文件类型识别：识别语言、扩展名和真实内容。
2. 编码和解析检查：排除损坏文件、二进制文件、无法解码文件。
3. 自动生成文件过滤：如压缩 JS、protobuf 生成文件、锁文件、构建产物。
4. vendor 和依赖目录处理：避免重复训练第三方库镜像。
5. fork 和镜像去重：降低重复和记忆风险。
6. license 识别：判断是否允许训练使用。
7. secrets 检测：识别并移除凭据、密钥、token、证书和私有端点。
8. benchmark 污染检测：排除或隔离评测题、参考解和测试样例。
9. 质量评分：根据语法可解析性、star、维护度、测试覆盖、文档质量等信号评分。

这里的 secrets 检测是防御性数据治理，目标是避免敏感信息进入训练集，而不是发现、利用或复原敏感信息。

---

## 7. 代码数据的结构化处理

代码不只是文本。更深入的处理可以利用结构信息。

常见结构化粒度包括：

1. 仓库级：项目、依赖、license、README、测试目录。
2. 文件级：语言、路径、模块、导入依赖。
3. 函数级：签名、docstring、实现、调用关系。
4. 类级：属性、方法、继承关系。
5. AST 级：语法结构、控制流、表达式树。
6. 测试级：输入、断言、预期行为。

这些结构可以用于构造训练样本。例如：

1. docstring 到函数实现。
2. 函数实现到解释。
3. 单元测试到函数实现。
4. bug 描述到 patch。
5. API 文档到调用示例。
6. 错误日志到修复建议。

这类样本比单纯拼接代码更接近真实代码助手任务。

---

## 8. 单测为什么重要？

单元测试是代码数据里非常有价值的一类。它把“代码应该做什么”以可执行约束表达出来。

单测的价值包括：

1. 提供输入输出约束。
2. 让模型学习边界条件。
3. 支持功能正确性评估。
4. 连接需求描述和实现。
5. 可用于生成候选代码后的自动验证。

HumanEval 这类评估强调 functional correctness，即代码是否通过测试，而不是文本相似度。这对代码模型非常关键。因为两个实现可以完全不同，但功能都正确；反过来，文本相似不代表代码可运行。

训练数据中如果有高质量测试和实现对，模型更容易学习“写能跑的代码”，而不只是“写像代码的文本”。

---

## 9. 代码数据的风险

代码数据风险很集中：

1. license 风险：不同开源许可证对使用和再分发限制不同。
2. secrets 风险：仓库中可能误提交密钥、token、证书和密码。
3. 安全风险：代码可能包含漏洞、不安全 API、过时依赖。
4. 污染风险：编程评测题、参考答案、测试用例可能进入训练集。
5. 重复风险：fork、vendor、镜像和模板工程大量重复。
6. 质量风险：toy project、未维护代码、错误示例和过时写法很多。

对代码数据的正确态度不是“全收”，而是“来源治理 + license 策略 + secrets 过滤 + 去重 + benchmark 隔离 + 功能评估”。

---

## 10. Math data：数学数据为什么重要？

数学数据训练的不只是计算能力，而是约束下的推理能力。

数学任务要求模型：

1. 理解题意。
2. 抽象变量。
3. 建立关系。
4. 分解步骤。
5. 执行计算。
6. 检查答案。
7. 用清晰语言解释过程。

这些能力和很多高级任务相关：规划、科学推理、代码调试、财务分析、逻辑问答和工具调用。

但数学数据的难点是质量。一个错误解答比没有解答更糟，因为模型会学习错误推理模式。一个跳步答案会让模型学到表面格式，却没有学会中间推理。

---

## 11. 数学数据从哪里来？

数学数据常见来源包括：

1. 教材和讲义。
2. 题库和练习册。
3. 竞赛题和解答。
4. 数学论坛。
5. 课程材料。
6. 论文和证明。
7. 程序生成题。
8. 强模型生成的解题数据。
9. 人工标注的 step-by-step 解答。
10. verifier 或 judge 生成的正确性标签。

不同来源服务不同目标。

教材适合概念学习；题库适合练习模式；竞赛题适合难题推理；程序生成题适合可控覆盖；人工 step-by-step 解答适合训练过程表达；verifier 数据适合学习判断候选解正确性。

---

## 12. 数学数据清洗

数学数据清洗比普通文本难，因为公式、符号、编号和短句都很常见，不能用“符号比例高”这种普通规则粗暴过滤。

重点包括：

1. 公式解析：LaTeX、MathML、图片 OCR 公式可能损坏。
2. 题解对齐：题目、步骤、答案必须对应。
3. 答案校验：尽量用规则、计算器、CAS 或人工抽检验证。
4. 重复题检测：同一题不同版本、翻译版、改数字版。
5. 难度分级：小学、初中、高中、竞赛、本科、研究生。
6. 领域分类：代数、几何、概率、微积分、离散数学等。
7. 污染检测：与 GSM8K、MATH、竞赛评测集等进行 overlap 检测。
8. 过程质量评分：步骤是否完整、是否跳步、是否存在错误推理。

数学数据最怕“答案对但过程错”或“过程看似合理但答案错”。因此需要结果验证和过程验证共同使用。

---

## 13. 过程数据、答案数据和 verifier 数据

数学训练数据可以分三类：

1. final-answer 数据：只有题目和最终答案。
2. rationale 数据：包含解题步骤和解释。
3. verifier 数据：包含候选解及其正确性判断。

final-answer 数据便宜，但教不会模型完整过程。rationale 数据更有价值，但质量要求高。verifier 数据让模型学习判断一个推理链是否可靠，适合和多候选采样结合。

GSM8K/verifier 的思路说明：生成多个候选解，再用 verifier 选择更可信答案，是数学推理中的重要范式。对数据工程来说，这意味着不仅要收集“正确解”，还要构造“正确/错误候选 + 判断标签”的数据。

---

## 14. 数学数据的污染风险

数学 benchmark 很容易污染。原因是题目、答案、解析经常被发布到博客、课程、题解网站、GitHub 和论坛。

污染形式包括：

1. 题目原文出现。
2. 答案出现。
3. 完整解析出现。
4. 改写或翻译版本出现。
5. 同源题库中的相似题出现。
6. 合成数据生成时意外复现 benchmark 结构。

数学污染比普通问答更难，因为“改数字版”和“同题型变体”很常见。不是所有相似题都必须删除，但评测集原题、答案和解析必须严格隔离。

---

## 15. Domain data：专业领域数据为什么重要？

专业领域数据解决的是模型的“行业知识”和“专业表达”问题。

例如医学模型要理解症状、疾病、检查、药物、指南和风险提示；法律模型要理解法条、判例、合同、程序和司法解释；金融模型要理解财报、风险、市场、监管和产品结构。

通用 web 数据里也有这些内容，但通常不够系统、可信和及时。专业领域需要更可靠的数据来源、更严格的质量控制和更清晰的边界。

专业数据的目标不是让模型“取代专家”，而是让模型具备领域语言理解、资料检索辅助、初步分析和规范表达能力。

---

## 16. 专业领域数据来源

常见专业数据包括：

1. 教材和专业书。
2. 行业标准和规范。
3. 法律法规和公开判例。
4. 医学指南和药品说明。
5. 金融公告、年报和监管文件。
6. 学术论文和综述。
7. 专利和技术白皮书。
8. 企业知识库和内部文档。
9. 专家标注问答。
10. 经授权的真实业务数据。

不同数据源的合规要求差异很大。公开可访问不等于可以随意训练；企业内部数据也必须处理授权、隐私、保密和访问控制。

---

## 17. 专业领域数据清洗和治理

专业数据治理要比普通网页更严格。

关键步骤包括：

1. 来源分级：官方、权威机构、教材、论文、论坛、用户内容分开。
2. 时间戳管理：医学、法律、金融知识会过期。
3. 版本管理：法规、指南、标准经常更新。
4. PII 和敏感信息处理：尤其是医疗、金融和企业数据。
5. 专家抽检：普通标注员可能无法判断专业正确性。
6. 引用和出处保留：便于后续 RAG、审计和更新。
7. 术语标准化：同义词、缩写、别名、跨语言术语。
8. 风险标签：高风险建议、诊断、投资、法律结论等要单独标注。

专业数据不能只追求数量。低质量专业数据会让模型学到危险的自信表达。

---

## 18. 医学、法律、金融数据的差异

### 18.1 医学数据

医学数据强调安全、证据等级、时效性和患者隐私。指南、药品说明、医学教材和综述比论坛问答更可靠。真实病历必须严格脱敏、授权和审计。

医学模型不能只学“给建议”，还要学会表达不确定性、建议就医、识别高风险症状和避免越权诊断。

### 18.2 法律数据

法律数据强调地域、时效、层级和适用条件。不同法域规则不同，同一法规也会修订。判例、法条、司法解释、合同模板和法律问答要分开处理。

法律模型要避免把一般信息说成确定法律意见，尤其要保留出处、适用范围和不确定性。

### 18.3 金融数据

金融数据强调时效、市场环境、监管边界和风险披露。财报、公告、研报、宏观数据和产品材料各自有不同用途。

金融模型要避免未经依据的投资建议，训练数据中也要区分事实信息、观点、预测和营销材料。

---

## 19. 专业数据与 RAG 的关系

很多专业能力不适合完全依赖预训练记忆。

原因包括：

1. 知识更新快。
2. 需要引用出处。
3. 错误成本高。
4. 领域边界复杂。
5. 企业私有数据不能全部进入 base model。

因此，专业数据常见用法是：

1. 预训练或继续预训练学习领域语言和概念。
2. SFT 学习专业问答格式和安全边界。
3. RAG 提供最新、可引用、可控的知识。
4. 工具调用完成计算、检索、校验和流程操作。

专业数据工程要同时服务训练和检索。保留文档结构、标题、章节、出处、日期和权限信息非常重要。

---

## 20. 继续预训练 vs SFT vs RAG

面对专业领域需求，常见问题是：该继续预训练、SFT，还是做 RAG？

继续预训练适合让模型熟悉领域语言、术语、文体和基础知识。它改变模型参数，但成本高，也可能造成通用能力遗忘。

SFT 适合让模型学会任务格式、问答风格、拒答边界和流程规范。它不能凭空补足大规模知识。

RAG 适合提供动态知识、私有知识和需要引用的内容。它不一定增强模型内在推理能力，但能提高可控性和可更新性。

实际项目通常三者结合，而不是选一个。

---

## 21. 三类数据的配比策略

代码、数学和专业领域数据都不能简单越多越好。

代码数据提高太多，可能损害自然语言和多语言能力。数学数据提高太多，可能造成模板化推理或 benchmark 过拟合。专业数据提高太多，可能让模型风格过窄，甚至在非专业场景中也过度严肃。

更稳妥的策略：

1. base 预训练中保留适量专项数据。
2. 对目标能力做小规模 ablation。
3. 用继续预训练强化目标领域。
4. 用 SFT 学习任务格式。
5. 用 RAG 和工具处理高准确、强时效和私有知识。
6. 用评估矩阵监控副作用。

---

## 22. 机制与边界：专项数据改变模型的能力拓扑

从机制上看，代码、数学和领域数据不是简单增加几个能力点，而是改变模型内部能力之间的连接方式。

代码数据把自然语言意图连接到可执行结构；数学数据把语言理解连接到符号约束和验证；专业数据把通用语言连接到领域本体和规范表达。

这些数据可能产生迁移效应。例如代码训练可能帮助工具调用和结构化输出，数学训练可能帮助规划和验证，领域训练可能帮助术语消歧和长文档理解。

但迁移不是免费的。专项数据也会带来风格偏移、过拟合、污染、记忆和安全边界问题。专家级数据工程的关键不是“加更多专项数据”，而是“用数据、评估和训练阶段控制迁移方向”。

---

## 23. 一个可落地的专项数据建设方案

一个同时需要代码、数学和专业知识的模型，首先要把目标能力拆开。代码池可能服务补全、修复、测试生成或工程问答；数学池可能服务文字题、竞赛推理或证明；领域池可能服务术语理解、文档检索、引用回答或流程辅助。目标不同，样本结构、质量阈值和评估分母都会不同。

三类数据应建立独立数据池，并在每条记录上保留来源、授权、质量、语言、时间、版本、去重和污染状态。代码池记录 license、secret、fork、vendor、生成文件和解析结果；数学池记录公式、题解对齐、答案验证、难度和评测重叠；领域池记录权威性、时效、PII、出处、权限和专家审计状态。

结构化样本让数据更接近真实任务。代码可以构造 docstring-code、test-code 和 bug-fix 对；数学可以区分 problem-solution、step-by-step 和 verifier 样本；领域数据可以构造 question-answer、document-grounded answer 和 citation-aware answer。每种结构都需要保留原始证据与验证结果，避免把生成模板误当成事实。

训练路径随后按责任边界组合：base 预训练提供广泛语言和背景，继续预训练强化领域术语与文体，SFT 学习任务格式和安全边界，RAG/工具提供实时、私有、可引用或需要计算的知识。对 HumanEval、GSM8K、MATH、领域 benchmark、题库、标准答案和私有 holdout 的 overlap 检查要独立于内部去重运行。

评估矩阵也必须按类型拆开。代码看编译、功能正确性、测试通过率、依赖和安全；数学看答案正确率、过程质量、改写题和难度切片；领域数据看事实性、引用、时效、拒答边界和专家评分。最终版本应同时记录过滤规则、采样权重、授权信息、评估结果、专家抽检和数据变更血缘。

### 23.1 最小可运行专项数据审计 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy code / math / domain 样本；输出包括保留样本、拒绝原因、分类型 token 保留率、最终 mixture、质量分预览、并列检查信号、后续动作和结论。示例会把“没有测试证据”与“测试失败”分开；它的质量分和阈值仍然只是教学抽象。

它演示的是专项数据治理机制，不是生产级 license scanner、secret scanner、医学 / 法律 / 金融审核器或数学 verifier。真实系统需要接入许可证审查、secret 扫描器、测试执行器、CAS / verifier、专家审计、权限系统和数据版本管理。

~~~python
from collections import Counter, defaultdict


samples = [
    {"id": "code_api_doc", "kind": "code", "tokens": 1400, "license_ok": True, "syntax_ok": True, "tests_pass": 8, "tests_total": 8, "doc_score": 0.85, "secret": False, "contam": False, "generated": False, "duplicate": False},
    {"id": "code_secret_config", "kind": "code", "tokens": 700, "license_ok": True, "syntax_ok": True, "tests_pass": 3, "tests_total": 3, "doc_score": 0.50, "secret": True, "contam": False, "generated": False, "duplicate": False},
    {"id": "code_eval_solution", "kind": "code", "tokens": 900, "license_ok": True, "syntax_ok": True, "tests_pass": 6, "tests_total": 6, "doc_score": 0.70, "secret": False, "contam": True, "generated": False, "duplicate": False},
    {"id": "code_vendor_bundle", "kind": "code", "tokens": 2200, "license_ok": False, "syntax_ok": True, "tests_pass": 0, "tests_total": 0, "doc_score": 0.20, "secret": False, "contam": False, "generated": True, "duplicate": True},
    {"id": "math_word_verified", "kind": "math", "tokens": 650, "answer_ok": True, "process_score": 0.92, "difficulty": 0.55, "contam": False, "duplicate": False},
    {"id": "math_wrong_steps", "kind": "math", "tokens": 500, "answer_ok": False, "process_score": 0.35, "difficulty": 0.40, "contam": False, "duplicate": False},
    {"id": "math_benchmark_leak", "kind": "math", "tokens": 620, "answer_ok": True, "process_score": 0.88, "difficulty": 0.60, "contam": True, "duplicate": False},
    {"id": "math_proof_note", "kind": "math", "tokens": 820, "answer_ok": True, "process_score": 0.84, "difficulty": 0.75, "contam": False, "duplicate": False},
    {"id": "domain_med_guideline", "kind": "domain", "tokens": 1500, "authority": 0.95, "recency": 0.90, "citation": True, "pii": False, "stale": False, "risk": 0.08},
    {"id": "domain_forum_advice", "kind": "domain", "tokens": 750, "authority": 0.25, "recency": 0.60, "citation": False, "pii": False, "stale": False, "risk": 0.35},
    {"id": "domain_fin_report", "kind": "domain", "tokens": 1200, "authority": 0.90, "recency": 0.85, "citation": True, "pii": False, "stale": False, "risk": 0.08},
    {"id": "domain_private_case", "kind": "domain", "tokens": 900, "authority": 0.80, "recency": 0.75, "citation": True, "pii": True, "stale": False, "risk": 0.25},
]

THRESHOLDS = {"code": 0.72, "math": 0.70, "domain": 0.74}


def safe_div(a, b):
    return a / b if b else None


def safe_ratio(a, b):
    return round(a / b, 3) if b else None


def score(item):
    if item["kind"] == "code":
        test_rate = safe_div(item["tests_pass"], item["tests_total"])
        test_component = test_rate if test_rate is not None else 0.0
        penalty = 0.25 * item["generated"] + 0.20 * item["duplicate"]
        return round(
            0.35 * item["syntax_ok"]
            + 0.35 * test_component
            + 0.20 * item["doc_score"]
            - penalty,
            3,
        )
    if item["kind"] == "math":
        return round(
            0.45 * item["answer_ok"]
            + 0.40 * item["process_score"]
            + 0.15 * item["difficulty"],
            3,
        )
    cite = 1.0 if item["citation"] else 0.0
    return round(
        0.42 * item["authority"]
        + 0.25 * item["recency"]
        + 0.20 * cite
        - 0.35 * item["risk"]
        - 0.20 * item["stale"],
        3,
    )


def reject_reason(item, q):
    if item["kind"] == "code":
        if not item["license_ok"] or item["secret"]:
            return "license_or_secret"
        if item["contam"]:
            return "eval_contamination"
        if item["tests_total"] <= 0:
            return "missing_test_evidence"
        if item["generated"] or item["duplicate"]:
            return "generated_or_duplicate"
    elif item["kind"] == "math":
        if item["contam"]:
            return "eval_contamination"
        if item["duplicate"]:
            return "duplicate_math"
        if not item["answer_ok"]:
            return "unverified_answer"
    else:
        if item["pii"]:
            return "privacy_or_sensitive"
        if item["stale"]:
            return "stale_domain"
    if q < THRESHOLDS[item["kind"]]:
        return "low_quality"
    return None


kept, rejected, rows = [], {}, []
for item in samples:
    q = score(item)
    reason = reject_reason(item, q)
    rows.append({"id": item["id"], "kind": item["kind"], "score": q, "reason": reason or "kept"})
    if reason:
        rejected[item["id"]] = reason
    else:
        kept.append(item)

raw_tokens = sum(item["tokens"] for item in samples)
kept_tokens = sum(item["tokens"] for item in kept)
kind_tokens, raw_kind_tokens = defaultdict(int), defaultdict(int)
for item in samples:
    raw_kind_tokens[item["kind"]] += item["tokens"]
for item in kept:
    kind_tokens[item["kind"]] += item["tokens"]

reason_counts = dict(sorted(Counter(rejected.values()).items()))
checks = {
    "code_has_tests": all(
        item["tests_total"] > 0 for item in kept if item["kind"] == "code"
    ) and any(item["kind"] == "code" for item in kept),
    "math_verified": all(item.get("answer_ok", True) for item in kept if item["kind"] == "math"),
    "domain_no_pii": all(not item.get("pii", False) for item in kept if item["kind"] == "domain"),
    "contamination_blocked": reason_counts.get("eval_contamination", 0) == 2,
    "secret_blocked": reason_counts.get("license_or_secret", 0) == 2,
    "coverage": set(kind_tokens) == {"code", "domain", "math"},
}
signals = {
    "missing_code_tests": not checks["code_has_tests"],
    "unverified_math_kept": not checks["math_verified"],
    "pii_kept": not checks["domain_no_pii"],
    "contamination_not_blocked": not checks["contamination_blocked"],
    "secret_or_license_not_blocked": not checks["secret_blocked"],
    "kind_coverage_gap": not checks["coverage"],
    "no_retained_tokens": kept_tokens == 0,
}
actions = []
if signals["missing_code_tests"]:
    actions.append("add_executable_code_tests")
if signals["unverified_math_kept"]:
    actions.append("quarantine_unverified_math")
if signals["pii_kept"]:
    actions.append("remove_or_deidentify_sensitive_domain_data")
if signals["contamination_not_blocked"]:
    actions.append("quarantine_eval_overlap_and_retest")
if signals["secret_or_license_not_blocked"]:
    actions.append("repair_license_and_secret_scan")
if signals["kind_coverage_gap"]:
    actions.append("rebalance_specialized_data_pools")
if signals["no_retained_tokens"]:
    actions.append("restore_or_collect_specialized_records")
decision = "continue_to_mixture_ablation" if not actions else "hold_for_repair"

report = {
    "kept_ids": [item["id"] for item in kept],
    "rejected": dict(sorted(rejected.items())),
    "reason_counts": reason_counts,
    "retention": safe_ratio(kept_tokens, raw_tokens),
    "kind_retention": {
        k: safe_ratio(kind_tokens[k], raw_kind_tokens[k])
        for k in sorted(raw_kind_tokens)
    },
    "mixture": {
        k: safe_ratio(kind_tokens[k], kept_tokens)
        for k in sorted(kind_tokens)
    },
    "score_preview": {row["id"]: row["score"] for row in rows},
    "checks": checks,
    "signals": signals,
    "actions": actions,
    "decision": decision,
}

for key, value in report.items():
    print(f"{key}=", value)

assert report["kept_ids"] == [
    "code_api_doc",
    "math_word_verified",
    "math_proof_note",
    "domain_med_guideline",
    "domain_fin_report",
]
assert report["reason_counts"] == {
    "eval_contamination": 2,
    "license_or_secret": 2,
    "low_quality": 1,
    "privacy_or_sensitive": 1,
    "unverified_answer": 1,
}
assert report["retention"] == 0.459
assert report["kind_retention"] == {"code": 0.269, "domain": 0.621, "math": 0.568}
assert report["mixture"] == {"code": 0.251, "domain": 0.485, "math": 0.264}
assert all(report["checks"].values())
assert not any(report["signals"].values())
assert report["actions"] == []
assert report["decision"] == "continue_to_mixture_ablation"
assert safe_div(0, 0) is None
assert safe_ratio(0, 0) is None
assert safe_ratio(0, 100) == 0.0
untested_code = {
    **samples[0],
    "id": "code_missing_tests",
    "tests_pass": 0,
    "tests_total": 0,
}
assert reject_reason(untested_code, score(untested_code)) == "missing_test_evidence"
~~~

运行后会看到类似输出：

~~~text
kept_ids= ['code_api_doc', 'math_word_verified', 'math_proof_note', 'domain_med_guideline', 'domain_fin_report']
rejected= {'code_eval_solution': 'eval_contamination', 'code_secret_config': 'license_or_secret', 'code_vendor_bundle': 'license_or_secret', 'domain_forum_advice': 'low_quality', 'domain_private_case': 'privacy_or_sensitive', 'math_benchmark_leak': 'eval_contamination', 'math_wrong_steps': 'unverified_answer'}
reason_counts= {'eval_contamination': 2, 'license_or_secret': 2, 'low_quality': 1, 'privacy_or_sensitive': 1, 'unverified_answer': 1}
retention= 0.459
kind_retention= {'code': 0.269, 'domain': 0.621, 'math': 0.568}
mixture= {'code': 0.251, 'domain': 0.485, 'math': 0.264}
score_preview= {'code_api_doc': 0.87, 'code_secret_config': 0.8, 'code_eval_solution': 0.84, 'code_vendor_bundle': -0.06, 'math_word_verified': 0.901, 'math_wrong_steps': 0.2, 'math_benchmark_leak': 0.892, 'math_proof_note': 0.899, 'domain_med_guideline': 0.796, 'domain_forum_advice': 0.133, 'domain_fin_report': 0.762, 'domain_private_case': 0.636}
checks= {'code_has_tests': True, 'math_verified': True, 'domain_no_pii': True, 'contamination_blocked': True, 'secret_blocked': True, 'coverage': True}
signals= {'missing_code_tests': False, 'unverified_math_kept': False, 'pii_kept': False, 'contamination_not_blocked': False, 'secret_or_license_not_blocked': False, 'kind_coverage_gap': False}
actions= []
decision= continue_to_mixture_ablation
~~~

这个 demo 的重点是把三类专项数据的质量逻辑分开：代码看 license、secret、污染和测试；数学看答案验证、过程质量和题库污染；领域数据看权威性、时效、引用和 PII。`safe_div(0, 0)` 返回 `None`，表示没有测试证据；如果样本存在但所有 token 都被过滤，`retention` 才是 `0.0`，并且会触发恢复或补充专项数据的动作。主示例中的每类都有保留样本，所以 `mixture` 可计算；空类型或空保留集合不能被伪装成正常的零比例。

---

## 24. 决策边界：三类数据如何选择训练路径

### 24.1 代码数据的价值来自可执行反馈

代码数据提供形式语言、可执行逻辑、API 使用、结构化输出和测试反馈。它可能迁移到工具调用、JSON/SQL 输出和结构化推理，但迁移强弱取决于样本是否同时包含意图、实现、依赖和验证。单纯增加 `.py` 文件数量，不能等价于增加工程能力。

### 24.2 代码清洗必须把 license 与功能质量分开

一个代码文件可以语法正确却没有可用授权，也可以许可证清楚却包含 secret、漏洞或未维护依赖。解析、测试、文档和维护度回答“它是否有训练价值”，license、secret 和来源策略回答“它是否可以进入这个数据池”；两类判断不能合成一个质量分。

### 24.3 单测把文本相似度连接到功能正确性

两个实现的文本可以完全不同，却都满足同一组测试；一个看起来很像参考实现的片段，也可能因为边界条件而失败。因此代码评估和数据选择都应保留测试输入、断言、运行环境和依赖版本。测试通过率是重要信号，但并不证明代码没有安全问题或覆盖了所有行为。

### 24.4 数学数据的核心是结果与过程同时可信

只有最终答案的数据适合训练结果映射，却不足以说明模型掌握了推导；带过程的数据提供更多监督，但错误步骤会传播错误模式。verifier 数据把候选解和正确性判断分开，使系统可以生成多个候选再进行选择，但 verifier 自身也需要独立验证，不能把自动评分器当作绝对真值。

### 24.5 数学污染要区分原题、同源题和一般题型

原题、答案和解析进入训练会直接削弱 benchmark 的独立性；改数字、翻译或同一题库的结构相似样本提供的是不同强度的污染证据；所有涉及同一数学方法的题目则不应自动删除。去重、时间信息、字段级匹配和人工复核需要共同决定处置。

### 24.6 专业数据的训练路径取决于知识责任

继续预训练适合学习术语、文体和背景，SFT 适合学习问答格式、引用行为和安全边界，RAG 适合最新、私有、可追溯的事实，工具适合计算、检索和流程操作。医学、法律和金融场景还要加入专家审计、适用地域、版本时间和不确定性表达；单一训练阶段不能承担全部责任。

### 24.7 为什么专业数据不应无条件大量混入预训练

专业数据可能带来高价值知识，也可能带来隐私、版权、过期规范、错误自信和风格偏移。大量混入还会提高小池重复暴露，令模型把特定机构或单一法域的表达误当成普遍事实。分层使用、保留出处、独立评估和 RAG/工具协同，通常比单纯扩大预训练比例更可控。

---

## 26. 常见误区

误区一：代码数据就是 `.py`、`.java` 文件。

更完整的代码数据包括文档、测试、issue、PR、README、API reference 和问答。

误区二：能编译的代码就是高质量代码。

能编译只是基础，代码还可能过时、不安全、无 license、无测试或来自重复 fork。

误区三：数学数据只要题目和答案。

高质量数学训练更需要步骤、验证、难度、领域、答案校验和错误候选。

误区四：专业数据越多越专业。

低质量专业数据会制造危险自信，且过度配比会损害通用能力。

误区五：领域模型只靠继续预训练。

继续预训练、SFT、RAG、工具和安全策略各自解决不同问题。

误区六：公开数据一定可训练。

公开访问不等于授权训练，尤其是代码 license、领域文档版权和个人敏感信息。

---

## 27. 资料与证据边界

1. [Evaluating Large Language Models Trained on Code（Codex/HumanEval）](https://arxiv.org/abs/2107.03374)：说明代码模型可以用功能测试评估程序合成；HumanEval 分数不等于生产代码安全性，也不证明训练数据没有题目重叠。
2. [Training Verifiers to Solve Math Word Problems（GSM8K）](https://arxiv.org/abs/2110.14168)：支持数学题、候选解和 verifier 的数据与评估讨论；论文结果绑定其任务、模型和验证设置。
3. [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)：提供分层数学问题和解答评估入口；公开题目和解析需要纳入污染审计，不能直接作为独立 holdout。
4. [The Stack: 3 TB of Permissively Licensed Source Code](https://arxiv.org/abs/2211.15533) 与 [StarCoder](https://arxiv.org/abs/2305.06161)：提供开放代码数据构造、许可过滤和代码模型训练的公开案例；具体授权解释仍需回到项目版本和法律审查。
5. [GitHub secret scanning 官方文档](https://docs.github.com/en/code-security/secret-scanning/introduction/about-secret-scanning)：说明 secrets 检测的治理背景；它不是训练数据中所有凭据、隐私或高风险内容的完整检测证明。
6. [Large Language Models Encode Clinical Knowledge（Med-PaLM）](https://arxiv.org/abs/2212.13138)、[PubMedQA](https://arxiv.org/abs/1909.06146) 和 [LegalBench](https://arxiv.org/abs/2308.11462)：提供医学、医学问答和法律能力评估入口；这些 benchmark 不能替代临床、法律或金融场景的专家审核、时效检查和责任边界。

这些论文和官方文档分别支持代码功能评估、数学验证、代码数据构造、秘密治理和专业 benchmark 的特定事实。本文的质量函数、检查向量和 demo 是教学抽象，不是任何论文公布的统一数据配方；对于 FinGPT、企业知识库和经授权真实业务数据，还必须补充具体 revision、授权文件、脱敏记录、数据血缘和专家审计。

## 28. 结语

Code、Math 与 Domain Data 是大模型能力定向增强的核心数据。代码把自然语言意图连接到可执行结构，数学把语言理解连接到符号约束和验证，专业资料把通用表达连接到领域本体、时效和责任边界。

三类数据的共同原则不是追求更大的文件数，而是保留结构、证据和用途：代码要有授权和功能反馈，数学要有结果与过程验证，专业资料要有来源、版本、引用和专家边界。它们进入预训练、SFT、RAG 或工具系统的路径不同，必须由目标能力、风险和评估结果共同决定。
