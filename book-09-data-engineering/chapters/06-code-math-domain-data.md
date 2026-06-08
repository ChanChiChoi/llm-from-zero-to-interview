# 第六章：Code、Math 与 Domain Data

前一章讨论了 data mixture 与配比。本章进一步拆开三类在大模型训练中特别重要的数据：代码数据、数学数据和专业领域数据。

这三类数据有一个共同点：它们的规模通常不如通用网页文本大，但对模型能力的边际价值很高。代码数据能显著增强程序合成、结构化推理和工具使用能力；数学数据能增强多步推理、符号操作和验证意识；专业领域数据能让模型理解医学、法律、金融、科研、工程等高价值场景。

但它们也有共同风险：采集难、清洗难、授权难、污染风险高、质量判断成本高、过度上采样容易记忆和过拟合。

本章重点：代码仓库数据、单测、数学推理数据、医学法律金融等领域数据。

合规边界：本章讨论数据治理、质量控制、授权合规、安全评估和污染检测，不提供利用代码漏洞、泄露凭据、绕过系统或获取敏感领域数据的操作方法。

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 Codex / HumanEval、GSM8K / verifier、MATH、The Stack / BigCode / StarCoder、GitHub secret scanning、Med-PaLM、PubMedQA、FinGPT 和 LegalBench 等公开资料。

本讲聚焦 code / math / domain data 的防御性数据治理：专项数据池、许可证和来源审查、secret / PII 过滤、题库和 benchmark 污染隔离、功能测试和 verifier、领域权威性 / 时效性 / 引用审计，以及这些数据如何进入 mixture、继续预训练、SFT 和 RAG。

```text
专项数据池 -> 授权 / 来源 -> 结构解析 -> 质量验证 -> 风险扫描 -> 污染隔离 -> 采样配比 -> 评估闭环 -> 版本审计
```

本讲不提供漏洞利用、凭据提取、真实隐私复原、医疗法律金融决策建议或绕过数据权限的操作方法。高风险领域数据只讨论训练数据治理和模型边界，不把模型输出替代专家判断。

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

```math
D_{\mathrm{spec}}=D_{\mathrm{code}}\cup D_{\mathrm{math}}\cup D_{\mathrm{domain}}
```

每个样本可以表示为：

```math
x_i=(m_i,T_i,q_i,r_i,z_i)
```

其中 `m_i` 是数据类型，取值为 code、math 或 domain；`T_i` 是 token 数；`q_i` 是质量分；`r_i` 是风险分；`z_i` 是来源、许可证、时间戳、去重簇、污染标记等元数据。

代码数据常见的功能测试通过率为：

```math
R_{\mathrm{test},i}=\frac{n_{\mathrm{pass},i}}{\max(n_{\mathrm{test},i},1)}
```

一个可解释的代码样本质量分可以写成：

```math
q_i^{\mathrm{code}}=w_pI_{\mathrm{parse},i}+w_tR_{\mathrm{test},i}+w_d d_i-\lambda_g g_i-\lambda_u u_i
```

其中 `I_parse` 表示语法或解析检查是否通过，`d_i` 是文档 / 注释 / 测试配套分，`g_i` 是生成文件、vendor、bundle 等低价值标记，`u_i` 是重复或 fork 风险。

数学样本可以同时看答案验证和过程质量：

```math
q_i^{\mathrm{math}}=w_aI_{\mathrm{ans},i}+w_s s_i+w_h h_i-\lambda_c c_i
```

其中 `I_ans` 表示答案可验证，`s_i` 是过程完整性分，`h_i` 是难度或覆盖价值，`c_i` 是 benchmark / 题库污染风险。

领域样本更强调权威性、时效性、引用和隐私风险：

```math
q_i^{\mathrm{domain}}=w_AA_i+w_R\rho_i+w_CI_{\mathrm{cite},i}-\lambda_PP_i-\lambda_SS_i-\lambda_VV_i
```

其中 `A_i` 是来源权威性，`\rho_i` 是时效性，`I_cite` 表示是否保留出处或引用，`P_i` 是 PII 风险，`S_i` 是过期或 stale 风险，`V_i` 是高风险建议或未经验证观点风险。

专项数据的最终门禁可以写成：

```math
G_i=I(q_i^{m_i}\ge \tau_{m_i})I_{\mathrm{license},i}I_{\mathrm{secret},i}I_{\mathrm{contam},i}I_{\mathrm{privacy},i}I_{\mathrm{verify},i}
```

这里不同类型的数据使用不同阈值 `\tau_m`。代码要重视 license、secret、测试和 fork；数学要重视答案、过程和题库污染；领域数据要重视来源、时间、PII、引用和专家审计。

按类型计算 token 保留率：

```math
R_{\mathrm{keep}}(m)=\frac{\sum_{i:m_i=m}G_iT_i}{\sum_{i:m_i=m}T_i}
```

按类型计算风险命中率：

```math
R_{\mathrm{risk}}(m)=\frac{\sum_{i:m_i=m}I(r_i>0)}{\sum_i I(m_i=m)}
```

如果专项数据计划采样 token 数为 `b_m`，清洗后可用 token 数为 `N_m`，则 effective epoch 为：

```math
e_m=\frac{b_m}{N_m}
```

对于代码、数学、专业文档和合成推理数据，`e_m` 过高通常意味着更高的记忆和污染风险，需要降权、扩充数据、增强去重或重新设计采样策略。

专项数据上线前可以设置门禁：

```math
G_{\mathrm{spec}}=I(R_{\mathrm{keep}}(m)\ge r_m^{\min})I(R_{\mathrm{risk}}(m)\le r_m^{\max})I(e_m\le e_m^{\max})
```

面试里要强调：代码、数学和领域数据不是“多多益善”，而是要用类型专属质量指标、风险指标和评估矩阵一起控制。

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

## 22. 面向专家：专项数据改变模型的能力拓扑

从专家视角看，代码、数学和领域数据不是简单增加几个能力点，而是改变模型内部能力之间的连接方式。

代码数据把自然语言意图连接到可执行结构；数学数据把语言理解连接到符号约束和验证；专业数据把通用语言连接到领域本体和规范表达。

这些数据可能产生迁移效应。例如代码训练可能帮助工具调用和结构化输出，数学训练可能帮助规划和验证，领域训练可能帮助术语消歧和长文档理解。

但迁移不是免费的。专项数据也会带来风格偏移、过拟合、污染、记忆和安全边界问题。专家级数据工程的关键不是“加更多专项数据”，而是“用数据、评估和训练阶段控制迁移方向”。

---

## 23. 一个可落地的专项数据建设方案

如果面试官问：“如何为大模型建设代码、数学和专业领域数据？”可以按下面回答。

第一步，明确目标能力。代码是补全、修复、测试生成还是工程问答？数学是小学 word problem、竞赛还是证明？专业领域是知识问答、文档检索还是流程辅助？

第二步，分池建设数据。代码、数学、医学、法律、金融等各自建立独立数据池，记录来源、授权、质量、语言、时间、版本、去重和污染状态。

第三步，专项清洗。代码处理 license、secrets、fork、vendor、生成文件和语法解析；数学处理公式、题解对齐、答案校验和评测污染；领域数据处理权威性、时效性、PII、出处和专家审计。

第四步，构造结构化样本。代码构造 docstring-code、test-code、bug-fix；数学构造 problem-solution、step-by-step、verifier；领域构造 question-answer、document-grounded answer、citation-aware answer。

第五步，设计配比和训练阶段。base 阶段适量混入，继续预训练强化领域，SFT 学习任务格式，RAG/工具解决实时和高风险知识。

第六步，做污染和记忆检测。对 HumanEval、GSM8K、MATH、领域 benchmark、题库、标准答案和私有数据做 overlap 检测。

第七步，建立评估矩阵。代码看功能正确性、编译运行、测试通过率；数学看答案正确率、过程质量、鲁棒变体；领域看事实性、引用、时效、安全边界和专家评分。

第八步，版本化和审计。记录数据版本、过滤规则、采样权重、授权信息、评估结果和专家抽检报告。

### 23.1 最小可运行专项数据审计 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy code / math / domain 样本；输出包括保留样本、拒绝原因、分类型 token 保留率、最终 mixture、质量分预览和门禁检查。

它演示的是专项数据治理机制，不是生产级 license scanner、secret scanner、医学 / 法律 / 金融审核器或数学 verifier。真实系统需要接入许可证审查、secret 扫描器、测试执行器、CAS / verifier、专家审计、权限系统和数据版本管理。

```python
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
    return a / b if b else 0.0


def score(item):
    if item["kind"] == "code":
        test_rate = safe_div(item["tests_pass"], item["tests_total"])
        penalty = 0.25 * item["generated"] + 0.20 * item["duplicate"]
        return round(
            0.35 * item["syntax_ok"]
            + 0.35 * test_rate
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
gates = {
    "code_has_tests": any(item["kind"] == "code" and item["tests_total"] > 0 for item in kept),
    "math_verified": all(item.get("answer_ok", True) for item in kept if item["kind"] == "math"),
    "domain_no_pii": all(not item.get("pii", False) for item in kept if item["kind"] == "domain"),
    "contamination_blocked": reason_counts.get("eval_contamination", 0) == 2,
    "secret_blocked": reason_counts.get("license_or_secret", 0) == 2,
    "coverage": set(kind_tokens) == {"code", "domain", "math"},
}

report = {
    "kept_ids": [item["id"] for item in kept],
    "rejected": dict(sorted(rejected.items())),
    "reason_counts": reason_counts,
    "retention": round(kept_tokens / raw_tokens, 3),
    "kind_retention": {k: round(kind_tokens[k] / raw_kind_tokens[k], 3) for k in sorted(raw_kind_tokens)},
    "mixture": {k: round(kind_tokens[k] / kept_tokens, 3) for k in sorted(kind_tokens)},
    "score_preview": {row["id"]: row["score"] for row in rows},
    "gates": gates,
    "gate_pass": all(gates.values()),
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
assert report["gate_pass"] is True
```

运行后会看到类似输出：

```text
kept_ids= ['code_api_doc', 'math_word_verified', 'math_proof_note', 'domain_med_guideline', 'domain_fin_report']
rejected= {'code_eval_solution': 'eval_contamination', 'code_secret_config': 'license_or_secret', 'code_vendor_bundle': 'license_or_secret', 'domain_forum_advice': 'low_quality', 'domain_private_case': 'privacy_or_sensitive', 'math_benchmark_leak': 'eval_contamination', 'math_wrong_steps': 'unverified_answer'}
reason_counts= {'eval_contamination': 2, 'license_or_secret': 2, 'low_quality': 1, 'privacy_or_sensitive': 1, 'unverified_answer': 1}
retention= 0.459
kind_retention= {'code': 0.269, 'domain': 0.621, 'math': 0.568}
mixture= {'code': 0.251, 'domain': 0.485, 'math': 0.264}
score_preview= {'code_api_doc': 0.87, 'code_secret_config': 0.8, 'code_eval_solution': 0.84, 'code_vendor_bundle': -0.06, 'math_word_verified': 0.901, 'math_wrong_steps': 0.2, 'math_benchmark_leak': 0.892, 'math_proof_note': 0.899, 'domain_med_guideline': 0.796, 'domain_forum_advice': 0.133, 'domain_fin_report': 0.762, 'domain_private_case': 0.636}
gates= {'code_has_tests': True, 'math_verified': True, 'domain_no_pii': True, 'contamination_blocked': True, 'secret_blocked': True, 'coverage': True}
gate_pass= True
```

这个 demo 的重点是把三类专项数据的质量逻辑分开：代码看 license、secret、污染和测试；数学看答案验证、过程质量和题库污染；领域数据看权威性、时效、引用和 PII。

---

## 24. 常见面试题

### 24.1 代码数据为什么对大模型重要？

代码数据提供形式语言、可执行逻辑、API 使用、结构化输出和测试反馈。它不仅提升代码生成，也可能增强工具调用、JSON/SQL 输出和结构化推理能力。

### 24.2 代码数据清洗重点是什么？

重点是语言识别、语法解析、license、secrets 过滤、fork 和 vendor 去重、自动生成文件过滤、benchmark 污染检测、测试和文档对齐，以及质量评分。

### 24.3 为什么单测对代码模型重要？

单测提供可执行约束，可以判断功能正确性。代码生成不能只看文本相似度，应该看是否通过测试。单测还能帮助模型学习边界条件和需求到实现的映射。

### 24.4 数学数据和普通文本有什么不同？

数学数据更强调步骤、符号、答案校验和过程正确性。普通文本清洗规则可能误伤公式和短句，数学数据还要防止错误解答、跳步、题库污染和 benchmark 泄漏。

### 24.5 verifier 数据有什么价值？

verifier 数据让模型学习判断候选解是否正确。对于数学题，可以生成多个候选答案，再用 verifier 选择更可靠的解，比只训练单一答案更能利用计算和数据。

### 24.6 专业领域数据应该怎么用？

预训练或继续预训练用于学习术语和领域语言，SFT 用于学习专业问答格式和边界，RAG 用于最新、私有和可引用知识。高风险领域还需要专家审核和安全策略。

### 24.7 为什么不能把医学法律金融数据直接大量混进预训练？

因为这些数据合规要求高、时效性强、错误成本大，且可能包含隐私和专业误导。大量混入还可能带来风格偏移和过拟合。应分层使用、保留出处、做专家审计，并结合 RAG。

---

## 25. 常见误区

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

## 26. 本章小结

Code、Math 与 Domain Data 是大模型能力定向增强的核心数据。它们不是普通 web 文本的附属品，而是需要独立治理的数据资产。

本章要记住几句话：

1. 代码数据训练可执行逻辑、结构化输出和工程能力。
2. 数学数据训练多步推理、符号约束和验证意识。
3. 专业领域数据训练术语、规范表达和行业语境。
4. 三类数据都需要专项清洗、授权治理、去重和污染检测。
5. 单测、verifier、专家审计和 RAG 是这些数据走向可靠应用的重要桥梁。
6. 专项数据不是越多越好，而要按目标、阶段、配比和评估闭环使用。

如果面试中被问到专项数据建设，最好的回答是：先明确目标能力，再分池治理代码、数学和领域数据，分别处理 license/secrets、题解验证/污染、专业合规/时效，用结构化样本、配比实验、评估矩阵和版本审计形成闭环。
