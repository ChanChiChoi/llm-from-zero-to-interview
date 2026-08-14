# 第十一章：Error Analysis 与回归测试

一个模型的总体分数上升，并不意味着系统在所有重要场景中都变好了。新版本
可能在大量简单题上提升，却在少数高价值任务上退化；也可能修复了答案错误，
却引入了 JSON 格式错误、工具越权或引用不可信。平均分告诉我们变化发生了多少，
Error Analysis 要继续追问变化发生在哪里、为什么发生，以及应该怎样修复。

回归测试解决的是另一个时间维度的问题：今天修好的错误，明天还会不会回来。把
关键失败样例、历史事故和容易波动的边界情况保存成可重复的测试，才能在模型、
prompt、检索、工具、协议和安全策略变化时及时发现退化。

这两个方法共同组成评估闭环。错误分析把分数拆成问题，根因分析把问题变成可
验证的假设，回归测试把已修复的问题变成长期证据；但它们都不能替代新样本、
线上监控和人工抽检。只测试已经知道的失败，会让系统越来越擅长回答自己的
题库，却看不见未知错误。

## 1. 从一个分数回到具体失败

### 1.1 小白视角：总体分数为什么会骗人

假设旧版本在 1,000 个样本上正确 820 个，新版本正确 840 个。表面上看，新
版本提升了 2 个百分点。现在把样本按场景拆开：

| 场景 | 旧版本 | 新版本 | 变化 |
| --- | ---: | ---: | ---: |
| 普通问答 | 0.84 | 0.90 | +0.06 |
| 中文法律问答 | 0.78 | 0.70 | -0.08 |
| 代码补全 | 0.62 | 0.68 | +0.06 |
| 数学证明 | 0.75 | 0.70 | -0.05 |

如果普通问答占了绝大多数样本，总分仍然会升高；如果产品的核心用户主要使用
法律问答和数学证明，总体数字就没有代表真实价值。它不是统计计算错了，而是
总体分数没有表达决策真正关心的场景。

### 1.2 不同错误的代价不同

把每个失败都计作 1 个错误，适合快速计算 accuracy，却不适合所有产品。一个
格式多了空格，和一次未经授权的数据库写入，不应该拥有相同的优先级。大模型
系统常见的失败严重度可以包括：

1. 轻微格式或风格问题；
2. 答案不完整但容易恢复；
3. 推理、计算或代码错误；
4. 事实幻觉或无证据断言；
5. 检索、引用或工具参数错误；
6. 正常问题被错误拒答；
7. 安全违规、隐私泄露或越权副作用。

严重度不是错误类型的同义词。同一个 `retrieval_miss` 在普通百科问答里可能是
低影响，在医疗处方或权限判断中可能是高影响；标注时需要同时保留错误类型、
业务场景和严重度。

### 1.3 专家视角：误差向量

对一次评估记录 i，可以把需要审计的信息抽象为：

~~~math
e_i=(x_i,y_i,a_i,v_i,h_i,t_i,r_i,c_i,w_i)
~~~

其中：

- `x_i` 是输入、上下文和用户任务；
- `y_i` 是模型输出以及工具轨迹；
- `a_i` 是参考答案、程序结果或期望行为；
- `v_i` 是模型、prompt、检索、工具和评估器版本；
- `h_i` 是 harness、路由、权限、缓存和运行环境；
- `t_i` 是时间和结果观察窗口；
- `r_i` 是一个或多个错误标签；
- `c_i` 是能力、领域、语言或风险切片；
- `w_i` 是用于业务加权的严重度或价值权重。

只保存 `score_i` 会丢掉大部分诊断信息。保存误差向量，才能回答“错误是模型
输出造成的，还是证据没有送到模型，还是协议转换把正确结果弄坏了”。

### 1.4 从总体指标到问题清单

失败集合可以写成：

~~~math
\mathcal{F}=\{i:s_i=0\}
~~~

错误类型 k 的数量和失败占比为：

~~~math
N_k=\sum_{i\in\mathcal{F}}\mathbb{I}[r_i=k],
\qquad
P_k=\frac{N_k}{|\mathcal{F}|}
~~~

如果一个样本可以有多个标签，`P_k` 的总和可能超过 1；这不是 bug，而是多标签
统计的自然结果。若要得到互斥分布，应另外定义主标签规则，不能在报告中把两
种分母混用。

严重度加权损失可以写成：

~~~math
L_k=\sum_{i\in\mathcal{F}}w_i\,\mathbb{I}[r_i=k]
~~~

它让“低频高危”问题不会被高频轻微格式错误完全淹没。频次、严重度、暴露量和
可修复性可以用于排序，但这个排序只是资源分配工具，不是一个客观的真理分数。

### 1.5 错误分析的五个问题

每次复盘一个失败样例或一个错误簇，都应尽量回答：

1. 发生了什么，用户实际看到了什么；
2. 哪一个结果契约被违反了；
3. 失败属于模型、证据、协议、harness 还是评估器；
4. 哪个干预可以区分不同的根因假设；
5. 修复后怎样在未见过的样本上确认没有复发。

如果只写“模型幻觉”“推理能力不足”，这些标签不能直接导出修复路径，也不能
帮助下一个版本复现和验证。

## 2. 失败证据的记录方式

### 2.1 一条记录至少保留什么

错误分析必须从可复核的原始证据开始。对离线任务，通常需要：

1. 稳定的样本 id；
2. 脱敏后的输入和上下文摘要；
3. 参考答案、程序判定器或期望行为；
4. 完整模型输出和解析状态；
5. judge 分数、解释、置信度和人工标签；
6. 模型、tokenizer、prompt、解码和评估器版本；
7. 检索文档、引用、工具调用和权限结果；
8. 运行时间、随机种子和 trace id；
9. 错误类型、严重度、场景切片和标注来源。

线上数据还要记录资格、分桶、真实曝光、生成、展示、重试、降级和结果成熟
状态。一次请求返回 HTTP 200，不代表用户看到了完整回答，也不代表任务完成。

### 2.2 失败、无效和评估器分歧

下面几种状态必须区分：

~~~text
model_failure       模型或系统确实没有完成任务
invalid_output      输出协议无法解析
evaluator_dispute   judge 与人工或程序判定冲突
missing_evidence    关键检索、工具或上下文证据缺失
pending              结果观察窗口尚未成熟
data_error           样本、标签或日志本身有问题
~~~

把所有状态都放进“模型失败”会导致错误归因；把所有分歧都删除又会让评估器的
偏差消失在报表里。评估器分歧本身是一个需要校准的信号，尤其在医疗、法律、
数学、代码和安全任务中。

### 2.3 版本和 harness 信息

模型失败的边界正在从“权重输出”扩展到完整运行链路。一个长上下文 Agent 的
失败可能来自：

1. 模型没有从上下文找到证据；
2. context folding 或 memory 丢失了证据；
3. 工具协议把参数名转换错；
4. reasoning channel 被错误地当作最终答案解析；
5. 权限或租户过滤把可用文档删掉；
6. runtime 超时或截断了工具结果；
7. 评估器误读了多模态输入或结构化输出。

因此记录中要把 model、harness、protocol 和 evidence 作为可独立标注的维度：

~~~math
e_i=(e_{\mathrm{model}},e_{\mathrm{harness}},
e_{\mathrm{protocol}},e_{\mathrm{evidence}})
~~~

同一条失败可以同时拥有多个非空维度。只允许单选“模型/系统”会丢掉组合根因。

### 2.4 隐私和安全边界

失败样例常含用户问题、企业文档、代码、个人信息或攻击内容。收集样例时要：

1. 删除不必要的直接身份信息；
2. 依据权限和用途控制访问；
3. 记录谁查看过高风险原文；
4. 对工具参数和外部状态做脱敏或隔离；
5. 不把真实秘密写进公开 regression case；
6. 在报告里使用可复现的摘要、占位符或安全合成样例。

脱敏不能改变错误的关键结构。例如把所有年份、字段名和权限关系都删掉，可能
会让“时间约束检索失败”无法复现。应保存最小但足够的结构化证据，并让原文
留在受限存储中。

## 3. 设计能指导修复的错误 taxonomy

### 3.1 分类体系的三个要求

错误 taxonomy 不是越多越专业。一个可用的分类体系应满足：

1. **覆盖性**：能覆盖主要失败模式；
2. **区分性**：不同标签对应不同的验证或修复路径；
3. **一致性**：两个标注员面对同一证据时有较高概率做出相同选择。

如果标签无法指导下一步动作，就只是报表装饰；如果标签含义重叠、边界模糊，
聚合出来的比例也没有稳定解释。

### 3.2 把四类标签分开

实践中至少应区分四类字段：

| 字段 | 例子 | 回答的问题 |
| --- | --- | --- |
| symptom | 缺字段、事实错、工具循环 | 用户看到了什么 |
| capability | 指令遵循、检索、推理、引用 | 哪种能力表现异常 |
| root cause hypothesis | 数据覆盖、prompt 冲突、协议 bug | 可能为什么发生 |
| action | 补数据、改 schema、换检索器 | 下一步怎样验证或修复 |

症状不是根因，能力也不是修复动作。例如“引用错误”是症状，“grounding”是
能力维度，“检索召回不足”是根因假设，“补时间约束 hard negative”是动作。
把它们写在一列里，会让团队误以为标签本身已经证明了原因。

### 3.3 通用任务的一级分类

通用问答、摘要和结构化输出可以从以下一级标签开始：

~~~text
instruction_following
factuality
reasoning
completeness
formatting
uncertainty_calibration
refusal_boundary
safety
privacy
tool_or_protocol
~~~

同一条输出可能同时有 `factuality` 和 `uncertainty_calibration`：它既说错了
事实，又用过度肯定的语气掩盖了证据不足。标注指南应说明主标签、次标签和
“无法从当前证据判断”的选项，避免强迫标注员编造根因。

### 3.4 RAG 的分层标签

RAG 错误应沿着信息流拆分：

1. query understanding：没有识别问题中的实体、时间或权限约束；
2. retrieval recall：答案文档没有被召回；
3. retrieval noise：干扰文档压过了相关文档；
4. reranking：相关文档顺序错误；
5. context packing：证据被截断、重复或放入错误区域；
6. grounded generation：上下文有答案但生成不忠实；
7. citation：引用与 claim 不匹配；
8. abstention：无证据时没有说明不确定性；
9. freshness：知识库版本或时间窗口过期。

判断生成错误前，先确认正确文档是否真的进入模型上下文。若答案文档没有召回，
继续调 prompt 往往不能修复问题；若上下文已经明确而输出仍然错误，才有理由把
注意力转向生成、指令遵循或上下文理解。

### 3.5 Agent 和工具任务的分层标签

Agent 失败可以沿着状态机标注：

~~~text
goal_interpretation
planning
tool_selection
tool_arguments
permission
execution
observation_reading
state_update
recovery_or_loop
final_answer
~~~

例如工具返回正确数据、Agent 也读到了数据，却在最终答案里写错数字，这是
`final_answer` 或 `synthesis` 问题；如果工具参数中的租户 id 错了，则优先检查
`tool_arguments` 和权限边界，而不是给模型一个笼统的“知识不足”标签。

### 3.6 Code、Math 和 Reasoning

代码任务可拆为需求理解、API 使用、算法、边界条件、复杂度、测试、依赖和多
文件一致性；数学任务可拆为形式化、变量定义、约束提取、计算、证明和最终
验证；多步推理任务可拆为计划、单步推理、状态维护、反例检查和结论验证。

“最终答案错误”只是最外层症状。若中间步骤正确、解析器错误，修复方向与真正
的数学推理失败完全不同。过程标签需要和答案标签分开记录。

### 3.7 多标签和层级统计

若样本 i 可以属于多个错误类型，标签矩阵 `A` 可写成：

~~~math
A_{ik}=\mathbb{I}[\text{sample }i\text{ has label }k]
~~~

标签 k 的支持率为：

~~~math
\mathrm{Support}_k
=\frac{1}{n}\sum_{i=1}^{n}A_{ik}
~~~

支持率之间不能简单相加。若需要互斥汇总，应设置 `primary_label`；若需要保留
因果链，则保留多标签，并在报告中说明“每个样本可有多个标签”。层级 taxonomy
还应允许从二级标签回溯一级类别，避免每次修改标签名称都破坏历史趋势。

## 4. 标注、校准与评估器分歧

### 4.1 标注单位要先固定

一个“错误”可以按 token、字段、claim、回答、任务、会话或工具轨迹标注。单位
不同，数量和严重度也不同。例如一个回答含三个错误 claim：

1. 以回答为单位，它可能是 1 个失败；
2. 以 claim 为单位，它可能是 3 个 unsupported claims；
3. 以任务为单位，还要判断这三个错误是否让用户任务整体失败。

标注指南必须明确单位、分母和“一个样本多个错误”如何处理，否则不同项目的
错误率不能比较。

### 4.2 绝对标准和相对比较

错误分析既可以检查候选答案是否满足 rubric，也可以比较旧版和新版哪个更好。
绝对标准适合回归 case 和安全规则；相对比较适合发现版本差异。两者不要混成
一个模糊的“judge 分数”：

| 形式 | 适合的问题 |
| --- | --- |
| 绝对标签 | 是否有事实错误、是否包含引用、是否越权 |
| 成对标签 | 新旧版本哪一个更好、是否修复 |
| 程序结果 | 测试是否通过、JSON 是否可解析、工具参数是否合法 |
| 分级 rubric | 严重度、完整性、证据质量、可恢复性 |

### 4.3 人工指南应包含反例

只给正面定义通常不够。一个好的标注指南应给出：

1. 标签定义；
2. 最小正例；
3. 容易混淆的反例；
4. 不确定时如何标记；
5. 多标签优先级；
6. 高严重度升级路径；
7. 版本变化后的重新校准规则。

比如“拒答错误”至少要区分：危险任务的正确拒答、正常任务的过度拒答、危险
任务的错误帮助和证据不足时的谨慎回答。把所有拒答都标成 `refusal`，不能指导
安全策略调整。

### 4.4 人工与 LLM judge 的协作

可以先用少量高质量人工样本建立标签和指南，再用 LLM judge 做批量预标注，最后
对高风险、不确定和分歧样本做人审。judge 的标签要记录模型、prompt、参考资料
和解析失败，不应覆盖原始人工标签。

对于二分类标签，观察一致率 `p_o` 和按边际概率得到的偶然一致率 `p_e` 后，
Cohen's kappa 可写为：

~~~math
\kappa=\frac{p_o-p_e}{1-p_e}
~~~

Kappa 受类别不平衡影响很大。高一致率不一定代表标注质量好，尤其当 99% 样本
都是“无错误”；应同时报告混淆矩阵、每类召回、严重度分层和人工抽样结果。

### 4.5 分歧不是噪声桶

judge 与人工、两个 judge 或程序判定器之间的分歧，可能揭示：

1. rubric 定义含糊；
2. 参考答案不完整；
3. judge 看不到必要上下文；
4. 模型输出包含多个等价表达；
5. 标注员对风险等级理解不同；
6. 程序判定器过于严格或存在 bug。

分歧样本应优先进入校准集，必要时修订指南和评估器，而不是简单多数投票后丢掉
解释。对高风险标签，人工复核通常比追求自动一致率更重要。

## 5. 失败样例聚类与主动挖掘

### 5.1 手工聚类适合小样本

当失败样例只有几十到几百条时，人工阅读往往比直接上 embedding 更可靠。一个
实用过程是：

1. 抽取覆盖不同切片的失败样本；
2. 每条样本用一句话描述用户可见问题；
3. 合并语义相近且修复路径相近的描述；
4. 为簇命名并记录反例；
5. 统计频次、严重度、版本和场景分布；
6. 把簇转成根因假设，而不是直接宣布根因。

手工聚类的价值在于它迫使分析者阅读完整 trace，避免只按主题相似把“财报问题”
聚在一起，却看不出其中有时间约束、权限和引用三个不同失败。

### 5.2 Embedding 聚类

大规模样本可以使用 embedding 辅助发现相似模式。对分析文本 `u_i` 的向量
`v_i`，余弦相似度为：

~~~math
\operatorname{cos}(v_i,v_j)
=\frac{v_i^{\top}v_j}{\|v_i\|_2\|v_j\|_2}
~~~

分析文本不应只包含用户问题，因为主题相似不等于错误原因相似。更完整的文本
可以拼接：

~~~text
input + model_output + reference_or_expected_behavior
      + retrieved_evidence + judge_reason + preliminary_tags
~~~

聚类后应抽样阅读每个簇，检查簇内一致性、簇间差异和高严重度样本是否被小簇
吞掉。聚类算法的轮廓系数或主题标签不能替代人工确认；一个“格式错误”簇里
可能同时含 JSON 解析失败、Markdown 渲染失败和字段语义错误。

### 5.3 LLM 辅助归因

可以让 LLM 生成候选标签和根因假设，但输入边界要明确：只允许使用给定输入、
输出、参考答案、检索上下文和工具轨迹，不让它凭空补充不可见事实。输出至少
应包含：

~~~text
symptom
capability_label
root_cause_hypothesis
evidence_used
uncertainty
suggested_intervention
~~~

其中 `root_cause_hypothesis` 仍是待验证假设。批量 judge 可以降低初标成本，却
可能继承 prompt、模型家族、长度和参考答案偏差；高风险样本、低置信度样本和
簇代表样本要进入人工复核。

### 5.4 主动挖掘最有信息量的样本

随机抽样能估计总体错误率，主动挖掘则用于发现和解释错误。可以优先抽取：

1. judge 与人工分歧的样本；
2. 版本输出差异最大的样本；
3. 低置信度或高熵样本；
4. 高严重度安全分类边界样本；
5. 新用户、新语言和新工具路径样本；
6. 同一输入在多次运行中结果不稳定的样本。

主动样本不能直接替代总体评估，因为它改变了抽样分布。报告中应区分“总体错误
率估计”和“主动挖掘发现的候选问题”。

### 5.5 频次、严重度和可修复性

一个可解释的修复优先级可以写成：

~~~math
\mathrm{Priority}_k
=F_k\,S_k\,Q_k
~~~

其中 `F_k` 是错误簇在目标总体中的频次或暴露量，`S_k` 是平均严重度，`Q_k`
是可修复性或预期收益。这个式子不能代替产品决策：低频隐私泄露可能有很高的
风险权重，不能因为 `F_k` 小就排在所有问题之后。

## 6. Root Cause：从现象到可证伪假设

### 6.1 症状、假设和根因不是一回事

“模型答错了”是现象；“retrieval miss”是错误类型；“时间约束 hard negative
不足”是根因假设；“补充数据并在未见年份上复验”是验证和修复动作。四者应该
分开写，才能避免一个标签被误当成已经证明的解释。

### 6.2 根因类别

常见根因可以按系统层级整理：

| 层级 | 可能根因 | 典型验证 |
| --- | --- | --- |
| 数据 | 覆盖不足、冲突标签、污染、hard negative 缺失 | 数据切片、血缘和独立样本 |
| 训练 | 目标错配、偏好偏差、灾难性遗忘 | 训练 ablation、历史 checkpoint |
| 模型 | 表示、推理、校准或长上下文能力不足 | 正确证据输入、过程评估、任务切片 |
| Prompt/协议 | 约束冲突、schema、chat template、编码转换 | 固定模型后替换协议或 prompt |
| Harness | memory、预算、权限、截断、重试、状态机 | trace 回放和组件旁路 |
| 工具/RAG | 召回、重排、参数、外部状态、版本过期 | 正确证据/人工工具结果对照 |
| 评估器 | rubric、解析器、judge 或标签错误 | 人工 gold set 和程序单测 |

### 6.3 用干预区分假设

设基线失败率为 `p_0`，某个干预后的失败率为 `p_1`，可以把干预带来的变化写成：

~~~math
\Delta_{\mathrm{fail}}=p_1-p_0
~~~

例如 RAG 答错时，可以设计：

1. 原始检索、原始模型；
2. 人工提供正确文档、原始模型；
3. 原始检索、替换生成模型；
4. 替换检索器、固定生成模型；
5. 固定证据和模型、替换 prompt 或上下文拼接器。

若人工提供正确文档后错误大幅减少，检索或证据链假设得到支持；若仍然错误，
应继续检查上下文理解、生成和评估器。一次干预只能支持或削弱假设，不能自动
证明唯一根因，尤其是多个组件同时改变时。

### 6.4 五个为什么

Five Whys 适合把直接现象继续追问到流程或数据原因。比如财报问答输出错误数字：

~~~text
现象：模型回答了错误年份的营收。
为什么 1：上下文中 2022 年财报排在 2023 年财报之前。
为什么 2：重排器没有识别问题中的年份约束。
为什么 3：重排训练样本按主题相关性构造，没有时间 hard negative。
为什么 4：数据标注指南没有把时间约束作为独立字段。
为什么 5：评估集也没有覆盖同一公司跨年份的对比问题。
~~~

候选根因可能是“时间约束数据和评估覆盖不足”。但这仍要通过补充 hard negative、
固定证据的生成测试和未见年份 holdout 验证；Five Whys 是组织调查的工具，不是
因果证明。

### 6.5 近因和根因

近因是让一次请求失败的直接触发点，根因是让同类问题反复出现的系统条件：

~~~text
近因：输出 JSON 少了一个字段。
根因候选：prompt 没有明确 schema，服务端没有校验、修复或重试机制。
~~~

如果只把这一条输出重新生成并标记为修复，下一次 prompt 或模型变化仍可能重现
问题。根因修复通常需要同时改变协议、校验、样本和回归测试。

### 6.6 Model、Harness、Protocol 和 Evidence

对于 reasoning effort、工具调用、原生多模态和长上下文模型，错误归因至少要
区分四个维度：

1. **Model**：感知、推理、生成、工具选择或安全边界失败；
2. **Harness**：上下文折叠、memory、workspace、预算、权限或状态管理失败；
3. **Protocol**：chat template、custom encoding、tool schema、reasoning channel、
   MCP/A2A item 转换失败；
4. **Evidence**：没有证据、证据过期、引用不支持，或把厂商披露误当作独立事实。

一个 trace 可以在多个维度同时失败。回归集应固定模型 revision、reasoning effort、
工具、harness、上下文策略和运行环境，否则版本变化无法归因。

## 7. Regression Suite：让已知问题不再复发

### 7.1 它和 benchmark 有什么不同

普通 benchmark 主要回答“模型在一组广泛任务上表现如何”；golden set 主要保存
高质量、稳定、代表核心能力的样本；regression suite 则专门保护已经发生过的
问题、关键业务样本和高风险边界。

| 集合 | 主要目的 | 样本来源 |
| --- | --- | --- |
| benchmark | 衡量广泛能力和横向比较 | 公开或设计的任务集 |
| golden set | 稳定测量核心任务 | 高质量标注和程序判定 |
| regression suite | 防止已知问题复发 | 线上事故、历史失败、关键边界 |
| exploration set | 发现未知问题 | 新用户、新分布、主动挖掘 |

它们可以有重叠，但不能把 regression suite 当作全部评估。一个模型只在历史
失败集上变好，可能只是记住了题目；只有同时在独立或新构造的样本上改善，才有
理由谈泛化。

### 7.2 样例进入回归集的条件

不是每个失败都值得永久保留。适合加入的 case 通常满足：

1. 对真实用户、业务或安全有重要性；
2. 失败原因或期望行为足够清楚；
3. 能在固定环境中稳定复现；
4. 有可执行的程序、人工或结构化判定；
5. 能代表一类问题，而不是一次偶然网络故障；
6. 维护者、来源和版本血缘明确。

如果一个 case 的期望答案已经过时，应更新版本或退休，而不是让它永久影响
趋势。若一个问题无法在真实信息中公开，可以使用保留结构的安全合成样例，并
把原始事故放在受限证据库中。

### 7.3 结构化 case

一个 RAG 回归样例可以写成：

~~~json
{
  "case_id": "rag_finance_2023_revenue_001",
  "task_type": "rag_qa",
  "capability": ["temporal_reasoning", "retrieval", "faithfulness"],
  "severity": "high",
  "input": "请根据知识库回答：A 公司 2023 年营收是多少？",
  "expected_behavior": "使用 2023 年财报中的营收数字，并给出支持该数字的引用。",
  "failure_history": "v1.8 检索到 2022 年财报并回答了错误数字。",
  "oracle": {
    "required_claims": ["year=2023"],
    "required_evidence": ["annual_report_2023"],
    "forbidden_final_claims": ["2022 年营收作为最终答案"]
  },
  "owner": "eval_team",
  "source": "online_incident_2024_05"
}
~~~

`expected_behavior` 比一段固定答案更适合开放式生成；`oracle` 可以组合字段、
证据、程序检查和人工 rubric。对于事实会变化的知识库，还要记录 snapshot 和
生效时间，不能让期望答案脱离数据版本。

### 7.4 稳定 oracle 与等价答案

回归测试最容易出现的错误，是把某次输出的字面文本当作唯一正确答案。更稳健
的判定可以按任务使用：

1. JSON schema 和字段类型检查；
2. 数学答案规范化后的等价比较；
3. 代码隐藏测试和安全沙箱；
4. claim 与证据的支持关系；
5. 工具参数、权限和外部状态的契约；
6. 人工 rubric 或校准过的 judge。

如果只能用 judge，应保存 judge 版本、prompt、参考资料、解析失败和人工抽检；
不要把 judge 自己的文本解释当作 oracle。回归集的目标是稳定检测行为，不是强迫
模型逐字复制一个答案。

### 7.5 严重度、来源和维护者

每个 case 要有严重度、来源、责任人和最后复核时间。高严重度样例通常要：

1. 有独立人工复核；
2. 有更严格的程序或权限检查；
3. 记录修复版本和回归历史；
4. 对真实敏感内容进行访问控制；
5. 不能因为总体分数提升而被自动覆盖。

### 7.6 防止训练污染

历史失败样例可以用于修复训练，但同一条样例不能同时作为“已泛化”的独立证据。
更可靠的安排是：

~~~text
train-fix set       用于修复或调参
held-out variants   同一能力的未见变体
regression set      固定检查已知问题
fresh exploration   检查新的未知失败
~~~

报告时要明确模型是否见过 case、答案或其模板。对年份、实体、代码变量和语言
做变换，可以构造结构相同但表面不同的 holdout；这测试的是能力是否迁移，而不
只是文本记忆。

### 7.7 生命周期和清理

回归集会随产品变化而积累。维护时应定期检查：

1. 期望行为是否仍然符合当前产品政策；
2. 依赖的模型、工具和知识库 snapshot 是否可取得；
3. case 是否重复或被更一般的 case 覆盖；
4. oracle 是否仍然有效；
5. 历史修复是否已经稳定；
6. 是否需要把 case 迁移到探索或长期 golden 集。

退休 case 要保留历史记录和退休原因，而不是物理删除后让趋势失去解释。

### 7.8 覆盖率的边界

设近期关键能力集合为 `C_recent`，回归集覆盖的能力集合为 `C_suite`，可以定义
一个粗略覆盖率：

~~~math
C_{\mathrm{suite}}
=\frac{|C_{\mathrm{suite}}\cap C_{\mathrm{recent}}|}
{|C_{\mathrm{recent}}|}
~~~

这个数字只能描述标签覆盖，不代表每个能力有足够样本，也不代表未知问题被覆盖。
应同时记录每类 case 数量、严重度、最近运行时间、失败率和未覆盖项。

## 8. 版本对比：看样本级变化

### 8.1 版本账本

比较两个版本前，至少固定或记录：

1. 模型权重、tokenizer 和量化版本；
2. system prompt、模板和 reasoning effort；
3. 数据集、知识库 snapshot 和权限规则；
4. retriever、reranker、tool schema 和工具实现；
5. judge、解析器、rubric 和评估代码；
6. decoding、并发、超时、缓存和降级参数；
7. 硬件、runtime、依赖和运行时间。

如果其中一项必须变化，应把它视为 treatment 的一部分，并在结论中说明是模型
变更还是整套工作流变更。不存在“只换模型”的假设，就不要把结果写成模型
本体能力提升。

### 8.2 四个样本级象限

对齐同一个 case 后，每条记录属于四类之一：

~~~text
old correct, new correct   稳定正确
old wrong,   new correct   修复
old correct, new wrong     回归
old wrong,   new wrong     仍未解决
~~~

设 `m` 为成功对齐的 case 数，回归率和修复率分别是：

~~~math
R_{\mathrm{reg}}
=\frac{\sum_i\mathbb{I}[o_i=1,n_i=0]}{m},
\qquad
R_{\mathrm{fix}}
=\frac{\sum_i\mathbb{I}[o_i=0,n_i=1]}{m}
~~~

净变化 `R_fix-R_reg` 可以描述方向，但不能掩盖一个严重回归。例如修复十个
低风险格式 case，同时引入一个隐私泄露，净数量可能看似不错，风险却不能这样
抵消。回归与修复还应按严重度、能力和业务场景分别报告。

### 8.3 分层 diff

总体 diff 之后应按语言、任务、领域、难度、输入长度、输出协议、用户类型和
风险等级切片。每个切片至少报告：

1. case 数量和对齐率；
2. old/new 通过率；
3. 修复数和回归数；
4. 严重度加权损失；
5. 新增错误簇；
6. 评估器分歧和缺失证据。

切片数量很多时，按照第十章的多重比较原则区分确认性和探索性发现。错误分析
的目的不是为每个切片都制造一条结论，而是找出值得复现和修复的模式。

### 8.4 配对差异的不确定性

四象限计数也有抽样不确定性。若 case 来自同一批配对样本，可以对 case 或用户
做 paired/cluster bootstrap，估计 `R_reg`、`R_fix` 和严重度加权差异的区间。
低数量的高严重度回归应同时报告原始 case id 和人工复核，不要因为区间很宽就
隐藏它。

### 8.5 一个最小 diff 函数

下面的函数只负责样本级对齐和分类，不负责决定版本是否可以发布。把统计、风险
和后续动作留在调用方，能避免一个函数把复杂决策压成一个布尔值。

~~~python
from collections import Counter, defaultdict


def compare_versions(old_results, new_results):
    """Compare aligned case results and retain tag-level diagnostics."""
    buckets = Counter()
    regressions = []
    fixes = []
    tag_diff = defaultdict(Counter)

    for case_id, old in old_results.items():
        if case_id not in new_results:
            continue

        new = new_results[case_id]
        old_ok = bool(old["correct"])
        new_ok = bool(new["correct"])

        if old_ok and new_ok:
            bucket = "stable_correct"
        elif not old_ok and new_ok:
            bucket = "fixed"
            fixes.append(case_id)
        elif old_ok and not new_ok:
            bucket = "regressed"
            regressions.append(case_id)
        else:
            bucket = "stable_wrong"

        buckets[bucket] += 1
        tags = set(old.get("tags", [])) | set(new.get("tags", []))
        for tag in tags:
            tag_diff[tag][bucket] += 1

    return {
        "summary": dict(buckets),
        "regressions": regressions,
        "fixes": fixes,
        "by_tag": {tag: dict(counts) for tag, counts in tag_diff.items()},
    }
~~~

这段函数没有检查缺失 case 是否改变总体分布，也没有处理用户聚类和 judge 不确定
性；生产分析需要在它的输出上继续做对齐率、分层、区间和人工复核。

## 9. 从错误分析到修复和复验

### 9.1 数据修复

当根因是覆盖不足、冲突标注或 hard negative 缺失时，可以：

1. 补充真实失败样例并脱敏；
2. 构造边界条件和对抗性但合法的样本；
3. 清理重复、冲突和错误标签；
4. 调整多语言、领域或任务配比；
5. 针对时间、权限、格式和实体变化构造 hard negative；
6. 用新的等价 holdout 检查泛化。

不能把最终测试答案直接回填训练，然后继续用同一测试集证明提升。数据血缘、
训练修复集和未见评估集要分开记录。

### 9.2 Prompt、协议和输出约束

如果症状是格式错误、拒答边界或工具参数错误，修改 prompt 可能有效，但应
同时考虑结构化输出和服务端保护：

1. 明确字段、类型和必填项；
2. 把互相冲突的规则拆开；
3. 对不确定性和拒答条件给出可判断的边界；
4. 使用 schema validation 或 constrained decoding；
5. 对可恢复的解析失败设计有限重试；
6. 将工具描述、权限和错误返回写进协议测试。

prompt 修复一个 JSON case 后，可能让回答变长、过度拒答或影响多轮状态。所有
改动都要通过完整回归集和新的探索样本，而不是只重跑被修复的那一条。

### 9.3 RAG 和工具链修复

当根因位于证据或工具链时，训练模型往往不是第一选择。可以分别验证：

1. chunking 和文档版本；
2. embedding、retriever recall 和 reranker 顺序；
3. 时间、实体和权限过滤；
4. context packing、截断和重复内容；
5. claim 与 citation 的支持关系；
6. tool schema、参数校验和权限；
7. 外部状态幂等、错误恢复和重试；
8. observation summarization 和最终答案合成。

修复后必须保存“证据是否进入上下文”“工具是否收到正确参数”“最终输出是否
使用了证据”这三层结果，不能只看最终回答分数。

### 9.4 训练和对齐修复

如果控制变量实验显示模型本体在正确证据、稳定协议下仍然失败，才考虑：

1. SFT 数据补强；
2. 过程监督或 verifier；
3. 偏好数据和奖励模型校准；
4. 拒答与有用性之间的平衡；
5. 多任务或多语言配比；
6. 长上下文、代码、数学和工具轨迹训练。

训练改动需要更长的验证链：训练集、未见变体、regression suite、广泛 benchmark、
安全评估和线上实验。一个局部错误修复不能抵消另一个能力维度的退化。

### 9.5 评估器和 harness 修复

若人工审阅发现 judge 或解析器错判，应该修复评估器并重新计算历史结果，而不是
把评估器错误当成模型回归。若 trace 显示 context folding、工具返回或权限路由
错误，则应修 harness，并分别记录 harness 修复前后的结果。

模型、评估器和 harness 的版本要能独立回滚，否则新版本通过率提升可能只是评分
规则放宽，或者失败被吞掉。

### 9.6 修复后的确认

每个修复动作都要产生可验证的预测：

~~~text
root cause hypothesis
  -> targeted intervention
  -> expected changed slice
  -> held-out check
  -> regression check
  -> fresh exploration
~~~

例如假设“时间 hard negative 缺失”导致财报年份错误，那么修复后应在未见公司、
未见年份和跨年份比较任务上改善，而不只是原始 case 通过。若只有原始 case 改善，
可能是记忆或规则过拟合。

## 10. 自动化闭环和人工边界

### 10.1 最小批处理流程

一个可维护的评估流水线可以按以下顺序运行：

~~~text
版本或配置变更
  -> 核心 benchmark
  -> regression suite
  -> 样本级版本 diff
  -> 错误标签和切片汇总
  -> 高严重度与评估器分歧人工复核
  -> 形成修复假设
  -> 针对性实验
  -> 独立 holdout 和新样本复验
~~~

流水线的每个阶段都应产出中间 artifact。只保留最后的 dashboard 数字，会让
分析者无法回到某一条 trace 查看证据。

### 10.2 哪些任务适合自动化

适合自动化的内容包括：

1. 样本 schema 校验；
2. JSON、代码、数学和工具参数的程序判定；
3. 版本四象限和分层计数；
4. 标签频次、严重度和覆盖率汇总；
5. 已知 case 的回归运行；
6. 新增回归和高风险标签提醒；
7. trace、版本和数据血缘关联。

### 10.3 必须保留人工判断的地方

以下内容不能只交给自动分类：

1. 高影响安全和隐私事件；
2. 新出现且 taxonomy 未覆盖的失败；
3. judge 与人工或程序判定的严重分歧；
4. 医疗、法律、金融等领域的事实和建议；
5. 根因假设的最终确认；
6. 对损失函数和风险权重的业务解释。

自动化的目标是扩大证据覆盖，不是把判断责任藏进一个颜色或布尔字段。

### 10.4 线上反馈进入回归集

用户举报、重复提问、复制后修改、人工接管和工具回滚都可以成为候选失败来源。
但回流前应完成：

1. 隐私和权限检查；
2. 去重和事件关联；
3. 人工确认真实失败；
4. 归类错误类型和严重度；
5. 检查是否已经进入训练或调参；
6. 构造可公开或受限的回归版本。

线上反馈不是天然的事实标签。用户可能因为延迟、风格或 UI 点踩，也可能根本
不知道答案是否正确；要结合任务状态、证据和人工抽检。

## 11. 可运行的版本回归审计案例

下面的 toy 数据表示 10 个对齐后的 case。它故意包含两个修复、四个回归、两个
高严重度隐私/安全回归，以及 regression suite 对 reasoning 和 tool 能力覆盖不足
的情况。代码输出的是诊断信号和后续动作，不把所有信息压成一个总开关。

### 11.1 版本和错误数据

~~~python
from collections import Counter, defaultdict


cases = [
    {"id": "rag_finance", "old_ok": False, "new_ok": True,
     "error": "none", "severity": 0, "fixability": 0.0,
     "capability": "rag", "in_suite": True},
    {"id": "json_schema", "old_ok": True, "new_ok": False,
     "error": "formatting", "severity": 3, "fixability": 0.9,
     "capability": "schema", "in_suite": True},
    {"id": "safety_refusal", "old_ok": True, "new_ok": False,
     "error": "safety", "severity": 5, "fixability": 0.6,
     "capability": "safety", "in_suite": True},
    {"id": "math_word", "old_ok": False, "new_ok": False,
     "error": "reasoning", "severity": 4, "fixability": 0.5,
     "capability": "reasoning", "in_suite": False},
    {"id": "tool_call", "old_ok": False, "new_ok": True,
     "error": "none", "severity": 0, "fixability": 0.0,
     "capability": "tool", "in_suite": False},
    {"id": "citation", "old_ok": True, "new_ok": False,
     "error": "citation_error", "severity": 4, "fixability": 0.8,
     "capability": "rag", "in_suite": True},
    {"id": "stale_knowledge", "old_ok": False, "new_ok": False,
     "error": "retrieval_miss", "severity": 4, "fixability": 0.7,
     "capability": "rag", "in_suite": False},
    {"id": "harmless_chat", "old_ok": True, "new_ok": True,
     "error": "none", "severity": 0, "fixability": 0.0,
     "capability": "chat", "in_suite": False},
    {"id": "privacy", "old_ok": True, "new_ok": False,
     "error": "privacy", "severity": 5, "fixability": 0.7,
     "capability": "privacy", "in_suite": True},
    {"id": "code_edge", "old_ok": False, "new_ok": False,
     "error": "edge_case", "severity": 3, "fixability": 0.8,
     "capability": "code", "in_suite": False},
]


root_map = {
    "formatting": "prompt_schema",
    "safety": "policy_boundary",
    "privacy": "policy_boundary",
    "citation_error": "rag_grounding",
    "retrieval_miss": "rag_retrieval",
    "reasoning": "reasoning_data",
    "edge_case": "code_tests",
}
~~~

### 11.2 统计版本象限和错误簇

~~~python
buckets = Counter()
regressions = []
fixes = []
new_failures = []

for case in cases:
    if case["old_ok"] and case["new_ok"]:
        bucket = "stable_correct"
    elif not case["old_ok"] and case["new_ok"]:
        bucket = "fixed"
        fixes.append(case["id"])
    elif case["old_ok"] and not case["new_ok"]:
        bucket = "regressed"
        regressions.append(case["id"])
    else:
        bucket = "stable_wrong"

    buckets[bucket] += 1
    if not case["new_ok"]:
        new_failures.append(case)

error_counts = Counter(case["error"] for case in new_failures)
weighted_loss = defaultdict(float)
fixability = defaultdict(list)

for case in new_failures:
    error = case["error"]
    weighted_loss[error] += case["severity"]
    fixability[error].append(case["fixability"])

priority = {}
for error, count in error_counts.items():
    average_severity = weighted_loss[error] / count
    average_fixability = sum(fixability[error]) / len(fixability[error])
    priority[error] = round(
        count * average_severity * average_fixability, 3
    )

root_counts = Counter(root_map[case["error"]] for case in new_failures)
~~~

### 11.3 覆盖和后续动作

~~~python
recent_capabilities = {
    "rag", "safety", "schema", "tool", "reasoning", "privacy"
}
suite_capabilities = {
    case["capability"] for case in cases if case["in_suite"]
}
coverage = (
    len(recent_capabilities & suite_capabilities)
    / len(recent_capabilities)
)
missing_capabilities = sorted(
    recent_capabilities - suite_capabilities
)

high_regressions = [
    case["id"]
    for case in cases
    if case["old_ok"]
    and not case["new_ok"]
    and case["severity"] >= 4
]
safety_regressions = [
    case["id"]
    for case in cases
    if case["old_ok"]
    and not case["new_ok"]
    and case["error"] in {"safety", "privacy"}
]

signals = {
    "high_severity_regression": len(high_regressions) > 0,
    "safety_regression": len(safety_regressions) > 0,
    "suite_coverage": round(coverage, 3),
    "missing_capabilities": missing_capabilities,
    "fix_count": len(fixes),
    "regression_count": len(regressions),
}

actions = []
if high_regressions:
    actions.append("review_high_severity_cases")
if safety_regressions:
    actions.append("restore_safety_and_privacy_behavior")
if coverage < 0.80:
    actions.append("add_reasoning_and_tool_regression_cases")
if len(fixes) <= len(regressions):
    actions.append("retest_after_fix_before_broader_traffic")

decision = (
    "hold_for_high_severity_review"
    if actions
    else "continue_with_sliced_report"
)

print("version_buckets=", dict(buckets), sep="")
print("regressions=", regressions, sep="")
print("fixes=", fixes, sep="")
print("error_counts=", dict(error_counts), sep="")
print(
    "priority=",
    dict(sorted(priority.items(), key=lambda item: item[1], reverse=True)),
    sep="",
)
print("root_counts=", dict(root_counts), sep="")
print(
    "suite_coverage=",
    {"coverage": round(coverage, 3), "missing": missing_capabilities},
    sep="",
)
print("signals=", signals, sep="")
print("actions=", actions, sep="")
print("decision=", decision, sep="")
~~~

### 11.4 读懂输出

这组数据的版本象限应为：稳定正确 1、修复 2、回归 4、仍未解决 3。新版本的
四个回归中有安全、隐私和引用问题；回归集只覆盖近期六种能力中的四种，缺少
reasoning 和 tool。即便修复数为 2，也不能用修复数量抵消高严重度回归。

错误簇优先级把频次、严重度和可修复性放在一起，帮助安排工作顺序；它不代表
隐私问题的风险可以被格式问题抵消。正确的后续动作是先复核高严重度 case，恢复
安全和隐私行为，补齐回归覆盖，再在独立样本上确认修复没有造成新的退化。

## 12. 生产闭环中的常见坑

### 12.1 分类太粗或太细

只有“正确/错误”无法指导修复；但如果 taxonomy 有数百个互相重叠的标签，
标注员无法保持一致，历史趋势也会失真。可以先从 8 到 15 个一级标签开始，
对高价值任务增加少量二级标签，再根据真实分歧调整边界。

### 12.2 把 judge 解释当作根因

LLM judge 能生成“可能因为检索错误”的解释，但它通常没有看到完整运行链路。
解释应被保存为候选假设，并由上下文检查、日志、对照实验和人工复核验证。

### 12.3 Regression suite 越积越脏

长期运行的 suite 可能出现：

1. 过时政策和知识库；
2. 重复或近重复 case；
3. 期望答案已不再正确；
4. 只针对旧系统实现的脆弱规则；
5. 缺少 owner、来源和复核时间。

维护 suite 和增加 suite 同样重要。退休 case 需要保留原因和历史结果，避免趋势
断裂后无法解释。

### 12.4 只加测试，不修根因

把失败样例加入回归集只能防止已知问题再次出现。若不修根因，模型仍然会在同
类新样本上失败。修复动作必须有明确的预期切片和 holdout 验证。

### 12.5 用训练过的样例证明泛化

失败样例可以帮助构造训练数据，但同一条样例和其答案不能继续作为独立测试。应
保留等价但未见过的实体、年份、变量、语言或文档结构，并在报告里标明训练血缘。

### 12.6 只看已知问题

Regression suite 对历史错误很有效，却不覆盖未知未知。它必须和新样本探索、
线上反馈、长尾监控、人工抽检和安全红队组合使用。

### 12.7 忽略系统而只怪模型

RAG、Agent、多工具和多模态系统中，错误可能来自检索、上下文、权限、协议、
后处理、UI 或外部状态。只有在正确证据、稳定协议和可用工具下仍复现，才更有
理由把根因归到模型能力。

## 13. 方法的边界和适用场景

### 13.1 Error Analysis 的价值

它能把分数变化转成具体问题，帮助团队：

1. 发现局部退化；
2. 区分错误类型和业务严重度；
3. 形成可证伪的根因假设；
4. 指导数据、训练、prompt、检索和工具修复；
5. 产生新的能力切片和回归 case。

### 13.2 Error Analysis 的限制

它也有明确边界：

1. 主动挖掘样本不代表总体分布；
2. LLM 标注和 embedding 聚类可能带来偏差；
3. 少数失败样例不能证明普遍根因；
4. 根因假设需要干预或复现实验验证；
5. 已知回归集不能替代新分布评估；
6. 高严重度事件需要独立安全和人工流程。

### 13.3 什么时候最有用

错误分析尤其适合模型、prompt、RAG 索引、工具 schema、安全策略和多模态处理
频繁变化的系统，也适合企业客户场景稳定、单个失败代价高的产品。对于全新
产品，它应与探索性数据和线上观察同时建立，而不是等到 suite 完成后才开始。

## 14. 从 case 扩展为能力评估

### 14.1 不要只修一个例子

假设发现一个“年份约束检索失败”的样例。要测试模型是否真正学会能力，可以
生成同一结构的变体：

1. 不同公司和不同年份；
2. 同一年多个财务指标；
3. 跨年份比较；
4. 文档标题相似但年份不同；
5. 问题包含干扰年份；
6. 需要引用页码或段落；
7. 用户无权访问其中一个年份。

原始 case 通过，只能说明这个 case 在当前配置下通过。变体集合通过，才更接近
对时间、实体、权限和证据组合能力的验证。

### 14.2 从错误簇形成 benchmark

当一个错误簇持续出现，可以把它从 regression case 扩展成能力 benchmark：

~~~text
failure cluster
  -> capability definition
  -> controlled variants
  -> independent labels or verifier
  -> difficulty slices
  -> benchmark + regression coverage
~~~

这样错误分析会反哺评估体系，而不是让回归集无限堆积单条样本。

### 14.3 Eval-driven development

类似软件工程中先写测试再实现，大模型系统可以先明确失败行为、判定方法和变体
集合，再修改模型或链路。这个过程不是要求每个需求都提前写出完美测试，而是
让改动者在开始前知道：

1. 哪个行为预计会改变；
2. 哪些旧行为必须保持；
3. 哪些新风险需要观察；
4. 用什么证据确认修复泛化。

## 15. 练习：从失败样例到长期测试

### 练习一：总体提升与局部退化

一个客服模型总体满意度提升 1.5%，但投诉中出现更多“答非所问”。请设计日志
字段、错误 taxonomy、用户和任务切片、人工抽样以及根因干预，并说明怎样判断
这是新颖效应、延迟变化、答案长度变化还是模型理解退化。

### 练习二：企业知识库回归集

为一个企业 RAG 系统设计至少 10 个 regression case。每个 case 写出来源、权限、
知识库 snapshot、期望行为、证据要求、严重度、oracle 和变体；说明如何避免把
真实敏感文档直接暴露给训练或公开评估。

### 练习三：样本级版本 diff

两个版本在 1,000 个样本上的总分分别是 83% 和 85%。请构造四象限统计，按语言、
领域、难度和安全等级分层，并说明为什么一个高严重度回归不能被多个低风险修复
抵消。

### 练习四：Agent 工具错误

一个 Agent 经常调用错工具。请设计从 goal interpretation、planning、tool selection、
tool arguments、permission、observation reading 到 final answer 的标签链；再为
每个可能根因设计一个最小干预和一个未见 holdout。

### 练习五：评估器分歧

人工标注与 LLM judge 在高风险摘要任务上只有 70% 一致。请说明如何抽取分歧样本、
修订 rubric、计算每类指标、保留人工 gold set，并判断是 judge 偏差、参考答案
不足还是任务本身存在多种等价答案。

## 16. 资料与证据边界

错误分析和回归测试依赖完整 trace、稳定标签、可执行 oracle 和正确的版本对齐。
论文、开源评估框架和测试工具可以提供方法入口，但不会自动告诉我们某个企业
RAG、Agent 或多模态产品的根因。特别是 LLM judge 生成的解释、用户点赞、自动
聚类标签和厂商模型卡，都需要按证据等级使用。

以下资料可作为延伸阅读：

1. [OpenAI Evals](https://github.com/openai/evals)：开放式模型评估框架和 eval registry 入口。
2. [Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/)：HELM 的多维度模型评估和可复核报告。
3. [HELM 论文](https://arxiv.org/abs/2211.09110)：语言模型整体评估的任务、指标和场景框架。
4. [CheckList](https://arxiv.org/abs/2005.04118)：行为测试和能力切片设计方法。
5. [The ML Test Score](https://doi.org/10.1109/BigData.2017.8258038)：机器学习系统测试维度和工程检查思路的 IEEE 正式出版记录。
6. [pytest Documentation](https://docs.pytest.org/en/stable/how-to/usage.html)：软件回归测试运行与组织的基础工具文档。
7. [NIST Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/)：抽样、统计分析和不确定性方法的参考资料。
8. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：风险识别、测量、管理和治理的官方框架。

其中，CheckList 和 Google ML Test Score 更接近行为测试与工程质量维度；OpenAI
Evals 和 HELM 提供评估框架或评测结果入口；pytest 只解决测试运行，不等于模型
质量判断；NIST 框架提供风险治理语言，也不能替代具体业务的错误 oracle。资料
用途和证据等级需要在正文之外继续维护。

## 17. 结语：让失败成为可积累的知识

评估分数只能告诉我们系统在某个样本和某个指标上的平均表现。Error Analysis
把失败样例还原成用户可见症状、能力标签、证据链和根因假设；聚类和主动挖掘让
团队在大量日志中找到值得阅读的模式；对照实验和 trace 回放则把“可能原因”
变成可以被支持或推翻的假设。

Regression suite 把已经确认的重要失败保留下来，用稳定的 oracle、版本血缘、
严重度和维护者保护它们不再复发。样本级 diff 进一步告诉我们新版本修复了什么、
退化了什么、哪些能力仍然没有变化。但回归集不是全部世界，它必须和新样本探索、
线上反馈、人工复核、统计分析和安全评估一起使用。

真正成熟的闭环不是“发现一个错误，加入一个 case”这么简单，而是：先确认用户
看到的失败，再区分模型、harness、协议、证据和评估器，提出可证伪根因，设计
针对性干预，用未见变体验证泛化，最后把高价值问题沉淀成长期测试。这样每一次
失败都会改善下一次实验，而不是只在报告里留下一个短暂的红色数字。
