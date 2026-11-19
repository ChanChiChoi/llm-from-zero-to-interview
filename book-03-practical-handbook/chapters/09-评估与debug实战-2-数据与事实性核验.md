# 第 9 章（第二部分）：数据可信度与事实性核验

固定题目只是比较的起点，还不能证明这批题目没有进入训练过程。如果测试答案经由训练语料、开发过程、检索上下文或评分器提前暴露，模型即使没学会目标能力，也可能得到漂亮分数。本节沿着数据和评测的信息流检查泄漏，并说明训练集不可见时能下到什么结论。

## 9.2 数据污染与评测泄漏：先确认分数测的是什么

### 9.2.1 一个看似优秀的结果

某模型在一个公开问答集上从 72 分升到 86 分。团队先后做过继续预训练、提示词优化、检索增强和评分器重写。若没有进一步检查，我们无法知道 14 分提升来自模型能力、检索带来的外部答案、评测脚本的宽松处理，还是测试样例已经进入训练语料。

数据污染不是一个只存在于论文里的词。企业项目中同样会发生：把线上回流数据未经隔离地加入训练集，把 test 的人工修订答案放入提示词，把同一客户的相邻订单同时分到训练和验证，把检索库的未来版本带入历史评估，或者因为调参反复查看 private holdout 而逐渐适应它。

### 先问答案有没有提前进入评测流程

最容易理解的例子是训练集和测试集出现同一道题。模型即使不是逐字记住，也可能记住题目的特殊措辞、答案顺序或上下文模板。评估时它表现很好，但遇到同一知识的不同表达未必同样可靠。

泄漏还可能发生在评估程序内部。例如参考答案被拼进 prompt，评分器把模型输出中的任何数字都当作正确，或者 RAG 系统在评估时检索到了包含参考答案的文档。此时问题不是模型“学得好”，而是测量路径给了它额外信息。

### 沿信息流逐段找边界

对每个样例，画出信息可能经过的路径：

```text
原始数据
  ├─> 预训练 / SFT / 偏好数据
  ├─> 提示词模板与 few-shot 示例
  ├─> 检索索引与工具返回
  ├─> 评估样本与参考答案
  └─> 评分器、人工指南和错误修订
```

如果从评估结果反向流入训练、提示词、检索库或评分器，就形成了不同程度的泄漏。需要区分：

1. 训练污染：评测内容或近似内容出现在模型训练语料。
2. 切分泄漏：同一实体、同一文档或高度相似样本跨越 train/validation/test。
3. 时间泄漏：用评估时点之后才产生的信息训练或检索。
4. 检索泄漏：评估上下文包含参考答案或由参考答案构造的文本。
5. 提示词泄漏：调参时把 test 失败样例、答案或标签写入固定模板。
6. 评分泄漏：评分规则与模型输出形式过度耦合，造成投机行为。
7. 标注泄漏：人工复核后的答案被当作普通训练数据，未保留独立版本。

这些类别的风险不同，但共同特征是评估时模型获得了不应获得的信息，或者评估者利用了测试结果改变了被测系统。

### 9.2.2 文档切分与时间切分

随机按行切分对独立同分布的简单数据有用，但对文档、用户、订单和会话数据常常不够。应先确定泄漏单元：同一文档、同一客户、同一问题模板、同一事件或同一时间窗口的数据应尽量放在同一分区。

时间敏感任务可以使用时间切分：

```math
D_{\mathrm{train}}=\{z:t(z)\lt T_{1}\},\qquad
D_{\mathrm{valid}}=\{z:T_{1}\le t(z)\lt T_{2}\},\qquad
D_{\mathrm{test}}=\{z:t(z)\ge T_{2}\}
```

如果任务需要回答“上线后未来数据表现如何”，时间切分比随机切分更接近真实部署。它也可能让分布变化变得明显，因此报告里要同时给出时间段、业务版本和样本量。

对于实体泄漏，先按实体分组再切分：

```math
\mathrm{group}(z_{i})=g_{i},\qquad
\mathrm{partition}(g_{i})\in\{\mathrm{train},\mathrm{valid},\mathrm{test}\}
```

同一个 $g_{i}$ 不得出现在多个分区。客服场景中的 `customer_id`、合同场景中的 `contract_id`、代码场景中的 `repository_id` 都可能是这种分组键。

### 9.2.3 近重复检测：从精确哈希到 shingle 相似度

完全相同的文本可以用规范化后哈希检测。规范化不能删除会改变含义的数字、否定词和单位，只能处理确定的空格、换行和 Unicode 形式。

对改写或格式变化，需要使用 token 或字符 shingle。设文本 $(a,b)$ 的 shingle 集合为 $S(a),S(b)$，Jaccard 相似度为：

```math
J(a,b)=\frac{|S(a)\cap S(b)|}{|S(a)\cup S(b)|}
```

例如比较“退款需要三个工作日到账”和“退款需要 3 个工作日到账”，去掉空格和标点后按连续 3 个字符切分：两边各有 9 个 shingle，其中 6 个相同，联合集合有 12 个，因此 $J=6/12=0.5$。若候选阈值设为 0.35，这一对会进入人工复核；“三”和“3”的表达差异只说明存在近似，不足以证明训练污染。

当 $J(a,b)$ 超过预设阈值，可以把它们标记为候选近重复，再由规则或人工确认。阈值不是定律：代码、数字表格和中文短句的相似度分布不同，应在标注样本上校准。

大规模数据可以用 MinHash、LSH 或 embedding 检索降低全量两两比较成本；这些方法可能漏检或误报，因此最后仍需保留原文片段和人工抽样。

### 9.2.4 一个零依赖近重复扫描器

下面的程序演示规范化、字符 shingle 和跨分区候选发现。它不是生产级去重器，却足以说明为什么“改了空格”不能被当作新样本，以及为什么结果需要人工确认。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import hashlib
import re


ROWS = [
    {"id": "a1", "split": "train", "text": "退款需要三个工作日到账。"},
    {"id": "a2", "split": "test", "text": "退款需要 3 个工作日到账。"},
    {"id": "b1", "split": "train", "text": "LayerNorm 稳定激活分布。"},
    {"id": "b2", "split": "test", "text": "模型使用 LayerNorm 来稳定激活分布。"},
    {"id": "c1", "split": "test", "text": "完全不同的查询。"},
]


def normalize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    text = text.lower().replace("３", "3")
    text = re.sub(r"[\s，。,:：；;]+", "", text)
    if not text:
        raise ValueError("text is empty after normalization")
    return text


def shingles(text, size=3):
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("shingle size must be a positive integer")
    text = normalize(text)
    if len(text) <= size:
        return {text}
    return {text[i:i + size] for i in range(len(text) - size + 1)}


def jaccard(left, right):
    if not isinstance(left, set) or not isinstance(right, set):
        raise TypeError("jaccard expects two sets")
    union = left | right
    if not union:
        raise ValueError("Jaccard similarity is undefined for an empty union")
    return len(left & right) / len(union)


def digest(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


for row in ROWS:
    row["digest"] = digest(row["text"])
    row["shingles"] = shingles(row["text"])

for index, left in enumerate(ROWS):
    for right in ROWS[index + 1:]:
        if left["split"] == right["split"]:
            continue
        similarity = jaccard(left["shingles"], right["shingles"])
        if left["digest"] == right["digest"] or similarity >= 0.35:
            print(left["id"], right["id"], round(similarity, 3))
```
这个示例中的“3”和“三”未必能被同一套规范化规则判断为相同事实；如果业务语料经常出现数字变体，应增加领域规则或使用 token 级比较。不要为了提高召回把所有数字都删除，因为“3 天”和“30 天”会因此产生危险的假相似。

### 9.2.5 训练集未知时如何调查污染

闭源模型通常不会公开完整训练语料，因此不能从“没有看到训练数据清单”推出“没有污染”。更稳妥的证据分层是：

```text
强证据：训练快照、数据清单、哈希和时间记录可审计
中等证据：去重报告、时间切分、成员样本审查、私有 holdout
弱证据：只观察公开 benchmark 分数或模型输出风格
```

可以采用几种互补方法：

- 在评估集建立私有、后加入、不可公开的样本，并通过访问权限隔离；
- 对训练语料做 exact hash 和近重复扫描；
- 用模型对训练样例和新改写样例的置信度、生成概率或逐 token loss 做差异分析；
- 将题目改写、改变实体和顺序，观察性能是否只对原始表面形式高；
- 用时间切分和新近数据做外部验证；
- 对异常高分样例进行逐条人工追踪，确认是否是评分器或检索路径泄漏。

单独的 membership inference 结果也不能直接给出“污染比例”。它可能受模型规模、采样方式、温度、重复程度和背景频率影响。调查结论应写成证据与不确定性，而不是把一次统计异常宣布为确定事实。

### 9.2.6 评估程序也需要被测试

数据泄漏不只在数据文件里。应给评分器和 runner 增加负向测试：

```text
输入为空时不得读取 reference
reference 改变时，模型 prompt 不应自动改变
context 不包含答案时，检索器不能读到隐藏 gold 文件
输出添加无关正确关键词时，分数不应无条件上升
test 样例不应写入训练缓存、few-shot 缓存或 prompt 日志模板
```

可以把评估系统看作被测软件，给它做单元测试和审计日志。一个分数只有在“数据来源、执行路径、评分器行为”都清楚时，才有解释价值。

### 9.2.7 评测污染的报告写法

研究或工程报告至少应说明：

1. 评测集发布日期、版本和是否公开；
2. 训练、提示词示例、检索库和评测集的时间关系；
3. 是否做过 exact/near-duplicate 检查以及阈值；
4. 是否保留私有 holdout 和时间外推集；
5. 哪些结果是污染调查，哪些只是风险提示；
6. 评测脚本是否能访问 reference、未来文档或人工修订结果。

这样即使无法证明“完全没有污染”，读者也能判断证据强度和剩余风险。

近重复扫描只能生成候选，不能直接给污染盖章。人工复核时要同时看题目、答案或解释、来源日期和训练路径：同一模板但不同条件可能是合法重叠，换了实体或表达方式却复用了完整答案则可能被精确匹配漏掉。实际报告应把候选比例与确认重叠比例分开，说明抽查方法和分母；如果 12,000 条样本里扫描器标出 47 个候选，不能把 47 条全当作已污染，也不能因为只确认 3 条就声称其余数据没有问题。

### 9.2.8 资料与边界

[MMLU 论文](https://arxiv.org/abs/2009.03300)和 [BIG-bench 论文](https://arxiv.org/abs/2206.04615)说明了公开 benchmark 的规模化价值，但公开题目也带来训练可见性和复现条件问题。[HELM](https://arxiv.org/abs/2211.09110)强调透明记录和多维度报告。数据污染研究没有一个适用于所有模型、所有数据形态的单一检测器；本节的近重复公式和代码用于建立排查路径，不应被解释为“阈值以上就证明训练污染”。

针对固定题面，2025 年的 [Data Contamination Quiz](https://aclanthology.org/2025.tacl-1.37/) 将原题、三个词级扰动版本和“以上都不是”选项组成选择题，以模型辨认原题的行为作为特定数据分区的污染信号；题面变体的语义与难度、位置偏差仍需检查，不能把分数直接当成语料重叠率。2026 年 ICLR 的 [污染检测脆弱性研究](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a3860475ddbfb8c644c43f0dcd266b66-Abstract-Conference.html) 在其推理模型设置中报告，短程 GRPO 可隐藏部分检测信号，某些 CoT 污染设置下多数检测器接近随机水平；这说明未检出不能证明未污染，结论仍限于论文测试的训练流程和模型。另一个 2026 年 ICLR 工作 [BeyondBench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22b4e30a7a660b17f1fb58ab49671e77-Abstract-Conference.html) 以算法生成题和确定性 verifier 降低固定题面记忆风险，适用于可形式化生成和验证的任务，不因此覆盖开放问答或真实业务流程。

排除已知的数据泄漏后，模型仍可能在事实题上答错。接下来读事实性、主张与证据，把流畅回答拆成 claim，逐条核对证据与完整性。

即使测试集没有泄漏，分数也不一定等于事实正确。开放式回答可能说得流畅、覆盖了部分要求，却在关键条件上补错了内容。接下来把回答拆成可核验的主张（claim），再分别看证据支持、必需信息覆盖和拒答是否合适。

## 9.3 幻觉与事实性：把流畅回答拆成可核对的主张

### 9.3.1 “说得像真的”不是事实性

语言模型的训练目标鼓励它生成在语言和上下文中最可能出现的 token，而不是为每一句话附上经过外部世界验证的证明。因此，语法流畅、语气自信、段落结构完整，都不能单独说明内容为真。

在产品中，“幻觉”常常被混成一个过大的标签。至少应区分四类现象：

1. 事实错误：回答与可靠事实或给定证据矛盾。
2. 无依据陈述：回答没有直接矛盾，但证据并不支持它。
3. 证据错配：引用存在，却没有支持对应的 claim，或引用了错误版本。
4. 任务偏离：内容可能为真，但没有回答问题，或者把推测写成确定结论。

例如，用户问“政策中规定的退款到账时间是多少”，模型回答“通常三个工作日，节假日可能延长”。如果文档只写了“三个工作日”，后半句可能是常识性推测，也可能是业务规则；在事实性评估里，它不能因为听起来合理就自动被视为正确。

### 把一段回答拆开逐条核对

一整段回答通常包含多个事实。先把它拆成 claim，再为每个 claim 找支持它的证据：

```text
回答：退款通常三个工作日到账，特殊支付渠道可能需要五个工作日。

claim 1：通常三个工作日到账。
claim 2：特殊支付渠道可能需要五个工作日。
```

如果给定资料只支持 claim 1，就不能因为 claim 1 正确而把整段判为完全正确。相反，一条回答也可能有少量措辞差异，但所有关键 claim 都受到证据支持。

### 回答、证据和事实状态是三件事

对回答 $a$ 做 claim 分解，得到 $C(a)=\{c_{j}\}_{j=1}^{k}$。给定证据集合 $E$，定义支持关系：

```math
\mathrm{support}(e,c_{j})\in\{0,1,\mathrm{contradict}\}
```

这不是语言模型天然提供的标签，而是需要规则、信息抽取、检索匹配、人工或另一个评估模型判断。至少要保存“哪一段证据支持哪一个 claim”，否则一个总分无法定位问题。

对于给定证据的 groundedness，可以定义加权支持率：

```math
P_{\mathrm{support}}(a\mid E)=
\frac{\sum_{j} w_{j}\mathbb{1}[\exists e\in E:\mathrm{support}(e,c_{j})=1]}{\sum_{j} w_{j}}
```

这衡量回答中有多少内容有依据，不衡量回答是否完整。若要衡量问题所需事实的覆盖，应另外定义由问题和参考资料确定的必需事实集 $Q=\{q_{j}\}$，并为其指定有限正权重 $v_{j}$。完整性可以写成：

```math
R_{\mathrm{required}}(a\mid Q,E)=
\frac{\sum_{q_{j}\in Q}v_{j}\mathbb{1}[a\ \mathrm{covers}\ q_{j}]\mathbb{1}[E\ \mathrm{supports}\ q_{j}]}{\sum_{q_{j}\in Q}v_{j}}
```

两式分别要求回答 claim 的总权重大于 0、必需事实集的总权重大于 0；缺少相应集合时报告“不适用”。若回答有 claim，但证据不支持其中任何一条，支持率是 0，而不是“不适用”。例如，问题要求普通和特殊渠道两种到账时间，等权必需事实集含 2 条；回答只正确说明普通渠道，则完整性为 $1/2=0.5$。若回答还加入一个“当天到账”的无依据 claim，回答两条 claim 中只有一条受支持，则等权支持率也为 $1/2=0.5$；没有这个额外 claim 时支持率为 $1/1=1$。这里的完整性是教学定义，不能当作 FActScore 的原始指标；写清分子、分母和 claim 抽取规则，比背一个指标名更重要。

### 9.3.2 开放世界、闭合证据集与拒答

“不知道”在不同任务中有不同含义。

- 闭合证据问答：答案必须由给定文档支持，文档没有说明时应说明证据不足。
- 开放世界问答：允许使用模型知识和外部搜索，但需要记录信息来源和时间。
- 创作任务：事实性不是唯一目标，虚构内容可能是任务要求；仍需避免把虚构叙述伪装成事实。
- 工具任务：模型应报告工具真实返回的状态，不能把计划或调用意图写成已完成动作。

因此，评估“幻觉率”前要先定义可用信息边界。对闭合证据集，一个保守决策是：

```math
d(a,E)=
\begin{cases}
\mathrm{answer}, & \mathrm{all\_supported}\\
\mathrm{qualified}, & \mathrm{partially\_supported}\\
\mathrm{abstain}, & \mathrm{insufficient\_or\_contradictory}
\end{cases}
```

拒答不是无条件越多越好。一个总是说“资料不足”的系统可能没有幻觉，却也无法完成有证据的任务。应同时测量有效回答率、正确回答率、证据不足时的拒答率和有证据时的过度拒答率。

设正确回答收益为 $U_{c}$，无依据回答损失为 $L_{h}$，合理拒答收益为 $U_{a}$，过度拒答损失为 $L_{o}$，则一个简化的期望效用可以写成：

```math
U=\Pr(c)U_{c}-\Pr(h)L_{h}+\Pr(a)U_{a}-\Pr(o)L_{o}
```

高风险领域通常令 $L_{h}$ 很大，于是系统宁可在不确定时请求更多证据；低风险创作任务则可能更重视完成率。阈值应由任务后果决定，不应把一个领域的拒答策略复制到所有领域。

### 9.3.3 事实性评估的四个层次

#### 层次一：字符串与结构检查

日期、金额、编号、枚举值和工具状态适合先做确定性检查。例如从回答中提取订单号，验证它是否属于当前用户；解析 JSON，检查金额是否为非负数；检查回答中的引用 ID 是否存在于检索结果。确定性检查便宜且可重复，应优先于昂贵的语义评估。

#### 层次二：claim 与证据对齐

将回答拆成原子事实，分别判断支持、矛盾或无法判断。一个 claim 可能需要多个证据片段共同支持；“有引用”不等于“引用支持”。例如引用文档的标题而不引用具体条款，不能证明数字和例外条件正确。

#### 层次三：与外部事实或参考答案比较

TruthfulQA 这类任务关注模型是否会复述常见但错误的说法；FActScore 则把长回答拆成原子事实后计算可验证程度。参考答案本身也可能过时或不完整，所以评估时要记录参考来源、更新时间和审校责任。

#### 层次四：不确定性与行为稳定性

同一问题重复采样时，如果答案在关键事实之间来回变化，说明模型的置信表达可能不可靠。SelfCheckGPT 等方法利用多次采样的一致性作为无外部知识时的风险信号，但一致并不等于正确：模型可能稳定地产生同一个错误。多次采样是诊断线索，不是事实证明。

### 9.3.4 一个可解释的 claim 评估表

对每个回答建立如下记录：

```text
case_id
claim_id
claim_text
claim_type: fact / number / condition / action_state / opinion
evidence_ids
relation: supported / contradicted / insufficient / not_applicable
severity: low / medium / high
annotator_or_judge
confidence
```

数字、时间、权限、医疗建议和外部动作状态通常应提高严重度。一个无关的形容词错误，和“退款已经完成”这种动作状态错误，不应在总分里拥有相同权重。

可以把错误严重度纳入风险加权：

```math
R_{\mathrm{hall}}=\frac{\sum_{j} w_{j}\mathbb{1}[\mathrm{unsupported}(c_{j})]}{\sum_{j} w_{j}},\qquad
w_{j}=w_{\mathrm{base}}(c_{j})\cdot w_{\mathrm{risk}}(c_{j})
```

这里同样要求 claim 集合非空且 $\sum_{j}w_{j}>0$；否则幻觉风险率没有定义。这个权重设计让“低风险细节很多、高风险错误一个”的回答不会被简单平均掩盖。

### 9.3.5 零依赖 claim 支持检查器

下面的程序不是自然语言推理模型，只是把人工定义的关键词证据映射成可重复的初筛结果。它的价值在于展示数据结构和失败标签，而不是替代人工或专业审查。

> 代码状态：静态审阅，未运行（2026-10-30）

```python
import json
import math


CASES = [
    {
        "id": "r1",
        "answer": "退款通常三个工作日到账，特殊支付渠道可能需要五个工作日。",
        "evidence_text": "退款通常三个工作日到账，特殊支付渠道可能需要五个工作日。",
        "claims": [
            {"text": "通常三个工作日到账", "evidence": ["三个工作日"], "weight": 1.0},
            {"text": "特殊支付渠道可能五个工作日", "evidence": ["特殊支付渠道", "五个工作日"], "weight": 1.5},
        ],
    },
    {
        "id": "r2",
        "answer": "退款会在当天到账。",
        "evidence_text": "退款通常三个工作日到账，特殊支付渠道可能需要五个工作日。",
        "claims": [
            {"text": "当天到账", "evidence": ["当天到账"], "weight": 1.0},
        ],
    },
]


def evaluate_case(case):
    if not isinstance(case, dict):
        raise TypeError("case must be a dictionary")
    if not isinstance(case.get("id"), str) or not case["id"].strip():
        raise ValueError("case id must be a non-empty string")
    if not isinstance(case.get("answer"), str) or not case["answer"].strip():
        raise ValueError("answer must be a non-empty string")
    if not isinstance(case.get("evidence_text"), str) or not case["evidence_text"].strip():
        raise ValueError("evidence_text must be a non-empty string")
    claims = case.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("a case must contain at least one claim")
    supported_weight = 0.0
    total_weight = 0.0
    details = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise TypeError("each claim must be a dictionary")
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            raise ValueError("claim text must be a non-empty string")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("each claim needs non-empty evidence pieces")
        if any(not isinstance(piece, str) or not piece.strip() for piece in evidence):
            raise ValueError("evidence pieces must be non-empty strings")
        evidence_hit = all(piece in case["evidence_text"] for piece in evidence)
        weight = claim.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise TypeError("claim weight must be numeric")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("claim weight must be a finite positive number")
        total_weight += weight
        supported_weight += weight * evidence_hit
        details.append({
            "claim": claim["text"],
            "relation": "supported" if evidence_hit else "insufficient",
        })
    if total_weight <= 0:
        raise ValueError("claim weight sum must be positive")
    score = supported_weight / total_weight
    return {"case_id": case["id"], "score": round(score, 3), "details": details}


report = [evaluate_case(case) for case in CASES]
print(json.dumps(report, ensure_ascii=False, indent=2))
```
若按给定数据手算，r1 的支持权重为 $1.0+1.5=2.5$，总权重也是 2.5，支持率为 1.0；r2 的支持权重为 0，总权重为 1.0，支持率为 0.0。答案中的 claim 和证据短语都是人工给定的；程序不检查 claim 是否忠实摘自答案、不自动抽取 claim，也不判断短语与 claim 是否语义等价。把政策改成“退款不是当天到账”，r2 仍会误判为支持，这说明字符串存在不等于语义支持。真实实现应保存文档 ID、版本和片段，并对否定、条件、时间和数字做语义判断；无检索结果时也应保留样例，单独标记证据缺失。

### 9.3.6 定位幻觉是在哪一步产生的

看到错误答案时，不要直接把原因归为“模型幻觉”。至少沿着以下路径复现：

```text
用户输入是否歧义？
  ↓
检索是否找到了正确文档、版本和权限范围？
  ↓
上下文是否包含支持 claim 的完整条款？
  ↓
提示词是否要求区分事实、推测和证据不足？
  ↓
模型是否正确读取上下文？
  ↓
解码是否引入随机差异或过长续写？
  ↓
后处理是否删掉了引用、否定或单位？
```

常见问题和修复方向如下：

- 没检索到证据：改进查询改写、chunk、过滤和召回，不要只提高生成温度。
- 检索到旧版本：保存文档版本、更新时间和有效期，并在检索层做时间过滤。
- 证据冲突：让系统显式呈现冲突，要求模型说明无法确定，而不是强行合并。
- 证据充分但回答错：检查上下文顺序、长上下文位置、提示词约束和模型能力。
- 正确回答后追加无依据内容：限制输出结构，要求每个外部事实绑定证据，增加 claim-level 检查。
- 工具未执行却声称完成：让执行器返回不可伪造的状态，并在程序层阻止模型自行生成成功标记。

### 9.3.7 评估工具的证据边界

[TruthfulQA](https://arxiv.org/abs/2109.07958) 适合观察模型是否会复述常见错误观念，但它不是企业知识库事实性测试；[FActScore](https://arxiv.org/abs/2305.14251)把长文本拆成原子事实并检查支持关系，强调了 claim 粒度的重要性；[SelfCheckGPT](https://arxiv.org/abs/2303.08896)把多次采样不一致作为无外部知识时的风险信号，但不能把一致性当作真值证明。

事实性评估的最终结论应注明：证据来自给定文档还是开放网络，claim 如何抽取，否定和条件如何处理，评估模型是否经过人工校准，以及模型拒答是否算作成功。只有这样，“幻觉率下降”才是可解释的工程结论。

当评分规则和证据口径都清楚后，下一步是判断失败发生在模型训练过程还是评测器。接下来读[训练不收敛的系统排查](09-评估与debug实战-3-训练调试与能力回归.md)，从可复现的小数据训练开始定位。
