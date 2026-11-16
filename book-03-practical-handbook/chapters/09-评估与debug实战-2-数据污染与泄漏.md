# 第 9 章（2）：数据污染与评测泄漏

固定题目只是比较的起点，还不能证明这批题目没有进入训练过程。如果测试答案经由训练语料、开发过程、检索上下文或评分器提前暴露，模型即使没学会目标能力，也可能得到漂亮分数。这一篇沿着数据和评测的信息流检查泄漏，并说明训练集不可见时能下到什么结论。

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
D_{\mathrm{train}}=\{z:t(z)<T_1\},\qquad
D_{\mathrm{valid}}=\{z:T_1\le t(z)<T_2\},\qquad
D_{\mathrm{test}}=\{z:t(z)\ge T_2\}
```

如果任务需要回答“上线后未来数据表现如何”，时间切分比随机切分更接近真实部署。它也可能让分布变化变得明显，因此报告里要同时给出时间段、业务版本和样本量。

对于实体泄漏，先按实体分组再切分：

```math
\mathrm{group}(z_i)=g_i,\qquad
\mathrm{partition}(g_i)\in\{\mathrm{train},\mathrm{valid},\mathrm{test}\}
```

同一个 $g_i$ 不得出现在多个分区。客服场景中的 `customer_id`、合同场景中的 `contract_id`、代码场景中的 `repository_id` 都可能是这种分组键。

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

排除已知的数据泄漏后，模型仍可能在事实题上答错。接下来读[事实性、主张与证据](09-评估与debug实战-3-事实性与claim.md)，把流畅回答拆成 claim，逐条核对证据与完整性。
