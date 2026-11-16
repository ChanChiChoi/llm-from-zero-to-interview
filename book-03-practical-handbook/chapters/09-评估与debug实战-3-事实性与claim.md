# 第 9 章（3）：事实性、主张与证据

即使测试集没有泄漏，分数也不一定等于事实正确。开放式回答可能说得流畅、覆盖了部分要求，却在关键条件上补错了内容。这一篇把回答拆成可核验的主张（claim），再分别看证据支持、必需信息覆盖和拒答是否合适。

## 9.3 幻觉与事实性：从流畅文本追溯到可核验 claim

### 9.3.1 “说得像真的”不是事实性

语言模型的训练目标鼓励它生成在语言和上下文中最可能出现的 token，而不是为每一句话附上经过外部世界验证的证明。因此，语法流畅、语气自信、段落结构完整，都不能单独说明内容为真。

在产品中，“幻觉”常常被混成一个过大的标签。至少应区分四类现象：

1. 事实错误：回答与可靠事实或给定证据矛盾。
2. 无依据陈述：回答没有直接矛盾，但证据并不支持它。
3. 证据错配：引用存在，却没有支持对应的 claim，或引用了错误版本。
4. 任务偏离：内容可能为真，但没有回答问题，或者把推测写成确定结论。

例如，用户问“政策中规定的退款到账时间是多少”，模型回答“通常三个工作日，节假日可能延长”。如果文档只写了“三个工作日”，后半句可能是常识性推测，也可能是业务规则；在事实性评估里，它不能因为听起来合理就自动被视为正确。

### 把一段回答拆成可核对的主张

一整段回答通常包含多个事实。先把它拆成 claim，再为每个 claim 找支持它的证据：

```text
回答：退款通常三个工作日到账，特殊支付渠道可能需要五个工作日。

claim 1：通常三个工作日到账。
claim 2：特殊支付渠道可能需要五个工作日。
```

如果给定资料只支持 claim 1，就不能因为 claim 1 正确而把整段判为完全正确。相反，一条回答也可能有少量措辞差异，但所有关键 claim 都受到证据支持。

### 回答、证据和事实状态是三件事

对回答 $a$ 做 claim 分解，得到 $C(a)=\{c_j\}_{j=1}^{k}$。给定证据集合 $E$，定义支持关系：

```math
\mathrm{support}(e,c_j)\in\{0,1,\mathrm{contradict}\}
```

这不是语言模型天然提供的标签，而是需要规则、信息抽取、检索匹配、人工或另一个评估模型判断。至少要保存“哪一段证据支持哪一个 claim”，否则一个总分无法定位问题。

对于给定证据的 groundedness，可以定义加权支持率：

```math
P_{\mathrm{support}}(a\mid E)=
\frac{\sum_j w_j\mathbb{1}[\exists e\in E:\mathrm{support}(e,c_j)=1]}{\sum_j w_j}
```

这衡量回答中有多少内容有依据，不衡量回答是否完整。若要衡量问题所需事实的覆盖，应另外定义由问题和参考资料确定的必需事实集 $Q=\{q_j\}$，并为其指定有限正权重 $v_j$。完整性可以写成：

```math
R_{\mathrm{required}}(a\mid Q,E)=
\frac{\sum_{q_j\in Q}v_j\mathbb{1}[a\ \mathrm{covers}\ q_j]\mathbb{1}[E\ \mathrm{supports}\ q_j]}{\sum_{q_j\in Q}v_j}
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

设正确回答收益为 $U_c$，无依据回答损失为 $L_h$，合理拒答收益为 $U_a$，过度拒答损失为 $L_o$，则一个简化的期望效用可以写成：

```math
U=\Pr(c)U_c-\Pr(h)L_h+\Pr(a)U_a-\Pr(o)L_o
```

高风险领域通常令 $L_h$ 很大，于是系统宁可在不确定时请求更多证据；低风险创作任务则可能更重视完成率。阈值应由任务后果决定，不应把一个领域的拒答策略复制到所有领域。

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
R_{\mathrm{hall}}=\frac{\sum_j w_j\mathbb{1}[\mathrm{unsupported}(c_j)]}{\sum_j w_j},\qquad
w_j=w_{\mathrm{base}}(c_j)\cdot w_{\mathrm{risk}}(c_j)
```

这里同样要求 claim 集合非空且 $\sum_jw_j>0$；否则幻觉风险率没有定义。这个权重设计让“低风险细节很多、高风险错误一个”的回答不会被简单平均掩盖。

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

### 9.3.6 幻觉的故障归因

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

常见归因与修复方向如下：

- 没检索到证据：改进查询改写、chunk、过滤和召回，不要只提高生成温度。
- 检索到旧版本：保存文档版本、更新时间和有效期，并在检索层做时间过滤。
- 证据冲突：让系统显式呈现冲突，要求模型说明无法确定，而不是强行合并。
- 证据充分但回答错：检查上下文顺序、长上下文位置、提示词约束和模型能力。
- 正确回答后追加无依据内容：限制输出结构，要求每个外部事实绑定证据，增加 claim-level 检查。
- 工具未执行却声称完成：让执行器返回不可伪造的状态，并在程序层阻止模型自行生成成功标记。

### 9.3.7 评估工具的证据边界

[TruthfulQA](https://arxiv.org/abs/2109.07958) 适合观察模型是否会复述常见错误观念，但它不是企业知识库事实性测试；[FActScore](https://arxiv.org/abs/2305.14251)把长文本拆成原子事实并检查支持关系，强调了 claim 粒度的重要性；[SelfCheckGPT](https://arxiv.org/abs/2303.08896)把多次采样不一致作为无外部知识时的风险信号，但不能把一致性当作真值证明。

事实性评估的最终结论应注明：证据来自给定文档还是开放网络，claim 如何抽取，否定和条件如何处理，评估模型是否经过人工校准，以及模型拒答是否算作成功。只有这样，“幻觉率下降”才是可解释的工程结论。

当评分规则和证据口径都清楚后，下一步是判断失败发生在模型训练过程还是评测器。接下来读[训练不收敛的系统排查](09-评估与debug实战-4-训练不收敛.md)，从可复现的小数据训练开始定位。
