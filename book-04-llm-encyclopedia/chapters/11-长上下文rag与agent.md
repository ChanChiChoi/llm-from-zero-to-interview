# 第 11 章 长上下文、RAG 与 Agent：从信息组织到可靠行动

一个语言模型的输入窗口变长，并不意味着它突然获得了数据库、搜索引擎和操作系统的全部能力。它只是能够在一次调用中接收更多 token。系统仍然要回答几个具体问题：哪些信息值得放入窗口，证据是否真的被模型使用，外部动作是否经过授权，失败后如何恢复，最后的答案和动作能否被复核。

长上下文、检索增强生成（Retrieval-Augmented Generation，RAG）和 Agent 经常一起出现，但它们解决的是三个不同层面的问题：长上下文扩大一次推理可见的信息容量，RAG 从外部数据中选择证据，Agent 则围绕目标重复进行判断、调用工具和更新状态。把三者混成“把更多文本塞给模型”会掩盖最危险的故障：证据没有进来、证据进来了但没有被使用、工具被调用了但权限不对，以及系统已经失败却仍然声称完成。

本章将这些概念放入同一条可验证的系统链路中。初学者可以先沿着“材料 -> 证据 -> 状态 -> 动作 -> 结果”的故事理解整体；有经验的读者可以进一步检查每个公式的变量、每个协议对象的边界、每次重试的副作用和每个评估指标的分母。论文用于说明方法最初的研究结论，协议和官方文档用于说明接口语义，代码示例只验证可复现的小规模数量关系，生产系统的吞吐、可靠性和安全性仍必须用目标环境实测。

~~~text
一次带知识和工具的任务：

用户目标
  -> 任务状态与约束
  -> 选择上下文或检索来源
  -> 取得并标注证据
  -> 生成答案或工具调用
  -> 权限、参数和副作用检查
  -> 执行并获得观察结果
  -> 更新状态、引用证据、继续或结束
  -> 记录 trace、成本和评估结果
~~~

## 11.1 长上下文：容量、位置与有效使用

### 输入窗口不是记忆

对初学者而言，context window 可以看成一次模型调用允许携带的“工作台”。系统提示、用户问题、历史消息、检索片段、工具返回和模型输出都要在这个工作台中占位置。窗口外的内容不会因为曾经在更早的请求中出现过，就自动成为当前模型可访问的事实；除非应用把它重新摘要、检索或写入模型参数。

设输入 token 数为 `n_in`，预留输出 token 数为 `n_out`，模型和服务共同声明的最大窗口为 `L`，最基本的可行条件是：

~~~math
n_{\mathrm{in}}+n_{\mathrm{out}}\leq L.
~~~

这个不等式只描述接口是否接受请求。它没有说明模型是否能找到位于中间的证据，也没有说明一次请求的延迟和价格是否可以接受。更准确的系统账本还应记录系统提示 `n_sys`、历史 `n_hist`、检索证据 `n_evi`、工具结果 `n_tool` 和用户问题 `n_user`：

~~~math
n_{\mathrm{in}}
=n_{\mathrm{sys}}+n_{\mathrm{hist}}+n_{\mathrm{evi}}+n_{\mathrm{tool}}+n_{\mathrm{user}}+n_{\mathrm{format}}.
~~~

其中 `n_format` 包括 XML、JSON、角色标记和引用标签等模板开销。实际可用的证据预算不是 `L`，而是扣除输出、固定指令和安全策略后的剩余量：

~~~math
B_{\mathrm{evidence}}
=L-n_{\mathrm{sys}}-n_{\mathrm{hist}}-n_{\mathrm{user}}-n_{\mathrm{tool}}-n_{\mathrm{format}}-n_{\mathrm{out}}.
~~~

这个预算是工程估算，不是模型能力定理。当 `B_{\mathrm{evidence}}\leq 0` 时，请求已经没有可用证据空间，系统应缩短历史、减少输出预算、分阶段处理或明确失败，不能把负预算静默截成零。工具结果通常具有突发性，历史摘要也会增长，所以系统应按最坏或高分位输入长度预留空间，而不是只用平均值。

### 注意力代价与 KV cache

标准自注意力需要让序列中的 token 互相比较。若序列长度为 `n`，隐藏维度为 `d`，单层注意力中形成注意力分数的矩阵规模近似为 `n x n`，因此朴素实现的时间和中间显存压力会随 `n^2` 增长。FlashAttention 等实现通过分块和更好的内存访问减少了中间矩阵的读写，但没有把所有长序列任务的总成本变成常数。

自回归生成时，已经处理过的 key 和 value 通常会放入 KV cache。设层数为 `N_layer`，每层每个 token 的 key/value 头维度总和为 `2H_kv`，元素字节数为 `b`，batch 为 `B`，当前序列长度为 `S`，则一个粗略的 KV cache 估算为：

~~~math
M_{\mathrm{KV}}
\approx
B\times S\times N_{\mathrm{layer}}\times 2H_{\mathrm{kv}}\times b.
~~~

`H_kv` 取决于注意力头数、每个头的维度以及是否使用 multi-query attention 或 grouped-query attention；不同实现可能把张量按页、块或其他布局保存。这个公式适合判断“长度翻倍为什么会明显增加显存”，不应当替代目标推理引擎的 profile。

### 位置表示和长度外推

Transformer 的注意力本身并不知道 token 的先后顺序，因此需要位置表示。绝对位置嵌入、相对位置方法、RoPE 和 ALiBi 对距离的表达方式不同。训练时只见过 `L_train` 长度的模型，直接在 `L_test > L_train` 上运行，可能遇到位置分布外推、注意力衰减和训练数据覆盖不足等问题。

位置插值、RoPE scaling、YaRN 或继续长序列训练，能够改变模型在更长位置上的数值范围或训练分布，但它们只解决了“如何表示位置”的一部分问题。模型仍可能没有学会在几十万 token 中寻找分散的约束，也可能在长输入中被重复和噪声淹没。因而，模型卡中写的 context length 应当理解为接口或训练配置的声明，不能直接当作所有任务的有效理解长度。

### Lost in the Middle

长文本研究中反复出现一种现象：证据放在输入开头或结尾时，模型往往比证据放在中间时更容易回答正确。Liu 等人在 *Lost in the Middle: How Language Models Use Long Contexts* 中用多种长上下文任务测量了这种位置敏感性，论文入口见 [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)。这并不意味着所有模型、所有任务都以同样幅度退化，而是说明“窗口足够大”与“证据可以被稳定利用”是两个可分离的实验问题。

可以把一条证据在位置 `p` 的使用概率记为 `u(p)`。一个最简单的有效上下文分数是对任务所需证据集合 `E` 的加权平均：

~~~math
U_{\mathrm{context}}
=\frac{\sum_{e\in E}w_e\,u(e)}{\sum_{e\in E}w_e},
~~~

其中 `w_e` 表示证据对任务的重要程度，要求证据集合非空、权重有限且总和大于零。这个量不是通用 benchmark 的标准定义，而是帮助工程团队区分容量和使用效果的教学模型。实际测量时，可以固定问题和噪声，只改变 needle 的位置，分别记录检索率、答案正确率和引用正确率。

### 一个可运行的位置敏感性实验

下面的代码不调用模型，只构造一个可重复的测试计划。它把证据放到序列的不同位置，并输出相对位置；真正的 `answer()` 应由目标模型或目标系统提供。这样做的价值是先固定实验设计，避免看到某一次回答后临时改变样本。

~~~python
import math
from numbers import Real


def make_needle_context(total_tokens, needle, fraction):
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens < 1
        or not isinstance(needle, str)
        or not needle
        or isinstance(fraction, bool)
        or not isinstance(fraction, Real)
    ):
        raise ValueError("invalid context size or position")
    try:
        fraction = float(fraction)
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid context size or position") from exc
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("invalid context size or position")
    position = min(total_tokens - 1, round((total_tokens - 1) * fraction))
    tokens = ["noise"] * total_tokens
    tokens[position] = needle
    return tokens, position


fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
cases = [make_needle_context(101, "TARGET=42", p) for p in fractions]
assert [case[1] for case in cases] == [0, 25, 50, 75, 100]
for invalid in (
    (0, "TARGET=42", 0.5),
    (101, "TARGET=42", float("nan")),
    (101, "", 0.5),
    (True, "TARGET=42", 0.5),
):
    try:
        make_needle_context(*invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid needle case was accepted")
print([(p, position) for p, (_, position) in zip(fractions, cases)])
~~~

初学者可以把它理解成“同一句话换五个位置”；进阶实验还应加入多条相互干扰的 needle、不同文档格式、不同问题类型、不同输出长度和多次随机采样。最终报告应同时保留请求 token、实际输出 token、位置、答案、引用和延迟，否则无法判断错误来自位置利用还是生成截断。

### 长上下文的适用边界

材料规模中等、要求完整阅读、证据之间需要互相参照时，直接使用长上下文可能比复杂检索更自然。例如审阅一份长度可控的合同，可以让模型同时看到定义、例外条款和附件。材料远大于窗口、知识经常更新、权限过滤复杂或问题只需要少量局部事实时，先检索再构造上下文通常更经济。

长上下文也不能替代结构化数据库。金额求和、库存扣减、权限判断和事务更新应交给程序或数据库完成，模型只负责解释结果或选择查询。窗口越大，越需要保留来源、版本和位置元数据，因为长文本中的错误内容也会获得更多机会影响生成。

## 11.2 上下文工程：预算、压缩与排序

### 上下文是有限资源

把历史消息、检索片段和工具输出直接拼接起来，是最容易写出的原型，也是最容易失控的设计。上下文管理的目标不是最大化 token 数，而是在给定预算下最大化任务相关信息、约束保留率和证据可追溯性。

设候选上下文片段为 `c_i`，每个片段的长度为 `l_i`，相关性为 `r_i`，可信度为 `t_i`，冗余惩罚为 `d_i`，总预算为 `B`。一个简化的选择问题可以写成：

~~~math
\max_{S}\sum_{i\in S}(\alpha r_i+\beta t_i-\gamma d_i)
\quad
\text{s.t.}\quad
\sum_{i\in S}l_i\leq B.
~~~

`alpha`、`beta` 和 `gamma` 是业务权重，不是模型自动给出的真值。法律问答可能提高可信度和版本权重，创意写作可能降低证据约束权重。真实系统还要加入租户权限、文档新鲜度、任务阶段和必须保留的硬约束。

### 历史压缩的三种层次

第一层是删除确定无用的内容，例如已经完成的工具原始输出和重复的系统提示；第二层是保留结构的摘要，例如任务目标、已完成步骤、未解决问题、关键 ID 和失败原因；第三层是把可查询事实外置到状态库，只在需要时重新读取。摘要不是无损压缩，尤其容易丢掉否定条件、例外条款、数字和权限范围。

一份可用的任务摘要至少应包含：原始目标、不可改变的约束、已确认事实、待验证事实、已执行动作、动作结果、失败与重试次数、当前预算和结束条件。把摘要写成自然语言段落并不代表它可恢复；对于金额、时间、ID、版本和权限，优先使用结构化字段。

### 证据排序与分区

将所有内容按相似度排序并塞入 prompt，可能把同一文档的重复片段放在一起，反而丢掉第二个独立来源。常见的上下文组织方式是：先放任务和输出约束，再放按来源分组的证据，最后放需要回答的问题和引用格式。关键证据可以在开头和结尾各保留一次，但必须标注其为同一来源的副本，避免模型把重复次数误认为独立支持。

长文本中还要区分“指令”和“数据”。来自网页、邮件、PDF 或工具的文本即使包含“请忽略之前规则”等句子，也应作为不可信数据，而不是系统指令。这个区分会在后面的工具安全和提示注入章节继续展开。

### 上下文构造的可观测字段

每次请求至少应记录候选片段 ID、选择原因、排序分数、版本、权限过滤结果、最终 token 数和被截断的片段。生产日志不应默认保存未经脱敏的全部敏感内容，但要保留能重建决策的哈希、版本号、范围和必要摘要。没有这些字段，出现“模型没看到证据”时只能重新猜测。

## 11.3 RAG 总体架构：把外部知识变成可追溯证据

### RAG 解决什么问题

Lewis 等人在 2020 年提出 Retrieval-Augmented Generation，将参数化语言模型与外部非参数记忆结合，论文入口见 [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)。工程上，RAG 常用于私有知识、不断更新的制度、产品文档、代码库和需要引用来源的问答。它不能保证答案正确，也不能自动解决来源错误、权限错误和文档版本冲突。

一个典型链路包含离线和在线两部分。

~~~text
离线：采集 -> 解析 -> 清洗 -> 切片 -> 元数据与权限 -> embedding/倒排索引 -> 版本发布
在线：问题 -> 查询改写 -> 权限过滤 -> 召回 -> 重排 -> 上下文构造 -> 生成 -> 引用/拒答 -> trace
~~~

初学者容易把向量数据库看成 RAG 的主体。实际上，向量库只负责索引和候选查询；文档解析、权限、版本、引用、拒答和评估决定了系统能否进入真实工作流。专家系统还需要回答删除传播、索引延迟、租户隔离、缓存失效和事故回滚等问题。

### 文档对象与 chunk 对象

一个文档对象不应只有正文。至少要保留 `document_id`、来源 URI、owner、创建时间、更新时间、版本、语言、数据级别、访问范围和删除状态。一个 chunk 除了 `chunk_id` 与文本，还应保留文档版本、页码或代码行号、父子层级、相邻 chunk 关系和切片算法版本。

切片算法改变后，即使文本没有改变，chunk ID 和召回分布也可能改变，因此切片配置应当像模型配置一样纳入版本。引用最终绑定的是“文档版本 + chunk 范围”，而不是一个无法追溯的字符串。

### 数据治理先于生成质量

过期制度、未标 owner 的 wiki、互相矛盾的产品文档和权限不清的附件，会让 RAG 产生看起来有出处、实际不可信的答案。建立知识库时应明确权威来源优先级、冲突处理规则和失效策略。若某个来源只适合内部草稿，就应在元数据中标记，而不是让模型自己猜测其权威性。

### RAG 与微调的边界

知识更新频繁、需要保留原文和引用时，RAG 通常比把事实写入参数更适合。模型需要学习固定的输出格式、领域表达风格或稳定任务行为时，微调更合适。两者可以组合：微调模型学会如何使用证据，RAG 提供随时间变化的事实。微调不能替代权限过滤，也不能保证模型忠实引用检索内容。

## 11.4 检索：召回、排序与证据质量

### 召回率与排序位置

设问题集合为 `Q`，每个问题的标准证据集合为 `G_q`，检索得到的前 `k` 个结果为 `R_k(q)`。Recall@k 可以写成：

~~~math
\mathrm{Recall@k}
=\frac{1}{|Q|}\sum_{q\in Q}
\frac{|G_q\cap R_k(q)|}{|G_q|}.
~~~

它回答“正确证据有没有进入候选集”，不回答证据是否排在前面。若每个问题只有一个被标注的第一正确结果，MRR 为：

~~~math
\mathrm{MRR}
=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{\mathrm{rank}_q},
~~~

当正确结果没有出现在候选中时，该项记为 0。这里假定评估集 `Q` 非空，且每个 `G_q` 至少有一个标准证据；如果某个问题没有可标注的证据，应单独标记为“不适用”，不能把空集合偷偷当成满分。Recall@k 和 MRR 的分母、标准证据标注和去重规则必须写清楚，否则不同团队的数字不能比较。

### Dense、sparse 和 hybrid retrieval

Dense retrieval 用 embedding 把问题和文档映射到向量空间，适合表达同义和语义相关性；sparse retrieval 依赖词项、倒排索引或 BM25，对产品编号、错误码、人名和精确术语常常更稳。DPR 论文 [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) 展示了密集段落检索的经典路线，但论文数据集上的收益不能直接代表中文企业语料或代码库的结果。

混合检索通常先分别取得 dense 排名和 sparse 排名，再用加权、倒数排名融合或学习排序合并候选。融合前应对分数尺度和去重键做处理，因为两个检索器的原始分数往往不可直接相加。

### 切片不是越细越好

切片过小，定义与例外会被拆散，模型拿到的片段缺乏上下文；切片过大，召回结果包含大量无关内容，占用上下文预算。对文档而言，标题层级、段落和表格边界比固定字符数更有意义；对代码而言，函数、类、调用关系和行号比任意窗口更有意义。

可以同时保存父 chunk 和子 chunk：先用子 chunk 精确召回，再根据需要补入标题、定义和相邻段落。这样做增加了上下文构造逻辑，但比把所有文档都切成相同长度更容易保留结构。

### Reranker 的位置

第一阶段召回器追求覆盖率，通常允许较多候选；reranker 再用 cross-encoder 或其他更贵的模型对问题—片段对进行精排。Cross-encoder 能直接联合读取问题和片段，往往比单独向量相似度更能识别细粒度约束，但它的每对计算成本更高，因此常放在候选集而不是全库上。

检索系统应分别测量召回器输出、reranker 输出和最终上下文，而不是只报告最终答案准确率。若正确片段从未进入第一阶段，继续调 reranker 没有意义；若候选正确但最终被上下文预算删掉，问题发生在另一个阶段。

### 一个可手算的检索融合示例

假设 sparse 排名为 `[A, B, C]`，dense 排名为 `[B, D, A]`，采用倒数排名融合：

~~~math
\mathrm{RRF}(x)=\sum_{l\in\{s,d\}}\frac{1}{k+\mathrm{rank}_l(x)}.
~~~

这里 `k` 是防止头部排名差异过大的常数。若 `k=60`，则 `A` 的分数为 `1/61 + 1/63`，`B` 的分数为 `1/62 + 1/61`，`C` 只有 `1/63`，`D` 只有 `1/62`。`B` 因为在两个列表都出现而领先。融合规则是教学示例，生产系统应使用离线标注集比较不同参数，而不是凭直觉选择。

~~~python
import math
from numbers import Real


def reciprocal_rank_fusion(rankings, constant=60):
    if isinstance(constant, bool) or not isinstance(constant, Real):
        raise ValueError("constant must be positive")
    try:
        constant = float(constant)
    except (TypeError, OverflowError) as exc:
        raise ValueError("constant must be positive") from exc
    if not math.isfinite(constant) or constant <= 0:
        raise ValueError("constant must be positive")
    if isinstance(rankings, (str, bytes)):
        raise TypeError("rankings must be an iterable of rankings")
    scores = {}
    for ranking in rankings:
        if isinstance(ranking, (str, bytes)):
            raise TypeError("each ranking must be a sequence of document ids")
        seen = set()
        for rank, item in enumerate(ranking, start=1):
            if not isinstance(item, str) or not item:
                raise ValueError("document ids must be non-empty strings")
            if item in seen:
                raise ValueError("a ranking cannot contain duplicate document ids")
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "D", "A"]])
assert fused[0][0] == "B"
assert reciprocal_rank_fusion([]) == []
try:
    reciprocal_rank_fusion([["A", "A"]])
except ValueError:
    print("duplicate ranking item rejected")
else:
    raise AssertionError("duplicate ranking item was accepted")
for invalid_constant in (0, float("nan"), True):
    try:
        reciprocal_rank_fusion([], invalid_constant)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid RRF constant was accepted")
print(fused)
~~~

## 11.5 证据状态、Agentic RAG 与答案支持

### 从一次检索到多轮证据收集

简单问题可以采用一次检索；复杂问题往往需要先找到一个实体，再用这个实体查询第二个来源，最后检查时间、版本或反例。Agentic RAG 把检索放进决策循环：系统根据当前证据缺口选择下一条查询，并在证据足够或预算耗尽时停止。

多轮检索的关键不是“调用次数更多”，而是每一轮都要改变证据状态。系统可以保存以下字段：当前问题、已覆盖的 claim、未解决的 gap、相互冲突的来源、查询历史、权限过滤结果、预算和停止原因。没有结构化状态时，Agent 很容易重复搜索相同词语，或者在多轮改写后忘记原始限制。

### 证据图与 claim

将最终答案拆成若干可检查的 claim，比把整段答案当作一个整体更容易评估。对每个 claim 保存支持它的 chunk、来源版本、时间和冲突来源。一个答案可以正确回答主体事实，但错误地补充未被证据支持的时间、范围和因果关系；claim 级记录能够暴露这种“半正确”。

设答案 claim 集合为 `C`，被证据支持的 claim 集合为 `S`，可以定义支持率：

~~~math
\mathrm{SupportRate}=\frac{|S|}{|C|}.
~~~

如果 claim 有不同重要性，使用权重 `w_i`：

~~~math
\mathrm{WeightedSupport}
=\frac{\sum_i w_i\mathbf{1}[c_i\text{ 有支持}]}{\sum_i w_i}.
~~~

这两个指标只衡量支持关系，不等于事实真值。这里要求 `C` 非空；加权版本还要求所有权重有限、非负且总和大于零。错误来源也可能“支持”一个错误 claim，因此还要结合来源权威性、版本和外部事实核验。

### 查询漂移与停止条件

多轮检索常见的 query drift 是：系统从原问题中的一个实体出发，逐渐把探索方向当成新的目标，最后回答了一个相关但不同的问题。控制方法包括保留不可变的原始问题、为每轮查询记录目的、限制主题变化、在生成最终答案前重新比对原始约束。

停止条件可以是所有必需 claim 都有足够支持、检索边际收益低于阈值、预算耗尽、发现权限不足或证据冲突需要人工处理。设第 `t` 轮新覆盖 claim 数为 `g_t`，新增成本为 `c_t`，可以用 `g_t/c_t` 作为粗略边际收益；它不是通用最优策略，但能帮助系统避免为了追求“再搜一点”而无限循环。

### RAG 评估的分层

检索层看 Recall@k、MRR 和候选去重；上下文层看最终证据覆盖、上下文精度和版本正确性；生成层看答案正确、claim 支持、引用准确和拒答；产品层再看延迟、成本、权限泄露和用户任务完成率。只测最后一层，无法定位是检索、拼接还是生成导致失败。

可以把一次 RAG 请求的总成功事件拆为：

~~~math
P(\mathrm{success})
\approx
P(R)\times P(C\mid R)\times P(G\mid C)\times P(A\mid G),
~~~

其中 `R` 表示正确证据被召回，`C` 表示证据正确进入上下文，`G` 表示生成内容忠实，`A` 表示答案满足业务协议。这个乘法不是独立性假设下的严格系统概率，只是说明每一层的损失都会压低最终成功率。

### 拒答是 RAG 的正常输出

没有权限、证据过期、来源冲突或问题超出知识库范围时，系统应澄清、拒答或转人工。拒答准确率可以定义为在“不应回答”的样本中，系统正确拒答或请求补充信息的比例；它必须和强答率一起看。若系统只追求回答覆盖率，可能把不确定性转化成自信的错误。

## 11.6 Grounding、引用与版本一致性

### 从“有引用”到“引用支持”

答案末尾放几个链接不等于答案被引用支持。引用准确性至少包含两个问题：引用指向的来源是否真的包含相关信息，引用范围是否覆盖 claim 的全部限定条件。一个来源支持“功能在版本 1.2 存在”，不一定支持“所有租户都可以使用”，更不一定支持“性能提升 30%”。

对每个引用建立 `claim -> evidence span` 映射。若引用只绑定整个长文档，用户仍然需要自己搜索，系统也难以自动检查。页面、章节、代码行、表格单元格或数据库记录都可以作为 evidence span，具体粒度随来源类型变化。

### 冲突证据

冲突不是简单地取相似度最高的片段。系统需要比较来源权威性、更新时间、适用范围和版本。两个来源分别描述不同时间点时，它们可能并不冲突；一个是草稿、一个是已发布制度时，则应有明确优先级。无法消解时，应把冲突暴露给用户，而不是悄悄拼成一个折中结论。

### 新鲜度和删除传播

设文档版本的生效时间为 `t_v`，请求时间为 `t_q`，业务允许的最大年龄为 `Delta`，可以定义新鲜度约束：

~~~math
t_q-t_v\leq\Delta.
~~~

这只是时间条件，不能替代“版本是否适用于当前产品线”的语义判断。文档更新后，缓存、向量索引、reranker 候选、引用服务和 trace 里的内容都要有传播策略。删除文档只从前端列表中隐藏是不够的；它还应从检索结果、缓存和后续引用中失效。

### 权限过滤的位置

权限过滤应尽量在候选产生前或至少在进入模型上下文前完成，并将租户、用户、资源和操作范围绑定到一次请求。仅在 UI 中隐藏链接仍可能泄露文本摘要，模型上下文中已有的无权内容也无法靠前端撤回。日志和 trace 同样属于数据出口，需要单独设置访问控制与脱敏规则。

## 11.7 Agent 的抽象：状态、动作、观察与完成条件

### Agent 不是一个更长的 prompt

一个 Agent 系统至少有目标 `g`、当前状态 `s_t`、可选动作集合 `A(s_t)`、策略或控制器 `pi`、外部环境 `E` 和停止条件 `F`。在第 `t` 步，系统根据目标和状态选择动作：

~~~math
a_t=\pi(g,s_t),
\qquad
o_{t+1}=E(a_t),
\qquad
s_{t+1}=U(s_t,a_t,o_{t+1}).
~~~

`a_t` 可以是回答、检索、调用 API、修改文件或请求人工确认；`o_{t+1}` 是工具返回、错误、页面变化或用户补充；`U` 是状态更新函数。若系统只保留对话文本而不保存结构化状态，就很难判断某个动作是否已经完成、是否重复执行以及下一步依赖什么结果。

初学者可以把 Agent 看成“会看结果再决定下一步的程序”；进阶视角会进一步区分模型策略、运行时、工具执行器、权限策略和外部环境。模型生成了一个看似合理的工具调用，不代表调用已经发生；工具返回了成功字符串，也不代表业务事务已经提交。

### Agent loop

一个最小循环通常包含：读取目标和状态、构造上下文、调用模型、解析模型输出、校验动作、执行工具、记录观察、更新状态、判断是否完成。运行时应给循环设置最大步数、总 token 预算、时间截止、工具调用预算和人工升级路径。

~~~text
READY -> THINKING -> ACTION_VALIDATION -> EXECUTING -> OBSERVING
  ^                                                        |
  +-------------------- CONTINUE <-------------------------+
  |
  +--> SUCCEEDED
  +--> FAILED / NEEDS_HUMAN / TIMED_OUT
~~~

状态名称可以不同，但每条转移都应有明确的触发条件和持久化记录。尤其不能把“模型输出了完成语句”当作 `SUCCEEDED`；完成状态应由可验证的业务条件、工具结果或人工确认触发。

### Harness 的作用

Harness 是包围模型的运行控制层，负责上下文拼装、工具注册、权限决策、状态持久化、重试、超时、日志和恢复。它决定模型能看到什么、能调用什么、一次任务可以消耗多少资源。一个能力更强的模型，如果被没有预算和恢复语义的 harness 包围，仍可能形成不可靠系统。

Anthropic 的工程文章 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 区分了预定义 workflow 与更开放的 agent loop，并强调应根据任务需要选择复杂度。文章是工程经验而非跨模型定理；它的价值在于提醒我们先判断任务是否真的需要动态决策。

## 11.8 规划、分解与 Workflow-Agent 边界

### 什么时候使用固定 workflow

如果步骤、输入输出和异常分支都能提前写清楚，固定 workflow 往往更容易测试、估算和审计。例如“读取订单 -> 查询库存 -> 计算折扣 -> 返回结果”适合由程序编排，模型只承担意图解析或自然语言解释。

如果任务需要根据新证据动态选择工具、处理未知网页结构、反复修复代码或探索开放问题，Agent loop 才有合理性。但动态性会带来更大的测试空间、成本波动和副作用风险。使用 Agent 不是架构升级的同义词，而是把部分控制权交给不确定的策略。

### 子任务图

把任务分解成子任务节点 `v_i` 和依赖边 `e_{ij}`，可以形成有向图 `G=(V,E)`。若节点 `j` 依赖 `i`，则 `e_{ij}` 表示 `i` 的结果必须先于 `j`。关键路径长度决定最短串行延迟，互不依赖的节点可以并行，但并行会增加资源占用和合并冲突。

一个可用的子任务定义应包含输入、输出 schema、完成条件、失败处理、最大重试、权限范围和证据要求。只写“请完成研究”不是可执行的子任务，因为没有办法判断结果是否覆盖目标，也没有定义什么情况下应该停止。

### 动态重规划的代价

遇到工具错误或新证据时，系统可能需要改变计划。重规划前要保留已完成动作和不可逆副作用，不能像改一段文本一样把过去的外部操作抹掉。对于资金、删除、发布和权限变更，重规划应先进入确认或补偿流程。

## 11.9 工具契约：Schema、选择与参数语义

### Function calling 的本质

工具调用不是模型直接执行函数，而是模型生成一个结构化请求，运行时验证请求并决定是否执行。一个工具定义至少包含名称、用途、参数 schema、返回结构、权限范围、风险级别、超时和版本。工具名称和描述会影响模型选择，但描述不是安全边界；真正的安全边界应在执行器和策略层。

JSON Schema 中的 `type`、`required`、`enum`、`pattern`、`minimum`、`maximum` 和 `additionalProperties` 可以限制数据形状。例如转账工具不能只要求 `amount` 是数字，还应限制币种、收款账户格式、金额范围和幂等键。schema 校验通过只说明形状合法，业务校验还要确认账户归属、余额、额度和用户授权。

### 选择工具与不调用工具

工具路由可以分成四步：先根据任务类型筛掉明显不相关工具，再按租户和权限过滤，再按风险和当前状态过滤，最后把小规模候选集交给模型选择。系统应保留“不调用工具”和“先澄清”的选项，避免模型在没有足够参数时强行调用。

可定义工具选择精度和召回：在需要调用工具的样本中，选中可完成任务的工具比例是工具召回；在所有被调用的工具中，真正必要且合适的比例是工具精度。两者都高并不保证参数正确和副作用安全，因此还要独立测量参数和执行结果。

### 参数来源映射

对每个关键参数保存来源：用户明确给出、可信数据库返回、上一工具结果、模型推断或默认值。模型推断的账户 ID、金额和删除范围不应与用户明确确认的值享有同样信任等级。参数值在修复、重试和跨 Agent 转交时，来源映射也要继续传递。

~~~python
import json


def validate_tool_args(raw, required, allowed):
    if not isinstance(required, (list, tuple, set, frozenset)):
        raise TypeError("required must be a collection of field names")
    if not isinstance(allowed, (list, tuple, set, frozenset)):
        raise TypeError("allowed must be a collection of field names")
    if any(not isinstance(name, str) or not name for name in required):
        raise ValueError("required contains an invalid field name")
    if any(not isinstance(name, str) or not name for name in allowed):
        raise ValueError("allowed contains an invalid field name")
    required_set = set(required)
    allowed_set = set(allowed)
    if not required_set <= allowed_set:
        raise ValueError("required fields must be allowed")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("arguments must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("arguments must be an object")
    if any(not isinstance(name, str) for name in raw):
        raise ValueError("argument names must be strings")
    missing = [name for name in required if name not in raw]
    unknown = sorted(set(raw) - allowed_set)
    if missing or unknown:
        raise ValueError(f"missing={missing}, unknown={unknown}")
    return raw


args = validate_tool_args(
    '{"account_id":"u-7","amount":12}',
    required=("account_id", "amount"),
    allowed=("account_id", "amount", "currency"),
)
assert args["amount"] == 12
try:
    validate_tool_args('{"amount":12,"debug":true}', ("account_id",), ("account_id", "amount"))
except (ValueError, json.JSONDecodeError):
    print("invalid tool arguments rejected")
try:
    validate_tool_args("not-json", (), ())
except ValueError:
    print("malformed JSON rejected")
else:
    raise AssertionError("malformed JSON was accepted")
try:
    validate_tool_args({}, ("account_id",), ("amount",))
except ValueError:
    print("inconsistent schema rejected")
else:
    raise AssertionError("inconsistent schema was accepted")
~~~
代码只是 schema 的最小模拟。生产执行器应使用目标语言的正式 JSON Schema 校验器，并把解析失败、类型错误、业务拒绝和权限拒绝分成不同错误类别。

## 11.10 工具执行：副作用、幂等与未知状态

### 读取和写入不是同一种工具

查询天气、读取文档通常是低副作用操作；发送邮件、修改文件、创建订单、删除资源和转账可能产生不可逆影响。风险级别应影响审批、日志、重试和并发策略。工具名称里写一个 `delete` 并不能自动触发安全机制，风险必须是注册信息和执行策略的一部分。

### 幂等键与重复执行

网络超时并不等于服务端没有执行。若客户端在未知状态下直接重试，可能重复扣款、重复发信或创建多个资源。对支持幂等的写操作，客户端生成稳定的 `idempotency_key`，服务端保存 key 与结果的映射；同一个 key 的重试返回原结果，而不是再次执行。

若服务端不支持幂等，运行时应把未知状态升级给人工或使用查询接口确认，不能仅靠指数退避解决。幂等也有作用域和过期时间，key 的碰撞、跨租户复用和缓存过期都需要处理。

### 重试与退避

对第 `n` 次重试，可以使用带抖动的退避：

~~~math
t_n=\min(t_{\max},t_0\times 2^n)+U(0,j),
~~~

其中 `n` 是从零开始的重试序号，`t_0` 是初始等待，`t_max` 是上限，`j` 是抖动范围。只有明确标记为暂时性错误的失败才应进入重试；参数无效、权限拒绝和业务条件不满足通常不应重试。每次重试都要消耗任务预算，并在 trace 中保留原始错误。下面的示例只生成不带随机抖动的基础退避序列，并把 `budget` 作为显式上限；生产实现再根据并发和截止时间加入抖动。

### 结构化错误

工具错误至少区分：参数校验失败、认证失败、授权失败、资源不存在、限流、暂时性网络错误、服务端内部错误、执行未知和业务拒绝。统一把所有错误压成“调用失败”会诱导 Agent 不断重复同一个动作，也会让用户无法采取补救措施。

~~~python
import math
from numbers import Real


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, OverflowError):
        return False


def retry_delays(attempts, initial, maximum, *, budget):
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise ValueError("attempts must be a non-negative integer")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ValueError("budget must be a non-negative integer")
    if (
        attempts < 0
        or budget < 0
        or attempts > budget
        or not _finite_number(initial)
        or not _finite_number(maximum)
        or initial < 0
        or maximum < initial
    ):
        raise ValueError("invalid retry policy")
    delays = []
    delay = initial
    for _ in range(attempts):
        delays.append(min(maximum, delay))
        delay = min(maximum, delay * 2)
    return delays


assert retry_delays(5, 1, 4, budget=5) == [1, 2, 4, 4, 4]
assert retry_delays(0, 0, 0, budget=0) == []
try:
    retry_delays(6, 1, 4, budget=5)
except ValueError:
    print("retry budget enforced")
else:
    raise AssertionError("retry budget was ignored")
try:
    retry_delays(1, float("nan"), 4, budget=1)
except ValueError:
    print("non-finite retry delay rejected")
else:
    raise AssertionError("non-finite retry delay was accepted")
print(retry_delays(5, 1, 4, budget=5))
~~~

### 结果投影与不可信输出

工具返回的 HTML、邮件正文、数据库字段和第三方 API 文本都可能包含提示注入或敏感数据。执行器应把原始结果、结构化字段、展示摘要和允许进入模型的字段分开。结果投影不是简单截断；需要保留状态码、版本、分页、来源和错误语义，同时阻止不必要的密钥、内部路径和跨租户数据进入上下文。

## 11.11 MCP：工具、资源和提示的协议边界

### MCP 解决的连接问题

Model Context Protocol（MCP）试图标准化 AI 应用与外部服务器之间的上下文和能力连接。官方规范 [2025-06-18 版本](https://modelcontextprotocol.io/specification/2025-06-18) 将 Host、Client、Server 以及初始化、能力协商、工具、资源、Prompt 等对象放在明确的协议边界内。规范页面会随版本演进，工程实现必须绑定具体版本，不能只写“MCP 兼容”。

初学者可以把 MCP 看成“让不同应用用统一方式发现和调用外部能力”的协议；它不是 Agent 自己的规划算法，也不是权限系统的替代品。Host 仍然拥有用户交互和整体策略，Client 负责与某个 Server 建立连接，Server 暴露能力并执行自身逻辑。不同实现可能把多个 Server 接入同一 Host，但每条连接的能力和信任边界都应单独管理。

### Tools、Resources 和 Prompts

Tool 表示可被调用的动作，通常有名称、描述、输入 schema 和结果；Resource 表示可读取的上下文资源，使用 URI 和元数据定位；Prompt 表示可发现、可参数化的提示模板。三者的语义不能互换：读取一个资源不应隐含修改动作，把一个有副作用的操作包装成普通资源会误导调用方。

MCP 解决了“能力如何发现、消息如何表达和连接如何协商”的一部分问题，但 Host 仍需执行用户授权、租户隔离、输出投影、工具预算和危险动作确认。Server 的描述字段也不能被当作可信指令，来自外部内容的文本仍然要经过数据与指令分离。

### 生命周期与能力协商

连接初始化通常需要协商协议版本和双方能力。版本协商成功不代表每一个扩展都可用；调用方应根据声明的 capability 决定是否发送对应消息，并对未知字段保持兼容。生命周期事件、连接关闭、取消和错误都应进入 trace，否则断线后的半完成任务无法恢复。

### MCP 的工程评估

评估 MCP 集成时，不能只测试“工具能否调用”。还要检查 Server 身份和版本、工具 schema 稳定性、Resource URI 访问范围、Prompt 参数校验、Roots 或沙箱边界、结果大小、连接超时、取消语义和跨 Server 数据流。协议层面通过不等于业务层面安全，业务系统必须在自己的执行器中再次校验权限和副作用。

## 11.12 MCP 资源、提示与安全生命周期

### Resource URI 不是权限票据

一个 URI 能定位资源，不代表请求者拥有读取权限。Resource 访问需要把 URI、租户、用户、会话、用途和字段投影绑定在一起。路径规范化、符号链接、远程 URI、资源模板和订阅更新都可能扩大访问范围。服务端不能只检查字符串前缀来实现目录隔离。

### Prompt 能力的边界

可发现的 Prompt 模板有助于复用领域工作流，但模板本身可能包含敏感策略、过时约束或未经验证的工具指令。调用方应校验参数、版本、用户可见范围和最终渲染结果；模板中的外部文本仍按不可信数据处理。Prompt 更新也应与评估集和回滚版本绑定。

### 服务器供应链

接入第三方 MCP Server 等同于接入一段能够读取数据或调用外部服务的软件。需要记录发布者、版本、依赖、权限范围、网络出口、文件根目录、密钥来源和变更历史。Server 描述中声称“只读”不能替代代码和运行时验证；只读数据库连接也可能暴露敏感行。

### 连接级隔离

不同 Server 的结果不能默认互相信任。系统可以给结果打上来源、租户、敏感等级和可信度标签，并在跨 Server 传递时重新检查权限。一个搜索 Server 返回的网页内容可能包含恶意指令，不能直接变成另一个邮件 Server 的发送参数。

## 11.13 A2A：跨 Agent 的任务与产物协作

### A2A 解决的对象不同

Agent2Agent（A2A）协议关注 Agent 之间发现、委派任务、交换消息和产物。当前规范入口为 [a2a-protocol.org/latest/specification](https://a2a-protocol.org/latest/specification/)。A2A 的具体版本和字段可能继续演进，生产系统要固定版本并记录协议适配层。

MCP 更接近“一个运行时如何连接外部工具和资源”，A2A 更接近“一个 Agent 如何把任务交给另一个 Agent”。前者的核心对象是 Server capability 和工具/资源，后者的核心对象是 Agent Card、Task、Message、Part、Artifact 和任务状态。一个系统可以用 A2A 委派任务，再由被委派 Agent 通过 MCP 调用工具；两种协议不是互相替代。

### Agent Card 与发现

Agent Card 描述 Agent 的身份、能力、接口、输入输出类型和认证信息。发现到一个 Card 只说明对方宣称具备某项能力，不说明当前租户已授权，也不说明对方真的能完成具体任务。Card 应有版本、过期和缓存策略；路由决策应记录选择依据和当时看到的 Card 版本。

### Task、Message 与 Artifact

Task 表示一个可持续跟踪的委派工作，通常拥有状态、上下文和生命周期；Message 携带用户或 Agent 之间的交流；Part 表示文本、结构化数据或其他内容单元；Artifact 表示任务产出的文件、报告或结构化结果。把完整上下文无选择地复制到下游 Agent，会扩大隐私和注入传播范围。更稳妥的做法是传递最小必要上下文、来源和约束。

### 任务状态与幂等

跨 Agent 请求同样可能超时和重复。委派任务需要唯一 ID、调用方身份、截止时间、取消语义、重试策略和最终状态。`submitted`、`working`、`completed`、`failed`、`canceled` 等状态的含义必须和产物是否已经生成区分开。下游报告 `completed` 时，上游仍应校验产物 schema、权限和证据，而不是只相信状态字符串。

### 跨 Agent 的权限衰减

下游 Agent 不应因为上游拥有更高权限就自动继承全部权限。委派时应明确 audience、scope、资源范围和有效期，必要时让权限只能减少不能扩大。跨租户、跨系统和跨数据等级传递时，身份链必须能在 trace 中复原。

## 11.14 Memory：从对话历史到可治理状态

### 记忆的四种用途

短期记忆保存当前任务的消息和工具观察；情景记忆保存过去某次任务的事件；语义记忆保存相对稳定的事实；偏好记忆保存用户明确表达并允许长期使用的偏好。它们的保留期限、访问范围和纠错方式不同，不能把所有历史都放进一个向量库。

### 记忆读写循环

记忆系统通常包含候选生成、相关性排序、权限过滤、冲突检查、上下文投影和写入审核。查询相关性可以用向量相似度、时间衰减和任务关系共同决定：

~~~math
S(m,q,t)=\alpha\,\mathrm{sim}(m,q)+\beta\,\mathrm{recency}(m,t)+\gamma\,\mathrm{tasklink}(m,q)-\delta\,\mathrm{risk}(m).
~~~

`m` 是记忆，`q` 是当前查询，`t` 是当前时间；各项分数需要在同一评测集上校准。高相关不代表可以使用，权限和敏感等级应当是硬过滤条件，而不是仅仅扣一点分。

### 写入比读取更危险

模型从用户话语中推断“用户喜欢某品牌”并自动长期保存，可能把一次性的上下文变成错误画像。写入策略应区分用户明确要求保存、系统观察到的短期事实和模型猜测；对敏感属性、凭据、医疗与财务信息通常禁止默认长期写入。用户应能够查看、修正和删除可持久化记忆。

### 冲突、过期与污染

新旧记忆冲突时，需要保存来源、时间和置信度，不能只按向量相似度选一条。过期策略可以按 TTL、事件失效或新事实覆盖；删除要传播到索引、缓存、摘要和备份。外部网页或工具输出中的指令不能因为被写进记忆就升级为可信策略，否则一次提示注入可能长期污染所有会话。

## 11.15 Code Agent：文件、补丁与命令执行

### 代码 Agent 的真实任务

Code Agent 不只是生成代码，还要定位文件、理解依赖、编辑内容、运行测试、阅读错误、修复并确认差异。每个动作都改变工作区或执行环境，因而需要保存当前分支、用户已有改动、补丁范围、测试命令和结果。

### 文件路径约束

文件工具应把工作区根目录规范化，并检查目标路径是否仍然位于根目录内。仅用字符串前缀判断有漏洞，例如 `/workspace/project2` 也可能以 `/workspace/project` 开头；符号链接和 `..` 更会绕过简单检查。

~~~python
from pathlib import Path


def contained_path(root, requested):
    root_path = Path(root).resolve()
    target = (root_path / requested).resolve()
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    return target


safe = contained_path("/tmp/project", "src/main.py")
assert str(safe).endswith("/tmp/project/src/main.py")
try:
    contained_path("/tmp/project", "../secrets.txt")
except ValueError:
    print("path traversal rejected")
~~~

这段代码只演示路径语义，生产系统还要处理根目录不存在、符号链接创建竞态、挂载点、Windows 路径、文件权限和并发编辑。读取工具也应有大小上限、分页和截断标记，避免把整个二进制文件或秘密配置送入上下文。

### Patch 比整文件重写更可审计

补丁包含目标文件、上下文、增删行和预期基线，便于审查、应用和回滚。应用前应检查上下文是否匹配，应用后应重新读取关键区域并运行格式化和测试。用户在 Agent 工作期间产生的无关改动不能被覆盖；如果基线发生变化，应暂停应用并重新定位，而不是强行合并。

### 命令沙箱

命令执行器要区分只读分析、编译测试、网络访问、写文件、进程控制和破坏性操作。allowlist/denylist 只是入口，真正的隔离还需要独立用户、工作目录、资源限制、网络策略、超时取消、秘密脱敏和可回收环境。命令输出中的文字同样是不可信数据，不能直接改变 Agent 的权限或系统提示。

### 代码任务的完成条件

“代码生成成功”不是任务完成。完成条件可以包括：补丁只触及允许文件、静态检查通过、目标测试通过、没有新增未解释的失败、差异符合用户目标、运行结果和环境版本已记录。测试通过也不代表没有安全问题，尤其是测试只覆盖公开路径时。

## 11.16 Browser 与 Computer Use Agent：状态不是截图

### 网页动作的状态空间

浏览器 Agent 面对的是会变化的页面、登录会话、异步加载、弹窗、分页和外部副作用。截图是观察的一种形式，但不能完整表达 DOM、可访问性树、当前 URL、表单值、网络请求和焦点状态。点击前应确认元素身份和当前页面状态，而不是只依靠上一次截图中的坐标。

### 观察—动作闭环

每个动作应记录目标元素、定位依据、前置状态、执行时间、结果变化和失败原因。页面改变后，旧的元素引用可能失效；重复提交表单前应检查是否已经产生订单或发送消息。对于支付、删除、发布和发送等动作，要把预览和用户确认放在不可逆步骤之前。

### 网页内容中的提示注入

网页文本、邮件正文和文档内容可能直接告诉 Agent“点击发送”或“把 cookie 发给某地址”。这些内容属于外部数据，不拥有系统指令等级。浏览器工具应阻止页面文本直接触发高风险动作，敏感数据也不应因为页面请求就自动填入。

### 计算机使用的评估

评估不仅看最终页面是否正确，还要看动作轨迹是否安全、是否访问了不必要的站点、是否在失败后重复提交、是否能识别登录和权限边界。可以用任务成功率、误操作率、危险动作拦截率和人工接管率分层记录。

## 11.17 Multi-Agent：并行协作与冲突合并

### 为什么拆成多个 Agent

多 Agent 可以把不同领域、不同工具权限或不同工作阶段隔离开，也可以并行处理互不依赖的子任务。例如一个 Agent 提取证据，另一个 Agent 检查数值，第三个 Agent 负责格式化报告。拆分并不自动提高质量，通信、合并和重复推理会增加成本。

### 协作拓扑

常见拓扑包括中心协调器、层级委派、流水线和共享黑板。中心协调器便于控制，但可能成为瓶颈；完全对等的网络更灵活，却更难限制循环和权限。选择拓扑时要看任务依赖、并发程度、结果合并复杂度和失败隔离要求。

若有 `N` 个 Agent，平均每个 Agent 产生 `m` 条需要互相交换的消息，完全互联的潜在连接数可达 `O(N^2)`；中心协调器通常把连接控制在 `O(N)`，但协调器的上下文和吞吐成为瓶颈。这是复杂度直觉，不代表实际消息量一定达到上界。

### 结果合并

合并器不能只把多个自然语言回答拼在一起。每个子结果应包含任务 ID、来源、证据、版本、置信度、未完成项和错误。冲突时可按来源权威性、独立性、时间和验证结果处理；无法判定时保留冲突并升级，而不是用多数票掩盖共同错误。

### 循环、重复与预算

跨 Agent 委派必须带有 trace ID、父任务 ID、剩余预算和最大深度。收到相同任务时要检查幂等键和当前状态。共享记忆会让一个错误结果快速传播，因此跨 Agent 写入应有来源标记和审核状态，不能把下游推测直接写成全局事实。

## 11.18 RAG、工具、Agent 与 Memory 的组合边界

### 四类对象的职责

RAG 提供带来源的外部证据，Tool 提供可执行的输入输出契约，Agent 负责根据目标和状态选择下一步，Memory 保存跨步骤或跨会话的部分状态。四者组合后，任何一类错误都可能被另一类放大：错误检索会诱导错误工具参数，工具结果写入记忆会造成长期污染，过长记忆又会让 Agent 忽略当前证据。

### 优先级与数据流

一次请求可以按以下顺序组织：不可变系统约束 -> 用户目标与授权 -> 当前任务状态 -> 经过权限过滤的证据 -> 工具 schema -> 外部观察 -> 可选记忆。这个顺序不是所有应用的固定模板，但它体现了约束、目标、证据和观察的职责差异。任何来自文档、网页、工具和记忆的自然语言都不应自动获得系统指令权限。

### 数据流标签

对跨层对象保存来源、租户、敏感等级、版本、可信状态和用途。数据从 RAG 进入 Agent，再进入 Tool 参数时，标签不能丢失；工具结果进入 Memory 时，应该判断是否允许持久化。可用 taint propagation 的方式表示：若输入集合中含有敏感或不可信标签，输出默认继承该标签，只有经过明确校验才可降级。

### 组合系统的故障定位

出现错误答案时，先问五个问题：正确事实是否存在，是否被检索，是否进入上下文，模型是否据此生成，是否经过工具或记忆改变。出现错误动作时，再追踪工具选择、参数来源、权限决定、执行状态和重试链。每个环节都有独立 trace span，才能避免把所有问题归因于“模型幻觉”。

## 11.19 Trace、Replay 与可观测性

### Trace 的最小结构

OpenTelemetry 的 [observability primer](https://opentelemetry.io/docs/concepts/observability-primer/) 将 trace、metric 和 log 放在统一的可观测性语境中。Agent 系统可以把一次任务作为 root span，再把上下文构造、模型调用、检索、重排、工具校验、工具执行、人工确认和最终输出作为子 span。

每个 span 至少需要开始和结束时间、状态、输入输出摘要、版本、调用方、租户和父子关系。原始 prompt 和工具参数可能含敏感信息，应采用字段级脱敏、哈希或受控存储；脱敏不能让 trace 失去定位所需的 ID 和版本。

### Replay 的含义

回放不是重新调用所有真实工具。安全回放通常使用原始模型响应和工具观察的录制件，或使用确定性模拟器重建状态转移，从而检查新版本的解析、权限和合并逻辑。需要真实副作用时，必须使用隔离账户、幂等键和人工确认。

### 失败归因

一个 Agent incident 可以有多个根因：上下文截断、检索漏召回、schema 解析失败、权限策略错误、工具未知状态、模型错误规划和评估集遗漏。trace 应保存每次状态变化和决策依据，而不只是最终文本。最终状态也必须和工具事实一致，不能在工具失败后写成成功。

### 成本和延迟归因

按任务记录输入输出 token、模型、检索、重排、工具、重试、人工和存储成本。延迟要拆成排队、首 token、生成、检索、工具等待和重试。只看端到端平均值会掩盖 Agent 长尾，至少还要看 P50、P95 和超时率。

## 11.20 评估：从答案分数到任务可靠性

### 评估对象分层

模型评估关注生成能力，检索评估关注证据召回，工具评估关注选择与参数，Agent 评估关注状态转移和任务完成，产品评估再加入权限、成本、延迟和用户体验。一个总分不能替代这些分层指标，因为某一层的高分可能掩盖另一层的严重失败。

HELM（[Stanford CRFM HELM](https://crfm.stanford.edu/helm/latest/)）强调多维度、可复现的语言模型评估；SWE-bench 论文 [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) 用真实软件仓库问题评估代码 Agent 的补丁能力。公开 benchmark 适合比较方法和建立共同语言，但目标业务的权限、数据版本和工具协议仍需自建评测集。

### Tool-use 指标

工具调用可以拆成：

~~~math
\mathrm{ToolCallRecall}
=\frac{\text{需要调用且正确调用的样本}}{\text{需要调用的样本}},
~~~

~~~math
\mathrm{ToolCallPrecision}
=\frac{\text{必要且正确的调用}}{\text{所有发生的调用}}.
~~~

参数正确率、schema 通过率、授权阻断率、执行成功率和观察使用正确率应分别统计。Recall 的分母必须是非空的“确实需要工具”的样本集；Precision 的分母必须是非空的实际调用集。没有适用样本时应报告“不适用”，不能用零或默认值伪造比较。工具调用次数多不代表 Agent 更强，可能只是重复调用或没有理解结果。

### 任务可靠性

定义任务成功需要同时满足业务结果、约束、安全和证据条件。若 `Y` 是业务结果正确，`C` 是约束满足，`S` 是安全，`P` 是证据或协议正确，则严格成功可以写成：

~~~math
\mathrm{Success}=Y\land C\land S\land P.
~~~

可以报告成功率、错误完成率、人工接管率、重复副作用率、预算超限率和恢复率。错误完成（系统声称完成但业务条件不满足）通常比显式失败更危险，应单独计数。

### Pass@k 与重复采样

代码和可验证任务有时会产生 `k` 个候选。若一次候选成功概率为 `p`，且假设候选独立，至少一个成功的理论概率为：

~~~math
\mathrm{Pass@k}=1-(1-p)^k.
~~~

这里要求 `0<=p<=1` 且 `k` 是非负整数；`k=0` 时结果为 0。真实候选往往相关，独立假设会高估收益；采样还增加 token、延迟和验证成本。因此报告 Pass@k 时要同时报告候选生成预算、验证器、去重规则和单位成功成本。

### 评审模型的局限

LLM-as-a-Judge 可以降低人工成本，但会受到长度、格式、模型偏好和提示设计影响。高风险任务应抽样进行人工复核，并测量评审模型与人工的一致性。自动 judge 的分数适合做回归信号，不应在没有校准的情况下当作事实真值。

## 11.21 提示注入与 Agent 安全

### 注入的核心问题

提示注入不是某个特殊字符串，而是把低信任数据伪装成高优先级指令，影响模型或运行时决策。它可能出现在网页、PDF、代码注释、邮件、工具返回、检索 chunk、记忆和下游 Agent 消息中。模型越能使用外部数据和工具，攻击面越大。

### 指令与数据分离

系统应在数据结构上区分 system policy、user instruction、retrieved evidence、tool observation 和 memory，而不是只靠自然语言说“不要相信下面内容”。模型上下文中可以加入来源标签和边界，但真正的高风险动作仍需运行时策略拦截。工具参数要从结构化字段生成，不能让页面文本直接写入 shell 命令或邮件收件人。

### 高风险动作的多层保护

对删除、支付、外发、权限变更和发布等动作，可以组合使用：最小权限、目标和参数白名单、敏感字段脱敏、独立执行器、用户确认、双人审批、网络隔离、审计和可补偿事务。任何单层保护都可能被绕过。NIST AI RMF 的[官方框架入口](https://www.nist.gov/itl/ai-risk-management-framework)适合用来组织风险识别、测量和治理，但它不是某个 Agent 的具体安全配置。

### SSRF、路径和数据外发

浏览器或 HTTP 工具要限制目标域名、解析后的 IP、内网地址、重定向和响应大小；文件工具要做路径包含检查和根目录隔离；数据库工具要限制语句类型、表和行范围；外发工具要检查收件人、附件和敏感标签。安全日志要记录“为什么允许”或“为什么阻断”，否则只能看到失败结果，看不到策略是否工作。

### 安全评估

安全集应覆盖直接注入、间接注入、跨轮污染、跨 Agent 传播、工具结果注入、越权检索、敏感记忆写入和高风险动作绕过。指标包括注入成功率、未授权动作率、敏感数据外泄率、危险动作拦截率和人工确认覆盖率。攻击样本通过一次并不证明系统安全，版本更新后还要做回归。

## 11.22 成本、延迟、并发与可靠性

### 单位任务成本

Agent 的成本不仅是一次模型调用。设第 `i` 个执行阶段的模型成本为 `c_i`，检索和重排成本为 `r_i`，工具成本为 `t_i`，人工成本为 `h_i`，则一次任务成本可以写成：

~~~math
C_{\mathrm{task}}
=\sum_i(c_i+r_i+t_i+h_i).
~~~

要按成功任务而不是请求数量比较成本：

~~~math
C_{\mathrm{success}}=\frac{C_{\mathrm{total}}}{N_{\mathrm{successful\ tasks}}}.
~~~

这里要求成功任务数大于零；没有成功任务时应报告“暂无单位成功成本”，而不是输出无穷大或 NaN。若失败任务仍消耗大量重试和人工，单纯看每请求价格会得出错误结论。成本记录还要绑定模型版本、工具版本、租户和任务类型。

### 延迟分解

端到端延迟可近似拆成排队、输入处理、检索、模型生成、工具等待、重试和后处理：

~~~math
T_{\mathrm{e2e}}
=T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{retrieval}}+T_{\mathrm{decode}}+T_{\mathrm{tool}}+T_{\mathrm{retry}}+T_{\mathrm{post}}.
~~~

长上下文主要增加 prefill 和输入传输压力，多轮 Agent 主要增加调用次数和等待链。并行化可以降低关键路径，但会增加并发资源、合并和限流复杂度。

### 预算和截止时间

每个任务应携带剩余 token、金钱、工具调用次数、最大深度和截止时间。下游委派只能消耗父任务剩余预算的一部分，不能重新获得完整预算。接近截止时间时，系统应选择摘要、部分结果或人工升级，并诚实说明未完成项，而不是继续生成一个貌似完整的答案。

### 缓存与一致性

可以缓存 embedding、检索结果、模型前缀和只读工具结果，但缓存键要包含租户、权限、文档版本、模型/提示版本和相关参数。权限或文档更新后，旧缓存若仍可被命中，就会产生数据泄露或陈旧答案。缓存命中率提高不代表系统更好，必须同时检查命中内容的版本和权限正确性。

### 降级策略

模型不可用时可以切换到较小模型、固定 workflow、关键词搜索或人工处理；工具不可用时可以返回查询链接或保存待办。降级路径也要经过安全和权限测试，不能因为“只是备用逻辑”就跳过校验。用户可见的状态应区分完整成功、部分成功、等待外部系统和明确失败。

## 11.23 一个可审计的端到端设计

设想一个企业技术支持系统：用户询问某版本产品的故障，并要求查询工单、读取内部文档、必要时创建诊断任务。一个可审计的设计可以这样组织。

第一步，身份服务把用户、租户、产品范围和数据级别绑定到请求。第二步，RAG 根据产品版本和权限召回文档，并保留文档版本、段落和更新时间。第三步，Agent 将问题拆成“解释现象”和“是否需要查询工单”两个子任务；前者要求引用，后者需要工具权限。

第四步，工具路由只暴露当前租户可用的工单查询工具，schema 要求产品 ID、时间范围和分页，缺失关键字段时先澄清。第五步，执行器将工具结果投影成工单状态和允许展示的字段，屏蔽内部备注和其他租户数据。第六步，Agent 将文档证据和工单观察分别标记来源，检查是否存在版本冲突，再生成带引用的诊断说明。

如果用户要求自动重启服务，系统要把这个动作从“查询”切换到高风险写操作：重新验证权限、目标实例、变更窗口和幂等键，必要时请求人工确认。重启返回超时后，系统不能直接再发一次；应先查询任务状态，确认未知状态，再决定重试或转人工。

这条流程的 trace 至少能回答：用户看到了什么版本的文档，检索为什么选中这些片段，模型提出了什么工具调用，谁批准了动作，工具是否真正执行，最终答案的每个 claim 来自哪里。这样的记录比一段完整的最终文本更有诊断价值。

## 11.24 资料、证据与持续验证

本章使用的主要资料按证据类型分层：

- RAG 的总体方法见 Lewis 等人的 [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)，密集检索的经典路线见 [DPR](https://arxiv.org/abs/2004.04906)。论文说明研究方法和实验范围，不保证目标企业语料上的效果。
- 长上下文位置利用见 [Lost in the Middle](https://arxiv.org/abs/2307.03172)。它支持进行位置敏感性测试，不应被解读为所有模型都有同一条退化曲线。
- Agent 循环和工具使用的研究入口包括 [ReAct](https://arxiv.org/abs/2210.03629) 与 [Toolformer](https://arxiv.org/abs/2302.04761)。研究中的工具集合、训练方式和任务分布与生产系统不同。
- Agent 工程组织方式可参考 Anthropic 的 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)；它是公开工程经验，适合帮助选择 workflow 与 agent 的边界。
- 工具协议的对象和生命周期以 [MCP 2025-06-18 规范](https://modelcontextprotocol.io/specification/2025-06-18)为准；跨 Agent 协作以 [A2A 规范](https://a2a-protocol.org/latest/specification/)为准，并在实现中固定版本。
- 代码 Agent 的公开评测可参考 [SWE-bench 论文](https://arxiv.org/abs/2310.06770)，评测系统与目标代码库、测试质量和补丁策略之间仍有差异。
- 评测维度可参考 [HELM](https://crfm.stanford.edu/helm/latest/)，可观测性术语可参考 [OpenTelemetry observability primer](https://opentelemetry.io/docs/concepts/observability-primer/)，风险治理可参考 [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)。这些资料提供框架和术语，不替代具体系统的安全审查。

资料的可信度还取决于用途：协议字段应查规范版本，模型行为应查模型卡和目标版本实测，论文 benchmark 应查原始实验设置，安全保证应查攻击样本和回归记录。网页摘要、营销文案和模型自己的解释可以作为线索，不能单独作为机制或性能结论。

## 11.25 长上下文、RAG 与 Agent 的统一理解

长上下文回答“当前一次推理最多能携带多少材料”，RAG 回答“从外部知识中选择哪些材料”，Agent 回答“如何围绕目标反复读取状态、采取动作并处理结果”。它们的组合关系可以用一条数据流表示：

~~~text
任务目标与约束
  -> 上下文预算分配
  -> RAG/工具取得带来源的观察
  -> Agent 更新状态与选择下一步
  -> 工具执行器验证权限和副作用
  -> 结果投影回上下文或持久化记忆
  -> 证据、状态、成本和安全事件进入 trace
~~~

当材料固定且需要整体互相参照时，长上下文可能最简单；当知识库巨大、更新频繁或权限复杂时，RAG 更适合先筛选；当任务需要多步外部行动时，Agent 才有必要。现实系统经常三者并用，但每增加一层，就增加新的状态、故障和评估维度。

真正可靠的系统不是回答得最像人，而是能说明自己看到了哪些证据、做了哪些动作、哪些结果已经确认、哪些仍然未知，并在权限、预算、时间和安全边界内停止。下一章将从 reasoning 与评估继续讨论模型如何生成、验证和分配测试时计算。
