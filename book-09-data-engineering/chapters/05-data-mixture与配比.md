# 第五章：Data Mixture 与配比

前几章讨论了数据从哪里来、如何清洗、如何去重、如何避免污染。本章进入一个更隐蔽但更关键的问题：数据配比，也就是 data mixture。

所谓 data mixture，是指训练集中不同数据来源、语言、领域、模态和任务类型之间的比例安排。它回答的问题不是“有没有这些数据”，而是“这些数据各占多少、以什么顺序出现、用什么采样权重参与训练”。

很多人第一次接触大模型训练时，会以为数据配比只是工程配置：网页多少、书籍多少、代码多少、中文多少、英文多少，随便设一个比例就可以。实际上，data mixture 是训练目标的一部分。模型最终学到什么能力、偏向什么语言、擅长什么领域、输出什么风格，都会受到配比影响。

本章重点：通用文本、代码、数学、多语言、专业领域数据的混合策略和实验方法。

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 Chinchilla、T5 / mT5、Gopher / MassiveText、RefinedWeb、FineWeb、Dolma、DataComp-LM / DCLM、Llama 系列公开报告和 phi-1 / textbook-quality data 相关资料。

本讲聚焦 data mixture 作为训练目标设计的一部分：数据池标签、token 配比、温度采样、质量加权、目标能力权重、风险约束、effective epoch、配比 ablation 和版本审计。

```text
数据池 -> 清洗去重后 token 数 -> 质量 / 风险 / 能力标签 -> 采样权重 -> 训练预算分配 -> 评估矩阵 -> ablation -> 版本化
```

本讲不把任何公开模型的数据比例写成通用最佳答案。真实项目的 mixture 要由模型目标、token budget、数据质量、tokenizer、许可证、安全约束和评估结果共同决定。

---

## 1. 先建立直觉：配比决定模型“读过什么世界”

可以把预训练想象成给一个学生安排阅读书单。你不能只问“读了多少页”，还要问读了什么。

如果 90% 是网络论坛，学生会很熟悉口语和情绪表达，但可能缺少严谨知识。如果 90% 是论文，学生会更正式、更专业，但可能不懂普通用户怎么提问。如果 50% 是代码，学生会很会写程序，但普通对话和文学表达可能被压缩。如果英语占绝对多数，其他语言能力会被牺牲。

大模型也是一样。模型不是抽象地学习“智能”，它是在训练分布中学习统计规律。data mixture 决定了训练分布的形状。

因此，配比不是训练前随便拼数据，而是能力设计。

一个通用助手模型、代码模型、数学模型、医疗模型、金融模型、多语言模型、儿童教育模型，它们的理想 data mixture 都不一样。

---

## 2. 来龙去脉：从“越多越好”到“配比优化”

早期语言模型规模较小，数据主要来自新闻、百科、书籍或任务数据集。那时数据配比的重要性已经存在，但没有今天这么突出，因为模型容量和训练数据规模有限。

GPT-3 之后，web-scale 数据成为大模型训练主流。论文强调大规模无监督预训练带来 few-shot 能力，但也暴露出大型网页语料的质量、污染、偏见和来源分布问题。模型能力不只由参数量决定，也由训练语料决定。

Chinchilla 进一步提醒大家：在固定 compute budget 下，参数量和训练 token 数需要共同扩展。它把行业注意力从“模型越大越好”拉回到“同样算力下，模型大小和数据 token 数如何平衡”。但 Chinchilla 讨论的是 token 数和模型规模的宏观关系，真正落地时还要问：这些 token 来自哪里？什么质量？什么语言？什么领域？

phi-1 则从另一个角度强调高质量数据的重要性。它使用 textbook-quality web 数据和合成教材/练习数据，在较小模型规模下取得较强代码能力。这说明在某些目标上，数据质量和配比可以显著改变训练效率。

所以大模型数据工程的发展路线大致是：

1. 先追求规模：尽可能收集更多网页、书籍和代码。
2. 再追求质量：清洗、过滤、去重、降低污染。
3. 再追求配比：让不同数据类型按目标能力合理组合。
4. 最后追求闭环：用训练实验和评估反馈反向优化 data mixture。

---

## 3. Data mixture 到底在配什么？

data mixture 不只是“网页占多少”。它至少包含以下维度：

1. 来源维度：Common Crawl、书籍、论文、百科、代码仓库、论坛、问答、文档、课程、合成数据。
2. 语言维度：英语、中文、日语、法语、西班牙语、低资源语言、混合语言。
3. 领域维度：通用知识、科学、工程、法律、医学、金融、教育、娱乐、生活服务。
4. 能力维度：事实知识、推理、代码、数学、写作、翻译、对话、工具使用、安全拒答。
5. 质量维度：高质量教材、普通网页、低质量网页、人工标注、合成数据。
6. 风格维度：正式、口语、学术、新闻、社交媒体、教程、文档、对话。
7. 阶段维度：预训练、继续预训练、SFT、偏好训练、安全训练、领域适配。

同一份数据在不同维度上可能有不同标签。例如一份 Python 教程同时属于代码、英文、教育、技术文档、高质量结构化文本。配比系统最好支持多标签，而不是把数据硬塞进一个单一类别。

---

## 4. 为什么配比会影响能力？

模型训练本质上是在最小化训练分布上的预测误差。某类数据出现得越多，模型越有机会优化这类模式。

如果代码 token 占比提高，模型会更多看到缩进、函数、变量、API、注释、测试和错误信息，因此代码生成能力通常会增强。但如果代码占比过高，模型在普通自然语言上的训练机会相对减少，可能影响聊天、写作和开放问答。

如果数学数据占比提高，模型更常见到题目、证明、公式、推理步骤和答案格式，数学能力可能增强。但数学数据质量不高时，模型也可能学到错误推理模板。

如果多语言数据占比提高，非英语能力会提升，但在固定 token budget 下，英语和高资源领域的比例会下降。

这就是配比的基本 trade-off：训练 token 是有限预算。提高一种数据的采样权重，通常意味着降低另一种数据的有效权重。

### 4.1 关键公式与审计指标

设清洗、去重和污染隔离后的数据池集合为：

```math
\mathcal{P}=\{P_1,P_2,\ldots,P_K\}
```

每个数据池 `P_k` 有 token 数 `n_k`、质量分 `q_k`、风险分 `r_k` 和能力标签向量：

```math
a_k=(a_{k,\mathrm{general}},a_{k,\mathrm{code}},a_{k,\mathrm{math}},a_{k,\mathrm{multi}},a_{k,\mathrm{safety}})
```

最简单的自然配比是按清洗后 token 数采样：

```math
p_k^{\mathrm{nat}}=\frac{n_k}{\sum_j n_j}
```

多语言和多领域常用平滑采样。用 `0\le \alpha\le 1` 表示平滑强度，可以写成：

```math
p_k(\alpha)=\frac{n_k^\alpha}{\sum_j n_j^\alpha}
```

其中 `\alpha=1` 接近自然规模采样，`\alpha=0` 接近均匀采样。实际工程里也可以把 `\alpha` 写成温度的倒数，面试时重点讲清“平滑大池、抬高小池”的作用。

如果还要考虑质量、目标能力和风险，可以定义目标能力权重 `v_m`，数据池效用为：

```math
u_k=\sum_m v_m a_{km}
```

一个可解释的采样分数可以写成：

```math
s_k=n_k^\alpha \exp(\beta q_k+\gamma u_k-\lambda r_k)
```

最终采样配比为：

```math
p_k=\frac{s_k}{\sum_j s_j}
```

如果总训练预算是 `B` 个 token，则第 `k` 个数据池计划采样 token 数为：

```math
b_k=Bp_k
```

effective epoch 用于衡量小数据池被重复使用的程度：

```math
e_k=\frac{b_k}{n_k}
```

如果 `e_k` 过高，说明该数据池被重复上采样，可能带来记忆、过拟合或污染风险。

对于某个能力维度 `m`，mixture 的能力覆盖估计可以写成：

```math
A_m(p)=\sum_k p_k a_{km}
```

配比方案的多目标效用可以粗略写成：

```math
U(p)=\sum_m v_m A_m(p)-\lambda_r\sum_k p_kr_k-\lambda_e\sum_k \max(0,e_k-e_{\max})
```

这里第一项鼓励目标能力覆盖，第二项惩罚风险，第三项惩罚过度重复小数据池。

配比相对 baseline 的漂移可以用 KL divergence 监控：

```math
D_{\mathrm{KL}}(p\Vert p^0)=\sum_k p_k\log\frac{p_k}{p_k^0}
```

它不是越小越好，而是帮助审计“这次 mixture 改动到底有多大”。

上线训练前可以把门禁写成：

```math
G_{\mathrm{mix}}=I(\min_m A_m(p)\ge a_m^{\min})I(\max_k e_k\le e_{\max})I(\sum_k p_kr_k\le r_{\max})I(D_{\mathrm{KL}}\le d_{\max})
```

面试里要强调：这些公式只是把配比思路显式化。真实训练仍要靠 ablation、小模型预实验、多维评估和大规模复验来校准。

---

## 5. 自然分布 vs 人工配比

一种简单做法是按数据自然规模混合：哪个来源 token 多，就占更大比例。比如网页最多，网页就占绝大多数；英语最多，英语就占绝大多数。

这种做法有好处：

1. 简单。
2. 接近真实 web 分布。
3. 不需要太多人工假设。
4. 覆盖范围广。

但缺点很明显：

1. web 分布不等于理想训练分布。
2. 高资源语言会压制低资源语言。
3. 娱乐、营销、模板和低质内容可能过多。
4. 数学、代码、专业知识等高价值数据可能被稀释。
5. 模型能力会被“互联网上什么最多”支配。

因此，成熟系统通常会人工调配。比如上采样代码、数学、高质量书籍、技术文档和低资源语言；下采样低质量网页、重复来源和模板化内容。

人工配比的风险是引入主观偏置。团队认为重要的数据，不一定对最终模型最优。所以配比必须通过实验验证，而不是只靠直觉。

---

## 6. 通用文本的角色

通用文本通常包括网页正文、百科、书籍、新闻、论坛、问答、博客和文档。它是 base model 的主体，因为它提供世界知识、语言建模能力、常识、事实、表达风格和多样话题。

通用文本的优点：

1. 覆盖广。
2. 规模大。
3. 风格多样。
4. 能支撑开放域问答和通用语言能力。

缺点：

1. 质量不均。
2. 噪声多。
3. 重复和污染风险高。
4. 事实可信度不稳定。
5. 可能包含偏见、毒性和隐私信息。

配比时要避免两个极端：通用文本太少，模型缺少世界知识；通用文本太多，高价值专业能力被稀释。

---

## 7. 代码数据的配比

代码数据对现代大模型非常重要。即使模型不是专门代码模型，适量代码数据也可能提升结构化推理、符号处理、工具使用和长程依赖建模能力。

代码数据包括：

1. 源代码文件。
2. README 和技术文档。
3. issue 和 pull request。
4. 单元测试。
5. API 文档。
6. 编程问答。
7. 教程和代码解释。

提高代码比例通常会增强：

1. 代码补全。
2. bug 修复。
3. API 使用。
4. 结构化输出。
5. 算法题能力。

但代码占比过高可能带来：

1. 普通自然语言能力下降。
2. 输出风格更像代码或模板。
3. 许可证和版权风险上升。
4. secrets、凭据和私有片段风险增加。
5. benchmark contamination 风险增加，尤其是编程题库。

代码数据配比要结合模型定位。如果是通用助手，可以保留适中比例；如果是代码模型，需要显著提高代码及相关解释数据；如果是行业模型，代码可能只是辅助能力。

---

## 8. 数学和推理数据的配比

数学数据通常比普通 web 文本更稀缺，但对推理能力很重要。它包括教材、题库、解答、证明、竞赛题、合成推理题、符号推导和数学论坛。

数学数据的特殊性在于：质量比数量更重要。错误解答、跳步解答、格式混乱和答案泄漏会严重误导模型。

提高数学数据比例可能带来：

1. 算术能力提升。
2. 多步推理能力提升。
3. 公式理解能力提升。
4. 解题格式更稳定。
5. 对 Chain-of-Thought 类数据更敏感。

但风险包括：

1. 题库污染。
2. 模板化推理。
3. 错误步骤被学习。
4. 对自然语言表达挤占。
5. 合成数据同质化。

数学配比要特别重视评估闭环。不能只看数学 benchmark 提升，还要看通用能力是否下降、污染是否可控、推理是否真实泛化。

---

## 9. 多语言数据的配比

多语言配比是大模型训练中最难的问题之一。互联网上英语数据规模和质量通常远超许多语言。如果按自然分布训练，模型会强烈偏向英语。

常见策略包括：

1. 对低资源语言上采样。
2. 对高资源语言下采样。
3. 按语言质量分设置不同阈值。
4. 使用平滑采样，而不是完全按 token 数采样。
5. 对关键目标市场语言设置最低比例。
6. 使用翻译数据或平行语料增强跨语言能力。

多语言配比的 trade-off 很明显：低资源语言比例提高，会改善这些语言能力，但会占用固定 token budget。对于通用英文能力、代码能力和专业领域能力，可能产生机会成本。

还要注意，低资源语言的数据质量过滤不能直接套用英语规则。否则会因为语言识别误差、字符分布差异、短文本特征不同而误删大量有效数据。

---

## 10. 专业领域数据的配比

专业领域数据包括医学、法律、金融、材料、生物、芯片、机器人、教育等。它的价值在于提升专业问答、术语理解、领域推理和行业落地能力。

但专业数据配比有几个难点：

1. 高质量数据稀缺。
2. 授权和合规要求更高。
3. 专家审核成本高。
4. 错误知识风险更大。
5. 与通用能力存在 trade-off。

专业领域数据通常有三种用法：

1. 预训练阶段少量混入，提供基本术语和背景。
2. 继续预训练阶段提高比例，做领域适配。
3. SFT/RAG/工具阶段解决高准确率和实时知识问题。

不要把所有专业能力都压到预训练配比上。对于医学、法律、金融这类高风险领域，单靠预训练记忆不是可靠方案，通常需要检索、工具、引用、审核和安全策略配合。

---

## 11. 合成数据的配比

合成数据可以来自规则生成、程序生成、模拟器、强模型生成、专家模板或数据增强。它在代码、数学、指令跟随、安全训练和低资源语言中很常见。

合成数据的优点：

1. 可控。
2. 可定向补能力短板。
3. 格式统一。
4. 能生成稀缺任务。
5. 可用于构造难例和安全样本。

缺点也明显：

1. 同质化。
2. 事实错误。
3. 风格模板化。
4. 可能继承教师模型偏差。
5. 过量使用会让模型学习生成器的分布，而不是现实分布。

phi-1 的经验说明，高质量合成教材和练习对代码能力很有帮助。但这不意味着合成数据越多越好。关键在于质量、覆盖、多样性、验证和配比。

一个稳妥原则是：合成数据要有明确目的、明确来源、明确标签、明确评估，不要不可追踪地混入自然语料。

---

## 12. 数据配比和训练阶段

不同训练阶段的 data mixture 应该不同。

### 12.1 预训练

预训练目标是建立广泛语言能力、世界知识和基础模式。配比要重视覆盖面、多样性和有效 token 数。

### 12.2 继续预训练

继续预训练常用于领域适配、语言适配或能力增强。此时可以提高目标领域数据比例，但要防止 catastrophic forgetting，即模型在新领域变强的同时通用能力下降。

### 12.3 SFT

SFT 的数据更偏指令、问答、对话、工具调用和格式规范。这里的配比决定助手风格、任务覆盖和指令遵循能力。

### 12.4 偏好训练

偏好训练数据配比影响模型对回答风格、安全边界、简洁性、帮助性和拒答策略的偏好。

### 12.5 安全训练

安全数据不能简单按自然分布采样。需要覆盖风险类别、拒答边界、合规场景、误拒问题和多语言安全表达。

---

## 13. 静态配比 vs 动态配比

静态配比是在整个训练过程中使用固定采样比例。例如网页 60%、代码 15%、数学 5%、书籍 10%、多语言 10%。

优点是简单、可复现、易分析。缺点是不够灵活。

动态配比会随训练阶段调整。例如：

1. 前期更多通用文本，建立基础语言能力。
2. 中期增加代码、数学和高质量知识数据。
3. 后期增加高质量、低噪声和目标能力数据。
4. 对 loss 下降慢或评估薄弱的数据域增加采样。

动态配比更像 curriculum learning，但也更难评估。因为最终效果来自配比、顺序、学习率、数据质量和训练阶段共同作用。

面试中可以说：静态配比适合第一版稳定训练，动态配比适合有充分实验体系的成熟团队。

---

## 14. Temperature sampling 和多语言平滑

多语言训练常用一种思想：不要完全按原始 token 数采样，而是对语言分布做平滑。

如果按原始规模采样，英语等高资源语言会占据绝大多数。若完全均匀采样，低资源语言可能被过度重复，导致过拟合和质量下降。因此需要介于两者之间的策略。

直观做法是：对每种语言的 token 占比做平滑，降低大语言的优势，提高小语言的采样概率。

温度越高，分布越接近均匀；温度越低，越接近原始分布。不同团队记号可能不同，面试中不必纠结公式，关键是讲清目的：在高资源语言和低资源语言之间做折中。

这种思想也可以用于领域配比：对小而高价值的数据源上采样，对大而低价值的数据源下采样。

---

## 15. 配比实验怎么做？

data mixture 不能只靠直觉，必须通过实验闭环。

典型流程：

1. 定义目标能力：通用问答、代码、数学、多语言、专业领域、安全等。
2. 建立候选数据池：每个池有来源、质量分、语言、领域、去重状态和许可证信息。
3. 设计 baseline mixture：基于经验和目标设初始比例。
4. 做小规模训练：使用较小模型和较少 token 快速验证趋势。
5. 做 ablation：单独提高或降低某类数据比例。
6. 建立评估矩阵：不要只看一个 benchmark。
7. 分析 trade-off：某项能力提升是否损害其他能力。
8. 固化版本：记录 mixture 配置、采样权重、数据版本和训练结果。
9. 放大验证：在更大模型或更多 token 上验证是否保持趋势。

配比实验最难的是外推。小模型上的最佳配比不一定适用于大模型；短训练上的最佳配比不一定适用于长训练。因此需要多尺度实验，而不是一次 ablation 就决定最终配比。

---

## 16. 评估矩阵：配比优化看什么指标？

一个配比方案不能只看平均分。它应该看多维指标：

1. 通用语言 loss。
2. 各来源 validation loss。
3. 下游 benchmark。
4. 代码能力。
5. 数学能力。
6. 多语言能力。
7. 专业领域能力。
8. 安全评估。
9. 幻觉和事实性。
10. 输出风格。
11. 记忆和污染风险。
12. 人工评测和产品样例。

例如，提高代码比例后，HumanEval 类指标可能提升，但多语言聊天能力下降。提高数学合成数据后，GSM 类指标可能提升，但模型回答变得模板化。提高论坛数据后，聊天自然度提升，但安全风险上升。

因此配比优化本质是多目标优化，而不是单指标爬山。

---

## 17. 数据上采样和过拟合

当某类数据很重要但规模小，团队常会对它上采样。比如低资源语言、数学题、代码解释、专业文档。

上采样能提高模型 exposure，但也带来过拟合和记忆风险。

如果一个小数据池被重复训练太多次，模型可能记住具体样本，而不是学到泛化规律。尤其是题库、隐私文本、专业文档和代码数据，更要谨慎。

缓解方法包括：

1. 增加数据多样性。
2. 做去重和污染检测。
3. 控制 epoch 数。
4. 使用数据增强或合成变体，但要验证质量。
5. 在评估中加入记忆检测。
6. 对小数据池使用更高质量阈值。

---

## 18. 配比和 tokenizer 的关系

不同语言和数据类型在 tokenizer 下的 token 化效率不同。同样 1000 个字符，英文、中文、代码、数学公式和低资源语言可能对应不同 token 数。

如果配比按 token 计算，tokenizer 会影响实际信息量。如果某种语言 token 化效率低，它在相同 token budget 下能表达的信息更少，训练成本更高。

所以多语言配比不能只看原始字符数或文档数，也要看 token 分布、压缩效率和模型实际 loss。

代码和数学也类似。符号、缩进、特殊字符会影响 token 数。配比实验需要基于最终 tokenizer 统计，而不是原始文本大小。

---

## 19. 配比和去重清洗的关系

配比发生在清洗和去重之后，但也会反过来影响清洗策略。

比如某个领域数据很稀缺，如果用普通规则过滤，可能被误删很多；为了保留这个领域，团队可能降低某些过滤阈值，同时加强人工审计。又比如某个来源规模巨大且重复率高，去重后占比下降，需要重新计算采样权重。

因此，data mixture 不是简单地在最后拼接文件，而是和数据清洗、质量评分、去重、污染检测共同设计。

成熟的数据表通常会保存：source、language、domain、quality score、dedup cluster、safety label、license、token count、sampling weight。训练时 sampler 根据这些字段动态采样。

---

## 20. 面向专家：Data mixture 是隐式目标函数设计

从专家视角看，data mixture 实际上是在设计隐式目标函数。

预训练通常形式上只有一个 next-token prediction loss，但这个 loss 是在 mixture 分布上求期望。改变 mixture，就改变了优化目标。

如果把代码采样权重加倍，优化器会更频繁地沿着代码样本梯度更新。如果把数学数据放到训练后期，模型后期参数会更受数学分布影响。如果把低质量网页下采样，模型对这些风格的拟合会减少。

因此，配比不是数据团队的外围工作，而是训练目标、模型能力和产品定位之间的接口。

专家级 trade-off 包括：

1. 多样性 vs 高质量。
2. 通用能力 vs 专项能力。
3. 高资源语言 vs 低资源语言。
4. 自然数据 vs 合成数据。
5. 预训练吸收 vs 后训练适配。
6. 短期 benchmark 提升 vs 长期泛化。
7. 上采样能力短板 vs 记忆和污染风险。

---

## 21. 一个可落地的 data mixture 方案

如果面试官问：“你要训练一个通用中文/英文大模型，如何设计数据配比？”可以按以下框架回答。

第一步，定义模型目标。明确是通用助手、代码助手、数学模型、多语言模型还是行业模型，以及核心评估指标。

第二步，建立数据池。把数据按来源、语言、领域、质量、安全、许可证和去重状态打标签。

第三步，设计初始配比。通用文本作为主体，适量加入高质量书籍、百科、论文、技术文档、代码、数学、多语言和合成数据。

第四步，设置采样权重。不要只按原始 token 数采样，对高质量、小规模、目标能力相关数据上采样，对低质量、大规模、重复来源下采样。

第五步，做小规模实验。训练多个小模型，对比不同代码比例、数学比例、多语言比例和高质量数据比例。

第六步，建立评估矩阵。覆盖通用问答、事实性、代码、数学、多语言、安全、长文本和人工样例。

第七步，分析 trade-off。看专项能力提升是否损害通用能力，是否增加幻觉、污染、记忆或安全风险。

第八步，放大验证。在更大模型和更多 token 上复验趋势，避免小规模结论误导。

第九步，版本化记录。保存 mixture 配置、数据版本、采样权重、过滤规则、训练曲线和评估结果。

第十步，迭代优化。根据训练结果、产品反馈和安全评估继续调整。

这套回答比直接报一个比例更好，因为真实项目中没有放之四海皆准的比例，只有目标驱动的实验闭环。

### 21.1 最小可运行 data mixture 审计 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy 数据池，每个池有清洗后 token 数、质量分、风险分和能力标签；输出包括自然配比、目标采样配比、计划采样 token、effective epoch、上采样倍数、能力覆盖、平均质量、风险和门禁结果。

它演示的是配比审计机制，不是生产级 sampler。真实训练还需要和 tokenizer 统计、数据版本、分布式采样器、训练曲线、验证 loss、下游评估和安全审计联动。

```python
import math


pools = [
    {"name": "general_web", "tokens": 500_000, "quality": 0.72, "risk": 0.06, "cap": {"general": 0.85, "code": 0.10, "math": 0.10, "multilingual": 0.25, "safety": 0.25}},
    {"name": "books_reference", "tokens": 120_000, "quality": 0.88, "risk": 0.02, "cap": {"general": 0.75, "code": 0.05, "math": 0.25, "multilingual": 0.15, "safety": 0.15}},
    {"name": "code_docs", "tokens": 80_000, "quality": 0.90, "risk": 0.04, "cap": {"general": 0.25, "code": 0.95, "math": 0.30, "multilingual": 0.10, "safety": 0.15}},
    {"name": "math_reasoning", "tokens": 30_000, "quality": 0.93, "risk": 0.03, "cap": {"general": 0.20, "code": 0.25, "math": 0.95, "multilingual": 0.08, "safety": 0.10}},
    {"name": "zh_multilingual", "tokens": 70_000, "quality": 0.82, "risk": 0.04, "cap": {"general": 0.45, "code": 0.05, "math": 0.15, "multilingual": 0.95, "safety": 0.20}},
    {"name": "domain_science", "tokens": 40_000, "quality": 0.87, "risk": 0.05, "cap": {"general": 0.55, "code": 0.10, "math": 0.45, "multilingual": 0.10, "safety": 0.10}},
    {"name": "synthetic_reasoning", "tokens": 20_000, "quality": 0.86, "risk": 0.08, "cap": {"general": 0.35, "code": 0.30, "math": 0.85, "multilingual": 0.08, "safety": 0.20}},
    {"name": "safety_data", "tokens": 10_000, "quality": 0.92, "risk": 0.02, "cap": {"general": 0.15, "code": 0.05, "math": 0.15, "multilingual": 0.20, "safety": 0.98}},
]

target = {"general": 0.35, "code": 0.20, "math": 0.20, "multilingual": 0.15, "safety": 0.10}
BUDGET = 200_000
SMOOTHING_POWER = 0.65
QUALITY_BETA = 1.2
TARGET_GAMMA = 0.9
RISK_LAMBDA = 1.8
MAX_EPOCH = 2.5


def normalize(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def utility(pool):
    return sum(target[k] * pool["cap"][k] for k in target)


raw_mix = normalize({p["name"]: p["tokens"] for p in pools})
raw_ability = {
    cap: round(sum(raw_mix[p["name"]] * p["cap"][cap] for p in pools), 3)
    for cap in target
}

scores = {}
for p in pools:
    scores[p["name"]] = (
        (p["tokens"] ** SMOOTHING_POWER)
        * math.exp(
            QUALITY_BETA * (p["quality"] - 0.75)
            + TARGET_GAMMA * utility(p)
            - RISK_LAMBDA * p["risk"]
        )
    )
mixture = normalize(scores)

plan = {}
for p in pools:
    name = p["name"]
    planned = round(BUDGET * mixture[name])
    plan[name] = {
        "mix": round(mixture[name], 3),
        "raw_mix": round(raw_mix[name], 3),
        "planned_tokens": planned,
        "effective_epoch": round(planned / p["tokens"], 3),
        "utility": round(utility(p), 3),
    }

ability = {
    cap: round(sum(mixture[p["name"]] * p["cap"][cap] for p in pools), 3)
    for cap in target
}
risk = round(sum(mixture[p["name"]] * p["risk"] for p in pools), 3)
avg_quality = round(sum(mixture[p["name"]] * p["quality"] for p in pools), 3)
up_sampling = {
    name: round(plan[name]["mix"] / max(plan[name]["raw_mix"], 1e-9), 2)
    for name in sorted(plan)
}
watchlist = sorted(
    name for name, row in plan.items()
    if row["effective_epoch"] > MAX_EPOCH
)
gates = {
    "code_floor": ability["code"] >= 0.21,
    "math_floor": ability["math"] >= 0.28,
    "multilingual_floor": ability["multilingual"] >= 0.25,
    "safety_floor": ability["safety"] >= 0.21,
    "risk_ok": risk <= 0.055,
    "epoch_ok": not watchlist,
}

print("raw_mix=", {k: round(raw_mix[k], 3) for k in sorted(raw_mix)})
print("target_mix=", {k: plan[k]["mix"] for k in sorted(plan)})
print("planned_tokens=", {k: plan[k]["planned_tokens"] for k in sorted(plan)})
print("effective_epoch=", {k: plan[k]["effective_epoch"] for k in sorted(plan)})
print("upsampling=", up_sampling)
print("raw_ability=", raw_ability)
print("target_ability=", ability)
print("avg_quality=", avg_quality, "risk=", risk)
print("watchlist=", watchlist)
print("gates=", gates)
print("gate_pass=", all(gates.values()))

assert round(sum(mixture.values()), 6) == 1.0
assert watchlist == []
assert ability == {"general": 0.591, "code": 0.218, "math": 0.285, "multilingual": 0.256, "safety": 0.212}
assert avg_quality == 0.823
assert risk == 0.045
assert all(gates.values())
```

运行后会看到类似输出：

```text
raw_mix= {'books_reference': 0.138, 'code_docs': 0.092, 'domain_science': 0.046, 'general_web': 0.575, 'math_reasoning': 0.034, 'safety_data': 0.011, 'synthetic_reasoning': 0.023, 'zh_multilingual': 0.08}
target_mix= {'books_reference': 0.174, 'code_docs': 0.133, 'domain_science': 0.078, 'general_web': 0.351, 'math_reasoning': 0.072, 'safety_data': 0.032, 'synthetic_reasoning': 0.049, 'zh_multilingual': 0.11}
planned_tokens= {'books_reference': 34896, 'code_docs': 26671, 'domain_science': 15534, 'general_web': 70245, 'math_reasoning': 14412, 'safety_data': 6421, 'synthetic_reasoning': 9755, 'zh_multilingual': 22066}
effective_epoch= {'books_reference': 0.291, 'code_docs': 0.333, 'domain_science': 0.388, 'general_web': 0.14, 'math_reasoning': 0.48, 'safety_data': 0.642, 'synthetic_reasoning': 0.488, 'zh_multilingual': 0.315}
upsampling= {'books_reference': 1.26, 'code_docs': 1.45, 'domain_science': 1.7, 'general_web': 0.61, 'math_reasoning': 2.12, 'safety_data': 2.91, 'synthetic_reasoning': 2.13, 'zh_multilingual': 1.38}
raw_ability= {'general': 0.693, 'code': 0.176, 'math': 0.206, 'multilingual': 0.261, 'safety': 0.218}
target_ability= {'general': 0.591, 'code': 0.218, 'math': 0.285, 'multilingual': 0.256, 'safety': 0.212}
avg_quality= 0.823 risk= 0.045
watchlist= []
gates= {'code_floor': True, 'math_floor': True, 'multilingual_floor': True, 'safety_floor': True, 'risk_ok': True, 'epoch_ok': True}
gate_pass= True
```

这个 demo 刻意把 `general_web` 从自然分布的 `0.575` 下调到 `0.351`，同时上采样代码、数学、安全和合成推理数据。结果是通用能力覆盖下降，但代码和数学覆盖上升；这正是 data mixture 的 trade-off，不能只看单项指标。

---

## 22. 常见面试题

### 22.1 什么是 data mixture？

data mixture 是训练集中不同来源、语言、领域、能力类型和质量层级数据的混合比例与采样策略。它决定模型在哪些数据分布上优化 next-token prediction，因此会影响模型能力、风格和安全表现。

### 22.2 为什么不能按数据自然规模混合？

因为互联网自然分布不等于理想训练分布。按自然规模混合会让高资源语言、大规模网页和低质来源占据过高比例，稀释代码、数学、专业领域和低资源语言等高价值数据。

### 22.3 代码数据比例提高一定好吗？

不一定。提高代码比例通常提升代码和结构化推理能力，但会占用 token budget，可能影响自然语言、多语言和对话能力，还会增加 license、secrets、重复和题库污染风险。

### 22.4 多语言配比怎么做？

不能完全按原始 token 数采样，也不能简单均匀采样。通常对低资源语言上采样、对高资源语言适当下采样，用平滑采样在覆盖和质量之间折中，并分语言评估 loss、下游任务和人工样例。

### 22.5 合成数据应该占多少？

没有固定答案。合成数据要看目标任务、质量、覆盖、多样性和教师模型偏差。通常应明确标注来源，单独评估，逐步增加比例，通过 ablation 判断收益和副作用，而不是无追踪地混入训练集。

### 22.6 如何优化 data mixture？

先定义目标能力和评估矩阵，再构建带标签的数据池，设计 baseline mixture，做小规模 ablation，观察各领域 loss、benchmark、安全和人工评测，分析 trade-off，最后在更大规模上验证并版本化记录。

### 22.7 配比和 Chinchilla scaling 有什么关系？

Chinchilla 说明在固定 compute budget 下，模型参数和训练 token 数都很重要。但 token 数只是总量问题，data mixture 解决的是这些 token 来自哪里、质量如何、各能力如何分布。两者是互补关系。

---

## 23. 常见误区

误区一：data mixture 只是数据拼接。

它实际上决定训练分布，是隐式目标函数设计。

误区二：比例有标准答案。

不同模型目标、token budget、数据质量和评估体系下，最佳比例不同。

误区三：高质量数据越多越好。

高质量很重要，但过度偏向单一高质量风格会损失真实世界多样性。

误区四：专项数据越多专项能力越强。

超过一定比例后可能收益递减，并带来过拟合、记忆、污染和通用能力下降。

误区五：小模型 ablation 结论可以直接放大。

小模型趋势有参考价值，但大模型、长训练和不同 tokenizer 下可能变化。

误区六：SFT 可以弥补所有预训练配比问题。

SFT 能调整行为和格式，但很难补齐预训练阶段缺失的大规模知识和语言能力。

---

## 24. 本章小结

Data mixture 与配比是大模型数据工程中最接近“模型能力设计”的环节。它不是把数据随便混在一起，而是在有限 token budget 下决定模型读什么、读多少、什么时候读、以多大权重读。

本章要记住几句话：

1. data mixture 决定训练分布，也就改变隐式优化目标。
2. 通用文本提供覆盖，代码和数学提供结构化能力，多语言和专业数据提供目标能力扩展。
3. 配比不是自然规模决定的，而是目标、质量、风险和实验共同决定的。
4. 合成数据有价值，但必须可追踪、可评估、可控比例使用。
5. 配比优化是多目标问题，要看通用、代码、数学、多语言、安全、事实性和风格的 trade-off。
6. 最成熟的做法不是报固定比例，而是建立数据池、采样权重、评估矩阵、ablation 和版本化闭环。

如果面试中被问到 data mixture，最好的回答是：先定义模型目标，再建立带标签数据池，设计 baseline mixture，通过小规模实验和多维评估找 trade-off，最后在大规模训练中验证并版本化记录。
