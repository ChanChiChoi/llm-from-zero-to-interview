# 第六章：Reasoning、Code、Math Eval

数学题有标准答案，代码可以运行测试，推理任务也常常给出一个最终结论。这种
表面上的确定性很容易让人以为评估很简单：答对就是会，答错就是不会。

实际情况要复杂得多。数学答案可能只是格式不同，代码可能通过公开样例却在
隐藏测试上失败，推理模型可能猜对最终选项却给出错误过程，多个采样路径也
可能共同复现训练集里的题面。一个“正确”标签只有在答案抽取、测试覆盖、
过程证据、题目新鲜度和运行协议都清楚时才有意义。

本章先把三类任务的被测对象拆开，再说明它们共享的实验原则。读者会从最直观
的“答案对不对”开始，逐步看到答案解析、过程证据、测试覆盖和题目迁移如何改变
结论；随后还会把这些概念落到 verifier、`pass@k`、隐藏测试和错误归因上。需要
特别保留的一条边界是：可观察到的过程不一定是真实的内部推理，长
chain-of-thought 也不自动等于更强的 reasoning。

## 1. 评估对象：答案、过程还是能力迁移

### 1.1 小白视角：三种“做对了”

一道题的最终答案正确，可能有三种来源：

1. 模型理解了问题并完成了有效推导；
2. 模型使用了模式匹配或猜测，碰巧得到正确结果；
3. 模型见过原题、答案或模板，直接复现了结果。

如果只记录最终答案，这三种情况会被合并。对于低风险、答案唯一且测试充分
的任务，这种合并可能是可接受的工程近似；对于研究 reasoning、训练 verifier
或评估安全规划，它会隐藏重要差异。

### 1.2 专家视角：三个层次

可以把被测能力拆成三个层次：

~~~text
outcome：最终答案或最终任务状态是否正确
process：中间步骤、约束、证据和工具轨迹是否合理
generalization：改变题面、数字、上下文、难度和时间后是否仍然正确
~~~

三层不是简单的包含关系。一个代码候选可以通过所有测试，却没有可读的解释；
一个数学解法过程正确，最后因算术笔误失败；一个模型在原题和轻微变体上都答对，
却在新领域完全失效。报告要把这些结果分开保存。

### 1.3 用任务契约确定分母

每类任务先写清楚：

~~~text
输入是什么
允许哪些资源和重试
最终成功状态是什么
哪些中间行为必须正确
哪些失败属于解析、执行、超时或安全错误
样本、候选和用户的统计单位是什么
~~~

如果代码任务允许 10 次候选生成，`pass@10` 可能有业务意义；如果用户只允许
一次响应，主指标应是 `pass@1`。如果 Agent 可以调用工具，最终文本正确却产生
错误副作用，任务仍然失败。指标必须从契约推导，而不是从流行 benchmark 名称
反推。

这份契约还决定了“什么算一次尝试”。例如，代码助手先生成补丁、再运行测试，
是一次包含验证阶段的任务；如果系统允许自动修复后重试，重试次数就属于资源
预算，而不是可以从报告中省略的实现细节。数学题若允许调用计算器，答案的正确
性和工具轨迹也应按同一协议记录。只有把成功状态、资源预算和失败分类写在同一
份契约里，读者才能判断一个分数究竟衡量了模型，还是衡量了搜索和外部工具。

### 1.4 评估样本 schema

一个 reasoning/code/math 样本可以抽象为：

~~~math
e_i=(x_i,y_i,r_i,z_i)
~~~

`x_i` 是题目或任务描述，`y_i` 是标准答案、测试、最终状态或期望行为，`r_i` 是
可选的 rubric、过程标签、参考证明或验证器配置，`z_i` 是任务类型、难度、来源、
时间、语言、资源预算和污染风险等元数据。

评估记录还应保存模型 revision、prompt、token budget、temperature、采样数、
verifier revision、sandbox 版本、失败状态和原始输出。没有这些字段，两个分数
即使看起来可以相减，也不一定来自同一量尺。

## 2. 最终答案评估的共同基础

### 2.1 Accuracy 是必要但有限

若有 `N` 个样本，模型预测为 `\hat a_i`，标准答案为 `a_i`，最终答案准确率是：

~~~math
A_{\mathrm{ans}}
=
\frac{1}{N}
\sum_{i=1}^{N}I(\hat a_i=a_i)
~~~

这个指标简单、可复现、适合封闭题。它没有告诉我们答案是否通过猜测得到，
也没有告诉我们中间过程、答案解析和测试覆盖是否可靠。

### 2.2 解析失败是独立结果

模型可能给出一长段说明，却没有按要求输出最后答案；也可能输出多个互相冲突
的答案。设无法从输出中解析目标答案的样本数为 `N_{\mathrm{parseFail}}`，解析
失败率为：

~~~math
R_{\mathrm{parseFail}}
=
\frac{N_{\mathrm{parseFail}}}{N}
~~~

解析失败不能悄悄从分母中删除。它可能代表模型能力不足、格式控制失败、token
预算不足或解析器过于脆弱；不同原因对应不同修复路径。

### 2.3 等价答案

数学、代码和结构化输出通常存在多个等价表示。评估器需要明确：

1. 哪些空白、标点和解释可以忽略；
2. 分数、小数、百分比是否等价；
3. 集合和字典的顺序是否有意义；
4. 浮点误差容忍范围是多少；
5. 代码是比较文本、AST、行为还是最终状态；
6. 单位和数量级是否必须显式出现。

答案归一化函数 `\nu` 也要有版本，并用正例和反例测试。一个过于宽松的归一化
可能把错误答案判成正确，一个过于严格的归一化则会把等价答案判错。

### 2.4 最终答案正确但过程错误

例如题目要求计算：

~~~text
若 x + 3 = 10，求 x。
~~~

模型先写“两个数相加会变大”，随后给出 `x=7`。最终答案正确，但解释包含
不相关或错误的规则。对于普通问答可以只记结果；对于数学教学、科学推理和
安全决策，就需要抽样检查过程，不能让正确结果掩盖错误逻辑。

### 2.5 最终答案错误但局部能力存在

一个多步解法前八步正确，最后一步把 `7*8` 算成 `54`；另一个模型第一步就
误解了题目，但碰巧选中了正确选项。两者最终 accuracy 都记为 0，却需要不同
的训练和调试策略。过程标签、错误类别和人工复核可以区分它们。

## 3. Math Eval：答案等价、过程与证明

### 3.1 数学任务的几种形态

数学评估至少包括：

1. 单选或判断题；
2. 整数、分数、小数和区间答案；
3. 多步应用题；
4. 形式化证明或证明草图；
5. 符号推导、方程和不等式；
6. 几何、组合、概率和竞赛问题；
7. 需要工具或代码辅助的数学任务。

答案越开放，字符串匹配越不可靠，过程、约束和 verifier 的作用越大。

### 3.2 答案抽取

模型输出可能是：

~~~text
先移项得到 x=7，所以最终答案为 7。
~~~

评估器需要明确“最终答案”的位置和格式。可以要求模型在 `final_answer` 标签
中输出，也可以使用可靠 parser 抽取最后一个候选，但必须测试：

1. 输出中间步骤里出现多个数字；
2. 同时出现草稿答案和最终修正答案；
3. 最终答案放在 LaTeX、列表或句子中；
4. 模型明确表示无法作答；
5. 模型给出两个互相冲突的结果。

抽取成功不代表答案正确，抽取失败也不一定代表推理失败；二者要分开计数。

### 3.3 数值和分数归一化

下面这些答案在特定题目条件下可能等价：

~~~text
1/2
0.5
50%
\frac{1}{2}
~~~

可以将输入解析成有理数或符号表达式，再比较数值或结构。浮点答案要定义容忍
误差 `\epsilon`：

~~~math
I_{\mathrm{num}}
=
I\left(|\hat a-a|\le\epsilon\right)
~~~

`\epsilon` 不能对所有题固定。金额、概率、物理量、近似算法和精确整数需要
不同口径；单位、有效数字和范围条件也可能是答案的一部分。

### 3.4 符号等价与集合答案

`x^2-1` 与 `(x-1)(x+1)` 在多项式领域可能等价；`[1,2]` 与 `{2,1}` 是否
等价取决于题目要求的是列表还是集合；矩阵的行列顺序通常不能随意交换。符号
化简器能帮助判断，却也可能有定义域、分支和数值稳定性问题。不能把某个
CAS 输出直接当作所有数学题的通用真理。

### 3.5 最终答案与过程标签

对多步题，可以把第 `l` 个步骤是否被参考约束支持记为 `v_{il}`，样本过程分为：

~~~math
S_{\mathrm{proc},i}
=
\frac{1}{L_i}
\sum_{l=1}^{L_i}v_{il}
~~~

`L_i` 是被标注的步骤数，`v_{il}` 可以是 0/1，也可以是带不确定性的等级。
这个分数只在步骤切分和标注规范稳定时可比。不同解法的步骤数量不同，不能
因为写得更长就获得更高过程分；需要以关键 claim、必要约束或证明义务为单位。

### 3.6 证明题的多路径问题

证明题往往没有唯一推理路径。评估器可以：

1. 用形式化证明系统检查可验证结论；
2. 用符号或数值 verifier 检查关键等式和不等式；
3. 让领域专家按 claim 和缺口标注；
4. 用 LLM judge 做低成本初筛，再以人工 gold set 校准；
5. 把“结论正确”和“证明完整”分开报告。

一个过程看起来清楚，不等于它满足证明义务；一个形式化证明通过，也不等于
语言解释适合目标读者。

### 3.7 数学错误分类

建议至少区分：

~~~text
题意理解错误
变量或单位定义错误
公式选择错误
代数变形错误
算术计算错误
边界条件遗漏
逻辑跳步
答案抽取或格式错误
题目/参考答案/评估器错误
~~~

错误分类比单纯降低分数更能指导修复。若大多数错误来自抽取，应该改协议或
parser；若来自边界条件，应该改数据和过程监督；若来自算术，可考虑工具或
verifier，而不是笼统增加训练量。

## 4. Code Eval：让程序在真实约束下执行

### 4.1 代码任务的成功定义

代码评估不是判断代码看起来是否合理，而是定义程序在一组输入、环境和资源
限制下是否完成预期行为。最小流程是：

~~~text
题目和接口
  -> 候选代码
  -> 语法/类型检查
  -> 隔离沙箱
  -> 公开与隐藏测试
  -> 超时、资源和安全检查
  -> 通过、失败或不可判定
~~~

“能运行一次”不等于“实现了函数契约”。测试、资源和安全边界是代码评估的一部分。

### 4.2 测试通过的数学定义

对题目 `i` 的第 `j` 个候选程序 `g_{ij}`，第 `m` 个测试是否通过可以记为：

~~~math
z_{ijm}=I\left(T_{im}(g_{ij})=1\right)
~~~

如果必须通过 `M_i` 个测试，候选通过全部测试的标记是：

~~~math
p_{ij}
=
\prod_{m=1}^{M_i}z_{ijm}
~~~

任何一个关键测试失败，整体行为就失败。若测试包含不同严重度，可以额外报告
按测试类型加权的结果，但不能把一个安全违规用多个普通测试通过抵消。

### 4.3 公开测试与隐藏测试

公开测试帮助模型理解输入输出，隐藏测试检验泛化。只使用公开样例容易产生：

1. hard-code 样例；
2. 忽略边界；
3. 只实现示例路径；
4. 利用测试实现细节；
5. 对公开函数签名和答案的记忆。

隐藏测试也不是越多越好。它们需要覆盖任务契约中的行为，而不是随机堆数字。
测试集还应保存生成种子、分布、边界类型、预期错误和版本，便于定位“模型错”
与“测试漏了”之间的区别。

### 4.4 测试覆盖的多个维度

代码评估至少要覆盖：

| 维度 | 例子 |
| --- | --- |
| 正常输入 | 常见合法请求 |
| 边界输入 | 空列表、零、最大长度、重复值 |
| 异常输入 | 缺失字段、非法格式、错误类型 |
| 规模 | 小规模与接近资源上限的输入 |
| 行为 | 多个等价实现和状态变化 |
| 性能 | 时间复杂度、内存和 I/O |
| 安全 | 网络、文件、进程和环境变量访问 |

公开测试和隐藏测试应按这些维度分层，而不是只报告测试条数。

### 4.5 沙箱和副作用

模型生成的代码可能删除文件、访问网络、读取环境变量、启动进程、制造死循环
或消耗全部内存。代码评估环境至少需要限制：

1. CPU 时间和墙钟时间；
2. 内存、磁盘和输出大小；
3. 文件系统读写范围；
4. 网络和 DNS；
5. 子进程和系统调用；
6. 可导入库和运行用户；
7. 测试数据与宿主机的隔离。

沙箱失败要和普通单测失败分开记录。否则代码质量和环境防护会被混在一个
“不通过”里，既无法比较模型，也无法修复基础设施。

### 4.6 补丁和大型仓库任务

对于 SWE 类任务，单个函数的 unit test 不够。需要记录 patch 是否能应用、
构建是否成功、已有测试是否回归、修改范围是否合理、是否改变不相关文件，
以及最终 issue 是否关闭。最终文本解释只能作为辅助，仓库状态和测试结果
才是主要证据。

### 4.7 代码错误分类

代码失败可以拆成：

~~~text
parse_or_compile_error
runtime_error
wrong_output
edge_case_failure
complexity_timeout
resource_exhaustion
environment_or_dependency_error
unsafe_side_effect
test_harness_error
~~~

这种分类可以直接映射到生成、测试、沙箱、依赖和题目设计。隐藏测试失败还要
保存最小失败输入或可脱敏的错误摘要，避免只留一个通过率数字。

## 5. pass@k：多次采样下的成功概率

### 5.1 指标回答什么问题

代码模型经常具有采样随机性。一次生成失败，不代表候选空间里没有正确程序；
如果系统允许生成多个候选并运行 verifier，至少一个通过的概率就有实际意义。

`pass@1` 近似一次请求的成功率，`pass@k` 描述最多观察 `k` 个候选时的潜在成功
率。二者不要互相替代：高 `pass@10` 可能只说明搜索有效，不能保证首个答案可用。

### 5.2 无偏估计

对同一道题生成 `n` 个候选，其中 `c` 个通过隐藏测试，从中随机取 `k` 个，
至少一个正确的无偏估计常写成：

~~~math
\widehat{\operatorname{pass@}k}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
~~~

`n` 是候选总数，`c` 是通过数，`k` 是允许尝试数，`k\le n`。当 `c=0` 时结果为
0；当 `n-c<k` 且 `c>0` 时，任何 `k` 个候选都会含有通过者，结果为 1。

### 5.3 手算例子

假设 `n=10`、`c=2`：

~~~math
\operatorname{pass@1}
=
1-\frac{\binom{8}{1}}{\binom{10}{1}}
=0.2
~~~

而 `k=5` 时：

~~~math
\operatorname{pass@5}
=
1-\frac{\binom{8}{5}}{\binom{10}{5}}
\approx0.778
~~~

这不是说模型一次回答有 77.8% 的概率正确，而是说在这 10 个候选的经验分布
下，随机选 5 个至少包含一个正确候选的估计值。

### 5.4 采样协议

比较 `pass@k` 必须固定或报告：

1. `n` 和 `k`；
2. temperature、top-p 和随机种子；
3. 候选是否独立生成；
4. 是否先用公开测试筛选；
5. 测试、超时和资源限制；
6. 候选 token budget 和最大代码长度；
7. 重复候选和解析失败的处理。

如果先用公开测试筛出候选，再在隐藏测试上报告 `pass@k`，这测量的是“带公开
筛选器的搜索流程”，不是单纯的模型采样能力，必须单独命名。

### 5.5 `pass@k` 的代价和边界

多次采样会增加模型 token、编译、测试和沙箱成本；候选高度相似时，`k` 增加
带来的收益很小；弱测试则会把错误候选误判为成功。一个工程系统还要报告：

~~~math
C_{\mathrm{success}}
=
\frac{C_{\mathrm{generation}}+C_{\mathrm{verification}}}
{N_{\mathrm{successful\ tasks}}}
~~~

如果通过提高 `k` 得到更高 `pass@k`，但单位成功任务成本暴涨，实际产品可能
更适合改进单次生成或 verifier，而不是继续增加候选数。

## 6. Self-Consistency：从答案分布观察稳定性

### 6.1 基本流程

Self-consistency 对同一道题采样多条推理路径，抽取并归一化每个最终答案，
再投票：

~~~math
\hat a_{\mathrm{sc}}
=
\operatorname{mode}
(\hat a^{(1)},\ldots,\hat a^{(K)})
~~~

例如五条路径得到 `42,42,40,42,41`，最终投票为 42。它既可以是推理时策略，
也可以是评估工具，用来观察单次输出的稳定性。

### 6.2 为什么可能有效

如果错误路径较分散而正确路径具有较高概率，多次采样投票可以降低偶然错误。
它依赖一个重要假设：不同路径的错误不能高度相关。如果所有路径都使用同一
个错误模板、同一条记忆或同一个错误前提，投票只会把共同错误放大。

### 6.3 答案分布指标

设归一化答案 `a` 的经验概率为 `p(a)`，答案熵为：

~~~math
H_{\mathrm{ans}}
=
-\sum_a p(a)\log p(a)
~~~

前两名答案的投票差可以写成：

~~~math
M_{\mathrm{vote}}
=
p_{\mathrm{top1}}-p_{\mathrm{top2}}
~~~

熵高、margin 小通常表示不稳定，但低熵也不保证正确：模型可能非常一致地
复现一个错误答案。应同时查看 gold correctness、题目难度和污染风险。
本章 demo 使用自然对数，因此熵的单位是 nat；换用以 2 为底的对数时，数值会变成
bit，但不会改变同一组样本中“更稳定”或“更不稳定”的排序。

### 6.4 评估预算

Self-consistency 的成本大约随 `K` 增长，但缓存、批处理和共享前缀可能改变
实际成本。报告应同时给 greedy accuracy、self-consistency accuracy、`K`、
解码参数、答案熵、投票 margin 和单位成功成本。不能只把投票后的分数拿来与
单次 greedy 的分数比较，却不说明多花了多少计算。

## 7. Verifier：把生成与检查分开

### 7.1 Outcome verifier

Outcome verifier 只检查最终答案或最终状态。代码单元测试、数学数值代入、
JSON schema 和数据库状态断言都属于这一类。它通常客观、可执行、成本低，
但无法定位哪一步推理出错，也可能奖励“错误过程 + 正确答案”。

### 7.2 Process verifier

Process verifier 检查中间步骤、关键声明或工具轨迹：

~~~text
Step 1：正确
Step 2：正确
Step 3：违反题目约束
Step 4：结论偶然正确
~~~

它能帮助错误归因、过程监督和 verifier 训练，但需要步骤边界、替代解法和
专家标注。不能因为一个步骤没有出现在参考解法中就判错。

### 7.3 Rule-based verifier

优先使用规则或执行器的任务包括：

1. 代码编译和单元测试；
2. 数学表达式求值和符号约束；
3. JSON schema 和结构约束；
4. 形式化逻辑或定理证明；
5. 工具权限和最终数据库状态；
6. 输出长度、格式和资源限制。

规则 verifier 的优点是确定性和可复现，缺点是覆盖范围有限。测试本身有 bug、
参考答案错误或沙箱不一致时，verifier 也会产生错误标签。

### 7.4 LLM-based verifier

开放式证明、系统设计和复杂过程可能需要 LLM verifier。使用时应：

1. 用人工 gold set 校准；
2. 明确输入证据和 rubric；
3. 测量 false positive 和 false negative；
4. 进行候选注入和顺序偏差测试；
5. 不把 verifier 分数当作最终事实来源。

### 7.5 Verifier 选择偏差

如果从多个候选中选择 verifier 分数最高者，再用同一个 verifier 报告最终质量，
会产生选择偏差：verifier 既挑选答案，又评价答案。应该使用独立测试、不同
verifier、人工 holdout 或交叉验证，避免对同一候选池过度乐观。

## 8. 过程评估与结果评估

### 8.1 Outcome evaluation

Outcome 适合答案唯一、测试充分、最终状态可检查的任务。优点是成本低、自动化
程度高、横向比较容易；缺点是看不到投机策略和错误步骤。

### 8.2 Process evaluation

Process 适合多步数学、证明、规划、多工具调用和教学解释。它能定位错误，但
标注成本高、正确路径不唯一、可观察 chain-of-thought 可能不是模型内部真正
推理，也容易被语言流畅度影响。

因此 process label 应尽量针对关键状态和可验证 claim，而不是评价一段文字
“看起来像不像好推理”。

### 8.3 组合方式

一个完整的评估结果可以包含：

~~~text
最终答案/任务成功
解析与格式
关键过程或 claim
verifier 结果
错误类型
题目变体泛化
污染风险
成本和延迟
~~~

主指标根据业务目标选择；过程和错误指标用于解释，不能把不同分母的结果强行
压成一个总分。

## 9. Benchmark 设计：让题目测迁移能力

### 9.1 难度分层

数学可以按算术、代数、几何、组合、概率和证明分层；代码可以按字符串、数据
结构、动态规划、图算法、并发和大型仓库分层；reasoning 可以按单跳、多跳、
约束数、工具数和状态长度分层。

难度标签应来自专家、历史人类成功率、多个基线和任务结构，不能只把某个模型
答错就称为 hard。模型分数用于观察区分度，而不是单独定义真难度。

### 9.2 参数化变体

变体改变数字、变量名、实体名、语言表达和上下文顺序，但保持目标能力和难度
尽量相近。变体可以检验模型是否掌握规则，也可以降低题面记忆；不过变体
生成器本身要经过人工抽样和难度校准。

### 9.3 反模板和新鲜题

不要让所有题拥有相同叙述、字段、答案位置和推理步骤。加入新写题目、时间
切分题、私有题、对抗题和自然用户表达，才能减少模板捷径。公开 benchmark
仍然有可比性价值，但不应成为唯一证据。

### 9.4 题目和参考答案的污染

Reasoning/code/math benchmark 需要检查题面、解析、测试、函数签名、verifier、
答案格式和示例输出，而不是只检查题目文本。具体污染路径和时间/近重复方法
见上一章；本章只需把污染信号作为最终报告的一部分。

### 9.5 题目、候选和用户的统计单位

数学 accuracy 通常以题目为单位；代码 `pass@k` 以题目和候选池为单位；线上
任务成功可能以用户或会话为单位。多个候选、多个采样和多个用户不能不加说明
地当成独立样本。置信区间应在实际随机化单位上重采样。

## 10. 统一报告和实验协议

### 10.1 必须固定的条件

至少记录：

~~~text
model_revision
prompt_revision
tokenizer_revision
temperature / top_p
max_output_tokens
sampling_count
answer_parser_revision
verifier_revision
test_harness_revision
sandbox_revision
dataset_revision
seed_policy
~~~

代码还要记录编译器、运行时、依赖和硬件；数学还要记录数值精度、符号引擎和
容忍误差；reasoning 还要记录是否展示过程、是否允许工具和如何抽取答案。

### 10.2 子指标

建议同时报告：

1. 最终答案 accuracy 和解析失败率；
2. 按难度、语言、题型和时间切片的结果；
3. self-consistency accuracy、熵和 vote margin；
4. `pass@1`、`pass@k`、测试通过、超时和沙箱错误；
5. 过程关键步骤正确率和错误类别；
6. verifier 的一致性、漏报和误报；
7. 污染命中、变体差异和人工复核；
8. token、延迟、候选数和单位成功成本。

### 10.3 不确定性

accuracy、slice score 和错误率可以用 bootstrap 或二项区间估计不确定性；`pass@k`
还受每道题采样数量和候选相关性影响；过程标签可能共享评审员和参考解法。
报告要写清重采样单位和缺失处理，不能把一个小题集的三位小数当作精确真理。

### 10.4 结果之间的冲突

下面这种结果并不矛盾：

~~~text
greedy accuracy 高，但 self-consistency entropy 也高
pass@10 高，但 pass@1 低且单位成本高
公开测试通过率高，但隐藏测试差距大
最终答案正确，但过程关键步骤错误
judge 认为过程好，但 verifier 失败
~~~

冲突意味着不同指标测到了不同层次，下一步应分析样本和协议，而不是挑一个
最漂亮的数字。

## 11. 错误定位：分数之后要问为什么

### 11.1 数学错误

题意理解、变量定义、公式选择、代数变形、算术、单位、边界、逻辑跳步、答案
抽取和参考答案错误应分开。每个错误类别都应连接到样本、步骤和修复动作。

### 11.2 代码错误

语法、运行时、输出、边界、复杂度、资源、依赖、环境、安全副作用、测试 harness
和参考实现错误应分开。公开通过而隐藏失败通常说明覆盖或泛化问题，不一定是
采样问题。

### 11.3 Reasoning 错误

常见错误包括：

1. 误读条件；
2. 丢失中间约束；
3. 把相关性当因果性；
4. 选择错误工具或参数；
5. 计划步骤不满足最终状态；
6. 过程文本流畅但关键 claim 无依据；
7. 通过记忆答案却无法处理变体。

### 11.4 评估系统错误

答案 parser、测试用例、沙箱、超时、符号计算器、参考答案和 judge 也可能错。
每次评估都应抽样复核“模型失败”和“评估器失败”的边界，否则会训练系统
去修复不存在的问题。

## 12. 一个可运行的 Reasoning/Code/Math 诊断 demo

下面的 demo 不调用模型，也不执行外部生成代码。它处理 toy 数学输出、代码
候选统计和评估元数据，演示答案归一化、greedy 与 self-consistency、entropy、
过程分、`pass@k`、公开/隐藏测试差距、污染风险以及具体后续动作。示例数据
故意同时包含一个过程错误、一个弱隐藏测试集和一个污染风险样本。

~~~python
from collections import Counter
from fractions import Fraction
from math import comb, log
from pprint import pprint
import re


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def extract_number(text):
    lowered = text.lower()
    frac = re.findall(r"-?\d+\s*/\s*-?\d+", lowered)
    if frac:
        return Fraction(frac[-1].replace(" ", ""))
    pct = re.findall(r"-?\d+(?:\.\d+)?\s*%", lowered)
    if pct:
        value = float(pct[-1].replace("%", ""))
        return Fraction(value / 100).limit_denominator(1000)
    nums = re.findall(r"-?\d+(?:\.\d+)?", lowered)
    if nums:
        return Fraction(float(nums[-1])).limit_denominator(1000)
    return None


def equivalent(pred, gold, tolerance=1e-3):
    pred_value = extract_number(pred)
    gold_value = extract_number(gold)
    if pred_value is None or gold_value is None:
        return False
    return abs(float(pred_value - gold_value)) <= tolerance


def canonical(text):
    value = extract_number(text)
    if value is None:
        return "unparsed"
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * log(count / total) for count in counts.values())


math_cases = [
    {
        "id": "math_frac",
        "type": "fraction",
        "gold": "2/3",
        "greedy": "The answer is 0.6666667.",
        "samples": ["2/3", "0.6666667", "66.7%", "1/2", "2/3"],
        "process_steps": [True, True, False],
        "contamination_risk": False,
    },
    {
        "id": "math_algebra",
        "type": "algebra",
        "gold": "7",
        "greedy": "x = 7",
        "samples": ["7", "7", "8", "7", "no final answer"],
        "process_steps": [True, True, True],
        "contamination_risk": False,
    },
    {
        "id": "math_distractor",
        "type": "distractor",
        "gold": "18",
        "greedy": "The final answer is 20.",
        "samples": ["20", "18", "20", "18", "20"],
        "process_steps": [True, False, False],
        "contamination_risk": True,
    },
]

math_reports = []
for case in math_cases:
    sample_labels = [canonical(sample) for sample in case["samples"]]
    counts = Counter(sample_labels)
    top_two = counts.most_common(2)
    top_label, top_count = top_two[0]
    second_count = top_two[1][1] if len(top_two) > 1 else 0
    vote_margin = (top_count - second_count) / len(sample_labels)
    math_reports.append(
        {
            "id": case["id"],
            "greedy_correct": equivalent(case["greedy"], case["gold"]),
            "sc_answer": top_label,
            "sc_correct": equivalent(top_label, case["gold"]),
            "entropy": round(entropy(sample_labels), 3),
            "vote_margin": round(vote_margin, 3),
            "process_score": round(sum(case["process_steps"]) / len(case["process_steps"]), 3),
            "parse_failed": canonical(case["greedy"]) == "unparsed",
            "contamination_risk": case["contamination_risk"],
        }
    )

code_tasks = [
    {
        "id": "reverse_words",
        "n": 10,
        "correct": 3,
        "public_pass": 4,
        "hidden_pass": 3,
        "tests": {"public": 3, "hidden": 8},
        "error": "edge_case",
    },
    {
        "id": "two_sum",
        "n": 10,
        "correct": 1,
        "public_pass": 3,
        "hidden_pass": 1,
        "tests": {"public": 2, "hidden": 10},
        "error": "complexity",
    },
    {
        "id": "parse_table",
        "n": 10,
        "correct": 0,
        "public_pass": 2,
        "hidden_pass": 0,
        "tests": {"public": 2, "hidden": 6},
        "error": "runtime",
    },
]

code_reports = []
for task in code_tasks:
    public_pass_rate = task["public_pass"] / task["n"]
    hidden_pass_rate = task["hidden_pass"] / task["n"]
    code_reports.append(
        {
            "id": task["id"],
            "pass@1": round(pass_at_k(task["n"], task["correct"], 1), 3),
            "pass@5": round(pass_at_k(task["n"], task["correct"], 5), 3),
            "public_pass_rate": round(public_pass_rate, 3),
            "hidden_pass_rate": round(hidden_pass_rate, 3),
            "public_hidden_gap": round(public_pass_rate - hidden_pass_rate, 3),
            "hidden_tests": task["tests"]["hidden"],
            "error": task["error"],
        }
    )

greedy_accuracy = sum(report["greedy_correct"] for report in math_reports) / len(math_reports)
sc_accuracy = sum(report["sc_correct"] for report in math_reports) / len(math_reports)
parse_fail_rate = sum(report["parse_failed"] for report in math_reports) / len(math_reports)
avg_entropy = sum(report["entropy"] for report in math_reports) / len(math_reports)
avg_vote_margin = sum(report["vote_margin"] for report in math_reports) / len(math_reports)
avg_process_score = sum(report["process_score"] for report in math_reports) / len(math_reports)
avg_pass1 = sum(report["pass@1"] for report in code_reports) / len(code_reports)
avg_pass5 = sum(report["pass@5"] for report in code_reports) / len(code_reports)
contamination_flags = [
    report["id"]
    for report in math_reports
    if report["contamination_risk"]
]
weak_hidden_tests = [
    report["id"]
    for report in code_reports
    if report["hidden_tests"] < 8
]
public_hidden_gaps = [
    (report["id"], report["public_hidden_gap"])
    for report in code_reports
    if report["public_hidden_gap"] >= 0.20
]
error_counts = Counter(report["error"] for report in code_reports)

signals = {
    "answer_accuracy_ok": greedy_accuracy >= 0.60,
    "parse_rate_ok": parse_fail_rate <= 0.05,
    "self_consistency_not_worse": sc_accuracy >= greedy_accuracy,
    "code_pass1_ok": avg_pass1 >= 0.20,
    "hidden_tests_sufficient": not weak_hidden_tests,
    "process_score_ok": avg_process_score >= 0.70,
    "contamination_clear": not contamination_flags,
}

actions = []
if not signals["code_pass1_ok"]:
    actions.append("improve_single_sample_code_reliability")
if not signals["hidden_tests_sufficient"]:
    actions.append("expand_weak_hidden_test_slices")
if not signals["process_score_ok"]:
    actions.append("review_process_error_annotations")
if not signals["contamination_clear"]:
    actions.append("remove_or_rebuild_contaminated_math_items")

decision = "revise_eval_before_comparison" if actions else "continue_with_sliced_report"
summary = {
    "math_summary": {
        "greedy_accuracy": round(greedy_accuracy, 3),
        "self_consistency_accuracy": round(sc_accuracy, 3),
        "parse_fail_rate": round(parse_fail_rate, 3),
        "avg_answer_entropy": round(avg_entropy, 3),
        "avg_vote_margin": round(avg_vote_margin, 3),
        "avg_process_score": round(avg_process_score, 3),
    },
    "math_reports": math_reports,
    "code_reports": code_reports,
    "code_summary": {
        "avg_pass@1": round(avg_pass1, 3),
        "avg_pass@5": round(avg_pass5, 3),
        "weak_hidden_tests": weak_hidden_tests,
        "public_hidden_gaps": public_hidden_gaps,
        "error_counts": dict(error_counts),
    },
    "contamination_flags": contamination_flags,
    "signals": signals,
    "actions": actions,
    "decision": decision,
}

pprint(summary, sort_dicts=False)
~~~

实际输出为：

~~~text
{'math_summary': {'greedy_accuracy': 0.667,
                  'self_consistency_accuracy': 0.667,
                  'parse_fail_rate': 0.0,
                  'avg_answer_entropy': 0.858,
                  'avg_vote_margin': 0.333,
                  'avg_process_score': 0.667},
 'math_reports': [{'id': 'math_frac',
                   'greedy_correct': True,
                   'sc_answer': '2/3',
                   'sc_correct': True,
                   'entropy': 0.95,
                   'vote_margin': 0.4,
                   'process_score': 0.667,
                   'parse_failed': False,
                   'contamination_risk': False},
                  {'id': 'math_algebra',
                   'greedy_correct': True,
                   'sc_answer': '7',
                   'sc_correct': True,
                   'entropy': 0.95,
                   'vote_margin': 0.4,
                   'process_score': 1.0,
                   'parse_failed': False,
                   'contamination_risk': False},
                  {'id': 'math_distractor',
                   'greedy_correct': False,
                   'sc_answer': '20',
                   'sc_correct': False,
                   'entropy': 0.673,
                   'vote_margin': 0.2,
                   'process_score': 0.333,
                   'parse_failed': False,
                   'contamination_risk': True}],
 'code_reports': [{'id': 'reverse_words',
                   'pass@1': 0.3,
                   'pass@5': 0.917,
                   'public_pass_rate': 0.4,
                   'hidden_pass_rate': 0.3,
                   'public_hidden_gap': 0.1,
                   'hidden_tests': 8,
                   'error': 'edge_case'},
                  {'id': 'two_sum',
                   'pass@1': 0.1,
                   'pass@5': 0.5,
                   'public_pass_rate': 0.3,
                   'hidden_pass_rate': 0.1,
                   'public_hidden_gap': 0.2,
                   'hidden_tests': 10,
                   'error': 'complexity'},
                  {'id': 'parse_table',
                   'pass@1': 0.0,
                   'pass@5': 0.0,
                   'public_pass_rate': 0.2,
                   'hidden_pass_rate': 0.0,
                   'public_hidden_gap': 0.2,
                   'hidden_tests': 6,
                   'error': 'runtime'}],
 'code_summary': {'avg_pass@1': 0.133,
                  'avg_pass@5': 0.472,
                  'weak_hidden_tests': ['parse_table'],
                  'public_hidden_gaps': [('two_sum', 0.2), ('parse_table', 0.2)],
                  'error_counts': {'edge_case': 1, 'complexity': 1, 'runtime': 1}},
 'contamination_flags': ['math_distractor'],
 'signals': {'answer_accuracy_ok': True,
             'parse_rate_ok': True,
             'self_consistency_not_worse': True,
             'code_pass1_ok': False,
             'hidden_tests_sufficient': False,
             'process_score_ok': False,
             'contamination_clear': False},
 'actions': ['improve_single_sample_code_reliability',
             'expand_weak_hidden_test_slices',
             'review_process_error_annotations',
             'remove_or_rebuild_contaminated_math_items'],
 'decision': 'revise_eval_before_comparison'}
~~~

这里的 `public_hidden_gap` 是同一批 `n` 个候选在公开测试与隐藏测试上的通过率差，
不是两套测试的题目数量差。若公开测试和隐藏测试并不作用于同一批候选，或者两套
测试的契约不同，就不能把这个差值直接解释成泛化下降，应该保存候选与测试切片
的联合结果后再分析。

这个 demo 的重点不是阈值本身，而是读者能看到不同证据如何互相补充：数学总体
准确率为 0.667，但过程平均分只有 0.667；代码 `pass@5` 高于 `pass@1`，却存在
公开/隐藏测试缺口；self-consistency 没有改善答案；同时有污染风险题。系统因此
输出具体动作，而不是把一组互不相同的测量压成一个总分。

## 13. 真实项目的评估闭环

### 13.1 离线运行

离线评估要准备公开 benchmark、私有 holdout、新鲜变体、历史 bad case 和任务
契约，先做污染和 parser 检查，再固定解码与 verifier 运行。原始输出、运行
日志、失败样本和版本 manifest 必须保存。

### 13.2 回归分析

模型、prompt、tokenizer、工具、sandbox、测试和 verifier 任一变化后，都要查看：

1. overall 和切片 accuracy；
2. 解析、编译、运行和超时错误；
3. hidden/public gap；
4. `pass@1` 与 `pass@k` 的变化；
5. 过程和 verifier 分歧；
6. 污染和题目新鲜度；
7. token、延迟和单位成功成本。

### 13.3 线上结果

benchmark 不能替代线上任务指标。代码助手还要看用户采纳、测试通过、回滚和
人工修改；数学或研究助手要看用户重试、引用和纠错；Agent 要看任务完成、
工具副作用、人工接管和恢复率。线上成功的统计单位可能是会话或用户，不能
直接与题目级 accuracy 混用。

### 13.4 失败闭环

一个可解释的闭环是：

~~~text
评估结果
  -> 错误样本与错误分类
  -> 判断是模型、数据、prompt、verifier、测试还是环境问题
  -> 修复并新增回归样本
  -> 在独立变体和 holdout 上复核
~~~

如果每次修复都只针对公开题面，系统可能越修越会背题，却没有提升泛化。

## 14. 常见误区

### 14.1 只看排行榜分数

公开 benchmark 可以提供历史对比，却可能被污染、饱和或过度调参。要结合私有、
时间切分、变体和线上证据。

### 14.2 把长 CoT 当作强 reasoning

长过程可能包含重复、错误跳步和事后编造。应评估关键 claim、最终状态和变体
泛化，而不是按 token 数奖励解释长度。

### 14.3 忽略答案解析

解析器可能把等价答案判错，也可能把错误输出最后一个数字误认为答案。parser
应有独立测试集和人工抽查。

### 14.4 公开测试太弱

代码可以 hard-code 样例，数学可以套模板，推理可以利用固定选项位置。隐藏
测试和参数化变体必须覆盖真实任务契约。

### 14.5 `pass@k` 不报告成本

`pass@10` 高不代表用户愿意等待十次生成和十次沙箱执行。报告候选数、延迟、
token、verifier 成本和成功任务成本。

### 14.6 用 LLM judge 替代执行验证

能用编译、单测、符号约束和状态断言时，应优先使用程序验证；judge 适合作为
开放过程的辅助信号，并要用人工 gold set 校准。

### 14.7 把污染风险当成模型结论

相似题、异常高分和扰动下降都只是风险证据。需要结合来源、时间、行为和人工
复核，才能决定样本是否移出最终报告。

## 15. 练习：从一个分数追到证据

### 练习一：数学答案归一化

给定 gold `2/3`，预测为 `0.6666667`、`\frac{2}{3}` 和 `66.7%`。说明在什么
数值容忍、单位和题目要求下可以判为等价，并指出 parser 可能的误判。

### 练习二：解释 `pass@1` 和 `pass@10`

一个代码模型 `pass@1` 很低而 `pass@10` 很高。分析单次稳定性、候选多样性、
verifier 搜索成本和真实产品是否允许十次尝试。

### 练习三：过程错误

设计一个数学步骤标注表，区分题意、公式、推导、计算、边界和答案格式错误，
并说明如何处理两条都正确但步骤不同的证明。

### 练习四：公开/隐藏测试差距

代码候选公开测试通过率 0.90，隐藏测试通过率 0.55。列出至少四种可能原因，
分别判断应修改测试、模型、提示、代码环境还是污染检测。

### 练习五：变体评估

设计一个保持难度的数学或代码题变体，写出原题、变体、能力契约、等价性检查、
污染风险和统计报告方式。

## 16. 资料与证据边界

论文和 benchmark 只在其公布的任务、数据、模型和 harness 条件下支持结论。
HumanEval 的代码通过率不能自动代表大型仓库修复能力，GSM8K 的数学结果不能
自动代表证明能力，Self-Consistency 的提升也依赖采样和答案抽取协议。

1. [HumanEval / Codex](https://arxiv.org/abs/2107.03374)：代码生成和单元测试评估。
2. [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)：数学解题与 verifier 的训练背景。
3. [MATH](https://arxiv.org/abs/2103.03874)：竞赛数学数据集与难度背景。
4. [Self-Consistency](https://arxiv.org/abs/2203.11171)：多条推理路径采样和一致性投票。
5. [APPS](https://arxiv.org/abs/2105.09938)：更开放的代码生成任务与测试评估。
6. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)：过程监督和结果验证的研究背景。
7. [RULER](https://arxiv.org/abs/2404.06654)：长上下文合成任务和能力退化评估。

使用这些资料时，要分开记录论文方法、数据集版本、运行 harness、解码预算、
测试强度、污染检查和项目实测。资料能说明评估设计的由来，不能替当前系统
承担未经验证的质量承诺。

## 17. 结语：让每个“正确”都有证据

Reasoning、Code、Math eval 的共同原则，是把最终答案、过程、verifier、测试
覆盖、泛化、污染和成本拆开。数学需要答案等价和过程边界，代码需要隐藏测试、
沙箱和行为验证，reasoning 需要多路径稳定性、过程 claim 和变体迁移。

`pass@k`、self-consistency 和 LLM verifier 都能扩大评估视野，却都会改变成本、
采样分布或误差结构。真正可比较的报告会保存完整协议，并把高分背后的候选数、
测试、题目新鲜度和失败样本讲清楚。

最终要回答的不是“排行榜上是多少分”，而是：模型在什么任务契约下成功，失败
发生在哪一步，改变题面后是否仍然成功，评估器是否可靠，以及得到这些证据
付出了多少计算和人工成本。
