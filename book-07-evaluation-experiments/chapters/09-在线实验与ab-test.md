# 第九章：在线实验与 A/B Test

离线评估可以回答“模型在一组固定样本上表现如何”，在线实验要回答的是
“真实用户使用整套产品时，结果是否变得更好”。这两个问题有关联，却不能互相
替代。模型在 benchmark 上提升，可能因为线上检索、工具、延迟、价格和界面
改变而没有带来价值；模型在离线质量上没有明显提升，也可能因为它减少了重试、
提高了代码采纳或降低了人工处理成本而更适合产品。

在线实验的对象也不是一个模型文件。一次用户请求可能经过路由、system prompt、
安全分类器、RAG、缓存、工具、流式输出、前端展示和人工审核。实验必须先确定
哪些组件属于 treatment，哪些保持不变，哪个用户或请求被随机到哪一组，以及
什么结果算一次曝光和一次成功。

本章从实验契约开始，依次解释随机单位、稳定分桶、曝光日志、主指标、护栏、
用户反馈、灰度、回滚和离线在线差异，再用一个可运行的 toy 实验展示：主指标
显著提升，并不意味着系统应该直接扩大流量。统计显著性、样本量和功效分析会
在下一章展开；这里保留上线实验必须理解的最小统计工具。

## 1. 在线实验测量的是端到端系统

### 1.1 小白视角：为什么换了模型却不一定换来价值

假设一个客服系统换了更强的模型。离线测试显示：

- 正确率从 0.82 提升到 0.85；
- 安全拒答没有明显变化；
- 公开测试的平均长度基本不变。

上线后却出现：

- 用户等待时间增加一倍；
- RAG 检索仍然返回错误文档；
- 新模型更喜欢长篇解释，用户更难找到结论；
- 代码块格式变化，前端无法渲染；
- 成功请求的成本大幅增加。

从用户角度看，产品可能没有变好，甚至变差。离线分数只覆盖了模型在固定
输入上的一部分行为，不能代替对完整链路的观察。

### 1.2 线上系统的组成

真实大模型产品通常包含：

1. 模型和 tokenizer；
2. system prompt 与路由策略；
3. RAG 检索、重排和引用；
4. 工具选择、参数校验和权限；
5. 安全分类器和人工审核；
6. 缓存、限流和重试；
7. 流式输出和前端展示；
8. 计费、配额和成本控制；
9. 日志、监控和反馈采集。

在线实验的 treatment 如果同时改变了多个组件，就需要把它们视为一个系统
版本，不能把效果归因给模型本身。若只想研究模型，就尽量固定其他链路；
若产品目标是整套新工作流，则应明确实验单位是端到端方案。

### 1.3 用户价值不是能力分数

能力、体验和业务价值可能朝不同方向变化：

| 变化 | 可能的用户结果 |
| --- | --- |
| 推理更强 | 复杂任务完成率上升，但延迟和成本上升 |
| 拒答更稳 | 风险降低，但正常敏感问题误拒增加 |
| 输出更长 | 解释更完整，也可能让结论更难找到 |
| 工具更积极 | 自动化成功上升，也可能增加副作用 |
| 缓存更激进 | 成本下降，也可能返回旧权限下的内容 |

因此在线实验需要同时测任务完成、用户行为、系统资源和安全风险。

## 2. 实验契约：在分流之前确定分母

### 2.1 样本 schema

一次可分析的曝光记录可以抽象为：

~~~math
e_i=(u_i,z_i,x_i,y_i,c_i,t_i)
~~~

其中：

- u_i 是随机单位，可以是用户、会话、请求、租户或设备；
- z_i 是 control 或 treatment 分组；
- x_i 是输入、场景和可用产品功能；
- y_i 是主指标观测值，如任务是否完成；
- c_i 是成本、延迟、错误、安全和人工处理等护栏结果；
- t_i 是曝光和结果发生的时间。

记录还应保存 model revision、prompt revision、实验配置、用户是否真正看到
输出、请求是否被重试、是否触发工具和是否被过滤。只有真正曝光给用户的
请求才能进入主要分析；后台 shadow 请求不能和真实用户结果混为一谈。

### 2.2 什么算一次曝光

不同产品的曝光定义不同：

1. 页面加载不代表用户看到了回答；
2. 请求到达模型不代表模型输出成功；
3. 输出生成不代表前端展示完整；
4. 展示回答不代表用户有机会使用工具；
5. 会话中的第二次重试不能无说明地当作新的独立用户。

实验开始前要定义 exposure、eligible、success、failure 和 missing。缺少
结果的请求也需要有状态，例如超时、用户离开、网络失败和人工接管，而不是
静默地从分母删除。

### 2.3 随机单位

随机单位的选择决定了独立性和用户体验：

| 单位 | 优点 | 风险 |
| --- | --- | --- |
| 请求 | 样本多、收敛快 | 同一会话切换版本，存在串扰 |
| 会话 | 体验较一致 | 跨会话学习和长期反馈被拆开 |
| 用户 | 多轮和长期行为一致 | 需要更多流量，用户间差异较大 |
| 租户 | 适合企业权限和协作 | 单位少，实验周期更长 |
| 地区 | 适合区域策略 | 地区差异可能与 treatment 混淆 |

聊天、多轮 Agent 和企业知识库通常更适合用户或租户级分桶；无状态的单次
分类请求可以考虑请求级分桶。无论选择什么单位，都要说明为什么。

### 2.4 假设和分析窗口

实验不是“先上流量再看数字”。开始前应写清：

1. treatment 改变了什么；
2. 预期改善的主指标；
3. 不能明显恶化的护栏；
4. 随机单位和流量比例；
5. 结果观察窗口；
6. 是否包含周末、节假日和高峰；
7. 缺失、重试和人工接管如何处理；
8. 实验何时结束以及谁负责解释。

这些约定不等于提前保证结果，而是防止看到结果后任意改变分母和目标。

## 3. A/B Test 的基本结构

### 3.1 Control 和 Treatment

Control 通常是当前线上版本，Treatment 是待比较的新模型、prompt、路由或
完整工作流。control 也可能需要版本冻结；如果对照组在实验中持续变化，
treatment effect 就不再对应一个稳定的比较对象。

### 3.2 稳定分桶

设随机单位为 u，实验标识为 e，哈希函数输出归一化到 [0,1)，treatment 比例
为 q，则可写成：

~~~math
z_i
=
I\left(h(u_i,e)<q\right)
~~~

分桶要依赖稳定的随机单位和 experiment id，而不是请求到达顺序、进程内随机数
或客户端临时状态。相同用户重试请求应回到同一组，除非实验契约明确使用
请求级随机。

不同实验要使用独立 experiment id 或 salt，避免多个实验共享相同的桶边界而
产生意外相关。分桶结果、版本和流量比例要写入曝光日志。

### 3.3 曝光日志和 Sample Ratio Mismatch

如果预期 control/treatment 比例为 1:1，实际曝光却是 60:40，不能直接比较
业务指标。设第 k 组观测数量为 o_k，按比例期望数量为 e_k，SRM 的卡方统计量为：

~~~math
X_{\mathrm{srm}}^2
=
\sum_k\frac{(o_k-e_k)^2}{e_k}
~~~

SRM 告警可能来自：

1. 分桶实现错误；
2. treatment 启动失败导致曝光丢失；
3. 前端或网关漏记日志；
4. 过滤条件只作用于一组；
5. 用户资格在分流后改变；
6. 重试或缓存重复计数。

SRM 不是“统计上不显著的小问题”，而是实验数据生成过程可能有缺陷的信号。
在原因没有解释前，业务差异不应被当作可靠 treatment effect。

### 3.4 实验隔离和干扰

多个实验同时运行时，一个用户可能同时进入模型、prompt、UI 和价格实验。
可以使用互斥实验层、正交设计、分层随机或明确记录交互项。

企业和协作产品还可能存在 interference：一个租户中的用户使用 treatment，
会影响同租户中 control 用户的文档、流程或协作行为。此时按请求随机会低估
或扭曲效果，需要按租户、团队或网络关系设计实验。

## 4. 主指标：把用户目标变成可观察结果

### 4.1 主指标的条件

一个好的主指标应：

1. 直接对应用户或业务目标；
2. 有稳定的分母；
3. 能在实验窗口内观察；
4. 不容易被简单投机；
5. 具有足够的信号和可解释性。

点赞率常常很容易采集，却未必直接代表任务完成；任务成功可能更接近价值，
但需要外部状态、人工标签或后续行为。

### 4.2 Chat 场景

可以考虑：

1. 任务完成率；
2. 有帮助回答率；
3. 用户在无需重复改写的情况下完成目标的比例；
4. 多轮会话的成功状态；
5. 人工转接率或重试率。

留存和会话长度可以作为长期结果，但它们受大量产品因素影响，不能自动归因
给单次模型变化。

### 4.3 Coding Assistant

代码助手可以观察：

1. 建议采纳率；
2. 采纳后代码通过测试的比例；
3. 用户编辑距离；
4. 从 issue 到可合并 patch 的时间；
5. 回滚、修复和人工 review 成本。

“用户接受了建议”不是最终质量；如果接受后很快被修改或导致回归，应该在
更长观察窗口中反映出来。

### 4.4 RAG 和 Agent

RAG 可观察答案采纳、引用点击、人工转接、重复提问和 claim-level faithfulness。
Agent 更适合看端到端任务成功、人工接管、工具失败、恢复率和外部状态。

工具调用次数通常不是越多越好。少调用可能是模型更高效，也可能是它漏掉了
必要步骤；必须结合任务成功和副作用判断。

### 4.5 数学抽象

均值型指标的 treatment effect 是：

~~~math
\hat{\tau}
=
\bar y_{\mathrm{treat}}
-\bar y_{\mathrm{ctrl}}
~~~

二分类成功率的差值为：

~~~math
\hat{\tau}
=
\hat p_{\mathrm{treat}}
-\hat p_{\mathrm{ctrl}}
~~~

如果 n_t 和 n_c 是两组样本数，s_t 和 s_c 是成功数，pooled rate 为：

~~~math
\hat p
=
\frac{s_t+s_c}{n_t+n_c}
~~~

大样本近似下，比例差的标准误可以写成：

~~~math
\mathrm{SE}(\hat{\tau})
=
\sqrt{
\hat p(1-\hat p)
\left(\frac{1}{n_t}+\frac{1}{n_c}\right)
}
~~~

对应的 z 值为：

~~~math
z
=
\frac{\hat{\tau}}{\mathrm{SE}(\hat{\tau})}
~~~

95% 近似区间为：

~~~math
\mathrm{CI}_{95}
=
\left[
\hat{\tau}-1.96\,\mathrm{SE},
\hat{\tau}+1.96\,\mathrm{SE}
\right]
~~~

下一章会讨论这些近似的条件、功效、样本量和多重检验。当前最重要的是不把
一个点估计当作确定事实。

## 5. 护栏指标：提升不能用风险换来

### 5.1 常见护栏

大模型实验的护栏通常包括：

1. P95 或 P99 延迟；
2. 首 token 时间和超时率；
3. 每请求成本和每成功任务成本；
4. 错误率、工具失败和回滚率；
5. unsafe output、privacy leakage 和 prompt injection；
6. over-refusal 和正常任务失败；
7. 人工审核量、用户举报和投诉；
8. 缓存命中、资源利用和配额耗尽。

### 5.2 用差值写清方向

对护栏 j：

~~~math
\Delta_j
=
m_{j,\mathrm{treat}}
-m_{j,\mathrm{ctrl}}
~~~

对于延迟、成本、错误率和安全违规，通常希望 Delta_j 不超过允许恶化
阈值 tau_j。对于成功率和安全完成率，方向相反，需要在报告中显式说明。

护栏还要记录置信区间和样本数。只因为当前点估计没有超过阈值，就宣布没有
风险，可能忽略了尾部不确定性。

### 5.3 硬约束与软信号

有些风险不适合用主指标抵消，例如未授权写入、明确隐私泄露和重大安全事故。
有些指标可以进入人工 review，例如少量延迟波动或某个低流量场景的负反馈。

硬约束与软信号的区别应由业务风险和恢复能力决定，而不是由指标名字决定。
报告可以同时给出：

~~~text
point_estimate：点估计
uncertainty：区间或误差
severity：失败严重度
action：调查、暂停、回滚或继续观察
~~~

### 5.4 成本和成功任务

只看每请求成本可能奖励一个经常失败但很便宜的系统。更有意义的指标是：

~~~math
C_{\mathrm{success}}
=
\frac{C_{\mathrm{generation}}+C_{\mathrm{verification}}}
{N_{\mathrm{successful\ tasks}}}
~~~

如果系统产生多个候选、工具调用或人工审核，分子应包含这些成本。成功任务
分母的定义必须和主指标相同，否则不同版本的成本不能直接比较。

## 6. 用户反馈：有信号，但不是事实标签

### 6.1 显式反馈

点赞、点踩、星级、举报和文本评论直接表达用户反应，但通常很稀疏。留下反馈
的用户可能是最满意或最不满意的一小部分，UI 位置、回答长度和新奇效应也会
影响点击。

### 6.2 隐式反馈

复制、引用点击、重新生成、继续追问、编辑、放弃和人工转接提供更多样本，但
同一个行为可能有多种解释。例如会话变长可能说明用户更投入，也可能说明
模型一直没有解决问题。

隐式反馈要与任务状态、答案质量和系统延迟结合，不能单独作为正负标签。

### 6.3 反馈偏差

常见偏差包括：

1. 高价值用户和普通用户反馈概率不同；
2. 长答案更容易获得“看起来完整”的评价；
3. 用户不知道事实是否正确；
4. 反馈按钮位置影响选择；
5. 新版本的短期新奇效应；
6. 失败后离开的用户没有留下反馈。

可以按用户、任务、语言、回答长度和结果状态分层，查看反馈是否在各组具有
相同含义。

### 6.4 反馈进入训练

线上反馈进入训练前要做隐私过滤、去重、质量标注、刷量检测和独立 holdout。
否则系统可能学会讨好反馈按钮、迎合高频用户或重复线上评估集，而不是提升
任务能力。

## 7. 流量切分、灰度和 shadow

### 7.1 用户级、请求级和租户级

用户级分桶能保持多轮体验一致，但需要更多样本；请求级分桶适合无状态任务，
却可能在同一会话中切换模型；租户级分桶适合权限和知识库绑定的企业产品，
但租户数量少、实验周期长。

选择分桶单位时，要同时考虑独立性、串扰、用户体验和结果观察窗口。

### 7.2 灰度不是统计结论

灰度的主要作用是控制风险和发现明显事故，不是替代完整 A/B 统计。一个 1%
流量的 canary 可以发现模型启动失败、成本异常和高危工具调用，却通常没有
足够样本判断小幅满意度差异。

常见放量顺序是内部用户、小流量、低风险场景、逐步扩大和全量，但每一步都
要保存版本、流量比例、主指标、护栏、异常和停止理由。

### 7.3 Shadow mode

Shadow 让新版本接收与线上相同的请求，但不把结果展示给用户。它适合比较：

1. 生成延迟和尾部；
2. token、GPU 和缓存成本；
3. 解析、工具和超时错误；
4. 离线 judge 或程序 verifier 结果；
5. 新旧输出的差异。

Shadow 不能测真实用户对新答案的反馈，也可能因为没有真实后续工具状态而
低估或高估端到端效果。

### 7.4 版本和路由

路由层必须能按用户、租户、地区和场景稳定地选择版本。缓存 key、重试和故障
降级不能把 control 和 treatment 的结果混在一起。实验曝光日志应记录最终
实际提供服务的版本，而不是只记录最初的配置。

## 8. 回滚和停止：先定义动作再扩大流量

### 8.1 触发信号

需要暂停或回滚的信号可能包括：

1. 安全违规或隐私泄露；
2. 高影响工具副作用；
3. P95/P99 延迟和超时明显上升；
4. 成本超过预算；
5. 错误、回滚或人工接管增加；
6. 关键任务成功下降；
7. 某个高风险群体出现严重退化；
8. SRM 或曝光日志持续异常。

触发条件需要区分自动回滚、人工 review 和继续观测。不能等实验结束后才决定
发生问题时怎么办。

### 8.2 可回滚对象

回滚不一定是整个模型：

1. 切回旧模型；
2. 切回旧 prompt 或路由；
3. 关闭有风险的工具；
4. 降级到安全模式；
5. 限制高风险场景；
6. 暂停 treatment 流量；
7. 清理或隔离错误缓存。

回滚操作本身也要可审计，尤其是已经产生外部副作用的 Agent。

### 8.3 回滚后的证据

回滚后要保存曝光、trace、版本、失败样本、外部状态和人工处置。判断问题
来自模型、prompt、检索、工具、流量、日志还是评估器后，才进入修复和复测。
回滚不是实验失败的终点，而是保留证据、恢复系统和改进实验设计的开始。

## 9. 离线与在线不一致

### 9.1 常见解释

离线提升而线上不提升，可能因为：

1. 离线样本不代表真实用户分布；
2. 主指标没有对应真实目标；
3. 延迟、成本或格式抵消了质量提升；
4. RAG、工具或权限成为瓶颈；
5. 新模型改变了拒答、长度或风格；
6. 实验流量不足或观察窗口过短；
7. 分桶、曝光或结果日志有 bug；
8. treatment 与其他线上实验发生干扰。

线上提升而离线不提升，也可能说明离线指标没有测到真实用户价值，或者用户
行为本身改变了任务完成方式。不要先选择一个更符合预期的结果。

### 9.2 排查顺序

可以按以下顺序建立证据链：

1. 检查实验配置、分桶、曝光和版本；
2. 检查 SRM、缺失和重复计数；
3. 对比 control/treatment 的用户、任务和语言分布；
4. 查看主指标、护栏和分层结果；
5. 抽样做人审和 claim-level 检查；
6. 检查延迟、成本、检索、工具和错误；
7. 对比线上输入与离线数据的长度、难度和风险；
8. 将真实 bad case 加入独立离线回归集。

### 9.3 反哺离线评估

线上失败样本应经过脱敏、归类、去重和污染检查，再进入离线评估。不能把
所有用户反馈直接变成训练样本，也不能把已经用于调优的样本继续当作无偏
holdout。

## 10. 实验中的统计和工程陷阱

### 10.1 样本量和功效

样本太少时，点估计波动很大；样本很多时，极小的无业务意义差异也可能显著。
需要同时考虑最小实际效果、基线率、方差、样本量、实验周期和成本。

### 10.2 提前停止

反复查看结果并在偶然变好时停止，会增加假阳性。实验要预先定义观察窗口、
允许的中途检查方法和停止规则。真正的事故信号可以立即停止，但不能把安全
停止和追求统计显著混为一谈。

### 10.3 多指标和多切片

看了很多主指标、用户群和时间段后，总会找到一个变好的结果。报告应区分
预注册的主要分析、探索性分析和后续假设，并说明多重比较和独立复验。

### 10.4 新奇效应

新模型或新 UI 可能短期提高活跃和反馈，长期效果却回落。实验窗口应覆盖
新奇效应消退的时间，或者至少把短期和长期结果分开。

### 10.5 Logging bug

曝光缺失、事件延迟、重复计数、客户端版本不一致、缓存命中未记录和错误重试
都可能污染结论。实验平台本身需要用已知流量和合成结果做审计。

### 10.6 实验前协变量：降低噪声，不改变因果对象

在线实验中，随机化的目的不是让两组用户在每一个属性上完全相同，而是让这些
属性在平均意义上不再与 treatment 相关。即使随机化正确，某次实验仍可能因为
样本量有限而出现“实验组恰好有更多重度用户”的情况。两组结果的差异此时会有
较大的偶然波动。

如果在实验开始之前已经知道用户过去的任务完成率、历史延迟、账户规模或上一
个周期的活跃度，就可以把这些变量作为协变量来降低方差。关键限制是：协变量
必须在 treatment 生效之前确定。实验产生的点击、重新提问或工具调用不能拿来
做这种调整，因为它们已经可能受到 treatment 影响。

设 Y_i 是实验期间的结果，X_i 是实验前测量的、和 Y_i 相关的协变量。一个常见
的调整形式是：

~~~math
Y_i^{\mathrm{adj}}
=
Y_i-\theta(X_i-\bar X),
\qquad
\theta
=
\frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)}
~~~

其中 \bar X 是全体分析单位的协变量均值。直观地说，调整先扣除“这个用户本来
就会产生的结果”，再比较两组剩余的变化。因为随机分组与 X 独立，正确使用
实验前协变量不会把 treatment effect 改成另一个问题；它主要减少估计量的噪声。

在理想的线性情形下，若 X 和 Y 的相关系数为 rho，调整后的方差大致按
`1-rho^2` 缩小。历史行为越能预测本次结果，方差降低越明显。例如，代码助手
的实验可以使用用户实验前一周的测试通过率作为 X；但不能使用实验期间“是否
采纳新模型建议”作为 X。后者是 treatment 之后的变量，会把真正的效果从分析
中扣掉。

实际使用时还要保存协变量的生成时间、版本和缺失规则，并在实验报告中同时给出
未调整结果和调整结果。调整模型的参数不能只在 treatment 组上拟合，也不应因为
调整后数字更好看就隐藏原始分析。协变量方法降低方差的前提仍然是随机化、曝光
和结果定义没有被破坏，它不能修复错误分桶或漏记日志。

### 10.7 比例指标：分子和分母必须一起解释

“每个请求平均花费多少 token”“每次成功任务平均成本多少”这类指标经常是比例
指标。它们的难点在于：分子和分母可能同时受到 treatment 影响，而且一个用户
可以贡献数量极不相等的请求。如果直接把所有请求拼在一起，重度用户会获得更大
权重；如果先算每个用户的比例再取平均，轻度用户又会和重度用户拥有相同权重。
两种做法回答的是不同问题。

对一组用户 g，整体比例可以写成：

~~~math
R_g
=
\frac{\sum_{i\in g} A_i}{\sum_{i\in g} B_i},
\qquad B_i>0
~~~

例如，若 A_i 是成功任务数、B_i 是合资格任务数，R_g 回答“这组全部合资格任务
中有多少成功”；若先计算每个用户的成功率再平均，回答的则是“一个典型用户的
成功率”。产品目标决定应选哪一个，但报告必须把定义写出来。

成本指标尤其容易被误读。新模型可能让每次回答更贵，却减少了用户重试；也可能
让单次生成更便宜，却需要更多工具调用和人工接管。应分别展示：

1. 每请求成本；
2. 每用户或每会话成本；
3. 每成功任务成本；
4. 成功任务率和失败原因。

如果用成功任务成本作为主结果，分母里的“成功”必须先定义，并采用固定的结果
观察窗口。不能在 treatment 失败更多时把失败请求从成本分母中删掉，再得出它
更便宜的结论。

重试和重复曝光也会改变比例。一次用户请求由于超时被重新发送时，需要记录原始
请求与重试的关联，并决定成本算一次任务还是算两次生成。只看生成服务日志会
高估请求量，只看前端点击又可能漏掉后台失败。比例指标的分析表应同时保留
分子、分母、去重键和聚合层级，以便复算。

### 10.8 延迟结果和观察窗口

很多大模型任务不是在响应结束时就知道结果。代码建议可能要等用户运行测试，
企业流程可能要等人工审批，Agent 的工具写入可能在数小时后才产生可验证状态。
如果实验在这些结果成熟之前结束，最早进入实验的用户有完整结果，最后进入的
用户却只有“尚未观察到”的状态。把后者当失败会低估成功率，把后者直接删除又
会造成选择偏差。

可以为每个分析单位记录结果成熟时间，并用固定窗口定义结果：

~~~math
Y_i(w)
=
I\left(\text{任务在曝光后的 }w\text{ 时间内完成}\right)
~~~

分析截止日为 T 时，只有满足 `exposure_time_i + w <= T` 的单位才拥有完整的
w 窗口。其余单位应标记为 pending，而不是悄悄进入失败分母。报告可以同时展示
短窗口结果、成熟队列结果和删失比例。

窗口长度不能只凭方便选择。短窗口更快，却可能漏掉长期修复或副作用；长窗口
更接近真实价值，却会延迟决策，并受到用户流失、权限变化和其他版本的影响。
例如代码助手可以在响应结束时记录“建议被采纳”，在七天窗口记录“采纳后测试
通过”，在更长窗口记录“是否回滚”。这些指标不是同一个结果，应分别命名。

还要注意 treatment 可能改变结果发生的速度。若新 Agent 更快完成任务，但总的
七天成功率相同，只看最终成功率会漏掉体验改善；若它先给出看似完成的结果、后续
修复成本更高，只看短窗口又会得出相反的结论。因此结果窗口与延迟指标应放在同
一份实验报告中，并标出数据成熟状态。

### 10.9 聚类、重复观测与长期结果

随机化单位和分析单位必须一致。若按用户分桶，却把一个用户的每条请求当作完全
独立样本，标准误会通常过小，因为同一用户的任务、设备、订阅状态和使用习惯
存在相关性。企业产品按租户分桶时，租户内部的相关性更强，一个大型租户甚至
可能决定整体均值的方向。

用每个随机单位的结果先聚合，再比较两组，通常比把请求当成独立样本更容易解释。
也可以使用按用户或租户聚类的标准误、cluster bootstrap，或在模型中显式建模
层级。一个帮助理解的近似是：每个单位平均有 m 条观测、组内相关系数为 rho 时，
有效样本量大致为：

~~~math
n_{\mathrm{eff}}
\approx
\frac{n}{1+(m-1)\rho}
~~~

当 rho 接近 0 时，多条观测确实提供更多信息；当 rho 较大时，继续增加同一用户
的请求并不能等价替代更多独立用户。这个公式不是所有实验的最终估计方法，但能
提醒我们：日志行数不等于独立证据量。

长期指标还需要避免“实验结束即停止记录”的误区。可以保留一小部分稳定用户作
长期 holdout，观察留存、复购、人工成本、回滚和投诉；也可以按时间段切换版本，
但时间段切换必须考虑周几、节假日、流量趋势和外部事件。若版本影响了共享缓存、
知识库或团队协作，一个用户的结果还可能改变另一个用户的结果，此时简单的用户
级 A/B 分流并不再满足无干扰假设。

因此，实验报告至少要写清三件事：随机化单位是什么，结果在哪一层聚合，长期
结果是否已经成熟。若这三者没有对齐，显著性计算再精细，也只是对错误数据结构
进行了精确计算。

## 11. 一个可运行的在线实验诊断 demo

下面的 demo 使用 Python 标准库模拟一次模型上线实验。control 和 treatment 的
任务成功率有差异，但 treatment 的延迟、成功任务成本、安全率和误拒率超出
示例护栏。代码不依赖真实用户数据，阈值仅用于说明分析过程。

~~~python
import hashlib
import math


def stable_bucket(user_id, experiment_id):
    key = f"{experiment_id}:{user_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % 100


def assign(user_id, experiment_id="exp_model_v2", treatment_pct=50):
    bucket = stable_bucket(user_id, experiment_id)
    return "treatment" if bucket < treatment_pct else "control"


def rate(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def p95(values):
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[index]


def two_prop_z(success_t, n_t, success_c, n_c):
    p_t = success_t / n_t
    p_c = success_c / n_c
    delta = p_t - p_c
    pooled = (success_t + success_c) / (n_t + n_c)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    z = 0.0 if se == 0 else delta / se
    ci = (delta - 1.96 * se, delta + 1.96 * se)
    return p_t, p_c, delta, se, z, ci


def srm_chi_square(observed, expected_ratio):
    total = sum(observed.values())
    chi = 0.0
    for group, ratio in expected_ratio.items():
        expected = total * ratio
        chi += (observed[group] - expected) ** 2 / expected
    return chi


bucket_preview = {
    user: assign(user)
    for user in ["u001", "u006", "u007", "u008", "u010"]
}

control_latencies = [620, 700, 740, 810, 850, 910, 980, 1050, 1220, 1350]
treatment_latencies = [760, 820, 910, 1040, 1180, 1320, 1490, 1680, 1880, 2100]

experiment = {
    "control": {
        "users": 1000,
        "success": 520,
        "unsafe": 2,
        "over_refusal": 35,
        "errors": 9,
        "cost_usd": 120.0,
        "latencies": control_latencies,
    },
    "treatment": {
        "users": 1000,
        "success": 570,
        "unsafe": 6,
        "over_refusal": 70,
        "errors": 11,
        "cost_usd": 180.0,
        "latencies": treatment_latencies,
    },
}

p_t, p_c, delta, se, z, ci = two_prop_z(
    experiment["treatment"]["success"],
    experiment["treatment"]["users"],
    experiment["control"]["success"],
    experiment["control"]["users"],
)

observed = {group: data["users"] for group, data in experiment.items()}
srm_chi = srm_chi_square(observed, {"control": 0.5, "treatment": 0.5})

summary = {}
for group, data in experiment.items():
    users = data["users"]
    summary[group] = {
        "success_rate": round(rate(data["success"], users), 3),
        "unsafe_rate": round(rate(data["unsafe"], users), 3),
        "over_refusal_rate": round(rate(data["over_refusal"], users), 3),
        "error_rate": round(rate(data["errors"], users), 3),
        "cost_per_success": round(data["cost_usd"] / data["success"], 3),
        "p95_latency_ms": p95(data["latencies"]),
    }

primary = {
    "control_rate": round(p_c, 3),
    "treatment_rate": round(p_t, 3),
    "delta": round(delta, 3),
    "se": round(se, 4),
    "z": round(z, 3),
    "ci95": (round(ci[0], 3), round(ci[1], 3)),
}

signals = {
    "srm_valid": srm_chi < 10.83,
    "primary_supported": primary["delta"] > 0 and abs(primary["z"]) >= 1.96,
    "latency_within_budget": (
        summary["treatment"]["p95_latency_ms"]
        <= summary["control"]["p95_latency_ms"] * 1.25
    ),
    "cost_within_budget": (
        summary["treatment"]["cost_per_success"]
        <= summary["control"]["cost_per_success"] * 1.30
    ),
    "unsafe_within_budget": (
        summary["treatment"]["unsafe_rate"]
        <= summary["control"]["unsafe_rate"] + 0.002
    ),
    "over_refusal_within_budget": (
        summary["treatment"]["over_refusal_rate"]
        <= summary["control"]["over_refusal_rate"] + 0.02
    ),
    "error_within_budget": (
        summary["treatment"]["error_rate"]
        <= summary["control"]["error_rate"] + 0.005
    ),
}

actions = []
if not signals["srm_valid"]:
    actions.append("debug_exposure_and_assignment_logs")
if not signals["primary_supported"]:
    actions.append("collect_more_samples_or_revisit_primary_metric")
if not signals["latency_within_budget"]:
    actions.append("reduce_tail_latency_or_route_selectively")
if not signals["cost_within_budget"]:
    actions.append("optimize_cost_per_successful_task")
if not signals["unsafe_within_budget"]:
    actions.append("review_safety_regression_before_more_traffic")
if not signals["over_refusal_within_budget"]:
    actions.append("review_benign_boundary_and_refusal_behavior")
if not signals["error_within_budget"]:
    actions.append("debug_runtime_and_timeout_regressions")

decision = "continue_with_sliced_report" if not actions else "hold_for_guardrail_review"

print("bucket_preview=", bucket_preview, sep="")
print("summary=", summary, sep="")
print("primary=", primary, sep="")
print("srm_chi_square=", round(srm_chi, 3), sep="")
print("signals=", signals, sep="")
print("actions=", actions, sep="")
print("decision=", decision, sep="")
~~~

示例输出为：

~~~text
bucket_preview={'u001': 'control', 'u006': 'treatment', 'u007': 'treatment', 'u008': 'control', 'u010': 'treatment'}
summary={'control': {'success_rate': 0.52, 'unsafe_rate': 0.002, 'over_refusal_rate': 0.035, 'error_rate': 0.009, 'cost_per_success': 0.231, 'p95_latency_ms': 1350}, 'treatment': {'success_rate': 0.57, 'unsafe_rate': 0.006, 'over_refusal_rate': 0.07, 'error_rate': 0.011, 'cost_per_success': 0.316, 'p95_latency_ms': 2100}}
primary={'control_rate': 0.52, 'treatment_rate': 0.57, 'delta': 0.05, 'se': 0.0223, 'z': 2.245, 'ci95': (0.006, 0.094)}
srm_chi_square=0.0
signals={'srm_valid': True, 'primary_supported': True, 'latency_within_budget': False, 'cost_within_budget': False, 'unsafe_within_budget': False, 'over_refusal_within_budget': False, 'error_within_budget': True}
actions=['reduce_tail_latency_or_route_selectively', 'optimize_cost_per_successful_task', 'review_safety_regression_before_more_traffic', 'review_benign_boundary_and_refusal_behavior']
decision=hold_for_guardrail_review
~~~

这个结果的含义不是 treatment “失败”或“成功”二选一。任务成功率有提升，
且在这个近似计算下区间没有跨过 0；但延迟、每成功任务成本、安全率和误拒率
都恶化，应该进入护栏复核、分层分析、路由或修复，而不是直接扩大流量。

### 11.1 代码中的统计边界

demo 的 p95 使用十个 toy 延迟值中的经验分位点，不能替代生产 histogram 或
更大窗口的分位数估计。两比例 z 检验依赖大样本近似，真实低基线安全事件、
强聚类用户数据和连续监测需要更合适的区间或分层方法。

SRM 为 0 只说明这组示例的用户数量正好按 1:1 分配，不代表曝光日志、用户资格、
结果事件和 treatment 实际生效都没有问题。

## 12. 真实项目的在线实验闭环

### 12.1 从实验假设开始

实验假设应包含：

1. treatment 改变了哪些组件；
2. 哪类用户或任务会受影响；
3. 预期改善的主指标；
4. 可能恶化的护栏；
5. 观察窗口和随机单位；
6. 最小有意义效果；
7. 结果不确定时的下一步。

例如，“新模型更强”不是可检验假设；“在企业知识库问答中，新模型提高带
正确引用的任务完成率，同时 P95 延迟增加不超过 25%、安全违规不增加”更接近
可执行的实验陈述。

### 12.2 数据集与线上抽样

线上样本可能包含隐私、权限和高风险内容。抽样做人审或写入离线集前要脱敏、
授权、去重和记录数据血缘。反馈、日志和失败案例不能直接同时用于调参和最终
评估。

### 12.3 运行清单

每次实验至少保存：

~~~text
experiment_id
control_revision
treatment_revision
prompt_and_router_revision
randomization_unit
assignment_salt
eligibility_rule
exposure_definition
primary_metric_definition
guardrail_definitions
analysis_window
sample_ratio_expectation
stopping_and_rollback_rules
privacy_and_log_policy
~~~

如果实验同时涉及 RAG、工具和安全分类器，还要保存它们的版本和权限配置。

### 12.4 结果报告

报告先给实验是否有效，再给业务结果：

1. SRM 和曝光完整性；
2. control/treatment 样本数；
3. 主指标点估计、区间和实际意义；
4. 护栏点估计、区间和严重度；
5. 用户、任务、语言、地区和风险切片；
6. 延迟、成本、错误和人工量；
7. 失败样本和评估器分歧；
8. 继续、暂停、路由、修复或回滚的理由。

不要用一个“显著/不显著”标签替代整份报告。

## 13. 常见误区

### 13.1 把 A/B Test 当成上线仪式

A/B Test 的价值是识别真实净收益和副作用，不是按流程盖章。若实验假设、
分母和护栏没有写清，分流本身不会自动产生因果证据。

### 13.2 只看主指标

主指标提升但安全、隐私、延迟、成本或关键群体明显退化，整体体验可能更差。
护栏和分层结果是结论的一部分。

### 13.3 请求级随机破坏多轮体验

同一用户在一个会话中频繁切模型，会改变用户行为和后续输入。多轮产品通常
应优先考虑用户或会话级稳定分桶。

### 13.4 不记录真实曝光

配置中的 treatment 比例不代表用户实际看到的版本。必须记录资格、分桶、曝光、
生成、展示和结果的状态。

### 13.5 用点赞替代成功

用户可能无法判断事实正确性，也可能因为 UI、长度或新奇效应改变点赞。任务
完成、外部状态、人工抽检和长期行为应共同使用。

### 13.6 中途挑最好看的切片

多个指标、多个用户群和多个时间窗会产生选择空间。预先定义主要分析，探索性
发现要标明并在新数据上复验。

### 13.7 线上 bad case 不回流

没有把真实失败加入脱敏回归集，下一版本仍可能重复同一问题。回流前要防止
训练污染和评估集过拟合。

## 14. 练习：从假设到实验报告

### 练习一：RAG 问答

为企业知识库设计一次 A/B Test，写出 treatment、随机单位、主指标、引用
护栏、权限护栏、人工抽样和回滚条件。

### 练习二：质量与成本冲突

新模型让任务完成率提升 5%，但 P95 延迟增加 40%，每成功任务成本增加 60%。
请按用户群和任务类型设计分层分析，并说明何时选择路由而不是全量发布。

### 练习三：会话变长

实验组平均会话长度增加。列出至少三种相反解释，并设计能够区分它们的事件
和结果指标。

### 练习四：代码助手

设计代码助手的主指标和护栏，至少包括建议采纳、测试通过、编辑距离、回滚、
延迟、成本和安全。

### 练习五：SRM 诊断

实验计划按 50:50 分流，但曝光日志显示 62:38。列出需要检查的分桶、资格、
前端、缓存、重试和日志路径，并说明为什么在修复前不能比较业务指标。

## 15. 资料与证据边界

在线实验方法依赖随机化单位、曝光定义、分析窗口、指标分母和运行环境。论文
或平台文档中的结果不能自动代表大模型产品；尤其是用户反馈、长会话、工具
副作用、隐私和安全指标，都可能与传统网页实验不同。

可作为方法入口的资料包括：

1. [Controlled Experiments on the Web](https://ai.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf)：大规模网页在线实验的基础经验。
2. [Trustworthy Online Controlled Experiments](https://experimentguide.com/)：在线受控实验的实践资料。
3. [Sample Ratio Mismatch Detection at Scale](https://arxiv.org/abs/2208.07766)：大规模实验中随机化校验和分流比例异常的诊断方法。
4. [CUPED](https://arxiv.org/abs/1512.04922)：利用实验前数据降低方差的实验方法。
5. [Online Controlled Experiments: Lessons from Running A/B/n Tests for 12 Years](https://doi.org/10.1145/2783258.2785464)：大规模在线实验基础设施和长期运行经验。
6. [Interference in Experiments](https://arxiv.org/abs/1706.07741)：网络和用户间干扰对随机实验的影响。
7. [Sequential Testing](https://arxiv.org/abs/2102.12352)：连续观测和停止规则的统计背景。

这些资料能说明实验设计、方差降低、分流诊断和干扰问题的来源；当前项目仍
需要根据用户单位、产品风险、隐私要求和实际流量重新验证。下一章会进一步
解释统计显著性、功效、样本量和多重检验的边界。

## 16. 结语：验证真实净收益

在线实验把模型能力放回真实用户、真实系统和真实成本中观察。它要求实验者
先写清 treatment、control、随机单位、曝光和成功状态，再选择主指标和护栏；
要求分桶稳定、日志完整、实验隔离，并在灰度、shadow 和全量之间保持版本可追溯。

对于大模型，任务成功不能脱离延迟、成本、拒答、安全、隐私、工具副作用和
用户反馈。一个主指标的提升只有在这些条件下仍然成立，才可能代表产品净收益。
即使结果显著，也要问提升来自哪个用户群、哪类任务、哪个时间窗，以及是否
牺牲了另一个更重要的结果。

好的 A/B Test 不是宣布哪个模型“更强”，而是用可复核的随机化、结果和失败
证据回答：新版本在什么真实场景中更有价值，代价是什么，风险是否可接受，
下一步应该全量、路由、修复、继续观察还是回滚。
