# 第 10 章（3）：灰度发布、护栏与实验报告

统计计算给出的是一段流量、一个版本和一段时间里的估计。若实验日志不完整、用户被错误分组，或高风险事件没有明确的暂停规则，再精确的区间也无法变成可靠的发布决定。这一篇把指标接到日志、成本与安全底线、分阶段灰度、分组分析和复核报告。

本篇承接[线上 A/B 实验设计](10-agent轨迹与线上评估实战-2-ab实验设计.md)：即便随机化和统计计算都正确，仍要确认分组、曝光、日志和用户状态符合协议，再依据预先写下的护栏决定是否扩量。

## 实验运行后，怎样把数据变成发布决定

### 10.2.7 日志字段决定实验能否解释

至少保存以下字段：

```text
request_id
pseudonymous_randomization_unit_id
randomization_unit_type (request / session / user / tenant)
experiment_name
eligibility, assignment_time and bucket
first_exposure_time and actual version
model_revision
prompt_revision
retriever_revision
tool_revision
tokenizer and decoding parameters
input and output token counts
retrieved document IDs and versions
tool requests, returns and retries
latency breakdown
cost estimate
error code and timeout
user feedback and later correction
timestamp and locale
```

关联 ID 应使用受控的伪名，身份映射留在隔离的业务系统中。主分析按随机分配单元保留分母（intention-to-treat，按分配分析）；实际曝光另记事件，用于解释未曝光或路由异常，不能在看过 Treatment 结果后只删除未按预期曝光的样本。主指标和曝光规则均应在实验开始前确定。

如果 prompt、检索器和模型同时变化，只记录 model_revision，就无法解释结果。缓存 key 也必须包含会改变输出的实验条件，否则 Control 可能命中 Treatment 产生的缓存。

### 10.2.8 大模型实验的特殊变量

#### 输出随机性

固定或记录 temperature、top_p、max_tokens、seed、停止条件和并行采样策略。不同版本即使使用同一 seed，也不一定产生相同输出，因为 logits、tokenizer 或后端 kernel 可能变化。

#### 多轮状态

会话级分桶后，整个历史仍需保持一致。若摘要器、记忆服务或工具版本独立变化，模型版本的因果解释仍会被污染。

#### 成本和延迟

更长的回答、更多的 reasoning token、更多工具轮次和更低缓存命中率都会提高成本。应分开看首 token 延迟、生成间隔、完整响应时间、队列时间、检索时间和工具时间，而不是只看平均总延迟。

#### 低频高后果事件

安全违规、隐私泄露、错误退款和错误删除可能在小流量里没有发生，但不能据此证明风险为零。上线前应结合离线红队、定向高风险样本、线上分类器、人工抽检和举报通道。

### 10.2.9 灰度、暂停和回退

一个可操作的灰度序列可以是：

```text
离线回归与安全测试
内部用户或低风险场景
1% 受控流量
5% 低风险全场景
10%--25% 分层扩量
更大比例对照
达到预先定义的收益与风险条件后逐步扩大
```

每个阶段都应记录进入下一阶段的条件和观察窗口。回退动作要可执行：切回旧模型、按渠道或场景关闭新版本、暂停外部工具、降低输出预算、切换保守提示词、增加人工复核或冻结高风险动作。

触发回退的不应只有一个平均指标。可以包括安全事件、错误率连续异常、P95 延迟、成本突增、关键用户群退化、集中投诉和工具副作用。回退后还要保留实验日志和失败样本，避免只把流量切回旧版本却丢失根因。

### 10.2.10 分组分析：总体提升可能掩盖局部退化

至少按以下维度分组统计：

```text
任务类型与业务渠道
query 长度、上下文长度和语言
新用户与老用户
低风险与高风险
有证据与无证据
工具成功与工具失败
低延迟与高延迟
不同模型路由和缓存状态
```

如果总体问题解决率提升 2%，但高价值客户下降 3%，总体平均不能直接代表产品体验。对样本较少的组要同时报告样本量和区间，避免把几个异常样本过度解读；对后果严重的明确退化，即使样本不大也应进入人工复核。

### 10.2.11 结果报告应包含什么

一份可复查的报告可以按以下顺序组织：

```text
实验范围、随机化单元和运行时间
Control / Treatment 的流量与 SRM 检查
主指标：点估计、置信区间、样本量
护栏：安全、事实、格式、延迟、成本、错误率
诊断：长度、工具、检索、缓存和任务类别
人工抽检：样本来源、盲评结果和严重错误
异常：日志缺失、协议变化、缓存污染或业务活动
决策：扩大、保持、暂停、回退或定向路由
后续：待修复问题和下一轮实验条件
```

例如，“Treatment 问题解决率提升 2.3%，95% 区间为 1.1%--3.5%；P95 延迟增加 8%，成本增加 12%；高风险任务没有明确改善，长问题提升明显；人工抽检发现多轮对话仍有少量过度解释，因此扩大到 25% 并继续观察”比“B 比 A 好”更接近可执行的工程结论。示例报告中的“扩大”仍须满足预先定义的高风险底线，不能仅凭总体区间作决定。

### 10.2.12 资料与边界

[Trustworthy Online Controlled Experiments: Five Puzzling Outcomes Explained](https://exp-platform.com/puzzling-outcomes/)讨论了线上实验中容易误读的结果和可信度问题；[Microsoft SRM 排查说明](https://www.microsoft.com/en-us/research/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/)提供了比例异常的诊断入口；[NIST 比例置信区间说明](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)解释了 Wilson 和精确区间。CUPED 的原始论文 [Improving the sensitivity of online controlled experiments by utilizing pre-experiment data](https://exp-platform.com/cuped/)展示了如何用实验前协变量降低方差；连续监测可参考 [Always Valid Inference](https://arxiv.org/abs/1512.04922) 中与停止过程匹配的推断方法。不应把一段 Python demo 当作完整统计软件。

A/B 实验的证据范围始终有限：它回答的是某个版本、某组流量、某段时间和某套系统条件下的差异。模型、提示词、检索器、工具、用户群和业务活动改变后，结论需要重新验证。连续查看结果时，应使用与停止规则相匹配的序贯推断方法，不能把固定样本检验反复套用。

## 本章练习与面试迁移

1. 用[第一篇的订单地址轨迹](10-agent轨迹与线上评估实战-1-轨迹诊断.md)分别标注模型提议、guard 判定、工具执行和状态变化；指出哪一处是首个不可恢复失败，并计算 20 条任务中的提前提议率、拦截率和真实副作用率。

2. 按[第二篇的样例数据](10-agent轨迹与线上评估实战-2-ab实验设计.md)手算 Control 与 Treatment 的解决率差、95% 区间、P95 延迟和成本变化；说明为何即使总体收益过线，6/7 个安全事件仍使示例决策保持暂停。

3. 将 50:50 分流的 12,000/12,080 个会话代入[第二篇的 SRM 例子](10-agent轨迹与线上评估实战-2-ab实验设计.md)，计算 SRM 统计量并列出应优先检查的分桶、曝光和日志环节。

回答“新模型离线提升了，为什么不扩量”时，先明确评估版本和随机化单元，再报告区间、SRM、成本延迟与安全证据。样本数不足以证明安全时，应明确保留风险，不把“未观察到事故”写成“系统安全”。

从一条轨迹到线上对照实验，交付物应能还原任务状态、随机化单元、版本、时间窗、样本分母、指标区间、异常和决策理由。[第十五章作品集实战](15-作品集实战.md)会把这些实验记录连同前面章节的实现与失败证据整理成可复核的作品集。
