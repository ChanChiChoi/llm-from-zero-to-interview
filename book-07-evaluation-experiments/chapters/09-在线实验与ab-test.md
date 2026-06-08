# 第九章：在线实验与 A/B Test

重点：线上指标、流量切分、用户反馈、护栏指标、灰度发布、回滚策略。

面试重点：线上效果不一定等于离线 benchmark 分数。

## 本章资料边界

本章第二轮精修前，按 `WRITING_PLAN.md` 联网核对了 Kohavi 等关于 web controlled experiments 和 large-scale online experiments 的论文、Microsoft / LinkedIn 关于实验基础设施、sample ratio mismatch 和 CUPED 的公开资料，以及前序评估、安全和部署章节的指标口径。

本章聚焦大模型产品上线实验的可迁移方法：随机单位、稳定分桶、曝光日志、主指标、护栏指标、灰度、回滚、用户反馈偏差、离线与在线不一致排查和实验报告。

统计显著性、样本量、方差、功效分析和多重检验会在下一章更系统展开；本章只保留上线实验设计必须掌握的最小公式和可运行审计 demo。

## 本章目标

学完本章，你要能回答：

1. 为什么离线 benchmark 分数高，不代表线上一定更好？
2. 大模型产品常见线上指标有哪些？
3. A/B Test 的基本流程是什么？
4. 如何做流量切分、用户分桶和实验隔离？
5. 什么是护栏指标，为什么大模型上线尤其需要护栏？
6. 如何处理用户反馈、灰度发布和回滚？
7. 面试中如何设计一个模型上线实验？

离线评估回答的是：“模型在固定测试集上表现如何？”

在线实验回答的是：“真实用户使用时，产品目标有没有变好，风险有没有变大？”

这两个问题相关，但不是同一个问题。

一个模型在 benchmark 上更强，线上可能没有提升，甚至变差。

原因可能是：

1. Benchmark 不覆盖真实用户任务。
2. 线上 prompt、工具、RAG、系统延迟和成本不同。
3. 用户对回答长度、风格、速度、稳定性更敏感。
4. 新模型可能更安全但更容易拒答。
5. 新模型可能更聪明但更慢、更贵。
6. 离线样本没有覆盖线上分布和边界情况。

面试表达：离线 eval 是上线前筛选，在线 A/B test 是验证真实用户价值；两者要结合，不能互相替代。

## 1. 为什么需要在线实验

大模型系统不是单个模型文件。

线上产品通常包含：

1. 模型。
2. System prompt。
3. RAG 检索。
4. Tool use。
5. 安全分类器。
6. 缓存。
7. 路由策略。
8. UI 交互。
9. 价格和限流策略。
10. 监控和人工审核。

离线 benchmark 往往只评估其中一部分。

在线实验评估的是端到端系统。

### 1.1 真实用户分布不同

Benchmark 通常是干净、结构化、任务明确的样本。

真实用户输入可能是：

1. 模糊问题。
2. 拼写错误。
3. 多轮上下文。
4. 半截需求。
5. 含有隐私信息。
6. 带有工具依赖。
7. 同时要求准确、快速、便宜。

### 1.2 用户价值不等于模型能力

模型能力提升不一定转成产品价值。

例如：

1. 回答更长，但用户只想要结论。
2. 推理更强，但延迟太高。
3. 安全性更强，但正常问题被拒答。
4. 代码能力更强，但格式不符合产品工作流。

### 1.3 系统链路会放大问题

RAG 检索错了，模型再强也可能回答错。

工具调用权限错了，模型再安全也可能造成风险。

前端展示不清楚，用户也可能觉得体验差。

面试表达：在线实验的价值是评估真实系统和真实用户，而不是只评估模型本身。

## 2. A/B Test 基本概念

A/B Test 是把用户或请求随机分成不同组，比较不同版本的线上效果。

例如：

```text
A 组：旧模型
B 组：新模型
```

然后比较两组指标是否有显著差异。

### 2.1 Control 和 Treatment

Control 是对照组，通常是当前线上版本。

Treatment 是实验组，通常是新模型、新 prompt 或新系统策略。

### 2.2 Randomization

随机分配是为了让两组用户在期望上相似。

如果 B 组刚好都是高价值用户，实验结果就会偏。

### 2.3 Unit of Randomization

随机单位可以是：

1. 用户。
2. 会话。
3. 请求。
4. 组织或企业租户。
5. 地区。

大模型产品通常优先按用户或租户分桶，而不是每个请求随机。

因为同一个用户在多轮对话中切换模型会造成体验不一致。

### 2.4 Primary Metric 和 Guardrail Metric

Primary metric 是实验主要优化目标。

Guardrail metric 是不能恶化的护栏指标。

例如：

```text
主指标：任务完成率提升。
护栏指标：延迟、成本、安全违规率、崩溃率不能恶化。
```

面试表达：A/B Test 的核心是随机分流、对照组、实验组、主指标、护栏指标和统计显著性。

### 2.5 核心数学抽象

在线实验可以抽象成一批曝光记录：

```math
D_{\mathrm{ab}}=\{e_i\}_{i=1}^{n},\qquad e_i=(u_i,z_i,y_i,c_i,t_i)
```

其中 `u_i` 是用户、会话、请求或租户等随机单位，`z_i` 是实验分组，`y_i` 是主指标观测值，`c_i` 是成本、延迟、安全等护栏指标，`t_i` 是曝光时间或事件时间。

稳定分桶可以写成：

```math
z_i=\mathbb{I}[h(u_i,e)\le q]
```

其中 `h` 是对随机单位和实验 ID 的稳定 hash，`q` 是实验组流量比例。这个公式强调：分桶必须由随机单位和实验 ID 决定，而不是由请求到达顺序或客户端临时状态决定。

对均值型主指标，treatment effect 可以写成：

```math
\hat{\tau}=\bar{y}_{\mathrm{treat}}-\bar{y}_{\mathrm{ctrl}}
```

二分类指标，例如任务完成率、点赞率或错误率，可以用比例差：

```math
\hat{\tau}=\hat{p}_{\mathrm{treat}}-\hat{p}_{\mathrm{ctrl}}
```

在大样本近似下，比例差的标准误可以写成：

```math
\mathrm{SE}(\hat{\tau})=
\sqrt{
\hat{p}(1-\hat{p})
\left(\frac{1}{n_{\mathrm{treat}}}+\frac{1}{n_{\mathrm{ctrl}}}\right)
}
```

其中 `p_hat` 是两组 pooled success rate。对应的 z-score 是：

```math
z=\frac{\hat{\tau}}{\mathrm{SE}(\hat{\tau})}
```

一个常见的 95% 置信区间写法是：

```math
\mathrm{CI}_{95}=
\left[
\hat{\tau}-1.96\,\mathrm{SE}(\hat{\tau}),
\hat{\tau}+1.96\,\mathrm{SE}(\hat{\tau})
\right]
```

真实上线判断不能只看主指标。每个护栏指标可以写成：

```math
\Delta_j=m_{j,\mathrm{treat}}-m_{j,\mathrm{ctrl}}
```

如果第 `j` 个护栏是越低越好，例如延迟、成本、安全违规率或错误率，就要求：

```math
\Delta_j\le \tau_j
```

Sample ratio mismatch 用来检查分流是否异常。两组实验的卡方统计量可以写成：

```math
X_{\mathrm{srm}}^2=
\sum_{k\in\{\mathrm{ctrl},\mathrm{treat}\}}
\frac{(o_k-e_k)^2}{e_k}
```

其中 `o_k` 是观测曝光数，`e_k` 是按预期流量比例计算的曝光数。SRM 命中通常优先怀疑分桶、曝光日志、过滤条件或实验配置有 bug。

上线门禁可以写成：

```math
G_{\mathrm{ab}}=
\mathbb{I}[\hat{\tau}>0]\,
\mathbb{I}[|z|\ge 1.96]\,
\mathbb{I}[X_{\mathrm{srm}}^2\le \tau_{\mathrm{srm}}]\,
\prod_{j=1}^{J}\mathbb{I}[\Delta_j\le \tau_j]
```

直觉是：主指标要有足够证据改善，分流要可信，所有硬护栏都不能破。

## 3. 大模型线上指标

大模型产品的线上指标要同时覆盖质量、体验、成本和安全。

### 3.1 任务成功指标

任务成功指标衡量用户目标是否完成。

常见指标：

1. Task success rate。
2. Conversation success rate。
3. 用户是否继续下一步。
4. 工具调用是否成功。
5. 代码是否通过测试。
6. RAG 答案是否被用户采纳。

任务成功率通常比“回答是否看起来好”更接近业务价值。

### 3.2 用户反馈指标

用户反馈包括：

1. 点赞率。
2. 点踩率。
3. 评分。
4. 举报。
5. 文本反馈。
6. 用户是否重新提问。
7. 用户是否复制或使用答案。

但用户反馈有偏差。

只有少数用户会主动反馈，且负反馈可能更容易发生。

### 3.3 行为指标

行为指标可以间接反映满意度：

1. Retention。
2. Session length。
3. Follow-up rate。
4. Regeneration rate。
5. Edit rate。
6. Abandon rate。
7. 用户是否切换到人工或其他工具。

这些指标要结合场景解释。

例如 session length 变长可能是用户更投入，也可能是模型没解决问题。

### 3.4 质量抽检指标

可以对线上样本抽样做人审或 LLM judge。

维度包括：

1. Correctness。
2. Helpfulness。
3. Faithfulness。
4. Instruction following。
5. Safety。
6. Style。

### 3.5 系统指标

系统指标包括：

1. Latency。
2. Time to first token。
3. Tokens per second。
4. Error rate。
5. Timeout rate。
6. Cost per request。
7. GPU utilization。
8. Cache hit rate。

大模型上线时，系统指标非常重要。

新模型质量更好，但延迟翻倍、成本翻倍，未必可上线。

### 3.6 安全指标

安全指标包括：

1. Unsafe compliance rate。
2. Refusal rate。
3. Over-refusal rate。
4. Safety classifier trigger rate。
5. User report rate。
6. Prompt injection success rate。
7. Privacy leakage risk。

面试表达：大模型线上指标要同时看任务成功、用户反馈、行为信号、人工质量抽检、系统性能、成本和安全。

## 4. 主指标设计

主指标应该直接对应产品目标。

如果主指标选错，实验可能优化错误方向。

### 4.1 Chatbot 场景

可能主指标：

1. 用户满意度。
2. 有帮助回答率。
3. 多轮任务完成率。
4. 用户留存。
5. 负反馈率下降。

### 4.2 Coding Assistant 场景

可能主指标：

1. 代码接受率。
2. 生成代码通过测试比例。
3. 用户编辑距离下降。
4. Pull request 完成时间下降。
5. 开发者留存。

### 4.3 RAG 问答场景

可能主指标：

1. 答案采纳率。
2. Faithfulness 评分。
3. 引用点击率。
4. 人工客服转接率下降。
5. 用户重复提问率下降。

### 4.4 Agent 场景

可能主指标：

1. 端到端任务成功率。
2. 工具调用成功率。
3. 人工接管率下降。
4. 平均完成步数下降。
5. 高风险操作误触发率不升高。

面试表达：主指标要从具体产品目标出发，不同场景的线上成功定义不同，不能统一用点赞率或留存率替代。

## 5. 护栏指标

护栏指标是上线实验中不能明显恶化的指标。

大模型系统尤其需要护栏，因为新模型可能在某些维度提升，同时在成本、安全或稳定性上退化。

### 5.1 常见护栏

常见护栏指标：

1. P95 latency。
2. Error rate。
3. Timeout rate。
4. Cost per successful task。
5. Unsafe output rate。
6. Privacy leakage rate。
7. Over-refusal rate。
8. User report rate。
9. Tool failure rate。
10. 用户流失率。

### 5.2 硬护栏和软护栏

硬护栏：超过阈值必须停止实验或回滚。

软护栏：超过阈值需要人工 review。

例如安全和隐私通常是硬护栏。

成本和延迟可能根据业务阶段设置软硬阈值。

### 5.3 护栏冲突

护栏之间可能冲突。

例如降低 unsafe output 可能提高 refusal rate。

这时要看 policy 和产品目标，不能机械优化单一指标。

面试表达：护栏指标保证实验不会用安全、稳定性、成本或用户体验的退化换取表面主指标提升。

## 6. 流量切分与分桶

流量切分是 A/B Test 的工程基础。

### 6.1 用户级分桶

用户级分桶把同一个用户固定分到同一实验组。

优点：

1. 体验一致。
2. 多轮对话不混模型。
3. 适合长期指标。

缺点：需要更多流量才能收敛。

### 6.2 请求级分桶

请求级分桶把每个请求随机到不同组。

优点：样本多，收敛快。

缺点：同一用户体验可能不一致。

适合无状态、低耦合请求。

### 6.3 租户级分桶

企业产品常按组织或租户分桶。

原因是同一企业用户之间会互相影响，且权限、知识库、工作流通常绑定租户。

### 6.4 分桶稳定性

分桶必须稳定。

常见做法是对用户 ID 或租户 ID 做 hash。

例如：

```text
bucket = hash(user_id + experiment_id) % 100
```

不同实验应使用不同 salt 或 experiment_id，避免分桶相关性过强。

### 6.5 实验隔离

多个实验同时运行时，要注意相互干扰。

例如一个实验改模型，另一个实验改 prompt，两者叠加后很难解释结果。

解决方法包括：

1. 互斥实验层。
2. 正交实验设计。
3. 限制同一用户参与实验数量。
4. 记录完整实验曝光日志。

面试表达：流量切分要明确随机单位，保证分桶稳定、实验隔离和曝光日志完整，否则实验结果不可解释。

## 7. 灰度发布

灰度发布是逐步扩大新版本流量，降低上线风险。

### 7.1 常见灰度流程

常见流程：

1. 内部 dogfood。
2. 小流量线上灰度。
3. 低风险用户或场景放量。
4. 扩大到 5%、10%、25%、50%。
5. 全量上线。
6. 持续监控。

每一步都要检查主指标和护栏指标。

### 7.2 大模型灰度特别注意

大模型灰度要特别关注：

1. 安全违规。
2. 成本突增。
3. 延迟尾部。
4. 工具调用异常。
5. RAG 引用错误。
6. 用户投诉。
7. 过度拒答。

### 7.3 Canary Release

Canary release 是先让极小比例流量使用新版本。

它适合发现明显线上问题。

但 canary 流量小，不能判断所有统计显著差异。

### 7.4 Shadow Mode

Shadow mode 是让新模型在后台接收同样请求，但不把结果展示给用户。

优点：低风险。

缺点：无法观察真实用户交互反馈。

适合上线前检查延迟、成本、错误率和离线质量。

面试表达：灰度发布不是替代 A/B test，而是风险控制手段；先小流量发现事故，再逐步放量验证效果。

## 8. 回滚策略

上线前必须设计回滚策略。

没有回滚的实验不是合格实验。

### 8.1 什么时候回滚

常见触发条件：

1. 安全违规率超过阈值。
2. 隐私风险上升。
3. P95 latency 明显恶化。
4. Error rate 或 timeout rate 上升。
5. 成本超预算。
6. 用户投诉激增。
7. 关键业务指标明显下降。
8. 工具调用出现高风险异常。

### 8.2 回滚方式

回滚方式包括：

1. 切回旧模型。
2. 切回旧 prompt。
3. 关闭某个工具。
4. 降级到安全模式。
5. 限制高风险场景。
6. 暂停实验流量。

### 8.3 回滚后做什么

回滚不是结束。

还要做：

1. 保存实验日志。
2. 抽样失败案例。
3. 判断是模型、prompt、检索、工具还是系统问题。
4. 更新回归测试。
5. 修复后重新灰度。

面试表达：上线实验必须提前定义回滚阈值和操作路径，尤其是安全、隐私、成本和可用性指标不能等事故发生后再临时决策。

## 9. 用户反馈处理

用户反馈是在线实验的重要信号，但不能直接当作真相。

### 9.1 显式反馈

显式反馈包括点赞、点踩、评分、举报、文本评论。

优点：直接表达用户感受。

缺点：稀疏、有选择偏差、容易受 UI 影响。

### 9.2 隐式反馈

隐式反馈包括：

1. 用户是否复制答案。
2. 是否继续追问。
3. 是否点击引用。
4. 是否重新生成。
5. 是否放弃会话。
6. 是否使用生成代码。

隐式反馈样本更多，但解释更困难。

### 9.3 反馈偏差

常见偏差：

1. 极端用户更愿意反馈。
2. 长答案更容易看起来有帮助。
3. 用户不一定知道答案是否正确。
4. UI 位置影响点击。
5. 新奇效应影响短期行为。

### 9.4 反馈到训练的风险

线上反馈不能直接无脑进入训练。

原因：

1. 反馈有噪声。
2. 可能被刷。
3. 可能包含隐私。
4. 可能强化短期讨好用户行为。
5. 可能污染评估集。

面试表达：用户反馈很重要，但要和人工抽检、任务成功指标、日志分析结合，并经过隐私过滤和质量控制后才能用于训练闭环。

## 10. 离线和在线不一致怎么办

离线分数提升但线上不提升，是常见问题。

### 10.1 可能原因

常见原因：

1. 离线评估集不匹配线上分布。
2. 主指标选错。
3. 用户更关注延迟或格式。
4. 新模型回答风格不适合产品。
5. RAG 或工具链瓶颈掩盖模型提升。
6. 安全策略导致过度拒答。
7. 实验流量太少，统计不稳定。
8. 存在分桶或日志 bug。

### 10.2 排查顺序

建议按顺序排查：

1. 检查实验配置和曝光日志。
2. 检查流量分布是否均衡。
3. 检查主指标和护栏指标。
4. 按用户群、任务类型、语言、场景分层。
5. 抽样做人审。
6. 对比离线样本和线上样本分布。
7. 检查系统延迟、成本和错误率。
8. 分析 bad case。

### 10.3 反向更新离线评估

线上 bad case 应该沉淀回离线评估集。

这样离线 benchmark 才能逐步贴近真实产品。

面试表达：离线和在线不一致时，不要简单否定某一方，要检查实验配置、流量、指标、系统链路和样本分布，并把线上失败反哺离线评估。

## 11. A/B Test 的常见坑

### 11.1 样本量不足

样本量太少时，指标波动可能只是随机噪声。

下一章会详细讲统计显著性和样本量。

### 11.2 提前停止实验

看到中途结果好就停止，容易产生假阳性。

实验应提前定义运行时长和停止规则。

### 11.3 多指标挑选

如果看了很多指标，只挑一个变好的报告，会夸大效果。

应提前定义主指标和护栏指标。

### 11.4 Novelty Effect

用户刚看到新功能可能短期更活跃。

长期效果可能回落。

### 11.5 Interference

用户之间可能互相影响。

企业产品、协作工具和社区产品尤其明显。

### 11.6 Logging Bug

实验结果经常被日志问题污染。

例如曝光记录缺失、重复计数、事件延迟、客户端版本不一致。

面试表达：A/B test 常见坑包括样本量不足、提前停止、多指标挑选、新奇效应、实验干扰和日志 bug。

## 12. 真实项目实验设计模板

设计一个模型上线实验，可以按下面模板回答。

### 12.1 实验目标

明确要验证什么：

```text
新模型是否在不恶化延迟、成本和安全的前提下，提高用户任务完成率？
```

### 12.2 实验组和对照组

定义：

1. Control：当前线上模型和 prompt。
2. Treatment：新模型或新策略。
3. 其他组件是否保持不变。

### 12.3 随机单位

选择用户级、请求级或租户级分桶。

说明理由。

### 12.4 指标

定义：

1. 主指标。
2. 护栏指标。
3. 分层指标。
4. 人工抽检指标。

### 12.5 实验周期

说明：

1. 预估样本量。
2. 运行周期。
3. 是否覆盖工作日和周末。
4. 是否有提前停止规则。

### 12.6 上线策略

说明：

1. 先 shadow 或 canary。
2. 小流量灰度。
3. 检查护栏。
4. 逐步放量。
5. 异常回滚。

面试表达：一个合格的 A/B test 设计要包括假设、对照组、实验组、随机单位、主指标、护栏指标、样本量、实验周期、灰度和回滚方案。

## 13. 最小可运行在线实验审计 Demo

下面这个 demo 用纯标准库模拟一轮模型上线实验。它演示：稳定分桶正常，主指标有统计改善，但 treatment 的延迟、成本、安全和误拒护栏失败，因此不能上线。

```python
import hashlib
import math


def stable_bucket(user_id, experiment_id):
    key = f"{experiment_id}:{user_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % 100


def assign(user_id, experiment_id="exp_model_v2", treatment_pct=50):
    bucket = stable_bucket(user_id, experiment_id)
    return "treatment" if bucket < treatment_pct else "control"


def rate(numer, denom):
    return 0.0 if denom == 0 else numer / denom


def p95(values):
    ordered = sorted(values)
    idx = math.ceil(0.95 * len(ordered)) - 1
    return ordered[idx]


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

guardrails = {
    "srm": srm_chi < 10.83,
    "latency": summary["treatment"]["p95_latency_ms"] <= summary["control"]["p95_latency_ms"] * 1.25,
    "cost": summary["treatment"]["cost_per_success"] <= summary["control"]["cost_per_success"] * 1.30,
    "unsafe": summary["treatment"]["unsafe_rate"] <= summary["control"]["unsafe_rate"] + 0.002,
    "over_refusal": summary["treatment"]["over_refusal_rate"] <= summary["control"]["over_refusal_rate"] + 0.02,
    "error": summary["treatment"]["error_rate"] <= summary["control"]["error_rate"] + 0.005,
}

primary_pass = primary["delta"] > 0 and abs(z) >= 1.96

print("bucket_preview=", bucket_preview, sep="")
print("summary=", summary, sep="")
print("primary=", primary, sep="")
print("srm_chi_square=", round(srm_chi, 3), sep="")
print("primary_pass=", primary_pass, sep="")
print("guardrails=", guardrails, sep="")
print("ship_decision=", primary_pass and all(guardrails.values()), sep="")
```

一组可能输出如下：

```text
bucket_preview={'u001': 'control', 'u006': 'treatment', 'u007': 'treatment', 'u008': 'control', 'u010': 'treatment'}
summary={'control': {'success_rate': 0.52, 'unsafe_rate': 0.002, 'over_refusal_rate': 0.035, 'error_rate': 0.009, 'cost_per_success': 0.231, 'p95_latency_ms': 1350}, 'treatment': {'success_rate': 0.57, 'unsafe_rate': 0.006, 'over_refusal_rate': 0.07, 'error_rate': 0.011, 'cost_per_success': 0.316, 'p95_latency_ms': 2100}}
primary={'control_rate': 0.52, 'treatment_rate': 0.57, 'delta': 0.05, 'se': 0.0223, 'z': 2.245, 'ci95': (0.006, 0.094)}
srm_chi_square=0.0
primary_pass=True
guardrails={'srm': True, 'latency': False, 'cost': False, 'unsafe': False, 'over_refusal': False, 'error': True}
ship_decision=False
```

这个结果的重点是：treatment 的任务成功率确实提升，并且粗略 z-score 已经超过 1.96；但是 P95 latency、cost per successful task、unsafe rate 和 over-refusal rate 都破了护栏，所以上线决策仍然是 `False`。真实项目里，这类结果通常进入分层分析、模型路由、成本优化、安全修复或小范围场景限定，而不是直接全量发布。

## 14. 面试官会怎么问

### 14.1 离线 benchmark 提升，线上没提升，你怎么分析？

回答要点：

1. 先检查实验配置、分桶和日志。
2. 看样本量和统计显著性。
3. 分析线上用户分布是否匹配离线评估。
4. 分层看任务类型、语言、用户群。
5. 检查延迟、成本、安全和拒答率。
6. 抽样做人审和 bad case 分析。
7. 把线上失败样本加入离线回归集。

### 14.2 如何为一个 ChatGPT 类产品设计 A/B Test？

回答要点：

1. 用户级分桶，保持会话体验一致。
2. Control 是旧模型，Treatment 是新模型。
3. 主指标可以是有帮助回答率、任务成功率或用户满意度。
4. 护栏包括延迟、成本、安全、过度拒答和错误率。
5. 小流量灰度，监控异常，逐步放量。
6. 抽样做人审校准用户反馈。

### 14.3 为什么不能只看点赞率？

回答要点：

1. 点赞反馈稀疏。
2. 有选择偏差。
3. 用户未必能判断事实正确性。
4. UI 和回答长度会影响点赞。
5. 要结合任务成功率、人工评估、负反馈、安全和留存。

### 14.4 如何设置护栏指标？

回答要点：

1. 从安全、隐私、成本、延迟、稳定性和用户体验出发。
2. 区分硬护栏和软护栏。
3. 预先定义阈值和回滚条件。
4. 对高风险指标不能用平均提升抵消。

## 15. 标准回答模板

如果面试官问：“你会如何设计一个大模型上线 A/B Test？”

可以这样回答：

```text
我会先明确实验假设，例如新模型是否能在不恶化安全、延迟和成本的前提下提升任务完成率。然后设置 control 为当前线上版本，treatment 为新模型或新策略，并尽量保持其他系统组件不变。

分流上我会根据产品选择随机单位。对多轮聊天或企业产品，我倾向于用户级或租户级分桶，保证体验一致，并记录完整曝光日志。指标上提前定义一个主指标，例如任务成功率、答案采纳率或代码接受率，同时设置护栏指标，包括 P95 latency、error rate、cost per request、unsafe output、privacy leakage、over-refusal 和用户投诉。

上线策略上，我会先做 shadow 或 canary，再小流量灰度，观察护栏指标后逐步放量。实验结束后不仅看总体指标，还会按任务类型、语言、用户群和场景分层分析，并抽样做人审。如果离线和在线不一致，我会检查日志、分桶、样本分布和系统链路，并把线上 bad case 沉淀到离线回归集。
```

## 16. 常见误区

### 16.1 把 A/B Test 当成上线仪式

A/B Test 不是走流程，而是验证真实用户价值和风险。

### 16.2 只看主指标不看护栏

主指标提升但安全、成本或延迟恶化，可能不能上线。

### 16.3 不记录曝光日志

没有曝光日志，就无法知道用户实际看到哪个版本。

### 16.4 请求级随机破坏多轮体验

聊天产品中，同一对话频繁切模型会影响用户体验和实验解释。

### 16.5 不分层分析

总体提升可能掩盖某个重要用户群或任务类型的大幅下降。

### 16.6 实验后不更新离线评估

线上 bad case 如果不沉淀，下一次还会重复踩坑。

## 17. 小练习

### 练习 1

为一个 RAG 问答产品设计 A/B Test。

要求说明：主指标、护栏指标、分桶单位、人工抽检方案和回滚条件。

### 练习 2

一个新模型让点赞率提升 3%，但 P95 latency 增加 40%，成本增加 60%。你会如何判断是否上线？

思考方向：业务价值、用户群分层、成本预算、延迟护栏、是否可路由给部分场景。

### 练习 3

如果实验组用户平均会话更长，这是好事还是坏事？

请列出至少三种可能解释。

### 练习 4

设计一个 coding assistant 的线上指标体系。

至少包含：代码接受率、编辑距离、测试通过率、延迟、错误率和安全指标。

## 18. 本章总结

在线实验是把离线模型能力转化为真实产品判断的关键环节。

离线 benchmark 能帮助筛选模型，但不能替代真实用户实验。

A/B Test 要明确实验假设、对照组、实验组、随机单位、主指标、护栏指标、样本量、实验周期和回滚策略。

大模型线上指标必须同时覆盖任务成功、用户反馈、行为信号、人工质量抽检、系统性能、成本和安全。

灰度发布和回滚策略是风险控制手段，尤其适合大模型这种行为复杂、成本高、风险高的系统。

面试中要强调：线上效果不等于离线分数，A/B Test 的目标不是证明新模型更强，而是验证它是否在真实场景中带来净收益。
