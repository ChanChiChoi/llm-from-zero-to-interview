# 第 47 章 SLO、SLA、错误预算和生产值班体系

上一章讲了推理故障定位。本章继续讲可靠性治理中的核心概念：SLO、SLA、错误预算和生产值班体系。

AI Infra 不只是把训练和推理跑起来，还要稳定地服务研发和线上业务。稳定性不能只靠救火，而要通过目标、指标、预算、告警、值班和复盘形成体系。

先记住一句话：

> 可靠性治理的核心，不是追求永不失败，而是明确什么程度的失败可接受，并围绕这个目标组织工程、发布和值班。

## 47.0 本讲资料边界与第二轮精修口径

本章按通用 AI Infra 生产可靠性治理来写，不绑定某个云厂商、监控系统、告警平台、工单系统、Kubernetes 发行版或内部 SRE 组织形态。资料校准时，主要参考 Google SRE Book 对 SLI、SLO、SLA、用户体验指标、错误预算和“不追求 100% SLO”的定义，参考 Google SRE Workbook 对 burn rate、多窗口多 burn rate 告警、on-call、incident response 和 postmortem action item 的实践口径，并结合前文可观测性、训练故障定位、推理故障定位、发布治理、回滚降级、成本治理和事故复盘章节。

第二轮精修只做三件事：

1. 把 SLI / SLO / SLA、错误预算、burn rate、告警、值班、runbook、事故分级、变更、复盘和成本取舍统一成可审计的可靠性治理样本。
2. 补齐 availability、latency pass rate、平台失败率、数据构建 SLO、错误预算、预算消耗、burn rate、告警可行动率、MTTA / MTTR、事故影响、runbook 覆盖、变更关联、行动项关闭率和 SLO 成本收益公式。
3. 增加一个 0 依赖 Python demo，用 toy SLO / on-call cases 检查生产值班体系是否只是“有人收告警”，还是能真正把用户体验、错误预算、告警路由、止损权限、复盘行动项和发布策略串成闭环。

## 47.1 SLI、SLO、SLA 的区别

先区分三个概念。

SLI 是 Service Level Indicator，服务水平指标。

它是实际观测到的指标。

例如：

1. 推理请求成功率。
2. p99 latency。
3. TTFT p95。
4. 训练任务成功率。
5. checkpoint 恢复成功率。

SLO 是 Service Level Objective，服务水平目标。

它是内部设定的目标。

例如：

```text
99.9% 推理请求在 3 秒内返回首 token
```

SLA 是 Service Level Agreement，服务水平协议。

它是对客户或业务承诺的协议，通常有赔偿或约束。

简单说：SLI 是指标，SLO 是目标，SLA 是承诺。

一个可靠性治理样本可以抽象成：

$$
R_i=(s_i,o_i,a_i,b_i,g_i,p_i,r_i,c_i,m_i,z_i)
$$

其中 `s_i` 是 SLI 定义，`o_i` 是 SLO 目标，`a_i` 是 SLA 或业务承诺边界，`b_i` 是错误预算，`g_i` 是告警和 burn rate 规则，`p_i` 是值班排班和升级路径，`r_i` 是 runbook，`c_i` 是变更事件，`m_i` 是止损 / 回滚 / 降级动作，`z_i` 是复盘和行动项。

这组变量的核心要求是“可测量、可行动、可归责、可复盘”。如果一个 SLO 不能落到实际观测数据和明确动作，就只是口号。

## 47.2 为什么 AI Infra 需要 SLO

没有 SLO，平台团队很难回答：

1. 推理慢到什么程度算故障？
2. 训练任务失败率多少可接受？
3. 评估平台排队多久算异常？
4. GPU 集群利用率低是否要处理？
5. 发布导致 p99 上升多少必须回滚？
6. 成本上升多少需要告警？

SLO 给工程决策提供边界。

没有目标，所有问题都变成“看情况”。

SLO 的控制闭环可以写成：

$$
G_{\mathrm{slo}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land B_{\mathrm{remain}}>0 \land P_0=0\right]
$$

其中 `C_j` 是第 `j` 个关键 SLO 或治理门禁覆盖率，`\tau_j` 是最低阈值，`B_{\mathrm{remain}}` 是剩余错误预算比例，`P_0` 是未关闭的最高优先级可靠性阻断项数量。这个式子表达的是：SLO 不是单个 dashboard 数字，而是目标、预算和阻断项共同组成的上线控制条件。

## 47.3 AI Infra 常见 SLI

训练平台 SLI：

1. 任务提交成功率。
2. 任务调度等待时间。
3. 训练任务成功率。
4. 节点失败恢复时间。
5. checkpoint 保存成功率。
6. checkpoint 恢复成功率。
7. GPU 利用率。

推理平台 SLI：

1. 请求成功率。
2. TTFT。
3. TPOT。
4. p95 / p99 延迟。
5. timeout rate。
6. error rate。
7. admission reject rate。

数据和评估平台 SLI：

1. dataset 构建成功率。
2. 数据质量门禁通过率。
3. eval job 成功率。
4. eval report 生成时延。

RAG/Agent SLI：

1. retrieval success rate。
2. tool call success rate。
3. agent run success rate。
4. citation available rate。

统一的 SLI 覆盖率可以写成：

$$
C_{\mathrm{sli}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[s_i\ \mathrm{defined}\land s_i\ \mathrm{measured}\land s_i\ \mathrm{owned}]
$$

其中 `N` 是关键服务或关键工作流数量。这个指标要求每个重要工作流不仅有指标名，还要有采集口径、时间窗口、owner 和对应动作。

## 47.4 推理 SLO 示例

推理平台可以定义：

```text
Availability SLO:
99.9% 请求返回非 5xx 和非 timeout

Latency SLO:
95% 请求 TTFT < 1s
99% 请求 end-to-end latency < 10s

Streaming SLO:
95% 请求 TPOT < 80ms

Quality Guardrail:
线上安全违规率低于阈值
```

注意：推理 SLO 不只看可用性，还要看 TTFT、TPOT 和质量 guardrail。

大模型服务如果 100% 返回但全部很慢，用户仍然认为不可用。

可用性可以定义为好请求占有效请求的比例：

$$
A=\frac{N_{\mathrm{good}}}{N_{\mathrm{valid}}}
$$

其中 `N_{\mathrm{good}}` 是非 5xx、非 timeout、非平台拒绝且满足基本响应契约的请求数，`N_{\mathrm{valid}}` 是纳入 SLO 统计的有效请求数。

延迟 SLO 常用通过率，而不是只看平均值：

$$
C_{\mathrm{lat}}(\tau)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[T_i\le \tau]
$$

其中 `T_i` 可以是 TTFT、TPOT 或 end-to-end latency，`\tau` 是目标阈值。推理平台的上线门禁可以写成：

$$
G_{\mathrm{infer\_slo}}=\mathbf{1}\left[A\ge a_0 \land C_{\mathrm{ttft}}(\tau_f)\ge c_f \land C_{\mathrm{tpot}}(\tau_p)\ge c_p \land R_{\mathrm{err}}\le r_0 \land Q_{\mathrm{guard}}\ge q_0\right]
$$

其中 `a_0` 是可用性目标，`c_f` 是 TTFT 通过率目标，`c_p` 是 TPOT 通过率目标，`R_{\mathrm{err}}` 是错误率，`Q_{\mathrm{guard}}` 是质量 / 安全 guardrail 得分。这个门禁能防止“请求都返回了，但慢、错或不安全”的假可用。

## 47.5 训练 SLO 示例

训练平台可以定义：

```text
Submission SLO:
99.5% 训练任务提交请求在 5s 内完成校验

Scheduling SLO:
高优先级任务 95% 在 10 分钟内开始运行

Reliability SLO:
训练任务因平台原因失败率 < 1%

Checkpoint SLO:
99% checkpoint 保存成功，恢复成功率 > 99%
```

训练平台 SLO 和推理平台不同。

训练任务本身可能因为用户代码失败，这不一定算平台错误。

因此要区分用户错误和平台错误。

平台原因失败率可以写成：

$$
R_{\mathrm{platform\_fail}}=\frac{N_{\mathrm{platform\_fail}}}{N_{\mathrm{train\_job}}}
$$

其中 `N_{\mathrm{platform\_fail}}` 只统计调度、节点、镜像、存储、网络、权限、checkpoint、平台 launcher 等平台责任导致的失败，用户代码、用户数据格式错误和显式超配资源不应直接算进平台失败率。

训练平台的可靠性门禁可以写成：

$$
G_{\mathrm{train\_slo}}=\mathbf{1}\left[R_{\mathrm{platform\_fail}}\le r_t \land C_{\mathrm{submit}}\ge c_s \land C_{\mathrm{schedule}}\ge c_q \land C_{\mathrm{ckpt}}\ge c_k\right]
$$

其中 `C_{\mathrm{submit}}` 是提交校验通过率，`C_{\mathrm{schedule}}` 是调度等待时间达标率，`C_{\mathrm{ckpt}}` 是 checkpoint 保存 / 恢复达标率。

## 47.6 数据平台 SLO 示例

数据平台可以定义：

```text
Dataset Build SLO:
95% dataset build 在预期时间内完成

Data Quality SLO:
生产 dataset 必须通过质量门禁

Access SLO:
99.9% dataset manifest 查询成功

Serving SLO:
训练数据读取吞吐满足任务声明需求
```

数据平台 SLO 不只是 API 可用，还要看数据质量和训练供给能力。

数据构建 SLO 可以写成：

$$
C_{\mathrm{build}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[T_{\mathrm{build},i}\le \tau_i \land Q_i\ge q_i \land M_i=1]
$$

其中 `T_{\mathrm{build},i}` 是第 `i` 个 dataset build 的耗时，`\tau_i` 是预期完成时间，`Q_i` 是数据质量门禁得分，`M_i` 表示 manifest、checksum、lineage 和权限记录是否完整。这个定义提醒面试时不要把数据平台 SLO 讲成“API 还能访问”，而要包含数据是否可训练、可追溯、可治理。

## 47.7 错误预算是什么

错误预算是 SLO 允许的失败空间。

例如 SLO 是 99.9% 可用性，那么一个周期内允许 0.1% 请求失败。

如果一个月有 1000 万请求：

```text
允许失败请求 = 1000 万 * 0.1% = 1 万
```

这 1 万就是错误预算。

错误预算用来平衡稳定性和迭代速度。

如果 SLO 目标是 `S_{\mathrm{slo}}`，错误预算比例是：

$$
B=1-S_{\mathrm{slo}}
$$

给定统计窗口内有效请求数 `N_{\mathrm{valid}}`，允许的坏事件数是：

$$
N_{\mathrm{bad,allowed}}=N_{\mathrm{valid}}B
$$

如果统计对象不是请求，也可以把 `N_{\mathrm{valid}}` 换成训练任务数、dataset build 数、eval job 数或 agent run 数。关键是先定义“什么算坏事件”，再谈预算。

## 47.8 错误预算怎么用

错误预算可以指导决策：

1. 预算充足，可以正常发布新功能。
2. 预算消耗过快，减少高风险发布。
3. 预算耗尽，进入稳定性优先模式。
4. 某个模型持续消耗预算，必须修复或降级。
5. 某个租户异常消耗预算，可能需要限流或隔离。

错误预算不是财务预算，而是可靠性预算。

它让发布速度和稳定性之间有量化边界。

预算已使用比例和剩余比例可以写成：

$$
U_{\mathrm{budget}}=\frac{N_{\mathrm{bad}}}{\max(1,N_{\mathrm{bad,allowed}})}
$$

$$
B_{\mathrm{remain}}=\max(0,1-U_{\mathrm{budget}})
$$

其中 `N_{\mathrm{bad}}` 是当前窗口已经发生的坏事件数。发布策略可以由 `B_{\mathrm{remain}}` 驱动：剩余预算高时允许常规灰度，剩余预算低时减速或冻结高风险发布，预算耗尽时只允许可靠性修复。

## 47.9 Burn Rate

Burn rate 是错误预算消耗速度。

如果一天就消耗了一个月 50% 的错误预算，说明问题很严重。

常见告警方式：

1. 快速 burn rate 告警：短时间内大量消耗预算。
2. 慢速 burn rate 告警：长期小幅超标。

快速 burn 适合发现事故。

慢速 burn 适合发现慢性退化。

AI Infra 中，p99 慢性上升、成本慢性上涨、某个模型错误率小幅升高，都可能是慢速 burn。

某个时间窗口 `w` 的 burn rate 可以写成：

$$
\beta_w=\frac{R_{\mathrm{bad},w}}{B}
$$

其中 `R_{\mathrm{bad},w}` 是窗口内坏事件比例，`B` 是错误预算比例。若 99.9% SLO 的预算比例是 0.1%，窗口内错误率是 0.6%，则 `\beta_w=6`，意味着当前错误消耗速度是预算允许速度的 6 倍。

多窗口告警通常同时看快窗口和慢窗口：

$$
G_{\mathrm{burn}}=\mathbf{1}\left[(\beta_{\mathrm{1h}}\ge \beta_f \land \beta_{\mathrm{5m}}\ge \beta_f)\lor(\beta_{\mathrm{6h}}\ge \beta_s \land \beta_{\mathrm{30m}}\ge \beta_s)\right]
$$

其中 `\beta_f` 是快速 burn 阈值，`\beta_s` 是慢速 burn 阈值。这样既能发现突发事故，也能降低已经恢复后的告警残留。

## 47.10 SLO 不能太多

很多团队会定义几十个 SLO，最后没人看。

好的 SLO 应该：

1. 反映用户体验。
2. 可测量。
3. 可行动。
4. 能区分平台责任和用户责任。
5. 数量有限。

例如推理平台优先选择：

1. Availability。
2. TTFT。
3. TPOT。
4. p99 latency。
5. timeout rate。

不要把所有内部指标都升级成 SLO。

SLO 数量本身也可以治理：

$$
G_{\mathrm{slo\_set}}=\mathbf{1}\left[N_{\mathrm{slo}}\le n_0 \land C_{\mathrm{user}}\ge c_u \land C_{\mathrm{action}}\ge c_a \land C_{\mathrm{owner}}\ge c_o\right]
$$

其中 `N_{\mathrm{slo}}` 是某个服务的正式 SLO 数量，`C_{\mathrm{user}}` 是用户体验相关覆盖率，`C_{\mathrm{action}}` 是 SLO 触发后有明确动作的比例，`C_{\mathrm{owner}}` 是有 owner 的比例。这个门禁的直觉是：SLO 太多会稀释注意力，SLO 太虚会无法驱动工程动作。

## 47.11 告警和 SLO 的关系

告警应围绕 SLO 和用户影响设计。

不好的告警：

```text
GPU utilization > 90%
```

因为 GPU 高利用率可能是正常高吞吐。

更好的告警：

```text
TTFT p95 超过 SLO 且 queue wait 持续上升
```

告警要告诉值班人员：用户体验是否受影响，可能原因是什么，下一步看哪里。

告警可行动率可以写成：

$$
A_{\mathrm{action}}=\frac{N_{\mathrm{actionable}}}{N_{\mathrm{alert}}}
$$

告警噪音率可以写成：

$$
R_{\mathrm{noise}}=\frac{N_{\mathrm{no\_action}}+N_{\mathrm{duplicate}}}{N_{\mathrm{alert}}}
$$

其中 `N_{\mathrm{actionable}}` 是触发明确排查、止损、回滚、扩容、限流或修复动作的告警数，`N_{\mathrm{no\_action}}` 是无人需要处理的告警数，`N_{\mathrm{duplicate}}` 是重复告警数。好的告警体系应提高 `A_{\mathrm{action}}`，降低 `R_{\mathrm{noise}}`。

## 47.12 生产值班体系

生产值班是保障线上系统的组织机制。

值班体系至少包括：

1. 值班排班。
2. 告警路由。
3. 升级机制。
4. Runbook。
5. 事故分级。
6. 状态页或内部通报。
7. 事故复盘。
8. 后续行动项跟踪。

没有值班体系，再好的监控也没人处理。

值班响应的两个基本指标是 MTTA 和 MTTR：

$$
\mathrm{MTTA}=\frac{1}{N}\sum_{i=1}^{N}(t_{\mathrm{ack},i}-t_{\mathrm{detect},i})
$$

$$
\mathrm{MTTR}=\frac{1}{N}\sum_{i=1}^{N}(t_{\mathrm{recover},i}-t_{\mathrm{detect},i})
$$

其中 `t_{\mathrm{detect},i}` 是第 `i` 个事故被检测到的时间，`t_{\mathrm{ack},i}` 是值班确认时间，`t_{\mathrm{recover},i}` 是用户影响恢复时间。MTTR 不等于根因完全修复时间，线上事故通常先止损恢复，再做完整根因修复。

## 47.13 事故分级

事故可以分级。

示例：

1. SEV1：核心推理服务大面积不可用。
2. SEV2：关键租户或关键模型严重受影响。
3. SEV3：部分功能退化，有降级方案。
4. SEV4：低影响问题或内部平台异常。

分级决定响应速度、通知范围和复盘要求。

AI Infra 中，训练平台大面积不可用和线上推理不可用影响不同，分级要结合业务场景。

事故影响分数可以写成：

$$
S_{\mathrm{incident}}=w_uU+w_dD+w_sS+w_rR
$$

其中 `U` 是受影响用户或租户比例，`D` 是持续时间，`S` 是服务重要性权重，`R` 是风险权重，例如安全、隐私、收入、合规或关键客户影响。事故分级不应只看错误率，还要结合业务影响和风险。

## 47.14 On-call Runbook

Runbook 是值班手册。

一个好的 runbook 包括：

1. 告警含义。
2. 影响范围判断。
3. 需要查看的 dashboard。
4. 常见根因。
5. 应急处理步骤。
6. 回滚或降级方法。
7. 升级联系人。
8. 复盘记录模板。

Runbook 要实际可执行，不是概念文档。

Runbook 覆盖率可以写成：

$$
C_{\mathrm{runbook}}=\frac{N_{\mathrm{alert\_with\_runbook}}}{N_{\mathrm{critical\_alert}}}
$$

执行就绪门禁可以写成：

$$
G_{\mathrm{runbook}}=\mathbf{1}\left[C_{\mathrm{runbook}}\ge c_r \land C_{\mathrm{step}}\ge c_s \land C_{\mathrm{rollback}}\ge c_b \land C_{\mathrm{owner}}\ge c_o\right]
$$

其中 `C_{\mathrm{step}}` 是 runbook 步骤可执行覆盖率，`C_{\mathrm{rollback}}` 是回滚 / 降级动作覆盖率，`C_{\mathrm{owner}}` 是升级联系人覆盖率。

## 47.15 推理事故处理流程

推理事故可以这样处理：

```text
1. 确认告警和影响范围
2. 查看是否有发布或流量事件
3. 按模型、租户、endpoint、地域切分指标
4. 查看 TTFT、TPOT、queue、KV cache、error、timeout
5. 决定是否限流、降级、摘除实例、扩容或回滚
6. 恢复服务后定位根因
7. 记录复盘和行动项
```

线上事故中，先恢复服务，再做完整根因分析。

推理事故的止损门禁可以写成：

$$
G_{\mathrm{mitigation}}=\mathbf{1}\left[M_{\mathrm{rate}}\lor M_{\mathrm{degrade}}\lor M_{\mathrm{rollback}}\lor M_{\mathrm{scale}}\lor M_{\mathrm{isolate}}\right]
$$

其中 `M_{\mathrm{rate}}` 是限流，`M_{\mathrm{degrade}}` 是降级，`M_{\mathrm{rollback}}` 是回滚，`M_{\mathrm{scale}}` 是扩容，`M_{\mathrm{isolate}}` 是摘除异常实例、租户、模型或路由。这个公式不是说所有动作都要做，而是要求至少存在一条已验证的止损路径。

## 47.16 训练事故处理流程

训练事故可以这样处理：

```text
1. 判断是单任务失败还是平台级故障
2. 区分用户代码错误和平台错误
3. 查看调度、节点、镜像、数据、通信和存储事件
4. 如果影响多个任务，暂停相关队列或迁移资源
5. 恢复 checkpoint 或重试任务
6. 修复根因并补充检测
```

训练平台要避免某个节点、存储或网络问题影响大量任务。

训练事故造成的丢失 GPU 小时可以写成：

$$
K_{\mathrm{lost}}=\sum_{j=1}^{J}G_jT_j
$$

其中 `G_j` 是第 `j` 个受影响训练任务占用的 GPU 数，`T_j` 是从故障发生到恢复或重试成功之间损失的小时数。这个指标能把“训练任务失败”转成资源浪费和调度影响，便于决定是否要升级事故级别。

## 47.17 值班中的权限

值班人员需要足够权限处理事故，但权限不能无限大。

常见值班权限：

1. 查看 dashboard。
2. 查看脱敏日志。
3. 摘除 endpoint。
4. 触发回滚。
5. 修改限流配置。
6. 暂停队列。
7. 扩缩容。

高风险操作要有审计，必要时需要双人确认。

值班权限门禁可以写成：

$$
G_{\mathrm{oncall\_perm}}=\mathbf{1}\left[C_{\mathrm{least}}\ge c_l \land C_{\mathrm{audit}}\ge c_a \land C_{\mathrm{breakglass}}\ge c_b \land H_{\mathrm{confirm}}=1\right]
$$

其中 `C_{\mathrm{least}}` 是最小权限覆盖率，`C_{\mathrm{audit}}` 是操作审计覆盖率，`C_{\mathrm{breakglass}}` 是紧急授权流程覆盖率，`H_{\mathrm{confirm}}` 表示高风险动作是否需要确认或审批。值班权限的目标不是让人“什么都不能做”，而是让值班能止损，同时让高风险动作可追踪、可回放、可复盘。

## 47.18 变更管理

很多事故来自变更。

变更包括：

1. 模型版本发布。
2. prompt 变更。
3. runtime 升级。
4. 路由规则变更。
5. 数据版本变更。
6. 安全策略变更。
7. 集群配置变更。
8. 网络或存储升级。

值班系统应该能看到最近变更时间线。

排查事故时，第一问题通常是：最近变了什么？

变更关联覆盖率可以写成：

$$
C_{\mathrm{change}}=\frac{N_{\mathrm{incident\_with\_change\_timeline}}}{N_{\mathrm{incident}}}
$$

一次 AI Infra 事故至少应能关联 `model`、`prompt`、`runtime`、`route`、`data`、`cluster`、`cache`、`safety` 和 `owner` 等字段。没有变更时间线，事故排查会退化为猜测。

## 47.19 事故复盘

事故复盘不是追责，而是改进系统。

复盘应包括：

1. 时间线。
2. 影响范围。
3. 用户影响。
4. 检测方式。
5. 响应过程。
6. 根因。
7. 哪些防线失效。
8. 修复措施。
9. 后续行动项。
10. 负责人和截止时间。

AI Infra 复盘还应包括：模型、数据、runtime、GPU、路由、缓存、成本和安全是否相关。

行动项关闭率可以写成：

$$
C_{\mathrm{action}}=\frac{N_{\mathrm{action\_closed}}}{N_{\mathrm{action\_total}}}
$$

复盘质量门禁可以写成：

$$
G_{\mathrm{postmortem}}=\mathbf{1}\left[C_{\mathrm{timeline}}\ge c_t \land C_{\mathrm{impact}}\ge c_i \land C_{\mathrm{root}}\ge c_r \land C_{\mathrm{action}}\ge c_a \land C_{\mathrm{regression}}\ge c_g\right]
$$

其中 `C_{\mathrm{timeline}}` 是时间线完整率，`C_{\mathrm{impact}}` 是影响量化覆盖率，`C_{\mathrm{root}}` 是根因分类覆盖率，`C_{\mathrm{regression}}` 是修复后回归验证覆盖率。复盘的价值不在于文档写完，而在于行动项真正关闭并转成监控、回归、runbook 或上线门禁。

## 47.20 错误预算和发布策略

错误预算可以影响发布策略。

例如：

1. 预算充足：正常灰度发布。
2. 预算消耗较快：降低灰度速度。
3. 预算接近耗尽：冻结高风险发布。
4. 预算耗尽：只允许修复稳定性问题。

对 AI 平台来说，模型发布、runtime 升级和路由策略变更都应该受错误预算约束。

基于错误预算的发布门禁可以写成：

$$
G_{\mathrm{release\_by\_budget}}=\mathbf{1}\left[B_{\mathrm{remain}}\ge b_0 \land \beta_{\mathrm{fast}}<\beta_f \land \beta_{\mathrm{slow}}<\beta_s \land R_{\mathrm{rollback}}=1\right]
$$

其中 `b_0` 是发布所需的最低剩余预算，`\beta_{\mathrm{fast}}` 和 `\beta_{\mathrm{slow}}` 是快慢窗口 burn rate，`R_{\mathrm{rollback}}` 表示是否有可执行回滚。这个门禁能把“最近稳定吗”转成可执行发布条件。

## 47.21 SLO 和成本的关系

更高 SLO 通常意味着更高成本。

例如：

1. 更低 TTFT 需要更多 warm pool。
2. 更低 p99 需要更多冗余。
3. 更高可用性需要多地域容灾。
4. 更快训练调度需要预留 GPU。

所以 SLO 不是越高越好。

要根据业务价值设定合理目标。

对低优先级批处理任务，99.99% 可用性可能没有意义。

SLO 提升的价值 / 成本比可以粗略写成：

$$
V_{\mathrm{slo}}=\frac{\Delta U_{\mathrm{user}}+\Delta R_{\mathrm{risk}}}{\Delta K_{\mathrm{infra}}+\Delta K_{\mathrm{ops}}}
$$

其中 `\Delta U_{\mathrm{user}}` 是用户体验或业务收益提升，`\Delta R_{\mathrm{risk}}` 是风险降低收益，`\Delta K_{\mathrm{infra}}` 是额外基础设施成本，`\Delta K_{\mathrm{ops}}` 是额外运维和值班成本。面试中要能说明：99.99% 并不天然比 99.9% 更好，只有当价值 / 成本比足够高时才值得追求。

## 47.22 SLO 值班体系审计指标和最小 demo

把本章落到平台验收时，可以用 16 个门禁：

1. SLI Contract Completeness：关键服务的 SLI 是否有定义、采集、窗口、owner 和坏事件口径。
2. SLO Target Measurability：SLO 是否可测量、可回放、可解释，并能对应具体用户体验。
3. SLA Boundary Clarity：对外承诺、内部目标、赔偿或业务责任边界是否清楚。
4. Inference SLO Coverage：推理是否覆盖 availability、TTFT、TPOT、p99、error、timeout 和质量 / 安全 guardrail。
5. Training SLO Coverage：训练是否区分用户错误和平台错误，并覆盖提交、调度、平台失败、checkpoint 和恢复。
6. Data Eval SLO Coverage：数据、评估、RAG / Agent 是否覆盖 build、manifest、质量、eval report、retrieval / tool / run success。
7. Error Budget Accounting：错误预算是否按窗口、坏事件和服务维度计算，并能驱动发布策略。
8. Burn Rate Alerting：是否有快慢窗口 burn rate 告警，并能减少恢复后的告警残留。
9. Alert Actionability：告警是否可行动、低噪音、有 owner、有 runbook、有下一步。
10. Oncall Ownership Escalation：值班排班、升级链路、ACK SLO、负责人和状态沟通是否完整。
11. Incident Severity Routing：事故分级是否结合用户影响、持续时间、服务重要性和风险。
12. Runbook Executability：runbook 是否包含影响判断、dashboard、根因路径、止损步骤、回滚和升级联系人。
13. Change Event Linkage：事故系统是否能关联模型、prompt、runtime、route、data、cluster、cache 和 owner 变更。
14. Mitigation Rollback Authority：值班是否有可审计的限流、降级、扩容、摘除、回滚和暂停队列权限。
15. Postmortem Action Closure：复盘是否包含时间线、影响、根因、防线失效、行动项 owner、截止时间和回归验证。
16. SLO Cost Tradeoff Gate：SLO 目标是否经过用户价值、风险降低、基础设施成本和值班成本取舍。

综合门禁：

$$
G_{\mathrm{slo\_oncall}}=\prod_{j=1}^{16}G_j
$$

下面是一个 0 依赖 demo，用 toy SLO / on-call cases 检查生产可靠性治理是否只是“有人收告警”，还是能真正把 SLI、SLO、SLA、错误预算、burn rate、值班、runbook、变更、止损和复盘行动项串成闭环。

```python
from copy import deepcopy


class MiniSLOOncallAudit:
    GATES = [
        "sli_contract_completeness",
        "slo_target_measurability",
        "sla_boundary_clarity",
        "inference_slo_coverage",
        "training_slo_coverage",
        "data_eval_slo_coverage",
        "error_budget_accounting",
        "burn_rate_alerting",
        "alert_actionability",
        "oncall_ownership_escalation",
        "incident_severity_routing",
        "runbook_executability",
        "change_event_linkage",
        "mitigation_rollback_authority",
        "postmortem_action_closure",
        "slo_cost_tradeoff_gate",
    ]

    SLI_FIELDS = ["name", "definition", "source", "window", "owner", "bad_event_rule"]
    SLO_FIELDS = ["target", "window", "threshold", "query", "business_reason", "action"]
    SLA_FIELDS = ["customer_scope", "promise", "exclusion", "remedy", "owner"]
    INFERENCE_FIELDS = ["availability", "ttft", "tpot", "p99", "error", "timeout", "quality_guardrail"]
    TRAINING_FIELDS = ["submission", "scheduling", "platform_fail", "checkpoint_save", "checkpoint_restore"]
    DATA_EVAL_FIELDS = ["dataset_build", "manifest_access", "quality_gate", "eval_report", "rag_agent_success"]
    CHANGE_FIELDS = ["model", "prompt", "runtime", "route", "data", "cluster", "cache", "owner"]
    RUNBOOK_FIELDS = ["meaning", "impact", "dashboards", "root_causes", "mitigation", "rollback", "escalation"]
    POSTMORTEM_FIELDS = ["timeline", "impact", "detection", "response", "root_cause", "failed_defenses", "actions", "regression"]

    @staticmethod
    def present(record, key):
        return key in record and record[key] is not None and record[key] != ""

    def coverage(self, record, fields):
        if not record:
            return 0.0
        return sum(1 for field in fields if self.present(record, field)) / len(fields)

    def availability(self, case):
        infer = case["inference"]
        return infer["good_requests"] / infer["valid_requests"]

    def latency_pass_rate(self, case):
        infer = case["inference"]
        return infer["latency_pass"] / infer["valid_requests"]

    def bad_allowed(self, case):
        budget = case["error_budget"]
        return int(round(case["inference"]["valid_requests"] * (1.0 - budget["slo_target"])))

    def budget_used(self, case):
        budget = case["error_budget"]
        return budget["bad_events"] / max(1, self.bad_allowed(case))

    def budget_remaining(self, case):
        return max(0.0, 1.0 - self.budget_used(case))

    def burn_rate(self, case, key):
        budget = case["error_budget"]
        return budget[key] / max(1e-12, 1.0 - budget["slo_target"])

    def actionability(self, case):
        alerts = case["alerts"]
        return alerts["actionable"] / alerts["total"]

    def noise_rate(self, case):
        alerts = case["alerts"]
        return (alerts["no_action"] + alerts["duplicate"]) / alerts["total"]

    def mtta_minutes(self, case):
        oncall = case["oncall"]
        return oncall["ack_minutes_total"] / oncall["incidents"]

    def action_item_closure(self, case):
        postmortem = case["postmortem"]
        return postmortem["action_closed"] / postmortem["action_total"]

    def value_cost_ratio(self, case):
        tradeoff = case["cost_tradeoff"]
        value = tradeoff["user_value_delta"] + tradeoff["risk_reduction_value"]
        cost = tradeoff["infra_cost_delta"] + tradeoff["ops_cost_delta"]
        return value / cost

    def sli_contract_completeness(self, case):
        return self.coverage(case.get("sli_contract", {}), self.SLI_FIELDS) == 1.0

    def slo_target_measurability(self, case):
        slo = case.get("slo_target", {})
        return (
            self.coverage(slo, self.SLO_FIELDS) == 1.0
            and slo.get("replayable") is True
            and slo.get("user_visible") is True
        )

    def sla_boundary_clarity(self, case):
        return self.coverage(case.get("sla_boundary", {}), self.SLA_FIELDS) == 1.0

    def inference_slo_coverage(self, case):
        infer = case.get("inference", {})
        return (
            self.coverage(infer.get("slo_fields", {}), self.INFERENCE_FIELDS) == 1.0
            and self.availability(case) >= infer.get("availability_target", 1.0)
            and self.latency_pass_rate(case) >= infer.get("latency_pass_target", 1.0)
            and infer.get("quality_guardrail_pass") is True
        )

    def training_slo_coverage(self, case):
        train = case.get("training", {})
        return (
            self.coverage(train.get("slo_fields", {}), self.TRAINING_FIELDS) == 1.0
            and train.get("user_platform_error_split") is True
            and train.get("platform_fail_rate", 1.0) <= train.get("max_platform_fail_rate", 0.0)
            and train.get("checkpoint_restore_success", 0.0) >= train.get("checkpoint_restore_target", 1.0)
        )

    def data_eval_slo_coverage(self, case):
        data_eval = case.get("data_eval", {})
        return (
            self.coverage(data_eval.get("slo_fields", {}), self.DATA_EVAL_FIELDS) == 1.0
            and data_eval.get("build_success_rate", 0.0) >= data_eval.get("build_target", 1.0)
            and data_eval.get("eval_report_latency_pass", 0.0) >= data_eval.get("eval_report_target", 1.0)
        )

    def error_budget_accounting(self, case):
        budget = case.get("error_budget", {})
        return (
            budget.get("window_days", 0) > 0
            and budget.get("bad_event_rule") is True
            and budget.get("service_dimension") is True
            and self.budget_used(case) <= budget.get("max_budget_used", 0.0)
            and budget.get("release_policy_bound") is True
        )

    def burn_rate_alerting(self, case):
        burn = case.get("burn_alerts", {})
        return (
            self.burn_rate(case, "fast_window_bad_rate") <= burn.get("fast_threshold", 0.0)
            and self.burn_rate(case, "slow_window_bad_rate") <= burn.get("slow_threshold", 0.0)
            and burn.get("multi_window") is True
            and burn.get("recovery_suppression") is True
        )

    def alert_actionability(self, case):
        alerts = case.get("alerts", {})
        return (
            self.actionability(case) >= alerts.get("min_actionability", 1.0)
            and self.noise_rate(case) <= alerts.get("max_noise_rate", 0.0)
            and alerts.get("owner") is True
            and alerts.get("runbook") is True
            and alerts.get("next_step") is True
        )

    def oncall_ownership_escalation(self, case):
        oncall = case.get("oncall", {})
        return (
            oncall.get("schedule") is True
            and oncall.get("primary") is True
            and oncall.get("secondary") is True
            and oncall.get("escalation_policy") is True
            and self.mtta_minutes(case) <= oncall.get("ack_slo_minutes", 0.0)
            and oncall.get("status_comms") is True
        )

    def incident_severity_routing(self, case):
        severity = case.get("severity", {})
        score = (
            severity["user_impact"] * severity["w_user"]
            + severity["duration_minutes"] * severity["w_duration"]
            + severity["service_weight"] * severity["w_service"]
            + severity["risk_weight"] * severity["w_risk"]
        )
        return (
            score >= severity.get("sev_threshold", 10**9)
            and severity.get("sev_level") == severity.get("expected_sev")
            and severity.get("notification_scope") is True
            and severity.get("review_required") is True
        )

    def runbook_executability(self, case):
        runbook = case.get("runbook", {})
        return (
            self.coverage(runbook, self.RUNBOOK_FIELDS) == 1.0
            and runbook.get("last_drill_days", 10**9) <= runbook.get("max_drill_age_days", 0)
            and runbook.get("tested_by_oncall") is True
        )

    def change_event_linkage(self, case):
        change = case.get("change_events", {})
        return (
            self.coverage(change.get("fields", {}), self.CHANGE_FIELDS) == 1.0
            and change.get("timeline_linked") is True
            and change.get("diff_ready") is True
            and change.get("owner_present") is True
        )

    def mitigation_rollback_authority(self, case):
        mitigation = case.get("mitigation", {})
        required = ["rate_limit", "degrade", "scale", "isolate", "rollback", "pause_queue"]
        return (
            all(mitigation.get(field) is True for field in required)
            and mitigation.get("audit_log") is True
            and mitigation.get("high_risk_confirmed") is True
            and mitigation.get("rollback_tested") is True
        )

    def postmortem_action_closure(self, case):
        postmortem = case.get("postmortem", {})
        return (
            self.coverage(postmortem.get("fields", {}), self.POSTMORTEM_FIELDS) == 1.0
            and self.action_item_closure(case) >= postmortem.get("min_action_closure", 1.0)
            and postmortem.get("owners_and_due_dates") is True
            and postmortem.get("regression_verified") is True
        )

    def slo_cost_tradeoff_gate(self, case):
        tradeoff = case.get("cost_tradeoff", {})
        return (
            self.value_cost_ratio(case) >= tradeoff.get("min_value_cost_ratio", 1.0)
            and tradeoff.get("business_priority_bound") is True
            and tradeoff.get("capacity_plan_ready") is True
            and tradeoff.get("ops_load_accepted") is True
        )

    def audit_case(self, case):
        return {gate: getattr(self, gate)(case) for gate in self.GATES}

    def run_all(self, cases):
        results = {case["case_id"]: self.audit_case(case) for case in cases}
        metrics = {}
        for gate in self.GATES:
            passed = sum(1 for result in results.values() if result[gate])
            metrics[gate] = round(passed / len(cases), 3)
        failed_cases = [
            case_id
            for case_id, result in results.items()
            if not all(result.values())
        ]
        failed_gates = [
            gate
            for gate in self.GATES
            if any(not result[gate] for result in results.values())
        ]
        return {
            "metrics": metrics,
            "hard_blocker_count": len(failed_cases),
            "failed_cases": failed_cases,
            "failed_gates": failed_gates,
            "slo_oncall_gate_pass": metrics["slo_cost_tradeoff_gate"] == 1.0,
        }

    def example_outputs(self, case):
        return {
            "availability": round(self.availability(case), 4),
            "latency_pass_rate": round(self.latency_pass_rate(case), 4),
            "bad_allowed": self.bad_allowed(case),
            "budget_used": round(self.budget_used(case), 3),
            "budget_remaining": round(self.budget_remaining(case), 3),
            "fast_burn": round(self.burn_rate(case, "fast_window_bad_rate"), 3),
            "slow_burn": round(self.burn_rate(case, "slow_window_bad_rate"), 3),
            "actionable_alerts": case["alerts"]["actionable"],
            "ack_slo_minutes": case["oncall"]["ack_slo_minutes"],
            "sev1_response_minutes": case["oncall"]["sev1_response_minutes"],
            "runbook_count": case["runbook"]["count"],
            "change_fields": sorted(case["change_events"]["fields"]),
            "mitigation_ready": self.mitigation_rollback_authority(case),
            "action_item_closure": round(self.action_item_closure(case), 3),
            "value_cost_ratio": round(self.value_cost_ratio(case), 3),
        }


def build_good_case():
    return {
        "case_id": "full_slo_oncall_system",
        "sli_contract": {
            "name": "chat_availability",
            "definition": "non_5xx_non_timeout",
            "source": "request_metrics",
            "window": "30d",
            "owner": "serving-sre",
            "bad_event_rule": "5xx_or_timeout_or_contract_violation",
        },
        "slo_target": {
            "target": "99.9 availability and 95 TTFT under 1s",
            "window": "30d",
            "threshold": 0.999,
            "query": "slo_query",
            "business_reason": "interactive chat user experience",
            "action": "release_freeze_or_rollback",
            "replayable": True,
            "user_visible": True,
        },
        "sla_boundary": {
            "customer_scope": "paid_enterprise_chat",
            "promise": "monthly_availability_commitment",
            "exclusion": "customer_code_and_policy_blocks",
            "remedy": "service_credit",
            "owner": "platform-owner",
        },
        "inference": {
            "slo_fields": {
                "availability": True,
                "ttft": True,
                "tpot": True,
                "p99": True,
                "error": True,
                "timeout": True,
                "quality_guardrail": True,
            },
            "good_requests": 9992,
            "valid_requests": 10000,
            "latency_pass": 9700,
            "availability_target": 0.999,
            "latency_pass_target": 0.95,
            "quality_guardrail_pass": True,
        },
        "training": {
            "slo_fields": {
                "submission": True,
                "scheduling": True,
                "platform_fail": True,
                "checkpoint_save": True,
                "checkpoint_restore": True,
            },
            "user_platform_error_split": True,
            "platform_fail_rate": 0.006,
            "max_platform_fail_rate": 0.01,
            "checkpoint_restore_success": 0.995,
            "checkpoint_restore_target": 0.99,
        },
        "data_eval": {
            "slo_fields": {
                "dataset_build": True,
                "manifest_access": True,
                "quality_gate": True,
                "eval_report": True,
                "rag_agent_success": True,
            },
            "build_success_rate": 0.97,
            "build_target": 0.95,
            "eval_report_latency_pass": 0.96,
            "eval_report_target": 0.95,
        },
        "error_budget": {
            "slo_target": 0.9,
            "bad_events": 800,
            "window_days": 30,
            "bad_event_rule": True,
            "service_dimension": True,
            "max_budget_used": 0.85,
            "release_policy_bound": True,
            "fast_window_bad_rate": 0.6,
            "slow_window_bad_rate": 0.15,
        },
        "burn_alerts": {
            "fast_threshold": 10.0,
            "slow_threshold": 2.0,
            "multi_window": True,
            "recovery_suppression": True,
        },
        "alerts": {
            "total": 4,
            "actionable": 3,
            "no_action": 0,
            "duplicate": 1,
            "min_actionability": 0.7,
            "max_noise_rate": 0.3,
            "owner": True,
            "runbook": True,
            "next_step": True,
        },
        "oncall": {
            "schedule": True,
            "primary": True,
            "secondary": True,
            "escalation_policy": True,
            "ack_minutes_total": 15,
            "incidents": 3,
            "ack_slo_minutes": 5,
            "sev1_response_minutes": 5,
            "status_comms": True,
        },
        "severity": {
            "user_impact": 0.65,
            "duration_minutes": 30,
            "service_weight": 4,
            "risk_weight": 5,
            "w_user": 10,
            "w_duration": 0.1,
            "w_service": 2,
            "w_risk": 2,
            "sev_threshold": 20,
            "sev_level": "SEV1",
            "expected_sev": "SEV1",
            "notification_scope": True,
            "review_required": True,
        },
        "runbook": {
            "meaning": True,
            "impact": True,
            "dashboards": True,
            "root_causes": True,
            "mitigation": True,
            "rollback": True,
            "escalation": True,
            "last_drill_days": 20,
            "max_drill_age_days": 90,
            "tested_by_oncall": True,
            "count": 1,
        },
        "change_events": {
            "fields": {
                "model": True,
                "prompt": True,
                "runtime": True,
                "route": True,
                "data": True,
                "cluster": True,
                "cache": True,
                "owner": True,
            },
            "timeline_linked": True,
            "diff_ready": True,
            "owner_present": True,
        },
        "mitigation": {
            "rate_limit": True,
            "degrade": True,
            "scale": True,
            "isolate": True,
            "rollback": True,
            "pause_queue": True,
            "audit_log": True,
            "high_risk_confirmed": True,
            "rollback_tested": True,
        },
        "postmortem": {
            "fields": {
                "timeline": True,
                "impact": True,
                "detection": True,
                "response": True,
                "root_cause": True,
                "failed_defenses": True,
                "actions": True,
                "regression": True,
            },
            "action_closed": 9,
            "action_total": 10,
            "min_action_closure": 0.85,
            "owners_and_due_dates": True,
            "regression_verified": True,
        },
        "cost_tradeoff": {
            "user_value_delta": 160,
            "risk_reduction_value": 110,
            "infra_cost_delta": 70,
            "ops_cost_delta": 30,
            "min_value_cost_ratio": 2.0,
            "business_priority_bound": True,
            "capacity_plan_ready": True,
            "ops_load_accepted": True,
        },
    }


def build_bad_cases(good_case):
    cases = []

    case = deepcopy(good_case)
    case["case_id"] = "sli_contract_missing_bad"
    case["sli_contract"].pop("bad_event_rule")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "slo_target_unmeasurable_bad"
    case["slo_target"]["query"] = ""
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "sla_boundary_unclear_bad"
    case["sla_boundary"].pop("remedy")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "inference_slo_missing_bad"
    case["inference"]["slo_fields"]["tpot"] = None
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "training_slo_missing_bad"
    case["training"]["user_platform_error_split"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "data_eval_slo_missing_bad"
    case["data_eval"]["slo_fields"].pop("eval_report")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "error_budget_exhausted_bad"
    case["error_budget"]["bad_events"] = 1100
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "burn_rate_alerting_bad"
    case["burn_alerts"]["multi_window"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "alert_actionability_bad"
    case["alerts"]["actionable"] = 1
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "oncall_escalation_bad"
    case["oncall"]["secondary"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "severity_routing_bad"
    case["severity"]["sev_level"] = "SEV3"
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "runbook_executability_bad"
    case["runbook"]["rollback"] = None
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "change_event_linkage_bad"
    case["change_events"]["fields"].pop("runtime")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "mitigation_rollback_bad"
    case["mitigation"]["rollback"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "postmortem_action_bad"
    case["postmortem"]["action_closed"] = 5
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "slo_cost_tradeoff_bad"
    case["cost_tradeoff"]["infra_cost_delta"] = 250
    cases.append(case)

    return cases


audit = MiniSLOOncallAudit()
good = build_good_case()
cases = [good] + build_bad_cases(good)
summary = audit.run_all(cases)

print("slo_oncall_examples=" + repr(audit.example_outputs(good)))
print("metrics=" + repr(summary["metrics"]))
print("hard_blocker_count=" + repr(summary["hard_blocker_count"]))
print("failed_cases=" + repr(summary["failed_cases"]))
print("failed_gates=" + repr(summary["failed_gates"]))
print("slo_oncall_gate_pass=" + repr(summary["slo_oncall_gate_pass"]))
```

参考输出应类似：

```text
slo_oncall_examples={'availability': 0.9992, 'latency_pass_rate': 0.97, 'bad_allowed': 1000, 'budget_used': 0.8, 'budget_remaining': 0.2, 'fast_burn': 6.0, 'slow_burn': 1.5, 'actionable_alerts': 3, 'ack_slo_minutes': 5, 'sev1_response_minutes': 5, 'runbook_count': 1, 'change_fields': ['cache', 'cluster', 'data', 'model', 'owner', 'prompt', 'route', 'runtime'], 'mitigation_ready': True, 'action_item_closure': 0.9, 'value_cost_ratio': 2.7}
metrics={'sli_contract_completeness': 0.941, 'slo_target_measurability': 0.941, 'sla_boundary_clarity': 0.941, 'inference_slo_coverage': 0.941, 'training_slo_coverage': 0.941, 'data_eval_slo_coverage': 0.941, 'error_budget_accounting': 0.941, 'burn_rate_alerting': 0.941, 'alert_actionability': 0.941, 'oncall_ownership_escalation': 0.941, 'incident_severity_routing': 0.941, 'runbook_executability': 0.941, 'change_event_linkage': 0.941, 'mitigation_rollback_authority': 0.941, 'postmortem_action_closure': 0.941, 'slo_cost_tradeoff_gate': 0.941}
hard_blocker_count=16
failed_cases=['sli_contract_missing_bad', 'slo_target_unmeasurable_bad', 'sla_boundary_unclear_bad', 'inference_slo_missing_bad', 'training_slo_missing_bad', 'data_eval_slo_missing_bad', 'error_budget_exhausted_bad', 'burn_rate_alerting_bad', 'alert_actionability_bad', 'oncall_escalation_bad', 'severity_routing_bad', 'runbook_executability_bad', 'change_event_linkage_bad', 'mitigation_rollback_bad', 'postmortem_action_bad', 'slo_cost_tradeoff_bad']
failed_gates=['sli_contract_completeness', 'slo_target_measurability', 'sla_boundary_clarity', 'inference_slo_coverage', 'training_slo_coverage', 'data_eval_slo_coverage', 'error_budget_accounting', 'burn_rate_alerting', 'alert_actionability', 'oncall_ownership_escalation', 'incident_severity_routing', 'runbook_executability', 'change_event_linkage', 'mitigation_rollback_authority', 'postmortem_action_closure', 'slo_cost_tradeoff_gate']
slo_oncall_gate_pass=False
```

这个 demo 的面试价值是把 SLO 和值班体系从“定义几个百分比指标”升级为生产可靠性治理闭环：SLI 必须有契约，SLO 必须可测量，SLA 边界必须清楚，推理、训练、数据和评估要各自有用户体验指标，错误预算要能影响发布，burn rate 告警要可行动，值班要能升级和止损，事故复盘要把失败转成行动项、回归样本、runbook 和下一次发布门禁。

## 47.23 常见误区

误区一：SLA、SLO、SLI 是一回事。

SLI 是指标，SLO 是目标，SLA 是对外承诺。

误区二：SLO 越高越好。

更高 SLO 需要更高成本和更慢发布速度，要和业务价值匹配。

误区三：错误预算只是 SRE 概念，AI 平台不用。

AI 平台同样需要平衡模型迭代、runtime 升级和线上稳定性。

误区四：告警越多越安全。

告警太多会造成疲劳。告警应该围绕用户影响和 SLO。

误区五：事故复盘就是找人背锅。

复盘目标是改进系统防线、检测能力和响应流程。

## 47.24 面试常见追问

问题一：SLI、SLO、SLA 有什么区别？

可以回答：SLI 是实际观测指标，SLO 是内部服务目标，SLA 是对客户或业务承诺的协议，通常带有责任或赔偿。

问题二：推理平台如何定义 SLO？

可以回答：可以定义请求成功率、TTFT、TPOT、p99 延迟、timeout rate、错误率，并结合安全和质量 guardrail。SLO 要反映用户体验。

问题三：错误预算有什么用？

可以回答：错误预算量化了 SLO 允许的失败空间，用来平衡稳定性和迭代速度。预算消耗过快时应降低发布风险或优先修复可靠性问题。

问题四：AI Infra 值班体系应该包含什么？

可以回答：包括告警、排班、事故分级、runbook、升级机制、回滚/降级权限、变更时间线、事故复盘和行动项跟踪。

## 47.25 小练习

1. 为什么 SLO 不能只看平均延迟？
2. 为推理服务设计 3 个 SLI 和 3 个 SLO。
3. 为训练平台设计一个 checkpoint SLO。
4. 错误预算耗尽后应该如何调整发布策略？
5. Burn rate 告警有什么价值？
6. 一个推理 p99 告警的 runbook 应包含什么？
7. 为什么值班操作需要审计？
8. 事故复盘应该记录哪些内容？

## 47.26 本章小结

本章讲了 SLO、SLA、错误预算和生产值班体系。

你需要记住：

1. SLI 是指标，SLO 是目标，SLA 是承诺。
2. AI Infra 的 SLO 要覆盖推理、训练、数据、评估、RAG/Agent 和成本相关体验。
3. 错误预算用于平衡稳定性和迭代速度，不是追求零故障。
4. 告警应围绕 SLO 和用户影响设计，避免噪音。
5. 生产值班体系需要告警、runbook、事故分级、权限、变更管理和复盘机制。
6. 更高可靠性通常意味着更高成本，SLO 要和业务价值匹配。

下一章我们会讲成本治理：GPU 成本、存储成本、网络成本和推理成本。
