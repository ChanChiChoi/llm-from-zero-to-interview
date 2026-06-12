# 第四章：成本收益与 ROI

## 0. 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做公式和 demo 精修。联网资料主要核对四类口径：OpenAI 官方 pricing / Batch / prompt caching / latency optimization 资料提醒我们，模型调用成本由输入、输出、缓存命中、批处理、工具和延迟策略共同决定，而且价格会变化；OpenAI evals / evaluation best practices 强调 ROI 不能脱离任务成功率和回归评估；Google SRE 的 SLO / error budget 口径提醒我们成本优化不能破坏可靠性和尾延迟；FinOps Foundation 的云成本管理口径强调成本要按单位经济账、责任归属和持续优化来管理。

本章不写任何长期固定的 API 价格，也不替代企业财务模型、采购合同或真实云账单分析。正文中的价格都是 toy 参数，用来说明算法工程师在面试和项目复盘中如何把 token、RAG、工具、人审、固定研发摊销、风险成本和业务收益放进同一张账。

大模型项目不能只问“效果好不好”，还要问“值不值得”。如果一个系统每次回答都很准确，但调用成本太高、人工审核太重、延迟太长、收益无法覆盖投入，就很难成为可持续产品。成本收益与 ROI，是大模型产品化绕不开的问题。

本章系统讲大模型项目的成本和收益：token 成本、GPU 成本、工程成本、数据成本、人工审核成本、运维成本；收益侧包括人工替代、效率提升、质量提升、收入增长和风险降低；最后讲 ROI 估算、单位经济账、成本优化和面试表达。

## 4.1 为什么要算 ROI

ROI，投入产出比，回答的是一个项目是否值得持续投入。

一个大模型 demo 可以不算 ROI，但产品必须算。原因是：

1. 模型调用有持续成本。
2. 用户规模扩大后成本会线性甚至超线性增长。
3. 人工审核和运维不能忽略。
4. 模型效果提升不一定带来业务收益。
5. 业务方需要判断优先级。

面试回答：

```text
大模型项目必须算 ROI，因为模型能力只有转化为业务收益才可持续。ROI 要同时看收益和成本：收益包括节省人工、提升效率、增加收入、降低风险；成本包括 token、GPU、检索、工具调用、人审、数据治理、研发和运维。不能只看离线指标，也不能只看 API 单价。
```

## 4.2 成本不只是 Token

很多人只算 token 成本，这是不够的。

完整成本包括：

1. 模型调用成本。
2. GPU 推理成本。
3. RAG 检索成本。
4. 向量库和存储成本。
5. reranker 和 verifier 成本。
6. Agent 工具调用成本。
7. 人工审核成本。
8. 数据清洗和标注成本。
9. 工程开发成本。
10. 运维监控成本。
11. 合规和安全成本。
12. 失败和事故成本。

产品化要看全生命周期成本，而不是单次模型调用报价。

## 4.3 Token 成本

Token 成本来自输入和输出。

影响因素：

1. Prompt 长度。
2. 用户输入长度。
3. RAG 拼接文档长度。
4. 历史上下文长度。
5. 输出长度。
6. 重试次数。
7. 多样本采样次数。
8. Agent 多轮调用。

例如普通问答一次调用可能只消耗几千 token；Agentic RAG 可能多轮检索、多次阅读、多次生成，token 成本显著更高。

## 4.4 GPU 成本

如果自部署模型，要考虑 GPU 成本。

包括：

1. GPU 采购或租用。
2. 推理服务部署。
3. 显存占用。
4. 吞吐和并发。
5. 量化和加速优化。
6. 空闲资源浪费。
7. 运维人员成本。

自部署不一定比 API 便宜。只有当调用量足够大、模型大小合适、团队有推理优化能力时，自部署才可能有成本优势。

## 4.5 RAG 成本

RAG 的成本包括：

1. 文档解析。
2. 分块。
3. embedding。
4. 向量库存储。
5. 检索。
6. reranking。
7. 文档更新。
8. 权限过滤。
9. 引用校验。

RAG 不是免费增强。企业知识库越大、权限越复杂、更新越频繁，成本越高。

## 4.6 Agent 成本

Agent 成本通常比普通问答高。

来源：

1. 多轮模型调用。
2. 工具调用。
3. 搜索和检索。
4. 代码执行或浏览器操作。
5. 失败重试。
6. trace 日志存储。
7. 安全检查。
8. 人工确认。

Agent 适合高价值任务，不适合所有简单请求默认开启。否则成本和延迟都会失控。

## 4.7 人工审核成本

很多大模型产品需要人在回路。

人工审核成本包括：

1. 审核时间。
2. 审核人员培训。
3. 质检流程。
4. 复核争议样本。
5. 处理用户投诉。
6. 标注反馈数据。

如果模型节省了 5 分钟，但审核要花 4 分钟，实际收益就很有限。人机协同设计要尽量让人只审核关键点，而不是重新做一遍。

## 4.8 工程和运维成本

产品上线后需要持续维护。

工程成本包括：

1. 前端和后端开发。
2. 模型服务接入。
3. 数据管道。
4. 权限系统。
5. 监控告警。
6. 日志和审计。
7. 回归评估。
8. 灰度发布。
9. 线上问题排查。

大模型产品不是接一个 API 就结束。越接近企业级应用，工程和运维成本越重要。

## 4.9 收益类型

大模型项目收益可以分为几类：

1. 节省人工。
2. 提升效率。
3. 提高质量。
4. 增加收入。
5. 降低风险。
6. 提升用户体验。
7. 加速知识沉淀。
8. 扩大服务覆盖。

不同收益的可量化程度不同。节省人工最容易算；用户满意度和知识沉淀较难直接量化，但仍然可以通过代理指标衡量。

## 4.10 人工替代收益

最常见收益是节省人工时间。

例如：

```text
每月 10000 个客服问题
每个问题原来人工处理 5 分钟
AI 能自动解决 40%
每小时人工成本 60 元
```

粗略收益：

```text
10000 * 40% * 5 / 60 * 60 = 20000 元/月
```

这里还要扣除模型成本、系统成本和人审成本。

## 4.11 效率提升收益

有些场景不是替代人工，而是提高效率。

例如：

1. 销售写方案从 2 小时降到 30 分钟。
2. 工程师查文档从 15 分钟降到 3 分钟。
3. 法务初审合同从 1 小时降到 20 分钟。
4. 数据分析师生成初稿从半天降到 1 小时。

效率提升不一定减少人数，但可以提升产能、缩短交付周期、支持更多客户。

## 4.12 质量收益

大模型也可能提升质量。

例如：

1. 减少漏检风险。
2. 提高回复一致性。
3. 提升文档完整性。
4. 降低新人上手成本。
5. 让专家经验可复用。

质量收益较难量化，但可以通过错误率、投诉率、返工率、审核通过率等指标间接衡量。

## 4.13 收入增长

大模型也可能带来收入增长。

例如：

1. 提高销售转化率。
2. 提升用户留存。
3. 增加付费功能。
4. 支持更多客户同时服务。
5. 提升客单价。

收入增长类项目要特别注意归因。转化率提升可能来自多个因素，不能轻易把所有增长都归因给大模型。

## 4.14 风险降低收益

有些收益来自降低风险。

例如：

1. 合同风险提示。
2. 内容合规审核。
3. 安全告警总结。
4. 财务异常检测辅助。
5. 客诉预警。

风险降低的收益可能不高频，但单次事故代价大。评估时可以看历史事故成本、风险暴露数量和人工审核覆盖率。

## 4.15 单位经济账

单位经济账是看单次任务是否划算。

例如每处理一个客服问题：

```text
模型成本：0.05 元
检索成本：0.01 元
人审成本：0.20 元
总成本：0.26 元
节省人工成本：0.80 元
净收益：0.54 元
```

如果任务量很大，单次净收益很关键。小小的成本差异会被规模放大。

## 4.16 ROI 估算模板

一个简单模板：

```text
月任务量 = N
自动化比例 = A
单次节省时间 = T 小时
人工时薪 = W
月收益 = N * A * T * W

月成本 = 模型成本 + 检索成本 + 人审成本 + 运维成本 + 研发摊销
ROI = 月收益 / 月成本
```

如果是收入增长场景，可以把月收益换成增量收入；如果是风险降低场景，可以用预期风险损失下降来估算。

## 4.16.1 关键公式与 ROI 指标速查

为了把 ROI 讲清楚，可以把候选产品化场景记成一个账本样本：

```math
b_i=(N_i,A_i,Q_i,T_i,W_i,V_i,P_i,C_i,F_i,R_i)
```

其中，$N_i$ 是月任务量，$A_i$ 是采用率或覆盖率，$Q_i$ 是任务成功率提升，$T_i$ 是单次节省时间，$W_i$ 是小时成本，$V_i$ 是单次业务价值，$P_i$ 是当前单次失败或返工概率，$C_i$ 是单次可变成本，$F_i$ 是月固定成本，$R_i$ 是预期风险成本或风险下降收益。

模型调用成本可以先按输入和输出 token 拆开：

```math
C_{\mathrm{model},i}=M_i\left(\frac{I_i p_{\mathrm{in}}(1-H_i)}{K}+\frac{O_i p_{\mathrm{out}}}{K}\right)
```

其中，$M_i$ 是单任务模型调用次数，$I_i$ 是平均输入 token，$O_i$ 是平均输出 token，$p_{\mathrm{in}}$ 和 $p_{\mathrm{out}}$ 是每 $K$ 个 token 的 toy 单价，$H_i$ 是缓存命中带来的输入折扣比例。这里故意写成变量，因为真实价格、缓存折扣和批处理折扣要随官方计费口径更新。

单次任务的全可变成本可以写成：

```math
C_{\mathrm{var},i}=C_{\mathrm{model},i}+C_{\mathrm{rag},i}+C_{\mathrm{tool},i}+C_{\mathrm{review},i}+C_{\mathrm{retry},i}+C_{\mathrm{risk},i}
```

其中，$C_{\mathrm{rag}}$ 包含 embedding、检索、rerank、向量库和引用校验，$C_{\mathrm{tool}}$ 是工具调用，$C_{\mathrm{review}}$ 是人工审核，$C_{\mathrm{retry}}$ 是重试，$C_{\mathrm{risk}}$ 是预期失败成本。

月度总成本是：

```math
C_{\mathrm{month},i}=N_iA_iC_{\mathrm{var},i}+F_i
```

其中，$F_i$ 包括研发摊销、运维、监控、日志、合规、安全评估、数据更新和固定基础设施。很多 ROI 误判来自只算 $C_{\mathrm{model}}$，不算 $F_i$。

单次收益可以粗略拆成节省时间、质量提升、收入增长和风险下降：

```math
B_{\mathrm{task},i}=A_i(T_iW_i+Q_iV_i+G_i)+R_i
```

其中，$G_i$ 是单次增量收入，$R_i$ 是单次任务对应的预期风险损失下降。注意 $A_i$ 是采用率：模型生成了结果但用户不采用，不能直接算成收益。

月度收益是：

```math
B_{\mathrm{month},i}=N_iB_{\mathrm{task},i}
```

净收益、收益成本比和净 ROI 分别是：

```math
P_{\mathrm{net},i}=B_{\mathrm{month},i}-C_{\mathrm{month},i}
```

```math
R_{\mathrm{bc},i}=\frac{B_{\mathrm{month},i}}{C_{\mathrm{month},i}}
```

```math
R_{\mathrm{roi},i}=\frac{B_{\mathrm{month},i}-C_{\mathrm{month},i}}{C_{\mathrm{month},i}}
```

面试中要先说明自己采用哪个口径：$R_{\mathrm{bc}}$ 是收益成本比，$R_{\mathrm{roi}}$ 是扣除成本后的净 ROI。两者都可以用，但不能混着讲。

如果有一次性投入 $U_i$，回本周期可以写成：

```math
P_{\mathrm{back},i}=\frac{U_i}{\max(P_{\mathrm{net},i},\epsilon)}
```

其中，$\epsilon$ 是避免除零的极小正数。真实表达时如果净收益小于等于 0，就不要报一个看似精确的回本月数，而应该说“当前假设下无法回本”。

盈亏平衡任务量可以写成：

```math
N_{\mathrm{be},i}=\frac{F_i}{\max(B_{\mathrm{task},i}-A_iC_{\mathrm{var},i},\epsilon)}
```

如果单次毛利为负，规模越大亏得越多；这时不能靠“用户量增长”解决问题，必须先降成本、提高采用率或重新选场景。

敏感性分析至少要看三个变量：

```math
\Delta P_{\mathrm{net}}=\frac{\partial P_{\mathrm{net}}}{\partial A}\Delta A+\frac{\partial P_{\mathrm{net}}}{\partial Q}\Delta Q+\frac{\partial P_{\mathrm{net}}}{\partial C_{\mathrm{var}}}\Delta C_{\mathrm{var}}
```

直觉是：ROI 最容易被采用率、质量提升和可变成本打穿。一个看似盈利的方案，如果采用率下降、重试率上升或人审成本增加，很快就会变成亏损。

上线门禁可以写成：

```math
G_{\mathrm{roi},i}=\mathbb{1}(P_{\mathrm{net},i}>0,\ R_{\mathrm{bc},i}\ge \tau_b,\ P_{\mathrm{back},i}\le \tau_p,\ C_{\mathrm{var},i}<B_{\mathrm{task},i})
```

它表达的是：月度净收益要为正，收益成本比要过阈值，回本周期要可接受，且单次任务本身不能越做越亏。

## 4.17 成本优化手段

常见优化：

1. 小模型处理简单任务。
2. 大模型只处理复杂任务。
3. 缓存高频问题。
4. 控制上下文长度。
5. 优化 RAG top-k。
6. 减少无效重试。
7. 使用结构化输出减少后处理。
8. 对低价值请求降级。
9. 批处理离线任务。
10. 推理加速和量化。

成本优化不能牺牲关键体验和安全。要看 quality-cost trade-off。

## 4.18 ROI 的常见误区

1. 只算 API 成本，不算工程和人审。
2. 只算节省时间，不算错误成本。
3. 夸大自动化比例。
4. 忽略用户采用率。
5. 忽略长尾和失败处理。
6. 不做 A/B 测试。
7. 把所有业务增长都归因于 AI。
8. 没有持续监控成本。

ROI 估算是决策工具，不是包装项目的数字游戏。

## 4.19 面试题：如何估算大模型项目 ROI

回答要点：

```text
我会先明确收益来源，是节省人工、提升效率、增加收入还是降低风险。然后估算任务量、自动化比例、单次节省时间或增量收入，再计算模型调用、RAG、工具、人审、研发、运维和合规成本。最后看单位经济账和整体 ROI，并通过试点和线上数据校准假设。
```

## 4.20 面试题：如何降低大模型产品成本

回答要点：

```text
可以从模型、上下文、调用策略和系统架构优化。简单任务用小模型或规则，大模型只处理复杂任务；控制 prompt 和 RAG 上下文长度；缓存高频问题；减少无效重试；对 Agent 设置最大步骤和工具调用预算；离线任务用批处理；自部署场景可以考虑量化和推理加速。但优化要保证质量、安全和用户体验不被明显损害。
```

## 4.21 最小可运行 ROI / 单位经济账审计 demo

下面的 0 依赖 demo 演示一个教学版 ROI audit：输入 toy 场景的任务量、采用率、节省时间、质量提升、输入输出 token、RAG / 工具 / 人审 / 固定成本，输出单次可变成本、月收益、月成本、净收益、收益成本比、净 ROI、回本周期、盈亏平衡任务量、敏感性分析和 ROI 门禁。

```python
def safe_div(num, den, default=None):
    if den == 0:
        return default
    return num / den


def round_or_none(value, digits=3):
    return None if value is None else round(value, digits)


SCENARIOS = [
    {
        "name": "support_rag",
        "monthly_tasks": 4200,
        "adoption_rate": 0.62,
        "quality_uplift": 0.24,
        "time_saved_minutes": 4.5,
        "hourly_cost": 38,
        "value_per_quality_success": 5.0,
        "incremental_revenue_per_task": 0.8,
        "risk_reduction_per_task": 0.4,
        "model_calls": 1.2,
        "input_tokens": 1800,
        "output_tokens": 360,
        "cache_discount_rate": 0.25,
        "input_price_per_1k": 0.002,
        "output_price_per_1k": 0.008,
        "rag_cost_per_task": 0.035,
        "tool_cost_per_task": 0.0,
        "review_minutes": 0.4,
        "review_hourly_cost": 24,
        "retry_cost_per_task": 0.012,
        "risk_cost_per_task": 0.02,
        "fixed_monthly_cost": 4200,
        "upfront_cost": 9000,
        "latency_ok": True,
        "quality_gate": True,
    },
    {
        "name": "contract_review",
        "monthly_tasks": 90,
        "adoption_rate": 0.55,
        "quality_uplift": 0.18,
        "time_saved_minutes": 55,
        "hourly_cost": 130,
        "value_per_quality_success": 280,
        "incremental_revenue_per_task": 0.0,
        "risk_reduction_per_task": 18.0,
        "model_calls": 2.5,
        "input_tokens": 6200,
        "output_tokens": 900,
        "cache_discount_rate": 0.15,
        "input_price_per_1k": 0.004,
        "output_price_per_1k": 0.012,
        "rag_cost_per_task": 0.18,
        "tool_cost_per_task": 0.08,
        "review_minutes": 8.0,
        "review_hourly_cost": 95,
        "retry_cost_per_task": 0.10,
        "risk_cost_per_task": 0.35,
        "fixed_monthly_cost": 8200,
        "upfront_cost": 32000,
        "latency_ok": True,
        "quality_gate": True,
    },
    {
        "name": "code_agent",
        "monthly_tasks": 650,
        "adoption_rate": 0.38,
        "quality_uplift": 0.20,
        "time_saved_minutes": 22,
        "hourly_cost": 90,
        "value_per_quality_success": 40,
        "incremental_revenue_per_task": 0.0,
        "risk_reduction_per_task": 0.0,
        "model_calls": 5.0,
        "input_tokens": 4800,
        "output_tokens": 1200,
        "cache_discount_rate": 0.10,
        "input_price_per_1k": 0.004,
        "output_price_per_1k": 0.012,
        "rag_cost_per_task": 0.12,
        "tool_cost_per_task": 0.25,
        "review_minutes": 5.0,
        "review_hourly_cost": 80,
        "retry_cost_per_task": 0.35,
        "risk_cost_per_task": 0.25,
        "fixed_monthly_cost": 12000,
        "upfront_cost": 45000,
        "latency_ok": False,
        "quality_gate": False,
    },
    {
        "name": "generic_chatbot",
        "monthly_tasks": 300,
        "adoption_rate": 0.25,
        "quality_uplift": 0.04,
        "time_saved_minutes": 1.5,
        "hourly_cost": 35,
        "value_per_quality_success": 2.0,
        "incremental_revenue_per_task": 0.0,
        "risk_reduction_per_task": 0.0,
        "model_calls": 1.0,
        "input_tokens": 900,
        "output_tokens": 260,
        "cache_discount_rate": 0.05,
        "input_price_per_1k": 0.002,
        "output_price_per_1k": 0.008,
        "rag_cost_per_task": 0.0,
        "tool_cost_per_task": 0.0,
        "review_minutes": 0.2,
        "review_hourly_cost": 24,
        "retry_cost_per_task": 0.01,
        "risk_cost_per_task": 0.03,
        "fixed_monthly_cost": 2500,
        "upfront_cost": 6000,
        "latency_ok": True,
        "quality_gate": False,
    },
]


def model_cost(s):
    discounted_input = s["input_tokens"] * (1.0 - s["cache_discount_rate"])
    input_cost = discounted_input * s["input_price_per_1k"] / 1000.0
    output_cost = s["output_tokens"] * s["output_price_per_1k"] / 1000.0
    return s["model_calls"] * (input_cost + output_cost)


def variable_cost(s):
    review_cost = s["review_minutes"] / 60.0 * s["review_hourly_cost"]
    return (
        model_cost(s)
        + s["rag_cost_per_task"]
        + s["tool_cost_per_task"]
        + review_cost
        + s["retry_cost_per_task"]
        + s["risk_cost_per_task"]
    )


def benefit_per_task(s, adoption=None, quality=None):
    adoption = s["adoption_rate"] if adoption is None else adoption
    quality = s["quality_uplift"] if quality is None else quality
    time_value = s["time_saved_minutes"] / 60.0 * s["hourly_cost"]
    quality_value = quality * s["value_per_quality_success"]
    return adoption * (time_value + quality_value + s["incremental_revenue_per_task"]) + s["risk_reduction_per_task"]


def audit(s):
    var_cost = variable_cost(s)
    benefit_task = benefit_per_task(s)
    adopted_tasks = s["monthly_tasks"] * s["adoption_rate"]
    monthly_benefit = s["monthly_tasks"] * benefit_task
    monthly_cost = adopted_tasks * var_cost + s["fixed_monthly_cost"]
    net_benefit = monthly_benefit - monthly_cost
    bcr = safe_div(monthly_benefit, monthly_cost)
    roi = safe_div(net_benefit, monthly_cost)
    payback = safe_div(s["upfront_cost"], net_benefit) if net_benefit > 0 else None
    unit_margin = benefit_task - s["adoption_rate"] * var_cost
    break_even_tasks = safe_div(s["fixed_monthly_cost"], unit_margin) if unit_margin > 0 else None

    sensitivity = {}
    for delta in (-0.15, 0.0, 0.15):
        adjusted_adoption = max(0.0, min(1.0, s["adoption_rate"] + delta))
        adjusted_benefit = s["monthly_tasks"] * benefit_per_task(s, adoption=adjusted_adoption)
        adjusted_cost = s["monthly_tasks"] * adjusted_adoption * var_cost + s["fixed_monthly_cost"]
        sensitivity[f"adoption_{delta:+.2f}"] = round(adjusted_benefit - adjusted_cost, 2)
    for multiplier in (0.8, 1.0, 1.2):
        adjusted_cost = adopted_tasks * var_cost * multiplier + s["fixed_monthly_cost"]
        sensitivity[f"variable_cost_x{multiplier:.1f}"] = round(monthly_benefit - adjusted_cost, 2)

    failed = []
    if net_benefit <= 0:
        failed.append("net_benefit")
    if bcr is None or bcr < 1.25:
        failed.append("benefit_cost_ratio")
    if payback is None or payback > 6:
        failed.append("payback")
    if unit_margin <= 0:
        failed.append("unit_margin")
    if not s["latency_ok"]:
        failed.append("latency")
    if not s["quality_gate"]:
        failed.append("quality")

    return {
        "model_cost_per_task": round(model_cost(s), 4),
        "variable_cost_per_task": round(var_cost, 4),
        "benefit_per_task": round(benefit_task, 3),
        "unit_margin": round(unit_margin, 3),
        "monthly_benefit": round(monthly_benefit, 2),
        "monthly_cost": round(monthly_cost, 2),
        "net_benefit": round(net_benefit, 2),
        "benefit_cost_ratio": round_or_none(bcr),
        "roi": round_or_none(roi),
        "payback_months": round_or_none(payback),
        "break_even_tasks": round_or_none(break_even_tasks, 1),
        "sensitivity": sensitivity,
        "roi_gate": not failed,
        "failed_gates": failed,
    }


audits = {scenario["name"]: audit(scenario) for scenario in SCENARIOS}
ranked = sorted(
    ((name, result["net_benefit"], result["roi_gate"]) for name, result in audits.items()),
    key=lambda item: item[1],
    reverse=True,
)

print("ranked=", ranked)
print("roi_pass=", [name for name, result in audits.items() if result["roi_gate"]])
print("needs_rework=", {name: result["failed_gates"] for name, result in audits.items() if not result["roi_gate"]})
for name in [item[0] for item in ranked]:
    print(name, audits[name])
```

这段 demo 的关键结论是：`support_rag` 在 toy 假设下通过 ROI 门禁；`contract_review` 单次价值高，但固定成本和人审成本太重，回本周期不过线；`code_agent` 质量和延迟没过线，不能只看效率收益；`generic_chatbot` 任务价值太低，规模也不够，单位经济账和净收益都不成立。

## 4.22 本章小结

大模型项目要可持续，必须算成本收益。成本不仅是 token，还包括 GPU、RAG、Agent 工具、人审、数据、工程、运维、安全和失败成本。收益也不只是一句“提升效率”，而要尽量连接到人工节省、质量提升、收入增长或风险降低。

下一章会进入企业级 LLM 应用，讨论大模型在企业场景中如何接入知识库、权限系统、业务流程、审计合规和组织协作。
