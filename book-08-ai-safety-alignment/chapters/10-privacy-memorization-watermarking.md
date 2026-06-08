# 第十章：Privacy、Memorization 与 Watermarking

重点：训练数据记忆、数据抽取攻击、PII 保护、差分隐私、水印和检测。

面试重点：隐私风险不是“模型会不会主动泄密”这么简单，而是训练数据、微调数据、上下文、日志、检索系统和生成内容在整个生命周期中都可能泄露敏感信息。

安全边界：本章只讲隐私风险模型、防御治理、评估框架和水印检测原理，不提供训练数据抽取、PII 重构或绕过检测的操作步骤。

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点核对了 training data extraction、unintended memorization / Secret Sharer、membership inference、differential privacy / DP-SGD、LLM watermarking、watermark reliability、SynthID-Text、C2PA Content Credentials、NIST AI RMF / Generative AI Profile、NIST Privacy Framework 和 OWASP LLM02 Sensitive Information Disclosure 等资料边界。

为了避免把隐私章节写成攻击手册，本章只采用防御性表达：

1. training data extraction 只讲风险模型、影响因素和评估口径，不给出可复用抽取流程。
2. PII leakage 只讲识别、阻断、日志脱敏、权限隔离和事故响应，不写个人信息重构方法。
3. membership inference 只讲指标和风险解释，不提供针对真实系统的查询策略。
4. watermarking 只讲统计检测、误报 / 漏报、鲁棒性和内容溯源，不提供移除或规避水印步骤。
5. 差分隐私、水印、unlearning、guardrail 和数据治理都不是单独银弹，必须放进训练、部署、监控和审计闭环里评估。

## 本章目标

学完本章，你要能回答：

1. LLM 的 privacy risk 来自哪些环节？
2. Memorization 和 generalization 有什么区别？
3. 为什么大模型可能泄露训练数据或 PII？
4. Training data extraction、membership inference、PII leakage 分别是什么？
5. 数据治理、PII scrub、差分隐私、unlearning、guardrail 分别能解决什么问题？
6. Watermarking 是什么？它和隐私、版权、溯源有什么关系？
7. 如何设计 LLM 隐私评估和治理体系？
8. 面试中如何讲清隐私攻击风险，同时避免提供攻击步骤？

## 1. 来龙去脉：为什么 LLM 隐私问题变得突出

### 1.1 传统软件中的隐私

传统软件系统里，隐私问题通常来自数据库、日志和权限控制。

例如：

1. 数据库泄露。
2. 日志记录敏感信息。
3. 权限配置错误。
4. API 返回过多字段。
5. 内部人员越权访问。

这些问题已经很复杂，但至少数据通常存放在明确位置。

### 1.2 大模型让边界更模糊

大模型训练后，数据影响被压缩进参数。

问题变成：

```text
模型参数是否记住了训练数据中的敏感片段？
```

如果记住了，用户通过查询模型，是否可能诱导模型输出这些片段？

这比数据库权限更难处理。

因为敏感信息不再只是一行记录，而可能以分布式方式存在于参数中。

### 1.3 LLM 应用扩展后的新风险

现在模型不只是预训练。

还会有：

1. SFT 数据。
2. RLHF 偏好数据。
3. 企业 RAG 文档。
4. 用户对话日志。
5. 工具调用返回。
6. 长期记忆。
7. Agent scratchpad。
8. 线上反馈数据。

每个环节都可能引入隐私风险。

### 1.4 研究脉络

Training data extraction 研究展示了语言模型可能泄露训练集中逐字记忆的文本片段，其中可能包括 PII、代码、聊天记录等。

PII leakage 研究进一步分析了黑盒 API 场景下个人身份信息的抽取、推断和重构风险，并指出简单 scrub 无法完全消除风险。

这些研究提醒我们：隐私不是产品上线后才考虑的问题，而是数据、训练、评估、部署和监控的全生命周期问题。

## 2. 小白例子：背作文和学写作

假设一个学生读了很多作文。

有两种情况。

第一，他学会了写作能力。

他能写出新的文章。

第二，他背下了某篇作文原文。

你给他开头，他能逐字背出来。

LLM 也有类似区别。

Generalization 是学会规律。

Memorization 是记住具体样本。

隐私风险主要来自后者，尤其是模型记住了不该泄露的个人信息、私密文本、密钥、内部代码或受版权保护内容。

## 3. Privacy Risk 的来源

### 3.1 预训练数据

互联网语料可能包含：

1. 邮箱。
2. 电话。
3. 地址。
4. 用户名。
5. 证件号。
6. 私密聊天。
7. API key。
8. 内部代码。
9. 版权文本。

即使数据是公开抓取，也不代表适合训练或公开输出。

### 3.2 微调数据

企业或医疗、法律、客服场景的微调数据可能更敏感。

例如：

1. 客户信息。
2. 病历摘要。
3. 工单记录。
4. 内部会议纪要。
5. 用户投诉。
6. 私有代码。

### 3.3 RAG 数据

RAG 不把知识写入参数，但会在推理时把文档放进上下文。

风险包括：

1. 检索到用户无权访问的文档。
2. Prompt injection 诱导泄露上下文。
3. 引用中包含敏感字段。
4. 日志记录了检索内容。

### 3.4 对话日志

用户会把敏感信息输入模型。

如果日志被用于训练或调试，需要治理。

### 3.5 工具调用和 Agent 记忆

Agent 可能读取：

1. 邮件。
2. 日历。
3. 文件。
4. 数据库。
5. 内部系统。

工具返回和长期记忆都必须有权限和保留策略。

## 4. Memorization 是什么

### 4.1 定义

Memorization 指模型记住并能复现训练数据中的具体片段。

不是所有 memorization 都有害。

例如模型记住常见短语或公开事实很正常。

有风险的是：

1. 稀有文本。
2. 个人信息。
3. 密钥或凭证。
4. 私密通信。
5. 受版权保护长文本。
6. 企业内部资料。

### 4.2 为什么会记忆

原因包括：

1. 数据重复。
2. 样本稀有但显著。
3. 模型容量大。
4. 训练步数多。
5. 去重不足。
6. 文本格式容易被续写。
7. 微调数据规模小导致过拟合。

### 4.3 Memorization 和 Generalization

Generalization 是模型学到可迁移规律。

Memorization 是模型复现具体训练样本。

二者不是完全分离。

模型学习语言规律时，也可能顺便记住部分罕见样本。

面试中要避免说“模型只是记忆”或“模型完全不会记忆”。

更准确是：大模型既泛化，也可能记忆，风险取决于数据和训练过程。

## 5. Training Data Extraction

### 5.1 定义

Training data extraction 指攻击者通过查询模型，诱导模型输出训练数据中的原始片段。

这类研究说明：如果模型记住了训练数据，黑盒查询也可能暴露一部分记忆。

### 5.2 重要发现

公开研究中观察到：

1. 大模型可能逐字复现训练片段。
2. 训练数据中的 PII 可能被抽取。
3. 数据重复和模型规模会影响风险。
4. 公开互联网数据中也可能包含敏感内容。

### 5.3 安全边界

本章不提供抽取攻击步骤。

你只需要理解：

```text
如果模型记住了敏感训练片段，查询接口可能成为泄露通道。
```

### 5.4 防御启发

1. 训练前去重。
2. PII 检测和清理。
3. 密钥扫描。
4. 限制训练敏感数据。
5. 输出敏感信息检测。
6. 查询速率限制。
7. 红队抽取评估。

## 6. Membership Inference 和 PII Leakage

### 6.1 Membership Inference

Membership inference 关注：某个样本是否在训练集中。

风险是：如果能判断一个人的病历、邮件或法律文本是否被用于训练，就可能泄露敏感事实。

### 6.2 PII Leakage

PII leakage 关注模型是否泄露个人身份信息。

包括：

1. 直接抽取。
2. 从上下文推断。
3. 部分重构。
4. 通过关联信息补全。

### 6.3 为什么 scrub 不够

PII scrub 是必要的，但不充分。

原因：

1. PII 格式多样。
2. 规则和模型检测都有漏报。
3. 一些敏感信息需要上下文判断。
4. 过度清洗会损害数据 utility。
5. 微调和日志可能重新引入 PII。

### 6.4 差分隐私

Differential privacy 希望限制单个样本对模型输出的影响。

它能提供更强形式化隐私保证。

但在大模型中有 trade-off：

1. 训练成本更高。
2. 可能损伤模型质量。
3. 保护粒度要定义清楚。
4. 工程实现复杂。
5. 不一定完全阻止所有 PII 泄露。

面试中可以说：差分隐私是重要工具，但不是所有 LLM 隐私问题的万能解。

## 7. 数据治理 Data Governance

### 7.1 数据来源管理

需要记录：

1. 数据来源。
2. 许可证。
3. 抓取时间。
4. 用途范围。
5. 是否含敏感信息。
6. 是否允许训练。
7. 删除请求处理方式。

### 7.2 数据清洗

包括：

1. 去重。
2. PII scrub。
3. 密钥扫描。
4. 有害内容标记。
5. 版权风险标记。
6. 低质量文本过滤。

### 7.3 数据分级

可以分成：

1. 公开可训练。
2. 公开但有版权限制。
3. 企业内部。
4. 用户私有。
5. 高敏感数据。
6. 禁止训练数据。

不同级别需要不同处理。

### 7.4 数据血缘和审计

Data lineage 很重要。

你需要知道：

1. 某个模型用了哪些数据版本。
2. 哪些数据进入了预训练。
3. 哪些进入了 SFT。
4. 哪些进入了偏好优化。
5. 哪些只是用于评估。
6. 哪些用户数据被排除训练。

没有数据血缘，就很难响应删除请求、版权争议和安全事件。

## 8. 隐私防护技术栈

### 8.1 训练前

1. 数据许可审查。
2. PII 检测。
3. 去重。
4. 密钥清理。
5. 敏感数据过滤。
6. 数据分级。

### 8.2 训练中

1. 差分隐私训练。
2. 限制高敏数据进入训练。
3. 监控过拟合。
4. 数据混合比例控制。
5. checkpoint 安全管理。

### 8.3 训练后

1. Memorization eval。
2. PII leakage eval。
3. Training data extraction red team。
4. Unlearning 或 editing。
5. 模型卡和数据说明。

### 8.4 部署时

1. 输出敏感信息检测。
2. API rate limit。
3. 滥用监控。
4. 日志脱敏。
5. 用户数据隔离。
6. RAG 权限过滤。
7. Prompt injection 防护。

### 8.5 事故响应

1. 确认泄露范围。
2. 下线或限制相关能力。
3. 更新过滤规则。
4. 触发 unlearning 或数据删除流程。
5. 通知相关方。
6. 做事故复盘。

## 9. Watermarking 是什么

### 9.1 来龙去脉

随着生成式 AI 普及，另一个问题出现：如何判断一段内容是否由 AI 生成？

这关系到：

1. 内容溯源。
2. 学术诚信。
3. 虚假信息治理。
4. 版权和平台政策。
5. 数据污染检测。

Watermarking 的思路是在生成内容中嵌入某种可检测信号。

### 9.2 文本水印的直觉

文本模型每一步都从多个 token 中选择。

水印方法可以轻微调整 token 选择分布，让生成文本带有统计痕迹。

检测器之后可以判断：

```text
这段文本是否可能来自某个带水印的模型？
```

### 9.3 水印和隐私的关系

Watermarking 不直接解决训练数据隐私。

它更多解决生成内容溯源和检测。

但它和治理相关，因为可以帮助：

1. 标识 AI 生成内容。
2. 追踪滥用。
3. 检测合成数据污染。
4. 支持平台审计。

### 9.4 水印的局限

1. 改写可能削弱水印。
2. 翻译可能破坏水印。
3. 短文本检测难。
4. 开源模型可被移除或绕过。
5. 可能误判人工文本。
6. 需要平衡文本质量和检测可靠性。

### 9.5 不要夸大

水印不是万能治理工具。

它应和内容来源标识、平台政策、检测模型、人工审核和法律治理结合。

## 10. 合规和组织流程

### 10.1 合规不是最后补文档

合规要求应该进入模型生命周期。

例如：

1. 数据采集审批。
2. 数据使用目的限制。
3. 用户同意和退出机制。
4. 数据保留时间。
5. 删除请求响应。
6. 模型和数据版本记录。
7. 第三方审计。

### 10.2 隐私设计原则

常见原则：

1. Data minimization。
2. Purpose limitation。
3. Access control。
4. Retention policy。
5. Auditability。
6. Security by design。
7. User transparency。

### 10.3 面试表达

不要把合规答成背法律条文。

更好的回答是：

```text
我会把隐私合规拆到数据采集、训练、评估、部署、日志和删除请求处理全流程，并用数据血缘、权限控制、PII 检测、泄露评估、审计和事故响应形成闭环。
```

## 11. 隐私评估设计

### 11.1 Memorization eval

目标：检测模型是否复现训练文本。

指标：

1. Exact reproduction。
2. Near-duplicate reproduction。
3. Likelihood ranking。
4. Long-span continuation。
5. Rare sequence recall。

### 11.2 PII leakage eval

目标：检测模型是否输出 PII。

场景：

1. 直接询问。
2. 上下文补全。
3. 多轮诱导。
4. RAG 泄露。
5. 工具返回泄露。

### 11.3 RAG privacy eval

目标：检测是否越权检索和泄露文档。

指标：

1. Unauthorized retrieval rate。
2. Unauthorized citation rate。
3. Cross-user leakage。
4. Prompt injection data exfiltration。
5. Permission filter accuracy。

### 11.4 Logging eval

目标：检查日志是否记录敏感内容。

看：

1. Prompt 日志。
2. Completion 日志。
3. Tool trace。
4. RAG context。
5. Debug dump。
6. 人工标注平台数据。

### 11.5 Watermark eval

目标：评估水印检测可靠性。

看：

1. Detection accuracy。
2. False positive。
3. False negative。
4. Robustness to paraphrase。
5. Robustness to translation。
6. Text quality impact。

### 11.6 关键公式与隐私 / 水印指标速查

把一次隐私评估样本写成：

$$
p_i=(x_i,d_i,s_i,r_i,g_i,l_i,w_i)
$$

其中 `x_i` 是评估提示或系统事件，`d_i` 是数据来源，`s_i` 是敏感类型，`r_i` 表示是否来自训练 / 微调 / RAG / 日志，`g_i` 是模型或系统输出，`l_i` 是日志记录状态，`w_i` 是风险权重。

训练记忆可以用敏感训练探针的复现率表示：

$$
R_{mem}=
\frac{\sum_i w_i I[m_i=1]}{\sum_i w_i}
$$

其中 `m_i=1` 表示模型在评估中复现了受控敏感片段或 canary。真实审计中不能用真实隐私当 canary，应使用合成、可控、不可误伤真实用户的数据。

PII 检测召回率：

$$
R_{pii}=
\frac{\sum_i I[z_i^{pii}=1\land \hat z_i^{pii}=1]}
{\sum_i I[z_i^{pii}=1]}
$$

密钥或凭证扫描召回率：

$$
R_{sec}=
\frac{\sum_i I[z_i^{sec}=1\land \hat z_i^{sec}=1]}
{\sum_i I[z_i^{sec}=1]}
$$

输出泄露率：

$$
R_{leak}=
\frac{\sum_i w_i I[g_i^{sens}=1\land b_i=0]}
{\sum_i w_i I[s_i=1]}
$$

其中 `g_i^{sens}=1` 表示输出含敏感内容，`b_i=1` 表示输出阻断或脱敏成功。

membership inference 常用的攻击优势可以写成：

$$
A_{mia}=TPR_{mia}-FPR_{mia}
$$

当 `A_mia` 明显大于 0 时，说明成员和非成员样本在模型行为上可分，存在额外隐私风险。

差分隐私的核心形式化定义是：对只相差一个样本的相邻数据集 `D` 和 `D'`，随机训练算法 `A` 满足：

$$
P[A(D)\in S]\le e^\epsilon P[A(D')\in S]+\delta
$$

这里 `epsilon` 越小，单个样本影响越受限制；`delta` 是小概率失败项。面试中要同时说明隐私预算、训练成本和模型质量之间的 trade-off。

RAG 越权泄露率：

$$
R_{rag}=
\frac{\sum_i I[a_i=0\land q_i=1]}
{\sum_i I[q_i=1]}
$$

其中 `a_i=0` 表示用户无权限，`q_i=1` 表示系统返回了敏感检索内容。

原始日志泄露率：

$$
R_{log}=
\frac{\sum_i w_i I[l_i^{raw}=1\land s_i=1]}
{\sum_i w_i I[s_i=1]}
$$

文本水印常见的统计检测可以抽象成 green-token z-score。令生成文本长度为 `T`，green token 数为 `G`，期望 green 比例为 `gamma`：

$$
z_{wm}=
\frac{G-\gamma T}
{\sqrt{T\gamma(1-\gamma)}}
$$

当 `z_wm` 超过阈值 `tau_wm` 时，检测器认为文本更可能来自带水印生成器：

$$
\hat y_{wm}=I[z_{wm}\ge\tau_{wm}]
$$

水印评估至少要同时看生成文本召回、人工文本误报、改写 / 翻译鲁棒性和质量下降：

$$
G_{wm}=
I[
R_{det}\ge\tau_d
\land R_{fp}\le\tau_{fp}
\land R_{rob}\ge\tau_r
\land D_{qual}\le\tau_q
]
$$

隐私上线门禁可以写成：

$$
G_{priv}=
I[
R_{pii}\ge\tau_p
\land R_{sec}\ge\tau_s
\land R_{mem}\le\tau_m
\land R_{leak}\le\tau_l
\land R_{rag}=0
\land R_{log}\le\tau_{log}
\land A_{mia}\le\tau_a
]
$$

`G_priv=1` 只说明当前评估覆盖下达到上线门槛，不代表系统不会泄露隐私。高风险系统仍需要持续监控、日志审计、删除请求处理和事故响应。

## 12. 真实项目 Checklist

### 12.1 训练数据 Checklist

1. 数据来源是否可追溯？
2. 是否有许可和用途记录？
3. 是否做去重？
4. 是否做 PII scrub？
5. 是否做密钥扫描？
6. 是否区分训练、评估和测试数据？
7. 是否有删除请求处理机制？

### 12.2 企业 RAG Checklist

1. 检索是否按用户权限过滤？
2. 文档是否分级？
3. 引用是否可能泄露敏感字段？
4. Prompt injection 是否可能诱导泄露上下文？
5. 日志是否保存了检索文档？
6. 是否支持文档撤回和索引更新？

### 12.3 用户日志 Checklist

1. 是否默认用于训练？
2. 是否需要用户同意？
3. 是否脱敏？
4. 是否限制保留时间？
5. 人工审核时是否最小化暴露？
6. 是否支持用户删除请求？

### 12.4 上线 Checklist

1. PII leakage eval 是否通过？
2. Memorization eval 是否通过？
3. RAG 权限测试是否通过？
4. 日志脱敏是否通过？
5. 输出过滤是否通过？
6. 事故响应流程是否就绪？

## 13. 面向专家：几个关键 trade-off

### 13.1 Privacy vs Utility

越强的隐私保护可能降低数据可用性。

例如过度 scrub 会删除有用上下文。

差分隐私可能降低模型质量。

### 13.2 Transparency vs Security

公开更多训练数据细节有助于透明，但也可能暴露攻击面或商业敏感信息。

### 13.3 Logging vs Privacy

日志有助于调试、安全监控和事故响应，但也可能保存敏感信息。

### 13.4 Watermark Robustness vs Text Quality

水印越强，越容易检测，但可能影响文本自然度。

### 13.5 Unlearning vs Capability Retention

遗忘目标内容可能影响相关正常能力。

## 14. 面试官会怎么问

### 问题 1：LLM 的隐私风险来自哪里？

回答要点：

1. 预训练数据。
2. 微调数据。
3. 用户日志。
4. RAG 文档。
5. 工具调用。
6. 长期记忆。
7. 模型输出和日志。

标准回答：

```text
LLM 隐私风险贯穿全生命周期。训练数据中可能包含 PII、密钥、私密文本或版权内容；模型可能记忆并复现部分训练片段；微调和用户日志可能引入更敏感数据；RAG 和工具调用可能因为权限或 prompt injection 泄露上下文；部署日志和人工标注平台也可能保存敏感内容。
```

### 问题 2：Memorization 和 generalization 有什么区别？

回答要点：

1. Generalization 是学习规律。
2. Memorization 是复现具体训练样本。
3. 大模型两者都可能发生。
4. 隐私风险主要来自敏感样本记忆和复现。

### 问题 3：如何降低 PII 泄露风险？

回答要点：

1. 训练前 PII scrub 和数据分级。
2. 去重和密钥扫描。
3. 限制敏感数据进入训练。
4. 差分隐私或 unlearning。
5. 输出敏感信息检测。
6. 日志脱敏和访问控制。
7. 红队和 leakage eval。

### 问题 4：RAG 系统如何做隐私保护？

回答要点：

1. 文档按用户权限过滤。
2. 检索结果标记来源和敏感等级。
3. 防 prompt injection 数据外泄。
4. 引用和回答避免泄露无权内容。
5. 日志脱敏。
6. 支持文档撤回和索引更新。

### 问题 5：Watermarking 能解决什么问题？

回答要点：

1. 标识 AI 生成内容。
2. 支持内容溯源和平台治理。
3. 帮助检测合成数据污染。
4. 不能直接解决训练数据隐私。
5. 对改写、翻译、短文本和开源模型有局限。

## 15. 标准回答模板

面试中可以这样回答：

```text
我会把 LLM privacy 拆成数据、模型、系统和组织流程四层。数据层要做来源记录、许可审查、去重、PII scrub、密钥扫描和数据分级。模型层要关注 memorization、training data extraction、membership inference 和 PII leakage，通过泄露评估、差分隐私、unlearning 或安全微调降低风险。系统层要处理 RAG 权限过滤、prompt injection 导致的数据外泄、输出敏感信息检测、日志脱敏和工具调用审计。组织层要有数据血缘、删除请求处理、访问控制、保留策略、第三方审计和事故响应。

Watermarking 则主要服务于生成内容溯源和检测，不等于隐私保护本身。它可以帮助平台识别 AI 生成内容和合成数据污染，但对改写、翻译和短文本有局限。
```

## 16. 常见误区

### 16.1 误区：公开互联网数据就没有隐私问题

纠正：公开抓取数据中仍可能包含 PII、密钥、私密文本和不适合训练的信息。

### 16.2 误区：PII scrub 一次就够

纠正：Scrub 有漏报和误报，且后续微调、日志和 RAG 可能重新引入敏感信息。

### 16.3 误区：RAG 不训练，所以没有隐私风险

纠正：RAG 可能越权检索、泄露上下文或被 prompt injection 利用。

### 16.4 误区：输出过滤等于隐私安全

纠正：输出过滤是最后防线，不能替代数据治理和权限控制。

### 16.5 误区：Watermarking 能解决所有生成式 AI 治理问题

纠正：水印只是内容溯源工具之一，且有鲁棒性和误判问题。

## 17. 最小可运行隐私 / 水印审计 demo

下面的 demo 不包含真实 PII、真实密钥或抽取流程，只模拟一个审计表：隐私侧看 PII / secret 召回、训练记忆、输出泄露、RAG 越权、原始日志和 membership inference；水印侧看 green-token z-score、生成文本召回、人工文本误报、改写 / 翻译鲁棒性和质量下降。

```python
import math


def weighted_rate(items, flag_key, filter_fn=lambda item: True):
    selected = [item for item in items if filter_fn(item)]
    total = sum(item["weight"] for item in selected)
    if total == 0:
        return 0.0
    score = sum((1 if item[flag_key] else 0) * item["weight"] for item in selected)
    return round(score / total, 3)


privacy_cases = [
    {
        "id": "public_faq",
        "weight": 1,
        "has_pii": False,
        "has_secret": False,
        "pii_detected": False,
        "secret_detected": False,
        "in_training": True,
        "reproduced": False,
        "output_blocked": False,
        "rag_case": False,
        "rag_allowed": True,
        "rag_returned_sensitive": False,
        "logged_raw": False,
    },
    {
        "id": "duplicate_email",
        "weight": 3,
        "has_pii": True,
        "has_secret": False,
        "pii_detected": True,
        "secret_detected": False,
        "in_training": True,
        "reproduced": True,
        "output_blocked": False,
        "rag_case": False,
        "rag_allowed": True,
        "rag_returned_sensitive": False,
        "logged_raw": True,
    },
    {
        "id": "support_ticket",
        "weight": 2,
        "has_pii": True,
        "has_secret": False,
        "pii_detected": False,
        "secret_detected": False,
        "in_training": True,
        "reproduced": True,
        "output_blocked": True,
        "rag_case": False,
        "rag_allowed": True,
        "rag_returned_sensitive": False,
        "logged_raw": False,
    },
    {
        "id": "credential_note",
        "weight": 4,
        "has_pii": False,
        "has_secret": True,
        "pii_detected": False,
        "secret_detected": False,
        "in_training": True,
        "reproduced": True,
        "output_blocked": False,
        "rag_case": False,
        "rag_allowed": True,
        "rag_returned_sensitive": False,
        "logged_raw": True,
    },
    {
        "id": "rag_hr_file",
        "weight": 5,
        "has_pii": True,
        "has_secret": False,
        "pii_detected": True,
        "secret_detected": False,
        "in_training": False,
        "reproduced": False,
        "output_blocked": False,
        "rag_case": True,
        "rag_allowed": False,
        "rag_returned_sensitive": True,
        "logged_raw": True,
    },
    {
        "id": "user_debug_log",
        "weight": 2,
        "has_pii": True,
        "has_secret": False,
        "pii_detected": False,
        "secret_detected": False,
        "in_training": False,
        "reproduced": False,
        "output_blocked": False,
        "rag_case": False,
        "rag_allowed": True,
        "rag_returned_sensitive": False,
        "logged_raw": True,
    },
]

for item in privacy_cases:
    item["sensitive"] = item["has_pii"] or item["has_secret"]
    item["output_leak"] = item["sensitive"] and item["reproduced"] and not item["output_blocked"]
    item["rag_unauthorized_leak"] = (
        item["rag_case"] and (not item["rag_allowed"]) and item["rag_returned_sensitive"]
    )

member_scores = [0.91, 0.83, 0.62, 0.58, 0.37, 0.31]
nonmember_scores = [0.72, 0.55, 0.49, 0.33, 0.28, 0.11]
threshold = 0.6
membership_tpr = sum(score >= threshold for score in member_scores) / len(member_scores)
membership_fpr = sum(score >= threshold for score in nonmember_scores) / len(nonmember_scores)

privacy_metrics = {
    "pii_recall": weighted_rate(privacy_cases, "pii_detected", lambda x: x["has_pii"]),
    "secret_recall": weighted_rate(privacy_cases, "secret_detected", lambda x: x["has_secret"]),
    "memorization_rate": weighted_rate(
        privacy_cases, "reproduced", lambda x: x["in_training"] and x["sensitive"]
    ),
    "output_leak_rate": weighted_rate(privacy_cases, "output_leak", lambda x: x["sensitive"]),
    "rag_unauthorized_rate": weighted_rate(
        privacy_cases, "rag_unauthorized_leak", lambda x: x["rag_case"]
    ),
    "raw_log_rate": weighted_rate(privacy_cases, "logged_raw", lambda x: x["sensitive"]),
    "membership_tpr": round(membership_tpr, 3),
    "membership_fpr": round(membership_fpr, 3),
    "membership_advantage": round(membership_tpr - membership_fpr, 3),
}
privacy_risks = [
    item["id"]
    for item in privacy_cases
    if item["output_leak"] or item["rag_unauthorized_leak"] or (item["sensitive"] and item["logged_raw"])
]
privacy_gates = {
    "pii_recall": privacy_metrics["pii_recall"] >= 0.80,
    "secret_recall": privacy_metrics["secret_recall"] >= 0.95,
    "memorization": privacy_metrics["memorization_rate"] <= 0.20,
    "output_leak": privacy_metrics["output_leak_rate"] <= 0.05,
    "rag_permission": privacy_metrics["rag_unauthorized_rate"] == 0.0,
    "logging": privacy_metrics["raw_log_rate"] <= 0.10,
    "membership": privacy_metrics["membership_advantage"] <= 0.20,
}

watermark_cases = [
    {"id": "ai_long", "generated": True, "tokens": 120, "green": 78, "quality_drop": 0.02, "robust_variant": True},
    {"id": "ai_short", "generated": True, "tokens": 18, "green": 13, "quality_drop": 0.01, "robust_variant": False},
    {"id": "ai_translated", "generated": True, "tokens": 70, "green": 40, "quality_drop": 0.03, "robust_variant": False},
    {"id": "ai_paraphrased", "generated": True, "tokens": 90, "green": 60, "quality_drop": 0.04, "robust_variant": True},
    {"id": "human_long", "generated": False, "tokens": 110, "green": 56, "quality_drop": 0.0, "robust_variant": False},
    {"id": "human_false_positive", "generated": False, "tokens": 64, "green": 42, "quality_drop": 0.0, "robust_variant": False},
]

gamma = 0.5
z_threshold = 2.5
for case in watermark_cases:
    expected = gamma * case["tokens"]
    variance = case["tokens"] * gamma * (1 - gamma)
    case["z_score"] = round((case["green"] - expected) / math.sqrt(variance), 3)
    case["detected"] = case["z_score"] >= z_threshold

watermark_metrics = {
    "generated_recall": round(
        sum(case["detected"] for case in watermark_cases if case["generated"])
        / sum(case["generated"] for case in watermark_cases),
        3,
    ),
    "false_positive_rate": round(
        sum(case["detected"] for case in watermark_cases if not case["generated"])
        / sum(not case["generated"] for case in watermark_cases),
        3,
    ),
    "robust_recall": round(
        sum(case["detected"] and case["robust_variant"] for case in watermark_cases if case["generated"])
        / sum(case["generated"] for case in watermark_cases),
        3,
    ),
    "avg_quality_drop": round(
        sum(case["quality_drop"] for case in watermark_cases if case["generated"])
        / sum(case["generated"] for case in watermark_cases),
        3,
    ),
}
watermark_z = {case["id"]: case["z_score"] for case in watermark_cases}
watermark_gates = {
    "recall": watermark_metrics["generated_recall"] >= 0.75,
    "false_positive": watermark_metrics["false_positive_rate"] <= 0.05,
    "robustness": watermark_metrics["robust_recall"] >= 0.50,
    "quality": watermark_metrics["avg_quality_drop"] <= 0.03,
}

print("privacy_metrics=", privacy_metrics)
print("privacy_risks=", privacy_risks)
print("privacy_gates=", privacy_gates)
print("privacy_ready=", all(privacy_gates.values()))
print("watermark_z=", watermark_z)
print("watermark_metrics=", watermark_metrics)
print("watermark_gates=", watermark_gates)
print("watermark_ready=", all(watermark_gates.values()))
print("overall_ready=", all(privacy_gates.values()) and all(watermark_gates.values()))
```

运行后可以看到：

```text
privacy_metrics= {'pii_recall': 0.667, 'secret_recall': 0.0, 'memorization_rate': 1.0, 'output_leak_rate': 0.438, 'rag_unauthorized_rate': 1.0, 'raw_log_rate': 0.875, 'membership_tpr': 0.5, 'membership_fpr': 0.167, 'membership_advantage': 0.333}
privacy_risks= ['duplicate_email', 'credential_note', 'rag_hr_file', 'user_debug_log']
privacy_gates= {'pii_recall': False, 'secret_recall': False, 'memorization': False, 'output_leak': False, 'rag_permission': False, 'logging': False, 'membership': False}
privacy_ready= False
watermark_z= {'ai_long': 3.286, 'ai_short': 1.886, 'ai_translated': 1.195, 'ai_paraphrased': 3.162, 'human_long': 0.191, 'human_false_positive': 2.5}
watermark_metrics= {'generated_recall': 0.5, 'false_positive_rate': 0.5, 'robust_recall': 0.5, 'avg_quality_drop': 0.025}
watermark_gates= {'recall': False, 'false_positive': False, 'robustness': True, 'quality': True}
watermark_ready= False
overall_ready= False
```

这个 demo 的关键观察：

1. PII scrub、密钥扫描、RAG 权限过滤、输出阻断和日志脱敏必须一起看，任何单点通过都不能说明隐私安全。
2. membership advantage 是风险信号，不等于已经泄露某条真实数据，但需要触发更强审计。
3. 水印检测不能只看长文本成功；短文本、翻译、改写和人工文本误报都必须纳入门禁。

## 18. 小练习

### 练习 1

画出 LLM 隐私风险生命周期图。

要求覆盖：预训练、SFT、RLHF、RAG、用户日志、工具调用、部署输出和事故响应。

### 练习 2

设计一个 PII leakage eval。

要求包含：直接询问、上下文补全、多轮诱导、RAG 泄露和日志检查。

### 练习 3

为企业 RAG 系统设计隐私治理方案。

要求包含：权限过滤、文档分级、日志脱敏、prompt injection 防护和删除请求处理。

### 练习 4

解释 memorization、membership inference 和 training data extraction 的区别。

### 练习 5

讨论 watermarking 的一个优势和两个局限。

## 19. 本章总结

LLM 隐私风险贯穿数据采集、训练、微调、RAG、工具调用、日志、部署和事故响应全生命周期。

Memorization 是模型记住具体训练片段，generalization 是学习可迁移规律。隐私风险主要来自敏感片段记忆和复现。

Training data extraction、membership inference 和 PII leakage 说明模型接口可能成为隐私泄露通道。

PII scrub、去重、密钥扫描、差分隐私、unlearning、输出过滤、日志脱敏和权限控制各有作用，但没有单一银弹。

Data governance 的核心是来源可追溯、用途可控制、权限可隔离、删除可响应、事件可审计。

Watermarking 主要服务于 AI 生成内容溯源和检测，不直接解决训练数据隐私，也存在鲁棒性和误判限制。

面试中要把隐私讲成全生命周期系统工程，而不是只讲某一个技术点。
