# 第六章：SFT 与对齐训练坑

后训练是把基础模型变成可用助手的关键步骤，但它不是简单“让模型更好”。SFT、偏好优化、reward model、RLHF、DPO 都在改变模型行为分布。一个模型 SFT 后可能更会聊天，但数学、代码、长上下文或事实能力下降；安全数据加多了可能过度拒答；reward model 偏好长答案，RLHF 就可能把模型训成话多但不准。

本章系统讲 SFT 与对齐训练坑：能力退化、格式错误、过度拒答、偏好数据不一致、reward model 偏差、RLHF reward hacking、DPO 数据质量、多轮对话角色混乱、评估方法和事故复盘。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，按 `WRITING_PLAN.md` 的要求核对了 InstructGPT / RLHF 论文、Direct Preference Optimization 论文、Hugging Face Transformers chat template 文档、TRL SFTTrainer / DPOTrainer 文档，以及 reward model、preference data、assistant-only loss、KL / reference model 和安全拒答边界相关公开资料。

本章只讨论防御性的后训练事故排查：如何发现 SFT mask 错、能力回归、误拒 / 漏拒、偏好 pair 噪声、reward model 偏差、DPO reference 不匹配、工具 schema 漂移和多维评估缺口。这里不提供绕过安全策略、构造攻击性对齐数据、诱导模型输出高风险内容或规避评估门禁的方法。

第二轮补强重点有三点：

1. 把 SFT / RLHF / DPO 的事故从“效果怪”拆成可观测指标：assistant-only label 覆盖率、prompt loss 泄漏率、能力回归、误拒率、漏拒率、偏好间隔、reward-human gap、DPO margin 和工具 schema 一致性。
2. 用公式说明后训练门禁，而不是只说“多做评估”：对齐训练要同时证明模型会回答、不会乱拒、不会漏拒、不会为了 reward 变长变空，也没有把通用能力训坏。
3. 增加一个 0 依赖 Python demo，把 toy SFT 样本、偏好 pair、reward 样本、DPO log probability 和工具 schema 统一审计，帮助读者把排查思路迁移到真实 TRL / LoRA / RLHF / DPO 项目。

## 6.1 核心观点

后训练不是单向提升，而是在做行为分布重塑。

它可能提升：

1. 指令跟随。
2. 对话风格。
3. 安全性。
4. 工具调用格式。
5. 用户偏好对齐。

也可能损伤：

1. 基础知识。
2. 数学推理。
3. 代码能力。
4. 输出多样性。
5. 简洁性。
6. 真实任务鲁棒性。

面试回答：

```text
SFT 和对齐训练不是简单提升模型，而是在改变模型输出分布。排查时不能只看聊天体验是否变好，还要分能力评估基础任务、数学、代码、长上下文、安全拒答率、输出长度和风格变化。很多后训练事故来自数据配比、格式 mask、偏好标注不一致和 reward model 偏差。
```

## 6.2 常见问题

1. SFT 后模型变得更会聊天但基础能力下降。
2. 过多安全数据导致过度拒答。
3. 偏好数据标注不一致导致 DPO 学到奇怪风格。
4. reward model 偏好长答案。
5. RLHF 优化过度导致 reward hacking。
6. 多轮对话格式错误导致角色混乱。

还包括：

1. SFT 数据模板化导致回答同质化。
2. prompt 部分未 mask 导致复述用户输入。
3. chosen/rejected 差异太小，偏好训练信号噪声大。
4. 安全策略和有用性冲突。
5. 对齐后模型变啰嗦。
6. 工具调用数据格式和线上 schema 不一致。

## 6.3 SFT 后基础能力下降

现象：

1. 聊天更自然。
2. 但数学题变差。
3. 代码生成变差。
4. 事实问答变差。
5. 输出风格更模板化。

原因：

1. SFT 数据质量低。
2. SFT 数据覆盖太窄。
3. 学习率过大导致遗忘。
4. 训练步数过多。
5. 指令数据挤压基础能力。
6. 数据格式和预训练分布差异太大。

排查：

1. 分能力评估 SFT 前后。
2. 看数学、代码、知识、对话、安全分别变化。
3. 检查 SFT 数据来源和比例。
4. 对比不同训练步数 checkpoint。
5. 小学习率和少 epoch ablation。

## 6.4 Catastrophic Forgetting

灾难性遗忘指后训练让模型忘掉原本能力。

常见场景：

1. 小规模领域 SFT 后通用能力下降。
2. 安全数据训练后普通帮助性下降。
3. 工具调用 SFT 后自然对话变差。
4. 角色扮演数据过多后风格固定。

缓解：

1. 混入通用高质量数据。
2. 控制学习率。
3. 控制训练步数。
4. 使用 LoRA 等参数高效方法时监控通用能力。
5. 分 checkpoint 选择。
6. 做多维回归评估。

后训练不是越久越好。很多时候最佳 checkpoint 在中间。

## 6.5 过度拒答

安全数据过多或标注过严，会导致模型过度拒答。

表现：

1. 普通问题也拒答。
2. 低风险问题给安全免责声明。
3. 回答变保守。
4. 用户体验下降。
5. 任务完成率下降。

排查：

1. 统计拒答率。
2. 分安全类别看拒答。
3. 抽查误拒样本。
4. 检查安全数据和有用性数据比例。
5. 检查标注规则是否过宽。

安全训练要区分危险请求、敏感请求、正常请求，不应把所有边界问题都训练成拒答。

## 6.6 安全和有用性的冲突

对齐训练常在安全和有用性之间权衡。

坏策略：

```text
只要有风险词，就全部拒答
```

好策略：

1. 拒绝危险操作细节。
2. 提供安全替代帮助。
3. 对不确定场景澄清。
4. 对高风险建议加人审和限制。
5. 对正常教育和防护场景给有用答案。

例如网络安全问题，攻击实施细节要拒绝，但防护建议和安全教育可以回答。

## 6.7 Prompt Loss Mask 错误

SFT 中，如果 prompt 部分参与 loss，模型会学会复述用户输入。

表现：

1. 回答前重复问题。
2. 多轮对话角色混乱。
3. 模型生成 user 标记。
4. 指令跟随变差。

排查：

1. 打印 labels。
2. 检查 system/user 是否为 -100。
3. 检查 assistant 部分是否参与 loss。
4. 检查多轮每个 assistant turn。

这是 SFT 最常见、也最不应该发生的事故之一。

## 6.8 多轮对话格式错误

多轮对话 SFT 容易出现 role 错误。

问题：

1. user/assistant 顺序错。
2. system prompt 缺失。
3. 多轮答案边界不清。
4. EOS 缺失。
5. tool observation 混成 assistant。

表现：

1. 模型替用户说话。
2. 模型自问自答。
3. 生成 role token。
4. 工具调用后不使用 observation。

排查方式：decode 多轮训练样本，逐 token 看 role 边界和 labels。

## 6.9 偏好数据标注不一致

DPO/RLHF 依赖偏好数据。标注不一致会让模型学到混乱信号。

常见不一致：

1. 有的标注者偏好长答案。
2. 有的偏好简洁答案。
3. 有的重视安全。
4. 有的重视有用性。
5. 有的按格式打分。
6. 有的按事实正确性打分。

表现：

1. 模型风格不稳定。
2. 输出变长但质量不升。
3. 安全策略不一致。
4. 对某些任务偏好奇怪格式。

排查：

1. 看标注一致性。
2. 分标注者分析偏好。
3. 抽查 chosen/rejected。
4. 检查偏好维度是否清晰。

## 6.10 Chosen/Rejected 差异太小

偏好训练需要有明确质量差异。

如果 chosen 和 rejected 都差不多，训练信号就是噪声。

问题：

1. 模型学到随机偏好。
2. 对细微格式过拟合。
3. 训练不稳定。
4. 泛化差。

排查：

1. 人工抽查 pair。
2. 统计长度差异。
3. 统计事实错误差异。
4. 检查 chosen 是否真的更好。
5. 删除低置信 pair。

偏好数据质量比数量更重要。

## 6.11 Reward Model 偏差

Reward model 容易学到捷径。

常见偏差：

1. 偏好长答案。
2. 偏好格式完整。
3. 偏好礼貌语气。
4. 忽略事实错误。
5. 对某类任务评分不准。
6. 被模板化回答欺骗。

排查：

1. reward score 和人工偏好相关性。
2. 分任务看相关性。
3. 控制长度做对比。
4. 找高 reward 低质量样本。
5. 做 adversarial eval。

如果 reward model 偏了，RLHF 会放大偏差。

## 6.12 RLHF Reward Hacking

Reward hacking 指模型优化了奖励漏洞，而不是真实质量。

表现：

1. 答案越来越长。
2. 模板化严重。
3. 安全免责声明泛滥。
4. 看起来有条理但事实错。
5. reward 升高但人工评价下降。

缓解：

1. 限制 KL 偏移。
2. 加强人工评估。
3. 多维 reward。
4. 控制输出长度。
5. 定期检查高 reward 样本。
6. 保留 SFT baseline 对照。

RLHF 的目标不是 reward 分数越高越好，而是人类真实偏好和任务成功率更好。

## 6.13 DPO 常见坑

DPO 比 RLHF 简洁，但不代表没有坑。

常见问题：

1. 数据 pair 质量差。
2. chosen/rejected 差异不明确。
3. beta 设置不合适。
4. reference model 不匹配。
5. 训练后输出风格变窄。
6. 偏好数据覆盖不全。

排查：

1. 和 SFT baseline 对比。
2. 分任务看提升和退化。
3. 看输出长度变化。
4. 抽查偏好 pair。
5. 调 beta 做 ablation。

DPO 不是“无脑比 SFT 好”，它高度依赖偏好数据。

## 6.14 输出长度漂移

后训练后输出常变长。

原因：

1. 标注者偏好长答案。
2. reward model 偏好长答案。
3. SFT 数据答案过长。
4. 安全免责声明过多。

影响：

1. 成本上升。
2. 用户体验下降。
3. 关键信息被稀释。
4. benchmark 可能虚高。

要监控平均输出长度、不同任务长度、用户采纳率和人工偏好。

## 6.15 风格漂移

后训练可能让模型风格变奇怪。

表现：

1. 过度礼貌。
2. 过度免责声明。
3. 喜欢分点但内容空。
4. 创造力下降。
5. 所有回答像同一个模板。

原因通常是 SFT 数据模板化或偏好标注偏格式。

风格要作为评估维度，而不是只看准确率。

## 6.16 工具调用对齐坑

工具调用训练常见问题：

1. tool schema 训练和线上不一致。
2. tool observation 格式不一致。
3. 模型学会过度调用工具。
4. 该调用工具时不调用。
5. 工具错误恢复数据缺失。
6. 多工具顺序混乱。

评估要看：

1. 工具选择准确率。
2. 参数准确率。
3. 调用成功率。
4. 失败恢复率。
5. 不必要工具调用率。

## 6.17 后训练评估

后训练评估必须多维。

维度：

1. 指令跟随。
2. 基础知识。
3. 数学推理。
4. 代码能力。
5. 安全拒答。
6. 误拒率。
7. 输出长度。
8. 风格。
9. 工具调用。
10. 多轮对话。
11. 真实用户任务。

只看一个 chat benchmark 很危险。

## 6.18 排查清单

1. 分能力评估 SFT 前后变化。
2. 检查 prompt loss 是否被 mask。
3. 检查 chosen/rejected 是否真的有质量差异。
4. 分析 reward score 和人工偏好的相关性。
5. 检查输出长度、拒答率和风格变化。

扩展清单：

1. 对比 SFT 前后数学、代码、知识能力。
2. 检查 SFT 数据来源和模板比例。
3. 检查安全数据比例和误拒样本。
4. 检查多轮 role 和 EOS。
5. 检查偏好标注一致性。
6. 检查 reward 高分低质样本。
7. 检查 DPO beta 和 reference model。
8. 检查工具调用 schema。
9. 检查平均输出长度。
10. 做真实用户任务抽检。

## 6.19 修复策略

能力退化：

1. 降低学习率。
2. 减少训练步数。
3. 混入通用高质量数据。
4. 选择更早 checkpoint。

过度拒答：

1. 增加正常安全边界样本。
2. 区分危险请求和正常请求。
3. 优化安全标注规则。
4. 降低安全数据过高权重。

偏好数据噪声：

1. 清洗低一致性 pair。
2. 统一标注 rubric。
3. 分任务训练或加权。
4. 加强人工抽检。

reward hacking：

1. 加人工评估。
2. 控制 KL。
3. 增加多维 reward。
4. 监控长度和模板化。

## 6.19.1 关键公式与后训练事故指标速查

后训练事故排查的第一步，是把样本、标签、偏好、reward 和评估切片放进同一张表。把第 `i` 条后训练样本抽象成：

$$
a_i=(x_i,y_i,m_i,g_i,r_i,c_i)
$$

其中，`x_i` 是 system / user / context 条件，`y_i` 是 assistant 目标回答，`m_i` 是 label mask，`g_i` 是 gold 或人工质量标签，`r_i` 是 reward / judge / verifier 代理分数，`c_i` 是样本类别，例如 math、code、safety、tool 或 normal chat。

**1. assistant-only SFT loss**

$$
L_{\mathrm{sft}}(\theta)=
-\frac{1}{N_{\mathrm{asst}}}
\sum_i\sum_t m_{i,t}\log p_\theta(y_{i,t}\mid x_{i,1:t-1})
$$

这里 `m_{i,t}=1` 表示该 token 参与 loss。SFT 中通常只让 assistant answer 和必要 EOS 参与 loss，system / user / padding 是条件或占位，不应成为模型要模仿输出的目标。

**2. assistant label 覆盖率**

$$
C_{\mathrm{asst}}=
\frac{\sum_i\sum_t z^{\mathrm{asst}}_{i,t}m_{i,t}}
{\max(1,\sum_i\sum_t z^{\mathrm{asst}}_{i,t})}
$$

`z_asst=1` 表示该位置属于 assistant 回答。`C_asst` 太低，说明回答 token 被误 mask，模型学不到答案；超过 1 不可能，接近 1 才是正常口径。

**3. prompt loss 泄漏率**

$$
R_{\mathrm{prompt}}=
\frac{\sum_i\sum_t z^{\mathrm{prompt}}_{i,t}m_{i,t}}
{\max(1,\sum_i\sum_t z^{\mathrm{prompt}}_{i,t})}
$$

如果 `R_prompt` 大于 0，模型就在学习复述 system / user / role token。这类事故会导致模型替用户说话、复述问题或生成角色标记。

**4. pad loss 泄漏率与 EOS 覆盖率**

$$
R_{\mathrm{pad}}=
\frac{\sum_i\sum_t z^{\mathrm{pad}}_{i,t}m_{i,t}}
{\max(1,\sum_i\sum_t z^{\mathrm{pad}}_{i,t})}
$$

$$
C_{\mathrm{eos}}=
\frac{\sum_i\sum_t z^{\mathrm{eos}}_{i,t}m_{i,t}}
{\max(1,\sum_i\sum_t z^{\mathrm{eos}}_{i,t})}
$$

padding 参与 loss 是硬错误；EOS 完全缺失则会让模型学不会停止，尤其在多轮 SFT 和工具调用场景里影响很大。

**5. 能力回归**

$$
\Delta_k=M^{\mathrm{after}}_k-M^{\mathrm{before}}_k
$$

`k` 表示能力切片，例如 instruction、math、code、safety、tool、long context。后训练不能只看聊天分数提升，必须监控每个能力切片的 `Delta_k`。

**6. 误拒率与漏拒率**

$$
R_{\mathrm{false\_refuse}}=
\frac{N_{\mathrm{safe,refused}}}{\max(1,N_{\mathrm{safe}})}
$$

$$
R_{\mathrm{unsafe\_leak}}=
\frac{N_{\mathrm{unsafe,answered}}}{\max(1,N_{\mathrm{unsafe}})}
$$

好的安全对齐不是拒答越多越好。误拒高会损害帮助性，漏拒高会损害安全性，两者都要进门禁。

**7. 偏好间隔**

$$
\Delta^{\mathrm{pref}}_j=h(c_j)-h(r_j)
$$

`c_j` 是 chosen response，`r_j` 是 rejected response，`h` 是人工或高可信评审质量分。`Delta_pref` 太小，偏好 pair 的训练信号就接近噪声；如果小于 0，说明 chosen 可能根本不该被选中。

**8. reward-human gap 与长度偏差**

$$
G_{\mathrm{rh}}=\frac{1}{N}\sum_i |r_i-h_i|
$$

$$
B_{\mathrm{len}}=\mathrm{corr}(\ell_i,r_i)
$$

`G_rh` 衡量 reward 与人工质量差距，`B_len` 衡量 reward 是否偏好长回答。reward 高但人工质量低，是 RLHF / rerank / best-of-N 都会放大的风险。

**9. DPO margin 与 loss**

$$
m_j=
\left[\log \pi_\theta(c_j\mid x_j)-\log \pi_\theta(r_j\mid x_j)\right]
-
\left[\log \pi_{\mathrm{ref}}(c_j\mid x_j)-\log \pi_{\mathrm{ref}}(r_j\mid x_j)\right]
$$

$$
L_{\mathrm{dpo}}=-\log \sigma(\beta m_j)
$$

DPO 不是简单提高 chosen 概率，而是提高 policy 相对 reference 对 chosen 的偏好优势。reference model 选错、beta 不合适、pair 质量差，都会让 DPO 学到错误风格或长度偏差。

**10. 后训练事故门禁**

$$
G_{\mathrm{align}}=\mathbf{1}\left[
C_{\mathrm{asst}}\ge\tau_{\mathrm{asst}}
\land R_{\mathrm{prompt}}=0
\land R_{\mathrm{pad}}=0
\land C_{\mathrm{eos}}\ge\tau_{\mathrm{eos}}
\land \min_k \Delta_k\ge-\tau_{\mathrm{reg}}
\land R_{\mathrm{false\_refuse}}\le\tau_{\mathrm{fr}}
\land R_{\mathrm{unsafe\_leak}}\le\tau_{\mathrm{ul}}
\land \overline{\Delta}^{\mathrm{pref}}\ge\tau_{\mathrm{pref}}
\land G_{\mathrm{rh}}\le\tau_{\mathrm{gap}}
\land |B_{\mathrm{len}}|\le\tau_{\mathrm{len}}
\right]
$$

这个门禁的意义不是用一个数替代人工评估，而是防止团队只看 SFT loss、reward curve 或 DPO loss。只要任一门禁失败，就应该先定位数据、mask、偏好、reward 或评估切片，而不是继续扩大训练。

## 6.19.2 最小可运行 SFT / 对齐训练事故审计 demo

下面的 demo 不依赖外部库。它故意构造多个事故：prompt 和 padding 参与 loss、assistant token 被误 mask、训练 / 推理 chat template 不一致、math / code / tool 能力回归、误拒和漏拒同时存在、偏好 pair 间隔过小、reward model 偏好长回答、DPO reference 不匹配以及工具 schema 漂移。

```python
from math import exp, log, sqrt

sft_samples = [
    {
        "id": "good_math",
        "tokens": {"system": 5, "user": 8, "assistant": 12, "eos": 1, "pad": 4},
        "labels": {"system": 0, "user": 0, "assistant": 12, "eos": 1, "pad": 0},
    },
    {
        "id": "prompt_leak",
        "tokens": {"system": 4, "user": 10, "assistant": 9, "eos": 1, "pad": 2},
        "labels": {"system": 0, "user": 10, "assistant": 9, "eos": 1, "pad": 2},
    },
    {
        "id": "masked_answer",
        "tokens": {"system": 3, "user": 7, "assistant": 11, "eos": 1, "pad": 0},
        "labels": {"system": 0, "user": 0, "assistant": 6, "eos": 0, "pad": 0},
    },
]
train_template = "<system>{system}</system><user>{user}</user><assistant>{assistant}</assistant>"
serving_template = "<|system|>{system}<|user|>{user}<|assistant|>{assistant}"

prompt_tokens = sum(sample["tokens"][role] for sample in sft_samples for role in ("system", "user"))
prompt_labels = sum(sample["labels"][role] for sample in sft_samples for role in ("system", "user"))
assistant_tokens = sum(sample["tokens"]["assistant"] for sample in sft_samples)
assistant_labels = sum(sample["labels"]["assistant"] for sample in sft_samples)
pad_tokens = sum(sample["tokens"]["pad"] for sample in sft_samples)
pad_labels = sum(sample["labels"]["pad"] for sample in sft_samples)
eos_tokens = sum(sample["tokens"]["eos"] for sample in sft_samples)
eos_labels = sum(sample["labels"]["eos"] for sample in sft_samples)

mask_audit = {
    "assistant_coverage": round(assistant_labels / assistant_tokens, 3),
    "prompt_loss_leak_rate": round(prompt_labels / prompt_tokens, 3),
    "pad_loss_leak_rate": round(pad_labels / max(1, pad_tokens), 3),
    "eos_coverage": round(eos_labels / eos_tokens, 3),
    "template_match": train_template == serving_template,
}

before = {"instruction": 0.62, "math": 0.70, "code": 0.66, "safety": 0.78, "tool": 0.60}
after = {"instruction": 0.76, "math": 0.55, "code": 0.49, "safety": 0.83, "tool": 0.52}
capability_delta = {name: round(after[name] - before[name], 3) for name in before}
regressions = {name: delta for name, delta in capability_delta.items() if delta <= -0.08}

refusal_cases = [
    {"id": "defensive_security", "should_refuse": False, "refused": True},
    {"id": "medical_general", "should_refuse": False, "refused": True},
    {"id": "homework_math", "should_refuse": False, "refused": False},
    {"id": "high_risk_steps", "should_refuse": True, "refused": False},
    {"id": "privacy_exposure", "should_refuse": True, "refused": True},
    {"id": "dangerous_operation", "should_refuse": True, "refused": True},
]
safe_cases = [case for case in refusal_cases if not case["should_refuse"]]
unsafe_cases = [case for case in refusal_cases if case["should_refuse"]]
false_refusals = [case["id"] for case in safe_cases if case["refused"]]
unsafe_leaks = [case["id"] for case in unsafe_cases if not case["refused"]]
refusal_audit = {
    "over_refusal_rate": round(len(false_refusals) / len(safe_cases), 3),
    "unsafe_leak_rate": round(len(unsafe_leaks) / len(unsafe_cases), 3),
    "false_refusals": false_refusals,
    "unsafe_leaks": unsafe_leaks,
}

preference_pairs = [
    {"id": "fact_fix", "chosen_q": 0.90, "rejected_q": 0.40, "chosen_len": 60, "rejected_len": 70, "agreement": 0.90},
    {"id": "small_margin", "chosen_q": 0.62, "rejected_q": 0.59, "chosen_len": 80, "rejected_len": 75, "agreement": 0.52},
    {"id": "length_bias", "chosen_q": 0.55, "rejected_q": 0.80, "chosen_len": 220, "rejected_len": 65, "agreement": 0.60},
    {"id": "safety_boundary", "chosen_q": 0.70, "rejected_q": 0.65, "chosen_len": 120, "rejected_len": 90, "agreement": 0.55},
]
preference_margins = {pair["id"]: round(pair["chosen_q"] - pair["rejected_q"], 3) for pair in preference_pairs}
low_margin_pairs = [pair_id for pair_id, margin in preference_margins.items() if margin < 0.10]
bad_chosen_pairs = [pair_id for pair_id, margin in preference_margins.items() if margin < 0]
preference_audit = {
    "avg_margin": round(sum(preference_margins.values()) / len(preference_margins), 3),
    "avg_agreement": round(sum(pair["agreement"] for pair in preference_pairs) / len(preference_pairs), 3),
    "low_margin_pairs": low_margin_pairs,
    "bad_chosen_pairs": bad_chosen_pairs,
}

reward_samples = [
    {"id": "concise_correct", "quality": 0.90, "length": 60, "reward": 0.42},
    {"id": "verbose_shallow", "quality": 0.45, "length": 240, "reward": 0.88},
    {"id": "safe_alt", "quality": 0.82, "length": 110, "reward": 0.74},
    {"id": "long_refusal", "quality": 0.50, "length": 210, "reward": 0.86},
]

def corr(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = sqrt(sum((y - mean_y) ** 2 for y in ys))
    return num / max(1e-12, den_x * den_y)

length_reward_corr = corr([sample["length"] for sample in reward_samples], [sample["reward"] for sample in reward_samples])
reward_human_gap = sum(abs(sample["reward"] - sample["quality"]) for sample in reward_samples) / len(reward_samples)
high_reward_low_quality = [
    sample["id"] for sample in reward_samples if sample["reward"] >= 0.80 and sample["quality"] < 0.60
]
reward_audit = {
    "length_reward_corr": round(length_reward_corr, 3),
    "reward_human_gap": round(reward_human_gap, 3),
    "high_reward_low_quality": high_reward_low_quality,
}

dpo_pairs = [
    {"id": "clear_win", "pi_c": -1.2, "pi_r": -1.9, "ref_c": -1.4, "ref_r": -1.6},
    {"id": "policy_prefers_bad", "pi_c": -2.5, "pi_r": -2.2, "ref_c": -2.1, "ref_r": -2.0},
    {"id": "ref_mismatch_case", "pi_c": -3.1, "pi_r": -3.0, "ref_c": -2.7, "ref_r": -2.9},
]
beta = 0.8
train_reference = "sft_v2"
expected_reference = "sft_v3"
dpo_margins = {}
dpo_losses = {}
for pair in dpo_pairs:
    margin = (pair["pi_c"] - pair["pi_r"]) - (pair["ref_c"] - pair["ref_r"])
    dpo_margins[pair["id"]] = round(margin, 3)
    dpo_losses[pair["id"]] = round(-log(1 / (1 + exp(-beta * margin))), 3)
negative_dpo_margins = [pair_id for pair_id, margin in dpo_margins.items() if margin <= 0]
dpo_audit = {
    "margins": dpo_margins,
    "losses": dpo_losses,
    "negative_margins": negative_dpo_margins,
    "reference_match": train_reference == expected_reference,
    "beta": beta,
}

train_tool_schema = {"search": ["query"], "calculator": ["expression"]}
serving_tool_schema = {"web_search": ["query"], "calculator": ["expr"]}
tool_schema_match = train_tool_schema == serving_tool_schema

gates = {
    "sft_mask_ok": mask_audit["assistant_coverage"] >= 0.95
    and mask_audit["prompt_loss_leak_rate"] == 0
    and mask_audit["pad_loss_leak_rate"] == 0
    and mask_audit["eos_coverage"] >= 0.95,
    "template_ok": mask_audit["template_match"],
    "capability_regression_ok": not regressions,
    "refusal_ok": refusal_audit["over_refusal_rate"] <= 0.20 and refusal_audit["unsafe_leak_rate"] == 0,
    "preference_data_ok": preference_audit["avg_margin"] >= 0.15
    and preference_audit["avg_agreement"] >= 0.70
    and not preference_audit["bad_chosen_pairs"],
    "reward_model_ok": abs(reward_audit["length_reward_corr"]) <= 0.50
    and reward_audit["reward_human_gap"] <= 0.20
    and not reward_audit["high_reward_low_quality"],
    "dpo_ok": not dpo_audit["negative_margins"] and dpo_audit["reference_match"],
    "tool_schema_ok": tool_schema_match,
}

print("mask_audit=", mask_audit)
print("capability_delta=", capability_delta)
print("regressions=", regressions)
print("refusal_audit=", refusal_audit)
print("preference_audit=", preference_audit)
print("reward_audit=", reward_audit)
print("dpo_audit=", dpo_audit)
print("tool_schema_match=", tool_schema_match)
print("gates=", gates)
print("gate_pass=", all(gates.values()))
```

一次输出示例：

```text
mask_audit= {'assistant_coverage': 0.844, 'prompt_loss_leak_rate': 0.27, 'pad_loss_leak_rate': 0.333, 'eos_coverage': 0.667, 'template_match': False}
capability_delta= {'instruction': 0.14, 'math': -0.15, 'code': -0.17, 'safety': 0.05, 'tool': -0.08}
regressions= {'math': -0.15, 'code': -0.17, 'tool': -0.08}
refusal_audit= {'over_refusal_rate': 0.667, 'unsafe_leak_rate': 0.333, 'false_refusals': ['defensive_security', 'medical_general'], 'unsafe_leaks': ['high_risk_steps']}
preference_audit= {'avg_margin': 0.083, 'avg_agreement': 0.643, 'low_margin_pairs': ['small_margin', 'length_bias', 'safety_boundary'], 'bad_chosen_pairs': ['length_bias']}
reward_audit= {'length_reward_corr': 0.91, 'reward_human_gap': 0.338, 'high_reward_low_quality': ['verbose_shallow', 'long_refusal']}
dpo_audit= {'margins': {'clear_win': 0.5, 'policy_prefers_bad': -0.2, 'ref_mismatch_case': -0.3}, 'losses': {'clear_win': 0.513, 'policy_prefers_bad': 0.776, 'ref_mismatch_case': 0.82}, 'negative_margins': ['policy_prefers_bad', 'ref_mismatch_case'], 'reference_match': False, 'beta': 0.8}
tool_schema_match= False
gates= {'sft_mask_ok': False, 'template_ok': False, 'capability_regression_ok': False, 'refusal_ok': False, 'preference_data_ok': False, 'reward_model_ok': False, 'dpo_ok': False, 'tool_schema_ok': False}
gate_pass= False
```

这段输出故意让所有门禁失败。它说明后训练事故不能只盯着一个 loss：SFT mask、模板、能力回归、安全拒答、偏好数据、reward model、DPO reference 和工具 schema 都可能单独造成线上行为异常。

## 6.20 事故复盘模板

```text
现象：SFT/RLHF/DPO 后出现什么退化
影响：影响哪些任务、用户或版本
对照：与 base model、SFT model 或旧版本对比
排查：数据、mask、偏好 pair、reward、长度、拒答率
根因：数据配比、格式、标注、reward 或训练超参
修复：清洗数据、调配比、改 mask、调 beta、回滚 checkpoint
验证：多维评估和线上灰度结果
预防：新增回归集、标注规范和监控指标
```

## 6.21 面试题：SFT 后模型基础能力下降怎么办

回答要点：

```text
我会先分能力评估 SFT 前后变化，确认是哪些能力下降，比如数学、代码、知识还是长上下文。然后检查 SFT 数据质量、数据覆盖、训练步数、学习率和是否存在格式或 mask 问题。常见修复是降低学习率、减少 epoch、混入通用高质量数据、选择更早 checkpoint，并建立多维回归评估避免只看聊天效果。
```

## 6.22 面试题：DPO 训练效果怪怎么排查

回答要点：

```text
我会先抽查 chosen/rejected pair，确认 chosen 是否真的更好，差异是否足够明显，标注标准是否一致。然后看 DPO 后输出长度、风格、拒答率和分任务能力变化，检查 beta、reference model 和数据覆盖。如果偏好数据本身噪声很大，DPO 会学到格式、长度或标注者偏好，而不是真实质量。
```

## 6.23 经验法则

后训练不是简单提升模型，而是在改变模型行为分布。任何提升都可能伴随能力、风格或安全性的副作用。

更具体地说：

1. SFT 后一定做多维回归。
2. 安全提升要同时看误拒率。
3. 偏好数据质量比数量重要。
4. Reward model 偏差会被 RLHF 放大。
5. DPO 简洁，但仍然依赖 pair 质量。
6. 输出长度和风格漂移要监控。
7. 对齐不是一次训练完成，而是持续评估和修正。

下一章会进入评估指标陷阱。后训练里很多问题之所以难发现，是因为团队只看单一指标，而没有看分任务、分场景、分风险和真实用户反馈。
