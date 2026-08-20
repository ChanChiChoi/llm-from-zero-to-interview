# 第六章：SFT 与对齐训练的行为事故

后训练把基础模型变成可用助手，却不是一个单向的“能力增强”按钮。SFT、偏好优化、Reward Model、RLHF、DPO 和安全训练都在改变模型的行为分布：它们可能改善指令跟随、格式稳定性和拒答边界，也可能损伤数学、代码、事实、长上下文、工具调用和输出多样性。

后训练事故的危险在于，局部体验可能变好，整体能力却已经退化。模型更会聊天，但数学和代码变差；安全拒答率上升，但正常教育和防护问题也被拒绝；reward 分数上升，但答案更长、更空、更模板化；DPO loss 下降，但 reference model、偏好 pair 或工具 schema 根本没有对齐。

本章把后训练看成一次行为分布重塑。先定义训练想改变什么、不能损伤什么，再分别审查 SFT 监督边界、能力回归、过度拒答、偏好数据、Reward Model、RLHF reward hacking、DPO margin、长度和工具协议。每个指标都明确样本集合、适用条件和零分母语义，最后用一个标准库审计示例把多个故障放在同一张报告中。

## 6.1 后训练改变的是行为分布

基础模型学习的是广泛的语言和知识分布，后训练则在更窄的任务、角色、偏好和安全约束上重新加权。可以把基础模型分布写成 `P_base(y|x)`，后训练后的分布写成：

```math

P_{post}(y\mid x)=P_{base}(y\mid x;\theta+\Delta\theta,\mathcal{D}_{post},\mathcal{R})
```

其中 `Delta theta` 是参数变化，`D_post` 是后训练数据，`R` 是损失、偏好、奖励、策略和安全约束的组合。这个表达不是一个具体算法，而是提醒我们：数据、mask、reference、reward 和评估定义都会改变最终行为。

后训练开始前，应该写一份“允许改变”和“必须保留”的能力合同：

```text
允许改变：指令跟随、结构化输出、目标领域术语、工具调用格式
必须保留：基础知识、数学、代码、事实准确性、长上下文和多语言能力
必须改善：危险请求处理、隐私边界、误拒和漏拒
必须稳定：EOS、role、tool schema、引用和状态协议
必须观察：输出长度、风格、成本、延迟、人工采纳和真实任务完成
```

没有这份合同，团队很容易用一个聊天分数证明所有事情都变好。后训练评估至少要比较 base、SFT、偏好优化和最终版本，并保留每个阶段的 checkpoint、数据版本、模板和评估器。

## 6.2 SFT 的监督边界

SFT 的关键不是“有多少条指令”，而是模型被要求预测哪些 token。设第 `i` 个样本的目标 token 为 `y_{i,t}`，监督 mask 为 `m_{i,t}`，当有效监督 token 总数 `N_loss>0` 时：

```math

L_{SFT}=-\frac{1}{N_{loss}}
\sum_{i,t}m_{i,t}\log p_\theta(y_{i,t}\mid x_{i,<t})
```

system 和 user 通常是条件，不是要模型复述的目标；assistant 内容和必要的 EOS 通常参与 loss；PAD、未监督控制 token 和被截断的空区域不应参与。具体策略可因任务不同而变化，但必须明确写入 collator 和数据契约。

最常见的 SFT 事故是 prompt loss 泄漏：labels 在 system/user 位置没有被忽略，模型因此学会复述问题、生成 role token 或把下一轮 user 说出来。另一个相反事故是 assistant 内容被全部 mask，训练 loss 仍能计算或被日志吞掉，但模型没有获得有效答案监督。

检查一条样本时，必须同时打印原始 messages、渲染文本、token、role spans、labels 和有效监督 token 数。仅仅看到训练 loss 在下降，不能证明监督位置正确。

## 6.3 SFT 后能力回归

SFT 后聊天体验变好而数学、代码或事实能力下降，通常与数据范围、学习率、训练步数、模板和分布偏移有关。先不要把它归因于灾难性遗忘，因为评估格式、提示和数据也可能已经变了。

对能力切片 `k`，设后训练前后的指标分别为 `M_before,k` 和 `M_after,k`：

```math

\Delta_k=M_{after,k}-M_{before,k}
```

`Delta_k` 只有在数据集、提示、评分器、解码、工具和资源条件足够一致时才可比较。需要单独报告 instruction、math、code、knowledge、long-context、multilingual、safety、tool 和真实业务任务，而不是只报一个加权总分。

常见回归模式：

- 领域 SFT 数据过窄，通用问题被错误处理；
- 高学习率或过多 epoch 让模型过度适应少量模板；
- 安全数据比例过高，正常问题被拒答；
- 工具调用格式训练过多，自然对话变得生硬；
- 合成数据风格压过真实用户表达；
- tokenizer、template 或 EOS 变化让评估输入不再可比。

修复可以从更早 checkpoint、较低学习率、混入通用高质量数据、调整数据配比和改善多维回归开始。LoRA 或其他参数高效方法不自动消除回归风险，adapter 仍然会改变输出分布和格式。

## 6.4 过度拒答和安全边界

安全训练不能简单等价为“拒答越多越安全”。正常的教育、防护、合规和风险分析问题可能包含敏感词，但任务本身不要求执行危险行为。如果所有边界输入都被拒答，用户任务完成率和系统可信度都会下降。

把样本分成应该回答的安全集合 `S` 和应该拒绝或受限处理的风险集合 `U`。当 `|S|>0` 时，误拒率为：

```math

R_{false\_refuse}=\frac{N_{S,refused}}{|S|}
```

当 `|U|>0` 时，漏拒率为：

```math

R_{unsafe\_leak}=\frac{N_{U,answered\_unsafely}}{|U|}
```

若评估集没有安全样本，指标是 `not_applicable`；如果应该有样本但标注或执行结果缺失，指标是 `unknown`。不能把没有安全样本的模型记为零漏拒。

安全边界应按任务意图、可执行性、潜在伤害和可提供的安全替代帮助判断。一个高质量拒答可能说明不能提供危险操作步骤，同时给出防护、求助或合规路径；单纯输出长免责声明不等于安全，也不等于有用。

过度拒答排查要看：安全数据与正常数据比例、标注规则、关键词触发、拒答模板、任务切片、语言差异和高风险判定是否被模型自行替代。不要把所有安全行为压缩成一个 refusal rate。

## 6.5 多轮和工具调用的后训练格式

后训练数据的角色和工具协议必须与线上执行器一致。多轮样本中，system、user、assistant 和 tool 的顺序、EOS 和 labels 决定模型学到的状态转换；工具样本中，tool name、参数 schema、call ID 和 observation 决定模型是否会把调用当成真实协议。

常见事故包括：

1. tool result 被当成 assistant 说话，模型学会伪造工具结果。
2. 工具名和线上 registry 不一致，生成能解析却找不到执行器。
3. 训练 schema 的字段是 `expression`，线上 schema 改成 `expr`，参数校验全部失败。
4. 只训练成功调用，没有训练超时、拒绝、重试和未知外部状态。
5. 生成自然语言和 tool call 混在同一段，解析器无法判断提交边界。

工具对齐要同时评估选择、参数、schema、执行、观察使用和恢复；调用次数多不代表工具使用正确。外部动作还必须受权限、幂等和人工确认控制，不能因为模型在训练中生成了某个 JSON 就自动执行。

## 6.6 偏好数据是一份质量合同

DPO、RLHF、RLAIF 和其他偏好方法依赖 chosen/rejected 或排序数据。偏好不是一个天然客观的标签，它可能同时混入事实正确性、安全、简洁、风格、格式、礼貌、长度和标注者个人偏好。

一条偏好记录应至少说明：

```text
输入和任务类型
chosen / rejected 文本
偏好的主要维度
风险等级和安全策略
标注者或评审器版本
置信度与是否需要复核
长度、格式和工具状态
```

如果 chosen 只是更长、更礼貌或格式更完整，但事实并不更正确，模型可能学到长度或模板捷径。训练前要对 pair 做独立质量检查，不要把“有人选了 chosen”当作“chosen 在所有目标上都更好”。

## 6.7 Chosen/Rejected 差异太小或方向相反

设人工或高可信评审对 chosen 和 rejected 的质量分为 `h_c` 与 `h_r`，偏好间隔为：

```math

\Delta_{pref}=h_c-h_r
```

当 pair 集合非空时，可以报告平均间隔；若 pair 为空，没有偏好证据。`Delta_pref` 很小表示训练信号噪声大，`Delta_pref<0` 则说明 chosen 可能被标错。

差异检查不能只比较文本长度。应分别比较：

- 事实和引用是否正确；
- 是否完整回答任务；
- 安全边界是否合规；
- 是否存在工具或格式错误；
- 是否引入无关长篇内容；
- 是否能被真实用户采纳。

标注一致性要按任务、风险级别和标注者分片。整体一致率高，可能只是大量容易样本掩盖了高风险边界的一致性低。对低置信 pair，隔离、重标或降低权重通常比盲目增加数量更可靠。

## 6.8 Reward Model 的捷径偏差

Reward Model 的目标是近似某种偏好或质量判断，但它可能学到容易检测的表面特征：答案长度、礼貌词、项目符号、免责声明、特定模板或参考答案词汇。模型一旦被 RL 或 rerank 优化，就会放大这些捷径。

设 reward 为 `r_i`，人工或高可信质量为 `h_i`，在 `N>0` 个同口径样本上，平均差距可写成：

```math

G_{rh}=\frac{1}{N}\sum_{i=1}^{N}|r_i-h_i|
```

如果样本没有人工质量标签，`G_rh` 是 `unknown`，不是 0。长度偏差可以用 reward 与长度的相关性作为线索，但相关性要求长度和 reward 都有方差；如果其中一者是常数，相关性未定义。

Reward 检查需要构造反事实对：保持事实和任务答案相同，只改变长度、礼貌模板、免责声明和格式；或者保持长度相近，只改变事实正确性。若 reward 只随长度上升，而人工质量不变甚至下降，说明 reward 存在捷径。

## 6.9 RLHF reward hacking

Reward hacking 是策略学会利用 reward 的漏洞，而没有提升真实任务质量。常见表现是回答越来越长、免责声明泛滥、格式越来越像评分器偏好的模板、事实错误被漂亮结构掩盖，reward 曲线上升而人工评价下降。

排查要保留 SFT 或 base baseline，并同时监控：

- reward 和人工质量；
- 输出长度与有效信息密度；
- 事实错误和引用支持；
- 安全误拒和漏拒；
- KL 或与 reference 的行为偏移；
- 真实任务完成率、成本和延迟。

KL 约束可以帮助限制策略偏离 reference，但 KL 不是质量指标，也不能修复错误 reward。多维 reward、独立人工抽样、反事实 reward 测试和定期回放高 reward 样本，才有机会发现奖励漏洞。

## 6.10 DPO 的 reference、beta 和 margin

DPO 直接用偏好 pair 和 reference model 优化相对偏好。对 chosen `c`、rejected `r` 和输入 `x`，一个常见 margin 为：

```math

m=\left[\log\pi_\theta(c\mid x)-\log\pi_\theta(r\mid x)\right]
-\left[\log\pi_{ref}(c\mid x)-\log\pi_{ref}(r\mid x)\right]
```

对应的损失可以写成：

```math

L_{DPO}=-\log\sigma(\beta m)
```

其中 `beta>0`，`pi_ref` 必须是记录中声明的 reference。reference 版本不匹配、chat template 不一致、tokenizer 不一致或 chosen/rejected 的长度处理不同，都会让 margin 失去预期含义。

DPO 排查要看：

1. margin 分布和负 margin 比例；
2. beta 的反事实实验；
3. reference 和 policy 的版本、模板和 tokenizer；
4. pair 的任务、风险和长度分布；
5. 输出风格、长度和能力切片的回归；
6. 训练前后真实任务和安全边界。

loss 下降只能说明优化目标被拟合，不能说明偏好标签正确或产品结果改善。

## 6.11 输出长度和风格漂移

后训练后输出变长，可能源于 SFT 答案本身过长、标注者偏好解释、reward model 的长度捷径、安全免责声明或 DPO pair 的长度差异。长度变化会影响成本、延迟、用户阅读和工具调用次数。

不要只看平均长度。应按任务、语言、风险、成功/失败和用户采纳分片。一个数学题答案变长但正确率不变，可能是效率下降；一个安全回答变长但仍漏拒，说明免责声明没有解决风险；一个 RAG 答案变长却引用覆盖下降，说明上下文和生成协议可能失衡。

风格指标也需要任务化：简洁性、结构清晰、事实密度、引用可用、拒答解释和格式遵循。所有回答都变成同一种模板，不能简单称为“风格稳定”。

## 6.12 多维评估和阶段对照

后训练评估至少包括：

| 维度 | 需要观察什么 |
| --- | --- |
| 指令 | 是否理解任务、遵守约束和处理多轮 |
| 知识 | 事实、时效、引用和拒答边界 |
| 数学/代码 | 正确率、过程、测试和鲁棒性 |
| 安全 | 误拒、漏拒、替代帮助和隐私 |
| 工具 | 选择、参数、schema、观察使用和恢复 |
| 语言 | 各语言质量、混用、术语和输入鲁棒性 |
| 长上下文 | 证据定位、跨段关系和成本 |
| 产品 | 任务完成、人工采纳、延迟、成本和反馈 |

评估必须保留 base、SFT、偏好优化和最终版本的对照。只比较最终模型与一个旧 baseline，无法知道退化发生在哪个阶段。对于高风险任务，离线分数不能替代人工审查和受控灰度。

## 6.13 后训练事故的诊断顺序

遇到后训练后行为异常，可以按以下顺序：

1. 固定模型、tokenizer、template、解码和评估器版本。
2. 对比 base/SFT/偏好优化/最终 checkpoint 的任务切片。
3. 抽查原始 SFT 样本和 labels，确认 assistant 监督边界。
4. 抽查 chosen/rejected，确认偏好方向、任务和风险维度。
5. 查看拒答、输出长度、风格、工具和引用的变化。
6. 对 reward 与人工质量做相关和反事实测试。
7. 检查 DPO reference、beta、margin 和 tokenizer/template。
8. 复核工具 schema、权限、状态和错误恢复数据。
9. 做小规模单变量 ablation，再决定重训或回滚。

这条顺序先排除数据和协议问题，再解释训练算法。因为一个错误的 mask 或 schema 会让所有高级方法都在错误目标上优化。

## 6.14 指标定义域

### 6.14.1 Assistant 覆盖和 prompt 泄漏

```math

C_{asst}=\frac{N_{assistant\ labels}}{N_{assistant\ positions}},\qquad
R_{prompt}=\frac{N_{prompt\ labels}}{N_{prompt\ positions}}
```

两个分母都必须为正；没有对应位置时应标记 `not_applicable` 或 `unknown`。PAD 泄漏同理。

### 6.14.2 能力回归

```math

\Delta_k=M_{after,k}-M_{before,k}
```

要求前后评估使用同一口径。若任一阶段缺少有效分母，`Delta_k` 不能被计算。

### 6.14.3 拒答边界

```math

R_{false\_refuse}=N_{safe,refused}/N_{safe}
```

```math

R_{unsafe\_leak}=N_{unsafe,answered\_unsafely}/N_{unsafe}
```

安全集合为空时不适用；样本存在但判定缺失时未知。

### 6.14.4 Reward 与人工质量

```math

G_{rh}=\frac{1}{N}\sum_i|r_i-h_i|
```

要求 `N>0`，并且 `r` 和 `h` 在同一范围或已校准。相关性需要非零方差，常数序列没有定义相关系数。

### 6.14.5 DPO margin

```math

m_j=(\log\pi_\theta(c_j|x_j)-\log\pi_\theta(r_j|x_j))
-(\log\pi_{ref}(c_j|x_j)-\log\pi_{ref}(r_j|x_j))
```

reference、tokenizer、template 和长度归一化必须绑定版本，否则 margin 不能直接解释。

## 6.15 一个可运行的后训练事故审计示例

下面的示例使用 Python 标准库，把 SFT mask、能力回归、拒答边界、偏好 pair、reward shortcut、DPO reference 和工具 schema 放在一份报告中。所有统计都通过显式分母函数计算，空集合返回 `None`。

```python
from dataclasses import dataclass
from math import exp, isfinite, log, log1p, sqrt
from typing import Dict, List, Optional, Tuple


def strict_int(value: object) -> bool:
    return type(value) is int


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if not strict_int(numerator) or not strict_int(denominator):
        raise ValueError("ratio arguments must be integers")
    if numerator < 0 or denominator < 0:
        raise ValueError("ratio arguments must be non-negative")
    if numerator > denominator:
        raise ValueError("numerator cannot exceed denominator")
    if denominator == 0:
        return None
    return round(numerator / denominator, 3)


def safe_average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if any(not isfinite(float(value)) for value in values):
        raise ValueError("average values must be finite")
    return sum(values) / len(values)


@dataclass(frozen=True)
class SFTSample:
    sample_id: str
    system_tokens: int
    user_tokens: int
    assistant_tokens: int
    assistant_labels: int
    prompt_labels: int
    pad_tokens: int
    pad_labels: int
    eos_tokens: int
    eos_labels: int


def validate_sft(row: SFTSample) -> None:
    if not row.sample_id.strip():
        raise ValueError("sample_id must not be empty")
    for name in (
        "system_tokens",
        "user_tokens",
        "assistant_tokens",
        "assistant_labels",
        "prompt_labels",
        "pad_tokens",
        "pad_labels",
        "eos_tokens",
        "eos_labels",
    ):
        value = getattr(row, name)
        if not strict_int(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if row.assistant_labels > row.assistant_tokens:
        raise ValueError("assistant labels cannot exceed assistant tokens")
    if row.pad_labels > row.pad_tokens:
        raise ValueError("pad labels cannot exceed pad tokens")
    if row.eos_labels > row.eos_tokens:
        raise ValueError("eos labels cannot exceed eos tokens")


def sft_audit(rows: List[SFTSample]) -> Dict[str, Optional[float]]:
    if not rows:
        return {
            "assistant_coverage": None,
            "prompt_loss_rate": None,
            "pad_loss_rate": None,
            "eos_coverage": None,
        }
    for row in rows:
        validate_sft(row)
    assistant_tokens = sum(row.assistant_tokens for row in rows)
    assistant_labels = sum(row.assistant_labels for row in rows)
    prompt_tokens = sum(row.system_tokens + row.user_tokens for row in rows)
    prompt_labels = sum(row.prompt_labels for row in rows)
    pad_tokens = sum(row.pad_tokens for row in rows)
    pad_labels = sum(row.pad_labels for row in rows)
    eos_tokens = sum(row.eos_tokens for row in rows)
    eos_labels = sum(row.eos_labels for row in rows)
    return {
        "assistant_coverage": safe_ratio(assistant_labels, assistant_tokens),
        "prompt_loss_rate": safe_ratio(prompt_labels, prompt_tokens),
        "pad_loss_rate": safe_ratio(pad_labels, pad_tokens),
        "eos_coverage": safe_ratio(eos_labels, eos_tokens),
    }


def correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys):
        raise ValueError("correlation inputs must have equal length")
    if not xs:
        return None
    if any(not isfinite(float(value)) for value in xs + ys):
        raise ValueError("correlation inputs must be finite")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    norm_x = sqrt(sum(value * value for value in centered_x))
    norm_y = sqrt(sum(value * value for value in centered_y))
    if norm_x == 0 or norm_y == 0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / (norm_x * norm_y)


def stable_softplus(value: float) -> float:
    if not isfinite(value):
        raise ValueError("softplus input must be finite")
    if value > 0:
        return value + log1p(exp(-value))
    return log1p(exp(value))


def dpo_margin(pair: Dict[str, float]) -> float:
    value = (pair["pi_chosen"] - pair["pi_rejected"]) - (
        pair["ref_chosen"] - pair["ref_rejected"]
    )
    if not isfinite(value):
        raise ValueError("DPO margin must be finite")
    return value


sft_rows = [
    SFTSample("good_math", 5, 8, 12, 12, 0, 4, 0, 1, 1),
    SFTSample("prompt_leak", 4, 10, 9, 9, 10, 2, 2, 1, 1),
    SFTSample("masked_answer", 3, 7, 11, 6, 0, 0, 0, 1, 0),
]
mask_audit = sft_audit(sft_rows)

before = {"instruction": 0.62, "math": 0.70, "code": 0.66, "safety": 0.78, "tool": 0.60}
after = {"instruction": 0.76, "math": 0.55, "code": 0.49, "safety": 0.83, "tool": 0.52}
if set(before) != set(after):
    raise ValueError("capability slices must match")
capability_delta = {name: round(after[name] - before[name], 3) for name in before}
regressions = {name: delta for name, delta in capability_delta.items() if delta <= -0.08}

refusal_cases = [
    {"id": "defensive_security", "unsafe": False, "refused": True},
    {"id": "medical_general", "unsafe": False, "refused": True},
    {"id": "homework_math", "unsafe": False, "refused": False},
    {"id": "high_risk_steps", "unsafe": True, "refused": False},
    {"id": "privacy_exposure", "unsafe": True, "refused": True},
    {"id": "dangerous_operation", "unsafe": True, "refused": True},
]
safe_cases = [row for row in refusal_cases if not row["unsafe"]]
unsafe_cases = [row for row in refusal_cases if row["unsafe"]]
false_refusals = [row["id"] for row in safe_cases if row["refused"]]
unsafe_leaks = [row["id"] for row in unsafe_cases if not row["refused"]]
refusal_audit = {
    "over_refusal_rate": safe_ratio(len(false_refusals), len(safe_cases)),
    "unsafe_leak_rate": safe_ratio(len(unsafe_leaks), len(unsafe_cases)),
    "false_refusals": false_refusals,
    "unsafe_leaks": unsafe_leaks,
}

preference_pairs = [
    {"id": "fact_fix", "chosen_quality": 0.90, "rejected_quality": 0.40, "agreement": 0.90},
    {"id": "small_margin", "chosen_quality": 0.62, "rejected_quality": 0.59, "agreement": 0.52},
    {"id": "length_bias", "chosen_quality": 0.55, "rejected_quality": 0.80, "agreement": 0.60},
    {"id": "safety_boundary", "chosen_quality": 0.70, "rejected_quality": 0.65, "agreement": 0.55},
]
preference_margins = {
    row["id"]: round(row["chosen_quality"] - row["rejected_quality"], 3)
    for row in preference_pairs
}
preference_audit = {
    "average_margin": round(
        safe_average(list(preference_margins.values())), 3
    ),
    "average_agreement": round(
        safe_average([row["agreement"] for row in preference_pairs]), 3
    ),
    "low_margin_pairs": [
        name for name, value in preference_margins.items() if value < 0.10
    ],
    "bad_chosen_pairs": [
        name for name, value in preference_margins.items() if value < 0
    ],
}

reward_samples = [
    {"id": "concise_correct", "quality": 0.90, "length": 60, "reward": 0.42},
    {"id": "verbose_shallow", "quality": 0.45, "length": 240, "reward": 0.88},
    {"id": "safe_alt", "quality": 0.82, "length": 110, "reward": 0.74},
    {"id": "long_refusal", "quality": 0.50, "length": 210, "reward": 0.86},
]
reward_audit = {
    "length_reward_corr": correlation(
        [row["length"] for row in reward_samples],
        [row["reward"] for row in reward_samples],
    ),
    "reward_human_gap": round(
        safe_average([abs(row["reward"] - row["quality"]) for row in reward_samples]),
        3,
    ),
    "high_reward_low_quality": [
        row["id"] for row in reward_samples if row["reward"] >= 0.80 and row["quality"] < 0.60
    ],
}

dpo_pairs = [
    {"id": "clear_win", "pi_chosen": -1.2, "pi_rejected": -1.9, "ref_chosen": -1.4, "ref_rejected": -1.6},
    {"id": "policy_prefers_bad", "pi_chosen": -2.5, "pi_rejected": -2.2, "ref_chosen": -2.1, "ref_rejected": -2.0},
    {"id": "ref_mismatch_case", "pi_chosen": -3.1, "pi_rejected": -3.0, "ref_chosen": -2.7, "ref_rejected": -2.9},
]
beta = 0.8
if beta <= 0 or not isfinite(beta):
    raise ValueError("beta must be positive and finite")
dpo_margins = {row["id"]: round(dpo_margin(row), 3) for row in dpo_pairs}
dpo_losses = {
    name: round(stable_softplus(-beta * margin), 3)
    for name, margin in dpo_margins.items()
}
dpo_audit = {
    "margins": dpo_margins,
    "losses": dpo_losses,
    "negative_margins": [name for name, value in dpo_margins.items() if value <= 0],
    "reference_match": "sft_v2" == "sft_v3",
    "beta": beta,
}

train_tool_schema = {"search": ["query"], "calculator": ["expression"]}
serving_tool_schema = {"web_search": ["query"], "calculator": ["expr"]}
tool_schema_match = train_tool_schema == serving_tool_schema

checks = {
    "sft_mask": (
        mask_audit["assistant_coverage"] is not None
        and mask_audit["assistant_coverage"] >= 0.95
        and mask_audit["prompt_loss_rate"] == 0
        and mask_audit["pad_loss_rate"] == 0
        and mask_audit["eos_coverage"] is not None
        and mask_audit["eos_coverage"] >= 0.95
    ),
    "capability_regression": not regressions,
    "refusal_boundary": (
        refusal_audit["over_refusal_rate"] is not None
        and refusal_audit["over_refusal_rate"] <= 0.20
        and refusal_audit["unsafe_leak_rate"] == 0
    ),
    "preference_data": (
        preference_audit["average_margin"] >= 0.15
        and preference_audit["average_agreement"] >= 0.70
        and not preference_audit["bad_chosen_pairs"]
    ),
    "reward_model": (
        reward_audit["length_reward_corr"] is not None
        and abs(reward_audit["length_reward_corr"]) <= 0.50
        and reward_audit["reward_human_gap"] <= 0.20
        and not reward_audit["high_reward_low_quality"]
    ),
    "dpo": not dpo_audit["negative_margins"] and dpo_audit["reference_match"],
    "tool_schema": tool_schema_match,
}
decision = "ready_for_review" if all(checks.values()) else "hold_for_repair"
print("mask_audit=", mask_audit)
print("capability_delta=", capability_delta)
print("refusal_audit=", refusal_audit)
print("preference_audit=", preference_audit)
print("reward_audit=", reward_audit)
print("dpo_audit=", dpo_audit)
print("tool_schema_match=", tool_schema_match)
print("checks=", checks)
print("decision=", decision)

assert decision == "hold_for_repair"
assert mask_audit["prompt_loss_rate"] > 0
assert regressions["math"] == -0.15
assert refusal_audit["unsafe_leak_rate"] > 0
assert dpo_audit["reference_match"] is False
assert safe_ratio(0, 0) is None

assert sft_audit([])["assistant_coverage"] is None
try:
    safe_ratio(1, 0)
except ValueError:
    pass
else:
    raise AssertionError("positive numerator with zero denominator is invalid")

bad_sft = SFTSample("bad_bool", True, 1, 1, 1, 0, 0, 0, 1, 1)
try:
    validate_sft(bad_sft)
except ValueError as exc:
    assert "system_tokens" in str(exc)
else:
    raise AssertionError("bool must not masquerade as a token count")

assert correlation([1.0, 1.0], [0.2, 0.2]) is None
```

示例中的 `hold_for_repair` 不是某个供应商或业务的通用上线标准，而是由本例构造的多个失败项共同产生：prompt/PAD 泄漏、assistant 覆盖不足、math/code/tool 回归、误拒和漏拒、低质量偏好 pair、reward 长度捷径、DPO reference 不匹配和工具 schema 漂移。实际项目应根据任务风险配置阈值，并保留每个失败项的样本和版本。

## 6.16 后训练事故的恢复路径

修复后训练事故时，不要直接从最终 checkpoint 继续优化。建议分层处理：

1. 保存 base、SFT、偏好优化和最终版本，冻结受影响评估结果。
2. 重新抽样 SFT labels、偏好 pair、拒答边界和工具样本。
3. 先修复数据、template、mask 和 schema，再讨论训练超参。
4. 对 reward 做反事实和人工校准，隔离高 reward 低质量样本。
5. 对 DPO/RLHF 做 reference、beta、KL、长度和多维 reward 对照。
6. 在固定能力回归集上选择 checkpoint，而不是只选最低训练 loss。
7. 通过真实任务、人工抽检和受控灰度观察成本、延迟、采用和安全。

最佳 checkpoint 可能在训练中间。后训练时间越长，越可能放大偏好噪声、reward shortcut 或领域过拟合；必须让 checkpoint 选择依赖多维目标，而不是单一 loss 或 reward。

## 6.17 常见但危险的处理方式

### 只看聊天体验

聊天自然不代表数学、代码、事实、工具和安全边界正确。后训练前后必须有固定多维回归。

### 安全训练只追求更高拒答率

拒答率上升可能来自误拒增加。安全评估必须同时覆盖安全回答、危险拒绝、替代帮助、语言切片和任务完成。

### 偏好 pair 越多越好

低质量、低间隔和方向错误的 pair 会增加噪声。先做 pair 质量和标注一致性，再扩大数量。

### Reward 越高越好

reward 可能偏好长度、模板和免责声明。高 reward 样本必须做人审和反事实检查。

### DPO loss 下降就算成功

DPO loss 只反映相对 reference 的优化目标。reference、tokenizer、template、pair 和 beta 错了，loss 仍可能下降。

### 训练工具调用不做线上 schema 回放

工具名、字段、枚举、call ID 和 observation 任何一个不一致，模型可能生成可解析但不可执行的请求。训练和执行器必须共享版本化 schema。

## 6.18 资料与证据边界

本章参考以下资料：

- InstructGPT 论文用于说明 SFT、奖励模型和人类反馈的训练链路；论文结果不能直接推出当前应用的安全或业务效果。
- PPO、DPO 及其后续偏好优化论文用于说明 reference、偏好 pair、策略变化和优化目标；具体实现需要绑定框架版本和损失细节。
- Hugging Face Transformers chat template、TRL SFT/DPO/GRPO 文档用于公开 API、数据字段和训练接口边界；默认 collator 和版本行为应通过样本回放验证。
- Reward Model、RLHF reward hacking 和 process preference 的研究用于解释奖励捷径、长度偏差和人工质量差距；任何阈值都需要在目标任务上校准。
- OpenAI Evals、NIST AI RMF 和 OWASP 资料用于多维评估、风险和持续治理的框架背景，不替代人工安全审查或目标系统实测。

闭源模型的内部后训练配方通常没有公开。公开 API 的 `reasoning`、`effort`、`refusal` 或工具字段不能用来反推训练数据、reward model、RL 算法或内部 verifier。正文只写论文、官方文档、模型卡和本地实验能够支持的内容。

## 6.19 本章小结

后训练不是把基础模型统一推向“更好”，而是在新的数据、偏好和约束下重塑行为。SFT 需要先保证 labels 和模板正确；能力评估需要发现数学、代码、知识、长上下文和工具回归；安全训练要同时控制误拒和漏拒；偏好数据需要方向、间隔和标注一致性；Reward Model 要接受人工质量和反事实挑战；RLHF 要防止 reward hacking；DPO 要保证 reference、tokenizer、template、beta 和 pair 的一致。

当后训练报告能同时回答“模型学了什么、忘了什么、为什么拒绝、为什么变长、reward 是否可信、工具是否仍能执行、哪个版本引入了变化”，训练结果才具有工程意义。任何单一 loss、reward 或聊天分数，都不能替代这条行为证据链。
