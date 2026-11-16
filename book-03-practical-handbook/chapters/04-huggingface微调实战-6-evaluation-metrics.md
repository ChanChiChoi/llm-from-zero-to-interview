# 第 4 章（第六篇）：定义比较协议与任务指标

训练结束只说明优化过程停了下来，不说明目标任务已经改善。要比较 base 与 SFT，先固定模型输入、模板、评估样本和生成配置，再定义任务指标、胜率与回归率。本篇建立比较协议；下一篇再把协议写成脚本和报告。

## 4.6 评估 SFT 前后的行为变化

### 4.6.1 训练完成不等于任务完成

SFT 训练最容易得到的结果是一个下降的训练 loss，但真正想知道的是模型行为是否改善：

```text
是否更愿意遵循指令。
是否能稳定输出指定格式。
答案是否更正确、更完整。
是否增加了幻觉、重复或不必要的拒答。
原有通用能力和安全边界是否回归。
```

因此评估至少包含两条线：

```text
token-level：masked validation loss、有效 token 数、困惑度。
task-level：生成结果、格式解析、事实检查、人工评分和坏例分析。
```

训练 loss 是必要的诊断信号，但不是任务成功的同义词。

### 4.6.2 固定比较对象

对每条评估样本 $i$，记录 prompt、类别、参考答案或检查点：

```math
\mathcal{D}_{\mathrm{eval}}
=
\{(p_i,c_i,r_i)\}_{i=1}^{N}
```

base 和 SFT 模型在同一 prompt、同一模板和同一生成参数下产生：

```math
y_i^{\mathrm{base}}
=
G(\theta_{\mathrm{base}},p_i,g)
```

```math
y_i^{\mathrm{sft}}
=
G(\theta_{\mathrm{sft}},p_i,g)
```

其中 $g$ 包括 max_new_tokens、do_sample、temperature、top-p、停止 token 和随机种子。比较时只应改变模型参数；如果同时改变 prompt 或采样策略，就不能把结果差异归因于 SFT。

base 与 SFT 应尽量使用相同 tokenizer 和模板。LoRA/QLoRA 的 SFT 模型通常是同一个 base 加 adapter；全参数 SFT 则应从保存目录加载对应 tokenizer。若更换了 base、词表或模板，报告中必须把它作为独立变量。

### 4.6.3 评估集应覆盖不同风险

一个只由训练集复制样本组成的评估集会产生过于乐观的结果。至少应分桶：

```text
训练分布内：确认模型是否学会目标格式和任务。
相似但未见：观察近邻泛化。
分布外：观察是否过度依赖模板。
格式任务：JSON、列表、表格、代码。
事实与来源任务：检查无依据断言。
安全和边界任务：检查拒答、隐私和风险内容。
旧能力回归：确认不相关能力没有明显损失。
```

样本元数据可以这样记录：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
eval_item = {
    "id": "format-json-001",
    "category": "format",
    "instruction": "用 JSON 输出三个机器学习术语。",
    "input": "",
    "expected_points": ["术语"],
    "format": "json",
    "source_requirement": False,
}
```
类别数量也应报告：

```math
N=\sum_{c\in\mathcal{C}}N_c
```

总分掩盖类别回归时，分桶结果能告诉我们是格式变好了、事实性变差了，还是某个小类别样本太少。

### 4.6.4 统一生成函数

评估时必须只比较新生成的 token：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch


@torch.no_grad()
def generate_new_text(model, tokenizer, prompt, max_new_tokens=128):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)
    prompt_length = inputs["input_ids"].shape[-1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    new_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
```
如果模型通过 device_map 分片，next(model.parameters()).device 只用于把输入放到起始设备，不能再把整个模型移动到单卡。对采样评估，应固定随机种子并对同一 prompt 重复多次，报告均值和方差，而不是只挑一条好看的输出。

### 4.6.5 任务级指标

对于明确格式，可以使用解析器：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import json


def format_pass(output, format_name):
    if format_name is None:
        return None
    if format_name == "json":
        try:
            value = json.loads(output)
        except json.JSONDecodeError:
            return False
        return isinstance(value, (dict, list))
    raise ValueError(f"unsupported format: {format_name}")
```
格式通过率：

令 $N_{\mathrm{fmt}}=|\mathcal{I}_{\mathrm{fmt}}|$。只有评估集中确实存在格式任务，即 $N_{\mathrm{fmt}}>0$，这个比例才有定义；没有格式样本时应记录为 `None`，而不是把空集合当作 100% 通过。

```math
A_{\mathrm{fmt}}
=
\frac{1}{N_{\mathrm{fmt}}}
\sum_{i\in\mathcal{I}_{\mathrm{fmt}}}
\mathbf{1}[\mathrm{valid}(y_i)]
```

关键词或检查点命中率只能作为粗粒度辅助：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
def keyword_score(output, expected_points):
    if not expected_points:
        return None
    if any(
        not isinstance(point, str) or not point.strip()
        for point in expected_points
    ):
        raise ValueError("expected_points must contain non-empty strings")
    output_lower = output.lower()
    hits = sum(
        point.lower() in output_lower
        for point in expected_points
    )
    return hits / len(expected_points)
```
其数学形式为：

对每个样本，令 $M_i$ 为非空检查点的数量。只有 $M_i>0$ 时，下面的关键词得分才有定义；它是便于定位坏例的字符串指标，不是语义正确率。

```math
K_i
=
\frac{1}{M_i}
\sum_{j=1}^{M_i}
\mathbf{1}[q_{i,j}\subset y_i]
```

关键词命中不等于语义正确：模型可能使用同义词而没有命中，也可能在错误语境中重复关键词。事实性、推理质量和安全性需要人工 rubric、程序验证或经过审查的评审模型共同判断。

### 4.6.6 胜率与回归率

如果每条输出都按同一份 rubric 得到评分 $s(y)$，就能逐 prompt 比较 base 与 SFT。令 $N>0$ 为两边都得到有效评分的样本数；平局保留在 $N$ 的分母中，但既不算胜出，也不算回归：

```math
W_{\mathrm{sft}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[s(y_i^{\mathrm{sft}})>s(y_i^{\mathrm{base}})]
```

更重要的是回归率：

```math
R_{\mathrm{reg}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[s(y_i^{\mathrm{sft}})<s(y_i^{\mathrm{base}})]
```

平均分上升仍可能伴随某个类别严重变差，因此报告应保存每个样本的 base 输出、SFT 输出、评分、类别和评语。

同时报告胜出、回归和平局的计数。若 100 条样本中 SFT 赢 70 条、输 20 条、平 10 条，含平局胜率是 70%；只在非平局样本中计算则为 $70/(70+20)\approx77.8\%$。两种口径都可以，但必须写明分母；只报较高的那一个会改变读者对结果的理解。

一个简单的人工 rubric 可以把每一维打成 0 到 3 分：

```text
指令遵循：是否完成了用户要求。
内容正确：事实和推理是否成立。
格式遵循：是否满足 JSON、字段、数量等约束。
表达质量：是否清晰、完整、无不必要重复。
边界行为：是否对无依据问题保持谨慎。
```

0 到 3 只是记录工具，不是普适的科学量表。评审前应固定评分说明，并尽量让评审者看到相同的 prompt 和输出。

### 4.6.7 validation loss 与行为评估的关系

使用 assistant-only labels 时，验证 loss 为：

令 $S_{\mathrm{val}}=\sum_{i,t}m_{i,t}$。只有 $S_{\mathrm{val}}>0$ 时，验证 loss 和由它得到的困惑度才有定义。

```math
\mathcal{L}_{\mathrm{val}}
=
-
\frac{
\sum_{i,t}m_{i,t}
\log p_\theta(x_{i,t}\mid x_{i,0:t-1})
}{
\sum_{i,t}m_{i,t}
}
```

它衡量模型对目标 token 的概率预测，不能直接衡量：

```text
回答是否真的完成任务。
JSON 是否能解析。
事实是否有来源。
是否出现危险或隐私泄露。
多轮交互是否保持角色边界。
```

困惑度可以由平均负对数似然得到：

```math
\mathrm{PPL}
=
\exp(\mathcal{L}_{\mathrm{val}})
```

只有当 tokenization、mask、评估文本和归一化方式一致时，PPL 才适合做相对比较。不同 tokenizer、不同回答长度和不同 mask 不能直接横向比较。

比较对象、评估切片和指标口径已经固定。下一篇把它们落成可保存逐样本输出的脚本，并写出有边界的实验结论。
