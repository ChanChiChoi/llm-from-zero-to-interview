# 第 4 章（下）：评估协议与实验报告

训练结束只说明优化过程停了下来，不说明目标任务已经改善。要比较 base 与 SFT，先固定模型输入、模板、评估样本和生成配置，再定义任务指标、胜率与回归率。本部分先建立比较协议，再把它写成脚本和报告。

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
\{(p_{i},c_{i},r_{i})\}_{i=1}^{N}
```

base 和 SFT 模型在同一 prompt、同一模板和同一生成参数下产生：

```math
y_{i}^{\mathrm{base}}
=
G(\theta_{\mathrm{base}},p_{i},g)
```

```math
y_{i}^{\mathrm{sft}}
=
G(\theta_{\mathrm{sft}},p_{i},g)
```

其中 $g$ 包括 max_new_tokens、do_sample、temperature、top-p、停止 token 和随机种子。比较时只应改变模型参数；如果同时改变 prompt 或采样策略，就不能认定结果差异是 SFT 带来的。

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
N=\sum_{c\in\mathcal{C}}N_{c}
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
\mathbf{1}[\mathrm{valid}(y_{i})]
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

对每个样本，令 $M_{i}$ 为非空检查点的数量。只有 $M_{i}>0$ 时，下面的关键词得分才有定义；它是便于定位坏例的字符串指标，不是语义正确率。

```math
K_{i}
=
\frac{1}{M_{i}}
\sum_{j=1}^{M_{i}}
\mathbf{1}[q_{i,j}\subset y_{i}]
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
[s(y_{i}^{\mathrm{sft}})>s(y_{i}^{\mathrm{base}})]
```

更重要的是回归率：

```math
R_{\mathrm{reg}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[s(y_{i}^{\mathrm{sft}})\lt s(y_{i}^{\mathrm{base}})]
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

比较对象、评估分组和指标口径已经固定。下一节把它们落成可保存逐样本输出的脚本，并写出有边界的实验结论。

前一节定义了 base/SFT 的比较条件、评估数据和指标。本节把它们落实为脚本，保留原始输出与逐样本分数，再从失败样本和分桶结果写出可复核的结论。示例使用教学规模数据，只演示评估流程，不代表真实模型效果。

## 4.6（续）：自动化与报告

### 4.6.8 自动化 base/SFT 对比脚本

下面的骨架使用 Alpaca 风格模板，实际聊天模型应把 `build_prompt` 换成目标 tokenizer 的 chat template：

它假设 `sft_dir` 是已经保存好的全参数 SFT 目录；如果实际产物是 LoRA/QLoRA adapter，应先用训练时相同的 base model 挂载 adapter，不能把 adapter 目录当成完整 CausalLM 加载。

示例中的 `sshleifer/tiny-gpt2` 固定到 `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`，只适合验证加载、生成和结果落盘的接口流程；它不能支持任何任务质量或微调收益结论。真实评估应替换成固定 revision 的目标 base/SFT 模型，并使用独立、分桶且经过审查的评估集。`sft_dir` 是本地训练产物，其训练 base revision、tokenizer 和文件 hash 应随评估报告保存。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(item):
    if item.get("input", "").strip():
        return (
            "### Instruction:\n"
            + item["instruction"].strip()
            + "\n\n### Input:\n"
            + item["input"].strip()
            + "\n\n### Response:\n"
        )
    return (
        "### Instruction:\n"
        + item["instruction"].strip()
        + "\n\n### Response:\n"
    )


@torch.no_grad()
def generate(model, tokenizer, prompt, device):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[-1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=pad_id,
    )
    return tokenizer.decode(
        output_ids[0, prompt_length:],
        skip_special_tokens=True,
    )


def keyword_score(output, points):
    if not points:
        return None
    if any(not isinstance(point, str) or not point.strip() for point in points):
        raise ValueError("points must contain non-empty strings")
    lower = output.lower()
    return sum(point.lower() in lower for point in points) / len(points)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = "sshleifer/tiny-gpt2"
    base_revision = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
    sft_dir = "outputs/full_sft_tiny/final"

    base_tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        revision=base_revision,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        revision=base_revision,
    ).to(device)
    sft_tokenizer = AutoTokenizer.from_pretrained(sft_dir)
    sft_model = AutoModelForCausalLM.from_pretrained(sft_dir).to(device)

    eval_items = [
        {
            "id": "overfit",
            "instruction": "解释什么是过拟合。",
            "input": "",
            "expected_points": ["训练集", "泛化"],
            "format": None,
        },
        {
            "id": "translation",
            "instruction": "翻译成英文。",
            "input": "我喜欢机器学习。",
            "expected_points": ["machine learning"],
            "format": None,
        },
    ]

    rows = []
    if not eval_items:
        raise ValueError("evaluation set must not be empty")
    for eval_tokenizer in (base_tokenizer, sft_tokenizer):
        if eval_tokenizer.pad_token_id is None:
            if eval_tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs pad_token_id or eos_token_id")
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
    if base_tokenizer.get_vocab() != sft_tokenizer.get_vocab():
        raise ValueError(
            "base and SFT tokenizers differ; compare with an explicit protocol"
        )
    for item in eval_items:
        prompt = build_prompt(item)
        base_output = generate(base_model, base_tokenizer, prompt, device)
        sft_output = generate(sft_model, sft_tokenizer, prompt, device)
        rows.append({
            "id": item["id"],
            "category": item.get("category", "general"),
            "prompt": prompt,
            "base_output": base_output,
            "sft_output": sft_output,
            "base_tokenizer": base_tokenizer.name_or_path,
            "sft_tokenizer": sft_tokenizer.name_or_path,
            "base_keyword": keyword_score(
                base_output,
                item["expected_points"],
            ),
            "sft_keyword": keyword_score(
                sft_output,
                item["expected_points"],
            ),
        })

    output_path = Path("outputs/eval_sft.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```
脚本保存原始输出而不是只保存分数，是因为任何自动指标都需要事后抽查。尤其是关键词得分高但语义错误的样本，只有保留原文才能查出评分器哪里判断错了。

### 4.6.9 不依赖模型的评估指标实验

下面的程序模拟已经生成的 base/SFT 输出，展示格式通过率、检查点、胜出样本和回归样本：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import json


items = [
    {
        "id": "definition",
        "points": ["训练集", "泛化"],
        "format": None,
    },
    {
        "id": "json",
        "points": ["正则化", "dropout"],
        "format": "json",
    },
    {
        "id": "unknown",
        "points": ["无法确认"],
        "format": None,
    },
]

base_outputs = {
    "definition": "模型记住了训练集。",
    "json": "可以多训练。",
    "unknown": "这篇论文证明了一个新算法。",
}
sft_outputs = {
    "definition": "训练集表现好但未见数据泛化差。",
    "json": '{"methods": ["正则化", "dropout"]}',
    "unknown": "无法确认，需要提供论文来源。",
}

if not items:
    raise ValueError("metric toy experiment requires a non-empty evaluation set")
item_ids = {item["id"] for item in items}
if set(base_outputs) != set(sft_outputs) or set(base_outputs) != item_ids:
    raise ValueError("items and base/SFT outputs must have identical ids")


def point_score(output, points):
    if not points:
        return None
    if any(not isinstance(point, str) or not point.strip() for point in points):
        raise ValueError("points must contain non-empty strings")
    lower = output.lower()
    return sum(point.lower() in lower for point in points) / len(points)


def format_ok(output, format_name):
    if format_name is None:
        return None
    if format_name != "json":
        raise ValueError(f"unsupported format: {format_name}")
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


def score(output, item):
    point = point_score(output, item["points"])
    format_result = format_ok(output, item["format"])
    if point is None:
        raise ValueError(f"item {item['id']} has no scoring points")
    return point + (0.1 * int(format_result) if format_result is not None else 0.0)


rows = []
for item in items:
    base = base_outputs[item["id"]]
    sft = sft_outputs[item["id"]]
    base_score = score(base, item)
    sft_score = score(sft, item)
    rows.append({
        "id": item["id"],
        "base": base_score,
        "sft": sft_score,
        "win": sft_score > base_score,
        "regression": sft_score < base_score,
        "format_base": format_ok(base, item["format"]),
        "format_sft": format_ok(sft, item["format"]),
    })

format_rows = [row for row in rows if row["format_sft"] is not None]
format_rate = (
    sum(row["format_sft"] for row in format_rows) / len(format_rows)
    if format_rows
    else None
)
wins = sum(row["win"] for row in rows)
regressions = sum(row["regression"] for row in rows)

print(
    "sft_format_rate=",
    None if format_rate is None else round(format_rate, 3),
)
print("sft_wins=", wins)
print("sft_regressions=", regressions)
print("rows=", rows)
```
该实验不证明 SFT 一定优于基座，只说明评估脚本应该输出可解释的逐样本信息，而不是只输出一个总分。

### 4.6.10 找出失败样本的问题原因

微调后变差时，按现象分类比盲目增加数据更有效：

```text
输出为空或极短：检查 EOS、pad_token、生成长度和模板末尾。
回答只复述 prompt：检查 labels mask 是否把 prompt 当成目标。
格式通过率下降：检查训练样本格式和模板一致性。
训练 loss 降而独立任务不变：检查数据覆盖、评估分布和过拟合。
通用能力下降：检查学习率、训练步数、数据混合比例和回归集。
无依据断言增加：加入来源要求、拒答样本和事实性检查。
```

每个坏例应保留 prompt、base 输出、SFT 输出、评分、数据类别和可能原因。这样下一轮数据或配置变化才能验证是否真正修复了问题。

### 4.6.11 评估报告的最小结构

一份可复查的报告应包含：

```text
模型与 base revision。
tokenizer、模板和微调方法。
数据规模、有效监督 token 数和划分规则。
训练配置、dtype、设备和 checkpoint。
评估集类别、每类样本数和生成参数。
base/SFT 原始输出与指标。
胜出样本、回归样本和失败原因分析。
已知限制与下一步实验。
```

结论应避免把单一 loss 说成能力证明。例如：

```text
在同一模板和固定生成参数下，SFT 模型在格式任务上的解析通过率提高，
但在未覆盖的推理类别上没有稳定提升；同时保留了三条回归样本，
因此当前结果只能支持“目标格式适配改善”，不能支持“通用能力全面提升”。
```

这类结论把证据范围和未知范围分开，是书写实验记录时比“微调成功”更可靠的表达。

若 20 条安全样本中有 2 条从 base 合规变成 SFT 不合规，即使整体平均分上升，也要报告这两条回归及其严重度。

评估集的分桶需要在看到 SFT 输出之前定义，例如格式、事实性、拒答边界、长输入、少数语言和安全风险。样本少的桶要同时写计数：2/5 与 40/100 的比例相同，证据强度却不同。人工评分应盲化模型身份、使用固定 rubric，并保留评审分歧。也可以让另一个大模型按 rubric 初筛（LLM judge），但要用人工样本校准并回看原始输出；它不能独自证明回答正确。

成对比较应保留每个 prompt 的 base/SFT 结果，而不仅是两个独立平均值。若一条样本两边都成功，它不应被计为 SFT 的“胜出”；若同一用户任务有多个相关 prompt，统计时还要考虑样本相关，不能把它们全部当成独立用户。报告中至少展示胜出、回归、平局计数和关键类别分母，并让评审者能回到原始输出。

一次 SFT 实验可以用固定 40 条样本做探索性回归：若 overall 由 25/40 提到 29/40，但代码格式桶从 8/10 降到 6/10，且安全桶出现 1 条新违规，结论不能只写“整体提升 10 个百分点”。还要查看变化是否来自样本难度、评分规则或采样方差，并按预先定义的安全护栏处理风险。这个样例的计数只演示报告结构，不代表统计显著性或真实模型效果。

### 4.6.12 本章资料与证据

本章关于接口行为的依据优先采用官方文档，关于 LoRA 和 QLoRA 机制的依据采用原始论文。文档参数会随版本变化，复现实验时仍需记录本地版本和模型 revision。

- Transformers v5.17.0 模型加载与保存：<https://huggingface.co/docs/transformers/v5.17.0/main_classes/model>
- Transformers v5.17.0 文本生成：<https://huggingface.co/docs/transformers/v5.17.0/main_classes/text_generation>
- Transformers v5.17.0 tokenizer 与 chat template：<https://huggingface.co/docs/transformers/v5.17.0/main_classes/tokenizer>
- Transformers v5.17.0 Trainer：<https://huggingface.co/docs/transformers/v5.17.0/main_classes/trainer>
- PEFT v0.18.0 LoRA API：<https://huggingface.co/docs/peft/v0.18.0/package_reference/lora>
- PEFT v0.18.0 量化训练准备：<https://huggingface.co/docs/peft/v0.18.0/developer_guides/quantization>
- Transformers v5.17.0 bitsandbytes 量化：<https://huggingface.co/docs/transformers/v5.17.0/quantization/bitsandbytes>
- TRL v1.14.1 SFTTrainer：<https://huggingface.co/docs/trl/v1.14.1/sft_trainer>
- PyTorch v2.14.0 CrossEntropyLoss：<https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html>
- tiny-gpt2 固定 revision 文件树：<https://huggingface.co/sshleifer/tiny-gpt2/tree/5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be>
- Qwen2.5-0.5B 固定 revision 文件树：<https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/b937cc073b55e980e6f21d5f8b7a77818efcf0eb>
- LoRA 原论文：<https://arxiv.org/abs/2106.09685>
- QLoRA 原论文：<https://arxiv.org/abs/2305.14314>
- LoFT 官方论文摘要：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html>

官方文档适合确认 API、参数和版本说明；论文适合确认低秩适配、NF4、double quantization 和量化基座训练的原始定义；博客、二手教程和模型卡可以补充实践经验，但不应在没有核对原始来源时被当作普遍规律。
