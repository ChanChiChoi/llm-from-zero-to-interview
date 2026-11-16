# 第 4 章（第七篇）：自动评估与实验报告

前一篇定义了 base/SFT 的比较条件、评估数据和指标。本篇把它们落实为脚本，保留原始输出与逐样本分数，再从失败样本和分桶结果写出可复核的结论。示例使用教学规模数据，只演示评估流程，不代表真实模型效果。

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
脚本保存原始输出而不是只保存分数，是因为任何自动指标都需要事后抽查。尤其是关键词得分高但语义错误的样本，只有保留原文才能归因。

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

### 4.6.10 失败样本要归因

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
胜出样本、回归样本和失败归因。
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
