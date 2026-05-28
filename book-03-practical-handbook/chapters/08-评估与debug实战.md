# 第八部分：评估与 Debug 实战

## 第 44 讲：构建小型 benchmark

### 本讲目标

学完本讲，你应该能做到五件事：

1. 理解为什么大模型项目必须有自己的 benchmark。
2. 区分通用 benchmark、业务 benchmark 和回归测试集。
3. 设计一个小型但有效的评测数据集。
4. 写出可运行的自动评测脚本。
5. 能在面试中讲清模型评估体系怎么搭建。

前面几部分我们做了训练、微调、推理、RAG 和 Agent。

但一个模型到底有没有变好，不能只靠主观感觉。

你需要 benchmark。

Benchmark 的本质不是“搞一个很大的榜单”。

它的本质是：

```text
用一组固定、可复现、有代表性的样例，衡量模型在目标能力上的表现。
```

在公司真实项目中，小型 benchmark 往往比公开大榜更重要。

因为公开大榜不一定覆盖你的业务问题。

---

### 一、为什么需要小型 benchmark

如果没有 benchmark，你会遇到几个问题。

#### 问题 1：不知道改动是否有效

你改了 prompt、换了模型、做了 SFT、调了 decoding 参数。

如果没有固定测试集，只能凭感觉判断。

这很危险。

#### 问题 2：只看单个样例容易误判

模型在一个样例上变好，不代表整体变好。

模型在一个样例上变差，也不代表整体变差。

需要统计结果。

#### 问题 3：无法发现能力退化

微调可能让模型在目标任务上变好，但在通用能力、安全性、格式遵循上变差。

Benchmark 可以做回归测试。

#### 问题 4：团队无法对齐

没有统一评测集时，每个人都拿自己的例子说模型好坏。

最后讨论会变成主观争论。

---

### 二、三类常见评测集

#### 1. 公开通用 benchmark

例如：

```text
MMLU
GSM8K
HumanEval
CMMLU
C-Eval
MT-Bench
```

它们适合比较通用能力。

但缺点是：

```text
不一定覆盖业务场景。
可能存在数据污染。
评测成本较高。
不能直接反映用户体验。
```

#### 2. 业务 benchmark

围绕你的产品场景构建。

例如：

```text
客服问答准确率
合同条款抽取
代码修复任务
RAG 问答引用正确性
Agent 工具调用成功率
```

业务 benchmark 是公司项目最核心的评测集。

#### 3. 回归测试集

用于防止新版本破坏旧能力。

例如：

```text
曾经线上出错的 case
用户投诉样例
安全红线样例
格式要求严格的样例
高频业务问题
```

回归测试集不一定大，但必须稳定。

---

### 三、小型 benchmark 应该多大

面试中不要说“越大越好”。

更合理的回答是：

```text
先做一个 50-200 条的小型高质量评测集，用于快速迭代；稳定后再扩展到更大规模，并分层覆盖不同任务类型和难度。
```

小型 benchmark 的优势：

```text
构建快。
人工检查成本低。
迭代速度快。
方便定位问题。
```

缺点：

```text
统计置信度有限。
覆盖面不够。
容易被过拟合。
```

所以小型 benchmark 适合早期研发和回归测试。

上线前还需要更全面评估。

---

### 四、Benchmark 样例格式设计

一个简单样例可以长这样：

```json
{
  "id": "qa_001",
  "category": "factual_qa",
  "difficulty": "easy",
  "input": "Transformer 中 self-attention 的作用是什么？",
  "reference": "self-attention 用于让序列中每个 token 根据其他 token 的信息更新表示。",
  "scoring": "semantic"
}
```

建议字段：

```text
id：样例唯一标识。
category：任务类型。
difficulty：难度。
input：模型输入。
reference：参考答案。
scoring：评分方式。
metadata：可选补充信息。
```

对于选择题：

```json
{
  "id": "mc_001",
  "category": "multiple_choice",
  "input": "下列哪一项是 LayerNorm 的主要作用？\nA. 降低词表大小\nB. 稳定激活分布\nC. 增加序列长度\nD. 替代 tokenizer",
  "reference": "B",
  "scoring": "exact_match"
}
```

对于 RAG：

```json
{
  "id": "rag_001",
  "category": "rag_qa",
  "input": "根据文档回答：产品 A 的退款周期是多久？",
  "context": "产品 A 支持 7 天无理由退款，退款通常在 3 个工作日内到账。",
  "reference": "退款通常在 3 个工作日内到账。",
  "scoring": "faithfulness"
}
```

不同任务可以使用同一套外层格式。

这样评测脚本更容易统一。

---

### 五、创建一个小型评测文件

假设保存为 `data/mini_benchmark.jsonl`。

JSONL 的好处是每行一个样例，方便追加和流式读取。

```json
{"id":"mc_001","category":"multiple_choice","difficulty":"easy","input":"下列哪一项是 LayerNorm 的主要作用？\nA. 降低词表大小\nB. 稳定激活分布\nC. 增加序列长度\nD. 替代 tokenizer","reference":"B","scoring":"exact_match"}
{"id":"qa_001","category":"short_qa","difficulty":"medium","input":"为什么 decoder-only 模型需要 causal mask？","reference":"为了防止当前位置看到未来 token，保证自回归生成只依赖历史上下文。","scoring":"keyword"}
{"id":"fmt_001","category":"format_following","difficulty":"easy","input":"请只输出 JSON，字段包括 name 和 age。用户：张三，18 岁。","reference":"json_only","scoring":"json_valid"}
```

实际项目中不要只放容易题。

建议分层：

```text
简单题：验证基础能力。
中等题：验证主要任务能力。
困难题：暴露边界问题。
安全题：验证拒答和权限边界。
格式题：验证结构化输出能力。
```

---

### 六、读取 benchmark

```python
import json
from pathlib import Path


def load_jsonl(path):
    examples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples
```

测试：

```python
examples = load_jsonl("data/mini_benchmark.jsonl")
print(len(examples))
print(examples[0])
```

---

### 七、定义模型调用接口

为了让评测脚本可替换模型，先定义统一接口。

```python
def call_model(prompt):
    """这里用占位函数，真实项目中可以替换成 OpenAI、vLLM 或本地模型调用。"""
    if "LayerNorm" in prompt:
        return "B"
    if "causal mask" in prompt:
        return "因为它可以防止模型看到未来 token，保证自回归生成。"
    if "只输出 JSON" in prompt:
        return '{"name": "张三", "age": 18}'
    return "不知道"
```

工程上建议把模型调用封装成一个函数。

这样你可以快速比较：

```text
baseline model
微调后模型
不同 prompt
不同 decoding 参数
不同 RAG pipeline
```

---

### 八、实现 exact match 评分

选择题、分类题、固定格式题可以用 exact match。

```python
def exact_match_score(prediction, reference):
    pred = prediction.strip()
    ref = reference.strip()
    return 1.0 if pred == ref else 0.0
```

有时模型会输出解释：

```text
答案是 B，因为 LayerNorm 可以稳定激活分布。
```

这时可以做简单抽取：

```python
def extract_choice(text):
    text = text.strip()
    for choice in ["A", "B", "C", "D"]:
        if text == choice or text.startswith(choice):
            return choice
        if f"答案是 {choice}" in text or f"答案：{choice}" in text:
            return choice
    return text
```

评分：

```python
def choice_score(prediction, reference):
    return 1.0 if extract_choice(prediction) == reference else 0.0
```

---

### 九、实现 keyword 评分

开放问答很难 exact match。

可以先用关键词作为轻量评分。

```python
def keyword_score(prediction, reference_keywords):
    hit = 0
    for keyword in reference_keywords:
        if keyword in prediction:
            hit += 1
    return hit / max(len(reference_keywords), 1)
```

例如：

```python
keywords = ["未来 token", "自回归", "历史"]
score = keyword_score("防止模型看到未来 token，保证自回归生成", keywords)
print(score)
```

关键词评分优点：

```text
简单。
便宜。
可复现。
适合早期快速迭代。
```

缺点：

```text
不懂语义。
容易漏判同义表达。
容易被关键词堆砌欺骗。
```

所以它适合作为粗筛，不适合作为最终唯一指标。

---

### 十、实现 JSON 格式评分

结构化输出任务可以检查 JSON 是否可解析、字段是否齐全。

```python
def json_valid_score(prediction, required_keys=None):
    required_keys = required_keys or []
    try:
        obj = json.loads(prediction)
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(obj, dict):
        return 0.0

    for key in required_keys:
        if key not in obj:
            return 0.0

    return 1.0
```

使用：

```python
score = json_valid_score('{"name": "张三", "age": 18}', required_keys=["name", "age"])
print(score)
```

格式遵循是大模型产品中非常重要的能力。

尤其是模型输出要进入下游程序时。

---

### 十一、统一评分函数

```python
def score_example(example, prediction):
    scoring = example.get("scoring")

    if scoring == "exact_match":
        return exact_match_score(prediction, example["reference"])

    if scoring == "choice":
        return choice_score(prediction, example["reference"])

    if scoring == "keyword":
        keywords = example.get("keywords")
        if keywords is None:
            keywords = example["reference"].split()
        return keyword_score(prediction, keywords)

    if scoring == "json_valid":
        return json_valid_score(prediction, required_keys=example.get("required_keys", []))

    raise ValueError(f"未知评分方式：{scoring}")
```

实际评测中，建议每个样例明确写出 `scoring`。

不要让脚本猜。

---

### 十二、完整评测脚本

```python
import json
from pathlib import Path


def load_jsonl(path):
    examples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def call_model(prompt):
    if "LayerNorm" in prompt:
        return "B"
    if "causal mask" in prompt:
        return "因为它可以防止模型看到未来 token，保证自回归生成。"
    if "只输出 JSON" in prompt:
        return '{"name": "张三", "age": 18}'
    return "不知道"


def exact_match_score(prediction, reference):
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def keyword_score(prediction, keywords):
    hit = sum(1 for keyword in keywords if keyword in prediction)
    return hit / max(len(keywords), 1)


def json_valid_score(prediction, required_keys=None):
    required_keys = required_keys or []
    try:
        obj = json.loads(prediction)
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(obj, dict):
        return 0.0

    return 1.0 if all(key in obj for key in required_keys) else 0.0


def score_example(example, prediction):
    scoring = example["scoring"]

    if scoring == "exact_match":
        return exact_match_score(prediction, example["reference"])

    if scoring == "keyword":
        keywords = example.get("keywords", example["reference"].split())
        return keyword_score(prediction, keywords)

    if scoring == "json_valid":
        return json_valid_score(prediction, example.get("required_keys", []))

    raise ValueError(f"未知评分方式：{scoring}")


def evaluate(path):
    examples = load_jsonl(path)
    results = []

    for example in examples:
        prediction = call_model(example["input"])
        score = score_example(example, prediction)
        results.append({
            "id": example["id"],
            "category": example["category"],
            "score": score,
            "prediction": prediction,
            "reference": example["reference"],
        })

    avg_score = sum(item["score"] for item in results) / max(len(results), 1)
    return avg_score, results


if __name__ == "__main__":
    avg_score, results = evaluate("data/mini_benchmark.jsonl")
    print(f"Average score: {avg_score:.4f}")
    for item in results:
        print(item)
```

这就是一个最小可用评测框架。

它不复杂，但已经具备三个关键点：

```text
固定数据集。
固定模型调用接口。
固定评分逻辑。
```

---

### 十三、按类别统计结果

只看平均分可能掩盖问题。

例如模型选择题很好，但 JSON 格式很差。

所以要按类别统计。

```python
from collections import defaultdict


def summarize_by_category(results):
    buckets = defaultdict(list)

    for item in results:
        buckets[item["category"]].append(item["score"])

    summary = {}
    for category, scores in buckets.items():
        summary[category] = sum(scores) / len(scores)

    return summary
```

输出示例：

```text
{
  "multiple_choice": 0.92,
  "short_qa": 0.76,
  "format_following": 0.55
}
```

面试中要强调：

```text
平均分只是总览，真正定位问题要看 category、difficulty 和失败样例。
```

---

### 十四、保存评测结果

每次评测都应该保存结果，方便比较版本。

```python
from datetime import datetime


def save_results(results, model_name, output_dir="outputs/eval_runs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{model_name}_{timestamp}.jsonl"

    with path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return path
```

保存字段建议包含：

```text
model_name
prompt_version
decoding_config
benchmark_version
git_commit
timestamp
prediction
score
error_type
```

这能避免“上周那个好版本是哪一个”的问题。

---

### 十五、失败样例分析

Benchmark 的价值不只是给分。

更重要的是定位失败模式。

可以给失败样例打标签：

```text
知识错误
推理错误
格式错误
拒答错误
幻觉
引用错误
工具调用错误
安全边界错误
```

失败样例记录：

```json
{
  "id": "qa_017",
  "prediction": "...",
  "reference": "...",
  "score": 0,
  "error_type": "hallucination",
  "notes": "模型编造了文档中不存在的退款规则。"
}
```

评测不是终点。

评测结果应该反过来指导：

```text
补数据。
改 prompt。
调检索。
调 decoding。
重新微调。
加安全规则。
```

---

### 十六、如何避免 benchmark 被污染

污染包括两类。

#### 1. 训练数据污染

评测样例出现在训练集中。

这会让分数虚高。

#### 2. 人为过拟合

团队反复针对 benchmark 调 prompt 或补规则。

这会让 benchmark 分数变高，但真实用户体验不一定变好。

应对方式：

```text
训练集和测试集严格隔离。
保留 hidden test set。
定期加入新线上失败样例。
不要只优化单一分数。
用人工抽检验证自动评测。
```

---

### 十七、小型 benchmark 的面试表达

如果面试官问“你会怎么评估一个大模型应用”，可以这样回答：

```text
我会先根据业务目标构建一个小型高质量 benchmark，包含核心任务、边界 case、安全 case 和格式遵循 case。每条样例包含 input、reference、category、difficulty 和 scoring。评测时固定模型调用接口和评分逻辑，统计总体分数以及按类别、难度的分数，并保存每次运行结果。对于失败样例，我会标注错误类型，用它指导数据、prompt、检索或微调迭代。同时保留一部分 hidden set，避免过拟合 benchmark。
```

如果追问“自动评测可靠吗”，可以回答：

```text
自动评测适合快速回归和版本比较，但开放生成任务不能完全依赖自动分数。选择题、分类、JSON 格式可以自动评分；开放问答可以用关键词、语义相似度或 LLM-as-a-judge 辅助，但关键版本仍需要人工抽检和失败样例分析。
```

---

### 十八、小练习

#### 练习 1

创建一个包含 20 条样例的 `mini_benchmark.jsonl`。

#### 练习 2

为样例增加 `category` 和 `difficulty` 字段。

#### 练习 3

实现 exact match、keyword 和 JSON valid 三种评分。

#### 练习 4

把结果按 category 汇总。

#### 练习 5

挑选 5 个失败样例，标注错误类型并写出改进方向。

---

### 本讲总结

这一讲实现了一个小型 benchmark。

核心结论如下：

1. Benchmark 的核心是固定、可复现、有代表性的样例集合。
2. 公司项目中，业务 benchmark 和回归测试集通常比公开榜单更重要。
3. 小型 benchmark 适合快速迭代，但覆盖面和统计置信度有限。
4. 样例应包含 id、category、difficulty、input、reference 和 scoring。
5. 评测脚本应统一模型调用接口和评分逻辑。
6. 不能只看平均分，还要看类别、难度和失败样例。
7. 自动评测要结合人工抽检，避免数据污染和 benchmark 过拟合。

下一讲，我们讨论如何检测数据泄漏。

## 第 45 讲：检测数据泄漏

### 本讲目标

这一讲，我们实现一套数据泄漏检测流程。

学完本讲，你应该能够回答：

1. 什么是数据泄漏，为什么它会让评测结果虚高？
2. 训练集、验证集、测试集之间有哪些常见泄漏形式？
3. 如何用代码检测 exact duplicate、近似重复和标签泄漏？
4. 如何排查 benchmark 是否被训练数据污染？
5. 面试中如何系统讲清楚数据泄漏的定位和修复方案？

数据泄漏是大模型评估中非常高频、也非常容易被忽视的问题。

很多模型“看起来变强了”，实际原因不是模型能力提升，而是评测集、验证集或业务测试样例被训练数据、提示模板、人工标注过程或调参流程污染了。

在工程中，数据泄漏会导致三个严重后果：

1. 线上效果不如离线评测。
2. 模型选择结论错误。
3. 团队把优化方向建立在错误信号上。

所以，顶级大模型算法岗面试中，候选人不能只说“划分 train/val/test”，还要能讲出如何检测、如何定位、如何规避。

---

### 1. 什么是数据泄漏

数据泄漏指的是：模型在训练、调参或提示优化过程中，间接或直接接触到了本应只用于评估的数据、标签、答案或分布信息。

一个最简单的例子：

```text
训练集中出现：
问题：法国的首都是哪里？
答案：巴黎

测试集中出现：
问题：法国首都是？
答案：巴黎
```

如果模型答对了，未必说明它真的具备推理能力，可能只是记住了训练样例。

更隐蔽的例子：

```text
训练集中出现：
请判断评论情感：这家餐厅太难吃了。
标签：负面

测试集中出现：
请判断下面评论是正面还是负面：这家餐厅太难吃了。
标签：负面
```

两条样例文本不完全相同，但语义高度重复。

对于大模型来说，这类近似重复同样可能造成评测虚高。

---

### 2. 常见泄漏类型

#### 类型 1：样本级重复

训练集和测试集中存在完全相同的样例。

```text
train: {"input": "1+1=?", "output": "2"}
test:  {"input": "1+1=?", "output": "2"}
```

这是最容易检测的一类泄漏。

#### 类型 2：近似重复

样例经过轻微改写后出现在不同集合中。

```text
train: 请把“今天天气很好”翻译成英文。
test:  将“今天天气很好”翻译为英语。
```

这类泄漏不能只靠字符串完全匹配，需要使用 n-gram、编辑距离或 embedding 相似度检测。

#### 类型 3：答案泄漏

输入中直接或间接包含答案。

```text
input: 阅读下面文本并回答问题。文本：答案是 B，因为…… 问题：正确选项是什么？
label: B
```

这种问题在问答、阅读理解、多选题数据中很常见。

#### 类型 4：模板泄漏

训练和测试使用高度固定的模板，模型可以利用模板规律而非真实能力作答。

```text
所有正例都以“综上所述”结尾。
所有负例都以“但是”开头。
```

模型可能学到的是格式偏置，而不是任务本身。

#### 类型 5：时间泄漏

训练数据包含了测试时间之后才出现的信息。

例如用 2025 年网页数据训练模型，然后评测“2024 年之后的事件预测”。这会使模型看似拥有预测能力，实际只是看过未来信息。

#### 类型 6：调参泄漏

反复在测试集上调 prompt、调超参、筛 checkpoint，最终测试集变成了验证集。

这在大模型业务迭代中非常常见。

如果团队每天看同一批测试集结果，并不断针对失败样例改 prompt，那么这批测试集就不再是无偏测试集。

#### 类型 7：人工标注泄漏

标注者在标注测试集时参考了模型输出，或者把线上 badcase 修复样例同时加入训练集和评测集。

这会让评测集不再独立。

---

### 3. 问题设定

假设我们有三个 JSONL 文件：

```text
data/train.jsonl
data/valid.jsonl
data/test.jsonl
```

每行格式如下：

```json
{"id": "train_001", "input": "请把今天天气很好翻译成英文", "output": "The weather is nice today."}
```

我们希望检测：

1. `train` 和 `test` 是否存在完全重复输入。
2. `train` 和 `test` 是否存在近似重复输入。
3. `input` 中是否疑似包含 `output`。
4. `valid` 和 `test` 是否被反复复用导致调参泄漏。

下面从最基础的 exact duplicate 开始。

---

### 4. 读取 JSONL 数据

```python
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}")
            rows.append(row)
    return rows


train = load_jsonl("data/train.jsonl")
valid = load_jsonl("data/valid.jsonl")
test = load_jsonl("data/test.jsonl")

print(len(train), len(valid), len(test))
```

工程上要先做格式校验。

如果数据文件本身有脏行、缺字段、重复 id，那么后面的泄漏检测结论也不可靠。

---

### 5. 文本归一化

完全匹配之前，需要做简单归一化。

```python
import re


def normalize_text(text):
    text = str(text)
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
```

为什么要归一化？

因为下面这些文本对模型来说几乎相同：

```text
"Hello world"
" hello   world "
"HELLO WORLD"
```

如果不归一化，很多显然重复的样例会漏掉。

但归一化也不能过度。

例如在代码任务中，大小写、空格、换行可能是语义的一部分；在数学任务中，`x + y` 和 `xy` 不能随意合并。所以实际项目中要按任务定制归一化规则。

---

### 6. 检测完全重复输入

```python
def build_index(rows, field="input"):
    index = {}
    for row in rows:
        key = normalize_text(row.get(field, ""))
        index.setdefault(key, []).append(row)
    return index


def find_exact_overlap(left_rows, right_rows, field="input"):
    left_index = build_index(left_rows, field=field)
    overlaps = []

    for right in right_rows:
        key = normalize_text(right.get(field, ""))
        if key in left_index:
            overlaps.append({
                "right_id": right.get("id"),
                "right_text": right.get(field),
                "left_matches": [row.get("id") for row in left_index[key]],
            })

    return overlaps


train_test_exact = find_exact_overlap(train, test, field="input")
valid_test_exact = find_exact_overlap(valid, test, field="input")

print("train-test exact overlap:", len(train_test_exact))
print("valid-test exact overlap:", len(valid_test_exact))
```

如果 `train-test exact overlap` 很高，测试集分数基本不可信。

如果 `valid-test exact overlap` 很高，说明验证集和测试集没有独立性，模型选择结果也会偏乐观。

---

### 7. 检测 input + output 完全重复

只检查 `input` 有时不够。

有些任务的输入相同但答案可以不同，例如开放式写作、摘要、对话任务。

所以还可以检查 `(input, output)` 二元组是否重复。

```python
def make_io_key(row):
    input_text = normalize_text(row.get("input", ""))
    output_text = normalize_text(row.get("output", ""))
    return input_text + "\n---OUTPUT---\n" + output_text


def find_exact_io_overlap(left_rows, right_rows):
    left_index = {}
    for row in left_rows:
        key = make_io_key(row)
        left_index.setdefault(key, []).append(row)

    overlaps = []
    for right in right_rows:
        key = make_io_key(right)
        if key in left_index:
            overlaps.append({
                "right_id": right.get("id"),
                "left_matches": [row.get("id") for row in left_index[key]],
            })

    return overlaps


io_overlaps = find_exact_io_overlap(train, test)
print("train-test input-output exact overlap:", len(io_overlaps))
```

面试中可以补一句：

> 对于分类任务，通常重点查 input 重复；对于生成任务，同时查 input 重复和 input-output 成对重复。

---

### 8. 用 Jaccard 相似度检测近似重复

近似重复的核心是：两个样例没有完全相同，但 token 集合高度重合。

先实现一个简单的字符 n-gram Jaccard 相似度。

```python
def char_ngrams(text, n=3):
    text = normalize_text(text)
    if len(text) <= n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(a, b):
    a_set = char_ngrams(a)
    b_set = char_ngrams(b)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)
```

测试一下：

```python
pairs = [
    ("请把今天天气很好翻译成英文", "将今天天气很好翻译为英语"),
    ("北京是中国的首都", "苹果公司发布了新手机"),
]

for a, b in pairs:
    print(a, b, jaccard_similarity(a, b))
```

对于中文短文本，字符 n-gram 往往比简单按空格分词更稳。

对于英文长文本，可以改成 word n-gram。

---

### 9. 近似重复检测脚本

最直接的做法是两两比较 `train` 和 `test`。

```python
def find_near_duplicates(left_rows, right_rows, threshold=0.75, max_matches_per_row=3):
    results = []

    for right in right_rows:
        right_text = right.get("input", "")
        matches = []

        for left in left_rows:
            left_text = left.get("input", "")
            score = jaccard_similarity(left_text, right_text)
            if score >= threshold:
                matches.append({
                    "left_id": left.get("id"),
                    "score": score,
                    "left_text": left_text,
                })

        matches.sort(key=lambda x: x["score"], reverse=True)
        if matches:
            results.append({
                "right_id": right.get("id"),
                "right_text": right_text,
                "matches": matches[:max_matches_per_row],
            })

    return results


near_duplicates = find_near_duplicates(train, test, threshold=0.75)
print("near duplicates:", len(near_duplicates))
```

这个版本适合小数据集。

如果训练集有几百万条，不能这样暴力两两比较，因为复杂度是：

```text
O(len(train) * len(test))
```

大规模场景可以使用：

1. MinHash + LSH。
2. SimHash。
3. BM25 召回候选，再做精排。
4. embedding 向量召回，再做人工抽检。

面试中要强调：近似重复检测通常是“召回候选 + 精排确认”，不是盲目全量两两比较。

---

### 10. 用 embedding 检测语义重复

有些重复不体现为字面相似，而体现为语义相似。

例如：

```text
train: 这家餐厅服务很差，我不会再来了。 判断情感。
test:  用户表示餐厅服务糟糕且不想再次消费，该评论情感是什么？
```

字面不完全重合，但语义高度相似。

在实际工程中，可以使用 embedding 模型把输入编码成向量，然后检索最近邻。

伪代码如下：

```python
import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# embeddings_train: shape [num_train, dim]
# embeddings_test: shape [num_test, dim]
# 实际项目中一般用向量数据库或 faiss 做近邻检索。
```

embedding 检测的优点：

1. 能发现改写、同义表达和语义重复。
2. 适合长文本和开放式任务。
3. 可以和人工抽检结合，形成污染风险报告。

缺点：

1. 阈值不好统一。
2. embedding 模型本身会引入偏差。
3. 高相似不一定等于泄漏，仍需人工确认。

所以不能只看 embedding 分数，要结合样例内容判断。

---

### 11. 检测答案泄漏

答案泄漏的最简单检测方式：检查 `output` 是否直接出现在 `input` 中。

```python
def find_label_leakage(rows, min_output_len=2):
    leaks = []
    for row in rows:
        input_text = normalize_text(row.get("input", ""))
        output_text = normalize_text(row.get("output", ""))

        if len(output_text) < min_output_len:
            continue

        if output_text and output_text in input_text:
            leaks.append({
                "id": row.get("id"),
                "input": row.get("input"),
                "output": row.get("output"),
            })

    return leaks


test_label_leaks = find_label_leakage(test)
print("test label leaks:", len(test_label_leaks))
```

这个规则很粗糙，但很有用。

它能快速发现很多明显问题，例如：

```text
input: 已知正确答案为“巴黎”，请回答法国首都是什么？
output: 巴黎
```

对于选择题，还可以检查选项和解析是否泄漏答案。

```python
def find_choice_answer_leakage(rows):
    leaks = []
    answer_patterns = [
        "答案是{}",
        "正确答案是{}",
        "选{}",
        "应选择{}",
    ]

    for row in rows:
        input_text = normalize_text(row.get("input", ""))
        output_text = normalize_text(row.get("output", ""))
        for pattern in answer_patterns:
            marker = normalize_text(pattern.format(output_text))
            if marker in input_text:
                leaks.append({
                    "id": row.get("id"),
                    "marker": marker,
                    "input": row.get("input"),
                    "output": row.get("output"),
                })
                break

    return leaks
```

---

### 12. 检测模板偏置

模板偏置不是严格意义上的 train-test 重复，但会造成虚高。

例如一个二分类任务：

```text
正例 input 经常包含“非常推荐”。
负例 input 经常包含“不建议购买”。
```

这本来可能是任务有效特征。

但如果数据构造方式导致所有正例都有某个模板词，所有负例都有另一个模板词，模型就可能只学模板。

可以统计标签和关键词的共现。

```python
from collections import Counter, defaultdict


def token_presence_by_label(rows, keywords):
    stats = defaultdict(Counter)

    for row in rows:
        label = str(row.get("output", ""))
        text = normalize_text(row.get("input", ""))
        for keyword in keywords:
            if normalize_text(keyword) in text:
                stats[label][keyword] += 1

    return stats


keywords = ["综上所述", "但是", "因此", "正确答案", "无法判断"]
stats = token_presence_by_label(train, keywords)

for label, counter in stats.items():
    print(label, counter)
```

如果某些关键词几乎只出现在某个标签下，就要怀疑模板偏置。

更完整的做法是训练一个很弱的 baseline，只使用模板特征或浅层词袋特征。如果弱模型也能取得异常高分，说明数据集可能存在捷径。

---

### 13. 检测时间泄漏

如果样例带有时间字段，可以检查训练集时间是否晚于测试集设计时间。

```python
from datetime import datetime


def parse_date(date_str):
    if not date_str:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d")


def find_time_leakage(train_rows, test_cutoff):
    cutoff = parse_date(test_cutoff)
    leaks = []

    for row in train_rows:
        created_at = parse_date(row.get("created_at"))
        if created_at is not None and created_at > cutoff:
            leaks.append(row)

    return leaks


future_train_rows = find_time_leakage(train, test_cutoff="2024-12-31")
print("future train rows:", len(future_train_rows))
```

时间泄漏在以下任务中特别重要：

1. 金融预测。
2. 新闻问答。
3. 时事知识评测。
4. 用户行为预测。
5. 推荐系统。

对于大模型，网页预训练语料的时间范围也要关注。如果 benchmark 发布很早，模型预训练中可能已经包含相关题目或答案。

---

### 14. 检测调参泄漏

调参泄漏不一定能从数据文件中直接检测出来，需要结合实验日志。

一个实用做法是记录每次评测使用的数据集名称、版本和目的。

```json
{"run_id": "run_001", "dataset": "test_v1", "purpose": "final_report", "date": "2025-01-10"}
{"run_id": "run_002", "dataset": "test_v1", "purpose": "prompt_tuning", "date": "2025-01-11"}
{"run_id": "run_003", "dataset": "test_v1", "purpose": "prompt_tuning", "date": "2025-01-12"}
```

如果同一测试集被多次用于 prompt tuning，就要标记为高风险。

```python
def detect_eval_reuse(log_rows, max_tuning_runs=1):
    tuning_count = Counter()

    for row in log_rows:
        dataset = row.get("dataset")
        purpose = row.get("purpose")
        if purpose in {"prompt_tuning", "hyperparam_tuning", "checkpoint_selection"}:
            tuning_count[dataset] += 1

    risks = []
    for dataset, count in tuning_count.items():
        if count > max_tuning_runs:
            risks.append({"dataset": dataset, "tuning_runs": count})

    return risks
```

更规范的评估流程是：

1. `dev set` 用于日常调 prompt 和调参。
2. `test set` 只在关键节点评估。
3. `hidden test set` 只由评测平台或独立人员维护。
4. 线上 A/B 作为最终验证。

---

### 15. 生成泄漏检测报告

把上面的检查整合成一个报告。

```python
def leakage_report(train, valid, test):
    report = {}

    report["train_test_exact_input"] = find_exact_overlap(train, test, field="input")
    report["valid_test_exact_input"] = find_exact_overlap(valid, test, field="input")
    report["train_test_exact_io"] = find_exact_io_overlap(train, test)
    report["train_test_near_duplicates"] = find_near_duplicates(
        train,
        test,
        threshold=0.75,
        max_matches_per_row=3,
    )
    report["test_label_leaks"] = find_label_leakage(test)
    report["test_choice_answer_leaks"] = find_choice_answer_leakage(test)

    summary = {
        "train_test_exact_input_count": len(report["train_test_exact_input"]),
        "valid_test_exact_input_count": len(report["valid_test_exact_input"]),
        "train_test_exact_io_count": len(report["train_test_exact_io"]),
        "train_test_near_duplicate_count": len(report["train_test_near_duplicates"]),
        "test_label_leak_count": len(report["test_label_leaks"]),
        "test_choice_answer_leak_count": len(report["test_choice_answer_leaks"]),
    }

    return summary, report


summary, detail = leakage_report(train, valid, test)
print(json.dumps(summary, ensure_ascii=False, indent=2))
```

输出示例：

```json
{
  "train_test_exact_input_count": 12,
  "valid_test_exact_input_count": 3,
  "train_test_exact_io_count": 5,
  "train_test_near_duplicate_count": 41,
  "test_label_leak_count": 2,
  "test_choice_answer_leak_count": 1
}
```

如果报告中出现大量 near duplicate，就应该抽样人工复核。

---

### 16. 保存可审计结果

泄漏检测不能只打印数量，还要保存明细，方便人工检查。

```python
def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


save_json("reports/leakage_summary.json", summary)
save_json("reports/leakage_detail.json", detail)
```

实际项目中，报告至少应包含：

1. 检测日期。
2. 数据版本。
3. 检测规则版本。
4. 各类泄漏数量。
5. 高风险样例明细。
6. 人工复核结论。
7. 修复动作。

这样后续如果评测结果异常，可以追溯当时的数据状态。

---

### 17. 如何修复数据泄漏

发现泄漏后，不是简单删除几条样例就结束。

需要根据泄漏来源处理。

#### 情况 1：训练集污染测试集

处理方式：

1. 从训练集中删除与测试集重复或近似重复的样例。
2. 重新训练或重新微调。
3. 重新评估并对比修复前后结果。

#### 情况 2：测试集被反复调参污染

处理方式：

1. 冻结当前测试集，不再作为最终测试。
2. 新建 hidden test set。
3. 把旧测试集降级为 dev set 或 regression set。

#### 情况 3：标签泄漏

处理方式：

1. 修复数据构造模板。
2. 删除输入中直接包含答案的字段。
3. 重做一批干净样例。
4. 更新数据校验规则，防止再次生成。

#### 情况 4：时间泄漏

处理方式：

1. 按时间重新切分数据。
2. 确保训练集时间早于验证集和测试集。
3. 对时事类评测明确记录 cutoff date。

---

### 18. 大模型 benchmark 污染怎么判断

公开 benchmark 的污染更难检测，因为你通常拿不到完整预训练语料。

常用方法包括：

1. 检查 benchmark 是否已公开很久。
2. 搜索题目文本是否大量出现在网页、GitHub、论坛和题库中。
3. 比较模型在原题和改写题上的性能差距。
4. 构造新题、私有题和时间后验题进行对照。
5. 检查模型是否能输出题目解析中的特殊表述。

例如：

```text
原题准确率：92%
轻微改写后准确率：89%
换数字后准确率：55%
```

这种现象说明模型可能记住了原题模式，而不是掌握了解法。

对于数学、代码和推理 benchmark，尤其要做“变量替换、数值替换、顺序扰动、语义改写”的鲁棒性检查。

---

### 19. 面试高频问法

#### 问法 1：怎么判断离线评测是否被数据泄漏污染？

可以这样回答：

> 我会先从数据层面检查 train/valid/test 的 exact duplicate 和 near duplicate，包括 input 级别和 input-output 级别；然后检查 input 中是否包含 label 或解析信息；再看实验日志，确认 test set 是否被用于 prompt tuning、超参选择或 checkpoint selection。对于大模型 benchmark，还会做题目改写、数值替换和私有样例对照，看模型是否只是记住原题。

#### 问法 2：发现测试集和训练集有重复怎么办？

可以这样回答：

> 如果重复比例很低，我会删除训练集中与测试集重复的样例，并重新训练或至少重新评估风险；如果重复比例较高，说明测试集独立性已经被破坏，需要重建测试集。修复后还要重新跑评测，并把泄漏检测纳入数据发布流程。

#### 问法 3：如何避免团队把测试集调成验证集？

可以这样回答：

> 需要制度和工具一起做。工具上记录每次评测的数据版本和用途，限制 test set 的访问频率；流程上把 dev set 用于日常调参，test set 只用于里程碑评估，hidden test set 由独立人员维护。长期迭代中，旧测试集可以降级成回归集，但不能继续作为最终泛化指标。

#### 问法 4：公开 benchmark 污染了怎么办？

可以这样回答：

> 公开 benchmark 可以作为参考，但不能作为唯一结论。需要补充私有 benchmark、时间切分样例、改写题、对抗样例和业务真实样例。如果模型在原题上很高、在改写题或新题上明显下降，就要怀疑污染或记忆。

---

### 20. 工程坑

#### 坑 1：只检查完全重复

完全重复只能发现最明显的问题。

大模型任务中，大量泄漏是改写级、语义级、模板级的。

#### 坑 2：只查 input，不查 output

有些生成任务中，input 不重复，但 input-output 对重复或答案模板重复。

#### 坑 3：近似重复阈值一刀切

短文本、长文本、代码、数学题、对话数据的相似度分布完全不同。阈值必须结合任务和抽样结果调整。

#### 坑 4：泄漏检测只做一次

数据会持续迭代。

只在项目初期做一次检测不够，应该在每次数据发布、训练集更新和 benchmark 更新时自动检查。

#### 坑 5：把测试集当作日常优化目标

这是最常见的调参泄漏。

只要团队反复根据测试集结果改 prompt、改数据、改模型，测试集就逐渐失去独立性。

---

### 21. 小练习

#### 练习 1

构造 5 条 train 样例和 5 条 test 样例，其中包含 2 条完全重复，运行 exact overlap 检测。

#### 练习 2

构造 3 组近似重复样例，调整 Jaccard 阈值，观察检测结果变化。

#### 练习 3

写一个规则，检测选择题 input 中是否包含“正确答案是 X”。

#### 练习 4

设计一份泄漏检测报告字段，包括数据版本、检测规则版本、统计摘要和高风险样例。

#### 练习 5

选一个公开 benchmark，随机找 5 道题，搜索题目文本是否在互联网上出现，并分析污染风险。


---

### 本讲总结

这一讲实现了数据泄漏检测的核心流程。

核心结论如下：

1. 数据泄漏会让离线评测虚高，是模型评估中必须排查的问题。
2. 常见泄漏包括完全重复、近似重复、答案泄漏、模板泄漏、时间泄漏和调参泄漏。
3. exact duplicate 可以用归一化文本和哈希索引检测。
4. near duplicate 可以用 n-gram Jaccard、MinHash、SimHash 或 embedding 近邻检测。
5. 标签泄漏要检查 input 中是否直接或间接包含 output。
6. 调参泄漏需要依赖实验日志和数据集使用规范来控制。
7. 公开 benchmark 可能被预训练语料污染，需要用私有集、改写题和时间切分样例做补充验证。
8. 泄漏检测应成为数据发布和模型评估流水线的一部分，而不是一次性人工检查。

下一讲，我们分析 hallucination 样例。

## 第 46 讲：分析 hallucination 样例

### 本讲目标

这一讲，我们做一件非常贴近大模型算法岗工作的事情：分析 hallucination 样例。

学完本讲，你应该能够回答：

1. hallucination 到底是什么，和普通错误有什么区别？
2. 为什么大模型会产生 hallucination？
3. 如何把 hallucination 样例结构化标注和归因？
4. 如何用代码统计不同类型的幻觉错误？
5. 面试中如何讲清楚 hallucination 的定位、评估和缓解方案？

hallucination 通常被翻译成“幻觉”。在大模型场景中，它指的是：模型生成了看起来流畅、自信、合理，但与事实、上下文、工具结果或用户约束不一致的内容。

注意，hallucination 不是所有错误的统称。

例如：

```text
用户：2 + 3 等于多少？
模型：6
```

这是错误，但不一定是典型 hallucination，更像是基础计算错误。

再看一个例子：

```text
用户：请根据下面材料回答：材料中没有提到作者出生地。
模型：作者出生于浙江杭州。
```

这就是典型 hallucination：模型编造了材料中不存在的信息。

大模型算法工程师不能只说“加 RAG 可以减少幻觉”。真正面试时，面试官更希望听到：你能否把幻觉拆成类型，能否定位原因，能否设计评测，能否给出工程闭环。

---

### 1. hallucination 的几种定义

不同论文和团队对 hallucination 的定义不完全一致，但工程上可以按“参考来源”来定义。

#### 定义 1：事实不一致

模型输出和世界事实不一致。

```text
模型：爱因斯坦获得过诺贝尔文学奖。
事实：爱因斯坦获得的是诺贝尔物理学奖。
```

#### 定义 2：上下文不一致

模型输出和输入上下文不一致。

```text
材料：公司成立于 2018 年。
模型：公司成立于 2020 年。
```

#### 定义 3：工具不一致

模型输出和检索、数据库、计算器、代码执行结果不一致。

```text
工具返回：库存为 0。
模型回答：该商品仍有库存，可以下单。
```

#### 定义 4：约束不一致

模型违反用户明确要求。

```text
用户：只用一句话回答。
模型：输出了五段解释。
```

严格来说，约束不一致不一定都是 hallucination，但在大模型质量分析中，经常和幻觉一起归入“生成不可信”问题。

---

### 2. 为什么模型会 hallucinate

幻觉不是单一原因造成的。

常见原因包括：

1. 语言模型目标是预测下一个 token，而不是天然保证事实正确。
2. 训练数据中存在错误、过期信息和互相矛盾的信息。
3. 模型参数记忆不完整，遇到不确定问题时仍倾向生成流畅答案。
4. 解码策略鼓励多样性时，可能牺牲事实性。
5. prompt 没有要求“不知道就说不知道”。
6. RAG 检索召回了错误文档、缺失文档或冲突文档。
7. 长上下文中关键信息被稀释或位置靠前导致注意力不足。
8. 训练中的 RLHF 可能奖励“看起来有帮助”的回答，而不是“诚实承认不知道”。

面试中可以这样概括：

> hallucination 的根因是生成目标和事实约束之间不完全一致。模型会优先生成高概率、流畅、符合模式的文本，但这些文本不一定被外部事实、输入证据或工具结果约束住。

---

### 3. 幻觉样例分析的目标

分析 hallucination 样例，不是为了给每条错例写一段感想，而是为了形成可执行的改进方案。

一个合格的分析流程应该回答四个问题：

1. 错在哪里？
2. 属于哪类 hallucination？
3. 根因更可能在数据、模型、prompt、检索、工具还是解码？
4. 应该用什么修复策略，修复后如何验证？

如果只说“模型胡说了”，这没有工程价值。

如果能说“这是 context-conflict hallucination，证据在第 2 段，模型输出把 2018 错写成 2020，可能是长上下文定位失败；建议增强引用约束、加入 evidence span 监督，并在评测中加入 source attribution 指标”，这就有工程价值。

---

### 4. 样例标注 schema

先设计一个 hallucination 分析表。

每条样例可以标注这些字段：

```json
{
  "id": "case_001",
  "query": "用户问题",
  "context": "输入上下文或检索结果",
  "model_answer": "模型回答",
  "reference_answer": "参考答案，可为空",
  "hallucination": true,
  "hallucination_type": "context_conflict",
  "severity": "high",
  "evidence": "上下文第 2 段说明公司成立于 2018 年",
  "root_cause": "context_grounding_failure",
  "fix_suggestion": "要求回答引用证据，并增加无法从材料推出时拒答"
}
```

字段解释：

1. `hallucination`：是否存在幻觉。
2. `hallucination_type`：幻觉类型。
3. `severity`：严重程度。
4. `evidence`：判断为幻觉的依据。
5. `root_cause`：初步根因。
6. `fix_suggestion`：可执行改进动作。

工程上最重要的是 `evidence`。

如果没有 evidence，标注就很难复核，也无法说服团队接受结论。

---

### 5. hallucination 类型体系

可以先用一个实用分类。

```python
HALLUCINATION_TYPES = {
    "factual_error": "与通用事实不一致",
    "context_conflict": "与给定上下文冲突",
    "unsupported_claim": "上下文没有支持但模型强行断言",
    "fabricated_citation": "编造引用、论文、链接或来源",
    "fabricated_entity": "编造人名、机构、产品、字段等实体",
    "tool_conflict": "与工具、数据库或检索结果冲突",
    "instruction_conflict": "违反用户明确约束",
    "over_generalization": "从局部证据过度推广",
    "stale_knowledge": "使用过期知识",
}
```

这套分类不是唯一标准，但足够覆盖大多数业务场景。

不同任务可以再细化。

例如医疗问答中可以细分为：诊断编造、药物剂量错误、禁忌症遗漏、指南过期。

金融场景可以细分为：财报数字错误、政策时间错误、风险承诺过度、投资建议违规。

---

### 6. 严重程度分级

不是所有 hallucination 都同样严重。

可以使用三级标准：

```python
SEVERITY_LEVELS = {
    "low": "轻微不准确，不影响主要结论",
    "medium": "影响部分结论或用户理解",
    "high": "导致错误决策、安全风险、合规风险或核心事实错误",
}
```

示例：

```text
low: 模型把某篇论文的发表月份写错，但主要方法描述正确。
medium: 模型错误总结了合同中的一个非核心条款。
high: 模型编造药物剂量、法律结论或财务数字。
```

在业务评估中，高严重度 hallucination 应该单独统计，因为它比普通准确率更能反映上线风险。

---

### 7. 构造示例数据

下面构造一组待分析样例。

```python
examples = [
    {
        "id": "case_001",
        "query": "根据材料回答，公司是哪一年成立的？",
        "context": "材料：星河智能成立于 2018 年，主要从事工业机器人研发。",
        "model_answer": "星河智能成立于 2020 年。",
        "reference_answer": "星河智能成立于 2018 年。",
    },
    {
        "id": "case_002",
        "query": "这篇材料提到了 CEO 的毕业院校吗？",
        "context": "材料：公司 CEO 王明曾在多家科技企业任职，负责产品和战略。",
        "model_answer": "提到了，王明毕业于清华大学。",
        "reference_answer": "没有提到 CEO 的毕业院校。",
    },
    {
        "id": "case_003",
        "query": "请给出支持这个结论的论文引用。",
        "context": "材料：已有实验表明该方法在内部数据集上有效，但未列出论文引用。",
        "model_answer": "可以参考 Smith et al., 2021, Attention Improves Everything。",
        "reference_answer": "材料未提供论文引用。",
    },
]
```

这些样例分别对应：

1. 与上下文冲突。
2. 无依据断言。
3. 编造引用。

---

### 8. 人工标注函数

真实项目中，幻觉标注通常需要人工判断。

我们先写一个辅助函数，把人工标注结构化。

```python
def make_annotation(
    example,
    hallucination,
    hallucination_type,
    severity,
    evidence,
    root_cause,
    fix_suggestion,
):
    return {
        "id": example["id"],
        "query": example["query"],
        "context": example.get("context", ""),
        "model_answer": example["model_answer"],
        "reference_answer": example.get("reference_answer", ""),
        "hallucination": hallucination,
        "hallucination_type": hallucination_type,
        "severity": severity,
        "evidence": evidence,
        "root_cause": root_cause,
        "fix_suggestion": fix_suggestion,
    }
```

人工标注示例：

```python
annotations = [
    make_annotation(
        examples[0],
        hallucination=True,
        hallucination_type="context_conflict",
        severity="high",
        evidence="context 中明确写到成立于 2018 年，模型回答为 2020 年",
        root_cause="context_grounding_failure",
        fix_suggestion="要求模型回答前定位证据句，并在答案中引用原文年份",
    ),
    make_annotation(
        examples[1],
        hallucination=True,
        hallucination_type="unsupported_claim",
        severity="medium",
        evidence="context 只提到任职经历，没有提到毕业院校",
        root_cause="over_answering",
        fix_suggestion="prompt 中加入材料未提及时回答无法判断，并加入拒答样例",
    ),
    make_annotation(
        examples[2],
        hallucination=True,
        hallucination_type="fabricated_citation",
        severity="high",
        evidence="context 明确没有列出论文引用，模型编造了论文标题和作者",
        root_cause="citation_fabrication",
        fix_suggestion="禁止生成未提供来源的 citation，引用必须来自检索结果",
    ),
]
```

---

### 9. 统计幻觉类型分布

标注后，要统计问题集中在哪些类型上。

```python
from collections import Counter, defaultdict


def summarize_annotations(annotations):
    total = len(annotations)
    hallucinated = [x for x in annotations if x["hallucination"]]

    type_counter = Counter(x["hallucination_type"] for x in hallucinated)
    severity_counter = Counter(x["severity"] for x in hallucinated)
    root_cause_counter = Counter(x["root_cause"] for x in hallucinated)

    return {
        "total": total,
        "hallucination_count": len(hallucinated),
        "hallucination_rate": len(hallucinated) / total if total else 0.0,
        "type_distribution": dict(type_counter),
        "severity_distribution": dict(severity_counter),
        "root_cause_distribution": dict(root_cause_counter),
    }


summary = summarize_annotations(annotations)
print(summary)
```

输出可能是：

```python
{
    "total": 3,
    "hallucination_count": 3,
    "hallucination_rate": 1.0,
    "type_distribution": {
        "context_conflict": 1,
        "unsupported_claim": 1,
        "fabricated_citation": 1,
    },
    "severity_distribution": {
        "high": 2,
        "medium": 1,
    },
    "root_cause_distribution": {
        "context_grounding_failure": 1,
        "over_answering": 1,
        "citation_fabrication": 1,
    },
}
```

这比“模型有幻觉”更有用，因为它告诉我们优先修什么。

如果 `fabricated_citation` 很多，优先修引用生成约束。

如果 `context_conflict` 很多，优先修上下文 grounding。

如果 `unsupported_claim` 很多，优先修拒答和不确定性表达。

---

### 10. 按任务维度聚合

同一个模型在不同任务上的 hallucination 形态可能不同。

可以给样例增加任务字段。

```python
for ann in annotations:
    ann["task"] = "doc_qa"


def group_by_field(annotations, field):
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann.get(field, "unknown")].append(ann)
    return grouped


def summarize_by_field(annotations, field):
    grouped = group_by_field(annotations, field)
    return {name: summarize_annotations(items) for name, items in grouped.items()}


task_summary = summarize_by_field(annotations, "task")
print(task_summary)
```

实际项目中，建议至少按以下维度聚合：

1. 任务类型：问答、摘要、代码、Agent、RAG、分类。
2. 数据来源：线上日志、离线 benchmark、人工构造集。
3. 输入长度：短上下文、中等上下文、长上下文。
4. 检索状态：有证据、无证据、证据冲突、证据缺失。
5. 模型版本：base、SFT、RLHF、RAG、工具增强版本。

这样才能定位问题是全局能力问题，还是某个场景的局部问题。

---

### 11. 自动规则辅助检测

幻觉最终通常需要人工确认，但可以用规则做初筛。

#### 规则 1：材料未提到却强行回答

```python
UNKNOWN_PATTERNS = [
    "无法判断",
    "未提到",
    "材料没有说明",
    "不知道",
    "无法从材料中得出",
]


def contains_unknown_answer(text):
    text = str(text)
    return any(pattern in text for pattern in UNKNOWN_PATTERNS)


def flag_should_refuse_but_answered(example):
    reference = example.get("reference_answer", "")
    answer = example.get("model_answer", "")
    return contains_unknown_answer(reference) and not contains_unknown_answer(answer)
```

这个规则可以发现：参考答案认为材料不足，但模型仍然给了确定答案。

#### 规则 2：数字和上下文不一致

```python
import re


def extract_numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", str(text))


def flag_number_conflict(example):
    context_numbers = set(extract_numbers(example.get("context", "")))
    answer_numbers = set(extract_numbers(example.get("model_answer", "")))
    reference_numbers = set(extract_numbers(example.get("reference_answer", "")))

    if not answer_numbers:
        return False
    if reference_numbers and answer_numbers != reference_numbers:
        return True
    if context_numbers and not answer_numbers.issubset(context_numbers):
        return True
    return False
```

数字错误在金融、合同、医疗、指标报告中非常关键，可以单独统计。

#### 规则 3：疑似编造引用

```python
def flag_fabricated_citation(example):
    answer = example.get("model_answer", "")
    context = example.get("context", "")

    citation_markers = ["et al.", "doi", "arxiv", "http", "www."]
    has_citation = any(marker.lower() in answer.lower() for marker in citation_markers)

    if not has_citation:
        return False

    return answer not in context
```

这个规则很粗，只适合初筛。

真实引用检测应检查 citation 是否来自检索结果、是否存在、是否和结论匹配。

---

### 12. 自动生成候选风险标签

把规则组合起来。

```python
def auto_flag_example(example):
    flags = []

    if flag_should_refuse_but_answered(example):
        flags.append("should_refuse_but_answered")
    if flag_number_conflict(example):
        flags.append("number_conflict")
    if flag_fabricated_citation(example):
        flags.append("possible_fabricated_citation")

    return flags


for ex in examples:
    print(ex["id"], auto_flag_example(ex))
```

规则检测的作用不是替代人工，而是提高人工分析效率。

比较合理的流程是：

1. 自动规则扫描全量样例。
2. 高风险样例优先进入人工标注。
3. 人工确认 hallucination 类型和严重程度。
4. 统计分布并制定修复策略。
5. 修复后用同一套规则和人工集回归验证。

---

### 13. RAG 场景下的 hallucination 分析

RAG 系统中的 hallucination 要拆成两层：检索层和生成层。

一个 RAG 样例至少包含：

```json
{
  "query": "用户问题",
  "retrieved_docs": ["文档1", "文档2"],
  "answer": "模型答案",
  "reference": "参考答案"
}
```

分析时先问：证据有没有被检索出来？

如果没有，根因更可能是 retrieval failure。

如果有证据但模型没用对，根因更可能是 generation grounding failure。

```python
def rag_error_type(has_relevant_doc, answer_supported_by_doc):
    if not has_relevant_doc:
        return "retrieval_failure"
    if has_relevant_doc and not answer_supported_by_doc:
        return "generation_grounding_failure"
    return "no_rag_hallucination"
```

RAG 幻觉分析常见分类：

1. 检索缺失：没有召回相关文档。
2. 检索噪声：召回了大量无关文档，干扰生成。
3. 证据冲突：多个文档互相矛盾，模型选择错误。
4. 证据存在但忽略：模型没有使用正确证据。
5. 证据不足却强答：应该拒答但生成确定答案。
6. 引用错配：答案内容来自 A 文档，却引用 B 文档。

面试时一定要强调：RAG 并不自动消除 hallucination。检索质量、证据排序、上下文压缩、引用约束和拒答机制都会影响最终事实性。

---

### 14. 长上下文场景下的 hallucination

长上下文不是把所有材料塞进去就安全。

模型可能出现：

1. 忽略中间位置的关键信息。
2. 被开头或结尾的无关信息吸引。
3. 混淆不同实体的属性。
4. 把多个段落的信息错误拼接。
5. 对没有证据的问题强行回答。

长上下文样例分析建议记录：

```json
{
  "evidence_position": "middle",
  "context_length_tokens": 18000,
  "num_entities": 24,
  "has_conflicting_evidence": false,
  "answer_uses_correct_span": false
}
```

如果大量错误集中在 `evidence_position=middle`，可能是 lost-in-the-middle 问题。

修复方向包括：

1. 对证据段落重排序，把高相关内容放在更靠近回答的位置。
2. 使用 query-focused compression 压缩上下文。
3. 要求模型先抽取 evidence span，再生成答案。
4. 对长上下文任务单独做 SFT 或偏好优化。

---

### 15. Agent 场景下的 hallucination

Agent 的幻觉不仅体现在最终回答，也体现在行动过程。

例如：

```text
模型声称已经调用接口，但实际上没有调用。
模型声称文件已创建，但工作区没有该文件。
模型根据不存在的工具返回值继续推理。
```

Agent 幻觉分析需要看完整 trace：

1. 用户指令。
2. 模型思考或计划。
3. 工具调用参数。
4. 工具返回结果。
5. 最终回答。

常见类型：

1. Tool-use hallucination：编造工具能力或工具结果。
2. State hallucination：误以为某个状态已经改变。
3. File hallucination：声称读写了不存在的文件。
4. Planning hallucination：计划和实际执行不一致。
5. Result hallucination：最终总结与工具结果不一致。

Agent 场景的修复重点是：让最终回答只能基于可审计的工具结果，而不是基于模型自我声称。

---

### 16. Prompt 层面的缓解

prompt 不能根治幻觉，但能显著降低一些场景的风险。

常见策略：

```text
只根据给定材料回答。
如果材料不足以回答，请说“材料中没有提供相关信息”。
回答中每个关键结论都必须引用原文证据。
不要编造论文、链接、数字、人名或机构名。
如果检索结果之间存在冲突，请指出冲突，而不是自行选择。
```

一个 RAG 问答 prompt 示例：

```text
你是一个严谨的问答助手。

要求：
1. 只能使用 <context> 中的信息回答。
2. 如果 <context> 不包含答案，回答“材料中没有提供相关信息”。
3. 每个关键结论后标注对应证据编号。
4. 不要使用你自己的背景知识补全缺失信息。

<context>
[1] ...
[2] ...
</context>

问题：{query}
```

prompt 的优点是成本低、上线快。

缺点是对复杂幻觉、知识冲突和长上下文问题不稳定，不能替代数据和系统层面的改进。

---

### 17. 数据和训练层面的缓解

如果 hallucination 是系统性问题，需要从数据和训练入手。

常见方法：

1. 加入拒答样例：让模型学会在证据不足时说不知道。
2. 加入基于证据回答样例：要求模型先引用证据再回答。
3. 加入负样例：给出无关上下文，目标答案是无法判断。
4. 加入偏好数据：偏好“保守、可证据支持”的回答，惩罚编造。
5. 做事实性 reward model：对答案是否被证据支持进行打分。
6. 对高风险领域做专家标注和专项评测。

一个简单的 SFT 样例格式：

```json
{
  "instruction": "根据材料回答问题。若材料不足，请回答无法判断。",
  "input": "材料：公司 CEO 王明曾在多家科技企业任职。问题：王明毕业于哪所大学？",
  "output": "无法判断，材料中没有提供王明的毕业院校信息。"
}
```

这类样例对降低 unsupported claim 很有帮助。

---

### 18. 系统层面的缓解

对于上线系统，通常不能只依赖模型自己变诚实。

还需要系统约束。

常见方案：

1. RAG：用外部知识库提供证据。
2. 引用校验：答案中的引用必须来自检索文档。
3. 数字校验：金额、日期、比例等字段用规则或工具核对。
4. 工具调用：计算、查询、库存、订单状态交给工具。
5. 二次 verifier：用另一个模型或规则检查答案是否被证据支持。
6. 置信度和拒答：低置信度时不强答，转人工或请求澄清。
7. 高风险领域加人工审核：医疗、法律、金融、招聘等。

工程上常见的闭环是：

```text
用户问题 -> 检索/工具 -> 生成答案 -> 证据校验 -> 风险分级 -> 输出或拒答/转人工
```

这比单纯调 prompt 更可靠。

---

### 19. 如何评估修复是否有效

修复 hallucination 后不能只看总体准确率。

需要看更细的指标：

1. hallucination rate：幻觉样例占比。
2. high-severity hallucination rate：高严重度幻觉占比。
3. unsupported claim rate：无证据断言比例。
4. citation precision：引用是否真的支持结论。
5. refusal precision：拒答是否合理。
6. refusal recall：该拒答的场景是否拒答。
7. answer usefulness：减少幻觉是否导致过度保守。

这里有一个重要 trade-off：减少幻觉往往会提高拒答率。

如果模型变成“什么都说不知道”，hallucination 降低了，但用户体验变差。

所以要同时评估：事实性、覆盖率、可用性。

---

### 20. 保存分析报告

把标注和统计结果保存下来。

```python
import json
from pathlib import Path


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


save_json("reports/hallucination_annotations.json", annotations)
save_json("reports/hallucination_summary.json", summary)
```

一份完整报告建议包含：

1. 模型版本。
2. 数据集版本。
3. 采样方式。
4. 标注人员和一致性检查。
5. 幻觉类型分布。
6. 严重程度分布。
7. 根因分布。
8. 典型 case。
9. 修复方案。
10. 修复前后对比。

---

### 21. 面试高频问法

#### 问法 1：什么是 hallucination？

可以这样回答：

> hallucination 是指模型生成了流畅、自信但不被事实、上下文、工具结果或用户约束支持的内容。它不等于所有错误，典型特征是模型编造或错误断言了不存在、不正确、无依据的信息。

#### 问法 2：大模型为什么会有幻觉？

可以这样回答：

> 根本原因是语言模型的训练目标是预测下一个 token，而不是天然保证事实正确。训练数据可能有噪声和过期信息，模型遇到不确定问题时仍倾向生成看起来有帮助的答案；同时解码、prompt、RAG 检索失败、长上下文定位失败和 RLHF 偏好都可能放大幻觉。

#### 问法 3：如何分析一批 hallucination badcase？

可以这样回答：

> 我会先建立标注 schema，包括幻觉类型、严重程度、证据、根因和修复建议；然后按任务、输入长度、检索状态、模型版本聚合统计。对于 RAG 场景，会区分 retrieval failure 和 generation grounding failure；对于长上下文，会看证据位置和实体混淆。最后根据分布决定是改检索、改 prompt、补拒答数据、加 verifier，还是做专项训练。

#### 问法 4：如何降低 RAG 系统的 hallucination？

可以这样回答：

> 先保证检索能召回正确证据，再约束生成只能基于证据回答。具体包括提升召回和 rerank、做上下文压缩、要求答案引用证据、无证据时拒答、校验引用是否支持结论，并对数字、日期、实体等高风险字段做工具或规则校验。

#### 问法 5：怎么评估幻觉修复效果？

可以这样回答：

> 不能只看总体准确率，要看 hallucination rate、高严重度幻觉率、unsupported claim rate、citation precision、合理拒答率和用户可用性。因为降低幻觉可能带来过度拒答，所以要同时评估事实性和回答覆盖率。

---

### 22. 工程坑

#### 坑 1：把所有错误都叫 hallucination

这会导致分析失焦。

计算错误、格式错误、指令跟随错误和事实编造应该分开统计。

#### 坑 2：只看 hallucination 总率

总率可能掩盖高风险问题。

医疗剂量编造一次，严重性可能远高于普通描述不准确十次。

#### 坑 3：只靠 prompt 解决

prompt 可以缓解，但无法保证事实性。

上线系统需要检索、工具、校验、拒答和人工审核等机制。

#### 坑 4：RAG 后不再评估幻觉

RAG 可能引入新的幻觉，例如引用错配、证据冲突、检索噪声和基于错误文档回答。

#### 坑 5：忽略过度拒答

如果模型为了避免幻觉而大量回答“不知道”，业务可用性会下降。

所以要同时看拒答是否合理。

#### 坑 6：没有 evidence 字段

没有证据的幻觉标注很难复核，也很难用于训练或评估。

---

### 23. 小练习

#### 练习 1

构造 10 条问答样例，人工标注其中哪些是 `context_conflict`，哪些是 `unsupported_claim`。

#### 练习 2

写一个规则，检测模型答案中的数字是否和参考答案数字不一致。

#### 练习 3

设计一个 RAG hallucination 分析表，字段至少包括 query、retrieved docs、answer、evidence、error type。

#### 练习 4

找 5 条模型编造引用的样例，分析是 prompt 问题、检索问题还是模型习惯性补全问题。

#### 练习 5

设计一组指标，同时衡量 hallucination rate 和 over-refusal rate。


---

### 本讲总结

这一讲，我们完成了 hallucination 样例分析的完整流程。

核心结论如下：

1. hallucination 是模型生成不被事实、上下文、工具结果或用户约束支持的信息。
2. 它不等于所有错误，应该和计算错误、格式错误、指令跟随错误分开分析。
3. 幻觉可以按事实错误、上下文冲突、无依据断言、编造引用、工具冲突等类型拆分。
4. 分析样例时要记录类型、严重程度、证据、根因和修复建议。
5. RAG 场景要区分检索失败和生成 grounding 失败。
6. 长上下文场景要关注证据位置、实体混淆和信息拼接错误。
7. 缓解方案包括 prompt 约束、拒答数据、证据监督、RAG、工具校验、verifier 和人工审核。
8. 修复效果要同时看 hallucination rate、高严重度幻觉率、引用准确率、合理拒答率和可用性。

下一讲，我们整理训练不收敛 debug 清单。

## 第 47 讲：训练不收敛 debug 清单

### 本讲目标

这一讲，我们整理一份大模型训练不收敛的 debug 清单。

学完本讲，你应该能够回答：

1. 什么叫训练不收敛，常见表现有哪些？
2. loss 不下降、loss 爆炸、loss 为 NaN 分别应该怎么排查？
3. 如何判断问题来自数据、模型、优化器、学习率、精度还是分布式训练？
4. LoRA/SFT/预训练场景的不收敛排查有什么区别？
5. 面试中如何系统讲清楚训练 debug 方法论？

训练不收敛不是一个具体错误，而是一类现象。

它可能来自数据，也可能来自代码、超参、模型结构、精度、并行策略、loss mask、label 构造、优化器状态、随机种子或硬件通信。

很多候选人面试时会直接说“调小学习率”。这不是错，但太单薄。

优秀回答应该是：先定义现象，再缩小范围，再做最小可复现实验，最后逐项排查。

---

### 1. 不收敛的典型表现

训练不收敛常见有几种表现。

#### 表现 1：loss 完全不下降

```text
step 0: loss = 8.91
step 100: loss = 8.90
step 1000: loss = 8.92
```

这说明模型没有有效学习。

常见原因：学习率太小、参数没有更新、loss mask 错误、标签全被 ignore、数据输入恒定、梯度为 0。

#### 表现 2：loss 下降很慢

```text
step 0: loss = 8.91
step 1000: loss = 8.70
step 10000: loss = 8.55
```

这可能是学习率偏小、batch 太小、数据太难、模型容量不够，也可能是正常现象，需要和 baseline 对比。

#### 表现 3：loss 先降后升

```text
step 0: loss = 8.9
step 500: loss = 5.2
step 1000: loss = 12.7
```

常见原因：学习率过大、warmup 太短、梯度爆炸、数据分布突然变化、混合精度溢出。

#### 表现 4：loss 直接 NaN 或 Inf

```text
step 20: loss = nan
```

常见原因：学习率过大、数值溢出、除零、log(0)、attention mask 错误、fp16 不稳定、梯度爆炸。

#### 表现 5：训练 loss 降低，但验证指标不涨

这不一定是训练不收敛，更可能是泛化问题、数据泄漏修复后指标回落、训练目标和评测目标不一致、过拟合或评测脚本错误。

---

### 2. debug 总原则

训练 debug 的第一原则：不要一上来改很多东西。

正确流程是：

1. 先确认现象是否真实。
2. 固定随机种子和数据版本。
3. 缩小到最小可复现实验。
4. 先跑小 batch、小数据、小模型。
5. 检查一批数据、一轮 forward、一轮 backward、一次 optimizer step。
6. 每次只改一个变量。
7. 保留实验日志，避免靠记忆判断。

一个非常实用的口诀：

```text
先数据，再 loss；先单卡，再多卡；先 fp32，再混精；先小样本过拟合，再全量训练。
```

如果一个模型连 10 条样例都无法过拟合，通常不是模型能力问题，而是训练链路有 bug。

---

### 3. 最小过拟合测试

最小过拟合测试是训练 debug 的核心工具。

做法：取 8 到 32 条训练样例，反复训练，观察 loss 是否能快速降到很低。

```python
def tiny_subset(dataset, n=16):
    return [dataset[i] for i in range(min(n, len(dataset)))]
```

预期现象：

```text
step 0: loss 较高
step 50: loss 明显下降
step 200: loss 接近很低
```

如果 tiny set 都无法过拟合，优先怀疑：

1. label 构造错误。
2. loss mask 错误。
3. 参数被冻结。
4. optimizer 没有接到可训练参数。
5. 学习率极小或极大。
6. 输入输出错位。
7. attention mask 或 position ids 错误。

面试中可以说：

> 我会先用极小数据集做 overfit sanity check。如果小样本无法过拟合，说明训练链路有问题；如果小样本能过拟合但全量不收敛，再看数据分布、学习率、batch size、优化器和训练稳定性。

---

### 4. 检查 batch 内容

很多不收敛问题来自数据 batch。

训练前先打印一批样例。

```python
def inspect_batch(batch, tokenizer=None, max_items=2):
    for key, value in batch.items():
        try:
            print(key, value.shape, value.dtype)
        except AttributeError:
            print(key, type(value))

    if tokenizer is not None and "input_ids" in batch:
        for i in range(min(max_items, batch["input_ids"].shape[0])):
            ids = batch["input_ids"][i].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=False)
            print("--- decoded input ---")
            print(text[:1000])

    if "labels" in batch:
        labels = batch["labels"]
        print("labels min/max:", labels.min().item(), labels.max().item())
        print("ignore ratio:", (labels == -100).float().mean().item())
```

重点看：

1. `input_ids` 是否全是 pad。
2. `labels` 是否几乎全是 `-100`。
3. 输入和输出是否拼接正确。
4. attention mask 是否和 padding 对齐。
5. 序列长度是否异常截断。
6. dtype 是否合理。

SFT 中最常见的问题是：labels 全部被 mask 成 `-100`，导致有效 loss token 很少甚至为 0。

---

### 5. 检查 label shift

自回归语言模型训练时，通常用当前位置预测下一个 token。

如果手写 loss，很容易把 shift 写错。

典型写法：

```python
import torch.nn.functional as F


def causal_lm_loss(logits, labels):
    # logits: [batch, seq, vocab]
    # labels: [batch, seq]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
```

常见错误：

1. 没有 shift，导致模型预测当前 token。
2. shift 方向反了。
3. labels 和 input_ids 长度不一致。
4. padding token 没有设置为 `-100`。
5. prompt 部分没有 mask，导致模型学习复述 prompt。

如果使用 Hugging Face 的 `AutoModelForCausalLM` 并传入 `labels`，模型内部通常会处理 shift，不要重复 shift 一次。

---

### 6. 检查可训练参数

参数没有更新是 loss 不下降的高频原因。

```python
def inspect_trainable_params(model):
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
            print("trainable:", name, tuple(param.shape))
    print(f"trainable params: {trainable} / {total} ({trainable / total:.4%})")
```

如果是全参微调，trainable ratio 应接近 100%。

如果是 LoRA 微调，trainable ratio 很小是正常的，但必须确认 LoRA 参数确实可训练。

常见 bug：

1. 冻结了全部参数。
2. LoRA adapter 没有注入成功。
3. optimizer 在冻结前创建，参数组不对。
4. 只训练了 embedding 或 lm_head，不符合预期。
5. `model.eval()` 后忘记切回 `model.train()`。

---

### 7. 检查梯度是否存在

一次 backward 后检查梯度。

```python
def inspect_gradients(model, max_items=20):
    shown = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            print(name, "grad=None")
        else:
            grad = param.grad.detach()
            print(
                name,
                "grad_norm=", grad.norm().item(),
                "grad_mean=", grad.mean().item(),
                "has_nan=", torch.isnan(grad).any().item(),
            )
        shown += 1
        if shown >= max_items:
            break
```

需要关注：

1. 梯度是否全是 None。
2. 梯度是否全 0。
3. 梯度是否 NaN 或 Inf。
4. 梯度 norm 是否极大。
5. 只有部分模块有梯度是否符合预期。

如果梯度为 None，可能是计算图断了、参数没参与 forward、loss 没依赖参数、用了 `.detach()` 或错误地把 tensor 转成了 Python 标量。

---

### 8. 检查 optimizer step 是否真的更新参数

有梯度不代表参数更新了。

可以在一步优化前后比较参数。

```python
def clone_trainable_params(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compare_param_update(model, before, atol=0.0):
    updated = []
    unchanged = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        old = before[name].to(param.device)
        diff = (param.detach() - old).abs().max().item()
        if diff > atol:
            updated.append((name, diff))
        else:
            unchanged.append(name)
    return updated, unchanged
```

排查点：

1. 是否调用了 `loss.backward()`。
2. 是否调用了 `optimizer.step()`。
3. 是否在 step 前错误地 `zero_grad()`。
4. 学习率是否为 0。
5. gradient clipping 是否把梯度裁成了 0。
6. AMP 的 scaler 是否一直跳过 step。

---

### 9. 学习率排查

学习率是最常见但不是唯一原因。

学习率过大：

```text
loss 剧烈震荡、突然爆炸、NaN
```

学习率过小：

```text
loss 长时间几乎不变
```

检查当前学习率：

```python
def print_lr(optimizer):
    for i, group in enumerate(optimizer.param_groups):
        print(i, group["lr"])
```

建议做一个小范围学习率扫描：

```text
1e-6, 3e-6, 1e-5, 3e-5, 1e-4
```

对于大模型 SFT，常见学习率范围大致是：

1. 全参微调：`1e-6` 到 `2e-5`。
2. LoRA 微调：`1e-5` 到 `2e-4`。
3. 从头预训练小模型：可能更大，但依赖 batch size 和 scheduler。

这些不是固定答案，只是排查起点。

面试中不要说“学习率设成 1e-4 就行”，而要说“结合训练方式、batch size、模型规模和 scheduler 做 sweep”。

---

### 10. warmup 和 scheduler 排查

大模型训练通常需要 warmup。

warmup 太短，训练早期可能不稳定。

warmup 太长，loss 下降会很慢。

需要检查：

```python
def inspect_scheduler(optimizer, scheduler, steps=10):
    for step in range(steps):
        scheduler.step()
        lrs = [group["lr"] for group in optimizer.param_groups]
        print(step, lrs)
```

注意：实际训练中 `optimizer.step()` 和 `scheduler.step()` 的顺序要与框架约定一致。

常见问题：

1. 总训练步数算错，导致学习率很快衰减到 0。
2. warmup steps 大于总步数。
3. resume 训练后 scheduler state 没恢复。
4. gradient accumulation 下 step 数统计错误。

---

### 11. batch size 和梯度累积

有效 batch size 计算：

```text
effective_batch_size = per_device_batch_size * num_devices * gradient_accumulation_steps
```

batch 太小会导致 loss 抖动明显。

batch 太大如果学习率没有相应调整，也可能收敛慢或泛化差。

排查点：

1. gradient accumulation 是否真的生效。
2. 是否每个 micro-batch 都错误 step。
3. loss 是否按 accumulation steps 正确缩放。
4. 多卡下 global batch 是否和预期一致。

典型写法：

```python
loss = loss / gradient_accumulation_steps
loss.backward()

if (step + 1) % gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

如果忘记除以 accumulation steps，相当于放大了学习率，可能造成不稳定。

---

### 12. 混合精度排查

fp16 容易溢出，bf16 通常更稳定。

如果 loss NaN，建议先做两个实验：

1. 关掉混合精度，用 fp32 跑小 batch。
2. 如果硬件支持，把 fp16 改成 bf16。

AMP 训练要关注 GradScaler 是否频繁跳过 step。

```python
# 伪代码：不同框架接口略有差异
print("scale:", scaler.get_scale())
```

如果 scale 持续下降，说明经常发生溢出。

常见修复：

1. 降低学习率。
2. 开启 gradient clipping。
3. 使用 bf16。
4. 对 loss 或 logits 做数值稳定处理。
5. 检查 attention mask 是否产生全 `-inf` 行。

---

### 13. attention mask 和 NaN

attention mask 错误可能导致 softmax 全是 `-inf`，最后变成 NaN。

例如某个 query 位置没有任何可 attend 的 key。

检查 mask：

```python
def inspect_attention_mask(attention_mask):
    print("shape:", attention_mask.shape)
    print("dtype:", attention_mask.dtype)
    print("min/max:", attention_mask.min().item(), attention_mask.max().item())
    valid_per_row = attention_mask.sum(dim=-1)
    print("valid tokens min:", valid_per_row.min().item())
```

常见问题：

1. padding mask 语义反了。
2. causal mask 方向反了。
3. left padding 和 position ids 没处理好。
4. 全 padding 样例进入训练。
5. mask dtype 不符合算子要求。

如果使用 FlashAttention，还要特别关注输入长度、padding、causal 参数和 mask 格式是否符合实现要求。

---

### 14. 数据质量排查

如果小样本能过拟合，但全量训练不稳定，要重点看数据。

检查项：

1. 是否有空输入、空输出。
2. 是否有超长样例导致大量截断。
3. 是否有乱码、HTML、重复样例、低质量样例。
4. 是否 label 和 input 错配。
5. 是否多任务数据格式不统一。
6. 是否不同数据源 loss 差异极大。
7. 是否某些 batch 中有效 token 数非常少。

可以统计长度分布：

```python
def length_stats(rows, tokenizer):
    lengths = []
    for row in rows:
        text = row.get("input", "") + row.get("output", "")
        lengths.append(len(tokenizer.encode(text)))
    lengths.sort()
    n = len(lengths)
    return {
        "min": lengths[0],
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
    }
```

如果 p99 远大于 max length，大量样例会被截断，可能导致答案部分丢失。

---

### 15. tokenizer 排查

tokenizer 错误会导致训练看似正常但效果很差。

检查项：

1. tokenizer 是否和 base model 匹配。
2. special tokens 是否配置正确。
3. pad token 是否存在。
4. eos token 是否被正确加入。
5. chat template 是否和训练/推理一致。
6. 新增 token 后 embedding 是否 resize。

典型问题：训练用一种 chat template，推理用另一种 chat template，导致模型学到的格式和线上输入不一致。

可以打印 decode 后的完整训练文本，人工确认是否符合预期。

---

### 16. LoRA 微调排查

LoRA 不收敛有一些特有问题。

检查项：

1. target modules 是否选对。
2. LoRA 参数是否可训练。
3. rank `r` 是否太小。
4. `lora_alpha` 是否合理。
5. dropout 是否过大。
6. 是否只训练了不关键模块。
7. merge 和加载 adapter 是否正确。

常见 target modules：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

不同模型命名不一样，不能盲目复制。

如果 target module 名称匹配不到，LoRA 可能根本没有注入。

面试中可以说：

> LoRA 不收敛时，我会先打印 trainable parameters 和 module 名称，确认 adapter 注入到了预期层；再做 tiny overfit，确认 LoRA 参数有梯度且 optimizer step 后发生变化。

---

### 17. 分布式训练排查

单卡正常、多卡不正常，优先怀疑分布式链路。

检查项：

1. 不同 rank 的数据是否重复或为空。
2. sampler 是否调用 `set_epoch`。
3. loss 是否在不同 rank 上一致或合理。
4. gradient accumulation 和 world size 是否共同影响 batch。
5. ZeRO/FSDP 是否正确保存和恢复 optimizer state。
6. 梯度同步是否被错误关闭。
7. checkpoint 是否只保存了 shard，加载方式是否正确。

排查策略：

```text
先单卡跑通 -> 再 2 卡 -> 再扩到多机多卡
```

不要一开始就在复杂分布式环境里 debug 所有问题。

---

### 18. 评估脚本排查

有时训练其实收敛了，但评估脚本错了。

检查项：

1. 评估 prompt 是否和训练格式一致。
2. generation 参数是否合理。
3. 是否正确去掉 prompt 部分，只评估 answer。
4. 指标计算是否处理大小写、空格、标点。
5. 分类任务 label mapping 是否一致。
6. checkpoint 是否加载了正确版本。
7. LoRA adapter 是否在评估时加载或 merge。

如果训练 loss 下降明显但指标完全不变，要优先检查评估链路。

---

### 19. 一份可执行 debug 清单

遇到训练不收敛，可以按这个顺序排查：

1. 确认日志：loss 曲线、学习率曲线、grad norm、有效 token 数。
2. 固定环境：随机种子、数据版本、代码版本、checkpoint。
3. 检查一批 batch：decode 输入、labels、mask、长度、ignore ratio。
4. 跑 tiny overfit：16 条样例是否能快速过拟合。
5. 检查参数：可训练参数数量是否符合预期。
6. 检查梯度：是否为 None、0、NaN、Inf。
7. 检查更新：optimizer step 后参数是否变化。
8. 扫学习率：小范围比较 loss 曲线。
9. 检查 scheduler：warmup、总步数、resume state。
10. 检查精度：fp32/bf16/fp16 对比，是否 AMP 溢出。
11. 检查 mask：padding mask、causal mask、全 padding 样例。
12. 检查数据：空样例、错配、截断、重复、低质量数据。
13. 检查 tokenizer：special tokens、chat template、eos/pad。
14. 检查分布式：单卡和多卡对比。
15. 检查评估：确认不是评估脚本或加载 checkpoint 错误。

这份清单的顺序很重要：先排除低级链路 bug，再讨论复杂优化问题。

---

### 20. 面试高频问法

#### 问法 1：训练 loss 不下降，你怎么排查？

可以这样回答：

> 我会先确认不是日志或评估问题，然后做 tiny overfit sanity check。如果 16 条样例都无法过拟合，就检查 batch decode、label mask、可训练参数、梯度和 optimizer step；如果 tiny set 能过拟合但全量不行，再看学习率、batch size、scheduler、数据质量和混合精度稳定性。

#### 问法 2：loss 变成 NaN 怎么办？

可以这样回答：

> 我会先定位 NaN 出现的 step，保存对应 batch，然后检查输入、label、attention mask 和 logits；同时降低学习率、打开 grad clipping、尝试 fp32 或 bf16，查看梯度和参数是否已有 NaN。大模型里还要重点查 fp16 溢出、mask 全 -inf、loss 除零和异常长样例。

#### 问法 3：LoRA 微调不生效怎么排查？

可以这样回答：

> 先打印 trainable parameters，确认 LoRA adapter 注入到正确 target modules；再检查 LoRA 参数是否有梯度、optimizer 是否包含这些参数、一步 step 后参数是否变化；最后做 tiny overfit。如果这些都正常，再调整 rank、alpha、学习率和训练数据。

#### 问法 4：训练 loss 下降但评估指标不涨，可能是什么原因？

可以这样回答：

> 这可能不是不收敛，而是训练目标和评估目标不一致、过拟合、数据分布不一致或评估脚本错误。我会检查评估 prompt、label mapping、generation 参数、是否加载正确 checkpoint/adapter，以及训练集和验证集分布差异。

---

### 21. 工程坑

#### 坑 1：不做 tiny overfit

直接在全量数据上调参，会把简单链路 bug 伪装成复杂优化问题。

#### 坑 2：只盯着学习率

学习率重要，但 label mask、参数冻结、optimizer 参数组错误同样常见。

#### 坑 3：不打印 decode 后的样例

很多数据格式错误只有 decode 成文本后才能发现。

#### 坑 4：混合精度 NaN 只靠降学习率

还要检查 mask、异常 batch、GradScaler、bf16 支持和数值稳定实现。

#### 坑 5：单卡没验证就上多卡

多卡会引入 sampler、同步、ZeRO/FSDP、checkpoint shard 等额外复杂度。

#### 坑 6：训练和推理模板不一致

SFT 中 chat template 不一致会导致训练 loss 正常下降，但线上效果很差。

---

### 22. 小练习

#### 练习 1

写一个函数，统计 batch 中 labels 为 `-100` 的比例，并判断有效监督 token 是否过少。

#### 练习 2

构造 16 条样例，做 tiny overfit，记录 loss 曲线并判断训练链路是否正常。

#### 练习 3

写代码检查一次 backward 后所有可训练参数的梯度 norm、NaN 和 Inf。

#### 练习 4

故意把 LoRA target module 写错，观察 trainable parameters 的变化。

#### 练习 5

分别用 fp16、bf16、fp32 跑一个小训练，比较 NaN 风险和显存占用。


---

### 本讲总结

这一讲整理了训练不收敛的系统 debug 清单。

核心结论如下：

1. 不收敛包括 loss 不下降、下降慢、震荡、爆炸、NaN，以及训练 loss 降但评估不涨。
2. debug 要先做最小可复现实验，每次只改一个变量。
3. tiny overfit 是判断训练链路是否正常的关键 sanity check。
4. batch、labels、loss mask、label shift 和 tokenizer 是高频问题来源。
5. 参数冻结、梯度为空、optimizer 没更新会导致 loss 完全不动。
6. 学习率、warmup、batch size、梯度累积和 scheduler 会影响稳定性。
7. fp16、attention mask、异常 batch 常导致 NaN。
8. LoRA 要重点检查 adapter 是否注入成功、参数是否可训练、target modules 是否匹配。
9. 多卡问题要先用单卡排除，再逐步扩展。
10. 训练 loss 下降但指标不涨时，要同时检查评估脚本和数据分布。

下一讲，我们分析微调后能力退化问题。

## 第 48 讲：微调后能力退化分析

微调后能力退化，是大模型训练和落地中非常常见的问题。

你本来只是想让模型学会某个垂直任务，比如客服问答、SQL 生成、医疗问诊、代码审查或企业知识库问答。训练完成后，模型在目标任务上的表现可能确实提升了，但你很快发现一些副作用：

1. 通用问答能力下降。
2. 推理题答得更差。
3. 代码能力退化。
4. 语言表达变得机械。
5. 模型更容易拒答。
6. 模型更容易幻觉。
7. 模型只会输出训练数据里的固定模板。
8. 原来能遵循的系统指令现在不遵循。
9. 原来多轮对话能接住上下文，现在容易忘。
10. 中文任务提升了，但英文能力掉了。

面试中如果被问到“微调后模型能力退化怎么办”，不能只回答“降低学习率、加数据”。这只是局部手段。更好的回答方式是：先定义退化现象，再定位退化范围，然后从数据、训练、模型、推理、评估五条链路逐层排查。

这一讲给出一套可执行的分析框架。

### 本讲目标

学完本讲，你需要掌握：

1. 什么叫微调后能力退化。
2. 能力退化和训练不收敛有什么区别。
3. 灾难性遗忘、过拟合、分布偏移、格式错配分别是什么。
4. 如何设计退化定位实验。
5. 如何判断退化来自数据、训练、推理还是评估。
6. 如何缓解微调后的通用能力下降。
7. 面试中如何系统回答“微调后能力退化怎么办”。

### 1. 问题设定

假设你有一个基础模型 `base_model`，经过 SFT 或 LoRA 微调后得到 `ft_model`。

微调目标是提升某个业务任务，例如：

```text
输入：用户问题 + 检索到的企业知识
输出：符合客服规范的回答
```

训练后你发现：

```text
业务 FAQ 准确率：72% -> 86%
通用数学题准确率：61% -> 45%
代码生成通过率：38% -> 26%
多轮对话满意度：80% -> 63%
拒答率：5% -> 18%
```

这就是典型的微调收益和能力退化同时出现。

### 2. 能力退化不等于训练失败

训练不收敛通常表现为：

```text
训练 loss 不下降
训练 loss NaN
评估指标完全不涨
模型输出乱码
```

能力退化则更隐蔽：

```text
训练 loss 正常下降
目标任务指标提升
但非目标能力下降
或者目标任务某些子能力下降
```

所以二者的定位思路不同。

训练不收敛优先检查训练链路是否跑通；能力退化则要检查“模型学到了什么，以及忘掉了什么”。

### 3. 常见退化类型

微调后的能力退化可以分成几类。

#### 3.1 通用能力退化

表现为：

```text
常识问答变差
数学推理变差
代码能力变差
翻译能力变差
开放式写作质量下降
```

常见原因：

1. 微调数据太窄。
2. 学习率过大。
3. 训练步数过多。
4. LoRA rank 或可训练参数范围过大。
5. 没有混入通用能力保持数据。

本质是模型参数被目标任务数据过度牵引，原有能力被覆盖。

#### 3.2 目标任务内部退化

有时整体指标提升，但某些子类任务下降。

例如客服模型：

```text
售前咨询提升
售后投诉下降
简单 FAQ 提升
复杂流程类问题下降
短问题提升
长上下文问题下降
```

这通常不是“模型整体坏了”，而是数据分布或评估切片有问题。

#### 3.3 指令遵循退化

表现为：

```text
不按格式输出
忽略 system prompt
不遵守 JSON schema
越权回答
拒答策略异常
```

常见原因：

1. 训练模板和推理模板不一致。
2. 微调数据缺少 system 指令。
3. answer 中混入解释性文本，破坏格式学习。
4. loss mask 把 prompt 也纳入训练目标。
5. SFT 数据和基础模型 chat template 不匹配。

#### 3.4 语言风格退化

表现为：

```text
回答变得啰嗦
回答变得模板化
总是使用固定开头
语气过度客服化
原来的自然表达消失
```

常见原因是训练数据风格单一，且重复样本过多。

#### 3.5 安全与拒答能力异常

表现为两类：

```text
过度拒答：正常问题也拒绝
拒答不足：危险问题也回答
```

这类问题常见于安全数据比例失衡、对齐数据质量差、DPO/RLHF 偏好对构造不合理，或者业务 SFT 数据覆盖了原有安全行为。

### 4. 核心原因一：灾难性遗忘

灾难性遗忘是最常被提到的原因。

简单说，模型原来在预训练和对齐阶段学到很多通用能力；微调时如果只在很窄的数据上更新参数，模型会向新数据分布迁移，从而损失旧能力。

可以用一句话理解：

```text
微调不是只“加能力”，也可能“改写能力”。
```

尤其是全参数微调时，所有层都被更新，遗忘风险更大。

LoRA 虽然只更新低秩增量参数，但如果学习率高、rank 大、训练久，也会明显改变模型行为。

#### 4.1 灾难性遗忘的典型信号

```text
目标任务越训越好，通用 benchmark 越训越差
训练后期目标验证集不再提升，但通用能力持续下降
降低推理温度也不能恢复原能力
去掉业务 prompt 后仍然表现异常
```

#### 4.2 如何验证是否遗忘

最直接的方法是做多 checkpoint 评估。

例如保存：

```text
step_100
step_300
step_500
step_1000
step_2000
```

然后同时评估：

```text
业务任务指标
通用能力指标
安全指标
格式遵循指标
```

如果曲线是这样：

```text
业务指标：持续上升后趋平
通用指标：持续下降
```

说明训练继续进行会增加遗忘，可以考虑 early stopping 或数据混合。

### 5. 核心原因二：微调数据分布太窄

微调数据决定模型被拉向哪里。

如果训练数据都是同一种格式、同一种语气、同一种问题类型，模型就会倾向于在所有场景都套用这种模式。

例如训练数据全部是：

```text
用户：如何办理退款？
助手：您好，关于您咨询的问题，您可以按照以下步骤操作：...
```

模型训练后可能对任何问题都回答：

```text
您好，关于您咨询的问题，您可以...
```

即使用户问的是：

```text
请证明根号 2 是无理数。
```

这不是模型不会数学，而是微调让它形成了强业务风格偏置。

#### 5.1 数据窄的常见形态

```text
领域窄：只覆盖一个业务场景
格式窄：全部是固定模板
长度窄：全部是短问短答
语言窄：只覆盖中文或英文
难度窄：只有简单样本
风格窄：全部是同一种客服话术
```

#### 5.2 如何诊断数据分布问题

可以统计训练集：

```text
问题长度分布
答案长度分布
任务类型占比
语言占比
拒答样本占比
JSON/非 JSON 样本占比
多轮/单轮样本占比
重复样本比例
模板化开头占比
```

如果某类样本占比异常高，就要警惕模型被该模式牵引。

### 6. 核心原因三：过拟合

过拟合和灾难性遗忘相关，但角度不同。

灾难性遗忘关注旧能力下降；过拟合关注模型过度贴合训练集。

过拟合的表现：

```text
训练 loss 持续下降
验证 loss 先降后升
训练集输出很好
相似但未见过的问题表现差
生成结果复读训练集模板
```

#### 6.1 微调中容易过拟合的情况

```text
数据量很小
训练 epoch 太多
学习率太高
LoRA rank 太大
dropout 太低
样本重复太多
验证集和训练集分布差异大
```

#### 6.2 过拟合的排查方式

至少看三条曲线：

```text
train loss
validation loss
业务评估指标
```

更好的方式是额外加：

```text
通用能力指标
格式遵循指标
安全指标
```

只看 train loss 很危险，因为 train loss 越低不代表模型越好。

### 7. 核心原因四：训练模板和推理模板不一致

这是实战中非常高频的问题。

训练时样本可能是：

```text
<|system|>
你是一个客服助手。
<|user|>
{question}
<|assistant|>
{answer}
```

推理时却是：

```text
用户问题：{question}
请回答：
```

或者训练时用了基础模型的 chat template，推理时服务端又手写了一套 prompt。

这会导致模型学到的条件分布和推理时看到的输入分布不一致。

#### 7.1 模板错配的表现

```text
线下评估好，线上效果差
同一问题在训练脚本中正常，在服务端异常
模型输出特殊 token
模型不遵循 role 边界
模型把用户问题续写下去，而不是回答
```

#### 7.2 模板错配检查清单

检查以下内容是否一致：

```text
tokenizer
chat_template
bos token
eos token
pad token
system/user/assistant role 标记
训练时是否 add_generation_prompt
推理时是否 add_generation_prompt
多轮拼接方式
截断方向
```

面试中提到这一点很加分，因为很多候选人只会从学习率和数据量解释退化，忽略模板错配。

### 8. 核心原因五：loss mask 错误

SFT 中通常只希望模型学习 assistant answer，不希望模型学习 user prompt。

正确的 labels 一般是：

```text
input:  [system tokens][user tokens][assistant tokens]
label:  [-100        ][-100       ][assistant tokens]
```

如果把 prompt 部分也纳入 loss，模型会学着预测用户问题、系统指令和格式标记，可能导致生成时角色混乱。

#### 8.1 loss mask 错误的表现

```text
模型复述用户输入
模型生成 user role
模型输出 system prompt
模型提前停止
模型格式混乱
```

#### 8.2 最小检查代码

下面的代码用于检查一个 batch 中 labels 是否正确 mask。

```python
def inspect_sft_batch(tokenizer, batch, n=1):
    input_ids = batch["input_ids"][:n]
    labels = batch["labels"][:n]

    for i in range(input_ids.size(0)):
        ids = input_ids[i].tolist()
        lbs = labels[i].tolist()

        visible_label_ids = [x for x in lbs if x != -100]
        print("=" * 80)
        print("FULL INPUT:")
        print(tokenizer.decode(ids, skip_special_tokens=False))
        print("\nLOSS TOKENS:")
        print(tokenizer.decode(visible_label_ids, skip_special_tokens=False))
```

你希望 `LOSS TOKENS` 基本只包含 assistant 的回答部分。

### 9. 核心原因六：学习率和训练步数过激

微调不是从零训练，基础模型已经有很强的能力。学习率过高会快速破坏已有参数分布。

常见现象：

```text
前几百 step 目标任务提升很快
继续训练后通用能力快速下降
输出风格越来越单一
模型越来越自信但错误更多
```

对于 LoRA 微调，很多项目会使用比全参微调更高的学习率，但这不代表越高越好。

#### 9.1 建议排查实验

做一个小网格：

```text
learning_rate: 1e-5, 2e-5, 5e-5, 1e-4
epochs: 1, 2, 3
lora_rank: 8, 16, 32
```

不要只比较目标任务指标，还要比较通用能力保持率。

### 10. 核心原因七：数据质量问题

低质量数据会让模型学习错误行为。

常见问题：

```text
答案事实错误
答案和问题不匹配
多个答案风格互相冲突
拒答标签错误
安全边界不一致
工具调用格式不统一
JSON 不合法
答案中混入来源、标注员备注或脏字符
```

微调数据不需要无限大，但需要足够干净。对于小规模高质量 SFT，脏数据的影响尤其明显。

#### 10.1 数据抽检方法

至少做三类抽检：

```text
随机抽样：看整体质量
低 loss 样本：看是否大量模板化或重复
高 loss 样本：看是否脏样本、长尾样本或标注冲突
```

如果高 loss 样本中大量是标注错误，不应该简单继续训练，而应该清洗数据。

### 11. 核心原因八：对齐数据副作用

如果微调包含偏好优化，例如 DPO、IPO、KTO 或 RLHF，能力退化可能来自偏好数据。

偏好优化不是简单让模型“更好”，而是让模型更偏向 chosen、远离 rejected。

如果偏好对构造不合理，模型可能学到错误偏好。

例如：

```text
chosen 总是很短，rejected 总是很长
```

模型可能学会“短就是好”。

再例如：

```text
chosen 总是拒答，rejected 总是尝试回答
```

模型可能学会过度拒答。

#### 11.1 DPO 后退化的典型表现

```text
回答变短
信息量下降
过度安全
不愿意推理
不愿意给明确答案
语言更圆滑但准确率下降
```

#### 11.2 排查方向

检查偏好数据中：

```text
chosen/rejected 长度差
chosen/rejected 安全标签分布
chosen/rejected 格式差异
chosen 是否真的更正确
rejected 是否只是风格不同而非错误
```

### 12. 建立能力退化评估集

没有评估集，就无法谈退化。

微调项目至少需要四类评估集：

```text
目标任务评估集
通用能力保持集
安全评估集
格式/指令遵循评估集
```

如果是垂直业务，还应该做切片：

```text
高频问题
低频问题
简单问题
复杂问题
短上下文
长上下文
单轮问题
多轮问题
有检索证据
无检索证据
```

### 13. 一个可执行的退化对比脚本

下面给出一个简化评估框架，用于比较 base model 和 finetuned model。

实际项目中可以替换为自己的推理服务或 vLLM 接口。

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class EvalCase:
    category: str
    prompt: str
    reference: str | None = None


def exact_or_contains_score(output: str, reference: str | None) -> float:
    if reference is None:
        return 0.0
    return 1.0 if reference.strip() in output else 0.0


def run_eval(
    generate: Callable[[str], str],
    cases: list[EvalCase],
    score_fn: Callable[[str, str | None], float],
) -> dict[str, float]:
    bucket_scores: dict[str, list[float]] = {}

    for case in cases:
        output = generate(case.prompt)
        score = score_fn(output, case.reference)
        bucket_scores.setdefault(case.category, []).append(score)

    return {
        category: sum(scores) / len(scores)
        for category, scores in bucket_scores.items()
    }


def compare_models(base_generate, ft_generate, cases):
    base_result = run_eval(base_generate, cases, exact_or_contains_score)
    ft_result = run_eval(ft_generate, cases, exact_or_contains_score)

    categories = sorted(set(base_result) | set(ft_result))
    for category in categories:
        base_score = base_result.get(category, 0.0)
        ft_score = ft_result.get(category, 0.0)
        delta = ft_score - base_score
        print(f"{category:20s} base={base_score:.3f} ft={ft_score:.3f} delta={delta:+.3f}")
```

这个脚本很简单，但体现了一个关键思想：不要只看整体平均分，要按能力类别看差异。

### 14. 退化定位实验设计

发现退化后，不要马上改训练参数。先定位。

推荐按以下顺序做实验。

#### 14.1 对比 base 与 ft

同一批 prompts，分别跑：

```text
base model
finetuned model
```

比较：

```text
准确率
输出长度
拒答率
格式错误率
幻觉率
人工偏好胜率
```

如果所有维度都变差，可能是训练链路或推理模板有严重问题。

如果只有某些能力变差，进入切片分析。

#### 14.2 对比多个 checkpoint

不要只看最终模型。

对比：

```text
base
checkpoint_early
checkpoint_middle
checkpoint_final
```

如果早期 checkpoint 最好，说明训练过头。

#### 14.3 对比不同训练数据版本

例如：

```text
clean_data
raw_data
with_general_replay
without_general_replay
without_duplicate_samples
```

如果加入通用 replay 后退化明显缓解，说明遗忘是主因。

如果清洗数据后目标任务和通用能力都提升，说明数据质量是主因。

#### 14.4 对比不同推理模板

用同一个模型，分别测试：

```text
训练时 chat template
线上服务 prompt
裸 prompt
带 system prompt
不带 system prompt
```

如果差异巨大，优先修模板。

### 15. 缓解方法一：混入通用能力保持数据

最直接的方法是加入 replay data，也就是通用能力保持数据。

训练数据变成：

```text
业务 SFT 数据 + 通用指令数据 + 安全数据 + 格式遵循数据
```

比例需要实验，一般不是越多越好。

可以从小比例开始：

```text
业务数据 90% + 通用保持数据 10%
业务数据 80% + 通用保持数据 20%
业务数据 70% + 通用保持数据 30%
```

如果通用数据比例过高，业务能力提升可能变弱。

### 16. 缓解方法二：降低更新强度

可以从以下角度降低微调对模型的扰动：

```text
降低 learning rate
减少 epoch
使用 early stopping
降低 LoRA rank
增大 LoRA dropout
减少 target modules
只训练部分层
使用更强正则
```

工程上通常先做三件事：

```text
降低学习率
减少训练轮数
保存并评估多个 checkpoint
```

这是性价比最高的组合。

### 17. 缓解方法三：控制 LoRA 作用范围

LoRA 常见 target modules 包括：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

如果业务任务主要是格式、风格或轻量领域适配，不一定要训练所有模块。

训练 MLP 相关模块可能带来更强表达改变，也可能更容易影响通用能力。训练 attention 投影模块通常更保守，但具体仍要实验。

可对比：

```text
attention only
mlp only
attention + mlp
```

然后看目标任务收益和通用能力损失。

### 18. 缓解方法四：改进数据配比和采样

数据不是简单拼接，采样比例会显著影响训练结果。

如果某类模板样本重复出现太多，模型会强烈学习该模板。

可以使用：

```text
去重
按任务类型均衡采样
按难度分层采样
限制同模板样本比例
提高长尾任务采样权重
控制拒答样本比例
```

对于企业知识库问答，尤其要避免训练集中大量“根据资料无法回答”导致模型上线后过度拒答。

### 19. 缓解方法五：保留基础模型能力的评估门禁

微调上线不应该只设置业务指标门禁。

推荐设置如下门禁：

```text
业务指标必须提升
通用能力下降不得超过阈值
安全指标不得下降
格式错误率不得上升
拒答率不得异常变化
线上核心 query 人工胜率必须提升
```

例如：

```text
业务准确率 +8%
通用能力平均下降 <= 2%
安全违规率不升高
JSON 格式错误率 <= 1%
```

这样可以避免“业务 benchmark 提升，但整体产品体验下降”。

### 20. 常见工程坑

#### 20.1 只评估目标任务

只看目标任务会掩盖遗忘。

正确做法是同时评估目标任务和保留能力。

#### 20.2 只看自动指标

自动指标不能完全反映生成质量。

开放式问答、客服质量、代码解释、长文写作都需要人工评测或 LLM-as-judge 辅助。

#### 20.3 验证集泄漏

如果训练集和验证集高度重复，目标任务指标会虚高，真实线上能力可能下降。

#### 20.4 推理参数变化

有时不是模型退化，而是推理参数变了。

需要检查：

```text
temperature
top_p
top_k
repetition_penalty
max_new_tokens
stop words
```

#### 20.5 tokenizer 或 special token 不一致

如果训练和推理加载了不同 tokenizer，或者 special token 配置不一致，会导致输出异常。

#### 20.6 合并 LoRA 后效果变化

有时 adapter 推理正常，merge 后异常。

需要检查：

```text
base model 是否一致
dtype 是否一致
merge 过程是否正确
量化后是否重新评估
```

### 21. 面试高频问法

#### 问法 1：微调后通用能力下降怎么办？

可以这样答：

```text
我会先确认下降是否真实存在，而不是评估或推理模板问题。具体会用 base model 和 finetuned model 在同一套通用能力集、业务集、安全集上对比，并按任务类型切片。

如果确认是通用能力下降，我会进一步看多 checkpoint 曲线，判断是不是训练过头或灾难性遗忘。缓解上可以降低学习率、减少 epoch、early stopping、降低 LoRA rank 或 target modules 范围，同时混入一定比例的通用 replay 数据和安全数据。最后用业务收益和通用能力保持率共同做上线门禁。
```

#### 问法 2：怎么区分灾难性遗忘和过拟合？

可以这样答：

```text
灾难性遗忘强调旧能力下降，过拟合强调对训练集过度贴合。两者可能同时出现。

我会看训练集、验证集、目标任务测试集和通用能力集的曲线。如果训练 loss 持续下降、验证 loss 上升，并且相似未见样本变差，这是过拟合信号。如果目标任务仍提升但通用数学、代码、常识、安全能力持续下降，这是遗忘信号。多 checkpoint 对比和 replay data ablation 可以进一步定位。
```

#### 问法 3：为什么 LoRA 微调也会造成能力退化？

可以这样答：

```text
LoRA 虽然冻结了原始参数，但它在前向中加入了可训练的低秩增量，本质上仍然改变了模型的函数。如果 rank 较大、学习率较高、训练步数较多，或者 target modules 覆盖 attention 和 MLP 的多个投影层，LoRA 的增量也可能显著改变输出分布。因此 LoRA 不是天然不会遗忘，只是通常比全参微调更可控。
```

#### 问法 4：微调后模型过度拒答，怎么排查？

可以这样答：

```text
我会先统计拒答率在 base 和 finetuned model 上的变化，并按安全问题、正常业务问题、无答案问题分开看。然后检查训练数据中拒答样本比例和标注是否合理，尤其是“无法回答”的样本是否过多。还要检查 DPO 或偏好数据里 chosen 是否大量是拒答。如果是 RAG 场景，还要区分模型拒答和检索无证据导致的拒答。缓解上可以调整拒答数据比例、补充正常可答样本、明确拒答边界，并设置拒答率门禁。
```

### 22. 一套完整排查清单

遇到微调后能力退化，可以按下面顺序排查：

```text
1. 确认退化现象
   - 哪些能力下降
   - 下降幅度多少
   - 是否有统计显著性

2. 排除评估问题
   - 评估集是否变化
   - 指标计算是否一致
   - decoding 参数是否一致
   - prompt 模板是否一致

3. 对比 base 和 finetuned
   - 目标任务
   - 通用能力
   - 安全能力
   - 格式遵循

4. 做 checkpoint 曲线
   - early checkpoint 是否更好
   - 是否训练过头
   - 是否目标收益和通用损失存在 trade-off

5. 检查数据
   - 分布是否过窄
   - 是否重复
   - 是否脏数据
   - 是否拒答比例异常
   - 是否模板化严重

6. 检查训练配置
   - learning rate
   - epoch
   - LoRA rank
   - target modules
   - dropout
   - warmup

7. 检查 SFT 细节
   - chat template
   - loss mask
   - special tokens
   - truncation

8. 做缓解实验
   - 降学习率
   - 减少 epoch
   - 加 replay data
   - 数据去重清洗
   - 调整 LoRA 范围
   - early stopping

9. 建立上线门禁
   - 业务收益
   - 通用能力保持
   - 安全指标
   - 格式错误率
   - 人工评测
```

### 23. 本讲小结

这一讲分析了微调后能力退化的原因和排查方法。

核心结论如下：

1. 能力退化不等于训练失败，目标任务提升也可能伴随其他能力下降。
2. 常见退化包括通用能力退化、目标任务内部退化、指令遵循退化、风格退化和安全拒答异常。
3. 灾难性遗忘来自微调数据对模型已有能力的覆盖。
4. 数据分布太窄会让模型形成强业务偏置和模板化输出。
5. 过拟合会导致训练集表现好，但未见样本和通用能力变差。
6. 训练模板、推理模板、chat template、special token 和 loss mask 是高频工程坑。
7. DPO 等偏好优化可能导致回答变短、过度拒答或信息量下降。
8. 排查退化要对比 base、finetuned 和多个 checkpoint，并按能力切片。
9. 缓解方法包括混入通用 replay 数据、降低更新强度、控制 LoRA 范围、清洗数据和设置上线门禁。
10. 面试中要体现系统性：先确认现象，再定位来源，最后给出可验证的缓解实验。

下一讲，我们设计生成质量人工评测表。

## 第 49 讲：生成质量人工评测表设计

大模型生成质量评估不能只依赖自动指标。

对于分类、抽取、检索召回这类任务，准确率、F1、Recall、NDCG 等指标通常比较直接。但对于开放式生成任务，例如客服问答、长文总结、代码解释、RAG 问答、智能体规划、多轮对话，很多问题没有唯一标准答案。

这时人工评测仍然非常重要。

但人工评测不能随便让几个人“凭感觉打分”。如果评测表设计不好，会出现几个问题：

1. 标注员标准不一致。
2. 分数不可复现。
3. 不同模型差距看起来很大，但实际没有统计意义。
4. 评测维度混在一起，无法定位问题。
5. 模型上线决策缺乏依据。

这一讲，我们设计一套可以直接用于项目的生成质量人工评测表。

### 本讲目标

学完本讲，你需要掌握：

1. 为什么生成任务需要人工评测。
2. 人工评测表应该包含哪些维度。
3. 如何设计 1-5 分打分标准。
4. 如何做 pairwise 偏好评测。
5. 如何控制标注一致性。
6. 如何统计人工评测结果。
7. 面试中如何回答“如何评估生成质量”。

### 1. 问题设定

假设你负责评估两个客服问答模型：

```text
Model A：当前线上模型
Model B：新微调模型
```

输入包括：

```text
用户问题
检索证据
历史对话
模型回答
```

你需要回答：

```text
Model B 是否比 Model A 更好？
能不能上线？
如果不能上线，主要问题是什么？
```

这不是一个单一准确率能解决的问题。

你需要评估：

```text
是否回答正确
是否基于证据
是否有幻觉
是否完整
是否简洁
是否符合格式
是否安全
是否符合业务语气
用户是否更满意
```

### 2. 人工评测的两种基本范式

生成质量人工评测常见两种范式。

#### 2.1 Absolute Rating

Absolute rating 是绝对打分。

标注员看到一个模型输出后，对多个维度分别打分，例如：

```text
事实正确性：1-5 分
完整性：1-5 分
简洁性：1-5 分
格式遵循：1-5 分
安全性：1-5 分
```

优点：

1. 能定位具体维度问题。
2. 适合做质量看板。
3. 适合分析模型短板。

缺点：

1. 标注员之间标准容易漂移。
2. 不同人对 3 分、4 分理解可能不同。
3. 对细微模型差异不够敏感。

#### 2.2 Pairwise Preference

Pairwise preference 是成对偏好评测。

标注员同时看到两个模型回答，但不知道哪个来自哪个模型，然后选择：

```text
A 更好
B 更好
差不多
都不好
```

优点：

1. 更接近用户真实选择。
2. 对模型差异更敏感。
3. 标注员更容易判断“哪个更好”。

缺点：

1. 不一定能说明为什么更好。
2. 需要盲评和随机化，避免位置偏差。
3. 不适合单模型长期质量监控。

实际项目中常常两者结合：先用 absolute rating 定位问题，再用 pairwise preference 做上线对比。

### 3. 评测表的核心字段

一个基础评测表至少包含以下字段。

```text
case_id
task_type
user_query
context_or_evidence
conversation_history
model_name
model_output
reference_answer
factuality_score
groundedness_score
completeness_score
instruction_following_score
format_score
conciseness_score
safety_score
tone_score
overall_score
error_tags
annotator_id
annotation_time
comment
```

如果是 pairwise 评测，还需要：

```text
answer_a
answer_b
position_random_seed
winner
preference_reason
```

### 4. 维度一：事实正确性

事实正确性评估回答是否真实、准确、没有明显错误。

对于闭卷问答，事实正确性看回答是否符合常识或标准答案。

对于 RAG 问答，事实正确性还要结合检索证据。

#### 4.1 1-5 分标准

```text
5 分：回答完全正确，没有事实错误，关键结论清楚。
4 分：基本正确，只有轻微不影响结论的小瑕疵。
3 分：部分正确，但遗漏或混淆了一些重要信息。
2 分：存在明显事实错误，用户按此执行可能受影响。
1 分：主要结论错误，或回答与问题完全不相关。
```

#### 4.2 标注注意事项

事实正确性不要被语言流畅度干扰。

一个回答写得很流畅，但事实错了，不能给高分。

### 5. 维度二：证据一致性

证据一致性也可以叫 groundedness，尤其适用于 RAG。

它评估模型回答是否基于给定资料，而不是凭空发挥。

#### 5.1 1-5 分标准

```text
5 分：所有关键结论都能在证据中找到支持。
4 分：大部分结论有证据支持，少量非关键表述略有扩展。
3 分：部分结论有证据支持，但也包含未证实信息。
2 分：大量内容无法从证据中推出。
1 分：回答主要是编造，或与证据矛盾。
```

#### 5.2 常见错误标签

```text
unsupported_claim：无证据断言
contradiction：与证据矛盾
over_generalization：过度概括
source_misuse：错误使用资料
```

RAG 场景中，证据一致性通常比语言优美更重要。

### 6. 维度三：完整性

完整性评估回答是否覆盖用户问题中的关键需求。

例如用户问：

```text
请比较 LoRA 和全参数微调的优缺点，并说明适用场景。
```

如果模型只回答了 LoRA 优点，没有比较全参微调，也没有适用场景，就不完整。

#### 6.1 1-5 分标准

```text
5 分：完整覆盖用户所有关键需求，结构清楚。
4 分：覆盖大部分需求，只有少量非关键遗漏。
3 分：覆盖一部分需求，但遗漏明显。
2 分：只回答了很小一部分问题。
1 分：基本没有回答用户真正的问题。
```

完整性和简洁性可能冲突。好回答不是越长越好，而是在必要范围内覆盖关键点。

### 7. 维度四：指令遵循

指令遵循评估模型是否遵守用户和系统要求。

例如：

```text
请用三点回答
请只输出 JSON
请不要解释过程
请用中文回答
请站在面试官角度评价
```

#### 7.1 1-5 分标准

```text
5 分：完全遵守所有明确指令。
4 分：基本遵守，只有轻微偏离。
3 分：遵守了部分指令，但有明显遗漏。
2 分：大部分指令未遵守。
1 分：完全无视指令，或违反关键约束。
```

指令遵循在工具调用、结构化输出、Agent 场景中尤其关键。

### 8. 维度五：格式正确性

格式正确性评估输出是否满足预期格式。

常见格式包括：

```text
JSON
Markdown 表格
SQL
YAML
函数调用参数
固定客服话术模板
```

对于结构化输出任务，格式错误可能直接导致系统不可用。

#### 8.1 1-5 分标准

```text
5 分：格式完全合法，可被下游系统直接解析。
4 分：格式基本正确，只有轻微不影响解析的问题。
3 分：格式有问题，需要少量人工修复。
2 分：格式错误明显，下游系统难以解析。
1 分：完全不符合要求格式。
```

如果是 JSON 输出，最好不要只靠人工判断，而要配合自动 parser 检查。

### 9. 维度六：简洁性与可读性

简洁性评估回答是否避免废话和重复。

可读性评估结构是否清晰，用户是否容易理解。

#### 9.1 1-5 分标准

```text
5 分：表达清楚、简洁、有结构，没有明显废话。
4 分：整体清晰，略有冗余。
3 分：能读懂，但结构一般或废话较多。
2 分：表达混乱，用户需要反复阅读。
1 分：难以理解，或大量重复、复读。
```

注意：简洁不是短。复杂问题需要足够解释，不能因为回答长就扣分。

### 10. 维度七：安全性

安全性评估回答是否包含危险、违规、隐私泄露、歧视或不当建议。

常见风险：

```text
违法操作指导
危险品制作
医疗误导
金融误导
隐私泄露
仇恨歧视
越权承诺
```

#### 10.1 1-5 分标准

```text
5 分：安全合规，拒答边界合理。
4 分：基本安全，有轻微措辞风险。
3 分：存在潜在风险，需要人工复核。
2 分：包含明显不安全建议。
1 分：严重违规或可能造成现实伤害。
```

安全性通常是上线门禁维度，不应该简单和其他维度平均后抵消。

### 11. 维度八：业务语气与用户体验

很多业务模型不仅要答对，还要符合场景语气。

例如客服场景需要：

```text
礼貌
清楚
不过度承诺
不推责
不机械
```

医疗问答需要：

```text
谨慎
不替代医生诊断
给出就医建议边界
```

代码助手需要：

```text
直接
可执行
解释必要假设
```

#### 11.1 1-5 分标准

```text
5 分：语气完全符合业务场景，用户体验好。
4 分：整体符合，有轻微风格问题。
3 分：能接受，但语气不够自然或不够专业。
2 分：语气明显不符合场景。
1 分：语气严重不当，可能引起用户反感或投诉。
```

### 12. Overall Score 不能简单平均

很多团队会把各维度求平均，得到 overall score。

这很方便，但有风险。

例如：

```text
事实正确性：1 分
语言流畅性：5 分
格式正确性：5 分
安全性：5 分
平均分：4 分
```

这个回答平均分很高，但事实完全错误，不能上线。

更合理的方式是设置硬门禁。

例如：

```text
事实正确性 < 3：整体不合格
安全性 < 4：整体不合格
格式正确性 < 4：结构化任务不合格
证据一致性 < 3：RAG 问答不合格
```

然后再对合格样本计算加权分。

### 13. 推荐评测表模板

下面是一份可直接使用的评测表模板。

```markdown
## 生成质量人工评测表

### 样本信息

- case_id：
- task_type：
- user_query：
- context/evidence：
- conversation_history：
- model_output：
- reference_answer：

### 打分维度

| 维度 | 分数 1-5 | 是否硬门禁 | 备注 |
|---|---:|---|---|
| 事实正确性 |  | 是 | 是否有事实错误 |
| 证据一致性 |  | RAG 场景是 | 是否基于给定证据 |
| 完整性 |  | 否 | 是否覆盖关键需求 |
| 指令遵循 |  | 是 | 是否遵守用户/系统要求 |
| 格式正确性 |  | 结构化任务是 | 是否能被下游解析 |
| 简洁可读 |  | 否 | 是否清晰、无冗余 |
| 安全合规 |  | 是 | 是否存在安全风险 |
| 业务语气 |  | 否 | 是否符合业务场景 |

### 错误标签

- hallucination：
- unsupported_claim：
- missing_key_point：
- instruction_violation：
- format_error：
- unsafe_content：
- over_refusal：
- too_verbose：
- irrelevant_answer：

### 总体结论

- overall_score：
- pass/fail：
- 主要问题：
- 标注员：
- 标注时间：
```

### 14. Pairwise 偏好评测表模板

当你比较两个模型版本时，推荐使用盲评。

```markdown
## 成对偏好评测表

### 样本信息

- case_id：
- task_type：
- user_query：
- context/evidence：

### 回答 A

{answer_a}

### 回答 B

{answer_b}

### 选择

- A 明显更好
- A 略好
- 差不多
- B 略好
- B 明显更好
- 都不好

### 判断依据

- 正确性：
- 完整性：
- 证据一致性：
- 指令遵循：
- 安全性：
- 其他原因：

### 错误标签

- A 的主要问题：
- B 的主要问题：
```

成对评测必须随机 A/B 位置，否则标注员可能产生位置偏差。

### 15. 标注一致性控制

人工评测最大的问题是主观性。

需要通过流程降低主观噪声。

#### 15.1 编写详细标注指南

标注指南至少包括：

```text
每个维度定义
每个分数对应标准
正例
反例
边界案例
常见错误标签
遇到不确定情况如何处理
```

不要只给一个表格就开始标注。

#### 15.2 标注员校准

正式标注前，先让所有标注员共同标一批样本。

然后讨论：

```text
为什么这个样本是 3 分不是 4 分
为什么这个回答算幻觉
为什么这个拒答是合理还是过度拒答
```

校准后再正式标注。

#### 15.3 多人复标

关键样本建议至少两人标注。

如果分歧过大，交给仲裁人。

例如：

```text
标注员 1：事实正确性 5 分
标注员 2：事实正确性 2 分
```

这种样本必须复核，因为二者至少有一方误解了标准或事实。

#### 15.4 计算一致性

可以计算：

```text
一致率
Cohen's Kappa
Krippendorff's Alpha
Spearman 相关
```

实际面试中不需要推公式，但要知道一致性不是靠感觉保证的。

### 16. 评测样本怎么选

评测样本质量决定结论可信度。

不要只抽简单样本。

推荐分层采样：

```text
高频问题
低频长尾问题
简单问题
复杂问题
短上下文
长上下文
单轮对话
多轮对话
有明确证据
证据不足
安全敏感问题
格式约束问题
```

如果样本只来自高频简单问题，评测结果会高估模型能力。

### 17. 结果统计方法

人工评测完成后，不要只报告一个平均分。

至少报告：

```text
各维度平均分
各维度通过率
硬门禁失败率
错误标签分布
按任务类型切片结果
pairwise 胜率
置信区间或 bootstrap 结果
典型 bad cases
```

例如：

```text
Model B 相比 Model A：
事实正确性 +0.3
证据一致性 +0.4
完整性 +0.2
格式错误率 3.1% -> 1.2%
安全失败率 0.5% -> 0.6%
pairwise 胜率 58%，败率 31%，持平 11%
主要问题：长上下文下仍有 unsupported_claim
```

这样的结论比“B 平均分更高”有用得多。

### 18. 一个简单统计脚本

假设人工评测结果保存为 CSV，每行一个样本。

```python
import pandas as pd


HARD_GATES = {
    "factuality_score": 3,
    "safety_score": 4,
    "instruction_following_score": 3,
}


def summarize_absolute_eval(path: str) -> None:
    df = pd.read_csv(path)
    score_cols = [c for c in df.columns if c.endswith("_score")]

    print("== Mean Scores ==")
    print(df.groupby("model_name")[score_cols].mean().round(3))

    print("\n== Hard Gate Failure Rate ==")
    for model, group in df.groupby("model_name"):
        fail = pd.Series(False, index=group.index)
        for col, threshold in HARD_GATES.items():
            fail = fail | (group[col] < threshold)
        print(model, round(fail.mean(), 4))

    if "error_tags" in df.columns:
        print("\n== Error Tags ==")
        tags = (
            df.assign(error_tags=df["error_tags"].fillna("").str.split(";"))
            .explode("error_tags")
        )
        tags = tags[tags["error_tags"].str.len() > 0]
        print(tags.groupby(["model_name", "error_tags"]).size().sort_values(ascending=False))


def summarize_pairwise_eval(path: str) -> None:
    df = pd.read_csv(path)
    print("== Winner Distribution ==")
    print(df["winner"].value_counts(normalize=True).round(4))

    if "task_type" in df.columns:
        print("\n== Winner by Task Type ==")
        table = pd.crosstab(df["task_type"], df["winner"], normalize="index")
        print(table.round(4))
```

这个脚本体现两个重点：

1. 各维度分开统计。
2. 硬门禁失败率单独统计。

### 19. LLM-as-Judge 和人工评测的关系

LLM-as-Judge 可以降低评测成本，但不能完全替代人工评测。

它适合：

```text
大规模初筛
回归测试
辅助错误标签
给人工标注提供参考
```

它不适合完全替代：

```text
高风险安全判断
复杂业务规则判断
法律医疗金融等强专业场景
最终上线决策
```

更稳妥的做法是：

```text
人工小样本高质量评测 + LLM-as-Judge 大规模回归评测
```

并定期用人工结果校准 judge prompt。

### 20. 常见工程坑

#### 20.1 标注员知道模型身份

如果标注员知道哪个是新模型，可能产生预期偏差。

应尽量盲评。

#### 20.2 A/B 位置不随机

如果新模型总在 B 位置，标注员可能形成位置偏好。

必须随机化回答顺序。

#### 20.3 维度定义重叠

例如把事实正确性、证据一致性、完整性混成一个“质量分”，会导致无法定位问题。

#### 20.4 分数没有锚点

如果没有说明 3 分和 4 分差别，标注员只能凭感觉。

#### 20.5 只看平均分

平均分会掩盖安全、格式、事实错误等硬问题。

#### 20.6 样本太少或太简单

样本量过小会导致结论不稳定，样本太简单会高估模型。

#### 20.7 没有 bad case 复盘

人工评测最有价值的不只是分数，而是 bad cases。bad cases 能直接指导下一轮数据清洗、prompt 优化和微调策略。

### 21. 面试高频问法

#### 问法 1：如何评估一个问答模型的生成质量？

可以这样答：

```text
我不会只看 BLEU 或 ROUGE 这类自动指标，因为开放式问答往往没有唯一标准答案。我会建立人工评测和自动评测结合的体系。

人工评测上，我会按事实正确性、证据一致性、完整性、指令遵循、格式正确性、安全性、简洁可读性和业务语气等维度打分。对于 RAG 场景，证据一致性和幻觉率是重点。对于结构化输出，格式可解析是硬门禁。对于上线对比，我会做 blind pairwise preference，随机 A/B 位置，统计新模型相对旧模型的胜率、败率和持平率，并按任务类型切片。
```

#### 问法 2：人工评测如何保证可靠？

可以这样答：

```text
首先要有清晰的标注指南，每个维度给出 1-5 分锚点、正例、反例和边界案例。正式标注前要做标注员校准，让大家对标准达成一致。关键样本可以多人复标，对分歧大的样本做仲裁。最后统计一致率或 Kappa 等一致性指标，避免完全依赖主观感觉。
```

#### 问法 3：为什么不能只用平均分判断模型上线？

可以这样答：

```text
因为不同维度的重要性不一样。比如一个回答事实错误但语言流畅，如果简单平均，可能分数仍然不低，但实际不能上线。安全性、事实正确性、格式正确性这类维度应该设置硬门禁，先判断是否合格，再对合格样本做加权评分或 pairwise 比较。
```

#### 问法 4：LLM-as-Judge 能否替代人工评测？

可以这样答：

```text
不能完全替代。LLM-as-Judge 很适合低成本大规模回归评测和初筛，但它本身也可能有偏差，尤其在专业领域、安全边界和复杂业务规则上不一定可靠。更合理的方案是用人工小样本高质量评测作为校准集，再用 LLM-as-Judge 做大规模自动化评估，并定期用人工结果校准 judge prompt 和判分标准。
```

### 22. 一套可落地流程

实际项目中可以按下面流程执行：

```text
1. 明确评测目标
   - 是上线对比还是质量诊断
   - 是通用问答、RAG、客服还是工具调用

2. 设计评测样本
   - 分层采样
   - 覆盖高频、长尾、复杂、安全、格式约束样本

3. 设计评测表
   - absolute rating 维度
   - pairwise preference 表
   - 错误标签体系

4. 编写标注指南
   - 分数锚点
   - 正例反例
   - 边界案例

5. 标注员校准
   - 小批量试标
   - 讨论分歧
   - 修订指南

6. 正式标注
   - 盲评
   - A/B 随机
   - 关键样本多人复标

7. 统计结果
   - 维度均分
   - 硬门禁失败率
   - pairwise 胜率
   - 错误标签分布
   - 切片分析

8. 复盘 bad cases
   - 数据问题
   - prompt 问题
   - 模型能力问题
   - 检索问题

9. 做上线决策
   - 业务收益是否显著
   - 安全和事实硬门禁是否满足
   - 是否需要灰度实验
```

### 23. 本讲小结

这一讲设计了一套生成质量人工评测表。

核心结论如下：

1. 开放式生成任务不能只依赖自动指标，人工评测仍然关键。
2. Absolute rating 适合定位问题，pairwise preference 适合比较两个模型版本。
3. 评测维度应包括事实正确性、证据一致性、完整性、指令遵循、格式正确性、安全性、简洁可读性和业务语气。
4. 每个维度都要有明确的 1-5 分锚点，不能让标注员凭感觉打分。
5. Overall score 不能简单平均，事实、安全、格式等关键维度应设置硬门禁。
6. 人工评测需要标注指南、标注员校准、多人复标和一致性统计。
7. 评测样本要分层采样，覆盖高频、长尾、复杂、多轮、安全和格式约束场景。
8. 结果统计要看维度均分、门禁失败率、错误标签、切片结果和 pairwise 胜率。
9. LLM-as-Judge 可以辅助大规模评估，但不能完全替代人工评测。
10. 面试中要体现工程闭环：设计评测表、保证一致性、统计结果、复盘 bad cases、支撑上线决策。

下一讲，我们分析如何 A/B 测试两个模型版本。

## 第 50 讲：A/B 测试两个模型版本

离线评测通过，不代表模型一定能上线。

人工评测、自动评测、benchmark、bad case 回归都只能说明模型在已有样本上的表现。真实用户流量更复杂：问题分布会变，用户表达会变，系统链路会变，成本和延迟也会影响体验。

所以模型上线前，常常需要做 A/B 测试。

A/B 测试的核心问题是：

```text
在真实线上环境中，新模型 B 是否显著优于旧模型 A？
```

这里的“优于”不是一句主观判断，而是要落到指标、实验设计、统计显著性、风险控制和上线决策上。

### 本讲目标

学完本讲，你需要掌握：

1. 为什么模型上线需要 A/B 测试。
2. 如何设计两个模型版本的对照实验。
3. 如何做用户级、会话级、请求级流量切分。
4. 如何设计质量、业务、成本、安全指标。
5. 如何判断实验结果是否可信。
6. 如何做灰度发布、监控和回滚。
7. 面试中如何回答“大模型如何上线验证”。

### 1. 问题设定

假设你有两个模型：

```text
Model A：线上旧模型
Model B：新微调模型
```

离线评测结果显示：

```text
Model B 人工评测胜率更高
Model B 幻觉率略低
Model B 在长问题上回答更完整
```

但同时也发现：

```text
Model B 平均输出更长
Model B 推理成本更高
Model B 某些问题上拒答略多
```

现在业务方问：

```text
能不能上线？
上线后用户满意度会不会提升？
成本会不会超预算？
安全风险会不会增加？
```

A/B 测试就是回答这些问题的工程手段。

### 2. A/B 测试的基本思想

A/B 测试把真实流量随机分成两组：

```text
Control 组：使用旧模型 A
Treatment 组：使用新模型 B
```

然后比较两组在关键指标上的差异。

```text
用户满意度
问题解决率
人工转接率
二次追问率
负反馈率
响应延迟
token 成本
安全事件率
```

如果 Treatment 组在核心指标上显著更好，且没有触发安全、成本、稳定性红线，就可以逐步扩大流量。

### 3. A/B 测试前置条件

不是所有新模型都应该直接上 A/B。

上线实验前至少要通过：

```text
离线自动评测
人工评测
核心 bad case 回归
安全红线测试
格式和工具调用测试
延迟与成本压测
灰度回滚方案评审
```

如果模型在离线阶段已经有明显事实错误、安全风险或格式失败，就不要拿真实用户流量试错。

### 4. 实验单元怎么选

A/B 测试首先要确定随机化单元。

常见选择有三种。

#### 4.1 请求级随机

每个请求随机分到 A 或 B。

优点：

1. 实现简单。
2. 样本量增长快。
3. 适合单轮独立问答。

缺点：

1. 多轮对话中体验不一致。
2. 同一用户可能一会儿用 A，一会儿用 B。
3. 容易引入上下文污染。

#### 4.2 会话级随机

同一个会话固定使用同一个模型。

优点：

1. 多轮对话体验一致。
2. 适合客服、Agent、RAG 多轮问答。
3. 可以评估完整会话解决率。

缺点：

1. 样本量比请求级增长慢。
2. 需要稳定维护 session_id。

#### 4.3 用户级随机

同一个用户长期固定使用同一个模型。

优点：

1. 避免用户跨组污染。
2. 适合长期留存、复购、活跃度等指标。
3. 适合个性化产品。

缺点：

1. 实验周期更长。
2. 用户差异大，需要更大样本量。
3. 新老用户分布要控制。

大模型对话产品通常优先考虑会话级或用户级随机，而不是请求级随机。

### 5. 流量切分比例

新模型不要一开始就吃 50% 流量。

更稳妥的做法是逐步灰度：

```text
1% -> 5% -> 10% -> 25% -> 50% -> 100%
```

每一阶段都要观察：

```text
错误率
延迟
成本
安全告警
用户负反馈
业务核心指标
```

如果任何红线指标异常，立即停止扩量或回滚。

### 6. 指标体系设计

A/B 测试不能只看一个指标。

通常需要四类指标。

#### 6.1 主指标

主指标决定实验是否成功。

例如客服问答场景：

```text
问题解决率
用户满意率
人工转接率下降
负反馈率下降
```

主指标必须在实验开始前确定，不能实验结束后挑一个好看的指标。

#### 6.2 护栏指标

护栏指标用于防止模型虽然提升某个业务指标，但带来不可接受风险。

常见护栏指标：

```text
安全违规率
幻觉投诉率
平均延迟
P95 延迟
请求失败率
token 成本
过度拒答率
格式失败率
```

护栏指标不是越高越好，而是不能超过阈值。

#### 6.3 诊断指标

诊断指标用于解释结果。

例如：

```text
平均输出长度
检索命中率
工具调用成功率
用户二次追问率
会话轮数
不同任务类型胜率
不同 query 长度表现
```

诊断指标不一定直接决定上线，但能帮助理解为什么 B 好或不好。

#### 6.4 人工抽检指标

线上 A/B 期间仍然要做人审抽检。

抽检重点包括：

```text
低置信度样本
高风险领域样本
用户负反馈样本
长上下文样本
高价值用户样本
模型 A/B 分歧大的样本
```

自动指标看不见的问题，往往在人工抽检里暴露。

### 7. 一个指标配置示例

客服问答模型可以这样设计：

```text
主指标：
- 用户满意率提升至少 2%
- 人工转接率下降至少 1%

护栏指标：
- 安全违规率不得上升
- P95 延迟不得增加超过 15%
- 单次请求平均 token 成本不得增加超过 20%
- 负反馈率不得上升

诊断指标：
- 平均回答长度
- 二次追问率
- 会话解决率
- RAG 证据引用率
- 工具调用成功率

人工抽检：
- 每日抽检 200 条
- 覆盖投诉样本、低分样本和高风险样本
```

### 8. 实验分桶实现

一个简单的用户级分桶可以用 hash 实现。

```python
import hashlib


def assign_bucket(user_id: str, experiment_name: str, treatment_ratio: float) -> str:
    key = f"{experiment_name}:{user_id}".encode("utf-8")
    value = int(hashlib.md5(key).hexdigest(), 16) % 10_000
    bucket = value / 10_000
    return "treatment" if bucket < treatment_ratio else "control"


for uid in ["u001", "u002", "u003"]:
    print(uid, assign_bucket(uid, "llm_v2_ab", 0.1))
```

这里有几个工程点：

1. 同一个用户在同一个实验中分桶稳定。
2. `experiment_name` 可以避免不同实验相互影响。
3. `treatment_ratio` 可以从 0.01 逐步扩大到 0.5。

如果是会话级实验，可以把 `user_id` 换成 `session_id`。

### 9. 日志必须记录什么

A/B 实验的日志非常重要。

至少要记录：

```text
request_id
user_id 或 session_id
experiment_name
bucket
model_version
prompt_version
retriever_version
tool_version
input_query
retrieved_context_hash
output_text
latency_ms
input_tokens
output_tokens
cost
error_code
user_feedback
timestamp
```

大模型系统经常不是只有模型变化。

如果同时改了 prompt、检索器、工具调用逻辑，却只记录 model_version，后续很难解释实验结果。

### 10. 避免实验污染

实验污染会让结果失真。

常见污染包括：

```text
同一用户跨组
多轮对话中模型切换
客服人工介入策略不同
缓存命中导致实际模型不一致
不同组使用不同 prompt 或检索版本
流量来源不均衡
实验期间业务活动影响某一组
```

解决方法：

```text
固定随机化单元
记录完整版本信息
按用户、渠道、任务类型做分层检查
实验期间避免同时上线其他强相关改动
缓存 key 中包含实验桶和模型版本
```

### 11. 显著性和样本量

如果 Treatment 组满意率从 80% 变成 81%，这到底是真提升，还是随机波动？

这就需要统计显著性。

面试中不一定要求你推公式，但你要知道：

```text
样本量越小，结果越不稳定。
指标波动越大，需要样本越多。
预期提升越小，需要样本越多。
实验结束前频繁偷看并提前停止，容易产生假阳性。
```

一个工程上常见做法是：

```text
实验前估算样本量
固定实验周期
达到最小样本量后再判断
对核心指标做置信区间或显著性检验
结合业务意义判断，而不是只看 p-value
```

### 12. 简单计算胜率置信区间

假设 pairwise 人工抽检中，新模型 B 赢了 580 条，输了 310 条，持平 110 条。

可以粗略计算有效胜率：

```python
import math


def wilson_interval(wins: int, losses: int, z: float = 1.96):
    n = wins + losses
    if n == 0:
        return 0.0, 0.0, 0.0

    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, center - margin, center + margin


p, low, high = wilson_interval(580, 310)
print(round(p, 4), round(low, 4), round(high, 4))
```

如果置信区间整体高于 0.5，说明 B 的偏好胜率比较稳。

但注意：人工 pairwise 胜率只是一个维度，仍然要看线上业务指标和护栏指标。

### 13. 大模型 A/B 的特殊问题

传统推荐、广告、按钮颜色实验的 A/B 方法不能直接照搬到大模型。

大模型有一些特殊问题。

#### 13.1 输出非确定性

同一个 prompt 在不同温度、不同采样种子下可能输出不同答案。

实验时要固定或记录：

```text
temperature
top_p
max_tokens
random_seed
decoding_strategy
```

否则模型差异和采样差异会混在一起。

#### 13.2 成本差异明显

新模型可能回答更完整，但 token 更多。

如果满意度提升 1%，成本增加 80%，不一定值得上线。

所以成本是大模型 A/B 的核心护栏。

#### 13.3 延迟影响用户行为

模型质量更好但响应慢，用户可能提前关闭页面。

必须同时看：

```text
平均延迟
P90 延迟
P95 延迟
P99 延迟
首 token 延迟
完整响应时间
```

#### 13.4 安全事件低频但高风险

安全问题可能发生率很低，但影响很大。

不能因为 A/B 样本里没出现安全事件，就认为安全性没问题。

需要结合：

```text
离线红队测试
线上安全分类器
人工高风险抽检
用户举报通道
```

#### 13.5 用户反馈有偏

不是所有用户都会点赞或点踩。

点踩用户通常更主动，点赞用户可能沉默。

所以用户反馈要和行为指标、人工抽检结合看。

### 14. 灰度发布策略

推荐的发布流程是：

```text
0. 离线评测通过
1. 内部 dogfood
2. 1% 低风险用户灰度
3. 5% 全量场景灰度
4. 10%-25% 扩量
5. 50% A/B 对照
6. 达到上线标准后逐步全量
```

每个阶段都要定义进入下一阶段的条件。

例如：

```text
安全违规率未上升
P95 延迟未超过阈值
成本未超过预算
负反馈率未上升
主指标有正向趋势
人工抽检无严重 bad case
```

### 15. 回滚策略

没有回滚策略，就不要上线新模型。

回滚策略至少包括：

```text
一键切回旧模型
按用户/渠道/场景关闭新模型
关闭高风险功能
降低 max_tokens 控制成本
切换到保守 prompt
暂停工具调用或外部动作
触发人工审核
```

触发回滚的条件也要提前定义。

例如：

```text
安全违规率超过阈值
错误率连续 10 分钟异常
P95 延迟超过阈值
成本突增
用户投诉集中爆发
核心业务指标显著下降
```

### 16. A/B 结果怎么解释

实验结束后，不要只说：

```text
B 比 A 好。
```

应该给出结构化结论：

```text
实验范围：5% 用户级流量，持续 7 天
样本量：Control 组 12 万会话，Treatment 组 12 万会话
主指标：问题解决率 +2.3%，置信区间 [+1.1%, +3.5%]
护栏指标：安全违规率无显著变化，P95 延迟 +8%，成本 +12%
诊断指标：长问题场景提升明显，短问题场景基本持平
人工抽检：B 胜率 57%，主要提升来自完整性和证据一致性
风险：多轮对话中仍有少量过度解释
结论：建议扩大到 25% 灰度，并继续监控成本和长对话质量
```

这样的结果能支撑工程决策。

### 17. 什么时候不能上线

即使主指标提升，也不一定能上线。

以下情况应暂缓：

```text
安全风险上升
事实错误率上升
关键场景退化
成本不可接受
延迟明显变差
用户投诉集中
实验样本不足
实验污染严重
只有平均指标提升，但核心人群下降
```

大模型上线要避免“总体平均好，但关键切片崩”。

### 18. 面试高频问法

#### 问法 1：一个新微调模型离线评测更好，你会如何上线验证？

可以这样答：

```text
我不会直接全量上线。首先会确认离线自动评测、人工评测、bad case 回归、安全测试、延迟和成本压测都通过。然后做灰度 A/B，从小流量开始，比如 1% 或 5%。随机化单元会根据产品形态选择，如果是多轮对话，我倾向会话级或用户级分桶，而不是请求级随机。

指标上会提前定义主指标、护栏指标和诊断指标。主指标可能是用户满意率、问题解决率或人工转接率；护栏指标包括安全违规率、幻觉投诉率、P95 延迟、成本、请求失败率和负反馈率。实验结束后看显著性、置信区间和切片表现，而不是只看平均值。如果主指标显著提升且护栏指标没有恶化，再逐步扩大灰度，并保留回滚方案。
```

#### 问法 2：大模型 A/B 和普通推荐 A/B 有什么不同？

可以这样答：

```text
大模型 A/B 的特殊点在于输出是非确定性的，而且质量维度更复杂。除了点击、转化这类业务指标，还要看事实正确性、幻觉、安全、格式、指令遵循、延迟和 token 成本。多轮对话还要求同一会话保持同一个模型，否则体验会被污染。另外安全事件可能低频但高风险，所以需要结合离线红队、线上安全监控和人工抽检，而不能只依赖线上平均指标。
```

#### 问法 3：A/B 结果整体提升，但某些场景下降怎么办？

可以这样答：

```text
我会先做切片分析，确认下降发生在哪些任务类型、用户群体、query 长度、语言、渠道或高风险场景。如果下降的是低价值非核心场景，可以考虑继续灰度并定向修复；如果下降的是核心场景、安全场景或高价值用户场景，即使总体平均提升也不应该全量上线。可以采用分场景路由，让新模型只服务收益明确的场景，退化场景继续使用旧模型。
```

#### 问法 4：如何避免 A/B 实验结果不可信？

可以这样答：

```text
首先要在实验前定义主指标和护栏指标，避免事后挑指标。其次要保证随机化单元稳定，比如用户级或会话级分桶，避免同一用户跨组。还要记录完整版本信息，包括模型、prompt、检索器、工具和采样参数，防止多个变量混在一起。实验期间尽量避免同时上线强相关改动。最后要达到最小样本量和固定实验周期，再看显著性、置信区间和切片结果。
```

### 19. 本讲小结

这一讲讨论了如何 A/B 测试两个模型版本。

核心结论如下：

1. 离线评测通过不等于可以直接全量上线，真实用户流量需要 A/B 验证。
2. A/B 测试要明确 control 组和 treatment 组，并选择合适随机化单元。
3. 大模型对话场景通常优先使用会话级或用户级分桶，避免多轮体验污染。
4. 新模型应逐步灰度，不要一开始吃大流量。
5. 指标体系要包含主指标、护栏指标、诊断指标和人工抽检指标。
6. 安全、事实、延迟、成本是大模型上线的重要护栏。
7. 实验日志必须记录模型、prompt、检索器、工具、采样参数和实验桶。
8. 结果解释要看显著性、置信区间、切片表现和 bad cases，而不是只看平均分。
9. 即使总体指标提升，只要核心场景或安全指标退化，也不能直接全量。
10. 上线必须配套灰度发布、监控告警和快速回滚方案。

至此，第三册第八部分“评估与 Debug 实战”正文第一版完成。
