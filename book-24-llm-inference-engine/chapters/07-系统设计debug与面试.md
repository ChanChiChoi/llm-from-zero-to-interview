# 第 7 章 实现最小 tokenizer、model wrapper 和 generate loop

从本章开始，我们进入第二部分：从 0 实现一个最小推理框架。目标不是立刻复刻 vLLM 或 SGLang，而是先把最小链路跑通：tokenizer 把文本转成 token，model wrapper 调用模型前向，generate loop 一步步生成 token。

如果你连最小 generate loop 都没拆开过，直接看 PagedAttention、continuous batching、RadixAttention，很容易只记住名词。本章先把最底层的生成过程写清楚。

一句话概括：

> 最小推理框架的第一步，是把“输入文本到输出文本”拆成 tokenizer、model wrapper、sampling 和 generate loop 四个可控环节。

## 7.0 本讲资料边界与第二轮精修口径

本章第二轮精修前，先用公开资料校准口径：Hugging Face Transformers 的生成接口把 `generate()`、logits 处理、stopping criteria、cache 和不同 decoding 策略封装成高层 API；Hugging Face tokenizer 文档强调 encode / decode、special tokens、padding / truncation 和 chat template 等输入输出契约；PyTorch 的 `eval()` 与 `inference_mode()` 分别对应模块推理状态和关闭 autograd 开销。

因此，本章不复刻 Transformers 源码，也不要求读者安装大模型权重才能理解机制。正文里的 PyTorch / Transformers 代码是工程对照，新增的 0 依赖 demo 用 toy tokenizer 和 toy model wrapper 展示最小生成链路。读者需要先掌握：文本如何变成 token id，模型如何把当前 token 序列变成最后位置 logits，generate loop 如何选择 token、追加 token、检查停止条件并留下 trace。

## 7.1 本章目标

本章要实现的是教学版最小链路，不追求性能。

它应该能完成：

1. 加载 tokenizer。
2. 加载模型。
3. 把 prompt 转成 token ids。
4. 调用模型得到 logits。
5. 从 logits 里选出下一个 token。
6. 把新 token 追加到序列。
7. 重复直到结束。
8. 把 token ids 解码成文本。

暂时不做：

1. batch。
2. KV Cache。
3. streaming。
4. scheduler。
5. HTTP API。
6. 多请求并发。

这些能力会在后续章节逐步加入。

## 7.2 最小组件拆分

最小推理框架可以先拆成三层：

```text
TokenizerWrapper
  encode(text) -> input_ids
  decode(token_ids) -> text

ModelWrapper
  forward(input_ids) -> logits

GenerateLoop
  encode -> forward -> select token -> append -> decode
```

为什么要拆 wrapper？

因为我们后续会不断替换内部实现：

1. tokenizer 以后要支持 chat template。
2. model wrapper 以后要支持 KV Cache。
3. generate loop 以后要支持 batch 和 scheduler。
4. sampling 以后要支持 top-k、top-p、temperature。

如果一开始所有逻辑都塞进一个函数，后面升级会很痛苦。

## 7.3 TokenizerWrapper

Tokenizer 的职责是文本和 token 之间互转。

教学版 wrapper 可以这样写：

```python
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def encode(self, text):
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded.input_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
```

这里有几个注意点。

第一，`input_ids` 是 token id 序列，不是原始字符串。

第二，`eos_token_id` 用来判断生成是否结束。

第三，真实聊天模型还需要 chat template。也就是把 messages 转成模型训练时熟悉的格式。

第四，decode 时要处理 special tokens，否则输出里可能出现模型内部控制符。

## 7.4 ModelWrapper

ModelWrapper 的职责是隐藏具体模型调用细节，对外提供统一 forward 接口。

教学版可以先这样写：

```python
class ModelWrapper:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids)
        return outputs.logits
```

几个关键点：

1. 推理时要用 `eval()`。
2. 推理时要关闭梯度。
3. 输入要放到模型所在设备。
4. 输出 logits 的形状通常是 `[batch, seq_len, vocab_size]`。

本章暂时不使用 KV Cache，所以每轮 forward 都会重新计算完整序列。这很低效，但便于理解。

## 7.5 Logits 是什么

模型 forward 的输出不是直接文本，而是 logits。

对于每个位置，logits 表示词表中每个 token 的未归一化分数。

形状示例：

```text
input_ids shape: [1, seq_len]
logits shape:    [1, seq_len, vocab_size]
```

生成下一个 token 时，我们只关心最后一个位置的 logits：

```python
next_token_logits = logits[:, -1, :]
```

因为自回归模型的含义是：基于前面所有 token，预测下一个 token。

如果 prompt 是：

```text
大模型推理的核心是
```

最后一个位置的 logits 就是在回答：“下一个 token 最可能是什么？”

## 7.6 最简单的 greedy 选择

本章先用 greedy decoding，也就是每次选择分数最高的 token。

```python
def greedy_select(logits):
    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    return next_token_id
```

greedy 的优点：

1. 简单。
2. 确定性强。
3. 适合教学和 debug。

greedy 的缺点：

1. 输出可能重复。
2. 多样性较差。
3. 不适合所有生成任务。

下一章会加入 temperature、top-k 和 top-p。

## 7.7 最小 generate loop

现在可以写最小生成循环：

```python
def generate(tokenizer_wrapper, model_wrapper, prompt, max_new_tokens=128):
    input_ids = tokenizer_wrapper.encode(prompt)
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        logits = model_wrapper.forward(generated_ids)
        next_token_id = greedy_select(logits)

        generated_ids = torch.cat([generated_ids, next_token_id.cpu()], dim=-1)

        if next_token_id.item() == tokenizer_wrapper.eos_token_id:
            break

    return tokenizer_wrapper.decode(generated_ids[0])
```

这就是最小 generate loop。

它的核心逻辑是：

```text
已有 tokens -> 模型预测下一个 token -> 追加 token -> 再预测下一个 token
```

注意，这段代码把 `next_token_id` 移回 CPU 后再拼接，是为了保持示例简单。真实实现里会尽量减少 CPU/GPU 来回拷贝。

## 7.8 更合理的设备处理

上面的示例为了易懂，设备处理略粗糙。更合理的写法是让 token 始终在模型设备上，最后再 decode。

```python
def generate(tokenizer_wrapper, model_wrapper, prompt, max_new_tokens=128):
    input_ids = tokenizer_wrapper.encode(prompt).to(model_wrapper.device)
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        logits = model_wrapper.forward(generated_ids)
        next_token_id = greedy_select(logits)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer_wrapper.eos_token_id:
            break

    return tokenizer_wrapper.decode(generated_ids[0].tolist())
```

教学版代码要尽量清晰，但也要让读者知道真实系统会避免不必要的数据搬运。

## 7.9 停止条件

最小停止条件有两个：

1. 达到 `max_new_tokens`。
2. 生成 EOS token。

真实系统还会支持：

1. stop words。
2. stop token ids。
3. 请求超时。
4. 用户取消。
5. 最大上下文长度。
6. 安全策略中止。
7. engine 内部错误。

停止条件看起来简单，但在 streaming 和 batch 场景会变复杂。

例如 batch 中有 8 个请求，其中 3 个已经 EOS，另外 5 个还在生成。框架要把已结束请求移出 running set，并释放它们的 KV Cache，而不是等所有请求一起结束。

## 7.10 为什么这个版本很慢

本章的 generate loop 有一个明显问题：每生成一个 token，都把完整历史序列重新送进模型。

如果 prompt 长度是 1000，生成 100 个 token，计算大致像这样：

```text
第 1 轮：处理 1000 tokens
第 2 轮：处理 1001 tokens
第 3 轮：处理 1002 tokens
...
第 100 轮：处理 1099 tokens
```

大量历史 token 被重复计算。

这就是下一步要引入 KV Cache 的原因。KV Cache 会让 decode 阶段只处理新 token，并复用历史 key/value。

但在引入 KV Cache 之前，先写清楚 naive loop 是必要的。因为 KV Cache 优化的对象，正是这个重复计算问题。

## 7.11 和 transformers.generate 的关系

Hugging Face 的 `model.generate()` 已经封装了大量功能：

1. greedy decoding。
2. sampling。
3. beam search。
4. KV Cache。
5. stopping criteria。
6. logits processor。
7. batch generation。

我们本章不是为了替代它，而是为了拆开它。

学习 serving engine 时，不能永远把 `generate()` 当黑盒。你需要知道黑盒里至少有：

1. forward。
2. logits 处理。
3. token 选择。
4. cache 更新。
5. 停止判断。
6. 序列管理。

只有拆开之后，才能继续理解 scheduler 和 continuous batching。

## 7.12 最小代码骨架

把本章组件合起来，代码骨架如下：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class ModelWrapper:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits


def greedy_select(logits):
    next_token_logits = logits[:, -1, :]
    return torch.argmax(next_token_logits, dim=-1, keepdim=True)


def generate(tokenizer_wrapper, model_wrapper, prompt, max_new_tokens=128):
    generated_ids = tokenizer_wrapper.encode(prompt).to(model_wrapper.device)

    for _ in range(max_new_tokens):
        logits = model_wrapper.forward(generated_ids)
        next_token_id = greedy_select(logits)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer_wrapper.eos_token_id:
            break

    return tokenizer_wrapper.decode(generated_ids[0].tolist())


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer_wrapper = TokenizerWrapper(tokenizer)
model_wrapper = ModelWrapper(model, device="cuda" if torch.cuda.is_available() else "cpu")

print(generate(tokenizer_wrapper, model_wrapper, "LLM inference is", max_new_tokens=32))
```

这段代码不是生产框架，但它已经具备推理框架的雏形。

## 7.13 这个骨架如何升级

后续章节会逐步升级它。

第 8 章会替换 `greedy_select`，加入 sampling。

第 9 章会修改 `ModelWrapper.forward`，加入 KV Cache。

第 10 章会让 prefill 支持 batch。

第 11 章会让 decode 支持 batch。

第 12 章会引入 request queue 和 scheduler。

第 13 章会把完整输出改成 token streaming。

第 14 章会加 HTTP API。

第 15 章会压测 TTFT、TPOT、吞吐和显存。

也就是说，本章代码是最小种子，后面每章都在它上面补一块真实 serving engine 能力。

## 7.14 生成循环公式、契约和可运行 demo

自回归生成可以写成：

```math
P(y_1,\ldots,y_M\mid x)=\prod_{t=1}^{M}P(y_t\mid x,y_1,\ldots,y_{t-1})
```

其中，$x$ 是输入 prompt，$y_t$ 是第 $t$ 个新生成 token。直觉上，模型不是一次写完整段文本，而是每一步根据已有 token 预测下一个 token。

tokenizer 契约可以写成：

```math
s_0=\mathrm{encode}(x),\qquad \hat{x}=\mathrm{decode}(s_0)
```

$s_0$ 是 prompt 的 token id 序列。真实 tokenizer 的 decode 可能不完全等于原字符串，因为空格、大小写、特殊 token 和规范化规则都会影响结果；但 wrapper 至少要保证 encode / decode 的输入输出类型清楚。

第 $t$ 轮模型前向输出最后位置 logits：

```math
z_t=f_{\theta}(s_{t-1})_{-1},\qquad z_t\in\mathbb{R}^{V}
```

$V$ 是词表大小，$z_t(v)$ 是候选 token $v$ 的未归一化分数。greedy decoding 可以写成：

```math
y_t=\mathrm{argmax}_{v\in\mathcal{V}} z_t(v),\qquad s_t=[s_{t-1},y_t]
```

停止条件可以抽象为：

```math
G_{\mathrm{stop}}=\mathbf{1}[y_t=y_{\mathrm{eos}}\ \mathrm{or}\ t=M_{\max}]
```

naive generate loop 的重复计算量可以粗略写成：

```math
W_{\mathrm{naive}}=\sum_{t=0}^{M-1}(N+t)
```

其中 $N$ 是 prompt token 数，$M$ 是生成 token 数。这个式子解释了为什么本章版本会慢：每一轮都把历史 token 重新送进模型。后续 KV Cache 会把主要重复计算变成“prefill 处理 prompt，decode 每轮处理新增 token”。

下面是一个完全不依赖 PyTorch 和 Transformers 的最小 demo。它用 toy tokenizer、toy model wrapper 和 greedy generate loop 模拟“LLM inference -> is -> a -> loop -> . -> EOS”的生成过程。

```python
class ToyTokenizer:
    def __init__(self):
        self.id_to_token = ["<eos>", "<unk>", "LLM", "inference", "is", "a", "loop", "."]
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.eos_token_id = self.token_to_id["<eos>"]
        self.unk_token_id = self.token_to_id["<unk>"]

    def encode(self, text):
        return [self.token_to_id.get(piece, self.unk_token_id) for piece in text.split()]

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if skip_special_tokens and token.startswith("<"):
                continue
            pieces.append(token)
        return " ".join(pieces)


class ToyModelWrapper:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.transition = {
            2: 3,  # LLM -> inference
            3: 4,  # inference -> is
            4: 5,  # is -> a
            5: 6,  # a -> loop
            6: 7,  # loop -> .
            7: 0,  # . -> <eos>
        }

    def forward(self, input_ids):
        last_token = input_ids[-1]
        next_token = self.transition.get(last_token, 0)
        logits = [-10.0] * self.vocab_size
        logits[next_token] = 10.0
        return logits


def greedy_select(logits):
    best_id = 0
    best_score = logits[0]
    for token_id, score in enumerate(logits):
        if score > best_score:
            best_id = token_id
            best_score = score
    return best_id


def generate(tokenizer, model, prompt, max_new_tokens=8):
    generated_ids = tokenizer.encode(prompt)
    trace = []
    naive_token_work = 0

    for step in range(1, max_new_tokens + 1):
        naive_token_work += len(generated_ids)
        logits = model.forward(generated_ids)
        next_token_id = greedy_select(logits)
        generated_ids.append(next_token_id)

        trace.append(
            {
                "step": step,
                "input_len": len(generated_ids) - 1,
                "next_token": tokenizer.id_to_token[next_token_id],
                "stop": next_token_id == tokenizer.eos_token_id,
            }
        )

        if next_token_id == tokenizer.eos_token_id:
            break

    generated_new_tokens = len(generated_ids) - len(tokenizer.encode(prompt))
    kv_like_work = len(tokenizer.encode(prompt)) + generated_new_tokens

    metrics = {
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "generated_tokens": generated_new_tokens,
        "forward_calls": len(trace),
        "naive_token_work": naive_token_work,
        "kv_like_work": kv_like_work,
        "duplicate_work": naive_token_work - kv_like_work,
    }

    audit = {
        "tokenizer_contract": tokenizer.decode(tokenizer.encode(prompt)) == prompt,
        "model_wrapper_contract": len(model.forward(tokenizer.encode(prompt))) == model.vocab_size,
        "generate_loop_trace": len(trace) == generated_new_tokens,
        "stop_condition": trace[-1]["stop"],
        "naive_recompute_visible": metrics["duplicate_work"] > 0,
    }
    audit["minimal_generate_gate"] = all(audit.values())

    return tokenizer.decode(generated_ids), generated_ids, trace, metrics, audit


tokenizer = ToyTokenizer()
model = ToyModelWrapper(vocab_size=len(tokenizer.id_to_token))
text, token_ids, trace, metrics, audit = generate(tokenizer, model, "LLM inference")

print("generated_text=", text)
print("token_ids=", token_ids)
print("trace=", trace)
print("metrics=", metrics)
print("audit=", audit)
```

一组典型输出如下：

```text
generated_text= LLM inference is a loop .
token_ids= [2, 3, 4, 5, 6, 7, 0]
trace= [{'step': 1, 'input_len': 2, 'next_token': 'is', 'stop': False}, {'step': 2, 'input_len': 3, 'next_token': 'a', 'stop': False}, {'step': 3, 'input_len': 4, 'next_token': 'loop', 'stop': False}, {'step': 4, 'input_len': 5, 'next_token': '.', 'stop': False}, {'step': 5, 'input_len': 6, 'next_token': '<eos>', 'stop': True}]
metrics= {'prompt_tokens': 2, 'generated_tokens': 5, 'forward_calls': 5, 'naive_token_work': 20, 'kv_like_work': 7, 'duplicate_work': 13}
audit= {'tokenizer_contract': True, 'model_wrapper_contract': True, 'generate_loop_trace': True, 'stop_condition': True, 'naive_recompute_visible': True, 'minimal_generate_gate': True}
```

这个 demo 说明四件事：

1. tokenizer wrapper 要有明确的 encode / decode 契约。
2. model wrapper 对外只暴露“输入 token ids，输出最后位置 logits”。
3. generate loop 的本质是 forward、select、append、stop。
4. naive loop 的重复计算能被量化，后续 KV Cache 不是魔法，而是在减少 `duplicate_work`。

## 7.15 常见误区

误区一：会调用 `model.generate()` 就等于懂生成循环。

`generate()` 是封装接口。理解 serving engine 需要知道 forward、logits、token selection、cache 和 stopping 是如何配合的。

误区二：logits 就是概率。

logits 是未归一化分数，经过 softmax 后才是概率分布。

误区三：每轮都重新计算完整序列也没关系。

对短 demo 可以接受；对长上下文和高并发服务不可接受，所以必须引入 KV Cache。

误区四：tokenizer 不重要。

tokenizer 决定 token 数、chat template、特殊 token 和停止条件，直接影响推理成本和结果格式。

误区五：教学代码不需要考虑设备。

即使是教学代码，也要知道 CPU/GPU 数据搬运会影响性能，真实系统要尽量减少不必要拷贝。

## 7.16 面试追问

1. 最小 generate loop 包含哪些步骤？
2. logits 的形状是什么，为什么只取最后一个位置？
3. greedy decoding 是什么，有什么优缺点？
4. 为什么 naive generate loop 很慢？
5. tokenizer 在推理系统中负责什么？
6. `eval()` 和 `torch.inference_mode()` 为什么重要？
7. `model.generate()` 和手写 generate loop 的关系是什么？
8. 这个最小框架如何升级到支持 KV Cache？

参考回答思路：

1. 先说 encode、forward、取最后 logits、选 token、append、停止判断、decode。
2. 再解释 logits 是 `[batch, seq_len, vocab_size]`，最后位置预测下一个 token。
3. 然后指出 naive loop 重复计算历史，所以需要 KV Cache。
4. 最后说明后续要加入 sampling、batch、scheduler 和 streaming。

## 7.17 小练习

1. 手写一个 greedy generate loop，不调用 `model.generate()`。
2. 打印每轮 `generated_ids.shape`，观察序列长度如何增长。
3. 打印 `logits.shape`，确认最后一维是词表大小。
4. 把 `max_new_tokens` 改成不同值，观察输出和耗时变化。
5. 思考如果每轮只输入最后一个 token，为什么还需要 KV Cache？
6. 扩展 `ToyModelWrapper`，让它在不同 prompt 下走不同 transition，并观察 trace 如何帮助定位生成错误。

## 7.18 本章小结

本章实现了最小 tokenizer、model wrapper 和 generate loop。

我们把生成过程拆成了文本编码、模型 forward、logits 选择、token 追加、停止判断和文本解码。这个版本故意不使用 KV Cache、不支持 batch、不支持 streaming，因此性能很差，但结构清晰。

下一章会在这个基础上加入 greedy、top-k、top-p 和 temperature sampling，让 token 选择从“永远取最大值”升级成可控采样策略。
