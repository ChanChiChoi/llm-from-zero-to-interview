# 第 6 章（第五篇）：Speculative Decoding——先提议，再验证

量化主要改变权重或缓存的数值表示；推测解码改变的是生成算法。目标模型仍决定最终分布，小模型或 draft head 先提出若干候选，目标模型再并行核验。它只有在少做的目标模型串行计算足以抵消 draft、验证和调度开销时才有价值。

本篇先区分贪心生成和采样生成的正确性要求，再推导接受率与每轮成本，最后给出怎样比较单请求收益与服务级吞吐的方法。读者会得到一个“是否值得启用”的测量清单，而不是只记住一个接受率数字。

## 6.5 Speculative Decoding：让小模型先提议，大模型批量验证

### 6.5.1 一次核验为什么可能接受多个 token

目标模型通常很大，生成一个 token 就要运行一次昂贵的 decode。另一个更小的 draft model 可以先连续生成若干 token；目标模型把这段候选作为一个小序列一次性计算并验证。如果候选大部分正确，就能用一次 target forward 接受多个 token。

```text
已有前缀
  ↓
draft：提出 d1, d2, d3, d4
  ↓
target：并行计算这些位置的概率
  ↓
接受共同前缀，例如 d1, d2
  ↓
在第一个不一致位置使用 target 的结果继续
```

收益来自两个事实：

1. 小模型生成多个 token 的成本低于目标模型逐 token 生成；
2. target 可以在更适合并行的形状上同时验证多个候选位置。

如果 draft 与 target 差异太大，接受率低，额外的 draft 计算、候选整理和验证 kernel 可能抵消收益。

### 6.5.2 Greedy 验证的教学版

对于贪心解码，可以用较直观的规则说明：

```math
\hat y_i^{\mathrm{draft}}
=
\arg\max_v q_i(v)
```

```math
\hat y_i^{\mathrm{target}}
=
\arg\max_v p_i(v)
```

如果从当前前缀开始的前 $k$ 个候选满足：

```math
\hat y_i^{\mathrm{draft}}
=
\hat y_i^{\mathrm{target}},
\qquad i=1,\ldots,k
```

就可以接受这段共同前缀；遇到第一个不一致位置时，输出 target 的 token，而不是 draft 的 token。

这只是机制演示。真实 target 会用因果 mask 和 cache 计算每个候选位置的 logits，处理 EOS、stop、logits processor 和 batch 元数据；不能把对每个 token 单独调用 target 的 Python 循环当成高性能实现。

### 6.5.3 采样时的分布校正

如果 draft token $x$ 来自分布 $q(x)$，target 分布为 $p(x)$，简单地“相等就接受、不相等就换成 target argmax”不能保证最终样本仍来自 $p$。经典 speculative sampling 使用接受概率：

```math
a(x)
=
\min\left(1,\frac{p(x)}{q(x)}\right)
```

若拒绝，需要从 residual distribution 采样：

```math
r(x)
=
\frac{\max(p(x)-q(x),0)}
{\sum_v\max(p(v)-q(v),0)}
```

这里的 $p$ 和 $q$ 都是在同一个有限词表上的非负归一化分布；只有当 draft 实际采样的 token 满足 $q(x)>0$ 时，接受概率中的比值才有定义。residual 分母记为 $D$。当 $D=0$ 时，说明这一步没有需要补偿的质量差异，应直接从共同的 target 分布采样，不能执行除以零；当 $D>0$ 时才按上式归一化 residual。实现还应检查概率有限、非负且归一化误差在容忍范围内。

这套规则需要在每一步正确维护 draft/target 的条件分布、随机数和已接受前缀。temperature、top-k、top-p、repetition penalty、bad words、grammar constraint 等处理器必须在两个模型的分布计算中保持一致或采用 runtime 明确支持的方案。

### 6.5.4 接受率与期望收益

设一次 draft 提议长度为 $\gamma$，每个位置近似以概率 $a$ 被接受。接受的 draft token 数的一个简化期望是：

```math
\mathbb{E}[K]
\approx
\sum_{i=1}^{\gamma}a^i
```

验证步骤通常还能产生一个 target token，因此一次循环的输出 token 数可以粗略表示为：

```math
\mathbb{E}[N_{\mathrm{cycle}}]
\approx
1+\sum_{i=1}^{\gamma}a^i
```

若 draft 生成 $\gamma$ 个 token 的成本为 $C_d(\gamma)$，target 验证一段候选的成本为 $C_t(\gamma+1)$，则每循环的近似时间为：

```math
T_{\mathrm{cycle}}
\approx
C_d(\gamma)+C_t(\gamma+1)+C_{\mathrm{overhead}}
```

相对普通 target 单 token decode 的速度比可以粗略写成：

```math
S
\approx
\frac{
\mathbb{E}[N_{\mathrm{cycle}}]/T_{\mathrm{cycle}}
}{
1/C_t(1)
}
```

这个式子不是 runtime 的精确性能模型。$a$ 会随任务、语言、temperature、位置和输出阶段变化；target 的批量验证成本也不等于 $(\gamma+1)C_t(1)$。它的作用是提醒我们必须同时测接受长度、draft 成本、target 验证成本和调度开销。

例如设 $\gamma=4$，为方便手算，假设每个候选位置独立地以概率 $a=0.8$ 被接受，则每轮期望输出 $1+0.8+0.8^2+0.8^3+0.8^4=3.3616$ 个 token。假设 draft 用 16 ms 提出四个候选、target 批量验证用 25 ms，暂不计额外调度开销，那么一轮需 41 ms，约为 $41/3.3616=12.2$ ms/token。若普通 target decode 是 20 ms/token，理想化单请求速度约提升 1.64 倍。把接受率降到 0.4，期望输出变为 $1+0.4+0.16+0.064+0.0256=1.6496$ 个 token，同一轮成本约为 $41/1.6496=24.9$ ms/token，反而慢于普通 decode。这个算例忽略了接受率随位置变化和调度开销，只能说明成本与接受长度要一起测，不能预测真实 runtime 收益。

这里的 $0\leq a\leq1$、$\gamma$ 为正整数，且 $C_d(\gamma)$、$C_t(\gamma+1)$ 和普通 decode 的单位成本都必须为正数；否则“接受率”和速度比没有定义。

### 6.5.5 无依赖接受率 demo

下面用两个确定性 token 序列模拟 greedy 验证，输出接受前缀长度和循环数。它不表示真实模型的接受率。

> 代码状态：静态审阅，未运行（2026-10-29）
```python
def verify_greedy(draft_tokens, target_tokens):
    draft_tokens = list(draft_tokens)
    target_tokens = list(target_tokens)
    if len(target_tokens) < len(draft_tokens) + 1:
        raise ValueError("target must contain a token after every draft position")

    accepted = 0
    while accepted < len(draft_tokens):
        if draft_tokens[accepted] != target_tokens[accepted]:
            return accepted, target_tokens[accepted]
        accepted += 1
    return accepted, target_tokens[accepted]


cases = [
    ([], ["a"]),
    (["a", "b", "c"], ["a", "b", "c", "d"]),
    (["a", "x", "c"], ["a", "b", "c", "d"]),
    (["x", "b"], ["a", "b", "c"]),
]

for draft, target in cases:
    accepted, replacement = verify_greedy(draft, target)
    print(
        "draft=", draft,
        "accepted=", accepted,
        "replacement=", replacement,
    )
```
第一组的共同前缀较长，第二组在第二个位置发生分歧，第三组一开始就分歧。真实系统还要处理“target 验证到第一个分歧后是否已经算出了后继位置”的细节，因此接受 token 数和 target forward 次数要从 runtime trace 读取。

### 6.5.6 Transformers assisted generation 的工程位置

Transformers v5.17.0 提供 assisted generation 接口，常规 assistant-model 路线要求 target 与 draft 使用同一个 tokenizer，并且文档注明该路线不支持 batched inputs。下面用 Pythia-160M 与 Pythia-70M 的固定 Hub commit 展示单条贪心生成；[Pythia 项目说明](https://github.com/EleutherAI/pythia)该系列使用共享的 GPT-NeoX tokenizer。小模型对只演示接口和身份固定，不构成速度收益证据。

> 代码状态：静态审阅，未运行（2026-10-29）
```python
from transformers import AutoModelForCausalLM, AutoTokenizer


target_id = "EleutherAI/pythia-160m-deduped"
target_revision = "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e"
assistant_id = "EleutherAI/pythia-70m-deduped"
assistant_revision = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"

tokenizer = AutoTokenizer.from_pretrained(
    target_id,
    revision=target_revision,
)
target = AutoModelForCausalLM.from_pretrained(
    target_id,
    revision=target_revision,
)
assistant = AutoModelForCausalLM.from_pretrained(
    assistant_id,
    revision=assistant_revision,
)
inputs = tokenizer("Explain KV cache briefly.", return_tensors="pt")
outputs = target.generate(
    **inputs,
    assistant_model=assistant,
    max_new_tokens=64,
)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```
这段代码需要 Transformers v5.17.0 和 PyTorch；示例为单条 CPU/GPU 生成，模型需放在可相互配合的设备上。默认路线使用共同 tokenizer；若目标与 draft tokenizer 不同，v5.17.0 另提供 Universal Assisted Decoding (UAD)，通过候选重编码和最长公共子序列对齐处理 token 边界，但这会引入额外步骤，也不表示任意模型对都兼容。启用 UAD 时须同时传入 target 与 assistant tokenizer，并按该版本文档确认模型和解码设置。具体见[Transformers v5.17.0 assisted decoding](https://huggingface.co/docs/transformers/v5.17.0/assisted_decoding) 与[Generation API](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/text_generation)；Pythia 模型快照见[160M](https://huggingface.co/EleutherAI/pythia-160m-deduped/tree/582159a2dfe3e712a8d47ae83dec95ae3bde8e7e)和[70M](https://huggingface.co/EleutherAI/pythia-70m-deduped/tree/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c)。

### 6.5.7 Draft 来源不止一种

draft 可以来自不同路线：

```text
独立的小语言模型：
  部署简单，词表和分布兼容性需要检查。

同一模型的轻量副本：
  tokenizer 更容易一致，但仍可能存在结构或配置差异。

额外训练的 draft head：
  与 target 共享部分表示，减少独立模型成本。

多 token prediction 或相关辅助头：
  直接学习未来多个位置的候选，验证机制和训练目标更紧密。

树状候选或多分支提议：
  提高候选覆盖，但验证和调度复杂度增加。
```

这些方法都可能被归到“推测解码”附近，但 draft 参数来源、训练目标、验证图和适用 runtime 并不相同。资料中写成一个统称时，正文仍应说明具体实现。

### 6.5.8 失败模式：接受率高也不代表总收益高

需要同时检查：

1. draft 本身是否占用了过多 GPU 或 CPU 时间；
2. target 验证候选的 kernel 是否真的比逐 token 更高效；
3. 接受率是否随输出位置、任务类别和采样温度变化；
4. batch 并发下 draft 与 target 是否争抢资源；
5. streaming 是否必须等待一整个验证循环才发 token；
6. EOS、stop sequence、grammar 和 tool call 是否在分歧位置正确处理；
7. 采样 residual 是否实现了分布校正；
8. 输出 token 是否与不使用 speculative 的参考实现一致或在分布上相符；
9. 取消请求后 draft 和 target 的中间状态是否释放；
10. 失败回退到普通 decode 时是否出现 cache 或随机状态污染。

质量测试可以分成两类。贪心模式比较逐步 token；采样模式比较大量样本的分布统计、长度、停止和任务指标。不能因为一条文本看起来相同，就宣布采样分布保持不变。

### 6.5.9 资料与证据边界

- Speculative Decoding 原论文：<https://arxiv.org/abs/2211.17192>
- Fast Inference from Transformers via Speculative Decoding：<https://arxiv.org/abs/2302.01318>
- Transformers v5.17.0 assisted generation：<https://huggingface.co/docs/transformers/v5.17.0/assisted_decoding>
- vLLM speculative decoding（v0.32.0）：<https://docs.vllm.ai/en/v0.32.0/features/speculative_decoding/>
- Medusa：<https://arxiv.org/abs/2401.10774>
- EAGLE：<https://arxiv.org/abs/2401.15077>
- EAGLE-2：<https://arxiv.org/abs/2406.16858>

原论文支持接受/拒绝机制和分布保持条件；Transformers 文档支持具体 API 入口；Medusa、EAGLE 和多 token prediction 属于不同 draft 结构，不能用一个接受率数字概括所有路线。真实收益需要在同一模型、同一请求分布和同一输出协议下实测。

因此，接受率只是解释单轮工作量的一个信号。实际启用前还要在同一到达率下复核分布保持、draft 与 target 的 GPU 竞争、队列和尾延迟。下一篇把视野从“每一步怎样加速”扩展到“很长的输入怎样完成任务”。
