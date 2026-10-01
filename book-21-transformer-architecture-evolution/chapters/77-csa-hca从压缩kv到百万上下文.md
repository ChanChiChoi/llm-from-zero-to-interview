# CSA/HCA：从压缩 KV 到百万上下文

> 资料来源：DeepSeek V4 官方模型卡、V4 Preview 公告与技术报告 `arXiv:2606.19348`（核验日期 2026-09-09）。本章解释报告公开的架构，不把发布方的 FLOPs 比例外推为所有硬件的保证。

## 先算一笔 KV Cache 账

全注意力在生成时需要保留历史 key/value。假设有 `T` 个 token、每个 KV 条目有 `d` 个元素、使用 `b` 字节存储，那么单层缓存近似为 `2Tdb`。当 `T` 从 8K 变成 1M 时，缓存不是“多一点”，而是按序列长度线性增长。多层、多请求和 batch 会进一步放大显存压力。

单纯把上下文窗口字段改成 1M，只说明接口允许更长输入；真正困难是让模型在这个长度下仍有可接受的检索质量、延迟、显存和成本。DeepSeek V4 的 CSA/HCA 试图在 KV 表示层面压缩历史，再在压缩表示上做选择或注意力。

## CSA：先压缩，再稀疏选择

Compressed Sparse Attention（CSA）先把连续 token 的 KV 沿序列维压缩。报告源码描述为：每 `m` 个 token 汇聚成一个压缩条目，序列长度大致变成原来的 `1/m`。随后，DeepSeek Sparse Attention 的 indexer 为每个 query 计算压缩条目分数，只保留 top-k 压缩块进入核心注意力。

这包含两个不同动作：

1. **压缩**减少候选数量，但可能损失块内细节。
2. **稀疏选择**进一步减少每个 query 实际访问的候选。

如果把两步混成“稀疏注意力”，就无法分析错误来源：压缩可能让相关 token 在汇聚时被稀释，top-k 可能在选择阶段漏掉相关块。

一个教学化的压缩可以写成：

```math
c_j=\sum_{r=0}^{m-1}z_{j,r}x_{jm+r},
```

其中 `x` 是原始 KV 条目，`z` 是压缩权重。真实报告还定义了压缩 KV、indexer key、因果可见性和局部滑动窗口；上式只是帮助理解“多个 token 变成一个条目”的简化表达。

## HCA：更激进压缩，但保留 dense attention

Heavily Compressed Attention（HCA）使用比 CSA 更大的压缩跨度 `m'`，把更多 token 汇聚成单个条目，但在压缩后的条目上保持 dense attention。它没有 CSA 的 top-k 稀疏选择，因此每个 query 会访问可见的压缩条目集合。

CSA 更像“较温和压缩 + 稀疏检索”，HCA 更像“重压缩 + 密集读取”。两者可以交错放在不同层，用不同的误差和计算路径换取整体效率。报告还保留滑动窗口分支，为最近 token 提供未被大幅压缩的局部信息。

## 为什么需要局部滑动窗口

压缩条目适合远距离趋势和粗粒度检索，却可能损失刚刚出现的变量、标点、局部语法或代码缩进。滑动窗口让 query 同时访问最近的一段原始 KV，形成“远处压缩、近处精细”的组合。因果约束要求当前位置只能看到此前已经完成压缩的块，以及允许的局部历史，不能因压缩实现而偷看未来 token。

## KV Cache 不再是单一数组

在普通 GQA 中，不同层通常有相对统一的 KV 布局；CSA/HCA 混合后，不同层的压缩倍率、indexer 维度、滑动窗口长度和未完成尾部都可能不同。DeepSeek V4 报告因此描述了异构 KV Cache：

- CSA/HCA 的压缩 KV。
- 稀疏选择所需的 indexer 状态。
- 滑动窗口最近 token。
- 尚未达到压缩块边界的尾部 token。
- 共享前缀和磁盘复用所需的状态。

Serving engine 不能只按“每 token 固定字节数”分配缓存，而需要记录不同层、不同请求和不同缓存段的形状与淘汰规则。

## 一个最小压缩与 top-k 示例

```python

def compress(values, weights, block_size):
    compressed = []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        w = weights[start:start + block_size]
        z = sum(w) or 1.0
        compressed.append(sum(x * a for x, a in zip(block, w)) / z)
    return compressed


def top_k_indices(scores, k):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

values = [1.0, 2.0, 10.0, 11.0, 3.0, 4.0]
compressed = compress(values, [1.0] * len(values), block_size=2)
scores = [0.2, 1.4, 0.7]
chosen = top_k_indices(scores, k=2)
print("compressed:", compressed)
print("chosen blocks:", chosen)
```

这个示例没有实现真实的向量 KV、causal mask、indexer 或 GPU kernel，只用于演示压缩和选择是两个阶段。工程实现需要明确压缩权重、精度、块边界、top-k、局部窗口和 cache eviction。

## 成本指标必须绑定基线

DeepSeek V4 报告给出 1M 场景下相对 V3.2 的单 token FLOPs 和 KV Cache 比例，也给出相对 BF16 GQA8 基线的缓存比例。它们回答的是不同问题：一个是同系列版本对比，一个是与特定注意力配置对比。正式评估时至少要记录：模型版本、上下文长度、精度、batch、硬件、是否包含 indexer、是否包含滑动窗口和 cache 命中情况。

## 局限与面试追问

压缩会带来信息损失；稀疏 top-k 会带来漏检风险；异构 cache 增加调度和实现复杂度；FP4/FP8 还会引入量化误差。CSA/HCA 不是无条件替代全注意力，而是把计算和存储预算集中在更有价值的历史表示上。

**问：CSA 和 HCA 的核心区别？** CSA 在压缩后还做稀疏 top-k 选择；HCA 进行更强压缩，但在压缩条目上保持 dense attention。

**问：为什么还需要滑动窗口？** 为近期 token 保留精细信息，弥补压缩远程表示对局部细节的损失。

**问：1M context 是否意味着模型能准确检索任意位置？** 不是。窗口容量、有效检索、延迟、显存和评测分布是不同指标。

**问：Serving engine 最难的地方是什么？** 不是只把 KV Cache 做大，而是管理不同层的压缩倍率、稀疏索引、局部窗口、未完成尾部和共享前缀。

## 小练习

1. 将示例改为二维 key/value 向量，并实现按块平均压缩。
2. 加入 causal mask，验证当前位置不会读取未来压缩块。
3. 比较无压缩、CSA 风格和 HCA 风格的缓存元素数。
4. 设计一个检索评测，分别测压缩损失、top-k 漏检和局部窗口补偿。

## 2026-09-20 更新：V4 Pro 的模型、effort 与 harness 账本

本章的架构主线现在以 `DeepSeek V4 Pro 0813` 作为活动锚点，但必须把三种身份分开：

| 层 | 本轮证据 | 能回答什么 |
|---|---|---|
| 榜单配置 | Artificial Analysis 的 `deepseek-v4-pro`，标题为 `Reasoning, Max Effort` | 第三方目录、速度/指数、context 和 release date 的观察 |
| 模型与 API | 官方 V4 Pro 公告、模型卡、配置和 Responses/Thinking 文档 | `deepseek-v4-pro`、`low/high/max`、1.6T/49B、1M、CSA/HCA 及协议边界 |
| Agent 评测系统 | DataCurve 的 `mini_swe_agent_deepseek_v4_pro_max` | 该模型配置在固定 harness、工具、环境和 verifier 下的 Pass@1/Pass@4、成本和 steps |

DataCurve 本轮记录为 Pass@1 `62.831858%`、Pass@4 `88.495575%`、平均成本 `$1.6660232187`、平均输出 `105998.9` token、平均 `154.71` steps，`n_runs=4`。这些数字不是“V4 Pro 的裸能力”，因为它们同时包含 max effort、`mini-swe-agent`、工具宿主、任务集、执行环境和 verifier。面试中若只说“V4 Pro 的 SWE 是 62.8%”，就丢掉了最重要的实验条件。

官方配置补充了本章的实现账本：61 层、384 routed experts、每 token 6 个 routed experts、1 个 shared expert、`q_lora_rank=1536`、`o_lora_rank=1024`、`index_topk=1024`、YaRN factor 16 和 FP8 quantization。它们是公开 artifact 的配置字段，不等于某次 API 服务一定使用完全相同的 kernel，也不能由配置直接推出实际吞吐。

API 层还提出一个容易被忽视的面试点：DeepSeek Responses 是 stateless，不支持 `previous_response_id`、`conversation`、`background` 或 `store`；function tools、`apply_patch` 和并行工具调用仍需要宿主执行器、权限、幂等和 verifier。也就是说，百万上下文的 cache/state 设计不能把“模型能读到历史”误写成“服务端自动替用户保存会话”。

**面试追问：为什么 `max` 不是一个新模型？** 因为官方将其定义为同一 V4 Pro API 的 reasoning effort，改变请求级 test-time compute/预算行为；只有当权重、model ID 或 revision 明确变化时，才有理由单独建立模型身份。`max` 的 DataCurve 行仍必须作为独立配置记录，不能和 low/high 或 V4 Flash 的结果混用。

**面试追问：CSA/HCA、mHC、Muon 和 on-policy distillation 是同一层面的技术吗？** 不是。CSA/HCA 是注意力与 KV serving 结构，mHC 是残差流的几何约束，Muon 是优化器/参数更新路径，on-policy distillation 是后训练中的能力合并流程。把它们按“V4 的四个模块”并列，会掩盖架构、优化、后训练和 serving 的责任边界。

## 官方 inference implementation：从架构描述到可追踪执行路径

模型卡和论文告诉我们为什么要压缩 KV；固定 Hugging Face artifact 则让我们看到“压缩、索引、局部窗口、专家和残差混合”在参考实现中如何连起来。2026-09-20 固定的 V4 Pro revision 是 `b5968e9190ef611bbf34a7229255be88a0e937c1`。本节只讨论公开代码路径，不宣称完整权重已下载或本机 GPU 推理已成功。

### 参考实现的状态图

```text
token hidden state
  -> gated KV compressor (ratio 128/4, overlap tail)
  -> learned indexer score + causal mask + top-k
  -> compressed sparse attention
  -> 128-token local sliding window
  -> MLA low-rank Q/O path
  -> MoE top-6 routed + 1 shared expert
  -> Hyper-Connections / Sinkhorn mixing
  -> next block / MTP prediction
```

这里的“状态”不是一块统一的 KV 数组。压缩 KV、indexer 所需的候选状态、局部窗口和压缩块边界的 overlap state 必须分别管理。对 serving engine 来说，prefix hit、eviction、decode continuation 和恢复都要知道当前层属于 CSA/HCA 哪条路径，以及尾部是否已经完成压缩。

### 从 `model.py` 读出的实现证据

- `Compressor` 使用 gated KV pooling；ratio=4 的路径保留 overlap state，避免块边界切断因果历史。
- `Indexer` 对压缩 KV 计算 learned score，施加 causal mask 后做 top-k；参考 inference 路径中的 indexer 还有 FP4 模拟量化。压缩损失和 top-k 漏检因此必须分开测量。
- `Attention` 同时维护 MLA 低秩投影、compressed-KV sparse attention 和 `window_size=128` 的局部窗口；窗口不是“额外的完整注意力”，而是对近期细节的补偿分支。
- `Gate` 的前三层使用 token-id hash routing，后续使用 `sqrtsoftplus` score routing；selection bias 只改变候选专家选择，不改变 routing weight。该字段来自公开 inference code，不应外推成完整训练策略。
- `MoE` 为 top-6 routed experts + 1 shared expert，专家按 tensor parallel 分片；`MTPBlock` 公开了 multi-token prediction block。MTP 仍要单独记录 draft、verify、accepted length、rollback 与 committed cache，不能看到类名就宣称端到端 speculative decoding 已验收。
- `Block` 用 `hc_mult=4` 和 20 轮 Sinkhorn 近似双随机残差混合，与第 78 章的 mHC 数学主线相连；这是残差流约束，不是 CSA/HCA 的一种变体。

### `kernel.py` 的低精度与部署边界

TileLang 参考 kernel 显示了 `[128,128]` block 的 FP8 activation quantization、FP4 quantization、FP8/FP4 GEMM、稀疏 attention online softmax 和 HC Sinkhorn 路径；FP4 权重沿 K 维打包后参与 GEMM。官方 inference README 的 `EXPERTS=384`、`MP=8` 是转换示例参数，不能改写为所有硬件的生产 TP/EP 结论。FP4/FP8 的 scale、KV cache dtype、专家 dispatch、workspace 和通信需要与 GPU 型号、并行拓扑及 batch 一起 profiling。

因此本章新增的实现证据可以回答“公开参考代码怎么把概念串起来”，但还不能回答“线上 1M context 一定达到多少 tokens/s”。后者需要固定权重、commit、GPU、batch、prefill/decode 比例、cache 命中、压缩召回和 verifier，并把发布方 benchmark 与本地结果分栏。

## 2026-09 更新：SGLang 的 SWA 分支点缓存与 V4 serving kernel

### 共享前缀的两个状态不能混成一个 cache hit

把一个大 system prompt 后接多个 Agent 子任务想成树：共同前缀只 prefill 一次，每个分支追加自己的 question 和 tool history。普通 radix cache 可能保留完整 KV，但 DeepSeek V4 还要维护滑动窗口分支的状态。chunked prefill 清理窗口外的 SWA slots 后，后续 sibling 即使命中 Full KV，也未必能从正确的分叉位置恢复 SWA；此时服务端需要重新计算相应前缀，或者会错误地把不完整状态当作完整命中。

SGLang `v0.5.20` 的统一 radix tree PR [#34565](https://github.com/sgl-project/sglang/pull/34565) 让 SWA 状态在分支点继续保留：先按分支位置调整窗口外 slots，把新分支插入 tree，再清理其余无用状态。这样 branch point 成为一等缓存边界。它需要额外状态占用，因此优化目标不是“永远不释放”，而是在复用概率、缓存容量和后续请求之间做生命周期管理。

PR 的 DeepSeek-V4-Flash-0731 shared-prefix workload 固定 TP=2、FlashInfer MXFP4、DSpark，system prompt 24,576 tokens，question 8,192，output 128；64 条请求分成 8 个共享前缀组。开启 out-of-window free 时，branch-point caching 把 token hit rate 从 `43.81%` 提到 `60.75%`，mean TTFT 从 `1.570s` 降至 `1.070s`，p95 TTFT 从 `3.427s` 降至 `2.373s`；input throughput 从 `66.3K` 到 `70.5K tokens/s`。这是 SGLang PR 自报的指定负载结果，不是本地复现，也不能替代真实 Agent workload 的 p95/p99 与单位成功成本。

面试追问应落到状态账本：Full KV、压缩 KV、indexer state、SWA slots 和 overlap tail 是否分别有 owner、branch key、TTL、eviction 与 restore 规则？工具调用造成分支后，`call_id`、工具结果和 verifier artifact 是否也绑定在同一个 task lineage？只报一个 prefix hit rate 会掩盖这些问题。

### CSA/HCA kernel 优化必须绑定硬件与 baseline

SGLang `v0.5.20` release 还收录了两类 DeepSeek V4 serving 路径：

| 后端/硬件 | 发布的实现证据 | 数字应如何理解 |
|---|---|---|
| B200（SM100/103） | TRT-LLM attention kernel 覆盖 V4 的 CSA/HCA，与 FlashMLA 比较 | PR #30805 报告 FP8/TP1 单元 kernel prefill 约 `1.2x`、decode 约 `1.45x`；不是 Agent 端到端或所有 V4 变体结果 |
| 4× RTX PRO 6000（SM120） | DeepGEMM paged-MQA sparse indexer、FlashInfer sparse prefill、DeepGEMM FP4 MoE 与 SWA page-split 改进 | PR #29927 报告相对 torch fallback 的单请求 TPOT 最多 `3.4x`；同一 PR 将 HC prenorm 的附加贡献约 `3.2%` 单独披露，baseline/backend 条件必须同时报告 |

硬件专用实现说明 attention 的收益来自多层协作：稀疏索引 kernel 影响候选准备，CSA/HCA kernel 影响实际读取，MoE backend 影响 expert compute，SWA page layout 影响每步内存搬运。不能把某一个 kernel 的倍数当成模型能力提升。SGLang release/PR 记录的是框架实现证据；本机没有相同 GPU、完整权重和固定 benchmark 时，只能标为发布方结果，不能声称复现或生产 acceptance。
