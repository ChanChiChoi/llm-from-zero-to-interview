# 第 92 章 Qwen3-VL：Interleaved-MRoPE、DeepStack 与视频时间戳

> 核验日期：2026-09-22。模型发现入口是 [Artificial Analysis 的 Qwen3-VL-235B-A22B instruct 条目](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-instruct) 和同一基础模型的 reasoning 条目。DataCurve 当前没有精确的 `mini_swe_agent_qwen3_vl_*` 行，因此本章不迁移其他 Qwen 模型的 Agent 成绩。

## 92.1 先明确模型身份：两个配置，不是两个基础模型

Artificial Analysis 同时展示 `Qwen3-VL-235B-A22B-Instruct` 和 `Qwen3-VL-235B-A22B-Reasoning`。前者页面给出约 `235B total / 22B active`、`262,144` context 和约 `50.94 tokens/s`；后者给出同样的总量/激活量和上下文字段，但 Intelligence Index、输出价格和推理配置不同。

这两个页面应归并为同一个 Qwen3-VL 基础模型的运行配置或 artifact。`Instruct` 与 `Thinking` 的模型卡/配置也不能被计作两个独立基础模型。面试时先固定以下 manifest：

| 字段 | 本轮记录 |
|---|---|
| 模型锚点 | `Qwen3-VL-235B-A22B` |
| AA 配置 | `instruct`、`reasoning` |
| 总参数/激活参数 | AA 页面字段 `235B / 22B`，属于第三方目录口径 |
| 上下文 | AA/config/论文 S3 分别绑定自己的来源，当前均为约 262K，不外推 1M |
| DataCurve | 无精确 `mini_swe_agent_qwen3_vl_*` 行，不迁移其他 Qwen 结果 |
| 官方报告 | [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) |

这里有一个常见错误：看到 instruct/reasoning 两个页面，就把它们当作两个模型，再把其中一个配置的 benchmark 迁移给另一个。正确做法是把模型身份、推理开关、输出预算、provider 和测量时间分开记录。

## 92.2 先看完整数据流，而不是只看一个视觉编码器

Qwen3-VL 的结构可以先画成三段：

```text
image / video patches
        |
        v
  SigLIP2 vision encoder
        |
        v
  two-layer MLP merger
        |
        +--> visual tokens into Qwen3 decoder
        +--> DeepStack residuals into early LLM layers
                         |
text tokens + position ids -> Qwen3 MoE language model
                         |
                 answer / reasoning / tool proposal
```

视觉编码器负责把 patch 变成视觉表示；merger 把视觉 hidden size 投影到语言模型 hidden size；Qwen3 decoder 再处理文本、视觉 token 和多维位置。DeepStack 是额外的残差注入路径，不是把视觉 encoder 复制成三个完整副本。

当前 HF config 给出的关键结构字段是：

| 子系统 | 字段 |
|---|---|
| 语言模型 | 94 layers，hidden size 4096，64 个 query heads、4 个 KV heads，head dim 128 |
| MoE FFN | 128 experts，top-8 routing，intermediate size 1536 |
| 位置与精度 | `max_position_embeddings=262144`、RoPE theta `5,000,000`、BF16 |
| 视觉模型 | depth 27，hidden size 1152，patch size 16，temporal patch size 2 |
| 合并器 | spatial merge size 2，output hidden size 4096 |
| DeepStack | 中间层索引 `[8,16,24]` |

这些字段能说明公开实现的形状，不能推出完整权重已经在本机加载、专家负载是否均衡、生产 kernel 是否高效，或论文 benchmark 是否可独立复现。

## 92.3 为什么普通 RoPE 不够用

文本 token 可以用一个位置整数表示顺序。视频 token 却至少有三类坐标：时间 `t`、图像高度 `h` 和图像宽度 `w`。一帧图像里的两个 patch 可能高度相邻，但来自不同时间的两个 patch 也可能在序列中相邻。把它们全部压成一条位置轴，模型很难区分“空间邻近”和“时间邻近”。

多模态模型因此常用 multi-dimensional RoPE，把位置写成：

```text
position = (temporal_id, height_id, width_id)
```

每个轴分到一部分 rotary dimensions。问题在于，若 temporal、height、width 完全占用连续的频率块，某一轴可能集中在特定频率范围；长视频中这种频谱分工会造成表示偏置，尤其当输入长度和采样密度改变时更明显。

## 92.4 Interleaved-MRoPE：交错频率分配

Qwen3-VL 的 Interleaved-MRoPE 在低频和高频维度中交错分配 temporal、height、width 轴。它保留三维位置表示，但改变不同轴进入旋转频谱的方式：

```text
旧式直觉： [temporal block][height block][width block]
新式直觉： [t][h][w][t][h][w] ... 交错到不同频率范围
```

图示是概念化表达，实际维度排列应以论文和对应实现为准。它的重点不是删掉空间轴，也不是把视频变成无序集合，而是让三个轴在低频和高频中更均衡地出现。

可以把一个 rotary pair 的位置变换抽象为：

```text
R(theta_axis(position_axis)) * [x_2i, x_2i+1]
```

其中 `axis` 可能是 temporal、height 或 width；Interleaved-MRoPE 改变的是不同轴如何映射到 `theta_axis` 的维度集合。它希望在局部纹理、区域关系和长时间跨度之间保持更稳定的频率覆盖。

不要从 `mrope_section=[24,20,20]` 单独推导全部频率设计。配置可以固定三个轴的维度账本，论文和上游代码才共同支持完整的生成规则。面试中的严谨回答应是：“它是三轴多模态 RoPE 的频率交错分配，用于缓解长视频频谱偏置；不是上下文扩展算法，也不是视频检索器。”

## 92.5 DeepStack：为什么要使用视觉中间层

视觉 Transformer 的最后一层具有较强的语义抽象，但局部纹理、边缘、几何和区域关系可能在多层变换中被重新编码。若只把最后一层送入 merger，语言模型接收到的是单一深度的视觉摘要。

DeepStack 的做法是从视觉 encoder 的三个中间层取出特征，对每层使用专用 merger，再将它们残差注入语言模型早期层。当前配置给出 `[8,16,24]`，上游 Transformers modeling 代码也确认视觉 embedding 会在语言模型前几层注入。

可以用简化公式表示：

```text
h_0 = text_and_visual_embeddings
h_l = Transformer_l(h_{l-1} + P_l(v_{k_l}))
```

`v_{k_l}` 是视觉 encoder 的中间层表示，`P_l` 是对应投影；公式省略了真实实现中的 token 对齐和层映射。它表达的直觉是：语言模型早期层在做基础文本—视觉对齐时，同时看见不同抽象层级的视觉证据。

DeepStack 的一个重要工程性质是“不增加额外上下文 token 长度”。视觉 token 序列仍由 processor 和 patch/grid 决定，新增的是早期层的 residual 计算、投影和激活保存。因此它可能改善视觉信息流，却仍会增加显存和算力；不能把“不增加 token”回答成“没有额外成本”。

## 92.6 Video Timestamp：给视觉 patch 一个可读的时间锚点

只用绝对 position id 时，长视频经过抽帧、压缩和 chunk 切分后，模型看到的是离散 token 序列。第 200 个视频 token 并不天然等于第 3.0 秒：不同帧率、丢帧和采样策略都会改变 token 与真实时间的对应关系。

Qwen3-VL 在视频时间 patch 前加入文本时间戳，例如：

```text
<3.0 seconds> [visual tokens for this temporal patch]
```

训练同时使用 seconds 和 HMS 格式，让模型学习时间的可读表达。这让模型有机会回答“事件发生在第几秒”，也让长视频里的时间证据更容易被语言推理路径引用。

它与 Interleaved-MRoPE 分工不同：

| 机制 | 主要职责 | 仍然需要宿主处理的部分 |
|---|---|---|
| Interleaved-MRoPE | 在模型内部表示 temporal/height/width 的位置关系 | 采样率、frame index、chunk offset、状态恢复 |
| Video Timestamp | 把真实时间锚点显式写入输入上下文 | 视频读取权限、丢帧检测、重复帧、时间戳可信度 |

所以 `<3.0 seconds>` 不是一个视频数据库查询，也不能自动保证时间定位正确。生产系统必须将媒体 revision、时间戳来源、采样策略、frame hash 和 chunk 边界放进回放 manifest。

## 92.7 从 S0 到 S3：长上下文是训练 curriculum

Qwen3-VL 的 262K 能力不是只修改 `max_position_embeddings`。报告描述的预训练路线可以简化为：

| 阶段 | 训练方式 | 公开 token/context |
|---|---|---|
| S0 | 冻结大部分语言模型，只训练 merger | 约 67B tokens，8K |
| S1 | 全参数多模态预训练 | 约 1T tokens，8K |
| S2 | 全参数长上下文预训练 | 约 1T tokens，32K |
| S3 | ultra-long adaptation | 约 100B tokens，262K |

报告还使用 square-root normalized per-token loss，平衡 text-only 与 multimodal 数据的 token 贡献。它的直觉是：如果直接把 token loss 按数据量相加，样本量更大的数据源会淹没其他模态；归一化后再聚合，可以让每种数据源的训练影响更可控。

这仍不是完整可复现 recipe。报告没有因此公开所有数据筛选、采样权重、optimizer schedule、并行配置、长序列 packing 和每个模态的 batch 细节。面试时应区分“训练阶段与目标已公开”和“完整训练配方已公开”。

后训练还把 SFT 从 32K 扩展到 256K，并包含 strong-to-weak distillation、Reasoning RL 和 General RL。Reasoning RL 使用 SAPO；General RL 使用规则 reward 与 model-based reward。算法名称能定位知识点，但不能证明候选人已经知道奖励权重、rollout 规模或线上策略。

## 92.8 Thinking with Images：为什么要奖励工具调用过程

视觉推理 Agent 往往需要多次读取图像区域、搜索文字、执行 GUI 操作或验证文档。只奖励最终答案会产生一个危险捷径：模型可能固定调用一次工具，或者直接猜答案，只要偶然得到正确结果就获得高 reward。

Qwen3-VL 报告描述了约 10K grounding cold-start examples，用 Qwen2.5-VL-32B 进行 visual-agent SFT 和 tool-integrated RL，并蒸馏约 120K multi-turn agent interactions。reward 至少分为：

1. **answer accuracy**：最终回答是否正确；
2. **multi-turn reasoning**：是否合理读取证据、规划步骤和处理多轮反馈；
3. **tool-calling reward**：调用工具的时机、参数、次数是否与任务复杂度匹配。

第三项尤其重要。报告指出只使用前两种 reward 可能让模型形成固定的一次工具调用策略；tool-call reward 约束探索行为，减少“答案正确但过程不可靠”或“每个任务都重复调用一次”的策略。

这也是面试中很好的 reward design 例子：

```text
最终正确 != 轨迹高质量
工具调用更多 != 探索更好
一次调用成功 != 过程可恢复
```

实际评测还需要检查工具 schema、权限、错误返回、重试、幂等、超时和独立 verifier。论文中的交互轨迹数量属于训练数据披露，不是某个生产 GUI Agent 的成功率。

## 92.9 从模型输出到 GUI/工具动作

官方 README 列出了 GUI、手机 Agent、OCR、文档解析、视频理解、2D/3D grounding、Thinking with Images 和 multimodal coding 等方向。系统设计上，Qwen3-VL 应被放在“提出计划和动作”的位置：

```text
image/video/document
       -> model reasoning
       -> tool proposal / coordinates / search query
       -> schema + permission + safety gate
       -> host executor
       -> observation replay
       -> independent verifier / artifact
```

模型给出点击坐标，不代表点击已经发生；模型给出代码 patch，不代表测试通过；模型找到文档区域，不代表租户权限允许导出。宿主仍负责浏览器或手机控制、文件和网络沙箱、凭据隔离、人工确认、超时、幂等、审计和副作用回滚。

GUI Agent 还要多记几类状态：截图或页面 revision、窗口尺寸、设备像素比、坐标缩放、动作序号、页面加载状态和动作后的截图 hash。否则模型在上一帧生成的坐标可能被错误地重放到下一帧页面。

## 92.10 一个最小的时间戳与 DeepStack 教学例子

下面的 Python 代码只演示两个概念：将视频 timestamp 映射到固定时间 bin，以及把不同视觉深度的投影作为 residual 加入早期层。它不实现 Qwen3-VL 的真实 RoPE、视觉 encoder、MoE routing 或生产 kernel。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualEvent:
    timestamp_ms: int
    token: str


def temporal_bin(timestamp_ms: int, step_ms: int = 1000) -> int:
    if timestamp_ms < 0:
        raise ValueError("timestamp must be non-negative")
    return timestamp_ms // step_ms


def deepstack_add(hidden: list[float], projected: list[float]) -> list[float]:
    if len(hidden) != len(projected):
        raise ValueError("residual dimensions must match")
    return [left + right for left, right in zip(hidden, projected)]


events = [VisualEvent(3010, "frame-c"), VisualEvent(990, "frame-a")]
ordered = sorted(events, key=lambda item: (temporal_bin(item.timestamp_ms), item.timestamp_ms))
print([(temporal_bin(item.timestamp_ms), item.token) for item in ordered])
print(deepstack_add([1.0, 2.0], [0.1, -0.2]))
```

输出类似：

```text
[(0, 'frame-a'), (3, 'frame-c')]
[1.1, 1.8]
```

教学实现忽略了 timestamp jitter、视频帧重复/丢失、真实 HMS token、三维 rotary index、视觉 token 对齐和 batch offset。生产实现必须把 bin 规则、媒体 revision 和 frame hash 固定在 manifest 中；不能因为 toy 代码能排序，就声称完成长视频定位或跨 chunk replay。

## 92.11 Serving 账本：视觉 token、专家、状态和工具要分开

Qwen3-VL 的服务成本至少由四部分组成：视觉预处理与 patch 数量、文本/视觉 token 的 prefill、MoE experts 的 dispatch，以及多轮 Agent 的工具和回执。`235B total / 22B active` 不能直接等价于单卡 22B 显存或固定每 token 延迟。还要记录 resident weights、expert placement、通信、KV cache、视觉 embedding、workspace 和并发。

一个可审计的请求 manifest 可以包含：

```text
model/revision/variant
media hash/format/duration/frame sampling/timestamp source
visual grid/token count/DeepStack layers/M-RoPE axes
prompt/tool schema/reasoning effort/output budget
expert routing/cache/TP-EP topology
tool call/permission/result/retry/idempotency
TTFT/TPOT/first evidence/quality/verifier/artifact digest
```

部署验收按层推进：

```text
config and tokenizer
  -> processor and visual preprocessing
  -> full-weight load
  -> text/vision numerical correctness
  -> M-RoPE/DeepStack/timestamp replay
  -> target backend and hardware profile
  -> GUI/tool permission and verifier acceptance
```

HF config 通过只说明字段存在；上游 Transformers raw main 通过只说明代码入口存在；都不能替代 vLLM/SGLang 的完整多模态 serving、目标 GPU profiling 或端到端工具验收。本轮 upstream 文件是 2026-09-22 raw main 快照，没有固定 commit SHA，不能把它当成不可变生产版本。

## 92.12 常见误区

### 误区一：AA 的 22B active 就是实际显存

错误。它是第三方目录或模型规模字段。MoE 总权重、KV/cache、expert placement、通信、workspace 和并发要单独计账。

### 误区二：Instruct 和 Thinking 是两个基础模型

错误。当前证据更支持同一 Qwen3-VL 家族的不同 artifact/运行配置。模型 identity、reasoning mode、provider 和评测 harness 要分开记录。

### 误区三：Interleaved-MRoPE 就是 1M context 扩展

错误。它是 temporal/height/width 位置频率的交错分配。当前论文、HF config 和 AA 证据绑定的是约 262K；不能借用其他 Qwen 模型的 1M 字段。

### 误区四：DeepStack 增加三倍视觉 token

错误。它从中间视觉层提取特征并作为 residual 注入早期语言层，不增加额外上下文 token 长度，但会增加投影、激活和计算成本。

### 误区五：模型输出 GUI action 就代表动作成功

错误。动作必须经过宿主 schema/权限/安全门禁、执行器、观察回灌和独立 verifier；坐标还要绑定截图和窗口 revision。

### 误区六：论文里的 Agent 轨迹数就是生产成功率

错误。轨迹数量属于训练数据披露。成功率必须绑定任务、工具、环境、权限、超时、verifier、模型 revision 和统计方法。

## 92.13 面试追问

**问：Interleaved-MRoPE 具体解决什么问题？**

答：Qwen3-VL 仍然使用 temporal/height/width 三轴位置表示，但把轴对应的 rotary frequency 在低频和高频中交错分配，缓解长视频中的频谱偏置。它不等于上下文扩展、视频检索或 sampling 策略。

**问：DeepStack 为什么要用视觉中间层？**

答：最后视觉层偏向高层语义，中间层保留不同深度的局部、区域和结构信息。专用 merger 将 `[8,16,24]` 等中间特征投影后残差注入语言模型早期层，增强早期跨模态对齐；它不复制三份序列 token，但会增加投影和激活成本。

**问：视频时间戳和 M-RoPE 是否重复？**

答：不重复。M-RoPE 是模型内部的多轴位置表示；文本时间戳是显式可读的绝对时间证据。一个负责表示关系，一个帮助模型引用真实秒数，两者都不能代替宿主的时间轴和丢帧审计。

**问：为什么 tool-calling reward 不能只看最终答案？**

答：只看答案可能奖励固定一次调用、猜测或不可恢复的捷径。tool-calling reward 应约束调用时机、参数和次数与任务复杂度匹配；还要用独立 verifier 检查工具结果和副作用。

**问：Qwen3-VL 的 262K 是不是每个视频都能有效使用？**

答：不是。它是接口/config/训练阶段的上下文边界，不保证有效视频召回。还要测采样率、视觉 token 数、时间戳、position、长视频定位、工具结果和质量—成本曲线。

**问：Transformers 支持 Qwen3-VL，是否等于生产 serving 已完成？**

答：不等于。还要验证固定 revision 的完整权重、processor、视觉数值正确性、DeepStack/M-RoPE、TP/EP、KV/cache、目标硬件性能、GUI/tool 权限和 verifier。上游 raw main 也要绑定日期或 commit。

## 92.14 小练习

1. 画出 SigLIP2、merger、DeepStack、Qwen3 MoE、GUI executor 和 verifier 的数据/权限边界。
2. 设计一个长视频时间定位实验，对比只用 position id、只用 timestamp、两者同时使用的召回和误差。
3. 固定视觉 token 数，比较打开/关闭 DeepStack 时的早期层激活、TTFT、显存和视觉问答质量。
4. 为 GUI Agent 写一份 replay manifest，至少包含截图 hash、窗口尺寸、像素比、坐标、动作序号、权限决定和动作后验证。
5. 设计三个 reward：最终答案、证据读取轨迹、工具调用策略，并构造“固定调用一次工具”的 shortcut 测试。
6. 用 262K 上限设计 text-only、image-heavy、long-video 和 tool-heavy 四种 budget，报告未覆盖证据而不是静默截断。

## 92.15 来源

- [Artificial Analysis Qwen3-VL-235B-A22B instruct](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-instruct)
- [Artificial Analysis Qwen3-VL-235B-A22B reasoning](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-reasoning)
- [Qwen3-VL Technical Report, arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
- [Qwen3-VL 官方 GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-235B-A22B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
- [Qwen3-VL-235B-A22B-Thinking model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)
- [Transformers Qwen3-VL configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py)
- [Transformers Qwen3-VL modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [Transformers Qwen3-VL processing](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
- 完整证据和待核验项见 [`qwen3-vl-source-notes.md`](../../research/model-update-2026-09/qwen3-vl-source-notes.md)。
