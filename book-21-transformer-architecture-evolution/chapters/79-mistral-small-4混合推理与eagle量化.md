# Mistral Small 4：把指令、推理和部署优化放进一个模型

> 资料来源：[Mistral Small 4 119B A6B 官方模型卡](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)，核验日期：2026-09-09。本章只把模型卡明确披露的字段写成模型事实；教学公式和代码是帮助理解的抽象。

## 先从一个产品问题开始

在线助手通常有两类请求。用户问“今天北京天气如何”时，希望很快得到一句话；用户让模型修复一个跨文件的并发 bug 时，却愿意等待更长的推理。过去工程师常用两个模型分别处理这两类请求：一个 instruct 模型追求延迟，一个 reasoning 模型追求质量。路由器要维护两套权重、两套 tokenizer 配置和两套监控口径。

Mistral Small 4 的模型卡给出的思路是把 Instruct、Reasoning（此前称 Magistral）和 Devstral 三个家族的能力合并到一个模型，再把“是否进行较长推理”暴露为每请求的 `reasoning_effort` 参数。这样，模式选择变成一次请求级策略，而不是重新部署另一套模型。

## 已核验的模型结构

模型卡将它标为 **Mistral Small 4 119B A6B**，给出以下字段：稀疏 MoE 共有 128 个专家、每个 token 激活 4 个专家；总参数约 119B、每 token 激活约 6.5B；上下文长度 256K；输入支持文本和图像，输出为文本。这里的 A6B 是“约 6.5B 激活参数”的产品命名线索，不能把它误读成总参数只有 6B。

MoE 的 total parameters 与 active parameters 回答不同问题。总参数决定权重存储和分片压力，激活参数更接近每 token 的矩阵乘计算量。推理时仍要把所有专家权重放在设备或可访问的分片中；“每 token 只算 6.5B”并不表示一张小显卡就能容纳 119B 权重。

## `reasoning_effort` 是模式开关

模型卡规定 `reasoning_effort="none"` 时不启用 reasoning，`reasoning_effort="high"` 时启用较深的推理。它不是一个跨模型的统一标尺，不能拿 `high` 和另一个厂商的 `high` 直接比较。

可以把一次调用的决策写成：

```math
u^*=argmax_{u\in\{\mathrm{none},\mathrm{high}\}}
\bigl(Q(u)-\lambda C(u)-\mu L(u)\bigr),
```

其中 `Q` 是任务质量，`C` 是 token、GPU 或 API 成本，`L` 是延迟，`λ` 和 `μ` 是业务权重。真实系统不一定计算这个公式，但它提醒我们：模式选择是质量、成本和延迟的联合决策。

下面的策略代码不调用模型，只演示如何把任务类型、预算和失败风险转成模式，并把选择写入 trace：

```python
def choose_effort(task, latency_budget_ms, risk_level):
    if risk_level == "high" and latency_budget_ms >= 2_000:
        return "high"
    if task in {"code_review", "math_proof", "multi_step_agent"}:
        return "high" if latency_budget_ms >= 1_500 else "none"
    return "none"


for case in [
    ("weather", 500, "low"),
    ("code_review", 3_000, "medium"),
    ("medical_summary", 3_000, "high"),
]:
    print(case, "->", choose_effort(*case))
```

生产系统还要记录实际输出 token、工具次数、超时和人工复核结果。只记录参数而不记录结果，无法知道 `high` 是否真的带来单位成功成本下降。

## 统一模型的工程收益和代价

统一权重可以减少路由和发布矩阵，但不会自动消除运行时复杂度。`high` 模式可能产生更长的思考文本或更多工具调用，连续批处理时会改变请求的 token 形状；`none` 模式的短请求和 `high` 模式的长请求混在一起，调度器仍需按剩余 token 和 KV cache 预算做 admission。

如果一个服务把所有请求都固定为 `high`，就失去了模式切换的成本收益；如果所有请求都固定为 `none`，又无法使用模型卡所说的 reasoning 能力。应在离线切片评测中比较两个模式的质量、长度、TTFT、TPOT、p95、显存和失败类型。

## EAGLE 草稿头和 NVFP4 检查点

模型卡还链接了训练好的 EAGLE speculative-decoding head，以及 NVFP4 量化检查点。它们作用在部署层：EAGLE 让草稿头先提出多个候选 token，再由目标模型验证；NVFP4 用 4-bit 浮点表示降低权重或激活存储。二者都可能提高吞吐或降低显存，但需要单独测量接受率、回退比例和质量差异。

推测解码的简化成本模型为：目标模型每次验证 `k` 个草稿 token，若平均接受 `a` 个，则单位目标模型调用的有效 token 约为 `a+1`。只有当草稿生成成本、验证并行度和接受率组合起来有利时，吞吐才会提高。不能看到“有 EAGLE head”就承诺固定倍数加速。

NVFP4 同样需要校准和误差审计。对权重、KV cache 和激活采用不同精度时，误差来源不同；应按任务切片比较 exact-match、代码测试通过率、视觉问答和长上下文检索，而不是只看平均 loss。

## 多模态和工具边界

模型卡确认文本和图像输入、文本输出，并描述 function calling、JSON 输出和 agent 场景。模型发出函数调用只是计划，真正执行仍由 parser、权限策略、沙箱和工具宿主负责。图像输入能被模型分析，也不等于模型自身生成图像；如果需要生成图像，应另接图像生成器。

## 许可证与部署

官方模型卡标注 Apache 2.0，给出 vLLM、Transformers、SGLang 和 llama.cpp 等部署入口。许可证适用范围、第三方依赖和模型卡附带文件仍应在发布版本中逐项核对。模型卡的 benchmark 是发布方测试，硬件、批大小、量化和后端版本不同，40% 的完成时间降低与 3 倍吞吐提升不能直接外推。

## 面试追问

**问：119B 模型为什么能按 6.5B 的计算量运行？**  MoE 每 token 只路由到少数专家，矩阵乘激活量接近 6.5B；但权重存储、专家并行和 all-to-all 通信仍按总专家规模产生压力。

**问：`reasoning_effort=high` 是新模型吗？** 不是。它是同一模型的请求级推理模式，应在评测中作为配置维度记录。

**问：EAGLE 和量化能叠加吗？** 可以尝试，但需要验证草稿头与量化目标模型的接受率、数值误差和后端支持，不能把两个独立宣传点相乘成固定加速比。

**问：统一三类模型能力是否等于三套模型简单拼接？** 不是。模型卡只说明能力统一，没有披露完整训练配方，不能据此推断内部如何融合。

## 小练习

1. 为 `none/high` 两种模式设计同任务评测表，加入质量、输出长度、TTFT、TPOT 和单位成功成本。
2. 估算 119B BF16 权重的理论存储，并比较 4-bit 权重的存储下界；说明实际还要加 scale、路由和运行时开销。
3. 写一个 toy MoE 路由器，统计 128 个专家的负载和 top-4 溢出。
4. 在固定质量门槛下，设计 EAGLE 接受率与目标模型调用次数的消融实验。
