# 第 14 章 Thinking Levels：离散档位如何成为可验证的服务契约

本章讨论用户或调用方看到的 low、medium、high 等离散控制项。它们是具体 provider、模型和 API 版本上的接口选项，不是跨模型通用的智力刻度。本章沿“接口映射—质量与成本—容量和权限—协议回归”展开；预算计量见[第十三章](13-reasoning-effort与推理预算.md)，在线策略见[第十八章](18-adaptive-thinking与动态预算.md)，跨轮状态见[第十五章](15-preserve-thinking与推理状态.md)。

## 14.1 档位是产品控制面

大多数调用方不希望逐项配置推理 token 上限、验证次数、工具次数和截止时间。产品可以将一组请求选项合并成“快速”“标准”“深入”等档位，降低选择成本。这样的档位有用，但它首先是一个接口约定：请求方选择名称，provider 按该模型和 API 的公开合同解释它，服务端再记录实际模型、事件和 usage。

档位名称本身并不说明内部算法。仅凭 `high`，不能断定后端增加了多少 token、运行了几轮验证或使用了树搜索。不同模型甚至可能让同一字段控制不同对象。读档位文档时，应依次确认准确的 model ID、API surface、支持值、默认值、输出上限、usage 字段和状态延续要求；没有公开的内部映射就应保留为未知。

这也是接口与测量的分界：请求的档位是输入条件，实际消耗和任务结果是观测值。最终可见文本长度不能替代 reasoning usage；一次回答更长，可能只是解释更长，并不能证明它检验了更多候选。

## 14.2 把档位展开成资源向量

为了说明某个产品如何实现档位，设 \(\ell\) 是离散 level，\(m\) 是模型及其 revision，\(x\) 是任务类型，\(p\) 是产品策略，\(\mathbf b\) 是该请求的资源约束向量。它可以分别记录推理 token 上限、允许的工具调用数、验证次数和 deadline；计数上限是非负整数，时间以秒或毫秒记录，这些量单位不同，不能未经换算相加。产品策略可以把这些输入映射成请求配置：

```math
\mathbf b = g(\ell,m,x,p)
```

例如，某个**合成的产品配置**可把后端 A 的 `medium` 展开为“推理 token 目标 2,048、最多 1 次工具调用、1 次 verifier、4 秒 deadline”；另一个后端 B 的 `medium` 则可能只对应“推理 token 目标 1,024、无工具调用、0 次 verifier、2 秒 deadline”。这些数字只是展示字段如何分开记录，不是 provider 的真实预算或性能承诺。它们说明 `medium` 只能在各自合同内解释，不能据名称说 A 与 B 投入了相同计算。

产品还应区分“请求上限”和“实际消耗”。一次请求即便获准最多进行两轮验证，也可能首轮通过而只执行一轮；deadline 则可能先于 token 上限触发。账本应记录请求配置、实际事件、实际 usage 和结束原因，具体计量口径与单位沿用第十三章。

## 14.3 建立 capability matrix 与有损适配规则

跨 provider 迁移时，直接转发 `high` 字符串会把接口名称误当成语义兼容。适配层需要对每个 model ID 维护能力矩阵，至少说明原生档位字段、usage 能否拆分、thinking 状态是否需要回传、工具事件是否有结构化表达、截止/取消语义，以及不支持时的返回方式。返回给调用方的结果也要包含实际采用的路径。

| 状态 | 适配器能承诺什么 | 需要暴露的限制 |
| --- | --- | --- |
| `native` | 使用后端文档明确支持的档位字段 | 不推断未公开的内部预算；仍记录 usage 与 model revision |
| `adapted` | 按公开能力做有边界的转换 | 逐项列出语义损失；不得称为与原生档位等价 |
| `unsupported` | 拒绝该请求或转到调用方认可的后端 | 明确返回不支持或实际改用的 model ID，不静默伪装 |

例如，后端没有 native thinking level、只有 `max_output_tokens` 时，适配器只能提供一个**总生成上限**，不能把该上限命名为 reasoning budget。它限制思考与可见输出的总量或后端文档规定的输出范围；它不保证更多验证，也不保证留下足够的可见答案空间。因此应记录 `adapted` 及其损失，并把真实质量交给对照评测。

### 14.3.1 教学适配器

下面的代码用三个合成后端展示 `native`、`adapted` 与 `unsupported` 三种返回状态。`output_only` 后端只收到总输出 token 上限；该数字不代表内部思考预算。工具权限由调用方单独传入，不由档位推导。

代码状态：**已验证**（2026-11-13；Python 3.12.9，标准库，无第三方依赖）。

```python
LEVELS = {
    "low": {"max_output_tokens": 1024},
    "medium": {"max_output_tokens": 3072},
    "high": {"max_output_tokens": 6144},
}
BACKENDS = {"native", "output_only", "plain"}
PERMISSIONS = {"none", "read_only", "approved_write"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compile_level(level, backend, tool_permission, allow_approximation=False):
    require(type(level) is str and level in LEVELS, "unknown level")
    require(type(backend) is str and backend in BACKENDS, "unknown backend")
    require(
        type(tool_permission) is str and tool_permission in PERMISSIONS,
        "unknown tool permission",
    )
    require(type(allow_approximation) is bool, "allow_approximation must be bool")

    if backend == "native":
        return {
            "status": "native",
            "request_fields": {"thinking_level": level},
            "tool_permission": tool_permission,
        }

    if backend == "output_only" and allow_approximation:
        return {
            "status": "adapted",
            "request_fields": {
                "max_output_tokens": LEVELS[level]["max_output_tokens"]
            },
            "tool_permission": tool_permission,
            "losses": [
                "no_native_thinking_level",
                "total_output_cap_is_not_reasoning_budget",
                "no_added_verifier_or_tool_round",
            ],
        }

    return {
        "status": "unsupported",
        "requested_level": level,
        "tool_permission": tool_permission,
    }


native = compile_level("high", "native", "read_only")
adapted = compile_level(
    "high", "output_only", "read_only", allow_approximation=True
)
unsupported = compile_level("high", "plain", "read_only")

print(native["status"], adapted["status"], unsupported["status"])
print(adapted["request_fields"]["max_output_tokens"], "total-output-cap")
print(native["tool_permission"], adapted["tool_permission"])
```

预期输出：

```text
native adapted unsupported
6144 total-output-cap
read_only read_only
```

输入校验先检查类型再进行集合查找，因此列表等不可哈希值会转成明确的 `ValueError`，而不是泄漏底层 `TypeError`。这段代码只验证适配器如何表达状态和请求字段；它不连接 provider，也不证明后端按预期消耗了推理资源。

## 14.4 用同一任务族测质量与成本

档位比较必须先固定任务集、成功定义、验证器、工具权限、deadline、重试规则和 model revision。设 \(Q(\ell)\) 是档位 \(\ell\) 在固定评测集上的成功率，\(C(\ell)\) 是同一批请求按统一口径统计的成本点，\(\lambda\) 表示每个成本点对应的质量效用折减；因此 \(Q\in[0,1]\)、成本有限且非负、\(\lambda\ge0\)。相邻档位的边际效用可写为：

```math
\Delta U_{\mathrm{low}\rightarrow\mathrm{high}}
=[Q(\mathrm{high})-Q(\mathrm{low})]
-\lambda[C(\mathrm{high})-C(\mathrm{low})]
```

用一组合成教学数据复算：low 的成功率为 \(0.70\)，high 为 \(0.81\)，所以质量增量是 \(0.81-0.70=0.11\)；low 与 high 的成本分别为 1.4 和 5.9 成本点，成本增量为 \(5.9-1.4=4.5\)。若本例设 \(\lambda=0.02\)（每成本点折减 0.02 个效用点），则：

```math
\Delta U = 0.11 - 0.02\times4.5 = 0.02
```

该合成例中升级的净效用为 0.02，刚好为正；若 \(\lambda=0.03\)，结果是 \(0.11-0.135=-0.025\)，则该评测目标下升级不合算。权重来自产品目标，不是模型常数。支付、发布等高风险任务还要保留越权率、严重错误率等硬门槛，不能让平均质量分抵销安全失败。

下面的表也只是合成教学数据，用来说明任务族会改变档位收益。p95 延迟和成本列须按同一任务集、版本与测量窗口复算，不能作为任何产品的经验基线。

| 任务族 | low 成功率 | high 成功率 | low p95 | high p95 | 成本判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| 简单问答 | 0.70 | 0.72 | 1.1 s | 3.9 s | 质量增量小，成本可能不值 |
| 跨文件代码 | 0.55 | 0.76 | 2.8 s | 9.2 s | 需结合成功任务成本判断 |
| 高风险写操作 | 0.63 | 0.71 | 3.0 s | 10.1 s | 仍须通过独立审批和权限门槛 |

跨任务合并的平均分可能掩盖收益只出现在某些任务上。应分别报告任务族的成功率、证据正确率、格式通过率、工具次数、p95、单位成功成本和安全失败；当数据量足以支持时，再给出不确定区间。若只有某个稀有任务从 high 受益，可以由第十八章的在线策略考虑何时升级；本章只规定档位合同和评测方法。

## 14.5 档位比例会改变活跃资源占用

档位提升不只影响单次调用账单，也会改变同时运行请求持有的资源。设 \(N\) 为正整数形式的活跃请求数，\(q\in[0,1]\) 为其中 high 档位的占比，\(M_{\mathrm{low}}\) 和 \(M_{\mathrm{high}}\) 是同一窗口与单位下有限非负的单请求平均资源占用；若没有活跃请求，则总占用为 0，无需定义 \(q\)。用线性均值作初步容量估算：

```math
M_{\mathrm{active}}
\approx N\bigl((1-q)M_{\mathrm{low}}+qM_{\mathrm{high}}\bigr)
```

例如，假设有 100 个活跃请求，25% 使用 high；合成假设 low 每请求占 1 GiB、high 占 3 GiB，则 low 侧有 75 个请求，占 \(75\times1=75\) GiB；high 侧有 25 个请求，占 \(25\times3=75\) GiB；合计约 \(150\) GiB。这里的输入和结果都是玩具估算，实际系统必须用目标模型、批处理、缓存、抢占和工具等待数据替换。

该均值只能帮助估算资源总量，不能推出 p95/p99 延迟，也不能替代队列模拟或负载实验。容量不足时，系统应明示排队、异步、已批准的降级或请求重选；若实际档位改变，就把实际执行状态回传并记入账本，避免继续显示原请求档位。

## 14.6 档位与权限分别决策

档位可能控制文档明确说明的推理或响应资源。产品也可以把某个档位映射为更多验证轮次或更长 deadline，但这属于产品自己的服务契约，必须可观测且经过评测。档位字符串本身不授予身份、数据范围或工具权限。

例如，调用方只持有只读权限时，即使请求 `high`，适配器也只能在该权限下增加允许的分析或验证工作。写入、发送、支付和发布等外部副作用仍须由独立授权规则决定；需要人工确认时，档位不能跳过确认。能力矩阵应同时记录“档位允许的资源”和“外部策略批准的动作”，避免 UI 或日志把两者混成一个值。

取消与超时也属于契约：若客户端 deadline 到期，宿主应停止后续工具动作，并记录请求是否取消成功、工具是否已经产生副作用。若 fallback 改了模型或档位，响应须带上实际 model ID、实际 level 和降级原因。不同供应商的具体状态协议另见第十五章。

## 14.7 升级时回放协议与质量回归

模型、endpoint 或适配器升级后，字段名可能仍相同，事件顺序和 usage 口径却可能变化。回归集应覆盖直接回答、长上下文取证、数学/代码、结构化输出、工具成功与拒绝、取消、超时及多轮状态。每个用例保存准确 model ID/revision、输入、请求档位、返回事件、usage、结束状态、成本和 verifier 结果，才能区分质量回退、计量变化与协议破坏。

shadow 回放可以比较候选档位的输出和账单，但不能再次触发真实工具副作用。涉及写入或外发的案例，应使用无副作用的模拟工具或只比较到动作草稿。评估通过后，才按产品自己的发布门槛逐步切换；门槛由任务成功、尾延迟、成本和独立安全条件组成，不由“请求没有报错”替代。

客户端提示也应选择可验证的语义，例如“使用了外部检索”“通过了单元测试”“需要人工确认”。不要向用户承诺“更聪明”或某个未公开算法。切档位后的策略选择如何利用任务特征和反馈交给第十八章，本章不重复定义在线控制器。

## 14.8 三类公开接口如何解释档位

以下案例展示字段虽相似，合同对象却可能不同。页面会随 provider、model ID 和日期变化；事实均限定在对应官方材料的公开范围内，访问日期为 2026-11-13。

### 14.8.1 Google Gemini：默认值、输出硬上限与 thought state

Google 当前 Thinking 文档将 `gemini-3.8-flash` 列为默认 `medium`，支持 `low`、`medium`、`high`，不支持 `minimal`；`gemini-3.5-flash-lite` 默认 `minimal`，支持 `minimal`、`low`、`medium`、`high`。这张表说明即使同属 Gemini 系列，也不能把一个模型的默认档位直接复制给另一个 model ID。见 [Gemini Thinking 文档](https://ai.google.dev/gemini-api/docs/thinking)及[Gemini 3.5 Flash-Lite 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)。

在 Interactions API 文档中，`max_output_tokens` 是包括 thought tokens 在内的总生成硬上限，不会替模型重新分配 thinking level；若模型在思考中触顶，请求会以 `incomplete` 返回，可能没有完整最终答案，而已生成的 thought tokens 仍计费。官方建议需要降低成本或延迟时先降低 `thinking_level`，不要把总输出上限压到截断答案。usage 可分别检查 `total_thought_tokens` 与 `total_output_tokens`。这些字段对应当前 Interactions API 文档，不应未经核实外推到其它 API surface。

同一 API 的 `thought` step 必须带加密 signature；summary 可缺省或为空。多轮 stateful 请求由服务端管理 signature，stateless 请求则需按文档原样回送 thought blocks。Interaction 的 steps 按时间顺序记录 thought、工具调用/结果及最终 model output。signature 是延续协议所需的 opaque state，summary 是可见摘要；二者都不能当作完整内部推理文本。更完整的跨轮状态分工见第十五章。相关入口：[Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)与[Interactions API 概览](https://ai.google.dev/gemini-api/docs/interactions-overview)。

### 14.8.2 xAI Grok：effort 也可能控制协作规模

xAI 的 reasoning 文档列出 `grok-4.7` 的 `reasoning.effort` 支持 `low`、`medium`、`high`（默认）和 `xhigh`，并说明 reasoning 不能关闭。Responses API 对 `grok-4.7` 会返回 `reasoning.encrypted_content`；后续请求需要把 reasoning item 原样回传，供多轮连续性使用。该字段是协议状态，不是可读 CoT；Chat Completions 不提供同一 ciphertext 字段。来源：[xAI Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)、[Grok 4.7 文档](https://docs.x.ai/developers/models/grok-4.7)。

同一份官方文档对 `grok-4.20-multi-agent` 给出不同的字段语义：`reasoning.effort` 为 low/medium 时使用 4 个协作 agent，high/xhigh 时使用 16 个。xAI 将多 Agent Research 标为 beta；其文档说明更大的协作规模会消耗更多 token 和延迟，子 Agent 的中间状态默认不直接展示给用户。此处的 level 控制 agent 数，不应讲成普通单模型的推理深度档位；协作和验证成本留在第十六章展开。来源：[xAI Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)与[Multi Agent 文档](https://docs.x.ai/developers/model-capabilities/text/multi-agent)。

### 14.8.3 Qwen3-Omni：模型变体不等于单一模型的档位

Qwen3-Omni 技术报告与官方仓库发布 Instruct、Thinking 和 Captioner 等不同变体；报告把 Thinking 描述为面向多模态推理的变体，把 Captioner 描述为从 Qwen3-Omni-30B-A3B 微调出的音频描述模型。它们是用途和训练不同的公开 artifact，不是一个 checkpoint 上 low/medium/high 三个运行档位。来源：[Qwen3-Omni 技术报告](https://arxiv.org/html/2509.17765)、[官方仓库](https://github.com/QwenLM/Qwen3-Omni)。

该报告描述 Thinker 处理多模态输入与文本生成，Talker 生成流式语音；音频 codec 速率为 12.5 Hz，即每秒 12.5 个 codec frame，帧间隔为 \(1/12.5=0.08\) 秒，也就是 80 ms。这个是音频时间粒度，不是推理预算。报告称在冷启动、没有先前上下文时，理论首包延迟为 234 ms；其并发表进一步给出并发为 1 时音频/视频 234/547 ms，并发为 4 时 728/1517 ms，并发为 6 时 1172/2284 ms。作者说明这些是特定 vLLM、torch.compile 和 CUDA Graph 设置下的理论数据，不是 p99，也不是不同 reasoning level 的质量—成本比较。

这个案例提醒我们，分析多模态系统时需分别记录 Thinker 的处理、Talker 的流式输出和任务成功条件。若把 codec 帧率、首包延迟或模型变体名称直接当成 thinking level，就会混淆时间单位、解码架构和推理控制。

## 14.9 小结与练习

thinking level 是具体服务接口上的离散选择。读者要先查 model ID 与 API surface，再识别它控制的对象、总输出上限、usage、事件和状态协议；迁移时用 capability matrix 标记 `native`、有损 `adapted` 或 `unsupported`。档位收益用固定任务族上的质量、成本、尾延迟与安全条件判断，外部权限必须独立决策。第十三章提供计量账本，第十八章处理在线预算策略，第十五章继续讨论状态延续。

练习一：为两个不同后端写一张 capability matrix。每一项标注“官方可确认”“产品自定义”或“未知”，并说明 usage 和状态如何回放。

练习二：某后端只有 `max_output_tokens`，没有 native level。设计一条 `adapted` 响应，解释为什么该字段不能称作 reasoning budget，并给出一个验证其成本与可见答案完整性的实验。

练习三：某档位升级使跨文件代码成功率由 0.55 升至 0.76，p95 从 2.8 秒升至 9.2 秒。列出仍需收集的成本、安全和样本不确定性信息，再判断是否适合默认启用。
