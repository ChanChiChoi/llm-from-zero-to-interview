# 第 15 章 Preserve Thinking：协议状态如何跨轮延续

推理模型在一次请求中生成了 reasoning item，下一次请求怎样接着走？“保存状态”至少涉及两件不同的事：把模型协议要求的历史交回去，以及让宿主确认工具、权限和外部世界的实际状态。前者可能帮助模型沿原上下文继续；后者决定系统能不能安全地继续执行。把二者都叫作“保留思维”，容易把 opaque 状态误当成可读思维链，也容易把旧计划误当成已完成的动作。

## 15.1 先区分三类状态

| 状态 | 例子 | 谁负责 | 保存它能解决什么 |
|---|---|---|---|
| 可见对话 | 用户请求、最终回答、允许展示的工具摘要 | 应用与产品界面 | 让用户理解前因后果 |
| Provider 推理状态 | reasoning item、thinking block、thought signature、续接 ID | 模型服务和 API 客户端按协议协作 | 在受支持的接口上延续原请求的上下文 |
| 宿主执行状态 | 工具调用 ID、外部执行结果、权限版本、artifact 和待确认副作用 | 调用模型的应用、工具服务及其持久化系统 | 判断动作是否执行、是否能重试以及当前是否仍获授权 |

这三类内容可以有关联，但不能互相替代。Provider 状态可能是不透明或加密的协议数据，并不等于可展示的思维链；它通常也不包含文件系统、数据库或浏览器的权威现状。反过来，宿主保存了工具结果，也不代表某个模型 API 能接收或继续使用此前的推理状态。

因此，本章说的 preserve thinking，限定为：按特定 provider、模型和 API surface 的公开协议，保留或引用可续接的模型上下文。它不表示模型拥有永久记忆，不表示客户端可以解密全部内部推理，也不自动恢复工具执行。

## 15.2 续接方式由 API 合同决定

多轮 API 大致有两种续接路径。第一种由服务端保存会话或响应，调用方在后续请求中引用前一状态标识。第二种由调用方管理历史，并按接口要求回放响应项。具体合同可不同，不能把一种供应商的做法当成通用标准。

| API 示例 | 状态如何续接 | 必须留意的边界 |
|---|---|---|
| OpenAI Responses | 可用 Conversations 或 previous_response_id 延续服务端状态；自行无状态回放时，保留响应 output 中的全部项目，包括加密 reasoning item 和 phase 信息。支持的模型还可用 reasoning.context=all_turns。 | 对话对象、响应保存期限和 store 设置有不同语义；新一代模型遇到前缀变化时可能丢弃后续 reasoning item。详细规则见 [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state) 和 [Preserve reasoning](https://developers.openai.com/api/docs/guides/preserved-reasoning)。 |
| Anthropic Messages | 客户端将每轮返回的 content blocks 按原顺序带回，包括 thinking、redacted_thinking 和工具项；部分模型按 system、tools 和此前 messages 检查前缀。 | 是否可回放还取决于生产模型、目标模型及部分模型的账号绑定。历史编辑可能让 block 失效或被丢弃；处理方式按模型和参数而异。见 [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)。 |
| Gemini Interactions | stateful 请求可用 previous_interaction_id，由服务端管理 thoughts 和 signatures；stateless 请求需要按原样回送 thought blocks。 | 此说明针对 Interactions API；其他 Gemini API surface 有自己的字段与回放规则。见 [Interactions 概览](https://ai.google.dev/gemini-api/docs/interactions-overview) 和 [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)。 |
| xAI Responses（Grok 4.7） | 文档说明 Responses 返回 encrypted reasoning content；续接时按文档回传相关项目。 | 这条规则绑定 Grok 4.7 与 Responses API，不能外推到 Chat Completions。见 [Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)。 |

表中合同依据 2026-11-13 阅读的官方文档；接口、模型支持表和存储策略可能变化，实际集成应按目标 model ID 与 API surface 复核。

这些例子说明，“拿到一个状态 ID”与“客户端必须回传完整响应项”是不同接口设计；即便都叫 reasoning 或 thinking，数据结构和兼容条件也不同。发送完整历史不等于把完整历史展示给用户，返回的加密或签名字段也不等于可读取的自然语言推理。Provider 文档说明的是 API 合同，不披露模型内部状态的完整表示。

## 15.3 历史编辑会改变状态的含义

许多续接协议会把推理项与它产生前的上下文绑定。若调用方重写早先的指令、工具 schema 或工具结果，新的输入已经不是原来的上下文。继续带上旧状态，可能被服务拒绝、静默丢弃，或使模型失去原有推理上下文。Anthropic 的前缀检查明确依赖 system、tools 和先前消息；OpenAI 文档也说明，部分较新的模型会在前缀变化后丢弃后续 reasoning item。具体行为以相应模型和 API 文档为准。

这也是为什么“追加新消息”通常比“重写旧历史”更容易满足续接合同。需要删改长历史时，应区分两种操作：

1. 若 provider 为目标 API 提供签名压缩或正式 compaction 协议，按其规定保存压缩块、版本和回放顺序。
2. 若应用自行生成摘要，应把它当成新的、由应用负责的上下文；按 provider 规则丢弃不再兼容的旧推理项，再用摘要、来源引用和明确的未完成事项重新开始。不能宣称旧推理状态仍然连续。

模型切换也不是简单替换 model 字段。目标模型可能无法读取源模型的状态；有的接口会丢弃不兼容项，有的只允许特定兼容对。即使同一个 provider 内切换，也要查看准确 model ID、API surface 和版本。第十四章讨论档位配置，本章讨论状态续接，两者都不能用一个跨模型通用映射来代替实测。

## 15.4 模型续接与工具恢复是两条链

考虑一个测试工具：宿主发出运行请求后，网络断开，客户端没有收到响应。此时至少有两个未知：模型 API 是否保存了可续接状态；测试服务是否已经收到并执行请求。前一个问题由 provider 协议回答，后一个必须向测试服务查询，或通过工具自己的幂等键防止重复提交。

| 宿主观察到的动作状态 | 可以得出的结论 | 恢复时的动作 |
|---|---|---|
| 尚未发送 | 已知外部工具没有收到本次提交 | 重新检查授权后再发送 |
| 已发送或执行中 | 请求可能正在执行，不能当成失败 | 查询原 call ID 或幂等键，等待权威结果 |
| 已成功或已失败 | 有工具侧结果可以读取 | 校验结果来源与目标任务，再更新任务状态 |
| 未知 | 当前证据不足以判断外部结果 | 继续查询、安全暂停或转人工；不盲目重发 |

“未知”描述的是宿主目前掌握的信息，不是工具执行的一种确定结果。若调用方把超时直接当失败，下一轮可能重复发信、付款、发布或写文件。若旧授权已经撤销，保存下来的状态也不能自行恢复授权；每个新的外部副作用都要由当前策略和执行器重新判断。

这里仅划清责任边界：Provider item 负责模型上下文的协议续接；事件顺序、工具结果、幂等和错误恢复由工具协议与宿主处理。工具交错生命周期见[第十九章](19-interleaved-thinking与工具协议.md)，长任务 Runtime 与 checkpoint 实现见[第二十册第十七章](../../book-20-agent-harness-runtime/chapters/17-long-running-agent与checkpoint.md)。

## 15.5 保存边界还包括隐私和生命周期

不透明状态不等于无敏感性。它可能受模型、消息前缀、账号、租户或 provider 存储策略约束；响应 ID 也不等于永久可取回的记忆。以 OpenAI 文档为例，Responses 对象默认保存 30 天，可用 store=false 关闭该项保存；Conversation 对象及其项目不受同一 30 天 TTL 规则约束。实际部署仍需核对账户设置和适用的数据保留条款，不能只凭字段名判断数据生命周期。以上行为依据 2026-11-13 查阅的[官方状态文档](https://developers.openai.com/api/docs/guides/conversation-state)。

宿主应按任务需要保存最少且可核验的信息：provider 与 model 标识、接口版本、续接 ID 或原始响应项、事件与 artifact 的引用、外部执行状态、授权版本以及保留期限。凭证、无关个人信息和完整的用户可见之外推理文本，不应因为“方便续接”而无条件落盘。用户需要解释时，给出经过验证的结论、来源和暂停原因；不要把内部协议字段直接当成解释文本。

## 15.6 用中断演练衡量能否正确续接

只测正常的多轮请求，看不出恢复能力。应在模型响应前后、工具已提交后、权限变化后、历史压缩后和模型切换后设置中断，检查恢复后的 artifact、工具副作用、授权判断和用户可见状态。一个任务只有同时满足以下条件，才计为正确恢复：模型上下文通过目标 API 的兼容检查；外部执行状态已确认或被安全地保持为未知；新副作用重新检查当前授权；没有重复提交；最终 artifact 满足任务契约。

令中断任务总数为 N_{interrupted}，其中恢复后完成任务契约、且满足上述条件的任务数为 N_{correct\_resume}。只有中断样本数大于零时，恢复率才有定义：

~~~math
R_{\mathrm{resume}}
=\frac{N_{\mathrm{correct\_resume}}}
{N_{\mathrm{interrupted}}}
~~~

例如一次演练中断了 100 个任务：82 个恢复后通过 artifact、权限和副作用检查；12 个因外部状态未知而安全暂停；6 个出现错误结果或恢复不安全。此时 N_{correct_resume}=82、N_{interrupted}=100，代入得到 R_resume=82/100=0.82，即 82%。12 个安全暂停不算成功续接，但应另报安全暂停率 12%；否则单看恢复率，可能鼓励系统冒险继续执行。若演练中没有中断任务，分母为零，恢复率应记为 N/A，而不是 0% 或 100%。

恢复率还不能替代分项指标。至少应同时记录重复副作用数、越权动作数、错误 artifact 数、状态不兼容率、人工接管比例和恢复延迟。每项都要保留明确分母和“未发生”的表示方法；例如没有任何外部副作用尝试时，重复副作用率应记为 N/A。上述是评测设计，不是行业统一 SLO。

## 15.7 一个最小的前缀兼容性演示

以下 Python 例子用哈希比较保存时和续接时的上下文前缀，演示“只追加”与“编辑旧内容”会得到不同检查结果。它刻意采用严格的教学合同，不模拟任何 provider 的签名、加密、模型兼容矩阵、compaction 或账号规则；真实请求必须按对应官方文档处理。

**代码状态：已验证（2026-11-13；CPython 3.12.9；仅 Python 标准库）。**

~~~python
import hashlib
import json
from dataclasses import dataclass


def prefix_digest(system, tools, messages):
    prefix = {
        "system": system,
        "tools": tools,
        "messages_before_state": messages,
    }
    encoded = json.dumps(
        prefix,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SavedState:
    provider: str
    model: str
    prefix_hash: str


def check_replay(saved, provider, model, current_prefix_hash):
    if provider != saved.provider:
        return "provider_mismatch"
    if model != saved.model:
        return "model_mismatch"
    if current_prefix_hash != saved.prefix_hash:
        return "prefix_mismatch"
    return "compatible"


system = "只查公开文档，保留来源。"
tools = [{"name": "search", "schema_version": 1}]
messages = [{"role": "user", "content": "查找接口状态规则"}]
saved_hash = prefix_digest(system, tools, messages)
saved = SavedState("Provider-A", "Model-X", saved_hash)

append_only = check_replay(
    saved, "Provider-A", "Model-X",
    prefix_digest(system, tools, messages),
)
edited_history = check_replay(
    saved, "Provider-A", "Model-X",
    prefix_digest(
        system,
        tools,
        [{"role": "user", "content": "查找另一个接口状态规则"}],
    ),
)
changed_tools = check_replay(
    saved, "Provider-A", "Model-X",
    prefix_digest(system, [{"name": "search", "schema_version": 2}], messages),
)
model_switch = check_replay(
    saved, "Provider-A", "Model-Y", saved_hash,
)

assert append_only == "compatible"
assert edited_history == "prefix_mismatch"
assert changed_tools == "prefix_mismatch"
assert model_switch == "model_mismatch"

print(
    f"append_only={append_only} "
    f"edited_history={edited_history} "
    f"changed_tools={changed_tools} "
    f"model_switch={model_switch}"
)
~~~

预期输出：

~~~text
append_only=compatible edited_history=prefix_mismatch changed_tools=prefix_mismatch model_switch=model_mismatch
~~~

例子中没有真实 reasoning payload；它只展示调用方可计算的上下文指纹。实际 provider 可能允许特定模型切换、签名 compaction 或显式丢弃不兼容状态，所以这段代码不能用来判定真实请求是否会成功。

## 15.8 本章结论与练习

跨轮推理状态是带有模型和历史约束的 API 协议对象，不是可以随意编辑的普通文本，也不是永久记忆。正确恢复还需要宿主核实工具的权威结果、当前权限和 artifact。设计时先识别状态归属和 API surface，再决定服务端续接、精确回放或安全重建；遇到未知外部结果时，宁可暂停，也不把超时猜成失败。

1. 选择 OpenAI Responses、Anthropic Messages 或 Gemini Interactions，画出一次工具调用前后需要回传或引用的状态，并标出服务端状态与客户端状态。
2. 构造一个历史前缀被编辑、一个工具结果超时的恢复案例，分别判断 provider 状态兼容性和工具执行状态；说明为何一个判断不能替代另一个。
3. 为含个人数据的任务设定 reasoning item、工具原文、artifact 和审计元数据的访问者、保留期与删除方式，并写出模型切换及 provider 不兼容时的降级策略。
