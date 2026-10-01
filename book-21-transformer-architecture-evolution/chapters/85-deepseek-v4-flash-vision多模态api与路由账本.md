# DeepSeek V4 Flash Vision：多模态 API、工具回灌与路由账本

> 本章核验日期：2026-09-15。Artificial Analysis 中的 `deepseek-v4-flash-vision` 是本章唯一的模型发现锚点；DataCurve DeepSWE 当前快照没有该条目。历史实验模型的发布信息来自 [DeepSeek-V4-Flash-Vision-Exp 官方公告](https://api-docs.deepseek.com/news/news260821)，当前接口行为来自 [Vision](https://api-docs.deepseek.com/guides/vision)、[Files API](https://api-docs.deepseek.com/guides/files_api)、[Responses API](https://api-docs.deepseek.com/guides/responses_api) 和 [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing) 文档。文中把历史榜单配置、当前服务路由、教学估算和真实 API 实测严格分开。

## 85.1 先从一个会“看截图”却交付失败的 Agent 开始

设想你让 coding Agent 根据一张后台管理页面截图修复前端：

1. 它读入截图，生成 React/CSS 补丁；
2. 它调用浏览器启动页面并截图；
3. 它发现按钮颜色接近了，于是宣布完成；
4. 但按钮点不开，窄屏标题溢出，提交表单后也没有成功提示。

这不是单纯的视觉识别问题。一个可交付的视觉 Agent 至少有四类状态：

```text
image input -> visual evidence -> model decision -> tool action
      ^                                      |
      |                                      v
      +-------- screenshot / observation <- host executor
```

模型负责根据输入和观察提出下一步意图；宿主负责上传或读取媒体、执行工具、控制权限和返回观察；验证器负责判断代码、行为、视觉结果和最终 artifact 是否真的满足任务。任何一层缺失，都可能出现“截图看起来对，但产品不能用”。

DeepSeek V4 Flash Vision 的面试价值正好在这里：它让我们同时讨论多模态输入协议、图像 token 预算、文件句柄、Responses 工具回灌、模型 alias 漂移和 Agent 评测证据，而不必把未公开的视觉 backbone 结构编造成事实。

## 85.2 先确认锚点：榜单配置不是当前 API 后端

### 85.2.1 Artificial Analysis 条目

Artificial Analysis 详情页的 canonical slug 是 `deepseek-v4-flash-vision`，配置名为 `DeepSeek V4 Flash Vision (Reasoning, Max Effort)`。2026-09-24 当前快照字段为：

| 字段 | 页面值 | 应如何解释 |
|---|---:|---|
| 发布日期字段 | `2026-08-21` | 第三方 `releaseDate`，与官方实验公告日期相符 |
| 推理配置 | `reasoning=true`、`effort=max` | 一次测量的推理配置，不是新的基础权重 |
| 参数目录字段 | `284` total、`13` active | 第三方目录口径；不能据此证明 Vision 变体有独立公开权重 |
| 上下文 | `1,000,000` | 配置窗口，不等于百万 token 召回或并发保证 |
| Intelligence Index | `34.8390628035969` | Artificial Analysis 自有评测下的配置级指数 |
| 输出速度 | `217.762710468086 tokens/s` | 页面测量字段，受 provider、输入和时间影响 |
| median TTFC | `0.970116780000126s` | 页面测量字段，不是所有请求的固定首响 |
| 价格字段 | `$0.44/M` input、`$1.32/M` output、`$0.014/M` cache hit | 页面引用的 provider/API 价格字段 |

9 月 15 日快照曾记录 Index `35.0122378035969`、速度 `215.179167697513 tokens/s`、TTFT `1.29855545700002s`。这些是不同采集时点的榜单/provider 数据，不作为模型 revision 变化证据。本次 AA 详情为 `3,967,107` bytes / SHA-256 `a5e5259ee84eac5aa88915dd6436ba155e265ede940ab663b52c4a681ae43ed7`。

这些字段很有用，但它们回答的是“这个榜单配置当时如何被测”。它们没有回答“当前请求由哪个后端处理”。特别是 `release` 字段关联到 `DeepSeek V4 Flash 0731`，而官方当前文档已经把旧 Vision alias 路由到 V4.1-Flash；复现报告必须同时保存请求 ID、模型请求名、响应中的实际 model、API 文档日期和价格表版本。

### 85.2.2 DataCurve 当前没有 Vision 行

2026-09-24 DataCurve DeepSWE v1.1 快照（HTTP 200，`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）中，`mini_swe_agent_deepseek_v4_flash_vision_*` 精确行仍不存在。页面有 V4 Pro 与 V4 Flash 的 `mini-swe-agent` 配置，但它们不能迁移成 Vision 的结果。

因此本章不写 Vision 的 DataCurve Pass@1。若面试官问“它在 DeepSWE 有多少分”，正确回答是：当前保存的页面没有该配置，不能用同系列其他行补齐；需要等榜单出现同名配置，或者自行固定模型/路由/harness 做新实验。

## 85.3 历史发布：实验模型引入的是多模态 Agent 闭环

[官方公告](https://api-docs.deepseek.com/news/news260821) 标注 2026/08/21，公开内容包括：

- `deepseek-v4-flash-vision-exp` 是实验性多模态模型；发布方称它在 agents、reasoning 和 world knowledge 等文本能力上匹配 V4-Flash；
- 发布方称它在多模态 Agent benchmark 上相对 V4-Flash 有提升并接近 Opus-4.8；公告没有给完整任务集、harness、硬件和原始分数，因此这是发布方声明；
- 支持混合文本和图像输入，可以通过 Chat Completions、Messages 与 Responses 调用；
- 图片可以用 base64、外部 URL 或 Files API；
- 公告写明图片按 V4-Flash 价格计费、每张最多 384 image tokens；
- 公告还宣布 DeepSeek Harness 0.1.1 对该模型开箱支持，并提供可以复用 `file_id` 的 Files API。

这里的“新技术”更准确地说是产品和系统闭环：视觉观察进入模型决策，工具负责改变环境，新的观察再回灌模型。公告没有公开视觉 encoder 层数、patch size、projector、视觉训练 loss 或完整训练数据，不能由“Vision”这个名称推导这些字段。

## 85.4 当前服务路由：requested、catalog、served 三个身份

2026-09-24 复读 [Quick Start](https://api-docs.deepseek.com/quick_start)、[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing) 和 [V4.1-Flash 发布页](https://api-docs.deepseek.com/news/news260910)：主模型名为 `deepseek-flash`，Model Version 是 `DeepSeek-V4.1-Flash`，支持 Vision、1M context 和最多 384K 输出。官方写明 V4-Flash-Vision-Exp 已 retired；旧 alias 仍接受，但请求暂时由 V4.1-Flash 服务并按 Flash 价格计费。V4.1-Flash 发布页称路由目标来自新架构系列且具备 native visual understanding；这是对 V4.1-Flash 的产品说明，不是旧 Vision-Exp 的视觉架构披露。Flash 当前峰/谷每百万 token 价格为 cache hit `$0.006/$0.003`、cache-miss input `$0.30/$0.15`、output `$1.20/$0.60`；峰时是工作日 01:00–04:00 与 06:00–10:00 UTC（中国法定节假日除外）。AA 页面仍列旧锚点的 `$0.44/$1.32/$0.014` provider 字段；两者不是可合并的一张价目表。

因此一次实验至少要保留三种身份：

```text
requested_model = deepseek-v4-flash-vision-exp
catalog_anchor  = deepseek-v4-flash-vision (Artificial Analysis, 2026-08-21)
served_model    = response.model (runtime observation)
```

这三个字段可能不同：

| 身份 | 来源 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| requested | 客户端请求 | 调用方希望访问的 alias | 服务端实际使用的权重 |
| catalog | 榜单详情 | 某配置曾被第三方测量 | 当前请求仍走同一后端 |
| served | API 响应/trace | 本次请求返回的模型标识 | 该标识背后的完整架构和训练 recipe |

如果只记录 requested model，alias 更新后会发生一种很危险的假复现：代码看起来完全相同，实际上模型已经换了。这个问题与 prompt、tokenizer、工具 schema、价格和 benchmark 结果都会发生耦合。

## 85.5 图像 token：预算取决于预处理，不是每张图的常数

当前 [Vision guide](https://api-docs.deepseek.com/guides/vision) 支持 JPEG、PNG、GIF 和 WebP，并说明服务端会根据尺寸做图像处理。三种输入路径是：

1. **inline base64/data URL**：请求自包含，适合一次性小图；代价是 body 膨胀和日志泄露风险。
2. **external URL**：请求只携带地址；代价是 URL 可用性、内容变化、SSRF、隐私和 provenance 风险。
3. **Files API `file_id`**：先上传再引用；适合大图、多轮和重复使用，但要管理 key、过期、删除和审计。

当前 guide 给出 `detail` 级别：`low` 会缩放到约 512×512，`high` 与 `original` 保留原图，`auto` 当前等价于 `original`。当前文档给出 resize 后最多 1024 tokens/image；2000×2000 与 5000×5000 图像可能被 resize 到同一预算，多图逐张独立计数。限制还包括每请求最多 600 张，单图 base64/URL 32 MiB、`file_id` 64 MiB，请求体 48 MiB，总图片体积不含 `file_id` 最多 64 MiB、含 `file_id` 最多 200 MiB；单边最多 8192 px，15 张及以上图片时单边限制降至 4096 px。历史实验公告的 384 tokens/image 是另一时点、另一 alias 的产品规则，不能把两个数字合成一个永久模型属性。

对一组图片，更好的预算式是：

```math
T_{image}=sum_{i=1}^{N}f(H_i,W_i,detail_i,resize_i),
\qquad
T_{request}=T_{text}+T_{image}+T_{tool}+T_{history}.
```

其中 `f` 应由目标服务当前的图片 token calculator 或响应 usage 校准。实际容量还要保留：

- 图片顺序与对应证据；
- 文本 placeholder 与 image block 的一一对应；
- tool result 中再次回传的截图；
- 多轮历史中已经复用的图片；
- 视觉 token 与 reasoning/output token 共同占用的 context；
- `detail`、resize、压缩和媒体解码失败。

“图片 token 少”不自动等于“任务便宜”：如果 Agent 因看不清而多执行三轮截图和工具调用，端到端 token、延迟和单位成功成本反而可能上升。

## 85.6 Files API：文件句柄不是模型记忆

当前 [Files API guide](https://api-docs.deepseek.com/guides/files_api) 将媒体生命周期独立出来：

```text
upload -> file_id -> reference in model input -> response/tool trace
    |        |                 |                       |
 quota    expiry/key       token/usage              evidence/audit
```

公开的运行语义包括：

- 通过 `POST /files` 的 multipart 请求上传，通常使用 `purpose=user_data`；
- 返回 `file-api-...`、文件大小、创建时间、文件名和过期字段；
- 在 Chat 请求中以 `{"type":"file","file_id":"..."}` 引用；
- 可以 list、retrieve 和 delete，文件归属于 API key；
- 文件可以在多个请求中复用，适合大于 inline 限制或需要反复观察的图像；
- Anthropic-compatible endpoint 也可使用文件，但 content block 与 header 形状不同。

需要分开验证的四件事：

1. 文件是否上传成功；
2. 本次请求是否成功引用并计入模型输入；
3. 模型是否在输出中使用了该图片证据；
4. Agent 是否基于该输出采取了正确且有权限的动作。

不能因为同一个 `file_id` 可重复发送，就说模型拥有永久视觉记忆；也不能把重复引用的上传成本直接当作零。文件句柄是资源标识，模型状态、上下文状态和宿主权限仍要按请求审计。

## 85.7 Responses API：让截图成为工具回灌

当前 [Responses API guide](https://api-docs.deepseek.com/guides/responses_api) 支持 `input_image`。一个典型的视觉工具循环是：

```text
user task
   -> model: function_call(take_screenshot)
   -> host: permission + browser execution
   -> function_call_output(input_image)
   -> model: visual observation + next action
   -> host: execute / verify / rollback
```

在这里必须区分三种消息：

- **function_call**：模型提出一个带 `call_id`、函数名和参数的动作意图；
- **function_call_output**：宿主执行后返回文本、结构化结果或 `input_image`；
- **最终 message/artifact**：模型给出的解释或交付物，仍需要外部验证。

流式解析也不能只拼接文本。官方文档列出 reasoning、message、function call、custom tool、`apply_patch`、output text 和 usage 等事件。一个可靠的 harness 至少维护：

```text
event sequence -> call_id -> tool arguments -> executor result
             -> image observation -> next model turn -> artifact state
```

工具事件中的参数片段可能跨多个 delta；宿主需要按 call index 或 call ID 聚合，再做 JSON/schema、业务规则、权限、用户确认和超时检查。模型产生 `function_call` 不是工具已执行，工具返回截图也不是页面已经满足验收条件。

## 85.8 API 兼容不等于能力兼容

DeepSeek 文档提供 OpenAI/Anthropic 兼容接口以及 Responses API。兼容主要解决请求形状和 SDK 接入，不保证所有字段的语义完全相同。

当前 Responses 文档特别值得面试时追问：部分不支持的参数可能被静默忽略而不返回错误。于是“HTTP 200”不一定意味着期望的 reasoning、truncation、tool 或 cache 行为已经启用。

迁移时应做 capability probe：

| 检查 | 记录内容 |
|---|---|
| 模型 | requested、served、响应 model |
| 输入 | image block 类型、顺序、detail、file_id、文本 placeholder |
| 推理 | effort、reasoning event、usage 中 reasoning token |
| 工具 | function schema、tool choice、parallel、call_id、超时 |
| 输出 | response event sequence、text/tool/image result |
| 不支持字段 | 是否报错、静默忽略还是降级 |
| 版本 | SDK、API 文档、harness、模型 alias 和时间 |

这也是为什么面试中不能只说“换 base URL 就能把 OpenAI Agent 迁过去”。真正需要迁移的是状态协议、parser、权限策略、失败恢复和 capability contract。

## 85.9 定价和端到端成本：不要只算图片 token

当前价格页给 `deepseek-flash` 的峰谷价格表，cache hit、cache miss input 和 output 分开计费，且峰值时段与非峰值时段不同。Artificial Analysis 详情页给出的 `$0.44/$1.32` 是它当前记录的 provider/API 字段，两者不能无条件拼接。

对视觉 Agent，单位成功成本可以写成：

```math
C_{task}=C_{text}+C_{image}+C_{tool calls}+C_{retries}+C_{storage/transfer},
```

如果只用输出 token 估算，至少会漏掉：

- 图片预处理和上传；
- Files API 生命周期与重复引用；
- reasoning token；
- 截图工具和浏览器等待；
- 失败后重新生成和重试；
- 不同 detail 级别带来的视觉 token 差异；
- Agent 每一轮的上下文和工具结果回灌。

成本报告必须带着 `model/revision/route/provider/effort/harness/task/verifier/time`，否则一个看似更便宜的数字可能只是少算了重试或把失败任务排除了。

## 85.10 零依赖教学 demo：把路由、媒体、事件和验收放进同一账本

下面的代码不访问网络，不调用真实模型，也不实现 DeepSeek 的生产图像 tokenizer。它用明确的教学估算模拟四个面试重点：请求身份、图像预算、Files 生命周期、Responses 事件回放和最终交付门禁。

```python
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class ModelRoute:
    requested: str
    catalog_anchor: str
    served: str


def audit_route(route):
    values = (route.requested, route.catalog_anchor, route.served)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("route identities must be non-empty strings")
    if route.requested.endswith("-vision-exp") and route.served != "deepseek-v4.1-flash":
        raise ValueError("legacy vision alias must record its served model")
    return {
        "requested": route.requested,
        "catalog": route.catalog_anchor,
        "served": route.served,
        "changed_backend": route.requested != route.served,
    }


def estimate_image_tokens(width, height, detail="auto", patch=32, max_tokens=1024):
    if not all(isinstance(value, int) for value in (width, height, patch, max_tokens)):
        raise ValueError("image dimensions and limits must be integers")
    if width <= 0 or height <= 0 or patch <= 0 or max_tokens <= 0:
        raise ValueError("image dimensions and limits must be positive")
    if detail not in {"low", "high", "original", "auto"}:
        raise ValueError("unknown detail")
    if detail == "low":
        width, height = min(width, 512), min(height, 512)
    raw = ceil(width / patch) * ceil(height / patch)
    return min(max(raw, 1), max_tokens)


def request_budget(text_tokens, images, tool_tokens, history_tokens, context_limit):
    fields = (text_tokens, tool_tokens, history_tokens, context_limit)
    if not all(isinstance(value, int) for value in fields):
        raise ValueError("token counts must be integers")
    if any(value < 0 for value in fields) or context_limit <= 0:
        raise ValueError("token counts are out of range")
    image_tokens = sum(
        estimate_image_tokens(image["width"], image["height"], image["detail"])
        for image in images
    )
    total = text_tokens + image_tokens + tool_tokens + history_tokens
    return {"text": text_tokens, "image": image_tokens, "tool": tool_tokens,
            "history": history_tokens, "total": total,
            "over_budget": total > context_limit}


def file_reference(file_id, owner, expires_at, now):
    values = (file_id, owner)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("file identity must be non-empty")
    if not isinstance(expires_at, int) or not isinstance(now, int):
        raise ValueError("timestamps must be integers")
    if expires_at <= 0 or now < 0:
        raise ValueError("timestamps are out of range")
    if now >= expires_at:
        return {"status": "expired", "file_id": file_id, "owner": owner}
    return {"status": "available", "file_id": file_id, "owner": owner}


def replay_events(events):
    if not isinstance(events, list) or not events:
        raise ValueError("events must be a non-empty list")
    calls = {}
    text_parts = []
    image_observations = 0
    for event in events:
        if not isinstance(event, dict) or not event.get("type"):
            raise ValueError("each event needs a type")
        kind = event["type"]
        if kind == "function_call":
            call_id = event.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("function call needs call_id")
            calls[call_id] = {"name": event.get("name"), "output": False}
        elif kind == "function_call_output":
            call_id = event.get("call_id")
            if call_id not in calls:
                raise ValueError("tool output has no matching call")
            output = event.get("output")
            calls[call_id]["output"] = True
            if isinstance(output, list):
                image_observations += sum(
                    item.get("type") == "input_image"
                    for item in output if isinstance(item, dict)
                )
        elif kind == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise ValueError("text delta must be a string")
            text_parts.append(delta)
        elif kind in {"response.reasoning_text.delta", "response.completed"}:
            continue
        else:
            raise ValueError("unsupported event in teaching parser")
    if any(not call["output"] for call in calls.values()):
        raise ValueError("a function call is missing its output")
    return {"calls": len(calls), "images": image_observations,
            "text": "".join(text_parts)}


def release_gate(compile_ok, behavior_ok, visual_ok, artifact_ok, permissions_known):
    checks = (compile_ok, behavior_ok, visual_ok, artifact_ok, permissions_known)
    if not all(isinstance(value, bool) for value in checks):
        raise ValueError("release checks must be booleans")
    return "release" if all(checks) else "repair"


route = ModelRoute(
    requested="deepseek-v4-flash-vision-exp",
    catalog_anchor="deepseek-v4-flash-vision",
    served="deepseek-v4.1-flash",
)
print("route:", audit_route(route))
print("budget:", request_budget(
    text_tokens=120, images=[{"width": 1600, "height": 900, "detail": "auto"}],
    tool_tokens=80, history_tokens=200, context_limit=1024))
print("file:", file_reference("file-api-demo", "api-key-a", 100, 40))
print("events:", replay_events([
    {"type": "function_call", "call_id": "fc1", "name": "take_screenshot"},
    {"type": "function_call_output", "call_id": "fc1",
     "output": [{"type": "input_image", "image_url": "data:image/png;base64,..."}]},
    {"type": "response.reasoning_text.delta", "delta": "inspect"},
    {"type": "response.output_text.delta", "delta": "修复后再验证"},
    {"type": "response.completed"},
]))
print("release_gate:", release_gate(True, True, True, True, True))
```

这段代码中的 `patch=32`、`max_tokens=1024` 和 token 估算只是教学模型，不是 DeepSeek 生产 tokenizer 或计费公式。它的价值在于把容易被忽略的账本显式化：requested/catalog/served 三种身份、图片预算、文件句柄、工具 call/output 对、截图观察以及最终门禁。

## 85.11 Demo 应该如何做边界测试

正常路径只是第一步，至少还要验证：

| 输入 | 应拒绝/返回什么 | 为什么 |
|---|---|---|
| legacy alias 没有 served model | `ValueError` | 无法复现实验后端 |
| 宽或高为 0 | `ValueError` | 没有合法媒体尺寸 |
| 未知 `detail` | `ValueError` | 不能默默采用错误预处理 |
| context limit 为 0 | `ValueError` | 预算没有定义域 |
| expired `file_id` | `expired` | 资源生命周期结束 |
| tool output 没有匹配 call | `ValueError` | 防止伪造观察结果 |
| function call 没有 output | `ValueError` | 工具循环未闭合 |
| 非布尔 release check | `ValueError` | 防止字符串/整数隐式通过 |
| 视觉检查失败但编译通过 | `repair` | 代码正确不等于视觉 artifact 正确 |

面试时可以把这些边界归纳成一句话：视觉 Agent 的正确性不是一个字符串，而是“资源、协议、工具、观察和交付物”五类状态的合取。

## 85.12 如何评估视觉 Agent，而不是只评一张图片问答

一个可复现的评估矩阵至少要拆开以下指标：

1. **输入层**：图片解码成功率、resize/detail、image token、上传时间、文件复用命中率；
2. **协议层**：placeholder 对齐、event sequence、call ID 配对、schema 错误、静默忽略字段；
3. **工具层**：工具调用成功率、权限拒绝、超时、重试、浏览器/终端等待和副作用；
4. **视觉层**：OCR 字段、布局溢出、颜色/位置、交互后的截图差异和视觉证据支持率；
5. **交付层**：编译、行为测试、文件可重开、artifact hash、回滚和 verifier 通过；
6. **效率层**：TTFT、TPOT、总 token、reasoning token、工具轮数、端到端延迟和单位成功成本。

对于任务成功率，可用最简单的定义：

```math
Success=mathbf{1}(code\_ok\land behavior\_ok\land visual\_ok\land artifact\_ok\land permission\_ok).
```

这并不意味着所有业务都必须使用同一套门禁；它强调不能用“模型输出了完成文本”替代行为和 artifact 检查。若某项不适用于任务，应明确记录 `not_applicable`，不能把它当作自动成功。

Artificial Analysis 的 Intelligence Index、TTFT、速度和价格，只是一个第三方配置观测；官方公告的“接近 Opus-4.8”是发布方比较；DataCurve 当前没有 Vision 行。三者不能拼成一个视觉 Agent 总分。

## 85.13 面试追问

### 问题 1：为什么不能只记录 `deepseek-v4-flash-vision-exp` 这个 model ID？

因为它可能是 requested alias，而官方当前文档已经说明旧 alias 由 V4.1-Flash 服务。要记录 catalog anchor、served model、时间、API 文档和响应 usage，否则无法判断复现对象。

### 问题 2：发布公告的 384 image tokens 和当前文档的约 1024 tokens/image 哪个是真的？

两者都可能在各自时间和 alias 语境下成立。前者是 2026-08-21 实验公告对当时实验 alias 的产品规则，后者在 2026-09-24 Vision guide 复核中仍是 resize 后每图上限；不能当作同一模型的永久常数，实时实验要绑定当前路由、detail、resize 和 usage。

### 问题 3：Files API 的 `file_id` 是否等于模型记住了这张图？

不等于。`file_id` 是受 API key、过期、删除和权限约束的资源句柄；请求引用它只说明输入资源可寻址，模型是否使用视觉证据、Agent 是否正确行动还要看 response、trace 和 verifier。

### 问题 4：Responses API 把截图放在 `function_call_output` 里有什么好处？

它把“工具动作”和“视觉观察”接成同一条可回放状态链：模型提出截图动作，宿主执行并检查权限，再把 `input_image` 回灌给模型。这样可以做多轮观察—修复，但必须维护 call ID、事件顺序、工具超时和 artifact 验证。

### 问题 5：OpenAI-compatible API 迁移为什么还需要 capability probe？

兼容通常只保证请求形状，当前文档还说明部分不支持参数可能静默忽略。必须检查模型、输入 block、reasoning/tool event、usage、错误与降级行为，不能只看 HTTP 200。

### 问题 6：为什么“视觉 self-verification”仍不是 verifier？

模型可以根据截图和环境反馈提出下一步修复，但编译、行为测试、视觉 diff、文件可重开、权限和安全策略需要宿主/独立验证器。模型的 call 是意图，不是执行授权。

### 问题 7：没有 DataCurve Vision 行时，怎样回答它的代码能力？

明确说当前 DataCurve 快照没有该配置，不能把 V4 Flash/Pro 的行迁移过来；可以分别报告 Artificial Analysis 的配置指标和官方发布方声明，并说明它们的评测协议不同。

## 85.14 小练习与实验设计

1. 写一个 model registry，分别记录 requested、catalog、served、provider、API 文档日期和 response model；模拟 alias 切换，验证旧实验不能被静默归档为新模型。
2. 固定三张不同分辨率图片，分别使用 `low/high/original/auto`，记录 toy image token、上传字节、视觉证据召回和端到端工具轮数；不要把 toy 计数写成官方计费。
3. 实现 Files API 的 upload/reference/expire/delete 状态机，加入错误 API key、过期文件、重复引用、删除后引用和跨 endpoint content block 的边界。
4. 为 Responses 事件写一个可回放 parser，测试多段 function arguments、错 call ID、截图 output、reasoning delta、静默忽略字段和取消后的幂等清理。
5. 对同一前端任务比较“代码测试通过”和“代码 + 行为 + 视觉 diff + artifact”两套门禁，报告误通过率、工具轮数、视觉修复次数和单位成功成本。
6. 设计 requested alias 与 `response.model` 不同的 A/B 实验，固定任务、effort、工具、provider、时间和 verifier，比较模型路由变化对质量、延迟、价格和错误率的影响。

## 85.15 证据边界与来源

### 可以直接说

- Artificial Analysis 有 `deepseek-v4-flash-vision` 的 `max` 配置；本次页面给出 1M context、第三方指数、速度、TTFT、价格和 284/13 目录字段。
- DeepSeek 官方在 2026-08-21 发布了 `deepseek-v4-flash-vision-exp` 实验 API，支持混合文本图像、Chat/Messages/Responses、base64/URL/Files API，并描述了视觉 Agent workflow。
- 当前官方文档使用 `deepseek-flash` 作为主模型名，并说明旧 Vision alias 由 V4.1-Flash 服务；当前 Vision/Files/Responses 文档定义了图片、文件和工具回灌接口。
- DataCurve 当前快照没有 `deepseek-v4-flash-vision` 行，因此不记录它的 DataCurve 分数。

### 不能直接说

- 不能从 Vision 名称推导独立视觉 encoder、patch size、projector、训练数据或 loss。
- 不能把 AA 的 `35.012` 写成裸模型能力，也不能把 V4 Flash/Pro 的 DataCurve 结果迁移给 Vision。
- 不能把官方“接近 Opus-4.8”写成独立复现；不能把 384 和 1024 写成无时间条件的永久图像 token 上限。
- 不能把 `file_id` 写成模型永久记忆，不能把 function call 写成工具执行成功。
- 不能把 API 兼容、HTTP 200 或静默忽略字段写成所有能力已经生效。

研究证据见 [`deepseek-v4-flash-vision-source-notes.md`](../../research/model-update-2026-09/deepseek-v4-flash-vision-source-notes.md)；DeepSeek V4 家族的压缩注意力、mHC、KV 和多模态模型卡背景见第 77、78、81 章及 [`deepseek-v4-source-notes.md`](../../research/model-update-2026-09/deepseek-v4-source-notes.md)。
