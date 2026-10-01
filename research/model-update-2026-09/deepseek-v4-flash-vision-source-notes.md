# DeepSeek V4 Flash Vision：多模态 API、榜单锚点与版本路由漂移

首次核验日期：2026-09-15；当前时点复验：2026-09-24。本笔记只把 Artificial Analysis 中已经出现的 `deepseek-v4-flash-vision` 作为模型锚点；DataCurve DeepSWE 当前页面没有该条目。DeepSeek 官方发布页、Vision/Files/Responses 文档和价格页用于核验这个锚点及其周边技术，不作为新的模型发现入口。

## 1. 锚点身份与两个榜单的证据边界

### Artificial Analysis（2026-09-15 首次快照）

首次抓取的详情页是 [DeepSeek V4 Flash Vision](https://artificialanalysis.ai/models/deepseek-v4-flash-vision)，canonical slug 为 `deepseek-v4-flash-vision`，配置名为 `DeepSeek V4 Flash Vision (Reasoning, Max Effort)`。该快照结构化字段给出：

| 字段 | 页面值 | 正确读法 |
|---|---:|---|
| `releaseDate` | `2026-08-21` | 第三方目录日期，与官方实验发布页日期相符，但仍是 AA 字段 |
| reasoning | `true`，`effort=max` | 请求/评测配置，不是新的基础权重 |
| total/active parameters | `284` / `13` | AA 目录字段；与 V4 Flash 家族公开的 284B/13B 口径相近，不能据此证明视觉变体的完整内部结构 |
| context | `1,000,000` | 配置上下文窗口，不等于百万 token 的召回、延迟或并发保证 |
| Intelligence Index | `35.0122378035969` | AA 自有评测协议下的配置级指数 |
| median output speed | `215.179167697513 tokens/s` | AA 测量字段；当前 API 路由变化后不能直接当作历史实验模型的固定速度 |
| median TTFT | `1.29855545700002s` | 详情页测量字段，受 provider、输入和测量时间影响 |
| API price fields | input `$0.44/M`、output `$1.32/M`、cache hit `$0.014/M` | 页面引用的 provider/API 价格字段；与当前 DeepSeek Pricing 页的峰谷价格不能直接拼接 |

详情页还把该条目的 release 关联到 `DeepSeek V4 Flash 0731`，并标为 proprietary、非 open weights。这里的 `284/13` 与开放 V4 家族模型卡的高层参数口径不能扩展为“V4 Flash Vision 有一份独立公开权重”；当前官方公开入口是 API 实验模型。

### DataCurve DeepSWE

本次通过两个代理获取的 [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 快照中，字符串 `deepseek-v4-flash-vision` 出现次数为 0。页面可见 `deepseek-v4-pro` 与 `deepseek-v4-flash` 的 `mini-swe-agent` 配置，但这些行不能迁移成 Vision 变体的结果。因而本锚点不记录 DataCurve Pass@1，不把 V4 Pro/Flash 的 Agent 分数归因给 V4 Flash Vision。

### 本次断点复验快照

| 页面 | 临时文件 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis 中文首页 | `/tmp/aa-home-1234-20260915-r2.html` | 1,773,715 bytes | `4e089738feed2330ffce50ba7cd4d58141641e647ddbf0e50e1d8c4a3b975cf0` |
| Artificial Analysis Vision 详情 | `/tmp/aa-dsv4vision-1234-20260915-r2.html` | 3,609,278 bytes | `a3ab00557356d09aca3729c2fead3900459ab73b1c63d31b03e7c3267200ea68` |
| Artificial Analysis Vision 详情（8098） | `/tmp/aa-dsv4vision-8098-20260915-r2.html` | 3,609,278 bytes | 同上 |
| DataCurve DeepSWE（1234/8098） | `/tmp/deepswe-1234-20260915-r2.html` | 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| 官方 Vision 发布页 | `/tmp/deepseek-v4vision-official-1234-20260915-r2.html` | 25,247 bytes | `f381bb106aa6d046a7a32015715e4e283320ad1fcf6b608ad439599600cf5784` |

两个代理抓到的 Artificial Analysis 详情页字节和哈希一致；DataCurve 也一致。该一致性只说明本次页面传输可复验，不说明榜单数据永久不变。

### 2026-09-15 当前时点恢复复验（r3）

考虑到夜间工作时段可能造成代理中断，本次再次对同一组入口做完整抓取：

| 页面 | 代理/临时文件 | HTTP/大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis Vision 详情 | `1234` `/tmp/1234-aa-20260915-r3.html` | 200 / 3,609,278 bytes | `a3ab00557356d09aca3729c2fead3900459ab73b1c63d31b03e7c3267200ea68` |
| Artificial Analysis Vision 详情 | `8098` `/tmp/8098-aa-20260915-r3.html` | 200 / 3,609,278 bytes | 同上 |
| Artificial Analysis Vision 详情 | `7890` `/tmp/7890-aa-20260915-r3.html` | TLS EOF / 0 bytes | 无完整快照 |
| DataCurve DeepSWE | `1234`/`8098`/`7890` `/tmp/*-deepswe-20260915-r3.html` | 均 200 / 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| DeepSeek Quick Start | 三个代理 `/tmp/*-ds-20260915-r3.html` | 均 200 / 46,116 bytes | `6e2eb037db92ebef6a8f6408d87c12318c973388d6e27321606bb0e67dd67a6c` |

`1234` 与 `8098` 对 Artificial Analysis 返回逐字节一致的完整页面；`7890` 的大型页面仍在 TLS 阶段 EOF，但它对 DataCurve 和 Quick Start 成功。Quick Start 当前页面仍同时出现旧的 `deepseek-v4-flash-vision-exp`、`retired` 和 `DeepSeek-V4.1-Flash`，与本笔记的 alias 漂移结论一致。该复验确认昨晚的中断没有造成当前可用线路的截断证据，但不把一次成功传输当作榜单永久快照。

## 2. 官方实验发布：Vision 不是“把图片塞进普通聊天”

[DeepSeek-V4-Flash-Vision-Exp Release](https://api-docs.deepseek.com/news/news260821) 的正文标注 2026/08/21，公开描述包括：

- `deepseek-v4-flash-vision-exp` 是实验性多模态模型；发布方称它在 agents、reasoning、world knowledge 等文本能力上对齐 V4-Flash；
- 发布方称其在 multimodal agent benchmark 上相对 V4-Flash 有跃升，并接近 Opus-4.8；这是发布方表述，没有公开完整任务、harness、硬件和原始分数；
- 官方同时发布 DeepSeek Harness 0.1.1，并称对新模型提供开箱支持；这说明模型效果依赖模型、视觉输入、工具循环和 harness 的组合；
- 支持混合文本 + 图像输入，入口包括 Chat Completions、Messages 与 Responses；图像可以用 base64、外部 URL 或 Files API；
- 发布时公告写明图像按 V4-Flash 价格计费、每张最多 384 image tokens；这属于当时实验 alias 的产品规则，不是视觉编码器架构披露；
- Files API 支持图像上传一次后通过 `file_id` 重用，且公告写明免费。

因此，“视觉模型”在面试中应拆成四层：视觉输入协议、图像 token/预处理账本、模型生成的工具动作、宿主环境中的观察与验证。公告足以证明产品 workflow 和 API 能力，不足以证明新增了什么视觉 encoder、projector、训练 loss 或参数结构。

## 3. 当前 API 的版本/路由漂移

2026-09-15 复读 [Quick Start](https://api-docs.deepseek.com/quick_start) 和 [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing) 得到当前状态：

1. 当前示例主模型名为 `deepseek-flash`，价格页的 Model Version 标为 `DeepSeek-V4.1-Flash`，并列出 Vision、1M context 和 maximum 384K output。
2. Quick Start 明确说明旧的 `deepseek-v4-flash` 与 `deepseek-v4-flash-vision-exp` 已 retired；兼容请求由 V4.1-Flash 服务。这个服务端路由有日期语境，客户端仍应记录响应中的实际 model/usage 字段。
3. 当前价格页给出 `deepseek-flash` 的峰谷价：cache hit 为 `$0.003/$0.006` 每百万 token，cache miss input 为 `$0.15/$0.30`，output 为 `$0.60/$1.20`；峰值与非峰值的顺序要结合表头读取。产品页同时提醒价格可调整。
4. AA 历史/第三方详情页的 `$0.44/$1.32` 与当前官方峰谷价不一致，不能强行取平均或认为页面错误；它们可能对应不同时间、provider、旧 alias 或测量配置。面试和实验报告必须写清采集日期、模型 ID、后端路由和价格表版本。

这形成一个重要的模型注册表字段：

```text
requested_model = deepseek-v4-flash-vision-exp
catalog_anchor = deepseek-v4-flash-vision (AA, 2026-08-21)
served_model = <response.model, record at runtime>
api_doc_date = 2026-09-15
route_status = legacy alias -> DeepSeek-V4.1-Flash
```

不记录 `served_model` 就无法判断一次实验究竟测的是历史 Vision 实验模型还是后续 V4.1-Flash 路由。

## 4. Vision guide：图像请求的三条输入路径

当前 [Vision guide](https://api-docs.deepseek.com/guides/vision) 以 `deepseek-flash` 为主要模型示例，支持 JPEG、PNG、GIF 和 WebP。三条路径的系统含义不同：

| 路径 | 适合场景 | 主要风险 |
|---|---|---|
| inline base64/data URL | 单次、小图、请求自包含 | body 膨胀、重复上传、密钥/日志泄露 |
| external `https` URL | 可公开访问的媒体、快速试验 | URL 可用性、内容变化、SSRF/隐私与 provenance |
| Files API `file_id` | 大文件、多轮复用、统一生命周期 | 文件权限、过期、删除、API key 绑定和跨请求审计 |

当前 guide 还公开了 `detail` 级别：`low` 会缩放到约 512×512，`high` 与 `original` 保留原图（high 是兼容别名），`auto` 当前等价于 original。它同时说明图像会按尺寸 resize，并给出约 1024 tokens/image 的当前上限，以及每请求图像数量、单图和请求体大小限制。

“图像 token”不是固定的每张图片常数。对于图片集合 `I`，更可靠的预算写法是：

```math
T_{image}=sum_{i=1}^{|I|} f(H_i,W_i,mathrm{detail}_i,mathrm{resize policy}),
\qquad
T_{request}=T_{text}+T_{image}+T_{tool}+T_{history}.
```

`f` 应由目标 API 的当前 calculator/usage 规则确定；不能用发布公告的 384 或当前 guide 的 1024 去推断任意尺寸、任意 alias、任意 provider 的计费。多图请求还要记录每张图片的顺序、来源和对应回答证据。

## 5. Files API：把媒体生命周期从 prompt 中拆出来

当前 [Files API guide](https://api-docs.deepseek.com/guides/files_api) 把媒体上传变成独立资源：

1. `POST /files` 以 `multipart/form-data` 上传，`purpose=user_data`；响应返回 `file-api-...`、大小、创建时间、文件名和过期字段。
2. 请求可以用 `{"type":"file","file_id":"..."}` 引用同一张图，减少重复上传，也允许大于 inline body 的媒体进入服务。
3. 可以 list/retrieve/delete；文件属于 API key，生命周期与 `expires_after` 绑定。
4. 同一文件还可以在 Anthropic-compatible endpoint 通过 beta Files API 使用，但 content block 形状不同。协议兼容不等于权限、计费和错误语义完全相同。

这给 Agent harness 带来一个可审计的资源图：

```text
local image -> upload -> file_id -> model input -> tool observation
     |            |          |             |              |
  provenance   quota     expiry/key     usage/model     artifact/trace
```

上传成功不等于模型已经看见图片；模型看见图片也不等于工具已经执行。每一层都要有独立事件和失败原因。

## 6. Responses API：视觉观察可以成为工具回灌的一部分

当前 [Responses API guide](https://api-docs.deepseek.com/guides/responses_api) 支持 `input_image`，并给出一个重要的 Agent 形态：工具先产生 `function_call`，宿主执行截图工具，再把 `function_call_output` 中的 `input_image` 回传给模型。抽象循环是：

```text
user task
   -> model function_call(take_screenshot)
   -> host validates permission and executes
   -> function_call_output(input_image)
   -> model observes image and chooses next action
   -> final artifact / another tool call
```

Responses 的流式事件还拆出 reasoning、message、function_call、custom tool、`apply_patch`、output text 和 usage 等类型。对 coding Agent 来说，parser 必须按事件序列和 `call_id` 重建状态，不能只拼接 text delta。

官方文档还列出部分兼容参数会被静默忽略。这是迁移风险：调用方即使得到 HTTP 成功，也可能没有真正启用期望的 tool、reasoning 或 truncation 行为。因此应对请求和响应都做 capability probe，并记录 `unsupported-but-ignored`，而不是只看状态码。

## 7. 可以转成面试答案的新技术点

### 7.1 模型身份要分 requested、catalog 和 served

榜单回答“哪个配置被测”，发布公告回答“哪个实验 alias 曾经发布”，API 响应回答“这次请求由哪个后端服务”。三者不同就必须写三个字段；否则榜单复现、价格分析和线上回滚都可能失真。

### 7.2 多模态 Agent 的瓶颈不只在视觉 encoder

图像 resize/tokenize、上传、文件复用、prefill、视觉证据回灌、工具执行、截图再次输入和最终 artifact 都可能成为瓶颈。应分别测 image preprocessing time、upload time、vision token count、TTFT、tool wait、visual recheck latency 和单位成功成本。

### 7.3 file_id 是资源句柄，不是模型记忆

`file_id` 只代表宿主/API 资源的可引用身份。它受 key、过期、删除和权限约束，不能当作模型永久记忆，也不能把重复引用的网络传输成本自动当作零。模型是否命中、看到了什么、是否依据它作出动作，仍要靠 usage、trace 和证据检查。

### 7.4 self-visual judgment 仍需要外部 verifier

模型可以根据截图或交互结果决定下一步修复，但编译、行为测试、像素/布局 diff、文件可重开、权限和安全策略必须由宿主验证器负责。模型输出的 `function_call` 是意图，不是执行授权。

## 8. 状态与证据等级

当前本锚点具备：Artificial Analysis 条目、官方实验发布页、当前 Vision/Files/Responses/价格文档、研究笔记和可写成面试题的协议/账本技术点，状态为**资料级闭环**。不新增“独立视觉架构”结论，原因是公开资料没有为 `deepseek-v4-flash-vision-exp` 提供专属参数、视觉 encoder 结构、训练配方、完整技术报告或 DataCurve 结果。

可直接说：2026-08-21 发布了 V4 Flash Vision 实验 API；它支持混合文本图像、Files API、Chat/Messages/Responses 和 agent framework；当前旧 alias 已由官方文档说明路由到 V4.1-Flash；图像预算、价格和 served model 必须按文档/响应时间绑定。

不能直接说：Vision 变体一定有独立的视觉 backbone；AA 的 35.012 指数是裸模型能力；DataCurve 的 V4 Flash/Pro 分数属于 Vision；发布方“接近 Opus-4.8”是独立 benchmark 复现；`file_id` 是长期模型记忆；Responses API 的兼容字段都实际生效。

相关正式章节：第二十一册第 81 章的 V4.1-Flash cache/多模态路径、第二十一册第 85 章的本锚点视觉 API 与路由账本，以及第四册第 19 章的模型谱系条目。

## 2026-09-24：Vision alias 路由与当前图片预算复验

通过 `10.24.27.134:7890` 重新获取两榜及 DeepSeek 官方页面。快照如下；SHA-256 用于识别本次页面内容，不代表页面永久不变：

| 页面 | HTTP / 大小 | SHA-256 |
|---|---:|---|
| Artificial Analysis `deepseek-v4-flash-vision` 详情 | 200 / `3,967,107` bytes | `a5e5259ee84eac5aa88915dd6436ba155e265ede940ab663b52c4a681ae43ed7` |
| Artificial Analysis 中文首页 | 200 / `1,781,428` bytes | `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59` |
| DataCurve DeepSWE | 200 / `268,036` bytes | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` |
| DeepSeek V4 Flash Vision 实验公告 | 200 / `22,032` bytes | `56babe7f597f37b31c4440174272d4090f32056902c76e3984be3d54d5864de8` |
| DeepSeek V4.1-Flash 9/10 发布页 | 200 / `24,438` bytes | `f18dc22d37393381b31c9069996138f45aa7b02b08442d43af7c6c57f587bdce` |
| Quick Start / Pricing / Vision / Files / Responses | `48,088` / `23,982` / `80,467` / `62,188` / `57,105` bytes | `7ce9db1b1cc7e2efafe7cbfd57b9d46d240c20399f7bd87672c7e3a5250ccdd0` / `210f102275ccf1a6542f08a3bc9e4b4c7c83278cb74b35217bffa112df6363b2` / `5654a198302edd80aea443fb123d14b14d675ccc76730959dd55414d9baee1e2` / `1a825256f4dfae25b40751044bc069860897e30582d5b492e8edeea96b37297a` / `3af115c64774731d42e29b8b6982b914351d501882aaa9440dfa55c83f549ce0` |

AA 当前仍保留 `deepseek-v4-flash-vision` canonical 条目。Index 为 `34.8390628035969`，median output speed `217.762710468086 tokens/s`，median TTFC `0.970116780000126s`，cost per Intelligence Index task `$0.314372044499279`；1M context 和 `$0.44/$1.32/$0.014` 的 input/output/cache-hit provider 字段仍在页面上。与 9/15 的 Index `35.0122378035969`、速度 `215.179167697513`、TTFT `1.29855545700002s` 相比，按榜单/provider 测量的当前时点变化记录，不推断模型权重或 served route 发生变化。当前 DataCurve 快照中没有精确 `mini_swe_agent_deepseek_v4_flash_vision_*` 行。

9/10 官方 V4.1-Flash 发布页称 V4-Flash 与 V4-Flash-Vision-Exp 已 retired，两个旧名称暂时路由到 V4.1-Flash；该发布页还把路由目标描述为新架构系列中最小、具备 native visual understanding 的模型。这个视觉能力声明属于 V4.1-Flash 服务目标，不能反向当成旧 Vision-Exp 的架构披露。9/24 Quick Start 与 Pricing 进一步明确：旧名仍被接受，但对应模型已退役，请求由 V4.1-Flash 服务，并按 Flash 价格计费。当前首选 API 名 `deepseek-flash`，Pricing 将版本标为 `DeepSeek-V4.1-Flash`，给出 1M context、384K 最大输出；Flash 的峰/谷每百万 token 价格分别为 cache hit `$0.006/$0.003`、cache-miss input `$0.30/$0.15`、output `$1.20/$0.60`，峰时为工作日 01:00–04:00 与 06:00–10:00 UTC（中国法定节假日除外）。这与 AA 对旧锚点列出的 `$0.44/$1.32` 不是同一计费快照或服务身份，不能拼成一个价格。

9/24 Vision guide 仍明确每图经 resize 后最多 `1024` tokens：例如 2000×2000 与 5000×5000 图像可 resize 到相同预算，多图按每张分别计算。请求最多 600 张图；单图 base64/URL 为 32 MiB、`file_id` 为 64 MiB；总请求体 48 MiB，总图片大小不含 `file_id` 图为 64 MiB、含 `file_id` 图最高 200 MiB；单边最大 8192 px，包含至少 15 张图时降为 4096 px。发布公告的 384 image tokens 是 8/21 实验 alias 的历史产品规则，必须与当前 1024 上限及当次 served model 分开。

工程结论不变：排行榜模型名、客户端请求 alias 和响应 `model` 必须分账。官方文档确认 alias 当前会路由至 V4.1-Flash，但本轮没有真实 API key 或已授权 endpoint，未观察实际响应 `model`；也没有获得 Vision 变体独立架构/训练报告或 DataCurve 结果，故不新增独立视觉架构结论。
