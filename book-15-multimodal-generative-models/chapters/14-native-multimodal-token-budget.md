# 第十四章：Native Multimodal 的有效上下文账本——能接收不等于能使用

多模态模型的产品资料经常给出一个很大的 context length，或者写“支持图片、音频、视频和长文档”。这类信息有用，却不能直接回答用户真正关心的问题：一张高分辨率发票、一段长视频和二十轮工具历史同时输入时，模型究竟看到了多少内容？哪些内容被压缩、抽帧或截断？输出和推理还剩多少预算？一次任务的延迟和成本是多少？

本章把这些问题放进一套可计算的账本。重点不是猜某个闭源模型的内部 token 数，而是建立通用的核算方法：先测目标 processor，区分文本 token、媒体 token、协议 token、工具历史和输出预留；再根据任务的信息密度做动态预算；最后报告被保留和被丢弃的证据范围，以及质量、延迟、缓存和单位成功成本的变化。

“Native multimodal”在本章中也不是一个架构保证。它可以表示模型原生接受多种媒体，也可以表示训练时使用多模态数据，或者只是产品接口把多种请求统一起来。除非资料明确披露 tokenizer、processor、主干和输出路径，否则只写公开能力，不倒推内部实现。

## 14.0 一张发票和一段视频的预算事故

用户上传四张高分辨率发票图片、一段 20 分钟会议视频，并要求模型比较发票金额和定位会议中的审批决定。会话里还保留了 20K token 的工具历史。

系统有 32K 的有效上下文上限。若应用只按文本长度计数，可能以为输入还有空间；实际 processor 可能产生：

- 文本：4K token。
- 四张图片：每张 2K 视觉 token，共 8K。
- 视频：抽帧后 6K token。
- 工具历史：20K token。
- 协议和媒体边界：几百 token。
- 输出和推理预留：至少 2K token。

输入和输出相加已经超过预算。更危险的是，服务端可能静默丢弃部分历史或媒体，而用户只看到一个语法完整的答案。

### 14.0.1 初学者先记住四个区别

1. 文件大小不是 token 数。一个压缩得很小的高分辨率 JPEG 仍可能产生很多视觉 token。
2. 输入 token 不是完整请求预算。输出、工具结果和推理预留也会占用上下文。
3. 能上传媒体不等于媒体被完整处理。processor 可能 resize、crop、抽帧或降采样。
4. token 预算不是质量预算。减少 token 可能只损害小字，也可能直接丢掉关键事件。

### 14.0.2 专家要记录的字段

专家会在 trace 中记录：

- 原始媒体 hash、尺寸、时长和采样率。
- processor revision 和实际输出尺寸。
- 每种媒体 token 数、special token 和工具历史长度。
- 输入、输出和总上下文长度。
- 被裁剪的区域、被跳过的页面和未采样的时间段。
- TTFT、TPOT、p95/p99、峰值显存和 cache 命中。
- 关键字段准确率、证据支持率和单位成功成本。

没有这些字段，“模型为什么突然不理解长图片”通常无法定位。

## 14.1 总预算的组成

设一条多模态请求包含文本、图像、音频、视频、协议字段、工具历史和输出预留，总预算可以写成：

~~~math
B_{\mathrm{total}}
=B_{\mathrm{text}}+B_{\mathrm{image}}+B_{\mathrm{audio}}
+B_{\mathrm{video}}+B_{\mathrm{protocol}}+B_{\mathrm{tool}}
+B_{\mathrm{output}}+B_{\mathrm{reason}}
~~~

不同服务对 `B_reason` 的定义可能不同，有的把隐藏推理、草稿或内部计算放在上下文中，有的只在系统层面管理。教材中的公式是资源账本，不代表某个 API 的计费口径。

这条等式要求各项预算使用同一计数单位，并且都是有限的非负数；如果某一项没有被服务公开，就应记录为“未知”，不能默认为零。`B_{\mathrm{total}}` 只是一次请求的资源估计，仍要区分硬性的上下文上限、计费 token、显存占用和实际延迟。若预算中包含输出预留，就不能在输入已经装满后再把输出当成额外空间。

### 14.1.1 文本预算

文本 token 数由 tokenizer、语言、数字、代码、格式和模板决定。系统消息、角色标记、引用、JSON schema 和工具描述也属于文本或协议预算。只计算用户可见文本，会低估真实输入。

### 14.1.2 图像预算

如果图像被切成边长为 `p` 的 patch，分辨率为 `H × W`，一个理想化 patch 数为：

~~~math
N_{\mathrm{patch}}
=\left\lceil\frac{H}{p}\right\rceil
\left\lceil\frac{W}{p}\right\rceil
~~~

如果有 `J` 张图片：

~~~math
B_{\mathrm{image}}=\sum_{j=1}^{J}N_{\mathrm{patch},j}+B_{\mathrm{image\ special}}
~~~

真实 processor 可能使用多尺度缩放、thumbnail、动态分辨率、patch merge、换行 token 或视觉 resampler，因此公式只能作为预算起点。最终数量必须调用目标 processor 或与其完全一致的实现测量。

理想化公式假设 `H>0`、`W>0`、`p>0`，并把不完整 patch 向上取整；无 padding 卷积、裁剪丢弃和动态 tile 会使用不同的网格。`B_{\mathrm{image\ special}}` 应按图片数量和协议版本展开，不能把一项固定常数套到所有多图请求上。

### 14.1.3 音频预算

设音频时长为 `D` 秒，每秒 codec 时间位置为 `q_a`，codebook 数为 `K_a`：

~~~math
B_{\mathrm{audio}}
=\lceil Dq_a\rceil K_a+B_{\mathrm{audio\ special}}
~~~

音频还可能有声道、采样率、说话人、时间戳和转写字段。对实时语音，输入窗口不是一次性固定长度，而是随流式会话持续增长，需要滑动窗口或摘要策略。

该式要求 `D\ge0`、`q_a>0`、`K_a>0`，并且三者有限。`D=0` 可以表示缺失或空窗口，但它不应被解释为已经观察到一段音频；是否保留媒体边界 token 要由协议明确。多声道、重采样和转写字段如果也进入模型，必须单独计入预算，不能只按波形 codec token 估算。

### 14.1.4 视频预算

若视频抽取 `F` 帧，每帧有 `N_frame` 个空间 token，时间和媒体边界开销为 `B_v,special`：

~~~math
B_{\mathrm{video}}=F N_{\mathrm{frame}}+B_{v,\mathrm{special}}
~~~

时空 tokenizer 也可以把连续帧合并，近似写成：

~~~math
N_{\mathrm{video}}
=\left\lceil\frac{T}{p_t}\right\rceil
\left\lceil\frac{H}{p_h}\right\rceil
\left\lceil\frac{W}{p_w}\right\rceil
~~~

`p_t` 增大可以降低 token 数，却可能漏掉短暂动作。视频预算必须同时记录时间覆盖和空间细节。

这里要求 `T,H,W,p_t,p_h,p_w` 为正，并明确 `T` 是实际参与处理的帧或时间长度。`F=0` 的空视频不能被当成“完整视频中没有事件”；它应记录为未覆盖或缺失媒体。不同抽帧策略得到的 `F` 即使相同，也可能覆盖不同时间位置，因此 token 数不能代替 coverage report。

### 14.1.5 文档和工具预算

文档可能同时包含页面视觉 token、OCR 文本、表格结构、页码和引用坐标。工具历史包含函数 schema、调用参数、返回结果、错误和确认记录。工具 token 有时比用户问题更长，且不能随意截断，否则当前计划可能缺少关键上下文。

## 14.2 计算一个完整请求

考虑一个教学配置：

- 文本：4200 token。
- 四张 `672 × 672` 图片，patch 为 14。
- 10 秒音频，每秒 50 个时间位置，4 个 codebook。
- 32 帧视频，空间 `224 × 224`，patch 为 16，时间 patch 为 2。
- 工具历史：20000 token。
- 协议：64 token。
- 输出预留：2048 token。

图片 token：

~~~math
N_i=4\times\left\lceil\frac{672}{14}\right\rceil^2
=4\times48^2=9216
~~~

音频 token：

~~~math
N_a=\lceil10\times50\rceil\times4=2000
~~~

视频 token：

~~~math
N_v=\left\lceil\frac{32}{2}\right\rceil
\left\lceil\frac{224}{16}\right\rceil^2
=16\times14^2=3136
~~~

总预算为：

~~~math
B_{\mathrm{total}}
=4200+9216+2000+3136+20000+64+2048
=40664
~~~

如果有效上下文为 32768，这份请求不能完整保留。系统必须在输入前做预算路由，而不是等服务端静默截断。

这里的算术假设图片、音频和视频都按本节的理想化规则计数，尚未加入图片边界、音频边界、视频时间边界和真实模板的额外 token。因此 `40664` 是教学配置下的下界式估算；调用真实 processor 后若数量更大，路由器必须以实测值为准。

## 14.3 上下文上限和有效能力不是一回事

“上下文上限”至少可能指四种不同口径：

1. API 接受的最大序列长度。
2. 模型训练时见过的最大序列长度。
3. processor 产生的媒体和文本序列能否放入运行时。
4. 在给定成本、延迟和任务质量下实际可用的长度。

它们不能互相替代。一个服务可以接受 1M token，却在长视频时间定位、跨段比较或多轮证据引用上表现不稳定；一个模型可能限制 API 长度，却对 32K 内的关键字段表现可靠。

### 14.3.1 接收、保留和使用

可以把一次任务的有效能力拆成三个事件：

~~~math
E_{\mathrm{usable}}
=E_{\mathrm{accepted}}
\land E_{\mathrm{retained}}
\land E_{\mathrm{used}}
~~~

`accepted` 表示接口没有拒绝，`retained` 表示关键媒体没有被截断或压缩到不可用，`used` 表示模型的答案确实受到相关证据影响。只有三个事件都成立，才有资格谈“使用了长上下文”。

这里的合取是任务级事件，不是一个可以从 API 状态码直接得到的布尔字段。`retained` 要依赖 coverage、processor 和质量检查，`used` 至少需要反事实、引用或对照实验支持；三项中的任意一项未知，都不应在报告中写成已成立。

### 14.3.2 长度声明的资料边界

模型卡或 API 文档通常能支持上限和输入格式；论文能支持训练上下文或实验设置；目标系统的有效检索、跨段推理和单位成功成本需要独立复测。教材不能把一项产品数字写成所有任务都拥有同样有效能力。

## 14.4 Processor 是输入协议的一部分

同一张图片在不同 processor revision、resize、颜色空间、裁剪、patchifier 和多尺度策略下，可能产生不同 token 数和不同视觉证据。多模态系统不能只把 processor 当作前端工具，它是模型输入协议的一部分。

### 14.4.1 可回放 manifest

一个可复现请求至少保存：

```json
{
  "media_digest": "sha256:...",
  "media_kind": "invoice_page",
  "original_shape": [2048, 1536],
  "processed_shape": [672, 672],
  "processor_revision": "processor-r7",
  "sampling": "full_page_plus_local_crop",
  "media_tokens": 2304,
  "special_tokens": 12,
  "model_revision": "model-r12",
  "tenant": "tenant-a"
}
```

只保存原始 URL 无法说明线上是否发生了缩放、裁剪、转码、抽帧或颜色转换。发生事故时，manifest 是判断“模型没看懂”还是“模型没有收到关键证据”的基础。

### 14.4.2 Processor 回归

每次升级 processor 都应使用固定媒体集回归：

1. 输出尺寸和帧数。
2. 各模态 token 数。
3. placeholder 和特殊 token 结构。
4. 位置 schema 和 attention mask。
5. OCR、grounding 和关键字段质量。
6. TTFT、峰值显存和缓存命中。

文件格式能成功解析不等于 processor 产生了模型期望的序列。

## 14.5 分辨率、帧率和细节的取舍

### 14.5.1 分辨率翻倍为什么可能带来四倍 token

如果 patch 大小不变，`H` 和 `W` 同时翻倍，则：

~~~math
N'_{\mathrm{patch}}
\approx(2H/p)(2W/p)
=4N_{\mathrm{patch}}
~~~

全注意力的二次项可能接近十六倍数量级。实际实现可能有局部注意力和压缩，因此不能把这个比例直接当成耗时，但它解释了为什么高分辨率必须按任务路由。

这个近似要求原始网格足够大，且两次请求采用相同的取整、边界 token 和连接模式。若 `H` 或 `W` 不能被 `p` 整除，精确比值应使用两个 ceiling 后的网格计算；若 processor 先 resize 到固定画布，原始分辨率翻倍甚至可能不改变模型 token 数。

### 14.5.2 帧率和时间覆盖

长视频减少帧数可以节省 token，却可能漏掉短暂事件。若视频长度为 `D` 秒、抽样率为 `r` 帧/秒：

~~~math
F=\lceil Dr\rceil
~~~

对动作识别，帧间变化和时间覆盖比每一帧的极高分辨率更重要；对合同视频中的一张小屏幕，局部高分辨率可能更重要。预算策略应由问题的信息密度决定。

`D` 应为有限的非负时长，`r` 应为有限的正采样率。`D=0` 时没有可覆盖的时间区间；若视频存在但采样结果为零帧，系统应返回缺失或不可评估，而不是把事件召回率记为零后继续比较。

### 14.5.3 音频时间率和 codebook

增加 codec 时间率和 codebook 数通常能保留更多声学细节，却会增加序列和解码成本。语音内容理解可能只需低码率语义 token，音色和实时 speech-to-speech 则可能需要更高保真表示。不能用“每秒 token 越少越好”作为单一目标。

## 14.6 动态预算分配

静态把预算平均分给文本、图像、音频和视频通常不合理。设总预算为 `B_total`，模态 `m` 的任务重要度为 `I_m`，成本权重为 `w_m`，一个教学上的分配形式为：

~~~math
B_m
=\frac{w_mI_m}{\sum_jw_jI_j}B_{\mathrm{media}}
~~~

该分配式要求 `B_{\mathrm{media}}\ge0`，每个 `w_m` 和 `I_m` 都是有限的非负数，且 `\sum_jw_jI_j>0`。如果所有重要度或成本权重都为零，分母没有定义；工程实现应拒绝配置或采用显式的备用分配规则，而不是产生 NaN。这个式子也只分配预算，不保证分配后的 token 能覆盖关键证据。

这不是生产算法，而是说明预算应随任务变化。发票 OCR 应增加局部图像预算；视频摘要可以降低帧率；语音打断任务应保留最近音频窗口；历史工具结果则可以按引用相关性压缩。

### 14.6.1 预算路由的三步

1. 低成本预览：识别媒体类型、疑似关键区域和时间段。
2. 任务相关加密：对关键区域或候选时间段提高分辨率/采样率。
3. 输出前检查：确认关键证据覆盖、输出预算和安全条件。

预算路由必须对用户可见或在内部记录降级范围。只看了视频的 10% 帧，不能把答案写成“完整视频中没有该事件”。

## 14.7 长文档和长视频的两阶段处理

### 14.7.1 长视频例子

20 分钟视频以每秒 2 帧粗扫：

~~~math
F_{\mathrm{coarse}}=20\times60\times2=2400
~~~

若检索到一个 30 秒候选窗口，再以每秒 8 帧精采样：

~~~math
F_{\mathrm{fine}}=30\times8=240
~~~

精采样省掉了对全部视频的高密度编码，但它不能证明候选窗口之外没有事件。输出要带覆盖说明、候选规则和未覆盖范围。

### 14.7.2 长文档例子

合同可以先按页和区域建立 OCR/视觉索引，再根据问题检索付款条款、签署页和版本号。检索表示适合发现候选，回答表示必须保留原文、坐标和版本。摘要可以帮助定位，不能自动替代高精度证据。

### 14.7.3 覆盖报告

每次选择性输入都产生 coverage report：

```text
媒体总范围：
实际处理范围：
采样间隔：
保留的页面/区域：
未覆盖的页面/时间段：
估算 token：
实际 token：
关键字段是否覆盖：
是否允许自动结论：
```

覆盖报告不是给用户增加负担，而是避免模型把“没看到”误说成“没有发生”。

## 14.8 Attention、KV 和内存成本

如果统一序列长度为 `N`，全 self-attention 的注意力矩阵数量级为：

~~~math
C_{\mathrm{attn}}\propto N^2
~~~

在 decoder 推理中，历史 token 还会占用 KV cache。设层数为 `L`、KV 头数为 `H_kv`、每个 head 维度为 `d_h`、每个元素占 `b` 字节，粗略 KV 大小为：

~~~math
M_{\mathrm{KV}}
\approx2LNH_{\mathrm{kv}}d_hb
~~~

因子 2 表示 key 和 value。真实系统还会考虑 batch、分页 KV、量化、共享 KV 和内核布局，但线性依赖 `N` 仍然是重要趋势。

上述 KV 近似要求 `L,N,H_{kv},d_h,b` 为有限的非负或正参数，并且明确 `b` 是每个元素的字节数。`N=0` 时没有历史 cache，但一个可执行请求通常仍需要协议或输入 token；batch 维度、beam 数、跨请求共享和多层不同 KV 结构应单独计入，不能把该式当作完整显存账单。

### 14.8.1 Prefill 和 decode

媒体 token 往往集中在 prefill 阶段，影响首 token 延迟和显存峰值；输出 token 影响 decode 时间和持续带宽。一个请求可以出现“输入很长、输出很短”或“输入不长、音频输出很长”两种完全不同的瓶颈。

应分别记录 TTFT、TPOT、总延迟和输出长度，不能只报告一次端到端平均耗时。

## 14.9 Cache：同一媒体不一定是同一表示

媒体缓存至少有四层：原始媒体、processor 输出、视觉/音频表示、文本 prefix/KV。复用前需要检查：

- 媒体 digest。
- processor revision。
- resize、crop、帧采样和音频窗口。
- position schema 和 mask。
- model revision。
- tenant 和权限边界。

一个可审计的 key 可以写成：

~~~math
K=H(\mathrm{digest},\mathrm{processor},\mathrm{resize},
\mathrm{sampling},\mathrm{model},\mathrm{position},\mathrm{tenant})
~~~

公共图片的视觉 embedding 可以共享，不代表包含私有问题、工具结果和用户上下文的 KV 可以跨租户共享。缓存命中率高但答案变旧，通常是失效字段缺失；命中率低但每次重新编码，通常是 key 过度包含请求级字段。

## 14.10 超预算时的回退语义

预算不足时可以：

1. 缩小图片或只保留局部 crop。
2. 降低视频帧率或先检索候选时间段。
3. 压缩音频或缩短实时窗口。
4. 摘要工具历史并保留引用。
5. 分批处理或转异步。
6. 请求用户缩小范围或转人工。

每种回退都有不同信息损失。静默按序列末尾截断尤其危险，因为可能删除否定条件、页码、时间限定或工具权限。回退响应应明确说明覆盖范围和不确定性。

## 14.11 有效能力的质量曲线

设预算配置为 `b`，任务质量为 `Q(b)`，延迟为 `L(b)`，成本为 `C(b)`，证据支持率为 `R(b)`。部署选择不是让 `Q` 无条件最大，而是在约束下选择：

~~~math
b^*=\arg\max_b Q(b)
\quad\mathrm{s.t.}\quad
L(b)\le L_{\max},\quad
C(b)\le C_{\max},\quad
R(b)\ge R_{\min}
~~~

`Q` 应拆为关键字段准确率、OCR、grounding、时间定位、引用支持和安全处置。主题回答保持不变，不代表数字、坐标和事件证据保持不变。

这个优化问题还要求候选预算集合非空，且 `Q(b)`、`L(b)`、`C(b)`、`R(b)` 在候选配置上可比较并且有限。若没有任何配置同时满足延迟、成本和证据约束，`\arg\max` 没有可接受解，系统应回退、转人工或返回无法完成，而不能从不可行集合中挑一个“最优”配置。

### 14.11.1 实验矩阵

固定模型、processor、硬件和并发，逐步改变：

- 图片分辨率和裁剪。
- 视频帧数和时间 patch。
- 音频窗口和 codec rate。
- 文本历史和工具结果。
- 输出上限和缓存状态。

每个配置记录 token、编码时间、TTFT、TPOT、p95、峰值显存、关键字段、引用、失败和单位成功成本。这样得到的是一条质量—成本曲线，而不是一个脱离条件的“支持长度”。

## 14.12 Native multimodal 的证据边界

### 14.12.1 可以确认什么

官方 API 或模型卡可能明确说明：

- 支持哪些输入媒体和格式。
- 最大文件大小、时长或上下文上限。
- 是否支持流式输入和输出。
- 公开的 processor 或使用方式。

论文或技术报告可能进一步说明：

- 模态 tokenizer 或 patch 表示。
- 训练时的序列和目标。
- 公开实验的 token 或压缩设置。

### 14.12.2 不能直接推出什么

不能仅凭“原生多模态”推出：

- 没有专用 encoder。
- 所有模态使用同一 tokenizer。
- 输入可以等精度地使用到最大上下文。
- 长视频每一帧都被完整读取。
- 图像、音频和视频 token 与文本 token 成本相同。
- 工具执行是端到端模型内部行为。

资料没有披露的内部细节，应该保留限定语；目标系统的有效能力要通过 processor、任务集和延迟/质量实验测量。

## 14.13 案例：发票 OCR 的高分辨率回退

输入是一张 `2048 × 1536` 的发票。若 patch 为 32，理想 patch 数为：

~~~math
N=\left\lceil\frac{2048}{32}\right\rceil
\left\lceil\frac{1536}{32}\right\rceil
=64\times48=3072
~~~

如果整张图片高分辨率输入超过预算，系统可以先用缩略图定位“总额”“日期”“币种”等字段，再对这些区域做局部 crop。局部高分辨率路径的输出必须保留原页面坐标，方便引用和人工复核。

如果局部检索没有找到金额，系统不能把低分辨率结果当成“发票没有金额”。正确处理是扩大区域、请求更清晰图片或转人工。

## 14.14 案例：长视频的时间覆盖

如果 20 分钟视频粗采样每秒 2 帧，采样间隔为 0.5 秒。持续 0.2 秒的事件可能完全落在两个采样点之间。模型回答“没有发生”之前，需要估计采样策略对该事件的召回能力。

可以引入短事件测试集，改变事件持续时间、发生位置、运动速度和背景相似度，记录：

- 事件召回率。
- 时间定位误差。
- 未覆盖事件比例。
- 精采样触发率。
- 平均 token 和延迟。

这比单独报告长视频能否被接口接收更接近有效能力。

## 14.15 案例：多模态 RAG 的预算边界

多模态 RAG 不应先把所有候选页面、截图和 OCR 文本无差别拼入上下文。更可靠的流程是：

1. 保存媒体 provenance 和版本。
2. 用便宜表示检索候选页面和区域。
3. 根据问题类型分配精细视觉和文本预算。
4. 把页码、坐标、时间和来源一起传给推理模型。
5. 对最终 claim 做证据验证。

检索摘要适合发现候选，原始区域适合回答高精度问题。两者的表示不能混为一谈。

## 14.16 最小可运行的多模态预算审计

下面的 Python 示例模拟一个超预算请求和一个预算回退路径。它不会调用真实模型，只计算不同媒体配置的 token，并把降级造成的覆盖风险显式记录出来。

```python
import math
from math import ceil


def ceil_div(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("ceil_div arguments must be integers")
    if a <= 0 or b <= 0:
        raise ValueError("ceil_div arguments must be positive")
    return (a + b - 1) // b


def require_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def require_positive_int(value, name):
    value = require_nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def validate_sample_config(sample, config):
    required_sample = {"text_tokens", "images", "audios", "videos", "tool_history"}
    required_config = {
        "image_patch", "video_patch", "video_temporal_patch", "audio_rate",
        "audio_codebooks", "protocol_tokens", "output_tokens", "context_limit",
        "fallback_image_size", "fallback_audio_seconds", "fallback_video_frames",
        "fallback_tool_history",
    }
    if set(sample) != required_sample or set(config) != required_config:
        raise ValueError("sample or config has an invalid schema")
    require_nonnegative_int(sample["text_tokens"], "text_tokens")
    require_nonnegative_int(sample["tool_history"], "tool_history")
    if not isinstance(sample["images"], list) or not isinstance(sample["audios"], list) or not isinstance(sample["videos"], list):
        raise TypeError("media collections must be lists")
    require_positive_int(config["image_patch"], "image_patch")
    require_positive_int(config["video_patch"], "video_patch")
    require_positive_int(config["video_temporal_patch"], "video_temporal_patch")
    require_positive_int(config["audio_rate"], "audio_rate")
    require_positive_int(config["audio_codebooks"], "audio_codebooks")
    require_nonnegative_int(config["protocol_tokens"], "protocol_tokens")
    require_nonnegative_int(config["output_tokens"], "output_tokens")
    require_positive_int(config["context_limit"], "context_limit")
    require_positive_int(config["fallback_image_size"], "fallback_image_size")
    require_finite_nonnegative(config["fallback_audio_seconds"], "fallback_audio_seconds")
    require_positive_int(config["fallback_video_frames"], "fallback_video_frames")
    require_nonnegative_int(config["fallback_tool_history"], "fallback_tool_history")
    for index, image in enumerate(sample["images"]):
        if set(image) != {"height", "width"}:
            raise ValueError(f"images[{index}] has an invalid schema")
        require_positive_int(image["height"], f"images[{index}].height")
        require_positive_int(image["width"], f"images[{index}].width")
    for index, audio in enumerate(sample["audios"]):
        if set(audio) != {"seconds"}:
            raise ValueError(f"audios[{index}] has an invalid schema")
        require_finite_nonnegative(audio["seconds"], f"audios[{index}].seconds")
    for index, video in enumerate(sample["videos"]):
        if set(video) != {"frames", "height", "width"}:
            raise ValueError(f"videos[{index}] has an invalid schema")
        require_positive_int(video["frames"], f"videos[{index}].frames")
        require_positive_int(video["height"], f"videos[{index}].height")
        require_positive_int(video["width"], f"videos[{index}].width")


def image_tokens(images, patch):
    require_positive_int(patch, "patch")
    return sum(
        ceil_div(image["height"], patch) * ceil_div(image["width"], patch)
        for image in images
    )


def video_tokens(videos, patch, temporal_patch):
    require_positive_int(patch, "patch")
    require_positive_int(temporal_patch, "temporal_patch")
    return sum(
        ceil_div(video["frames"], temporal_patch)
        * ceil_div(video["height"], patch)
        * ceil_div(video["width"], patch)
        for video in videos
    )


def audio_tokens(audios, rate, codebooks):
    require_positive_int(rate, "rate")
    require_positive_int(codebooks, "codebooks")
    for index, audio in enumerate(audios):
        require_finite_nonnegative(audio["seconds"], f"audios[{index}].seconds")
    return sum(ceil(audio["seconds"] * rate) * codebooks for audio in audios)


def budget_plan(sample, config, detail):
    if not isinstance(sample, dict) or not isinstance(config, dict):
        raise TypeError("sample and config must be mappings")
    validate_sample_config(sample, config)
    if detail not in {"full", "fallback"}:
        raise ValueError("detail must be 'full' or 'fallback'")
    if detail == "full":
        images = sample["images"]
        audios = sample["audios"]
        videos = sample["videos"]
        tool_history = sample["tool_history"]
    else:
        images = [
            {"height": config["fallback_image_size"], "width": config["fallback_image_size"]}
            for _ in sample["images"]
        ]
        audios = [
            {"seconds": config["fallback_audio_seconds"]}
            for _ in sample["audios"]
        ]
        videos = [
            {"frames": config["fallback_video_frames"], "height": 224, "width": 224}
            for _ in sample["videos"]
        ]
        tool_history = config["fallback_tool_history"]

    counts = {
        "text": sample["text_tokens"],
        "image": image_tokens(images, config["image_patch"]),
        "audio": audio_tokens(audios, config["audio_rate"], config["audio_codebooks"]),
        "video": video_tokens(videos, config["video_patch"], config["video_temporal_patch"]),
        "tool": tool_history,
        "protocol": config["protocol_tokens"],
        "output": config["output_tokens"],
    }
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("budget must contain at least one token")
    warnings = []
    if detail == "fallback":
        warnings.extend([
            "local_small_text_needs_recheck",
            "short_video_event_may_be_missed",
            "audio_history_window_reduced",
        ])
    return {"detail": detail, "counts": counts, "total": total, "warnings": warnings}


sample = {
    "text_tokens": 4200,
    "images": [{"height": 672, "width": 672}] * 4,
    "audios": [{"seconds": 10.0}],
    "videos": [{"frames": 32, "height": 224, "width": 224}],
    "tool_history": 20000,
}
config = {
    "image_patch": 14,
    "video_patch": 16,
    "video_temporal_patch": 2,
    "audio_rate": 50,
    "audio_codebooks": 4,
    "protocol_tokens": 64,
    "output_tokens": 2048,
    "context_limit": 32768,
    "fallback_image_size": 448,
    "fallback_audio_seconds": 5.0,
    "fallback_video_frames": 16,
    "fallback_tool_history": 8000,
}

full = budget_plan(sample, config, "full")
fallback = budget_plan(sample, config, "fallback")
selected = fallback if full["total"] > config["context_limit"] else full
report = {
    "full_total": full["total"],
    "fallback_total": fallback["total"],
    "selected_plan": selected,
    "context_limit": config["context_limit"],
    "coverage_report_required": bool(selected["warnings"]),
    "audit_consistent": (
        full["total"] > config["context_limit"]
        and fallback["total"] <= config["context_limit"]
        and selected["detail"] == "fallback"
    ),
}
for key, value in report.items():
    print(f"{key}={value}")
```

参考输出为：

```text
full_total=40664
fallback_total=20976
selected_plan={'detail': 'fallback', 'counts': {'text': 4200, 'image': 4096, 'audio': 1000, 'video': 1568, 'tool': 8000, 'protocol': 64, 'output': 2048}, 'total': 20976, 'warnings': ['local_small_text_needs_recheck', 'short_video_event_may_be_missed', 'audio_history_window_reduced']}
context_limit=32768
coverage_report_required=True
audit_consistent=True
```

这个结果表达了三个事实：完整方案超预算，回退方案可以放入预算，回退方案伴随明确的证据风险。`audit_consistent=True` 只说明账本和路由规则符合示例设定，不代表回退后的发票 OCR 或视频问答一定正确。

## 14.17 练习：把长度数字变成任务判断

### 练习一：图像 token

计算 `336 × 336`、`672 × 672` 和 `1024 × 768` 在 patch=14 时的理想 token 数，并说明 processor 的 resize 和多尺度策略为什么会改变真实数量。

### 练习二：多模态总预算

给定文本 4K、三张图片各 1.5K、10 秒音频 2K、视频 2K、工具历史 12K 和输出预留 2K，判断 32K 上限是否足够，并说明协议 token 还应如何计算。

### 练习三：有效上下文

设计一个实验，区分“接口接受了长视频”“关键帧被保留”和“模型真正使用了时间证据”。

### 练习四：动态路由

为普通图片描述、发票 OCR、长视频摘要和短事件定位分别选择分辨率、帧率和局部回读策略，列出每条路径的覆盖报告字段。

### 练习五：缓存 key

设计视觉 embedding、projector output、文本 prefix 和完整 KV 四层缓存的 key，说明哪些字段变化后必须失效。

### 练习六：质量曲线

固定模型和数据，改变图片分辨率或视频帧率，记录关键字段准确率、引用支持、TTFT、显存和成本。画出至少一条质量—成本曲线，并解释拐点。

### 练习七：静默截断事故

构造一个把合同最后否定条款截掉的请求。说明为什么主题摘要可能仍然正确，以及如何用 coverage report 防止错误结论。

### 练习八：预算和安全

设计一个包含恶意图片文字的超预算请求。说明回退策略如何保证媒体来源、未覆盖范围和工具权限不会被混淆。

## 14.18 资料与证据边界

本章主要依据前面章节已经核验的论文和公开资料：

1. Gemini: A Family of Highly Capable Multimodal Models，https://arxiv.org/abs/2312.11805
2. Chameleon: Mixed-Modal Early-Fusion Foundation Models，https://arxiv.org/abs/2405.09818
3. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action，https://arxiv.org/abs/2312.17172
4. Emu3: Next-Token Prediction is All You Need，https://arxiv.org/abs/2409.18869
5. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation，https://arxiv.org/abs/2408.12528
6. Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model，https://arxiv.org/abs/2408.11039
7. ViT: An Image is Worth 16x16 Words，https://arxiv.org/abs/2010.11929
8. Hugging Face Transformers multimodal chat templates，https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal

论文和模型卡可以支持公开的 token 化、上下文设置和实验条件；processor/API 文档可以支持某个版本的输入协议；目标系统的有效上下文、质量曲线、缓存行为和单位成功成本必须在目标版本和硬件上独立测量。Native multimodal 是能力或产品标签，不应被写成未公开的内部架构事实。
