# 第十四章：Native Multimodal 的有效上下文账本——能接收不等于能使用

多模态模型的产品资料经常给出一个很大的 context length，或者写“支持图片、音频、视频和长文档”。这类信息有用，却不能直接回答用户真正关心的问题：一张高分辨率发票、一段长视频和二十轮工具历史同时输入时，模型究竟看到了多少内容？哪些内容被压缩、抽帧或截断？输出和推理还剩多少预算？一次任务的延迟和成本是多少？

本章把这些问题放进一套可计算的账本。重点不是猜某个闭源模型的内部 token 数，而是建立通用的核算方法：先测目标 processor，区分文本 token、媒体 token、协议 token、工具历史和输出预留；再根据任务的信息密度做动态预算；最后报告被保留和被丢弃的证据范围，以及质量、延迟、缓存和单位成功成本的变化。

“Native multimodal”在本章中也不是一个架构保证。它可以表示模型原生接受多种媒体，也可以表示训练时使用多模态数据，或者只是产品接口把多种请求统一起来。除非资料明确披露 tokenizer、processor、主干和输出路径，否则只写公开能力，不倒推内部实现。

## 14.0 一张发票和一段视频的预算事故

设一个合成请求：用户上传四张高分辨率发票图片和一段 20 分钟会议视频，要求比较发票金额并定位会议中的审批决定；会话还保留 20K token 的工具历史。

假设目标服务的上下文合同上限是 32K。若应用只按文本长度计数，可能以为输入还有空间；实际 processor 可能产生：

下面的 token 数是用于演示路由问题的合成值，不对应任何模型、服务或真实发票/视频测量。

- 文本：4K token。
- 四张图片：每张 2K 视觉 token，共 8K。
- 视频：抽帧后 6K token。
- 工具历史：20K token。
- 协议和媒体边界：几百 token。
- 输出和推理预留：至少 2K token。

在这些假设下，输入和输出相加已经超过预算。更危险的是，服务端可能静默丢弃部分历史或媒体，而用户只看到一个语法完整的答案。

### 14.0.1 初学者先记住四个区别

1. 文件大小不是 token 数。一个压缩得很小的高分辨率 JPEG 仍可能产生很多视觉 token。
2. 输入 token 不是完整请求预算。在 API 把这些内容计入同一上下文合同的模式下，输出、工具结果和推理预留也会占用预算。
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

先把同一计数口径下、互不重叠的请求项放入账本。若媒体和工具结果已经计入服务返回的输入 token，就不能再把它们重复相加：

~~~math
B_{\mathrm{total}}
=B_{\mathrm{text}}+B_{\mathrm{image}}+B_{\mathrm{audio}}
+B_{\mathrm{video}}+B_{\mathrm{protocol}}+B_{\mathrm{tool}}
+B_{\mathrm{output}}+B_{\mathrm{reason}}
~~~

不同服务对隐藏推理、草稿或内部计算的计数方式不同。只有当它是独立且不会与输出预留重叠的一项时，才额外记入 `B_{\mathrm{reason}}`；服务没有公开该值时标为未知，不要猜成零。教材公式是同一口径下的请求资源账本，不代表某个 API 的计费规则。

这条等式要求各项预算使用同一计数单位、互不重叠，且都是有限的非负数；如果某一项没有被服务公开，就应记录为“未知”，不能默认为零。若接口只给出一个包含媒体展开结果的输入 token 总数，应采用这个总数并避免再次加上估算的媒体子项。`B_{\mathrm{total}}` 只是一次请求的 token 账本，仍要区分服务上下文合同、计费 token、显存占用和实际延迟。若预算中包含输出预留，就不能在输入已经装满后再把输出当成额外空间；若 provider 将隐藏推理计入输出，也不能重复加一遍 `B_{\mathrm{reason}}`。

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

如果教学配置的上下文上限为 32768，这份请求不能完整保留。示例把 2048 token 作为生成阶段的总预留，没有另行猜测隐藏推理 token；若目标服务将该预留拆成多个计数项，应按其合同改写账本。系统必须在输入前做预算路由，而不是等服务端静默截断。

这里的算术假设图片、音频和视频都按本节的理想化规则计数，尚未加入图片边界、音频边界、视频时间边界和真实模板的额外 token。因此 40664 是教学配置下的理想化估算，不是目标 processor 的实测下界或 provider 计费数；调用真实 processor 后应以实际输入结构和服务返回的用量为准。

## 14.3 上下文上限和有效能力不是一回事

“上下文上限”至少可能指四种不同口径：

1. API 接受的最大序列长度。
2. 模型训练时见过的最大序列长度。
3. processor 产生的媒体和文本序列能否放入运行时。
4. 在给定成本、延迟和任务质量下实际可用的长度。

它们不能互相替代。即使服务接受 1M token，这一字段本身也不能证明长视频时间定位、跨段比较或多轮证据引用稳定；一个模型也可能限制 API 长度，却对 32K 内的关键字段表现可靠。

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

模型卡或 API 文档通常能支持特定版本、区域和模式的输入格式及上限；论文能支持作者报告的训练上下文或实验设置。长上下文实验也表明，输入长度本身不能代替按位置、任务和证据类型测量检索质量；例如 LVOmniBench 将长音视频记忆、时间定位和细粒度理解分开评估，而不是仅以文件能否送入模型作为成功标准（[论文 v1](https://arxiv.org/abs/2603.19217v1)）。目标系统的有效检索、跨段推理和单位成功成本仍需在固定模型、processor、采样协议和任务集上复测。教材不能把一项产品数字写成所有任务都拥有同样有效能力。

## 14.4 Processor 是输入协议的一部分

同一张图片在不同 processor revision、resize、颜色空间、裁剪、patchifier 和多尺度策略下，可能产生不同 token 数和不同视觉证据。多模态系统不能只把 processor 当作前端工具，它是模型输入协议的一部分。Transformers 的多模态模板文档也把对话模板与 processor 组合在一起处理文本、图像、音频和视频输入；这些接口示例说明怎样构造输入，不自动保证模型使用了全部媒体证据（[Transformers v5.21.0 文档](https://huggingface.co/docs/transformers/v5.21.0/chat_templating_multimodal)）。

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

例如 patch 为 14 时：

~~~math
N_{\mathrm{image}}=4\times(672/28)^2=4\times24^2=2304
~~~

两条边都翻倍为 1344×1344 后：

~~~math
N_{1344}=\frac{1344}{14}\times\frac{1344}{14}=96\times96=9216=4N_{672}
~~~

这里计算的是几何 patch 数，不含多尺度分支、边界 token 或模型内部合并。

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

静态把预算平均分给文本、图像、音频和视频通常不合理。设总预算为 `B_total`，模态 `m` 的任务重要度为 `I_m`，成本权重为 `w_m`，则从文本、协议、工具和生成预留中扣除后，可供媒体使用的剩余额度记为 `B_{\mathrm{media}}`。一个教学上的分配形式为：

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

把分配式代入一组明确的教学权重。假设可分配媒体预算为 12,000 token，发票图片、短视频事件和音频转写的任务权重分别为 3、2、1，示例中成本权重都取 1，则三类媒体分别获得 6,000、4,000、2,000 token。这个比例只展示如何把任务优先级变成账本初值；实际配置还要用覆盖率和质量曲线校准，不能把权重当成模型质量保证。

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

取一个教学配置：`L=32`、`N=4096`、`H_{kv}=8`、`d_h=128`、`b=2` 字节，则

~~~math
M_{\mathrm{KV}}
=2\times32\times4096\times8\times128\times2
=536{,}870{,}912\ \mathrm{bytes}
~~~

即约 512 MiB（按 1 MiB=1,048,576 字节）。上下文长度从 4096 增至 8192 时，这个未量化单请求近似值翻倍到约 1 GiB；attention score 的稠密二次代理则变为原来的 4 倍。该账本不含运行时分配器、临时激活、模型权重和其他请求。

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

对于自建的媒体表示缓存，一个可审计的 key 可以写成：

~~~math
K=H(\mathrm{digest},\mathrm{processor},\mathrm{resize},
\mathrm{sampling},\mathrm{model},\mathrm{position},\mathrm{tenant})
~~~

例如，相同原始图片 digest 在 processor 从 r7 升到 r8、或者 resize 从 672 改成 448 后，处理结果 key 必须变化；否则缓存可能返回旧网格的媒体表示。完整 KV 还依赖文本前缀、工具结果、权限和模型状态，不能只用图片 digest 作 key。实际缓存键还应按共享范围区分用户/租户，避免把一个请求的私有上下文复用给另一个请求。

公共图片的视觉 embedding 可以共享，不代表包含私有问题、工具结果和用户上下文的 KV 可以跨租户共享。provider-managed prefix cache 命中可能复用重复前缀的计算或计费，但不会缩短逻辑请求本身的上下文占用；命中也不保证发生。缓存命中率高但答案变旧，通常是失效字段缺失；命中率低但每次重新编码，通常是 key 过度包含请求级字段。

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

表中是一组用于说明约束筛选的合成实验值，不是模型测量。若要求延迟不超过 2.5 秒、单次成本不超过 0.02 美元、证据支持率至少 0.90，A 因证据门槛不合格，B 才是可行选择；若延迟上限降至 2 秒，两者都不合格，系统应拒绝自动完成。

| 合成候选 | 任务质量 Q | 延迟 L | 单次成本 C | 证据支持率 R | 约束结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 0.91 | 1.2 秒 | 0.012 美元 | 0.86 | R 未达 0.90，不可选 |
| B | 0.96 | 2.1 秒 | 0.018 美元 | 0.94 | 满足 2.5 秒/0.02 美元/0.90 门槛 |

单位成功成本可按任务门槛定义为：

~~~math
C_{\mathrm{success}}=\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
~~~

其中 `C_{\mathrm{total}}` 包含成功与失败请求的总成本。假设 100 个请求共花费 2 美元，其中 80 个同时通过正确性、证据和安全门槛，则单位成功成本为 2/80=0.025 美元。分母不应只算有答案的请求；失败、重试和人工复核成本也应纳入相应系统口径。

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

如果整张图片高分辨率输入超过预算，系统可以先用 512×384 缩略图定位“总额”“日期”“币种”，再对三个 512×512 局部区域做高分辨率回读。按同一 patch=32 的理想网格：

~~~math
N_{\mathrm{thumb}}=\frac{512}{32}\times\frac{384}{32}=16\times12=192
~~~

~~~math
N_{\mathrm{crops}}=3\times\frac{512}{32}\times\frac{512}{32}
=3\times16\times16=768
~~~

两部分合计 960；与整页 3072 相比，预算减少 `1-960/3072=68.75\%`。这不是质量改善保证：若缩略图没定位到字段，预算器必须扩大区域或转人工，且局部输出要映射回原页面坐标以供引用和复核。

如果局部检索没有找到金额，系统不能把低分辨率结果当成“发票没有金额”。正确处理是扩大区域、请求更清晰图片或转人工。

## 14.14 案例：长视频的时间覆盖

如果 20 分钟视频粗采样每秒 2 帧，采样间隔为 0.5 秒。持续 0.2 秒的事件可能完全落在两个采样点之间。再作一个理想化假设：事件开始相位在采样周期内均匀分布，采样点是瞬时且每个采样点都能正确识别事件，则：

~~~math
P(\mathrm{sampled})=\frac{0.2}{0.5}=0.4,\qquad
P(\mathrm{missed})=1-0.4=0.6
~~~

真实帧有曝光时长、识别错误和事件边界，不能把这个玩具概率当成系统召回率；它只说明采样间隔本身会形成覆盖盲区。模型回答“没有发生”之前，需要在目标事件分布上估计采样策略的召回能力。

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

下面的 Python 示例模拟一个超预算请求和一个预算回退路径。它不会调用真实模型，只用 patch 网格、音频率和视频帧数做代理估算；协议字段是一个合成总数，没有逐一展开各类媒体边界 token。正式请求必须用目标 processor 的实际输出替换这些假设。

程序先检查样本和配置的字段、类型及取值范围，再分别计算图像、音频和视频代理 token，将文本、工具历史、协议及输出预留相加。它先测试完整方案；若超过 context limit，则构造一个带降级警告的候选方案。改变 patch、音频时间率、codebook、帧数或工具历史字段，可以观察各项账本如何变化。任何候选方案都还要对照覆盖报告和质量评估，不能仅凭总数通过就直接回答。

输入中的 `text_tokens` 和 `tool_history` 是已经计数的 token 长度；`images` 逐张给出高度、宽度，`audios` 给出秒数，`videos` 给出帧数和每帧的空间尺寸。配置给出空间 patch、时间 patch、音频时间率、codebook 数、协议和输出预留、context limit 以及回退值。输出的 `counts` 按模态列账，`total` 用于 preflight 检查，`warnings` 标出降级后必须重新核验的证据风险。

本例完整方案中图像为 9,216、音频为 2,000、视频为 3,136，再加 4,200 文本、20,000 工具历史、64 协议和 2,048 输出预留，合计 40,664，超过 32,768。回退候选把图像降到 4,096、音频降到 1,000、视频降到 1,568、工具历史字段降到 8,000，合计 20,976；相对完整账本少 19,688 token，留出 11,792 token 账面空间。减少的不是抽象长度：图像缩小可能抹掉小字，降低视频帧数会漏掉短事件，缩短音频窗口会丢失较早语句，压缩工具历史也可能损坏计划或权限状态。

这里工具历史只是把计数字段改为 8,000，没有真的摘要历史，也没有证明关键信息被保留；因此 warning 特意包含 `tool_history_summary_unverified`，整个方案仍标成 `candidate_only`。实际系统应在 token 预检后生成记录原始范围、保留范围、丢弃范围和质量门槛的 coverage report，再决定能否答复。输入足够放入 context，只证明账本没有超限。

即使完整方案没有触发回退，`routing_consistent` 也只核对预算算术与所选分支；它不检查权限、媒体是否正确解码、摘要是否保真或答案是否正确。因此这个例子不会输出“可自动答复”的结论；生产门禁还要串接媒体覆盖、任务质量和安全校验，并保留失败时的转人工或补采样路径。

可以再改一个配置，观察总量是否真的跨过准入线。把 image patch 从 14 改到 28，四张 672×672 图片的代理 token 变为：

~~~math
N_{\mathrm{image}}=4\times(672/28)^2=4\times24^2=2304
~~~

总账本随之变为：

~~~math
B_{\mathrm{total}}=40{,}664-9{,}216+2{,}304=33{,}752
~~~

它仍比 32,768 上限多 984。单独提高 patch 尺寸虽减少了视觉预算，却没有解决超限；还要重新分配其他项，或按任务要求拒绝/拆分请求。真实 processor 可能不会按这个理想网格计数，所以任何通过都要以目标 processor 输出复核。

音频时长由十进制输入转换为分数后再乘以 rate 并向上取整，避免直接用浮点乘法时把 0.1 秒乘 50 得到略大于 5、错误进位到 6；整数时长也不会先转换成浮点数，因此超大整数仍可被安全计为超预算。布尔值、负数、NaN 和无穷值被拒绝。复制整段代码围栏可在 Python 3.12.9 运行，不需要第三方包；它只验证账本结构，不访问真实媒体，不运行 processor、模型、缓存或 GUI executor。

接入目标模型时，应把代理数与 processor 实际输出按模态对照，记录绝对误差和相对误差；若误差会改变准入结论，就改用实际展开数量重新路由，而不要用一个未经校准的统一安全系数掩盖偏差。输出预留也要按目标 API 的输入/输出合同复核，避免把已包含的隐藏推理预算再加一次。

代码状态：已验证（2026-11-12；Python 3.12.9；仅标准库）。

```python
import math
from fractions import Fraction


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def ceil_scaled_nonnegative(value, rate):
    scaled = Fraction(str(value)) * rate
    return (scaled.numerator + scaled.denominator - 1) // scaled.denominator


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
    total = 0
    for index, audio in enumerate(audios):
        seconds = require_finite_nonnegative(audio["seconds"], f"audios[{index}].seconds")
        total += ceil_scaled_nonnegative(seconds, rate) * codebooks
    return total


def budget_plan(sample, config, detail):
    if not isinstance(sample, dict) or not isinstance(config, dict):
        raise TypeError("sample and config must be mappings")
    validate_sample_config(sample, config)
    if not isinstance(detail, str) or detail not in {"full", "fallback"}:
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
            "tool_history_summary_unverified",
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
    "plan_status": "candidate_only",
    "routing_consistent": (
        selected["detail"] == ("fallback" if full["total"] > config["context_limit"] else "full")
        and selected["total"] <= config["context_limit"]
    ),
}
for key, value in report.items():
    print(f"{key}={value}")
```

参考输出为：

```text
full_total=40664
fallback_total=20976
selected_plan={'detail': 'fallback', 'counts': {'text': 4200, 'image': 4096, 'audio': 1000, 'video': 1568, 'tool': 8000, 'protocol': 64, 'output': 2048}, 'total': 20976, 'warnings': ['local_small_text_needs_recheck', 'short_video_event_may_be_missed', 'audio_history_window_reduced', 'tool_history_summary_unverified']}
context_limit=32768
coverage_report_required=True
plan_status=candidate_only
routing_consistent=True
```

这个结果表达了三个事实：完整方案超预算，回退方案在合成账本中可以放入预算，回退方案伴随明确的证据风险。`plan_status=candidate_only` 表示它只是预检候选；`coverage_report_required=True` 只是要求后续生成覆盖报告，代码本身并未验证裁剪区域、摘要保真度、短事件召回或最终答案质量。`routing_consistent=True` 只说明示例按配置选中了可放入账本的候选，不代表可以直接回答。

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

## 14.18 模型文档与预算证据边界

本章的几何、音频率和账本算例是明确标注的教学假设，不是论文或产品的 token 实测。外部资料用于核对 processor 接口、具体版本能力和作者报告的测量口径：

1. [Transformers v5.21.0 多模态 chat template 文档](https://huggingface.co/docs/transformers/v5.21.0/chat_templating_multimodal)：消息结构与 processor 输入示例。
2. [Qwen3-VL Technical Report v2](https://arxiv.org/abs/2511.21631v2)：作者报告中的 interleaved multimodal context、时空位置、DeepStack 和视频时间戳。
3. [Qwen3-Omni Technical Report v1](https://arxiv.org/abs/2509.17765v1)：音频表示率、时间位置与特定 serving 配置下的理论首包延迟。
4. [Qwen3.5-Omni Technical Report v2](https://arxiv.org/abs/2604.15804v2)：另一模型版本的上下文、视频长度与流式延迟设置。
5. [Alibaba Cloud Qwen3.7 Plus 托管文档](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)，访问日期 2026-11-12：动态的 region/scope capability 和 context limit 表。
6. [Alibaba Cloud Context Cache 文档](https://www.alibabacloud.com/help/en/model-studio/context-cache)，访问日期 2026-11-12：provider-managed prefix cache 的公开行为和命中边界。
7. [LVOmniBench v1](https://arxiv.org/abs/2603.19217v1)：长音视频记忆、时间定位和细粒度理解的任务化评估。

论文、模型卡和 API 文档分别支持特定版本公开的机制、配置或服务字段，不互相替代。动态文档要按使用时的 region、scope、endpoint 和 mode 复核；目标系统的有效上下文、质量曲线、缓存行为和单位成功成本必须在固定版本与硬件上独立测量。Native multimodal 是能力或产品标签，不应被写成未公开的内部架构事实。

以下三例按同一顺序阅读：先确认字段来自技术报告、模型配置还是托管 API，再判定它描述的是媒体 token、位置元数据、服务上限还是实测延迟，最后才决定能否放进请求预算。Qwen3-VL 展示位置与视觉 token 需要分账；Qwen3-Omni 展示 codec 时间率与首包关键路径不能互换；Qwen3.7 Plus 展示 API 的最大长度和 GUI 产品能力仍须经过目标 workload 验证。它们合起来是一套预算证据审计，不是三个模型的能力排名。

### 14.18.1 Qwen3-VL：视频时间戳与媒体 token 分账

Qwen3-VL Technical Report v2 声称该模型系列支持 256K token 的文本和交错多模态上下文，并描述 Interleaved-MRoPE、DeepStack 和显式文本时间戳。[技术报告](https://arxiv.org/abs/2511.21631v2)中的模型设计信息不能当成所有部署接口的长度合同，也不保证关键视频事件已被覆盖或正确使用。

为该类视频请求记账时，实际视觉 token 和显式时间戳文本应按 processor 输出计数；temporal/height/width position 是位置元数据，DeepStack 是多层视觉特征注入，两者不能未经实现证据就另算成文本 token。DeepStack 即使不增加序列长度，也可能改变层间计算和激活流量，需从目标实现单独 profile。回放记录应保存媒体 revision、帧采样率、frame index、真实 timestamp、视觉 grid、实际 token 数、chunk offset 和未覆盖区间。不要把 Qwen3-Omni 的音频 token rate 或另一模型的 context 字段迁移到 Qwen3-VL。

### 14.18.2 Qwen3-Omni：token rate 与流式延迟分账

Qwen3-Omni Technical Report v1 报告输入和输出音频 code rate 为 12.5 Hz；按该报告的表示约定，一个时间步约对应 80 ms 音频。报告还把视频 timestamp 映射到相同的时间尺度。这些数值描述特定模型的位置/codec 粒度，不能推导成通用音频 token 预算或客户端可播放延迟。[原始报告](https://arxiv.org/abs/2509.17765v1)的单并发配置报告理论首包延迟为音频 234 ms、视频 547 ms；并发升至 4 时报告为 728/1,517 ms，升至 6 时为 1,172/2,284 ms，且测试使用报告指定的 vLLM、torch.compile 与 CUDA Graph 设置。

输出路径还要分别测量 Talker codec steps、MTP 补码、Code2Wav 解码、packet 和播放器缓冲。首包延迟是串行关键路径的测量值，不是 token 数；它必须绑定 variant、输入采样、并发、运行时和硬件，不能与 80 ms 时间步相加或互换。另一个版本 Qwen3.5-Omni 报告最高 256K context，并报告 720P 视频 1 FPS、超过 400 秒的处理设置；该版本和测试协议的字段不能覆盖 Qwen3-Omni 的配置。[Qwen3.5-Omni 报告 v2](https://arxiv.org/abs/2604.15804v2)

### 14.18.3 Qwen3.7 Plus：托管 API 合同不等于有效容量

Qwen3.7 Plus 官方托管文档把输入模态列为文本、图像和视频，并将读屏、GUI 交互、视觉参考生成代码和移动端导航列为产品能力；该页面将当前 alias 对应到 `qwen3.7-plus-2026-05-26` snapshot。[官方文档](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)的 Context Limits 分别列出 1,000,000 context window、991,808 max input、131,072 max output；thinking mode 另列 983,616 max input、131,072 max output 和 262,144 max chain-of-thought length。这些是不同的托管合同字段，不应机械相加，也不说明任意长度下关键截图都被保留。

同一文档按 region/scope 列出 capability；页面中的服务范围行对 Structured Outputs、Web Search 和 Batch 等字段给出不同支持状态。复现实验至少要固定 snapshot、region、scope、endpoint 和请求 mode，不能只记录模型 alias。

GUI 截图还携带窗口尺寸、设备像素比、页面 revision、坐标系和动作后的观察证据。以下字段是本书建议的回放清单，不是厂商公开的内部输入 schema：

一次可回放的 GUI 请求至少应有：

```text
media hash/size/resize/crop
frame index/timestamp/visual token estimate
text/tool schema/tool result/thinking/output reservation
screen revision/window/viewport/device pixel ratio
action proposal/permission/executor receipt/observation/verifier
```

1M context 只能说明服务合同的窗口上限，不能说明所有截图细节都会被保留，更不能说明坐标动作一定成功。预算器应先把媒体处理结果和工具历史放进账本，再为 action schema、权限回执、观察和 verifier 预留空间。若预算不足，回退策略必须明确丢弃了哪些帧、分辨率或历史，并生成 coverage report；不能静默截掉包含按钮状态或错误提示的关键截图。

还要把媒体预算和执行预算分开。Structured Outputs 可约束返回格式满足 schema，Function Calling 可承载工具调用协议；它们不能证明 action 语义正确、坐标仍有效或操作已经获准。在自建系统中，真实点击、输入和导航应由有权限边界的宿主 executor 执行。页面 revision 变化后，旧坐标应失效并触发重新观测。于是有效能力应按四层验收：媒体被接受、关键证据被保留、模型提出了合法动作、宿主执行并验证了副作用。

Qwen3.7 Plus 官方文档还把 Context Caching、Prefix Completion、Function Calling 和 Web Search 按 region/scope 列出。Alibaba 的 Context Cache 文档描述 provider-managed prefix matching，并说明满足最小前缀长度也不保证命中；这与 GPU KV cache 和应用 memory 属于不同层次。[Context Cache 文档](https://www.alibabacloud.com/help/en/model-studio/context-cache)

对于自己维护的回放或应用缓存，key 应包含所需的模型 alias/snapshot、region/scope、媒体 revision 和 tool schema hash；provider 托管缓存的内部 key 未公开，不能把这份应用设计写成服务端实现。输入长度分段、cache 状态和价格应以实际 provider 合同为准。
