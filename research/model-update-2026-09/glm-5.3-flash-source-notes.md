# GLM-5.3-Flash：榜单锚点、混合架构与视觉闭环资料摘记

核验日期：2026-09-22。本文件是研究底稿，不把榜单配置、发布方自报 benchmark 或模型卡配置字段升级成未公开的训练事实。

## 1. 锚点和证据范围

GLM-5.3-Flash 是本轮从两个允许的排行榜中确认的重点锚点。发现入口和官方核验入口的职责不同：Artificial Analysis 与 DataCurve DeepSWE 负责说明“应当研究哪个条目”；Z.ai 官方文档、官方博客、Hugging Face 模型卡/配置和相关实现负责说明“该条目公开了什么”。

| 层级 | 当前证据 | 结论边界 |
|---|---|---|
| Artificial Analysis | [GLM-5.3-Flash](https://artificialanalysis.ai/models/glm-5-3-flash)，canonical slug `glm-5-3-flash`，本次页面快照 SHA-256 `7800ff202ced5e5cc170d7f1858d8070cf8d41c47b3ab6bace60b75c596b6319` | `releaseDate`、Intelligence Index、速度、TTFT、价格和 effort 是第三方页面/配置字段，不是官方参数或训练证明 |
| DataCurve DeepSWE | [DeepSWE v1.1](https://deepswe.datacurve.ai/)，本次快照 SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` | 结果绑定 `mini-swe-agent`、工具、任务集、环境、超时、重试和 verifier，不是裸模型能力 |
| Z.ai 官方文档 | [GLM-5.3-Flash 文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)，快照 SHA-256 `a127bf7eff2780aacebfc4ffdcadfac5820b75caeaafdb932da0c8942eee879f` | 支持模型定位、公开架构摘要、模态、接口和视觉 coding 工作流 |
| Z.ai 官方博客 | [GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash)，正文由页面脚本加载，本次 JS 资源 SHA-256 `225196c63b5944629606d26982c0a43c4a8fbd6edb8a7a2e9bf8abfacd35fcbb` | 支持架构动机、IndexPool、mHC、Serving 和视觉闭环的发布方描述；speedup 仍是自报 |
| 官方模型卡/实现 | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)，固定 revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | README、`config.json` 和许可证是公开 artifact；配置字段不自动等于完整训练报告 |

本次复验使用的临时文件位于 `/tmp/glm53flash-*current-8098-20260915.*` 和 `/tmp/glm53flash-*current-7890-20260915.*`，只用于记录采集身份，不作为仓库依赖。`10.237.126.170:1234` 与 `10.24.27.134:8098` 可完成榜单/官方页面访问，7890 作为备用线路时对大页面偶发 TLS EOF 或读取超时；网络失败不被解释为页面不存在。

## 2. 榜单配置：先固定条件，再谈数字

### 2.1 Artificial Analysis

本次页面可解析到：

- canonical slug：`glm-5-3-flash`；页面 release date 字段为 `2026-08-26`。
- 主配置为 `max reasoning effort`。
- Artificial Analysis Intelligence Index：`41.907366113455`。
- median output speed：`114.22108687545 tokens/s`。
- median TTFT：`2.45458272199994s`。
- context 字段：`1M`。
- 输入/输出/缓存命中价格字段：约 `$0.15/$0.50/$0.026` 每百万 token。

这些是当前页面快照中与配置相关的字段。尤其不能把 `max` 行当成一个新的 checkpoint，也不能把页面价格当成所有地区、账户、时间和 API 路由的长期价格承诺。

2026-09-21 当前时点复验：页面大小 `3,942,546` bytes，SHA-256 为 `42f880600d3637489a0c48ff27357fe7986e53510ad5ee016c7d6170bd201fae`；release date 仍为 `2026-08-26`，context `1,048,576`，Intelligence Index `41.807466113455`，median output speed `95.0131572798129 tokens/s`，cost per Intelligence Index task `0.2532595604307378`，页面价格约 `$0.15/$0.50` 每百万输入/输出 token。与旧快照相比指数和速度有测量漂移；没有证据表明这是模型版本或训练变化。

### 2.2 DataCurve DeepSWE v1.1

对应配置记录为：

```text
model=glm-5-3-flash
harness=mini-swe-agent
reasoning_effort=max
config=mini_swe_agent_glm_5_3_flash_max
n_passed=284
n_attempted=448
Pass@1=63.392857%
Pass@4=84.955752%
mean_cost=$0.2409818562
mean_output_tokens=72829.77
mean_agent_steps=122.89
n_runs=4
```

`n_passed/n_attempted`、Pass@1/Pass@4、成本、输出 token 和 steps 描述的是一个“模型配置 + Agent harness + 工具执行 + 环境 + verifier”的系统。它可以作为面试中的可复现实验索引，但不能写成“GLM-5.3-Flash 裸模型在代码任务上有 63.4% 能力”。

2026-09-21 当前时点复验仍命中精确行 `mini_swe_agent_glm_5_3_flash_max`，结果为 `284/448`、Pass@1 `0.6339285714285714`、Pass@4 `0.8495575221238938`、平均成本 `$0.24098185622767856`、平均输出 `72829.77008928571` token、平均 Agent steps 约 `122.89`。DataCurve 页面快照为 `268,571` bytes，SHA-256 为 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；这些数字仍绑定 `mini-swe-agent`、工具、任务集、环境和 verifier。

## 3. 官方身份与结构化配置

### 3.1 版本边界

GLM-5.3 主模型页面称其沿用 GLM-5.2 基础模型、改进来自后训练；GLM-5.3-Flash 的官方模型卡则称它从 newly trained base model 开始，重新设计架构和训练 recipe。这两条描述适用于不同条目，不能合并成一句“Flash 只是 GLM-5.3 的后训练版”。本专题按 Flash 模型卡和 Flash 专属文档记录。

官方 README/配置公开了 `Glm5NextForConditionalGeneration` 和 `glm5_next`；模型卡的 front matter 标注 MIT。许可证字段证明的是该固定 artifact 的许可声明，不替代仓库完整 LICENSE 的法律解释，也不能推导云 API 的商业条款。

### 3.2 文本主干字段

固定 revision 的 `config.json` 给出以下可直接核对的字段：

| 字段 | 值 | 解读 |
|---|---:|---|
| total parameters | 320B | 模型卡/官方文档的总参数口径；表示权重容量，不等于一次 token 的 FLOPs |
| activated parameters | 18B | 发布方的 active 口径；不等于单卡显存或端到端延迟 |
| hidden size | 4096 | 文本主干 hidden width |
| hidden layers | 45 | 配置中的 Transformer-like layer 数 |
| first dense MLP layers | 3 | `first_k_dense_replace=3`；前 3 层 MLP 为 dense |
| routed experts | 288 | sparse MoE 的 routed expert 数 |
| experts per token | 8 | 每 token top-8 routed experts；另有 1 shared expert |
| MoE intermediate size | 2048 | routed MoE expert 的配置字段 |
| max position | 1,048,576 | 1M 上下文接口/配置上限；不是有效长上下文质量证明 |
| residual/mHC | enabled | `mhc=true`、`hc_mult=4`、Sinkhorn iterations `20` |

attention 的 layer type 列表包含 34 个 `linear_attention` 和 11 个 `deepseek_sparse_attention`；11 个稀疏层出现在配置索引 `3, 7, 11, ..., 43`。这里使用“配置命名”和层列表的事实，不把它扩写成一份未公开的完整算子论文。线性注意力的官方高层解释是以递归状态承载局部/大部分历史，稀疏路径借助轻量 indexer 恢复远程精确检索。

### 3.3 视觉配置

`vision_config` 给出 24 层视觉模块、hidden size 1024、16 heads、image size 448、patch size 14、temporal patch size 2、spatial merge size 2，并把视觉输出投影到 4096。模型卡/官方文档把输入模态写为 video/image/text/file，输出为 text。视觉配置说明“有一个原生视觉路径”，不等于它能在任意视频长度、分辨率、GUI 或文件格式下稳定完成任务。

## 4. 混合线性/稀疏注意力

### 4.1 为什么只用一种注意力不够

长任务 coding Agent 会先读取大量代码，再在很晚的时候回到早期的接口定义。全量 causal attention 可以逐 token 精确访问历史，但每个 query 都面对变长的 key/value 集合；把全部历史写进常规 KV cache 后，decode 阶段还要反复读取越来越大的缓存。

纯线性注意力把历史压进固定形状的递归状态，单步成本和状态大小更容易控制，却可能丢掉某个早期变量名、错误堆栈或工具回执的精确 token。GLM-5.3-Flash 的公开路线是让两种路径分工：

```text
大部分层：linear attention -> 递归状态 / 便宜历史汇总
少数层：sparse attention -> indexer -> 远程精确候选 -> 显式检索
```

这不是“把 Transformer 变成 RNN”，也不是“稀疏层可以免费读取全部历史”。indexer、候选排序、gather、mask、cache 和 kernel 仍然要付费。

### 4.2 线性路径的教学抽象

对一个 attention head，可以用固定状态 `S_t` 表示递归历史：

```math
S_t = lambda_t S_{t-1} + eta_t \; k_t v_t^{\top},
\qquad y_t = S_t^{\top}q_t .
```

`k_t`、`v_t`、`q_t` 是当前 token 的 key/value/query，`\lambda_t` 表示旧状态保留程度，`\eta_t` 表示新写入强度。这个式子只是帮助理解“固定状态 + 更新 + 查询”的信息流，不是 Flash 的完整生产 kernel 或训练目标。真实实现还需要门控、卷积、归一化、位置处理、状态布局和融合 kernel；配置给出的线性 attention 部分包含 64 heads、head dim 128 和 short-convolution kernel size 4。

### 4.3 稀疏路径与 IndexPool

稀疏层要先回答“哪些历史值得展开”。官方博客称它使用 lightweight indexer，并引入 **IndexPool**：把 4 个 indexer key vectors 通过加权 pooling 压成 1 个，以降低 1M context 下 indexer 的延迟和内存压力。固定配置进一步给出：

- `index_n_heads=32`；
- `index_head_dim=128`；
- `index_topk=2048`；
- `index_kpool=4`；
- `index_kpool_compress=true`；
- `index_kpool_always_select_tail=true`；
- `index_share_for_mtp_iteration=true`。

IndexPool 的抽象可以写成：

```math
\bar{k}_{b,h}=\sum_{j=1}^{4}\alpha_{b,h,j} k_{b,h,j},
\qquad \alpha_{b,h,j}\ge 0,\quad \sum_j\alpha_{b,h,j}=1 .
```

`b` 是一个被压缩的局部单元，`h` 是 indexer head。这个式子只表达“加权池化”，没有假设生产实现一定使用 softmax、固定权重或某种具体归一化。之后 query 与 `\bar{k}` 产生候选分数，再执行 top-k、因果约束和稀疏 gather。`index_topk=2048` 是配置字段；在没有完整 kernel 和端到端 tracing 时，不应擅自把它解释为最终 attention 一定读取 2048 个 token。

### 4.4 与普通 KV cache 的区别

线性层维护递归 state，稀疏层仍有显式的 key/value 和 indexer 中间结果。因此不能用一个“所有层都按 `2 × L × T × H_kv × d`”的公式描述整个模型。普通 attention 的教学账本是：

```math
\mathrm{KVBytes}_{\mathrm{explicit}}
=2_{K,V}\times L\times T\times H_{KV}\times d_{head}\times b_{elem} .
```

对 GLM-5.3-Flash，至少要分成：

1. 11 个显式稀疏 attention 层的 KV/cache 与 indexer 临时张量。
2. 34 个线性 attention 层的递归状态、短卷积状态和实现所需 metadata。
3. mHC widened residual 的激活/通信存储。
4. MoE dispatch 的 token、expert、combine buffer。
5. 视觉编码、文件/视频预处理和 EPD 之间的中间结果。

因此官方博客给出的“相对 GLM-5.3 attention computation 约 3.0×、KV cache 约 4.4×降低”是特定比较口径下的发布方结果，不是把每一层都代入单一 KV 公式后的数学定理。

## 5. Sparse MoE：总容量和每 token 计算解耦

设 token 表示为 `x`，router 产生 288 个 expert score，选择 top-8 routed experts，再加 shared expert。教学抽象为：

```math
p=\mathrm{router}(x),\qquad I=\mathrm{TopK}(p,8),
\qquad y=E_{shared}(x)+\sum_{i\in I}\tilde p_i E_i(x).
```

`320B total / 18B activated` 的表达说明模型把权重容量和每 token 的主要计算预算拆开；但 active 口径通常不包含所有 attention、router、shared expert、通信、padding、workspace 和 cache，所以不能直接把 18B 当成 18B dense 模型的显存或 tokens/s。

面试中应追问四个系统问题：

- top-8 token-expert assignment 是否导致热门 expert 过载？
- 跨设备 dispatch/combine 的 all-to-all 是否抵消了少算的 FLOPs？
- shared expert 与 routed expert 的输出如何保持数值稳定？
- indexer、MoE router、MTP 和 cache 的 token 顺序是否一致？

官方配置证明了专家数量和 top-k 字段；没有公开完整训练 loss、capacity factor、丢 token 规则和线上负载分布时，不能补写这些细节。

## 6. mHC：在更宽的残差流上加几何约束

### 6.1 从标准 residual 到多流 residual

标准 residual 是：

```math
x_{l+1}=x_l+F_l(x_l).
```

更宽的 Hyper-Connections 允许多路 residual stream 先混合、送入主分支，再写回多路状态。表达力增加的同时，连续矩阵混合可能放大或衰减某些流。mHC（Manifold-Constrained Hyper-Connections）的公开思想是把关键流混合映射约束到双随机矩阵集合：

```math
\mathcal{B}_n=\{M\mid M\mathbf 1=\mathbf 1,\;\mathbf 1^\top M=\mathbf 1^\top,\;M\ge0\}.
```

每行和每列都为 1，意味着流量既不会在结构上任意凭空放大，也不会任意被吸走。它不是“每个神经元范数恒定”的保证；非线性、写回矩阵、归一化、量化和训练更新仍会影响实际激活。

### 6.2 Flash 配置能证明什么

固定配置开启 `mhc=true`，并给出 `hc_mult=4`、`hc_sinkhorn_iters=20`。这足以确认该公开 artifact 启用了 mHC 相关配置，以及使用了 4 倍 residual 宽度/多流相关的配置口径和 20 次 Sinkhorn 迭代字段。它不公开每层 `A/B/C` 的全部参数、融合 kernel、通信布局或训练消融，不能从字段推断这些未公开实现细节。

Sinkhorn 教学投影为：

```text
M^(0) = exp(raw_scores)
M^(t) = column_normalize(row_normalize(M^(t-1)))
```

正矩阵经过交替行/列归一化后趋近双随机集合；有限 20 次只说明一个实现配置，实际误差还依赖数值精度、logit 范围和反向传播。mHC 与 IndexPool 优化的是不同对象：前者约束深度 residual 流，后者压缩稀疏 indexer 的历史 key。

## 7. 原生多模态与视觉 self-verification

### 7.1 视觉不是一次性的输入编码

普通“图像问答”可以只把图片编码成一次 embedding，再生成文本。GLM-5.3-Flash 的官方定位更接近一个视觉参与的 Agent loop：

```text
参考截图/需求
      -> 生成代码或操作
      -> 启动应用、渲染页面、运行游戏/Blender/Office 工具
      -> 观察界面、渲染结果和交互反馈
      -> 判断差异
      -> 修改并再次验证
```

官方文档具体把 frontend、game、Blender、BUA/CUA、PPTX/PDF/DOCX/XLSX、视频编辑和金融工作流作为示例；官方博客称其训练/数据合成关注 self-visual judgment、test-time improvement，并在 GUI workflow 中使用 agent-based verification。面试时要把它说成“公开的模型与系统工作流主张”，而不是“模型凭视觉天然知道产品是否正确”。

### 7.2 渲染正确不等于功能正确

一个网页截图可能颜色和布局正确，却无法点击；一个 PPTX 可能能打开，却有文字溢出；一个游戏画面可能漂亮，却无法完成胜负循环。因此应把验证拆成：

| 层 | 检查对象 | 典型 verifier |
|---|---|---|
| 语义/代码 | API、类型、单元测试、构建 | compiler/test runner |
| 行为/交互 | 点击、输入、状态转移、错误恢复 | browser/game/CUA trace |
| 视觉 | 布局、裁剪、溢出、遮挡、风格 | screenshot/视觉比较 |
| 交付 artifact | 文件可打开、引用、权限、可复现 | file validator + audit |

模型可以提出下一步观察和修改，但浏览器、终端、文件系统和外部网络权限仍由宿主执行器授权。视觉 self-judgment 也不能代替独立的行为 verifier；否则模型可能只学会“看起来完成了”的捷径。

## 8. Serving：架构、硬件和调度一起设计

### 8.1 官方披露的优化栈

Z.ai 官方文档/博客公开列出：

- 在 SGLang 之上为该架构构建 dedicated inference engine；
- `ReplaySSM`，用于线性/递归状态相关的 serving 路径；
- W8A8，以及 INT8/FP8/BF16 的 hybrid cache quantization；
- Layer Split；
- 对 Linear Attention 和 LM head 使用 intra-node tensor parallelism；
- 在集群层使用 EPD（Encode–Prefill–Decode）把多模态 encode、prompt prefill 和 token-by-token decode 分成可独立调度的 worker pool。

这些名字描述的是公开 serving 组件/策略，不表示任意 SGLang 版本开箱即用，也不等于已经获得同样性能。

### 8.2 EPD 为什么比简单 PD 更适合多模态

传统 PD（Prefill/Decode disaggregation）主要拆分 prompt prefill 和逐 token decode。多模态请求还包括图像、视频、文件的 encode；如果 encode 长时间占用与 decode 相同的 worker pool，短文本请求的首 token 延迟会被拖慢。EPD 把三个阶段单独建账：

```text
Encode worker pool  --encoded representation-->  Prefill worker pool
Prefill worker pool  --KV/state metadata------->  Decode worker pool
Decode worker pool   --streaming text---------->  client
```

跨池传递的不是“免费数据”：需要记录版本兼容、序列 ID、位置偏移、cache dtype、取消/超时、传输带宽、失败重算和租户隔离。prefill/decode 的分离只有在负载具有足够阶段差异、传输成本可接受且运维边界成熟时才有收益。

### 8.3 3× 结果的证据边界

官方博客称在相同硬件、从初始 baseline 到优化栈后端到端 serving performance 约提升 3×，并称每 token 成本/硬件效率接近主流 NVIDIA GPU。该数字是 Z.ai 对特定国产加速器、网络、batch、请求分布和实现的发布方自报；需要固定 backend commit、硬件、并行度、输入长度、输出长度、视觉比例、并发、TTFT/TPOT 定义和 p95/p99 后才可复现。不能从“3×”推导一般部署收益。

### 8.4 2026-09-21 serving/runtime 复核

本次复核把“模型公开配置”“runtime 实现入口”和“目标硬件验收”分开记录。SGLang 页面是当前最具体的 serving 证据，但页面中的 benchmark 和硬件支持仍属于 recipe/发布方结果，不是本机复现。

#### SGLang：两类状态池和 MTP 开关

当前 [SGLang GLM-5.3-Flash cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash.md) 页面更新时间为 `2026-09-21T12:21:10.188Z`，快照大小 `247,030` bytes，SHA-256 为 `fef983feab9a25311de8cf0b62c539b8450ef9acef9015a288ab6639ecd89019`。页面描述的模型/服务组合包括：45 个文本层（MLA、DSA、KDA 组合）、24 层视觉 encoder、288 routed experts/top-8、原生 MTP。

对调度最重要的不是“有 MTP”这四个字，而是 runtime 同时管理两类历史状态：

1. paged KV pool：显式 attention 路径使用的分页 K/V；
2. KDA state pool：KDA/线性路径的递归状态。

页面明确提示 KDA state pool 可能先成为并发上限。因此混合模型不能只以 `kv_tokens` 作为恢复和容量字段；请求 manifest 至少还应保存 state pool 的句柄、长度、dtype/backend、MTP 位置和模型 revision。

低延迟配置使用 MTP `5/1/6`，高吞吐配置可以关闭 speculative decoding。这说明 MTP 是带 workload 取舍的 serving 策略，不是每个请求都必须启用的静态模型属性。关闭 speculative 后仍可使用主体模型路径，但吞吐、接受率和调度账本必须重新测量。

#### KV dtype 与 DSA backend 必须成对切换

该页面给出的硬件路线具有明确的配对约束：Blackwell 默认 FP8 KV + TRT-LLM DSA；H100/H200 默认 BF16 KV + TileLang DSA。页面同时标明 FP8 KV + TileLang DSA 是无效组合。面试中应把这回答成“cache dtype 和 sparse-attention backend 共享 kernel/数值契约”，而不是泛泛地说“FP8 更省显存”。

页面发布方在 GB300 上报告 FP8 KV + TRT-LLM DSA 相比 BF16 路线约有 `2.9%--5.7%` throughput 差异、KV token capacity 约 `1.8x`，GSM8K 差异在噪声范围内。这些数字未在本机复现，也不是跨 GPU、batch、context 或 provider 的普遍保证。

#### 多模态 EPD/PD 的门禁

SGLang 页面把视频默认采样写为 2 FPS，最多约 240,000 visual tokens，并要求 `torchcodec`；4x GB300 encoder disaggregation 的页面结果把最大 decode gap 从 `5.53s` 降到 `1.79s`。这些结果说明 EPD 的收益要看 encode 阶段是否阻塞 decode，不能只拿文本 PD 的 TTFT/TPOT 推导。

同一页面对 PD disaggregation 的证据边界更谨慎：当前只是 dummy weights 的机械验证，没有 load/accuracy 验证；普通 speculative decoding 还不支持，不同 TP 的数值正确性也尚未验证。因此“页面有 PD 命令”只能说明接线入口存在，不能写成生产 correctness 或多卡一致性已经通过。

#### vLLM 和 Transformers 的边界

当前 vLLM recipe 页面于 2026-09-18 更新，要求 vLLM `0.29.0+`，描述 native FP8、MTP、1M context 和 hybrid KDA+sparse MLA，并列出 H100、B200、GB200、MI355X、Ascend 950PR 8x 等路线。当前快照暂存为 `/tmp/glm53flash-vllm-recipe-20260921.out`，SHA-256 为 `cfd032baa545b0749c171bebac7ad9fab30e3478ef20b5233f1ab44971539103`。这些是 recipe 的验证/支持声明，不是本机加载或 profiling 结果。页面自身还出现 FlashInfer `0.6.17+` 与 troubleshooting `0.6.18+` 两个门槛，迁移时应锁定实际报错路径和完整依赖，而不是只抄一个版本号。

Ascend 950PR 路线的页面说明 native FP8 约需 306 GiB、使用 `--block-size 512`，且不应传 `--quantization ascend`；A2/A3/BF16 尚未验证。MI355X 路线支持 MTP，并关联 vLLM PR `#55239`。这些信息可以进入部署 checklist，但不能升级成目标设备已验收。

Transformers 的 [GLM5-Next 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm5_next.md) 当前实现明确不包含 MTP layer；快照大小 `2,783` bytes，SHA-256 为 `e08af95e1b8cab25e11e0623ee0e14ab8280396bdd014f9d1a035aeb48f2fdf2`。因此要区分：Transformers 是模型接入/基础前向实现，vLLM/SGLang recipe 可以在 serving 层提供 MTP；后者不能反向证明 Transformers checkpoint 本身含有 MTP layer。

#### FlashX 不是新模型候选

Z.ai 文档把 `GLM-5.3-FlashX` 列为关联服务入口，并标注约 200 tokens/s；Flash 页面同时给出 1M context、128K maximum output、`thinking.type=enabled`、推荐 `temperature=1`/`top_p=0.95`，以及 Flash Coding Plan 约为 GLM-5.3 配额的 3 倍（FlashX 尚未加入该 plan）。这些是 endpoint/service 差异。`GLM-5.3-FlashX` 不在本轮两个排行榜的新增候选中，因此不另建模型条目，也不把它的服务速度当成新 checkpoint 的架构事实。

API 迁移时，Coding/Agent 场景建议 `clear_thinking=false` 并完整、原样回传 reasoning content；流式工具参数仍按 tool-call `index` 拼接。推理内容回传是协议状态要求，不是权限凭证，tool call 也不等于宿主已经执行成功。

## 8.5 2026-09-22 upstream implementation 与 stable/main 对照

本轮用 GitHub Contents/tree API 固定了 SGLang 与 vLLM 的源码路径和提交历史。源码存在、测试类存在和测试门禁写在 upstream 中，都是 **implementation evidence**；本机没有安装对应框架、Flash 权重或目标 GPU，因此不把它们写成已运行的硬件结果。

### SGLang：v0.5.20 已有 Flash 入口，main 继续收敛 kernel/融合

SGLang `v0.5.20` tag（tag commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，发布时间 `2026-09-18T22:41:33Z`）已经包含：

- `python/sglang/srt/models/glm5_next.py`，源码 `61,466` bytes，SHA-256 `12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2`；
- `python/sglang/srt/configs/glm5_next.py`，`12,026` bytes，SHA-256 `3b3c7aa3ae60e1cf59edf91e1c11a6aa49be7532340c8e2482f7f75ed859f3e0`；
- `glm5_next_nextn.py`、GLM-5.3-Flash cookbook，以及 B200/H200 注册测试。

因此“Flash 只有文档 recipe、没有 SGLang 模型入口”的说法不成立；但 tag 中存在源码不等于本机权重加载或目标卡验收通过。当前 `main` tree 为 `9d58189c12e4e14a7eea20f24f9aa7b17e221778`，main 模型文件 `68,097` bytes，SHA-256 `1cd324533aa0827e7542e27fc39c5901c2330823c4b3a32b79bcb26521dbcfec`。main 相对 v0.5.20 的可观察增量包括：

1. `Glm5NextLinearAttention` 增加按量化配置决定的 q/k/v/b/f/g projection fusion；LoRA 或不兼容 quant method 时回退到非融合路径。
2. `c8eb54c41da1`（2026-09-20）加入 GLM-5.3-Flash KDA projection 与 prefill metadata 融合；`2fa6b94e3440`（2026-09-20）加入 mHC attention-to-MLP boundary fusion；`b44e2486824e`（2026-09-22）加入 gfx950 的 FP8/Quark MXFP4 MoE 路径。
3. main 的 B200/H200 测试使用真实模型路径 `zai-org/GLM-5.3-Flash`。B200 是 TP4/EP4，FP8 KV + TRT-LLM DSA，要求 GSM8K 500 题/20-shot 达到 `0.93`，低延迟还检查 EAGLE MTP `5/1/6` 的接受长度 `4.0` 与单请求速度阈值；另有 DFlash2 和 high-throughput DP/Deepep 变体。H200 对应 TP8/EP8、BF16 KV + TileLang DSA，也有 `0.93` GSM8K 门禁。测试文件存在不代表本轮已经运行这些测试。

SGLang cookbook 的提交历史显示 serving 组合仍在演进：初始 recipe 为 `dfc40e0efe10`（2026-08-26），`e27a7fac772b` 固定 Blackwell 的 FP8 KV + TRT-LLM DSA，`b5a2aebc7ecc` 固定 MTP `5/1/6`，`a0781f271462`（2026-09-22）将 reasoning/tool-call parser 默认改为 `auto`。这些提交说明 recipe 是版本化工程 artifact，不应把 mutable main 的当前开关迁移到所有旧镜像。

### vLLM：main 已拆出 GLM5Next runtime，v0.29.0 tag 没有同名专属路径

vLLM `main` tree（`81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`）包含 `vllm/models/glm5next/`，包括：

- `common/attention.py`：`Glm5NextIndexerCache` 把 `tokens_per_state` 设为 `index_kpool`，让 KV metadata 以 pool 粒度寻址；它与同 block 的 MLA 共用 block table，并对 `cache_config.block_size` 与 `index_kpool * 32` 的整除关系做前置检查。
- `common/attention.py`：`Glm5NextTailCache` 单独保存未完成 pool 的 raw BF16 K 与 gate score。prefill 先 seed tail，decode/spec-decode 原地推进，PD connector 传递 tail；已压缩的 indexer entry 则在另一份 `Glm5NextIndexerCache` 中保存。这证明“indexer cache”和“递归 KDA state”之外，pool 边界还有第三类需要恢复的临时状态。
- `common/kda.py`：`Glm5NextLinearAttention` 通过独立的 Mamba/GDN state dtype/shape 接口分配 KDA state；conv state 宽度包含 `num_spec`，以覆盖 draft/verify 的滑动更新；FlashKDA 只在 CUDA SM90/SM10x/SM12x、BF16、head dim 128 且 gate 有 lower bound 时启用，否则回退 Triton。KDA projection 保持 BF16，因为 FP8 checkpoint 没有对应 scales。
- `common/mtp.py`：MTP 有显式的 layer/weight mapping；`set_skip_topk()` 让 MTP 后续迭代复用 sparse index top-k，`compact_topk_indices()` 按 slot id 重排 buffer。MTP logits 路径可以只做 vocab-parallel local argmax，不必每个 draft step all-gather 全词表。

相关 main 路径的快照大小/SHA-256 为：`attention.py` `24,145`/`a7554347a8e91215a9cf884f74bdcbaa4eb4a3870dbd016994b92d93293b1c57`，`kda.py` `31,136`/`37745b45892cb26d9193160c4f446191f6cc276031fe8f7de5dbb37784cf8e8b`，`model.py` `51,591`/`9d30ec0bf2eb052b96c6fc995435979993a68cced3376734d0ea72002e0f5dfc`，`mtp.py` `17,490`/`db158eec6731fb11e3072cd9d9a76f34f3267007a821277ce3463090eb8f85c0`，`common/sparse_indexer.py` `6,294`/`a3ab1edda8490b8e21c1c240e07e8c8fcd0bb34246a9ed1f64acfe067d15067c`。关键历史为：`4fe9e6f6e564`（2026-09-16）将 sparse indexer kpool 移入模型目录并拆分 AMD/NVIDIA，`c8d1cf077a78`（2026-09-16）修复 ROCm Quark MXFP4 load/inference，`36fa72d2d0d2`（2026-09-19）为 decode workspace 节省约 3072 MiB 的提交说明。

对照 vLLM `v0.29.0` tag tree：快照 `1,979,822` bytes、SHA-256 `7131879ae9592d90738776d24ae213f789577317b2c5427f350761a87ca05070`，路径搜索没有 `vllm/models/glm5next/` 或 GLM-5.3-Flash 专属 common runtime 文件；该 tag 只有 GLM-4/通用 GLM 与 DeepSeek V3.2 相关条目。因而现有 vLLM recipe 的“`0.29.0+`”是 recipe 的版本门槛/部署声明，不能单独证明公开 `v0.29.0` tag 已包含这些专属源码。正确状态是：vLLM GLM5Next 为 main/post-tag upstream evidence；recipe、stable tag、wheel 和目标硬件 acceptance 必须分开记录。

### 面试上应保留的实现边界

可以确认：GLM-5.3-Flash 的 runtime 不是一个只按 `kv_tokens` 计数的缓存；至少要管理显式稀疏/MLA KV、KDA recurrent/conv state、indexer compressed pool、in-progress tail pool、MTP top-k/slot mapping、MoE dispatch 和视觉/EPD metadata。不能确认：这些源码在本机完整权重、各硬件 backend、PD/EPD recovery、数值正确性和线上 tool/verifier 上已经通过。源码提交、recipe benchmark 和 CI threshold 都不能替代这些 gate。

## 9. 三本账：参数、状态、带宽

研究和面试回答至少要同时画出以下三本账：

| 账本 | 需要记录 | Flash 公开线索 |
|---|---|---|
| 权重/计算 | total、active、专家数、top-k、矩阵 dtype、分片 | 320B total、18B activated、288 routed、top-8、1 shared |
| 历史状态 | 线性 state、显式 sparse KV、indexer key、MTP 复用、residual state | 34 linear layers、11 sparse layers、IndexPool=4、1M position |
| 系统带宽 | expert dispatch、state read/write、KV read、EPD transfer、量化/反量化 | W8A8、hybrid cache quantization、ReplaySSM、Layer Split、EPD |

一个请求的粗略成本可写成：

```math
C_{request}=C_{encode}+C_{prefill}+C_{decode}
              +C_{dispatch}+C_{state/KV}+C_{transfer}+C_{verify}.
```

对任意一项未测量的 `C`，应记录为 `unknown`，而不是用 0 或一个“理论上很小”的常数填充。特别是线性 attention 的固定 state 降低了随 `T` 增长的历史账本，但不会自动消除状态读写、batch 对齐和跨卡通信。

## 10. API 与 Agent 协议

官方文档把 Flash 的 model code 写为 `glm-5.3-flash`，上下文 1M、最大输出 128K，输入支持 video/image/text/file，输出为 text。公开请求约束包括：

- `thinking.type` 只支持 `enabled`，不能用 `disabled` 关闭 thinking；
- `reasoning_effort` 支持 `low`、`high`、`max`，默认 `max`；
- 官方建议 `temperature=1`、`top_p=0.95`，但迁移时应按服务文档固定采样参数；
- 流式返回中应区分 `delta.reasoning_content`、`delta.content` 和 `delta.tool_calls`；
- 使用 `tool_stream=true` 时，工具参数必须按 tool-call `index` 拼接，不能按到达顺序粗暴覆盖；
- interleaved thinking 允许模型在多个 tool call 之间继续思考；保留 thinking 时要把完整、未修改、原序的 reasoning content 回传；
- `clear_thinking=false` 与 Coding Plan/标准 API 的默认值和用途有 endpoint 差异，不能把一个端点的默认值抄给另一个端点；
- 结构化输出文档公开的是 `response_format={"type":"json_object"}` JSON mode，工程上仍要做 JSON parse 和 schema validation，不能写成严格 JSON Schema 保证。

这些字段把“模型产生 reasoning/tool event”和“宿主如何执行、回传、审计工具”连接起来。reasoning content 不是权限凭证，tool call 不是已经执行成功的事实。

## 11. 可运行教学 demo：把四个问题放进一个小实验

下面的零依赖 Python 示例只模拟四个概念：IndexPool 的加权压缩、稀疏候选选择、递归状态、mHC 的 Sinkhorn 投影，以及 EPD/视觉验证的账本。它不加载 320B 权重，也不是生产 kernel；它的价值是让面试者能解释“为什么混合架构需要多本账”。

```python
import math


def dot(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("vectors must have the same non-zero length")
    return sum(a * b for a, b in zip(left, right))


def index_pool(keys, group_size=4):
    if group_size <= 0 or len(keys) == 0:
        raise ValueError("invalid key groups")
    pooled = []
    groups = []
    for start in range(0, len(keys), group_size):
        group = keys[start:start + group_size]
        weights = [float(i + 1) for i in range(len(group))]
        total = sum(weights)
        pooled.append([
            sum(weight * row[col] for weight, row in zip(weights, group)) / total
            for col in range(len(group[0]))
        ])
        groups.append((start, start + len(group)))
    return pooled, groups


def sparse_visible(query, keys, block_size=4, index_topk=2):
    pooled, groups = index_pool(keys, group_size=block_size)
    complete = []
    for block_id, (start, end) in enumerate(groups):
        # A block is eligible only when it is fully before the query.
        if end <= len(keys) - 1:
            complete.append((dot(query, pooled[block_id]), block_id, start, end))
    if not complete:
        return [], []
    complete.sort(reverse=True)
    selected = complete[:index_topk]
    token_ids = [token for _, _, start, end in selected for token in range(start, end)]
    return token_ids, [(block_id, round(score, 3)) for score, block_id, _, _ in selected]


def linear_state(values, decay=0.8):
    if not values or not 0.0 < decay <= 1.0:
        raise ValueError("values and decay must be valid")
    state = [0.0 for _ in values[0]]
    for value in values:
        if len(value) != len(state):
            raise ValueError("state width changed")
        state = [decay * old + new for old, new in zip(state, value)]
    return [round(item, 3) for item in state]


def sinkhorn(raw, steps=20):
    if not raw or any(len(row) != len(raw) for row in raw) or steps <= 0:
        raise ValueError("raw must be a non-empty square matrix")
    matrix = [[math.exp(value) for value in row] for row in raw]
    for _ in range(steps):
        for row_id, row in enumerate(matrix):
            total = sum(row)
            matrix[row_id] = [item / total for item in row]
        for col_id in range(len(matrix)):
            total = sum(matrix[row_id][col_id] for row_id in range(len(matrix)))
            for row_id in range(len(matrix)):
                matrix[row_id][col_id] /= total
    return matrix


def cache_ledger(sequence_length):
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    # Toy values: they are deliberately not GLM-5.3-Flash production values.
    explicit_layers = 2
    linear_layers = 4
    kv_heads = 2
    head_dim = 8
    bf16_bytes = 2
    explicit_kv = 2 * explicit_layers * sequence_length * kv_heads * head_dim * bf16_bytes
    recurrent_state = linear_layers * kv_heads * head_dim * head_dim * bf16_bytes
    return {"explicit_kv_bytes": explicit_kv, "linear_state_bytes": recurrent_state}


def epd_plan():
    return [
        ("encode", "image/video/file -> visual representation"),
        ("prefill", "text + visual representation -> KV/state metadata"),
        ("decode", "metadata -> streamed text/tool events"),
    ]


def visual_verification_loop():
    errors = [3, 2, 1, 0]
    for before, after in zip(errors, errors[1:]):
        if after > before:
            raise AssertionError("refinement made the visual diff worse")
    return errors[-1]


keys = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
visible, ranked_blocks = sparse_visible([0.95, 0.05], keys)
state = linear_state(keys)
projected = sinkhorn([[1.4, 0.2], [0.1, 1.3]])
assert all(abs(sum(row) - 1.0) < 1e-3 for row in projected)
assert all(abs(sum(projected[row][col] for row in range(2)) - 1.0) < 1e-3 for col in range(2))
assert visible and ranked_blocks
assert visual_verification_loop() == 0
print("visible_tokens:", visible)
print("ranked_blocks:", ranked_blocks)
print("linear_state:", state)
print("cache_ledger:", cache_ledger(16))
print("epd_stages:", [stage for stage, _ in epd_plan()])
print("visual_diff_after_refinement:", visual_verification_loop())
```

预期现象是：IndexPool 把 4 个 key 聚成一个候选表示，稀疏路径只返回排序靠前的完整 block；递归 state 的形状不随序列长度增长；Sinkhorn 后行和/列和接近 1；cache ledger 把显式 KV 和线性 state 分开；视觉 loop 只有在“观察—修改—重新观察”的差异不恶化时才宣称完成。示例的 `explicit_layers=2`、`linear_layers=4`、`kv_heads=2` 等都是教学数字，不能当作 Flash 的实际配置。

## 12. 面试问答主线

### 问题 1：GLM-5.3-Flash 为什么同时需要 linear attention 和 sparse attention？

回答：线性路径用固定形状的递归 state 降低多数历史处理的成本，稀疏路径通过 indexer 从远程历史中恢复少量显式 token，补足固定 state 的精确检索短板。它是按信息访问模式分预算，而不是宣称任意历史都能免费读取。

### 问题 2：IndexPool 的价值是什么？

回答：1M context 下 indexer 的 key 数量和随机访问本身会成为开销。IndexPool 把 4 个 indexer key vectors 加权压成一个候选表示，减少 indexer latency/memory；随后仍要做 query scoring、top-k、mask、gather 和验证，不能把 pooling 等同于最终 attention 的 token 选择。

### 问题 3：18B active 是否意味着它等价于 18B dense 模型？

回答：不是。18B 是发布方的每 token active 口径，total 320B 还决定权重容量和专家分片；attention、router、shared expert、dispatch/combine、padding、KV/state 和量化开销都要另算。

### 问题 4：mHC 和 IndexPool 是同一个稳定化技术吗？

回答：不是。mHC 约束多路 residual stream 的混合矩阵，使其接近双随机集合；IndexPool 压缩稀疏 indexer 的历史 key。一个作用在深度残差流，一个作用在长上下文检索候选。

### 问题 5：EPD 为什么不是简单的 PD？

回答：多模态请求在 prefill 前还有 encode 阶段。EPD 把视觉/文件编码、文本 prefill、逐 token decode 分开调度，能减少不同阶段互相干扰，但增加跨池传输、版本、取消、重算、流量和运维复杂度。

### 问题 6：DataCurve 63.392857% 能否说明模型代码能力？

回答：它说明在 `mini-swe-agent`、`max` effort、固定任务集、工具、环境和 verifier 组合下的 284/448 结果。要讨论基础模型能力，必须换成同一 harness/工具/任务/超时/版本的对照，不能把它当裸模型分数或与 Artificial Analysis 指数直接拼接。

## 13. 待核验项

当前仍缺少或不应从公开资料推断的内容：

1. 完整 GLM-5.3-Flash 技术报告中的训练数据配方、loss、后训练 objective、RL/环境细节和每层完整算子定义。
2. 生产 linear-attention/ReplaySSM、IndexPool、MoE dispatch、mHC 和 EPD 的完整 kernel、通信拓扑与 profiler trace。
3. `index_topk=2048` 在不同 batch/context/MTP 场景中究竟对应何种最终可见 token 预算，以及 MTP index 复用的准确性条件。
4. 线性 state、稀疏 KV、indexer 临时张量、mHC residual 和视觉中间结果的真实 bytes/token 与并发容量。
5. Z.ai 自报的 3.01×/4.44×、3× serving、DeepSWE/AutomationBench 与视觉工作流数字在目标硬件上的独立复现。
6. API 不同 endpoint 对 `clear_thinking`、`tool_stream`、JSON mode、缓存命中和多轮 reasoning replay 的真实错误码、限流和版本行为。
7. 公开 artifact 的权重分片、完整许可证文本、量化误差、目标硬件 profiling、线上接受率和第三方独立 benchmark。

后续补证时仍以本锚点已有官方 artifact 为范围；只有 Artificial Analysis 或 DataCurve 出现的新独立条目，才可以进入新的模型候选盘点。

## 14. 来源清单

1. [Artificial Analysis GLM-5.3-Flash](https://artificialanalysis.ai/models/glm-5-3-flash)：候选发现、配置级指数/速度/TTFT/价格和页面日期。
2. [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：`mini_swe_agent_glm_5_3_flash_max` 的配置级 Agent 评测。
3. [Z.ai GLM-5.3-Flash 官方文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)：模型概览、混合架构、视觉 coding、接口和 serving 摘要。
4. [Z.ai GLM-5.3-Flash 官方博客](https://z.ai/blog/glm-5.3-flash)：IndexPool、mHC、30T multimodal corpus、视觉 self-judgment、SGLang/EPD/量化和发布方 speedup 描述。
5. [GLM-5.3-Flash Hugging Face 模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a)：公开模型卡、许可证和运行入口。
6. [固定 revision `config.json`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)：层数、层类型、MoE、IndexPool、mHC、视觉和量化配置。
7. [GLM-5 Technical Report](https://arxiv.org/abs/2602.15763)：模型卡引用的关联技术报告；仅用于关联路线和术语，不把报告中未明确归属 Flash 的数字写成 Flash 内部事实。
8. [Z.ai Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)、[Streaming](https://docs.z.ai/guides/capabilities/stream-tool)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling)、[Context Caching](https://docs.z.ai/guides/capabilities/cache)、[Structured Output](https://docs.z.ai/guides/capabilities/struct-output)：API/Agent 协议核验。
9. [SGLang GLM-5.3-Flash cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash.md)、[vLLM recipes](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)、[Transformers GLM5-Next docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm5_next.md)：部署入口，不替代目标硬件实测；Transformers 当前文档不包含 MTP layer。
