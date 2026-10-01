# DeepSeek-V4.1-Flash 官方资料摘记

首轮核验日期：2026-09-14；固定 revision 实现复验：2026-09-20；当前时点联网复验：2026-09-23。本文把 DeepSeek 官方发布页、DeepSeek Harness 官方预览文档、Hugging Face 模型卡、模型配置和随仓库发布的编码/评测说明分开记录。技术报告 PDF 已下载、校验哈希并按页提取 51 页正文；报告中的内部实验和部署数字仍按发布方自报记录，不等同于本项目复现。

## 来源与快照

| 来源 | 作用 | 本轮状态 |
|---|---|---|
| [DeepSeek-V4.1-Flash 发布页](https://api-docs.deepseek.com/news/news260910) | 发布日期、API 模型名、兼容路由、非对称激活和 KV/cache 产品描述 | 2026-09-23 通过 `10.24.27.134:7890` 返回 HTTP 200，`27,838` bytes，SHA-256 `bea79d60a0712f1971554724c94e8ff2145a048d3c0e80326efa4bacd2bf8e11`；页面导航标注 `2026/09/10`。此前 `420cbb...` 快照保留为历史页面版本 |
| [DeepSeek Harness 快速开始](https://deepseek-harness.github.io/deepseek-harness/en/guide/quickstart) | Web UI、workspace、session、审批和 Agent 工作流 | 2026-09-23 HTTP 200，`83,023` bytes，SHA-256 `1e816c2bbead769f57b2c344334c02d6136eeb5831d46a9a3d5cc46f9f135966`；官方站点标为 Preview |
| [DeepSeek Harness model providers](https://deepseek-harness.github.io/deepseek-harness/en/guide/providers) | provider ID、密钥脱敏、协议兼容、模型发现和 reasoning/image 声明 | `113,025` bytes，SHA-256 `1761dd552158cfe6e01245744414ef5c81273f2ba09026d4a5b6e94a3065182d` |
| [DeepSeek Harness Python SDK](https://deepseek-harness.github.io/deepseek-harness/en/guide/python-sdk) | isolated home/workspace、profile、插件和 native runtime wheel | `110,733` bytes，SHA-256 `3459ca57567d8a3ff982674ac57a390ecd8215336a5e426d40e4b4ead6c51edf` |
| [DeepSeek Harness architecture/reference](https://deepseek-harness.github.io/deepseek-harness/en/reference/) | Cordis plugin tree、profiles/bundles、session log、agent loop和事件域 | `130,135` bytes，SHA-256 `3595d42386804b571bb5976592e07f10a1342764d64485c08e37ea1604b74837` |
| [DeepSeek Harness MCP memory](https://deepseek-harness.github.io/deepseek-harness/en/guide/mcp-memory) | MCP namespace、secret 环境过滤、重连和工具注销 | `96,150` bytes，SHA-256 `43751eb6e0f5a363397d26844c2068a6d820adf6d65cbf8381fe993d3764fa33` |
| [DeepSeek Harness GitHub review](https://deepseek-harness.github.io/deepseek-harness/en/guide/github-review) | signed webhook、异步 admission、workspace 和重复投递语义 | `94,772` bytes，SHA-256 `3dfae298c9af189c1965a1ff19e78ccbab3f575fd98b0f1ced45925497b0f31f` |
| [Hugging Face 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 架构、训练、评测、协议和开源入口 | revision `dba1be0a40aa45a94ad051997016db3960a90277`；README 快照 SHA-256 为 `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251` |
| [模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) | 可机器读取的结构字段 | `DeepseekV41ForCausalLM`、`deepseek_v41`；配置快照 SHA-256 为 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| [技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | 架构、训练基础设施、推理系统、后训练和评测细节 | 51 页正文已逐页提取；SHA-256 为 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d` |
| [encoding README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md) | prompt 格式、工具标签和 reasoning effort | 已读取；快照 SHA-256 为 `a2f0fc3baea318c9cfbceca68cbfe50d37cf7da6605ace887f33148bcff7e3ae` |
| [evaluation README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/evaluation/README.md) | DeepSWE 复现实验的 harness 说明 | 已读取；明确区分 `mini-swe-agent` 与 `dsh-minimal` |

页面哈希只用于识别本轮资料快照，不代表页面或模型权重永久不变。模型卡 revision 是权重/仓库的版本锚点，不能用首页当前内容替代。

## 2026-09-23：官方发布页与 DeepSeek Harness preview

本轮通过已验证的 `10.24.27.134:7890` 重新取得 DeepSeek V4.1-Flash 发布页和官方 DeepSeek Harness 文档。这里把模型发布合同、Harness runtime 和本地教学审计分开，避免把 Harness 的工程能力写成 V4.1 的内部网络结构。

### 发布页新增的产品级确认

官方发布页用产品层语言补充了模型卡的架构账本：V4.1-Flash 是 552B MoE，采用 Causal Encoder--Decoder（CED），prefill 每 token 约 8B active、decode 每 token 约 16B active；发布页同时宣称相对上一代约需要 `1/4 HBM` 和 `1/8 SSD` 的 KV/cache 存储。这里的 `8B/16B`、HBM 比例和 SSD 比例分别来自激活与产品存储口径，不能合并为单卡显存或端到端加速保证。

发布页还确认了服务迁移边界：首选 API model name 是 `deepseek-flash`；旧的 `deepseek-v4-flash` 与 `deepseek-v4-flash-vision-exp` 在兼容期路由到 V4.1-Flash；`deepseek-v4-pro` 从 2026-09-14 04:00 UTC 起临时路由到 V4.1-Flash，直到 V4.1-Pro 发布。客户端必须同时记录 requested alias、served model、时间、价格和响应能力，不能把 alias redirect 当作权重身份。

### Harness 的可观察责任边界

官方 DeepSeek Harness 文档把它描述为构建 Agent Harness 的插件化 SDK，并将 Web UI、headless/SDK profile、workspace、session、agent loop、tools 和 MCP 放在同一 Cordis plugin tree 中。可用于面试的事实和边界如下：

1. **Provider 与 session identity**：provider ID 是持久身份，已保存请求、session、默认模型和 credential reference 都依赖它；改名应新建 provider 后删除旧 provider。保存 key 后 UI 只返回脱敏描述，不能把明文 secret 回显。已有请求的 session 保留其日志中的 model，不能在中途静默切换 provider/model。
2. **协议兼容不是探测结果**：内置 provider 使用安装时 catalog；custom provider 可以声明 `openai-completions`、`openai-responses` 或 `anthropic-messages`。reasoning level、image input、header、timeout、retry 和兼容开关属于 route configuration。文档明确说手工声明的 image/reasoning 是对 endpoint 的 claim，不能替代真实 capability probe。
3. **插件和生命周期**：插件通过 `apply` 注册配置与服务，依赖未满足时不应加载；由 `ctx` 注册的 listener/resource 在卸载时自动清理，外部连接等资源要使用显式 effect cleanup。profile 的 bundle 顺序和 patch 路径决定最终运行树，不能把“插件源码存在”当成应用已经加载。
4. **session log 是上下文来源**：reference 文档把 session log 作为模型可见 context 的来源，并通过 `turn/*`、`step/*`、`tool/*`、`agent/*` 和 `llm/stream` 事件扩展 runtime。恢复时应保留 workspace、工具、权限、模型和 session 版本；一个可读摘要不能替代原始状态或副作用 receipt。
5. **MCP 重连和最小权限**：MCP 工具使用 `mcp__<server>__<tool>` 命名空间；stdio bridge 会过滤通常表示 credential 的环境变量及 `DSH_*` 变量。断线时调用失败，重连会重新发现/同步工具；达到重连预算后工具被注销，不能继续使用旧 registry。文档给出的 reference memory 还明确是本地 JSONL/substring search，不自动提供 embedding、摘要、冲突解决或 forgetting policy。
6. **Webhook admission 不等于任务完成**：GitHub review overlay 使用签名 webhook，把请求送入 workspace/session；HTTP `202` 只代表规则接纳，不能代表 Agent 已完成、测试已通过或 artifact 已 verified。官方文档还说明 webhook runtime 不保存 delivery/execution state，重复投递可能创建多个 session；入站 webhook secret 认证不自动授予 Agent 对 GitHub 的出站权限。

这些内容是 DeepSeek Harness 的官方 preview/runtime 合同，不是 V4.1 的参数、训练 recipe、DSpark acceptance 或生产 SLO 证据。它们适合放入面试中的“模型 API -> Harness -> tool/MCP -> workspace -> verifier”责任图。

### 本地协议审计

新增 [`deepseek_harness_protocol_audit.py`](code/deepseek_harness_protocol_audit.py)，只使用 Python 标准库和合成事件，验证 provider 脱敏与稳定 ID、session model 固化、plugin dependency/cleanup、MCP credential 过滤与 reconnect budget，以及 webhook `202`/重复投递/入站认证与出站权限分离。它不启动 DeepSeek Harness、不调用 API、不运行 MCP 或 GitHub，证据等级固定为 **local_protocol_toy**。

## 2026-09-20：固定 revision 的 HF inventory 与 reference implementation

本轮通过可用代理 `10.24.27.134:7890` 读取固定 revision 的 Hugging Face API metadata 和公开源码；没有下载任何 safetensors 权重。API 响应本身为 6,687 bytes，SHA-256 为 `fb3aefa7794da9101d0253ccc4e6e36fbace841778f6857b1370cb2880235d63`，记录的 `lastModified` 为 `2026-09-10T08:18:10Z`，revision 仍为 `dba1be0a40aa45a94ad051997016db3960a90277`。

API inventory 列出 48 个 safetensors 分片，并报告 dtype bucket 的参数计数合计 `763,205,315,794`（BF16 `1,976,441,856`、F32 `42,307,282`、F8_E4M3 `204,015,223,296`、I8 `557,171,343,360`）。这是仓库清单的 dtype/存储元数据，不是去重后的架构参数定义，不能覆盖模型卡的 `552B backbone`，也不能据此推导单卡显存；完整权重仍未下载。

固定 revision 的关键文本 artifact 哈希如下。哈希用于锁定源码快照，不等于已经在本机执行了 CUDA/TileLang 推理。

| artifact | bytes | SHA-256 |
|---|---:|---|
| `inference/README.md` | 2,029 | `2834402823199ee24e9a42bdf36a0fc6daf94448f444cb062c042a057a798f1c` |
| `inference/config.json` | 1,982 | `2e84f45cf1dac8c7fcbb200e96667d4b913275690ed496f24c7747207a809` |
| `inference/convert.py` | 9,458 | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` |
| `inference/engram.py` | 8,138 | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `inference/kernel.py` | 23,790 | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `inference/model.py` | 61,549 | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `inference/generate.py` | 8,722 | `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` |
| `inference/image_processor.py` | 7,699 | `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272` |
| `inference/vision.py` | 4,457 | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| `evaluation/README.md` | 3,977 | `b1367cba184ce632e1e24e4dfc29cf40b02fdc297ef316bb82599e38eba6dae0` |
| `evaluation/dsh-minimal.patch` | 28,725 | `11f934726bdffb2111072c7dc121fe4734f5948311dad1ab9aa4b76e2ae2f24c` |
| `encoding/encoding.py` | 37,316 | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |

### 实现证据能确认什么

- `inference/config.json` 是一个可运行路径的 runtime snapshot：包含 `n_mtp_layers=3`、`dspark_block_size=5`、DSpark target layers `[37, 38, 39]`、Markov rank `256`、DSpark `128/3` routed/active experts、`window_size=128`、`index_topk=512`、`candidate_topk_blocks=2048`、`candidate_block_size=8`、`hc_mult=4`、20 次 Sinkhorn、Engram layers `[1, 14]` 和 FP4 expert dtype。它比模型卡高层名称更接近参考代码的执行账本，但仍绑定该 revision 和该 runtime。
- `model.py` 显式实现 ring-buffer 的 SWA window、压缩 KV 的 overlap state、两级 candidate/index top-k、sparse attention、MoE、Engram、Hyper-Connections/Sinkhorn 和 DSpark block；`kernel.py` 提供 block FP8/FP4 quantization、FP8/FP4 GEMM、sparse online softmax 和 Sinkhorn 的 TileLang 路径。源码存在证明“参考实现公开”，不证明所有生产 kernel、召回率或目标硬件性能已经公开。
- `convert.py` 将 HF safetensors 按 model-parallel rank 分片，按权重名称推断 backbone/MTP expert 数，并可选择 FP4 或 FP8 expert 路径；`inference/README.md` 给出 `MP=8` 转换示例，但示例并不等于我们已经下载权重或完成多卡运行。
- `generate.py` 的实际生成循环调用普通 `model.forward`，README 也明确称 generation 是 plain autoregressive sampling。源码虽有 `forward_spec`/DSpark forward path，但本参考生成入口没有 acceptance scheduler、目标模型 verify/rollback 账本，因此不能把“DSpark 模块存在”写成“本地已复现 speculative throughput”。
- `evaluation/README.md` 和 `dsh-minimal.patch` 把 `mini-swe-agent` 与 `dsh-minimal` 分开；它们是评测 harness/环境补丁，不是 V4.1 裸模型分数。模型卡中的 DeepSWE 数字仍必须绑定任务集、工具、容器、verifier、effort、超时和 scaffold。

本轮本地只对下载的 Python 源码做 `py_compile`，并对 `encoding.py` 做 DSML 前导空格、`reasoning_effort=max`、中途 system message 和 thinking parse 的 smoke test；没有 CUDA、TileLang、完整权重或线上 API 推理验收。

## 技术报告逐页核验

本轮从固定 revision 下载报告并用本地 PDF 解析器逐页读取 51 页；报告书签包含 Architecture、General Infrastructures、Pre-Training、Post-Training、Evaluation 和附录。下面只记录报告明确写出的补充事实，页码指 PDF 页码，不把图表中的发布方结果改写成独立实测。

### 架构与精确排布

- CED 的 global attention 路径把底部 `L/2` 层作为 causal encoder；decoder 层从第 `L/2` 层 hidden state 通过逐层投影得到 global KV 和压缩权重。SWA 仍在每层从当前 hidden state 计算；报告给出的长序列复杂度近似为 `O(NL/2 + n_win*L/2)`（PDF p.9）。
- CSA2 去掉 CSA 中相邻压缩条目的重叠源 token 和绝对位置编码，并从 main KV 投影 indexer K；`Full`、`Reindex`、`Reuse` 的跨层依赖和 HSI 的候选池流程见 PDF pp.10-12。
- 训练设置给出明确层排布：encoder 前两层为纯 SWA，余下 18 层按三个六层组使用 `Full + 5 Reuse` 的 CSA2、压缩率 `m=2`；decoder 的 20 层按五个四层组使用 `Full + 3 Reuse` 或 `Reindex + 3 Reuse`，压缩率 `m=1`。HSI 最多选 2,048 个八位置 block，即最多 16,384 个候选位置；最终 sparse attention 仍选 Top-512（PDF pp.11-12, 22）。
- Single-Pass mHC 将当前层 input mixing 使用的系数从 `A_l` 移到前一层的 `A_{l-1}`，消除跨完整 hidden reduction 的依赖；报告称部署 kernel 的 activation traffic 从原实现的约 `(4n+4)d` 降到 `(2n+2)d`，预训练仍保留原多 kernel 路径（PDF pp.12-13）。
- Engram 的报告实现描述为两个均分参数的模块，使用 n-gram `{2,3,4}`、每阶 8 个 hash head、每阶总 embedding dimension 2048、约 16M 条目的不同素数表，embedding 与投影为 FP8，模块放在 zero-indexed 的第 1 和第 14 层；报告还描述了 host memory RDMA 预取和省略 short causal convolution（PDF p.13）。
- DSpark 使用三个 Transformer drafter block、128 token 的 SWA，一次前向并行产生 5 个 draft positions，并用 Markov head、confidence head 和吞吐曲线调度验证长度；backbone 预训练后单独训练 DSpark，后训练时随 backbone 更新但不把 DSpark loss 梯度传回 backbone（PDF pp.13-14）。

### 量化、训练与部署

- main KV 的 FP4 采用 E2M1、每 16 个 channel 一个 E4M3 scale；报告说明这是 post-training QAT，cache 在 RoPE 后量化，attention 前反量化，SWA KV 保留 FP8，并省略 NVFP4 的第二级 global scale（PDF p.14）。
- 多模态训练基础设施把 vision encoder 放在 LLM 参数树之外，按 vision forward、LLM forward/backward、vision backward 三阶段执行；报告还描述 image sharding、CSA2 的 shadow indexer/pipeline payload/shared-state 管理和 Engram row sharding/prefetch（PDF pp.16-18）。
- 推理部署采用 Encoder-Prefill-Decode（EPD）解耦；报告称多数 CSA2 Reuse layer 需要 15 个 prefill kernels 和 11 个 decode kernels。其持久化策略把 SWA KV 移出持久 KV cache，放入每台机器约 10% host DRAM 的短 TTL 分布式池，而 global KV 的报告配置保证至少 72 小时生命周期（PDF pp.18-20）。这些是报告描述的部署实现，不是任意硬件上的 SLO。
- 预训练设置给出 45T token、固定 100.6M token batch、64K 稀疏 attention 起训并在 34T token 扩展到 1M；视觉编码器另有约 47B image-text pairs 的对比预训练和 236B token 的高分辨率自回归微调阶段（PDF pp.21-23）。

### 后训练与评测限制

- 报告把 Agent 任务形式化为 `(problem, environment, verification system)`，强调任务合成、环境构造、质量复审和 RL 数据扩展，而不是新的后训练算法；最后的全词表 OPD 使用 40 多个 teacher models（PDF pp.25-32）。
- DSec 将 sandbox 与 worker 分离，用 scale units、最终一致的 placement 和本地 admission 控制大规模并发；报告报告过单机从约 1,000 到超过 2,500 个 live containers 的密度变化，并用 AppArmor 与 eBPF 网络策略处理越权、破坏文件系统和 reward hacking（PDF pp.27-29）。这些数字和安全措施仍是作者环境中的报告结果。
- 报告 Table 1/3 补充了 base 与 Agent 结果，但评测使用内部框架、指定 scaffold、温度/top-p、context、样本数、容器和 verifier；报告结论自身也承认 CSA2 选择错误和近似 SWA replay 在未测试边界上可能造成能力下降（PDF pp.23-24, 32-37）。

## 可以写成事实的模型信息

### 架构和计算路径

模型卡称 DeepSeek-V4.1-Flash 是带原生图像输入的 MoE 模型，backbone 为 552B 参数，最长上下文为 1M token。其语言主干采用 Causal Encoder-Decoder（CED）：共 40 层，前 20 层是 causal encoder，后 20 层是 decoder。模型卡进一步说明 decoder 的 global KV cache 从 encoder 最终 hidden states 投影得到，而不是由每个 decoder 层自己的 hidden states 逐层产生。

公开说明给出两个不同的每 token 激活账本：prefill 每 token 约 8B，decode 每 token 约 16B。这里的 8B/16B 是激活参数口径，不是 FLOPs、显存或总参数；prefill 与 decode 的 token 数、算子和通信模式也不同。

模型卡将 SWA Bounded Replay 描述为：不把滑动窗口注意力的 KV 全部持久化到 SSD，而是在需要时只重放最近的 `n_win` 个 token 来重建缺失状态；持久 KV footprint 约为 DeepSeek-V4-Flash 的 1/8。官方发布页以产品层语言给出 V4.1-Flash 相比上一代约 1/4 HBM、1/8 SSD 的说法。二者的比较对象、指标名称和部署层级不同，不能合成一个没有基线的“总压缩率”。

### CSA2、层间复用和 KV 量化

模型卡把 Compressed Sparse Attention 2（CSA2）拆成三种静态层模式：`Full`、`Reindex`、`Reuse`。它们用于在不同层之间共享 main KV 与 indexer K，并复用 Top-K 稀疏注意力索引。decoder 还有 Hierarchical Sparse Indexer：后续 indexing layer 使用第一个 Full Mode layer 构造的候选池，从而把更深层的 indexer 成本限制在候选池规模，而不是随完整上下文长度无界增长。

模型卡称 main KV 使用 FP4 cache，格式为 E2M1，每 16 个 channel 共用一个 E4M3 scale；global KV cache footprint 为每 token 890 bytes，约为 DeepSeek-V4-Flash 的 1/4。这里的 890 bytes 是 global KV 的公开口径，不是单请求全部 HBM，也不包含所有 indexer、SWA 尾部、workspace、通信 buffer 和权重。

### MoE、残差和条件记忆

配置给出每个 MoE 层 1 个 shared expert、384 个 routed experts，每 token 激活 6 个 routed experts。总参数、激活参数、专家 dispatch、all-to-all 通信和负载均衡分别属于不同账本，不能仅用 active parameters 估算整机内存或尾延迟。

模型卡还列出三类额外组件：

- Single-Pass mHC，用于残差流混合，并配套高效的 Mega-mHC kernel；
- Engram conditional memory，披露为 196B 参数、通过 token-based lookup 稀疏访问；
- DSpark speculative decoding，描述为半自回归草稿生成与按置信度调度的验证。

这些名称和高层作用来自模型卡；它们的完整数学定义、训练损失、硬件 kernel 细节和线上接受率不能由名称反推。本仓库配置还提供了 `num_nextn_predict_layers=3`、DSpark 目标层 `[37, 38, 39]` 等实现快照字段，但这些字段不等价于一个跨后端的吞吐保证。

### 原生多模态和训练

视觉路径使用从头训练的 DeepSeek-ViT，采用 2D-RoPE、3x3 pixel-unshuffle 下采样和两层 MLP projector，把视觉 embedding 与文本 embedding 一起送入语言模型预训练。配置快照给出视觉编码器 32 层、hidden size 1024、16 个 attention heads、patch size 14、downsample ratio 3、最多 1024 个 image tokens；这些是该 revision 的配置字段。

模型卡称多模态预训练语料包含 45T tokens；稀疏注意力在 64K sequence length 上训练，并使用 34T tokens 将上下文扩展到 1M。这里的“训练长度”和“服务端最大输入”必须分开记录：后者是接口容量，前者是训练分布和泛化证据的一部分。

后训练部分，模型卡写明总体流程为 `SFT -> RL -> on-policy distillation (OPD)`，没有宣称算法级改动；实质变化被描述为自动合成 Agent 任务、环境与 rollout 的数据管线，以及任务、数据和 rollout 的渐进扩展。官方发布页则用更概括的语言提到新的预训练方法和更大规模 RL。两段话处于不同粒度，不能据此编造具体奖励函数、优化器、教师结构或 loss 权重；报告明确披露的“40 多个 teacher models”只作为报告训练设置记录。

## 配置字段摘录

以下字段来自该 revision 的 `config.json`，用于教学和部署账本，不替代模型卡对机制的叙述：

| 字段 | 值 | 解释边界 |
|---|---:|---|
| `hidden_size` | 5120 | 主干 hidden size |
| `num_hidden_layers` | 40 | 与 20 encoder + 20 decoder 的公开架构描述一致 |
| `num_attention_heads` | 64 | attention head 配置 |
| `num_key_value_heads` | 1 | 配置字段；不能把整套 CSA2 直接简化为普通 MQA |
| `head_dim` | 512 | attention head dimension |
| `max_position_embeddings` | 1048576 | 1M token 配置上限 |
| YaRN `factor` | 16 | 长上下文位置配置字段 |
| `n_routed_experts` | 384 | routed expert 数 |
| `n_shared_experts` | 1 | shared expert 数 |
| `num_experts_per_tok` | 6 | 每 token routed expert 选择数 |
| `sliding_window` | 128 | 配置中的局部窗口字段，不等于所有注意力层都只看 128 |
| `index_n_heads` / `index_head_dim` | 32 / 128 | indexer 配置 |
| `index_topk` | 512 | indexer Top-K 配置 |
| `candidate_topk_blocks` / `candidate_block_size` | 2048 / 8 | hierarchical candidate pool 相关字段 |
| `engram_max_ngram_size` | 4 | Engram 配置字段 |
| `num_nextn_predict_layers` | 3 | DSpark/预测层配置字段 |

## Prompt、工具和 reasoning effort

仓库没有提供 Jinja chat template，而是提供独立的 `encoding/encoding.py` 与测试。V4.1 相对 V4 的公开格式变化包括：

1. DSML 工具标签带前导空格，例如 `<｜DSML｜ calls>`、`<｜DSML｜ invoke>`、`<｜DSML｜ parameter>`；不能把 V4 的无空格标签原样复用。
2. thinking 模式支持整数 `reasoning_effort`，范围为 1--100；字符串别名映射为 `low=50`、`high=75`、`max=100`，默认 `high=75`。
3. 支持中途 system message；它使用 `<｜System｜>`，并影响后续 assistant generation header 的拼接。

`deepseek-recipe` 提供 Rust/Python 协议工具包，可在 Messages、Chat Completions、Responses 与模型 prompt 之间转换，并解析 thinking、工具调用、图像和流式输出。它负责协议编码，不替调用方执行工具、发起 HTTP 请求或授予宿主权限。

官方 API 发布页标明 API 模型名为 `deepseek-flash`。页面还写明 `deepseek-v4-flash` 与 `deepseek-v4-flash-vision-exp` 在兼容期暂时路由到 V4.1-Flash，并从 2026-09-14 04:00 UTC 起将 `deepseek-v4-pro` 请求路由到 V4.1-Flash，直到 V4.1-Pro 发布。路由策略属于带时间语境的服务端行为，客户端应记录实际返回的 model 字段和日期。

## 官方自报评测及其边界

模型卡的 base 表给出：MMLU-Pro 74.1、HumanEval 79.4、GSM8K 93.0、MMMU-Pro 56.5、DocVQA 95.6。Instruct/Agent 表给出 Terminal-Bench 2.1 Pass@1 90.6、DeepSWE v1.1 Resolved 74.2、AutomationBench 54.8 和 Agent's Last Exam 31.8。

这些数字是发布方模型卡中的结果，不是本项目独立复现。模型卡明确绑定了条件：instruct 结果使用 `reasoning_effort=100`、`temperature=1.0`、`top_p=0.95`；代码 Agent 使用 DeepSeek Harness 的 Minimal mode 和 1M context，DeepSWE v1.1 对齐 `mini-swe-agent`，视觉 Agent 使用 Claude Code harness 和 512K context，其他任务使用各自官方 scaffold。Agent 分数观测的是模型 revision、effort、工具、harness、环境、verifier 和超时策略的组合。

因此正文只把这些数字作为“官方自报、协议绑定的参考点”，不把 74.2 写成基础模型单独能力，不将它与不同 harness 的 DeepSWE、SWE-bench 或 Artificial Analysis 指数拼接成一张排行榜。

## 2026-09-20：CPU 教学实验补充召回与 FP4 误差账本

新增 [`deepseek_v41_cache_demo.py`](code/deepseek_v41_cache_demo.py)，文件大小 `6,317` bytes，SHA-256 为 `6c65d44f2f092369ac8b0cdf83a58fe4c2dad38e46698e5d7f1badc209c9517e`。脚本只依赖 Python 标准库和合成分数/向量，不加载 V4.1 权重、不调用 CUDA/TileLang，也不模拟生产 DSpark scheduler。

实验把层次化 indexer 的两个召回门拆开：先按 block 最大分数构造 candidate pool，再在候选池内做最终 Top-K；同时用分组 E2M1-like codebook 计算 toy KV 向量的 MSE 和最大绝对误差。固定合成输入的输出为 candidate recall `0.600`、conditional Top-K recall `0.667`、端到端 recall `0.400`、toy FP4-like MSE `0.025862`、最大绝对误差 `0.300000`。这些结果只证明指标分解和代码可运行，不能替代真实 gold-evidence recall、FP4 logits/质量对照、acceptance length 或 GPU profiling。

这项实验补足了面试中的一个重要边界：`candidate recall × conditional Top-K recall` 才能近似描述两级检索的端到端证据召回；仅报告最终 Top-K 或 cache bytes 都不足以证明稀疏注意力和 FP4 cache 的收益。

## 2026-09-21：`deepseek-recipe` 固定 commit 的协议与流式实现证据

本轮补齐模型卡和 V4.1 `encoding/encoding.py` 之外的官方协议工具链，固定 [DeepSeek 官方 `deepseek-recipe` 仓库](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea) 到 commit `8cadfede7063c896b944e7bae05daa3549ae97ea`。官方 `main.atom` 的最新条目是 2026-09-10 10:39:01 UTC 的 `docs: update project documentation`；提交记录快照为 2,280 bytes、SHA-256 `70e83744340349e637a6569557439a493d6ce5483540d172ea2a6560c255fa05`。固定 commit 的源码归档为 3,996,119 bytes、SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`，不是可变 `main` 页面或模型权重。

关键文件快照如下：

| artifact（固定 commit） | bytes | SHA-256 |
|---|---:|---|
| `README.md` | 5,261 | `0cccc69baa118d7689fc3ff2c2e47ab652e05410b777744c43a424f4db5fc0af` |
| `docs/streaming.md` | 3,495 | `654091c8f5b79075c5232bcf8314d5a1baeea540a1d472496ba1de40493c5c8d` |
| `docs/tokenizer.md` | 6,918 | `9fb1dd444abe6729c9cafe29a2e3766d9e14b02a28e68e07d06e6d9aaac0c0db` |
| `deepseek-recipe/src/request/mod.rs` | 3,639 | `3481c3a6cfee9b98063831314ee1b96eac3e7dc121d65984060c535bc699ac07` |
| `deepseek-recipe/src/stream/state_machine.rs` | 18,527 | `e6b65c2d2d91191ac97378376bd666b537db32147f6c522a2363f4cd2e080154` |
| `deepseek-recipe/src/stream/processor.rs` | 14,029 | `7d3f35c6278c5fdd8b33592db9506d1b2eef28f80ebb555353e900941eea4c1e` |
| `deepseek-recipe-encoding/src/v4/dsv41.rs` | 2,072 | `3408554e8a4ade05e034231cab8f87efcee7459d7fdd28fbb926f652e3b56786` |
| `deepseek-recipe-image/src/resolver.rs` | 15,977 | `1ef5e98d262cb6ba96af35173e39551d147521aa323bc1d55d8e75384779f367` |

### 协议规范化层，而不是推理引擎

README 将仓库定位为 Rust libraries + Python bindings：把 Messages、Chat Completions 和 Responses 转换为共享的 `Conversation`，再渲染成 DeepSeek V4/V4.1 prompt 或 token IDs，并把后端输出转换回目标协议。源码的 `ConversationRequest` 进一步保存 `InferenceOptions`、`ParsingOptions`、原始 model、stream 标志和共享 conversation；未指定的温度、`top_p`、token budget 等仍由调用方/模型后端决定。这一层次让“API 兼容”不再等于“模型推理已经实现”。

固定 commit 的执行链可以写成：

```text
protocol request
  -> validate/convert
  -> ConversationRequest
  -> V4.1 prompt rendering
  -> optional tokenizer -> backend inference (external)
  -> InferenceChunk
  -> streaming state machine / StreamProcessor
  -> Messages, Chat Completions, or Responses events
```

这条链支持思考、工具调用、图像和流式响应，但官方 README 明确把模型推理、工具执行和 HTTP transport 留给外部应用。仓库同时列出当前不支持的协议能力：`logprobs`/`top_logprobs`、document/audio/video/file retrieval、server-side `web_search`、JSON Schema/正则约束和 `strict` 强制、`n > 1`、除 `apply_patch` 外的 Responses custom tool、`previous_response_id` 存储恢复以及 `encrypted_content`。这些是该 commit 的适配器边界，不能反推 V4.1 模型本身不具备相应能力。

### 流式解析是有状态的，不是按 chunk 做字符串替换

`stream/state_machine.rs` 把 `<think>`、DSML tool-call 标签、JSON fence/raw JSON、stop sequence 和普通回答拆为 `Common`、`Json`、`ToolCalls`、`ToolName`、参数解析、`Reasoning`、`Finished` 等状态。`feed()` 会保留跨 chunk 尚未闭合的 marker，`finish()` 再处理输入尾部；工具参数被分成 name/type/value，字符串值和非字符串值走不同转义路径。`StreamProcessor` 再把这些 segment 与 backend 的 `InferenceChunk`、token usage 和 finish reason 组合成协议事件。

因此面试中要问的不是“能不能 parse 一段完整字符串”，而是：标签被拆在两个 SSE/token chunk 之间时是否丢失；reasoning、tool markup、JSON 和 stop sequence 是否分别计入输出；token ID 流是否附着了匹配 tokenizer；异常结束时 parser 是否重复发 tool call 或把未验证 JSON 当成有效结构。仓库的 parser 能把文本分段并生成协议事件，但不执行工具、不校验业务结果，也不充当 verifier。

### tokenizer、图像和安全门禁仍是独立账本

`docs/tokenizer.md` 明确要求先渲染 prompt，再显式附加与模型匹配的 tokenizer；由于 prompt 已经含有 special-token 文本，编码时关闭 tokenizer 的额外 special-token 注入。没有 tokenizer 时，`encode` 或 token-ID 流会返回错误；这可以避免把 tokenizer mismatch 隐藏为模型质量回归。

图像路径由 `deepseek-recipe-image` 负责 URL/data URL/bytes 的解析、并发 resolve、重试、字节预算和 OpenCV 预处理。固定 commit 的默认 `ImageLimits` 是最多 600 张图、单图 32 MiB、单请求 64 MiB、每次 resolve 最多 8 个并发源；HTTP fetcher 默认 5 次重定向、10 秒连接超时和 60 秒请求超时。源码特别警告默认 fetcher 不过滤私网、loopback 或 link-local 地址，因此接入外部 URL 时仍必须由宿主增加 SSRF 防护；“支持图像 URL”绝不等于“可以安全地抓取任意 URL”。

仓库的 `server-py`/`server-rs` 只是带 mock inference 的示例应用，展示如何把准备好的 prompt、图像和 `InferenceChunk` 接到协议层；它不下载权重，也不证明 V4.1 的 CUDA/TileLang、线上 alias、工具 acceptance 或生产 SLO 已经验收。本轮只完成固定 commit 的源码静态阅读与哈希核验，没有把示例服务器运行结果写成模型推理结果。

## 待核验与后续动作

- 技术报告正文已逐页读取并纳入上面的页码记录；报告没有公开完整 production kernel source、所有权重/参数分片、线上接受率或目标硬件上的独立 profiling。固定 HF 仓库另有 reference `inference/kernel.py`，其证据边界见上面的 2026-09-20 小节。
- 公开资料没有给出完整参数分片与服务端并发账本；552B backbone、196B Engram 和权重仓库 tensor dtype 不能简单相加后当作单卡显存。
- CED、CSA2、SWA Bounded Replay、Engram、mHC 和 DSpark 的生产级 kernel、容错、接受率和硬件依赖需要按 revision/后端复测。
- API 价格图片、账户限流、流式错误码、实际 alias 路由和多模态 token 计费需要通过实时 API 文档或不产生费用的接口探测继续核验。

## 书系映射

- 第二十一册第 81 章：CED、SWA Bounded Replay、CSA2、Hierarchical Sparse Indexer、FP4 KV、MoE、Engram、DSpark 和原生多模态。
- 第四册第 19 章：作为架构百科条目，强调模型卡证据与推理部署边界。
- 第五册：45T 多模态预训练、SFT -> RL -> OPD 与 Agent 数据管线。
- 第六册：1M context、异构 KV/cache tier、HBM/SSD/重算和 numeric reasoning effort 预算。
- 第七册、第十七册和第二十册：官方 Agent benchmark 的 harness、verifier、工具权限、恢复和复现账本。

## 2026-09-22：三代理当前时点复验与 revision 边界

本轮重新发起此前可能落在夜间工作时段的联网任务。Artificial Analysis 中文首页通过三条代理均返回 HTTP 200，快照为 `1,762,420` bytes、SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DeepSWE 页面三条代理也均返回 HTTP 200，快照为 `268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。这两个页面没有产生新的重点厂商 canonical 模型。

沿既有榜单锚点复验时，DeepSeek V4.1 发布页三条代理均返回 HTTP 200，快照为 `27,838` bytes、SHA-256 `bea79d60a0712f1971554724c94e8ff2145a048d3c0e80326efa4bacd2bf8e11`；Artificial Analysis 的 [`deepseek-v4-1-flash`](https://artificialanalysis.ai/models/deepseek-v4-1-flash) 详情页三条代理均返回 HTTP 200，快照为 `3,948,908` bytes、SHA-256 `114cc90d1cb8125174d9464141cbfd0faeac76f52b132f78630c0375c9fcf8fe`。当前 AA 条目仍是 `deepseek-v4-1-flash`，Intelligence Index 为 `39.456167472527`，约 1M context，第三方价格约 `$0.30/$1.20`；这些是榜单/provider 字段，不是本项目实测。

DataCurve 当前没有精确的 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此不把 V4 Pro/V4 Flash 或其他 DeepSeek 版本的 Pass@1、成本、输出 token、steps 和 verifier 结果迁移给 V4.1-Flash。这里的“无 DataCurve 行”是榜单覆盖边界，不是模型能力为零。

Hugging Face API 经 `10.24.27.134:7890` 成功返回 `6,714` bytes，SHA-256 为 `df3cb8b368d3a77a4eb3b96c8a4f85abfb3a199245bf9006a68d374a4b425ca3`；revision 仍为 `dba1be0a40aa45a94ad051997016db3960a90277`，`lastModified` 仍为 `2026-09-10T08:18:10Z`，文件清单仍包含 48 个 safetensors 分片。8098 对该 API 返回 503/连接失败，1234 未成功；这只能记录为代理线路边界，不能解释成 HF 页面或 revision 不存在。`deepseek-recipe` README 经 1234/7890 仍可获取，8098 对 GitHub raw 出现 TLS EOF，内容没有观察到变化。

本轮结论是：已确认的 V4.1-Flash canonical identity、官方发布页、HF revision、reference inference、技术报告和 recipe commit 均未产生新的模型或 revision；当前状态保持“内容专题 + reference implementation + recipe protocol evidence（AA 单榜）”。后续仍只补完整权重加载、production kernel、candidate/index Top-K recall、真实 FP4 误差、DSpark draft/verify/rollback、EPD 调度、目标硬件 profiling、tool acceptance 和独立 benchmark，不把 reference 代码或 README 写成生产验收。

## 2026-09-22：vLLM upstream V4.1 专用 runtime 补证

本轮仍以 Artificial Analysis 的 `deepseek-v4-1-flash` 为模型发现锚点，没有从 vLLM 仓库另发现模型。通过 1234 和 7890 代理取得的 vLLM `main` registry 内容逐字节一致；8098 对同一组 raw 文件超时。这里的 `main` 是未绑定 release tag 的 upstream 实现快照，必须与 vLLM `0.29.0` stable source 分开记录。

### 快照与入口

| artifact | 作用 | bytes / SHA-256 |
|---|---|---|
| [vLLM main `registry.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py) | 专用 model registry 入口 | `64,391` / `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e` |
| [vLLM main `deepseek_v41/__init__.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v41/__init__.py) | V4.1 NVIDIA/ROCm 包分流 | `625` / `6b928f07c6f67f4fd1b599a143cd15a1309a177c877f7f08b29d5fa30db659fe` |
| [`quant_config.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v41/quant_config.py) | V4.1 expert dtype 与量化路径 | `9,006` / `bfc500c4989607809577cbd10512b96e9162a7359ad407f772b7f695eaa34cd9` |
| [`nvidia/vl_model.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v41/nvidia/vl_model.py) | NVIDIA 视觉 wrapper | `13,027` / `627a321c526295547c065778b610b62b16d4cacc9f73572e987b5cfc8949ea80` |
| [`nvidia/dspark.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v41/nvidia/dspark.py) | NVIDIA DSpark draft runtime | `23,356` / `30e5362e96491a2b7c99b03998de3bf359e053f2d189bbffc250084e17678326` |
| [`amd/vl_model.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v41/amd/vl_model.py) | ROCm 视觉 wrapper | `13,035` / `0e17395ab5720085ebaaec30977cf1fd37e2a9802bc1f09ca80e2e3d4a16db34` |
| [`amd/dspark.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v41/amd/dspark.py) | ROCm DSpark draft runtime | `22,731` / `a109e581711a74a7c5597b3f5a07d81ed05aac0ed2619dfa3f859050efc0b9d9` |
| [vLLM `v0.29.0` stable `registry.py`](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/models/registry.py) | stable 对照 | `63,102` / `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7` |

`main` registry 明确登记 `DeepseekV41ForCausalLM -> vllm.models.deepseek_v41` 和 `DSparkV41DraftModel -> vllm.models.deepseek_v41`。正确的实现路径是 `vllm/models/deepseek_v41/` 包；先前猜测的 `vllm/model_executor/models/deepseek_v41.py` 404 不能解释成“vLLM 没有实现”。相同的 v0.29.0 stable registry 快照只有通用 V4/DSpark 入口，没有上述两个 V4.1 专用类名，因此当前证据只能写成“upstream main 已出现专用入口，stable 0.29.0 专用登记尚未证明”。

### 代码证据对应的技术点

- **Expert dtype 分支**：`quant_config.py` 把 `expert_dtype` 限定为 `fp4`/`fp8`。FP4 expert 走 MXFP4，并使用 `ue8m0`/`e8m0fnu` 风格的 FP8 linear scale；FP8 expert 走 block-FP8 与 float32 scale。V4.1 的量化配置通过 `deepseek_v4_fp8` 路径接入，不能只看一个通用 `fp8` 字符串就假设两种 checkpoint 的权重布局相同。
- **Vision wrapper**：`vl_model.py` 将 ViT 与 aligner 产生的 image embedding 注入 `inputs_embeds`，同时保留原始 `input_ids`，让 MoE router 能依据 image token 应用 `bias_vl`。代码还暴露 encoder CUDA graph 与 ViT data-parallel 的 serving 接口。视觉权重映射显式把 `mtp.*` 标为跳过，当前 vision variant 不支持 MTP/DSpark draft heads；这比“V4.1 有视觉输入且有 DSpark”更精确，因为两条路径在这个 wrapper 中并未合并为一个可直接验收的组合。
- **DSpark draft runtime**：`dspark.py` 从 `num_nextn_predict_layers` 读取预测层数，按 `dspark_target_layer_ids` 绑定目标层，并分配 `[max_num_batched_tokens, index_topk]` 的 Top-K buffer。V4.1 当前配置为 3 个预测层；draft 权重从目标 checkpoint 的 `mtp.{0,1,2}.*` 前缀加载，embedding 与 lm head 与 target 共享。代码还实现 Markov head、confidence head 和逐位置 `sigmoid` confidence、context KV 预计算、SWA cache 插入，以及 FP4/FP8 scale 的分支处理。
- **恢复/验证边界**：这些文件说明 vLLM upstream 已经为 V4.1 的视觉、量化和 draft runtime 建立了实现入口；它们没有在本轮证明完整权重加载、目标 GPU kernel 正确性、speculative acceptance length、EPD 调度、FP4 质量、p99 或生产 SLO。尤其是 `main` registry、Python/CUDA 扩展入口与 stable wheel、可运行镜像是三种不同证据等级。

### 本轮结论

V4.1-Flash 现在可标记为“官方内容 + HF reference implementation + vLLM upstream main runtime evidence + recipe protocol evidence（AA 单榜）”。仍不升级为 stable serving 闭环，也不把 vLLM `main` 的类注册写成已经下载权重或完成 GPU/线上验收。下一步继续补具体 revision、stable release、硬件和 harness 绑定的加载、召回、量化误差、DSpark acceptance 与 EPD profiling；若外部条件不允许，则保持待核验状态并选择两榜已有 canonical 条目推进。

## 2026-09-23：vLLM `v0.30.0` stable release surface

2026-09-23 重新核验了 vLLM 正式 release，而不是只看 mutable `main`。vLLM `v0.30.0` 于 `2026-09-22T05:20:54Z` 发布，tag commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。固定的 [release API 响应](https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.30.0) 为 `65,334` bytes / SHA-256 `bc5d0dee9296de133c54209afab0ae4eb9d2c7c3f331261f1dfdd4d0ab23a48`；[tag registry](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/models/registry.py) 为 `64,420` bytes / SHA-256 `a08a98aaae52ced32226aa647f58d682ac600b9572651a9b377b97846bc99212`。registry 已把 `DeepseekV41ForCausalLM` 和 `DSparkV41DraftModel` 登记到 `vllm.models.deepseek_v41`，所以 V4.1 现在有明确的 stable release/source entry；这比 v0.29.0 的“专用类名未出现”更强，但仍只证明 release surface。

v0.30.0 tag 的 V4.1 包文件也已固定：`__init__.py` 为 `625` bytes / `6b928f07c6f67f4fd1b599a143cd15a1309a177c877f7f08b29d5fa30db659fe`；`quant_config.py` 为 `9,006` bytes / `bfc500c4989607809577cbd10512b96e9162a7359ad407f772b7f695eaa34cd9`；NVIDIA `vl_model.py` 为 `13,728` bytes / `a5a3f477225990946093092d4781db181b59b52102aff2e0e345e631973817f1`，`dspark.py` 为 `23,059` bytes / `4d9c2bfa4c123aa5b95b637f24dc3d04748227455857bdc1376b27cda8af5954`；ROCm `vl_model.py` 为 `13,736` bytes / `6f3fcc8a5896432ef51f809348097e92c7782ab226adb5ecb32cdc599ce05af4`，`dspark.py` 为 `22,731` bytes / `a109e581711a74a7c5597b3f5a07d81ed05aac0ed2619dfa3f859050efc0b9d9`。因此 stable source 已同时呈现 NVIDIA/ROCm 平台分支，但不等于两种平台都已被本机或生产环境验收。

release notes 还把一些实现变化显式列为 v0.30.0 内容：DeepSeek V4.1-Flash 接入；SM100 上 FlashMLA V4.1 record 的 MXFP8 whole-KV；DeepGEMM Mega-mHC；mHC post block folded into delayed pre projection；Triton-fused input metadata preparation；CPU-offloaded Engram 的 async prefetch 与 Engram DP sharding；DSpark draft states 在 sequence-parallel all-gather 前折叠；DSpark 不继承未初始化的 EPLB state；V4.1 strict tool parameters 的 XGrammar 约束；Responses text parts；Vision-Exp 专用 image sentinel padding。这些是 vLLM release/runtime 的实现证据，不能改写成 DeepSeek 独立论文 benchmark、完整模型卡结论或本机吞吐。

[PyPI `vllm/0.30.0` metadata](https://pypi.org/pypi/vllm/0.30.0/json) 快照为 `13,218` bytes / SHA-256 `43020551808911e4cabfca5ea71951766c25101b3817a10d306d88fe42d860b8`；x86_64 wheel 为 `314,883,777` bytes / `ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4`，aarch64 wheel 为 `309,984,160` bytes / `eb3e11bab695d085098579a6eda2d602419adec3826ebfbcce9a3ffa543eb62e`，sdist 为 `42,432,229` bytes / `5f8f4e890c042ffa1c3e103f81c35e2d96f60a0a175ac43a4adbae7006bef62b`。wheel/SDist 元数据证明可分发的 release artifact 存在，不证明当前环境已安装、完整权重已加载或目标 GPU/ROCm/NPU 路径可运行。

GitHub release API 还列出 CUDA 12.9 的平台 artifact：`vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` 为 `545,459,905` bytes / `e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc`；aarch64 wheel 为 `519,981,036` bytes / `fdb57ab5fa1c3ac4c94a6eff52579df32cc9aab5da9863880d03e7167e088e77`。这说明 release API 的平台 artifact 与 PyPI 默认 wheel 是不同的分发对象，部署 manifest 不能只记录 `vllm==0.30.0`，还要记录 CUDA/ROCm、架构、Python ABI 和 wheel 名称。

release body 还把几个 vLLM-wide serving 技术列为 v0.30.0 的周边变化，适合放进 V4.1 的系统面试上下文，但不能写成 DeepSeek 专属结构：

- **Fast Start**：持久化的 per-GPU weight-cache daemon 保存 post-quantized、TP-sharded weights，重启时通过 CUDA IPC 和 `--load-format ipc_cache` 复用，减少从磁盘重新加载的冷启动路径；这需要额外验证 cache revision、GPU 拓扑、权限和失效协议。
- **HiSparse**：在 GPU 压力下把 sparse-MLA decode 的 KV page 溢出到 pinned host memory，并用每请求 GPU hot buffer 服务 top-k miss；`HiSparseConnector`、Prometheus counters 和 TP-shared host cache 是 serving 组件，不等于 V4.1 的 CSA2 candidate recall。
- **Model Runner V2 与 adaptive verification**：release body 同时列出 dual-batch overlap、PP 下的 MTP/EAGLE3/DFlash/DSpark，以及通过 online acceptance estimator 做自适应验证。这些可以解释 DSpark 的运行时演进，但仍不能替代 V4.1 的 draft/target trace、接受长度、rollback 和目标硬件测量。

目前 V4.1 的 serving 证据阶梯应写成：固定 HF reference -> vLLM main -> vLLM v0.30.0 stable release/source -> SGLang main -> stable release/wheel 安装 -> 完整权重与目标硬件验收。`v0.29.0` 仍作为历史负证据保存，不能用它覆盖 v0.30.0 的新 release surface，也不能反过来把 release surface 当作 acceptance。完整权重、真实 FP4 质量、candidate/index Top-K recall、DSpark acceptance/rollback、EPD、tool acceptance、独立 benchmark 和生产 SLO 继续为 `unverified`。

## 2026-09-22：SGLang main/stable 的 V4.1 runtime 边界

本轮继续沿 Artificial Analysis 的 `deepseek-v4-1-flash` 锚点补 SGLang serving 证据，没有从 SGLang 仓库另发现模型。通过 `10.237.126.170:1234` 的 GitHub Contents API 取得 SGLang `main` 文件并解码；GitHub raw 端点曾超时，不能把 raw 线路失败解释成源码不存在。固定快照如下：

| artifact | Git blob | bytes / SHA-256 |
|---|---|---|
| [`deepseek_v4.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4.py) | `a3b8b610235e84353e2e0b5c4680441295bffedd` | `241,421` / `4164c354f38e7b35e6bf5445b4d506e610b9f956f3e85f1431946ff6b0471ded` |
| [`deepseek_v4_dspark.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4_dspark.py) | `baebc2de6fcc0188cb2eddb90bd7ef15c99a8acf` | `47,640` / `61dc79f075c9e1e5a68a466de5eb95a85ff1fa2e5a9f4d66c2fa69e956fd7b61` |
| [`deepseek_v41_vit.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v41_vit.py) | `d5777a4f007eb2a12cace0d514de5028529d516d` | `5,126` / `29f4d98802b28ac443699c9a899e3b66406d54cc33a757017d506549a45c94eb` |
| [`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4_nextn.py) | `6694abb603a8cda9e5a2b12d1ce29d731ce90a3a` | `9,459` / `a3ca101dfec7ba4aa1577066e26c58dd0daa8f26b05016ac742cc543ddc5d90b` |

`deepseek_v4.py` 在 `config.model_type == "deepseek_v41"` 且 `vision_n_layers > 0` 时实例化 V4.1 ViT、Aligner 和图像边界 embedding。V4.1 vision 路径公开支持 TP/EP/DP，但代码明确限制 CP、PP 和 MoE A2A；图像特征经过 patch embedding、full bidirectional attention、2D RoPE、ViT blocks 和 Aligner 后写入多模态 embedding/span。`deepseek_v41_vit.py` 还暴露 attention data parallel 路径，说明这是 serving/runtime 中独立的视觉子图，而不是仅靠 prompt 拼接实现的图片输入。

同一份 main 源码还给出几个可用于面试的实现细节：

- `_dequant_fp8` 的注释区分 V4 的 `128x128` block size 与 V4.1 的 `32x32` block size；scale 可以是 `fp8_e8m0fnu` 或 `float32`。这只能说明当前 SGLang 代码的解量化分支，不能直接推出 checkpoint 已在本机加载成功。
- 主路径包含 MXFP8/FP8 prefill autotune、FlashInfer backend、unified KV、DSV4 sparse indexer 和 cache 写入；这些是 upstream code surface，不是目标 GPU 的 profiling 结果。
- `deepseek_v4_dspark.py` 读取 target layer ids 或 `num_nextn_predict_layers`，构造 Markov/confidence head、`mtp.*` draft 权重映射，并共享 target embedding/lm head。它的 `_dspark_stage_config` 在存在 vision 时将 `vision_n_layers` 设为 `0`，所以 draft stage 不实例化 vision tower；V4.1 draft hidden states 还走 collapsed forward，最后阶段处理 mHC head。
- confidence head 缺失权重会直接报错。这是 artifact completeness gate，不等于 speculative acceptance、rollback 或吞吐已经验收。

main 中可定位到的相关提交包括 `a6cf05817f11d22023fd951a76255ef50fb09f49`（2026-09-18，`dsv4.1: remaining model and runtime integration (#38798)`）和 `7fac84b6391b56b4469e0c29fd1ae0a3c0d871c6`（2026-09-19，`[DSV4.1] Reduce mHC, metadata and small-batch router overhead (#39704)`）。它们支持“main 正在补齐 V4.1 runtime integration”的时间线判断，但不改变 release 证据等级。

### stable 对照

SGLang 最新 release `v0.5.20` 的 tag commit 是 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，发布时间为 `2026-09-18T22:41:33Z`。固定 tag 中 [`deepseek_v4.py`](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/deepseek_v4.py) 为 `170,193` bytes / SHA-256 `252f6176338bf7849c9297de930745f46b761d22ff4161073a1031340f60caa4`，Git blob `af76c73f7225d14379906a341b2f51adb4447dcc`；[`deepseek_v4_dspark.py`](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/deepseek_v4_dspark.py) 为 `40,396` bytes / SHA-256 `59ac079e28d4d0a90b82c1ca641ba89f2851d081ce4fddf2bec2e003dda9fd04`，Git blob `4994a92547b6bec244b1f6e20fa276cd80e79640`。tag 中 `deepseek_v41_vit.py` HTTP 404，两个文件中 `deepseek_v41` 和 `dsv41` 出现次数均为 0，未形成 V4.1 专用 vision/runtime 证据。

因此当前准确结论是：SGLang `main` 已有 V4.1 vision、DSV4.1 sparse/indexer、FP8/unified KV 和 DSpark integration；`v0.5.20` stable 除通用 V4/DSpark 文件外，还包含 V4.1 专属 FlashMLA fork/rebase pin（PR #39171），但这仍不能证明 V4.1 专用 vision/runtime 的完整实现或验收已经进入 stable。SGLang main 与 stable tag 的 compare 为 `diverged`，main ahead 403、stable ahead 5，merge base `d58342deab27e6f8bb507d9db2a0f87c6f42cee3`；不能用当前 main 文件替代已发布包。

本轮将 V4.1 状态更新为“官方内容 + HF reference implementation + vLLM `v0.30.0` stable release/source evidence + SGLang `v0.5.20` stable 的 FlashMLA dependency-pin evidence + SGLang upstream main runtime evidence + recipe protocol evidence（AA 单榜）”。vLLM `main` 与 v0.29.0 仍作为时间线对照保留。仍未完成完整权重加载、GPU/ROCm/NPU 运行、视觉 token 的 draft/target verify、真实 FP4/FP8 质量、candidate/index recall、DSpark acceptance/rollback、EPD 调度、目标硬件 profiling、tool acceptance 和生产 SLO。

## 2026-09-24：SGLang latest stable 与 V4.1 FlashMLA pin

经 `10.24.27.134:7890` 读取 GitHub 官方 `/repos/sgl-project/sglang/releases/latest`，截至本次检查仍为 `v0.5.20`（发布于 `2026-09-18T22:41:33Z`），API 响应 `87,872` bytes / SHA-256 `c75daa307a5ead07993a00b9d8367579a9146dfc63b12ccfe31ae29cd4154bdc`。

release 中的 [PR #39171](https://github.com/sgl-project/sglang/pull/39171) 名为“Bump FlashMLA to the fork's rebase head (v4.1 kernels)”，于 `2026-09-13T22:43:20Z` 合并。PR API 响应 `31,159` bytes / SHA-256 `c150a9f705fbb5f730ddcaecf3d157f2157795dec723a2caf1125896109ec452`；正文说明它是 #38942 的 main-branch rebase，未提供独立 benchmark/accuracy 数据。因此这可证明稳定 release 包含 V4.1 相关 FlashMLA kernel dependency 更新；不能升级成 V4.1 vision wrapper、完整模型 forward、全部 sparse/cache runtime 或生产 acceptance 已验证。

## 2026-09-23：DeepSeek 官方 API capability contract

本轮通过 `10.237.126.170:1234` 重新取得 DeepSeek 官方 API 文档。下面的内容是 API/provider contract，不能反推 V4.1-Flash 的参数、attention 结构、训练算法或线上权重快照。

| 官方页面 | 用途 | 快照 |
|---|---|---|
| [Pricing](https://api-docs.deepseek.com/quick_start/pricing) | 价格、模型版本与计费字段 | `23,149` bytes / `2fecee48bf6ad791bce38d1d5504d8ad5c8b0fd4da93e6dc198ae88ff1a4506a` |
| [Rate Limit](https://api-docs.deepseek.com/quick_start/rate_limit) | 并发、速率和请求限制 | `35,133` bytes / `1190b37c138b132b45ed3fa6e19861e7fa40be95a9a9c5a36104cc88ad26413a` |
| [Error Codes](https://api-docs.deepseek.com/quick_start/error_codes) | HTTP/API 错误语义 | `20,387` bytes / `0df2698a3c67c567476e476c75c74f69313b9025ded16eeb324514c52ae5094a` |
| [Vision](https://api-docs.deepseek.com/guides/vision) | 图像输入与媒体预算 | `78,174` bytes / `a805d8a40388ee238c626b83419c2cf786ac3002187c072699710132fc77dac7` |
| [Files](https://api-docs.deepseek.com/guides/files) | `file_id` 生命周期与配额 | `61,828` bytes / `1020efca2be22faf0ec88f04aed7ee701b290c0d5ff0a465f7ec617314240345` |
| [Responses API](https://api-docs.deepseek.com/guides/responses_api) | typed items、SSE 和状态终结 | `56,250` bytes / `1719ac1b05e29579acd0cbc5ba0bcdb629cf7722eb551b5cd6d6d62c271e3ca2` |
| [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls) | function tool 与中途工具回灌 | `70,636` bytes / `5ee72ac00e5594bffac058cfef8872beb121b97e122f914c114a618f6a2b4027` |

### 可观察的服务合同

- 当前价格/模型文档把 `deepseek-flash` 映射为 V4.1-Flash，记录为 1M context、384K maximum output 和最高 2500 concurrency。`requested_model`、文档 alias 与响应中的 `model` 必须分开存储，旧 alias 的兼容路由不能写成新的 checkpoint。
- Vision guide 的当前限制包括：外部 URL 最长 8192 字符、请求超时 60 秒、URL 单图 32 MiB、`file_id` 单图 64 MiB、请求体 48 MiB、最多 600 张图；单图约 1024 image tokens 是当前 guide 的预算提示。历史实验公告中的 384 image tokens 仍保留为旧 alias/旧文档语境，不能混成一个统一数字。
- Files API 支持 `purpose=user_data`，单文件上限 64 MiB，保存期可为约 1 小时至 30 天或永久，用户配额为 25 GiB、最多 10,000 个文件。资源过期、删除和权限必须进入宿主的 artifact manifest；`file_id` 存在不代表模型已经使用了完整图像证据。
- Responses 是 stateless。SSE 使用 semantic event 类型和递增 `sequence_number`，以 `response.completed`、`response.incomplete` 或 `response.failed` 等终态结束，不发送 `[DONE]`。`previous_response_id`、`conversation`、`background`、context management 等不支持项不能被客户端默认为持久会话；部分不支持字段可能被静默忽略。
- `function_call_output` 与 `custom_tool_call_output` 可以回灌文本或图像；响应解析器必须按 item、index、call id 和 turn lineage 重放，而不是按到达顺序拼接字符串。客户端仍负责权限、执行、重试、幂等和 verifier。
- Tool Calls 文档支持在 Responses/兼容 Anthropic 的回路中插入客户端工具调用；Chat Completions 不提供同样的中途插入协议。因而“模型会生成 tool call”与“某 endpoint 能完成多轮 tool loop”是两个 capability probe。
- `/beta` endpoint 的 `strict=true` 可启用 JSON Schema 约束；这属于 API 端的 schema enforcement。固定 `deepseek-recipe` adapter 的 strict enforcement 尚未证明，协议转换器能生成字段不等于服务端或宿主已验证 JSON、权限与业务结果。

### 面试与验收边界

应把 V4.1-Flash 的闭环拆成四本账：排行榜 canonical identity、官方 API contract、reference/runtime source、完整权重与目标硬件 acceptance。API 文档可以支持 alias、媒体预算、SSE 状态机、文件生命周期、schema 和工具回灌问题；它不能替代 FP4/FP8 误差、candidate/index Top-K recall、DSpark acceptance/rollback、EPD 调度、p99、独立 benchmark 或生产 SLO。后续实验应至少记录 `requested_model`、`served_model`、API snapshot、tool/schema hash、权限决定、executor receipt、verifier receipt 和最终成本。

## 2026-09-23：API contract toy audit

新增零依赖脚本 [`deepseek_v41_api_contract_audit.py`](code/deepseek_v41_api_contract_audit.py)，不访问网络、不调用付费 API、不加载 V4.1 权重，只把官方 contract 中可复现的协议门禁缩成教学实验：

- `SemanticSSEParser` 按 semantic event、JSON data、递增 `sequence_number` 和 `response.completed/incomplete/failed` 维护状态；把输入按 5 字符切块，验证事件边界不能依赖网络 chunk，乱序序号会被拒绝，终态之后的事件也会被拒绝。
- `StrictObjectSchema` 拒绝缺失字段、错误类型和额外字段；这是 JSON Schema 的最小 toy，不冒充 DeepSeek `/beta strict=true` 的完整实现。
- `PermissionPolicy` 在 executor 之前检查工具名与路径前缀；权限拒绝不会进入执行或重试。
- `ToolHarness` 只对执行前的合成 transient failure 做有限重试；成功 receipt 按 idempotency key 保存，重复 call 返回 duplicate receipt 而不重复副作用。独立 verifier 再比较 receipt 结果，故 `executed=true` 不自动等于 `verified=true`。
- `tool_output_item` 同时演示文本与 `input_image/file_id` 结构化回灌，强调 tool output 是 observation，不是权限授予或业务成功。

本次脚本运行结果：SSE 3 个事件、文本 `contract audit`、终态 `response.completed`；首次工具调用因一次预执行 transient failure 经过 2 次尝试后完成，重复调用没有增加副作用；额外字段触发 `SchemaError`，越权路径触发 `PermissionDenied`；`network_called=false`。这些结果只证明 toy 的状态机和审计逻辑，不证明真实 endpoint、alias 路由、模型质量、strict enforcement、生产 executor 或 verifier 已验收。

## 2026-09-28：7890 复验与 vLLM stable/预发布边界

当前工作区显式经 `10.24.27.134:7890` 请求百度、Artificial Analysis `/zh`、DataCurve DeepSWE、DeepSeek V4.1 公告及 GitHub release 页面均取得 HTTP 200。榜单本轮只确认可达，没有重新解析 canonical 集合，不据此声称模型清单无变化。DeepSeek 公告最终 URL 为 [`news260910/`](https://api-docs.deepseek.com/news/news260910/)，`24,438` bytes / SHA-256 `f18dc22d37393381b31c9069996138f45aa7b02b08442d43af7c6c57f587bdce`。

vLLM [`releases/latest`](https://github.com/vllm-project/vllm/releases/latest) 最终重定向到 `v0.30.0` 稳定版页面，`615,462` bytes / SHA-256 `ff2ddd43e2e969a5ccd9b141c43c9aa0dc36af24402cca781eb2e9760f4f8aa2`。官方 [release Atom feed](https://github.com/vllm-project/vllm/releases.atom) 为 `734,203` bytes / `30131b0cf21894d9e8bf4cbd4eb0f51d8e47b364e3849db0ee1c5f8724810de5`；最新 feed entry 是 `v0.30.1rc0`，时间 `2026-09-23T08:06:15Z`。其固定页面为 `224,871` bytes / `8034c62c3dbc34f8fb7897e8c2c5ed763ca8538884cc0965714a71b5629fb015`，可见标题为 `[ROCm][CI] Add MI355 dense NVFP4 and MoRI kernel mirrors (#58281)`；这是 RC/ROCm-CI 条目，不是新的稳定版，也没有显示 V4.1 专属 release-note 增量。`api.github.com/repos/vllm-project/vllm/releases/latest` 在同一代理下返回 HTTP 403，因此以可用的 GitHub release 页面和 Atom feed 交叉确认，不把 API 403 当成 release 不存在。

本日另有已保存的 SGLang/PyPI 快照：GitHub latest release 仍为 `v0.5.20`（`2026-09-18T22:41:33Z`；`433,353` bytes / `11c41065c3d9dff6d381437e33f1c8be3600ed9066c0fa5d5c2e51217fd49cd1`），PyPI `info.version` 仍为 `0.5.20`（`350,431` bytes / `267bc13e757f4ca70d0cf0f5de5c463ee819b34097475c7d7e55bf02fab7a05c`）。因此 V4.1 serving 证据仍是 vLLM `v0.30.0` stable source、SGLang `v0.5.20` stable/source 与各自上游更新分开记录；没有安装 wheel、加载权重或做硬件验收，完整模型与生产验收待核验项不变。

## 2026-09-29：SGLang v0.5.20 的 DeepSeek-V4 AMD/HIP cache 与 DSpark 增量

### 榜单复验与 release 边界

本轮仅沿已有 Artificial Analysis 锚点 `deepseek-v4-1-flash` 继续查 runtime，没有从代码仓库发现模型。经 `10.24.27.134:7890` 取得 AA `/zh`（1,747,860 bytes / SHA-256 `0933d776618e8530dccb05f28c0f3ce2398a3184552c8c988df9a60e6da5d0c5`）、[AA V4.1-Flash 详情](https://artificialanalysis.ai/models/deepseek-v4-1-flash)（3,969,971 bytes / `630f1c9df1abd6ce900d6a7016daf1038d5482c349fd5f4f2a419973ef790a91`）和 [DataCurve](https://deepswe.datacurve.ai/)（268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`）。详情首请求遇 TLS EOF，重试 HTTP 200；AA 路由集合与同日快照一致。DataCurve 有 `mini_swe_agent_deepseek_v4_flash_max`，没有精确 `mini_swe_agent_deepseek_v4_1_flash_*`；V4 Flash 行不迁移成 V4.1 成绩。

7890 获得 [vLLM release Atom](https://github.com/vllm-project/vllm/releases.atom)，734,377 bytes / SHA-256 `9933c44ff0413dce0b7565f5df9f4046eebea4fb121b95efc19ca94fd245933e`；最新项为 2026-09-29 `v0.31.0rc1`，标题是跳过 CUDA 12 镜像中的 snapshot runtime（CI/build），不是新的稳定版。当前 stable 仍 `v0.30.0`。 [SGLang release Atom](https://github.com/sgl-project/sglang/releases.atom) 为 1,058,795 bytes / `9c9c5e9bcc5fc55fea6be1d8e9c77fd84d330522638083a80e74f2134ddd5eb2`；latest stable `v0.5.20` 发布于 2026-09-18。

### 三项可迁移的 serving 知识

这些是 v0.5.20 release body 与固定 PR patch 的 V4 家族 ROCm/HIP serving 证据，不是 V4.1 专属模型结构，也不是本机/独立性能测试。

1. **DSpark graph replay 的指针语义与 host bubble（SGLang [PR #39116](https://github.com/sgl-project/sglang/pull/39116)）**：release 标题称“修复 DSV4 DSpark accept length 并降低 host bubble”。patch 解释了一个具体状态 bug：CUDA/HIP graph 捕获的 store kernel 按记录时的 `swa_loc` 指针读写；eager target-verify 每步新建 tensor 后，Python 层重新绑定引用并不会更新图保留的 buffer，因此 replay 可能持续写旧 SWA ring slots。修复把 `swa_loc` 改为 copy-field，在固定地址缓冲区内覆盖新值。另将统一长度 DSpark verify 的 metadata materialization 移入图内，并用 `exact_num_tokens` 区分真实 token 数与 padded tier：只有长度精确时才给 `repeat_interleave(output_size=...)`，避免用 D2H `.item()` 同步换取速度却错误声明输出长度。patch 附有原位更新与 graph-routing 测试；本轮未运行这些 AMD 测试。

2. **统一 KV 中的 request-owned SWA ring（SGLang [PR #38192](https://github.com/sgl-project/sglang/pull/38192)）**：15-commit patch 将压缩比为 4 的 SWA recurrent/compress state 放到每请求 ring（按 request slot 与 position 模运算寻址），并让 unified-KV pool sizing、scheduler capacity check 和状态回收共同识别 ring 布局。关键不是“多塞一些普通 KV”，而是避免把每请求、循环覆盖的 SWA state 错算成统一 radix tree 中可跨分支复用的 full-attention token slots。release note 将路径标为 AMD/DSV4、fully gated，并报告可容纳的 full-attention KV tokens `+83.6%`；该数字是指定 release workload 的发布方结果，feed 未给出完整硬件/负载表，不能推广成一般吞吐或 V4.1 模型分数。

3. **FP4 indexer 的 schedule 准备融合（SGLang [PR #37764](https://github.com/sgl-project/sglang/pull/37764)）**：HIP patch 将原先约 27 个小 Torch 调度操作收敛为融合的 prefill-schedule prep kernel，再接 AITER CTA-info kernel；page-table padding 与 schedule buffers 显式保活。由于 AITER fallback 会释放其 scratch，该路径要求 schedule build 在 CUDA-graph capture 外执行，超过 `MAX_FUSED_ROWS` 时回退到 AITER 旧 preamble。release note 报告 concurrency 4 的 output throughput `+15.3%`；这是 AMD FP4 indexer 指定实现/负载结果，不代表所有 GPU、并发或端到端任务速度。

### 不把同一 release feed 的模型实现混归

SGLang v0.5.20 还列出 DSA cooperative exact top-k PR [#37591](https://github.com/sgl-project/sglang/pull/37591)，但该 patch 的代码注释和测试样例明确指向 DeepSeek V3.2 / GLM-5.2 的 index_topk workload（有例子来自 GLM-5.2、134,849-context decode），不是 V4.1 的专属证据，因此不并入本模型技术结论。类似地，V4.1 官方模型结构、V4-family runtime patch、AMD HIP kernel 和发布方测量必须分层写入。

本轮更新把 V4.1 serving 证据推进到 SGLang `v0.5.20` 已发布的 V4-family ROCm/HIP cache/DSpark 路径；它不关闭完整权重加载、V4.1-specific vision/runtime、FP4/FP8 quality、index recall、实际 DSpark acceptance/rollback、目标硬件 profile、工具验收或生产 SLO。

## 2026-09-29：SGLang main PR #39313 混合精度 MegaMoE shared-expert fusion

本节继续沿既有 Artificial Analysis canonical `deepseek-v4-1-flash` 追踪 V4-family runtime，没有从 SGLang 代码中发现模型。fresh GitHub commits API（156,291 bytes / SHA-256 `f7529dd2c868848510b3bdbb1e003d91d5527da1b70a9c7e6b973cdea72bba44`）显示 `deepseek_v4.py` 的 2026-09-29 最新相关提交 `0e586fd12d63f06306ec20beb637bfeb331f1088`，标题为 `fuse shared experts with routed experts in MegaMoE's DeepGEMM (#39313)`。fresh PR API（45,675 bytes / `fbe33b6da871321cf7445d948086d7ce9456a3e211865ef7e4ddfe953db00970`）确认 PR 于 `2026-09-29T09:25:11Z` 合并进 `main`；merge commit 为上述 SHA，head 为 `1aff3627e28222654bac599e20be27d90ef2bb86`。固定 `.patch` 为 35,120 bytes / `2ceee8fb5122d9e571a1608416e5699d824ee703b4ad641e9ab50f8fc3192e6b`。

### 机制：让 mixed-precision shared expert 进入 routed MegaMoE

PR 描述的 DeepSeek-V4 checkpoint 将 shared expert 保持 block-FP8，而 routed experts 使用 MXFP4/QAT；Flash 路径为 1 shared + 256 routed、每 token 选 6 个 routed experts，Pro 路径为 1 shared + 384 routed。旧的 shared/routed fusion gate 要求两边量化精度兼容，因此这组异构布局无法直接走原融合路径。PR 将 FP8 shared L1/L2 权重作为独立输入交给 DeepGEMM `fp8_fp4_mega_moe`，转换 shared scale 到 kernel 所需布局，并在 pre-dispatch 依据实际 `BLOCK_M` 写入 shared activation scales；routed scaling factor 则可前移并折入 top-k weights。

该路径由门禁控制：MegaMoE A2A backend、SM100-supported device、已构建的 MegaMoE expert weights、`fp8xfp4` MMA、FP8 shared expert、`[128, 128]` shared weight block size，以及默认开启的 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_FUSE_SHARED_EXPERTS`。原始 shared weights 仍保留给不满足 token-cap/kernel 条件时的普通 fallback。代码目标是把 shared expert 原先每层的 6 个独立 kernel 收入 routed MegaMoE；这会省 launch，但不自动意味着端到端更快。

### overlap 与评测账本

PR 作者报告，若融合后直接从 `forward_mega_moe` 返回，就会绕过现有 alternate-stream fork；未融合时 shared expert 的第二个 GEMM 在所测 184/184 次/rank 与 routed MegaMoE 重叠，中位覆盖其耗时的 94.1%–95.6%。最终实现因此在 CUDA-graph capture 时把完整 router/top-k/pre-dispatch/MegaMoE 区域 fork 到 alternate stream，再 join。PR 给出的 4×B300（SM103）、`sgl-deep-gemm 0.1.7`、输入 1,024/output 256 tokens 的 overall throughput（tokens/s）如下；DSpark 使用 `SGLANG_SIMULATE_ACC_LEN=6`，数字均是 PR 发布方结果：

| 路径 | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|
| DSpark main | 3,052.04 | 19,255.64 | 53,236.74 | 76,347.80 |
| DSpark fusion + fork | 3,125.89 | 19,624.04 | 54,022.42 | 81,739.39 |
| Regular main | 737.25 | 5,147.25 | 16,123.12 | 27,562.54 |
| Regular fusion + fork | 748.13 | 5,252.39 | 16,245.61 | 27,506.76 |

结果依 batch 与 prefill/decode 组成变化；例如 regular batch 64 的 overall throughput 略低于 main。PR 同时报出 greedy accuracy `0.899 -> 0.897`（198 题）、sampled pass@1 `0.900 -> 0.910`（792 samples），并有逐请求 answer flips。不能把小样本的点估计写成统计等价、质量提升或一般性能保证；完整 prefill/decode 表、paired comparison 和任务设置以 PR body 为准。

证据边界尤其重要：PR 的准确性/速度实验使用 `deepseek-ai/DeepSeek-V4-Flash-0731`，不是 `DeepSeek-V4.1-Flash`；PR prose 里的 Flash/Pro expert counts 也不能覆盖 V4.1 固定 config。本段是 **V4-family SGLang main serving 实现上下文**，不是 V4.1 专属结构、V4.1 分数或新架构论文。SGLang 当前 release Atom（1,058,795 bytes / `9c9c5e9bcc5fc55fea6be1d8e9c77fd84d330522638083a80e74f2134ddd5eb2`）仍指向 stable `v0.5.20`（2026-09-18），故 9 月 29 日合入的 #39313 不属于该 stable tag。

CI 也不应被简化成“全绿”：Checks API 快照为 129,420 bytes / `b045fed811e051416ed84de214b4fbc2e80c28173c1b33115f620225ebe818b8`，`total_count=171`，默认页返回 30 项，其中 15 success、9 skipped、6 failure；PR body 的最新 Base、Extra、AMD finish 项显示失败标记。未逐页审计全部 171 条检查，因此不声称完整 CI 汇总。没有在本机运行 SGLang 测试、安装依赖或执行 GPU benchmark。

面试要点：kernel fusion 的优化对象不只是 kernel launch 数，还包括它是否破坏原有 stream overlap；正确的优化单位可能是“融合后的整段调度区域”，而不是单个 fused kernel。部署 manifest 应显式绑定权重布局、scale layout、SM capability、MegaMoE backend、token cap、graph capture 和 fallback 条件。当前仍保持 **DeepSeek V4.1 AA 单榜内容专题 + family-level main source 增量**，不提升为 stable、目标硬件或生产验收闭环。
