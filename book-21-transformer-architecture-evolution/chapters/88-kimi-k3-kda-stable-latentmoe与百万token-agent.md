# 第 88 章 Kimi K3：KDA、AttnRes、Stable LatentMoE 与百万 token Agent

> 证据边界：Kimi K3 的具体配置来自 [MoonshotAI/Kimi-K3 官方仓库](https://github.com/MoonshotAI/Kimi-K3) 的 README、`k3_tech_report.pdf` 和固定 Hugging Face `config.json` revision。Kimi Linear、Attention Residuals 和 FlashKDA 是关联机制/实现资料。Artificial Analysis 与 DataCurve 只用于确认榜单候选和配置级评测，不提供内部架构证明。

## 88.1 为什么需要三条扩展轴

把 Transformer 做大，至少会同时遇到三类瓶颈：序列越长，token 间全连接的计算、KV cache 和状态恢复越贵；网络越深，普通残差会把所有历史层压进一个固定加法路径；模型越宽，激活更多专家会带来路由倾斜、通信和低精度溢出。K3 技术报告把这三类问题分别映射到：

| 扩展轴 | K3 设计 | 要解决的工程问题 |
|---|---|---|
| sequence length | KDA + Gated MLA | 固定状态的长序列混合与全局内容交互 |
| network depth | Block Attention Residuals | 深度方向的选择性信息累积与跨 stage 通信 |
| model width | Stable LatentMoE | 大专家池的容量、负载和低精度稳定性 |

这是一种系统设计分解，不表示三个组件能互相替代。KDA 沿 token 序列维护状态，AttnRes 沿层深度选择表示，LatentMoE 沿专家维度稀疏计算。

## 88.2 K3 的配置账本

官方 README/报告给出：2.8T total parameters、104B activated parameters、93 layers、69 个 KDA layer、24 个 Gated MLA layer、1 个 dense layer、7168 attention hidden dimension、96 attention heads、3584 latent MoE dimension、3072 routed expert hidden dimension、896 routed experts、每 token 选择 16 个 routed experts、2 个 shared experts、160K vocabulary 和 1,048,576 context。视觉侧使用 MoonViT-V2，报告/README 给出约 401M 参数。

“104B activated”是官方模型账本中的口径；它不是任意 batch 的显存占用，也不包含 KV cache、通信 buffer、workspace 和并发请求。面试时应把 total、activated、resident weight、KV 和 runtime workspace 分成五本账。

固定 Hugging Face revision `f831ab66814297da540d832a5235f8e904f29d06` 的 `config.json` 进一步确认了 `q_lora_rank=1536`、`kv_lora_rank=512`、`max_position_embeddings=1048576`、`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、69 个 KDA layer 和 24 个 full-attention layer；官方 API 文件清单包含 96 个 safetensors 分片。本地没有下载完整权重，因此这些是仓库 metadata/config 证据，不是本地加载或端到端复现证据。

## 88.3 3:1 KDA/Gated MLA

K3 的 backbone 每三个 KDA 层接一个 Gated MLA 层，末尾额外放置一个 Gated MLA，保证最终层具备全局注意力。MLA 层使用 NoPE。直觉上，KDA 用递归状态低成本保存序列中的可更新记忆，Gated MLA 周期性提供全局 token-to-token 交互，门控再决定全局路径写回多少。

K3 report 还披露 lower-bounded decay：令 `g_min=5`，将衰减参数通过 scaled sigmoid 映射，使 retention factor 不会过小。这样 diagonal/off-diagonal tile 可以统一使用 dense TensorCore 矩阵乘法，减少专门 position-pair diagonal path。它是数值参数化与 kernel 形状共同设计的例子：一个看似训练稳定性的下界，也影响 serving kernel 是否能保持规则矩阵计算。

```math
S_t = \mathrm{Decay}(\alpha_t) \odot S_{t-1} + \mathrm{Update}(k_t,v_t,q_t),
```

上式只表示 KDA 的状态更新抽象，不是 K3 report 中完整 kernel 公式。完整实现还包括逐通道衰减、delta-rule 更新、chunkwise parallel form、input-dependent full-rank output gate，以及训练时 MLA 输出的 FP32 保持。不要拿这段抽象式冒充 FlashKDA 的生产实现。

## 88.4 Block AttnRes：沿深度保留可选历史

普通残差把每层输出直接累加。AttnRes 则让当前层用 learned pseudo-query 对 embedding 和历史层表示做选择性 attention。K3 采用 Block AttnRes，把 93 层组织成 8 个 block，每 block 12 层，并把 embedding representation 作为额外的 block-level representation。这样报告中的深度状态数量是 9 个 block-level representation，而不是 93 个独立层输出。

块级设计的收益是跨 pipeline stage 的保存、传输和读取从按层增长变为按 block 增长；代价是 block 内仍然需要维护 partial sum，细粒度的层信息会被聚合。它不等于沿 token 的 self-attention，也不等于把所有层输出永久保留。

## 88.5 Stable LatentMoE 与 Quantile Balancing

K3 的 routed expert 在 latent width 中运行：先把 full-width hidden 投影到 latent width，再经过稀疏 expert，聚合后投影回 full width；shared experts 保留 full-width 路径。聚合后追加 RMSNorm，控制 routed branch 的尺度。这个设计把专家内部的矩阵乘法和激活宽度从主干宽度中解耦，但增加了投影和路由账本。

SiTU-GLU 对 gate/up 两个分支使用 scaled `tanh` soft cap，降低 activation outlier 和低精度溢出风险。Quantile Balancing 不依赖 learning-rate-like 的平衡超参，而是从 router score quantiles 得到 expert bias，以最大-score balanced assignment 为目标；报告描述通过交替 coordinate minimization 求 token/expert 方向的量化阈值。训练大规模统计时可用 per-expert histogram，只需 global histogram all-reduce，通信量不随 token 数线性增长；推理阶段使用冻结 bias 和 fixed Top-k，不运行 quantile 计算。

面试中不要只说“MoE 节省 FLOPs”。应继续回答：专家容量是否均衡、dispatch 是否需要 all-to-all、shared branch 是否始终驻留、latent projection 是否成为瓶颈、低精度下 router 排序是否改变，以及 top-k/负载倾斜如何进入 p99。

## 88.6 QAT、Muon 与长轨迹 RL

报告披露从 SFT 阶段开始使用 MXFP4 weights、MXFP8 activations 的量化感知训练，而不是训练完再单独做一次 PTQ。Per-Head Muon 把 Muon 的优化结构扩展到独立 attention head。公开材料支持这些路线存在，但不支持补写完整 optimizer 超参、训练数据比例、量化 kernel 误差消融或硬件专属吞吐。

后训练覆盖 general reasoning、agentic tasks、coding 和 multiple reasoning efforts，并使用 multi-teacher on-policy distillation 合并 domain/effort-specialized policies。长轨迹环境包括 web search、professional knowledge work、软件工程/kernel optimization、vision-in-the-loop tool use、persistent assistant workflows、web development 和 autonomous execution。轨迹可包含数百/数千次工具调用和百万级上下文 token；这说明 RL 的状态、恢复和外部副作用管理必须成为训练系统的一部分。

## 88.7 从 MoonEP 到可恢复 sandbox

K3 report 把模型结构和系统基础设施一起讨论：KDA fused kernels、KDA context parallelism、state-aware prefix caching、MoonEP 静态形状/zero-copy expert parallel、partial rollouts、external KV-cache retention、adaptive throttling 和 resumable microVM sandboxes。Serving 侧还按 cache-aware、budget-aware 的 effort scheduling 管理成本和质量。

这些名称对应的是工程方向，不等于本地已有可复现实装。评估实现时应分别核对：仓库 commit、后端支持、硬件型号、batch/sequence、缓存命中、通信量、失败恢复和 verifier。当前官方 vLLM recipe 已公开 K3 的 TP/TEP/DEP、FP8 KV、hybrid KV manager、prefix-match unit 128 和 DCP 配置；recipe 使用 K3-enabled nightly/image 说明的是特定优化部署路径，不能替代目标环境验收。

## 88.8 XTM 与 Agent 状态回放

报告描述 XTM 风格模板，使用 `[open]`、`[sep]`、`[close]`、`[end_of_msg]`。全局 option 可声明工具和 reasoning effort；one-shot option 包括 `tool_choice` 和 `response_format`。后续 `tool-declare` message 可以动态扩展工具集，assistant channel 分为 `think`、`response`、`tool`，tool call/result 使用 `tool/index` 配对。

因此 K3 的长任务恢复至少要保存：模型/revision、channel、tool index、工具 schema 版本、reasoning effort、思考状态引用、工具结果、权限决定、sandbox checkpoint 和 artifact。仅保存 visible response 会破坏重试幂等性和跨模型迁移；“保留思考历史”在这里首先是协议状态完整性问题，不是把隐藏推理任意暴露给下游。

## 88.9 评测证据边界

Artificial Analysis 的 K3 行和 DataCurve 的 `mini-swe-agent` 行是模型配置与 Agent harness 的外部观察。发布文章/报告中的 benchmark 还绑定 reasoning effort、temperature、top-p、硬件、harness 和 fallback。正确的比较键至少包括 model revision、provider、harness、hardware、effort、task set、tool version、verifier 和 sampling。

README 的“完整权重已发布”和 License 已经是官方一手证据；本轮进一步固定了 Hugging Face revision、96 个权重分片的 API 清单、config hash、FlashKDA master commit，以及 vLLM K3 recipe。FlashKDA README 要求 SM90+、CUDA 12.9+、PyTorch 2.4+，支持 recurrent state/`cu_seqlens` 和 FLA backend dispatch；H20/GB200 benchmark 仍只是仓库发布结果。vLLM `0.29.0` stable release/source 已有 K3 registry/package/model implementation，但线上 acceptance rate、独立 benchmark 和目标硬件 profiling 仍待核验，报告和 config 也不能被扩写成未公开训练 recipe。

## 88.10 面试追问

**问：为什么 K3 同时需要 KDA 和 Gated MLA？**

答：KDA 提供低成本、可递归更新的序列状态，Gated MLA 周期性提供全局内容交互；3:1 混合是在长序列成本和精确全局访问之间做结构化折中，不是简单把全注意力删掉。

**问：Block AttnRes 解决的是上下文长度问题吗？**

答：主要解决深度方向的历史表示选择、保存和通信；它不替代 token 维度的长上下文注意力，也不直接消除 KV cache。

**问：Quantile Balancing 与常见 auxiliary loss 有什么不同？**

答：报告描述它直接从 router-score quantiles 产生 bias，并用 coordinate minimization 对齐最大得分分配；推理时冻结 bias，不需要运行时 quantile。是否在所有训练阶段都优于 auxiliary loss，仍需同条件消融。

**问：104B activated 是否等于每 token 只读 104B 权重？**

答：不等于。它是官方激活参数口径；实际 resident weights、shared experts、通信 buffer、KV cache、workspace 和 batch 并发需要单独计算。

**问：K3 的工具协议为什么要保留空的 thinking channel？**

答：固定结构降低解析歧义，使 replay、tool/index 配对和跨轮状态恢复更稳定；空 channel 的存在不意味着下游获得了全部隐藏推理。

## 88.11 小练习

1. 按 total、activated、resident weight、KV cache、workspace 五本账估算一个 8 请求 batch 的内存，说明哪些数字不是模型卡直接给出的。
2. 画出 3 个 KDA + 1 个 Gated MLA 的 token 状态路径，标记递归状态、全局 KV 和输出 gate。
3. 模拟 896 experts、top-16 和冻结 bias，比较均匀路由、热点路由和 expert overflow 的通信与 p99。
4. 构造 XTM trace：`think -> tool call -> tool result -> response`，删除 tool/index 或 thinking channel 后检查 replay 为什么失败。
5. 固定 harness、硬件、effort 和 verifier，比较 K3 与另一个模型；报告中分开写榜单字段、官方 benchmark 和本地 toy 结果。

6. 对照 FlashKDA 的 H20/GB200 benchmark，固定 `T=8192`、`H`、`D`、warmup/iters/repeats，分别记录 kernel latency、基线实现和硬件；不要把局部 KDA forward speedup 写成完整 K3 serving 吞吐。
7. 画出 vLLM hybrid KV manager 的两个状态组：MLA attention cache 与 KDA recurrent state；解释为什么 prefix-match unit、DCP、prefill/decode backend 和 TP/TEP/DEP 拓扑必须绑定在同一 serving manifest 中。

## 88.13 固定权重 manifest：从模型卡到可审计分片

“官方发布了完整权重”与“我的服务已经加载了完整权重”是两个不同命题。前者可以由 README 或模型仓库声明支持，后者至少还需要固定 revision、下载清单、索引映射、加载日志和目标后端验收。K3 的完整 safetensors 约为 TB 级，本项目本轮只读取 JSON metadata 和源码，没有下载分片。

固定 revision 的 `model.safetensors.index.json` 通过 SHA-256 `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd` 固定。索引包含 `497,220` 个 tensor 映射，使用连续的 96 个分片；`config.json` 负责解释这些名字属于 93 层、69 个 KDA 层和 24 个 full-attention 层。索引还显示每个 routed expert 的 packed weight 都有对应 scale tensor，共 `247,296` 对。

这里还有一个容易造成错误的“模型大小”陷阱。index 的 `metadata.total_size=1,560,860,324,864` 是 packed MXFP4 分片口径；HF API 的 `safetensors.total=2,779,931,837,184` 是按 U8/BF16/F32 dtype 统计的参数口径。两者用于不同问题：前者用于文件分发和加载计划，后者用于参数/存储统计。它们不能相加，也不能在面试中互换。

本轮的零依赖脚本 [`kimi_k3_manifest_audit.py`](../../research/model-update-2026-09/code/kimi_k3_manifest_audit.py) 只做以下门禁：

1. 分片编号连续且总数一致；
2. index 的 layer id 与 config 的 93 层一致；
3. 1-based 的 KDA/full-attention 配置是互斥且完整的 partition；
4. 92 个 MoE layer 各自包含 0 到 895 的 expert id；
5. packed weight 与 scale tensor 成对出现；
6. 输出中明确 `full_weights_downloaded=false`。

这类审计器的价值不是替代加载，而是先阻止“revision 错配、缺分片、专家不全、量化 scale 丢失”进入加载阶段。它也把模型身份、文件完整性和运行时性能分成三道不同门禁。

## 88.14 KDA state 与 MLA cache 不是一个字段

固定 revision 的 `modeling_kimi_linear.py` 把两种状态分开维护：

| 路径 | 代码状态 | 作用 | 单 token decode |
|---|---|---|---|
| full attention / Gated MLA | `key_cache`、`value_cache` | 追加可检索的 full-attention K/V 历史 | 使用 cache update |
| KDA | `conv_states`、`recurrent_states` | 保存短卷积边界和递归矩阵状态 | 使用 `fused_recurrent_kda` |
| KDA prefill/chunk | `cu_seqlens`、chunk state | 处理批量/变长输入 | 走 `chunk_kda` |

因此 `past_length` 不能代表 K3 的全部历史状态。恢复一个长任务时，至少要绑定模型 revision、KDA state layout、MLA KV dtype、cache position、prefix-match unit、backend 和并行拓扑。只恢复 MLA cache 会丢掉 KDA 的递归连续性；只恢复 KDA state 又无法复现 full-attention 的精确历史访问。

FlashKDA 的 H20/GB200 数字仍只是固定输入、特定 backend 和仓库 baseline 下的局部 kernel benchmark。它们不能证明 hybrid cache 的恢复正确性、K3 的端到端吞吐或线上 Agent 成功率。生产验收还要加入 cache eviction、beam reorder、变长 batch、跨 revision 拒绝、tool-call parser、权限和 verifier 测试。

## 88.12 官方实现证据：FlashKDA 与 vLLM K3 recipe

K3 的公开资料已经能把“架构名词”连接到实现路径，但必须保持证据分层。官方 [FlashKDA](https://github.com/MoonshotAI/FlashKDA) master 最新可见 commit 为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`（2026-09-01）。README 把 KDA kernel 建立在 CUTLASS 上，要求 SM90+、CUDA 12.9+ 和 PyTorch 2.4+，并通过 `flash-linear-attention` 的 `chunk_kda` 自动 dispatch；`FLA_FLASH_KDA=0` 可以回退到 Triton。它支持 initial/final recurrent state、BF16/FP32 state 和 `cu_seqlens` 变长批处理，当前要求 `K=V=128`。

仓库发布的 `T=8192,H=96,D=128` forward benchmark 在 H20 上给出 `flash_kda=2.6220 ms`、`fla_chunk_kda=4.8388 ms`，在 GB200 上给出 `1.0087 ms`、`2.3271 ms`；两组分别是 1.85× 和 2.31× 的局部基线加速。它们绑定 warmup 30、iters 200、repeats 5 和特定 GPU，不证明 K3 的端到端 TTFT/TPOT、MoE 通信或百万 token 质量。

官方 [vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3) 则处理部署组合：K3-enabled CUDA 13 image、至少 8 张 GB300 或 MI355X/MI350X、Blackwell 的 FP8 KV/TOKENSPEED MLA、hybrid KV manager 的 `--prefix-match-unit 128`、decode context parallelism，以及 RDMA/NVLink 的不同 all-to-all 后端。recipe 还警告 K3 偶尔会产生 parser 不期望的 tool-call 格式，宿主需 schema validation/retry。面试时应把这条链回答成：

```text
Kimi K3 config/revision
    -> KDA recurrent state + MLA attention cache
    -> FlashKDA / MLA backend
    -> hybrid KV manager + prefix match / DCP
    -> TP/TEP/DEP/PP + RDMA/NVLink topology
    -> tool parser/schema/retry + verifier
```

这条链把模型结构、kernel、缓存、并行和 Agent harness 连起来；任何一层的局部结果都不能替代其他层的生产验收。

## 88.15 vLLM 实现入口与硬件分叉：从类名到可运行 artifact

K3 的公开实现证据现在可以向前推进一层，但仍不能跨过部署验收的证据鸿沟。vLLM stable supported-models 页面已经列出 `KimiK3ForConditionalGeneration`、`Kimi-K3` 和 `moonshotai/Kimi-K3`；stable API 页面还暴露 `KimiK3MTP`。这类页面回答的是“文档和 API 是否知道这个模型”，不是“本地 stable wheel 是否能加载这份 revision”。

main registry 进一步登记 `KimiLinearForCausalLM`、`KimiK3ForConditionalGeneration`、`K3DSparkModel` 和 `KimiK3MTPModel`。registry 是源码发现表，不能把 MTP/DSpark 类名解读为 draft/verify/rollback 或 speculative acceptance 已经在目标硬件上通过。K3 package 的 `__init__.py` 按 `current_platform` 分流 NVIDIA 和 ROCm，并让 TPU 路径不主动加载 GPU 实现；这是一种硬件隔离设计，仍需分别测试 CUDA/ROCm kernel、通信、dtype 和 cache 状态布局。

recipe 的证据层更接近部署，但其 YAML 仍是 `Pre-release`，最低版本字段为 `0.29.0`，实际要求 K3-enabled nightly/image、CUDA 13/cu130 和 NVIDIA r580+ driver。它给出了 TP/TEP/DEP/PP、DCP、Blackwell FP8 KV、TOKENSPEED MLA、prefix-match unit 128 以及 hybrid KV manager 的组合。这里的 `hybrid` 不是宣传词：full-attention 侧要恢复 `key_cache/value_cache`，KDA 侧要恢复 `conv_states/recurrent_states`；两个状态组的 dtype、位置、eviction、prefix hit 和故障恢复都要分别验证。

一个可靠的 K3 artifact manifest 应至少分出：

```yaml
docs: {stable_model_page: observed, stable_api: observed}
source: {registry: main, platform_branch: nvidia_or_rocm}
recipe: {status: pre_release, vllm_min: "0.29.0", cuda: cu130}
artifact: {revision: pinned, full_weights_loaded: false}
runtime: {mla_cache: unverified, kda_state: unverified}
acceptance: {hardware_profile: missing, tool_call: schema_retry_verifier_pending}
```

这个 manifest 的重点是“不把缺失写成成功”。stable docs/API 和 main registry/package entry 已经足够支撑“实现入口存在”的表述；它们还不足以支撑完整权重加载、stable wheel、目标 GPU profiling、hybrid cache recovery、线上 tool-call acceptance 或 MTP/DSpark 的独立验收。FlashKDA 的局部 H20/GB200 benchmark 也只能留在 kernel evidence 栏，不能填入端到端 K3 throughput 栏。

## 88.16 stable release 与优化 recipe 的边界

通过 PyPI metadata 可以确认 `vllm 0.29.0` 是已发布的 stable artifact；对应的 v0.29.0 source 还包含 K3 registry、K3 package 分支和 NVIDIA K3 model implementation。因此现在可以说“stable release 中已有 K3 实现入口”。但这句话仍然不等于“我的服务已经把 K3 跑起来”。

要把 stable artifact 变成可运行服务，还需要三类证据：第一，安装的 wheel 版本与模型 revision、tokenizer、quantization manifest 一致，并有完整权重加载日志；第二，目标 NVIDIA/ROCm 平台真正选择了对应分支，KDA kernel、MLA backend、MoE dispatch 和通信拓扑通过 profiling；第三，MLA `key_cache/value_cache` 与 KDA `conv_states/recurrent_states` 的恢复、变长 batch、故障注入和 tool-call schema/retry/幂等/verifier 通过。

recipe 继续要求 K3-enabled nightly/image、CUDA 13/cu130 和特定 driver，是因为它记录了面向特定硬件的优化组合。这与 PyPI stable release 并不矛盾：一个回答“通用发行版里是否已有代码”，另一个回答“某套优化路径怎样部署”。面试时应分别说“stable implementation entry”“optimized recipe”“target runtime acceptance”，不要用其中一个替代另外两个。

## 88.17 SGLang：KDA、LatentMoE 与 EP 通信如何落到 runtime

vLLM 资料回答了 K3 是否有 stable implementation entry；SGLang 源码则更适合追踪一次 forward 中 token、状态和通信怎样排队。本节对照了 SGLang `main` 与 `v0.5.20` 的 K3 文件：`main` 的 `kimi_k3.py` 为 171,101 bytes、Git blob `383a6f47812bccd1cb91b76814cd0730ff945dd7`，`v0.5.20` 为 168,114 bytes、Git blob `b0ede48c88264d518351a66abf623f1bcf8a730e`；两个版本的 `kimi_k3_vl.py` 都是 32,790 bytes、Git blob `c423027994645e4a839ec47bd26bcc1b2a7cb012`。tag `v0.5.20` 的 commit 是 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。

### 88.17.1 LatentMoE 的三段式数据流

SGLang `KimiK3MoE` 明确把 routed expert 的计算宽度设为 `routed_expert_hidden_size`：

```text
hidden(full width)
    -> router / grouped Top-K
    -> down projection(full -> latent)
    -> EP/A2A dispatch + expert GEMM(latent)
    -> latent reduction + RMSNorm
    -> up projection(latent -> full)
    -> shared expert branch + residual/prefix add
```

这段代码把“稀疏专家减少计算”拆成了四个独立问题：router 的 FP32 排序、token dispatch 的通信、latent expert 的矩阵乘、shared branch 的聚合。只回答“每个 token 选 16 个 expert”是不够的；还要问 token 在哪个 rank、shared expert 权重是否复制、latent reduction 在 norm 前还是后、以及 tail add 何时发生。

SGLang 支持 MegaMoE、DeepEP、Mooncake、Ascend-FuseEP 和 MoRI 等 A2A backend。A2A 路径下，rank 直接处理自己持有的 token rows，避免把已经分片的 token 重新做普通 DP gather；这正是 K3 LatentMoE 与服务拓扑绑定的地方。SGLang `main` 还为 shared expert 提供可配置 TP group：先 all-gather 输入，执行 TP-sharded shared MLP，再 reduce-scatter 回 token rows，并检查 shared intermediate size 可被 TP size 整除。`v0.5.20` 已有较专用的 DP-attention/Ascend shared-expert 通信路径，但覆盖条件不同。

### 88.17.2 SBO 与“把通信藏在计算下面”

两个版本都包含 single-batch overlap（SBO）分支：把 shared expert 放到 side stream，使它与 routed A2A 或 latent tail 重叠，直到真正消费 shared output 前才 join。这个优化依赖 shared branch 与 routed branch 的输入/权重独立，以及 CUDA stream event 的生命周期正确；它不是打开一个 flag 就能保证收益。

`main` 还把 shared-expert TP communication 和 NPU fine-grained dual-stream 单独建模。面试中可以用下面的时序追问实现正确性：

```text
main stream: gate -> Top-K -> latent down -> A2A -> expert GEMM -> latent up
side stream:                         shared gather -> shared MLP -> reduce-scatter
join point:                                                    tail add
```

如果 join 太早，SBO 隐藏不了通信；如果 shared 输出没有正确记录 stream，allocator 可能复用仍在写的 buffer；如果把 TP-sharded shared output 当成完整和，结果会出现重复或缺少 TP partial sum。源码分支能证明设计存在，不能证明所有 batch、capture mode、NPU/CUDA/ROCm 组合都通过。

### 88.17.3 KDA decode 的 capability gate

`KimiK3DeltaAttention` 同时准备普通路径和 fused decode 路径。权重加载后，SGLang 检查卷积权重、`A_log`、`dt_bias`、head 数和 dtype 是否符合编译 kernel 的固定 layout；满足时把 conv、decay、output-norm 参数交给 `kda_fused_decode`，不满足时回退到普通递归 attention chain。K3 的 full-rank gate 还把 output-norm gate 通过 stash 交给 backend，backend 未消费时再由模型层执行 gated RMSNorm。

因此 KDA serving 的正确问题不是“有没有 fused kernel”，而是：

1. prefill/chunk、decode 和 target verify 是否走相同的 beta/gate 语义；
2. recurrent state、short-conv state 和 MLA KV 是否在同一个请求 manifest 中分别保存；
3. 固定 shape 不满足时 fallback 是否仍与 reference path 数值一致；
4. CUDA graph、side stream 和 allocator 是否在 cache restore/replay 后保持顺序。

### 88.17.4 视觉实现的版本边界

`kimi_k3_vl.py` 在 `main` 与 `v0.5.20` 逐字节一致，因此视觉路径不能被误写成 `main` 新增。代码已经包含 MoonViT3d patch embedding、2D RoPE、temporal position、`cu_seqlens` 变长 segment、SDPA/Triton/FA4/FlashInfer-CuDNN backend 选择、按形状的 FA4 fallback、fused RoPE、可选 CUDA graph 和 PatchMerger projector。

这个对照本身也是面试知识点：文本 serving 的通信/量化演进与视觉 encoder backend 的版本变化可以不同步。看到 `KimiK3ForConditionalGeneration` 和视觉类名，只能说明入口存在；还要分别验收视觉权重、patch/grid 对齐、图像数值、CUDA graph replay、文本 KV/KDA state 注入和多模态 tool trace。

### 88.17.5 证据分层

| 证据 | 可以说明 | 不能说明 |
|---|---|---|
| SGLang `v0.5.20` K3 文件 | stable tag 中有文本/视觉实现入口 | 目标 wheel、完整权重和目标卡已通过 |
| SGLang `main` K3 文件 | DP/EP/shared-TP/ModelSlim/KDA overlap 的当前源码路径 | 这些改动已经进入 stable 或线上 SLO |
| HF config/index/reference | 模型结构、分片和双状态语义 | SGLang kernel 数值与生产恢复 |
| 本地 full-weight load/profile | 某 revision、硬件、依赖组合的实测结果 | 可迁移到所有 provider、GPU 和 batch |

所以 K3 的完整 serving 结论应保持为：**stable 有基础实现，main 有更近的 runtime 演进，完整权重加载、双状态恢复、目标硬件 profiling、视觉正确性和 tool/verifier acceptance 仍需单独验收。**

## 88.18 SGLang 对照题

**问：SGLang `main` 有 K3 的 shared-expert TP，是否说明 v0.5.20 也有同样能力？**

答：不能直接说明。两个版本都有 K3 和部分 shared-expert communication，但 `main` 的 `shared_experts_tp_size`/process-group 路径更通用；应以固定 stable 文件、配置开关和目标 backend 逐项比对。

**问：为什么 LatentMoE 的 latent reduction 必须发生在 RMSNorm 之前？**

答：TP 或 EP rank 可能各自产生 latent partial sum，而 `norm(sum(x_i))` 不等于 `sum(norm(x_i))`。因此先完成 latent space 的正确归约，再做 routed branch 的 RMSNorm，最后 up projection；通信顺序是数值语义的一部分。

**问：KDA fused decode 被启用就能证明 K3 serving 已验收吗？**

答：不能。它只证明当前 shape/dtype/backend 命中了 capability gate。还要测试 prefill/decode/verify 语义、recurrent/conv state 恢复、MLA cache、fallback 数值、目标硬件 profiling、tool parser 和 verifier。

**问：视觉文件 main 与 stable 一致，是否意味着多模态服务已经稳定？**

答：只说明这两个源码快照的视觉实现内容一致；完整权重、patch/grid 对齐、视觉数值、CUDA graph、文本双状态注入和端到端多模态 acceptance 仍然是独立门禁。

## 88.20 main commit history：runtime 是移动中的证据

上一节只比较了两个源码快照；继续追踪 commit history 后，可以看到 K3 serving 的难点并不是“把模型类注册进去”，而是不断修正状态、stream、权重布局和硬件拓扑之间的契约。SGLang commit API 快照覆盖到 2026-09-22T06:35:12Z，响应哈希为 2f7d8402aa104fe9a5b7839e85c9eb56060e98f7aeb4aecb787005f775351e04。

首先，KDA 的 deferred gate 需要明确投影责任。8ac19cc 把 f_a 交给 fused decode，让 kernel 自己应用 f_b；如果模型层仍提前做完整 f_a @ f_b，就会重复投影。c4d3770 则修正 CUDA graph 中的 fork/join：主 stream 先捕获宽 QKVG projection，side stream 处理小 GEMV，最终在同一 capture segment 内等待。这个修改说明 side-stream 优化必须同时验证数值依赖、capture 图拓扑和 replay 后的 stream 数量。

其次，权重加载本身是 runtime correctness 的一部分。c2c3629 为 experts.<id>.w1/w2/w3 建立直接查找路径，并把 ModelSlim 的原始 q/k/v/g 名称映射到 fused QKVG；72d5c5b 让 fused MoE finalize 兼容 FP32 routing weights。模型 config 正确但 tensor-name、packed/scale 配对或 dtype contract 错误时，仍可能在加载阶段产生错误结果。

再次，拓扑不能只用一个 TP 数字描述。f4c2563 把 PP prefill、DCP decode、PD disaggregation 和 DSpark draft/verify 放进同一 runtime 组合；2d0e94e 修正 shared-expert process group 的读取；8ac39c6 与 cb32dbc 分别增加 Ascend A5/NPU 和 ROCm 的 KDA/MoE 分支。正确的 manifest 至少要绑定 model revision、源码 commit/blob、hardware backend、process groups、prefill/decode/verify mode、state layout 和 fallback policy。

这些提交证明 upstream 在持续修正实现，并不证明某个 stable wheel、完整 K3 权重或目标硬件已经通过验收。面试中应把“代码修复存在”“该 commit 的 CI/测试通过”“本机数值复现”“目标卡 profiling”“线上 tool/verifier acceptance”分成五个命题。

**追问：为什么 K3 的 stream 修复也属于模型正确性？**

答：side stream 读写的是与主 stream 共享的 hidden state、gate 或输出 buffer；等待位置错误会造成未定义数据、重复计算或 allocator 提前复用。即使最终 tensor shape 正确，也必须检查 event/wait、CUDA graph capture/replay、异常路径清理和 cache restore 后的顺序。

**追问：PP prefill、DCP decode 和 DSpark verify 能否只复用一个 KDA state？**

答：不能直接假设。KDA 的 recurrent/conv state、MLA 的 key/value cache、draft/target 的接受长度与 rollback 都有各自的生命周期。必须按 mode 写入和恢复 manifest，并用 prefill、decode、verify、拒绝 token、跨节点传输和故障恢复做数值回归。

## 88.19 小练习补充

8. 画出 `hidden -> router -> latent down -> EP/A2A -> expert -> latent reduce/norm -> up` 的 tensor shape 和通信边界，分别标出普通 TP、EP A2A、shared-expert TP 三种情况下的 collective。
9. 构造一个 KDA fused-decode capability manifest，分别记录 conv layout、`A_log`、`dt_bias`、gate、state dtype、fallback 和目标 GPU；故意改坏一个字段，验证系统拒绝或回退。
10. 对照 SGLang `v0.5.20` 与 `main` 的 K3 文件，写出“源码存在”“stable release entry”“完整权重加载”“目标硬件验收”四个状态，不允许用一个 `supported=true` 替代它们。

## 88.21 当前时点复验：不要把稳定页面写成模型升级

2026-09-23 的三代理复验得到以下分层结果：

| 观察 | 正确写法 | 不能越级写成 |
|---|---|---|
| AA K3 `max` 的 `43.5938` Intelligence Index、速度、成本、1M context | 当前 provider/configuration 观察字段 | 官方参数、架构或独立 benchmark |
| DataCurve `309/451`、Pass@1/Pass@4、平均 token/steps/cost | `mini-swe-agent + tools + environment + verifier` 系统结果 | K3 裸模型能力或可迁移到其他 harness 的分数 |
| README/HF metadata 哈希稳定，revision 仍为 `f831ab...` | 当前没有观察到 artifact revision 漂移 | 发生了新版本发布或架构升级 |
| SGLang main 只出现 import/type annotation 变化 | 当前源码快照无实质 runtime 技术变化 | 新增了可宣称的 serving 能力 |

这里的“稳定”是证据快照稳定，不是生产系统已经验收。完整权重加载、KDA recurrent state 与 MLA cache 的双状态恢复、目标 GPU profiling、视觉正确性、tool/schema/idempotency/verifier 和 SLO 仍需独立门禁。

### 88.21.1 发布限制如何转成系统设计

- `2.5x scaling efficiency`：记录为发布方声明；复现时锁定 hardware、backend、序列长度、harness、warmup/iters 和 verifier。
- preserved thinking history：保留 channel、tool/index、thinking 引用和工具结果等结构化状态；不能用 visible response 的拼接替代。
- 跨模型切换不稳定：迁移 manifest 需检查 model/revision、协议 schema、reasoning effort、工具状态和 cache 状态，必要时拒绝 continuation。
- excessive proactive：模型输出是提案，不是授权；仍需权限、预算、sandbox、幂等和 artifact verifier。

因此 K3 的面试主线可以从“新架构名词”推进到“证据和控制面”：榜单负责发现，官方报告/仓库负责解释，harness 负责可比性，runtime manifest 负责恢复，verifier 负责把可解析输出变成可接受结果。

## 88.22 vLLM v0.30.0：模型接入之后，runtime 还在演进什么

K3 的 vLLM 支持不能用一个 `supported=true` 概括。公开 tag 显示 v0.29.0 已有 K3 registry、NVIDIA model 和 DSpark 实现入口；v0.30.0 是这些实现继续变化，不是 K3 首次进入 stable。固定 release commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`（2026-09-22）；release、PyPI metadata 和源码快照详见[研究笔记](../../research/model-update-2026-09/kimi-k3-source-notes.md#2026-09-24-vllm-v0300stable-release-与-k3-runtime-实现演进)。

### 88.22.1 权重加载是一个有结束边界的生命周期

模型权重不是一口气到齐的。父级 loader 可能因文件分片或非连续 stream 多次调用 `load_weights`。如果每次调用后都做 MegaMoE packed-weight 转换，当前阶段看到的不一定是完整权重，过早 finalize 还可能重复做昂贵转换。v0.30 把这一步放到统一的 `process_weights_after_loading` hook：先完成全部映射/装载，再构造推理用的权重表示。这是典型的两阶段生命周期：

```text
streamed checkpoint tensors
    -> map/load all parameter fragments
    -> post-load finalize exactly once
    -> runtime-ready transformed weights
```

v0.30 还允许 MegaMoE weight-cache daemon 导出的 transformed buffers 通过 IPC 零拷贝复用，并释放原始 packed 参数。面试时除了问“是否省了一次转换”，还应问缓存 buffer 的 owner、生命周期、失败回退和 raw/transformed 两份内存何时释放。本轮只核验源码，不是启动时延或显存的本地实测。

### 88.22.2 PP 与 speculative decode 需要携带辅助状态

K3 同时包含 AttnRes 的深度状态、MLA attention cache、KDA recurrent/conv state，以及 DSpark/MTP draft-target 的验证和回滚边界。v0.30 的 K3 模型实现支持 auxiliary hidden state 跨 PP stage 传递，并在开启 AttnRes auxiliary stream 时拒绝不安全的非末级 stage 边界配置。含义是：状态不能只由“当前 token 到第几个”或单个 KV cache 长度表达；还要说明捕获了哪些层表示、属于 draft 还是 target、PP rank 如何传递，以及拒绝 token 时怎样回滚。

同一版本的 DSpark context-KV 快速路径按 cache dtype、量化 scale、各层 dtype 一致性和 block layout 设置 gate。只有满足条件时才批量投影和插入；其他情况必须走兼容路径。KDA state dtype 也显式考虑 `mamba_ssm_cache_dtype`。因此 serving manifest 应分别登记 attention KV、context KV、KDA SSM state、AttnRes auxiliary hidden states 和 speculative state，而不是把它们折叠成一个 `cache_dtype/cache_length`。

### 88.22.3 版本证据表与部署结论

| 证据 | v0.29.0 | v0.30.0 | 能得出的结论 |
|---|---|---|---|
| K3 registry/model/DSpark entry | 已存在 | 仍存在且源码有演进 | stable source entry 不是 v0.30 首次加入 |
| NVIDIA K3 model source | 87,313 bytes，`e74026a8…` | 88,517 bytes，`27cbd7f0…` | 实现变化，不能只依据文件变大推断性能/正确性 |
| DSpark source | 19,451 bytes，`e86bb984…` | 20,001 bytes，`ba5c555c…` | draft/context-KV 和状态路径继续发展 |
| PyPI x86_64 wheel | 0.29.0 wheel 已有发布记录 | 0.30.0 wheel metadata 可用；314,883,777 bytes | artifact 可获取不等于本地安装或硬件验收 |

面试回答应分开说四件事：tag 中有代码、PyPI 有 wheel、目标环境能装载完整权重并正确运行、生产负载达到 SLO。当前证据支持前两层的 release/source 事实，不支持后两层。

**追问：为什么 stable v0.29 已有 K3，而 v0.30 的 model.py 仍明显变化？**

答：模型注册只是入口。权重流式加载、post-load 转换、IPC cache ownership、PP 辅助状态、DSpark context-KV layout 和 KDA state dtype 都会在实现演进中变化。应固定 tag 比较源码，并分别运行完整权重加载、数值回归、cache recovery、目标硬件 profile 和 draft acceptance；不能把 `supported` 文档标签或 registry 命中当作生产验收。

## 88.23 DSpark 自适应 verify：动态长度如何落入固定图

vLLM main 的 K3 DSpark 实现加入可选 confidence head，为每个 draft 位置估计接受概率。scheduler 可据此为不同 request 选择不同 draft/verify 长度；但 CUDA graph 希望形状稳定。实现折中是 capture 到 `k+1` 的上界，再让 device-side offsets/masks 在一次图回放中表达同批请求各自 1 到 `k+1` 的长度。它不是把所有请求强行补成相同的逻辑长度，而是用固定执行外形承载变长 request metadata。

这会影响 row 分类和状态容量：adaptive scheduler 会重写 `num_scheduled_tokens`，所以不能再用它与 draft 数的简单等式判断是否处于 verify；实现改看 request 的 prefill/decode 状态。RecoverSSM 只为 `k+1` 窗口准备状态，较长的 profiling dummy rows 必须被 mask 出去。capture 使用 scheduler 承诺的 `max_query_len`，而非某个 dummy batch 的实测最大行长，以保证 graph 对之后的 replay 仍成立。以上来自 2026-09-23 main commit；vLLM stable `v0.30.0` 早于该改动。

另一个 2026-09-28 合入 main 的修复为 DSpark context KV pointer 加入 buffer-address key。cache rebinding 导致任一层 `data_ptr()` 改变时，旧 pointer cache 必须失效并重建；图捕获期间不允许更新。PR 描述的失败包括 GPU memory fault 与静默内存破坏，说明 context cache 的正确性依赖分配生命周期，而不是仅依赖模型层或逻辑 cache 名称。

## 88.24 ROCm 后端契约与 benchmark 证据

Kimi K3 的 MI355X recipe 显示，模型接入后仍有量化布局与 attention backend 的双向契约。SiTUv2 的新 AITER 路径是 a4w4，要求匹配的 vLLM flag、AITER `v0.1.20+` 和对应 tuned config；兼容保留的 flag 名字仍含 `A8W4`，不能从变量名猜测实际 kernel。另一个 AITER flag 若单独选择 interleaved weight kernel，而 vLLM 没有做对应权重 shuffle，PR #53954 记录可能出现无报错的退化输出；截至本次快照该 PR 仍 open。

DSpark verify 也需要后端能力 gate。ROCm AITER MLA 对非因果 draft block 检查 `causal` 接口与受支持形状；例如 FP8 的 2-token block 无匹配 kernel 时应显式拒绝并切换到 `TRITON_MLA`，而不是冒险进入不兼容分支。Backend 是否支持某个 dtype、query length、mask 与 DCP 组合，要作为 capability tuple 测试。

发布方 AgentX 配置还把真实 block verification 与 synthetic-acceptance throughput sweep 分开。synthetic sweep 注入从 B300/SPEED-Bench 得到的 golden acceptance length、绕开真实 target verification，只能估计给定接受长度假设下的吞吐；正确性、线上 acceptance rate 与用户服务都必须用真实校验路径。完整源码与约束见[研究笔记](../../research/model-update-2026-09/kimi-k3-source-notes.md#2026-09-29-vllm-recipeadaptive-dspark-rocm-serving-)。

**面试追问：为什么 `synthetic_acceptance_length` 下的 TPS 不能和真实 block verification 的 Agent 分数并列？**

答：synthetic 路径人为指定了 draft 接受长度并跳过 target verification，改变了计算量，也不测量 draft 与 target 是否逐 token 一致。它隔离的是给定接受率假设下的调度/吞吐；真实任务还要执行 verifier、处理拒绝与回滚，并受工具 schema、环境和任务 verifier 影响。

## 88.25 小练习补充

11. 画出 K3 adaptive DSpark 的长度账本：request 的 confidence、draft tokens、verify query length、`k+1` graph bound、KDA offsets 和 RecoverSSM capacity。分别构造 1-token、`k`-token、`k+1`-token、prefill 与超界 profiling dummy case，解释哪些要执行、mask 或拒绝。
12. 对照 vLLM stable `v0.30.0`、main commit 和 K3 recipe pinned commit，标注每项功能是 stable source、mutable upstream 还是 pre-release recipe；再把真实 block rejection 与 synthetic acceptance 分成两个 benchmark manifest。

## 88.26 DFLASH：把 K3 的 target features 变成并行 draft

Kimi K3 的 DFLASH 接入提供了一个很适合面试的推测解码案例。公开的 `modal-labs/Kimi-K3-DFlash` 是约 2.6B BF16 的 draft-only 模型，不能脱离 K3 单独提供语言模型服务。其 config 使用 6 层 draft Transformer，读取 K3 的 93 层 target 中的 `[19,37,54,66,78,90]` 层输出，推荐 block size 为 8；这些字段描述的是 draft 与 target 的接口合同，不是 K3 本体参数规模或新的排行榜模型。

DFlash 的关键不是简单地把一个小模型接在 K3 前面。论文描述的流程是：K3 prefill 时抽取多层 hidden states，经 projection 融合为 target context feature；draft 将该 feature 投影进每一层的 Key/Value，并把它保存在 draft KV cache 中。之后 block diffusion 在一次 forward pass 中同时生成一个 block 的 mask positions，K3 再并行验证。这样 draft 不必从 token embedding 重新猜完整未来，且 draft 层数增加时 target context 不会只作为输入信号逐层稀释。

训练也模拟了真实 verify 形状。每个 response 随机选 anchor，anchor 后的 `block_size - 1` 个 token 同时 mask；不同 block 拼成一条序列，用 sparse mask 让同一 block 内可双向注意、不同 block 互相隔离。由于第一个错误会阻断后续接受，论文用 `w_k = exp(-(k-1)/gamma)` 强调 block 前部位置，并冻结与 target 共享的 embedding 和 LM head。面试中可以把它和普通 AR drafter 对照：AR draft 的成本随 draft token 数线性增长，DFlash 的 draft forward 更接近一次并行 block 成本；最终速度仍取决于 draft latency、verify latency 和平均 acceptance length。

SGLang PR #40794 解决的是接入契约。K3 原有 DSpark capture 的是 layer outputs，DFLASH 的 `target_layer_ids` 也命名 layer outputs，所以新 hook 复用 DSpark taps 且不做通常的 `+1` layer-input 偏移。cookbook 固定公开 draft、block size 8，并按 pipeline parallel、DP attention、NPU、Hopper/AMD 等条件禁用；PR 的 8xB300 数字使用未公开 production draft 和多个合并 PR，不能当成单 PR 加速或公开 checkpoint 的复现结果。SGLang stable 仍为 `v0.5.20`，该功能属于 main source evidence。

| 面试问题 | 应检查的账本 |
|---|---|
| 为什么 target hidden feature 要注入每层 K/V？ | 条件信息是否跨 draft depth 保持；draft KV 的生命周期与 block iteration 是否一致 |
| 为什么 target layer id 不能随意 `+1`？ | hook 捕获的是 layer input 还是 layer output；目标索引的语义必须和 capture tap 对齐 |
| 为什么 DFLASH 的 TPS 不能直接由 accept length 推出？ | `T_draft`、`T_verify`、并行 block size、GPU backend、graph/capture、拒绝回滚和请求分布都影响吞吐 |

## 88.27 小练习补充

13. 根据 K3 的 93 个 target layers 和 DFLASH 的六个 `target_layer_ids`，画出 prefill hidden extraction、projection、draft KV injection、block mask 和 target verification 的时序；分别标注 draft-only 权重、target 权重和共享 frozen embedding 的所有权。
14. 构造 block size 8/16 的 acceptance manifest，记录公开 draft revision、target revision、硬件、backend、真实 verifier、平均 accept length 和 wall-clock；把 SGLang PR 的未公开 production checkpoint 结果单独标记为不可复现发布方 evidence。
