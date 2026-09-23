# Kimi K3：官方资料摘记与扩写入口

首轮核验日期：2026-09-22；当前时点复验：2026-09-23。本文是研究记录，尚非正式书籍章节；本次补充字段来自 Kimi 官方仓库、技术报告 PDF、许可证、vLLM/SGLang 官方源码和排行榜快照。

## 已读取来源

- [官方发布文章](https://www.kimi.com/en/blog/kimi-k3)：已读取正文。
- [官方博客索引](https://www.kimi.com/blog)：将该文章标为 2026-07-16。
- [Artificial Analysis 条目](https://artificialanalysis.ai/models/kimi-k3)：当前仍有 `max` 和 `low` 配置；本轮详情页与 `/zh` 首页均已重新抓取。
- [DeepSWE](https://deepswe.datacurve.ai/)：页面数据包含 kimi-k3、mini-swe-agent 和 max 配置。
- [Kimi K3 官方仓库](https://github.com/MoonshotAI/Kimi-K3)：包含 README、`k3_tech_report.pdf`、许可证和完整权重发布说明。
- [Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)：本轮从 GitHub API 获取 PDF 并提取正文文本核验。
- [Kimi K3 License](https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE)：核验了权重、配置、代码和文档的授权范围及商业条件。
- [Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)；[FlashKDA](https://github.com/MoonshotAI/FlashKDA)：报告关联的官方实现仓库。

## 2026-09-20 官方权重 revision 与 KDA serving 补证

本轮仍以 Artificial Analysis 的 `kimi-k3` 与 DataCurve 的 K3 精确行作为模型锚点；以下官方仓库、模型卡和 serving recipe 只用于核验这个已发现的锚点，不引入新的模型候选。

### 1. Hugging Face 权重仓库与配置已固定

- [Hugging Face API 元数据](https://huggingface.co/api/models/moonshotai/Kimi-K3) 返回官方 `moonshotai/Kimi-K3`，revision/commit `f831ab66814297da540d832a5235f8e904f29d06`，`lastModified=2026-09-02T02:22:41Z`；本轮响应 9,377 bytes，SHA-256 `24b2606b8f2604828ce078a93f76209b1ce97e366e4cfa0ea20176975d9ac07e`。
- API 的文件清单包含 `model-00001-of-000096.safetensors` 到 `model-00096-of-000096.safetensors`、`model.safetensors.index.json`、`encoding_k3.py`、`configuration_kimi_k3.py`、`modeling_kimi_k3.py` 和视觉处理文件；共 119 个文件。元数据声明 safetensors 参数总量 2,779,931,837,184 bytes，其中 U8 2,722,740,830,208、BF16 57,179,884,544、F32 11,122,432。这个证据证明官方仓库存在具体分片和 revision，但本轮没有把约 2.8 TB 权重下载到本地。
- 固定 revision 的 `config.json` 响应为 7,006 bytes，SHA-256 `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`。配置级字段包括 `KimiK3ForConditionalGeneration`、BF16、`max_position_embeddings=1048576`、`hidden_size=7168`、`num_hidden_layers=93`、`first_k_dense_replace=1`、`q_lora_rank=1536`、`kv_lora_rank=512`、96 attention heads，以及 `num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`。
- `linear_attn_config` 列出 69 个 KDA layer 和 24 个 full-attention layer，`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`、full-rank gate；视觉配置还明确 27 层视觉塔、14×14 patch、1024 vision hidden size 和 7168 text hidden size。量化字段是 compressed-tensors 的 MXFP4 packed weights、4 bit、group size 32，并显式忽略 attention、shared experts、lm head、vision tower 和 projector 等模块。以上是固定 config 的结构/部署字段，不等于完整训练 recipe 或端到端性能。

### 2. FlashKDA 的代码版本、状态接口与硬件边界

- [FlashKDA README](https://github.com/MoonshotAI/FlashKDA/blob/master/README.md) 本轮 4,431 bytes，SHA-256 `fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2`；master 的 GitHub Atom feed 显示最新 commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934`，时间 `2026-09-01T08:16:56Z`，提交说明是将 fp16 Neumann inverse 换成 `8x8 fp32 forward substitution + 16x16 bf16 merge`。这是实现历史证据，不应写成 K3 训练算法。
- README 要求 SM90+、CUDA 12.9+、PyTorch 2.4+；FlashKDA 作为 `flash-linear-attention >= 0.5.0` 的 `chunk_kda` backend 自动 dispatch，也允许用 `FLA_FLASH_KDA=0` 回退 Triton。debug logging 会明确显示命中或拒绝原因，适合做 serving path audit。
- kernel API 以 BF16 的 `[B,T,H,K]`/`[B,T,H,V]` q/k/v 和 FP32 的 `A_log`、`dt_bias` 为输入，当前要求 `K=V=128`；支持 BF16/FP32 的 initial/final recurrent state 和 `cu_seqlens` 变长批处理。变长模式下 `B=1`、状态形状为 `[N,H,V,K]`，不能把普通 batch 的 `[B,H,V,K]` 约定直接套用。
- 官方 benchmark 文档绑定 `T=8192`、`D=128`、warmup 30、iters 200、repeats 5。H20/H96 固定长度的 `flash_kda` 为 2.6220 ms，相对 `fla_chunk_kda` 的 4.8388 ms 为 1.85×；GB200/H96 固定长度为 1.0087 ms 对 2.3271 ms，为 2.31×。这些是仓库发布的硬件/基线结果，不是本地复现，也不能外推为 K3 端到端吞吐或所有 GPU 的收益。
- Atom feed 还记录了 `cu_seqlens` prefix-sum/binary-search 优化、TMA proxy fence 修复和更多架构支持；面试中应把算法状态更新、变长准备、异步内存栅栏和硬件专用 kernel 分开回答。

### 3. vLLM recipe 已公开，stable release 与优化 recipe 要分开记录

- [vLLM Kimi K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3) 页面标注 `Updated 2026-09-10`；对应 YAML 19,414 bytes，SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`，要求 vLLM `0.29.0`，但说明实际使用 K3-enabled nightly/image，CUDA 路径是 cu130、宿主需要 r580+ driver。
- recipe 将 K3 分成 H200/B200/B300/GB200/GB300/MI355X/Ascend 910C 适配，并提供 TP8、TEP16、TP8×PP2、DEP 和 disaggregated prefill/decode 账本。Blackwell profile 使用 FP8 KV、`TOKENSPEED_MLA`、prefill query quantization 和 prefix caching；由于 hybrid KV manager 同时管理 MLA attention 与 KDA recurrent state，recipe 建议 `--prefix-match-unit 128` 细化 prefix hit 边界。
- 长上下文 decode 可选 DCP：`--decode-context-parallel-size 8`、A2A 通信和 TOKENSPEED MLA；PD cluster 将 prefill 的 TEP/TP8 与 decode 的 DEP/TP1 分开，不能把单一 TP 数字当成所有阶段的并行策略。跨节点 RDMA 使用 `deepep_v2`，NVLink 使用 `flashinfer_nvlink_one_sided`；DeepGEMM MegaMoE 适用于跨节点 NVLink DEP，但不适用于跨节点 RDMA。
- recipe 还显式提示 K3 偶尔会产生自身 parser 不期望的 tool-call 格式，宿主应做 schema validation 与 retry。这是 runtime 兼容性证据，不应把解析失败率写成模型能力指标。vLLM `0.29.0` stable release/source entry 已由 PyPI metadata 与 v0.29.0 tag 固定；本地 wheel 安装、完整生产 profiling 和线上 tool-call acceptance 仍未由本轮证据确认。

- [PyPI vLLM release metadata](https://pypi.org/pypi/vllm/json) 本轮通过 `10.24.27.134:7890` 获取，完整 JSON 为 `253,652` bytes，SHA-256 `a232f3e31b111ebfb0d99cbe69b71da682a680ccb4cbf1c57f8cdab3801e5b94`；`info.version=0.29.0`，x86_64 stable wheel 为 `315,961,042` bytes、SHA-256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`，aarch64 wheel 为 `310,033,787` bytes、SHA-256 `e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b`，均于 `2026-09-09` 上传。metadata 是发布 artifact 证据；本轮没有下载或安装 wheel，也没有把失败的截断 source-tar 下载当作证据。
- 与公开 `v0.29.0` tag 对照：`registry.py` 为 `63,102` bytes、SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`，K3 package `__init__.py` 为 `1,444` bytes、SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`，NVIDIA K3 `model.py` 为 `87,313` bytes、SHA-256 `e74026a83c28cce7ee62889e9ed434d1fff6076dbd263d388aa379f4cf987a4d`。tag 源码包含 K3 registry、NVIDIA/ROCm/TPU package 分支和 K3 model implementation；这证明 stable release source entry 存在，不证明本地 wheel 在目标卡上加载完整权重或通过生产验收。

## 官方文章披露的内容及边界

官方发布文章和技术报告共同支持模型总参数 2.8T、原生多模态与 1,048,576 context；README/报告给出 104B activated parameters、93 layers、69 KDA + 24 Gated MLA、1 个 dense layer，以及 Stable LatentMoE 的 896 routed experts、每 token 16 个 routed experts 和 2 个 shared experts。这里的参数口径仍按官方定义记录，不能据此推导任意 batch、精度或并发下的显存占用。

技术报告进一步给出 Quantile Balancing、Per-Head Muon、Sigmoid Tanh Unit（SiTU）、Gated MLA，以及从 SFT 阶段起使用 MXFP4 权重与 MXFP8 激活的量化感知训练的实现语境。报告披露足以支持教学级公式和配置总结，但不支持把这些组件都归因成 K3 首创，也不支持补写未公开的完整 optimizer/数据配方。

基础设施方面，文章提出静态形状、关键路径无主机同步的专家并行训练方法，建议在至少 64 个加速器的超节点部署，并称正在向 vLLM 贡献适配 KDA 的前缀缓存实现。这是发布文章的时间语境，不能据此认定相关代码截至今日已合并。

发布文章的 Availability 段还给出了接入层字段：Kimi Work 桌面端要求 3.1.0 或更高版本，Kimi Code 可通过 `/model` 选择 K3，API 平台提供 `kimi-k3`；文章列出的价格为 cache-hit 输入 $0.30/百万 token、cache-miss 输入 $3.00/百万 token、输出 $15.00/百万 token。文章将这些推理服务描述为由 Mooncake 的 disaggregated inference architecture 支撑，并自报官方 API 在编码工作负载上的缓存命中率超过 90%。这些是发布文章快照中的产品/价格声明，价格、客户端版本和命中率都应按账户、时间、工作负载和平台重新核对，不能当作永久费率或普遍性能保证。

2026-09-18 已从官方仓库核验 `k3_tech_report.pdf`、README 和 Kimi K3 License。README 声明完整权重已发布，并给出 Hugging Face/ModelScope 入口；本轮没有稳定取得具体权重仓库、文件清单和 revision，因此只能写成“官方 README 声明已发布”，不能写成“本地已下载权重”。

## 2026-09-18 技术报告增补

### 三条扩展轴与混合注意力

报告把 K3 的扩展拆成 sequence length、network depth、model width 三条轴：KDA + Gated MLA 处理序列长度，Attention Residuals 处理深度方向信息流，Stable LatentMoE 处理宽度和专家容量。每 3 个 KDA layer 后接 1 个 Gated MLA layer，backbone 末尾再放置一个 Gated MLA，保证最后一层具备 global attention。KDA 层使用 NoPE；KDA 负责递归状态、位置/新近性和高效长序列混合，Gated MLA 负责全局内容交互。

报告披露 KDA 的 lower-bounded decay：`g_min=5`，通过 scaled sigmoid 把 retention factor 的下界纳入参数化，使 diagonal/off-diagonal tile 都能走 dense TensorCore 矩阵乘法，避免额外的 position-pair diagonal path。KDA 与 Gated MLA 都使用 input-dependent full-rank output gate，MLA 输出训练时保持 FP32。这里的 lower bound、3:1 比例和 K3 层数属于 K3 report 证据，不应回写成 Kimi Linear 论文的普遍配置。

### Block Attention Residuals 与 Stable LatentMoE

K3 使用 Block AttnRes：8 个 block、每 block 12 层，加 embedding representation 后形成 9 个 block-level representations。这个配置把深度方向的保存和通信从按层增长降到按 block 增长，但不能把它误写成论文所有模型都固定使用的参数。

Stable LatentMoE 在 latent width 中运行 routed experts，再投影回 full width；shared experts 保留 full-width 路径，aggregation 后增加 RMSNorm 稳定 routed branch 的尺度。SiTU-GLU 对 gate/up 两支使用 scaled `tanh` soft cap，控制 activation outlier 和低精度溢出。Quantile Balancing 从 router-score quantiles 直接得到 expert bias，以最大-score balanced assignment 为目标，通过交替 coordinate minimization 求 token/expert 方向的量化阈值；推理时使用冻结 bias + fixed Top-k，不需要运行时 quantile 计算。

### 训练、RL 与基础设施

报告披露 Per-Head Muon、SFT 起步的 MXFP4/MXFP8 QAT，以及覆盖 general reasoning、agentic tasks、coding 和 multiple reasoning efforts 的 RL。长轨迹环境包含 web search、professional knowledge work、software engineering/kernel optimization、vision-in-the-loop tool use、persistent assistant workflows、web development 和 autonomous execution；multi-teacher on-policy distillation 用于把 domain/effort-specialized policies 合并为统一模型。

工程侧包括 KDA fused kernels、KDA context parallelism、state-aware prefix caching、MoonEP 的静态形状/zero-copy expert parallel、partial rollouts、external KV-cache retention、adaptive throttling、resumable microVM sandboxes，以及 cache-aware/budget-aware effort scheduling。关联仓库和 vLLM `0.29.0` stable source 已证明实现入口；本轮尚未完成 backend 本地执行、目标硬件 profiling 和生产 serving 验收。

### XTM 风格消息协议

报告第 46-47 页附近描述 XTM 风格模板和 `[open]`、`[sep]`、`[close]`、`[end_of_msg]` 特殊 token。全局 option 可声明工具和 reasoning effort；one-shot option 包括 `tool_choice` 与 `response_format`。支持通过后续 `tool-declare` message 动态扩展工具集，不必重建此前上下文；assistant channel 分为 `think`、`response`、`tool`，tool call/result 使用 `tool/index` 配对，thinking channel 即使为空也保留结构。

这说明 K3 的“保留思考历史”是消息协议和状态回放约束，而不是简单把隐藏推理当普通字符串拼接。跨模型切换、摘要、工具重试和缓存恢复都必须保留 channel、tool index、reasoning effort 及结果归属。

## 值得扩写的知识链

| 技术入口 | 应讲清的问题 | 优先对应书册 |
|---|---|---|
| KDA 与 Gated MLA | 序列状态如何保存；混合注意力如何取舍计算、记忆与检索能力 | 21 架构、24 推理框架 |
| AttnRes | 深度方向的信息累积与选择；与普通残差连接的区别 | 21 架构、13 数学 |
| Stable LatentMoE 与 Quantile Balancing | 专家稀疏化、路由负载和通信成本为什么相互牵制 | 21 架构、05 训练、23 Infra |
| Per-Head Muon 与 SiTU | 优化器作用的矩阵结构、激活控制和训练稳定性 | 05 训练、13 数学 |
| MXFP4/MXFP8 量化感知训练 | 训练时模拟低精度误差如何影响部署；与训练后量化的区别 | 05 训练、06 部署 |
| KDA 前缀缓存 | 注意力 KV 与递归状态各自需要什么缓存；共享状态如何恢复 | 24 推理框架 |
| 保留思考历史 | 多轮工具协议、历史回传和跨模型切换的兼容性 | 17 Agent、20 Runtime、22 协议 |
| 长任务评测 | harness、硬件、预算和任务版本如何影响可比性 | 07 评估、20 Runtime |

这些是调查与教学方向，不是对未公开实现的结论。后续每个主题按背景、旧方法问题、直觉、数学机制、最小例子、实现、工程取舍、局限和面试追问展开。

## 评测与使用限制

官方脚注明确混用了不同 harness；部分测试将 H100 环境改为 H20，部分比较对象发生 fallback。因此不能把该文所有分数拼成同条件排名。文章引述发布时 mini-swe-agent 下 DeepSWE 分数为 67.3，而本次榜单快照记录不同数值，后续需核对版本与更新时间，不能直接判为矛盾。

该文章的统一评测脚注还声明 K3 结果使用 `reasoning effort=max`、`temperature=1.0`、`top-p=1.0`，并按基准选择 Kimi Code、Claude Code 或 Codex harness。应把这组采样与 harness 条件随结果保存；缺少它们时，不能复算文章中的绝对分数或把不同条目拼接成排名。

文章特别提示 K3 依赖保留历史思考内容：没有按要求回传、或在其他模型的会话中途切换到 K3，质量可能不稳定。它还提示模型可能过度主动。这两点适合扩写成运行时兼容性与行为边界案例，不能仅靠更长上下文来解释可靠性。

## 下一步核验

1. 已固定 Hugging Face 的具体权重仓库、119 项文件 metadata、safetensors 分片清单和 revision；完整权重未下载，仍需在不下载完整权重的前提下核对 safetensors index、分片与 config 的一致性。
2. 查 KDA、AttnRes、Muon 及其相关论文，区分前作、复用技术与 K3 新改动。
3. 已固定 FlashKDA master 的最新 commit、README/API、H20/GB200 benchmark、vLLM 0.29.0 stable source entry、K3 recipe 和 SGLang main/v0.5.20 K3 model source；仍待核对具体 backend 本地执行、目标硬件 profiling 和 recipe 优化路径的实际运行结果。
4. 对报告中的 benchmark、硬件和 acceptance rate 做条件化复现，不把发布方数字当独立结论。

## 已完成的教学落地

已新增第十七册第 15 章，将发布文章披露、KDA/AttnRes 论文机制、长任务 harness、思考状态 manifest、fallback 和 benchmark 条件化评测串成正式教材。2026-09-18 又完成官方技术报告、仓库和许可证核验，并将 K3 配置收口到第二十一册第 88 章；2026-09-20 又固定 HF 权重 revision、配置字段、FlashKDA kernel/API/benchmark 和 vLLM recipe；2026-09-21 又固定 PyPI vLLM 0.29.0 和 v0.29.0 K3 source entry。具体权重下载、目标 hardware runtime、独立复现、完整优化 recipe、线上 acceptance rate 和目标硬件 profiling 仍待核验。

## 2026-09-23 当前时点复验：榜单指标与官方 revision 均未漂移

本轮重新抓取两个允许的模型发现入口，并重新核对 K3 详情页。结果没有产生新的模型候选，也没有发现 K3 的模型 revision 或架构变化。

- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) 三条代理逐字节一致：`1,783,605` bytes，SHA-256 `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 三条代理逐字节一致：`268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商按 canonical slug 去重后没有新增模型。
- [Artificial Analysis Kimi K3](https://artificialanalysis.ai/models/kimi-k3) 三条代理逐字节一致：`4,080,091` bytes，SHA-256 `340a36ac5922f354aa42553fe3da2ce4acf4231bbb6af97fd6a88b974dbea4ee`。当前 `max` 配置的 Intelligence Index 为 `43.5938229518782`，median output speed 为 `36.9982439081499 tokens/s`，cost per Intelligence Index task 为 `2.0001323004425493`，context 为 `1M`；这些仍是 Artificial Analysis/provider 观察字段。
- DataCurve 的精确行是 `mini_swe_agent_kimi_k3_max`：`309/451` 通过，Pass@1 为 `0.6851441241685144`，Pass@4 为 `0.8938053097345132`，平均成本 `$4.654682129933482`，平均输出 `81,499.84` tokens，平均 Agent steps `97.5876`，`4 runs`、`113 tasks`。这些数字必须完整标注为 `mini-swe-agent + tools + environment + verifier` 的系统结果，不能写成 K3 裸模型能力。

### 官方资料复验结果

- Kimi K3 [README](https://github.com/MoonshotAI/Kimi-K3/blob/main/README.md) 为 `45,004` bytes，SHA-256 `849a303d849486aac61d1f0c253e3a1148ca59e801be807151c2dc3cce57ccb2`；与 2026-09-18 快照逐字节一致。
- [Hugging Face API metadata](https://huggingface.co/api/models/moonshotai/Kimi-K3) 为 `9,436` bytes，SHA-256 `6dfa5b1680074f4dfdd79a36b3342a191aacc32e15a9bd4a3895b98570773939`；revision 仍为 `f831ab66814297da540d832a5235f8e904f29d06`。
- 固定 `config.json`、vLLM recipe 和 FlashKDA README 的前述哈希保持不变；SGLang `main` 当前 K3 文本文件为 `171,130` bytes，SHA-256 `9af22b45f8d8f8a5931c3bd60310316090973fd5cf9626aabd2a618a9e4baa6d`。相较 9 月 22 日的 `171,101` bytes，仅见 `array` import 和 `pad_input_ids` 类型注解变化，没有实质 runtime 技术变化。
- 因而本轮应写成“榜单指标当前复验、官方 revision/config/source 未漂移”，不能写成“发布了新 K3 版本”或“架构发生升级”。此前的 2.8T/104B、93 层、69 KDA + 24 Gated MLA、Stable LatentMoE、双状态 cache、vLLM/SGLang source evidence 继续有效；完整权重加载、目标硬件 profiling、hybrid cache recovery、tool acceptance 和生产 SLO 仍未验收。

### 适合面试的限制表述

1. 官方材料中的“约 `2.5x scaling efficiency`”是发布方声明，不能当成独立复现；回答时要说明比较对象、硬件、序列长度、实现和测量口径。
2. K3 依赖 preserved thinking history 时，宿主应原样回传所需的结构化思考/工具历史；这不是把隐藏推理任意暴露给下游，也不是仅追加 visible text 就能替代。
3. 在其他模型会话中途切换到 K3 可能造成质量不稳定，说明跨模型 continuation 是协议和状态兼容问题，不是单纯扩大 context window。
4. 官方还提示 K3 可能 excessive proactive；这应作为行为限制和 verifier/权限设计问题记录，不能直接归因于某个未公开的内部模块。
5. K3 发布评测混用了 Kimi Code、Claude Code、Codex 等 harness，以及 H20/H100 等硬件和 compaction 条件；DataCurve 的 `309/451` 与 Pass@4 也只能在其精确 harness、工具、环境和 verifier 下解释。

## 2026-09-21 固定权重 manifest、混合 cache 与源码边界

本轮没有从 HF 或 GitHub 另发现模型；所有实现资料继续绑定两个排行榜已经发现的 `kimi-k3`。通过 `10.24.27.134:7890` 重新读取官方 HF revision `f831ab66814297da540d832a5235f8e904f29d06`，不下载任何 safetensors 分片。

- `model.safetensors.index.json` 为 `59,764,096` bytes，SHA-256 `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`。索引的 `metadata.total_size` 为 `1,560,860,324,864` bytes，包含 `497,220` 个 tensor 映射，覆盖连续的 `model-00001-of-000096.safetensors` 到 `model-00096-of-000096.safetensors`。
- 固定 `config.json` 仍为 `7,006` bytes、SHA-256 `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`。索引审计确认 93 个 layer id 为 `0..92`；`first_k_dense_replace=1` 使 layer 0 为 dense MLP，后续 92 层具有 896 个 routed expert 的分片键；每个专家层均出现 `w1/w2/w3.weight_packed` 与配对的 `.weight_scale`，分别为 `247,296` 个。
- [HF API metadata](https://huggingface.co/api/models/moonshotai/Kimi-K3) 的 `safetensors.total=2,779,931,837,184` bytes 由 `U8/BF16/F32` 参数 dtype 统计组成；它不能与 MXFP4 packed safetensors 的 `metadata.total_size` 相加或互换。前者是 API 参数统计口径，后者是索引声明的分片文件权重大小。
- 官方 `modeling_kimi_linear.py`（固定文件 SHA-256 `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`）把混合状态实现为两类不同对象：`KimiDynamicCache` 为 full-attention 层保存 `key_cache/value_cache`，KDA 层保存 `conv_states/recurrent_states`；KDA 在带 cache 且单 token 时走 `fused_recurrent_kda`，prefill/chunk 路径走 `chunk_kda`，并通过 `cu_seqlens` 处理变长输入。
- `KimiMLAAttention` 通过 `past_key_values.update()` 追加 full-attention K/V；`KimiDeltaAttention` 则把短卷积状态、递归矩阵状态、衰减/`beta` 和 output gate 写回 `KimiDynamicCache`。因此 serving manifest 必须同时保存 MLA attention cache 与 KDA recurrent/conv state，不能用一个“KV cache 长度”字段代替两类状态。
- 新增零依赖审计脚本 [`kimi_k3_manifest_audit.py`](code/kimi_k3_manifest_audit.py)，在本轮完整 index/config 上通过：分片连续性、layer partition、expert 覆盖、packed/scale 配对、量化字段和“未下载权重”声明均通过。脚本只读取 JSON，不加载权重；本地没有完成模型加载、CUDA kernel 执行或端到端 serving。

本轮把 K3 状态从“实现资料补证”推进为“固定 manifest/config/runtime source evidence”。仍待核验：完整权重加载、目标 GPU 的真实 profiling、完整训练/optimizer recipe、线上 tool-call acceptance、hybrid cache 恢复回归和独立 benchmark。仓库 README/recipe/benchmark 和 stable source entry 仍不能改写成生产验收。

## 2026-09-21 vLLM upstream/runtime recheck：实现入口不等于稳定 artifact

本轮继续只核验两个排行榜已经发现的 `kimi-k3`，没有把 vLLM registry 中的关联类名或 FlashKDA 仓库另记为新模型。联网证据把“文档/API 可见”“main 源码有入口”“recipe 可部署”和“stable wheel/生产验收”分成四层：

- [vLLM stable supported-models](https://docs.vllm.ai/en/stable/models/supported_models/) 页面快照为 `738,169` bytes，SHA-256 `5420be70a431de9d6e25ba8322c27c1397239fc11643544b354591b8de80d069`，页面更新时间为 2026-08-29；其中列出 `KimiK3ForConditionalGeneration`、`Kimi-K3` 和 `moonshotai/Kimi-K3`。这证明 stable 文档目录已有 K3 条目，但不证明当前安装的 stable wheel 在目标硬件上成功加载完整权重。
- [vLLM stable K3 API](https://docs.vllm.ai/en/stable/api/vllm/models/kimi_k3/) 快照为 `799,430` bytes，SHA-256 `d17b052921ae5ac2216453edf2034854f45a7d1e46a37ef90eff13e25e5cab5b`，页面更新时间为 2026-09-09；API 页面暴露 `KimiK3ForConditionalGeneration` 和 `KimiK3MTP`。API reference 是类/方法可发现性证据，不是 wheel、权重、kernel 或端到端 serving 通过证据。latest supported-models 页面另存为 `792,208` bytes，SHA-256 `4604b83e2dffa3d6cf154003bf4aa1a29d3b7d7503ed278b556c90f7c29d098f`，不把 latest 与 stable 混写。
- [vLLM main registry.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py) 快照为 `64,391` bytes，SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`；registry 目前包含 `KimiLinearForCausalLM`、`KimiK3ForConditionalGeneration`、`K3DSparkModel` 和 `KimiK3MTPModel`。main 分支类名能证明源码注册入口存在，不能倒推 stable release 的版本号、完整实现覆盖或 speculative/tool-call acceptance。
- [K3 package `__init__.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/kimi_k3/__init__.py) 快照为 `1,444` bytes，SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`；入口按 `current_platform` 分流 NVIDIA 与 ROCm，TPU 不主动加载 GPU 实现。这是硬件隔离设计证据，仍不替代各分支在目标卡上的运行测试。本轮曾尝试读取完整 NVIDIA 源文件，但代理中断导致只保存截断内容，因此不引用其为完整源码证据。
- [K3 vLLM recipe YAML](https://raw.githubusercontent.com/vllm-project/recipes/main/models/moonshotai/Kimi-K3.yaml) 快照为 `19,414` bytes，SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`；`date_updated: 2026-09-10`、`min_vllm_version: 0.29.0`，但 description/performance 字段仍明确为 `Pre-release`，并要求 K3-enabled nightly/image、CUDA 13/cu130 和 NVIDIA r580+ driver。它提供硬件、TP/TEP/DEP/PP、hybrid KV manager、prefix-match unit 128 和 tool-call parser 风险的部署账本，不能写成 stable wheel 或生产 SLO。
- [FlashKDA README](https://github.com/MoonshotAI/FlashKDA) 当前快照仍为 `4,431` bytes、SHA-256 `fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2`；官方 Atom 当前快照为 `8,454` bytes、SHA-256 `2155ff08883c6240b3fb53920a8ecdabfd79d2045f41db35d15e4b9d699f07c5`，最新可见 commit 仍为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`（2026-09-01）。因此本轮没有新的 kernel 版本或 upstream merge 证据。

当前 K3 证据状态应写成：PyPI stable `vLLM 0.29.0` 及其 `v0.29.0` tag 已有 K3 registry/package/model implementation；K3 recipe 仍以 pre-release/nightly image 和特定 CUDA/driver 作为优化部署路径。完整权重加载、目标硬件 runtime、hybrid cache recovery、端到端 profiling、线上 tool-call acceptance 和独立 benchmark 仍未证明。面试回答时，不能把“stable release 有代码”回答成“生产 serving 已完成”。

## 2026-09-22 SGLang main/stable runtime 对照

说明：本节早先的代理失败句子属于首次尝试的历史状态；随后网络恢复并补齐了 commit history，最新结果见本节的 commit history recheck 小节。

本轮继续只沿 Artificial Analysis 与 DataCurve 已发现的 `kimi-k3` 锚点推进；SGLang 仓库只作为该锚点的 serving/runtime 证据，不从源码目录另发现模型。固定的两个文本模型源码快照如下：

| 版本 | 文件 | Git blob | bytes | SHA-256 |
|---|---|---|---:|---|
| `main` | [`kimi_k3.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py) | `383a6f47812bccd1cb91b76814cd0730ff945dd7` | 171,101 | `54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e` |
| `v0.5.20` | [`kimi_k3.py`](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/kimi_k3.py) | `b0ede48c88264d518351a66abf623f1bcf8a730e` | 168,114 | `7a3ef867394a2fd52b3a71a979c053b51e9f2310c35cd4c12ef3aa8e7be172e5` |

`v0.5.20` 的 release tag commit 是 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，发布时间为 `2026-09-18T22:41:33Z`。`main` 是可变分支，本轮固定的是 Contents API 返回的文件内容和 blob，不把它写成不可变 release。当前代理对 GitHub commits API 的重新请求均连接失败，因此没有把未固定的 commit history 当作证据。

### 1. 文本 backbone：KDA、MLA 与两条状态/计算路径

- 两个文本快照都包含 `KimiK3DeltaAttention`、`KimiK3MLAAttention`、`KimiK3DecoderLayer` 和 `KimiK3ForConditionalGeneration`。KDA 路径从配置读取 `short_conv_kernel_size`、`num_heads`、`head_dim`、`A_log` 和 `dt_bias`，使用 `RadixLinearAttention`；K3 的 full-rank gate 由 `q/k/v/g` 投影组合，并在输出侧使用 gated RMSNorm。
- 两个快照都提供 KDA fused decode 的准备路径：权重加载后检查固定的卷积、`A_log`、`dt_bias` 和 head layout，满足 backend 形状时把卷积权重、衰减参数和 output-norm 权重放入 kernel 参数；不满足时保留普通链路。这个“检查后启用、否则回退”的结构是 kernel capability gate，不是性能保证。
- `main` 的 KDA decode 还把 HIP fused path 的 `f_b` 计算延迟给 decode kernel，并把 output-norm gate 通过 attempt-and-verify stash 交给 attention backend；普通 prefill/extend 与 target-verify 仍有不同的 beta/gate 处理。面试中应把 prefill、decode、verify 和 backend fallback 分开回答。
- `KimiK3MLAAttention` 对 full-attention 侧保留 MLA cache 语义，并在 decoder 层与 KDA/AttnRes/MLP 的通信路径组合。SGLang 源码实现因此与 HF reference 中的 `key_cache/value_cache` 对 `conv_states/recurrent_states` 的双状态结论一致，但源码入口本身仍不等于恢复回归通过。

### 2. Latent MoE 与 EP/A2A 通信

- `KimiK3MoE` 将 routed expert 的 hidden size 设为 `routed_expert_hidden_size`，由 `routed_expert_down_proj` 压到 latent width，专家计算后经 `routed_expert_up_proj` 回到 full width；shared experts 仍在原始 hidden width 运行。router 输出保持 FP32，Top-K 使用 grouped routing 和 correction bias。
- 代码枚举了 MegaMoE、DeepEP、Mooncake、Ascend-FuseEP 和 MoRI 等 A2A backend。EP A2A 下每个 rank 处理自己持有的 token rows，MoE 区域不再先做普通 DP gather 或多余 TP reduce；MegaMoE 进一步把 dispatch、grouped GEMM、SiTU 和 combine 放入对称内存路径。这里记录的是实现分支，不是任何 backend 的本地吞吐结果。
- `main` 对 shared expert 的通信抽象比 `v0.5.20` 更通用：通过 `shared_experts_tp_size` 和独立的 shared-expert process group 支持 TP-sharded shared experts，并在 shared branch 中执行 all-gather、TP-sharded MLP 和 reduce-scatter；同时检查 shared intermediate size 能否被 TP size 整除。`v0.5.20` 已有面向 DP-attention/Ascend 兼容路径的 shared-expert gather/reduce-scatter，但条件更专用，不能把两者写成同一覆盖范围。
- 两个快照都包含 single-batch overlap（SBO）思路：shared expert 计算放到 side stream，与 routed A2A/latent tail 重叠，在消费结果前 join。`main` 进一步为 NPU fine-grained dual-stream 和 shared-expert TP communication 留出分支；是否真正命中取决于 device、backend、batch shape、capture mode 和环境开关。
- `main` 还支持把 router gate、shared gate/up 和 latent down projection 在满足 dtype/backend 条件时合并成 decode front weight，并在加载后建立 view；量化或混合 dtype 会保留 unfused path。该实现减少 GEMM/launch 账本，但本轮没有目标硬件 benchmark，不能把代码注释中的潜在收益写成实测加速。

### 3. DP/SP、AttnRes 与通信顺序

`main` 的 K3 decoder 将 attention 输出的 reduce-scatter、MoE token shard、EP dispatch 和 tail all-gather 组织为一条 SP-MoE 路径；普通 attention/AttnRes 的 local rows、delayed `prefix_sum` 和 MLP/MoE 的 global DP buffer 也分别处理。实现重点不是“把所有层都 DP”，而是确保 token shard 在 MoE A2A 中每个 token 只 dispatch 一次，并在尾部恢复后续层所需的行布局。死锁、重复 dispatch、prefix add 先后顺序和 padded extend rows 都是必须单独测试的 runtime 门禁。

### 4. ModelSlim/量化加载和权重映射

相对 `v0.5.20`，`main` 新增 `ModelSlimConfig` 相关的 packed-module mapping，并把 `q_proj/k_proj/v_proj/g_proj` 映射到 fused QKVG module；expert weight loader 也通过正则识别 `experts.<id>.w1/w2/w3` 片段，区分普通参数、packed weight 和 scale。它说明“模型结构正确”还不够，量化 checkpoint 的原始命名、融合模块、分片 loader 和 post-load merge 必须互相一致。源码存在映射不等于 MXFP4/ModelSlim 在本机已经加载或数值正确。

### 5. Vision 文件的稳定性与 serving backend

`kimi_k3_vl.py` 在两个版本中逐字节一致：Git blob `c423027994645e4a839ec47bd26bcc1b2a7cb012`，32,790 bytes，SHA-256 `2924c38f652a6ebff2ef79c49f2f336ba18723ea4b854d3ac14d95292988acdb`。因此本轮不能把视觉文件写成 `main` 独有新增。

这份视觉实现包含 MoonViT3d patch embedding、可插值的 2D learnable position embedding 加 temporal sin/cos、2D RoPE、`cu_seqlens` 变长 segment metadata、SDPA/Triton/FA4/FlashInfer-CuDNN backend 选择，以及在 SM103/B300/GB300 上按形状选择 Triton 或 FA4 的逻辑；CUDA major >= 9 时还可使用 fused complex RoPE。`KimiK3VisionTower` 支持 data-parallel image owner、可选 CUDA graph 和 PatchMerger/后置 RMSNorm projector。它们是源码实现事实，不等于视觉权重加载、图像数值正确性、CUDA graph replay 或端到端多模态 SLO 已通过。

### 6.1 SGLang main commit history recheck

本轮网络恢复后，使用 10.237.126.170:1234 重新读取 [SGLang kimi_k3.py 的 commit history API](https://api.github.com/repos/sgl-project/sglang/commits?path=python/sglang/srt/models/kimi_k3.py&per_page=10)。响应为 51,251 bytes，SHA-256 为 2f7d8402aa104fe9a5b7839e85c9eb56060e98f7aeb4aecb787005f775351e04，覆盖截至 2026-09-22T06:35:12Z 的最近 10 个提交。重新抓取 [当前 main 文本源码](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/models/kimi_k3.py) 后仍为 171,101 bytes、Git blob 383a6f47812bccd1cb91b76814cd0730ff945dd7、SHA-256 为 54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e，与上一轮 Contents API 快照逐字节一致；因此下面记录的是同一源码快照的 upstream 变更历史，不另造一个模型或文件版本。

| commit | 时间 | 提交说明 | 可核验的 K3 影响 |
|---|---|---|---|
| [8ac19cc](https://github.com/sgl-project/sglang/commit/8ac19cc19f8ade51ed203478f17c9a09c706d730) | 2026-09-22 | Fix deferred KDA gate projection and update DCP cookbook | defer_f_b 时把 f_a 交给 fused decode，由 kernel 自己应用 f_b；修正 deferred gate 的所有权边界 |
| [c4d3770](https://github.com/sgl-project/sglang/commit/c4d3770a6850171beb660576cd003e8001a3f75b) | 2026-09-22 | Fix CUDA graph stream explosion | capture 中先发起宽 QKVG projection，再在 side stream 处理小 GEMV；用 pending stream 在同一 capture segment 内 join，避免 replay 时不断扩张 stream |
| [c2c3629](https://github.com/sgl-project/sglang/commit/c2c3629f2dc0d4fa9386e90ea1a63e6ed5d50580) | 2026-09-20 | O(1) expert weight lookup in load_weights | 用 expert tensor-name pattern 和 packed-module mapping 直接定位权重，覆盖 experts.<id>.w1/w2/w3、ModelSlim fused QKVG 和 NPU packed-weight 分支 |
| [72d5c5b](https://github.com/sgl-project/sglang/commit/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a) | 2026-09-09 | Accept fp32 routing weights in fused MoE finalize | fused finalize 同时接受 packed routing id 与未打包的 FP32 routing weights；不满足 push window 时仍回退 in-op finalize |

这几条提交的共同面试点是“融合路径的契约由谁负责”：KDA gate 的投影、CUDA stream 的 fork/join、量化权重的命名映射和 MoE finalize 的输入 dtype 都不能只看 kernel 名称。提交存在和对应测试文件存在可以证明 upstream 正在修正这些边界，但没有给出本环境的数值、吞吐或端到端 acceptance 结果。

并行与后端方向也有明确提交：f4c2563 将 PP prefill、DCP decode、PD disaggregation 与 DSpark worker 组合起来；2d0e94e 在 K3 hunk 中改用 parallel.shared_experts_tp_group，说明 shared-expert process group 必须进入 manifest；8ac39c6 增加 Ascend A5/NPU 的 K3 KDA/MLA cache、MXFP4 MoE、NPU graph 和 DSpark verify 路径；cb32dbc 在 ROCm 小 token 路径把 KDA [q,k,v,g|f_a|b] 合成单 GEMM，并提供布局测试。以上是 source/commit evidence，代码注释中的硬件收益不是本轮独立 benchmark。

commit history 让 main 的证据从“当前源码有哪些分支”推进到“这些分支近期为何变化”：KDA deferred gate 和 graph capture 属于正确性/调度边界，expert lookup 和 ModelSlim mapping 属于加载器复杂度，PP/DCP/DSpark 属于状态与拓扑组合，A5/ROCm 属于后端分支。它们仍然全部属于 mutable upstream source evidence。

### 6. 证据等级与下一步

本轮结论应写成 **SGLang `v0.5.20` stable 已有 K3 文本/视觉实现入口，SGLang `main` 在文本 serving 的 DP/EP/shared-TP/ModelSlim/KDA overlap 路径上继续演进；完整权重、目标硬件、稳定 wheel/环境安装、双状态恢复、视觉正确性、profiling、tool acceptance 和生产 SLO 仍待核验**。不能把 `main` 的新代码升级为 stable 功能，也不能把 `v0.5.20` 的类名升级为目标硬件已验收。

面试回答可以用下面的证据链：

```text
AA/DataCurve Kimi K3 identity
    -> HF revision/config/index
    -> SGLang stable model/vision entry
    -> SGLang main communication/quantization evolution
    -> full-weight load + numerical check
    -> dual-state recovery + target profiling
    -> tool/schema/idempotency/verifier acceptance
```
