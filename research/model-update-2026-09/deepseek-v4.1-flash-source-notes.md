# DeepSeek-V4.1-Flash 官方资料摘记

核验日期：2026-09-14。本文把 DeepSeek 官方发布页、Hugging Face 模型卡、模型配置和随仓库发布的编码/评测说明分开记录。技术报告 PDF 已下载、校验哈希并按页提取 51 页正文；报告中的内部实验和部署数字仍按发布方自报记录，不等同于本项目复现。

## 来源与快照

| 来源 | 作用 | 本轮状态 |
|---|---|---|
| [DeepSeek-V4.1-Flash 发布页](https://api-docs.deepseek.com/news/news260910) | 发布日期、API 模型名、兼容路由、官方产品描述 | 页面快照返回 HTTP 200；页面导航标注 `2026/09/10`，快照 SHA-256 为 `420cbb7b5e8e97632fa45cb49cd2b5f22b57c8f9e125d1c34a22bd67bbc33705` |
| [Hugging Face 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 架构、训练、评测、协议和开源入口 | revision `dba1be0a40aa45a94ad051997016db3960a90277`；README 快照 SHA-256 为 `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251` |
| [模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) | 可机器读取的结构字段 | `DeepseekV41ForCausalLM`、`deepseek_v41`；配置快照 SHA-256 为 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| [技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | 架构、训练基础设施、推理系统、后训练和评测细节 | 51 页正文已逐页提取；SHA-256 为 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d` |
| [encoding README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md) | prompt 格式、工具标签和 reasoning effort | 已读取；快照 SHA-256 为 `a2f0fc3baea318c9cfbceca68cbfe50d37cf7da6605ace887f33148bcff7e3ae` |
| [evaluation README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/evaluation/README.md) | DeepSWE 复现实验的 harness 说明 | 已读取；明确区分 `mini-swe-agent` 与 `dsh-minimal` |

页面哈希只用于识别本轮资料快照，不代表页面或模型权重永久不变。模型卡 revision 是权重/仓库的版本锚点，不能用首页当前内容替代。

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

## 待核验与后续动作

- 技术报告正文已逐页读取并纳入上面的页码记录；报告没有公开完整 kernel source、所有权重/参数分片、线上接受率或目标硬件上的独立 profiling。
- 公开资料没有给出完整参数分片与服务端并发账本；552B backbone、196B Engram 和权重仓库 tensor dtype 不能简单相加后当作单卡显存。
- CED、CSA2、SWA Bounded Replay、Engram、mHC 和 DSpark 的生产级 kernel、容错、接受率和硬件依赖需要按 revision/后端复测。
- API 价格图片、账户限流、流式错误码、实际 alias 路由和多模态 token 计费需要通过实时 API 文档或不产生费用的接口探测继续核验。

## 书系映射

- 第二十一册第 81 章：CED、SWA Bounded Replay、CSA2、Hierarchical Sparse Indexer、FP4 KV、MoE、Engram、DSpark 和原生多模态。
- 第四册第 19 章：作为架构百科条目，强调模型卡证据与推理部署边界。
- 第五册：45T 多模态预训练、SFT -> RL -> OPD 与 Agent 数据管线。
- 第六册：1M context、异构 KV/cache tier、HBM/SSD/重算和 numeric reasoning effort 预算。
- 第七册、第十七册和第二十册：官方 Agent benchmark 的 harness、verifier、工具权限、恢复和复现账本。
