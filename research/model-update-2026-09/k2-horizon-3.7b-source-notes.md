# K2 Horizon 3.7B：dense 长上下文模型的证据摘记

核验日期：2026-09-22。本文只记录已经由 Artificial Analysis 发现的 `K2 Horizon 3.7B` 及其官方周边资料。DataCurve DeepSWE 没有精确的 K2 Horizon 3.7B 行，因此不迁移其他 K2 或其他模型的 Agent 分数。

## 1. 候选身份和证据边界

| 层级 | 已核验事实 |
|---|---|
| 发现入口 | [Artificial Analysis K2 Horizon 3.7B](https://artificialanalysis.ai/models/k2-horizon-3-7b)，slug `k2-horizon-3-7b` |
| 榜单日期 | 第三方字段 `releaseDate=2026-09-03`；不能当作 IFM 官方发布日期 |
| AA 规格字段 | reasoning、open weights、3.7B、524,288 context、Apache 2.0；AA Index `15.6103951330976` |
| DataCurve | `/tmp/ds-current-1234.html` 当前快照没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 或 K2 Horizon 行 |
| 官方模型 | [IFM/K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B) |
| 当前 revision | `6360f705b2e57d542959e6a2e67ebeb95dae0373`，`lastModified=2026-09-21T01:40:06Z` |
| 当前状态 | **AA 单榜资料级闭环**；有官方模型卡、配置、迁移和 serving 资料，但没有精确 DataCurve Agent 结果或独立复现 |

三条代理取得的 AA 详情页逐字节一致：

- `/tmp/aa-k2-horizon-3-7b-1234-20260922.html`
- `/tmp/aa-k2-horizon-3-7b-7890-20260922.html`
- `/tmp/aa-k2-horizon-3-7b-8098-20260922.html`
- 三份均为 `3,756,408` bytes，SHA-256 `72b4f94c55add582b0399333552e92b7b4aebd2f493c26830c00e09e58afc39d`

DataCurve 快照为 `/tmp/ds-current-1234.html`，`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。页面没有精确 K2 Horizon 3.7B 配置，因而不把 K2 36B、K2.7 Code、K3 或其他模型的 Pass@1、成本和 Agent steps 写到 3.7B 名下。

## 2. 当前实现和固定 artifact

当前配置的模型类型是 `K2HorizonForCausalLM`。固定配置/实现字段为：

| 字段 | 3.7B 当前 revision |
|---|---:|
| decoder layers | 36 |
| hidden size | 2560 |
| query / KV heads | 32 / 8 |
| head dimension | 128 |
| intermediate size | 10240 |
| vocabulary | 250,624 |
| max position | 524,288 |
| RoPE | default，`rope_theta=10,000,000` |
| sliding window | `null` |
| routed experts | `num_experts=0` |
| MoVA experts | `mova_num_experts=0` |
| dtype | BF16 |

所有层都是 dense；没有证据支持把 3.7B 写成 MoE、MoVA、sliding-window 或带 value router 的模型。GQA 的 32Q/8KV 仍然会影响 KV cache，但它不是专家稀疏化。

固定文件的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `config.json` | `a98a4dc771aadcbe03a390d825723a42eaee2682758b29b7d44341d4f33d8ab4` |
| `configuration_k2_horizon.py` | `5c2f993c1053d9462ebea6dea416c897fddfbb4a5edd904e486936b20d4badc5` |
| `modeling_k2_horizon.py` | `fb09e010956bd51cfa7d4055b4381cff34c9e06164066b49e3546f38b2e6242f` |
| `model.safetensors.index.json` | `d3b5c4c42227590b76382a9bcc54f868a725bc3c46bea5cdc82e218494759599` |

权重索引显示 36 个 shard、327 个 tensors、`total_size=10,116,510,720` bytes。按 BF16 的两字节元素计算约为 5.06B 个存储参数；这支持 vLLM 页面所写的 5.06B dense 存储口径，但不把 AA 的 `3.7B` 名称、核心参数和含 embedding 参数混成同一个未经说明的数字。

## 3. 参数账本和文档冲突

不同 artifact 的参数口径需要保留：

1. AA 名称是 `K2 Horizon 3.7B`，属于第三方目录身份和规模标签。
2. vLLM recipe 页面写作 `5.06B`、`DENSE`、`512K ctx`；权重索引的 BF16 字节数与“含 embedding 的存储参数”口径相符。
3. 同一仓库较旧的 `APPENDIX.md` 仍写 `3.78B core / 5.06B including embeddings`，并写 `XllmForCausalLM`、FP32。3.78B core 与 5.06B including embeddings 可以解释为 core/embedding 账本不同，但旧类名和 FP32 与当前 `K2HorizonForCausalLM`、BF16 config/migration 不一致。

因此当前结论是：**以当前固定 config、migration manifest 和权重 index 作为运行时 artifact 事实；把 APPENDIX 的 Xllm/FP32 字段标为旧 revision 或残留文档，不能覆盖当前 config。** 参数总量仍按 core、embedding、dtype 和 revision 分栏，不用一个“3.7B/5.06B”数字回答所有问题。

## 4. 与 K2 Horizon MoVA 36B/A4B 的结构对照

| 维度 | 3.7B 当前实现 | 36B/A4B 当前已核验实现 |
|---|---|---|
| 总体路径 | 36 层 dense decoder | 48 层，前 3 层 dense，后 45 层 sparse |
| 参数标签 | AA 3.7B；权重存储约 5.06B BF16 参数 | 约 36B total、约 4B/token active proxy |
| value 路径 | 普通 dense value projection；`mova_num_experts=0` | 64 value experts，top-4 MoVA |
| FFN | dense，intermediate size 10240 | 100 routed FFN experts，top-8，另有 1 shared expert |
| attention | 32Q/8KV GQA，head dim 128 | 32Q/8KV GQA，head dim 128，并有 attention gate |
| 上下文 | 524,288，`sliding_window=null` | 524,288，`sliding_window=null` |
| serving 重点 | 单路 dense GEMM、KV、reasoning/tool parser | MoVA/FFN 双路 dispatch、EP/TP、router metadata、通信和 KV |

两者都使用 GQA 和 512K 配置，但这不意味着 3.7B 继承 36B 的 MoVA、MoE 或 active-parameter 口径。3.7B 是很有价值的 dense 对照：可以在相同 hidden size、head 布局和长上下文目标下，隔离“专家 dispatch/通信”与“dense GEMM/KV”之间的差异。

## 5. 分阶段长上下文训练

官方模型卡给出的 token 是阶段增量，后续阶段从前一阶段 checkpoint 继续，不能直接相加后称作一份独立数据集：

| 阶段 | 额外 token | 序列长度 |
|---|---:|---:|
| Pretraining | 22.9T | 8K |
| Midtraining 1 | 1.1T | 32K |
| Midtraining 2 | 498B | 128K |
| Midtraining 3 | 110B | 512K |
| Midtraining 4 | 199B | 512K |
| SFT Phase 1 | 199B | 512K |
| SFT Phase 2 | 50B | 512K |

RL 阶段按方向分出 Math expert、Code expert 和 STEM-Code expert；模型卡还描述了 RL merge：self-attention 使用 ISO merge，其他权重使用 RAM。这个描述能支持“领域 RL 分支再合并”的训练主线，但不等于公开了完整 loss、采样器、权重系数、教师数量或分布式实现。

公开中间 checkpoint 的价值在于可以研究能力随 midtraining、RL 和 SFT 阶段变化，而不是只比较最终模型。面试时应把“阶段增量 token”“checkpoint 继承关系”和“RL expert merge 的权重规则”分别问清楚。

## 6. Migration manifest：artifact 迁移不是重新训练

固定迁移清单 `/tmp/k2-migration-fixed-20260922.out` 给出：

- source model type：`k2_aurora`
- target model type：`k2_horizon`
- target architecture：`K2HorizonForCausalLM`
- weight mode：`copy`
- `weights_reencoded=false`
- dtype：BF16
- 36 shards、327 tensors
- 文件 SHA-256：`dd8e209002e7d69163e2dfde3e3490b8a6b1bf2aa70361088307079efa45b515`

这说明迁移工具如何把 artifact 重新登记到目标模型实现，不能被写成“训练从 K2 Aurora 重新开始”，也不能单独证明完整模型行为等价。迁移后的 tokenizer、chat template、reasoning 参数、tool parser、revision 和输出协议仍需单独回归。

## 7. Serving 资料和发布方结果

### vLLM

官方 [vLLM recipe](https://recipes.vllm.ai/IFM/K2-Horizon-3.7B) 更新时间为 `2026-09-02`，描述 5.06B dense、512K、H200 recipe，并使用 `k2_horizon` reasoning/tool parser。它是部署入口和参数契约，不是本机性能复现。

### SGLang

官方 Markdown/PR `#37654` 的 3.7B 配方使用 H200、TP1、BF16 和 FlashAttention-3；文档固定 revision 为 `c177771836a4c460743c00002c22483f6f18d1eb`。该 revision 在本轮通过 HF raw/API 当前返回 404，结论是“部署文档引用的旧 revision 当前不可解析”，不是模型不存在。

发布方给出的结果应单独带 recipe 标签：

| 配置 | TTFT | TPOT | 吞吐/结果 |
|---|---:|---:|---:|
| 8,192 input / 1,024 output，concurrency 1 | 158.71 ms | 5.10 ms | 1,712.15 tok/s/GPU |
| 同一长度，concurrency 64 | 5,174.71 ms | 28.83 ms | 16,998.94 tok/s/GPU |
| GSM8K | - | - | 92.00%，两次记录 92.12% / 91.89% |

这些是发布方 recipe 结果，不是本机实测；不能与 AA Index 或 DataCurve Pass@1 拼成统一排名。迁移到其他 GPU、backend、batch 或 parser 后，必须重新测 TTFT、TPOT、输出一致性、峰值显存和工具调用成功率。

## 8. 关联资料和未确认项

- [K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) 的 MOPD/领域专家合并只作为同系列训练资料；不能把 0.9B 的流程自动写成 3.7B recipe。
- [K2-Horizon-7B-Uno](https://huggingface.co/IFM/K2-Horizon-7B-Uno) 和 [Uno 论文](https://arxiv.org/abs/2609.04010) 是 K2 周边的 diffusion/LoRA speculative decoding 技术，不是 3.7B 榜单条目，也不是 3.7B 的架构证明。
- IFM blog 当前被 Cloudflare challenge 拦截；不能声称已经读取正文。
- xLLM raw README 只有 `# xllm`，不足以作为 3.7B 训练实现或生产 kernel 证据。
- 完整训练 recipe、完整训练代码、生产 kernel、目标硬件 profiling、线上 tool acceptance、独立 benchmark 和精确 DataCurve Agent 结果仍待核验。

## 9. 面试复述模板

> K2 Horizon 3.7B 是 K2 Horizon 家族中的 dense 长上下文对照模型。当前固定 revision 使用 `K2HorizonForCausalLM`，36 层、2560 hidden、32Q/8KV GQA、128 head dim、524,288 position，所有层的 MoE/MoVA expert 数都为 0。它和 K2 Horizon MoVA 36B/A4B 共享 GQA 与 512K 目标，但后者在后 45 层加入 value top-4 和 FFN top-8 路由；因此比较两者时，要把 dense GEMM、KV、dispatch、通信和 parser/serving 版本分开记账。3.7B 的 3.78B core/5.06B including embeddings 旧文档口径与当前 BF16/K2Horizon config 存在版本冲突，应按 revision 解释，不能混写。
