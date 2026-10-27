# 第六册第十六章：FP8 KV Cache 资料驱动精修研究记录

日期：2026-10-27  
正文：[第十六章：FP8 KV Cache](../book-06-llm-deployment/chapters/16-fp8-kv-cache.md)

## 研究问题与结论

原稿已有显存账本、K/V 分开量化、page 生命周期、prefix sharing、灰度回滚和任务路由等完整主线。本轮研究聚焦需要从“经验提醒”深入到可解释机制的部分：FP8 两种编码到底差在哪；scale 由谁估计、保存在哪、何时能更新；K/V 为什么可能需要不同粒度；cache 字节减半何时能变成延迟或吞吐收益；一次长上下文精度失败如何分辨量化误差和 kernel 累加误差。

研究后的核心判断：FP8 是 8-bit payload 编码，不是完整的 KV 量化方案。scale 选择、轴/粒度、QK/PV 计算精度、页面布局和具体 attention backend 都参与最终质量和速度。serving 证据尤其说明，相同的“FP8 KV”开关在不同 head dimension、attention 结构、上下文和 kernel 下可能表现迥异；必须报告实际后端与工作负载。

## 来源阅读与可支持结论

| 来源 | 读到的具体内容 | 支持正文的判断 | 限制与使用方式 |
| --- | --- | --- | --- |
| [OCP OFP8 v1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) | §5.1 的 E4M3/E5M2 位字段、正常数/次正规数、特殊值、最大/最小值；§5.2 的饱和和舍入转换 | E4M3 与 E5M2 都是 1 sign bit，但 E4M3 用 4 exponent + 3 mantissa，最大有限值 448、无 Inf；E5M2 用 5 + 2，最大有限值 57,344、支持 Inf/NaN。E4M3 更细的尾数精度换较窄动态范围；E5M2 相反 | 规范是二进制 interchange 格式，不规定 scale 选择、存放或维护，也不规定 KV 应用；不能由 dtype 名推导 runtime 算法。实际 GPU 还需确认是否采用 OCP 变体而非 FNUZ 等编码 |
| [ROCm FP8 precision support](https://rocm.docs.amd.com/en/docs-7.2.3/reference/precision-support.html) 与 [PyTorch dtype table](https://docs.pytorch.org/docs/main/tensor_attributes.html) | 分别列出 OCP 与 FNUZ FP8 类型；PyTorch 将 `fn` / `fnuz` 后缀解释为与特殊值、signed zero 有关的编码属性 | 同样写作 E4M3/E5M2 的类型名并不保证二进制编码完全相同；跨设备保存、转换或调用 kernel 时要核实具体 variant | 这是类型与支持边界的官方说明，不代表对应类型在所有 GPU/kernel 上都可用；应按目标版本复核 |
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | §2 和 §3 说明格式选择、scale 与训练/推理数据流；实验涉及图像、语言模型和 FP8 training/PTQ | FP8 不足以覆盖不同 tensor 的分布时，软件 scale 是独立设计维度；E4M3/E5M2 在精度和范围之间取舍 | 论文不是 KV-specific serving benchmark；本章仅用它解释格式背景，不外推训练结果 |
| [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750) | §3.2 按 Key/Value 分析离群结构；§3.3 设计 key per-channel、value per-token 的 2-bit 非对称量化和 streaming 缓冲；§4 给模型、数据、效率实验 | 代表“先分析两种状态张量的分布，再分别选择量化轴”的研究路线。作者报告其指定 Llama/Falcon/Mistral workload 上峰值总内存 2.6× 减少、batch 最大 4×、吞吐 2.35–3.47× | 这是 2-bit 方法，不等于 FP8 scale 粒度；研究模型、kernel、评测和代际固定。per-channel K 需要跨 token 聚组，与逐 token append 冲突，论文用分组 cache 和尚未满组的高精度 residual 处理 streaming |
| [KVQuant (arXiv v6, 2025-05-28)](https://arxiv.org/abs/2401.18079) | §3 的 per-channel key、pre-RoPE key、per-layer sensitivity-weighted non-uniform datatype、per-vector dense/sparse outlier；§4 长上下文、性能和内存实验 | 代表比 FP8 更低 bit 的研究路线：从 key 通道结构、RoPE 前后差异、层敏感度和 outlier 分别处理。作者报告指定 Llama/Mistral 的 3-bit Wikitext-2/C4 perplexity 增量小于 0.1，并展示长上下文容量实验 | 量化位数、模型、校准和自定义 CUDA kernel 均特定；不能直接把其 perplexity 或 1M/10M context 结果与 vLLM 的 FP8 生产路径比为同一指标 |
| [KV cache serving survey (2026-07-09)](https://arxiv.org/abs/2607.08057) | 按 execution/scheduling、placement/migration、representation/retention 建立系统行为 taxonomy；representation 之下区分 quantization、low-rank 和 structural compression，并把 eviction/reuse 单列 | 用来核对本章边界：FP8 是 representation/quantization；分页、迁移和淘汰是不同但要协同的行为，不应把“更大 cache 容量”与“少保留 token”混成同一种优化 | 综述用于地图，不替代原论文或 serving 实测；新领域持续扩展，引用时标记版本日期 |
| [vLLM Quantized KV Cache docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) | 2026-10-27 latest 文档快照：per-tensor 与 per-attention-head scales；K/V 轴长度对应 KV heads；head scales 需 llm-compressor 校准并限于 FlashAttention backend；提供 skip layers；默认 scale=1 | scale 轴、calibration 和层混合是可观察的 runtime 配置，不是 FP8 格式自动决定。文档也明确 FlashAttention 3 FP8 cache 路径会把 Q 一并量化 | rolling docs；feature/backend/CLI 随 release 改动，使用前固定目标版本。只证明公开接口声明，不证明任意模型/shape 实际可用或速度更快 |
| [vLLM/LLM Compressor FP8 KV calibration example](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_kv_cache/) | one-shot dataset calibration 示例，包含 tensor/attention-head strategy、校准样本和序列长度，以及在 vLLM 侧加载校准 scales | 校准必须覆盖目标数据长度与请求形态；可复用校准 scale 需要与 model/tokenizer/config revision 同步 | 示例给出的是公开 recipe，不证明某校准集能代表目标业务或具有最优泛化 |
| [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) | 2026-04-22 vLLM 技术报告；H100/FlashAttention-3、B200/FlashInfer；长上下文精度、ITL slope、break-even、hybrid sliding-window、head_dim=256 prefill 和并发吞吐 | 具体案例说明“量化误差”与“kernel accumulation error”必须区分：H100/FA3 一个 128k needle case 中 BF16 91%、原 FP8 路径 13%；双级累加后 FP8 恢复至 89%，但寄存器压力使部分 prefill 变慢。Llama-3.1-8B H100 decode ITL slope 降至 BF16 的 54%，但 break-even 约 7k；gpt-oss-20b 小滑窗层全量化 break-even 22k，跳过滑窗层后约 7.7k。并发 8、150 个约 20k input/2k output 请求时，Llama-3.1-8B 输出吞吐 450.3→517.5 tok/s（+14.9%） | 全是作者公开 workload/backend/版本的工程结果；该文明确部分表格用未校准 scale=1.0，不能当作所有 release 或 workload 的预测。博客同时说明 head_dim=256 的 H100 prefill 仍可能回退，以及 B200/FlashInfer 有不同路径 |
| [TensorRT-LLM FP8 quantization](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html) | 当前官方性能指南中 weight/model quant 与 `kv_cache_quant_algo=FP8` / `--kv_cache_dtype fp8` 分开配置；文档提醒单独量化 KV 也需质量验证 | 权重精度、Q/K/V 计算路径、KV 存储精度应分别记录；支持接口不等于默认启用 | 页面 benchmark 表格上下文不足以支撑跨 workload 数字比较，本轮只把它用作接口层证据 |
| [Transformer Engine 2.20.2 FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) | current scaling recipe 先求 amax，再缩放并转成 FP8；文档说明需两次读取 tensor | 动态 scale 的统计与转换存在额外数据访问；用于解释一般 tensor quantization 的成本组成 | 这是训练库的量化 recipe，不是 KV-cache manager 或在线 serving benchmark；不能据此声称具体 KV runtime 使用相同路径 |

## 机制与可复算数字

- 对普通 MHA/GQA/MQA attention，payload 下界为 `M_KV = 2 L B T H_kv D_h b`。示例取 `L=64, B=1, T=131072, H_kv=8, D_h=128`：BF16 是 32 GiB，FP8 payload 是 16 GiB；若仅作教学假设额外加 8% scale/metadata/alignment，则是 17.28 GiB，不代表 runtime 实际增量。
- `T=1,048,576`、其余不变时 BF16 payload 是 256 GiB，FP8 payload 是 128 GiB；实际容量还受 MLA/混合 attention、分页粒度、scale、副本、workspace 和 allocator 余量影响。
- OCP FP8：E4M3 最大有限 magnitude 448，最小 normal `2^-6`、最小 subnormal `2^-9`；E5M2 最大有限 magnitude 57,344，最小 normal `2^-14`、最小 subnormal `2^-16`。E4M3 没有 Inf 编码而 E5M2 有；scale 应被视为独立层。
- 可读的 E4M3 scale 算例：输入组最大绝对值为 500，若按最大码值缩放，`s=500/448≈1.1161`，归一化最大值为 448；组内 `x=1` 变成约 0.896，E4M3 最近码值为 0.875，反量化约 0.9766。这个例子说明 scale 会影响同组内值的重构；不代表所有 runtime 都采用 absmax/单组 scale。
- K 误差通过 `q·k` 进入 logits 并经 softmax 非线性改变权重；两位置 logits `[1,0]` 的权重约 `[0.731,0.269]`，若第一项因 K 误差变为 `[0.7,0]`，权重变为 `[0.668,0.332]`。若 `v=[1,0]`，输出从约 0.731 变为 0.668。V 上的误差则直接按 attention 权重加权进入输出。该例是机制推演，不是模型级质量预测。
- 如果一个 page/quantization group 共用 scale，已有 payload `q` 的解释是 `x_hat=sD(q)`。scale 从 0.1 改为 0.2 而不重编码 payload，会把 `q=10` 的重构从 1 改成 2；因此更新组 scale 必须同步更新/重编码组内全部值，或者使用分组/静态尺度方案。共享 page 的所有 reader 还必须拿到匹配的 scale revision。

## 原稿知识点去向

| 原稿区域 | 修订处理 |
| --- | --- |
| §1 量化 KV 与权重量化的区别，容量/带宽/调度收益 | 保留；补充收益成立条件以及论文 survey 的 representation 与 execution 层次区分 |
| §2 显存公式和 128k/1M 数量级 | 保留原数；核对 MHA/GQA/MQA 假设，补实际账本项目与公式适用边界 |
| §3 FP8、scale 粒度、static/dynamic/calibration | 深化 E4M3/E5M2 编码、特殊值、范围和可复算 scale；用实际 vLLM scale axes/校准接口说明格式与 runtime 分离 |
| §4 K/V 不对称、质量回归 | 保留四组消融；加入 softmax 数值推演，比较 KIVI/KVQuant 的具体研究问题、方法、证据与边界 |
| §5 cache 生命周期、prefix、swap、speculative KV | 保留完整状态链；补充 scale 在 append/page share/restore 中的一致性约束 |
| §6 分页和 FP8 组合 | 保留；补充真实 vLLM 案例，说明 paged allocator/量化/attention backend 不能独立推断速度 |
| §7 容量、路由、单位成功成本 | 保留；将人为设定的质量数据明确标成门限演示，不作为产品报告 |
| §8 长上下文、K/V 消融、敏感层 | 保留；加入 128k 数值失败的 kernel 归因、双级累加修复及不同 workload 的 ITL/break-even 对照 |
| §9 manifest、灰度、回滚 | 保留；丰富 scale 轴/形状、转换/饱和规则和 backend version 记录字段 |
| §10 教学代码 | 保留容量与任务 gate 示例；补输入校验，并在正文明确 overhead/quality 是合成值 |
| §11—13 排查、来源、总结 | 保留；将来源注释从“入口链接”升级为对应研究解决的问题和证据限制；修订低精度收益总结 |

## 资料同步状态

- `EXERCISES.md`：已增加 E4M3/E5M2 范围、scale 计算、K/V softmax 扰动和 vLLM serving case 解释练习。
- `INTERVIEW_BANK.md`：已增加 KV payload 与 attention 计算精度、K/V 非对称量化和 FP8 cache 可能变慢/精度骤降等问答。
- `PAPERS.md`：已补 2026 KV-cache survey、OCP FP8 spec、vLLM FP8 serving report，并扩充 KIVI/KVQuant 等相关研究条目。
- `GLOSSARY_EN_ZH.md`：已核对并更新 E4M3/E5M2、FP8 KV Cache，增加 OFP8/OCP FP8 和 two-level accumulation。
- 本章资料段已加入 ROCm/PyTorch OCP 与 FNUZ 类型边界，以及 Transformer Engine current scaling recipe；这些来源仅支撑格式/接口/一般量化流程，不作为 KV runtime 性能证据。

## 验收和边界

- `source_status`：阅读 OCP FP8 spec、FP8 格式原论文、ROCm/PyTorch FP8 类型说明、KIVI、KVQuant、2026 KV survey、vLLM 最新 KV 量化文档、LLM Compressor 校准 recipe、vLLM 2026-04 工程验证、TensorRT-LLM FP8 指南和 Transformer Engine 2.20.2 current scaling 文档。Rolling docs 记录访问日期并按实际部署 release 重核。
- `implementation_status`：未在目标 GPU、目标模型、vLLM/TensorRT engine 或生产 trace 实测；文中所有性能和质量数字均绑定各来源明确给出的模型、task、hardware、kernel、version 和配置。
- `formula_status`：容量、scale 和 attention score 案例为理论手算；教学脚本只验证账本/质量门禁逻辑，不编码 IEEE/OCP FP8、attention kernel 或生产 cache manager。
- `validation_status`：章节 §1—§13 编号连续；代码 fence 成对，本章和研究记录的本地 Markdown 链接目标存在；示例标准输出与正文 fenced output 完全匹配；18 类非法输入均按预期抛出 `ValueError`。公式、输入边界和链接均完成静态核对。
- `open_risks`：该章是一般 MHA/GQA/MQA 的 FP8 KV 章节，MLA/混合注意力、滑窗、state-space cache 和模型专用 paged layout 需按目标架构另建公式。OCP/FNUZ 编码变体、scale 配置、head/block grain 和后端 kernel 均随 release/hardware 变化。
