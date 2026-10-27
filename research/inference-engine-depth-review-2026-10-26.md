# 推理引擎章节资料驱动精修记录（2026-10-26）

## 状态与研究问题

- `status`：资料驱动精修和声明范围内验收完成。
- `chapter`：`book-06-llm-deployment/chapters/05-推理引擎.md`，§5.1—§5.20。
- `review_basis`：当前 `WRITING_SPEC.md` 的知识点深度、代表方案比较、教学代码验证、证据范围、旧知识点去向与配套同步要求。
- `date_boundary`：截至 2026-10-26 能访问的论文和官方文档快照。论文、`main`、`stable`、`latest` 文档并非同一运行版本；部署判断必须再固定 release 和硬件。

本章要回答的不是“有哪些推理框架”，而是：一个在线 LLM runtime 怎样从模型执行代码扩展成请求生命周期管理系统；每类系统机制分别解决了什么瓶颈；论文结果在何种实验边界内成立；框架的能力、稳定性与实际性能如何通过 workload 回放验证。

研究将问题拆成五条链：

1. **请求生命周期**：一次 `generate()` 与在线服务之间还缺哪些排队、并发、取消、cache 释放和事件语义？
2. **执行与调度**：iteration-level scheduling、paged KV、chunked prefill、程序级共享状态分别改善什么瓶颈？
3. **模型和协议兼容**：权重加载成功为何不等于 template、stop、tool event、长上下文和质量都正确？
4. **benchmark 设计**：闭环/开环流量、冷热 cache、不同 TPOT/ITL 分母怎样改变结论？
5. **框架状态**：截至查阅日期，各项目的执行后端、服务能力和维护状态有哪些会影响新部署/存量迁移的变化？

## 代表性研究：问题、方案、证据和局限

| 研究 | 原问题与核心方法 | 作者报告的实验结果 | 主要边界与后续问题 |
| --- | --- | --- | --- |
| Orca（OSDI 2022） | 静态请求批次被最长生成拖住，新请求要等整组结束。以 iteration 为调度边界重新安排请求，并提出 selective batching，只对适合合批的运算做 batching。 | GPT-3 175B 的论文设置中，相同 latency 下相对 FasterTransformer 报告 36.9× throughput。 | 特定模型、硬件/部署和 2022 年 baseline；它主要解决请求批次生命周期与 Transformer batching，并未独立解决动态 KV 分配、碎片和 prefix 共享。不能拿这个倍率与其它论文直接排名。 |
| PagedAttention / vLLM（SOSP 2023） | KV 长度随请求动态增长，连续分配容易碎片化或预留浪费；将 KV 划分为 token blocks，用 block table 间接寻址并允许共享。 | 论文在其模型和 workload 下报告：相同 latency 时吞吐较 FasterTransformer/Orca 提高 2–4×。 | block 化会要求 attention kernel 支持分页布局；cache 容量增长与 prefill 对 decode 的干扰是不同问题。具体收益取决于序列长度、模型、解码策略和 runtime 实现。 |
| Sarathi-Serve（2024） | 大 prefill iteration 会让已有 decode 请求长时间等不到下一个 token；将 prefill 切成 chunks，并安排 stall-free schedule，使新请求加入时不暂停已有 decode。 | 作者报告 Mistral-7B 单 A100 serving capacity 为 vLLM 的 2.6×，Yi-34B 双 A100 最高 3.7×；论文还报告 Falcon-180B pipeline-parallel 路径最高 5.6×。 | 容量结果受模型、GPU 数、chunk size、pipeline、尾延迟 SLO 和 baseline 版本约束。它说明 chunk 与调度协同的价值，不证明所有 continuous batching 都有同等收益。 |
| SGLang（2023/2024） | 多次生成、并行分支、Agent/检索程序反复计算共享前缀，也需要更快的结构化输出；提供程序前端和 runtime，以 RadixAttention 复用 KV、compressed FSM 加速受约束解码。 | 原论文在多类模型和 Agent、推理、few-shot、JSON、RAG、多轮对话 workload 上报告最高 6.4× throughput。 | 收益依赖程序形状、prefix 重复率、runtime cache hit 与对比系统；论文上限不代表当前通用请求都加速。还需处理 eviction、session 生命周期、格式解码及 end-to-end 任务质量。 |
| vAttention（ASPLOS 2025） | PagedAttention 的非连续虚拟布局要求 kernel 读分页 KV；使用 CUDA VMM 将虚拟地址保留与物理页映射解耦，保留虚拟连续布局并处理物理碎片。 | 作者报告相对其 FlashAttention/FlashInfer paged kernels 配置最高 1.23× serving throughput。 | 依赖 CUDA VMM 和指定 kernel/hardware，对照范围较窄；它替换 KV 内存管理路径，不是请求 scheduler，也不能把论文倍率外推至所有引擎。 |

这些数字各自回答“在作者自己的实验中，相对某 baseline、在某个指标上改善多少”，而非共同排行榜。它们跨越不同年代、模型、硬件、输入/输出分布、SLO、实现和指标；正文明确禁止裸倍率横向排名。

## 当前工程资料核对

| 来源 | 本轮确认的事实 | 正文中的使用范围 |
| --- | --- | --- |
| [Transformers v5.6.0 serving guide](https://huggingface.co/docs/transformers/v5.6.0/serve-cli/serving) | `transformers serve` 文档列出 continuous batching 与 OpenAI SDK 兼容 API，并定位于评估、实验、中等负载本地/自托管；大型生产部署建议看 vLLM/SGLang。 | 区分 `model.generate()` reference path 与服务 CLI，不把功能存在写成高并发性能保证。 |
| [vLLM V1 architecture](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md)、[系统剖析](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) | 架构文档描述 API server、engine core、GPU worker 的边界；V1 调度和 KV 管理跨进程/执行组件协作。 | 用来解释调度、服务接口、GPU 前向不是同一层；`main` 和博客不固定某 release 契约。 |
| [vLLM Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) | `request-rate`、`burstiness`、`max-concurrency` 分别控制到达率、到达波动和并发上限；官方文档包含 prefix caching 重复负载场景。 | 支持基准测试必须明确 arrival model、并发和 cache 状态；stable 文档要绑定版本。 |
| [SGLang session-aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache) | 应用传入的 `session_id` 标记可复用 KV 的会话引用，不会自动拼接上下文；会话引用是 eviction preference，并非固定驻留；会话结束可关闭引用。 | 强化 cache/session 生命周期边界；明确 session ID 不等于对话存储、鉴权或 hard pin。 |
| [SGLang Radix Cache eviction](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy) | 当前页面区分 LRU、LFU、SLRU、priority、TLRU 等淘汰规则及可淘汰 leaf/引用状态；策略排序不能保证节点不被回收。 | 支持 cache hit、驻留、淘汰、公平/优先级与会话语义都要共同测量。 |
| [TensorRT-LLM migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)、[PyTorch architecture](https://nvidia.github.io/TensorRT-LLM/latest/torch/arch_overview.html) | 当前文档称 TensorRT engine backend 已移除，PyTorch 为唯一执行 backend；旧 `trtllm-build`/逐模型 checkpoint conversion 流程不再是当前默认部署路径。PyExecutor 描述 scheduler、model engine 和 decoder。 | 明确标注 date/release-sensitive 的架构变更；保留 TensorRT-LLM 的 NVIDIA kernel/优化角色，不把历史 build 流程写为 current。 |
| [LMDeploy installation](https://lmdeploy.readthedocs.io/en/stable/get_started/installation.html)、[supported models](https://lmdeploy.readthedocs.io/en/stable/supported_models/supported_models.html) | 文档将 TurboMind 与 PyTorch 列为不同 engine 路径，模型支持按后端和平台区分。 | 选型按模型、backend、device、dtype/量化组合检查，不能只看总模型表。 |
| [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | server 功能列出并发 parallel decoding、continuous batching、结构化输出及多 API 能力。 | 纠正“端侧/本地引擎没有 serving 并发”的笼统表述；性能仍绑定设备和请求负载。 |
| [Hugging Face TGI status](https://huggingface.co/docs/text-generation-inference/index) | 官方文档称 TGI 自 2025-12-11 起处于 maintenance mode，主要接受小型 bug fix、文档和轻量维护。 | 新部署选型和存量迁移分开处理，不推导为既有部署立即失效。 |

## 原章节知识点覆盖图

| 原节 | 保留并深化的内容 |
| --- | --- |
| §5.1—§5.4 | 模型 artifact / runtime / 平台边界、朴素 generate 的不足、请求生命周期、workload 坐标系。保留旧有选型前提，增加从单次执行到服务生命周期之间缺失的能力。 |
| §5.5—§5.11 | 原有 Transformers、vLLM、SGLang、TensorRT-LLM、LMDeploy、llama.cpp、TGI 逐框架介绍保留；更新为版本定位、执行路径、机制、适用场景和必须核验的组合。 |
| §5.12 | 保留引擎能力矩阵；新增 Orca、PagedAttention/vLLM、Sarathi-Serve、SGLang、vAttention 的问题—方法—作者证据—限制对照，并单列当前框架路线矩阵。 |
| §5.13 | 保留模型兼容与接入；细化 weight、kernel、cache、protocol、quality 五层 contract 及最小回放和失败定位。 |
| §5.14 | 保留性能指标与压测框架；增开环/闭环 arrival、burstiness、prefix cache 冷热、goodput 定义和 TPOT/ITL 口径。 |
| §5.15 | 保留零依赖 toy 选型审计；把 continuous batching 标为能力布尔项，剥离 toy 数值与现实 benchmark 含义，验证无候选工作负载路径。 |
| §5.16—§5.18 | 保留选型步骤、常见误解、模型接入和引擎迁移案例；把回放、目标 workload benchmark、维护成本、灰度与 rollback 放进连续决策链。 |
| §5.19—§5.20 | 扩充论文与官方文档资料，并保留结论边界。 |

本轮没有删除原有主要框架、性能比较、协议验证、选型案例和误区等问题。低价值的泛化能力宣称改为有条件的框架定位；关键机制、代表研究和新的版本边界均有正文或参考入口。

## 公式、代码与链接验收

- §5.14 新增 SLO goodput：以满足请求完成、质量门槛和指定 latency SLO 的请求数除以 wall time。正文要求显式定义 `L_i` 是 TTFT、TPOT、E2E 还是业务 deadline；因此它不会与未限定的 completed-request QPS 混用。
- §5.15 的公式和评分全部是教学模型，固定示例只用来解释候选筛选，不表示真实引擎排名。
- 从 Markdown 提取 §5.15 的完整 Python demo 并运行。输出与 fenced `text` 完全一致；把 throughput 门槛设成不可达后，候选集为空且筛选逻辑正常返回空结果。该验证不覆盖真实 runtime、模型、GPU 和 benchmark。
- 外部链接已按原论文、官方文档和项目仓库逐类核查；动态页面以 2026-10-26 访问内容为准。论文数字只按原始研究设置描述。章节没有新增相对 Markdown 文件引用。
- 静态检查小节顺序、代码/文本/数学围栏配对、公式定界符、目录映射、表格列数和本章引入的术语入口；`git diff --check` 通过。

## 配套文件与未验证边界

- `PAPERS.md` 补入 vLLM、SGLang 会话/eviction、TensorRT-LLM、LMDeploy、llama.cpp、Transformers 和 TGI 的当前工程入口。
- `GLOSSARY_EN_ZH.md` 补 Iteration-Level Scheduling、Selective Batching、Radix Cache、Session-Aware Radix Cache、Serving Goodput、PyExecutor、TurboMind Engine、Transformers Serve、Maintenance Mode 等术语。
- `INTERVIEW_BANK.md` 增补框架版本变化和 serving benchmark 设计问题；`EXERCISES.md` 增加代表研究比较和开环/闭环 benchmark 设计练习。
- 第六册目录原有第五章映射存在，无需另改。
- 未启动任何真实模型 server，也没有目标 GPU/集群/线上 trace；没有测吞吐、容量、成本、SLO 或多租户隔离。所有论文数据只代表作者评测；官方文档只证明文档当前声明的实现和状态，不构成目标环境性能或生产就绪证明。
