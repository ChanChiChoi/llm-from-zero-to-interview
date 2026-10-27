# 第六册第一章：部署总览资料驱动精修记录

## 研究范围与问题图谱

- 日期：2026-10-26。
- 本章起点：`progress_v2.md` 将第六册第一章列为下一目标。本轮从该章继续，没有重新精修此前已登记完成的章节。
- 主文件：[`book-06-llm-deployment/chapters/01-部署总览.md`](../book-06-llm-deployment/chapters/01-部署总览.md)。
- 研究目的：把原章的部署组件总览推进到“读者能从请求和负载推导配置、从观测信号定位瓶颈，并解释代表方案解决的约束”的程度。总览章仍负责建立决策地图；KV、调度、引擎、多 GPU 和生产系统的细节由后续章节主讲。

核心问题图谱：

1. checkpoint 为什么不等于服务：服务还依赖 tokenizer、chat template、runtime、API、资源、策略、观测和生命周期。
2. 一次请求如何变成 TTFT、TPOT/ITL、E2E、队列、prefill、decode 和 KV 观测；这些指标的采样单位与分母是什么。
3. 部署容量如何从工作负载、到达过程和请求级 SLO 推导；为什么峰值 tokens/s 或平均请求长度不能单独证明容量。
4. Continuous/iteration scheduling、chunked prefill 和 Prefill/Decode disaggregation 各自改变哪项约束，付出什么代价。
5. 多卡时 KV 状态怎样分布到 TP rank、PP stage；副本总量和单 GPU 容量不能混为一谈。
6. 监控指标怎样定位排队、prefill、decode、KV 压力或请求失败；分位数如何跨副本聚合。
7. 共享前缀缓存涉及什么租户隔离语义；缓存 key 隔离为什么不等于物理删除。
8. 教学 trace 能证明什么、不能证明什么；如何把一次有限成功请求 cohort 与生产稳态容量区分。

## 来源与实际核验内容

以下来源分别承担研究地图、方法证据、当前公开接口和标准语义的角色。动态文档以 2026-10-26 可见内容为准；部署时仍应按固定版本重查。

| 来源 | 实际核读内容及其支持的判断 | 证据边界 |
| --- | --- | --- |
| [A Survey of LLM Inference Systems](https://arxiv.org/abs/2506.21901) | 读取摘要、引言及与请求处理、batching/调度、KV、执行和部署架构相关的分类部分，作为本章议题图谱 | 综述用于发现分类和引用链，不替代每项工作的原论文，也不证明某实现当前可用 |
| [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | 核对静态 request-level batching 的等待问题、iteration-level scheduling 和 selective batching；会议页面给出 GPT-3 175B 上相对 FasterTransformer 的 36.9× throughput 同延迟结果 | 数字是论文特定模型、基线与实验设置的作者结果，不能外推为任何 serving workload 的倍率 |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 作为 KV block 管理与 serving 系统的原始研究入口；章节保留其机制主讲位置并链接后续 KV 章节 | 论文评测结果不能替代当前引擎版本和目标硬件测试 |
| [Sarathi-Serve, arXiv v3](https://arxiv.org/html/2403.02310) | 阅读动机、chunked prefill、stall-free batching、token budget 与消融/评测部分；方案围绕在已有 decode 中插入部分 prefill，降低长 prefill 对 decode 的整段阻塞 | 论文报告的 capacity 提升绑定 Mistral-7B/A100、Yi-34B/双 A100、Falcon-180B/PP 等设置；chunk 也会带来额外调度/执行开销，不将倍率当通用收益 |
| [DistServe, OSDI 2024](https://arxiv.org/abs/2401.09670) | 核对共置 prefill/decode 的干扰和耦合、分别配置资源/并行策略、跨阶段 KV 传输及以 SLO goodput 衡量的设计 | 作者的最大 request-rate 与 SLO 数字绑定其模型、请求 trace、硬件、流量和阈值；拆分也引入 KV 传输、网络带宽、编排和资源碎片成本 |
| [Revisiting SLOs and System-Level Metrics, arXiv v2](https://arxiv.org/html/2410.14257) | 阅读请求级 SLO、旧指标的反例、smooth-goodput 定义、评估设置与敏感性讨论。论文提出按用户读取速度定义 token deadline，并讨论人为延迟 token、主动放弃难请求可能优化旧指标的问题 | 研究指标方案而非行业统一标准；实验使用 A100-SXM4-80GB、CUDA 12.2、vLLM 0.6.3、Llama 3.1 8B/Qwen2.5 14B 及 ShareGPT/LooGLE 等工作负载，不能直接推出目标服务用户体验 |
| [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/) 与 [production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) | 核对事件级 ITL、请求级 TPOT、单 token 的 Prometheus 记值、prefill/decode/queue/KV 等观测维度；usage 页面用于 histogram bucket 配置与成本取舍 | `latest` 页面会变化，具体指标名称、默认 bucket 和版本行为要按运行版本核对；文档不代表目标部署的指标已采集 |
| [SGLang bench_serving guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx) | 核对当前 `main` 对 `--request-rate`（有限速率按 Poisson 生成到达）、`--max-concurrency`（限制在途并发）、streaming、TTFT/ITL/TPOT 和 goodput 的说明。Good request 需请求成功且满足所有所列 SLO；失败请求算未达标；单 token TPOT 记 0；同时输出 req/s 和 attainment | GitHub `main` 是动态文档；使用时须记录 commit 和 bench 命令。工具的 TPOT 占位语义不可与“无后续间隔”这一数学定义混淆 |
| [OpenMetrics specification](https://prometheus.io/docs/specs/om/open_metrics_spec/) | 核对预计算 quantile 不能跨实例聚合；章节因此说明先合并 histogram bucket 再估算 fleet percentile | 标准说明数据结构语义，不保证某套 dashboard 已正确配置 |
| [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html) | 作为当前厂商 runtime 对 context/generation 分离、KV transfer 和受支持拓扑的官方实现入口 | 官方页面随版本变化；不能证明目标 GPU、网卡、模型组合已通过验证 |
| [vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) | 核对 `cache_salt` 纳入首 block hash、用于隔离不同信任组的 prefix reuse | salt 影响缓存键/可复用性，不等于显存清理、日志删除、备份到期或完整数据擦除 |
| [Hugging Face TGI documentation](https://huggingface.co/docs/text-generation-inference/main/en/index) | 核对官方文档标注的 maintenance mode；正文将其作为选型和迁移维护状态信息 | 文档维护状态不意味着现有服务必须立刻下线；此状态必须按官方文档日期复核 |

## 研究带来的正文变化

1. 把 ITL 定义为相邻 streamed output event 的间隔，把 TPOT 定义为单请求首 token 后的平均 token 间隔；用 5 token、3 个输出事件的 30 ms 轨迹展示 ITL `[30,30] ms` 与 request-level TPOT `60/4=15 ms` 的区别。
2. 用 21-token 轨迹展示平均 TPOT 掩盖长停顿：19 个 20 ms 间隔加一个 500 ms 间隔，平均为 `(19×20+500)/20=44 ms`，但最大空档仍为 500 ms。补上这不等于 ITL/TBT 总能直接代表用户主观阅读体验，并介绍 smooth-goodput 论文的限定范围。
3. 容量段将 `S_replica` 限定为在明确 workload 与 SLO 下实测的可持续吞吐。输入和输出 token 需求可作初始负载账本，但共置 prefill/decode 的单阶段峰值不能简单拼接成实测容量。
4. 增加 request-level SLO goodput 与 attainment 定义，并用同为 100 个请求、10 秒窗口的例子说明 6,000 output tokens/s / 60 good requests 得 6 req/s，而 5,000 tokens/s / 90 good requests 得 9 req/s。指出 failed/cancelled/time-out 请求的处理规则必须先定义，并与完成率、取消率一起看。
5. 给出到达率与在途并发两种负载生成模型，说明固定 arrival rate 与固定 concurrency 不能互换；把 Poisson arrival、warm-up、长度分布、窗口和 SLO 判定留作可复现压测协议字段。
6. 增加 iteration scheduling、chunked prefill、P/D separation 的方法对照，说明每种方法针对哪个干扰/排队瓶颈及其代价；P/D 分离被放在 serving 架构层讨论，与 TP/PP 并行方式区分。
7. 把指标排查串成 trace 因果路径：queue 先升、prefill 随 prompt token 变、decode/ITL 同 KV 使用或抢占变、tokens/s 与 goodput 背离；避免只看 GPU 利用率作诊断。
8. 明确 histogram bucket/时序成本，以及不能平均各副本预计算 P95 的聚合边界；补充 TP/PP 下总 KV 与每 GPU/rank/stage 账本不同的原因。
9. 在 RAG 租户例子中用 cache identity/salt 描述 prefix reuse 隔离，并明确换 salt 不是物理清除缓存。
10. toy code 从四条成功请求 trace 计算队列、TTFT、TPOT、延迟、有限 cohort throughput 和重叠 KV token envelope；将输出命名为 cohort rate，并限定它不是稳态 QPS、GPU 容量或真实 allocated KV blocks。

## 原有知识点覆盖图

精修保留原章 §1.0—§1.19 的问题和材料。总览仍链接到后续主讲章节；没有把复杂机制压缩后悄悄删去。

| 原章节 | 处理方式 | 本轮延续/增强 |
| --- | --- | --- |
| §1.0 部署决策从 workload 开始 | 保留并加路线图 | 仍以 workload、SLO、质量、安全和资源作为决策输入；链接推理基础、KV、引擎、调度、多 GPU、观测和综合推导章节 |
| §1.1 checkpoint 不是服务 | 保留 | 模型 artifact、协议、runtime 和运营生命周期仍在部署对象中 |
| §1.2 离线推理和在线服务 | 保留 | 保持离线批处理与在线交互服务目标差异 |
| §1.3 请求过程、TTFT、TPOT、E2E | 保留并深入 | 增 ITL 与 TPOT 的分母/采样单位、单 token 边界、停顿反例 |
| §1.4 延迟与吞吐资源画像 | 保留并增强 | 增按 queue/prefill/decode/KV/preemption 的可证伪定位链 |
| §1.5 SLO 和容量估算 | 保留并深入 | 增 SLO goodput、attainment、实际负载模型、可持续能力和单阶段拼接边界 |
| §1.6 单机与分布式部署 | 保留并扩展 | 保留 TP/PP 等部署选项，加入 P/D 分离的独立机制和代价 |
| §1.7 推理引擎 | 保留并扩展 | 仍讲 vLLM、SGLang、TensorRT-LLM、llama.cpp、TGI 选型；增代表方案问题—机制—限制对照和维护状态 |
| §1.8 Tokenizer/template/API 协议 | 保留 | 保持输入序列化与模型协议作为部署契约 |
| §1.9 KV Cache 与显存 | 保留并澄清 | 加 TP/PP 各 rank/stage 的账本边界；细节仍交给第三章 |
| §1.10 生产请求路径 | 保留 | 维持 API、tokenize、queue、engine、stream 和工具等请求链路 |
| §1.11 监控指标 | 保留并增强 | 增观测信号诊断、histogram 桶、跨实例分位数与基数成本 |
| §1.12 质量/安全/发布 | 保留 | 维持离线质量和在线服务门禁分开验收 |
| §1.13 灰度、回滚、生命周期 | 保留 | 仍包含 drain、请求完成/取消、artifact 和版本回退 |
| §1.14 最小指标 demo | 保留并修订 | 从合成成功 trace 计算指标和 KV 上界，说明有限样本与生产容量的边界 |
| §1.15 服务 SLO | 保留并扩展 | 增质量、延迟、容量、安全、成本证据向量与 workload replay |
| §1.16 常见部署误区 | 保留 | 原“能跑即部署”“高吞吐即好”等工程问题继续在位 |
| §1.17 企业 RAG 推导 | 保留并加深 | 保持 workload 分型、资源、SLO、副本数、权限、缓存隔离和灰度回滚；P95 压力示例明确不是真实 P95 流量分布 |
| §1.18 资料入口 | 保留并扩展 | 增综述、原始研究、当前 benchmark 文档及厂商 disaggregation 入口 |
| §1.19 小结 | 保留 | 仍按服务生命周期归纳，不把部署缩减为模型加载或单一 tokens/s |

## 公式和数值复核

- 指标语义：5 output tokens 有 4 个 token 间隔；60 ms/4=15 ms request-level TPOT。流式事件间隔仍为 `[30,30] ms`。
- 停顿示例：`(19×20+500)/20=44 ms`；平均数没有消除 500 ms 最大空档。
- goodput 示例：`60/10=6 req/s`、`90/10=9 req/s`；attainment 分别为 60% 和 90%。这是定义演示，不是压测结果。
- 容量粗算：`ceil(50×120/(0.7×2400))+1=5`；0.7 是示例安全利用率，不是行业常数。
- RAG 场景：`30×220=6,600 output token/s`；假设每副本 4,000 token/s 且目标利用率 0.7，则 `ceil(6,600/(0.7×4,000))+1=4`。把每个请求都假设成 500-token 输出得到 7 个副本，只是压力包络；不是 P95 request-rate 推导，而且仍假设副本能力不变。
- toy trace：4 个成功请求的 cohort window 是 3.03 s，得到 1.32 req/s 和 87.46 output tokens/s；平均 TTFT 230 ms，nearest-rank P95 E2E 2,830 ms、eligible TPOT P95 20 ms。仅四个样本时 P95 等于最大值，不代表线上 tail probability。
- toy KV envelope：峰值同时活跃 token `3885`；按 `2×layers×tokens×KV_heads×head_dim×bytes` 得 485.62 MiB。这个模型把每个活跃请求按最终观测长度计，未建模 block 对齐、prefix sharing、取消、碎片和 runtime 实际 allocation。
- 已检查本章 display-math 作为 Markdown 内容；未使用外部 LaTeX/MathJax 渲染器。

## 代码、链接和配套同步

- `code_status`：用 Python 3 标准库提取并编译执行 §1.14 toy demo；输出与章节示例完全对应。非法输入覆盖负 queue、bool token count、成功请求零输出和重复 trace ID。这里只验证本地有限 cohort 逻辑。
- `link_status`：本章 7 个本地 Markdown 链接目标均存在；新增核心论文、SGLang benchmark、vLLM 指标/缓存、OpenMetrics、TensorRT-LLM 和 TGI 页面已打开核对支持范围。`latest`、`main` 和厂商 docs 属于动态来源，链接可访问不代表未来版本稳定。
- `sync_status`：`PAPERS.md` 加入 Orca/Sarathi/DistServe、LLM inference survey 和 SLO/goodput 研究入口；`GLOSSARY_EN_ZH.md` 增 ITL、Request-level TPOT、SLO Goodput、Smooth Goodput；`INTERVIEW_BANK.md` 增 tokens/s 与 goodput 比较题；`EXERCISES.md` 增 SLO goodput 练习并修复相邻部署题编号重复。
- `navigation_status`：第六册目录已有第一章映射；无需重复改导航。
- `validation`：§1.0—§1.19 连续；48 个 `~~~` fence marker 成对；本地链接核验无缺失；新增 SLO 算式与 demo 数值复核；运行代码 demo；`git diff --check` 通过。未运行 GPU、真实模型、集群容量、线上流量、论文 benchmark、PDF 构建或外部公式渲染。

## 未验证项和外推限制

1. 没有目标模型、请求 trace、GPU/网络和生产指标，因此本章不报告本地 serving 吞吐、成本、质量或可用容量。
2. Orca、Sarathi-Serve、DistServe 的论文性能数字各自绑定数据、模型、基线、硬件和 SLO；跨论文倍率不可直接排名，也不能外推目标部署。
3. vLLM、SGLang、TensorRT-LLM、TGI 的当前 docs/`main`/`latest` 会更新；版本行为须按实际运行版本重新核对，尤其是 histogram defaults、单 token TPOT 编码、P/D transfer backend 和 maintenance 状态。
4. SLO goodput 是实用的容量读数之一，但单一硬阈值会隐藏距离阈值很近的请求，若调度器可取消/放弃请求还可能产生目标偏置；应配 completion/cancel/error、分布指标和必要的质量评估。
5. 本章仍是部署总览，不替代第六册后续章节对 CUDA/kernel、KV block allocator、scheduler、parallel runtime、RAG/Agent 权限或生产 rollout 的详细机制说明。
