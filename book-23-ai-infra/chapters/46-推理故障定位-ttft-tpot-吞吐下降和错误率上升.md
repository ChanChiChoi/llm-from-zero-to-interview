# 第 46 章 推理故障定位：TTFT、TPOT、吞吐下降和错误率上升

上一章讲了训练故障定位。本章讲推理故障定位。

推理服务是在线系统，用户直接感知延迟、失败和质量变化。训练任务可以排队和重试，线上推理不行。推理故障定位的关键，是把端到端请求拆成阶段，并结合模型、runtime、路由、缓存、GPU、KV cache、网络和业务流量一起分析。

先记住一句话：

> 推理故障定位不要只看 QPS 和 GPU 利用率，而要按 gateway、router、queue、prefill、decode、streaming、cache、runtime 和下游依赖逐段拆解。

## 46.0 本讲资料边界与第二轮精修口径

本章按通用大模型推理平台故障定位抽象来写，不绑定 vLLM、SGLang、TensorRT-LLM、TGI、Triton Inference Server、Kubernetes、某个云厂商网关或具体 GPU 型号。资料校准时，主要参考 vLLM production metrics 对 TTFT、TPOT、queue、prefill、decode、KV cache、prefix cache 和调度状态的指标口径，参考 Triton Inference Server metrics 对 request、failure、pending、queue、compute 和 first response 的服务端指标拆分，参考 OpenTelemetry trace / span 对跨服务阶段拆解的通用模型，并结合前文推理平台、runtime 选型、Prefill / Decode / KV 资源画像、可观测性和发布治理章节中的 trace、版本、SLO、回滚和成本门禁。

第二轮精修只做三件事：

1. 把 TTFT、TPOT、p99、吞吐下降、错误率、timeout、KV cache、模型加载、streaming、cache、route、tool 和质量异常统一成推理故障样本。
2. 补齐端到端延迟拆分、TTFT、TPOT、尾延迟、token 吞吐、错误分类、timeout stage、KV 压力、stream abort、cache hit、tool failure 和质量漂移公式。
3. 增加一个 0 依赖 Python demo，用 toy inference incident cases 检查推理故障定位是否有完整请求 trace、版本 diff、容量证据和应急动作。

## 46.1 推理故障的常见类型

推理故障常见包括：

1. TTFT 升高。
2. TPOT 升高。
3. p99 延迟升高。
4. 吞吐下降。
5. 错误率上升。
6. 超时率上升。
7. OOM。
8. KV cache 不足。
9. 模型加载失败。
10. streaming 中断。
11. 路由异常。
12. cache 命中率下降。
13. 工具调用失败。
14. 质量或安全指标异常。

推理故障不仅是系统故障，也可能是模型版本、prompt、路由或数据依赖变化导致。

一个推理故障样本可以写成：

$$
I_i=(q_i,s_i,t_i,k_i,m_i,r_i,c_i,e_i,u_i,d_i,p_i,z_i)
$$

其中 `q_i` 是请求与 token 画像，`s_i` 是影响范围切片，`t_i` 是阶段 trace，`k_i` 是 KV cache 证据，`m_i` 是模型和 runtime 证据，`r_i` 是路由证据，`c_i` 是 cache 证据，`e_i` 是错误和 timeout 证据，`u_i` 是下游工具证据，`d_i` 是发布 / 配置 diff，`p_i` 是应急动作，`z_i` 是根因和复盘结论。

推理故障证据覆盖率：

$$
C_{\mathrm{infer\_fault}}=\frac{1}{N}\sum_{i=1}^{N}I(q_i,s_i,t_i,k_i,m_i,r_i,c_i,e_i,u_i,d_i,p_i,z_i\ \mathrm{present})
$$

## 46.2 推理请求链路回顾

一次典型推理请求链路：

```text
Client
  -> API Gateway
  -> Auth / Rate Limit
  -> Safety Input Check
  -> Model Router
  -> Cache Layer
  -> Admission Controller
  -> Runtime Queue
  -> Prefill
  -> Decode
  -> Safety Output Check
  -> Streaming Response
```

故障定位时，要先判断问题发生在哪一段。

端到端慢，不代表模型慢。

可能是网关排队、路由异常、cache miss、runtime queue 堆积、prefill 变长、decode 变慢或客户端接收慢。

端到端延迟可以拆成：

$$
T_{\mathrm{e2e}}=T_{\mathrm{gw}}+T_{\mathrm{auth}}+T_{\mathrm{router}}+T_{\mathrm{cache}}+T_{\mathrm{queue}}+T_{\mathrm{tok}}+T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{stream}}+T_{\mathrm{safety}}+T_{\mathrm{down}}
$$

故障定位时，至少要能把一次慢请求的每个阶段挂到同一个 trace id 上。

## 46.3 先看故障范围

排查第一步是确定影响范围。

要问：

1. 是所有模型还是某个模型？
2. 是所有租户还是某个租户？
3. 是所有地域还是某个集群？
4. 是所有请求还是长上下文请求？
5. 是 streaming 还是 non-streaming？
6. 是新版本还是旧版本？
7. 是某个 runtime 类型还是全部 runtime？
8. 是高峰期才发生还是持续发生？

范围判断能快速缩小根因。

如果只有某个模型版本出问题，优先看模型、runtime 配置和路由。

如果全平台都慢，优先看集群、网关、存储、网络或流量突增。

范围切片覆盖率：

$$
C_{\mathrm{scope}}=\frac{N_{\mathrm{slice\ fields}}}{N_{\mathrm{required\ slice\ fields}}}
$$

必要切片至少包括 model、tenant、region、endpoint、runtime、streaming、input length bucket、output length bucket 和 release version。

## 46.4 TTFT 升高怎么排查

TTFT 是从请求进入到第一个 token 返回的时间。

TTFT 升高可能来自：

1. 网关排队。
2. 鉴权或限流服务慢。
3. Router 慢。
4. Cache 查询慢或 miss。
5. runtime queue 等待。
6. prefill 输入变长。
7. tokenizer 慢。
8. KV cache 分配慢。
9. 模型实例冷启动。
10. safety input check 慢。

排查指标：

1. gateway latency。
2. router latency。
3. queue wait time。
4. prefill latency。
5. input token length。
6. active requests。
7. KV cache usage。
8. cold start count。

TTFT 是前半段链路的综合指标，必须拆阶段看。

TTFT 可以写成：

$$
T_{\mathrm{TTFT}}=T_{\mathrm{gw}}+T_{\mathrm{auth}}+T_{\mathrm{router}}+T_{\mathrm{cache}}+T_{\mathrm{queue}}+T_{\mathrm{tok}}+T_{\mathrm{prefill}}+T_{\mathrm{first\_decode}}+T_{\mathrm{first\_flush}}
$$

如果 `T_TTFT` 升高，但 `T_prefill` 没变，根因更可能在 gateway、router、cache、queue、cold start 或 safety input check。

## 46.5 TPOT 升高怎么排查

TPOT 是每生成一个输出 token 的平均耗时。

TPOT 升高通常与 decode 阶段有关。

可能原因：

1. decode batch 过大。
2. KV cache 访问变慢。
3. 显存带宽瓶颈。
4. GPU 负载过高。
5. batch 中长序列过多。
6. continuous batching 策略变化。
7. 量化或 runtime 版本变化。
8. streaming flush 策略变化。
9. sampling 参数变化。

排查指标：

1. decode latency。
2. output tokens/s。
3. active sequences。
4. batch size。
5. batch token count。
6. KV cache usage。
7. GPU memory bandwidth 相关指标。
8. runtime version。

TPOT 高，用户会感到输出一顿一顿。

TPOT 可以写成：

$$
T_{\mathrm{TPOT}}=\frac{T_{\mathrm{last\ token}}-T_{\mathrm{first\ token}}}{\max(1,N_{\mathrm{out}}-1)}
$$

其中 `N_out` 是输出 token 数。TTFT 高不一定 TPOT 高；前者更多受 prefill 和排队影响，后者更多受 decode、KV cache、显存带宽和调度影响。

## 46.6 p99 延迟升高

p99 升高表示尾部请求变慢。

常见原因：

1. 少量超长请求混入。
2. 某个租户突增。
3. 长短请求混跑。
4. 个别实例拥塞。
5. 某个 endpoint OOM 或重启。
6. cache 命中率下降。
7. downstream tool 慢。
8. 网络抖动。
9. autoscaling 来不及。

排查方法：

1. 按模型切分 p99。
2. 按租户切分 p99。
3. 按 endpoint 切分 p99。
4. 按输入长度切分 p99。
5. 按输出长度切分 p99。
6. 查看慢请求 trace。

平均延迟正常不代表系统健康，线上更要关注尾延迟。

尾延迟放大系数：

$$
A_{\mathrm{tail}}=\frac{Q_{0.99}(T_{\mathrm{e2e}})}{\max(\epsilon,Q_{0.50}(T_{\mathrm{e2e}}))}
$$

`A_tail` 明显升高时，优先按长上下文、租户、endpoint、region、runtime instance 和 tool call 切片找尾部请求。

## 46.7 吞吐下降怎么排查

吞吐下降可能表现为 QPS 下降或 tokens/s 下降。

要区分：

1. 请求数下降。
2. input tokens/s 下降。
3. output tokens/s 下降。
4. 每 GPU tokens/s 下降。
5. 有效吞吐下降但流量没变。

可能原因：

1. batch 变小。
2. queue 策略变化。
3. GPU 利用率下降。
4. runtime 版本退化。
5. 模型版本变大。
6. 输入输出长度变化。
7. cache 命中率下降。
8. 实例数量减少。
9. 某些实例不健康。

吞吐必须按 token 看，不要只看 QPS。

token 吞吐可以写成：

$$
X_{\mathrm{tok}}=\frac{N_{\mathrm{in}}+N_{\mathrm{out}}}{\Delta t}
$$

每 GPU 输出吞吐：

$$
X_{\mathrm{out/gpu}}=\frac{N_{\mathrm{out}}}{G\Delta t}
$$

其中 `G` 是 GPU 数。请求 QPS 不变但 `X_out/gpu` 下降，通常说明输出长度、decode、KV cache、batch 或 runtime 发生变化。

## 46.8 错误率上升怎么排查

错误率上升先分类错误。

常见错误类型：

1. 4xx 参数错误。
2. 401/403 权限错误。
3. 429 限流。
4. 5xx runtime 错误。
5. timeout。
6. OOM。
7. model load failed。
8. tokenizer error。
9. safety blocked。
10. tool call failed。

不同错误对应不同处理。

429 上升可能是限流配置或流量突增。

OOM 上升可能是上下文变长或并发过高。

5xx 上升可能是 runtime、模型或依赖异常。

按类别错误率：

$$
R_c=\frac{N_c}{N_{\mathrm{request}}}
$$

总错误率：

$$
R_{\mathrm{err}}=\frac{\sum_c N_c}{N_{\mathrm{request}}}
$$

其中 `c` 可以是 4xx、429、5xx、timeout、OOM、model load、tool failure 或 safety block。错误率上升要先分类，再决定是限流、回滚、扩容、摘除实例还是修复依赖。

## 46.9 Timeout 上升

Timeout 可能发生在多个阶段：

1. 网关超时。
2. 排队超时。
3. prefill 超时。
4. decode 超时。
5. streaming idle 超时。
6. tool call 超时。
7. safety check 超时。

排查时要看 timeout stage。

如果所有 timeout 都只记录成一个错误码，就很难定位。

Timeout budget 应该在 trace 中记录每个阶段消耗。

timeout stage 覆盖率：

$$
C_{\mathrm{timeout}}=\frac{N_{\mathrm{timeout\ with\ stage}}}{N_{\mathrm{timeout}}}
$$

如果 timeout 都只有一个错误码，`C_timeout` 很低，系统就无法区分 queue timeout、prefill timeout、decode timeout、streaming idle timeout 和 tool timeout。

## 46.10 KV Cache 不足

KV cache 不足会导致：

1. admission reject 上升。
2. OOM。
3. p99 抖动。
4. 长上下文请求失败。
5. active sequence 上限提前触达。

可能原因：

1. 输入上下文变长。
2. 输出变长。
3. 并发变高。
4. batch 策略变化。
5. prefix cache 保留太多。
6. KV block 碎片。
7. 模型层数或 hidden size 变大。

排查指标：

1. KV cache usage。
2. free blocks。
3. active sequences。
4. average context length。
5. eviction count。
6. OOM count。

KV cache 是大模型推理特有的重要瓶颈。

KV cache 粗略显存预算：

$$
M_{\mathrm{kv}}=2L_{\mathrm{layer}}H_{\mathrm{kv}}d_hbN_{\mathrm{token}}
$$

其中 `2` 表示 K 和 V，`L_layer` 是层数，`H_kv` 是 KV head 数，`d_h` 是 head dimension，`b` 是每个元素字节数，`N_token` 是当前保留的 KV token 数。

KV 压力：

$$
P_{\mathrm{kv}}=\frac{M_{\mathrm{kv,used}}}{M_{\mathrm{kv,budget}}}
$$

当 `P_kv` 接近 1 时，admission reject、OOM、eviction 和 p99 抖动都会变多。

## 46.11 模型加载失败

模型加载失败常见原因：

1. 权重文件缺失。
2. checksum 不匹配。
3. tokenizer 缺失或不匹配。
4. config 错误。
5. runtime 不支持该模型结构。
6. 量化格式不兼容。
7. CUDA 或 driver 版本不匹配。
8. 显存不足。
9. 权限不足。
10. 权重下载超时。

排查要看 deployment package、model registry、runtime logs 和节点环境。

不要只看 Pod restart。

模型加载门禁：

$$
G_{\mathrm{load}}=I(M_{\mathrm{manifest}}=1)\cdot I(C_{\mathrm{weight}}=1)\cdot I(C_{\mathrm{tokenizer}}=1)\cdot I(C_{\mathrm{runtime}}=1)\cdot I(C_{\mathrm{permission}}=1)
$$

如果模型加载失败，排查要从 model registry、deployment package、checksum、tokenizer、runtime compatibility、量化格式、权限和节点环境一起看。

## 46.12 Streaming 中断

Streaming 中断可能来自：

1. 客户端断开。
2. 网关 idle timeout。
3. 后端 decode 超时。
4. 网络抖动。
5. runtime 崩溃。
6. safety output check 中断。
7. 代理层缓冲问题。

排查指标：

1. stream open count。
2. stream duration。
3. client disconnect count。
4. server abort count。
5. tokens sent before abort。
6. idle timeout count。

Streaming 问题要区分服务端生成慢和客户端消费慢。

stream abort rate：

$$
R_{\mathrm{abort}}=\frac{N_{\mathrm{server\ abort}}+N_{\mathrm{client\ disconnect}}+N_{\mathrm{idle\ timeout}}}{N_{\mathrm{stream}}}
$$

如果 abort 主要来自 client disconnect，优先看客户端网络和消费速度；如果来自 server abort 或 idle timeout，优先看 decode、gateway、proxy buffer、timeout budget 和 runtime。

## 46.13 Cache 命中率下降

Cache 命中率下降会导致延迟和成本上升。

可能原因：

1. prompt 模板变化。
2. model version 变化。
3. tokenizer 变化。
4. cache key 变化。
5. TTL 过短。
6. cache 被驱逐。
7. 流量分布变化。
8. 权限域变化。

排查要看：

1. prompt cache hit rate。
2. semantic cache hit rate。
3. result cache hit rate。
4. cache key diff。
5. eviction rate。
6. latency before/after cache。

缓存问题常常在发布后出现，因为 prompt、模型或安全策略版本变化会影响 key。

cache hit rate：

$$
H_{\mathrm{cache}}=\frac{N_{\mathrm{hit}}}{N_{\mathrm{lookup}}}
$$

缓存节省的 prefill 时间可以近似写成：

$$
S_{\mathrm{cache}}=N_{\mathrm{hit}}\bar{T}_{\mathrm{prefill\ saved}}
$$

Cache 命中率下降时，要看 cache key 是否绑定 model、tokenizer、prompt、safety policy、tenant 和 permission 版本。

## 46.14 路由异常

路由异常可能表现为：

1. 请求打到错误模型。
2. 灰度比例异常。
3. 高优先级租户被降级。
4. 过载实例仍然接流量。
5. fallback 频繁触发。
6. 成本突然上升。

排查需要 route trace。

Route trace 应记录：

1. 候选模型。
2. 选择模型。
3. endpoint。
4. route reason。
5. 灰度规则。
6. fallback chain。
7. 实时负载。
8. 租户策略。

没有 route trace，路由问题很难复盘。

路由正确性门禁：

$$
G_{\mathrm{route}}=I(m_{\mathrm{selected}}\in M_{\mathrm{candidate}})\cdot I(P_{\mathrm{tenant}}=1)\cdot I(H_{\mathrm{endpoint}}=1)\cdot I(C_{\mathrm{trace}}=1)
$$

其中 `P_tenant` 是租户策略满足，`H_endpoint` 是 endpoint 健康，`C_trace` 是 route trace 完整。

## 46.15 工具调用失败

Agent 或 tool-use 场景中，错误率可能来自工具。

常见原因：

1. tool schema 变化。
2. 参数生成错误。
3. 工具权限失败。
4. 工具超时。
5. 工具返回格式变化。
6. 下游服务不可用。
7. 重试策略错误。

排查要看 tool call trace：

1. tool name/version。
2. input arguments。
3. output。
4. status。
5. latency。
6. error message。
7. retry count。

不要把所有 Agent 失败都归因于模型。

工具失败率：

$$
R_{\mathrm{tool}}=\frac{N_{\mathrm{tool\ fail}}}{N_{\mathrm{tool\ call}}}
$$

工具失败率上升时，要看 schema 版本、权限决策、timeout、retry、下游状态和 tool call trace，而不是只看最终 HTTP 状态码。

## 46.16 质量异常怎么排查

推理服务可能系统指标正常，但质量下降。

可能原因：

1. 模型版本变化。
2. prompt 版本变化。
3. retrieval index 变化。
4. reranker 变化。
5. tool description 变化。
6. safety 策略变化。
7. sampling 参数变化。
8. fallback 到小模型。

排查需要：

1. model version。
2. prompt version。
3. route trace。
4. retrieval trace。
5. tool trace。
6. eval/online feedback。

HTTP 200 不代表模型服务质量正常。

质量漂移可以写成：

$$
\Delta Q=Q_{\mathrm{current}}-Q_{\mathrm{baseline}}
$$

当 `\Delta Q` 为负且超过阈值时，要把 model、prompt、retrieval、rerank、tool、safety、sampling、fallback 和 release event 放进同一张 diff 表。

## 46.17 发布后故障

发布后常见故障：

1. 新模型延迟更高。
2. 新 tokenizer 不兼容。
3. prompt cache 失效。
4. 输出格式变化。
5. tool call 参数变化。
6. 安全拒答率变化。
7. 成本上升。
8. 灰度分桶异常。

发布后排查应先看变更：

1. model version。
2. tokenizer。
3. prompt。
4. runtime。
5. route policy。
6. safety policy。
7. cache policy。

变更时间线和事件记录非常重要。

发布差异集合：

$$
\Delta_{\mathrm{release}}=\{k:v^{\mathrm{new}}_k\ne v^{\mathrm{old}}_k\}
$$

推理故障常常是多个小变更共同导致，例如 prompt 变化让 cache miss，runtime 变化让 TPOT 升高，route policy 变化又把长请求集中到同一组实例。

## 46.18 推理故障排查 Runbook

一个通用 runbook：

1. 确认影响范围。
2. 查看最近变更和发布事件。
3. 按模型、租户、endpoint、地域切分指标。
4. 查看 TTFT、TPOT、p99、error、timeout。
5. 查看 queue、KV cache、GPU、active sequences。
6. 抽样慢请求 trace。
7. 查看 runtime logs。
8. 查看 route/cache/tool/retrieval trace。
9. 判断是否需要限流、降级、回滚或扩容。
10. 记录复盘。

Runbook 的价值是减少线上故障时的临场猜测。

## 46.19 应急处理手段

常见应急手段：

1. 限流异常租户。
2. 降级到小模型。
3. 回滚模型版本。
4. 关闭有问题的灰度。
5. 摘除异常 endpoint。
6. 扩容实例。
7. 降低 max output tokens。
8. 限制最大上下文长度。
9. 启用缓存兜底。
10. 暂停高风险工具调用。

应急处理要优先恢复服务，再深入定位根因。

## 46.20 推理故障定位审计指标和最小 demo

把本章落到平台验收时，可以用 16 个门禁：

1. Inference Fault Evidence Coverage：请求画像、影响范围、阶段 trace、KV、模型、路由、cache、错误、工具、发布 diff、应急动作和诊断是否齐全。
2. Request Scope Slice：是否按 model、tenant、region、endpoint、runtime、streaming、输入长度、输出长度和版本切片。
3. Trace Stage Coverage：gateway、auth、router、cache、queue、tokenize、prefill、first decode、decode、stream、safety 和 downstream 是否有 span。
4. TTFT Decomposition：TTFT 是否能拆到 queue、tokenize、prefill、first decode 和 first flush，并有 SLO 与根因解释。
5. TPOT Decode Health：TPOT 是否能绑定 decode time、输出 token、active sequences、KV 和 runtime 版本。
6. Tail Latency Attribution：p99 / p50 是否可计算，尾部慢请求是否有 trace sample 和切片归因。
7. Throughput Token Capacity：QPS、input tokens/s、output tokens/s、per GPU output tokens/s 是否都达标。
8. Error Taxonomy Timeout Stage：错误是否按 4xx、429、5xx、timeout、OOM、tool、safety 分类，timeout 是否有阶段。
9. KV Cache Pressure Guard：KV pressure、free blocks、eviction、admission reject、fragmentation 是否可判断。
10. Model Artifact Load Readiness：manifest、checksum、tokenizer、config、runtime、quantization、权限和 warmup 是否完整。
11. Streaming Reliability：server abort、client disconnect、idle timeout、tokens before abort 和 cancellation cleanup 是否可观测。
12. Cache Hit Key Governance：cache hit、key version、prompt / tokenizer / safety / tenant 版本和 eviction 是否可解释。
13. Route Trace Correctness：候选模型、选择模型、路由原因、灰度比例、租户策略、endpoint 健康和 fallback 是否在 trace 中。
14. Tool Dependency Trace：tool schema、权限、timeout、retry、下游错误和 tool call trace 是否完整。
15. Quality Release Diff Readiness：模型、prompt、runtime、route、cache、safety、sampling diff 与质量指标是否关联。
16. Inference Fault Diagnosis Gate：最终是否有 owner、runbook、应急动作、回滚、postmortem 和 P0 风险阻断。

综合门禁：

$$
G_{\mathrm{inference\_fault}}=\prod_{j=1}^{16}G_j
$$

下面是一个 0 依赖 demo，用 toy inference incident cases 检查推理故障定位是否只是“看几条 dashboard 曲线”，还是能真正把慢请求、错误、容量、版本和应急动作串成证据链。

```python
from copy import deepcopy


class MiniInferenceFaultDiagnosisAudit:
    GATES = [
        "inference_fault_evidence_coverage",
        "request_scope_slice",
        "trace_stage_coverage",
        "ttft_decomposition",
        "tpot_decode_health",
        "tail_latency_attribution",
        "throughput_token_capacity",
        "error_taxonomy_timeout_stage",
        "kv_cache_pressure_guard",
        "model_artifact_load_readiness",
        "streaming_reliability",
        "cache_hit_key_governance",
        "route_trace_correctness",
        "tool_dependency_trace",
        "quality_release_diff_readiness",
        "inference_fault_diagnosis_gate",
    ]

    EVIDENCE_FIELDS = [
        "request_profile",
        "scope",
        "trace",
        "kv_cache",
        "model_runtime",
        "route",
        "cache",
        "errors",
        "tool",
        "release_diff",
        "mitigation",
        "diagnosis",
    ]
    REQUIRED_SCOPE_FIELDS = [
        "model",
        "tenant",
        "region",
        "endpoint",
        "runtime",
        "streaming",
        "input_length_bucket",
        "output_length_bucket",
        "release_version",
    ]
    REQUIRED_STAGES = [
        "gateway",
        "auth",
        "router",
        "cache",
        "queue",
        "tokenize",
        "prefill",
        "first_decode",
        "first_flush",
        "decode",
        "stream",
        "safety",
        "downstream",
    ]

    @staticmethod
    def present(record, key):
        return key in record and record[key] is not None and record[key] != ""

    def coverage(self, record, fields):
        if not record:
            return 0.0
        return sum(1 for field in fields if self.present(record, field)) / len(fields)

    def stage_map(self, case):
        return {stage["name"]: stage["ms"] for stage in case.get("trace", {}).get("stages", [])}

    def ttft_ms(self, case):
        stages = self.stage_map(case)
        names = [
            "gateway",
            "auth",
            "router",
            "cache",
            "queue",
            "tokenize",
            "prefill",
            "first_decode",
            "first_flush",
        ]
        return sum(stages[name] for name in names)

    def tpot_ms(self, case):
        trace = case["trace"]
        return trace["decode_total_ms"] / max(1, trace["output_tokens"] - 1)

    def tail_amplification(self, case):
        latency = case["latency"]
        return latency["p99_ms"] / max(1e-12, latency["p50_ms"])

    def total_tokens_per_s(self, case):
        traffic = case["traffic"]
        return (traffic["input_tokens"] + traffic["output_tokens"]) / traffic["window_s"]

    def per_gpu_output_tokens_s(self, case):
        traffic = case["traffic"]
        return traffic["output_tokens"] / (traffic["gpu_count"] * traffic["window_s"])

    def error_rate(self, case):
        errors = case["errors"]
        return sum(errors["counts"].values()) / errors["requests"]

    def timeout_stage_coverage(self, case):
        errors = case["errors"]
        timeout_count = errors["counts"].get("timeout", 0)
        if timeout_count == 0:
            return 1.0
        return sum(errors["timeout_stages"].values()) / timeout_count

    def kv_pressure(self, case):
        kv = case["kv_cache"]
        return kv["used_gib"] / kv["budget_gib"]

    def stream_abort_rate(self, case):
        stream = case["streaming"]
        aborts = stream["server_abort"] + stream["client_disconnect"] + stream["idle_timeout"]
        return aborts / stream["open_streams"]

    def cache_hit_rate(self, case):
        cache = case["cache"]
        return cache["hits"] / cache["lookups"]

    def tool_failure_rate(self, case):
        tool = case["tool"]
        return tool["failures"] / tool["calls"]

    def quality_delta(self, case):
        quality = case["quality"]
        return quality["current_score"] - quality["baseline_score"]

    def inference_fault_evidence_coverage(self, case):
        evidence = case.get("evidence", {})
        return self.coverage(evidence, self.EVIDENCE_FIELDS) == 1.0 and all(
            evidence[field] is True for field in self.EVIDENCE_FIELDS
        )

    def request_scope_slice(self, case):
        scope = case.get("scope", {})
        return self.coverage(scope, self.REQUIRED_SCOPE_FIELDS) == 1.0

    def trace_stage_coverage(self, case):
        stages = set(self.stage_map(case))
        return set(self.REQUIRED_STAGES).issubset(stages) and bool(case.get("trace", {}).get("trace_id"))

    def ttft_decomposition(self, case):
        trace = case.get("trace", {})
        ttft = self.ttft_ms(case)
        return (
            ttft <= trace.get("ttft_slo_ms", 0)
            or (
                trace.get("ttft_root_cause_ready") is True
                and trace.get("queue_ms_explained") is True
                and trace.get("prefill_ms_explained") is True
            )
        )

    def tpot_decode_health(self, case):
        decode = case.get("decode", {})
        tpot = self.tpot_ms(case)
        return (
            tpot <= decode.get("tpot_slo_ms", 0)
            or (
                decode.get("decode_root_cause_ready") is True
                and decode.get("kv_evidence_ready") is True
                and decode.get("runtime_version_bound") is True
            )
        )

    def tail_latency_attribution(self, case):
        latency = case.get("latency", {})
        return (
            latency.get("p99_ms", 10**9) <= latency.get("p99_slo_ms", 0)
            or (
                latency.get("slow_trace_sampled") is True
                and latency.get("slice_attribution_ready") is True
                and self.tail_amplification(case) <= latency.get("max_tail_amplification", 10**9)
            )
        )

    def throughput_token_capacity(self, case):
        traffic = case.get("traffic", {})
        return (
            traffic.get("qps", 0.0) >= traffic.get("qps_target", 10**9)
            and self.total_tokens_per_s(case) >= traffic.get("total_tokens_s_target", 10**9)
            and self.per_gpu_output_tokens_s(case) >= traffic.get("per_gpu_output_tokens_s_target", 10**9)
        )

    def error_taxonomy_timeout_stage(self, case):
        errors = case.get("errors", {})
        return (
            errors.get("taxonomy_ready") is True
            and self.error_rate(case) <= errors.get("max_error_rate", 0.0)
            and self.timeout_stage_coverage(case) >= 0.95
        )

    def kv_cache_pressure_guard(self, case):
        kv = case.get("kv_cache", {})
        return (
            self.kv_pressure(case) <= kv.get("max_pressure", 0.0)
            and kv.get("free_blocks", 0) > kv.get("min_free_blocks", 10**9)
            and kv.get("eviction_count", 10**9) <= kv.get("max_evictions", 0)
            and kv.get("admission_reject_rate", 1.0) <= kv.get("max_admission_reject_rate", 0.0)
            and kv.get("fragmentation", 1.0) <= kv.get("max_fragmentation", 0.0)
        )

    def model_artifact_load_readiness(self, case):
        model = case.get("model_runtime", {})
        required = [
            "manifest",
            "weight_checksum",
            "tokenizer",
            "config",
            "runtime_compatible",
            "quantization_compatible",
            "permission",
            "warmup_passed",
        ]
        return all(model.get(field) is True for field in required) and model.get("load_latency_ms", 10**9) <= model.get("load_slo_ms", 0)

    def streaming_reliability(self, case):
        stream = case.get("streaming", {})
        return (
            stream.get("open_streams", 0) > 0
            and self.stream_abort_rate(case) <= stream.get("max_abort_rate", 0.0)
            and stream.get("cancellation_cleanup") is True
            and stream.get("tokens_before_abort_median", 0) >= stream.get("min_tokens_before_abort_median", 0)
        )

    def cache_hit_key_governance(self, case):
        cache = case.get("cache", {})
        return (
            self.cache_hit_rate(case) >= cache.get("min_hit_rate", 1.0)
            and cache.get("key_versions_bound") is True
            and cache.get("prompt_version_bound") is True
            and cache.get("tokenizer_version_bound") is True
            and cache.get("safety_version_bound") is True
            and cache.get("tenant_permission_bound") is True
            and cache.get("eviction_rate", 1.0) <= cache.get("max_eviction_rate", 0.0)
        )

    def route_trace_correctness(self, case):
        route = case.get("route", {})
        canary_gap = abs(route.get("actual_canary_percent", 0.0) - route.get("expected_canary_percent", 0.0))
        return (
            route.get("selected_model") in route.get("candidate_models", [])
            and route.get("selected_model") == route.get("expected_model")
            and bool(route.get("route_reason"))
            and canary_gap <= route.get("max_canary_gap", 0.0)
            and route.get("tenant_policy_ok") is True
            and route.get("endpoint_health_ok") is True
            and route.get("trace_complete") is True
        )

    def tool_dependency_trace(self, case):
        tool = case.get("tool", {})
        return (
            tool.get("calls", 0) > 0
            and self.tool_failure_rate(case) <= tool.get("max_failure_rate", 0.0)
            and tool.get("schema_version_bound") is True
            and tool.get("permission_decisions_recorded") is True
            and tool.get("timeout_budget_recorded") is True
            and tool.get("retry_policy_recorded") is True
            and tool.get("trace_complete") is True
        )

    def quality_release_diff_readiness(self, case):
        quality = case.get("quality", {})
        diff = case.get("release_diff", {})
        required = ["model", "prompt", "runtime", "route", "cache", "safety", "sampling"]
        return (
            self.quality_delta(case) >= -quality.get("max_allowed_drop", 0.0)
            and all(field in diff.get("changed_fields", []) for field in required)
            and quality.get("eval_feedback_linked") is True
            and diff.get("release_event_timeline") is True
            and diff.get("rollback_ready") is True
        )

    def inference_fault_diagnosis_gate(self, case):
        gate = case.get("platform_gate", {})
        return (
            gate.get("enabled") is True
            and bool(gate.get("owner"))
            and bool(gate.get("runbook"))
            and bool(gate.get("mitigation_actions"))
            and gate.get("rollback_ready") is True
            and gate.get("postmortem_required") is True
            and gate.get("p0_open") is False
        )

    def audit_case(self, case):
        return {gate: getattr(self, gate)(case) for gate in self.GATES}

    def run_all(self, cases):
        results = {case["case_id"]: self.audit_case(case) for case in cases}
        metrics = {}
        for gate in self.GATES:
            passed = sum(1 for result in results.values() if result[gate])
            metrics[gate] = round(passed / len(cases), 3)
        failed_cases = [
            case_id
            for case_id, result in results.items()
            if not all(result.values())
        ]
        failed_gates = [
            gate
            for gate in self.GATES
            if any(not result[gate] for result in results.values())
        ]
        return {
            "metrics": metrics,
            "hard_blocker_count": len(failed_cases),
            "failed_cases": failed_cases,
            "failed_gates": failed_gates,
            "inference_fault_diagnosis_gate_pass": metrics["inference_fault_diagnosis_gate"] == 1.0,
        }

    def example_outputs(self, case):
        return {
            "evidence_coverage": round(self.coverage(case["evidence"], self.EVIDENCE_FIELDS), 3),
            "scope_coverage": round(self.coverage(case["scope"], self.REQUIRED_SCOPE_FIELDS), 3),
            "stage_coverage": round(len(self.stage_map(case)) / len(self.REQUIRED_STAGES), 3),
            "ttft_ms": self.ttft_ms(case),
            "tpot_ms": round(self.tpot_ms(case), 3),
            "p99_ms": case["latency"]["p99_ms"],
            "tail_amplification": round(self.tail_amplification(case), 3),
            "tokens_per_s": round(self.total_tokens_per_s(case), 3),
            "per_gpu_output_tokens_s": round(self.per_gpu_output_tokens_s(case), 3),
            "error_rate": round(self.error_rate(case), 3),
            "timeout_stage_coverage": round(self.timeout_stage_coverage(case), 3),
            "kv_pressure": round(self.kv_pressure(case), 3),
            "model_load_ready": self.model_artifact_load_readiness(case),
            "stream_abort_rate": round(self.stream_abort_rate(case), 3),
            "cache_hit_rate": round(self.cache_hit_rate(case), 3),
            "route_correct": self.route_trace_correctness(case),
            "tool_failure_rate": round(self.tool_failure_rate(case), 3),
            "quality_delta": round(self.quality_delta(case), 3),
        }


def build_good_case():
    return {
        "case_id": "full_inference_fault_diagnosis",
        "evidence": {
            "request_profile": True,
            "scope": True,
            "trace": True,
            "kv_cache": True,
            "model_runtime": True,
            "route": True,
            "cache": True,
            "errors": True,
            "tool": True,
            "release_diff": True,
            "mitigation": True,
            "diagnosis": True,
        },
        "scope": {
            "model": "chat-v8",
            "tenant": "all",
            "region": "us-east",
            "endpoint": "chat",
            "runtime": "vllm",
            "streaming": True,
            "input_length_bucket": "4k-8k",
            "output_length_bucket": "256-512",
            "release_version": "rel_2026_06_15",
        },
        "trace": {
            "trace_id": "tr_infer_1",
            "stages": [
                {"name": "gateway", "ms": 40},
                {"name": "auth", "ms": 20},
                {"name": "router", "ms": 35},
                {"name": "cache", "ms": 45},
                {"name": "queue", "ms": 180},
                {"name": "tokenize", "ms": 30},
                {"name": "prefill", "ms": 280},
                {"name": "first_decode", "ms": 45},
                {"name": "first_flush", "ms": 45},
                {"name": "decode", "ms": 580},
                {"name": "stream", "ms": 60},
                {"name": "safety", "ms": 35},
                {"name": "downstream", "ms": 20},
            ],
            "decode_total_ms": 580,
            "output_tokens": 11,
            "ttft_slo_ms": 900,
            "ttft_root_cause_ready": True,
            "queue_ms_explained": True,
            "prefill_ms_explained": True,
        },
        "decode": {
            "tpot_slo_ms": 80,
            "active_sequences": 96,
            "decode_root_cause_ready": True,
            "kv_evidence_ready": True,
            "runtime_version_bound": True,
        },
        "latency": {
            "p50_ms": 720,
            "p99_ms": 1900,
            "p99_slo_ms": 2000,
            "max_tail_amplification": 3.0,
            "slow_trace_sampled": True,
            "slice_attribution_ready": True,
        },
        "traffic": {
            "qps": 10.0,
            "qps_target": 9.0,
            "window_s": 100,
            "input_tokens": 300000,
            "output_tokens": 220000,
            "gpu_count": 8,
            "total_tokens_s_target": 5000,
            "per_gpu_output_tokens_s_target": 250,
        },
        "errors": {
            "requests": 10000,
            "counts": {"4xx": 40, "429": 60, "5xx": 35, "timeout": 30, "oom": 10, "tool": 5},
            "timeout_stages": {"queue": 12, "decode": 18},
            "taxonomy_ready": True,
            "max_error_rate": 0.02,
        },
        "kv_cache": {
            "used_gib": 65.6,
            "budget_gib": 80.0,
            "max_pressure": 0.9,
            "free_blocks": 320,
            "min_free_blocks": 200,
            "eviction_count": 2,
            "max_evictions": 5,
            "admission_reject_rate": 0.01,
            "max_admission_reject_rate": 0.02,
            "fragmentation": 0.08,
            "max_fragmentation": 0.15,
        },
        "model_runtime": {
            "manifest": True,
            "weight_checksum": True,
            "tokenizer": True,
            "config": True,
            "runtime_compatible": True,
            "quantization_compatible": True,
            "permission": True,
            "warmup_passed": True,
            "load_latency_ms": 90000,
            "load_slo_ms": 120000,
        },
        "streaming": {
            "open_streams": 5000,
            "server_abort": 40,
            "client_disconnect": 20,
            "idle_timeout": 0,
            "max_abort_rate": 0.02,
            "tokens_before_abort_median": 64,
            "min_tokens_before_abort_median": 32,
            "cancellation_cleanup": True,
        },
        "cache": {
            "hits": 4200,
            "lookups": 10000,
            "min_hit_rate": 0.4,
            "key_versions_bound": True,
            "prompt_version_bound": True,
            "tokenizer_version_bound": True,
            "safety_version_bound": True,
            "tenant_permission_bound": True,
            "eviction_rate": 0.03,
            "max_eviction_rate": 0.05,
            "latency_saved_ms": 180,
        },
        "route": {
            "candidate_models": ["chat-v8", "chat-v7-small"],
            "selected_model": "chat-v8",
            "expected_model": "chat-v8",
            "route_reason": "quality_slo_primary",
            "expected_canary_percent": 5.0,
            "actual_canary_percent": 5.2,
            "max_canary_gap": 0.5,
            "tenant_policy_ok": True,
            "endpoint_health_ok": True,
            "fallback_chain": ["chat-v7-small"],
            "trace_complete": True,
        },
        "tool": {
            "calls": 500,
            "failures": 10,
            "max_failure_rate": 0.03,
            "schema_version_bound": True,
            "permission_decisions_recorded": True,
            "timeout_budget_recorded": True,
            "retry_policy_recorded": True,
            "trace_complete": True,
        },
        "quality": {
            "baseline_score": 0.86,
            "current_score": 0.845,
            "max_allowed_drop": 0.03,
            "eval_feedback_linked": True,
        },
        "release_diff": {
            "changed_fields": ["model", "prompt", "runtime", "route", "cache", "safety", "sampling"],
            "release_event_timeline": True,
            "rollback_ready": True,
        },
        "platform_gate": {
            "enabled": True,
            "owner": "serving-oncall",
            "runbook": "runbook://inference-fault",
            "mitigation_actions": ["rate_limit_spike", "rollback_cache_policy"],
            "rollback_ready": True,
            "postmortem_required": True,
            "p0_open": False,
        },
    }


def build_bad_cases(good_case):
    cases = []

    case = deepcopy(good_case)
    case["case_id"] = "inference_evidence_missing_bad"
    case["evidence"]["diagnosis"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "scope_slice_missing_bad"
    case["scope"].pop("runtime")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "trace_stage_missing_bad"
    case["trace"]["stages"] = [stage for stage in case["trace"]["stages"] if stage["name"] != "downstream"]
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "ttft_decomposition_bad"
    for stage in case["trace"]["stages"]:
        if stage["name"] == "queue":
            stage["ms"] = 520
    case["trace"]["ttft_root_cause_ready"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "tpot_decode_bad"
    case["trace"]["decode_total_ms"] = 980
    case["decode"]["decode_root_cause_ready"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "tail_latency_unattributed_bad"
    case["latency"]["p99_ms"] = 2600
    case["latency"]["slow_trace_sampled"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "throughput_capacity_bad"
    case["traffic"]["output_tokens"] = 120000
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "error_taxonomy_bad"
    case["errors"]["taxonomy_ready"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "kv_pressure_bad"
    case["kv_cache"]["used_gib"] = 77.5
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "model_artifact_load_bad"
    case["model_runtime"]["weight_checksum"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "streaming_reliability_bad"
    case["streaming"]["cancellation_cleanup"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cache_key_governance_bad"
    case["cache"]["key_versions_bound"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "route_trace_bad"
    case["route"]["selected_model"] = "chat-v7-small"
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "tool_trace_bad"
    case["tool"]["trace_complete"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "quality_release_diff_bad"
    case["quality"]["current_score"] = 0.79
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "inference_fault_gate_missing_bad"
    case["platform_gate"]["enabled"] = False
    cases.append(case)

    return cases


audit = MiniInferenceFaultDiagnosisAudit()
good = build_good_case()
cases = [good] + build_bad_cases(good)
summary = audit.run_all(cases)

print("inference_fault_examples=" + repr(audit.example_outputs(good)))
print("metrics=" + repr(summary["metrics"]))
print("hard_blocker_count=" + repr(summary["hard_blocker_count"]))
print("failed_cases=" + repr(summary["failed_cases"]))
print("failed_gates=" + repr(summary["failed_gates"]))
print("inference_fault_diagnosis_gate_pass=" + repr(summary["inference_fault_diagnosis_gate_pass"]))
```

参考输出应类似：

```text
inference_fault_examples={'evidence_coverage': 1.0, 'scope_coverage': 1.0, 'stage_coverage': 1.0, 'ttft_ms': 720, 'tpot_ms': 58.0, 'p99_ms': 1900, 'tail_amplification': 2.639, 'tokens_per_s': 5200.0, 'per_gpu_output_tokens_s': 275.0, 'error_rate': 0.018, 'timeout_stage_coverage': 1.0, 'kv_pressure': 0.82, 'model_load_ready': True, 'stream_abort_rate': 0.012, 'cache_hit_rate': 0.42, 'route_correct': True, 'tool_failure_rate': 0.02, 'quality_delta': -0.015}
metrics={'inference_fault_evidence_coverage': 0.941, 'request_scope_slice': 0.941, 'trace_stage_coverage': 0.941, 'ttft_decomposition': 0.941, 'tpot_decode_health': 0.941, 'tail_latency_attribution': 0.941, 'throughput_token_capacity': 0.941, 'error_taxonomy_timeout_stage': 0.941, 'kv_cache_pressure_guard': 0.941, 'model_artifact_load_readiness': 0.941, 'streaming_reliability': 0.941, 'cache_hit_key_governance': 0.941, 'route_trace_correctness': 0.941, 'tool_dependency_trace': 0.941, 'quality_release_diff_readiness': 0.941, 'inference_fault_diagnosis_gate': 0.941}
hard_blocker_count=16
failed_cases=['inference_evidence_missing_bad', 'scope_slice_missing_bad', 'trace_stage_missing_bad', 'ttft_decomposition_bad', 'tpot_decode_bad', 'tail_latency_unattributed_bad', 'throughput_capacity_bad', 'error_taxonomy_bad', 'kv_pressure_bad', 'model_artifact_load_bad', 'streaming_reliability_bad', 'cache_key_governance_bad', 'route_trace_bad', 'tool_trace_bad', 'quality_release_diff_bad', 'inference_fault_gate_missing_bad']
failed_gates=['inference_fault_evidence_coverage', 'request_scope_slice', 'trace_stage_coverage', 'ttft_decomposition', 'tpot_decode_health', 'tail_latency_attribution', 'throughput_token_capacity', 'error_taxonomy_timeout_stage', 'kv_cache_pressure_guard', 'model_artifact_load_readiness', 'streaming_reliability', 'cache_hit_key_governance', 'route_trace_correctness', 'tool_dependency_trace', 'quality_release_diff_readiness', 'inference_fault_diagnosis_gate']
inference_fault_diagnosis_gate_pass=False
```

这个 demo 的面试价值是把推理故障定位从“看 QPS 和 GPU utilization”升级为请求级证据链：TTFT 拆前半段，TPOT 拆 decode，p99 拆尾部切片，吞吐按 token 归一化，错误按类别和阶段归因，KV、cache、route、tool、模型版本和 release diff 都能落到同一个 trace 里。

## 46.21 常见误区

误区一：GPU 利用率高就是过载。

要结合 queue、TTFT、TPOT、KV cache 和错误率判断。

误区二：TTFT 高就是模型慢。

TTFT 可能来自网关、路由、排队、cache miss、prefill 或冷启动。

误区三：错误率上升就是 runtime bug。

可能是限流、权限、输入参数、工具、模型加载或安全策略。

误区四：吞吐下降只看 QPS。

要看 input/output tokens/s 和每 GPU tokens/s。

误区五：质量问题不属于故障。

模型输出质量、安全、工具调用和 RAG 引用错误都是生产故障的一部分。

## 46.22 面试常见追问

问题一：TTFT 突然升高怎么排查？

可以回答：拆分 gateway、auth、router、cache、queue、tokenizer、prefill 和 safety 阶段，重点看 queue wait、input tokens、prefill latency、KV cache、cold start 和流量变化。

问题二：TPOT 升高说明什么？

可以回答：通常说明 decode 阶段变慢，可能与 batch、KV cache、显存带宽、active sequences、runtime 版本、sampling 参数或 GPU 负载有关。

问题三：p99 升高但平均延迟没变怎么办？

可以回答：按模型、租户、endpoint、输入长度、输出长度切分，查看慢请求 trace，排查长请求、局部实例拥塞、cache miss、工具调用和网络抖动。

问题四：推理错误率上升如何分类？

可以回答：先按 4xx、429、5xx、timeout、OOM、model load failed、tool failed、safety blocked 分类，再分别定位权限、限流、runtime、资源、模型或下游依赖。

## 46.23 小练习

1. TTFT 和 TPOT 分别对应推理链路中的哪些阶段？
2. 为什么 p99 延迟比平均延迟更重要？
3. KV cache 不足会导致哪些现象？
4. 模型加载失败应检查哪些 artifact 和环境？
5. Cache 命中率下降可能由哪些发布变更导致？
6. Route trace 应该记录哪些信息？
7. Agent 工具调用失败如何排查？
8. 如何设计一个推理故障 runbook？

## 46.24 本章小结

本章讲了推理故障定位。

你需要记住：

1. 推理故障要先判断影响范围，再按请求链路拆阶段。
2. TTFT 主要受前置链路、排队和 prefill 影响，TPOT 主要受 decode 和 KV cache 影响。
3. p99、tokens/s、queue、KV cache、active sequences 和 route trace 比单纯 QPS 更关键。
4. 错误率要先分类，timeout 要记录发生阶段。
5. RAG/Agent 场景还要看 retrieval trace、prompt assembly trace 和 tool call trace。
6. 生产故障处理要结合限流、降级、回滚、扩容和摘除异常 endpoint。

下一章我们会讲 SLO、SLA、错误预算和生产值班体系。
