# 第 46 章 推理故障定位：TTFT、TPOT、吞吐下降和错误率上升

上一章讲了训练故障定位。本章讲推理故障定位。

推理服务是在线系统，用户直接感知延迟、失败和质量变化。训练任务可以排队和重试，线上推理不行。推理故障定位的关键，是把端到端请求拆成阶段，并结合模型、runtime、路由、缓存、GPU、KV cache、网络和业务流量一起分析。

先记住一句话：

> 推理故障定位不要只看 QPS 和 GPU 利用率，而要按 gateway、router、queue、prefill、decode、streaming、cache、runtime 和下游依赖逐段拆解。

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

## 46.20 常见误区

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

## 46.21 面试常见追问

问题一：TTFT 突然升高怎么排查？

可以回答：拆分 gateway、auth、router、cache、queue、tokenizer、prefill 和 safety 阶段，重点看 queue wait、input tokens、prefill latency、KV cache、cold start 和流量变化。

问题二：TPOT 升高说明什么？

可以回答：通常说明 decode 阶段变慢，可能与 batch、KV cache、显存带宽、active sequences、runtime 版本、sampling 参数或 GPU 负载有关。

问题三：p99 升高但平均延迟没变怎么办？

可以回答：按模型、租户、endpoint、输入长度、输出长度切分，查看慢请求 trace，排查长请求、局部实例拥塞、cache miss、工具调用和网络抖动。

问题四：推理错误率上升如何分类？

可以回答：先按 4xx、429、5xx、timeout、OOM、model load failed、tool failed、safety blocked 分类，再分别定位权限、限流、runtime、资源、模型或下游依赖。

## 46.22 小练习

1. TTFT 和 TPOT 分别对应推理链路中的哪些阶段？
2. 为什么 p99 延迟比平均延迟更重要？
3. KV cache 不足会导致哪些现象？
4. 模型加载失败应检查哪些 artifact 和环境？
5. Cache 命中率下降可能由哪些发布变更导致？
6. Route trace 应该记录哪些信息？
7. Agent 工具调用失败如何排查？
8. 如何设计一个推理故障 runbook？

## 46.23 本章小结

本章讲了推理故障定位。

你需要记住：

1. 推理故障要先判断影响范围，再按请求链路拆阶段。
2. TTFT 主要受前置链路、排队和 prefill 影响，TPOT 主要受 decode 和 KV cache 影响。
3. p99、tokens/s、queue、KV cache、active sequences 和 route trace 比单纯 QPS 更关键。
4. 错误率要先分类，timeout 要记录发生阶段。
5. RAG/Agent 场景还要看 retrieval trace、prompt assembly trace 和 tool call trace。
6. 生产故障处理要结合限流、降级、回滚、扩容和摘除异常 endpoint。

下一章我们会讲 SLO、SLA、错误预算和生产值班体系。
