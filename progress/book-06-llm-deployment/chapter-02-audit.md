# 第六册第 2 章《推理基础》章级审计

审计日期：2026-11-02。状态：完成章级扫描、正文精修、配套同步与静态门禁；待整册连续复读时再次核对前后章节交接。维护决定只记在 progress/；联网发现的技术事实和来源证据记在 research/。

## Scope 与中心问题

- 完整审阅 book-06-llm-deployment/chapters/02-推理基础.md，保留原有 2.1—2.19 节和文件入口。
- 中心问题：一次 decoder-only 请求怎样从 prompt 前向走到流式 token；不同延迟指标具体计到哪里、按什么分母聚合；如何结合计算/数据移动估算与 trace 证据把性能诊断推进到容量决策。
- 章节职责与第一章部署决策地图一致：本章统一推理生命周期、性能账本和测量方法；KV 分页与回收、采样策略、调度器和生产观测留给后续主讲章节。

## 章级扫描与内容去向

- **扫描日期与范围**：2026-11-02；聚焦首 token 的阶段归属、TTFT/TPOT/ITL 的采样边界、roofline 的解释范围、压测到达模型和容量决策。核对 2023—2026 serving 论文、固定版本 runtime benchmark 文档及官方 profiler 文档；不是对推理服务研究的穷尽检索。
- **Prefill 与首 token**：Sarathi-Serve 和 Splitwise 描述 prompt computation/prefill 产生首个输出 token，后续 token 才进入 decode。正文删除“首 token 通常还要一次完整 decode 前向”的错误描述，改为常见 decoder-only 路径从最后 prompt 位置 logits 采样首 token，并注明 runtime span/采样计时边界可能不同。证据范围见 [推理阶段与指标研究记录](../../research/inference-serving-metrics-and-disaggregation-2026.md)。
- **TPOT 与 ITL**：vLLM v0.32.0 固定版 CLI 分别说明客户端 TTFT、成功请求内的相邻输出事件 ITL、按请求计算的 (E2E - TTFT)/(output_tokens - 1) TPOT，以及各自的聚合方式。正文区分事件间隔、请求等权平均和响应完成尾部，并说明多 token 单事件、单 token 请求和尾部时间的处理必须跟随具体工具定义。
- **来源分歧**：TensorRT-LLM 动态 benchmark guide 对 TPOT request-average 的文字说明，与其 ITL 聚合公式中的输出 token 分母和“相邻间隔”样本数存在表述歧义。未把该公式作为通用定义，也未在章节中引用为已确定口径；分歧和待核验边界留在上述 research 记录。
- **Roofline 与 profiler**：保留 kernel/明确阶段级 roofline 估算，不据其断言整个服务瓶颈；补充 Nsight Compute 的采集开销会随指标、section、kernel 数和架构变化，profile 诊断与 headline benchmark 分开。
- **workload 和容量**：在压测步骤中要求保留输入/输出长度联合分布、到达时间和 burst 结构；P95 只是摘要。并发曲线写成一种可能测量结果，容量案例将 70% 利用率目标与一台故障冗余明确列为策略输入，不写成通用常数。以上扩写均纳入本章已有推理测量—压测—容量主线，不新增章节。
- **研究记录**：首 token 阶段边界、vLLM v0.32.0 指标、TensorRT-LLM 文档歧义及 Nsight roofline/overhead 的来源、版本、阅读位置和证据边界已写入 [research 记录](../../research/inference-serving-metrics-and-disaggregation-2026.md)。

## 教学链与篇幅

按连续论证而非每个小标题自动拆分独立单元；计数为解释性汉字静态计数，排除标题、表格、公式、代码、引用清单和状态标记：

| 单元 | 类型 | 汉字数 | 下限 | 判断 |
| --- | --- | ---: | ---: | --- |
| 章首及 §§2.1—2.4：条件生成、请求生命周期、prefill/decode 与 KV 状态 | 核心 | 2,665 | 1,800 | 通过；用请求状态与资源账本连接两个阶段 |
| §§2.5—2.7：计时起止、TPOT/ITL、吞吐及聚合口径 | 核心 | 2,977 | 1,800 | 通过；以同一组请求数例比较事件权重与请求权重 |
| §§2.8—2.10：roofline、优化谱系与 trace 诊断 | 核心 | 1,945 | 1,800 | 通过；由理想上界回到 kernel/服务级证据边界 |
| §§2.11—2.13：训练/推理对照、workload 与请求状态 | 支持过渡 | 526 | 不独立计数 | 不单独承担完整机制教学；为后续测量案例提供术语和输入 |
| §§2.14—2.17：指标 demo、误区、压测设计与容量决策 | 核心 | 2,852 | 1,800 | 通过；构成从 toy 计算、测量协议到容量/故障决策的连续应用链 |
| §§2.18—2.19：资料、证据边界与总结 | 导航/收束 | 不适用 | 不适用 | 不作为独立知识讲解单元 |

## 例子与复算状态

- **KV 状态账本（§2.4）**：假设 L=32、H_kv=8、D_h=128、BF16/FP16 每值 2 bytes、上下文 8,192 token，逐项代入 2L × T_ctx × H_kv × D_h × b 得 1,073,741,824 bytes = 1,024 MiB；MHA 的 32 个 KV head 对应约 4 GiB。数值为结构示例，不代表可用显存配置。
- **Roofline（§2.8）**：用 P_peak=120 TFLOP/s、BW=1.6 TB/s 手算 ridge point 75 FLOPs/byte；对比短上下文不同 batch 及长上下文 KV 流量，明确结果是理想下界而非目标 GPU 测量。
- **请求 trace（§2.9）**：长 RAG 假设 queue 40 ms、prefill 260 ms、输出 100 token、后续间隔 18 ms，手算 TTFT 300 ms、E2E 2,082 ms；短问答 queue 10 ms、prefill 35 ms、输出 40 token、间隔 12 ms，手算 TTFT 45 ms、E2E 513 ms。假设项、单位和首 token/后续间隔的分界均有说明。
- **零依赖 demo（§2.14）**：手算 short_chat 的最终上下文 320 token、KV proxy 40 MiB、TTFT 45.35 ms、TPOT 8.38 ms、E2E 583.29 ms；手算 long_rag 的上下文 2,144 token、KV proxy 268 MiB、TTFT 159.85 ms、TPOT 14.23 ms、E2E 1,521.51 ms。8 条假设请求的输出总量 600 token、cohort wall time 2,484.8 ms，输出吞吐 241.47 token/s、请求速率 3.22 request/s；线性插值 P95 的零起点位置为 6.65，对应约 2,147.6 ms。脚本中的趋势系数和尾部均为合成教学输入，不拟合 GPU。
- **容量决策（§2.17）**：在单副本目标负载实测 3,200 output token/s、峰值窗口需求 7,200 token/s 的明确假设下，按 ceil(7200/(0.70×3200))+1 得 5 副本；70% 余量和单副本故障冗余是策略假设。

## 公式、代码、链接与视觉状态

- **公式**：静态扫描 41 个 math 围栏，花括号平衡；单美元行内公式分隔符无未配对项。禁用的 \operatorname、尖括号下标、\text 中带下划线代码变量及缺失反斜线的 qquad 检查无命中。核心公式均有变量边界、单位/条件和对应手算数例。
- **代码**：仅 §2.14 有一段独立 Python 教学 demo。代码状态为“静态审阅，未运行（2026-11-02）”；旧正文“已验证”标记没有可追溯运行记录。本轮没有运行它、章节代码、测试、模型或 benchmark；代码块后展示的结果是手算预期值。demo 仅覆盖合成的正输入/输出成功请求，失败、取消、超时和零输出的真实统计合同不在代码覆盖范围。
- **围栏**：94 个 ~~~ 标记配为 47 对，其中 math 41、text 5、python 1；无未闭合围栏。
- **链接**：28 个 Markdown 链接均为外链，无本地目标；本轮没有全量在线复核外链可达性。动态 Transformers/SGLang 文档仅用于一般接口/能力入口，正文已提醒固定目标版本；核心 TPOT/ITL 事实链接到固定 vLLM v0.32.0 页面。
- **视觉**：保留线性请求生命周期示意、阶段/方案对比表、手算轨迹和容量约束式；连续请求路径不需要再增加装饰性 Mermaid 图。目标平台 Mermaid 渲染不适用。
- **补充资料**：简介与目录的第二章短摘要已按修订内容同步。GLOSSARY_EN_ZH.md 已更新 TPOT、ITL 与 request-level TPOT 的口径提示；现有推理优化练习已覆盖单事件多 token 时 event-level ITL 与 request-level TPOT 的分母比较，故本轮不重复粘贴题目。PAPERS.md 和 KNOWLEDGE_GRAPH.md 已有主要 serving 指标/论文入口，无需复制同一来源清单。
- git diff --check 通过；外链全量访问性和目标硬件实测仍不在本轮范围。

## 遗留风险与交接

- §2.14 教学代码仍未运行；若未来修改可执行语句或 API/输入输出契约，应按模块 04 第 6.3 节重新验证。当前不能把手算预期写成程序实测。
- 客户端事件时间、服务端 span 和框架 histogram 的边界随具体版本/配置变化；引用工具定义不能替代部署版本核对。容量和 roofline 的假设值不能转述为生产建议或硬件 benchmark。
- 本轮未构建或更新 PDF。第 2 章完成后，后续册级连续复读仍要复核第 1→2→3 章衔接以及第 2 章向第 4、5、6、8、11、12 章移交的指标定义。
- **下一步**：开始第六册第 3 章《KV Cache 与内存管理》的定向审视；全套书精修继续进行，goal 保持 active。
