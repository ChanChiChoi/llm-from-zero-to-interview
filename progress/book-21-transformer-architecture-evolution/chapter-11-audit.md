# 第十一章《KV Cache、长上下文推理和显存增长问题》精修审计

审阅日期：2026-11-20。状态：章节职责审视、外部证据核验、正文重写、配套同步和静态验收完成。未运行代码、测试、模型、API、benchmark 或 PDF；GitHub MathJax 页面渲染、目标读者试读和第二十一册连续复读保留在后续门禁。

## 读者契约与章节边界

- **前置：** 理解第 1—10 章建立的 causal attention、Q/K/V、head 结构、位置、mask 和 attention 复杂度；不假设读者有 serving 框架经验。
- **离章结果：** 能跟踪一次 prefill 和增量 decode 中各层 K/V 的生成与复用；在明确维度、精度、请求长度和并发条件下估算显式 KV payload；区分模型如何表示/选择历史、serving 如何分配状态，以及哪些吞吐/延迟结论仍需具体环境测量。
- **章节边界：** 本章主讲 cache 语义、增量计算、容量账本与机制分层。MHA/MQA/GQA/MLA 的完整结构和转换证据在第 4 章；attention pair 与 prefill 计算由第 10 章承接；长距离有效检索见第 13 章；滑窗、PagedAttention 与 kernel/system 细节分别由第 19、20 章及专门的混合架构/cache 章节展开。第六册与第二十四册承接 serving 部署、并行实现和实际性能口径。

## 外部研究与讲解材料

技术来源、版本、实际阅读位置和限制记录在[KV Cache 容量、分页管理与压缩证据](../../research/kv-cache-capacity-paging-and-compression-2026.md)与[MHA/MQA/GQA/MLA 结构和证据边界](../../research/mha-mqa-gqa-mla-evidence-2026-11-19.md)。本轮复用的原始/官方证据包括 Shazeer 的 MQA 带宽动机、Ainslie 等人的 GQA 定义与论文实验范围、DeepSeek-V2 v5 的 MLA latent 与 decoupled RoPE key、Kwon 等人的 PagedAttention 方法与系统评测；论文速度及质量数据均未推广为普遍收益。

新增对照 Stanford MS&E 319 Autumn 2026 Lecture 5 讲义。讲义按单个新位置的 Q/K/V 更新、cache 等价条件、缓存与不缓存的复杂度、容量公式、窗口 cache 及跨层 receptive field 组织解释。它用于参考教学推导，不作为在线性能测量或普遍学习效果证据。课程原文、阅读范围与边界已补入研究记录。

| 外部主题 | 读者结果与证据边界 | 本章决定 |
| --- | --- | --- |
| Cache 等价与逐层更新 | 固定参数、旧 token、位置和可见规则下，因果模型的历史状态可复用；低层 per-position 运算的确定性是等价前提 | 以 prompt、首个输出和后续新增位置走查，并指出浮点实现会有舍入差异 |
| Prefill 与 decode 账本 | Prefill 并行处理 prompt 位置；decode 每个新 query 读取增长的历史。理论位置对数不等于完整层 FLOPs 或墙钟延迟 | 给出长度关系和仅计 query-key pair 的 decode 合计式，其他成本留给第 10 章及具体 profile |
| Cache payload | 全局显式 K/V payload 随各请求实际保留长度、层数、K/V head 数、维度和元素字节数变化 | 定义均匀全局 attention 公式、变量长度形式及 32 层/16 请求/8K 的 64 GiB 教学计算；说明被排除的内存项 |
| 架构改变历史表示 | MQA/GQA 改变 KV head 数；MLA 存 latent 与位置相关成分；windowed/recurrent layers 不能套统一显式 KV 公式 | 表格区分改变表示、直接可读历史和状态形状；不重复第 4 章的完整结构推导 |
| Serving 内存管理 | PagedAttention 用 blocks 与映射按需分配并支持前缀共享；历史论文结果绑定其模型、硬件和 baseline | 解释逻辑 payload 与物理分配的区别；共享时可避免重复，分页本身不缩小同一历史的数学 payload |
| 长上下文边界 | 输入长度声明不等于目标并发下的设备容量，也不证明任意位置检索有效 | 回链第 13 章；FlashAttention 与持久 KV 状态分开说明 |

## 正文与配套修改

- 将旧版 918 行改为单条主线：逐层 cache 更新 → prefill/decode → payload 估算 → decode 数据搬运 → 架构状态类型 → block 分配 → 长上下文容量/能力边界。
- 删除日期化资料清单、宽泛的近期模型 1M context 表、未充分定义的混合 history-state 公式、项目作者流程、重复 FAQ/练习，以及将固定 20/80 GiB、8% 浪费和带宽数值写成 pass/fail 的自拟成本门禁与 Python demo。具体模型披露回到本册的 KDA、hybrid/cache 和模型专题章节。
- 修正“KV cache 是高效推理必要条件”为“可关闭但会重算”；说明新增位置的 K/V 必须进入本次可见集合；把 MLA 描述为 latent 与位置相关 key 的 cache 设计，不暗示推理必须恢复完整 K/V 张量。
- 明确估算公式适用均匀、显式全局 K/V 且 K/V head dimension 和精度一致的条件；异长请求按实际长度求和。重算的 64 GiB 结果是 payload，不是整卡容量；按 GQA 8 KV heads 同条件估得 16 GiB，单请求单 token 为 128 KiB。
- 修正笼统的带宽瓶颈结论：说明负载、batch、结构、缓存层次、kernel 和权重访问均可能改变瓶颈；不得从公式直接外推吞吐或延迟。
- 同步[练习 31P](../../EXERCISES.md)为逐层 cache 轨迹与等价条件；[面试题 31N](../../INTERVIEW_BANK.md)更新容量公式、结构分工及长上下文边界；[项目路线](../../PROJECTS.md)将固定阈值 demo 改为带明确配置与 workload 的 KV Cache Capacity Brief；术语表将 KV Cache Gate 限定为由场景条件推导的项目检查项。
- 章节编号不变，无需修改简介或目录。

## 静态验收与遗留项

- H2 `11.1—11.7` 连续；7 个 `math` 围栏成对；72 个行内数学定界符成对，且未发现定界符与全角标点相邻；13 个本地 Markdown 链接目标存在；未发现控制字符或尾随空白。
- 复核缓存逐层顺序、decode pair 公式、64 GiB/16 GiB/128 KiB 计算、uniform-layer 假设、MLA/分页/滑窗边界，以及本地第 4、10、13、19、20、61、71 章链接。相关文件 `git diff --check` 通过。
- 未运行章节代码、自动化测试、模型、API、kernel、benchmark 或 PDF；未检查代理设置。外部核心论文/讲义已定向核读，但本地检查不等于全书外链可达性或目标平台公式渲染验收。
- 下一入口为第十二章《In-Context Learning 与显式 token 检索能力》；本册第 1—11 章已逐章精修，84 章待审。全套书精修 goal 继续 active，不处理 PDF 或 `plan.md`。
