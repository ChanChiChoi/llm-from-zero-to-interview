# 第六册第 7 章《量化与模型压缩》章级审计

审计日期：2026-11-02。状态：完成定向扫描、正文精修、配套同步与静态门禁。保留原有第 1—19 节及 14.1 编号；将“常见误判”收束为入口表，没有新增、删除或重排章节。第 15 节代码仅静态审阅，未运行；本轮没有运行测试、模型、benchmark 或 PDF，也没有检查代理设置。

## Scope 与章节职责

- **中心问题**：当目标服务受权重、activation、active KV、冷 KV、计算或质量约束时，怎样挑选压缩对象和数值表示，并通过校准、任务回归、kernel 与运行时门槛证明候选可部署。
- **本章职责**：建立压缩收益与误差的通用决策框架，沿 weight-only、activation、PTQ/QAT、GPTQ/AWQ/SmoothQuant、KV 量化、校准与质量验收解释机制；蒸馏、剪枝作为支持分支，提供小型代码演示和长上下文服务案例。
- **章节边界**：第 3 章主讲 active KV 容量、分页、复用和淘汰；本章只讲 KV 数值量化与质量/部署边界。第 15—17 章分别深讲 FP4 系列、FP8 KV 和 native INT4 artifact；本章给它们共同的评估坐标，不重复格式实现。第 5 章主讲 runtime 兼容性，第 2 章主讲指标口径，第 12 章主讲经济优化，第 7 册主讲评估方法；训练中的蒸馏路线交训练/小模型相关篇章。
- 简介和目录原有第 7 章摘要已准确说明“精度、质量、校准与部署性能的通用压缩决策框架”，并将第 15—17 章列作后续分支；本轮核对后一致，无需修改。

## 章级扫描与候选去向

- **日期、范围与资料类型**：2026-11-02。围绕低比特 weight/activation/KV 的误差目标、校准分布、scale/metadata、格式与 kernel 的配套关系、长上下文 cache 生命周期、质量分桶、候选方法演进及 2025—2026 新增方法定向检索。阅读原始论文/预印本的相关章节与实验、ICLR/ICML 会议入口及 NVIDIA、Transformers、vLLM 等官方动态文档；扫描限于本章主题，不声称穷尽全部量化方法或 runtime。
- **GPTQ、AWQ、SmoothQuant、LLM.int8()、ZeroQuant、KIVI 与 TurboQuant**：保留为现有主要路线，分别解释校准二阶误差补偿、activation-aware 通道缩放、activation outlier 迁移、混合精度旁路、动态/细粒度配置和 K/V cache 目标差异。论文结果按各自模型、数据、硬件和指标限定，不横向排序为通用速度结论。
- **SliderQuant**：纳入 §§14.1、18，作为按层敏感度与滑窗选择的近期方法案例。只阅读 ICLR 2026 官方会议摘要；其浅/深层及首末层敏感性为作者报告，不据此声称目标 runtime 已兼容，也不补写未阅读的算法实现细节。
- **AAAC**：纳入 §§14.1、18。阅读 arXiv v2 的相关方法/实验及 ICML 2026 AdaptFM workshop 页面；说明双 scalar codebook、activation-weighted group 选择，并限定 scale 与选择 bit 的共享条件。artifact 编码和 runtime 支持仍未知，不将论文结果写成部署事实。
- **D²Quant**：保留在研究记录为 abstract-only 候选；摘要不足以支撑本章需要的机制推演和实现边界，故不纳入正文。
- **InnerQ**：纳入 §9.3，用于说明 KV cache 内维分组、混合量化/高精度窗口与 fused dequant-GEMV 的研究路线。只把作者报告的 Jetson Xavier NX、batch 1、单层 microbenchmark 作为该实验设置的结果；预印本不同部分对 Llama 3.1/3.2-8B 标注不一致，已记录该分歧，不外推端到端 serving 收益。
- **TensorRT-LLM KV cache compression**：在 §9.2 区分 active KV 量化、NVFP4 cold-page 编码和 TriAttention token retention。访问日 2026-11-02 的动态官方文档把 cold-page 的 Host/Disk 迁移与返回 GPU 后的 runtime dtype、TriAttention 离线校准、PyTorch backend/KVCM V2/SM100 系列条件分别说明；这些是当前条件化能力，不推广为其他 runtime 的通用行为。
- **统一量化评估**：正文及参考资料更新到 arXiv v4（页面标记 2026-10-21 修订并被 Pattern Recognition 接收），用于解释变换、误差补偿、粒度、scale 精度与 MXFP4/NVFP4 结果的实验边界。
- **蒸馏与剪枝**：保留为压缩主题的支持性分支，突出“改变 student/权重结构”与“改变数值表示”的差异，链接到第二十一册小模型与结构改造主线及第五册 on-policy 蒸馏；不在本章展开训练 recipe，也不把稀疏率当作 kernel 加速。
- 新发现论文/文档的版本、实际阅读范围、结论支持范围与未核验边界仅记入 `research/weight-quantization-formats-and-ptq-2026.md`、`research/kv-cache-capacity-paging-and-compression-2026.md`；本记录只记候选去向、编辑决定和剩余任务。

## 篇幅、例子与代码状态

解释性汉字静态计数口径：排除标题、表格、代码围栏、公式、行内代码、Markdown 链接文本、状态标记与 URL，只计中文汉字。核心单元下限 1,800，辅助单元下限 1,000。

| 教学单元 | 汉字数 | 类型 | 下限 | 状态与合并理由 |
| --- | ---: | --- | ---: | --- |
| §§1—2 瓶颈预算与数值表示 | 1,925 | 核心连续链 | 1,800 | 通过；从容量/性能约束推进到量化格式、映射误差和选择对象。 |
| §§3—5 weight-only、activation、PTQ/QAT | 2,156 | 核心连续链 | 1,800 | 通过；按“量化什么，再何时适配”递进，解释静态权重、动态输入与训练/校准假设。 |
| §§6—8 GPTQ、AWQ、SmoothQuant | 2,182 | 核心连续比较链 | 1,800 | 通过；三节共同回答不完整重训练时如何利用激活/变换控制低比特误差，节首说明机制递进。 |
| §9 KV Cache quantization | 2,046 | 核心 | 1,800 | 通过；单独处理 cache 数值误差、活跃/冷数据路径和性能结论的证据边界。 |
| §§10—11 校准与质量评估 | 1,893 | 核心连续链 | 1,800 | 通过；由校准覆盖推进至分桶回归、质量门槛和 runtime 观测。 |
| §12 Distillation | 1,078 | 辅助分支 | 1,000 | 通过；用于比较改变 student 与改变表示，不承担训练主线，给出跨册主讲入口。 |
| §13 剪枝和稀疏化 | 1,010 | 辅助分支 | 1,000 | 通过；聚焦稀疏率、artifact 与真实 kernel 的区别，并链接结构改造主线。 |
| §§14—14.1 选型与下一轮实验 | 1,855 | 核心连续决策链 | 1,800 | 通过；从 profile 信号选择压缩对象，经门槛/假设比较推进到元数据成本和实验 go/no-go，是本章中心问题的收束。 |
| §15 最小 Python demo | 1,004 | 辅助代码演示 | 1,000 | 通过；复用 §§2.4、4、8 的输入/手算结果，展示映射和存储账本，不独立引入新算法。 |
| §17 长上下文服务决策案例 | 1,888 | 核心综合案例 | 1,800 | 通过；作为完整工作案例，增加联合长度分布、显存 headroom、配对回放、结果分支和上线门槛推演。 |
| §16 常见误判、§§18—19 | 不单独计数 | 入口表、参考资料与收束 | 不适用 | 不承担独立教学；§16 表格只链接前文已讲机制。 |

手算与状态案例覆盖：7B 参数 FP16/INT8/INT4 payload；8 个权重的对称 INT4 与 group-wise scale/MSE；非对称 INT8 activation 的 scale、zero point、整数码与反量化；outlier 改变统一刻度；校准输入方向放大线性层误差；group scale 数量；W4A16 从转换 artifact 到 loader/kernel/fallback 的数据路径；KIVI K/V 的 residual 窗口状态；分桶加权成功率 97.45% 对 96.4%（长文档桶下降 7 个百分点，全部标明为合成教学输入）；三 token 蒸馏 KL；稀疏矩阵的 60% 稀疏率与 dense storage/kernel 边界；7B group-size scale 约 0.10 GiB 的估算；§17 中 131,072 bytes/token、8,704 token/请求、16 个请求的 17 GiB 理论 KV 与 24 GiB 卡的容量门槛。假设参数、合成数字和作者报告的测量严格区分。

- §15 Python 代码只用标准库；本轮对照函数输入、INT4 scale/整数码/MSE、activation zero point、常量输入退化分支、SmoothQuant 幅值和 7B storage 输出，逐步手算预期值。代码附近标注“静态审阅，未运行（2026-11-02）”，输出写为“手算预期输出（未执行）”。它不执行真实矩阵乘、模型导出、GPU kernel 或模型质量评估；Python `round` 与目标库舍入、packing、scale 布局可能不同。
- 公式静态状态：23 对围栏（46 个 fence marker）；行内单美元标记 106 个、全部成对；math fence 花括号净差为 0。禁用 `\operatorname`、`\(` / `\)`、`\[` / `\]`、尖括号下标、`\text` 内代码下划线及无反斜线 `qquad` 检查均为零命中。数例按正文精度手算，未用代码替代。
- 链接状态：76 个 Markdown 链接；18 个相对本地链接目标均存在（含新增的第二十一册跨册入口），无失效本地文件目标。外链已阅读与引用的关键来源见 `research/` 记录，但本章全部外链未逐条在线可达性复查。
- 图示机会：保留 KV cache 生命周期状态表、方法/格式对照、实验决策表和评估分桶表；请求/权重/cache 数据流已用连续步骤讲解，表格均在正文引出并有读法说明，没有为数量新增 Mermaid 图。

## 配套同步

- 第六册简介、目录的第 7 章短摘要与通用框架职责一致，本轮核对后不改。
- `PAPERS.md` 更新统一评估到 arXiv v4，新增 SliderQuant、AAAC、InnerQ 和 TensorRT-LLM 文档入口。
- `GLOSSARY_EN_ZH.md` 增加 InnerQ、SliderQuant、AAAC、TriAttention；Active KV 与 Cold-Page 已有词条。
- `EXERCISES.md` 新增第六册量化练习 26，对照 active KV quantization、cold-page compression 与 TriAttention 的生命周期和观测指标。
- `INTERVIEW_BANK.md` 新增 40ZZZZL20D（SliderQuant/AAAC 方法差异）及 40ZZZZL20E（active KV、cold-page、TriAttention）。
- `KNOWLEDGE_GRAPH.md` 增加压缩决策链，连接权重/activation/KV、质量验收和部署路径。

## 遗留边界与交接

- 第六册连续复读中复算 §4 非对称 INT8 手算例，发现 `2.5` 的反量化值应为约 `2.5020`，原文 `2.5098` 为算术笔误，已修正。量化整数、scale、zero point、MSE 与 §15 手算预期输出未变；没有修改代码，也未运行代码。
- 章节代码仍为静态审阅、未运行；没有目标 GPU/runtime 的模型质量、kernel、延迟、吞吐、显存峰值或灰度数据。代码语句或预期输出若后续改变，需按规范判断实际受影响的代码块；仅改正文不触发重跑。
- SliderQuant 仅核对官方会议摘要，未读全文或代码；D²Quant 仅读摘要并暂不纳入正文。InnerQ 的模型版本标记冲突保留为来源边界。TensorRT-LLM 文档动态变化，实际部署前须固定目标 release/revision 与 backend/hardware。
- 全部外链未在线逐一检查；公开论文与厂商文档的实验/兼容性不代替目标 workload 回放。本章未更新 PDF。
- 第六册连续复读已覆盖简介、目录和第 1—7 章；确认第 7 章通用量化框架与第 15—17 章格式/部署专题入口清楚。下一步按顺序复读第 8 章《并行推理与多 GPU 服务》，继续核对量化 artifact/KV 状态与 TP/PP/EP、cache transfer 的边界；第 8—18 章、全册复读及最终门禁均未完成，全套书 goal 继续 active。
