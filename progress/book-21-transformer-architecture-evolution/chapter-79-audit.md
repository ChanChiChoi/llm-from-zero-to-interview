# 第七十九章精修审计

日期：2026-11-25。范围：第 79 章大纲核对、正文主题思想不变重写、读者/设计目录及相关配套资料同步；不处理 PDF。

## 旧稿诊断与写作决定

旧稿把模型结构、reasoning_effort、EAGLE 和 NVFP4 并列成部署功能清单，没有解释 Mistral4 文本主干的 MLA 缓存表示；把模型卡的 6.5B active 当成发布口径唯一答案，未处理发布文章的 6B 与含嵌入/输出层 8B 说法；用未核验的请求策略代码和简化成本式代替对工作负载与解码机制的说明。旧稿把 EAGLE 与量化合并描述成可尝试叠加，却未区分 verifier 分布、量化误差及制品/后端兼容；还扩展多模态、tools、API 与许可证，稀释本册的架构责任，正文标题也与目录标题不一致。

保留 Mistral Small 4 作为模型架构案例，按公开对象分层、固定配置中的 MLA/MoE、参数和 cache 资源账本、请求模式、EAGLE 机制与 artifact、NVFP4 表示与制品边界、部署对照组织。通过 EAGLE 的两 token 离散例子讲清接受质量和拒绝残差如何恢复当前 verifier 的分布；把该保证限定为目标模型实际使用的分布。没有 Mistral-specific 证据时，不给 EAGLE 制品推定代际，不声称 EAGLE+NVFP4 已有兼容或联合性能结果。

## 研究依据

Mistral 官方发布、模型文档与固定模型卡/config revision 支持发布主张、公开上下文及结构字段；Transformers 引入 Mistral4 的固定实现支持指定实现中的 MLA cache 与共享专家数据流。EAGLE ICML/PMLR 原论文与 EAGLE-3 论文用于区分方法代际、树验证和校正规则；Stanford 课程讲义与 NVIDIA 技术文章用于教学解释。NVIDIA Transformer Engine 2.19 只定义通用 NVFP4 层级格式；Mistral 固定 NVFP4 model-card revision 支持制品来源描述，但没有完整逐层 recipe。

模型发布文章写 6B active 及含 embedding/output 时 8B，模型卡和产品页写约 6.5B activated/token；差异未由可见资料对账。模型卡/产品页标称 256K，固定配置的位置字段为 1,048,576；字段本身不证明 1M 有效上下文。详细来源、revision、推导和边界见[第 79 章研究记录](../../research/mistral-small-4-eagle-nvfp4-2026-11-25.md)。

## 本轮改动

- 完整重写[第 79 章正文](../../book-21-transformer-architecture-evolution/chapters/79-mistral-small-4混合推理与eagle量化.md)：将请求配置与模型结构分开；补入 Mistral3 外层/Mistral4 文本配置边界、MoE routed/shared expert 账本、指定 Transformers MLA cache 形状、按公开窗口估算的 KV 字节数、EAGLE 接受/拒绝校正及缓存回退、NVFP4 尺度层级和分阶段评估方法。移除未验证策略代码和与本册架构职责不符的 API、工具及许可证展开。
- 建立完整的[第 79 章结构大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/79-Mistral-Small-4混合推理与EAGLE-NVFP4部署.md)，并修正 cache 标题的粒度为“每层每 token 320 个数值”。
- 同步[读者目录](../../book-21-transformer-architecture-evolution/目录.md)和[设计目录](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/目录.md)的 79.1—79.8、30 个三级标题；设计目录增加章节大纲链接。
- 修订 `PAPERS.md` 对模型卡、EAGLE head 和 NVFP4 checkpoint 的索引边界；更新相关三道练习、三道面试题及知识图谱中的 MLA cache、active 参数分歧、EAGLE verifier 和配对验证节点。
- 更新本册[架构审视入口](architecture-audit.md)，将第 80 章设为下一入口；第 78 章原有渲染待验状态保留。

## 静态验收与遗留项

- 正文和章节大纲各有 8 个 H2、30 个 H3；两份目录也各登记 8 个 H2、30 个 H3，标题文字、层级和顺序与正文/大纲一致。
- 8 个顶层 `math` 围栏闭合，行内美元定界符逐行平衡；未发现 `\(`、`\)` 或 `\operatorname`。手工核对 cache 账本：每层每 token 320 values，36 层 BF16 为 22.5 KiB/token；262,144 token 约 5.625 GiB；显式 K/V 元素对照为 25.6 倍。EAGLE 教学分布的接受质量与拒绝残差相加后恢复目标分布；4-bit 权重值明确标为 payload 下界。
- 第 79 章正文、章节大纲、两份目录、新研究记录及本轮配套入口的本地链接检查无失效项。全量 `PAPERS.md` 链接扫描另发现 23 个既有失效本地引用，均指向其他主题的旧 research/model-update 文件或代码，不由本章新增；后续按共享索引维护范围处理。
- `git diff --check` 通过。未新增可执行代码；未运行测试、模型、API、kernel 或 benchmark，未构建或更新 PDF，未改 `plan.md`。
- GitHub MathJax 页面未实际查看，公式视觉验收仍待完成；模型制品未下载，EAGLE/NVFP4 联合运行、接受率和量化质量均未实测。外链终审、目标读者试读及第二十一册连续复读仍属于全册门禁。

下一章为第 80 章《Step 3.5 Flash：MTP-3、滑动窗口和 11B 激活参数》。第二十一册第 1—79 章已完成首轮正文审阅与修订，第 80—95 章共 16 章待精修；全套书精修 goal 继续 active。

