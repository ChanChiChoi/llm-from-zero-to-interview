# 第十五册第十四章精修审计：Native Multimodal Token Budget

审计日期：2026-11-12

## scope

- 正文：[第十四章《Native Multimodal 的有效上下文账本》](../../book-15-multimodal-generative-models/chapters/14-native-multimodal-token-budget.md)，覆盖 §§14.0–14.18.3。
- 本章职责是把 processor 展开、图像/音频/视频、协议、工具历史、输出预留、缓存和任务质量放入请求级预算；第十章讲架构 taxonomy，第十三章讲前端到共享序列的数据路径。
- 保留十四章骨架和第十四章专题定位。简介与目录复核后保持不变；未更新 PDF 或 plan.md。

## chapter_scan：候选与决定

- 2026-11-12 完成全文审阅与定向证据核验。范围包括文本/媒体 token 记账、processor 输入协议、上下文声明与有效能力、图像分辨率和视频采样、音频 codec rate、动态分配、长文档/长视频覆盖、attention/KV 成本、缓存层次、超限回退、质量—成本约束及托管 API 字段。
- 核对 Transformers v5.21.0 多模态模板文档；Qwen3-VL Technical Report v2、Qwen3-Omni Technical Report v1、Qwen3.5-Omni Technical Report v2；Alibaba Cloud Qwen3.7 Plus 和 Context Cache 动态托管文档（访问日期 2026-11-12）；LVOmniBench arXiv v1。搜索摘要只用于定位，涉及正文的关键数字回到论文或官方页面核验。

| 候选问题 | 正文决定 | 证据与边界 |
|---|---|---|
| 媒体 token 数常由通用 patch/帧公式代替真实 processor 数 | 保留 patch、codec rate 和帧网格作为带假设的教学估算；要求以目标 processor 实际输出替换。 | Transformers 文档支持模板和 processor 的输入接口，不支持跨模型通用 token 规则。 |
| context window 被误当成已保留或已使用证据 | 明确 accepted、retained、used 三个任务事件；引用 LVOmniBench 作为长音视频记忆、时间定位和细粒度任务的评测例。 | benchmark 结果绑定其模型、采样和评分协议；不从输入长度推导有效能力。 |
| 位置元数据、视觉特征注入和 token 数混在一个预算项 | Qwen3-VL 案例区分显式 timestamp 文本/视觉 token、位置 ID 和 DeepStack 特征注入。 | 报告公开的是 Qwen3-VL 指定架构；DeepStack 不应自动算成额外文本 token，但目标实现可能有独立计算/激活成本。 |
| token rate 和音频首包延迟被当成同一单位 | Qwen3-Omni 案例分别记录 12.5 Hz/约 80 ms 表示步长与论文的理论首包延迟。 | 234/547 ms 为报告单并发音频/视频设置；4/6 并发另有延迟数，绑定 vLLM、torch.compile、CUDA Graph 和报告环境。删除没有在所引原始报告中核实的“66K context”表述。 |
| 模型产品能力被扩写为内部架构保证 | Qwen3.7 Plus 案例引用托管页列出的模态、snapshot、context/input/output 字段和 region/scope capability，并区分宿主 action executor。 | 页面是动态 API/产品合同；不同长度字段不可机械相加。GUI 产品定位不证明内部 policy、权限、设备动作成功率或质量。 |
| provider cache 被混同为 GPU KV 或应用 memory | 区分媒体表示缓存、完整 KV 和 provider-managed prefix cache；补充命中不保证发生。 | Alibaba Context Cache 文档按访问日支持公开行为；服务端内部 key 和实现未披露。应用自建 key 是建议，不写成 provider 实现。 |
| 单一降级 token 总量被当成足以答复 | 预算代码改为输出候选计划和 coverage warning，工具历史摘要未验证时明确阻止直接答复。 | 代码只验证合成账本，不构造真实摘要、媒体覆盖、质量、授权或模型输出。 |
| 近期模型新增功能可能扩大本册范围 | 不新增章节、不列跨 provider 价格或排名；只取能说明请求字段和证据边界的实例。 | 产品字段时效性高，未读资料和目标 endpoint 行为不外推。 |

## length_status

按连续中文解释性正文计数；不含标题、代码、表格、公式、引用清单、行内代码和 URL。

| 单元 | 类型 | 汉字数 | 下限 | 结果 |
|---|---|---:|---:|---|
| §§14.0–14.15：预算组成、processor、有效上下文、分配、覆盖、缓存、质量曲线和案例 | 核心 | 5,540 | 1,800 | 通过 |
| §14.16：独立预算审计 demo 的输入/输出、逐项算例和限制 | 辅助 | 1,168 | 1,000 | 通过 |
| §§14.18–14.18.3：三种模型资料如何映射到预算证据 | 辅助 | 1,343 | 1,000 | 通过 |

## example_status 与 formula_status

- §§14.1–14.2：明确同一账本不可重复计数；四图、音频、视频、协议、工具历史和输出预留合计 40,664 token，超过教学上限 32,768。所有输入都是合成假设，理想 patch 结果不冒充 processor/API 用量。
- §14.5：patch 14 下 672×672 为 2,304 个理想 patch，边长翻倍到 1,344 后为 9,216（4 倍）。
- §14.6：媒体预算 12,000、任务权重 3:2:1、成本权重相同时，示例分为 6,000/4,000/2,000；明示分配权重不是质量保证。
- §§14.7、14.14：20 分钟视频 2 FPS 为 2,400 帧；30 秒窗口 8 FPS 为 240 帧。采样间隔 0.5 秒、事件持续 0.2 秒、均匀相位和理想瞬时采样假设下，示例采中概率 0.4、漏采概率 0.6；不外推为系统召回率。
- §14.8：32 层、4096 token、8 KV 头、head dim 128、FP16 每元素 2 bytes，KV 代理为 536,870,912 bytes，约 512 MiB；序列长度翻倍后 KV 代理翻倍，稠密 attention score-cell 代理变为 4 倍。排除权重、激活、其他请求和运行时开销。
- §14.11：合成两配置在质量/延迟/成本/证据率门槛下筛选；100 次、总成本 2 美元、80 次同时通过正确性/证据/安全门槛时，单位成功成本为 0.025 美元。均非服务测量。
- §14.13：2048×1536、patch=32 为 3,072 patch；512×384 缩略图加三块 512×512 crop 为 960，教学代理减少 68.75%；保留漏定位时扩大区域/转人工的边界。
- §14.16：fallback 总量 20,976 token；patch 改为 28 时完整账本 33,752，仍比 32,768 多 984。代码与手算对应。
- 30 个 display-math 块花括号平衡；无 operatorname 命令、无行内美元公式。KV、分配、优化、成本、回退均有可复算数例。

## code_status

- §14.16 唯一独立 Python 标准库 demo 在 Python 3.12.9 执行；用标准库从 Markdown 提取代码围栏并编译/运行，实际 stdout 与正文参考输出逐行一致。
- 12 项输入/数值边界检查通过，覆盖 bool、NaN、正负无穷、负时长、十进制 0.1×50 的取整、超大整数、1e308 有限浮点、错误样本/配置类型、错误 detail 类型及有效配置；额外验证 patch=28 时图像代理 2,304、总量 33,752。
- code gate 输出 candidate_only；fallback 明示低分辨率小字、短视频事件、音频窗口和未验证工具摘要风险。它不执行 processor、摘要、模型、媒体、缓存、GPU、GUI、质量或权限验证。
- 章节状态标记为“已验证（2026-11-12；Python 3.12.9；仅标准库）”。后续只改解释文字，未改已执行语句。

## visual_status 与 link_status

- 一张 Markdown 表对照合成预算配置的质量、延迟、成本和证据支持率；流程步骤由有序列表、数据流由账本公式和覆盖报告解释，未额外添加 Mermaid。
- 14 个 Markdown 链接均为外链；无本地链接断裂。未做全量外链 HTTP crawl。
- 30 组波浪围栏配对；5 组反引号围栏配对；对比表四行列结构一致。未在 GitHub 页面验证 MathJax 或表格渲染。

## sync_status 与 open_risks

- 第十五册简介的第十四章承诺和目录职责与精修后内容一致，均复核后保持不变。
- research/model-update-2026-09/qwen3.7-plus-source-notes.md 新增访问日的官方 Qwen3.7 Plus / Context Cache 事实、来源和边界；此文件也修复 PAPERS.md 中指向该来源笔记的既有入口。Qwen3-VL、Qwen3-Omni、LVOmniBench 与 Transformers 来源直接引向版本化论文/官方页面。
- 公式和 token 数都是教学代理；processor 实际输出、hosted API 命中、GUI 操作、质量、延迟和真实单位成本未在本地测量。Qwen3.7 Plus 与 Context Cache 页面为动态文档，使用时需复核当前 region/scope/endpoint/mode。
- 按简介主路径完成第 1—14 章整册连续复读。本轮重新通读 §§14.0–14.18.3，并对照第 3、4、8—10、13 章确认 processor、媒体表示、架构 taxonomy、序列布局和请求预算的职责交接清楚；没有发现新的结构、顺序、导航或简介摘要问题。
- 本轮未修改第十四章正文或代码；沿用 2026-11-12 的 Python 3.12.9 执行验证及 12 项边界检查，不重复运行。简介与目录复核后保持不变。
- 第十五册最终门禁仍未关闭：第十三章 Mermaid 的 GitHub 页面打开返回 cache miss；本地没有浏览器或 Mermaid 渲染器，且 GitHub 默认分支不是当前未提交稿，故不能据源码或旧版页面宣称目标平台渲染通过。待可检查当前版本的 GitHub 渲染时复核。未更新 PDF 或 `plan.md`。

## 下一步

- 第十四章正文和章级门禁完成，整册连续复读第 1—14 章完成。第十六册架构审视已完成，下一入口为第一章《Reasoning 总览》；第十五册保留一项 GitHub Mermaid 目标平台渲染门禁待关闭。按用户要求不更新 PDF 或 `plan.md`。
