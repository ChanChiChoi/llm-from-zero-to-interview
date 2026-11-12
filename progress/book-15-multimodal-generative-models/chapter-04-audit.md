# 第十五册第四章精修审计：VLM 架构

审阅日期：2026-11-11

状态：本章精修、章级扫描和正文门禁通过；2026-11-12 整册复读时补齐计数类型合同并复验代码；继续第五章整册复读。按用户要求不更新 PDF。

目标章节：[第四章《VLM 架构》](../../book-15-multimodal-generative-models/chapters/04-vlm架构.md)

## scope 与章节结构

- 全文复读 §§0–15，保留整篇为一个连续核心讲解单元；没有增删章节或改变第十五册十四章骨架。
- 主问题是视觉信息如何进入语言模型的生成路径，以及 prefix、cross-attention、Q-Former、Perceiver Resampler 如何改变信息访问、序列长度、训练责任与成本。
- 承接第三章的视觉表示输出；把图文样本与多模态监督细节交给第五章，把模型级统一路线交给第十章，把媒体请求预算的跨模型账本交给第十四章。

## chapter_scan：范围、证据与去向

- **日期与范围：** 2026-11-11；检查视觉连接器路线、视觉 memory 的跨层读取、压缩的长度与成本边界、多图映射和监督 mask。资料类型为原始论文、已有研究记录和论文索引；不穷尽全部 VLM 变体、商业 API 或生产 serving 实现。
- **研究问题：** 视觉特征如何接到语言模型；哪些路线改变主序列长度、哪些保留外部视觉 memory；固定 query 数能否推出计算或质量收益；连接器组合与训练策略的论文证据可外推到什么范围。

| 候选主题 | 来源证据 | 去向与决定 |
| --- | --- | --- |
| Prefix projector 与两阶段对齐 | LLaVA 原论文 §3–4；[多模态系统研究记录](../../research/multimodal-systems-and-evaluation-2026.md) 已记录可训练投影矩阵及论文两阶段训练。 | 保留为 prefix 和训练路线的具体实例；明确是 LLaVA 配置，不写成所有 VLM 的通用 recipe。 |
| Cross-attention 与固定长度 resampler | Flamingo 原论文 §§2–3；研究记录保存 Perceiver Resampler 和 gated cross-attention 的证据边界。 | 讲清视觉 memory 与主序列的区别；论文采用的 64 个 latent 是 Flamingo 特定值，不推广至其他实现。 |
| Query bridge 与长度压缩 | BLIP-2 v3 §§3.1–3.3；研究记录含 32 个 query、`257×1024` 到 `32×768` 的论文示例。 | 加入可复算的位置数比例，并限定为序列长度变化，不等于 FLOPs、延迟或质量比例。 |
| 连接器与多层注入的组合 | Qwen3-VL Technical Report v2 §§2.1–2.2；来源边界见研究记录。 | 用两层 merger 合并相邻 `2×2` 视觉特征并结合前三层 DeepStack 作为单一实例；DeepStack 的主要解释留在第三章，通过本地链接复用。 |
| LoRA 作为语言侧适配选项 | Hu et al., [LoRA 原论文 v2](https://arxiv.org/abs/2106.09685v2)；仓库[研究方法证据记录](../../research/research-methods-evidence-and-reproducibility-2026-11-03.md)已记载其冻结底座和低秩可训练更新。 | 在训练策略表中补齐这一行；论文效果绑定其模型与任务，不外推到 VLM 多模态质量。 |
| 多图预算、动态长度和评估 | 论文架构描述与本章明确标注的教学输入。 | 用假设 token 预算说明算式；产品上下文规格及跨媒体统一预算留给第十四章，反事实测试写成评估方法而非能力结论。 |

**扫描结论：** 代表路线足以覆盖本章问题；新增示例用于区分机制和证据边界，无须增章、重排或建立新研究笔记。候选特性均按论文实例表达，没有外推成行业默认架构。

## 修改与教学链

- 补全开篇的视觉输入—编码—连接器—多模态上下文—生成数据流，区分 CLIP 的相似度输出与 VLM 的条件生成。
- 统一视觉张量、projector 参数和位置预算的形状契约；区分主序列、视觉 memory、image placeholder 和图片对象映射。
- 重写 cross-attention 的单头/多头张量说明、投影维度、softmax 归一化轴和 image-level mask；加入逻辑 score-cell 预算及融合注意力不物化完整矩阵的边界。
- 补充 Q-Former、Perceiver Resampler 和 Qwen3-VL merger/DeepStack 的对照，突出压缩长度不等于计算量或保真度。
- 校准 SFT 的 assistant-only 监督、有效标签分母、零有效标签边界与 attention mask 的不同职责；LLaVA 两阶段配方明确限定为该论文选择。
- 动态长度一节将 padding、packed sequence 和 ragged 首次出现处释义；训练策略表修复 LoRA 行并标明资源等级和语言论文实验不能外推至 VLM。延迟分解先定义各阶段时间及单位，限定为不含排队/网络/工具往返的串行教学账本。
- 将练习和回顾中的 `<image>` 标成行内代码，避免 Markdown 把占位符解释为 HTML 标签。

## length_status

- 全篇一个核心讲解单元；解释性中文字符实计 **8,250**，高于 1,800 下限。统计排除标题、表格、代码/文本示意与数学围栏、行内代码/公式、链接目标、状态行及编号参考文献；保留解释正文、练习和回顾。长度达标不替代机制与证据的人工审阅。

## example_status

- 通过。正文至少包含以下可复算输入和结果：
  - projector：`1024×4096 + 4096 = 4,198,400` 个参数；`[2,576,1024]` 投影后为 `[2,576,4096]`。
  - cross-attention 教学输入 `B=2, H_a=32, T=128, N_m=64`，逻辑 score cell 为 `524,288`；按 FP32 完整物化为 `2,097,152` 字节，即 2 MiB。该值不代表实际峰值显存。
  - BLIP-2 论文特定示例 `257×1024 → 32×768`：少 225 个位置，保留约 12.45%，长度因子约 8.03；明确不是 FLOPs 或延迟收益。
  - 教学 resampler 示例 `576 → 64`，保留位置比例约 0.111，位置数减少约 88.9%；视觉编码和读取仍然执行。
  - 三图预算 `3×576+512+8=2,248`；每图压到 64 后为 `3×64+512+8=712`；另有单图对照 `576+128+4=708`。这些是显式假设，不是产品规格。
  - 监督示例逐项构造 labels/mask 并核算有效 assistant label；章节代码输出 32 个有效标签。代码不能替代正文手算。

## code_status

- §12 是唯一独立运行的 Python 标准库教学代码块，旁边标注“已验证（2026-11-12；Python 3.12.9；仅 Python 标准库）”。整册复读发现预算和 shape 函数会接受浮点/布尔计数，现加入正整数校验、布尔 `bias` 校验及 `VLMConfig` 字段验证；正常输入输出不变。
- 验证方式：Python 3.12.9 从 Markdown 提取唯一 `~~~python` 块并执行，再对各 helper 执行额外输入边界检查；原有 shell 一次性提取命令仍可用于独立重跑标准输出。
- 实际输出含 projector 参数 `4198400`、shape `(2, 576, 4096)`、预算 `708/2248/712`、长度比例 `0.111`、cross-attention cell `524288`、有效标签 `32`；10 个块内 `boundary_checks` 与全部 `checks` 为 `True`，另 4 个浮点输入及一个已构造后被改坏的配置均按约定拒绝，`signals` 与 `actions` 为空，决策为 `continue_to_grounded_vlm_evaluation`。
- 块内边界检查仍拒绝 placeholder/image 数不一致、零维 projector、零 cross-attention 维度及无 assistant 监督；新增拒绝 bool image/assistant count、浮点 token/attention 计数、非布尔 bias 和 bool 配置字段。演示不读取图片、不调用 VLM、不测 GPU、真实 OCR/grounding 质量或延迟/显存。

## formula_status、link_status 与 visual_status

- **formula_status：** 30 个 display-math 块，121 对行内 `$` 标记；花括号平衡。未发现 `\\operatorname`、`\\text{}` 中带下划线变量或 `x_{<t}` / `y_{<t}`。数学源码结构已检查，未做 GitHub/浏览器 MathJax 渲染验证。
- **link_status：** 唯一本地章节链接指向第三章且目标存在；外链目标未做 HTTP 全量检查。LoRA 原论文 v2 作为策略表近邻来源链接。
- **visual_status：** 保留数据流文本图、张量/成本公式、路线对照表、budget 逐步账本和监督标签示例，足以表达本章形状、数据流与方案差异；本章不需要再加装饰图或 Mermaid。无 Mermaid 渲染门禁。
- **源码结构检查：** 52 个 fenced blocks 的开闭配对正确；1 个 Python 块，30 个数学块；Markdown 表格数据行列数一致；无断裂本地链接。

## sync_status 与 open_risks

- **sync_status：** 简介和目录的第 4 章摘要已覆盖连接器路线，顺序与职责未变，无需改写。`PAPERS.md` 已包含 LLaVA、Flamingo、BLIP-2、Perceiver IO、Qwen3-VL 和 LoRA；术语表已有 Q-Former、Perceiver Resampler、DeepStack 等条目；`INTERVIEW_BANK.md` 与 `EXERCISES.md` 已覆盖 VLM 架构、预算、mask 与评估练习，本轮无须重复同步。新事实和版本仍由既有研究记录承载。
- **open_risks：** 尚未对真实图像、真实模型或扫描文档做 OCR/grounding/反事实质量评估；未测 GPU 峰值显存、吞吐、生产端到端延迟；论文配置不代表当前产品 API；未做外链 HTTP 全量巡检或目标平台公式渲染验证。
- **handoff：** 第十五册前四章的章级门禁均已通过，且已按主路径完成第一至第四章整册复读。下一步继续[第五章《多模态 Instruction Tuning》](../../book-15-multimodal-generative-models/chapters/05-多模态instruction-tuning.md)整册复读，沿用本章的图片对象映射、视觉条件、有效标签和连接器边界，核查样本 schema、chat template、多轮监督、拒答及安全训练，并继续区分第五册训练方法和第九册数据治理职责。全套书修书 goal 继续 active；不更新 PDF 或 `plan.md`。
