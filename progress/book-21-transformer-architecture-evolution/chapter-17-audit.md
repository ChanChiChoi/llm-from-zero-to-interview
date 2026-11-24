# 第十七章《LLaMA、GPT、Mistral 风格架构对比》精修审计

审阅日期：2026-11-20。状态：章节职责复核、结构设计、版本化来源核验、正文修订、配套同步和静态验收完成。全套书精修 goal 继续 active；本章不更新 PDF 或 `plan.md`。

## 读者契约与章节边界

- **前置：** 已理解 decoder-only Transformer、自回归 attention、位置表示、LayerNorm/RMSNorm、FFN、MHA/GQA 和 KV cache 的基本概念；本章在比较时唤回必要关系，机制推导回链第 4、6、7、11、18、21、22 章。
- **离章结果：** 能为具体 checkpoint 或公开报告确定版本与来源，再比较位置表示、归一化、Q/KV heads、注意力可见范围、FFN/MoE 和上下文字段；能区分披露事实、实现解释、来源推断与未知项。
- **本章职责：** 接续第十六章 decoder-only 通用生成接口，讨论 GPT、LLaMA 与 Mistral 的指定公开版本采用什么 block 设计。第十八章主讲 RoPE/长上下文扩展，第五十一章主讲模型谱系、scaling 与 ICL；注意力、归一化、KV cache 和 MoE 的完整机制分别由第 4、6、11、21、22 章承担。
- **范围外：** 不猜测未披露 GPT 服务模型的内部结构；不从异模型 benchmark 推断某个 block 设计优越；不扩写 serving、runtime、kernel 或硬件性能教程。

## 旧稿问题与结构决定

旧稿有 24 个 H2、约 822 词，重复介绍架构特点，并混有 Mistral 滑窗跨度与上下文配置字段的混淆、Python KV-cache demo、自拟 Pareto 公式以及 serving/runtime 内容。读者难以沿一个稳定维度比较具体模型，也容易把家族名误当成固定配置。

结合第十六章交接和第十七章完整大纲，将正文重组为六节：先固定模型版本与证据来源，再建立 decoder block 比较坐标，依次比较位置/norm/FFN、Q/KV heads、局部窗口与上下文字段，最后以版本化表格收束有限结论。章名、文件路径和目录入口保持不变。有效结构见[第十七章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/17-LLaMA-GPT-Mistral架构对比.md)。

## 研究核验与正文决定

版本、来源阅读位置和证据范围记于[GPT、LLaMA 与 Mistral 架构比较研究记录](../../research/gpt-llama-mistral-architecture-comparison-2026-11-20.md)。正文主要决定如下：

- GPT-3 175B 作为论文披露的历史基线；GPT-4 技术报告没有披露 block 规格，相关字段保留为未知；gpt-oss-20b 的 MoE/GQA/局部与全局 attention 配置只用于该开放模型版本。
- GPT-2 参考实现用于补充 GPT-3 架构基线中的 LayerNorm、可学习位置表、GELU 与 MHA，并明确这是对指定公开材料的来源推断，不用于外推闭源 GPT。
- 将 LLaMA 原始论文、Llama 3 技术报告与 Llama 3.1 模型卡的事实按各自版本拆开引用，不拼成一个无版本的“LLaMA 配方”。
- Mistral 7B v0.1 区分论文 `context_len=8,192`、固定 revision 的 `max_position_embeddings=32,768`、`sliding_window=4,096` 与论文计算的约 131K 理论 attention span。新增配置及 Transformers v4.34.0 参考实现交叉核验，说明该固定版本的 RMSNorm、RoPE 和 SiLU 门控 MLP；实现证据只解释该配置，不推广到所有 Mistral 版本。
- GQA 的速度/质量表述限定在原论文指定 T5 uptraining 实验；滑窗性能只按 Mistral 论文报告的配置说明。未据此给模型排名。

## 配套同步

- `EXERCISES.md` 新增 31V：按固定模型版本填写架构字段、证据来源和未知项，并解释 Mistral 的窗口/上下文字段差异。
- `INTERVIEW_BANK.md` 新增 31S：回答为何不能只凭 GPT、LLaMA、Mistral 家族名推断 block，并说明未披露项与实现证据边界。
- `PAPERS.md` 增加 GPT-3/GPT-4、GPT-2 源码、LLaMA/Llama 3、Mistral 7B、gpt-oss 与固定 Mistral 实现入口。
- 术语表已有 GQA、RMSNorm、RoPE、SwiGLU、MoE 等条目，本轮不重复添加；目录与简介中的章名和导航无需变化。

## 静态验收与遗留项

- H2 `17.1—17.6` 连续；比较坐标表和版本对照表均为三列；本章没有公式、代码围栏或 Mermaid 图。
- 本章与大纲中的本地 Markdown 链接目标存在；相关修改文件 `git diff --check` 通过。PAPERS.md 另有本章以前的失效本地链接，本轮未改动；不将其计作新增缺陷。
- 本轮未运行代码、测试、模型、API、benchmark 或 PDF 构建。已核对关键一手资料和固定实现来源，但没有对本章全部外链做逐条可达性终审。
- **后续门禁：** 目标读者试读、第二十一册连续复读、全章外链复核；第十二章两处及第十四章一处既有块级公式的 GitHub MathJax 渲染仍待验收。本章无公式，不触发公式渲染门禁。
- 下一入口为[第十八章《RoPE 与长上下文位置外推》](../../book-21-transformer-architecture-evolution/chapters/18-RoPE与长上下文位置外推.md)。第二十一册已完成第 1—17 章，尚有第 18—95 章 78 章待逐章审阅。

## 跨章导航校正

2026-11-20 将 GPT-OSS 的 MoE 路由与 Dense/MoE 成本回链由第二十三章改为第二十一、二十二章，并把第二十三章指向小模型与端侧效率；同步修正本章大纲和正文导航。此处仅校正章节职责链接，不改变本章技术内容。
