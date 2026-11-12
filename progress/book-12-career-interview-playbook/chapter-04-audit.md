# 第十二册第四章审计：Coding Interview

审阅日期：2026-11-05  
状态：本轮章级扫描、正文精修和静态门禁完成；第十二册连续复读已通过第一至第四章。所有章节代码均未运行，代码执行验证门禁仍未完成。PDF 不更新。

## Scope 与章节扫描

- 全篇审阅《Coding Interview：把模型语义写成可靠程序》，覆盖 4.1—4.21：面试契约、Python 数据处理、张量 shape、因果语言模型损失、mask/attention/MHA/RoPE、采样与生成、序列分数与 DPO、归一化与 MLP、debug、数值稳定、内存复杂度、训练评估、测试、练习和准备计划。
- 章级扫描聚焦：章节是否把现场拆题、实现、复杂度、错误定位、测试和沟通连成可练习的回答链；重要实现是否有手算输入和中间结果；API 行为是否指明版本；代码是否明确完整度、依赖及运行状态。
- 定向核阅日期为 2026-11-05，问题覆盖 Python 堆排序接口、PyTorch 张量视图/广播/gather/cross entropy/SDPA/inference mode、Transformer/RoPE/RMSNorm/DPO，以及 FlashAttention 与 PagedAttention 的内存机制。使用 Python 3.15 文档、PyTorch 2.15 官方文档及固定 arXiv 版本；具体链接、版本和证据边界列于正文 §4.20。
- 扫描未发现需要新增章节或改变本册边界的主题。现有章节内容均保留；把原本已有的 FlashAttention 与 PagedAttention 相关内容补上机制说明和来源，不扩成独立专题。API 版本差异、缺失的算例和代码依赖通过当前小节补齐。

## 候选与处置

| 候选或问题 | 处置 | 理由 |
| --- | --- | --- |
| Coding 题容易停留在代码长度或题型清单 | 保留程序契约主线，串起输入输出、shape、数值、错误和测试 | 面试表达应解释决策如何保证语义，不靠背 API 或罗列技巧。 |
| Python 字符数、JSONL、堆、生成器缺少手算输入 | 补充小样本统计、Top-K 输出和峰值载荷估算 | 让复杂度和异常处理对应具体数据；教学样本不代表真实训练数据质量。 |
| gather、causal loss、mask、attention、RoPE 等公式/接口没有完整手算 | 增加类别选取、label shift、因果矩阵、attention 权重/输出和旋转结果 | 数值例展示关键中间量，不由代码替代手算。 |
| SDPA mask 语义和非方形 causal 行为过于笼统 | 按 PyTorch 2.15 文档明确 True 的含义、反向 key_padding_mask 语义、mask broadcast、is_causal 冲突及左上角对齐 | 这些是具体版本的接口契约，不外推到全部 PyTorch 版本。 |
| 采样、DPO、归一化、SwiGLU 与训练/评估代码缺少可复算例或依赖说明 | 补充概率、DPO loss、LN/RMSNorm、SwiGLU、梯度累积和 masked mean 示例；标明调用片段依赖 | 读者能核对方法，也能辨认不可独立运行的片段。 |
| 位置/cache、数值掩码、截断和调试内容偏抽象 | 增加绝对位置、全屏蔽行、KV cache、token accuracy、生成状态和 token budget 的具体轨迹 | 把 shape 正确但语义错误的失败模式显式化。 |

## 篇幅与教学例子

- **length_status：** 按连续论证分为五个核心教学单元；计数仅含解释性汉字，不含标题、表格、代码、math/text 围栏、行内代码和代码状态标记。§4.20 资料索引与 §4.21 结语不承担独立完整教学任务，不单列为核心单元。

| 核心单元 | 汉字数 | 下限 | 状态 |
| --- | ---: | ---: | --- |
| §§4.1—4.4：面试契约、张量/数据与语言模型损失 | 4,065 | 1,800 | 通过 |
| §§4.5—4.7：mask、attention、head 与位置/cache | 2,043 | 1,800 | 通过 |
| §§4.8—4.10：采样、生成偏好与模型内部模块 | 1,987 | 1,800 | 通过 |
| §§4.11—4.14：debug、数值、资源和训练状态 | 2,740 | 1,800 | 通过 |
| §§4.15—4.19：评估、复盘、练习和准备计划 | 2,054 | 1,800 | 通过 |

- **example_status：** 已给出可复算示例，包括 gather 选出 −0.6931/−1.2040；JSONL 的 3 行计数与 5 字文本；Top-K 的 `a:3,b:2`；shift 后 LM loss 约 0.1269；T=3 causal mask；attention 权重约 0.6698/0.3302 和输出 `[1.3396,1.3208]`；D=8、H=2 的 shape 轨迹；RoPE `[1,0]→[0.5403,0.8415]`；temperature/top-p 分布；序列 log probability −0.9163 与 DPO loss 约 0.6782；LN/RMSNorm 与 SwiGLU；稳定 softmax 和全屏蔽行；T 翻倍导致 score 元素数变 4 倍；KV cache 约 192/768 MiB；token accuracy 0.5；梯度累计 3；masked mean `[2,2]`；生成/cache/mask 和 token budget 的逐步轨迹。全部是构造输入，不作论文、生产或招聘绩效主张。

## 公式、代码、链接与同步

- **formula_status：** 19 个 math 围栏，花括号逐块配平；未发现 `\operatorname`；没有行内美元公式。2026-11-06 用 MathJax 4.1.3 将本册 244 个 display 与 22 个 inline 数学表达式统一渲染；首轮定位到本章 RMSNorm 公式中缺失的分式分母括号，修正后全部解析通过。GitHub 当前页面仍未能实际核验本地未发布修改。
- **code_status：** 共 30 个 Python 围栏。`analyze_jsonl`、RMSNorm 和 `masked_mean` 于 2026-11-06 在 Python 3.12.9、PyTorch 2.12.0+cpu 环境执行；其余 27 块仍如实标为静态审阅、未运行。依赖前文 helper、通用模型协议或调用方对象的片段均有说明；训练步骤反例标作故意不完整。对采样、生成、RoPE、DPO、损失、调试和 masked mean 等代码补了输入边界、设备/类型检查、依赖导入或数值样例。`analyze_jsonl` 验证三行文档样例（两条有效重复记录加一条非法 JSON）、空文件、非对象 JSON 和无效字段；输出计数、均值/中位数与正文相符。RMSNorm 验证 `[1,3]` 输出约 `[0.4472,1.3416]`，并检查标量、末维不匹配、整数输入和非法 hidden size。`masked_mean` 验证 `[2,2]` 样例、屏蔽位置 NaN、全 False mask、空时间维、shape 与 dtype 边界；结果符合正文约定。GitHub 页面实际呈现留待本地修改可读后复核；未运行模型、benchmark 或 PDF。
- **link_status：** 共 29 个 Markdown 链接、15 个唯一外部目标、无本地链接和无断链。API 入口固定到 Python 3.15 与 PyTorch 2.15；论文固定到 Transformer v7、RoFormer v5、RMSNorm v1、DPO v3、FlashAttention v2、PagedAttention v1。正文链接贴近所支持的接口或方法主张；论文结论不外推到当前生产性能。
- **静态结构门禁：** H2 4.1—4.21 连续，共 21 个；H3 58 个；53 个围栏均配对（19 math、30 python、4 text）；Math 花括号平衡；30 个 Python 状态标记与代码块一一对应。`git diff --check` 通过。
- **sync_status：** 章节名、顺序和入口未变，无需改简介或目录。第 3 章到本章的既有导航保留；正文结尾继续引向第 5 章。连续复读已通过第 1—4 章；第 5—13 章复读、代码运行验证、公式实际页面渲染和本册整册最终门禁仍待处理。

## 遗留风险与交接

- 三个更新过的独立示例已有 Python 3.12.9 / PyTorch 2.12.0+cpu 的本地执行记录；其余 27 个代码块保持静态审阅状态，尤其是依赖调用方接口或外部对象的片段仍需在目标调用环境核验。该本地结果不代表 PyTorch 2.15、GPU 或生产实现。
- 本地 MathJax 4.1.3 对全册公式的渲染已通过；由于当前正文改动尚未发布，GitHub 当前版本页不能代表本工作区，目标平台页面核验保留为后续复核项。外部引用按版本核阅，历史链接巡检已完成。
- **下一步入口：** 第五章《ML 与 LLM 基础面试》，按目录继续连续复读；之后完成第 5—13 章、GitHub Mermaid / MathJax 渲染、历史外链巡检及适用的代码验证门禁。全套书修书 goal 保持 active，不更新 PDF。

## 整册连续复读记录（2026-11-06）

- 按目录读完 §§4.1—4.21，并核对第 3 章项目证据到编码面试程序契约的交接，以及本章通向第 5 章基础面试的收束；章节作为按岗位选读的技术面试单元，与本册主线相符，无需改简介或目录。
- 复核了 token 序列 log probability、DPO 单对损失、prompt mask、LN/RMSNorm、SwiGLU、attention 复杂度、MHA/GQA KV 容量、masked token accuracy、梯度累积和 masked mean 的算例与口径。复读发现 JSONL 统计把可解析对象称为 valid record 且空统计返回 0.0、RMSNorm 未拒绝整数/标量输入、masked mean 用乘零屏蔽会被无效位 NaN 污染；已在正文改为明确的计数/空统计语义、补充输入检查，并以 `where` 清零无效值。三个改动块于 2026-11-06 静态复核，均未运行。
- 未改本章链接、公式、导航或章节顺序；既有代码与未运行状态保持如实记录。GitHub MathJax 页面渲染、历史外链复巡、其余适用代码的执行验证以及第 5—13 章连续复读仍未完成。下一步从第 5 章继续。
