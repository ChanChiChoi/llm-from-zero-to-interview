# 第三册第十三、十四章结构收束与章级扫描（2026-10-28）

## 审视范围

按 `WRITING_SPEC.md` 第 2.1 节的顺序，先检查新版简介、目录和册级章节地图，再回到第十三、十四章全文。两章的角色需要在书内分开：

- 第十三章回答“已完成的实践如何变成可信、可复核的项目交付”，不能重新讲一遍前三至八章机制。
- 第十四章回答“怎样从机制、约束和证据组织技术解释与系统设计答辩”，可以用小型推演示范表达，但底层实现回链到主讲章节。

## 旧结构问题与内容去向

旧第十三章虽然在开篇宣称要讲证据，却有六个大段重新教学 miniGPT、SFT/LoRA、DPO、Serving、RAG 和数学推理，正文职责与章节定位冲突，且重复第三至八章。新版把六类实践保留为索引表，详细机制回到第三至八章；第十三章改为一条交付链：

1. 写出可证伪主张和发布护栏；
2. 固定数据、模型、代码、环境、指标和分母；
3. 让基线与一次主要改动一一对应；
4. 检查整体分数、关键切片、成本和失败样例；
5. 形成复现说明、README 和诚实的简历表述。

原章七个 Python 教程/审计块随重复教程一起移除，新章没有可执行 Python。没有必要为了保留旧代码而继续让作品集章承担教程职责；同类模型实现仍在对应主讲章可读可运行。

旧第十四章前两节是 Attention 和 GPT 的完整技术教程，复述第二章及第三章。新第十四章只保留两个用来演示回答结构的数值例：二位置 Attention 手算，以及 KV Cache 容量估算。Attention 的完整实现回到第二章，Serving 的测量和实现回到第六章。其后的项目证据链、开放问题比较、系统设计、论文评议与统一表达纪律保留，并衔接成“解释机制 → 举证 → 方案决策 → 系统设计 → 论文评议”。

第十四章前两节原有两个 Python 实现块因重复教学而删除；余下五个标准库记录/约束审计例未改动，行内既有验证状态仍对应原代码，无需重跑。两章没有新增 Python 代码。公式、表格和 Mermaid 图直接解释章节主线，不以代码或目录代替论证。

## 新版数值例

第十三章使用同一份 100 题 RAG 评估演示整体收益与切片退化：有效引用成功数从 72/100 到 80/100，但旧政策切片从 16/20 到 12/20；在 75% 的旧政策最低门槛下，候选方案不能发布。再以每请求成本 0.018 美元和 0.024 美元计算每个成功回答成本 0.025 与 0.030 美元，说明总成功数增加不等于单位成功成本下降。数字仅为教学样例，不宣称统计显著或线上收益。

第十四章的 Attention 例把两个点积、缩放、softmax 和 value 混合逐步算出约 (1.34, 1.32)，再屏蔽第二个 key 得到 (2, 0)。KV Cache 例使用 32 层、8 个 KV head、head dim 128、FP16、4,096 token 和 24 路并发，逐项算出每条序列 512 MiB、合计 12 GiB。该账本不包含权重、临时激活、碎片和运行时预留，明确只是容量估算。

## 章级外部扫描与取舍

检索日期：2026-10-28。针对第十三章检索模型卡、运行/工件追踪和 Agent trace 评测的官方文档；针对第十四章检索 Serving 指标和新近 KV cache 压缩方向。

| 发现 | 证据 | 决定 |
| --- | --- | --- |
| 可复核模型项目需要同时描述用途、限制、训练参数、数据和评估，并保存运行条件、指标与输出工件 | Hugging Face Model Card 文档列出模型用途/限制、训练信息、数据集和评估；MLflow Tracking 文档说明 run 可记录 parameters、code version、metrics 和 output artifacts | 纳入第十三章 manifest 与模型说明，但不把特定平台当成唯一实现；工具不会自动证明评分规则或因果结论正确 |
| Agent 调试与回归应区分单条轨迹定位和固定数据上的重复测量 | OpenAI Agent Evals 官方指南说明先检查 trace 定位 workflow 问题，在标准明确后使用数据集与评估运行；当前 OpenAI Evals 文档同时说明该平台/API 正进入迁移退场计划 | 只采用“轨迹诊断 → 固定样本回归”的通用流程；不在第三册绑定该平台接口，也不新增独立章节 |
| Serving 答辩不能把一个端到端延迟数字当作完整诊断 | vLLM 当前指标文档分别列出 queue、prefill、decode、TTFT、端到端延迟及 token 指标 | 第十四章系统设计段增加实现版本边界，并要求 trace 按关键路径拆解；字段名随 runtime 版本变化 |
| KV cache 降本仍有新型表示研究 | 2026 年 arXiv 预印本提出 Grouped Value Attention 与 Kronecker Product Attention 等表示；其摘要结果绑定论文模型规模、训练预算和指标 | 作为第六册/二十册推理专题的后续候选，不把未经独立部署验证的架构塞进第三册面试表达章，也不据摘要声称工程收益 |

### 来源

- [Hugging Face Model Cards](https://huggingface.co/docs/hub/en/model-cards)：用途和限制、训练信息、数据集与评估结果等模型说明字段。
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)：实验 run、参数、代码版本、指标和 artifacts。
- [OpenAI Agent Evals](https://developers.openai.com/api/docs/guides/agent-evals)：trace 诊断与固定数据评估的工作流。
- [OpenAI Evals deprecations](https://developers.openai.com/api/docs/deprecations)：当前产品迁移状态；因此书中不绑定平台 API。
- [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)：实现版本中的排队、prefill、decode、TTFT 与端到端观测。
- [Grouped Value Attention, arXiv:2609.13285](https://arxiv.org/abs/2609.13285) 与 [Kronecker Product Attention, arXiv:2610.12657](https://arxiv.org/abs/2610.12657)：近期预印本，主张和测量条件仍由论文设置约束。

## 验收与剩余边界

更新后第十三章与全册项目主线的关系清晰：前文负责完成项目，第十三章只负责把项目变成可复核证据，第十四章使用证据完成表达和设计答辩。按当前规范口径静态计数，第十三章有 3,344 个解释性汉字，第十四章有 8,264 个；两篇均高于 1,800 字下限。第十三章每个示例继续使用第三至八章作为机制主讲入口，避免重复教学。

第三册全量本地静态检查共覆盖 16 个 Markdown 文件、972 个围栏、217 个 Python 块和 9 个 Mermaid 图；相对文件链接目标存在、标题无重复、围栏闭合、Python AST 可解析、每个 Python 围栏邻近均有状态行。`git diff --check` 通过。检查不等于目标平台渲染；GitHub 上 Mermaid rich display 与 PDF 构建仍待推送及 pandoc/XeLaTeX 环境支持后验收。
