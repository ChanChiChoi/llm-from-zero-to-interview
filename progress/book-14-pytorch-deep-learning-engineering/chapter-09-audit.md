# 第九章精修审计：实现 Transformer 组件

审计日期：2026-11-11

## 范围与章级扫描（scope / chapter_scan）

- 完成《实现 Transformer 组件》§§9.0–9.18 全文审阅与精修；章节仍回答“如何把张量、模块、训练目标与生成状态组成一个可检查的小型 decoder-only Transformer”，没有改变第十四册十章骨架。
- 2026-11-11 定向复核 PyTorch 2.15 官方 SDPA、`causal_lower_right`、RMSNorm、LayerNorm、Embedding 和 MultiheadAttention 文档，并核读 Transformer 与 RoPE 原始论文的固定 arXiv 版本。PyTorch API 事实、版本和证据边界记入[Transformer 组件 API 研究记录](../../research/transformer-component-apis-2026.md)；研究记录只保存 API 新知识，不记本章审阅经过。
- 核验问题：训练方阵与 cache 非方阵的 causal 对齐如何区别；SDPA 和 MultiheadAttention 的 bool mask 语义是否相同；SDPA 的 dropout 是否跟随 `.eval()`；PyTorch RMSNorm 默认 epsilon 与教学实现有何差别；Embedding 的 `padding_idx` 是否同时处理 attention 和 loss；RoPE/cache 的绝对位置、loss shift 和有效目标分母能否端到端核对。

| 候选主题 | 去向与理由 |
| --- | --- |
| PyTorch 2.15 SDPA 非方阵 causal bias、lower-right prototype、bool mask 与 dropout | 纳入 §9.5.5。它们直接影响 cache decode 的可见位置和 eval 正确性，不能由“使用 causal attention”一句话带过。`causal_lower_right` 的 prototype 状态和设备/版本边界也一并保留。 |
| PyTorch RMSNorm epsilon、LayerNorm 方差语义与 Embedding padding 行 | 纳入 §§9.3–9.4，并区分官方 API 行为和本章显式 `eps=1e-6` 的手写类；明确 `padding_idx` 不替代 attention mask 或 loss ignore index。 |
| GQA/MQA、MLA、FlashAttention、PagedAttention、分页分配和连续 batching | 不扩写为本章实现；§9.0、§9.11、§9.15保留机制边界与延伸方向。它们改变 KV 头数、kernel 或生产状态管理，超出本章同长 batch 的教学模型范围，并分别承接第三册项目和第五册规模化训练/推理实践。 |
| RoPE scaling 与各模型的变体配置 | 保留基础二维旋转、相对相位直觉、实现布局和 cache 位置偏移；不从范数守恒或一个 base 值推断长上下文效果。具体模型仍须读其配置和实现。 |

## 篇幅与解释链（length_status）

计数方式：统计连续解释文字中的汉字；剔除标题、代码、数学围栏、表格、引用链接、行内代码和相邻状态标记。按完整教学链而不是每个连续小标题拆分；四条均为核心单元，门槛为 1,800 汉字。

| 教学链 | 范围 | 实计汉字 | 类型 / 门槛 | 状态 |
| --- | --- | ---: | --- | --- |
| 从 token 表示到归一化与模型参数 | §§9.1–9.4 | 2,417 | 核心 / 1,800 | 达标；从 `[B,T]` 输入和 shape contract 推到 embedding/LM head、权重绑定与 Norm 差异。 |
| 注意力、head 与位置关系 | §§9.5–9.7 | 2,744 | 核心 / 1,800 | 达标；从缩放点积和 causal mask 推到融合 API、多头 shape、RoPE 与 cache 位置 offset。 |
| 逐位置变换、残差与训练目标 | §§9.8–9.10 | 2,553 | 核心 / 1,800 | 达标；串联 SwiGLU、Pre-Norm residual、logits/labels shift、有效 token loss 与 perplexity。 |
| 生成状态、实现、诊断与边界 | §§9.11–9.15 | 3,651 | 核心 / 1,800 | 达标；从 KV 预算和请求状态推到连续/分块 mask、完整实现、回归路径及生产边界。 |

## 数值例子（example_status）

- §9.2：`V=32,d=16,m=32,L=2`、无 bias、每层两个 RMSNorm 时，绑定 embedding/LM head 总参数为 `5,712`；独立 LM head 再加 `Vd=512`，合计 `6,224`。
- §9.3：`E=[[1,0],[0,1],[1,1]]`；输入 id 2 得 `[1,1]`，若绑定输出权重且 hidden 为 `[1,0]`，logits 是 `[1,0,1]`。同时区分 lookup padding 梯度、attention mask 和 loss ignore。
- §9.4：对 `[3,4]`，LayerNorm 有偏方差 `0.25`、输出约 `[-1,1]`；RMSNorm 均方值 `12.5`、RMS `3.5355`、输出约 `[0.8485,1.1314]`。示例值按忽略微小 epsilon 的精度舍入。
- §9.5：`q=[0,1]`、`k_0=[1,0]`、`k_1=[0,1]`、`v_0=[10,0]`、`v_1=[0,20]`；位置 1 的缩放分数约 `[0,0.7071]`、权重 `[0.3302,0.6698]`、输出 `[3.302,13.396]`；位置 0 屏蔽第二个 key 后输出 `[10,0]`。
- §9.6：`d=4,H=2,D_h=2,T=3` 时，多头 Q/K/V 为 `[1,2,3,2]`，score 为 `[1,2,3,3]`，拼接后回到 `[1,3,4]`。
- §9.7：二维 RoPE 中 `[1,0]` 位于位置 1 和 3，旋转后点积为 `cos(2)≈-0.4161`；范数保持只能检查旋转正交性，还需位置/相对相位用例。
- §9.8–9.9：SwiGLU 例子 `x=1,W_g=1,W_u=2,W_d=1` 得 `SiLU(1)≈0.7311`、输出 `1.4622`；残差例子 `[1,2]+[0.5,-1]=[1.5,1]`，再加 `[-0.25,0.5]` 得 `[1.25,1.5]`。
- §9.10：输入 `[3,1,2,0]`、labels `[3,1,2,-100]`，shift 后目标 `[1,2,-100]`；前两目标概率 `0.8`、`0.5` 时平均 CE 约 `0.4581`，PPL 约 `1.581`，忽略项不进分母。
- §9.11：`L=32,B=2,H_kv=4,T=512,D_h=64,b=2` 的 KV payload 为 `32 MiB`；KV heads 减半为 `16 MiB`。`P=3` 并生成两个 token 时，无 cache 因果 pair 数 `6+10=16`，带 cache 为 `6+4=10`；但 decode query 仍需读取全部历史 key。`p=3,q=2` 的 mask 两行为 `[1,1,1,1,0]` 和 `[1,1,1,1,1]`。
- §9.15：序列长度从 `512` 增至 `1024`，dense score grid 变为 4 倍而 KV 长度和 payload 约变为 2 倍；此数量关系不代表具体 kernel 的耗时曲线。

## 证据、公式、图表和链接（evidence / formula_status / link_status）

- 原始定义来源固定为 Vaswani et al. *Attention Is All You Need* arXiv:1706.03762v7、Su et al. *RoFormer* arXiv:2104.09864v5；接口结论对应 PyTorch 2.15 官方文档。官方文档说明接口语义，不支持对其他版本、后端选择、生产吞吐或特定厂商模型架构作外推。
- 20 个 display-math 围栏均闭合且花括号平衡；逐项复查变量和假设在正文邻近说明。attention 方差推导标为初始化近似；参数量、cache bytes、操作 pair 数及 PPL 均注明计算口径。
- 三反引号围栏共 16 对（14 个 Python 块和 2 个 text 示意块），所有 Python 块均通过 `ast.parse` 并有紧邻状态；另有 20 个 math 块和 1 个 text 的波浪围栏示意，均成对。未宣称 MathJax 页面渲染或完整外链 HTTP crawl。
- 本地 Markdown 链接 14 个，所有相对路径解析到现存文件；无 `docs/stable`、`\operatorname` 或维护目录路径。公式/代码和链接为源码级检查。
- 视觉机会已审视：§9.1 的数据流文本图、shape contract 表，§9.5 的逐步注意力算例、§9.11 的 cache mask 逐行案例和 §9.15 的复杂度对比已经分别呈现模块顺序、维度关系、可见状态和数量级。本轮未新增 Mermaid；现有线性数据流图与表格足以支撑本章的主要关系，没有声称目标平台渲染验收。

## 代码状态（code_status）

共 14 个 Python 代码块。§§9.2–9.10 的代码块 01–12 均标为静态审阅，未运行；其中包含依赖前文变量/模块的局部片段，不把 AST 解析写作运行验证。块 06、09 的相同 RMSNorm/RoPE 机制由完整实现覆盖，并在后续集成 demo 中检查，但它们仍按单独代码块状态标为未独立运行。

| 稳定位置 / 代码块 | 状态与验证记录 |
| --- | --- |
| §9.2–§9.10，块 01–12 | 静态审阅，未运行；14 个 Python 块全部 AST parse。若读者独立执行局部片段，需按状态标记补齐前文依赖。 |
| §9.12 块 13 + §9.13 块 14 | **已验证**：2026-11-11，Python 3.12.9 / PyTorch 2.12.0+cpu，CPU。原样从章节提取两块合并至临时脚本，以 `.tmp/llm-book-cpu-check/bin/python /tmp/llm_book_ch9_demo.py` 执行。14 项检查全部通过：RMSNorm fp16/bf16/fp32/fp64 输出有限且 dtype 保留、方阵/分块 mask 行正确、attention 行和为 1 且未来权重为 0、RoPE 范数保持、logits `[2,5,32]`、loss `13.7062` 有限、权重绑定、prefill/decode logits 一致、decode attention 可见 5 个 cache key。 |

验证边界：本地环境是 PyTorch 2.12 CPU，不是正文所引用的 PyTorch 2.15 runtime；没有 CUDA、融合 kernel、长上下文、训练 step、GQA/MLA、连续 batching 或真实 serving 验证。CPU 输出不支撑硬件性能结论。源码中 `.item()` 防御性值域检查会在 CUDA 引入同步，正文已标明这是教学实现取舍。

## 配套同步与遗留项（sync_status / open_risks）

- 已同步[第十四册简介](../../book-14-pytorch-deep-learning-engineering/简介.md)和[目录](../../book-14-pytorch-deep-learning-engineering/目录.md)的第九章摘要，明确非方阵 causal mask、RoPE offset 和 prefill/decode 一致性。
- 已更新 `INTERVIEW_BANK.md` Q68 的 PyTorch 2.15 非方阵 mask 边界，并新增 Q72A 说明 `.eval()` 不会自动关闭 SDPA dropout；新增[第十四册第九章练习](../../EXERCISES.md)，覆盖参数量、Norm、attention、RoPE、shift loss、cache 预算和回归 demo；在术语表增加 Causal Lower-Right Bias。
- API 新事实及证据边界已写入既有研究文件；本章审计状态、候选去向和修改结果只在此 `progress/` 维护记录。
- 遗留范围：尚无 PyTorch 2.15 runtime、CUDA/GPU、目标融合 backend 或生产 serving 复核；`causal_lower_right` 仍属 prototype，因此正文保留显式绝对位置 mask。未更新 PDF 和 `plan.md`。

## 工作交接（work_handoff）

第十四册第一至第九章均已完成章级扫描、正文精修和逐章审计。下一步进入[第十章《工程问题与判断》](../../book-14-pytorch-deep-learning-engineering/chapters/10-工程面试题.md)，检查综合工程案例是否回用前九章真实接口、能否沿成功条件—假设—最小实验—证据链组织，而不是变成脱离约束的术语问答。
