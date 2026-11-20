# 第 10 章《O(n²) Attention 的计算与显存瓶颈》精修审计

审阅日期：2026-11-20。状态：章级核验、正文重写、配套同步和当前静态验收完成。未运行代码、测试、kernel、模型、API、benchmark 或 PDF。目标平台 MathJax 渲染、外链全量复核、目标读者试读和第二十一册连续复读属于后续门禁。

## 读者契约与章节职责

- **前置：** 理解 scaled dot-product attention、Q/K/V shape、causal mask 和常见矩阵乘记号；第 1—9 章已建立 Transformer 结构、位置、mask 与并行路径入口。
- **离章结果：** 能从 $[B,H_q,T_q,T_k]$ 推出 attention pair 数与主要矩阵乘 FLOPs；把平方计算复杂度与特定实现是否保存 $T\times T$ 矩阵分开；解释 online softmax 的跨块重标定；区分训练、prefill、decode 与稀疏/近似路径改变的对象。
- **章节边界：** 本章说明成本账本与原理。KV cache 容量、分页和 decode 内存由第 11 章主讲；FlashAttention 实现和 GPU kernel 由第 20 章主讲；滑窗/稀疏注意力由第 19 章主讲；线性/核近似与替代架构由第 24—28 章主讲。

## 研究证据与正文处理

联网发现的技术知识、来源位置与限制见[Attention 复杂度、工作内存与在线 softmax 研究记录](../../research/attention-complexity-memory-online-softmax-2026-11-20.md)；FlashAttention 版本演进另见[既有 IO 与硬件研究记录](../../research/flashattention-io-softmax-and-hardware-evolution-2026.md)。

| 主题 | 证据支持及边界 | 正文处理 |
| --- | --- | --- |
| Dense attention 计算量 | $S/P$ 的 shape 给出 $BH_qT_qT_k$ 个位置对，$QK^\top$ 与 $PV$ 合计约 $4BH_qT_qT_kd_h$ FLOPs；不包含投影、softmax 和其他层运算 | 10.1—10.2 明确 shape、计数口径及 self/cross-attention 差异 |
| Attention 额外内存 | 显式 materialize 的 $S/P$ 可占 $O(T_qT_k)$ 存储；Rabe/Staats 与 FlashAttention 说明可用流式/重计算避免保存完整矩阵 | 把“attention memory 必然二次”改为实现条件；用 32K、32 query heads、BF16 做 payload 手算 |
| FlashAttention | Dense 算法以 tiling、在线 softmax 和 backward 重算处理同一 full-softmax 数学算子；主 pair 算术仍是二次，浮点舍入次序可能不同 | 给出 $(m,\ell,u)$ 全局更新规则及两块加权教学追踪；分开 dense exact 与 block-sparse 路径 |
| Causal mask | 有效位置对数是 $T(T+1)/2$；密集张量的分配 shape 与具体 kernel 是否跳过无效 tile 是不同事实 | 不把“约一半可见 pair”直接写成“约一半时间/显存” |
| 训练、prefill、decode | 前两者一般使用整段 query；单步 decode 用 $T_q=1$ 且历史 key 长度递增，KV 状态和带宽另有成本 | 用阶段表和总 decode pair 公式解释，并交接第 11 章 |
| Sparse / low-rank / kernel feature | Longformer、Linformer、Performer 分别支持指定稀疏图、低秩近似、随机特征近似及各自论文结论 | 只比较它们改变的对象和核对边界，不泛化任务质量或硬件速度 |
| 教学组织 | UW CSE 599M 的 online-softmax 推导与 UC Berkeley Fall 2026 Attention and MoE Kernels 讲义提供状态合并和 tile 讲解样例 | 先给 pair/存储账本，再解释 online-softmax 状态与合并；课程顺序作为讲解参考，事实仍回核原论文 |

## 正文与配套修改

- 删除旧的日期化资料段、长参考清单、重复的 IO-aware/FlashAttention 解释、作者工作流、简化损失与跨模型版本盘点。
- 修正“标准 attention 必须有 $O(T^2)$ 额外显存”的混淆；用实现是否 materialize score/probability、attention workspace 和全模型训练总显存三个层次说明内存。
- 用在线 softmax 的最大值、归一化质量和加权和解释分块过程；新增两块合并教学值，避免把局部 softmax 平均误当全局 attention。
- 删除任意资源阈值、统一 attention cost gate 与 0 依赖 Python demo；保留可复算的 pair、FLOPs、显存公式和阶段账本。
- 将 PagedAttention/GQA 详细职责转交第 20、11、4 章；滑窗与替代路线转交第 19、24—28 章。
- 同步[练习 31N—31O](../../EXERCISES.md)、[面试题 31M](../../INTERVIEW_BANK.md)、Attention Cost Evidence Brief 项目、术语表、知识图谱和[论文路线](../../PAPERS.md)。章节编号未变，目录无需修改。

## 静态验收与遗留项

- H2 10.1—10.7 连续；数学围栏标记 24 个、成对形成 12 个 math 块；行内 $ 定界符计数为 164（偶数）；6 个本地 Markdown 链接目标存在；未发现控制字符或正文中的工作流/demo 残留。
- 核对主要公式、变量定义、causal pair 与 materialized memory 的区分、online-softmax 状态更新和跨章链接；修改文件的定向 git diff --check 通过。
- 当前静态检查不能替代 GitHub MathJax 渲染、外链全量可达性、读者复述测试和本册连续复读。未运行代码、测试、kernel、模型、API、benchmark 或 PDF。
- 下一入口为第十一章《KV Cache、长上下文推理和显存增长问题》；第二十一册第 1—10 章完成，85 章待逐章审阅。
