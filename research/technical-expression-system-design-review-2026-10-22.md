# 第三册第十章技术表达与系统设计精修证据

- 日期：2026-10-22；范围：`book-03-practical-handbook/chapters/10-面试实战训练.md`，10.1—10.7 全文。
- 证据类型：paper、official、local-demo；评分权重、合成项目和容量场景为教学设计。

## 来源与边界

正文 13 个唯一外链经维护者指定代理检查，均返回 HTTP 200：Attention、LoRA、DPO、FlashAttention、RAG、Mamba 六篇 arXiv 论文，GPT-2 报告，PyTorch SDPA，Transformers causal LM 与 generation，OpenAI evals，Google SRE SLO，以及 NIST AI RMF。

arXiv Atom API 核对六篇论文题名及版本：Attention v7、LoRA v2、DPO v3、FlashAttention v2、RAG v4、Mamba v2，均与引文对应。另读取 FlashAttention HTML 的 IO 优化相关内容、Mamba HTML 的选择机制与离散化相关内容，以及 Google SRE 的 SLI 测量窗口和 SLO 定义。未逐篇全文或实验复现。GPT-2 PDF、Transformers、evals 和 NIST 本轮只确认入口可达。

PyTorch stable URL 本日返回跳转到 2.14 的 HTML 页面，因此 HTTP 200 不作为该 API 正文已读取的证据；本地行为另外用 PyTorch 2.12.0+cpu 实测。版本不能混为一谈。

## 修订与本地验证

- Attention 说明全 mask 行、padding query、共享 batch toy mask 和 cache 绝对位置；缩放前后熵使用相同 mask。CPU SDPA 与标准库结果一致；额外确认缓存显式 mask 与非方阵 upper-left causal mask 结果不同。
- GPT 说明 shift 需要 T+1 个 token，区分绝对位置加法与 RoPE，明确 KV 复用的模型、历史及位置配置条件。
- 项目证据区分已登记引用与语义支持；候选选择允许无合格方案，限制率和权重定义域，点估计不承担安全非劣效证明。
- Little's Law 使用同一稳定窗口的均值；保留精度到 ceil，说明串行时延预算、关键路径和分位数不可直接相加；ACL 下推后再复核。
- 增加在线 softmax 未归一化分子和分母的递推及可运行分块加权示例；说明 Mamba 的输入相关 B/C/Delta 与连续 A；论文卡片字段完整不等于理解、复现或证据核验，负结果可形成完整记录。

七段标准库 Python 分别编译和独立执行通过。边界覆盖空/不齐输入、NaN/Inf、全 mask、shift 长度、head 维度、零基线、未登记引用、零权重、非法率、无合格方案、peak factor、零请求、ceil 精度、全 mask tile 和空实验字段。负收益但材料齐备的论文记录仍可完整。

可核对输出：attention 最后一行权重 `[0.2483,0.2483,0.5035]`，缩放/未缩放熵 1.037277/0.975328；KV 0.0078125 MiB；候选 router 分数 0.5572；平均/峰值规划 QPS 0.463/2.7778，并发 8.89，阶段预算 2.42 秒，KV 1265.62 MiB；三个 tile 大小的加权输出均为 2.897438；LoRA adapter 参数 65536，相对全矩阵比例约 0.0039，`record_complete=True` 且 `evidence_verified=False`。

61 个 math 围栏、花括号和环境结构检查通过，内部链接目标存在。未使用外部公式渲染器；PDF 构建仍受 Pandoc 缺失限制。上述结果不证明真实模型训练、语义事实核验、GPU kernel 性能、生产容量或安全隔离。
