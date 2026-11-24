# Latent Cache 表示压缩与任务保真研究记录

检索与原文核验日期：2026-11-24。

## 研究问题与检索范围

本轮为第二十一册第七十章精修，研究低秩/latent 表示如何存放逐 token 的 KV 历史、attention 如何读取这类表示、它与 head sharing、量化、token selection 和递归状态的区别，以及压缩率、任务质量和推理成本应怎样一起解释。重点核对低秩表示的原始架构、RoPE 处理、读取时重构/吸收计算、公开结果的模型与实验边界；教学来源用于改进概念路线，不作为性能结论的替代证据。

检索平台：arXiv HTML、ACL Anthology 论文 PDF、NeurIPS Proceedings 论文 PDF、研究者技术博客、公开课程讲义。

检索词包括：`latent KV cache low-rank compression MLA`、`KV-Latent ACL 2025 dimensional reduction RoPE`、`SALS sparse attention latent space KV cache`、`MLA weight absorption inference tutorial`、`DeepSeek MLA lecture notes dataflow`。

## 一手研究来源

### DeepSeek-V2：每个历史 token 的联合 K/V latent

- **来源与版本**：[DeepSeek-V2 arXiv v5](https://arxiv.org/html/2405.04434v5)，DeepSeek-AI，2024-06-19；阅读 §2.1.1–2.1.4，特别是 §2.1.2 的 Eqs. 9–13、§2.1.3 与 Table 1。
- **支持事实**：输入 hidden state 经 $W^{DKV}$ 投影成每 token 的共享 latent $c_t^{KV}$，K/V 分别由 $W^{UK}$ 和 $W^{UV}$ 从该 latent 投影出来；推理缓存保留每个历史位置的 latent。论文指出 K 上投影可吸收到 Q 路径，V 上投影可吸收到输出投影，避免必须显式展开整段 K/V。DeepSeek-V2 另将 RoPE 位置 key 与 content latent 解耦，因此 cache 元素账本为 $(d_c+d_h^R)$ 每 token、每层，而非只算 $d_c$。
- **作者报告及边界**：摘要报告相较 DeepSeek 67B 的 93.3% KV-cache 减少和最高 5.76 倍 generation throughput；这是包含 MLA、MoE 和其他模型设计的整模型对比，不能归因于 latent cache 单一改动。Table 1 的“capability”评价是作者对其配置的判断，不作普遍质量定律。

### KV-Latent：缩减既有注意力头维度并继续训练

- **来源与版本**：[KV-Latent ACL 2025 正式论文](https://aclanthology.org/2025.acl-long.77/)，Luohe Shi 等，ACL 2025，pp. 1535–1550；[论文 PDF](https://aclanthology.org/2025.acl-long.77.pdf)。阅读 §2.1、§3.2–3.3、§4.1–4.3、Tables 1–4 与 Limitations。
- **支持事实**：与 DeepSeek-V2 以共同下投影学习 K/V latent 的方式不同，KV-Latent 直接缩减既有 attention head 的 Q/K 和 V/O 维度，再以两阶段继续训练恢复能力；论文按任务将 $d_{qk}$ 与 $d_{vo}$ 分开，并针对低维 RoPE 修改频率采样。作者报告额外训练量低于原预训练量的 1%；实际实验包含 1B FineWeb-Edu tokens、LLaMA-3-8B GQA 与 LLaMA-2-7B MHA 配置，在单节点 8 张 A100 80G 上训练。LLaMA-3-8B 的 64/64 配置在其 Table 1 中将 4K 序列 BF16 cache 从 491 MB 降到 245 MB，MMLU/OBQA/ARC 平均分为 42.1 到 41.3，所报 NIAH pass ratio 均为 92%；缩到 16/16 时 NIAH 降至 6%。论文还报告其测试中 $d_{vo}$ 对性能更敏感。
- **证据边界**：指标和训练/硬件预算绑定该论文的 checkpoint、1B-token continuation recipe 和评估任务。NIAH 将一句短文本随机放入 3,840-token 上下文、重复 50 次，不能代表普遍长文档检索；作者明确指出更低维配置在有限继续训练下无法恢复能力。$d_{vo}$ 敏感性是该实验的发现，不是对所有模型的通用 K/V 信息定律。论文未与所有相关方法进行可比直接实验。

### SALS：把低秩表示压缩与 token 选择组合

- **来源与版本**：[SALS NeurIPS 2025 正式论文](https://papers.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html)，Mu 等；[论文 PDF](https://papers.neurips.cc/paper_files/paper/2025/file/00a0ebcad584c59dbc439c2af8793638-Paper-Conference.pdf)。阅读 §3.1–3.2、§4.1、§5.2、§5.4–5.5。
- **支持事实**：SALS 用低秩投影把 pre-RoPE 多头 key 映射到紧凑 latent 空间，在该空间打分并选 token，只对选出的 key 重建、再施加 RoPE 后做候选集 attention；论文称 value 张量在其分析中近 full-rank，因此采用按通道量化而非同样的低秩投影。它清楚展示表示维度压缩、token 选择和数值精度可组合，但各自影响 cache 维度、可访问位置和精度。
- **作者报告及边界**：评测使用 LLaMA2-7B-Chat、Mistral-7B-v0.2 的 GSM8K、CoQA、LongBench；RULER-128K 使用 LLaMA3.1-8B-Instruct。端到端 throughput 与 GPT-fast 的差距随长度变化；作者报告 8K 场景在其配置下的 speedup 为 2.13 倍，32K 为 4.57 倍，并指出选中 token 的 reconstruction overhead 会影响结果。数值绑定其单张 Ampere GPU、batch、kernel、比例和 baseline，不泛化为 latent cache 的普遍加速。

## 教学与实现材料

### DeepSeek MLA 的执行讲解

- **来源**：[DeepSeek’s Multi-Head Latent Attention](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)，Lior Sinai，原文 2025-02-22，作者于 2026-11-08 更新；阅读低秩下投影、KV cache、weight absorption 和 Julia 实现段。
- **用途与边界**：展示 $q^T(W^{UK}c)$ 如何把 K 投影与 query 计算重排，以及 V 投影可与输出矩阵合并；其 2026 更新特别修订了未吸收投影时的 decode 成本说明。适合作为机制与实现路径的讲解材料；Julia 代码与作者的时延观察不代表所有实现或 kernel。

### RPI Machine Learning and Optimization Lecture 19

- **来源**：[Spring 2026 Lecture 19 handout](https://cs.rpi.edu/~gittea/old-site/teaching/spring2026/files/lecture19-handout.pdf)，Appendix B，PDF pp. 53–55。
- **用途与边界**：以“下投影—仅缓存 latent—上投影 K/V”图解讲清数据流，并将 latent cache 与 GQA 的结构性 head sharing 对照。讲义中的尺寸/缓存比是教学摘要，需按 DeepSeek-V2 的位置分支和具体维度重新核算；不直接引用其简化比例作为目标模型账本。

## 结构结论与内容去向

- 第七十章聚焦逐 token latent 表示：latent 如何形成与被读取、位置分支怎样影响 cache、哪些方法改变特征维/每值位宽/可访问 token/历史时间表示、任务保真和系统成本怎样共同评估。
- MLA 的完整公式、DeepSeek-V2 位置路径和 Kimi K3 Gated MLA 的具体配置留在第 63 章及第 58/88 章；第 70 章只复用理解通用压缩所需的最小机制，并解释具体证据能支撑到哪里。
- 第 69 章负责混合架构中逐层 cache 表示与 ordinary KV payload；第 71 章负责递归 state 的生命周期、快照、迁移和事务；第 20 章、第二十四册负责 attention kernel、分页和服务运行时。通用 fallback、安全、snapshot 与多租户流程不属于第 70 章。
- 删除没有来源的通用“latent compressor encoder/decoder”、玩具文档场景、动态压缩比保证、可逆性/隐私/provenance 定律及通用错误预算。把“信息瓶颈”限作表示容量/任务保真的直观比喻；所核论文没有把 canonical Information Bottleneck 互信息目标列为 MLA 的训练目标。

## 未解决边界

- 没有在目标模型或 serving backend 上运行复现、kernel benchmark 或生产 trace；本文仅据公开论文与教学材料报告。
- 不同来源采用不同模型规模、训练数据、缓存定义和指标，不能横向拼接成一个普遍压缩率或质量/时延曲线。
- 目标读者复述与实际 GitHub MathJax 渲染尚待书系最终验收。
