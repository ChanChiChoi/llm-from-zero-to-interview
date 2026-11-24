# MoE 总参数与激活参数：来源和证据边界

检索日期：2026-11-24  
研究问题：Total Parameters 与 Active/Activated Parameters 分别统计什么；MoE 的逐 token 路径、模型发布数字、参数存储字节和计算量之间有哪些可确认的关系？  
范围：以 MoE 每层专家计数和公开模型披露为主；不把本记录扩展为训练预算或推理服务性能综述。

## 检索范围

通过 arXiv、Hugging Face 模型卡、Stanford MS&E 319 课程资料及 Hugging Face 技术博客查找一手报告、教学讲义和实现解释。检索词包括：MoE total parameters / active parameters / activated per token、expert count and top-k、mixture of experts lecture notes、Hugging Face MoE weight loading。2026-11-24 的网页搜索还使用了以下原始查询：site:sebastianraschka.com mixture of experts total parameters active parameters MoE blog；site:huggingface.co/blog mixture of experts active parameters total parameters explanation；site:newsletter.maartengrootendorst.com mixture of experts active parameters total parameters。课程讲义和技术博客用于核对解释路径与其作者直接报告的实现；模型数字回到论文或模型卡。

## 来源与支持范围

1. **Mixtral of Experts**，Mistral AI 作者，arXiv:2401.04088，v1，2024-01-08 提交。阅读摘要及架构概述。原文说明每层由 8 个 feed-forward experts 组成，router 对每个 token 选择两个 experts；论文报告 47B 参数总量和推理时 13B active parameters。它支持该报告版本中的架构和报告值，不支持从 8:2 比例精确重算全模型数字，也不证明 active parameters 等于显存或实测 FLOPs。[论文](https://arxiv.org/abs/2401.04088)

2. **DeepSeek-V3 Technical Report**，DeepSeek-AI，arXiv:2412.19437v2，2025-02-18 修订。阅读 HTML 摘要、§2.1.2 和 §4.2。报告给出 671B total、每 token 37B activated；§4.2 说明除前三层外使用 MoE，MoE 层含 1 个 shared expert 和 256 个 routed experts，每 token 选择 8 个 routed experts。shared 与 routed 路径、dense 前三层和模型其他参数共同影响发布的整模型数字；不应只用 8/256 推导 37B。[技术报告 v2](https://arxiv.org/abs/2412.19437v2) · [HTML §4.2](https://arxiv.org/html/2412.19437v2)

3. **Qwen3-30B-A3B model card**，Qwen 官方 Hugging Face 模型页，模型 ID Qwen/Qwen3-30B-A3B，访问日期 2026-11-24。阅读 “Model Overview” 项。模型卡当时列出 30.5B total、3.3B activated、29.9B non-embedding、128 experts、8 activated experts。它说明同一模型页可同时披露不同统计字段；此处是访问时的在线页面快照，未固定 Git revision，不能据此声称今后页面字段不变，也不能把各字段相减当作完整组件账本。[官方模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B)

4. **MS&E 319: Efficient Generative Language Models, Lecture #4: Mixtures of Experts**，Anay Mehrotra，Stanford，2026-10-01。阅读 §1 和 §1.1。讲义把每个 expert 的参数量记作 $P$、expert 数记作 $E$、每个输入选出的数量记作 $k$，给出专家库 $EP$ 与单 token 选中部分 $kP$ 的区分；并指出 router 和通信有额外代价、整模型仍需保存全部专家参数。它是课程教学定义和简化计算依据，不代表所有 MoE 报告采用相同的全模型计数口径。[课程讲义 PDF](https://web.stanford.edu/class/msande319/lectures/lecture-4-notes.pdf)

5. **Slicing and Dicing: Configuring Optimal Mixtures of Experts**，Margaret Li、Sneha Kudugunta、Danielle Rothermel、Luke Zettlemoyer，arXiv:2605.11689v1，2026-05-12。阅读摘要、§3、§3.1。论文报告超过 2,000 次预训练、总参数最高 6.6B；实验规模约为 10M–300M active parameters，并按 active FFN 参数对每步 FLOPs 作匹配。作者在其被测设置中报告增加 inactive experts 参数能改善 held-out loss。该结论是 arXiv v1 研究结果，绑定其模型、数据、预算和评估设置；不能外推为 frontier 模型的普遍规律或质量随 total 参数单调增长的定律。[arXiv v1](https://arxiv.org/abs/2605.11689v1) · [HTML §3–3.1](https://arxiv.org/html/2605.11689)

6. **Mixture of Experts (MoEs) in Transformers**，Hugging Face Transformers 团队博客，2026-02-26。阅读 “From Dense to Sparse” 和 “Weight Loading Refactor”。它把逐 token router 选择作为入门解释，并以 Transformers 的 WeightConverter 描述 checkpoint 中分开的 expert tensors 如何转换为 runtime packed tensor。后者支持“checkpoint 排列方式与 runtime tensor 布局不是同一问题”的实现例子，只限定于该 Transformers 加载路径；博客中基于 active 参数估算特定设备 tokens/s 的数字是指定模型、设备和 workload 的案例，不作为普遍速度公式。[技术博客](https://huggingface.co/blog/moe-transformers)

## 可支持的稳定区分

- Total 参数统计所有被发布口径纳入的共享权重、所有 experts 及其他参数；口径可能对 embedding、output head、router、shared expert、tied weights 和非嵌入参数作不同处理。
- 在等大小专家的 MoE 层，expert bank 有 $EP$ 个参数，而一个 token 选择 $k$ 个时，选中专家部分为 $kP$。当专家大小不相等或每层路由规则不同时，应按层、按选中集合求和；$k/E$ 只表示等大小 expert bank 的比例。
- 一个 token 在不同 MoE 层可选择不同专家。批次/请求内所有 token 触及的专家并集可能大于任意一个 token 的路径；并集不是逐 token active 参数，也不是每张设备上的 resident bytes。
- 参数个数乘存储格式每参数字节数，只估算声明格式下的原始权重载荷。量化 metadata、scale、padding、布局转换、sharding/replication、KV cache、activation、workspace 与 runtime 分配会改变实际存储和显存。
- Active parameters 是按 token 路径统计的参数口径，不等于严格 FLOPs、tokens/s、延迟、服务成本或质量。相同尺寸矩阵下可与专家矩阵乘计算量近似相关；跨架构/任务/实现比较仍需相同口径和实测。
