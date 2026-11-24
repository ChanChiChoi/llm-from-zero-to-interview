# 第二十章《FlashAttention、PagedAttention 与 Attention Kernel 优化》精修审计

审阅日期：2026-11-20。章节职责复核、主题研究、正文主题重写、结构大纲、配套同步和本轮静态验收已完成。两处块公式的 GitHub MathJax 实际渲染、目标读者试读、外链逐条终审和第二十一册连续复读仍待后续门禁；按用户要求不更新 PDF 或 `plan.md`。全套书精修 goal 继续 active。

## 读者契约与章节边界

- **前置：** 第十章的 attention pair 数与在线 softmax；第十一章的逐请求 KV 状态增长。读者不需要 CUDA 编程经验。
- **离章结果：** 能说明 dense FlashAttention 如何保留完整 attention 语义并减少中间数据搬运；区分 FA2/FA3/FA4 的并行与硬件适配；解释 Flash-Decoding 为何沿 K/V 维分段、如何合并 softmax 状态；读懂 PagedAttention 的逻辑 block、block table、物理 KV block 与共享前缀写时复制；区分 kernel latency、KV 空间和端到端服务指标。
- **范围：** 本章负责 dense attention kernel、decode 分段并行、Paged KV 映射和共享前缀正确性。第十章负责 pair 计算及在线 softmax完整推导；第十一章负责 KV payload 与请求状态；第十九章负责改变可见连接的 pattern；第二十六章负责块稀疏布局和动态稀疏执行；第六册与第二十四册负责服务调度和端到端性能框架。

## 旧稿诊断与结构决定

旧稿基线为 291 行、23 个 H2。FlashAttention、Flash-Decoding、PagedAttention、分配器和 serving 性能主题多次重复；一个 blocked-attention 教学代码未实现在线 softmax，公式记号与归一化解释不完整；多个测试 gate、benchmark 流程和服务指标越出本章架构/算子读者结果。

按章节精修采用主题重写，保留章名、路径和目录入口，将正文重组为六节：优化对象分类、FlashAttention IO 路径、FA 版本与 Flash-Decoding、PagedAttention 地址映射、共享前缀写时复制、后端组合与证据口径。完整章节设计见[第二十章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/20-FlashAttention与PagedAttention.md)。

## 研究核验与正文决定

来源版本、实际阅读范围和证据边界见[Attention kernel 与分页 KV 研究记录](../../research/attention-kernel-and-paged-kv-evidence-2026-11-20.md)。正文采用的主要判断：

- FlashAttention 的 dense exact 路径改变 tile 计算与 HBM/片上存储之间的数据移动，不删除允许的 query-key pairs；浮点次序变化不承诺逐位一致。详细 online softmax 推导回链第十章。
- FA2 的 work partition、FA3 的 Hopper 异步数据/计算流水与低精度路径、FA4 的 Blackwell 特定流水分别按论文设备、shape、dtype 与 baseline 限定。FA4 的 cuDNN 对照版本变化也写明，未外推为当前通用排名。
- Flash-Decoding 针对少 query、长历史 KV 的并行度，将各 K/V 分段的 $(m_j,\ell_j,u_j)$ 统计按共同最大值重标定后合并，保留同一可见 key 集合；归并成本和收益依 workload。
- PagedAttention 使用 0 起点的 block index、block 内 offset 和 request block table 表达逻辑到物理映射；分页减少最大长度预留浪费，不缩小活跃 token 的 KV payload。block size 的元数据与尾块浪费取舍、非连续访问成本均保留。
- 原论文在其特定设置下报告 PagedAttention kernel latency 高于 FasterTransformer 20%–26%；该结果与系统吞吐分开解释，不外推到当前 kernel。
- 共享前缀要求 K/V 计算条件兼容；引用计数与 copy-on-write 保护仍被其他请求引用的 block。prefix cache key 和淘汰行为限定为具体 engine 版本。
- PyTorch SDPA 以 2.15 文档为版本边界，说明融合 backend 的限制、受控选择入口和浮点差异；API 名称本身不能证明实际运行的 kernel。

## 删除与同步

- 删除旧稿重复的 20.x 主题、错误的 blocked-attention toy code、公式/输出重复解释、泛化 serving benchmark gate 和面向不同书册的调度验收流程；这些形式不服务本章读者结果。
- 新建结构大纲；本章目录标题与文件路径未变，不改目录项。
- 同步 `PAPERS.md` 的 FA4、Flash-Decoding 来源与研究记录入口；在术语表增加 Flash-Decoding，在知识图谱增加 dense attention IO 与 decode 分段归并关系。
- 在 `EXERCISES.md` 增加 31Y（Paged KV 映射、COW 和方法辨析），在 `INTERVIEW_BANK.md` 增加 31V（decode 并行、统计归并和 kernel/系统边界）。
- 正文链接到第十九章、第二十六章、第六册调度及第二十四册 serving benchmark 入口。

## 静态检查与遗留门禁

- 正文 H2 为连续的 20.1—20.6；两处独立 `math` 块公式为分段 softmax 统计合并和 KV block 地址映射。已检查符号定义、归一化含义、0 起点索引与 block table 对应关系。
- 正文、结构大纲、审计和本轮相关同步文件的本地 Markdown 链接目标均已检查；围栏闭合，行内数学定界成对，表格列结构一致；练习 31Y 与面试题 31V 各自唯一。
- 旧稿小节序号在第二十一册当前导航和本轮配套索引中无残留。相关文件 `git diff --check` 通过。
- 正文无代码块或 Mermaid；未运行测试、代码、模型、API、kernel benchmark 或 PDF 构建。
- 两处公式的目标平台视觉渲染仍待完成；目标读者试读、外链逐条终审、第二十一册连续复读及本册原先遗留公式的 GitHub 渲染仍待全册门禁。

下一入口为[第二十一章](../../book-21-transformer-architecture-evolution/chapters/21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)。全套书精修 goal 继续 active。
