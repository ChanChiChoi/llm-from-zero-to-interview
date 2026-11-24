# 第三十五章《Mamba-2 与 State-Space Duality》精修审计

## 范围与诊断

按全书精修的主题重写要求完成本章，保留章节文件、章名和目录位置。旧稿有 27 个 H2，反复讲述 duality 的定义、chunk size、状态连续性、数值等价和验证流程；还把 serving snapshot/restore、错误恢复、hybrid 调度和生产回归写入机制章。旧代码逐 token 循环处理输入，并不实现 SSD 的分块矩阵算法，容易让读者把演示代码误当成算法支架。

先按 02a 重组为六节，完整大纲见[第三十五章结构大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/35-Mamba-2与State-Space-Duality.md)。章节现在沿“SSD 对象区分 → Mamba-2 状态参数化 → 递归展开 → attention 类比边界 → 分块状态传递 → 证据适用范围”推进。serving、恢复协议和 hybrid 架构细节不属于本章读者结果，分别交给第六册、第二十四册和第四十三章。

## 内容与证据

正文说明单 head 的状态形状、状态维度 $N$ 与 head 维度 $P$、标量乘单位矩阵转移，以及从递归到 $M=L\odot(CB^\mathsf{T})$ 的矩阵表达。半可分离的秩限制落在相关三角子块；没有称整张矩阵低秩或把块外作用概括为秩一。正文明确区分该 attention-like 形式与标准 softmax attention，并指出输入相关参数使 $M$ 不是对所有输入都固定的矩阵。

分块部分以 chunk 的仿射状态摘要解释局部输出、摘要扫描和真实边界状态读出，明确 chunk 边界不能重置状态，也不需要显式物化完整 $T\times T$ 矩阵。论文的 $2$–$8$ 倍结果限定为其专门实现的 SSD 核心层相对优化 selective scan 的条件性比较；没有写成完整模型或 serving 的通用加速保证。

原始来源、作者教学材料、官方实现入口及其各自边界见[Mamba-2/SSD 研究记录](../../research/mamba-2-ssd-duality-2026-11-23.md)。没有本地运行实现或复现论文 benchmark。

## 配套同步

已将第二十一册目录标题与正文一致，并加上大纲链接；新增练习 31AN、面试题 31AK、Structured State Space Duality / SSD 与 Semiseparable Matrix 术语；补充 PAPERS.md 的正式论文入口和 KNOWLEDGE_GRAPH.md 中从状态转移到 SSD 分块算法的路径。

## 验收与遗留

正文 H2 35.1—35.6 与大纲标题逐项一致。公式源码检查覆盖 11 个顶层 math 块，所有块公式围栏和行内美元定界符均成对；未使用旧式括号行内定界符或 operatorname 命令。递归展开、矩阵尺寸、因果索引、边界状态形状和教学算例的数值关系已复核。练习 31AN 与面试题 31AK 各唯一；章节、大纲和本轮新增索引目标存在。全文件 PAPERS.md 链接扫描另外发现 23 个既有缺失本地路径，均与本轮新增的第 19 条无关，未在本章任务中扩展修复。

git diff --check 通过。GitHub MathJax 页面实际渲染尚未验收，不能标为公式渲染完成；外链最终逐条复核、目标读者试读和第二十一册连续复读仍待全册门禁。没有代码需运行；未运行测试、模型、API、kernel、benchmark 或 PDF 构建，不更新 PDF 或 plan.md。下一入口为第三十六章《SSM 会不会替代 Transformer：证据、争议与面试回答》；全套书精修 goal 继续 active。
