# 第二十五章精修审计

日期：2026-11-20。

## 任务与读者结果

本轮按全套书精修执行主题重写，并先按 02a 完成本章结构设计。保留章节标题、路径与位置；旧稿 25 个 H2 存在重复和职责混杂，无法清楚区分对所选 kernel 的精确重排、Performer 对 softmax kernel 的随机近似、输出误差及理论边界，因此先形成六节完整大纲，再按新结构重写。

读者应能从 scaled dot-product softmax 导出对应指数内积，推导 FAVOR+ 正值特征的无偏 kernel 估计，解释 PRF 与 ORF 的不同作用，说明 kernel 估计无偏不推出归一化 attention 输出无偏，并按定理的估计器、假设和实验条件解读其误差保证。

## 旧稿诊断与结构决定

旧稿将 Performer 与 Linear Transformer 混为一条泛称“线性 attention”的路线；用通用随机 Fourier cosine 映射示例和代码介绍随机特征，却未准确区分 FAVOR+ 的正值特征构造。后续多个小节反复讨论 feature 数、随机种子、kernel MSE、top-k evidence、部署回退、线上 trace、fallback、benchmark matrix 和实验决策伪代码，也重复第二十四章的前缀状态、状态容量和复杂度说明。

现结构按精确重排与随机近似、正值特征推导、FAVOR+ 的正值/正交设计、归一化输出误差、定理和实验边界、成本与 FlashAttention 区分六节组织。保留了有读者价值的有限样本波动、MSE/收敛假设、多层误差传播和固定特征数成本；删除不属于架构教学承诺的可执行代码、部署选择与回退流程、日志字段和实验门禁。第二十四章继续负责基本状态推导，第二十六至二十八章负责稀疏路线。

## 研究与事实核验

新增[Performer/FAVOR+ 正值正交随机特征研究记录](../../research/performer-favor-plus-random-features-2026-11-20.md)。核对 Choromanski 等人的 ICLR 2021 原论文 arXiv v4 摘要、§2.1—2.4、Theorems 1—4、§4.1—4.3、Figure 4/5 与 Appendix B.1—B.3；Google Research 官方出版页确认 ICLR 2021 oral，作者博客和 Teddy Koker 教学文章用于比较讲解顺序，Rahimi–Recht NeurIPS 2007 记录用于限定一般随机 Fourier 特征的背景范围。

正文统一使用列向量和 $d_k$ 缩放口径；以高斯矩母函数导出 PRF，无偏性限定为对随机方向取期望。解释 ORF 方向而非特征向量的正交性、$m\le d_k$ 的基础构造条件和 $m>d_k$ 的分块方式。区分 Theorem 2 的正值 ORF kernel estimator MSE 比较与 Theorem 4 的有 query/key 范数上界的三角 ORF attention-matrix 收敛结论；不把任一结论扩为归一化输出误差或通用任务质量保证。实验数字只保留 Figure 4 的 $L=4096,d=16$、$m$ 扫描及 15 组归一化随机矩阵输入条件；多层实验按原论文模型与数据集限定。

## 同步

- 新建[第二十五章结构大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/25-Performer与Linear-Transformer.md)，并在第二十一册目录增加大纲入口。
- 新增练习 31AD、面试题 31AA；术语表增加 FAVOR+、PRF、ORF；PAPERS 架构索引新增一般随机特征原始工作第 40 项；知识图谱扩展 Performer/FAVOR+ 到正值正交特征和归一化输出误差的关系。
- 更新第二十一册架构入口为第 1—25 章完成、26—95 章共 70 章待审；下一入口为第二十六章。

## 静态验收与遗留项

- 正文 H2 连续为 25.1—25.6；七个顶格 `math` 块围栏配对。源码复核了 scaled dot-product 的 $d_k^{1/2}$ 缩放分配、Gaussian MGF、有限特征 $1/\sqrt{m}$ 系数、状态 $S_t\in\mathbb{R}^{d_v\times m}$ 与 $z_t\in\mathbb{R}^{m}$、分子分母误差恒等式和固定 $m$ 的复杂度口径。
- 正文、大纲、目录、研究记录、配套索引和本章审计中的新增本地链接目标均可定位；练习 31AD、面试题 31AA、FAVOR+/PRF/ORF 词条和 PAPERS 第 40 项唯一。`git diff --check` 通过。
- 以上属于源码、公式含义和静态链接复核。GitHub MathJax 页面未逐式查看，本章公式渲染仍待验收；目标读者试读、外链逐项复核和第二十一册连续复读留待后续全册门禁。
- 未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。
