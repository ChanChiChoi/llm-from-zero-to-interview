# Mamba-2 与 State-Space Duality：结构、算法和证据

研究日期：2026-11-23

## 研究问题与范围

核实 SSD（Structured State Space Duality）和 Mamba-2 的关系：递归形式在什么条件下能写成结构化矩阵；这种矩阵为何与 attention 相似但不等同于标准 softmax attention；分块算法如何把局部矩阵乘法与跨块状态连接起来；论文中的效率结果支持到什么范围。本记录供第二十一册第三十五章使用，不讨论 serving 系统设计。

## 检索与阅读

- 查阅 Dao 与 Gu 的 ICML 2024 正式论文页及 arXiv HTML 版。重点阅读 arXiv 版本 §§2.3–2.4、3.1–3.4、4.3、5.1–5.3、6、7.1–7.3、9.3 及附录中的 state-passing 算法。
- 并行查阅方法作者 Tri Dao 关于 Mamba-2 的四篇技术说明，分别核对模型、理论、算法和系统视角。作者文章用于理解与组织机制；涉及定理和性能的判断回到论文。
- 查阅官方 Mamba 仓库中的最小 SSD 模块，作为实现入口。所查链接指向可变的 main 分支，只说明当前文件位置，不能代表论文定义、固定发布版本或实际运行结果。
- 检索入口为 web 检索及直接打开的论文、作者站和官方仓库页面；英文关键词围绕 Mamba-2、State-Space Duality、semiseparable matrix、SSD chunk algorithm 和 speed benchmark。筛选范围限于能直接说明代数条件、块算法、Mamba-2 block 或论文效率结果的材料。论文和作者说明对应 2024 年工作；官方 main 分支按 2026-11-23 页面状态阅读。

## 来源与证据边界

| 来源 | 阅读位置 | 可支持的内容与限制 |
| --- | --- | --- |
| Dao & Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality,” ICML 2024, PMLR | [正式论文页](https://proceedings.mlr.press/v235/dao24a.html)；[arXiv HTML](https://arxiv.org/html/2405.21060)，重点见 §§2.3–2.4、3、4.3、5、6、7、9.3 | 支持论文定义、推导、算法、实验设置及作者报告结果。论文中的结果不能直接推出其他模型、硬件、完整模型或 serving 的普遍表现。 |
| Tri Dao, “Mamba-2: Transformers are SSMs” 系列 | [Part I: Model](https://tridao.me/blog/2024/mamba2-part1-model/)；[Part II: Theory](https://tridao.me/blog/2024/mamba2-part2-theory/)；[Part III: Algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/)；[Part IV: Systems](https://tridao.me/blog/2024/mamba2-part4-systems/) | 方法作者对模型、理论、分块算法和系统动机的讲解，可用于核对教学解释；跨模型结果仍应按论文中的条件表述。 |
| State Spaces Lab, 官方 Mamba 仓库 | [ssd_minimal.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py) | 当前 main 分支的实现示例入口。main 可变；未在本地读取或运行该模块，不用它声称任何本地复现或稳定版本行为。 |

## 核心发现

### 递归与结构化矩阵

论文研究的 SSD 核心层把每个时间步的状态转移矩阵限制为标量乘单位矩阵；输入相关的写入向量和状态读出向量仍随时间变化。在论文采用的离散参数化中，标量状态转移 $a_t$ 取值在 $[0,1]$。对一个 head，可将状态记为 $H_t\in\mathbb{R}^{N\times P}$，输入和输出记为 $X_t,Y_t\in\mathbb{R}^{P}$，写入和读出向量记为 $b_t,c_t\in\mathbb{R}^{N}$。因此 $N$ 是状态维度，$P$ 是 head 的通道维度，两者不能混称。

展开递归后，位置 $i$ 到 $t$ 的影响由状态衰减乘以读写向量内积决定。序列变换矩阵是因果下三角的半可分离矩阵；其下三角子块秩受状态维度 $N$ 限制。半可分离不表示整张矩阵低秩，也不能把所有跨块贡献说成秩一。

### Attention-like 矩阵的边界

在论文给出的参数化下，可将矩阵写为 $M=L\odot(CB^\mathsf{T})$，输出为 $Y=MX$。这里 $L$ 由标量状态转移的累积乘积构成并提供因果掩码；$B,C,X$ 分别可与 key、query、value 作受限类比。该形式不含标准 softmax，并增加了特定的结构化掩码。因此它只表明特定 SSM 子类与特定 structured masked attention 形式重合，不表示任意 SSM、任意 attention 或标准 Transformer 全部等价。

矩阵视图若显式构造完整的 $T\times T$ 矩阵仍需二次存储/计算；它主要揭示结构并帮助设计算法，不意味着直接展开矩阵就高效。输入相关的 $B,C,a$ 也意味着该线性变换矩阵本身随输入变化，不能把“对 $X$ 的矩阵乘法”误读为整个网络对输入是固定线性的。

### 分块算法与效率结果

SSD 的分块算法将序列分为若干 chunk：先计算各 chunk 内、假设初始状态为零时的局部输出和状态摘要；再按真实因果顺序扫描 chunk 摘要，求得每个 chunk 的真实入边界状态；最后将边界状态的贡献投影回该 chunk 的输出。块对角部分对应 chunk 内局部计算，块外部分通过状态摘要因式分解并跨块传递。算法不必物化完整 $T\times T$ 矩阵，但跨块状态仍须保持连续，不能在 chunk 边界重置。

论文摘要报告，专门实现的 SSD 核心层相对 Mamba 的优化 selective scan 实现快 $2$–$8$ 倍，并可使用更大的状态维度；这是作者在论文模型、实现、硬件和序列设置下的核心层比较，不是所有完整模型或服务请求的速度保证。Mamba-2 block 还包含投影和归一化等设计变化；它们不都属于 SSD 对偶关系本身。

## 使用限制与待核

- 公式说明是论文特定标量状态转移子类的结果；参数化、因果范围、初始状态和张量维度需与具体模型一致。
- PMLR 与 arXiv 论文支持机制和论文实验；作者博客支持其明确给出的讲解；官方可变分支源码只作当前实现入口。
- 没有运行本地代码或模型，也没有复现论文 benchmark。不能据本记录宣称本地数值一致、吞吐改进或生产性能。
- 论文的 2–8 倍结果不能外推为全模型、任意硬件、任意 chunk、训练配方或 serving 的通用收益。
