# Performer/FAVOR+ 正值正交随机特征：研究记录

核对日期：2026-11-20。

## 研究问题与范围

核查 Performer/FAVOR+ 如何近似 scaled dot-product softmax kernel；正值随机特征、正交采样、有限样本误差、归一化 attention 输出与多层误差分别能得到什么结论；并与一般随机 Fourier 特征、对所选 kernel 的精确结合律重排、精确 attention 的 IO 优化划定边界。

## 检索范围与筛选

- 使用 web 搜索及来源直达页面；检索词包括 `Rethinking Attention with Performers ICLR 2021 FAVOR+ positive orthogonal random features`、`Performer FAVOR+ tutorial softmax kernel positive random features lecture notes`、`Random Features for Large-Scale Kernel Machines Rahimi Recht NeurIPS 2007` 和 `Rethinking Attention with Performers positive orthogonal random features theorem 2 theorem 4`。
- 优先核对原始论文和正式出版记录；并行阅读作者团队技术博客、技术讲解文章和一般随机特征原论文，以区分方法事实与教学组织。
- OpenReview PDF 直达页触发浏览器验证，未作为已读证据；改读可访问的 arXiv HTML v4。仅凭搜索摘要发现的其他教程不用于本记录的事实结论。

## 来源与阅读位置

### 原始方法论文

- Krzysztof Choromanski et al., [Rethinking Attention with Performers](https://arxiv.org/html/2009.14794), ICLR 2021；所读 arXiv v4，版本日期 2022-11-19。阅读摘要、§2.1—2.4（常规 attention、可核化 attention、softmax kernel 特征、正交随机特征）、§3 Theorems 1—4、§4.1—4.3（计算量、近似误差、误差传播）、图 4/5，以及 Appendix B.1—B.3（因果前缀和、ORF 扩展、复杂度）。该来源用于方法定义、定理条件、作者实验设置和复杂度口径。
- [Google Research publication entry](https://research.google/pubs/rethinking-attention-with-performers/) 列出论文作者并确认其为 ICLR 2021 oral presentation；用于核对正式出版身份，方法细节以论文原文为准。

### 原作者解释与教学组织

- Krzysztof Choromanski、Lucy Colwell，[Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/)，Google Research，2020-10-23。阅读 attention 的二次矩阵问题、稀疏路线背景、广义 kernel attention 和随机正值映射相关段落。作者博客适合建立“成对相似度—随机特征—低维计算”的解释地图；性能与理论保证不以博客替代论文。
- Teddy Koker，[Performers: The Kernel Trick, Random Fourier Features, and Attention](https://teddykoker.com/2020/11/performers/)，2020-11-11。阅读 Kernel Trick、Random Fourier Features、Attention、Softmax Kernel、Orthogonal Random Features 和 Positive Random Features 相关段落。其组织顺序适合教学参考；文章有时将随机 Fourier 特征作为 Performer 的总称，并展示了三角特征示例，故仅用于比较讲解路线，不将该实现等同于 FAVOR+，事实以原论文为准。

### 基础随机特征来源

- Ali Rahimi、Benjamin Recht，[Random Features for Large-Scale Kernel Machines](https://papers.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)，NeurIPS 2007。阅读正式会议页面摘要：随机低维特征内积近似用户指定的平移不变 kernel。它支持一般随机 Fourier 特征的背景，不能单独支持 softmax kernel 的正值映射、Performer 公式或相应误差界。

## 核验发现

1. 标准 scaled dot-product 的未归一化相似度为 $\exp(q^{\mathsf T}k/\sqrt{d_k})$。令 $x=q/d_k^{1/4}$、$y=k/d_k^{1/4}$，目标 kernel 写为 $\exp(x^{\mathsf T}y)$。缩放必须同时应用于 query 和 key，不能把标准 softmax 的 $1/\sqrt{d_k}$ 漏掉。
2. 对 $\omega\sim\mathcal{N}(0,I)$，令 $\psi_\omega(x)=\exp(\omega^{\mathsf T}x-\lVert x\rVert^2/2)$。高斯矩母函数给出 $\mathbb{E}[\psi_\omega(x)\psi_\omega(y)]=\exp(x^{\mathsf T}y)$。用 $m$ 个方向构造 $\phi_m(x)=m^{-1/2}[\psi_{\omega_1}(x),\ldots,\psi_{\omega_m}(x)]^{\mathsf T}$，其内积是未归一化 softmax kernel 的无偏 Monte Carlo 估计；单次有限样本仍有随机误差。
3. “正值”描述映射分量是正指数值，因此由内积形成的未归一化估计分数为正。它避免三角随机特征直接产生负的 noisy kernel estimate，但不保证指数运算不会溢出，也不自动保证归一化后的输出准确。
4. “正交”描述随机投影方向 $\omega_i$，不是正值特征向量。论文讨论在各向同性采样下保持边际分布并施加方向正交，以降低估计误差；基础构造的 $m\leq d$，Appendix B.2 对 $m>d$ 使用分块扩展，不能说超过 $d$ 个方向在 $d$ 维空间中全部互相正交。
5. 论文 Theorem 2 给出正值 ORF kernel estimator 相对于 IID 正值 estimator 的 MSE 改进式，条件包括该 ORF 构造与 $m\leq d$。论文 Theorem 4 则针对**三角特征加 ORF** 的 attention matrix approximation；其 uniform convergence 假设 query/key 的 $L_2$ 范数受 $R$ 限制，特征数依赖维度、半径和目标精度，不依赖序列长度 $L$。该结论不能改写成所有 FAVOR+ 设置的统一保证，更不能直接当作归一化输出或任务准确率定理。
6. 论文 Figure 4 在 $L=4096,d=16$、变化的 $m$ 与 15 组适当归一化随机矩阵输入下比较 attention approximation MSE；图中 ORF 优于 IID、正值特征优于三角特征的结果限定于该设计。§4.3 报告了 attention 局部误差在多层模型中的传播，并研究 LM1B、PG-19 上不同 feature/redrawing 设置；这些不是所有训练任务的统一结果。
7. 在固定 $m$ 时，映射与聚合的主要成本随序列长度 $T$ 线性增长；分别计入 query/key 特征计算与 value 聚合，可记为 $O(Tm d_k+Tm d_v)$。因果逐步使用的累计状态约为 $O(md_v+m)$，随机方向存储、Q/K/V 投影、MLP、训练反向中间量和具体实现开销需另计。渐近成本不推出特定硬件的 wall-clock 优势。
8. 若 kernel 分子和分母分别记为 $N,D$，其估计为 $\hat N,\hat D$，则归一化输出误差满足 $\hat y-y=((\hat N-N)-y(\hat D-D))/\hat D$（$\hat D>0$）。因此 kernel estimator 无偏不意味着比值输出无偏；小分母会放大联合估计误差，多层网络还可能继续传播局部误差。

## 证据边界与未核事项

论文支持其随机映射定义、估计器特定的 MSE/收敛结果、算法复杂度和所报告实验条件；不支持将这些结果外推为任意有限特征 attention 的保证、所有模型的任务质量结论或任意 GPU 的速度优势。一般随机 Fourier 特征的原始工作主要研究平移不变 kernel，不能替代 Performer 对 softmax kernel 的专门构造。随机采样误差与浮点数值稳定是不同问题；正特征并不能单独解决数值溢出、underflow 或低精度累积。未运行本地实现或 benchmark，也未检查 GitHub/PDF 公式视觉渲染。
