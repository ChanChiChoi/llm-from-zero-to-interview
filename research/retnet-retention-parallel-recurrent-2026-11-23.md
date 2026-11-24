# RetNet retention、并行递归与 chunkwise 研究记录

## 研究范围

- 日期：2026-11-23。
- 对象：Sun 等人 2023 年 RetNet 论文中的 retention、multi-scale retention、parallel/recurrent/chunkwise recurrent 表示及论文实验边界；不将视觉版、Gated RetNet 或后续结构并入基础 RetNet 定义。
- 问题：三种计算形式如何对应同一映射；位置相位、head-wise decay、门控与归一化在论文中的职责；固定状态成本可以说明什么；原论文性能比较可以外推到哪里。
- 检索入口与查询：Web 搜索了 “Retentive Network A Successor to Transformer for Large Language Models”、 “RetNet retention mechanism lecture notes chunkwise recurrence”、 “RetNet explained multi-scale retention chunkwise recurrent blog”、 “site:github.com/microsoft RetNet TorchScale retention implementation” 和 “RetNet paper teaching explanation decay matrix GroupNorm swish gate”。随后阅读原论文 HTML、讲解 notes、官方项目说明及有固定 SHA 的实现修复记录。

## 核心发现

### Retention 的并行式与递归式

主要依据为 [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/html/2307.08621)，arXiv v4，2023-08-09，§§2.1—2.3、3.1—3.4。它是提出 RetNet 的一手预印本；§2.1 给出三种表示，§2.2 给出 multi-scale retention、门控和 score normalization，§2.3 给出 block 结构，§3 报告模型和效率实验。

对某个 head，令 query/key 宽度为 $d_k$、value 宽度为 $d_v$，$q_t,k_t\in\mathbb{R}^{d_k}$，$v_t\in\mathbb{R}^{d_v}$。忽略归一化细节时，causal decay matrix 的元素为 $D_{t,j}=\gamma^{t-j}$（$j\leq t$），未来位置为零。并行式写为 $Y=((QK^{\mathsf T})\odot D)V$。令状态 $S_t\in\mathbb{R}^{d_k\times d_v}$，递归式为 $S_t=\gamma S_{t-1}+k_t^{\mathsf T}v_t$、$y_t=q_tS_t$；从 $S_0=0$ 展开可得 $S_t=\sum_{j=1}^{t}\gamma^{t-j}k_j^{\mathsf T}v_j$。因此并行式与递归式来自同一历史加权和，状态尺寸不随已读 token 数增长，但仍依赖 head 的 key/value 宽度。

论文为位置引入相位变换：$\Theta_t=e^{it\theta}$，$Q_t=(XW_Q)_t\odot\Theta_t$，$K_t=(XW_K)_t\odot\overline{\Theta_t}$。$D$ 的衰减与因果结构和相位编码各有职责；不要把 $D$ 当成完整的位置表示。

### Multi-scale block 与 normalization

§2.2 将不同固定衰减率分配给 retention heads，论文默认设为 $\gamma_h=1-2^{-5-h}$（$h=0,\ldots,H-1$）。慢衰减 head 的单项贡献下降较慢，但半衰期只描述权重尺度，不等价于可准确检索事实的长度。多头结果按 head 分组做 GroupNorm，再经 Swish gate 和输出投影；完整网络在 retention、FFN 与残差连接之间组合。

RetNet 的 score normalization 不是 softmax，也不是泛化的“对 key/value 分母再求商”。论文 §2.2 先将 $QK^{\mathsf T}$ 按 $\sqrt{d_k}$ 缩放，将衰减矩阵按因果行和的平方根缩放，再将每行 score 除以 $\max(|\sum_j R_{t,j}|,1)$，之后与 $V$ 相乘；GroupNorm 另行处理各 head 的输出尺度。正文可以用文字概括这些操作，并明确它们稳定数值尺度且不产生 softmax 概率。

### Chunkwise recurrent 表示

论文 §2.1 Eq. 7 在 chunk 内使用 parallel retention，在 chunk 间递归传递状态。若一块长为 $B$，块内位置从 $r=0$ 编号，且 $S_{\mathrm{in}}$ 是该块前一 token 后的状态，则未写入归一化细节的等价展开为：

- 块内位置状态：$S_{i,r}=\gamma^{r+1}S_{\mathrm{in}}+\sum_{j=0}^{r}\gamma^{r-j}k_{i,j}^{\mathsf T}v_{i,j}$。
- 块结束状态：$S_{\mathrm{out}}=\gamma^B S_{\mathrm{in}}+\sum_{j=0}^{B-1}\gamma^{B-1-j}k_{i,j}^{\mathsf T}v_{i,j}$。

这表明 chunkwise 不是互相独立的窗口；块内计算可并行，块间状态不能因切块而清零。固定块长时，沿序列的块数随 $T$ 线性增加；块大小、head 尺寸和具体 kernel 仍影响内存与吞吐。论文的推导是映射等价，不保证不同实现浮点逐位一致。

### 论文结果的证据范围

论文 §3.1 使用 1.3B、2.7B、6.7B 模型，在 The Pile、C4、The Stack 的整理语料上训练 100B tokens。§3.3 的训练成本比较使用序列长度 8192、chunk 大小 512、8 张 Nvidia A100-80GB，并比较 vanilla PyTorch RetNet 与 Transformer/FlashAttention；作者明确说明 RetNet 当时未使用 kernel fusion。§3.4 的推理成本实验为 6.7B 模型、A100-80GB。正文可以把它们作为架构提出时的具体报告条件，不应写成现代 GPU、任意实现或所有工作负载上的普遍速度优势。

## 教学来源与实现版本

- [Calvin Woo, “Linear Attention, Chunkwise Training, and Neural Memory”](https://calwoo.github.io/notes/concepts/attention-mechanisms/linear-attention/index.html)，阅读 §5.2 及 chunkwise retention 推导。该 notes 用状态展开、衰减权重与块边界状态帮助建立讲解路径；没有明确发布日期。仅用于教学组织和符号交叉检查，论文公式仍以原论文为准。
- [Microsoft UniLM RetNet README](https://github.com/microsoft/unilm/blob/master/retnet/README.md) 记录官方 TorchScale 项目的实现入口和历史修正：2023-08-04 的 chunkwise bug 修复、recurrent 数值精度修正，以及 2023-11 的 retention normalization 修复。README 指向的 [chunkwise 修复 commit `0b1f113`](https://github.com/microsoft/torchscale/commit/0b1f113985a0339bc322b0c7df91be0f745cb311)、[recurrent 数值修正 commit `7f0bf80`](https://github.com/microsoft/torchscale/commit/7f0bf80a7e41e6fe2d3bf1fda570fbbf8ecc13a4) 和 [normalization 修复 commit `fdd8838`](https://github.com/microsoft/torchscale/commit/fdd8838a756c7c435d7f8a1e4303e150dfac7442) 说明实现需固定版本核验。官方 TorchScale 仓库页面截至 2026-11-23 显示已于 2026-10-26 归档；这只说明该仓库当前只读，不据此推断 RetNet 研究或其他实现停止。
- 未运行代码、benchmark 或模型；代码只作为指定 revision 的实现来源核读。

## 结论边界

- 支持：RetNet 论文给出的带固定相对位置衰减的 retention 可在并行、递归和 chunkwise recurrent 视角下计算；状态按模型宽度固定，而非按历史 token 数累积；multi-scale decay、GroupNorm 与门控属于其 block 设计。
- 不支持：RetNet 对所有任务/硬件/现代 Transformer 都更快或更强；有限矩阵状态可以精确保留任意长度中每个 token 的独立可寻址副本；论文公式等于每一版官方实现的归一化和数值路径。
- 章节分工：RWKV 状态式由第三十七章承接；线性注意力基础与 Performer 由第二十四、二十五章承接；全面的替代架构对照由第四十二章承接；服务状态生命周期与部署回归由推理和系统书册承接。
