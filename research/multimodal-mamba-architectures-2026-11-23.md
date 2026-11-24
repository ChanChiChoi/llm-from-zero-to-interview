# 多模态模型中的 Mamba 替代与混合架构

## 研究问题与范围

- 日期：2026-11-23。
- 任务：为第二十一册第 48 章寻找多模态视觉语言与长视频模型中 Mamba/SSM 的具体架构位置、替代与混合方式、任务表现证据和讲解材料；同时核实与第十五册一般 VLM/统一多模态内容、第 43 章通用 Attention–SSM 混合、第 44 章通用模块职责及第 50 章架构演进判断的分工。
- 检索词：`Mamba multimodal architecture vision language model`、`VL-Mamba Cobra connector`、`Mamba visual grounding controlled comparison`、`query-based cross-modal projector Mamba`、`Mamba multimodal fusion alignment`、`hybrid Mamba Transformer long video VLM`、`Mamba VLM tutorial lecture notes`；检索并阅读英文原始论文、会议出版页、项目说明及作者/研究者报告。
- 范围判断：候选主题归为视觉语言语言主干替换、视觉序列化/连接器、跨模态对齐与扫描融合、长视频分流和混合。多模态生成模型有不同去噪目标，本章不把它们与自回归理解模型混为一类。

## 发现与证据

### 1. 语言主干替换不自动解决视觉连接与跨模态检索

- **Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference**，Zhao et al.，AAAI 2025；[arXiv 页面](https://arxiv.org/abs/2403.14520)，[AAAI 正式论文 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/33131/35286)。原文 §§1、3：视觉编码器拼接 DINOv2 与 SigLIP 特征，projector 对齐视觉和语言宽度，Mamba 作为语言主干；比较了多种模态融合方式。支持“将语言主干改为 Mamba、同时保留视觉塔和连接器”这一具体路线，不支持所有多模态模型都有线性端到端成本或普遍质量/速度优势。模型、数据和基线必须按论文设置读取。
- **VL-Mamba: Exploring State Space Models for Multimodal Learning**，Qiao et al.，NeurIPS Efficient Natural Language and Speech Processing Workshop 2024；[PMLR 出版页](https://proceedings.mlr.press/v262/qiao24a.html)，[论文 PDF](https://raw.githubusercontent.com/mlresearch/v262/main/assets/qiao24a/qiao24a.pdf)。原文摘要及模型图：预训练 Mamba 语言模型加 MultiModal Connector，其中包括 Vision Selective Scan (VSS)；其视觉编码器和语言主干与 connector 是不同结构位置。属于 workshop 论文，性能主张限于作者的 benchmark、模型及实验配置。
- **Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling**，Pantazopoulos et al.，EMNLP 2024；[ACL Anthology 页面](https://aclanthology.org/2024.emnlp-main.793/)，[论文 PDF](https://aclanthology.org/2024.emnlp-main.793.pdf)。原文摘要、§3—5：在最多 3B 参数的受控模型中，两类模型使用相同数据及顺序和相同训练超参数；Mamba-VL 在 captioning、VQA 和 reading comprehension 优势明显或较强，Transformer-based Pythia-VL 在 grounding 和 in-context multimodal retrieval 上更强，差距随规模扩大。属于特定基模型、数据和任务，不证明所有 SSM 或所有 VLM 的普遍表现。

### 2. 视觉连接器必须处理二维网格到一维因果序列的映射

- **Query-based Cross-Modal Projector Bolstering Mamba Multimodal LLM**，Eom et al.，Findings of EMNLP 2024；[ACL Anthology 页面](https://aclanthology.org/2024.findings-emnlp.827/)，[论文 PDF](https://aclanthology.org/2024.findings-emnlp.827.pdf)。原文摘要、§3.2：Querying Mamba (Q-Mamba) 由可学习 query、Mamba 序列混合、跨模态 attention 与 FFN 构成；query 读取视觉编码特征，把二维视觉信息投影为一维因果 token，可学习压缩长度并避免手工指定原始图像的扫描顺序。其结果支持该论文中的 connector 设计，不能外推为 query connector 对所有分辨率/任务更好。
- **Vision Mamba | Multimodal Weekly 35**，TwelveLabs 于 2024-03-21 发布的研究者专题报告：[视频](https://www.youtube.com/watch?v=qPJJS-9fbRc)。报告由 Vision Mamba 作者/研究者讲解 SSM 与视觉表征、扫描方案、效率实验和消融。用于辅助理解视觉序列化和作者的解释路线，不替代论文证据，也不是 VLM 任务结果。

### 3. 顺序扫描与跨模态对齐是两个不同设计问题

- **AlignMamba: Enhancing Multimodal Mamba with Local and Global Cross-modal Alignment**，Li et al.，CVPR 2025；[CVF 正式论文 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_AlignMamba_Enhancing_Multimodal_Mamba_with_Local_and_Global_Cross-modal_Alignment_CVPR_2025_paper.pdf)。原文摘要、§§2—4：各模态先通过专用 encoder；局部对齐以 relaxed Optimal Transport 建立 token 对应，全局对齐用 Maximum Mean Discrepancy 约束分布，之后再拼接并输入 Mamba backbone。论文覆盖 complete/incomplete multimodal representation fusion，并报告 CMU-MOSI 等任务。它是音频/视频/语言情感等表示融合案例，不是通用自回归 VLM；效率数字只比较其特定 cross-modal fusion 模块、数据集与硬件设置。

### 4. 长视频中可按模态分配不同的信息路径

- **VAMBA: Understanding Hour-Long Videos with Hybrid Mamba-Transformers**，Ren et al.，ICCV 2025；[ICCV 正式论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Ren_Vamba_Understanding_Hour-Long_Videos_with_Hybrid_Mamba-Transformers_ICCV_2025_paper.html)，[arXiv 页面](https://arxiv.org/abs/2503.11579)，[ICCV supplementary](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Ren_Vamba_Understanding_Hour-Long_ICCV_2025_supplemental.pdf)。原文 §§3—4：视频 token 由 Mamba/Mamba-2 更新，文本 token 保留 self-attention 并通过 cross-attention 读取全部视频 token；作者报告在其视频帧数、输入分辨率、模型和单卡/训练配置下可处理 1024 帧以上，并报告对照条件下的显存、FLOPs 与速度差异。不能将这些数值当成所有硬件、任务或实现的收益。研究者 Weiming Ren 在 Toronto Machine Learning Society 的[作者报告](https://www.youtube.com/watch?v=nTEnQnW0zZQ)有助于理解该架构设计；数值和结构仍以正式论文为准。
- **TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding**，Xu et al.，CVPR 2026；[CVF 正式论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_TimeViper_A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_CVPR_2026_paper.pdf)，[项目页](https://xuboshen.github.io/TimeViper/)。原文摘要、§§3—4：ViT 与 projector/ToMe 生成视频 token，混合语言主干包含 Mamba-2 和少量 self-attention；TransV 通过 gated cross-attention 把视觉信息转入 instruction token 并在主干内压缩视觉 token。论文提出 vision-to-text 信息聚合现象，并在 VideoQA、grounding 和 captioning 等任务做不同控制。摘要中的帧数、吞吐与性能只绑定报告中的 backbone、token、batch、输入/输出长度及 benchmark；不能把压缩、SSM 和硬件实现收益拆为互相独立的普遍因果结论。
- **Multimodal Mamba: Decoder-only Multimodal State Space Model via Quadratic to Linear Distillation**，Liao et al.，arXiv 2025；[预印本](https://arxiv.org/abs/2502.13145)。摘要说明从 Transformer VLM 逐步蒸馏到线性 Mamba 与 hybrid 变体，结果绑定 HoVLE teacher 和论文报告的 103K-token设置。作为架构转换路线的预印本线索，不据此称蒸馏转换已普遍解决多模态能力保持。

## 教学资料检索

- 首轮同时检索了论文/官方出版资料与解释型文章、讲义和作者报告。未发现对本章全部架构家族进行独立、完整推导的高质量长文；因此使用 Vision Mamba 研究者专题报告和 VAMBA 作者报告辅助观察空间扫描及长视频分路的解释方式。它们是讲解来源，不充当结果证据。
- 第十五册第 4、10、13、14 章已经讲解一般 VLM 连接器、统一模型、编码形式与 token 预算；本章只回顾理解 SSM 设计位置所需的部分，不重复其通用机制。

## 决定及边界

- 主线按“主干替换—视觉连接器—跨模态对齐—长视频混合—按任务判读证据”组织。
- 不使用纯图像 Mamba backbone 当作完整多模态模型证据；不把特定情感融合、3D-VL 或多模态扩散模型的结果直接迁移到 VLM；不根据算法复杂度单独断言墙钟速度。
- 主要未决项：不同模型的参数量、预训练数据/初始化、图像/帧 token 化、任务集合、硬件、kernel 和评估脚手架仍不一致，无法组成统一排名。后续发现新模型时按其中的结构位置和证据质量更新候选，不按发布日期堆目录。
