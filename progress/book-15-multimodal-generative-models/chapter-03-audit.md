# 第十五册第三章精修审计：Vision Encoder

审阅日期：2026-11-11

状态：本章精修、章级扫描与正文门禁通过；2026-11-12 整册复读时收紧尺寸/索引类型合同并复验代码；继续第十五册第四章整册复读。按用户要求不更新 PDF。

目标章节：[第三章《Vision Encoder》](../../book-15-multimodal-generative-models/chapters/03-vision-encoder.md)

## 范围与结构

- 全文复读 §§0–15，保留整篇作为一个连续的核心讲解单元；没有改变十四章结构，也没有新增独立讲次。
- 主问题是：图像怎样从像素变成全局、局部或多层视觉表示；CNN、ViT、预训练目标、分辨率和预处理如何共同决定这些表示的空间契约、能力边界和计算成本；视觉输出怎样接入 VLM。
- 与第四章的连接器架构、第十四章的请求级多模态 token 预算保持分工。本章解释视觉表示、输入接口和成本量级，不展开完整 VLM 训练、生产 serving 或模型级预算规则。

## 章级扫描（chapter_scan）

- **日期与范围：** 2026-11-11；围绕视觉塔的 dense feature、分辨率迁移、processor 输入契约和多层视觉特征定向检查。资料类型包括原始论文、固定 revision 官方代码及官方处理器文档；不穷尽所有视觉塔和 OCR 模型。
- **研究问题：** 长时间自监督训练是否会让全局指标和 patch 局部一致性产生取舍；如何保持稠密 patch 结构；高分辨率与多层特征如何改变输入/下游接口；公开代码和 processor 文档可以支持到哪个版本边界。

| 候选主题 | 来源证据 | 去向与决定 |
| --- | --- | --- |
| 全局指标与 patch 级 dense feature 的取舍 | Siméoni et al., [DINOv3 arXiv v1](https://arxiv.org/abs/2508.10104v1)，核对摘要、引言、§§4.1–4.3、§§5.1、5.3。作者报告其长时程训练中全局指标提高而 patch 级相似结构/密集任务表现可能退化，并提出 Gram anchoring、高分辨率适配和文本对齐阶段。 | 纳入 §6.4，解释论文特定机制与阶段；明确不推广为所有视觉塔的通用正则项，也不声称本地复现。来源细节和边界写入[研究记录](../../research/multimodal-systems-and-evaluation-2026.md)。 |
| 多层 ViT 特征注入 | Qwen Team, [Qwen3-VL Technical Report v2](https://arxiv.org/abs/2511.21631v2)，报告描述 DeepStack 使用多层视觉特征加强视觉—语言对齐。 | 在 §4.4 作为架构实例，说明可能在最终视觉层之前接入中间层；不推成所有 VLM 的必要设计或独立效果证据。论文入口已在 `PAPERS.md`，术语已有 `GLOSSARY_EN_ZH.md` 条目。 |
| 固定视觉塔的实现和输入预处理 | OpenAI CLIP [固定 revision `model.py`](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py) 与 [`clip.py`](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py)；Transformers [CLIPImageProcessor 官方文档](https://huggingface.co/docs/transformers/model_doc/clip)，访问日期 2026-11-11。 | 在 §§0.2、3.5、7.4 贴近实现结论引用；固定源码事实只限该 revision，动态 API 文档和默认值绑定库/权重版本，不推广至所有模型。 |
| 视觉主线的基础机制 | ResNet、ViT、CLIP、SigLIP、MAE 与 Swin 原论文分别支持残差 CNN、patch Transformer、图文对齐、sigmoid 配对目标、掩码重建和窗口层级 attention 的机制说明。 | 保留为已有主线的机制来源；不把论文实验结果横向拼成单因素排名，也不据此推断任意产品模型的 OCR 或 VLM 质量。 |

**扫描结论：** 新证据补足密集特征退化/保持、多层特征注入及 processor 版本边界，无须增章或重排；章节间职责不变。

## 修改与教学链

- 开篇明确视觉塔输入/输出形式与本章问题；补足 CNN、ViT、patch、CLS、SigLIP、MAE、LLM hidden size、OCR、pooling 等首现释义，并统一图像块、全局与局部表示的术语。
- §2.1 将不分组 Conv2d 的索引、步幅、边界填充、膨胀率、边界零值和偏置写清；补 224 输入、3×3 核、padding=1、stride=2、dilation=1 的逐步尺寸代入，得到 112×112 输出。按常见深度学习框架互相关约定限定公式。
- §§3.1–3.5 补全 225/patch14 的补齐与有效卷积两种网格计算、行优先空间坐标、CLS/位置表长度，以及 RGB/patch14/hidden1024 patch embedding 参数代入。
- §§4–5 解释 CLS、平均/注意力池化、patch 和多层特征的责任；补 Qwen3-VL DeepStack 的有限实例；明确 attention score 的计数口径为每样本、每 head 的 $L^2$，并区分 batch 与 head 汇总的 $BhL^2$ 逻辑元素数。
- §§5–7 补分辨率和 token 的面积/平方成本、图像 processor 版本合同及归一化边界。§6.4 纳入 DINOv3 Gram anchoring 与高分辨率/文本对齐路线，限于论文设置。
- §§8–11 保留 OCR、图表、空间关系、局部验证、视觉塔迁移和 VLM 接口主线；清楚说明全局向量不能代替局部证据，projector 不能恢复前序已丢弃的信息。
- §12 将教程代码结果与前文算例连接，说明 `detail_need` 和阈值 0.7 是合成分支输入，不是校准分数或分辨率推荐。

## 篇幅、例子与公式

- **length_status：** 全篇一个核心讲解单元，解释性中文字符实计 **8,492**，高于 1,800 下限。计数排除标题、表格、代码/文本图示围栏、公式、行内代码与公式、链接目标、状态行及编号参考文献；保留章节解释、案例、练习和回顾文字。计数脚本只用于统计，不替代人工判断。
- **example_status：** 通过。正文覆盖 224 输入卷积输出尺寸 112×112；225×225、patch14 时补齐网格 17×17=289、无 padding 网格 16×16=256；RGB/patch14/hidden1024 的 embedding 权重 602,112、含 bias 共 603,136；224/336/448 的 patch 数 256/576/1,024，带 CLS 长度 257/577/1,025；对应单样本单 head score cell 66,049/332,929/1,050,625，448 相对 224 约 15.91 倍；projector 1024→4096 含 bias 共 4,198,400 参数；第 25 个含 CLS 序号对应 patch 网格第 1 行第 0 列，中心坐标为 $(x,y)=(7,21)$ 像素。所有数字是显式代入或教学代码确定性输出，不是模型识别质量或硬件实测。
- **formula_status：** 37 个 display-math 围栏、99 对行内 `$` 标记；花括号脚本检查无未配对项。无 `\operatorname`、`\text` 下划线变量或 `x_{<t}`。MathJax/GitHub 页面未实际渲染检查。
- **visual_status：** 数据流与文本流程图、token 网格和成本公式、方案对照表及代码结果表承担结构表达；未增装饰图或 Mermaid。现有表达足以解释章节关系，本轮无必须绘图项。未声称 GitHub 页面渲染通过。

## 代码、链接与配套同步

- **code_status：** §12 唯一独立运行的 Python 块带相邻标记“已验证（2026-11-12；Python 3.12.9；仅 Python 标准库）”。复读发现原实现允许浮点数和布尔值进入图像尺寸、patch、hidden size 与 token 索引，并用浮点除法计算补齐网格；现改为正整数合同、布尔开关校验和纯整数 ceiling 计算，未改变正常输入的结果。
  - 运行环境/方式：Python 3.12.9 虚拟环境解释器执行本章 §12 fenced block；输入为代码内固定的 patch=14、RGB channels=3、vision hidden=1024、LLM hidden=4096 和四组分辨率。
  - 预期/实际：224、336、448、672×448 网格分别为 16×16、24×24、32×32、48×32；CLS 后序列长度为 257、577、1025、1537；score-cell 数为 66,049、332,929、1,050,625、2,362,369；相对 224 倍数为 1、5.04、15.91、35.77。225×225、patch14 补齐/有效网格分别为 17×17/16×16；patch 向量维 588；权重形状 `(1024,3,14,14)`；位置表 `(577,1024)`；投影 token `(2,576,4096)`；两个参数量为 603,136 和 4,198,400；坐标和所有逻辑断言通过，决策输出 `continue_to_encoder_benchmark`。
  - 边界：原有零/负维度、输入小于 patch 的有效卷积网格、零 token、非法坐标等边界继续拒绝；本次另验证 25 个非法输入，覆盖浮点/布尔维度、非布尔开关、负/CLS/越界坐标和浮点网格尺寸；$10^{500}$ 尺寸的补齐网格及 $10^{200}$ token 的 attention cell 数均以整数精确计算。实际 stdout、参数量和所有逻辑断言与预期一致。本演示不运行 Transformer、不读取真实图像，也不验证 OCR。
- **link_status：** 全章 16 个 Markdown 链接均已解析；相对本地链接缺失数为 0。未做外链 HTTP 全量巡检。
- **sync_status：** `PAPERS.md` 增加 DINOv3 视觉主线入口；`GLOSSARY_EN_ZH.md` 增加 DINOv3 与 Gram Anchoring 的范围定义；`INTERVIEW_BANK.md` 更新 vision encoder 答题框架并连接本章与 DINOv3 论文；`research/multimodal-systems-and-evaluation-2026.md` 只记录 DINOv3、固定 CLIP 源码和 processor 文档的新事实、来源及边界。简介/目录未改，因为章节职责和顺序未变；Qwen3-VL 的论文及 DeepStack 词条原已在配套索引。
- **检查：** 代码围栏共 48 对；math 围栏 37 个；行内美元标记 198 个/99 对；公式花括号配平；唯一 Python 块有验证状态；`git diff --check` 通过。

## 遗留边界与交接

- 未验证任意真实图像、OCR、图表问答、grounding、VLM 对话、GPU 吞吐、峰值显存、动态分辨率或 processor 跨版本兼容。
- 未检查 GitHub 渲染和外链全量 HTTP 状态；正文中的 CLIP processor 动态文档已标访问日期，后续涉及默认值时按具体 revision 复查。
- 对配套索引作的扩展本地链接检查发现 `PAPERS.md` 与 `INTERVIEW_BANK.md` 另有指向当前不存在的旧研究笔记/代码路径的引用；本章新增的 DINOv3/章节链接均有效，这些既有条目不属于本章改动范围，留待对应索引维护时逐项处理。
- 下一步：继续第十五册整册复读第四章《VLM 架构》，以本章视觉输出契约为前置，核查 projector、cross-attention、Q-Former、resampler 等接入路线；同时对照第十章统一模型和第十四章预算章，若发现章节关系变化则回到架构审计。
