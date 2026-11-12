# 第十五册第六章精修审计：Diffusion 基础

审阅日期：2026-11-11

状态：本章精修、章级扫描和正文门禁通过；继续第七章。按用户要求不更新 PDF。

目标章节：[第六章《Diffusion 基础》](../../book-15-multimodal-generative-models/chapters/06-diffusion基础.md)

## scope 与章节结构

- 全文复读 §§0–16，保留整篇为一个连续核心讲解单元；不拆章、不改变十四章骨架。
- 主问题是如何把生成任务表示为前向随机路径、可监督的局部去噪任务和有顺序的反向状态更新，并判断预测目标、噪声 schedule、denoiser 和 sampler 是否匹配。
- 教学链从生成模型与扩散直觉开始，推导 DDPM 闭式加噪和 SNR，连接 epsilon、`x_0`、`v` 与 score 参数化，再进入反向均值/方差、连续 SDE/ODE、sampler、U-Net/DiT、CFG、latent diffusion、少步生成和 Flow Matching；以训练/推理数据流、失败诊断、可运行算术审计及练习收束。
- 与第七章《Stable Diffusion 与 DALL·E》分工：本章讲基础机制、代表性训练/采样方法和适用边界；产品级架构、完整生成 pipeline、图像编辑和具体系统组合留给下一章。全规模训练 recipe 交第五册，端到端项目交第三册。

## chapter_scan：范围、证据与去向

- **日期与范围：** 2026-11-11。定向核验扩散训练目标、反向方差、timestep weighting、终端 SNR、连续设计空间、采样器与 Flow Matching/Rectified Flow 边界，并复查 CFG 参数记法及固定版本 scheduler 接口。未穷尽扩散模型、checkpoint、采样器和产品 pipeline。
- **研究问题：** 简化 epsilon MSE 遗漏什么训练信号；不同预测目标下如何解释 Min-SNR；训练终点与纯噪声生成起点何时不匹配；训练方法、denoiser 结构和采样 solver 分别改变什么；Flow Matching 与 Rectified Flow 如何避免被当作同义词。

| 候选主题 | 来源与证据范围 | 去向与决定 |
| --- | --- | --- |
| 学习反向方差 | Nichol & Dhariwal, Improved DDPM v1；核对 arXiv HTML 摘要、§2.2、§3.1。来源支持方差输出参数化、`L_simple+λL_vlb` 和论文实验内少一个数量级前向调用；未本地复现。 | 纳入反向分布小节，解释方差头和混合目标如何改变模型训练；强调输出未被约束在 `[0,1]`、零方差端点需要实现约定，少步结果绑定论文模型和协议。 |
| 目标相关的 Min-SNR-γ | Hang et al., v3；核对 HTML §§3.1、3.4、4.2。来源支持 `x_0`、epsilon、`v` 各自对应的权重形式及指定实验的收敛报告。 | 纳入时间步加权小节，给出 `γ=5、SNR=20` 的三种手算值；不把收敛倍数或单一权重推广到其他参数化、schedule 或数据。 |
| 非零终端 SNR | Lin et al., v4；核对摘要、§2、§3.1 和 Stable Diffusion 2.1 特定配置。 | 纳入边界例子，给出论文报告的 `sqrt(bar_alpha_T)=0.068265`；限定到被检查配置，并说明零信号系数下 epsilon 到 `x_0` 的直接反解不适用。 |
| EDM 设计空间 | Karras et al., v2 与 NeurIPS 2022 正式论文 PDF；核对摘要、§§1–2、5。 | 在 sampler/solver 小节用作设计维度拆分：噪声/时间参数化、采样与离散、训练分布/损失和网络预条件；明确不保证任意交叉组合兼容。 |
| Flow Matching 与 Rectified Flow | Lipman et al., Flow Matching v2（§§2–4）；Liu et al., Rectified Flow v1（摘要、§§2.1–2.2）。 | 修正噪声到数据的直线路径方向并加入单样本速度误差；说明前者是所选概率路径上的向量场回归框架，后者是具体 ODE 运输/轨迹整流路线。 |
| CFG 参数记法与 scheduler 文档 | Ho & Salimans CFG v1；Diffusers v0.35.1 scheduler 文档和经典 Stable Diffusion pipeline 固定源码。 | 明确论文 `w` 与差分系数 `s=1+w` 的对应关系；接口描述限定于固定版本，negative prompt 分支不称为字面 null condition。 |

**扫描结论：** 上述主题补足训练目标与推理更新的关键区别，不要求增章或改变章节职责。完整 Stable Diffusion/DALL·E 路线留第七章；未把论文、接口、教学算例或局部 checkpoint 观察写成通用性能保证。

## 修改与教学链

- 在来源边界表补充 Improved DDPM、EDM、Min-SNR/terminal-SNR、Flow Matching/Rectified Flow 和 Diffusers v0.35.1 的来源类型与不可外推边界。
- 新增训练与生成对照表，比较各自已知输入、当前状态的构造方式及 denoiser 输出用途；说明训练随机抽取时间步不等于生成时可以跳过有序状态更新。
- SNR 公式改为显式含 `σ_data²` 的坐标方差比，并说明单位方差假设；给出 `bar_alpha=0.81` 时 `SNR≈4.263` 的数值例。
- 补充论文特定的 Stable Diffusion 2.1 非零终端系数及 `x_T` 近似；说明终端边界、预测目标和采样起点需要配套。
- 为 Min-SNR-γ 列出 `x_0`、epsilon、`v` 目标各自的权重，说明 SNR=0 时 epsilon 权重取连续极限；手算 `γ=5、SNR=20` 得 `5、0.25、0.2381`。
- 补充 Improved DDPM 的对数方差头、混合训练目标、论文 `λ=0.001` 设置和末步零后验方差约定；区分模型方差学习与换数值 solver。
- 解释 EDM 的可拆分设计选择和边界；补齐 DDIM、DPM-Solver、DiT、CFG、LDM、Progressive Distillation、Consistency Models 等关键段落的近邻来源。
- CFG 中统一用 `s` 表示差分系数，并对照原论文 `w`；补充 Diffusers v0.35.1 经典 pipeline 的 `guidance_scale=1` 基线及 negative-prompt 分支边界。
- 将 Flow Matching 教学路径统一为噪声到数据，给出 `x_noise=-1、x_data=2、t=0.25` 的 `x_t=-0.25`、速度 `3` 和模型预测 `2.5` 时平方误差 `0.25`；另述 Rectified Flow 的方法边界。
- 公式函数名改为 GitHub MathJax 更稳妥的 `\mathrm` 写法；补充 Min-SNR、学习方差、终端 SNR 和 Flow Matching 边界练习。

## length_status

- 全篇一个核心讲解单元；解释性中文实计 **9,215 字**，高于 1,800 字下限。
- 统计纳入连贯正文、解释性列表、练习和回顾文字；排除标题、两张表格、围栏代码/数学、代码状态、行内代码与公式、链接目标及 §15 参考资料。篇幅达标不替代机制链与证据边界的人工审阅。

## example_status

- 通过。可复算例子均写明教学输入和中间步骤，不作为 checkpoint 或产品实测：
  - 数据坐标方差取 1、`bar_alpha_t=0.81`，得到 `SNR=0.81/(1-0.81)=0.81/0.19≈4.263`。
  - Stable Diffusion 2.1 论文示例中 `sqrt(bar_alpha_T)=0.068265`，`sqrt(1-bar_alpha_T)≈0.997667`，故末步状态写作 `x_T≈0.068265x_0+0.997667ε`；结果只绑定论文所检查配置。
  - Min-SNR-γ 取 `γ=5、SNR=20`：`x_0` 权重 `5`，epsilon 权重 `0.25`，`v` 权重 `5/21≈0.2381`。
  - Flow Matching 端点 `-1→2`、`t=0.25` 得 `x_t=-0.25`、目标速度 `3`；预测速度 `2.5` 的平方误差为 `0.25`。
  - 章节既有 DDPM 标量/向量加噪、反向均值、CFG 与 latent 元素数例仍保留，并与 §13 四维算术 demo 的公式和输出对应。

## code_status

- §13 唯一独立 Python 标准库代码块状态为“已验证（2026-11-12；Python 3.12.9；仅 Python 标准库；CPU）”。整册复读发现向量允许 bool/非数值元素、guidance scale 和反向公式系数允许 bool、latent 维度允许浮点数或 bool；现增加有限实数向量、schedule 系数与 guidance scale 检查，并要求 latent 高宽、通道和压缩因子为正整数，拒绝 bool 充当数值或尺寸。
- 从 Markdown 提取完整代码块执行，stdout 与正文预期输出逐字节一致。十项主检查和十三项边界检查全部为 True；新增边界覆盖 bool/字符串向量元素、bool guidance scale、非整数尺寸、bool 压缩因子和 bool `previous_alpha_bar`。正常算术结果不变。
- 本地教学输入为四维向量、五个 beta 和 `512×512` latent 示例；验证只核对公式算术和输入契约，不调用模型、不生成图片、不训练，不验证真实 denoiser、checkpoint 质量、GPU、吞吐、显存或生产延迟。

## formula_status、link_status 与 visual_status

- **formula_status：** 51 个 math 围栏，花括号平衡；围栏外 36 个行内 `$` 标记（18 对）；124 个波浪围栏标记配成 62 对（51 math、10 text、1 Python）；未发现 `\operatorname`。完成源码结构检查，未声称 GitHub/浏览器 MathJax 实际渲染。
- **link_status：** 36 个 Markdown 链接，均为外链；无本地链接。本轮没有对全部外链作 HTTP crawl；固定论文版本、官方文档版本及阅读范围已列入正文和研究笔记。
- **visual_status：** 增加训练/生成对照表；保留来源/证据边界表和带变量的训练/推理数据流。两张表对应输入状态差异和证据范围；没有 Mermaid，未额外绘制扩散时间线，公式、数例和数据流足以表达本章重点关系。
- **结构检查：** 两张 Markdown 表格列数一致；围栏配对；正文无 `\operatorname`；`git diff --check` 通过。

## sync_status 与 open_risks

- **sync_status：** `PAPERS.md` 增补/固定 Improved DDPM v1、EDM v2、Flow Matching v2、Rectified Flow v1、CFG v1、DDIM v4、DPM-Solver v3、Progressive Distillation v2 及 schedule 分析 v4，并将既有 Min-SNR 入口固定到 v3；`EXERCISES.md` 增补 Min-SNR、学习方差、终端 SNR 练习并精确 Flow Matching/Rectified Flow 端点命名；`GLOSSARY_EN_ZH.md` 新增 Terminal SNR、Learned Reverse Variance、EDM；`INTERVIEW_BANK.md` 更新 scheduler 问答并新增学习方差/权重/solver 对照题。论文机制、固定版本、阅读范围和非复现边界写入既有 diffusion 及 Flow Matching 研究记录；没有把章节审阅流水写入 `research/`。简介与目录章节职责保持准确，无需修改。
- **open_risks：** 未运行扩散 checkpoint、未训练或复现论文图像质量/采样加速结论；Min-SNR 的收益、EDM 组合和 terminal-SNR 改动需在目标模型/schedule 下独立评估；没有目标平台数学渲染检查或外链全量 HTTP 检查。具体结果仍绑定论文设置或固定文档版本。
- **handoff：** 下一步进入第七章《Stable Diffusion 与 DALL·E》，沿基础机制检查 VAE/text encoder/denoiser/scheduler/CFG/control 的完整系统数据流和论文路线；避免重复本章公式推导，并继续不更新 PDF。

## 整册连续复读追加记录（2026-11-12）

- 已沿第十五册主路径读完本章 §§0–16，并对照第二章图文对齐、第五章训练数据契约与第七章完整文生图系统职责；未发现需要调整章节边界或顺序的跨章问题。
- 连续复读中发现 §13 输入类型边界未充分拒绝 bool、非数值向量项及浮点 latent 尺寸，已按上述 code_status 修复并复验。练习、正文结论和配套主题无须调整；原证据范围和研究结论不变。
- **下一步：** 进入第七章《Stable Diffusion 与 DALL·E》做整册连续复读；读完第十四章后执行书册最终门禁。
