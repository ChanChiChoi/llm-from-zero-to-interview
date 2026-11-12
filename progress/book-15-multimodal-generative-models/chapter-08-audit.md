# 第十五册第八章精修审计：视频生成与 World Model

审阅日期：2026-11-12

状态：本章全文精修、定向扫描与正文门禁通过；第十五册其余章节仍待精修。按用户要求不更新 PDF。

目标章节：[第八章《视频生成与 World Model》](../../book-15-multimodal-generative-models/chapters/08-视频生成与world-model.md)

## scope 与章节职责

- 全文复读 §§8.0–8.23；保留一个贯穿“时空表示—视频生成—状态预测—交互 rollout—任务评估”的核心讲解单元和既有十四章结构。
- 本章承接第六章 diffusion 与第七章文生图系统，回答为什么视频多出时间状态、模型怎样生成时空内容、什么条件下能进一步预测动作后果，以及如何区分视觉相似、任务物理和控制证据。
- 本章讲生成机制、latent/patch 成本、时序关系、world model 状态转移与评估边界；第十一章承担多模态评估与安全体系，第七册承担通用评估科学，真实机器人/VLA 控制和生产服务交由相邻专题。
- 章级候选可以由现有职责容纳，不增章、不拆分或迁移章节；简介与目录承诺仍准确，无需修改。

## chapter_scan：范围、证据与候选去向

- **日期与范围：** 2026-11-12。针对视频时空生成、交互式 world model、视频物理测量、条件执行和分布/任务评估定向扫描；复用既有原始论文和研究记录，不追求穷尽视频模型或服务 API。
- **检索问题：** 图像扩散如何延展到时空 latent；视频质量、物理测量和交互 rollout 分别需要什么输入；仅有视觉逼真度是否足以称为 world model；新 benchmark 是扩展本章机制边界还是应留给第十一章完整评估主讲；框架 forward 次数与条件分支计算量怎样分开记账。
- **资料类型：** 视频生成和 world model 原始论文、benchmark 预印本及官方研究说明。采用事实均有现存来源记录；benchmark 的样本数、维度与实验结果绑定对应版本、数据和评分器。

| 候选主题 | 来源证据 | 去向与决定 |
| --- | --- | --- |
| 视频 diffusion、时空 U-Net/Transformer、级联和离散 token 路线 | Video Diffusion Models、Make-A-Video、Imagen Video、Lumiere、VideoPoet、CogVideoX 等论文及既有[视频生成研究记录](../../research/video-generation-world-modeling-and-sora-lifecycle-2026.md)。 | 保留机制和路线对照，不用单篇论文配置推成通用规格，也不新增章节。 |
| 分布级视频生成距离与逐样本任务指标的区别 | FVD v2（[arXiv 版本页](https://arxiv.org/abs/1812.01717v2)）定义特征分布距离；本章用一维高斯输入核对公式。 | 保留 FVD 作为总体分布信号，另配任务/条件遵循指标；不把 toy 数值当 benchmark 结果。 |
| 交互式视频 world model | [WBench v1](https://arxiv.org/abs/2605.25874v1)：289 案例、1,058 轮交互，按视频质量、场景/交互遵循、一致性和物理符合度评估；资料见既有[来源记录](../../research/video-generation-world-modeling-and-sora-lifecycle-2026.md)。 | 纳入 §8.17 的任务选择表，用于说明多轮动作/视角交互如何评估；明确 benchmark 不证明真实环境闭环控制。 |
| 视频忠实度自动评估 | [VBench-2.0 v2](https://arxiv.org/abs/2503.21755v2)：五类、18 项能力，并报告通用 VLM 的人体、计数和 3D 评估局限；详见[视频评估研究记录](../../research/video-generation-evaluation-2026.md)。 | 扩展 §8.17.3 的维度化思路，并把自动分数和单视频可靠性边界写清；完整评估分类留给第十一章。 |
| 对象级物理测量 | [PVBench v1](https://arxiv.org/abs/2610.26711v1)：600 个提示、七领域、46 维度；物理量可涉及姿态、尺度、运动等。 | 纳入物理评测任务例，明确结果依赖可观测性、抽取器和校准，不将不可测状态推为正确。 |
| 受控物理现象与观测边界 | [World Models’ Last Exam in Physics v1](https://arxiv.org/abs/2610.08791v1)：40 个任务、九类可观测现象，区分可观测性筛查和任务专属物理测量；研究记录已有摘要、方法和边界。 | 与 PVBench 并列作为受控物理测试示例，结论严格限于该预印本任务和测量流程。 |
| 关键帧条件执行 | [KeyFrame-Compass v1](https://arxiv.org/abs/2607.14202v1)：386 个样例，关键帧执行拆为六项。 | 只作为 storyboard/关键帧输入时的专门协议；明确不代表普通文生视频能力。 |
| 控制相关状态预测与规划 | World Models、DreamerV3、V-JEPA 2，以及既有交互路线材料。 | 以状态、动作、belief、rollout 和 MPC 主线解释；视频像素生成与真实动作执行仍分开验证。 |

**扫描结论：** 近期候选要求在本章增加“按任务协议选择评估”的桥接，不改变十四章骨架；WBench/物理测量与关键帧评测的完整指标体系交第十一章延伸。没有把机器人控制、真实环境安全或服务上线门禁扩进本册主线。

证据范围复用[视频生成评估研究记录](../../research/video-generation-evaluation-2026.md)和[视频生成、交互式世界建模研究记录](../../research/video-generation-world-modeling-and-sora-lifecycle-2026.md)。这些记录只保存来源事实；本审计保存候选去向、正文决策、验证状态和未决边界，因此本轮没有新增 research 文件。

## 教学链、篇幅与数值例子

- **length_status：** 全篇一个连续核心单元，按解释性汉字实计 **12,769 字**，高于 1,800 字下限。统计排除 §§8.22–8.23、标题、代码/数学围栏、表格、状态行、行内代码/公式和链接目标。人工复读确认各节沿“表示—生成—条件—状态—评估”推进，篇幅由机制、边界、实例和评测解释构成。
- **example_status：** 通过。例子中的数值均为明确标注的教学输入，不是模型评测或产品规格：
  - 视频 latent 加噪：$z_0=0.8$、$\bar\alpha_t=0.64$、$\epsilon=-0.5$，逐项代入得到 $z_t=0.34$。
  - 时间戳对齐：24 fps 的第 2 帧 $\tau=2/24\approx0.0833$ 秒；条件点在 0 与 0.1 秒分别为 0/10 像素，插值得约 8.33 像素。
  - CFG 成本：batch=1、24 步、两条件分支是 48 个样本级前向等价量；`1×24×4096×14×2=2,752,512` 是单头、单层 cross-attention score-cell 数，不等于框架函数调用数、FLOPs 或显存。
  - MPC：在 $x_{k+1}=x_k+a_k$、$x_0=0$、目标 1、动作代价系数 0.1 下，两步动作 $(0.6,0.4)$ 的总代价为 0.212，$(0.8,0.8)$ 为 0.528；选择前者后只执行第一步并重观测。
  - FVD 一维高斯例：均值 1/2、方差 4/1，均值项 1、协方差项 1，总距离 2；只验证矩阵平方根公式的标量代入。
  - 原章节还包含时空 patch、latent 压缩、长视频误差、身份/运动指标和 rollout 例；全部使用合成或声明的输入。

## code_status

- §8.20 的独立 Python 标准库 demo 于 2026-11-12 完成边界加固并重新运行，代码状态为“已验证”。新增有限实数检查，拒绝 bool 充当帧数、尺寸、patch/token 计数、guidance、中心坐标、身份特征或亮度；计数、向量长度和 shape 的正整数/有限值约定与可见性布尔值分开处理。
- 运行命令：`./.tmp/llm-book-cpu-check/bin/python -c 'from pathlib import Path; p=Path("book-15-multimodal-generative-models/chapters/08-视频生成与world-model.md"); s=p.read_text(encoding="utf-8"); code=s.split("~~~python",1)[1]; code=code[:code.find("~~~")].strip(); exec(compile(code,"chapter_08_demo","exec"))'`。
- 输入为 batch=1、16 帧、256×256、空间 patch=16、时间 patch=2、时空 latent 压缩和合成五帧指标窗口；不加载模型或视频。
- 执行结果与正文完整 stdout 逐字节一致；9 项主检查和 14 项边界检查均通过。边界覆盖零 patch、过短/超长指标窗口、零向量、错配向量、非法可见值、非有限亮度、关闭 CFG，以及 bool 计数/数值字段。原边界样例曾因窗口过短先触发别的校验；此前已改为三帧输入，确保每项实际命中目标路径。

~~~text
raw_shape=(1, 3, 16, 256, 256)
framewise_tokens=4096
spatiotemporal_grid=(8, 16, 16)
spatiotemporal_tokens=2048
token_ratio=0.5
latent_shape=(1, 4, 4, 32, 32)
latent_element_ratio=0.005208
latent_tokens=4096
cfg_branches=2
denoiser_sample_forward_equivalents=48
cross_attention_score_cells_one_head_layer=2752512
motion_second_difference=0.0133
minimum_identity_cosine=0.9987
average_brightness_change=0.015
object_visibility=1.0
metric_window_frames=5
linear_rollout=[0.1, 0.21, 0.32, 0.43, 0.54]
linear_rollout_mae=0.002
checks={'tokens_within_budget': True, 'latent_ratio_under_demo_limit': True, 'cfg_forward_equivalents_match': True, 'metric_window_within_video': True, 'identity_above_demo_threshold': True, 'brightness_change_under_demo_limit': True, 'motion_change_under_demo_limit': True, 'object_visible_in_demo': True, 'linear_rollout_error_under_demo_limit': True}
audit_consistent=True
boundary_checks={'zero_patch_rejected': True, 'too_short_metric_window_rejected': True, 'oversized_metric_window_rejected': True, 'zero_identity_rejected': True, 'mismatched_identity_rejected': True, 'invalid_visibility_rejected': True, 'nonfinite_brightness_rejected': True, 'disabled_cfg_uses_one_branch': True, 'disabled_cfg_forward_equivalents_match': True, 'boolean_count_rejected': True, 'boolean_guidance_rejected': True, 'boolean_center_rejected': True, 'boolean_identity_value_rejected': True, 'boolean_brightness_rejected': True}
all_checks_passed=True
~~~

- **验证边界：** 仅核对教学 shape、算术、简化的身份/运动/亮度窗口指标和线性 rollout；没有真实视频、扩散权重、GPU、物理测量或生成质量评估。演示阈值不是通用验收门槛。

## formula_status、link_status 与 visual_status

- **formula_status：** 49 个 display-math 块花括号平衡；围栏外 74 个未转义 `$` 标记，配成 37 对；无 `\operatorname`。54 对波浪围栏（49 math、4 text、1 Python）配平；源码检查通过，未宣称 GitHub MathJax 渲染通过。
- **link_status：** 40 个 Markdown 链接均为外链，没有本地链接目标；未做全量 HTTP crawl。
- **visual_status：** 保留时空表示、视频路线、开放环/闭环矩阵和任务评估表；用时间轴算例、rollout 表和任务—协议表承载关系。未使用 Mermaid；现有表格与逐步案例足以表达当前结构。未做 GitHub 渲染。
- **结构检查：** 5 张 Markdown 表格列数一致（3、4、4、4、4 列）；围栏、display math 花括号、行内 `$` 配对均通过；`git diff --check` 无输出。

## sync_status 与 open_risks

- **sync_status：** `PAPERS.md` 增加 World Models’ Last Exam in Physics v1，并固定 FVD v2、VBench v1、VBench-2.0 v2、PVBench v1、KeyFrame-Compass v1；`EXERCISES.md` 新增第十五册第八章的评测选型、FVD 手算、MPC 和 CFG 预算练习；`GLOSSARY_EN_ZH.md` 增加 VBench-2.0、PVBench、物理现象任务和 KeyFrame-Compass 的范围定义；`INTERVIEW_BANK.md` 更新视频生成审计问题，将样本级前向等价量与框架调用数分开，并按任务映射 benchmark；`KNOWLEDGE_GRAPH.md` 增加视频评测子链。简介、目录与章节骨架复核后保持不变。
- **open_risks：** 没有执行真实视频生成或机器人闭环评估；预印本任务和自动评估器不能替代本地业务集校准。全章公式未做 GitHub MathJax 渲染，40 个外链未全量 HTTP 检查。第十一章仍需决定多模态视频评估的全面分类、安全探针和证据门禁。
- **handoff：** 第八章正文门禁通过；第十五册架构审计现为前八章完成、剩余六章待审。**下一步：** 进入第九章《语音与音频生成》，按连续信号、ASR/TTS、离散 codec、文本/语音联合生成和实时会话的主线做全文扫描与精修。全套书修书 goal 继续 active；不更新 PDF 或 `plan.md`。

## 整册连续复读追加记录（2026-11-12）

- 已沿主路径完整复读 §§8.0–8.23，对照第六章 diffusion、第七章文生图系统和第九章音频生成的前后关系；未发现需要调整章节边界、顺序或目录入口的问题。
- 复读发现 §8.20 的输入验证允许 Python bool 混入计数与连续指标；现按字段语义拒绝 bool 和非有限/非数值项，布尔可见性仍作为显式状态处理。改动后的代码输出和正常数值不变。
- **下一步：** 进入第九章《语音与音频生成》继续整册连续复读。
