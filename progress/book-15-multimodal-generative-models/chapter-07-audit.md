# 第十五册第七章精修审计：Stable Diffusion 与 DALL·E

审阅日期：2026-11-11

状态：本章全文精修、定向扫描与正文门禁通过；第十五册其余章节仍待精修。按用户要求不更新 PDF。

目标章节：[第七章《Stable Diffusion 与 DALL·E》](../../book-15-multimodal-generative-models/chapters/07-stable-diffusion与dalle.md)

## scope 与章节职责

- 全文复读 §§0–16；保留一个连续核心讲解单元和既有十四章结构。
- 本章承接第六章扩散机制，回答如何将 VAE、文本编码、denoiser、cross-attention、CFG、scheduler、编辑/控制和安全评估组织成可追踪的文生图系统；比较 Stable Diffusion 类路线与 DALL·E 公开论文路线。
- 第六章负责扩散训练/采样原理，本章负责完整图像生成系统与产品边界；端到端项目交第三册，完整扩散训练 recipe 交第五册，多模态评估与治理方法回接第七、八册。
- 发现的新路线可由现有章节职责容纳，不新增、删除、拆分或迁移章节；简介和目录的第七章承诺仍准确，无需修改。

## chapter_scan：范围、证据与候选去向

- **日期与范围：** 2026-11-11。定向核对 latent diffusion 系统结构、SDXL/SD3 路线、img2img/inpainting、ControlNet、个性化、DALL·E 公开系统、文生图评估和 provenance。阅读原始论文、官方系统卡、固定版本 Diffusers 文档及 C2PA 规范；不穷尽 Stable Diffusion checkpoint、产品 API 和生成评测。
- **检索问题：** 文本条件在系统组件间如何传递；CFG 和编辑参数应如何与具体接口区分；结构条件怎样初始化并注入；低秩参数更新和主体驱动微调是否同一概念；DALL·E 2 的 prior/decoder 与 DALL·E 3 的产品层证据各支持什么；质量、组合遵循和来源凭证有哪些不能互相替代的边界。

| 候选主题 | 来源证据 | 去向与决定 |
| --- | --- | --- |
| SDXL 双文本编码器与 pooled prompt embedding | SDXL v1 §2.1/Table 1；Diffusers v0.35.1 SDXL pipeline 参数与 `encode_prompt` 接口。 | 纳入文本条件接口段，用 SDXL 作特定路线示例，不泛化到所有 Stable Diffusion。 |
| SD3 的 rectified-flow Transformer 与 MM-DiT | SD3 v1 §4；独立文本/图像参数及联合注意力信息交换。 | 纳入技术路线演进，限定到该论文模型和实验，不改变章节结构。 |
| ControlNet 的 frozen copy 与 zero convolution | ControlNet v1 §3.1 公式 (1)–(4)。 | 纳入控制机制和零初始化前向例，条件图对齐保留为独立输入契约。 |
| DreamBooth 与 LoRA | DreamBooth v1 摘要和 §§4.1–4.2；LoRA v2 §4.1；Diffusers v0.35.1 training/loader 文档。 | 在既有个性化段区分 subject-driven 方法与权重更新参数化，并说明可组合；不另开章节。 |
| img2img strength 与 inpainting mask | Diffusers v0.35.1 img2img API 和 inpainting 指南。 | 用 strength 及两位置 mask 手算说明接口语义；将 scheduler timestep 和 mask 约定绑定版本/实现。 |
| DALL·E 公开路线与 DALL·E 3 产品编排 | DALL·E 早期论文 v2、DALL·E 2 v1 §§2–2.1、OpenAI DALL·E 3 System Card (2023-10-03)。 | 修正 DALL·E 2 的 prior/decoder 公式和可选 caption 条件；用 DALL·E 3 说明 GPT-4 prompt 编排与安全公开披露，不反推未披露架构。 |
| 文生图评估和 provenance | FID v6、CLIPScore v3、GenEval v1、T2I-CompBench 会议版 v2、C2PA 2.5 §1.2–1.3。 | 分开总体分布、caption-derived proxy、组合任务和来源完整性；不把任一指标或签名当作整体质量/真实性证明。 |
| 扩散系统的 attention 数量口径 | LDM v2 的条件 cross-attention 机制；按 shape 推导逻辑 score-cell 数。 | 统一按 batch、采样步、层、head、latent query 和条件分支计数；明确逻辑 score cells 不等于 FLOPs 或峰值显存。 |

**扫描结论：** 候选主题均可落入现有“系统组件—条件/控制—模型路线—评估/安全”教学链，不改变本册职责或章间关系。T2I-CompBench++ 作为后续不同版本，不与原会议版混称；DALL·E 3 系统卡不是生成架构说明。

来源事实、固定版本、阅读位置和证据边界单独保存在[文生图系统研究记录](../../research/text-to-image-systems-control-and-evaluation-2026-11.md)；C2PA 的已有规范核验记录见[来源凭证研究记录](../../research/c2pa-provenance-validation-2026.md)。本审计只保存书籍取舍和工作状态。

## 教学链、篇幅与数值例子

- **length_status：** 全篇一个核心单元；按解释性汉字计数脚本得到 **8,527 字**，下限 1,800。排除 §§15–16、标题、代码/数学围栏、表格、状态行、行内代码/公式和链接目标。字数达标之外，已人工复读检查因果、术语、证据和适用边界。
- **example_status：** 通过。核心手算覆盖输入、变量与中间步骤，所有数字均标为教学输入而非 checkpoint/生产实测：
  - latent shape：512×512、scale=8、4 通道得到 `[1,4,64,64]`；空间位置 4096，元素比 0.020833。
  - CFG/cross-attention：768×512、scale=8 得 6144 个 latent query；单 head/layer/单分支/单步为 `6144×77=473,088` 个逻辑 score cells；正负两支为 946,176，30 步累计 28,385,280，再乘实际 heads/layers 才得到对应完整逻辑计数。
  - img2img：30 步、strength 0.55，教学映射 `floor(30×0.55)=16` 个活动步，列表起始索引 `30−16=14`；不是 scheduler 实际 timestep。
  - inpainting：`m=[1,0]`、新状态 `[0.9,0.3]`、同一噪声步原图状态 `[0.4,0.8]`，混合为 `[0.9,0.8]`；第二位置保留。
  - ControlNet：`h_base=[0.2,0.8]`、残差 `[-0.2,0.4]`、`λ=0.5`，逐元素加法得到 `[0.1,1.0]`；零初始化卷积的初始输出边界由论文支持，玩具向量不是论文输出。
  - LoRA：单矩阵 `4096×4096`、rank 8，低秩更新参数 `8×(4096+4096)=65,536`，占该矩阵 0.390625%；没有据此推断训练总显存/算力比例。
  - DALL·E 自回归 token：32×32 image-token grid 得 1,024 个 image tokens；DALL·E 2 另以 prior 和 decoder 两阶段说明条件路径。
  - FID 一维高斯 toy：均值 1/2、方差 4/1，均值项 1、协方差项 `4+1−2√(2×1×2)=1`，总 FID 为 2；只检查公式代入。
  - 单位成功任务成本：合成请求成本 0.02/0.03/0.05 美元、2 个通过 rubric，总成本 0.10 美元，除以 2 得每次成功 0.05 美元；失败成本保留在分子。

## code_status

- §14 唯一独立 Python 标准库代码块状态更新为“已验证（2026-11-12；Python 3.12.9；仅 Python 标准库；CPU）”。整册复读发现 `latent_shape` 用浮点除法计算 ceiling、`elem_count` 接受浮点/bool shape 元素，strength/guidance 检查也会把 bool 当有限数值；现改为整数 ceiling、正整数 shape 校验和有限实数标量校验。
- 运行命令：`./.tmp/llm-book-cpu-check/bin/python -c 'from pathlib import Path; p=Path("book-15-multimodal-generative-models/chapters/07-stable-diffusion与dalle.md"); s=p.read_text(encoding="utf-8"); code=s.split("~~~python\n",1)[1].split("\n~~~",1)[0]; exec(compile(code,str(p),"exec"))'`。
- 示例输入：单个 batch、512×512、VAE scale 8、4 latent channels、30 steps、CFG 7.5、strength 0.55、32×32 DALL·E token grid；不加载模型、不生成图片。
- 从 Markdown 提取并运行完整代码；latent shape、token 形状、分支 forward-equivalents、score-cell 算术、img2img 索引、ControlNet shape 和 DALL·E token 账本均与正文一致。9 项主检查与 13 项边界检查全部通过；新增覆盖浮点/bool 尺寸、bool 强度参数和 $10^{500}$ 尺寸下的精确整数 ceiling。正常输出保持不变，完整 stdout 为：

~~~text
prompt_token_count=14
negative_token_count=6
text_condition_shape=(1, 16, 768)
negative_condition_shape=(1, 8, 768)
latent_shape=(1, 4, 64, 64)
latent_spatial_cells=4096
spatial_ratio=0.015625
element_ratio=0.020833
denoiser_forward_equivalents=60
cross_attention_score_cells_per_fullres_head_layer=2949120
img2img_active_steps=16
img2img_start_index=14
control_shape=(1, 1, 512, 512)
dalle_total_tokens=1040
dalle_image_token_ratio=64.0
checks={'latent_height_matches': True, 'latent_width_matches': True, 'denoiser_forward_equivalents_match_cfg_branches': True, 'cross_attention_counts_conditions': True, 'img2img_start_index_in_range': True, 'img2img_active_steps_in_range': True, 'latent_ratio_small': True, 'control_batch_matches': True, 'dalle_image_tokens_1024': True}
boundary_checks={'zero_steps_rejected': True, 'invalid_strength_rejected': True, 'negative_guidance_rejected': True, 'non_finite_guidance_rejected': True, 'non_finite_strength_rejected': True, 'zero_latent_channels_rejected': True, 'non_string_prompt_rejected': True, 'empty_shape_rejected': True, 'non_integer_latent_height_rejected': True, 'boolean_shape_dimension_rejected': True, 'boolean_guidance_scale_rejected': True, 'boolean_denoising_strength_rejected': True, 'huge_integer_ceil_exact': True}
all_checks_passed=True
~~~

- 验证边界：仅检查教学 shape、参数合同与算术；未加载模型、运行真实 sampler、检查 GPU/吞吐/峰值显存、训练 LoRA、验证 ControlNet 姿态或评估生成质量。

## formula_status、link_status 与 visual_status

- **formula_status：** 33 个 `math` 围栏，花括号平衡；围栏外 66 个未转义 `$` 标记，配成 33 对；无 `\operatorname`。45 个 fenced blocks（33 math、11 text、1 Python）均配对。源码检查通过；未宣称 GitHub MathJax 渲染通过。
- **link_status：** 47 个 Markdown 链接，全部为外链；本章没有本地链接目标需要存在性核验。本轮没有外链全量 HTTP crawl。
- **visual_status：** 保留来源/证据边界表和自回归/latent diffusion 工程对照表；用文本数据流和逐步算例呈现 pipeline 与状态变化。无 Mermaid；当前关系不需要额外图才能跟随。未做目标平台渲染检查。
- **结构检查：** 2 张 Markdown 表格列数一致；围栏配对；行内数学标记配对；无 `\operatorname`；`git diff --check` 通过。

## sync_status 与 open_risks

- **sync_status：** `PAPERS.md` 固定 SDXL v1、SD3/MM-DiT v1、LoRA v2、DreamBooth v1、ControlNet v1、FID v6、CLIPScore v3、GenEval v1、T2I-CompBench 会议版 v2 和 DALL·E 论文版本，并说明 DALL·E 3 系统卡边界；`EXERCISES.md` 更新 pipeline audit 题的 batch、CFG 分支、heads/layers、score-cell 与 img2img 口径；`GLOSSARY_EN_ZH.md` 增补 MM-DiT、LoRA、DreamBooth、C2PA；`INTERVIEW_BANK.md` 同步 SDXL/SD3、ControlNet 初始化、DALL·E 2/3 证据边界、DreamBooth/LoRA 区别及成本计数；`KNOWLEDGE_GRAPH.md` 增加文生图系统子链。已有 C2PA 研究记录复用；新增文生图来源事实记入 research 记录。简介、目录与十四章骨架不变。
- **open_risks：** 未运行真实文生图 pipeline、真实图像质量评估、GPU/显存/延迟或目标产品 API；FID 与 benchmark 示例不替代业务 prompt suite。公式未做 GitHub MathJax 渲染，47 个外链未全量 HTTP crawl。scheduler timestep 和 mask 约定仍必须按目标实现版本核对。
- **handoff：** 第七章门禁通过；第十五册架构审计已更新为前七章完成、剩余七章待审。下一步进入第八章《视频生成与 World Model》，重点核验时空生成、状态预测、交互 rollout 与真实具身控制边界；全套书修书 goal 继续 active。

## 整册连续复读追加记录（2026-11-12）

- 已沿第十五册主路径完整复读本章 §§0–17，对照第六章 diffusion 机制与第八章视频生成/World Model 的前后边界；系统章节职责清楚，无需改章节顺序或目录。
- 连续复读修正 §14 的输入合同和整数 shape ceiling；本地 demo 的正常形状/预算/输出未变，验证范围和来源结论不变。
- **下一步：** 进入第八章《视频生成与 World Model》继续整册连续复读。
