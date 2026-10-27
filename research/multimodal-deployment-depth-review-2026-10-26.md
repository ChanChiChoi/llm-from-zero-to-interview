# 第六册第十章：多模态部署深度精修研究记录

- 日期：2026-10-26
- 正文：[`book-06-llm-deployment/chapters/10-多模态部署.md`](../book-06-llm-deployment/chapters/10-多模态部署.md)
- 依据：当前 [`WRITING_SPEC.md`](../WRITING_SPEC.md)、[`progress_v2.md`](../progress_v2.md) 和仓库 `AGENTS.md`。
- 结论：原章围绕媒体输入边界、VLM、OCR、ASR、TTS、视频、安全、容量、评测、发票/语音案例、demo、排障和总结的内容均保留；精修加入模型机制、代表性方法的目标/代价/证据、服务运行时交接方式和可执行边界检查。没有把整章改成模型或论文名录。

## 本轮要回答的问题

1. 图片尺寸、视觉 token 和上下文预算如何联系？动态分辨率会不会自动保证上限？
2. 文档理解该把 OCR 当作可定位中间产物，还是直接训练 OCR-free 模型？两者分别省掉什么、又失去什么？
3. 流式 ASR 的 chunk 和 padding 如何分别影响更新频率、声学上下文、重复工作与并发？RTF 是否能代表交互延迟？
4. 级联 ASR→LLM→TTS、原生 speech/omni 与多阶段 TTS runtime 在何种约束下更合适？
5. 视频抽帧和编码后 token pruning 分别在哪一段省成本？EVS、VidCom2 的论文数字和当前 serving 支持有什么边界？
6. 多模态媒体 URL、图片内文字和视频字幕如何把风险带到网络访问或工具副作用？
7. 原 Python demo 的输出、预算上限和非有限输入是否与代码行为一致？

## 代表性研究与工程资料

| 来源 | 解决的问题与机制 | 本章使用的作者证据/工程事实 | 限制与落地验证 |
| --- | --- | --- | --- |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | Naive Dynamic Resolution 允许不同图像展开为不同数量视觉 token；M-RoPE 统一文本、图像和视频的位置表示。 | 论文明确将视觉序列长度设计成随输入变化的资源分配变量。 | 动态不等于无限；patch 数、merge、crop、processor 和真实模型上下文合同不同。必须测目标 processor 输出、padding、长尾上下文和关键小字质量。 |
| [LayoutLMv3](https://arxiv.org/abs/2204.08387) 与 [Donut](https://arxiv.org/abs/2111.15664) | LayoutLMv3 联合文本与二维版面/图像表示；Donut 直接从文档图像生成结构化文本，避免外部 OCR 的成本、文种/文档适配困难和错误传播。 | Donut 论文在其任务上报告了有竞争力的速度与准确率；两种路线代表了显式 OCR/layout 与 OCR-free image-to-structure 的目标差异。 | 文本框便于审计和定位，依赖 OCR；OCR-free 减少一个服务却不自动返回像素证据。用字段关系、证据区域、schema 有效性、人工复核和单位成功成本在目标文档集比较。 |
| [NVIDIA Riva ASR pipeline configuration](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/public/asr/asr-pipeline-configuration.html) 与 [ASR overview](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html) | 通过 chunk 频率提供 partial 更新，用左右 padding 提供声学上下文；streaming 和 offline 是不同调用/延迟路径。 | Riva 文档中的低延迟配置示例 `chunk=0.16s,padding=1.92s`，高吞吐示例 `chunk=0.8s,padding=1.6s`；两者的窗口均约 4 秒，更新间隔相差 5 倍。 | 这是 Riva Conformer pipeline 参数，不应当作所有 ASR 的固定配方。报告 WER/CER、partial/stable/final 延迟、GPU 并发、窗口重算/特征复用和 endpoint 结果。 |
| [Qwen3-Omni](https://arxiv.org/abs/2509.17765) 与 [Qwen3.5-Omni](https://arxiv.org/abs/2604.15804) | Thinker–Talker 和 codec/token decoder 把多模态理解与语音生成组合；ARIA 针对文本/语音 tokenizer 速率失配导致的流式韵律不稳做自适应交织对齐。 | Qwen3-Omni 报告 234 ms 理论冷启动首包。 | 这是作者定义的理论首包，不是跨系统生产 TTFA；ARIＡ结论绑定该模型。级联系统在审计和替换上更清晰，原生模型减少文字边界但不替代宿主权限和事务控制。 |
| [EVS](https://arxiv.org/abs/2510.14624) | 通过相邻帧 patch 的时间静态性剪枝，推理期保留位置身份，减少视觉序列冗余。 | 论文摘要报告其设置下 LLM TTFT 最多降低 4×，准确率损失较小。 | 静态字幕、仪表或小目标也可能是任务证据；TTFT 不等于完整任务延迟。按任务测 temporal recall、grounding、质量—token 曲线。 |
| [VidCom2](https://arxiv.org/abs/2505.14454) | 量化每帧独特性，并按帧调整压缩强度，避免平均删 token 时误删独特信号。 | 作者报告 LLaVA-OneVision 保留 25% 视觉 token 时达到原性能的 99.6%，并减少 70.8% 的 LLM generation latency。 | 数字绑定指定模型、benchmark、压缩策略和 generation 指标，不等于所有视频可无损压到 25%。需验证少见事件、静态证据和长视频分层。 |
| [当前 vLLM Multimodal Inputs 文档](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | 把视频 pruning 放在 vision encoder 后；提供媒体 URL 域限制和禁用重定向的入口。 | 文档说明 EVS 支持实现多模态 pruning 的模型；VidCom2 当前仅支持 Qwen3-VL；动态保留长度会禁用 encoder CUDA graphs。文档建议 `--allowed-media-domains` 并将 `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0`，以减小任意 URL fetch 的 SSRF 风险。 | `latest` 持续变化；服务接受参数不代表目标模型/输入可用或性能为正。框架域限制仍需结合下载器隔离、服务端网络出口策略和目标 revision 验证。 |
| [Image-based Prompt Injection](https://arxiv.org/abs/2603.03637) | 在自然图像中隐藏指令，以黑盒方式诱导 MLLM 覆盖预期行为。 | 论文在 COCO/GPT-4 Turbo 上测试 12 种嵌入策略，最隐蔽配置最高报告 64% 攻击成功率。 | 只适用于论文模型、数据和 stealth 条件，不是当前模型通用攻击率。红队覆盖小字、背景融合、裁剪边界和正常任务效用；OCR 过滤不是唯一防线。 |
| [Video-MME-v2](https://arxiv.org/abs/2604.05015) | 用多点视觉聚合→时序动态→复杂多模态推理的三级任务，及相关问题组的非线性评估，检查单题正确掩盖的跨题不一致。 | 论文描述 12 位 annotator、50 位独立 reviewer 和最多 5 轮质量检查。 | 该 benchmark 定义只适用于其数据与评分协议；业务评估还需时间定位、证据支持、关键事件召回和目标视频切片。 |
| [vLLM-Omni Technical Report](https://arxiv.org/abs/2610.09307) | 把请求编排和状态放在 orchestrator，计算放入 stage pool/engine，connector 承担 stage 间 payload；async-chunk 让下游读取部分 codec 输出并行工作。 | Qwen3-Omni Fixed-Len-2500/900、并发 32、2×H100 测量中，async-chunk/no-async 的平均 TTFT 为 507.8/3420.1 ms，E2EL 为 72.63/136.62 s，音频 RTF 为 0.630/0.927。TTFT 是首个文本 token。另一个 H200 Qwen3-TTS CustomVoice 并发 8 测量中，融合单 GPU/默认双 GPU 吞吐为 111.6/81.1 audio-s/s。 | 论文报告有固定 workload、hardware、commit、warm protocol 和 nightly 条件，不能外推成通用 runtime 排名。TTFT 与音频 TTFP 不同；融合与解耦须在同一工作负载验证吞吐、尾延迟、显存、排队和失败恢复。 |

## 原章节覆盖图

| 原位置 | 保留内容与精修后的去向 |
| --- | --- |
| §1 请求边界 | 保留原文件、规范化输入、组件分层和同步/异步 job；作为后续预算、安全、trace 的输入合同。 |
| §2 VLM | 保留视觉 patch、KV、processor、多图引用和 embedding cache；新增 §2.6 动态分辨率/tiling/固定 resize 的预算与手算。 |
| §3 OCR | 保留 OCR pipeline、PDF 坏例、OCR/VLM 路由、CER/F1；新增 §3.5 LayoutLMv3、Donut、OCR+VLM 路由的目标/证据/限制。 |
| §4 ASR | 保留 Whisper、分段、partial/stable/final、WER/CER/RTF 和隐私；新增 §4.5 chunk/padding/endpointing 的窗口级成本演算。 |
| §5 TTS | 保留概念链路、首包、韵律、打断、音色安全；新增 §5.4 级联、原生 speech/omni 和专用流式 TTS 对照。 |
| §6 视频 | 保留时间采样、时序引用、视频生成状态、worker 隔离；新增 §6.5 抽帧与编码后剪枝的阶段差异、EVS/VidCom2 与 runtime 支持。 |
| §7 安全 | 保留解析器隔离、跨模态注入、输出审核、缓存/日志/删除；加入 URL fetch 的 SSRF 威胁面和图片内隐藏 prompt-injection 研究证据。 |
| §8 性能/容量 | 保留瓶颈、batch/cache、队列和逐段 trace；新增 §8.5 控制面、数据面、完整交接、chunk overlap 与 stage 融合的比较。 |
| §9 评估 | 保留文档、ASR/TTS、视频 tIoU、回放切片；在 §9.4 加跨题一致性/证据支持与 Video-MME-v2 的限制。 |
| §10 案例 | 保留发票图片问答与实时语音助手两个完整案例；新知识回到这些场景的证据、路由、取消和门禁。 |
| §11 demo | 保留标准库 patch/context/KV/latency/RTF/video/routes 演示；修复图像 token cap 超限、非有限值、上下文超限路由及输出示例不一致。 |
| §12 排障 | 保留效果、性能和稳定性问题的逐段定位；与新 trace 字段及压缩/partial 失败方式对应。 |
| §13 联合发布案例 | 保留发票和语音工作流、业务状态、隐私、灰度和失败回放；未压缩为简单的“端到端架构介绍”。 |
| §14 资料 | 扩至 27 项，加入动态分辨率、OCR-free、Riva ASR、Qwen Omni/TTS、视频 pruning、vLLM-Omni、Video-MME-v2 和图像注入研究。 |
| §15 小结 | 保留原结论并具体加入动态预算、chunk/padding、视频后编码剪枝、多阶段 handoff 和跨题一致性。 |

## 公式、数值与代码核验

- 图像 patch：`ceil(1800/32)×ceil(2400/32)=57×75=4275`。教学算法按 `cap=1536` 得到 1056×1408，即 33×44=1452 patch；结果满足上限，但不代替模型真实 resize/crop/merge 输出。
- Riva 窗口：`0.16+2×1.92=4.00s`；`0.8+2×1.6=4.00s`。在每次完整重算的简化假设下，更新间隔分别 0.16s/0.8s，对应约 25×/5× 窗口输入覆盖量；这不是 GPU FLOPs，runtime 特征复用需实测。
- 视频账本：45s×0.5 FPS 向上取整为 23 帧，输入假设为 5888 token；帧 cap=12 后为 3072 token。对编码后 token 均匀剪 75% 得 768 是理想化上界推算，非 EVS/VidCom2 的真实保留数。
- demo context：`620+2372+128=3120`，相对 4096 尚余 976。KV 粗账本为 `2×32×1×3120×8×128×2 / 1024²=390 MiB`。12 秒音频处理 4.2 秒，RTF=`4.2/12=0.35`。视频帧数 `ceil(45×0.5)=23`，cap 后 12 帧、3072 token。
- §11 Python block 从 Markdown 提取后成功编译和运行；实际 stdout 与正文 `text` fence 逐字一致，包含 `diagnostics` 输出。边界检查覆盖正常/极端长宽比与 cap=1、非正/非整数/布尔尺寸、NaN/Inf/超大尺寸、KV 非有限/负输入、路由 NaN/Inf/负数/布尔输入，以及上下文限制降至 1000 时的 `trim_or_reject` 分支。
- 代码仅为标准库教学账本，不连接真实图像 processor、VLM、OCR、ASR、TTS、video decoder、serving runtime、GPU、租户存储或生产流量；所有容量与时延数据是合成值。
- 未执行外部公式渲染；`git diff --check` 通过。无新增相对链接断点；第六册目录已有第十章条目。

## 配套资料同步

- `PAPERS.md`：增加视觉 token 预算/EVS/VidCom2 与多模态 serving/Video-MME-v2/image-injection 入口。
- `GLOSSARY_EN_ZH.md`：增加 Dynamic Resolution、M-RoPE、EVS、VidCom2、Video-MME-v2、async chunk handoff、RTF/RTFx。
- `INTERVIEW_BANK.md`：新增视频剪枝阶段比较、ASR chunk/padding 与多阶段 serving/async-chunk 三题，包含来源数字和定义边界。
- `EXERCISES.md`：新增五项与第六册第十章对应的预算、剪枝、Riva 配置、语音路线和 stage handoff 练习。

## 遗留风险

没有目标文档/音频/视频数据集、模型 processor、视觉编码器、ASR/TTS runtime、目标 GPU、真实请求 trace 或安全 red-team 环境。不能据此宣称目标部署的质量、费用、实时性、吞吐、SSRF 防护完整性或 prompt-injection 攻击率。EVS/VidCom2、Donut、Riva、Qwen Omni、Video-MME-v2 与 vLLM-Omni 的数值/机制分别受论文模型、任务、版本和 workload 限制；vLLM/Riva/Hugging Face `latest` 文档需在目标 release 重新核对。
