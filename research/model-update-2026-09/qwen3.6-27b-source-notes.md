# Qwen3.6-27B：Dense 混合注意力与 Agent 评测口径

核验日期：2026-09-24。模型锚点来自 Artificial Analysis；DataCurve DeepSWE 用来查有无精确 Agent 行。技术字段来自 Qwen 的官方 ModelScope 仓库、固定配置和 chat template。本条是 Qwen3.6-35B-A3B 之后的独立榜单锚点；两者有共享协议，但主体架构不同。

## 1. 榜单身份

- Artificial Analysis 有 [`qwen3-6-27b`](https://artificialanalysis.ai/models/qwen3-6-27b)（Reasoning）与 `qwen3-6-27b-non-reasoning`（Non-reasoning）两条，model inventory 标注日期 2026-04-22；按同一 Qwen3.6-27B 基础模型归并。
- 本轮 AA 详情 HTTP 200，4,051,359 bytes，SHA-256 `97375509bc5e428c1886206338a512e3ffb5bccd72b46bc92ed8cd5df8f17e44`。AA 的 Intelligence Index、provider、速度、价格等只记作榜单/provider 字段，不替代官方模型规格。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照 HTTP 200，268,036 bytes，SHA-256 `14436c31be1e50a0b62171e4ee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；未检出精确 `mini_swe_agent_qwen3_6_27b_*` / Qwen3.6-27B 行，不迁移 Qwen3.6-35B-A3B 或其他 Qwen 的 Agent 分数。

## 2. 官方资料和 revision

1. [Qwen 官方 ModelScope 模型卡](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B)：ModelScope 文件清单 API 快照 12,158 bytes / SHA-256 `363ecd222fc2b94adc8e8a743a0b3e51cc4d4035c374bdb552052a4d6e110690`。文件清单给出 README revision `cea40373b9214dd387123e68841890af30dcd469`、config/template revision `c53c4820996523bb6413f1002e24c5dfb0bad548`。
2. 固定 [README/model card](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B/resolve/cea40373b9214dd387123e68841890af30dcd469/README.md)：62,593 bytes / SHA-256 `bb936d6da51014f1edc9aa4cf9abf28d98695b7616ad56adfeeebfa752051d3d`。卡片引用文章题名 *Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model*，但 BibTeX URL 指向 Qwen 博客，不是 arXiv/可下载报告。
3. 固定 [config.json](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B/resolve/c53c4820996523bb6413f1002e24c5dfb0bad548/config.json)：4,308 bytes / SHA-256 `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c`；固定 [chat_template.jinja](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B/resolve/c53c4820996523bb6413f1002e24c5dfb0bad548/chat_template.jinja)：7,764 bytes / SHA-256 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`。同一 chat-template 内容哈希也见于 Qwen3.6-35B-A3B 的固定 template，故 `preserve_thinking` 是共享协议能力，不应重复包装成 27B 独有发明。
4. [Qwen3.6-27B 官方博客](https://qwen.ai/blog?id=qwen3.6-27b)：博客路由的页面仍是客户端 shell，但通过页面公开调用的 [Qwen 文章 API](https://qwen.ai/api/v2/article/?language=zh-CN&path=qwen3.6-27b&type=qwen_ai) 取得完整文章。2026-09-28 经 `10.24.27.134:7890` 返回 HTTP 200，JSON 94,527 bytes；本次响应 SHA-256 `036a9cfc6a38b04fed0b72aaf9339296296356dd9441a9e63d4b720283ed90c0`（含动态 `request_id`，不适合作为正文稳定标识）。其中 HTML 正文 91,758 bytes，SHA-256 `7748e75a7c5a47943d6abe4e6415cb6b8d4749713eeff39323eb831f2d8ae367`。API 元数据为《Qwen3.6-27B：270亿参数稠密模型，旗舰级编程能力》，Qwen Team，发布/修改时间 `2026-04-22T10:00:00+08:00`。文章本身注明引用标题 *Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model*，引用 URL 仍指向博客，不是独立技术报告。
5. arXiv 复查经备用代理 `10.237.126.170:1234` 成功：title-field 查询 `ti:Qwen3.6-27B` 与 `ti:Qwen3.6` 各返回 0 条；all-field 查询返回 50 条，其中有直接讨论该模型 serving 的第三方预印本，见下节。all-field HTML 为 258,469 bytes / SHA-256 `c46c1ece5e012fd42a04818b3317b3af663155a8fe6244326c07f9f5ca88b19b`，API XML 为 25,632 bytes / `83f0064fd8625927c5c93b44e9137711626444d97fea4ad7af878235ad48bf72`。标题查询的零结果不证明没有相关论文；ModelScope 卡没有给出 Qwen 自有技术报告的 arXiv ID，只把 citation 指向博客。

ModelScope 文件清单显示仓库有 15 个 safetensors 权重分片；本轮没有下载权重。ModelScope 仓库清单、ModelScope 文本卡、Qwen 官方文章 API、AA 页面和下节 arXiv 预印本是可复核证据。2026-09-28 的 agent 路径可访问百度和 Qwen 文章 API，但直抓 `qwenlm.github.io/zh/blog/qwen3.6-27b/` 仍连接失败；这只描述特定主机/路径，不代表 `7890` 全局不可用。

## 3. 架构：同系列里的 dense 路线

Qwen 官方卡称 Qwen3.6-27B 是 27B dense 模型，语言模型 hidden size 5,120、64 层、FFN intermediate size 17,408。公开层布局为：

```text
16 × [3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN)]
```

对应 config 每四层三项 `linear_attention`、一项 `full_attention`。Gated DeltaNet 为 16 个 QK heads / 48 个 V heads、head dimension 128；Gated Attention 为 24Q/4KV、head dimension 256、RoPE dimension 64。卡片称其模型类型为带 vision encoder 的 causal LM；config 中 vision tower 为 27 层、hidden 1,152、16 heads、patch 16、temporal patch 2、spatial merge 2，输出维度 5,120。

| 对照项 | Qwen3.6-27B | Qwen3.6-35B-A3B |
|---|---|---|
| 参数/FFN | 27B dense FFN，intermediate 17,408 | 35B total / 3B active，MoE 256 experts、8 routed + 1 shared |
| Transformer | 64 层；每 4 层 3×Gated DeltaNet + 1×Gated Attention | 40 层；相同 3+1 混合节奏，但每层接 MoE |
| 榜单配置 | AA Reasoning / Non-reasoning，2026-04-22 | AA Reasoning / Non-reasoning，2026-04-16 |
| thinking 模板 | `preserve_thinking` 与 `enable_thinking` 共用同一 chat-template 内容哈希 | 同一共享模板协议 |

所以不能只用模型名中的“27B/35B”作吞吐推断：一个是 dense FFN，一个是稀疏 MoE；层数、active parameter、路由/通信、resident weights 和显存账目均不同。config 中 `architectures: Qwen3_5ForConditionalGeneration`、`model_type: qwen3_5` 是实现/配置命名，不应覆盖官方仓库、排行榜与模型卡确认的 Qwen3.6-27B 身份。

## 4. MTP、thinking 和长上下文接口

- config 声明 `mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=false`；模型卡称 MTP 是 multi-step trained，并提供 serving 示例。但它没有发布跨引擎 acceptance rate，不能把示例 flags 当成同一个算法效果或吞吐承诺。
- ModelScope README 的 SGLang recipe 示例用 `NEXTN`、3 speculative steps、4 draft tokens；vLLM 示例用 `qwen3_next_mtp`、2 speculative tokens。引擎参数、draft 数和执行路径不同，不能横向比较；本轮未安装框架或载入权重。
- 默认 thinking 开启；卡片说 Qwen3 的 `/think`、`/nothink` soft switch 不适用于 Qwen3.6，应由 API 参数控制。`enable_thinking` 管当前生成，`preserve_thinking` 管输入 transcript 中旧 assistant reasoning 的保留。该 template 与 35B-A3B 相同；保留历史仍不是 server-side memory。
- native context 为 262,144，卡片以 YaRN `factor=4.0` 举例扩到 1,010,000；static YaRN 可能影响短文本。模型卡提醒 OOM 时降低窗口，但又建议 Agent/复杂任务保留至少 128K 以发挥其 thinking 能力；这是发布方的 serving 建议，不是独立测得的最低有效 context。

官方发布博客称 `preserve_thinking` 可把先前轮次的 thinking 内容保留在消息历史中，并推荐给 Agent 任务；这与固定 chat template 的职责一致，是 transcript 序列化/接口能力，不是持久记忆或新架构。博客展示 OpenAI-compatible Chat Completions/Responses 与 Anthropic-compatible API，并给出 OpenClaw、Qwen Code、Claude Code 集成示例。OpenClaw 示例配置 `contextWindow=131072`、`maxTokens=16384`，是该客户端/harness 的预算示例，低于模型卡的 262,144 native context，不能据此改写模型规格。博客对百炼 API 上线状态的表述前后不完全一致（介绍段称即将上线，后文又给调用示例）；本项目未对真实账号或 endpoint 发起 probe，不推断当前实际可用性。

## 5. 评测方法：分数必须带上 harness

Qwen 官方博客报告 Qwen3.6-27B 在 SWE-bench Verified、SWE-bench Pro、Terminal-Bench 2.0、SkillsBench 上分别为 77.2、53.5、59.3、48.2，并与 Qwen3.5-397B-A17B 的 76.2、50.9、52.5、30.0 作对比；这些是发布方结果，不是本项目独立复现，也不能单凭它们归因于参数规模或某项未披露技术。Qwen 模型卡提供多个 benchmark 的脚注条件，能直接抽成面试中的评测设计问题：

- SWE-Bench 系列用内部 bash + file-edit agent scaffold、temperature 1.0、top-p 0.95、200K context。卡片说对公开 SWE-Bench Pro 的部分问题作修正，并在 refined set 上评估全部 baselines。因此它不应无条件与原始 public set 的数字直接对比，至少应标 benchmark revision 和重跑基线事实。
- Terminal-Bench 2.0 绑定 Harbor/Terminus-2、3 小时 timeout、32 CPU / 48 GB RAM、256K context、最高 80K output、5 runs 均值。任务分数依赖系统资源、超时和 harness，而不是一个脱离环境的裸模型常数。
- SkillsBench 只取 78 个 self-contained 子集，排除 API-dependent tasks，跑 5 次平均；其评测覆盖范围不是完整原榜。
- QwenClawBench 被描述为 real-user-distribution Claw agent benchmark，但模型卡没有给出足以独立重建的数据/完整 verifier；应标为 publisher-described internal benchmark。
- QwenWebBench 是内部双语前端生成评测：7 类任务，自动渲染后由多模态 judge 判断 code/visual correctness，再用 Bradley–Terry/Elo 汇总偏好。面试讨论点是 judge consistency/calibration、browser/render version、pairwise 样本与置信区间、可复现任务集；Elo 不是 pass rate，也不能代替客观 verifier。

## 6. 相关外部 serving 预印本（非 Qwen 官方报告）

arXiv 于 2026-09-20 收录 Zhiyuan Ma 的单作者预印本 [GDN Tree-Scan: Served Tree Verification for Recurrent-Hybrid Language Models](https://arxiv.org/abs/2609.23900v1)。本轮 arXiv 页面 HTTP 200，275,937 bytes / SHA-256 `8b935c50f17284cb6ad85cf03454ce321ec50f6c80da6269f7fd48639034c0f7`；API XML 为 2,757 bytes / `7a149feb604a9450bd06f03b59bdc7762b50fd2b6ab2cd6b9f0458ae00e89986`。它是外部作者的研究预印本，不是 Qwen 发布材料、同行评审结论或本项目复现。

论文在公开 Qwen3.6-27B-FP8 上描述一条 vLLM serving 路径：FA2 tree-bias 让 attention 行只看 prompt 和祖先；每个 Gated DeltaNet 候选行则从其 parent recurrent state 做 branch-local scan/replay；MTP draft tree 提交后，只发布被接受 root-to-leaf 链上的状态。候选树若仅有正确 attention ancestry、却沿 packed row 顺序继承 recurrent state，兄弟分支就会带入不可能的历史。论文的 cat6root 在原生五步 MTP spine 上加一个 root sibling，并采用 device-side multidraft committer。

作者报告的 clean B=1、temperature 0.6、四个 SWE/Codex tasks 结果：native E5 与 cat6root committed tokens/event 为 4.11 与 4.82（+17.2%），verify-forward 为 0.137 与 0.138 秒；token-weighted decode TPS 为 18.80 与 23.88（+27.0%），per-request-equal TPS 为 17.80 与 18.51（+4.0%）。这是 decode 指标，不是 end-to-end 任务加速；论文说明重复的约 11K–14K prompt prefill 且未开 prefix cache，未给出可推广的 task-wall speedup。

等价性证据是每组 40-turn 的 recurrent-oracle p-rescore：cat6root 的 clear-margin flip 为 13.09%，native E5 为 12.90%，作者称差值落在观察到的 native numerical flip floor 内。论文明确这不是 full distribution-distance proof；位置样本相关，request-cluster bootstrap、更多 seeds、B=4 和 Stage-D phase timing 仍未完成。arXiv 正文所指的实现/测量 artifact 固定在 [Lumo_FlyWheel commit `55f55854328b37f262e97d57b5863d8fadd7ff76`](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/55f55854328b37f262e97d57b5863d8fadd7ff76)；本项目没有执行该代码、下载权重或复核其 GPU 测量。

## 7. 当前闭环状态

- 状态：**AA 单榜身份核验 + 官方 ModelScope model card/config/chat-template、Qwen 发布博客与 benchmark 方法摘录 + 一篇直接相关的第三方 serving 预印本**；DataCurve 当前无精确行。
- Qwen 官方博客正文现已通过官方文章 API 取得，但未检出标题直接命中 Qwen3.6 的 Qwen 自有技术报告。发布博客不披露新的内部架构或完整训练 recipe；外部预印本不证明 Qwen 官方采用其 serving 路线，也不能从发布方 benchmark 反推完整训练配方、内部 Agent scaffold 或质量置信区间。
- 未下载权重、未安装推理框架、未做真实 API probe、复现实验、MTP acceptance、GPU profiling 或独立 benchmark。

正式教学映射：[第二十一册第 83 章 83.16](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)；推测解码补充：[第二十四册第 61 章 61.31](../../book-24-llm-inference-engine/chapters/61-mtp与多token预测.md)；评测方法补充：[第二十册第 19 章 19.39](../../book-20-agent-harness-runtime/chapters/19-harness-aware-evaluation.md)。
