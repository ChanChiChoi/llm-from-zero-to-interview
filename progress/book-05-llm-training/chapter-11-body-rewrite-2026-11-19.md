# 第五册第十一章：正文教学路径修订

日期：2026-11-19

## 范围与阅读路径

- 按用户要求，仅处理 `book-05-llm-training/chapters/11-后训练与对齐训练.md` 的正文编写，不重新审查书册定位或章节定位/组织。
- 沿正文现有承诺复核后训练反馈、偏好优化、在线 RL/RLVR、方法比较、训练风险、工程案例、评估与诊断。保留原有章节覆盖、外部链接和第 17 节已验证的分析示例代码；不改训练代码、目录或配套资料。
- 讲解主线继续使用算术题 `$3/4+1/4=?$`：示范回答解释 SFT，chosen/rejected 解释 RM、DPO 和 SimPO，同题多条采样解释 GRPO，PPO-RLHF 保留独立的六步贯穿算例。长回答的 ratio 粒度、采样、裁剪和长度归一化方法则用对应的 token/response 小例子说明。

## 正文改动

- 开篇先区分示范、离线偏好、在线采样与可验证反馈，修正容易误读为固定先后关系的路线图，明确 PPO-RLHF 与 DPO 是不同路线，不是必须顺序经过的阶段。
- SFT 小节改用同一算术题说明示范 token、mask 和两位置交叉熵，并保留 EOS、工具字段与空 mask 的必要边界。
- 为 DPO、SimPO、GRPO、GSPO、DAPO、DCPO 增加有具体内容的背景、目的、问题或意义说明；补齐/统一适用场景与优缺点表达。例子标明假设值，不把 reward 分数说成正确性证明。
- 压缩 SimPO 中重复解释的 margin、平均 log probability 和长度影响，保留 token sum/mean 对目标的区别、调参含义、reference-free 漂移风险与 mask 边界。
- 根据 arXiv 页面元数据更新 GAPO 状态：该论文已被 EMNLP 2026 主会录用，不再与仍为早期预印本的 DHPO 一并称作“早期预印本候选”。正文及研究记录仍将其结果限定为作者报告的模型、任务和协议；来源证据补入 `research/rlhf-preference-and-policy-optimization-2026.md`。
- PPO-RLHF 六步示例及其中的公式、六个教学 token 片段、reward、GAE、returns 和 ratio 数值均保留；不改已验证的第 17 节代码。

## 联网核验与证据边界

- 本轮依模块 03 联网核对正文的原始论文和官方披露；直接复核 PPO/InstructGPT、DPO、SimPO、DeepSeekMath/GRPO、DAPO、DrGRPO、GSPO、DCPO、DHPO、GAPO 的论文页面，以及 gpt-oss、Grok 4.7 的发布页。GLM-5.3 内容沿用 `research/llm-training-and-post-training-systems-2026.md` 已记录的官方来源和证据边界。
- 重点复核论文原文与版本元数据：DAPO 的组合式 recipe 和 AIME24 设定；GSPO 的 sequence likelihood ratio 与 sequence-level clipping；DCPO 的动态 clipping、平滑 advantage 与论文自定义 RUR；DHPO 的 token/sequence ratio 混合；GAPO 的 advantage-adaptive clipping 与 EMNLP 2026 录用状态。
- 论文和发布方报告仅支持各自公开的定义、实验和披露；章节保留“作者在其模型、任务、数据、评估协议下报告”的边界，不据此拼成跨论文总排名或推断未公开训练配方。

## 检查与状态

- 手工复核本轮新增的 SFT 平均损失、DPO margin/logit、GRPO 组内均值/标准差/advantage 示例。
- 完成后检查 Markdown 公式、围栏、链接和 `git diff --check`；未运行测试或 PDF 构建。
- 本记录只表示本轮正文教学修订，不代表重新审定书册或章节结构；若正文发现会改变主题范围或章节职责的新证据，应另按模块 01/02a 处理。

## 2026-11-19：按更新后的模块 02 重写初学者讲解路径

本节记录本日后续重写；它取代此前同日版本中对若干偏好方法采用变量代入和公式推导的讲解安排。

- **范围：** 只重写第五册第十一章正文，不重新审查书册定位或章节结构。
- **读者路径：** 以“3/4 + 1/4 = ?”贯穿示范、偏好比较、可验证奖励和在线采样；安全与领域任务使用各自完整场景。每个主知识点先说明真实对象、输入、动作和结果，再总结术语、一般规则与边界。
- **方法讲解：** 将 SFT、RM、DPO、SimPO、GRPO、GSPO、DAPO、DCPO、DeepSeek-R1/RLVR、领域专家和 on-policy distillation 改为例子、完整回答对、rollout 表或决策表主导；保留方法选择、适用条件、优缺点及论文证据边界。
- **PPO-RLHF：** 保留用户已认可的伪代码与六 token 贯穿算例；把 value、终止 mask、TD 误差、六步 GAE 和 returns 改成独立 math 公式块，修正已失效的 §1.3 裁剪交叉引用。
- **代码：** 第 17 节已经验证的分析 demo 及说明原样保留。
- **联网核验：** 对照 PPO、InstructGPT、DPO、DeepSeekMath、SimPO、DAPO、DrGRPO、GSPO、DCPO、DHPO、DeepSeek-R1 的原始论文，以及 GLM-5.3、gpt-oss、Grok 4.7 官方披露；补核 GAPO 已收录 EMNLP 2026 主会的 ACL Anthology 正式记录。新增来源证据写入 research/rlhf-preference-and-policy-optimization-2026.md。
- **验收：** 检查章节标题、公式块、围栏、链接和补丁空白；本轮不运行测试或 PDF 构建。


## 2026-11-19：按独立算法重做第 4–9 节的讲解路径

- 根据读者反馈，撤回让 DPO、SimPO、GRPO、GSPO、DAPO、DCPO 共用“3/4 + 1/4”例子的组织方式。前面的反馈概念与 PPO-RLHF 继续沿用分数题；第 4–9 节每个算法各用不同任务，避免用单一玩具例子承载不同机制。
- 六节都改为“问题/目的 → 算法伪代码 → 按伪代码走一遍本算法专属完整例子 → 适用场景与优缺点”。伪代码定义输入、状态、核心操作和输出；例子继续追踪数据、reward、ratio、advantage 或状态变化，不以一组孤立数字代替算法 walkthrough。
- DPO 使用临时加班回复偏好对；SimPO 使用缓存解释及长度平均 log probability；GRPO 使用保持顺序去重的代码与单测；GSPO 使用值班回滚说明并计算序列级 ratio；DAPO 使用 1 到 100 求和来走动态采样、Clip-Higher、token loss reduction 与长度处理；DCPO 使用 19×21 来走历史 advantage、动态 clipping 和 OTM。
- 根据 DCPO 原论文补齐 OTM loss 的 response 内 token mean 解释，并修正 advantage 平滑说明：统计来自同一 prompt 的当前与累计 reward，平滑后选择绝对值较小的候选；论文 objective 不含 PPO-RLHF 的 reference KL 项。教学数值明确标注为假设，不冒充论文默认参数。
- 联网对照 DPO、SimPO、DeepSeekMath/GRPO、GSPO、DAPO、DCPO 原始论文；沿用 research/rlhf-preference-and-policy-optimization-2026.md 中已有的机制与证据边界。第 17 节 demo 不在本次修改范围。

## 2026-11-19：进一步按 PPO-RLHF 形式走查独立算法

- 按用户反馈复核 §4–§9：每节都以本算法的伪代码呈现全局路径，再沿不同任务的专属例子追踪偏好/奖励、相对优势或概率比、裁剪/损失和更新方向；不再让一个例子兼任六种算法的说明。
- 加细 GRPO 的 token ratio 与 clipped objective、GSPO 的序列 ratio 与序列级裁剪、DAPO 的动态采样和 token 级 loss、DCPO 的历史优势平滑、动态裁剪与 OTM 走查，让例子能接上各自伪代码的更新阶段。
- 进一步核对原论文：DAPO 的正确性 reward 刻度、动态采样、token-level loss 与 soft overlong shaping；GSPO 的长度归一化序列 importance ratio；DCPO 的动态边界、平滑优势与 OTM。相关数值均标为教学设定，论文结果只限其公开的模型和评测协议。
- 本轮仍只改第十一章 §4–§9 及本进度记录；§3 PPO-RLHF 和第 17 节 demo 保持原样。

## 本轮统一 §4–§9 伪代码形式（2026-11-19）

- 按读者反馈，将 DPO、SimPO、GRPO、GSPO、DAPO、DCPO 六个独立算法的伪代码统一改成 PPO-RLHF 的函数调用风格，并将代码围栏标为 `python` 以显示缩进、循环和调用关系。
- 为偏好打分、组内 advantage、token/sequence ratio、动态采样、动态 clipping 与 OTM 等核心调用补充 `【核心】` 注释；其余注释说明输入约定、mask、评分器、终止/截断或不训练额外模型等要点。
- 保留每个算法自己的完整例子、变量语义和原有正文解释；不改 §3 PPO-RLHF、数学公式、研究结论或第 17 节 demo。伪代码仍是机制骨架，不声明为可执行实现。
- 静态检查 `git diff --check`、Markdown 围栏和公式/链接边界；未运行测试或 PDF 构建。

## 2026-11-19：重写 §4–§9 伪代码配套讲解

- 按用户指出的问题，重新通读完整模块 02，并复核第十一章伪代码和下方步骤是否逐个对应。六种独立方法继续使用各自的完整例子，不共用一个例子代替机制说明。
- DPO、SimPO、GRPO、GSPO、DAPO、DCPO 的步骤现在按伪代码调用顺序展开，明确指出每一步由哪个函数处理、输入输出怎样变化，以及这些信号如何到达 policy 更新；补足核心函数注释。
- 修正 GSPO 手算中“从长回答抽五个代表位置就计算整句 ratio”的不严谨说法，改为明确的五位置教学回答；为 DCPO 伪代码补上 prompt 历史的初始化，并在算例中对齐历史写入、累计标准化、动态 clip、token ratio、OTM 和 RUR/evaluate。
- 复核并清除 DPO 的重复例子标题；检查其余算法例子与调用步骤的顺序。PPO-RLHF 的既有六步讲解和第 17 节已验证 demo 保留。
- 机制边界依据本轮核对的 DPO、SimPO、DeepSeekMath/GRPO、GSPO、DAPO、DCPO 原始论文；研究记录同步补充 DAPO 的 token reduction/超长处理与 DCPO OTM 的逐回答 mean 后求和定义。未运行测试或 PDF 构建。

## 2026-11-20：依据教学型联网来源重写算法讲解步骤

- 依用户要求重读当前模块 02、03、04、05 与新增《联网数据来源》，并重新核对恢复入口和本章交接记录。将课程讲义、公开 notes 与实践 walkthrough 作为教学路径的默认检索来源，与论文机制核验并行使用。
- 新增研究资料记录在 `research/rlhf-preference-and-policy-optimization-2026.md`：阅读 RLHF & Post-Training Course 的 PPO/RLHF、DPO 讲义，Hugging Face GRPO worked example 和 OpenEnv v0.6.0 端到端教程；记录它们适用于讲解顺序和过程追踪的范围，并注明不能替代原论文事实核验。
- 重写 §4–§9 的例子步骤：各段仍以自身伪代码为全局骨架并使用独立任务；把 DPO/SimPO 改为追踪偏好对与 policy/reference 比较，把 GRPO 改为追踪整组候选如何收到测试反馈并变成 token 更新，把 GSPO 的局部 token ratio 改为位置表后汇总成回答级 ratio，把 DAPO 的长度处理改为自然结束/硬截断状态表，把 DCPO 的 advantage 平滑、动态 clipping 和 OTM 改为状态/资源账本。只在能推广机制时保留公式或数值关系，删去单独换算 logistic loss 的段落。
- 将 §11 的 DrGRPO 长度归一化说明改为短答/长答在三种 loss reduction 下的对照表；在 §20 增列课程与实现型教学资料供读者继续学习。保留已认可的 §3 PPO-RLHF 伪代码与六 token 算例，保留 §17 已验证 demo 的可执行代码和输出。
- 按公式规范给行内数学与全角标点留出分界空格，并在 DPO 例子之后补一条汇总目标式，将公式放在具体过程之后作一般化。
- 本轮核验课程和教学资料的具体页面；算法事实沿用并对照 research 中已经记录的 PPO、DPO、DeepSeekMath/GRPO、GSPO、DAPO、DCPO 与 DrGRPO 原始来源。更新后检查 `git diff --check`、Markdown 围栏、行内公式定界符、全角标点边界和本地链接；未运行测试或 PDF 构建。
