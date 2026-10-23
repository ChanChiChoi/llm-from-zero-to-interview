# 第四册后训练与对齐精修核验（2026-10-22）

## 整章收口

本章声明范围内验收完成，以下旧检查点的未完成列表为历史状态。最终验证：唯一 Python 块独立运行，六组非法账本输入（超预期计数、bool 计数、负计数、非布尔标签、切片重叠、不一致直接危险回答）均拒绝；DPO 非法输入、数值溢出、零分母、小 softplus 和方法例子通过。35 个公式块结构、围栏及 git diff --check 通过。

`python3 /tmp/check_posttraining_review.py --links` 经指定代理取得 24 个唯一外部页面 HTTP200，29 个目标的本地路径和外部引用锚点通过；16 篇 abs 题名与所引方法核对一致。版本 HTML 的 title 有时为章节标题（IPO 返回 1Introduction），该论文身份由 abs 题名及已读方法段落共同确认，未只依赖 HTTP 状态。目录/简介映射与下一章路径核对，练习、题库及论文索引同步。

来源证据限于下列明确读取的摘要、方法段落和官方说明；其余论文入口仅题名/可达性核验。没有全文阅读所有论文、运行真实训练、GPU 或生产系统，也未使用外部公式渲染器。RLHF 目标补最大化符号，隐私账本限制与 reference-free 成本表述一并修订。PDF 仍受环境缺少 Pandoc 限制。

## 第四个局部检查点：偏好目标与 V3.2

- [IPO v2](https://arxiv.org/html/2310.12036v2#S5.SS2)：读取 sampled loss、τ 正则说明和过拟合讨论；核对目标 1/(2τ)。正文区分原文 generation 动作概率与训练器 token 均值约定，补间隔两侧导数。
- [SimPO v2](https://arxiv.org/html/2405.14734v2)：读取长度归一化隐式奖励、目标 margin、下游退化及安全/诚实性限制段落。正文损失与原式一致，补数值及长度翻倍不变反例；不将结果外推成普遍简洁性或安全保证。
- [RLAIF 摘要](https://arxiv.org/abs/2309.00267)：核对 AI 反馈用于 RL 的定义与动机，未阅读全文；正文将纯规则奖励、AI 偏好离线优化分别说明。
- [V3.2 报告 v1](https://arxiv.org/html/2512.02556v1#S3)：读取第 3 节 specialist distillation/mixed RL、第 3.1 节 KL 估计、负优势 off-policy 屏蔽、Keep Routing 和 Keep Sampling Mask；屏蔽作用于策略项，不能推成所有失败数据及 KL 都删除。
- [官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/README.md)：读取三条技术主线、chat template 与 thinking-with-tools、Speciale 不支持工具调用的说明。main 为动态来源；本次取得相应文本，不代表固定版本或部署验证。

`python3 /tmp/check_posttraining_review.py` 通过，新增 IPO 两侧平方误差与 SimPO loss≈0.513015 数值断言；35 个公式块结构及 git diff --check 通过。EXERCISES.md 同步 5 项验收练习。未进行模型训练或外部数学渲染。全章外链、论文题名、诊断器完整输入边界与导航仍待验收，旧的待办列表以此处缩小后的范围为准。

## 第三个局部检查点：DAPO 与 GSPO

- [DAPO v1 第 3.2 节](https://arxiv.org/html/2503.14476v1#S3.SS2)：读取动态采样相关段落，明确过采样并过滤准确率等于 0 和 1 的题目组，不是只过滤全零奖励。正文加入三组二元奖励例子，区分零优势与包括辅助损失在内的总梯度。
- [GSPO v1 第 4.1 节](https://arxiv.org/html/2507.18071v1#S4.SS1)：读取式 6—8 及长度归一化说明，确认序列目标与 token ratio 几何平均。正文修正 GRPO 需连乘的暗示，说明共享 outcome 优势不为 GSPO 独有，且 clip 范围不可直接照搬。

以上为局部方法核验，未复现训练。`python3 /tmp/check_posttraining_review.py` 独立执行本章唯一 Python 块，并检查 DAPO 保留组及优势、GSPO 两步/四步几何平均、固定历史下交叉熵减教师熵等于 forward KL。35 个公式块结构、围栏及 git diff --check 通过。蒸馏部分仅澄清现有简化公式的数学边界，尚未完成具体方法论文对照。整章来源/外链和配套验收继续保留。

## 当前范围

全文初读7.1—7.15、小结及章末DeepSeek V3.2补充，局部核验SFT、LoRA、PPO与DPO。整章尚未验收。

## 来源与修订依据

- [DPO v2 HTML](https://arxiv.org/html/2305.18290v2)：读取KL正则目标、最优策略π∝π_ref exp(r/beta)及DPO推导相关公式。修正beta越大越快的无条件说法，区分KL惩罚含义与分类margin/梯度缩放。原始序列log probability为token求和，改均值会改变目标。
- [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：读取PPO-Clip没有硬约束、移除部分更新激励及KL early stopping的说明，正文不再宣称clip硬性限制ratio。
- [TRL DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)：读取相对reference目标和指标相关说明，动态main不当作本地安装或所有版本行为证明。

SFT改成内部shift的可核对示意：stored labels[-100,-100,a0,a1,EOS]，前四位置logits预测后四目标；LoRA用矩阵维度说明增量秩至多r，不保证等于r。公式宏和历史下标机械统一。

## 本地验证

唯一标准库Python块独立编译执行通过，输出assistant coverage0.917、mean DPO loss0.688、positive margin0.667、FRR0.500、ULR1.000。

softplus使用log1p，保留很小正值；dpo_loss拒绝bool、正log probability、非法beta和非有限缩放margin。首条DPO loss0.673347167228034复算通过，log1p_exp(-100)>0，零分母ratio(0,0)返回None。34个math块花括号/环境及总围栏结构通过，git diff --check通过。未运行外部数学渲染器、模型训练或GPU。

## 待验收

继续核验reward-human gap的共同量纲、偏好长度偏差、IPO/KTO/ORPO/SimPO的目标边界、GRPO/DAPO/GSPO/Dr.GRPO归一化、RLAIF与RLVR区分、on-policy蒸馏、诊断比例输入及章末V3.2证据。全章来源题名/外链、导航和配套同步未完成；PDF仍受Pandoc缺失限制。

## 第二个局部检查点

本轮读取并修订：

- [KTO v3 HTML](https://arxiv.org/html/2402.01306v3)第4节目标、KL参考点及类权重相关段落。正文移除只表达方向的二元交叉熵toy目标，改为有界sigmoid价值及E[λ-v]，区分全局KL参考点与具体实现估计/截零/梯度规则。
- [ORPO v2 HTML](https://arxiv.org/html/2403.07691v2)的长度平均log probability、odds和odds-ratio偏好项。补充P为平均token logprob的指数，不是原始序列概率；稳定实现和非空长度作为边界。
- [Understanding R1-Zero-Like Training v1 HTML](https://arxiv.org/html/2503.20783v1)关于长度偏置、question-level difficulty bias及固定MAX_TOKENS相关段落。正文补Dr.GRPO去std和固定最大长度归一化，不把动态有效token总数称为固定分母。

以上读取局部方法段落，未复现论文训练或声称全文核验。奖励-人工gap改为共同事件上校准后的胜率差，避免任意raw reward与人工分直接相减；无统一量纲时并列报告。

唯一代码块重新独立编译执行通过，正常五项输出不变；KTO好/坏样本loss约0.268941/0.731059，与好样本BCE0.313262区分，ORPO odds4/1的loss0.223144复算通过。35个math块的花括号/环境检查通过，git diff --check通过。仍未使用外部公式渲染器或真实训练。

整章保持进行中。后续重点为IPO/SimPO、GRPO/DAPO/GSPO、RLAIF、蒸馏及V3.2的来源、诊断器完整边界、全章外链与配套同步；首个检查点的未完成清单由此缩小，不表示其他方法自动完成。
