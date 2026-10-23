# 第四册预训练与数据精修核验（2026-10-22）

## 当前范围

全文初读6.1—6.14与小结，局部修订训练token口径、去重计数、packing边界和合成样本过滤。整章尚未验收。

## 来源与边界

- [TRL SFT Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)：读取packing及packing_strategy配置相关内容，不同策略存在截断/切分等差异；仅凭packing开关不能推出完整attention/loss隔离。动态main文档不是固定安装版本，本地未安装或执行TRL。
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)：读取摘要中近重复/长重复子串、记忆和train-test overlap的观察；不将论文特定比例或收益外推为通用去重效果，未复现工具或训练。

新增packing数值按causal shift定义直接推导，不声称目标框架默认自动处理；独立样本目标与可见前文的文档流目标分别说明。普通hash是候选匹配而非匿名化或零碰撞保证；本轮未验证生产hash/隐私系统。

## 本地验证

五个标准库Python块独立编译执行通过：quality记录score0.82；去重样例共有hash1、命中文档1；packing输出[[1,2,0,3,0],[4,5,6,0]]；边界矩阵B首query只允许自身；labels[-100,2,0,-100,0]，shift有效目标3；合成过滤输出[True,False]。

补查同一hash命中两条评估文档，distinct overlap=1而matched_eval_documents=2。合成答案“2”通过，空白答案拒绝；答案“3”若上游仍标verified=True也能通过，明确暴露教学检查器不验证事实的边界。

13个math块花括号/环境和总围栏结构通过，宏和历史下标统一，git diff --check通过。未调用外部数学渲染器、真实模型或训练。

## 整章验收补充

接续复读剩余主题，完成mixture温度定义域/空池/文档与token比例、有效epoch、collapse条件、scaling拟合、数据删除/恢复、消融预算与价值估计边界修订。扫描器命中与真实风险分开，空桶不算零风险；不可变版本不等于敏感原文永久保留。数据级删除与模型参数遗忘仍明确区分，未声称本地实现了遗忘或生产恢复。

新增来源读取：

- [The Curse of Recursion](https://arxiv.org/abs/2305.17493)摘要：递归生成数据训练中的长尾丢失和collapse风险，不复现其理论或实验。
- [Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413)摘要：其替换/累积设置的区别，保留原始真实数据并累积后续合成数据时的理论与实验结论；不外推成任意训练配比的保证。
- 复读此前下载的[Chinchilla v1 HTML](https://arxiv.org/html/2203.15556v1)拟合与compute约束公式：L(N,D)的两个幂律项及C≈6ND，分开训练compute-optimal与部署生命周期成本。

Atom核对8篇正文arXiv题名/版本：Chinchilla2203.15556v1、SelfInstruct2212.10560v2、Curse2305.17493v3、Datasheets1803.09010v8、ModelCards1810.03993v2、Influence1703.04730v3、DataShapley1904.02868v2、Accumulation2404.01413v2。元数据不代表逐篇全文阅读，C4/NIST/Datasets等入口可达不等于验证目标模型使用数据或平台合规。

最终临时检查器`/tmp/check_pretraining_review.py`通过：5个Python块独立编译执行；非法quality score/风险flag、空白文本/空训练集、超长packing/非法token与长度、合成verified类型等边界断言；packing矩阵/shift后的3个目标；mixture平方根权重1/3和6ND算术。原有正常输出保持一致。

13个math块和围栏结构、章节相对入口、本册目录/专题/练习/下一章目标存在，git diff --check通过。13个唯一外链最终HTTP200；首次沙箱脚本无法连接代理，获准后重试通过。EXERCISES.md新增5项计数与边界验收；题库/术语已有相关内容，不复制摘要。

本章声明范围内精修验收完成。未运行真实语料处理平台、近重复系统、TRL、模型训练、GPU、多节点恢复、遗忘或生产安全验证；合成样本检查仍只是字段审核。未调用外部数学渲染器，PDF仍受Pandoc缺失限制。全书和第二册遗留审计未完成。
