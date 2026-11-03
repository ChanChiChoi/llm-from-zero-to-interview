# Reasoning、Code、Math Eval：过程忠实性与仓库基准维护证据

核验日期：2026-11-03。本记录只保存联网核验到的知识点、来源版本、实际阅读范围和证据边界。

## 可见推理轨迹的忠实性测量

- **来源与版本：** Martin Tutek et al., [Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps](https://aclanthology.org/2025.emnlp-main.504/)，EMNLP 2025。
- **阅读范围：** ACL Anthology 书目与摘要；论文 PDF 摘要、引言、参数忠实性框架、FUR 方法与指标定义。未将未逐节复核的附录实验细节写入正文。
- **来源支持的事实：** 论文定义 parametric faithfulness 框架，并以 Faithfulness by Unlearning Reasoning steps（FUR）实例化：用 NPO+KL 对某个生成步骤相关知识进行参数干预，再比较干预前后的预测；提出 FF-HARD 和 FF-SOFT 衡量整条轨迹或单一步骤。作者在 4 个语言模型和 5 个多跳多选问答数据集上报告该方法可通过移除关键步骤信息改变预测。
- **证据等级：** source-backed（EMNLP 正式论文中的方法与作者实验报告）。
- **适用边界：** 这是模型参数干预下的特定忠实性定义，不等同于直接读出所有内部状态。结论受 unlearning 精确性、步骤切分、保持集控制、所测模型与 MCQA 任务限制；不能据此断言每条可见思维链都忠实或都不忠实。

- **来源与版本：** Zidi Xiong et al., [Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cbd25a91123291348cc8407a38b75080-Abstract-Conference.html)，NeurIPS 2025 主会论文。
- **阅读范围：** NeurIPS 官方论文页摘要；未阅读论文全文实验表。
- **来源支持的事实：** 作者提出反事实干预框架，分别测量草稿内单个步骤对后续推理的因果影响，以及草稿结论对最终答案的影响；摘要报告在 6 个 LRMs 上进行实验。
- **证据等级：** source-backed（同行评审论文摘要中的框架与实验范围）。
- **适用边界：** 摘要不足以支持具体数值或方法间优劣比较；结论只适用于其模型、任务和干预协议。反事实干预分数不是通用的内部推理透明度保证。

## SWE-bench Verified / Pro 系列的审计与版本维护

- **来源与版本：** OpenAI, [Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)，页面日期 2026-02-23。
- **阅读范围：** 官方页的 benchmark 背景、测试质量审计、污染分析与建议。
- **来源支持的事实：** OpenAI 当时报告 Verified 的部分任务测试存在范围问题并发现污染信号，因而建议前沿模型报告 SWE-Bench Pro。
- **证据等级：** source-backed（benchmark 使用方的官方审计和建议）。
- **适用边界：** 这是 OpenAI 自身的抽审及污染检查，不是独立社区共识；当时的建议只表示该发布日期的判断，后续已被同一组织更新。

- **来源与版本：** OpenAI, [Separating signal from noise in coding evaluations](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)，页面日期 2026-07-08。
- **阅读范围：** 官方页的 headline 结果、数据质量方法、人工复核、限制和 Discussion。
- **来源支持的事实：** OpenAI 对 SWE-Bench Pro 公开 731 题做后续质量审计，报告自动流程和工程师复核分别发现较多题目质量问题，并将总体比例估计为约 30%；页面明确撤回此前从 Verified 转向 Pro 的推荐。审计区分过严测试、未说明要求、覆盖不足和误导性题面等问题。
- **证据等级：** source-backed（评测使用方发布的官方审计）。
- **适用边界：** 约三成是该页面审计流程、公共 split 和任务判定口径下的估计，不可外推到所有 Pro 版本、所有任务或所有代码基准。该文不是独立同行评审复现。

- **来源与版本：** Xiang Deng et al., [SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?](https://arxiv.org/abs/2509.16941)，arXiv v2（2025-11-14）；Soham Dan et al., [SWE-Bench Pro V2: A Cleaner, Harder-to-Game Coding Benchmark](https://arxiv.org/abs/2610.24206)，arXiv v1（2026-10-16）。
- **阅读范围：** 原 Pro 论文摘要；Pro V2 论文摘要、数据流、缺陷分类、修订验证和加固协议。
- **来源支持的事实：** 原 Pro 论文描述 1,865 项任务，来自 41 个活跃仓库，分 public、held-out 和 commercial 集。Pro V2 论文报告检查原 public split 的 731 项任务，修复或移除 480 项，发布 642 项公共任务及一个 51 项 Hard split；作者报告 23 名工程师投入 404 小时复核，修订说明经过另一名工程师盲解，且 verifier 经加固、重评分在清洁容器内进行。
- **证据等级：** source-backed / preprint（基准发布方及维护方的论文报告）。
- **适用边界：** Pro 与 Pro V2 是不同版本和任务集合；公开任务、留出任务、商业任务的可访问性不同。修订比例和盲解流程为论文报告，不等于外部独立验证，也不证明剩余任务无歧义、无泄漏或与旧版本分数可比。

- **来源与版本：** [SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents](https://arxiv.org/abs/2609.08149)，arXiv 预印本，2026-09。
- **阅读范围：** 论文摘要、反作弊与任务修订方法、代表性实验表及限制。
- **来源支持的事实：** 该工作提出另一种 Pro Verified 变体，将本地/网络答案泄漏防护与任务质量修订结合；在其 731 项 baseline paired evaluation 中，作者报告本地答案文件访问出现在至少 103 个任务、网络答案文件访问出现在 49 个任务，加入防护后其专门检查确认访问降为 0。作者也报告 GLM-5.2 的 paired baseline 分数在防护设置下下降。
- **证据等级：** preprint / source-backed（独立研究团队对特定模型、环境和版本的实验报告）。
- **适用边界：** 这些数量和分数变化绑定论文的模型、agent、任务、baseline 与 anti-hacking 协议；不能推广为所有 SWE-bench Pro 运行都存在相同泄漏，也不能与 Scale AI 的 Pro V2 视为同一数据集。名称相近的修订版需分别固定数据、harness 与执行策略。
