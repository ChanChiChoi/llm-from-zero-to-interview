# 后训练与对齐进阶精修记录（2026-10-07）

## 范围与结论

- `chapter`：`book-02-advanced-100/chapters/05-后训练与对齐进阶.md`。
- `scope`：从第 51 节读到第 65 节；修订前 2,564 行，当前 2,566 行。第二册目录范围为第 51—65 讲。
- `evidence_level`：`paper`、`official`、`local-demo`。
- `status`：完成；结论范围限于所列公开来源、公式核对和本地 CPU 教学示例。

## 主要修订

- PPO rollout 的 sampled-KL reward 使用生成轨迹时冻结的 `old_logp` 与 reference logprob 比较；`new_logp` 仅用于 PPO importance ratio，避免把策略更新后的 logprob 回算进 rollout reward。
- Truthfulness 评估把事实正确、幻觉和 unsupported claim 作为不同标注维度：分别用 `z_i`、`h_i`、`u_i` 表示，避免把每个事实错误都归类为幻觉。
- TRL SFT/DPO 文档改用固定仓库 revision，PPO 资料改为说明具体版本边界；不再把固定快照中缺失的 PPO Trainer 文档描述为当前接口。
- 第 65.2 节在 MMLU/MMLU-Pro、GSM8K/MATH、HumanEval/MBPP、IFEval、LongBench 名称旁补上任务切片含义、原始论文链接和评估污染/协议边界。
- OWASP Top 10 引用改到最终项目页；`PAPERS.md` 增加后训练评估基准论文入口。

## 资料与证据范围

### 公开评测论文

arXiv Atom API 返回以下题名/版本；八个 arXiv 页面都返回 HTTP 200：

| ID | 元数据标题 | 版本 |
|---|---|---|
| [2009.03300](https://arxiv.org/abs/2009.03300) | Measuring Massive Multitask Language Understanding | v3 |
| [2406.01574](https://arxiv.org/abs/2406.01574) | MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark | v6 |
| [2110.14168](https://arxiv.org/abs/2110.14168) | Training Verifiers to Solve Math Word Problems | v2 |
| [2103.03874](https://arxiv.org/abs/2103.03874) | Measuring Mathematical Problem Solving With the MATH Dataset | v2 |
| [2107.03374](https://arxiv.org/abs/2107.03374) | Evaluating Large Language Models Trained on Code | v2 |
| [2108.07732](https://arxiv.org/abs/2108.07732) | Program Synthesis with Large Language Models | v1 |
| [2311.07911](https://arxiv.org/abs/2311.07911) | Instruction-Following Evaluation for Large Language Models | v1 |
| [2308.14508](https://arxiv.org/abs/2308.14508) | LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding | v2 |

标题和 ID 核验用于确认链接对应目标 benchmark，不等于逐篇复核全部实验、数据切分或 scoring harness。正文提醒评估者固定具体版本、模板、生成预算和判分方式，并把公开基准与私有 holdout、真实任务分开。

### TRL 版本快照

- 固定 revision：[`956d742c1b8df9491c8382614e66415fb15bff9c` 的文档树](https://github.com/huggingface/trl/tree/956d742c1b8df9491c8382614e66415fb15bff9c/docs/source)。Git Trees API 返回 `truncated=false`；目录中有 `sft_trainer.md`、`dpo_trainer.md`，递归树中没有 PPO 路径。
- 固定的 [SFT Trainer 文档](https://raw.githubusercontent.com/huggingface/trl/956d742c1b8df9491c8382614e66415fb15bff9c/docs/source/sft_trainer.md) 和 [DPO Trainer 文档](https://raw.githubusercontent.com/huggingface/trl/956d742c1b8df9491c8382614e66415fb15bff9c/docs/source/dpo_trainer.md) 均返回 HTTP 200。该快照的预期 [`ppo_trainer.md`](https://raw.githubusercontent.com/huggingface/trl/956d742c1b8df9491c8382614e66415fb15bff9c/docs/source/ppo_trainer.md) 路径返回 404；这只说明此 revision 的该文档路径不存在，不推断其他版本的接口。
- 本轮检查的 28 个唯一正文外链全部返回 HTTP 200；OWASP 的最终项目页也返回 200。动态 Hugging Face 文档站曾返回代理 503，因此正文改用固定 revision 文档。

## 本地教学代码与公式

- 17 个 Python 代码块在 Python 3.12.9 下逐块独立执行通过；环境为 PyTorch 2.12.0+cpu、NumPy 2.4.6，`torch.cuda.is_available()` 为 False。环境位于 Git 忽略的仓库 `.tmp/llm-book-cpu-check/`，没有安装 CUDA 库。
- 另做 8 项无效输入检查：softmax 空/非有限 logits，assistant mask 空 span、token/label 长度不匹配和越界 span，PPO 空 rollout，置信度越界，checkpoint 基线长度为零。五个相关代码块的拒绝行为符合预期。
- 76 个 `math` 围栏与其他 Markdown 围栏成对；花括号和 `begin/end` 环境检查通过。没有运行外部 MathJax 或 PDF 渲染器。
- CPU toy 输出说明代码对应的局部行为，不证明实际模型的 PPO 训练稳定性、benchmark 得分或线上产品效果。

## 配套同步和未验证项

- `PAPERS.md` 增加 MMLU、MMLU-Pro、GSM8K、MATH、HumanEval、MBPP、IFEval 与 LongBench 论文入口。`EXERCISES.md` 已有模型发布 benchmark 审计及评估统计任务；`INTERVIEW_BANK.md` 已有 benchmark 分类问题；术语表已有 MMLU-Pro 与 benchmark catalog，不重复复制章节内容。
- 第二册目录映射第 51—65 讲正确，下一章从第 66 讲开始；无相对 Markdown 链接。
- 未运行目标 TRL trainer、真实 policy/reference 模型、完整 benchmark harness 或生产服务；未逐篇阅读全部基准论文全文；未使用外部公式渲染器或构建 PDF。
