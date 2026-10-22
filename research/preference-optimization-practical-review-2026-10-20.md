# 偏好优化实战精修记录（2026-10-20）

## 范围

开始从头复读 `book-03-practical-handbook/chapters/05-偏好优化实战.md`，覆盖偏好 pair 与数据审计、Bradley–Terry Reward Model、手写 reward 训练、DPO logprob/mask、TRL DPOTrainer、LoRA/QLoRA 和行为评估。

## 已完成检查与修订

- 明确 5.3.4 的 Reward Model 训练示例使用 CPU toy backbone 和内存 tokenized pair；实际 Transformers backbone 保留为可替换的工程路径。toy 结果只解释梯度和数据流。checkpoint 示例写入 `.tmp/reward_model.pt`，避免落入未创建的仓库 `outputs/` 路径。
- 在 5.4.6 标明 reference 代码片段依赖前文创建的 `ref_model`，避免读者把上下文变量误认为已在本段定义。
- 通过指定代理读取 TRL DPOTrainer 动态文档，确认当前常见调用使用 `processing_class`、`eval_strategy`；日志中的 `logps/chosen` 和 `logps/rejected` 是有效 token 的平均 log-probability，而手写 `sequence_log_probs` 返回 response token log-probability 之和。正文说明长度效应和 beta 尺度不同，loss 数值不可直接比较。
- 当前文档接受标准与 conversational preference 格式，并指出 VLM 图像 token 可能被 `max_length` 截断；正文限定固定长度配置适用于纯文本示例。
- 通过指定代理查询 5 个 arXiv Atom 条目：DPO v3、InstructGPT v1、Learning to Summarize v3、PPO v2、A General Theoretical Paradigm v2。题名和摘要主题与引用一致；结论仍受各论文任务/假设范围约束。
- 按写作规范检查主要定义域和边界，包括空 batch、平票、截断后零 response token、prompt 分组切分、DPO response mask 和评估适用样本集合。
- `PAPERS.md` 已有 DPO、InstructGPT 等入口；新增 QLoRA 来源已随第四章精修同步。

## 本地验证

- 44 个 Python 围栏全部语法编译通过；20 个不需外部模型/运行上下文的片段在仓库 `.tmp/llm-book-cpu-check`（Python 3.12.9、PyTorch 2.12.0+cpu）独立运行通过。
- 组合 torch 片段及 5.3.4 mock Reward Model CPU 训练、评估和 checkpoint 保存闭环通过；保存到 `.tmp/reward_model.pt` 后由验证脚本清理。
- `sequence_log_probs` 的手造均匀 logits 返回预期 `-log(4)`，全 response mask 样本按契约抛出 `ValueError`。
- 46 个 `math` 围栏的花括号和方括号计数平衡；`git diff --check` 通过。
- 未执行依赖 Transformers、Datasets、TRL、PEFT、bitsandbytes 或真实 checkpoint 的块；未下载模型、未安装 CUDA。

## 来源状态与待办

- 本章 16 个唯一来源入口经指定代理最终均 HTTP 200；PyTorch 旧域重定向至 `docs.pytorch.org` 后返回 HTTP 200。TRL 动态文档和 5 个 arXiv 元数据/摘要本轮已核验。
- `EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md` 已覆盖偏好 pair、DPO、Reward Model、评估和失败模式，本轮没有重复添加条目；书册目录映射正确。
- 全文章节复读、来源映射、外链检查、公式结构、CPU 教学代码和配套索引审查完成。未执行真实模型/TRL/PEFT 端到端训练、GPU/QLoRA，也未逐篇阅读全文、附录或完整实验 harness。
