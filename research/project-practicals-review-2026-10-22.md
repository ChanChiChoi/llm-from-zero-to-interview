# 第三册第九章项目实作精修证据

- 日期：2026-10-22；范围：`book-03-practical-handbook/chapters/09-简历项目实战.md` 全文，9.1—9.7。
- `evidence_level`：paper、official、local-demo；项目目录与部分决策流程是教学设计。
- `lifecycle`：active；未运行真实模型训练、向量检索服务或 GPU serving。
- 使用维护者指定代理。链接可达、元数据、原文核对与本地验证分开记录。

## 来源核验

以下十九个正文唯一入口最终均 HTTP 200。动态 main/latest 文档是本日快照，不是固定依赖或未来接口保证。

| 资料 | 核验内容与边界 |
| --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Atom 题名一致，v7；attention 原始入口，未复现实验 |
| [GPT-2 报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | PDF 入口可达，未逐页核对 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 仓库入口可达，未固定 revision 或执行训练 |
| [Transformers chat template](https://huggingface.co/docs/transformers/main/en/chat_templating) | 读取训练/生成提示区别；main 文档不等于本地安装版本 |
| [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) | 官方概念入口可达，不作为本地微调结果 |
| [LoRA 论文](https://arxiv.org/abs/2106.09685) | Atom 题名一致，v2；低秩更新定义 |
| [TRL SFT](https://huggingface.co/docs/trl/main/en/sft_trainer) | 读取 assistant_only_loss 与模板 generation span 条件；最后一轮 mask 不等价于全部 assistant span |
| [DPO 论文](https://arxiv.org/abs/2305.18290) | Atom 题名一致，v3；另读取该版本 HTML 的目标与推导相关内容 |
| [TRL DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer) | 读取 policy/reference 相对 log probability 与 loss 定义；未运行 trainer |
| [Transformers generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) | 官方生成入口可达，未运行模型 API |
| [vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | 读取 Limits：主要减少 prefill，不减少新 token decode；旧 .html 入口改为当前路径 |
| [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/) | 原型接口入口可达，未启动真实服务 |
| [RAG](https://arxiv.org/abs/2005.11401) | Atom 题名一致，v4；检索增强原始定义入口 |
| [FAISS](https://arxiv.org/abs/1702.08734) | Atom 题名一致，v1；向量搜索入口，未评测目标语料 |
| [Sentence Transformers retrieve-rerank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) | 官方检索/重排入口可达，未下载模型 |
| [Chain-of-Thought](https://arxiv.org/abs/2201.11903) | Atom 题名一致，v6；非所有模型通用效果保证 |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | Atom 题名一致，v4；多路径与聚合定义 |
| [GSM8K](https://arxiv.org/abs/2110.14168) | Atom 题名 Training Verifiers to Solve Math Word Problems，v2；数据与 verifier 入口 |
| [HumanEval/pass@k](https://arxiv.org/abs/2107.03374) | Atom 题名 Evaluating Large Language Models Trained on Code，v2；另读取该版 HTML 的无偏估计公式与稳定计算说明 |

九个正文 arXiv ID 均由 Atom 核准题名与引用主题。另读取相邻推理章节的 PagedAttention 元数据，不冒充本章新增实验。DPO 与 pass@k 原文读取不等于完整论文复现。数学 toy 将通过单元测试替换为任务答案判定，是教学指标迁移，不是 HumanEval 测量。

## 修订依据与结果

1. miniGPT 实作明确为序列审计，实际模型训练回指第三章；统一零基词表公式、温度和 shape 定义域，检查整个概率分布，而非只检查目标概率。
2. SFT 审计仅监督最后一轮 assistant，并保留同长度 labels 供内部 shift；拒绝伪造教学分隔符和不支持的角色。真实多轮 mask 需按模型模板重新验证。
3. DPO 拒绝正 log probability、非法 beta 与非有限输入，报告重复 ID；固定 log probability 的 beta 对比不当作训练收益。
4. 压测增加全部请求结束时间，防止慢失败被排除；全失败正窗口中成功吞吐为零，成功延迟不适用。占用达到容量时不能新增请求，合法空输出需真实系统单独建模。
5. RAG 空 ACL 默认拒绝，无候选返回空集合；RRF 拒绝重复 ID，普通稀疏 dict 余弦使用缺失项零值。引用短语命中不当作语义支持或完整性。
6. 数学评估保留单位，严格最终答案行，不取中间数字；Decimal 避免六位小数舍入误判；解析失败不投票但保留采样计数。错误类型是候选归因，增加工具与最终答案一致性字段。
7. 补充题目文本、前后章节链接、项目交付目录、正常/边界练习与限定实际贡献的面试表达；不填造真实模型成绩。

## 本地验证

- Python 3.12.9；七段标准库代码分别语法编译、独立执行通过；未新增训练或 CUDA 依赖。
- 字符词表 10，今天编码 `[3,5]`；四个窗口，目标概率 0.2 时 NLL 1.609438，0.8 时 0.223144。
- SFT 两样本有效比例 0.3139；三个 LoRA 模块参数合计 448。
- DPO 长度差 `[4,9]`，beta=0.05 平均 loss 0.664830。
- 服务 P50/P95 666/2205 ms，QPS 1.3363、输出吞吐 203.1180 token/s、KV 384 MiB；晚失败结束于 10000 ms 时成功 QPS=0.3。
- RAG employee 可见两个 chunk，返回 travel_0，Recall/reciprocal rank 均为 1；不是模型检索效果。
- 数学单次准确率 0.8，合成候选投票准确率 1.0、pass@3=1.0；这不是策略实测对照。
- 边界验证覆盖未知 token/非法 ID、窗口无目标、概率不归一/NaN/目标为零、SFT 无目标/分隔符、非法 beta/NaN、空或非法 trace、全失败/晚失败、容量等值、零向量、空 ACL/无证据、重复 RRF、无 gold、错误单位/百分比/数字舍入/抽取失败、全失败投票、非法 pass@k 和重复问题 ID。
- 83 个 math 围栏结构平衡；无 operatorname、尖括号下标、text 下划线、旧式 $$；目录与新增相邻正文链接目标存在。
- `git diff --check` 通过；未运行外部数学渲染器或 PDF 构建，后者已有 Pandoc 缺失限制。

配套 PROJECTS.md 与 EXERCISES.md 同步具体边界验收；论文索引、题库和术语已有相关主题，不复制摘要。所有本地数字只支持本章合成数据和当前代码行为。
