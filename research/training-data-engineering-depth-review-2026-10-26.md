# 第五册第二章：数据工程深度精修研究记录

## 研究范围与问题图谱

- `date`：2026-10-26。
- `chapter`：`book-05-llm-training/chapters/02-数据工程.md`，全文 2.0—2.23。
- `goal`：保留原章中数据生命周期、manifest / lineage、解析、质量、PII / secret、安全、exact / near dedup、评测污染、mixture、合成数据、SFT / preference / process data、代码 / 数学 / 领域、多语言 / 多模态、packing / shard、版本 / 删除、训练收益评估、toy audit、故障诊断和练习；对主流问题补充实际算法、代表研究、实验条件和失效边界。
- `research_questions`：
  1. “高质量数据”由谁定义，过滤器实际学习了什么目标，怎样避免总体分数提升却损失代码、多语种或长尾覆盖？
  2. MinHash / LSH 的召回概率如何从参数推出；不同 near-dedup 算法怎样在内存、召回、顺序依赖和误删之间取舍？
  3. 温度采样与自动 mixture 优化分别解决什么问题，proxy 的收益怎样迁移到目标模型？
  4. benchmark 命中是否等于测分虚高，如何用受限的去污染实验建立因果判断？
  5. synthetic data 的递归替换与累积原始数据有何区别，哪些结论仍受实验条件限制？

## 旧知识点覆盖图

原文 2.0—2.23 的主题均留在本章，没有删除已有知识路径。本轮重点深化如下；其余内容继续按原章节保留：

| 原有主题 | 本轮去向 |
| --- | --- |
| 生命周期、来源契约、manifest / lineage、解析规范化、语言 / 域 / 长度统计 | 原 2.0—2.5 保留；用于界定后续过滤和采样操作依赖的统计口径。 |
| 质量维度、规则 / 分类器 / 模型信号、分桶抽样 | 2.6.5—2.6.7 深化为分类器目标解释、多个公开数据构建方案、过滤收益与覆盖损失。 |
| PII、secret、安全过滤、误删 / 漏删 | 2.7 保留；新增 Dolma 的 PII 替换 / 删除策略作为单一语料实例，并明确不外推。 |
| exact hash、MinHash、SimHash、LSH、重复簇、误删和保留策略 | 2.8—2.9 保留；新增 MinHash 估计式、LSH 候选召回概率和数值表，补 BFF 与 MinHash / Suffix Array 的规模对照。 |
| benchmark contamination、检索、人工复核、新鲜测试集 | 2.10 保留；新增 DCLM 的具体去污染实验及其误报 / 结论边界。 |
| 原始比例、temperature、重复倍数、动态 mixture | 2.11 保留；新增手算比例表及 DoReMi / RegMix 的代表路线比较。 |
| 合成数据、蒸馏、Model Collapse 和验证链 | 2.12 保留；新增“递归替换 vs. 累积真实数据”的实验假设对照。 |
| SFT、偏好、过程监督、代码 / 数学、领域、多语言、多模态、tokenize / packing、shard、删除、消融和 toy audit | 2.13—2.20 保留原主讲位置，本轮不搬迁。 |
| 资料边界和练习 | 2.21—2.23 保留；增加综述、原始论文、代码库和当年预印本入口。 |

## 代表性研究与证据

### 数据选择框架与过滤

- [A Survey on Data Selection for Language Models, arXiv:2402.16827 v3 (2024-08-02)](https://arxiv.org/html/2402.16827)：读了统一 selection framework、distribution matching vs. diversification、pretraining 章节中的语言 / 启发式 / 质量 / 域选择 / 去重 / mixture，以及污染讨论。综述把选择函数拆成 utility function 与 selection mechanism，并指出目标可包括模型表现、成本、评估完整性和 bias / toxicity；这支持正文把“quality score”写成目标条件分数，而非全局真值。
- [The FineWeb Datasets, arXiv:2406.17557 v2 (2024-10-31)](https://arxiv.org/html/2406.17557)：读了 Common Crawl 过滤、MinHash、跨快照去重、C4 规则、FineWeb-Edu 和实验配置。FineWeb 用 5-gram、112 MinHash、14 bands × 8 hashes；LSH 概率为 `1-(1-s^8)^14`，论文报告 `s=0.70/0.75/0.80/0.85` 时召回概率约为 `56%/77%/92%/98.8%`。论文在 1.71B GPT-2 tokenizer 模型、每数据集随机抽样 350B token 的设置下比较公开语料；它不是所有规模和目标上的固定排名。
- FineWeb 的全局跨 snapshot MinHash 实验移除早期 crawl 中最高约 90% 内容；整体得到约 4T tokens 的结果在抽样 350B 训练时提升有限。作者拆出 2013-48 数据后发现，被保留的旧页（约 31B）质量低于被删除内容（筛后约 171B），并据此改成逐 snapshot 去重，得到约 20T tokens，在作者评测中匹配 RefinedWeb。该结果证明去重顺序会改变“哪类数据留下”，不是“去重一定增益”的泛化定律。
- FineWeb 自定义启发式过滤的另一组 28B-token 消融中，三项规则合用删除约 22% tokens，aggregate score 约升 1%；不要与 350B-token 跨数据集比较混为一个实验。FineWeb-Edu 由 Llama-3-70B-Instruct 给 46 万网页打教育分，embedding 冻结并训练线性回归器，再扩展为 1.3T tokens；“教育价值”是目标域标签，不是通用质量。
- [Dolma, arXiv:2402.00159](https://arxiv.org/html/2402.00159)：读了设计目标、开放性、web / code / social 流程、过滤、PII、去重和 data ablation。web 语料约 2.28T tokens，整体英文混合语料约 3T。URL 去重先移除 53.2% 文档，文档级 exact dedup 再移除 URL 去重后 14.9% 文档，段落 dedup 过滤 18.7% 段落；URL / 文档阶段省处理成本，段落级放后面避免提前破坏上下文。Dolma 的 data-ablation models 最多训练 150B tokens；其过滤效果不能自动外推到大规模最终预训练。
- Dolma 的一例 PII 规则：不超过 5 个命中 span 时替换为类型 token，超过 5 个则删除整篇；论文报告分别约影响 0.02% 与 0.001% 文档，在其评估中没有可测性能影响。极小受影响比例使这个结果不支持“PII 操作无损”的普遍结论。
- [DataComp-LM, arXiv:2406.11794](https://arxiv.org/html/2406.11794)：读了 DCLM-Pool、固定训练 / 评估协议、抽取、去重、fastText 质量过滤、mixture、decontamination 和 limitation。DCLM-Pool 为 Common Crawl 抽取的 240T GPT-NeoX tokens；benchmark 以标准化模型规模、配方和 53 个任务控制变量，目标是提高数据实验可比性。论文在 1B-1x 设置比较 PageRank、SemDedup、BGE 特征分类、AskLLM、PPL、top-k logits 与 fastText；fastText 的正例路线包括 OpenHermes-2.5 + ELI5，负例来自 RefinedWeb reproduction 的随机样本。原论文报告的 Core / Extended 是 v1 聚合；[官方 DCLM README](https://github.com/mlfoundations/dclm) 明示 2025-09-05 起修正 baseline bugs，旧新分数不直接可比。正文明确保留这一版本边界。
- DCLM 的公开 limitations 写到：消融受算力限制，多数维度逐项测试，规模最高 7B，缺少充分 run-to-run variation，主要使用 GPT-NeoX tokenizer，第一版对语言理解较强但 code / math 较弱；这支持正文把它当作 benchmark 起点而不是“行业最终答案”。
- [Nemotron-CC, arXiv:2412.02595 v2 (2025-05-30)](https://arxiv.org/html/2412.02595)：读了抽取、分类器、质量分桶、合成改写、数据 blend 和训练实验。作者发布的 6.3T token 集含 4.4T globally deduplicated real tokens 和 1.9T synthetic tokens；在作者的 8B、15T 训练评估中报告多项能力超过 Llama 3.1 8B。这个方向回应“高质量过滤可能耗尽长 horizon 唯一 token”，但模型、配方与比较均是作者设置，不能当作独立复现；改写质量和覆盖保留还需逐任务检查。
- [Removing Noise, not Finding Gold, arXiv:2510.00866 v4 (2026-06-24)](https://arxiv.org/html/2510.00866)：读了 CQF 定义、下游评估、HQ-set modeling 差异与作者建议。论文显示分类器过滤可改善特定 downstream evaluation，却不保证更好地建模整个 reference high-quality set；作者的解释是 classifier 也隐式筛选 reference set 内的子分布。因此正文明确说分数对齐某个参考集和任务，而非数据普适真值。它是新近预印本分析，未声称是被所有语料采用的生产标准。
- [Leverage Score Sampling, arXiv:2609.32484 v1 (2026-09-26)](https://arxiv.org/html/2609.32484)：读了 Leverage / DiSF 方法动机、CC 与 StarCoderData 设置、代码和 efficiency results。作者报告在其 Common Crawl 筛选上最高 72× 相对 DiSF 的 speedup、Vendi score 提升 9.2%、7 个 downstream task 最多提升 1.31%；在代码设置中质量筛选和多样性选择存在任务 tradeoff，DCLM-fastText 的筛选代码 recall 不足。该结果为 v1 预印本，数据、selection ratio、embedding 和评估均限制其外推。

### 去重算法、污染和评估

- [Deduplicating Training Data Makes Language Models Better, arXiv:2107.06499 v2 (ACL 2022)](https://arxiv.org/abs/2107.06499)：摘要报告数据集存在 near duplicates 和长重复子串；论文作者发现去重后模型生成记忆文本频率约低十倍、在相同或更好准确率下需要更少训练步数，同时超过 4% 的标准验证样本和训练数据有 overlap。文中结论绑于论文数据、模型和评估，正文没有把它写成通用固定比例。
- [DataComp-LM BFF / MinHash + Suffix Array 对照](https://arxiv.org/html/2406.11794)：BFF 以 corpus-stream membership 的段落 / 文档 n-gram 比例做判断；和 document-pair MinHash 的 duplicate definition 不同。7B-2x 表格中 BFF(13-token minimum, 10 shards) 的 MMLU / Core 是 44.3 / 45.3，projected yield 3.8T；MinHash + SA 为 44.4 / 45.5，projected yield 3.2T。5-token minimum 虽保住 Core，却使 MMLU 明显下降；作者最终采用 13-token minimum。数字均为 DCLM 受控实验，yield 是以 1/10 Common Crawl pool 结果投射。
- DCLM contamination 例子：基于题干末句及至少一个选项识别 MMLU，题目与选项查找 HellaSwag；作者明确看到 MMLU 规则误报较多。7B-2x 去掉检测内容后，论文版 MMLU 为 51.8→52.7，HellaSwag 为 77.9→78.4。它排除了“检测到这些重叠就是该分数优势原因”的简单解释，但不能证明无 contamination 或评估污染无害。
- [Contamination Survey (2024)](https://aclanthology.org/2024.findings-acl.951/) 与 [Static to Dynamic Evaluation Survey (2025)](https://aclanthology.org/2025.emnlp-main.511/) 用于污染类型 / 检测 / remediation 和 static / dynamic benchmark 的概念地图。动态评测减少题目长期公开产生的重用，但本身也需要评估任务、数据版本与公平条件；没有把动态 benchmark 写成完全解决污染。

### 数据配比与合成数据

- [DoReMi, arXiv:2305.10429 v4 (2023-11-21)](https://arxiv.org/html/2305.10429)：Group DRO 最大化各域相对 reference model 的 excess loss，通过小 proxy 估计 domain weights，再用于 full-size training。The Pile 实验中 280M proxy → 8B main model，五项 one-shot task 平均提升 6.5 个百分点，75K 步达到 baseline 200K 步表现；proxy weights 搜索约占大模型训练 compute 的 8%。作者指出 domain 按数据 provenance 粗分，细粒度 domain 可能改善；权重跨任务与模型设置仍需验证。
- [RegMix, arXiv:2407.01492 v2 (2025-01-23)](https://arxiv.org/html/2407.01492)：对许多小模型的 mixture 结果拟合回归，假设 mixture rank 在 model / token scale 间相对稳定。作者训练 512 个 1M 参数模型、每个 1B tokens，估算 FLOPs 约等于一个 1B 模型；用模型预测 1B/25B token 的 mixture 后，在 64 个目标配比候选里选优。作者还报告扩到 7B/100B；rank invariance、domain definition、tokenizer、任务指标及有限 benchmark 是限制。与 DoReMi 是不同数据 / 预算 / 评测，正文只对照机制，不对比绝对提升数字。
- DataComp-LM mixing experiment 在 1B-1x 设置将 67% CC + 33% Wikipedia / Books / StackExchange / arXiv / GitHub 的配比用于多个底库：补充数据提升 C4、RedPajama-CC 和 RefinedWeb，但降低 DCLM-baseline 的平均 Core / Extended。该结果支持“互补性依赖 base corpus”，不支持固定域配比。
- [The Curse of Recursion, arXiv:2305.17493](https://arxiv.org/abs/2305.17493)：多模态生成模型工作中研究递归使用 model output 并报告分布尾部消失；[Is Model Collapse Inevitable?, arXiv:2404.01413](https://arxiv.org/abs/2404.01413) 对比 replace 与 accumulate，报告在其语言模型和其他生成模型的累积真实数据设置中没有观察到相同 collapse。两组前提并不矛盾：前者着重递归替换，后者保留原数据并累积；不能推广到任意合成来源、比例或筛选规则。
- [Nemotron-CC](https://arxiv.org/abs/2412.02595) 展示分类器质量标注、真实网页改写及合成数据按 provenance 合并的一个工程方案。作者报告值不等于已验证的通用 generation verifier。

## 官方代码和工程材料

- [Hugging Face DataTrove](https://github.com/huggingface/datatrove)：检查了 FineWeb 示例；代码体现多阶段过滤、每个 crawl 独立 MinHash、签名配置和集群执行依赖。该链接为会变的 `main` 分支，正文只作为源码入口，不把当前 HEAD 字段写成固定长期事实。
- [DataComp-LM repository](https://github.com/mlfoundations/dclm)：README 说明当前 score aggregation 修订、fastText filtering、Ray 本地过滤与 Rust BFF 的分层；其当前实现用于定位仓库能力，论文性能数字仍来自原论文对应固定训练协议。
- [Dolma Toolkit](https://github.com/allenai/dolma)：README / docs 表明提供可扩展数据处理、C4 / Gopher taggers、Rust Bloom-filter dedup 和云存储支持。运行版在本轮未安装或执行。

## 正文变化

1. 2.6.5 说明 classifier-based quality score 从正例参考集学到“像什么”，并明确参考集、目标和阈值决定留下的数据；补 DCLM / FineWeb-Edu 的实施和实验边界。
2. 2.6.6 加入过滤目标与语言 / 领域覆盖的张力；将 2026 两篇研究作为预印本证据，区分作者观察和本章的工程建议。
3. 2.6.7 用 C4、Dolma、FineWeb、DCLM、Nemotron-CC 对照语料目标、机制、结果与边界；明确禁止跨论文裸分排名。
4. 2.7.2 增加 Dolma PII 阈值和影响比例，并明确影响极小不代表一般处理无损。
5. 2.9.2 推导 MinHash estimator 与 LSH 候选召回概率，给出 FineWeb 参数下的数值表；2.9.5 加 BFF 操作流程、误报和短 n-gram 造成的能力损失案例。
6. 2.10.4 加 DCLM 的题目/选项 overlap 扫描、去污染前后分数与误报边界。
7. 2.11.5 加 `100:10:1` 的自然比例 vs. `alpha=0.5` 温度采样和 200-token-unit exposure 推演；比较 DoReMi 的 Group DRO proxy 与 RegMix 的 regression proxy，再用 DCLM mixture ablation 说明互补性依赖底库。
8. 2.12.4 对照“每代替换数据”与“保留原始数据并累积合成样本”，解释模型坍塌论文看似相冲突的实验前提。
9. 2.21 补综述、原论文、预印本与官方 GitHub pipeline 入口。原有 SFT / preference、code / math、multilingual / multimodal、packing / shard、lineage / deletion、评估、toy audit、故障和练习均保留原章节。

## 数值与证据边界

- 手算 LSH 公式使用 `b=14,r=8`：`s=.70` 得 `1-(1-.7^8)^14≈0.557`，`.75≈0.772`，`.80≈0.923`，`.85≈0.988`，与 FineWeb appendix 四舍五入数据一致。
- 温度示例 `n=(100,10,1), alpha=.5`：归一化权重约 `(0.706,0.223,0.071)`；训练预算 200 后约 `(141.2,44.6,14.2)`，对应重复倍数约 `(1.41,4.46,14.2)`。
- 未重新执行或修改 2.19 toy Python pipeline；未下载数据、运行去重器、训练 proxy / LM、复现任一论文或跑 GPU / 分布式作业。
- 未调用外部公式渲染器；公式结构、变量和手算结果做静态核对。
- 论文公开训练设置、论文版 benchmark 分数、仓库当前版本和本地实验严格分账。DCLM Core / Extended 旧分数尤其不能与仓库 v2 聚合直接比较。

## 本轮来源入口

- Survey：[A Survey on Data Selection for Language Models](https://arxiv.org/abs/2402.16827)
- Corpora：[C4 / T5](https://arxiv.org/abs/1910.10683)、[Dolma](https://arxiv.org/abs/2402.00159)、[FineWeb](https://arxiv.org/abs/2406.17557)、[DataComp-LM](https://arxiv.org/abs/2406.11794)、[Nemotron-CC](https://arxiv.org/abs/2412.02595)
- Quality / diversity：[Removing Noise, not Finding Gold](https://arxiv.org/abs/2510.00866)、[Leverage Score Sampling](https://arxiv.org/abs/2609.32484)
- Dedup / contamination：[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)、[Unveiling the Spectrum of Data Contamination](https://aclanthology.org/2024.findings-acl.951/)、[Static to Dynamic Contamination Survey](https://aclanthology.org/2025.emnlp-main.511/)
- Mixture / synthetic data：[DoReMi](https://arxiv.org/abs/2305.10429)、[RegMix](https://arxiv.org/abs/2407.01492)、[The Curse of Recursion](https://arxiv.org/abs/2305.17493)、[Is Model Collapse Inevitable?](https://arxiv.org/abs/2404.01413)
- Source code：[DataTrove](https://github.com/huggingface/datatrove)、[DCLM](https://github.com/mlfoundations/dclm)、[Dolma](https://github.com/allenai/dolma)
