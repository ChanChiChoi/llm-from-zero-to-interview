# 第十一章：Dataset Versioning 与 Governance

前面章节讲了数据采集、清洗、去重、配比、专项数据、合成数据、偏好安全数据、多模态数据和数据价值评估。本章讨论把这些能力组织成长期可运行系统所必须的底座：dataset versioning 与 governance。

如果没有数据版本和治理，团队可能训练出一个模型，却说不清楚它到底用了哪些数据、哪些规则、哪些过滤器、哪些来源、哪些授权、哪些删除请求是否生效。模型出了问题，也无法追溯到数据版本；想复现实验，也找不到当时的数据快照。

大模型时代，数据治理不再是合规部门的附属工作，而是训练工程的一部分。没有治理的数据系统，规模越大，风险越大。

本章重点：数据版本、元数据、可追溯性、权限、审计、删除请求和数据治理。

合规边界：本章讨论数据治理、权限控制、审计、删除机制和合规文档，不提供绕过访问控制、规避审计、保留应删除数据或滥用个人信息的方法。

## 0. 本章范围与资料

数据版本和治理解决的是同一个可复查问题：一次训练究竟使用了哪些对象、哪些处理规则、哪些权限和哪些责任判断。版本号只解决“这是什么时候的快照”，治理还要解决“谁可以看、为什么可以用、如何删除、出了问题如何证明处理过”。

本章沿着不可变快照、manifest、lineage、schema 演进、权限分级、删除请求、datasheet、model card 输入、审计指标和训练复现记录展开。Datasheets、Model Cards、W3C PROV、DVC、OpenLineage、Hugging Face Dataset Cards、NIST AI RMF 和 GDPR Article 17 作为公开资料入口，放在章末区分研究框架、工程文档和法律原文。

一条治理链路可以写成：

~~~text
来源登记 -> 处理流水线 -> 不可变快照 -> manifest -> lineage -> 权限与删除 -> 审计记录 -> 训练日志 -> 发布文档
~~~

本章不提供规避权限、保留应删除数据、绕过审计或滥用个人信息的方法。删除请求和隐私治理的具体法律义务取决于适用地区、主体关系和数据用途，正文只讨论工程机制、证据和审计口径。

---

## 1. 先建立直觉：为什么数据需要版本？

代码工程里，每次改代码都会有 git commit。你可以知道某个 bug 是哪个 commit 引入的，也可以回退到旧版本。数据工程也需要类似能力。

假设你训练了两个模型：Model A 和 Model B。B 在数学上更好，但安全误拒变多。你需要知道：

1. B 是否用了新的数学数据？
2. 安全数据比例是否变化？
3. 清洗规则是否改过？
4. 去重阈值是否不同？
5. 是否新增了某个低质量合成数据源？
6. 某些用户删除请求是否在 B 的训练集中生效？

如果数据没有版本，这些问题都无法回答。

所以 dataset versioning 的核心目标是：让每一次训练都能追溯到确定的数据快照和处理流程。

### 1.1 一个具体的复现实验

假设数据团队每天把最新数据写入 `train/latest/`，训练脚本也永远读取这个目录。周一模型 A 使用了 100 万条样本，周三清洗规则调整后，目录里变成 120 万条样本。研究员在周五重新运行周一的训练命令，代码、超参数和随机种子都没有变化，结果却对不上。问题不在随机数，而在“周一的数据”从来没有被定义成一个可以再次取得的对象。

如果周一的目录在训练开始时生成 `data-v1`，并把 shard、hash、样本统计和处理配置写入 manifest，周三的新数据则生成 `data-v2`，两次运行就可以明确写成“模型 A 使用 `data-v1`，模型 B 使用 `data-v2`”。即使对象存储中的文件后来被迁移，训练日志仍然可以通过版本和 checksum 找回当时的内容。

从初学者角度看，版本号像文件的存档标签；从工程角度看，它更像一次训练可以引用的输入契约。这个契约至少要固定三层：数据内容本身、改变数据的处理过程，以及决定哪些对象进入训练的选择规则。只有把三层绑定，才能解释“同一批原文为什么在不同版本中得到不同样本”。

还要注意一个边界：版本化只能回答“使用了什么”，不能自动回答“这些内容是否高质量、是否有权使用”。一个 hash 完整、可下载、可复现的数据版本，仍然可能包含错误标签、隐私信息或不适合当前用途的授权。版本系统是治理的证据底座，不是质量和合规的替代品。

---

## 2. 来龙去脉：从数据集文档到基础模型治理

传统机器学习时代，数据集通常比较小，研究者可以在论文中简单描述数据来源、样本数、划分方式和标注方法。但随着模型进入高风险应用，大家逐渐意识到数据集本身需要标准化文档。

Datasheets for Datasets 提出每个数据集都应配套 datasheet，记录动机、组成、采集过程、推荐用途等信息，提升透明度和责任意识。

Model Cards 则强调模型发布时要说明模型用途、评估结果、适用边界和风险。虽然 model card 面向模型，但它离不开数据治理：模型的行为风险很多来自训练数据。

基础模型进一步放大了问题。Foundation model 训练在宽泛数据上，随后被大量下游应用复用。上游数据缺陷会被下游系统继承和放大。因此，数据版本、来源、权限、风险和文档不再是“最好有”，而是模型可控、可审计、可复现的基础。

这条发展脉络中有一个容易被忽略的变化：早期数据文档主要帮助研究者理解一个公开数据集，基础模型时代的治理则要处理“同一份数据会被许多训练运行和产品复用”的传播关系。一个标签错误在单个实验中可能只是噪声，在基础模型中却可能被蒸馏到多个下游模型；一个来源权限判断的遗漏，也可能随着 checkpoint、embedding 和评测服务被复制。

Datasheets 和 Model Cards 提供的是提问框架，而不是自动生成的证明。Datasheet 促使团队说明数据为什么存在、如何得到、适合什么用途；model card 促使发布者说明模型能做什么、不能做什么、如何评估。工程系统要做的是把这些问题连接到具体版本、统计和运行记录。比如“训练数据包含多语言网页”只是概述，只有语言分布、采集范围、过滤版本和评测集引用齐全后，读者才有机会判断这句话的边界。

对于专家读者，真正的难点在于治理对象的范围。治理既不能退化成只给文件贴标签，也不能试图把所有社会和法律判断都编码成一个字段。合理的做法是把可验证的事实、组织决策、例外和不确定性分开记录，并让每一层都能回到来源或责任人。这样，文档才不会在规模扩大后变成无法维护的宣传页。

---

## 3. Dataset versioning 到底版本化什么？

很多人以为版本化就是给数据文件起个版本号。实际远不止如此。

需要版本化的内容包括：

1. 原始数据快照。
2. 数据来源列表。
3. 采集时间和采集方式。
4. 解析器版本。
5. 清洗规则和阈值。
6. 质量评分模型版本。
7. PII 和安全过滤器版本。
8. 去重算法和参数。
9. benchmark contamination 检测版本。
10. 数据 schema。
11. 数据分桶和标签。
12. data mixture 配置。
13. 采样权重。
14. 删除请求和处理状态。
15. license 和授权状态。

一个训练数据版本不是一个文件，而是一组数据、代码、配置、元数据和审计记录的组合。

可以把这些对象分成三类。第一类是内容对象，例如原始文件、清洗后的样本和 shard；第二类是变换对象，例如解析器、过滤器、去重模型、采样器和配置；第三类是证据对象，例如 manifest、运行日志、授权记录、删除工单和质量报告。只有第一类被 hash，而后二、三类没有版本，仍然不能复现训练。

一个常见反例是只给清洗后的数据打版本号。团队后来修改了 PII 过滤器，但继续把结果写到同一个数据版本目录；模型行为发生变化时，大家能看到样本文件的 hash，却不知道过滤器发生了变化。另一个反例是只版本化代码，不版本化采样配置：数据集合相同，mixture 权重改变，低资源语言或安全数据在训练中的有效曝光量仍然会变化。

因此，版本差异应当既有内容层，也有语义层。内容层比较新增、删除和修改的对象；语义层比较 schema、处理规则、授权状态、数据配比和风险分层。两个版本即使共享大量 shard，也可能因为选择规则不同而代表不同的训练输入。发布变更说明时，应明确哪些 shard 被复用、哪些处理器变化、哪些统计发生偏移，以及这些变化是否会影响下游模型。

---

## 4. 数据 lineage：数据从哪里来，到哪里去

Lineage 指数据血缘，也就是数据从源头到最终训练样本的完整路径。

例如一条训练样本可能经历：

1. 从某网页采集。
2. HTML 解析。
3. 正文抽取。
4. 语言识别。
5. 质量评分。
6. PII 过滤。
7. 去重聚类。
8. 分配到中文技术文档数据池。
9. 按某个采样权重进入训练 shard。

Lineage 要回答：这条样本来自哪里，经过哪些处理，为什么被保留，最后用于哪个模型训练。

没有 lineage，数据问题无法追责；有 lineage，模型异常可以追溯到具体来源、规则或数据版本。

更准确地说，lineage 是一个带方向的图，而不是一列来源字符串。图中的节点可以是原始对象、样本、处理任务、shard、数据版本、训练运行和模型；边表示“由某对象产生”“被某任务读取”“被某版本收录”或“被某运行使用”。当问题从模型回溯到来源时沿图反向走，从一个来源寻找受影响的模型时沿图正向走。

以一条技术文章为例，系统不应只记录 `source=web_blog`。它还应记录原始响应或归档对象的 hash、抓取时间、解析器版本、正文抽取结果、语言识别结果、质量评分、PII 扫描结果、去重聚类和最终 shard。若质量模型在第二天更新，新的处理事件要引用新的模型版本，而不是覆盖旧事件。这样才能解释同一 URL 在两个数据版本中为何得到不同结果。

血缘粒度也需要取舍。所有原始字节都保存会带来存储、索引和隐私成本；只保存数据集级关系又无法处理单个来源的删除请求。工程上常用分层策略：版本和 shard 级关系保证宏观追踪，风险较高或可能收到撤回请求的对象保留样本级关系，低风险大规模来源可以保存可重建的聚合和抽样证据。粒度选择必须和用途、风险以及保留期限一起设计。

对专家而言，血缘记录的关键不是图形化展示，而是事件的可验证性。一个处理事件至少需要输入引用、输出引用、处理器版本、配置摘要、运行身份、时间和状态；如果任务失败或部分重试，也要区分已完成对象和未完成对象。没有这些字段，图看起来连通，实际上无法判断结果是否被重写、是否漏处理或是否来自未经授权的输入。

---

## 5. 元数据 schema

大模型数据治理依赖元数据。每条样本至少应包含一些基础字段。

常见字段包括：

1. sample_id。
2. source_id。
3. source_url 或来源描述。
4. collection_time。
5. license。
6. language。
7. domain。
8. modality。
9. token_count。
10. quality_score。
11. safety_label。
12. pii_label。
13. dedup_cluster_id。
14. contamination_flag。
15. processing_version。
16. dataset_version。
17. access_level。
18. deletion_status。

这些字段不是为了好看，而是服务训练、审计、删除、配比和复现。

例如，如果要删除某个来源的数据，需要 source_id；如果要分析低资源语言表现，需要 language；如果要隔离 benchmark 污染，需要 contamination_flag；如果要重现实验，需要 processing_version 和 dataset_version。

字段设计首先要解决“未知”和“否定”的区别。`pii_label=false` 表示已经执行了检测并得到否定结果，字段缺失则只表示没有证据；`license=unknown` 也不能等同于“没有限制”。如果下游筛选器把缺失值当成安全值，治理系统会在数据规模扩大时静默放行风险对象。建议为关键字段定义允许值、缺失语义、生产者、更新时间和校验规则，并在 schema 中保留未知状态。

字段还要有稳定的语义边界。`quality_score` 是某个模型在某个版本、某个分数范围和某个数据切分上产生的结果；它不是脱离上下文的客观质量。若评分模型更新，应生成新的 `quality_score_version` 或处理版本，并在统计报告中说明分数不可直接和旧版本比较。类似地，`access_level` 描述当前访问策略，不代表原始内容天然公开；`deletion_status` 描述处理状态，也不应被解释成法律结论。

### 5.1 关键公式与治理指标

一个数据版本可以抽象为：

~~~math
V_t = (D_t, M_t, P_t, S_t, A_t)
~~~

D_t 是数据快照，M_t 是 manifest，P_t 是处理 pipeline 和配置版本，S_t 是 schema 版本，A_t 是审计记录。训练日志必须引用这个整体，而不是只记录一个会被原地修改的数据目录名。

Manifest 可以写成 shard 清单：

~~~math
M_t = {(h_j, n_j, b_j, c_j) for j = 1..K_t}
~~~

h_j 是 shard 的内容 hash 或 checksum，n_j 是样本数，b_j 是 token 数或字节数，c_j 是来源、语言、license、质量和风险统计。checksum 只能证明当前字节内容与记录一致，不能证明内容本身合法或质量合格。

样本级 lineage 可以写成有序处理路径：

~~~math
L_i = (s_i, o_i, p_i1, p_i2, ..., p_im, v_i)
~~~

s_i 是 source id，o_i 是原始对象 hash，p_ij 是第 j 个处理步骤，例如 parse、quality、PII scan、dedup 和 contamination scan，v_i 是样本进入的数据版本。lineage 同时支持“从样本追来源”和“从来源追模型版本”。

lineage 覆盖率可以写成：

~~~math
C_lineage = complete_lineage_samples / total_samples
~~~

license 记录完整率可以写成：

~~~math
C_license = samples_with_license_record / total_samples
~~~

这两个分数只衡量记录是否存在，不代表记录内容一定正确。抽样时还要检查来源、许可文本、处理动作和实际样本是否相互匹配。

删除请求的执行时延可以写成：

~~~math
T_delete_r = t_closed_r - t_received_r
~~~

如果请求 r 对应的来源、hash、用户或 license 状态无法映射到版本和 shard，就不能声称删除机制可靠。还要记录待处理、已隔离、已从未来版本排除和已完成回放等状态，避免把“从当前目录消失”误当成删除完成。

访问阻断率可以写成：

~~~math
R_blocked = denied_or_invalid_attempts / all_high_risk_attempts
~~~

这个指标不是越低越好：高风险数据的访问尝试应该被阻断，而正常授权访问不应被无故拒绝。因此要同时报告授权成功率、未授权阻断率和审计日志完整率。

数据文档完整率可以写成：

~~~math
C_doc = completed_required_sections / required_sections
~~~

最终不要把这些指标压成一个治理总开关，而应保留并列状态向量：

~~~math
C_gov = (M_manifest, L_lineage, L_license, A_access, D_delete, D_doc)
~~~

每个分量都应有责任人、版本、分母、例外记录和下一步动作。一个数据版本可以在 manifest 完整但删除链路不完整，也可以 lineage 完整但 license 尚未确认；并列状态比一个 True/False 更能支持处置。

公式中的分母尤其重要。假设一个版本有 100 万条样本，其中 99 万条来自低风险公开来源，1 万条来自需要逐条核验的高风险来源。总体 `C_license=0.99` 看起来很高，但如果那 1 万条全部没有授权记录，发布判断仍然不能只看总体比例。实践中应同时报告总体值、按来源/语言/风险桶分层的值，以及高风险桶的最差值。

这些指标也不能只在训练结束后计算。训练入口需要在读取 manifest 时核验快照 hash、schema 版本和删除状态；数据流水线在生成 shard 时记录 lineage 和 license；发布流程再把训练实际引用与已批准版本进行比对。指标一旦脱离这些状态转换，就会变成事后填表，无法阻止错误数据继续流入。

---

## 6. 数据快照和不可变性

训练数据版本最好是不可变快照。也就是说，一旦某个版本用于训练，就不应该原地修改。

如果发现错误，应创建新版本，而不是偷偷改旧版本。否则后续无法解释为什么同一个版本训练结果不同。

不可变快照可以通过以下方式实现：

1. 内容 hash。
2. 文件 manifest。
3. shard checksum。
4. 对象存储版本。
5. 数据库快照。
6. 配置文件版本。

训练日志应记录使用的数据 manifest、shard 列表、采样配置和过滤器版本。

不可变性有两个层次。内容不可变要求已经发布的对象不会被原地覆盖；引用不可变要求训练运行保存的是明确的版本 ID 或 digest，而不是会随时间解析到不同内容的 `latest` 标签。很多对象存储提供版本控制，但打开版本控制并不等于应用已经使用了不可变引用：如果脚本仍然读取一个可移动的路径，复现风险依旧存在。

不可变也不等于永远不能删除。删除请求可能要求从后续使用路径、缓存或派生对象中移除内容，因此系统要把“历史证据保留”和“内容访问/继续使用”分开建模。在适用的保留政策下，审计记录可以只保存必要的 hash、状态和责任信息，而不是继续保存不应访问的正文。具体保留期限和删除义务要由项目适用的政策与法律判断，工程设计不能用“不可变”作为拒绝处理变更的理由。

如果必须修正一个错误版本，可以采用 copy-on-write：保留旧版本及其状态，新版本只重写受影响的 shard，并在 manifest 中记录 parent version 和变更摘要。训练入口默认只接受新版本；旧版本仍可用于复盘历史运行，但不应被新的训练任务误选。

---

## 7. 数据 manifest

Manifest 是数据版本的目录清单。它描述一个数据集版本包含哪些文件、每个文件的 hash、大小、样本数、token 数和来源统计。

一个 manifest 可以包含：

1. dataset_name。
2. dataset_version。
3. created_at。
4. creator。
5. parent_versions。
6. processing_pipeline_version。
7. shard_list。
8. shard_hash。
9. sample_count。
10. token_count。
11. language_distribution。
12. domain_distribution。
13. license_distribution。
14. quality_summary。
15. known_risks。

Manifest 是训练可复现的关键。如果没有 manifest，只知道“用了某个数据目录”，复现几乎不可靠。

manifest 还承担供应链清单的作用。它不只告诉训练器去哪里读取文件，也让审计者知道某个版本由哪些对象组成、统计是否合理、哪些风险被声明。可以先做一个手算：如果三个 shard 的 token 数分别是 `1,130`、`820` 和 `1,040`，则版本总量应为：

~~~math
N_{mathrm{token}} = 1130 + 820 + 1040 = 2990
~~~

如果训练日志声称读取了 3,200 token，至少有一个引用、统计口径或计数器不一致。这个简单的总量检查不能证明数据合法，却能及时发现 shard 漏读、重复读取或 manifest 与实际文件不匹配。

大型数据系统通常需要分层 manifest。顶层 manifest 引用数据版本、schema、pipeline、父版本和统计；分片 manifest 再记录每个 shard 的 hash、样本数、token 数、压缩格式和位置；必要时还保存样本索引或可重建的范围。分层设计可以降低单个清单的大小，但必须保证父子清单的 hash 和版本关系可验证。若只把 shard 路径写入文本而不记录 digest，清单会随着对象迁移失去身份。

manifest 的统计也有口径边界。`token_count` 可能是 tokenizer A 的 token 数，`byte_size` 是压缩后大小，`sample_count` 是清洗后的样本数；如果不记录计算方式，跨版本比较会产生假差异。发布报告应标明 tokenizer、过滤状态和是否包含被隔离对象，避免把“存储总量”“训练候选量”和“实际消费量”混成一个数字。

---

## 8. 权限控制

不是所有数据都应该对所有人开放。

权限控制要区分：

1. 原始数据。
2. 清洗后数据。
3. 脱敏数据。
4. 高风险数据。
5. 用户数据。
6. 企业内部数据。
7. 标注数据。
8. 评测集和 benchmark。

常见策略包括：

1. 最小权限原则。
2. 按数据敏感度分级。
3. 访问日志。
4. 审批流程。
5. 加密存储。
6. 临时访问令牌。
7. 数据导出限制。
8. 定期权限审计。

评测集尤其要保护。训练团队如果随意访问测试集，可能造成污染和评估失真。

访问策略可以用主体、对象、动作、目的和有效期来描述。比如“研究员 R 在实验 E 期间可以读取脱敏数据版本 v3 的聚合统计，但不能导出原文”，比“研究员 R 有数据权限”更可审计。角色型访问控制适合稳定的团队职责，属性型访问控制更适合按数据敏感度、项目、地区和用途变化的场景；实际系统常把两者组合，并把最终授权决定写入日志。

最小权限不是把所有人都拒绝，而是让工作所需的最小数据暴露出来。训练工程师可能只需要读取已批准 shard，质量分析师可能只需要读取标签和统计，审计人员可能只需要读取 hash、事件和审批记录。为每个角色提供合适的视图，通常比让所有人访问原始库再依赖口头约定更可靠。

评测集还要有独立的生命周期。它可以被用于评估，但不应因为同一套存储接口方便而自动出现在训练数据目录中；访问评测集的研究运行应标记用途，导出和复制应受限。污染不仅来自恶意行为，也可能来自调试脚本把测试样例写入缓存、模型生成数据再被回收进训练池。权限日志、数据 lineage 和 contamination 标记需要联动，才能定位这种非故意污染。

一个常见失败是把权限分级写成数据字段，却没有在存储、服务和导出路径真正执行。此时 `access_level=restricted` 只是一句声明。验证权限时应使用不同主体和动作做负向测试，确认未授权请求被拒绝、拒绝事件可追踪、授权到期后确实失效，并抽查导出文件是否仍带有版本和敏感度标识。

---

## 9. 删除请求和数据撤回

大模型数据治理必须考虑删除请求。用户、数据提供方、版权方或合规团队可能要求删除某些数据。

删除机制要回答：

1. 如何定位相关数据？
2. 哪些版本包含这些数据？
3. 是否进入过训练？
4. 是否在后续数据版本中删除？
5. 是否需要重新训练、继续训练或模型层面处理？
6. 如何记录删除证明？

删除不是简单从当前目录删文件。因为旧版本、缓存、shard、索引、embedding、训练日志和派生数据都可能包含相关内容。

因此，删除请求需要 lineage、source_id、hash、索引和版本记录支持。

一个可靠的处理流程可以从请求登记开始。系统先给请求分配稳定的 request id，保存接收时间、请求范围、验证状态和负责团队；随后通过 source、对象 hash、用户标识或其他允许的索引寻找命中对象；命中后建立受影响版本和派生对象清单，并把对象标记为不可进入新快照。每个阶段都应有明确状态，例如 `received`、`targeted`、`isolated`、`replayed`、`exception`，而不是只写一个最终日期。

删除的难点通常在派生关系。原始文档可能已经被切成样本、合并到 shard、写入检索索引、生成 embedding，甚至参与过合成数据或偏好数据。系统至少要知道哪些派生对象可以直接定位、哪些只能通过重建得到、哪些历史训练运行受影响。对无法逐个定位的对象，应把重建或重新导出作为动作，并记录旧对象何时停止使用。

还要区分“未来不再使用”和“历史影响已经消除”。把样本从下一个版本排除，是一个有价值的控制动作，但它不等于旧版本、缓存和模型参数已经不再包含相关影响。工程文档应分别报告这两个状态，避免向请求方或内部决策者给出超出证据的结论。是否需要模型重训或其他补救，需要结合适用规则和产品风险进行专门判断。

---

## 10. 数据审计

数据审计是定期检查数据系统是否符合质量、合规和安全要求。

审计内容包括：

1. 来源是否合法合规。
2. license 是否记录完整。
3. PII 处理是否生效。
4. 删除请求是否执行。
5. 访问权限是否合理。
6. 数据版本是否可复现。
7. 高风险数据是否隔离。
8. benchmark 是否被污染。
9. 标注规范是否执行。
10. 数据文档是否完整。

审计不是训练结束后的形式流程，而应该嵌入数据 pipeline。

审计可以分成三种互补的工作。自动校验适合检查 hash、字段类型、枚举值、数量守恒和状态转换；抽样复核适合判断 license 文本、PII 标签、质量标签和处理结果是否真的相符；面向事件的回放适合验证一个来源、删除请求或权限工单能否沿 lineage 找到所有受影响对象。只有自动校验，容易把错误输入当成正确事实；只有人工抽样，又无法覆盖大规模数据。

审计报告要保存范围和分母。例如“PII 检测通过”需要说明检测覆盖的是全部样本、某个 shard，还是只抽样了 10,000 条；“删除完成”需要说明请求命中了多少对象、哪些派生层已经处理、哪些仍在例外队列。没有范围的通过语句无法被复核，也无法比较两个版本的风险变化。

审计本身也可能失败。任务被中断、重试产生重复事件、日志写入和数据写入不在同一个事务中，都会制造“日志说完成、对象实际未更新”的分离状态。生产系统要为审计事件提供幂等键、运行状态、失败重试记录和最终一致性检查；对关键动作，还要从结果存储反向抽查，而不是只相信执行端返回值。

---

## 11. Datasheet：给数据集写说明书

Datasheets for Datasets 的核心思想是：数据集像硬件组件一样，也需要说明书。

一个 datasheet 应该回答：

1. 为什么创建这个数据集？
2. 数据由什么组成？
3. 如何采集？
4. 如何清洗和标注？
5. 包含哪些人群、语言、领域和模态？
6. 有哪些偏差和风险？
7. 适合什么用途？
8. 不适合什么用途？
9. 是否包含敏感信息？
10. 如何维护、更新和删除？

对大模型训练数据，datasheet 可以是内部文档，也可以在开源或对外合作时提供简化版本。重点是透明和可沟通。

说明书的价值在于把“数据看起来是什么”与“数据可以被怎样使用”分开。组成部分要描述数量、语言、领域、模态和时间分布；采集部分要说明来源类型、筛选范围和可能的选择偏差；处理部分要说明解析、去重、质量和安全过滤；用途部分要给出适合与不适合的任务；维护部分要说明版本、更新、纠错和删除请求如何进入流程。每一项都应指向具体版本，而不是只写一份永远不更新的总说明。

对于内部数据集，datasheet 可以包含不宜公开的来源细节，但仍应对可见范围做权限控制。对外 dataset card 则可能需要隐藏敏感来源或个人信息，同时保留足够的限制和风险说明。简化发布不应删掉所有不确定性；如果某类来源尚未完成授权核验，应该明确标成限制或待确认，而不是用“公开数据”四个字一笔带过。

写说明书还要防止文档与数据脱节。新版本生成时，系统可以自动填充样本数、token 数、语言分布和处理版本，但偏差、适用范围和已知风险仍需要责任人审阅。自动生成解决的是遗漏和口径统一，不能替代对数据含义的判断。

---

## 12. 数据治理和 model card 的关系

Model card 说明模型的用途、评估、限制和风险。数据治理为 model card 提供事实基础。

例如 model card 中常见内容：

1. 训练数据概述。
2. 适用语言和领域。
3. 不适用场景。
4. 已知偏差。
5. 安全评估。
6. 隐私风险。
7. 更新和删除策略。

这些都需要数据版本和治理支撑。如果训练数据来源、过滤策略和风险标签都不清楚，model card 就只能写空话。

两种文档的关系可以理解成“数据集的证据”与“模型的解释”。数据集文档回答训练输入是什么、如何形成、有什么风险；model card 进一步回答模型在这些输入和后训练过程下表现如何。模型卡可以引用多个数据版本和评测版本，因为同一模型可能经历不同训练运行；也可能需要说明某个能力来自后训练或工具系统，而不能全部归因于预训练数据。

一个实用的写法是为模型卡的每个关键断言保存证据引用。例如“支持中文技术问答”应关联语言/领域统计和对应评测；“不适合医疗诊断”应关联风险分析、评测结果和产品限制；“已处理删除请求”应关联数据版本、请求记录和影响评估。若证据只有定性描述，就应降低措辞强度，使用“在有限评测中观察到”而不是普遍保证。

Model card 也不是法律合规证书。它可以提高透明度，帮助用户理解限制，但不能替代授权审查、隐私影响评估、访问控制或事故响应。发布前应检查模型卡引用的训练版本、评测版本和实际发布 checkpoint 是否一致，否则文档即使内容完整，也可能描述的是另一个模型。

---

## 13. 数据版本和训练可复现

训练可复现需要记录：

1. 模型代码版本。
2. tokenizer 版本。
3. 数据版本。
4. data mixture 配置。
5. 采样随机种子。
6. 训练超参。
7. 过滤规则版本。
8. 评测集版本。
9. checkpoint 版本。
10. 环境和依赖版本。

其中数据版本经常被低估。很多训练无法复现不是因为模型代码变了，而是数据目录被重写、过滤器变了、数据源更新了、shard 顺序变了。

复现还要先区分目标。严格复现关心同一输入和同一实现下的训练轨迹；工程复现关心重新运行后指标是否在约定区间；科学复现则关心结论是否在合理数据和随机性扰动下成立。前者可能需要固定硬件、依赖和算子确定性，后者不一定要求每个浮点数相同。没有明确目标，团队会把“loss 曲线不完全一致”误报成失败，或把“最终分数相近”误报成完全复现。

数据侧常见的隐性变量包括 shard 顺序、过滤后的空样本、tokenizer 版本、packing 规则、重复样本、采样权重和 worker 的随机种子。比如两个版本包含相同文本，但一个版本把长文档切成不同窗口，训练 token 数和边界上下文就已经改变。训练日志除了记录数据版本，还要记录如何从版本得到实际 batch 的规则。

可以把一次运行的可复现引用写成：

~~~math
R = (V_{mathrm{code}}, V_{mathrm{data}}, V_{mathrm{tokenizer}}, V_{mathrm{config}}, V_{mathrm{eval}}, E_{mathrm{env}})
~~~

其中前五项是版本化对象，`E_env` 是依赖、硬件和运行环境摘要。这个表达式不是要求所有系统都使用同一工具，而是提醒读者：只保存 checkpoint 文件，无法说明 checkpoint 是在什么输入和环境上产生的。

---

## 14. 数据 schema 演进

数据系统会不断新增字段。例如一开始只有 source 和 language，后来增加 license、quality_score、pii_label、safety_label、dedup_cluster_id。

Schema 演进要注意兼容性：

1. 新字段默认值。
2. 旧数据如何回填。
3. 字段含义是否变化。
4. 标签体系是否更新。
5. 训练 sampler 是否依赖该字段。
6. 下游评估是否使用该字段。

字段含义变化必须版本化。比如 quality_score v1 和 v2 使用不同模型，不能混在一起当同一个分数解释。

schema 演进至少有三种变化。加一个有明确默认值的可选字段，通常是向后兼容的；删除字段或改变字段类型，可能让旧消费者直接失败；字段仍叫同一个名字但含义变化，则是最危险的语义不兼容，因为程序可能继续运行，却把不同含义的数据混在一起。治理字段不应只检查解析是否成功，还要检查生产者和消费者理解的语义是否一致。

以 `quality_score` 为例，v1 可能是规则分数，范围是 `[0, 1]`；v2 可能是分类模型校准后的概率，样本分布和阈值都不同。如果简单把 v2 回填到旧记录中，训练 sampler 会把两个分数当作同一个量比较。更稳妥的迁移方式是保留原字段、增加评分器版本和计算时间，或者生成新的 schema 版本，让下游显式选择转换规则。

回填也要区分“没有历史证据”和“根据新规则推断”。如果旧数据没有 license 记录，批量回填 `license=permissive` 是危险的；即使根据来源默认值推断，也应把它标记为推断值，保留推断规则和置信度。这样后续的授权审计不会把自动补齐误认为原始证明。

---

## 15. 数据治理中的角色分工

数据治理不是某一个人的事。

常见角色包括：

1. 数据工程：采集、清洗、存储、版本。
2. 训练工程：采样、配比、训练日志。
3. 研究团队：数据实验、估值、评估。
4. 安全团队：风险分类、安全过滤、红队数据。
5. 法务合规：license、隐私、删除请求。
6. 标注团队：标注规范、质量审计。
7. 产品团队：用户反馈、真实场景。
8. 平台团队：权限、审计、基础设施。

成熟组织会把这些责任写进流程，而不是靠口头沟通。

角色分工的重点不是把问题转给别人，而是明确每个对象的 owner、reviewer 和执行者。数据工程团队可以负责生成 manifest 和 lineage 事件，但不应独自决定一个来源是否适合公开训练；合规团队可以确认适用政策，却需要数据工程提供可查询的命中范围；训练团队负责固定运行引用，也要在发现输入异常时停止消费，而不是把问题留到模型评估阶段。

一次数据版本发布至少需要几个交接点：来源登记完成后由授权责任人确认，处理流水线完成后由质量/安全责任人抽样复核，manifest 生成后由训练或平台责任人验证可读性，进入训练前由运行系统保存版本引用，发布前由文档和产品责任人核对限制。每个交接点都应留下状态、时间和证据链接。没有 owner 的字段最终会变成“大家以为别人会维护”的空字段。

责任划分还要覆盖异常。某个来源突然出现 PII 比例上升时，谁有权暂停数据发布？删除请求未找到对象时，谁负责解释缺口？评测集被写入训练缓存时，谁负责隔离受影响运行？把这些问题提前写进 runbook，比发生事故后临时寻找负责人更能缩短响应时间。

---

## 16. 数据治理指标

数据治理也需要指标。

常见指标包括：

1. 数据版本可复现率。
2. 样本 lineage 覆盖率。
3. license 记录完整率。
4. PII 检测覆盖率。
5. 删除请求处理时延。
6. 权限审计通过率。
7. 数据文档完整率。
8. benchmark 污染检测覆盖率。
9. 高风险数据隔离率。
10. 数据质量回归通过率。

这些指标让治理从抽象要求变成可执行工程。

指标必须带着口径、分层和行动阈值。比如 lineage 覆盖率可以按全部样本、训练候选样本和高风险样本分别计算；删除处理时延要区分目标定位时延、隔离时延和最终回放时延；权限审计通过率要报告审计对象数和被豁免对象数。只有一个总体百分比，无法判断风险集中在哪里。

还要防止指标被“优化”成好看的数字。把难以处理的样本从分母中排除，会提高覆盖率；把所有请求一律拒绝，会提高高风险阻断率，却让授权任务无法运行；把文档字段填入默认文本，会提高完整率，却没有增加事实证据。因此指标应与样本抽查、异常队列和失败动作配套，并保留原始分母和排除理由。

一个简单的分层例子是：总体 license 记录完整率为 99%，但高风险来源完整率只有 80%。发布决策应优先关注后一个数字，因为风险不是按样本平均分配的。实际系统可以为不同风险桶设置不同阈值，也可以要求高风险桶必须逐项确认，而不是依赖总体平均值。指标是观察仪表盘，最终动作仍需要结合版本用途和责任边界。

---

## 17. 机制与边界：治理是模型行为控制面的底座

从机制上看，dataset governance 是模型行为控制面的底座。

模型行为问题经常被归因于算法，但很多问题来自数据：某批安全数据过度保守、某个代码源污染评测、某个合成数据版本质量下降、某些低资源语言被误删、某个旧法规文本未更新。

如果没有数据版本和 lineage，团队只能在模型层面盲目调参。如果有治理系统，就可以追溯、回滚、重洗、重采样、隔离和审计。

治理的价值不是让流程变慢，而是让大规模模型迭代可控。

可以用一个具体回归来理解控制面的作用。模型新版本在数学题上提升，但对正常用户请求的拒答率突然上升。没有治理记录时，团队可能继续调学习率、改 system prompt 或增加后训练数据；有了版本和 lineage，就可以先比较新旧数据版本的安全样本比例、拒答标签、合成数据来源和过滤器版本，再用分层评测验证变化是否集中在某个数据桶。

这不意味着数据治理可以替代算法分析。它只能缩小搜索范围，提供可回滚的输入和可解释的变化点；最终仍需要研究人员判断数据变化是否导致目标行为改变。治理控制面与模型控制面之间的边界也要写清：访问策略、数据版本和发布审批控制的是输入和使用方式，模型参数、损失函数和推理策略控制的是学习与输出机制。

从专家角度看，治理系统的价值还在于保留反事实实验的条件。若新模型退化，可以固定代码和训练配置，只替换数据版本；也可以固定数据，只替换过滤器或 mixture。每次实验都引用明确对象，团队才能把相关性逐步推进到可验证的因果假设，而不是凭感觉归因。

---

## 18. 一个可落地的数据版本与治理方案

一个可运行的数据治理系统，先把样本、派生对象、数据版本和训练运行分成不同对象，再用 manifest 和 lineage 把它们连接起来。schema 记录 source、license、language、domain、quality、safety、PII、dedup、version 和 deletion_status；快照记录 hash、shard、父版本和统计摘要；pipeline 记录采集、解析、清洗、过滤、去重、质量评分和污染检测的代码与配置版本。

权限要按敏感度和用途分级，评测集、高风险数据、用户数据和原始对象不能与公开脱敏数据使用同一访问边界。删除机制要能按来源、hash、用户请求或 license 状态定位到版本、shard、索引和派生对象，并把每个处理状态写入审计记录。

文档和训练集成是同一系统的两端。datasheet 解释数据为何存在、如何采集和适合什么用途，model card 解释模型使用这些数据后表现如何、有哪些限制；训练日志则把数据版本、mixture、sampler、tokenizer 和评测集版本固定下来。审计不是训练后的附加审批，而是对这些对象关系的周期性回放。

### 18.1 最小可运行数据版本治理 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy 样本、访问请求、删除请求和文档完成状态；输出包括 manifest、可训练样本、license 分布、lineage 覆盖率、访问阻断、删除请求命中、datasheet 完成度、model card 输入是否完整、检查信号和后续动作。

它演示的是治理机制，不是真实 DVC、对象存储、权限系统、合规系统或生产数据平台。真实系统需要接入对象存储版本、数据目录、权限服务、审计日志、删除工单、dataset card / datasheet 文档和训练日志。demo 的 decision 只表示当前版本下一步应做什么，不把一组检查压成“治理完成”的绝对结论。

~~~python
import hashlib
from collections import Counter, defaultdict


samples = [
    {"id": "s1", "source": "web_blog", "shard": "shard-a", "tokens": 520, "license": "permissive", "pii": False, "contam": False, "access": "public", "lineage": ["crawl", "parse", "quality", "dedup"], "deleted": False},
    {"id": "s2", "source": "math_docs", "shard": "shard-a", "tokens": 610, "license": "permissive", "pii": False, "contam": False, "access": "public", "lineage": ["crawl", "parse", "quality", "dedup"], "deleted": False},
    {"id": "s3", "source": "user_forum", "shard": "shard-b", "tokens": 430, "license": "review", "pii": True, "contam": False, "access": "restricted", "lineage": ["crawl", "parse", "quality"], "deleted": False},
    {"id": "s4", "source": "benchmark_site", "shard": "shard-b", "tokens": 390, "license": "blocked", "pii": False, "contam": True, "access": "quarantine", "lineage": ["crawl", "parse", "quality", "contam_scan"], "deleted": False},
    {"id": "s5", "source": "code_repo", "shard": "shard-c", "tokens": 740, "license": "permissive", "pii": False, "contam": False, "access": "internal", "lineage": ["crawl", "parse", "secret_scan", "dedup"], "deleted": False},
    {"id": "s6", "source": "old_vendor", "shard": "shard-c", "tokens": 300, "license": "expired", "pii": False, "contam": False, "access": "blocked", "lineage": ["import", "quality"], "deleted": True},
]

required_lineage = {"crawl", "parse", "quality"}
required_docs = {"purpose", "composition", "collection", "processing", "risks", "maintenance"}
datasheet = {"purpose": True, "composition": True, "collection": True, "processing": True, "risks": True, "maintenance": False}
model_card_inputs = {"training_data": True, "eval_data": True, "limitations": True, "risk_summary": True, "deletion_policy": False}

access_requests = [
    {"user": "trainer", "sample": "s1", "allowed_levels": {"public", "internal"}},
    {"user": "contractor", "sample": "s5", "allowed_levels": {"public"}},
    {"user": "auditor", "sample": "s3", "allowed_levels": {"public", "internal", "restricted"}},
    {"user": "researcher", "sample": "s4", "allowed_levels": {"public", "internal"}},
]
deletion_requests = [{"request_id": "del-001", "source": "old_vendor"}, {"request_id": "del-002", "source": "missing_source"}]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sample_hash(item):
    payload = f"{item['id']}|{item['source']}|{item['tokens']}|{item['license']}|{item['access']}"
    return digest(payload)


shards = defaultdict(list)
for item in samples:
    shards[item["shard"]].append(item)

manifest = []
for shard, rows in sorted(shards.items()):
    joined = ";".join(sample_hash(row) for row in sorted(rows, key=lambda x: x["id"]))
    manifest.append({"shard": shard, "sample_count": len(rows), "tokens": sum(row["tokens"] for row in rows), "checksum": digest(joined)})

eligible = [item for item in samples if item["license"] == "permissive" and not item["pii"] and not item["contam"] and not item["deleted"]]
license_counts = Counter(item["license"] for item in samples)
lineage_ok = [item["id"] for item in samples if required_lineage.issubset(set(item["lineage"]))]
lineage_coverage = round(len(lineage_ok) / len(samples), 3)

blocked_access = []
for req in access_requests:
    item = next(row for row in samples if row["id"] == req["sample"])
    if item["access"] not in req["allowed_levels"]:
        blocked_access.append((req["user"], req["sample"], item["access"]))

deletion_hits = {}
for req in deletion_requests:
    deletion_hits[req["request_id"]] = [item["id"] for item in samples if item["source"] == req["source"]]

datasheet_completion = round(sum(datasheet.values()) / len(required_docs), 3)
model_card_ready = all(model_card_inputs.values())

report = {
    "dataset_version": "data-v2026-06-06.1",
    "manifest": manifest,
    "eligible_ids": [item["id"] for item in eligible],
    "license_counts": dict(sorted(license_counts.items())),
    "lineage_coverage": lineage_coverage,
    "lineage_missing": [item["id"] for item in samples if item["id"] not in lineage_ok],
    "blocked_access": blocked_access,
    "deletion_hits": deletion_hits,
    "datasheet_completion": datasheet_completion,
    "model_card_ready": model_card_ready,
}

deletion_status = {}
for req in deletion_requests:
    request_id = req["request_id"]
    hits = deletion_hits[request_id]
    if not hits:
        deletion_status[request_id] = "unresolved_target"
    elif all(item["deleted"] for item in samples if item["id"] in hits):
        deletion_status[request_id] = "completed"
    else:
        deletion_status[request_id] = "needs_isolation"

checks = {
    "manifest_checksums": all(row["checksum"] for row in manifest),
    "lineage_minimum": report["lineage_coverage"] >= 0.65,
    "training_eligibility": set(report["eligible_ids"]) == {"s1", "s2", "s5"},
    "access_controls": len(blocked_access) == 2,
    "deletion_targets_resolved": all(bool(hits) for hits in deletion_hits.values()),
    "deletion_completed": all(status == "completed" for status in deletion_status.values()),
    "datasheet_documented": datasheet_completion >= 0.80,
    "model_card_inputs": model_card_ready,
}

signals = {
    "lineage_missing_ids": report["lineage_missing"],
    "unresolved_deletion_requests": [
        request_id for request_id, status in deletion_status.items() if status == "unresolved_target"
    ],
    "deletion_status": deletion_status,
    "missing_datasheet_fields": [key for key, value in datasheet.items() if not value],
    "missing_model_card_inputs": [key for key, value in model_card_inputs.items() if not value],
}

actions = []
if signals["lineage_missing_ids"]:
    actions.append("repair_lineage_before_claiming_full_reproducibility")
if signals["unresolved_deletion_requests"]:
    actions.append("resolve_deletion_request_targets")
if not checks["deletion_completed"]:
    actions.append("replay_deletion_across_affected_objects")
if signals["missing_datasheet_fields"]:
    actions.append("complete_datasheet_maintenance_section")
if signals["missing_model_card_inputs"]:
    actions.append("document_deletion_policy_in_model_card_inputs")

decision = "hold_for_governance_repair" if actions else "proceed_to_release_review"
report["deletion_status"] = deletion_status
report["checks"] = checks
report["signals"] = signals
report["actions"] = actions
report["decision"] = decision

for key, value in report.items():
    print(f"{key}=", value)

assert report["eligible_ids"] == ["s1", "s2", "s5"]
assert report["lineage_coverage"] == 0.667
assert report["lineage_missing"] == ["s5", "s6"]
assert report["blocked_access"] == [("contractor", "s5", "internal"), ("researcher", "s4", "quarantine")]
assert report["deletion_hits"] == {"del-001": ["s6"], "del-002": []}
assert report["datasheet_completion"] == 0.833
assert report["deletion_status"] == {"del-001": "completed", "del-002": "unresolved_target"}
assert report["checks"]["deletion_targets_resolved"] is False
assert report["checks"]["model_card_inputs"] is False
assert report["decision"] == "hold_for_governance_repair"
~~~

运行后会看到类似输出：

~~~text
dataset_version= data-v2026-06-06.1
manifest= [{'shard': 'shard-a', 'sample_count': 2, 'tokens': 1130, 'checksum': '8ee90eff217c'}, {'shard': 'shard-b', 'sample_count': 2, 'tokens': 820, 'checksum': '83cc9bdbbafe'}, {'shard': 'shard-c', 'sample_count': 2, 'tokens': 1040, 'checksum': 'b5bf80f8852a'}]
eligible_ids= ['s1', 's2', 's5']
license_counts= {'blocked': 1, 'expired': 1, 'permissive': 3, 'review': 1}
lineage_coverage= 0.667
lineage_missing= ['s5', 's6']
blocked_access= [('contractor', 's5', 'internal'), ('researcher', 's4', 'quarantine')]
deletion_hits= {'del-001': ['s6'], 'del-002': []}
datasheet_completion= 0.833
model_card_ready= False
deletion_status= {'del-001': 'completed', 'del-002': 'unresolved_target'}
checks= {'manifest_checksums': True, 'lineage_minimum': True, 'training_eligibility': True, 'access_controls': True, 'deletion_targets_resolved': False, 'deletion_completed': False, 'datasheet_documented': True, 'model_card_inputs': False}
signals= {'lineage_missing_ids': ['s5', 's6'], 'unresolved_deletion_requests': ['del-002'], 'deletion_status': {'del-001': 'completed', 'del-002': 'unresolved_target'}, 'missing_datasheet_fields': ['maintenance'], 'missing_model_card_inputs': ['deletion_policy']}
actions= ['repair_lineage_before_claiming_full_reproducibility', 'resolve_deletion_request_targets', 'replay_deletion_across_affected_objects', 'complete_datasheet_maintenance_section', 'document_deletion_policy_in_model_card_inputs']
decision= hold_for_governance_repair
~~~

这个 demo 的重点是把几个经常被混为一谈的问题拆开。`eligible_ids` 只说明样本满足当前训练集筛选条件；`checks` 描述已经核验的事实，`signals` 保留缺失对象和未完成事项，`actions` 把缺口翻译成可执行的修复工作，`decision` 才表示当前版本是否可以进入下一步。样本 s6 已经从训练候选中排除，但请求 del-002 没有找到对应对象，所以系统不能把“当前候选集没有它”解释成“删除请求已经完成”。同样，datasheet 的维护字段和 model card 的删除策略事实尚未补齐，版本应停留在治理修复阶段，而不是被一个总布尔值掩盖。

---

## 19. 从数据变更到模型发布：六个必须分开的判断

当一个数据版本准备进入训练或发布流程时，工程团队最容易犯的错误，是把“文件存在”“样本可读”“训练跑通”和“可以对外负责”当成同一个判断。本章前面的对象和指标，最终要落到几个互相独立的决策边界上。它们之间有先后关系，但不能互相替代。

### 19.1 版本边界：我们到底在使用什么？

版本边界首先要回答的是对象身份，而不是时间。`data-v2026-06-06.1` 这样的名字只能帮助人阅读，真正决定身份的是 manifest、shard checksum、schema、处理代码和配置的组合。只要其中任意一项改变，就要判断这是同一版本的修订，还是一个新的版本对象。对已经参与训练的快照，最稳妥的做法是保持不可变，修正问题时创建新版本并记录 parent version。

例如，团队把一个 shard 里的十万条文本重新排序，样本集合没有变化，许多任务的结果可能看起来不受影响；但如果 sampler 依赖全局顺序、随机种子没有固定，训练批次和梯度累积就可能改变。反过来，只有文件名改变而内容、处理配置和 manifest 都未改变，通常只是仓库命名变化，不应制造虚假的数据差异。版本审查因此要同时看内容 hash 和语义元数据，不能只看目录时间戳。

### 19.2 血缘边界：一个结果能否回到来源？

版本告诉我们“使用了哪一个快照”，lineage 告诉我们“快照中的对象是怎样形成的”。从训练样本向前追溯时，至少应能找到原始对象、采集或导入动作、每个改变内容或标签的处理步骤，以及最终进入的 shard 和数据版本；从一个来源向后追踪时，则要能列出它影响过哪些版本、训练运行和发布模型。两条方向都只记录一半，排查问题时仍然会断链。

血缘完整不等于处理正确。一个错误的 PII 分类器也可以留下完整的执行记录，一个错误的 license 映射也可以拥有漂亮的图。审计要抽取样本检查“记录的动作”和“样本的实际状态”是否相符，并把处理器版本、输入 hash、输出 hash 和运行时间放在同一条事件中。这样才能区分“没有记录”与“记录了错误事实”这两类完全不同的问题。

### 19.3 删除边界：缺席、隔离和完成不是一回事

删除请求通常包含三个阶段。第一阶段是定位：把请求中的来源、对象标识、用户标识或权利凭据映射到内部对象。第二阶段是隔离：阻止命中的对象继续进入新的训练快照、索引、缓存或导出任务。第三阶段才是完成：在适用的存储层和派生对象上执行约定的删除或不可用化动作，并留下可审计的结果。请求没有命中对象时，系统应该报告“目标未解析”，而不是把空列表当成成功。

已经训练过的模型还要单独处理。数据从训练集删除，并不自动证明模型参数中不再包含相关影响；是否需要重训、继续训练、遗忘处理或仅限制未来使用，取决于数据类型、模型用途、适用法律和组织承诺。工程系统能做的是提供影响范围：命中的版本、shard、索引、缓存、训练运行和发布物。法律结论和具体补救措施必须由有权限的合规与产品责任人结合实际场景判断，本章不把一种工程实现写成普遍法律答案。

### 19.4 权限边界：谁可以在什么目的下看到什么？

权限不是一个目录读写位，而是主体、对象、动作、目的和时间的组合。一个训练工程师可能被允许读取经过脱敏的 shard，却不能读取原始用户记录；一个审计人员可能可以查看 lineage 和 hash，却不应获得不必要的正文；一个研究人员可以运行聚合统计，却不能把 benchmark 原文导出到训练目录。授权记录若只写“某人有数据权限”，后续无法判断权限是否超出目的。

最小权限还需要和可用性一起设计。过度收紧的权限会推动团队复制数据、使用个人目录或绕开正式流程，反而扩大不可审计的副本。较好的做法是提供有期限、可撤回、带用途和导出限制的访问方式，并让访问日志能关联到数据版本和工单。对评测集尤其要把“可评估”与“可训练”分开，避免评估集因为方便读取而被无意混入数据混合物。

### 19.5 文档边界：事实、解释和承诺必须分层

datasheet、dataset card 和 model card 不是同一份文档的不同标题。数据集说明书主要描述数据的动机、组成、采集、处理、限制和维护方式；模型卡要说明模型的用途、评测、已知风险和不适用场景；训练运行记录则负责固定某一次实验实际使用的版本和配置。它们可以互相引用，但不能用模型卡里的概括性描述替代样本级或版本级证据。

写文档时还要区分三种句子。事实句应能回到 manifest、日志或评测结果，例如“该版本包含三个 shard”；解释句说明为什么采用某个过滤规则；承诺句说明未来如何处理删除、更新或限制。承诺句不能因为当前版本有一个字段就被写成已经实现的能力。示例中的 `deletion_policy` 缺失，意味着发布材料没有足够事实支撑该段说明，应先补记录、完成责任人确认，再决定如何对外表述。

### 19.6 复现边界：复现什么，允许多大差异？

“可复现”需要先定义目标。严格复现要求同一数据、代码、依赖、随机性和硬件路径下得到可比较的训练轨迹；工程复现可能只要求在固定数据版本和代码版本上重跑，指标落在容许区间；科学复现则更关注结论是否在合理扰动下仍成立。没有这个定义，团队很容易把一次成功重跑误报成完全复现。

可以把一次训练运行写成引用集合：模型代码 commit、tokenizer、数据版本、mixture、随机种子、超参数、评测集、checkpoint、容器和依赖版本。发生差异时，先比较这些引用，再判断是数据差异、实现差异还是非确定性。版本、血缘、权限、删除和文档都服务于这个边界，但任何一个单独完善的组件都不能替代完整的运行记录。

---

## 20. 治理失败模式与修复顺序

治理系统通常不是一次性建成的。更常见的情况是：训练已经运行了几轮，团队才发现来源、权限或删除记录不完整。此时最危险的做法是继续添加更多表格和审批，把没有证据的字段填成“已完成”。修复顺序应该由影响范围和不可逆程度决定。

第一类失败是身份不稳定。数据目录被原地覆盖，manifest 没有 hash，训练日志只留下一个路径。修复时先冻结当前目录并制作一次带时间、内容摘要和负责人确认的取证快照；不能把今天读到的文件直接宣称为历史版本。随后把后续流水线改为新版本写入，并让训练入口只接受 manifest 引用。

第二类失败是血缘断裂。样本有 `source_id`，但没有记录解析器、过滤器或去重聚类；或者 shard 有清单，样本级记录却无法关联。修复重点不是先画一张漂亮的图，而是选择一条端到端路径建立最小闭环：原始对象 hash、处理事件、输出对象 hash、版本和训练运行全部可互相引用。对历史数据无法补出的部分要标记为未知，并在评估和发布文档中说明覆盖范围。

第三类失败是资格和权限混用。团队把“能读到”当成“可以训练”，把“公开可见”当成“没有授权限制”，把脱敏副本和原始副本放在同一访问边界。修复时要先定义数据分类和允许动作，再回收超范围权限；同时提供能完成工作的脱敏视图和聚合接口，避免权限修复把用户推向不可记录的副本。

第四类失败是删除只覆盖主存储。旧版本、对象存储版本、缓存、索引和导出文件仍然存在，系统却只把当前目录中的文件删除。修复应先建立影响清单，按“定位、隔离、派生对象、训练运行、发布物”的顺序回放，并为每一步记录状态。无法处理的对象不能静默跳过，要进入例外队列，由责任人决定后续动作。

第五类失败是文档滞后。datasheet 和 model card 在发布前临时拼接，导致“训练数据概述”“删除策略”“适用边界”等句子没有对应证据。修复顺序是先让文档字段引用版本级统计和工单，再让发布流程检查引用是否存在；文档检查发现的是事实缺口，不应靠改措辞来掩盖。

第六类失败是只报一个分数。`lineage_coverage=0.98` 可能很高，但剩下的 2% 恰好是高风险来源；访问阻断率也可能因为所有请求都被拒绝而看起来很好。修复时保留分母、分层统计、例外样本和风险权重，至少分别观察覆盖、正确性、时延和未解决对象。指标是定位工具，不是把责任判断自动化的替代品。

在资源有限时，可以按以下顺序推进：先冻结并识别版本，再保证高风险数据不会继续流入，接着补齐 lineage 和删除影响范围，然后恢复权限与审计的关联，最后把文档、训练入口和发布流程接上。这个顺序的理由是先保护未来状态，再处理历史证据；若直接从文档开始，写出来的结论仍然没有可靠的底层对象支撑。

---

## 21. 从数据变更到模型发布的可回放闭环

把本章内容串起来，可以得到一条可回放的发布链路。新来源先登记并保存授权、敏感度和采集时间；原始对象进入不可变存储后生成 hash；解析、清洗、PII 扫描、质量评估、去重和污染检测各自产生处理事件；这些事件共同构成样本 lineage；经过资格判断的对象写入新的 shard 和 manifest；训练运行引用 manifest、pipeline、tokenizer、mixture、评测集和环境；最后，model card 引用训练事实、评测结果和限制，发布记录保存对应的模型版本。

这条链路的关键不是步骤数量，而是每个步骤都留下能被下一步引用的对象。来源登记没有对象身份，后面就无法做删除映射；处理事件没有输入输出 hash，无法判断结果是否被替换；manifest 没有 schema 和统计，训练日志无法解释数据构成；发布文档没有运行引用，模型卡里的概述就无法回到实际训练。系统设计时应把这些关系当作数据结构，而不是只写在 wiki 里的流程图。

当发生新的删除请求时，闭环应该支持一次受控回放：根据请求找到对象和受影响版本，阻止它进入新快照，重新生成受影响的 manifest 和统计，标记哪些训练运行和派生索引受到影响，再由责任人决定是否重训或采用其他模型层面的措施。回放的产物不仅是“删掉了某个文件”，还应包含命中的对象数量、未解析请求、处理时间、例外原因和新的版本引用。这样下一次审计才能判断动作是否覆盖了真实影响范围。

当模型行为出现回归时，闭环也应该反向工作。先比较模型运行引用的数据版本和前一版本的 manifest，再沿 lineage 找到新增、删除或规则改变的来源，按语言、领域、风险和采样桶切分评测，最后把结论写回数据版本和 model card。若只在模型层面盲目调学习率，可能把一个数据治理问题变成更难解释的训练变化。

工程上可以把一次可回放运行概括为：

~~~text
source snapshot
  -> processing events
  -> sample lineage
  -> immutable dataset version
  -> training run
  -> evaluation evidence
  -> model documentation
  -> release record
~~~

其中每个箭头都应能通过 ID、hash、版本或工单回到前一个对象。回放不要求所有环境永远完全相同，但必须清楚记录哪些部分可以严格重现、哪些部分只能在指标范围内复现、哪些历史信息已经不可获得。治理的成熟度，最终体现在面对变更、删除和回归时，系统能否给出有证据的影响范围和下一步动作。

## 22. 资料与证据边界

本章使用的资料承担不同角色，不能把它们混写成同一种权威。Gebru 等人的 Datasheets for Datasets 是数据集文档化的研究框架，强调动机、组成、采集、处理、用途和限制；Mitchell 等人的 Model Cards for Model Reporting 是模型发布文档框架，强调模型用途、评估、限制和风险。它们回答“应该说明哪些问题”，不替代组织对具体数据的事实核验。

W3C PROV 提供描述实体、活动和责任主体之间关系的通用血缘模型，适合作为 lineage 语义参考；DVC 和 OpenLineage 的官方文档则更接近工程实现，分别帮助理解数据/实验版本管理和跨系统运行事件。它们能说明一种工具如何建模、记录或传递信息，但不能自动证明某个项目已经完成授权审查、隐私评估或删除义务。

Hugging Face Dataset Cards 体现了开放数据集常见的说明字段和发布习惯，NIST AI RMF 提供风险管理的组织化语言。两者适合帮助团队设计文档和风险流程，但仍需结合数据来源、使用目的、主体关系和产品部署环境判断。公开平台上的卡片是发布者提供的声明，阅读时要区分“作者自述”“可验证的统计”“独立评测”和“法律结论”。

GDPR Article 17 常被称为删除权的法律资料入口，但它包含适用范围、例外、责任主体和具体情境判断，不能从条文标题直接推出所有模型都必须采取同一种技术措施。本章只借它说明“删除请求需要被识别、处理和留证”的治理问题；实际项目应由适用法域中的专业人员判断义务、期限、例外和模型层面的补救方式。

因此，资料审查至少要记录四件事：来源是研究论文、标准、工具文档还是法律原文；资料描述的是概念、接口、推荐做法还是强制义务；结论是否有版本和适用范围；当前项目是否有自己的实测证据。工程文档中可以写“依据某资料设计了字段”，但不能写成“该资料证明当前数据合法”或“某工具自动完成了删除”。

### 22.1 资料入口

下面的入口用于继续核对原始定义和版本变化：

1. [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) 和 [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)：研究论文，分别对应数据集说明和模型报告框架。
2. [W3C PROV Overview](https://www.w3.org/TR/prov-overview/)：标准族概览，用于理解实体、活动和责任主体之间的 provenance 关系。
3. [DVC Documentation](https://dvc.org/doc) 与 [OpenLineage Documentation](https://openlineage.io/docs/)：工程工具文档，用于对照数据/实验版本和跨系统运行事件的实现方式。
4. [Hugging Face Dataset Cards](https://huggingface.co/docs/hub/datasets-cards)：开放数据集发布文档的字段和维护方式示例。
5. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：风险管理框架，不是某个项目的自动合规证明。
6. [GDPR Article 17](https://eur-lex.europa.eu/eli/reg/2016/679/art_17/oj)：法律原文入口，具体适用范围、例外和补救措施仍需结合实际法域判断。

阅读这些入口时，应优先查看原论文、标准原文和官方文档的发布日期与版本；第三方文章适合帮助定位问题，但不应单独支撑本章关于工具能力、法律义务或项目事实的确定结论。

## 23. 结语

数据版本化解决的是对象身份，lineage 解决的是过程和来源，manifest 解决的是快照组成，权限和审计解决的是谁在什么条件下使用，删除机制解决的是变更如何传播，datasheet 和 model card 解决的是事实如何被解释和交付。它们共同构成训练系统的可追溯基础。

真正成熟的数据治理并不是把所有不确定性伪装成一个“通过”标记，而是在证据不足时明确说出缺什么、影响谁、下一步如何处理。一个版本可以适合内部探索，却还不适合对外发布；一个样本可以满足质量条件，却没有足够授权证据；一次删除可以已经阻止未来流入，却仍需要评估旧版本和训练运行。这些状态都应该被保留下来。

当每一次训练都能引用不可变快照，每个关键样本都能追到来源和处理动作，每个高风险访问都有目的和日志，每次删除都有影响范围和处理记录，模型发布才不再是一次无法解释的打包动作。数据工程由此从“把数据喂给模型”变成一套能够复查、修复和持续负责的系统。
