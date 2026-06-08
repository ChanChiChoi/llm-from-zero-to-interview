# 第十一章：Dataset Versioning 与 Governance

前面章节讲了数据采集、清洗、去重、配比、专项数据、合成数据、偏好安全数据、多模态数据和数据价值评估。本章讨论把这些能力组织成长期可运行系统所必须的底座：dataset versioning 与 governance。

如果没有数据版本和治理，团队可能训练出一个模型，却说不清楚它到底用了哪些数据、哪些规则、哪些过滤器、哪些来源、哪些授权、哪些删除请求是否生效。模型出了问题，也无法追溯到数据版本；想复现实验，也找不到当时的数据快照。

大模型时代，数据治理不再是合规部门的附属工作，而是训练工程的一部分。没有治理的数据系统，规模越大，风险越大。

本章重点：数据版本、元数据、可追溯性、权限、审计、删除请求和数据治理。

合规边界：本章讨论数据治理、权限控制、审计、删除机制和合规文档，不提供绕过访问控制、规避审计、保留应删除数据或滥用个人信息的方法。

## 0. 本讲资料边界与第二轮精修口径

按照 `WRITING_PLAN.md` 的要求，本讲精修前核对了 Datasheets for Datasets、Model Cards、W3C PROV、DVC 数据版本控制文档、OpenLineage、Hugging Face Dataset Cards、NIST AI RMF / Generative AI Profile 和 GDPR Article 17 删除权等公开资料。

本讲聚焦训练数据版本和治理的工程闭环：不可变快照、manifest、checksum、lineage、schema 演进、权限分级、删除请求、datasheet、model card 输入、审计指标和训练可复现记录。

```text
数据来源 -> 处理流水线 -> 不可变快照 -> manifest -> lineage -> 权限 / 删除 / 审计 -> 训练日志 -> 发布文档
```

本讲不提供规避权限、保留应删除数据、绕过审计或滥用个人信息的方法。涉及删除请求和隐私治理时，正文只讨论工程机制和审计口径；具体法律义务应由合规和法务团队结合适用地区判断。

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

---

## 2. 来龙去脉：从数据集文档到基础模型治理

传统机器学习时代，数据集通常比较小，研究者可以在论文中简单描述数据来源、样本数、划分方式和标注方法。但随着模型进入高风险应用，大家逐渐意识到数据集本身需要标准化文档。

Datasheets for Datasets 提出每个数据集都应配套 datasheet，记录动机、组成、采集过程、推荐用途等信息，提升透明度和责任意识。

Model Cards 则强调模型发布时要说明模型用途、评估结果、适用边界和风险。虽然 model card 面向模型，但它离不开数据治理：模型的行为风险很多来自训练数据。

基础模型进一步放大了问题。Foundation model 训练在宽泛数据上，随后被大量下游应用复用。上游数据缺陷会被下游系统继承和放大。因此，数据版本、来源、权限、风险和文档不再是“最好有”，而是模型可控、可审计、可复现的基础。

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

### 5.1 关键公式与治理指标

一个数据版本可以抽象为：

```math
V_t=(D_t,M_t,P_t,S_t,A_t)
```

其中 `D_t` 是数据快照，`M_t` 是 manifest，`P_t` 是处理 pipeline 和配置版本，`S_t` 是 schema 版本，`A_t` 是审计记录。训练日志必须能引用这个整体，而不是只记录一个数据目录名。

Manifest 可以写成 shard 清单：

```math
M_t=\{(h_j,n_j,b_j,c_j)\}_{j=1}^{K_t}
```

其中 `h_j` 是第 `j` 个 shard 的内容 hash 或 checksum，`n_j` 是样本数，`b_j` 是 token 数或字节数，`c_j` 是来源、语言、license、质量和风险统计。

样本级 lineage 可以写成有序处理路径：

```math
L_i=(s_i,o_i,p_{i1},p_{i2},\ldots,p_{im},v_i)
```

其中 `s_i` 是 source id，`o_i` 是原始对象 hash，`p_ij` 是第 `j` 个处理步骤，例如 parse、quality、PII scan、dedup、contamination scan，`v_i` 是样本进入的数据版本。lineage 的价值是同时支持“从样本追来源”和“从来源追模型版本”。

Lineage 覆盖率可以写成：

```math
C_{\mathrm{lineage}}=\frac{|\{i:L_i\ \mathrm{complete}\}|}{N}
```

License 记录完整率可以写成：

```math
C_{\mathrm{license}}=\frac{|\{i:\ell_i\neq \varnothing\}|}{N}
```

删除请求的执行时延可以写成：

```math
T_{\mathrm{delete},r}=t_{\mathrm{closed},r}-t_{\mathrm{received},r}
```

如果请求 `r` 对应的来源、hash、用户或 license 状态无法映射到版本和 shard，就不能声称删除机制可靠。

权限违规率可以写成：

```math
R_{\mathrm{access}}=\frac{N_{\mathrm{denied\ or\ invalid}}}{N_{\mathrm{access}}}
```

这个指标不是越低越好：如果高风险数据被低权限角色访问成功，说明权限系统失效。审计时要同时看违规访问尝试是否被阻断。

数据文档完整率可以写成：

```math
C_{\mathrm{doc}}=\frac{N_{\mathrm{completed\ sections}}}{N_{\mathrm{required\ sections}}}
```

最终治理门禁可以写成：

```math
G_{\mathrm{gov}}=G_{\mathrm{manifest}}\land G_{\mathrm{lineage}}\land G_{\mathrm{license}}\land G_{\mathrm{access}}\land G_{\mathrm{delete}}\land G_{\mathrm{doc}}
```

其中任何一项失败，都意味着该数据版本不应直接进入正式训练或对外发布说明。

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

---

## 17. 面向专家：治理是模型行为控制面的底座

从专家视角看，dataset governance 是模型行为控制面的底座。

模型行为问题经常被归因于算法，但很多问题来自数据：某批安全数据过度保守、某个代码源污染评测、某个合成数据版本质量下降、某些低资源语言被误删、某个旧法规文本未更新。

如果没有数据版本和 lineage，团队只能在模型层面盲目调参。如果有治理系统，就可以追溯、回滚、重洗、重采样、隔离和审计。

治理的价值不是让流程变慢，而是让大规模模型迭代可控。

---

## 18. 一个可落地的数据版本与治理方案

如果面试官问：“如何设计大模型训练数据的 versioning 和 governance？”可以按下面回答。

第一步，建立统一数据 schema。每条样本记录 source、license、language、domain、quality、safety、PII、dedup、version 和 deletion_status。

第二步，建立不可变数据快照。每个训练数据版本有 manifest、hash、shard 列表、统计摘要和父版本。

第三步，记录 pipeline 版本。采集、解析、清洗、过滤、去重、质量评分、污染检测都要有代码和配置版本。

第四步，建立 lineage。支持从训练样本追溯到原始来源，也支持从来源追踪到哪些模型版本用过。

第五步，做权限分级。区分公开数据、授权数据、用户数据、敏感数据、评测集和高风险数据。

第六步，建立删除机制。支持按来源、hash、用户请求、license 状态定位和删除，并记录执行证明。

第七步，配套文档。为关键数据集写 datasheet，为模型发布提供 model card 所需的数据说明。

第八步，审计和监控。定期检查 license、PII、权限、污染、删除请求和数据版本可复现性。

第九步，训练集成。训练日志必须记录数据版本、mixture、sampler、tokenizer 和评测集版本。

### 18.1 最小可运行数据版本治理 demo

下面这个 demo 不依赖外部库，也不读写文件。输入是一组 toy 样本、访问请求、删除请求和文档完成状态；输出包括 manifest、可训练样本、license 分布、lineage 覆盖率、访问阻断、删除请求命中、datasheet 完成度、model card 输入是否完整和治理门禁。

它演示的是治理机制，不是真实 DVC、对象存储、权限系统、合规系统或生产数据平台。真实系统需要接入对象存储版本、数据目录、权限服务、审计日志、删除工单、dataset card / datasheet 文档和训练日志。

```python
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

deletion_compliant = all(
    item["deleted"] for req in deletion_requests for item in samples if item["source"] == req["source"]
)

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
    "deletion_compliant": deletion_compliant,
    "datasheet_completion": datasheet_completion,
    "model_card_ready": model_card_ready,
}

gates = {
    "manifest_checksums": all(row["checksum"] for row in manifest),
    "lineage_minimum": report["lineage_coverage"] >= 0.65,
    "blocked_bad_data": set(report["eligible_ids"]) == {"s1", "s2", "s5"},
    "access_controls": len(blocked_access) == 2,
    "deletion_traceable": deletion_hits["del-001"] == ["s6"],
    "deletion_compliant": deletion_compliant,
    "datasheet_ready": datasheet_completion >= 0.80,
    "model_card_ready": model_card_ready,
}
report["gates"] = gates
report["governance_ready"] = all(gates.values())

for key, value in report.items():
    print(f"{key}=", value)

assert report["eligible_ids"] == ["s1", "s2", "s5"]
assert report["lineage_coverage"] == 0.667
assert report["lineage_missing"] == ["s5", "s6"]
assert report["blocked_access"] == [("contractor", "s5", "internal"), ("researcher", "s4", "quarantine")]
assert report["deletion_hits"] == {"del-001": ["s6"], "del-002": []}
assert report["datasheet_completion"] == 0.833
assert report["governance_ready"] is False
```

运行后会看到类似输出：

```text
dataset_version= data-v2026-06-06.1
manifest= [{'shard': 'shard-a', 'sample_count': 2, 'tokens': 1130, 'checksum': '8ee90eff217c'}, {'shard': 'shard-b', 'sample_count': 2, 'tokens': 820, 'checksum': '83cc9bdbbafe'}, {'shard': 'shard-c', 'sample_count': 2, 'tokens': 1040, 'checksum': 'b5bf80f8852a'}]
eligible_ids= ['s1', 's2', 's5']
license_counts= {'blocked': 1, 'expired': 1, 'permissive': 3, 'review': 1}
lineage_coverage= 0.667
lineage_missing= ['s5', 's6']
blocked_access= [('contractor', 's5', 'internal'), ('researcher', 's4', 'quarantine')]
deletion_hits= {'del-001': ['s6'], 'del-002': []}
deletion_compliant= True
datasheet_completion= 0.833
model_card_ready= False
governance_ready= False
```

这个 demo 的重点是：`eligible_ids` 只能说明样本通过了训练准入门禁，不代表整个数据版本已经治理完备。这里 `model_card_ready=False`，说明发布文档所需的删除策略事实基础仍缺失，因此 `governance_ready=False` 是正确结果。

---

## 19. 常见面试题

### 19.1 为什么大模型数据需要 versioning？

因为训练结果依赖具体数据快照、清洗规则、采样权重和过滤版本。没有 versioning，模型不可复现，问题不可追溯，删除和审计也无法可靠执行。

### 19.2 什么是 data lineage？

data lineage 是数据从原始来源到最终训练样本的血缘记录，包括采集、解析、清洗、过滤、去重、标签、版本和使用去向。

### 19.3 manifest 有什么作用？

manifest 是数据版本清单，记录 shard、hash、样本数、token 数、统计分布、pipeline 版本和风险摘要，是训练可复现和审计的基础。

### 19.4 删除请求为什么难？

因为数据可能存在于原始数据、清洗数据、shard、缓存、索引、embedding、旧版本和已训练模型中。需要 lineage、hash、版本和处理记录支持。

### 19.5 Datasheet for Datasets 解决什么问题？

它为数据集提供标准化说明，记录动机、组成、采集、清洗、用途、限制和风险，提升透明度和责任意识。

### 19.6 数据治理和 model card 有什么关系？

model card 中关于训练数据、适用范围、风险、偏差和评估的信息，都需要数据治理提供事实基础。

### 19.7 如何设计数据权限？

按数据敏感度和用途分级，遵循最小权限原则，记录访问日志，审批高风险数据访问，定期审计权限，并保护评测集避免污染。

---

## 20. 常见误区

误区一：数据版本就是文件名加日期。

真正的数据版本包括数据快照、manifest、pipeline、配置、schema、过滤器、采样权重和审计记录。

误区二：训练完再补文档也可以。

很多信息训练后无法补齐，文档和 lineage 应嵌入 pipeline。

误区三：公开数据不需要治理。

公开可见不等于授权使用，也可能包含隐私、版权、偏见和删除请求风险。

误区四：删除就是删当前目录。

还要处理旧版本、缓存、索引、派生数据和训练记录。

误区五：治理会拖慢研发。

缺少治理才会在模型出问题时拖慢研发。好的治理让迭代可追溯、可回滚、可审计。

误区六：model card 可以脱离数据系统编写。

没有数据版本和统计，model card 很容易变成泛泛而谈。

---

## 21. 本章小结

Dataset Versioning 与 Governance 是大模型数据工程从“能训练”走向“能复现、能审计、能治理”的关键。

本章要记住几句话：

1. 大模型训练必须绑定不可变数据快照和 manifest。
2. Lineage 让样本来源、处理过程和模型使用去向可追溯。
3. 元数据 schema 是清洗、配比、权限、删除和审计的基础。
4. Datasheet 和 model card 是透明治理的重要文档形式。
5. 删除请求、权限控制、license 和隐私处理必须进入工程流程。
6. 治理不是研发阻力，而是大规模模型迭代的控制面。

如果面试中被问到数据治理，最好的回答是：不仅讲存储和版本号，还要讲 schema、manifest、lineage、pipeline 版本、权限、删除请求、审计、datasheet、model card 和训练可复现闭环。
