# 第五章：数据、模型与实验平台

## 5.1 实验平台的核心对象是证据链

“这次模型变好了”只有在输入、代码、配置、运行、评估和产物都能关联时才有工程意义。数据平台、模型仓库和实验追踪不是三个孤立的后台，它们共同维护一条证据链：

```text
数据集版本 -> 训练 run -> checkpoint -> 评估 run -> 发布包 -> 线上版本
```

如果只保存最终权重，读者无法知道它使用了哪份数据、哪个 tokenizer、什么超参和哪一版评估集；如果只保存实验指标而没有样本和错误分析，指标变化也无法解释。

## 5.2 数据版本和血缘

一个数据集版本至少应包含 schema、来源、过滤规则、去重版本、切分、统计摘要、许可证/权限信息和 manifest checksum。数据文件名不是版本，`latest` 也不是版本。

可以把一个样本集合表示为：

```math
D_v=(source, transform_{1:k}, split, manifest, policy)
```

`transform` 记录清洗、tokenize、采样和打包过程；`policy` 记录版权、PII、租户和用途限制。训练 run 只引用 `D_v` 的不可变 manifest，不能在运行期间静默改变文件内容。

血缘图要回答两个方向的问题：一个模型使用了哪些数据？一份敏感数据进入了哪些模型和评估？前者服务复现，后者服务回溯、删除请求和安全治理。

## 5.3 模型仓库不是文件目录

模型仓库需要管理权重、配置、tokenizer、chat template、adapter、量化产物、推理引擎兼容性和安全审批。一个模型版本的执行契约可以写成：

```text
model_revision + config_revision + tokenizer_revision
+ template_revision + adapter_revision + quantization_revision
```

缺少其中任一项，模型可能“能加载但行为不对”。例如 tokenizer 不匹配会改变 token 长度，template 不匹配会改变 role 边界，量化版本不兼容会让某些 kernel 退回慢路径，adapter 绑定错误会把一个租户的风格带进另一个服务。

模型仓库的发布状态应区分 `candidate`、`validated`、`staged`、`production`、`deprecated` 和 `revoked`。只有通过离线能力、安全、协议和资源测试的 artifact 才能进入下一状态。

## 5.4 Artifact manifest 和可复现 run

一个 run 的 manifest 可抽象为：

```math
R=(D,C,E,M,H,O)
```

`D` 是数据版本，`C` 是代码 revision，`E` 是训练/评估配置，`M` 是模型与 tokenizer 依赖，`H` 是硬件和运行环境，`O` 是产出 artifact。记录这些字段后，实验平台才能比较两个 run 是“只改变了一个变量”，还是实际上同时改变了数据、镜像和评估集。

最小 manifest 示例：

```json
{
  "run_id": "run-2026-08-06-001",
  "code_commit": "abc123",
  "dataset_manifest": "sha256:...",
  "base_model": "model-rev-17",
  "tokenizer": "tokenizer-rev-4",
  "config": "config-sha256:...",
  "hardware": "8xGPU",
  "outputs": ["checkpoint-step-12000", "eval-report-..." ]
}
```

示例中的 ID 只是结构示意；真正系统要校验 checksum、权限和提交时间，不能接受用户直接填入的任意字符串。

## 5.5 实验追踪和样本证据

指标曲线适合发现趋势，样本记录适合解释原因。一次评估 run 除了保存 accuracy、latency 和 cost，还应保存输入 hash、模型版本、prompt/template、输出、判定结果、错误类型和是否人工复核。

对生成任务，结果表可以使用下面的结构：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 评估样本稳定 ID |
| `input_hash` | 输入与媒体的去敏 hash |
| `model_revision` | 实际执行模型 |
| `harness_revision` | 评估脚本与 prompt 版本 |
| `output_ref` | 输出 artifact 或脱敏内容 |
| `judge` | 规则、人评或 judge 版本 |
| `label` | 正确、错误、拒答或不适用 |
| `error_type` | 检索、格式、事实、安全等切片 |

没有 `harness_revision` 时，同一个模型在不同 prompt、工具和采样设置下的分数不能直接比较。没有错误样本时，团队会反复优化平均分，却无法知道真实用户遇到的失败类型。

## 5.6 评估平台的分层

离线批评估适合回归和大规模对照，人评适合开放质量和偏好，在线评估适合真实行为、成本和长期留存。三者的样本、指标和延迟不同，不能用一个分数替代。

一个发布准入条件可以形式化为：

```math
Release=I(Q\ge\tau_q)\land I(Safety\le\tau_s)
\land I(Latency\le\tau_l)\land I(Cost\le\tau_c)
```

这是合取验收条件，不是把所有指标简单加权。某个模型平均质量提高，但安全违规率超过阈值，不能因为总分较高就发布。不同业务还应加入引用正确率、工具成功率、拒答质量、人工接管率或领域关键字段准确率。

## 5.7 Embedding store 和向量索引

Embedding store 保存的不是“一个向量就够了”。向量必须绑定文本或媒体版本、模型 revision、维度、归一化方式、权限、分块策略和更新时间。若 embedding 模型变了，旧向量和新向量通常不能直接混合比较。

向量检索的基础相似度可以写成：

```math
sim(q,d)=\frac{q\cdot d}{\|q\|\|d\|}
```

当向量已归一化时，cosine similarity 等于 inner product；但索引类型、量化、分片和过滤条件会改变实际召回。RAG/Agent 平台还要在检索前应用租户权限，在结果返回时保留 source ID 和版本，避免把无权文档或过期内容送进模型。

## 5.8 一个回归定位例子

某次发布后客服答案正确率下降 4 个百分点。实验平台应沿血缘反查：

1. 模型 revision 是否变化。
2. chat template 或 tokenizer 是否变化。
3. 评估集 manifest 是否变化。
4. RAG embedding model、chunk size 或权限过滤是否变化。
5. judge/harness 是否变化。
6. 错误是否集中在某个租户、语言、文档版本或问题类型。

如果只有“发布前 82%，发布后 78%”两条曲线，无法判断是模型退化、数据变化还是评估脚本错误。保存样本级证据后，可以把差异拆成输入变化、检索变化、模型输出变化和判定变化。

## 5.9 治理、隐私和清理

实验平台经常保存 prompt、用户文档、模型输出和错误日志，这些可能包含个人信息和商业秘密。应使用数据分级、字段脱敏、访问控制、保留期限、删除索引和访问审计。hash 不能自动消除隐私风险：原文仍可能出现在 artifact、日志或缓存中。

数据删除或撤回请求要能沿血缘找到派生数据、embedding、checkpoint、评估报告和线上缓存。能追踪并不等于一定能完成模型级遗忘，但至少可以明确影响范围和剩余风险。

## 5.10 从实验记录到因果结论

实验平台不应只保存最终分数，还要保存输入样本、模型/数据/代码 revision、随机种子、硬件、工具、评测器和失败 trace。改变多个因素的实验只能说明系统结果变化，不能证明某个模块因果有效。

数据、模型、评估和部署 artifact 之间需要 lineage，才能回答一个线上回归来自哪个数据版本或 checkpoint。

## 5.11 实验实体和 lineage

实验平台至少需要区分 Run、ModelVersion、DatasetVersion、Prompt/Template、Environment、Artifact 和 Evaluation。一个 run 的结果必须能回到输入数据、代码、权重、配置、硬件和 grader。

如果只保存一个 accuracy 数字，后续无法判断 tokenizer、prompt、模型 revision、随机种子或评估器是否变化。元数据 schema 本身是可复现性的基础设施。

## 5.12 artifact 和指标的可信度

日志、checkpoint、评估输出、trace 和模型卡都属于 artifact。平台要给 artifact hash、来源、访问权限、保留期限和状态；指标要带样本集、聚合方式、置信区间和失败样本链接。

派生指标不能掩盖原始事实。一个总分提高时，读者仍应能查看分桶结果、bad case、输入输出和评估版本。

## 5.13 从实验到发布的验收条件

发布前把训练、评估、模型仓库和 serving 的 lineage 连接起来：数据通过治理，模型通过质量/安全，runtime 通过加载和性能，发布通过权限和回滚。任何缺失都进入人工审核或阻断。

线上反馈还要回写为匿名化 bad case、回归集和数据版本，避免只追求一次实验的漂亮结果。

## 5.14 实验对象与证据链

实验平台至少要区分 dataset version、run、checkpoint、adapter、eval report、deployment package 和 incident。每个对象有 immutable id、producer、输入依赖、checksum、权限、生命周期和状态。仅保存一个“最好分数”无法回答这个分数来自哪个 prompt、哪份数据、哪个代码 commit 和哪种评估脚本。

可以把一条证据链表示为：

~~~text
dataset_snapshot
  -> training_run
  -> checkpoint
  -> evaluation_run
  -> release_candidate
  -> production_observation
~~~

箭头不是简单的文件引用，而是带版本、时间和权限的 lineage edge。数据删除、模型回滚或评估重跑时，平台应能沿图查询影响范围。

## 5.15 用因果问题组织实验

实验追踪不等于指标收集。一个实验应该先写假设、干预变量、对照、主要指标、护栏指标和停止条件。例如要判断 LoRA rank 是否带来收益，不能只比较两个最终分数，还要固定 base checkpoint、数据、训练步数和评估 prompt，并记录 rank、学习率、adapter merge 和显存。

若主指标为 Y，干预为 X，最小差异可写成：

~~~math
\Delta
=\mathbb{E}[Y\mid X=1]-\mathbb{E}[Y\mid X=0].
~~~

真实实验还要考虑 seed、样本切片、评估方差和成本。平台应保存 paired sample、错误标签和原始输出，使“提升”可以被复核，而不是只保留聚合均值。

## 5.16 数据、模型和评估的可复现性

数据版本需要记录内容 checksum、schema、过滤规则、去重规则、split、权限和时间快照；模型需要记录权重、config、tokenizer、template、adapter、量化和 runtime；评估需要记录样本、prompt、解码参数、judge 版本、后处理和指标定义。缺少任一层，复现结果都可能漂移。

对于在线检索和向量索引，还要保存 embedding model、chunker、metadata filter、index build revision 和 point-in-time 时间。否则同一文档在不同时间重建出的召回结果可能不同，实验平台却无法解释。

## 5.17 样本级证据和隐私边界

生产 trace 能帮助发现回归，但原始 prompt、检索内容、工具参数和媒体可能包含敏感数据。平台要支持字段级脱敏、采样、访问审批、retention 和删除，而不是把所有原文永久写入日志。

样本级审计可以保存：

~~~text
sample_hash
task_slice
model_revision
prompt_revision
retrieved_source_ids
prediction_hash
error_labels
review_status
~~~

只有经过权限校验的角色才能读取原文。哈希、来源 id 和错误标签通常足以支持回归定位；需要复核时再通过受控方式取回原始样本。

## 5.18 血缘查询：从线上回归追到样本和版本

设线上版本 `model-r42` 的引用支持率下降，平台不能只查询模型文件。它需要沿着 `production_observation -> release -> evaluation_run -> model_artifact -> training_run -> dataset_version` 逆向遍历，并同时查询 prompt、retriever、embedding、tool schema 和 judge 版本。

一个可审计的 lineage edge 至少包含：

```text
from_id, to_id, relation
created_at, actor, revision, input_hash
policy_scope, deletion_status
```

若回归集中在某个文档版本，平台可以定位到具体 source；若只集中在某个模板 revision，模型权重可能无需回滚；若评估器本身升级，先重跑旧样本和旧 grader，才能判断质量变化是真实的。血缘图的价值是缩小因果搜索空间，而不是把所有文件贴在一个页面上。

## 5.19 评估样本、聚合指标与不确定性

聚合指标要能回到样本级。每个样本记录预测、标签、引用、错误类型、是否拒答、耗时和成本；聚合层再按任务、语言、租户、长度、风险和数据新旧切片。一个整体平均分上升，可能是简单样本增加而关键长尾样本变差。

对于比例指标，可以记录样本数和置信区间，而不是只保存一个百分比：

```math
\hat p=\frac{k}{n},
\qquad
SE(\hat p)\approx\sqrt{\frac{\hat p(1-\hat p)}{n}}.
```

这是近似的不确定性说明，不替代针对分层、相关样本和人工评分的正式方法。评估平台还要记录停用规则、缺失样本和重试样本，避免失败请求从分母中消失而虚高质量。

## 5.20 从 candidate 到 production 的证据验收条件

发布候选要有一条明确的状态机：`built -> validated -> staged -> canary -> production -> deprecated/revoked`。每次状态变化绑定 actor、策略、输入 artifact 和证据报告。`validated` 不是“所有任务都正确”，而是通过预先声明的质量、安全、协议、成本和资源验收条件。

可以把发布条件写成：

```math
G_{\mathrm{release}}
=G_{\mathrm{lineage}}
 G_{\mathrm{quality}}
 G_{\mathrm{safety}}
 G_{\mathrm{protocol}}
 G_{\mathrm{resource}}
 G_{\mathrm{rollback}}
```

数据撤回或权限变化时，`G_lineage` 要能列出受影响模型、embedding、索引、评估和缓存；如果无法确定影响范围，候选不能继续扩大流量。回滚时也要保留新版本已产生的线上 trace 和外部 artifact，不能让版本切换抹掉证据。

## 5.21 实验到发布的基础验收条件

发布候选应通过数据血缘、模型 artifact、离线评估、切片回归、安全检查、成本/延迟压测和回滚准备。MLflow、OpenTelemetry、模型仓库、向量索引和数据版本系统可以组成实现，但不能替代实体契约。

数据、模型和实验平台的共同目标是让每个结论都有可追溯证据，让每次变更都有可比较基线；同时让敏感数据的访问、删除和导出都可审计。公开工具文档说明接口能力，具体治理强度仍需结合组织权限和业务合规验证。

## 5.22 先把实验对象建模清楚

实验平台至少需要区分 dataset、model artifact、prompt/template、retriever/index、tool schema、policy、evaluation suite、run、sample result 和 release。它们之间是有向血缘关系，不是都挂在一个 `experiment_id` 下的字符串。

例如一次 RAG 回归可能固定模型权重，只更新 embedding 和 index；一次 prompt 实验可能固定模型和数据，只替换模板；一次安全发布可能不改权重，却改了 policy 和工具权限。如果实体没有独立版本，平台会把所有变化归因给模型，得出错误结论。

每个 artifact 应有不可变 revision、内容 hash、owner、license、敏感等级、创建时间、父依赖和失效/删除状态。`latest` 只能作为人类界面别名，不能作为评估或发布的输入。

## 5.23 Dataset snapshot 与 point-in-time 语义

数据集版本不只是文件路径。它还要记录采集时间、过滤规则、去重规则、标签版本、split、采样权重、授权和删除请求。在线特征、检索索引和知识库还需要 point-in-time 语义，否则训练/评估可能读取未来信息。

一个可复现的样本 manifest 可以包含：

```text
sample_id, source_id, snapshot_time, split
content_hash, label_revision, filter_revision
permission_scope, deletion_status, lineage_parent
```

当某个源被撤回时，平台应能查询哪些 dataset、embedding、index、eval run 和 release 受影响；若无法查询，就不能声称数据治理是可追溯的。

## 5.24 评估运行要防止分母漂移

评估系统常见的虚高来自失败样本被重试、超时样本被删除、缺失标签不计入分母或只报告成功请求。每个 run 都应保存计划样本数、实际开始数、成功数、失败数、跳过原因、重试次数和最终分母。

对比例指标，至少保留 `k/n` 而不是只保留百分比；对 paired experiment，保留同一 `sample_id` 在 baseline 和 candidate 的两条结果；对人工评估，保留标注规范、评审员、校准和仲裁记录。这样才能区分模型真的提升，还是样本构成和执行成功率改变。

## 5.25 隐私、删除和可复现性并不矛盾

实验可复现不等于永久保存所有原文。平台可以把内容与身份分离：用 content hash、source id、样本标签和脱敏特征支持大多数回归；原文放在受控存储中，按租户、用途和 retention 访问。删除请求发生时，删除原文和派生 artifact，同时保留不含敏感内容的审计事件。

如果评估必须回看原文，应记录访问主体、目的、审批、时间和导出范围。日志中的 prompt、工具参数和媒体不能因为“为了复现”就无限保留；否则实验平台本身会成为数据泄露面。

## 5.26 从实验到发布的最小证据链

一个候选 release 至少要能反向查询：它使用了哪个模型/adapter、tokenizer/template、数据 snapshot、retriever/index、tool schema、policy、评估 suite、硬件/runtime 和成本配置。正向查询还要能列出它产生的线上请求、错误样本、工具动作和事故。

发布准入条件可以形式化为：

```math
G=G_{\mathrm{artifact}}G_{\mathrm{lineage}}G_{\mathrm{eval}}
 G_{\mathrm{privacy}}G_{\mathrm{security}}G_{\mathrm{rollback}}.
```

只有当血缘、质量、权限、隐私和回滚都可验证时，候选才从研究状态进入生产状态。平台的价值不是把实验结果画成漂亮曲线，而是让曲线能被重新计算、被解释，并在问题发生时支持最小范围的回滚。

## 5.27 小结

实验平台的核心对象是带版本和血缘的证据链：数据 snapshot、模型 artifact、评估样本、运行环境、指标分母和发布结果必须能够互相追溯。隐私删除不能靠永久保留原文解决，可用受控访问、hash、脱敏和派生 artifact 支持复现；发布条件还要把质量、安全、权限和回滚作为同一条链验收。
