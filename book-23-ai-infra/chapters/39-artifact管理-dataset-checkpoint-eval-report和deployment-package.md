# 第 39 章 Artifact 管理：dataset、checkpoint、eval report 和 deployment package

上一章讲了模型仓库。本章讲一个更通用的概念：Artifact 管理。

在 AI Infra 中，训练、评估、部署、回滚、审计都围绕各种产物流转。dataset、checkpoint、eval report、deployment package、日志、配置、模型权重都可以看作 artifact。

先记住一句话：

> Artifact 管理的目标，是让 AI 研发过程中的每个关键产物都可定位、可校验、可复现、可追溯、可治理。

## 39.0 本讲资料边界与第二轮精修口径

本讲按截至 2026-06 的稳定公开资料校准：MLflow Tracking 对 run、artifact store、backend store、参数、指标和 artifact 的实验追踪抽象，Weights & Biases Artifacts 对数据集、模型、文件集合、版本和 lineage 的产物管理口径，OpenLineage 对 Job、Run、Dataset 和事件血缘的抽象，Amazon S3 object integrity 对 checksum 的对象完整性口径，PyTorch Distributed Checkpoint 对分布式 checkpoint 保存 / 加载的状态管理边界，以及 OCI image / manifest 对 deployment package 中容器镜像与内容寻址的工程边界。

本章只抽象大模型平台里的通用 artifact 管理能力，不绑定某个实验追踪产品、对象存储、镜像仓库、模型仓库、评估平台或内部发布系统。

和上一章的分工是：

1. 第 38 章聚焦模型仓库，回答“权重、tokenizer、config、adapter 和量化版本如何作为模型资产被管理”。
2. 本章扩展到更通用的 artifact 管理，回答“dataset、checkpoint、eval report、deployment package、日志、配置和 release manifest 如何被统一注册、校验、追踪、晋级、回滚和清理”。

## 39.1 什么是 Artifact

Artifact 可以理解为一次计算过程产生或依赖的可保存产物。

在 AI Infra 中常见 artifact 包括：

1. dataset。
2. dataset manifest。
3. checkpoint。
4. model weights。
5. tokenizer。
6. config。
7. adapter。
8. eval report。
9. benchmark report。
10. deployment package。
11. container image。
12. training log。
13. trace。
14. experiment summary。

Artifact 不是简单文件，它还应该带有元数据、版本、校验和血缘。

## 39.2 为什么 Artifact 管理重要

没有 Artifact 管理，会出现这些问题：

1. 不知道某个 checkpoint 是哪个训练任务产出的。
2. 不知道某个模型用了哪个 dataset。
3. eval report 和模型版本对不上。
4. 部署包里 tokenizer 和权重不匹配。
5. 线上出问题无法回滚。
6. 存储里堆满没人敢删的文件。
7. 审计时无法证明模型来源。

Artifact 管理是 AI 工程可复现性的基础。

## 39.3 Artifact 的基本元数据

一个 artifact 至少要有：

1. artifact ID。
2. artifact type。
3. name。
4. version。
5. storage URI。
6. checksum。
7. size。
8. created time。
9. created by。
10. producer job。
11. input dependencies。
12. permissions。
13. lifecycle state。
14. tags。

有了这些字段，平台才能进行查询、校验、权限控制和生命周期管理。

一个 artifact 记录可以抽象为：

```math
A_i=(u_i,\tau_i,n_i,v_i,r_i,h_i,s_i,o_i,d_i,p_i,l_i,g_i,e_i,z_i)
```

其中 `u_i` 是 artifact id，`tau_i` 是 artifact type，`n_i` 是名称，`v_i` 是版本，`r_i` 是存储 URI，`h_i` 是 checksum 或 digest，`s_i` 是大小，`o_i` 是 owner / producer job，`d_i` 是输入依赖，`p_i` 是权限策略，`l_i` 是 lifecycle state，`g_i` 是 lineage 事件，`e_i` 是评估、发布或审计证据，`z_i` 是 tag 和审计日志。

元数据完整率可以写成：

```math
C_{\mathrm{meta}}=\frac{K_{\mathrm{filled}}}{K_{\mathrm{required}}}
```

其中 `K_filled` 是已填写的必需字段数，`K_required` 是该 artifact type 的必需字段总数。这个公式的直觉是：artifact 不是“有路径就够”，而是要有足够证据支撑复现、权限、血缘、回滚和删除判断。

## 39.4 Artifact 类型

常见类型可以分为几类。

数据类：

1. raw dataset。
2. cleaned dataset。
3. dataset manifest。
4. annotation file。
5. synthetic data。

模型类：

1. checkpoint。
2. final weights。
3. adapter。
4. tokenizer。
5. model config。
6. quantized weights。

评估类：

1. eval report。
2. benchmark result。
3. safety report。
4. human review result。

部署类：

1. deployment package。
2. container image。
3. runtime config。
4. release manifest。

## 39.5 Dataset Artifact

Dataset artifact 是训练和评估的输入。

它不一定是数据本体，也可以是 manifest。

Dataset artifact 应记录：

1. dataset name。
2. version。
3. split。
4. file list。
5. schema。
6. sample count。
7. token count。
8. source composition。
9. cleaning rule version。
10. dedup rule version。
11. quality metrics。
12. permissions。
13. checksum。

训练任务引用 dataset artifact，而不是裸路径。

Dataset manifest 完整率可以写成：

```math
C_{\mathrm{dataset}}=\frac{N_{\mathrm{manifest,filled}}}{N_{\mathrm{manifest,required}}}
```

其中 `N_manifest,filled` 是 dataset manifest 中已填写的必要字段数，`N_manifest,required` 是 dataset artifact 发布前必须记录的字段总数。对训练输入来说，manifest 比目录路径更重要，因为它冻结了文件集合、schema、样本数、token 数、来源组成、规则版本和权限快照。

## 39.6 Checkpoint Artifact

Checkpoint artifact 是训练过程中保存的模型状态。

它可能包含：

1. model weights。
2. optimizer state。
3. scheduler state。
4. dataloader state。
5. random state。
6. training step。
7. config。
8. distributed training metadata。

Checkpoint 有两种用途：

1. 断点恢复。
2. 作为候选模型版本。

两者要求不同。

断点恢复 checkpoint 需要 optimizer 和随机状态。

候选模型 checkpoint 更关注权重、tokenizer、config 和评估。

## 39.7 Checkpoint 元数据

Checkpoint artifact 应记录：

1. training job ID。
2. global step。
3. epoch。
4. loss。
5. learning rate。
6. dataset version。
7. code version。
8. config version。
9. parent checkpoint。
10. file shards。
11. checksum。
12. save time。
13. retention policy。

这样才能判断某个 checkpoint 是否可恢复、是否值得评估、是否可以清理。

Checkpoint 恢复完整率可以写成：

```math
C_{\mathrm{ckpt}}=\frac{N_{\mathrm{state,present}}}{N_{\mathrm{state,required}}}
```

其中 `N_state,present` 是 checkpoint 中实际保存的状态项数，`N_state,required` 是恢复训练所需状态项数。用于断点恢复时，模型权重、optimizer、scheduler、dataloader cursor、随机数状态和分布式 metadata 都应进入检查；用于候选模型时，权重、tokenizer、config、eval report 和 release metadata 更关键。

## 39.8 Eval Report Artifact

Eval report artifact 是模型评估结果。

它应绑定：

1. model version 或 checkpoint。
2. eval dataset version。
3. eval code version。
4. metric definitions。
5. prompt template version。
6. inference config。
7. runtime version。
8. result summary。
9. sample-level outputs。
10. failure cases。
11. evaluator version。

如果 eval report 不记录这些信息，评估结果不可复现。

同一个模型用不同 prompt、不同 temperature、不同 eval 集，分数可能完全不同。

Eval report 可复现门禁可以写成：

```math
G_{\mathrm{eval}}=\mathbf{1}[h_{\mathrm{model}}=h_{\mathrm{model,expected}} \land h_{\mathrm{data}}=h_{\mathrm{data,expected}} \land h_{\mathrm{code}}=h_{\mathrm{code,expected}} \land h_{\mathrm{prompt}}=h_{\mathrm{prompt,expected}}]
```

其中四个 hash 分别对应模型、评估数据、评估代码和 prompt / template。这个门禁强调：eval report 不是一个孤立分数，而是一组可复现实验条件和样本级证据。

## 39.9 Deployment Package Artifact

Deployment package 是部署推理服务所需的一组文件和配置。

它可以包含：

1. model weights。
2. tokenizer。
3. model config。
4. runtime config。
5. container image。
6. health check config。
7. default generation config。
8. safety policy。
9. route config。
10. release manifest。

部署包的目标是保证线上部署可复现。

不要让线上服务临时拼装权重、tokenizer 和配置。

Deployment package 完整率可以写成：

```math
C_{\mathrm{release}}=\frac{N_{\mathrm{artifact,present}}}{N_{\mathrm{artifact,required}}}
```

其中 `N_artifact,present` 是部署包中实际存在并通过校验的 artifact 数，`N_artifact,required` 是上线所需 artifact 总数。生产部署通常要求权重、tokenizer、model config、runtime config、image digest、health check、generation config、安全策略、route policy 和 release manifest 都可定位、可校验。

## 39.10 Release Manifest

Release manifest 描述一次发布使用了哪些 artifact。

例如：

```text
release: chat-model-prod-2026-05-30
model_version: chat-model:v1.3.0
weights: artifact://weights/abc
tokenizer: artifact://tokenizer/def
runtime_image: registry/runtime:v0.8.1
runtime_config: artifact://config/ghi
safety_policy: policy:v12
route_policy: route:v7
eval_report: eval:v1.3.0-prod-gate
```

有了 release manifest，回滚和审计都会简单很多。

## 39.11 Artifact 血缘

Artifact 之间应该有依赖关系。

例如：

```text
dataset:v1 + config:v3 + code:commit_x
  -> training_job:123
  -> checkpoint:step_10000
  -> eval_report:abc
  -> model_version:v2
  -> deployment_package:p1
  -> deployment:prod
```

这条链路就是 artifact lineage。

它能帮助你从线上部署追溯到训练数据，也能从某个数据问题找到受影响的模型。

Artifact 血缘图可以写成：

```math
G_{\mathrm{art}}=(V_{\mathrm{art}},E_{\mathrm{dep}})
```

其中 `V_art` 是 artifact、job、run、deployment 等节点集合，`E_dep` 是输入、输出、派生、评估、发布和部署依赖边。血缘覆盖率可以写成：

```math
C_{\mathrm{lineage}}=\frac{N_{\mathrm{linked}}}{N_{\mathrm{expected}}}
```

其中 `N_linked` 是已经能追踪的依赖节点或边数量，`N_expected` 是按产物类型应追踪的依赖数量。

## 39.12 Artifact Store 和 Metadata Store

Artifact 管理通常分成两层：

1. Artifact Store：存大文件。
2. Metadata Store：存元数据和索引。

Artifact Store 可以是对象存储、共享文件系统、镜像仓库或专用模型存储。

Metadata Store 存：

1. artifact ID。
2. 类型。
3. 版本。
4. URI。
5. checksum。
6. 依赖关系。
7. 权限。
8. 生命周期状态。

不要试图把大模型权重直接塞进数据库。

## 39.13 Checksum 和完整性校验

Artifact 必须支持完整性校验。

原因：

1. 大文件传输可能失败。
2. 分片文件可能缺失。
3. 存储可能出现不一致。
4. 部署前需要确认文件没被篡改。
5. 审计需要证明 artifact 未变化。

常见做法：

1. 每个文件有 checksum。
2. manifest 有整体 checksum。
3. 上传完成后校验。
4. 下载或部署前校验。
5. 校验失败则拒绝使用。

checksum 是 AI 资产可信的底线。

文件级 checksum 覆盖率可以写成：

```math
C_{\mathrm{checksum}}=\frac{N_{\mathrm{checked}}}{N_{\mathrm{files,total}}}
```

其中 `N_checked` 是已经完成 checksum 校验的文件或 shard 数，`N_files,total` 是 manifest 中声明的文件总数。大文件、分片 checkpoint、deployment package 和 eval sample outputs 都应该进入这个口径。

## 39.14 Artifact 权限

Artifact 权限和数据权限、模型权限一样重要。

权限问题包括：

1. 谁能读取 dataset artifact。
2. 谁能下载 checkpoint。
3. 谁能查看 eval sample outputs。
4. 谁能部署 deployment package。
5. 谁能删除 artifact。
6. 哪些 artifact 不能跨租户使用。

Eval report 中的 sample-level outputs 可能包含敏感 prompt 或模型输出，也需要权限控制。

## 39.15 Artifact 生命周期

Artifact 生命周期可以是：

```text
created -> validated -> active -> deprecated -> archived -> deleted
```

不同 artifact 生命周期策略不同。

Dataset：生产数据版本需要长期保留。

Checkpoint：中间 checkpoint 可以按策略清理。

Eval report：与发布相关的报告要保留。

Deployment package：线上和可回滚版本必须保留。

删除前要做依赖检查。

## 39.16 Retention Policy

保留策略可以按类型制定。

示例：

1. production deployment package 永久保留或长期保留。
2. 最近 5 个 production model version 保留。
3. 每个训练任务保留 best checkpoint 和 latest checkpoint。
4. 中间 checkpoint 保留 30 天。
5. failed job artifact 保留 7 天。
6. 安全审计相关 artifact 冻结删除。

没有 retention policy，存储成本会不可控。

策略太激进，又可能破坏复现和回滚能力。

存储保留成本可以粗略写成：

```math
K_{\mathrm{store}}=\sum_{a=1}^{A} S_a P_a T_a+K_{\mathrm{request}}+K_{\mathrm{egress}}+K_{\mathrm{ops}}
```

其中 `S_a` 是第 `a` 类 artifact 的存储大小，`P_a` 是单位存储价格，`T_a` 是保留时长，后面三项分别是请求、跨区传输和运维成本。这个公式说明 retention policy 是工程和成本问题，不是“全部保留”或“按时间删除”二选一。

删除安全门禁可以写成：

```math
G_{\mathrm{delete}}=\mathbf{1}[N_{\mathrm{dependents}}=0 \land T_{\mathrm{age}}\ge T_{\mathrm{retention}} \land H_{\mathrm{legal}}=0 \land A_{\mathrm{audit}}=1]
```

其中 `N_dependents` 是依赖该 artifact 的下游对象数量，`T_age` 是 artifact 年龄，`T_retention` 是保留策略要求，`H_legal` 是合规或审计冻结标记，`A_audit` 是删除审计记录是否完整。

## 39.17 Artifact Promotion

Artifact promotion 是把产物从低级状态提升到高级状态。

例如：

```text
checkpoint -> candidate model -> evaluated model -> release candidate -> production package
```

每次 promotion 都应有门禁：

1. checkpoint 完整性校验。
2. eval 通过。
3. safety 通过。
4. runtime 兼容性通过。
5. 审批通过。
6. 发布状态更新。

Promotion 让产物流转有流程，而不是谁都能拿一个 checkpoint 去部署。

Artifact 晋级门禁可以写成：

```math
G_{\mathrm{promo}}=\mathbf{1}[\min_j C_j\ge\tau_j \land C_{\mathrm{checksum}}=1 \land C_{\mathrm{lineage}}\ge\rho \land R_{\mathrm{permission}}=0 \land P_0=0]
```

其中 `C_j` 是每个 artifact 检查维度的覆盖率，`tau_j` 是阈值，`R_permission` 是权限违规率，`P0` 是未解决的 P0 风险数量。这个门禁适用于 checkpoint 晋级为候选模型、eval report 晋级为发布证据、deployment package 晋级为生产发布包。

## 39.18 Artifact 和实验追踪的关系

实验追踪记录一次实验的参数、指标和日志。

Artifact 管理记录实验产生和依赖的产物。

二者应该关联。

例如一个 experiment run 关联：

1. 输入 dataset artifact。
2. 输入 config artifact。
3. 输出 checkpoint artifact。
4. 输出 eval report artifact。
5. 输出 logs artifact。

这样才能从实验结果直接定位产物。

## 39.19 Artifact 和回滚

回滚依赖 artifact 可用。

如果旧版本 deployment package 被删除，回滚就会失败。

回滚需要：

1. 旧模型权重。
2. 旧 tokenizer。
3. 旧 runtime image。
4. 旧配置。
5. 旧安全策略。
6. 旧路由策略。
7. 旧 release manifest。

所以 production artifact 不能随便清理。

## 39.20 Artifact 命名和路径规范

命名规范能降低混乱。

示例：

```text
artifact://model/chat-model/v1.3.0/weights
artifact://dataset/pretrain-mix/2026-05-30-v1/manifest
artifact://eval/chat-model/v1.3.0/safety-eval-2026-05-30
artifact://release/chat-model/prod/2026-05-30-r1
```

路径中应体现类型、名称、版本和用途。

但不要只依赖路径表达元数据，核心元数据仍要进入 metadata store。

## 39.21 Artifact 管理系统架构

一个 Artifact 管理系统可以这样设计：

```text
Artifact API
  -> Metadata DB
  -> Artifact Store
  -> Manifest Store
  -> Checksum Service
  -> Lineage Graph
  -> Access Control
  -> Lifecycle Manager
  -> Audit Log
```

Artifact API 提供注册、查询、下载、校验、promote、archive 和删除接口。

Lifecycle Manager 负责 retention 和依赖检查。

## 39.22 Artifact 管理审计指标和最小 demo

Artifact 管理的最终门禁可以写成：

```math
G_{\mathrm{artifact}}=\mathbf{1}[\min_j C_j\ge\tau_j \land C_{\mathrm{checksum}}=1 \land C_{\mathrm{lineage}}\ge\rho \land C_{\mathrm{release}}=1 \land G_{\mathrm{delete}}=1 \land P_0=0]
```

其中 `C_j` 是各类 artifact 的审计覆盖率，`C_checksum` 是文件级完整性校验，`C_lineage` 是血缘覆盖率，`C_release` 是 deployment package / release manifest 完整率，`G_delete` 是删除安全门禁，`P0` 是未解决的高风险问题。这个门禁强调：artifact 管理不是文件系统整理，而是训练、评估、发布、回滚、审计和成本治理的共同基础。

下面是一个 0 依赖 toy demo。它把 dataset、checkpoint、eval report、deployment package 和 release manifest 放进同一张审计表，并构造 16 个 bad case，覆盖元数据缺失、dataset manifest 错误、checkpoint 不可恢复、eval report 不可复现、部署包缺文件、release manifest 可变、checksum 未验证、血缘断边、metadata / blob 分层错误、权限放开、retention 缺失、promotion 缺门禁、实验追踪未绑定、回滚 artifact 缺失、删除依赖不安全和最终 artifact gate 缺失。

```python
from copy import deepcopy
from pprint import pprint


REQUIRED_ARTIFACT_FIELDS = [
    "artifact_id",
    "artifact_type",
    "name",
    "version",
    "uri",
    "checksum",
    "size_gib",
    "created_at",
    "created_by",
    "producer_job",
    "dependencies",
    "permissions",
    "lifecycle_state",
    "tags",
]

REQUIRED_RELEASE_REFS = {
    "weights",
    "tokenizer",
    "model_config",
    "runtime_config",
    "runtime_image",
    "safety_policy",
    "route_policy",
    "eval_report",
    "deployment_package",
}

REQUIRED_LINEAGE_NODES = {
    "dataset_artifact",
    "config_artifact",
    "training_job",
    "checkpoint_artifact",
    "eval_job",
    "eval_report_artifact",
    "model_version",
    "deployment_package",
    "release_manifest",
    "deployment",
}

REQUIRED_LINEAGE_EDGES = [
    ("dataset_artifact", "training_job"),
    ("config_artifact", "training_job"),
    ("training_job", "checkpoint_artifact"),
    ("checkpoint_artifact", "eval_job"),
    ("eval_job", "eval_report_artifact"),
    ("eval_report_artifact", "model_version"),
    ("model_version", "deployment_package"),
    ("deployment_package", "release_manifest"),
    ("release_manifest", "deployment"),
]


def complete_case():
    return {
        "name": "complete_case",
        "artifact": {
            "artifact_id": "artifact_release_chat_v120",
            "artifact_type": "deployment_package",
            "name": "chat-model-prod-package",
            "version": "2026-06-01-r1",
            "uri": "artifact://release/chat-model/prod/2026-06-01-r1",
            "checksum": "sha256:release-package",
            "size_gib": 18.0,
            "created_at": "2026-06-01T10:00:00Z",
            "created_by": "release-bot",
            "producer_job": "release_job_009",
            "dependencies": ["weights", "tokenizer", "runtime_image", "eval_report"],
            "permissions": ["assistant-prod", "model-platform"],
            "lifecycle_state": "active",
            "tags": ["prod", "chat", "rollback-ready"],
        },
        "dataset": {
            "dataset_id": "pretrain_mix",
            "version": "2026-06-01-v2",
            "split": "train",
            "schema_version": "schema_v3",
            "sample_count": 2000,
            "token_count": 1320000,
            "source_composition": {"web": 0.5, "code": 0.3, "books": 0.2},
            "cleaning_rule_version": "clean_v17",
            "dedup_rule_version": "dedup_v9",
            "quality_metrics": {"schema_valid_rate": 0.9995, "duplicate_rate": 0.02},
            "permissions": {"classification": "internal", "allowed_purpose": ["training"]},
            "checksum": "sha256:dataset-manifest",
            "files": [
                {"path": "shard-000.parquet", "checksum": "sha256:ds0", "samples": 1000, "tokens": 700000},
                {"path": "shard-001.parquet", "checksum": "sha256:ds1", "samples": 1000, "tokens": 620000},
            ],
        },
        "checkpoint": {
            "training_job_id": "train_042",
            "global_step": 12000,
            "epoch": 2,
            "loss": 1.92,
            "learning_rate": 0.00012,
            "dataset_version": "pretrain_mix:2026-06-01-v2",
            "code_version": "commit_abc123",
            "config_version": "cfg_v8",
            "parent_checkpoint": "ckpt_11000",
            "model_state": True,
            "optimizer_state": True,
            "scheduler_state": True,
            "dataloader_state": True,
            "rng_state": True,
            "distributed_metadata": True,
            "shards": ["rank0.pt", "rank1.pt"],
            "checksums": {"rank0.pt": "sha256:c0", "rank1.pt": "sha256:c1"},
            "restore_tested": True,
            "purpose": ["resume", "candidate_model"],
        },
        "eval_report": {
            "model_ref": "checkpoint:train_042:12000",
            "eval_dataset_version": "eval_chat_2026_06",
            "eval_code_version": "eval_commit_777",
            "metric_definitions": ["chat_quality", "tool_call", "safety", "latency"],
            "prompt_template_version": "prompt_v5",
            "inference_config": {"temperature": 0.0, "max_tokens": 512},
            "runtime_version": "vllm_0_9",
            "result_summary": {"chat_quality": 0.86, "tool_call": 0.91, "safety": 0.99, "latency": 0.94},
            "sample_outputs_uri": "artifact://eval/chat/v120/samples",
            "failure_cases": ["tool_arg_schema_edge"],
            "evaluator_version": "judge_v4",
            "report_checksum": "sha256:eval-report",
        },
        "deployment_package": {
            "weights": "artifact://model/chat/v120/weights",
            "tokenizer": "artifact://model/chat/v120/tokenizer",
            "model_config": "artifact://model/chat/v120/config",
            "runtime_config": "artifact://runtime/chat/v120/config",
            "container_image_digest": "sha256:runtime-image-v120",
            "health_check": "artifact://release/chat/v120/health",
            "generation_config": "artifact://release/chat/v120/generation",
            "safety_policy": "policy:v12",
            "route_config": "route:v7",
            "release_manifest": "artifact://release/chat/prod/2026-06-01-r1/manifest",
        },
        "release_manifest": {
            "release_id": "chat-model-prod-2026-06-01-r1",
            "artifact_refs": {
                "weights": "artifact://model/chat/v120/weights",
                "tokenizer": "artifact://model/chat/v120/tokenizer",
                "model_config": "artifact://model/chat/v120/config",
                "runtime_config": "artifact://runtime/chat/v120/config",
                "runtime_image": "sha256:runtime-image-v120",
                "safety_policy": "policy:v12",
                "route_policy": "route:v7",
                "eval_report": "artifact://eval/chat/v120/report",
                "deployment_package": "artifact://release/chat/prod/2026-06-01-r1",
            },
            "checksum": "sha256:release-manifest",
            "immutable": True,
            "rollback_target": "chat-model-prod-2026-05-25-r3",
        },
        "integrity": {
            "checksums_verified": True,
            "manifest_checksum_verified": True,
            "download_verify_required": True,
            "failed_files": [],
        },
        "lineage": {
            "nodes": sorted(REQUIRED_LINEAGE_NODES),
            "edges": list(REQUIRED_LINEAGE_EDGES),
            "events": [
                {"job": "train", "run_id": "train_042", "inputs": ["dataset_artifact", "config_artifact"], "outputs": ["checkpoint_artifact"]},
                {"job": "eval", "run_id": "eval_120", "inputs": ["checkpoint_artifact"], "outputs": ["eval_report_artifact"]},
                {"job": "package", "run_id": "release_job_009", "inputs": ["model_version", "eval_report_artifact"], "outputs": ["deployment_package"]},
            ],
        },
        "store": {
            "artifact_store_for_blobs": True,
            "metadata_store_for_index": True,
            "large_blob_in_metadata_db": False,
            "indexed_fields": ["artifact_id", "artifact_type", "version", "checksum", "lifecycle_state", "owner"],
        },
        "access": {
            "rbac_enabled": True,
            "sample_outputs_restricted": True,
            "cross_tenant_blocked": True,
            "delete_requires_approval": True,
            "public_download_allowed": False,
            "audit_log": True,
        },
        "lifecycle": {
            "policies": {
                "dataset": {"retention_days": 1095, "dependency_check": True},
                "checkpoint": {"retention_days": 30, "keep_best": True, "keep_latest": True},
                "eval_report": {"retention_days": 1095, "dependency_check": True},
                "deployment_package": {"retention_days": 1095, "dependency_check": True},
            },
            "legal_hold": False,
            "archive_tier": "warm_then_cold",
        },
        "promotion": {
            "sequence": ["checkpoint", "candidate_model", "evaluated_model", "release_candidate", "production_package"],
            "checks": {"integrity": True, "eval": True, "safety": True, "runtime": True, "approval": True},
            "approvers": ["model-owner", "safety-owner"],
        },
        "experiment": {
            "run_id": "run_train_042",
            "params_logged": True,
            "metrics_logged": True,
            "logs_uri": "artifact://logs/run_train_042",
            "input_artifacts": ["dataset_artifact", "config_artifact"],
            "output_artifacts": ["checkpoint_artifact", "eval_report_artifact"],
        },
        "rollback": {
            "rollback_package": "artifact://release/chat/prod/2026-05-25-r3",
            "required_artifacts_available": True,
            "rollback_manifest_checksum": "sha256:rollback-manifest",
            "load_test_passed": True,
        },
        "delete_plan": {
            "candidate_artifact": "checkpoint:train_042:step_8000",
            "dependent_count": 0,
            "retention_met": True,
            "legal_hold": False,
            "audit_record": True,
            "protected_artifact_blocked": True,
        },
        "inventory": [
            {"artifact_type": "dataset", "size_gib": 1200, "price_per_gib_month": 0.012, "retention_months": 36},
            {"artifact_type": "checkpoint", "size_gib": 600, "price_per_gib_month": 0.012, "retention_months": 1},
            {"artifact_type": "eval_report", "size_gib": 2, "price_per_gib_month": 0.012, "retention_months": 36},
            {"artifact_type": "deployment_package", "size_gib": 18, "price_per_gib_month": 0.012, "retention_months": 36},
        ],
        "artifact_management_gate": True,
    }


def ratio(num, den):
    return 0.0 if den == 0 else num / den


def artifact_storage_cost(case):
    return sum(item["size_gib"] * item["price_per_gib_month"] * item["retention_months"] for item in case["inventory"])


def artifact_metadata_contract(case):
    artifact = case["artifact"]
    fields_ok = all(artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS)
    return fields_ok and artifact["uri"].startswith("artifact://") and artifact["lifecycle_state"] in {"created", "validated", "active", "deprecated", "archived"}


def dataset_artifact_manifest(case):
    dataset = case["dataset"]
    files = dataset["files"]
    samples_match = sum(f["samples"] for f in files) == dataset["sample_count"]
    tokens_match = sum(f["tokens"] for f in files) == dataset["token_count"]
    fields = ["dataset_id", "version", "split", "schema_version", "source_composition", "cleaning_rule_version", "dedup_rule_version", "quality_metrics", "permissions", "checksum"]
    return all(dataset.get(field) for field in fields) and samples_match and tokens_match


def checkpoint_artifact_recoverability(case):
    checkpoint = case["checkpoint"]
    required_states = ["model_state", "optimizer_state", "scheduler_state", "dataloader_state", "rng_state", "distributed_metadata"]
    state_ok = all(checkpoint[state] for state in required_states)
    checksum_ok = all(shard in checkpoint["checksums"] for shard in checkpoint["shards"])
    return state_ok and checksum_ok and checkpoint["restore_tested"] and "resume" in checkpoint["purpose"]


def eval_report_reproducibility(case):
    report = case["eval_report"]
    required = ["model_ref", "eval_dataset_version", "eval_code_version", "metric_definitions", "prompt_template_version", "inference_config", "runtime_version", "result_summary", "sample_outputs_uri", "failure_cases", "evaluator_version", "report_checksum"]
    metrics_ok = set(report["metric_definitions"]) == set(report["result_summary"])
    return all(report.get(field) for field in required) and metrics_ok


def deployment_package_completeness(case):
    package = case["deployment_package"]
    required = ["weights", "tokenizer", "model_config", "runtime_config", "container_image_digest", "health_check", "generation_config", "safety_policy", "route_config", "release_manifest"]
    return all(package.get(field) for field in required) and package["container_image_digest"].startswith("sha256:")


def release_manifest_integrity(case):
    manifest = case["release_manifest"]
    refs_ok = REQUIRED_RELEASE_REFS.issubset(set(manifest["artifact_refs"]))
    return refs_ok and bool(manifest["checksum"]) and manifest["immutable"] and bool(manifest["rollback_target"])


def checksum_integrity_gate(case):
    integrity = case["integrity"]
    dataset_files = all(f.get("checksum") for f in case["dataset"]["files"])
    checkpoint_files = all(case["checkpoint"]["checksums"].get(shard) for shard in case["checkpoint"]["shards"])
    return integrity["checksums_verified"] and integrity["manifest_checksum_verified"] and integrity["download_verify_required"] and not integrity["failed_files"] and dataset_files and checkpoint_files


def artifact_lineage_completeness(case):
    lineage = case["lineage"]
    nodes_ok = REQUIRED_LINEAGE_NODES.issubset(set(lineage["nodes"]))
    edges_ok = all(edge in lineage["edges"] for edge in REQUIRED_LINEAGE_EDGES)
    events_ok = all(e.get("job") and e.get("run_id") and e.get("inputs") and e.get("outputs") for e in lineage["events"])
    return nodes_ok and edges_ok and events_ok


def artifact_store_metadata_split(case):
    store = case["store"]
    required_index = {"artifact_id", "artifact_type", "version", "checksum", "lifecycle_state"}
    return store["artifact_store_for_blobs"] and store["metadata_store_for_index"] and not store["large_blob_in_metadata_db"] and required_index.issubset(set(store["indexed_fields"]))


def permission_access_control(case):
    access = case["access"]
    return access["rbac_enabled"] and access["sample_outputs_restricted"] and access["cross_tenant_blocked"] and access["delete_requires_approval"] and not access["public_download_allowed"] and access["audit_log"]


def lifecycle_retention_policy(case):
    lifecycle = case["lifecycle"]
    policies = lifecycle["policies"]
    needed = {"dataset", "checkpoint", "eval_report", "deployment_package"}
    deps_ok = all(policies[name].get("dependency_check", True) for name in ["dataset", "eval_report", "deployment_package"])
    checkpoint_ok = policies["checkpoint"]["keep_best"] and policies["checkpoint"]["keep_latest"] and policies["checkpoint"]["retention_days"] >= 7
    return needed.issubset(set(policies)) and deps_ok and checkpoint_ok and bool(lifecycle["archive_tier"])


def promotion_gate_readiness(case):
    promotion = case["promotion"]
    checks = promotion["checks"]
    expected_sequence = ["checkpoint", "candidate_model", "evaluated_model", "release_candidate", "production_package"]
    return promotion["sequence"] == expected_sequence and all(checks.values()) and len(promotion["approvers"]) >= 2


def experiment_tracking_linkage(case):
    experiment = case["experiment"]
    return (
        bool(experiment["run_id"])
        and experiment["params_logged"]
        and experiment["metrics_logged"]
        and bool(experiment["logs_uri"])
        and {"dataset_artifact", "config_artifact"}.issubset(set(experiment["input_artifacts"]))
        and {"checkpoint_artifact", "eval_report_artifact"}.issubset(set(experiment["output_artifacts"]))
    )


def rollback_artifact_readiness(case):
    rollback = case["rollback"]
    return bool(rollback["rollback_package"]) and rollback["required_artifacts_available"] and bool(rollback["rollback_manifest_checksum"]) and rollback["load_test_passed"]


def deletion_dependency_safety(case):
    plan = case["delete_plan"]
    return plan["dependent_count"] == 0 and plan["retention_met"] and not plan["legal_hold"] and plan["audit_record"] and plan["protected_artifact_blocked"]


def artifact_management_gate(case):
    return case["artifact_management_gate"] is True


CHECKS = [
    ("artifact_metadata_contract", artifact_metadata_contract),
    ("dataset_artifact_manifest", dataset_artifact_manifest),
    ("checkpoint_artifact_recoverability", checkpoint_artifact_recoverability),
    ("eval_report_reproducibility", eval_report_reproducibility),
    ("deployment_package_completeness", deployment_package_completeness),
    ("release_manifest_integrity", release_manifest_integrity),
    ("checksum_integrity_gate", checksum_integrity_gate),
    ("artifact_lineage_completeness", artifact_lineage_completeness),
    ("artifact_store_metadata_split", artifact_store_metadata_split),
    ("permission_access_control", permission_access_control),
    ("lifecycle_retention_policy", lifecycle_retention_policy),
    ("promotion_gate_readiness", promotion_gate_readiness),
    ("experiment_tracking_linkage", experiment_tracking_linkage),
    ("rollback_artifact_readiness", rollback_artifact_readiness),
    ("deletion_dependency_safety", deletion_dependency_safety),
    ("artifact_management_gate", artifact_management_gate),
]


def make_bad_cases(base):
    cases = []

    def add(name, change):
        case = deepcopy(base)
        case["name"] = name
        change(case)
        cases.append(case)

    add("artifact_metadata_missing_bad", lambda c: c["artifact"].update({"created_by": ""}))
    add("dataset_manifest_mismatch_bad", lambda c: c["dataset"].update({"sample_count": 1999}))
    add("checkpoint_restore_state_missing_bad", lambda c: c["checkpoint"].update({"rng_state": False}))
    add("eval_report_prompt_missing_bad", lambda c: c["eval_report"].update({"prompt_template_version": ""}))
    add("deployment_package_missing_bad", lambda c: c["deployment_package"].update({"health_check": ""}))
    add("release_manifest_mutable_bad", lambda c: c["release_manifest"].update({"immutable": False}))
    add("checksum_not_verified_bad", lambda c: c["integrity"].update({"checksums_verified": False}))
    add("artifact_lineage_edge_missing_bad", lambda c: c["lineage"]["edges"].remove(("model_version", "deployment_package")))
    add("metadata_blob_mixed_bad", lambda c: c["store"].update({"large_blob_in_metadata_db": True}))
    add("permission_public_bad", lambda c: c["access"].update({"public_download_allowed": True}))
    add("retention_policy_missing_bad", lambda c: c["lifecycle"]["policies"]["deployment_package"].update({"dependency_check": False}))
    add("promotion_safety_missing_bad", lambda c: c["promotion"]["checks"].update({"safety": False}))
    add("experiment_outputs_missing_bad", lambda c: c["experiment"].update({"output_artifacts": ["checkpoint_artifact"]}))
    add("rollback_artifact_missing_bad", lambda c: c["rollback"].update({"required_artifacts_available": False}))
    add("delete_dependency_unsafe_bad", lambda c: c["delete_plan"].update({"dependent_count": 3}))
    add("artifact_gate_missing_bad", lambda c: c.update({"artifact_management_gate": False}))
    return cases


def summarize_examples(case):
    artifact = case["artifact"]
    dataset = case["dataset"]
    package = case["deployment_package"]
    release = case["release_manifest"]
    checksum_files = len(dataset["files"]) + len(case["checkpoint"]["shards"])
    checksum_present = sum(1 for f in dataset["files"] if f["checksum"]) + sum(1 for s in case["checkpoint"]["shards"] if case["checkpoint"]["checksums"].get(s))
    package_required = ["weights", "tokenizer", "model_config", "runtime_config", "container_image_digest", "health_check", "generation_config", "safety_policy", "route_config", "release_manifest"]
    return {
        "metadata_completeness": round(ratio(sum(1 for f in REQUIRED_ARTIFACT_FIELDS if artifact.get(f)), len(REQUIRED_ARTIFACT_FIELDS)), 3),
        "dataset_manifest_completeness": 1.0,
        "checksum_coverage": round(ratio(checksum_present, checksum_files), 3),
        "checkpoint_recoverable": checkpoint_artifact_recoverability(case),
        "eval_metric_count": len(case["eval_report"]["metric_definitions"]),
        "deployment_package_completeness": round(ratio(sum(1 for f in package_required if package.get(f)), len(package_required)), 3),
        "release_ref_coverage": round(ratio(len(set(release["artifact_refs"]) & REQUIRED_RELEASE_REFS), len(REQUIRED_RELEASE_REFS)), 3),
        "lineage_node_count": len(case["lineage"]["nodes"]),
        "monthly_storage_cost_usd": round(artifact_storage_cost(case), 2),
        "candidate_delete_safe": deletion_dependency_safety(case),
        "artifact_size_gib": artifact["size_gib"],
    }


def audit_artifact_management(cases):
    per_case = {
        case["name"]: {name: check(case) for name, check in CHECKS}
        for case in cases
    }
    metrics = {
        name: round(sum(result[name] for result in per_case.values()) / len(cases), 3)
        for name, _ in CHECKS
    }
    failed_cases = [name for name, result in per_case.items() if not all(result.values())]
    failed_gates = [name for name, score in metrics.items() if score < 1.0]
    return {
        "smoke": {
            "complete_case_passes": all(per_case["complete_case"].values()),
            "caught_dataset_gap": "dataset_manifest_mismatch_bad" in failed_cases,
            "caught_checkpoint_gap": "checkpoint_restore_state_missing_bad" in failed_cases,
            "caught_eval_gap": "eval_report_prompt_missing_bad" in failed_cases,
            "caught_release_gap": "release_manifest_mutable_bad" in failed_cases,
            "caught_delete_gap": "delete_dependency_unsafe_bad" in failed_cases,
        },
        "metrics": metrics,
        "hard_blocker_count": len(failed_cases),
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "artifact_management_gate_pass": len(failed_cases) == 0,
    }


base = complete_case()
cases = [base] + make_bad_cases(base)
report = audit_artifact_management(cases)

pprint({"artifact_examples": summarize_examples(base)})
pprint(report["smoke"])
pprint(report["metrics"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("artifact_management_gate_pass=", report["artifact_management_gate_pass"])

assert report["smoke"]["complete_case_passes"]
assert report["smoke"]["caught_checkpoint_gap"]
assert report["smoke"]["caught_delete_gap"]
assert report["metrics"]["artifact_metadata_contract"] == 0.941
assert report["metrics"]["artifact_management_gate"] == 0.941
assert report["hard_blocker_count"] == 16
assert report["artifact_management_gate_pass"] is False
```

这个 demo 的重点不是复刻真实平台，而是把 Artifact 管理拆成可检查证据：元数据契约、dataset manifest、checkpoint recoverability、eval report 可复现性、deployment package 完整性、release manifest、checksum、lineage、store / metadata 分层、权限、生命周期、promotion、experiment tracking、rollback、删除依赖安全和最终 artifact gate。

## 39.23 常见误区

误区一：Artifact 就是文件路径。

文件路径只是定位方式，artifact 还需要类型、版本、checksum、权限、血缘和生命周期。

误区二：checkpoint 都应该长期保留。

中间 checkpoint 成本很高，应按策略保留 best、latest 和关键版本。

误区三：eval report 只是一个分数表。

Eval report 还要记录 eval 数据版本、代码版本、prompt、runtime、样本输出和失败案例。

误区四：部署时现场拼装 artifact。

生产部署应使用 release manifest 或 deployment package，保证可复现和可回滚。

误区五：删除 artifact 只看时间。

删除前必须检查 lineage、deployment、回滚和合规依赖。

## 39.24 面试常见追问

问题一：Artifact 管理和模型仓库有什么区别？

可以回答：模型仓库专注模型资产，如权重、tokenizer、config、adapter 和量化版本；Artifact 管理更通用，覆盖 dataset、checkpoint、eval report、deployment package、日志和配置等所有关键产物。

问题二：为什么 checkpoint artifact 要记录 optimizer state？

可以回答：如果用于断点恢复，除了模型权重，还需要 optimizer、scheduler、dataloader 和随机状态，否则恢复后的训练轨迹可能不一致。

问题三：Eval report 为什么要版本化？

可以回答：评估结果依赖模型版本、eval 数据、评估代码、prompt、推理参数和 runtime。缺少版本信息，评估结果不可复现，也无法比较。

问题四：如何安全删除 artifact？

可以回答：先检查 artifact 是否被 dataset、model version、eval report、deployment、release manifest 或审计流程依赖；如果无依赖且超过 retention policy，再归档或删除，并记录审计。

## 39.25 小练习

1. Artifact 和普通文件有什么区别？
2. Dataset artifact 应该记录哪些元数据？
3. Checkpoint 用于断点恢复和候选模型时有什么不同？
4. Eval report 为什么必须绑定 eval dataset 和代码版本？
5. Deployment package 应该包含哪些内容？
6. Release manifest 对回滚有什么价值？
7. Artifact retention policy 如何设计？
8. 删除 artifact 前应该检查哪些依赖？

## 39.26 本章小结

本章讲了 Artifact 管理。

你需要记住：

1. Artifact 是 AI 研发和生产过程中的关键产物，不只是文件路径。
2. Artifact 应该有类型、版本、URI、checksum、权限、血缘和生命周期状态。
3. Dataset、checkpoint、eval report 和 deployment package 是 AI Infra 中最重要的 artifact 类型。
4. Release manifest 能保证部署可复现和可回滚。
5. Artifact 管理连接训练、评估、部署、审计和成本治理，是平台可复现性的基础设施。

下一章我们会讲实验追踪：参数、指标、日志、样本和代码版本。
