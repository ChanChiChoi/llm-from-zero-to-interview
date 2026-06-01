# 第 39 章 Artifact 管理：dataset、checkpoint、eval report 和 deployment package

上一章讲了模型仓库。本章讲一个更通用的概念：Artifact 管理。

在 AI Infra 中，训练、评估、部署、回滚、审计都围绕各种产物流转。dataset、checkpoint、eval report、deployment package、日志、配置、模型权重都可以看作 artifact。

先记住一句话：

> Artifact 管理的目标，是让 AI 研发过程中的每个关键产物都可定位、可校验、可复现、可追溯、可治理。

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

## 39.22 常见误区

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

## 39.23 面试常见追问

问题一：Artifact 管理和模型仓库有什么区别？

可以回答：模型仓库专注模型资产，如权重、tokenizer、config、adapter 和量化版本；Artifact 管理更通用，覆盖 dataset、checkpoint、eval report、deployment package、日志和配置等所有关键产物。

问题二：为什么 checkpoint artifact 要记录 optimizer state？

可以回答：如果用于断点恢复，除了模型权重，还需要 optimizer、scheduler、dataloader 和随机状态，否则恢复后的训练轨迹可能不一致。

问题三：Eval report 为什么要版本化？

可以回答：评估结果依赖模型版本、eval 数据、评估代码、prompt、推理参数和 runtime。缺少版本信息，评估结果不可复现，也无法比较。

问题四：如何安全删除 artifact？

可以回答：先检查 artifact 是否被 dataset、model version、eval report、deployment、release manifest 或审计流程依赖；如果无依赖且超过 retention policy，再归档或删除，并记录审计。

## 39.24 小练习

1. Artifact 和普通文件有什么区别？
2. Dataset artifact 应该记录哪些元数据？
3. Checkpoint 用于断点恢复和候选模型时有什么不同？
4. Eval report 为什么必须绑定 eval dataset 和代码版本？
5. Deployment package 应该包含哪些内容？
6. Release manifest 对回滚有什么价值？
7. Artifact retention policy 如何设计？
8. 删除 artifact 前应该检查哪些依赖？

## 39.25 本章小结

本章讲了 Artifact 管理。

你需要记住：

1. Artifact 是 AI 研发和生产过程中的关键产物，不只是文件路径。
2. Artifact 应该有类型、版本、URI、checksum、权限、血缘和生命周期状态。
3. Dataset、checkpoint、eval report 和 deployment package 是 AI Infra 中最重要的 artifact 类型。
4. Release manifest 能保证部署可复现和可回滚。
5. Artifact 管理连接训练、评估、部署、审计和成本治理，是平台可复现性的基础设施。

下一章我们会讲实验追踪：参数、指标、日志、样本和代码版本。
