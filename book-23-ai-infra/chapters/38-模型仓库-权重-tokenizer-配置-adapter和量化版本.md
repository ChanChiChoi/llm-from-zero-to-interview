# 第 38 章 模型仓库：权重、tokenizer、配置、adapter 和量化版本

上一章讲了数据版本、血缘和质量监控。本章继续讲模型资产管理中的核心系统：模型仓库。

训练平台产出 checkpoint，评估平台产出 eval report，推理平台需要部署模型版本。中间必须有一个可信的模型仓库，把权重、tokenizer、配置、adapter、量化版本和元数据统一管理起来。

先记住一句话：

> 模型仓库不是存权重文件的网盘，而是模型资产的版本、元数据、权限、血缘、评估和发布状态管理系统。

## 38.1 为什么需要模型仓库

小实验中，模型可能只是一个目录：

```text
./outputs/checkpoint-10000/
```

但生产环境里会有大量模型资产：

1. base model。
2. SFT model。
3. RLHF / DPO model。
4. LoRA / adapter。
5. merge 后模型。
6. 量化模型。
7. embedding model。
8. reranker。
9. 多模态模型。
10. 不同 runtime 格式的模型。

如果没有模型仓库，团队很快会搞不清哪个模型能用、哪个模型评估过、哪个模型线上在跑、哪个模型可以删除。

## 38.2 模型仓库解决什么问题

模型仓库要解决：

1. 模型资产注册。
2. 权重和相关文件存储。
3. 模型版本管理。
4. tokenizer 和配置绑定。
5. adapter 和 base model 关系管理。
6. 量化版本管理。
7. 模型血缘追踪。
8. 评估报告关联。
9. 权限和安全管理。
10. 发布状态管理。
11. 生命周期管理。

它是训练平台、评估平台和推理平台之间的资产枢纽。

## 38.3 模型仓库中的核心对象

模型仓库通常包含这些对象：

1. Model。
2. ModelVersion。
3. ModelArtifact。
4. TokenizerVersion。
5. ModelConfig。
6. Adapter。
7. QuantizedVersion。
8. EvalReport。
9. DeploymentRecord。
10. LineageRecord。

其中 Model 是逻辑模型，ModelVersion 是具体版本，Artifact 是实际文件集合。

不要把 Model 和 ModelVersion 混在一起。

## 38.4 Model 和 ModelVersion

Model 表示一个模型系列，例如：

```text
company-chat-model
company-code-model
company-embedding-model
```

ModelVersion 表示具体版本，例如：

```text
company-chat-model:v1.0.0
company-chat-model:v1.1.0-sft
company-chat-model:v1.2.0-dpo
```

一个 Model 可以有多个 ModelVersion。

线上部署、评估、回滚都应该指向具体 ModelVersion。

## 38.5 ModelVersion 应该记录什么

ModelVersion 至少记录：

1. 模型名称。
2. 版本号。
3. 模型类型。
4. 参数规模。
5. 架构类型。
6. 权重路径。
7. 权重 checksum。
8. tokenizer 版本。
9. model config。
10. 上下文长度。
11. 支持精度。
12. 支持 runtime。
13. 训练任务 ID。
14. 训练数据版本。
15. 评估报告。
16. 安全等级。
17. 发布状态。
18. 创建人和审批人。

这些字段决定模型是否可复现、可部署、可审计。

## 38.6 权重文件管理

权重文件是模型仓库最核心的 artifact。

常见权重格式包括：

1. PyTorch `.bin`。
2. safetensors。
3. TensorRT engine。
4. GGUF。
5. ONNX。
6. 自定义 runtime 格式。

大模型权重通常很大，管理时要考虑：

1. 分片文件。
2. checksum 校验。
3. 上传断点续传。
4. 下载加速。
5. 本地缓存。
6. 跨地域复制。
7. 生命周期和冷存储。

权重文件损坏或不一致，会导致模型加载失败或线上行为异常。

## 38.7 Safetensors 为什么常见

Safetensors 是常见权重格式之一。

它的优点包括：

1. 加载更安全。
2. 避免 pickle 反序列化风险。
3. 支持快速加载。
4. 适合大模型分片存储。

模型仓库可以要求生产模型优先使用 safetensors 或经过安全校验的格式。

权重格式不是细枝末节，它关系到安全、加载速度和 runtime 兼容性。

## 38.8 Tokenizer 管理

Tokenizer 是模型不可分割的一部分。

同一个权重配错 tokenizer，会导致：

1. 输出乱码。
2. 特殊 token 错误。
3. 上下文长度异常。
4. stop token 失效。
5. 工具调用格式异常。
6. 多语言效果退化。

模型仓库必须记录 tokenizer 版本，并和 ModelVersion 绑定。

Tokenizer artifact 通常包括：

1. tokenizer config。
2. vocab。
3. merges。
4. special tokens map。
5. chat template。

不要把 tokenizer 当成可随意替换的小文件。

## 38.9 Model Config 管理

Model config 描述模型结构和推理参数。

常见字段：

1. hidden size。
2. layer 数。
3. attention head 数。
4. vocab size。
5. rope 参数。
6. max position embeddings。
7. bos/eos token。
8. dtype。
9. quantization config。
10. architecture name。

runtime 加载模型时依赖这些配置。

配置错误会导致加载失败、性能异常或输出错误。

## 38.10 Adapter 管理

Adapter 常见于 LoRA、QLoRA 和其他 PEFT 方法。

Adapter 不是完整模型，而是依赖 base model 的增量参数。

模型仓库要记录：

1. adapter 名称。
2. adapter 版本。
3. base model 版本。
4. adapter 权重路径。
5. 训练任务 ID。
6. 训练数据版本。
7. merge 状态。
8. 兼容性约束。
9. 评估报告。

如果 adapter 和 base model 不匹配，结果可能完全不可用。

## 38.11 Merge 后模型

LoRA adapter 可以在推理时动态加载，也可以 merge 到 base model 中生成完整权重。

两种方式各有利弊。

动态加载 adapter：

1. 存储更省。
2. 方便切换多个 adapter。
3. 对 runtime 支持有要求。
4. 可能增加推理复杂度。

Merge 后模型：

1. 部署更简单。
2. runtime 兼容性更好。
3. 存储占用更大。
4. 需要记录 merge 过程。

模型仓库应记录 merge lineage：哪个 base model 加哪个 adapter 生成了哪个 merged model。

## 38.12 量化版本管理

量化版本是同一模型的低精度部署形态。

例如：

1. FP16。
2. BF16。
3. FP8。
4. INT8。
5. INT4。
6. GPTQ。
7. AWQ。
8. GGUF variants。

量化版本要记录：

1. 原始模型版本。
2. 量化方法。
3. 校准数据。
4. 量化工具版本。
5. 量化参数。
6. 目标 runtime。
7. 质量评估。
8. 延迟和吞吐 benchmark。

量化不是简单转换格式，必须评估质量和性能。

## 38.13 Runtime 兼容性

不同 runtime 支持的格式和能力不同。

模型仓库要记录：

1. 是否支持 vLLM。
2. 是否支持 TGI。
3. 是否支持 Triton。
4. 是否支持 TensorRT-LLM。
5. 是否支持 SGLang。
6. 需要的 CUDA 版本。
7. 需要的 driver 版本。
8. 需要的 tokenizer 配置。
9. 是否支持 tensor parallel。
10. 是否支持特定量化格式。

推理平台部署前应检查 runtime compatibility。

不能让部署阶段才发现模型格式不支持。

## 38.14 模型血缘

模型血缘记录模型从哪里来。

常见血缘关系：

1. base model -> SFT model。
2. SFT model -> DPO model。
3. base model + adapter -> merged model。
4. FP16 model -> INT4 quantized model。
5. checkpoint -> model version。
6. model version -> deployment。

血缘还要关联训练数据、训练代码、训练配置和评估报告。

当某个模型出现问题时，平台需要追溯到训练任务和数据版本。

## 38.15 评估报告关联

模型仓库不应该只保存模型文件，还要关联 eval report。

评估报告可以包括：

1. 通用能力评估。
2. 领域评估。
3. 安全评估。
4. 工具调用评估。
5. 结构化输出评估。
6. 长上下文评估。
7. 延迟和吞吐 benchmark。
8. 成本估算。

模型发布前，平台应检查是否有必要评估报告，且指标满足门禁。

## 38.16 发布状态管理

模型版本可以有状态：

```text
registered -> validated -> evaluated -> staged -> production -> deprecated -> archived
```

状态变化应该有权限控制和审计。

例如：

1. 未评估模型不能进入 production。
2. 安全评估未通过不能发布。
3. deprecated 模型不能新建部署。
4. archived 模型只能保留审计，不提供线上服务。

模型仓库和发布系统要联动。

## 38.17 权限控制

模型本身也有权限。

权限问题包括：

1. 谁能下载权重。
2. 谁能部署模型。
3. 谁能查看评估报告。
4. 谁能使用敏感数据训练出的模型。
5. 哪个租户能调用哪个模型。
6. 哪些模型不能导出。

企业内部训练出的模型可能包含敏感知识，不能默认所有人可见和可用。

## 38.18 安全扫描

模型仓库可以对 artifact 做安全扫描：

1. 文件完整性校验。
2. 反序列化风险检查。
3. 可疑文件检查。
4. 权重格式检查。
5. tokenizer 特殊 token 检查。
6. 配置合法性检查。
7. 许可证元数据检查。

模型 artifact 是供应链的一部分，也需要安全治理。

## 38.19 生命周期管理

大模型 artifact 很占存储。

生命周期管理要回答：

1. 哪些 checkpoint 保留？
2. 哪些中间版本可以删除？
3. 哪些 production 模型必须长期保留？
4. 哪些模型可以转冷存储？
5. 删除前是否有部署依赖？
6. 删除是否影响回滚？

策略示例：

1. production 版本长期保留。
2. 最近 N 个候选版本保留。
3. 失败实验 checkpoint 定期清理。
4. 有审计或合规要求的版本冻结。
5. 删除前检查 lineage 和 deployment。

模型仓库如果不做生命周期管理，存储成本会快速增长。

## 38.20 模型仓库和推理平台的接口

推理平台部署模型时，应该从模型仓库读取：

1. 权重路径。
2. tokenizer 路径。
3. model config。
4. runtime 兼容性。
5. 默认推理参数。
6. 上下文长度。
7. 量化版本。
8. 安全等级。
9. 发布状态。

如果模型状态不是 production 或 staged，推理平台应该拒绝部署或要求审批。

## 38.21 模型仓库和训练平台的接口

训练平台完成训练后，可以将 checkpoint 注册成候选模型版本。

注册过程包括：

1. 上传或引用权重。
2. 计算 checksum。
3. 绑定 tokenizer。
4. 绑定训练配置。
5. 绑定 dataset version。
6. 绑定训练任务 ID。
7. 创建 ModelVersion。
8. 触发评估任务。

这样训练产物才能进入标准化资产管理流程。

## 38.22 模型仓库和评估平台的接口

评估平台从模型仓库读取候选模型，执行评估后写回 eval report。

流程：

```text
ModelVersion -> EvalJob -> EvalReport -> Quality Gate -> Release Candidate
```

评估报告写回后，模型状态可以从 registered 变成 evaluated 或 validated。

没有评估报告的模型不应该直接上线。

## 38.23 一个模型仓库系统架构

可以这样设计：

```text
Model Registry API
  -> Metadata DB
  -> Artifact Store
  -> Checksum Service
  -> Access Control
  -> Lineage Graph
  -> Eval Report Store
  -> Release State Machine
  -> Audit Log
```

Artifact Store 存大文件，Metadata DB 存模型元数据，Lineage Graph 存依赖关系，Audit Log 记录变更。

## 38.24 常见误区

误区一：模型仓库就是对象存储目录。

对象存储只能存文件，模型仓库还要管理版本、元数据、血缘、评估、权限和发布状态。

误区二：tokenizer 可以随便换。

tokenizer 和模型权重强绑定，换错会导致严重输出问题。

误区三：量化版本只看速度。

量化还要看质量退化、runtime 兼容性和目标场景。

误区四：adapter 是独立模型。

adapter 依赖 base model，必须记录兼容关系。

误区五：模型删除只看文件是否没人用。

还要看 deployment、回滚、审计、合规和 lineage 依赖。

## 38.25 面试常见追问

问题一：模型仓库和对象存储有什么区别？

可以回答：对象存储只负责保存文件，模型仓库还管理 ModelVersion、tokenizer、config、adapter、量化版本、评估报告、权限、血缘、发布状态和审计。

问题二：为什么 tokenizer 要版本化？

可以回答：tokenizer 决定文本和 token 的映射，特殊 token、chat template 和 stop token 都会影响模型行为。tokenizer 与权重不匹配会导致乱码、格式错误或能力退化。

问题三：如何管理 LoRA adapter？

可以回答：记录 adapter 版本、base model 版本、训练数据、训练任务、权重路径、兼容性、是否 merge 以及评估报告。adapter 不能脱离 base model 单独管理。

问题四：量化模型上线前要检查什么？

可以回答：检查原始模型版本、量化方法、校准数据、量化工具版本、runtime 兼容性、质量评估、延迟吞吐 benchmark 和成本收益。

## 38.26 小练习

1. Model 和 ModelVersion 有什么区别？
2. ModelVersion 应该记录哪些元数据？
3. 为什么权重文件需要 checksum？
4. tokenizer 和 chat template 为什么要绑定模型版本？
5. LoRA adapter 和 merge 后模型如何管理血缘？
6. 量化版本为什么需要单独评估？
7. 模型仓库如何和训练平台、评估平台、推理平台联动？
8. 模型生命周期管理要考虑哪些依赖？

## 38.27 本章小结

本章讲了模型仓库。

你需要记住：

1. 模型仓库不是网盘，而是模型资产的版本、元数据、权限、血缘、评估和发布状态管理系统。
2. Model 是逻辑模型，ModelVersion 是具体版本，Artifact 是实际文件集合。
3. 权重、tokenizer、config、adapter 和量化版本都必须被纳入版本管理。
4. Adapter 依赖 base model，量化版本依赖原始模型和量化配置，它们都需要血缘关系。
5. 模型仓库连接训练、评估和推理平台，是模型资产生命周期的中心。

下一章我们会讲 Artifact 管理：dataset、checkpoint、eval report 和 deployment package。
