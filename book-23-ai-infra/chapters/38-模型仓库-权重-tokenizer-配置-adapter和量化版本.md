# 第 38 章 模型仓库：权重、tokenizer、配置、adapter 和量化版本

上一章讲了数据版本、血缘和质量监控。本章继续讲模型资产管理中的核心系统：模型仓库。

训练平台产出 checkpoint，评估平台产出 eval report，推理平台需要部署模型版本。中间必须有一个可信的模型仓库，把权重、tokenizer、配置、adapter、量化版本和元数据统一管理起来。

先记住一句话：

> 模型仓库不是存权重文件的普通目录，而是模型资产的版本、元数据、权限、血缘、评估和发布状态管理系统。

## 38.0 本讲资料边界与第二轮精修口径

本讲按截至 2026-06 的稳定公开资料校准：MLflow Model Registry 对 registered model、model version、alias、tag 和 stage 的模型生命周期抽象，Hugging Face Hub 对模型仓库、revision、model card 和文件版本的资产管理口径，Safetensors 对安全权重格式和避免 pickle 反序列化风险的说明，Hugging Face PEFT 对 adapter、base model、merge 的关系管理口径，以及 vLLM 等推理 runtime 对权重格式、tokenizer、量化和加载兼容性的公开工程边界。

本章只抽象大模型平台里的模型产物仓库能力，不绑定某个 registry 产品、对象存储、推理引擎、量化工具、模型卡模板或内部平台实现。

和前面章节的分工是：

1. 第 36 章和第 37 章回答“训练数据如何被版本化、追踪和治理”。
2. 本章回答“训练、评估和部署之间的模型产物如何被冻结、校验、发布、回滚和审计”。

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

一个模型版本可以抽象为：

```math
a_i=(m_i,v_i,w_i,t_i,c_i,b_i,p_i,q_i,r_i,e_i,s_i,l_i,o_i,z_i)
```

其中 `m_i` 是逻辑 model id，`v_i` 是 model version，`w_i` 是权重 manifest，`t_i` 是 tokenizer version，`c_i` 是 model config，`b_i` 是 base model 或训练 checkpoint，`p_i` 是权限与安全策略，`q_i` 是量化配置，`r_i` 是 runtime 兼容性，`e_i` 是评估报告，`s_i` 是发布状态，`l_i` 是 lineage，`o_i` 是 owner / approver，`z_i` 是审计记录。

元数据完整率可以写成：

```math
C_{\mathrm{artifact}}=\frac{K_{\mathrm{filled}}}{K_{\mathrm{required}}}
```

其中 `K_filled` 是已填写的必要字段数，`K_required` 是模型发布前必须存在的字段数。这个公式的直觉是：模型版本不是文件路径，而是一组发布证据；字段缺失越多，复现和回滚风险越高。

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

权重 checksum 覆盖率可以写成：

```math
C_{\mathrm{weight}}=\frac{N_{\mathrm{checked}}}{N_{\mathrm{weight,total}}}
```

其中 `N_checked` 是已经校验过 checksum 的权重分片数，`N_weight,total` 是 manifest 中声明的权重分片总数。生产发布时通常要求 `C_weight=1`，否则同一个 ModelVersion 在不同机器上可能加载到不一致内容。

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

tokenizer 和 config 的绑定可以用一个二值门禁表示：

```math
B_{\mathrm{tok}}=\mathbf{1}[h_{\mathrm{tok}}=h_{\mathrm{tok,expected}} \land h_{\mathrm{cfg}}=h_{\mathrm{cfg,expected}}]
```

其中 `h_tok` 是实际 tokenizer artifact 的 hash，`h_tok,expected` 是 ModelVersion 记录的期望 tokenizer hash，`h_cfg` 是实际 config hash，`h_cfg,expected` 是期望 config hash。这个门禁的直觉是：权重、tokenizer、chat template、special token 和 config 是同一个行为契约，不能只靠文件名匹配。

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

adapter 与 base model 的最小兼容门禁可以写成：

```math
G_{\mathrm{adapter}}=\mathbf{1}[v_{\mathrm{base}}=v_{\mathrm{base,expected}} \land d_{\mathrm{rank}}\le d_{\mathrm{hidden}}]
```

其中 `v_base` 是 adapter 实际绑定的 base model 版本，`v_base,expected` 是仓库记录的期望 base model 版本，`d_rank` 是 adapter rank，`d_hidden` 是 base model hidden size。真实系统还会检查 target modules、dtype、RoPE / vocab / tokenizer、PEFT 方法和训练配置。

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

量化前后的质量退化可以写成：

```math
\Delta Q=Q_{\mathrm{base}}-Q_{\mathrm{quant}}
```

其中 `Q_base` 是原始模型在同一评估集上的质量指标，`Q_quant` 是量化版本质量指标。上线门禁不应该只看吞吐提升，还要约束 `Delta Q` 不超过业务可接受阈值。

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

模型加载耗时可以粗略拆成：

```math
T_{\mathrm{load}}=\frac{S_{\mathrm{artifact}}}{B_{\mathrm{read}}}+T_{\mathrm{verify}}+T_{\mathrm{init}}
```

其中 `S_artifact` 是模型产物大小，`B_read` 是有效读取带宽，`T_verify` 是 checksum / manifest / 安全扫描校验时间，`T_init` 是 runtime 初始化、权重映射、kernel 准备和缓存预热时间。这个公式用于解释：模型仓库的缓存、manifest 和格式选择会直接影响推理平台冷启动。

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

## 38.24 模型仓库审计指标和最小 demo

模型仓库的最终门禁可以写成：

```math
G_{\mathrm{registry}}=\mathbf{1}[\min_j C_j\ge\tau_j \land C_{\mathrm{weight}}=1 \land B_{\mathrm{tok}}=1 \land \Delta Q\le\rho_Q \land T_{\mathrm{load}}\le\tau_{\mathrm{load}} \land P_0=0]
```

其中 `C_j` 是每个审计维度的覆盖率，`tau_j` 是该维度阈值，`rho_Q` 是可接受量化质量退化，`tau_load` 是加载耗时阈值，`P0` 是未解决的 P0 级发布风险数量。这个门禁强调：模型仓库不是“文件可下载”就合格，而是权重、tokenizer、config、adapter、merge、量化、runtime、评估、安全、发布、回滚和生命周期都能被平台验证。

下面是一个 0 依赖 toy demo。它把本章的模型仓库治理拆成 16 个可审计门禁，并构造 16 个 bad case，覆盖权重 manifest 缺失、unsafe 权重格式、tokenizer / config 绑定错误、adapter base 不匹配、merge lineage 缺失、量化质量退化、runtime 不兼容、评估报告缺失、权限安全不足、发布治理缺失、回滚别名缺失、lineage 断边、加载缓存不达标、生命周期策略缺失和最终 registry gate 缺失。

```python
from copy import deepcopy
from pprint import pprint


REQUIRED_MODEL_FIELDS = [
    "model_id",
    "version",
    "artifact_type",
    "architecture",
    "parameter_count",
    "owner",
    "stage",
    "created_at",
    "train_job_id",
    "dataset_version",
    "eval_report_id",
    "safety_report_id",
    "release_alias",
    "rollback_alias",
]

REQUIRED_LINEAGE_NODES = {
    "dataset_version",
    "training_job",
    "checkpoint",
    "model_version",
    "adapter",
    "merged_model",
    "quantized_model",
    "eval_report",
    "deployment",
}

REQUIRED_LINEAGE_EDGES = [
    ("dataset_version", "training_job"),
    ("training_job", "checkpoint"),
    ("checkpoint", "model_version"),
    ("model_version", "adapter"),
    ("adapter", "merged_model"),
    ("merged_model", "quantized_model"),
    ("quantized_model", "eval_report"),
    ("eval_report", "deployment"),
]


def complete_case():
    return {
        "name": "complete_case",
        "model_version": {
            "model_id": "company-chat-model",
            "version": "v1.2.0-dpo-int4",
            "artifact_type": "quantized_merged_model",
            "architecture": "decoder_only_transformer",
            "parameter_count": 7000000000,
            "owner": "model-platform",
            "stage": "production",
            "created_at": "2026-06-01T09:00:00Z",
            "train_job_id": "train_sft_042",
            "dataset_version": "pretrain_mix:2026-06-01-v2",
            "eval_report_id": "eval_chat_v120",
            "safety_report_id": "safety_chat_v120",
            "release_alias": "chat-prod",
            "rollback_alias": "chat-prod-previous",
        },
        "weights": {
            "format": "safetensors",
            "pickle_required": False,
            "security_scan_passed": True,
            "manifest_checksum": "sha256:manifest-model-v120",
            "total_size_gib": 14.0,
            "files": [
                {"path": "model-00001-of-00002.safetensors", "checksum": "sha256:w0", "size_gib": 7.0},
                {"path": "model-00002-of-00002.safetensors", "checksum": "sha256:w1", "size_gib": 7.0},
            ],
        },
        "tokenizer_config": {
            "tokenizer_hash": "sha256:tok-v4",
            "expected_tokenizer_hash": "sha256:tok-v4",
            "config_hash": "sha256:cfg-v120",
            "expected_config_hash": "sha256:cfg-v120",
            "chat_template_hash": "sha256:chat-template-v3",
            "special_tokens": ["bos", "eos", "pad"],
            "stop_tokens": ["<eos>"],
        },
        "adapter": {
            "method": "lora",
            "adapter_version": "adapter-dpo-v7",
            "base_version": "company-chat-model:v1.1.0-sft",
            "expected_base_version": "company-chat-model:v1.1.0-sft",
            "rank": 16,
            "hidden_size": 4096,
            "target_modules": ["q_proj", "v_proj"],
            "train_job_id": "train_dpo_007",
        },
        "merge": {
            "base_version": "company-chat-model:v1.1.0-sft",
            "adapter_version": "adapter-dpo-v7",
            "merged_model_version": "company-chat-model:v1.2.0-dpo",
            "merge_tool_version": "peft-merge-v2",
            "lineage_recorded": True,
            "eval_after_merge": True,
        },
        "quantization": {
            "source_version": "company-chat-model:v1.2.0-dpo",
            "quant_method": "awq",
            "bits": 4,
            "calibration_dataset": "calib_chat_2026_06",
            "tool_version": "awq-0.2",
            "runtime_target": "vllm",
            "base_quality": 0.842,
            "quant_quality": 0.834,
            "max_quality_drop": 0.015,
            "latency_speedup": 1.7,
            "eval_report_id": "eval_quant_v120",
        },
        "runtime": {
            "target_runtime": "vllm",
            "supported_runtimes": ["vllm", "tgi"],
            "quantization_supported": True,
            "required_context": 8192,
            "context_length": 8192,
            "tensor_parallel": 2,
            "available_gpus": 2,
            "artifact_size_gib": 14.0,
            "read_bandwidth_gib_s": 2.0,
            "verify_time_s": 2.0,
            "init_time_s": 11.0,
            "load_time_slo": 25.0,
            "cache_prepared": True,
            "prewarm_ready": True,
            "smoke_loaded": True,
        },
        "eval": {
            "required_tasks": ["chat", "tool_call", "safety", "latency"],
            "reports": {
                "chat": {"report_id": "eval_chat_v120", "score": 0.86, "threshold": 0.82},
                "tool_call": {"report_id": "eval_tool_v120", "score": 0.91, "threshold": 0.88},
                "safety": {"report_id": "safety_chat_v120", "score": 0.99, "threshold": 0.98},
                "latency": {"report_id": "bench_chat_v120", "score": 0.94, "threshold": 0.90},
            },
        },
        "security_permission": {
            "safety_approved": True,
            "license_ok": True,
            "classification": "internal",
            "authorized_projects": ["assistant-prod"],
            "download_roles": ["model-platform-admin"],
            "export_public": False,
            "audit_log": True,
        },
        "release": {
            "stage": "production",
            "approvers": ["model-owner", "safety-owner"],
            "state_transition_audit": True,
            "no_mutable_latest": True,
            "versions": ["company-chat-model:v1.1.0-sft", "company-chat-model:v1.2.0-dpo-int4"],
            "rollback_version": "company-chat-model:v1.1.0-sft",
            "rollback_load_test": True,
        },
        "lineage": {
            "nodes": sorted(REQUIRED_LINEAGE_NODES),
            "edges": list(REQUIRED_LINEAGE_EDGES),
            "events": [
                {"job": "train", "run_id": "train_sft_042", "inputs": ["dataset_version"], "outputs": ["checkpoint"]},
                {"job": "adapt", "run_id": "train_dpo_007", "inputs": ["model_version"], "outputs": ["adapter"]},
                {"job": "merge", "run_id": "merge_003", "inputs": ["model_version", "adapter"], "outputs": ["merged_model"]},
                {"job": "quantize", "run_id": "quant_002", "inputs": ["merged_model"], "outputs": ["quantized_model"]},
                {"job": "eval", "run_id": "eval_009", "inputs": ["quantized_model"], "outputs": ["eval_report"]},
            ],
        },
        "lifecycle": {
            "production_retention_years": 3,
            "candidate_retention_days": 30,
            "dependency_check_before_delete": True,
            "protected_from_delete": True,
            "archive_policy": "retain_manifest_and_eval",
        },
        "model_registry_gate": True,
    }


def ratio(num, den):
    return 0.0 if den == 0 else num / den


def load_time(runtime):
    return runtime["artifact_size_gib"] / runtime["read_bandwidth_gib_s"] + runtime["verify_time_s"] + runtime["init_time_s"]


def model_version_contract(case):
    version = case["model_version"]
    return all(version.get(field) for field in REQUIRED_MODEL_FIELDS) and version["stage"] in {"registered", "evaluated", "staged", "production"}


def weight_manifest_integrity(case):
    weights = case["weights"]
    files = weights["files"]
    unique_paths = len({f["path"] for f in files}) == len(files)
    size_match = abs(sum(f["size_gib"] for f in files) - weights["total_size_gib"]) < 1e-9
    checksums = bool(weights["manifest_checksum"]) and all(f.get("checksum") for f in files)
    return unique_paths and size_match and checksums


def safe_weight_format(case):
    weights = case["weights"]
    return weights["format"] == "safetensors" and not weights["pickle_required"] and weights["security_scan_passed"]


def tokenizer_config_binding(case):
    binding = case["tokenizer_config"]
    token_ok = binding["tokenizer_hash"] == binding["expected_tokenizer_hash"]
    config_ok = binding["config_hash"] == binding["expected_config_hash"]
    special_ok = {"bos", "eos"}.issubset(set(binding["special_tokens"])) and bool(binding["stop_tokens"])
    return token_ok and config_ok and bool(binding["chat_template_hash"]) and special_ok


def adapter_base_compatibility(case):
    adapter = case["adapter"]
    return (
        adapter["base_version"] == adapter["expected_base_version"]
        and adapter["rank"] <= adapter["hidden_size"]
        and bool(adapter["target_modules"])
        and bool(adapter["train_job_id"])
    )


def merge_lineage_completeness(case):
    merge = case["merge"]
    return (
        bool(merge["base_version"])
        and bool(merge["adapter_version"])
        and bool(merge["merged_model_version"])
        and bool(merge["merge_tool_version"])
        and merge["lineage_recorded"]
        and merge["eval_after_merge"]
    )


def quantization_eval_gate(case):
    quant = case["quantization"]
    quality_drop = quant["base_quality"] - quant["quant_quality"]
    return (
        bool(quant["source_version"])
        and bool(quant["calibration_dataset"])
        and bool(quant["tool_version"])
        and bool(quant["runtime_target"])
        and quality_drop <= quant["max_quality_drop"]
        and quant["latency_speedup"] > 1.0
        and bool(quant["eval_report_id"])
    )


def runtime_compatibility_fit(case):
    runtime = case["runtime"]
    return (
        runtime["target_runtime"] in runtime["supported_runtimes"]
        and runtime["quantization_supported"]
        and runtime["context_length"] >= runtime["required_context"]
        and runtime["available_gpus"] >= runtime["tensor_parallel"]
        and runtime["smoke_loaded"]
    )


def eval_report_linkage(case):
    eval_info = case["eval"]
    reports = eval_info["reports"]
    required = eval_info["required_tasks"]
    task_ok = all(task in reports for task in required)
    score_ok = all(report["score"] >= report["threshold"] for report in reports.values())
    ids = {report["report_id"] for report in reports.values()}
    version = case["model_version"]
    linked = version["eval_report_id"] in ids and version["safety_report_id"] in ids
    return task_ok and score_ok and linked


def safety_permission_gate(case):
    policy = case["security_permission"]
    return (
        policy["safety_approved"]
        and policy["license_ok"]
        and policy["classification"] in {"internal", "public"}
        and bool(policy["authorized_projects"])
        and bool(policy["download_roles"])
        and not policy["export_public"]
        and policy["audit_log"]
    )


def release_status_governance(case):
    release = case["release"]
    return (
        release["stage"] == "production"
        and len(release["approvers"]) >= 2
        and release["state_transition_audit"]
        and release["no_mutable_latest"]
    )


def rollback_alias_readiness(case):
    release = case["release"]
    version = case["model_version"]
    return (
        bool(version["rollback_alias"])
        and release["rollback_version"] in release["versions"]
        and release["rollback_load_test"]
    )


def artifact_lineage_completeness(case):
    lineage = case["lineage"]
    nodes_ok = REQUIRED_LINEAGE_NODES.issubset(set(lineage["nodes"]))
    edges_ok = all(edge in lineage["edges"] for edge in REQUIRED_LINEAGE_EDGES)
    events_ok = all(e.get("job") and e.get("run_id") and e.get("inputs") and e.get("outputs") for e in lineage["events"])
    return nodes_ok and edges_ok and events_ok


def load_cache_readiness(case):
    runtime = case["runtime"]
    return (
        load_time(runtime) <= runtime["load_time_slo"]
        and runtime["cache_prepared"]
        and runtime["prewarm_ready"]
    )


def lifecycle_retention_policy(case):
    lifecycle = case["lifecycle"]
    return (
        lifecycle["production_retention_years"] >= 1
        and lifecycle["candidate_retention_days"] >= 7
        and lifecycle["dependency_check_before_delete"]
        and lifecycle["protected_from_delete"]
        and bool(lifecycle["archive_policy"])
    )


def model_registry_gate(case):
    return case["model_registry_gate"] is True


CHECKS = [
    ("model_version_contract", model_version_contract),
    ("weight_manifest_integrity", weight_manifest_integrity),
    ("safe_weight_format", safe_weight_format),
    ("tokenizer_config_binding", tokenizer_config_binding),
    ("adapter_base_compatibility", adapter_base_compatibility),
    ("merge_lineage_completeness", merge_lineage_completeness),
    ("quantization_eval_gate", quantization_eval_gate),
    ("runtime_compatibility_fit", runtime_compatibility_fit),
    ("eval_report_linkage", eval_report_linkage),
    ("safety_permission_gate", safety_permission_gate),
    ("release_status_governance", release_status_governance),
    ("rollback_alias_readiness", rollback_alias_readiness),
    ("artifact_lineage_completeness", artifact_lineage_completeness),
    ("load_cache_readiness", load_cache_readiness),
    ("lifecycle_retention_policy", lifecycle_retention_policy),
    ("model_registry_gate", model_registry_gate),
]


def make_bad_cases(base):
    cases = []

    def add(name, change):
        case = deepcopy(base)
        case["name"] = name
        change(case)
        cases.append(case)

    add("model_contract_missing_bad", lambda c: c["model_version"].update({"owner": ""}))
    add("weight_checksum_missing_bad", lambda c: c["weights"]["files"][0].update({"checksum": ""}))
    add("unsafe_weight_format_bad", lambda c: c["weights"].update({"format": "bin", "pickle_required": True}))
    add("tokenizer_hash_mismatch_bad", lambda c: c["tokenizer_config"].update({"tokenizer_hash": "sha256:wrong"}))
    add("adapter_base_mismatch_bad", lambda c: c["adapter"].update({"base_version": "company-chat-model:v0.9.0"}))
    add("merge_lineage_missing_bad", lambda c: c["merge"].update({"lineage_recorded": False}))
    add("quant_quality_drop_bad", lambda c: c["quantization"].update({"quant_quality": 0.80}))
    add("runtime_incompatible_bad", lambda c: c["runtime"].update({"supported_runtimes": ["tgi"]}))
    add("eval_report_missing_bad", lambda c: c["eval"]["reports"].pop("tool_call"))
    add("permission_open_bad", lambda c: c["security_permission"].update({"export_public": True}))
    add("release_unapproved_bad", lambda c: c["release"].update({"approvers": ["model-owner"]}))
    add("rollback_alias_missing_bad", lambda c: c["release"].update({"rollback_version": "company-chat-model:missing"}))
    add("artifact_lineage_edge_missing_bad", lambda c: c["lineage"]["edges"].remove(("merged_model", "quantized_model")))
    add("load_cache_not_ready_bad", lambda c: c["runtime"].update({"cache_prepared": False}))
    add("lifecycle_policy_missing_bad", lambda c: c["lifecycle"].update({"protected_from_delete": False}))
    add("registry_gate_missing_bad", lambda c: c.update({"model_registry_gate": False}))
    return cases


def summarize_examples(case):
    weights = case["weights"]
    quant = case["quantization"]
    runtime = case["runtime"]
    required_fields = sum(1 for field in REQUIRED_MODEL_FIELDS if case["model_version"].get(field))
    return {
        "artifact_metadata_completeness": round(ratio(required_fields, len(REQUIRED_MODEL_FIELDS)), 3),
        "weight_checksum_coverage": round(ratio(sum(1 for f in weights["files"] if f["checksum"]), len(weights["files"])), 3),
        "weight_total_size_gib": weights["total_size_gib"],
        "tokenizer_config_binding": tokenizer_config_binding(case),
        "adapter_rank_ratio": round(ratio(case["adapter"]["rank"], case["adapter"]["hidden_size"]), 4),
        "quant_quality_drop": round(quant["base_quality"] - quant["quant_quality"], 3),
        "quant_latency_speedup": quant["latency_speedup"],
        "estimated_load_time_s": round(load_time(runtime), 2),
        "load_time_slo": runtime["load_time_slo"],
        "lineage_node_count": len(case["lineage"]["nodes"]),
        "release_alias": case["model_version"]["release_alias"],
        "rollback_alias": case["model_version"]["rollback_alias"],
    }


def audit_model_artifact_registry(cases):
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
            "caught_weight_gap": "weight_checksum_missing_bad" in failed_cases,
            "caught_tokenizer_gap": "tokenizer_hash_mismatch_bad" in failed_cases,
            "caught_adapter_gap": "adapter_base_mismatch_bad" in failed_cases,
            "caught_quant_gap": "quant_quality_drop_bad" in failed_cases,
            "caught_runtime_gap": "runtime_incompatible_bad" in failed_cases,
            "caught_release_gap": "release_unapproved_bad" in failed_cases,
        },
        "metrics": metrics,
        "hard_blocker_count": len(failed_cases),
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "model_registry_gate_pass": len(failed_cases) == 0,
    }


base = complete_case()
cases = [base] + make_bad_cases(base)
report = audit_model_artifact_registry(cases)

pprint({"model_registry_examples": summarize_examples(base)})
pprint(report["smoke"])
pprint(report["metrics"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("model_registry_gate_pass=", report["model_registry_gate_pass"])

assert report["smoke"]["complete_case_passes"]
assert report["smoke"]["caught_runtime_gap"]
assert report["metrics"]["model_version_contract"] == 0.941
assert report["metrics"]["model_registry_gate"] == 0.941
assert report["hard_blocker_count"] == 16
assert report["model_registry_gate_pass"] is False
```

这个 demo 的重点不是复刻真实模型平台，而是把“模型仓库”拆成可检查证据：ModelVersion 契约、权重 manifest、safetensors 安全格式、tokenizer / config 绑定、adapter 和 base model 兼容、merge lineage、量化评估、runtime 兼容、评估报告、安全权限、发布状态、回滚 alias、artifact lineage、加载缓存、生命周期和最终 registry gate。

## 38.25 常见误区

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

## 38.26 面试常见追问

问题一：模型仓库和对象存储有什么区别？

可以回答：对象存储只负责保存文件，模型仓库还管理 ModelVersion、tokenizer、config、adapter、量化版本、评估报告、权限、血缘、发布状态和审计。

问题二：为什么 tokenizer 要版本化？

可以回答：tokenizer 决定文本和 token 的映射，特殊 token、chat template 和 stop token 都会影响模型行为。tokenizer 与权重不匹配会导致乱码、格式错误或能力退化。

问题三：如何管理 LoRA adapter？

可以回答：记录 adapter 版本、base model 版本、训练数据、训练任务、权重路径、兼容性、是否 merge 以及评估报告。adapter 不能脱离 base model 单独管理。

问题四：量化模型上线前要检查什么？

可以回答：检查原始模型版本、量化方法、校准数据、量化工具版本、runtime 兼容性、质量评估、延迟吞吐 benchmark 和成本收益。

## 38.27 小练习

1. Model 和 ModelVersion 有什么区别？
2. ModelVersion 应该记录哪些元数据？
3. 为什么权重文件需要 checksum？
4. tokenizer 和 chat template 为什么要绑定模型版本？
5. LoRA adapter 和 merge 后模型如何管理血缘？
6. 量化版本为什么需要单独评估？
7. 模型仓库如何和训练平台、评估平台、推理平台联动？
8. 模型生命周期管理要考虑哪些依赖？

## 38.28 本章小结

本章讲了模型仓库。

你需要记住：

1. 模型仓库不是普通目录，而是模型资产的版本、元数据、权限、血缘、评估和发布状态管理系统。
2. Model 是逻辑模型，ModelVersion 是具体版本，Artifact 是实际文件集合。
3. 权重、tokenizer、config、adapter 和量化版本都必须被纳入版本管理。
4. Adapter 依赖 base model，量化版本依赖原始模型和量化配置，它们都需要血缘关系。
5. 模型仓库连接训练、评估和推理平台，是模型资产生命周期的中心。

下一章我们会讲 Artifact 管理：dataset、checkpoint、eval report 和 deployment package。
