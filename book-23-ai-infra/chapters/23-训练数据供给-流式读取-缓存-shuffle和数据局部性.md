# 第 23 章 训练数据供给：流式读取、缓存、shuffle 和数据局部性

上一章讲了训练容错。本章讲训练平台中另一个决定训练效率的关键环节：数据供给。

很多训练任务 GPU 利用率低，不是模型算得慢，而是数据喂得慢。大模型训练需要持续把海量 token、样本、图像、视频或多模态数据送到 GPU。如果数据读取、预处理、shuffle、缓存任何一环跟不上，GPU 就会等待。

先记住一句话：

> 训练数据供给的目标，是在保证数据版本、权限、shuffle 和可复现性的前提下，把数据以足够高、足够稳定的吞吐送到训练进程。

## 23.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径主要对齐几类公开工程资料：PyTorch `torch.utils.data` 对 `Dataset`、`IterableDataset`、`DataLoader`、`DistributedSampler`、`num_workers`、`pin_memory`、`prefetch_factor` 和 worker 初始化的稳定抽象；Hugging Face Datasets streaming 对超大数据集流式读取、`IterableDataset`、shuffle buffer、`set_epoch`、sharding 和 Parquet 列裁剪 / 过滤的公开口径；NVIDIA DALI 对高吞吐数据加载、CPU / GPU preprocessing、pipeline、reader、sharding 和预取的工程边界；前文存储、checkpoint 和可观测性章节对对象存储、缓存、manifest、checksum、恢复状态和指标采集的稳定抽象。

这里不把某个数据湖、对象存储、分布式缓存产品、训练框架、文件格式、内部平台或具体吞吐数值写成通用标准。正文只抽象训练数据供给里的稳定问题：manifest、shard、格式、小文件、离线 / 在线预处理、流式读取、预取、本地缓存、分布式缓存、shuffle、rank 切分、数据局部性、DataLoader、H2D copy、坏样本、监控、checkpoint 恢复和权限版本治理。

第二轮精修重点放在三个方面：

1. 把“数据读得快”拆成可计算的吞吐、等待比例、缓存命中、rank lag、坏样本率和恢复连续性。
2. 补充 GitHub MathJax 兼容公式，解释 shard、shuffle、缓存、DataLoader 和 checkpoint 数据状态之间的关系。
3. 增加一个 0 依赖 Python demo，把训练数据供给从经验 checklist 变成可运行门禁检查。

## 23.1 为什么数据供给重要

大模型训练通常很贵。

如果 GPU 每秒能处理 10000 token，但数据 pipeline 只能提供 6000 token，那么 GPU 会空等。

数据供给问题会导致：

1. GPU 利用率低。
2. step time 抖动。
3. tokens/s 下降。
4. dataloader time 占比高。
5. 多个 rank 不同步。
6. 存储压力暴涨。
7. 训练任务偶发失败。

所以训练数据供给是 AI Infra 的主路径，不是数据工程的边角问题。

## 23.2 训练数据供给链路

一个典型链路：

```text
Raw Data
  -> Data Lake / Object Storage
  -> Cleaning / Dedup / Filtering
  -> Tokenization / Transform
  -> Sharding / Packing
  -> Dataset Manifest
  -> Cache / Local Disk
  -> Dataloader
  -> CPU Memory
  -> GPU
```

每一层都可能成为瓶颈。

平台需要知道：

1. 数据从哪里来。
2. 使用哪个版本。
3. 预处理规则是什么。
4. shard 如何组织。
5. cache 是否命中。
6. dataloader 是否够快。
7. 数据是否正确送到 GPU。

## 23.3 离线预处理和在线预处理

数据预处理可以分为离线和在线。

离线预处理：训练前完成。

包括：

1. 清洗。
2. 去重。
3. 过滤。
4. tokenization。
5. packing。
6. shard 生成。
7. index 生成。

优点：

1. 训练时更快。
2. 数据版本更稳定。
3. 便于复现。
4. 减少 CPU 压力。

缺点：

1. 预处理耗时。
2. 需要额外存储。
3. 数据更新不够实时。

在线预处理：训练时动态处理。

优点：

1. 灵活。
2. 节省预处理产物存储。
3. 适合快速实验。

缺点：

1. CPU 容易成为瓶颈。
2. 训练吞吐不稳定。
3. 可复现性更难。
4. 错误更晚暴露。

大规模训练通常倾向离线预处理，小实验可以保留在线灵活性。

## 23.4 数据格式和 shard

训练数据格式会影响吞吐。

常见格式：

1. JSONL。
2. Parquet。
3. Arrow。
4. WebDataset。
5. TFRecord。
6. 自定义二进制格式。

小文件太多会严重拖慢训练。

因此大规模训练常把数据打成 shard。

Shard 的好处：

1. 减少文件数量。
2. 提高顺序读取效率。
3. 降低 metadata 压力。
4. 便于分布式切分。
5. 便于缓存和预取。

Shard 设计要考虑：

1. 单个 shard 大小。
2. shard 数量。
3. 每个 shard 样本数。
4. 是否压缩。
5. 是否带 index。
6. 是否可随机访问。

Shard 太小会有小文件问题，太大又不利于并行和恢复。

## 23.5 Dataset Manifest

Dataset manifest 是数据集的目录和说明书。

它记录：

1. dataset_id。
2. version。
3. shard 列表。
4. 每个 shard 大小。
5. 样本数。
6. token 数。
7. checksum。
8. 数据来源。
9. preprocessing_version。
10. 权限和数据等级。

示例：

```json
{
  "dataset_id": "web_text_pretrain",
  "version": "2026-05-30",
  "preprocessing_version": "dedup_v4_tokenize_v2",
  "shards": [
    {
      "uri": "s3://datasets/web_text/shard_00001.bin",
      "num_tokens": 120000000,
      "checksum": "sha256:..."
    }
  ]
}
```

没有 manifest，训练平台很难保证数据版本可复现。

## 23.6 流式读取

流式读取是从远端存储或数据服务边读边训。

适合：

1. 数据集很大。
2. 本地盘放不下。
3. 数据持续更新。
4. 多任务共享数据。

挑战：

1. 网络波动影响训练。
2. 存储吞吐成为瓶颈。
3. 随机读取效率低。
4. cache 命中率关键。
5. 错误恢复复杂。

流式读取要配合：

1. 预取。
2. 本地缓存。
3. 分布式缓存。
4. 重试。
5. shard 顺序规划。
6. 吞吐监控。

否则很容易拖慢 GPU。

## 23.7 本地缓存

本地缓存是把热点数据放到训练节点本地盘。

优点：

1. 读取快。
2. 减少远端存储压力。
3. 降低网络波动影响。
4. 提高训练稳定性。

挑战：

1. 本地盘容量有限。
2. 缓存一致性管理复杂。
3. 节点故障后缓存丢失。
4. 多任务缓存竞争。

常见策略：

1. 任务启动前预热数据。
2. 按 shard 缓存。
3. LRU 清理。
4. 按数据版本隔离缓存。
5. 缓存命中率监控。

本地缓存适合重复使用的数据集和热点 shard。

## 23.8 分布式缓存

分布式缓存位于远端存储和训练节点之间。

它可以：

1. 缓解对象存储压力。
2. 提高多节点读取吞吐。
3. 减少重复下载。
4. 支持跨任务共享热点数据。

挑战：

1. 部署和运维复杂。
2. 容量规划困难。
3. 一致性和版本隔离要做好。
4. 可能成为新的瓶颈。

分布式缓存适合大规模预训练和多任务共享数据场景。

## 23.9 Shuffle

Shuffle 是训练数据随机化的重要环节。

Shuffle 不好会影响训练质量。

常见 shuffle 层次：

1. shard 级 shuffle。
2. shard 内样本 shuffle。
3. batch 内 shuffle。
4. 多数据源采样 shuffle。

大规模流式读取中，全局完全随机 shuffle 成本很高。

常见做法是近似 shuffle：

1. 随机打乱 shard 顺序。
2. 使用 shuffle buffer。
3. 每个 epoch 使用不同 seed。
4. 多 rank 分配不同 shard。

Shuffle 要兼顾：

1. 随机性。
2. 吞吐。
3. 可复现。
4. 数据均衡。

## 23.10 多 rank 数据切分

分布式训练中，不同 rank 不能重复处理同一批数据，除非算法允许。

需要处理：

1. rank 到 shard 的分配。
2. epoch 边界。
3. 数据不足时是否 drop。
4. 断点恢复时数据位置。
5. world size 变化时重新切分。

常见问题：

1. 不同 rank 读到重复数据。
2. 某些 shard 被漏读。
3. 某个 rank 数据读取慢。
4. 恢复后重复训练一段数据。
5. 数据分布不均导致训练偏差。

平台要记录 dataloader state，支持恢复。

## 23.11 数据局部性

数据局部性是指计算尽量靠近数据。

在训练中，数据局部性包括：

1. 数据是否在本地缓存。
2. 数据是否在同机柜存储节点。
3. 数据是否在同区域对象存储。
4. 训练节点到数据路径是否拥塞。

数据局部性差会导致：

1. 读取延迟高。
2. 跨区域流量贵。
3. 网络拥塞。
4. step time 抖动。

调度系统可以利用数据局部性：

1. 优先调度到有缓存的节点。
2. 把同一数据集任务放在缓存附近。
3. 避免跨区域读取训练数据。
4. 对热点数据提前预热。

## 23.12 Dataloader 性能

Dataloader 是训练数据到模型之间的最后一段。

常见瓶颈：

1. worker 数太少。
2. CPU 预处理太重。
3. Python 解析慢。
4. 解压耗时。
5. tokenization 在线执行。
6. batch padding 浪费。
7. CPU 到 GPU 拷贝慢。

优化方法：

1. 增加 worker。
2. 预处理离线化。
3. 使用二进制格式。
4. 使用 pinned memory。
5. 异步预取。
6. 减少动态解析。
7. 合理 batch packing。

数据 pipeline 优化常常比改模型更能提升 GPU 利用率。

## 23.13 坏样本处理

训练数据中可能有坏样本。

例如：

1. 格式错误。
2. 解码失败。
3. 空文本。
4. token 超长。
5. 图片损坏。
6. 视频无法读取。
7. 标签缺失。

处理策略：

1. 离线校验。
2. 训练中跳过并记录。
3. 超过阈值失败。
4. 记录坏样本位置。
5. 生成数据质量报告。

不能无限跳过坏样本，否则训练数据分布可能被悄悄改变。

## 23.14 数据供给监控

关键指标：

1. dataloader time。
2. data wait time。
3. storage read throughput。
4. cache hit rate。
5. shard read latency。
6. bad sample count。
7. tokens loaded per second。
8. CPU preprocessing time。
9. H2D copy time。
10. per-rank data lag。

告警场景：

1. dataloader time 占 step time 过高。
2. cache hit rate 下降。
3. 某个 rank 数据读取明显慢。
4. 坏样本比例过高。
5. 存储吞吐接近上限。
6. 数据读取错误率上升。

这些指标能帮助判断 GPU 是不是在等数据。

## 23.15 数据供给和 checkpoint 恢复

断点恢复时，不只要恢复模型，也要恢复数据位置。

否则可能：

1. 重复训练一段数据。
2. 跳过一段数据。
3. shuffle 顺序变化。
4. 多 rank 数据分配变化。

Checkpoint 应记录：

1. epoch。
2. global step。
3. shard index。
4. sample offset。
5. shuffle seed。
6. dataloader worker state。

严格复现要求更高，平台要根据任务需求选择支持程度。

## 23.16 训练数据供给审计指标与最小 demo

这一节把训练数据供给改写成可审计指标。真实平台可以从 DataLoader profile、对象存储日志、缓存服务、dataset manifest、训练事件、rank trace 和 checkpoint metadata 中取数；教学 demo 里先用 toy 字段模拟。

一个训练数据供给样本可以抽象成：

```math
d_i=(m_i,s_i,a_i,u_i,c_i,q_i,r_i,\ell_i,p_i,h_i,b_i,o_i,k_i,v_i,z_i)
```

其中，`m_i` 是 manifest 证据，`s_i` 是 shard 和格式，`a_i` 是小文件和 metadata 压力，`u_i` 是流式读取和预取，`c_i` 是缓存，`q_i` 是 shuffle，`r_i` 是 rank 切分，`\ell_i` 是数据局部性，`p_i` 是 DataLoader 并行与预处理，`h_i` 是 H2D copy，`b_i` 是坏样本，`o_i` 是观测指标，`k_i` 是 checkpoint 数据状态，`v_i` 是权限和版本治理，`z_i` 是最终门禁。

统一覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(d_i)=1]
```

数据供给吞吐可以写成：

```math
Q_{\mathrm{data}}=\frac{N_{\mathrm{tok}}}{T_{\mathrm{load}}}
```

其中，`N_tok` 是数据 pipeline 在时间窗口内真正送到训练循环的 token 数，`T_load` 是对应的数据加载耗时。

GPU 等数据的比例可以写成：

```math
R_{\mathrm{wait}}=\frac{T_{\mathrm{data\_wait}}}{T_{\mathrm{step}}}
```

如果 `R_wait` 长期很高，说明瓶颈可能在对象存储、缓存、解析、预处理、DataLoader worker、H2D copy 或 rank 不均衡，而不一定在模型计算。

缓存命中率可以写成：

```math
R_{\mathrm{cache}}=\frac{B_{\mathrm{cache}}}{B_{\mathrm{cache}}+B_{\mathrm{remote}}}
```

其中，`B_cache` 是从本地或分布式缓存读取的字节数，`B_remote` 是从远端存储读取的字节数。

rank 间数据滞后可以写成：

```math
A_{\mathrm{rank}}=\max_r T_{\mathrm{data},r}-\min_r T_{\mathrm{data},r}
```

如果某个 rank 数据读取慢，其他 rank 可能在 collective 前等待，最后表现成 GPU 利用率低或 NCCL 同步变慢。

坏样本率可以写成：

```math
R_{\mathrm{bad}}=\frac{N_{\mathrm{bad}}}{N_{\mathrm{seen}}}
```

坏样本不是越过越好。超过阈值时应该失败并输出数据质量报告，否则训练分布会被悄悄改变。

近似 shuffle 的桶分布熵可以写成：

```math
H_{\mathrm{shuffle}}=-\frac{1}{\log K}\sum_{k=1}^{K}p_k\log p_k
```

其中，`K` 是桶数，`p_k` 是第 `k` 个来源桶在一个窗口中的比例。这个指标不能证明全局完全随机，但能发现明显的 shard 顺序偏置或单来源连续读取。

最终训练数据供给门禁可以写成：

```math
G_{\mathrm{data}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land Q_{\mathrm{data}}\ge \gamma Q_{\mathrm{train}} \land R_{\mathrm{wait}}\le \rho_{\mathrm{wait}} \land R_{\mathrm{cache}}\ge \rho_{\mathrm{cache}} \land R_{\mathrm{bad}}\le \rho_{\mathrm{bad}} \land P_0=0\right]
```

这个门禁背后的含义是：数据供给不是“能读到文件”，而是 manifest、shard、缓存、shuffle、rank 切分、DataLoader、H2D、坏样本、观测指标和 checkpoint 数据状态都要可审计。

下面的 0 依赖 Python demo 演示一个最小训练数据供给审计器。

```python
from copy import deepcopy


GATES = [
    "dataset_manifest_integrity",
    "shard_format_size_fit",
    "small_file_amplification_control",
    "streaming_prefetch_retry",
    "local_cache_policy",
    "distributed_cache_policy",
    "shuffle_reproducibility",
    "rank_shard_assignment",
    "data_locality_fit",
    "dataloader_parallelism_fit",
    "h2d_copy_readiness",
    "bad_sample_threshold",
    "data_supply_observability",
    "checkpoint_dataloader_state",
    "permission_version_governance",
    "training_data_supply_gate",
]


def base_case():
    return {
        "name": "complete_data_supply_case",
        "manifest": {
            "dataset_id": "web_text_pretrain",
            "version": "2026-05-30",
            "immutable_version": True,
            "preprocessing_version": "dedup_v4_tokenize_v2_pack_v3",
            "shard_count": 64,
            "checksum_coverage": 1.0,
            "permission_metadata": True,
            "token_count": 320_000_000_000,
        },
        "shards": {
            "format": "packed_binary",
            "avg_size_mib": 512,
            "min_size_mib": 420,
            "max_size_mib": 620,
            "indexed": True,
            "compression_splittable": True,
            "total_files": 64,
            "metadata_open_qps": 32,
        },
        "streaming": {
            "enabled": True,
            "prefetch_mib": 2048,
            "retry_policy": True,
            "timeout_s": 60,
            "observed_mib_s": 640,
            "required_mib_s": 512,
        },
        "local_cache": {
            "enabled": True,
            "hit_bytes": 800,
            "remote_bytes": 200,
            "version_isolation": True,
            "lru_eviction": True,
            "capacity_ok": True,
        },
        "distributed_cache": {
            "enabled": True,
            "hot_shard_shared": True,
            "version_consistent": True,
            "capacity_ok": True,
            "not_single_bottleneck": True,
        },
        "shuffle": {
            "shard_shuffle": True,
            "buffer_size": 100_000,
            "seed_recorded": True,
            "set_epoch": True,
            "bucket_entropy": 0.96,
        },
        "rank_assignment": {
            "world_size": 8,
            "rank_shard_counts": [8, 8, 8, 8, 8, 8, 8, 8],
            "duplicate_shards": 0,
            "missing_shards": 0,
            "resume_cursor_saved": True,
        },
        "locality": {
            "same_region": True,
            "cache_affinity": True,
            "cross_region_read": False,
            "topology_aware_schedule": True,
            "locality_score": 0.9,
        },
        "dataloader": {
            "num_workers": 8,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "cpu_preprocess_ms": 80,
            "dataloader_ms": 140,
            "step_ms": 1200,
        },
        "h2d": {
            "pin_memory": True,
            "async_copy": True,
            "copy_ms": 18,
            "copy_budget_ms": 40,
        },
        "bad_samples": {
            "seen": 1_000_000,
            "bad": 1000,
            "threshold": 0.002,
            "logged_locations": True,
            "fail_over_threshold": True,
        },
        "observability": {
            "metrics": [
                "dataloader_time",
                "data_wait_time",
                "storage_read_throughput",
                "cache_hit_rate",
                "shard_read_latency",
                "bad_sample_count",
                "tokens_loaded_per_second",
                "cpu_preprocess_time",
                "h2d_copy_time",
                "per_rank_data_lag",
            ]
        },
        "checkpoint_data_state": {
            "epoch": 2,
            "global_step": 1200,
            "shard_index": 41,
            "sample_offset": 9984,
            "shuffle_seed": 20260612,
            "worker_state": True,
        },
        "permission_version": {
            "access_checked": True,
            "data_classification": "internal",
            "dataset_version_bound": True,
            "preprocessing_version_bound": True,
        },
        "tokens_loaded": 524_288,
        "load_seconds": 1.2,
        "train_tokens_per_second": 400_000,
        "data_wait_ms": 100,
        "training_data_supply_gate": True,
    }


def contains_all(values, required):
    return set(required).issubset(set(values))


def rate(num, den):
    return num / max(den, 1e-9)


def evaluate(case):
    manifest = case["manifest"]
    shards = case["shards"]
    streaming = case["streaming"]
    local_cache = case["local_cache"]
    distributed_cache = case["distributed_cache"]
    shuffle = case["shuffle"]
    rank_assignment = case["rank_assignment"]
    locality = case["locality"]
    dataloader = case["dataloader"]
    h2d = case["h2d"]
    bad_samples = case["bad_samples"]
    checkpoint_state = case["checkpoint_data_state"]
    permission_version = case["permission_version"]

    shard_counts = rank_assignment["rank_shard_counts"]
    rank_balance = max(shard_counts) / max(min(shard_counts), 1)
    data_wait_ratio = rate(case["data_wait_ms"], dataloader["step_ms"])
    cache_hit = rate(local_cache["hit_bytes"], local_cache["hit_bytes"] + local_cache["remote_bytes"])
    dataloader_ratio = rate(dataloader["dataloader_ms"], dataloader["step_ms"])
    bad_rate = rate(bad_samples["bad"], bad_samples["seen"])

    required_metrics = [
        "dataloader_time",
        "data_wait_time",
        "storage_read_throughput",
        "cache_hit_rate",
        "shard_read_latency",
        "bad_sample_count",
        "tokens_loaded_per_second",
        "cpu_preprocess_time",
        "h2d_copy_time",
        "per_rank_data_lag",
    ]
    required_checkpoint_fields = ["epoch", "global_step", "shard_index", "sample_offset", "shuffle_seed", "worker_state"]

    checks = {}
    checks["dataset_manifest_integrity"] = (
        manifest["immutable_version"]
        and manifest["shard_count"] > 0
        and manifest["checksum_coverage"] >= 1.0
        and bool(manifest["preprocessing_version"])
        and manifest["permission_metadata"]
    )
    checks["shard_format_size_fit"] = (
        shards["format"] in ["packed_binary", "parquet", "arrow", "webdataset"]
        and 128 <= shards["avg_size_mib"] <= 4096
        and shards["indexed"]
        and shards["compression_splittable"]
    )
    checks["small_file_amplification_control"] = (
        rate(shards["total_files"], manifest["shard_count"]) <= 2
        and shards["metadata_open_qps"] <= 200
    )
    checks["streaming_prefetch_retry"] = (
        streaming["enabled"]
        and streaming["prefetch_mib"] >= 1024
        and streaming["retry_policy"]
        and streaming["timeout_s"] > 0
        and streaming["observed_mib_s"] >= streaming["required_mib_s"]
    )
    checks["local_cache_policy"] = (
        local_cache["enabled"]
        and local_cache["version_isolation"]
        and local_cache["lru_eviction"]
        and local_cache["capacity_ok"]
        and cache_hit >= 0.6
    )
    checks["distributed_cache_policy"] = (
        distributed_cache["enabled"]
        and distributed_cache["hot_shard_shared"]
        and distributed_cache["version_consistent"]
        and distributed_cache["capacity_ok"]
        and distributed_cache["not_single_bottleneck"]
    )
    checks["shuffle_reproducibility"] = (
        shuffle["shard_shuffle"]
        and shuffle["buffer_size"] >= 10_000
        and shuffle["seed_recorded"]
        and shuffle["set_epoch"]
        and shuffle["bucket_entropy"] >= 0.8
    )
    checks["rank_shard_assignment"] = (
        rank_assignment["duplicate_shards"] == 0
        and rank_assignment["missing_shards"] == 0
        and rank_balance <= 1.25
        and rank_assignment["resume_cursor_saved"]
    )
    checks["data_locality_fit"] = (
        locality["same_region"]
        and locality["cache_affinity"]
        and not locality["cross_region_read"]
        and locality["topology_aware_schedule"]
        and locality["locality_score"] >= 0.8
    )
    checks["dataloader_parallelism_fit"] = (
        dataloader["num_workers"] >= 4
        and dataloader["persistent_workers"]
        and dataloader["prefetch_factor"] >= 2
        and dataloader_ratio <= 0.2
    )
    checks["h2d_copy_readiness"] = (
        h2d["pin_memory"]
        and h2d["async_copy"]
        and h2d["copy_ms"] <= h2d["copy_budget_ms"]
    )
    checks["bad_sample_threshold"] = (
        bad_rate <= bad_samples["threshold"]
        and bad_samples["logged_locations"]
        and bad_samples["fail_over_threshold"]
    )
    checks["data_supply_observability"] = contains_all(case["observability"]["metrics"], required_metrics)
    checks["checkpoint_dataloader_state"] = (
        contains_all(checkpoint_state.keys(), required_checkpoint_fields)
        and checkpoint_state["worker_state"]
    )
    checks["permission_version_governance"] = (
        permission_version["access_checked"]
        and permission_version["dataset_version_bound"]
        and permission_version["preprocessing_version_bound"]
        and bool(permission_version["data_classification"])
    )
    checks["training_data_supply_gate"] = (
        bool(case["training_data_supply_gate"])
        and data_wait_ratio <= 0.1
        and case["tokens_loaded"] / case["load_seconds"] >= case["train_tokens_per_second"]
    )
    return checks


def summarize_examples(case):
    shard_counts = case["rank_assignment"]["rank_shard_counts"]
    return {
        "data_tokens_per_second": round(case["tokens_loaded"] / case["load_seconds"]),
        "observed_data_mib_s": case["streaming"]["observed_mib_s"],
        "data_wait_ratio": round(rate(case["data_wait_ms"], case["dataloader"]["step_ms"]), 3),
        "cache_hit_rate": round(rate(case["local_cache"]["hit_bytes"], case["local_cache"]["hit_bytes"] + case["local_cache"]["remote_bytes"]), 3),
        "bad_sample_rate": round(rate(case["bad_samples"]["bad"], case["bad_samples"]["seen"]), 4),
        "rank_balance": round(max(shard_counts) / max(min(shard_counts), 1), 3),
    }


def mutate(name, editor):
    case = deepcopy(base_case())
    case["name"] = name
    editor(case)
    return case


def build_cases():
    cases = [base_case()]
    cases.append(mutate("manifest_missing_bad", lambda c: c["manifest"].update({"checksum_coverage": 0.5, "immutable_version": False})))
    cases.append(mutate("shard_format_bad", lambda c: c["shards"].update({"format": "raw_jsonl", "indexed": False})))
    cases.append(mutate("small_file_amplification_bad", lambda c: c["shards"].update({"total_files": 20000, "metadata_open_qps": 5000})))
    cases.append(mutate("streaming_prefetch_missing_bad", lambda c: c["streaming"].update({"prefetch_mib": 0, "retry_policy": False, "observed_mib_s": 128})))
    cases.append(mutate("local_cache_policy_bad", lambda c: c["local_cache"].update({"hit_bytes": 100, "remote_bytes": 900, "version_isolation": False})))
    cases.append(mutate("distributed_cache_policy_bad", lambda c: c["distributed_cache"].update({"version_consistent": False, "not_single_bottleneck": False})))
    cases.append(mutate("shuffle_not_reproducible_bad", lambda c: c["shuffle"].update({"seed_recorded": False, "set_epoch": False, "bucket_entropy": 0.4})))
    cases.append(mutate("rank_assignment_duplicate_bad", lambda c: c["rank_assignment"].update({"duplicate_shards": 2, "missing_shards": 1})))
    cases.append(mutate("data_locality_cross_region_bad", lambda c: c["locality"].update({"same_region": False, "cross_region_read": True, "locality_score": 0.3})))
    cases.append(mutate("dataloader_parallelism_bad", lambda c: c["dataloader"].update({"num_workers": 1, "persistent_workers": False, "dataloader_ms": 420})))
    cases.append(mutate("h2d_copy_blocking_bad", lambda c: c["h2d"].update({"pin_memory": False, "async_copy": False, "copy_ms": 90})))
    cases.append(mutate("bad_sample_threshold_bad", lambda c: c["bad_samples"].update({"bad": 20_000, "logged_locations": False})))
    cases.append(mutate("observability_missing_bad", lambda c: c["observability"].update({"metrics": ["dataloader_time", "cache_hit_rate"]})))
    cases.append(mutate("checkpoint_dataloader_state_missing_bad", lambda c: c["checkpoint_data_state"].pop("worker_state")))
    cases.append(mutate("permission_version_unbound_bad", lambda c: c["permission_version"].update({"access_checked": False, "dataset_version_bound": False})))
    cases.append(mutate("data_supply_gate_missing_bad", lambda c: c.update({"training_data_supply_gate": False})))
    return cases


def main():
    cases = build_cases()
    evaluations = {case["name"]: evaluate(case) for case in cases}
    metrics = {
        gate: round(sum(1 for case in cases if evaluations[case["name"]][gate]) / len(cases), 3)
        for gate in GATES
    }
    failed_cases = [
        case["name"]
        for case in cases
        if not all(evaluations[case["name"]].values())
    ]
    failed_gates = [
        gate
        for gate in GATES
        if any(not evaluations[case["name"]][gate] for case in cases)
    ]
    complete = cases[0]
    smoke = {
        "complete_case_passes": all(evaluations[complete["name"]].values()),
        "caught_small_files": not evaluations["small_file_amplification_bad"]["small_file_amplification_control"],
        "caught_cache_gap": not evaluations["local_cache_policy_bad"]["local_cache_policy"],
        "caught_shuffle_gap": not evaluations["shuffle_not_reproducible_bad"]["shuffle_reproducibility"],
        "caught_rank_overlap": not evaluations["rank_assignment_duplicate_bad"]["rank_shard_assignment"],
        "caught_checkpoint_state_gap": not evaluations["checkpoint_dataloader_state_missing_bad"]["checkpoint_dataloader_state"],
    }
    training_data_supply_gate_pass = not failed_cases and all(v >= 1.0 for v in metrics.values())

    print("training_data_supply_examples=", summarize_examples(complete))
    print("smoke=", smoke)
    print("metrics=", metrics)
    print("hard_blocker_count=", len(failed_cases))
    print("failed_cases=", failed_cases)
    print("failed_gates=", failed_gates)
    print("training_data_supply_gate_pass=", training_data_supply_gate_pass)


if __name__ == "__main__":
    main()
```

预期输出类似：

```text
training_data_supply_examples= {'data_tokens_per_second': 436907, 'observed_data_mib_s': 640, 'data_wait_ratio': 0.083, 'cache_hit_rate': 0.8, 'bad_sample_rate': 0.001, 'rank_balance': 1.0}
smoke= {'complete_case_passes': True, 'caught_small_files': True, 'caught_cache_gap': True, 'caught_shuffle_gap': True, 'caught_rank_overlap': True, 'caught_checkpoint_state_gap': True}
metrics= {'dataset_manifest_integrity': 0.941, 'shard_format_size_fit': 0.941, 'small_file_amplification_control': 0.941, 'streaming_prefetch_retry': 0.941, 'local_cache_policy': 0.941, 'distributed_cache_policy': 0.941, 'shuffle_reproducibility': 0.941, 'rank_shard_assignment': 0.941, 'data_locality_fit': 0.941, 'dataloader_parallelism_fit': 0.941, 'h2d_copy_readiness': 0.941, 'bad_sample_threshold': 0.941, 'data_supply_observability': 0.941, 'checkpoint_dataloader_state': 0.941, 'permission_version_governance': 0.941, 'training_data_supply_gate': 0.941}
hard_blocker_count= 16
failed_cases= ['manifest_missing_bad', 'shard_format_bad', 'small_file_amplification_bad', 'streaming_prefetch_missing_bad', 'local_cache_policy_bad', 'distributed_cache_policy_bad', 'shuffle_not_reproducible_bad', 'rank_assignment_duplicate_bad', 'data_locality_cross_region_bad', 'dataloader_parallelism_bad', 'h2d_copy_blocking_bad', 'bad_sample_threshold_bad', 'observability_missing_bad', 'checkpoint_dataloader_state_missing_bad', 'permission_version_unbound_bad', 'data_supply_gate_missing_bad']
failed_gates= ['dataset_manifest_integrity', 'shard_format_size_fit', 'small_file_amplification_control', 'streaming_prefetch_retry', 'local_cache_policy', 'distributed_cache_policy', 'shuffle_reproducibility', 'rank_shard_assignment', 'data_locality_fit', 'dataloader_parallelism_fit', 'h2d_copy_readiness', 'bad_sample_threshold', 'data_supply_observability', 'checkpoint_dataloader_state', 'permission_version_governance', 'training_data_supply_gate']
training_data_supply_gate_pass= False
```

面试里可以把这个 demo 压缩成一句话：训练数据供给要证明数据“版本正确、读得够快、shuffle 可复现、rank 不重不漏、缓存有效、坏样本可控、恢复不乱序”，而不是只证明路径能打开。

## 23.17 面试中如何回答训练数据供给

如果面试官问：

```text
如何设计大模型训练的数据供给系统？
```

可以这样回答：

```text
我会把训练数据供给设计成从数据湖或对象存储到训练进程的高吞吐链路。原始数据先离线清洗、去重、过滤、tokenize、packing，并生成训练友好的 shard 和 dataset manifest。Manifest 记录 shard、token 数、checksum、数据版本、预处理版本和权限信息，保证可复现。

训练时根据任务规模选择流式读取、本地缓存或分布式缓存。大规模预训练会用 shard 顺序读取、预取和缓存来减少对象存储压力；重复使用的数据可以预热到本地 NVMe 或缓存集群。

Shuffle 采用 shard 级 shuffle、shuffle buffer 和固定 seed，平衡随机性、吞吐和可复现。分布式训练中要按 rank 切分数据，避免重复或漏读，并在 checkpoint 中记录 dataloader state 支持断点恢复。

平台还要监控 dataloader time、data wait time、cache hit rate、storage throughput、bad sample count 和 per-rank data lag。如果 GPU 利用率低，要能判断是不是数据供给跟不上。
```

## 23.18 常见误区

误区一：数据在对象存储里，训练直接读就行。

大规模训练直接读对象存储可能受延迟、请求开销和吞吐限制，需要 shard、预取和缓存。

误区二：shuffle 越随机越好。

完全全局随机成本很高，要在随机性、吞吐和可复现性之间平衡。

误区三：跳过坏样本没影响。

坏样本比例高会改变数据分布，必须记录和设阈值。

误区四：训练恢复只需要模型 checkpoint。

严格恢复还需要数据位置、shuffle seed 和 dataloader state。

误区五：GPU 利用率低一定是模型代码问题。

也可能是数据读取、预处理、缓存、存储或 CPU 到 GPU 拷贝问题。

## 23.19 面试题

### 题 1：为什么大规模训练要做数据 shard？

答：Shard 可以减少小文件数量，提高顺序读取效率，降低 metadata 压力，便于分布式切分、缓存、预取和恢复。小文件过多会严重拖慢 dataloader。

### 题 2：Dataset manifest 有什么作用？

答：Manifest 记录数据集版本、shard 列表、样本数、token 数、checksum、预处理版本、数据来源和权限信息，是数据可复现、完整性校验和分布式读取的基础。

### 题 3：流式读取和本地缓存如何取舍？

答：流式读取适合超大数据和动态数据，但依赖网络和远端存储吞吐。本地缓存速度快、稳定，但容量有限且管理复杂。大规模训练常用对象存储作为源，本地或分布式缓存做加速。

### 题 4：shuffle 在流式训练中如何做？

答：通常采用近似 shuffle，例如 shard 级随机顺序、shuffle buffer、固定 seed、多 rank 不同 shard 分配。要平衡随机性、吞吐和可复现性。

### 题 5：如何判断 GPU 在等数据？

答：看 dataloader time、data wait time、storage throughput、cache hit rate、CPU preprocessing time、H2D copy time、per-rank data lag 和 GPU utilization。如果 dataloader 占 step time 高，可能是数据供给瓶颈。

## 23.20 小练习

练习一：设计一个 dataset manifest。

要求：包含 dataset_id、version、preprocessing_version、shards、num_tokens、checksum 和 data_classification。

练习二：设计一个数据缓存策略。

要求：说明哪些数据预热到本地 NVMe，哪些走流式读取，如何清理缓存。

练习三：分析 GPU 利用率低。

假设 GPU utilization 只有 45%，dataloader time 占 step time 40%，请从 shard、小文件、缓存、CPU 预处理、H2D 拷贝和存储吞吐角度排查。

练习四：设计断点恢复的数据状态。

要求：说明 checkpoint 中应该保存哪些 dataloader 和 shuffle 状态。

## 23.21 本章小结

本章讲了训练数据供给。

你需要掌握：

1. 数据供给目标是在保证版本、权限、shuffle 和可复现性的前提下高吞吐喂给 GPU。
2. 数据链路包括数据湖、清洗、去重、tokenization、shard、manifest、缓存、dataloader 和 GPU。
3. 大规模训练倾向离线预处理，减少训练时 CPU 和解析开销。
4. Shard 和 manifest 是训练数据供给的基础。
5. 流式读取需要预取、缓存和重试配合。
6. 本地缓存和分布式缓存能提升吞吐，但需要容量和一致性管理。
7. Shuffle 要平衡随机性、吞吐和可复现性。
8. 分布式训练要按 rank 切分数据，并支持断点恢复。
9. 坏样本要记录、限阈值和生成质量报告。
10. 数据供给监控能帮助判断 GPU 是否在等数据。

下一章我们会讲训练平台权限、安全和审计。
