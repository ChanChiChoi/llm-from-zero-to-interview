# 第四册第 10 章内容精修记录（2026-10-23）

## 背景与前轮状态

- 前轮（commit `3ec6661` 及之前）对本章只做了核验级处理：`\operatorname→\mathrm`、链接核对、代码跑通。内容深度未达 `WRITING_SPEC.md` §四的标准。
- 本轮为内容精修补做，属于 progress_v2 中第 10 章验收的覆盖与升级。

## 本轮修改（全部为内容修改，非格式修复）

修改文件：`book-04-llm-encyclopedia/chapters/10-分布式训练系统.md`（+约 50 行）。

### 1. 11 个 python 块全部补齐"前观察目标 + 后输出解释"

每个代码块前新增一句观察目标说明；代码后新增输出解释、可修改的关键数值、教学实现与真实系统的差距。逐块内容：

| 位置（节） | 前文 | 后文要点 |
| --- | --- | --- |
| 10.1 global_batch | 验证两个 batch/并行关系 | 输出 64；改 DP=8→128；world size 不符触发 ValueError；真实中 packing 使 token 浮动 |
| 10.2 ring_all_reduce | 观察发送量与时间 | 输出 14.0；rank 8→64 趋向 16 GiB（ring 带宽优势）；1 TB/s→25 GB/s 约 14ms→0.6s；NCCL 树/环、chunk、overlap 使实测高于下界 |
| 10.3 column/row shape | 观察每卡张量 shape | 输出两组 shape；parts=3 报错；真实 Megatron 还需 head 分配、QKV 布局、gated MLP 同分片 |
| 10.4 pipeline_bubble | 公式数值代入（3/15=0.2；M=48→0.059） | 输出 0.2；stage 8 时 7/19≈0.368；1F1B/interleaved 改变实际值 |
| 10.5 sharded peak | 两笔账算成数字 | 240/8=30 GiB；峰值 56 GiB 中 all-gather 占 16；ranks=16 时峰值 41 GiB（all-gather 项不随 DP 缩小）；wrap policy/profiler 决定真实峰值 |
| 10.6 梯度累积+NaN | 归一化分母错误 vs 正确；非有限跳过 | 4.0 vs 3.0 反例；NaN→updated=False；GradScaler 动态 scale、DDP no-sync |
| 10.7 step_breakdown | 观察 data/collective 比例 | 0.167；data 1→4 时 data ratio 1/3；真实各项 overlap，相加是上界 |
| 10.8 classify_fault | 第一分类器=路由到证据类型 | communication 输出与阈值 0.3 的可改性；真实需时间序列、rank 维度、变更记录 |
| 10.11 manifest_digest | 提交语义 | 前缀 314b8834b309；committed=False 拒绝；真实需逐文件 checksum、分片映射、原子 rename |
| 10.13 dominant_share | max 取最紧缺资源 + 数值例子 0.25 | can_borrow 预留边界；真实调度器（DRF/fair-tree/Kueue）加时间衰减与 gang |
| 10.17 burn_rate | 数值代入（B=0.01，β=2） | 浮点尾差说明；slo=1→inf 边界；真实用多窗口 SLO 与坏事件口径 |

### 2. 公式改进

- 10.4 bubble 公式：补变量代入式数值例子（P=4,M=12→0.2）。
- 10.7 MFU：补 `F_model/T_compute/F_peak` 三个变量的说明；新增完整数值例子（400k tok/s × 6N(7B)=1.68e16 FLOPs/s ÷ 989 TFLOPs/s ≈ 17%，并说明 40% 已属高效的口径原因）。
- 10.3 column parallel：将"每张卡可以独立计算自己的输出列"改写为有具体机制的说法（同一输入乘不同权重列、各得部分输出列、此步无需交换数据）。
- 10.13 dominant share：补"max 决定于最紧缺资源"的直觉 + 16 GPU/256 CPU 租户 4/32 的数值例子。

## 验证

- 11 个 python 块在本机全部 exec 通过，打印值与新增后文说明逐一核对（含 burn_rate 浮点尾差 1.9999999999999982、manifest 前缀 314b8834b309 已按实测值写）。
- 新增数值例子全部手工/脚本复核：MFU≈0.17、bubble(8,12)=7/19、ring 64 rank=15.75 GiB、14 GiB@25GB/s≈0.60 s、峰值 41 GiB（初稿写 31 GiB 系笔误，已修正）。
- 结构检查：32 math 块、16 text 块、11 python 块，围栏配对平衡；无 `$$`、无 `\operatorname`、无禁用句式（显然/本质上就是等，grep 0 命中）。
- `git diff --check` 干净。

## 遗留

- PDF dry-run：环境无 Pandoc（长期遗留）。
- 第 11—18 章需按同一内容精修标准重做（前轮仅核验级）。
