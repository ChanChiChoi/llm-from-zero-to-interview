# 第四册第十份正文《分布式训练系统》历史验证记录与本轮精修证据（2026-10-23）

> 本文件前半部分是此前的验证记录，不等同于按 `WRITING_SPEC.md` 完成全章精修。本轮从第四册第十章重新开始；本轮仍在进行中，不能据历史验证条目把章节标记为精修完成。

对象：`book-04-llm-encyclopedia/chapters/10-分布式训练系统.md`（10.1—10.24，1,249 行）。

## 结构与格式

- 全章 59 个 `~~~` 围栏成对闭合（32 math、16 text、11 python），与第四册已验收章节的围栏风格一致，GitHub 可稳定渲染。
- 修正 1 处 `\operatorname{Optimizer}` 为 `\mathrm{Optimizer}`（WRITING_SPEC 禁用项）；转换后无 `\operatorname`、无 `$$`、无尖括号下标；32 个 math 围栏花括号平衡。

## 来源核验

- 18 个唯一外链经指定代理 `10.24.27.134:7890` 全部 HTTP 200（5 篇 arXiv、5 个 PyTorch 文档、NCCL、Kubernetes、Kueue、MLflow、OpenTelemetry、NIST）。
- 5 篇 arXiv 题名经 abs 页面核对，发现并修正 2 处错误：
  1. GPipe 题名由正文的 "Easy Scaling with Micro-Batch Pipeline Parallelism" 修正为实际题名 "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"（1811.06965）。
  2. 10.23 的 Data Cards 引用错链 `2203.15556`（实为 Chinchilla 论文 Training Compute-Optimal Large Language Models），修正为 `2204.01075`（Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI）。
- ZeRO 1910.02054、Megatron-LM 2104.04473、ZeRO-Infinity 2104.07857 题名与引用主题一致。

## 代码与数值核验

- 11 个 Python 块逐块独立执行，11/11 通过（标准库，无第三方依赖，未安装 CUDA）。
- 数值复算：ring all-reduce 8 GiB/8 rank=14 GiB、bubble(4,12)=0.2、ZeRO 240/8=30 GiB、dominant share=0.25、burn rate=2.0、global batch 2×8×4=64、column/row 并行切分 11008/8=1376 与 4096/8=512，均与正文一致。
- 公式边界检查：全局 batch 与并行维度乘积、ring 通信量近似、GPipe bubble 近似、token 级全局 loss 汇总、梯度累积归一化、loss scaling、step time 分解上界、MFU/HFU 口径区分、dominant share、碎片率、错误预算/burn rate 定义域均有正文限定。

## 配套同步

- `PAPERS.md` 分布式训练条目补入 Megatron-LM GPU 集群训练（2104.04473）与 ZeRO-Infinity（2104.07857）。
- `EXERCISES.md` 已覆盖分布式事故审计、训练效率、K8s GPU、调度治理、平台设计等主题；`INTERVIEW_BANK.md` 已有 DDP/FSDP、ZeRO、MFU/HFU 问答；术语表已有 ZeRO/MFU/张量并行等条目；不重复添加。
- 第四册目录第 10 章映射正确；正文无相对链接。

## 未验证项

- 未在真实 GPU/NCCL/多节点环境验证通信量、MFU、吞吐或故障恢复；未运行 ZeRO/FSDP/Megatron 实际训练；未复现论文实验。
- 公式为教学近似（ring、bubble、显存账本、碎片率），不能预测目标集群实测值，正文已声明。
- PDF 构建受环境缺少 Pandoc 限制，未执行。

## 重新启动精修时的来源复核（2026-10-23）

通过指定代理重新读取以下五个 arXiv abs 页面并核对页面正式题名：1910.02054（ZeRO）、2104.07857（ZeRO-Infinity）、2104.04473（Megatron-LM GPU clusters）、1811.06965（GPipe）和 2204.01075（Data Cards）。据此将正文中 2104.04473 和 2104.07857 的链接文字补为完整题名。此次仅核对页面题名与引用对应关系，不代表通读论文、复核其全部结论或重跑论文实验；本轮其余动态官方文档尚未重新读取。

此后通过同一代理对本章列出的 13 个官方文档入口执行 HTTP 可达性检查，均返回 200；PyTorch Distributed 文档跳转到 `docs.pytorch.org`。该结果只证明检查时链接可访问，不证明文档所有版本语义已与正文逐条比对。

## 本轮精修检查点（2026-10-23）

- 深度复读范围：10.1—10.24，分别复读训练账本与并行策略、状态分片/重计算/混合精度、效率、故障、拓扑网络、checkpoint、调度、平台状态机、血缘/观测、SLO/成本、安全和综合实验；本轮仍未完成所有版本级来源及最终章节验收。
- 正文修订：将 ZeRO/FSDP 内存说明限定为“被切分的状态分片到组内总量约一份”，避免暗示所有参数、梯度、optimizer 状态均在所有 stage 一起缩至 `1/D`。梯度累积与有限值教学代码拒绝溢出归一化值/更新量；故障分类器明确是取证路由，比例限于 `[0,1]`，状态字段与 OOM 阶段校验类型。checkpoint 教学摘要只覆盖文件名映射，原说明将其误称为内容寻址并暗示可证明完整提交；现已改成明确的教学 manifest gate，说明它不校验文件字节、不证明原子提交、不保证逐位恢复。调度 toy 校验资源使用量不超过容量、预留不超过容量，并限定整卡 GPU 用整数；dominant share 同样拒绝半卡输入。SLO/burn-rate 补充符合条件事件分母、互斥/完备坏事件口径，并纠正把请求错误比例直接换算成坏状态天数的说法。
- 官方语义抽查：读取 PyTorch 2.14 Distributed Checkpoint 页面中的 state-dict API 说明，核对 model/optimizer state 的获取与加载、分片状态约束；读取 Kubernetes device plugin 文档中 extended resource 注册、`ListAndWatch` 和 `Allocate` 职责；读取 Google SRE Workbook 的 SLI/error budget/burn-rate 段落，确认 burn rate 是相对 SLO 的预算消耗速率，并需按 SLI 的 good/total 或 bad/total 口径计算。来源分别为 `https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html`、`https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/`、`https://sre.google/workbook/alerting-on-slos/`。这些是当前公开页面的局部语义核验，不代表所有 PyTorch 版本或完整 Kubernetes/SRE 实现均被审计。
- 代码与格式：从本章直接提取 11 个 Python 围栏并逐块编译执行，11/11 通过；调度代码新增超容量使用、超容量预留、半卡 GPU、GPU 计数、故障 ratio/flag/phase、梯度/更新溢出边界；输出保留 `manifest_digest=314b8834b309`、`dominant_share=0.25`、`burn_rate≈2.0`。直接结构检查发现 118 个 fenced block 标记成对（16 text、32 math、11 python），math 花括号平衡，无禁用 `\\operatorname`、旧式 `$$` 或相对链接；`git diff --check` 通过。未运行外部公式渲染器。
- 配套：第四册目录第十章入口有效；PAPERS/练习/题库/术语已有相关主题，本轮未发现必须追加的读者入口；未新增重复条目。
- 外链收口：章节链接目标在本地核对为 19 个唯一 URL；sampler 引用收敛到稳定的 PyTorch data 文档入口，不再依赖未能在线复核的深层锚点。本轮执行环境无法连接外网，不能把先前 HTTP 可达性记录当作本轮逐 URL 可达性结果，也未完成所有文档的版本语义复核。
- 未完成：虽已逐节复读 10.1—10.24，仍需做全章最终口径一致性与章末/配套收口、所有官方文档的版本级语义复核及逐链接目标再检查。没有运行真实 GPU、NCCL、多节点、FSDP/Megatron 训练、checkpoint 故障注入或目标集群性能测试；PDF 构建未运行。
- 状态：第四册第十章仍为进行中，`progress_v2.md` 的 `next` 保持本章，不进入第十一章。
