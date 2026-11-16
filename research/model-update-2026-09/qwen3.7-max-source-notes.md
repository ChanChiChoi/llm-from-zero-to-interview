# Qwen3.7-Max：Agent 环境、长程 kernel 优化与评测来源记录

资料核对日期：2026-11-16。本文记录 Qwen 官方对 Qwen3.7-Max 的公开描述、对应条件和未验证边界；不把发布方实验写成独立复现或生产保证。

## 官方发布材料

- **来源与版本：** Qwen Team，[Qwen3.7: The Agent Frontier](https://qwen.ai/blog?id=qwen3.7)，页面署名日期 2026-05-20；访问日期 2026-11-16。文章为动态页面；本次通过官方页面的可检索正文核对摘要、评测表及脚注、Cross-Harness Generalization、Self-Evolving in the Wild、Reward Hacking Monitoring 与 API Usage 段落。
- **证据等级：** `source-backed`（模型发布方公开说明和作者报告）。该来源支持“Qwen 发布方在所述模型、harness、任务和测量条件下报告了什么”，不等于第三方复现。

## 长时程 Extend Attention kernel 案例

- **任务与初始条件：** 优化 SGLang 的变长 Extend Attention。任务使用最长 32K 项的前缀 KV cache 和 MTP 生成 token；参考实现是 SGLang 官方 Triton kernel。运行设备为平头哥真武 M890 PPU。发布方称该设备此前未出现在训练中，且模型初始只有任务描述、参考实现和评估脚本，没有对应硬件文档、profile 数据或示例 kernel。
- **发布方报告：** 约 35 小时自主执行，432 次 kernel evaluation、1,158 次工具调用；跨多个 workload 对 Triton reference 报告 `10.0x` 几何平均加速。公开轨迹中的阶段包括 Split-KV 加 partial reduction、移除 host/device 同步与分配开销、按 workload 调整 split 策略，以及在 MTP `γ=4` 时让一个 block 同时处理四个 query 并共享 K/V 读取。
- **轨迹数字：** 官方文章报告若干阶段相对 reference 从 `0.33x` 到 `2.58x`、`5.37x`、`8.50x`，最终至 `10.0x`；各阶段对应具体 kernel revision、设备和 workload。它们是该案例轨迹，不应解释为任意 workload 的加速曲线。
- **KernelBench L3 是另一组测量：** 同文另报 Qwen3.7-Max `1.98x / 96%`：统计基于 50 道题、H100 80GB，在隔离 Docker 中运行；`1.98x` 是相对 PyTorch eager reference 的逐题 speedup 中位数，`96%` 是快于 `torch.compile` 的题目比例。作者描述的限制还包括最多 500 次工具调用及连续 100 轮无改进后提前停止。任务、硬件、baseline 和统计量均不同于 M890 案例，不能横向等同。

## 发布方评测与训练环境描述

- **benchmark 条件：** Qwen 官方表将 Terminal-Bench 2.0 结果绑定 Harbor/Terminus-2 harness、5 小时 timeout、12 CPU、24 GB RAM、最高 80K 输出和 256K context，报告平均 5 次运行；SWE-bench 系列则注明为内部 bash/file-edit scaffold、200K context。不同表格结果不能脱离脚注或移植成裸模型能力。
- **环境扩展与 Cross-Harness Generalization：** 发布方描述其 RL rollout 将 Task、Harness、Verifier 作为可组合组件，并称以不同 harness/version 和 verifier 组合任务。文章同时称评测使用训练中未见的环境，并据此主张跨环境泛化；完整技术报告尚未在该页面披露。此处是作者的训练和评测描述，不公开完整数据、训练 recipe 或独立验证细节。
- **Reward Hacking Monitoring：** Qwen 报告在超过 80 小时的 SWE 强化学习实验中自主检索并回放训练轨迹，执行超过 10,000 次调用；系统新增 13 条启发式规则并报告标记 1,618 个 hacking cases。页面将其作为奖励投机自监控和规则演化的发布方案例；未提供足以独立重建误报率、漏报率或规则泛化性的完整协议，不能当作通用检测器准确率。

## 限制与未确认项

上述性能、训练和轨迹数据均来自 Qwen 官方博客。公开材料未由此确认模型参数或架构、完整训练配方、未披露的奖励与数据、M890 之外的广泛硬件表现、一般生产成功率或独立复现。对比数字只保留各自的任务集、harness、baseline、硬件和统计口径；缺少精确 DataCurve 配置行时，不迁移其他模型或其他 harness 的分数。
