# 第10章《checkpoint 评估与实验管理》本轮精修交接（2026-11-24）

## 精修范围

按用户要求忽略既有 `progress/` 状态记录，依当前写作规范和第10章章节契约处理本章；不据旧状态跳章。保留“哪个 checkpoint 值得继续”的中心，按章节文件顺序完成本章后进入第11章。

## 正文调整

- 将恢复 loss 探针改为对照同一 checkpoint 派生的未中断参考分支与保存/恢复分支；固定 batch、mask、归约和随机状态。说明相邻日志常来自不同 batch，不能直接当作恢复偏差；学习率按恢复步对应的调度器预期值比较。同步重写偏差定义和数值例子，并要求单独核对数据游标与下一批样本标识。
- 校正 PyTorch 2.15 DCP 异步保存描述：默认 CPU staging 与后台写入的边界、GPU 到主机内存的 staging 可能阻塞、启用 `AsyncStager` 后的 staging/upload 两个完成状态、更新模型前等待 staging、发布版本前等待写入，以及该 API 的实验性和跨版本兼容限制。
- 明确 `N_save=ceil(S/f)` 的条件是训练结束时保存最终 checkpoint；若不额外保存，严格周期保存数是 `floor(S/f)`。为最大回退 token 公式补充按已完成、已发布周期 checkpoint 计算的假设。
- 统一 PPL 滑动窗口示例的 token 下标与计分口径；将数学数量、单位、比例及成对分数表达按模块 04 规范呈现。
- 说明选择例中 code loss 的两个参照：`ckpt_2000` 的 2.74 是相邻版本值，选择门槛以 `ckpt_1000` 的 2.90 为基线，升至 2.98 后超出 0.05 界限。
- 将 lm-eval 研究引用固定为论文 v3（2026-05-31），并区分题目抽样 bootstrap、重复模型运行与 prompt/评分器选择带来的波动。

## 研究证据

本章事实、来源版本及边界记在 [checkpoint-evaluation-experiment-management-evidence-2026-11-24.md](../../research/checkpoint-evaluation-experiment-management-evidence-2026-11-24.md)。本轮核对 PyTorch 2.15 DCP/API 与异步保存教程、PyTorch `torch.load` 和可复现性文档、Hugging Face PPL/Trainer 文档、HELM、lm-eval v3、Dror et al.、Oren et al. 与 ConStat 一手来源。

## 静态验收与遗留

- 第10章契约标题与正文的 32 个章节标题顺序一致；`load_state_dict` 仅有反引号代码字体差异，归一化后匹配。
- 24 个 Markdown 链接，缺失本地目标 0；代码围栏未闭合数 0；行内数学定界异常数 0；检查 330 个公式表达式，未发现大括号或 `\\left`/`\\right` 配对问题；尾随空白 0。
- `git diff --check` 对本章正文与研究证据文件通过。
- CPU demo 保持“本轮未执行”；未运行代码示例、测试、训练或 PDF 构建。
- 公式源码和数学含义已静态检查；目标平台逐式渲染尚未目视确认，因此不能将公式渲染门禁标记为完成。

## 后续顺序

下一步进入第11章《后训练与对齐训练》，不从既有 `progress/` 状态推导跳章。
