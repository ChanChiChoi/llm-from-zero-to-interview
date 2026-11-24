# 第三十八章《RetNet：Retention 机制与并行/递推统一》精修审计

## 范围与诊断

按全套书精修的主题重写要求，保留章名、章节文件路径和目录位置。完整复读旧稿后发现 32 个 H2 多次重复 parallel/recurrent/chunkwise 一致性、chunk 边界、固定状态容量和长程记忆边界，并把状态 snapshot/rollback、batch reorder、恢复测试、服务资源和 benchmark 验收扩写成多个独立小节；这些内容与第六册、第二十四册和本册第四十二章的职责交叠。旧稿还把一个自拟的分子/分母归一化状态式呈现为 RetNet 可能采用的归一化方式，混淆它与原论文的 retention score normalization、逐头 GroupNorm；教学 toy code 也没有兑现本章读者结果。

按 02a 新建六节结构并据此重写正文，保留章节名称、路径和目录位置。主线改为：训练/生成约束，因果衰减的并行与递归式，multi-scale block 与数值尺度，三种执行形态，状态表达边界和相邻路线，原论文效率证据边界。服务生命周期、恢复演练和通用 benchmark 流程不再作为 RetNet 机制正文。

## 研究与内容

研究问题是 RetNet 论文中的 retention、multi-scale retention 及 parallel/recurrent/chunkwise recurrent 三种计算形式，不把 Gated RetNet、视觉派生模型或后续架构并入基础定义。证据阅读位置与来源角色见[RetNet 研究记录](../../research/retnet-retention-parallel-recurrent-2026-11-23.md)。

正文根据原论文 §2.1 推导衰减矩阵、状态外积累积及 chunk 边界的输入/输出状态；根据 §2.2 分开解释固定 head decay、位置相位、Swish gate、逐头 GroupNorm 和 score normalization；根据 §2.3 呈现 RetNet 的 residual/FFN block。全章显式区分 retention score 与 softmax 概率，并说明固定状态只相对历史长度成立。

根据原论文 §3.1—3.4，效率结果只按论文模型规模、训练语料/token 量、chunk/序列长度及 A100 条件叙述。Microsoft TorchScale README 和固定 revision 修复记录只支持“具体实现曾进行 chunkwise 一致性与数值修正”这一版本边界，不支撑跨实现行为或新的性能主张。未运行代码或 benchmark。

## 配套同步

第三十八章结构大纲已新增，第二十一册目录补充其链接及 38.1—38.6 标题。PAPERS.md 加入原论文和研究记录入口；KNOWLEDGE_GRAPH.md 增加 RetNet 的因果衰减、矩阵状态、三种执行形态及与相邻路线的关系。新增练习 31AQ、面试题 31AN。术语表已有 RetNet 条目，未重复添加。

## 验收与遗留

- 正文、大纲和目录的六个小节标题及顺序一致；章节标题与路径保持不变。
- 源码复核覆盖因果衰减指数和 mask 方向、$d_k\times d_v$ 状态维度、块内/块间指数、multi-scale decay 序列、位置相位及论文的 score normalization。正文无代码、Mermaid 或图片。
- 新增题号在各自题库唯一；新增章节、目录、大纲、研究和审计链接均已检查目标。
- `git diff --check` 通过。GitHub MathJax 实际页面逐式渲染尚未验收，公式门禁仍开；不能标为公式渲染完成。
- 外链最终逐条复核、目标读者试读和第二十一册连续复读留待全册门禁。
- 未运行测试、代码、模型、API、kernel、benchmark 或 PDF 构建；不更新 PDF 或 `plan.md`。

第三十八章完成首轮修订。下一章为第三十九章《Hyena：隐式长卷积与长序列建模》。
