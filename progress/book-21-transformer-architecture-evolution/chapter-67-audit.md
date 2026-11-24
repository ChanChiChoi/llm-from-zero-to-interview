# 第二十一册第六十七章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行。旧稿约 281 行，以 27 个 H2 反复展开 RoPE 外推、长训稳定性、serving/cache 契约和泛化评测清单；这些主题互相重复，未围绕 Llama 4 iRoPE 的具体机制、来源口径和读者理解路径组织。现行章内结构为 6 个 H2、17 个 H3，把位置变换、mask 可见范围、query-temperature scaling、频率缩放、长上下文训练披露和运行时成本分别处理。

## 结构与内容决定

- 以 Meta 发布资料作为本章 iRoPE 的主命名口径，并说明 ICLR 2026 论文采用更宽的术语范围；不把任一来源口径说成所有模型共用的标准定义。
- 分开解释 Q/K 的 RoPE 变换、局部/全局 causal mask、NoPE 路径上的位置依赖 query scaling、Scout scaled RoPE 和长上下文训练；通过跨 chunk 教学轨迹说明全局层提供的远程信息路径，但不把结构可达性等同于检索正确率。
- 保留 Meta query scale 公式，并分别说明 256K 预训练/后训练、专门数据 mid-training、10M 输入窗口、needle 与代码 NLL 的发布披露；不从这些结果归因到单一组件。
- 把计算量写成分块 mask 生效时的条件估算，并明确局部 mask 与 KV eviction 不等价。
- 复核 Meta `main` 分支后补充调用边界：参考实现只在单次输入长度大于 1 时建立 local/global mask；长度为 1 时 mask 保持 `None`，block 会传递 `None` 给 attention。因此该源码的多 token mask 路径不能证明逐 token decode 仍限制在局部 chunk。此项为源码静态阅读结论，未本地执行，也不外推到其他 runtime。

## 研究与同步

来源、版本、阅读范围和证据边界见[iRoPE 与长上下文研究记录](../../research/irope-long-context-evidence-2026-11-24.md)。已同步正文、[第六十七章结构大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/67-iRoPE与超长上下文.md)、读者目录、设计目录、练习 31BZ、面试题 31CA、`PAPERS.md` 第 64 项、术语表和知识图谱。

## 静态验收与遗留门禁

- 正文、大纲、读者目录和设计目录共 23 个章内标题（6 个 H2、17 个 H3）的名称、层级与顺序一致。
- 本轮涉及的 18 个本地链接目标均存在；练习 31BZ、面试题 31CA 和论文索引第 64 项各自唯一。
- 两个 `math` 块级公式围栏闭合；行内数学定界符逐行成对。手工核对 query scale 与 Meta 参考代码中的取整、位置起点和乘数关系，并核对教学 softmax 算例及分块/全局 attention 复杂度口径。
- `git diff --check` 通过。GitHub MathJax 页面实际渲染未查看，故公式渲染门禁仍未关闭。目标读者试读、外链终审和第二十一册连续复读仍留在全册门禁。
- 未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

## 下一步

第二十一册第 1—67 章已完成首轮正文修订，第 68—95 章共 28 章待精修。下一章为第六十八章《MetaP 与长上下文训练稳定性》。
