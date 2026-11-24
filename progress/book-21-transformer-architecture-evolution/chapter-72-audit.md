# 第二十一册第七十二章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行。旧稿约 276 行、31 个二级标题，把尚未证实的局部/全局位置处理设想、长上下文评估清单和通用 serving/runtime 事项混在一起，也没有明确辨认论文中的 P-RoPE 是 Periodic RoPE。重写后以已发布为 arXiv v2 的 P-RoPE/HP-RoPE 方法为对象，避免从缩写推测机制。

正文先区分位置旋转与 attention 可见范围，再推导周期索引、跨界 wrap error 和 HP-RoPE 的 $R_W=I$ 条件；随后交代 MiniWin 的局部 SWA/全局 NoPE 路径、具体八层设置和 v2 实验。保留“周期位置索引可循环”这一机制主张，同时排除“模型具有无限资源或任意长文理解能力”的外推。

## 结构与分工

现行结构为 5 个二级标题、17 个三级标题。第 18 章负责标准 RoPE 与长上下文扩展，第 64 章负责 NoPE 与隐式顺序，第 65 章负责局部/全局/递归路径的一般比较，第 67 章负责 Llama 4 iRoPE；本章只讲周期 RoPE 的边界机制、HP-RoPE 和 MiniWin 的特定实验，不重复这些章节，也不展开 serving/cache 运维。

完整结构见[第七十二章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/72-P-RoPE周期性RoPE与局部全局注意力.md)。

## 研究依据与边界

方法和数字依据 Huo 与 Li 的 [Periodic and Harmonic Periodic RoPE 论文 arXiv v2](https://arxiv.org/html/2605.27980v2)；有关 NoPE、局部/全局结构与教学解释的来源及其支持边界，见[周期性 RoPE 与局部/全局位置路径研究记录](../../research/periodic-rope-local-global-position-evidence-2026-11-24.md)。论文结果绑定八层 MiniWin、指定训练长度和至 16k 的评测；边界移动诊断支持旋转一致性的机制判断，不替代更广泛的任务评测或独立复现。

## 正文、配套与验收

- 已重写正文并新增完整大纲；同步第二十一册读者目录和设计目录。正文、大纲及两份目录的 22 个编号标题（5 个 H2、17 个 H3）名称、层级和顺序一致。
- 已同步练习 31CE、修订原有 MiniWin schedule 练习、面试题 31CF 和原有 P-RoPE 面试回答、`PAPERS.md` 第 69 项及术语说明、`KNOWLEDGE_GRAPH.md`；术语表新增 HP-RoPE 并区分 partial/fractional RoPE。
- 逐式静态检查公式源码及数学含义：频率基数、head dimension、缩放因子说明、周期边界差 $d-W$、$R_W=I$ 条件、谐波频率和实验数字均与正文解释对应。公式围栏闭合；正文、大纲与目录未发现标题错位，新增配套编号在各自索引中唯一。`git diff --check` 通过。
- GitHub MathJax 页面未实际查看，公式视觉渲染门禁仍待验；外链逐条终审、目标读者试读和第二十一册连续复读仍留待全册门禁。
- 未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

第二十一册第 1—72 章已完成首轮正文修订，第 73—95 章共 23 章待精修。下一入口为第七十三章《Gated Attention：门控显式注意力》。全套书精修 goal 继续 active。
