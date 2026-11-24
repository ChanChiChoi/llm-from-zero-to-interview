# 第二十一册第七十章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行。旧稿 341 行、34 个 H2，除逐 token latent 表示外，还混入泛化的压缩器/解码器方案、玩具文档案例、动态压缩保证、通用错误预算、来源与隐私规则、fallback、安全、snapshot/rollback 和端到端 serving 清单。多处内容与第 63 章 Gated MLA、第 69 章混合架构中的逐层表示、第 71 章递归状态生命周期，以及第二十章/第二十四册的 kernel、分页和服务职责交叠；若干可靠性与性能保证也没有对应证据。

重写保留“逐 token 的低维 latent 怎样承载并供 attention 读取历史信息”这一中心思想，收束到表示形成、投影吸收、位置分支、压缩轴区分、任务保真、cache payload/读取成本与可比实验。将通用 latent encoder/decoder、安全与来源协议、事务和 fallback 流程移出本章。

## 结构与范围决定

- 章节现行主线为：显式 K/V 到逐 token latent；query/key 与 value/output 投影的吸收；RoPE 位置分支及缓存账本；latent 与 head sharing、量化、token selection/eviction、递归 state 的差别；用任务保真评估容量；将逻辑 payload、读取代价和任务质量共同比较；按条件解读 DeepSeek-V2、KV-Latent 与 SALS 结果。
- 将“信息瓶颈”限定为有限表示容量的直观类比，不把经典互信息目标写成 MLA 的通用训练目标。明确每个位置仍有 latent 时缓存仍按上下文长度线性增长；递归状态、snapshot、迁移和事务留给第 71 章。
- DeepSeek-V2 的 93.3% cache 降幅和最高 5.76× generation throughput 绑定其与 DeepSeek 67B 的整模型比较；KV-Latent 质量与 NIAH 数据绑定论文的模型、继续训练及任务设置；SALS 的结果绑定低秩 key、token selection、key reconstruction 和 value quantization 组合及其测量条件，不归因于单一压缩轴。

## 研究、实施与验收

研究问题、原文位置、教学来源、证据边界及未解决项见[Latent Cache 表示压缩与任务保真研究记录](../../research/latent-cache-representation-and-fidelity-evidence-2026-11-24.md)。

- 重写正文并新增完整章节大纲；同步第二十一册 reader TOC 与设计 TOC、练习 31CC、面试题 31CD、`PAPERS.md` 第 67 项和 `KNOWLEDGE_GRAPH.md`。术语表已有 Latent KV Cache / Latent KV 与位置缓存相关条目，本轮未重复添加。
- 正文、章节大纲及两份目录的 26 个编号标题（6 个 H2、20 个 H3）名称、层级和顺序一致；标题核对覆盖四份文件。正文的 13 个 Markdown 链接均无失效本地目标，新增配套编号各自在对应索引中唯一。
- 数学源码检查：6 个独立 `math` 围栏闭合，行内 `$` 定界符成对，无单行未配对定界符。普通 KV 教学账本为 $2\times32\times128\times2=16{,}384$ bytes/token/layer；latent 加位置项为 $(512+64)\times2=1{,}152$ bytes/token/layer，比例为 $7.03125\%$，约低 14.22 倍。此例仅为逻辑 payload 教学设定，不含运行时额外开销。
- 公式源码与含义已静态复核，`git diff --check` 通过。GitHub MathJax 页面未实际查看，公式渲染验收仍待目标平台门禁；目标读者试读、外链终审及第二十一册连续复读也仍待全册验收。
- 未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

第七十章正文精修完成。第二十一册第 1—70 章已完成首轮正文修订，第 71—95 章共 25 章待精修；下一章为第七十一章《Recursive State Cache：递归状态缓存》。全套书精修 goal 继续 active。
