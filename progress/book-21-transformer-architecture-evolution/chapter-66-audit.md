# 第六十六章精修审计

精修日期：2026-11-24  
章节：Total Parameters 与 Active Parameters  
方式：主题思想不变重写

## 旧稿诊断

旧稿用 34 个层级标题反复讲 total、active、MoE 路由、负载均衡、训练与服务成本。多个小节重复同一组定义和公式，也并列了 660B/40B、330.1B/20.1B 等不同教学算例，却没有始终保留统一的参数统计约定。Active 参数有时被写成计算量、显存或端到端成本的直接代理；后文又重复讲权重常驻、KV、通信、p99、训练状态和服务监控，使本章越过了参数口径本身，和第五册第 13 章、第二十一册第 47/55 章及后续 serving/cache 内容重叠。

## 修订决定

重写为六节：total 与 active 的统计对象；逐层 total/active 公式；单 token 路径、请求内 route union 与权重字节；Mixtral、DeepSeek-V3、Qwen3 的来源数字；参数量不能直接推出 FLOPs、显存、延迟或质量；以及审核发布数字的字段清单。新增参数计数约定 C，按互不重叠的参数集合汇总，并对 embedding、output head、router、shared expert 和 tied weights 保留来源差异。用一个明确标注的教学 MoE 计算 10B total、4B active，说明 k/E 仅为等大小 expert bank 的比例。

本章不再重讲 router load-balancing loss、训练 optimizer state、capacity/drop 的服务监控、p99 dashboard 或通用容量规划。MoE 组合和模型案例回链第 47、55 章；训练 FLOPs 回链第五册第 13 章。新正文：[第六十六章](../../book-21-transformer-architecture-evolution/chapters/66-total-parameters与active-parameters.md)，结构：[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/66-Total-Parameters与Active-Parameters.md)。

## 研究依据

本轮查阅来源及阅读位置、支持范围和边界见[MoE 参数量研究记录](../../research/moe-parameter-count-evidence-2026-11-24.md)。正文采用 Mixtral 论文的 47B/13B 与 8-expert/top-2 报告、DeepSeek-V3 v2 的 671B/37B 与 shared/routed 路由配置、Qwen3-30B-A3B 模型卡并列的 total/activated/non-embedding 字段；Stanford MS&E 319 讲义支撑 EP/kP 和近似 expert FLOPs 的教学关系；Hugging Face Transformers 技术博客只用于说明 MoE checkpoint tensor 与该 runtime 的打包布局；2026 arXiv v1 配置研究的质量结果限定在论文模型规模、active FFN 匹配和 held-out loss 设置内。

## 配套同步

已新增第六十六章详细大纲，并同步第二十一册正文目录和设计目录；新增练习 31BY、面试题 31BZ、PAPERS.md 第 63 项、KNOWLEDGE_GRAPH 参数计数链和 Total Parameters 术语。现有 Active Parameters 术语改为标明 per-token 发布口径及不等同的资源/质量指标。

## 静态验收

正文、大纲与两份目录的 6 个 H2、20 个 H3 名称、层级和顺序一致。6 个 math 围栏配对；公式以源码检查并逐式核对含义，98 个行内美元定界符为 49 对。手工核对教学算例：4 个 MoE 层、每层 8 个 0.25B expert、每 token top-2、2B 共同路径得到 10B total 和 4B active；专家库比例 25%，全模型比例 40%；BF16 原始 10B 权重载荷按十进制估算约 20GB，不作为 GPU 显存值。

章节正文、大纲、两份目录、练习和面试题范围内新增本地链接目标均存在；练习 31BY、面试题 31BZ、PAPERS.md 第 63 项编号各唯一。EXERCISES.md 和 INTERVIEW_BANK.md 中未发现误写入的字面反斜杠-n。git diff --check 通过。

## 未关闭事项

公式源码和含义已静态检查，但 GitHub MathJax 实际页面渲染未查看，不能标为渲染门禁通过。目标读者复述/试读、第二十一册整册连续复读和历史外链终审仍待全册门禁。未运行测试、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 plan.md。全套书精修 goal 继续 active，下一章为第六十七章《iRoPE 与超长上下文》。
