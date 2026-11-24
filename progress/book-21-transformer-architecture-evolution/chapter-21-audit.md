# 第二十一章《MoE 路线：Switch、GShard、Mixtral 与 DeepSeek-MoE》精修审计

审阅日期：2026-11-20。章节职责复核、MoE 路由与专家结构研究、结构大纲、正文主题重写及配套同步已完成。块公式的 GitHub MathJax 实际渲染、目标读者试读、外链逐条终审和第二十一册连续复读仍待后续门禁；按用户要求不更新 PDF 或 plan.md。全套书精修 goal 继续 active。

## 读者契约与章节边界

- **前置：** 本册第五章的 Transformer FFN/SwiGLU 与 block 数据流；第五册第四章的 MoE 总参数、active 计算、路由训练与 capacity 口径。
- **离章结果：** 能追踪 token 从 router 选择、dispatch、expert 计算到 combine 的完整路径；能区分 GShard top-2 变体、Switch top-1、Mixtral 每层 8 expert/top-2 与 DeepSeekMoE 的细粒度切分、shared expert；能限制专家专业化与模型收益的结论范围。
- **范围：** 本章负责 Transformer MoE 的 FFN 路由和代表性 expert 结构。第五册第四章负责辅助 loss、z-loss 与容量训练公式；第二十二章负责 Dense/MoE 的训练、服务、通信、SLO 和成本；MLA 与 KV 表示回链本册第四、十一章。

## 旧稿诊断与结构决定

旧稿有 21 个 H2，混合 MoE 结构、训练 recipe、服务成本、生产 gate、checkpoint 回滚和多份资源目录。容量公式、负载指标、p99、公平比较、服务账本与第二十二章及第五册第四章重复；Python router demo 不服务本章的架构读者结果；DeepSeek-V2 的 MLA 与 MoE 路由也被放在同一条解释线上。另有把 selected top-k weights 一律重新归一化、把 capacity overflow 说成“等待下一处理”，以及将 expert specialization 写成稳定学科语义的表述需要收窄。

章节结构研究确认将原有混合主题收束为六节：MoE FFN 的条件路径、token 完整路由、top-k/容量规则、GShard/Switch/Mixtral、DeepSeekMoE 细粒度与 shared 路径、专家分工证据和跨章边界。保留章节名、文件路径和本册目录入口。完整现行结构见[第二十一章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)。

## 研究核验与正文决定

来源版本、实际阅读位置、教学组织材料及证据范围见[MoE 路由与专家结构研究记录](../../research/moe-routing-and-specialist-architecture-2026-11-20.md)。主要决定如下：

- 把 MoE 的路由对象限定为每层 token hidden state，不写成请求级领域模型选择。
- 用一个完整路由式和不含真实分数的教学轨迹连接 expert selection、按 expert 打包、token identity 恢复与输出组合；组合权重是否归一化由具体模型决定。
- 说明 Switch 论文先对全部 expert logits 做 softmax 再取 top-1 gate；GShard 是含特定二路 dispatch/容量处理的 top-2 变体，不把路由数量误作完整算法定义。
- 对 GShard、Switch、Mixtral 分别按原文说明 top-2 变体、top-1 和每层 8 FFN experts/top-2；三者不是统一控制实验，论文结果不作跨模型性能排序。
- 对 DeepSeekMoE 限定为论文提出的细粒度切分和始终激活的 shared experts；其消融结果限于论文模型与实验，不声称架构会自动产生可解释的通用 expert。
- 将 DeepSeek-V2 中 MLA 的 KV 表示变化与 MoE FFN 路由明确分开；不以联合模型结果单独归因任何一项结构。
- expert-choice 仅作为与 token-choice top-k 不同的路由方向对照，不移入一般 Dense/MoE 成本比较。

## 删除与同步

- 删除旧稿中重复的 MoE load-balance 公式与损失配方、Python router 审计、通信量估算、训练/推理资源账本、p99 诊断、通用公平实验清单、checkpoint 迁移/回滚流程、生产发布 gate、资源目录及重复面试答案。
- 这些内容已有主讲位置时回链第五册第四章或本册第二十二章；未在其他正文重新安置的服务实现细节不在本章补造。
- 新建本章大纲与研究记录；本册目录入口与章名保持不变。
- 在 PAPERS.md 的 DeepSeekMoE 条目添加本章研究记录链接；在 GLOSSARY_EN_ZH.md 增加 Fine-grained Expert Segmentation、Shared Expert 与 Expert Specialization；在 KNOWLEDGE_GRAPH.md 建立从路由选择到 dispatch/combine、细粒度与共享路径的依赖链。
- 在 EXERCISES.md 增加 31Z 的 token 路由轨迹练习；在 INTERVIEW_BANK.md 增加 31W 的四类路线与证据边界题。

## 静态检查与遗留门禁

- 正文 H2 为连续的 21.1—21.6；使用一处独立 math 块公式，无代码块或 Mermaid。公式变量与 combine-weight 口径已检查；MathJax 视觉渲染待验。
- 正文含一个教学路由表；示例明确不代表真实 router 输出，忽略容量溢出与设备放置。
- 静态检查通过：正文 H2 `21.1—21.6` 连续；1 个 `math` 围栏成对；54 个行内 `$` 定界符成对；正文表格列数一致；正文、大纲、研究记录、章节审计和册级审计中的本地链接均可解析，交接新增段与 `PAPERS.md` 本章来源链接也有效。
- 配套编号检查通过：`EXERCISES.md` 的 31Z、`INTERVIEW_BANK.md` 的 31W 各自唯一；三个新增术语各出现一次。正文、大纲、研究记录和章节审计的新内容无尾随空格、均有结尾换行且围栏配对；`git diff --check` 通过。
- 未运行代码、测试、模型、框架 API、GPU kernel、benchmark 或 PDF。
- GitHub MathJax 实际渲染、目标读者试读、外链逐条终审及第二十一册连续复读仍待后续门禁。

下一入口为[第二十二章](../../book-21-transformer-architecture-evolution/chapters/22-Dense与MoE训练服务trade-off.md)。全套书精修 goal 继续 active。
