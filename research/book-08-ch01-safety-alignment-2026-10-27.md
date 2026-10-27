# 第八册第一章外部扫描与精修记录（2026-10-27）

## 研究问题

检查总览章的安全框架和规范来源是否仍指向有效版本，区分行为规范、能力风险框架、
治理框架、工程控制和实测证据能支持的结论；同时根据第八册稳定后的章节地图消除
章内重复并补全后续交接。

## 主要来源及适用边界

- [OpenAI Model Spec](https://model-spec.openai.com/) 是 OpenAI 对模型期望行为的公开说明。其[版本记录](https://github.com/openai/model_spec/blob/main/CHANGELOG.md)截至 2026-10-27 列出 v2026.08.18。它支持分析行为边界与指令优先级，不是某个模型实测遵循率。
- [OpenAI Preparedness Framework Version 2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) 的官方更新说明日期为 2025-04-15；OpenAI 的 [Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/) 发布于 2026-05-28，说明其安全与安保实践如何映射到相关法律要求。两者都是发布组织自己的风险治理资料，不能当作独立认证或跨组织通用阈值。
- [NIST AI RMF 官方入口](https://www.nist.gov/itl/ai-risk-management-framework) 说明 AI RMF 1.0 正在修订，且 2026-04-07 发布了 Trustworthy AI in Critical Infrastructure Profile 概念说明。正文因此明确引用 AI RMF 1.0，并把新 Profile 写成开发中的概念资料，而非已定稿标准。
- [OWASP Top 10 for LLM Applications 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) 是社区驱动的应用安全风险分类，可帮助建立覆盖面，不能估计特定模型风险率。OWASP [Agent Control Standard](https://genai.owasp.org/resource/agent-control-standard-acs/) 页面日期为 2026-09-01，说明 ACS 现由 OWASP GenAI Security Project 托管，并以 middleware hooks 和可移植声明式策略描述运行时控制接口；该接口说明不证明具体 Agent 已实现或执行正确。
- [METR Frontier Risk Report（2026 年 2—3 月）](https://metr.org/blog/2026-05-19-frontier-risk-report/) 是与 Anthropic、Google、Meta、OpenAI 合作的组织级外部评估试点，评估窗口为 2026-02-16 至 2026-03-16，并结合参与组织提供的非公开信息和模型访问。结论绑定试点设计、窗口与参与组织，不推广成公开模型的一般风险率。
- [Anthropic Responsible Scaling Policy 页面](https://www.anthropic.com/responsible-scaling-policy) 提供当前政策、版本历史与风险报告；引用具体门槛时须定位版本，政策文本及自评不等于独立认证。[Google DeepMind Frontier Safety Framework 页面](https://deepmind.google/frontier-safety/) 当前列出 Version 3.1（2026-04-17）；其阈值与缓解流程是组织自己的框架。
- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026) 用于区分 misuse、malfunctions、systemic risks 等影响维度；它是跨领域研究综述，不是部署评估。

## 结构与正文决定

- 不新增第一章内容主题；保持本章作为全册地图和系统安全判断入口。
- 第5节解释模型能力为什么有双重用途；第10.6节专讲如何结合任务、授权、工具和后果选择运行时路由，增加代码助手进入生产发布动作的例子。
- 第6节用风险 taxonomy 标注潜在问题和证据位置；第10.1节把分类转换为对照样本、期望动作和判定记录，避免重复列举风险标签。
- 第11节保留上线系统的运营失效；第13节改为评审中常见的推断错误，解释拒答率、训练方法、注入检测、规范遵循和 benchmark 结果不能单独证明哪些结论。
- 修正开篇对“适用边界见第9节”的错引，补全第12章综合复习和第13—17章安全工程专题交接，更新 ACS 的当前组织归属，去除第7.1节重复句。
- 更新官方来源版本与链接：Model Spec 版本记录、Preparedness Framework v2、NIST AI RMF 修订状态、OWASP 2026 资料、METR 原始地址、Anthropic RSP 当前页面、DeepMind FSF 3.1，并补入 OpenAI Frontier Governance Framework。

## 验收状态与遗留项

- 第一章正文已按新版册级结构重读并做针对性重组；全章仍是一篇核心讲解单元。按排除标题、表格、引用清单、围栏内容、状态注释和练习正文后的机械汉字计数为 9,896，超过 1,800 字下限；这不替代对解释链的人工判断。
- §10.5 的标准库代码和其预期输出没有变化，可复用此前记录的 Python 3.12.9 运行验证；本轮未重跑代码。
- `git diff --check` 通过；简介、目录、第一章、第17章和本研究记录的静态相对链接检查无断链，Markdown 代码/数学围栏无未闭合项。已复读本轮正文差异。
- GitHub 页面 Mermaid 与修改后的 §10.6 数学公式渲染仍未确认，因此当前状态不是第一章验收通过。web 页面读取返回缓存未命中；随后通过只读 GET 获取了远端 `main` 页面 HTML。HTML 将旧版章首图放入 GitHub 的 `data-type="mermaid"` 客户端渲染容器，但没有提供最终 SVG；远端内容也不含本地这轮修改，故不能据此声称本地改动已通过 GitHub 页面渲染。
- 若 GitHub 页面渲染发现语法问题，修复后再复核；不得把未做的外部渲染或真实系统评估记为完成。
