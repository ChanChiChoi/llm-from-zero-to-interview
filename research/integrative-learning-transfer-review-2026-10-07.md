# 第一册第五份正文：学习迁移资料核验

核验日期：2026-10-07。范围：`book-01-core-30/chapters/05-综合复习与技术表达.md` 中检索练习、间隔练习和复习计划的证据表述。数值型诊断案例、权重和学习调度器均明确标为教学构造，不作为实测结果或学习科学定律。

## 一手论文与支持范围

| 论文 | 核验材料 | 可支持的正文表述 | 证据边界 |
| --- | --- | --- | --- |
| Roediger & Karpicke, *Test-Enhanced Learning* (2006), [DOI](https://doi.org/10.1111/j.1467-9280.2006.01693.x) | [Crossref 元数据与摘要](https://api.crossref.org/works/10.1111%2Fj.1467-9280.2006.01693.x)；摘要记录两项 prose-passage 实验、5 分钟/2 天/1 周最终测试间隔。DOI 页面在本次访问返回 HTTP 200，但呈现人机识别页。 | 该实验中，5 分钟最终测试时重复学习成绩较好；延迟 2 天或 1 周时，先前测试带来更好的保持。 | 不能把这组 prose-passage 实验直接推广为所有技术主题的最佳复习策略或间隔参数。 |
| Cepeda et al., *Distributed practice in verbal recall tasks: A review and quantitative synthesis* (2006), [DOI](https://doi.org/10.1037/0033-2909.132.3.354) | [Crossref 元数据](https://api.crossref.org/works/10.1037%2F0033-2909.132.3.354)与 [OpenAlex 索引摘要](https://api.openalex.org/works/https://doi.org/10.1037/0033-2909.132.3.354)。摘要称覆盖 184 篇文章、317 项实验、839 项评估；APA DOI 页面返回 HTTP 403，Crossref 未提供摘要。 | 在该 verbal-recall 研究综合中，练习间隔（ISI）与最终保持间隔共同影响最终测试保持；保持间隔增长时，达到最高保持效果的 ISI 也增长。 | 不推出固定复习日历，也不将 verbal-recall 综合自动迁移为技术课程或所有学习者的最优参数。 |

正文依照上述范围改写：Roediger/Karpicke 部分给出论文任务和测试间隔；Cepeda 部分明确限定为 verbal recall，并只描述摘要直接支持的 ISI—保持间隔关系。语义学习、代码技能和面试表现需要各自的延迟测量与变体任务评估。

## 访问记录

- Crossref API 的两条 DOI 元数据请求均成功；Roediger 记录提供摘要，Cepeda 记录没有摘要。
- Roediger DOI 重定向页返回 HTTP 200，但内容是人机验证页面；Cepeda DOI 页面返回 HTTP 403。
- OpenAlex 的 Cepeda 条目返回标题、年份和索引摘要；Semantic Scholar API 本轮返回 HTTP 429，未作为正文依据。
- 正文继续保留出版商 DOI 作为读者追溯入口；本记录不把元数据索引替代为论文全文。
