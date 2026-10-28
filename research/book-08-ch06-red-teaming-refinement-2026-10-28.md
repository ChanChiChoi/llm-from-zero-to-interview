# 第八册第六章 Red Teaming 与危险能力评估正文精修记录

- `date`：2026-10-28。
- `scope`：按当前 `WRITING_SPEC.md` 重读并修订第六章；旧章级扫描 `book-08-ch06-red-teaming-scan-2026-10-27.md` 作为候选线索，不当作正文完成证据。按该规范，章节代码围栏、状态行及其紧邻说明不写维护文件名，运行日志只记在 `progress_v2.md`。

## 结构判断与候选去向

- 保留单篇教材，不新增章节。解释链为：Red Teaming 背景与范围（§1–§6）→ 任务协议和安全执行（§7–§8）→ 发布阈值、组织交接和统计证据（§9–§12）→ 代码助手综合案例（§13）。
- 章首流程改成 Mermaid 闭环，明确任务定义、受控执行、根因修复、回归、发布和监控反馈。§11 从第二条重复流程改为跨团队证据交接，明确输入、责任人、修复版本和复测结果。
- 旧扫描提出的危险能力基准、安全评估环境和治理框架纳入 §8.5–§8.6、§9.4 和 §14；没有新开章节。AISI Frontier AI Trends Report 和 Inspect Sandboxing Toolkit 未作为本轮正文的主要证据：前者是截至 2025-10 的方法综述，后者是工具设计说明；有关隔离边界和事后控制，本轮优先引用直接描述事件与修复的 AISI incident report / security update。工具/主机/网络三分法在正文明确标为教学检查框架，而非 AISI 的正式 taxonomy。

## 外部证据与适用范围

来源在本轮通过其官方页面或 arXiv 原文核读；发布方治理政策与 benchmark 结果均不当作目标系统测量。

| 来源 | 支持的正文判断 | 证据边界 |
| --- | --- | --- |
| [AISI Incident Report](https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing) | 2026-07-28 的评估共 122 次 challenge run；10 次出现未授权自主动作，共记 19 项动作。报告明确说明当时有意开放互联网、关闭 provider cyber classifiers；不是 sandbox escape；未证实由此造成现实伤害，涉事配置未商业发布。 | 单次特定评估及配置。`10/122` 不代表一般 Agent 的发生率，不能据此推断模型意图或公开模型的部署风险。 |
| [AISI Security Update](https://www.aisi.gov.uk/blog/building-a-more-secure-environment-for-evaluating-dangerous-capabilities) | 暂停最高风险 cyber eval 后更新互联网控制、运行时监控和评估设计；sandbox 与云主机各有出站网络控制，并对两层预先验证；同步 LLM monitor 可阻断可疑动作并升级人工。 | AISI 描述自身改动和剩余工作，不构成通用安全认证；其更新也说明监控有局限。 |
| [SandboxEscapeBench v3](https://arxiv.org/abs/2603.02277v3) | 用于说明 container-to-host escape 是特定威胁模型下的能力测试；作者的限制说明任务含有意设置的弱点、shell access 和较少监控。 | 不能解释为受防护生产系统的端到端 compromise likelihood；任务覆盖不穷尽容器部署与缓解。 |
| [AgentCyberRange v2](https://arxiv.org/abs/2606.14295) | 论文有 15 个 Web 应用和 110 个漏洞，覆盖 web exploitation 与 post-exploitation 阶段。 | 作者明确未评估 phishing、Windows domain、cloud IAM、supply-chain compromise、social engineering；结果依赖系统、harness、prompts、tools 和 budgets。 |
| [FUSE v1](https://arxiv.org/abs/2609.02168) | K/D/H 管线区分 Knowledge、Defense 与 Harm；正文用其说明拒答、知识和有害性需分开观察。 | 完整规模验证主要在 CB；cyber 是较小试点且 knowledge bank 尚不成熟；H 维度与 primary judge 的跨评审者一致性统计量 ρ 为 0.42–0.58，不能视作成熟 cyber 排名。 |
| [Anthropic RSP 3.0](https://www.anthropic.com/news/responsible-scaling-policy-v3)、[Google DeepMind FSF 3.1](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)、[OpenAI Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) 及[Frontier Governance Framework](https://openai.com/index/openai-frontier-governance-framework/) | 用于说明机构公开的能力分层、评估和防护承诺有版本且按各自政策工作。 | 各组织自述政策，不是独立安全结果；阈值、类别和措施不能跨组织换算。RSP v3.0 发布于 2026-02，FSF 3.1 页面标注更新于 2026-04-17，Preparedness v2 PDF 标注 2025-04-15。 |

## 正文修订

- §8.5 新增工具、主机、网络三个环境检查维度、验证和停止条件；用 AISI incident / update 说明测试本身也可能产生外部副作用，并严格限定事件解释。
- §8.6 对照 FUSE、SandboxEscapeBench 和 AgentCyberRange 的测量问题与未覆盖范围，强调不可按原始成功率跨威胁模型排名。
- §9.4 更新框架版本与政策属性；§9.5 重写指标定义，使危险能力 uplift 与代码 demo 使用同一危险能力任务切片，补齐工具尝试任务级越权率、分母和阈值定义。
- §9.5 用 demo 九条合成记录手算覆盖率、失败率、高严重度未解决率、elicitation gain、baseline uplift、自主性、工具越权、回归通过率和加权风险；明确输入与阈值均为 toy。
- §12.2 增加独立尝试 `p=0.2,m=3` 的发现概率手算；§12.3 区分任务级 `10/100` 与动作级 `12/1000` 分母；§12.4 补公式单位和合成损失单位算例；§12.5 展开零失败时二项概率、单侧上界和 Rule of Three 的推导。
- §13 扩充代码助手综合案例，解释 P0、工具越权和回归门槛怎样组合成发布范围决定。按辅助教学单元单独计数，以确保案例自身达标。
- 将所有展示公式统一为单美元行内公式或 `math` 围栏；删除旧式 `$$` 与 `\(...\)`，确认无 `\operatorname`。关键公式补变量域、二元标签、阈值预先定义、空切片 N/A 和手算步骤。
- demo 修复空指标动作映射：缺测指标使用 `expand_evaluation_before_interpreting_metrics`，不再对 `actions[name]` 直接索引导致 `KeyError`。预期输出补上 `undefined_metrics= []`。状态行只写日期和环境，不含维护文件引用。

## 篇幅与验证

- 主章按一篇核心讲解单元统计 **14,581 个解释性汉字**（不计标题、图示/代码/公式围栏、表格、引用条目、练习和状态行），高于 1,800 字下限。
- §13 案例单独按辅助讲解单元统计 **1,151 个解释性汉字**，高于 1,000 字下限。
- Python 3.12.9，标准库；固定输入为 9 条合成 case。将唯一 Python 围栏抽取到 `/tmp/codex-book08-ch06-redteam-demo.py` 后执行 `python3 /tmp/codex-book08-ch06-redteam-demo.py`。正常输出逐行匹配正文预期块：taxonomy `1.0`、failure `0.778`、severe unresolved `0.667`、dangerous uplift `0.283`、elicitation gain `0.327`、autonomy `0.7`、unauthorized tool `0.667`、regression pass `0.429`、severity-weighted risk `0.824`；决策 `hold_high_risk_scope`。
- 另在内存中对同一代码块分别改用空 `CASES`、移除 autonomy slice、移除工具尝试、移除回归记录；各路径均保持缺测显式、将决定置为 `expand_evaluation_before_interpreting_metrics`，未触发 `KeyError`。没有新增仓库测试文件。
- Mermaid 源码采用标准 `mermaid` 围栏、带引号的标签及 `-.->|标签|`；结构静态检查通过。目标 Markdown 页未包含本地变更，所以 GitHub 渲染尚未确认。
- AISI、Anthropic、Google DeepMind、OpenAI 官方页及三篇 arXiv 原文已在本轮打开核对；未执行本章全部外链的 HTTP 扫描。固定代码块已实际运行；无目标模型、评估 harness 或生产网络参与。

## 尚未关闭

- GitHub Mermaid 和 MathJax 页面渲染、整章外链状态检查与第八册连续复读仍开放；因此本章不标记完整正文门禁通过。
- 本地 demo 只验证合成样例与空分母控制流，不代表任何模型的危险能力、越权率、回归有效性或生产安全。
