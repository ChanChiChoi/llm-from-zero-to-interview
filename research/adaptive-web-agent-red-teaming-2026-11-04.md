# 自适应 Web Agent 红队与执行状态评估：来源证据

核对日期：2026-11-04。本文只记录公开论文中的方法、实验事实和适用边界，不记录章节审阅决定或书籍修改状态。

## ANGLER / REEF

- **来源与版本：** Huang et al., *ANGLER: Verifier-Grounded Benchmarking with an Adaptive Attacker for Web Agents*, arXiv:2610.41272v1，2026-10-30 提交；[论文 HTML](https://arxiv.org/html/2610.41272)。
- **阅读范围：** 摘要、§§3.1—3.2 威胁模型、§§4.2—4.3 benchmark 构造与执行状态核验、§§5.1—5.3 自适应攻击流程、§§6.1—6.5 实验和讨论、伦理与隐私说明。
- **来源支持：** REEF 包含 9 个状态化网站、8,316 个可执行案例和 5 类攻击；攻击可经用户指令或网站环境内容注入，程序化 verifier 检查恶意状态变化和结束状态。ANGLER 使用此前 rollout 的 verifier 结果更新候选攻击与选择策略，并优先从 near-miss 案例生成后续候选。在 VisualWebArena scaffold 下对 8 个 VLM agent 配置、9 个网站进行匹配比较，每个 website–backbone 条件为两种攻击方各分配 100 次 rollout；作者报告自适应方法的平均 ASR 为 20.36%，非自适应基线为 15.24%，差 5.12 个百分点、相对增幅 33.6%。同一研究报告攻击条件下的平均 benign-task success rate 为 44.4%，无攻击条件为 61.9%。
- **证据边界：** 这是 2026-10-30 的 arXiv v1 预印本结果，绑定 9 个重置式网站、合成账号/实体、8 个 VLM 配置、VisualWebArena scaffold、攻击候选、验证器和 rollout 预算；论文称实验不触达真实网站、用户或账号。ASR 和 BSR 不能解释成真实部署发生率，也不能据此将模型按普遍安全性排序。5.12 个百分点是该 benchmark 匹配预算下的差异，不能外推到其他攻击机制或环境。
