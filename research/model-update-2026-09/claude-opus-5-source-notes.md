# Claude Opus 5：System Card 版本与 Agentic Safety 来源记录

资料核对日期：2026-11-03。本文记录 Anthropic 对 Opus 5 的公开 System Card 与评测材料，不是章节审计、独立复现或生产风险率估计。

## System Card 修订

- 官方入口：[Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card)；发布方 [Transparency Hub](https://www.anthropic.com/transparency) 汇总部分评估结果。
- System Card 的 2026-08-19 changelog 记录：发现 Cowork browser harness 与旧模型基线不一致后，统一 Cowork harness 并重跑基线；由于 Cowork 不支持关闭 thinking，修订版不再报告 `thinking disabled` 条件。该变化属于测量协议与比较条件修正，不能写成模型权重升级。
- 对适应性红队结果应保留模型 snapshot、surface、thinking/effort、攻击策略、预算、probe、Auto mode、harness、验证器和成功定义；attempt-level 攻击率与 scenario-level 成功率回答不同问题。System Card 报告的数字不能当作生产用户风险概率。

## Gray Swan IPI 与 Cowork Browser

- Anthropic 的官方摘要说明 Gray Swan Indirect Prompt Injection benchmark 采用 28 个场景和去重后的 1,130 个高迁移攻击，并报告攻击者在 1、10、15 次尝试内成功的概率。Opus 5 在一次与十五次尝试下分别为 0.2% 和 2.0%；其比较绑定评测版本和配置，不能和不同 endpoint 或 safeguard 状态的模型直接排名。
- System Card 的 Cowork browser 评估记录 129 个未见环境、每个环境十次尝试，并区分 Auto mode 条件。公开表中 3.84% 与 0/129 分别对应不同成功单位/判定口径；其中 Auto mode 包含对不可信工具结果和危险工具调用的检查。该结果描述产品保护配置下的系统行为，不等同于裸模型攻击成功率。
- 这组结果支持“系统卡数字要绑定环境、harness、保护措施和分母”的结论；它不证明其他产品 surface、未测攻击或生产流量具有相同表现。完整背景见上述官方卡片及 [Anthropic System Card 摘要页](https://www.anthropic.com/transparency)。
