# Gemini 4 Argon：排行榜新锚点与官方资料缺口

核验日期：2026-10-01。本笔记只记录两个指定排行榜中的 `gemini-4-argon`，不把榜单字段扩写成 Google 官方模型事实。

## 当前证据

## 2026-10-01 再次刷新

- Artificial Analysis `/zh` 经用户提供的 `http://10.24.27.134:7890` 显式代理取得 HTTP 200，快照 `1,930,483` bytes，SHA-256 `3a962c79cec028baf1e642d8e14fa20928c6b8470d2f70b3568b5fff7e559ee6`；页面仍包含 `gemini-4-argon`，未观察到其从榜单移除。
- DataCurve DeepSWE 同日 HTTP 200，快照 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；页面仍有 70 个 `mini_swe_agent_*` 配置，未出现精确 `mini_swe_agent_gemini_4_argon_*`。
- Google AI Developers 模型目录 HTTP 200，快照 `155,241` bytes，SHA-256 `a0491f5f0b9f9fc107c42183fb6cf9d22927e1af03dbbdfa0ccc38e4e4ac31`；正文没有 `argon` 或 `Gemini 4 Argon`。Google sitemap HTTP 200，快照 `253` bytes，SHA-256 `8dadac65fc82806836873c16fc1f813597e99b3e5f838d61c0e4435b8cfc8f7b`，同样没有 Argon 条目。

本次刷新没有获得 Google 一手模型页、模型卡、技术报告、博客或代码。状态保持：**AA 新锚点 + 官方资料缺口**；不新增正式章节、参数表、架构结论或 Agent 评测结论。

## Google 官方扩展入口复查

- Google Blog 搜索页 `https://blog.google/search/?query=Gemini%204%20Argon` HTTP 200，`236,062` bytes / SHA-256 `f476efb4056e53283469acf04d3b5f2aaf75f48a637cf4dcb0f73ef7c13073ea`；页面正文没有 `argon` 或 `Gemini 4` 命中。
- Google AI Blog 首页 HTTP 200，`405,476` bytes / SHA-256 `95f83f7704c64b995cab1899c76003a88f7284d13a327c65e85d5949a64c852b`；本次抓取正文没有 Argon 命中。
- Vertex AI 模型目录 `https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models` HTTP 200，`424,706` bytes / SHA-256 `cde940b9cf0c1447b3f4e2a6384f3596aeebe8d2193da9b7942eb2695d891475`；页面出现 Gemini 3.5/3.6/3.7/3.8 条目，没有 Gemini 4 Argon。
- Gemini API 模型目录 `https://ai.google.dev/gemini-api/docs/models/gemini` HTTP 200，`155,237` bytes / SHA-256 `191979fb43d7735ee22072a35adc802ce1cc6d1aa90e50a858d8688c93504e6c`；页面列出 Gemini 3.x/3.8 系列，没有 `argon` 或 Gemini 4 条目。

这些页面属于 Google 官方入口，但负检索不能证明模型不存在，只能证明本次抓取未找到可引用的一手 Argon 资料。状态继续保持官方资料缺口。

## 2026-10-01 详情页与公开搜索复验

- Artificial Analysis Argon 详情页 HTTP 200，`3,813,795` bytes / SHA-256 `d1548a30f1de26f5f8a528bf7d133d191b68ec3e559bc5e3ec30a67405351388`；页面标题为 `Gemini 4 Argon (high)`，并提供 provider 子页入口。详情页仍属于第三方目录/测量来源，不能替代 Google 发布资料。
- Google 搜索 exact phrase `"Gemini 4 Argon"` HTTP 200，`91,696` bytes / SHA-256 `f9f1fea30a6d26f1388ea5d7204d36b778c2da028ab8e420f4a596948c829026`；Bing exact phrase 搜索 HTTP 200，`120,933` bytes / SHA-256 `37afde007528953b316c9e0c01682f703c7cb5a2ce210d955dc57927c7762274`。搜索结果页没有形成可引用的 Google 官方 Argon 页面；搜索摘要不纳入技术证据。

结论不变：Argon 仍只有 AA/provider 侧发现证据，尚无 Google 官方模型页、模型卡、技术报告、博客或代码可用于内容闭环。

## Google 限定域名检索

- 通过 7890 代理对 `site:ai.google.dev "Gemini 4 Argon"`、`site:cloud.google.com "Gemini 4 Argon"`、`site:deepmind.google "Gemini 4 Argon"`、`site:blog.google "Gemini 4 Argon"` 和 `site:google.com "gemini-4-argon"` 分别请求 Google 搜索，均 HTTP 200；对应快照 SHA-256 依次为 `3d397d7d261ade8bd53037b6adb9607462161e02ef19617f8bb042630552c0fc`、`cb6d97c3684bbfe3a20d077381f26c81f74bf46ff972834e9506b8fb97b097c4`、`75b5758aa36adab9f71f64c39edb5a411622f3ad43ba6ab49d5b74f3292582ea`、`c202f41be059f9b5ef412227934d0e0f62af14ce2cb0c966e17eba9c9e1aef6d`、`7e1452b37fa6ce1739fec30102968ce418b7a6c36133b979a0652d605e8be107`。
- 限定域名结果页没有提取出可引用的 Google 官方 Argon URL；`google.com` 结果中的查询词回显不构成命中。搜索索引结果仍只能作为负检索证据，不能证明模型不存在。

状态保持：**AA 新锚点 + 官方资料缺口**。

## 2026-10-01 Google 官方发布：首次形成资料闭环

Google The Keyword 官方文章：[Introducing Gemini 4 Argon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/) 经 7890 代理 HTTP 200，`408,929` bytes，SHA-256 `1c09a8019b06eada8703e67c7da2d4269ff3dc9169ae66226784dbec2d3043df`。文章发布时间为 2026-09-30T20:00:00Z，作者 Koray Kavukcuoglu。

### 可引用的发布方事实

- Argon 是 Google 新的 frontier model，面向真实软件工程、企业知识工作（法律、金融等）和网络安全防御；首批通过 Fairwind Program 向受信任的网络防御者逐步开放，尚未面向开发者、企业和消费者全面开放。
- 官方宣称输出上限为 1M tokens（由此前 64K 提升），用于长程、多步骤推理；这是发布方 API/产品声明，不是独立 endpoint 验收。
- 官方公布计划内价格为输入 `$2 / 1M`、输出 `$10 / 1M`，缓存输入折扣 95%；广泛发布后输入 `$4 / 1M`、输出 `$20 / 1M`。价格属于产品公告，不能推导模型规模或吞吐。
- Google 报告内部工作流案例：量子算法资源优化、数据中心内存优化（已部署节省超过 300 TiB，估计总节省 500 TiB–1 PiB）、C/C++ 到 Rust 的大规模迁移；libgav1 Rust 迁移案例称在保持输出一致的情况下达到 Rust port 的 2.7x。以上均为 Google 自报案例，未作为独立 benchmark。
- Google 报告 DeepSWE v1.1 `77.9%`、AutomationBench `51.3%`、LVBench `91.7%`，以及 CWE-bench v1 `68%`；这些是发布方选择的评测/结果，不能与 DataCurve 的精确 Argon 行混同（DataCurve 当前无该行），也不能当作独立复现。
- 官方称 Argon 可自主发现、验证并修复关键漏洞；Wiz Scan for Good 和 Google 内部测试属于合作方/内部案例，安全影响需按其测试边界理解。
- 发布前安全措施包括分阶段访问、自动与人工 red team、adversarial training、间接 prompt injection 防护、Frontier Safety Framework，以及对内部 activation 的监测研究；文章没有披露完整训练 recipe、参数量、层结构、权重或 kernel。

### 闭环边界

Argon 现在达到：**AA 单榜发现 + Google 官方发布博客 + 发布方评测/安全边界**。仍未获得公开模型卡、技术报告、权重、完整 API 文档、独立复现或精确 DataCurve Agent 配置；不要把 Google 自报数字写成独立测量，也不要把 1M 输出上限写成已完成生产验收。

- Artificial Analysis `/zh` 经 `http://10.24.27.134:7890` 显式代理取得 HTTP 200，快照 `1,930,493` bytes，SHA-256 `4fa2574617fc125da6e0cf6a8575d5b81084baf22a134fb69a5ad450504c8c8d`；页面包含 `gemini-4-argon` canonical 路由。
- DataCurve DeepSWE 经同一代理取得 HTTP 200，快照 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；没有精确 `mini_swe_agent_gemini_4_argon_*`。
- Google AI Developers 模型目录经同一代理取得 HTTP 200，快照 `155,241` bytes，SHA-256 `032840f32f36ad8cfa30d0b2767f9ae17dbaca5a17ed3acfc8a1e6fa97b95c47`；目录没有 `gemini-4-argon` 或 `Gemini 4 Argon`。

## 状态与边界

当前状态：**AA 新锚点 + Google 官方资料缺口**。它还不是资料级闭环，不新增技术章节、参数表、架构结论或 Agent 评测结论。AA/provider 字段只能作为第三方发现字段；取得 Google 官方模型页、模型卡、技术报告、博客或代码前，不写成 Google 正式规格。

## 2026-10-01 官方入口复查

- Google AI Developers sitemap `sitemap_0_of_1.xml` HTTP 200，`15,361,787` bytes / SHA-256 `f74529bc6bc9685feab9c9dba0116a2dfcebe0ea16c2b7a6b764c3857da00e98`，全文没有 `argon`。
- 猜测的 AI Developers 模型页 `/gemini-4-argon` 返回 HTTP 404（`83,882` bytes）；猜测的 DeepMind Model Card 返回 HTTP 404（`116,775` bytes）。404 页面中出现的搜索词或通用站点文案不构成模型资料。
- 同日 AA `/zh` 仍 HTTP 200，`1,930,493` bytes / SHA-256 `a47cc50856c95c96d04679f1d52c6cf76dacb26a78248d8e278ccacdf5590efe`；DataCurve 仍 HTTP 200，`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，无精确 Argon 行。

结论不变：Argon 仍是排行榜锚点，尚未形成 Google 权威资料闭环。

## Artificial Analysis 详情页边界

- AA 详情页本轮 HTTP 200，`3,813,855` bytes / SHA-256 `aaaf7cbe1b821904de6149a9dfd0810c54c9db878054e30b9014676d77db341`。页面把它标为 Google、proprietary、2026 年 9 月发布，并称价格/可用性来自 Google API、当前有 1 个 API provider。
- 这些字段仍是 Artificial Analysis 的目录与 provider 观测，不是 Google 官方模型卡或发布公告。详情页明确写出 Google 未披露参数规模；本项目不把其 Intelligence Index、价格、context 或 provider 描述升级为官方规格。
- Google 搜索结果页本轮 HTTP 200，但没有形成可核验的 Google 官方 Argon 页面；搜索壳和摘要不作为技术来源。
