# Artificial Analysis 2026-09-14 实时快照

采集日期：2026-09-14。来源：[Artificial Analysis](https://artificialanalysis.ai/)。本文件是独立快照记录，不覆盖此前的 2026-09-09 `/tmp/artificialanalysis.html` 盘点。

## 快照身份

- 页面：`https://artificialanalysis.ai/`
- HTTP：200，经可用网页代理 `10.24.27.134:7890` 获取。
- 文件：`/tmp/proxy-live-artificialanalysis.html`
- 大小：1,771,961 bytes。
- SHA-256：`ebda1f3ff7fc1956dc629b82ab56400300683a8a134446882d12dfbb4d814423`。
- 页面结构化数据：约 702 个配置条目、673 个唯一名称，最新 `releaseDate` 字段为 `2026-09-11`。
- 旧快照对照：2026-09-09 文件大小 1,863,708 bytes，SHA-256 `bd8cfd2e4120012b7f58fe48e6c37117667fe223077a4448b675de8bb91f51f3`。

页面中的配置行可能因为 reasoning effort、非 reasoning、fallback、provider 或 canonical release 而重复表示同一个基础模型。这里的“新增 13 个 slug/别名”是当前页面数据相对旧快照的结构变化，不等于 13 个独立基础模型。

## 新增结构条目

| 页面显示名 | 新增 slug/别名 | 页面 releaseDate | 当前证据状态 |
|---|---|---:|---|
| DeepSeek V4.1 Flash (Reasoning, Max Effort) | `deepseek-v4-1-flash` | 2026-09-10 | 已由 DeepSeek 官方发布页和 Hugging Face 模型卡核验 |
| Agnes 3.0 Flash | `agnes-3-0-flash` | 2026-09-11 | 仅榜单发现，待官方核验 |
| Ling-3.0-flash-VL | `ling-3-0-flash-vl` | 2026-09-10 | 仅榜单发现，待官方核验 |
| K2 Horizon MoVA 36B A4B | `k2-mova-36b-mid5`, `k2-horizon-mova-36b-a4b` | 2026-09-03 | 榜单候选已由 IFM 官方模型卡、固定配置和实现核验 |
| K2 Horizon 7B | `k2-7b-ph2`, `k2-horizon-7b` | 2026-09-03 | 榜单候选；官方 7B 卡用于核验 Uno base 关联，完整模型专题待定 |
| K2 Horizon 3.7B | `k2-4b-ph1`, `k2-horizon-3-7b` | 2026-09-03 | 仅榜单发现，完整官方资料待核验 |
| K2 Horizon 0.9B | `k2-1b-final`, `k2-horizon-0-9b` | 2026-09-03 | 榜单候选；官方同系列卡片仅用于记录 MOPD 边界，完整规格待核验 |
| MBZUAI provider/目录条目 | `mbzuai` | 页面关联字段 | 仅榜单/目录结构，不能当作模型 |
| DeepSeek V4 Flash (Non-reasoning) | `deepseek-v4-flash-0420-non-reasoning` | 2026-04-24 | 配置发现；不替代官方 V4.1 API 路由说明 |

表格按 canonical 名称合并了成对的 slug，因此显示行少于 13 个结构新增项。K2 的成对项尤其不能直接相加参数量；它们更可能是评测配置与 release alias 的不同记录。

## 证据边界

Artificial Analysis 的 `releaseDate`、开放性标签、参数字段和 Intelligence Index 都是第三方页面字段。它们适合发现候选、比较同一页面内配置和记录采集时间，不替代：

- 官方发布日期、模型卡 revision 和许可证；
- 参数量、激活参数、架构和训练配方；
- 固定 harness 下的独立 benchmark；
- API 当前 alias、价格、限流和实际可用性。

本轮 DeepSeek V4.1-Flash 与 K2 Horizon MoVA 36B/A4B 已获得相应官方资料支持并进入正式专题；Agnes、Ling、K2 Horizon 3.7B 和 `mbzuai` 仍停留在发现层。K2 7B/0.9B 只在关联资料边界内引用官方卡片，不能据此扩写完整规格。`deepseek-v4-flash-0420-non-reasoning` 也只表示榜单配置发现；DeepSeek API 对 alias 的当前路由以官方发布页为准。

## 后续复核

1. 保存下一次页面快照并比较哈希、条目总数、最新日期和新增/消失 slug。
2. 对 Agnes、Ling 和 K2 Horizon 3.7B 分别寻找官方模型卡、研究页、技术报告和许可证；对 K2 36B/A4B 继续核验完整训练报告、kernel 和实测性能。
3. 若引用 Artificial Analysis 指数，必须同时保留页面配置名、effort/fallback、采集日期和 canonical slug，且不与 DeepSWE 或其他 harness 分数直接合并。
