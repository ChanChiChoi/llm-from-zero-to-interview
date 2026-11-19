# 第二十一册本地链接首轮扫描（2026-11-19）

## 范围与结果

- 扫描范围：第二十一册目录内全部 Markdown 文件，包括 95 章、简介、目录和独立时间线，共 98 份文件。
- 发现 139 个本地 Markdown 链接实例，其中 28 个实例目标缺失，涉及 20 个不同目标文件；失效引用分布在 12 份文件中。
- 95 章共 2,092 个 Markdown 围栏标记，均成对；将 10—12 章的 `x.0` 作为范围/资料前言后，发现第 88 章部分新增 H2 次序倒置。围栏配对和标题发现只是结构静态扫描，不代表内容已验收。
- 所有失效链接集中在 source-note 路径和旧 `code/` 示例路径；不以创建空白占位文件方式修复。是否删除旧脚本链接、合并到其他研究记录或重建来源卡，要结合对应章节正文和现有证据逐项判断。

## 按文件分组的缺失目标

| 正文/资料 | 扫描发现的目标 |
| --- | --- |
| `主流大模型架构创新时间线.md` | `qwen3-omni-source-notes.md`、`qwen3-vl-source-notes.md`、`k2-horizon-3.7b-source-notes.md` |
| `chapters/81-deepseek-v4.1-flash-causal-encoder-decoder.md` | `deepseek-v4.1-flash-source-notes.md`、`code/deepseek_v41_api_contract_audit.py`、`code/deepseek_harness_protocol_audit.py`、`code/deepseek_v41_cache_demo.py` |
| `chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md` | `code/qwen_max0902_cache_contract_audit.py`、`qwen3.5-397b-a17b-source-notes.md`、`qwen3.6-35b-a3b-source-notes.md`、`qwen3.6-27b-source-notes.md` |
| `chapters/84-glm-5.3-flash混合注意力与视觉闭环.md` | `glm-5.3-flash-source-notes.md` |
| `chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md` | `deepseek-v4-flash-vision-source-notes.md`、`deepseek-v4-source-notes.md` |
| `chapters/86-glm-5.2-indexshare-mtp与长轨迹rl.md` | `glm-5.2-source-notes.md` |
| `chapters/87-gpt-oss开放权重moe-harmony与可变推理.md` | `gpt-oss-source-notes.md` |
| `chapters/88-kimi-k3-kda-stable-latentmoe与百万token-agent.md` | `code/kimi_k3_manifest_audit.py`、`kimi-k3-source-notes.md` |
| `chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md` | `qwen3-omni-source-notes.md` |
| `chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md` | `qwen3-vl-source-notes.md` |
| `chapters/93-qwen3.5-omni-aria-aut-timestamp与流式语音.md` | `qwen3.5-omni-source-notes.md` |
| `chapters/94-kimi-k2.7-code-long-horizon-coding-agent.md` | `code/kimi_k27_contract_budget_toy.py` |

目标路径均位于 `research/model-update-2026-09/` 下。相同 source note 在正文与时间线可能有多条引用，表格按目标文件去重，不表示已合并或替代了它们。

## 逐章修复方法

1. 审阅对应章节全文、已有研究记录、共享索引与配套题目后，确认每项目标记录的来源和边界。
2. 若证据存在于其他研究记录，校准该记录覆盖范围后改链接；若缺少研究来源，按模块 03 建立有来源、有版本、有边界的 source note；若链接指向已删除脚本，删除旧脚本声明并保留必要的数值推导或人工核验依据。
3. 每完成一章，更新该章审计和本清单的状态；全部章节结束后重扫 98 份文件并核对研究事实与索引落点。

本扫描不访问外链、不运行示例脚本或 API，也不更新 PDF。
