# 共享索引本地链接复核（2026-11-19）

## 范围与结果

- 在第二十册整册终验中，对第二十册正文和配套维护文件进行本地 Markdown 链接扫描：53 份文件、261 个本地链接目标，缺失 0。
- 同一轮扩展到共享索引 `PAPERS.md`、`BOOK_SERIES.md` 等入口后，59 份文件共检查 527 个本地链接目标，发现 30 个失效链接实例、涉及 28 个不同目标；失效项全部位于 `PAPERS.md` 和 `BOOK_SERIES.md`，不在第二十册正文或其进度目录中。
- 此扫描是本地 Markdown 导航检查，不表示外链可达性、GitHub 图表/公式渲染或整仓库其他未纳入文件均已通过。

## 待逐项处置的目标

### `PAPERS.md`（21 个不同目标，23 个失效链接实例）

- `research/book-08-ai-safety-alignment-architecture-scan-2026-10-27.md`
- `research/book-09-book-level-architecture-2026-10-28.md`
- `research/evaluation-debug-practical-review-2026-10-22.md`
- `research/model-update-2026-09/claude-opus-4.6-source-notes.md`
- `research/model-update-2026-09/claude-opus-4.7-source-notes.md`
- `research/model-update-2026-09/code/gpt56_luna_state_replay_audit.py`
- `research/model-update-2026-09/code/gpt6_sol_contract_audit.py`
- `research/model-update-2026-09/code/qwen_max0902_cache_contract_audit.py`
- `research/model-update-2026-09/deepseek-v4-source-notes.md`
- `research/model-update-2026-09/deepswe-snapshot-notes.md`
- `research/model-update-2026-09/gemini-3.8-flash-source-notes.md`
- `research/model-update-2026-09/glm-5.1-source-notes.md`
- `research/model-update-2026-09/glm-5.3-flash-source-notes.md`
- `research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md`
- `research/model-update-2026-09/gpt-5.6-source-notes.md`
- `research/model-update-2026-09/k2-horizon-3.7b-source-notes.md`
- `research/model-update-2026-09/qwen3-omni-source-notes.md`
- `research/model-update-2026-09/qwen3-vl-source-notes.md`
- `research/model-update-2026-09/qwen3.6-27b-source-notes.md`
- `research/model-update-2026-09/qwen3.6-35b-a3b-source-notes.md`
- `research/safety-robustness-evaluation-depth-review-2026-10-27.md`

### `BOOK_SERIES.md`（7 个不同目标，7 个失效链接实例）

- `research/model-update-2026-09/claude-opus-4.8-source-notes.md`
- `research/model-update-2026-09/deepswe-snapshot-notes.md`
- `research/model-update-2026-09/gpt-6-luna-source-notes.md`
- `research/model-update-2026-09/gpt-6-sol-source-notes.md`
- `research/model-update-2026-09/kimi-k2.7-code-source-notes.md`
- `research/model-update-2026-09/qwen3.5-omni-source-notes.md`
- `research/model-update-2026-09/qwen3.6-35b-a3b-source-notes.md`

## 处置规则与交接

- 随对应书册和模型主题审阅时，查明每项是目标文件移动、来源记录缺失、研究记录已合并，还是链接残留在已删除的审计脚本；按证据更新索引或补齐真实研究记录。
- 不为消除扫描报错创建空白占位文件，也不在未读对应正文/研究材料前推断替代目标。
- `BOOK_SERIES.md` 中 Gemini 3.8 链接已由第二十册第二十三章工作单独修复；本清单只保留当前扫描仍失效的目标。后续处置时按扫描当日状态重算，不把本清单视作永久链接结果。
