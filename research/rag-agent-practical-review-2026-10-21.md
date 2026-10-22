# 第三册第七章 RAG 与 Agent 项目实战审阅记录

## 范围与证据

- 正文：`book-03-practical-handbook/chapters/07-rag与agent项目实战.md`，从头复读 7.1—7.8，覆盖文档清洗与切分、embedding/FAISS、reranker、引用与证据、分层评估、Tool Calling、ReAct、Agent 安全和贯通案例。
- 配套：第三册目录第 7 条指向正文文件；本轮在 `PAPERS.md` 的 RAG 与 Agent 资料组补入 FActScore、AttributionBench、RAGAS、AgentDojo 和 ToolSandbox。题库、练习、术语表已有 RAG、引用、工具调用和 Agent 安全主题，本轮未复制章节摘要。
- 联网方式：通过指定代理 `http://10.24.27.134:7890` 检查正文来源。正文去重后 34 个唯一 URL，最终均 HTTP 200；OpenAI、Anthropic 和 OWASP 页面有重定向，正文更新到本轮检查到的官方落地地址。
- 论文元数据：17 个唯一 arXiv ID 的 `citation_title` 均与正文引用主题一致。发现并修正 FActScore 原误链 `2305.14250` 为 `2305.14251`；`2402.04397` 是无关物理学论文，AttributionBench 改为 ACL Anthology 正式页面 `2024.findings-acl.886`。两条更正后的页面均返回 HTTP 200。题名核验不等于逐篇全文、附录或源码复核。

## 修订

- 7.1.10 的 Sentence Transformers semantic-search 链接去掉失效/过时的 README 路径，改为当前可达页面。
- 补充 Sentence Transformers、NumPy、FAISS 和模型文件的外部依赖说明；3 个外部模型/索引围栏不在无依赖 CPU 环境执行，但均通过语法编译。
- 将工具注册、多个工具调用和 Tool Calling 审计 demo 改为自包含代码，避免读者复制单块时依赖未定义的前文状态。
- `execute_registered` 不把 Python 异常原文返回给模型，只返回稳定的 `tool_execution_failed`；注释说明生产系统需要单独的受保护日志。
- 安全 guard 的高风险动作返回 `confirmation_required` 和未执行错误；员工查询结果在返回值中脱敏，审计事件与模型可见结果保持一致。
- 修正 `PAPERS.md` 的 RAG/Agent 论文入口和证据范围说明。

## 代码与公式验证

- 28 个 Python 围栏全部通过语法编译；25 个无外部模型依赖的代码块在 `.tmp/llm-book-cpu-check`（Python 3.12.9、PyTorch 2.12.0+cpu 环境，可执行示例本身主要使用标准库）独立运行通过。3 个需要 `numpy`、`faiss-cpu`、`sentence-transformers` 和模型文件的块只做语法验证，未下载模型或额外库。
- 边界断言全部通过：chunk 覆盖与非法窗口、空检索/非法 top-k、零上下文预算/负 token 数、空 gold 指标和非法 k、计算器代码/除零/非有限输入、工具异常细节不泄露、多工具 call id 与结果绑定、Agent 零步预算、高风险动作不执行和敏感字段脱敏。
- 检查 51 个 `math` 围栏的成对结构、花括号/方括号平衡、无 `$$`、无旧式 `\[`/`\]`、无 `\operatorname`、无 `\text{...}` 下划线和尖括号下标；正文没有重复的一级/二级/三级标题。
- `git diff --check` 通过；章节与第三册目录的相对路径有效。未运行外部数学渲染器和 PDF 构建，后者仍受环境缺少 Pandoc 限制。

## 证据边界与遗留项

- sparse RAG、关键词 reranker、引用正则、mock ReAct 和安全 guard 只验证局部数据流、状态和边界，不代表真实 embedding 召回、cross-encoder 排序、LLM 生成质量、工具服务可靠性或生产安全。
- 未执行真实模型、FAISS 大规模索引、向量数据库、付费 API、真实浏览器/代码沙箱或非幂等外部写操作；相关接口、权限、网络和成本结论仍需按目标 revision、租户、工具和负载实测。
- 评估公式对空集合、零分母和拒答分母已明确未定义/不适用；benchmark 论文和公开威胁模型不替代目标系统的人工抽查、回放和权限审计。
