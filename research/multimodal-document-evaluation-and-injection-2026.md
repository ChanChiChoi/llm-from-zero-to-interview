# 多模态文档评测与感知通道注入：来源证据记录（2026）

核验日期：2026-11-05。本文只记录联网发现的新资料、来源支持的事实及证据边界，不记录章节审阅进度或正文修改决定。

## PureDocBench v2：干净、数字退化与真实拍摄的配对评测

- **来源与版本：** [PureDocBench: A Source-Traceable Benchmark across Clean, Degraded, and Real-World Settings](https://arxiv.org/html/2605.07492v2)，arXiv:2605.07492v2，页面版本日期 2026-09-27。阅读摘要、§1、结论与附录 C 的 Real-track 可读性审计。
- **来源支持的事实：** 作者发布 1,475 个页面，覆盖 10 个领域和 66 个子类；每页有 Clean、Digital、Real 三种对应图像，共 4,425 张，文本、公式、表格和阅读顺序标注由共同 HTML/CSS 源生成。v2 评估 58 个模型；作者报告三轨平均分最佳为 79.69/100、整体均值 63.84，平均分相对 Clean 在 Digital 下降 3.34 分、在 Real 下降 11.61 分。
- **作者声明的限制：** 评测沿用 OmniDocBench 的指标框架，作者认为这些指标仍需改善，以表达语义遗漏和结构一致性；HTML 生成方式对手写材料和历史书籍覆盖有限。两名标注者的可读性审查标记了 83 张 Real 图像（约 5.63%）难以可靠核对转写；排除这些图像后的敏感性分析中，作者称模型排序大体稳定。
- **证据等级与边界：** source-backed（arXiv 预印本及其公开数据/代码声明）。性能和退化差异绑定该版本的数据生成、图像轨道、模型集合和评测实现；不能作为任意企业文档或生产 OCR/VLM 准确率，也不能把其作者评估的指标问题扩展成所有文档 benchmark 均无效。

## MMPIBench：多模态提示注入的阶段轨迹

- **来源与版本：** [An Experimental Evaluation of Multimodal Prompt Injection Attacks on Agentic AI Frameworks](https://arxiv.org/abs/2609.09404)，Viet K. Nguyen、Mohammad I. Husain，2026-09 预印本。阅读摘要、基准设计、图像实验结果及讨论/限制。
- **来源支持的事实：** 该研究以 24 个攻击样本（六种视觉载体与四种攻击目标）交叉评估五个模型和六个 Agent 框架，在温度 0 下报告 720 次运行。作者将 attempted 作为汇总指标：92/720 次（12.8%）被判定为按注入指令行动，其中包含 8 次完成攻击、72 次部分执行和 12 次未完成的尝试；completed 为 8/720（1.11%），partial execution 为 72/720（10.0%）。这几项存在包含关系，不能相加。阶段追踪记录载荷到达感知、规划和工具调用的最远位置。
- **重要限制：** 工具环境为 mock sandbox，completed 表示发出了研究定义下的有害调用，不表示真实系统受到破坏；多数实验格只运行一次，159 格的重复执行集中在两个框架，且其中一个框架使用了不同的提示适配。审查标签包含模型评审器。音频实验有不同的任务 framing 和工具邀请，作者明确不支持把视觉与音频攻击率作直接横向比较；所有结论是固定模型、框架和配置的快照。
- **证据等级与边界：** source-backed（arXiv 预印本，研究者自建 harness 和 mock tools）。这些分母和类别只适用于其协议。它支持“评估需分别观察攻击识别、动作尝试、部分执行、完成和被阻断阶段”这一诊断思路，不提供企业系统的实际攻击率或安全认证。

## OWASP File Upload Cheat Sheet：上传内容验证

- **来源：** [OWASP File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)，访问日期 2026-11-05；阅读 Content-Type Validation、File Signature Validation 和 File Content Validation。
- **来源支持的事实：** Cheat Sheet 指出上传方提供的 Content-Type 可以伪造，文件签名检查可与类型检查配合，但单独使用文件签名不足以保证安全。图像解码、重新编码和移除不需要的元数据可作为措施；处理器本身仍在处理不可信输入，应使用维护中的库并限制格式、资源和执行环境。
- **证据等级与边界：** source-backed（OWASP 官方安全建议）。它提供防御纵深建议，不证明特定解析器、沙箱或上传系统已安全，也不取代目标业务的文件类型清单、威胁建模和部署验证。

## OWASP LLM01:2025：多模态间接提示注入

- **来源与版本：** [OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)，访问日期 2026-11-05；阅读风险说明、间接注入、多模态示例和缓解建议。
- **来源支持的事实：** 官方风险条目指出，外部文件可携带会改变模型行为的间接指令；多模态系统可能在看似正常的文本旁边处理图像隐藏指令。建议措施包括区分并标记不可信外部内容、输入/输出过滤、最小权限、高风险动作人工批准和持续对抗测试；页面说明 prompt injection 没有已知的万无一失防护方法。
- **证据等级与边界：** source-backed（OWASP 官方风险分类和安全建议）。该页面支持风险类别及纵深缓解思路，不证明特定模型会读取某个图像载体，也不认证某一系统已经阻断注入。
