# 第十一章：Error Analysis 与回归测试

重点：失败样例聚类、能力维度拆解、regression suite、版本对比、root cause 分析。

面试重点：评估的目的不只是排名，还要指导下一步改进。

## 本章目标

学完本章，你要能回答：

1. 为什么平均分提升后仍然必须做 error analysis？
2. 如何把失败样例拆成可行动的能力维度？
3. 如何对错误样例做聚类和 root cause 分析？
4. 什么是 regression suite，它和普通 benchmark 有什么区别？
5. 模型、prompt、RAG、工具链版本迭代时，如何做版本对比？
6. 如何把 error analysis 结果转成数据、训练、prompt、系统和评估改进？
7. 面试中如何讲清楚一个完整的评估闭环？

前面几章讨论了 benchmark、human eval、LLM-as-a-judge、线上实验和统计显著性。

这些方法可以回答一个问题：模型 A 是否比模型 B 更好？

但真实项目里，仅知道“更好”通常不够。

你还需要回答：

1. 好在哪里？
2. 坏在哪里？
3. 哪些问题是新版本引入的回归？
4. 哪些错误最影响用户价值？
5. 下一步应该改数据、改训练、改 prompt、改检索，还是改工具链？

Error analysis 的目标，就是把评估结果从一个分数拆成可诊断、可定位、可行动的证据。

Regression testing 的目标，是防止已经修好的问题在新版本中重新出现。

面试表达：评估不是为了得到一个 leaderboard 分数，而是为了形成“发现问题、定位原因、修复问题、防止复发”的闭环。

## 1. 为什么平均分不够

假设两个模型在同一个测试集上的结果如下：

```text
Model A accuracy = 82.0%
Model B accuracy = 84.0%
```

Model B 看起来更好。

但这个结论仍然有很多盲区。

### 1.1 平均分会掩盖局部退化

Model B 可能在多数简单样本上更好，但在某些关键场景明显变差。

例如：

1. 总体提升 2%。
2. 中文法律问答下降 8%。
3. 代码补全提升 6%。
4. 数学证明下降 5%。
5. 安全拒答率过高。

如果产品重点用户是法律和数学用户，总体提升就可能误导决策。

### 1.2 不同错误的业务成本不同

一个拼写错误和一个医疗建议错误，不能只按“错一题”同等处理。

大模型错误通常需要区分严重程度：

1. 轻微格式问题。
2. 答案不完整。
3. 推理链条错误。
4. 事实幻觉。
5. 工具调用错误。
6. 安全风险。
7. 隐私泄露。

面试中要强调：error analysis 不只是统计错误数量，还要评估错误类型、严重程度和业务影响。

### 1.3 分数不能直接告诉你怎么改

模型分数下降可能来自很多原因：

1. 训练数据覆盖不足。
2. SFT 数据格式不一致。
3. 偏好数据奖励了错误风格。
4. RLHF 或 DPO 导致过度拒答。
5. Prompt 约束不清楚。
6. RAG 检索召回失败。
7. Tool schema 设计不合理。
8. Judge prompt 有偏差。
9. 测试集污染或标注错误。

如果没有 error analysis，团队只能凭感觉尝试修复。

## 2. Error Analysis 的基本流程

一个实用的 error analysis 流程通常包括七步。

```text
评估运行
  -> 收集失败样例
  -> 标注错误类型
  -> 按能力维度和场景聚合
  -> 找 root cause
  -> 设计修复方案
  -> 加入 regression suite
```

### 2.1 收集失败样例

首先要保存足够完整的评估日志。

至少包括：

1. 样本 ID。
2. 输入 prompt。
3. 参考答案或评分标准。
4. 模型输出。
5. 评估指标和 judge 解释。
6. 模型版本。
7. Prompt 版本。
8. 检索结果或工具调用记录。
9. Decoding 参数。
10. 时间戳和实验 ID。

如果日志不完整，后面很难定位问题。

### 2.2 定义错误 taxonomy

Taxonomy 是错误分类体系。

它应该满足两个条件：

1. 能覆盖主要失败模式。
2. 能指导后续修复动作。

例如，通用问答可以使用：

```text
错误类型：
- instruction_following：没有遵循用户指令
- factuality：事实错误或幻觉
- reasoning：推理步骤错误
- completeness：答案缺漏
- formatting：格式不符合要求
- safety：安全策略错误
- refusal：不该拒答却拒答
- over_answering：回答了用户没问的内容
```

RAG 系统可以使用：

```text
错误类型：
- retrieval_miss：没有召回相关文档
- retrieval_noise：召回了太多干扰文档
- citation_error：引用和答案不匹配
- faithfulness_error：答案不忠实于上下文
- synthesis_error：文档里有答案但模型没有整合好
- stale_knowledge：知识库内容过期
```

Agent 系统可以使用：

```text
错误类型：
- planning_error：任务拆解错误
- tool_selection_error：选错工具
- tool_argument_error：工具参数错误
- observation_error：没有正确读取工具返回
- loop_error：陷入重复动作
- permission_error：越权或权限处理错误
- final_answer_error：工具结果正确但最终回答错误
```

面试表达：错误分类不能只为了好看，而要和可执行修复路径绑定。

### 2.3 标注错误样例

错误标注可以由人工、LLM judge 或二者结合完成。

人工标注更可靠，但成本高。

LLM judge 成本低，但需要校准。

实战中常见做法是：

1. 先抽样人工标注一批。
2. 用人工样本定义 taxonomy 和标注指南。
3. 用 LLM judge 批量预标注。
4. 对高风险类别和不确定样本做人审。
5. 定期计算人和 judge 的一致性。

### 2.4 按维度聚合

单个错误样例只能说明局部问题。

聚合后才能发现系统性模式。

常见聚合维度包括：

1. 任务类型：问答、摘要、翻译、代码、数学、写作。
2. 语言：中文、英文、小语种、混合语言。
3. 难度：简单、中等、困难、长链推理。
4. 领域：医疗、法律、金融、教育、客服。
5. 输入长度：短上下文、长上下文、多轮对话。
6. 输出形态：JSON、表格、代码、引用、结构化报告。
7. 用户意图：信息查询、决策建议、创作、执行任务。
8. 安全等级：普通、敏感、高风险。

聚合目标是找到高频、高严重度、可修复的问题簇。

## 3. 能力维度拆解

大模型错误通常不是单一能力问题。

一个失败样例可能同时涉及理解、推理、知识、格式和安全。

因此需要把任务拆成能力维度。

### 3.1 通用能力维度

常见能力维度包括：

1. 指令遵循。
2. 事实准确性。
3. 推理正确性。
4. 复杂约束满足。
5. 长上下文定位。
6. 多轮一致性。
7. 格式稳定性。
8. 领域知识。
9. 安全合规。
10. 不确定性表达。

例如，一个答案“格式正确但事实错误”，和“事实正确但格式不符合 JSON schema”，修复路径不同。

前者可能需要数据、检索或知识增强。

后者可能需要 prompt、schema validation、constrained decoding 或后处理。

### 3.2 RAG 能力维度

RAG 评估中常见维度是：

1. Query understanding：是否理解用户问题。
2. Retrieval recall：是否召回包含答案的文档。
3. Reranking：相关文档是否排在前面。
4. Context packing：是否把关键信息放入上下文。
5. Grounded generation：是否忠实根据上下文回答。
6. Citation：引用是否精确。
7. Abstention：没有证据时是否拒答或说明不确定。

RAG 的 error analysis 不能只看最终答案。

必须拆开看检索、重排、上下文构造和生成。

### 3.3 Code 能力维度

代码任务可以拆成：

1. 需求理解。
2. API 使用。
3. 算法正确性。
4. 边界条件。
5. 复杂度。
6. 代码风格。
7. 测试生成。
8. 调试能力。
9. 多文件修改一致性。

例如，单元测试失败可能来自需求理解错，也可能来自边界条件漏掉。

如果只记录 pass/fail，就无法知道应该补哪类数据。

### 3.4 Reasoning 能力维度

推理任务可以拆成：

1. 问题形式化。
2. 中间变量定义。
3. 约束提取。
4. 计划生成。
5. 单步推理。
6. 多步组合。
7. 反例检查。
8. 最终答案验证。

很多 reasoning 错误不是模型“不会算”，而是第一步把问题理解错。

面试中如果能把失败拆到这个粒度，会比只说“模型推理能力不够”更有说服力。

## 4. 失败样例聚类

当失败样例很多时，人工逐条分析会很慢。

可以用聚类帮助发现模式。

### 4.1 手工聚类

小规模样本可以直接人工聚类。

做法是：

1. 抽取 50 到 200 个失败样例。
2. 每个样例写一句失败原因。
3. 把相似原因合并。
4. 给每个簇命名。
5. 统计每个簇的样本数和严重程度。

这种方法简单，但非常有效。

### 4.2 Embedding 聚类

大规模样本可以用 embedding 聚类。

基本流程：

```text
失败样例文本
  -> 构造分析文本：input + output + judge_reason
  -> 生成 embedding
  -> 聚类
  -> 每个 cluster 抽样查看
  -> 用人工或 LLM 总结 cluster 标签
```

用于聚类的文本不一定只包含用户输入。

更好的做法是包含：

1. 用户输入。
2. 模型输出。
3. 参考答案。
4. Judge 解释。
5. 初步错误类型。

这样聚类更容易按失败原因而不是只按主题聚合。

### 4.3 LLM 辅助归因

可以让 LLM 对失败样例生成候选原因。

示例 prompt：

```text
你是模型评估分析员。
请阅读一个失败样例，输出：
1. 错误类型
2. 失败原因
3. 可能 root cause
4. 建议修复方向

要求：只基于给定输入、模型输出、参考答案和检索上下文判断。
不要编造额外信息。
```

但要注意：LLM 给出的 root cause 只是候选假设，不是最终结论。

真正的 root cause 需要通过日志、对照实验和复现实验验证。

### 4.4 聚类结果怎么用

聚类结果应该转成优先级列表。

可以按下面公式粗略排序：

```text
priority = frequency * severity * fixability
```

含义是：

1. frequency：这个问题出现多频繁。
2. severity：这个问题有多严重。
3. fixability：这个问题是否容易修复。

高频、高严重度、容易修复的问题，应该优先处理。

低频但高风险的问题，也应该进入 regression suite。

## 5. Root Cause 分析

Root cause analysis 不是给错误贴标签，而是找出导致错误的系统原因。

### 5.1 常见 root cause 类别

大模型项目常见 root cause 包括：

1. 数据覆盖不足。
2. 数据标注不一致。
3. 训练目标和真实需求不一致。
4. Reward model 或 judge 偏差。
5. Prompt 约束冲突。
6. Decoding 参数不合适。
7. 检索召回不足。
8. 上下文截断导致关键信息丢失。
9. Tool schema 表达不清。
10. 系统集成 bug。
11. 安全策略过严或过松。
12. 评估集本身有问题。

### 5.2 用对照实验验证 root cause

不要只凭直觉判断 root cause。

例如，RAG 答错了，不一定是模型生成能力差。

可以做对照：

```text
实验 1：原始 RAG 链路
实验 2：人工提供正确文档，只测生成
实验 3：使用原始检索文档，但换更强模型
实验 4：固定模型，换检索器
实验 5：固定检索器，调整 prompt
```

如果人工提供正确文档后模型能答对，说明主要问题可能在 retrieval。

如果提供正确文档后仍然答错，说明可能是 generation、instruction following 或上下文理解问题。

### 5.3 五个为什么

“Five Whys” 是简单但实用的追问方法。

示例：

```text
现象：模型在财报问答中回答了错误数字。

为什么 1：模型使用了错误年份的数据。
为什么 2：检索结果里 2022 年财报排在 2023 年财报前面。
为什么 3：reranker 没有识别用户问题中的年份约束。
为什么 4：reranker 训练数据缺少时间敏感查询。
为什么 5：数据构造时只按主题相关性标注，没有按时间约束标注 hard negative。

root cause：检索训练数据缺少时间约束 hard negative。
修复：补充时间敏感查询和 hard negative，增加 reranker 特征或训练样本。
```

这种分析比“模型幻觉了”更有行动价值。

### 5.4 区分近因和根因

近因是直接表现。

根因是导致问题反复出现的系统原因。

例如：

```text
近因：模型输出 JSON 少了一个字段。
根因：prompt 没有明确 schema，服务端也没有 schema validation 和重试机制。
```

只修近因，问题容易复发。

修根因，才是真正改善系统。

## 6. Regression Suite

Regression suite 是一组专门用于防止能力退化的测试集。

它和普通 benchmark 不完全一样。

普通 benchmark 主要衡量总体能力。

Regression suite 更关注历史问题、关键场景和上线门禁。

### 6.1 为什么需要 regression suite

大模型系统频繁迭代：

1. 模型权重更新。
2. SFT 或偏好数据更新。
3. System prompt 更新。
4. RAG 索引更新。
5. Tool schema 更新。
6. 安全策略更新。
7. Decoding 和服务参数更新。

每次改动都可能修好一些问题，同时引入新问题。

Regression suite 的作用是：

1. 保护已经修好的关键样例。
2. 发现局部能力退化。
3. 作为上线门禁。
4. 让版本比较可复现。
5. 帮助团队积累真实失败经验。

### 6.2 Regression suite 应该包含什么

一个好的 regression suite 通常包含：

1. 历史线上事故样例。
2. 高频用户失败样例。
3. 高风险安全样例。
4. 重要客户或核心业务场景。
5. 已经修复过的 bug 样例。
6. 模型容易波动的边界样例。
7. 格式、工具、RAG、多轮等系统集成样例。

不是所有失败样例都应该加入 regression suite。

否则测试集会越来越大、越来越噪。

适合加入的样例通常满足：

1. 真实重要。
2. 失败原因明确。
3. 期望行为清楚。
4. 能稳定复现。
5. 有防止复发的价值。

### 6.3 样例格式

Regression case 应该结构化保存。

示例：

```json
{
  "case_id": "rag_finance_2023_revenue_001",
  "task_type": "rag_qa",
  "capability": ["temporal_reasoning", "retrieval", "faithfulness"],
  "severity": "high",
  "input": "请根据知识库回答：A 公司 2023 年营收是多少？",
  "expected_behavior": "必须使用 2023 年财报中的营收数字，并给出引用。",
  "failure_history": "v1.8 检索到 2022 年财报并回答错误数字。",
  "acceptance_criteria": {
    "must_include": ["2023", "引用"],
    "must_not_include": ["2022 年营收作为最终答案"]
  },
  "owner": "eval_team",
  "added_from": "online_incident_2024_05"
}
```

关键是让样例可复现、可评估、可追踪。

### 6.4 Golden set 和 regression set 的区别

Golden set 是高质量、稳定、代表核心能力的评估集。

Regression set 是从历史失败和关键风险中沉淀出来的防回归集合。

两者可以重叠，但目标不同。

```text
Golden set：衡量模型是否整体达标。
Regression set：检查已知关键问题是否复发。
```

面试中可以说：benchmark 看广度，golden set 看核心能力，regression suite 看历史问题和上线风险。

## 7. 版本对比

大模型评估经常要比较多个版本。

版本对比不能只比较总分。

### 7.1 需要记录哪些版本

至少记录：

1. 模型权重版本。
2. Tokenizer 版本。
3. System prompt 版本。
4. User prompt template 版本。
5. RAG 索引版本。
6. Retriever 和 reranker 版本。
7. Tool schema 版本。
8. 安全策略版本。
9. Judge 模型和 judge prompt 版本。
10. Eval code 版本。
11. Decoding 参数。
12. 数据集版本。

缺少版本记录，就很难解释分数变化。

### 7.2 Diff 分析

版本对比最重要的是看样本级 diff。

常见四象限：

```text
old correct, new correct：稳定正确
old wrong,   new correct：修复
old correct, new wrong：回归
old wrong,   new wrong：仍未解决
```

其中最需要关注的是：

1. old correct, new wrong：新引入回归。
2. old wrong, new correct：新版本修复了什么。
3. old wrong, new wrong：长期短板。

只看平均分会漏掉这些信息。

### 7.3 分层对比

版本对比应该按维度分层。

例如：

```text
总体：+2.0%
代码：+5.5%
数学：-1.8%
中文长上下文：-3.2%
安全拒答：+4.0%
JSON 格式：-6.0%
```

这种结果比单一总分更适合决策。

如果新版本要服务代码助手，代码提升可能值得上线。

如果新版本要服务结构化信息抽取，JSON 格式下降可能是阻断问题。

### 7.4 Pairwise diff 示例代码

下面是一个最小版本对比示例。

```python
from collections import Counter, defaultdict


def compare_versions(old_results, new_results):
    """old_results/new_results: dict[case_id] -> {correct: bool, tags: list[str]}"""
    buckets = Counter()
    regressions = []
    fixes = []
    tag_diff = defaultdict(Counter)

    for case_id, old in old_results.items():
        if case_id not in new_results:
            continue

        new = new_results[case_id]
        old_ok = bool(old["correct"])
        new_ok = bool(new["correct"])

        if old_ok and new_ok:
            bucket = "stable_correct"
        elif (not old_ok) and new_ok:
            bucket = "fixed"
            fixes.append(case_id)
        elif old_ok and (not new_ok):
            bucket = "regressed"
            regressions.append(case_id)
        else:
            bucket = "stable_wrong"

        buckets[bucket] += 1

        for tag in set(old.get("tags", [])) | set(new.get("tags", [])):
            tag_diff[tag][bucket] += 1

    return {
        "summary": dict(buckets),
        "regressions": regressions,
        "fixes": fixes,
        "by_tag": {tag: dict(counts) for tag, counts in tag_diff.items()},
    }
```

真实系统里，`correct` 可能来自 exact match、单元测试、human label、LLM judge 或业务指标。

关键是做样本级对齐，而不是只比较两个聚合分数。

## 8. 从错误分析到改进动作

Error analysis 的最终产物不是报告，而是改进决策。

### 8.1 数据改进

如果 root cause 是数据覆盖不足，可以：

1. 补充真实失败样例。
2. 构造 hard negative。
3. 增加边界条件样本。
4. 清洗冲突标注。
5. 做数据重采样或提高领域数据权重。
6. 增加多语言、多轮、长上下文样本。

注意不要把测试集直接泄露进训练集。

如果使用失败样例做训练，要保留独立 holdout 或新构造等价样例做验证。

### 8.2 Prompt 和系统改进

如果 root cause 是指令不清或系统约束不足，可以：

1. 明确输出 schema。
2. 增加拒答和不确定性规则。
3. 拆分复杂 prompt。
4. 增加 few-shot 示例。
5. 在服务端增加 validation 和 retry。
6. 调整 tool description。

Prompt 改动也要做 regression test。

一个 prompt 修好格式问题，可能同时让回答变啰嗦或过度拒答。

### 8.3 检索和工具链改进

RAG 或 Agent 错误常常不是模型本体问题。

可以改：

1. Chunking 策略。
2. Embedding 模型。
3. Retriever recall。
4. Reranker。
5. Context packing。
6. Citation 约束。
7. Tool schema。
8. 工具权限和错误处理。
9. Observation summarization。

面试中要展示系统视角：不是所有错误都靠继续训练解决。

### 8.4 训练和对齐改进

如果错误来自模型能力或偏好，可以考虑：

1. SFT 数据补强。
2. 偏好数据重构。
3. DPO 或 RLHF 目标调整。
4. Reward model 校准。
5. 拒答策略重平衡。
6. 多任务数据配比调整。

但训练改动成本较高，应该由 error analysis 支撑，而不是盲目扩大训练。

## 9. 自动化评估闭环

成熟团队会把 error analysis 和 regression testing 平台化。

### 9.1 最小闭环

一个最小闭环可以是：

```text
每次模型或 prompt 改动
  -> 跑核心 eval
  -> 跑 regression suite
  -> 生成版本 diff
  -> 输出新增回归样例
  -> 人工审核高风险 case
  -> 决定上线、灰度或回滚
```

### 9.2 报告应该包含什么

评估报告至少包含：

1. 总体指标变化。
2. 分层指标变化。
3. 统计不确定性。
4. 修复样例数量。
5. 回归样例数量。
6. 高严重度失败样例列表。
7. Top error clusters。
8. Root cause 假设。
9. 建议改进动作。
10. 是否通过上线门禁。

### 9.3 上线门禁

Regression suite 常用于上线门禁。

示例规则：

```text
上线条件：
- 主指标不下降超过 0.5%。
- 高严重度 regression 数量为 0。
- Safety regression 数量为 0。
- 核心业务 golden set 通过率不低于线上版本。
- JSON/schema 相关 case 通过率不低于 99%。
- 人工审核通过所有 P0/P1 case。
```

门禁规则不能只看总分。

高风险业务中，一个严重 regression 就可能阻止上线。

## 10. 真实项目中的坑

### 10.1 错误分类太粗

如果只有“正确”和“错误”，就无法指导修复。

但分类太细也不好，会导致标注不一致。

建议从 8 到 15 个一级错误类型开始，再为重点任务增加二级标签。

### 10.2 把 judge 的解释当成事实

LLM judge 可以辅助分析，但它也会误判。

尤其在数学、代码、安全、医疗、法律等任务中，高风险样例需要人工复核。

### 10.3 Regression suite 越积越脏

长期维护中，regression suite 可能出现：

1. 过时样例。
2. 期望答案不再正确。
3. 重复样例。
4. 过度针对旧系统的样例。
5. 标注标准不一致。

因此需要定期清理和版本化。

### 10.4 只加测试，不修 root cause

把失败样例加入 regression suite 只是防止复发。

如果不修 root cause，模型仍然会在同类新样例上失败。

### 10.5 用训练集污染 regression suite

如果把 regression suite 直接用于训练，然后再用同一套 suite 报告提升，评估结论会失真。

更好的做法是：

1. 历史失败样例可以用于训练修复。
2. 同时构造未见过的等价 holdout 样例。
3. 报告时区分 train-fix set 和 held-out regression set。

### 10.6 忽略线上新分布

Regression suite 主要覆盖已知问题。

它不能替代线上监控和新样本抽检。

新用户、新产品形态和新攻击方式会带来未知错误。

## 11. 优点、缺点和适用场景

### 11.1 Error analysis 的优点

1. 能把分数变化转成具体原因。
2. 能指导数据、训练、prompt 和系统改进。
3. 能发现局部退化和高风险问题。
4. 能帮助团队建立共享问题语言。
5. 能提高评估报告的可信度。

### 11.2 Error analysis 的局限

1. 人工成本高。
2. 错误 taxonomy 需要迭代。
3. Root cause 不一定容易验证。
4. 聚类和 LLM 分析可能引入偏差。
5. 很难覆盖未知未知问题。

### 11.3 Regression testing 的适用场景

特别适合：

1. 模型频繁迭代。
2. Prompt 频繁调整。
3. RAG 知识库持续更新。
4. Agent 工具链复杂。
5. 安全和合规要求高。
6. 企业客户场景稳定且高价值。

不适合作为唯一评估手段。

它必须和 broader benchmark、线上监控、人工抽检、新样本探索结合。

## 12. 常见改进方向

### 12.1 Active error mining

主动挖掘容易失败的样例。

例如：

1. 找低置信度样本。
2. 找 judge 分歧样本。
3. 找用户差评样本。
4. 找模型版本输出差异大的样本。
5. 找安全分类器边界样本。

这些样本通常比随机样本更有分析价值。

### 12.2 自动标签和人工复核结合

用 LLM 做初标，再让人工复核关键样例。

这样可以降低成本，同时保留质量。

### 12.3 从 case 到 capability benchmark

单个失败 case 修复后，可以扩展成一组能力测试。

例如，一个“年份约束检索失败”可以扩展成：

1. 不同公司。
2. 不同年份。
3. 同一年多个指标。
4. 跨年份比较。
5. 问题中有干扰年份。

这样可以测试模型是否真正掌握能力，而不是记住单个样例。

### 12.4 Eval-driven development

类似软件工程中的 test-driven development，大模型系统可以采用 eval-driven development。

即先明确失败样例、指标和上线门禁，再做模型或系统修改。

这样可以减少“改了很多但不知道是否更好”的情况。

## 13. 面试官会怎么问

### 问题 1：模型总体分数提升了，为什么还要做 error analysis？

回答要点：

1. 平均分会掩盖局部退化。
2. 不同错误业务成本不同。
3. Error analysis 能定位 root cause。
4. 能指导下一步数据、训练、prompt 或系统改进。
5. 能沉淀 regression suite 防止复发。

### 问题 2：你会如何设计一个 regression suite？

回答要点：

1. 从历史线上事故、高频失败、高风险场景和核心业务 case 中选样本。
2. 每个 case 要有明确输入、期望行为、错误类型、严重程度和验收标准。
3. 对 suite 做版本管理和定期清理。
4. 每次模型、prompt、RAG 或工具链更新都运行。
5. 作为上线门禁的一部分，但不替代完整 benchmark。

### 问题 3：RAG 系统答错了，你如何定位原因？

回答要点：

1. 先看检索是否召回正确文档。
2. 再看 reranker 是否把相关文档排在前面。
3. 检查上下文构造是否截断关键信息。
4. 如果上下文正确，再看生成是否忠实、引用是否正确。
5. 通过人工提供正确文档、替换检索器、替换模型等对照实验验证 root cause。

### 问题 4：如何发现新版本引入的 regression？

回答要点：

1. 在同一批样本上运行旧版本和新版本。
2. 固定 prompt、数据、judge、decoding 参数和评估脚本。
3. 做样本级 diff，重点看 old correct, new wrong。
4. 按任务、领域、难度、安全等级等维度分层。
5. 对高严重度 regression 做人工复核并决定是否阻断上线。

### 问题 5：失败样例能不能直接加入训练？

回答要点：

1. 可以用于修复，但要避免评估污染。
2. 加入训练前要确认标注和期望行为正确。
3. 应保留独立 holdout 或构造等价新样例验证泛化。
4. 报告结果时区分训练修复集和未见过的 regression set。

## 14. 标准回答模板

面试中可以这样回答：

```text
我不会只看总体分数。首先我会保存完整评估日志，包括输入、输出、参考答案、judge 解释、模型版本、prompt 版本、检索和工具调用记录。然后把失败样例按任务、能力维度、错误类型和严重程度做标注，例如 instruction following、factuality、reasoning、format、safety、retrieval miss、tool error 等。

接着我会对失败样例做聚类和分层统计，找出高频、高严重度、可修复的问题簇。对关键问题不会只停留在错误标签，而会通过对照实验做 root cause 分析，比如 RAG 场景会拆开检索、重排、上下文构造和生成，判断问题到底来自 retrieval、prompt、模型能力还是系统集成。

对于已经确认的重要失败样例，我会沉淀到 regression suite。每个 case 都要有 case id、输入、期望行为、错误类型、严重程度、验收标准和来源。之后每次模型、prompt、RAG 索引、工具 schema 或安全策略更新，都跑 regression suite，并和旧版本做样本级 diff，重点关注 old correct, new wrong 的回归样例。

最后我会把 error analysis 结果转成具体改进动作：数据补强、hard negative 构造、prompt/schema 修改、retriever/reranker 优化、工具链修复、训练或对齐目标调整。上线前不仅看主指标，还要看分层指标、严重 regression、护栏指标和人工复核结果。
```

## 15. 常见误区

### 15.1 只报告总体分数

总体分数不能说明局部能力变化，也不能指导修复。

### 15.2 错误 taxonomy 不连接修复动作

如果错误标签不能指导下一步行动，分类价值有限。

### 15.3 把所有失败都加入 regression suite

这样会让 suite 膨胀、噪声增加、维护成本变高。

### 15.4 不做样本级 diff

只比较聚合指标会漏掉新版本修好的样例和退化的样例。

### 15.5 不区分模型问题和系统问题

RAG、Agent、多工具系统中，很多错误来自检索、上下文、工具、权限、后处理或 UI，而不是模型权重。

### 15.6 用被训练过的样例证明模型变强

这会造成评估污染。

修复样例可以进训练，但最终结论要靠未见过的 holdout 或独立测试集支持。

## 16. 小练习

### 练习 1

一个客服模型总体满意度提升 1.5%，但投诉中出现更多“答非所问”。你会如何做 error analysis？

要求说明：日志字段、错误分类、分层维度和 root cause 验证方法。

### 练习 2

为一个企业知识库 RAG 系统设计 regression suite。

至少包含：样例来源、case 格式、验收标准、版本管理和上线门禁规则。

### 练习 3

两个模型版本在 1000 个样本上的总分分别是 83% 和 85%。你如何判断新版本是否有严重 regression？

要求说明：样本级 diff、四象限统计、分层分析和人工复核。

### 练习 4

一个 Agent 经常调用错工具。请设计错误 taxonomy 和 root cause 分析流程。

参考方向：planning、tool selection、tool arguments、observation reading、permission、final answer。

## 17. 本章总结

Error analysis 的核心是把评估结果从“一个分数”拆成“可定位、可修复、可复查的问题”。

平均分可能掩盖局部退化，不同错误的业务成本也不同。

好的错误分类体系应该连接修复动作，而不是只做标签统计。

失败样例聚类可以帮助发现高频问题簇，但 root cause 需要通过日志、对照实验和复现实验验证。

Regression suite 用来防止历史问题复发，特别适合作为模型、prompt、RAG、工具链和安全策略更新时的上线门禁。

版本对比要做样本级 diff，重点关注 old correct, new wrong 的回归样例，并按任务、领域、难度和风险分层。

面试中要强调：评估闭环不是“跑分”，而是发现问题、解释问题、修复问题、验证修复，并把关键问题沉淀为长期回归测试。
