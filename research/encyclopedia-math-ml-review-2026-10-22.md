# 第四册数学与机器学习百科精修证据

- 日期：2026-10-22；范围：`book-04-llm-encyclopedia/chapters/01-数学与机器学习.md` 全文、简介/目录定位与专题入口。
- 按百科类型验收定义、条件、关系、反例和深入入口，不用术语条目替代第十三册完整推导。
- 证据等级：official、local-demo、教材；未运行模型训练、GPU或生产实验。

## 来源与实际读取范围

| 来源 | 实际核验与限制 |
| --- | --- |
| [Deep Learning 概率章](https://www.deeplearningbook.org/contents/prob.html) | 读取条件概率、非零条件事件、链式分解等相关段落；PDF转HTML文本按连续字符抽取，未逐页复核整本教材 |
| [PyTorch 2.12 Dropout](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Dropout.html) | 读取训练时1/(1-p)缩放、eval恒等映射；stable入口先返回HTML跳转页，因此改用明确版本正文 |
| [NIST 均值置信区间](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) | 读取t区间及重复抽样覆盖含义；不将均值区间迁移为所有指标通用方法 |
| [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | 读取裁剪代理目标、正负优势与仍可能发生过大策略变化的提示；未运行PPO |
| [scikit-learn指标指南](https://scikit-learn.org/stable/modules/model_evaluation.html) | 读取指标组织、二分类排序和precision/recall相关材料；未执行该库评估器 |
| [SciPy FDR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html) | 读取BH对独立或正回归依赖的要求以及BY的区别；未运行多重检验实验 |

最终11个正文唯一外部入口均HTTP200。其余入口包括Deep Learning首页、ProbML、Sutton/Barto教材页、PyTorch目录和label smoothing论文，本轮主要检查可达性，不能写成完整原文核验。额外尝试Penn State Bayes页面时出现TLS证书失败，候选statproofbook路径返回404，均未作为正文依据；概率关系改由已成功读取教材和本地数值推导支撑。未关闭TLS验证。

## 修订内容

1. 特征向量非零与方阵条件，负特征值反向及零值；SVD误差平方/能量比与任务能力分开。
2. 条件概率非零分母、离散KL支持集/无穷边界；困惑度绑定目标似然而非分布熵。
3. 似然支持方向用似然比，不保证检查通过就提高后验；修正Bayes推导和MAP适用边界。
4. Dropout训练缩放与推理恒等、warmup无固定比例、global batch的等长worker前提、测试集独立性不自动等于无偏。
5. ROC-AUC平局、precision非单调、F1直接计数式及零分母；PPO代理目标不冒充硬KL约束。
6. 标准误的独立性与n>1、频率区间含义、配对/聚类bootstrap和稀有事件退化、McNemar二元结果而非二类任务限制、BH依赖条件。
7. 交叉熵先中心化避免大数消减；mask先排除无监督位置，LoRA参数比不冒充全模型压缩，单位成功成本包含失败开销且统一请求/任务分母。

## 本地验证

- Python 3.12.9下从正文提取唯一标准库Python块，语法编译与独立执行通过。weak_loss=0.939831、strong_loss=0.229358，整体加100后loss差为0；三个1e20 logits得到1.098612，而非错误的0。
- 空输入、越界/布尔target、NaN/Inf/bool logits拒绝；单类别loss=0，±1e20同值三分类loss=log(3)，正负1000极端正确/错误目标loss=0/2000。
- 贝叶斯三组教学输入后验为0.875、0.7、0.6747；先验固定0.7，通过本身不保证后验提高。
- 熵/PPL反例：`[.99,.01]`熵0.0560015、第二类目标PPL100；均匀分布熵0.693147、PPL2。
- LoRA参数131072、全矩阵16777216、比值128；token平均loss=6/11；20元/80次成功=0.25元，均复算通过。diag(3,1)截断误差为1、保留能量0.9。
- 6个math围栏与总围栏配对、花括号/环境检查通过；无operatorname、尖括号下标、text下划线或旧式美元公式块。113个二级标题含索引/方法说明，不能当作113个独立技术专题。
- 本章所有相对链接目标存在，第十三册与第五册入口有效；并未因此将链接目标正文标为本轮已精修。git diff --check通过。

EXERCISES.md新增5项反例验收；INTERVIEW_BANK.md第62题同步PPO边界；术语表已有相关名词，未重复添加。未运行外部公式渲染器或PDF构建，后者仍有Pandoc缺失限制。本章结果只支持定义修订、所读来源和局部数值行为。
