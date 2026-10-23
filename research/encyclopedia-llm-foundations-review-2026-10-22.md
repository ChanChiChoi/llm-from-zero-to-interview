# 第四册大语言模型基础精修核验（2026-10-22）

## 当前范围

全文初读5.1—5.15与小结，局部修订概率目标、loss、PPL、采样及scaling。整章尚未验收。

## 来源读取

- [Chinchilla v1 HTML](https://arxiv.org/html/2203.15556v1)：读取参数化loss拟合与compute-optimal相关公式，L(N,D)=E+A/N^α+B/D^β，FLOPs≈6ND。将原正文独立叠加N/D/C三项的无来源公式换成上述形式，并说明dense近似和适用限制。未复现论文拟合或训练。
- [PyTorch2.12 CrossEntropyLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html)：读取class-index targets、ignore_index与有效目标平均相关说明；正文教学函数接收log probabilities，与真实CE的logits接口区分。
- [Python random](https://docs.python.org/3/library/random.html)：读取可单独实例化Random的说明；本地验证局部随机流不修改全局状态。文档为动态入口。

PPL反例为直接算术：参考目标为概率0.01时PPL100，均匀二类PPL2；预测分布的熵和参考目标负对数似然不是同一量。

## 代码与结构

两个标准库Python块独立编译执行通过：masked NLL=0.458145；temperature0.7下概率约[0.771,0.1848,0.0443]，seed7采样索引0。新增未归一化log probabilities、bool目标、全忽略目标的拒绝断言通过；[0,1,0]只选中索引1，调用前后全局random状态一致。

新增说明：有限且归一化的log probabilities、固定词表宽度、非空有效目标；采样函数反复传入同seed会重用分位点，不能替代持续推进的生成随机流。N=10^9、D=2×10^10时6ND=1.2×10^20FLOPs复算通过。

18个math块花括号/环境、围栏结构通过，宏和历史下标机械统一。git diff --check通过；未调用外部数学渲染器。

## 整章验收补充

接续复读剩余主题和小结，补ICL冲突分母非零及标注口径、greedy的确定性条件；修复极端温度差值溢出；限定记忆和泛化证据、涌现指标解释；成本分类不重叠并保留失败/重试支出，成功条件逐任务合取而非边际相乘；日志按权限与保留策略管理，不承诺有trace就能逐字复现。

新增读取范围：

- [Quantifying Memorization](https://arxiv.org/abs/2202.07646)摘要：模型容量、重复次数和提示上下文长度与逐字提取记忆的关系，跨家族外推更复杂；本次不把独特/常见文本单独作为普遍记忆判据。
- [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004)摘要：固定模型输出、改变非连续/非线性指标可产生看似涌现的曲线；不等于所有能力转变均被否定。

两项均为摘要阅读，不冒充全文或论文实验复现。通过Atom核对全部11篇题名/版本：Attention1706.03762v7、GPT3 2005.14165v4、Chinchilla2203.15556v1、InstructGPT2203.02155v1、CoT2201.11903v6、TruthfulQA2109.07958v2、FActScore2305.14251v2、SelfCheckGPT2303.08896v3、HELM2211.09110v2、Memorization2202.07646v3、Emergence2304.15004v2。未逐篇重读其他论文全文；NIST与OWASP入口可达不等于逐条标准合规审计。

最终临时检查器`/tmp/check_foundations_review.py`通过：两个标准库Python块独立编译执行；非法行/目标/ignore_index、变动词表、全忽略目标、非法logits/温度/概率检查；极端±1e308、温度1e308得到0.880797/0.119203，相等超大logits与极小温度得到0.5/0.5；零概率不被选中且全局random状态不变。两个相关布尔条件的联合率0.5不同于边际乘积0.25，单位成功成本20/80=0.25。

18个math块与围栏结构通过；本章相对链接、练习标题、下一章目标存在，本册目录仍指向当前文件；git diff --check通过。14个唯一外链最终全部HTTP200，首次沙箱脚本连代理失败，获准后重试通过。EXERCISES.md增加5项目标/分母反例，题库/术语已有相关主题，不复制摘要。

本章声明范围内精修验收完成；未运行真实模型、训练、tokenizer、生产采样、安全或运营系统，未调用外部公式渲染器，PDF仍受Pandoc缺失限制。教学数值不证明模型能力，动态页面不保证所有版本接口。全书与第二册遗留审计未完成。
