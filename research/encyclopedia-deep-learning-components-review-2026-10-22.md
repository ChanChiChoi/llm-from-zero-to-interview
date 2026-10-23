# 第四册深度学习组件精修核验（2026-10-22）

## 范围与结论

全文复读 `book-04-llm-encyclopedia/chapters/02-深度学习组件.md` 的 2.1—2.17 及小结，按百科定义、机制、反例和专题入口验收。重点修复 AMP 裁剪调用、PAD/EOS 共享 ID、shift 后有效目标、变长累积归一化、DDP no_sync、归一化统计和 checkpoint 状态语义。正文不承担完整训练或生产系统复现。

## 来源与读取边界

PyTorch 文档固定为 2.12，与本地 CPU 版本对应；文档版本号不是源码 commit。实际读取了以下页面的相关段落：

- [AMP examples](https://docs.pytorch.org/docs/2.12/notes/amp_examples.html)：unscale 后调用参数梯度裁剪，累积窗口内缩放保持一致。
- [BatchNorm1d](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BatchNorm1d.html)：前向有偏方差、running variance 无偏估计及 track_running_stats。
- [LayerNorm](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html)：normalized_shape、仿射参数和方差定义。
- [DDP](https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html)：no_sync 必须包含对应 forward。
- [Module](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html)：state_dict 为含引用的浅拷贝、load_state_dict strict 语义。
- [MultiheadAttention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html)：布尔 mask 的 True 禁止参加 attention；与 SDPA 的 True 允许语义相反，并用 CPU 实例对照。

正文 18 个唯一外链（8 个 arXiv、Xavier PMLR、9 个 PyTorch 文档）均返回 HTTP 200。通过 arXiv Atom 核对以下题名和版本：Residual 1512.03385v1、LayerNorm 1607.06450v1、RMSNorm 1910.07467v1、GLU 2002.05202v1、Kaiming 1502.01852v1、Adam 1412.6980v9、AdamW 1711.05101v3、Attention 1706.03762v7。元数据核对和入口可达不代表论文全文阅读或实验复现；Xavier 本轮仅验证入口。

## 本地验证

环境为 Python CPU 虚拟环境、PyTorch 2.12.0+cpu。临时检查器 `/tmp/check_components_review.py` 提取正文代码，各块独立编译、执行；该临时工具不是仓库长期测试设施。

- 8 个 Python 块全部通过；参数计数为 40/128，SiLU/门控为 -0.268941/-0.537883，残差方差为 1.026/1.032/1.128。
- CPU GradScaler 演示未缩放梯度范数 2、裁剪后参数 0.9；未启用 autocast，不代表 GPU 混合精度验证。
- CPU float64、foreach=False、fused=False 的 AdamW 首步为 0.9890000004999999，与公式差小于 1e-14。
- 转置 tensor 的 shape 为 (2,4,3)、stride 为 (12,1,4)，不连续。
- batch 原始有效 label 为 5，shift 后为 3，padding 为 1，pad ID 命中为 3；非法 mask、被监督 padding、全忽略目标均拒绝。
- 参数维度、非有限 sigmoid/max_norm、非法 checkpoint step 等边界通过。
- LayerNorm bias=1 时输出均值为 1；BatchNorm 输入 [1,3]、momentum=0.1 时 running_var 为 1.1。
- strict=False 仍拒绝同名 shape 不匹配；模型更新后普通 state_dict 引用跟随变化，deepcopy 快照不变。
- 同一布尔 mask 下 SDPA 输出 1，恒等投影 MultiheadAttention 输出 3。
- 两个 micro-batch 的 token 数 1/3、均值 1/3，直接平均为 2，全局 token 平均为 2.5。
- 34 个 math 块的围栏、花括号和环境结构通过；无旧式 operatorname 等禁用格式。相对链接目标存在，git diff --check 通过。

## 配套与限制

`EXERCISES.md` 新增五项接口和状态验收；`INTERVIEW_BANK.md` 同步 no_sync 和 token 加权边界。未运行真实训练、多进程 DDP、GPU、性能测试、数据恢复或生产 serving，未调用外部数学渲染器。PDF 构建仍受已知 Pandoc 缺失限制；本次不重复安装或扩展依赖。
