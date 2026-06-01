# 项目路线

## 当前状态

24 本主书正文第一版已完成，第三册已完成项目作品集和多模态实战。本文件用于把可写进简历的项目统一整理成路线、产出物和面试表达。第二轮精修时，本文件需要继续补齐 Agent Harness、工具协议生态、AI Infra 和推理框架方向的项目任务。

## 项目 1：miniGPT from scratch

目标：从零实现 decoder-only Transformer，并完成小语料自回归语言建模实验。

简历表达：基于 PyTorch 实现 miniGPT，包括 tokenizer、causal attention、Transformer block、训练循环、验证 loss、perplexity 和采样生成，并分析重复生成、过拟合和上下文长度限制。

## 项目 2：手写 LLM 核心组件

目标：实现 tokenizer、attention、causal mask、multi-head attention、RoPE、top-k、top-p、KV Cache 和 DPO/CLIP loss。

简历表达：手写并验证大模型关键训练与推理组件，能解释每个模块的输入输出 shape、复杂度和工程坑。

## 项目 3：LoRA/SFT 中文助手

目标：使用开源小模型完成中文 instruction tuning。

简历表达：构建中文 SFT 数据集，正确处理 chat template 与 assistant-only loss mask，使用 PEFT/LoRA 完成监督微调，并评估 base vs SFT 的指令遵循、格式合法率、人工偏好和 bad case。

## 项目 4：DPO 偏好优化实验

目标：构造 chosen/rejected preference pair，使用 TRL/LoRA 完成 DPO 偏好优化。

简历表达：基于中文 SFT 助手构造偏好数据，覆盖事实性、完整性、格式遵循、安全拒答和拒答边界，比较 SFT 与 DPO 的 pairwise 胜率、输出长度、安全拒答和通用能力回归。

## 项目 5：高性能推理服务

目标：部署开源 LLM，支持 Chat API、SSE 流式输出、限流、监控和压测。

简历表达：构建大模型推理服务，测量 TTFT、P95/P99、tokens/s、QPS、显存和错误率，对比 Transformers generate、vLLM、量化和不同并发下的性能差异。

## 项目 6：企业知识库 RAG 系统

目标：完成文档解析、清洗、chunk、embedding、混合召回、rerank、引用生成、权限过滤和评估。

简历表达：构建可评估的企业知识库问答系统，使用 Recall@K、MRR、groundedness、citation accuracy 和人工评测分析检索与生成质量，并分类复盘解析、切分、召回、重排、生成和权限类 bad cases。

## 项目 7：数学推理增强实验

目标：构造数学推理评测集，比较 zero-shot、few-shot、CoT、self-consistency 和工具校验。

简历表达：实现答案抽取与标准化、self-consistency 投票、Python 表达式校验和错误类型分析，按题型统计准确率、成本和 bad cases。

## 项目 8：CLIP 图文检索

目标：使用 CLIP 实现 text-to-image 和 image-to-text 检索，并手写 mini CLIP loss。

简历表达：使用 image/text encoder 生成归一化 embedding，通过 cosine similarity 和 FAISS 实现图文检索，使用 Recall@K、MRR、Top-1 Accuracy 评估，并分析中文 prompt、细粒度识别和 OCR 场景局限。

## 项目 9：开源 VLM 图片问答

目标：使用 LLaVA、Qwen-VL、InternVL 或 MiniCPM-V 跑通图片输入、文本问题和多模态回答流程。

简历表达：构造包含描述、OCR、图表、截图、计数和空间关系的小型 VQA 评测集，分析视觉幻觉、OCR 错误、图片分辨率和 prompt 对结果的影响。

## 项目 10：多模态 Instruction Tuning 数据集

目标：构造 image-text conversation 数据，理解 image token、chat template 和 label mask。

简历表达：设计多模态 SFT 样本格式，覆盖单图问答、多轮对话、OCR、图表理解和拒答边界，并说明 assistant-only loss mask 和图片占位符一致性。

## 项目 11：图像 + 文本 RAG

目标：把 OCR、图片描述、文本检索、图像检索和 LLM 生成结合起来。

简历表达：构建多模态 RAG demo，将图片 OCR、caption、metadata 与文本知识库统一索引，支持图文混合检索和基于证据的答案生成。

## 项目 12：语音转文本再问答 Demo

目标：串联 Whisper ASR、LLM 问答和文本/TTS 输出。

简历表达：实现语音输入到文本转写、问题理解、知识库问答和结果输出的端到端 demo，并分析 ASR 错误如何影响下游问答。

## 项目 13：大模型评估平台 Mini Version

目标：管理 prompt set、模型版本、自动评估、人工评估、LLM-as-Judge 和 bad case 回归。

简历表达：构建可复现的大模型评估和回归测试工具，支持版本对比、pairwise 评测、硬门禁、评测报告和 bad case 管理。

## 项目 14：Coding Agent Harness Mini Version

目标：实现一个最小 coding agent runtime，支持任务输入、文件读写、命令执行、权限检查、trace 记录和 replay。

简历表达：设计并实现 coding agent harness，包含 planner、tool executor、workspace patch、shell sandbox、permission gate、trace/replay 和失败复盘机制，能够分析 Claude Code、OpenCode、Codex 类系统的工程取舍。

## 项目 15：工具协议与 MCP Server Demo

目标：实现一个最小工具协议 demo，覆盖 tool schema、参数校验、权限边界、MCP server 接入和工具调用评估。

简历表达：构建企业工具调用 demo，支持结构化 function calling、工具注册、参数 schema 校验、权限控制、调用 trace 和错误处理，并设计工具选择、参数正确率、执行成功率和安全性的评估表。

## 项目 16：AI Infra 可观测性与成本 Dashboard

目标：模拟训练和推理平台的核心指标，构建可观测性、容量规划和成本分析 dashboard。

简历表达：设计 AI Infra dashboard，覆盖 GPU 利用率、MFU/HFU、队列等待、训练故障、TTFT、TPOT、tokens/s、KV Cache、P95/P99、错误率和单位 token 成本，并能用指标定位训练和推理瓶颈。

## 项目 17：Mini LLM Serving Engine

目标：从 naive generate loop 升级到支持 continuous batching、KV Cache 管理、prefix cache、流式输出和简单调度的教学版 serving engine。

简历表达：实现 mini LLM serving engine，包含 request queue、prefill/decode 分离、token budget 调度、KV block 管理、流式返回和压测脚本，并对比 naive generate、vLLM/SGLang 思路下的延迟、吞吐和显存 trade-off。
