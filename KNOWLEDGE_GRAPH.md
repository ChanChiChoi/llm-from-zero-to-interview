# 大模型知识图谱

## 核心依赖链

### 语言模型链

Tokenization -> Embedding -> Transformer -> Next-Token Prediction -> Cross Entropy -> Pretraining -> SFT -> RLHF/DPO -> Chat Model

### 推理优化链

Self-Attention -> Causal Mask -> KV Cache -> Prefill/Decode -> Continuous Batching -> Quantization -> Speculative Decoding -> Serving System

### 对齐链

Instruction Data -> SFT -> Preference Data -> Reward Model -> RLHF -> DPO -> Safety Training -> Model Behavior Evaluation

### 多模态链

Vision Encoder -> CLIP -> VLM Connector -> Multimodal Instruction Tuning -> Multimodal Reasoning -> Multimodal Safety

### 评估链

Benchmark -> Metrics -> Human Eval -> LLM-as-a-Judge -> Contamination Detection -> Error Analysis -> Regression Test

### 研究链

Paper Reading -> Hypothesis -> Baseline -> Experiment -> Ablation -> Error Analysis -> Reproduction Report -> New Research Idea
