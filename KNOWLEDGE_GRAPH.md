# 大模型知识图谱

## 核心依赖链

### 语言模型链

Tokenization -> Embedding -> Transformer -> Next-Token Prediction -> Cross Entropy -> Pretraining -> SFT -> RLHF/DPO -> Chat Model

### 推理优化链

Self-Attention -> Causal Mask -> KV Cache -> Prefill/Decode -> Continuous Batching -> Quantization -> Speculative Decoding -> Serving System

### 数学基础链

Linear Algebra -> Vector -> Dot Product -> Vector Norm -> Cosine Similarity -> Matrix Multiplication -> Projection -> Eigenvalue -> Matrix Rank -> Singular Value Decomposition -> Low-Rank Approximation -> PCA -> Low-Rank Compression -> LoRA -> QLoRA -> Attention Shape -> Embedding Retrieval

Probability Theory -> Random Variable -> Probability Distribution -> Conditional Probability -> Chain Rule of Probability -> Maximum Likelihood Estimation -> Negative Log-Likelihood -> Perplexity -> Sampling -> Calibration -> Hallucination

Information Theory -> Information Content -> Entropy -> Cross Entropy -> KL Divergence -> Perplexity -> Mutual Information -> RAG Evidence Quality -> KL Penalty -> DPO Reference Model

Optimization Basics -> Gradient -> Gradient Descent -> SGD -> Momentum -> Adam -> AdamW -> Learning Rate Schedule -> Warmup -> Cosine Decay -> Gradient Clipping -> Global Batch Size -> Hessian -> Loss Spike Debugging -> Training Stability

Statistical Learning -> True Risk -> Empirical Risk -> ERM -> Generalization Gap -> Bias-Variance Trade-off -> Overfitting -> Underfitting -> Regularization -> Early Stopping -> Distribution Shift -> Data Leakage -> Benchmark Contamination -> Generalization Audit

Bayesian Thinking -> Prior -> Likelihood -> Evidence -> Posterior -> Bayesian Update -> MAP -> Aleatoric Uncertainty -> Epistemic Uncertainty -> Confidence -> Calibration -> ECE -> Brier Score -> Selective Prediction -> Abstention -> Human Review

Reinforcement Learning Math -> MDP -> State -> Action -> Policy -> Reward -> Return -> Value Function -> Q Function -> Advantage -> Policy Gradient -> PPO Ratio -> Clipped Surrogate Objective -> RLHF KL Penalty -> Reward Model Pairwise Loss -> DPO Loss -> Reward Hacking Audit

Evaluation Statistics -> Sample Mean -> Sample Variance -> Standard Error -> Confidence Interval -> Hypothesis Test -> p-value -> Paired Evaluation -> Bootstrap -> McNemar Test -> Sample Size -> Statistical Power -> Multiple Comparisons -> Bonferroni Correction -> Benjamini-Hochberg -> Experiment Gate

Math Interview Readiness -> Formula Coverage -> Formula Accuracy -> Intuition Clarity -> LLM Scenario Mapping -> Caveat Coverage -> Demo Coverage -> Weak Question -> Revision Plan

### PyTorch 工程链

PyTorch Tensor -> Tensor Shape -> Dtype -> Device -> Broadcasting -> Tensor Stride -> Contiguous Tensor -> View vs Reshape -> Matmul -> Einsum -> Tensor Mask -> LM Loss Flatten -> Tensor Shape Audit -> Autograd -> Dynamic Computation Graph -> Requires Grad -> Leaf Tensor -> Backward -> Vector-Jacobian Product -> Grad Accumulation -> Zero Grad -> Detach -> No Grad -> In-place Operation Risk -> Autograd Audit -> nn.Module -> Parameter -> Module Registration -> Submodule -> ModuleList -> ModuleDict -> Buffer -> state_dict -> load_state_dict -> train/eval Mode -> Module Hook -> Module Audit -> Dataset/DataLoader -> Map-style Dataset -> IterableDataset -> Collate Function -> Dynamic Padding -> Ignore Index -> Sampler -> Batch Sampler -> DistributedSampler -> DataLoader Worker -> Pin Memory -> Padding Waste -> Data Pipeline Audit -> Training Loop -> Training Step -> Raw Loss -> Scaled Loss -> Optimizer Step -> Scheduler Step -> Evaluation Loop -> Checkpoint Resume -> Non-Finite Loss -> Training Loop Audit -> Mixed Precision -> FP16 -> BF16 -> Autocast -> GradScaler -> Loss Scaling -> Activation Memory -> Activation Checkpointing -> CUDA Memory Stats -> OOM Audit -> Distributed Training -> Rank/World Size -> Process Group -> DDP -> All-Reduce -> Gradient Synchronization -> Global Batch Size -> DDP no_sync -> FSDP -> Distributed Training Audit -> Debug/Profiling -> Debug Tensor Metadata -> Finite Check -> Gradient Debug -> Forward Hook Debug -> Anomaly Detection -> CUDA Synchronize Timing -> torch.profiler -> DataLoader Bottleneck -> Debug/Profiling Audit -> Transformer Components -> Token Embedding -> LM Head Weight Tying -> RMSNorm -> Causal Mask -> Scaled Dot-Product Attention -> Multi-Head Self-Attention -> SwiGLU MLP -> Pre-Norm Decoder Block -> RoPE -> KV Cache -> Transformer Component Audit -> PyTorch Engineering Interview Readiness -> Engineering Interview Gate -> Revision Plan

### 对齐链

Human Intent -> Safety Policy -> Proxy Objective -> Instruction Data -> SFT -> Preference Data -> Reward Model -> RLHF/DPO -> Reward Hacking Audit -> Scalable Oversight -> AI Feedback -> Verifier -> Human Audit -> Outer Alignment Check -> Inner Alignment Check -> Goal Misgeneralization Eval -> Model Behavior Evaluation -> Alignment Gate

### 安全治理链

AI Safety -> Alignment -> Alignment Problem -> HHH -> Risk Taxonomy -> Safety Policy -> Instruction Hierarchy -> Untrusted Content Boundary -> Jailbreak Eval -> Prompt Injection Eval -> Prompt Injection Gate -> Reward Hacking Gate -> Scalable Oversight -> Oversight Gate -> Safety Evaluation -> Red Teaming -> Capability Elicitation -> Dangerous Capability Eval -> Red Team Regression Suite -> Red Team Gate -> Mechanistic Interpretability -> Activation Patching -> Sparse Autoencoder -> Interpretability Gate -> Representation Engineering -> Steering Vector -> Activation Steering -> Steering Gate -> Model Editing -> Knowledge Editing -> Editing Gate -> Machine Unlearning -> Forget Set -> Retain Set -> Unlearning Gate -> Data Privacy -> Memorization -> Training Data Extraction Eval -> Membership Inference -> PII Leakage Eval -> Privacy Gate -> Watermarking -> Watermark Detection -> Content Credentials -> Watermark Gate -> Model Card -> System Card -> Risk Disclosure -> Responsible Scaling -> Model Governance -> Governance Gate -> Safety Gate -> Release Gate -> Guardrail -> Tool Safety -> Audit Log -> Incident Response -> Safety Interview Readiness -> Safety Interview Rubric

### 多模态链

Multimodal Data -> Image-Text Alignment -> OCR/ASR/Video Temporal Alignment -> Vision Encoder -> Patch Embedding -> CLS/Patch Tokens -> Vision Encoder Shape Audit -> Visual Token Budget -> Multimodal Context Budget -> CLIP -> CLIP Loss -> Zero-Shot Classification -> Image-Text Retrieval -> VLM Connector -> VLM Connector Audit -> Image Placeholder -> Visual Token Compression -> Assistant-Only Multimodal Loss -> Chat Template -> Multimodal SFT Data Audit -> Multimodal Task Coverage -> Evidence Support Rate -> Missing Refusal Rate -> Multimodal Instruction Tuning -> Diffusion Model -> DDPM -> Forward Diffusion -> Reverse Denoising -> Noise Scheduler -> Noise Prediction Loss -> Classifier-Free Guidance -> Latent Diffusion -> Stable Diffusion -> VAE Compression Ratio -> Text-to-Image Pipeline Audit -> Negative Prompt -> ControlNet -> Image-to-Image -> Inpainting -> DALL-E -> Autoregressive Image Tokens -> Video Generation -> Spatiotemporal Patch -> Video Diffusion -> Temporal Consistency -> Identity Drift -> Flickering -> World Model -> Physics Consistency -> Video Generation Evaluation -> FVD -> Video Token Audit -> Audio Generation -> Waveform -> Log-Mel Spectrogram -> ASR -> Whisper -> WER -> CER -> TTS -> Vocoder -> Voice Cloning -> Audio Codec -> Speech Token -> Codec Language Model -> Speech-to-Speech -> VAD -> MOS -> Audio Token Audit -> Unified Multimodal Model -> Unified Tokenization -> Any-to-Any Multimodal Model -> Early-Fusion Multimodal Transformer -> Center-LLM Multimodal System -> Multimodal Router -> Multimodal Loss Mixture -> Modality Conflict -> Unified Multimodal Audit -> Multimodal Evaluation -> VQA Accuracy -> Chart Relaxed Accuracy -> Grounding IoU -> Multimodal Hallucination Rate -> Multimodal Prompt Injection -> Biometric Identity Safety -> Content Provenance -> Multimodal Safety Audit -> Multimodal Reasoning -> Multimodal Cost Audit -> Multimodal Safety

### 评估链

Benchmark -> Metrics -> Human Eval -> LLM-as-a-Judge -> Contamination Detection -> Error Analysis -> Regression Test

### 数据工程链

Source Registry -> Web-Scale Collection -> Parsing -> Data Cleaning -> Quality Scoring -> Exact Deduplication -> Near Deduplication -> Code/Math/Domain Audit -> Synthetic/Distillation Audit -> Preference/Safety Audit -> Multimodal Data Audit -> PII/Secret Filtering -> Safety Filtering -> Train-Eval Overlap -> Contamination Detection -> Data Attribution -> Data Valuation -> Data Mixture -> Data Sampling -> Dataset Versioning -> Data Lineage -> Data Governance -> Data Interview Readiness -> Pretraining

### 研究链

Paper Reading -> Hypothesis -> Baseline -> Experiment -> Ablation -> Error Analysis -> Reproduction Report -> New Research Idea
