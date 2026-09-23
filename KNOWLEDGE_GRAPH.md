# 大模型知识图谱

## 核心依赖链

### 语言模型链

Tokenization -> Special Token -> Chat Template -> Assistant-Only Label Mask -> EOS Coverage -> Embedding -> Transformer -> Transformer Backbone -> Residual Stream -> Norm Placement -> Pre-LN -> Post-LN -> RMSNorm -> Residual Scaling -> Early Update Pressure -> Normalization Stability Gate -> Position Encoding -> Sinusoidal Position Encoding -> Learned Absolute Position Embedding -> Relative Position Representation -> Relative Position Bias -> RoPE -> ALiBi -> Position Interpolation -> RoPE Scaling -> Position ID Audit -> Position Encoding Gate -> Attention Mask -> Loss Mask -> Bidirectional Attention -> Prefix LM -> Packed Sequence Mask -> Block-Diagonal Causal Mask -> Sliding Window Attention -> Mask Visibility Audit -> Mask Gate -> Parallel Path Length -> Attention Pair Count -> Quadratic Attention Cost -> Attention Score Memory -> Prefill Attention Cost -> Decode Attention Cost -> Attention Cost Gate -> Training FLOPs Estimate -> Compute-Optimal Training -> Tokens per Parameter -> Undertrained Model -> Serving Pressure -> Scaling Gate -> Position-wise FFN -> SwiGLU -> Gated MLP -> MoE FFN -> Top-k Routing -> Expert Capacity -> Active Parameters -> FFN / MoE Gate -> Architecture Selection Gate -> In-Context Learning -> Label Space Coverage -> Demonstration Format Consistency -> Context Evidence Use -> Middle Evidence Risk -> Token Retrieval Audit -> Retrieval Label Match -> ICL Conflict Rate -> ICL Gate -> Route Score -> Training Parallelism -> Content Routing -> Scaling Evidence -> Ecosystem Readiness -> Deployment Pressure -> Attention-SSM Hybrid Readiness -> Next-Token Prediction -> Cross Entropy -> Pretraining -> SFT -> RLHF/DPO -> Chat Model

### 推理优化链

Self-Attention -> Attention Routing Audit -> Route Hit Rate -> Future Leak Rate -> Softmax Row Sum Check -> Scale Stability -> Interpretation Boundary -> QK Circuit -> OV Circuit -> Head Subspace Audit -> Head Redundancy Rate -> KV Head Ratio -> Multi-head Latent Attention -> Latent KV Cache -> KV Cache Saving Rate -> Attention Architecture Gate -> Causal Mask -> KV Cache -> KV Cache Budget -> KV Cache Per Token -> Decode Bandwidth Pressure -> Paged KV Block -> KV Cache Gate -> Prefill/Decode -> Continuous Batching -> Request Scheduling -> TTFT Regression -> TPOT Regression -> KV Pressure -> Prompt Cost Drift -> Cache Effectiveness -> Serving Gate -> Quantization -> Speculative Decoding -> Serving System

### Reasoning 系统链

Reasoning Model -> Reasoning Candidate Set -> Chain-of-Thought -> Few-shot CoT -> Zero-shot CoT -> Scratchpad -> Hidden CoT -> Visible Explanation -> CoT Faithfulness -> CoT Step Accuracy -> CoT Regression -> CoT Routing -> CoT Audit -> Self-Consistency -> Sampling Temperature -> Top-p Sampling -> Answer Normalization -> Majority Vote -> Weighted Vote -> Candidate Diversity -> Majority Failure -> Self-Consistency Cost -> Self-Consistency Accuracy -> Pass@k -> Verifier -> Programmatic Verifier -> Hybrid Verifier -> Verifier Reranking -> Pairwise Accuracy -> Hard Negative -> Verifier Calibration -> Reward Model Bias -> Process Supervision -> Step Label -> First-Error Detection -> Process Reward Model -> Process Search Pruning -> Process Supervision Audit -> Search Reasoning -> Search State -> Search Action -> Beam Search -> Best-First Search -> Tree-of-Thought -> UCT -> MCTS -> Prune False Negative -> Search Budget -> Search Audit -> Test-Time Compute Scaling -> Compute Budget Vector -> Adaptive Compute -> Budget Router -> Cost per Correct -> Marginal Accuracy per Cost -> P95 Latency -> Wasted High Compute -> TTC Audit -> Math Reasoning Training -> Math Training Sample -> Answer Supervision -> Synthetic Math Data -> Math Curriculum -> Math Contamination Audit -> Template Diversity -> Math Training Gate -> Code Reasoning -> Execution Feedback -> Unit Test Verifier -> Public Test -> Hidden Test -> Public-Hidden Gap -> Self-Debug -> Repair Success Rate -> Sandbox Violation Rate -> Code Execution Audit -> Code Reasoning Gate -> Reasoning Evaluation -> Evaluation Sample -> Variant Evaluation -> Robustness Drop -> Paired Lift -> Bootstrap Confidence Interval -> Reasoning Eval Gate -> Reasoning Safety -> Pseudo Reasoning -> Overconfident Error -> Hidden CoT Exposure -> Tool Misuse -> Human Review Coverage -> Severity-Weighted Risk -> Reasoning Safety Gate -> Outcome Reward Model -> Process Step Accuracy -> Test-Time Compute -> Test-Time Compute Cost -> Reasoning Audit -> Reasoning Gate -> Reasoning Interview Readiness -> Reasoning Interview Rubric -> Reasoning Formula Coverage -> Reasoning Demo Coverage -> Weak Reasoning Question -> Reasoning Revision Plan

### Agent 系统链

Agent -> Agent Harness -> Agent Runtime -> Session Manager -> Task Manager -> Context Builder -> Model Adapter -> Action Parser -> State Store -> Error Handler -> Runtime Gate -> Goal -> Agent Loop -> Harness Loop Audit -> Agent State -> Agent Action -> Tool Registry -> Tool Registry Completeness -> Tool Schema Strictness -> Tool Permission Binding -> Side Effect Level -> Risk Protection Coverage -> Output Normalization Coverage -> Tool Version Coverage -> Tool Trace Readiness -> Untrusted Tool Output Boundary -> Tool Registry Gate -> Tool Executor -> Observation -> State Update -> Controller -> Agent Budget -> Permission Gate -> Replay Readiness -> Harness Gate -> Agent Trace -> Trace Schema Completeness -> Span Schema Completeness -> Span Tree Validity -> Timeline Validity -> Artifact Reference Coverage -> Version Capture Coverage -> Replay Readiness Rate -> Privacy Masking Coverage -> Error Attribution Coverage -> Final Status Consistency -> Metric Export Coverage -> Eval Export Coverage -> Trace Replay Gate -> Agent Incident -> Plan Feasibility Rate -> Task Success Rate -> Tool Selection Accuracy -> Argument Validity -> Tool Execution Success Rate -> Observation Use Rate -> State Update Coverage -> False Completion Rate -> Budget Overrun Rate -> Unauthorized Action Rate -> Tool Result Injection Block Rate -> Stop Correctness -> Agent Incident Gate -> Agent Gate -> Tool Use -> Tool Schema -> JSON Schema -> Required Field -> Additional Properties -> Enum Constraint -> Pattern Constraint -> Range Constraint -> Business Validation -> Schema Repair -> Schema Gate -> Tool Call -> Tool Choice -> Tool Choice Policy -> Candidate Tool Set -> Auto Tool Choice -> None Tool Choice -> Required Tool Choice -> Forced Tool Choice -> Allowed Tools -> Tool Choice Mode Accuracy -> No-Tool Clarification Block Rate -> Forced Missing Argument Block Rate -> Parallel Safety Rate -> Rate Limit Pass Rate -> Confirmation Enforcement Rate -> Cost Budget Pass Rate -> Loop Control Rate -> Tool Choice Gate -> Structured Function Calling -> JSON Mode -> Structured Outputs -> Function Calling -> Function Calling Protocol -> Tool Call ID -> Tool Result -> Finish Reason -> Tool Loop -> Parallel Tool Calls -> Streaming Tool Call Delta -> Tool Call Trace -> Protocol Chain Validity -> Tool Result ID Match Rate -> Finish Reason Consistency -> Streaming Safety Rate -> Parallel Alignment Rate -> Idempotency Protection Rate -> Schema Valid Rate -> Required Field Pass Rate -> Type Valid Rate -> Enum Valid Rate -> Pattern Valid Rate -> Range Valid Rate -> Additional Properties Block Rate -> Business Rule Pass Rate -> Schema Repair Success Rate -> Tool Selection Accuracy -> Argument Exact Match -> Argument Validity -> Execution Success Rate -> Error Recovery Rate -> Unnecessary Tool Rate -> Unauthorized Attempt Rate -> Unauthorized Attempt Block Rate -> Tool Result Injection -> Tool Result Injection Rate -> Tool Calling Gate -> ReAct -> Plan-Act-Observe -> ReAct Trace -> Plan Update -> Thought-Action Alignment -> Action Accuracy -> Plan Adherence Rate -> Plan Update Coverage -> Parse Failure Rate -> Repeat Action Rate -> Premature Final Rate -> Blocked Recovery Rate -> ReAct Gate -> Planning -> Task Decomposition -> Subgoal -> Acceptance Criteria -> Dependency Graph -> Topological Order -> Dynamic Replanning -> Critical Path -> Goal Coverage -> Dependency Violation Rate -> Risk Confirmation Coverage -> Planning Gate -> Memory -> Short-Term Memory -> Long-Term Memory -> Episodic Memory -> Semantic Memory -> Procedural Memory -> Preference Memory -> Memory Write Gate -> Memory Retrieval Score -> Memory Permission Gate -> Memory Decay -> Memory Conflict -> Memory Pollution -> Stale Memory Use Rate -> Unauthorized Memory Retrieval Rate -> Unsafe Write Block Rate -> Memory Gate -> Agentic RAG -> Active Retrieval -> Retrieval Controller -> Multi-Round Retrieval -> Evidence State -> Evidence Gap -> Query Drift -> New Evidence Gain -> Context Precision -> Evidence Support Rate -> Citation Accuracy -> Agentic RAG Gate -> Code Agent -> Coding Agent Workflow -> Task Type Classification -> Repository Exploration Coverage -> Plan Coverage -> Workflow Validation Coverage -> Feedback Use Rate -> Repository Understanding -> File System Tool -> Workspace Root -> Path Containment -> Read File Tool -> Search Tool -> Apply Patch Tool -> Patch Context Match -> Secret File Block Rate -> Read Truncation Clarity -> Concurrent Edit Protection -> Unrelated Diff Rate -> Diff Faithfulness -> Rollback Readiness -> File Edit Gate -> Terminal Execution -> Bash Tool -> Test Runner Safety -> Command Risk Level -> Command Allowlist -> Command Denylist -> Command Auto Execution Precision -> Dangerous Command Block Rate -> Command Confirmation Coverage -> Network Access Control -> Timeout Cancellation Coverage -> Command Output Truncation -> Secret Masking Coverage -> Command Trace -> Command Safety Gate -> Context Management -> Context Candidate -> Context Budget Utilization -> Key Context Recall -> Constraint Retention -> Task State Compression -> Summary Faithfulness -> Stale Summary Rate -> Tool Output Compression Fidelity -> Current Diff Coverage -> Trust Boundary Coverage -> Memory Write Safety -> Context Builder Gate -> Patch Generation -> Patch Localization -> Minimal Patch -> Test Execution Feedback -> User Change Protection -> Code Sandbox -> Coding Agent Workflow Gate -> Code Agent Gate -> Claude Code Architecture -> Terminal Coding Agent -> CLI Control Command -> Architecture Evidence Coverage -> Required Module Coverage -> Access Governance Coverage -> Core Loop Governance Coverage -> Permission Control Coverage -> State Recovery Coverage -> Extension Governance Coverage -> Observability Coverage -> Architecture Eval Readiness -> High-Risk Surface Governance -> Claude Code Architecture Gate -> OpenCode Architecture -> Open Runtime Control Plane -> Open Runtime Module Coverage -> Config Governance Coverage -> Agent Permission Isolation -> Tool Permission Binding Coverage -> MCP Tool Namespace Coverage -> Custom Tool Schema Coverage -> Server API Governance -> Snapshot Recovery Coverage -> Provider Adapter Coverage -> OpenCode Architecture Gate -> Coding Agent Comparison -> Product Surface Coverage -> Context Source Coverage -> Tool Execution Coverage -> Permission Governance Coverage -> Edit Recovery Coverage -> Agent Eval Readiness -> Enterprise Governance Coverage -> Cross-Agent Risk Governance -> Coding Agent Comparison Gate -> MCP Background Audit -> Direct Integration Count -> MCP Integration Count -> Integration Reduction -> Capability Model Coverage -> Context Object Coverage -> Discovery Standardization -> MCP Schema Contract Coverage -> Host Server Boundary Clarity -> Local Context Control -> Cross Client Reuse -> Governance Boundary Clarity -> MCP Trace Eval Readiness -> MCP A2A Distinction -> MCP Background Gate -> MCP Concept Audit -> Host Policy Ownership -> MCP Client Server Boundary -> Server Capability Declaration -> MCP Tool Schema and Result -> MCP Resource URI Metadata -> MCP Prompt Argument Review -> MCP Lifecycle Negotiation -> MCP Transport Policy -> MCP Roots Boundary -> MCP Sampling Control -> Host Capability Filtering -> MCP Context Budget -> MCP Concept Gate -> MCP Server Implementation Audit -> MCP Server Metadata Readiness -> MCP Server Capability Declaration -> MCP Tool Registry Readiness -> MCP Strict Schema Coverage -> MCP Handler Execution Coverage -> MCP Argument Validation Coverage -> MCP Structured Result Coverage -> MCP Structured Error Coverage -> MCP Resource Scope Coverage -> MCP Prompt Template Coverage -> MCP Server Transport Policy Coverage -> MCP Host Connection Readiness -> MCP Server Safety Baseline -> MCP Server Trace Readiness -> MCP Server Gate -> MCP Tool Function Calling Comparison Audit -> Protocol Layer Clarity -> Tool Discovery Boundary -> MCP Capability Scope Coverage -> MCP Execution Boundary Clarity -> MCP Projection Mapping Coverage -> Adapter Separation Coverage -> MCP Lifecycle Version Awareness -> MCP Governance Registry Import -> MCP Security Boundary Enforcement -> MCP Use Case Selection Fit -> MCP Error Surface Separation -> MCP Latency Availability Tradeoff -> MCP Function Calling Gate -> MCP A2A Harness Integration -> Protocol Capability Registry -> MCP Integration -> A2A Integration -> Capability Discovery Coverage -> Namespace Isolation Coverage -> Protocol Schema Validity -> Protocol Permission Binding -> High-Risk Approval Coverage -> Context Output Budget Coverage -> External Trust Boundary Coverage -> A2A Lifecycle Coverage -> Protocol Trace Coverage -> Protocol Replay Readiness -> Protocol Version Capture -> Protocol Integration Gate -> Agent Harness System Design -> Runtime Module Coverage -> Interface Contract Coverage -> Agent State Machine Coverage -> Permission Integration Coverage -> Context Control Coverage -> Execution Isolation Coverage -> System Trace Coverage -> System Replay Readiness -> System Eval Readiness -> Recovery Coverage -> Harness Version Capture -> Enterprise Governance Readiness -> Harness System Gate -> Harness Pitfall Audit -> Harness Triage Coverage -> Permission Overreach -> Context Overload Incident -> User Edit Overwrite -> Shell Hang Incident -> Prompt Injection Boundary Miss -> Trace Missing Incident -> Eval Flakiness Incident -> Doom Loop Guard -> MCP Tool Overload -> Environment Parity Check -> Harness Pitfall Gate -> Browser Agent -> Computer Use Agent -> GUI Action -> Screenshot Observation -> Accessibility Tree -> UI State Tracking -> Misclick Rate -> Form Accuracy -> High-Risk Action Protection -> UI Agent Gate -> Multi-Agent -> Coordinator -> Role Assignment -> Communication Protocol -> Shared State / Blackboard -> Agent Debate -> Judge Agent -> Consensus -> Conflict Resolution -> Duplicate Work Rate -> Single-Agent Lift Rate -> Multi-Agent Gate -> Agent Evaluation -> Evaluation Harness -> Eval Dataset Bucket Coverage -> Environment Reproducibility -> Validator Coverage -> Weighted Task Success -> Partial Success Score -> Diff Scope Safety -> Trace Completeness -> Baseline Fairness -> Regression Pass Rate -> Unsafe Execution Rate -> Cost Budget Pass Rate -> Flaky Task Rate -> Eval Version Capture -> Eval Report Completeness -> Evaluation Harness Gate -> Agent Benchmark -> Eval Sample -> Trajectory Evaluation -> Summary Faithfulness -> Claim Support Rate -> Recovery Success Rate -> Cost-Quality Trade-off -> Regression Suite -> Agent Eval Gate -> Agent Safety -> Least Privilege -> Tool Permission Matrix -> Permission Request -> Least Privilege Coverage -> Permission Matrix Coverage -> Unauthorized Action Block Rate -> Secret Access Block Rate -> Network Egress Control -> Sandbox Enforcement Coverage -> Dry Run Coverage -> Audit Log Completeness -> Irreversible Action Protection -> Permission Sandbox Gate -> Untrusted Content Boundary -> Data Flow Guard -> Sensitive Data Filter -> External Transfer Gate -> Sandbox Policy -> Human Confirmation -> Dry Run -> Audit Log -> Irreversible Action -> Memory Safety -> Supply Chain Risk -> Safety Alternative -> Agent Safety Gate -> Agent Interview Readiness -> Agent Interview Rubric -> Agent Answer Coverage -> Agent Formula Coverage -> Agent Demo Coverage -> Trace Metric Coverage -> Project Evidence Score -> Trade-off Depth -> Weak Agent Question -> Agent Revision Plan -> Agent Interview Gate

Tool Choice Gate -> Argument Parsing -> Parse Success Rate -> Validation Error -> Normalization -> Argument Evidence Map -> Argument Evidence Coverage -> Safe Normalization Rate -> Repair Attempt -> Model Self-Repair -> Clarification Coverage -> Unsafe Argument Block Rate -> Retry Budget Pass Rate -> Idempotency Key Coverage -> Repair Trace Completeness -> Argument Repair Gate -> Structured Function Calling
Tool Registry -> Tool Registry Identity Coverage -> Tool Description Quality -> Schema Contract Coverage -> Runtime Metadata Coverage -> Tool Permission Binding Coverage -> Risk Annotation Coverage -> Version Trace Coverage -> Lifecycle Policy Pass Rate -> Owner SLO Coverage -> Tool Eval Binding Coverage -> Provider Projection Readiness -> Registry Audit Completeness -> Tool Registry Gate -> Tool Router
Tool Router -> Scenario Filter Pass Rate -> Permission Filter Pass Rate -> Risk Filter Pass Rate -> Candidate Recall -> Candidate Precision -> Candidate Size Pass Rate -> Clarification Accuracy -> Tool Choice Mode Accuracy -> Forced Tool Accuracy -> Router Parallel Safety -> Provider Capability Compatibility -> Router Trace Completeness -> Tool Router Gate -> Tool Executor
Tool Executor -> Execution Request Validity -> Schema Validation Pass Rate -> Permission Enforcement Pass Rate -> Execution Mode Accuracy -> Async Completion Tracking -> Timeout Cancellation Coverage -> Idempotency Protection Rate -> Unknown State Escalation Rate -> Retry Safety Rate -> Side Effect Confirmation Coverage -> Structured Result Coverage -> Executor Trace Completeness -> Tool Executor Gate -> Tool Permission Model
Tool Permission Model -> Authorization Decision Accuracy -> Trusted Context Injection -> Tenant Isolation Pass Rate -> Tool Permission Enforcement -> Object Permission Accuracy -> Field Projection Safety -> Action Context Policy Accuracy -> Prompt Injection Block Rate -> Least Privilege Coverage -> Token Audience Validation -> Revocation Cache Safety -> Error Disclosure Safety -> Permission Audit Completeness -> Tool Permission Gate -> Tool Security -> Prompt Injection Containment -> Untrusted Content Isolation -> Unauthorized Access Block Rate -> Sensitive Data Protection -> Tool External Transfer Gate -> Dangerous Action Confirmation -> SSRF Block Rate -> SQL Risk Block Rate -> Path Traversal Block Rate -> Shell Sandbox Enforcement -> Data Flow Policy Pass Rate -> Tool Security Audit Completeness -> Security Alert Completeness -> Safety Eval Regression Pass Rate -> Tool Security Gate -> Tool Trace Replay Audit -> Trace Schema Completeness -> ID Tree Integrity -> Version Capture Coverage -> Argument Lineage Coverage -> Permission Trace Completeness -> Tool Result Trace Completeness -> Privacy Masking Coverage -> Audit Event Completeness -> Replay Readiness Rate -> Replay Side Effect Safety -> Metric Export Coverage -> Alert Owner Coverage -> Eval Linkage Coverage -> Trace Replay Gate -> Tool Version Release Audit -> Spec Lint Pass Rate -> Schema Backward Compatibility -> Description Behavior Guard -> Output Compatibility -> Permission Policy Compatibility -> Version Matrix Capture -> Offline Eval Pass Rate -> Canary Routing Stability -> Canary Quality Guard -> Canary Safety Guard -> Cost Latency Guard -> Tool Release Rollback Readiness -> Lifecycle Coverage -> Tool Release Gate -> Enterprise Tool Platform Audit -> Platform Module Coverage -> Platform Interface Contract Coverage -> Registry Readiness -> Router Governance Coverage -> Executor Safety Coverage -> Platform Permission Integration Coverage -> Platform Safety Guard Coverage -> Platform Trace Audit Coverage -> Platform Eval Service Coverage -> Platform Release Governance Coverage -> Platform Tenant Isolation Coverage -> Platform Provider Adapter Coverage -> High Availability Readiness -> Admin Ops Governance Coverage -> Enterprise Tool Platform Gate -> MCP Background Audit -> Direct Integration Count -> MCP Integration Count -> Integration Reduction -> Capability Model Coverage -> Context Object Coverage -> Discovery Standardization -> MCP Schema Contract Coverage -> Host Server Boundary Clarity -> Local Context Control -> Cross Client Reuse -> Governance Boundary Clarity -> MCP Trace Eval Readiness -> MCP A2A Distinction -> MCP Background Gate -> MCP Concept Audit -> Host Policy Ownership -> MCP Client Server Boundary -> Server Capability Declaration -> MCP Tool Schema and Result -> MCP Resource URI Metadata -> MCP Prompt Argument Review -> MCP Lifecycle Negotiation -> MCP Transport Policy -> MCP Roots Boundary -> MCP Sampling Control -> Host Capability Filtering -> MCP Context Budget -> MCP Concept Gate -> MCP Server Implementation Audit -> MCP Server Metadata Readiness -> MCP Server Capability Declaration -> MCP Tool Registry Readiness -> MCP Strict Schema Coverage -> MCP Handler Execution Coverage -> MCP Argument Validation Coverage -> MCP Structured Result Coverage -> MCP Structured Error Coverage -> MCP Resource Scope Coverage -> MCP Prompt Template Coverage -> MCP Server Transport Policy Coverage -> MCP Host Connection Readiness -> MCP Server Safety Baseline -> MCP Server Trace Readiness -> MCP Server Gate -> MCP Tool Function Calling Comparison Audit -> Protocol Layer Clarity -> Tool Discovery Boundary -> MCP Capability Scope Coverage -> MCP Execution Boundary Clarity -> MCP Projection Mapping Coverage -> Adapter Separation Coverage -> MCP Lifecycle Version Awareness -> MCP Governance Registry Import -> MCP Security Boundary Enforcement -> MCP Use Case Selection Fit -> MCP Error Surface Separation -> MCP Latency Availability Tradeoff -> MCP Function Calling Gate -> Agent Trace
MCP Function Calling Gate -> MCP Resource Exposure Audit -> Resource URI Validity -> Resource Metadata Completeness -> Resource List Filtering Boundary -> Resource Read Scope Enforcement -> MCP Roots Containment -> MCP Resource Permission Enforcement -> MCP Resource Field Projection Safety -> Resource Context Budget Control -> Resource Citation Traceability -> Untrusted Resource Labeling -> Resource Freshness Version Awareness -> Resource Template Boundary -> Resource Subscription Awareness -> Resource Trace Eval Readiness -> MCP Resource Gate -> MCP Prompt Exposure Audit -> Prompt Capability Declaration -> Prompt Discovery Coverage -> Prompt Argument Schema Coverage -> Prompt Required Argument Validation -> Prompt Rendering Safety -> Prompt Role Boundary Enforcement -> Prompt Dependency Alignment -> Prompt Version Governance -> Prompt Eval Binding -> Prompt Permission Enforcement -> Prompt Injection Containment -> Prompt User Control -> Prompt List Change Awareness -> Prompt Trace Readiness -> MCP Prompt Gate -> MCP Security Sandbox Audit -> Server Connection Governance -> Authorization Token Binding -> Scope Minimization -> Roots Sandbox Containment -> File Secret Blocking -> Network SSRF Protection -> MCP Shell Sandbox Enforcement -> Prompt Injection Data Boundary -> High Risk Confirmation -> Sensitive Data Flow Control -> Local Credential Isolation -> MCP Tenant Isolation -> Server Supply Chain Governance -> MCP Security Trace Readiness -> MCP Security Eval Coverage -> MCP Security Gate -> MCP Integration Surface Audit -> MCP Capability Registration Coverage -> MCP Namespace Isolation Coverage -> MCP IDE Context Routing -> MCP Knowledge Citation Traceability -> MCP Database Query Governance -> MCP Browser Action Governance -> MCP Terminal Sandbox Governance -> MCP Integration Context Budget Control -> MCP Cross Server Data Flow Control -> MCP High Risk Approval Coverage -> MCP Output Projection Coverage -> MCP Integration Trace Readiness -> MCP Integration Eval Coverage -> MCP Integration Gate -> MCP A2A Harness Integration
MCP Security Gate -> MCP Integration Surface Audit -> MCP Integration Gate -> Protocol Capability Registry -> MCP Integration -> Agent Trace
Skill Definition Gate -> Ecosystem Boundary Audit -> Required Field Coverage -> Abstraction Classification -> Granularity Boundary -> Action Trace -> Tool Contract -> Workflow Control -> Skill Bundle -> Plugin Packaging -> Permission Layering -> Eval Layering -> Audit Trace Layering -> Naming Alias Documentation -> Governance Lifecycle -> High Risk Approval -> Eval Ready -> Ecosystem Boundary Gate -> MCP A2A Harness Integration
Ecosystem Boundary Gate -> Skill Manifest Audit -> Identity Metadata -> Description Quality -> Capability Granularity -> Input Contract -> Output Contract -> Tool Dependency Purpose -> Resource Prompt Binding -> Manifest Workflow Readiness -> Permission Least Privilege -> Configuration Schema -> Safety Policy -> Eval Gate -> Examples Trigger Coverage -> Version Lifecycle -> Audit Readiness -> Skill Manifest Gate -> MCP A2A Harness Integration
Skill Manifest Gate -> Skill Lifecycle Audit -> Review Before Publish -> Install Approval -> Enable Scope Control -> Permission Reapproval -> Configuration Versioning -> Running Task Policy -> Lifecycle Compatibility Check -> Rollout Guard -> Skill Rollback Readiness -> Dependency Versioning -> Lifecycle Audit Log Completeness -> Emergency Suspend Readiness -> Uninstall Retention -> Lifecycle Eval Monitoring -> Update Policy Accuracy -> Skill Lifecycle Gate -> Skill Marketplace Audit -> Catalog Metadata -> Search Discovery -> Detail Page Completeness -> Review Workflow -> Permission Transparency -> Install Approval Flow -> Rating Quality Balance -> Operations Metrics -> Duplicate Capability Governance -> Admin Permission View -> Developer Console Readiness -> Security Center Readiness -> Recommendation Policy Safety -> Lifecycle Visibility -> Owner Maintenance -> Portal Audit Trace -> Skill Marketplace Gate -> Tool Skill Quality Safety Audit -> Tool Schema Clarity -> Argument Validation -> Execution Reliability -> Output Stability -> Side Effect Control -> Skill Task Quality -> Factual Grounding -> Completeness Format -> Offline Eval Coverage -> Online Monitoring -> Permission Least Privilege -> Data Security Review -> Prompt Injection Resilience -> High Risk Action Control -> Supply Chain Governance -> Human Review Readiness -> Regression Release Gate -> Audit Trace Readiness -> Quality Safety Gate -> MCP A2A Harness Integration
MCP Integration Gate -> A2A Background Audit -> Agent Card Completeness -> Agent Discovery Readiness -> Task Delegation Contract -> A2A Task Lifecycle Coverage -> A2A Message Structure Coverage -> A2A Artifact Reference Coverage -> A2A Context Boundary Control -> A2A Permission Boundary -> A2A MCP Distinction -> A2A Failure Handling Coverage -> A2A Trace Readiness -> A2A Eval Coverage -> A2A Background Gate -> Agent Card Discovery Audit -> Agent Card Field Completeness -> Agent Skill Declaration Quality -> Supported Interface Readiness -> Agent Card Security Coverage -> Agent Card Version Cache Readiness -> Agent Discovery Match Quality -> Agent Routing Decision Quality -> Extended Agent Card Control -> Agent Card Trace Readiness -> Agent Card Eval Coverage -> Agent Card Gate -> A2A Task Delegation Audit -> A2A Task Contract Coverage -> A2A State Transition Validity -> A2A Input Required Handling -> A2A Artifact Metadata Coverage -> A2A Error Semantics Coverage -> A2A Retry Idempotency Coverage -> A2A Cancellation Coverage -> A2A Delegation Permission Boundary -> A2A Parallel Aggregation Readiness -> A2A Task Trace Readiness -> A2A Task Eval Coverage -> A2A Task Gate -> A2A Message Boundary Audit -> A2A Message Contract Coverage -> A2A Part Typing Coverage -> A2A Source Trust Labeling -> A2A Instruction Data Separation -> A2A Minimal Context Coverage -> A2A Reference Over Copy Coverage -> A2A Context Policy Enforcement -> A2A Sensitive Redaction Coverage -> A2A Claim Grounding Coverage -> A2A Summary Constraint Retention -> A2A Message Trace Readiness -> A2A Message Eval Coverage -> A2A Message Gate -> A2A MCP Boundary Audit -> A2A MCP Protocol Classification -> A2A Tool Agent Boundary -> A2A Autonomy Fit -> A2A Lifecycle Placement -> A2A MCP Discovery Split -> A2A MCP Context Ownership -> A2A MCP Permission Separation -> A2A Result Artifact Boundary -> A2A MCP Trace Linkage -> A2A MCP Version Eval Coverage -> A2A MCP Boundary Gate -> Cross-Agent Security Audit -> Cross-Agent Identity Chain Coverage -> OBO Scope Binding -> Delegation Allowlist Coverage -> Permission Attenuation Coverage -> Cross-Agent Context Policy Enforcement -> Cross-Agent Redelegation Control -> Cross-Agent High Risk Confirmation -> Cross-Agent Tool Permission Binding -> Cross-Agent Result Release Control -> Cross-Agent Audit Trace Completeness -> Cross-Agent Trust Evidence Verification -> Cross-Agent Tenant Isolation -> Cross-Agent Security Gate -> Multi-Agent Failure Audit -> Delegation Loop Control -> Role Conflict Arbitration -> Hallucination Propagation Containment -> Context Drift Control -> Duplicate Work Budget Control -> Multi-Agent State Consistency -> Artifact Conflict Control -> Multi-Agent Accountability Trace -> Policy Chain Enforcement -> Collaboration Fit -> Termination Handoff Readiness -> Multi-Agent Failure Eval Coverage -> Multi-Agent Failure Gate -> A2A System Design Audit -> A2A Requirement Clarification -> A2A Architecture Module Coverage -> A2A Protocol Contract Coverage -> A2A Discovery Routing Governance -> A2A Runtime State Readiness -> A2A Context Boundary Control -> A2A Permission Model Coverage -> A2A MCP Boundary Clarity -> A2A Artifact Governance Coverage -> A2A Trace Audit Completeness -> A2A Failure Handling Coverage -> A2A Eval Observability Coverage -> A2A Scalability Idempotency Readiness -> A2A System Design Gate -> Cross-Agent Collaboration System Audit -> Cross-Agent Requirement Goal Clarity -> Cross-Agent Collaboration Module Coverage -> Cross-Agent Registry Routing Governance -> Cross-Agent Task Graph Validity -> Cross-Agent A2A Lifecycle Alignment -> Cross-Agent Context Minimization Enforcement -> Cross-Agent Permission Attenuation Binding -> Cross-Agent MCP Tool Boundary Governance -> Cross-Agent Artifact Evidence Grounding -> Cross-Agent Conflict Arbitration Readiness -> Cross-Agent Hallucination Propagation Control -> Cross-Agent Delegation Loop Containment -> Cross-Agent Human Handoff Approval Readiness -> Cross-Agent Trace Audit Replay Coverage -> Cross-Agent Eval Baseline Regression Coverage -> Cross-Agent Cost Latency Budget Control -> Cross-Agent Scalability Idempotency Readiness -> Cross-Agent Collaboration Fit Control -> Cross-Agent Collaboration System Gate -> Tool Protocol Future Audit -> Layered Protocol Boundary Clarity -> Standardization Extension Balance -> Long Task Lifecycle Readiness -> Capability Package Completeness -> Agent-Centric Autonomy Governance -> Tool Safety Metadata Coverage -> Provenance Verifiability -> Marketplace Governance Readiness -> Provider Adapter Migration Control -> Auto Generated Tool Review Gate -> Workflow Runtime Integration -> Behavior Eval Coverage -> Human Model Documentation Split -> Autonomy Risk Tiering -> Responsibility Attribution Trace -> Natural Language API Boundary -> Agent Operating Layer Governance -> Engineer Readiness Coverage -> Unsupported Speculation Rate -> Tool Protocol Future Gate -> Skill Definition Audit -> Skill Manifest Metadata -> Skill Task Goal Clarity -> Skill Bundle Completeness -> Tool Skill Boundary -> Skill Instruction Strategy -> Skill Resource Prompt Grounding -> Skill Workflow Readiness -> Skill Permission Safety -> Skill Configuration Reuse -> Skill Eval Coverage -> Skill Lifecycle Governance -> Skill Product Install Governance -> Skill Progressive Disclosure -> Skill Definition Gate -> MCP A2A Harness Integration

Argument Repair Gate -> Tool Result -> Tool Result Context -> Tool Result Projection -> Source Metadata Coverage -> Redaction Coverage -> Tool Result Context Budget Pass Rate -> Injection Containment Rate -> Error Status Fidelity -> Compression Fidelity -> Conflict Labeling Rate -> Citation Support Rate -> Tool Result Freshness Pass Rate -> Memory Boundary Pass Rate -> Tool Result Context Gate -> Context Builder
Tool Result Context Gate -> Tool Failure Recovery -> Structured Tool Error -> Retryable Error -> Non-Retryable Error -> Retry Policy Precision -> Exponential Backoff with Jitter -> Retry Budget -> Unknown Execution State -> Fallback Honesty -> Circuit Breaker Containment -> Timeout Cancellation Coverage -> Human Handoff Readiness -> User-Visible Error Clarity -> Failure Trace Completeness -> Tool Failure Recovery Gate -> Agent Evaluation
Tool Failure Recovery Gate -> Tool Calling Evaluation -> Tool Call Recall -> Tool Call Precision -> Tool Selection Accuracy -> Tool Set Precision -> Tool Set Recall -> Argument Value Accuracy -> Argument Source Coverage -> Execution Success Rate -> Observation Use Correctness -> Safe Failure Rate -> Unsafe Failure Rate -> Tool Eval Regression Pass Rate -> Tool Calling Eval Gate -> Agent Evaluation
Tool Calling Eval Gate -> Tool Skill Developer Experience Audit -> Quickstart Path Clarity -> Scaffold Completeness -> Schema Contract -> Documentation Example Coverage -> Local Debug Readiness -> Trace Replay Readiness -> Review Feedback Actionability -> Security Documentation Coverage -> Migration Documentation Coverage -> CLI SDK Consistency -> Documentation Freshness -> Owner Maintenance -> Portal Path Clarity -> DX Documentation Gate -> Skill Marketplace
DX Documentation Gate -> Composable Workflow Audit -> Workflow Graph Validity -> Dependency Acyclicity -> Workflow IO Contract -> Condition Determinism -> Workflow State Coverage -> Workflow Permission Boundary -> Workflow Data Flow Policy -> Workflow Idempotency Coverage -> Retry Safety -> Compensation Readiness -> Workflow Human Approval Coverage -> Workflow Trace Replay Readiness -> Workflow Eval Coverage -> Workflow Gate -> Skill Definition Gate -> MCP A2A Harness Integration
Workflow Gate -> Prompt Injection Defense Audit -> Instruction Data Separation -> Source Trust Labeling -> Taint Propagation -> Policy Pre-Tool Gate -> Risky Tool Isolation -> Untrusted Action Blocking -> Sensitive Data Control -> Prompt Injection High Risk Confirmation -> Sandbox Enforcement -> Output Redaction Projection -> RAG Source Boundary -> Multi-Agent Taint Propagation -> Prompt Injection Trace Readiness -> Prompt Injection Eval Coverage -> Prompt Injection Defense Gate -> Tool Security Gate -> MCP A2A Harness Integration
Prompt Injection Defense Gate -> Tool Output Trust Audit -> Source Metadata Coverage -> Trust Level Coverage -> Freshness Version Coverage -> Access Scope Disclosure -> Citation Binding -> Citation Support Accuracy -> Evidence Chain Completeness -> Claim Type Calibration -> Confidence Limitation Disclosure -> Conflict Resolution Readiness -> Summary Provenance Retention -> RAG Chunk Citation Accuracy -> Multi-Agent Evidence Propagation -> Provenance Trace Replay -> Citation Forgery Block -> Tool Output Trust Eval Coverage -> Tool Output Trust Gate -> Tool Result Context Gate -> MCP A2A Harness Integration
Tool Output Trust Gate -> RAG Tool Agent Memory Integration Audit -> Capability Boundary Clarity -> Orchestration Mode Fit -> Context Priority Enforcement -> Context Budget Allocation -> Evidence Preservation -> Tool Observation Use -> Agent State Update Discipline -> Memory Read Relevance -> Memory Write Gate -> Memory Scope Permission -> Conflict Resolution Policy -> Injection Propagation Control -> Sensitive Data Memory Block -> Trace Linkage Coverage -> Layered Eval Coverage -> RAG Tool Agent Memory Integration Gate -> Agentic RAG Gate -> Memory Gate -> MCP A2A Harness Integration
RAG Tool Agent Memory Integration Gate -> Tool Cost Latency Concurrency Audit -> Cost Attribution Coverage -> Latency Breakdown Coverage -> Tool Call Budget Enforcement -> Timeout Deadline Coverage -> Retry Classification Accuracy -> Idempotent Retry Safety -> Concurrency Limit Enforcement -> Queue Priority Fairness -> Rate Limit Quota Handling -> Cache Safety Correctness -> Batching Partial Success Readiness -> Result Trimming Budget Control -> Degradation Transparency -> Router Performance Awareness -> Performance Trace Readiness -> Cost Latency Eval Coverage -> Tool Performance Gate -> Tool Calling Eval Gate -> MCP A2A Harness Integration
Tool Performance Gate -> Tool Use Eval Benchmark Audit -> Tool Need Accuracy -> No-Tool Overcall Control -> Tool Selection Accuracy -> Tool Set Precision -> Tool Set Recall -> Argument Schema Validity -> Argument Semantic Accuracy -> Argument Source Coverage -> Sequence Order Accuracy -> Observation Grounding Accuracy -> Error Recovery Readiness -> Safe Failure Handling -> Safety Policy Compliance -> Tool Simulator Determinism -> Trace Replay Coverage -> Cost Latency Regression Control -> Benchmark Slice Coverage -> Regression Gate Readiness -> Tool Use Eval Benchmark Gate -> Tool Calling Eval Gate -> MCP A2A Harness Integration
Tool Use Eval Benchmark Gate -> Function MCP A2A Comparison Audit -> Layer Boundary Clarity -> Function Calling Fit -> MCP Integration Fit -> A2A Delegation Fit -> Object Contract Coverage -> Capability Discovery Fit -> Lifecycle State Alignment -> Context Transfer Boundary -> Permission Governance Split -> Host Runtime Ownership -> Trace Chain Continuity -> Eval Gate Linkage -> Overengineering Control -> Severe Protocol Misuse Rate -> Protocol Composition Gate -> MCP A2A Harness Integration
Protocol Composition Gate -> Provider Framework Tool Protocol Audit -> Provider Framework Type Clarity -> Tool Schema Projection -> Tool Choice Mapping -> Tool Result Round Trip -> Streaming Event Assembly -> Parallel Call Alignment -> Built-In Tool Boundary -> Error Normalization -> Provider Adapter Isolation -> Framework Escape Hatch -> RAG Tool Traceability -> Permission Safety Consistency -> Migration Eval Coverage -> Provider Trace Replay Readiness -> Vendor Lock-In Control -> Provider Runtime Gate -> Enterprise MCP Platform Audit -> MCP Gateway Readiness -> MCP Registry Metadata Completeness -> MCP Capability Namespace Isolation -> MCP Tool Resource Prompt Contract -> MCP Host Client Server Boundary Clarity -> MCP OBO Scope Binding -> MCP Tenant Isolation Enforcement -> MCP Policy Engine Coverage -> MCP Sandbox Roots Containment -> MCP Context Output Projection -> MCP Prompt Injection Taint Propagation -> MCP Trace Audit Replay Continuity -> MCP Cost Latency Quota Control -> MCP Developer Portal Review Readiness -> MCP Release Lifecycle Governance -> MCP Provider Adapter Compatibility -> MCP Eval Regression Coverage -> MCP High Availability Readiness -> Enterprise MCP Platform Gate -> Enterprise Tool Platform Gate -> MCP A2A Harness Integration

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

PyTorch Tensor -> Tensor Shape -> Dtype -> Device -> Broadcasting -> Tensor Stride -> Contiguous Tensor -> View vs Reshape -> Matmul -> Einsum -> Tensor Mask -> LM Loss Flatten -> Tensor Shape Audit -> Autograd -> Dynamic Computation Graph -> Requires Grad -> Leaf Tensor -> Backward -> Vector-Jacobian Product -> Grad Accumulation -> Zero Grad -> Detach -> No Grad -> In-place Operation Risk -> Autograd Audit -> nn.Module -> Parameter -> Module Registration -> Submodule -> ModuleList -> ModuleDict -> Buffer -> state_dict -> load_state_dict -> train/eval Mode -> Module Hook -> Module Audit -> Dataset/DataLoader -> Map-style Dataset -> IterableDataset -> Collate Function -> Dynamic Padding -> Ignore Index -> Sampler -> Batch Sampler -> DistributedSampler -> DataLoader Worker -> Pin Memory -> Padding Waste -> Data Pipeline Audit -> Training Loop -> Training Step -> Raw Loss -> Scaled Loss -> Optimizer Step -> Scheduler Step -> Evaluation Loop -> Checkpoint Resume -> Non-Finite Loss -> Training Stability -> Loss Spike Audit -> Effective Label Tokens -> LR Continuity -> Rank Loss Skew -> Training Stability Gate -> Training Loop Audit -> Mixed Precision -> FP16 -> BF16 -> Autocast -> GradScaler -> Loss Scaling -> Activation Memory -> Activation Checkpointing -> CUDA Memory Stats -> OOM Audit -> Distributed Training -> Rank/World Size -> Process Group -> DDP -> All-Reduce -> Gradient Synchronization -> Global Batch Size -> DDP no_sync -> FSDP -> Distributed Training Audit -> Rank Step Alignment -> Collective Mismatch -> Token Shard Imbalance -> Straggler Ratio -> Communication Ratio -> Checkpoint Shard Coverage -> Resume Continuity -> Pipeline Bubble Ratio -> Distributed Incident Gate -> Debug/Profiling -> Debug Tensor Metadata -> Finite Check -> Gradient Debug -> Forward Hook Debug -> Anomaly Detection -> CUDA Synchronize Timing -> torch.profiler -> DataLoader Bottleneck -> Debug/Profiling Audit -> Transformer Components -> Token Embedding -> LM Head Weight Tying -> RMSNorm -> Causal Mask -> Scaled Dot-Product Attention -> Multi-Head Self-Attention -> SwiGLU MLP -> Pre-Norm Decoder Block -> RoPE -> KV Cache -> Transformer Component Audit -> PyTorch Engineering Interview Readiness -> Engineering Interview Gate -> Revision Plan

### 对齐链

Human Intent -> Safety Policy -> Proxy Objective -> Instruction Data -> SFT -> Preference Data -> Reward Model -> RLHF/DPO -> Reward Hacking Audit -> Scalable Oversight -> AI Feedback -> Verifier -> Human Audit -> Outer Alignment Check -> Inner Alignment Check -> Goal Misgeneralization Eval -> Model Behavior Evaluation -> Alignment Gate

### 安全治理链

AI Safety -> Alignment -> Alignment Problem -> HHH -> Risk Taxonomy -> Safety Policy -> Instruction Hierarchy -> Untrusted Content Boundary -> Jailbreak Eval -> Prompt Injection Eval -> Prompt Injection Gate -> Reward Hacking Gate -> Scalable Oversight -> Oversight Gate -> Safety Evaluation -> Red Teaming -> Capability Elicitation -> Dangerous Capability Eval -> Red Team Regression Suite -> Red Team Gate -> Mechanistic Interpretability -> Activation Patching -> Sparse Autoencoder -> Interpretability Gate -> Representation Engineering -> Steering Vector -> Activation Steering -> Steering Gate -> Model Editing -> Knowledge Editing -> Editing Gate -> Machine Unlearning -> Forget Set -> Retain Set -> Unlearning Gate -> Data Privacy -> Memorization -> Training Data Extraction Eval -> Membership Inference -> PII Leakage Eval -> Privacy Gate -> Privacy Governance -> Data Minimization Rate -> PII Redaction Coverage -> Sensitive Data Block Rate -> Permission Isolation -> Log Redaction Coverage -> Training Consent Coverage -> Retention Compliance Rate -> Deletion SLA Pass Rate -> External Transfer Approval Coverage -> Audit Coverage -> DPA Ready -> Incident Response Ready -> Privacy Governance Gate -> Safety Compliance Incident -> Unsafe Pass Rate -> Safety Compliance Gate -> Watermarking -> Watermark Detection -> Content Credentials -> Watermark Gate -> Model Card -> System Card -> Risk Disclosure -> Responsible Scaling -> Model Governance -> Governance Gate -> Safety Gate -> Release Gate -> Guardrail -> Tool Safety -> Audit Log -> Incident Response -> Safety Interview Readiness -> Safety Interview Rubric

### 多模态链

Multimodal Data -> Image-Text Alignment -> OCR/ASR/Video Temporal Alignment -> Vision Encoder -> Patch Embedding -> CLS/Patch Tokens -> Vision Encoder Shape Audit -> Visual Token Budget -> Multimodal Context Budget -> CLIP -> CLIP Loss -> Zero-Shot Classification -> Image-Text Retrieval -> VLM Connector -> VLM Connector Audit -> Image Placeholder -> Visual Token Compression -> Assistant-Only Multimodal Loss -> Chat Template -> Multimodal SFT Data Audit -> Multimodal Task Coverage -> Evidence Support Rate -> Missing Refusal Rate -> Multimodal Instruction Tuning -> Diffusion Model -> DDPM -> Forward Diffusion -> Reverse Denoising -> Noise Scheduler -> Noise Prediction Loss -> Classifier-Free Guidance -> Latent Diffusion -> Stable Diffusion -> VAE Compression Ratio -> Text-to-Image Pipeline Audit -> Negative Prompt -> ControlNet -> Image-to-Image -> Inpainting -> DALL-E -> Autoregressive Image Tokens -> Video Generation -> Spatiotemporal Patch -> Video Diffusion -> Temporal Consistency -> Identity Drift -> Flickering -> World Model -> Physics Consistency -> Video Generation Evaluation -> FVD -> Video Token Audit -> Audio Generation -> Waveform -> Log-Mel Spectrogram -> ASR -> Whisper -> WER -> CER -> TTS -> Vocoder -> Voice Cloning -> Audio Codec -> Speech Token -> Codec Language Model -> Speech-to-Speech -> VAD -> MOS -> Audio Token Audit -> Unified Multimodal Model -> Unified Tokenization -> Any-to-Any Multimodal Model -> Early-Fusion Multimodal Transformer -> Center-LLM Multimodal System -> Multimodal Router -> Multimodal Loss Mixture -> Modality Conflict -> Unified Multimodal Audit -> Multimodal Evaluation -> Multimodal Incident -> Input Fidelity Rate -> Multimodal Evidence Recall -> OCR Critical Field Accuracy -> ASR Critical Field Accuracy -> Temporal Evidence Recall -> VQA Accuracy -> Chart Relaxed Accuracy -> Grounding IoU -> Multimodal Hallucination Rate -> Multimodal Prompt Injection -> Biometric Identity Safety -> Content Provenance -> Multimodal Safety Audit -> Multimodal Incident Gate -> Multimodal Interview Readiness -> Multimodal Interview Rubric -> Multimodal Formula Coverage -> Multimodal Demo Coverage -> Weak Multimodal Question -> Multimodal Revision Plan -> Multimodal Reasoning -> Multimodal Cost Audit -> Multimodal Product -> Input Quality Pass Rate -> Multimodal Evidence Support Rate -> OCR Quality Score -> ASR Quality Score -> Generation Adoption Rate -> Media Safety Pass Rate -> Privacy Pass Rate -> Copyright Pass Rate -> Multimodal Product Gate -> Multimodal Safety

### 评估链

Benchmark -> Metrics -> Evaluation Metric Incident -> Aggregate Score Trap -> Evaluation Slicing -> Slice Regression -> Clean Eval Lift -> Human Eval -> LLM-as-a-Judge -> Judge-Human Agreement -> Judge Length Bias -> Paired Evaluation -> Bootstrap Confidence Interval -> Contamination Detection -> Error Analysis -> Regression Test -> Online Evaluation -> Cost-Quality Trade-off -> Evaluation Gate

### 数据工程链

Source Registry -> Web-Scale Collection -> Parsing -> Data Cleaning -> Quality Scoring -> Exact Deduplication -> Near Deduplication -> Code/Math/Domain Audit -> Synthetic/Distillation Audit -> Preference/Safety Audit -> Multimodal Data Audit -> PII/Secret Filtering -> Safety Filtering -> Train-Eval Overlap -> Contamination Detection -> Data Attribution -> Data Valuation -> Data Mixture -> Mixture Shift -> Data Sampling -> Dataset Versioning -> Data Lineage -> Lineage Coverage -> Data Governance -> Data Risk Rate -> Data Incident -> Data Incident Gate -> Data Interview Readiness -> Pretraining

### 产品化链

LLM Productization -> Demo to Product -> User Persona -> Stakeholder Map -> Real Demand -> User Pain -> Task Journey -> Scenario Selection -> Frequency-Value Matrix -> LLM Fit -> Data Readiness -> Verifiability -> Workflow Fit -> Automation Level -> Pilot Scenario -> Scenario Selection Gate -> Model-to-Experience Mapping -> LLM Product Experience -> Experience Metric -> Task Success Rate -> P95 Latency -> Latency SLO -> Structured Output Stability -> Citation Support Rate -> Controllability -> Error Recovery Rate -> Trust Calibration -> Over-Refusal Rate -> UX Score -> Experience Gate -> Product Metric -> Business Metric -> Task Success Uplift -> Adoption Rate -> Unit Economics -> Total Cost of Ownership -> Model Cost -> Token Cost -> RAG Cost -> Tool Cost -> Review Cost -> Retry Cost -> Risk Cost -> Variable Cost -> Fixed Cost -> Monthly Benefit -> Monthly Cost -> Net Benefit -> Unit Margin -> Benefit-Cost Ratio -> ROI -> Payback Period -> Break-Even Point -> Sensitivity Analysis -> ROI Gate -> RAG Product -> RAG Incident -> RAG Error Attribution -> Document Governance -> Retrieval Recall -> Mean Reciprocal Rank / MRR -> Retrieval Miss -> Context Drop -> Context Recall -> Context Precision -> Evidence Support Rate -> Citation Accuracy -> Unsupported Claim Rate -> Permission Leak Rate -> Abstention Accuracy -> Stale Evidence Rate -> RAG Freshness Gate -> RAG Product Gate -> RAG Incident Gate -> Agent Product -> Workflow-Agent Hybrid -> Agent Automation Level -> Agent Product Trace -> Tool Execution Success Rate -> Human Confirmation Coverage -> Agent Recovery Rate -> Observation Use Rate -> State Update Coverage -> Unauthorized Action Rate -> Budget Overrun Rate -> Agent Product Gate -> Multimodal Product -> Input Quality Pass Rate -> Multimodal Evidence Support Rate -> OCR Quality Score -> ASR Quality Score -> Generation Adoption Rate -> Media Safety Pass Rate -> Privacy Pass Rate -> Copyright Pass Rate -> Multimodal Product Gate -> Privacy Governance -> Data Minimization Rate -> PII Redaction Coverage -> Sensitive Data Block Rate -> Permission Isolation -> Log Redaction Coverage -> Training Consent Coverage -> Retention Compliance Rate -> Deletion SLA Pass Rate -> External Transfer Approval Coverage -> Audit Coverage -> DPA Ready -> Incident Response Ready -> Privacy Governance Gate -> Safety Compliance Gate -> Enterprise LLM Application -> Enterprise Knowledge Base -> Enterprise Integration -> SSO -> IAM -> RBAC -> Tenant Isolation -> Permission-Aware RAG -> RAG Permission Filter -> Tool Permission Gate -> Audit Log Coverage -> PII Redaction -> Data Freshness -> Workflow Integration -> SLO Pass Rate -> Human Review Coverage -> Enterprise Gate -> Human-in-the-Loop -> Safety / Privacy Gate -> Evaluation Gate -> Feedback Loop -> Feedback Action Rate -> Bad Case Triage Coverage -> Regression Pass Rate -> Slice Coverage -> Canary Rollout -> Shadow Traffic -> Feature Flag -> Online Monitoring -> Trace Coverage -> Incident Response Coverage -> Rollback Plan -> SLO Burn Rate -> Ops Gate -> Project Collaboration Incident -> Objective Clarity Rate -> Metric Tree Coverage -> Baseline Coverage -> Experiment Reproducibility Rate -> Version Trace Coverage -> RACI Coverage -> Change Control Coverage -> Risk Escalation Coverage -> Project Collaboration Gate -> Product Interview Readiness -> Formula Coverage -> Demo Evidence Coverage -> Trade-off Coverage -> Weak Product Question -> Product Interview Gate -> Productization Gate

### AI Infra 链

AI Infra -> AI Infra Overview Audit -> Compute Accelerator Readiness -> Network Communication Readiness -> Storage Data Checkpoint Readiness -> Scheduler Resource Governance -> Training Platform Reproducibility -> Inference Platform SLO Readiness -> Data Platform Lineage Quality -> Model Artifact Registry Governance -> Eval Experiment Tracking Coverage -> Observability Signal Coverage -> Security Governance Coverage -> Cost Capacity Governance -> Developer Self-Service Readiness -> AI Infra Boundary Clarity -> MLOps LLMOps Platform Boundary Clarity -> Algorithm Infra Collaboration Readiness -> AI Infra Overview Gate -> AI Infra MLOps LLMOps Boundary Audit -> AI Infra Scope Accuracy -> MLOps Lifecycle Accuracy -> LLMOps Application Accuracy -> Platform Engineering DX Accuracy -> DevOps SRE Boundary Accuracy -> Data Platform Boundary Accuracy -> Model Platform Boundary Accuracy -> Primary Owner Clarity -> Interface Contract Coverage -> Artifact Lineage Handoff -> Observability SLO Handoff -> Security Cost Governance Handoff -> Lifecycle Stage Mapping -> Anti Tool Name Confusion -> Incident Routing Accuracy -> Collaboration Handoff Readiness -> Boundary Gate -> Accelerator Selection Audit -> Peak Compute Fit -> Memory Capacity Fit -> Memory Bandwidth Fit -> Interconnect Bandwidth Fit -> Low Precision Support -> Software Stack Maturity -> Kernel Library Readiness -> Distributed Communication Readiness -> Training Memory Budget -> Inference KV Cache Budget -> Workload Hardware Fit -> Cloud Self Build Decision -> Cost Power Capacity Awareness -> Profiling Observability Readiness -> Fallback Portability Plan -> Selection Risk Governance -> Accelerator Selection Gate -> Bandwidth Bottleneck Audit -> VRAM Capacity Accounting -> HBM Bandwidth Model -> PCIe Transfer Awareness -> NVLink Topology Awareness -> NVSwitch All-to-All Awareness -> Inter-Node Network Awareness -> KV Cache Growth Accounting -> Training State Memory Accounting -> Communication Volume Accounting -> Topology-Aware Parallel Group -> Dataloader Storage IO Awareness -> Checkpoint IO Awareness -> Offload Penalty Awareness -> Overlap Fusion Optimization -> Observability Metric Coverage -> Bandwidth Bottleneck Gate -> Training Efficiency Audit -> Tokens Throughput Accounting -> Step Time Breakdown Coverage -> GPU Utilization Interpretation -> MFU Estimation -> HFU Estimation -> Model FLOPs Accounting -> Hardware Peak Accounting -> Communication Ratio Tracking -> IO Dataloader Tracking -> Checkpoint Overhead Tracking -> Rank Skew Detection -> Scaling Efficiency Tracking -> Padding Waste Awareness -> Recompute Overhead Awareness -> Loss Correctness Coupling -> Training Efficiency Gate -> Task Profile Audit -> Workload Type Classification -> Resource Shape Completeness -> Pretraining Profile Accuracy -> SFT Iteration Profile Accuracy -> RLHF Pipeline Profile Accuracy -> Evaluation Reproducibility Profile -> Serving SLO Profile -> RAG Freshness Retrieval Profile -> Agent Tool Runtime Profile -> Multimodal Resource Profile -> Scheduler Policy Fit -> Observability Metric Fit -> Cost Model Fit -> Artifact Lineage Fit -> Safety Governance Fit -> Task Profile Gate -> GPU Cluster Topology Audit -> Scale-Up Domain Fit -> PCIe NUMA Locality -> NVLink NVSwitch Locality -> GPU NIC Affinity -> Inter-Node Fabric Readiness -> Rack Locality Awareness -> Oversubscription Awareness -> Collective Communication Fit -> Parallel Group Placement -> Storage Checkpoint Locality -> Fault Domain Isolation -> Power Cooling Capacity Fit -> Resource Pool Isolation -> Topology-Aware Scheduling -> Observability Topology Coverage -> GPU Cluster Gate -> Network Communication Audit -> Bandwidth Unit Accounting -> Latency Jitter Tracking -> RDMA Capability Fit -> GPUDirect RDMA Path -> InfiniBand Fabric Readiness -> RoCE Congestion Losslessness -> Ethernet Fallback Scope -> Collective Operation Modeling -> NCCL Topology Runtime Fit -> AllReduce Cost Estimation -> AllGather ReduceScatter Cost -> Rank Straggler Detection -> Packet Error Retransmit Tracking -> Topology Congestion Awareness -> Scheduler Network Locality -> Network Communication Gate -> Storage System Audit -> Capacity Tier Fit -> Throughput IOPS Fit -> Local NVMe Cache Fit -> Shared FS Metadata Fit -> Object Store Authority Fit -> Data Lake Governance Fit -> Training Shard Format Fit -> Small File Amplification Control -> Dataloader Cache Hit Tracking -> Checkpoint Write Recovery Fit -> Model Weight Load Cache Fit -> Artifact Lineage Metadata Fit -> Consistency Commit Integrity -> Security Compliance Fit -> Lifecycle Cost Governance -> Storage System Gate -> Checkpoint Lifecycle Audit -> Checkpoint Object Completeness -> Shard Layout Fit -> Async Save Overlap -> Write Bandwidth Fit -> Metadata Commit Integrity -> Checksum Manifest Validation -> Restore Replay Readiness -> Dataloader RNG State Capture -> Distributed Rank State Capture -> Retention Policy Fit -> Lifecycle Cost Governance -> Cross-Region Replication Fit -> Security Access Control -> Resume SLO Tracking -> Failure Drill Coverage -> Checkpoint Lifecycle Gate -> Checkpoint Strategy Audit -> RPO RTO Budget -> Save Trigger Policy -> Checkpoint Scope Separation -> Async Snapshot Consistency -> Manifest Commit Integrity -> Restore Selection Policy -> Preemption Checkpoint Coupling -> Retention Tier Lifecycle -> Storage Cost Budget -> Checkpoint Strategy Gate -> Container Environment Audit -> Image Digest Reproducibility -> Base Layer Cache Fit -> CUDA Driver Framework Compatibility -> GPU Runtime Visibility -> Dependency Lock Coverage -> Training Serving Image Separation -> Image Size Startup Fit -> Build Pipeline Smoke Test -> Security Scan Signing -> Secret Data Exclusion -> Runtime Hardening Fit -> Registry Access Governance -> Environment Metadata Capture -> Multi-Tenant Mount Isolation -> NCCL RDMA Runtime Fit -> Container Environment Gate -> GPU Cluster -> Network Fabric -> Storage / Checkpoint -> Scheduler / Resource Manager -> Training Platform -> Inference Platform -> Data Platform -> Artifact Registry -> Eval Platform -> Observability / Security / Cost Governance -> Platform Engineering

AI Infra Kubernetes GPU resource management subchain: Container Environment Gate -> Kubernetes GPU Resource Management Audit -> Device Plugin Readiness -> GPU Extended Resource Fit -> GPU Request Limit Integrity -> Gang Scheduling Readiness -> Fragmentation Control -> Topology Aware Placement -> Node Label Affinity Fit -> Taint Toleration Fit -> MIG Sharing Policy Fit -> Quota Namespace Governance -> Training Operator Readiness -> Inference Service Readiness -> GPU Monitoring Mapping -> Multi-Tenant Isolation -> Pending Troubleshooting Coverage -> Kubernetes GPU Gate -> Scheduler / Resource Manager

AI Infra training scheduler governance subchain: Kubernetes GPU Gate -> Training Scheduler Governance Audit -> Workload Resource Shape -> Queue Policy Coverage -> Priority Governance -> Quota Usage Control -> Fair Share Accounting -> Gang Scheduling Fit -> Backfilling Safety -> Preemption Checkpoint Safety -> Scheduler Fragmentation Control -> Scheduler Topology Awareness -> Checkpoint Scheduler Coupling -> Cost Attribution Control -> Scheduler Observability -> Failure Requeue Readiness -> Anti Starvation Control -> Training Scheduler Gate -> Multi-Tenant Isolation

AI Infra multi-tenant isolation subchain: Training Scheduler Gate -> Multi-Tenant Isolation Audit -> Resource Quota Isolation -> Identity Namespace Binding -> RBAC ABAC Permission Fit -> Data Access Boundary -> Data Classification Lineage -> Network Policy Isolation -> Runtime Security Boundary -> Image Supply Chain Boundary -> Secret Scope Rotation -> Logs Trace Redaction -> Cost Attribution Coverage -> Prod Experiment Separation -> Cross Tenant Sharing Governance -> Audit Evidence Readiness -> Blast Radius Control -> Multi-Tenant Isolation Gate -> Cluster Capacity Planning Audit

AI Infra cluster capacity planning subchain: Multi-Tenant Isolation Gate -> Cluster Capacity Planning Audit -> Workload Forecast Coverage -> GPU Count Capacity Fit -> GPU Type Mix Fit -> Training Queue SLO Fit -> Serving Peak SLO Fit -> Network Bandwidth Capacity Fit -> Storage Throughput Capacity Fit -> Storage Capacity Lifecycle Fit -> Utilization Headroom Fit -> Failure Redundancy Fit -> Growth Forecast Fit -> Cost Budget Fit -> Pool Separation Fit -> Quota Burst Governance -> Observability Forecast Feedback -> Cluster Capacity Planning Gate -> Training Platform Lifecycle Audit

AI Infra training platform lifecycle subchain: Cluster Capacity Planning Gate -> Training Platform Lifecycle Audit -> TrainingJob Contract -> Config Validation -> Resource Quota Binding -> Image Code Reproducibility -> Dataset Lineage Permission -> Launcher Distributed Fit -> Checkpoint Resume Policy -> Observability Events Metrics -> Experiment Tracking Lineage -> Artifact Registry Linkage -> Failure Recovery Classification -> Security Audit Control -> Cost Attribution Control -> Developer Self Service -> Lifecycle State Machine -> Training Platform Gate -> Training Submission Audit -> TrainingJob Schema Completeness -> Required Field Coverage -> Image Digest Binding -> Code Commit Diff Binding -> Dataset Version Permission -> Structured Command Safety -> Final Config Snapshot -> Distributed Resource Consistency -> Checkpoint URI Resume Validation -> Logging Metrics Destination -> Queue Priority Policy Binding -> Quota Dry Run Admission -> Idempotency Duplicate Control -> Template Version Governance -> Submit State Machine Validity -> Submission Audit Trace -> Training Submission Gate -> Distributed Launcher Audit -> Resource World Size Consistency -> Rank Local Rank Mapping -> Rendezvous Endpoint Readiness -> GPU Binding Visibility -> Launcher Adapter Fit -> Torchrun Argument Fit -> DeepSpeed Config Fit -> Megatron Parallel Consistency -> Ray Runtime Fit -> Network NCCL Readiness -> Log Rank Aggregation -> Failure Stage Classification -> Elastic Training Safety -> Checkpoint Launcher Coupling -> Scheduler Launcher Handoff -> Launcher Audit Trace -> Distributed Launcher Gate -> Training Config Management Audit -> Config Source Order -> Final Config Snapshot Freeze -> Required Config Field Coverage -> Config Schema Type Validation -> Batch Parallel Consistency -> Immutable Version Binding -> Environment Version Capture -> Reproducibility Seed Boundary -> Experiment Tracking Linkage -> Hyperparameter Sweep Trace -> Checkpoint Resume Compatibility -> Release Config Coupling -> Config Permission Approval -> Config Diff Readiness -> Config Audit Trace -> Training Config Gate -> Training Observability Audit -> Structured Log Context -> Rank Log Coverage -> Metric Taxonomy Coverage -> Metric Dimension Scope -> Event Schema Lifecycle -> State Machine Validity -> Attempt Retry Traceability -> Trace Phase Coverage -> Dashboard Diagnostic Readiness -> Alert Rule Readiness -> Anomaly Detection Rules -> Experiment Tracking Linkage -> Retention Cost Governance -> Privacy Redaction Access -> Cost Attribution Signal -> Training Observability Gate -> Checkpoint Strategy Audit -> RPO RTO Budget -> Save Trigger Policy -> Checkpoint Scope Separation -> Async Snapshot Consistency -> Manifest Commit Integrity -> Shard Metadata Resharding -> Restore Selection Policy -> Preemption Checkpoint Coupling -> Evaluation Best Linkage -> Release Checkpoint Readiness -> Retention Tier Lifecycle -> Storage Cost Budget -> Checkpoint Monitoring Alerts -> Restore Drill Coverage -> Permission Release Protection -> Checkpoint Strategy Gate -> Training Platform

AI Infra training fault tolerance subchain: Checkpoint Strategy Gate -> Training Fault Tolerance Audit -> Fault Taxonomy Coverage -> Retryability Classification -> Retry Budget Bounded -> Retry Backoff Policy -> Checkpoint Resume Readiness -> Node Failure Reschedule -> GPU Health Isolation -> Communication Hang Detection -> OOM Handling Policy -> Data Permission Fast Fail -> Checkpoint Commit Safety -> Numeric Instability Stop Rollback -> Diagnosis Report Evidence -> Blacklist Precision -> User Action Clarity -> Training Fault Tolerance Gate -> Training Platform

AI Infra training data supply subchain: Training Fault Tolerance Gate -> Training Data Supply Audit -> Dataset Manifest Integrity -> Shard Format Size Fit -> Small File Amplification Control -> Streaming Prefetch Retry -> Local Cache Policy -> Distributed Cache Policy -> Shuffle Reproducibility -> Rank Shard Assignment -> Data Locality Fit -> Dataloader Parallelism Fit -> H2D Copy Readiness -> Bad Sample Threshold -> Data Supply Observability -> Checkpoint Dataloader State -> Permission Version Governance -> Training Data Supply Gate -> Training Platform

AI Infra training platform security subchain: Training Data Supply Gate -> Training Platform Security Audit -> Identity Runtime Binding -> RBAC ABAC Policy Fit -> Dataset Permission Purpose -> Artifact Classification Propagation -> Resource Quota Priority Governance -> Image Supply Chain Trust -> Runtime Code Isolation -> Secret Scope Rotation -> Network Egress Control -> Log Redaction Access -> Model Artifact Permission -> High Risk Approval -> Audit Log Completeness -> Audit Integrity Retention -> Incident Response Readiness -> Training Platform Security Gate -> Training Platform

AI Infra training platform system design subchain: Training Platform Security Gate -> Training Platform System Design Audit -> Requirement Clarification -> TrainingJob Contract -> Lifecycle State Machine -> Admission Quota Policy -> Scheduler Resource Fit -> Distributed Launcher Fit -> Data Config Lineage -> Checkpoint Recovery Coupling -> Observability Experiment Linkage -> Fault Recovery Flow -> Security Audit Governance -> Cost Capacity Governance -> Developer Self Service -> Interface Artifact Contract -> Tradeoff Boundary Reasoning -> Training Platform Design Gate -> Training Platform

AI Infra inference platform overview subchain: Training Platform Design Gate -> Inference Platform Overview Audit -> Request Lifecycle Coverage -> SLO Metric Contract -> Model Registry Readiness -> Model Router Policy Fit -> Runtime Prefill Decode Fit -> Continuous Batching Readiness -> KV Cache Capacity Governance -> Autoscaling Warm Pool Fit -> Release Rollback Control -> Rate Limit Degrade Protection -> Safety Governance Fit -> Cost Attribution Optimization -> Observability Trace Coverage -> Multi Model Governance -> Developer API Self Service -> Inference Platform Gate -> Inference Platform

AI Infra model serving runtime selection subchain: Inference Platform Gate -> Model Serving Runtime Selection Audit -> Model Hardware Fit -> Runtime Boundary Clarity -> Model Format Tokenizer Fit -> Prefill Decode Scheduler Fit -> Continuous Batching Fit -> KV Cache Memory Manager Fit -> Streaming API Fit -> Quantization Accuracy Fit -> Distributed Inference Fit -> Observability Metrics Fit -> Benchmarking Method Fit -> Deployment Rollback Fit -> Ecosystem Maintenance Fit -> Cost Capacity Fit -> Self Development Threshold -> Runtime Selection Gate -> Inference Platform

AI Infra inference resource profile subchain: Runtime Selection Gate -> Prefill Decode KV Resource Profile Audit -> Request Token Profile -> Prefill Phase Accounting -> Decode Phase Accounting -> TTFT TPOT Contract -> KV Cache Formula Fit -> KV Capacity Admission -> Paged Block Management -> Prefix Cache Reuse -> Long Context Policy -> Tenant KV Isolation -> Continuous Batching Policy -> PD Disaggregation Fit -> Streaming Backpressure Fit -> Observability Phase Metrics -> Cost Capacity Model -> Resource Profile Gate -> Inference Platform

AI Infra inference scheduling subchain: Resource Profile Gate -> Inference Scheduling Audit -> Request Arrival Profile -> Continuous Batching Core -> Token Budget Policy -> Prefill Decode Balance -> Paged KV Block Management -> Scheduler KV Admission -> Long Short Isolation -> Tenant Priority Fairness -> Cancellation Cleanup -> Streaming Backpressure Fit -> Scheduler Observability Metrics -> Scheduler Gate -> Inference Platform

AI Infra model routing subchain: Scheduler Gate -> Model Routing Audit -> Request Intent Profile -> Model Capability Profile -> Permission Candidate Filter -> Cost Budget Policy -> Latency SLO Policy -> Quality Safety Policy -> Fallback Chain Governance -> Canary Routing Stability -> Realtime Health Load -> Route Trace Coverage -> Degrade Policy Control -> Route Config Versioning -> Model Routing Gate -> Inference Platform

AI Infra inference cache system subchain: Model Routing Gate -> Inference Cache System Audit -> Request Cache Profile -> Cache Layer Boundary -> Key Version Fingerprint -> Prompt Prefix Reuse -> KV Lifecycle Management -> Semantic Similarity Guard -> Result Cache Determinism -> Tenant Permission Isolation -> TTL Staleness Control -> Eviction Quota Policy -> Streaming Cache Policy -> Cache Observability Metrics -> Cost Latency Savings -> Cache Governance Gate -> Inference Platform

AI Infra inference autoscaling subchain: Cache Governance Gate -> Inference Autoscaling Audit -> Traffic Token Profile -> SLO Latency Contract -> Queue Backlog Signal -> GPU KV Capacity Signal -> Cold Start Lead Time -> Warm Pool Readiness -> Multi Metric Recommendation -> Scale Up Down Policy -> Draining Streaming Safety -> Router Admission Coupling -> Tenant Quota Priority -> Cost Budget Guard -> Autoscaling Trace Coverage -> Autoscaling Gate -> Inference Platform

AI Infra inference protection policy subchain: Autoscaling Gate -> Inference Protection Policy Audit -> Request Token Rate Limit -> Concurrency Quota Control -> Admission Capacity Guard -> Circuit Breaker State Machine -> Retry Error Classification -> Retry Budget Bound -> Idempotency Stage Guard -> Timeout Budget Allocation -> Degradation Policy Control -> Priority Tenant Fairness -> Streaming Cancellation Cleanup -> Protection Trace Coverage -> Protection Gate -> Inference Platform

AI Infra inference release governance subchain: Protection Gate -> Inference Release Governance Audit -> Release Bundle Trace -> Model Artifact Integrity -> Tokenizer Prompt Runtime Binding -> Offline Eval Quality Gate -> Safety Eval Gate -> Contract Test Compatibility -> Stable Bucket Assignment -> Canary Ramp Control -> AB Experiment Design -> Sample Ratio Health -> Guardrail Metric Control -> Rollback Readiness -> Release Audit Trace -> Post Release Monitoring -> Release Governance Gate -> Inference Platform

AI Infra inference platform system design subchain: Release Governance Gate -> Inference Platform System Design Audit -> Requirement Clarification -> Request Lifecycle Coverage -> SLO Token Contract -> Model Registry Release -> Router Policy Fit -> Runtime Prefill Decode -> KV Scheduler Capacity -> Cache Isolation Strategy -> Autoscaling Capacity Plan -> Protection Policy Design -> Release Governance Design -> Tenant Security Governance -> Observability Trace Readiness -> Cost Capacity Governance -> Tradeoff Boundary Reasoning -> Inference Platform Design Gate -> Inference Platform

AI Infra data platform supply chain subchain: Inference Platform Design Gate -> Data Platform Supply Chain Audit -> Source Registry Coverage -> Raw Data Lake Integrity -> Cleaning Policy Versioning -> Dedup Contamination Guard -> PII Redaction Coverage -> Quality Scoring Calibration -> Dataset Builder Recipe -> Dataset Manifest Completeness -> Dataset Version Immutability -> Streaming Shard Readiness -> Permission Purpose Binding -> Data Lineage Coverage -> Data Platform Supply Observability -> Read Throughput Fit -> Data Wait Fit -> Data Platform Supply Gate -> Data Platform

AI Infra data version lineage quality subchain: Data Platform Supply Gate -> Data Version Lineage Quality Audit -> Dataset Version Contract -> Manifest Checksum Integrity -> Immutable Snapshot Policy -> Delta Version Diff -> Lineage Graph Completeness -> Impact Analysis Readiness -> Schema Quality Gate -> Content Quality Gate -> Data Safety Compliance Gate -> Distribution Drift Monitoring -> Train Eval Leakage Monitoring -> Annotation Quality Monitoring -> Synthetic Data Monitoring -> Quality Alert Routing -> Training Model Version Linkage -> Data Version Quality Gate -> Data Platform

AI Infra model artifact registry subchain: Data Version Quality Gate -> Model Artifact Registry Audit -> Model Version Contract -> Weight Manifest Integrity -> Safe Weight Format -> Tokenizer Config Binding -> Adapter Base Compatibility -> Merge Lineage Completeness -> Quantization Eval Gate -> Runtime Compatibility Fit -> Eval Report Linkage -> Safety Permission Gate -> Release Status Governance -> Rollback Alias Readiness -> Artifact Lineage Completeness -> Load Cache Readiness -> Lifecycle Retention Policy -> Model Registry Gate -> Artifact Registry

AI Infra artifact management subchain: Model Registry Gate -> Artifact Management Audit -> Artifact Metadata Contract -> Dataset Artifact Manifest -> Checkpoint Artifact Recoverability -> Eval Report Reproducibility -> Deployment Package Completeness -> Release Manifest Integrity -> Checksum Integrity Gate -> Artifact Lineage Completeness -> Artifact Store Metadata Split -> Permission Access Control -> Lifecycle Retention Policy -> Promotion Gate Readiness -> Experiment Tracking Linkage -> Rollback Artifact Readiness -> Deletion Dependency Safety -> Artifact Management Gate -> Eval Platform

AI Infra experiment tracking subchain: Artifact Management Gate -> Experiment Tracking Audit -> Run Metadata Contract -> Parameter Config Capture -> Metric Curve Step Capture -> Log Run Linkage -> Sample Result Diagnosis -> Code Version Reproducibility -> Data Version Reproducibility -> Model Checkpoint Linkage -> Environment Capture -> Prompt Eval Config Capture -> Artifact Output Linkage -> Cost Attribution -> Run State Transition -> Search Index Readiness -> Lineage Graph Readiness -> Decision Note Capture -> Experiment Tracking Gate -> Eval Platform

AI Infra evaluation platform subchain: Experiment Tracking Gate -> Evaluation Platform Audit -> Eval Dataset Contract -> Offline Batch Reproducibility -> Metric Definition Versioning -> LLM Judge Calibration -> Human Review Quality Control -> Pairwise Blinding Randomization -> Online Eval Guardrail -> Release Gate Readiness -> Eval Report Completeness -> Slice Regression Detection -> Regression Suite Coverage -> Eval Job Scheduler Reliability -> Eval Cache Key Integrity -> Eval Observability -> Artifact Tracking Linkage -> Evaluation Platform Gate -> Feature Store / Embedding Store

AI Infra feature embedding index subchain: Evaluation Platform Gate -> Feature Embedding Index Infrastructure Audit -> Feature Definition Contract -> Offline Online Consistency -> Point in Time Correctness -> Embedding Version Contract -> Chunk Embedding Lineage -> Vector Index Build Readiness -> ANN Quality Latency Gate -> Metadata Permission Filter -> Shadow Index Switch Readiness -> Retrieval Trace Completeness -> Feature Embedding Lineage Graph -> Multi Tenant Isolation -> Quality Monitoring Metrics -> Cost Capacity Governance -> RAG Agent Integration -> Feature Embedding Index Gate -> RAG / Agent Platform

AI Infra RAG agent storage subchain: RAG / Agent Platform -> RAG Agent Storage Audit -> Knowledge Base Contract -> Document Chunk Version Contract -> Sync Delete Propagation -> ACL Permission Enforcement -> Retrieval Trace Completeness -> Prompt Assembly Trace -> Citation Version Binding -> Agent Definition Versioning -> Tool Definition Contract -> Tool Permission Gate -> Tool Call Trace Completeness -> Execution Trace Replay Readiness -> Memory Privacy Lifecycle -> Trace Privacy Retention -> Cost Attribution Governance -> RAG Agent Storage Gate -> AI Observability Platform

AI Infra observability subchain: AI Observability Platform -> AI Infra Observability Audit -> Signal Inventory Coverage -> Metric Contract Completeness -> SLO Error Budget Readiness -> Latency Quantile Guard -> Trace Span Coverage -> Log Event Structure -> Correlation ID Coverage -> Cardinality Budget Control -> Training Observability -> Inference Observability -> Data Quality Observability -> RAG Agent Observability -> Cost Observability -> Privacy Redaction Retention -> Alert Actionability -> Observability Platform Gate -> Training Fault Diagnosis

AI Infra training fault diagnosis subchain: Observability Platform Gate -> Training Fault Diagnosis Audit -> Training Fault Evidence Coverage -> Lifecycle Event Timeline -> Rank Log Completeness -> Loss Anomaly Diagnosis -> NaN Inf Guard -> Update Health Check -> Hang Straggler Detection -> OOM Phase Memory Evidence -> Communication Bottleneck Diagnosis -> IO Data Bottleneck Diagnosis -> Checkpoint Integrity -> Training Resume Continuity -> Data Fault Isolation -> Config Code Diff Readiness -> Minimal Reproduction Readiness -> Training Fault Diagnosis Gate -> Inference Fault Diagnosis

AI Infra inference fault diagnosis subchain: Training Fault Diagnosis Gate -> Inference Fault Diagnosis Audit -> Inference Fault Evidence Coverage -> Request Scope Slice -> Trace Stage Coverage -> TTFT Decomposition -> TPOT Decode Health -> Tail Latency Attribution -> Throughput Token Capacity -> Error Taxonomy Timeout Stage -> KV Cache Pressure Guard -> Model Artifact Load Readiness -> Streaming Reliability -> Cache Hit Key Governance -> Route Trace Correctness -> Tool Dependency Trace -> Quality Release Diff Readiness -> Inference Fault Diagnosis Gate -> SLO Oncall Audit

AI Infra SLO oncall subchain: Inference Fault Diagnosis Gate -> SLO Oncall Audit -> SLI Contract Completeness -> SLO Target Measurability -> SLA Boundary Clarity -> Inference SLO Coverage -> Training SLO Coverage -> Data Eval SLO Coverage -> Error Budget Accounting -> Burn Rate Alerting -> Alert Actionability -> Oncall Ownership Escalation -> Incident Severity Routing -> Runbook Executability -> Change Event Linkage -> Mitigation Rollback Authority -> Postmortem Action Closure -> SLO Cost Tradeoff Gate -> AI Infra Cost Governance Audit

AI Infra cost governance subchain: SLO Cost Tradeoff Gate -> AI Infra Cost Governance Audit -> Usage Metering Coverage -> Cost Attribution Labels -> GPU Cost Efficiency -> Training Waste Control -> Inference Unit Cost -> Cache Savings Accounting -> Storage Lifecycle Governance -> Network Egress Governance -> Artifact Dependency Safety -> Budget Quota Enforcement -> Cost Anomaly Alerting -> Tenant Model Chargeback -> Dashboard Drilldown Readiness -> Optimization Recommendation Trace -> SLO Quality Cost Tradeoff -> Cost Governance Gate -> Resource Utilization Optimization Audit

AI Infra resource utilization subchain: Cost Governance Gate -> Resource Utilization Optimization Audit -> Resource Metric Contract -> Effective Utilization Accounting -> Fragmentation Control -> Topology Binpacking Readiness -> Colocation SLO Guard -> Preemption Checkpoint Safety -> Low Priority Reclaim Policy -> Elastic Training Safety -> Inference Capacity Utilization -> Warm Pool Headroom Policy -> Quota Fairness Borrowing -> Scheduler Observability -> Dashboard Drilldown Readiness -> Reliability Headroom Guard -> Cost SLO Tradeoff Control -> Resource Utilization Gate -> AI Infra Security Governance Audit

AI Infra security governance subchain: Resource Utilization Gate -> AI Infra Security Governance Audit -> Identity Workload Binding -> RBAC ABAC Policy Fit -> Secret Lifecycle Management -> Data Compliance Classification -> PII Sensitive Data Control -> Tenant Isolation Boundary -> Model Access Control -> Model Output Safety Gate -> RAG Permission Enforcement -> Agent Tool Safety Gate -> Supply Chain Integrity -> Model Artifact Integrity -> Runtime Security Boundary -> Log Trace Privacy Governance -> Audit Incident Response Readiness -> Security Governance Gate -> Audit Change Governance Audit

AI Infra audit change governance subchain: Security Governance Gate -> Audit Change Governance Audit -> Audit Event Schema Completeness -> Critical Operation Audit Coverage -> Audit Log Integrity Retention -> Log Retention Policy Fit -> Log Minimization Redaction -> Trace Privacy Sampling -> Incident Timeline Completeness -> Impact Detection Response Metrics -> Root Cause Systemic Analysis -> Postmortem Action Item Closure -> Change Record Completeness -> High Risk Change Approval -> Rollout Rollback Readiness -> Change Freeze Error Budget Policy -> Audit Postmortem Change Linkage -> Governance Loop Gate -> Training Platform

AI Infra enterprise LLMOps platform subchain: Governance Loop Gate -> Enterprise LLMOps Platform Audit -> LLMOps Application Resource Model -> Model Gateway Policy -> Prompt Registry Versioning -> RAG KB Permission Governance -> Tool Agent Runtime Control -> LLMOps Eval Feedback Loop -> LLMOps Release Manifest -> LLMOps Trace Observability Readiness -> LLMOps Cost Budget Governance -> Security Audit Policy -> Multi Environment Promotion -> Developer Interface Readiness -> Production Monitoring Guardrail -> Tradeoff Boundary Reasoning -> LLMOps Platform Gate -> Multi Tenant GPU Scheduler

AI Infra multi tenant GPU scheduler subchain: LLMOps Platform Gate -> Multi Tenant GPU Scheduler Design Audit -> Tenant Queue Contract -> Quota Borrowing Policy -> Dominant Fairness Accounting -> Gang Scheduling Fit -> Topology Aware Placement -> Scheduler Fragmentation Control -> Preemption Checkpoint Safety -> Low Priority Reclaim -> Inference SLO Isolation -> Heterogeneous GPU Policy -> Resource Snapshot Health -> Placement Explainability -> Cost Utilization Attribution -> Scheduler Observability Audit -> Tradeoff Boundary Reasoning -> Multi Tenant GPU Scheduler Gate -> Scheduler / Resource Manager

AI Infra interview readiness subchain: Multi Tenant GPU Scheduler Gate -> AI Infra Interview Readiness Audit -> AI Infra Interview Rubric -> AI Infra Topic Coverage -> AI Infra Formula Coverage -> AI Infra Demo Evidence Coverage -> AI Infra Risk Coverage -> AI Infra Tradeoff Coverage -> Weak AI Infra Question -> AI Infra Revision Plan -> AI Infra Interview Gate -> AI Infra Future Trends

LLM serving engine overview subchain: AI Infra Future Trends -> LLM Serving Engine Overview Audit -> Serving Engine Request Lifecycle -> Prefill Decode Phase Contract -> KV Cache Footprint Estimate -> Scheduler Batching Readiness -> Streaming State Management -> Engine Metrics Observability -> Runtime Boundary Clarity -> Cost Capacity Model -> Serving Engine Gate -> Mini LLM Serving Engine

From scratch serving engine subchain: Serving Engine Gate -> From Scratch Mini Serving Engine -> Naive Generate Loop -> Toy Tokenizer -> Toy Sampler -> Request State Machine -> Waiting Queue -> Running Set -> Minimal Scheduler -> Prefill Decode Metrics -> Token Streaming Trace -> KV Cleanup Gate -> Minimal Engine Gate -> Continuous Batching

Request lifecycle subchain: Minimal Engine Gate -> Request Lifecycle Audit -> Request Lifecycle Object -> Parameter Validation -> Tokenization -> Waiting Queue -> Scheduler Admission -> Prefill State -> Decode State -> Token Streaming Trace -> Finish Reason Coverage -> Abort Timeout Cleanup -> Lifecycle Metrics Trace -> Request Lifecycle Gate -> Prefill Decode Phase Contract

Prefill decode KV streaming subchain: Request Lifecycle Gate -> Prefill Decode KV Streaming Audit -> Prefill Phase Accounting -> Decode Phase Accounting -> TTFT Steps -> TPOT Steps -> KV Cache Footprint Estimate -> KV Pressure Estimate -> Streaming Backpressure Signal -> Finish Reason Coverage -> Phase Gate -> Request Scheduling

Serving metric cost subchain: Phase Gate -> Serving Metric Cost Audit -> TTFT SLO Gate -> TPOT SLO Gate -> Token Throughput Window -> Active Sequence Capacity -> KV Pressure Estimate -> Cost Per 1k Tokens -> Tail Latency Gate -> Timeout Rate Gate -> Metric Cost Gate -> Cost Capacity Model

Serving boundary subchain: Metric Cost Gate -> Serving Boundary Audit -> Engine Platform Infra Boundary -> Role Coverage -> Interface Contract Coverage -> Metric Handoff Coverage -> Incident Layer Routing -> Misroute Rate -> Serving Boundary Gate -> Request Scheduling

Minimal generate loop subchain: Serving Boundary Gate -> Minimal Generate Loop Audit -> Tokenizer Contract -> Model Wrapper Contract -> Last Logits Selection -> Greedy Select -> Append Token -> Stop Condition -> Naive Recompute Work -> Minimal Generate Gate -> Sampling Strategy

Sampling strategy subchain: Minimal Generate Gate -> Sampling Strategy Audit -> Stable Softmax -> Temperature Softmax -> Greedy Argmax -> Top-k Candidate Filter -> Top-p Nucleus Filter -> Renormalization -> Seeded Multinomial Sampling -> Sampling Gate -> KV Cache

Minimal KV cache subchain: Sampling Gate -> Minimal KV Cache Audit -> Past Key Values Contract -> Prefill Cache Build -> Decode Cache Append -> KV Cache Equivalence Check -> KV Cache Memory Formula -> KV Cache Gate -> Batched Prefill

Batched prefill subchain: KV Cache Gate -> Batched Prefill Audit -> Padding Side Policy -> Attention Mask Contract -> Last Real Token Logits -> Padding Waste Ratio -> Prefill Token Budget -> Batched Prefill Gate -> Batched Decode

Batched decode subchain: Batched Prefill Gate -> Batched Decode Audit -> Finished Mask -> Active Batch Compaction -> Batch Cache Alignment -> Decode Position Tracking -> Decode Row Savings -> Batched Decode Gate -> Request Scheduling

Simple scheduler subchain: Batched Decode Gate -> Simple Scheduler Audit -> Waiting Queue -> Running Set -> Scheduler Admission -> Max Active Sequences -> Scheduler Token Budget -> Queue Wait Metric -> TTFT Steps -> Scheduler Trace -> Simple Scheduler Gate -> Token Streaming

Token streaming subchain: Simple Scheduler Gate -> Token Streaming Audit -> Streaming Event Contract -> Incremental Detokenization -> First Token Chunk -> Finish Event Contract -> Stop Sequence Buffer -> Client Cancellation Cleanup -> Streaming Backpressure -> Streaming Gate -> Minimal HTTP API

Minimal HTTP API subchain: Streaming Gate -> Minimal HTTP API Audit -> Generate Request Contract -> Request Validation Gate -> Queue Admission Control -> Sync Generate Response -> SSE Frame Format -> Streaming Generate Response -> HTTP Cancellation Cleanup -> HTTP API Gate -> Serving Benchmark Audit

Serving benchmark subchain: HTTP API Gate -> Serving Benchmark Audit -> TTFT Metric -> TPOT Metric -> E2E Latency Metric -> Token Throughput Report -> Queue Length Percentile -> Active Request Percentile -> KV Memory Peak -> Benchmark Bottleneck Classification -> Benchmark Gate -> vLLM Motivation

vLLM motivation subchain: Benchmark Gate -> vLLM Motivation Audit -> Naive KV Reservation -> Paged KV Block Allocation -> KV Waste Ratio -> Logical Physical Block Mapping -> Block Reuse After Cleanup -> Static Decode Rows -> Continuous Decode Rows -> vLLM Motivation Gate -> PagedAttention

PagedAttention core subchain: vLLM Motivation Gate -> PagedAttention Core Audit -> Block Table Address Translation -> Logical Block -> Physical KV Block -> Block Size Tradeoff -> Internal Block Waste -> Prefix Block Ref Count -> Freed Block Reuse -> PagedAttention Gate -> KV Cache Block Manager

KV block manager subchain: PagedAttention Gate -> KV Block Manager Audit -> Block Free List -> Block Reference Count -> Prefill Block Admission -> Decode Block Extension -> Shared Prefix Cleanup -> Block Allocation Failure -> Block Manager Metrics -> Block Manager Gate -> Continuous Batching

Continuous batching subchain: Block Manager Gate -> Continuous Batching Audit -> Iteration Level Scheduling -> Dynamic Request Join -> Dynamic Request Exit -> Decode First Scheduling -> Continuous Token Budget -> Scheduler KV Budget -> Deferred Scheduling Reason -> Static Batch Hole -> Continuous Batching Gate -> vLLM Scheduler Flow

vLLM scheduler flow subchain: Continuous Batching Gate -> vLLM Scheduler Flow Audit -> Engine Request Contract -> Request State Machine Trace -> Scheduler Output Metadata -> Model Runner Execution Metadata -> Slot Mapping Metadata -> Output Processor Finish Gate -> Abort Cleanup Path -> Request Flow Metrics -> Request Flow Gate -> vLLM Memory Management

vLLM memory management subchain: Request Flow Gate -> vLLM Memory Management Audit -> GPU Memory Breakdown -> KV Page Size -> Prefix Cache Hash -> Cached Free Block -> Prefix Cache Hit Rate -> LRU Cache Eviction -> Preemption Recompute Path -> Hybrid KV Cache Group -> Memory Management Gate -> vLLM Worker Executor

vLLM worker executor subchain: Memory Management Gate -> vLLM Executor Architecture Audit -> Engine Core Dispatch Loop -> Executor Dispatch Layer -> Worker Rank Mapping -> Model Runner Metadata Gate -> Tensor Parallel Merge -> CPU Process Budget -> Worker Failure Recovery -> Executor Architecture Gate -> Prefix Caching Audit

Prefix caching subchain: Executor Architecture Gate -> Prefix Caching Audit -> Full Block Prefix Reuse -> Parent Hash Chain -> Extra Hash Isolation -> Cache Salt Isolation -> Cached Free Touch -> Prefix Saved Prefill Tokens -> Prefix Cache Eviction -> Prefix Cache Gate -> Serving Parallelism Audit

Serving parallelism subchain: Prefix Cache Gate -> Serving Parallelism Audit -> Tensor Parallel Topology -> Pipeline Parallel Bubble -> Data Parallel Replica -> Expert Parallel Placement -> Worker Group Size -> Per Rank Memory Fit -> Cross Node TP Risk -> Prefix Cache Locality -> Parallel Serving Gate -> vLLM Performance Tuning Audit

vLLM performance tuning subchain: Parallel Serving Gate -> vLLM Performance Tuning Audit -> TTFT Bottleneck Diagnosis -> TPOT Bottleneck Diagnosis -> KV Pressure Tuning -> Preemption Rate -> Prefix Cache Effectiveness -> Config Trade-off Plan -> Rollback Guard -> vLLM Tuning Gate -> SGLang Motivation Audit

SGLang motivation subchain: vLLM Tuning Gate -> SGLang Motivation Audit -> Complex LLM Program -> Frontend Language -> RadixAttention Motivation -> Radix Prefix Reuse -> Branch Sharing Visibility -> Structured Decoding Retry Saving -> OpenAI-compatible API Boundary -> SGLang Motivation Gate -> SGLang Runtime Audit

SGLang runtime subchain: SGLang Motivation Gate -> SGLang Runtime Audit -> Runtime Entrypoint Unification -> Request State Contract -> Radix Prefix Lookup -> KV Memory Pool Budget -> Prefill Decode Scheduler -> Model Runner Boundary -> Grammar Mask Step -> Streaming Event Trace -> SGLang Runtime Gate -> RadixAttention Prefix Sharing Audit

RadixAttention prefix sharing subchain: SGLang Runtime Gate -> RadixAttention Prefix Sharing Audit -> Compressed Prefix Tree -> Longest Prefix Match -> Radix Tree Split -> Page Aligned Prefix Hit -> Radix KV Ref Count -> Leaf LRU Eviction -> Cache Aware Scheduling Cost -> RadixAttention Gate -> SGLang Scheduler Audit

SGLang scheduler subchain: RadixAttention Gate -> SGLang Scheduler Audit -> Decode First Scheduling -> Cache Aware Admission -> Suffix Cost Scheduling -> Scheduler Token Budget -> Scheduler Sequence Budget -> Scheduler KV Admission -> Scheduler Aging Fairness -> Chunked Prefill Scheduling -> Scheduler Grammar Cost -> Scheduler Abort Cleanup -> SGLang Scheduler Gate -> Structured Generation Audit

Structured generation subchain: SGLang Scheduler Gate -> Structured Generation Audit -> Grammar State -> Valid Token Mask -> JSON Schema Constraint -> Regex Choice Constraint -> EBNF Grammar Constraint -> Structural Tag Constraint -> Empty Valid Token Set -> Format Not Fact Gate -> Structured Streaming Partial -> Structured Generation Gate -> Speculative Decoding Audit

Speculative decoding subchain: Structured Generation Gate -> Speculative Decoding Audit -> Draft Source -> Target Verify Call -> Accept Length -> Draft Acceptance Rate -> Target Call Reduction -> Speculative Fallback Token -> Speculative KV Cleanup -> Adaptive Speculative Steps -> Speculative Decoding Gate -> Agent Serving Audit

Agent serving subchain: Speculative Decoding Gate -> Agent Serving Audit -> Multi-turn Runtime State -> Session Aware Routing -> Agent Trajectory Prefix Sharing -> Tool Parser Fit -> Tool Schema Validation -> Tool Result Backfill -> Tool Wait GPU Release -> Agent Round Budget -> Agent Serving Gate -> Serving Architecture Comparison Audit

Serving architecture comparison subchain: Agent Serving Gate -> Serving Architecture Comparison Audit -> Common Serving Core -> Paged Prefix Cache Evidence -> Radix Program Reuse Evidence -> Workload Fit Score -> Runtime Workload Router -> vLLM Not Replaced Gate -> SGLang Program Fit Gate -> Architecture Comparison Gate -> mini-sglang Source Path Audit

mini-sglang source path subchain: Architecture Comparison Gate -> mini-sglang Source Path Audit -> Runtime Module Map -> Request Lifecycle Order -> Source Resource Coverage -> Source Experiment Coverage -> Source Observable Signal -> Radix Split Experiment -> Abort Cleanup Experiment -> Source Path Gate -> Prefill Decode Resource Profile Audit

Prefill decode resource profile subchain: Source Path Gate -> Prefill Decode Resource Profile Audit -> Request Token Profile -> Run Prefill Tokens -> Prefill Compute Profile -> Prefill KV Write -> Decode KV Read -> TTFT TPOT Separation -> Long Prefill Stall Risk -> Decode-first Starvation Risk -> PD KV Transfer Cost -> Prefill Decode Profile Gate -> PD Disaggregation Motivation Audit

PD disaggregation motivation subchain: Prefill Decode Profile Gate -> PD Disaggregation Motivation Audit -> Unified Engine Interference -> Prefill Interruption Evidence -> Decode-first Starvation Evidence -> Independent P/D Scaling -> KV Transfer Cost Gate -> Slow Transfer Counterexample -> PD Motivation Gate -> PD Architecture Audit

PD architecture subchain: PD Motivation Gate -> PD Architecture Audit -> PD Router -> Prefill Worker Pool -> Decode Worker Pool -> KV Transfer Backend -> Bootstrap Metadata -> Decode Reservation -> Cross Component State Machine -> Transfer Failure Cleanup -> Client Abort Cleanup -> Model Version Compatibility -> PD Architecture Gate -> KV Transfer Routing Audit

KV transfer routing subchain: PD Architecture Gate -> KV Transfer Routing Audit -> KV Metadata Compatibility -> Decode Capacity Reservation -> KV Routing Cost Score -> Tenant Cache Isolation -> Recompute Fallback Path -> Transfer Failure Plan -> KV Transfer Routing Gate -> Chunked Disaggregated Prefill Audit

Chunked disaggregated prefill subchain: KV Transfer Routing Gate -> Chunked Disaggregated Prefill Audit -> Run Prefill Tokens -> Prefill Chunk Count -> Decode Interleaving -> Prefill Token Budget -> Position Continuity -> Short Request Fairness -> PD Transfer Pipeline -> Backpressure Cleanup -> Chunked Disagg Prefill Gate -> Multi Level KV Cache Audit

Multi level KV cache subchain: Chunked Disagg Prefill Gate -> Multi Level KV Cache Audit -> KV Residency Level -> Active GPU Block Protection -> CPU KV Promote -> Remote KV Fetch -> Recompute Fallback -> KV Tenant Isolation -> KV Demotion Policy -> Residency Metrics -> Multi Level KV Gate -> Cross Node Network Audit

Cross node network subchain: Multi Level KV Gate -> Cross Node Network Audit -> Cross Node Link Profile -> TP Cross Node Risk -> Pipeline Activation Cost -> PD KV Transfer Cost -> Remote Recompute Decision -> Topology Aware Routing -> Transfer Backpressure -> Control Data Plane Separation -> Cross Node Network Gate -> PD Tradeoff Audit

PD tradeoff subchain: Cross Node Network Gate -> PD Tradeoff Audit -> PD Benefit Cost Score -> PD Positive Case -> Short Request Anti Pattern -> Decode Bottleneck Anti Pattern -> Slow Transfer Anti Pattern -> Ops Readiness Gate -> PD Alternative Path -> PD Tradeoff Gate -> Single Engine To PD Migration Audit

Single engine to PD migration subchain: PD Tradeoff Gate -> Single Engine To PD Migration Audit -> Explicit Request Stage -> Model Runner Interface Split -> KV Metadata Contract -> Prefill Decode Scheduler Split -> PD Router State Machine -> Same Node PD Prototype -> Cleanup Cancel Path -> Observability Fallback Readiness -> PD Migration Gate -> nano-vLLM Source Learning Audit

nano-vLLM source learning subchain: PD Migration Gate -> nano-vLLM Source Learning Audit -> Source Module Map -> Generate Trace Path -> Engine Step Trace -> Sequence State Audit -> Scheduler Decision Trace -> KV Block Lifecycle -> Model Runner Batch Contract -> Source Experiment Signal -> nano-vLLM Source Gate -> tiny-LLM Learning Audit

tiny-LLM learning subchain: nano-vLLM Source Gate -> tiny-LLM Learning Audit -> Operator To Model Path -> Attention Shape Trace -> RoPE Position Alignment -> GQA KV Head Audit -> Generate Sampling Loop -> KV Cache Equivalence -> Serving Optimization Ladder -> Chunk Position Continuity -> tiny-LLM Learning Gate -> mini-sglang Learning Audit

mini-sglang learning subchain: tiny-LLM Learning Gate -> mini-sglang Learning Audit -> SGLang Runtime Capability -> Radix Cache Runtime Evidence -> Chunked Prefill Ablation -> Overlap Scheduling Ablation -> Online Serving Trace -> Structured Tool Boundary -> Abort Cleanup Check -> Production Gap Map -> mini-sglang Learning Gate -> Teaching Project Core Module Audit

Teaching project core module subchain: mini-sglang Learning Gate -> Teaching Project Core Module Audit -> Core Module Boundary -> Request State Ownership -> Scheduler ModelRunner Boundary -> KV Manager Ownership -> BatchBuilder Metadata Contract -> Output Processor Cleanup Boundary -> Metrics Observability Contract -> Replaceable Upgrade Path -> Core Module Gate -> Naive Scheduler To Continuous Batching Audit

Naive scheduler to continuous batching subchain: Core Module Gate -> Naive Scheduler To Continuous Batching Audit -> Request Level Batch Lock -> Iteration Boundary Scheduling -> Dynamic Request Admission -> Dynamic Request Exit -> Decode First Upgrade -> Bounded Prefill Budget -> KV Capacity Admission -> Scheduler Upgrade Gate -> Paged KV Cache Upgrade Audit

Paged KV cache upgrade subchain: Scheduler Upgrade Gate -> Paged KV Cache Upgrade Audit -> List KV Private Cache -> Global KV Block Pool -> Request Block Table -> Allocate Until Contract -> Paged Slot Mapping -> Decode Block Extension Gate -> Idempotent KV Free -> Double Free Guard -> Paged KV Upgrade Gate -> Prefix Prompt Cache Upgrade Audit

Prefix prompt cache upgrade subchain: Paged KV Upgrade Gate -> Prefix Prompt Cache Upgrade Audit -> Prompt Prefix Cache Boundary -> Full Block Cache Reuse -> Parent Hash Cache Chain -> Extra Hash Cache Isolation -> Suffix Prefill Start Position -> Cached Ref Count Lifecycle -> Full Prompt Hit Fallback -> Prefix Prompt Cache Upgrade Gate -> Preemption Recompute Swap Upgrade Audit

Preemption recompute swap upgrade subchain: Prefix Prompt Cache Upgrade Gate -> Preemption Recompute Swap Upgrade Audit -> KV Pressure Trigger -> Cached Block Eviction Before Preemption -> Victim Selection Policy -> Recompute Context Preservation -> Recompute Resume Path -> Swap Failure Rollback -> Swap In State Restore -> Preemption Upgrade Gate -> Unified Scheduler Loop Audit

Unified scheduler loop subchain: Preemption Upgrade Gate -> Unified Scheduler Loop Audit -> Engine Step Order -> Prefix Lookup Once -> Decode First Commit -> Suffix Prefill Plan -> KV Budget Commit -> Memory Pressure Order -> Batch Metadata Invariant -> Output State Update -> Unified Scheduler Loop Gate -> Serving Benchmark Framework Audit

Serving benchmark framework subchain: Unified Scheduler Loop Gate -> Serving Benchmark Framework Audit -> Benchmark Workload Coverage -> Benchmark Experiment Fingerprint -> Request Trace Summary -> Engine Step Trace Summary -> SLO Regression Gate -> Throughput Regression Check -> KV Cleanup Check -> Prefix Effect Check -> Preemption Risk Check -> Benchmark Decision Gate -> Serving Benchmark Framework Gate -> Async Serving Architecture Audit

Async serving architecture subchain: Serving Benchmark Framework Gate -> Async Serving Architecture Audit -> Async Boundary Design -> API Admission Boundary -> Tokenizer Worker Queue -> Engine Input Queue -> Bounded Engine Drain -> Engine Output Queue -> Per Client Stream Queue -> Stream Queue Backpressure -> Cancel Signal Queue -> Engine State Ownership -> Idempotent Async Cleanup -> Async Serving Architecture Gate -> Multi Worker Router Audit

Multi worker router subchain: Async Serving Architecture Gate -> Multi Worker Router Audit -> Router Worker Capability Filter -> Request Cost Block Estimate -> Load Aware Worker Score -> Sticky Route Fallback -> Global Router Admission -> Worker Heartbeat Timeout -> Safe Retry Boundary -> Request Worker Map -> Worker Selection Imbalance -> Multi Worker Router Gate -> Distributed Parallel KV Audit

Distributed parallel KV subchain: Multi Worker Router Gate -> Distributed Parallel KV Audit -> Parallel Group Boundary -> TP Head Shard Assignment -> TP Block Table Consistency -> PP Layer Ownership -> Distributed KV Byte Accounting -> Collective Communication Cost -> Pipeline Bubble Ratio -> KV Migration Decision -> Cross Rank Cleanup -> Distributed Parallel KV Gate -> OpenAI Compatible API Audit

OpenAI compatible API subchain: Distributed Parallel KV Gate -> OpenAI Compatible API Audit -> API Compatibility Contract -> Chat Request Schema Validation -> SSE Stream Frame Contract -> OpenAI Style Error Shape -> Bearer Auth Gate -> Model Permission Gate -> Token Rate Limit Gate -> Concurrency Limit Gate -> Usage Accounting Check -> API Privacy Log Check -> OpenAI Compatible API Gate -> Production Deployment Audit

Production deployment subchain: OpenAI Compatible API Gate -> Production Deployment Audit -> Runtime Compatibility Matrix -> Model Artifact Manifest -> Startup Readiness Gate -> Warmup Probe Check -> Worker Registration Gate -> Drain Completion Gate -> Rolling Update Capacity Gate -> Canary Traffic Split -> Canary Regression Gate -> Rollback Readiness Gate -> Production Deployment Gate -> Capacity SLO Fault Drill Audit

Capacity SLO fault drill subchain: Production Deployment Gate -> Capacity SLO Fault Drill Audit -> Workload Token Profile -> Benchmark Stable Capacity -> GPU Count Estimate -> KV Concurrency Limit -> Cost Attribution Model -> SLO Error Budget -> Admission Overload Policy -> Fault Drill Scenario -> Runbook Coverage Check -> Capacity SLO Fault Drill Gate -> Inference Engine Project Portfolio

Inference engine interview readiness subchain: Capacity SLO Fault Drill Gate -> Inference Engine Interview Readiness Audit -> Interview Question Rubric -> Concept Coverage Score -> Request Lifecycle Answer -> KV Cache Answer Evidence -> Scheduler Answer Evidence -> Serving Metrics Coverage -> Ordered Debug Path -> Production Governance Answer -> Project Evidence Portfolio -> Tradeoff Coverage -> Inference Engine Interview Gate

### 实战排查链

Practitioner Playbook -> Problem Boundary -> Evidence Collection -> Minimal Reproduction -> Baseline Comparison -> Data Check -> Data Incident -> Duplicate Rate -> Contamination Rate -> Mixture Shift -> Lineage Coverage -> Data Incident Gate -> Tokenizer Check -> Tokenizer Format Incident -> Chat Template Drift -> Assistant-Only Label Mask -> Prompt Loss Leak -> PAD Loss Leak -> EOS Coverage -> Format Gate -> Training Stability -> Loss Spike Audit -> Non-Finite Loss Rate -> Effective Label Tokens -> LR Continuity -> Rank Loss Skew -> Training Stability Gate -> Distributed Incident -> Rank Step Alignment -> Collective Mismatch -> Token Shard Imbalance -> Straggler Ratio -> Communication Ratio -> Checkpoint Shard Coverage -> Resume Continuity -> Pipeline Bubble Ratio -> Distributed Incident Gate -> Post-Training Incident -> Evaluation Metric Incident -> Aggregate Score Trap -> Slice Regression -> Clean Eval Lift -> Judge-Human Agreement -> Evaluation Gate -> Inference Performance Incident -> TTFT Regression -> TPOT Regression -> KV Pressure -> Prompt Cost Drift -> Cache Effectiveness -> Serving Gate -> RAG Incident -> Retrieval Miss -> Context Drop -> Unsupported Claim Rate -> Permission Leak Rate -> RAG Freshness Gate -> RAG Incident Gate -> Agent Incident -> Plan Feasibility Rate -> False Completion Rate -> Tool Result Injection Block Rate -> Agent Incident Gate -> Multimodal Incident -> Input Fidelity Rate -> Multimodal Evidence Recall -> Temporal Evidence Recall -> Multimodal Incident Gate -> Safety Compliance Incident -> Unsafe Pass Rate -> Safety Compliance Gate -> Project Collaboration Incident -> Objective Clarity Rate -> Metric Tree Coverage -> Baseline Coverage -> Experiment Reproducibility Rate -> Version Trace Coverage -> RACI Coverage -> Change Control Coverage -> Risk Escalation Coverage -> Project Collaboration Gate -> Experiment Retrospective -> Hypothesis Coverage -> Slice Analysis Coverage -> Badcase Taxonomy Coverage -> Evidence Coverage -> Timeline Coverage -> Impact Quantification Rate -> Root Cause Rate -> Prevention Coverage -> Action Item Closure -> Regression Verification Rate -> Change Failure Rate -> Retrospective Gate -> Log Check -> Trace Check -> Version Diff -> Root Cause Analysis -> Fix Priority -> Rollback Plan -> Regression Test -> Postmortem Completeness -> Debug Gate -> Incident Triage Audit -> Practitioner Interview Readiness -> Practitioner Interview Rubric -> Project Evidence Score -> STAR Reflection Score -> Trade-off Depth -> Expert Follow-up Readiness -> Weak Practitioner Question -> Practitioner Revision Plan -> Practitioner Interview Gate

### 研究链

Paper Reading -> Hypothesis -> Baseline -> Experiment -> Ablation -> Error Analysis -> Reproduction Report -> New Research Idea

### 2026-09 新模型架构分支

Long Context -> KV Cache -> 压缩表示 -> CSA/HCA -> 异构 KV Cache -> Serving Engine

Sequence Modeling -> Linear Attention -> Delta Rule -> Gated DeltaNet -> KDA -> KDA/MLA Hybrid

Residual Stream -> Hyper-Connections -> 双随机矩阵 -> Sinkhorn 投影 -> mHC -> 深层信号稳定性

Transformer Depth -> 标准残差 -> Attention Residuals -> Block AttnRes -> 深度历史选择 -> Pipeline/Inference 优化

Post-Training -> 领域专家培养 -> GRPO -> on-policy distillation -> 统一模型能力

每条边都要区分“公开模型/论文披露”与“本项目教学抽象”；不能由排行榜名称反推内部机制。

### GPT-6 Astra 运行时预算分支

GPT-6 Astra -> Context Window -> Maximum Input / Maximum Output -> Reasoning Effort -> Configuration Update -> Reasoning Item / Phase Replay -> Prompt Cache Prefix -> Deferred Tool Search -> Async Function/Custom Tool -> Response Lineage / call_id -> WebSocket Mid-turn Steering -> Compaction Canonical Context -> Skills / AGENTS.md Progressive Disclosure -> Tool Host -> Sandbox Policy -> Misalignment Monitoring -> Permission Trace -> Token Cost Threshold -> Unit Success Cost -> Long-Context Evaluation

该分支描述 OpenAI 官方模型页、模型指南、运行时文档和开发者博客已公开的接口、容量、状态、提示词路由和安全控制关系；模型架构、参数规模和训练方法仍属于待核验信息。`phase` 的现行专节以 GPT-5.5/GPT-5.4 为示例，不把它单独连接成 GPT-6 专属能力。

GLM-5.3 -> GLM-5.2 Base -> Post-Training -> Executable Environment -> Judge Agent -> Oracle Check -> No-op Check -> Unsolved-state Check -> Reward Shortcut Audit -> SAO with Compaction inheritance -> Context Compaction implementation (5.3-specific, unverified) -> Long-Task Evaluation
GLM-5.2 -> DSA -> IndexShare -> shared top-k index -> MTP -> speculative decoding -> acceptance length -> critic-based PPO -> compacted sub-traces -> anti-hack verifier

该分支把官方继承声明、SAO 论文公开算法和产品侧实现分开；论文提供 single-rollout/DIS/critic/Skip-Observation GAE 证据，但不等于 GLM-5.3 专属 compaction 或完整 post-training recipe。

### GLM-5.3 继承的 SAO 算法边界

GLM-5.3 -> official inheritance claim -> GLM-5.2 `SAO with compaction`
SAO -> single-rollout per prompt -> immediate async update -> reduced group barrier/straggler wait
SAO -> rollout engine logprob -> token ratio -> double-sided clipping/masking -> policy-lag control
SAO -> value model -> critic K=2 -> frozen attention + MoE projection update -> critic stability
SAO -> action/observation segments -> Skip-Observation GAE -> cross-segment value bootstrap -> credit assignment
SAO paper -> Qwen3-30B-A3B experiments + GLM-5.2 deployment statement -> not GLM-5.3 benchmark/architecture proof
GLM-5.3 product compaction -> serialization/state boundary/verifier -> unverified

Kimi K3 -> 发布文章披露 -> KDA / AttnRes / Stable LatentMoE -> 论文机制解释 -> State Manifest -> Harness Revision -> Tool/Permission Trace -> Fallback Audit -> Harness-Aware Evaluation

该分支把模型发布信号、独立论文机制和 Agent 运行时证据分开；K3 report 与固定 HF config 已确认主要配置，完整权重未下载，线上服务和独立 benchmark 仍需独立核验。

### Mistral Small 4 与 Step 3.5 Flash

Mistral Small 4 -> 稀疏 MoE -> total/active parameter ledger -> reasoning_effort -> EAGLE draft head -> NVFP4 -> speculative serving audit

Step 3.5 Flash -> 稀疏 MoE/top-8 routing -> 3:1 SWA/full attention -> MTP-3 -> acceptance length -> Context Manager -> harness-aware long-task evaluation

模型卡公开的结构字段、推测解码机制和 Agent harness 策略分别记录；吞吐、接受率、完整训练配方和后端支持仍需绑定版本与实验核验。

### Grok 4.6

Grok 4.6 -> 500K Context -> reasoning effort (`low/medium/high/xhigh`) -> structured output/function calling -> Web Search/X Search -> Tool Host/Sandbox -> Harness-aware evaluation

上述链路只连接 xAI 官方模型页公开的接口和运行时概念；参数量、训练架构、许可证、完整报告与首次发布日期仍待核验。

### Claude Opus 5

Claude Opus 5 -> 1M Context -> 128K Max Output -> 300K Batch Output -> Adaptive Thinking -> Default High Effort -> Thinking Block/Signature Replay -> Refusal/Fallback Ledger -> Tool Host/Sandbox -> Harness-aware evaluation

上述链路只连接 Anthropic 官方模型目录公开的接口、平台和运行时字段；参数量、训练架构、训练配方、后训练算法和独立 benchmark 复现仍待核验。

### Claude Fable 5.1

Claude Fable 5.1 -> 1M Context -> Adaptive Thinking (always on) -> Default High Effort -> Preserved Thinking -> Thinking Block Compatibility -> Thinking Block Invalidation on History Edit -> Forced Tool Capability Error -> Per-message Effort (beta) -> Turn-scoped System Message (beta) -> Tool Progress Updates (`display: "updates"`) -> Structured Tool Result -> Content Provenance -> Tool Host/Sandbox -> Long-horizon Harness Evaluation

Claude Fable 5.1 -> Mythos 5.1 -> Same Underlying Model -> Different Safeguards/Access Plan -> benchmark and refusal policy boundary -> not a separate architecture claim

Claude Fable 5.1 -> no exact `mini_swe_agent_claude_fable_5_1_*` DataCurve row -> no migration of Fable 5 `316/452` or other Claude Agent scores -> source-aware evaluation manifest

上述链路连接 Anthropic 专属模型页公开的接口、状态协议和产品定位；参数量、训练架构、训练配方、完整推理机制和独立 benchmark 复现仍待核验。

### Claude Sonnet 5

Claude Sonnet 5 -> 1M Context -> 128K Max Output -> 300K Batch Max Output -> Adaptive Thinking -> Default High Effort -> Fast Latency Field -> Multi-platform API -> Tool Host/Sandbox -> Fixed-harness Evaluation

Claude Haiku 4.5 -> 200K Context -> 64K Max Output -> Extended Thinking -> Fastest Latency Field -> Low-cost Multi-platform API -> Tool Host/Sandbox -> Latency/Quality/Unit-cost Evaluation

DeepSeek-R1-0528 -> Official Release Page -> JSON Output/Function Calling -> Open Weights Entry -> Thinking Mode/API Compatibility -> Revision/License Audit -> Harness-aware Evaluation

上述链路只连接 Anthropic 官方模型目录公开的接口字段；参数量、训练架构、训练配方和独立 benchmark 复现仍待核验。

### DeepSWE v1.1

DeepSWE -> 113 Tasks / 91 Repositories / 5 Languages -> mini-SWE-agent -> Tool Host -> Verifier -> Timeout/Retry/Context Policy -> Pass@1 + Interval -> Cost / Output Tokens / Agent Steps -> Harness-Aware Evaluation -> Artifact and Failure Audit

上述链路描述公开页面快照中的评测组成，不把任何一行分数当作基础模型能力。模型 revision、供应商、工具 schema、硬件、重试和 verifier 版本仍需绑定并独立复现；原始快照见 `research/model-update-2026-09/deepswe-snapshot-notes.md`。

### DeepSeek V4.1-Flash

DeepSeek V4.1-Flash -> 552B Backbone -> CED (20-layer Causal Encoder + 20-layer Decoder) -> 8B Prefill / 16B Decode Active Proxy -> Global KV Projection

DeepSeek V4.1-Flash -> CSA2 -> Full/Reindex/Reuse -> Shared Main KV + Indexer K -> Hierarchical Sparse Indexer -> Candidate Pool Recall -> Top-K Recall -> Global KV 890 Bytes/Token

DeepSeek V4.1-Flash -> FP4 Main KV (E2M1 + E4M3 Scale) -> Quantization Error -> Long-Context Retrieval Evaluation

DeepSeek V4.1-Flash -> SWA Bounded Replay -> Recent Window Replay -> Replay Compute -> Cache Revision/Position/Boundary Check -> Restore Gate

DeepSeek V4.1-Flash -> 384 Routed Experts + 1 Shared Expert -> 6 Routed Experts/Token -> Dispatch/All-to-All -> Load Balance/Latency

DeepSeek V4.1-Flash -> Engram Conditional Memory -> Token Lookup -> Conditional Model Memory -> Ablation/Hit-Miss Audit

DeepSeek V4.1-Flash -> Single-Pass mHC -> Residual Stream Mixing -> Mega-mHC Kernel -> Deep Signal Stability

DeepSeek V4.1-Flash -> DSpark -> Semi-autoregressive Draft -> Confidence-Scheduled Verification -> Acceptance Length -> Effective Tokens/Target Call -> Speculative Serving Audit

DeepSeek V4.1-Flash -> DeepSeek-ViT -> 2D-RoPE + 3x3 Pixel-Unshuffle -> MLP Projector -> Joint Vision/Text Tokens -> Multimodal Prompt/Permission Audit

DeepSeek V4.1-Flash -> 45T Multimodal Pretraining Tokens -> 64K Sparse Attention Training -> 34T Context Extension Data -> 1M Context Evaluation

DeepSeek V4.1-Flash -> SFT -> RL -> On-Policy Distillation -> Agent Task/Environment/Rollout Data Pipeline -> Verifier/Harness Evaluation

DeepSeek V4.1-Flash -> Numeric Reasoning Effort (1-100) -> Token/Latency/Quality Budget -> Tool Calls/Timeout/Unit Success Cost

DeepSeek V4.1-Flash -> Fixed HF Revision -> Reference `model.py`/TileLang `kernel.py` -> SWA Ring + Compressed KV + Candidate/Indexer + mHC/Sinkhorn -> Source/Runtime Audit

DeepSeek V4.1-Flash -> `forward_spec`/DSpark Block Exists -> `generate.py` Plain Autoregressive Entry -> Draft/Verify/Rollback Scheduler Missing -> No Speculative Throughput Claim

上述链路使用模型卡、固定 `config.json`、encoding README、官方 API 发布页和已逐页读取的技术报告支持的字段；报告补充的具体 kernel 名称、训练/部署设置和评测边界仍是发布方自报，完整 kernel source、线上接受率和独立 profiling 仍待核验。Agent benchmark 必须连接 model revision、effort、harness、工具、环境、verifier、timeout/retry 和 context policy，不能把组合结果归因给基础模型。

### K2 Horizon MoVA 与 Uno

Artificial Analysis -> K2 Horizon MoVA 36B/A4B -> IFM official model card -> Fixed Revision `de2d2efb32ed7639b7140bccbefe131a0063a982` -> 36B Total / 4B Active Proxy -> 48 Layers -> Dense First 3 / Sparse Last 45

K2 Horizon -> GQA (32Q/8KV) -> 64 MoVA Value Experts -> Top-4 Value Routing -> Attention -> 100 Routed FFN Experts -> Top-8 + 1 Shared Expert -> Dual Dispatch/Load Balance/Latency

K2 Horizon -> 524,288 Context -> 8K/32K/128K/512K Staged Training -> BF16 KV/Weights -> Long-Context Prefill -> TTFT/TPOT/Memory Evaluation

K2 Horizon -> Sigmoid Router -> Selection-only Bias -> Raw-score Normalization -> Scaling 2.5 -> Softplus Attention Gate -> Checkpoint Semantic Regression

K2 Horizon 0.9B Card -> Domain Expert Merge -> MOPD Disclosure -> On-policy Distillation -> Merge Interference Audit; this edge cannot be copied to 36B MoVA Training Recipe

K2 Horizon 7B -> Uno Adapter -> Frozen AR Base + LoRA Diffusion Path -> Parallel Draft -> `Psi-Spec` AR Rejection Verification -> Acceptance Length/Target Calls -> Speculative Serving Audit

该分支中 K2 36B/A4B 是两个排行榜候选链路中的 Artificial Analysis 发现并经官方资料核验的锚点；DataCurve DeepSWE 本地快照未检出 K2。Uno 是沿官方资料追踪到的关联 adapter/论文技术，不是新的排行榜候选。完整训练报告、接受率、kernel 性能和目标硬件 profiling 仍待核验。

### Qwen3.8

Artificial Analysis -> Qwen3.8 27B / 2.4T-A95B / Flash-Next / Max -> Qwen official model cards and Qwen Cloud -> hosted/open checkpoint boundary

Qwen3.8 27B -> Dense 27B -> 64 Layers -> 3 GDN + 1 Gated Attention -> Native Vision-Language -> Request-level Thinking Control

Qwen3.8 2.4T-A95B -> 2.4T Total / 95B Active Proxy -> 92 Layers -> 512 Experts -> 10 Routed + 1 Shared -> Text-only -> Forced Thinking

Qwen3.8 Flash-Next -> 125B Main / 6B Active Proxy -> GDN + QSA -> 4-branch Gated Residual -> 51B N-gram Table -> 4B MTP

GDN -> Gated Forgetting -> Delta Correction -> Fixed-size Recurrent State -> Long-prefix Cost Reduction

QSA -> Micro-block Pooling -> Partial RoPE -> Block-causal Indexer -> Top-k Complete Blocks -> Token Expansion + Causal Tail -> Sparse Explicit Attention

QSA -> Dense Teacher Attention -> Block Max Pooling + KL Distillation -> Sparse Training -> Backbone/Indexer Adaptation -> Candidate Recall / Long-context Evaluation

Gated Residual -> Per-branch RMSNorm -> Elementwise Read Gate -> Branch-scalar Write Gate -> Four Residual Paths -> Reduced Branch Mixing/Memory Traffic

N-gram Embedding -> Local Token Address -> 20M Slots / 51B Parameters -> Host-memory Prefetch -> Random Read/Bandwidth/Hit-rate Trade-off; this is neither RAG nor KV cache

Muon + AdamW/Adam -> Semantic Parameter Groups -> Per-matrix Newton-Schulz vs Embedding/Router/Table Updates -> Fused-parameter Split -> Distributed Optimizer/Kernel Constraints

Qwen3.8 -> enable_thinking / reasoning_effort / preserve_thinking -> Request Budget and Context Protocol -> Agent Quality/Latency/Unit-cost Evaluation

Flash-Next Report -> Self-reported Loss/Benchmark/7.6x Prefill/4.9x Decode -> Fixed Report Conditions -> Independent Kernel, Hardware Profiling and Acceptance-rate Verification Pending

该分支的模型候选来自两个排行榜，官方模型卡、Flash-Next 报告和仓库只用于核验与扩展；Qwen3.8-Max 是基于 A95B 的 hosted version，不作为独立 open checkpoint。

### Qwen3.8 Max (0902) 服务 revision

Artificial Analysis `qwen3-8-max` -> `Qwen3.8 Max (0902)` -> release slug `qwen3-8-max-0902` -> Qwen Cloud alias `qwen3.8-max-2026-09-02` -> upgraded snapshot -> hosted revision, not new open checkpoint

Qwen3.8 Max 0902 -> `1M context` -> `991K` normal input / `983K` thinking input / `131K` output -> reasoning/tool/cache/workspace budget -> TTFT/TPOT/p95/unit-success-cost ledger

`reasoning_effort` low/medium/xhigh -> default xhigh -> mutually exclusive with `thinking_budget` -> request capability validation -> effort is configuration, not model identity

Thinking mode -> `tool_choice auto/none only` -> forced tool requires non-thinking path -> MultiModalConversation -> host schema/permission/confirmation -> executor receipt -> final artifact

Qwen Context Cache -> explicit cache / implicit cache / session cache -> minimum 1,024 tokens -> different hit/billing/validity semantics -> revision/tokenizer/template/tenant/session key audit -> reuse or recompute decision

QwenCloud endpoint migration -> `dashscope-intl.aliyuncs.com` -> `maas.qwencloudapi.com` -> provider adapter/transport/auth regression -> not model architecture or capability evidence

QwenCloud account+model quota -> workspace override -> monthly TPM tier -> soft limit -> guaranteed TPM vs observed TPM -> 429/Retry-After/queue latency -> retry/idempotency/unit-success-cost audit

Qwen3.8 Max family -> generic DataCurve `qwen3_8_max_xhigh` -> 258/449 -> Pass@1 `57.4610%` -> mini-swe-agent + tools + environment + verifier -> no exact 0902 row -> no revision-level score migration

Qwen 0902 product claims -> coding / engineering-scale project / long autonomous development / multi-tool Agent / vision -> fixed harness + task contract + verifier -> recovery/quality/cost evaluation; product description is not architecture or training evidence

### GLM-5.3-Flash

Artificial Analysis/DataCurve -> `glm-5-3-flash` max -> Z.ai official docs/blog -> fixed revision model card/config -> 320B total / 18B activated -> 45 layers -> 34 linear + 11 sparse attention

Hybrid linear-sparse attention -> recurrent state + explicit sparse KV -> Indexer -> 4-key weighted IndexPool -> candidate top-k -> causal retrieval/evidence recall

GLM-5.3-Flash -> 288 routed experts -> top-8 + 1 shared expert -> token-expert dispatch -> all-to-all/load balance -> active/total/communication ledger

GLM-5.3-Flash -> mHC -> widened residual flow -> Sinkhorn projection -> doubly stochastic constraint -> deep-signal stability audit

GLM-5.3-Flash -> visual input -> Encode -> representation transfer -> Prefill -> KV/state metadata -> Decode -> tool stream -> Render/Observe -> Verify -> Refine -> artifact gate

Tool stream -> host parser -> schema/policy/permission check -> executor -> timeout/cancel/retry -> tool result -> model continuation; model output is not execution authority

GLM-5.3-Flash -> paged KV pool + KDA state pool -> dual-state scheduler -> prefix/cache recovery -> state-pool concurrency bottleneck -> separate capacity ledger

GLM-5.3-Flash -> SGLang MTP 5/1/6 -> low-latency policy -> draft/verify/rollback -> accepted tokens/tool boundary/task success; high-throughput route may disable speculative decoding

KV dtype + page/layout -> DSA backend pairing -> Blackwell FP8 KV/TRT-LLM DSA or H100/H200 BF16 KV/TileLang DSA -> invalid pairing gate -> numerical/recall/profile validation

GLM-5.3-Flash -> EPD -> encode/prefill/decode transfer -> representation/KV/state metadata -> cancellation/recompute/version/tenant isolation -> PD dummy-weight gate != production correctness

Transformers GLM5-Next -> no MTP layer; vLLM/SGLang -> serving-layer MTP recipe -> framework feature surface != checkpoint structure != target hardware acceptance

SGLang v0.5.20 -> fixed GLM5Next source entry -> stable source evidence -> full-weight/target-hardware gates remain open
SGLang main -> projection fusion + KDA prefill metadata + mHC boundary fusion + AMD FP8/Quark MXFP4 -> mutable upstream evolution -> not stable release proof
vLLM main `glm5next` -> IndexerCache(pool metadata) + TailCache(raw BF16 K/gate tail) + KDA state + MTP top-k/slot mapping -> multi-state manifest -> recovery/precision/profile gate
vLLM v0.29.0 tree -> no `vllm/models/glm5next/` in fixed snapshot -> negative stable-tag evidence -> recipe `0.29.0+` is not proof of stable GLM5Next path

GLM-5.3-FlashX -> associated service endpoint -> speed/quota metadata -> not a new Artificial Analysis/DataCurve model candidate

该分支的模型锚点来自两个排行榜；Z.ai 的架构、视觉 workflow、serving 组件和 speedup 是官方资料/发布方口径，不能由配置字段推导完整 kernel、训练 recipe、真实 state bytes、硬件 profiling 或独立 benchmark。DataCurve 的 `Pass@1` 仍是 `mini-swe-agent` 组合系统结果。

### DeepSeek V3.2

Artificial Analysis `deepseek-v3-2` -> `Non-reasoning` 配置 -> 官方 V3.2 模型卡/技术报告 -> DSA + scalable RL + large-scale agentic task synthesis -> 长上下文检索、后训练计算和工具任务数据闭环

DSA -> lightweight indexer -> candidate positions -> main attention -> index recall / final evidence recall -> KV/indexer bytes -> TTFT/TPOT -> dense fallback and failure audit

Thinking with Tools -> revised message/encoding protocol -> reasoning boundary -> tool call -> host schema/permission/confirmation -> tool result replay -> final answer -> trace and artifact gate

Agentic task synthesis -> task contract + environment + tools -> trajectory generation -> verifier/filter -> hard/failed sample audit -> contamination/shortcut check -> post-training data admission

V3.2 -> tool-calling capable checkpoint; V3.2-Speciale -> deep-reasoning variant -> no tool calling -> not a direct coding-agent substitute

DSA -> dense indexer warm-up -> KL alignment to dense attention -> sparse training -> 2048 KV candidates/query -> detached indexer loss + LM loss

V3.2 post-training -> specialist distillation -> mixed GRPO -> unbiased KL estimate + negative off-policy masking -> Keep Routing + Keep Sampling Mask -> more stable MoE RL

thinking with tools -> tool-only message retains reasoning -> new user message drops reasoning -> harness message encoding determines token reuse -> serving must preserve event boundaries

agentic task synthesis -> search/code/general/interpreter environments -> solution function restricted to tools -> independent verifier -> pass@100 admission -> contamination/shortcut audit

该分支是 Artificial Analysis 单榜资料级闭环；DataCurve 当前没有精确 V3.2 行，因此不挂接 DeepSWE Pass@1、成本或 Agent steps。公开资料未确认的 DSA kernel/indexer loss、完整 RL/synthesis recipe、硬件 profiling 和线上 acceptance rate 保持待核验。

DeepSeek V3.2-Exp inference -> FP8 Indexer -> non-interleaved Indexer RoPE -> FP8 Q/K cache -> `fp8_index` -> causal mask -> top-k candidate positions

DeepSeek V3.2 fixed config -> `q_lora_rank=1536` / `kv_lora_rank=512` -> 61 layers / DSA index fields -> final-model structure evidence

DeepSeek V3.2-Exp -> `q_lora_rank=1536` experimental inference config -> `kv_lora_rank=512` latent KV + positional cache -> prefill MHA / decode MQA -> FP8 KV deployment cache; same-valued final config and experiment remain separate revisions/artifacts

FP8 Index Score -> radix/histogram Top-k Selector -> selected KV gather -> causal sparse MLA -> index recall / final evidence recall -> TTFT/TPOT / KV bytes

TileLang `deepseek_v32` -> Lightning Indexer -> Top-k Selector -> Sparse MLA -> pipelined producer/consumer -> double buffering -> FP8 K-major V layout / shared-memory transpose

DeepGEMM PR #200 -> FP8 MQA logits / paged MQA logits -> MoE + MQA serving path -> SM90/SM100 kernel boundary

FlashMLA PR #98 -> sparse prefill / sparse FP8 decode -> SM90 sparse MLA -> metadata/combine/quantization -> hardware-specific profiling gate

vLLM V3.2-Exp recipe -> DeepGEMM dependency -> `DP=8, EP=8, TP=1` recommendation -> TP fallback -> FP8/BF16 KV choice -> `max-num-seqs` tuning -> recipe/harness result, not base-model score

V3.2 implementation evidence -> model artifact / inference demo / kernel / serving recipe / eval harness separation -> q_lora rank conflict audit -> benchmark attribution gate

DeepSeek V3.2 current AA page -> 685B/37B, 128K, Intelligence Index 16.0435, price 0.28/0.42 -> third-party directory/provider fields -> 9/20 648B vs 9/21 685B is page drift, not model revision

V3.2-Exp README -> V3.1-Terminus aligned comparison -> indexer non-interleaved RoPE / MLA layout correction -> implementation reproducibility evidence

V3.2-Exp README -> TileLang readable kernel / DeepGEMM indexer logits / FlashMLA sparse MLA -> implementation-layer separation -> no automatic full-weight, hardware, or SLO acceptance

SGLang dsv32 tags -> tp=8, dp=8, enable-dp-attention -> serving recipe entry -> hardware/dependency/full-weight/numerical/tool acceptance gates

vLLM recipe current URL -> HTTP 404 access boundary -> historical recipe evidence retained -> no inference that vLLM lacks V3.2 implementation

### GPT-5.3 Codex

Artificial Analysis `gpt-5-3-codex` -> `GPT-5.3 Codex (xhigh)` -> OpenAI official model page -> `gpt-5.3-codex` -> Responses-only -> agentic coding model

`low/medium/high/xhigh` -> reasoning effort configuration -> reasoning/visible output/tool/workspace budget -> TTFT/TPOT/task-success/unit-cost evaluation

400K context -> 272K maximum input + 128K maximum output -> input/output/reasoning/tool schema/workspace accounting -> admission and truncation gate

Codex Prompting Guide -> autonomy/persistence + codebase exploration + fixed workdir + `apply_patch` + tool schema + parallel calls -> Codex harness -> model/harness attribution boundary

Responses output items -> assistant `phase` (`commentary`/`final_answer`) + tool call/result + encrypted reasoning item -> full replay -> next-turn recovery gate

Server-side compaction -> encrypted compaction item; standalone compact endpoint -> canonical context -> preserve goals/receipts/permissions/artifacts -> duplicate-side-effect and recovery audit

Prompt caching -> stable rendered prefix -> KV state reuse -> cache hit/saved prefill -> tool schema/order/template/revision invalidation audit; cache is not compaction or permanent memory

Model proposes tool -> schema/parser -> host permission/approval -> sandbox executor -> receipt/result -> artifact verifier -> committed side effect; model call is not authorization or execution

GPT-5.3 Codex -> DataCurve no exact `mini_swe_agent_gpt_5_3_codex_*` row -> no migration of other GPT/Codex Pass@1/cost/steps -> configuration/harness evaluation boundary

GPT-5.3 Codex -> no public parameters/architecture/training recipe/system card/kernel/online acceptance rate -> no duplicate Transformer chapter -> map to deployment, reasoning, agent/tool, evaluation and inference-serving chapters

### OpenAI gpt-oss

Artificial Analysis `gpt-oss-120b`/`gpt-oss-20b` -> `high` reasoning configuration -> OpenAI Model Card/arXiv + official repository + Harmony + Hugging Face/Cookbook -> open-weight autoregressive MoE family

120B/20B -> total parameters vs active parameters -> 128/32 experts -> top-4 routing -> dispatch/load-balance/communication ledger -> weight/KV/workspace/concurrency memory audit

Alternating sliding-window/full attention -> local window cost + periodic dense global exchange -> GQA -> RoPE/YaRN -> long-context recall vs TTFT/TPOT experiment

MoE weights -> post-training MXFP4 -> group scale/kernel/quantization error -> checkpoint and single-GPU deployment boundary -> hardware/provider profiling gate

`o200k_harmony` -> Harmony roles/instruction hierarchy -> `analysis`/`commentary`/`final` channels -> recipient/tool schema/structured output -> render/parse/replay contract

`low/medium/high` -> variable-effort reasoning -> CoT/visible output/tool/workspace budget -> quality-latency-cost curve -> effort is configuration, not checkpoint

Model proposes tool -> Harmony parser -> schema/permission/approval -> sandbox/browser/Python executor -> tool result replay -> verifier/artifact -> committed side effect; model output is not execution authority

gpt-oss raw CoT -> Responses `reasoning.content[].reasoning_text` -> `response.reasoning_text.delta/done` -> item/index/turn lineage replay -> Harmony continuation; raw CoT is not end-user display content

gpt-oss provider -> API shape/channel/schema smoke test -> official `compatibility-test` -> AIME/GPQA/HealthBench quality eval -> kernel/precision/hardware/production gates; 0 invalid requests and >90% pass@k/pass^k are signals, not full proof

gpt-oss -> open weights -> downstream fine-tuning/copy/safety drift -> deployment permissions/sandbox/output filter/audit -> open-weight safety responsibility

gpt-oss -> DataCurve no exact `mini_swe_agent_gpt_oss_*` row -> no migration of other OpenAI/Codex Pass@1/cost/steps -> benchmark evidence boundary

gpt-oss -> no full training recipe/router balance/kernel profiling/online acceptance rate -> map to MoE, quantization, reasoning, Agent protocol and serving chapters -> no second architecture chapter for the sibling size

### Claude Opus 4.6

Artificial Analysis `claude-opus-4-6-adaptive`/`claude-opus-4-6` -> same base model + runtime configuration -> `effort`/adaptive thinking -> reasoning/tool/visible-output budget -> quality/latency/cost audit

Thinking block + encrypted signature -> exact replay -> tool call/result -> server-side `compact-2026-01-12` -> compaction block -> continuation state -> duplicate-side-effect/recovery audit

`defer_loading` -> regex/BM25 tool search -> up to 5 `tool_reference` -> schema gate -> permission/approval -> executor -> receipt -> verifier; tool discovery is not authorization

`computer_20251124` -> model action proposal -> host sandbox/allowlist/human confirmation -> screenshot/result -> prompt-injection defense -> artifact verifier; computer use is not browser permission

Opus 4.6 -> AA configuration fields + Anthropic product/runtime facts + publisher-reported benchmark -> separate evidence ledgers; DataCurve no exact row -> no migration of Opus 4.8/5 Agent scores -> no internal architecture inference

### Claude Opus 4.7

Artificial Analysis `claude-opus-4-7`/`claude-opus-4-7-non-reasoning` -> same base model + runtime configuration -> `high/xhigh/max` effort -> step/loop/request budget separation -> quality/latency/cost audit

`effort` -> step policy; task budget -> thinking/tool/result/output loop budget; `max_tokens` -> single-response hard cap -> independent accounting -> no direct concurrency addition

Updated tokenizer -> same input about `1.0-1.35x` legacy token range -> cache/input/thinking/output/retry remeasurement -> migration cost audit

Opus 4.7+ high-resolution vision -> `2576 px` max edge / `4784` visual tokens -> image resize/patch/token budget -> dense screenshot/document recall vs TTFT/cache/cost evaluation

Server-side compaction -> continuation state -> preserve budget/accounting/tool receipts/permissions/artifacts -> replay and duplicate-side-effect audit

Cyber safeguards -> policy detection/blocking + Cyber Verification Program -> authorization/sandbox/network isolation/audit/human escalation -> model capability is not execution permission

Claude Opus 4.7 -> DataCurve no exact `mini_swe_agent_claude_opus_4_7_*` row -> no migration of other Claude Agent scores -> runtime/control-plane evidence boundary -> no internal architecture inference

GLM-5.3 -> Z.ai Code Bench -> completion/checklist accuracy -> output-token efficiency -> private benchmark evidence boundary

GLM-5.3 -> executable long-horizon environment -> judge agent -> reference-free verifier -> solver trajectory -> reward-shortcut audit -> oracle/no-op/unsolved-state -> binary reward gate

GLM-5.3 -> CyberGym discovery/validation -> ExploitBench exploitation reasoning -> ExploitGym TPS-normalized time budget -> discovery/verification/exploitation-chain separation

GLM-5.3 benchmark footnotes -> harness/tool/container/timeout/domain whitelist/Tool Search/verifier -> model-plus-system result -> no cross-benchmark score merge

### Kimi K3

Kimi K3 -> sequence-length scaling -> KDA + Gated MLA (3:1, terminal Gated MLA) -> lower-bounded decay `g_min=5` -> recurrent state + global content interaction -> KDA context parallelism/state-aware prefix cache

Kimi K3 -> depth scaling -> Block AttnRes (8 blocks x 12 layers + embedding) -> block-level state -> cross-stage communication/memory reduction -> depth-selective residual flow

Kimi K3 -> width scaling -> Stable LatentMoE -> latent routed experts + full-width shared experts -> RMSNorm + SiTU-GLU -> Quantile Balancing -> frozen bias/fixed Top-k -> expert load/communication ledger

Kimi K3 -> MXFP4 weights + MXFP8 activations QAT -> low-precision deployment contract -> MoonEP/static shapes/zero-copy EP -> hardware and kernel profiling gate

Kimi K3 -> XTM -> `think/response/tool` channels + dynamic `tool-declare` + `tool/index` -> replay/permission/artifact state -> long-horizon RL + resumable microVM sandbox

Kimi K3 -> HF revision `f831ab...` + `config.json` -> model identity/config manifest -> 93 layers / 69 KDA + 24 Gated MLA / q-lora 1536 / kv-lora 512 -> revision-aware evidence boundary

Kimi K3 -> FlashKDA commit `7afb9f...` -> `8x8 fp32 forward substitution + 16x16 bf16 merge` -> recurrent-state kernel path -> SM90+/CUDA 12.9+/PyTorch 2.4+ gate -> H20/GB200 benchmark ledger

Kimi K3 -> vLLM recipe -> hybrid KV manager -> MLA attention cache + KDA recurrent state -> prefix caching / `prefix-match-unit=128` / DCP-TP-DEP topology -> schema validation + retry + verifier for tool-call parser drift

Kimi K3 -> fixed HF revision -> safetensors index -> 497,220 tensor mappings / 96 contiguous shards -> artifact manifest audit -> no full-weight-download claim

Kimi K3 -> MXFP4 packed weights + scale tensors -> 247,296 packed/scale pairs -> quantization artifact gate -> kernel/runtime gate remains separate

Kimi K3 -> `KimiDynamicCache` -> full-attention `key/value` cache + KDA `conv/recurrent` state -> `chunk_kda` prefill vs `fused_recurrent_kda` decode -> hybrid cache recovery

### Qwen3.5-397B-A17B

Qwen3.5-397B-A17B -> 397B total / 17B active -> 512-expert sparse MoE -> total/active/resident/KV/communication serving ledger

Qwen3.5 -> 15 x [`3 x Gated DeltaNet -> MoE` + `1 x Gated Attention -> MoE`] -> recursive state + periodic explicit retrieval -> hybrid cache and long-context trade-off

Qwen3.5 -> vision encoder + early-fusion multimodal tokens -> unified text/image/video path -> visual token budget/alignment/throughput evaluation

Qwen3.5 -> MTP training -> draft/verify/accepted length/rollback/committed KV -> speculative decoding acceptance and serving SLO

Qwen3.5 -> official million-agent RL/asynchronous RL claim -> rollout/environment/verifier/policy freshness ledger -> distinguish publisher claim from independent reproduction

Qwen3.5 -> Qwen3.8 architectural foundation -> QSA/Gated Residual/N-gram/Muon only supported by Qwen3.8 evidence -> version-isolated technology attribution

### GLM-5

Artificial Analysis `glm-5` -> `GLM-5 (Reasoning)` historical AA anchor -> DataCurve no exact `mini_swe_agent_glm_5_*` row -> no migration of GLM-5.2/5.3 Agent scores -> configuration/evidence boundary

GLM-5 -> 744B total / 40B active -> 256 routed experts + top-8 + 1 shared expert -> total/active/resident/communication/cache/workspace ledger -> serving capacity and p99 audit

GLM-5 -> DSA -> lightweight indexer -> candidate scores -> top-k positions -> main attention -> index recall/final evidence recall -> long-context needle/task-success evaluation

GLM-5 -> `q_lora_rank=2048` / `kv_lora_rank=512` -> latent/cache bandwidth hypothesis -> indexer/top-k/gather buffer -> end-to-end TTFT/TPOT and memory profiling gate

GLM-5 -> `slime` -> async rollout workers -> tool/environment trace -> verifier/reward -> trainer -> policy version -> policy lag/sample freshness/checkpoint consistency -> long-trajectory RL audit

GLM-5 -> Agentic Engineering -> plan/edit/execute/observe/test/diagnose/repair -> host permission/sandbox -> tool receipt -> verifier -> artifact digest -> completion gate

GLM-5 -> model output is proposal -> host executor is authority -> permission/approval/side-effect/replay -> independent verifier -> no-op/shortcut/false-completion audit

### Gemini 3.5 Flash-Lite

Artificial Analysis `gemini-3-5-flash-lite` -> AA 单榜候选 -> DataCurve no exact `mini_swe_agent_gemini_3_5_flash_lite_*` row -> no migration of Gemini 3.5/3.6 Flash Agent scores -> configuration/evidence boundary

Gemini 3.5 Flash-Lite -> low-latency/high-throughput multimodal API -> subagent/document parsing/classification -> input/media/token/latency/cost ledger -> serving SLO audit

Gemini 3.5 Flash-Lite -> default `thinking_level=minimal` + `minimal/low/medium/high` -> request-level test-time compute -> quality/TTFT/TPOT/tool-round/unit-success-cost comparison -> not four checkpoints

video input -> static fixed sampling -> one context build; video input -> agentic timeline exploration -> transcript/frame/audio on demand -> `processing_call` -> `processing_result` -> auditable media evidence state

agentic video -> fewer irrelevant media tokens hypothesis -> extra processing steps/latency/retry risk -> evidence recall/timeout/replay audit -> host authorization remains separate

Gemini 3.5 Flash-Lite -> Model Card says based on Gemini 3.1 Flash-Lite -> architecture/training/hardware/software point to predecessor -> version dependency boundary -> no independent architecture inference

Lite Model Card benchmark -> Google publisher harness/price/safety evidence -> AA Intelligence Index -> third-party configuration evidence; DataCurve exact row absent -> separate benchmark ledgers -> no score concatenation

### Kimi K2.6

Artificial Analysis `kimi-k2-6` -> AA 单榜锚点 -> DataCurve no exact `mini_swe_agent_kimi_k2_6_*` row -> no migration of K2.7 Code/K3 Agent scores -> configuration/evidence boundary

Kimi K2.6 -> 1T total / 32B active -> 384 routed experts + top-8 + 1 shared -> total/active/resident/communication/cache/workspace ledger -> serving capacity and p99 audit

Kimi K2.6 -> MLA -> `q_lora_rank=1536` + `kv_lora_rank=512` -> latent/position cache + gather/dispatch + tool/media state -> long-context TTFT/TPOT/recall evaluation

Kimi K2.6 -> MoonViT + native INT4 -> multimodal tokens + group-size-32 compressed weights + unquantized modules -> vision/quantization/kernel/compute-dtype audit

Kimi K2.6 -> Agent Swarm -> 300 sub-agents + 4,000 coordinated steps -> task DAG/parallel workers/isolated workspace/permission/budget/rollback -> artifact verifier and unit-success-cost audit

Kimi K2.6 -> `preserve_thinking` + `reasoning_content` -> interleaved thinking + multi-step tool call -> replay/schema/permission/executor receipt -> context and token-cost ledger

Kimi K2.6 -> Kimi Vendor Verifier -> pre-flight/OCRBench/MMMU-Pro/AIME2025/ToolCall/SWE-Bench -> model-vs-serving-vs-harness diagnosis -> deployment trust chain

### GPT-5.4 mini/nano

Artificial Analysis `gpt-5-4-mini` / `gpt-5-4-nano` -> AA 单榜精确 sibling -> DataCurve only `mini_swe_agent_gpt_5_4_xhigh` base row -> no migration of base Agent scores -> sibling evidence boundary

GPT-5.4 mini/nano -> `2026-03-17` snapshots -> 400K context / 272K maximum input / 128K maximum output -> separate manifest and cost ledger -> no inheritance of base 1.05M context

mini -> coding/computer-use/Agent workflow -> task-shape classifier + explicit prompt contract -> tool/verifier/escalation loop -> high-throughput route

nano -> classification/extraction/ranking/narrow sub-agent -> fixed schema + bounded tools + abstain/stop condition -> verifier -> upgrade to mini/base for open planning

mini model page lists `tool_search`/`computer_use`; nano page does not -> model_id/snapshot/endpoint capability probe -> precise tool catalog -> host authorization/sandbox/receipt -> no family-name inheritance

reasoning effort -> request-level behavior/investment knob -> input/reasoning/output/tool/retry/cache/executor ledger -> capacity and unit-success-cost evaluation -> not a hard token budget

GPT-5.4 mini/nano -> no public parameter/architecture/training report/exact DataCurve row -> reuse reasoning/Agent serving/tool/evaluation chapters -> no duplicate Transformer chapter

### DeepSeek V4 Pro 0813

Artificial Analysis `deepseek-v4-pro` -> `Reasoning, Max Effort` -> official `deepseek-v4-pro` API identity -> DataCurve `mini_swe_agent_deepseek_v4_pro_max` -> model/config/harness evidence ledger

V4 Pro -> 1.6T total / 49B active -> 384 routed experts / 6 selected / 1 shared -> resident/dispatch/communication/cache/workspace ledger -> serving capacity and p99 analysis

V4 Pro -> CSA -> compressed KV blocks + indexer top-k -> sparse remote retrieval -> compression loss/index recall/evidence recall separation

V4 Pro -> HCA -> heavier KV compression + dense compressed read -> local sliding-window branch -> remote efficiency versus local fidelity trade-off

V4 Pro -> mHC -> doubly stochastic residual mixing -> stable deep signal propagation -> residual geometry distinct from CSA/HCA and Attention Residuals

V4 Pro -> Muon + AdamW -> module-specific optimizer path -> training throughput/communication/optimizer-state ledger -> no inference-kernel inference

V4 Pro -> domain SFT + GRPO experts -> on-policy distillation -> unified student -> post-training teacher/student boundary -> not inference-time MoE routing

V4 Pro -> `low/high/max` -> request-level reasoning effort -> reasoning/visible output/tool/retry/cache ledger -> same model identity, different harness configuration

V4 Pro -> stateless Responses -> no `previous_response_id`/`conversation`/`background`/`store` -> host persistence + tool authorization + executor receipt + verifier -> protocol/serving responsibility boundary

### GLM-5.1 长周期 Agent 与过程质量

GLM-5.1 -> Artificial Analysis `glm-5-1`/`glm-5-1-non-reasoning` -> AA 单榜资料级闭环 -> DataCurve 无精确 `mini_swe_agent_glm_5_1_*` -> 不迁移相邻 GLM 版本 Agent 结果

GLM-5.1 -> 200K context / 128K output -> `glm-5.1` API -> text in/text out -> thinking/function calling/MCP/structured output/cache capability manifest

GLM-5.1 -> `GlmMoeDsaForCausalLM` -> 78 layers -> first 3 dense -> 256 routed experts / top-8 / 1 shared -> config-level DSA/MoE ledger -> no complete production-kernel inference

GLM-5.1 -> `q_lora_rank=2048` + `kv_lora_rank=512` + `index_topk=2048` -> low-rank/indexer implementation fields -> cache/indexer/attention cost questions -> no automatic migration of GLM-5 report details

GLM-5.1 -> long-horizon Agent -> up to 8-hour publisher claim -> experiment/analyze/optimize -> goal alignment + strategy revision + tool loop -> external tests/verifier/artifact gate

GLM-5.1 -> multi-turn SFT + RL + process-quality evaluation framework -> release-note training/assessment direction -> concrete RL algorithm/reward/verifier undisclosed -> evidence boundary

GLM-5.1 -> `thinking.type=enabled/disabled` -> request-level reasoning mode -> GLM-5.2+ `reasoning_effort` must not be inherited -> capability probe and version-specific API manifest

GLM-5.1 -> Function Calling/MCP -> model tool proposal -> host schema/permission -> executor -> tool result replay -> verifier -> final artifact

GLM-5.1 -> implicit context caching -> repeated prompt/history -> `cached_tokens` usage -> provider cache/billing layer -> not identical to permanent GPU KV cache

GLM-5.1 -> Z.ai self-reported SWE-Bench/KernelBench/Linux desktop -> benchmark/harness/measurement conditions -> not combinable with AA index or DataCurve -> fair evaluation ledger

GLM-5.1 -> model card links GLM-5 report + official blog JS recovered on 2026-09-21 + no precise GLM-5.1 arXiv report -> blog is publisher experiment evidence, not an independent technical report -> no duplicate Transformer chapter -> reuse GLM-5 DSA/MoE and Agent serving chapters

GLM-5.1 -> VectorDBBench -> Recall >= 95% + QPS feedback -> outer edit/compile/test/profile loop -> 600+ iterations / 6,000+ tool calls / 21.5k QPS publisher claim -> strategy transitions and constraint recovery

GLM-5.1 -> KernelBench Level 3 -> 50 problems + H100/Docker + 1,200-turn cap -> correctness tolerance + Claude Opus 4.6/GPT-5.4 anti-exploitation audits -> lower audited speedup -> verifier-aware performance evaluation

GLM-5.1 -> Linux desktop -> no single scalar objective -> self-review harness -> 8-hour iterative artifact refinement -> self-evaluation is not an independent verifier -> external tests/artifact acceptance

### Grok 4.20

Artificial Analysis `grok-4-20` -> `Grok 4.20 0309 v2 (Reasoning)` -> AA 单榜资料级锚点 -> DataCurve no exact `mini_swe_agent_grok_4_20_*` row -> no migration of Grok 4.5/4.6 Agent scores

Grok 4.20 -> xAI `grok-4.20-0309-reasoning` / non-reasoning / `grok-4.20-multi-agent-0309` -> endpoint-specific capability manifest -> model identity distinct from product display name

Grok 4.20 Multi-agent -> `agent_count=4/16` -> parallel specialist Agents -> leader synthesis -> orchestration topology, not MoE expert count or ordinary reasoning depth

Grok 4.20 Multi-agent -> sub-Agent reasoning/tool state hidden by default -> `use_encrypted_content` -> opaque encrypted replay state -> no readable chain-of-thought or client-side editing

Grok 4.20 -> AA 2M context vs xAI official 1M prompt/context -> source/snapshot/endpoint manifest + capability probe -> evidence discrepancy preserved -> no unconditional single context claim

Grok 4.20 -> Responses compaction -> single opaque `compaction` item -> whole-item ordered replay -> compaction cost/information-loss/recovery ledger -> not ordinary editable summary

Grok 4.20 -> prompt caching -> repeated prefix reuse -> provider compute/billing optimization -> not permanent GPU KV cache, conversation memory or reasoning state

Grok 4.20 -> server-side web/X/code/collections tools + client-side function calling -> server execution vs host authorization/execution/receipt -> hybrid Agent state machine -> verifier/artifact gate

Grok 4.20 -> Remote MCP `allowed_tools` -> smaller tool schema/context + smaller callable surface -> context-cost and least-privilege improvement -> host policy still required

Grok 4.20 -> no dedicated arXiv technical report -> architecture/training/kernel/independent benchmark undisclosed -> reuse reasoning/Agent/tool/serving/evaluation chapters -> evidence boundary

### Gemini 3.8 Flash

Artificial Analysis `gemini-3-8-flash` low/medium/high -> one canonical Gemini 3.8 Flash model -> effort is request configuration, not three checkpoints -> source-aware capability manifest

Gemini 3.8 Flash -> `thinking_level=low/medium/high` + shared output budget -> reasoning/tool/output budget trade-off -> thinking ablation -> TTFT/TPOT/token/cost/success ledger

Gemini 3.8 Flash -> thought summary + opaque thought signature -> partial visible explanation plus cross-turn reasoning continuity -> stateful/stateless replay -> not full chain-of-thought, permanent memory or GPU KV cache

Gemini 3.8 Flash -> Interactions API -> `thought/tool_call/tool_result/model_output` steps + SSE -> observable Agent trace -> host executes tools and verifies artifacts

Gemini 3.8 Flash -> Search/URL Context/File Search/Code Execution/function calling -> external evidence or computation re-enters context -> plan/act/observe loop -> citation/schema/business verifier

Gemini 3.8 Flash -> Computer Use -> screenshot + action intent + normalized coordinates -> client approval/execution -> new screenshot/result -> UI drift, permission and side-effect risk

Gemini 3.8 Flash -> 1M input + implicit caching -> long-context and repeated-prefix serving trade-off -> recall/TTFT/billing/TPOT experiment -> not equivalent to model architecture or GPU KV cache

Gemini 3.8 Flash -> DataCurve high `mini-swe-agent` row -> Pass@1/4 + cost + Agent steps -> model+harness+tools+environment+verifier result -> no migration to low/medium or adjacent Gemini

Gemini 3.8 Flash -> no independent 3.8 architecture/training report -> API/runtime evidence only -> no parameter/MoE/RL/verifier claim -> evidence boundary

Gemini 3.8 Flash -> `store=true` + `previous_interaction_id` -> server-side conversation history -> tools/system/generation config remain interaction-scoped -> replay manifest must re-specify runtime controls

Gemini 3.8 Flash -> `store=false` / 55-day paid / 1-day free / delete -> lifecycle and privacy control -> not permanent memory -> retention/deletion test

Gemini 3.8 Flash -> `max_output_tokens` -> thought + visible output hard cutoff -> `incomplete`/truncation risk -> use thinking-level routing for cost control

Gemini 3.8 Flash -> Thinking signature scope vs Tool-combination signature scope -> official documentation field-range conflict -> preserve returned opaque fields and call/result id -> endpoint/schema capability probe

Gemini 3.8 Flash -> tool context circulation -> built-in/custom server/client tools -> host permission/executor/verifier -> `validated` mode and side-effect audit

Gemini 3.8 Flash -> replay toy -> local protocol evidence -> state/signature/id/SSE/retention/modality checks -> not real API/model/SLO evidence

### DeepSeek V4 Pro：实现证据扩展

HF revision `b5968e...` -> fixed config/encoding/inference artifact -> reproducible file identity -> no full-weight download claim

`Compressor` -> gated compressed KV + overlap state -> block-boundary continuity -> heterogeneous KV/cache lifecycle

`Indexer` -> learned score + causal mask + top-k -> candidate recall -> evidence recall -> task success

`window_size=128` -> local uncompressed/less-compressed branch -> recent-token fidelity -> remote compressed retrieval trade-off

`n_hash_layers=3` + `sqrtsoftplus` -> hash/score routing split -> expert selection vs routing weight -> dispatch/communication ledger

FP4/FP8 block quantization -> TileLang GEMM/online softmax/Sinkhorn -> dtype/scale/workspace constraints -> hardware-specific profiling

DSML encoding -> tool role / `<tool_result>` / string-vs-JSON parameter -> parser -> schema/permission/executor/verifier separation

`MP=8` conversion example -> artifact-specific parallel configuration -> not universal production deployment claim

### DeepSeek V4.1-Flash：deepseek-recipe 协议实现

DeepSeek V4.1-Flash -> Artificial Analysis anchor -> pinned `deepseek-recipe` commit `8cadfede...` -> protocol adapter -> `ConversationRequest` -> V4.1 prompt/tokenizer -> external inference backend

`InferenceChunk` -> incremental state machine -> reasoning/DSML/JSON/stop segmentation -> `StreamProcessor` -> Messages/Chat Completions/Responses events -> transport/tool executor/verifier remain host responsibilities

V4.1 recipe -> tokenizer attached explicitly -> token IDs require matching tokenizer -> prompt special tokens not injected twice -> tokenizer revision enters golden manifest

Image URL/data URL/bytes -> quota/concurrency/retry/preprocess -> 600 images / 32 MiB image / 64 MiB request / 8 concurrent defaults -> default fetcher lacks SSRF/private-address filtering -> host security gate

recipe README unsupported features -> `logprobs`/server web search/JSON Schema strict/`n>1`/Responses storage/encrypted thinking -> adapter capability boundary -> not model architecture or model capability negative evidence

### Kimi K3 upstream/runtime evidence graph

Kimi K3 -> vLLM stable supported-models -> `KimiK3ForConditionalGeneration` / `Kimi-K3` -> stable documentation discoverability -> not stable wheel proof

Kimi K3 -> vLLM stable K3 API -> `KimiK3MTP` -> API/class visibility -> not MTP acceptance or speculative serving proof

Kimi K3 -> PyPI vLLM `0.29.0` + v0.29.0 registry -> `KimiK3ForConditionalGeneration` + `K3DSparkModel` + `KimiK3MTPModel` -> stable release/source entry -> not target hardware runtime or production proof

Kimi K3 -> `kimi_k3/__init__.py` -> `current_platform` -> NVIDIA/ROCm branch isolation -> platform-specific kernel/load/profile gates

Kimi K3 -> pre-release vLLM recipe -> K3-enabled nightly + CUDA 13/cu130 + r580+ driver -> optimized hybrid KV / TP-TEP-DEP-PP / DCP path -> not target hardware runtime or production SLO

Kimi K3 -> `KimiDynamicCache` -> MLA `key_cache/value_cache` + KDA `conv_states/recurrent_states` -> dual-state recovery -> revision/dtype/backend/prefix/topology gates

FlashKDA commit -> recurrent/chunk KDA kernel -> local H20/GB200 benchmark -> kernel evidence -> not end-to-end K3 throughput, cache recovery or tool-call acceptance

docs/API -> stable release/source entry -> optimized recipe -> full-weight load -> target hardware profile -> hybrid cache recovery -> tool schema/retry/idempotency/verifier -> production serving evidence

Kimi K3 -> SGLang `v0.5.20` `kimi_k3.py` -> stable text implementation entry -> not full-weight load or target hardware acceptance
Kimi K3 -> SGLang `main` `kimi_k3.py` -> LatentMoE + EP/A2A + shared-expert TP/reduce-scatter + SBO/ModelSlim/KDA gate -> mutable source evolution -> not stable release or production evidence
Kimi K3 -> SGLang `main`/`v0.5.20` `kimi_k3_vl.py` same blob -> MoonViT3d/2D RoPE/varlen vision backend -> source parity -> not visual numerical correctness or multimodal SLO
LatentMoE -> latent down projection -> A2A expert GEMM -> latent reduce/RMSNorm -> up projection -> communication and numerical-order gate
shared experts -> replicated or TP-sharded branch -> gather/MLP/reduce-scatter -> SBO side stream -> join before tail add -> stream/allocator/capture gate

### GPT-5.6 Luna：运行时与证据分层

GPT-5.6 Luna -> Artificial Analysis `max` -> Intelligence Index/speed/price/context -> third-party provider measurement
GPT-5.6 Luna -> DataCurve `mini_swe_agent_gpt_5_6_luna_max` -> Pass@1/4/cost/output/Agent steps -> mini-swe-agent + tools + environment + verifier

GPT-5.6 Sol/Terra/Luna -> service tier/model ID -> `reasoning.effort` -> `standard/pro` mode -> source-aware evaluation manifest
`reasoning.context` -> `current_turn` / `all_turns` -> opaque reasoning item replay -> same-family state compatibility -> not visible CoT/permanent memory/GPU KV cache
Stable developer prefix -> prompt-cache breakpoint -> cache read/write/TTL -> dynamic tool result -> compaction can change prefix -> cost/latency/replay audit
Tool schema -> deferred tool search / Programmatic Tool Calling -> model intent -> host schema/permission/sandbox/executor -> idempotency/timeout/retry -> verifier
1.05M context -> 922K maximum input + 128K maximum output + reasoning/tool items -> 272K whole-request price threshold -> unit success cost

OpenAI official page 403/timeout/DNS -> access-path evidence -> preserve historical official snapshot dates -> no claim of model absence/API change -> wait for fresh official verification

### Claude Sonnet 5：System Card 与 adaptive Agent

Claude Sonnet 5 -> Artificial Analysis `max` -> Intelligence Index/provider measurement -> third-party configuration evidence
Claude Sonnet 5 -> DataCurve five effort rows -> Pass@1/4/cost/steps -> `mini-swe-agent` + tools + environment + verifier -> Agent system evidence
Claude Sonnet 5 -> `thinking: adaptive` + `output_config.effort` -> behavior signal -> `max_tokens` hard cap -> task budget remains separate
Claude Sonnet 5 -> thinking block/signature + tool call/result -> opaque protocol state -> exact replay/compatibility gate -> not visible CoT or permanent memory
Claude Sonnet 5 -> context awareness + server-side compaction -> long-task state transition -> tool receipt/permission/artifact recovery -> not infinite context
Claude Sonnet 5 -> System Card -> RSP/cyber/agentic safety + benchmark harness -> safeguards/task/environment-bound evidence -> not internal training recipe or universal safety score
Claude Sonnet 5 -> no exact arXiv title / no public architecture recipe -> evidence boundary -> no parameter/MoE/adaptive algorithm claim

### Grok 4.7：长轨迹与运行时状态

Grok 4.7 -> Artificial Analysis `grok-4-7` / xhigh -> Intelligence Index/provider measurement -> AA configuration evidence
Grok 4.7 -> no exact DataCurve `mini_swe_agent_grok_4_7_*` -> no Agent score migration -> Grok 4.6 rows remain non-transferable
Grok 4.7 -> xAI release -> larger base + longer RL + harder long-horizon tasks -> publisher training disclosure -> not complete optimizer/reward/rollout recipe
Grok 4.7 -> `reasoning.encrypted_content` -> opaque state -> exact Responses replay -> not visible CoT / prompt cache / application memory
Grok 4.7 -> server-side tool encrypted output -> same replay responsibility -> preserve item lineage/call id -> not tool execution proof
Grok 4.7 -> `response.reasoning_text.delta` / `response.reasoning_summary_text.delta` -> developer-visible reasoning summary stream -> observation layer -> not complete CoT or replay substitute
Grok 4.7 -> `store` -> `previous_response_id` response storage behavior -> server-side state reference -> not permanent application memory
Grok 4.7 -> `/v1/responses/compact` -> opaque `compaction` item -> immutable context restart -> not over-limit rescue / budget reset
Grok 4.7 -> `reasoning_effort` low/medium/high/xhigh -> one model's runtime configuration -> effort sweep manifest -> not four checkpoints or MoE expert count
Grok 4.7 -> function calling / structured outputs -> model proposal/schema -> host permission/executor -> verifier/artifact
Grok 4.7 -> Remote MCP `allowed_tools` -> smaller schema + least privilege -> authorization/audit/network/idempotency gates
Remote MCP -> Responses `allowed_tools`/`headers` -> xAI SDK `allowed_tool_names`/`extra_headers` -> adapter capability matrix -> not unified security policy
Remote MCP -> Streaming HTTP/SSE only -> transport gate -> `require_approval`/`connector_id` unsupported in Responses API -> explicit rejection
Grok 4.7 -> xAI publisher benchmarks -> benchmark/harness/effort/environment/verifier labels -> not AA/DataCurve unified score
Grok 4.7 -> arXiv exact-title search no result -> evidence boundary -> no parameter/architecture/full recipe claim

### DeepSeek V3.2：deprecated identity merge

`deepseek-v3-2-reasoning-0925` -> V3.2 Exp reasoning -> deprecated/redirect -> historical revision
`deepseek-v3-2-0925` -> V3.2 Exp non-reasoning -> deprecated/redirect -> historical revision
`deepseek-v3-2-reasoning` / `deepseek-v3-2` -> V3.2 canonical configurations -> deprecated/redirect -> same family, not new models
`deepseek-v3-2-speciale` -> V3.2 special reasoning checkpoint -> deprecated/redirect -> same family checkpoint
V3.2 family -> no exact DataCurve `mini_swe_agent_deepseek_v3_2_*` -> no V4/V3.1 Agent score migration

### K2 Horizon 3.7B：dense 对照与 artifact 迁移

K2 Horizon 3.7B -> Artificial Analysis `k2-horizon-3-7b` -> AA 单榜 identity -> DataCurve 无精确 `mini_swe_agent_k2_horizon_3_7b_*` -> 不迁移其他 K2/模型 Agent 分数
K2 Horizon 3.7B -> fixed revision `6360f705b2e57d542959e6a2e67ebeb95dae0373` -> `K2HorizonForCausalLM` -> 36 layers -> dense path
`num_experts=0` + `mova_num_experts=0` -> no forward-time MoE/MoVA routing -> dense GEMM + 32Q/8KV GQA + KV cache -> 512K long-context serving
K2 3.7B -> 22.9T/8K pretraining -> 32K/128K/512K midtraining -> 512K SFT -> intermediate checkpoints -> stage-aware capability comparison
Math/Code/STEM-Code RL branches -> self-attention ISO merge / other weights RAM -> unified checkpoint -> not runtime MoE expert routing
`k2_aurora` -> `k2_horizon` -> copy + `weights_reencoded=false` + BF16 -> 36 shards/327 tensors -> artifact migration evidence -> not retraining or backend equivalence
K2 3.7B -> vLLM 5.06B dense/H200/`k2_horizon` parser -> SGLang TP1/BF16/FA3 -> publisher recipe evidence -> not local profiling
K2 3.7B -> current config/BF16 vs old APPENDIX Xllm/FP32 -> revision conflict -> core/embedding/dtype parameter ledger -> do not force one total parameter number
K2 3.7B dense baseline -> compare K2 36B/A4B -> MoVA value top-4 + FFN top-8 + shared expert -> dual dispatch/EP communication/workspace -> dense-vs-sparse serving experiment

### Qwen3-Omni：Thinker-Talker 与流式多模态

Qwen3-Omni -> Artificial Analysis `qwen3-omni-30b-a3b-instruct` -> AA 单榜模型身份 -> DataCurve 无精确 `mini_swe_agent_qwen3_omni_*` -> 不迁移其他 Qwen Agent 分数
Qwen3-Omni -> Instruct / Thinking / Captioner -> same family artifacts -> 不按三个基础模型计数
Qwen3-Omni -> AuT -> 12.5 Hz audio token rate -> 约 80 ms temporal granularity -> 不等于端到端 first-packet latency
Qwen3-Omni -> Qwen3-VL/SigLIP2-So400m vision encoder -> multimodal representation -> image/video spatial-temporal evidence
Qwen3-Omni -> TM-RoPE -> temporal/height/width axes + real timestamp -> cross-media temporal alignment -> 仍需 timestamp/chunk/state replay
Qwen3-Omni -> Thinker -> understanding/reasoning/tool proposal -> RAG/function calling/safety/verifier host boundary -> proposal != execution
Qwen3-Omni -> Talker -> multimodal-conditioned first codebook AR -> residual codebook MTP -> reduced codebook serial depth
Qwen3-Omni -> Code2Wav causal ConvNet -> audio code -> waveform -> packet/codec/realtime acceptance
Qwen3-Omni -> asynchronous chunked prefill -> Thinker/Talker scheduling -> high-concurrency serving -> needs cancel/retry/state manifest
Qwen3-Omni -> report first audio/video packet `234/547 ms` -> publisher/theory measurement -> not local p99/SLO
Qwen3-Omni -> vLLM README -> Thinker mainly supported, Instruct audio output progressing -> repository/runtime evidence -> not production audio acceptance
Qwen3-Omni -> S1 freeze LLM + train encoder/adapter -> S2 mixed multimodal ~2T tokens -> S3 8K->32K long audio/video -> training curriculum
Qwen3-Omni -> Thinker post-training -> SFT/distillation/GSPO/rule+model reward -> multimodal reasoning/tool behavior -> full recipe unverified
Qwen3-Omni -> Talker post-training -> continual pretraining/long-context/multilingual DPO/speaker fine-tuning -> voice quality/control -> full recipe unverified

### Qwen3-VL-235B-A22B

Artificial Analysis `qwen3-vl-235b-a22b-instruct/reasoning` -> same Qwen3-VL base model configurations -> AA identity -> DataCurve no exact `mini_swe_agent_qwen3_vl_*` -> no migration of other Qwen Agent scores

Qwen3-VL -> SigLIP2 vision encoder -> two-layer MLP merger -> Qwen3 MoE decoder -> visual/text token stream
Qwen3-VL config -> 94 layers + 64Q/4KV + 128 experts/top-8 + 262K position -> implementation fields -> not complete parameter/training proof
Interleaved-MRoPE -> interleaved temporal/height/width rotary frequencies -> lower long-video spectral bias -> position representation -> not 1M context or video retrieval proof
DeepStack `[8,16,24]` -> intermediate vision features + dedicated merger -> residual injection into early LLM layers -> multi-level visual alignment -> no extra visual sequence length but extra projection/activation cost
Video Timestamp -> seconds/HMS text before video temporal patch -> explicit temporal evidence -> long-video grounding -> still needs sampling/frame/chunk/replay audit
Qwen3-VL -> S0 merger alignment 67B/8K -> S1 multimodal pretraining 1T/8K -> S2 long-context 1T/32K -> S3 ultra-long 100B/262K -> curriculum -> not complete training recipe
Qwen3-VL -> square-root normalized per-token loss -> balance text-only/multimodal sources -> data mixture ledger -> not independent optimizer proof
Thinking with Images -> grounding cold start + visual-agent SFT/RL + 120K interaction distillation -> multimodal reasoning/tool behavior -> answer/multi-turn/tool-call rewards -> verifier and anti-shortcut audit
answer accuracy + multi-turn reasoning only -> fixed one-tool-call shortcut -> tool-calling reward -> action count/timing/complexity alignment -> trajectory quality separate from final answer
Qwen3-VL -> GUI/search/code proposal -> schema/permission/sandbox/executor -> observation replay -> artifact/verifier -> model proposal != external action success
HF config/Transformers raw main -> class/position/DeepStack/processor entry -> implementation evidence -> raw main no fixed commit -> not production kernel/full-weight/hardware acceptance

### Qwen3.7 Plus：托管多模态 Agent 合同

Artificial Analysis `qwen3-7-plus` -> exact leaderboard identity -> AA third-party fields -> DataCurve no exact `mini_swe_agent_qwen3_7_plus_*` -> no Agent-score migration
Qwen3.7 Plus -> text/image/video input + text output -> multimodal interactive hybrid Agent -> read screen/GUI/mobile navigation/visual-reference coding positioning -> product contract, not architecture disclosure
visual evidence -> resize/crop/frame/timestamp/token budget -> context/input/output/thinking reservation -> evidence coverage -> effective capability != accepted context window
model action proposal -> structured schema -> region/scope capability -> host permission -> GUI/mobile executor -> observation/idempotency/retry -> verifier/artifact
Function Calling / Structured Outputs / Web Search -> API capability matrix -> Beijing/Global/Virginia US differences -> authorization/executor -> supported endpoint != universal model behavior
Prefix Completion -> continuation protocol; Context Caching -> provider prefix reuse; GPU KV cache -> runtime attention state; application memory -> business state -> four different layers
Qwen3.7 alias -> `qwen3.7-plus-2026-05-26` snapshot -> hosted model identity -> no public checkpoint/parameters/technical report -> do not back-port Qwen3.5/Qwen3.8/Qwen3-VL/Qwen3-Omni architecture

### GLM-5.3 标准 DSA runtime

GLM-5.3 standard -> `glm_moe_dsa` -> `GlmMoeDsaForCausalLM` -> 78 layers -> first 3 dense -> 256 routed/top-8/1 shared -> config evidence != full training ledger
GLM-5.3 standard -> 21 Full indexer layers -> interleaved indexer RoPE + causal score/top-k -> 57 Shared indexer layers -> reuse `prev_topk_indices` -> lower indexer cost + possible candidate miss
`index_topk=2048` -> candidate selection field -> index recall -> evidence recall -> Agent task success -> three separate metrics
MLA latent cache + indexer top-k state + RoPE offset + block table + MTP iteration -> request recovery manifest -> cannot collapse into `kv_length`
`GlmMoeDsaForCausalLM` -> vLLM `deepseek_v32` registry -> DeepSeek-V3.2 DSA runtime reuse -> source routing evidence -> not checkpoint/training/kernel equivalence
`is_glm_moe_dsa` -> SGLang `DeepseekV32ForCausalLM` -> cross-layer top-k state -> stable/main source evidence -> not full-weight/target-hardware/production acceptance
GLM-5.3 standard -> stable tag vs mutable main -> registry/source entry -> full-weight load -> numerical check -> index/evidence recall -> MTP/cache recovery -> target profile -> tool/verifier/SLO
GLM-5.3 standard != GLM-5.3-Flash -> no automatic KDA/RadixLinearAttention/vision/dual-state-pool/EPD transfer

### DeepSeek V4.1-Flash vLLM `v0.30.0` stable release surface

DeepSeek V4.1-Flash -> Artificial Analysis anchor -> fixed HF revision -> vLLM main -> vLLM `v0.30.0` stable tag -> `DeepseekV41ForCausalLM` / `DSparkV41DraftModel` registry -> stable release/source evidence
vLLM `v0.29.0` registry -> no V4.1-specific class names -> historical negative evidence -> cannot be filled by mutable main
vLLM `v0.30.0` registry/package -> NVIDIA/ROCm `vl_model.py` + `dspark.py` + `quant_config.py` -> release surface -> not full-weight/target-hardware proof
PyPI vLLM `0.30.0` -> wheel/sdist metadata -> distributable artifact -> not installed dependency/kernel/numerical proof
v0.30.0 release notes -> FlashMLA V4.1 MXFP8 whole-KV + Mega-mHC + Engram async prefetch/DP sharding + DSpark state folding/EPLB isolation + XGrammar strict tools -> runtime release evidence -> not DeepSeek independent benchmark
stable release/source -> full-weight load -> numerical/FP4/recall -> DSpark verify/rollback/acceptance -> EPD -> target profile -> tool/verifier/SLO

Artificial Analysis `claude-opus-5-5` -> canonical Claude Opus 5.5 -> `max with fallback` configuration -> DataCurve exact row absent -> no neighboring Agent score migration
Anthropic Opus 5.5 release -> token efficiency + fewer steps/tool calls -> quality-cost curve -> requires fixed effort/tools/verifier
Opus 5.5 benchmark table -> adaptive/max, xhigh, medium/default -> effort/harness/provider mismatch -> no bare-model ranking
Cyber/Life Sciences/Distillation safeguards -> risk classification -> fallback/verification program/preserved thinking -> actual-model and permission trace
long-horizon coding -> context snapshot -> complete patch -> test receipt -> artifact verifier -> verified success
WANDR/OSWorld/Terminal/Automation -> model + tools + environment + verifier -> system result -> not bare model capability
Opus 5.5 always-on adaptive thinking -> effort control -> no disabled/manual budget -> compatibility gate
thinking block -> model/conversation/prefix binding -> replay/drop/error -> state-aware harness trace
forced tool_choice any/tool -> 400 -> auto + strict tool/structured output -> final tool verifier
computer_20251124 -> provider migration -> computer_toolset_20260801 -> platform capability manifest
on-demand compaction + inline tool_addition -> signed summary/tool schema update -> cache/state replay gate
fast mode -> same weights/faster inference config -> usage.speed + independent rate limit -> latency/cost ledger

### GPT-6 Sol：预算与 Agent runtime

Artificial Analysis `gpt-6-sol` -> canonical model -> `max` effort configuration -> DataCurve exact row absent -> no neighboring GPT Agent score migration
`gpt-6-sol` -> 1.05M context -> 922K maximum input -> 128K maximum output -> reasoning/tool/output budget ledger
GPT-6 family -> `reasoning.mode=standard|pro` + `reasoning.effort=none|low|medium|high|xhigh|max` -> independent control planes -> fair evaluation manifest
`configuration_update` -> mid-conversation effort change -> stateful protocol item -> not checkpoint/model-weight switch
reasoning tokens -> output/context accounting -> `incomplete` on budget exhaustion -> reserve reasoning/output capacity
Responses API -> typed response items + function calling -> application executor -> permission -> artifact verifier
Agents API -> OpenAI-managed Codex harness; Agents SDK -> application-managed loop/storage/approval; Responses API -> application-managed response/history/tool loop
tool search -> deferred schema -> loaded-tool registry/schema hash -> permission/executor/verifier -> cache-prefix and replay state
server-side compaction -> `context_management.compact_threshold` -> encrypted compaction item -> canonical continuation state
standalone `/responses/compact` -> full window -> canonical next context -> no arbitrary pruning
272K input threshold -> whole-request input/cache 2x + output 1.5x -> Batch/Flex 50% -> Fast mode 2x -> serving cost ledger
GPT-6 Sol API/runtime evidence -> model contract/observable state -> does not prove parameters, architecture, training recipe or production SLO

GPT-6 Luna -> Artificial Analysis `gpt-6-luna` canonical -> max configuration -> no exact DataCurve `mini_swe_agent_gpt_6_luna_*` row -> no Agent-score migration
GPT-6 Luna -> focused/high-volume contract -> task-shape router -> separate sibling ledger from GPT-6 Sol
GPT-6 Luna -> 1.05M context / 922K maximum input / 128K maximum output -> reasoning + tool + compaction budget
GPT-6 Luna -> 272K whole-request threshold -> input/cache 2x + output 1.5x -> Batch/Flex 50% -> Fast mode 2x -> unit-success-cost ledger
GPT-6 Luna -> GPT-6 family runtime docs -> `standard/pro` mode + `none`--`max` effort -> `configuration_update` -> later-turn budget change, not checkpoint swap
GPT-6 Luna -> Responses tools -> capability surface -> permission -> executor -> verifier -> model intent is not host authority
GPT-6 Luna -> official model/runtime evidence -> does not prove parameters, architecture, training recipe or production SLO

### DeepSeek V4.1-Flash API contract

Artificial Analysis `deepseek-v4-1-flash` -> official API `deepseek-flash` -> requested model / served model -> route and reproducibility ledger
`deepseek-flash` -> 1M context + 384K max output + 2500 concurrency -> provider contract -> not parameter/FLOPs evidence
Vision guide -> URL/file/request limits + 600 images + ~1024 image tokens/image -> media evidence budget -> not visual quality/full evidence use
Files API -> `purpose=user_data` + lifecycle/quota -> file artifact manifest -> expiry/permission/replay gate
Responses API -> stateless semantic SSE + `sequence_number` + terminal events -> typed item state machine -> no `[DONE]` assumption
`function_call_output` / `custom_tool_call_output` -> text/image observation -> client tool replay -> permission -> executor -> verifier
`/beta strict=true` -> JSON Schema enforcement -> structure valid -> host authorization -> execution -> business verifier
API contract -> reference/runtime source -> full-weight load -> FP4/FP8 quality + Top-K recall -> DSpark verify/rollback -> hardware/profile/SLO
