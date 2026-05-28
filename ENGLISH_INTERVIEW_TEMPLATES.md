# 英文面试表达模板

## 自我介绍

I am an algorithm engineer focusing on large language models. My preparation covers model architecture, training, alignment, inference optimization, evaluation, and multimodal systems.

## 不确定时的回答

I am not fully certain about the exact implementation detail, but I would reason about it from the objective, data, optimization, and system constraints.

## 解释 Transformer

The core idea of Transformer is self-attention. Each token builds a contextual representation by attending to other tokens, and causal masking ensures that an autoregressive model cannot look into the future.

## 解释 RLHF

RLHF usually consists of supervised fine-tuning, reward model training from human preferences, and policy optimization with a KL constraint to keep the model close to the reference model.

## 解释 Trade-off

The main trade-off is between quality, latency, throughput, memory, and cost. A better model may improve output quality, but it can increase inference latency and serving cost.
