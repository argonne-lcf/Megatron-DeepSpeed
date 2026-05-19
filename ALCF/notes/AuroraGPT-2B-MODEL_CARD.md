---
language:
  - en
tags:
  - auroragpt
  - pretraining
  - megatron-deepspeed
  - sophiag
  - aurora
  - alcf
  - llama
model_name: AuroraGPT-2B
pipeline_tag: text-generation
---

# AuroraGPT-2B

AuroraGPT-2B is a 1.986 billion parameter language model developed at the
[Argonne Leadership Computing Facility (ALCF)](https://www.alcf.anl.gov/) as
part of the AuroraGPT project. It was pre-trained and continually pre-trained
across three stages on the
[Aurora supercomputer](https://www.alcf.anl.gov/aurora) using
[Megatron-DeepSpeed](https://github.com/argonne-lcf/Megatron-DeepSpeed) with
the SophiaG optimizer.

## Model Details

### Architecture

AuroraGPT-2B uses a Llama 3-style transformer architecture with grouped query
attention.

| Parameter | Value |
|:---|:---|
| Parameters | ~1.986B |
| Hidden size | 2,048 |
| Layers | 12 |
| Attention heads | 16 |
| KV heads (GQA) | 4 |
| FFN hidden size | 11,008 |
| Sequence length | 8,192 |
| Vocabulary size | 256,000 |

### Tokenizer

The model uses the
[google/gemma-7b](https://huggingface.co/google/gemma-7b) SentencePiece
tokenizer (HuggingFace `HFTokenizer`) with a vocabulary size of 256,000.

## Training Details

### Infrastructure

- **System**: [Aurora](https://www.alcf.anl.gov/aurora) supercomputer at ALCF
- **Nodes**: 256
- **GPUs per node**: 12 (Intel Data Center GPU Max)
- **Total GPUs**: 3,072
- **Precision**: BF16

### Framework

- [Megatron-DeepSpeed](https://github.com/argonne-lcf/Megatron-DeepSpeed)
- ZeRO Stage 0

### Optimizer

SophiaG with the following configuration:

| Hyperparameter | Value |
|:---|:---|
| Optimizer | SophiaG |
| &beta;&#x2081; | 0.9 |
| &beta;&#x2082; | 0.95 |
| &rho; | 0.01 |
| Weight decay | 0.1 |
| LR warmup | 5% of total iterations |
| Gradient clipping | 1.0 |

### Training Stages

The model was trained in three stages with a constant learning rate schedule
(infinite scheduler: warmup &rarr; constant &rarr; cooldown). All stages use
micro-batch size 1 and gradient accumulation steps 2.

#### Stage 1 &mdash; Pre-training

| Setting | Value |
|:---|:---|
| Learning rate | 2.28e-5 (constant) |
| Data | [OLMo Mix 1124](../data-lists/aurora/olmo-mix-1124.txt) |

**Data sources**: `wiki`, `algebraic-stack`, `pes2o`, `open-web-math`, `arxiv`,
`dclm`, `starcoder`

#### Stage 2 &mdash; Continued Pre-training

| Setting | Value |
|:---|:---|
| Learning rate | 2.17e-5 (constant) |
| Cumulative tokens | ~7.064T |
| Data | [Dolmino Mix 1124](../data-lists/aurora/dolmino-mix-1124-fused-file-list.txt) |

**Data sources**: `wiki`, `dclm`, `stackexchange`, `math`, `pes2o`, `flan`

#### Stage 3 &mdash; Math + Code Specialization

| Setting | Value |
|:---|:---|
| Learning rate | 2.17e-5 (constant) |
| Cumulative tokens | ~7.770T (+0.706T from Stage 2) |
| Data | [NVIDIA Math + Code](../data-lists/aurora/nvidia-math1-code2.txt) |

**Data sources** (from NVIDIA CC-MATH V1 and Nemotron Pretraining Code v2):
`4plus`, `4plus_MIND`, `3` (level-3 math), `Synthetic-Code`

## Intended Use

AuroraGPT-2B is intended for **research and scientific computing** purposes.
It serves as a base model for studying pre-training dynamics, continual
pre-training strategies, and optimizer comparisons at scale on the Aurora
supercomputer.

## Limitations

- This is a **base model** and has **not** been instruction-tuned or aligned
  with RLHF/DPO.
- It may generate inaccurate, biased, or harmful content.
- It is not intended for deployment in production or safety-critical
  applications without further fine-tuning, evaluation, and guardrails.

## Citation

```bibtex
@misc{auroragpt2b,
  title  = {AuroraGPT-2B},
  author = {Argonne Leadership Computing Facility},
  year   = {2025},
  url    = {https://github.com/argonne-lcf/Megatron-DeepSpeed}
}
```
