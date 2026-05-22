# AuroraGPT-2B Pre-Training Report

Local reproduction of the
[AuroraGPT-2B Pre-Training](https://api.wandb.ai/links/aurora_gpt/zyabwz9i)
Weights & Biases report.

The plots below are regenerated from the live `aurora_gpt/AuroraGPT` W&B
project using
[`scripts/generate_report.py`](scripts/generate_report.py) and rendered with
[ambivalent](https://github.com/saforem2/ambivalent) styling. Run the script
to refresh the figures whenever new runs land:

```bash
.venv-report/bin/python ALCF/AuroraGPT/2B/scripts/generate_report.py
```

## Runs included

The same 22-clause filter as the report:

- `username = foremans`
- `createdAt >= 2025-08-20`
- `optimizer = sophiag`
- `rope_theta = 50000`, `zero_stage = 0`, `checkpoint_activations = false`
- `micro_batch_size = 1`, `gradient_accumulation_steps >= 2`
- `world_size IN {3072, 6240}`, `num_layers IN {4..20}`
- `tokenizer_model = google/gemma-7b`
- summary `loss/lm loss` is not `NaN`
- tag is not `cooldown`
- `data_file_list NOT IN {books.txt, nvidia-math1-code2.txt}`
- excluded: `treasured-thunder-4350`, `grateful-dust-4351`, `glowing-disco-4352`,
  `happy-surf-4362`, `laced-puddle-4445`, `sage-universe-4443`, `dandy-sea-4440`,
  `twilight-dream-4439`, `vibrant-glitter-4358`, `dulcet-forest-4361`,
  `copper-pyramid-4359`, `lemon-aardvark-4360`, `sleek-tree-4390`,
  `autumn-sea-4394`

Curves are grouped by
`(machine, NHOSTS, torch_version, global_batch_size, optimizer, data_file_list)`
so each line represents one training segment / data mixture.

## Loss

### LM loss vs. consumed tokens (log y)

![](assets/loss_lm_loss_vs_tokens.svg)

### LM loss vs. consumed tokens (linear y)

![](assets/loss_lm_loss_vs_tokens_linear.svg)

### Gradient norm vs. consumed tokens

![](assets/grad_norm_vs_tokens.svg)

### Validation LM loss vs. iteration

![](assets/val_lm_loss_vs_iter.svg)

## Learning rate

### Learning rate vs. iteration (linear)

![](assets/lr_vs_iter_linear.svg)

### Learning rate vs. iteration (log-log)

![](assets/lr_vs_iter_loglog.svg)

## Throughput

### Iteration time vs. wall time

![](assets/iter_time_vs_runtime.svg)

### Iteration time vs. wall time (zoomed to 0–5 s)

![](assets/iter_time_vs_runtime_zoom.svg)

### Tokens / sec vs. iteration

![](assets/tokens_per_sec_vs_iter.svg)

### Tokens / GPU / sec vs. iteration

![](assets/tokens_per_gpu_per_sec_vs_iter.svg)

### TFLOPs vs. wall time

![](assets/tflops_lm_vs_runtime.svg)

## Progress

### Approx. parameters in billions vs. wall time

![](assets/params_in_billions_vs_time.svg)

### Consumed tokens vs. wall time

![](assets/consumed_tokens_vs_time.svg)
