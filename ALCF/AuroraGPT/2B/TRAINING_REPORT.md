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
- excluded: `treasured-thunder-4350`, `grateful-dust-4351`, `glowing-disco-4352`

Curves are grouped by
`(machine, NHOSTS, torch_version, global_batch_size, optimizer, data_file_list)`
so each line represents one training segment / data mixture.

## Loss

### LM loss vs. consumed tokens (log y)

![](assets/loss_lm_loss_vs_tokens.png)

### LM loss vs. consumed tokens (linear y)

![](assets/loss_lm_loss_vs_tokens_linear.png)

### Gradient norm vs. consumed tokens

![](assets/grad_norm_vs_tokens.png)

### Validation LM loss vs. iteration

![](assets/val_lm_loss_vs_iter.png)

## Learning rate

### Learning rate vs. iteration (linear)

![](assets/lr_vs_iter_linear.png)

### Learning rate vs. iteration (log-log)

![](assets/lr_vs_iter_loglog.png)

## Throughput

### Iteration time vs. wall time

![](assets/iter_time_vs_runtime.png)

### Tokens / sec vs. iteration

![](assets/tokens_per_sec_vs_iter.png)

### Tokens / GPU / sec vs. iteration

![](assets/tokens_per_gpu_per_sec_vs_iter.png)

### TFLOPs (LM) vs. wall time

![](assets/tflops_lm_vs_runtime.png)

## Progress

### Approx. parameters in billions vs. wall time

![](assets/params_in_billions_vs_time.png)

### Consumed tokens vs. wall time

![](assets/consumed_tokens_vs_time.png)
