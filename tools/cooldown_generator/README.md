# `make_cooldown_cmds.py`

Generate **ready-to-run** Megatron-DeepSpeed commands to *cool down* a training run **starting exactly at a given checkpoint iteration**.

Given:

* a checkpoint iteration **S** (the `global_step` you resume from), and
* a cooldown length **R** (steps to spend decaying LR),

the script emits commands that set:

* `TRAIN_ITERS = T = S + R`
* `--lr_constant_plus_cooldown_frac = f = S / T`

So the **constant LR** phase ends at the resume step, and the **cooldown** covers the remaining `R` steps.

---

## What it prints

For each `(S, R)` pair, the script prints a small annotated block:

```
# id=<ID> resume_step=<S> cooldown_steps=<R> total_iters=<T> frac=<f>
LR_DECAY_STYLE=constant \
OPT=ipex.fusedlamb \
OVERRIDE_CKPT_OPT_PARAM=1 \
TRAIN_ITERS=<T> \
GRAD_ACC_STEPS=2 \
LOAD=<...> \
DATA_FILE_LIST=<...> \
bash train_alcf.sh \
  --override-opt_param-scheduler \
  --min-lr=2e-5 \
  --lr_constant_plus_cooldown \
  --lr_constant_plus_cooldown_frac=<f> \
  [any extra args...]
```

You can copy/paste the printed commands, or write all of them to a single `.sh` via `--emit-sh`.

---

## Requirements

* Python 3.7+
* Your training wrapper (default: `train_alcf.sh`) accepts the same environment variables/flags as shown above.
* The checkpoint parent directory (`--load`) is the path you normally pass to Megatron/DeepSpeed for resuming.

---

## Basic usage

### Single checkpoint, single cooldown

```bash
./make_cooldown_cmds.py \
  --load /proj/checkpoints_parent \
  --data-file-list ALCF/data-lists/olmo-mix-1124.txt \
  -S 72500 \
  -R 2000
```

### Multiple checkpoints, one cooldown value

```bash
./make_cooldown_cmds.py \
  --load /proj/checkpoints_parent \
  --data-file-list ALCF/data-lists/olmo-mix-1124.txt \
  -S 12900 32800 52650 72500 \
  -R 2000
```

### Multiple checkpoints × multiple cooldowns (grid)

```bash
./make_cooldown_cmds.py \
  --load /proj/checkpoints_parent \
  --data-file-list ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
  -S 92400 112250 132150 \
  -R 1000 2000 5000 \
  --emit-sh cooldown_grid.sh
# => prints to stdout and writes all commands to cooldown_grid.sh
```

### Using explicit ID tags (optional)

IDs label the comment header above each command; they’re handy for grouping in dashboards or logs.

```bash
./make_cooldown_cmds.py \
  --load /proj/checkpoints_parent \
  --data-file-list ALCF/data-lists/olmo-mix-1124.txt \
  --checkpoint-ids 1 2 3 4 \
  -S 12900 32800 52650 72500 \
  -R 2000
```

### Using `--pairs` (compact)

`--pairs` accepts `S:R` or `id:S:R` entries; when `id:` is omitted, IDs auto-increment.

```bash
./make_cooldown_cmds.py \
  --load /proj/checkpoints_parent \
  --data-file-list ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
  --pairs 92400:2000 112250:5000 7:132150:2000
```

---

## Common flags

* `--load` *(required)*: Parent directory that contains `global_stepXXXXX/` checkpoints.
* `--data-file-list` *(required)*: The data list file your wrapper expects.
* `--train-script` (default: `train_alcf.sh`)
* `--grad-acc-steps` (default: `2`)
* `--opt` (default: `ipex.fusedlamb`)
* `--min-lr` (default: `2e-5`)
* `--no-override-ckpt-opt` (use if you **do not** want `OVERRIDE_CKPT_OPT_PARAM=1`)
* `--extra-args "..."` (anything you want appended to the train command; e.g., W&B tags)
* `--emit-sh <path>` (write all printed commands to a runnable shell script)

---

## Why this formulation?

Megatron-DeepSpeed’s constant-plus-cooldown helper treats `--lr_constant_plus_cooldown_frac=f` as the fraction of **total** training reserved for the constant phase. By setting:

* `T = S + R` and `f = S/T`,
  we guarantee the **cooldown starts exactly at resume** and lasts `R` steps—no ambiguity about “10% of total vs. remaining.”

---

## Quick sanity checks

* If you resume at **S** and choose **R=2000**, you should see:

  * `TRAIN_ITERS = S + 2000`
  * `lr_constant_plus_cooldown_frac ≈ S / (S + 2000)`
    (It will be a high fraction like `0.98…` if you resume late.)

* If you accidentally swap the meanings (e.g., set `f=0.9` without adjusting `T`), you’ll be scheduling “last 10% of **total** job” rather than “R steps after resume.” This script avoids that pitfall.

---

## Examples with real numbers

Assume:

* tokens/step = `8192 * 6144 = 50,331,648`
* 5% rollback from final ~7T run corresponds to ≈ *~7k steps* earlier.

**ID 5** (≈ 5T tokens): `S=92400`, `R=2000`

```bash
./make_cooldown_cmds.py \
  --load /proj/checkpoints_parent \
  --data-file-list ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
  --pairs 5:92400:2000
```

**ID 5 rollback**: `S=~(92400 - ~7000) ≈ 85400` (rounded per your table)

```bash
./make_cooldown_cmds.py \
  --load /proj/checkpoints_parent \
  --data-file-list ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
  --pairs 5:85400:2000
```

---

## Troubleshooting

* **“Provide either --pairs OR both --checkpoint-iters and --cooldown-steps.”**
  You must supply `(S, R)` pairs. Use `--pairs` *or* `-S ... -R ...`.

* **Nothing about the data list changes across IDs.**
  This script is intentionally **generic**. If your data list changes by phase/ID, handle that in a wrapper (e.g., `run_cooldown_per_id_split.sh`) and pass the correct `--data-file-list` for each call.

* **Scheduler doesn’t seem to take effect.**
  Ensure `--override-opt_param-scheduler` is present (it is) and `OVERRIDE_CKPT_OPT_PARAM=1` isn’t disabled unless you need to.

---

# Cooldown Wrapper Scripts

These helper scripts automate checkpoint enumeration and cooldown‐command generation using the generic [`make_cooldown_cmds.py`](./make_cooldown_cmds.py).

---

## 1. `build_checkpoints_from_tokens.py`

### Purpose

Converts **token milestones** (0 T → 7 T) into **training iterations** and computes a “rollback” checkpoint offset by a percentage of the final total run.

### How it works

1. Each iteration processes `8192 × 6144 = 50 331 648 tokens`.
2. For every integer trillion token milestone (1 T → 7 T) it computes:

   * `steps_mod` = rounded step count at that token milestone
   * `steps_rollback` = `steps_mod – (cooldown_percent × final_steps)`
     (rounded to nearest multiple of `--round`)
3. Writes a tab-separated file:

```
id    steps_mod    steps_rollback
1     12900        12900
2     32800        32800
...
7     132150       132150
```

### CLI

```bash
python build_checkpoints_from_tokens.py \
  --ttokens 8 \
  --tokens-per-step $((8192*6144)) \
  --cooldown-percent 0.05 \
  --round 50 \
  --out checkpoints.tsv
```

**Arguments**

| Flag                 | Default           | Description                                       |
| -------------------- | ----------------- | ------------------------------------------------- |
| `--ttokens`          | `8`               | Number of trillion-token milestones (0..N).       |
| `--tokens-per-step`  | `8192*6144`       | Tokens processed per optimizer step.              |
| `--cooldown-percent` | `0.05`            | Fraction of final total used for rollback offset. |
| `--round`            | `50`              | Round step counts to nearest N.                   |
| `--out`              | `checkpoints.tsv` | Output TSV file.                                  |

### Output

Creates `checkpoints.tsv` for use by the generator script.

---

## 2. `gen_cooldown_sweep.sh`

### Purpose

Automates cooldown job generation:

* Reads `checkpoints.tsv` from the step builder.
* Creates **one `.sh` per checkpoint ID** for both *exact* and *rollback* resume points.
* Uses phase-based data lists:

  * IDs 1–4 → `olmo-mix-1124.txt`
  * IDs 5–7 → `aurora/dolmino-mix-1124-fused-file-list.txt`

### What it produces

```
cooldown_out/
  cooldown_id1_exact.sh
  cooldown_id1_rollback.sh
  cooldown_id2_exact.sh
  ...
  cooldown_id7_exact.sh
  cooldown_id7_rollback.sh
```

Each file contains **one** fully-formed Megatron command block generated via `make_cooldown_cmds.py`.

### CLI

```bash
./gen_cooldown_sweep.sh \
  --load /proj/checkpoints_parent \
  --cool-R 2000 \
  --emit-dir ./cooldown_out \
  --phase1-list ALCF/data-lists/olmo-mix-1124.txt \
  --phase2-list ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt \
  --train-script train_alcf.sh \
  --extra-args "--wandb-tag cooldown --wandb-note sweep_remaining"
```

### Key options

| Flag                                                              | Meaning                                                        |
| ----------------------------------------------------------------- | -------------------------------------------------------------- |
| `--load`                                                          | Parent directory containing checkpoints (`global_stepXXXXX/`). |
| `--cool-R`                                                        | Cooldown length in steps (`R`).                                |
| `--emit-dir`                                                      | Output directory for generated scripts.                        |
| `--phase1-list`, `--phase2-list`                                  | Data lists for phase 1 (IDs 1–4) and phase 2 (IDs 5–7).        |
| `--train-script`                                                  | Training wrapper (`train_alcf.sh` by default).                 |
| `--extra-args`                                                    | Extra flags to append (e.g. W&B tags).                         |
| `--tokens-per-step`, `--ttokens`, `--cooldown-percent`, `--round` | Passed to `build_checkpoints_from_tokens.py`.                  |
| `--python`                                                        | Python interpreter (default: `python`).                        |

### Output behavior

* Calls `build_checkpoints_from_tokens.py` internally.
* For each ID:

  * Uses the proper data list by phase.
  * Generates `cooldown_id<ID>_exact.sh` (always) and `cooldown_id<ID>_rollback.sh` (if rollback > 0).
* Each script is executable and self-contained.

---

### Typical workflow

1. **Enumerate checkpoints**

   ```bash
   python build_checkpoints_from_tokens.py --out checkpoints.tsv
   ```
2. **Generate per-checkpoint cooldown scripts**

   ```bash
   ./gen_cooldown_sweep.sh --load /proj/checkpoints_parent --cool-R 2000
   ```
3. **Submit or batch-run** any of the emitted scripts on your training cluster.

---

These two wrappers + [`make_cooldown_cmds.py`](./make_cooldown_cmds.py) together form a reproducible, parameterized pipeline for generating and managing cooldown experiments from multi-trillion-token Megatron-DeepSpeed runs.

