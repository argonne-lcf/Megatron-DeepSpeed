# AuroraGPT-70B Performance Study

```bash
TOTAL_PARAMS=68976648192 (68976.648M)
```

## Personal Experiments

### Pipeline + ZeRO-1

```bash
PP="${NHOSTS}" \
  TP=1 \
  HEADS=64 \
  NLAYERS=80 \
  NUM_KV_HEAD=8 \
  FFN_HIDDEN_SIZE=28672 \
  HIDDEN=8192 \
  MICRO_BATCH=1 \
  GRAD_ACC_STEPS="${GAS}" \
  SEQ_LEN=8192 \
  DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt \
  USE_ACTIVATION_CHECKPOINTING=1 \
  bash train_alcf.sh
```

- **Note**:
  - `sps`: samples per second
  - `tpgps`: tokens per GPU per second


|   Nodes   | NGPU | PP  | DP  | TP  | MBS | GAS | GBS | ACKPT | TFLOPS | `sps` | `tpgps` | wandb                                                                          |
| :-------: | :--: | :-: | :-: | :-: | :-: | :-: | --: | :---: | -----: | ----: | ------: | :----------------------------------------------------------------------------- |
| 8[^2125]  |  96  |  8  | 12  |  1  |  1  |  1  |  12 | True  |    OOM |   OOM |     OOM | [chocolate-meadow-2125](https://wandb.ai/aurora_gpt/AuroraGPT/runs/uhfkwmp2)   |
| 8[^2126]  |  96  |  8  |  6  |  2  |  1  |  1  |   6 | True  |  10.26 | 0.541 |  23.084 | [glamorous-darkness-2126](https://wandb.ai/aurora_gpt/AuroraGPT/runs/tdehvqey) |
| 8[^2153]  |  96  |  8  |  6  |  2  |  1  |  1  |   6 | False |  13.16 | 0.694 |  29.608 | [proud-frog-2153](https://wandb.ai/aurora_gpt/AuroraGPT/runs/ho1lwmer)         |
| 8[^2127]  |  96  |  8  |  6  |  2  |  1  |  2  |  12 | True  |  17.75 | 0.936 |  39.927 | [stoic-dragon-2127](https://wandb.ai/aurora_gpt/AuroraGPT/runs/2vay044x)       |
| 8[^2128]  |  96  |  8  |  6  |  2  |  1  |  4  |  24 | True  |  28.93 | 1.525 |  65.081 | [drawn-wildflower-2128](https://wandb.ai/aurora_gpt/AuroraGPT/runs/qjaqpbxg)   |
| 8[^2129]  |  96  |  8  |  6  |  2  |  1  |  8  |  48 | True  |  42.27 | 2.229 |  95.103 | [fresh-waterfall-2129](https://wandb.ai/aurora_gpt/AuroraGPT/runs/8m38fq95)    |
| 8[^2130]  |  96  |  8  |  6  |  2  |  1  | 16  |  96 | True  |  54.66 | 2.882 | 122.974 | [divine-waterfall-2130](https://wandb.ai/aurora_gpt/AuroraGPT/runs/on3m4isp)   |
| 8[^2131]  |  96  |  8  |  6  |  2  |  1  | 32  | 192 | True  |  64.02 | 3.376 | 144.037 | [wild-bee-2131](https://wandb.ai/aurora_gpt/AuroraGPT/runs/hitrbi6d)           |
|  &nbsp;   |      |     |     |     |     |     |     |       |        |       |         |                                                                                |
| 16[^2118] | 192  | 16  | 12  |  1  |  1  |  1  |  12 | True  |    5.6 | 0.594 |  12.695 | [fluent-surf-2118](https://wandb.ai/aurora_gpt/AuroraGPT/runs/0y250j0i)        |
| 16[^2119] | 192  | 16  | 12  |  1  |  1  |  2  |  24 | True  |  10.25 | 1.081 |  23.064 | [dulcet-salad-2119](https://wandb.ai/aurora_gpt/AuroraGPT/runs/5f1rdn9p)       |
| 16[^2120] | 192  | 16  | 12  |  1  |  1  |  4  |  48 | True  |  18.10 | 1.908 |  40.713 | [rose-blaze-2120](https://wandb.ai/aurora_gpt/AuroraGPT/runs/9obt1iqi)         |
| 16[^2121] | 192  | 16  | 12  |  1  |  1  |  8  |  96 | True  |  29.49 | 3.110 |  66.342 | [azure-jazz-2121](https://wandb.ai/aurora_gpt/AuroraGPT/runs/oaft4n5p)         |
|    16     | 192  | 16  | 12  |  1  |  1  | 16  | 192 | True  |    OOM |   OOM |     OOM | OOM                                                                            |


[^2125]: [chocolate-meadow-2125](https://wandb.ai/aurora_gpt/AuroraGPT/runs/uhfkwmp2)
[^2126]: [glamorous-darkness-2126](https://wandb.ai/aurora_gpt/AuroraGPT/runs/tdehvqey)
[^2153]: [proud-frog-2153](https://wandb.ai/aurora_gpt/AuroraGPT/runs/ho1lwmer)
[^2127]: [stoic-dragon-2127](https://wandb.ai/aurora_gpt/AuroraGPT/runs/2vay044x)
[^2128]: [drawn-wildflower-2128](https://wandb.ai/aurora_gpt/AuroraGPT/runs/qjaqpbxg)
[^2129]: [fresh-waterfall-2129](https://wandb.ai/aurora_gpt/AuroraGPT/runs/8m38fq95)
[^2130]: [divine-waterfall-2130](https://wandb.ai/aurora_gpt/AuroraGPT/runs/on3m4isp)
[^2131]: [wild-bee-2131](https://wandb.ai/aurora_gpt/AuroraGPT/runs/hitrbi6d)
[^2118]: [fluent-surf-2118](https://wandb.ai/aurora_gpt/AuroraGPT/runs/0y250j0i)
[^2119]: [dulcet-salad-2119](https://wandb.ai/aurora_gpt/AuroraGPT/runs/5f1rdn9p)
[^2120]: [rose-blaze-2120](https://wandb.ai/aurora_gpt/AuroraGPT/runs/9obt1iqi)
[^2121]: [azure-jazz-2121](https://wandb.ai/aurora_gpt/AuroraGPT/runs/oaft4n5p)


## ZeRO-3 + HPZ

- Command:

    ```bash
    TP=1 \
      GRAD_ACC_STEPS=4 \
      ZERO_HPZ_PARTITION_SIZE="${NGPU_PER_HOST}" \
      PP=1 \
      ZERO_STAGE=3 \
      HEADS=64 \
      NLAYERS=80 \
      NUM_KV_HEAD=8 \
      FFN_HIDDEN_SIZE=28672 \
      HIDDEN=8192 \
      MICRO_BATCH=1 \
      SEQ_LEN=8192 \
      USE_ACTIVATION_CHECKPOINTING=1 \
      DATA_FILE_LIST="ALCF/data-lists/aurora/books.txt" \
      bash train.sh
    ```


|  Nodes  | NGPU | PP  | DP  | TP  | MBS | GAS | GBS | ACKPT | TFLOPS | `sps` | `tpgps` | wandb                                                                      |
| :-----: | :--: | :-: | :-: | :-: | :-: | :-: | --: | :---: | -----: | ----: | ------: | :------------------------------------------------------------------------- |
| 4[^z3h] |  48  |  1  |  1  |  1  |  1  |  1  |  48 | True  |    ??? |   ??? |     ??? | ???                                                                        |
| &nbsp;  |      |     |     |     |     |     |     |       |        |       |         |                                                                            |
|    6    |  72  |  1  |  1  |  1  |  1  |  1  |  72 | True  |  39.05 | 1.544 |  87.849 | [charmed-fire-2203](https://wandb.ai/aurora_gpt/AuroraGPT/runs/actny2dl)   |
| &nbsp;  |      |     |     |     |     |     |     |       |        |       |         |                                                                            |
|    8    |  96  |  1  |  1  |  1  |  1  |  1  |  96 | True  |  47.60 | 2.510 | 107.087 | [glamorous-wood-2199](https://wandb.ai/aurora_gpt/AuroraGPT/runs/bgu31497) |
|    8    |  96  |  1  |  1  |  1  |  1  |  2  | 192 | True  |  55.69 | 2.936 | 125.287 | [solar-plant-2200](https://wandb.ai/aurora_gpt/AuroraGPT/runs/gnc48o99)    |
|    8    |  96  |  1  |  1  |  1  |  1  |  4  | 384 | True  |  59.66 | 3.146 | 134.216 | [smooth-eon-2207](https://wandb.ai/aurora_gpt/AuroraGPT/runs/awt6d825)     |
|    8    |  96  |  1  |  1  |  1  |  1  |  8  | 768 | True  |  62.19 | 3.279 | 139.919 | [glamorous-eon-2209](https://wandb.ai/aurora_gpt/AuroraGPT/runs/jmukcjdc)  |
| &nbsp;  |      |     |     |     |     |     |     |       |        |       |         |                                                                            |
|   16    | 192  |  1  |  1  |  1  |  1  |  1  | 192 | True  |   57.6 | 6.074 | 129.577 | [gallant-field-2208](https://wandb.ai/aurora_gpt/AuroraGPT/runs/gb94ahi3)  |
|   16    | 192  |  1  |  1  |  1  |  1  |  2  | 384 | True  |  61.86 | 6.523 | 139.161 | [swept-shadow-2210](https://wandb.ai/aurora_gpt/AuroraGPT/runs/riy7y3k1)   |
|   16    | 192  |  1  |  1  |  1  |  1  |  4  | 768 | True  |  64.28 | 6.779 | 144.621 | [restful-oath-2212](https://wandb.ai/aurora_gpt/AuroraGPT/runs/kzmhlcju)   |

[^z3h]: Seems to get hung on 4 nodes (waited ~30 min)

## 42B Model (Pure Tensor Parallelism) (TP=6)

- 42B param model

```bash
#[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
#[/f/d/f/p/a/Megatron-DeepSpeed][🌱 saforem2/dev][📦📝🤷✓] [⏱️ 4m15s]
#[06/14/25 @ 16:33:06][x4515c6s0b0n0]
; TP=6 PP=1 HEADS=$((TP * $((48 / TP)))) NLAYERS=$((TP * $((48 / TP)))) NUM_KV_HEAD=$((TP * $((8 / TP)))) FFN_HIDDEN_SIZE=$((TP * $((28672 / TP)))) HIDDEN=$((HEADS * $((8192 / HEADS)))) ZERO_STAGE=2 MICRO_BATCH=1 GRAD_ACC_STEPS=1 DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt bash train_alcf.sh
```

```bash
==== ARCHITECTURE ====
NLAYERS: 48
GAS: 1
PP: 1
HEADS: 48
USE_ACTIVATION_CHECKPOINTING: 0
FFN_HIDDEN_SIZE: 28668
SEQ: 4096
GBS: 16
DP: 16
NUM_KV_HEAD: 6
HIDDEN: 8160
TP: 6
SP: 1
MBS: 1
======================
```

| Nodes | NGPU |  PP |  DP |  TP | MBS | GAS | GBS | TFLOPS | samples / s | tok / gpu / s | wandb                                                                          |
| ----: | :--: | --: | --: | --: | --: | --: | --: | -----: | ----------: | ------------: | :----------------------------------------------------------------------------- |
|     8 |  96  |   1 |  16 |   6 |   1 |   1 |  16 |  19.76 |       1.740 |        74.257 | [royal-forest-2135](https://wandb.ai/aurora_gpt/AuroraGPT/runs/o3cyy1mq)       |
|     8 |  96  |   1 |  16 |   6 |   1 |   2 |  32 |  22.13 |       1.949 |        83.170 | [balmy-terrain-2136](https://wandb.ai/aurora_gpt/AuroraGPT/runs/rpo1fbk9)      |
|     8 |  96  |   1 |  16 |   6 |   1 |   4 |  64 |  23.60 |       2.078 |        88.676 | [eternal-wildflower-2137](https://wandb.ai/aurora_gpt/AuroraGPT/runs/faxpu0r6) |

## From Deepak

### Pipeline + ZeRO-1

| Nodes | NGPU |  PP |  DP |  TP | MBS | GBS | TFLOPS | samples / s |
| ----: | ---: | --: | --: | --: | --: | --: | -----: | ----------: |
|     8 |   96 |   8 |  12 |   1 |   1 |  96 |    OOM |         OOM |
|     8 |   96 |   8 |   6 |   2 |   1 |  48 |   58.2 |         1.1 |
|     8 |   96 |   8 |   6 |   2 |   1 |  96 |   72.5 |         1.4 |
|     8 |   96 |   8 |   6 |   2 |   1 | 192 |   85.6 |         1.7 |
|    16 |  192 |  16 |  12 |   1 |   1 | 192 |   59.9 |         2.3 |
|    16 |  192 |  16 |  12 |   1 |   1 | 384 |   76.1 |         3.0 |
|    16 |  192 |  16 |  12 |   1 |   1 | 768 |    OOM |         OOM |
|    32 |  384 |  16 |  24 |   1 |   1 | 384 |    OOM |         OOM |
|    32 |  384 |  16 |  24 |   1 |   1 | 768 |    OOM |         OOM |

### ZeRO-3 MiCS Performance Data

| Nodes | DP  | TP  | MBS | GBS | TFLOPS | sample/s |
| :---: | :-: | :-: | :-: | :-: | :----: | :------: |
|   4   | 48  |  1  |  1  | 48  |  98.5  |   0.96   |
|   8   | 96  |  1  |  1  | 96  |  96.7  |   1.88   |
|  16   | 192 |  1  |  1  | 192 |  95.4  |   3.70   |

| Nodes | DP  | TP  | MBS | GBS | TFLOPS | sample/s |
| :---: | :-: | :-: | :-: | :-: | :----: | :------: |
| 4     | 48  | 1   | 1   | 48  | 96.8   | 0.94     |
| 8     | 96  | 1   | 1   | 96  | 99.1   | 1.92     |
| 16    | 192 | 1   | 1   | 192 | 97.2   | 3.77     |
