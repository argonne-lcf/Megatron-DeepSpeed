# CPT
This document serves as a strategy cookbook for our current runs. 
CPT is the process of continually pre-training a model on new data. The goal of CPT is to continue training on new data while retaining previously learned knowledge and avoiding forgetting. When compared to finetuning, the goal of CPT is not to improve the model knowledge and performance on a given downstream task but to retain and improve current knowledge as more data are being streamed while avoiding catastrophic forgetting.
In this document, we focus on three CPT approaches: a **data centric strategy**, an **optimization (LR) strategy**, and a third one mixing both. This means that the following are **fixed** across all runs:
- Model architecture
- Sequence length
- Optimizer (although it might be interesting testing how changing optimizers across stages affect training)
- Evaluation/validation tasks (needs to be fixed from the start and be consistent across stages)

In all that follows, we suppose the base model was trained on dataset $D_0$ and label the subsequent datasets $D_i$, $i = 1\cdotsN$. What it means for us is that stage 1 is training with $D_0$, stage 2 is training with $D_1$, stage 3 with $D_3$ and stage 4 with $D_3$. A CPT strategy for the legacy model (agpt-7B) can be found at the end of the document.

## AuroraGPT V1 (Stages 1 to 4)
For these runs, we have 4 stages of training with the first stage producing the pretrained or base model. A key element here is the learning rate scheduler that we are using. In fact, as opposed to the legacy model, we used an infinite scheduler where the LR was warmed up to $LR_{max}$ then kept constant before cooling it down to convergence. The main advantage of this is to avoid rewarming the LR when doing CPT which can lead to instabilities. Therefore, we mostly adopt a data centric strategy here but might resort to playing with the LRs if needed. The dataset $D_0$ for pretraining is Olmo-mix and has 4 Trillion tokens, then $D_1$ has 2 Trillion tokens from Dolmino and fineweb Edu meaning the data distribution between these two stages is weak. We then have $D_2$ for stage 3 that has 1.5 trillion tokens from math, code, ans science papers. Finally, we have $D_3$ stage 4 made of 0.5 trillion tokens from reasoning traces. 

| Stage | Dataset Symbol | Size | Source / Path | Notes |
|------:|----------------|----------------------|---------------|-------|
| 0 | D₀ |  4T | Olmo-mix | Pretraining dataset |
| 1 | D₁ | 2T | Dolmino and fineweb Edu | CPT stage 1 |
| 2 | D₂ | 1.5T | Olmo-Mix, Open Alex, and proof pile II | CPT stage 2 |
| 3 | D₃ | 0.5T |Olmo-Mix, OpenMathInstruct, CoT Collection, AQUA-RAT, Llama-Nemotron Dataset, GSM8K, OpenHermes  | CPT stage 3 |

## Data centric strategy ##
The main thing to figure out here is the data mixing strategy. To avoid catastrophic forgetting, we need to sample from the pretraining dataset $D_0$, the current one $D_i$, and we also might need to sample from a buffer $B$ that contains data from the previous stages $D_1,\cdots,D_{i-1}$. Which means we need sampling weights $\alpha_0$ for the pretraining data, $\alpha_D$ for the current dataset, and $\alpha_B$ for the buffer dataset with $\alpha_0 + \alpha_D + \alpha_D = 1$.
See the figure below from this [paper](https://arxiv.org/pdf/2408.14471)
![data mixing](./assets/CPT_data_mixing.png)
Note that you add data to the buffer B after the current step to be used for the next one i.e at sampling time, B only contains data from previous stages.

#### Stage 1 to stage 2 (weak distribution shift)
##### First strategy to try
**YOU NEED TO USE A CHECKPOINT AT LR=LR_max i.e. BEFORE COOLING DOWN**
Just naively continue training with D_1, no replay data. This might work here because the datasets are similar.
##### Second strategy 
 **YOU NEED TO USE A CHECKPOINT AT LR=LR_max i.e. BEFORE COOLING DOWN**. Then,
 1. ***Replay the pretraining dataset*** To that aim, use [mix_datasets.py](https://github.com/zhenghh04/blendcorpus/blob/main/utils/mix_datasets.py) function to build your cpt dataset. You need to mix data from the pretraining set $D_0$ and the current CPT set $D_1$. Always start with a small percent of the pretraing dataset i.e $\alpha_0 = 0.1$ and $\alpha_D = 0.9$ then increase $\alpha_0$ to 25-30%, **$alpha_0=0.05** is a common suitable choice. For example, to mix the lucid papers with weight 0.9 and the dolma dataset with weight 0.1, you do
 ```bash
python3 mix_datasets.py --input 0.9 /flare/Aurora_deployment/AuroraGPT/datasets/papers/papers.txt 0.1 /flare/Aurora_deployment/AuroraGPT/datasets/dolma/dolma_v1_7_file_list_v2.txt > ${debug_dir}/Megatron-DeepSpeed/ALCF/data-lists/aurora/mix_lucid_papers09_dolma01.txt
```
For convenience, here is a copy of the ***mix_datasets.py*** script
```bash
#!/usr/bin/env python3
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="Mix multiple file-lists, normalize internal weights, and apply global file weights."
    )
    p.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help="Pairs of file_list and global_weight, e.g.: --inputs 0.3  f1.txt 0.7 f2.tx"
    )
    return p.parse_args()


def main():
    args = parse_args()
    inp = args.inputs
    if len(inp) % 2 != 0:
        sys.exit("Error: --inputs must be an even number of arguments (file weight pairs).")

    # Group into (file_path, global_weight)
    pairs = []
    for i in range(0, len(inp), 2):
        file_path = inp[i+1]
        try:
            gw = float(inp[i])
        except ValueError:
            sys.exit(f"Error: global weight must be a number, got '{inp[i]}'")
        if gw <= 0:
            sys.exit(f"Error: global weight must be positive, got {gw}")
        pairs.append((gw, file_path))

    # Compute sum of all global weights (if normalization across files is desired)
    sum_global = sum(gw for gw, _ in pairs)

    for gw, file_path in pairs:
        # Normalized file-level fraction (optional across all files)
        file_fraction = gw / sum_global

        # Read entries and sum file-local weights
        entries = []
        file_sum = 0.0
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        sys.exit(f"Error: each line must have prefix weight corpus, got: '{line}'")
                    prefix = parts[1]
                    try:
                        w = float(parts[0])
                    except ValueError:
                        sys.exit(f"Error: weight must be numeric, got '{parts[1]}' in file {file_path}")
                    corpus = parts[2]
                    entries.append((prefix, w, corpus))
                    file_sum += w
        except FileNotFoundError:
            sys.exit(f"Error: cannot open file '{file_path}'")

        if file_sum <= 0:
            sys.exit(f"Error: sum of weights in file '{file_path}' is non-positive: {file_sum}")

        # Print header only once
        # Compute and print normalized weights
        for prefix, w, corpus in entries:
            new_w = (w / file_sum) * file_fraction
            print(f"{new_w:.6f} {prefix} {corpus}")

if __name__ == '__main__':
    main()
```
2. Start building the buffer $B$ in prevision of the next stages.
3. Load your checkpoint and run CPT with the --finetube flag.
Note that you might need to convert your checkpoints following [these instructions](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/notes/universal_checkpoint_bug.md).

##### If loss is not recovering after increasing the pretraining data weight, one can switch to a LR centric approach:
   - take a converged checkpoint **i.e after cooling it down** and experiment with rewarming the LR to a different value.
   - If still no luck, one should:
     a. take an earlier not converged checkpoint
     b. Continue training with the base dataset with (i) a cosine scheduler decaying to **LR_max/100** or (ii) cooldown to **LR_max/100**. (I would experiment with both if resources allow)
     c. Introduce the new dataset at **LR=LR_max/5**. When introducing the new dataset, you use a mixed one i.e $\alpha_0 > 0$ as you should not exclusively use the new dataset. This is basically the recipe here [recipe](https://arxiv.org/pdf/2407.07263v1)

My guess is that since the distribution shift is not too strong between stage I and stage II data, the naive approach or the second one might work and you will not need to experiment with the latter.

#### Stage 2 to stage 3 (shift to math/code datasets)
1. You can try the naive approach but it might not work here. Then,
2. Try the second strategy above i.e mixing from the previous stage 1 training set (the one obtained after mixing) then follow the same steps.
3. If loss is not recovering, use the buffer. Mix from $D_0$, the current dataset $D_2$ and the buffer B. You should try weights 0.05 for D_0, 0.48 for D_1, and 0.47 for B then 0, 0.1, 0.9. Some explorations might be needed here. Do not forget to add data from D_2 to the buffer for the next training stage
4. If all fail, do the following
 - take a checkpoint before convergence **i.e before cooldown**
 - Continue training with the base dataset with (a cosine scheduler decaying to **LR_max/100**) or (cooldown to **LR_max/100**). (I would experiment with both if resources allow)
 - Introduce the new dataset at **LR=LR_max/5**. When introducing the new dataset, you use a mixed one i.e you should not exclusively use the new dataset. This is basically the recipe here [recipe](https://arxiv.org/pdf/2407.07263v1)
   
4. **If that does not work**,take a converged checkpoint **i.e after cooling it down** and experiment with rewarming the LR to a different value and the data mixing strategy by increasing the pretraining data weight.

#### Stage 3 to stage 4 (shift to reasoning tracex)
Try the same strategies as above. My guess here is you will (need the buffer and balance the weights across the 3 data sources) OR (need to do step 3)



## Legacy agpt-7b checkpoints
This is for doing CPT on the initial agpt-7B checkpoint where a cosine scheduler was used from `lr=0.0002` to 0. Here, the CPT stratregy followed is the [replay+rewarm one](https://arxiv.org/pdf/2403.08763) where we replay a small amount of data from the initial pretraining dataset and mix it with the cpt one. The steps are as follows:
1. First, if running on resources different than in base pretraining i.e smaller num of gpus, we need to train from an **universal checkpoint**. If you don't have the universal checkpoint, you can follow [the instructions](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/notes/universal_checkpoint_bug.md) here.
2. Use [mix_datasets.py](https://github.com/zhenghh04/blendcorpus/blob/main/utils/mix_datasets.py) function to build your cpt dataset. Here we are mixing the lucid papers with weight 0.9 and dolma with weight 0.1 (you can play with the weights if needed):
```bash
python3 mix_datasets.py --input 0.9 /flare/Aurora_deployment/AuroraGPT/datasets/papers/papers.txt 0.1 /flare/Aurora_deployment/AuroraGPT/datasets/dolma/dolma_v1_7_file_list_v2.txt > ${debug_dir}/Megatron-DeepSpeed/ALCF/data-lists/aurora/mix_lucid_papers09_dolma01.txt
```

3. Then, we can run the following cpt command from the Megatron-deepspeed folder (you can modify GRAD_ACC_STEPS according to the batch size you want to do CPT with):
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/mix_lucid_papers_dolma.txt LOAD=/flare/AuroraGPT/AuroraGPT-v0/checkpoint-copies/checkpoints/ws768_ds_stage1_nl32_hs4096_mb1_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR=0.0002 LR_WARMUP_FRACTION=0.01 bash train_alcf.sh --universal-checkpoint --finetune
```
Here, we are rewarming to the original learning but you can rewarm to any LR you seem fit. by just setting a different value for LR For example, we tested rewarming to LR/2 i.e **LR=0.0001** and 2LR as well.
Here the following options options/flags should be:
```bash
DATA_FILE_LIST=path/to/your/tokenized/data
LOAD=path/to/your/universal/checkpoint
SAVE=path/to/where/you/want/to/save/checkpoints
--universal-checkpoint to load a universal checkpoint (not needed if checkpoint not universal)
```
 
     

## Worth exploring
Follow and implement the [recipe](https://arxiv.org/pdf/2407.07263v1) where the new dataset is incrementally introduced. This might be advantageous for example when the new dataset is QAs as opposed to pure text. Here:
1. If base model LR was decayed to 0, one might need to rewarm it before following the recipe
2. Constant/infinite LR schedule was used, one might experiment with the recipe as is.
