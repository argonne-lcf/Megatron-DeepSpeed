# CPT
## Legacy agpt-7b checkpoints
This is for doing CPT on the initial agpt-7B checkpoint where a cosine scheduler was used from `lr=0.0002` to 0. Here, the CPT stratregy followed is the [replay+rewarm one](https://arxiv.org/pdf/2403.08763) where we replay a small amount of data from the initial pretraining dataset and mix it with the cpt one. The steps are as follows:
1. First, if running on resources different than in base pretraining i.e smaller num of gpus, we need to train from an **universal checkpoint**. If you don't have the universal checkpoint, you can follow [the instructions](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/notes/universal_checkpoint_bug.md) here.
2. Use [mix_datasets.py](https://github.com/zhenghh04/blendcorpus/blob/main/utils/mix_datasets.py) function to build your cpt dataset. Here we are mixing the lucid papers with weight 0.9 and dolma with weight 0.1 (you can play with the weights if needed):
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
## New agpt runs (phase 1 -> phase 2: weak distribution shift)
For the new runs, we are using a constant LR with cooldowns. The advantage of using a constant LR is to forego the need of rewarming. Furthermore, with the cooldown, one can train a model to convergence at any point of training without committing to a token budget.

To do CPT here, 
1. Convert checkpoint to an universal checkpoint (if running for example on smaller num of gpus), **YOU NEED TO USE A CHECKPOINT AT LR=LR_max i.e. BEFORE COOLING DOWN**
2.  Mix the datasets as above, I would try different mixing weights here to experiment with
3. set `export LR_WARMUP_FRAC=0.0` in order to not rewarm
4. Run
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/mix_lucid_papers_dolma.txt LOAD=/flare/AuroraGPT/AuroraGPT-v0/checkpoint-copies/checkpoints/ws768_ds_stage1_nl32_hs4096_mb1_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR=0.0002 LR_DECAY_STYLE=constant bash train_alcf.sh --universal-checkpoint --finetune --lr_constant_plus_cooldown 
```
5. If loss curve not recovering after a while or loss is diverging, one can:
   - take a converged checkpoint **i.e after cooling it down** and experiment with rewarming the LR to a different value, data mixing strategy etc as we did with the legacy model.
   - If still no luck, I'd:
     a. take an earlier not converged checkpoint
     b. Continue training with the base dataset with a cosine scheduler decaying to **LR_max/100** or cooldown to **LR_max/100**. (I would experiment with both if resources allow)
     c. Introduce the new dataset at **LR=LR_max/5**. When introducing the new dataset, you use a mixed one i.e you should not exclusively use the new dataset.This is basically the recipe here [recipe](https://arxiv.org/pdf/2407.07263v1)

My guess is that since the distribution shift is not too strong between phase I and phase II data, you will not need to experiment with step 5.

## New agpt runs (phase 2 -> phase 3 or phase 3 -> phase 4: strong distribution shift)
I'd try steps 1-4 above and put more weight on phase 2 (then phase 3) data when mixing in the new datasets. If no luck, step 5 should work here.
 
     

## To do list
Follow and implement the [recipe](https://arxiv.org/pdf/2407.07263v1) where the new dataset is incrementally introduced. This might be advantageous for example when the new dataset is QAs as opposed to pure text. Here:
1. If base model LR was decayed to 0, one might need to rewarm it before following the recipe
2. Constant/infinite LR schedule was used, one might experiment with the recipe as is.
