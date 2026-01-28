# CPT
This document serves as a **strategy cookbook** for our current runs.

Continual pre-training (CPT) is the process of training a model on new data over time while retaining previously learned knowledge and avoiding forgetting. Unlike fine-tuning, the goal of CPT is **not** to optimize performance on a specific downstream task. Instead, CPT aims to **retain and incrementally improve general model knowledge** as new data are streamed, while mitigating catastrophic forgetting.

In this document, we focus on two CPT approaches: a **data-centric strategy** and an **optimization (learning-rate) strategy**. As a result, the following components are **held fixed** across all runs:

- Model architecture  
- Sequence length  
- Optimizer  
  *(although it may be interesting to explore how changing optimizers across stages affects training)*  
- All hyperparameters except the learning rate  
- Evaluation and validation tasks  
  *(these must be fixed from the start and remain consistent across stages)*


In what follows, we assume that the base model was trained on dataset $D_0$, and we denote subsequent datasets by $D_i$, with $i = 1, \ldots, N$.

Under this convention:
- **Stage 1** corresponds to training on $D_0$,
- **Stage 2** corresponds to training on $D_1$,
- **Stage 3** corresponds to training on $D_2$,
- **Stage 4** corresponds to training on $D_3$,

We denote by $D^{CPT}_i$ the **training data actually used for CPT at stage $i$** (which may be a mixture of multiple datasets, depending on the strategy).

A CPT strategy for the legacy model (**agpt-7B**) is provided at the end of this document.

## Recommended CPT strategies by stage

| CPT Stages | Distribution shift | Primary strategy | Fallbacks | Notes |
|------|--------------------|------------------|-------------|-------|
| Stage 1 → 2 | Weak | Naive CPT | 5% Replay: $D^{CPT}_{2} = 0.05D_0 + 0.95D_1$| Add $D_1$ to buffer $B$ |
| Stage 2 → 3 | Strong | 5-30% replay of $D^{CPT}_{2}$ and monitor loss  | Use buffer: $0.33D_0 + 0.33D_2 + 0.34B$| Add $D_2$ to buffer $B$, you might need to switch to LR centric strategy, see below |
| Stage 3 → 4 | Strong | Cooldown with mix  $0.33D_0 + 0.33D_3 + 0.34B$ |  Cooldown with mix  $0.05D_0 + 0.47D_3 + 0.48B$  | You will need to continue decay if used in previous stage. Play with decay function,stages, and final LR value |



## AuroraGPT V1 (Stages 1 to 4)
![different stages](./assets/cpt_images/stages_training_initial-1.png)
For these runs, we consider **four stages of training**, with the first stage producing the pretrained (base) model.

A key component of our setup is the **learning-rate scheduler**. Unlike the legacy model, we use an **infinite scheduler**, in which the learning rate is warmed up to $LR_{\max}$, held constant, and then cooled down to convergence. The main advantage of this approach is that it **avoids rewarming the learning rate during CPT**, which can otherwise introduce instabilities.

As a result, we primarily adopt a **data-centric strategy** throughout these stages, resorting to learning-rate adjustments only when necessary.

The dataset $D_0$ for pretraining is Olmo-mix and has 4 Trillion tokens, then $D_1$ has 2 Trillion tokens from Dolmino and fineweb Edu meaning the data distribution between these two stages is weak. We then have $D_2$ for stage 3 that has 1.5 trillion tokens from math, code, ans science papers. Finally, we have $D_3$ stage 4 made of 0.5 trillion tokens from reasoning traces. 

| Stage | Dataset Symbol | Size | Source / Path | Notes |
|------:|----------------|----------------------|---------------|-------|
| 1 | D₀ |  4T | Olmo-mix | Pretraining |
| 2 | D₁ | 2T | Dolmino and fineweb Edu | High quality focused data |
| 3 | D₂ | 1.5T |Open Alex, and proof pile II | Math, code, science focused |
| 4 | D₃ | 0.5T |OpenMathInstruct, CoT Collection, AQUA-RAT, Llama-Nemotron Dataset, GSM8K, OpenHermes  | reasoning traces |

## Data centric strategy ##
The main thing to determine is the **data-mixing strategy**. To avoid catastrophic forgetting, we sample from the pretraining dataset $D_0$, the current dataset $D_i$, and, when necessary, from a buffer $B$ containing data from previous stages $D_1, \ldots, D_{i-1}$.

This requires defining sampling weights:
- $\alpha_0$ for the pretraining data $D_0$,
- $\alpha_D$ for the current dataset $D_i$,
- $\alpha_B$ for the buffer $B$,

with the constraint
\[
\alpha_0 + \alpha_D + \alpha_B = 1.
\]
See the figure below from this [paper](https://arxiv.org/pdf/2408.14471)
![data mixing](./assets/cpt_images/CPT_data_mixing.png)
Note that data are added to the buffer $B$ **after** the current stage completes and are used only in subsequent stages. That is, at sampling time during stage $i$, the buffer $B$ contains data exclusively from **previous stages**.

#### Stage 1 to stage 2 (weak distribution shift)
##### Strategy 1: No replay
`Important: USE A CHECKPOINT AT LR=LR_max i.e. BEFORE COOLING DOWN`.

Naively continue training with $D_1$, no replay data. 
- Continue training using only the current dataset $D_1$
- No replay from $D_0$ or buffer data
This may be sufficient under weak distribution shift but there is potential risks of forgetting

##### Strategy 2: Replay from pretraining dataset
`Important: USE A CHECKPOINT AT LR=LR_max i.e. BEFORE COOLING DOWN`. Then,replay the pretraining dataset
 We mix data from:
- the pretraining dataset $D_0$,
- the current CPT dataset $D_1$.

No buffer data is used at this stage, $\alpha_B=0$.

###### Mixing weights
- Start conservatively:
  - $\alpha_0 = 0.05$–$0.10$
  - $\alpha_D = 1 - \alpha_0$
> In practice, $\alpha_0 = 0.05$ is often a safe starting point.
> Increase up to 25–30% only if forgetting is observed.
![stage 1 to 2](./assets/cpt_images/strategy_cpt_stage1tostage2-1.png)


**Dataset construction**
   Use [mix_datasets.py](https://github.com/zhenghh04/blendcorpus/blob/main/utils/mix_datasets.py) function to build your cpt dataset. For example, to mix the lucid papers with weight 0.9 and the dolma dataset with weight 0.1, you do
 ```bash
python3 mix_datasets.py --input 0.9 /flare/Aurora_deployment/AuroraGPT/datasets/papers/papers.txt 0.1 /flare/Aurora_deployment/AuroraGPT/datasets/dolma/dolma_v1_7_file_list_v2.txt > ${debug_dir}/Megatron-DeepSpeed/ALCF/data-lists/aurora/mix_lucid_papers09_dolma01.txt
```
2. **Start building the buffer $B$** in prevision of the next stages.
3. **Run CPT**: Load your checkpoint and run CPT with the --finetube flag.
Note that you might need to convert your checkpoints following [these instructions](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/notes/universal_checkpoint_bug.md) to a universal checkpoint.

At the end of this stage, we have $D^{CPT}_1$.


#### Stage 2 to stage 3 (shift to math/code datasets)
##### Naive strategy
You can try the naive approach but it might not work here, stop early if loss does not recover.
##### Strategy 2
Mix in the final dataset $D^{CPT}_1$ used in Stage 1.
1. Construct a mixed dataset containing the final dataset $D^{CPT}_1$ used in stage 2 and $D_3$.
2. Follow the same procedure as in the previous mixing strategy. At this point, the model has seen 6T tokens and $D_3$ contains 1.5T. Here, give $D_3$ less weight.
##### Strategy 3
![stage 2 to 3](./assets/cpt_images/strategy_cpt_stage2tostage3-1.png)
If the loss is not recovering, sample from $D_0$, $D_2$ (not the final mix after stage 1), and the buffer $B$.
Start with the following candidate weights (some exploration may be required):
 - **Mix A:**  
  - `D0`: 0.33  
  - `D2`: 0.33  
  - `B`: 0.34
   
- **Mix B:**  
  - `D0`: 0.05  
  - `D2`: 0.48  
  - `B`: 0.47  

- **Mix C:** (this is called IIDifying the dataset)  
  - `D0`: 0.00  
  - `D2`: 0.10  
  - `B`: 0.90
Notes:
- Even a small weight on `D0` can help stabilize optimization.
- The buffer should contain representative or difficult samples from earlier stages.
- **Important:** Add samples from `D2` to the buffer at the end of this stage for use in the next training stage.

##### Strategy 4 (if all else fails)
![stage 2 to 3 decay](./assets/cpt_images/strategy3_cpt_stage2tostage3_decay-1.png)
If all previous strategies fail, apply the following procedure:

- Take a checkpoint **before convergence** (i.e., **before cooldown**).
- Continue training on the **base dataset** using one of the following:
  - a cosine scheduler decaying to **`LR_3 = LR_max / N`**, or
  - a cooldown to **`LR_max / N`**.  
  *(If resources allow, experiment with both. Try N=10, 50,100)*
- Introduce the new dataset at **`LR = LR_max / 5`**.
- When introducing the new dataset, **do not train on it exclusively**; always use a **mixed dataset**. Here try $\alpha_0=0.8 - 0.6$

This follows the general recipe described in  
[https://arxiv.org/pdf/2407.07263v1](https://arxiv.org/pdf/2407.07263v1)

##### Strategy 5 (last resort)

If Strategy 4 does not work:

- Take a **converged checkpoint** (i.e., **after cooldown**).
- Experiment with:
  - rewarming the learning rate to a different max value, and
  - adjusting the data-mixing strategy by **increasing the weight of pretraining data**.

At the end of this stage, we have $D^{CPT}_2$.
#### Stage 3 to stage 4 (shift to reasoning tracex)
![stage 3 to 4](./assets/cpt_images/strategy_cpt_stage3tostage4-1.png)
At this point, we only have ~6% of training left and one should start the final decay.

***If we didn't use Strategy 4 above:***
1. Try
 **Mix A:**  
  - `D0`: 0.33  
  - `D_3`: 0.33  
  - `B`: 0.34
    
 **Mix B:**  
  - `D0`: 0.5  
  - `D_3`: 0.25  
  - `B`: 0.25
2. Cooldown/decay the LR to convergence.

***If we did use Strategy 4 above:***
We should keep decaying with $D^{CPT}_2$ until $LR_3/100$ then introduce the new mix at $LR_3/5$
![stage 3 to 4 previous devay](./assets/cpt_images/strategy3_cpt_stage3tostage4ifprevdecay-1.png)


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
 
     

## Things to keep in mind
- If new dataset is considerably smaller that previous ones, one need to put more weight on previous data.
- One can reduce/increase the batch size by a factor k but need to reduce/increase LR by a factor $\sqrt(k)$ or $k$.
