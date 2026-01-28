# Megatron-DeepSpeed, optimizers, hyperparameters

Single command to test and run Megatron-DeepSpeed:

```bash
now=$(date +'%Y-%m-%d-%H%M%S') && debug_dir="${now}" && mkdir -p "${debug_dir}"&& cd "${debug_dir}"&& git clone https://github.com/argonne-lcf/Megatron-DeepSpeed && cd Megatron-DeepSpeed && source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env && python3 -m pip install --require-virtualenv "git+https://github.com/saforem2/ezpz" "numpy<2" deepspeed tensorboard && ezpz-test && DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt bash train_alcf.sh
```
## Optimizers
The default optimizer is `adamw`. Go to this [list of optimizers](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1485) for a complete list of supported optimizers (note that dshampoo might throw checkpointing errors, we are working on fixing this). For example, to run with `muon`, you can do:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh
```
Here
```bash
DATA_FILE_LIST=path/to/your/tokenized/data
TRAIN_TOKENS= number of training tokens
GRAD_ACC_STEPS=number of grad accumulation steps
LR=learning rate
LR_WARMUP_FRACTION=warmup fraction
OPT=optimizer
```
Your global batch size will be: `num_gpus*micro_batch_size*GRAD_ACC_STEPS`, micro batch size is 1 by default, you can change it by adding `MICRO_BATCH=new_micro_batch_size` to your options. To have the corresponding number if tokens per step, you need to multiply the global batch size by the sequence length (set with `SEQ_LEN`, default is 4096)

### Adding custom optimizers
To add a custom optimizer, you have to modify the following files:
- `megatron/optimizer/__init__.py`: [muon example](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/optimizer/__init__.py#L434), note that you either heve to import the optimizer from a pre-installed package or add it in the `megatron/optimizer/` folder.
- `megatron/arguments.py`: [optimizer arguments](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1070), to add the optimizer arguments
- `megatron/arguments.py`: [list of valid optimizers](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1485), to add the new optimizer to the list of valid optimizers

### Schedulers
Note that the default scheduler is `cosine`. We also support `infinite cosine, infinite inverse square root, constant, constant with cooldown, inverse square root, linear` schedulers. For example to change the scheduler to `constant`, you can do so with the `LR_DECAY_STYLE` option:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh
```
To add cooldown, you need to add the `--lr_constant_plus_cooldown` flag and set the cooldown fraction with `--lr_constant_plus_cooldown_frac`. The default cooldown fraction is 0.05
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --lr_constant_plus_cooldown_frac 0.01
```
#### Adding custom schedulers
To add a custom scheduler, you have to modify the following files:
- `megatron/optimizer_param_scheduler.py`: [schedulers](megatron/optimizer_param_scheduler.py), to add the new scheduler
- `megatron/arguments.py`: [list of LR arguments](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1671), to add the new scheduler arguments.
- You might have to change [the function](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/training.py#L559) to incorporate your custom scheduler options.

## Hyperparameter tuning
#### Init variance
Weight initialization is key to training LLMs,and to avoid spikes in losses. Here, we initialize the weights following this [paper](https://arxiv.org/pdf/2312.16903). The default variance value at initialization is 0.02. To add custom variances, one can use `--init-method-std, `--adjust-word-embedding-init`, and `--word-embedding-init-std`. For our runs, we do
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --init-method-std ${sqrt(2/5d)}  --adjust-word-embedding-init --word-embedding-init-std 0.632
```
where `d=hidden size`. In general, the initialization should be 
```bash
--init-method-std sqrt{ 2 / (5 * d) }  --adjust-word-embedding-init --word-embedding-init-std sqrt{ 2 / 5 }
```
### Learning rate
For the learning rate, we implemented the learning rate finder routine [here](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) and [here](https://arxiv.org/pdf/1506.01186). This is activated with the `--lr-finder` and run for `TRAIN_ITERS` steps. For example, for a 1000 steps:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_ITERS=1000 GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --init-method-std ${sqrt(2/5d)}  --adjust-word-embedding-init --word-embedding-init-std 0.632 --lr-finder
```
This approach allows to find the largest LR one can train with without the model divergence. Training with large LR is crucial for infinite schedulers as well for large batch training. We increase the LR following a power law at each step and monitor the LR curve in particular the decaying phase and the blow up phase. To tune the LR, you identify the point where the LR start increasing and divide that LR value by 10. You can also pick the LR corresponding to the steepest descent phase.
![lr_finder](./assets/lb_optimizers/lr_finder_example.png)
### Maximal Update Parametrization/Complete Parametrization
We have MuP and CompleteP incorporated in AuroraGPT in the `lb-optimizers` branch.
